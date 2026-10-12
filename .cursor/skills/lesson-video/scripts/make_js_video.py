"""Build a lesson video whose picture is written in JavaScript (Remotion) and whose voice is local (Kokoro).

    PY=D:/StudyNotes-video/tts-venv/Scripts/python.exe
    $PY make_js_video.py all   deepseek/video/l01-s4-js            # tts, math, timeline, render, subs
    $PY make_js_video.py still deepseek/video/l01-s4-js --at 12.5  # one frame at 12.5 s, for layout checks
    $PY make_js_video.py sheet deepseek/video/l01-s4-js            # one frame per board state
    $PY make_js_video.py qa    deepseek/video/l01-s4-js            # Whisper listen-back (other venv, GPU)

An episode folder holds script.json (narration, chapters, levels), formulas.txt (`id | LaTeX` per line) and
index.tsx (the boards). Generated data goes to D:/StudyNotes-video/remotion/gen/<slug>/ and audio to
.../public/tts/; the Remotion workspace there holds node_modules and resolves the repo's imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(os.environ.get("BB_VIDEO_ROOT", r"D:\StudyNotes-video"))
RM = ROOT / "remotion"
TEX_BIN = ROOT / "TinyTeX" / "bin" / "windows"
WHISPER_PY = ROOT / "venv" / "Scripts" / "python.exe"
FPS, CARD, LEAD, GAP, TAIL = 30, 2.2, 0.5, 0.3, 1.0
SR = 24000
STRONG = "。！？；…"
os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))


VOICE_OVERRIDE: str | None = None   # set by --voice; gives the episode a separate slug, gen dir and output


def load(ep: Path) -> dict:
    s = json.loads((ep / "script.json").read_text(encoding="utf-8"))
    s.setdefault("slug", ep.name)
    if VOICE_OVERRIDE:
        s["voice"] = VOICE_OVERRIDE
        s["slug"] += "-" + re.sub(r"[^a-z0-9]+", "-", VOICE_OVERRIDE.split(":")[-1].lower()).strip("-")
        s["speed"] = s.get("speed_" + VOICE_OVERRIDE.split(":")[0], s.get("speed", 1.0))
    if s.get("speak_numbers") == "zh":
        os.environ["BB_SPEAK_NUMBERS"] = "zh"
    return s


def gen_dir(s: dict) -> Path:
    d = RM / "gen" / s["slug"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def all_lines(s: dict):
    for ln in s.get("intro", {}).get("lines", []):
        yield "intro", ln
    for ch in s["chapters"]:
        for ln in ch["lines"]:
            yield ch["id"], ln


# ---------------------------------------------------------------- voice (Kokoro, local)
_pipe = None


def _pipeline():
    global _pipe
    if _pipe is None:
        from kokoro import KModel, KPipeline
        repo = "hexgrad/Kokoro-82M-v1.1-zh"
        model = KModel(repo_id=repo).eval()
        en = KPipeline(lang_code="a", repo_id=repo, model=False)
        _pipe = KPipeline(lang_code="z", repo_id=repo, model=model, en_callable=lambda t: next(en(t)).phonemes)
    return _pipe


def sentences(speech: str) -> list[tuple[int, int]]:
    out, start = [], 0
    for m in re.finditer(f"[{STRONG}]+", speech):
        out.append((start, m.end()))
        start = m.end()
    if start < len(speech) and speech[start:].strip():
        out.append((start, len(speech)))
    return [(a, b) for a, b in out if speech[a:b].strip()]


# ---------------------------------------------------------------- voice (GPT-SoVITS, local GPU, personal-use voices)
# A voice folder D:/StudyNotes-video/voices/<name>/ holds voice.json {"sovits", "gpt", "ref_wav", "ref_text", "ref_lang"}.
# The engine is the official GPT-SoVITS Windows package (hash-checked), serving only on 127.0.0.1.
GSV_DIR = ROOT / "gpt-sovits" / "GPT-SoVITS-v2pro-20250604"
GSV_PORT = 9880
_gsv_proc = None


def _port_open(port: int) -> bool:
    import socket
    with socket.socket() as so:
        so.settimeout(0.5)
        return so.connect_ex(("127.0.0.1", port)) == 0


def ensure_gsv(name: str) -> None:
    global _gsv_proc
    if _port_open(GSV_PORT):
        return
    vdir = ROOT / "voices" / name
    v = json.loads((vdir / "voice.json").read_text(encoding="utf-8"))
    log = open(ROOT / "work" / f"gsv-{name}.log", "w", encoding="utf-8")
    _gsv_proc = subprocess.Popen(
        [str(GSV_DIR / "runtime" / "python.exe"), "api.py", "-a", "127.0.0.1", "-p", str(GSV_PORT),
         "-s", str(vdir / v["sovits"]), "-g", str(vdir / v["gpt"]), "-dr", str(vdir / v["ref_wav"]),
         "-dt", v["ref_text"], "-dl", v.get("ref_lang", "zh")],
        cwd=GSV_DIR, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        if _port_open(GSV_PORT):
            return
        if _gsv_proc.poll() is not None:
            sys.exit(f"GPT-SoVITS exited; see {log.name}")
        time.sleep(1)
    sys.exit("GPT-SoVITS did not start within 4 minutes")


def gsv_say(name: str, text: str, speed: float):
    """One sentence through the local GPT-SoVITS api.py; returns (float32 mono audio, sample rate)."""
    import io
    import urllib.request
    import soundfile as sf
    ensure_gsv(name)
    body = json.dumps({"text": text, "text_language": "zh", "cut_punc": "", "speed": speed}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{GSV_PORT}/", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        audio, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
    return (audio if audio.ndim == 1 else audio.mean(axis=1)), sr


RETAKE_CER = 0.12   # a GPT-SoVITS sentence whose first take scores worse than this in Whisper gets two more takes


def engine_text(chunk: str, voice: str) -> str:
    """What the engine is asked to say. GPT-SoVITS voices were fine-tuned on Chinese only and garble English, so their
    Latin words and letters are rewritten the way a Mandarin speaker reads them (translit.py). Subtitles keep the
    original; Kokoro reads English well and gets the text unchanged."""
    if voice.startswith("gsv:"):
        import translit
        return translit.to_spoken(chunk)
    return chunk


def _line_key(speech: str, voice: str, speed: float) -> tuple[str, str]:
    if voice.startswith("gsv:"):
        spoken = "|".join(engine_text(speech[a:b].strip(), voice) for a, b in sentences(speech))
        return "gsv-v2pro", hashlib.sha1(json.dumps([spoken, voice, speed, "gsv-v2pro", "retake", RETAKE_CER],
                                                    ensure_ascii=False).encode()).hexdigest()[:16]
    return "kokoro-v1.1-zh", hashlib.sha1(json.dumps([speech, voice, speed, "kokoro-v1.1-zh"], ensure_ascii=False).encode()).hexdigest()[:16]


def _cached(key: str) -> dict | None:
    wav, meta_p = RM / "public" / "tts" / f"{key}.wav", RM / "gen" / "tts-meta" / f"{key}.json"
    return json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() and wav.exists() else None


def _whisper_scores(items: list[dict]) -> list[float]:
    """Whisper (GPU, the other venv) on each take; score = CER against the original text or the engine text, whichever
    is lower (Whisper writes a well-said 达布溜欧 back as "WO")."""
    if not items:
        return []
    work = ROOT / "work" / "takes"
    work.mkdir(parents=True, exist_ok=True)
    src, dst = work / "score-in.json", work / "score-out.json"
    src.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    code = f"""
import json, sys
sys.path.insert(0, {str(HERE)!r})
import qa
items = json.load(open({src.as_posix()!r}, encoding="utf-8"))
out = []
for it in items:
    hyp = qa.transcribe(it["wav"])
    out.append(min(qa.cer(it["ref"], hyp), qa.cer(it["eng"], hyp)))
json.dump(out, open({dst.as_posix()!r}, "w"))
"""
    subprocess.run([str(WHISPER_PY), "-c", code], env={**os.environ, "PYTHONUTF8": "1"}, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(dst.read_text())


def gsv_batch(texts: list[str], voice: str, speed: float) -> int:
    """Voice many lines with a GPT-SoVITS voice: one take per sentence, Whisper scores every take, sentences above
    RETAKE_CER get two more takes, and each line is assembled from the clearest take of each sentence."""
    import numpy as np
    import soundfile as sf
    import tts
    todo = []
    for text in texts:
        line = tts.parse(text)
        _, key = _line_key(line.speech, voice, speed)
        if _cached(key) is None and key not in {k for k, _, _ in todo}:
            todo.append((key, line, [(a, b, line.speech[a:b].strip(), engine_text(line.speech[a:b].strip(), voice))
                                     for a, b in sentences(line.speech)]))
    if not todo:
        return 0
    takes = ROOT / "work" / "takes"

    def make(entries):
        for e in entries:
            p = takes / e["key"] / f"{e['si']}-{e['take']}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                audio, sr = gsv_say(voice[4:], e["eng"], speed)
                sf.write(p, audio, sr)
            e["wav"] = str(p)

    first = [{"key": k, "si": si, "take": 0, "ref": c, "eng": g} for k, _, ss in todo for si, (_, _, c, g) in enumerate(ss)]
    make(first)
    s1 = _whisper_scores(first)
    retry = [{**e, "take": t} for e, c in zip(first, s1) if c > RETAKE_CER for t in (1, 2)]
    make(retry)
    s2 = _whisper_scores(retry)
    best: dict[tuple[str, int], tuple[int, float]] = {}
    for e, c in list(zip(first, s1)) + list(zip(retry, s2)):
        k = (e["key"], e["si"])
        if k not in best or c < best[k][1]:
            best[k] = (e["take"], c)
    for key, line, ss in todo:
        pieces, sents, t, sr = [], [], 0.0, SR
        for si, (a, b, _, eng) in enumerate(ss):
            take, c = best[(key, si)]
            audio, sr = sf.read(takes / key / f"{si}-{take}.wav", dtype="float32")
            pause = np.zeros(int(0.22 * sr), np.float32)
            sents.append({"s0": a, "s1": b, "t0": round(t, 3), "t1": round(t + len(audio) / sr, 3), "spoken": eng,
                          "take": take, "cer": round(c, 3)})
            pieces += [audio, pause]
            t += (len(audio) + len(pause)) / sr
        full = np.concatenate(pieces[:-1]) if pieces else np.zeros(sr // 2, np.float32)
        sf.write(RM / "public" / "tts" / f"{key}.wav", full, sr)
        meta = {"key": key, "file": f"tts/{key}.wav", "speech": line.speech, "duration": round(len(full) / sr, 3),
                "sentences": sents, "voice": voice}
        (RM / "gen" / "tts-meta" / f"{key}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"gsv: {len(first)} sentences, {len(retry) // 2} retaken; mean first-take CER {sum(s1) / len(s1):.3f}, "
          f"chosen {sum(c for _, c in best.values()) / len(best):.3f}")
    return len(todo)


def synth(text: str, voice: str, speed: float) -> dict:
    import numpy as np
    import soundfile as sf
    import tts
    line = tts.parse(text)
    engine, key = _line_key(line.speech, voice, speed)
    if voice.startswith("gsv:") and _cached(key) is None:
        gsv_batch([text], voice, speed)
    wav, meta_p = RM / "public" / "tts" / f"{key}.wav", RM / "gen" / "tts-meta" / f"{key}.json"
    if meta_p.exists() and wav.exists():
        return json.loads(meta_p.read_text(encoding="utf-8"))
    wav.parent.mkdir(parents=True, exist_ok=True)
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    sr = SR
    pieces, sents, t = [], [], 0.0
    for a, b in sentences(line.speech):
        chunk = line.speech[a:b].strip()
        if voice.startswith("gsv:"):
            audio, sr = gsv_say(voice[4:], chunk, speed)
        else:
            audio = np.concatenate([r.audio.numpy() for r in _pipeline()(chunk, voice=voice, speed=speed)])
        pause = np.zeros(int(0.22 * sr), np.float32)
        sents.append({"s0": a, "s1": b, "t0": round(t, 3), "t1": round(t + len(audio) / sr, 3)})
        pieces += [audio.astype(np.float32), pause]
        t += (len(audio) + len(pause)) / sr
    full = np.concatenate(pieces[:-1]) if pieces else np.zeros(sr // 2, np.float32)
    sf.write(wav, full, sr)
    meta = {"key": key, "file": f"tts/{key}.wav", "speech": line.speech, "duration": round(len(full) / sr, 3),
            "sentences": sents, "voice": voice}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def char_time(meta: dict, i: int) -> float:
    """Seconds into the clip at which speech[i] is reached: exact per sentence, proportional inside one."""
    speech = meta["speech"]
    for s in meta["sentences"]:
        if s["s0"] <= i < s["s1"] or (i == len(speech) and s["s1"] == len(speech)):
            w = [1.6 if c in "，、：,:" else 1.0 for c in speech[s["s0"]:s["s1"]]]
            done = sum(w[: max(0, i - s["s0"])])
            return s["t0"] + (s["t1"] - s["t0"]) * done / max(sum(w), 1e-6)
    return meta["duration"] if i >= len(speech) else 0.0


def cmd_tts(ep: Path, args) -> None:
    s = load(ep)
    voice, speed = s.get("voice", "zm_010"), s.get("speed", 1.0)
    t0, n, chars = time.time(), 0, 0
    texts = [ln["text"] for _, ln in all_lines(s)]
    if voice.startswith("gsv:"):
        gsv_batch(texts, voice, speed)
    for text in texts:
        m = synth(text, voice, speed)
        n, chars = n + 1, chars + len(m["speech"])
    print(f"tts: {n} lines, {chars} chars in {time.time() - t0:.0f}s (local, free)")


# ---------------------------------------------------------------- formulas (LaTeX -> path-only SVG)
# A part \p{name}{...} is typeset in a marker colour rgb(1,2,k); dvisvgm keeps the colour as fill='#0102kk', which
# becomes a CSS hook (the board colours, dims or glows the part) and a box the board can point at.
PART = re.compile(r"\\p\{(\w+)\}\{")
_NUMS = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?")


def _path_box(d: str) -> tuple[float, float, float, float]:
    """Bounding box of an absolute-command path (M L H V C S Z, as dvisvgm writes them), control points included."""
    xs, ys, cx, cy = [], [], 0.0, 0.0
    for cmd, body in re.findall(r"([MLHVCSZ])([^MLHVCSZ]*)", d):
        v = [float(n) for n in _NUMS.findall(body)]
        if cmd == "H":
            for x in v:
                cx = x
                xs.append(cx); ys.append(cy)
        elif cmd == "V":
            for y in v:
                cy = y
                xs.append(cx); ys.append(cy)
        elif cmd != "Z":
            for i in range(0, len(v) - 1, 2):
                cx, cy = v[i], v[i + 1]
                xs.append(cx); ys.append(cy)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)


def _parts(svg: str, names: list[str]) -> dict:
    """Box of every part as fractions of the formula's box: {name: [x, y, w, h]}."""
    vx, vy, vw, vh = (float(n) for n in re.search(r"viewBox='([^']+)'", svg).group(1).split())
    glyph = {pid: _path_box(d) for pid, d in re.findall(r"<path id='([^']+)' d='([^']+)'", svg)}
    boxes: dict[str, list[float]] = {}

    def add(k: int, x0, y0, x1, y1):
        b = boxes.setdefault(names[k - 1], [x0, y0, x1, y1])
        b[:] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]

    for k, body in re.findall(r"<g fill='#0102([0-9a-f]{2})'>(.*?)</g>", svg, flags=re.S):
        for x, y, ref in re.findall(r"<use x='([^']+)' y='([^']+)' xlink:href='#([^']+)'", body):
            g = glyph[ref]
            add(int(k, 16), g[0] + float(x), g[1] + float(y), g[2] + float(x), g[3] + float(y))
    for attrs in re.findall(r"<rect ([^>]*fill='#0102[0-9a-f]{2}'[^>]*)/>", svg):
        a = dict(re.findall(r"(\w+)='([^']*)'", attrs))
        x, y, w, h = (float(a[n]) for n in ("x", "y", "width", "height"))
        add(int(a["fill"][-2:], 16), x, y, x + w, y + h)
    return {n: [round((b[0] - vx) / vw, 4), round((b[1] - vy) / vh, 4), round((b[2] - b[0]) / vw, 4),
                round((b[3] - b[1]) / vh, 4)] for n, b in boxes.items()}


def cmd_math(ep: Path, args) -> None:
    s = load(ep)
    os.environ["PATH"] = str(TEX_BIN) + os.pathsep + os.environ["PATH"]
    work = ROOT / "work" / "math"
    work.mkdir(parents=True, exist_ok=True)
    out = {}
    for raw in (ep / "formulas.txt").read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fid, tex = (x.strip() for x in raw.split("|", 1))
        names: list[str] = []

        def mark(m: re.Match) -> str:
            if m.group(1) not in names:
                names.append(m.group(1))
            return r"{\color[RGB]{1,2,%d}" % (names.index(m.group(1)) + 1)
        tex = PART.sub(mark, tex)
        stem = hashlib.sha1(tex.encode()).hexdigest()[:12]
        svg_p = work / f"{stem}.svg"
        if not svg_p.exists():
            src = (r"\documentclass[preview,border=1pt]{standalone}\usepackage{amsmath,amssymb}\usepackage{lmodern}"
                   r"\usepackage{xcolor}\begin{document}$\displaystyle " + tex + r"$\end{document}")
            (work / f"{stem}.tex").write_text(src, encoding="utf-8")
            r = subprocess.run(["latex", "-interaction=nonstopmode", f"{stem}.tex"], cwd=work, capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"LaTeX failed for {fid}: {tex}\n" + r.stdout[-800:])
            subprocess.run(["dvisvgm", "--no-fonts", "--exact-bbox", "-o", svg_p.name, f"{stem}.dvi"], cwd=work,
                           capture_output=True, check=True)
        svg = re.sub(r"<!--.*?-->|<\?xml[^>]*>", "", svg_p.read_text(encoding="utf-8"), flags=re.S).strip()
        w = float(re.search(r"width='([\d.]+)pt'", svg).group(1))
        h = float(re.search(r"height='([\d.]+)pt'", svg).group(1))
        parts = _parts(svg, names) if names else {}
        hook = lambda k: (f"data-p='{names[int(k, 16) - 1]}' style='fill:var(--pc-{names[int(k, 16) - 1]},currentColor);"
                          f"opacity:var(--po-{names[int(k, 16) - 1]},1);filter:var(--pf-{names[int(k, 16) - 1]},none)'")
        svg = re.sub(r"fill='#0102([0-9a-f]{2})'", lambda m: hook(m.group(1)), svg)
        svg = re.sub(r"(id=')", rf"\g<1>{fid}-", svg)
        svg = re.sub(r"(xlink:href='#)", rf"\g<1>{fid}-", svg)
        svg = re.sub(r"<svg ([^>]*?)width='[^']*' height='[^']*'", r"<svg \1width='100%' height='100%' fill='currentColor'", svg)
        out[fid] = {"svg": svg, "w": round(w, 3), "h": round(h, 3), **({"parts": parts} if parts else {})}
    (gen_dir(s) / "math.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"math: {len(out)} formulas, {sum(len(v.get('parts', {})) for v in out.values())} parts")


# ---------------------------------------------------------------- sound effects (synthesised, so no licences to track)
def cmd_sfx(ep: Path, args) -> None:
    """A small, quiet palette for the opening and the chapter cards: drone, whoosh, tick, ping, glitch.
    Deterministic numpy synthesis into public/sfx/; the episode decides when (and how loud) each plays."""
    import numpy as np
    import soundfile as sf
    sr, rng = 48000, np.random.default_rng(7)
    out = RM / "public" / "sfx"
    out.mkdir(parents=True, exist_ok=True)

    def t(sec):
        return np.arange(int(sec * sr)) / sr

    def env(n, a, r):                      # linear attack, exponential release over n samples
        e = np.ones(n)
        na = max(1, int(a * sr))
        e[:na] = np.linspace(0, 1, na)
        e[na:] = np.exp(-np.arange(n - na) / (r * sr))
        return e

    def lowpass(x, fc):                    # one-pole low-pass, enough for soft edges
        a = np.exp(-2 * np.pi * fc / sr)
        y = np.empty_like(x)
        acc = 0.0
        for i, v in enumerate(x):
            acc = (1 - a) * v + a * acc
            y[i] = acc
        return y

    def save(name, x, peak):
        x = x / (np.abs(x).max() + 1e-9) * peak
        sf.write(out / f"{name}.wav", np.stack([x, x], 1).astype(np.float32), sr)

    # drone: two low partials a fifth apart, slowly breathing, with a little filtered air; 90 s, faded at both ends
    tt = t(90)
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * tt / 11)
    drone = (np.sin(2 * np.pi * 55 * tt) + 0.6 * np.sin(2 * np.pi * 82.41 * tt + 1) + 0.25 * np.sin(2 * np.pi * 110.3 * tt)) * lfo
    air = np.convolve(rng.standard_normal(len(tt)), np.ones(400) / 400, mode="same") * 0.8
    fade = np.minimum(1, np.minimum(tt / 3, (90 - tt) / 3))
    save("drone", (drone + air) * fade, 0.5)
    # whoosh: noise through a band that sweeps up, swelling then falling away
    tw = t(1.4)
    noise = rng.standard_normal(len(tw))
    lo, hi = lowpass(noise, 700), lowpass(noise, 3200)
    sweep = np.clip(tw / 1.0, 0, 1)
    shape = np.sin(np.pi * np.clip(tw / 1.4, 0, 1)) ** 2
    save("whoosh", ((1 - sweep) * lo + sweep * (hi - lo) * 1.4) * shape, 0.7)
    # tick: a short, bright click
    tk = t(0.06)
    save("tick", np.sin(2 * np.pi * 2400 * tk) * env(len(tk), 0.001, 0.008), 0.6)
    # ping: a soft two-partial bell
    tp = t(1.2)
    save("ping", (np.sin(2 * np.pi * 880 * tp) + 0.35 * np.sin(2 * np.pi * 1318.5 * tp)) * env(len(tp), 0.004, 0.28), 0.6)
    # glitch: a few bit-crushed noise grains
    tg = t(0.28)
    g = np.round(rng.standard_normal(len(tg)) * 3) / 3
    gate = (np.floor(tg * 40) % 3 != 1).astype(float)
    save("glitch", lowpass(g * gate, 2500) * env(len(tg), 0.002, 0.08), 0.6)
    print(f"sfx: 5 sounds -> {out}")


# ---------------------------------------------------------------- the course map (opening of every episode)
REPO = HERE.parents[3]


def cmd_roadmap(ep: Path, args) -> None:
    """The course's tech tree as world geometry for the opening zoom: lanes, nodes, edges, axis, plus the lesson list
    and this lesson's pages. Data comes from the lesson-0 figure script (dates checked in lesson-00-doc.md)."""
    import importlib
    s = load(ep)
    cfg = s.get("roadmap")
    if not cfg:
        return
    course = REPO / cfg["course"]
    sys.path.insert(0, str(course / "figures" / "gen"))
    tt = importlib.import_module("l00_tech_tree")
    X0, W1, GAP_X, W2 = 0.0, 560.0, 34.0, 1740.0
    m_tb, m_t1 = tt.months(tt.TB), tt.months(tt.T1)
    xof = lambda d: (X0 + W1 * tt.months(d) / m_tb if d < tt.TB
                     else X0 + W1 + GAP_X + W2 * (tt.months(d) - m_tb) / (m_t1 - m_tb))
    FS, CH, PITCH, PAD, BADGE = 15.0, 34.0, 42.0, 10.0, 20.0
    tw = lambda t: sum(FS * (1.0 if ord(c) > 0x2E80 else 0.56) for c in t)
    nodes = []
    for nid, lane, date, label, lessons, strong in tt.NODES:
        w = 8 + len(lessons) * BADGE + 6 + tw(label) + 10
        nodes.append({"id": nid, "lane": lane, "date": date, "label": label, "lessons": lessons, "strong": strong,
                      "x": round(xof(tt.parse(date)), 1), "w": round(w, 1), "h": CH})
    src0 = (course / "src" / "lesson-00-tech-tree.md").read_text(encoding="utf-8")
    names = dict(re.findall(r"^# 主线 ([A-F])：(.+?) \|", src0, flags=re.M))
    lanes, y = [], 0.0
    for key, _, _ in tt.LANES:
        mine = sorted((n for n in nodes if n["lane"] == key), key=lambda n: n["x"])
        ends: list[float] = []
        for n in mine:
            for r, e in enumerate(ends):
                if e + 10 <= n["x"]:
                    n["row"], ends[r] = r, n["x"] + n["w"]
                    break
            else:
                n["row"] = len(ends)
                ends.append(n["x"] + n["w"])
        h = 2 * PAD + (max(1, len(ends)) - 1) * PITCH + CH
        for n in mine:
            n["y"] = round(y + PAD + n["row"] * PITCH, 1)
        lanes.append({"key": key, "name": names.get(key, "模型发布"), "y": round(y, 1), "h": round(h, 1)})
        y += h + (48 if key == "M" else 8)
    years = [{"year": yr, "x": round(xof(tt.dt.date(yr, 1, 1)), 1)} for yr in range(2017, 2027)]
    index = (course / "src" / "index.yaml").read_text(encoding="utf-8")
    lessons = [{"n": int(n), "title": t.strip()} for n, t in
               re.findall(r"file: lesson-(\d+)-[^,]+, tag: [^,]+,(?: tagclass: \w+,)? title: ([^,]+), desc:", index)]
    # the syllabus: parts in order with their lessons, and the maths refresher (opening of overview episodes)
    parts = []
    for name, body in re.findall(r"- name: (.+?)\n(.*?)(?=\n  - name: |\Z)", index, flags=re.S):
        nums = [int(n) for n in re.findall(r"file: lesson-(\d+)-", body)]
        if nums:
            parts.append({"name": name.strip(), "lessons": nums})
    math_idx = REPO / "math" / "src" / "index.yaml"
    maths = ([{"tag": tag, "title": t.strip()} for tag, t in
              re.findall(r"tag: (M\d+), tagclass: m, title: ([^,]+), desc:", math_idx.read_text(encoding="utf-8"))]
             if math_idx.exists() else [])
    maths.sort(key=lambda m: m["tag"])
    # lesson 0's full release timeline (date, name, threads; bold = a trunk model), placed on the same time axis
    timeline = []
    sec = src0.split("## 完整时间线", 1)[-1].split("\n## ", 1)[0]
    for date, tail, name, what, lanes_txt in re.findall(
            r"^\| <span[^>]*>(\d{4}-\d{2}-\d{2})</span>((?:（论文）| 起)?) \| (.+?) \| (.+?) \| (.+?) \|", sec, flags=re.M):
        timeline.append({"date": date, "paper": "论文" in tail, "name": name.replace("**", "").strip(), "what": what.strip(),
                         "trunk": name.startswith("**"), "lanes": [x.strip() for x in lanes_txt.split("、")],
                         "x": round(xof(tt.parse(date)), 1)})
    src = next((course / "src").glob(f"lesson-{cfg['lesson']:02d}-*.md")).read_text(encoding="utf-8")
    pages = [{"n": i + 1, "title": t, "sub": sub} for i, (t, sub) in
             enumerate(re.findall(r"^# (.+?) \| (.+?) \| \d+\s*$", src, flags=re.M))]
    out = {"lanes": lanes, "nodes": nodes, "edges": [list(e) for e in tt.EDGES], "years": years,
           "axis": {"x0": X0, "break": round(X0 + W1, 1), "gap": GAP_X, "x1": round(X0 + W1 + GAP_X + W2, 1),
                    "y": round(lanes[0]["y"] + lanes[0]["h"] + 24, 1)},
           "height": round(y - 8, 1), "lessons": lessons, "parts": parts, "maths": maths, "timeline": timeline, "lesson": cfg["lesson"],
           "page": cfg["page"], "pages": pages}
    (gen_dir(s) / "roadmap.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"roadmap: {len(nodes)} nodes, {len(lanes)} lanes, {len(lessons)} lessons, {len(pages)} pages in lesson {cfg['lesson']}")


# ---------------------------------------------------------------- timeline (single source for picture, sound, subtitles)
def cmd_timeline(ep: Path, args) -> dict:
    import tts
    from make_video import cue_chunks
    s = load(ep)
    voice, speed = s.get("voice", "zm_010"), s.get("speed", 1.0)
    f = lambda sec: int(round(sec * FPS))
    audio, cues = [], []

    def block(lines, start_s: float, lead: float):
        t, out = start_s + lead, {}
        for ln in lines:
            m = synth(ln["text"], voice, speed)
            line = tts.parse(ln["text"])
            marks = {k: f(t + char_time(m, i)) - f(start_s) for k, i in line.marks.items()}
            out[ln["id"]] = {"start": f(t) - f(start_s), "end": f(t + m["duration"]) - f(start_s), "marks": marks}
            audio.append({"file": m["file"], "from": f(t)})
            for spans in cue_chunks(line, max_len=20):
                (d0, _, s0, _), (_, d1, _, s1) = spans[0], spans[-1]
                text = line.display[d0:d1].strip().rstrip("，、：；。,")
                if text:
                    cues.append({"from": f(t + char_time(m, s0)), "to": f(t + char_time(m, s1)), "text": text})
            t += m["duration"] + GAP + ln.get("hold", 0)
        return out, t - GAP + TAIL

    tl = {"fps": FPS, "title": s.get("title", ""), "questions": s.get("questions", []), "chapters": []}
    intro_lines, t = block(s.get("intro", {}).get("lines", []), 0.0, 1.0)
    tl["intro"] = {"start": 0, "dur": f(t), "lines": intro_lines}
    for ch in s["chapters"]:
        start = t
        lines, end = block(ch["lines"], start + CARD, LEAD)
        tl["chapters"].append({"id": ch["id"], "title": ch["title"], "level": ch["level"], "tag": ch.get("tag", ""),
                               "q": ch.get("q", 0), "start": f(start), "card": f(CARD), "board": f(start + CARD),
                               "dur": f(end) - f(start), "lines": lines})
        t = end
    tl["total"] = f(t)
    for i in range(len(cues) - 1):  # a cue never overlaps the next; short gaps are bridged
        cues[i]["to"] = min(max(cues[i]["to"] + 6, cues[i]["from"] + 12), cues[i + 1]["from"])
    tl["cues"], tl["audio"] = cues, audio
    (gen_dir(s) / "timeline.json").write_text(json.dumps(tl, ensure_ascii=False), encoding="utf-8")
    print(f"timeline: {len(tl['chapters'])} chapters, {t / 60:.1f} min, {len(cues)} cues")
    for ch in tl["chapters"]:
        print(f"  {ch['start'] / FPS / 60:05.2f} {ch['id']} {ch['title']} [{ch['level']}] {ch['dur'] / FPS:.0f}s")
    return tl


# ---------------------------------------------------------------- render, stills, subtitles
def remotion(args_: list[str]) -> None:
    cmd = ["npx.cmd" if os.name == "nt" else "npx", "remotion", *args_]
    r = subprocess.run(cmd, cwd=RM, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        sys.exit(r.stdout[-3000:] + r.stderr[-3000:])


def cmd_render(ep: Path, args) -> None:
    s = load(ep)
    out = ROOT / "out" / s["slug"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    extra = ["--scale=0.5"] if args.preview else []
    name = f"{s['slug']}{'-preview' if args.preview else ''}.mp4"
    raw = out / f"raw-{name}"
    remotion(["render", str(ep / "index.tsx"), "episode", str(raw), f"--concurrency={args.jobs}",
              f"--props={props_file(s)}", "--codec=h264", "--crf=18", "--audio-codec=aac", *extra])
    # Kokoro speaks at about -28 LUFS: normalise to -16 (web video), and embed the chapters. Picture is copied.
    tl = json.loads((gen_dir(s) / "timeline.json").read_text(encoding="utf-8"))
    names = {"zero": "零基础", "eng": "工程师向"}
    ff = [";FFMETADATA1", f"title={s.get('title', s['slug'])}", "[CHAPTER]", "TIMEBASE=1/30", "START=0",
          f"END={tl['intro']['dur']}", "title=开场：科技树上的位置"]
    for c in tl["chapters"]:
        ff += ["[CHAPTER]", "TIMEBASE=1/30", f"START={c['start']}", f"END={c['start'] + c['dur']}",
               f"title={c['id']} {c['title']}（{names[c['level']]}）"]
    meta = out / "chapters.ffmeta"
    meta.write_text("\n".join(ff) + "\n", encoding="utf-8")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-f", "ffmetadata", "-i", str(meta), "-map", "0:v", "-map", "0:a",
                    "-map_metadata", "1", "-map_chapters", "1", "-c:v", "copy",
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(out / name)], check=True)
    raw.unlink()
    print(f"render: {out / name} in {time.time() - t0:.0f}s")


def props_file(s: dict) -> Path:
    """This voice's timeline and formulas as Remotion input props (the episode's static imports are only defaults)."""
    g = gen_dir(s)
    p = g / "props.json"
    props = {name: json.loads((g / f"{name}.json").read_text(encoding="utf-8"))
             for name in ("timeline", "math", "roadmap") if (g / f"{name}.json").exists()}
    p.write_text(json.dumps(props, ensure_ascii=False), encoding="utf-8")
    return p


def stills(ep: Path, out: Path, frames: list[int], scale: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["node", "stills.mjs", str(ep / "index.tsx"), str(out), str(scale), ",".join(map(str, frames)),
                        str(props_file(load(ep)))],
                       cwd=RM, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        sys.exit(r.stdout[-3000:] + r.stderr[-3000:])


def cmd_still(ep: Path, args) -> None:
    s = load(ep)
    out = ROOT / "out" / s["slug"] / "stills"
    frames = [int(float(x) * FPS) for x in args.at.split(",")]
    stills(ep, out, frames, 0.5)
    print("still:", ", ".join(str(out / f"{fr:05d}.png") for fr in frames))


def cmd_sheet(ep: Path, args) -> None:
    """One tile per board state: the frame just before each chapter ends, plus mid-chapter frames."""
    from PIL import Image, ImageDraw
    s = load(ep)
    tl = json.loads((gen_dir(s) / "timeline.json").read_text(encoding="utf-8"))
    shots = [(f"intro-{k}", ln["end"] - 2) for k, ln in tl["intro"]["lines"].items()] or [("cover", tl["intro"]["dur"] - 10)]
    for ch in tl["chapters"]:
        shots += [(f"{ch['id']}-card", ch["start"] + ch["card"] // 2)]
        mids = sorted({ln["end"] for ln in ch["lines"].values()})
        for e in mids[:: max(1, len(mids) // 2)]:
            shots.append((f"{ch['id']}", ch["board"] + e - 2))
        shots.append((f"{ch['id']}-end", ch["start"] + ch["dur"] - 3))
    if args.only:
        shots = [x for x in shots if x[0].split("-")[0] in args.only.split(",")]
    out = ROOT / "out" / s["slug"] / "sheet"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    stills(ep, out, [fr for _, fr in shots], 0.34)
    tiles = [(label, fr, out / f"{fr:05d}.png") for label, fr in shots]
    for old in (ROOT / "out" / s["slug"]).glob("sheet-*.png"):
        old.unlink()
    cols, tw, th = 3, 653, 367
    per = 12
    for k in range(0, len(tiles), per):
        batch = tiles[k:k + per]
        sheet = Image.new("RGB", (cols * tw, ((len(batch) + cols - 1) // cols) * th), (0, 0, 0))
        for i, (label, fr, p) in enumerate(batch):
            im = Image.open(p).convert("RGB").resize((tw, th))
            ImageDraw.Draw(im).text((6, th - 18), f"{label} @{fr / FPS:.1f}s", fill=(255, 80, 80))
            sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
        sheet.save(ROOT / "out" / s["slug"] / f"sheet-{k // per + 1}.png")
    print(f"sheet: {len(tiles)} tiles -> {ROOT / 'out' / s['slug']}/sheet-*.png")


def cmd_subs(ep: Path, args) -> None:
    s = load(ep)
    tl = json.loads((gen_dir(s) / "timeline.json").read_text(encoding="utf-8"))
    out = ROOT / "out" / s["slug"]
    out.mkdir(parents=True, exist_ok=True)

    def ts(fr: int, sep=",") -> str:
        ms = int(round(fr * 1000 / FPS))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"
    srt = [f"{i}\n{ts(c['from'])} --> {ts(c['to'])}\n{c['text']}\n" for i, c in enumerate(tl["cues"], 1)]
    (out / f"{s['slug']}.srt").write_text("\n".join(srt), encoding="utf-8")
    names = {"zero": "零基础", "eng": "工程师向"}
    lines = [f"{c['start'] // FPS // 60:02d}:{c['start'] // FPS % 60:02d} {c['id']} {c['title']}【{names[c['level']]}】"
             for c in tl["chapters"]]
    (out / "chapters.txt").write_text("00:00 开场：科技树上的位置\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("subs:", out / f"{s['slug']}.srt", "and chapters.txt")


def cmd_qa(ep: Path, args) -> None:
    """Whisper listen-back of every clip; runs in the Whisper venv on the GPU."""
    s = load(ep)
    used = {synth(ln["text"], s.get("voice", "zm_010"), s.get("speed", 1.0))["key"] for _, ln in all_lines(s)}
    code = f"""
import json, sys, glob
sys.path.insert(0, {str(HERE)!r})
import qa
rows = []
for key in {sorted(used)!r}:
    m = json.load(open({(RM / 'gen' / 'tts-meta').as_posix()!r} + '/' + key + '.json', encoding="utf-8"))
    hyp = qa.transcribe({(RM / 'public').as_posix()!r} + '/' + m["file"])
    spoken = "".join(x.get("spoken", "") for x in m["sentences"])
    c = min(qa.cer(m["speech"], hyp), qa.cer(spoken, hyp)) if spoken else qa.cer(m["speech"], hyp)
    rows.append((c, m["speech"], hyp))
rows.sort(reverse=True)
print("qa: %d clips, mean CER %.3f" % (len(rows), sum(r[0] for r in rows) / max(1, len(rows))))
for c, ref, hyp in rows[:8]:
    if c > 0.1: print("  %.2f\\n   ref: %s\\n   asr: %s" % (c, ref, hyp))
"""
    subprocess.run([str(WHISPER_PY), "-c", code], env={**os.environ, "PYTHONUTF8": "1"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["tts", "math", "roadmap", "sfx", "timeline", "render", "still", "sheet", "subs", "qa", "all"])
    ap.add_argument("episode", type=Path)
    ap.add_argument("--at", default="5", help="still: comma-separated seconds")
    ap.add_argument("--preview", action="store_true", help="render at half size")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--only", default="", help="sheet: comma-separated chapter ids (cover, 01, 02, ...)")
    ap.add_argument("--voice", default="", help="another voice for a separate variant: a Kokoro id (zf_001) or gsv:<name>")
    args = ap.parse_args()
    global VOICE_OVERRIDE
    VOICE_OVERRIDE = args.voice or None
    import atexit
    atexit.register(lambda: _gsv_proc and _gsv_proc.poll() is None and _gsv_proc.terminate())
    ep = args.episode.resolve()
    steps = {"tts": [cmd_tts], "math": [cmd_math], "roadmap": [cmd_roadmap], "sfx": [cmd_sfx], "timeline": [cmd_timeline], "render": [cmd_render],
             "still": [cmd_still], "sheet": [cmd_sheet], "subs": [cmd_subs], "qa": [cmd_qa],
             "all": [cmd_tts, cmd_math, cmd_roadmap, cmd_sfx, cmd_timeline, cmd_render, cmd_subs]}[args.step]
    for fn in steps:
        fn(ep, args)


if __name__ == "__main__":
    main()
