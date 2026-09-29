"""Build a blackboard lesson video from an episode file.

    python make_video.py all <episode.py> --preview   # 854x480 @15 fps: fast layout pass
    python make_video.py all <episode.py>             # 1920x1080 @30 fps: the real thing
    python make_video.py <step> <episode.py> [--preview] [--only S03,S05] [-j 8]

Steps: tts (synthesise every narration line, cached), qa (Whisper listen-back of every line),
render (one Manim process per segment, in parallel), assemble (concat, audio mix, board texture,
level strip, subtitles, chapters), sheet (contact sheet of every board state).

Big files live under BB_VIDEO_ROOT (default D:\\StudyNotes-video): cache/ (TTS clips), work/
(Manim media per segment) and out/<slug>/ (mp4, srt, chapters.txt, sheet-*.png).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BB_VIDEO_ROOT", r"D:\StudyNotes-video"))
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
TEX_BIN = ROOT / "TinyTeX" / "bin" / "windows"
QUALITY = {"dry": (854, 480, 15), "preview": (854, 480, 15), "final": (1920, 1080, 30)}
TAG = {"dry": "-dry", "preview": "-preview", "final": ""}
LEVEL_NAME = {"zero": "零基础", "eng": "工程师向"}
# per theme: level strip colours, subtitle font, whether to lay the chalk-dust texture over the frame
STYLE = {
    "3b1b": {"levels": {"zero": (131, 193, 103), "eng": (240, 172, 95)}, "sub_font": "Noto Sans SC", "texture": False},
    "chalk": {"levels": {"zero": (166, 219, 154), "eng": (245, 166, 107)}, "sub_font": "Microsoft YaHei", "texture": True},
}


def style() -> dict:
    return STYLE[os.environ.get("BB_THEME", "3b1b")]


def apply_episode_settings(ep: Path) -> dict:
    """EPISODE = {"theme": ..., "voice": ..., "speak_numbers": "zh"} -> environment for this process and Manim."""
    meta, _ = read_episode(ep)
    os.environ["BB_THEME"] = meta.get("theme", "3b1b")
    if meta.get("voice"):
        os.environ["BB_VOICE_ID"] = meta["voice"]
    if meta.get("speak_numbers"):
        os.environ["BB_SPEAK_NUMBERS"] = meta["speak_numbers"]
    return meta
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
SR = 48000


def _reexec_in_venv():
    try:
        import manim  # noqa: F401
    except ImportError:
        if VENV_PY.exists():
            sys.exit(subprocess.call([str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]]))
        raise


# ---------------------------------------------------------------- episode
def read_episode(path: Path):
    """Segments in file order with level, title and every literal narration line (no Manim import)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    meta, scenes = {}, []
    names = {"ZERO": "zero", "ENG": "eng"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "EPISODE" for t in node.targets):
            meta = ast.literal_eval(node.value)
        if isinstance(node, ast.ClassDef) and re.match(r"S\d+", node.name):
            info = {"name": node.name, "level": "zero", "title": "", "lines": []}
            for st in node.body:
                if not isinstance(st, ast.Assign):
                    continue
                tgt, val = st.targets[0], st.value
                pairs = (zip(tgt.elts, val.elts) if isinstance(tgt, ast.Tuple) and isinstance(val, ast.Tuple)
                         else [(tgt, val)])
                for t, v in pairs:
                    if getattr(t, "id", "") == "level":
                        info["level"] = names.get(getattr(v, "id", ""), getattr(v, "value", "zero"))
                    elif getattr(t, "id", "") == "title":
                        info["title"] = v.value
            for c in ast.walk(node):
                if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in ("say", "narrate")
                        and c.args and isinstance(c.args[0], ast.Constant) and isinstance(c.args[0].value, str)):
                    info["lines"].append(c.args[0].value)
            scenes.append(info)
    return meta, scenes


def narration_by_key(ep: Path) -> dict:
    """TTS cache key -> narration text, for timelines rendered before clips carried their text."""
    import tts
    _, scenes = read_episode(ep)
    return {tts.cache_key(tts.parse(t).speech, tts.VOICE_ID, tts.MODEL_ID, tts.SETTINGS): t
            for s in scenes for t in s["lines"]}


def dirs(ep: Path, quality: str):
    slug = ep.stem
    work = ROOT / "work" / slug / quality
    out = ROOT / "out" / slug
    for d in (work / "timeline", work / "logs", out):
        d.mkdir(parents=True, exist_ok=True)
    return slug, work, out


def selected(scenes, only):
    return [s for s in scenes if not only or s["name"] in only.split(",")]


# ---------------------------------------------------------------- tts / qa
def cmd_tts(ep, args):
    import tts
    _, scenes = read_episode(ep)
    lines = [t for s in selected(scenes, args.only) for t in s["lines"]]
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:  # the free plan allows 2 concurrent requests
        metas = list(ex.map(tts.synth, lines))
    chars = sum(len(m["speech"]) for m in metas)
    billed = sum(m.get("billed", 0) for m in metas)
    dur = sum(m["duration"] for m in metas)
    bad = [m["speech"][:30] for m in metas if m.get("alignment_mismatch")]
    print(f"tts: {len(lines)} lines, {chars} chars, {dur / 60:.1f} min of speech, {chars / max(dur, 1e-6):.2f} ch/s; "
          f"billed now {billed} chars ({time.time() - t0:.0f}s)" + (f"; alignment mismatch in {bad}" if bad else ""))


def cmd_qa(ep, args, threshold=0.12):
    import qa
    import tts
    _, scenes = read_episode(ep)
    rows = []
    for s in selected(scenes, args.only):
        for text in s["lines"]:
            m = tts.synth(text)
            side = Path(m["mp3"]).with_suffix(".qa.json")
            if side.exists():
                r = json.loads(side.read_text(encoding="utf-8"))
            else:
                hyp = qa.transcribe(m["mp3"])
                r = {"hyp": hyp, "cer": qa.cer(m["speech"], hyp)}
                side.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            rows.append((r["cer"], s["name"], m["speech"], r["hyp"]))
    flagged = sorted([r for r in rows if r[0] > threshold], reverse=True)
    mean = sum(r[0] for r in rows) / max(1, len(rows))
    print(f"qa: {len(rows)} lines, mean CER {mean:.3f}, {len(flagged)} above {threshold}")
    for c, name, ref, hyp in flagged:
        print(f"  {c:.2f} {name}\n     ref: {ref}\n     asr: {hyp}")


# ---------------------------------------------------------------- render
def manim_env(timeline_dir: Path):
    env = os.environ.copy()
    env["PATH"] = str(TEX_BIN) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    env["BB_OUT"] = str(timeline_dir)
    env["PYTHONUTF8"] = "1"
    return env


def segment_mp4(work: Path, name: str) -> Path | None:
    hits = sorted((work / "media" / name).glob(f"videos/**/{name}.mp4"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def quality_of(args) -> str:
    return "dry" if args.dry else "preview" if args.preview else "final"


def cmd_render(ep, args):
    quality = quality_of(args)
    w, h, fps = QUALITY[quality]
    slug, work, _ = dirs(ep, quality)
    _, scenes = read_episode(ep)
    todo = [s["name"] for s in selected(scenes, args.only)]
    env = manim_env(work / "timeline")
    if args.dry:
        env["BB_DRY_TTS"] = "1"

    def run(name):
        log = work / "logs" / f"{name}.log"
        cmd = [sys.executable, "-m", "manim", "render", str(ep), name, "-r", f"{w},{h}", "--fps", str(fps),
               "--media_dir", str(work / "media" / name), "-o", name, "--progress_bar", "none", "-v", "WARNING"]
        t = time.time()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        return name, rc, time.time() - t

    t0 = time.time()
    with ThreadPoolExecutor(min(args.jobs, len(todo))) as ex:
        results = list(ex.map(run, todo))
    for name, rc, dt in results:
        tl = work / "timeline" / f"{name}.timeline.json"
        info = json.loads(tl.read_text(encoding="utf-8")) if tl.exists() else {}
        print(f"  {name}: {'ok' if rc == 0 else 'FAILED'} in {dt:.0f}s, {info.get('duration', 0):.1f}s of video")
        if rc:
            tail = (work / "logs" / f"{name}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            print("    " + "\n    ".join(tail))
        for wmsg in info.get("warnings", []):
            print(f"    warn @{wmsg['t']}s: {wmsg['msg']}")
    print(f"render ({quality}): {len(todo)} segments in {time.time() - t0:.0f}s")
    if any(rc for _, rc, _ in results):
        sys.exit(1)


# ---------------------------------------------------------------- assemble
def probe_duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def load_segments(ep: Path, quality: str):
    _, work, _ = dirs(ep, quality)[0:3]
    _, scenes = read_episode(ep)
    segs, t = [], 0.0
    for s in scenes:
        mp4 = segment_mp4(work, s["name"])
        tl = work / "timeline" / f"{s['name']}.timeline.json"
        if not mp4 or not tl.exists():
            sys.exit(f"missing render for {s['name']}: run the render step first")
        d = probe_duration(mp4)
        segs.append({**s, "mp4": mp4, "tl": json.loads(tl.read_text(encoding="utf-8")), "dur": d, "t": t})
        t += d
    return segs, t


def mix_audio(segs, total: float, dst: Path):
    import numpy as np
    buf = np.zeros(int((total + 0.5) * SR), np.float32)
    for s in segs:
        for c in s["tl"]["clips"]:
            if not c["mp3"]:
                continue  # dry run: silent placeholder
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", c["mp3"], "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                                 capture_output=True, check=True).stdout
            pcm = np.frombuffer(raw, np.float32)
            i = int(round((s["t"] + c["t"]) * SR))
            n = max(0, min(len(pcm), len(buf) - i))
            buf[i:i + n] += pcm[:n]
    pcm16 = (np.clip(buf, -1, 1) * 32767).astype("<i2")
    with wave.open(str(dst), "wb") as wv:
        wv.setnchannels(1)
        wv.setsampwidth(2)
        wv.setframerate(SR)
        wv.writeframes(pcm16.tobytes())


def ts(t: float, sep=",") -> str:
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)) % 1000:03d}"


STRONG, WEAK = "。！？；…", "，、：,"


def _word_char(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in "._%")


def cue_chunks(line, max_len: int = 18):
    """Cut one narration line into subtitle chunks: at punctuation, short phrases merged, long
    ones halved evenly (never inside a Latin word or number). Yields lists of spans."""
    phrases, cur = [], []
    for sp in line.spans:
        cur.append(sp)
        ch = line.display[sp[0]:sp[1]]
        if ch and ch[-1] in STRONG + WEAK:
            phrases.append((cur, ch[-1] in STRONG))
            cur = []
    if cur:
        phrases.append((cur, True))
    size = lambda spans: spans[-1][1] - spans[0][0]
    merged = []
    for spans, strong in phrases:
        if merged and not merged[-1][1] and size(merged[-1][0]) + size(spans) <= max_len:
            merged[-1] = (merged[-1][0] + spans, strong)
        else:
            merged.append((spans, strong))
    for spans, _ in merged:
        n = -(-size(spans) // max_len)
        if n <= 1:
            yield spans
            continue
        start = 0
        for k in range(1, n):
            target = spans[0][0] + size(spans) * k / n
            best = None
            for i in range(start + 1, len(spans)):  # cut before spans[i]
                a, b = line.display[spans[i][0] - 1], line.display[spans[i][0]]
                if _word_char(a) and _word_char(b):
                    continue  # never inside a Latin word or a number
                dist = abs(spans[i][0] - target)
                if dist > 5:
                    continue
                score = 3 if " " in (a, b) else 2 if a in "的了是在和与就都也还把被给到着过" else 1 if b in "第从在把被给向" else 0
                if best is None or (score, -dist) > best[0]:
                    best = ((score, -dist), i)
            if best is None:
                continue
            yield spans[start:best[1]]
            start = best[1]
        yield spans[start:]


def clip_meta(clip: dict) -> dict:
    """Timing for one narration clip: the cached TTS json, or the silent dry-run estimate."""
    import tts
    if clip["mp3"]:
        return json.loads(Path(clip["mp3"]).with_suffix(".json").read_text(encoding="utf-8"))
    os.environ["BB_DRY_TTS"] = "1"
    try:
        return tts.synth(clip["text"])
    finally:
        del os.environ["BB_DRY_TTS"]


def write_subs(segs, out: Path, w: int, h: int, texts: dict):
    import tts
    cues = []
    for s in segs:
        for c in s["tl"]["clips"]:
            raw = c.get("text") or texts.get(Path(c["mp3"]).stem)
            if raw is None:
                continue
            line, meta = tts.parse(raw), clip_meta(c)
            ends = meta.get("ends") or []
            for spans in cue_chunks(line):
                (d0, _, s0, _), (_, d1, _, s1) = spans[0], spans[-1]
                text = line.display[d0:d1].strip().rstrip("，、：；。,")
                if not text:
                    continue
                a = s["t"] + c["t"] + tts.time_of(meta, s0)
                b = s["t"] + c["t"] + (ends[s1 - 1] if 0 < s1 <= len(ends) else tts.time_of(meta, s1))
                cues.append((a, max(b, a + 0.4), text))
    cues.sort()
    for i in range(len(cues) - 1):  # never overlap: end each cue by the next one's start
        a, b, txt = cues[i]
        cues[i] = (a, min(b + 0.25, cues[i + 1][0] - 0.02), txt)
    srt = [f"{i}\n{ts(a)} --> {ts(b)}\n{txt}\n" for i, (a, b, txt) in enumerate(cues, 1)]
    (out / "subs.srt").write_text("\n".join(srt), encoding="utf-8")
    fs = round(48 * h / 1080)
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Sub,{style()['sub_font']},{fs},&H00F0F0F0,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,1,0,1,"
            f"{max(1, round(3 * h / 1080))},0,2,{round(80 * w / 1920)},{round(80 * w / 1920)},{round(34 * h / 1080)},1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    ev = [f"Dialogue: 0,{ts(a, '.')[1:-1]},{ts(b, '.')[1:-1]},Sub,,0,0,0,,{txt}" for a, b, txt in cues]
    (out / "subs.ass").write_text(head + "\n".join(ev) + "\n", encoding="utf-8-sig")
    return len(cues)


def board_texture(w: int, h: int) -> Path:
    """RGBA overlay: vignette, chalk dust and fine grain (also roughens the chalk strokes)."""
    import numpy as np
    from PIL import Image
    p = ROOT / "cache" / f"board_texture_{w}x{h}.png"
    if p.exists():
        return p
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 * 0.9 + ((y - h / 2) / (h / 2)) ** 2 * 1.1)
    dark = np.clip((r - 0.6) / 0.7, 0, 1) ** 1.5 * 0.42 + rng.random((h, w), np.float32) ** 7 * 0.25
    lo = rng.random((h // 90 + 3, w // 90 + 3)).astype(np.float32)
    lo = np.asarray(Image.fromarray((lo * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255
    dust = np.clip((lo - 0.5) * 2.0, 0, 1) * 0.06
    a = np.clip(dark + dust, 0, 1)
    grey = np.where(a > 0, dust / np.maximum(a, 1e-6), 0) * 235
    rgba = np.dstack([grey, grey, grey, a * 255]).astype(np.uint8)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(p)
    return p


def level_strip(segs, total: float, w: int, h: int, out: Path) -> tuple[Path, list[str]]:
    """Bottom strip coloured by level; drawbox filters brighten each segment once it is done."""
    from PIL import Image, ImageDraw
    sh = max(4, round(8 * h / 1080))
    img = Image.new("RGBA", (w, sh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    boxes = []
    for s in segs:
        x0, x1 = round(w * s["t"] / total), round(w * (s["t"] + s["dur"]) / total)
        r, g, b = style()["levels"][s["level"]]
        d.rectangle([x0 + 1, 0, x1 - 2, sh - 1], fill=(r, g, b, 90))
        boxes.append(f"drawbox=x={x0 + 1}:y=ih-{sh}:w={max(1, x1 - x0 - 3)}:h={sh}:color=0x{r:02X}{g:02X}{b:02X}@0.95:"
                     f"t=fill:enable='gte(t,{s['t'] + s['dur']:.2f})'")
    p = out / "strip.png"
    img.save(p)
    return p, boxes


def write_chapters(segs, out: Path, meta: dict):
    lines = []
    for s in segs:
        m, sec = divmod(int(s["t"]), 60)
        num = CIRCLED[len(lines)] if len(lines) < len(CIRCLED) else str(len(lines) + 1)
        lines.append(f"{m:02d}:{sec:02d} {num} {s['title']}【{LEVEL_NAME[s['level']]}】")
    (out / "chapters.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ff = [";FFMETADATA1", f"title={meta.get('title', out.name)}"]
    for s in segs:
        ff += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(s['t'] * 1000)}", f"END={int((s['t'] + s['dur']) * 1000)}",
               f"title={s['title']}（{LEVEL_NAME[s['level']]}）"]
    (out / "chapters.ffmeta").write_text("\n".join(ff) + "\n", encoding="utf-8")
    return lines


def cmd_assemble(ep, args):
    quality = quality_of(args)
    w, h, fps = QUALITY[quality]
    slug, work, out = dirs(ep, quality)
    meta, _ = read_episode(ep)
    segs, total = load_segments(ep, quality)
    t0 = time.time()
    tag = TAG[quality]
    mix_audio(segs, total, work / "narration.wav")
    (work / "concat.txt").write_text("".join(f"file '{s['mp4'].as_posix()}'\n" for s in segs), encoding="utf-8")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "concat.txt"),
                    "-c", "copy", str(work / "concat.mp4")], check=True)
    n_cues = write_subs(segs, out, w, h, narration_by_key(ep))
    chapters = write_chapters(segs, out, meta)
    tex = board_texture(w, h)
    strip, boxes = level_strip(segs, total, w, h, out)
    ph_h = max(6, round(16 * h / 1080))
    first = "[0:v][2:v]overlay=0:0:shortest=1[a];" if style()["texture"] else "[0:v]null[a];"
    graph = (f"{first}[a][3:v]overlay=0:H-h:shortest=1[b];"
             f"[b]{','.join(boxes)}[c];"
             f"[4:v]format=rgba[ph];[c][ph]overlay=x='(W-w)*t/{total:.3f}':y=H-h:eval=frame:shortest=1[d];"
             f"[d]subtitles=subs.ass[v]")
    final = out / f"{slug}{tag}.mp4"
    enc = (["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "20", "-b:v", "0"]
           if not args.cpu else ["-c:v", "libx264", "-preset", "medium", "-crf", "19"])
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(work / "concat.mp4"), "-i", str(work / "narration.wav"),
           "-loop", "1", "-i", str(tex), "-loop", "1", "-i", str(strip),
           "-f", "lavfi", "-i", f"color=c=white:s=4x{ph_h}:r={fps}", "-f", "ffmetadata", "-i", str(out / "chapters.ffmeta"),
           "-filter_complex", graph, "-map", "[v]", "-map", "1:a", "-map_metadata", "5", "-map_chapters", "5",
           *enc, "-pix_fmt", "yuv420p", "-r", str(fps),
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-t", f"{total:.3f}", str(final)]
    r = subprocess.run(cmd, cwd=out, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode and not args.cpu:
        print("nvenc failed, falling back to libx264:", r.stderr[-300:])
        args.cpu = True
        return cmd_assemble(ep, args)
    if r.returncode:
        sys.exit(r.stderr[-2000:])
    summary = {"file": str(final), "duration": round(total, 2), "segments": [
        {"name": s["name"], "title": s["title"], "level": s["level"], "start": round(s["t"], 2), "dur": round(s["dur"], 2)}
        for s in segs]}
    (out / f"summary{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    zero = sum(s["dur"] for s in segs if s["level"] == "zero")
    print(f"assemble: {final} ({total / 60:.1f} min, {n_cues} subtitle cues, {zero / total:.0%} 零基础) in {time.time() - t0:.0f}s")
    print("  " + "\n  ".join(chapters))


# ---------------------------------------------------------------- per-chapter files
def cmd_split(ep, args, cap_mib: float = 19.0):
    """Cut the finished video into one file per chapter (each under cap_mib, the web page's per-file
    limit is 20 MiB) plus chapters.json with titles, levels and times."""
    quality = quality_of(args)
    slug, work, out = dirs(ep, quality)
    segs, _ = load_segments(ep, quality)
    final = out / f"{slug}{TAG[quality]}.mp4"
    cdir = out / "chapters"
    cdir.mkdir(exist_ok=True)
    items = []
    for i, s in enumerate(segs, 1):
        dst = cdir / f"{i:02d}-{s['name']}.mp4"
        for cq in (22, 26, 30, 34):
            enc = (["-c:v", "libx264", "-preset", "medium", "-crf", str(cq - 3)] if args.cpu else
                   ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"])
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s['t']:.3f}", "-i", str(final), "-t", f"{s['dur']:.3f}",
                            *enc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                            str(dst)], check=True)
            if dst.stat().st_size <= cap_mib * 2 ** 20:
                break
        items.append({"n": i, "file": dst.name, "title": s["title"], "level": s["level"],
                      "start": round(s["t"], 2), "dur": round(s["dur"], 2), "mib": round(dst.stat().st_size / 2 ** 20, 1)})
    (cdir / "chapters.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print("split:", ", ".join(f"{it['n']}:{it['mib']}MiB" for it in items))


def cmd_page(ep, args):
    """Build the private video page: index.html from assets/player.html, poster.jpg and chapters/.
    Artifact publishing only reads from the repo or the session scratchpad, so pass --page-dir there."""
    import shutil
    quality = quality_of(args)
    slug, _, out = dirs(ep, quality)
    meta, _ = read_episode(ep)
    items = json.loads((out / "chapters" / "chapters.json").read_text(encoding="utf-8"))
    dst = Path(args.page_dir) if args.page_dir else out / "page"
    (dst / "chapters").mkdir(parents=True, exist_ok=True)
    for it in items:
        shutil.copy2(out / "chapters" / it["file"], dst / "chapters" / it["file"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(meta.get("poster_at", 60)), "-i",
                    str(out / f"{slug}{TAG[quality]}.mp4"), "-frames:v", "1", str(dst / "poster.jpg")], check=True)
    chapters = [{"title": it["title"], "level": it["level"], "dur": it["dur"], "src": "chapters/" + it["file"]}
                for it in items]
    html = (HERE.parent / "assets" / "player.html").read_text(encoding="utf-8")
    fills = {"__CHAPTERS__": json.dumps(chapters, ensure_ascii=False), "__POSTER__": "poster.jpg",
             "__PAGE_TITLE__": meta.get("page_title", slug), "__KICKER__": meta.get("kicker", ""),
             "__HEADLINE__": meta.get("headline", meta.get("title", slug)), "__LEDE__": meta.get("lede", ""),
             "__SOURCE__": meta.get("source", "")}
    for k, v in fills.items():
        html = html.replace(k, v)
    (dst / "index.html").write_text(html, encoding="utf-8")
    print(f"page: {dst} (publish index.html with poster.jpg and chapters/*.mp4 as files)")


# ---------------------------------------------------------------- contact sheet
def cmd_sheet(ep, args, per_sheet=12, cols=3, tile_w=640):
    from PIL import Image, ImageDraw, ImageFont
    quality = quality_of(args)
    slug, work, out = dirs(ep, quality)
    segs, total = load_segments(ep, quality)
    final = out / f"{slug}{TAG[quality]}.mp4"
    shots = [(s["name"], min(s["t"] + t, s["t"] + s["dur"] - 0.15, total - 0.3)) for s in segs
             if not args.only or s["name"] in args.only.split(",") for t in s["tl"]["snaps"]]
    tile_h = tile_w * 9 // 16
    font = ImageFont.truetype("msyh.ttc", 22)
    for old in out.glob("sheet-*.png"):
        old.unlink()
    paths = []
    for k in range(0, len(shots), per_sheet):
        batch = shots[k:k + per_sheet]
        rows = (len(batch) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (0, 0, 0))
        for i, (name, t) in enumerate(batch):
            raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.2f}", "-i", str(final), "-frames:v", "1",
                                  "-vf", f"scale={tile_w}:{tile_h}", "-f", "image2pipe", "-vcodec", "png", "-"],
                                 capture_output=True, check=True).stdout
            from io import BytesIO
            im = Image.open(BytesIO(raw)).convert("RGB")
            ImageDraw.Draw(im).text((8, tile_h - 30), f"{name} @{int(t) // 60}:{int(t) % 60:02d}", fill=(255, 80, 80),
                                    font=font)
            sheet.paste(im, ((i % cols) * tile_w, (i // cols) * tile_h))
        p = out / f"sheet-{k // per_sheet + 1}.png"
        sheet.save(p)
        paths.append(p)
    print("sheet:", ", ".join(str(p) for p in paths))


def main():
    _reexec_in_venv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["tts", "qa", "render", "assemble", "sheet", "split", "page", "all"])
    ap.add_argument("episode", type=Path)
    ap.add_argument("--preview", action="store_true", help="854x480 at 15 fps")
    ap.add_argument("--dry", action="store_true", help="preview size with silent placeholder narration (no TTS, no cost)")
    ap.add_argument("--only", default="", help="comma-separated segment classes (tts/qa/render/sheet)")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--cpu", action="store_true", help="encode with libx264 instead of NVENC")
    ap.add_argument("--qa", action="store_true", help="with 'all': run the Whisper listen-back after tts")
    ap.add_argument("--page-dir", default="", help="'page': output folder (use the session scratchpad to publish)")
    args = ap.parse_args()
    ep = args.episode.resolve()
    sys.path.insert(0, str(HERE))
    apply_episode_settings(ep)
    steps = {"tts": [cmd_tts], "qa": [cmd_qa], "render": [cmd_render], "assemble": [cmd_assemble],
             "sheet": [cmd_sheet], "split": [cmd_split], "page": [cmd_page],
             "all": ([] if args.dry else [cmd_tts] + ([cmd_qa] if args.qa else []))
                    + [cmd_render, cmd_assemble, cmd_sheet]}[args.step]
    for f in steps:
        f(ep, args)


if __name__ == "__main__":
    main()
