"""ElevenLabs narration with a content-addressed cache.

Every narration line is synthesised once per (text, voice, model, settings) and stored as
<cache>/<key>.mp3 plus <key>.json (duration and per-character timing from the
/with-timestamps endpoint). Re-rendering an episode never re-bills a line whose spoken text did
not change, and a lock file keeps parallel renders from billing the same line twice.

Narration markup (shared by the Manim kit and the subtitle writer):
    [shown|spoken]   the subtitle shows "shown", the voice says "spoken", e.g. [√d_k|根号 d k]
    {name}           a bookmark; animations can wait until the voice reaches it
With BB_SPEAK_NUMBERS=zh (an episode's "speak_numbers": "zh"), bare numbers become
[64|六十四]-style pairs automatically.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("BB_VIDEO_ROOT", r"D:\StudyNotes-video"))
CACHE = ROOT / "cache" / "tts"
REPO = Path(__file__).resolve().parents[4]

# Chosen 2026-09-28 by a listen-back test (qa.py): the account is on the free plan, which cannot call
# Voice Library voices, so the voice is a premade one. On eleven_v3 the premade male voices read
# Mandarin at 3-5% CER (19-25% on eleven_multilingual_v2); "Daniel - Steady Broadcaster" scored the
# highest Whisper confidence. Swap the id (or set BB_VOICE_ID) for a library or cloned voice later.
VOICE_ID = os.environ.get("BB_VOICE_ID", "onwK4e9ZLuTAKqWW03F9")
MODEL_ID = os.environ.get("BB_MODEL_ID", "eleven_v3")
SETTINGS = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True, "speed": 1.15}

_PIECE = re.compile(r"\[([^\[\]|]*)\|([^\[\]]*)\]|\{(\w+)\}|(.)", re.S)
_MARKUP = re.compile(r"\[[^\[\]|]*\|[^\[\]]*\]|\{\w+\}")
_NUM = re.compile(r"(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%)?(?![A-Za-z0-9_])")
_DIG = "零一二三四五六七八九"


def _four(n: int) -> str:
    out, zero = "", False
    for i, unit in ((3, "千"), (2, "百"), (1, "十"), (0, "")):
        d = n // 10 ** i % 10
        if d == 0:
            zero = bool(out)
        else:
            out += ("零" if zero else "") + _DIG[d] + unit
            zero = False
    return out


def int2zh(n: int) -> str:
    """Integer as read aloud in Mandarin: 768 -> 七百六十八, 1024 -> 一千零二十四, 2362368 -> 二百三十六万二千三百六十八."""
    if n == 0:
        return "零"
    groups = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    out = ""
    for gi in range(len(groups) - 1, -1, -1):
        g = groups[gi]
        if g == 0:
            continue
        if out and g < 1000:
            out += "零"
        out += _four(g) + ("", "万", "亿", "万亿")[gi]
    return out[1:] if out.startswith("一十") else out


def speak_numbers(text: str) -> str:
    """Wrap every bare number in [digits|Chinese reading] so the subtitle keeps the digits and the voice
    reads it the Chinese way (premade voices misread Arabic numerals next to English words)."""
    def one(seg: str) -> str:
        def rep(m: re.Match) -> str:
            whole, frac, pct = m.group(1), m.group(2), m.group(3)
            after = seg[m.end():].lstrip()
            if frac is None and not pct and len(whole) == 4 and after.startswith("年"):
                spoken = "".join(_DIG[int(c)] for c in whole)  # years digit by digit
            elif whole == "2" and frac is None and not pct and after[:1] in tuple("个次位种项倍张条对组层"):
                spoken = "两"
            else:
                spoken = int2zh(int(whole.replace(",", "")))
            if frac:
                spoken += "点" + "".join(_DIG[int(c)] for c in frac)
            return f"[{m.group(0)}|{'百分之' + spoken if pct else spoken}]"
        seg = _NUM.sub(rep, seg)
        # a space between a Chinese character and a number reads as a pause: keep it on screen only
        seg = re.sub(r"(?<=[一-鿿]) \[([^\[\]|]*)\|", r"[ \1|", seg)
        return re.sub(r"\[([^\[\]|]*)\|([^\[\]]*)\] (?=[一-鿿])", r"[\1 |\2]", seg)

    out, pos = [], 0
    for m in _MARKUP.finditer(text):
        out += [one(text[pos:m.start()]), m.group(0)]
        pos = m.end()
    return "".join(out + [one(text[pos:])])


@dataclass
class Line:
    """One narration line split into what is spoken and what is shown."""
    speech: str
    display: str
    marks: dict[str, int] = field(default_factory=dict)          # bookmark -> index into speech
    spans: list[tuple[int, int, int, int]] = field(default_factory=list)  # (d0, d1, s0, s1) pieces


def parse(text: str) -> Line:
    if os.environ.get("BB_SPEAK_NUMBERS") == "zh":
        text = speak_numbers(text)
    speech, display, marks, spans = [], [], {}, []
    s = d = 0
    for m in _PIECE.finditer(text):
        shown, spoken, mark, ch = m.groups()
        if mark:
            marks[mark] = s
            continue
        if ch is not None:
            shown = spoken = ch
        speech.append(spoken)
        display.append(shown)
        spans.append((d, d + len(shown), s, s + len(spoken)))
        s += len(spoken)
        d += len(shown)
    return Line("".join(speech), "".join(display), marks, spans)


def load_env() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        if "=" in raw and not raw.lstrip().startswith("#"):
            k, v = raw.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def cache_key(speech: str, voice: str, model: str, settings: dict) -> str:
    blob = json.dumps({"t": speech, "v": voice, "m": model, "s": settings}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:20]


def _duration(mp3: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def _post(voice: str, body: dict) -> dict:
    load_env()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set (put it in the repo's .env)")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=mp3_44100_128"
    data = json.dumps(body).encode("utf-8")
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"xi-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode("utf-8", "replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt + 1)
                continue
            raise RuntimeError(f"ElevenLabs HTTP {e.code}: {detail}") from None
        except urllib.error.URLError:
            if attempt < 5:
                time.sleep(2 ** attempt + 1)
                continue
            raise
    raise RuntimeError("unreachable")


def synth(text: str, voice: str | None = None, model: str | None = None, settings: dict | None = None,
          cache: Path = CACHE) -> dict:
    """Return {"mp3", "duration", "starts", "ends", "speech", ...} for one narration line."""
    line = parse(text)
    if os.environ.get("BB_DRY_TTS"):  # layout passes: silent, 4.3 characters per second, nothing billed
        n, rate = len(line.speech), 4.3
        return {"speech": line.speech, "duration": n / rate + 0.2, "mp3": None, "dry": True,
                "starts": [i / rate for i in range(n)], "ends": [(i + 1) / rate for i in range(n)]}
    voice, model = voice or VOICE_ID, model or MODEL_ID
    settings = {**SETTINGS, **(settings or {})}
    k = cache_key(line.speech, voice, model, settings)
    cache.mkdir(parents=True, exist_ok=True)
    mp3, meta, lock = cache / f"{k}.mp3", cache / f"{k}.json", cache / f"{k}.lock"
    for _ in range(600):
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            m["mp3"] = str(mp3)
            return m
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.5)  # another process is synthesising this line
    try:
        body = {"text": line.speech, "model_id": model, "voice_settings": settings}
        r = _post(voice, body)
        mp3.write_bytes(base64.b64decode(r["audio_base64"]))
        a = r.get("alignment") or {}
        chars = a.get("characters") or []
        m = {"key": k, "speech": line.speech, "voice": voice, "model": model, "settings": settings,
             "duration": _duration(mp3), "chars": chars,
             "starts": a.get("character_start_times_seconds") or [],
             "ends": a.get("character_end_times_seconds") or []}
        if "".join(chars) != line.speech:
            m["alignment_mismatch"] = True
        tmp = meta.with_suffix(".tmp")
        tmp.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, meta)
        with open(cache / "usage.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{len(line.speech)}\t{voice}\t{model}\t{k}\n")
        m["mp3"] = str(mp3)
        m["billed"] = len(line.speech)
        return m
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def time_of(meta: dict, index: int) -> float:
    """Seconds into the clip at which the voice reaches speech[index] (index may equal len)."""
    starts, ends = meta.get("starts") or [], meta.get("ends") or []
    if not starts:
        return meta["duration"] * index / max(1, len(meta["speech"]))
    if index >= len(starts):
        return ends[-1] if ends else meta["duration"]
    return starts[index]
