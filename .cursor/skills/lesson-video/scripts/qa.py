"""Listen-back check for synthesised narration.

Nobody on the pipeline side can hear the audio, so each clip is transcribed with a local
Whisper model and compared with the text it was supposed to say. The character error rate (CER)
after normalisation flags clips where the voice skipped, repeated or garbled words; those lines
get re-worded or re-synthesised before a render. Runs on the GPU when CUDA libraries are present,
otherwise on the CPU (slower, same result).
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(os.environ.get("BB_VIDEO_ROOT", r"D:\StudyNotes-video"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))

_DIG = "零一二三四五六七八九"
_model = None


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


def _num(m: re.Match) -> str:
    whole, frac = m.group(1).replace(",", ""), m.group(2)
    s = int2zh(int(whole))
    if frac:
        s += "点" + "".join(_DIG[int(c)] for c in frac)
    return s


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"(\d[\d,]*(?:\.\d+)?)%", r"百分之\1", t)
    t = re.sub(r"(\d[\d,]*)(?:\.(\d+))?", _num, t)
    t = t.replace("两", "二").replace("〇", "零")
    return "".join(ch for ch in t if ch.isalnum())


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def _cuda_dlls():
    """Expose the pip-installed CUDA 12 cuBLAS/cuDNN DLLs (nvidia-*-cu12 wheels) to CTranslate2."""
    import site
    for sp in site.getsitepackages():
        for bin_dir in Path(sp, "nvidia").glob("*/bin"):
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(bin_dir))


def _load():
    global _model
    if _model is None:
        import numpy as np
        _cuda_dlls()
        from faster_whisper import WhisperModel
        root = str(ROOT / "models")
        try:
            _model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16", download_root=root)
            list(_model.transcribe(np.zeros(16000, np.float32), language="zh")[0])  # fails fast without CUDA libs
        except Exception:
            _model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", download_root=root)
    return _model


def transcribe(mp3: str) -> str:
    segs, _ = _load().transcribe(mp3, language="zh", beam_size=5, vad_filter=False,
                                 initial_prompt="以下是普通话的课程讲解，使用简体中文。")
    return "".join(s.text for s in segs).strip()
