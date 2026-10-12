#!/usr/bin/env python3
"""Screenshot pages of a built lesson with headless Chrome, then trim the blank tail.

    python shoot.py deepseek/lesson-08-mla.html                 # every page, light
    python shoot.py deepseek/lesson-08-mla.html --pages 0,3-5 --dark --out /tmp/shots
    python shoot.py deepseek/lesson-08-mla.html --width 420     # phone width

Writes <out>/<basename>-p<N>[-dark].png and prints the paths, so the PNGs can be
opened with an image viewer (or the Read tool) and inspected. KaTeX needs the
network (jsDelivr) and a virtual-time budget to finish typesetting; the default
budget of 8000 ms is enough on a normal connection.

Chrome is looked up in $CHROME, then the usual Windows / macOS / Linux locations.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse

CANDIDATES = [
    os.environ.get("CHROME", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]


def chrome() -> str:
    for c in CANDIDATES:
        if c and os.path.exists(c):
            return c
    sys.exit("Chrome not found; set $CHROME")


def file_url(path: str, query: str = "", frag: str = "") -> str:
    p = os.path.abspath(path).replace(os.sep, "/")
    if not p.startswith("/"):
        p = "/" + p
    return "file://" + urllib.parse.quote(p) + (("?" + query) if query else "") + (("#" + frag) if frag else "")


def count_pages(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        return len(re.findall(r'<section class="page', fh.read()))


def parse_pages(spec: str, n: int):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [p for p in out if 0 <= p < n]


def trim(png: str, bg_tol: int = 6, crop_w: int = 0):
    try:
        from PIL import Image
    except ImportError:
        return
    im = Image.open(png).convert("RGB")
    if crop_w:
        im = im.crop((0, 0, crop_w, im.size[1]))
    w, h = im.size
    bg = im.getpixel((w - 3, h - 3))
    px = im.load()
    last = h - 1
    step = max(1, w // 60)
    while last > 200:
        row_diff = any(max(abs(px[x, last][c] - bg[c]) for c in range(3)) > bg_tol for x in range(0, w, step))
        if row_diff:
            break
        last -= 1
    im.crop((0, 0, w, min(h, last + 40))).save(png)


def shoot(path, pages, dark, width, height, budget, out_dir, slice_h=1600):
    exe = chrome()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    results = []
    with tempfile.TemporaryDirectory() as prof:
        for p in pages:
            png = os.path.join(out_dir, "%s-p%d%s%s.png" % (base, p, "-dark" if dark else "",
                                                             "" if width >= 700 else "-w%d" % width))
            url = file_url(path, "shot&open&dark" if dark else "shot&open&light", "p%d" % p)
            win_w = width
            if width < 500:
                # headless Chrome will not make a window narrower than 500 px; render the page
                # inside an iframe of the requested width so media queries see a phone viewport
                wrapper = os.path.join(prof, "narrow-%d.html" % p)
                with open(wrapper, "w", encoding="utf-8") as fh:
                    fh.write('<!doctype html><html><body style="margin:0">'
                             '<iframe src="%s" style="border:0;width:%dpx;height:%dpx;display:block">'
                             '</iframe></body></html>' % (url.replace('"', "%22"), width, height))
                url = file_url(wrapper)
                win_w = 520
            cmd = [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                   "--allow-file-access-from-files",
                   "--user-data-dir=" + prof, "--virtual-time-budget=%d" % budget,
                   "--window-size=%d,%d" % (win_w, height), "--screenshot=" + png, url]
            subprocess.run(cmd, capture_output=True, timeout=120)
            if os.path.exists(png):
                trim(png, crop_w=width if width < 500 else 0)
                results.extend(slice_png(png, slice_h))
            else:
                print("failed:", url, file=sys.stderr)
    return results


def slice_png(png, slice_h):
    """Split a tall page screenshot into strips a viewer can read at full size."""
    if not slice_h:
        return [png]
    try:
        from PIL import Image
    except ImportError:
        return [png]
    im = Image.open(png)
    w, h = im.size
    if h <= slice_h * 1.25:
        return [png]
    out = []
    stem = png[:-4]
    for k, top in enumerate(range(0, h, slice_h)):
        part = "%s-s%d.png" % (stem, k + 1)
        im.crop((0, top, w, min(h, top + slice_h + 60))).save(part)
        out.append(part)
    im.close()
    os.remove(png)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("html")
    ap.add_argument("--pages", default="")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=16000)
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "lesson-shots"))
    ap.add_argument("--slice", type=int, default=1600,
                    help="split pages taller than this into strips (px, 0 = keep one image)")
    a = ap.parse_args()
    n = max(1, count_pages(a.html))
    for png in shoot(a.html, parse_pages(a.pages, n), a.dark, a.width, a.height, a.budget, a.out, a.slice):
        print(png)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    main()
