#!/usr/bin/env python3
"""Compile Tailwind for a lesson HTML file and inline the result.

    python3 build_css.py ../../../../mamba/lesson-01-....html

Scans the target HTML for utility classes, compiles assets/lesson.css with the
Tailwind standalone CLI, and replaces everything between

    <style id="tw">  ...  </style>

with the minified output. The published file therefore has zero runtime CSS
dependency. Re-run after any change to the markup's class attributes.

Requires the Tailwind standalone binary (no Node). Install once:

    curl -sL -o ~/.local/bin/tailwindcss \\
      https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-macos-arm64
    chmod +x ~/.local/bin/tailwindcss
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "assets", "lesson.css"))
MARK = re.compile(r'(<style id="tw"[^>]*>)(.*?)(</style>)', re.S)


def tailwind_bin() -> str:
    for cand in (os.path.expanduser("~/.local/bin/tailwindcss"),
                 shutil.which("tailwindcss") or ""):
        if cand and os.path.exists(cand):
            return cand
    sys.exit("tailwindcss standalone binary not found — see the header of this file")


def build(html_path: str) -> None:
    html_path = os.path.abspath(html_path)
    if not os.path.exists(html_path):
        sys.exit("no such file: " + html_path)
    with open(html_path, encoding="utf-8") as fh:
        html = fh.read()
    if not MARK.search(html):
        sys.exit('target must contain a <style id="tw"></style> block')

    with tempfile.TemporaryDirectory() as tmp:
        # Build in a scratch dir so Tailwind's automatic source detection cannot
        # pick up the repo; the one @source below is the only thing scanned.
        shutil.copy(SRC, os.path.join(tmp, "lesson.css"))
        entry = os.path.join(tmp, "entry.css")
        with open(entry, "w", encoding="utf-8") as fh:
            fh.write('@import "./lesson.css";\n@source "%s";\n'
                     % html_path.replace('"', '\\"'))
        out = os.path.join(tmp, "out.css")
        proc = subprocess.run(
            [tailwind_bin(), "-i", entry, "-o", out, "--minify", "--cwd", tmp],
            capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit("tailwind build failed:\n" + proc.stderr)
        with open(out, encoding="utf-8") as fh:
            css = fh.read().strip()

    new = MARK.sub(lambda m: m.group(1) + "\n" + css + "\n" + m.group(3), html, count=1)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(new)
    print("%s: inlined %.1f kB of CSS" % (os.path.basename(html_path), len(css) / 1000))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(sys.argv[1])
