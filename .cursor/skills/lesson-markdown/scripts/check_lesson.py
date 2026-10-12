#!/usr/bin/env python3
"""Validate a Markdown lesson and its built HTML.

    python check_lesson.py deepseek/src/lesson-08-mla.md          # build + static + DOM checks
    python check_lesson.py deepseek/src/*.md --no-dom             # skip the headless-Chrome pass

Checks
  build        the source builds (all citations resolve, every ref is cited, containers close)
  svg          every figure has a viewBox, no width/height, and no hard-coded colour outside a
               var(--fig-*, #fallback); text is at least 11 user units
  dom          headless Chrome renders the page with KaTeX, then the DOM is scanned for
               .katex-error nodes (bad TeX) and for raw \\( \\[ delimiters left untypeset
  prose        warnings only: English em dashes, 'clearly/obviously/simply', 显然/易证/不难看出
Exit status is 1 if any error was found.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_lesson  # noqa: E402
from shoot import chrome, file_url  # noqa: E402

HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
VAR_FALLBACK = re.compile(r"var\(--fig-[a-z0-9-]+\s*,\s*#[0-9a-fA-F]{3,8}\s*\)")
NAMED = re.compile(r'(?:fill|stroke|stop-color|color)\s*[:=]\s*"?\s*(black|white|red|blue|green|gray|grey|orange|purple)\b')


def check_svg(path):
    errs, warns = [], []
    with open(path, encoding="utf-8") as fh:
        svg = fh.read()
    root = re.search(r"<svg\b[^>]*>", svg, re.S)
    if not root:
        return ["no <svg> root"], []
    if "viewBox" not in root.group(0):
        errs.append("root <svg> has no viewBox")
    stripped = VAR_FALLBACK.sub("", svg)
    stripped = re.sub(r"<!--.*?-->", "", stripped, flags=re.S)
    for m in HEX.finditer(stripped):
        ctx = stripped[max(0, m.start() - 30):m.end() + 5].replace("\n", " ")
        if "url(" in ctx or 'href="#' in ctx:
            continue
        errs.append("hard-coded colour %s (use var(--fig-*, %s)): ...%s..." % (m.group(0), m.group(0), ctx))
        break
    for m in NAMED.finditer(stripped):
        errs.append("named colour %r; use var(--fig-*)" % m.group(1))
        break
    for m in re.finditer(r'font-size="([\d.]+)"', svg):
        if float(m.group(1)) < 11:
            warns.append("font-size %s < 11 user units" % m.group(1))
            break
    return errs, warns


def dom_check(html_path):
    errs = []
    with tempfile.TemporaryDirectory() as prof:
        cmd = [chrome(), "--headless=new", "--disable-gpu", "--no-first-run", "--user-data-dir=" + prof,
               "--virtual-time-budget=10000", "--dump-dom", file_url(html_path, "all&light")]
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
    dom = proc.stdout.decode("utf-8", "replace")
    if "math-ready" not in dom:
        return ["KaTeX never ran (offline? jsDelivr blocked?) - DOM check inconclusive"]
    for m in re.finditer(r'<span class="katex-error"[^>]*title="([^"]*)"[^>]*>(.*?)</span>', dom, re.S):
        errs.append("KaTeX: %s  in: %s" % (html.unescape(m.group(1))[:160], html.unescape(m.group(2))[:100]))
    # with throwOnError:false an undefined macro is not an error node: it is drawn in errorColor
    for m in re.finditer(r'mathcolor="#cc0000"><mtext>(.*?)</mtext>.*?<annotation encoding="application/x-tex">(.*?)</annotation>',
                         dom, re.S):
        errs.append("KaTeX: undefined control sequence %s  in: %s"
                    % (html.unescape(m.group(1)), html.unescape(m.group(2))[:100]))
    text = re.sub(r"<(script|style|annotation)\b.*?</\1>", "", dom, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    for m in re.finditer(r"\\\(|\\\[", text):
        errs.append("untypeset math near: " + html.unescape(text[m.start():m.start() + 80]).replace("\n", " "))
        if len(errs) > 20:
            break
    return errs


def prose_warnings(src):
    warns = []
    body = re.sub(r"```.*?```", "", src, flags=re.S)
    body = re.sub(r"\$\$.*?\$\$", "", body, flags=re.S)
    for pat, msg in ((r"[A-Za-z] ?— ?[A-Za-z]", "English em dash"),
                     (r"\b(clearly|obviously|simply)\b", "banned hedge word"),
                     (r"(显然|易证|不难看出|容易看出|众所周知)", "banned 'obviously' phrase")):
        for m in re.finditer(pat, body):
            line = body.count("\n", 0, m.start()) + 1
            warns.append("%s at line ~%d: %s" % (msg, line, body[max(0, m.start() - 20):m.end() + 20].replace("\n", " ")))
    return warns


def feynman_warnings(src):
    """Once a lesson has the Feynman layer (style-feynman.md), every teaching page needs a
    teach-it-back box, and the lesson needs its review page. Warnings, not errors."""
    body = re.sub(r"```.*?```", "", src, flags=re.S)
    if not re.search(r"^:{3,}\s*feynman\b", body, re.M):
        return []
    warns = []
    chunks = re.split(r"^# (.+)$", body, flags=re.M)
    titles = [chunks[j].split("|")[0].strip() for j in range(1, len(chunks), 2)]
    for n, (title, page) in enumerate(zip(titles, chunks[2::2]), 1):
        if re.search(r"练习|复盘|习题", title):
            continue
        if not re.search(r"^:{3,}\s*feynman\b", page, re.M):
            warns.append("Feynman layer: §%d「%s」has no ::: feynman box" % (n, title))
    if not any("复盘" in t for t in titles):
        warns.append("Feynman layer: no 费曼复盘 page")
    if not re.search(r"^:{3,}\s*predict\b[^\n]*\{#before-1\}", chunks[0], re.M):
        warns.append("Feynman layer: the cover has no ::: predict … {#before-1} box")
    return warns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="+")
    ap.add_argument("--no-dom", action="store_true")
    a = ap.parse_args()
    bad = 0
    for src in a.src:
        print("== %s" % src)
        try:
            out = build_lesson.build(src, quiet=True)
        except build_lesson.BuildError as exc:
            print("  ERROR build: %s" % exc)
            bad += 1
            continue
        with open(src, encoding="utf-8") as fh:
            text = fh.read()
        out_dir = os.path.dirname(out)
        errs, warns = [], []
        for path in sorted(set(re.findall(r"^:{3,}\s*fig\s+([^|\n]+)", text, re.M))):
            full = os.path.normpath(os.path.join(out_dir, path.strip()))
            e, w = check_svg(full)
            errs += ["svg %s: %s" % (path.strip(), x) for x in e]
            warns += ["svg %s: %s" % (path.strip(), x) for x in w]
        warns += prose_warnings(text)
        warns += feynman_warnings(text)
        for n, line in enumerate(text.splitlines(), 1):
            m = re.search(r"[\x00-\x08\x0b-\x1f\x7f]", line)
            if m:
                errs.append("control character %r at line %d (a stray tab or backspace escape eats TeX such as "
                            "the transpose macro): %s" % (m.group(0), n, line.strip()[:80]))
        with open(out, encoding="utf-8") as fh:
            built = fh.read()
        mc = re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", built)
        if mc:
            errs.append("built HTML contains control character %r near: %r"
                        % (mc.group(0), built[max(0, mc.start() - 60):mc.start() + 20]))
        if not a.no_dom:
            errs += ["dom: " + x for x in dom_check(out)]
        for e in errs:
            print("  ERROR " + e)
        for w in warns:
            print("  warn  " + w)
        n_eq = len(re.findall(r"\$\$", text)) // 2
        n_fig = len(re.findall(r"^:{3,}\s*fig\b", text, re.M))
        n_cjk = len(re.findall("[一-鿿]", text))
        print("  stats: %d CJK chars, %d display equations, %d figures, %s kB html"
              % (n_cjk, n_eq, n_fig, round(os.path.getsize(out) / 1000)))
        if errs:
            bad += 1
        else:
            print("  OK")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    main()
