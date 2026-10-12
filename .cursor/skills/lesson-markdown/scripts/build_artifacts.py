#!/usr/bin/env python3
"""Build the claude.ai Artifact variant of every Markdown lesson and both course indexes.

    python build_artifacts.py <out_dir>                                   # first pass, no cross-links
    python build_artifacts.py <out_dir> --link-map deepseek/reference-docs/artifacts.json

Writes <out_dir>/deepseek-lesson-09-v2-mla.html, <out_dir>/math-m02-rank-svd-lowrank.html, … and
<out_dir>/deepseek-index.html, <out_dir>/math-index.html. The file names are stable so that
republishing the same file keeps its Artifact URL within a session; across sessions, publish
with the URL recorded in artifacts.json. The link map (repo-relative path -> Artifact URL)
rewrites cross-lesson links so they open the other lessons' Artifacts.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_index  # noqa: E402
import build_lesson  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))

SHORT_INDEX = {"deepseek": "DeepSeek 科技树目录", "math": "数学补课目录"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--link-map")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    link_map = {}
    if a.link_map:
        with open(a.link_map, encoding="utf-8") as fh:
            link_map = json.load(fh)
    failed = 0
    for track in ("deepseek", "math"):
        for src in sorted(glob.glob(os.path.join(ROOT, track, "src", "*.md"))):
            slug = os.path.splitext(os.path.basename(src))[0]
            out = os.path.join(a.out_dir, "%s-%s.html" % (track, slug))
            try:
                build_lesson.build(src, out, quiet=True, link_map=link_map or None)
                print(out)
            except build_lesson.BuildError as exc:
                failed += 1
                print("ERROR %s: %s" % (src, exc), file=sys.stderr)
        idx = os.path.join(ROOT, track, "src", "index.yaml")
        build_index.build_artifact(idx, os.path.join(a.out_dir, "%s-index.html" % track), link_map, SHORT_INDEX[track])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    main()
