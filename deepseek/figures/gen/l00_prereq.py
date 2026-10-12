"""Figure 0.8: what each DeepSeek lesson assumes, read directly from the lesson sources.

For every deepseek/src/lesson-NN-*.md (NN >= 1) the script parses
  * the 'meta:' front-matter line: "前置：第 x、y 课 ... 数学 Mxx ..."  -> declared prerequisites
  * every in-text link '../math/mNN-...html#pK'                     -> math sections the lesson points to
and draws one row per lesson: filled dots = lessons named in 'meta', filled squares = math lessons named
in 'meta', hollow squares = math lessons only linked from the text. Re-run it after editing a lesson's meta.

    python deepseek/figures/gen/l00_prereq.py
"""

from __future__ import annotations

import glob
import os
import re

from l00_svg import SRC_DIR, SVG


def parse():
    rows = []
    for path in sorted(glob.glob(os.path.join(SRC_DIR, "lesson-[0-9][0-9]-*.md"))):
        n = int(os.path.basename(path)[7:9])
        # lesson 0 is this overview; lesson 20 (the retrospective) assumes every lesson 1-19 and M01-M11
        # ("第 1–19 课""M01–M11" ranges), which this one-row-per-lesson chart does not draw
        if n == 0 or n >= 20:
            continue
        text = open(path, encoding="utf-8").read()
        meta = re.search(r"^meta:(.*)$", text, re.M).group(1)
        pre = meta.split("前置", 1)[1] if "前置" in meta else ""
        lessons = set()
        for m in re.finditer(r"第\s*([\d、，,\s]+?)\s*课", pre):
            lessons |= {int(x) for x in re.split(r"[、，,\s]+", m.group(1)) if x.strip()}
        math_meta = {int(x) for x in re.findall(r"M(\d\d)", pre)}
        math_link = {int(x) for x in re.findall(r"\.\./math/m(\d\d)-", text)}
        rows.append((n, sorted(lessons), sorted(math_meta), sorted(math_link - math_meta)))
    return rows


W = 760
CELL = 21
LX = 70                 # first lesson column
MX_GAP = 26


def main():
    rows = parse()
    n_rows = len(rows)
    top = 64
    H = top + n_rows * CELL + 70
    s = SVG(W, H, title="每一课的前置课与数学课")
    mx0 = LX + 19 * CELL + MX_GAP
    s.text(LX, 18, "前置的 DeepSeek 课（第 0–18 课）", size=12.5, weight=700)
    s.text(mx0, 18, "数学补课 M01–M11", size=12.5, weight=700)
    for k in range(19):
        s.text(LX + k * CELL + CELL / 2, top - 10, str(k), size=11.5, anchor="middle", color="muted")
    for k in range(1, 12):
        s.text(mx0 + (k - 1) * CELL + CELL / 2, top - 10, "%d" % k, size=11.5, anchor="middle", color="muted")
    for i, (n, les, mm, ml) in enumerate(rows):
        y = top + i * CELL
        if i % 2 == 0:
            s.rect(LX - 56, y, mx0 + 11 * CELL - (LX - 56), CELL, fill="grid", fill_opacity=0.6)
        s.text(LX - 8, y + CELL / 2 + 4.5, "第 %d 课" % n, size=12, anchor="end")
        # the diagonal: a lesson can only depend on earlier lessons
        if n <= 18:
            s.rect(LX + n * CELL + 2, y + 2, CELL - 4, CELL - 4, fill="rule", fill_opacity=0.5, rx=2)
        for k in les:
            s.circle(LX + k * CELL + CELL / 2, y + CELL / 2, 6, fill="blue")
        for k in mm:
            s.rect(mx0 + (k - 1) * CELL + 4, y + 4, CELL - 8, CELL - 8, fill="green", rx=2)
        for k in ml:
            s.rect(mx0 + (k - 1) * CELL + 4.5, y + 4.5, CELL - 9, CELL - 9, fill="surface", stroke="green",
                   width=1.5, rx=2)
    s.line(mx0 - MX_GAP / 2, top - 4, mx0 - MX_GAP / 2, top + n_rows * CELL, color="rule", width=1)
    ly = top + n_rows * CELL + 28
    s.circle(LX, ly - 4, 6, fill="blue")
    s.text(LX + 12, ly, "封面 meta 写明的前置课", size=12)
    s.rect(LX + 190, ly - 11, 13, 13, fill="green", rx=2)
    s.text(LX + 210, ly, "meta 写明的数学课", size=12)
    s.rect(LX + 350, ly - 10.5, 12, 12, fill="surface", stroke="green", width=1.5, rx=2)
    s.text(LX + 370, ly, "只在正文里链接到的数学课", size=12)
    s.rect(LX + 550, ly - 11, 13, 13, fill="rule", fill_opacity=0.5, rx=2)
    s.text(LX + 570, ly, "这一课自己", size=12)
    s.text(LX, ly + 24, "由 l00_prereq.py 读取各课源文件的 meta 行与正文链接生成", size=11.5, color="muted")
    s.save("l00-prereq.svg")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
