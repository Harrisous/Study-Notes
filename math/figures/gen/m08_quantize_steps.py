"""Figure M8.8 (new in the teaching review): the §6 hand-worked example, step by step.

The 8-element vector x = (0.9, -1.7, 0.35, 3.0, -0.6, 1.2, 12.0, -0.2) is quantised to
FP4 E2M1 (a) with one scale for all 8 numbers, s = 12/6 = 2, and (b) with one scale per
block of 4, s1 = 3/6 = 0.5 and s2 = 12/6 = 2.  Each row is one element: the blue circle is
|x_i / s| on a log2 axis, the orange dot is the E2M1 grid point it rounds to (nearest, ties
to even, via m08_fp.quant), a red cross in the "0" slot means it rounded to 0.  The three
background bands are the zones of (M8.12) for E2M1: |y| <= 0.25 rounds to 0, 0.25 < |y| < 1
is the subnormal zone, 1 <= |y| <= 6 is the normal zone (relative error <= u = 25%).
Dividing by s shifts every point by the same distance on the log axis, which is why a
point's fate depends only on R_i = max|x_j| / |x_i|.
Writes math/figures/m08-quantize-steps.svg
"""

import math

import numpy as np

from m08_fp import quant
from m08_svg import SVG

x = np.array([0.9, -1.7, 0.35, 3.0, -0.6, 1.2, 12.0, -0.2])
QMAX = 6.0

# (a) one scale; (b) one scale per block of 4 -- the scales exactly as in the lesson text
s_a = np.full(8, np.max(np.abs(x)) / QMAX)
s_b = np.repeat([np.max(np.abs(x[:4])) / QMAX, np.max(np.abs(x[4:])) / QMAX], 4)
y_a, y_b = x / s_a, x / s_b
q_a, q_b = quant(y_a, "E2M1"), quant(y_b, "E2M1")
xh_a, xh_b = s_a * q_a, s_b * q_b
sse_a = float(np.sum((xh_a - x) ** 2))
sse_b = float(np.sum((xh_b - x) ** 2))
# the numbers quoted in the §6 worked example
assert s_a[0] == 2 and list(s_b[[0, 4]]) == [0.5, 2]
assert list(xh_a) == [1, -2, 0, 3, -1, 1, 12, 0], xh_a
assert list(xh_b) == [1.0, -1.5, 0.25, 3.0, -1, 1, 12, 0], xh_b
assert abs(sse_a - 0.4625) < 1e-9 and abs(sse_b - 0.30) < 1e-9, (sse_a, sse_b)
print("whole: y", y_a, "q", q_a, "sse %.4f" % sse_a)
print("block: y", y_b, "q", q_b, "sse %.4f" % sse_b)

W, H = 760, 590
ZX = 206                       # the "0" slot (0 has no place on a log axis)
LX0, LX1 = 238, 736            # log2 axis from 2^-4 to 2^3
LO, HI = -4.0, 3.0
ROW = 21
GRID = [0.5, 1, 1.5, 2, 3, 4, 6]


def px(v):
    return LX0 + (math.log2(v) - LO) / (HI - LO) * (LX1 - LX0)


def num(v):
    """Short decimal label; the minus sign is the typographic one."""
    s = ("%.3f" % v).rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s.replace("-", "−")


svg = SVG(W, H, "先缩放再舍入：8 个数量化成 FP4 E2M1，整组缩放与每 4 个一块")


def panel(top, title, right, rows, y, q, xh):
    """rows: list of (label_or_None, index_or_None) in drawing order."""
    n = len(rows)
    y0 = top + 60                          # baseline of the first row
    ybot = y0 + (n - 1) * ROW
    band_top, band_h = y0 - 14, ybot - y0 + 24
    svg.text(16, top + 16, title, size=13, weight="600")
    svg.text(W - 16, top + 16, right, size=12, anchor="end", color="red")
    # zones of (M8.12)
    svg.rect(ZX - 12, band_top, px(0.25) - ZX + 12, band_h, fill="red", fill_opacity=0.08)
    svg.rect(px(0.25), band_top, px(1) - px(0.25), band_h, fill="muted", fill_opacity=0.10)
    svg.rect(px(1), band_top, px(6) - px(1), band_h, fill="green", fill_opacity=0.08)
    zy = top + 38
    svg.text((ZX - 12 + px(0.25)) / 2, zy, "舍入成 0", size=12, anchor="middle", color="red")
    svg.text((px(0.25) + px(1)) / 2, zy, "次正规区", size=12, anchor="middle", color="muted")
    svg.text((px(1) + px(6)) / 2, zy, "正规区：相对误差 ≤ u = 25%", size=12, anchor="middle",
             color="green")
    # grid points of E2M1 and the rounding threshold 0.25
    for g in GRID:
        svg.line(px(g), band_top, px(g), band_top + band_h, color="rule", width=1)
    svg.line(ZX, band_top, ZX, band_top + band_h, color="rule", width=1)
    svg.line(px(0.25), band_top, px(0.25), band_top + band_h, color="red", width=1, dash="3 3")
    # axis break between the 0 slot and the log axis
    ty = band_top + band_h + 16
    for g in GRID:
        svg.text(px(g), ty, "%g" % g, size=11, anchor="middle", color="orange")
    svg.text(ZX, ty, "0", size=11, anchor="middle", color="orange")
    svg.text((ZX + LX0) / 2, ty, "//", size=11, anchor="middle", color="muted")   # axis break
    svg.text(px(0.25), ty, "0.25", size=11, anchor="middle", color="red")
    svg.text(px(1 / 8), ty, "0.125", size=11, anchor="middle", color="muted")
    # column heads
    hy = top + 38
    svg.text(58, hy, "原值 x_{i}", size=12, anchor="end", color="muted")
    svg.text(118, hy, "x_{i}/s", size=12, anchor="end", color="muted")
    svg.text(178, hy, "乘回 s·q_{i}", size=12, anchor="end", color="muted")
    r = 0
    for label, i in rows:
        yy = y0 + r * ROW
        r += 1
        if i is None:
            svg.text(16, yy + 4, label, size=12, color="violet")
            svg.line(ZX - 12, yy - 10, LX1, yy - 10, color="violet", width=1, dash="4 3")
            continue
        lost = q[i] == 0 and x[i] != 0
        svg.text(58, yy + 4, num(x[i]), size=12, anchor="end")
        svg.text(118, yy + 4, num(y[i]), size=12, anchor="end", color="blue")
        svg.text(178, yy + 4, num(xh[i]), size=12, anchor="end", color="red" if lost else "orange",
                 weight="700" if lost else None)
        bx = px(abs(y[i]))
        if lost:
            svg.arrow(bx - 7, yy, ZX + 8, yy, color="red", width=1.4)
            svg.text(ZX, yy + 4.5, "×", size=14, anchor="middle", color="red", weight="700")
        else:
            ox = px(abs(q[i]))
            if abs(ox - bx) > 9:
                svg.arrow(bx + (6 if ox > bx else -6), yy, ox, yy, color="orange", width=1.4, shorten=4)
            svg.circle(ox, yy, 4, fill="orange")
        svg.circle(bx, yy, 6, stroke="blue", width=1.8)
    return ty


row_a = [(None, i) for i in range(8)]
bottom_a = panel(0, "(a) 整组一个缩放因子：s = 12/6 = 2",
                 "0.35 与 −0.2 丢失；平方误差和 %.4g" % sse_a, row_a, y_a, q_a, xh_a)
row_b = ([("块 1：s_{1} = 3/6 = 0.5", None)] + [(None, i) for i in range(4)]
         + [("块 2：s_{2} = 12/6 = 2", None)] + [(None, i) for i in range(4, 8)])
bottom_b = panel(bottom_a + 22, "(b) 每 4 个一块，各配一个缩放因子",
                 "只有 −0.2 丢失；平方误差和 %.2f" % sse_b, row_b, y_b, q_b, xh_b)

# legend
ly = bottom_b + 30
svg.circle(24, ly - 4, 6, stroke="blue", width=1.8)
svg.text(36, ly, "缩放后的 |x_{i}/s|", size=12)
svg.circle(170, ly - 4, 4, fill="orange")
svg.text(180, ly, "舍入到的 E2M1 格点 q_{i}", size=12)
svg.text(356, ly, "×", size=14, color="red", weight="700", anchor="middle")
svg.text(366, ly, "舍入成 0", size=12)
svg.text(440, ly, "横轴：|x_{i}/s|，对数刻度（符号见左边的数）", size=12, color="muted")
assert ly + 8 <= H, ly
svg.save("m08-quantize-steps.svg")
