"""Figure M8.1: the same 32 codes (5 bits, no sign) spent two ways.

Fixed point with 3 integer bits and 2 fraction bits: k/4, k = 0..31 (0 .. 7.75, uniform).
A float with 3 exponent bits and 2 mantissa bits (bias 3, all codes finite, the FP6 E3M2
element format of MX without its sign bit): 0 .. 28, four values per power of two.
Panel (a) plots both on a linear axis 0..8, panel (b) on a log2 axis 1/16..32.
Every tick position is computed here.  Writes math/figures/m08-fixed-vs-float.svg
"""

import math

import numpy as np

from m08_fp import values
from m08_svg import SVG

fixed = np.arange(32) / 4.0
flt = values("E3M2")
assert len(flt) == 32 and flt[-1] == 28.0

W, H = 720, 452
X0, X1 = 150, 690
svg = SVG(W, H, "同样 32 个码：定点把刻度均匀铺开，浮点让刻度随数值变稀")

ROWS_A = (74, 148)      # y of the two rulers in panel (a)
ROWS_B = (284, 338)     # panel (b)


def lin(v):
    return X0 + v / 8.0 * (X1 - X0)


LG0, LG1 = -4, 5        # log2 range 1/16 .. 32


def lg(v):
    return X0 + (math.log2(v) - LG0) / (LG1 - LG0) * (X1 - X0)


def ruler(y, label, sub, color):
    svg.line(X0, y, X1, y, color="rule", width=1.2)
    svg.text(X0 - 12, y - 3, label, size=13, anchor="end", color=color, weight="600")
    svg.text(X0 - 12, y + 14, sub, size=11, anchor="end", color="muted")


# ---------- panel (a): linear
svg.text(16, 30, "(a) 线性刻度：0 到 8", size=13, weight="600")
for t in range(0, 9):
    svg.line(lin(t), ROWS_A[0] - 16, lin(t), ROWS_A[1] + 16, color="grid", width=1)
    svg.text(lin(t), ROWS_A[1] + 34, str(t), size=11, anchor="middle", color="muted")
ruler(ROWS_A[0], "定点 3.2", "3 位整数 + 2 位小数", "blue")
ruler(ROWS_A[1], "浮点 E3M2", "3 位指数 + 2 位尾数", "orange")
for v in fixed:
    svg.line(lin(v), ROWS_A[0] - 9, lin(v), ROWS_A[0] + 9, color="blue", width=1.6)
shown = 0
for v in flt:
    if v <= 8:
        svg.line(lin(v), ROWS_A[1] - 9, lin(v), ROWS_A[1] + 9, color="orange", width=1.6)
        shown += 1
svg.text(X1, ROWS_A[0] + 27, "0 到 7.75，处处步长 0.25", size=11, anchor="end", color="blue")
svg.text(X1, ROWS_A[1] - 18, "另有 %d 个值在 8 以外：10、12、14、16、20、24、28" % (32 - shown),
         size=11, anchor="end", color="orange")

# ---------- panel (b): log2
svg.text(16, 236, "(b) 对数刻度：1/16 到 32（每一格是一个 2 倍）", size=13, weight="600")
for e in range(LG0, LG1 + 1):
    x = lg(2.0 ** e)
    svg.line(x, ROWS_B[0] - 16, x, ROWS_B[1] + 30, color="grid", width=1)
    lab = ("1/%d" % 2 ** (-e)) if e < 0 else str(2 ** e)
    svg.text(x, ROWS_B[1] + 44, lab, size=11, anchor="middle", color="muted")
ruler(ROWS_B[0], "定点 3.2", "最小正数 0.25", "blue")
ruler(ROWS_B[1], "浮点 E3M2", "最小正数 0.0625", "orange")
for v in fixed[1:]:
    svg.line(lg(v), ROWS_B[0] - 9, lg(v), ROWS_B[0] + 9, color="blue", width=1.6)
for v in flt[1:]:
    svg.line(lg(v), ROWS_B[1] - 9, lg(v), ROWS_B[1] + 9, color="orange", width=1.6)

# relative step annotations (computed)
rel_fixed_lo = (fixed[2] - fixed[1]) / fixed[1]          # 0.25 -> 0.5 : 100 %
rel_fixed_hi = (fixed[31] - fixed[30]) / fixed[30]       # 7.5 -> 7.75 : 3.3 %
svg.text(lg(0.25), ROWS_B[0] - 20, "0.25→0.5：相对步长 %d%%" % round(rel_fixed_lo * 100),
         size=11, anchor="start", color="blue")
svg.text(lg(7.75), ROWS_B[0] - 20, "7.5→7.75：%.1f%%" % (rel_fixed_hi * 100),
         size=11, anchor="end", color="blue")
assert flt[4] == 0.25                                          # smallest normal value (e = 1, m = 0)
rel_f = [(flt[i + 1] - flt[i]) / flt[i] for i in range(4, 31)]  # normal range only
svg.text((X0 + X1) / 2, ROWS_B[1] + 66,
         "浮点在正规数范围内（0.25 以上）相邻两值的相对步长始终在 %.1f%% 到 %.0f%% 之间"
         % (min(rel_f) * 100, max(rel_f) * 100), size=12, anchor="middle", color="orange")

# bottom note
svg.text((X0 + X1) / 2, H - 14,
         "两种编法都只有 32 个码：定点换来均匀的绝对精度，浮点换来 9 个 2 倍的覆盖范围和均匀的相对精度",
         size=12, anchor="middle", color="muted")

svg.save("m08-fixed-vs-float.svg")
