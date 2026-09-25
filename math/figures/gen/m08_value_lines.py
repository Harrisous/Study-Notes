"""Figure M8.4: where the representable values of each format sit, on a log2 axis.

FP4 E2M1 (8 non-negative values), FP8 E4M3 (127 finite non-negative values incl. 0) and
FP8 E5M2 (124) are drawn tick by tick from m08_fp.values(); FP16 and BF16 have too many
values to draw, so they are shown as bands from the smallest subnormal to the largest value,
with the count per power of two (2^M) written on them.  Zero is not on a log axis.
Writes math/figures/m08-value-lines.svg
"""

import math

from m08_fp import params, values
from m08_svg import SVG

W, H = 760, 388
X0, X1 = 120, 730
L0, L1 = -26, 18              # log2 axis range


def px(v):
    return X0 + (math.log2(v) - L0) / (L1 - L0) * (X1 - X0)


svg = SVG(W, H, "各格式的可表示值在对数轴上的位置")
# grid
for e in range(L0, L1 + 1, 2):
    x = px(2.0 ** e)
    svg.line(x, 46, x, H - 52, color="grid", width=1)
    if e % 4 == 2:
        svg.text(x, H - 34, "2^{%d}" % e, size=11, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, H - 10, "数值大小（对数刻度；相邻两条竖线相差 4 倍）", size=12, anchor="middle",
         color="muted")

rows = [("FP4 E2M1", "E2M1", 70), ("FP8 E4M3", "E4M3", 130), ("FP8 E5M2", "E5M2", 190)]
for name, key, y in rows:
    v = values(key)[1:]
    p = params(key)
    svg.text(X0 - 12, y + 4, name, size=13, anchor="end", weight="600")
    for val in v:
        normal = val >= p["min_normal"]
        svg.line(px(val), y - 11, px(val), y + 11, color="orange", width=1.4 if normal else 1.0,
                 opacity=1 if normal else 0.55)
    lo, hi = v[0], v[-1]
    n_per = 2 ** p["M"]
    svg.text(px(hi) + 8, y + 4, "%g" % hi, size=12, color="orange")
    lab = "%d 个正值，每段 %d 个" % (len(v), n_per)
    svg.text(px(lo) - 8, y + 4, lab, size=12, anchor="end", color="muted")

# E4M3: the 7 values above 240 that IEEE-style encoding would not have
e4 = values("E4M3")
extra = [x for x in e4 if x > 240]
assert len(extra) == 7
xa, xb = px(extra[0]), px(extra[-1])
svg.path("M%.1f %.1f v-6 H%.1f v6" % (xa, 130 - 16, xb), stroke="blue", width=1.2)
svg.text(xa - 6, 130 - 20, "256…448：比 IEEE 式多出的 7 个值", size=12, anchor="end", color="blue")

# FP16 and BF16 as bands
for name, key, y in (("FP16", "FP16", 250), ("BF16", "BF16", 300)):
    p = params(key)
    svg.text(X0 - 12, y + 4, name, size=13, anchor="end", weight="600")
    a = px(p["min_sub"]) if p["min_sub"] > 2.0 ** L0 else X0
    b = min(px(p["max"]), X1)
    svg.rect(a, y - 10, b - a, 20, fill="blue", fill_opacity=0.14, stroke="blue", width=1)
    if p["min_sub"] > 2.0 ** L0:
        # normal part darker
        svg.rect(px(p["min_normal"]), y - 10, b - px(p["min_normal"]), 20, fill="blue", fill_opacity=0.16)
    txt = "每段 %d 个值" % 2 ** p["M"]
    if key == "FP16":
        txt += "；次正规 2^{−24} 起，最大 65,504"
        svg.text((a + b) / 2, y + 4.5, txt, size=12, anchor="middle")
    else:
        txt += "；正规范围与 FP32 几乎相同；次正规 2^{−133} 起，最大 3.39×10^{38}"
        svg.text((a + b) / 2, y + 4.5, txt, size=12, anchor="middle")
        svg.text(X1 - 4, y + 4.5, "→", size=13, anchor="end", color="blue")
        svg.text(X0 + 4, y + 4.5, "←", size=13, color="blue")

svg.text(X0, 30, "每一根橙色竖线是一个可表示的正数；浅色竖线是次正规数。0 不在对数轴上。", size=12, color="muted")
svg.save("m08-value-lines.svg")
