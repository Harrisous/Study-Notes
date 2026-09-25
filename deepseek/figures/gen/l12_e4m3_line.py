"""Figure 12.3: every positive E4M3 and E5M2 value on a log2 axis, plus a linear zoom.

The value grids come from l12_num.py (FP8 Formats for Deep Learning, Table 1). Top: the
126 positive E4M3 values span 2^-9 .. 448; the 7 values above 240 exist only because E4M3
gives up infinities. Middle: the 123 positive E5M2 values span 2^-16 .. 57344. Bottom: E4M3
on a linear axis between 1 and 8, where the spacing doubles at every power of two.
Writes deepseek/figures/l12-e4m3-line.svg
"""

import math

from l12_num import E4M3, E5M2
from l12_svg import SVG

pos4 = [v for v in E4M3 if v > 0]
pos5 = [v for v in E5M2 if v > 0]
assert len(pos4) == 126 and len(pos5) == 123
borrowed = [v for v in pos4 if v > 240]
assert borrowed == [256, 288, 320, 352, 384, 416, 448]

W, H = 760, 372
X0, X1 = 64, 736
L0, L1 = -16, 16


def px(v):
    return X0 + (math.log2(v) - L0) / (L1 - L0) * (X1 - X0)


svg = SVG(W, H, "E4M3 与 E5M2 的全部正值（对数轴）以及 E4M3 在 1 到 8 之间的线性放大")

# log axis
AY = 150
for k in range(L0, L1 + 1, 2):
    x = px(2.0 ** k)
    svg.line(x, 36, x, AY, color="grid", width=1)
    svg.text(x, AY + 16, "2^{%d}" % k, size=11, anchor="middle", color="muted")
svg.line(X0, AY, X1, AY, color="muted", width=1.2)
svg.text(X1, AY + 34, "数值（对数刻度，每条竖线相隔 4 倍）", size=11, anchor="end", color="muted")

# E4M3 row
R1 = 62
svg.text(X0 - 8, R1 + 5, "E4M3", size=13, anchor="end", weight="700", color="orange")
for v in pos4:
    x = px(v)
    sub = v < 2 ** -6
    col = "red" if v > 240 else "orange"
    svg.line(x, R1 - 9, x, R1 + 9, color=col, width=1.1, opacity=0.55 if sub else 1)
for v, lab, dy, anc in ((2 ** -9, "最小次正规 2^{-9}", -16, "middle"), (2 ** -6, "最小正规 2^{-6}", 26, "middle")):
    svg.text(px(v), R1 + dy, lab, size=11, anchor=anc, color="muted")
svg.text(px(448) + 4, R1 - 14, "448", size=12, anchor="start", color="red", weight="600")
svg.text(px(240) - 4, R1 + 26, "240（守 IEEE 惯例时的最大值）", size=11, anchor="end", color="muted")
svg.line(px(240), R1 + 11, px(240), R1 + 16, color="muted", width=1)

# E5M2 row
R2 = 112
svg.text(X0 - 8, R2 + 5, "E5M2", size=13, anchor="end", weight="700", color="orange")
for v in pos5:
    x = px(v)
    sub = v < 2 ** -14
    svg.line(x, R2 - 9, x, R2 + 9, color="orange", width=1.1, opacity=0.55 if sub else 1)
svg.text(px(57344) - 4, R2 - 14, "57,344", size=12, anchor="end", color="orange", weight="600")
svg.text(px(2 ** -16) + 4, R2 - 14, "2^{-16}", size=11, anchor="start", color="muted")

# linear zoom: E4M3 between 1 and 8
ZY = 280
Z0, Z1 = 70, 730


def zx(v):
    return Z0 + (v - 1) / 7 * (Z1 - Z0)


svg.text(Z0, 210, "E4M3 在 [1, 8) 上的线性放大：每个 2 的幂之间 8 个值，过一个 2 的幂间距就翻倍",
         size=12, weight="600")
svg.rect(px(1), 40, px(8) - px(1), AY - 40, fill="amber", fill_opacity=0.10)
svg.text((px(1) + px(8)) / 2, 34, "下方放大 [1, 8)", size=11, anchor="middle", color="amber")
svg.line(Z0, ZY, Z1, ZY, color="muted", width=1.2)
spans = [(1, 2, 0.125), (2, 4, 0.25), (4, 8, 0.5)]
for a, b, step in spans:
    n = int(round((b - a) / step))
    for i in range(n):
        v = a + i * step
        assert any(abs(v - g) < 1e-12 for g in pos4)
        big = i == 0
        svg.line(zx(v), ZY - (14 if big else 9), zx(v), ZY + (14 if big else 9), color="orange",
                 width=1.8 if big else 1.2)
    svg.text(zx(a), ZY + 30, "%g" % a, size=12, anchor="middle")
    mid = (zx(a) + zx(b)) / 2
    svg.text(mid, ZY - 24, "间距 %g" % step, size=12, anchor="middle", color="muted")
    assert abs(step / b - 0.0625) < 1e-12 and abs(step / a - 0.125) < 1e-12
svg.text(zx(8), ZY + 30, "8", size=12, anchor="middle")
svg.text(Z0, ZY + 58, "三段的相对间距（间距 ÷ 数值）都在 6.25% 到 12.5% 之间；舍入到最近格点，相对误差至多是它的一半。",
         size=12, color="ink")
svg.text(Z0, ZY + 78, "红色：240 以上的 7 个值（256…448）；半透明：次正规数，越靠近 0 相对间距越大。",
         size=11, color="muted")
svg.save("l12-e4m3-line.svg")
