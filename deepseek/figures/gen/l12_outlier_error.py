"""Figure 12.4: what one outlier does to FP8 (E4M3) quantization error, per-tile vs per-tensor.

Recipe (lesson-12 research notes, section 4.7): numpy default_rng(0), x = N(0,1,128) * 0.01,
then x[17] = 10 (the outlier). Scale s = amax / 448, quantize x / s to E4M3 (round to nearest),
dequantize, and measure |x_hat - x| / |x| for the ordinary (non-outlier) elements.
    A  the original tile (before x[17] = 10), scaled by its own amax        -> median 1.65 %
    B  the tile with the outlier, scaled by its own amax (10)               -> median 2.26 %
    C  the same tile, but scaled by a whole-tensor amax of 1000             -> median 16.4 %,
       21 of the 127 ordinary values become 0
Every number printed on the figure is computed below.
Writes deepseek/figures/l12-outlier-error.svg
"""

import math

import numpy as np

from l12_num import quantize_e4m3
from l12_svg import SVG

rng = np.random.default_rng(0)
x0 = rng.normal(0, 1, 128) * 0.01
x = x0.copy()
x[17] = 10.0
ordinary = np.arange(128) != 17


def case(xin, amax, mask):
    s = amax / 448.0
    q = quantize_e4m3(xin, s)
    rel = np.abs(q - xin) / np.abs(xin)
    return s, rel[mask], int(np.sum(q[mask] == 0))


cases = [
    ("A", "原始 tile，按本 tile 缩放", "（没有离群值）", *case(x0, np.abs(x0).max(), np.ones(128, bool))),
    ("B", "tile 里混进一个 10，", "仍按本 tile 缩放", *case(x, 10.0, ordinary)),
    ("C", "同一个 tile，按整张量缩放", "（别处有个 1000）", *case(x, 1000.0, ordinary)),
]
med = [float(np.median(c[4])) for c in cases]
assert abs(med[0] - 0.0165) < 5e-4 and abs(med[1] - 0.0226) < 5e-4 and abs(med[2] - 0.164) < 5e-4
assert cases[2][5] == 21 and cases[0][5] == 0 and cases[1][5] == 0
# caption claims: B leaves only 1 of the 127 ordinary values below the normal range (2^-6 after
# scaling); C pushes all 127 into the subnormal range; B's scale is ~430x A's.
n_sub = [int(np.sum(np.abs(xin[mask]) / (amax / 448.0) < 2.0 ** -6))
         for xin, amax, mask in ((x0, np.abs(x0).max(), np.ones(128, bool)), (x, 10.0, ordinary),
                                 (x, 1000.0, ordinary))]
assert n_sub == [0, 1, 127], n_sub
assert round(cases[1][3] / cases[0][3]) == 430
# caption claim: the 1.65 % -> 2.26 % step from A to B is NOT caused by the outlier. Inside the
# normal range every binade has the same relative precision; what changes is where each value
# lands inside its binade (430 is not a power of two). Multiplying each row's s by 2^u, u in [0, 1),
# moves both medians around the same band, about 1.6 % .. 2.8 %, and A's 1.65 % is at its low end.
sweep = {}
for tag, xin, amax, mask in (("A", x0, np.abs(x0).max(), np.ones(128, bool)), ("B", x, 10.0, ordinary)):
    ms = [float(np.median(np.abs(quantize_e4m3(xin, amax / 448.0 * 2.0 ** u) - xin)[mask] / np.abs(xin[mask])))
          for u in np.arange(100) / 100]
    sweep[tag] = (min(ms), max(ms))
    print(tag, "median over s*2^u, u in [0,1): %.4f .. %.4f" % sweep[tag])
    assert 0.016 <= min(ms) and max(ms) <= 0.028
assert abs(sweep["A"][0] - med[0]) < 5e-4
for c, m in zip(cases, med):
    print(c[0], "s=%.4g" % c[3], "median %.4f" % m, "zeros", c[5], "max %.3f" % c[4].max())

W, H = 760, 350
X0, X1 = 250, 720
L0, L1 = -4, 0          # 0.01 % .. 100 %


def px(r):
    r = max(r, 10 ** L0)
    return X0 + (math.log10(r) - L0) / (L1 - L0) * (X1 - X0)


svg = SVG(W, H, "一个离群值让同组的其它数被量化得多粗：按 tile 缩放 vs 按整张量缩放")
AY = 268
for k, lab in zip(range(L0, L1 + 1), ("0.01%", "0.1%", "1%", "10%", "100%")):
    x_ = px(10.0 ** k)
    svg.line(x_, 30, x_, AY, color="grid", width=1)
    svg.text(x_, AY + 18, lab, size=11, anchor="middle", color="muted")
svg.line(X0, AY, X1, AY, color="muted", width=1.2)
svg.text((X0 + X1) / 2, AY + 40, "每个普通元素的相对误差 |x̂ − x| / |x|（对数刻度）", size=12, anchor="middle",
         color="muted")

rows_y = [70, 142, 214]
jit = np.random.default_rng(1).uniform(-13, 13, 128)
for (tag, l1, l2, s, rel, zeros), y, m in zip(cases, rows_y, med):
    svg.text(16, y - 6, "%s  %s" % (tag, l1), size=13, weight="600")
    svg.text(34, y + 12, l2, size=12, color="muted")
    svg.text(34, y + 30, "s = %s" % ("%.3g" % s if s > 1e-3 else "%.2e" % s).replace("e-05", "×10^{-5}"),
             size=12, color="orange")
    for i, r in enumerate(rel):
        zero = r >= 0.999
        svg.circle(px(r), y + jit[i] * 0.9, 2.3, fill="red" if zero else "orange",
                   fill_opacity=0.9 if zero else 0.45)
    svg.line(px(m), y - 20, px(m), y + 20, color="ink", width=2.2)
    lab = "中位 %.2f%%" % (100 * m) if m < 0.1 else "中位 %.1f%%" % (100 * m)
    svg.text(px(m) + 6, y - 22, lab, size=12, weight="600")
    if zeros:
        svg.text(px(1.0) - 14, y + 38, "%d / 127 个变成 0（误差 100%%）" % zeros, size=12, anchor="end", color="red")

svg.text(16, H - 8, "数据：rng(0) 生成的 128 个 N(0,1)×0.01，第 17 个换成 10；s = amax / 448，E4M3 最近舍入。每点一个元素；误差 ≤0.01% 的画在最左。",
         size=11, color="muted")
svg.save("l12-outlier-error.svg")
