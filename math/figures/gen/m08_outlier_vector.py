"""Figure M8.9: one outlier, two ways of choosing the FP4 scale.

A 64-element vector: 64 draws from N(0,1) (numpy default_rng(7)) with element 21 replaced by
an outlier 20.  Row (a) quantises it to FP4 E2M1 with one scale for the whole vector
(s = 20/6), row (b) with one scale per block of 16 (s_b = amax_b / 6), both via
m08_block_scaling.qdq.  Grey bars: original values; orange dots: dequantised values;
red crosses: nonzero values that became 0.  The outlier bar is clipped at the top of the
axis in the generator and labelled.
Writes math/figures/m08-outlier-vector.svg
"""

import numpy as np

from m08_block_scaling import qdq
from m08_svg import SVG

N, POS, OUT = 64, 21, 20.0
x = np.random.default_rng(7).standard_normal(N)
x[POS] = OUT
xt, st = qdq(x, "E2M1", None)
xb, sb = qdq(x, "E2M1", 16)
zt = int(np.sum((xt == 0) & (x != 0)))
zb = int(np.sum((xb == 0) & (x != 0)))
print("per-tensor zeros", zt, "per-block zeros", zb, "scales", st, sb)
err_t = np.sqrt(np.mean(np.delete(xt - x, POS) ** 2))
err_b = np.sqrt(np.mean(np.delete(xb - x, POS) ** 2))
print("rms error excl. outlier: tensor %.3f block %.3f" % (err_t, err_b))

W, H = 760, 500
X0, X1 = 70, 730
YR = 3.6                                   # visible y range +-YR
ROWS = [(60, "(a) 整个向量一个缩放因子", xt, st, zt), (296, "(b) 每 16 个数一个缩放因子", xb, sb, zb)]
RH = 150                                   # row height (plot)
svg = SVG(W, H, "离群值与缩放粒度：FP4 E2M1 量化一个 64 维向量")
dx = (X1 - X0) / N

for top, title, xh, scales, nz in ROWS:
    mid = top + RH / 2
    fy = lambda v: mid - max(-YR, min(YR, v)) / YR * (RH / 2)
    svg.text(16, top - 18, title, size=13, weight="600")
    for v in (-3, -2, -1, 0, 1, 2, 3):
        svg.line(X0, fy(v), X1, fy(v), color="grid" if v else "rule", width=1)
        svg.text(X0 - 8, fy(v) + 4, "%d" % v, size=11, anchor="end", color="muted")
    # block boundaries and scales
    if len(scales) > 1:
        for k in range(1, len(scales)):
            xb_ = X0 + k * 16 * dx
            svg.line(xb_, top - 4, xb_, top + RH + 4, color="violet", width=1.2, dash="4 3")
        for k, s in enumerate(scales):
            svg.text(X0 + (k + 0.5) * 16 * dx, top + RH + 20, "s = %.3g" % s, size=12, anchor="middle",
                     color="violet")
    else:
        svg.text((X0 + X1) / 2, top + RH + 20, "s = 20/6 = %.3g：|x| ≤ 0.25·s = %.2f 的数都变成 0"
                 % (scales[0], 0.25 * scales[0]), size=12, anchor="middle", color="violet")
        # the zero band
        svg.rect(X0, fy(0.25 * scales[0]), X1 - X0, fy(-0.25 * scales[0]) - fy(0.25 * scales[0]),
                 fill="red", fill_opacity=0.07)
    for i in range(N):
        cx = X0 + (i + 0.5) * dx
        v = x[i]
        svg.line(cx, fy(0), cx, fy(v), color="muted", width=5, opacity=0.45)
        if v > YR:
            # two short lines, so the label ends before the block boundary at element 32
            svg.text(cx + 7, fy(YR) + 11, "离群值 %g" % v, size=12, color="ink", halo=True)
            svg.text(cx + 7, fy(YR) + 26, "（柱子截断）", size=12, color="muted", halo=True)
            continue
        if xh[i] == 0 and v != 0:
            svg.text(cx, fy(0) + 4.5, "×", size=13, anchor="middle", color="red", weight="700")
        else:
            svg.circle(cx, fy(xh[i]), 3.2, fill="orange")
    svg.text(X1, top - 18, "%d 个非零数变成 0（红叉）" % nz, size=12, anchor="end", color="red")

# legend
ly = H - 14
svg.rect(X0, ly - 10, 10, 12, fill="muted", fill_opacity=0.45)
svg.text(X0 + 16, ly, "原始值", size=12)
svg.circle(X0 + 90, ly - 4, 3.2, fill="orange")
svg.text(X0 + 98, ly, "量化后再乘回缩放因子的值", size=12)
svg.text(X0 + 290, ly, "×", size=13, color="red", weight="700")
svg.text(X0 + 302, ly, "被冲成 0", size=12)
svg.text(X0 + 380, ly, "紫色虚线：块的边界", size=12, color="violet")
svg.save("m08-outlier-vector.svg")
