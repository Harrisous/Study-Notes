"""Figure M7.8: McCandlish et al.'s gradient-noise-scale model of batch size.

Left:  steps vs examples to reach a fixed loss, eq. (2.11):  S/S_min − 1 = (E/E_min − 1)^{-1}.
       The point B = B_crit = E_min/S_min sits at (2, 2): twice the minimum steps and twice the minimum data.
Right: benefit of one optimally-sized step as a function of batch size, eq. (2.7):
       ΔL_opt(B)/ΔL_max = 1/(1 + B_noise/B)  (the optimal step size ε_opt/ε_max, eq. (2.6), is the same curve).
       B = B_noise/10 -> 1/11 = 0.0909;  B = B_noise -> 0.5;  B = 10 B_noise -> 10/11 = 0.909.
Every point is computed here. Writes math/figures/m07-batch-noise.svg
"""

import numpy as np

from m07_svg import SVG, Axes, clipped_path

W, H = 740, 382
svg = SVG(W, H, "梯度噪声尺度模型：步数与样本数的双曲线权衡，以及每步收益随 batch 的饱和")

ax = Axes(70, 44, 330, 296, (1, 6), (1, 6))
svg.text((ax.x0 + ax.x1) / 2, 22, "达到同一损失：步数 vs 样本数", size=14, anchor="middle", weight="600")
ax.ygrid(svg, [1, 2, 3, 4, 5, 6])
ax.xgrid(svg, [1, 2, 3, 4, 5, 6])
ax.frame(svg)
svg.text((ax.x0 + ax.x1) / 2, ax.y1 + 36, "用掉的样本数 E / E_{min}", size=12, anchor="middle", color="muted")
svg.text(26, (ax.y0 + ax.y1) / 2, "走的步数 S / S_{min}", size=12, anchor="middle", color="muted", rotate=-90)
e = np.linspace(1.0005, 6, 800)
s = 1 + 1 / (e - 1)
svg.path(clipped_path([ax(x, y) for x, y in zip(e, s)], (ax.x0, ax.y0, ax.x1, ax.y1)), stroke="blue", width=2.6)
svg.line(ax.X(1), ax.Y(1), ax.X(6), ax.Y(1), color="muted", width=1, dash="4 4")
svg.circle(ax.X(2), ax.Y(2), 4.5, fill="ink")
svg.text(ax.X(2) + 8, ax.Y(2) - 8, "B = B_{crit}：(2, 2)", size=12, weight="600")
svg.text(ax.X(1.5), ax.Y(5.4), "小 batch：样本省，步数多", size=11, color="blue")
svg.text(ax.X(5.9), ax.Y(1.55), "大 batch：步数省，样本多", size=11, color="blue", anchor="end")

ax2 = Axes(430, 44, 712, 296, (1e-2, 1e2), (0, 1.0), xlog=True)
svg.text((ax2.x0 + ax2.x1) / 2, 22, "一步最多能降多少：ΔL_{opt} / ΔL_{max}", size=14, anchor="middle", weight="600")
ax2.ygrid(svg, [0, 0.25, 0.5, 0.75, 1.0])
ax2.xgrid(svg, [1e-2, 1e-1, 1, 10, 100], labels=["0.01", "0.1", "1", "10", "100"])
ax2.frame(svg)
svg.text((ax2.x0 + ax2.x1) / 2, ax2.y1 + 36, "batch 大小 B / B_{noise}（对数刻度）", size=12, anchor="middle", color="muted")
r = np.logspace(-2, 2, 400)
y = 1 / (1 + 1 / r)
svg.path(clipped_path([ax2(a, b) for a, b in zip(r, y)], (ax2.x0, ax2.y0, ax2.x1, ax2.y1)), stroke="green", width=2.6)
yl = r  # small-batch asymptote y ≈ B/B_noise
svg.path(clipped_path([ax2(a, b) for a, b in zip(r, yl)], (ax2.x0, ax2.y0, ax2.x1, ax2.y1)), stroke="green",
         width=1.1, dash="4 4")
for rv, lab, dx, anc in ((0.1, "1/11 = 0.091", 8, "start"), (1, "0.5", -8, "end"), (10, "10/11 = 0.909", 8, "start")):
    yv = 1 / (1 + 1 / rv)
    svg.circle(ax2.X(rv), ax2.Y(yv), 4, fill="green")
    svg.text(ax2.X(rv) + dx, ax2.Y(yv) + (-8 if rv == 1 else 16), lab, size=12, color="green", anchor=anc,
             weight="600")
svg.text(ax2.X(0.011), ax2.Y(0.9), "虚线：≈ B/B_{noise}", size=11, color="green")
svg.text(ax2.X(0.011), ax2.Y(0.9) + 21, "（小 batch：收益 ∝ B）", size=11, color="green")
svg.text(ax2.X(95), ax2.Y(0.6), "饱和：再加 batch", size=11, color="muted", anchor="end")
svg.text(ax2.X(95), ax2.Y(0.6) + 14, "几乎不再多降", size=11, color="muted", anchor="end")
svg.text(70, H - 10, "左：式 (2.11)，横纵轴都是相对最小值的倍数；右：式 (2.7)，最优步长 η_{opt}/η_{∞} 是同一条曲线。",
         size=11, color="muted")
svg.save("m07-batch-noise.svg")
