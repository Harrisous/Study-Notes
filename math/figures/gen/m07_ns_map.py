"""Figure M7.11: Newton–Schulz acts on each singular value separately; how the three polynomials behave.

φ_J(x) = 3.4445x − 4.7750x³ + 2.0315x⁵ (Jordan), φ_B(x) = 2x − 1.5x³ + 0.5x⁵ (baseline quintic),
f(x) = 1.5x − 0.5x³ (classical cubic).
Left: φ_J^N(x) on [0, 1] for N = 1, 2, 3, 5, and DeepSeek-V4's hybrid (8 steps φ_J, then 2 steps φ_B).
      φ_J(1) = 0.701, max φ_J = 1.2024 at x = 0.5545; after 5 steps inputs ≥ 1e-3 land in [0.4705, 1.2024],
      the long-run band is [0.6818, 1.1344] (after N = 5 every x ≥ 0.05 is already inside it);
      for inputs x ≥ 1e-3 the hybrid output lies in [1.000000, 1.000538].
Right: one singular value starting at 0.01, iterate k = 0..12 under each map.
Every point is computed here. Writes math/figures/m07-ns-map.svg
"""

import numpy as np

from m07_svg import SVG, Axes, clipped_path

a, b, c = 3.4445, -4.7750, 2.0315


def phiJ(x):
    return a * x + b * x ** 3 + c * x ** 5


def phiB(x):
    return 2 * x - 1.5 * x ** 3 + 0.5 * x ** 5


def cub(x):
    return 1.5 * x - 0.5 * x ** 3


W, H = 740, 414
svg = SVG(W, H, "牛顿–舒尔茨迭代对奇异值的作用：φ^N(x) 的形状与从 0.01 出发的收敛")
ax = Axes(64, 44, 380, 310, (0, 1), (0, 1.4))
svg.text((ax.x0 + ax.x1) / 2, 22, "输出奇异值 φ^{N}(x)", size=14, anchor="middle", weight="600")
# bands
svg.rect(ax.x0, ax.Y(1.3), ax.x1 - ax.x0, ax.Y(0.7) - ax.Y(1.3), fill="green", fill_opacity=0.07)
svg.rect(ax.x0, ax.Y(1.1344), ax.x1 - ax.x0, ax.Y(0.6818) - ax.Y(1.1344), fill="orange", fill_opacity=0.08)
ax.ygrid(svg, [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax.xgrid(svg, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.frame(svg)
svg.text((ax.x0 + ax.x1) / 2, ax.y1 + 36, "输入奇异值 x（已除以 ‖G‖_{F}，落在 [0, 1]）", size=12, anchor="middle", color="muted")
box = (ax.x0, ax.y0, ax.x1, ax.y1)
xs = np.linspace(0, 1, 801)
y = xs.copy()
shades = {1: 0.35, 2: 0.55, 3: 0.75, 5: 1.0}
for n in range(1, 6):
    y = phiJ(y)
    if n in shades:
        svg.path(clipped_path([ax(u, v) for u, v in zip(xs, y)], box), stroke="orange", width=1.9,
                 opacity=shades[n])
yh = xs.copy()
for k in range(10):
    yh = phiJ(yh) if k < 8 else phiB(yh)
svg.path(clipped_path([ax(u, v) for u, v in zip(xs, yh)], box), stroke="green", width=2.6)
svg.line(ax.x0, ax.Y(1), ax.x1, ax.Y(1), color="ink", width=0.8, dash="2 3")
# annotations
svg.circle(ax.X(1), ax.Y(0.701), 3.5, fill="orange")
svg.text(ax.X(1) - 6, ax.Y(0.701) + 16, "φ_{J}(1) = 0.701", size=11, anchor="end", color="orange")
svg.circle(ax.X(0.5545), ax.Y(1.2024), 3.5, fill="orange")
svg.text(ax.X(0.5545) + 6, ax.Y(1.2024) - 6, "峰 1.2024 @ 0.5545", size=11, color="orange")
svg.text(ax.X(0.1), ax.Y(0.2), "N = 1（颜色越深 N 越大：1、2、3、5）", size=11, color="orange")
# the top strip above the green band is empty: the band legend on the left, the thick-line legend on
# the right (the thick green line itself runs along y = 1 through the orange curves, so no label there)
svg.text(ax.X(0.02), ax.Y(1.34), "绿带 [0.7, 1.3]：博客的目标", size=11, color="green")
svg.text(ax.X(0.2), ax.Y(0.52), "橙带 [0.68, 1.13]：φ_{J} 反复迭代的最终范围", size=11, color="orange")
svg.text(ax.X(0.99), ax.Y(1.34), "绿粗线：V4 混合 8+2 步", size=12, color="green", weight="600", anchor="end")

# right panel: track from 0.01
ax2 = Axes(452, 44, 716, 310, (0, 12), (0, 1.25))
svg.text((ax2.x0 + ax2.x1) / 2, 22, "一个很小的奇异值 0.01 的命运", size=14, anchor="middle", weight="600")
ax2.ygrid(svg, [0, 0.25, 0.5, 0.75, 1.0, 1.25])
ax2.xgrid(svg, [0, 2, 4, 6, 8, 10, 12])
ax2.frame(svg)
svg.text((ax2.x0 + ax2.x1) / 2, ax2.y1 + 36, "迭代次数 k", size=12, anchor="middle", color="muted")
maps = [(cub, "muted", "三次 (1.5, −0.5)：φ′(0) = 1.5"), (phiB, "blue", "五次 (2, −1.5, 0.5)：φ′(0) = 2"),
        (phiJ, "orange", "Jordan (3.4445, …)：φ′(0) = 3.4445")]
tracks = {}
for f, col, lab in maps:
    x = 0.01
    tr = [x]
    for k in range(12):
        x = f(x)
        tr.append(x)
    tracks[col] = tr
    svg.path(clipped_path([ax2(k, v) for k, v in enumerate(tr)], (ax2.x0, ax2.y0, ax2.x1, ax2.y1)), stroke=col,
             width=2.2)
    for k, v in enumerate(tr):
        svg.circle(ax2.X(k), ax2.Y(v), 2.4, fill=col)
ly = ax2.y1 + 58
for f, col, lab in maps:
    svg.line(ax2.x0, ly - 4, ax2.x0 + 18, ly - 4, color=col, width=2.4)
    svg.text(ax2.x0 + 24, ly, lab, size=11, color=col)
    ly += 16
svg.save("m07-ns-map.svg")
print("cubic 12:", tracks["muted"][12], "naive:", [round(v, 4) for v in tracks["blue"]])
print("jordan:", [round(v, 4) for v in tracks["orange"]])
xin = np.logspace(-3, 0, 20001)
yy = xin.copy()
for k in range(5):
    yy = phiJ(yy)
print("5-step range", yy.min(), yy.max(), "hybrid range", yh[xs >= 1e-3].min(), yh[xs >= 1e-3].max())
