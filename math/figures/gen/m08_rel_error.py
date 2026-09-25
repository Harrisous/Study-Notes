"""Figure M8.5: relative rounding error of FP8 E4M3 across its whole range.

For 6000 log-spaced x in [2^-11, 2^10] the script rounds x to E4M3 (nearest, ties to even,
saturating at 448, see m08_fp.quant) and plots |fl(x) - x| / x.  Dashed: the bound
u = 2^-(M+1) = 1/16 for normal numbers, the subnormal envelope (half the subnormal spacing
2^-10 divided by x) and the saturation curve (x - 448) / x.  Values above 20 % are clipped
in the generator (not with clipPath) and marked.
Writes math/figures/m08-rel-error.svg
"""

import math

import numpy as np

from m08_fp import params, quant
from m08_svg import SVG

p = params("E4M3")
u = p["u"]
L0, L1 = -11, 10
xs = 2.0 ** np.linspace(L0, L1, 6000)
rel = np.abs(quant(xs, "E4M3") - xs) / xs
YMAX = 0.20

W, H = 760, 400
X0, X1, Y0, Y1 = 78, 736, 48, 318


def px(x):
    return X0 + (math.log2(x) - L0) / (L1 - L0) * (X1 - X0)


def py(r):
    return Y1 - min(r, YMAX) / YMAX * (Y1 - Y0)


svg = SVG(W, H, "E4M3 的相对舍入误差随数值大小的变化")

# regions
half_sub = p["min_sub"] / 2
regions = [(2.0 ** L0, half_sub, "冲成 0（100%）", "red"),
           (half_sub, p["min_normal"], "次正规区", "amber"),
           (p["min_normal"], p["max"], "正规区：误差 ≤ u = 1/16", "green"),
           (p["max"], 2.0 ** L1, "饱和到 448", "red")]
for a, b, lab, col in regions:
    svg.rect(px(a), Y0, px(b) - px(a), Y1 - Y0, fill=col, fill_opacity=0.06)
    svg.text((px(a) + px(b)) / 2, Y0 - 8, lab, size=12, anchor="middle", color=col)

# grid and axes
for r in (0, 0.05, 0.10, 0.15, 0.20):
    svg.line(X0, py(r), X1, py(r), color="grid", width=1)
    svg.text(X0 - 8, py(r) + 4, "%d%%" % round(r * 100), size=11, anchor="end", color="muted")
for e in range(L0 + 1, L1 + 1, 2):
    x = px(2.0 ** e)
    svg.line(x, Y1, x, Y1 + 5, color="muted", width=1)
    svg.text(x, Y1 + 20, "2^{%d}" % e, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 44, "x（对数刻度）", size=12, anchor="middle", color="muted")
svg.text(20, (Y0 + Y1) / 2, "相对误差 |fl(x) − x| / x", size=12, anchor="middle", color="muted", rotate=-90)

# the error curve (clipped numerically at YMAX)
d = "M" + " L".join("%.1f %.1f" % (px(x), py(r)) for x, r in zip(xs, rel))
svg.path(d, stroke="orange", width=1.0)

# bound for normal range
svg.line(px(p["min_normal"]), py(u), px(p["max"]), py(u), color="ink", width=1.4, dash="6 4")
svg.text(px(2.0 ** 3), py(u) - 8, "上界 u = 2^{−4} = 6.25%", size=12, anchor="middle")

# subnormal envelope: (2^-10) / x, only where below YMAX
env_x = 2.0 ** np.linspace(math.log2(half_sub), math.log2(p["min_normal"]), 200)
env = (p["min_sub"] / 2) / env_x
pts = [(px(x), py(r)) for x, r in zip(env_x, env) if r <= YMAX]
svg.path("M" + " L".join("%.1f %.1f" % q for q in pts), stroke="ink", width=1.2, dash="3 3")
ex_, ey_ = 2.0 ** -7.2, (p["min_sub"] / 2) / 2.0 ** -7.2
svg.line(px(ex_) + 3, py(ey_) - 2, px(2.0 ** -5.3) - 4, py(0.155) + 4, color="ink", width=0.8)
svg.text(px(2.0 ** -5.3), py(0.155), "虚线：次正规区的上界 2^{−10}/x", size=12)

# annotation: clipped
svg.text(px(2.0 ** 8.6), py(0.18), "纵轴在 20% 处截断", size=11, anchor="end", color="muted")

# example point: 0.1
x01 = 0.1
r01 = abs(float(quant(x01, "E4M3")) - x01) / x01
svg.circle(px(x01), py(r01), 4, fill="blue")
svg.line(px(x01) + 3, py(r01) - 3, px(2.0 ** -2.4) - 2, py(0.105) + 4, color="blue", width=0.8)
svg.text(px(2.0 ** -2.4), py(0.105), "0.1 → 0.1015625，误差 1.56%", size=12, color="blue")

svg.save("m08-rel-error.svg")
