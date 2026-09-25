"""Figure 18.9: the two Newton-Schulz polynomials of V4's hybrid iteration and three singular-value
trajectories (8 steps of phi_A, then 2 steps of phi_B). V4 report eq. (28), sec. 2.4; research
notes sec. 4.5 (phi_A(1) = 0.701, max 1.2024 at 0.554, band [0.682, 1.134]).
Writes deepseek/figures/l18-newton-schulz.svg
"""

import numpy as np

from l18_svg import SVG


def phiA(x):
    return 3.4445 * x - 4.7750 * x ** 3 + 2.0315 * x ** 5


def phiB(x):
    return 2 * x - 1.5 * x ** 3 + 0.5 * x ** 5


xs = np.linspace(0, 1, 100001)
imax = int(np.argmax(phiA(xs)))
assert round(float(phiA(1.0)), 3) == 0.701 and round(float(phiA(xs[imax])), 4) == 1.2024
assert round(float(xs[imax]), 3) == 0.555 or round(float(xs[imax]), 3) == 0.554


def traj(x0):
    out = [x0]
    for k in range(10):
        out.append((phiA if k < 8 else phiB)(out[-1]))
    return out


TR = [(1e-3, "σ₀ = 0.001", "circle"), (0.05, "σ₀ = 0.05", "square"), (1.0, "σ₀ = 1", "tri")]
lo = min(traj(x)[8] for x in np.linspace(1e-3, 1, 2000))
hi = max(traj(x)[8] for x in np.linspace(1e-3, 1, 2000))
assert round(lo, 3) == 0.682 and round(hi, 3) == 1.134
fin = [traj(x)[10] for x in np.linspace(1e-3, 1, 2000)]
assert 0.99999 < min(fin) and max(fin) < 1.00054

W, H = 760, 424
svg = SVG(W, H, "牛顿–舒尔茨的两组系数与奇异值轨迹")

# ---------------------------------------------------------------- left: polynomials
LX0, LX1, LY0, LY1, VMAX = 60, 330, 44, 314, 1.3


def lx(v):
    return LX0 + v / VMAX * (LX1 - LX0)


def ly(v):
    return LY1 - v / VMAX * (LY1 - LY0)


svg.text(20, 24, "(a) 一步迭代对一个奇异值做什么", size=13, weight="700")
for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
    svg.line(LX0, ly(v), LX1, ly(v), color="grid", width=1)
    svg.line(lx(v), LY0, lx(v), LY1, color="grid", width=1)
    svg.text(LX0 - 6, ly(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
    svg.text(lx(v), LY1 + 16, "%.1f" % v, size=11, anchor="middle", color="muted")
svg.line(LX0, LY1, LX1, LY1, color="muted", width=1.2)
svg.line(LX0, LY0, LX0, LY1, color="muted", width=1.2)
svg.text((LX0 + LX1) / 2, LY1 + 34, "输入奇异值 σ", size=12, anchor="middle", color="muted")
svg.line(lx(0), ly(0), lx(1.3), ly(1.3), color="muted", width=1.2, dash="4 4")
svg.text(lx(1.22), ly(1.28), "y = σ", size=11, color="muted")
for f, col in ((phiA, "orange"), (phiB, "green")):
    pts = [(lx(x), ly(f(x))) for x in np.linspace(0, 1.18, 240) if 0 <= f(x) <= VMAX]
    svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke=col, width=2.4)
svg.circle(lx(1), ly(phiA(1)), 4, fill="orange")
svg.text(lx(1) + 8, ly(phiA(1)) + 14, "φ_{A}(1) = 0.701", size=12, color="orange")
svg.circle(lx(xs[imax]), ly(phiA(xs[imax])), 4, fill="orange")
svg.text(lx(xs[imax]) - 4, ly(phiA(xs[imax])) - 10, "max 1.20 @ 0.554", size=12, anchor="middle", color="orange")
svg.circle(lx(1), ly(1), 4, fill="green")
svg.line(lx(0.9), ly(1), lx(1.1), ly(1), color="green", width=1.6)
svg.text(lx(1.02), ly(1) - 10, "φ_{B}(1) = 1，切线水平", size=12, color="green")
svg.text(lx(0.52), ly(0.30), "φ_{A}：3.4445σ − 4.775σ³ + 2.0315σ⁵", size=11, color="orange")
svg.text(lx(0.52), ly(0.20), "φ_{B}：2σ − 1.5σ³ + 0.5σ⁵", size=11, color="green")

# ---------------------------------------------------------------- right: trajectories
RX0, RX1, RY0, RY1 = 440, 740, 44, 314


def rx(k):
    return RX0 + k / 10 * (RX1 - RX0)


def ry(v):
    return RY1 - v / VMAX * (RY1 - RY0)


svg.text(400, 24, "(b) 10 步混合迭代中的奇异值", size=13, weight="700")
svg.rect(RX0, ry(1.134), RX1 - RX0, ry(0.682) - ry(1.134), fill="muted", fill_opacity=0.14)
svg.text(rx(5.4), ry(0.40), "灰带：φ_{A}", size=11, color="muted")
svg.text(rx(5.4), ry(0.40) + 15, "的振荡带", size=11, color="muted")
svg.text(rx(5.4), ry(0.40) + 30, "[0.682, 1.134]", size=11, color="muted")
for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
    svg.line(RX0, ry(v), RX1, ry(v), color="grid", width=1)
    svg.text(RX0 - 6, ry(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
for k in range(11):
    svg.text(rx(k), RY1 + 16, str(k), size=11, anchor="middle", color="muted")
svg.line(RX0, RY1, RX1, RY1, color="muted", width=1.2)
svg.line(RX0, RY0, RX0, RY1, color="muted", width=1.2)
svg.line(RX0, ry(1), RX1, ry(1), color="green", width=1, dash="3 3")
svg.line(rx(8), RY0, rx(8), RY1, color="ink", width=1, dash="2 3")
svg.text(rx(8) + 4, RY0 + 12, "换成 φ_{B}", size=11, color="green")
svg.text((RX0 + RX1) / 2, RY1 + 34, "迭代步数", size=12, anchor="middle", color="muted")


def marker(x, y, kind, col):
    if kind == "circle":
        svg.circle(x, y, 3.6, fill=col)
    elif kind == "square":
        svg.rect(x - 3.4, y - 3.4, 6.8, 6.8, fill=col)
    else:
        svg.polygon([(x, y - 4.4), (x - 4, y + 3), (x + 4, y + 3)], fill=col)


for x0, lab, kind in TR:
    t = traj(x0)
    for k in range(10):
        col = "orange" if k < 8 else "green"
        svg.line(rx(k), ry(t[k]), rx(k + 1), ry(t[k + 1]), color=col, width=1.6, opacity=0.9)
    for k, v in enumerate(t):
        marker(rx(k), ry(v), kind, "orange" if k <= 8 else "green")
# legend for markers
svg.text(RX0, 374, "起点：", size=11, color="muted")
for n, (x0, lab, kind) in enumerate(TR):
    x = RX0 + 44 + n * 88
    marker(x, 370, kind, "ink")
    svg.text(x + 9, 374, lab, size=11)
svg.text(20, 398, "左图：一步迭代把一个奇异值映射到哪里；右图：三个起点的实际轨迹，每个点由 (18.20) 迭代算出。", size=11, color="muted")
svg.text(20, 416, "8 步 φ_{A} 后所有 σ₀ ∈ [0.001, 1] 落在 [0.682, 1.134]；再 2 步 φ_{B} 后全部落在 [1.0000, 1.0005]。", size=11, color="muted")

svg.save("l18-newton-schulz.svg")
