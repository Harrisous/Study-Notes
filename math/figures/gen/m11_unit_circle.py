"""Figure M11.4: what three 2x2 matrices do to the unit circle.

(a) doubly stochastic  [[0.8, 0.2], [0.2, 0.8]]  singular values 1, 0.6   -> ellipse inside the circle
(b) row stochastic     [[0.9, 0.1], [0.6, 0.4]]  singular values 1.1265, 0.2663 -> pokes out
(c) unconstrained H    [[1.2, 0.3], [0.1, 0.9]]  singular values 1.3048, 0.8048 (M06 / M11 §7)
The image ellipse, the most-stretched input v1 and its image A v1 are computed with numpy SVD.
Writes math/figures/m11-unit-circle.svg
"""

import math

import numpy as np

from m11_svg import SVG

cases = [
    ("(a) 双随机", [[0.8, 0.2], [0.2, 0.8]], "green", "行和 1, 1；列和 1, 1"),
    ("(b) 只有行和为 1", [[0.9, 0.1], [0.6, 0.4]], "orange", "行和 1, 1；列和 1.5, 0.5"),
    ("(c) 无约束", [[1.2, 0.3], [0.1, 0.9]], "red", "行和 1.5, 1；列和 1.3, 1.2"),
]

W, H = 760, 444
S = 78                      # pixels per unit
CY = 200
svg = SVG(W, H, "三个 2×2 矩阵把单位圆变成什么：双随机矩阵的像留在圆内，另外两个会伸出圆外")

for k, (title, Am, col, sums) in enumerate(cases):
    A = np.array(Am, float)
    cx = 130 + 250 * k
    U, s, Vt = np.linalg.svd(A)
    svg.text(cx, 30, title, size=13, anchor="middle", weight="600", color=col)
    svg.text(cx, 50, "A = [%s ; %s]" % (", ".join("%g" % v for v in A[0]), ", ".join("%g" % v for v in A[1])),
             size=12, anchor="middle", color="muted")
    # axes
    R = 1.45 * S
    svg.line(cx - R, CY, cx + R, CY, color="grid", width=1)
    svg.line(cx, CY - R, cx, CY + R, color="grid", width=1)
    # unit circle
    svg.circle(cx, CY, S, stroke="muted", width=1.2, dash="4 3")
    # image ellipse
    th = np.linspace(0, 2 * math.pi, 241)
    pts = A @ np.vstack([np.cos(th), np.sin(th)])
    d = "M" + " L".join("%.1f %.1f" % (cx + S * x, CY - S * y) for x, y in pts.T) + " Z"
    svg.path(d, stroke=col, width=2.2, fill=col, fill_opacity=0.12, join="round")
    # most stretched input v1 and its image
    v1 = Vt[0]
    if v1[0] < 0:
        v1 = -v1
    img = A @ v1
    svg.line(cx, CY, cx + S * v1[0], CY - S * v1[1], color="ink", width=1.4, dash="3 3")
    svg.circle(cx + S * v1[0], CY - S * v1[1], 3.2, fill="ink")
    svg.arrow(cx, CY, cx + S * img[0], CY - S * img[1], color=col, width=2.2)
    # the all-ones direction (fixed when rows sum to 1)
    o = np.array([1, 1]) / math.sqrt(2)
    Ao = A @ o
    if k < 2:
        svg.circle(cx + S * o[0], CY - S * o[1], 4.2, fill="surface", stroke="blue", width=1.6)
    svg.text(cx, CY + R + 22, "最大拉伸 σ₁ = %.4f" % s[0], size=13, anchor="middle", color=col, weight="600")
    svg.text(cx, CY + R + 42, "最小 σ₂ = %.4f" % s[1], size=12, anchor="middle", color="muted")
    svg.text(cx, CY + R + 62, sums, size=12, anchor="middle", color="muted")
    print("case", k, "sigma", s, "v1", v1, "A v1", img, "A(1,1)/sqrt2", Ao)

# legend
LY = H - 26
svg.line(40, LY, 64, LY, color="muted", width=1.2, dash="4 3")
svg.text(70, LY + 4, "单位圆（所有长度为 1 的输入）", size=12, color="muted")
svg.line(290, LY, 314, LY, color="ink", width=1.4, dash="3 3")
svg.text(320, LY + 4, "被拉得最长的输入 v₁", size=12, color="muted")
svg.circle(478, LY, 4.2, fill="surface", stroke="blue", width=1.6)
svg.text(488, LY + 4, "方向 (1, 1)/√2：行和为 1 时它是不动点", size=12, color="blue")

svg.save("m11-unit-circle.svg")
