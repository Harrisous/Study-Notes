"""Figure M3.7: the block-diagonal RoPE matrix R_{Theta,t} for d = 8, base b = 10000, at position t = 3.

theta_i = 10000^(-2i/8) = 10^(-i), i = 0..3, so the four 2x2 blocks rotate by
3, 0.3, 0.03, 0.003 rad (171.9, 17.2, 1.72, 0.17 degrees). Every entry is computed here.
Writes math/figures/m03-block-rotation.svg
"""

import math

import numpy as np

from m03_svg import SVG, arc_path, fmt

d, t, b = 8, 3, 10000.0
theta = b ** (-np.arange(0, d, 2) / d)
assert np.allclose(theta, [1, 0.1, 0.01, 0.001])
M = np.zeros((d, d))
for i, th in enumerate(theta):
    a = t * th
    M[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
assert np.allclose(M.T @ M, np.eye(d))


def num(v):
    s = "%.3f" % v
    if s in ("-0.000", "0.000"):
        s = "0"
    return s.replace("-", "−")


W, H = 740, 444
svg = SVG(W, H, "d = 8 的 RoPE 旋转矩阵在位置 t = 3：四个 2×2 旋转沿对角线排开，各转各的角度")
svg.text(W / 2, 22, "R_{Θ,t}（d = 8，b = 10000，t = 3）：4 个转速不同的 2×2 旋转沿对角线排开，其余全是 0",
         size=13.5, anchor="middle", weight="600")
cell = 44
X0, Y0 = 58, 54
cols = ["blue", "violet", "green", "orange"]
for i in range(4):
    svg.rect(X0 + 2 * i * cell + 1, Y0 + 2 * i * cell + 1, 2 * cell - 2, 2 * cell - 2, fill=cols[i], fill_opacity=0.12,
             stroke=cols[i], width=1.2, rx=4)
for r in range(d):
    for c in range(d):
        v = M[r, c]
        inblock = r // 2 == c // 2
        svg.text(X0 + c * cell + cell / 2, Y0 + r * cell + cell / 2 + 4, num(v) if inblock else "0", size=11.5 if inblock else 11,
                 anchor="middle", color="ink" if inblock else "muted")
# brackets
h = d * cell
svg.path("M%s %s h-6 v%s h6" % (fmt(X0), fmt(Y0), fmt(h)), stroke="ink", width=1.4)
svg.path("M%s %s h6 v%s h-6" % (fmt(X0 + h), fmt(Y0), fmt(h)), stroke="ink", width=1.4)
for i in range(4):
    svg.text(X0 - 12, Y0 + (2 * i + 1) * cell + 4, "对 %d" % i, size=12, anchor="end", color=cols[i])
# column index labels
for c in range(d):
    svg.text(X0 + c * cell + cell / 2, Y0 - 8, "%d" % c, size=11, anchor="middle", color="muted")

# clocks on the right
CX0 = X0 + h + 30
labels_theta = ["θ₀ = 1", "θ₁ = 0.1", "θ₂ = 0.01", "θ₃ = 0.001"]
for i in range(4):
    row, colp = divmod(i, 2)
    cx = CX0 + 64 + colp * 150
    cy = Y0 + 62 + row * 184
    r = 40
    col = cols[i]
    svg.circle(cx, cy, r, stroke="rule", width=1.4, fill="surface")
    for kk in range(12):
        a = kk * math.pi / 6
        svg.line(cx + (r - 4) * math.cos(a), cy - (r - 4) * math.sin(a), cx + r * math.cos(a), cy - r * math.sin(a),
                 color="muted", width=1)
    ang = t * theta[i]
    svg.line(cx, cy, cx + r * 0.95, cy, color="muted", width=1.3, dash="3 2")
    svg.path(arc_path(cx, cy, 0.42 * r, 0, ang), stroke="red", width=1.8)
    svg.arrow(cx, cy, cx + 0.92 * r * math.cos(ang), cy - 0.92 * r * math.sin(ang), color=col, width=2.4)
    svg.text(cx, cy + r + 18, "对 %d：%s" % (i, labels_theta[i]), size=12, anchor="middle", color=col, weight="600")
    svg.text(cx, cy + r + 36, "转 tθ = %s rad = %s°" % (fmt(ang) if ang >= 0.01 else "0.003", "%.2f" % math.degrees(ang) if ang < 1 else "%.1f" % math.degrees(ang)),
             size=11.5, anchor="middle")
    lam = 2 * math.pi / theta[i]
    svg.text(cx, cy + r + 53, "波长 2π/θ = %s token" % ("%.2f" % lam if lam < 100 else "%.0f" % lam), size=11.5, anchor="middle",
             color="muted")
svg.text(W / 2, 432, "每个 2×2 块都是 (M3.4) 的 R(tθᵢ)，只搅动自己那两个分量。虚线是 t = 0 时的指针，彩色箭头是 t = 3 时的指针。",
         size=11.5, anchor="middle", color="muted")
svg.save("m03-block-rotation.svg")
