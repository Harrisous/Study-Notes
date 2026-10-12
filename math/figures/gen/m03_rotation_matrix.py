"""Figure M3.2: where a rotation sends the two axes (= its columns), and a vector's angle adding up.

Left: phi = 30 deg. e1 -> (cos phi, sin phi) = (0.866, 0.5), e2 -> (-sin phi, cos phi) = (-0.5, 0.866);
      these two images are the two columns of R(phi).
Right: x = (2, 1) (angle alpha = 26.57 deg, length sqrt 5) is sent to R(30 deg) x = (1.232, 1.866)
      (angle 56.57 deg, same length).
Writes math/figures/m03-rotation-matrix.svg
"""

import math

import numpy as np

from m03_svg import SVG, Plane, arc_path, fmt

phi = math.radians(30)
R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
x = np.array([2.0, 1.0])
rx = R @ x
alpha = math.atan2(x[1], x[0])
assert abs(np.linalg.norm(rx) - math.sqrt(5)) < 1e-12

W, H = 740, 356
svg = SVG(W, H, "旋转矩阵的两列就是两个坐标轴被转到的位置；旋转一个向量就是给它的角度加上 φ")

# ---------------- left panel: the axes
s = 118
pl = Plane(40 + 0.75 * s, 62 + 1.4 * s, s)
svg.text(185, 24, "① 两个坐标轴转到哪里，矩阵的两列就是什么", size=14, anchor="middle", weight="600")
ax0, ay0 = pl(-0.75, 0)
ax1, _ = pl(1.28, 0)
svg.line(ax0, ay0, ax1, ay0, color="rule", width=1)
bx, by0 = pl(0, -0.22)
_, by1 = pl(0, 1.2)
svg.line(bx, by0, bx, by1, color="rule", width=1)
ox, oy = pl(0, 0)
svg.path(arc_path(ox, oy, s, math.radians(-8), math.radians(188)), stroke="muted", width=1, dash="4 3")
# original axes
ex, ey = pl(1, 0)
svg.arrow(ox, oy, ex, ey, color="muted", width=1.8)
svg.text(ex + 6, ey + 5, "*e*_{1}", size=13, color="muted")
ex2, ey2 = pl(0, 1)
svg.arrow(ox, oy, ex2, ey2, color="muted", width=1.8)
svg.text(ex2 + 8, ey2 + 6, "*e*_{2}", size=13, color="muted")
# images
c1 = R[:, 0]
c2 = R[:, 1]
p1 = pl(*c1)
p2 = pl(*c2)
# drop lines for column 1
svg.line(p1[0], p1[1], p1[0], oy, color="blue", width=1, dash="3 3", opacity=0.7)
svg.line(p1[0], p1[1], ox, p1[1], color="blue", width=1, dash="3 3", opacity=0.7)
svg.line(p2[0], p2[1], p2[0], oy, color="blue", width=1, dash="3 3", opacity=0.7)
svg.line(p2[0], p2[1], ox, p2[1], color="blue", width=1, dash="3 3", opacity=0.7)
svg.arrow(ox, oy, *p1, color="blue", width=2.6)
svg.arrow(ox, oy, *p2, color="blue", width=2.6)
svg.text(p1[0] + 6, p1[1] - 6, "R*e*_{1} = (cos φ, sin φ)", size=12.5, color="blue", weight="600")
svg.text(p2[0] - 30, p2[1] - 36, "R*e*_{2} = (−sin φ, cos φ)", size=12.5, color="blue", weight="600")
# coordinate readouts
svg.text(p1[0], oy + 16, "0.866", size=11.5, anchor="middle", color="blue")
svg.text(ox + 4, p1[1] + 13, "0.5", size=11.5, color="blue")
svg.text(p2[0], oy + 16, "−0.5", size=11.5, anchor="middle", color="blue")
svg.text(ox - 5, p2[1] - 5, "0.866", size=11.5, color="blue", anchor="end")
# angle arcs
svg.path(arc_path(ox, oy, 0.36 * s, 0, phi), stroke="red", width=1.8)
a = phi / 2
svg.text(ox + 0.46 * s * math.cos(a), oy - 0.46 * s * math.sin(a) + 5, "φ", size=13, color="red", anchor="middle")
svg.path(arc_path(ox, oy, 0.36 * s, math.pi / 2, math.pi / 2 + phi), stroke="red", width=1.8)
a = math.pi / 2 + phi / 2
svg.text(ox + 0.47 * s * math.cos(a), oy - 0.47 * s * math.sin(a) + 5, "φ", size=13, color="red", anchor="middle")
# matrix readout
mx, my = 40, 290
svg.text(mx, my + 23, "R(30°) =", size=13)
cell = 50
X0 = mx + 70
svg.rect(X0 + 2, my + 2, cell - 4, 2 * 26 - 4, fill="blue", fill_opacity=0.12, rx=4)
svg.rect(X0 + cell + 2, my + 2, cell - 4, 2 * 26 - 4, fill="blue", fill_opacity=0.12, rx=4)
vals = [["0.866", "−0.5"], ["0.5", "0.866"]]
for i in range(2):
    for j in range(2):
        svg.text(X0 + j * cell + cell / 2, my + 18 + 26 * i, vals[i][j], size=13, anchor="middle")
svg.path("M%s %s h-5 v%s h5" % (fmt(X0 + 1), fmt(my), 52), stroke="ink", width=1.3)
svg.path("M%s %s h5 v%s h-5" % (fmt(X0 + 2 * cell - 1), fmt(my), 52), stroke="ink", width=1.3)
svg.text(X0 + 2 * cell + 14, my + 18, "第 1 列 = R*e*_{1}", size=12, color="blue")
svg.text(X0 + 2 * cell + 14, my + 42, "第 2 列 = R*e*_{2}", size=12, color="blue")

# divider
svg.line(372, 40, 372, 330, color="rule", width=1)

# ---------------- right panel: a vector
s2 = 72
pr = Plane(410 + 0.35 * s2, 70 + 2.2 * s2, s2)
svg.text(560, 24, "② 转一个向量：角度加上 φ，长度不变", size=14, anchor="middle", weight="600")
ax0, ay0 = pr(-0.35, 0)
ax1, _ = pr(3.9, 0)
svg.line(ax0, ay0, ax1, ay0, color="rule", width=1)
bx, by0 = pr(0, -0.3)
_, by1 = pr(0, 2.3)
svg.line(bx, by0, bx, by1, color="rule", width=1)
ox, oy = pr(0, 0)
r = math.sqrt(5)
svg.path(arc_path(ox, oy, r * s2, alpha, alpha + phi), stroke="muted", width=1.2, dash="4 3")
px = pr(*x)
prx = pr(*rx)
svg.arrow(ox, oy, *px, color="muted", width=2.2)
svg.arrow(ox, oy, *prx, color="ink", width=2.4)
svg.text(px[0] + 6, px[1] + 4, "*x* = (2, 1)", size=12.5, color="muted")
svg.text(prx[0] + 6, prx[1] - 4, "R(30°)*x* = (1.232, 1.866)", size=12.5, color="ink")
svg.path(arc_path(ox, oy, 0.75 * s2, 0, alpha), stroke="muted", width=1.6)
svg.text(ox + 0.95 * s2 * math.cos(alpha / 2), oy - 0.95 * s2 * math.sin(alpha / 2) + 5, "α", size=13, color="muted", anchor="middle")
svg.path(arc_path(ox, oy, 1.05 * s2, alpha, alpha + phi), stroke="red", width=1.8)
m = alpha + phi / 2
svg.text(ox + 1.24 * s2 * math.cos(m), oy - 1.24 * s2 * math.sin(m) + 5, "φ", size=13, color="red", anchor="middle")
svg.text(560, 262, "α = 26.57°，α + φ = 56.57°", size=12.5, anchor="middle")
svg.text(560, 283, "两支箭头一样长：‖*x*‖ = ‖R*x*‖ = √5 ≈ 2.236（虚线圆弧）", size=12.5, anchor="middle")
svg.text(560, 312, "极坐标：(r cos α, r sin α) → (r cos(α+φ), r sin(α+φ))", size=12.5, anchor="middle", color="blue")
svg.text(560, 333, "与 R(φ) 乘出来的结果对比，就得到和角公式 (M3.5)", size=12, anchor="middle", color="muted")
svg.save("m03-rotation-matrix.svg")
