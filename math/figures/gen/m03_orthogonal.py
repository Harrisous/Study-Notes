"""Figure M3.1: an orthogonal map keeps the unit circle a unit circle; a shear does not.

Three 2x2 matrices act on the unit circle, on the basis vectors e1, e2 and on x = (2, 1):
  U = (1/5)[[3, -4], [4, 3]]  (rotation by atan2(4, 3) = 53.13 deg, orthogonal, det +1)
  P = [[0, 1], [1, 0]]        (swap the coordinates = mirror in y = x, orthogonal, det -1)
  S = [[1, 1], [0, 1]]        (shear, det +1 but NOT orthogonal)
Every point is computed here. Writes math/figures/m03-orthogonal.svg
"""

import math

import numpy as np

from m03_svg import SVG, Plane, fmt

U = np.array([[3, -4], [4, 3]]) / 5
P = np.array([[0, 1], [1, 0]], dtype=float)
S = np.array([[1, 1], [0, 1]], dtype=float)
x = np.array([2.0, 1.0])
for M in (U, P):
    assert np.allclose(M.T @ M, np.eye(2))
assert not np.allclose(S.T @ S, np.eye(2))

W, H = 740, 332
svg = SVG(W, H, "正交矩阵把单位圆送到单位圆，剪切矩阵把它压成椭圆")
panels = [
    (U, "blue", "U", "U：旋转 53.13°", ["U^{⊤}U = I，det U = +1", "‖U*x*‖ = ‖*x*‖ = 2.236"]),
    (P, "blue", "P", "P：交换两个坐标（镜像）", ["P^{⊤}P = I，det P = −1", "‖P*x*‖ = ‖*x*‖ = 2.236"]),
    (S, "red", "S", "S：剪切（不是正交矩阵）", ["S^{⊤}S ≠ I，det S = +1", "S*x* = (3, 1)，‖S*x*‖ = 3.162 ≠ 2.236"]),
]
PW = W / 3
scale = 42
theta = np.linspace(0, 2 * math.pi, 241)
circle = np.stack([np.cos(theta), np.sin(theta)])
# where to put the label of M x (dx, dy in screen units, anchor)
img_label = {0: (8, -4, "start"), 1: (8, -2, "start"), 2: (-2, -10, "end")}
x_label = {0: (6, 16), 1: (6, 16), 2: (6, -6)}
for k, (M, col, name, title, notes) in enumerate(panels):
    x0 = k * PW
    pl = Plane(x0 + 26 + 1.6 * scale, 50 + 2.45 * scale, scale)
    svg.text(x0 + PW / 2, 24, title, size=14, anchor="middle", weight="600", color="ink")
    ax0, ay0 = pl(-1.6, 0)
    ax1, _ = pl(3.25, 0)
    svg.line(ax0, ay0, ax1, ay0, color="rule", width=1)
    bx, by0 = pl(0, -1.55)
    _, by1 = pl(0, 2.45)
    svg.line(bx, by0, bx, by1, color="rule", width=1)
    pts = [pl(a, b) for a, b in circle.T]
    svg.path("M" + " L".join("%s %s" % (fmt(a), fmt(b)) for a, b in pts) + " Z", stroke="muted", width=1.2, dash="4 3")
    img = M @ circle
    pts = [pl(a, b) for a, b in img.T]
    svg.path("M" + " L".join("%s %s" % (fmt(a), fmt(b)) for a, b in pts) + " Z", stroke=col, width=2.2,
             fill=col, fill_opacity=0.06)
    ox, oy = pl(0, 0)
    for j in (0, 1):
        cvec = M[:, j]
        ex, ey = pl(cvec[0], cvec[1])
        svg.arrow(ox, oy, ex, ey, color=col, width=2.4)
        n = math.hypot(cvec[0], cvec[1])
        lx, ly = pl(cvec[0] + 0.32 * cvec[0] / n, cvec[1] + 0.32 * cvec[1] / n)
        if k == 2 and j == 0:  # S e1 = (1, 0): put the label under the axis, clear of the S x arrow
            lx, ly = pl(1.0, -0.42)
        svg.text(lx, ly + 5, "%s*e*_{%d}" % (name, j + 1), size=12, anchor="middle", color=col)
    xx, xy = pl(*x)
    svg.arrow(ox, oy, xx, xy, color="muted", width=1.6)
    dx, dy = x_label[k]
    svg.text(xx + dx, xy + dy, "*x*", size=13, color="muted")
    mx = M @ x
    mxx, mxy = pl(*mx)
    svg.arrow(ox, oy, mxx, mxy, color="ink", width=1.9)
    dx, dy, anc = img_label[k]
    lab = "%s*x* = (%s, %s)" % (name, fmt(mx[0]), fmt(mx[1])) if k < 2 else "%s*x*" % name  # no room at the right edge
    svg.text(mxx + dx, mxy + dy, lab, size=12, anchor=anc, color="ink")
    for i, s in enumerate(notes):
        svg.text(x0 + PW / 2, 250 + 21 * i, s, size=12.5, anchor="middle", color="ink" if i == 0 else col)
svg.text(W / 2, 302, "虚线：变换前的单位圆；实线：变换后的像。彩色粗箭头：矩阵的两列（e₁、e₂ 的去处）。",
         size=12, anchor="middle", color="muted")
svg.text(W / 2, 322, "灰箭头 x = (2, 1)；标着 Ux、Px、Sx 的箭头是它的像。",
         size=12, anchor="middle", color="muted")
svg.save("m03-orthogonal.svg")
