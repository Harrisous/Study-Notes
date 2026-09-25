"""Figure M3.6: a rotation does not commute with a stretch, but it does commute with a complex multiplication.

R = R(90 deg), Lambda = diag(2, 1) (stretch x by 2; S is the shear of figure M3.1), W = [[2, -1], [1, 2]] = 2I + J (multiplication by 2 + i).
x = (1, 1).
  (a) stretch then rotate: Lambda x = (2, 1), R Lambda x = (-1, 2)
  (b) rotate then stretch: R x = (-1, 1), Lambda R x = (-2, 1)
  (c) W and R in either order: W R x = R W x = (-3, 1)
The unit circle is pushed through the same two steps (dashed: after the first step, solid: after both).
Writes math/figures/m03-noncommute.svg
"""

import math

import numpy as np

from m03_svg import SVG, Plane, fmt

R = np.array([[0.0, -1.0], [1.0, 0.0]])
Lam = np.diag([2.0, 1.0])
Wm = np.array([[2.0, -1.0], [1.0, 2.0]])
x = np.array([1.0, 1.0])
assert np.allclose(R @ Lam @ x, [-1, 2]) and np.allclose(Lam @ R @ x, [-2, 1])
assert np.allclose(Wm @ R, R @ Wm) and np.allclose(Wm @ R @ x, [-3, 1])

W, H = 740, 352
svg = SVG(W, H, "旋转与拉伸交换次序后结果不同；旋转只与复数乘法交换")
theta = np.linspace(0, 2 * math.pi, 241)
circle = np.stack([np.cos(theta), np.sin(theta)])
PW = W / 3
s = 34
panels = [
    ("(a) 先拉伸 Λ，再旋转 R", Lam, R, "Λ*x* = (2, 1)", "RΛ*x* = (−1, 2)", "red"),
    ("(b) 先旋转 R，再拉伸 Λ", R, Lam, "R*x* = (−1, 1)", "ΛR*x* = (−2, 1)", "red"),
    ("(c) W = 2I + J 与 R：两种次序一样", Wm, R, "W*x* = (1, 3)", "RW*x* = WR*x* = (−3, 1)", "green"),
]


def poly(pl, pts, **kw):
    P = [pl(a, b) for a, b in pts.T]
    svg.path("M" + " L".join("%s %s" % (fmt(a), fmt(b)) for a, b in P) + " Z", **kw)


for k, (title, A, B, lab1, lab2, col) in enumerate(panels):
    x0 = k * PW
    pl = Plane(x0 + PW / 2 + 0.2 * s, 58 + 3.3 * s, s)
    svg.text(x0 + PW / 2, 24, title, size=13, anchor="middle", weight="600")
    a0, b0 = pl(-3.4, 0)
    a1, _ = pl(3.0, 0)
    svg.line(a0, b0, a1, b0, color="rule", width=1)
    c0, d0 = pl(0, -2.35)
    _, d1 = pl(0, 3.3)
    svg.line(c0, d0, c0, d1, color="rule", width=1)
    poly(pl, circle, stroke="muted", width=1, dash="2 3")
    poly(pl, A @ circle, stroke="muted", width=1.3, dash="5 3")
    poly(pl, B @ A @ circle, stroke=col, width=2.2, fill=col, fill_opacity=0.06)
    ox, oy = pl(0, 0)
    p1 = pl(*x)
    p2 = pl(*(A @ x))
    p3 = pl(*(B @ A @ x))
    svg.arrow(ox, oy, *p1, color="muted", width=1.6)
    svg.arrow(ox, oy, *p2, color="muted", width=1.6, dash="5 3")
    svg.arrow(ox, oy, *p3, color="ink", width=2.2)
    svg.text(p1[0] + 6, p1[1] + 4, "*x*", size=12.5, color="muted")
    svg.text(x0 + PW / 2, 272, lab1, size=12, anchor="middle", color="muted")
    svg.text(x0 + PW / 2, 292, lab2, size=12.5, anchor="middle", color="ink", weight="600")
svg.text(W / 3 / 2 + W / 6, 318, "(a)(b) 终点不同：RΛ ≠ ΛR，椭圆一个竖着、一个横着", size=12, anchor="middle", color="red")
svg.text(2 * PW + PW / 2, 318, "W 本身就是“拉伸 √5 倍 + 转 26.57°”", size=12, anchor="middle", color="green")
svg.text(W / 2, 344, "点线：单位圆；虚线：第一步后；实线：两步后。灰实线箭头 x = (1, 1)，灰虚线箭头是第一步的像，最粗的箭头是两步后的像。",
         size=11.5, anchor="middle", color="muted")
svg.line(PW, 44, PW, 300, color="rule", width=1)
svg.line(2 * PW, 44, 2 * PW, 300, color="rule", width=1)
svg.save("m03-noncommute.svg")
