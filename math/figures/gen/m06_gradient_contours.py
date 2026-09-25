"""Figure M6.3: contours of f(x, y) = x^2 + 3y^2, gradient arrows, and two gradient-descent paths.

Left: level sets f = c (ellipses, computed), gradient vectors at a grid of points (scaled down),
the point (2, 1) with grad f = (4, 6), and two directions from that point with their directional
derivatives. Right part of the same axes: gradient-descent paths from (2, 1) with eta = 0.1 and
eta = 0.3 (the second zig-zags in y because 1 - 6*eta = -0.8). All points are computed here.
Writes math/figures/m06-gradient-contours.svg
"""

import math

from m06_svg import SVG, fmt

f = lambda x, y: x * x + 3 * y * y
grad = lambda x, y: (2 * x, 6 * y)

W, H = 720, 430
X0, X1 = 50, 470          # plot area in px
Y0, Y1 = 40, 390
XMIN, XMAX = -3.0, 3.0
YMIN, YMAX = -2.5, 2.5
sx = (X1 - X0) / (XMAX - XMIN)
sy = (Y1 - Y0) / (YMAX - YMIN)
s = min(sx, sy)           # equal aspect so that "perpendicular" looks perpendicular
cxp = (X0 + X1) / 2
cyp = (Y0 + Y1) / 2
px = lambda x: cxp + x * s
py = lambda y: cyp - y * s

svg = SVG(W, H, "f(x, y) = x² + 3y² 的等高线、梯度与梯度下降路径")
# axes
svg.line(px(-3), py(0), px(3), py(0), color="rule", width=1.2)
svg.line(px(0), py(-2.5), px(0), py(2.5), color="rule", width=1.2)
for t in (-3, -2, -1, 1, 2, 3):
    svg.text(px(t), py(0) + 15, str(t), size=11, anchor="middle", color="muted")
for t in (-2, -1, 1, 2):
    svg.text(px(0) - 7, py(t) + 4, str(t), size=11, anchor="end", color="muted")
svg.text(px(3) + 4, py(0) + 4, "x", size=12, color="muted")
svg.text(px(0) + 6, py(2.5) + 4, "y", size=12, color="muted")

# contours f = c: ellipse x = sqrt(c) cos t, y = sqrt(c/3) sin t; split where it leaves the window
levels = [0.5, 1, 2, 4, 7, 10]
for c in levels:
    runs, cur = [], []
    for i in range(361):
        t = 2 * math.pi * i / 360
        x, y = math.sqrt(c) * math.cos(t), math.sqrt(c / 3) * math.sin(t)
        if XMIN <= x <= XMAX and YMIN <= y <= YMAX:
            cur.append("%.1f %.1f" % (px(x), py(y)))
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    for r in runs:
        svg.path("M" + " L".join(r), stroke="muted", width=1.8 if c == 7 else 1.1, opacity=0.9 if c == 7 else 0.55)
    if c in (1, 4, 10):
        lab_x, lab_y = math.sqrt(c) * math.cos(2.2), math.sqrt(c / 3) * math.sin(2.2)
        svg.text(px(lab_x) - 3, py(lab_y) - 4, "f=%s" % fmt(c), size=11, anchor="end", color="muted")

# gradient directions on the f = 2 and f = 10 contours (unit length 0.42, direction only)
for c, ts in ((2, (0.5, 1.2, 2.0, 3.6, 4.4, 5.4)), (10, (0.35, 1.2, 1.94, 2.8, 3.5, 4.35, 5.1, 5.93))):
    for t in ts:
        x, y = math.sqrt(c) * math.cos(t), math.sqrt(c / 3) * math.sin(t)
        if not (XMIN <= x <= XMAX and YMIN <= y <= YMAX):
            continue
        a, b = grad(x, y)
        n = math.hypot(a, b)
        svg.arrow(px(x), py(y), px(x + 0.42 * a / n), py(y + 0.42 * b / n), color="muted", width=1.2)

# point (2, 1): gradient and two directions
P = (2.0, 1.0)
g = grad(*P)
ng = math.hypot(*g)
k = 0.14
svg.arrow(px(P[0]), py(P[1]), px(P[0] + k * g[0]), py(P[1] + k * g[1]), color="red", width=2.4)
svg.text(px(P[0] + k * g[0]) - 8, py(P[1] + k * g[1]) + 4, "∇f = (4, 6)", size=13, anchor="end", color="red", weight="600")
# tangent to the contour at P: perpendicular to grad
tx, ty = -g[1] / ng, g[0] / ng
svg.line(px(P[0] - 0.6 * tx), py(P[1] - 0.6 * ty), px(P[0] + 0.6 * tx), py(P[1] + 0.6 * ty), color="red", width=1.3, dash="4 3")
# label the tangent at its lower-right end, in the gap between the f = 7 and f = 10 contours
# (the upper-left end lies on the thick f = 7 contour itself, so a label there collides with it)
svg.text(px(P[0] - 0.6 * tx) + 12, py(P[1] - 0.6 * ty) + 22, "切线", size=12, color="red")

# gradient descent paths
for eta, col, lab in ((0.1, "blue", "η = 0.1"), (0.3, "violet", "η = 0.3")):
    x, y = P
    path = [(x, y)]
    for _ in range(12):
        a, b = grad(x, y)
        x, y = x - eta * a, y - eta * b
        path.append((x, y))
    svg.path("M" + " L".join("%.1f %.1f" % (px(u), py(v)) for u, v in path), stroke=col, width=1.8)
    for u, v in path:
        svg.circle(px(u), py(v), 2.8, fill=col)
    print(eta, [("%.3f" % u, "%.3f" % v, "%.3f" % f(u, v)) for u, v in path[:4]])
svg.circle(px(P[0]), py(P[1]), 5, fill="ink")
svg.text(px(P[0]) + 9, py(P[1]) + 5, "P", size=13, weight="700")

# right panel: numbers
RX = 500
yy = 56
svg.text(RX, yy, "在点 P = (2, 1)，f = 7：", size=13, weight="600"); yy += 22
svg.text(RX, yy, "∂f/∂x = 2x = 4，∂f/∂y = 6y = 6", size=12); yy += 20
svg.text(RX, yy, "|∇f| = √52 = 7.211", size=12); yy += 28
svg.text(RX, yy, "方向导数 = ∇f · u（u 为单位向量）", size=12, color="muted"); yy += 20
svg.text(RX, yy, "u = (1, 0)：4", size=12); yy += 18
svg.text(RX, yy, "u = (0.6, 0.8)：7.2", size=12); yy += 18
svg.text(RX, yy, "u = ∇f/|∇f|：7.211（最大）", size=12, color="red"); yy += 18
svg.text(RX, yy, "u 沿等高线（红色虚线）：0", size=12, color="red"); yy += 30
svg.line(RX, yy - 4, RX + 22, yy - 4, color="blue", width=2)
svg.text(RX + 28, yy, "η = 0.1：(2,1) → (1.6, 0.4)", size=12, color="blue"); yy += 18
svg.text(RX + 28, yy, "f：7 → 3.04", size=12, color="blue"); yy += 24
svg.line(RX, yy - 4, RX + 22, yy - 4, color="violet", width=2)
svg.text(RX + 28, yy, "η = 0.3：(2,1) → (0.8, −0.8)", size=12, color="violet"); yy += 18
svg.text(RX + 28, yy, "y 方向每步乘 1 − 6η = −0.8，", size=12, color="violet"); yy += 18
svg.text(RX + 28, yy, "来回摆动", size=12, color="violet"); yy += 28
svg.text(RX, yy, "灰色箭头：各处 ∇f 的方向（只画方向），", size=12, color="muted"); yy += 18
svg.text(RX, yy, "处处垂直于等高线、指向 f 增大；", size=12, color="muted"); yy += 18
svg.text(RX, yy, "等高线 f = 0.5, 1, 2, 4, 7, 10", size=12, color="muted")

svg.save("m06-gradient-contours.svg")
