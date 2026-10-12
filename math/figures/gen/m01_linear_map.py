"""Figure M1.3: a 2x2 matrix as a map of the plane.

A = [[2, -1], [1, 1]] sends e1 -> (2, 1) (first column), e2 -> (-1, 1) (second column),
the unit square to a parallelogram of area det A = 3, and x = (1, 2) to A x = (0, 3).
Writes math/figures/m01-linear-map.svg
"""

import numpy as np

from m01_svg import SVG

A = np.array([[2.0, -1.0], [1.0, 1.0]])
e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
x = np.array([1.0, 2.0])
Ax = A @ x
det = float(np.linalg.det(A))

W, H = 700, 290
S = 42
LEFT = dict(o=(62, 238), xr=(-0.9, 3.9), yr=(-0.9, 4.3))
RIGHT = dict(o=(452, 238), xr=(-2.8, 4.9), yr=(-0.9, 4.3))

svg = SVG(W, H, "矩阵 A 把平面上的网格、基向量和一个点一起变换")


def P(panel, v):
    o = panel["o"]
    return o[0] + S * v[0], o[1] - S * v[1]


def panel_rect(panel):
    x0, y1 = P(panel, (panel["xr"][0], panel["yr"][0]))
    x1, y0 = P(panel, (panel["xr"][1], panel["yr"][1]))
    return x0, y0, x1, y1


def clipped_line(rect, a, b, **kw):
    """Liang-Barsky: draw only the part of segment a-b inside rect. Geometry is clipped here,
    not with an SVG clipPath, because clipped-away geometry still widens the page's scroll
    area once the SVG is inlined with overflow: visible."""
    x0, y0, x1, y1 = rect
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if p == 0:
            if q < 0:
                return
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 < t1:
        svg.line(ax + t0 * dx, ay + t0 * dy, ax + t1 * dx, ay + t1 * dy, **kw)


def grid(panel, rect, M, color, opacity):
    for i in range(-6, 8):
        for a, b in ((np.array([i, -6.0]), np.array([i, 8.0])), (np.array([-6.0, i]), np.array([8.0, i]))):
            clipped_line(rect, P(panel, M @ a), P(panel, M @ b), color=color, width=1, opacity=opacity)


I = np.eye(2)
for panel, M in ((LEFT, I), (RIGHT, A)):
    rect = panel_rect(panel)
    x0, y0, x1, y1 = rect
    svg.rect(x0, y0, x1 - x0, y1 - y0, stroke="rule", width=1, rx=6)
    if M is A:
        grid(panel, rect, I, "muted", 0.18)          # faint original grid for reference
        grid(panel, rect, A, "blue", 0.45)
    else:
        grid(panel, rect, I, "muted", 0.45)
    # axes
    clipped_line(rect, P(panel, (-9, 0)), P(panel, (9, 0)), color="muted", width=1.2)
    clipped_line(rect, P(panel, (0, -9)), P(panel, (0, 9)), color="muted", width=1.2)
    # unit square and its image
    sq = [M @ v for v in (np.zeros(2), e1, e1 + e2, e2)]
    svg.polygon([P(panel, v) for v in sq], fill="amber", fill_opacity=0.16, stroke="amber", width=1.2)
    # basis vectors
    svg.arrow(*P(panel, (0, 0)), *P(panel, M @ e1), color="blue", width=2.6)
    svg.arrow(*P(panel, (0, 0)), *P(panel, M @ e2), color="green", width=2.6)
    # x as a combination of the (images of the) basis vectors: 1 step of col 1, then 2 of col 2
    mid = M @ (1 * e1)
    svg.line(*P(panel, mid), *P(panel, M @ x), color="green", width=1.4, dash="5 4")
    svg.arrow(*P(panel, (0, 0)), *P(panel, M @ x), color="ink", width=2.2)
    svg.circle(*P(panel, M @ x), 3.4, fill="ink")

# labels, left
svg.text(LEFT["o"][0] - 24, 26, "变换前", size=14, weight="600")
svg.text(*(np.array(P(LEFT, e1)) + (6, 18)), "*e*_{1} = (1, 0)", size=13, color="blue")
svg.text(*(np.array(P(LEFT, e2)) + (-8, -6)), "*e*_{2}", size=13, color="green", anchor="end")
svg.text(*(np.array(P(LEFT, x)) + (8, -4)), "*x* = (1, 2)", size=13)
svg.text(*(np.array(P(LEFT, x)) + (8, 14)), "= 1·*e*_{1} + 2·*e*_{2}", size=12, color="muted")
svg.text(*(np.array(P(LEFT, (0.5, 0.5))) + (0, 5)), "面积 1", size=12, color="amber", anchor="middle")

# labels, right
svg.text(RIGHT["o"][0] - 100, 26, "变换后：每个点 *v* 移到 *Av*", size=14, weight="600")
svg.text(*(np.array(P(RIGHT, A @ e1)) + (8, 6)), "*Ae*_{1} = (2, 1)", size=13, color="blue")
svg.text(*(np.array(P(RIGHT, A @ e2)) + (-8, -8)), "*Ae*_{2}", size=13, color="green", anchor="end")
svg.text(*(np.array(P(RIGHT, A @ e2)) + (-8, 12)), "(−1, 1)", size=13, color="green", anchor="end")
# both lines sit left of the black arrow, clear of the dashed "2 steps of new green" segment
svg.text(*(np.array(P(RIGHT, Ax)) + (-10, 4)), "*Ax* = (0, 3)", size=13, anchor="end")
svg.text(*(np.array(P(RIGHT, Ax)) + (-10, 22)), "= 1·*Ae*_{1} + 2·*Ae*_{2}", size=12, color="muted", anchor="end")
svg.text(*(np.array(P(RIGHT, (0.5, 1.0))) + (0, 5)), "面积 %d" % round(det), size=12, color="amber", anchor="middle")

# the matrix between the panels, with its columns coloured like the basis images
cell = 28
mx, my = 256, 96
svg.text(mx + cell, my - 14, "乘以 *A*", size=14, anchor="middle", weight="600")
svg.rect(mx + 2, my + 2, cell - 4, 2 * cell - 4, fill="blue", fill_opacity=0.14, rx=4)
svg.rect(mx + cell + 2, my + 2, cell - 4, 2 * cell - 4, fill="green", fill_opacity=0.14, rx=4)
svg.matrix(mx, my, [[2, "−1"], [1, 1]], cell=cell, size=14)
svg.text(mx + 12, my + 2 * cell + 20, "列 1", size=12, color="blue", anchor="middle")
svg.text(mx + cell + 16, my + 2 * cell + 20, "列 2", size=12, color="green", anchor="middle")
svg.arrow(mx - 6, my + 2 * cell + 50, mx + 2 * cell + 8, my + 2 * cell + 50, color="muted", width=1.4)

svg.save("m01-linear-map.svg")
