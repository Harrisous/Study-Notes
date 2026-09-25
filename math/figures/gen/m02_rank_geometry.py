"""Figure M2.1: rank = the dimension of the set of outputs.

(a) A  = 2 u1 v1^T + 0.8 u2 v2^T  (rank 2): the unit circle goes to an ellipse; outputs fill the plane.
(b) A1 = 2 u1 v1^T                (rank 1): the same circle collapses onto a segment along u1;
    the direction v2 is sent to 0 (null space), so x and x + t v2 land on the same point.
(c) M = [[1,0,1],[0,1,2],[1,1,3]] (rank 2): its three columns lie in one plane through the origin,
    because column 3 = column 1 + 2 * column 2.
v1 at 30 deg, u1 at 170 deg; v2, u2 are v1, u1 turned by +90 deg.
Writes math/figures/m02-rank-geometry.svg
"""

import math

import numpy as np

from m02_svg import SVG, Plane, polyline_d

deg = math.radians
V1, U1 = 30, 170
v1 = np.array([math.cos(deg(V1)), math.sin(deg(V1))])
v2 = np.array([math.cos(deg(V1 + 90)), math.sin(deg(V1 + 90))])
u1 = np.array([math.cos(deg(U1)), math.sin(deg(U1))])
u2 = np.array([math.cos(deg(U1 + 90)), math.sin(deg(U1 + 90))])
A = 2.0 * np.outer(u1, v1) + 0.8 * np.outer(u2, v2)
A1 = 2.0 * np.outer(u1, v1)
assert np.linalg.matrix_rank(A) == 2 and np.linalg.matrix_rank(A1) == 1

W, H = 780, 318
svg = SVG(W, H, "秩是输出集合的维数：满秩映射把圆变成椭圆，秩 1 映射把圆压成线段，3×3 秩 2 矩阵的三列共面")

th = np.radians(np.arange(0, 360, 15))
X = np.stack([np.cos(th), np.sin(th)])            # 24 points on the unit circle
ts = np.linspace(0, 2 * np.pi, 241)
C = np.stack([np.cos(ts), np.sin(ts)])

S, R = 42, 2.5
TOP = 56
OY = TOP + R * S


def fmt_m(M):
    e = [[("%.2f" % M[i, j]).replace("-", "−") for j in range(2)] for i in range(2)]
    return "= [[%s, %s], [%s, %s]]" % (e[0][0], e[0][1], e[1][0], e[1][1])


def panel2d(ox, title, sub, M, rank1, formula):
    P = Plane(ox, OY, S)
    x0, y0 = P(-R, R)
    x1, y1 = P(R, -R)
    svg.rect(x0, y0, x1 - x0, y1 - y0, stroke="rule", width=1, rx=6)
    svg.text(x0, 22, title, size=14, weight="600")
    svg.text(x0, 40, sub, size=12, color="muted")
    svg.line(*P(-R, 0), *P(R, 0), color="grid", width=1.2)
    svg.line(*P(0, -R), *P(0, R), color="grid", width=1.2)
    svg.path(polyline_d([P(*p) for p in C.T]) + " Z", stroke="muted", width=1.1, dash="4 4")
    for p in X.T:
        svg.circle(*P(*p), 2.6, stroke="muted", width=1.1, fill="surface")
    if rank1:
        svg.line(*P(*(-2.35 * v2)), *P(*(2.35 * v2)), color="red", width=1.6, dash="6 4")
        svg.text(*(np.array(P(*(2.2 * v2))) + (-8, 4)), "零空间", size=12, color="red", anchor="end")
        # two inputs with the same v1-component (30 +- 60 deg) map to the same output
        pa = np.array([math.cos(deg(V1 + 60)), math.sin(deg(V1 + 60))])
        pb = np.array([math.cos(deg(V1 - 60)), math.sin(deg(V1 - 60))])
        out = M @ pa
        assert np.allclose(out, M @ pb)
        svg.line(*P(*(-2 * u1)), *P(*(2 * u1)), color="green", width=3.2, cap="round")
        for p in (pa, pb):
            svg.line(*P(*p), *P(*out), color="ink", width=1.1, dash="2 3", opacity=0.8)
            svg.circle(*P(*p), 4.2, fill="blue")
        svg.text(*(np.array(P(*pb)) + (2, 24)), "两个输入", size=12, color="blue")
        svg.circle(*P(*out), 5.2, fill="green")
        # 18 units up: at -8 the label sat on the thick green segment
        svg.text(*(np.array(P(*out)) + (-8, -18)), "同一个输出", size=12, color="green", anchor="end")
    else:
        img = M @ C
        svg.path(polyline_d([P(*p) for p in img.T]) + " Z", stroke="green", width=2, fill="green",
                 fill_opacity=0.08)
    for p in (M @ X).T:
        svg.circle(*P(*p), 2.8, fill="green")
    svg.arrow(*P(0, 0), *P(*v1), color="blue", width=2.2)
    # in (a) the tip of v1 lies on the ellipse: put its label above the outline, and 2u1's label above-left
    # of the ellipse's left end (below it the label crossed the outline)
    v1_off, u1_off, u1_anchor = ((6, -14), (-19, -22), "start") if not rank1 else ((5, -4), (2, 20), "middle")
    svg.text(*(np.array(P(*v1)) + v1_off), "*v*_{1}", size=13, color="blue")
    svg.arrow(*P(0, 0), *P(*(2 * u1)), color="green", width=2.2)
    svg.text(*(np.array(P(*(2 * u1))) + u1_off), "2*u*_{1}", size=13, color="green", anchor=u1_anchor)
    svg.text(x0, y1 + 22, formula, size=12)
    svg.text(x0, y1 + 40, fmt_m(M), size=12, color="muted")


panel2d(14 + R * S, "(a) 秩 2：圆 → 椭圆", "输出铺满整个平面", A, False,
        "*A* = 2*u*_{1}*v*_{1}^{⊤} + 0.8*u*_{2}*v*_{2}^{⊤}")
panel2d(244 + R * S, "(b) 秩 1：圆 → 线段", "输出全在 *u*_{1} 这条直线上", A1, True,
        "*A*_{1} = 2*u*_{1}*v*_{1}^{⊤}（0.8 换成 0）")

# ---- (c) 3-D column space of M
M3 = np.array([[1., 0, 1], [0, 1, 2], [1, 1, 3]])
assert np.linalg.matrix_rank(M3) == 2
assert np.allclose(M3 @ np.array([1, 2, -1.]), 0)
c1, c2, c3 = M3[:, 0], M3[:, 1], M3[:, 2]
ox, oy, s = 610, 236, 40
EX, EY, EZ = np.array([-0.62, 0.40]), np.array([0.95, 0.16]), np.array([0.0, -1.0])


def P3(v):
    q = v[0] * EX + v[1] * EY + v[2] * EZ
    return ox + s * q[0], oy + s * q[1]


svg.rect(474, TOP, 292, 2 * R * S, stroke="rule", width=1, rx=6)
svg.text(474, 22, "(c) 3×3 的矩阵 *M*：秩 2", size=14, weight="600")
svg.text(474, 40, "三列都落在一个过原点的平面里", size=12, color="muted")
for v, lab, dx, dy in ((np.array([1.2, 0, 0]), "x", -12, 6), (np.array([0, 2.6, 0]), "y", 5, 4),
                       (np.array([0, 0, 3.6]), "z", 6, 4)):
    svg.line(*P3(np.zeros(3)), *P3(v), color="muted", width=1)
    svg.text(*(np.array(P3(v)) + (dx, dy)), lab, size=12, color="muted")
corners = [(-0.25, -0.25), (1.35, -0.25), (1.35, 2.3), (-0.25, 2.3)]
svg.polygon([P3(a * c1 + b * c2) for a, b in corners], fill="green", fill_opacity=0.10, stroke="green", width=1,
            dash="5 4")
for a in (0, 0.5, 1.0):
    svg.line(*P3(a * c1 - 0.25 * c2), *P3(a * c1 + 2.3 * c2), color="green", width=0.8, opacity=0.35)
for b in (0, 1, 2):
    svg.line(*P3(-0.25 * c1 + b * c2), *P3(1.35 * c1 + b * c2), color="green", width=0.8, opacity=0.35)
svg.line(*P3(c1), *P3(c1 + 2 * c2), color="ink", width=1, dash="3 3")
svg.line(*P3(2 * c2), *P3(c1 + 2 * c2), color="ink", width=1, dash="3 3")
svg.line(*P3(c2), *P3(2 * c2), color="green", width=1.4, dash="3 3")
svg.arrow(*P3(np.zeros(3)), *P3(c1), color="blue", width=2.2)
svg.arrow(*P3(np.zeros(3)), *P3(c2), color="green", width=2.2)
svg.arrow(*P3(np.zeros(3)), *P3(c3), color="violet", width=2.4)
# labels kept off the dashed outline of the plane: its left corner sits right next to the tip of column 1,
# so column 1's label goes further left; column 2's label goes below the lower edge; column 3's to the right
svg.text(*(np.array(P3(c1)) + (-23, 4)), "列 1 = (1, 0, 1)", size=12, color="blue", anchor="end")
svg.text(*(np.array(P3(c2)) + (14, 24)), "列 2 = (0, 1, 1)", size=12, color="green")
svg.text(*(np.array(P3(c3)) + (7, -26)), "列 3 = (1, 2, 3)", size=12, color="violet")
svg.text(474, TOP + 2 * R * S + 22, "列 3 = 列 1 + 2 × 列 2，没有带来新方向", size=12)
svg.text(474, TOP + 2 * R * S + 40, "*M*(1, 2, −1) = *0*：零空间的方向", size=12, color="red")

svg.save("m02-rank-geometry.svg")
