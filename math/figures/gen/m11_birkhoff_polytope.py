"""Figure M11.6: the Birkhoff polytope for n = 2 (a segment) and a linear 2-D projection of B_3.

B_3 lives in a 4-dimensional affine subspace of R^{3x3}, so it cannot be drawn faithfully. We use the
linear map  M -> z(M) + e^{i*pi/3} w(M),  z(M) = (1/3) sum_ij M_ij omega^(j-i),  w(M) = (1/3) sum_ij M_ij omega^(i+j),
omega = e^{2 pi i/3}. It sends the three even permutations (cyclic shifts) to 1, omega, omega^2 (z-part) and
the three odd permutations (transpositions) to the three other vertices of a regular hexagon (w-part);
J_3 goes to 0. Because the map is linear, a convex combination of permutation matrices lands on the same
convex combination of their images, so the decomposition of the running example A can be drawn as weights.
Distances and angles are NOT preserved. Writes math/figures/m11-birkhoff-polytope.svg
"""

import cmath
import itertools
import math

import numpy as np

from m11_svg import SVG, fmt

om = cmath.exp(2j * math.pi / 3)
rot = cmath.exp(1j * math.pi / 3)


def P(pi):
    M = np.zeros((3, 3))
    for i in range(3):
        M[i, pi[i] - 1] = 1
    return M


def proj(M):
    z = sum(M[i, j] * om ** (j - i) for i in range(3) for j in range(3)) / 3
    w = sum(M[i, j] * om ** (i + j) for i in range(3) for j in range(3)) / 3
    c = (z + rot * w) * cmath.exp(1j * math.pi / 2)      # rotate so that I sits at the top
    return c.real, c.imag


perms = list(itertools.permutations((1, 2, 3)))
even = [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
odd = [(1, 3, 2), (3, 2, 1), (2, 1, 3)]
pts = {pi: proj(P(pi)) for pi in perms}
for pi, p in pts.items():
    print(pi, "%.3f %.3f" % p, "r=%.3f" % math.hypot(*p))
J = np.ones((3, 3)) / 3
print("J ->", proj(J))
A = np.array([[0.6, 0.4, 0], [0.1, 0.3, 0.6], [0.3, 0.3, 0.4]])
dec = [((1, 2, 3), 0.3), ((2, 3, 1), 0.3), ((1, 3, 2), 0.3), ((2, 1, 3), 0.1)]
assert np.allclose(sum(w * P(pi) for pi, w in dec), A)
pa = proj(A)
print("A ->", pa, "check", sum(w * np.array(pts[pi]) for pi, w in dec))
Dc = 0.5 * np.eye(3) + 0.3 * P((2, 3, 1)) + 0.2 * P((3, 1, 2))
pd = proj(Dc)

W, H = 760, 522
svg = SVG(W, H, "Birkhoff 多面体：n = 2 是一条线段；n = 3 是 4 维的，这里画的是一个线性投影")

# ---- (a) B_2
svg.text(24, 30, "(a) n = 2：一条线段", size=13, weight="600")
svg.text(24, 50, "全部 2×2 双随机矩阵 [a, 1−a; 1−a, a]", size=12, color="muted")
X1, X2, YS = 50, 250, 226
svg.line(X1, YS, X2, YS, color="violet", width=3)


def tile2(x, y, on, cell=18):
    for i in range(2):
        for j in range(2):
            svg.rect(x + j * cell + 1, y + i * cell + 1, cell - 2, cell - 2,
                     fill="violet" if on[i][j] else "grid", fill_opacity=0.85 if on[i][j] else 1, rx=3)


for x, on, lab, a in [(X1, [[1, 0], [0, 1]], "I（a = 1）", 1), (X2, [[0, 1], [1, 0]], "交换（a = 0）", 0)]:
    svg.circle(x, YS, 6, fill="violet", stroke="surface", width=1.5)
    tile2(x - 18, YS - 68, on)
    svg.text(x, YS + 26, lab, size=12, anchor="middle", color="violet")
xm = (X1 + X2) / 2
svg.circle(xm, YS, 5, fill="surface", stroke="green", width=2)
svg.text(xm, YS + 26, "J₂（a = ½）", size=12, anchor="middle", color="green")
a = 0.8
xa = X1 + (1 - a) * (X2 - X1)
svg.circle(xa, YS, 4.5, fill="blue")
svg.text(xa, YS - 12, "a = 0.8", size=12, anchor="middle", color="blue")
svg.text(24, 298, "维数 (2−1)² = 1，顶点 2! = 2 个", size=12, color="muted")
svg.text(24, 318, "线段上每一点都是两端的加权平均：", size=12, color="muted")
svg.text(24, 338, "[a, 1−a; 1−a, a] = a·I + (1−a)·交换", size=12, color="muted")

# ---- (b) B_3 projection
CX, CY, S = 540, 250, 112
svg.text(310, 30, "(b) n = 3：4 维的多面体，这里是一个线性投影", size=13, weight="600")
svg.text(310, 50, "6 个顶点 = 6 个置换矩阵；投影保持凸组合的权重，不保持距离", size=12, color="muted")


def xy(p):
    return CX + S * p[0], CY - S * p[1]


hexa = [xy(pts[pi]) for pi in [(1, 2, 3), (2, 1, 3), (3, 1, 2), (3, 2, 1), (2, 3, 1), (1, 3, 2)]]
# order the hexagon by angle
allp = sorted([xy(pts[pi]) for pi in perms], key=lambda q: math.atan2(q[1] - CY, q[0] - CX))
svg.polygon(allp, fill="violet", fill_opacity=0.05, stroke="rule", width=1.2)
svg.polygon([xy(pts[pi]) for pi in even], stroke="violet", width=1.6)
svg.polygon([xy(pts[pi]) for pi in odd], stroke="violet", width=1.2, dash="5 4")


def tile3(x, y, pi, cell=11):
    for i in range(3):
        for j in range(3):
            on = pi[i] - 1 == j
            svg.rect(x + j * cell + 0.5, y + i * cell + 0.5, cell - 1, cell - 1, fill="violet" if on else "grid",
                     fill_opacity=0.85 if on else 1, rx=2)


for pi in perms:
    x, y = xy(pts[pi])
    ux, uy = (x - CX) / S, (y - CY) / S
    svg.circle(x, y, 5.5, fill="violet", stroke="surface", width=1.5)
    tx, ty = x + ux * 30 - 16, y + uy * 34 - 16
    tile3(tx, ty, pi)
    lab = "(%d,%d,%d)" % pi + ("  I" if pi == (1, 2, 3) else "")
    svg.text(tx + 50 if ux > 0.3 else tx - 8 if ux < -0.3 else tx + 16, ty + 20 if abs(ux) > 0.3 else (ty - 6 if uy < 0 else ty + 50),
             lab, size=12, anchor="start" if ux > 0.3 else "end" if ux < -0.3 else "middle", color="violet")
# J_3 and the two decompositions
svg.circle(CX, CY, 5, fill="surface", stroke="green", width=2)
svg.text(CX + 10, CY + 18, "J₃", size=12, color="green", weight="600")
# circulant example
xd, yd = xy(pd)
svg.circle(xd, yd, 4.5, fill="orange")
svg.rect(xd + 6, yd - 19, 164, 18, fill="surface", fill_opacity=0.9, rx=3)
svg.text(xd + 9, yd - 6, "循环矩阵 0.5I + 0.3C + 0.2C²", size=12, color="orange")
# running example A with dashed spokes to its 4 permutations
xa, ya = xy(pa)
for pi, w in dec:
    x, y = xy(pts[pi])
    svg.line(xa, ya, x, y, color="blue", width=1, dash="3 3", opacity=0.8)
svg.circle(xa, ya, 5.5, fill="blue", stroke="surface", width=1.5)
svg.rect(xa - 70, ya + 7, 62, 18, fill="surface", fill_opacity=0.9, rx=3)
svg.text(xa - 10, ya + 20, "主算例 A", size=12, anchor="end", color="blue", weight="600")
svg.text(310, H - 50, "实线三角形：3 个循环移位 I、C、C²；虚线三角形：3 个对换。", size=12, color="muted")
svg.text(310, H - 31, "两个三角形的重心都是 J₃，所以 J₃ 至少有两种分解。", size=12, color="muted")
svg.text(310, H - 12, "蓝色虚线：主算例分解里用到的 4 个置换（权重 0.3、0.3、0.3、0.1）。", size=12, color="blue")

svg.save("m11-birkhoff-polytope.svg")
