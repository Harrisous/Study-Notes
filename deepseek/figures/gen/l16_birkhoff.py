"""Figure 16.7: the Birkhoff polytope, drawn honestly.

B_n (n x n doubly stochastic matrices) has dimension (n-1)^2 and n! vertices (the permutation matrices;
Birkhoff-von Neumann). B_2 is a segment; B_3 is 4-dimensional, so it cannot be drawn -- panel (b) shows an
exact 2-D slice, the circulant matrices a I + b C + c C^2 (a + b + c = 1), a triangle with centroid J_3.
Panel (c) computes the Sinkhorn-Knopp limit of A = [[4,1,1],[1,4,1],[2,2,2]] (research notes N23) and a
greedy Birkhoff decomposition of it (notes N24 list one such decomposition; decompositions are not unique).
Writes deepseek/figures/l16-birkhoff.svg and prints the decomposition.
"""

import itertools

import numpy as np

from l16_svg import SVG

# ---------------------------------------------------------------- numbers for panel (c)
A = np.array([[4.0, 1, 1], [1, 4, 1], [2, 2, 2]])
M = A.copy()
for _ in range(200):
    M = M / M.sum(0, keepdims=True)
    M = M / M.sum(1, keepdims=True)
LIMIT = M.copy()
R = M.copy()
decomp = []
perms = list(itertools.permutations(range(3)))
while R.max() > 1e-9:
    best = max(perms, key=lambda p: min(R[i, p[i]] for i in range(3)))
    w = min(R[i, best[i]] for i in range(3))
    if w <= 1e-9:
        break
    for i in range(3):
        R[i, best[i]] -= w
    decomp.append((w, best))
decomp.sort(key=lambda t: -t[0])
print("limit:\n", np.round(LIMIT, 5))
for w, p in decomp:
    print("%.5f  perm %s" % (w, p))
print("sum of weights %.6f" % sum(w for w, _ in decomp))

W, H = 760, 420
svg = SVG(W, H, "Birkhoff 多面体：n = 2 是线段，n = 3 画不出来，只画一个切片")


def glyph(x, y, perm_or_mat, cell=11, color="violet", label=None):
    """tiny matrix glyph: filled square where the entry is 1 (permutation) or shaded by value."""
    n = len(perm_or_mat)
    for i in range(n):
        for j in range(n):
            if isinstance(perm_or_mat[0], (int, np.integer)):
                v = 1.0 if perm_or_mat[i] == j else 0.0
            else:
                v = float(perm_or_mat[i][j])
            svg.rect(x + j * cell, y + i * cell, cell - 1, cell - 1, fill=color if v > 0 else "grid",
                     fill_opacity=(0.2 + 0.8 * v) if v > 0 else 1.0, rx=1.5)
    if label:
        svg.text(x + n * cell / 2, y + n * cell + 15, label, size=12, anchor="middle")


# ---------------------------------------------------------------- (a) B_2
svg.text(20, 26, "(a) 𝓑_{2}：一条线段", size=14, weight="700")
X0, X1, YA = 50, 210, 150
svg.line(X0, YA, X1, YA, color="green", width=3)
for x in (X0, X1):
    svg.circle(x, YA, 5, fill="violet")
svg.circle((X0 + X1) / 2, YA, 5, fill="green")
glyph(X0 - 11, YA - 60, [0, 1], cell=11, label=None)
svg.text(X0, YA + 24, "I（a = 1）", size=12, anchor="middle")
glyph(X1 - 11, YA - 60, [1, 0], cell=11)
svg.text(X1, YA + 24, "交换（a = 0）", size=12, anchor="middle")
glyph((X0 + X1) / 2 - 11, YA - 60, [[0.5, 0.5], [0.5, 0.5]], cell=11, color="green")
svg.text((X0 + X1) / 2, YA + 24, "J_{2}（a = ½）", size=12, anchor="middle", color="green")
svg.text(20, 214, "每个 2×2 双随机矩阵都是", size=12)
svg.text(20, 234, "[[a, 1−a], [1−a, a]]，0 ≤ a ≤ 1", size=12)
svg.text(20, 258, "1 个自由度 = (2−1)²；2! = 2 个顶点", size=12, color="muted")

# ---------------------------------------------------------------- (b) B_3 circulant slice
svg.line(262, 40, 262, 400, color="rule", width=1)
svg.text(280, 26, "(b) 𝓑_{3} 的一个二维切片", size=14, weight="700")
TI, TC, TC2 = (318, 312), (478, 312), (398, 150)
svg.polygon([TI, TC, TC2], fill="green", fill_opacity=0.10, stroke="green", width=2)
for (x, y), p, lab, gx, gy, lx, anchor in ((TI, [0, 1, 2], "I", -15, 12, -21, "end"),
                                           (TC, [1, 2, 0], "C", -15, 12, 21, "start"),
                                           (TC2, [2, 0, 1], "C²", 14, -15, 50, "start")):
    svg.circle(x, y, 5, fill="violet")
    glyph(x + gx, y + gy, p, cell=10)
    svg.text(x + lx, y + gy + 20, lab, size=13, anchor=anchor, color="violet", weight="600")
G = ((TI[0] + TC[0] + TC2[0]) / 3, (TI[1] + TC[1] + TC2[1]) / 3)
svg.circle(G[0], G[1], 5, fill="green")
svg.text(G[0] + 9, G[1] + 4, "J_{3}", size=12, color="green", weight="600")
# a sample point 0.6 I + 0.3 C + 0.1 C^2
a, b, c_ = 0.6, 0.3, 0.1
P = (a * TI[0] + b * TC[0] + c_ * TC2[0], a * TI[1] + b * TC[1] + c_ * TC2[1])
svg.circle(P[0], P[1], 4, fill="blue")
svg.text(P[0] + 8, P[1] - 6, "0.6I+0.3C+0.1C²", size=11, color="blue")
svg.text(280, 384, "C = 循环移位；切片内的点 = aI + bC + cC²", size=12)
svg.text(280, 404, "𝓑_{3} 本身是 4 维，有 3! = 6 个顶点", size=12, color="muted")

# ---------------------------------------------------------------- (c) decomposition bars
svg.line(512, 40, 512, 400, color="rule", width=1)
svg.text(528, 26, "(c) 一个双随机矩阵 = 置换的凸组合", size=14, weight="700")
glyph(540, 46, LIMIT, cell=15, color="green")
svg.text(596, 60, "SK(A) 的极限", size=12, color="green", weight="600")
svg.text(596, 78, "A = [[4,1,1],[1,4,1],[2,2,2]]", size=11, color="muted")
svg.text(596, 94, "（§5 的算例）", size=11, color="muted")
BY = 124
BW = 270  # px per unit weight
for k, (w, p) in enumerate(decomp):
    y = BY + k * 44
    glyph(532, y, list(p), cell=10)
    svg.rect(572, y + 6, w * BW, 18, fill="violet", fill_opacity=0.55, rx=3)
    svg.text(572 + w * BW + 6, y + 20, "%.5f" % w, size=12)
svg.text(528, BY + len(decomp) * 44 + 12, "权重之和 = %.4f（分解不唯一）" % sum(w for w, _ in decomp), size=12,
         color="muted")

svg.save("l16-birkhoff.svg")
