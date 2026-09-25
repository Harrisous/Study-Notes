"""Figure M11.8: the three cases of the Sinkhorn-Knopp (1967) theorem, on three small matrices.

(a) positive 3x3 (the worked example [[2,1,1],[1,3,1],[1,4,6]]): total support -> limit D1 A D2.
(b) [[1,1],[0,1]]: support (main diagonal) but not total support: the (1,2) entry lies on no positive
    diagonal; the iteration converges to I, slowly (error 1/(2t+1)); the limit is not of the form D1 A D2.
(c) [[1,1,1],[1,0,0],[1,0,0]]: no positive diagonal (rows 2 and 3 can only use column 1): the half-steps
    approach two different matrices, neither doubly stochastic.
Positive diagonals are enumerated with itertools; the limits are computed with numpy.
Writes math/figures/m11-support.svg
"""

import itertools

import numpy as np

from m11_svg import SVG, fmt

cases = [
    np.array([[2, 1, 1], [1, 3, 1], [1, 4, 6]], float),
    np.array([[1, 1], [0, 1]], float),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], float),
]


def pos_diagonals(A):
    n = len(A)
    return [p for p in itertools.permutations(range(n)) if all(A[i, p[i]] > 0 for i in range(n))]


def iterate(A, T):
    M = A.copy()
    for _ in range(T):
        Mc = M / M.sum(0, keepdims=True)
        M = Mc / Mc.sum(1, keepdims=True)
    return Mc, M


for k, A in enumerate(cases):
    print(k, "positive diagonals:", pos_diagonals(A))
    Mc, M = iterate(A, 200)
    print("   after col step:", np.round(Mc, 4).tolist(), " after row step:", np.round(M, 4).tolist())

W, H = 760, 418
svg = SVG(W, H, "Sinkhorn–Knopp 定理的三种情形：完全支撑、有支撑无完全支撑、没有支撑")


def grid(x, y, A, cell, diag=None, bad=None):
    n = len(A)
    for i in range(n):
        for j in range(n):
            cx, cy = x + j * cell, y + i * cell
            if A[i, j] > 0:
                svg.rect(cx + 2, cy + 2, cell - 4, cell - 4, fill="blue", fill_opacity=0.22, rx=4)
            else:
                svg.rect(cx + 2, cy + 2, cell - 4, cell - 4, fill="grid", rx=4)
            if bad and (i, j) in bad:
                svg.rect(cx + 2, cy + 2, cell - 4, cell - 4, stroke="red", width=2.2, rx=4)
            svg.text(cx + cell / 2, cy + cell / 2 + 5, fmt(A[i, j]), size=13, anchor="middle",
                     color="ink" if A[i, j] > 0 else "muted")
    if diag:
        pts = [(x + diag[i] * cell + cell / 2, y + i * cell + cell / 2) for i in range(n)]
        for i in range(n):
            svg.circle(pts[i][0], pts[i][1], cell / 2 - 5, stroke="green", width=2.2)


def small(x, y, M, cell=30, color="ink"):
    n = len(M)
    for i in range(n):
        for j in range(n):
            svg.text(x + j * cell + cell / 2, y + i * 18, fmt(M[i][j]) if abs(M[i][j]) > 5e-3 else "0", size=12,
                     anchor="middle", color=color)
    svg.path("M%s %s h-4 v%s h4" % (x + 2, y - 13, 18 * n), stroke=color, width=1)
    svg.path("M%s %s h4 v%s h-4" % (x + n * cell - 2, y - 13, 18 * n), stroke=color, width=1)


# (a)
X = 24
svg.text(X, 28, "(a) 元素全为正：完全支撑", size=13, weight="600", color="green")
grid(X + 20, 52, cases[0], 44, diag=(0, 1, 2))
svg.text(X, 208, "6 条对角线全为正（绿圈是其中一条）", size=12, color="muted")
svg.text(X, 228, "→ 收敛到唯一的 D₁AD₂，几何级数地快", size=12, color="green")
Mc, M = iterate(cases[0], 200)
svg.text(X, 256, "极限（两位小数）：", size=12, color="muted")
small(X + 10, 280, np.round(M, 2), cell=40)

# (b)
X = 280
svg.text(X, 28, "(b) 有支撑，无完全支撑", size=13, weight="600", color="violet")
grid(X + 40, 52, cases[1], 50, diag=(0, 1), bad={(0, 1)})
svg.text(X, 208, "正对角线只有主对角线（绿圈）；", size=12, color="muted")
svg.text(X, 228, "红框的 1 不在任何正对角线上", size=12, color="red")
svg.text(X, 248, "→ 它被压向 0，极限是 I，不是 D₁AD₂", size=12, color="violet")
svg.text(X, 268, "→ 慢：第 t 轮后还剩 1/(2t+1)", size=12, color="violet")
small(X + 30, 296, [[1, 0], [0, 1]], cell=40, color="violet")
svg.text(X + 120, 300, "（t → ∞）", size=12, color="muted")

# (c)
X = 520
svg.text(X, 28, "(c) 没有正对角线：没有支撑", size=13, weight="600", color="red")
grid(X + 20, 52, cases[2], 44)
svg.text(X, 208, "第 2、3 行都只能用第 1 列，", size=12, color="muted")
svg.text(X, 228, "凑不出一条全为正的对角线", size=12, color="muted")
svg.text(X, 248, "→ 永远到不了双随机矩阵：", size=12, color="red")
Mc, M = iterate(cases[2], 200)
svg.text(X, 270, "行步之后趋向", size=12, color="muted")
small(X + 110, 270, np.round(M, 2), cell=34, color="red")
svg.text(X, 340, "列步之后趋向", size=12, color="muted")
small(X + 110, 340, np.round(Mc, 2), cell=34, color="red")

svg.text(24, H - 8, "蓝格 = 正元素，灰格 = 0。定理与术语据 Sinkhorn 与 Knopp（1967）；极限由生成脚本迭代 200 轮得到。",
         size=11, color="muted")
svg.save("m11-support.svg")
