"""Figure M1.4: one product, three readings.

A = [[1, 2, 0], [0, 1, 3]] (2x3), B = [[2, 1], [1, 0], [1, 4]] (3x2), C = AB = [[4, 1], [4, 12]].
(1) entry = row . column; (2) column of C = combination of A's columns; (3) C = sum of outer products.
All numbers are computed here and asserted against numpy.
Writes math/figures/m01-three-readings.svg
"""

import numpy as np

from m01_svg import SVG

A = np.array([[1, 2, 0], [0, 1, 3]])
B = np.array([[2, 1], [1, 0], [1, 4]])
C = A @ B
outer = [np.outer(A[:, k], B[k]) for k in range(3)]
assert (sum(outer) == C).all()
assert (C == np.array([[4, 1], [4, 12]])).all()

W, H = 720, 520
cell = 30
svg = SVG(W, H, "同一个矩阵乘积的三种读法：批量点积、列的组合、外积之和")


def mat(x, y, M, **kw):
    return svg.matrix(x, y, [[int(v) for v in row] for row in M], cell=cell, size=14, **kw)


def op(x, y, s, size=18):
    svg.text(x, y, s, size=size, anchor="middle", color="muted")


# ---------------------------------------------------------------- panel 1
y0 = 30
svg.text(16, y0, "① 批量点积", size=15, weight="700", color="blue")
svg.text(118, y0, "*C* 的每一格 = *A* 的一行 与 *B* 的一列 的点积", size=13, color="muted")
by = y0 + 22
mat(40, by + 15, A, highlight={(1, j) for j in range(3)}, hl_color="blue")
svg.text(85, by + 97, "*A*  [2, 3]", size=12, color="muted", anchor="middle")
op(152, by + 52, "×")
mat(178, by, B, highlight={(i, 0) for i in range(3)}, hl_color="violet")
svg.text(208, by + 112, "*B*  [3, 2]", size=12, color="muted", anchor="middle")
op(264, by + 52, "=")
mat(290, by + 15, C, highlight={(1, 0)}, hl_color="green")
svg.text(320, by + 97, "*C*  [2, 2]", size=12, color="muted", anchor="middle")
tx = 400
svg.text(tx, by + 28, "*C*_{21} = (第 2 行) · (第 1 列)", size=14)
svg.text(tx, by + 54, "     = (0, 1, 3) · (2, 1, 1)", size=14)
svg.text(tx, by + 80, "     = 0×2 + 1×1 + 3×1 = %d" % C[1, 0], size=14, color="green")
svg.line(16, 172, W - 16, 172, color="rule", width=1)

# ---------------------------------------------------------------- panel 2
y0 = 200
svg.text(16, y0, "② 列的组合", size=15, weight="700", color="blue")
svg.text(118, y0, "*C* 的第 1 列 = *A* 的三列，按 *B* 第 1 列的系数 (2, 1, 1) 加权相加", size=13, color="muted")
cy = y0 + 28
x = 36
for k in range(3):
    coef = int(B[k, 0])
    svg.text(x + 8, cy + 37, "%d" % coef, size=17, color="violet", anchor="middle", weight="700")
    svg.text(x + 26, cy + 36, "×", size=15, color="muted", anchor="middle")
    mat(x + 44, cy, A[:, [k]], fill="blue", fill_opacity=0.10)
    svg.text(x + 59, cy + 82, "*a*_{%d}" % (k + 1), size=13, color="blue", anchor="middle")
    op(x + 92, cy + 37, "+" if k < 2 else "=")
    x += 106
mat(x + 4, cy, C[:, [0]], highlight={(0, 0), (1, 0)}, hl_color="green")
svg.text(x + 19, cy + 82, "*c*_{1}", size=13, color="green", anchor="middle")
tx = 420
svg.text(tx, cy + 18, "2·(1, 0) + 1·(2, 1) + 1·(0, 3)", size=14)
svg.text(tx, cy + 44, "= (2+2+0, 0+1+3) = (4, 4)", size=14, color="green")
svg.text(tx, cy + 70, "第 2 列同理：1·*a*_{1} + 0·*a*_{2} + 4·*a*_{3} = (1, 12)", size=13, color="muted")
svg.line(16, 336, W - 16, 336, color="rule", width=1)

# ---------------------------------------------------------------- panel 3
y0 = 364
svg.text(16, y0, "③ 外积之和", size=15, weight="700", color="blue")
svg.text(118, y0, "*C* = Σ_{k} (*A* 的第 k 列)(*B* 的第 k 行)，每一项都是一个 [2, 2] 的“秩 1”矩阵", size=13, color="muted")
oy = y0 + 30
x = 36
for k in range(3):
    mat(x, oy, outer[k], fill="orange", fill_opacity=0.08)
    col = "(%s)" % ", ".join(str(int(v)) for v in A[:, k])
    row = "(%s)" % ", ".join(str(int(v)) for v in B[k])
    svg.text(x + 30, oy + 86, "*a*_{%d}*b*_{%d}^{⊤}" % (k + 1, k + 1), size=13, color="orange", anchor="middle")
    svg.text(x + 30, oy + 110, "列%s × 行%s" % (col, row), size=12, color="muted", anchor="middle")
    op(x + 92, oy + 36, "+" if k < 2 else "=")
    x += 122
mat(x - 4, oy, C, highlight={(0, 0), (0, 1), (1, 0), (1, 1)}, hl_color="green")
svg.text(x + 26, oy + 86, "*C*", size=13, color="green", anchor="middle")
tx = 500
svg.text(tx, oy + 20, "左上角：2 + 2 + 0 = 4", size=13, color="muted")
svg.text(tx, oy + 42, "右下角：0 + 0 + 12 = 12", size=13, color="muted")
svg.text(tx, oy + 64, "三项相加，与 ①② 的结果相同", size=13, color="muted")

svg.save("m01-three-readings.svg")
