"""Figure M11.7: Sinkhorn-Knopp on the 3x3 worked example of M11 §5, step by step.

A = [[2, 1, 1], [1, 3, 1], [1, 4, 6]] (think of it as exp of a logit matrix). Each round: divide every column
by its sum (T_c), then every row by its sum (T_r), as in mHC eq. (9). Exact fractions are computed with
fractions.Fraction; entries with small denominators are shown as fractions, the rest as decimals.
Row / column sums equal to 1 are drawn green, the others red. Panel 6 is M^(20) in floating point: 20 rounds,
as in mHC; the column error is then 8.9e-16, so it agrees with the limit to every digit shown.
Writes math/figures/m11-sinkhorn-steps.svg
"""

from fractions import Fraction as Fr

import numpy as np

from m11_svg import SVG

A = [[Fr(v) for v in row] for row in [[2, 1, 1], [1, 3, 1], [1, 4, 6]]]


def Tc(M):
    cs = [sum(M[i][j] for i in range(3)) for j in range(3)]
    return [[M[i][j] / cs[j] for j in range(3)] for i in range(3)]


def Tr(M):
    rs = [sum(M[i]) for i in range(3)]
    return [[M[i][j] / rs[i] for j in range(3)] for i in range(3)]


S1 = Tc(A)
M1 = Tr(S1)
S2 = Tc(M1)
M2 = Tr(S2)
Mf = np.array(A, float)
for _ in range(20):
    Mf = Mf / Mf.sum(0, keepdims=True)
    Mf = Mf / Mf.sum(1, keepdims=True)
Minf = [[Fr(float(v)).limit_denominator(10 ** 9) for v in row] for row in Mf]
for name, M in [("A", A), ("Tc(A)", S1), ("M1", M1), ("Tc(M1)", S2), ("M2", M2)]:
    print(name, [[str(v) for v in row] for row in M],
          "rows", [str(sum(r)) for r in M], "cols", [str(sum(M[i][j] for i in range(3))) for j in range(3)])
print("limit", np.round(Mf, 5).tolist())


def show(v, force_dec=False):
    if not force_dec and v.denominator <= 30:
        return str(v.numerator) if v.denominator == 1 else "%d/%d" % (v.numerator, v.denominator)
    return "%.3f" % float(v)


W, H = 760, 520
svg = SVG(W, H, "Sinkhorn–Knopp 的一个 3×3 算例：每一步只能把一组和变成 1，另一组被打乱，但偏差一轮比一轮小")
CELL = 46


def block(x, y, M, title, dec=False, exact_one_tol=0):
    svg.text(x, y - 14, title, size=13, weight="600")
    for i in range(3):
        for j in range(3):
            cx, cy = x + j * CELL, y + i * CELL
            svg.rect(cx + 2, cy + 2, CELL - 4, CELL - 4, fill="blue", fill_opacity=0.06 + 0.5 * min(1, float(M[i][j])),
                     rx=4)
            svg.text(cx + CELL / 2, cy + CELL / 2 + 5, show(M[i][j], dec), size=12, anchor="middle")
    svg.path("M%s %s h-5 v%s h5" % (x + 1, y, 3 * CELL), stroke="ink", width=1.2)
    svg.path("M%s %s h5 v%s h-5" % (x + 3 * CELL - 1, y, 3 * CELL), stroke="ink", width=1.2)
    rows = [sum(M[i]) for i in range(3)]
    cols = [sum(M[i][j] for i in range(3)) for j in range(3)]
    for i, r in enumerate(rows):
        ok = abs(float(r) - 1) < 1e-9
        svg.text(x + 3 * CELL + 10, y + i * CELL + CELL / 2 + 5, show(r, dec and not ok), size=12,
                 color="green" if ok else "red", weight="600")
    for j, c in enumerate(cols):
        ok = abs(float(c) - 1) < 1e-9
        svg.text(x + j * CELL + CELL / 2, y + 3 * CELL + 18, show(c, dec and not ok), size=12, anchor="middle",
                 color="green" if ok else "red", weight="600")
    svg.text(x + 3 * CELL + 10, y + 3 * CELL + 18, "← 列和", size=11, color="muted")
    svg.text(x + 3 * CELL + 10, y - 14 + 0, "", size=11)


def arrow(x1, x2, y, label1, label2):
    svg.arrow(x1, y, x2, y, color="muted", width=1.6)
    svg.text((x1 + x2) / 2, y - 10, label1, size=13, anchor="middle", color="violet", weight="600")
    svg.text((x1 + x2) / 2, y + 20, label2, size=11, anchor="middle", color="muted")


Y1, Y2 = 70, 316
X = [34, 294, 548]
block(X[0], Y1, A, "① A（行和、列和都不是 1）")
arrow(X[0] + 3 * CELL + 42, X[1] - 12, Y1 + 70, "T_{c}", "除以列和")
block(X[1], Y1, S1, "② T_{c}(A)：列和变 1")
arrow(X[1] + 3 * CELL + 42, X[2] - 12, Y1 + 70, "T_{r}", "除以行和")
block(X[2], Y1, M1, "③ M^{(1)}：行和变 1")
# row 2
svg.path("M%s %s C %s %s %s %s %s %s" % (X[2] + 70, Y1 + 3 * CELL + 30, X[2] + 70, Y1 + 3 * CELL + 70,
                                         X[0] + 70, Y2 - 80, X[0] + 70, Y2 - 40), stroke="muted", width=1.4,
         dash="4 4", arrow=True)
svg.text((X[0] + X[2]) / 2 + 60, Y1 + 3 * CELL + 70, "第 2 轮：再做一次 T_{c}，再做一次 T_{r}", size=12, anchor="middle",
         color="muted")
block(X[0], Y2, S2, "④ T_{c}(M^{(1)})：列和又变 1", dec=False)
arrow(X[0] + 3 * CELL + 52, X[1] - 12, Y2 + 70, "T_{r}", "")
block(X[1], Y2, M2, "⑤ M^{(2)}（小数）", dec=True)
arrow(X[1] + 3 * CELL + 52, X[2] - 12, Y2 + 70, "……", "共 20 轮")
block(X[2], Y2, Minf, "⑥ 极限：行和、列和都是 1", dec=True)
svg.text(34, H - 14, "绿字 = 等于 1 的和，红字 = 不等于 1 的和。最大列和误差：第 1 轮后 1/6 ≈ 0.167，第 2 轮后 0.032，之后每轮约乘 0.18。",
         size=12, color="muted")
svg.save("m11-sinkhorn-steps.svg")
