"""Figure 1.7 of the blackboard version of lesson 1: the five hand-computed steps of sec. 3.

Three tokens (The, cat, sat), d_k = 2, with the q/k/v table of the lesson text. Every cell is
computed here: (1) raw dot products QK^T, (2) divided by sqrt(2), (3) plus the causal mask,
(4) row-wise softmax with the row sums, (5) the outputs O = A V. The sat row (the one computed
alone first in the text) is framed in blue in every panel; masked cells are red.
Writes deepseek/figures/l01b-board-attention.svg
"""

import math

from l01_svg import SVG

TOK = ["The", "cat", "sat"]
Q = [(1, 1), (0, 2), (2, 1)]
K = [(0, 1), (2, 0), (1, 1)]
V = [(1, 0), (0, 1), (1, 1)]
N = 3

raw = [[Q[t][0] * K[j][0] + Q[t][1] * K[j][1] for j in range(N)] for t in range(N)]
S = [[raw[t][j] / math.sqrt(2) for j in range(N)] for t in range(N)]
A = []
for t in range(N):
    e = [math.exp(S[t][j]) if j <= t else 0.0 for j in range(N)]
    tot = sum(e)
    A.append([x / tot for x in e])
O = [[sum(A[t][j] * V[j][d] for j in range(N)) for d in range(2)] for t in range(N)]

assert raw == [[1, 2, 2], [2, 0, 2], [1, 4, 3]]
assert abs(A[2][1] - 0.6200) < 5e-5 and abs(A[2][0] - 0.0743) < 5e-5 and abs(A[1][0] - 0.8044) < 5e-5
assert abs(O[2][0] - 0.3800) < 5e-5 and abs(O[2][1] - 0.9257) < 5e-5

W, H = 760, 400
svg = SVG(W, H, "黑板上的五步：3 个 token、d_k = 2")

CW, CH = 56, 30          # cell width / height
LAB = 40                 # row-label column


def panel(x, y, title, vals, fmt_cell, masked=None, tint=None, cols=None, row_sums=None):
    """Draw a 3 x n matrix of cells with row labels; returns the x just right of the panel."""
    ncol = len(vals[0])
    svg.text(x, y - 34, title, size=13, weight="700")
    col_labels = cols or ["j=%d" % (j + 1) for j in range(ncol)]
    for j, cl in enumerate(col_labels):
        svg.text(x + LAB + j * CW + CW / 2, y - 10, cl, size=11, anchor="middle", color="muted")
    for t in range(N):
        cy = y + t * CH
        svg.text(x + LAB - 6, cy + CH / 2 + 4, TOK[t], size=12, anchor="end", color="muted")
        for j in range(ncol):
            cx = x + LAB + j * CW
            if masked and masked(t, j):
                svg.rect(cx + 1, cy + 1, CW - 2, CH - 2, fill="red", fill_opacity=0.12, rx=3)
                col = "red"
            elif tint:
                col_name, op = tint(t, j)
                svg.rect(cx + 1, cy + 1, CW - 2, CH - 2, fill=col_name, fill_opacity=op, rx=3)
                col = "ink"
            else:
                svg.rect(cx + 1, cy + 1, CW - 2, CH - 2, fill="grid", fill_opacity=0.9, rx=3)
                col = "ink"
            svg.text(cx + CW / 2, cy + CH / 2 + 4.5, fmt_cell(t, j), size=12.5, anchor="middle", color=col)
    right = x + LAB + ncol * CW
    if row_sums:
        svg.text(right + 30, y - 10, "行和", size=11, anchor="middle", color="muted")
        for t in range(N):
            svg.text(right + 30, y + t * CH + CH / 2 + 4.5, row_sums[t], size=12, anchor="middle", color="muted")
        right += 56
    # frame the sat row
    svg.rect(x + LAB - 2, y + 2 * CH - 1, ncol * CW + 4, CH + 2, stroke="blue", width=1.8, rx=4)
    return right


def neg_inf(t, j):
    return j > t


# row 1: three panels
Y1 = 70
x = 10
r1 = panel(x, Y1, "① 点积 QKᵀ", raw, lambda t, j: "%d" % raw[t][j])
svg.arrow(r1 + 6, Y1 + 45, r1 + 30, Y1 + 45, color="ink")
svg.text(r1 + 18, Y1 + 34, "÷√2", size=12, anchor="middle")
x2 = r1 + 36
r2 = panel(x2, Y1, "② 缩放 S = QKᵀ/√2", S, lambda t, j: "%.4f" % S[t][j])
svg.arrow(r2 + 6, Y1 + 45, r2 + 30, Y1 + 45, color="ink")
svg.text(r2 + 18, Y1 + 34, "+M", size=12, anchor="middle")
x3 = r2 + 36
r3 = panel(x3, Y1, "③ 加掩码 S + M", S,
           lambda t, j: "−∞" if j > t else "%.4f" % S[t][j], masked=neg_inf)
assert r3 < W - 4, r3

# row 2: softmax and output
Y2 = 250
svg.line(10, Y2 - 62, W - 10, Y2 - 62, color="rule", width=1)
svg.text(W - 14, Y2 - 70, "逐行 softmax ↓", size=12, anchor="end", color="muted")
x4 = 10
r4 = panel(x4, Y2, "④ 逐行 softmax：权重 A", A,
           lambda t, j: ("0" if j > t else ("1" if A[t][j] == 1 else "%.4f" % A[t][j])),
           masked=neg_inf, tint=lambda t, j: ("blue", 0.08 + 0.5 * A[t][j]),
           row_sums=["%.4f" % sum(A[t]) for t in range(N)])
svg.arrow(r4 + 6, Y2 + 45, r4 + 30, Y2 + 45, color="ink")
svg.text(r4 + 18, Y2 + 34, "×V", size=12, anchor="middle")
x5 = r4 + 36
r5 = panel(x5, Y2, "⑤ 输出 O = AV", O, lambda t, j: "%.4f" % O[t][j],
           tint=lambda t, j: ("green", 0.14), cols=["第 1 维", "第 2 维"])
# the values, for reference
vx = r5 + 24
svg.text(vx, Y2 - 34, "值 V（黑板上的表）", size=12, color="green", weight="600")
for t in range(N):
    svg.text(vx, Y2 + t * CH + CH / 2 + 4.5, "*v*_{%d} = (%d, %d)  %s" % (t + 1, V[t][0], V[t][1], TOK[t]),
             size=12, color="green")
assert vx + 150 < W, vx

# legend / footnote
fy = H - 18
svg.rect(12, fy - 12, 26, 16, stroke="blue", width=1.8, rx=3)
svg.text(46, fy, "sat 这一行（正文先单独算的那一行）", size=12)
svg.rect(292, fy - 12, 26, 16, fill="red", fill_opacity=0.12, rx=3)
svg.text(326, fy, "被掩码盖住的“未来”：softmax 之后权重恰好为 0", size=12)
svg.save("l01b-board-attention.svg")
