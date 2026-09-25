"""Figure M11.5: Birkhoff decomposition of the running example, found greedily.

A = [[0.6, 0.4, 0], [0.1, 0.3, 0.6], [0.3, 0.3, 0.4]].
Greedy step: pick a permutation pi whose entries in the remainder R are all positive (a positive
diagonal), take theta = min_i R[i, pi(i)], subtract theta * P_pi. The order below is the one used in
the text; every weight and remainder is computed here with exact fractions.
Permutations are written in one-line notation (pi(1), pi(2), pi(3)): row i has its 1 in column pi(i).
Writes math/figures/m11-birkhoff-decomp.svg
"""

from fractions import Fraction as Fr

from m11_svg import SVG, fmt

A = [[Fr(6, 10), Fr(4, 10), Fr(0)], [Fr(1, 10), Fr(3, 10), Fr(6, 10)], [Fr(3, 10), Fr(3, 10), Fr(4, 10)]]
order = [(1, 2, 3), (2, 3, 1), (1, 3, 2), (2, 1, 3)]

R = [row[:] for row in A]
steps = []
for pi in order:
    theta = min(R[i][pi[i] - 1] for i in range(3))
    assert theta > 0
    before = [row[:] for row in R]
    for i in range(3):
        R[i][pi[i] - 1] -= theta
    steps.append((pi, theta, before, [row[:] for row in R]))
assert all(v == 0 for row in R for v in row)
assert sum(t for _, t, _, _ in steps) == 1
for pi, t, _, after in steps:
    print(pi, t, [[str(v) for v in row] for row in after])

W, H = 760, 476
svg = SVG(W, H, "把主算例 A 拆成 4 个置换矩阵的凸组合：每一步找一条全为正的对角线，减去它上面的最小值")


def num(v):
    return fmt(float(v)) if v else "0"


def value_matrix(x, y, M, cell=34, hl=None, hl_color="blue", mark=None):
    for i in range(3):
        for j in range(3):
            v = M[i][j]
            cx, cy = x + j * cell, y + i * cell
            if hl and (i, j) in hl:
                svg.rect(cx + 2, cy + 2, cell - 4, cell - 4, fill=hl_color, fill_opacity=0.16, stroke=hl_color,
                         width=1.6, rx=4)
            elif v:
                svg.rect(cx + 2, cy + 2, cell - 4, cell - 4, fill="grid", rx=4)
            svg.text(cx + cell / 2, cy + cell / 2 + 5, num(v), size=13, anchor="middle",
                     color="red" if mark and (i, j) in mark else ("ink" if v else "muted"),
                     weight="700" if mark and (i, j) in mark else None)
    svg.path("M%s %s h-5 v%s h5" % (x + 1, y, 3 * cell), stroke="ink", width=1.2)
    svg.path("M%s %s h5 v%s h-5" % (x + 3 * cell - 1, y, 3 * cell), stroke="ink", width=1.2)


def perm_tile(x, y, pi, cell=20):
    for i in range(3):
        for j in range(3):
            on = pi[i] - 1 == j
            svg.rect(x + j * cell + 1, y + i * cell + 1, cell - 2, cell - 2, fill="violet" if on else "grid",
                     fill_opacity=0.85 if on else 1, rx=3)
    svg.text(x + 1.5 * cell, y + 3 * cell + 16, "(%d,%d,%d)" % pi, size=12, anchor="middle", color="violet")


# ---- top row: the result
svg.text(24, 28, "(a) 结果：A 等于 4 个置换矩阵的加权和，权重非负、加起来是 1", size=13, weight="600")
TY = 50
value_matrix(40, TY, A, cell=32)
x = 40 + 3 * 32 + 16
svg.text(x, TY + 54, "=", size=20, anchor="middle")
x += 18
for k, (pi, t, _, _) in enumerate(steps):
    if k:
        svg.text(x + 4, TY + 54, "+", size=18, anchor="middle")
        x += 16
    svg.text(x + 20, TY + 54, fmt(float(t)), size=15, anchor="middle", color="blue", weight="700")
    x += 40
    perm_tile(x, TY + 18, pi)
    x += 70
svg.text(24, TY + 138, "紫格 = 1，灰格 = 0；每个小方块下面是置换的一行写法 (π(1), π(2), π(3))：第 i 行的 1 在第 π(i) 列。",
         size=12, color="muted")

# ---- bottom row: the greedy steps
svg.text(24, 226, "(b) 贪心过程：蓝框 = 选中的一条正对角线，红字 = 其中最小的那个数（它被减成 0）", size=13, weight="600")
BY = 262
for k, (pi, t, before, after) in enumerate(steps):
    x0 = 24 + k * 186
    diag = {(i, pi[i] - 1) for i in range(3)}
    mn = {(i, pi[i] - 1) for i in range(3) if before[i][pi[i] - 1] == t}
    svg.text(x0, BY - 10, "第 %d 步：余量 R%s" % (k + 1, "₀₁₂₃"[k]), size=12, color="ink", weight="600")
    value_matrix(x0 + 8, BY, before, cell=34, hl=diag, mark=mn)
    svg.text(x0, BY + 3 * 34 + 22, "对角线 (%d,%d,%d)：" % pi, size=12, color="blue")
    vals = "、".join(num(before[i][pi[i] - 1]) for i in range(3))
    svg.text(x0, BY + 3 * 34 + 40, "%s，最小 %s" % (vals, num(t)), size=12, color="muted")
    rs = sum(before[0])
    svg.text(x0, BY + 3 * 34 + 58, "每行、每列之和都是 %s" % num(rs), size=12, color="muted")
    if k < 3:
        svg.arrow(x0 + 8 + 3 * 34 + 12, BY + 51, x0 + 176, BY + 51, color="muted", width=1.4)
        svg.text(x0 + 8 + 3 * 34 + 36, BY + 42, "−%s·P" % num(t), size=12, anchor="middle", color="muted")
svg.text(24, H - 12, "第 4 步减完余量全为 0，权重 0.3 + 0.3 + 0.3 + 0.1 = 1。", size=12, color="muted")

svg.save("m11-birkhoff-decomp.svg")
