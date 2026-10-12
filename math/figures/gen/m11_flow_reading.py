"""Figure M11.2: one doubly stochastic matrix, two readings (columns distribute, rows average).

A = [[0.6, 0.4, 0.0], [0.1, 0.3, 0.6], [0.3, 0.3, 0.4]]  (the running example of M11 §2-§4)
Row i = where new stream i takes its goods from (weights sum to 1: a weighted average).
Column j = how old stream j is split among the new streams (sums to 1: nothing created or lost).
Edge widths are proportional to a_ij. Writes math/figures/m11-flow-reading.svg
"""

from fractions import Fraction

from m11_svg import SVG, fmt

A = [[0.6, 0.4, 0.0], [0.1, 0.3, 0.6], [0.3, 0.3, 0.4]]
n = 3
rows = [sum(r) for r in A]
cols = [sum(A[i][j] for i in range(n)) for j in range(n)]
assert all(abs(s - 1) < 1e-12 for s in rows + cols)

W, H = 760, 400
svg = SVG(W, H, "一个双随机矩阵的两种读法：按列是“分出去”，按行是“加权平均”")


def num(v):
    return fmt(v) if v else "0"


# ---- (a) the matrix with its sums
svg.text(30, 34, "(a) 矩阵 A", size=13, weight="600")
svg.text(30, 54, "第 i 行第 j 列：旧流 j 分给新流 i 的份额", size=12, color="muted")
CELL = 50
MX, MY = 70, 92
for j in range(n):
    svg.text(MX + j * CELL + CELL / 2, MY - 12, "旧 %d" % (j + 1), size=12, anchor="middle", color="violet")
for i in range(n):
    svg.text(MX - 10, MY + i * CELL + CELL / 2 + 4, "新 %d" % (i + 1), size=12, anchor="end", color="blue")
    for j in range(n):
        x, y = MX + j * CELL, MY + i * CELL
        a = A[i][j]
        if a > 0:
            svg.rect(x + 2, y + 2, CELL - 4, CELL - 4, fill="blue", fill_opacity=0.08 + 0.55 * a, rx=5)
        else:
            svg.rect(x + 2, y + 2, CELL - 4, CELL - 4, fill="grid", rx=5)
        svg.text(x + CELL / 2, y + CELL / 2 + 5, num(a), size=14, anchor="middle",
                 color="surface" if a >= 0.5 else "ink", weight="600" if a >= 0.5 else None)
    # row sum
    svg.text(MX + n * CELL + 12, MY + i * CELL + CELL / 2 + 5, "行和 %s" % fmt(rows[i]), size=12, color="blue")
for j in range(n):
    svg.text(MX + j * CELL + CELL / 2, MY + n * CELL + 20, fmt(cols[j]), size=12, anchor="middle", color="violet")
svg.text(MX + n * CELL + 12, MY + n * CELL + 20, "← 列和", size=12, color="violet")
svg.text(30, MY + n * CELL + 58, "非负；每行、每列加起来都是 1", size=12, color="muted")
svg.text(30, MY + n * CELL + 78, "（颜色越深，份额越大；灰格为 0）", size=12, color="muted")


# ---- bipartite helper
def bipartite(x0, title, sub, highlight, color, label_t, note_lines):
    svg.text(x0, 34, title, size=13, weight="600")
    svg.text(x0, 54, sub, size=12, color="muted")
    ox, nx = x0 + 10, x0 + 160
    ys = [110, 190, 270]
    BW, BH = 54, 30
    # faint background edges
    for i in range(n):
        for j in range(n):
            if A[i][j] > 0 and (i, j) not in highlight:
                svg.line(ox + BW, ys[j], nx, ys[i], color="rule", width=1 + 10 * A[i][j], opacity=0.9)
    for (i, j) in highlight:
        a = A[i][j]
        svg.line(ox + BW, ys[j], nx, ys[i], color=color, width=1.5 + 12 * a, opacity=0.55)
        x1, y1, x2, y2 = ox + BW, ys[j], nx, ys[i]
        dx, dy = x2 - x1, y2 - y1
        ln = (dx * dx + dy * dy) ** 0.5
        nx_, ny_ = dy / ln, -dx / ln          # unit normal
        if ny_ > 0:
            nx_, ny_ = -nx_, -ny_               # always offset upwards
        off = 8 + 6 * a
        lx = x1 + dx * label_t + nx_ * off
        ly = y1 + dy * label_t + ny_ * off + 4
        svg.text(lx, ly, fmt(a), size=13, anchor="middle", color=color, weight="700")
    for k in range(n):
        svg.rect(ox, ys[k] - BH / 2, BW, BH, fill="surface", stroke="violet", width=1.4, rx=6)
        svg.text(ox + BW / 2, ys[k] + 5, "旧 %d" % (k + 1), size=13, anchor="middle", color="violet")
        svg.rect(nx, ys[k] - BH / 2, BW, BH, fill="surface", stroke="blue", width=1.4, rx=6)
        svg.text(nx + BW / 2, ys[k] + 5, "新 %d" % (k + 1), size=13, anchor="middle", color="blue")
    for t, line in enumerate(note_lines):
        svg.text(x0, 320 + 20 * t, line, size=12, color=color if t == 0 else "muted")


bipartite(296, "(b) 按列读：旧流 1 分给谁", "第 1 列 (0.6, 0.1, 0.3)",
          {(0, 0), (1, 0), (2, 0)}, "violet", 0.72,
          ["0.6 + 0.1 + 0.3 = 1（列和）", "旧流 1 的货全部分出去，", "不多出、不丢失"])
bipartite(530, "(c) 按行读：新流 2 从哪来", "第 2 行 (0.1, 0.3, 0.6)",
          {(1, 0), (1, 1), (1, 2)}, "blue", 0.34,
          ["0.1 + 0.3 + 0.6 = 1（行和）", "新流 2 是三条旧流的", "加权平均"])

svg.save("m11-flow-reading.svg")
