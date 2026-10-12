"""Figure M11.1: a three-state weather Markov chain and its row-stochastic transition matrix.

P = [[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]  (sunny, cloudy, rainy; a teaching example)
Row sums are 1 (tomorrow must be something); column sums are 1.1, 1.05, 0.85 (not doubly stochastic).
The distributions p_t^T = p_0^T P^t are computed with numpy. Writes math/figures/m11-markov.svg
"""

import math

import numpy as np

from m11_svg import SVG, fmt

names = ["晴", "阴", "雨"]
P = np.array([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
assert np.allclose(P.sum(1), 1)
colsum = P.sum(0)

W, H = 760, 400
svg = SVG(W, H, "一个三状态的天气马尔可夫链：转移矩阵的每一行加起来是 1，每一列不必")

svg.text(24, 30, "(a) 状态图：箭头上的数 = 明天变成那个状态的概率", size=13, weight="600")
pos = {0: (232, 136), 1: (104, 322), 2: (360, 322)}
R = 30


def edge(i, j, bend=26):
    (x1, y1), (x2, y2) = pos[i], pos[j]
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux                       # left normal: i->j and j->i bend to opposite sides
    sx, sy = x1 + ux * R + nx * 8, y1 + uy * R + ny * 8
    ex, ey = x2 - ux * (R + 4) + nx * 8, y2 - uy * (R + 4) + ny * 8
    cx, cy = (x1 + x2) / 2 + nx * bend, (y1 + y2) / 2 + ny * bend
    svg.path("M%.1f %.1f Q%.1f %.1f %.1f %.1f" % (sx, sy, cx, cy, ex, ey), stroke="blue",
             width=1.2 + 5 * P[i, j], arrow=True)
    lx, ly = (x1 + x2) / 2 + nx * (bend + 14), (y1 + y2) / 2 + ny * (bend + 14)
    svg.text(lx, ly + 4, fmt(P[i, j]), size=12, anchor="middle", color="blue", weight="600")


for i in range(3):
    for j in range(3):
        if i != j:
            edge(i, j)
# self loops
loops = {0: (0, -1), 1: (-1, 0.35), 2: (1, 0.35)}
for i, (dx, dy) in loops.items():
    x, y = pos[i]
    L = math.hypot(dx, dy)
    dx, dy = dx / L, dy / L
    px, py = -dy, dx
    sx, sy = x + dx * R * 0.7 + px * R * 0.7, y + dy * R * 0.7 + py * R * 0.7
    ex, ey = x + dx * R * 0.7 - px * R * 0.7, y + dy * R * 0.7 - py * R * 0.7
    c1 = (x + dx * 2.2 * R + px * 1.3 * R, y + dy * 2.2 * R + py * 1.3 * R)
    c2 = (x + dx * 2.2 * R - px * 1.3 * R, y + dy * 2.2 * R - py * 1.3 * R)
    svg.path("M%.1f %.1f C%.1f %.1f %.1f %.1f %.1f %.1f" % (sx, sy, c1[0], c1[1], c2[0], c2[1], ex, ey),
             stroke="blue", width=1.2 + 5 * P[i, i], arrow=True)
    svg.text(x + dx * 2.35 * R + (6 if dx > 0.5 else -6 if dx < -0.5 else 0), y + dy * 2.35 * R + (4 if dy >= 0 else -2),
             fmt(P[i, i]), size=12, anchor="start" if dx > 0.5 else "end" if dx < -0.5 else "middle",
             color="blue", weight="600")
for i, (x, y) in pos.items():
    svg.circle(x, y, R, fill="surface", stroke="ink", width=1.6)
    svg.text(x, y + 6, names[i], size=16, anchor="middle", weight="600")

# ---- (b) the matrix with row and column sums
X0 = 466
svg.text(X0, 30, "(b) 转移矩阵 P：第 i 行 = 今天是 i 时明天的分布", size=13, weight="600")
CELL = 52
MX, MY = X0 + 44, 76
for j in range(3):
    svg.text(MX + j * CELL + CELL / 2, MY - 10, "明天" + names[j], size=12, anchor="middle", color="muted")
for i in range(3):
    svg.text(MX - 8, MY + i * CELL + CELL / 2 + 4, "今天" + names[i], size=12, anchor="end", color="muted")
    for j in range(3):
        x, y = MX + j * CELL, MY + i * CELL
        svg.rect(x + 2, y + 2, CELL - 4, CELL - 4, fill="blue", fill_opacity=0.08 + 0.5 * P[i, j], rx=5)
        svg.text(x + CELL / 2, y + CELL / 2 + 5, fmt(P[i, j]), size=14, anchor="middle")
    svg.text(MX + 3 * CELL + 10, MY + i * CELL + CELL / 2 + 5, "行和 1", size=12, color="green", weight="600")
for j in range(3):
    svg.text(MX + j * CELL + CELL / 2, MY + 3 * CELL + 20, fmt(colsum[j]), size=12, anchor="middle",
             color="red", weight="600")
svg.text(MX + 3 * CELL + 10, MY + 3 * CELL + 20, "列和 ≠ 1", size=12, color="red", weight="600")

# distributions
p = np.array([1.0, 0, 0])
ys = MY + 3 * CELL + 52
svg.text(X0, ys, "从“今天晴”出发，每天的分布（行向量乘 P）：", size=12, color="muted")
for t in range(0, 4):
    s = "p%s = (%s)" % ("₀₁₂₃"[t], ", ".join(("%.5f" % v).rstrip("0").rstrip(".") for v in p))
    svg.text(X0 + 8, ys + 22 * (t + 1), s + "，和为 1", size=12)
    p = p @ P

svg.save("m11-markov.svg")
