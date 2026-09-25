"""Figure 3.3: one ranking of K answers becomes C(K,2) pairwise comparisons.

Left : K = 4 answers ranked D > C > A > B -> 6 directed pairs (winner -> loser).
Right: C(K,2) for K = 4..9 (InstructGPT shows labelers 4 to 9 answers; sec. 3.5),
       and the batch bound 64 * C(9,2) = 2,304 comparisons (App. C.2).
Writes deepseek/figures/l03-rank-pairs.svg
"""
import math
from l03_svg import SVG, Axis

W, H = 720, 326
svg = SVG(W, H, "K 个回答的一次排序拆成 C(K,2) 对两两比较")

# ---------------- left: complete graph
pos = {"D": (80, 84), "C": (260, 84), "A": (260, 222), "B": (80, 222)}
rank = ["D", "C", "A", "B"]
svg.text(16, 26, "一次排序：D > C > A > B（K = 4）", size=14, weight="600")
for i, w in enumerate(rank):
    for l in rank[i + 1:]:
        (x1, y1), (x2, y2) = pos[w], pos[l]
        dx, dy = x2 - x1, y2 - y1
        n = math.hypot(dx, dy)
        sx, sy = x1 + dx / n * 26, y1 + dy / n * 26
        ex, ey = x2 - dx / n * 28, y2 - dy / n * 28
        svg.arrow(sx, sy, ex, ey, color="green", width=1.6)
for k, name in enumerate(rank):
    x, y = pos[name]
    svg.circle(x, y, 22, fill="surface", stroke="ink", width=1.6)
    svg.text(x, y + 6, name, size=17, anchor="middle", weight="700")
    svg.text(x, y + (-30 if y < 150 else 42), "第 %d 名" % (k + 1), size=12, anchor="middle", color="muted")
svg.text(16, 298, "箭头从胜者指向败者：C(4,2) = 6 对，", size=12.5)
svg.text(16, 316, "每个回答出现在 K − 1 = 3 对里。", size=12.5)

# ---------------- right: bar chart
Ks = list(range(4, 10))
vals = [math.comb(K, 2) for K in Ks]
ax = Axis(3.4, 9.6, 400, 700)
ay = Axis(0, 40, 262, 52)
svg.text(390, 26, "每个提示产生的比较对数 C(K,2) = K(K−1)/2", size=14, weight="600")
for v in (0, 10, 20, 30, 40):
    svg.line(ax(3.4), ay(v), ax(9.6), ay(v), color="grid", width=1)
    svg.text(ax(3.4) - 6, ay(v) + 4, str(v), size=11, anchor="end", color="muted")
svg.line(ax(3.4), ay(0), ax(9.6), ay(0), color="muted", width=1.1)
bw = 32
for K, v in zip(Ks, vals):
    svg.rect(ax(K) - bw / 2, ay(v), bw, ay(0) - ay(v), fill="green", fill_opacity=0.22, stroke="green", width=1.2)
    svg.text(ax(K), ay(v) - 6, str(v), size=12.5, anchor="middle", weight="600")
    svg.text(ax(K), ay(0) + 17, "K=%d" % K, size=12, anchor="middle", color="muted")
    svg.text(ax(K), ay(0) + 33, "%d 对/答" % (K - 1), size=11, anchor="middle", color="muted")
svg.text(390, 316, "一个 batch 64 个提示：384 至 2,304 对（64×6 至 64×36）", size=12.5)
svg.save("l03-rank-pairs.svg")
