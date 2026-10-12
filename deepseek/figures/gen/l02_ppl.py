"""Figure 2.2: cross-entropy loss (nats per token) versus perplexity PPL = e^L.

Marked points (all recomputed in Python, see lesson-02-doc.md section 4.4):
  worked example, token probabilities 0.5, 0.25, 0.1, 0.8 -> L = 1.1513 nats, PPL 3.162
  GPT-1 on BooksCorpus: token-level perplexity 18.4 -> L = ln 18.4 = 2.912 nats
  uniform guess over GPT-2's 50,257 tokens -> L = ln 50257 = 10.825 nats
Top axis: the same loss in bits (L / ln 2).
Writes deepseek/figures/l02-ppl.svg
"""

import math

import numpy as np

from l02_svg import SVG, LogAxis

W, H = 700, 372
X0, X1, Y0, Y1 = 78, 610, 62, 318
ax = LogAxis(0, 11.5, X0, X1, log=False)
ay = LogAxis(1, 1e5, Y1, Y0, log=True)

svg = SVG(W, H, "交叉熵损失与困惑度：PPL = e 的 L 次方")

# grid and ticks
for e in range(0, 6):
    y = ay(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    lab = {0: "1", 1: "10", 2: "100", 3: "1,000", 4: "10^{4}", 5: "10^{5}"}[e]
    svg.text(X0 - 8, y + 4, lab, size=11, anchor="end", color="muted")
for v in range(0, 12):
    x = ax(v)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 17, str(v), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X1, Y0, color="muted", width=1.0)
# top axis: bits
for b in range(0, 17, 2):
    L = b * math.log(2)
    if L > 11.5:
        break
    x = ax(L)
    svg.line(x, Y0 - 5, x, Y0, color="muted", width=1)
    svg.text(x, Y0 - 9, str(b), size=11, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, 24, "同一损失换成 bits（除以 ln 2 ≈ 0.693）", size=12, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, Y1 + 40, "平均交叉熵损失 L（nats / token）", size=12, anchor="middle", color="muted")
svg.text(20, (Y0 + Y1) / 2, "困惑度 PPL（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

# the curve
Ls = np.linspace(0, 11.5, 200)
d = "M" + " L".join("%.1f %.1f" % (ax(L), ay(math.exp(L))) for L in Ls)
svg.path(d, stroke="blue", width=2.6)
svg.text(ax(6.2), ay(math.exp(6.2)) - 12, "PPL = e^{L}", size=13, color="blue", weight="600", anchor="end")

pts = [
    (-math.log(0.5 * 0.25 * 0.1 * 0.8) / 4, "例题（四个 token）：L = 1.151，PPL = 3.16", None, "green", "right1"),
    (math.log(18.4), "GPT-1 在 BooksCorpus", "PPL 18.4 → L = 2.912", "violet", "leftup"),
    (math.log(50257), "均匀瞎猜 50,257 个 token", "L = ln 50,257 = 10.825", "red", "left"),
]
for L, t1, t2, col, side in pts:
    x, y = ax(L), ay(math.exp(L))
    svg.line(X0, y, x, y, color=col, width=1, dash="3 3")
    if side == "left":
        svg.line(x, y, x, Y1, color=col, width=1, dash="3 3")
    svg.circle(x, y, 4.5, fill=col)
    if side == "right1":
        svg.text(x + 10, y + 5, t1, size=12, color=col, weight="600")
    elif side == "leftup":
        svg.text(x - 8, y - 26, t1, size=12, color=col, weight="600", anchor="end")
        svg.text(x - 8, y - 10, t2, size=12, color="muted", anchor="end")
    else:
        # one line just below the dashed guide, ending well left of the point: right next to the point
        # the curve would cut through it, and above the guide it would sit on the top frame
        svg.text(x - 72, y + 17, "*%s*：%s" % (t1, t2), size=12, color=col, anchor="end")

# a reading aid: one nat less = PPL / e
x3, x4 = ax(8), ax(9)
svg.text(ax(8.5), ay(math.exp(4.1)), "损失每降 1 nat，", size=12, anchor="middle", color="ink")
svg.text(ax(8.5), ay(math.exp(4.1)) + 16, "困惑度除以 e ≈ 2.72", size=12, anchor="middle", color="ink")

svg.save("l02-ppl.svg")
