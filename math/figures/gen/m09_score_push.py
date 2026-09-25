"""Figure M9.3: the score function pushes and pulls; the weight decides which arrow wins.

Toy model of M09, first step only: pi(5 | "2+3=") = 0.6, pi(6 | "2+3=") = 0.4. The score of the
logit z_5 is 1[a=5] - 0.6, i.e. +0.4 when "5" is sampled and -0.6 when "6" is sampled (the z_6
component is always the negative). Each arrow is pi(a) * weight(a) * score(a), so the arrows of one
row add up to an expectation (M9.7). Weights: 1; the action values Q = 0.85, 0.10; Q - 0.55 (the
advantages); Q + 10. Every length and label is computed here. Writes math/figures/m09-score-push.svg
"""

from m09_svg import SVG

PI = {"5": 0.6, "6": 0.4}
SCORE = {"5": 1 - 0.6, "6": 0 - 0.6}
Q = {"5": 0.7 * 1 + 0.3 * 0.5, "6": 0.5 * 0 + 0.5 * 0.2}
assert abs(Q["5"] - 0.85) < 1e-12 and abs(Q["6"] - 0.10) < 1e-12
V = PI["5"] * Q["5"] + PI["6"] * Q["6"]
assert abs(V - 0.55) < 1e-12

rows = [
    ("① 权重 = 1", "只看得分函数本身", "推与拉正好抵消", {a: 1.0 for a in PI}, 0.3),
    ("② 权重 = 回报", "Q(5) = 0.85，Q(6) = 0.10", "合力就是真梯度", dict(Q), 0.3),
    ("③ 权重 = 回报 − 0.55", "即优势 +0.30、−0.45", "两支都往右，合力不变", {a: Q[a] - V for a in PI}, 0.3),
    ("④ 权重 = 回报 + 10", "10.85、10.10", "两支巨大且反向，合力不变", {a: Q[a] + 10 for a in PI}, 3.0),
]

contrib = []
for _, _, _, w, _ in rows:
    c = {a: PI[a] * w[a] * SCORE[a] for a in PI}
    contrib.append(c)
sums = [c["5"] + c["6"] for c in contrib]
assert abs(sums[0]) < 1e-12
for s in sums[1:]:
    assert abs(s - 0.18) < 1e-12
assert abs(contrib[1]["5"] - 0.204) < 1e-12 and abs(contrib[1]["6"] + 0.024) < 1e-12
assert abs(contrib[2]["5"] - 0.072) < 1e-12 and abs(contrib[2]["6"] - 0.108) < 1e-12
assert abs(contrib[3]["5"] - 2.604) < 1e-9 and abs(contrib[3]["6"] + 2.424) < 1e-9

W, H = 720, 416
svg = SVG(W, H, "得分函数的推与拉：权重决定哪支箭头更长")
AX0, AX1 = 262, 572
AXC = (AX0 + AX1) / 2
TOP = 56
RH = 90

svg.text(16, 18, "玩具模型第一步（π(5) = 0.6，π(6) = 0.4），只画 logit z_{5} 的分量（z_{6} 的分量正好相反）",
         size=12.5, weight="600")
svg.text(16, 42, "每支箭头 = 选到这个动作的概率 × 权重 × 得分函数（选“5”时 +0.4，选“6”时 −0.6）；一行箭头之和 = 期望",
         size=12, color="muted")


def sgn(v: float) -> str:
    return ("+%.3f" % v) if v > 1e-12 else ("−%.3f" % -v if v < -1e-12 else "0")


for i, ((title, sub, note, w, half), c, s) in enumerate(zip(rows, contrib, sums)):
    top = TOP + i * RH
    if i:
        svg.line(16, top, W - 12, top, color="rule", width=1)
    scale = (AX1 - AXC) / half
    y5, y6, ys = top + 18, top + 36, top + 54
    yaxis = top + 64

    # left: what this row weights by
    svg.text(16, top + 22, title, size=12.5, weight="600")
    svg.text(16, top + 41, sub, size=12, color="muted")
    svg.text(16, top + 60, note, size=12, color="blue" if i else "ink")
    if i == 3:
        svg.text(16, top + 78, "（这一行横轴刻度是上面的 10 倍）", size=11, color="muted")

    # axis: zero line, ticks
    svg.line(AXC, top + 6, AXC, yaxis, color="muted", width=1, dash="3 3")
    svg.line(AX0, yaxis, AX1, yaxis, color="rule", width=1)
    for t in (-half, -half / 2, 0, half / 2, half):
        x = AXC + t * scale
        svg.line(x, yaxis - 3, x, yaxis + 3, color="muted", width=1)
    for t in (-half, 0, half):
        x = AXC + t * scale
        lab = "0" if t == 0 else (("+%g" % t) if t > 0 else ("−%g" % -t))
        svg.text(x, yaxis + 16, lab, size=11, anchor="middle", color="muted")

    # the two pushes and their sum
    for a, y, col in (("5", y5, "green"), ("6", y6, "red")):
        v = c[a]
        x2 = AXC + v * scale
        if abs(x2 - AXC) > 1:
            svg.arrow(AXC, y, x2, y, color=col, width=2.6)
        else:
            svg.circle(AXC, y, 3, fill=col)
        svg.text(588, y + 4, "选“%s”：%s" % (a, sgn(v)), size=12, color=col, weight="600")
    if abs(s) < 1e-12:
        svg.circle(AXC, ys, 4.5, fill="blue")
    else:
        svg.arrow(AXC, ys, AXC + s * scale, ys, color="blue", width=3.4)
    svg.text(588, ys + 4, "合力：%s" % sgn(s), size=12, color="blue", weight="600")

svg.save("m09-score-push.svg")
