"""Figure 3.9 of the blackboard version of lesson 3: a baseline keeps the mean gradient and shrinks the spread.

One prompt, two answers. pi(a1) = sigmoid(theta) = 0.7, pi(a2) = 0.3; rewards R = 1.0 and 0.8.
Score directions: d/dtheta log pi(a1) = 1 - 0.7 = 0.3, d/dtheta log pi(a2) = -0.7.
One sample's gradient is (R - b) * direction, with b = 0 (top row) or b = E[R] = 0.94 (bottom row).
Circle areas are proportional to the sampling probabilities. Means and variances are computed here with
exact fractions (they are the numbers of the worked example in section 5 of the lesson).
Writes deepseek/figures/l03b-baseline-board.svg
"""

from fractions import Fraction as F

from l03_svg import SVG, Axis

P = F(7, 10)
REWARD = (F(1), F(4, 5))
DIRECTION = (1 - P, -P)
PROB = (P, 1 - P)


def row(b):
    s = [(REWARD[i] - b) * DIRECTION[i] for i in range(2)]
    mean = sum(PROB[i] * s[i] for i in range(2))
    var = sum(PROB[i] * s[i] ** 2 for i in range(2)) - mean ** 2
    return s, mean, var


BASE = P * REWARD[0] + (1 - P) * REWARD[1]
s0, m0, v0 = row(F(0))
s1, m1, v1 = row(BASE)
assert BASE == F(94, 100)
assert s0 == [F(3, 10), F(-56, 100)] and s1 == [F(18, 1000), F(98, 1000)]
assert m0 == m1 == F(42, 1000)
assert v0 == F(155316, 10 ** 6) and v1 == F(1344, 10 ** 6)
assert round(float(v0 / v1)) == 116


def num(x):
    s = "%+.3f" % float(x)
    s = s.rstrip("0").rstrip(".")
    return s.replace("-", "−")


W, H = 740, 372
svg = SVG(W, H, "黑板上的两个回答：减去基线，平均不变，样本却不再东倒西歪")
ax = Axis(-0.7, 0.45, 40, 500)

ROWS = [
    (110, "不减基线：梯度 = 分数 × 方向", s0, m0, v0,
     ["回答 1：1.0 × 0.3", "回答 2：0.8 × (−0.7)"]),
    (270, "减去基线 b = 0.94：梯度 = (分数 − 0.94) × 方向", s1, m1, v1,
     ["回答 1：(1.0 − 0.94) × 0.3", "回答 2：(0.8 − 0.94) × (−0.7)"]),
]

for y, title, s, m, v, labels in ROWS:
    svg.text(16, y - 62, title, size=13.5, weight="700")
    # axis with ticks; the tick labels sit below the mean marker
    svg.line(ax(-0.7), y, ax(0.45), y, color="muted", width=1.1)
    svg.line(ax(0), y - 14, ax(0), y + 4, color="rule", width=1.2, dash="3 3")
    for t in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4):
        svg.line(ax(t), y - 4, ax(t), y + 4, color="muted", width=1)
        svg.text(ax(t), y + 33, ("%.1f" % t).replace("-", "−"), size=11, anchor="middle", color="muted")
    # samples: circle area proportional to probability
    for i in range(2):
        x = ax(float(s[i]))
        r = 13 * float(PROB[i]) ** 0.5
        col = "green" if s[i] > 0 else "red"
        svg.circle(x, y, r, fill=col, fill_opacity=0.25, stroke=col, width=1.4)
        if i == 0:      # answer 1: label above the circle
            svg.text(x, y - 34, labels[0], size=12, anchor="middle", color=col)
            svg.text(x, y - 18, "= %s（概率 0.7）" % num(s[0]), size=12, anchor="middle", color=col,
                     weight="600")
        else:           # answer 2: label below the tick labels, nudged right when it sits next to answer 1
            lx = x if s[1] < 0 else x + 44
            svg.text(lx, y + 52, labels[1], size=12, anchor="middle", color=col)
            svg.text(lx, y + 68, "= %s（概率 0.3）" % num(s[1]), size=12, anchor="middle", color=col,
                     weight="600")
    # mean marker: a triangle under the axis, pointing up
    mx = ax(float(m))
    svg.polygon([(mx - 6, y + 19), (mx + 6, y + 19), (mx, y + 8)], fill="ink", stroke="ink", width=1)
    # variance panel
    svg.text(560, y - 36, "平均 = %s" % num(m), size=12.5, weight="700")
    svg.text(560, y - 16, "方差 = %s" % ("%.6f" % float(v)).rstrip("0"), size=12.5)
    bw = 160 * float(v) / float(v0)
    svg.rect(560, y - 4, max(bw, 2), 12, fill="violet", fill_opacity=0.3, stroke="violet", width=1)

svg.text(560, 318, "方差小了约 %d 倍" % round(float(v0 / v1)), size=13, color="violet", weight="700")
svg.text(16, 362, "▲ 平均梯度；圆点面积正比于采到它的概率。绿色：把 θ 往正确方向推（回答 1 更可能）；红色：往反方向推。",
         size=11.5, color="muted")
svg.save("l03b-baseline-board.svg")
