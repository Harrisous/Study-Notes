"""Figure 13.4: (a) GRPO advantage of a correct / wrong answer versus the group accuracy p (binary
rewards, population std): A+ = sqrt((1-p)/p), A- = -sqrt(p/(1-p)); dots at p = k/16.
(b) probability that a whole group is all-correct or all-wrong (zero learning signal),
s^G + (1-s)^G, for G = 16 and G = 64. All geometry computed here.
Writes deepseek/figures/l13-adv-vs-p.svg
"""

import numpy as np

from l13_svg import SVG, Axes

W, H = 760, 340
svg = SVG(W, H, "二值奖励下 GRPO 的优势随组内正确率变化，以及一组没有学习信号的概率")

# ---------------- panel (a)
a = Axes(66, 352, 44, 290, (0, 1), (-4.5, 4.5))
svg.text(16, 24, "(a) 答对者与答错者的优势（G = 16）", size=13, weight="600")
for v in (-4, -2, 0, 2, 4):
    y = a.py(v)
    svg.line(a.X0, y, a.X1, y, color="grid" if v else "muted", width=1 if v else 1.1)
    svg.text(a.X0 - 6, y + 4, ("%+d" % v).replace("-", "−") if v else "0", size=11, anchor="end", color="muted")
for p in (0, 0.25, 0.5, 0.75, 1):
    x = a.px(p)
    svg.line(x, a.Y0, x, a.Y1, color="grid", width=1)
    svg.text(x, a.Y1 + 16, "%g" % p, size=11, anchor="middle", color="muted")
svg.text((a.X0 + a.X1) / 2, a.Y1 + 36, "组内正确率 p = k / 16", size=12, anchor="middle", color="muted")
svg.text(24, (a.Y0 + a.Y1) / 2, "优势 Â", size=12, anchor="middle", color="muted", rotate=-90)
ps = np.linspace(1 / 16, 15 / 16, 200)
Ap = np.sqrt((1 - ps) / ps)
Am = -np.sqrt(ps / (1 - ps))
svg.polyline([(a.px(p), a.py(v)) for p, v in zip(ps, Ap)], stroke="green", width=2.4)
svg.polyline([(a.px(p), a.py(v)) for p, v in zip(ps, Am)], stroke="red", width=2.4)
for k in range(1, 16):
    p = k / 16
    svg.circle(a.px(p), a.py(np.sqrt((1 - p) / p)), 2.6, fill="green")
    svg.circle(a.px(p), a.py(-np.sqrt(p / (1 - p))), 2.6, fill="red")
# annotate k = 1, 4, 15
for k, txt, col, dy, anc in ((1, "k=1：+3.87", "green", -8, "start"), (4, "k=4：+1.73", "green", -10, "start"),
                             (4, "−0.58", "red", 18, "middle"), (15, "k=15：−3.87", "red", 18, "end")):
    p = k / 16
    v = np.sqrt((1 - p) / p) if col == "green" else -np.sqrt(p / (1 - p))
    svg.text(a.px(p) + (6 if anc == "start" else 0), a.py(v) + dy, txt, size=12, anchor=anc, color=col,
             weight="600")
svg.text(a.px(0.62), a.py(2.7), "答对者 √((1−p)/p)", size=12, color="green")
svg.text(a.px(0.08), a.py(-2.9), "答错者 −√(p/(1−p))", size=12, color="red")

# ---------------- panel (b)
b = Axes(448, 734, 44, 290, (0, 1), (0, 1))
svg.text(408, 24, "(b) 一组全对或全错（没有学习信号）的概率", size=13, weight="600")
for v in (0, 0.25, 0.5, 0.75, 1):
    y = b.py(v)
    svg.line(b.X0, y, b.X1, y, color="grid" if v else "muted", width=1 if v else 1.1)
    svg.text(b.X0 - 6, y + 4, "%g" % v, size=11, anchor="end", color="muted")
for p in (0, 0.25, 0.5, 0.75, 1):
    x = b.px(p)
    svg.line(x, b.Y0, x, b.Y1, color="grid", width=1)
    svg.text(x, b.Y1 + 16, "%g" % p, size=11, anchor="middle", color="muted")
svg.text((b.X0 + b.X1) / 2, b.Y1 + 36, "模型对这道题的单次正确率 s", size=12, anchor="middle", color="muted")
ss = np.linspace(0, 1, 400)
for Gg, col in ((16, "blue"), (64, "violet")):
    z = ss ** Gg + (1 - ss) ** Gg
    svg.polyline([(b.px(s), b.py(v)) for s, v in zip(ss, z)], stroke=col, width=2.4)
s = 0.9
z16, z64 = s ** 16 + (1 - s) ** 16, s ** 64 + (1 - s) ** 64
svg.line(b.px(s), b.Y0, b.px(s), b.Y1, color="muted", width=1, dash="3 3")
svg.circle(b.px(s), b.py(z16), 4, fill="blue")
svg.circle(b.px(s), b.py(z64), 4, fill="violet")
svg.text(b.px(s) - 14, b.py(z16) - 6, "G=16：%.3f" % z16, size=12, anchor="end", color="blue", weight="600")
svg.text(b.px(s) - 14, b.py(z64) - 12, "G=64：%.4f" % z64, size=12, anchor="end", color="violet", weight="600")
svg.text(b.px(0.5), b.py(0.62), "s^{16} + (1−s)^{16}", size=12, anchor="middle", color="blue")
svg.text(b.px(0.5), b.py(0.52), "s^{64} + (1−s)^{64}", size=12, anchor="middle", color="violet")
svg.save("l13-adv-vs-p.svg")
