"""Figure 2.12 of the blackboard version of lesson 2: the toy power law L = (64 / X)^0.5.

Left: ordinary (linear) axes, the curve flattens: from X = 1 to 4 the loss drops by 4, from 16 to 64 by 1.
Right: both axes logarithmic, the same four points (X = 1, 4, 16, 64 -> L = 8, 4, 2, 1) lie on a straight
line of slope -0.5: every factor 4 in X moves the point the same distance right and down.
All geometry is computed from the formula.
Writes deepseek/figures/l02b-power-law-board.svg
"""

import math

import numpy as np

from l02_svg import SVG, LogAxis

XC, ALPHA = 64.0, 0.5


def L(x):
    return (XC / x) ** ALPHA


PTS = [1, 4, 16, 64]
assert [L(x) for x in PTS] == [8.0, 4.0, 2.0, 1.0]

W, H = 720, 330
svg = SVG(W, H, "黑板上的幂律：普通坐标与双对数坐标")

# ---------------- left panel: linear axes
LX0, LX1, LY0, LY1 = 62, 330, 52, 270
ax = LogAxis(0, 70, LX0, LX1, log=False)
ay = LogAxis(0, 9, LY1, LY0, log=False)
svg.text(LX0 - 40, 24, "(a) 普通坐标：越走越平", size=13, weight="600")
for v in (0, 10, 20, 30, 40, 50, 60, 70):
    x = ax(v)
    svg.line(x, LY0, x, LY1, color="grid", width=1)
    svg.text(x, LY1 + 17, str(v), size=11, anchor="middle", color="muted")
for v in range(0, 10, 2):
    y = ay(v)
    svg.line(LX0, y, LX1, y, color="grid", width=1)
    svg.text(LX0 - 7, y + 4, str(v), size=11, anchor="end", color="muted")
svg.line(LX0, LY1, LX1, LY1, color="muted", width=1.2)
svg.line(LX0, LY0, LX0, LY1, color="muted", width=1.2)
xs = np.linspace(0.79, 70, 300)
svg.path("M" + " L".join("%.1f %.1f" % (ax(x), ay(L(x))) for x in xs), stroke="blue", width=2.4)
for x in PTS:
    svg.circle(ax(x), ay(L(x)), 4.5, fill="blue", stroke="surface", width=1.2)
svg.text(ax(1) + 8, ay(8) + 4, "(1, 8)", size=12, color="ink")
svg.text(ax(4) + 8, ay(4) - 6, "(4, 4)", size=12, color="ink")
svg.text(ax(16) + 4, ay(2) - 9, "(16, 2)", size=12, color="ink")
svg.text(ax(64), ay(1) - 10, "(64, 1)", size=12, anchor="middle", color="ink")
# drop annotations
svg.text(ax(24), ay(6.4), "1 → 4：掉了 4", size=12, color="red")
svg.text(ax(34), ay(3.1), "16 → 64：只掉 1", size=12, color="red")
svg.text((LX0 + LX1) / 2, LY1 + 40, "X", size=12, anchor="middle", color="muted")
svg.text(LX0 - 36, (LY0 + LY1) / 2, "L", size=12, anchor="middle", color="muted")

# ---------------- right panel: log-log axes
RX0, RX1, RY0, RY1 = 432, 690, 52, 270
bx = LogAxis(0.7, 100, RX0, RX1, log=True)
by = LogAxis(0.7, 12, RY1, RY0, log=True)
svg.text(RX0 - 36, 24, "(b) 双对数坐标：一条直线", size=13, weight="600")
for v in PTS:
    x = bx(v)
    svg.line(x, RY0, x, RY1, color="grid", width=1)
    svg.text(x, RY1 + 17, str(v), size=11, anchor="middle", color="muted")
for v in (1, 2, 4, 8):
    y = by(v)
    svg.line(RX0, y, RX1, y, color="grid", width=1)
    svg.text(RX0 - 7, y + 4, str(v), size=11, anchor="end", color="muted")
svg.line(RX0, RY1, RX1, RY1, color="muted", width=1.2)
svg.line(RX0, RY0, RX0, RY1, color="muted", width=1.2)
xs = np.geomspace(0.72, 100, 100)
svg.path("M" + " L".join("%.1f %.1f" % (bx(x), by(L(x))) for x in xs), stroke="blue", width=2.4)
for x in PTS:
    svg.circle(bx(x), by(L(x)), 4.5, fill="blue", stroke="surface", width=1.2)
# staircase: each x4 step right, halve down
for a, b in zip(PTS[:-1], PTS[1:]):
    svg.line(bx(a), by(L(a)), bx(b), by(L(a)), color="red", width=1.3, dash="4 3")
    svg.line(bx(b), by(L(a)), bx(b), by(L(b)), color="red", width=1.3, dash="4 3")
svg.text(bx(4) + 4, by(8) - 8, "×4", size=12, color="red")
svg.text(bx(4) + 5, by(5.6), "÷2", size=12, color="red")
svg.text(bx(10), by(6.5), "斜率 −0.5：每 ×4，L ÷2", size=12, color="blue", weight="600")
svg.text((RX0 + RX1) / 2, RY1 + 40, "X（对数刻度）", size=12, anchor="middle", color="muted")
svg.text(RX0 - 32, (RY0 + RY1) / 2, "L（对数）", size=12, anchor="middle", color="muted", rotate=-90)

svg.save("l02b-power-law-board.svg")
