"""Figure M4.3: the mean is the balance point, the standard deviation the typical deviation.

Three distributions from section 2: a constant 3.5, a fair die, and the loaded die
p = (0.1, 0.1, 0.1, 0.1, 0.1, 0.5). Each bar is a weight equal to its probability sitting
on a beam; the green fulcrum is at E[X]; the orange bar spans mu - sigma .. mu + sigma.
All numbers are computed here.
Writes math/figures/m04-mean-var.svg
"""

import math
from fractions import Fraction

from m04_svg import SVG

rows = [
    ("常数 3.5", "期望相同，但纹丝不动", {3.5: 1.0}, {3.5: "1"}, None),
    ("公平骰子", "每次在 1 到 6 之间跳", {k: 1 / 6 for k in range(1, 7)}, {k: "1/6" for k in range(1, 7)},
     "偏离 x − μ：−2.5, −1.5, −0.5, +0.5, +1.5, +2.5，各重 1/6；偏离² 的平均 = %.2f = σ²"),
    ("灌铅骰子", "六点重 0.5，支点右移一格", {**{k: 0.1 for k in range(1, 6)}, 6: 0.5},
     {**{k: "0.1" for k in range(1, 6)}, 6: "0.5"},
     "偏离 x − μ：−3.5 … +0.5 各重 0.1，+1.5 重 0.5；偏离² 的平均 = %.2f = σ²"),
]


def stats(p):
    mu = sum(x * w for x, w in p.items())
    var = sum(w * (x - mu) ** 2 for x, w in p.items())
    return mu, var, math.sqrt(var)


for name, _, p, _, _ in rows:
    mu, var, sd = stats(p)
    assert abs(sum(p.values()) - 1) < 1e-12
    print("%s: mean %.4f var %.4f sd %.4f" % (name, mu, var, sd))
# exact check for the fair die: var = 35/12
assert abs(stats(rows[1][2])[1] - float(Fraction(35, 12))) < 1e-12

W = 720
UNIT = 90                        # pixels per unit of probability (bar height)
XL, XR, V0, V1 = 150, 520, 0.3, 6.7


def X(v):
    return XL + (XR - XL) * (v - V0) / (V1 - V0)


heights = [14 + max(p.values()) * UNIT + 58 + (22 if dev else 0) for _, _, p, _, dev in rows]
H = int(58 + sum(heights) + 6)
svg = SVG(W, H, "期望是平衡点，标准差是典型偏离")

# legend
svg.text(20, 24, "竖条 = 概率（放在横梁上的砝码）", size=12, color="blue")
svg.text(232, 24, "绿三角 = 期望 μ（让横梁平衡的支点）", size=12, color="green")
svg.text(472, 24, "橙线 = μ ± σ（一次典型的偏离）", size=12, color="orange")
svg.line(20, 38, W - 20, 38, color="rule", width=1)

top = 50
for (name, note, p, lab, dev), rh in zip(rows, heights):
    mu, var, sd = stats(p)
    hmax = max(p.values()) * UNIT
    beam = top + 14 + hmax
    # beam and ticks
    svg.line(X(V0), beam, X(V1), beam, color="ink", width=1.6)
    for k in range(1, 7):
        svg.line(X(k), beam, X(k), beam + 4, color="muted", width=1)
        svg.text(X(k), beam + 17, str(k), size=11, anchor="middle", color="muted")
    # weights
    bw = 22
    for x, w in p.items():
        h = w * UNIT
        svg.rect(X(x) - bw / 2, beam - h, bw, h, fill="blue", fill_opacity=0.30, stroke="blue", width=1.2)
        svg.text(X(x), beam - h - 4, lab[x], size=11, anchor="middle", color="blue")
    # fulcrum at the mean
    xm = X(mu)
    svg.polygon([(xm, beam + 1.5), (xm - 8, beam + 14), (xm + 8, beam + 14)], fill="green")
    # mu +- sigma
    yb = beam + 34
    if sd > 1e-9:
        svg.line(X(mu - sd), yb, X(mu + sd), yb, color="orange", width=2.2)
        for v in (mu - sd, mu + sd):
            svg.line(X(v), yb - 6, X(v), yb + 6, color="orange", width=2.2)
        svg.circle(xm, yb, 3, fill="orange")
    else:
        svg.circle(xm, yb, 4, fill="orange")
    if dev:
        svg.text(XL - 40, yb + 24, dev % var, size=12, color="muted")
    # left label
    svg.text(20, beam - 6, name, size=13, weight="600")
    # right column
    rx = 548
    svg.text(rx, beam - 22, "μ = %s" % ("%.1f" % mu), size=12, color="green")
    if sd > 1e-9:
        svg.text(rx, beam - 4, "σ = %.2f（方差 %.2f）" % (sd, var), size=12, color="orange")
    else:
        svg.text(rx, beam - 4, "σ = 0（方差 0）", size=12, color="orange")
    svg.text(rx, beam + 14, note, size=12, color="muted")
    top += rh
    if top < H - 20:
        svg.line(20, top - 8, W - 20, top - 8, color="grid", width=1)

svg.save("m04-mean-var.svg")
