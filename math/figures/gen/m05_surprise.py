"""Figure M5.1: surprise h(p) = -log2 p (bits; nats on the right axis) on a linear p axis.

Marked events: p = 0.9, 1/2 (fair coin), 1/6 (die face), 0.1, 0.01. A side panel shows additivity:
two independent fair coins, p = 1/2 * 1/2 = 1/4, surprise 1 + 1 = 2 bits.
Every point and the curve are computed here. Writes math/figures/m05-surprise.svg
"""

import math

from m05_svg import SVG

W, H = 720, 380
X0, X1, Y0, Y1 = 64, 452, 34, 314
YMAX = 7.0                      # bits


def px(p):
    return X0 + p * (X1 - X0)


def py(b):
    return Y1 - b / YMAX * (Y1 - Y0)


svg = SVG(W, H, "信息量 h(p) = −log₂ p：概率越小，意外越大")

# grid and ticks
for b in range(0, 8):
    y = py(b)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, str(b), size=11, anchor="end", color="muted")
for nat in (1, 2, 3, 4):
    y = py(nat / math.log(2))
    svg.line(X1, y, X1 + 5, y, color="muted", width=1)
    svg.text(X1 + 8, y + 4, str(nat), size=11, color="muted")
for i in range(0, 11):
    p = i / 10
    x = px(p)
    svg.line(x, Y1, x, Y1 + 4, color="muted", width=1)
    svg.text(x, Y1 + 17, ("%.1f" % p) if i not in (0, 10) else str(i // 10), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.line(X1, Y0, X1, Y1, color="muted", width=1)
svg.text((X0 + X1) / 2, Y1 + 38, "事件的概率 p", size=12, anchor="middle", color="muted")
svg.text(20, (Y0 + Y1) / 2, "信息量（bits）", size=12, anchor="middle", color="muted", rotate=-90)
svg.text(X1 + 34, (Y0 + Y1) / 2, "信息量（nats）", size=12, anchor="middle", color="muted", rotate=90)

# curve, truncated at the top of the plot (p >= 2^-7)
pmin = 2 ** -YMAX
pts = []
n = 400
for k in range(n + 1):
    p = pmin * (1 / pmin) ** (k / n)      # geometric spacing, dense near 0
    pts.append((px(p), py(-math.log2(p))))
svg.path("M" + " L".join("%.1f %.1f" % xy for xy in pts), stroke="blue", width=2.6)

marks = [
    (0.9, "p = 0.9：0.15 bit", 14, -20, "end"),     # ends left of the nats axis, above the curve
    (0.5, "硬币正面 1/2：1 bit", 10, -8, "start"),
    (1 / 6, "骰子某一面 1/6：2.585 bits", 10, -8, "start"),
    (0.1, "p = 0.1：3.32 bits", 12, -12, "start"),
    (0.01, "p = 0.01：6.64 bits = 4.61 nats", 12, 6, "start"),
]
for p, lab, dx, dy, anc in marks:
    x, y = px(p), py(-math.log2(p))
    svg.circle(x, y, 4.2, fill="ink")
    svg.text(x + dx, y + dy, lab, size=12, anchor=anc)

svg.text(X1 - 10, Y0 + 20, "p → 1 时 h → 0（毫不意外）", size=12, anchor="end", color="muted")
svg.text(X1 - 10, Y0 + 38, "p → 0 时 h → ∞", size=12, anchor="end", color="muted")

# ---- side panel: additivity
BX, BY = 500, 44
svg.rect(BX, BY, 204, 262, fill="blue", fill_opacity=0.05, stroke="rule", width=1, rx=8)
svg.text(BX + 102, BY + 24, "独立事件：意外相加", size=13, anchor="middle", weight="600")
# two coins
for i, (cx, lab) in enumerate(((BX + 60, "硬币 1"), (BX + 144, "硬币 2"))):
    svg.circle(cx, BY + 70, 22, fill="amber", fill_opacity=0.15, stroke="amber", width=1.4)
    svg.text(cx, BY + 75, "正", size=14, anchor="middle", color="amber", weight="600")
    svg.text(cx, BY + 110, lab, size=12, anchor="middle", color="muted")
    svg.text(cx, BY + 130, "p = 1/2", size=12, anchor="middle")
    svg.text(cx, BY + 148, "h = 1 bit", size=12, anchor="middle", color="blue")
svg.text(BX + 102, BY + 76, "×", size=16, anchor="middle", color="muted")
svg.line(BX + 20, BY + 164, BX + 184, BY + 164, color="rule", width=1)
svg.text(BX + 102, BY + 188, "两枚都是正面", size=12, anchor="middle")
svg.text(BX + 102, BY + 208, "p = 1/2 × 1/2 = 1/4", size=12, anchor="middle")
svg.text(BX + 102, BY + 230, "h = −log₂(1/4) = 2 bits", size=12, anchor="middle", color="blue", weight="600")
svg.text(BX + 102, BY + 250, "= 1 + 1", size=12, anchor="middle", color="blue")

svg.save("m05-surprise.svg")
