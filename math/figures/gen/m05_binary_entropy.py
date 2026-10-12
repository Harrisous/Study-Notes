"""Figure M5.2: binary entropy H(p) = -p log2 p - (1-p) log2 (1-p), redrawn after Shannon (1948) Fig. 7.

Marked coins: p = 0.5 (1 bit), 0.75 (0.811), 0.9 (0.469), 0.99 (0.081); the level H = 0.5 bit is reached
at p = 0.110 and 0.890. Right axis in nats (1 bit = ln 2 nats). All values computed here.
Writes math/figures/m05-binary-entropy.svg
"""

import math

from m05_svg import SVG


def hb(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# p where hb = 0.5 (bisection on [0.5, 1))
lo, hi = 0.5, 1 - 1e-12
for _ in range(200):
    m = (lo + hi) / 2
    lo, hi = (m, hi) if hb(m) > 0.5 else (lo, m)
p_half = lo                      # 0.88997

W, H = 720, 400
X0, X1, Y0, Y1 = 70, 600, 30, 330


def px(p):
    return X0 + p * (X1 - X0)


def py(b):
    return Y1 - b * (Y1 - Y0)


svg = SVG(W, H, "二元熵曲线（据 Shannon 1948 图 7 重绘）")
for i in range(11):
    v = i / 10
    svg.line(X0, py(v), X1, py(v), color="grid", width=1)
    svg.line(px(v), Y0, px(v), Y1, color="grid", width=1)
    svg.text(X0 - 8, py(v) + 4, "%.1f" % v if i else "0", size=11, anchor="end", color="muted")
    svg.text(px(v), Y1 + 17, "%.1f" % v if i not in (0, 10) else str(i // 10), size=11, anchor="middle", color="muted")
for nat in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, math.log(2)):
    y = py(nat / math.log(2))
    svg.line(X1, y, X1 + 5, y, color="muted", width=1)
    svg.text(X1 + 8, y + 4, "%.1f" % nat if nat < 0.69 else "0.693", size=11, color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.line(X1, Y0, X1, Y1, color="muted", width=1)
svg.text((X0 + X1) / 2, Y1 + 38, "正面的概率 p", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "熵 H（bits）", size=12, anchor="middle", color="muted", rotate=-90)
svg.text(X1 + 54, (Y0 + Y1) / 2, "熵 H（nats）", size=12, anchor="middle", color="muted", rotate=90)

# half-bit level
yh = py(0.5)
svg.line(px(1 - p_half), yh, px(p_half), yh, color="violet", width=1.3, dash="5 4")
svg.text(px(0.5), yh - 7, "0.5 bit：p = %.3f 或 %.3f" % (1 - p_half, p_half), size=12, anchor="middle", color="violet")

n = 500
pts = [(px(k / n), py(hb(k / n))) for k in range(n + 1)]
svg.path("M" + " L".join("%.1f %.1f" % xy for xy in pts), stroke="blue", width=2.8)

marks = [(0.5, "公平硬币：1 bit", 0, -12, "middle"),
         (0.75, "p = 0.75：0.811", 10, -6, "start"),
         (0.9, "p = 0.9：0.469", -12, 18, "end"),
         (0.99, "p = 0.99：0.081", -10, -2, "end")]
for p, lab, dx, dy, anc in marks:
    x, y = px(p), py(hb(p))
    svg.circle(x, y, 4.3, fill="ink")
    svg.text(x + dx, y + dy, lab, size=12, anchor=anc)
svg.circle(px(1), py(0), 4.3, fill="ink")
svg.circle(px(0), py(0), 4.3, fill="ink")

svg.save("m05-binary-entropy.svg")
