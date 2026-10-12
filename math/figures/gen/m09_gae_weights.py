"""Figure M9.7: how much weight GAE puts on the TD residual l steps ahead, (gamma*lambda)^l.

Settings: (gamma, lambda) = (0.99, 0.95), (0.99, 1), (1, 0.95) and (0.99, 0); the sum of the
weights 1/(1 - gamma*lambda) is the "effective horizon" (shared notes F09-1, N71).
Writes math/figures/m09-gae-weights.svg
"""

import math

from m09_svg import SVG

LMAX = 60
sets = [
    ((0.99, 1.0), "violet", "γ = 0.99，λ = 1（蒙特卡洛回报减 V）"),
    ((1.0, 0.95), "green", "γ = 1，λ = 0.95（Stiennon 等人的取值）"),
    ((0.99, 0.95), "blue", "γ = 0.99，λ = 0.95（GAE 论文区间内）"),
]

W, H = 720, 360
svg = SVG(W, H, "GAE 给第 l 步之后的 TD 残差的权重 (γλ)^l")
X0, X1, Y0, Y1 = 64, 470, 30, 290


def px(l):
    return X0 + l / LMAX * (X1 - X0)


def py(v):
    return Y1 - v * (Y1 - Y0)


for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
    y = py(v)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "%.1f" % v, size=11, anchor="end", color="muted")
for l in range(0, LMAX + 1, 10):
    x = px(l)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 16, str(l), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 36, "往后第 l 步的 TD 残差 δ_{t+l}", size=12, anchor="middle", color="muted")
svg.text(18, (Y0 + Y1) / 2, "权重 (γλ)^{l}", size=12, anchor="middle", color="muted", rotate=-90)

for (gm, lm), col, lab in sets:
    pts = [(px(l), py((gm * lm) ** l)) for l in range(LMAX + 1)]
    svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke=col, width=2.4)
    for l in (0, 10, 20, 30, 40, 50, 60):
        svg.circle(px(l), py((gm * lm) ** l), 2.6, fill=col)
# lambda = 0: all weight on l = 0
svg.circle(px(0), py(1.0), 6, stroke="red", width=2)
svg.line(px(0), py(0), px(LMAX), py(0), color="red", width=2.2, dash="2 3")

# half-life marker for (0.99, 0.95)
gl = 0.99 * 0.95
half = math.log(0.5) / math.log(gl)
svg.line(px(half), py(0.5), px(half), Y1, color="blue", width=1, dash="3 3")
svg.line(X0, py(0.5), px(half), py(0.5), color="blue", width=1, dash="3 3")
# right of the vertical dashed line, near the axis: every curve is above 0.25 there
svg.text(px(half) + 5, py(0.06), "半衰期 %.1f 步" % half, size=12, color="blue")

# legend and effective horizons
LX = 488
svg.text(LX, Y0 + 8, "有效视界 1/(1−γλ)", size=12.5, weight="600")
y = Y0 + 36
for (gm, lm), col, lab in sets[::-1]:
    horizon = 1 / (1 - gm * lm)
    svg.line(LX, y - 4, LX + 22, y - 4, color=col, width=2.4)
    svg.text(LX + 28, y, "γλ = %.4f → %.1f 步" % (gm * lm, horizon), size=12.5, color=col, weight="600")
    svg.text(LX + 28, y + 18, lab.split("（")[0], size=12, color="muted")
    svg.text(LX + 28, y + 35, "（" + lab.split("（")[1], size=12, color="muted")
    y += 62
svg.circle(LX + 11, y - 4, 5, stroke="red", width=2)
svg.text(LX + 28, y, "λ = 0 → 只看 δ_{t}（1 步）", size=12.5, color="red", weight="600")
svg.text(LX + 28, y + 18, "（全部权重在 l = 0）", size=12, color="muted")

svg.save("m09-gae-weights.svg")
