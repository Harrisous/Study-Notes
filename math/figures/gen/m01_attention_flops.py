"""Figure M1.8: FLOPs of one attention layer versus sequence length T (log-log).

Projections (Q, K, V, O): 8 T d^2.  Scores QK^T plus weighted sum PV over all heads: 4 T^2 d.
d = 768 (GPT-2 small). The two curves cross at T = 2d = 1536. Every point is computed here.
Writes math/figures/m01-attention-flops.svg
"""

import math

import numpy as np

from m01_svg import SVG

d = 768
Ts = np.logspace(math.log10(64), math.log10(131072), 200)
proj = 8 * Ts * d * d
attn = 4 * Ts * Ts * d
total = proj + attn
cross = 2 * d
assert abs(8 * cross * d * d - 4 * cross * cross * d) < 1e-6

W, H = 720, 400
X0, X1, Y0, Y1 = 86, 590, 36, 332          # plot area
LX0, LX1 = math.log2(64), math.log2(131072)
LY0, LY1 = 7, 14                           # 10^7 .. 10^14 FLOPs


def px(T):
    return X0 + (math.log2(T) - LX0) / (LX1 - LX0) * (X1 - X0)


def py(F):
    return Y1 - (math.log10(F) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "一层注意力的 FLOPs 随序列长度 T 的变化（d = 768，双对数坐标）")

# grid + ticks
for e in range(LY0, LY1 + 1):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
for T, lab in ((64, "64"), (256, "256"), (1024, "1K"), (4096, "4K"), (16384, "16K"), (65536, "64K")):
    x = px(T)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, lab, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "序列长度 T（token，对数刻度，1K = 1024）", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "每层 FLOPs（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)


def poly(xs, ys):
    return "M" + " L".join("%.1f %.1f" % (px(x), py(y)) for x, y in zip(xs, ys))


svg.path(poly(Ts, total), stroke="ink", width=1.3, dash="5 4")
svg.path(poly(Ts, proj), stroke="blue", width=2.6)
svg.path(poly(Ts, attn), stroke="violet", width=2.6)

# GPT-2 context length marker
x1024 = px(1024)
svg.line(x1024, Y0 + 6, x1024, Y1, color="muted", width=1.3, dash="3 3")
svg.text(x1024 - 6, Y0 + 18, "GPT-2 的上下文 1024", size=12, anchor="end", color="muted")

# crossover
xc, yc = px(cross), py(8 * cross * d * d)
svg.circle(xc, yc, 4.5, fill="ink")
svg.line(xc + 3, yc + 4, xc + 26, yc + 38, color="ink", width=0.8)
svg.text(xc + 28, yc + 50, "交点 T = 2d = 1536", size=12, weight="600")
svg.text(xc + 28, yc + 66, "两项都是 %.2f×10^{9}" % (8 * cross * d * d / 1e9), size=12, color="muted")

# direct labels at the right end
Te = 131072
svg.text(X1 + 8, py(proj[-1]) + 4, "投影 8Td^{2}", size=13, color="blue", weight="600")
svg.text(X1 + 8, py(proj[-1]) + 21, "斜率 1", size=12, color="muted")
svg.text(X1 + 8, py(attn[-1]) + 12, "打分 + 加权求和", size=13, color="violet", weight="600")
svg.text(X1 + 8, py(attn[-1]) + 32, "4T^{2}d", size=13, color="violet", weight="600")
svg.text(X1 + 8, py(attn[-1]) + 50, "斜率 2", size=12, color="muted")
svg.text(X1 + 8, py(total[-1]) - 12, "合计（虚线）", size=12, color="ink")

# values at T = 1024 (GPT-2): dots on the curves, numbers in a callout in the empty upper left
p1024, a1024 = 8 * 1024 * d * d, 4 * 1024 * 1024 * d
for F, col in ((p1024, "blue"), (a1024, "violet")):
    svg.circle(x1024, py(F), 3.4, fill=col)
bx, by = X0 + 10, Y0 + 34
assert bx + 160 < x1024 - 4                 # the callout stays left of the dashed T = 1024 line
svg.rect(bx, by, 160, 70, fill="surface", stroke="rule", rx=6)
svg.text(bx + 10, by + 20, "T = 1024 时（圆点）：", size=12)
svg.text(bx + 10, by + 40, "投影 %.2f×10^{9}" % (p1024 / 1e9), size=12, color="blue", weight="600")
svg.text(bx + 10, by + 59, "打分+加权求和 %.2f×10^{9}" % (a1024 / 1e9), size=12, color="violet", weight="600")

svg.save("m01-attention-flops.svg")
