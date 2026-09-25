"""Figure 5.12: the YaRN ramp gamma(r) as written in the paper vs the reference-code ramp (research notes R8, E16).

x: r_i = L / lambda_i (turns inside the pre-training window), log axis; y: blend weight gamma
(1 = keep the original frequency, 0 = divide it by s). Paper: gamma linear in r between alpha = 1 and beta = 32.
Code (YaRN repo and DeepSeek): weight linear in the pair index i between floor(i(beta)) = 10 and ceil(i(alpha)) = 23.
DeepSeek setting d_h^R = 64, L = 4096, s = 40. Writes deepseek/figures/l05-ramp.svg
"""

import math

from l05_svg import SVG
from l05_yarn_common import ALPHA, BETA, N, gamma_paper, mask_code, r

W, H = 760, 420
X0, X1, Y0, Y1 = 80, 700, 40, 300
LX0, LX1 = math.log10(0.05), math.log10(1000)


def X(v):
    return X0 + (math.log10(v) - LX0) / (LX1 - LX0) * (X1 - X0)


def Y(g):
    return Y1 - g * (Y1 - Y0)


svg = SVG(W, H, "YaRN 的斜坡函数：论文式与参考代码")
for g in (0, 0.25, 0.5, 0.75, 1):
    svg.line(X0, Y(g), X1, Y(g), color="grid", width=1)
    svg.text(X0 - 8, Y(g) + 4, "%g" % g, size=11, anchor="end", color="muted")
for v in (0.1, 1, 10, 100, 1000):
    svg.line(X(v), Y0, X(v), Y1, color="grid", width=1)
    svg.text(X(v), Y1 + 16, "%g" % v, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 34, "r_{i} = L / λ_{i}：这一对在 4096 个位置内转过的圈数（对数刻度）", size=12, anchor="middle",
         color="muted")
svg.text(22, (Y0 + Y1) / 2, "混合权重 γ（1 = 保留，0 = ÷40）", size=12, anchor="middle", color="muted", rotate=-90)

# paper ramp as a continuous curve (piecewise linear in r, drawn on a log axis)
pts = []
for k in range(400):
    v = 10 ** (LX0 + (LX1 - LX0) * k / 399)
    pts.append((X(v), Y(gamma_paper(v))))
svg.polyline(pts, stroke="amber", width=2)
for v, lab, yl in ((ALPHA, "α = 1", Y0 + 12), (BETA, "β = 32", Y1 - 12)):
    svg.line(X(v), Y0, X(v), Y1, color="ink", width=1, dash="4 3")
    svg.text(X(v) + 4, yl, lab, size=12)
for i in range(N):
    svg.circle(X(r[i]), Y(gamma_paper(r[i])), 3.8, fill="amber")
    svg.circle(X(r[i]), Y(mask_code(i)), 5.2, stroke="blue", width=1.5)
# annotate i = 16: short labels beside the two points, so none of them crosses the alpha / beta lines
i = 16
svg.line(X(r[i]), Y(gamma_paper(r[i])), X(r[i]), Y(mask_code(i)), color="ink", width=1)
svg.text(X(r[i]) + 2, Y(mask_code(i)) - 10, "i = 16，r = %.2f" % r[i], size=11.5, anchor="end", color="ink")
svg.text(X(r[i]) + 9, Y(mask_code(i)) + 4, "代码 %.3f" % mask_code(i), size=11.5, color="blue", halo=True)  # the paper curve passes behind it
svg.text(X(r[i]) + 9, Y(gamma_paper(r[i])) + 4, "论文 %.3f" % gamma_paper(r[i]), size=11.5, color="amber")
# legend below the axis title (inside the plot the alpha line ran through it)
LG = Y1 + 58
svg.circle(X0 + 8, LG - 4, 3.8, fill="amber")
svg.text(X0 + 20, LG, "论文式：γ 对 r 线性（实心点 = 32 对各自的取值）", size=12, color="amber")
svg.circle(X0 + 8, LG + 16, 5.2, stroke="blue", width=1.5)
svg.text(X0 + 20, LG + 20, "参考代码 / DeepSeek：对下标 i 线性，i = 10 … 23（空心圈）", size=12, color="blue")
svg.text(X0, H - 12, "两者“全保留 / 过渡 / 全插值”的分界相同（11 / 12 / 9 对），过渡区里代码保留得更多。", size=12)
svg.save("l05-ramp.svg")
