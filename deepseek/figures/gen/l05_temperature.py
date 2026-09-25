"""Figure 5.14: YaRN attention temperature mu = sqrt(1/t) and the logit multiplier mu^2 vs scale s (research notes R4).

YaRN / DeepSeek-V3: mu = 0.1 ln s + 1; DeepSeek-V2: mu = 0.0707 ln s + 1 (config mscale_all_dim 0.707).
Points: s = 8 (YaRN appendix value 1.208) and s = 40 (DeepSeek). Log2 x axis 1..128.
Writes deepseek/figures/l05-temperature.svg
"""

import math

from l05_svg import SVG

W, H = 760, 400
X0, X1, Y0, Y1 = 80, 560, 30, 320
YMIN, YMAX = 1.0, 2.3


def X(s):
    return X0 + math.log2(s) / 7 * (X1 - X0)


def Y(v):
    return Y1 - (v - YMIN) / (YMAX - YMIN) * (Y1 - Y0)


def mu(s, c):
    return c * math.log(s) + 1


svg = SVG(W, H, "YaRN 的注意力温度：μ 与 logits 乘数 μ² 随扩展倍数 s 的变化")
for v in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25):
    svg.line(X0, Y(v), X1, Y(v), color="grid", width=1)
    svg.text(X0 - 8, Y(v) + 4, "%g" % v, size=11, anchor="end", color="muted")
for s in (1, 2, 4, 8, 16, 32, 64, 128):
    svg.line(X(s), Y0, X(s), Y1, color="grid", width=1)
    svg.text(X(s), Y1 + 16, str(s), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 34, "扩展倍数 s = L'_{ctx} / L_{ctx}（以 2 为底的对数刻度）", size=12, anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "乘数", size=12, anchor="middle", color="muted", rotate=-90)

ss = [2 ** (7 * k / 200) for k in range(201)]
for c, col, name in ((0.1, "blue", "YaRN / V3"), (0.0707, "violet", "V2")):
    svg.polyline([(X(s), Y(mu(s, c))) for s in ss], stroke=col, width=2.2)
    svg.polyline([(X(s), Y(mu(s, c) ** 2)) for s in ss], stroke=col, width=1.8, dash="6 4")
    svg.text(X1 + 6, Y(mu(128, c)) + 4, "μ（%s）= %.4f" % (name, mu(128, c)), size=12, color=col)
    svg.text(X1 + 6, Y(mu(128, c) ** 2) + 4, "μ²（%s）= %.4f" % (name, mu(128, c) ** 2), size=12, color=col)

s = 40
x = X(s)
svg.line(x, Y0, x, Y1, color="ink", width=1.1, dash="3 3")
svg.text(x + 4, Y0 + 12, "s = 40（DeepSeek）", size=12, weight="600")
for c, col in ((0.1, "blue"), (0.0707, "violet")):
    for v in (mu(s, c), mu(s, c) ** 2):
        svg.circle(x, Y(v), 3.8, fill=col)
svg.text(x - 6, Y(mu(s, 0.1) ** 2) - 8, "V3：μ² = %.4f" % mu(s, 0.1) ** 2, size=11.5, anchor="end", color="blue",
         weight="600")
svg.text(x + 10, Y(1.50), "V2：μ² = %.4f" % mu(s, 0.0707) ** 2, size=11.5, color="violet", weight="600")
svg.text(x - 6, Y(mu(s, 0.1)) - 8, "%.4f" % mu(s, 0.1), size=11.5, anchor="end", color="blue")
svg.text(x + 6, Y(mu(s, 0.0707)) + 16, "%.4f" % mu(s, 0.0707), size=11.5, color="violet")
x8 = X(8)
svg.circle(x8, Y(mu(8, 0.1)), 3.8, stroke="blue", width=1.4)
svg.line(x8, Y(mu(8, 0.1)), x8 + 16, Y(1.06) - 12, color="blue", width=0.9)
# short label: the longer "（YaRN 附录 A.3）" version ran into the s = 40 line; the source is in the caption
svg.text(x8 + 18, Y(1.06), "s = 8：μ = %.3f" % mu(8, 0.1), size=11.5, color="blue")
svg.text(X0, H - 12, "实线 μ：q、k 各乘的因子；虚线 μ²：logits 实际被乘的倍数。μ² > 1 让 softmax 更尖锐（等价温度 1/μ² < 1）。",
         size=12)
svg.save("l05-temperature.svg")
