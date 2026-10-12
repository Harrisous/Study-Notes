"""Figure 5.11: which of DeepSeek's 32 RoPE pairs YaRN keeps, blends or interpolates (research notes R3, R8).

d_h^R = 64, b = 10000, L = 4096, s = 40, alpha = 1, beta = 32, DeepSeek/YaRN reference-code ramp.
Bars: original wavelength lambda_i and YaRN wavelength lambda'_i = lambda_i / scale_code(i), log axis.
Reference lines: lambda = L / beta = 128 (r = 32), lambda = L = 4096 (r = 1), lambda = s * L = 163840.
Writes deepseek/figures/l05-yarn-bands.svg
"""

import math

from l05_svg import SVG
from l05_yarn_common import HIGH, L, LOW, N, S, lam, scale_code

W, H = 760, 440
X0, X1, Y0, Y1 = 80, 700, 58, 350
LY0, LY1 = 0, 7


def X(i):
    return X0 + (i + 0.5) / N * (X1 - X0)


def Y(v):
    return Y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "DeepSeek 的 YaRN：32 对 RoPE 频率中哪些保留、哪些插值")
BW = (X1 - X0) / N
bands = [(0, LOW, "green", "保留原频率（%d 对）" % (LOW + 1)),
         (LOW + 1, HIGH - 1, "amber", "过渡（%d 对）" % (HIGH - 1 - LOW)),
         (HIGH, N - 1, "red", "频率 ÷ %d（%d 对）" % (S, N - HIGH))]
for a, b, col, lab in bands:
    xa, xb = X(a) - BW / 2, X(b) + BW / 2
    svg.rect(xa, Y0, xb - xa, Y1 - Y0, fill=col, fill_opacity=0.07)
    svg.text((xa + xb) / 2, Y0 - 10, lab, size=12.5, anchor="middle", color=col, weight="600")
for e in range(LY0, LY1 + 1):
    svg.line(X0, Y(10 ** e), X1, Y(10 ** e), color="grid", width=1)
    svg.text(X0 - 8, Y(10 ** e) + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
for i in (0, 4, 8, 10, 11, 16, 22, 23, 27, 31):
    svg.text(X(i), Y1 + 16, str(i), size=11, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, Y1 + 34, "第 i 对（DeepSeek-V2/V3 的 64 维解耦 RoPE）", size=12, anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "波长（token，对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

for v, lab, col in ((L / 32, "λ = 4096/32 = 128（r = β）", "ink"), (L, "λ = 4096（r = α）", "ink"),
                    (S * L, "λ = 40 × 4096 = 163,840", "blue")):
    svg.line(X0, Y(v), X1, Y(v), color=col, width=1.1, dash="5 4")
    svg.text(X(0) - 6, Y(v) - 5, lab, size=11.5, color=col)

for i in range(N):
    new = lam[i] / scale_code(i)
    x = X(i)
    # original wavelength: short black tick; YaRN wavelength: blue dot, joined by a thin line if moved
    if new > lam[i] * 1.0001:
        svg.line(x, Y(lam[i]), x, Y(new), color="blue", width=1.4)
    svg.line(x - BW * 0.32, Y(lam[i]), x + BW * 0.32, Y(lam[i]), color="ink", width=2.2)
    svg.circle(x, Y(new), 3.4, fill="blue")

lx, ly = X0 + 12, Y0 + 20
svg.line(lx, ly, lx + 16, ly, color="ink", width=2.2)
svg.text(lx + 22, ly + 4, "原波长 λ_{i}", size=12)
svg.circle(lx + 8, ly + 20, 3.4, fill="blue")
svg.text(lx + 22, ly + 24, "YaRN 之后 λ'_{i}（DeepSeek 代码）", size=12, color="blue")
svg.text(X0, H - 32, "i ≤ 10：训练中转过 36 圈以上，保持不动；i ≥ 23：λ > 4096，训练中没转满一圈，频率除以 40，", size=12)
svg.text(X0, H - 14, "扩展后在 163,840 以内仍转不满一圈（新波长 ≥ %s）；中间 12 对按斜坡混合。" % "{:,.0f}".format(lam[HIGH] * S),
         size=12)
svg.save("l05-yarn-bands.svg")
