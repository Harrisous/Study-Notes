"""Figure M4.13: why PPO/GRPO use per-token ratios, not the whole-sequence weight.

Toy model (the author's illustration, same as exercise 12): tokens independent,
log rho_t ~ N(-s^2/2, s^2), so E[rho_t] = 1 exactly (as an importance weight must be).
Whole-sequence weight R = prod_t rho_t: log R ~ N(-T s^2/2, T s^2), E[R] = 1.
E[rho_t^2] = exp(s^2), so E[R^2] = exp(T s^2) and by (M4.29) ESS/N ~ exp(-T s^2);
a single per-token ratio keeps ESS/N ~ exp(-s^2).
Left: density of log10 R for T = 10, 100, 1000 (s = 0.05), each row scaled to its own peak.
Right: ESS/N = exp(-T s^2) against T for s = 0.02, 0.05, 0.1, and the per-token line.
Everything is analytic; no random numbers.
Writes math/figures/m04-seq-weight.svg
"""

import math

import numpy as np

from m04_svg import SVG, Axes

S = 0.05
TS = (10, 100, 1000)
LN10 = math.log(10)

W, H = 740, 372
svg = SVG(W, H, "整段权重随长度退化，逐 token 比值不会")

# ---- left: ridgeline of log10 R
A = Axes(svg, 70, 64, 330, 222, (-3, 1.5), (0, 3.3))
xt = [-3, -2, -1, 0, 1]
lab = {-3: "0.001", -2: "0.01", -1: "0.1", 0: "1", 1: "10"}
A.frame(xticks=xt, yticks=[], xfmt=lambda v: lab[int(v)], grid=True,
        xlabel="整段权重 R（对数轴）")
svg.text(A.x0 - 50, 26, "整段权重 R = ∏ ρ_{t} 的分布（s = 0.05）", size=13, weight="600")
svg.text(A.x0 - 50, 44, "每行按自己的峰高缩放", size=12, color="muted")
colors = {10: "green", 100: "blue", 1000: "red"}
rows = {10: 2.2, 100: 1.1, 1000: 0.0}           # baseline of each ridge (data units)
xs = np.linspace(-3, 1.5, 226)
for T in TS:
    m, sd = -T * S * S / 2, S * math.sqrt(T)   # log R (natural log)
    m10, sd10 = m / LN10, sd / LN10
    dens = np.exp(-0.5 * ((xs - m10) / sd10) ** 2)
    base = rows[T]
    ys = base + 0.95 * dens
    pts = [A.P(xs[0], base)] + [A.P(x, y) for x, y in zip(xs, ys)] + [A.P(xs[-1], base)]
    col = colors[T]
    svg.polygon(pts, fill=col, fill_opacity=0.16, stroke=None)
    A.curve(xs, ys, stroke=col, width=1.8)
    svg.line(A.X(-3), A.Y(base), A.X(1.5), A.Y(base), color="rule", width=1)
    svg.text(A.X(-3) - 8, A.Y(base + 0.35), "T = %d" % T, size=12, anchor="end", color=col, weight="600")
    svg.text(A.X(-3) - 8, A.Y(base + 0.35) + 16, "σ = %.2f" % sd, size=11, anchor="end", color=col)
    print("T", T, "log R mean %.4f sd %.4f" % (m, sd), "median R %.4f" % math.exp(m))
svg.line(A.X(0), A.Y(0), A.X(0), A.Y(3.3), color="ink", width=1.4, dash="5 4")
svg.text(A.X(0) + 6, A.Y(3.2), "R = 1 = E[R]", size=12, color="ink")
svg.text(A.x0 - 50, A.Y(0) + 56, "σ 是 log R 的标准差 s√T：长度翻 100 倍，宽度翻 10 倍", size=12, color="muted")
svg.text(A.x0 - 50, A.Y(0) + 74, "期望始终是 1，却越来越靠右边一条细长的尾巴撑着", size=12, color="muted")

# ---- right: ESS/N against T
B = Axes(svg, 490, 64, 220, 222, (1, 10000), (0, 1.0), logx=True)
tlab = {1: "1", 10: "10", 100: "100", 1000: "1000", 10000: "10000"}
B.frame(xticks=[1, 10, 100, 1000, 10000], yticks=[0, 0.25, 0.5, 0.75, 1.0], xlabel="回答长度 T（对数轴）",
        ylabel="有效样本比例 ESS/N", ylabel_dx=-40, xfmt=lambda v: tlab[int(round(v))])
svg.text(B.x0 - 50, 26, "ESS/N ≈ 1/E[R²] = e^{−T s²}", size=13, weight="600")
Tg = np.logspace(0, 4, 241)
CURVES = ((0.02, "violet", 1.5), (0.05, "blue", 2.4), (0.1, "orange", 1.5))
for s, col, wdt in CURVES:
    B.curve(Tg, np.exp(-Tg * s * s), stroke=col, width=wdt)
# legend in the empty lower-left corner (every curve is still above 0.6 for T < 50)
for k, (s, col, wdt) in enumerate(CURVES):
    y = B.Y(0.44) + k * 18
    svg.line(B.X(1.4), y - 4, B.X(4), y - 4, color=col, width=wdt)
    svg.text(B.X(4.6), y, "s = %g" % s, size=12, color=col, weight=("600" if s == S else None))
per_tok = math.exp(-S * S)
svg.line(B.X(1), B.Y(per_tok), B.X(10000), B.Y(per_tok), color="green", width=1.8, dash="6 4")
svg.text(B.x0 - 50, B.y0 - 8, "绿虚线 = 逐 token 比值：e^{−s²} ≈ %.4f" % per_tok, size=12, color="green")
ess = math.exp(-1000 * S * S)
svg.circle(*B.P(1000, ess), 4, fill="blue")
svg.text(B.X(1000) + 7, B.Y(ess) - 9, "%.3f" % ess, size=12, color="blue", weight="600")
svg.text(B.x0 - 50, B.Y(0) + 56, "1000 个回答只相当于约 %d 个直接采样的回答" % round(1000 * ess), size=12, color="blue")
svg.text(B.x0 - 50, B.Y(0) + 74, "玩具模型：各 token 独立，log ρ_{t} ~ N(−s²/2, s²)", size=12, color="muted")
print("ESS/N T=1000", ess, "T=100", math.exp(-100 * S * S), "T=10", math.exp(-10 * S * S), "per token", per_tok)

svg.save("m04-seq-weight.svg")
