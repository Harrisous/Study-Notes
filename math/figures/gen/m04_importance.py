"""Figure M4.12: importance sampling. Weights on a die, and how the effective sample size decays.

Left: target p = loaded die (0.1 x 5, 0.5), proposal q = fair die; weights w = p/q.
Right: p = N(mu, 1), q = N(0, 1), w(x) = exp(mu x - mu^2/2). Theory ESS/N = exp(-mu^2)
(E_q[w^2] = exp(mu^2)); dots are Monte Carlo with 10^6 draws from q (numpy default_rng(0),
the same draws for every mu), ESS = (sum w)^2 / sum w^2.
Writes math/figures/m04-importance.svg
"""

import math

import numpy as np

from m04_svg import SVG, Axes

x = np.arange(1, 7)
p = np.array([0.1] * 5 + [0.5])
q = np.ones(6) / 6
w = p / q
Ep = float((p * x).sum())
Eq_wx = float((q * w * x).sum())
var_p = float((p * x ** 2).sum() - Ep ** 2)
var_is = float((q * (w * x) ** 2).sum() - Ep ** 2)
Eq_w2 = float((q * w ** 2).sum())
print("w", w, "E_p X", Ep, "E_q wX", Eq_wx, "Var_p", var_p, "Var_IS", var_is, "E_q w2", Eq_w2, "ESS/N", 1 / Eq_w2)

rng = np.random.default_rng(0)
xs = rng.standard_normal(10 ** 6)
MUS = (0.0, 0.5, 1.0, 1.5, 2.0)
mc = {}
for mu in MUS:
    ww = np.exp(mu * xs - mu * mu / 2)
    mc[mu] = dict(ess=float(ww.sum() ** 2 / (ww ** 2).sum() / len(ww)), meanw=float(ww.mean()),
                  est=float((ww * xs).mean()), snis=float((ww * xs).sum() / ww.sum()),
                  maxshare=float(ww.max() / ww.sum()))
    print("mu", mu, {k: round(v, 5) for k, v in mc[mu].items()}, "theory", round(math.exp(-mu * mu), 4))

W, H = 720, 372
svg = SVG(W, H, "重要性采样：权重与有效样本数")

# ---- left: the die
A = Axes(svg, 60, 58, 300, 200, (0.4, 6.6), (0, 0.6))
A.frame(xticks=[], yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ylabel="概率", ylabel_dx=-40)
svg.text(A.x0 - 20, 26, "用公平骰子（q）估计灌铅骰子（p）", size=13, weight="600")
bw = 18
for i in range(6):
    xc = A.X(i + 1)
    svg.rect(xc - bw - 1, A.Y(q[i]), bw, A.Y(0) - A.Y(q[i]), fill="blue", fill_opacity=0.35, stroke="blue", width=1.1)
    svg.rect(xc + 1, A.Y(p[i]), bw, A.Y(0) - A.Y(p[i]), fill="green", fill_opacity=0.45, stroke="green", width=1.1)
    svg.text(xc, A.Y(0) + 16, str(i + 1), size=12, anchor="middle", color="muted")
    top = max(p[i], q[i])
    svg.text(xc, A.Y(top) - 8, "w=%g" % w[i], size=12, anchor="middle", color="orange", weight="600")
svg.rect(A.x0 + 8, A.y0 + 2, 12, 10, fill="blue", fill_opacity=0.35, stroke="blue", width=1)
svg.text(A.x0 + 26, A.y0 + 12, "q：采样用的公平骰子，1/6", size=12, color="blue")
svg.rect(A.x0 + 8, A.y0 + 22, 12, 10, fill="green", fill_opacity=0.45, stroke="green", width=1)
svg.text(A.x0 + 26, A.y0 + 32, "p：想估计的灌铅骰子", size=12, color="green")
svg.text(A.x0 - 20, A.Y(0) + 42, "E_{q}[wX] = E_{p}[X] = %.1f；方差 %.2f，是直接采 p 的 %.1f 倍" % (Ep, var_is, var_is / var_p),
         size=12, color="ink")
svg.text(A.x0 - 20, A.Y(0) + 66, "E_{q}[w²] = %.1f → 有效样本比例 1/1.8 ≈ %.2f" % (Eq_w2, 1 / Eq_w2), size=12, color="orange")

# ---- right: ESS vs mu
B = Axes(svg, 450, 58, 250, 200, (0, 2.5), (0, 1.05))
B.frame(xticks=[0, 0.5, 1, 1.5, 2, 2.5], yticks=[0, 0.25, 0.5, 0.75, 1.0],
        xlabel="两个分布均值之差 μ", ylabel="有效样本比例 ESS/N", ylabel_dx=-38)
svg.text(B.x0 - 20, 26, "p = N(μ, 1)，q = N(0, 1)", size=13, weight="600")
mus = np.linspace(0, 2.5, 200)
B.curve(mus, np.exp(-mus ** 2), stroke="orange", width=2.2)
for mu in MUS:
    svg.circle(*B.P(mu, mc[mu]["ess"]), 4, fill="ink")
svg.text(B.X(1.0) + 8, B.Y(math.exp(-1)) - 6, "μ = 1：e^{−1} ≈ 0.368", size=12, color="orange")
svg.text(B.X(1.55), B.Y(0.20), "μ = 2：0.018", size=12, color="orange")
svg.text(B.X(1.05), B.Y(0.93), "曲线：理论 e^{−μ²}", size=12, color="orange")
svg.text(B.X(1.05), B.Y(0.83), "圆点：10⁶ 个样本实测", size=12, color="ink")
svg.text(B.x0 - 20, B.Y(0) + 58, "均值只差 2 个标准差，100 万个样本只顶", size=12, color="ink")
svg.text(B.x0 - 20, B.Y(0) + 76, "约 %.1f 万个（理论）；这次实测 %.1f 万个"
         % (math.exp(-4) * 100, mc[2.0]["ess"] * 100), size=12, color="ink")

svg.save("m04-importance.svg")
