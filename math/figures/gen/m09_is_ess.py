"""Figure M9.8: effective sample size of a sequence-level importance ratio versus response length.

Model (author's derivation, m09-notes §2.4): per-token log ratios l_t = log(pi_theta / pi_old) are i.i.d.
N(-s^2/2, s^2), so each token ratio has mean 1 and the per-token KL(pi_old || pi_theta) is s^2/2.
The sequence ratio W = prod_t exp(l_t) has E[W] = 1 and E[W^2] = exp(T s^2), so ESS/N = exp(-T s^2).
A single token ratio has ESS/N = exp(-s^2), essentially 1.
Writes math/figures/m09-is-ess.svg
"""

import math

from m09_svg import SVG

sigmas = [(0.02, "green"), (0.05, "blue"), (0.10, "red")]
TMIN, TMAX = 1, 10_000
EMIN, EMAX = -6, 0

W, H = 720, 380
svg = SVG(W, H, "序列级重要性比的有效样本比例随回答长度的变化")
X0, X1, Y0, Y1 = 78, 500, 30, 300


def px(T):
    return X0 + (math.log10(T) - math.log10(TMIN)) / (math.log10(TMAX) - math.log10(TMIN)) * (X1 - X0)


def py(v):
    return Y1 - (math.log10(v) - EMIN) / (EMAX - EMIN) * (Y1 - Y0)


for e in range(EMIN, EMAX + 1):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    lab = "1" if e == 0 else "10^{%d}" % e
    svg.text(X0 - 8, y + 4, lab, size=11, anchor="end", color="muted")
for T, lab in ((1, "1"), (10, "10"), (100, "100"), (1000, "1000"), (10000, "10^{4}")):
    x = px(T)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, lab, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "回答长度 T（token，对数刻度）", size=12, anchor="middle", color="muted")
svg.text(20, (Y0 + Y1) / 2, "有效样本比例 ESS/N（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

# token-level ratio: flat, ~1
svg.line(X0, py(math.exp(-0.1 ** 2)), X1, py(math.exp(-0.1 ** 2)), color="ink", width=1.6, dash="6 4")
svg.line(X0 + 10, py(1e-5) - 4, X0 + 36, py(1e-5) - 4, color="ink", width=1.6, dash="6 4")
svg.text(X0 + 42, py(1e-5), "逐 token 的比值：不连乘，", size=12, color="ink")
svg.text(X0 + 42, py(1e-5) + 18, "ESS/N = exp(−σ²) ≈ 1", size=12, color="ink")
svg.text(X0 + 42, py(1e-5) + 36, "（σ = 0.1 时 0.990）", size=12, color="muted")

for s, col in sigmas:
    tcut = min(TMAX, -math.log(10 ** EMIN) / s ** 2)   # stop at the bottom of the box
    n = 300
    pts = []
    for i in range(n + 1):
        T = 10 ** (math.log10(TMIN) + (math.log10(tcut) - math.log10(TMIN)) * i / n)
        pts.append((px(T), py(math.exp(-T * s * s))))
    svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke=col, width=2.4)

# marked values at T = 100 and T = 1000
for s, col in sigmas:
    for T in (100, 1000):
        v = math.exp(-T * s * s)
        if v >= 10 ** EMIN:
            svg.circle(px(T), py(v), 3.6, fill=col)
labels = [
    (0.05, 1000, "0.082", 8, 4),
    (0.10, 100, "0.368", -8, 16),
    (0.10, 1000, "4.5×10^{−5}", 8, 4),
    (0.02, 1000, "0.670", 8, 20),     # below-right of the dot: clear of the blue curve
    (0.05, 100, "0.779", 8, 16),
]
for s, T, lab, dx, dy in labels:
    col = dict(sigmas)[s]
    v = math.exp(-T * s * s)
    svg.text(px(T) + dx, py(v) + dy, lab, size=12, color=col, weight="600", anchor="start" if dx > 0 else "end")

# legend
LX = 520
svg.text(LX, Y0 + 8, "每 token 的对数比 ℓ ~ N(−σ²/2, σ²)", size=12, weight="600")
svg.text(LX, Y0 + 26, "每 token 的 KL = σ²/2", size=12, color="muted")
y = Y0 + 58
for s, col in sigmas:
    svg.line(LX, y - 4, LX + 24, y - 4, color=col, width=2.4)
    svg.text(LX + 30, y, "σ = %.2f" % s, size=12.5, color=col, weight="600")
    svg.text(LX + 30, y + 17, "每 token KL = %.4g" % (s * s / 2), size=12, color="muted")
    y += 44
svg.text(LX, y + 6, "整段回答的比值 W = Π_{t} ρ_{t}：", size=12)
svg.text(LX, y + 24, "E[W] = 1，E[W²] = exp(Tσ²)，", size=12)
svg.text(LX, y + 42, "ESS/N ≈ 1/E[W²] = exp(−Tσ²)", size=12, weight="600")
svg.text(LX, y + 66, "每 token 的差别再小，", size=12, color="muted")
svg.text(LX, y + 83, "乘上几千次也会让权重退化。", size=12, color="muted")

svg.save("m09-is-ess.svg")
