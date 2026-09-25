"""Figure M4.6: the dot product of two random d-dimensional vectors, before and after /sqrt(d).

q, k have i.i.d. N(0, 1) components (the assumption of Vaswani et al. 2017, footnote 4).
200,000 pairs per d, numpy default_rng(seed 0 + d). Histograms are densities; the smooth
curves are the normal densities N(0, d) and N(0, 1) that the derivation predicts.
The exact probability of landing outside +-2 sqrt(d) is computed too: given k, q.k is
N(0, |k|^2) and |k|^2 is chi-square with d degrees of freedom, so
P(|q.k| > 2 sqrt(d)) = E[erfc(sqrt(2 d / |k|^2))], integrated numerically below.
Writes math/figures/m04-dot-variance.svg
"""

import math

import numpy as np

from m04_svg import SVG, Axes

DS = (16, 64, 256)
COL = {16: "green", 64: "blue", 256: "violet"}
NPAIR = 200_000

erfc = np.frompyfunc(math.erfc, 1, 1)


def exact_out2(d, n=200_001):
    """P(|q.k| > 2 sqrt(d)) for q, k with i.i.d. N(0, 1) components (trapezoid rule over |k|^2)."""
    x = np.linspace(1e-9, d + 40 * math.sqrt(2 * d), n)
    logpdf = (d / 2 - 1) * np.log(x) - x / 2 - (d / 2) * math.log(2) - math.lgamma(d / 2)
    f = erfc(np.sqrt(2 * d / x)).astype(float) * np.exp(logpdf)
    return float(((f[1:] + f[:-1]) / 2 * np.diff(x)).sum())


EXACT = {d: exact_out2(d) for d in DS}

stats = {}
raw, scl = {}, {}
for d in DS:
    rng = np.random.default_rng(d)
    q = rng.standard_normal((NPAIR, d), dtype=np.float32)
    k = rng.standard_normal((NPAIR, d), dtype=np.float32)
    s = (q.astype(np.float64) * k).sum(1)
    raw[d] = s
    scl[d] = s / math.sqrt(d)
    stats[d] = dict(var=float(s.var()), sd=float(s.std()), var_scaled=float((s / math.sqrt(d)).var()),
                    out2=float(np.mean(np.abs(s) > 2 * math.sqrt(d))), maxabs=float(np.abs(s).max()))

W, H = 720, 410
svg = SVG(W, H, "两个 d 维随机向量的点积：除以根号 d 之前与之后")


def normal(x, sd):
    return np.exp(-0.5 * (x / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def hist(ax, data, lo, hi, nb, col):
    h, e = np.histogram(data, bins=nb, range=(lo, hi), density=True)
    pts = [ax.P(e[0], 0)]
    for i, v in enumerate(h):
        pts += [ax.P(e[i], v), ax.P(e[i + 1], v)]
    pts.append(ax.P(e[-1], 0))
    svg.polygon(pts, fill=col, fill_opacity=0.12, stroke=col, width=1.3)


# left: raw
A = Axes(svg, 58, 50, 330, 220, (-60, 60), (0, 0.105))
A.frame(xticks=[-60, -40, -20, 0, 20, 40, 60], yticks=[0, 0.025, 0.05, 0.075, 0.1],
        xlabel="q·k（未缩放）", ylabel="概率密度", ylabel_dx=-44)
svg.text(A.x0, 26, "未缩放：宽度随 d 变", size=13, weight="600")
for d in DS:
    hist(A, raw[d], -60, 60, 96, COL[d])
xs = np.linspace(-60, 60, 400)
for d in DS:
    A.curve(xs, normal(xs, math.sqrt(d)), stroke="ink", width=1.1, dash="3 3")
svg.text(A.X(6), A.Y(0.098), "d = 16，σ = 4", size=12, color=COL[16])
svg.text(A.X(13), A.Y(0.052), "d = 64，σ = 8", size=12, color=COL[64])
svg.text(A.X(23), A.Y(0.030), "d = 256，σ = 16", size=12, color=COL[256])

# right: scaled
B = Axes(svg, 470, 50, 230, 220, (-4, 4), (0, 0.45))
B.frame(xticks=[-4, -2, 0, 2, 4], yticks=[0, 0.1, 0.2, 0.3, 0.4], xlabel="q·k / √d")
svg.text(B.x0, 26, "除以 √d 之后：三条几乎重合", size=13, weight="600")
for d in DS:
    hist(B, scl[d], -4, 4, 64, COL[d])
xb = np.linspace(-4, 4, 300)
B.curve(xb, normal(xb, 1.0), stroke="ink", width=1.4, dash="3 3")
svg.text(B.X(1.25), B.Y(0.36), "虚线：N(0, 1)", size=12, color="ink")

# table of the measured numbers
ty = 322
svg.text(58, ty, "20 万对样本的实测（理论值在括号里）：", size=12, color="ink", weight="600")
for i, d in enumerate(DS):
    st = stats[d]
    svg.text(58 + i * 222, ty + 22,
             "d = %d：方差 %.1f（%d）" % (d, st["var"], d), size=12, color=COL[d])
    svg.text(58 + i * 222, ty + 40,
             "缩放后方差 %.3f（1）" % st["var_scaled"], size=12, color=COL[d])
    svg.text(58 + i * 222, ty + 58,
             "±2σ 之外 %.2f%%（%.2f%%）" % (100 * st["out2"], 100 * EXACT[d]), size=12, color=COL[d])
svg.text(58, ty + 80, "虚线：推导预言的正态密度。正态分布落在 ±2σ 之外的比例是 4.55%；"
         "d 小时尾巴略重，d 越大越接近。", size=12, color="muted")

svg.save("m04-dot-variance.svg")
for d in DS:
    print(d, {k: round(v, 4) for k, v in stats[d].items()}, "exact out2 %.5f" % EXACT[d])
