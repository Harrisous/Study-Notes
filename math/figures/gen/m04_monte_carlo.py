"""Figure M4.11: Monte Carlo and the sample variance.

Left: running mean of fair-die rolls, 6 independent runs (numpy default_rng(7)), N = 1..10^4 on
a log axis, inside the band 3.5 +- 2 sigma/sqrt(N), sigma^2 = 35/12.
Right: the exact distribution (all 6^4 = 1296 equally likely outcomes) of two variance
estimators from n = 4 rolls: divide by n (biased, mean 3/4 sigma^2) and by n - 1 (unbiased).
Writes math/figures/m04-monte-carlo.svg
"""

import itertools
import math
from collections import Counter
from fractions import Fraction

import numpy as np

from m04_svg import SVG, Axes

SIG2 = 35 / 12
SIG = math.sqrt(SIG2)

# ---- exact sampling distribution of the variance estimators (n = 4)
ss_counts = Counter()
for t in itertools.product(range(1, 7), repeat=4):
    m = Fraction(sum(t), 4)
    ss = sum((Fraction(x) - m) ** 2 for x in t)
    ss_counts[ss] += 1
tot = sum(ss_counts.values())
E_n = sum(float(ss) / 4 * c for ss, c in ss_counts.items()) / tot
E_n1 = sum(float(ss) / 3 * c for ss, c in ss_counts.items()) / tot
P_below_n1 = sum(c for ss, c in ss_counts.items() if float(ss) / 3 < SIG2) / tot
print("E[SS/4] %.4f  E[SS/3] %.4f  sigma2 %.4f  P(SS/3 < sigma2) %.4f" % (E_n, E_n1, SIG2, P_below_n1))

# ---- running means
rng = np.random.default_rng(7)
NMAX = 10_000
runs = rng.integers(1, 7, size=(6, NMAX))
means = np.cumsum(runs, axis=1) / np.arange(1, NMAX + 1)
print("final means", np.round(means[:, -1], 4), "2se at 1e4", 2 * SIG / 100)

W, H = 720, 330
svg = SVG(W, H, "蒙特卡洛：样本均值收敛的速度，以及样本方差为什么除以 n−1")

A = Axes(svg, 60, 48, 330, 230, (1, NMAX), (1, 6), logx=True)
A.frame(xticks=[1, 10, 100, 1000, 10000], yticks=[1, 2, 3, 3.5, 4, 5, 6], xlabel="样本数 N（对数轴）",
        ylabel="前 N 次的平均点数", ylabel_dx=-40,
        xfmt=lambda v: {1: "1", 10: "10", 100: "100", 1000: "1000", 10000: "10⁴"}[v])
svg.text(A.x0, 26, "6 次独立的“掷骰子求平均”", size=13, weight="600")
Ns = np.unique(np.round(np.logspace(0, 4, 240)).astype(int))
upper = [min(6, 3.5 + 2 * SIG / math.sqrt(n)) for n in Ns]
lower = [max(1, 3.5 - 2 * SIG / math.sqrt(n)) for n in Ns]
band = [A.P(n, u) for n, u in zip(Ns, upper)] + [A.P(n, l) for n, l in zip(Ns[::-1], lower[::-1])]
svg.polygon(band, fill="green", fill_opacity=0.12)
A.curve(Ns, upper, stroke="green", width=1.2, dash="4 3")
A.curve(Ns, lower, stroke="green", width=1.2, dash="4 3")
svg.line(A.x0, A.Y(3.5), A.x0 + A.w, A.Y(3.5), color="ink", width=1.2)
for r in range(6):
    A.curve(Ns, means[r, Ns - 1], stroke="blue", width=1.2, opacity=0.75)
svg.text(A.X(40), A.Y(5.6), "绿带：3.5 ± 2σ/√N", size=12, color="green")
svg.text(A.X(300), A.Y(4.45), "N = 100：±0.34", size=12, color="green")
svg.text(A.X(1500), A.Y(3.85), "10⁴：±0.034", size=12, color="green")

# ---- right: histogram of the two estimators
B = Axes(svg, 450, 48, 250, 230, (0, 9), (0, 0.30))
B.frame(xticks=[0, 2, 4, 6, 8], yticks=[0, 0.1, 0.2, 0.3], xlabel="方差的估计值",
        ylabel="概率（每格宽 0.5）", ylabel_dx=-36)
svg.text(B.x0 - 20, 26, "4 次投掷估计方差（精确分布）", size=13, weight="600")
edges = np.arange(0, 9.01, 0.5)


def hist(div):
    h = np.zeros(len(edges) - 1)
    for ss, c in ss_counts.items():
        v = float(ss) / div
        i = min(int(v // 0.5), len(h) - 1)
        h[i] += c / tot
    return h


for div, col, fill in ((3, "blue", 0.30), (4, "orange", 0.0)):
    h = hist(div)
    pts = [B.P(edges[0], 0)]
    for i, v in enumerate(h):
        pts += [B.P(edges[i], v), B.P(edges[i + 1], v)]
    pts.append(B.P(edges[-1], 0))
    if fill:
        svg.polygon(pts, fill=col, fill_opacity=fill, stroke=col, width=1.3)
    else:
        svg.polyline(pts, stroke=col, width=1.8)
for v, col, lab, dy in ((E_n, "orange", "除以 n：均值 %.4f" % E_n, 0), (E_n1, "blue", "除以 n−1：均值 %.4f = σ²" % E_n1, 18)):
    svg.line(B.X(v), B.Y(0), B.X(v), B.Y(0.30), color=col, width=1.6, dash="5 3")
    svg.text(B.X(3.2), B.Y(0.27) + dy, lab, size=12, color=col)
svg.text(B.X(3.2), B.Y(0.27) + 38, "（真值 σ² = 35/12 ≈ 2.917）", size=12, color="ink")

svg.save("m04-monte-carlo.svg")
