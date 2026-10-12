"""Figure M4.5: the sum of n fair dice. Mean grows like n, spread only like sqrt(n).

Left: exact distributions (repeated convolution) for n = 1, 2, 4, 8 with mean +- one sd.
Right: sd of the sum against n: independent dice sqrt(n)*sigma vs "all dice equal" n*sigma.
Writes math/figures/m04-dice-sums.svg
"""

import math

import numpy as np

from m04_svg import SVG, Axes

die = np.ones(6) / 6
sigma = math.sqrt(35 / 12)


def pmf_sum(n):
    p = np.array([1.0])
    for _ in range(n):
        p = np.convolve(p, die)
    return p  # index k -> sum n + k


W, H = 720, 360
svg = SVG(W, H, "n 颗骰子的点数和：均值按 n 增长，标准差只按根号 n 增长")

ax = Axes(svg, 60, 46, 380, 230, (0, 50), (0, 0.18))
ax.frame(xticks=[0, 10, 20, 30, 40, 50], yticks=[0, 0.05, 0.10, 0.15], xlabel="点数和",
         ylabel="概率", ylabel_dx=-42)
svg.text(ax.x0, 24, "n 颗公平骰子之和的分布（精确卷积）", size=13, weight="600")

colors = {1: "muted", 2: "green", 4: "blue", 8: "violet"}
rows = []
for n, col in colors.items():
    p = pmf_sum(n)
    xs = np.arange(n, 6 * n + 1)
    mean = float((xs * p).sum())
    sd = float(math.sqrt(((xs - mean) ** 2 * p).sum()))
    assert abs(mean - 3.5 * n) < 1e-9 and abs(sd - sigma * math.sqrt(n)) < 1e-9
    rows.append((n, mean, sd, p.max()))
    pts = []
    for x, v in zip(xs, p):
        pts += [ax.P(x - 0.5, v), ax.P(x + 0.5, v)]
    pts = [ax.P(xs[0] - 0.5, 0)] + pts + [ax.P(xs[-1] + 0.5, 0)]
    svg.polygon(pts, fill=col, fill_opacity=0.10, stroke=col, width=1.6)
    # mean +- sd bracket above the peak (n = 1 and n = 2 share a peak height, so stagger them)
    yb = ax.Y(p.max()) - {1: 8, 2: 20, 4: 10, 8: 10}[n]
    svg.line(ax.X(mean - sd), yb, ax.X(mean + sd), yb, color=col, width=2)
    svg.line(ax.X(mean - sd), yb - 4, ax.X(mean - sd), yb + 4, color=col, width=2)
    svg.line(ax.X(mean + sd), yb - 4, ax.X(mean + sd), yb + 4, color=col, width=2)
    svg.circle(ax.X(mean), yb, 3, fill=col)

# legend in the empty upper-right corner of the panel
lx, ly = ax.X(32), ax.Y(0.172)
for i, (n, mean, sd, pk) in enumerate(rows):
    y = ly + i * 19
    svg.rect(lx, y - 10, 14, 11, fill=colors[n], fill_opacity=0.25, stroke=colors[n], width=1.2)
    svg.text(lx + 20, y, "n=%d：μ=%g，σ=%.2f" % (n, mean, sd), size=12, color=colors[n])
svg.text(ax.x0, ax.Y(0) + 52, "横线 = 均值 ± 1 个标准差；分布越来越宽、越来越像钟形，但宽得比均值慢",
         size=12, color="muted")

# right: sd vs n
bx = Axes(svg, 520, 46, 180, 230, (0, 16), (0, 28))
bx.frame(xticks=[0, 4, 8, 12, 16], yticks=[0, 7, 14, 21, 28], xlabel="骰子个数 n",
         ylabel="和的标准差", ylabel_dx=-34)
svg.text(bx.x0 - 20, 24, "标准差随 n 的增长", size=13, weight="600")
ns = np.linspace(0, 16, 81)
bx.curve(ns, sigma * ns, stroke="red", width=1.6, dash="5 4")
bx.curve(ns, sigma * np.sqrt(ns), stroke="blue", width=2.2)
for n in (1, 2, 4, 8, 16):
    svg.circle(*bx.P(n, sigma * math.sqrt(n)), 3.2, fill="blue")
svg.text(bx.X(0.6), bx.Y(25.5), "n·σ：若各骰子", size=12, color="red")
svg.text(bx.X(0.6), bx.Y(23.2), "总是同一个点数", size=12, color="red")
svg.text(bx.X(8.5), bx.Y(1.6), "√n·σ：独立", size=12, color="blue")
svg.text(bx.X(16), bx.Y(sigma * 4) - 8, "6.83", size=11, anchor="end", color="blue")
svg.text(bx.x0 - 20, bx.Y(0) + 52, "σ = √(35/12) ≈ 1.708", size=12, color="muted")

svg.save("m04-dice-sums.svg")
for r in rows:
    print("n=%d mean %.1f sd %.4f peak %.4f" % r)
print("sd n=16", sigma * 4)
