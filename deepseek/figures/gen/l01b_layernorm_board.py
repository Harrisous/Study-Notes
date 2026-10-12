"""Figure 1.15 of the blackboard version of lesson 1: three 'exam papers' through LayerNorm.

Inputs (2,4,6,8), (20,40,60,80) = 10x and (102,104,106,108) = +100, d = 4, gamma = 1, beta = 0.
Each input is drawn on its own number line (its own scale) with the mean and one standard
deviation marked; all three map to the same output (-1.3416, -0.4472, 0.4472, 1.3416), whose
length is 2 = sqrt(4). Everything is computed here with formula (1.12) (epsilon = 1e-5).
Writes deepseek/figures/l01b-layernorm-board.svg
"""

import math

from l01_svg import SVG

EPS = 1e-5
INPUTS = [((2, 4, 6, 8), (0, 10), "原答卷"), ((20, 40, 60, 80), (0, 100), "整体放大 10 倍"),
          ((102, 104, 106, 108), (100, 110), "整体加 100 分")]


def ln(x):
    mu = sum(x) / len(x)
    var = sum((v - mu) ** 2 for v in x) / len(x)
    return mu, var, [(v - mu) / math.sqrt(var + EPS) for v in x]


outs = []
for x, _, _ in INPUTS:
    mu, var, y = ln(x)
    outs.append(y)
    assert all(abs(a - b) < 1e-4 for a, b in zip(y, (-1.3416, -0.4472, 0.4472, 1.3416))), y
y0 = outs[0]
assert abs(math.sqrt(sum(v * v for v in y0)) - 2) < 1e-5

W, H = 760, 388
svg = SVG(W, H, "黑板上的 LayerNorm：三份答卷落到同一个点")

AX0, AX1 = 250, 600          # pixel extent of every number line


def axis(y, lo, hi, ticks, color="ink"):
    svg.line(AX0 - 10, y, AX1 + 10, y, color="muted", width=1.2)
    for tv in ticks:
        px = AX0 + (tv - lo) / (hi - lo) * (AX1 - AX0)
        svg.line(px, y - 4, px, y + 4, color="muted", width=1)
        svg.text(px, y + 18, ("%g" % tv).replace("-", "−"), size=11, anchor="middle", color="muted")


def pos(v, lo, hi):
    return AX0 + (v - lo) / (hi - lo) * (AX1 - AX0)


svg.text(10, 22, "三份 4 维“答卷”，各画在自己的数轴上（刻度不同）", size=13, weight="700")
for k, (x, (lo, hi), name) in enumerate(INPUTS):
    y = 72 + k * 78
    mu, var, _ = ln(x)
    sd = math.sqrt(var)
    svg.text(10, y - 4, name, size=12.5, weight="600")
    svg.text(10, y + 14, "x = (%s)" % ", ".join(str(v) for v in x), size=12, color="blue")
    step = (hi - lo) / 5
    axis(y, lo, hi, [lo + i * step for i in range(6)])
    # one standard deviation band around the mean
    svg.rect(pos(mu - sd, lo, hi), y - 12, pos(mu + sd, lo, hi) - pos(mu - sd, lo, hi), 24,
             fill="blue", fill_opacity=0.08, rx=3)
    svg.line(pos(mu, lo, hi), y - 16, pos(mu, lo, hi), y + 8, color="ink", width=1.6)
    svg.text(pos(mu, lo, hi), y - 20, "μ", size=12, anchor="middle")
    for v in x:
        svg.circle(pos(v, lo, hi), y, 5.5, fill="blue", stroke="blue", width=1)
    svg.text(AX1 + 22, y - 2, "μ = %g" % mu, size=12)
    svg.text(AX1 + 22, y + 15, "σ² = %g，σ = %.4g" % (var, sd), size=12, color="muted")

# the operation
oy = 72 + 3 * 78 - 20
svg.arrow((AX0 + AX1) / 2, oy - 18, (AX0 + AX1) / 2, oy + 22, color="ink", width=1.8)
svg.text((AX0 + AX1) / 2 + 14, oy + 6, "减均值 μ，再除以标准差 √(σ² + ε)", size=12.5, weight="600")

# the common output
y = oy + 62
svg.text(10, y - 4, "三份的输出", size=12.5, weight="600", color="green")
svg.text(10, y + 14, "完全一样", size=12, color="green")
axis(y, -2, 2, [-2, -1, 0, 1, 2])
for v in y0:
    px = pos(v, -2, 2)
    svg.circle(px, y, 6, fill="green", stroke="green", width=1)
    svg.text(px, y - 12, ("%.4f" % v).replace("-", "−"), size=11.5, anchor="middle", color="green")
svg.text(AX1 + 22, y - 2, "均值 0", size=12)
svg.text(AX1 + 22, y + 15, "长度 = 2 = √4", size=12, color="green", weight="600")
svg.save("l01b-layernorm-board.svg")
