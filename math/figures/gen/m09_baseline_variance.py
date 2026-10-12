"""Figure M9.5: the Bernoulli baseline experiment (shared notes N72), in closed form and simulated.

One Bernoulli action y ~ Bern(p), p = 0.3, reward r = 1[y=1] + 10, REINFORCE estimate of dJ/dp:
    g_b = (r - b) (y - p) / (p (1 - p)),   E[g_b] = 1 for every b,
    Var[g_b] = (b - b*)^2 / (p (1 - p)),   b* = 11 - p = 10.7   (author's derivation, m09-notes §2.2).
Left: Var[g_b] against b (log axis) with simulated points (numpy default_rng(0), 2e5 samples).
Right: running mean of g_b over the same sample sequence for b = 0 and b = 10.3, with +-2 sd/sqrt(n) bands.
Writes math/figures/m09-baseline-variance.svg
"""

import math

import numpy as np

from m09_svg import SVG

p = 0.3
bstar = 11 - p
N = 200_000
rng = np.random.default_rng(0)
y = (rng.random(N) < p).astype(float)
r = y + 10.0


def g_of(b):
    return (r - b) * (y - p) / (p * (1 - p))


def var_exact(b):
    return (b - bstar) ** 2 / (p * (1 - p))


sim = {b: (g_of(b).mean(), g_of(b).var()) for b in (0.0, 5.0, 10.0, 10.3, 12.0)}
for b, (m, v) in sim.items():
    assert abs(v - var_exact(b)) / var_exact(b) < 0.02, (b, v)

W, H = 720, 404
svg = SVG(W, H, "Bernoulli 基线实验：方差随基线 b 的变化，以及同一串样本上的滑动平均")

# ---------------- left panel ----------------
LX0, LX1, LY0, LY1 = 70, 330, 40, 300
BMIN, BMAX = 0.0, 14.0
EMIN, EMAX = -2, 3            # 10^-2 .. 10^3


def lx(b):
    return LX0 + (b - BMIN) / (BMAX - BMIN) * (LX1 - LX0)


def ly(v):
    return LY1 - (math.log10(v) - EMIN) / (EMAX - EMIN) * (LY1 - LY0)


svg.text(LX0, 22, "方差 Var[g_{b}]（对数刻度）随基线 b 的变化", size=13, weight="600")
for e in range(EMIN, EMAX + 1):
    yy = ly(10 ** e)
    svg.line(LX0, yy, LX1, yy, color="grid", width=1)
    lab = {-2: "0.01", -1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1000"}[e]
    svg.text(LX0 - 6, yy + 4, lab, size=11, anchor="end", color="muted")
for b in (0, 2, 4, 6, 8, 10, 12, 14):
    xx = lx(b)
    svg.line(xx, LY0, xx, LY1, color="grid", width=1)
    svg.text(xx, LY1 + 16, str(b), size=11, anchor="middle", color="muted")
svg.line(LX0, LY1, LX1, LY1, color="muted", width=1.2)
svg.line(LX0, LY0, LX0, LY1, color="muted", width=1.2)
svg.text((LX0 + LX1) / 2, LY1 + 36, "基线 b", size=12, anchor="middle", color="muted")

# curve, cut where it leaves the plot box (Var < 0.01 near b*), computed here instead of a clipPath
half = math.sqrt(10 ** EMIN * p * (1 - p))          # Var = 0.01  <=>  |b - b*| = half
for lo_b, hi_b in ((BMIN, bstar - half), (bstar + half, BMAX)):
    bs = np.linspace(lo_b, hi_b, 700)
    pts = [(lx(b), ly(max(var_exact(b), 10 ** EMIN))) for b in bs]
    svg.path("M" + " L".join("%.1f %.1f" % pt for pt in pts), stroke="blue", width=2.4)

# b* and the mean reward
xs = lx(bstar)
svg.line(xs, LY0 + 10, xs, LY1, color="green", width=1.4, dash="4 3")
svg.text(xs + 4, LY0 + 20, "b* = 10.7", size=12, color="green", weight="600")
svg.text(xs + 4, LY0 + 36, "方差 = 0", size=12, color="green")

for b, name, dx, dy, anchor in ((0.0, "b = 0：545", 8, -8, "start"), (5.0, "b = 5：155", 8, -6, "start"),
                               (10.0, "b = 10：2.33", -8, -2, "end"), (10.3, "b = 10.3（平均奖励）：0.76", -8, 18, "end"),
                               (12.0, "b = 12：8.05", 9, 20, "start")):   # below-right: the curve rises away from it
    v = var_exact(b)
    svg.circle(lx(b), ly(v), 4, fill="blue")
    svg.circle(lx(b), ly(sim[b][1]), 7, stroke="ink", width=1.2)
    svg.text(lx(b) + dx, ly(v) + dy, name, size=11.5, anchor=anchor)
svg.circle(LX0 + 12, LY1 + 58, 4, fill="blue")
svg.text(LX0 + 22, LY1 + 62, "闭式 (b−b*)²/(p(1−p))", size=12, color="muted")
svg.circle(LX0 + 170, LY1 + 58, 7, stroke="ink", width=1.2)
svg.text(LX0 + 182, LY1 + 62, "模拟 2×10⁵ 次", size=12, color="muted")

# ---------------- right panel ----------------
RX0, RX1, RY0, RY1 = 420, 700, 40, 300
NMIN, NMAX = 10, 10_000
YMIN, YMAX = -4.0, 6.0


def rx(n):
    return RX0 + (math.log10(n) - math.log10(NMIN)) / (math.log10(NMAX) - math.log10(NMIN)) * (RX1 - RX0)


def ry(v):
    return RY1 - (v - YMIN) / (YMAX - YMIN) * (RY1 - RY0)


svg.text(RX0 - 14, 22, "同一串样本的滑动平均（真值 1）", size=13, weight="600")
for v in (-4, -2, 0, 1, 2, 4, 6):
    yy = ry(v)
    svg.line(RX0, yy, RX1, yy, color="grid", width=1)
    svg.text(RX0 - 6, yy + 4, ("%d" % v).replace("-", "−"), size=11, anchor="end", color="muted")
for n, lab in ((10, "10"), (100, "100"), (1000, "1000"), (10000, "10⁴")):
    xx = rx(n)
    svg.line(xx, RY0, xx, RY1, color="grid", width=1)
    svg.text(xx, RY1 + 16, lab, size=11, anchor="middle", color="muted")
svg.line(RX0, RY1, RX1, RY1, color="muted", width=1.2)
svg.line(RX0, RY0, RX0, RY1, color="muted", width=1.2)
svg.text((RX0 + RX1) / 2, RY1 + 36, "样本数 n（对数刻度）", size=12, anchor="middle", color="muted")

ns = np.unique(np.round(np.logspace(math.log10(NMIN), math.log10(NMAX), 240)).astype(int))


def clip_poly(pts):
    """Split a polyline where it leaves [YMIN, YMAX]; interpolate the crossing points."""
    out, cur = [], []
    for (x0, v0), (x1, v1) in zip(pts[:-1], pts[1:]):
        inside0 = YMIN <= v0 <= YMAX
        inside1 = YMIN <= v1 <= YMAX
        if inside0:
            if not cur:
                cur.append((x0, v0))
        if inside0 and inside1:
            cur.append((x1, v1))
        elif inside0 and not inside1:
            lim = YMAX if v1 > YMAX else YMIN
            t = (lim - v0) / (v1 - v0)
            cur.append((x0 + t * (x1 - x0), lim))
            out.append(cur)
            cur = []
        elif not inside0 and inside1:
            lim = YMAX if v0 > YMAX else YMIN
            t = (lim - v0) / (v1 - v0)
            cur = [(x0 + t * (x1 - x0), lim), (x1, v1)]
    if cur:
        out.append(cur)
    return out


for b, col, sd_lab in ((0.0, "muted", None), (10.3, "blue", None)):
    g = g_of(b)
    run = np.cumsum(g)[ns - 1] / ns
    sd = math.sqrt(var_exact(b))
    upper = [1 + 2 * sd / math.sqrt(n) for n in ns]
    lower = [1 - 2 * sd / math.sqrt(n) for n in ns]
    up = [(rx(n), min(max(u, YMIN), YMAX)) for n, u in zip(ns, upper)]
    lo = [(rx(n), min(max(l_, YMIN), YMAX)) for n, l_ in zip(ns, lower)]
    poly = up + lo[::-1]
    svg.polygon([(x, ry(v)) for x, v in poly], fill=col, fill_opacity=0.12)
    for seg in clip_poly(list(zip([rx(n) for n in ns], run))):
        svg.path("M" + " L".join("%.1f %.1f" % (x, ry(v)) for x, v in seg), stroke=col, width=1.8)
svg.line(RX0, ry(1), RX1, ry(1), color="ink", width=1.2, dash="5 4")
# below the dashed line at the right end: both running means sit on 1 there, the band is light
svg.text(RX1 - 4, ry(1) + 22, "真值 1", size=12, anchor="end", weight="600")
svg.line(RX0 - 10, RY1 + 58, RX0 + 14, RY1 + 58, color="muted", width=1.8)
svg.text(RX0 + 20, RY1 + 62, "b = 0", size=12)
svg.line(RX0 + 70, RY1 + 58, RX0 + 94, RY1 + 58, color="blue", width=1.8)
svg.text(RX0 + 100, RY1 + 62, "b = 10.3", size=12, color="blue", weight="600")
svg.rect(RX0 + 170, RY1 + 51, 22, 14, fill="muted", fill_opacity=0.18)
svg.text(RX0 + 198, RY1 + 62, "±2 个标准误", size=12)
m0, v0 = sim[0.0]
m1, v1 = sim[10.3]
svg.text(RX0 - 10, RY1 + 86, "2×10⁵ 次后：b = 0 均值 %.3f，b = 10.3 均值 %.4f" % (m0, m1), size=12, color="muted")

svg.save("m09-baseline-variance.svg")
