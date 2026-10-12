"""Figure M5.10: how good are k1, k2, k3? Exact bias and spread as a function of the true KL(q||p).

Two families, both computed exactly (no sampling):
  * "vocabulary": V = 1000 tokens, q = softmax(z), p = softmax(z + s*g), z ~ 2*N(0,1), g ~ N(0,1) (seed 0),
    s swept so that KL(q||p) runs from about 1e-3 to 2.5; expectations are sums over the 1000 tokens.
  * "Gaussian": q = N(0,1), p = N(mu,1), KL = mu^2/2, closed forms (Lesson 8, research notes C21):
    rel.std k1 = 2/mu, k2 = sqrt(2 + mu^2), k3 = 2 sqrt(e^{mu^2} - 1 - mu^2)/mu^2; bias(k2)/KL = mu^2/4.
Markers: Schulman's blog table (mu = 0.1 and 1): rel.std 20 / 1.42 / 1.42 and 2 / 1.73 / 1.7.
Small-KL limits: k1 ~ sqrt(2/KL); k2, k3 -> sqrt(kappa - 1), kappa = kurtosis of w - 1 under q (w = p/q).
Writes math/figures/m05-estimator-compare.svg
"""

import math

import numpy as np

from m05_svg import SVG

rng = np.random.default_rng(0)
V = 1000
z = rng.standard_normal(V) * 2.0
g = rng.standard_normal(V)


def softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


q = softmax(z)
gc = g - (q * g).sum()
kappa = float((q * gc ** 4).sum() / (q * gc ** 2).sum() ** 2)
print("H(q) = %.3f nats, kappa = %.3f, sqrt(kappa-1) = %.3f" % (float(-(q * np.log(q)).sum()), kappa,
                                                                  math.sqrt(kappa - 1)))

disc = []                                   # (KL, relstd1, relstd2, relstd3, relbias2)
for s in np.logspace(math.log10(0.03), math.log10(2.0), 60):
    p = softmax(z + s * g)
    lr = np.log(p / q)                      # ln r
    r = p / q
    kl = float((q * -lr).sum())
    row = [kl]
    ests = (-lr, 0.5 * lr ** 2, (r - 1) - lr)
    for k in ests:
        m = float((q * k).sum())
        row.append(math.sqrt(float((q * (k - m) ** 2).sum())) / kl)
    row.append(float((q * ests[1]).sum()) / kl - 1)
    disc.append(row)
for row in disc[::12]:
    print("vocab KL %.4f  rel.std k1 %.2f k2 %.3f k3 %.3f  bias k2 %+.4f" % tuple(row))

gauss = []
for mu in np.logspace(math.log10(0.045), math.log10(2.2), 80):
    kl = mu * mu / 2
    gauss.append((kl, 2 / mu, math.sqrt(2 + mu * mu), 2 * math.sqrt(math.exp(mu * mu) - 1 - mu * mu) / (mu * mu),
                  mu * mu / 4))

# ------------------------------------------------------------------ drawing
W, H = 760, 420
svg = SVG(W, H, "k1、k2、k3 的偏差与相对标准差（精确计算）")
KX0, KX1 = -3, math.log10(3)


def frame(x0, x1, y0, y1, ylo, yhi, ylog, yticks, title, ylab):
    def PX(k):
        return x0 + (math.log10(k) - KX0) / (KX1 - KX0) * (x1 - x0)

    def PY(v):
        t = (math.log10(v) - ylo) / (yhi - ylo) if ylog else (v - ylo) / (yhi - ylo)
        return y1 - t * (y1 - y0)
    for k, lab in ((1e-3, "0.001"), (1e-2, "0.01"), (1e-1, "0.1"), (1, "1")):
        svg.line(PX(k), y0, PX(k), y1, color="grid", width=1)
        svg.text(PX(k), y1 + 16, lab, size=11, anchor="middle", color="muted")
    for v, lab in yticks:
        svg.line(x0, PY(v), x1, PY(v), color="grid", width=1)
        svg.text(x0 - 6, PY(v) + 4, lab, size=11, anchor="end", color="muted")
    svg.line(x0, y1, x1, y1, color="muted", width=1.2)
    svg.line(x0, y0, x0, y1, color="muted", width=1.2)
    svg.text((x0 + x1) / 2, y0 - 14, title, size=13, anchor="middle", weight="600")
    svg.text((x0 + x1) / 2, y1 + 36, "真实 KL(q‖p)（nats，对数刻度）", size=12, anchor="middle", color="muted")
    svg.text(x0 - 50, (y0 + y1) / 2, ylab, size=12, anchor="middle", color="muted", rotate=-90)
    return PX, PY


# (a) relative standard deviation, log-log
PX, PY = frame(80, 420, 40, 340, math.log10(0.5), 2, True,
               ((0.5, "0.5"), (1, "1"), (2, "2"), (5, "5"), (10, "10"), (20, "20"), (50, "50"), (100, "100")),
               "(a) 单样本相对标准差 = 标准差 ÷ 真值", "相对标准差（对数刻度）")


def poly(rows, idx, col, dash=None):
    pts = [(PX(r[0]), PY(r[idx])) for r in rows if 1e-3 <= r[0] <= 3 and 0.5 <= r[idx] <= 100]
    svg.path("M" + " L".join("%.1f %.1f" % xy for xy in pts), stroke=col, width=2.3 if not dash else 1.6, dash=dash)


for idx, col in ((1, "red"), (2, "amber"), (3, "blue")):
    poly(gauss, idx, col, dash="5 4")
    poly(disc, idx, col)
for kl, vals in ((0.005, (20, 1.42, 1.42)), (0.5, (2, 1.73, 1.7))):
    for v, col in zip(vals, ("red", "amber", "blue")):
        svg.circle(PX(kl), PY(v), 4.2, fill="surface", stroke=col, width=1.8)
svg.text(PX(0.0012), PY(60), "k1：≈ √(2/KL)，KL 越小越糟", size=12, color="red")
svg.text(PX(0.0012), PY(0.95), "k2、k3：→ 常数（高斯 √2 = 1.41，词表例 %.2f）" % math.sqrt(kappa - 1), size=12,
         color="blue")
svg.text(PX(0.0012), PY(3.3), "KL ≳ 0.73 时两族的排序相反：", size=12, color="muted")
svg.text(PX(0.0012), PY(3.3) + 16, "高斯例 k3 更散，词表例 k3 更稳", size=12, color="muted")

# (b) relative bias of k2
PXb, PYb = frame(520, 740, 40, 340, -0.05, 1.05, False,
                 ((0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1")),
                 "(b) k2 的相对偏差", "E[k2] ÷ KL − 1")


def poly_b(rows, col, dash=None):
    pts = [(PXb(r[0]), PYb(r[4])) for r in rows if 1e-3 <= r[0] <= 3 and -0.05 <= r[4] <= 1.05]
    svg.path("M" + " L".join("%.1f %.1f" % xy for xy in pts), stroke=col, width=2.3 if not dash else 1.6, dash=dash)


svg.line(520, PYb(0), 740, PYb(0), color="blue", width=1.4)
poly_b(gauss, "amber", dash="5 4")
poly_b(disc, "amber")
# Schulman's table prints bias/true 0.002 for k2 at mu = 0.1 (closed form mu^2/4 = 0.0025) and 0.25 at mu = 1
svg.circle(PXb(0.005), PYb(0.002), 4.2, fill="surface", stroke="amber", width=1.8)
svg.circle(PXb(0.5), PYb(0.25), 4.2, fill="surface", stroke="amber", width=1.8)
svg.text(PXb(0.0012), PYb(0.1), "k1、k3：恒为 0（无偏）", size=12, color="blue")
svg.text(PXb(0.0012), PYb(0.93), "高斯：= KL/2", size=12, color="amber")
svg.text(PXb(0.0012), PYb(0.93) + 16, "词表例：先略负后转正", size=12, color="amber")

# legend
LY = 400
svg.line(80, LY, 104, LY, color="ink", width=2.3)
svg.text(110, LY + 4, "实线：1000 词表例（精确求和）", size=12)
svg.line(330, LY, 354, LY, color="ink", width=1.6, dash="5 4")
svg.text(360, LY + 4, "虚线：高斯例闭式", size=12)
svg.circle(500, LY, 4.2, fill="surface", stroke="ink", width=1.8)
svg.text(510, LY + 4, "圆圈：Schulman 博客实验表", size=12)

svg.save("m05-estimator-compare.svg")
