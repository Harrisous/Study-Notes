"""Figure M5.7: fitting one Gaussian q = N(mu, sigma^2) to a two-peak target p by forward vs reverse KL.

Target p = 0.5 N(-2, 0.6^2) + 0.5 N(2, 0.6^2) (the setting of Minka 2005, Fig. 1; the numbers here are ours).
Forward KL(p||q) is minimised by moment matching: mu = E_p[x] = 0, sigma^2 = Var_p(x) = 0.36 + 4.
Reverse KL(q||p) is minimised numerically; it has two mirror-image minima near +-2.
Panel (c) plots min over sigma of each divergence as a function of mu.
Everything is integrated numerically here (trapezoid rule on [-12, 12], 24001 points).
Writes math/figures/m05-forward-reverse.svg
"""

import math

import numpy as np

from m05_svg import SVG

M, S = 2.0, 0.6
x = np.linspace(-12, 12, 24001)
dx = x[1] - x[0]


def npdf(t, m, s):
    return np.exp(-0.5 * ((t - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def trap(y):
    return float((y[:-1] + y[1:]).sum() * dx / 2)


p = 0.5 * npdf(x, -M, S) + 0.5 * npdf(x, M, S)
logp = np.log(np.maximum(p, 1e-300))


def fkl(mu, sg):              # KL(p || q)
    logq = -0.5 * ((x - mu) / sg) ** 2 - math.log(sg * math.sqrt(2 * math.pi))
    return trap(p * (logp - logq))


def rkl(mu, sg):              # KL(q || p)
    q = npdf(x, mu, sg)
    logq = -0.5 * ((x - mu) / sg) ** 2 - math.log(sg * math.sqrt(2 * math.pi))
    return trap(q * (logq - logp))


def golden(f, a, b, it=60):
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(it):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = f(d)
    m = (a + b) / 2
    return m, f(m)


# forward optimum (closed form) and check
mu_f = trap(x * p)
sg_f = math.sqrt(trap((x - mu_f) ** 2 * p))
assert abs(sg_f - math.sqrt(S * S + M * M)) < 1e-6
F_f, R_f = fkl(mu_f, sg_f), rkl(mu_f, sg_f)

# landscape over mu (sigma optimised for each mu)
mus = np.linspace(-3.2, 3.2, 129)
land_r, land_f = [], []
for mu in mus:
    land_r.append(golden(lambda s: rkl(mu, s), 0.15, 5.0)[1])
    land_f.append(golden(lambda s: fkl(mu, s), 0.15, 8.0)[1])
# reverse optimum: golden search on the profile (sigma optimised) around the best grid mu on the left branch
i0 = int(np.argmin(land_r[:65]))
mu_r, R_r = golden(lambda m: golden(lambda s: rkl(m, s), 0.15, 5.0)[1], mus[i0] - 0.05, mus[i0] + 0.05)
sg_r = golden(lambda s: rkl(mu_r, s), 0.15, 5.0)[0]
F_r = fkl(mu_r, sg_r)
bump = land_r[64]
print("forward fit mu %.4f sigma %.4f  KL(p||q) %.4f  KL(q||p) %.4f" % (mu_f, sg_f, F_f, R_f))
print("reverse fit mu %.4f sigma %.4f  KL(q||p) %.4f  KL(p||q) %.4f" % (mu_r, sg_r, R_r, F_r))
print("reverse at mu=0 (sigma optimised): %.4f; ln 2 = %.4f" % (bump, math.log(2)))
# local maxima of the reverse profile between the minima (a shallow local minimum sits at mu = 0)
lr = np.array(land_r)
for i in range(1, len(mus) - 1):
    if lr[i] > lr[i - 1] and lr[i] > lr[i + 1]:
        print("reverse profile local max at mu = %.2f: %.4f" % (mus[i], lr[i]))
    if lr[i] < lr[i - 1] and lr[i] < lr[i + 1]:
        print("reverse profile local min at mu = %.2f: %.4f (sigma %.3f)"
              % (mus[i], lr[i], golden(lambda s: rkl(mus[i], s), 0.15, 5.0)[0]))

# ------------------------------------------------------------------ drawing
W, H = 760, 540
svg = SVG(W, H, "前向 KL 与反向 KL：用一个高斯去拟合双峰分布")
XL, XR = -5.0, 5.0
DMAX = 0.7


def dens_panel(x0, x1, y0, y1, qmu, qsg, col, title, sub):
    def PX(t):
        return x0 + (t - XL) / (XR - XL) * (x1 - x0)

    def PY(v):
        return y1 - v / DMAX * (y1 - y0)
    for t in range(-4, 5, 2):
        svg.line(PX(t), y0, PX(t), y1, color="grid", width=1)
        svg.text(PX(t), y1 + 15, str(t).replace("-", "−"), size=11, anchor="middle", color="muted")
    svg.line(x0, y1, x1, y1, color="muted", width=1.2)
    sel = (x >= XL) & (x <= XR)
    xs, ps = x[sel][::20], p[sel][::20]
    d = "M%.1f %.1f " % (PX(xs[0]), PY(0)) + " ".join("L%.1f %.1f" % (PX(a), PY(b)) for a, b in zip(xs, ps)) + \
        " L%.1f %.1f Z" % (PX(xs[-1]), PY(0))
    svg.path(d, stroke="ink", width=1.6, fill="muted", fill_opacity=0.18)
    qs = npdf(xs, qmu, qsg)
    svg.path("M" + " L".join("%.1f %.1f" % (PX(a), PY(min(b, DMAX))) for a, b in zip(xs, qs)), stroke=col, width=2.6)
    svg.text((x0 + x1) / 2, y0 - 22, title, size=13, anchor="middle", weight="600", color=col)
    svg.text((x0 + x1) / 2, y0 - 6, sub, size=12, anchor="middle", color="muted")
    return PX, PY


PXa, PYa = dens_panel(30, 360, 60, 240, mu_f, sg_f, "blue", "(a) 最小化前向 KL(p‖q)：盖住两个峰",
                      "q = N(0, %.3f²)，均值与方差对上 p" % sg_f)
svg.text(PXa(0), PYa(0.235), "谷底也给了概率", size=12, anchor="middle", color="blue")
svg.text(PXa(-4.8), PYa(0.62), "灰色填充：目标 p", size=12, color="muted")

PXb, PYb = dens_panel(400, 730, 60, 240, mu_r, sg_r, "violet", "(b) 最小化反向 KL(q‖p)：只挑一个峰",
                      "q = N(%s, %.3f²)，另一个峰完全不管" % (("%.3f" % mu_r).replace("-", "−"), sg_r))
svg.text(PXb(2), PYb(0.40), "被放弃的峰", size=12, anchor="middle", color="muted")

# numbers table
TY = 282
svg.text(30, TY, "两个解在两种 KL 下的得分（nats）", size=12, weight="600")
rows = [("", "KL(p‖q)", "KL(q‖p)"),
        ("(a) 前向解", "%.3f（最小）" % F_f, "%.3f" % R_f),
        ("(b) 反向解", "%.2f" % F_r, "%.3f（最小）" % R_r)]
for i, (a_, b_, c_) in enumerate(rows):
    y = TY + 20 + i * 18
    col = ["ink", "blue", "violet"][i]
    svg.text(30, y, a_, size=12, color=col, weight="600" if i else None)
    svg.text(160, y, b_, size=12, color=col if i else "muted")
    svg.text(280, y, c_, size=12, color=col if i else "muted")

# ---- (c) landscape
x0, x1, y0, y1 = 430, 730, 300, 490
LMAX = 3.0


def PX(m):
    return x0 + (m + 3.2) / 6.4 * (x1 - x0)


def PY(v):
    return y1 - min(v, LMAX) / LMAX * (y1 - y0)


for m in range(-3, 4):
    svg.line(PX(m), y0, PX(m), y1, color="grid", width=1)
    svg.text(PX(m), y1 + 15, str(m).replace("-", "−"), size=11, anchor="middle", color="muted")
for v in range(0, 4):
    svg.line(x0, PY(v), x1, PY(v), color="grid", width=1)
    svg.text(x0 - 6, PY(v) + 4, str(v), size=11, anchor="end", color="muted")
svg.line(x0, y1, x1, y1, color="muted", width=1.2)
svg.line(x0, y0, x0, y1, color="muted", width=1.2)
svg.text((x0 + x1) / 2, y0 - 10, "(c) 固定 μ、对 σ 取最优后的 KL", size=13, anchor="middle", weight="600")
svg.text((x0 + x1) / 2, y1 + 32, "q 的均值 μ", size=12, anchor="middle", color="muted")
def curve(vals, col):
    seg = []
    for m, v in zip(mus, vals):
        if v <= LMAX:
            seg.append((PX(m), PY(v)))
        elif seg:
            svg.path("M" + " L".join("%.1f %.1f" % xy for xy in seg), stroke=col, width=2.4)
            seg = []
    if seg:
        svg.path("M" + " L".join("%.1f %.1f" % xy for xy in seg), stroke=col, width=2.4)


curve(land_f, "blue")
curve(land_r, "violet")
svg.circle(PX(mu_r), PY(R_r), 4.3, fill="violet")
svg.circle(PX(-mu_r), PY(R_r), 4.3, fill="violet")
svg.circle(PX(0), PY(F_f), 4.3, fill="blue")
svg.text(PX(0), PY(2.75), "紫＝反向：μ ≈ ±2 两个极小（%.3f）" % R_r, size=12, anchor="middle", color="violet")
svg.text(PX(0), PY(2.75) + 17, "μ = 0 处另有一个浅的局部极小（%.2f）" % bump, size=12, anchor="middle",
         color="violet")
svg.text(PX(0), PY(2.75) + 34, "蓝＝前向：只有 μ = 0 一个极小（%.3f）" % F_f, size=12, anchor="middle", color="blue")

# explanation under the table
EY = 370
notes = [("前向 KL(p‖q) = E_p[ln p/q]：", "blue"),
         ("  p 有质量而 q 很小的地方罚得最重 → 宁可铺开", "ink"),
         ("  （质量覆盖，Minka 称 inclusive）", "muted"),
         ("反向 KL(q‖p) = E_q[ln q/p]：", "violet"),
         ("  q 有质量而 p 很小的地方罚得最重 → 宁可收缩", "ink"),
         ("  （只挑一个峰，Minka 称 exclusive / zero-forcing）", "muted"),
         ("反向解离 ln 2 = 0.693 很近：q 所在的峰只占 p 的一半", "ink")]
for i, (s, col) in enumerate(notes):
    svg.text(30, EY + i * 19, s, size=12, color=col)

svg.save("m05-forward-reverse.svg")
