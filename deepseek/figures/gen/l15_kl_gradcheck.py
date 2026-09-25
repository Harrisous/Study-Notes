"""Figure 15.7: the KL term's gradient weight, original k3 vs V3.2's importance-corrected k3.

(a) Coefficient on grad log pi_theta(token) of the loss term +k (per unit beta), as a function of
    z = pi_theta / pi_ref (the lesson's (15.13); x is the token there), at rho = 1:
      original k3:   d k3 / d log pi_theta          = 1 - 1/z          (unbounded as z -> 0)
      V3.2 (rho*k3): d (rho*k3) / d log pi_theta    = rho * log z      (logarithmic)
    Plotted on a symmetric-log y axis: y' = sign(y) * log10(1 + |y|).
(b) Five-token softmax check (lesson-15-doc.md N23): logits and reference logits are two consecutive
    5-vectors from numpy.random.default_rng(0). On-policy sampling (pi_old = pi_theta). Expected gradients
    (w.r.t. the logits) computed by exact enumeration over the 5 tokens, and the two true KL gradients
    computed analytically and checked by central finite differences. An extra assertion (not drawn) repeats
    the check with a different sampling distribution pi_old (numpy.random.default_rng(1)).
Writes deepseek/figures/l15-kl-gradcheck.svg
"""

import math

import numpy as np

from l15_svg import SVG, fmt

# ---------------------------------------------------------------- data for (b)
rng = np.random.default_rng(0)
z = rng.normal(size=5)
zr = rng.normal(size=5)


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


p, q = softmax(z), softmax(zr)
I = np.eye(5)
grad_fwd = p * (np.log(p / q) - (p * np.log(p / q)).sum())      # d KL(p||q) / dz
grad_rev = p - q                                                # d KL(q||p) / dz
E_k3 = sum(p[x] * (1 - q[x] / p[x]) * (I[x] - p) for x in range(5))
E_rk3 = sum(p[x] * (-math.log(q[x] / p[x])) * (I[x] - p) for x in range(5))


def KL(a, b):
    return float((a * np.log(a / b)).sum())


fd = np.array([(KL(softmax(z + 1e-6 * I[i]), q) - KL(softmax(z - 1e-6 * I[i]), q)) / 2e-6 for i in range(5)])
assert np.allclose(fd, grad_fwd, atol=1e-7)
assert np.allclose(E_rk3, grad_fwd) and np.allclose(E_k3, grad_rev)
assert np.allclose(np.round(grad_fwd, 4), [-0.0195, -0.2027, -0.0565, 0.1880, 0.0908])
assert np.allclose(np.round(grad_rev, 4), [0.0330, -0.2783, 0.0344, 0.1398, 0.0712])

# Off-policy check (not drawn): sample from a different pi_old. The gradient of rho*k3 w.r.t. the logits is
# rho * (-log w) * (e_x - p) with rho = p/p_old; its expectation under pi_old is still grad KL(p||q), while the
# raw k3 gradient (1 - w) * (e_x - p) averaged under pi_old matches neither KL gradient.
p_old = softmax(np.random.default_rng(1).normal(size=5))
E_rk3_off = sum(p_old[x] * (p[x] / p_old[x]) * (-math.log(q[x] / p[x])) * (I[x] - p) for x in range(5))
E_k3_off = sum(p_old[x] * (1 - q[x] / p[x]) * (I[x] - p) for x in range(5))
assert np.allclose(E_rk3_off, grad_fwd)
assert not np.allclose(E_k3_off, grad_fwd, atol=1e-3) and not np.allclose(E_k3_off, grad_rev, atol=1e-3)

# single-token example (N21)
assert abs((1 - 0.1 / 0.001) - (-99)) < 1e-9 and abs(math.log(0.001 / 0.1) + 4.60517) < 1e-5

W, H = 760, 420
svg = SVG(W, H, "KL 项的梯度权重：原始 k3 与 V3.2 的重要性修正版")

# ================================================================ (a)
AX0, AX1, AY0, AY1 = 64, 356, 50, 330
svg.text(16, 22, "(a) 每个 token 的梯度权重（ρ = 1）", size=13, weight="700")
LX0, LX1 = -3.0, 1.0


def ax(lx):
    return AX0 + (lx - LX0) / (LX1 - LX0) * (AX1 - AX0)


def slog(y):
    return math.copysign(math.log10(1 + abs(y)), y)


SY0, SY1 = slog(-2500), slog(12)


def ay(y):
    return AY1 - (slog(y) - SY0) / (SY1 - SY0) * (AY1 - AY0)


for yv, lab in [(-1000, "−1000"), (-100, "−100"), (-10, "−10"), (-1, "−1"), (0, "0"), (1, "1"), (10, "10")]:
    svg.line(AX0, ay(yv), AX1, ay(yv), color="grid" if yv else "rule", width=1)
    svg.text(AX0 - 6, ay(yv) + 4, lab, size=11, anchor="end", color="muted")
for e in range(-3, 2):
    svg.line(ax(e), AY0, ax(e), AY1, color="grid", width=1)
    svg.text(ax(e), AY1 + 16, "10^{%d}" % e, size=11, anchor="middle", color="muted")
svg.text((AX0 + AX1) / 2, AY1 + 36, "z = π_{θ} / π_{ref}（对数刻度）", size=12, anchor="middle", color="muted")
svg.text(AX0 - 50, AY0 - 12, "纵轴：对称对数刻度", size=11, color="muted")


def curve(f, color, dash=None):
    pts = []
    for k in range(241):
        lx = LX0 + (LX1 - LX0) * k / 240
        pts.append("%s %s" % (fmt(ax(lx)), fmt(ay(f(10 ** lx)))))
    svg.path("M" + " L".join(pts), stroke=color, width=2.2, dash=dash)


curve(lambda x: 1 - 1 / x, "red")
curve(lambda x: math.log(x), "blue")
svg.text(ax(-1.25), ay(-40), "原始 k3：1 − 1/z", size=12, color="red", weight="700")
svg.text(ax(-2.95), ay(-2.6) - 30, "V3.2：ρ·log z", size=12, color="blue", weight="700")
for x in (0.001, 0.01):
    for f, col in ((lambda v: 1 - 1 / v, "red"), (lambda v: math.log(v), "blue")):
        svg.circle(ax(math.log10(x)), ay(f(x)), 3.8, fill=col, stroke="surface", width=1)
svg.text(ax(-2) - 8, ay(-99) - 6, "−99", size=12, anchor="end", color="red")
svg.text(ax(-2) + 8, ay(math.log(0.01)) + 16, "−4.61", size=12, color="blue")
svg.text(ax(-3) + 12, ay(-999) + 5, "−999", size=12, color="red")
svg.text(ax(-3) + 8, ay(math.log(0.001)) + 16, "−6.91", size=12, color="blue")
svg.text(ax(-2.1), ay(-900), "π_{θ} ≪ π_{ref} 时原始权重无界", size=11, color="muted")

# ================================================================ (b)
BX0, BX1, BY0, BY1 = 440, 746, 50, 330
svg.text(400, 22, "(b) 5 个 token 的 softmax：期望梯度等于哪个 KL 的梯度", size=13, weight="700")
YMAX = 0.3


def by(v):
    return (BY0 + BY1) / 2 - v / YMAX * (BY1 - BY0) / 2


for v in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
    svg.line(BX0, by(v), BX1, by(v), color="rule" if v == 0 else "grid", width=1)
    svg.text(BX0 - 6, by(v) + 4, ("%.1f" % v).replace("-", "−"), size=11, anchor="end", color="muted")
GW = (BX1 - BX0) / 5
BWID = GW / 5.2
series = [(grad_fwd, "blue", 0.55, None, "∇KL(π_{θ}‖π_{ref})"),
          (E_rk3, "blue", 0.0, None, "E[∇(ρ·k3)]"),
          (grad_rev, "red", 0.55, None, "∇KL(π_{ref}‖π_{θ})"),
          (E_k3, "red", 0.0, None, "E[∇k3]")]
for t in range(5):
    gx = BX0 + t * GW + GW / 2 - 2 * BWID
    for k, (vals, col, op, _, _) in enumerate(series):
        v = float(vals[t])
        x = gx + k * BWID
        y0, y1 = sorted((by(0), by(v)))
        svg.rect(x + 1, y0, BWID - 2, max(y1 - y0, 0.8), fill=col if op else None, fill_opacity=op or None,
                 stroke=col, width=1.4)
    svg.text(BX0 + t * GW + GW / 2, BY1 + 16, "token %d" % (t + 1), size=11, anchor="middle", color="muted")
svg.text((BX0 + BX1) / 2, BY1 + 36, "对 logits 的梯度分量（在策略采样：π_{old} = π_{θ}）", size=12, anchor="middle",
         color="muted")

# legend under the plots
LY = 390
lx = 64
for vals, col, op, _, lab in series:
    svg.rect(lx, LY - 11, 16, 14, fill=col if op else None, fill_opacity=op or None, stroke=col, width=1.4)
    svg.text(lx + 22, LY, lab, size=12)
    lx += 42 + 7.2 * len(lab.replace("_{", "").replace("}", ""))
svg.text(64, 412, "实心 = 真实的 KL 梯度（解析式，已用有限差分核对）；空心 = 把估计量当损失求导后在 π_{θ} 下的期望。两两重合到小数点后 4 位。",
         size=11, color="muted")

svg.save("l15-kl-gradcheck.svg")
