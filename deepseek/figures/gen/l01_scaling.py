"""Figure 1.8: why the scores are divided by sqrt(d_k).

Left: distribution of q.k for d_k = 64 when q, k have independent N(0, 1) components
(Vaswani et al. 2017, footnote 4 assumptions), before and after dividing by sqrt(64) = 8.
Monte Carlo, 200,000 pairs, seed 0 (reference-docs/lesson-01-doc.md sec. 4.4: var 63.77 / 0.996).
Right: average largest softmax weight when 512 logits are drawn i.i.d. N(0, s^2), as a function of
the logit scale s (2,000 draws per s, seed 1). s = 1 is the scaled case; s = 8 is d_k = 64 unscaled.
Writes deepseek/figures/l01-scaling.svg
"""

import math

import numpy as np

from l01_svg import SVG

rng = np.random.default_rng(0)
dk, n = 64, 200_000
q = rng.standard_normal((n, dk))
k = rng.standard_normal((n, dk))
dots = np.einsum("nd,nd->n", q, k)
print("var raw %.2f  var scaled %.3f" % (dots.var(), (dots / 8).var()))

rng2 = np.random.default_rng(1)
svals = 2 ** np.linspace(-2, 4, 25)
maxw = []
for sc in svals:
    z = rng2.standard_normal((2000, 512)) * sc
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    maxw.append(p.max(1).mean())
maxw = np.array(maxw)
m1 = maxw[np.argmin(abs(svals - 1))]
m8 = maxw[np.argmin(abs(svals - 8))]
print("max weight s=1 %.3f  s=8 %.3f" % (m1, m8))
assert abs(m1 - 0.027) < 0.004 and abs(m8 - 0.729) < 0.03

W, H = 760, 330
s = SVG(W, H, "为什么除以 √d_k：点积的尺度随维度增长，softmax 随之饱和")

# ---------------------------------------------------------------- left: histograms
X0, X1, Y0, Y1 = 50, 380, 50, 270
XL, XR = -32, 32
YMAX = 0.42


def px(x):
    return X0 + (x - XL) / (XR - XL) * (X1 - X0)


def py(y):
    return Y1 - y / YMAX * (Y1 - Y0)


s.text(10, 22, "(a) d_{k} = 64 时点积 q·k 的分布", size=13, weight="700")
for x in (-32, -16, -8, 0, 8, 16, 32):
    s.line(px(x), Y0, px(x), Y1, color="grid", width=1)
    s.text(px(x), Y1 + 16, str(x), size=11, anchor="middle", color="muted")
for y in (0.1, 0.2, 0.3, 0.4):
    s.line(X0, py(y), X1, py(y), color="grid", width=1)
    s.text(X0 - 6, py(y) + 4, "%.1f" % y, size=11, anchor="end", color="muted")
s.line(X0, Y1, X1, Y1, color="muted", width=1.2)
s.text((X0 + X1) / 2, Y1 + 34, "分数的取值（概率密度，纵轴）", size=12, anchor="middle", color="muted")


def hist_path(vals, lo, hi, nb):
    h, e = np.histogram(vals, bins=nb, range=(lo, hi), density=True)
    pts = []
    for i in range(nb):
        pts.append((px(e[i]), py(h[i])))
        pts.append((px(e[i + 1]), py(h[i])))
    return "M%.1f %.1f " % (px(lo), py(0)) + " ".join("L%.1f %.1f" % p for p in pts) + " L%.1f %.1f Z" % (px(hi), py(0))


s.path(hist_path(dots, -32, 32, 64), stroke="violet", width=1.4, fill="violet", fill_opacity=0.15)
s.path(hist_path(dots / 8, -4, 4, 64), stroke="blue", width=1.6, fill="blue", fill_opacity=0.18)
s.text(px(4.6), py(0.37), "除以 8 之后：标准差 ≈ 1", size=12, color="blue", weight="600")
s.text(px(10), py(0.10), "不缩放：标准差 ≈ 8", size=12, color="violet", weight="600")
s.text(px(10), py(0.10) + 16, "（实测方差 %.1f，理论 64）" % dots.var(), size=11, color="violet")

# ---------------------------------------------------------------- right: saturation curve
R0, R1 = 460, 740
LX0, LX1 = -2, 4


def rx(sc):
    return R0 + (math.log2(sc) - LX0) / (LX1 - LX0) * (R1 - R0)


def ry(v):
    return Y1 - v * (Y1 - Y0)


s.line(430, 12, 430, 320, color="rule", width=1)
s.text(446, 22, "(b) 512 个分数做 softmax：最大权重的平均值", size=13, weight="700")
for e in range(LX0, LX1 + 1):
    x = rx(2 ** e)
    s.line(x, Y0, x, Y1, color="grid", width=1)
    s.text(x, Y1 + 16, "%g" % (2 ** e), size=11, anchor="middle", color="muted")
for v in (0.25, 0.5, 0.75, 1.0):
    s.line(R0, ry(v), R1, ry(v), color="grid", width=1)
    s.text(R0 - 6, ry(v) + 4, "%g" % v, size=11, anchor="end", color="muted")
s.line(R0, Y1, R1, Y1, color="muted", width=1.2)
s.text((R0 + R1) / 2, Y1 + 34, "分数的标准差 s（对数横轴）", size=12, anchor="middle", color="muted")
s.path("M" + " L".join("%.1f %.1f" % (rx(a), ry(b)) for a, b in zip(svals, maxw)), stroke="ink", width=2.2)
# s = 1: the curve climbs right after the dot, so the label sits in the empty upper-left
# region (above the 0.25 grid line) with a thin leader down to the dot
s.circle(rx(1), ry(m1), 4.5, fill="blue")
s.line(rx(1), ry(m1) - 6, rx(1), ry(0.25) - 18, color="blue", width=0.8)
s.text(R0 + 8, ry(0.25) - 22, "s = 1（缩放后）：%.3f" % m1, size=12, color="blue", weight="600")
s.text(R0 + 8, ry(0.25) - 40, "每个分数都分到一点权重", size=11, color="muted")
s.circle(rx(8), ry(m8), 4.5, fill="violet")
s.text(rx(8) - 12, ry(m8) - 14, "s = 8（d_{k} = 64 不缩放）：%.2f" % m8, size=12, color="violet", weight="600",
       anchor="end")
s.text(rx(8) - 40, ry(m8) + 4, "几乎全压在一个位置上", size=11, color="muted", anchor="end")

s.save("l01-scaling.svg")
