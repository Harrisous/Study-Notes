"""Figure 1.12: the FFN read as a key-value memory, and the GELU non-linearity.

Left: FFN(x) = sum_m GELU(k_m . x + b_m) v_m with k_m = rows of W_1 and v_m = columns of W_2
(Geva et al. 2021, eq. 1, rewritten in the course's column convention), GPT-2 small shapes.
The 16 coefficient cells are illustrative (seeded random pre-activations through GELU).
Right: exact GELU x*Phi(x), its tanh approximation (GPT-2 code) and ReLU on [-4, 3]; the minimum
(-0.752, -0.170) and GELU(1) = 0.841 are computed and asserted (lesson-01-doc sec. 4.5).
Writes deepseek/figures/l01-ffn.svg
"""

import math

import numpy as np

from l01_svg import SVG


def gelu(x):
    return x * 0.5 * (1 + math.erf(x / math.sqrt(2)))


def gelu_tanh(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


xs = np.linspace(-4, 3, 1401)
g = np.array([gelu(x) for x in xs])
gt = np.array([gelu_tanh(x) for x in xs])
imin = int(np.argmin(g))
assert abs(xs[imin] + 0.752) < 0.01 and abs(g[imin] + 0.16997) < 1e-3
assert abs(gelu(1) - 0.841345) < 1e-5
grid = np.linspace(-6, 6, 120001)
err = max(abs(gelu(x) - gelu_tanh(x)) for x in grid[::10])
print("max |GELU - tanh approx| on [-6,6]: %.2e" % err)

W, H = 760, 340
s = SVG(W, H, "前馈网络 = 3072 条键值记忆；GELU 与 ReLU")

# ---------------------------------------------------------------- left: key-value memory
s.text(10, 22, "(a) FFN 的键值记忆读法（GPT-2 small，一个位置）", size=13, weight="700")
Y0, HB = 58, 220
# input x
s.rect(24, Y0, 22, HB, fill="blue", fill_opacity=0.12, stroke="blue", width=1)
s.text(35, Y0 + HB + 18, "*x*  [768]", size=12, anchor="middle")
s.text(35, Y0 - 8, "输入", size=12, anchor="middle", color="muted")
# keys
KX = 80
s.rect(KX, Y0, 70, HB, fill="violet", fill_opacity=0.08, stroke="violet", width=1)
for r in range(1, 16):
    s.line(KX, Y0 + r * HB / 16, KX + 70, Y0 + r * HB / 16, color="violet", width=0.4, opacity=0.6)
s.text(KX + 35, Y0 - 8, "键 k_{m}", size=12, anchor="middle", color="violet")
s.text(KX + 35, Y0 + HB + 18, "W_{1} 的行", size=12, anchor="middle", color="violet")
s.text(KX + 35, Y0 + HB + 37, "[3072, 768]", size=11, anchor="middle", color="muted")
s.arrow(48, Y0 + HB / 2, KX - 2, Y0 + HB / 2, width=1.3)
# coefficients
rng = np.random.default_rng(3)
pre = rng.standard_normal(16) * 1.2 - 0.6
coef = np.array([gelu(v) for v in pre])
CX = 186
s.text(CX + 20, Y0 - 8, "系数 c_{m}", size=12, anchor="middle", color="ink")
for r in range(16):
    c = coef[r]
    col = "green" if c > 0.05 else "muted"
    s.rect(CX, Y0 + r * HB / 16 + 1, 40, HB / 16 - 2, fill=col, fill_opacity=min(0.85, 0.08 + 0.45 * abs(c)))
s.text(CX + 20, Y0 + HB + 18, "GELU(k_{m}·x + b_{m})", size=11.5, anchor="middle")
s.text(CX + 20, Y0 + HB + 37, "3072 个数", size=11, anchor="middle", color="muted")
s.arrow(KX + 72, Y0 + HB / 2, CX - 2, Y0 + HB / 2, width=1.3)
# values
VX = 262
s.rect(VX, Y0, 70, HB, fill="green", fill_opacity=0.08, stroke="green", width=1)
for r in range(1, 16):
    s.line(VX, Y0 + r * HB / 16, VX + 70, Y0 + r * HB / 16, color="green", width=0.4, opacity=0.6)
s.text(VX + 35, Y0 - 8, "值 v_{m}", size=12, anchor="middle", color="green")
s.text(VX + 35, Y0 + HB + 18, "W_{2} 的列", size=12, anchor="middle", color="green")
s.text(VX + 35, Y0 + HB + 37, "[768, 3072]", size=11, anchor="middle", color="muted")
s.arrow(CX + 42, Y0 + HB / 2, VX - 2, Y0 + HB / 2, width=1.3)
# output
OX = 360
s.rect(OX, Y0, 22, HB, fill="ink", fill_opacity=0.06, stroke="ink", width=1)
s.text(OX + 11, Y0 - 8, "输出", size=12, anchor="middle", color="muted")
s.text(OX + 11, Y0 + HB + 18, "*m*  [768]", size=12, anchor="middle")
s.arrow(VX + 72, Y0 + HB / 2, OX - 2, Y0 + HB / 2, width=1.3)
s.text(10, 330, "*m* = Σ_{m} c_{m} *v*_{m}：3072 个值向量按系数加权相加（绿色 = 被激活的记忆）", size=12, color="muted")

# ---------------------------------------------------------------- right: GELU
s.line(404, 12, 404, 336, color="rule", width=1)
RX0, RX1, RY0, RY1 = 452, 742, 50, 280
XL, XR, YL, YR = -4, 3, -0.6, 3


def rx(x):
    return RX0 + (x - XL) / (XR - XL) * (RX1 - RX0)


def ry(y):
    return RY1 - (y - YL) / (YR - YL) * (RY1 - RY0)


s.text(418, 22, "(b) GELU(x) = x Φ(x)", size=13, weight="700")
for x in range(-4, 4):
    s.line(rx(x), RY0, rx(x), RY1, color="grid", width=1)
    s.text(rx(x), RY1 + 16, str(x).replace("-", "−"), size=11, anchor="middle", color="muted")
for y in (0, 1, 2, 3):
    s.line(RX0, ry(y), RX1, ry(y), color="grid", width=1)
    s.text(RX0 - 6, ry(y) + 4, str(y), size=11, anchor="end", color="muted")
s.line(RX0, ry(0), RX1, ry(0), color="muted", width=1.1)
s.line(rx(0), RY0, rx(0), RY1, color="muted", width=1.1)
relu = np.maximum(xs, 0)
s.path("M" + " L".join("%.1f %.1f" % (rx(a), ry(b)) for a, b in zip(xs, relu)), stroke="muted", width=1.6, dash="5 4")
s.path("M" + " L".join("%.1f %.1f" % (rx(a), ry(b)) for a, b in zip(xs, g)), stroke="ink", width=2.4)
s.path("M" + " L".join("%.1f %.1f" % (rx(a), ry(b)) for a, b in zip(xs[::20], gt[::20])), stroke="orange",
       width=1.2, dash="2 3")
s.circle(rx(xs[imin]), ry(g[imin]), 4, fill="red")
s.text(rx(xs[imin]) - 4, ry(g[imin]) + 22, "最小值 (−0.752, −0.170)", size=11.5, color="red", anchor="middle")
s.circle(rx(1), ry(gelu(1)), 3.5, fill="ink")
s.text(rx(1) + 8, ry(gelu(1)) + 14, "(1, 0.841)", size=11.5, color="ink")
s.text(rx(1.55), ry(2.55), "ReLU", size=12, color="muted")
s.text(rx(-3.9), ry(2.7), "黑实线：精确 GELU", size=11.5, color="ink")
s.text(rx(-3.9), ry(2.42), "橙点线：tanh 近似（GPT-2 代码）", size=11.5, color="orange")
s.text(rx(-3.9), ry(2.14), "两者最大差 %.1e" % err, size=11.5, color="muted")
s.text(418, 330, "负半轴不是一刀切成 0：小的负输入会“漏”出一点负值", size=12, color="muted")

s.save("l01-ffn.svg")
