"""Figure M3.8: RoPE frequencies and wavelengths, base 10000.

d = 128: theta_i = 10000^(-2i/128), lambda_i = 2 pi / theta_i, i = 0..63 -> lambda from 6.28 to 54,410.
d = 64 (DeepSeek-V2/V3 decoupled RoPE): pair i' equals the d = 128 pair 2i'.
Shaded: lambda > 4096 (DeepSeek's pretraining length): 18 pairs for d = 128, 9 for d = 64.
Every point is computed here. Writes math/figures/m03-frequencies.svg
"""

import math

import numpy as np

from m03_svg import SVG, fmt

b = 10000.0
i128 = np.arange(64)
lam128 = 2 * math.pi * b ** (2 * i128 / 128)
i64 = np.arange(32)
lam64 = 2 * math.pi * b ** (2 * i64 / 64)
L = 4096
n_slow128 = int((lam128 > L).sum())
n_slow64 = int((lam64 > L).sum())
assert n_slow128 == 18 and n_slow64 == 9
assert abs(lam128[-1] - 54410.14) < 0.01
# for comparison: the same end points, but spaced arithmetically instead of geometrically
theta_ari = np.linspace(1.0, 2 * math.pi / lam128[-1], 64)
lam_ari = 2 * math.pi / theta_ari
assert int((lam_ari < 100).sum()) == 60 and int((lam_ari > L).sum()) == 1
assert int(((lam128 >= 100) & (lam128 <= L)).sum()) == 26 and int((lam128 < 100).sum()) == 20

W, H = 740, 400
svg = SVG(W, H, "RoPE 各对维度的波长与频率（对数纵轴）")
X0, X1 = 92, 640
Y0, Y1 = 58, 322  # top, bottom of plot area
LOG_MIN, LOG_MAX = 0.0, 5.0  # lambda from 1 to 1e5


def X(i):
    return X0 + (X1 - X0) * i / 63


def Y(lam):
    return Y1 - (Y1 - Y0) * (math.log10(lam) - LOG_MIN) / (LOG_MAX - LOG_MIN)


svg.text(W / 2, 24, "RoPE 的“多尺度时钟”：波长随维度编号按几何级数增长（b = 10000）", size=14, anchor="middle", weight="600")
# slow region shading (lambda > 4096)
svg.rect(X0, Y0, X1 - X0, Y(L) - Y0, fill="amber", fill_opacity=0.10)
svg.text(X0 + 8, Y(L) - 8, "λ > 4096：在 DeepSeek 的 4096 预训练长度内转不满一圈", size=12, color="amber")
# grid and left axis (wavelength)
for e in range(0, 6):
    y = Y(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "10" if e == 1 else ("1" if e == 0 else "10^{%d}" % e), size=11.5, anchor="end", color="muted")
for e in range(0, 5):
    for m in range(2, 10):
        y = Y(m * 10 ** e)
        if Y0 <= y <= Y1:
            svg.line(X0 - 3, y, X0, y, color="muted", width=0.8)
svg.line(X0, Y0, X0, Y1, color="ink", width=1.2)
svg.line(X0, Y1, X1, Y1, color="ink", width=1.2)
svg.line(X1, Y0, X1, Y1, color="ink", width=1.2)
svg.text(28, (Y0 + Y1) / 2, "波长 λᵢ = 2π/θᵢ（token，对数）", size=12, anchor="middle", rotate=-90)
# right axis: theta = 2 pi / lambda
for e in range(0, 5):
    th = 10.0 ** (-e)
    y = Y(2 * math.pi / th)
    svg.line(X1, y, X1 + 4, y, color="ink", width=1)
    svg.text(X1 + 8, y + 4, "1" if e == 0 else "10^{−%d}" % e, size=11.5, color="muted")
svg.text(W - 18, (Y0 + Y1) / 2, "频率 θᵢ（rad/token）", size=12, anchor="middle", rotate=90)
# x axis ticks
for i in (0, 8, 16, 24, 32, 40, 48, 56, 63):
    x = X(i)
    svg.line(x, Y1, x, Y1 + 4, color="ink", width=1)
    svg.text(x, Y1 + 17, str(i), size=11.5, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, Y1 + 36, "d = 128 时的对号 i（顶部灰字：d = 64 时的对号 i′ = i/2）", size=12, anchor="middle")
for ip in (0, 8, 16, 24, 31):
    svg.text(X(2 * ip), Y0 - 6, str(ip), size=11, anchor="middle", color="muted")
# L line
yl = Y(L)
svg.line(X0, yl, X1, yl, color="amber", width=1.4, dash="6 3")
svg.text(X1 - 4, yl + 15, "λ = 4096", size=12, anchor="end", color="amber")
# curve d=128
pts = " L".join("%s %s" % (fmt(X(i)), fmt(Y(l))) for i, l in zip(i128, lam128))
svg.path("M" + pts, stroke="blue", width=1.4, opacity=0.6)
for i, l in zip(i128, lam128):
    svg.circle(X(i), Y(l), 2.6, fill="blue")
# arithmetic spacing (comparison): grey crosses
for i, l in zip(i128, lam_ari):
    x, y = X(i), Y(l)
    svg.line(x - 2.6, y - 2.6, x + 2.6, y + 2.6, color="muted", width=1.2)
    svg.line(x - 2.6, y + 2.6, x + 2.6, y - 2.6, color="muted", width=1.2)
svg.text(X(52), 229, "× 等差排列（对照）", size=12, anchor="end", color="muted")
# d=64 rings at even i
for ip, l in zip(i64, lam64):
    svg.circle(X(2 * ip), Y(l), 5.2, stroke="orange", width=1.5)
# annotations
for i, txt, dx, dy, anc in [
    (0, "i = 0：θ = 1，λ = 6.28", 8, 14, "start"),
    (32, "i = 32：θ = 0.01，λ = 628.3", 8, 16, "start"),
    (60, "i = 63：θ = 1.155 × 10^{−4}，λ = 54,410", -8, -2, "end"),
]:
    yy = Y(lam128[63]) + 4 if i == 60 else Y(lam128[i]) + dy
    svg.text(X(i) + dx, yy, txt, size=12, anchor=anc, color="ink")
svg.line(X(60) - 4, Y(lam128[63]), X(63) - 6, Y(lam128[63]), color="muted", width=1)
# decade bracket: 16 pairs per factor 10 (drawn above-left of the curve)
xa, xb = X(8), X(24)
ya = Y(lam128[8])
yb = Y(lam128[24])
svg.line(xa, ya - 8, xa, yb, color="muted", width=1)
svg.line(xa, yb, xb - 8, yb, color="muted", width=1)
svg.text((xa + xb) / 2 - 36, yb - 7, "从 i = 8 到 24（16 对），波长 ×10", size=12, anchor="middle", color="muted")
# legend
lx, ly = X(22), Y1 - 37  # low enough to clear the grey crosses above it
svg.circle(lx, ly, 3, fill="blue")
svg.text(lx + 10, ly + 4, "d = 128（如 DeepSeek LLM 7B 的头）：64 对，18 对 λ > 4096", size=12)
svg.circle(lx, ly + 20, 5.2, stroke="orange", width=1.5)
svg.text(lx + 10, ly + 24, "d = 64（V2/V3 解耦 RoPE）：隔一个取一个，9 对 λ > 4096", size=12)
svg.text(W / 2, 392, "左轴与右轴是同一条曲线的两种读法：λ = 2π/θ。纵轴取对数后，几何级数变成直线；等差排列的 × 几乎全挤在底部。", size=11.5, anchor="middle", color="muted")
svg.save("m03-frequencies.svg")
