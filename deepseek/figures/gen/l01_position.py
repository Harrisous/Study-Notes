"""Figure 1.5: the sinusoidal position encoding of Vaswani et al. (2017, sec. 3.5), d = 512.

Left: PE(pos, 2m) = sin(pos * omega_m), omega_m = 10000^(-2m/d), for four dimension pairs m,
pos = 0..128, drawn as stacked strips ("clocks" of different speeds).
Right: wavelength 2*pi/omega_m = 2*pi*10000^(2m/d) against m on a log axis, with the values
listed in reference-docs/lesson-01-doc.md sec. 4.5 marked and asserted.
Writes deepseek/figures/l01-position.svg
"""

import math

import numpy as np

from l01_svg import SVG

d = 512


def wavelength(m):
    return 2 * math.pi * 10000 ** (2 * m / d)


checks = {0: 6.283, 10: 9.004, 50: 37.97, 100: 229.4, 200: 8378.8, 255: 60611}
for m, v in checks.items():
    assert abs(wavelength(m) - v) / v < 1e-3, (m, wavelength(m))

W, H = 760, 330
s = SVG(W, H, "正弦位置编码：不同维度对是转速不同的时钟（d = 512）")

# ---------------------------------------------------------------- left: strips
X0, X1 = 70, 440
s.text(10, 20, "PE(t, 2m) = sin(ω_{m} t)，ω_{m} = 10000^{−2m/512}", size=13, weight="700")
ms = [0, 32, 64, 128]
cols = ["blue", "violet", "green", "amber"]
pos = np.linspace(0, 128, 800)


def px(p):
    return X0 + p / 128 * (X1 - X0)


for k, (m, col) in enumerate(zip(ms, cols)):
    yc = 70 + k * 60
    amp = 20
    s.line(X0, yc, X1, yc, color="grid", width=1)
    s.line(X0, yc - amp, X1, yc - amp, color="grid", width=0.8)
    s.line(X0, yc + amp, X1, yc + amp, color="grid", width=0.8)
    om = 10000 ** (-2 * m / d)
    ys = np.sin(pos * om)
    s.path("M" + " L".join("%.1f %.1f" % (px(p), yc - amp * y) for p, y in zip(pos, ys)), stroke=col, width=1.8)
    s.text(X0 - 8, yc + 4, "m=%d" % m, size=12, anchor="end", color=col, weight="600")
    s.text(X1 + 6, yc - 6, "波长", size=11, color="muted")
    s.text(X1 + 6, yc + 10, "%.1f" % wavelength(m) if wavelength(m) < 1000 else "{:,.0f}".format(wavelength(m)),
           size=11, color=col)
    # mark pos = 1 value for the first strip
yb = 70 + 3 * 60 + 36
s.line(X0, yb, X1, yb, color="muted", width=1)
for p in (0, 32, 64, 96, 128):
    s.line(px(p), yb, px(p), yb + 4, color="muted", width=1)
    s.text(px(p), yb + 17, str(p), size=11, anchor="middle", color="muted")
s.text((X0 + X1) / 2, yb + 34, "位置 t", size=12, anchor="middle", color="muted")

# ---------------------------------------------------------------- right: wavelength vs m (log)
RX0, RX1, RY0, RY1 = 540, 740, 40, 266
LY0, LY1 = 0, 5          # 10^0 .. 10^5


def rx(m):
    return RX0 + m / 256 * (RX1 - RX0)


def ry(v):
    return RY1 - (math.log10(v) - LY0) / (LY1 - LY0) * (RY1 - RY0)


s.text(RX0 - 40, 20, "每对维度的波长 2π/ω_{m}（对数纵轴）", size=13, weight="700")
for e in range(LY0, LY1 + 1):
    y = ry(10 ** e)
    s.line(RX0, y, RX1, y, color="grid", width=1)
    s.text(RX0 - 6, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
for m in (0, 64, 128, 192, 256):
    s.line(rx(m), RY1, rx(m), RY1 + 4, color="muted", width=1)
    s.text(rx(m), RY1 + 17, str(m), size=11, anchor="middle", color="muted")
s.line(RX0, RY1, RX1, RY1, color="muted", width=1)
s.line(RX0, RY0, RX0, RY1, color="muted", width=1)
s.text((RX0 + RX1) / 2, RY1 + 34, "维度对编号 m（0 … 255）", size=12, anchor="middle", color="muted")
mm = np.linspace(0, 255, 120)
s.path("M" + " L".join("%.1f %.1f" % (rx(m), ry(wavelength(m))) for m in mm), stroke="ink", width=2)
for m in (0, 100, 200, 255):
    s.circle(rx(m), ry(wavelength(m)), 3.2, fill="ink")
# 60,611 sits above-left of its point: below it, the rising curve would run through the label
lbl = {0: ("6.28 = 2π", 9, 14), 100: ("229", 8, 4), 200: ("8,379", -8, -8), 255: ("60,611", -10, -4)}
for m, (t, dx, dy) in lbl.items():
    s.text(rx(m) + dx, ry(wavelength(m)) + dy, t, size=11, anchor="start" if dx > 0 else "end", color="ink")
s.text(RX0 + 6, RY0 + 12, "上限 2π × 10^{4} ≈ 62,832", size=11, color="muted")

s.save("l01-position.svg")
