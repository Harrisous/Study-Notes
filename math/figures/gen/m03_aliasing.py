"""Figure M3.9: one frequency wraps around, many frequencies do not.

Score of a query with an identical key at relative distance D (normalised so D = 0 gives 1):
  single frequency theta = 0.5 (one 2-D pair):  cos(0.5 D)
  RoPE, d = 128, b = 10000, q = k = all-ones:   (2/d) * sum_i cos(D theta_i)
(a) D = 0..60 on a linear axis. (b) D = 1..65536 on a log axis: for every pixel column the min and
max over all integer D in that column are drawn as a band, and the column mean as a line.
Every value is computed here. Writes math/figures/m03-aliasing.svg
"""

import math

import numpy as np

from m03_svg import SVG, fmt

d, b = 128, 10000.0
theta = b ** (-np.arange(0, d, 2) / d)


def multi(D):
    D = np.asarray(D, dtype=float)
    return (2.0 / d) * np.cos(np.multiply.outer(D, theta)).sum(axis=-1)


def single(D):
    return np.cos(0.5 * np.asarray(D, dtype=float))


assert abs(multi(13) - 0.6600) < 5e-5 and abs(single(13) - 0.9766) < 5e-5
assert abs(multi(1000) - 0.1590) < 5e-5 and abs(multi(4096) + 0.0529) < 5e-5

W, H = 740, 382
svg = SVG(W, H, "单一频率会绕圈，多频率不会：同向查询与键的打分随相对距离的变化")
PY0, PY1 = 74, 292


def frame(x0, x1, title):
    svg.text((x0 + x1) / 2, 22, title, size=13.5, anchor="middle", weight="600")
    for v in (-1, -0.5, 0, 0.5, 1):
        y = PY1 - (PY1 - PY0) * (v + 1) / 2
        svg.line(x0, y, x1, y, color="grid" if v != 0 else "rule", width=1)
        svg.text(x0 - 6, y + 4, fmt(v).replace("-", "−"), size=11, anchor="end", color="muted")
    svg.line(x0, PY0, x0, PY1, color="ink", width=1.1)
    svg.line(x0, PY1, x1, PY1, color="ink", width=1.1)


def Yv(v):
    return PY1 - (PY1 - PY0) * (v + 1) / 2


# ---------------- (a) linear 0..60
AX0, AX1 = 60, 350
frame(AX0, AX1, "(a) 距离 0–60：一根指针转一圈就“回到原点”")


def Xa(D):
    return AX0 + (AX1 - AX0) * D / 60


for D in range(0, 61, 10):
    svg.line(Xa(D), PY1, Xa(D), PY1 + 4, color="ink", width=1)
    svg.text(Xa(D), PY1 + 17, str(D), size=11, anchor="middle", color="muted")
svg.text((AX0 + AX1) / 2, PY1 + 34, "相对距离 Δ = t − j（token）", size=12, anchor="middle")
period = 2 * math.pi / 0.5
k = 1
while k * period <= 60:
    x = Xa(k * period)
    svg.line(x, PY0, x, PY1, color="red", width=1, dash="2 3", opacity=0.7)
    k += 1
Ds = np.linspace(0, 60, 601)
svg.path("M" + " L".join("%s %s" % (fmt(Xa(D)), fmt(Yv(v))) for D, v in zip(Ds, single(Ds))), stroke="red", width=1.8)
svg.path("M" + " L".join("%s %s" % (fmt(Xa(D)), fmt(Yv(v))) for D, v in zip(Ds, multi(Ds))), stroke="blue", width=2.4)
# period label, centred on the first dotted line at score -0.8: the red curve's two troughs sit at
# D = 6.28 and 18.85, and between D = 7.4 and 17.7 it stays above -0.87, so the label box
# (D = 9.3..15.9, score -0.87..-0.74) covers no data. (It used to sit on the axis and hid the second trough.)
LBL_Y = Yv(-0.8)
assert single(np.linspace(9.2, 16.0, 200)).min() > -0.74
svg.rect(Xa(period) - 16, LBL_Y - 10.5, 32, 14, fill="surface", fill_opacity=0.9, rx=2)
svg.text(Xa(period), LBL_Y + 1, "12.57", size=11, color="red", anchor="middle", halo=True)
for D, fn, col, dx, dy, anc in [(13, single, "red", 8, 5, "start"), (13, multi, "blue", 6, 26, "start")]:
    v = float(fn(D))
    svg.circle(Xa(D), Yv(v), 3.6, fill=col, stroke="surface", width=1)
    if col == "blue":  # this label sits where the red curve swings through; give it a plain backing
        svg.rect(Xa(D) + dx - 3, Yv(v) + dy - 12, 92, 16, fill="surface", fill_opacity=0.9, rx=2)
    svg.text(Xa(D) + dx, Yv(v) + dy, ("%.4f" if col == "red" else "Δ = 13：%.4f") % v, size=11.5, color=col, anchor=anc,
             halo=True)
svg.line(AX0 + 4, 43, AX0 + 24, 43, color="red", width=1.8)
svg.text(AX0 + 30, 47, "一个频率 θ = 0.5：cos(0.5Δ)", size=12, color="red")
svg.line(AX0 + 4, 61, AX0 + 24, 61, color="blue", width=2.4)
svg.text(AX0 + 30, 65, "RoPE 64 个频率的平均（d = 128，q = k = 全 1）", size=12, color="blue")

# ---------------- (b) log 1..65536
BX0, BX1 = 420, 712
frame(BX0, BX1, "(b) 距离 1–65536（对数横轴）：64 个频率")
LMIN, LMAX = 0.0, math.log10(65536)


def Xb(D):
    return BX0 + (BX1 - BX0) * (math.log10(D) - LMIN) / (LMAX - LMIN)


# shade beyond 4096
svg.rect(Xb(4096), PY0, BX1 - Xb(4096), PY1 - PY0, fill="amber", fill_opacity=0.10)
svg.text(Xb(4096) + 4, PY0 + 14, "> 4096", size=11, color="amber")
svg.line(BX0 + 4, 61, BX0 + 24, 61, color="blue", width=1.8)
svg.rect(BX0 + 4, 40, 20, 12, fill="blue", fill_opacity=0.18)
svg.text(BX0 + 30, 50, "每个像素列内的最小–最大值", size=12, color="blue")
svg.text(BX0 + 30, 65, "每个像素列内的平均", size=12, color="blue")
for e in range(0, 5):
    x = Xb(10 ** e)
    svg.line(x, PY1, x, PY1 + 4, color="ink", width=1)
    svg.text(x, PY1 + 17, "1" if e == 0 else ("10" if e == 1 else "10^{%d}" % e), size=11, anchor="middle", color="muted")
svg.text((BX0 + BX1) / 2, PY1 + 34, "相对距离 Δ（对数）", size=12, anchor="middle")
cols = int(BX1 - BX0)
lo, hi, mean = [], [], []
xs = []
for c in range(cols):
    a = 10 ** (LMIN + (LMAX - LMIN) * c / cols)
    bnd = 10 ** (LMIN + (LMAX - LMIN) * (c + 1) / cols)
    Dint = np.arange(math.ceil(a), math.floor(bnd) + 1)
    if len(Dint) == 0:  # below D ~ 26 some pixel columns hold no integer distance: skip them, the
        continue        # band and the line then join the neighbouring integer columns
    v = multi(Dint)
    lo.append(v.min())
    hi.append(v.max())
    mean.append(v.mean())
    xs.append(BX0 + c + 0.5)
band = "M" + " L".join("%s %s" % (fmt(x), fmt(Yv(v))) for x, v in zip(xs, hi))
band += " L" + " L".join("%s %s" % (fmt(x), fmt(Yv(v))) for x, v in zip(xs[::-1], lo[::-1])) + " Z"
svg.path(band, stroke=None, fill="blue", fill_opacity=0.18)
svg.path("M" + " L".join("%s %s" % (fmt(x), fmt(Yv(v))) for x, v in zip(xs, mean)), stroke="blue", width=1.8)
for D, dx, dy, anc in [(10, 4, -8, "start"), (100, 4, -8, "start"), (1000, 4, -10, "start"), (4096, -4, 18, "end")]:
    v = float(multi(D))
    svg.circle(Xb(D), Yv(v), 3.4, fill="ink", stroke="surface", width=1)
    svg.text(Xb(D) + dx, Yv(v) + dy, ("%d：%.3f" % (D, v)).replace("-", "−"), size=11.5, anchor=anc, halo=True)
svg.text(W / 2, 350, "纵轴：q = k 时的打分除以 Δ = 0 时的值。(b) 的半透明带是每个像素列内所有整数 Δ 的最小到最大值，实线是它们的平均。",
         size=11.5, anchor="middle", color="muted")
far = multi(np.arange(100, 65537))
svg.text(W / 2, 370, "单个频率每 12.57 个 token 回到 1；64 个频率合起来近处高、远处低，Δ ≥ 100 之后（到 65536 为止）最高只有 %.3f。"
         % far.max(), size=11.5, anchor="middle", color="muted")
assert abs(far.max() - 0.4972) < 5e-4
svg.save("m03-aliasing.svg")
