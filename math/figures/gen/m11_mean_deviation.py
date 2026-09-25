"""Figure M11.3 (§2, property three): the mean stays, the deviation from the mean shrinks.

Repeatedly multiply the main example A (M11.5) into x = (9, 0, 3):
    x = (9, 0, 3), Ax = (5.4, 2.7, 3.9), A^2 x = (4.32, 3.69, 3.99), A^3 x = (4.068, 3.933, 3.999).
Every vector sums to 12 (mean 4); the deviation from the mean has length
6.481, 1.913, 0.446, 0.095, ... and the ratio tends to 0.2 (second eigenvalue of A),
while the guaranteed bound shrinks by sigma_2(A) = 0.5646 per step (M11.22).
All numbers are computed here with numpy. Writes math/figures/m11-mean-deviation.svg
"""

import math

import numpy as np

from m11_svg import SVG, fmt

A = np.array([[0.6, 0.4, 0.0], [0.1, 0.3, 0.6], [0.3, 0.3, 0.4]])
x0 = np.array([9.0, 0.0, 3.0])
sig = np.linalg.svd(A, compute_uv=False)
s2 = sig[1]

vecs = [x0]
for _ in range(8):
    vecs.append(A @ vecs[-1])
mean = x0.mean()
devlen = [float(np.linalg.norm(v - v.mean())) for v in vecs]
for k, v in enumerate(vecs[:7]):
    print(k, np.round(v, 5), "sum %.6f" % v.sum(), "dev len %.5f" % devlen[k],
          "ratio %.4f" % (devlen[k] / devlen[k - 1]) if k else "")
print("sigma", sig)

W, H = 760, 452
svg = SVG(W, H, "反复乘同一个双随机矩阵：均值 4 与总和 12 不动，偏离均值的部分一步步变短")

# ---------------- panel (a): bars
svg.text(40, 26, "(a) 反复乘主算例 A：均值与总和不动", size=13, weight="600")
svg.text(40, 45, "每组 3 根柱子依次是流 1、2、3 在同一个坐标上的值", size=12, color="muted")
X0, Y0, Y1 = 70, 320, 90            # plot box: y=Y0 is value 0, y=Y1 is value 10
AX1 = 500                            # right end of the bar axis
VMAX = 10.0


def yv(v):
    return Y0 - (Y0 - Y1) * v / VMAX


# axis + grid
for t in range(0, 11, 2):
    y = yv(t)
    svg.line(X0, y, AX1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, str(t), size=11, anchor="end", color="muted")
svg.line(X0, Y0, AX1, Y0, color="rule", width=1.2)
svg.line(X0, Y0, X0, Y1 - 6, color="rule", width=1.2)

names = ["x", "Ax", "A²x", "A³x"]
BW, GAP, GROUP = 24, 4, 108
for g in range(4):
    v = vecs[g]
    gx = X0 + 22 + g * GROUP
    for i in range(3):
        bx = gx + i * (BW + GAP)
        top = yv(v[i])
        h = Y0 - top
        if h > 0.5:
            svg.rect(bx, top, BW, h, fill="blue", fill_opacity=0.16, stroke="blue", width=1.2)
        else:
            svg.line(bx, Y0, bx + BW, Y0, color="blue", width=2)
        # deviation from the mean: tinted band between the mean line and the bar top
        ym = yv(mean)
        lo, hi = sorted([ym, top])
        if hi - lo > 0.8:
            svg.rect(bx, lo, BW, hi - lo, fill="red", fill_opacity=0.42)
        lab = ("%g" % v[i]) if g < 2 else ("%.2f" % v[i])
        svg.text(bx + BW / 2, Y0 + 15, lab, size=11, anchor="middle", color="blue")
    cx = gx + 1.5 * BW + GAP
    svg.text(cx, Y0 + 38, names[g], size=13, anchor="middle", weight="700", color="blue")
    svg.text(cx, Y0 + 57, "和 %g" % round(v.sum(), 6), size=12, anchor="middle", color="green")
    svg.text(cx, Y0 + 75, "偏离长度 %s" % ("%.3f" % devlen[g] if devlen[g] < 1 else "%.2f" % devlen[g]),
             size=12, anchor="middle", color="red")
    if g < 3:
        ax = gx + 3 * BW + 2 * GAP + 5
        alen = GROUP - 3 * BW - 2 * GAP - 10
        svg.arrow(ax, yv(8.2), ax + alen, yv(8.2), color="muted", width=1.3)
        svg.text(ax + alen / 2, yv(8.2) - 7, "×A", size=12, anchor="middle", color="muted")

# the mean line, drawn last so it sits on top of the bars
ym = yv(mean)
svg.line(X0, ym, AX1, ym, color="green", width=1.8, dash="6 4")
svg.text(AX1 + 4, ym + 4, "均值 4", size=12, color="green", weight="600")
svg.text(X0 - 30, (Y0 + Y1) / 2, "值", size=12, anchor="middle", color="muted")

# ---------------- panel (b): deviation length on a log axis
PX0, PX1, PY0, PY1 = 600, 740, 320, 90      # x: k = 0..6 ; y: 10^-4 .. 10^1
KMAX = 6
LO, HI = -4.0, 1.0
svg.text(565, 26, "(b) 偏离部分的长度（对数纵轴）", size=13, weight="600")
svg.text(565, 45, "每乘一次 A 最多剩 σ₂ = %.4f 倍" % s2, size=12, color="muted")


def px(k):
    return PX0 + (PX1 - PX0) * k / KMAX


def py(v):
    return PY0 - (PY0 - PY1) * (math.log10(v) - LO) / (HI - LO)


for e in range(-4, 2):
    y = py(10.0 ** e)
    svg.line(PX0, y, PX1, y, color="grid", width=1)
    lab = {1: "10", 0: "1"}.get(e, "10^{%d}" % e)
    svg.text(PX0 - 6, y + 4, lab, size=11, anchor="end", color="muted")
for k in range(0, KMAX + 1):
    svg.text(px(k), PY0 + 16, str(k), size=11, anchor="middle", color="muted")
svg.line(PX0, PY0, PX1, PY0, color="rule", width=1.2)
svg.line(PX0, PY0, PX0, PY1 - 6, color="rule", width=1.2)
svg.text((PX0 + PX1) / 2, PY0 + 36, "乘 A 的次数 k", size=12, anchor="middle", color="muted")

# guaranteed bound 6.48 * s2^k  (dashed)
bound = [devlen[0] * s2 ** k for k in range(KMAX + 1)]
d = "M" + " L".join("%.1f %.1f" % (px(k), py(b)) for k, b in enumerate(bound))
svg.path(d, stroke="muted", width=1.6, dash="5 4")
# actual
d = "M" + " L".join("%.1f %.1f" % (px(k), py(devlen[k])) for k in range(KMAX + 1))
svg.path(d, stroke="red", width=2.2, join="round")
for k in range(KMAX + 1):
    svg.circle(px(k), py(devlen[k]), 3.2, fill="red")
for k, (dx, dy) in zip(range(4), [(10, 2), (8, 12), (8, 12), (8, 12)]):
    val = devlen[k]
    lab = "%.2f" % val if val >= 1 else "%.3f" % val
    svg.text(px(k) + dx, py(val) + dy, lab, size=11, color="red")
svg.text(PX1, py(bound[2]) - 7, "界：每步 ×0.5646", size=11, anchor="end", color="muted")
svg.text(PX1, py(devlen[KMAX]) + 24, "实际：每步之比趋于 0.2", size=11, anchor="end", color="red")

# ---------------- legend
LY = H - 16
svg.rect(40, LY - 10, 18, 12, fill="blue", fill_opacity=0.16, stroke="blue", width=1.2)
svg.text(64, LY + 1, "各条流的值", size=12, color="muted")
svg.rect(160, LY - 10, 18, 12, fill="red", fill_opacity=0.42)
svg.text(184, LY + 1, "偏离均值的部分（被一步步压扁）", size=12, color="muted")
svg.line(390, LY - 4, 414, LY - 4, color="green", width=1.8, dash="6 4")
svg.text(420, LY + 1, "均值（不动）", size=12, color="muted")
svg.line(530, LY - 4, 554, LY - 4, color="muted", width=1.6, dash="5 4")
svg.text(560, LY + 1, "保证的界 6.48 × 0.5646^{k}", size=12, color="muted")

svg.save("m11-mean-deviation.svg")
