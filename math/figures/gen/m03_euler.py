"""Figure M3.4: Euler's formula on the unit circle, and the Taylor series of e^{i phi} walking there.

Left: the points e^{i phi} = cos phi + i sin phi for phi = 0, pi/6, 1, pi/2, 2, pi; the arc of length 1
      (one radian) is highlighted, with its cosine 0.5403 and sine 0.8415 dropped onto the axes.
Right: the partial sums S_N = sum_{n<=N} (2i)^n / n! for phi = 2. Each new term is the previous one
      turned by 90 degrees and multiplied by 2/n, so the path spirals into e^{2i} = -0.4161 + 0.9093i.
All points are computed here. Writes math/figures/m03-euler.svg
"""

import cmath
import math

from m03_svg import SVG, Plane, arc_path, fmt

W, H = 740, 404
svg = SVG(W, H, "欧拉公式：e 的 iφ 次方是单位圆上角度为 φ 的点；它的泰勒级数一项一项地转进这个点")

# ---------------- left: unit circle
s = 118
pl = Plane(40 + 1.25 * s, 70 + 1.2 * s, s)
svg.text(185, 26, "① e^{iφ} = cos φ + i sin φ：单位圆上角度为 φ 的点", size=13.5, anchor="middle", weight="600")
ox, oy = pl(0, 0)
a0, b0 = pl(-1.28, 0)
a1, _ = pl(1.3, 0)
svg.line(a0, b0, a1, b0, color="rule", width=1)
c0, d0 = pl(0, -0.5)
_, d1 = pl(0, 1.22)
svg.line(c0, d0, c0, d1, color="rule", width=1)
svg.path(arc_path(ox, oy, s, math.radians(-25), math.radians(205)), stroke="muted", width=1.2)
# arc of length 1 (one radian)
svg.path(arc_path(ox, oy, s, 0, 1.0), stroke="red", width=4, opacity=0.55)
p1 = pl(math.cos(1), math.sin(1))
svg.line(p1[0], p1[1], p1[0], oy, color="blue", width=1, dash="3 3", opacity=0.8)
svg.line(p1[0], p1[1], ox, p1[1], color="blue", width=1, dash="3 3", opacity=0.8)
svg.text(p1[0], oy + 15, "cos 1 = 0.540", size=11.5, anchor="middle", color="blue")
svg.text(ox - 4, p1[1] + 15, "sin 1 = 0.841", size=11.5, anchor="end", color="blue")
mid = pl(1.07 * math.cos(0.22), 1.07 * math.sin(0.22))
svg.text(mid[0] + 2, mid[1] + 4, "弧长 1", size=12, color="red")
pts = [
    (0.0, "1 = e^{i0}", (6, 16), "start"),
    (math.pi / 6, "e^{iπ/6}", (8, -2), "start"),
    (1.0, "e^{i·1}", (8, -6), "start"),
    (math.pi / 2, "i = e^{iπ/2}", (8, -8), "start"),
    (2.0, "e^{2i}", (-8, -8), "end"),
    (math.pi, "−1 = e^{iπ}", (-4, 16), "end"),
]
# velocity f'(phi) = i f(phi): the position turned by 90 degrees, i.e. tangent to the circle (derivation two)
for phi in (math.pi / 2, 2.6, math.pi):  # placed where no label sits
    e = cmath.exp(1j * phi)
    v = 1j * e * 0.36
    px, py = pl(e.real, e.imag)
    qx, qy = pl((e + v).real, (e + v).imag)
    svg.arrow(px, py, qx, qy, color="green", width=1.8)
for phi, lab, (dx, dy), anc in pts:
    e = cmath.exp(1j * phi)
    px, py = pl(e.real, e.imag)
    if phi == 1.0:
        svg.line(ox, oy, px, py, color="blue", width=1.6, opacity=0.8)
    svg.circle(px, py, 4.2, fill="blue", stroke="surface", width=1.2)
    svg.text(px + dx, py + dy, lab, size=12.5, anchor=anc, color="ink")
svg.text(185, 318, "角度 φ 用弧度量：沿单位圆走过的弧长就是 φ。", size=12, anchor="middle")
svg.text(185, 338, "|e^{iφ}|² = cos²φ + sin²φ = 1，所以这些点都在圆上。", size=12, anchor="middle")
svg.text(185, 358, "绿箭头：速度 i·e^{iφ}，永远垂直于半径（推导二）。", size=12, anchor="middle", color="green")
svg.text(185, 378, "e^{iπ} = −1：转半圈就是“掉头”。", size=12, anchor="middle", color="muted")

svg.line(372, 44, 372, 388, color="rule", width=1)

# ---------------- right: Taylor partial sums for phi = 2
phi = 2.0
s2 = 92
pr = Plane(392 + 1.45 * s2, 64 + 2.3 * s2, s2)
svg.text(556, 26, "② φ = 2 时，级数 Σ (2i)ⁿ/n! 一项一项走向 e^{2i}", size=13.5, anchor="middle", weight="600")
ox2, oy2 = pr(0, 0)
a0, b0 = pr(-1.4, 0)
a1, _ = pr(1.55, 0)
svg.line(a0, b0, a1, b0, color="rule", width=1)
c0, d0 = pr(0, -0.4)
_, d1 = pr(0, 2.3)
svg.line(c0, d0, c0, d1, color="rule", width=1)
svg.path(arc_path(ox2, oy2, s2, math.radians(-6), math.radians(186)), stroke="muted", width=1, dash="4 3")
target = cmath.exp(1j * phi)
S = 0
prev = pr(0, 0)
labels = {
    0: ("1", (0, 16), "middle"),
    1: ("+2i", (8, 4), "start"),
    2: ("−2", (0, -8), "middle"),
    3: ("−1.333i", (7, 4), "start"),
    4: ("+0.667", (14, 16), "middle"),
    5: ("+0.267i", (8, 2), "start"),
}
sums = []
for n in range(0, 12):
    term = (1j * phi) ** n / math.factorial(n)
    S += term
    cur = pr(S.real, S.imag)
    sums.append(S)
    col = "orange" if n % 2 == 0 else "violet"
    if n <= 7:
        svg.arrow(prev[0], prev[1], cur[0], cur[1], color=col, width=2.0 if n < 6 else 1.6)
    else:
        svg.line(prev[0], prev[1], cur[0], cur[1], color=col, width=1.4)
    if n in labels:
        lab, (dx, dy), anc = labels[n]
        mx, my = (prev[0] + cur[0]) / 2, (prev[1] + cur[1]) / 2
        svg.text(mx + dx, my + dy, lab, size=12, anchor=anc, color=col)
    prev = cur
tx, ty = pr(target.real, target.imag)
svg.circle(tx, ty, 5.5, fill="blue", stroke="surface", width=1.2)
svg.text(tx + 10, ty - 16, "e^{2i} = −0.416 + 0.909i", size=12, color="blue", weight="600", anchor="start")
svg.line(ox2, oy2, tx, ty, color="blue", width=1.2, opacity=0.5)
# partial sum markers for the first few
for n, S_ in enumerate(sums[:5]):
    px, py = pr(S_.real, S_.imag)
    svg.circle(px, py, 2.6, fill="ink")
svg.text(556, 318, "每一项 = 上一项转 90° 再乘 2/n（橙：实数项，紫：虚数项）", size=12, anchor="middle")
svg.text(556, 338, "前 8 项之和 −0.422 + 0.908i，离 e^{2i} 还差 0.006；", size=12, anchor="middle")
svg.text(556, 358, "前 12 项之和离 e^{2i} 只差 8.5 × 10⁻⁶", size=12, anchor="middle", color="muted")
svg.save("m03-euler.svg")
print("S7", sums[7], abs(sums[7] - target), "S11", abs(sums[11] - target))
