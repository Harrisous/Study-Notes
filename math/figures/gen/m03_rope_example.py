"""Figure M3.10: the worked 4-dimensional RoPE example.

d = 4, b = 10000 -> theta = (1, 0.01). q = (1, 2, 2, 1), k = (0.5, -1, 1, 1); query at t = 5, key at j = 2.
Pair 0 (fast hand): q0 = (1, 2) turned 5 rad -> (2.2015, -0.3916); k0 = (0.5, -1) turned 2 rad -> (0.7012, 0.8708);
                   dot 1.2027 (was -1.5).
Pair 1 (slow hand): q1 = (2, 1) turned 0.05 rad -> (1.9475, 1.0987); k1 = (1, 1) turned 0.02 rad -> (0.9798, 1.0198);
                   dot 3.0286 (was 3).
Total 4.2314 (was 1.5). Writes math/figures/m03-rope-example.svg
"""

import math

import numpy as np

from m03_svg import SVG, Plane, arc_path, fmt


def R(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


q = np.array([1.0, 2.0, 2.0, 1.0])
k = np.array([0.5, -1.0, 1.0, 1.0])
t, j = 5, 2
theta = 10000.0 ** (-np.arange(0, 4, 2) / 4)
parts = []
for i in range(2):
    qi, ki = q[2 * i:2 * i + 2], k[2 * i:2 * i + 2]
    qr, kr = R(t * theta[i]) @ qi, R(j * theta[i]) @ ki
    parts.append((qi, ki, qr, kr, qi @ ki, qr @ kr))
total = sum(p[5] for p in parts)
assert abs(total - 4.231394) < 1e-6

W, H = 740, 408
svg = SVG(W, H, "4 维 RoPE 算例：两对维度各自旋转，点积是两对小点积之和")
svg.text(W / 2, 22, "q = (1, 2, 2, 1) 在位置 t = 5，k = (0.5, −1, 1, 1) 在位置 j = 2；d = 4，θ = (1, 0.01)",
         size=13.5, anchor="middle", weight="600")
s = 56
titles = ["对 0（快针，θ₀ = 1）：q 转 5 rad，k 转 2 rad", "对 1（慢针，θ₁ = 0.01）：q 转 0.05 rad，k 转 0.02 rad"]
for i in range(2):
    qi, ki, qr, kr, d0, d1 = parts[i]
    x0 = i * W / 2
    pl = Plane(x0 + W / 4 + (0 if i == 0 else -30), 62 + 2.55 * s, s)
    svg.text(x0 + W / 4, 50, titles[i], size=12.5, anchor="middle", weight="600", color="ink")
    ox, oy = pl(0, 0)
    a0, b0 = pl(-2.4 if i == 0 else -1.2, 0)
    a1, _ = pl(2.6 if i == 0 else 3.2, 0)
    svg.line(a0, b0, a1, b0, color="rule", width=1)
    c0, d0_ = pl(0, -1.35)
    _, d1_ = pl(0, 2.45)
    svg.line(c0, d0_, c0, d1_, color="rule", width=1)
    # originals dashed
    for v, col in ((qi, "blue"), (ki, "violet")):
        px, py = pl(*v)
        svg.line(ox, oy, px, py, color=col, width=1.4, dash="4 3", opacity=0.75)
    # rotation arcs
    aq0, ak0 = math.atan2(qi[1], qi[0]), math.atan2(ki[1], ki[0])
    aq1, ak1 = aq0 + t * theta[i], ak0 + j * theta[i]
    if i == 0:
        svg.path(arc_path(ox, oy, 0.62 * s, aq0, aq1), stroke="red", width=1.5, dash="5 2")
        svg.path(arc_path(ox, oy, 0.38 * s, ak0, ak1), stroke="red", width=1.5)
    pq, pk = pl(*qr), pl(*kr)
    svg.arrow(ox, oy, *pq, color="blue", width=2.6)
    svg.arrow(ox, oy, *pk, color="violet", width=2.6)
    # angle between rotated vectors
    lo_, hi_ = sorted([math.atan2(qr[1], qr[0]), math.atan2(kr[1], kr[0])])
    svg.path(arc_path(ox, oy, 0.95 * s, lo_, hi_), stroke="green", width=2.2)
    ang = math.degrees(hi_ - lo_)
    if i == 0:
        svg.text(pq[0] + 4, pq[1] + 18, "(2.202, −0.392)", size=11.5, color="blue", anchor="end")
        svg.text(pk[0] + 6, pk[1] - 4, "(0.701, 0.871)", size=11.5, color="violet")
        qo = pl(*qi)
        svg.text(qo[0] - 6, qo[1] - 4, "q⁽⁰⁾ = (1, 2)", size=11.5, color="blue", anchor="end")
        ko = pl(*ki)
        svg.text(ko[0] + 8, ko[1] + 6, "k⁽⁰⁾ = (0.5, −1)", size=11.5, color="violet")
        svg.text(ox - 0.62 * s - 4, oy + 0.1 * s, "q 转 286.5°（虚线弧）", size=11, color="red", anchor="end")
        svg.text(ox - 0.62 * s - 4, oy + 0.1 * s + 17, "k 转 114.6°（实线弧）", size=11, color="red", anchor="end")
    else:
        svg.text(pq[0] + 8, pq[1] + 16, "q⁽¹⁾", size=12, color="blue")
        svg.text(pk[0] - 6, pk[1] - 6, "k⁽¹⁾", size=12, color="violet", anchor="end")
        svg.text(ox + 1.1 * s, oy - 0.2 * s, "转角只有 2.9° 与 1.1°", size=11, color="red")
        svg.text(ox + 14, oy + 0.62 * s, "q⁽¹⁾ = (2, 1) → (1.948, 1.099)", size=11.5, color="blue")
        svg.text(ox + 14, oy + 0.95 * s, "k⁽¹⁾ = (1, 1) → (0.980, 1.020)", size=11.5, color="violet")
    mid = (lo_ + hi_) / 2
    rr = 1.4 if i == 0 else 2.05
    svg.text(ox + rr * s * math.cos(mid), oy - rr * s * math.sin(mid) + 4,
             "%.2f°" % ang, size=11.5, color="green", anchor="middle" if i == 0 else "start")
    yb = 62 + 2.55 * s + 1.45 * s + 26
    svg.text(x0 + W / 4, yb, ("旋转前小点积 %s，旋转后 %.4f" % (fmt(d0), d1)).replace("-", "−"), size=12.5, anchor="middle",
             color="ink", weight="600")
svg.line(W / 2, 40, W / 2, 330, color="rule", width=1)
svg.text(W / 2, 368, "总打分 = 1.2027 + 3.0286 = 4.2314（旋转前是 −1.5 + 3 = 1.5）", size=13, anchor="middle", color="green",
         weight="600")
svg.text(W / 2, 392, "虚线：旋转前；实线：旋转后；红弧：转过的角度；绿弧：转完后 q 与 k 的夹角。快针的贡献从 −1.5 变成 1.2027，慢针几乎没动。",
         size=11.5, anchor="middle", color="muted")
svg.save("m03-rope-example.svg")
print([round(p[5], 6) for p in parts], total)
