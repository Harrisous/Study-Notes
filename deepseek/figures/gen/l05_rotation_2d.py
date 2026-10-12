"""Figure 5.2: the smallest unit of RoPE, a 2-D rotation (research notes R22).

theta = 0.3 rad per position, q = (1, 0.5), k = (0.8, -0.2).
Three (t, j) pairs with the same distance t - j = 3: (3, 0), (5, 2), (13, 10).
Every rotated coordinate, angle and dot product below is computed here.
Writes deepseek/figures/l05-rotation-2d.svg
"""

import math

from l05_svg import SVG

TH = 0.3
q = (1.0, 0.5)
k = (0.8, -0.2)
pairs = [(3, 0), (5, 2), (13, 10)]


def rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def ang(v):
    return math.atan2(v[1], v[0])


W, H = 750, 384
svg = SVG(W, H, "二维旋转：位置不同、距离相同的三对 (t, j) 得到同一个内积")
S = 86  # px per unit

dots = []
for n, (t, j) in enumerate(pairs):
    cx, cy = 125 + 250 * n, 168
    qr, kr = rot(q, t * TH), rot(k, j * TH)
    dot = qr[0] * kr[0] + qr[1] * kr[1]
    dots.append(dot)
    # axes and unit circle
    svg.line(cx - 112, cy, cx + 112, cy, color="grid", width=1.2)
    svg.line(cx, cy - 112, cx, cy + 112, color="grid", width=1.2)
    svg.circle(cx, cy, S, stroke="rule", width=1, dash="3 3")

    def P(v):
        return cx + v[0] * S, cy - v[1] * S

    # original vectors, faint
    for v, col in ((q, "blue"), (k, "violet")):
        x, y = P(v)
        svg.line(cx, cy, x, y, color=col, width=1.2, dash="4 3", opacity=0.45)
    # rotation arcs (red, thin): from original direction to rotated direction
    for v, a, rr in ((q, t * TH, 0.42), (k, j * TH, 0.30)):
        if a == 0:
            continue
        a0 = ang(v)
        steps = max(8, int(abs(a) / 0.05))
        pts = [P((rr * math.cos(a0 + a * s / steps), rr * math.sin(a0 + a * s / steps))) for s in range(steps + 1)]
        svg.polyline(pts[:-1], stroke="red", width=1.2)
        svg.arrow(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color="red", width=1.2)
    # angle between rotated vectors (ink arc)
    aq, ak = ang(qr), ang(kr)
    da = (aq - ak) % (2 * math.pi)
    rr = 0.62
    pts = [P((rr * math.cos(ak + da * s / 40), rr * math.sin(ak + da * s / 40))) for s in range(41)]
    svg.polyline(pts, stroke="ink", width=1.3)
    mid = ak + da / 2
    lx, ly = P((0.80 * math.cos(mid), 0.80 * math.sin(mid)))
    svg.text(lx, ly + 4, "%.1f°" % math.degrees(da), size=12, anchor="middle", weight="600")
    # rotated vectors
    xq, yq = P(qr)
    xk, yk = P(kr)
    svg.arrow(cx, cy, xq, yq, color="blue", width=2.4)
    svg.arrow(cx, cy, xk, yk, color="violet", width=2.4)
    # vector labels just beyond the tips
    for (x, y), v, lab, col in (((xq, yq), qr, "R(tθ)*q*", "blue"), ((xk, yk), kr, "R(jθ)*k*", "violet")):
        a = ang(v)
        tx, ty = x + 16 * math.cos(a), y - 16 * math.sin(a)
        anchor = "start" if math.cos(a) > 0.3 else ("end" if math.cos(a) < -0.3 else "middle")
        svg.text(tx, ty + 4, lab, size=12, anchor=anchor, color=col, weight="600")
    # captions
    svg.text(cx, 28, "t = %d，j = %d" % (t, j), size=14, anchor="middle", weight="700")
    svg.text(cx, 46, "q 转 %.1f rad，k 转 %.1f rad" % (t * TH, j * TH), size=12, anchor="middle", color="red")
    svg.text(cx, 300, "(%.4f, %.4f)·(%.4f, %.4f)" % (qr[0], qr[1], kr[0], kr[1]), size=11.5, anchor="middle",
             color="muted")
    svg.text(cx, 320, "内积 = %.4f" % dot, size=14, anchor="middle", weight="700", color="green")

assert max(dots) - min(dots) < 1e-12
a0 = math.degrees(ang(q) - ang(k))
# two lines: one line of 12-unit CJK text this long overflows the 750-unit viewBox
svg.text(W / 2, 350, "虚线是旋转前的 q = (1, 0.5)、k = (0.8, −0.2)：夹角 %.1f°，内积 %.1f。"
         % (a0, q[0] * k[0] + q[1] * k[1]), size=12, anchor="middle", color="ink")
svg.text(W / 2, 370, "三对的距离都是 t − j = 3：夹角都是 %.1f° + %.1f°（= 3 × 0.3 rad）= %.1f°。"
         % (a0, math.degrees(3 * TH), a0 + math.degrees(3 * TH)), size=12, anchor="middle", color="ink")

svg.save("l05-rotation-2d.svg")
