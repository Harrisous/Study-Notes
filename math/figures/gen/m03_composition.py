"""Figure M3.5: composing rotations adds angles; rotating both vectors keeps only the relative angle.

Left: e^{i30deg} * e^{i45deg} = e^{i75deg} = 0.2588 + 0.9659i, drawn as two successive turns of (1, 0).
Right: q = (1, 2), k = (0.5, -1), theta = 0.1 rad per position. The query at position t is turned by
       t*theta, the key at position j by j*theta. For (t, j) = (5, 2) and (13, 10) the angle between the
       turned vectors is 126.87 + 17.19 = 144.06 degrees and the dot product is -2.024045 in both cases.
Writes math/figures/m03-composition.svg
"""

import cmath
import math

import numpy as np

from m03_svg import SVG, Plane, arc_path, fmt


def R(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


W, H = 740, 392
svg = SVG(W, H, "旋转的复合是角度相加；两个向量各转各的，夹角只取决于转角之差")

# ---------------- left
s = 128
pl = Plane(34 + 1.12 * s, 64 + 1.2 * s, s)
svg.text(180, 24, "① 先转 30°、再转 45°，等于一次转 75°", size=13.5, anchor="middle", weight="600")
ox, oy = pl(0, 0)
a0, b0 = pl(-1.1, 0)
a1, _ = pl(1.25, 0)
svg.line(a0, b0, a1, b0, color="rule", width=1)
c0, d0 = pl(0, -0.25)
_, d1 = pl(0, 1.2)
svg.line(c0, d0, c0, d1, color="rule", width=1)
svg.path(arc_path(ox, oy, s, math.radians(-12), math.radians(192)), stroke="muted", width=1, dash="4 3")
e30, e45, e75 = (cmath.exp(1j * math.radians(a)) for a in (30, 45, 75))
assert abs(e30 * e45 - e75) < 1e-15
p0 = pl(1, 0)
p30 = pl(e30.real, e30.imag)
p45 = pl(e45.real, e45.imag)
p75 = pl(e75.real, e75.imag)
svg.arrow(ox, oy, *p0, color="muted", width=1.8)
svg.arrow(ox, oy, *p45, color="violet", width=1.4, dash="5 3")
svg.arrow(ox, oy, *p30, color="blue", width=2.4)
svg.arrow(ox, oy, *p75, color="green", width=2.6)
svg.path(arc_path(ox, oy, 0.42 * s, 0, math.radians(30)), stroke="red", width=2)
svg.text(ox + 0.54 * s * math.cos(math.radians(15)), oy - 0.54 * s * math.sin(math.radians(15)) + 5, "30°", size=12, color="red", anchor="middle")
svg.path(arc_path(ox, oy, 0.42 * s, math.radians(30), math.radians(75)), stroke="orange", width=2)
svg.text(ox + 0.55 * s * math.cos(math.radians(55)), oy - 0.55 * s * math.sin(math.radians(55)) + 5, "45°", size=12, color="orange", anchor="middle")
svg.path(arc_path(ox, oy, 0.24 * s, 0, math.radians(75)), stroke="green", width=1.4, dash="3 2")
svg.text(p0[0] + 4, p0[1] + 16, "1", size=12.5, color="muted")
svg.text(p30[0] + 8, p30[1] + 4, "e^{i30°}", size=12.5, color="blue")
svg.text(p45[0] + 8, p45[1] + 2, "e^{i45°}", size=12.5, color="violet")
svg.text(p75[0] - 6, p75[1] - 10, "e^{i30°}e^{i45°} = e^{i75°} = 0.259 + 0.966i", size=12, color="green", weight="600", anchor="middle")
svg.text(180, 300, "复数：e^{i30°} · e^{i45°} = e^{i(30°+45°)}", size=12.5, anchor="middle")
svg.text(180, 321, "矩阵：R(30°) R(45°) = R(75°)", size=12.5, anchor="middle")
svg.text(180, 342, "(0.866 + 0.5i)(0.707 + 0.707i) = 0.259 + 0.966i", size=12, anchor="middle", color="muted")

svg.line(362, 44, 362, 360, color="rule", width=1)

# ---------------- right: relative angle
q = np.array([1.0, 2.0])
k = np.array([0.5, -1.0])
th = 0.1
svg.text(551, 24, "② q、k 按位置各转 tθ、jθ（θ = 0.1）", size=13.5, anchor="middle", weight="600")
s2 = 36
for col_i, (t, j) in enumerate([(5, 2), (13, 10)]):
    cx0 = 378 + col_i * 182
    pr = Plane(cx0 + (2.35 if col_i == 0 else 1.85) * s2, 64 + 2.55 * s2, s2)
    ox2, oy2 = pr(0, 0)
    svg.text(cx0 + 86, 52, "(t, j) = (%d, %d)" % (t, j), size=12.5, anchor="middle", weight="600")
    a0, b0 = pr(-2.3 if col_i == 0 else -1.8, 0)
    a1, _ = pr(2.3 if col_i == 0 else 2.8, 0)
    svg.line(a0, b0, a1, b0, color="rule", width=1)
    c0, d0 = pr(0, -1.6)
    _, d1 = pr(0, 2.5)
    svg.line(c0, d0, c0, d1, color="rule", width=1)
    qr = R(t * th) @ q
    kr = R(j * th) @ k
    # originals, faint
    for v, col in ((q, "blue"), (k, "violet")):
        px, py = pr(*v)
        svg.line(ox2, oy2, px, py, color=col, width=1.2, dash="3 3", opacity=0.6)
    pq = pr(*qr)
    pk = pr(*kr)
    svg.arrow(ox2, oy2, *pq, color="blue", width=2.4)
    svg.arrow(ox2, oy2, *pk, color="violet", width=2.4)
    aq = math.atan2(qr[1], qr[0])
    ak = math.atan2(kr[1], kr[0])
    # rotation arcs (red) for q and k
    svg.path(arc_path(ox2, oy2, 1.75 * s2, math.atan2(q[1], q[0]), aq), stroke="red", width=1.4)
    svg.path(arc_path(ox2, oy2, 0.85 * s2, math.atan2(k[1], k[0]), ak), stroke="red", width=1.4)
    # angle between them (green)
    svg.path(arc_path(ox2, oy2, 0.5 * s2, ak, aq), stroke="green", width=2)
    ang = math.degrees(aq - ak)
    dot = qr @ kr
    svg.text(pq[0] + (6 if qr[0] >= 0 else -4), pq[1] - 6, "q 转 %.1f°" % math.degrees(t * th), size=11.5, color="blue", anchor="start" if qr[0] >= 0 else "end")
    if col_i == 0:
        svg.text(pk[0] + 4, pk[1] + 16, "k 转 %.1f°" % math.degrees(j * th), size=11.5, color="violet", anchor="end")
    else:
        svg.text(pk[0] + 6, pk[1] + 4, "k 转 %.1f°" % math.degrees(j * th), size=11.5, color="violet", anchor="start")
    svg.text(cx0 + 86, 262, "夹角 %.2f°" % ang, size=12.5, anchor="middle", color="green", weight="600")
    svg.text(cx0 + 86, 282, ("点积 %.6f" % dot).replace("-", "−"), size=12.5, anchor="middle")
    assert abs(dot - (-2.0240451470110883)) < 1e-12
svg.text(551, 314, "虚线：旋转前的 q = (1, 2)、k = (0.5, −1)，夹角 126.87°，点积 −1.5", size=12, anchor="middle", color="muted")
svg.text(551, 335, "两对位置都差 3：夹角都增加 3 × 0.1 rad = 17.19°", size=12, anchor="middle")
svg.text(551, 356, "绝对位置不同，旋转后的点积完全相同：−2.024045", size=12, anchor="middle", color="green")
svg.text(W / 2, 384, "红色弧线是各自被转过的角度，绿色弧线是转完之后 q 与 k 的夹角。", size=11.5, anchor="middle", color="muted")
svg.save("m03-composition.svg")
