"""Figure M3.3: a complex number is a point of the plane, and multiplying by one scales and rotates.

(a) z = 1 + 2i: modulus sqrt 5 = 2.236, argument 63.43 deg; its conjugate 1 - 2i is the mirror image.
(b) multiplying by i turns z by 90 deg: i z = -2 + i; doing it twice gives i^2 z = -z = -1 - 2i.
(c) multiplying by w = 1 + i (modulus sqrt 2, argument 45 deg): w z = -1 + 3i,
    modulus sqrt 10 = sqrt 2 * sqrt 5, argument 108.43 deg = 63.43 + 45.
All numbers are computed below. Writes math/figures/m03-complex-mult.svg
"""

import cmath
import math

from m03_svg import SVG, Plane, arc_path, fmt

z = 1 + 2j
iz = 1j * z
iiz = 1j * iz
w = 1 + 1j
wz = w * z
assert iz == -2 + 1j and iiz == -z and wz == -1 + 3j
assert abs(abs(wz) - abs(w) * abs(z)) < 1e-12


def deg(c):
    return math.degrees(cmath.phase(c))


W, H = 740, 372
svg = SVG(W, H, "复数是平面上的点；乘以一个复数 = 把长度乘上它的模、把角度加上它的辐角")
s = 38
YTOP = 64
Y_MAX, Y_MIN = 3.35, -2.35
oy = YTOP + Y_MAX * s
PW = W / 3


def axes(pl, x_min, x_max, names=False, y_top=Y_MAX):
    a0, b0 = pl(x_min, 0)
    a1, _ = pl(x_max, 0)
    svg.line(a0, b0, a1, b0, color="rule", width=1)
    c0, d0 = pl(0, Y_MIN)
    _, d1 = pl(0, y_top)
    svg.line(c0, d0, c0, d1, color="rule", width=1)
    if names:
        svg.text(a1 - 2, b0 + 15, "实轴", size=11, anchor="end", color="muted")
        svg.text(c0 + 5, d1 + 10, "虚轴", size=11, color="muted")
    # unit ticks
    for k in range(math.ceil(x_min), math.floor(x_max) + 1):
        if k == 0:
            continue
        tx, ty = pl(k, 0)
        svg.line(tx, ty - 3, tx, ty + 3, color="muted", width=1)
    for k in range(math.ceil(Y_MIN), math.floor(y_top) + 1):
        if k == 0:
            continue
        tx, ty = pl(0, k)
        svg.line(tx - 3, ty, tx + 3, ty, color="muted", width=1)


def vec(pl, c, color, width=2.4, dash=None):
    ox, oy_ = pl(0, 0)
    px, py = pl(c.real, c.imag)
    svg.arrow(ox, oy_, px, py, color=color, width=width, dash=dash)
    return px, py


# ---------- (a)
pl = Plane(0 * PW + 30 + 1.3 * s, oy, s)
svg.text(PW / 2, 24, "(a) z = 1 + 2i 是平面上的一个点", size=13.5, anchor="middle", weight="600")
axes(pl, -1.3, 3.0, names=True)
px, py = vec(pl, z, "blue")
svg.text(px + 6, py - 2, "z = 1 + 2i", size=12.5, color="blue", weight="600")
cx, cy = vec(pl, z.conjugate(), "muted", width=1.8, dash="5 3")
svg.text(cx + 6, cy + 6, "z̄ = 1 − 2i", size=12.5, color="muted")
ox, oy0 = pl(0, 0)
svg.path(arc_path(ox, oy0, 0.62 * s, 0, cmath.phase(z)), stroke="red", width=1.6)
svg.text(ox + 0.95 * s, oy0 - 0.62 * s, "63.43°", size=11.5, color="red")
# modulus label along the arrow
mx, my = pl(0.5 * z.real - 0.28, 0.5 * z.imag + 0.12)
svg.text(mx, my, "|z| = √5", size=12, color="blue", anchor="end")
# dotted coordinates
svg.line(px, py, px, oy0, color="blue", width=1, dash="2 3", opacity=0.6)
svg.line(px, py, ox, py, color="blue", width=1, dash="2 3", opacity=0.6)
svg.text(PW / 2, 318, "实部 1 = 横坐标，虚部 2 = 纵坐标", size=12, anchor="middle")
svg.text(PW / 2, 337, "共轭 z̄：关于实轴的镜像", size=12, anchor="middle", color="muted")

# ---------- (b)
pl = Plane(PW + 26 + 2.45 * s, oy, s)
svg.text(PW + PW / 2, 24, "(b) 乘以 i：逆时针转 90°", size=13.5, anchor="middle", weight="600")
axes(pl, -2.45, 1.9)
vec(pl, z, "muted", width=1.8)
zx, zy = pl(z.real, z.imag)
svg.text(zx + 5, zy + 2, "z", size=12.5, color="muted")
bx, by = vec(pl, iz, "blue")
svg.text(bx - 2, by - 8, "iz = −2 + i", size=12.5, color="blue", weight="600", anchor="middle")
dx_, dy_ = vec(pl, iiz, "violet", width=1.8, dash="5 3")
svg.text(dx_ - 4, dy_ + 14, "i²z = −z", size=12.5, color="violet", anchor="middle")
ox, oy0 = pl(0, 0)
svg.path(arc_path(ox, oy0, 0.9 * s, cmath.phase(z), cmath.phase(iz)), stroke="red", width=1.6)
m = (cmath.phase(z) + cmath.phase(iz)) / 2
svg.text(ox + 1.2 * s * math.cos(m), oy0 - 1.2 * s * math.sin(m) + 4, "90°", size=11.5, color="red", anchor="middle")
a0 = cmath.phase(iz)
a1 = cmath.phase(iiz) + 2 * math.pi
svg.path(arc_path(ox, oy0, 0.62 * s, a0, a1), stroke="violet", width=1.4, dash="3 2")
svg.text(PW + PW / 2, 318, "i(a + bi) = −b + ai：正是 R(90°)", size=12, anchor="middle")
svg.text(PW + PW / 2, 337, "再乘一次 i 共转 180°：i² = −1", size=12, anchor="middle", color="violet")

# ---------- (c)
pl = Plane(2 * PW + 30 + 1.75 * s, oy, s)
svg.text(2 * PW + PW / 2, 24, "(c) 乘以 w = 1 + i：拉长 √2 倍、转 45°", size=13.5, anchor="middle", weight="600")
axes(pl, -1.75, 2.35, y_top=2.75)
vec(pl, z, "muted", width=1.8)
zx, zy = pl(z.real, z.imag)
svg.text(zx + 6, zy + 4, "z（√5，63.43°）", size=12, color="muted")
wx, wy = vec(pl, w, "orange", width=2.0)
svg.text(wx + 6, wy + 12, "w（√2，45°）", size=12, color="orange")
rx, ry = vec(pl, wz, "blue")
svg.text(rx + 8, ry - 8, "wz = −1 + 3i", size=12.5, color="blue", weight="600", anchor="middle")
ox, oy0 = pl(0, 0)
svg.path(arc_path(ox, oy0, 1.05 * s, cmath.phase(z), cmath.phase(wz)), stroke="red", width=1.6)
m = (cmath.phase(z) + cmath.phase(wz)) / 2
svg.text(ox + 1.36 * s * math.cos(m), oy0 - 1.36 * s * math.sin(m) + 4, "+45°", size=11.5, color="red", anchor="middle")
svg.text(2 * PW + PW / 2, 318, "|wz| = √10 = √2 × √5", size=12, anchor="middle")
svg.text(2 * PW + PW / 2, 337, "辐角 108.43° = 63.43° + 45°", size=12, anchor="middle", color="red")

svg.line(PW, 44, PW, 340, color="rule", width=1)
svg.line(2 * PW, 44, 2 * PW, 340, color="rule", width=1)
svg.text(W / 2, 364, "每个面板的网格刻度间隔 1。蓝色是乘法的结果，红色弧线是辐角增加的量。", size=11.5, anchor="middle", color="muted")
svg.save("m03-complex-mult.svg")
print("check", deg(z), deg(wz), abs(wz))
