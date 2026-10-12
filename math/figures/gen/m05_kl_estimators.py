"""Figure M5.9: the single-sample KL estimators as functions of the ratio w = p(x)/q(x) (Lesson 8 notation; Schulman writes r).

(a) k1 = -ln w, k2 = (ln w)^2 / 2, k3 = (w - 1) - ln w on a log w axis, with the points w = 0.5, 1, 2
    (research notes N40: 0.6931/0.2402/0.1931 and -0.6931/0.2402/0.3069).
(b) k3 is the vertical gap between ln w and its tangent at w = 1, the line w - 1.
Colours follow Lesson 8's Fig. 8.9 (k1 red, k2 amber, k3 blue). Computed here.
Writes math/figures/m05-kl-estimators.svg
"""

import math

from m05_svg import SVG

W, H = 760, 400
svg = SVG(W, H, "三个单样本 KL 估计量随比值 w 的变化")


def k1(r):
    return -math.log(r)


def k2(r):
    return 0.5 * math.log(r) ** 2


def k3(r):
    return (r - 1) - math.log(r)


# ---------------- (a)
x0, x1, y0, y1 = 60, 400, 40, 330
LR0, LR1 = math.log10(0.1), math.log10(10)
YMIN, YMAX = -2.5, 4.0


def PX(r):
    return x0 + (math.log10(r) - LR0) / (LR1 - LR0) * (x1 - x0)


def PY(v):
    return y1 - (v - YMIN) / (YMAX - YMIN) * (y1 - y0)


for r, lab in ((0.1, "0.1"), (0.2, "0.2"), (0.5, "0.5"), (1, "1"), (2, "2"), (5, "5"), (10, "10")):
    svg.line(PX(r), y0, PX(r), y1, color="grid", width=1)
    svg.text(PX(r), y1 + 16, lab, size=11, anchor="middle", color="muted")
for v in range(-2, 5):
    svg.line(x0, PY(v), x1, PY(v), color="grid", width=1)
    svg.text(x0 - 6, PY(v) + 4, str(v).replace("-", "−"), size=11, anchor="end", color="muted")
svg.line(x0, PY(0), x1, PY(0), color="muted", width=1.2)
svg.line(x0, y0, x0, y1, color="muted", width=1.2)
svg.text((x0 + x1) / 2, 22, "(a) 单个样本上的取值", size=13, anchor="middle", weight="600")
svg.text((x0 + x1) / 2, y1 + 36, "w = p(x)/q(x)（对数刻度）", size=12, anchor="middle", color="muted")


def curve(f, col, dash=None):
    seg, n = [], 400
    for k in range(n + 1):
        r = 10 ** (LR0 + (LR1 - LR0) * k / n)
        v = f(r)
        if YMIN <= v <= YMAX:
            seg.append((PX(r), PY(v)))
        elif seg:
            svg.path("M" + " L".join("%.1f %.1f" % xy for xy in seg), stroke=col, width=2.4, dash=dash)
            seg = []
    if seg:
        svg.path("M" + " L".join("%.1f %.1f" % xy for xy in seg), stroke=col, width=2.4, dash=dash)


curve(k1, "red")
curve(k2, "amber")
curve(k3, "blue")
for r in (0.5, 1, 2):
    for f, col in ((k1, "red"), (k2, "amber"), (k3, "blue")):
        svg.circle(PX(r), PY(f(r)), 3.6, fill=col)
for i, (lab, col) in enumerate((("k1 = −ln w", "red"), ("k2 = ½(ln w)²", "amber"), ("k3 = (w − 1) − ln w", "blue"))):
    yy = PY(3.75 - 0.36 * i)
    svg.line(PX(0.108), yy - 4, PX(0.16), yy - 4, color=col, width=2.4)
    svg.text(PX(0.17), yy, lab, size=12, color=col, weight="600")
svg.text(PX(2) - 8, PY(k1(2)) + 18, "k1 = −0.693 < 0", size=12, color="red", anchor="end")
# below the three points at w = 0.5: no curve enters the region w < 1, value < 0
svg.text(PX(0.5), PY(-0.5), "w = 0.5", size=12, color="muted", anchor="middle")
svg.text(PX(1), PY(-1.9), "w = 1：三者都是 0", size=12, anchor="middle", color="muted")

# ---------------- (b)
bx0, bx1, by0, by1 = 470, 740, 40, 330
RMIN, RMAX, BYMIN, BYMAX = 0.0, 3.0, -2.0, 2.2


def QX(r):
    return bx0 + (r - RMIN) / (RMAX - RMIN) * (bx1 - bx0)


def QY(v):
    return by1 - (v - BYMIN) / (BYMAX - BYMIN) * (by1 - by0)


for r in range(0, 4):
    svg.line(QX(r), by0, QX(r), by1, color="grid", width=1)
    svg.text(QX(r), by1 + 16, str(r), size=11, anchor="middle", color="muted")
for v in (-2, -1, 0, 1, 2):
    svg.line(bx0, QY(v), bx1, QY(v), color="grid", width=1)
    svg.text(bx0 - 6, QY(v) + 4, str(v).replace("-", "−"), size=11, anchor="end", color="muted")
svg.line(bx0, QY(0), bx1, QY(0), color="muted", width=1.2)
svg.line(bx0, by0, bx0, by1, color="muted", width=1.2)
svg.text((bx0 + bx1) / 2, 22, "(b) k3 = 切线与 ln w 的竖直距离", size=13, anchor="middle", weight="600")
svg.text((bx0 + bx1) / 2, by1 + 36, "w（线性刻度）", size=12, anchor="middle", color="muted")
# ln r clipped at BYMIN
rs = [math.exp(BYMIN) + k * (RMAX - math.exp(BYMIN)) / 300 for k in range(301)]
svg.path("M" + " L".join("%.1f %.1f" % (QX(r), QY(math.log(r))) for r in rs), stroke="ink", width=2.2)
svg.text(QX(2.95), QY(1.32), "ln w", size=12, anchor="end")
# tangent r - 1 clipped to the box
ra, rb = max(RMIN, BYMIN + 1), min(RMAX, BYMAX + 1)
svg.line(QX(ra), QY(ra - 1), QX(rb), QY(rb - 1), color="orange", width=1.7, dash="7 4")
svg.text(QX(2.3), QY(1.3) - 16, "切线 w − 1", size=12, color="orange", anchor="end")
svg.circle(QX(1), QY(0), 4, fill="ink")
for i, r in enumerate((0.5, 2.0)):
    svg.line(QX(r), QY(math.log(r)), QX(r), QY(r - 1), color="blue", width=3)
    svg.text(QX(0.08), QY(1.95) + 18 * i, "w = %g：k3 = %.3f" % (r, k3(r)), size=12, color="blue",
             weight="600")
svg.text(QX(1.5), QY(-1.45), "ln 是凹函数，切线总在上方", size=12, anchor="middle", color="muted")
svg.text(QX(1.5), QY(-1.45) + 17, "所以 k3 ≥ 0，只在 w = 1 时为 0", size=12, anchor="middle", color="muted")

svg.save("m05-kl-estimators.svg")
