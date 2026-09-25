"""Figure 10.10: why the slow RoPE pairs break beyond the training length, and what YaRN's /40 does.

RoPE pair i of V2's 64-dim decoupled branch rotates by (relative distance) x theta_i, theta_i = 10000^(-2i/64).
Pre-training length L = 4096. Each dial shows the rotation angle as the relative distance grows from 0:
  (a) i = 0  : theta = 1, 4096 rad = 652 turns inside 4K -> every angle seen many times.
  (b) i = 31 : theta = 1.3335e-4, 4096 x theta = 0.546 rad = 31.3 deg -> only a 31.3-degree wedge is seen.
  (c) i = 31 at 131072 without YaRN: 17.48 rad = 2.78 turns -> most angles never seen in training.
  (d) i = 31 with YaRN (theta / 40): 131072 x theta / 40 = 25.0 deg, 163840 x theta / 40 = 31.3 deg.
The path is drawn as a spiral (radius grows with distance) and coloured green where the angle (mod 360)
lies inside the range seen in 4K pre-training, red where it does not. All geometry is computed here.
Writes deepseek/figures/l10-rope-clock.svg
"""

import math

from l10_svg import SVG, fmt

B, DH, L = 10000, 64, 4096
TH0 = 1.0
TH31 = B ** (-2 * 31 / DH)
SEEN31 = L * TH31                       # 0.546 rad: the wedge pair 31 covers in pre-training
assert abs(math.degrees(SEEN31) - 31.30) < 0.01
A_128K = 131072 * TH31                  # 17.48 rad without YaRN
A_YARN = 131072 * TH31 / 40             # 0.437 rad with YaRN
A_160K = 163840 * TH31 / 40             # = SEEN31 exactly
assert abs(A_160K - SEEN31) < 1e-12
assert abs(A_128K / (2 * math.pi) - 2.78) < 0.01 and abs(math.degrees(A_YARN) - 25.04) < 0.01

W, H = 760, 286
svg = SVG(W, H, "RoPE 最慢的一对维度：4K 训练见过的角度、128K 直接外推的角度、YaRN 之后的角度")
R, CY = 60, 124


def pt(cx, r, a):
    return cx + r * math.cos(a), CY - r * math.sin(a)


def wedge(cx, a, color="green", opacity=0.16):
    """Filled sector from angle 0 to a (counter-clockwise), radius R."""
    if a >= 2 * math.pi - 1e-9:
        svg.circle(cx, CY, R, fill=color, fill_opacity=opacity)
        return
    x1, y1 = pt(cx, R, 0)
    x2, y2 = pt(cx, R, a)
    large = 1 if a > math.pi else 0
    svg.path("M%s %s L%s %s A%s %s 0 %d 0 %s %s Z" % (fmt(cx), fmt(CY), fmt(x1), fmt(y1), R, R, large,
                                                       fmt(x2), fmt(y2)), stroke=None, fill=color,
             fill_opacity=opacity)


def dial(cx):
    svg.circle(cx, CY, R, stroke="rule", width=1.2)
    svg.line(cx, CY, cx + R + 6, CY, color="muted", width=1, dash="2 2")
    svg.text(cx + R + 8, CY + 4, "0°", size=11, color="muted")
    svg.circle(cx, CY, 2.2, fill="ink")


def spiral(cx, a_end, seen, r0=0.30, r1=0.90, step=0.02):
    """Angle path from 0 to a_end; radius grows linearly with the angle. Green inside [0, seen] mod 2pi."""
    n = max(2, int(a_end / step))
    run, run_col = [], None
    for k in range(n + 1):
        a = a_end * k / n
        col = "green" if (a % (2 * math.pi)) <= seen + 1e-9 else "red"
        x, y = pt(cx, R * (r0 + (r1 - r0) * k / n), a)
        if run_col is not None and col != run_col:
            run.append((x, y))
            svg.path("M" + " L".join("%s %s" % (fmt(px), fmt(py)) for px, py in run), stroke=run_col, width=2.2)
            run = []
        run.append((x, y))
        run_col = col
    svg.path("M" + " L".join("%s %s" % (fmt(px), fmt(py)) for px, py in run), stroke=run_col, width=2.2)
    return pt(cx, R * r1, a_end)


def pointer(cx, a, dash=None, color="ink"):
    x, y = pt(cx, R, a)
    svg.line(cx, CY, x, y, color=color, width=1.6, dash=dash)


CX = [95, 285, 475, 665]
titles = ["快的一对 i = 0", "慢的一对 i = 31", "i = 31 直接用到 128K", "i = 31 用 YaRN（θ ÷ 40）"]
lines = [("4K 内转了 652 圈，", "每个角度都见过很多次"),
         ("4K 内只转到 31.3°，", "训练只见过这一小段"),
         ("转了 2.78 圈，", "31.3° 以外的角度全没见过"),
         ("128K 只转到 25.0°，", "160K 恰好 31.3°")]
for cx, t, (l1, l2) in zip(CX, titles, lines):
    svg.text(cx, 30, t, size=13, anchor="middle", weight="600")
    svg.text(cx, CY + R + 32, l1, size=12, anchor="middle")
    svg.text(cx, CY + R + 50, l2, size=12, anchor="middle", color="muted")

# (a) i = 0: the whole circle is seen
wedge(CX[0], 2 * math.pi)
dial(CX[0])
svg.circle(CX[0], CY, R * 0.62, stroke="green", width=2.2)
svg.circle(CX[0], CY, R * 0.80, stroke="green", width=2.2)
svg.text(CX[0], CY - 10, "×652", size=12, anchor="middle", color="green", weight="600")

# (b) i = 31 inside 4K
wedge(CX[1], SEEN31)
dial(CX[1])
spiral(CX[1], SEEN31, SEEN31, r0=0.85, r1=0.85)
pointer(CX[1], SEEN31)
x, y = pt(CX[1], R + 10, SEEN31)
svg.text(x + 2, y, "4096", size=12, color="green", weight="600")

# (c) i = 31 at 131072 without YaRN
wedge(CX[2], SEEN31)
dial(CX[2])
spiral(CX[2], A_128K, SEEN31, r0=0.28, r1=0.92)
pointer(CX[2], A_128K, color="red")

# (d) i = 31 with YaRN
wedge(CX[3], SEEN31)
dial(CX[3])
spiral(CX[3], A_YARN, SEEN31, r0=0.85, r1=0.85)
pointer(CX[3], A_YARN)
pointer(CX[3], A_160K, dash="3 2", color="muted")
x, y = pt(CX[3], R + 10, A_YARN)
svg.text(x, y + 8, "128K", size=12, color="green", weight="600")

svg.text(W / 2, H - 12, "角度 = 相对距离 × θ_{i}；浅绿扇形 = 4K 预训练里出现过的角度；路径绿 = 见过，红 = 没见过", size=12,
         anchor="middle", color="muted")
svg.save("l10-rope-clock.svg")
