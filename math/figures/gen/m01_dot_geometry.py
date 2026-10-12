"""Figure M1.1: the dot product as |a| times the signed length of b's shadow on a.

Vectors a = (4, 1), b = (2, 3). Every coordinate below is computed, not eyeballed.
Writes math/figures/m01-dot-geometry.svg
"""

import math

import numpy as np

from m01_svg import SVG

a = np.array([4.0, 1.0])
b = np.array([2.0, 3.0])
dot = float(a @ b)
na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
cos = dot / (na * nb)
theta = math.degrees(math.acos(cos))
shadow = dot / na                      # signed length of b's shadow on a
p = (dot / (a @ a)) * a                # foot of the perpendicular from b to the line of a

W, H = 720, 352
O = (64, 300)                          # screen position of the origin
S = 56                                 # pixels per unit


def P(v):
    return O[0] + S * v[0], O[1] - S * v[1]


svg = SVG(W, H, "点积的几何意义：a·b 等于 |a| 乘以 b 在 a 方向上的影子长")

# grid and axes
for i in range(0, 6):
    x, _ = P((i, 0))
    svg.line(x, O[1], x, O[1] - 4.6 * S, color="grid", width=1)
    svg.text(x, O[1] + 18, str(i), size=11, anchor="middle", color="muted")
for j in range(0, 5):
    _, y = P((0, j))
    svg.line(O[0], y, O[0] + 5.4 * S, y, color="grid", width=1)
    if j:
        svg.text(O[0] - 10, y + 4, str(j), size=11, anchor="end", color="muted")
svg.line(O[0], O[1], O[0] + 5.5 * S, O[1], color="muted", width=1.2)
svg.line(O[0], O[1], O[0], O[1] - 4.7 * S, color="muted", width=1.2)

# the line through a (extended, faint)
far = a / na * 5.6
svg.line(*P((0, 0)), *P(far), color="blue", width=1, dash="2 4", opacity=0.6)

# the shadow of b on a: thick translucent band along a
svg.line(*P((0, 0)), *P(p), color="violet", width=7, opacity=0.28, cap="round")

# perpendicular drop from b to the line of a, with a right-angle mark
svg.line(*P(b), *P(p), color="violet", width=1.3, dash="4 3")
u = a / na
n = np.array([-u[1], u[0]])
k = 0.16
c1, c2, c3 = p - u * k, p - u * k + n * k, p + n * k
svg.path("M%.1f %.1f L%.1f %.1f L%.1f %.1f" % (*P(c1), *P(c2), *P(c3)), stroke="violet", width=1.1)

# angle arc between a and b
r = 0.9
ang_a, ang_b = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
x1, y1 = P((r * math.cos(ang_a), r * math.sin(ang_a)))
x2, y2 = P((r * math.cos(ang_b), r * math.sin(ang_b)))
svg.path("M%.1f %.1f A%.1f %.1f 0 0 0 %.1f %.1f" % (x1, y1, r * S, r * S, x2, y2), stroke="ink", width=1.2)
mid = (ang_a + ang_b) / 2
svg.text(*P((1.2 * math.cos(mid), 1.2 * math.sin(mid) - 0.1)), "θ", size=14, anchor="middle")

# the two vectors
svg.arrow(*P((0, 0)), *P(a), color="blue", width=2.4)
svg.arrow(*P((0, 0)), *P(b), color="violet", width=2.4)
svg.text(*(np.array(P(a)) + (8, 16)), "*a* = (4, 1)", size=14, color="blue")
svg.text(*(np.array(P(b)) + (-8, -10)), "*b* = (2, 3)", size=14, color="violet", anchor="end")

# label on the shadow
# bracket under the axis marking the shadow's horizontal extent is misleading (the shadow lies
# along a, not along the x axis), so the label sits below the ticks with a short leader instead
sx, sy = P(p * 0.62)
svg.line(sx, sy + 5, sx + 18, O[1] + 26, color="violet", width=1)
svg.text(sx + 20, O[1] + 38, "影子长 %.2f（沿 *a* 方向量）" % shadow, size=13, color="violet")

# worked numbers on the right
X0, Y0, L = 410, 58, 29
svg.rect(X0 - 16, Y0 - 30, 318, 262, fill="surface", stroke="rule", rx=10)
rows = [
    ("*a*·*b* = 4×2 + 1×3 = %d" % dot, "ink"),
    ("|*a*| = √17 ≈ %.2f，|*b*| = √13 ≈ %.2f" % (na, nb), "ink"),
    ("cos θ = %d ÷ (%.2f × %.2f) ≈ %.3f" % (dot, na, nb, cos), "ink"),
    ("θ ≈ %.1f°" % theta, "ink"),
    ("*b* 在 *a* 方向上的影子长", "violet"),
    ("   |*b*| cos θ = %d ÷ %.2f ≈ %.2f" % (dot, na, shadow), "violet"),
    ("*a*·*b* = |*a*| × 影子长", "blue"),
    ("   = %.2f × %.2f ≈ %d" % (na, shadow, round(na * shadow)), "blue"),
]
for i, (s, col) in enumerate(rows):
    svg.text(X0, Y0 + i * L, s, size=14, color=col)

svg.save("m01-dot-geometry.svg")
