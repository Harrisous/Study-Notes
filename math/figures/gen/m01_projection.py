"""Figure M1.9: orthogonal projection of h = (2, -1, 3) onto the plane spanned by the orthonormal
vectors u1 = (1, 0, 0) and u2 = (0, 1, 1)/sqrt(2) (a "ramp"). Coordinates U^T h = (2, sqrt 2),
projection p = U U^T h = (2, 1, 1), discarded residual r = h - p = (0, -2, 2).
The 3-D scene is drawn with an orthographic camera (azimuth, elevation) computed below.
Writes math/figures/m01-projection.svg
"""

import os

import numpy as np

from m01_svg import SVG

AZIMUTH_DEG = float(os.environ.get('M01_AZ', '-30'))
ELEVATION_DEG = float(os.environ.get('M01_EL', '25'))

h = np.array([2.0, -1.0, 3.0])
u1 = np.array([1.0, 0.0, 0.0])
u2 = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
U = np.stack([u1, u2], axis=1)
coords = U.T @ h
p = U @ coords
r = h - p
assert np.allclose(p, [2, 1, 1]) and np.allclose(r, [0, -2, 2])
assert abs(r @ u1) < 1e-12 and abs(r @ u2) < 1e-12

S = 78
ORIGIN = np.array([150.0, 262.0])
AZ, EL = np.radians(AZIMUTH_DEG), np.radians(ELEVATION_DEG)    # orthographic camera
VIEW = np.array([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)])
RIGHT = np.cross([0.0, 0.0, 1.0], VIEW)
RIGHT /= np.linalg.norm(RIGHT)
UP = np.cross(VIEW, RIGHT)


def P(v):
    v = np.asarray(v, dtype=float)
    return ORIGIN + S * np.array([v @ RIGHT, -(v @ UP)])


W, H = 720, 432
svg = SVG(W, H, "把 h 正交投影到由 u1、u2 张成的平面上：保留投影 p，丢弃垂直分量 r")

# axes
for axis, name, length in ((np.array([1, 0, 0]), "x", 3.6), (np.array([0, 1, 0]), "y", 3.4), (np.array([0, 0, 1]), "z", 3.3)):
    a, b = P(-0.2 * axis), P(length * axis)
    svg.line(*a, *b, color="muted", width=1.1)
    svg.text(*(b + np.array([6, 4])), name, size=13, color="muted", italic=True)

# the plane z = y (spanned by u1 and u2), as a parallelogram patch
corners = [P(a * u1 + b * u2) for a, b in ((-0.5, -0.5), (3.0, -0.5), (3.0, 2.6), (-0.5, 2.6))]
svg.polygon([tuple(c) for c in corners], fill="blue", fill_opacity=0.08, stroke="blue", width=1)
svg.text(*(corners[1] + np.array([-4, 20])), "子空间：*u*_{1}、*u*_{2} 张成的平面", size=13, color="blue", anchor="middle")

# basis vectors
svg.arrow(*P([0, 0, 0]), *P(u1), color="blue", width=2.4)
svg.arrow(*P([0, 0, 0]), *P(u2), color="blue", width=2.4)
svg.text(*(P(u1) + np.array([4, 16])), "*u*_{1}", size=13, color="blue", weight="700")
svg.text(*(P(u2) + np.array([-8, -6])), "*u*_{2}", size=13, color="blue", weight="700", anchor="end")

# h, its projection and the residual
svg.arrow(*P([0, 0, 0]), *P(h), color="ink", width=2.4)
svg.arrow(*P([0, 0, 0]), *P(p), color="blue", width=2.8)
svg.line(*P(p), *P(h), color="red", width=1.8, dash="6 4")
# right-angle mark at p between r and the in-plane direction towards the origin
rn = r / np.linalg.norm(r)
pn = -p / np.linalg.norm(p)
k = 0.28
m1, m2, m3 = p + rn * k, p + rn * k + pn * k, p + pn * k
svg.path("M%.1f %.1f L%.1f %.1f L%.1f %.1f" % (*P(m1), *P(m2), *P(m3)), stroke="red", width=1.1)

svg.circle(*P(h), 3.5, fill="ink")
svg.circle(*P(p), 3.5, fill="blue")
svg.text(*(P(h) + np.array([-10, -10])), "*h* = (2, −1, 3)", size=14, anchor="end", weight="600")
svg.text(*(P(p) + np.array([8, 22])), "*p* = (2, 1, 1)", size=14, color="blue", weight="600")
mid = P((p + h) / 2)
svg.text(*(mid + np.array([14, -10])), "*r* = (0, −2, 2)", size=13, color="red")
svg.text(*(mid + np.array([14, 7])), "垂直于平面，被丢弃", size=12, color="red")

# worked numbers
X0, Y0, L = 474, 64, 27
svg.rect(X0 - 14, Y0 - 28, 262, 250, fill="surface", stroke="rule", rx=10)
rows = [
    ("*U* = [*u*_{1} *u*_{2}]，*U*^{⊤}*U* = *I*", "ink"),
    ("坐标 *U*^{⊤}*h* = (%.0f, %.2f)" % tuple(coords), "blue"),
    ("   即 (2, √2)", "muted"),
    ("投影 *p* = *UU*^{⊤}*h* = (2, 1, 1)", "blue"),
    ("残差 *r* = *h* − *p* = (0, −2, 2)", "red"),
    ("*r*·*u*_{1} = 0，*r*·*u*_{2} = 0", "red"),
    ("|*h*|^{2} = 14 = |*p*|^{2} + |*r*|^{2}", "ink"),
    ("         = 6 + 8", "muted"),
]
for i, (s, col) in enumerate(rows):
    svg.text(X0, Y0 + i * L, s, size=13, color=col)

svg.save("m01-projection.svg")
