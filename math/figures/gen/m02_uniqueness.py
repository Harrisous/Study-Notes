"""Figure M2.6: in the spectral norm the best rank-1 approximation is not unique; in Frobenius it is.

A = diag(3, 2, 1). Rank-1 candidates B(b) = diag(b, 0, 0), b in [0, 6]:
    ||A - B||_2 = max(|3 - b|, 2, 1)   (computed with np.linalg.norm(., 2))
    ||A - B||_F = sqrt((3 - b)^2 + 4 + 1)
The spectral error equals the optimum sigma_2 = 2 for every b in [1, 5]; the Frobenius error has a unique
minimum sqrt5 = 2.236 at b = 3 (the truncated SVD). math-refs-doc N23 / F02-2.
Writes math/figures/m02-uniqueness.svg
"""

import numpy as np

from m02_svg import SVG, polyline_d

A = np.diag([3.0, 2.0, 1.0])
b = np.linspace(0, 6, 601)
e2 = np.array([np.linalg.norm(A - np.diag([x, 0, 0]), 2) for x in b])
eF = np.array([np.linalg.norm(A - np.diag([x, 0, 0]), "fro") for x in b])
assert np.allclose(e2, np.maximum(np.abs(3 - b), 2))
assert np.allclose(eF, np.sqrt((3 - b) ** 2 + 5))

W, H = 640, 300
svg = SVG(W, H, "谱范数下最优秩 1 近似不唯一，Frobenius 范数下唯一")
x0, x1, y0, y1 = 64, 600, 40, 240
ymax = 4.5


def X_(v):
    return x0 + (x1 - x0) * v / 6.0


def Y_(v):
    return y1 - (y1 - y0) * v / ymax


for v in range(0, 5):
    svg.line(x0, Y_(v), x1, Y_(v), color="grid", width=1)
    svg.text(x0 - 8, Y_(v) + 4, str(v), size=11, color="muted", anchor="end")
for v in range(0, 7):
    svg.line(X_(v), y1, X_(v), y1 + 5, color="muted", width=1)
    svg.text(X_(v), y1 + 19, str(v), size=11, color="muted", anchor="middle")
svg.text((x0 + x1) / 2, y1 + 40, "候选 *B* = diag(b, 0, 0) 中的 b", size=12, color="muted", anchor="middle")
svg.text(20, (y0 + y1) / 2, "误差", size=12, color="muted", anchor="middle", rotate=-90)
svg.rect(X_(1), Y_(4.4), X_(5) - X_(1), Y_(0) - Y_(4.4), fill="blue", fill_opacity=0.06)
svg.path(polyline_d([(X_(p), Y_(q)) for p, q in zip(b, e2)]), stroke="blue", width=2.4, dash="7 4")
svg.path(polyline_d([(X_(p), Y_(q)) for p, q in zip(b, eF)]), stroke="ink", width=2.2)
svg.circle(X_(3), Y_(np.sqrt(5)), 5, fill="ink")
# label sits just above the y = 3 grid line (not on it), with a thin leader down to the minimum
svg.text(X_(3), Y_(3) - 7, "唯一最小：b = 3，√5 = 2.236", size=12, anchor="middle")
svg.line(X_(3), Y_(3) - 3, X_(3), Y_(np.sqrt(5)) - 7, color="ink", width=1, dash="2 2", opacity=0.6)
svg.text(X_(3), Y_(2) + 18, "平台：b ∈ [1, 5] 全都是 2 = σ_{2}", size=12, color="blue", anchor="middle")
svg.text(X_(5.05), Y_(3.85), "‖*A* − *B*‖_{F}", size=12, anchor="end")
svg.text(X_(0.2), Y_(1.62), "‖*A* − *B*‖_{2}", size=12, color="blue")
svg.text(X_(0.1), Y_(4.25), "*A* = diag(3, 2, 1)，候选的秩都 ≤ 1", size=12, color="muted")

svg.save("m02-uniqueness.svg")
