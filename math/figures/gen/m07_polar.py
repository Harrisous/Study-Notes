"""Figure M7.10: what orthogonalisation does to a 2×2 update matrix, drawn as images of the unit circle.

G = [[3, 0], [4, 5]] = U Σ V^T with σ = (3√5, √5) = (6.708, 2.236), ‖G‖_F = √50 = 7.071,
v1 = (1, 1)/√2, v2 = (1, −1)/√2, u1 = (1, 3)/√10, u2 = (3, −1)/√10, and U V^T = rotation by 26.57°.
Panels: unit circle with v1, v2  ->  X0 = G/‖G‖_F (singular values 0.949, 0.316)
        -> X1 = one Newton–Schulz step with (3.4445, −4.7750, 2.0315) (0.752 along u1, 0.945 along u2)
        -> X10 = DeepSeek-V4 hybrid 8 + 2 steps (1.0002 along u1, 1.0005 along u2) ≈ U V^T.
Every point is computed here. Writes math/figures/m07-polar.svg
"""

import math

import numpy as np

from m07_svg import SVG, fmt

G = np.array([[3.0, 0.0], [4.0, 5.0]])
U, S, Vt = np.linalg.svd(G)
# fix signs so that u1 = (1,3)/√10 and v1 = (1,1)/√2 (up to the common sign)
if U[0, 0] < 0:
    U[:, 0] *= -1
    Vt[0] *= -1
if U[0, 1] < 0:
    U[:, 1] *= -1
    Vt[1] *= -1
assert np.allclose(U @ np.diag(S) @ Vt, G)
a, b, c = 3.4445, -4.7750, 2.0315


def ns(X, coeffs):
    A = X @ X.T
    return coeffs[0] * X + (coeffs[1] * A + coeffs[2] * A @ A) @ X


X0 = G / np.linalg.norm(G)
X1 = ns(X0, (a, b, c))
X = X0.copy()
for k in range(10):
    X = ns(X, (a, b, c) if k < 8 else (2, -1.5, 0.5))
X10 = X
# singular values of X10 along the SVD directions of G (u1 first), so the label order matches the text
Ug, _, Vgt = np.linalg.svd(G)
sv10 = np.diag(Ug.T @ X10 @ Vgt.T)
mats = [
    (np.eye(2), "单位圆", ["输入方向 *v*_{1}、*v*_{2}"]),
    (X0, "X_{0} = G / ‖G‖_{F}", ["奇异值 0.949、0.316"]),
    (X1, "X_{1}：一步 NS", ["沿 *u*_{1}：0.752", "沿 *u*_{2}：0.945"]),
    (X10, "X_{10}：V4 混合 8+2 步", ["奇异值 %.4f、%.4f" % (sv10[0], sv10[1]), "≈ UV^{⊤}（转 26.57°）"]),
]
W, H = 740, 300
svg = SVG(W, H, "正交化对一个 2×2 更新矩阵做了什么：单位圆的像")
PW = W / 4
sc = 66.0
th = np.linspace(0, 2 * math.pi, 241)
circ = np.stack([np.cos(th), np.sin(th)])
v1, v2 = Vt[0], Vt[1]
for k, (M, title, notes) in enumerate(mats):
    cx, cy = k * PW + PW / 2, 150.0

    def P(x, y):
        return cx + x * sc, cy - y * sc

    svg.text(cx, 26, title, size=13, anchor="middle", weight="600")
    svg.line(cx - 1.25 * sc, cy, cx + 1.25 * sc, cy, color="grid", width=1)
    svg.line(cx, cy - 1.25 * sc, cx, cy + 1.25 * sc, color="grid", width=1)
    # reference unit circle
    svg.path("M" + " L".join("%s %s" % (fmt(P(x, y)[0]), fmt(P(x, y)[1])) for x, y in circ.T) + " Z",
             stroke="muted", width=1, dash="3 3")
    img = M @ circ
    col = "ink" if k == 0 else ("green" if k == 3 else "orange")
    svg.path("M" + " L".join("%s %s" % (fmt(P(x, y)[0]), fmt(P(x, y)[1])) for x, y in img.T) + " Z",
             stroke=col, width=2.2, fill=col, fill_opacity=0.07)
    ox, oy = P(0, 0)
    for vec, vc, name in ((v1, "blue", "1"), (v2, "violet", "2")):
        w = M @ vec
        ex, ey = P(w[0], w[1])
        svg.arrow(ox, oy, ex, ey, color=vc, width=2.2)
    for i, line in enumerate(notes):
        svg.text(cx, 250 + 17 * i, line, size=12, anchor="middle", color="ink" if i == 0 else "muted")
    if k < 3:
        ax_ = (k + 1) * PW
        svg.arrow(ax_ - 12, cy, ax_ + 10, cy, color="muted", width=1.4)
# labels on the first panel
cx = PW / 2
svg.text(cx + v1[0] * sc + 4, 150 - v1[1] * sc - 6, "*v*_{1}", size=12, color="blue")
svg.text(cx + v2[0] * sc + 4, 150 - v2[1] * sc + 14, "*v*_{2}", size=12, color="violet")
svg.text(W / 2, H - 6, "蓝、紫箭头：*v*_{1}、*v*_{2} 被矩阵送到哪里（右奇异向量 → 左奇异向量 × 奇异值）。虚线圆：单位圆。",
         size=11, anchor="middle", color="muted")
svg.save("m07-polar.svg")
print("S", S, "U", U, "Vt", Vt)
print("X1 sv", np.linalg.svd(X1, compute_uv=False), "X1 v1->", X1 @ v1, np.linalg.norm(X1 @ v1),
      "v2->", np.linalg.norm(X1 @ v2))
print("X10", X10, np.linalg.svd(X10, compute_uv=False))
print("UVt", U @ Vt)
