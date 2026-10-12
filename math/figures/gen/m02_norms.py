"""Figure M2.4: two rulers for a matrix, for A = [[3, 0], [4, 5]].

Left: the stretch ||A x|| for unit inputs x = (cos t, sin t), t in [0, 180) degrees, computed pointwise.
The maximum is sigma_1 = 3 sqrt5 = 6.708 (at t = 45 deg, x = v1), the minimum sigma_2 = sqrt5 = 2.236
(at t = 135 deg, x = v2), and the root-mean-square over the circle is ||A||_F / sqrt2 = 5.
Right: ||A||_F^2 = sigma_1^2 + sigma_2^2 drawn as a right triangle (legs sigma_1, sigma_2).
Writes math/figures/m02-norms.svg
"""

import math

import numpy as np

from m02_svg import SVG, fmt, polyline_d

A = np.array([[3.0, 0.0], [4.0, 5.0]])
s = np.linalg.svd(A, compute_uv=False)
s1, s2 = float(s[0]), float(s[1])
fro = float(np.linalg.norm(A, "fro"))
t = np.linspace(0, 180, 721)
X = np.stack([np.cos(np.radians(t)), np.sin(np.radians(t))])
L = np.linalg.norm(A @ X, axis=0)
rms = float(np.sqrt((L ** 2).mean()))
assert abs(L.max() - s1) < 1e-4 and abs(L.min() - s2) < 1e-4 and abs(rms - fro / math.sqrt(2)) < 1e-3
tmax, tmin = float(t[L.argmax()]), float(t[L.argmin()])

W, H = 760, 330
svg = SVG(W, H, "谱范数与 Frobenius 范数：单位输入被拉长的倍数随方向变化")

# ---- left panel: line chart
x0, x1, y0, y1 = 64, 470, 52, 272          # plot box (y0 top)
ymax = 8.0


def X_(deg):
    return x0 + (x1 - x0) * deg / 180.0


def Y_(v):
    return y1 - (y1 - y0) * v / ymax


svg.text(20, 26, "① 单位输入 *x* = (cos θ, sin θ) 被拉长多少：‖*Ax*‖", size=14, weight="600")
for v in range(0, 9, 2):
    svg.line(x0, Y_(v), x1, Y_(v), color="grid", width=1)
    svg.text(x0 - 8, Y_(v) + 4, str(v), size=11, color="muted", anchor="end")
for d in range(0, 181, 45):
    svg.line(X_(d), y1, X_(d), y1 + 5, color="muted", width=1)
    svg.text(X_(d), y1 + 19, "%d°" % d, size=11, color="muted", anchor="middle")
svg.text((x0 + x1) / 2, y1 + 38, "输入方向 θ", size=12, color="muted", anchor="middle")
svg.text(20, (y0 + y1) / 2, "‖*Ax*‖", size=12, color="muted", anchor="middle", rotate=-90)
svg.line(x0, y0, x0, y1, color="rule", width=1)
svg.line(x0, y1, x1, y1, color="rule", width=1)
# reference lines
svg.line(x0, Y_(s1), x1, Y_(s1), color="blue", width=1.3, dash="6 4")
svg.line(x0, Y_(s2), x1, Y_(s2), color="violet", width=1.3, dash="6 4")
svg.line(x0, Y_(rms), x1, Y_(rms), color="amber", width=1.3, dash="3 3")
svg.path(polyline_d([(X_(a), Y_(b)) for a, b in zip(t, L)]), stroke="green", width=2.4)
svg.circle(X_(tmax), Y_(s1), 4.5, fill="blue")
svg.circle(X_(tmin), Y_(s2), 4.5, fill="violet")
svg.circle(X_(0), Y_(5), 4, fill="ink")
svg.circle(X_(90), Y_(5), 4, fill="ink")
svg.text(X_(tmax) + 8, Y_(s1) - 8, "最大 σ_{1} = %.3f（θ = 45°，*x* = *v*_{1}）" % s1, size=12, color="blue")
svg.text(X_(tmin) - 6, Y_(s2) + 18, "最小 σ_{2} = %.3f（θ = 135°）" % s2, size=12, color="violet",
         anchor="middle")
svg.text(X_(0) + 8, Y_(5) + 16, "*e*_{1}：‖(3, 4)‖ = 5", size=12)
svg.text(X_(90) + 8, Y_(5) - 8, "*e*_{2}：‖(0, 5)‖ = 5", size=12)
svg.text(X_(170), Y_(4.52), "均方根 = ‖*A*‖_{F}/√2 = %.2f" % rms, size=12, color="amber", anchor="end")

# ---- right panel: right triangle
svg.line(500, 40, 500, H - 16, color="rule", width=1)
svg.text(520, 26, "② 两把尺子", size=14, weight="600")
sc = 26
ox, oy = 536, 250
A_ = (ox, oy)
B_ = (ox + sc * s1, oy)
C_ = (ox + sc * s1, oy - sc * s2)
svg.polygon([A_, B_, C_], fill="green", fill_opacity=0.06)
svg.line(*A_, *B_, color="blue", width=3)
svg.line(*B_, *C_, color="violet", width=3)
svg.line(*A_, *C_, color="ink", width=2)
svg.rect(B_[0] - 10, B_[1] - 10, 10, 10, stroke="muted", width=1)
svg.text((A_[0] + B_[0]) / 2, oy + 20, "σ_{1} = %.2f = ‖*A*‖_{2}" % s1, size=12, color="blue", anchor="middle")
# sigma_2 label starts just left of the top vertex and the hypotenuse label is shifted down-left,
# so the end of "7.07" no longer runs into "σ2 = 2.24"
svg.text(C_[0] - 16, C_[1] - 10, "σ_{2} = %.2f" % s2, size=12, color="violet")
mx, my = (A_[0] + C_[0]) / 2, (A_[1] + C_[1]) / 2
svg.text(mx - 14, my - 8, "‖*A*‖_{F} = √50 = %.2f" % fro, size=12, anchor="middle",
         rotate=-math.degrees(math.atan2(s2, s1)))
svg.text(520, 64, "谱范数：最长的那条拉伸", size=12, color="blue")
svg.text(520, 82, "Frobenius：所有拉伸的平方和开方", size=12)
svg.text(520, 100, "σ_{1} ≤ ‖*A*‖_{F} ≤ √2·σ_{1}", size=12, color="muted")
svg.text(520, 118, "6.71 ≤ 7.07 ≤ 9.49", size=12, color="muted")
# Unicode superscript ² instead of ^{2}: a baseline-shifted "2" on the second line rose into the
# subscript F of the first line (the two lines are only 18 units apart).
svg.text(520, 298, "‖*A*‖_{F}² = 9 + 0 + 16 + 25", size=12, color="muted")
svg.text(520, 318, "= σ_{1}² + σ_{2}² = 45 + 5", size=12, color="muted")

svg.save("m02-norms.svg")
print("max %.4f at %.2f, min %.4f at %.2f, rms %.4f, fro %.4f" % (s1, tmax, s2, tmin, rms, fro))
