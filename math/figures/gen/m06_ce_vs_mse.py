"""Figure M6.7: gradient norm w.r.t. the logits, cross-entropy vs squared error, as softmax saturates.

Logits z = (m, 0, 0) for a margin m >= 0, p = softmax(z). Two targets:
  wrong:   true class 1 (the model is confidently wrong when m is large)   -> solid lines
  right:   true class 0 (the model is confidently right when m is large)   -> dashed lines
Cross-entropy  l = -log p_y        : dl/dz = p - y                                  (M6.16)
Squared error  l = 1/2 ||p - y||^2 : dl/dz = J^T (p - y) = p * (g - p.g), g = p - y  (M6.17)
The curves are the Euclidean norms of these gradients, computed here in float64.
Writes math/figures/m06-ce-vs-mse.svg
"""

import math

import numpy as np

from m06_svg import SVG, sci


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def grads(m, label):
    z = np.array([m, 0.0, 0.0])
    p = softmax(z)
    y = np.zeros(3)
    y[label] = 1.0
    g_ce = p - y
    g = p - y                                   # dl/dp of 1/2 ||p - y||^2
    g_mse = p * (g - p @ g)                     # softmax VJP (M6.17)
    return float(np.linalg.norm(g_ce)), float(np.linalg.norm(g_mse)), p


ms = np.linspace(0.0, 12.0, 241)
curves = {}
for label, tag in ((1, "wrong"), (0, "right")):
    ce, mse = zip(*[grads(m, label)[:2] for m in ms])
    curves[tag] = (np.array(ce), np.array(mse))

ce10, mse10, p10 = grads(10.0, 1)
print("m=10 wrong: p", p10, "CE", ce10, "MSE", mse10, "ratio", ce10 / mse10)
for m in (0, 2, 4, 6, 8, 10, 12):
    a = grads(m, 1)
    b = grads(m, 0)
    print("m=%2d  wrong CE %.4g MSE %.4g | right CE %.4g MSE %.4g" % (m, a[0], a[1], b[0], b[1]))

W, H = 720, 400
X0, X1, Y0, Y1 = 78, 500, 30, 340
MX = 12.0
LY0, LY1 = -6, 1


def px(m):
    return X0 + m / MX * (X1 - X0)


def py(v):
    v = min(max(v, 10.0 ** LY0), 10.0 ** LY1)
    return Y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "softmax 饱和时，交叉熵与平方误差对 logits 的梯度范数")
for e in range(LY0, LY1 + 1):
    y = py(10.0 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
for m in range(0, 13, 2):
    svg.line(px(m), Y0, px(m), Y1, color="grid", width=1)
    svg.text(px(m), Y1 + 18, str(m), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 42, "logits z = (m, 0, 0) 中的 m（类别 0 领先的幅度）", size=12, anchor="middle",
         color="muted")
svg.text(22, (Y0 + Y1) / 2, "‖∂ℓ/∂z‖（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

# the largest possible norm of p - y is sqrt(2)
svg.line(X0, py(math.sqrt(2)), X1, py(math.sqrt(2)), color="ink", width=1, dash="4 3")
svg.text(X0 + 6, py(math.sqrt(2)) - 6, "上限 √2", size=12, color="ink")


def curve(vals, col, dash=None):
    pts = ["%.1f %.1f" % (px(m), py(v)) for m, v in zip(ms, vals) if v >= 10.0 ** LY0]
    svg.path("M" + " L".join(pts), stroke=col, width=2.2, dash=dash)


ce_w, mse_w = curves["wrong"]
ce_r, mse_r = curves["right"]
curve(ce_r, "blue", dash="6 4")
curve(mse_r, "red", dash="6 4")
curve(ce_w, "blue")
curve(mse_w, "red")

# the worked example of the text: m = 10, true class 1
svg.line(px(10), Y0, px(10), Y1, color="muted", width=1, dash="2 3")
svg.circle(px(10), py(ce10), 4.5, fill="blue")
svg.circle(px(10), py(mse10), 4.5, fill="red")
svg.text(px(10) - 8, py(ce10) + 20, "1.414", size=12, anchor="end", color="blue", weight="600")
svg.text(px(10) + 8, py(mse10) + 4, sci(mse10, 1), size=12, color="red", weight="600")
svg.text(px(10) - 8, (py(ce10) + py(mse10)) / 2 + 4, "差约 %d 倍" % round(ce10 / mse10, -2), size=12,
         anchor="end", color="ink")

# right-hand legend
RX = X1 + 22
yy = Y0 + 10
svg.text(RX, yy, "实线：真类别是 1（猜错）", size=12, weight="600"); yy += 22
svg.line(RX, yy - 4, RX + 22, yy - 4, color="blue", width=2.4)
svg.text(RX + 28, yy, "交叉熵 p − y", size=12, color="blue", weight="600"); yy += 17
svg.text(RX + 28, yy, "越错越用力，趋于 √2", size=12, color="muted"); yy += 24
svg.line(RX, yy - 4, RX + 22, yy - 4, color="red", width=2.4)
svg.text(RX + 28, yy, "平方误差 Jᵀ(p − y)", size=12, color="red", weight="600"); yy += 17
svg.text(RX + 28, yy, "饱和后 J → 0，梯度消失", size=12, color="muted"); yy += 34
svg.text(RX, yy, "虚线：真类别是 0（猜对）", size=12, weight="600"); yy += 22
svg.line(RX, yy - 4, RX + 22, yy - 4, color="blue", width=2.4, dash="6 4")
svg.text(RX + 28, yy, "交叉熵", size=12, color="blue", weight="600"); yy += 20
svg.line(RX, yy - 4, RX + 22, yy - 4, color="red", width=2.4, dash="6 4")
svg.text(RX + 28, yy, "平方误差", size=12, color="red", weight="600"); yy += 17
svg.text(RX + 28, yy, "猜对了，两者都该变小", size=12, color="muted"); yy += 34
svg.text(RX, yy, "m = 10 时 p₀ = %.6f" % p10[0], size=12, color="muted"); yy += 17
svg.text(RX, yy, "（正文的例子）", size=12, color="muted")

svg.save("m06-ce-vs-mse.svg")
