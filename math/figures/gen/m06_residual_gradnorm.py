"""Figure M6.10: gradient norm at every layer of a 100-layer random network, with and without residual.

Width d = 256, depth L = 100, weights W_l ~ N(0, s^2 / d) (s = weight scale), input x ~ N(0, I), readout
loss = w_out . h_L with a unit vector w_out, so the gradient at the top has norm 1.
  plain:    h_l = tanh(W_l h_{l-1})               J_l = diag(tanh'(W_l h_{l-1})) W_l
  residual: h_l = h_{l-1} + alpha * W_l tanh(h_{l-1})   J_l = I + alpha * W_l diag(tanh'(h_{l-1}))
The backward pass is done exactly (vector-Jacobian products). Lines are seed 0; the light bands
span the minimum and maximum over seeds 0-4. Every number is computed here.
Writes math/figures/m06-residual-gradnorm.svg
"""

import math

import numpy as np

from m06_svg import SVG, sci

d, L = 256, 100


def run(kind, sw, alpha, seed):
    rng = np.random.default_rng(seed)
    Ws = [rng.standard_normal((d, d)) * sw / math.sqrt(d) for _ in range(L)]
    h = rng.standard_normal(d)
    saved = []
    for W in Ws:
        if kind == "plain":
            u = W @ h
            saved.append(u)
            h = np.tanh(u)
        else:
            saved.append(h)
            h = h + alpha * (W @ np.tanh(h))
    w_out = rng.standard_normal(d)
    w_out /= np.linalg.norm(w_out)
    g = w_out
    norms = [1.0]
    for l in range(L - 1, -1, -1):
        W = Ws[l]
        if kind == "plain":
            g = W.T @ (g * (1 - np.tanh(saved[l]) ** 2))
        else:
            g = g + alpha * (1 - np.tanh(saved[l]) ** 2) * (W.T @ g)
        norms.append(float(np.linalg.norm(g)))
    return np.array(norms[::-1])          # index l = layer, norms[L] = 1 at the top


configs = [
    ("plain", 0.8, 1.0, "red", "无残差，s = 0.8", "消失"),
    ("plain", 1.0, 1.0, "orange", "无残差，s = 1", ""),
    ("plain", 2.0, 1.0, "violet", "无残差，s = 2", "爆炸"),
    ("resid", 1.0, 1.0, "blue", "残差，分支 α = 1", ""),
    ("resid", 1.0, 0.1, "green", "残差，分支 α = 0.1", ""),
]
results = []
for kind, sw, a, col, lab, tag in configs:
    runs = [run(kind, sw, a, s) for s in range(5)]
    arr = np.array(runs)
    results.append((kind, sw, a, col, lab, tag, arr))
    print(lab, "seed0 layer0 = %.3e" % arr[0, 0], "range over seeds %.3e .. %.3e" % (arr[:, 0].min(), arr[:, 0].max()))

W, H = 720, 440
X0, X1, Y0, Y1 = 78, 540, 30, 380
LY0, LY1 = -12, 8


def px(l):
    return X0 + l / L * (X1 - X0)


def py(v):
    v = max(v, 10.0 ** LY0)
    return Y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "100 层随机网络里每一层的梯度范数：有无残差连接")
for e in range(LY0, LY1 + 1, 2):
    y = py(10.0 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
for l in range(0, L + 1, 20):
    svg.line(px(l), Y0, px(l), Y1, color="grid", width=1)
    svg.text(px(l), Y1 + 18, str(l), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.line(X0, py(1), X1, py(1), color="ink", width=1, dash="4 3")
svg.text((X0 + X1) / 2, Y1 + 40, "层号 l（0 = 最底层，100 = 顶层；梯度从右往左传）", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "‖∂𝓛/∂h_{l}‖（顶层为 1，对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

for kind, sw, a, col, lab, tag, arr in results:
    lo = np.log10(np.maximum(arr.min(0), 10.0 ** LY0))
    hi = np.log10(np.minimum(arr.max(0), 10.0 ** LY1))
    band = [(px(l), py(10 ** hi[l])) for l in range(L + 1)] + [(px(l), py(10 ** lo[l])) for l in range(L, -1, -1)]
    svg.polygon(band, fill=col, fill_opacity=0.12)
    pts = ["%.1f %.1f" % (px(l), py(arr[0, l])) for l in range(L + 1)]
    svg.path("M" + " L".join(pts), stroke=col, width=2.2)

# direct labels at the left end (layer 0), where the curves are furthest apart
RX = X1 + 16
order = sorted(results, key=lambda r: -r[6][0, 0])
ys_used = []
for kind, sw, a, col, lab, tag, arr in order:
    v = arr[0, 0]
    y = py(v)
    svg.circle(px(0), y, 3.5, fill=col)
yy = Y0 + 10
svg.text(RX, yy, "第 0 层的梯度范数（seed 0）", size=12, weight="600"); yy += 24
for kind, sw, a, col, lab, tag, arr in order:
    svg.line(RX, yy - 4, RX + 20, yy - 4, color=col, width=2.4)
    svg.text(RX + 26, yy, lab, size=12, color=col, weight="600"); yy += 17
    svg.text(RX + 26, yy, sci(arr[0, 0], 1), size=12, color="muted"); yy += 26
svg.text(RX, yy + 4, "宽 256、深 100、tanh，", size=11, color="muted"); yy += 16
svg.text(RX, yy + 4, "W 的元素 ~ N(0, s²/256)", size=11, color="muted"); yy += 16
svg.text(RX, yy + 4, "浅色带：5 个随机种子的范围", size=11, color="muted")

svg.save("m06-residual-gradnorm.svg")
