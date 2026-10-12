"""Figure M6.6: softmax Jacobian and the cross-entropy gradient p - y for z = (2, 1, 0), label 0.

Left: p = softmax(z). Middle: the 3x3 Jacobian diag(p) - p p^T as a heat map (blue positive,
red negative, opacity proportional to |value|), with the row sums. Right: dL/dz = p - y.
Every number is computed here (research notes N41, N42).
Writes math/figures/m06-softmax-jacobian.svg
"""

import math

import numpy as np

from m06_svg import SVG

z = np.array([2.0, 1.0, 0.0])
p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
J = np.diag(p) - np.outer(p, p)
y = np.array([1.0, 0.0, 0.0])
g = p - y
assert np.allclose(J.sum(1), 0)
print("p", p, "J", J, "g", g)

W, H = 720, 360
svg = SVG(W, H, "softmax 的雅可比 diag(p) − ppᵀ 与交叉熵梯度 p − y（z = (2, 1, 0)，真类别为 0）")


def num(v, d=3):
    s = "%.*f" % (d, v)
    return s.replace("-", "−")


# ---- left: z and p
LX = 24
svg.text(LX, 34, "① logits 与概率", size=13, weight="600")
top = 64
rowh = 62
bmax = 104
for i in range(3):
    yy = top + i * rowh
    svg.text(LX, yy + 16, "z_{%d} = %d" % (i, z[i]), size=12, color="muted")
    svg.rect(LX + 58, yy + 4, bmax * p[i], 16, fill="blue", fill_opacity=0.55)
    svg.text(LX + 58 + bmax * p[i] + 5, yy + 17, "p_{%d} = %s" % (i, num(p[i])), size=12, color="blue")
svg.text(LX, top + 3 * rowh + 6, "p = softmax(z)，和为 1", size=12, color="muted")

# ---- middle: heat map
MX = 250
cell = 66
svg.text(MX, 34, "② 雅可比 ∂p_{i}/∂z_{j} = p_{i}(δ_{ij} − p_{j})", size=13, weight="600")
for j in range(3):
    svg.text(MX + j * cell + cell / 2, top - 6, "z_{%d}" % j, size=12, anchor="middle", color="muted")
vmax = np.abs(J).max()
for i in range(3):
    svg.text(MX - 8, top + i * cell + cell / 2 + 4, "p_{%d}" % i, size=12, anchor="end", color="muted")
    for j in range(3):
        v = J[i, j]
        col = "blue" if v > 0 else "red"
        svg.rect(MX + j * cell + 1, top + i * cell + 1, cell - 2, cell - 2, fill=col,
                 fill_opacity=0.08 + 0.5 * abs(v) / vmax, rx=4)
        svg.text(MX + j * cell + cell / 2, top + i * cell + cell / 2 + 5, num(v), size=12, anchor="middle",
                 color="ink", weight="600" if i == j else None)
    svg.text(MX + 3 * cell + 10, top + i * cell + cell / 2 + 5, "行和 %s" % num(abs(J[i].sum()), 0), size=12,
             color="muted")
svg.text(MX, top + 3 * cell + 22, "对角（加粗）> 0：自己的 logit ↑，自己的概率 ↑", size=12, color="blue")
svg.text(MX, top + 3 * cell + 40, "非对角 < 0：别人的 logit ↑，自己的概率 ↓", size=12, color="red")
svg.text(MX, top + 3 * cell + 58, "对称；每行、每列之和都是 0", size=12, color="muted")

# ---- right: p - y
RX = 548
svg.text(RX, 34, "③ ∂ℓ/∂z = p − y", size=13, weight="600")
zero = top + 96
scale = 150
cw = 36
for i in range(3):
    cx = RX + 20 + i * 50
    v = g[i]
    col = "red" if v < 0 else "blue"
    y_top = zero - scale * v if v > 0 else zero
    svg.rect(cx - cw / 2, y_top, cw, abs(scale * v), fill=col, fill_opacity=0.55)
    if v < 0:
        svg.text(cx, zero + abs(scale * v) + 15, num(v), size=12, anchor="middle", color=col)
    else:
        svg.text(cx, y_top - 5, num(v), size=12, anchor="middle", color=col)
    svg.text(cx, top + 2, "类 %d" % i, size=12, anchor="middle", color="muted")
svg.text(RX + 20, top + 17, "（真）", size=11, anchor="middle", color="muted")
svg.line(RX, zero, RX + 140, zero, color="ink", width=1)
svg.text(RX, top + 3 * rowh + 6, "和为 0；真类别为负，", size=12, color="muted")
svg.text(RX, top + 3 * rowh + 24, "其余等于各自的 p", size=12, color="muted")
svg.text(RX, top + 3 * rowh + 42, "ℓ = −ln p_{0} = %s nats" % num(-math.log(p[0]), 4), size=12, color="ink")

svg.save("m06-softmax-jacobian.svg")
