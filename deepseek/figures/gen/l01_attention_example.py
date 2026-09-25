"""Figure 1.7: the hand-computed 3-token example of sec. 3 (d_k = 2), drawn in the plane.

(a) keys k_1..k_3 and the query q_3 of token 3; the scores are the dot products q_3 . k_j.
(b) values v_1..v_3 span a triangle; each output o_t = sum_j alpha_tj v_j is a point inside the
part of the triangle that token t is allowed to see (causal mask).
All numbers are computed here and match the table in the lesson text.
Writes deepseek/figures/l01-attention-example.svg
"""

import numpy as np

from l01_svg import SVG

Q = np.array([[1, 1], [0, 2], [2, 1]], float)
K = np.array([[0, 1], [2, 0], [1, 1]], float)
V = np.array([[1, 0], [0, 1], [1, 1]], float)
S = Q @ K.T / np.sqrt(2)
S[np.triu_indices(3, 1)] = -np.inf
A = np.exp(S - S.max(1, keepdims=True))
A /= A.sum(1, keepdims=True)
O = A @ V
assert np.allclose(O[2], [0.380, 0.926], atol=1e-3) and np.allclose(A[1, :2], [0.804, 0.196], atol=1e-3)
words = ["The", "cat", "sat"]

W, H = 760, 350
s = SVG(W, H, "手算例子的几何：打分是点积，输出是值向量的加权平均")

# ---------------------------------------------------------------- (a) keys and query
ox, oy, u = 46, 300, 92


def P(x, y):
    return ox + x * u, oy - y * u


s.text(10, 22, "(a) token 3（sat）的查询与三个键", size=13, weight="700")
for g in range(0, 3):
    x, _ = P(g, 0)
    s.line(x, P(0, -0.2)[1], x, P(0, 2.3)[1], color="grid", width=1)
    if g:  # the origin gets one shared "0" below-left, so no label sits on an axis line
        s.text(x, oy + 18, str(g), size=11, anchor="middle", color="muted")
    _, y = P(0, g)
    s.line(P(-0.2, 0)[0], y, P(2.4, 0)[0], y, color="grid", width=1)
    if g:
        s.text(ox - 10, y + 4, str(g), size=11, anchor="end", color="muted")
s.text(ox - 8, oy + 16, "0", size=11, anchor="end", color="muted")
s.line(*P(-0.2, 0), *P(2.4, 0), color="muted", width=1.2)
s.line(*P(0, -0.2), *P(0, 2.3), color="muted", width=1.2)
tip = {0: (8, 6, "start"), 1: (4, -10, "middle"), 2: (-4, -10, "middle")}
for j in range(3):
    kx, ky = P(*K[j])
    s.arrow(ox, oy, kx, ky, color="violet", width=1.8)
    dx, dy, an = tip[j]
    s.text(kx + dx, ky + dy, "*k*_{%d}" % (j + 1), size=13, anchor=an, color="violet", weight="600")
qx, qy = P(*Q[2])
s.arrow(ox, oy, qx, qy, color="blue", width=2.4)
s.text(qx + 6, qy - 8, "*q*_{3}", size=13, color="blue", weight="600")
LX = 282  # the text column must end left of the panel divider at x = 456
s.text(LX, 96, "分数 = 查询与键的点积：", size=12, color="ink")
for j in range(3):
    raw = Q[2] @ K[j]
    s.text(LX, 122 + j * 24, "*q*_{3}·*k*_{%d} = 2×%g + 1×%g = %d" % (j + 1, K[j, 0], K[j, 1], raw), size=12,
           color="violet" if j != 1 else "violet", weight="600" if j == 1 else None)
    s.text(LX + 128, 122 + j * 24, "（%s）" % words[j], size=12, color="muted")
s.text(LX, 208, "k_{3} 的方向更接近 q_{3}，", size=12, color="muted")
s.text(LX, 226, "但 k_{2} 更长（2 对 1.41），", size=12, color="muted")
s.text(LX, 244, "所以 k_{2}（cat）的点积最大。", size=12, color="muted")

# ---------------------------------------------------------------- (b) values and outputs
bx0, by0, ub = 504, 296, 186
s.line(456, 12, 456, 340, color="rule", width=1)
s.text(468, 22, "(b) 值向量围成的三角形与三个输出", size=13, weight="700")


def PB(x, y):
    return bx0 + x * ub, by0 - y * ub


for g in (0, 0.5, 1):
    x, _ = PB(g, 0)
    s.line(x, PB(0, -0.05)[1], x, PB(0, 1.12)[1], color="grid", width=1)
    s.text(x, by0 + 18, "%g" % g, size=11, anchor="middle", color="muted")
    _, y = PB(0, g)
    s.line(PB(-0.05, 0)[0], y, PB(1.12, 0)[0], y, color="grid", width=1)
    if g:
        s.text(bx0 - 8, y + 4, "%g" % g, size=11, anchor="end", color="muted")
s.polygon([PB(*V[0]), PB(*V[1]), PB(*V[2])], fill="green", fill_opacity=0.08)
s.line(*PB(*V[1]), *PB(*V[2]), color="green", width=1)
s.line(*PB(*V[2]), *PB(*V[0]), color="green", width=1)
s.line(*PB(*V[0]), *PB(*V[1]), color="green", width=1.4, dash="5 4")  # the only edge o_2 can live on
vlab = {0: (10, 4, "start"), 1: (0, -12, "middle"), 2: (10, 4, "start")}
for j in range(3):
    x, y = PB(*V[j])
    s.circle(x, y, 5, fill="green")
    dx, dy, an = vlab[j]
    s.text(x + dx, y + dy, "*v*_{%d}（%s）" % (j + 1, words[j]), size=12, anchor=an, color="green")
olab = {0: (14, 22, "start"), 1: (-12, 4, "end"), 2: (8, 22, "start")}  # o_1 sits under the v_1 label
for t in range(3):
    x, y = PB(*O[t])
    s.circle(x, y, 5.5, fill="surface", stroke="blue", width=2.2)
    dx, dy, an = olab[t]
    txt = "*o*_{1} = *v*_{1}" if t == 0 else "*o*_{%d} = (%.3f, %.3f)" % (t + 1, O[t, 0], O[t, 1])
    s.text(x + dx, y + dy, txt, size=12, anchor=an, color="blue", weight="600")
s.text(468, 340, "o_{2} 只能落在虚线 v_{1}–v_{2} 上；o_{3} 可在整个三角形内", size=12, color="muted")

s.save("l01-attention-example.svg")
