"""Figure 1.6: scaled dot-product attention for one head, redrawn top-down from Vaswani et al.
(2017) Fig. 2 (left) with the course colours and GPT-2 small shapes (d_k = d_h = 64).

Right panel: an illustrative 8-token example (seeded random Q, K) showing the score table S,
the masked table and the row-wise softmax A. The numbers are synthetic; only the structure matters.
Writes deepseek/figures/l01-attention-graph.svg
"""

import numpy as np

from l01_svg import SVG

rng = np.random.default_rng(7)
T, dk = 8, 4
Q = rng.standard_normal((T, dk))
K = rng.standard_normal((T, dk))
S = Q @ K.T / np.sqrt(dk)
mask = np.triu(np.ones((T, T), bool), 1)
Sm = np.where(mask, -np.inf, S)
A = np.exp(Sm - Sm.max(1, keepdims=True))
A /= A.sum(1, keepdims=True)
assert np.allclose(A.sum(1), 1) and np.all(A[mask] == 0)

W, H = 760, 470
s = SVG(W, H, "缩放点积注意力（单头）：计算图与每一步的形状")

# ---------------------------------------------------------------- left: the graph
BX, BW = 30, 300
s.box(BX, 16, 90, 44, [("Q", {"weight": "700", "color": "blue"}), ("[T, 64]", {"size": 12, "color": "muted"})],
      color="blue", fill_opacity=0.12, line_gap=17)
s.box(BX + 105, 16, 90, 44, [("K", {"weight": "700", "color": "violet"}), ("[T, 64]", {"size": 12, "color": "muted"})],
      color="violet", fill_opacity=0.12, line_gap=17)
s.box(BX + 210, 16, 90, 44, [("V", {"weight": "700", "color": "green"}), ("[T, 64]", {"size": 12, "color": "muted"})],
      color="green", fill_opacity=0.12, line_gap=17)
steps = [
    (92, "MatMul：S = QK^{⊤}", "violet", "[T, T]"),
    (152, "Scale：除以 √d_{k} = √64 = 8", "ink", "[T, T]"),
    (212, "Mask：j > t 的格子加上 −∞", "red", "[T, T]"),
    (272, "SoftMax：逐行归一化 → A", "ink", "[T, T]"),
    (352, "MatMul：O = A V", "green", "[T, 64]"),
]
s.arrow(BX + 45, 60, BX + 45, 92, color="blue")
s.arrow(BX + 150, 60, BX + 150, 92, color="violet")
for k, (y, lab, col, shp) in enumerate(steps):
    w = 195 if k < 4 else BW
    s.box(BX, y, w, 34, [(lab, {"size": 12.5})], color=col, fill_opacity=0.08)
    s.mono(BX + w + 8, y + 22, shp, size=12, color="muted")
    if 0 < k < 4:
        s.arrow(BX + 97, steps[k - 1][0] + 34, BX + 97, y)
s.arrow(BX + 97, 306, BX + 97, 352)
s.text(BX + 104, 334, "A  [T, T]", size=12, color="muted")
s.path("M%d 60 V330 H%d V352" % (BX + 255, BX + 255), stroke="green", width=1.6, arrow=True)
s.arrow(BX + 150, 386, BX + 150, 414, color="green")
s.box(BX + 90, 414, 120, 34, [("O  [T, 64]", {"size": 13, "weight": "600", "color": "green"})], color="green",
      fill_opacity=0.14)
s.text(BX + 222, 436, "第 t 行 = 位置 t 的输出", size=12, color="muted")

# ---------------------------------------------------------------- right: the three tables
C = 13
tops = 150
panels = [(384, "S = QK^{⊤}/√d_{k}", "打分表"), (510, "S + M", "加掩码"), (636, "A = softmax(S + M)", "按行归一化")]
s.text(384, 104, "示意：T = 8 个位置（随机数据）", size=12.5, weight="700")
s.text(384, 124, "第 t 行 = 位置 t 看向各位置 j 的分数或权重", size=12, color="muted")
smin, smax = S.min(), S.max()
for pi, (x0, lab, sub) in enumerate(panels):
    s.text(x0 + T * C / 2, tops - 8, sub, size=12, anchor="middle", color="muted")
    for i in range(T):
        for j in range(T):
            x, y = x0 + j * C, tops + i * C
            if pi == 0:
                v = (S[i, j] - smin) / (smax - smin)
                s.rect(x, y, C - 1, C - 1, fill="violet", fill_opacity=0.08 + 0.72 * v)
            elif mask[i, j]:
                s.rect(x, y, C - 1, C - 1, fill="red", fill_opacity=0.55)
            elif pi == 1:
                v = (S[i, j] - smin) / (smax - smin)
                s.rect(x, y, C - 1, C - 1, fill="violet", fill_opacity=0.08 + 0.72 * v)
            else:
                s.rect(x, y, C - 1, C - 1, fill="blue", fill_opacity=0.06 + 0.9 * A[i, j])
    s.rect(x0 - 1, tops - 1, T * C + 1, T * C + 1, stroke="rule", width=1)
    s.text(x0 + T * C / 2, tops + T * C + 18, lab, size=12, anchor="middle")
    if pi < 2:
        s.arrow(x0 + T * C + 4, tops + T * C / 2, x0 + 126 - 6, tops + T * C / 2, width=1.3)
s.text(384, tops + T * C + 50, "紫色深浅 = 分数高低；红色 = −∞，exp(−∞) = 0；", size=12, color="muted")
s.text(384, tops + T * C + 70, "蓝色深浅 = 权重大小，每一行的权重加起来等于 1。", size=12, color="muted")
# one line, kept above y = 362 so it clears the "[T, 64]" label of the last MatMul box
s.text(384, tops + T * C + 96, "第 1 行只剩一个格子：第一个 token 只能“看见”自己。", size=12, color="muted")

s.save("l01-attention-graph.svg")
