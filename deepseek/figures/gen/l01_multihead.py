"""Figure 1.11: multi-head attention in GPT-2 small (d = 768, n_h = 12, d_h = 64).

(a) splitting the 768-d query into 12 slices costs nothing (a reshape);
(b) every head runs its own softmax, so one position gets 12 different weightings of its prefix
    (the four weight patterns drawn are illustrative, generated here from fixed logits);
(c) concatenating and multiplying by W^O equals each head writing through its own 768 x 64 column
    block W^O_i, summed (block multiplication; Elhage et al. 2021).
Writes deepseek/figures/l01-multihead.svg
"""

import numpy as np

from l01_svg import SVG

d, nh = 768, 12
dh = d // nh
assert dh == 64

W, H = 760, 490
s = SVG(W, H, "多头注意力：拆头、各自 softmax、经 W^O 合头")

# ---------------------------------------------------------------- (a) split
s.text(10, 22, "(a) 拆头：位置 t 的查询 *q*_{t} ∈ ℝ^{768} 按顺序切成 12 段，每段 64 维（只是换一种方式读下标）", size=13,
       weight="700")
X0, X1 = 20, 740
sw = (X1 - X0) / nh
for i in range(nh):
    x = X0 + i * sw
    s.rect(x, 38, sw - 2, 26, fill="blue", fill_opacity=0.10 + (0.25 if i == 2 else 0.06 * (i % 2)),
           stroke="blue", width=0.8)
    s.text(x + sw / 2, 56, "*q*_{t,%d}" % (i + 1), size=12, anchor="middle", color="blue")
s.text(X0, 82, "第 i 段 = W^{Q}_{i} · h̄_{t}；W^{Q}_{i} 是 768×768 的 W^{Q} 的第 i 个 64 行块；键、值同理", size=12,
       color="muted")

# ---------------------------------------------------------------- (b) per-head softmax
s.line(10, 98, 750, 98, color="rule", width=1)
s.text(10, 120, "(b) 每个头各做一次 softmax：同一个位置 t = 8，得到 12 套不同的权重（示意）", size=13, weight="700")
T = 8
patterns = {
    "头 1": np.array([0, 0, 0, 0, 0, 0, 4.0, 1.0]),
    "头 2": np.array([4.0, 0, 0, 0, 0, 0, 0, 0.5]),
    "头 3": np.array([0.3, 0.2, 0.4, 0.1, 0.3, 0.2, 0.4, 0.3]),
    "头 12": np.array([0, 0, 3.0, 0, 0, 0, 0, 3.0]),
}
px0 = [30, 210, 390, 590]
BW, BH = 16, 90
base = 250
for (name, z), x0 in zip(patterns.items(), px0):
    w = np.exp(z - z.max())
    w /= w.sum()
    assert abs(w.sum() - 1) < 1e-12
    for j in range(T):
        h = w[j] * BH
        s.rect(x0 + j * (BW + 2), base - h, BW, h, fill="blue", fill_opacity=0.25 + 0.6 * w[j])
        s.text(x0 + j * (BW + 2) + BW / 2, base + 14, str(j + 1), size=11, anchor="middle", color="muted")
    s.line(x0 - 2, base, x0 + T * (BW + 2), base, color="muted", width=1)
    s.text(x0 + T * (BW + 2) / 2, base + 34, name, size=12.5, anchor="middle", weight="600")
    s.text(x0 + T * (BW + 2) / 2, base - BH - 12, "α_{8,j} 对 j = 1…8", size=11, anchor="middle", color="muted")
s.text(540, base - 30, "…", size=20, anchor="middle", color="muted")

# ---------------------------------------------------------------- (c) merge through W^O
s.line(10, 298, 750, 298, color="rule", width=1)
s.text(10, 320, "(c) 合头：拼接后乘 W^{O}，等于每个头经自己那一块 W^{O}_{i} 写回 768 维，再全部相加", size=13, weight="700")
Y = 336
SQ = 120
WX = 60
cb = SQ / nh
for i in range(nh):
    s.rect(WX + i * cb, Y, cb, SQ, fill="green" if i == 2 else "ink", fill_opacity=0.35 if i == 2 else 0.04,
           stroke="muted", width=0.6)
s.rect(WX, Y, SQ, SQ, stroke="ink", width=1.2)
s.text(WX + SQ / 2, Y + SQ + 18, "W^{O}  [768, 768]", size=12, anchor="middle")
s.text(WX - 8, Y + SQ / 2 + 4, "768", size=11, anchor="end", color="muted")
cx = WX + SQ + 16
s.text(cx - 6, Y + SQ / 2 + 5, "×", size=15, anchor="middle")
vx = cx + 8
for i in range(nh):
    s.rect(vx, Y + i * cb, 22, cb, fill="blue" if i == 2 else "ink", fill_opacity=0.35 if i == 2 else 0.05,
           stroke="muted", width=0.6)
s.text(vx + 11, Y + SQ + 18, "[*o*_{t,1}; …; *o*_{t,12}]", size=12, anchor="middle")
s.text(vx + 44, Y + SQ / 2 + 5, "=", size=15, anchor="middle")
ux = vx + 60
s.rect(ux, Y, 22, SQ, fill="green", fill_opacity=0.18, stroke="green", width=1)
s.text(ux + 11, Y + SQ + 18, "*u*_{t}", size=12, anchor="middle")
tx = ux + 64
eqs = [
    ("*u*_{t} = W^{O}[*o*_{t,1}; …; *o*_{t,12}] = Σ_{i} W^{O}_{i} *o*_{t,i}", "ink", "600"),
    ("W^{O}_{i}：W^{O} 的第 i 个列块，形状 [768, 64]（图中绿色一列）", "green", None),
    ("它把第 i 个头的 64 维结果“写”进 768 维的残差流；", "muted", None),
    ("12 个头写的增量相加，谁也不会被关在固定的 64 个坐标里。", "muted", None),
    ("参数：W^{O} 共 768 × 768 = 589,824 个，另有 768 个偏置。", "muted", None),
]
for k, (t, col, wt) in enumerate(eqs):
    s.text(tx, Y + 18 + k * 24, t, size=12.5 if k == 0 else 12, color=col, weight=wt)

s.save("l01-multihead.svg")
