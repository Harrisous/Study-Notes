"""Figure 2.9: where the 6 in C = 6ND comes from, and GPT-3 175B's per-token training FLOPs.

Left: one linear layer Y = X W with X in R^{T x d_in}, W in R^{d_in x d_out}. Forward is one matmul,
backward is two matmuls of the same size (dL/dX = dL/dY W^T and dL/dW = X^T dL/dY); each costs
2 T d_in d_out FLOPs = 2 FLOPs per parameter per token (Kaplan et al. 2020 section 2.1; GPT-3 table D.1).
Right (computed): GPT-3 175B, N = 174.6e9 (table D.1), n_layer = 96, d = 12288, n_ctx = 2048:
  forward weights 2N = 349.2 GFLOP, backward 4N = 698.4 GFLOP per token;
  attention scores and weighted sums, Kaplan's forward term 2 n_layer n_ctx d times 3 for training
  = 14.5 GFLOP per token (1.38% of 6N).
Writes deepseek/figures/l02-6nd.svg
"""

from l02_svg import SVG

N, L, T, d = 174.6e9, 96, 2048, 12288
fwd, bwd = 2 * N / 1e9, 4 * N / 1e9
att = 3 * 2 * L * T * d / 1e9
assert abs(att - 14.4955) < 1e-3

W, H = 720, 410
svg = SVG(W, H, "6ND 的来历：一个线性层的前向与反向，以及 GPT-3 每 token 的训练 FLOPs")

svg.text(16, 22, "一个线性层 Y = XW（W 有 d_{in}d_{out} 个参数）", size=13,
         color="ink", weight="600")


def mat(x, y, w, h, label, color, sub=None):
    svg.rect(x, y, w, h, fill=color, fill_opacity=0.14, stroke=color, width=1.3, rx=3)
    svg.text(x + w / 2, y + h / 2 + 5, label, size=13, anchor="middle", color=color, weight="600")
    if sub:
        svg.text(x + w / 2, y + h + 14, sub, size=11, anchor="middle", color="muted")


rows = [
    ("前向", "Y = X · W", [("X", "blue", 28, 48, "[T, d_{in}]"), ("W", "violet", 48, 48, "[d_{in}, d_{out}]"),
                          ("Y", "green", 48, 48, "[T, d_{out}]")], "blue"),
    ("反向 ①", "∂X = ∂Y · W^{⊤}", [("∂Y", "red", 48, 48, "[T, d_{out}]"), ("W^{⊤}", "violet", 48, 48, "[d_{out}, d_{in}]"),
                                  ("∂X", "red", 28, 48, "[T, d_{in}]")], "red"),
    ("反向 ②", "∂W = X^{⊤} · ∂Y", [("X^{⊤}", "blue", 48, 28, "[d_{in}, T]"), ("∂Y", "red", 48, 48, "[T, d_{out}]"),
                                  ("∂W", "red", 48, 48, "[d_{in}, d_{out}]")], "red"),
]
y0 = 46
for i, (tag, formula, mats, col) in enumerate(rows):
    y = y0 + i * 112
    svg.text(16, y + 20, tag, size=13, color=col, weight="600")
    svg.text(16, y + 38, formula, size=12, color="ink")
    x = 116
    for j, (lab, c, w, h, sub) in enumerate(mats):
        mat(x, y + (48 - h) / 2 + 6, w, h, lab, c, sub)
        x += w
        if j == 0:
            svg.text(x + 12, y + 35, "×", size=16, anchor="middle", color="ink")
            x += 24
        elif j == 1:
            svg.text(x + 12, y + 35, "=", size=16, anchor="middle", color="ink")
            x += 24
    svg.text(318, y + 24, "2 · T · d_{in} · d_{out}", size=13, color="ink", weight="600")
    svg.text(318, y + 48, "= 2 × 参数 × token", size=12, color="muted")
svg.line(16, y0 + 3 * 112 - 8, 450, y0 + 3 * 112 - 8, color="rule", width=1.2)
svg.text(16, y0 + 3 * 112 + 12, "合计：前向 2 + 反向 4 = 每参数每 token 6 次 → C ≈ 6ND", size=13, color="ink",
         weight="600")

# ---------------- right: stacked bar
BX, BW_ = 566, 56
Y0, Y1 = 60, 360
scale = (Y1 - Y0) / 1100.0
svg.text(450, 22, "GPT-3 175B：每训练 token 的 FLOPs", size=13, color="ink", weight="600")
yb = Y1
for val, lab, col in ((fwd, "前向 2N", "blue"), (bwd, "反向 4N", "red")):
    h = val * scale
    svg.rect(BX, yb - h, BW_, h, fill=col, fill_opacity=0.55 if col == "blue" else 0.45, stroke=col, width=1)
    svg.text(BX - 8, yb - h / 2 + 4, lab, size=12, anchor="end", color=col, weight="600")
    svg.text(BX + BW_ + 8, yb - h / 2 + 4, "%.1f GFLOP" % val, size=12, color="ink")
    yb -= h
h = att * scale
svg.rect(BX, yb - h, BW_, h, fill="violet", fill_opacity=0.85)
svg.line(BX, yb - h / 2, BX - 16, yb - 16, color="violet", width=1)
svg.text(BX - 18, yb - 34, "注意力打分 + 加权求和", size=12, anchor="end", color="violet", weight="600")
svg.text(BX - 18, yb - 18, "%.1f GFLOP（%.1f%%）" % (att, att / (6 * N / 1e9) * 100), size=12, anchor="end", color="violet")
svg.line(BX - 4, Y1, BX + BW_ + 4, Y1, color="muted", width=1.2)
svg.text(BX + BW_ / 2, Y1 + 18, "6N = %.1f GFLOP" % (6 * N / 1e9), size=12, anchor="middle", color="ink", weight="600")
svg.text(BX + BW_ / 2, Y1 + 36, "× 300B token = 3.14×10^{23}", size=12, anchor="middle", color="muted")

svg.save("l02-6nd.svg")
