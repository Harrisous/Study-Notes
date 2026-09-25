"""Figure M10.1: DeepSeek-V3 training FLOPs per token, split by component.

Parameter counts per component are the ones rebuilt from the released config in the
lesson-12 research notes (deepseek/reference-docs/lesson-12-doc.md §4.3):
  attention (MLA) per layer 187,105,280; dense FFN 396,361,728 (first 3 layers);
  one expert 44,040,192 (58 MoE layers, 8 routed + 1 shared active); gate 1,835,008;
  output head 926,679,040.  Matrix FLOPs = 6 x active parameters (forward 2 + backward 4).
  Attention scores + weighted sum (causal, mean context 2048, d_qk = 192, d_v = 128, 128 heads):
  3 * 61 * 2 * 128 * (192 + 128) * 4096 / 2.
Comparison bar: all 671.03B parameters active (256 routed experts per MoE layer), with the input
embedding (926,679,040) still a lookup, i.e. 6 x (671.03B - 0.93B) + the same attention term.
Writes math/figures/m10-v3-flops.svg
"""

from m10_svg import SVG

ATT, DENSE, EXPERT, GATE, HEAD = 187105280, 396361728, 44040192, 1835008, 926679040
parts = [
    ("路由专家（每 token 8 个）", 58 * 8 * EXPERT * 6, "blue"),
    ("MLA 的投影矩阵", 61 * ATT * 6, "orange"),
    ("注意力打分与加权求和", 3 * 61 * 2 * 128 * (192 + 128) * 4096 / 2, "green"),
    ("共享专家", 58 * EXPERT * 6, "violet"),
    ("前 3 层稠密 FFN", 3 * DENSE * 6, "muted"),
    ("输出头", HEAD * 6, "ink"),
    ("门控（路由打分）", 58 * GATE * 6, "violet"),
]
total = sum(v for _, v, _ in parts)
assert abs(total / 1e9 - 250.45) < 0.01, total
n_act = 61 * ATT + 3 * DENSE + 58 * (9 * EXPERT + GATE) + HEAD
EMB = HEAD                                          # input embedding, same shape as the output head
n_all = 61 * ATT + 3 * DENSE + 58 * (257 * EXPERT + GATE) + HEAD + EMB
assert abs(n_all / 1e9 - 671.03) < 0.01, n_all      # the "671B" total
# every one of the 671B parameters active, counted the same way as N_act: the embedding stays a
# table lookup (no FLOPs), so it is left out of the 6 x parameters term
dense_equiv = 6 * (n_all - EMB) + parts[2][1]

W, H = 720, 326
svg = SVG(W, H, "DeepSeek-V3 每个训练 token 的运算量拆分")
X0, X1 = 40, 680
BY, BH = 64, 46
scale = (X1 - X0) / total

svg.text(X0, 30, "DeepSeek-V3 每个训练 token：%.2f GFLOPs（前向 %.1f + 反向 %.1f）"
         % (total / 1e9, total / 3e9, 2 * total / 3e9), size=14, weight="600")
svg.text(X0, 50, "矩阵部分 = 6 × 每 token 激活的参数 %.2fB；注意力打分按因果掩码、平均上下文 2048 计"
         % (n_act / 1e9), size=12, color="muted")

def opacity(name, col):
    if name.startswith("门控"):
        return 0.4            # same hue as the shared expert (violet = routing), lighter tint
    return 0.85 if col in ("blue", "orange", "green", "violet") else 0.55


x = X0
label_rows = []
for i, (name, v, col) in enumerate(parts):
    w = v * scale
    op = opacity(name, col)
    svg.rect(x, BY, w, BH, fill=col, fill_opacity=op, stroke="surface", width=1)
    label_rows.append((name, v, col, x, w))
    x += w

# labels: big segments inside, small ones below with leader lines
below = []
for name, v, col, x0, w in label_rows:
    pct = 100 * v / total
    if w > 60:
        svg.text(x0 + w / 2, BY + 20, "%.1f" % (v / 1e9), size=13, anchor="middle", color="surface", weight="700")
        svg.text(x0 + w / 2, BY + 37, "%.0f%%" % pct, size=12, anchor="middle", color="surface")
    else:
        below.append((name, v, col, x0, w, pct))

# legend-style rows under the bar for every component (keeps text off the small segments)
LY = BY + BH + 30
colx = [X0, X0 + 330]
for i, (name, v, col, x0, w) in enumerate(label_rows):
    cx = colx[i % 2]
    cy = LY + (i // 2) * 24
    op = opacity(name, col)
    svg.rect(cx, cy - 11, 14, 14, fill=col, fill_opacity=op, rx=2, stroke=col, width=1)
    svg.text(cx + 22, cy, "%s  %.2f GF（%.1f%%）" % (name, v / 1e9, 100 * v / total), size=12)

# comparison: every parameter active
CY = LY + 4 * 24 + 16
svg.text(X0, CY, "同一刻度下的对比：如果 671B 个参数对每个 token 全部参与计算（嵌入层照旧只查表）", size=13,
         weight="600")
big_scale = (X1 - X0) / dense_equiv
bw1 = total * big_scale
svg.rect(X0, CY + 12, bw1, 20, fill="blue", fill_opacity=0.85)
svg.text(X0 + bw1 + 8, CY + 27, "V3 实际：%.0f GF" % (total / 1e9), size=12, color="blue", weight="600")
svg.rect(X0, CY + 40, X1 - X0, 20, fill="muted", fill_opacity=0.35)
svg.text(X1 - 6, CY + 55, "全部激活：约 %.0f GF（%.1f 倍）" % (dense_equiv / 1e9, dense_equiv / total),
         size=12, anchor="end", color="ink", weight="600")

svg.save("m10-v3-flops.svg")
print("total GF %.3f, active params %.4fB, dense-equivalent %.1f GF, ratio %.2f"
      % (total / 1e9, n_act / 1e9, dense_equiv / 1e9, dense_equiv / total))
