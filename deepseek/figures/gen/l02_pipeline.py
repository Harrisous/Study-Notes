"""Figure 2.1: the whole pre-training pipeline of a GPT, with GPT-3 175B's numbers.

Left column: the data pipeline (done once). Right: the loop that runs once per optimizer step.
Numbers: GPT-3 (Brown et al. 2020) section 2.2, appendix A and B; vocabulary 50,257 (GPT-2).
Schematic, but every number on it is from the research notes (lesson-02-doc.md).
Writes deepseek/figures/l02-pipeline.svg
"""

from l02_svg import SVG, box

W, H = 720, 434
svg = SVG(W, H, "GPT 预训练的完整流程：数据管线（一次）与训练循环（每步一次）")

# ---------------- left column: data pipeline
svg.text(16, 22, "数据管线（只做一次）", size=13, color="amber", weight="600")
col_x, col_w, bh, gap = 16, 168, 64, 18
data = [
    ("原始语料", ["Common Crawl 45 TB（压缩）", "+ 书、维基等精选语料"]),
    ("质量过滤 + 去重", ["分类器打分、MinHash", "CC 过滤后剩 570 GB"]),
    ("分词", ["字节级 BPE", "词表 V = 50,257"]),
    ("拼接打包", ["文档首尾相接", "用 <|endoftext|> 分隔"]),
    ("组成 batch", ["切成 T = 2048 的序列", "每步约 3.2M 个 token"]),
]
ys = []
for i, (t, ls) in enumerate(data):
    y = 36 + i * (bh + gap)
    ys.append(y)
    box(svg, col_x, y, col_w, bh, t, ls, color="amber", fill_opacity=0.10, line_size=12)
    if i:
        svg.arrow(col_x + col_w / 2, y - gap + 1, col_x + col_w / 2, y - 1, color="amber", width=1.5)
svg.text(col_x + col_w - 5, ys[0] + 13, "§4", size=11, anchor="end", color="muted")

# ---------------- right: the training loop
svg.text(214, 22, "训练循环（每个优化步重复一次）", size=13, color="blue", weight="600")
bw, bh2 = 146, 70
X = [214, 381, 548]            # three columns
Yt, Ym, Yb = 44, 196, 348       # top, middle, bottom rows

nodes = {
    # B_seq = sequences per batch; B = B_seq * T tokens (the lesson's B counts tokens, eq. 2.6)
    "fwd": (X[0], Yt, "前向传播", ["logits 形状 [B_{seq}, T, V]"], "blue"),
    "loss": (X[1], Yt, "逐位置损失", ["ℓ_{t} = −log p_{θ}(x_{t+1} | x_{≤t})"], "red"),
    "mean": (X[2], Yt, "平均损失 L", ["对 B_{seq}×T 个位置取平均", "单位 nats/token"], "red"),
    "bwd": (X[2], Ym, "反向传播", ["得到梯度 ∇_{θ}L", "FLOPs 约为前向的 2 倍"], "muted"),
    "clip": (X[2], Yb, "梯度裁剪", ["全局范数超过 1.0", "就等比例缩小"], "muted"),
    "adam": (X[1], Yb, "AdamW 更新", ["β_{1}=0.9，β_{2}=0.95", "权重衰减 0.1"], "green"),
    # step index s, as in the lesson's section 1 and the caption (section 5 switches to Adam's t)
    "lr": (X[0], Yb, "学习率调度 η_{s}", ["warmup 375M token", "余弦降到峰值的 10%"], "green"),
    "next": (X[0], Ym, "下一步 s → s+1", ["取下一个 batch"], "blue"),
}
for key, (x, y, t, ls, col) in nodes.items():
    box(svg, x, y, bw, bh2, t, ls, color=col, fill_opacity=0.08 if col != "muted" else 0.06,
        line_size=12, title_color="ink" if col == "muted" else col)


def mid(key, side):
    x, y = nodes[key][0], nodes[key][1]
    return {"l": (x, y + bh2 / 2), "r": (x + bw, y + bh2 / 2), "t": (x + bw / 2, y),
            "b": (x + bw / 2, y + bh2)}[side]


def link(a, sa, b, sb, color="ink"):
    x1, y1 = mid(a, sa)
    x2, y2 = mid(b, sb)
    svg.arrow(x1, y1, x2 - (2 if x2 > x1 else -2 if x2 < x1 else 0), y2 - (2 if y2 > y1 else -2 if y2 < y1 else 0),
              color=color, width=1.6)


link("fwd", "r", "loss", "l", "blue")
link("loss", "r", "mean", "l", "red")
link("mean", "b", "bwd", "t", "ink")
link("bwd", "b", "clip", "t", "ink")
link("clip", "l", "adam", "r", "ink")
link("lr", "r", "adam", "l", "green")      # the schedule feeds the step size eta_t into AdamW
link("next", "t", "fwd", "b", "blue")
# AdamW -> next step: up from the AdamW box, then left into the "next" box (below the centre text)
ax_, ay_ = mid("adam", "t")
nx_, ny_ = mid("next", "r")
svg.path("M%.1f %.1f V%.1f H%.1f" % (ax_, ay_ - 1, ny_, nx_ + 2), stroke="green", width=1.6, arrow=True)

# batch feeds the loop
bx, by = col_x + col_w, ys[4] + bh / 2
nx, ny = mid("next", "l")
svg.path("M%.1f %.1f H%.1f V%.1f H%.1f" % (bx + 1, by, 197, ny, nx - 2), stroke="amber", width=1.6, arrow=True)

# section tags
for key, tag in (("fwd", "§3"), ("mean", "§2"), ("bwd", "§3·§6"), ("adam", "§5"), ("clip", "§5"), ("lr", "§5")):
    x, y = nodes[key][0], nodes[key][1]
    svg.text(x + bw - 5, y + 13, tag, size=11, anchor="end", color="muted")

# centre of the loop
cx = X[1] + bw / 2
svg.text(cx, Yt + bh2 + 22, "GPT-3 175B：", size=12, anchor="middle", color="ink", weight="600")
svg.text(cx, Yt + bh2 + 40, "300B token ÷ 3.2M token/步", size=12, anchor="middle", color="muted")
svg.text(cx, Yt + bh2 + 58, "≈ 93,750 步（忽略 batch 渐增）", size=12, anchor="middle", color="muted")
svg.text(cx, Yt + bh2 + 84, "总算力 ≈ 6ND", size=12, anchor="middle", color="ink")
svg.text(cx, Yt + bh2 + 102, "= 3.14×10^{23} FLOPs", size=12, anchor="middle", color="ink")

svg.save("l02-pipeline.svg")
