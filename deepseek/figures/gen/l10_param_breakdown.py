"""Figure 10.2: where DeepSeek-V2's 235.7B parameters live, and which 20.9B one token activates.

Counted from the HF config.json of deepseek-ai/DeepSeek-V2 (hidden 5120, 60 layers, first layer dense
with intermediate 12288, 59 MoE layers of 2 shared + 160 routed experts of intermediate 1536, 6 routed
active, MLA q_lora_rank 1536, kv_lora_rank 512, qk_nope 128, qk_rope 64, v_head 128, 128 heads,
vocab 102400, untied embeddings). "Activated" counts the output head but not the input embedding lookup
(the paper does not define its convention; see lesson-10-doc sec. 4.1).
Writes deepseek/figures/l10-param-breakdown.svg
"""

from l10_svg import SVG, fmt

d, L, V = 5120, 60, 102400
mla = (d * 1536 + 1536 + 1536 * 128 * 192 + d * 576 + 512 + 512 * 128 * 256 + 128 * 128 * d)
expert = 3 * d * 1536
dense = 3 * d * 12288
n_moe = L - 1
attn = L * mla
routed = n_moe * 160 * expert
shared = n_moe * 2 * expert
gates = n_moe * 160 * d
norms = L * 2 * d + d
emb = V * d
misc = dense + gates + norms

total_parts = [
    ("路由专家（160 个 × 59 层）", routed, "blue"),
    ("MLA 注意力（60 层）", attn, "orange"),
    ("共享专家（2 个 × 59 层）", shared, "violet"),
    ("嵌入 + 输出头", 2 * emb, "amber"),
    ("第 1 层稠密 FFN、门控、RMSNorm", misc, "muted"),
]
act_routed = n_moe * 6 * expert
act_parts = [
    ("MLA 注意力", attn, "orange"),
    ("被选中的路由专家（6 个 × 59 层）", act_routed, "blue"),
    ("共享专家", shared, "violet"),
    ("输出头", emb, "amber"),
    ("第 1 层稠密 FFN、门控、RMSNorm", misc, "muted"),
]
T = sum(p[1] for p in total_parts)
A = sum(p[1] for p in act_parts)
assert abs(T / 1e9 - 235.74) < 0.01 and abs(A / 1e9 - 20.85) < 0.01, (T, A)

W, H = 740, 404
svg = SVG(W, H, "DeepSeek-V2 的参数构成：总参数 235.7B 与每 token 激活 20.9B")
X0, X1 = 30, 710
BH = 34


def bar(y, title, parts, tot):
    svg.text(X0, y - 12, title, size=14, weight="600")
    x = X0
    for name, v, col in parts:
        w = (X1 - X0) * v / tot
        svg.rect(x, y, w, BH, fill=col, fill_opacity=0.75 if col != "muted" else 0.45)
        if w > 60:
            svg.text(x + w / 2, y + BH / 2 + 5, "%.1f%%" % (100 * v / tot), size=12, anchor="middle",
                     color="surface", weight="600")
        x += w
    svg.rect(X0, y, X1 - X0, BH, stroke="ink", width=1)
    # legend rows
    ly = y + BH + 24
    for k, (name, v, col) in enumerate(parts):
        cx = X0 + (k % 2) * 350
        cy = ly + (k // 2) * 20
        svg.rect(cx, cy - 10, 12, 12, fill=col, fill_opacity=0.75 if col != "muted" else 0.45, rx=2)
        svg.text(cx + 18, cy, "%s：%s B（%s%%）" % (name, fmt(round(v / 1e9, 2)), fmt(round(100 * v / tot, 1))),
                 size=12)


bar(52, "总参数 %.1fB（论文：236B）" % (T / 1e9), total_parts, T)
bar(206, "每个 token 激活 %.1fB（论文：21B；含输出头、不含输入嵌入）" % (A / 1e9), act_parts, A)
# the two totals on one common scale
y3 = 352
svg.text(X0, y3 - 12, "放到同一把尺子上：激活参数只占总参数的 %.1f%%" % (100 * A / T), size=14, weight="600")
svg.rect(X0, y3, X1 - X0, 16, fill="muted", fill_opacity=0.18, stroke="muted", width=0.8)
svg.rect(X0, y3, (X1 - X0) * A / T, 16, fill="blue", fill_opacity=0.8)
svg.text(X0 + (X1 - X0) * A / T + 8, y3 + 13, "20.9B", size=12, color="blue", weight="600")
svg.text(X1 - 6, y3 + 13, "235.7B", size=12, anchor="end", color="muted")
svg.text(X0, y3 + 42, "按 HF config.json 逐项计数（嵌入与输出头不共享权重）。上面两根条各自按 100% 画。", size=12,
         color="muted")
svg.save("l10-param-breakdown.svg")
