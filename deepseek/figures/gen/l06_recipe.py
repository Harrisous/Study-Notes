"""Figure 6.1: the whole DeepSeek LLM recipe on one page (schematic).

Numbers on the boxes come from the lesson-06 research notes (dsllm §2.1, §2.2, §2.3, §3, §4):
91 dumps / 89.8 % removed (Table 1), vocabulary 102400, 30 layers MHA / 95 layers GQA,
1.5 M SFT instances, DPO 1 epoch, scaling experiments 1e17 .. 3e20 FLOPs.
Writes deepseek/figures/l06-recipe.svg
"""

from l06_svg import SVG, box

W, H = 760, 420
svg = SVG(W, H, "DeepSeek LLM 的完整配方：数据、分词、架构、预训练、对齐，以及为预训练定超参数的小规模 scaling 实验")

# ---- row 1: data
r1y, r1h = 34, 84
bw, gap = 130, 22
x0 = (W - (5 * bw + 4 * gap)) / 2
row1 = [
    (["原始语料", "Common Crawl 等", "（91 个 dump）"], "muted", None),
    (["① 去重", "跨 91 个 dump", "去掉 89.8% 文档"], "red", "§2"),
    (["② 过滤", "语言学 + 语义", "质量评估"], "ink", "§2"),
    (["③ 重混", "提高少数领域", "的占比"], "ink", "§2"),
    (["预训练语料", "2T tokens", "中英为主"], "green", "§2"),
]
svg.text(x0, r1y - 12, "数据（DeepSeek LLM §2.1）", size=12, color="muted")
cx1 = []
for i, (lines, col, tag) in enumerate(row1):
    x = x0 + i * (bw + gap)
    box(svg, x, r1y, bw, r1h, lines, color=col, fill_opacity=0.08)
    cx1.append(x + bw / 2)
    if tag:
        svg.text(x + bw - 6, r1y + 15, tag, size=11, anchor="end", color="muted")
    if i < 4:
        svg.arrow(x + bw + 2, r1y + r1h / 2, x + bw + gap - 2, r1y + r1h / 2, color="ink", width=1.5)

# ---- row 2: model and training
r2y, r2h = 196, 84
bw2, gap2 = 108, 18
x02 = (W - (6 * bw2 + 5 * gap2)) / 2
row2 = [
    (["分词器", "BBPE", "词表 102400"], "ink", "§2"),
    (["架构", "7B：30 层 MHA", "67B：95 层 GQA"], "ink", "§3"),
    (["预训练", "AdamW", "多步学习率"], "blue", "§4"),
    (["SFT", "150 万条", "指令数据"], "ink", "§8"),
    (["DPO", "偏好对", "1 个 epoch"], "ink", "§8"),
    (["DeepSeek", "Chat", "7B / 67B"], "green", None),
]
svg.text(x02, r2y - 12, "模型与训练（§2.1–§2.3、§4）", size=12, color="muted")
cx2 = []
for i, (lines, col, tag) in enumerate(row2):
    x = x02 + i * (bw2 + gap2)
    box(svg, x, r2y, bw2, r2h, lines, color=col, fill_opacity=0.08, title_size=13, size=12)
    cx2.append(x + bw2 / 2)
    if tag:
        svg.text(x + bw2 - 5, r2y + 15, tag, size=11, anchor="end", color="muted")
    if i < 5:
        svg.arrow(x + bw2 + 2, r2y + r2h / 2, x + bw2 + gap2 - 2, r2y + r2h / 2, color="ink", width=1.5)

# corpus feeds pretraining
svg.path("M%.1f %.1f V %.1f H %.1f V %.1f" % (cx1[4], r1y + r1h + 2, 160, cx2[2] + 20, r2y - 4),
         stroke="green", width=1.6, arrow=True)
svg.text(cx1[4] - 6, 152, "喂给预训练", size=12, anchor="end", color="green")

# ---- row 3: scaling-law study feeding hyperparameters and allocation
r3y, r3h = 330, 70
sx, sw = x02 + (bw2 + gap2) * 0.55, (bw2 + gap2) * 2.3
box(svg, sx, r3y, sw, r3h,
    ["小规模 scaling 实验（§5–§7）", "算力 10^{17}–3×10^{20} FLOPs，拟合幂律"],
    color="violet", fill_opacity=0.06, dash="5 4")
# arrows up
ax1 = cx2[2]
svg.arrow(ax1, r3y - 2, ax1, r2y + r2h + 4, color="violet", width=1.6)
svg.text(ax1 + 6, r3y - 14, "最优 η、B", size=12, color="violet")
ax2 = cx2[1]
svg.arrow(ax2, r3y - 2, ax2, r2y + r2h + 4, color="violet", width=1.6)
svg.text(ax2 - 6, r3y - 14, "模型/数据分配", size=12, anchor="end", color="violet")

# legend-ish note on the right of row 3
nx = sx + sw + 26
svg.text(nx, r3y + 22, "只在左侧小预算上做实验，", size=12, color="muted")
svg.text(nx, r3y + 40, "外推到 7B、67B 的", size=12, color="muted")
svg.text(nx, r3y + 58, "一次性大训练", size=12, color="muted")

svg.save("l06-recipe.svg")
