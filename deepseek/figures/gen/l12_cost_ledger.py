"""Figure 12.1: the DeepSeek-V3 training bill (V3 report Table 1) drawn to scale.

GPU hours: pre-training 2664K, context extension 119K, post-training 5K, total 2788K,
at the paper's assumed rental price of $2 per H800 GPU hour. Days are computed for the
2048-GPU cluster. The dashed boxes list what the number does not contain (the paper
excludes only "prior research and ablation experiments"; the rest is not discussed).
Writes deepseek/figures/l12-cost-ledger.svg
"""

from l12_svg import SVG

stages = [  # name, K GPU hours, colour
    ("预训练", 2664, "blue"),
    ("上下文扩展", 119, "violet"),
    ("后训练", 5, "green"),
]
PRICE = 2.0
GPUS = 2048
total = sum(k for _, k, _ in stages)
assert total == 2788
assert abs(14.8 * 180 - 2664) < 1e-9          # 180K GPU h per trillion tokens x 14.8T

W, H = 720, 356
X0, X1 = 40, 680
BAR_Y, BAR_H = 92, 44
scale = (X1 - X0) / total

svg = SVG(W, H, "DeepSeek-V3 正式训练的 GPU 小时与美元（按论文表 1，按比例绘制）")
svg.text(X0, 28, "正式训练一次：2,788K H800 GPU 小时 × 每小时 $2（假设租价）= $5.576M", size=14, weight="600")
svg.text(X0, 50, "柱长按 GPU 小时等比例；天数按 2048 张 GPU 同时工作计算", size=12, color="muted")

x = X0
geo = []
for name, k, col in stages:
    w = k * scale
    svg.rect(x, BAR_Y, w, BAR_H, fill=col, fill_opacity=0.22, stroke=col, width=1.4)
    geo.append((name, k, col, x, w))
    x += w

# pre-training label inside its segment
name, k, col, x0, w = geo[0]
days = k * 1000 / GPUS / 24
svg.text(x0 + 14, BAR_Y + 19, "预训练 2,664K GPU·h（95.55%）", size=13, color="blue", weight="600")
svg.text(x0 + 14, BAR_Y + 36, "$5.328M · 14.8T token · 约 %.1f 天" % days, size=12, color="ink")

# small segments: callouts above the bar
name, k, col, x1, w1 = geo[1]
d1 = k * 1000 / GPUS / 24
svg.line(x1 + w1 / 2, BAR_Y, x1 + w1 / 2, BAR_Y - 14, color="violet", width=1.2)
svg.text(x1 + w1 / 2 - 2, BAR_Y - 20, "上下文扩展 119K（4.27%%）· $0.238M · %.1f 天" % d1, size=12,
         color="violet", anchor="end")
name, k, col, x2, w2 = geo[2]
d2 = k * 1000 / GPUS / 24
svg.line(x2 + w2 / 2, BAR_Y + BAR_H, x2 + w2 / 2, BAR_Y + BAR_H + 16, color="green", width=1.2)
svg.text(x2 + w2 / 2 + 2, BAR_Y + BAR_H + 30, "后训练 5K（0.18%）", size=12, color="green", anchor="end")
svg.text(x2 + w2 / 2 + 2, BAR_Y + BAR_H + 46, "$0.01M · %.1f 天" % d2, size=12, color="green", anchor="end")

# consistency line
yy = BAR_Y + BAR_H + 30
svg.text(X0, yy, "对账：每万亿 token 180K GPU·h × 14.8T = 2,664K", size=12, color="ink")
svg.text(X0, yy + 16, "与预训练一格恰好相等", size=12, color="muted")

# what is not in the bill
BY = 240
svg.text(X0, BY - 12, "不在这张账里的：", size=13, weight="600")
boxes = [
    (["先前的研究", "与消融实验"], "论文明确排除", "red"),
    (["集群购置", "（$2 只是假设租价）"], "论文未讨论", "muted"),
    (["人员、数据", "采集与标注"], "论文未讨论", "muted"),
    (["R1 等配套模型", "的训练"], "论文未讨论", "muted"),
]
bw, gap = 148, 16
for i, (lines, tag, col) in enumerate(boxes):
    bx = X0 + i * (bw + gap)
    svg.rect(bx, BY, bw, 62, stroke="muted", width=1.2, dash="5 4", rx=6)
    svg.text(bx + bw / 2, BY + 26, lines[0], size=12, anchor="middle")
    svg.text(bx + bw / 2, BY + 44, lines[1], size=12, anchor="middle")
    svg.text(bx + bw / 2, BY + 80, tag, size=12, anchor="middle", color=col)

svg.text(X0, H - 6, "数据：DeepSeek-V3 技术报告表 1 与 §1；天数 = GPU 小时 ÷ 2048 ÷ 24", size=11, color="muted")
svg.save("l12-cost-ledger.svg")
