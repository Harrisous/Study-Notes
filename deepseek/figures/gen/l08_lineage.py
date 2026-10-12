"""Figure 8.1: the DeepSeekMath model lineage and the GSM8K / MATH scores at each stage.

Scores (few-shot CoT for Base, Top1 CoT without tools for Instruct / RL) are from
DeepSeekMath v3, Table 2 and Table 5 (research notes lesson-08-doc.md, C11, C27).
Bars are drawn to scale (0-100 %). Writes deepseek/figures/l08-lineage.svg
"""

from l08_svg import SVG, box

W, H = 740, 500
svg = SVG(W, H, "DeepSeekMath 的模型谱系与各阶段成绩")

# ---- left column: the chain of checkpoints
BX, BW, BH = 16, 250, 50
rows = [
    ("DeepSeek-LLM 7B Base", "第 6 课的通用模型", "muted"),
    ("DeepSeek-Coder-Base-v1.5 7B", "取“学习率衰减前”的检查点", "muted"),
    ("DeepSeekMath-Base 7B", "数学继续预训练", "amber"),
    ("DeepSeekMath-Instruct 7B", "监督微调（SFT）", "blue"),
    ("DeepSeekMath-RL 7B", "GRPO 强化学习", "green"),
]
steps = [
    "继续预训练（源代码占 70%）",
    "+500B tokens，56% 来自数学语料",
    "SFT：776K 条样本，500 步",
    "GRPO：约 144K 道题，每题 G = 64",
]
Y0, DY = 34, 92
ys = [Y0 + k * DY for k in range(len(rows))]
for (name, sub, col), y in zip(rows, ys):
    box(svg, BX, y, BW, BH, [name, sub], color=col, size=13, fill_opacity=0.10)
for k, s in enumerate(steps):
    y1 = ys[k] + BH
    y2 = ys[k + 1]
    xm = BX + 38
    svg.arrow(xm, y1 + 2, xm, y2 - 2, color="ink", width=1.6)
    svg.text(xm + 12, (y1 + y2) / 2 + 4, s, size=12, color="muted")

# ---- right panel: horizontal bars for the three stages that have scores
PX0, PX1 = 330, 700          # 0 % .. 100 %


def px(v):
    return PX0 + v / 100.0 * (PX1 - PX0)


scores = {2: (64.2, 36.2), 3: (82.9, 46.8), 4: (88.2, 51.7)}
top, bot = ys[2] - 14, ys[4] + BH + 10
for v in range(0, 101, 20):
    x = px(v)
    svg.line(x, top, x, bot, color="grid", width=1)
    svg.text(x, bot + 16, "%d%%" % v, size=11, anchor="middle", color="muted")
svg.line(PX0, top, PX0, bot, color="muted", width=1.2)

# GPT-4 reference lines (Table 5, CoT): GSM8K 92.0, MATH 52.9
for v, col, lab, dy in ((92.0, "blue", "GPT-4 GSM8K 92.0%", 0), (52.9, "violet", "GPT-4 MATH 52.9%", 0)):
    x = px(v)
    svg.line(x, top - 6, x, bot, color=col, width=1.3, dash="5 4")
    svg.text(x, top - 12 - dy, lab, size=11, anchor="middle", color=col)

for k, (g, m) in scores.items():
    y = ys[k]
    for j, (v, col, lab) in enumerate(((g, "blue", "GSM8K"), (m, "violet", "MATH"))):
        yy = y + 4 + j * 22
        svg.rect(PX0, yy, px(v) - PX0, 17, fill=col, fill_opacity=0.85, rx=2)
        svg.text(PX0 + 8, yy + 13, "%s %.1f%%" % (lab, v), size=12, color="surface", weight="600")

# Minerva 540B markers in the Base row (Table 2): GSM8K 58.8, MATH 33.6
yb = ys[2]
for j, v in enumerate((58.8, 33.6)):
    x, yy = px(v), yb + 4 + j * 22 + 8.5
    svg.polygon([(x, yy - 7), (x + 6, yy), (x, yy + 7), (x - 6, yy)], fill="surface", stroke="ink", width=1.4)
xl, yl = px(58.8) + 4, yb - 8
svg.polygon([(xl, yl - 5), (xl + 4, yl), (xl, yl + 5), (xl - 4, yl)], fill="surface", stroke="ink", width=1.2)
svg.text(xl + 9, yl + 4, "Minerva 540B（参数多 77 倍）", size=11, color="ink", halo=True)

svg.text(PX0, ys[0] + 14, "成绩：Base 为 few-shot 思维链；", size=12, color="muted")
svg.text(PX0, ys[0] + 32, "Instruct / RL 为不用工具、不投票的 Top1", size=12, color="muted")
svg.text(PX0, ys[1] + 22, "（前两个检查点的成绩不在本课范围）", size=12, color="muted")

svg.save("l08-lineage.svg")
