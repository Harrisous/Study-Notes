"""Figure 13.16: the published training bill of DeepSeek-R1 (Nature SI Table 4 = arXiv v2 Table 7,
H800 at $2 per GPU hour) and its scale against the full DeepSeek-V3 training bill (2.788M GPU hours,
DeepSeek-V3 technical report). Bars computed here.
Writes deepseek/figures/l13-cost.svg
"""

from l13_svg import SVG, Axes

R1 = [("R1-Zero 的 RL", 101, "512 张 H800 × 约 198 小时", "blue"),
      ("造 SFT 数据", 5, "5K GPU 小时", "amber"),
      ("R1 的训练", 41, "512 张 H800 × 约 80 小时", "blue")]
TOTAL = sum(v for _, v, _, _ in R1)
assert TOTAL == 147
V3 = 2788

W, H = 760, 380
svg = SVG(W, H, "DeepSeek-R1 公开的训练账单，以及它与 DeepSeek-V3 全部训练成本的比例")
svg.text(16, 22, "(a) R1 的三项成本（千 GPU 小时；按每 GPU 小时 2 美元折算）", size=13, weight="600")
ax = Axes(150, 600, 36, 170, (0, 160), (0, 1))
for v in (0, 40, 80, 120, 160):
    x = ax.px(v)
    svg.line(x, 34, x, 172, color="grid", width=1)
    svg.text(x, 188, "%dK" % v, size=11, anchor="middle", color="muted")
rows = R1 + [("合计", TOTAL, "", "ink")]
for k, (name, v, note, col) in enumerate(rows):
    y = 40 + k * 32
    svg.text(142, y + 17, name, size=12.5, anchor="end", weight="700" if name == "合计" else None)
    svg.rect(ax.px(0), y + 2, ax.px(v) - ax.px(0), 22, fill=col, fill_opacity=0.75 if col != "ink" else 0.6)
    svg.text(ax.px(v) + 6, y + 17, "%dK · $%dK" % (v, 2 * v), size=12, weight="600")
    if note:
        svg.text(744, y + 17, note, size=12, anchor="end", color="muted")

svg.text(16, 222, "(b) 放到同一把尺子上：DeepSeek-V3 的全部训练（第 12 课）与 R1 的后训练", size=13, weight="600")
bx = Axes(150, 700, 230, 300, (0, 3000), (0, 1))
for v in (0, 500, 1000, 1500, 2000, 2500, 3000):
    x = bx.px(v)
    svg.line(x, 234, x, 302, color="grid", width=1)
    svg.text(x, 318, "%dK" % v if v else "0", size=11, anchor="middle", color="muted")
svg.text(142, 256, "V3 全部训练", size=12.5, anchor="end")
svg.rect(bx.px(0), 242, bx.px(V3) - bx.px(0), 22, fill="muted", fill_opacity=0.55)
svg.text(bx.px(V3) - 6, 258, "2,788K · $5.576M", size=12, anchor="end", weight="600")
svg.text(142, 288, "R1 后训练", size=12.5, anchor="end")
svg.rect(bx.px(0), 274, bx.px(TOTAL) - bx.px(0), 22, fill="blue", fill_opacity=0.75)
svg.text(bx.px(TOTAL) + 6, 290, "147K · $294K，约为 V3 的 %.1f%%" % (100 * TOTAL * 2 / (V3 * 2)), size=12,
         weight="600")
svg.text(16, 346, "V3 的账单不含前期研究与消融实验。R1 这张只列了三项，没有列出：V3-Base 的预训练、30B 小模型在 A100",
         size=12, color="muted")
svg.text(16, 364, "上的准备实验、PRM 与 MCTS 等失败尝试、蒸馏 6 个小模型；奖励模型的训练是否计入，论文没有说明。",
         size=12, color="muted")
svg.save("l13-cost.svg")
