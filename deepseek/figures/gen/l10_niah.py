"""Figure 10.12: "Needle In A Haystack" on DeepSeek-V2 Base at up to 128K context, redrawn after Fig. 4 of
the V2 paper.

The paper gives the figure only as an image; the cell scores here were read off its colour scale
(lesson-10-doc sec. 3.5 and sec. 10.1 #10: pixel lookup gave ~2.9 for three cells in the 35K column,
~7.0 for one cell in the 104K column, ~10 elsewhere). They are approximate.
Writes deepseek/figures/l10-niah.svg
"""

from l10_svg import SVG

lengths = ["1K", "12K", "24K", "35K", "47K", "58K", "70K", "81K", "93K", "104K", "116K", "128K"]
depths = [0, 9, 18, 27, 36, 45, 55, 64, 73, 82, 91, 100]
score = {(l, d): 10 for l in lengths for d in depths}
score[("35K", 27)] = score[("35K", 36)] = score[("35K", 64)] = 3
score[("104K", 45)] = 7
assert sum(v < 10 for v in score.values()) == 4

W, H = 760, 430
svg = SVG(W, H, "大海捞针测试：DeepSeek-V2 Base 在 1K–128K 上下文、12 个插入深度上的得分（近似重绘）")
X0, Y0, CW, CH = 84, 30, 42, 28


def fill(v):
    if v >= 9.5:
        return "green", 0.55
    if v >= 5:
        return "amber", 0.55
    return "red", 0.6


for c, l in enumerate(lengths):
    for r_, d in enumerate(depths):
        v = score[(l, d)]
        col, op = fill(v)
        x, y = X0 + c * CW, Y0 + r_ * CH
        svg.rect(x + 1, y + 1, CW - 2, CH - 2, fill=col, fill_opacity=op, rx=2)
        if v < 10:
            svg.text(x + CW / 2, y + CH / 2 + 5, "≈%d" % v, size=12, anchor="middle", color="ink", weight="600")
for c, l in enumerate(lengths):
    svg.text(X0 + c * CW + CW / 2, Y0 + 12 * CH + 18, l, size=11, anchor="middle", color="muted")
for r_, d in enumerate(depths):
    svg.text(X0 - 8, Y0 + r_ * CH + CH / 2 + 4, "%d%%" % d, size=11, anchor="end", color="muted")
svg.text(X0 + 6 * CW, Y0 + 12 * CH + 40, "上下文长度（token）", size=12, anchor="middle", color="muted")
svg.text(24, Y0 + 6 * CH, "针插入的深度（占全文的百分比）", size=12, anchor="middle", color="muted", rotate=-90)

# legend
LX = X0 + 12 * CW + 22
svg.text(LX, Y0 + 10, "得分（1–10）", size=12, weight="600")
for k, (v, lab) in enumerate(((10, "≈10：找到"), (7, "≈7：部分"), (3, "≈3：失败"))):
    col, op = fill(v)
    svg.rect(LX, Y0 + 26 + k * 28, 22, 18, fill=col, fill_opacity=op, rx=2)
    svg.text(LX + 30, Y0 + 40 + k * 28, lab, size=12)
svg.text(LX, Y0 + 140, "144 格中 140 格", size=12, color="muted")
svg.text(LX, Y0 + 158, "约为满分；", size=12, color="muted")
svg.text(LX, Y0 + 176, "35K 列三格、", size=12, color="muted")
svg.text(LX, Y0 + 194, "104K 列一格失分", size=12, color="muted")
svg.text(LX, Y0 + 230, "测的是 Base 模型", size=12, color="ink", weight="600")
svg.text(LX, Y0 + 248, "（YaRN 续训后的基础", size=12, color="muted")
svg.text(LX, Y0 + 266, "模型，不是对话模型）", size=12, color="muted")
svg.save("l10-niah.svg")
