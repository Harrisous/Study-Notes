"""Figure 14.12: DSA's two continued-training stages (V3.2-Exp report section 2.1).

Stage 1 (dense warm-up): only the indexer trains, lr 1e-3, 1000 steps x 16 sequences x 128K tokens.
Stage 2 (sparse training): all parameters, lr 7.3e-6, top-2048, 15000 steps x 480 x 128K tokens.
128K = 131,072 (only this value reproduces the paper's 943.7B). Token counts computed below;
the bar lengths use a log scale.
Writes deepseek/figures/l14-training-stages.svg
"""

import math

from l14_svg import SVG

L128K = 131072
WARM = 1000 * 16 * L128K
SPARSE = 15000 * 480 * L128K
assert round(WARM / 1e9, 1) == 2.1 and round(SPARSE / 1e9, 1) == 943.7
assert round(15000 * 480 * 128000 / 1e9, 1) == 921.6        # 128K = 128,000 would not match
RATIO = SPARSE / WARM
assert RATIO == 450

W, H = 760, 480
svg = SVG(W, H, "DSA 的两阶段续训：先稠密预热只训 indexer，再全参数稀疏训练")

# ---------------------------------------------------------------- flow
Y = 40
svg.box(12, Y + 34, 118, 92, ["V3.1-Terminus", "基座检查点", ("已扩到 128K", {"size": 11, "color": "muted"})],
        color="ink", fill_opacity=0.05, size=12, line_gap=22)
svg.arrow(132, Y + 80, 150, Y + 80, color="ink", width=1.4)

svg.rect(152, Y, 250, 214, fill="violet", fill_opacity=0.08, stroke="violet", width=1.4, rx=8)
svg.text(166, Y + 22, "① 稠密预热（dense warm-up）", size=13, weight="700", color="violet")
lines1 = [
    ("注意力：仍是稠密的", "ink"),
    ("可训练：只有 indexer，其余全部冻结", "ink"),
    ("损失：KL(p_{t,:} ‖ softmax(I_{t,:}))，全部位置", "ink"),
    ("学习率 10^{-3}", "muted"),
    ("1000 步 × 16 条 × 128K", "muted"),
    ("= 2,097,152,000 ≈ 2.1B token", "violet"),
]
for i, (t, c) in enumerate(lines1):
    # 27-unit rows: a superscript on one row must clear the subscripts of the row above
    svg.text(166, Y + 50 + i * 27, t, size=12, color=c, weight="700" if i == len(lines1) - 1 else None)
svg.arrow(404, Y + 80, 420, Y + 80, color="ink", width=1.4)

svg.rect(422, Y, 250, 214, fill="blue", fill_opacity=0.07, stroke="blue", width=1.4, rx=8)
svg.text(436, Y + 22, "② 稀疏训练（sparse training）", size=13, weight="700", color="blue")
lines2 = [
    ("注意力：每个查询只看 top-2048", "ink"),
    ("可训练：全部参数（主模型 + indexer）", "ink"),
    ("主模型：语言模型损失；indexer：", "ink"),
    ("KL 只在被选集合 S_{t} 上；两者梯度分离", "ink"),
    ("学习率 7.3 × 10^{-6}", "muted"),
    ("15000 步 × 480 条 × 128K", "muted"),
    ("= 943,718,400,000 ≈ 943.7B token", "blue"),
]
for i, (t, c) in enumerate(lines2):
    svg.text(436, Y + 50 + i * 25, t, size=12, color=c, weight="700" if i == len(lines2) - 1 else None)
svg.arrow(674, Y + 80, 690, Y + 80, color="ink", width=1.4)
svg.box(692, Y + 34, 60, 92, ["后训练", "同样", "稀疏"], color="green", fill_opacity=0.10, size=12, line_gap=22)

svg.text(166, Y + 236, "两阶段的数据分布都与 V3.1-Terminus 的 128K 长上下文扩展数据完全对齐。", size=12, color="muted")

# ---------------------------------------------------------------- token bars (log scale)
BY = 320
X0, X1 = 150, 700
LO, HI = 9, 12            # 1e9 .. 1e12 tokens


def bx(v):
    return X0 + (math.log10(v) - LO) / (HI - LO) * (X1 - X0)


svg.text(12, BY - 16, "训练 token 数（对数刻度）", size=13, weight="700")
for e in range(LO, HI + 1):
    x = bx(10 ** e)
    svg.line(x, BY, x, BY + 104, color="grid", width=1)
    svg.text(x, BY + 122, ["1B", "10B", "100B", "1T"][e - LO], size=11, anchor="middle", color="muted")
for i, (lab, v, col) in enumerate((("① 稠密预热", WARM, "violet"), ("② 稀疏训练", SPARSE, "blue"))):
    y = BY + 12 + i * 48
    svg.text(X0 - 10, y + 22, lab, size=12, anchor="end", color=col, weight="700")
    svg.rect(X0, y + 4, bx(v) - X0, 28, fill=col, fill_opacity=0.55, stroke=col, width=1)
    svg.text(bx(v) + 8, y + 23, "%.1fB" % (v / 1e9), size=12, color=col, weight="700")
svg.text(12, BY + 146, "稀疏阶段的 token 数是预热的 %d 倍；预热只占两阶段合计的 %.2f%%。每步 batch：预热 2.10M token，稀疏 62.9M token。"
         % (RATIO, 100 * WARM / (WARM + SPARSE)), size=12, color="muted")

svg.save("l14-training-stages.svg")
