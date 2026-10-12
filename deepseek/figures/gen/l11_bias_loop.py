"""Figure 11.6: the bias feedback loop (redrawn after LossFree fig. 1, arXiv:2408.15664).

Toy numbers from the original figure (checked against the PNG in the research notes, sec. 4.5):
4 experts, loads [5, 2, 1, 4] (6 tokens x top-2), mean 3, old bias [0.1, -0.2, -0.1, -0.1],
update rate u = 0.1 (inferred from the figure), new bias = old + u * sign(mean - load)
= [0.0, -0.1, 0.0, -0.2].
Writes deepseek/figures/l11-bias-loop.svg
"""

import numpy as np

from l11_svg import SVG, box

load = np.array([5, 2, 1, 4])
old = np.array([0.1, -0.2, -0.1, -0.1])
u = 0.1
mean = load.mean()
delta = mean - load
new = old + u * np.sign(delta)
assert np.allclose(new, [0.0, -0.1, 0.0, -0.2])
maxvio = (load.max() - mean) / mean

W, H = 720, 420
svg = SVG(W, H, "偏置的反馈回路：按上一步的负载调整下一步的选择")

# ---------------------------------------------------------------- left: one training step
box(svg, 16, 34, 330, 330, "", color="blue", fill_opacity=0.04)
svg.text(30, 58, "① 训练一步", size=14, weight="700", color="blue")
svg.text(30, 80, "选择用 s + b（带偏置的 Top-2）", size=12, color="violet")
svg.text(30, 98, "加权用 s（偏置不进入门控值）", size=12, color="green")
PX0, PY1, PH = 70, 320, 180
unit = PH / 6


def py(v):
    return PY1 - v * unit


for v in range(0, 7):
    svg.line(PX0 - 4, py(v), PX0 + 206, py(v), color="grid", width=1)
    svg.text(PX0 - 10, py(v) + 4, str(v), size=11, anchor="end", color="muted")
for i, l in enumerate(load):
    x = PX0 + 10 + i * 50
    over = l > mean
    col = "red" if over else "blue"
    svg.rect(x, py(l), 34, PY1 - py(l), fill=col, fill_opacity=0.45, stroke=col, width=1)
    svg.text(x + 17, py(l) - 6, str(l), size=12, anchor="middle", color=col, weight="600")
    svg.text(x + 17, PY1 + 18, "专家 %d" % (i + 1), size=12, anchor="middle")
svg.line(PX0, py(mean), PX0 + 206, py(mean), color="green", width=1.8, dash="6 4")
svg.text(PX0 + 210, py(mean) + 4, "均值 3", size=12, color="green")
svg.text(38, (py(6) + PY1) / 2, "负载（token 数）", size=12, anchor="middle", rotate=-90)
svg.text(30, 356, "MaxVio = (5 − 3)/3 ≈ %.3f" % maxvio, size=12)

# ---------------------------------------------------------------- right: bias update table
RX = 380
box(svg, RX - 14, 34, 352, 330, "", color="violet", fill_opacity=0.04)
svg.text(RX, 58, "② 按负载更新偏置（γ = 0.1）", size=14, weight="700", color="violet")
svg.text(RX, 80, "b_{i} ← b_{i} + γ · sign(均值 − 负载_{i})", size=12, color="violet")
cols = [RX, RX + 46, RX + 96, RX + 186, RX + 248, RX + 298]
heads = ["专家", "负载", "均值−负载", "方向", "旧 b", "新 b"]
for x, h in zip(cols, heads):
    svg.text(x, 112, h, size=12, weight="600", color="muted")
svg.line(RX, 120, RX + 330, 120, color="rule", width=1)
for i in range(4):
    y = 148 + i * 44
    over = load[i] > mean
    col = "red" if over else "blue"
    svg.text(cols[0], y, str(i + 1), size=13)
    svg.text(cols[1], y, str(load[i]), size=13, color=col)
    svg.text(cols[2], y, "%+d" % delta[i], size=13)
    svg.text(cols[3], y, "↓ 调低" if over else "↑ 调高", size=13, color=col, weight="600")
    svg.text(cols[4], y, "%+.1f" % old[i], size=13, color="muted")
    svg.text(cols[5], y, ("%+.1f" % new[i]).replace("+0.0", "0.0"), size=13, color="violet", weight="700")
svg.text(RX, 330, "过载专家（红）的偏置调低，下一步更难被选中；", size=12)
svg.text(RX, 348, "欠载专家（蓝）的偏置调高，下一步更容易被选中。", size=12)

# ---------------------------------------------------------------- loop arrows
svg.arrow(346, 190, 364, 190, color="ink", width=1.8)
svg.path("M 540 364 V 392 H 180 V 367", stroke="violet", width=1.8, arrow=True)
svg.text(360, 408, "③ 下一步用新偏置做选择（只依赖已经过去的 batch，不偷看当前序列）", size=12, anchor="middle",
         color="violet")

svg.save("l11-bias-loop.svg")
