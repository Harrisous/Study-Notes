"""Figure 7.6: routing collapse is a self-reinforcing loop; a balance loss pushes back.

Left: the feedback loop described in Shazeer et al. (2017) section 4.
Right: Shazeer et al. (2017) Appendix A, Table 6 (MoE-256 language model), two rows:
  no balance loss:        test ppl 39.8, CV(Importance) 3.04, max/mean load 17.80
  w_imp = w_load = 0.1:   test ppl 35.6, CV(Importance) 0.06, max/mean load 1.14
Bar lengths are computed from these numbers (each metric on its own linear scale).
Writes deepseek/figures/l07-collapse-loop.svg
"""

import math

from l07_svg import SVG

W, H = 740, 360
svg = SVG(W, H, "路由坍缩的自我强化回路，以及均衡损失的效果（Shazeer 等 2017 表 6）")

# ---------------- left: loop (four boxes on a rectangle, clockwise)
boxes = [
    (20, 52, "路由器对专家 A", "的打分略高一点", "violet"),
    (236, 52, "A 分到更多 token", "", "blue"),
    (236, 204, "A 得到更多梯度更新", "变得更“好用”", "blue"),
    (20, 204, "更多 token 与 A 的", "亲和度变高", "violet"),
]
BWX, BHX = 150, 46
for x, y, l1, l2, col in boxes:
    svg.rect(x, y, BWX, BHX, fill=col, fill_opacity=0.1, stroke=col, width=1.3, rx=8)
    if l2:
        svg.text(x + BWX / 2, y + 19, l1, size=12, anchor="middle")
        svg.text(x + BWX / 2, y + 36, l2, size=12, anchor="middle")
    else:
        svg.text(x + BWX / 2, y + 28, l1, size=12, anchor="middle")
svg.arrow(20 + BWX + 4, 75, 236 - 4, 75, color="red", width=1.8)
svg.arrow(236 + BWX / 2, 52 + BHX + 4, 236 + BWX / 2, 204 - 4, color="red", width=1.8)
svg.arrow(236 - 4, 227, 20 + BWX + 4, 227, color="red", width=1.8)
svg.arrow(20 + BWX / 2, 204 - 4, 20 + BWX / 2, 52 + BHX + 4, color="red", width=1.8)
svg.text(203, 146, "自我强化", size=14, anchor="middle", color="red", weight="600")
svg.text(203, 164, "（路由坍缩）", size=12, anchor="middle", color="red")

# counter force: balance loss
svg.rect(20, 292, 366, 46, fill="green", fill_opacity=0.1, stroke="green", width=1.3, rx=8)
svg.text(34, 311, "均衡损失：一个专家分到的 token 越多，", size=12, color="green")
svg.text(34, 329, "它的平均亲和度就被往下压得越狠（推导见正文）", size=12, color="green")
svg.arrow(20 + BWX / 2, 292 - 2, 20 + BWX / 2, 204 + BHX + 4, color="green", width=1.8)

# ---------------- right: Shazeer table 6
RX = 420
svg.text(RX, 26, "Shazeer 等 2017 表 6（MoE-256 语言模型）", size=13, weight="600")
svg.text(RX, 44, "同一模型，只改均衡损失的权重", size=12, color="muted")
metrics = [
    ("最忙专家的负载 / 平均负载", 17.80, 1.14, 18.0, "%.2f"),
    ("importance 的变异系数 CV", 3.04, 0.06, 3.2, "%.2f"),
    ("测试困惑度（越低越好）", 39.8, 35.6, 42.0, "%.1f"),
]
BX0, BX1 = RX + 8, RX + 240
y = 76
for name, v0, v1, vmax, f in metrics:
    svg.text(RX, y, name, size=12, color="ink", weight="600")
    for j, (v, col, lab) in enumerate(((v0, "red", "无均衡损失"), (v1, "green", "w_{imp} = w_{load} = 0.1"))):
        yy = y + 10 + j * 22
        w = (BX1 - BX0) * v / vmax
        svg.rect(BX0, yy, max(w, 1.5), 15, fill=col, fill_opacity=0.75)
        svg.text(BX0 + max(w, 1.5) + 6, yy + 12, f % v, size=12, color=col)
    y += 84
svg.rect(RX, H - 36, 12, 12, fill="red", fill_opacity=0.75)
svg.text(RX + 18, H - 25, "无均衡损失", size=12)
svg.rect(RX + 106, H - 36, 12, 12, fill="green", fill_opacity=0.75)
svg.text(RX + 124, H - 25, "w_{imp} = w_{load} = 0.1", size=12)
svg.text(RX, H - 6, "困惑度坐标从 0 开始：39.8 → 35.6 是约 11% 的改善", size=11, color="muted")

svg.save("l07-collapse-loop.svg")
