"""Figure 7.13: the 2B validation experiments, compute (FLOPs per 2K tokens, log axis) vs Pile loss.

DeepSeekMoE arXiv v1 Tables 1 and 2 (research notes section 5.2, figure 7.10 there):
Dense 0.2B (2.9T, 2.060); Hash Layer (2.9T, 1.932); Switch (2.9T, 1.881); GShard (4.3T, 1.867);
DeepSeekMoE (4.3T, 1.808); GShard x1.5 (5.8T, 1.808); Dense x16 (24.6T, 1.806).
All models trained on 100B tokens. Positions computed from these numbers.
Writes deepseek/figures/l07-2b-results.svg
"""

import math

from l07_svg import SVG

pts = [
    ("Dense（0.2B）", 2.9, 2.060, "muted", "right"),
    ("Hash Layer", 2.9, 1.932, "muted", "right"),
    ("Switch", 2.9, 1.881, "muted", "right"),
    ("GShard", 4.3, 1.867, "ink", "right"),
    ("DeepSeekMoE", 4.3, 1.808, "blue", "leftb"),
    ("GShard×1.5", 5.8, 1.808, "ink", "above"),
    ("Dense×16", 24.6, 1.806, "ink", "above"),
]

W, H = 740, 400
X0, X1, Y0, Y1 = 80, 700, 30, 330
LX0, LX1 = math.log10(2.0), math.log10(32.0)
V0, V1 = 1.78, 2.09


def px(v):
    return X0 + (math.log10(v) - LX0) / (LX1 - LX0) * (X1 - X0)


def py(v):
    return Y1 - (v - V0) / (V1 - V0) * (Y1 - Y0)


svg = SVG(W, H, "DeepSeekMoE 2B 验证实验：计算量与 Pile 损失")
for v in (1.8, 1.85, 1.9, 1.95, 2.0, 2.05):
    svg.line(X0, py(v), X1, py(v), color="grid", width=1)
    svg.text(X0 - 8, py(v) + 4, "%.2f" % v, size=11, anchor="end", color="muted")
for v in (2, 4, 8, 16, 32):
    svg.line(px(v), Y0, px(v), Y1, color="grid", width=1)
    svg.text(px(v), Y1 + 18, "%dT" % v, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "每 2K token 的计算量（FLOPs，对数刻度）", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "Pile 损失（越低越好）", size=12, anchor="middle", color="muted", rotate=-90)

# iso-loss line at 1.808
yl = py(1.808)
svg.line(px(4.3), yl, px(24.6), yl, color="blue", width=1.2, dash="4 3")
svg.text(px(9.5), yl + 20, "同样的 Pile 损失：GShard×1.5 要 %.2f 倍计算，Dense×16 要 %.1f 倍" % (5.8 / 4.3, 24.6 / 4.3),
         size=12, anchor="middle", color="blue")
# arrow from GShard to DeepSeekMoE at equal compute
svg.arrow(px(4.3) + 14, py(1.867) + 10, px(4.3) + 14, py(1.808) - 6, color="green", width=1.6)
svg.text(px(4.3) + 6, (py(1.867) + py(1.808)) / 2 + 4, "同计算、同参数：−0.059", size=12, anchor="end", color="green")

for name, x, y, col, pos in pts:
    cx, cy = px(x), py(y)
    r = 6 if col == "blue" else 5
    svg.circle(cx, cy, r, fill=col if col != "muted" else "surface", stroke=col, width=1.8)
    lab = "%s %.3f" % (name, y)
    if pos == "left":
        svg.text(cx - 10, cy + 4, lab, size=12, anchor="end", color="ink")
    elif pos == "right":
        svg.text(cx + 10, cy + 4, lab, size=12, color="ink")
    elif pos == "leftb":
        svg.text(cx - 12, cy + 5, lab, size=13, anchor="end", color="blue", weight="600")
    else:
        svg.text(cx, cy - 12, lab, size=12, anchor="middle", color="ink")

svg.save("l07-2b-results.svg")
