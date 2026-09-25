"""Figure 3.8: PPO surrogate-objective ablation, Schulman et al. 2017, Table 1.

Average normalised score over 7 MuJoCo environments x 3 random seeds (21 runs each); 1 = best,
0 = random policy (normalisation as described in the paper's sec. 6.1).
Writes deepseek/figures/l03-ppo-ablation.svg
"""
from l03_svg import SVG, Axis, fmt

rows = [
    ("不裁剪、不惩罚", -0.39, "red"),
    ("裁剪 ε = 0.1", 0.76, "blue"),
    ("裁剪 ε = 0.2", 0.82, "blue"),
    ("裁剪 ε = 0.3", 0.70, "blue"),
    ("自适应 KL，d_{targ} = 0.003", 0.68, "violet"),
    ("自适应 KL，d_{targ} = 0.01", 0.74, "violet"),
    ("自适应 KL，d_{targ} = 0.03", 0.71, "violet"),
    ("固定 KL，β = 0.3", 0.62, "muted"),
    ("固定 KL，β = 1", 0.71, "muted"),
    ("固定 KL，β = 3", 0.72, "muted"),
    ("固定 KL，β = 10", 0.69, "muted"),
]
W, H = 720, 416
svg = SVG(W, H, "PPO 论文表 1：不同替代目标的平均归一化得分")
ax = Axis(-0.5, 1.0, 250, 690)
y0, bh, gap = 50, 22, 7
for v in (-0.5, 0, 0.5, 1.0):
    svg.line(ax(v), y0 - 8, ax(v), y0 + len(rows) * (bh + gap), color="grid", width=1)
    svg.text(ax(v), y0 + len(rows) * (bh + gap) + 16, fmt(v).replace("-", "−"), size=11, anchor="middle", color="muted")
svg.line(ax(0), y0 - 8, ax(0), y0 + len(rows) * (bh + gap), color="muted", width=1.2)
svg.text(ax(0.25), y0 + len(rows) * (bh + gap) + 36, "平均归一化得分（1 = 最好，0 = 随机策略）", size=12.5, anchor="middle")
svg.text(16, 26, "7 个 MuJoCo 环境 × 3 个随机种子的平均", size=13.5, weight="600")
y = y0
for lab, v, col in rows:
    xa, xb = sorted((ax(0), ax(v)))
    svg.rect(xa, y, xb - xa, bh, fill=col, fill_opacity=0.28 if v != 0.82 else 0.55, stroke=col, width=1.1)
    svg.text(240, y + 16, lab, size=12.5, anchor="end", weight="600" if v == 0.82 else None)
    tx = ax(v) + 6
    svg.text(tx, y + 16, ("%.2f" % v).replace("-", "−"), size=12, anchor="start",
             weight="700" if v == 0.82 else None)
    y += bh + gap
svg.save("l03-ppo-ablation.svg")
