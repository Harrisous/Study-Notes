"""Figure 3.10: how GAE weights future TD residuals, (gamma*lambda)^l.

  gamma*lambda = 0.9405  (gamma 0.99, lambda 0.95: PPO paper, MuJoCo, Table 3)
  gamma*lambda = 0.95    (gamma 1, lambda 0.95: Stiennon et al. 2020, App. B.1)
  gamma*lambda = 0.99    (e.g. gamma 0.99, lambda 1)
  gamma*lambda = 1       (gamma = lambda = 1: Monte-Carlo return minus baseline)
Effective horizon 1/(1 - gamma*lambda) = 16.8, 20, 100, infinite.
Writes deepseek/figures/l03-gae-weights.svg
"""
from l03_svg import SVG, Axis, polyline_d, fmt

W, H = 720, 346
svg = SVG(W, H, "GAE 对往后第 l 个 TD 残差的权重：γλ 的 l 次方")
ax = Axis(0, 60, 70, 500)
ay = Axis(0, 1.05, 270, 40)
for v in (0, 10, 20, 30, 40, 50, 60):
    svg.line(ax(v), ay(0), ax(v), ay(1.05), color="grid", width=1)
    svg.text(ax(v), ay(0) + 16, str(v), size=11, anchor="middle", color="muted")
for v in (0, 0.25, 0.5, 0.75, 1.0):
    svg.line(ax(0), ay(v), ax(60), ay(v), color="grid", width=1)
    svg.text(ax(0) - 6, ay(v) + 4, fmt(v), size=11, anchor="end", color="muted")
svg.line(ax(0), ay(0), ax(60), ay(0), color="muted", width=1.1)
svg.line(ax(0), ay(0), ax(0), ay(1.05), color="muted", width=1.1)
svg.text(ax(30), ay(0) + 36, "往后数第 l 个 token 的 TD 残差 δ_{t+l}", size=12.5, anchor="middle")
svg.text(ax(0) - 44, ay(1.05) - 14, "权重 (γλ)^{l}", size=12.5)
curves = [(0.9405, "blue", "γλ = 0.9405（PPO 论文，MuJoCo）", 16.81), (0.95, "violet", "γλ = 0.95（Stiennon 等，γ = 1）", 20.0),
          (0.99, "amber", "γλ = 0.99（有效视野 100）", 100.0)]
ls = [60 * i / 300 for i in range(301)]
svg.line(ax(0), ay(1), ax(60), ay(1), color="ink", width=2, dash="7 4")
for g, col, lab, hz in curves:
    svg.path(polyline_d([(ax(l), ay(g ** l)) for l in ls]), stroke=col, width=2.4)
    if hz <= 60:
        svg.line(ax(hz), ay(0), ax(hz), ay(g ** hz), color=col, width=1.1, dash="3 3")
        left = hz < 18
        svg.text(ax(hz) + (-4 if left else 4), ay(0.08), fmt(hz), size=11.5, color=col, anchor="end" if left else "start", weight="600")
# lambda = 0 : only l = 0 has weight
svg.rect(ax(0), ay(1), 4, ay(0) - ay(1), fill="red", fill_opacity=0.6)
lx, ly = 516, 60
items = [("ink", "γλ = 1（回报减 V，蒙特卡洛）", "7 4")] + [(c, lab, None) for g, c, lab, hz in curves] + [("red", "λ = 0（只用 δ_{t} 本身）", None)]
for k, (col, lab, dash) in enumerate(items):
    yy = ly + k * 44
    svg.line(lx, yy, lx + 26, yy, color=col, width=2.4, dash=dash)
    words = lab.split("（")
    svg.text(lx + 32, yy + 4, words[0], size=12, color=col if col != "amber" else "amber")
    if len(words) > 1:
        svg.text(lx + 32, yy + 20, "（" + words[1], size=11.5, color="muted")
svg.text(ax(0), ay(0) + 58, "竖虚线标出“有效视野” 1/(1−γλ)：在那里权重约为 0.36，接近 1/e。", size=12, color="muted")
svg.save("l03-gae-weights.svg")
