"""Figure 3.6 (schematic): reward over-optimisation.

Shape only, after Stiennon et al. 2020, Figure 5 ("eventually the reward model becomes anti-correlated
with human preferences"). No data points were extracted from the paper; both curves are illustrative:
  RM score    s(x) = 0.12 + 0.86 (1 - exp(-x/2.4))           keeps rising
  human pref  h(x) = 0.12 + 0.55 (x/3) exp(1 - x/3)          rises, peaks at x = 3, then falls
Writes deepseek/figures/l03-overopt.svg
"""
import math
from l03_svg import SVG, Axis, polyline_d

W, H = 720, 320
svg = SVG(W, H, "奖励过度优化（示意）：奖励模型的分数一直上升，真实偏好先升后降")
ax = Axis(0, 10, 70, 540)
ay = Axis(0, 1.05, 262, 46)
svg.line(ax(0), ay(0), ax(10), ay(0), color="muted", width=1.1)
svg.line(ax(0), ay(0), ax(0), ay(1.05), color="muted", width=1.1)
svg.text(ax(5), ay(0) + 24, "对奖励模型的优化强度（离参考模型越来越远，KL 增大）", size=12.5, anchor="middle")
svg.text(ax(0) - 10, ay(1.05) - 14, "分数（示意，无刻度）", size=12.5)
xs = [10 * i / 200 for i in range(201)]
s = lambda x: 0.12 + 0.86 * (1 - math.exp(-x / 2.4))
h = lambda x: 0.12 + 0.55 * (x / 3.0) * math.exp(1 - x / 3.0)
xpk = 3.0
svg.rect(ax(xpk), ay(1.05), ax(10) - ax(xpk), ay(0) - ay(1.05), fill="red", fill_opacity=0.05)
svg.path(polyline_d([(ax(x), ay(s(x))) for x in xs]), stroke="green", width=2.6)
svg.path(polyline_d([(ax(x), ay(h(x))) for x in xs]), stroke="ink", width=2.4, dash="7 4")
svg.line(ax(xpk), ay(0), ax(xpk), ay(1.02), color="muted", width=1.2, dash="3 3")
svg.text(ax(xpk) - 6, ay(1.0), "真实偏好的最高点", size=12, color="muted", anchor="end")
svg.text(ax(5.6), ay(0.24), "过度优化区：", size=12.5, color="red", weight="600")
svg.text(ax(5.6), ay(0.24) + 18, "分数还在涨，人却越来越不满意", size=12.5, color="red")
svg.text(ax(10) + 8, ay(s(10)) + 4, "奖励模型打的分", size=12.5, color="green", weight="600")
svg.text(ax(10) + 8, ay(h(10)) + 4, "人的真实偏好", size=12.5, weight="600")
svg.text(16, 308, "示意图：形状按 Stiennon 等人（2020）图 5 的定性结论绘制；曲线是示意函数，不是论文数据。", size=12, color="muted")
svg.save("l03-overopt.svg")
