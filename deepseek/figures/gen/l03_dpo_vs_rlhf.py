"""Figure 3.13: RLHF (reward model + RL loop) versus DPO (one maximum-likelihood step).
Redrawn after Rafailov et al. 2023, Figure 1, in the course colours; schematic.
Writes deepseek/figures/l03-dpo-vs-rlhf.svg
"""
from l03_svg import SVG, box

W, H = 740, 330
svg = SVG(W, H, "RLHF 与 DPO 的流程对比")
svg.line(380, 20, 380, 300, color="rule", width=1.2)

# ---------------- left: RLHF
svg.text(198, 26, "RLHF（奖励模型 + PPO）", size=14.5, anchor="middle", weight="700", color="blue")
box(svg, 24, 50, 150, 54, ["偏好数据", "(q, o_{w}, o_{l})"], stroke="ink", weight="600")
svg.arrow(99, 106, 99, 162, color="ink", width=1.6)
svg.text(106, 138, "最大似然 (3.3)", size=12)
box(svg, 24, 166, 150, 54, ["奖励模型 r_{φ}"], stroke="green", fill="green", fill_opacity=0.10, color="green", weight="700")
box(svg, 222, 166, 146, 54, ["策略 π_{θ}"], stroke="blue", fill="blue", fill_opacity=0.10, color="blue", weight="700", width=2.2)
box(svg, 222, 50, 146, 54, ["参考 π_{ref}（冻结）"], stroke="muted", dash="6 4", weight="600")
svg.arrow(295, 106, 295, 163, color="muted", width=1.4)
svg.text(301, 138, "KL 惩罚", size=12, color="muted")
# RL loop between RM and policy
svg.path("M222 180 C 205 172, 192 172, 177 180", stroke="blue", width=1.6, arrow=True)
svg.text(199, 160, "采样回答", size=11.5, anchor="middle", color="blue")
svg.path("M177 208 C 192 216, 205 216, 219 208", stroke="green", width=1.6, arrow=True)
svg.text(199, 236, "打分", size=11.5, anchor="middle", color="green")
svg.text(198, 262, "强化学习循环：采样 → 打分 → PPO 更新", size=12.5, anchor="middle", color="blue")
svg.text(198, 286, "两个阶段 · 四个网络 · 训练中在线采样", size=12.5, anchor="middle", weight="600")

# ---------------- right: DPO
svg.text(560, 26, "DPO（直接偏好优化）", size=14.5, anchor="middle", weight="700", color="green")
box(svg, 400, 50, 150, 54, ["偏好数据", "(q, o_{w}, o_{l})"], stroke="ink", weight="600")
box(svg, 572, 50, 150, 54, ["参考 π_{ref}（冻结）"], stroke="muted", dash="6 4", weight="600")
svg.arrow(475, 106, 475, 162, color="ink", width=1.6)
svg.text(484, 146, "最大似然：DPO 损失 (3.30)", size=12)
# the reference model's log-probabilities feed the maximum-likelihood step (arrow ends beside it)
svg.path("M647 106 V124 H482", stroke="muted", width=1.4, arrow=True)
svg.text(654, 121, "log π_{ref}", size=12, color="muted")
box(svg, 400, 166, 322, 54, ["最终的语言模型 π_{θ}"], stroke="blue", fill="blue", fill_opacity=0.10, color="blue", weight="700", width=2.2)
svg.text(561, 244, "隐式奖励 r̂_{θ}(q, o) = β·log(π_{θ}/π_{ref})", size=12.5, anchor="middle", color="green")
svg.text(561, 266, "（没有单独的奖励模型，也没有强化学习循环）", size=12, anchor="middle", color="muted")
svg.text(561, 286, "一个阶段 · 两个网络 · 不采样", size=12.5, anchor="middle", weight="600")
svg.text(16, 318, "据 DPO 论文图 1 重绘；颜色按本课程的约定。", size=12, color="muted")
svg.save("l03-dpo-vs-rlhf.svg")
