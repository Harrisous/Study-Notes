"""Figure 10.13: DeepSeek-V2's alignment pipeline -- SFT, then GRPO in two stages with different rewards.

Facts (V2 paper sec. 4.1-4.2): SFT on 1.5M instances (1.2M helpfulness + 0.3M safety), 2 epochs, lr 5e-6;
GRPO (no critic; baseline from group scores); stage 1 reasoning alignment with r = RM_reasoning(o);
stage 2 human-preference alignment with r = c1 RM_helpful + c2 RM_safety + c3 RM_rule (c's unpublished);
reward models initialised from DeepSeek-V2 Chat (SFT); code preference data from compiler feedback, math
preference data from ground-truth labels. Schematic, hand-laid-out.
Writes deepseek/figures/l10-grpo-stages.svg
"""

from l10_svg import SVG, box

W, H = 760, 506
svg = SVG(W, H, "DeepSeek-V2 的对齐流程：SFT，再用 GRPO 分两阶段做强化学习")

# ---- row 1: pipeline
row = [("V2 Base", "8.1T + YaRN", "ink"), ("SFT", "150 万条，2 轮", "ink"),
       ("RL 阶段 1", "推理对齐", "blue"), ("RL 阶段 2", "人类偏好对齐", "blue"), ("V2 Chat (RL)", "", "green")]
x = 16
for k, (a, b, col) in enumerate(row):
    w = 132
    box(svg, x, 22, w, 50, a, color=col, fill=col if col != "ink" else None, fill_opacity=0.10, size=13,
        weight="600", sub=b or None)
    if k < len(row) - 1:
        svg.arrow(x + w, 47, x + w + 17, 47)
    x += w + 18
svg.text(236, 92, "SFT：120 万条有用性 + 30 万条安全，学习率 5×10⁻⁶", size=12, anchor="middle", color="muted")
svg.text(530, 92, "两个阶段用同一个算法（GRPO），换不同的奖励", size=12, anchor="middle", color="muted")

# ---- row 2: one GRPO step
Y = 132
svg.rect(12, Y - 20, 736, 206, stroke="rule", width=1, rx=10)
svg.text(24, Y, "GRPO 的一步（不需要价值网络）", size=13, weight="600")
box(svg, 24, Y + 34, 92, 44, "问题 q", size=13)
svg.arrow(116, Y + 56, 146, Y + 56)
# samples
box(svg, 148, Y + 18, 132, 118, None, color="blue", fill="blue", fill_opacity=0.06)
svg.text(214, Y + 36, "旧策略 π_{θold}", size=12, anchor="middle", color="blue", weight="600")
svg.text(214, Y + 58, "采样 G 个回答", size=12, anchor="middle", color="blue")
for k, lab in enumerate(("o_{1}", "o_{2}", "⋮", "o_{G}")):
    svg.text(214, Y + 76 + k * 16, lab, size=12, anchor="middle")
svg.arrow(280, Y + 76, 312, Y + 76)
box(svg, 314, Y + 30, 120, 92, None, color="green", fill="green", fill_opacity=0.08)
svg.text(374, Y + 52, "奖励", size=13, anchor="middle", color="green", weight="600")
svg.text(374, Y + 72, "r_{1}, r_{2}, …, r_{G}", size=12, anchor="middle")
svg.text(374, Y + 92, "（来自奖励模型，", size=11, anchor="middle", color="muted")
svg.text(374, Y + 108, "见下方两种来源）", size=11, anchor="middle", color="muted")
svg.arrow(434, Y + 76, 464, Y + 76)
box(svg, 466, Y + 30, 124, 92, None, color="ink")
svg.text(528, Y + 52, "组内标准化", size=13, anchor="middle", weight="600")
svg.text(528, Y + 72, "Â_{i} = (r_{i} − 均值)", size=12, anchor="middle")
svg.text(528, Y + 96, "÷ 标准差", size=12, anchor="middle")   # clear of the subscripts above
svg.arrow(590, Y + 76, 620, Y + 76)
box(svg, 622, Y + 22, 118, 108, None, color="blue", fill="blue", fill_opacity=0.10)
svg.text(681, Y + 44, "更新 π_{θ}", size=13, anchor="middle", color="blue", weight="600")
svg.text(681, Y + 64, "裁剪比值 × Â", size=12, anchor="middle")
svg.text(681, Y + 82, "− β·KL(π_{θ}‖π_{ref})", size=12, anchor="middle")
svg.text(681, Y + 104, "（式 10.32）", size=11, anchor="middle", color="muted")
svg.path("M681 %d C 681 %d, 214 %d, 214 %d" % (Y + 130, Y + 162, Y + 162, Y + 138), stroke="muted", width=1.3,
         dash="4 3", arrow=True)
svg.text(448, Y + 178, "更新后的 π_{θ} 成为下一轮采样用的旧策略", size=11, anchor="middle", color="muted")

# ---- row 3: rewards of the two stages
RY = 352
box(svg, 16, RY, 350, 118, None, color="green", fill="green", fill_opacity=0.06)
svg.text(28, RY + 22, "阶段 1：推理对齐（代码、数学提示）", size=13, weight="600", color="green")
svg.text(28, RY + 46, "r_{i} = RM_{reasoning}(o_{i})", size=13)
svg.text(28, RY + 70, "代码偏好数据：由编译器反馈构造", size=12, color="muted")
svg.text(28, RY + 88, "数学偏好数据：由标准答案构造", size=12, color="muted")
svg.text(28, RY + 106, "数学与代码能力能在更长的训练中持续提升", size=12, color="muted")
box(svg, 386, RY, 358, 118, None, color="green", fill="green", fill_opacity=0.06)
svg.text(398, RY + 22, "阶段 2：人类偏好对齐（多奖励）", size=13, weight="600", color="green")
svg.text(398, RY + 46, "r_{i} = c_{1}·RM_{helpful} + c_{2}·RM_{safety} + c_{3}·RM_{rule}", size=13)
svg.text(398, RY + 70, "有用性、安全两个学出来的奖励模型", size=12, color="muted")
svg.text(398, RY + 88, "加一个基于规则的奖励；c_{1}、c_{2}、c_{3} 未公布", size=12, color="muted")
svg.text(398, RY + 106, "论文实验：在线 RL 明显优于离线", size=12, color="muted")
svg.text(380, 494, "学出来的奖励模型都从 V2 Chat (SFT) 初始化，用逐点（point-wise）或成对（pair-wise）损失训练", size=12, anchor="middle",
         color="ink")
svg.save("l10-grpo-stages.svg")
