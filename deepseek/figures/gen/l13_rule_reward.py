"""Figure 13.2: the R1-Zero training loop - template, rule-based rewards, group normalisation, GRPO.

Numbers (lr 3e-6, beta 0.001, G = 16, 32 questions/step, 8,192 outputs per rollout split into 16
mini-batches, reference reset every 400 steps, 10,400 steps, 32,768 -> 65,536 at 8.2k) are from the
R1 paper (Nature Methods / arXiv v2 section 2.1), as recorded in reference-docs/lesson-13-doc.md C8.
Writes deepseek/figures/l13-rule-reward.svg
"""

from l13_svg import SVG

W, H = 760, 486
svg = SVG(W, H, "R1-Zero 的训练回路：模板、两条规则奖励、组内标准化与 GRPO 更新")

# ---- row 1: template -> policy -> response
svg.rect(16, 16, 300, 150, fill="surface", stroke="blue", width=1.4, rx=10)
svg.rect(16, 16, 300, 150, fill="blue", fill_opacity=0.06, rx=10)
svg.text(28, 36, "训练模板（据论文表 1 意译）", size=12, color="blue", weight="600")
svg.text(28, 58, "用户与助手对话。助手先在心里思考推理", size=12)
svg.text(28, 76, "过程，再给出答案；两者分别放进标签：", size=12)
svg.mono(28, 100, "<think> 推理过程 </think>", size=12, color="ink")
svg.mono(28, 120, "<answer> 答案 </answer>", size=12, color="ink")
svg.text(28, 146, "User: ", size=12.5)
svg.rect(68, 132, 58, 20, fill="blue", fill_opacity=0.16, rx=4)
svg.text(97, 146, "{题目}", size=12.5, anchor="middle", color="blue", weight="600")
svg.text(134, 146, "Assistant:", size=12.5)

svg.arrow(318, 91, 334, 91, color="ink", width=1.5)
svg.box(336, 50, 112, 82, color="blue", lines=["策略 π_{θ}", "从 V3-Base 起步", "温度 T = 1"], size=14, sub_size=12)
svg.arrow(450, 91, 464, 91, color="ink", width=1.5)

svg.rect(466, 16, 278, 150, fill="surface", stroke="ink", width=1.2, rx=10)
svg.text(478, 36, "一个回答 o_{i}（示意）", size=12, color="muted", weight="600")
svg.mono(478, 58, "<think>", size=12, color="violet")
svg.text(490, 77, "两边平方，整理成四次方程……", size=12)
svg.text(490, 96, "等等，平方可能引入增根，逐个检验……", size=12)
svg.mono(478, 115, "</think>", size=12, color="violet")
svg.mono(478, 140, "<answer> \\boxed{…} </answer>", size=12, color="green")

# ---- row 2: checkers -> sum -> normalise -> update
svg.arrow(590, 168, 590, 196, color="ink", width=1.4)
svg.box(430, 198, 290, 66, color="green",
        lines=["准确性奖励 r_{acc} ∈ {0, 1}", "数学：\\boxed{} 中的答案等于参考答案？", "代码：编译并跑通预设的测试用例？"],
        size=13, sub_size=12, line_gap=18)
svg.path("M736 168 V299 H722", stroke="ink", width=1.4, arrow=True)
svg.box(430, 276, 290, 46, color="violet",
        lines=["格式奖励 r_{format} ∈ {0, 1}", "思考是否放在 <think> 与 </think> 之间？"], size=13, sub_size=12,
        line_gap=18)

svg.circle(390, 262, 15, fill="surface", stroke="ink", width=1.5)
svg.text(390, 268, "+", size=18, anchor="middle", weight="700")
svg.arrow(428, 231, 405, 254, color="green", width=1.4)
svg.arrow(428, 292, 403, 270, color="violet", width=1.4)
svg.text(196, 324, "两条奖励等权相加：r_{i} = r_{acc} + r_{format}", size=12, color="ink")

svg.arrow(374, 262, 350, 262, color="ink", width=1.4)
svg.box(196, 224, 152, 76, color="ink",
        lines=["组内标准化", "Â_{i} = (r_{i} − 均值) / 标准差", "G = 16 个回答一组"], size=13, sub_size=12,
        line_gap=19, fill_opacity=0.05)
svg.arrow(194, 262, 178, 262, color="ink", width=1.4)
svg.box(16, 224, 160, 76, color="blue",
        lines=["GRPO 更新 θ", "lr = 3×10^{-6}，β = 0.001", "每步 32 题 × 16 = 512 条"], size=13, sub_size=12,
        line_gap=19)
svg.path("M96 222 V190 H392 V134", stroke="blue", width=1.5, arrow=True)
svg.text(244, 184, "更新后的策略继续采样", size=12, anchor="middle", color="blue")

# ---- row 3: what is not used, and the schedule
svg.rect(16, 354, 360, 116, fill="red", fill_opacity=0.05, stroke="red", width=1.2, rx=10, dash="5 4")
svg.cross(34, 374, s=6)
svg.text(48, 379, "不用神经奖励模型（结果型或过程型都不用）", size=13, color="red", weight="600")
svg.text(30, 405, "① 大规模 RL 中会被 reward hacking（钻空子）", size=12)
svg.text(30, 427, "② 重新训练奖励模型要额外的训练资源", size=12)
svg.text(30, 449, "③ 让整条训练流程更复杂", size=12)

svg.rect(392, 354, 352, 116, fill="amber", fill_opacity=0.06, stroke="amber", width=1.2, rx=10)
svg.text(406, 377, "训练日程（Nature 版 Methods 首次公开）", size=13, color="amber", weight="600")
svg.text(406, 403, "每次 rollout 8,192 条回答 → 16 个小批 → 16 步", size=12)
svg.text(406, 425, "每 400 步把参考模型 π_{ref} 换成最新策略", size=12)
svg.text(406, 447, "共 10,400 步；8.2k 步起最大长度 32,768 → 65,536", size=12)

svg.save("l13-rule-reward.svg")
