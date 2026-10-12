"""Figure 18.11: V4 post-training pipeline (V4 report sec. 5.1-5.2; research notes C58-C68):
specialists per domain (SFT -> GRPO; GRM for hard-to-verify tasks; separate experts per reasoning
effort), then multi-teacher on-policy distillation (>10 teachers, full-vocabulary reverse KL),
with FP4 quantisation-aware training throughout. Schematic, no data.
Writes deepseek/figures/l18-post-training.svg
"""

from l18_svg import SVG

W, H = 760, 420
svg = SVG(W, H, "V4 的后训练流水线")

# base model
svg.box(16, 176, 120, 64, ["V4-Base", ("专家权重 FP8", {"size": 11, "color": "muted"})], color="muted",
        fill_opacity=0.12, size=13, line_gap=20)

# specialists
domains = ["数学", "代码", "智能体", "指令遵循", "……"]
SX0, SW, SH = 196, 220, 42
for n, dom in enumerate(domains):
    y = 62 + n * 54
    last = dom == "……"
    svg.box(SX0, y, SW, SH, ["专家：%s" % dom if not last else "更多领域（共十个以上）",
                              ("SFT → GRPO；三档推理强度各一版", {"size": 11, "color": "muted"})]
            if not last else ["更多领域（教师共十个以上）"],
            color="blue", fill_opacity=0.10, size=12, line_gap=17, dash="4 3" if last else None)
    svg.path("M136 208 C 166 208, 166 %d, %d %d" % (y + SH / 2, SX0 - 1, y + SH / 2), stroke="ink", width=1.3,
             arrow=True)
    # to OPD
    svg.path("M%d %d C %d %d, %d 208, 459 208" % (SX0 + SW, y + SH / 2, SX0 + SW + 30, y + SH / 2, 430),
             stroke="violet", width=1.3, arrow=True)

# reward note
svg.box(SX0 - 12, 14, SW + 24, 40, ["奖励：规则 / 测试用例；难验证任务用 GRM", ("（带评分细则，actor 兼任评委）", {"size": 11, "color": "muted"})],
        color="amber", fill_opacity=0.10, size=11, line_gap=15)
svg.text(SX0 + SW / 2, 388, "第一段：专家培养（GRPO 只在这里）", size=12, anchor="middle", color="blue", weight="700")

# OPD
svg.box(460, 130, 178, 156, ["多教师在线蒸馏", "（OPD）", ("① 学生自己采样轨迹", {"size": 11}),
                              ("② 每位教师在每个位置", {"size": 11}), ("给出全词表分布", {"size": 11}),
                              ("③ 最小化 Σ w_{i}·KL(学生 ‖ 教师 i)", {"size": 11})],
        color="violet", fill_opacity=0.10, size=13, line_gap=21)
svg.text(549, 388, "第二段：合成一个模型", size=12, anchor="middle", color="violet", weight="700")
svg.arrow(638, 208, 652, 208)
svg.box(652, 170, 100, 76, ["V4", "统一模型", ("三档推理强度", {"size": 11, "color": "muted"})], color="green",
        fill_opacity=0.12, size=13, line_gap=19)

# FP4 QAT band
svg.rect(196, 334, 552, 30, fill="orange", fill_opacity=0.12, stroke="orange", width=1.2, rx=6)
svg.text(472, 354, "贯穿后训练：FP4 量化感知训练（专家权重 + indexer 的 QK），推理与 RL 采样直接用 FP4", size=12,
         anchor="middle")
svg.text(16, 410, "OPD 用反向 KL，并在每个位置对 129,280 个词精确求和；不用“采样 token 的对数比当优势”的单样本估计。", size=11,
         color="muted")

svg.save("l18-post-training.svg")
