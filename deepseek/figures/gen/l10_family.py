"""Figure 10.14: the DeepSeek-V2 family -- how V2, V2-Lite, DeepSeek-Coder-V2 and V2.5 are related.

Facts (lesson-10-doc C34-C42, sec. 3.1-3.2): V2 pre-trained on 8.1T tokens, YaRN 32K x 576 x 1000 steps,
SFT, two-stage GRPO; V2 released 2024-05-06, V2-Lite 2024-05-16 (from scratch, 5.7T, 15.7B/2.4B, SFT only);
Coder-V2 continues from an intermediate V2 checkpoint trained on 4.2T with +6T (code 60 %, math 10 %,
natural language 30 %), YaRN two stages 32K x 1152 x 1000 and 128K x 288 x 1000, SFT 300M tokens, GRPO
on ~40k prompts (arXiv 2024-06-17); V2.5 (2024-09-05) merges V2-0628 and Coder-V2-0724. Whether the
Coder-V2-Lite checkpoint comes from V2-Lite is not stated (drawn dashed).
Writes deepseek/figures/l10-family.svg
"""

from l10_svg import SVG, box

W, H = 760, 470
svg = SVG(W, H, "DeepSeek-V2 家族：谁从谁那里来")

# ---- V2 main line: pre-training bar with token scale
BX0, BX1, BY = 30, 330, 70
svg.text(BX0, 26, "DeepSeek-V2 预训练（236B / 21B）", size=13, weight="600")
svg.rect(BX0, BY - 14, BX1 - BX0, 28, fill="blue", fill_opacity=0.14, stroke="blue", width=1.2, rx=4)
for t in (0, 2, 4, 6, 8.1):
    x = BX0 + t / 8.1 * (BX1 - BX0)
    svg.line(x, BY + 14, x, BY + 20, color="muted", width=1)
    if t != 4:
        svg.text(x, BY + 33, ("%gT" % t), size=11, anchor="middle", color="muted")
x42 = BX0 + 4.2 / 8.1 * (BX1 - BX0)
svg.line(x42, BY - 20, x42, BY + 14, color="orange", width=2)
svg.text(x42, BY - 24, "4.2T 中间检查点", size=11, anchor="middle", color="orange", weight="600")
svg.text((BX0 + BX1) / 2 + 60, BY + 5, "8.1T tokens，4K", size=12, anchor="middle")
# after pre-training
steps = [("YaRN 续训", "32K × 576 × 1000 步"), ("SFT", "150 万条"), ("两阶段 GRPO", "推理 → 偏好")]
x = 350
for a, b in steps:
    svg.arrow(x - 18, BY, x - 2, BY)
    box(svg, x, BY - 24, 116, 48, a, color="blue", size=12, weight="600", sub=b, sub_size=11)
    x += 134
svg.text(730, 20, "V2 Chat：2024-05-06", size=12, anchor="end", color="blue", weight="600")

# ---- Coder-V2 branch
CY = 200
svg.path("M%.1f %d C %.1f %d, 60 %d, 60 %d" % (x42, BY + 14, x42, CY - 60, CY - 60, CY - 26), stroke="orange",
         width=1.8, arrow=True)
svg.text(206, CY - 44, "DeepSeek-Coder-V2（236B / 21B）：从 V2 的中间检查点续训", size=13, weight="600", color="orange")
csteps = [("续训 +6T", "代码 60% · 数学 10% · 文本 30%"), ("长上下文两阶段", "32K×1152、128K×288 各 1000 步"),
          ("SFT", "300M tokens"), ("GRPO", "约 4 万个提示")]
x = 20
for k, (a, b) in enumerate(csteps):
    w = 190 if k < 2 else 120
    box(svg, x, CY - 24, w, 48, a, color="orange", size=12, weight="600", sub=b, sub_size=11)
    if k < len(csteps) - 1:
        svg.arrow(x + w + 2, CY, x + w + 16, CY)
    x += w + 18
svg.text(20, CY + 44, "总曝光 4.2T + 6T = 10.2T tokens；86 → 338 种编程语言；16K → 128K 上下文；arXiv 2024-06-17", size=12,
         color="muted")

# ---- V2-Lite line
LY = 318
svg.text(20, LY - 40, "DeepSeek-V2-Lite（15.7B / 2.4B）：与 V2 同一份语料，从零训练", size=13, weight="600", color="violet")
lsteps = [("从零预训练", "5.7T tokens，4K"), ("长上下文扩展", "到 32K"), ("SFT", "没有 RL")]
x = 20
for k, (a, b) in enumerate(lsteps):
    box(svg, x, LY - 22, 150, 46, a, color="violet", size=12, weight="600", sub=b, sub_size=11)
    if k < len(lsteps) - 1:
        svg.arrow(x + 152, LY, x + 166, LY)
    x += 168
svg.text(520, LY + 5, "V2-Lite Chat：2024-05-16", size=12, color="violet", weight="600")
svg.text(20, LY + 44, "Coder-V2-Lite（16B / 2.4B）同样写作“4.2T + 6T”；它的起点检查点来自 V2-Lite 还是 V2，原文没有说明", size=12,
         color="muted")

# ---- merge into V2.5
MY = 420
box(svg, 250, MY - 22, 260, 46, "DeepSeek-V2.5（2024-09-05）", color="green", fill="green", fill_opacity=0.10,
    size=13, weight="600", sub="合并 V2-0628 与 Coder-V2-0724", sub_size=11)
svg.path("M720 %d C 752 %d, 752 %d, 512 %d" % (BY + 24, BY + 60, MY + 10, MY + 8), stroke="green", width=1.6,
         dash="5 4", arrow=True)
# labels sit just left of their dashed curves (the V2 curve is at x ~734 at y 150, the Coder curve ~680 at y 250)
svg.text(726, 150, "V2-0628", size=11, anchor="end", color="green")
svg.path("M660 %d C 712 %d, 712 %d, 512 %d" % (CY + 24, CY + 60, MY - 14, MY - 8), stroke="green", width=1.6,
         dash="5 4", arrow=True)
svg.text(668, CY + 54, "Coder-V2-0724", size=11, anchor="end", color="green")
svg.save("l10-family.svg")
