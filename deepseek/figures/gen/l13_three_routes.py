"""Figure 13.1: three ways to get a reasoning signal into a language model (schematic).

Route A: imitate human-written chains of thought (SFT). Route B: score every step (PRM) or search
(MCTS) with a value model. Route C: reward only the final answer with a rule-based checker (R1-Zero).
Writes deepseek/figures/l13-three-routes.svg
"""

from l13_svg import SVG

W, H = 760, 412
svg = SVG(W, H, "让模型学会推理的三条路线：模仿人写的思维链、逐步打分或搜索、只奖励最终答案")

# column headers
svg.text(16, 22, "路线", size=12, color="muted", weight="600")
svg.text(162, 22, "推理信号从哪里来", size=12, color="muted", weight="600")
svg.text(560, 22, "代价 · 在 R1 里的下场", size=12, color="muted", weight="600")
svg.line(16, 30, 744, 30, color="rule", width=1)

lanes = [
    ("muted", "路线 A", "模仿人写的思维链",
     [("人类专家", "写逐步解答"), ("SFT：逐 token", "模仿这些解答"), ("模型学会", "“像人那样写”")],
     ["标注贵、难扩展；", "上限受人类写法约束"], "只用于数千条冷启动（§5）", "amber"),
    ("violet", "路线 B", "逐步打分或搜索",
     [("模型写出", "一个推理步骤"), ("PRM / 价值模型", "给这一步打分"), ("保留高分步骤", "继续往下展开")],
     ["“一步”难定义、逐步标注难；", "打分模型会被钻空子"], "试过，放弃（§7）", "red"),
    ("blue", "路线 C", "只奖励最终答案",
     [("题目 +", "标准答案"), ("模型写出", "完整回答"), ("规则核对", "给 1 或 0"), ("GRPO", "更新策略")],
     ["要有可自动验证的题目", "与足够的 RL 算力"], "R1-Zero 的全部配方（§2–4）", "green"),
]

y = 40
LH = 108
for color, tag, name, boxes, cost, fate, fate_col in lanes:
    svg.rect(16, y, 728, LH, fill=color, fill_opacity=0.05, stroke="rule", width=1, rx=10)
    svg.text(28, y + 44, tag, size=14, color=color if color != "muted" else "ink", weight="700")
    svg.text(28, y + 66, name, size=13, color="ink")
    n = len(boxes)
    x0, x1 = 162, 540
    gap = 20
    bw = (x1 - x0 - gap * (n - 1)) / n
    bh = 56
    by = y + (LH - bh) / 2 if color != "blue" else y + 12
    for i, (l1, l2) in enumerate(boxes):
        bx = x0 + i * (bw + gap)
        bcol = color
        if color == "blue" and i == 2:
            bcol = "green"
        svg.rect(bx, by, bw, bh, fill="surface", stroke=bcol, width=1.4, rx=8)
        svg.rect(bx, by, bw, bh, fill=bcol, fill_opacity=0.10, rx=8)
        svg.text(bx + bw / 2, by + 24, l1, size=12.5, anchor="middle", color="ink", weight="600")
        svg.text(bx + bw / 2, by + 43, l2, size=12, anchor="middle", color="ink")
        if i < n - 1:
            svg.arrow(bx + bw + 2, by + bh / 2, bx + bw + gap - 2, by + bh / 2, color="ink", width=1.4)
    if color == "blue":
        # feedback loop from the update back to the model
        bx_last = x0 + (n - 1) * (bw + gap) + bw / 2
        bx_model = x0 + 1 * (bw + gap) + bw / 2
        svg.path("M%.1f %.1f V%.1f H%.1f V%.1f" % (bx_last, by + bh + 2, by + bh + 12, bx_model, by + bh + 4),
                 stroke="blue", width=1.3, arrow=True)
        svg.text((bx_last + bx_model) / 2, by + bh + 25, "同一道题采 16 个回答，反复迭代", size=12,
                 anchor="middle", color="blue")
    svg.text(560, y + 36, cost[0], size=12, color="ink")
    svg.text(560, y + 54, cost[1], size=12, color="ink")
    svg.rect(556, y + 66, 184, 26, fill=fate_col, fill_opacity=0.10, stroke=fate_col, width=1, rx=13)
    svg.text(648, y + 84, fate, size=12, anchor="middle", color=fate_col, weight="600")
    y += LH + 10

svg.text(16, H - 8, "三条路线都从同一个预训练模型出发；区别只在“训练信号”由谁、在哪个粒度上给出。", size=12, color="muted")
svg.save("l13-three-routes.svg")
