"""Figure 13.11: which reward is used where - R1-Zero RL, R1 stage-1 RL and R1 stage-2 RL.

Sources (reference-docs/lesson-13-doc.md): C4-C6, C22, C24, section 2.3 (Eqs. 4-10 of arXiv v2),
section 3.1-3.3 and 3.6 (reward models: helpful pairwise 66,000 pairs, safety point-wise 106,000).
Writes deepseek/figures/l13-rewards-by-stage.svg
"""

from l13_svg import SVG

W, H = 760, 436
svg = SVG(W, H, "R1-Zero 与 R1 两个 RL 阶段分别用了哪些奖励")

COLX = [176, 364, 552]
CW = 184
ROWS = [(64, 78), (150, 80), (238, 80), (326, 84)]      # (y, height)


def chip(x, y, w, text, color, h=24):
    svg.rect(x, y, w, h, fill=color, fill_opacity=0.14, stroke=color, width=1.2, rx=12)
    svg.text(x + w / 2, y + 16.5, text, size=12, anchor="middle", color="ink", weight="600")


def plus(x, y):
    svg.text(x, y + 16.5, "+", size=14, anchor="middle", color="muted", weight="700")


heads = [("R1-Zero 的 RL", "从 V3-Base 直接开始"), ("R1 第一阶段 RL", "推理导向，得到 Dev2"),
         ("R1 第二阶段 RL", "全场景，得到 R1")]
for x, (a, b) in zip(COLX, heads):
    svg.text(x + CW / 2, 24, a, size=13.5, anchor="middle", weight="700")
    svg.text(x + CW / 2, 43, b, size=12, anchor="middle", color="muted")
svg.line(16, 54, 744, 54, color="rule", width=1)

labels = [("推理题", "数学 · 代码 · STEM · 逻辑"), ("一般题：有用性", "66K 条 RL 提示"),
          ("一般题：无害性", "12,000 条 RL 提示"), ("训练设置", "Nature 版 Methods")]
for (y, h), (a, b) in zip(ROWS, labels):
    svg.rect(16, y, 728, h, fill="grid", fill_opacity=0.35, rx=8)
    svg.text(26, y + h / 2 - 3, a, size=13, weight="600")
    svg.text(26, y + h / 2 + 15, b, size=12, color="muted")

# row 1: reasoning prompts
y, h = ROWS[0]
x = COLX[0]
chip(x, y + 10, 84, "准确性 0/1", "green"); plus(x + 90, y + 10); chip(x + 98, y + 10, 76, "格式 0/1", "violet")
svg.text(x + CW / 2, y + 60, "等权相加（式 13.9）", size=12, anchor="middle", color="muted")
for x in COLX[1:]:
    chip(x, y + 10, 84, "规则奖励", "green"); plus(x + 90, y + 10); chip(x + 98, y + 10, 82, "语言一致性", "blue")
    svg.text(x + CW / 2, y + 60, "直接相加", size=12, anchor="middle", color="muted")

# rows 2-3, R1-Zero: not trained
y2, _ = ROWS[1]
y3, h3 = ROWS[2]
svg.rect(COLX[0], y2 + 8, CW, y3 + h3 - y2 - 16, fill="surface", stroke="rule", width=1, rx=8, dash="4 4")
svg.text(COLX[0] + CW / 2, (y2 + y3 + h3) / 2 - 4, "不训练一般题", size=12.5, anchor="middle", color="muted")
svg.text(COLX[0] + CW / 2, (y2 + y3 + h3) / 2 + 14, "（只有可自动核对的题）", size=12, anchor="middle", color="muted")
# rows 2-3, stage 1
svg.rect(COLX[1], y2 + 8, CW, y3 + h3 - y2 - 16, fill="surface", stroke="rule", width=1, rx=8, dash="4 4")
svg.text(COLX[1] + CW / 2, (y2 + y3 + h3) / 2 - 12, "论文只说：语言一致性", size=12, anchor="middle", color="muted")
svg.text(COLX[1] + CW / 2, (y2 + y3 + h3) / 2 + 6, "奖励也加在非推理数据上，", size=12, anchor="middle", color="muted")
svg.text(COLX[1] + CW / 2, (y2 + y3 + h3) / 2 + 24, "其余未详述", size=12, anchor="middle", color="muted")
# rows 2-3, stage 2
x = COLX[2]
y, h = ROWS[1]
chip(x, y + 8, CW, "有用性 RM：只看最终总结", "amber")
svg.text(x + CW / 2, y + 51, "+ 格式 + 语言一致性", size=12, anchor="middle", color="ink")
svg.text(x + CW / 2, y + 69, "成对训练，66,000 对偏好", size=12, anchor="middle", color="muted")
y, h = ROWS[2]
chip(x, y + 8, CW, "安全 RM：看推理与总结全文", "red")
svg.text(x + CW / 2, y + 51, "+ 格式 + 语言一致性", size=12, anchor="middle", color="ink")
svg.text(x + CW / 2, y + 69, "逐点训练，106,000 条", size=12, anchor="middle", color="muted")

# row 4: settings
y, h = ROWS[3]
sets = [["温度 T = 1，G = 16", "10,400 步", "ε 未公开"],
        ["温度 T = 1，G = 16", "步数未公开（训到推理收敛）", "ε = 10"],
        ["温度 T = 0.7", "1,700 步；偏好奖励只在", "最后 400 步加入"]]
for x, lines in zip(COLX, sets):
    for k, s in enumerate(lines):
        svg.text(x + CW / 2, y + 24 + 19 * k, s, size=12, anchor="middle")

svg.text(16, H - 6, "共同设置：lr 3×10^{-6}，KL 系数 0.001，每步 32 题 × 16 个回答，每 400 步重置参考模型（第二阶段“沿用第一阶段的大部分参数”）",
         size=12, color="muted")
svg.save("l13-rewards-by-stage.svg")
