"""Figure 2.15: fine-tuning versus zero-, one- and few-shot in-context learning.

Redrawn after GPT-3 (Brown et al. 2020) figure 2.1 with a different example (English -> Chinese words).
Fine-tuning (GPT-1 style: batch 32, 3 epochs usually enough) updates the weights on labelled examples; the three
in-context settings only put the task description and K examples into the prompt, and run a forward
pass with the weights frozen. K is typically 10 to 100, limited by the 2048-token context (section 2).
Schematic; no data.
Writes deepseek/figures/l02-icl.svg
"""

from l02_svg import SVG

W, H = 720, 360
svg = SVG(W, H, "微调与零样本、单样本、少样本的上下文学习")

PW, GAP, X0 = 166, 12, 12
panels = [
    ("微调（GPT-1 的用法）", "red", [("apple → 苹果", "green"), ("⤷ 梯度更新 θ", "red"), ("book → 书", "green"),
                                ("⤷ 梯度更新 θ", "red"), ("… 更多带标签的例子 …", "muted"), ("测试：cat →", "blue")]),
    ("零样本 zero-shot", "blue", [("英译中：", "muted"), ("cat →", "blue")]),
    ("单样本 one-shot", "blue", [("英译中：", "muted"), ("apple → 苹果", "green"), ("cat →", "blue")]),
    ("少样本 few-shot", "blue", [("英译中：", "muted"), ("apple → 苹果", "green"), ("book → 书", "green"),
                               ("water → 水", "green"), ("…（K 个例子）", "muted"), ("cat →", "blue")]),
]
for i, (title, col, rows) in enumerate(panels):
    x = X0 + i * (PW + GAP)
    svg.rect(x, 14, PW, 290, fill="surface", fill_opacity=1, stroke="rule", width=1.2, rx=8)
    svg.text(x + PW / 2, 38, title, size=13, anchor="middle", color=col, weight="600")
    if i > 0:
        svg.rect(x + 12, 52, PW - 24, len(rows) * 30 + 14, fill="blue", fill_opacity=0.05, stroke="blue", width=1,
                 rx=6, dash="4 3")
        svg.text(x + PW - 16, 66, "提示", size=11, anchor="end", color="blue")
    for j, (txt, c) in enumerate(rows):
        y = 78 + j * 30
        svg.text(x + 22, y, txt, size=13 if c != "muted" else 12, color=c, weight="600" if c == "blue" else None)
    # footer
    if i == 0:
        svg.text(x + PW / 2, 262, "用带标签的例子改权重", size=12, anchor="middle", color="red")
        svg.text(x + PW / 2, 280, "GPT-1：batch 32，约 3 遍", size=12, anchor="middle", color="muted")
    else:
        svg.text(x + PW / 2, 262, "权重不变，只做前向", size=12, anchor="middle", color="green")
        svg.text(x + PW / 2, 280, "例子放在上下文里", size=12, anchor="middle", color="muted")

svg.line(X0 + PW + GAP / 2, 14, X0 + PW + GAP / 2, 304, color="ink", width=1.2, dash="4 4")
svg.text(X0, 330, "左：GPT-1 的“预训练 + 微调”。右三栏：GPT-3 的上下文学习，模型读完整个提示，", size=12, color="ink")
svg.text(X0, 348, "接着预测 “cat →” 后面的 token；K 一般取 10 到 100，受 2048 的上下文长度限制。", size=12, color="ink")

svg.save("l02-icl.svg")
