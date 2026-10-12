"""Figure 15.11: thinking retention in tool-calling scenarios (redrawn after V3.2 paper, Figure 4).

Rules (V3.2 paper section 3.2.1; encoding_dsv32.py drop_thinking_messages):
  - reasoning content is discarded only when a new user message arrives;
  - while only tool messages are appended, the reasoning content is kept;
  - when reasoning is removed, tool calls and their results stay in the context.
Schematic; block sizes carry no length information.
Writes deepseek/figures/l15-thinking-retention.svg
"""

from l15_svg import SVG

KIND = {
    "tools": ("muted", 0.10, "工具列表"),
    "user": ("ink", 0.06, None),
    "think": ("blue", 0.20, None),
    "call": ("violet", 0.14, None),
    "result": ("amber", 0.14, None),
    "answer": ("green", 0.16, None),
    "drop": ("red", 0.05, None),
}

COLS = [
    ("第 1 轮 · 第 1 步", [("tools", "工具列表"), ("user", "User 1")],
     [("think", "思考 1.1"), ("call", "工具调用 1.1")]),
    ("第 1 轮 · 第 2 步", [("tools", "工具列表"), ("user", "User 1"), ("think", "思考 1.1"), ("call", "工具调用 1.1"),
                        ("result", "工具结果 1.1")],
     [("think", "思考 1.2"), ("call", "工具调用 1.2")]),
    ("第 1 轮 · 第 3 步", [("tools", "工具列表"), ("user", "User 1"), ("think", "思考 1.1"), ("call", "工具调用 1.1"),
                        ("result", "工具结果 1.1"), ("think", "思考 1.2"), ("call", "工具调用 1.2"),
                        ("result", "工具结果 1.2")],
     [("think", "思考 1.3"), ("answer", "回答 1")]),
    ("第 2 轮 · 第 1 步", [("tools", "工具列表"), ("user", "User 1"), ("drop", "思考 1.1（删）"), ("call", "工具调用 1.1"),
                        ("result", "工具结果 1.1"), ("drop", "思考 1.2（删）"), ("call", "工具调用 1.2"),
                        ("result", "工具结果 1.2"), ("drop", "思考 1.3（删）"), ("answer", "回答 1"),
                        ("user", "User 2")],
     [("think", "思考 2.1"), ("call", "……")]),
]

W, H = 760, 490
svg = SVG(W, H, "工具调用中的思考保留：只有新的 user 消息才会删掉之前的思考")

CX, CW, GAP = 12, 172, 12
BH, STEP = 21, 25
TOP = 58
for c, (title, inp, out) in enumerate(COLS):
    x = CX + c * (CW + GAP)
    svg.text(x + CW / 2, 20, title, size=13, anchor="middle", weight="700")
    svg.text(x + 6, 44, "输入（上下文）", size=11, color="muted")
    y = TOP
    for kind, lab in inp:
        col, op, _ = KIND[kind]
        svg.rect(x + 6, y, CW - 12, BH, fill=col, fill_opacity=op, stroke=col, width=1.1, rx=4,
                 dash="4 3" if kind == "drop" else None)
        svg.text(x + CW / 2, y + 15, lab, size=12, anchor="middle", color="red" if kind == "drop" else "ink",
                 weight="700" if (c == 3 and lab == "User 2") else None)
        if kind == "drop":
            svg.line(x + 30, y + 11, x + CW - 30, y + 11, color="red", width=1)
        y += STEP
    yb = TOP + 11 * STEP + 6
    svg.line(x + 6, yb, x + CW - 6, yb, color="rule", width=1.2)
    svg.text(x + 6, yb + 18, "输出（本步生成）", size=11, color="muted")
    y = yb + 26
    for kind, lab in out:
        col, op, _ = KIND[kind]
        svg.rect(x + 6, y, CW - 12, BH, fill=col, fill_opacity=op + 0.1, stroke=col, width=1.5, rx=4)
        svg.text(x + CW / 2, y + 15, lab, size=12, anchor="middle", weight="700")
        y += STEP
    if c < 3:
        ax = x + CW + GAP / 2
        svg.arrow(ax - 5, 150, ax + 5, 150, color="ink", width=1.4)

# annotations

LY = 452
x = 12
for kind, lab in [("user", "用户消息"), ("think", "思考"), ("call", "工具调用"), ("result", "工具结果"), ("answer", "回答"),
                  ("drop", "被删除的思考"), ("tools", "工具列表（系统提示）")]:
    col, op, _ = KIND[kind]
    svg.rect(x, LY - 11, 16, 14, fill=col, fill_opacity=op + 0.05, stroke=col, width=1.1, rx=3,
             dash="3 2" if kind == "drop" else None)
    svg.text(x + 22, LY, lab, size=12)
    x += 34 + 12.4 * len(lab)
svg.text(12, 480, "第 2 轮开始时来了新的 User 2：此前三段思考全部删掉，工具调用、工具结果与回答 1 都留下。据 DeepSeek-V3.2 论文图 4 重绘。",
         size=11, color="muted")

svg.save("l15-thinking-retention.svg")
