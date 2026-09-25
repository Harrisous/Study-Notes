"""Figure 15.2: V3.1 hybrid thinking - one model, two chat-template prefixes.

Template strings from the DeepSeek-V3.1 model card ("Chat Template"); token ids 128798 / 128799
from V3.1 tokenizer.json (they were <|place_holder_no_798|> / 799 in V3). Special tokens are
abbreviated in the figure: <bos> = <｜begin▁of▁sentence｜>, <User> = <｜User｜>, <Assistant> = <｜Assistant｜>,
<eos> = <｜end▁of▁sentence｜>. Schematic; box widths follow the label length only.
Writes deepseek/figures/l15-hybrid-template.svg
"""

from l15_svg import SVG

W, H = 760, 400
svg = SVG(W, H, "V3.1 的混合思考：同一组参数，靠提示的最后一个 token 切换模式")

SIZE = 12
KIND = {
    "sp": ("ink", 0.06),       # special tokens of the template
    "txt": ("muted", 0.10),    # user-supplied text
    "mode": ("violet", 0.16),  # the mode-switch token
    "think": ("blue", 0.14),   # generated reasoning
    "ans": ("green", 0.14),    # generated answer
}


def width(s):
    w = 0.0
    for ch in s:
        if ord(ch) > 0x2E80:
            w += SIZE
        elif ch in "‹›":
            w += SIZE * 0.45
        else:
            w += SIZE * 0.6
    return w + 14


def row(x, y, items, gap=4, h=26):
    """items: (label, kind, note). Returns list of (x0, x1) per item."""
    spans = []
    for lab, kind, note in items:
        if kind == "cut":
            svg.line(x + 3, y - 8, x + 3, y + h + 8, color="ink", width=1.4, dash="4 3")
            spans.append((x, x + 6))
            x += 10
            continue
        col, op = KIND[kind]
        w = width(lab.replace("_{", "").replace("}", "")) + (4 if "_{" in lab else 0)
        svg.rect(x, y, w, h, fill=col, fill_opacity=op, stroke=col, width=1.6 if kind == "mode" else 1, rx=4)
        svg.text(x + w / 2, y + 17.5, lab, size=SIZE, anchor="middle",
                 color="violet" if kind == "mode" else "ink", weight="700" if kind == "mode" else None)
        if note:
            svg.text(x + w / 2, y + h + 14, note, size=11, anchor="middle", color="muted")
        spans.append((x, x + w))
        x += w + gap
    return spans


X0 = 16
# ---------------------------------------------------------------- A: non-thinking
svg.text(X0, 22, "① 非思考模式（首轮）：提示以 </think> 结尾，模型直接写回答", size=13, weight="700")
row(X0, 34, [("‹bos›", "sp", None), ("系统提示", "txt", None), ("‹User›", "sp", None), ("问题", "txt", None),
             ("‹Assistant›", "sp", None), ("</think>", "mode", "id 128799"), ("", "cut", None),
             ("回答", "ans", None), ("‹eos›", "sp", None)])

# ---------------------------------------------------------------- B: thinking
svg.text(X0, 112, "② 思考模式（首轮）：提示以 <think> 结尾，模型先写思考，自己写出 </think>，再写回答", size=13, weight="700")
row(X0, 124, [("‹bos›", "sp", None), ("系统提示", "txt", None), ("‹User›", "sp", None), ("问题", "txt", None),
              ("‹Assistant›", "sp", None), ("<think>", "mode", "id 128798"), ("", "cut", None),
              ("思考内容", "think", None), ("</think>", "sp", None), ("回答", "ans", None), ("‹eos›", "sp", None)])

# ---------------------------------------------------------------- C: multi-turn
svg.text(X0, 202, "③ 多轮：历史轮次一律写成 </think>回答‹eos›，思考内容被删掉，</think> 保留", size=13, weight="700")
sp = row(X0, 250, [("‹bos›", "sp", None), ("系统提示", "txt", None), ("‹User›", "sp", None), ("问题_{1}", "txt", None),
                   ("‹Assistant›", "sp", None), ("</think>", "sp", None), ("回答_{1}", "ans", None), ("‹eos›", "sp", None)])
# the deleted reasoning of turn 1, floating above the history
ax0, ax1 = sp[4][1], sp[5][0]
bx, bw = sp[5][0] - 44, 212
svg.rect(bx, 214, bw, 24, fill="red", fill_opacity=0.06, stroke="red", width=1.2, rx=4, dash="4 3")
svg.text(bx + bw / 2, 230.5, "<think> 第 1 轮的思考（已删除）", size=12, anchor="middle", color="red")
svg.line(bx + 8, 226, bx + bw - 8, 226, color="red", width=1.2)
svg.arrow(bx + 20, 238, (ax0 + ax1) / 2, 250, color="red", width=1.2)
svg.text(sp[7][1] + 10, 267, "← 第 1 轮（历史）", size=12, color="muted")

sp2 = row(X0, 300, [("‹User›", "sp", None), ("问题_{2}", "txt", None), ("‹Assistant›", "sp", None),
                    ("<think>", "mode", "或 </think>"), ("", "cut", None), ("思考内容_{2}", "think", None),
                    ("</think>", "sp", None), ("回答_{2}", "ans", None), ("‹eos›", "sp", None)])
svg.text(sp2[-1][1] + 10, 317, "← 第 2 轮（当前）", size=12, color="muted")

# ---------------------------------------------------------------- legend
LY = 370
items = [("sp", "模板的特殊 token"), ("txt", "用户给的文本"), ("mode", "切换模式的 token"),
         ("think", "模型生成：思考"), ("ans", "模型生成：回答")]
x = X0
for kind, lab in items:
    col, op = KIND[kind]
    svg.rect(x, LY - 11, 16, 14, fill=col, fill_opacity=op, stroke=col, width=1, rx=3)
    svg.text(x + 22, LY, lab, size=12, color="ink")
    x += 34 + width(lab) - 14
svg.line(x + 4, LY - 13, x + 4, LY + 3, color="ink", width=1.4, dash="4 3")
svg.text(x + 12, LY, "提示到此为止", size=12)

svg.save("l15-hybrid-template.svg")
