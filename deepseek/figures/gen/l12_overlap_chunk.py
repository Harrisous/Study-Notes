"""Figure 12.12: overlapping one forward chunk with one backward chunk
(redrawn after DeepSeek-V3 report Figure 4).

Component order is read from the paper's figure (research notes C21); the relative widths are
the pixel widths of the original figure rendered at 1900 px (notes 5.2), i.e. illustrative, not
measured times. Triangle up-open = forward chunk, filled = backward chunk.
    compute:  MLP(B) MLP(W) MLP(F) | ATTN(B) ATTN(W) ATTN(F) |
    comm:     DISPATCH(F) DISPATCH(B) | COMBINE(F) PP COMBINE(B) |
Writes deepseek/figures/l12-overlap-chunk.svg
"""

from l12_svg import SVG

comp = [("MLP(B)", "B", 170), ("MLP(W)", "W", 95), ("MLP(F)", "F", 182), ("|", "bar", 10),
        ("ATTN(B)", "B", 303), ("ATTN(W)", "W", 107), ("ATTN(F)", "F", 150), ("|", "bar", 10)]
comm = [("DISPATCH(F)", "F", 192), ("DISPATCH(B)", "B", 255), ("|", "bar", 10), ("COMBINE(F)", "F", 268),
        ("PP", "PP", 58), ("COMBINE(B)", "B", 234), ("|", "bar", 10)]
tot = sum(w for *_, w in comp)
assert tot == sum(w for *_, w in comm) == 1027

COL = {"F": "blue", "B": "green", "W": "violet", "PP": "amber", "bar": "red"}
W, H = 760, 300
X0, X1 = 118, 740
k = (X1 - X0) / tot

svg = SVG(W, H, "一对前向 chunk 与反向 chunk 的重叠：每段通信都被另一个 chunk 的计算盖住")


def row(items, y, h, label, sub, comm_row):
    svg.text(X0 - 10, y + h / 2 - 2, label, size=13, anchor="end", weight="600")
    svg.text(X0 - 10, y + h / 2 + 15, sub, size=11, anchor="end", color="muted")
    x = X0
    for name, kind, w in items:
        ww = w * k
        col = COL[kind]
        if kind == "bar":
            svg.rect(x, y - 4, ww, h + 8, fill="red", fill_opacity=0.9)
        else:
            svg.rect(x + 1, y, ww - 2, h, fill=col, fill_opacity=0.12 if comm_row else 0.24, stroke=col,
                     width=1.3, dash="5 3" if comm_row else None, rx=3)
            tri = "△" if kind == "F" else ("" if kind == "PP" else "▲")
            label = name + tri
            # about 8 units per Latin glyph at size 12 bold; narrow boxes get the label on two lines
            if 8.2 * len(label) > ww - 8 and "(" in name:
                head, tail = name.split("(", 1)
                svg.text(x + ww / 2, y + h / 2 - 2, head, size=12, anchor="middle", color=col, weight="600")
                svg.text(x + ww / 2, y + h / 2 + 13, "(" + tail + tri, size=12, anchor="middle", color=col,
                         weight="600")
            else:
                svg.text(x + ww / 2, y + h / 2 + 5, label, size=12, anchor="middle", color=col, weight="600")
        x += ww


row(comp, 74, 40, "计算", "大部分 SM", False)
row(comm, 146, 40, "通信", "少数 SM", True)
svg.text(X0, 30, "时间 →（宽度按原图比例，示意，不是实测）", size=12, color="muted")
svg.line(X0, 38, X1, 38, color="muted", width=1, arrow=True)
# legend
ly = 228
items = [("blue", "前向 chunk △"), ("green", "反向：对输入求梯度 ▲"), ("violet", "反向：对权重求梯度 ▲"),
         ("amber", "PP 通信"), ("red", "屏障（同步点）")]
x = 16
for col, lab in items:
    svg.rect(x, ly - 11, 14, 14, fill=col, fill_opacity=0.3, stroke=col, width=1)
    svg.text(x + 20, ly + 1, lab, size=12)
    x += 34 + 12 * len(lab)
svg.text(16, ly + 30, "实线框 = 计算，虚线框 = 通信（dispatch：把 token 发给专家；combine：收回专家输出）；计算与通信的 SM 比例手动调。",
         size=11, color="muted")
svg.text(16, ly + 50, "每个 chunk 拆成 attention、dispatch、MLP、combine 四块；反向的 attention 与 MLP 再拆成“对输入”“对权重”两半。据 V3 报告图 4 重绘。",
         size=11, color="muted")
svg.save("l12-overlap-chunk.svg")
