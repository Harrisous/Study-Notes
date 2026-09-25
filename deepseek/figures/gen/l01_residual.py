"""Figure 1.12: the residual stream as a conveyor belt (course analogy 6; Elhage et al. 2021).

Each of the 2L = 24 sub-layers of GPT-2 small reads a copy of the stream (through its own
LayerNorm) and adds an increment back. Schematic; the counts (12 blocks, 24 writers, 768 wide)
come from the GPT-2 small config.
Writes deepseek/figures/l01-residual.svg
"""

from l01_svg import SVG

L = 12
W, H = 760, 330
s = SVG(W, H, "残差流是一条传送带：每个子层读一份、加回一个增量")

BY = 160           # belt y
s.text(10, 24, "残差流 = 嵌入 + 所有子层写回的增量之和（GPT-2 small：12 块 × 2 个子层 = 24 个“工位”）", size=13,
       weight="700")
s.line(56, BY, 712, BY, color="ink", width=5)
s.arrow(712, BY, 736, BY, width=5)
s.text(48, BY + 5, "*h*^{(0)}", size=14, anchor="end", weight="600")
s.text(742, BY - 12, "*h*^{(12)}", size=14, anchor="end", weight="600")
s.text(10, 46, "传送带上的“件”：每个位置一条 768 维向量，从左往右流；工位先经 LN 读一份，算完把增量加回（⊕）", size=12, color="muted")

stations = [(1, 128), (2, 300), (12, 560)]
for blk, x in stations:
    # attention station above the belt
    for kind, xs, up in (("注意力", x, True), ("FFN", x + 86, False)):
        col = "blue" if kind == "注意力" else "muted"
        by = BY - 92 if up else BY + 48
        s.box(xs - 30, by, 60, 42, [(kind, {"size": 12, "weight": "600"}), ("第 %d 块" % blk, {"size": 11, "color": "muted"})],
              color=col, fill_opacity=0.12, line_gap=16)
        rx_, wx = xs - 16, xs + 16
        if up:
            s.arrow(rx_, BY - 4, rx_, by + 42 + 2, color="ink", width=1.2)
            s.text(rx_ - 4, BY - 22, "LN", size=11, anchor="end", color="muted")
            s.arrow(wx, by + 42, wx, BY - 9, color=col, width=1.6)
            s.plus(wx, BY, r=7)
            s.text(wx + 5, BY - 30, "+*u*" if kind == "注意力" else "+*m*", size=12, color=col)
        else:
            s.arrow(rx_, BY + 4, rx_, by - 2, color="ink", width=1.2)
            s.text(rx_ - 4, BY + 30, "LN", size=11, anchor="end", color="muted")
            s.arrow(wx, by, wx, BY + 9, color=col, width=1.6)
            s.plus(wx, BY, r=7)
            s.text(wx + 5, BY + 42, "+*m*", size=12, color=col)
s.text(468, BY - 14, "……", size=16, anchor="middle", color="muted")
s.text(468, BY + 34, "第 3–11 块", size=11, anchor="middle", color="muted")

s.line(10, 264, 750, 264, color="rule", width=1)
s.text(10, 286, "前向：*h*^{(12)} = *h*^{(0)} + Σ_{l=1}^{12}（*u*^{(l)} + *m*^{(l)}）。每个工位只加不改：它读得到前面所有工位写下的东西，", size=12.5)
s.text(10, 308, "却擦不掉它们，除非自己写一个方向相反的增量。反向：梯度沿传送带原样流回，∂*h*^{(12)}/∂*h*^{(l)} = I + （经过工位的各项）。",
       size=12.5)

s.save("l01-residual.svg")
