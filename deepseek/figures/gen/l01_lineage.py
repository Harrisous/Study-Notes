"""Figure 1.2: from the Transformer's encoder-decoder (Vaswani et al. 2017, Fig. 1) to the
decoder-only GPT-1 (Post-LN) and GPT-2 (Pre-LN).

Redrawn top-down in the course colours. Facts per reference-docs/lesson-01-doc.md:
Vaswani sec. 3.1/3.3/3.5 (Post-LN "LayerNorm(x + Sublayer(x))", ReLU FFN, sinusoidal PE, N = 6;
sec. 3.4: the embedding layers and the pre-softmax linear layer share one matrix, so weight tying
is inherited by GPT-1, not introduced by it);
GPT-1 sec. 4.1 (12-layer decoder-only, masked self-attention, GELU, learned positions, output
softmax(h_n W_e^T)); GPT-2 sec. 2.3 (LN moved to the input of each sub-block, extra final LN,
1/sqrt(N) residual init, vocab 50,257, context 1024). GPT-1's LN placement is not stated in the
report's text, but the report's Fig. 1 (left) draws Layer Norm after each residual add (Post-LN),
and the Hugging Face port agrees (lesson-01-doc C15 and sec. 11.7).
Writes deepseek/figures/l01-lineage.svg
"""

from l01_svg import SVG

W, H = 760, 548
s = SVG(W, H, "从编码器–解码器 Transformer 到仅解码器的 GPT-1 与 GPT-2")

BH = 24          # box height
Y0 = 112         # first row inside a block frame
P = 31           # row pitch


def stack(x, w, rows, frame_label, y_frame0=100, n_rows=6):
    """rows: list of (label, colour, kind) with kind in {"box", "addnorm", "struck"}; returns row y's."""
    ys = []
    frame_h = (n_rows - 1) * P + BH + 24
    s.rect(x - 16, y_frame0, w + 26, frame_h, fill="ink", fill_opacity=0.02, stroke="muted",
           width=1.1, rx=9, dash="5 4")
    s.text(x + w + 8, y_frame0 + frame_h - 6, frame_label, size=12, anchor="end", color="muted", weight="600")
    pitch = P if len(rows) == n_rows else (n_rows - 1) * P / (len(rows) - 1)
    for k, (lab, col, kind) in enumerate(rows):
        y = Y0 + k * pitch
        ys.append(y)
        if kind == "struck":
            s.box(x, y, w, BH, [(lab, {"size": 12, "color": "muted"})], color="red", fill_opacity=0.04,
                  dash="4 3", width=1.1)
            s.line(x + 8, y + BH / 2, x + w - 8, y + BH / 2, color="red", width=1.4)
        elif kind == "addnorm":
            s.box(x + 16, y, w - 32, BH, [(lab, {"size": 12})], color="muted", fill_opacity=0.10)
        else:
            s.box(x, y, w, BH, [(lab, {"size": 12})], color=col, fill_opacity=0.10)
        if k:
            s.arrow(x + w / 2, ys[k - 1] + BH, x + w / 2, y, width=1.2)
    return ys


def bypass(x, y_from, y_to, color="ink"):
    """Residual path on the left of a column, from just above row y_from into the Add box at y_to."""
    s.path("M%.1f %.1f H%.1f V%.1f H%.1f" % (x + 30, y_from - 4, x - 8, y_to + BH / 2, x + 14),
           stroke=color, width=1.1, arrow=True)


# ---------------------------------------------------------------- (a) Transformer 2017
s.text(8, 22, "(a) Transformer（2017）", size=14, weight="700")
s.text(8, 40, "编码器–解码器，Post-LN，N = 6", size=12, color="muted")
# encoder (removed by GPT), drawn muted
ex, ew = 18, 92
s.box(ex - 8, 52, ew + 16, 38, [("输入嵌入", {"size": 12, "color": "muted"}), ("+ 正弦位置", {"size": 12, "color": "muted"})],
      color="muted", fill_opacity=0.05, dash="4 3", width=1.1, line_gap=16)
s.rect(ex - 8, 100, ew + 16, 5 * P + BH + 24, fill="muted", fill_opacity=0.04, stroke="muted", width=1.1,
       rx=9, dash="4 3")
for k, lab in enumerate(["多头自注意力", "Add & Norm", "FFN（ReLU）", "Add & Norm"]):
    y = Y0 + 16 + k * 42
    s.box(ex, y, ew, BH, [(lab, {"size": 12, "color": "muted"})], color="muted", fill_opacity=0.05, width=1.0)
s.text(ex + ew / 2, 300, "编码器 ×6", size=12, anchor="middle", color="muted")
s.text(ex + ew / 2, 330, "✕ GPT 整个删掉", size=12, anchor="middle", color="red", weight="600")
s.line(ex + ew + 8, 190, 150, 190, color="red", width=1.1, dash="4 3", arrow=True)

dx, dw = 150, 142
s.box(dx - 4, 52, dw + 8, 38, [("输出嵌入（右移一位）", {"size": 12}), ("+ 正弦位置", {"size": 12})], color="ink",
      fill_opacity=0.05, line_gap=16)
s.arrow(dx + dw / 2, 90, dx + dw / 2, Y0)
ya = stack(dx, dw, [("掩码多头自注意力", "blue", "box"), ("Add & Norm", "muted", "addnorm"),
                    ("交叉注意力", "red", "struck"), ("Add & Norm", "muted", "struck"),
                    ("FFN（ReLU）", "muted", "box"), ("Add & Norm", "muted", "addnorm")], "×6")
bypass(dx, ya[0], ya[1])
bypass(dx, ya[4], ya[5])
yo = Y0 + 5 * P + BH + 24 + 16
s.arrow(dx + dw / 2, ya[5] + BH, dx + dw / 2, yo)
s.box(dx + 6, yo, dw - 12, BH, [("Linear（共享嵌入）", {"size": 12})], color="green", fill_opacity=0.10)
s.arrow(dx + dw / 2, yo + BH, dx + dw / 2, yo + BH + 12)
s.box(dx + 16, yo + BH + 12, dw - 32, BH, [("Softmax", {"size": 12})], color="green", fill_opacity=0.14)

# ---------------------------------------------------------------- (b) GPT-1
bx, bw = 346, 160
s.text(bx - 20, 22, "(b) GPT-1（2018）", size=14, weight="700")
s.text(bx - 20, 40, "仅解码器，Post-LN，12 层", size=12, color="muted")
s.box(bx - 4, 52, bw + 8, 38, [("词嵌入", {"size": 12}), ("+ 可学习位置", {"size": 12})], color="ink",
      fill_opacity=0.05, line_gap=16)
s.arrow(bx + bw / 2, 90, bx + bw / 2, Y0)
yb = stack(bx, bw, [("掩码多头自注意力", "blue", "box"), ("Add & Norm", "muted", "addnorm"),
                    ("FFN（GELU）", "muted", "box"), ("Add & Norm", "muted", "addnorm")], "×12")
bypass(bx, yb[0], yb[1])
bypass(bx, yb[2], yb[3])
s.arrow(bx + bw / 2, yb[3] + BH, bx + bw / 2, yo)
s.box(bx + 10, yo, bw - 20, BH, [("× W_{e}^{⊤}（共享）", {"size": 12})], color="green", fill_opacity=0.10)
s.arrow(bx + bw / 2, yo + BH, bx + bw / 2, yo + BH + 12)
s.box(bx + 10, yo + BH + 12, bw - 20, BH, [("Softmax", {"size": 12})], color="green", fill_opacity=0.14)

# ---------------------------------------------------------------- (c) GPT-2
cx, cw = 572, 160
s.text(cx - 20, 22, "(c) GPT-2（2019）", size=14, weight="700")
s.text(cx - 20, 40, "仅解码器，Pre-LN，L = 12–48 层", size=12, color="muted")
s.box(cx - 4, 52, cw + 8, 38, [("词嵌入", {"size": 12}), ("+ 可学习位置", {"size": 12})], color="ink",
      fill_opacity=0.05, line_gap=16)
s.arrow(cx + cw / 2, 90, cx + cw / 2, Y0)
yc = stack(cx, cw, [("LayerNorm", "muted", "addnorm"), ("掩码多头自注意力", "blue", "box"),
                    ("Add", "muted", "addnorm"), ("LayerNorm", "muted", "addnorm"),
                    ("FFN（GELU）", "muted", "box"), ("Add", "muted", "addnorm")], "×L")
bypass(cx, yc[0], yc[2])
bypass(cx, yc[3], yc[5])
s.arrow(cx + cw / 2, yc[5] + BH, cx + cw / 2, yo - 10)
s.box(cx + 16, yo - 10, cw - 32, BH, [("最终 LayerNorm", {"size": 12, "weight": "600"})], color="ink",
      fill_opacity=0.10, width=1.8)
s.arrow(cx + cw / 2, yo + BH - 10, cx + cw / 2, yo + BH + 2)
s.box(cx + 10, yo + BH + 2, cw - 20, BH, [("× *E*^{⊤}（共享）", {"size": 12})], color="green", fill_opacity=0.10)
s.arrow(cx + cw / 2, yo + 2 * BH + 2, cx + cw / 2, yo + 2 * BH + 14)
s.box(cx + 10, yo + 2 * BH + 14, cw - 20, BH, [("Softmax", {"size": 12})], color="green", fill_opacity=0.14)

# ---------------------------------------------------------------- change notes
NY = 432
s.line(8, NY - 16, 752, NY - 16, color="rule", width=1)
notes = [
    (8, ["每个子层之后：", "LayerNorm(x + 子层(x))", "编码器的输出经交叉注意力", "送进解码器"]),
    (326, ["相对 (a)：", "删去编码器与交叉注意力", "位置改为可学习的表 P", "ReLU → GELU；输出层仍共享嵌入"]),
    (552, ["相对 (b)：", "LN 移到每个子层的输入", "最后加一个 LayerNorm", "残差层初始化再乘 1/√N"]),
]
for x, lines in notes:
    for k, t in enumerate(lines):
        s.text(x, NY + 4 + k * 20, t, size=12, color="ink" if k == 0 else "muted",
               weight="600" if k == 0 else None)
s.text(8, NY + 96, "红色删除线 = GPT 去掉的部分；左侧细箭头 = 残差连接（跳过子层，直接加到 Add 上）", size=12,
       color="muted")

s.save("l01-lineage.svg")
