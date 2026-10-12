"""Figure 1.1: GPT-2 small, one forward pass, every edge labelled with its tensor shape.

Decoder-only, Pre-LN (GPT-2 report sec. 2.3 and openai/gpt-2 src/model.py). Shapes follow the
code's row convention (one row per token, batch axis B omitted). Parameter counts are computed
here from the config (L=12, d=768, n_h=12, d_ff=3072, V=50257, T_max=1024) and asserted against
the exact total 124,439,808 recorded in reference-docs/lesson-01-doc.md sec. 4.1.
Writes deepseek/figures/l01-architecture.svg
"""

from l01_svg import SVG

L, d, nh, dff, V, Tmax = 12, 768, 12, 3072, 50257, 1024
dh = d // nh
wte, wpe = V * d, Tmax * d
attn = d * 3 * d + 3 * d + d * d + d
mlp = d * dff + dff + dff * d + d
ln = 2 * d
block = attn + mlp + 2 * ln
total = wte + wpe + L * block + ln
assert total == 124_439_808 and block == 7_087_872

W, H = 760, 944
s = SVG(W, H, "GPT-2 small 的一次前向传播：每条边上的张量形状")
SX = 150          # x of the residual stream
MONO_C = "muted"


def lab(x, y, t, anchor="start", color=MONO_C, size=12):
    s.mono(x, y, t, size=size, anchor=anchor, color=color)


# ---------------------------------------------------------------- inputs
s.box(70, 14, 160, 34, "token id：x_{1}, …, x_{T}", color="ink", fill_opacity=0.06)
s.box(262, 14, 160, 34, "位置：0, 1, …, T−1", color="muted", fill_opacity=0.05, text_color="muted")
s.arrow(SX, 48, SX, 82)
lab(SX + 8, 70, "[T] 整数")
s.arrow(342, 48, 342, 82)
lab(350, 70, "[T] 整数")
s.box(70, 82, 160, 48, ["词嵌入 wte（查表）", "矩阵 *E*：[50257, 768]"], color="ink", fill_opacity=0.06, line_gap=20)
s.box(262, 82, 160, 48, ["位置嵌入 wpe（查表）", "矩阵 *P*：[1024, 768]"], color="ink", fill_opacity=0.06, line_gap=20)
s.arrow(SX, 130, SX, 162)
lab(SX - 8, 152, "[T, 768]", anchor="end")
s.path("M342 130 V172 H162", stroke="ink", width=1.6, arrow=True)
lab(350, 155, "[T, 768]")
s.plus(SX, 172)
s.line(SX, 181, SX, 214, color="ink", width=4)
s.text(SX + 10, 200, "*h*^{(0)}", size=13)
lab(SX + 42, 200, "[T, 768]")

# ---------------------------------------------------------------- the block
s.rect(34, 214, 390, 448, fill="ink", fill_opacity=0.02, stroke="muted", width=1.3, rx=12, dash="6 4")
s.text(412, 236, "Transformer 块 ×12", size=14, anchor="end", weight="700")
s.text(412, 254, "12 块结构相同、参数各自独立", size=12, anchor="end", color="muted")

# residual stream segments
s.line(SX, 214, SX, 421, color="ink", width=4)
s.plus(SX, 430)
s.line(SX, 439, SX, 621, color="ink", width=4)
s.plus(SX, 630)
s.line(SX, 639, SX, 662, color="ink", width=4)
s.text(SX - 10, 290, "*h*^{(l−1)}", size=13, anchor="end")
lab(SX - 10, 306, "[T, 768]", anchor="end")
s.text(SX - 10, 486, "*h*′", size=13, anchor="end")
lab(SX - 10, 502, "[T, 768]", anchor="end")
s.text(SX - 14, 262, "残差流", size=12, anchor="end", color="muted")

# branch 1: attention
s.path("M%d 272 H270 V294" % SX, stroke="ink", width=1.6, arrow=True)
s.box(200, 294, 140, 30, "LayerNorm₁（ln_1）", color="muted", fill_opacity=0.10, size=12)
s.arrow(270, 324, 270, 346)
lab(278, 340, "[T, 768]")
s.box(186, 346, 218, 58, ["掩码多头自注意力", ("12 头 × 64 维", {"size": 12, "color": "muted"})],
      color="blue", fill_opacity=0.10, weight="600")
s.path("M270 404 V430 H%d" % (SX + 11), stroke="ink", width=1.6, arrow=True)
s.text(278, 422, "*u*", size=13)
lab(290, 422, "[T, 768]")
s.text(SX + 18, 446, "相加（残差连接）", size=12, color="muted")

# branch 2: FFN
s.path("M%d 470 H270 V494" % SX, stroke="ink", width=1.6, arrow=True)
s.box(200, 494, 140, 30, "LayerNorm₂（ln_2）", color="muted", fill_opacity=0.10, size=12)
s.arrow(270, 524, 270, 546)
lab(278, 540, "[T, 768]")
s.box(186, 546, 218, 58, ["前馈网络 FFN（逐位置）", ("768 → 3072 → 768，GELU", {"size": 12, "color": "muted"})],
      color="muted", fill_opacity=0.12, weight="600")
s.path("M270 604 V630 H%d" % (SX + 11), stroke="ink", width=1.6, arrow=True)
s.text(278, 622, "*m*", size=13)
lab(292, 622, "[T, 768]")
s.text(SX + 18, 648, "相加（残差连接）", size=12, color="muted")

# ---------------------------------------------------------------- zoom: attention
PX = 446
s.rect(PX, 214, 302, 290, fill="blue", fill_opacity=0.04, stroke="blue", width=1.1, rx=10)
s.line(404, 375, PX, 300, color="blue", width=1, dash="3 3")
s.text(PX + 12, 234, "放大：一层掩码多头自注意力", size=13, weight="700", color="blue")
lab(PX + 12, 254, "输入：LN₁ 的输出 [T, 768]")
s.box(PX + 20, 262, 262, 26, "c_attn：乘 W_{qkv}、加偏置（768 → 2304）", size=12, color="ink", fill_opacity=0.05)
s.arrow(PX + 151, 288, PX + 151, 304)
lab(PX + 160, 300, "[T, 2304]")
for k, (nm, col) in enumerate((("Q", "blue"), ("K", "violet"), ("V", "green"))):
    x0 = PX + 20 + k * 90
    s.box(x0, 306, 82, 24, [("%s  [T, 768]" % nm, {"size": 12, "color": col, "weight": "600"})],
          color=col, fill_opacity=0.12)
s.text(PX + 163, 346, "拆头 → [12, T, 64]", size=12, color="muted")
s.arrow(PX + 61, 330, PX + 61, 352, color="blue", width=1.3)
s.arrow(PX + 151, 330, PX + 151, 352, color="violet", width=1.3)
s.box(PX + 20, 354, 262, 24, "S = QK^{⊤} / √64  →  [12, T, T]", size=12, color="violet", fill_opacity=0.08)
s.arrow(PX + 151, 378, PX + 151, 392)
s.box(PX + 20, 392, 262, 24, "因果掩码 + 逐行 softmax → A  [12, T, T]", size=12, color="red", fill_opacity=0.07)
s.path("M%d 318 H%d V442" % (PX + 282, PX + 294), stroke="green", width=1.3)
s.path("M%d 442 H%d" % (PX + 294, PX + 283), stroke="green", width=1.3, arrow=True)
s.arrow(PX + 151, 416, PX + 151, 430)
s.box(PX + 20, 430, 262, 24, "A V → [12, T, 64]；合头 → [T, 768]", size=12, color="green", fill_opacity=0.08)
s.arrow(PX + 151, 454, PX + 151, 468)
s.box(PX + 20, 468, 262, 26, "W^{O}（768×768）加偏置 → *u*  [T, 768]", size=12, color="ink", fill_opacity=0.05)

# ---------------------------------------------------------------- zoom: FFN
s.rect(PX, 516, 302, 146, fill="muted", fill_opacity=0.05, stroke="muted", width=1.1, rx=10)
s.line(404, 575, PX, 560, color="muted", width=1, dash="3 3")
s.text(PX + 12, 536, "放大：一层前馈网络", size=13, weight="700", color="muted")
lab(PX + 12, 556, "输入：LN₂ 的输出 [T, 768]")
s.box(PX + 20, 564, 262, 24, "c_fc：乘 W_{1}、加偏置（768 → 3072）", size=12, color="ink", fill_opacity=0.05)
s.arrow(PX + 151, 588, PX + 151, 600)
s.box(PX + 20, 600, 262, 24, "GELU（逐元素）→ [T, 3072]", size=12, color="ink", fill_opacity=0.05)
s.arrow(PX + 151, 624, PX + 151, 634)
s.box(PX + 20, 634, 262, 24, "c_proj：乘 W_{2}、加偏置 → *m*  [T, 768]", size=12, color="ink", fill_opacity=0.05)

# ---------------------------------------------------------------- output
s.line(SX, 662, SX, 686, color="ink", width=4)
s.arrow(SX, 686, SX, 700, width=2.2)
s.text(SX + 10, 684, "*h*^{(12)}", size=13)
lab(SX + 46, 684, "[T, 768]")
s.box(70, 700, 160, 32, "最终 LayerNorm（ln_f）", color="muted", fill_opacity=0.10, size=12)
s.arrow(SX, 732, SX, 760)
lab(SX + 8, 750, "[T, 768]")
s.box(60, 760, 180, 50, ["反嵌入：Z = H̃ *E*^{⊤}", ("768 → 50257，不新增参数", {"size": 12, "color": "muted"})],
      color="green", fill_opacity=0.08)
s.arrow(SX, 810, SX, 842)
s.text(SX + 8, 831, "logits Z", size=12, color="green")
lab(SX + 64, 831, "[T, 50257]")
s.box(70, 842, 160, 32, "softmax（逐行）", color="green", fill_opacity=0.12, size=12)
s.arrow(SX, 874, SX, 900)
lab(SX + 8, 892, "[T, 50257]")
s.text(40, 920, "第 t 行 = p(x_{t+1} | x_{1}, …, x_{t})：50257 个非负数，和为 1", size=12)

# weight tying
s.path("M70 106 H20 V785 H58", stroke="ink", width=1.3, dash="5 4", arrow=True)
s.text(14, 450, "同一个矩阵 E（权重共享）", size=12, anchor="middle", rotate=-90, color="ink")

# ---------------------------------------------------------------- legend (top right)
s.text(PX, 30, "怎么读这张图", size=13, weight="700")
leg = [
    ("粗黑竖线 = 残差流：每个位置一条 768 维向量", "ink"),
    ("[T, 768] = 张量形状，每行一个 token（省略 batch 维）", "muted"),
    ("T ≤ 1024（上下文长度）；d = 768；d_{h} = 64", "muted"),
    ("蓝 = 注意力；Q 蓝、K 紫、V 绿；红 = 掩码", "blue"),
    ("灰 = LayerNorm 与 FFN；绿 = 输出", "muted"),
    ("自上而下读；块内每个子层“读一份、加回去”", "muted"),
]
for k, (t, col) in enumerate(leg):
    s.text(PX, 52 + k * 22, t, size=12, color=col if col != "muted" else "ink")

# ---------------------------------------------------------------- parameter ledger (bottom right)
s.text(PX, 700, "参数（GPT-2 small，精确计数）", size=13, weight="700")
rows = [
    ("wte  50257×768", wte),
    ("wpe  1024×768", wpe),
    ("每块：注意力", attn),
    ("每块：FFN", mlp),
    ("每块：2 个 LayerNorm", 2 * ln),
    ("12 块合计", L * block),
    ("ln_f", ln),
    ("反嵌入（与 wte 共享）", 0),
]
for k, (nm, v) in enumerate(rows):
    y = 722 + k * 21
    s.text(PX, y, nm, size=12, color="ink")
    s.mono(PX + 296, y, "{:,}".format(v), size=12, anchor="end")
s.line(PX, 722 + 7 * 21 + 8, PX + 298, 722 + 7 * 21 + 8, color="rule", width=1)
s.text(PX, 722 + 8 * 21 + 8, "合计", size=13, weight="700")
s.mono(PX + 296, 722 + 8 * 21 + 8, "{:,}".format(total), size=13, anchor="end", weight="700")

s.save("l01-architecture.svg")
