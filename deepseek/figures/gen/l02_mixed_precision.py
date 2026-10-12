"""Figure 2.10: BF16 mixed precision - bit layouts (widths proportional to bit counts) and the data flow
of one training step with FP32 master weights.

Bit layouts: FP32 1/8/23 and FP16 1/5/10 (numpy.finfo: nmant 23 / 10, nexp 8 / 5, max 3.4e38 / 65504);
BF16 1/8/7 = the upper 16 bits of FP32 (Wikipedia "bfloat16 floating-point format"; Google Cloud TPU docs:
same dynamic range as float32). Machine epsilon: 2^-23, 2^-7, 2^-10.
Data flow: Chinchilla section 4.1 (Hoffmann et al. 2022): forward and backward in bfloat16, a float32 copy
of the weights kept in the (distributed) optimizer state. The paper does not state the format of the Adam
moments m, v (or of the gradients); the figure labels FP32 m, v as the author's assumption.
Rounding example (emulated with numpy bit manipulation, see lesson-02-doc.md 补充核实):
  bf16(0.02) = 0.02001953; adding 6e-5 and rounding to BF16 gives the same number back;
  1000 such additions in FP32 give 0.0800190, in BF16 the weight stays at 0.0200195.
Writes deepseek/figures/l02-mixed-precision.svg
"""

from l02_svg import SVG, box

W, H = 720, 500
svg = SVG(W, H, "BF16 混合精度：三种浮点格式的位布局与一步训练的数据流")

# ---------------- bit layouts
BITW = 13.5
X0 = 96
svg.text(16, 22, "浮点数 = 符号 × 2^{指数} × 1.尾数：指数位决定范围，尾数位决定精度", size=13, color="ink",
         weight="600")
fmts = [("FP32", 1, 8, 23, "范围 ≈ 10^{−38} … 3.4×10^{38}", "相对精度 2^{−23} ≈ 1.2×10^{−7}"),
        ("BF16", 1, 8, 7, "范围与 FP32 相同", "相对精度 2^{−7} ≈ 0.0078（2–3 位有效数字）"),
        ("FP16", 1, 5, 10, "最大只有 65,504", "相对精度 2^{−10} ≈ 0.00098")]
for i, (name, sb, eb, mb, r1, r2) in enumerate(fmts):
    y = 40 + i * 48
    svg.text(16, y + 19, name, size=13, color="ink", weight="600")
    x = X0
    for n, col, lab in ((sb, "muted", "符"), (eb, "blue", "指数 %d" % eb), (mb, "orange", "尾数 %d" % mb)):
        w = n * BITW
        svg.rect(x, y + 4, w, 22, fill=col, fill_opacity=0.22 if col != "muted" else 0.3, stroke=col, width=1)
        if w > 40:
            svg.text(x + w / 2, y + 20, lab, size=12, anchor="middle", color=col if col != "muted" else "ink")
        x += w
    xe = X0 + 32 * BITW + 12 if name == "FP32" else x + 12
    svg.text(xe, y + 11, r1, size=12, color="ink")
    svg.text(xe, y + 32, r2, size=12, color="muted")

# ---------------- data flow
yf = 206
svg.text(16, yf, "一个训练步里各种数用什么格式（按 Chinchilla §4.1；m、v 的格式论文未写）", size=13, color="ink",
         weight="600")
box(svg, 16, yf + 14, 200, 78, "FP32 主权重 θ", ["+ m、v（FP32，笔者假设）", "放在优化器状态里"], color="amber",
    fill_opacity=0.10, line_gap=18)
box(svg, 262, yf + 14, 170, 78, "BF16 权重副本", ["每步从主权重", "舍入得到"], color="orange", fill_opacity=0.12,
    line_gap=18)
box(svg, 478, yf + 14, 226, 78, "前向 + 反向（BF16）", ["前向、反向的计算用 BF16", "得到梯度 g"], color="blue",
    fill_opacity=0.08, line_gap=18)
box(svg, 262, yf + 124, 170, 62, "AdamW 更新", ["在 FP32 主权重上做"], color="green", fill_opacity=0.10, line_gap=18)
svg.arrow(217, yf + 53, 260, yf + 53, color="ink", width=1.5)
svg.arrow(433, yf + 53, 476, yf + 53, color="ink", width=1.5)
svg.path("M591 %d V%d H434" % (yf + 93, yf + 155), stroke="ink", width=1.5, arrow=True)
svg.text(596, yf + 128, "梯度 g", size=12, color="ink")
svg.path("M261 %d H116 V%d" % (yf + 155, yf + 94), stroke="green", width=1.5, arrow=True)
svg.text(124, yf + 148, "θ ← θ − η(…)", size=12, color="green")

# ---------------- why FP32 master weights: the rounding example
ye = yf + 212
svg.rect(16, ye, 688, 70, fill="red", fill_opacity=0.06, stroke="red", width=1, rx=6)
svg.text(28, ye + 20, "为什么主权重要 FP32：一个权重 0.02，每步更新约 η = 6×10^{−5}", size=12, color="red", weight="600")
svg.text(28, ye + 40, "BF16 在 0.02 附近的间隔是 2^{−13} ≈ 1.2×10^{−4}，加 6×10^{−5} 会被舍回原值。", size=12, color="ink")
svg.text(28, ye + 60, "连加 1000 次：FP32 得 0.0800，BF16 仍是 0.0200（NumPy 按位模拟）。", size=12, color="ink")

svg.save("l02-mixed-precision.svg")
