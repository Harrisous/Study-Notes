"""Figure 5.8: the SwiGLU feed-forward layer with LLaMA-7B dimensions, and the parameter-matching picture.

Left: data flow x (4096) -> W1 (gate) and W3 (up), both 4096 -> 11008; SiLU on the gate; elementwise product;
      W2 (down) 11008 -> 4096 (LLaMA code: w2(silu(w1 x) * w3 x); widths from LLaMA Table 2 / code rule).
Right: matrices drawn to scale. GELU FFN: two d x 4d matrices; SwiGLU: three d x d_ff matrices with
       d_ff = 8d/3 (exact match) and 11008 (LLaMA-7B after rounding up to a multiple of 256).
Writes deepseek/figures/l05-swiglu.svg
"""

from l05_svg import SVG

d = 4096
dff = 11008
exact = 8 * d / 3
p_gelu = 2 * d * 4 * d
p_swi_exact = 3 * d * exact
p_swi = 3 * d * dff
assert abs(p_gelu - p_swi_exact) < 1e-6
assert dff == ((int(2 * 4 * d / 3) + 255) // 256) * 256

W, H = 760, 426
svg = SVG(W, H, "SwiGLU 前馈层（LLaMA-7B 维度）与参数匹配")

# ---------------- left: data flow
svg.text(185, 24, "数据流（一个 token）", size=13, anchor="middle", weight="700")
svg.box(106, 40, 158, 38, ["输入 *x*", "RMSNorm 之后，4096 维"], color="muted", fill_opacity=0.08, size=13)
# branches
svg.arrow(150, 78, 90, 118, color="ink", width=1.4)
svg.arrow(220, 78, 280, 118, color="ink", width=1.4)
svg.box(20, 120, 140, 50, ["W_{1}（gate）", "4096 → 11008"], color="blue", size=13, weight="600", line_gap=20)
svg.box(210, 120, 140, 50, ["W_{3}（up）", "4096 → 11008"], color="blue", size=13, weight="600", line_gap=20)
svg.arrow(90, 170, 90, 196, color="ink", width=1.4)
svg.circle(90, 214, 18, fill="violet", fill_opacity=0.12, stroke="violet", width=1.4)
svg.text(90, 219, "SiLU", size=12, anchor="middle", color="violet", weight="600")
svg.text(20, 248, "SiLU(z) = z·σ(z)", size=12, color="violet")
svg.text(20, 300, "门：给每个分量一个“开度”", size=11.5, color="muted")
# product node
svg.arrow(104, 226, 172, 272, color="ink", width=1.4)
svg.arrow(280, 170, 198, 272, color="ink", width=1.4)
svg.circle(185, 286, 15, fill="surface", stroke="ink", width=1.5)
svg.text(185, 292, "⊙", size=17, anchor="middle")
svg.text(205, 312, "逐元素相乘，11008 维", size=11.5, color="muted")
svg.arrow(185, 301, 185, 328, color="ink", width=1.4)
# height 50 like the W1/W3 boxes, and a wider line gap in all three: with the default gap
# the subscript of W_k touched the second line
svg.box(115, 330, 140, 50, ["W_{2}（down）", "11008 → 4096"], color="green", size=13, weight="600", line_gap=20)
svg.arrow(185, 380, 185, 400, color="ink", width=1.4)
svg.text(185, 416, "输出，加回残差主干", size=12, anchor="middle", color="green")

# ---------------- right: parameter matching to scale
X0 = 420
svg.text(X0 + 160, 24, "参数匹配（按比例画，d = 4096）", size=13, anchor="middle", weight="700")
PX = 160 / (4 * d)          # px per unit of width
HH = 40                      # height of a d-row matrix


def mats(y, n, width_units, col, label, sub, total, tail=""):
    w = width_units * PX
    for k in range(n):
        svg.rect(X0 + k * (w + 8), y, w, HH, fill=col, fill_opacity=0.18, stroke=col, width=1.2)
    svg.text(X0, y - 8, label, size=12.5, weight="600", color=col)
    svg.text(X0, y + HH + 18, sub + " = " + "{:,}".format(round(total)) + tail, size=12, color="muted")


mats(64, 2, 4 * d, "muted", "GELU 前馈层：2 个 d × 4d 矩阵", "2 · d · 4d = 8d²", p_gelu)
mats(172, 3, exact, "blue", "SwiGLU，d_{ff} = 8d/3 = 10922.67：3 个矩阵", "3 · d · (8d/3) = 8d²", p_swi_exact)
mats(280, 3, dff, "blue", "LLaMA-7B 实际取 d_{ff} = 11008", "3 · d · 11008", p_swi, "（多 %.2f%%）" % ((p_swi / p_gelu - 1) * 100))
svg.text(X0, 380, "每个方块高 = d = 4096；宽 = 该矩阵的另一维。", size=12, color="muted")
svg.text(X0, 398, "面积 = 参数量。三个窄方块的总面积等于两个宽方块。", size=12, color="muted")

svg.save("l05-swiglu.svg")
