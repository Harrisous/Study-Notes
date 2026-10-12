"""Figure M1.5: the three matrix products of one attention head, drawn to scale.

GPT-2 small shapes (d = 768, n_h = 12, d_h = 64) with a sequence of T = 1024 tokens,
row convention (one token per row). Block sizes are proportional to the real dimensions;
FLOP counts use 2mnk (see section 5 of the lesson).
Writes math/figures/m01-attention-shapes.svg
"""

from m01_svg import SVG

T, d, dh = 1024, 768, 64
k = 0.13                                   # pixels per unit of dimension
PT, PD, PH = T * k, d * k, dh * k


def flops(m, kk, n):
    return 2 * m * kk * n


def sci(x):
    e = len(str(int(x))) - 1
    return "%.2f×10^{%d}" % (x / 10 ** e, e)


W, H = 720, 476
svg = SVG(W, H, "单个注意力头里的三次矩阵乘法，按 GPT-2 small 的真实形状等比例绘制")


def block(x, y, w, h, color, name, shape, nums, label_side="below", opacity=0.16, mask=False):
    svg.rect(x, y, w, h, fill=color, fill_opacity=opacity, stroke=color, width=1.3)
    if mask:
        # causal mask: the strict upper triangle is thrown away (red)
        svg.polygon([(x, y), (x + w, y), (x + w, y + h)], fill="red", fill_opacity=0.16)
        svg.line(x, y, x + w, y + h, color="red", width=1, dash="3 3")
    if label_side == "below":
        cx, ty = x + w / 2, y + h + 18
        svg.text(cx, ty, "%s  %s" % (name, shape), size=13, anchor="middle", color=color, weight="600")
        svg.text(cx, ty + 17, nums, size=12, anchor="middle", color="muted")
    elif label_side == "above":
        cx, ty = x + w / 2, y - 24
        svg.text(cx, ty, "%s  %s" % (name, shape), size=13, anchor="middle", color=color, weight="600")
        svg.text(cx, ty + 16, nums, size=12, anchor="middle", color="muted")


def op(x, y, s):
    svg.text(x, y, s, size=20, anchor="middle", color="muted")


# ---------------------------------------------------------------- step 1: projection
y1 = 44
svg.text(16, 22, "① 投影：每个 token 的 768 维隐藏状态 → 这个头的 64 维查询（键、值同理）", size=14, weight="600")
x = 40
block(x, y1, PD, PT, "ink", "*H*", "[T, d]", "[1024, 768]", opacity=0.07)   # H as in the lesson's table
op(x + PD + 22, y1 + PT / 2 + 7, "×")
xw = x + PD + 44
block(xw, y1 + (PT - PD) / 2, PH, PD, "blue", "", "", "")
svg.text(xw + PH + 8, y1 + (PT - PD) / 2 + 14, "(*W*^{Q}_{i})^{⊤}  [d, d_{h}]", size=13, color="blue", weight="600")
svg.text(xw + PH + 8, y1 + (PT - PD) / 2 + 36, "[768, 64]", size=12, color="muted")
op(xw + PH + 110, y1 + PT / 2 + 7, "=")
xq = xw + PH + 136
block(xq, y1, PH, PT, "blue", "", "", "")
svg.text(xq + PH + 8, y1 + 14, "*Q*_{i}  [T, d_{h}]", size=13, color="blue", weight="600")
svg.text(xq + PH + 8, y1 + 36, "[1024, 64]", size=12, color="muted")
svg.text(xq + PH + 8, y1 + PT - 6, "%s 次运算" % sci(flops(T, d, dh)), size=12, color="muted")

# a note about proportion
svg.rect(470, y1 + 4, 234, 92, fill="surface", stroke="rule", rx=8)
svg.text(484, y1 + 26, "等比例绘制：方块边长与维度成正比", size=12, color="muted")
svg.text(484, y1 + 46, "T = 1024（序列长度）", size=12, color="muted")
svg.text(484, y1 + 64, "d = 768，d_{h} = 64（GPT-2 small）", size=12, color="muted")
svg.text(484, y1 + 84, "红色：因果掩码丢掉的上三角", size=12, color="red")

svg.line(16, 238, W - 16, 238, color="rule", width=1)

# ---------------------------------------------------------------- step 2: scores
y2 = 290
svg.text(16, 262, "② 打分：*S*_{i} = *Q*_{i}*K*_{i}^{⊤} / √d_{h}", size=14, weight="600")
x = 40
block(x, y2, PH, PT, "blue", "", "", "")
svg.text(x + PH / 2, y2 + PT + 18, "*Q*_{i}", size=13, anchor="middle", color="blue", weight="600")
op(x + PH + 16, y2 + PT / 2 + 7, "×")
xk = x + PH + 32
block(xk, y2 + PT / 2 - PH / 2, PT, PH, "violet", "", "", "")
svg.text(xk + PT / 2, y2 + PT / 2 - PH / 2 - 10, "*K*_{i}^{⊤}  [d_{h}, T]", size=13, anchor="middle", color="violet", weight="600")
op(xk + PT + 16, y2 + PT / 2 + 7, "=")
xs = xk + PT + 32
block(xs, y2, PT, PT, "violet", "", "", "", opacity=0.10, mask=True)
svg.text(xs + PT / 2, y2 + PT + 18, "*S*_{i}  [T, T]", size=13, anchor="middle", color="violet", weight="600")
svg.text(xs + PT / 2, y2 + PT + 41, "%s 次运算" % sci(flops(T, dh, T)), size=12, anchor="middle", color="muted")

# ---------------------------------------------------------------- step 3: weighted sum
svg.line(394, 250, 394, H - 12, color="rule", width=1)
svg.text(408, 262, "③ 加权求和：*O*_{i} = softmax(*S*_{i}) *V*_{i}", size=14, weight="600")
x = 416
block(x, y2, PT, PT, "violet", "", "", "", opacity=0.10, mask=True)
svg.text(x + PT / 2, y2 + PT + 18, "*P*_{i}  [T, T]", size=13, anchor="middle", color="violet", weight="600")
op(x + PT + 16, y2 + PT / 2 + 7, "×")
xv = x + PT + 32
block(xv, y2, PH, PT, "green", "", "", "")
svg.text(xv + PH / 2, y2 + PT + 18, "*V*_{i}", size=13, anchor="middle", color="green", weight="600")
op(xv + PH + 16, y2 + PT / 2 + 7, "=")
xo = xv + PH + 32
block(xo, y2, PH, PT, "green", "", "", "")
svg.text(xo + PH / 2, y2 + PT + 18, "*O*_{i}", size=13, anchor="middle", color="green", weight="600")
svg.text(xo + PH + 8, y2 + 14, "[T, d_{h}]", size=12, color="muted")
svg.text(x + PT / 2 + 40, y2 + PT + 41, "%s 次运算" % sci(flops(T, T, dh)), size=12, anchor="middle", color="muted")

svg.save("m01-attention-shapes.svg")
