"""Figure M2.9: one MLA head at decode time, computed in the absorbed order, with DeepSeek-V2 dimensions.

d = 5120, d_c' = 1536, d_h = 128, d_c = 512 (DeepSeek-V2 §3.1.2 via deepseek/reference-docs/lesson-09-doc.md).
Row 1 (scores):  h_t -> c_t^Q (W^DQ) -> q_{t,i}^C (W^UQ_i) -> q~_{t,i} = (W^UK_i)^T q^C (512) -> dot with every cached c_j.
Row 2 (output):  weights p_{t,j} -> o~_{t,i} = sum_j p c_j (512) -> o_{t,i} = W^UV_i o~ (128) -> W^O_i o_{t,i} (5120).
T is the number of cached tokens (style guide: T = sequence length, L = layers).
Vector bars are drawn with height proportional to length (1 px = 40 numbers). The two fixed products that
must NOT be pre-multiplied are marked in red: (W^UK_i)^T W^UQ_i is 512 x 1536 but rank <= 128, and
W^O_i W^UV_i is 5120 x 512 but rank <= 128; stored whole they cost 3.0x and 3.6x the parameters of the factors
(numbers: lesson-09-doc §4.4). RoPE part omitted (lesson 9 §4-§5).
Writes math/figures/m02-absorb-chain.svg
"""

from m02_svg import SVG

d, dcq, dh, dc = 5120, 1536, 128, 512
K = 40.0
pre_q, sep_q = dc * dcq, dh * (dc + dcq)
pre_o, sep_o = d * dc, dh * (d + dc)
assert round(pre_q / sep_q, 1) == 3.0

W, H = 780, 436
svg = SVG(W, H, "吸收后的 MLA 单头解码：先把查询变成 512 维，再直接与缓存点积；两个固定乘积不预先乘出")


def bar(x, base, length, col, name, dim, name_dy=18, fill_op=0.22):
    h = max(length / K, 3)
    svg.rect(x - 9, base - h, 18, h, fill=col, fill_opacity=fill_op, stroke=col, width=1.3)
    svg.text(x, base + name_dy, name, size=13, anchor="middle", color=col)
    svg.text(x, base + name_dy + 19, dim, size=11, anchor="middle", color="muted")  # clear of the name's subscript


def step(x1, x2, y, label, col="ink", sub=None):
    svg.arrow(x1, y, x2, y, color=col, width=1.6)
    # baseline 12 above the arrow: a subscript such as the i in W^{UQ}_{i} drops ~5 below the baseline
    svg.text((x1 + x2) / 2, y - 12, label, size=12, anchor="middle", color=col)
    if sub:
        svg.text((x1 + x2) / 2, y + 16, sub, size=11, anchor="middle", color="muted")


# ---------------- row 1
b1 = 188
svg.text(14, 24, "① 打分：查询先变成 512 维（每步 1 次），再和缓存里的每个 *c*_{j} 点积", size=14, weight="600")
xs = [40, 150, 250, 350]
bar(xs[0], b1, d, "ink", "*h*_{t}", "5120", fill_op=0.08)
bar(xs[1], b1, dcq, "ink", "*c*_{t}^{Q}", "1536", fill_op=0.08)
bar(xs[2], b1, dh, "blue", "*q*_{t,i}^{C}", "128")
bar(xs[3], b1, dc, "blue", "*q̃*_{t,i}", "512")
ya = b1 - 30
step(xs[0] + 14, xs[1] - 14, ya, "*W*^{DQ}")
step(xs[1] + 14, xs[2] - 14, ya, "*W*^{UQ}_{i}")
step(xs[2] + 14, xs[3] - 14, ya, "(*W*^{UK}_{i})^{⊤}", col="violet")
# red bracket: do not pre-multiply
svg.path("M%d %d v-10 h%d v10" % (xs[1] + 4, b1 - 80, xs[3] - xs[1] - 8), stroke="red", width=1.4, dash="4 3")
svg.text((xs[1] + xs[3]) / 2, b1 - 116, "不预乘：(*W*^{UK}_{i})^{⊤}*W*^{UQ}_{i} 是 512×1536", size=12, color="red",
         anchor="middle")
svg.text((xs[1] + xs[3]) / 2, b1 - 96, "秩 ≤ 128；整存参数 %.1f 倍" % (pre_q / sep_q), size=12, color="red",
         anchor="middle")
# cache stack
cx0 = 450
svg.rect(cx0 - 12, b1 - dc / K - 22, 184, dc / K + 48, fill="amber", fill_opacity=0.06, stroke="amber", width=1,
         dash="4 3", rx=6)
# baseline 12 above the box's top border so the subscript j stays clear of it
svg.text(cx0 + 80, b1 - dc / K - 34, "KV cache：每个 token 只存 *c*_{j}（512 个数）", size=12, color="amber",
         anchor="middle")
for j in range(8):
    x = cx0 + j * 20
    svg.rect(x, b1 - dc / K, 12, dc / K, fill="orange", fill_opacity=0.35, stroke="orange", width=1)
# baseline at b1 + 14 so the subscripts stay above the dashed border of the cache box (b1 + 26)
svg.text(cx0 + 6, b1 + 14, "*c*_{1}", size=12, color="orange", anchor="middle")
svg.text(cx0 + 146, b1 + 14, "*c*_{T}", size=12, color="orange", anchor="middle")
svg.text(cx0 + 76, b1 + 14, "…", size=12, color="orange", anchor="middle")
svg.arrow(xs[3] + 14, b1 - 6, cx0 - 16, b1 - 6, color="blue", width=1.6)
svg.text((xs[3] + cx0) / 2, b1 - 14, "点积 ×T", size=12, color="blue", anchor="middle")
svg.text((xs[3] + cx0) / 2, b1 + 34, "2·512·T", size=11, color="muted", anchor="middle")
# scores
sx = 690
for j in range(6):
    svg.rect(sx - 30 + j * 11, b1 - 12, 9, 12, fill="violet", fill_opacity=0.3, stroke="violet", width=0.8)
svg.arrow(cx0 + 176, b1 - 6, sx - 36, b1 - 6, color="ink", width=1.4)
svg.text(sx, b1 + 18, "T 个分数", size=12, color="violet", anchor="middle")
svg.text(sx, b1 + 34, "→ softmax → *p*_{t,j}", size=11, color="muted", anchor="middle")

# ---------------- row 2
b2 = 392
svg.line(14, 246, W - 14, 246, color="rule", width=1)
svg.text(14, 272, "② 输出：先对缓存加权平均（仍是 512 维），最后才展开", size=14, weight="600")
x2 = [70, 250, 400, 560]
# weights feeding the average
for j in range(6):
    svg.rect(x2[0] - 30 + j * 11, b2 - 12, 9, 12, fill="violet", fill_opacity=0.3, stroke="violet", width=0.8)
svg.text(x2[0], b2 + 18, "*p*_{t,j} 与 *c*_{j}", size=12, color="violet", anchor="middle")
bar(x2[1], b2, dc, "green", "*õ*_{t,i}", "512")
bar(x2[2], b2, dh, "green", "*o*_{t,i}", "128")
bar(x2[3], b2, d, "green", "*W*^{O}_{i}*o*_{t,i}", "5120")
yb = b2 - 30
step(x2[0] + 36, x2[1] - 14, yb, "Σ_{j} *p*_{t,j}*c*_{j}", col="green", sub="2·512·T")
step(x2[1] + 14, x2[2] - 14, yb, "*W*^{UV}_{i}", col="green")
step(x2[2] + 14, x2[3] - 14, yb, "*W*^{O}_{i}", col="green")
svg.path("M%d %d v-10 h%d v10" % (x2[1] + 4, 326, x2[3] - 18 - x2[1] - 4), stroke="red", width=1.4, dash="4 3")
svg.text(380, 304, "不预乘：*W*^{O}_{i}*W*^{UV}_{i} 是 5120×512，秩 ≤ 128；整存 %.1f 倍"
         % (pre_o / sep_o), size=12, color="red", anchor="middle")
svg.text(640, b2 - 60, "对 128 个头求和", size=12, color="muted")
svg.text(640, b2 - 44, "= 层输出 *u*_{t}", size=12, color="muted")

svg.save("m02-absorb-chain.svg")
print("premult q %d vs %d (%.2f), o %d vs %d (%.2f)" % (pre_q, sep_q, pre_q / sep_q, pre_o, sep_o, pre_o / sep_o))
