"""Figure M10.7: the four ways to split training across GPUs (schematic, 4 GPUs each).

Data parallel: every GPU holds the whole model and a different slice of the batch; gradients
are summed with an all-reduce.  Tensor parallel: one matrix is cut into column blocks (and the
next one into row blocks, Megatron-LM style); partial results are summed with an all-reduce
inside every layer.  Pipeline parallel: consecutive layers live on consecutive GPUs; only the
activations at the stage boundaries move.  Expert parallel: different experts live on
different GPUs; every token is sent to the GPU holding its experts and back (all-to-all).
Writes math/figures/m10-parallelism.svg
"""

from m10_svg import SVG

W, H = 740, 540
svg = SVG(W, H, "四种并行：数据、张量、流水线、专家")
PW, PH = 352, 250
origins = [(12, 10), (376, 10), (12, 280), (376, 280)]


def panel(ox, oy, title, sub):
    svg.rect(ox, oy, PW, PH, fill="surface", stroke="rule", width=1.2, rx=8)
    svg.text(ox + 14, oy + 24, title, size=14, weight="700")
    svg.text(ox + 14, oy + 42, sub, size=12, color="muted")


def gpu(x, y, w, h, label):
    svg.rect(x, y, w, h, fill="grid", fill_opacity=0.6, stroke="muted", width=1.1, rx=5)
    svg.text(x + w / 2, y + h + 15, label, size=11, anchor="middle", color="muted")


# ---------------- data parallel
ox, oy = origins[0]
panel(ox, oy, "数据并行（DP）", "每卡一份完整模型，各吃不同的数据")
gw, gh = 62, 104
for i in range(4):
    x = ox + 22 + i * 82
    y = oy + 76
    gpu(x, y, gw, gh, "GPU %d" % i)
    for L in range(4):
        svg.rect(x + 8, y + 10 + L * 22, gw - 16, 16, fill="muted", fill_opacity=0.35, rx=2)
    svg.text(x + gw / 2, y + 10 + 3 * 22 + 12, "全 16 层", size=11, anchor="middle", color="ink")
    svg.rect(x + 12, y - 22, gw - 24, 16, fill="blue", fill_opacity=0.8, rx=3)
    svg.text(x + gw / 2, y - 10, "数据 %d" % i, size=11, anchor="middle", color="surface", weight="600")
# all-reduce bar
by = oy + 212
svg.line(ox + 22 + gw / 2, by, ox + 22 + 3 * 82 + gw / 2, by, color="ink", width=1.6)
for i in range(4):
    x = ox + 22 + i * 82 + gw / 2
    svg.line(x, oy + 76 + gh + 18, x, by, color="ink", width=1.3)
svg.text(ox + PW / 2, by + 20, "每步一次：梯度 all-reduce（求和）", size=12, anchor="middle", weight="600")

# ---------------- tensor parallel
ox, oy = origins[1]
panel(ox, oy, "张量并行（TP）", "把一层里的矩阵切开，每层都要通信")
# input X broadcast
svg.rect(ox + 20, oy + 100, 34, 64, fill="blue", fill_opacity=0.7, rx=3)
svg.text(ox + 37, oy + 137, "X", size=13, anchor="middle", color="surface", weight="700")
for i, (yy, lab) in enumerate(((oy + 64, "GPU 0"), (oy + 150, "GPU 1"))):
    gpu(ox + 76, yy, 196, 70, "")
    svg.text(ox + 268, yy + 64, lab, size=11, color="muted", anchor="end")
    svg.arrow(ox + 56, oy + 132, ox + 86, yy + 35, color="ink", width=1.2, shorten=2)
    # column block of A
    svg.rect(ox + 90, yy + 12, 34, 46, fill="violet", fill_opacity=0.35, stroke="violet", width=1)
    svg.text(ox + 107, yy + 40, "A_{%d}" % (i + 1), size=12, anchor="middle")
    svg.arrow(ox + 126, yy + 35, ox + 146, yy + 35, color="ink", width=1.2)
    svg.rect(ox + 148, yy + 12, 40, 46, fill="green", fill_opacity=0.25, stroke="green", width=1)
    svg.text(ox + 168, yy + 32, "GeLU", size=11, anchor="middle")
    svg.text(ox + 168, yy + 48, "(XA_{%d})" % (i + 1), size=11, anchor="middle")
    svg.arrow(ox + 190, yy + 35, ox + 208, yy + 35, color="ink", width=1.2)
    svg.rect(ox + 210, yy + 22, 50, 26, fill="violet", fill_opacity=0.35, stroke="violet", width=1)
    svg.text(ox + 235, yy + 40, "B_{%d}" % (i + 1), size=12, anchor="middle")
    # the partial result leaving this GPU
    svg.text(ox + 291, yy + 25, "Z_{%d}" % (i + 1), size=12, anchor="middle", weight="600")
svg.text(ox + 107, oy + 60, "按列切", size=11, anchor="middle", color="violet")
svg.text(ox + 235, oy + 60, "按行切", size=11, anchor="middle", color="violet")
# all-reduce of partial sums
svg.line(ox + 262, oy + 99, ox + 320, oy + 99, color="ink", width=1.3)
svg.line(ox + 262, oy + 185, ox + 320, oy + 185, color="ink", width=1.3)
svg.line(ox + 320, oy + 99, ox + 320, oy + 185, color="ink", width=1.6)
svg.text(ox + 176, oy + 240, "部分和 Z_{1} + Z_{2}：前向一次 all-reduce", size=12, anchor="middle", weight="600")

# ---------------- pipeline parallel
ox, oy = origins[2]
panel(ox, oy, "流水线并行（PP）", "按层切成几段，段与段之间传激活")
for i in range(4):
    x = ox + 22 + i * 82
    y = oy + 70
    gpu(x, y, gw, 110, "GPU %d" % i)
    svg.rect(x + 8, y + 10, gw - 16, 90, fill="muted", fill_opacity=0.35, rx=2)
    svg.text(x + gw / 2, y + 52, "第 %d–%d" % (4 * i + 1, 4 * i + 4), size=11, anchor="middle")
    svg.text(x + gw / 2, y + 68, "层", size=11, anchor="middle")
    if i < 3:
        svg.arrow(x + gw + 2, y + 36, x + 82 - 2, y + 36, color="blue", width=1.5)
        svg.arrow(x + 82 - 2, y + 80, x + gw + 2, y + 80, color="orange", width=1.5)
svg.text(ox + 20, oy + 212, "→ 前向传激活", size=12, color="blue", weight="600")
svg.text(ox + 130, oy + 212, "← 反向传梯度", size=12, color="orange", weight="600")
svg.text(ox + 20, oy + 234, "点对点、只在段的边界通信；代价是流水线气泡", size=12, color="muted")

# ---------------- expert parallel
ox, oy = origins[3]
panel(ox, oy, "专家并行（EP）", "不同专家放在不同卡上，token 去找专家")
for i in range(4):
    x = ox + 22 + i * 82
    y = oy + 110
    gpu(x, y, gw, 70, "GPU %d" % i)
    for e in range(2):
        svg.rect(x + 7 + e * 25, y + 10, 22, 50, fill="blue", fill_opacity=0.18, stroke="blue", width=1, rx=2)
        svg.text(x + 18 + e * 25, y + 40, "E%d" % (2 * i + e), size=11, anchor="middle")
    # tokens sitting on this GPU before dispatch, labelled with the GPU that holds their expert
    for t in range(4):
        dest = (i + t) % 4
        svg.circle(x + 8 + t * 15.3, y - 30, 6.6, fill="blue", fill_opacity=0.85)
        svg.text(x + 8 + t * 15.3, y - 26, str(dest), size=11, anchor="middle", color="surface", weight="600")
# a few crossing arrows (dispatch)
pairs = [(0, 2), (1, 3), (2, 0), (3, 1), (0, 1), (2, 3)]
for a, b in pairs:
    xa = ox + 22 + a * 82 + 8 + ((b - a) % 4) * 15.3
    xb = ox + 22 + b * 82 + gw / 2
    svg.arrow(xa, oy + 88, xb, oy + 108, color="muted", width=1, shorten=2)
svg.text(ox + 176, oy + 62, "每个圆点是一个 token，数字是它的专家所在的卡", size=11, anchor="middle", color="muted")
svg.text(ox + 176, oy + 216, "dispatch：all-to-all 发出去", size=12, anchor="middle", weight="600")
svg.text(ox + 176, oy + 236, "combine：算完再 all-to-all 发回来", size=12, anchor="middle", weight="600")

svg.save("m10-parallelism.svg")
