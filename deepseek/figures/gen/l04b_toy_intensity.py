"""Figure 4.9 of the blackboard version of lesson 4: the two intensities of a decode step, on toy numbers.

Left: a 1000 x 1000 bf16 weight matrix read once and shared by B vectors (B = 1, 4); the tiny input
vectors are ignored, as in the text. Right: a toy attention layer with 4 query heads, d_h = 2, context 3,
bf16, with 1 KV head (MQA) or 4 KV heads (MHA), for B = 1, 2 sequences that each read their own cache.
All FLOP and byte counts are computed here from the formulas of the lesson.
Writes deepseek/figures/l04b-toy-intensity.svg
"""

from l04_svg import SVG

S = 2          # bytes per number (bf16)
D = 1000       # weight matrix is D x D
NH, DH, T, L = 4, 2, 3, 1


def weight_row(b):
    flops = b * 2 * D * D
    byts = S * D * D
    return flops, byts


def attn_row(b, ng):
    flops = b * 4 * L * NH * DH * T
    byts = b * 2 * ng * DH * L * S * T
    return flops, byts


W, H = 740, 420
svg = SVG(W, H, "黑板上的两部分强度：权重被全 batch 共用，KV 每条序列各读各的")
PX = (10, 378)
PW = 352
for x in PX:
    svg.rect(x, 8, PW, H - 16, fill=None, stroke="rule", width=1, rx=8)
svg.text(PX[0] + 14, 32, "权重部分：一份 W，全 batch 共用", size=13, weight="700")
svg.text(PX[1] + 14, 32, "注意力部分：每条序列读自己的 KV", size=13, weight="700")

# ---- left schematic: W shared by 4 vectors
wx, wy, ws = PX[0] + 30, 62, 92
svg.rect(wx, wy, ws, ws, fill="muted", fill_opacity=0.14, stroke="ink", width=1.6, rx=4)
svg.text(wx + ws / 2, wy + ws / 2 - 4, "W", size=16, anchor="middle", weight="700")
svg.text(wx + ws / 2, wy + ws / 2 + 16, "1000×1000", size=12, anchor="middle", color="muted")
svg.text(wx + ws / 2, wy + ws + 20, "搬一次：2,000,000 字节", size=12, anchor="middle")
for i in range(4):
    cy = wy + 10 + i * 24
    svg.arrow(wx + ws + 6, wy + ws / 2, PX[0] + 232, cy, color="ink", width=1.3, shorten=8)
    svg.circle(PX[0] + 240, cy, 8, fill="blue", fill_opacity=0.25, stroke="blue", width=1.3)
    svg.text(PX[0] + 256, cy + 4, "序列 %d" % (i + 1), size=12, color="blue")
svg.text(wx - 6, wy + ws + 44, "B = 4 时，4 条序列都用这同一份", size=12, color="muted")

# ---- right schematic: two sequences, each with its own KV (MQA: 1 KV head)
for sidx in range(2):
    bx = PX[1] + 24 + sidx * 164
    by = 58
    svg.rect(bx - 6, by - 6, 140, 96, fill="amber", fill_opacity=0.05, stroke="amber", width=1.2, rx=6)
    svg.text(bx + 64, by + 140, "序列 %d 自己的缓存" % (sidx + 1), size=12, anchor="middle", color="amber")
    for j in range(T):
        ry = by + 4 + j * 22
        svg.text(bx + 2, ry + 13, "位置 %d" % (j + 1), size=11, color="muted")
        for c in range(DH):
            svg.rect(bx + 42 + c * 14, ry, 12, 16, fill="violet", fill_opacity=0.25, stroke="violet", width=0.7, rx=2)
            svg.rect(bx + 76 + c * 14, ry, 12, 16, fill="green", fill_opacity=0.25, stroke="green", width=0.7, rx=2)
    svg.text(bx + 55, by + 82, "K", size=12, anchor="middle", color="violet", weight="600")
    svg.text(bx + 89, by + 82, "V", size=12, anchor="middle", color="green", weight="600")
    for h in range(NH):
        hx = bx + 22 + h * 28
        svg.line(hx, by + 90, hx, by + 110, color="amber", width=1.1)
        svg.circle(hx, by + 116, 6, fill="blue", fill_opacity=0.25, stroke="blue", width=1.1)
svg.text(PX[1] + 14, 226, "每条序列 4 个查询头（蓝点）共用 1 个 KV 头（MQA）", size=12, color="muted")

# ---- tables
def table(x0, y0, head, rows):
    cols = [x0 + 14, x0 + 150, x0 + 236, x0 + 322]
    for i, hname in enumerate(head):
        svg.text(cols[i], y0, hname, size=12, weight="700", anchor="start" if i == 0 else "end")
    svg.line(x0 + 10, y0 + 8, x0 + PW - 10, y0 + 8, color="rule", width=1)
    for r, (name, fl, by, hi) in enumerate(rows):
        y = y0 + 30 + r * 28
        if hi:
            svg.rect(x0 + 8, y - 18, PW - 16, 26, fill="blue", fill_opacity=0.06, rx=4)
        svg.text(cols[0], y, name, size=12)
        svg.text(cols[1], y, "{:,}".format(fl), size=12, anchor="end")
        svg.text(cols[2], y, "{:,}".format(by), size=12, anchor="end", color="amber")
        iv = fl / by
        svg.text(cols[3], y, ("%d" % round(iv)) if abs(iv - round(iv)) < 1e-9 else "%.1f" % iv, size=13,
                 anchor="end", color="blue", weight="700")


lrows = []
for b in (1, 4):
    fl, by = weight_row(b)
    lrows.append(("B = %d" % b, fl, by, b == 4))
table(PX[0], 258, ["", "运算", "字节", "强度"], lrows)
svg.text(PX[0] + 14, 346, "字节不随 B 变，强度 = B", size=12, color="blue", weight="600")
svg.text(PX[0] + 14, 368, "（输入向量的字节不到 1%，忽略）", size=12, color="muted")

rrows = []
for ng, nm in ((1, "MQA"), (4, "MHA")):
    for b in (1, 2):
        fl, by = attn_row(b, ng)
        rrows.append(("%s，B = %d" % (nm, b), fl, by, False))
table(PX[1], 258, ["", "运算", "字节", "强度"], rrows)
svg.text(PX[1] + 14, 398, "运算和字节一起随 B 翻倍，强度 = n_{h}/n_{g}", size=12, color="blue", weight="600")
svg.save("l04b-toy-intensity.svg")
