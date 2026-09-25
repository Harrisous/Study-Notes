"""Figure 18.1: DeepSeek-V3.2 extrapolated to 1M tokens -- the three parts of the per-token FLOPs
(raw count, eq. 18.2) and the KV cache of one request (eq. 18.1).

Raw FLOPs: projections 2 * 37e9; core attention 61 * 128 * 2176 * min(t, 2048);
lightning indexer 61 * 64 * 256 * t. KV = 48,068 B/token * t. V3.2 officially supports 128K,
so the bars to the right of 128K are extrapolations. Writes deepseek/figures/l18-v32-bill.svg
"""

from l18_svg import SVG
from l18_data import K, RAW, flops_v32, kv_v32

POS = [8, 32, 128, 256, 512, 1024]           # in K = 1024 tokens
data = []
for p in POS:
    f = flops_v32(p * K, RAW)
    data.append((p, f["proj"] / 1e12, f["core"] / 1e12, f["idx"] / 1e12, kv_v32() * p * K / 1e9))
tot = [a + b + c for _, a, b, c, _ in data]
assert round(tot[-1], 3) == 1.157 and round(data[-1][3] / tot[-1], 3) == 0.906
assert round(data[-1][4], 1) == 50.4 and round(data[2][4], 2) == 6.30

W, H = 740, 450
svg = SVG(W, H, "V3.2 外推到 1M：单 token FLOPs 的三块与每个请求的 KV cache")
X0, X1, Y0, Y1 = 70, 520, 72, 366
YMAX = 1.2


def py(v):
    return Y1 - v / YMAX * (Y1 - Y0)


svg.text(20, 26, "单 token FLOPs（原始计数，TFLOPs）", size=13, weight="700")
for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
    y = py(v)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "%.1f" % v, size=11, anchor="end", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)

slot = (X1 - X0) / len(POS)
bw = 44
for n, ((p, a, b, c, kv), t) in enumerate(zip(data, tot)):
    cx = X0 + slot * (n + 0.5)
    x = cx - bw / 2
    extrap = p > 128
    op = 0.45 if extrap else 0.8
    y = Y1
    for v, col in ((a, "muted"), (b, "blue"), (c, "violet")):
        h = v / YMAX * (Y1 - Y0)
        if h > 0.05:
            svg.rect(x, y - h, bw, h, fill=col, fill_opacity=op, stroke=col, width=0.8,
                     dash="4 3" if extrap else None)
        y -= h
    svg.text(cx, y - 24, "%.3f" % t, size=12, anchor="middle", weight="700")
    svg.text(cx, y - 8, "KV %s GB" % ("%.2f" % kv if kv < 10 else "%.1f" % kv), size=11,
             anchor="middle", color="amber")
    svg.text(cx, Y1 + 18, "%dK" % p if p < 1024 else "1M", size=12, anchor="middle")
svg.text((X0 + X1) / 2, Y1 + 40, "查询位置 t（K = 1024 个 token）", size=12, anchor="middle", color="muted")

# 128K boundary
bx = X0 + slot * 3
svg.line(bx, Y0 - 6, bx, Y1, color="red", width=1.4, dash="5 4")
svg.text(bx - 6, Y0 + 14, "V3.2 官方上限 128K", size=12, anchor="end", color="red")
svg.text(bx + 6, Y0 + 36, "右侧为外推（虚线框）", size=12, color="red")

# legend + notes
LX = 548
items = [("muted", "投影与专家 2N（固定 0.074）"),
         ("blue", "核心注意力（t ≥ 2048 后固定 0.035）"),
         ("violet", "lightning indexer（扫全部前文，∝ t）")]
ly = 96
for col, lab in items:
    svg.rect(LX, ly - 11, 14, 14, fill=col, fill_opacity=0.8, stroke=col, width=0.8)
    words = lab.split("（")
    svg.text(LX + 22, ly, words[0], size=12)
    if len(words) > 1:
        svg.text(LX + 22, ly + 17, "（" + words[1], size=11, color="muted")
    ly += 50
svg.text(LX, ly + 6, "1M 处 indexer 占", size=12)
svg.text(LX, ly + 24, "%.1f%%" % (100 * data[-1][3] / tot[-1]), size=20, weight="700", color="violet")
svg.text(LX, ly + 46, "琥珀色数字：一个请求", size=11, color="amber")
svg.text(LX, ly + 62, "在该长度的 KV cache", size=11, color="amber")
svg.text(LX, ly + 78, "（48,068 B/token）", size=11, color="amber")
svg.text(20, 436, "维度：V3.2 激活 37B，61 层，128 头（MQA 模式键 576 / 值 512），indexer 64 头 × 128 维。一次乘加计 2 次运算。",
         size=11, color="muted")

svg.save("l18-v32-bill.svg")
