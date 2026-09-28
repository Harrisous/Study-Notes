"""Figure 3.13 of the blackboard version of lesson 3: the memory bill counted cell by cell.

Top: what one parameter costs in mixed-precision Adam training (ZeRO paper sec. 3.1): 16-bit weight 2 B,
16-bit gradient 2 B, fp32 master weight 4 B, momentum 4 B, variance 4 B = 16 B; a frozen parameter keeps
only its 16-bit weight, 2 B. One cell = 2 bytes.
Bottom: the four post-training recipes with every network at 1B parameters (GB = 1e9 bytes), block widths
proportional to bytes, and the 7B totals of the lesson (x 7). Only model states are counted.
Writes deepseek/figures/l03b-bytes-board.svg
"""

from l03_svg import SVG

TRAIN, FROZEN = 16, 2
# (name, precision, bytes, colour)
PARTS = [("权重", "16 位", 2, "blue"), ("梯度", "16 位", 2, "blue"), ("主权重", "32 位", 4, "amber"),
         ("动量", "32 位", 4, "amber"), ("方差", "32 位", 4, "amber")]
assert sum(b for _, _, b, _ in PARTS) == TRAIN

NET = {"policy": ("策略", "blue", TRAIN), "value": ("价值", "violet", TRAIN),
       "ref": ("参考", "muted", FROZEN), "rm": ("奖励", "green", FROZEN)}
RECIPES = [("SFT", ["policy"]), ("DPO", ["policy", "ref"]), ("GRPO 式", ["policy", "ref", "rm"]),
           ("PPO", ["policy", "value", "ref", "rm"])]
totals = [sum(NET[k][2] for k in keys) for _, keys in RECIPES]
assert totals == [16, 18, 20, 36]
assert [7 * t for t in totals] == [112, 126, 140, 252]

W, H = 740, 404
svg = SVG(W, H, "黑板上的显存账：先数一个参数的字节，再数四种做法各放了几个网络")

# ---------------- top: one parameter, one cell = 2 bytes
CELL, CH = 44, 26
svg.text(16, 26, "一个参数占几个字节（每格 2 字节）", size=13.5, weight="700")
y = 44
svg.text(16, y + 18, "训练的参数", size=12.5, weight="600")
x = 104
for name, prec, b, col in PARTS:
    n = b // 2
    for k in range(n):
        svg.rect(x + k * CELL, y, CELL - 3, CH, fill=col, fill_opacity=0.25 if col == "blue" else 0.18,
                 stroke=col, width=1, rx=3)
        svg.text(x + k * CELL + (CELL - 3) / 2, y + 18, "2", size=12, anchor="middle")
    cx = x + n * CELL / 2 - 1.5
    svg.text(cx, y + CH + 16, name, size=11.5, anchor="middle", color="muted")
    svg.text(cx, y + CH + 31, "%s · %d 字节" % (prec, b) if n > 1 else prec, size=11, anchor="middle",
             color="muted")
    x += n * CELL + 10
svg.text(x + 6, y + 18, "= 16 字节", size=13, weight="700")

y2 = 124
svg.text(16, y2 + 18, "冻结的参数", size=12.5, weight="600")
svg.rect(104, y2, CELL - 3, CH, fill="muted", fill_opacity=0.18, stroke="muted", width=1, rx=3)
svg.text(104 + (CELL - 3) / 2, y2 + 18, "2", size=12, anchor="middle")
svg.text(104 + CELL + 10, y2 + 18, "= 2 字节（只存 16 位权重，没有梯度和优化器状态）", size=12.5)
svg.line(10, 166, W - 10, 166, color="rule", width=1)

# ---------------- bottom: four recipes, every network 1B parameters
svg.text(16, 192, "每个网络 1B 参数时的模型状态（GB）", size=13.5, weight="700")
svg.text(W - 16, 192, "× 7 → 7B", size=12.5, anchor="end", color="muted", weight="600")
UNIT = 9.0          # pixels per GB
X0 = 104
for i, (label, keys) in enumerate(RECIPES):
    yy = 210 + i * 42
    svg.text(16, yy + 19, label, size=12.5, weight="600")
    x = X0
    for k in keys:
        name, col, gb = NET[k]
        w = gb * UNIT
        svg.rect(x, yy, w - 2, 28, fill=col, fill_opacity=0.28 if gb == TRAIN else 0.18, stroke=col,
                 width=1.1, rx=3)
        txt = "%s %d" % (name, gb) if gb == TRAIN else "%d" % gb
        svg.text(x + (w - 2) / 2, yy + 19, txt, size=12 if gb == TRAIN else 11.5, anchor="middle")
        x += w
    svg.text(X0 + 36 * UNIT + 16, yy + 19, "= %d GB" % totals[i], size=13, weight="700")
    svg.text(W - 16, yy + 19, "%d GB" % (7 * totals[i]), size=13, anchor="end", weight="700", color="blue")

# legend (the narrow frozen blocks carry only their number)
LEG = {"policy": "策略", "value": "价值网络", "ref": "参考模型", "rm": "奖励模型"}
ly = 386
for k, key in enumerate(("policy", "value", "ref", "rm")):
    name, col, gb = NET[key]
    lx = 16 + k * 142
    svg.rect(lx, ly - 10, 14, 12, fill=col, fill_opacity=0.28 if gb == TRAIN else 0.18, stroke=col, width=1)
    svg.text(lx + 20, ly, "%s（%s）" % (LEG[key], "训练 16" if gb == TRAIN else "冻结 2"), size=11.5)
svg.text(W - 16, 26, "只算模型状态：不含激活值与采样用的 KV cache", size=11.5, anchor="end", color="muted")
svg.save("l03b-bytes-board.svg")
