"""Figure 14.14: what DSA does to KV bytes: storage goes up, per-step reads go down.

Per token per layer: MLA entry in BF16 = 576 * 2 = 1152 B; V3.2 FP8 format = 512 FP8 + 4 FP32
scales + 64 BF16 = 656 B (FlashMLA README); indexer key = 128 FP8 + 1 FP32 scale = 132 B.
Decode reads per step per layer at position T: dense FP8 656 T, dense BF16 1152 T,
DSA 656 * min(T, 2048) + 132 T. All values computed below.
Writes deepseek/figures/l14-kv-bytes.svg
"""

from l14_svg import SVG

BF16, FP8, IDX, K = 576 * 2, 512 + 16 + 128, 128 + 4, 2048
assert (BF16, FP8, IDX) == (1152, 656, 132)


def dsa_read(T):
    return FP8 * min(T, K) + IDX * T


T128 = 131072
assert round(FP8 * T128 / 1e6, 1) == 86.0 and round(dsa_read(T128) / 1e6, 1) == 18.6
assert round(BF16 * T128 / 1e6, 1) == 151.0
X_FP8 = FP8 * K / (FP8 - IDX)          # where DSA reads equal dense FP8 reads
assert round(X_FP8) == 2564

W, H = 760, 424
svg = SVG(W, H, "DSA 让每 token 缓存变大 20%，却让每步解码的读取量降到约 1/4.6")

# ---------------------------------------------------------------- (a) storage per token per layer
AX0, AY0 = 20, 30
svg.text(AX0, 22, "(a) 每 token 每层存多少字节", size=13, weight="700")
BX, BW = 150, 190
rows = [
    ("MLA，BF16", [(BF16, "amber", "1152")]),
    ("MLA，FP8 格式", [(512, "orange", "512 FP8"), (16, "muted", ""), (128, "red", "128")]),
    ("DSA = FP8 + 索引键", [(512, "orange", ""), (16, "muted", ""), (128, "red", ""), (IDX, "violet", "132")]),
]
scale = BW / 1200
for i, (lab, segs) in enumerate(rows):
    y = AY0 + 30 + i * 62
    svg.text(BX - 10, y + 20, lab, size=12, anchor="end")
    x = BX
    for v, col, t in segs:
        svg.rect(x, y + 4, v * scale, 26, fill=col, fill_opacity=0.55, stroke=col, width=0.8)
        if t and v * scale > 30:
            svg.text(x + v * scale / 2, y + 22, t, size=11, anchor="middle")
        x += v * scale
    total = sum(v for v, _, _ in segs)
    svg.text(x + 6, y + 22, "%d B" % total, size=12, weight="700")
svg.text(BX - 130, AY0 + 222, "FP8 格式：512 个 FP8（内容）+ 4 个 FP32 缩放", size=11, color="muted")
svg.text(BX - 130, AY0 + 238, "+ 64 个 BF16（RoPE 部分，不量化）= 656 B", size=11, color="muted")
svg.text(BX - 130, AY0 + 262, "DSA 比不带索引键多存 132 B，即 +20.1%；", size=12, color="violet")
svg.text(BX - 130, AY0 + 280, "任何历史 token 都可能被选中，一个都不能丢。", size=12, color="violet")

# ---------------------------------------------------------------- (b) decode reads per step per layer
PX0, PX1, PY0, PY1 = 450, 740, 56, 330
YM = 160e6


def px(T):
    return PX0 + T / T128 * (PX1 - PX0)


def py(v):
    return PY1 - v / YM * (PY1 - PY0)


svg.text(400, 22, "(b) 位置 T 解码一步，每层读多少字节", size=13, weight="700")
for v in range(0, 161, 40):
    y = py(v * 1e6)
    svg.line(PX0, y, PX1, y, color="grid", width=1)
    svg.text(PX0 - 8, y + 4, "%d MB" % v, size=11, anchor="end", color="muted")
for xk in (0, 32, 64, 96, 128):
    svg.text(px(xk * 1024), PY1 + 18, "%dK" % xk, size=11, anchor="middle", color="muted")
svg.line(PX0, PY1, PX1, PY1, color="muted", width=1.2)
svg.line(PX0, PY0, PX0, PY1, color="muted", width=1.2)
svg.line(px(0), py(0), px(T128), py(BF16 * T128), color="amber", width=2.2, dash="6 4")
svg.line(px(0), py(0), px(T128), py(FP8 * T128), color="blue", width=2.4)
pts = [(0, 0), (K, dsa_read(K)), (T128, dsa_read(T128))]
svg.path("M" + " L".join("%.1f %.1f" % (px(a), py(b)) for a, b in pts), stroke="green", width=3)
svg.text(px(T128) - 4, py(BF16 * T128) - 8, "稠密 BF16：151.0", size=12, anchor="end", color="amber", weight="700")
svg.text(px(T128) - 4, py(FP8 * T128) - 8, "稠密 FP8：86.0", size=12, anchor="end", color="blue", weight="700")
svg.text(px(T128) - 4, py(dsa_read(T128)) - 10, "DSA：18.6", size=12, anchor="end", color="green", weight="700")
svg.text(PX0 + 8, PY0 + 12, "128K 处：FP8 读取降到 1/4.6，", size=12)
svg.text(PX0 + 8, PY0 + 30, "相对 BF16 稠密核降到 1/8.1", size=12)
svg.text((PX0 + PX1) / 2, PY1 + 38, "位置 T", size=12, anchor="middle", color="muted")
svg.text(20, 394, "DSA 的读取 = 2048 × 656（被选条目）+ T × 132（全部索引键）。T < %d 时，DSA 读得比稠密 FP8 还多。" % round(X_FP8),
         size=12, color="muted")
svg.text(20, 414, "字节格式来自 FlashMLA README 与推理代码；读取量是按格式计算的模型，不是实测。", size=11, color="muted")

svg.save("l14-kv-bytes.svg")
