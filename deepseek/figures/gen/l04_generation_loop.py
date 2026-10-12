"""Figure 4.1: the generation loop (prefill once, then one token per decode step) and where the time goes.

Top: schematic of prefill + three decode steps writing and reading the KV cache.
Bottom: PaLM 540B on 64 TPU v4 chips, Pope et al. 2023 Table 2 (lesson-04-doc C78):
  low-latency  : prefill 2048 tokens, batch 1, 0.29 s, MFU 43% | decode 64 tokens, batch 64, 1.82 s, MFU 14%
  high-through.: prefill 2048 tokens, batch 512, 85.2 s, MFU 76% | decode 64 tokens, batch 512, 6.0 s, MFU 33%
Bars show the share of tokens (per sequence) and the share of wall-clock time, computed below.
Writes deepseek/figures/l04-generation-loop.svg
"""

from l04_svg import SVG

CONF = [
    ("低延迟配置", 2048, 64, 0.29, 1.82, 43, 14, "prefill batch 1，decode batch 64"),
    ("高吞吐配置", 2048, 64, 85.2, 6.0, 76, 33, "prefill、decode 都是 batch 512"),
]
assert round(0.29 / 2048 * 1e3, 3) == 0.142 and round(1.82 / 64 * 1e3, 1) == 28.4

W, H = 720, 560
svg = SVG(W, H, "生成循环：prefill 一次、decode 每步一个 token，以及时间花在哪里")

# ------------------------------------------------------------------ top: schematic
svg.text(16, 24, "Ⓐ 生成循环：提示一次性 prefill，之后每个 decode 步只喂进 1 个 token", size=14, weight="600")
prompt = ["讲", "个", "故", "事"]
outs = ["从", "前", "有", "座"]
TK = 26  # token box size


def tok(x, y, s, col, fill_op=0.16):
    svg.rect(x, y, TK, TK, fill=col, fill_opacity=fill_op, stroke=col, width=1.2, rx=4)
    svg.text(x + TK / 2, y + TK / 2 + 5, s, size=13, anchor="middle", color="ink")


# prompt tokens
px0, py0 = 36, 44
for i, s in enumerate(prompt):
    tok(px0 + i * (TK + 6), py0, s, "blue")
svg.text(px0 + 3 * (TK + 6) - 4, py0 + TK + 16, "提示：4 个 token", size=12, anchor="end", color="muted")
# prefill block
bx, by, bw, bh = 20, 96, 200, 62
svg.rect(bx, by, bw, bh, fill="blue", fill_opacity=0.10, stroke="blue", width=1.6, rx=8)
svg.text(bx + bw / 2, by + 24, "prefill：一次前向", size=13, anchor="middle", weight="600", color="blue")
svg.text(bx + bw / 2, by + 44, "4 个位置并行（矩阵 × 矩阵）", size=12, anchor="middle")
svg.arrow(px0 + 3 * (TK + 6) + 13, py0 + TK + 3, px0 + 3 * (TK + 6) + 13, by - 2, color="blue", width=1.4)

# decode steps
SX = [250, 400, 550]
sw, sh = 136, 62
for k, x in enumerate(SX):
    svg.rect(x, by, sw, sh, fill="blue", fill_opacity=0.06, stroke="blue", width=1.2, rx=8, dash="5 3")
    svg.text(x + sw / 2, by + 24, "decode 第 %d 步" % (k + 1), size=13, anchor="middle", weight="600", color="blue")
    svg.text(x + sw / 2, by + 44, "1 个位置：矩阵 × 向量", size=12, anchor="middle")
# sampled tokens: produced by one block, fed into the next one
OY = py0
for k, s in enumerate(outs):
    xo = (bx + bw + 6) if k == 0 else (SX[k - 1] + sw + 6)
    tok(xo, OY, s, "green", 0.2)
    src_x = (bx + bw - 24) if k == 0 else (SX[k - 1] + sw - 24)
    svg.arrow(src_x, by - 2, xo + 6, OY + TK + 2, color="green", width=1.3)
    if k < 3:
        svg.arrow(xo + TK - 4, OY + TK + 2, SX[k] + 30, by - 2, color="ink", width=1.3)

# KV cache snapshots under each block: old slots pale, slots written in this step dark
cy = 186
slot, gap = 14, 2
svg.text(bx, cy + 12, "KV cache", size=12, weight="600", color="amber")


def snapshot(x0, n_old, n_new, label):
    for i in range(n_old + n_new):
        new = i >= n_old
        svg.rect(x0 + i * (slot + gap), cy, slot, slot, fill="amber", fill_opacity=0.42 if new else 0.14,
                 stroke="amber", width=1.4 if new else 0.9, rx=2)
    svg.text(x0, cy + slot + 16, label, size=12, color="amber")


snapshot(bx + 70, 0, 4, "写入 4 格")
svg.arrow(bx + 120, by + bh + 2, bx + 120, cy - 3, color="amber", width=1.3)
for k, x in enumerate(SX):
    snapshot(x + 6, 4 + k, 1, "读 %d 格，写 1 格" % (4 + k))
    svg.arrow(x + sw / 2, by + bh + 2, x + sw / 2, cy - 3, color="amber", width=1.3)
    svg.arrow(x + sw / 2 + 22, cy - 3, x + sw / 2 + 22, by + bh + 2, color="amber", width=1.1, dash="3 2")

svg.line(16, 240, W - 16, 240, color="rule", width=1)

# ------------------------------------------------------------------ bottom: Pope et al. Table 2
svg.text(16, 266, "Ⓑ 时间花在哪里：PaLM 540B，64 块 TPU v4，提示 2048 token，生成 64 token", size=14, weight="600")
BX0, BX1 = 150, 580
yy = 290
for name, npre, ndec, tpre, tdec, mpre, mdec, note in CONF:
    svg.text(16, yy + 14, name, size=13, weight="600")
    rows = (("token 数", npre / (npre + ndec), "%d" % npre, "%d" % ndec),
            ("时间", tpre / (tpre + tdec), "%s s" % tpre, "%s s" % tdec))
    for r, (lab, frac, a, b) in enumerate(rows):
        y = yy + r * 28
        svg.text(BX0 - 8, y + 15, lab, size=12, anchor="end", color="muted")
        xm = BX0 + frac * (BX1 - BX0)
        svg.rect(BX0, y, xm - BX0, 20, fill="blue", fill_opacity=0.45, rx=2)
        svg.rect(xm, y, BX1 - xm, 20, fill="amber", fill_opacity=0.55, rx=2)
        if xm - BX0 >= 130:
            svg.text(BX0 + 6, y + 15, "prefill %s（%.0f%%）" % (a, 100 * frac), size=12, weight="600")
        else:
            # a 14% segment is ~60 units wide: the full label would run into the amber part
            svg.text(BX0 + 5, y + 15, a, size=12, weight="600")
        tx = BX1 + 6
        svg.text(tx, y + 15, "decode %s（%.0f%%）" % (b, 100 * (1 - frac)), size=12, weight="600", color="amber")
    svg.text(BX0, yy + 72, "MFU：prefill %d%%，decode %d%%；%s" % (mpre, mdec, note), size=12)
    yy += 110
svg.text(16, H - 36, "同一个请求里，decode 只处理约 3% 的 token；低延迟配置下却占了 86% 的时间，", size=12, color="muted")
svg.text(16, H - 18, "而且硬件利用率（MFU）只有 prefill 的三分之一左右。数据：Pope 等人 2023 表 2。", size=12, color="muted")
svg.save("l04-generation-loop.svg")
