"""Figure 18.14: how one request's KV cache is laid out (redrawn from Fig. 6 of the V4 report).

Research notes C48-C50 and F14: a fixed-size "state cache" per request holds every layer's
sliding-window entries (n_win = 128, 584 B each) plus each compressed layer's not-yet-compressed
tail; the growing "classic" KV cache is stored in blocks of lcm(4, 128) = 128 original tokens,
each holding k1 = 32 CSA entries + 32 indexer keys per CSA layer and k2 = 1 entry per HCA layer.
The byte numbers are V4-Pro's (30 CSA + 31 HCA layers, 61 layers in all); the token strip at the
bottom is an illustration for a request that has read 582 tokens.
Writes deepseek/figures/l18-kv-layout.svg
"""

from math import lcm

from l18_data import M_CSA, M_HCA, MODELS, N_WIN, V4_ENTRY, V4_IDX, kv_v4
from l18_svg import SVG

PRO = MODELS["pro"]
BLOCK = lcm(M_CSA, M_HCA)                      # 128 original tokens per block
K1, K2 = BLOCK // M_CSA, BLOCK // M_HCA        # 32 CSA entries, 1 HCA entry per block
SWA_BYTES = PRO["L"] * N_WIN * V4_ENTRY        # 4,559,872 B
CSA_BLOCK = K1 * (V4_ENTRY + V4_IDX)           # 20,864 B per CSA layer per block
BLOCK_BYTES = PRO["nC"] * CSA_BLOCK + PRO["nH"] * K2 * V4_ENTRY
assert (BLOCK, K1, K2) == (128, 32, 1)
assert SWA_BYTES == 4_559_872 and CSA_BLOCK == 20_864 and BLOCK_BYTES == 644_024
assert abs(BLOCK_BYTES - BLOCK * kv_v4("pro")["total"]) < 1

W, H = 760, 552
svg = SVG(W, H, "V4 一个请求的 KV cache 布局：固定大小的状态缓存 + 按块增长的经典缓存")

# ---------------------------------------------------------------- panel frames
PY0, PY1 = 34, 328
svg.rect(14, PY0, 292, PY1 - PY0, fill="amber", fill_opacity=0.05, stroke="amber", width=1.3, rx=8)
svg.rect(320, PY0, 426, PY1 - PY0, fill="amber", fill_opacity=0.05, stroke="amber", width=1.3, rx=8)
svg.text(24, 24, "① 状态缓存：每个请求一块，大小固定", size=13, weight="700", color="amber")
svg.text(330, 24, "② 经典 KV cache：随长度增长，按块存放", size=13, weight="700", color="amber")

# ---------------------------------------------------------------- panel 1: state cache
svg.text(28, 58, "滑动窗口条目：每层最近 128 个 token，未压缩", size=12)
rows = ["层 0", "层 1", "层 2", "⋮", "层 60"]
GX0, GX1, CELLS = 72, 290, 16                  # one cell = 8 tokens
for n, lab in enumerate(rows):
    y = 70 + n * 17
    svg.text(GX0 - 8, y + 10, lab, size=11, anchor="end", color="muted")
    if lab == "⋮":
        continue
    cw = (GX1 - GX0) / CELLS
    for k in range(CELLS):
        svg.rect(GX0 + k * cw + 0.5, y, cw - 1, 12, fill="amber", fill_opacity=0.28, rx=1.5)
svg.text(28, 172, "%d 层 × %d 个条目 × %d B" % (PRO["L"], N_WIN, V4_ENTRY), size=12)
svg.text(28, 190, "= {:,} B ≈ {:.2f} MB（Pro）".format(SWA_BYTES, SWA_BYTES / 1e6), size=12, weight="700")

svg.text(28, 222, "每个压缩层尚未凑满一块的尾部，未压缩", size=12)
TX0, TW = 92, 190                              # bar for a full 127-token HCA tail
svg.text(TX0 - 8, 245, "HCA 层", size=11, anchor="end", color="muted")
svg.rect(TX0, 235, TW, 13, fill="amber", fill_opacity=0.28, rx=2)
svg.text(TX0 + TW / 2, 262, "最多 %d 个 token" % (M_HCA - 1), size=11, anchor="middle", color="muted")
svg.text(TX0 - 8, 283, "CSA 层", size=11, anchor="end", color="muted")
svg.rect(TX0, 273, TW * (M_CSA - 1) / (M_HCA - 1), 13, fill="amber", fill_opacity=0.5, rx=1)
svg.text(TX0 + 14, 284, "最多 %d 个 token（同一比例尺）" % (M_CSA - 1), size=11, color="muted")
svg.text(28, 316, "与上下文长度无关：像状态空间模型的定长“状态”", size=11, color="amber")

# ---------------------------------------------------------------- panel 2: growing blocks
BX0, BW, BG, BY = 338, 72, 8, 62         # last block ends at x = 730, inside the panel (edge 746)
blocks = ["块 0", "块 1", "块 2", "…", "块 8191"]
for n, lab in enumerate(blocks):
    x = BX0 + n * (BW + BG)
    if lab == "…":
        svg.text(x + BW / 2, BY + 20, "…", size=14, anchor="middle")
        continue
    hl = n == 1
    svg.rect(x, BY, BW, 30, fill="orange", fill_opacity=0.22 if hl else 0.12, stroke="orange",
             width=1.8 if hl else 1.1, rx=4)
    svg.text(x + BW / 2, BY + 19, lab, size=12, anchor="middle")
svg.text(BX0 - 2, 52, "每块覆盖 lcm(4, 128) = %d 个原始 token；1M 上下文共 %d 块" % (BLOCK, 2 ** 20 // BLOCK),
         size=11, color="muted")

# zoom lines from block 1 to the detail box
DX0, DX1, DY0, DY1 = 332, 734, 120, 318
b1x = BX0 + BW + BG
svg.line(b1x, BY + 30, DX0 + 6, DY0, color="orange", width=1, dash="3 3")
svg.line(b1x + BW, BY + 30, DX1 - 6, DY0, color="orange", width=1, dash="3 3")
svg.rect(DX0, DY0, DX1 - DX0, DY1 - DY0, fill="surface", stroke="orange", width=1.1, rx=6)

CX0, CW = 346, 8
svg.text(CX0, DY0 + 20, "一个 CSA 层（Pro 有 %d 层），每块" % PRO["nC"], size=12, weight="700")
for k in range(K1):
    svg.rect(CX0 + k * CW, DY0 + 28, CW - 1.5, 16, fill="orange", fill_opacity=0.55, rx=1.5)
svg.text(CX0 + K1 * CW + 8, DY0 + 41, "%d 个条目 × %d B" % (K1, V4_ENTRY), size=11)
for k in range(K1):
    svg.rect(CX0 + k * CW, DY0 + 50, CW - 1.5, 8, fill="violet", fill_opacity=0.55, rx=1)
svg.text(CX0 + K1 * CW + 8, DY0 + 59, "%d 个 indexer 键 × %d B" % (K1, V4_IDX), size=11)
svg.text(CX0, DY0 + 80, "= {:,} B".format(CSA_BLOCK), size=11, color="muted")

svg.text(CX0, DY0 + 110, "一个 HCA 层（Pro 有 %d 层），每块" % PRO["nH"], size=12, weight="700")
svg.rect(CX0, DY0 + 118, CW - 1.5, 16, fill="orange", fill_opacity=0.55, rx=1.5)
svg.text(CX0 + CW + 8, DY0 + 131, "%d 个条目 × %d B = %d B" % (K2, V4_ENTRY, K2 * V4_ENTRY), size=11)

svg.line(CX0, DY0 + 150, DX1 - 14, DY0 + 150, color="rule", width=1)
svg.text(CX0, DY0 + 170, "每块合计（Pro）：{} × {:,} + {} × {} = {:,} B".format(
    PRO["nC"], CSA_BLOCK, PRO["nH"], V4_ENTRY, BLOCK_BYTES), size=12, weight="700")
svg.text(CX0, DY0 + 188, "= 128 × 5,031.44 B，正是 (18.25) 的每 token 字节数", size=11, color="muted")

# ---------------------------------------------------------------- panel 3: where one request's tokens go
N_READ = 582                                    # tokens 0 .. 581 have been read; t = 581
T = N_READ - 1
SPAN = 5 * BLOCK                                # draw 0 .. 640
AX0, AX1 = 40, 720
ppt = (AX1 - AX0) / SPAN


def X(tok):
    return AX0 + tok * ppt


hca_tail = N_READ % M_HCA                       # 70 tokens not yet in an HCA entry
csa_tail = N_READ % M_CSA                       # 2 tokens not yet in a CSA entry
swa0 = N_READ - N_WIN                           # window = 454 .. 581
done = N_READ - hca_tail                        # 512: end of the last complete block
assert (hca_tail, csa_tail, swa0, done) == (70, 2, 454, 512)

svg.text(24, 360, "③ 一个请求的历史分到哪里（示意：已读 %d 个 token，当前位置 t = %d）" % (N_READ, T),
         size=13, weight="700")
BY3 = 402
for b in range(done // BLOCK):
    svg.rect(X(b * BLOCK), BY3, BLOCK * ppt, 22, fill="orange", fill_opacity=0.20, stroke="orange", width=1)
    svg.text(X(b * BLOCK + BLOCK / 2), BY3 + 15, "块 %d" % b, size=11, anchor="middle")
# the block being filled: CSA entries exist for complete groups of 4, no HCA entry yet
svg.rect(X(done), BY3, (N_READ - csa_tail - done) * ppt, 22, fill="orange", fill_opacity=0.08)
svg.rect(X(done), BY3, BLOCK * ppt, 22, stroke="orange", width=1, dash="4 3")
svg.rect(X(N_READ - csa_tail), BY3, csa_tail * ppt, 22, fill="amber", fill_opacity=0.6)
svg.text(X((done + N_READ - csa_tail) / 2), BY3 + 15, "块 4", size=11, anchor="middle")
svg.text(X((N_READ + done + BLOCK) / 2), BY3 + 15, "（未满）", size=11, anchor="middle", color="muted")
svg.line(X(N_READ), BY3 - 6, X(N_READ), BY3 + 28, color="blue", width=2)
svg.text(X(N_READ) + 4, BY3 - 8, "t = %d" % T, size=11, color="blue")
for tok in (0, 128, 256, 384, 512, 640):
    svg.text(X(tok), BY3 + 38, str(tok), size=11, anchor="middle", color="muted")

# brace above: what is in the classic cache
svg.line(X(0), 388, X(done), 388, color="orange", width=2.5, cap="round")
svg.text(X(done / 2), 382, "已完成的 %d 块：CSA 条目、indexer 键、HCA 条目 → ②" % (done // BLOCK),
         size=11, anchor="middle", color="orange")


def bracket(a, b, y, color, label):
    svg.line(X(a), y, X(b), y, color=color, width=2.5, cap="round")
    svg.line(X(a), y, X(a), y - 6, color=color, width=1.3)
    svg.line(X(b), y, X(b), y - 6, color=color, width=1.3)
    svg.text(X(a) - 6, y + 4, label, size=11, anchor="end", color=color)


bracket(swa0, N_READ, 462, "amber", "最近 %d 个 token 的滑动窗口条目 → ①" % N_WIN)
bracket(done, N_READ, 484, "amber", "HCA 尾部 %d mod %d = %d 个 → ①" % (N_READ, M_HCA, hca_tail))
bracket(N_READ - csa_tail, N_READ, 506, "amber", "CSA 尾部 %d mod %d = %d 个 → ①" % (N_READ, M_CSA, csa_tail))
svg.text(24, 538, "块 4 里已凑满的 %d 组（每组 4 个 token）已有 CSA 条目，属于 ②；它的 HCA 条目要等凑满 128 个才生成。"
         % ((N_READ - csa_tail - done) // M_CSA), size=11, color="muted")

svg.save("l18-kv-layout.svg")
