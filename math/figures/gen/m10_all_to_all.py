"""Figure M10.11: all-to-all as a transpose of blocks, and the bytes one DeepSeek-V3 token moves.

Left: p = 4 GPUs.  Before the all-to-all, GPU i holds p blocks, block j destined for GPU j; after it,
GPU j holds block j from every GPU (the block matrix is transposed).  The pairwise-exchange
algorithm (Thakur et al. 2005, section 4.3) does it in p - 1 steps: in step k, GPU i swaps with GPU i XOR k.
Right: bytes per token for the MoE dispatch / combine with DeepSeek-V3's hidden size 7168
(research notes N80, an illustration): dispatch in FP8 with one FP32 scale per 128 values
= 7168 + 56*4 = 7392 bytes per copy; combine in BF16 = 14336 bytes per copy.
8 copies (one per routed expert) vs at most 4 copies over InfiniBand under node-limited routing.
Writes math/figures/m10-all-to-all.svg
"""

from m10_svg import SVG

p = 4
W, H = 740, 350
svg = SVG(W, H, "all-to-all：块矩阵转置，以及 V3 一个 token 的通信字节")
C = 34


def blocks(x, y, title, transpose):
    svg.text(x + p * C / 2, y - 12, title, size=12, anchor="middle", weight="600")
    for i in range(p):
        svg.text(x - 8, y + i * C + C / 2 + 4, "GPU %d" % i, size=11, anchor="end", color="muted")
        for j in range(p):
            src, dst = (j, i) if transpose else (i, j)
            diag = src == dst
            svg.rect(x + j * C, y + i * C, C - 3, C - 3, fill="muted" if diag else "blue",
                     fill_opacity=0.25 if diag else 0.12 + 0.16 * dst, rx=3)
            svg.text(x + j * C + (C - 3) / 2, y + i * C + C / 2 + 3, "%d→%d" % (src, dst), size=11, anchor="middle")


blocks(88, 70, "发送前：第 i 行是 GPU i 要发出的块", False)
blocks(322, 70, "发送后：GPU j 收齐所有“→j”", True)
svg.arrow(88 + p * C + 8, 70 + p * C / 2, 322 - 56, 70 + p * C / 2, color="ink", width=1.8)
svg.text(20, 26, "① all-to-all：每张卡给其余每张卡各发一块不同的数据", size=13, weight="600")
svg.text(20, 232, "成对交换：第 k 步，GPU i 与 GPU (i XOR k) 互换一块，", size=12)
svg.text(20, 250, "共 p − 1 = 3 步。灰色对角块留在本卡，不上网络。", size=12)
svg.text(20, 274, "带宽项只看每卡要发出的总字节 S：", size=12, weight="600")
svg.text(20, 292, "时间 ≈ (p − 1)α + Sβ", size=12, weight="600")

# ---------------- right: V3 bytes per token
RX = 484
svg.text(RX - 10, 26, "② V3 一个 token 的 MoE 通信（示意）", size=13, weight="600")
disp, comb = 7168 + 56 * 4, 7168 * 2
rows = [("dispatch，8 份", 8 * disp, "blue"), ("combine，8 份", 8 * comb, "green"),
        ("跨节点 dispatch，≤ 4 份", 4 * disp, "blue"), ("跨节点 combine，≤ 4 份", 4 * comb, "green")]
scale = 180 / (8 * comb)
for k, (lab, b, col) in enumerate(rows):
    y = 62 + k * 50
    svg.text(RX - 10, y, lab, size=12)
    svg.rect(RX - 10, y + 7, b * scale, 18, fill=col, fill_opacity=0.55 if k < 2 else 0.85, rx=2)
    svg.text(RX - 10 + b * scale + 6, y + 21, "%.1f KiB" % (b / 1024), size=12, weight="600")
svg.text(RX - 10, 272, "每份 dispatch：FP8 的 7168 字节", size=11, color="muted")
svg.text(RX - 10, 288, "+ 每 128 个数一个 FP32 缩放 = 7392 字节；", size=11, color="muted")
svg.text(RX - 10, 304, "每份 combine：BF16，14,336 字节。", size=11, color="muted")
svg.text(RX - 10, 320, "节点受限路由把跨节点的份数压到 4 以内。", size=11, color="muted")

svg.save("m10-all-to-all.svg")
print("dispatch", disp, "combine", comb, "8 copies KiB", 8 * disp / 1024, 8 * comb / 1024,
      "4 copies KiB", 4 * disp / 1024, 4 * comb / 1024)
