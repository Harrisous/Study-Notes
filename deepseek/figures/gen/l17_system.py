"""Figure 17.10: the system side of Engram (redrawn after Engram fig. 2, with V4.1 details).

(a) Training: tables sharded by row across GPUs; an All-to-All gathers the active rows in the forward
    pass and sends the gradients back in the backward pass (Engram §2.5; V4.1 §3.1.3).
(b) Inference: the row indices depend only on token IDs, so they are known before the forward pass;
    the rows are prefetched from host DRAM (PCIe in the paper, background RDMA in V4.1) while the GPU
    computes the first block; the Engram module in block 2 then uses them (Engram §2.5, §6.4).
    The timeline is schematic (no durations are measured in the papers).
Numbers: rows per token 2 layers x 2 orders x 8 heads = 32 (27B), 2 x 3 x 8 = 48 (V4.1);
V4.1 bytes per token 48 x (256 + 8) = 12,672 (l17_num); throughput from Engram table 4.
Writes deepseek/figures/l17-system.svg
"""

from l17_svg import SVG
from l17_num import THR, LOSS, V41_FETCH

assert V41_FETCH == 12_672

W, H = 760, 548
svg = SVG(W, H, "Engram 的系统实现：训练时分片 + All-to-All，推理时从主机内存预取")

# ================================================================= (a) training
svg.text(20, 22, "(a) 训练：表按行切到各卡，前向用 All-to-All 取行，反向把梯度送回持有该行的卡", size=13, weight="700")
GX = [24, 204, 384, 564]
GW = 164
for g, x in enumerate(GX):
    svg.rect(x, 36, GW, 110, fill="ink", fill_opacity=0.03, stroke="rule", width=1.2, rx=6)
    svg.text(x + 10, 54, "GPU %d" % (g + 1), size=12.5, weight="700")
    svg.box(x + 8, 64, 74, 48, ["骨干", "本卡 token"], color="blue", fill_opacity=0.10, size=11.5)
    svg.box(x + 88, 64, 68, 48, ["表第 %d 片" % (g + 1), "1/4 的行"], color="amber", fill_opacity=0.2, size=11.5)
BUSY = 170
svg.rect(24, BUSY, 704, 24, fill="ink", fill_opacity=0.04, stroke="ink", width=1.2, rx=5)
svg.text(376, BUSY + 16.5, "All-to-All：紫 = 行号请求（反向时是梯度）；琥珀 = 取回的行向量",
         size=12, anchor="middle")
for x in GX:
    svg.arrow(x + 30, 114, x + 30, BUSY - 2, color="violet", width=1.4)
    svg.arrow(x + 60, BUSY, x + 60, 116, color="amber", width=1.4)
    svg.arrow(x + 106, BUSY, x + 106, 116, color="violet", width=1.4)
    svg.arrow(x + 136, 114, x + 136, BUSY - 2, color="amber", width=1.4)
svg.text(24, 222, "V4.1-Flash：表按行切到专门的 engram 并行组，优化器状态再按副本切分；每个流水段开始本步的微批之前就对整个本地 batch 发起预取。",
         size=11.5, color="muted")

# ================================================================= (b) inference timeline
TY = 262
svg.text(20, TY, "(b) 推理：行号只由 token ID 决定，开算之前就能预取（示意时间线，论文未报告各段耗时）", size=13, weight="700")
T0, T1 = 118, 560


def t(v):                         # schematic time 0..100
    return T0 + v / 100 * (T1 - T0)


lanes = [("主机 CPU", 290), ("PCIe / RDMA", 334), ("GPU", 378)]
for name, yy in lanes:
    svg.text(T0 - 10, yy + 19, name, size=12, anchor="end", color="muted")
    svg.line(T0, yy + 32, T1, yy + 32, color="grid", width=1)
svg.box(t(0), 290, t(9) - t(0), 28, ["哈希"], color="violet", fill_opacity=0.14, size=11.5)
svg.box(t(9), 290, t(34) - t(9), 28, ["从 DRAM 读行"], color="amber", fill_opacity=0.18, size=11.5)
svg.box(t(18), 334, t(44) - t(18), 28, ["拷到 GPU"], color="amber", fill_opacity=0.32, size=11.5)
svg.box(t(0), 378, t(10) - t(0), 28, ["嵌入"], color="blue", fill_opacity=0.08, size=11.5)
svg.box(t(10), 378, t(50) - t(10), 28, ["第 1 块（注意力 + FFN）"], color="blue", fill_opacity=0.14, size=11.5)
svg.box(t(50), 378, t(58) - t(50), 28, ["E"], color="amber", fill_opacity=0.55, size=11.5, weight="700")
svg.box(t(58), 378, t(92) - t(58), 28, ["第 2 块其余部分"], color="blue", fill_opacity=0.14, size=11.5)
svg.text(t(96), 396, "…", size=14, anchor="middle")
svg.path("M%s %s V%s" % (t(44), 362, 376), stroke="amber", width=1.6, arrow=True)
# overlap bracket
svg.path("M%s 428 v6 H%s v-6" % (t(10), t(50)), stroke="ink", width=1.2)
svg.text(t(30), 450, "预取被第 1 块的计算“盖住”", size=12, anchor="middle")
svg.text(t(54), 450, "E = 第 2 块里的 Engram", size=11.5, color="amber")
svg.text(T0, 480, "对比 MoE：某一层选哪些专家，要等这一层的 *h*_{t} 算出来才知道，没法这样提前取。", size=12, color="blue")
svg.text(T0, 500, "每 token 取的行数只由“层数 × 阶数 × 头数”决定：27B 为 32 行，V4.1-Flash 为 48 行（%s 字节，含缩放）。" % "{:,}".format(V41_FETCH),
         size=12)
svg.text(T0, 520, "与表有多大无关：表大 3 倍（Engram-40B），每 token 仍取 32 行。", size=12)

# throughput card
CX = 586
svg.rect(CX, 286, 160, 158, fill="ink", fill_opacity=0.03, stroke="rule", width=1.2, rx=6)
svg.text(CX + 10, 306, "实测（论文表 4）", size=12.5, weight="700")
svg.text(CX + 10, 324, "100B 参数表全在主机内存", size=11, color="muted")
yy = 348
for name, (a, b) in THR.items():
    svg.text(CX + 10, yy, name, size=12, weight="700")
    svg.text(CX + 10, yy + 18, "{:,.2f} → {:,.2f}".format(a, b), size=11.5)
    svg.text(CX + 150, yy, "−%.2f%%" % (LOSS[name] * 100), size=12, anchor="end", color="red", weight="700")
    yy += 44
svg.text(CX + 10, 432, "token/s；稠密骨干，H800", size=11, color="muted")

svg.save("l17-system.svg")
