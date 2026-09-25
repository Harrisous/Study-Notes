"""Figure M10.2: the memory and interconnect ladder, bandwidth on a log axis.

On-chip and HBM numbers: H100 SXM5, FlashAttention-3 (arXiv:2407.08608v2) Table 1:
  SMEM 228 KiB per SM, 31 TB/s per GPU; L2 50 MiB @ 12 TB/s; GMEM (HBM) 80 GiB @ 3.35 TB/s;
  RMEM 256 KiB per SM (no bandwidth given).
Interconnect: DeepSeek-V3 report, H800 cluster: NVLink ~160 GB/s, InfiniBand 50 GB/s per GPU.
Right column: the "ridge" of each level, F / BW with F = 989 TFLOP/s (dense BF16/FP16, H100 SXM5).
Writes math/figures/m10-memory-hierarchy.svg
"""

import math

from m10_svg import SVG

F = 989e12
SM = 132
levels = [
    # name, capacity text, bandwidth (B/s) or None, colour, group
    ("寄存器 RMEM", "每 SM 256 KiB（全卡约 %d MiB）" % round(256 * SM / 1024), None, "amber", "片上"),
    ("共享内存 SMEM", "每 SM 228 KiB（全卡约 %d MiB）" % round(228 * SM / 1024), 31e12, "amber", "片上"),
    ("L2 缓存", "50 MiB", 12e12, "amber", "片上"),
    ("显存 HBM", "80 GiB", 3.35e12, "amber", "片外"),
    ("NVLink（节点内）", "同节点另外 7 张卡", 160e9, "muted", "卡间"),
    ("InfiniBand（节点间）", "其他节点的卡", 50e9, "muted", "卡间"),
]

W, H = 740, 400
svg = SVG(W, H, "存储与互连的层级：带宽（对数刻度）与每一层的“脊点”")
LX = 44                       # label column
BX0, BX1 = 272, 566           # bar area
RX = 700                      # right column (anchor end)
TOP = 70
ROW = 44
LO, HI = 10.0, 14.0           # log10(B/s): 10 GB/s .. 100 TB/s


def bx(bw):
    return BX0 + (math.log10(bw) - LO) / (HI - LO) * (BX1 - BX0)


svg.text(LX, 26, "层级与容量", size=13, weight="600")
svg.text((BX0 + BX1) / 2, 26, "带宽（每秒字节，对数刻度）", size=13, weight="600", anchor="middle")
svg.text(RX, 26, "脊点 F/带宽", size=13, weight="600", anchor="end")
svg.text(RX, 43, "（FLOP/字节，F = 989 TFLOP/s）", size=11, anchor="end", color="muted")

# axis ticks
for e, lab in ((10, "10 GB/s"), (11, "100 GB/s"), (12, "1 TB/s"), (13, "10 TB/s"), (14, "100 TB/s")):
    x = bx(10 ** e)
    svg.line(x, TOP - 12, x, TOP + ROW * len(levels) - 8, color="grid", width=1)
    svg.text(x, TOP + ROW * len(levels) + 8, lab, size=11, anchor="middle", color="muted")

prev_group = None
for i, (name, cap, bw, col, grp) in enumerate(levels):
    y = TOP + i * ROW
    if grp != prev_group and i:
        svg.line(LX, y - 12, RX, y - 12, color="rule", width=1, dash="4 3")
    prev_group = grp
    svg.text(LX, y + 6, name, size=13, weight="600")
    svg.text(LX, y + 23, cap, size=11, color="muted")
    if bw is None:
        svg.text(BX0 + 6, y + 14, "（表中未给带宽）", size=11, color="muted")
        continue
    x1 = bx(bw)
    svg.rect(BX0, y - 4, x1 - BX0, 24, fill=col, fill_opacity=0.55 if col == "amber" else 0.45, rx=3)
    if bw >= 1e12:
        lab = "%g TB/s" % (bw / 1e12)
    else:
        lab = "%g GB/s" % (bw / 1e9)
    svg.text(x1 + 6, y + 13, lab, size=12, weight="600")
    svg.text(RX, y + 13, "{:,.0f}".format(F / bw) if F / bw >= 100 else "%.1f" % (F / bw), size=13,
             anchor="end", weight="600", color="blue" if F / bw < 1000 else "red")

# group brackets on the far left with rotated names
tags = [("片上", 0, 2), ("片外", 3, 3), ("卡间", 4, 5)]
for text, a, b in tags:
    ya = TOP + a * ROW - 6
    yb = TOP + b * ROW + 24
    svg.line(30, ya, 30, yb, color="muted", width=1.4)
    svg.text(20, (ya + yb) / 2, text, size=12, anchor="middle", color="muted", rotate=-90)
svg.text(20, H - 34, "从 SMEM 到 InfiniBand，带宽相差 %.0f 倍；脊点越大，这一层上的数据越“贵”：每搬 1 字节"
         % (31e12 / 50e9), size=12)
svg.text(20, H - 16, "必须做这么多次运算，才不至于让计算单元干等。片上与 HBM 为 H100 SXM5，互连为 V3 的 H800 集群。",
         size=12)

svg.save("m10-memory-hierarchy.svg")
for name, cap, bw, col, grp in levels:
    if bw:
        print(name, "ridge %.1f" % (F / bw), "1 GB takes %.1f us" % (1e9 / bw * 1e6))
