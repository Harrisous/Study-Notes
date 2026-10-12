"""Figure 12.6: increasing accumulation precision (redrawn after DeepSeek-V3 report Figure 7b).

Tensor Cores add FP8 x FP8 products with a limited-precision accumulator (about 14 significant
bits per V3 report 3.5.2; "13 fraction bits", FP22 registers per the Insights paper 3.1.1).
Every N_C = 128 elements along K (= 4 WGMMA instructions of 32 elements each) the partial sum
is copied to FP32 registers on the CUDA cores, multiplied by the two scaling factors, and
added to an FP32 accumulator. Two warpgroups alternate so the promotion of one overlaps the
MMA of the other. Schematic; widths are not measured times.
Writes deepseek/figures/l12-accum-promotion.svg
"""

from l12_svg import SVG

NC, WG = 128, 32
assert NC // WG == 4 and 7168 // NC == 56

W, H = 760, 400
svg = SVG(W, H, "提升累加精度：Tensor Core 里每累加 128 个乘积，就把部分和搬到 CUDA core 的 FP32 寄存器")

# Tensor Core box
TX, TY, TW, TH = 20, 20, 470, 118
svg.rect(TX, TY, TW, TH, fill="blue", fill_opacity=0.05, stroke="blue", width=1.3, rx=8)
svg.text(TX + 12, TY + 20, "Tensor Core（FP8 × FP8 乘法，低精度累加）", size=13, weight="600", color="blue")
bx = TX + 14
for i in range(4):
    svg.rect(bx + i * 80, TY + 38, 70, 34, fill="orange", fill_opacity=0.16, stroke="orange", width=1.1, rx=4)
    svg.text(bx + i * 80 + 35, TY + 55, "WGMMA %d" % (i + 1), size=12, anchor="middle")
    svg.text(bx + i * 80 + 35, TY + 68, "K 上 32 个", size=11, anchor="middle", color="muted")
    if i < 3:
        svg.arrow(bx + i * 80 + 70, TY + 55, bx + i * 80 + 80, TY + 55, color="ink", width=1.2)
# accumulator register
ax = bx + 4 * 80 + 8
svg.arrow(bx + 3 * 80 + 70, TY + 55, ax, TY + 55, color="ink", width=1.2)
svg.rect(ax, TY + 30, 110, 50, fill="amber", fill_opacity=0.14, stroke="amber", width=1.2, rx=4)
svg.text(ax + 55, TY + 48, "部分和 P_{g}", size=12, anchor="middle")
svg.text(ax + 55, TY + 72, "约 14 位有效位", size=11, anchor="middle", color="red")
svg.text(TX + 12, TY + 89, "“N_{C} 间隔”：沿 K 累加满 N_{C} = 128 个乘积", size=12, color="ink")
svg.text(TX + 12, TY + 109, "（= 4 条 WGMMA）就停下，把 P_{g} 交出去", size=12, color="ink")

# CUDA core box
CX_, CY_, CW, CH = 20, 176, 470, 92
svg.rect(CX_, CY_, CW, CH, fill="green", fill_opacity=0.05, stroke="green", width=1.3, rx=8)
svg.text(CX_ + 12, CY_ + 20, "CUDA core（FP32 寄存器，全精度累加）", size=13, weight="600", color="green")
svg.text(CX_ + 14, CY_ + 48, "y_{t,j}  ←  y_{t,j}  +  (s_{t,g} · S_{a,g}) × P_{g}", size=14, weight="600")
svg.text(CX_ + 14, CY_ + 72, "两个缩放因子在这里乘上（反量化）；y_{t,j} 是 FP32，g = 1…56（K = 7168）", size=12, color="muted")
svg.arrow(ax + 55, TY + 80, ax + 55, CY_ - 2, color="green", width=1.8)
svg.text(ax + 62, CY_ - 16, "复制到 FP32", size=12, color="green")

# right-hand note
svg.text(512, 44, "为什么要这样：", size=13, weight="600")
notes = ["Tensor Core 的 FP8 累加", "只保留约 14 位有效位", "（V3 报告 §3.5.2；Insights", "论文写作 13 位小数、", "FP22 寄存器）。",
         "K = 4096 时，论文测得最大", "相对误差接近 2%。"]
for i, t in enumerate(notes):
    svg.text(512, 66 + i * 19, t, size=12, color="muted" if i in (2, 3, 4) else "ink")
svg.text(512, 212, "代价：每 128 个元素一次", size=12)
svg.text(512, 231, "搬运与乘法。靠两个 warpgroup", size=12)
svg.text(512, 250, "交替来藏住它（下图）。", size=12)

# warpgroup timeline
LY = 300
svg.text(20, LY - 12, "时间 →（示意）", size=12, color="muted")
labels = ["warpgroup 1", "warpgroup 2"]
seq = [("MMA", "P"), ("P", "MMA")]
for r in range(2):
    y = LY + r * 36
    svg.text(20, y + 17, labels[r], size=12)
    x = 118
    for k in range(6):
        kind = "MMA" if (k + r) % 2 == 0 else "P"
        w = 96 if kind == "MMA" else 96
        if kind == "MMA":
            svg.rect(x, y, w - 4, 24, fill="blue", fill_opacity=0.16, stroke="blue", width=1, rx=3)
            svg.text(x + (w - 4) / 2, y + 16, "Tensor Core 乘加", size=11, anchor="middle")
        else:
            svg.rect(x, y, w - 4, 24, fill="green", fill_opacity=0.16, stroke="green", width=1, rx=3)
            svg.text(x + (w - 4) / 2, y + 16, "FP32 提升", size=11, anchor="middle")
        x += w
svg.text(20, H - 8, "据 DeepSeek-V3 技术报告图 7(b) 与 §3.3.2 重绘；宽度不代表实测时间。", size=11, color="muted")
svg.save("l12-accum-promotion.svg")
