"""Figure 14.5: NSA's group-centric selection kernel (redrawn after Yuan et al. 2025, Fig. 3).

Schematic. Shapes from NSA's experimental backbone: one GQA group has h = 16 query heads,
d_k = 192, d_v = 128; selected blocks have l_s = 64 keys; the kernel block size B_k divides
l_s (its value is not given in the paper).
Writes deepseek/figures/l14-nsa-kernel.svg
"""

from l14_svg import SVG

W, H = 760, 430
svg = SVG(W, H, "NSA 选择分支的组中心核：一次装入一组 16 个查询头，按索引依次装入连续的 KV 块（据 NSA 图 3 重绘）")

# ------------------------------------------------------------ HBM panel
svg.rect(16, 30, 330, 330, fill="amber", fill_opacity=0.05, stroke="amber", width=1.2, rx=8)
svg.text(30, 52, "显存 HBM（大、慢）", size=13, weight="700", color="amber")

# queries: rows = positions, each row has 16 heads of one group
QX, QY = 34, 76
svg.text(QX, QY - 6, "查询 Q：每个位置 × 一组 16 头", size=12, color="blue")
for r in range(6):
    y = QY + r * 16
    hl = r == 3
    svg.rect(QX, y, 128, 13, fill="blue", fill_opacity=0.45 if hl else 0.08, stroke="blue" if hl else "rule",
             width=1.2 if hl else 0.6)
    for h in range(1, 16):
        svg.line(QX + h * 8, y, QX + h * 8, y + 13, color="rule", width=0.4)
svg.text(QX + 136, QY + 3 * 16 + 11, "位置 t", size=12, color="blue", weight="600")

# keys and values: long strips of 12 blocks, some selected
KX, KY, BW = 34, 200, 24
sel = {1, 5, 6, 9}
svg.text(KX, KY - 8, "键 K（每块 ℓ_{s} = 64 个 token × 192 维）", size=12, color="violet")
for b in range(12):
    svg.rect(KX + b * BW, KY, BW - 2, 22, fill="violet", fill_opacity=0.5 if b in sel else 0.07,
             stroke="violet" if b in sel else "rule", width=1.1 if b in sel else 0.6)
VY = 268
svg.text(KX, VY - 8, "值 V（每块 64 个 token × 128 维）", size=12, color="green")
for b in range(12):
    svg.rect(KX + b * BW, VY, BW - 2, 22, fill="green", fill_opacity=0.5 if b in sel else 0.07,
             stroke="green" if b in sel else "rule", width=1.1 if b in sel else 0.6)
svg.text(KX, 318, "深色块 = 索引集 I_{t} 选中的块（组内 16 头共用）", size=12, color="muted")
svg.text(KX, 338, "每块在显存里是连续的一段，可以整段搬运", size=12, color="muted")

# ------------------------------------------------------------ SRAM panel
SX0 = 430
svg.rect(SX0, 30, 314, 330, fill="amber", fill_opacity=0.14, stroke="amber", width=1.4, rx=8)
svg.text(SX0 + 14, 52, "片上 SRAM（小、快）", size=13, weight="700", color="amber")
svg.box(SX0 + 20, 66, 130, 50, ["Q_{t}：16 × 192", ("每个程序只装一次", {"size": 11, "color": "muted"})],
        color="blue", fill_opacity=0.14, size=12, line_gap=20)
svg.box(SX0 + 20, 136, 130, 50, ["K 块：B_{k} × 192", ("B_{k} 整除 64", {"size": 11, "color": "muted"})],
        color="violet", fill_opacity=0.14, size=12, line_gap=20)
svg.box(SX0 + 20, 206, 130, 50, ["V 块：B_{k} × 128", ("与 K 块同步装入", {"size": 11, "color": "muted"})],
        color="green", fill_opacity=0.14, size=12, line_gap=20)
svg.box(SX0 + 170, 120, 126, 84, ["16 × B_{k} 的分数", "在线 softmax", "O += P V"], color="ink",
        fill_opacity=0.05, size=12, line_gap=20)
svg.arrow(SX0 + 150, 92, SX0 + 168, 136, color="blue", width=1.4)
svg.arrow(SX0 + 150, 162, SX0 + 168, 162, color="violet", width=1.4)
svg.arrow(SX0 + 150, 232, SX0 + 168, 190, color="green", width=1.4)
svg.box(SX0 + 170, 262, 126, 40, ["O_{t}：16 × 128"], color="green", fill_opacity=0.18, size=12)
svg.arrow(SX0 + 233, 204, SX0 + 233, 260, color="ink", width=1.4)
svg.text(SX0 + 20, 330, "内层循环：按 I_{t} 依次装入被选块，", size=12)
svg.text(SX0 + 20, 348, "每个块只装一次，供 16 个头同时使用", size=12)

# ------------------------------------------------------------ transfers
svg.path("M%d %d C %d %d, %d %d, %d %d" % (QX + 182, QY + 3 * 16 + 6, 320, 110, 380, 92, SX0 + 18, 92),
         stroke="blue", width=1.6, arrow=True)
svg.path("M%d %d C %d %d, %d %d, %d %d" % (KX + 9 * BW + 10, KY + 11, 360, 190, 390, 162, SX0 + 18, 162),
         stroke="violet", width=1.6, arrow=True)
svg.path("M%d %d C %d %d, %d %d, %d %d" % (KX + 9 * BW + 10, VY + 11, 360, 260, 390, 232, SX0 + 18, 232),
         stroke="green", width=1.6, arrow=True)
svg.path("M%d %d L%d %d" % (SX0 + 168, 290, 350, 290), stroke="green", width=1.6, dash="5 3", arrow=True)
svg.text(390, 282, "写回", size=12, color="green", anchor="middle")

svg.text(16, 404, "外层循环（Triton grid）：每个 (查询位置 t, GQA 组) 一个程序。与“每个头各选各的”相比，", size=12)
svg.text(16, 422, "同一块 KV 从显存读一次，就被 16 个头的打分与加权求和共用：同样的读取量，做 16 倍的计算。", size=12)

svg.save("l14-nsa-kernel.svg")
