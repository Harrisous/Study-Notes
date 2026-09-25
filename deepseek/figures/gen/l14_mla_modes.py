"""Figure 14.10: MLA's MHA mode vs MQA mode (redrawn after DeepSeek-V3.2 Fig. 7 / V3.2-Exp Fig. 4).

Per-key FLOPs (one query, one layer, QK^T + PV): MHA mode 2*128*(192+128) = 81,920;
MQA mode 2*128*(576+512) = 278,528; ratio 3.4. Numbers asserted below.
Writes deepseek/figures/l14-mla-modes.svg
"""

from l14_svg import SVG

NH, DH, DR, DC = 128, 128, 64, 512
MHA = 2 * NH * ((DH + DR) + DH)
MQA = 2 * NH * ((DC + DR) + DC)
assert (MHA, MQA) == (81920, 278528) and abs(MQA / MHA - 3.4) < 1e-9

W, H = 760, 470
svg = SVG(W, H, "MLA 的两种算法：MHA 模式与 MQA 模式（据 DeepSeek-V3.2 论文图 7 重绘）")


def cache_entry(x, y):
    svg.rect(x, y, 128, 26, fill="orange", fill_opacity=0.25, stroke="orange", width=1.2)
    svg.rect(x + 128, y, 20, 26, fill="red", fill_opacity=0.10, stroke="red", width=1.2)
    svg.text(x + 64, y + 18, "*c*_{s}^{KV}：512", size=12, anchor="middle")
    svg.text(x + 154, y + 18, "*k*_{s}^{R}：64", size=12, color="red")
    svg.text(x, y + 46, "缓存里的一个条目（576 个数）", size=12, color="amber")


# ------------------------------------------------------------------ left: MHA mode
LX = 20
svg.text(LX, 26, "MHA 模式（训练、预填充）", size=14, weight="700", color="blue")
svg.text(LX, 46, "先把缓存展开成每个头自己的键和值", size=12, color="muted")
cache_entry(LX + 10, 360)
# expansion
svg.box(LX + 10, 266, 170, 50, ["× W^{UK}、W^{UV}（每头一份）", ("展开：每 token 128 头", {"size": 11, "color": "muted"})],
        color="orange", fill_opacity=0.08, size=12, line_gap=22)
svg.arrow(LX + 28, 358, LX + 28, 316, color="orange", width=1.3)
# per-head keys/values
for h in range(4):
    x = LX + 10 + h * 42
    svg.rect(x, 172, 36, 30, fill="violet", fill_opacity=0.25, stroke="violet", width=1)
    svg.rect(x, 206, 36, 22, fill="green", fill_opacity=0.25, stroke="green", width=1)
svg.text(LX + 185, 192, "…", size=14)
svg.text(LX + 52, 160, "键 *k*_{s,i}：128 头 × 192", size=12, color="violet")
svg.text(LX + 52, 246, "值 *v*_{s,i}^{C}：128 头 × 128", size=12, color="green")
svg.arrow(LX + 28, 265, LX + 28, 232, color="ink", width=1.2)
# queries
svg.box(LX + 222, 172, 120, 56, ["查询 *q*_{t,i}", "128 头 × 192"], color="blue", fill_opacity=0.12, size=12,
        line_gap=20)
svg.box(LX + 10, 76, 332, 56, ["每个头：点积 192 + 加权 128",
                                ("每键 2 × 128 × (192 + 128) = 81,920 FLOPs", {"color": "blue", "weight": "700"})],
        color="blue", fill_opacity=0.08, size=12, line_gap=22)
svg.arrow(LX + 28, 170, LX + 28, 134, color="ink", width=1.2)
svg.arrow(LX + 282, 170, LX + 282, 134, color="blue", width=1.2)
svg.text(LX + 196, 318, "展开后每 token 每层：", size=11, color="muted")
svg.text(LX + 196, 334, "128 × 320 = 40,960 个数", size=11, color="muted")

# divider
svg.line(372, 20, 372, 420, color="rule", width=1.2)

# ------------------------------------------------------------------ right: MQA mode
RX = 392
svg.text(RX, 26, "MQA 模式（解码；DSA 的稀疏核）", size=14, weight="700", color="blue")
svg.text(RX, 46, "把 W^{UK} 吸收进查询，缓存条目原样当键和值", size=12, color="muted")
cache_entry(RX + 10, 360)
svg.box(RX + 10, 172, 160, 56, ["键 = 整个条目：576", "值 = 前 512 维"], color="orange", fill_opacity=0.14, size=12,
        line_gap=20)
svg.arrow(RX + 84, 358, RX + 84, 230, color="orange", width=1.4)
svg.text(RX + 92, 292, "不展开，", size=12, color="orange")
svg.text(RX + 92, 310, "128 个头共用", size=12, color="orange")
svg.box(RX + 196, 256, 156, 68, ["*q*_{t,i}^{C} 左乘 (W_{i}^{UK})^{⊤}", ("得 *q*_{t,i}^{A}：512 维", {"size": 11, "color": "muted"})],
        color="blue", fill_opacity=0.08, size=12, line_gap=30)
svg.box(RX + 196, 172, 156, 56, ["查询 [*q*^{A}; *q*^{R}]", "128 头 × 576"], color="blue", fill_opacity=0.12,
        size=12, line_gap=20)
svg.arrow(RX + 274, 256, RX + 274, 230, color="blue", width=1.2)
svg.box(RX + 10, 76, 342, 56, ["每个头：点积 576 + 加权 512，再乘 W_{i}^{UV}",
                                ("每键 2 × 128 × (576 + 512) = 278,528 FLOPs（× 3.4）", {"color": "blue", "weight": "700"})],
        color="blue", fill_opacity=0.08, size=12, line_gap=22)
svg.arrow(RX + 84, 170, RX + 84, 134, color="orange", width=1.2)
svg.arrow(RX + 274, 170, RX + 274, 134, color="blue", width=1.2)

svg.text(20, 446, "两种模式数学上等价（结合律），缩放都是 1/√192。MHA 模式每键算得少，但每个头要自己的键值；",
         size=12)
svg.text(20, 464, "MQA 模式每键多算 3.4 倍，换来“一个条目读一次、128 个头一起用”，这正是解码与 DSA 稀疏核需要的。", size=12)

svg.save("l14-mla-modes.svg")
