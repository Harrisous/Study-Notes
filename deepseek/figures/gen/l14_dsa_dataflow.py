"""Figure 14.9: DSA instantiated under MLA (redrawn after DeepSeek-V3.2 Fig. 2 / V3.2-Exp Fig. 1).

Schematic with the real shapes: d = 7168, d_c' = 1536, d_c = 512, d_h = 128, d_h^R = 64, n_h = 128;
indexer H^I = 64 heads x d^I = 128 (from config.json, not the paper); k = 2048;
KV cache 656 bytes/token/layer (FP8 format of FlashMLA), indexer key 132 bytes/token/layer.
Flow goes bottom-up, as in the paper's figure.
Layout note: two-line boxes whose lines both carry sub/superscripts use a 26-unit line gap, so a
superscript on the second line never touches a subscript on the first.
Writes deepseek/figures/l14-dsa-dataflow.svg
"""

from l14_svg import SVG

W, H = 760, 652
svg = SVG(W, H, "DSA 在 MLA 上的数据流（据 DeepSeek-V3.2 论文图 2 重绘，标出真实形状）")

LX, LW = 12, 252      # indexer column
MX, MW = 284, 212     # query column
RX, RW = 514, 234     # KV column
GAP2 = 26             # line gap for two-line boxes with scripts on both lines
MUTED = {"size": 11, "color": "muted"}

svg.text(LX + LW / 2, 22, "lightning indexer（打分）", size=13, anchor="middle", weight="700", color="violet")
svg.text(MX + MW / 2, 22, "主注意力的查询", size=13, anchor="middle", weight="700", color="blue")
svg.text(RX + RW / 2, 22, "主注意力的键值（MLA）", size=13, anchor="middle", weight="700", color="orange")

# ---------------------------------------------------------------- input
yIn = 608
svg.box(190, yIn, 380, 32, ["输入隐藏状态 *h*_{t} ∈ ℝ^{7168}"], color="ink", fill_opacity=0.06, size=13)

# ---------------------------------------------------------------- row A (projections from h_t)
yA, hA = 520, 60
svg.box(MX + 30, yA, MW - 60, hA, ["*c*_{t}^{Q} ∈ ℝ^{1536}", ("W^{DQ} + RMSNorm", MUTED)],
        color="orange", fill_opacity=0.14, size=12, line_gap=GAP2)
svg.box(RX, yA, 124, hA, ["*c*_{t}^{KV} ∈ ℝ^{512}", ("W^{DKV}", MUTED)],
        color="orange", fill_opacity=0.14, size=12, line_gap=GAP2)
svg.box(RX + 132, yA, 102, hA, ["*k*_{t}^{R} ∈ ℝ^{64}", ("W^{KR} + RoPE", MUTED)],
        color="red", fill_opacity=0.06, size=12, line_gap=GAP2)
svg.box(LX, yA, 118, hA, ["w_{t,j}^{I}：64 个", ("W^{Iw}，BF16", MUTED)],
        color="violet", fill_opacity=0.10, size=12, line_gap=GAP2)
svg.box(LX + 132, yA, 120, hA, ["*k*_{t}^{I} ∈ ℝ^{128}", ("W^{IK} + LayerNorm", MUTED)],
        color="violet", fill_opacity=0.10, size=12, line_gap=GAP2)
for x in (MX + MW / 2, RX + 62, RX + 183, LX + 59, LX + 192):
    svg.arrow(x, yIn, x, yA + hA + 2, color="muted", width=1.2)

# ---------------------------------------------------------------- row B
yB, hB = 422, 62
svg.box(MX, yB, 102, hB, ["*q*_{t,i}^{A}：128 × 512", ("吸收 W^{UK}", MUTED)],
        color="blue", fill_opacity=0.12, size=12, line_gap=GAP2)
svg.box(MX + 110, yB, 102, hB, ["*q*_{t,i}^{R}：128 × 64", ("W^{QR} + RoPE", MUTED)],
        color="red", fill_opacity=0.06, size=12, line_gap=GAP2)
svg.arrow(MX + MW / 2, yA, MX + 51, yB + hB + 2, color="blue", width=1.2)
svg.arrow(MX + MW / 2, yA, MX + 161, yB + hB + 2, color="red", width=1.2)
# KV cache
svg.rect(RX, yB - 6, RW, hB + 6, fill="amber", fill_opacity=0.14, stroke="amber", width=1.4, rx=6)
svg.text(RX + RW / 2, yB + 16, "KV cache：[*c*_{s}^{KV}; *k*_{s}^{R}] ∈ ℝ^{576}", size=12, anchor="middle")
svg.text(RX + RW / 2, yB + 38, "全部 s ≤ t 都要留着", size=11, anchor="middle", color="muted")
svg.text(RX + RW / 2, yB + 54, "FP8 格式 656 字节 / token / 层", size=11, anchor="middle", color="amber")
svg.arrow(RX + 62, yA, RX + 62, yB + hB + 2, color="orange", width=1.2)
svg.arrow(RX + 183, yA, RX + 183, yB + hB + 2, color="red", width=1.2)
# indexer query (from c^Q) and indexer key cache
svg.box(LX + 26, yB, 98, hB, ["*q*_{t,j}^{I}：64 × 128", ("由 *c*^{Q} 投影", MUTED)],
        color="violet", fill_opacity=0.10, size=12, line_gap=GAP2)
# elbow route (up, left, up) so the line stays in the gap between rows A and B
yElbow = yA - 16
svg.path("M%d %d L%d %d L%d %d L%d %d" % (MX + 40, yA, MX + 40, yElbow, LX + 90, yElbow, LX + 90, yB + hB + 2),
         stroke="violet", width=1.2, arrow=True, join="round")
svg.rect(LX + 132, yB - 6, 120, hB + 6, fill="amber", fill_opacity=0.14, stroke="amber", width=1.4, rx=6)
svg.text(LX + 192, yB + 16, "索引键缓存 *k*_{s}^{I}", size=12, anchor="middle")
svg.text(LX + 192, yB + 38, "FP8 + 1 个缩放", size=11, anchor="middle", color="muted")
svg.text(LX + 192, yB + 54, "132 字节 / token / 层", size=11, anchor="middle", color="amber")
svg.arrow(LX + 192, yA, LX + 192, yB + hB + 2, color="violet", width=1.2)

# ---------------------------------------------------------------- row C: indexer
yC, hC = 314, 80
svg.box(LX, yC, LW, hC, ["I_{t,s} = Σ_{j} w_{t,j}^{I} · ReLU(*q*_{t,j}^{I} · *k*_{s}^{I})",
                          ("对全部 s ≤ t 打分，FP8，仍是 O(t)", MUTED),
                          ("每键 2 × 64 × 128 = 16,384 FLOPs", {"size": 11, "color": "violet"})],
        color="violet", fill_opacity=0.14, size=12, line_gap=23)
svg.arrow(LX + 75, yB - 2, LX + 75, yC + hC + 2, color="violet", width=1.3)
svg.arrow(LX + 192, yB - 8, LX + 192, yC + hC + 2, color="violet", width=1.3)
svg.arrow(LX + 12, yA, LX + 12, yC + hC + 2, color="violet", width=1.3)

# ---------------------------------------------------------------- row D: top-k
yD, hD = 224, 54
svg.box(LX, yD, LW, hD, ["Top-k 选择器：取分数最高的 k = 2048 个", ("得到下标集合 S_{t}；t ≤ 2048 时等于全选", MUTED)],
        color="violet", fill_opacity=0.22, size=12, line_gap=22)
svg.arrow(LX + LW / 2, yC, LX + LW / 2, yD + hD + 2, color="violet", width=1.4)

# gather from the KV cache
yG = yD + hD / 2
svg.path("M%d %d L%d %d" % (LX + LW, yG, RX + RW / 2, yG), stroke="violet", width=1.4, dash="5 3")
svg.path("M%d %d L%d %d" % (RX + RW / 2, yB - 8, RX + RW / 2, yG + 7), stroke="amber", width=1.6, arrow=True)
svg.text(RX + RW / 2 - 8, yG + 36, "按 S_{t} 取出 2048 个条目", size=12, color="amber", anchor="end")
svg.text(RX + RW / 2 - 8, yG + 55, "2048 × 576", size=12, color="amber", anchor="end")
svg.circle(RX + RW / 2, yG, 4, fill="violet")

# ---------------------------------------------------------------- row E: core attention
yE, hE = 118, 60
svg.box(MX, yE, RX + RW - MX, hE, ["多查询注意力（核心注意力，MQA 模式）",
                                   ("128 个头共用同一组 2048 个条目：键 576 维，值取前 512 维", MUTED)],
        color="blue", fill_opacity=0.10, size=13, line_gap=22)
svg.path("M%d %d L%d %d" % (MX + 51, yB, MX + 51, yE + hE + 2), stroke="blue", width=1.3, arrow=True)
svg.path("M%d %d L%d %d" % (MX + 161, yB, MX + 161, yE + hE + 2), stroke="red", width=1.3, arrow=True)
svg.text(MX + 106, yE + hE + 34, "拼成 128 × 576", size=11, anchor="middle", color="blue")
svg.path("M%d %d L%d %d" % (RX + RW / 2, yG - 5, RX + RW / 2, yE + hE + 2), stroke="amber", width=1.6, arrow=True)

# ---------------------------------------------------------------- row F: output
yF = 46
svg.box(MX, yF, RX + RW - MX, 44,
        ["*o*_{t,i}^{C}：128 × 512 → W^{UV} → *o*_{t,i}：128 × 128 → W^{O} → *u*_{t} ∈ ℝ^{7168}"],
        color="green", fill_opacity=0.14, size=12)
svg.arrow(MX + 230, yE, MX + 230, yF + 46, color="green", width=1.4)

# notes on the left top
svg.text(LX + 4, 60, "只有左列是 DSA 新增的；", size=12, color="violet")
svg.text(LX + 4, 78, "其余与 V3.1-Terminus 的 MLA 相同。", size=12, color="violet")
svg.text(LX + 4, 106, "参数：每层 13,959,424 个，", size=11, color="muted")
svg.text(LX + 4, 122, "62 层（含 MTP 层）约 0.87B", size=11, color="muted")
svg.text(LX + 4, 150, "头数与维度来自 config.json，", size=11, color="muted")
svg.text(LX + 4, 166, "论文只说“少量的头”", size=11, color="muted")

svg.save("l14-dsa-dataflow.svg")
