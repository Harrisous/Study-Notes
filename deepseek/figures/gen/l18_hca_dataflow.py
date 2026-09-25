"""Figure 18.5: HCA data flow for one query token (redrawn from Fig. 4 of the V4 report),
V4-Pro dimensions (research notes sec. 2.5, 3.1): m' = 128 without overlap, all completed entries
plus 128 window entries, shared-KV MQA with 128 heads x 512, sink, inverse RoPE, grouped output
projection. Writes deepseek/figures/l18-hca-dataflow.svg
"""

from l18_svg import SVG

W, H = 720, 470
svg = SVG(W, H, "HCA 的数据流（V4-Pro 维度）")

LX0, LX1 = 24, 214
MX0, MX1 = 244, 476
RX0, RX1 = 506, 700


def mid(a, b):
    return (a + b) / 2


# inputs
svg.box(LX0, 420, LX1 - LX0, 36, ["查询 token：*h*_{t}（7168）"], color="blue", fill_opacity=0.08, size=12)
svg.box(RX0, 420, RX1 - RX0, 36, ["历史 token：*h*_{τ}，τ ≤ t"], color="muted", fill_opacity=0.10, size=12)

# cache side
svg.box(RX0, 330, RX1 - RX0, 50, ["压缩器：m′ = 128，不重叠", ("一路投影 + 逐通道 softmax", {"size": 11, "color": "muted"})],
        color="orange", fill_opacity=0.10, size=12, line_gap=17)
svg.arrow(mid(RX0, RX1), 420, mid(RX0, RX1), 380)
svg.box(RX0, 244, RX1 - RX0, 56, ["条目 *e*′_{s}：512 维，K = V", ("584 B，每 128 个 token 一条", {"size": 11}),
                                  ("1M 时共 8192 条", {"size": 11})],
        color="amber", fill_opacity=0.14, size=12, line_gap=16)
svg.arrow(mid(RX0, RX1), 330, mid(RX0, RX1), 300)
svg.box(RX0, 150, RX1 - RX0, 46, ["滑动窗口条目：最近 128 个", ("未压缩，512 维", {"size": 11, "color": "muted"})],
        color="amber", fill_opacity=0.14, size=12, line_gap=17)
svg.path("M%d 438 H%d V173 H%d" % (RX1, RX1 + 12, RX1 + 1), stroke="ink", width=1.6, arrow=True)

# query side
svg.box(LX0, 350, LX1 - LX0, 36, ["*W*^{DQ}：7168 → 1536"], color="blue", fill_opacity=0.08, size=12)
svg.arrow(mid(LX0, LX1), 420, mid(LX0, LX1), 386)
svg.box(LX0, 280, LX1 - LX0, 40, ["*c*^{Q}_{t}（1536）→ RMSNorm"], color="blue", fill_opacity=0.08, size=12)
svg.arrow(mid(LX0, LX1), 350, mid(LX0, LX1), 320)
svg.box(LX0, 180, LX1 - LX0, 58, ["*W*^{UQ}：主查询", ("128 头 × 512 维", {"size": 11, "color": "muted"}),
                                  ("RMSNorm + RoPE（后 64 维）", {"size": 11, "color": "muted"})],
        color="blue", fill_opacity=0.10, size=12, line_gap=16)
svg.arrow(mid(LX0, LX1), 280, mid(LX0, LX1), 238)

# middle
svg.box(MX0, 196, MX1 - MX0, 56, ["全部 ⌊(t+1)/128⌋ 个条目", ("∪ 128 个窗口条目", {"size": 12}),
                                  ("不打分、不挑选", {"size": 11, "color": "red"})],
        color="orange", fill_opacity=0.10, size=12, line_gap=17)
svg.path("M%d 272 H490 V232 H%d" % (RX0, MX1 + 1), stroke="orange", width=1.6, arrow=True)
svg.path("M%d 173 H%d V195" % (RX0, MX1 - 40), stroke="ink", width=1.6, arrow=True)
svg.box(MX0, 100, MX1 - MX0, 50, ["稠密的共享 KV MQA：128 头", ("softmax 分母另加 sink 项", {"size": 11, "color": "muted"})],
        color="blue", fill_opacity=0.10, size=12, line_gap=18)
svg.arrow(mid(MX0, MX1) - 30, 196, mid(MX0, MX1) - 30, 150)
svg.path("M%d 209 H229 V125 H%d" % (LX1, MX0 - 1), stroke="blue", width=1.6, arrow=True)
svg.box(MX0 - 40, 26, MX1 - MX0 + 80, 46, ["逆 RoPE（−t）→ 分组输出投影 → 7168"], color="green",
        fill_opacity=0.10, size=12)
svg.arrow(mid(MX0, MX1), 100, mid(MX0, MX1), 72)
svg.arrow(mid(MX0, MX1), 26, mid(MX0, MX1), 6)
svg.text(mid(MX0, MX1) + 10, 16, "注意力输出（7168 维）", size=12)

svg.text(mid(RX0, RX1), 134, "琥珀色框 = KV cache", size=11, anchor="middle", color="amber")
svg.text(mid(LX0, LX1), 160, "与 CSA 相同", size=11, anchor="middle", color="muted")

svg.save("l18-hca-dataflow.svg")
