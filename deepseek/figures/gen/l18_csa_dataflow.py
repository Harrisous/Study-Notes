"""Figure 18.4: CSA data flow for one query token (redrawn from Fig. 3 of the V4 report),
annotated with V4-Pro dimensions from sec. 4.2.1 and config.json (research notes sec. 3.1-3.2):
d = 7168, d_c' = 1536, 128 heads x 512, indexer 64 x 128 (FP4), m = 4, top-k 1024,
window 128, output groups 16 x (4096 -> 1024), entry 584 B, indexer key 68 B.
Writes deepseek/figures/l18-csa-dataflow.svg
"""

from l18_svg import SVG

W, H = 760, 640
svg = SVG(W, H, "CSA 的数据流（V4-Pro 维度）")

LX0, LX1 = 44, 222            # query-side column
MX0, MX1 = 250, 500           # middle column
AX0, AX1 = 524, 630           # cache sub-column A (indexer keys)
BX0, BX1 = 642, 748           # cache sub-column B (main entries)


def mid(a, b):
    return (a + b) / 2


# ---------------------------------------------------------------- inputs (bottom)
svg.box(LX0, 590, LX1 - LX0, 36, ["查询 token：*h*_{t}（7168）"], color="blue", fill_opacity=0.08, size=12)
svg.box(AX0, 590, BX1 - AX0, 36, ["历史 token：*h*_{τ}（7168），τ ≤ t"], color="muted", fill_opacity=0.10, size=12)

# ---------------------------------------------------------------- cache side
svg.box(AX0, 500, AX1 - AX0, 50, ["indexer 压缩器", ("m=4，128 维", {"size": 11, "color": "muted"})],
        color="violet", fill_opacity=0.08, size=12, line_gap=17)
svg.box(BX0, 500, BX1 - BX0, 50, ["主压缩器", ("m=4，两路重叠", {"size": 11, "color": "muted"})],
        color="orange", fill_opacity=0.10, size=12, line_gap=17)
svg.arrow(mid(AX0, AX1), 590, mid(AX0, AX1), 550)
svg.arrow(mid(BX0, BX1), 590, mid(BX0, BX1), 550)
svg.box(AX0, 410, AX1 - AX0, 58, ["indexer 键 *k*^{I}_{s}", ("128 维 FP4", {"size": 11}), ("68 B / 条", {"size": 11})],
        color="amber", fill_opacity=0.14, size=12, line_gap=16)
svg.box(BX0, 410, BX1 - BX0, 58, ["条目 *e*_{s}", ("512 维，K=V", {"size": 11}), ("584 B / 条", {"size": 11})],
        color="amber", fill_opacity=0.14, size=12, line_gap=16)
svg.arrow(mid(AX0, AX1), 500, mid(AX0, AX1), 468)
svg.arrow(mid(BX0, BX1), 500, mid(BX0, BX1), 468)
# sliding-window entries
svg.box(AX0, 196, BX1 - AX0, 46, ["滑动窗口条目 *u*_{τ}：最近 128 个", ("未压缩，512 维（单独的投影）", {"size": 11, "color": "muted"})],
        color="amber", fill_opacity=0.14, size=12, line_gap=17)
svg.path("M%d %d H754 V219 H%d" % (BX1, 608, BX1 + 1), stroke="ink", width=1.6, arrow=True)

# ---------------------------------------------------------------- query side
svg.box(LX0, 520, LX1 - LX0, 36, ["*W*^{DQ}：7168 → 1536"], color="blue", fill_opacity=0.08, size=12)
svg.arrow(mid(LX0, LX1), 590, mid(LX0, LX1), 556)
svg.box(LX0, 450, LX1 - LX0, 40, ["*c*^{Q}_{t}（1536）→ RMSNorm"], color="blue", fill_opacity=0.08, size=12)
svg.arrow(mid(LX0, LX1), 520, mid(LX0, LX1), 490)
svg.box(LX0, 350, LX1 - LX0, 50, ["*W*^{IUQ}：indexer 查询", ("64 头 × 128 维（FP4）", {"size": 11, "color": "muted"})],
        color="violet", fill_opacity=0.08, size=12, line_gap=17)
svg.arrow(mid(LX0, LX1), 450, mid(LX0, LX1), 400)
svg.box(LX0, 142, LX1 - LX0, 58, ["*W*^{UQ}：主查询", ("128 头 × 512 维", {"size": 11, "color": "muted"}),
                                  ("RMSNorm + RoPE（后 64 维）", {"size": 11, "color": "muted"})],
        color="blue", fill_opacity=0.10, size=12, line_gap=16)
svg.path("M%d 470 H24 V171 H%d" % (LX0, LX0 - 1), stroke="ink", width=1.6, arrow=True)

# head weights from h_t (middle column, bottom)
svg.box(MX0 + 40, 440, MX1 - MX0 - 80, 40, ["头权重 *w*^{I} = *W*^{w}*h*_{t}（64 个）"], color="violet",
        fill_opacity=0.08, size=12)
svg.path("M%d 608 H%d V482" % (LX1, mid(MX0, MX1)), stroke="ink", width=1.6, arrow=True)

# ---------------------------------------------------------------- middle column
svg.box(MX0, 350, MX1 - MX0, 50, ["lightning indexer",
                                  ("I_{t,s} = Σ_{j} w_{t,j} · ReLU(q_{t,j} · k_{s})", {"size": 12})],
        color="violet", fill_opacity=0.10, size=12, line_gap=19)
svg.arrow(LX1, 375, MX0, 375)
svg.arrow(mid(MX0, MX1), 440, mid(MX0, MX1), 400)
svg.path("M%d 410 V375 H%d" % (mid(AX0, AX1), MX1 + 1), stroke="violet", width=1.6, arrow=True)

svg.box(MX0, 268, MX1 - MX0, 44, ["top-k 选择：⌊(t+1)/4⌋ 个条目中选 1024 个"], color="violet",
        fill_opacity=0.10, size=12)
svg.arrow(mid(MX0, MX1), 350, mid(MX0, MX1), 312)
svg.path("M%d 410 V290 H%d" % (mid(BX0, BX1), MX1 + 1), stroke="orange", width=1.6, arrow=True)
svg.text(mid(BX0, BX1) - 6, 284, "按下标取条目", size=11, anchor="end", color="orange")

svg.box(MX0, 196, MX1 - MX0, 46, ["1024 个被选条目 ∪ 128 个窗口条目", ("每个条目同时当 K 和 V", {"size": 11, "color": "muted"})],
        color="orange", fill_opacity=0.10, size=12, line_gap=17)
svg.arrow(mid(MX0, MX1), 268, mid(MX0, MX1), 242)
svg.arrow(AX0, 219, MX1, 219)

svg.box(MX0, 110, MX1 - MX0, 50, ["共享 KV 的 MQA：128 头共用 1152 个条目",
                                  ("softmax 分母另加每头一个 sink 项 exp(z′_{i})", {"size": 11, "color": "muted"})],
        color="blue", fill_opacity=0.10, size=12, line_gap=18)
svg.arrow(mid(MX0, MX1), 196, mid(MX0, MX1), 160)
svg.path("M%d 171 H236 V135 H%d" % (LX1, MX0 - 1), stroke="blue", width=1.6, arrow=True)

svg.box(MX0 - 60, 30, MX1 - MX0 + 120, 52, ["逆 RoPE（后 64 维转 −t）→ 分组输出投影",
                                            ("16 组 × (4096 → 1024) → 拼成 16,384 → 7168", {"size": 11, "color": "muted"})],
        color="green", fill_opacity=0.10, size=12, line_gap=18)
svg.arrow(mid(MX0, MX1), 110, mid(MX0, MX1), 82)
svg.arrow(mid(MX0, MX1), 30, mid(MX0, MX1), 8)
svg.text(mid(MX0, MX1) + 10, 18, "注意力输出（7168 维），交给 mHC 的 H^{post}", size=12)

# column captions
svg.text(mid(LX0, LX1), 112, "查询侧：当前 token 自己算", size=12, anchor="middle", color="blue")
svg.text(mid(AX0, BX1), 104, "缓存侧：历史 token 留下的", size=12, anchor="middle", color="amber")
svg.text(mid(AX0, BX1), 122, "（琥珀色框 = KV cache）", size=11, anchor="middle", color="amber")

svg.save("l18-csa-dataflow.svg")
