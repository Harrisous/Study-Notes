"""Figure 18.1: one DeepSeek-V4 block with real dimensions (redrawn from Fig. 2 of the V4 report).

Dimensions from the V4 report sec. 4.2.1 and config.json (research notes sec. 3.1):
V4-Pro 61 layers, d = 7168, 128 query heads, entry dim 512, 1 shared + 384 routed experts
(top-6, expert width 3072), first 3 MoE layers hash-routed, n_hc = 4 residual streams, MTP depth 1.
Writes deepseek/figures/l18-architecture.svg
"""

from l18_svg import SVG

W, H = 760, 660
svg = SVG(W, H, "DeepSeek-V4 的一个块：4 条残差流，mHC 包住注意力与 MoE 两个子层")

SX = [112, 130, 148, 166]          # the four residual streams
SL, SR = SX[0] - 16, SX[-1] + 16   # bundle edges
Y_TOP, Y_BOT = 150, 548

# ---------------------------------------------------------------- input side
svg.text(140, 648, "token id", size=13, anchor="middle", color="muted")
svg.arrow(140, 634, 140, 616, color="ink")
svg.box(40, 584, 200, 32, ["Embedding：129,280 × 7168"], color="muted", fill_opacity=0.10, size=13)
svg.arrow(140, 584, 140, 566, color="ink")
svg.text(258, 564, "复制成 n_{hc} = 4 条残差流：X_{l} ∈ ℝ^{4×7168}", size=12, color="ink")

# residual streams
for x in SX:
    svg.line(x, Y_BOT + 14, x, Y_TOP, color="ink", width=2)
svg.path("M%d %d L%d %d" % (SL + 2, Y_BOT + 14, SR - 2, Y_BOT + 14), stroke="ink", width=2)

# block frame
svg.rect(28, 178, 710, 352, stroke="muted", width=1.2, rx=10, dash="6 5")
svg.text(728, 198, "× 61 层（Pro）/ × 43 层（Flash）", size=12, anchor="end", color="muted")


def sublayer(y0, y1, title_lines, colour, extra=None):
    """One mHC-wrapped sublayer occupying the band y1 (top) .. y0 (bottom)."""
    y_pre, y_post, y_mid = y0 - 22, y1 + 22, (y0 + y1) / 2
    # H^res on the bundle
    svg.rect(SL - 4, y_mid - 15, SR - SL + 8, 30, fill="surface", stroke="ink", width=1.3, rx=5)
    svg.text(140, y_mid + 5, "H^{res} 4×4", size=12, anchor="middle")
    svg.text(SL - 10, y_mid - 2, "双随机", size=12, anchor="end", color="muted")
    svg.text(SL - 10, y_mid + 14, "20 次迭代", size=11, anchor="end", color="muted")
    # pre branch
    svg.line(SX[0], y_pre, SR, y_pre, color="ink", width=1.2)
    for x in SX:
        svg.circle(x, y_pre, 3, fill="ink")
    svg.arrow(SR, y_pre, 236, y_pre, color="ink")
    svg.box(236, y_pre - 20, 150, 40, ["H^{pre}：4 条 → 1 条", ("σ(·) 加权，形状 1×4", {"size": 11, "color": "muted"})],
            color="green", fill_opacity=0.10, size=12, line_gap=16)
    # the sublayer itself
    bx, bw, by, bh = 432, 290, y1 + 8, (y0 - y1) - 16
    svg.box(bx, by, bw, bh, title_lines, color=colour, fill_opacity=0.10, size=12)
    svg.arrow(386, y_pre, bx, y_pre, color="ink")
    # post branch
    svg.arrow(bx, y_post, 386, y_post, color="ink")
    svg.box(236, y_post - 20, 150, 40, ["H^{post}：1 条 → 4 条", ("2σ(·) 加权，形状 4×1", {"size": 11, "color": "muted"})],
            color="green", fill_opacity=0.10, size=12, line_gap=16)
    svg.arrow(236, y_post, SR + 16, y_post, color="ink")
    svg.line(SX[0], y_post, SR + 1, y_post, color="ink", width=1.2)
    svg.circle(SR + 9, y_post, 8, fill="surface", stroke="ink", width=1.4)
    svg.line(SR + 4, y_post, SR + 14, y_post, color="ink", width=1.4)
    svg.line(SR + 9, y_post - 5, SR + 9, y_post + 5, color="ink", width=1.4)
    if extra:
        extra(bx, by, bw, bh)


# attention sub-layer (lower band)
sublayer(516, 368, ["RMSNorm → CSA 或 HCA（逐层交替）",
                    "查询 128 头，每头 512 维",
                    "KV 条目 512 维，K 与 V 共用（MQA）",
                    "另有 128 个未压缩的滑窗条目"], "orange")


def experts(bx, by, bw, bh):
    y = by + bh - 26
    svg.rect(bx + 16, y, 14, 14, fill="violet", fill_opacity=0.75, stroke="violet", width=0.8, rx=2)
    for i in range(24):
        on = i in (2, 5, 9, 14, 17, 21)
        svg.rect(bx + 42 + i * 10, y, 8, 14, fill="blue" if on else "muted",
                 fill_opacity=0.8 if on else 0.18, stroke=None, rx=1.5)
    svg.text(bx + bw - 10, y - 6, "… 共 384 个", size=11, anchor="end", color="muted")


sublayer(346, 198, ["RMSNorm → DeepSeekMoE",
                    "1 个共享专家 + 384 个路由专家选 6 个",
                    "专家中间维 3072；前 3 层用 Hash 路由", ""], "blue", extra=experts)

# ---------------------------------------------------------------- output side
svg.box(40, 104, 200, 32, ["hc_head：4 条 → 1 条"], color="green", fill_opacity=0.10, size=12)
for x in SX:
    svg.line(x, Y_TOP, x, 136, color="ink", width=2)
svg.arrow(140, 104, 140, 86, color="ink")
svg.box(40, 54, 200, 32, ["预测头：7168 → 129,280"], color="muted", fill_opacity=0.10, size=12)
svg.arrow(140, 54, 140, 34, color="ink")
svg.text(140, 26, "主损失：下一个 token", size=12, anchor="middle")
svg.arrow(240, 120, 432, 120, color="ink")
svg.box(432, 100, 290, 40, ["MTP 模块（深度 1，与 V3 相同）"], color="violet", fill_opacity=0.10, size=12)
svg.arrow(577, 100, 577, 80, color="ink")
svg.text(577, 72, "MTP 损失：再往后一个 token", size=12, anchor="middle")

# spec legend
svg.text(258, 596, "V4-Pro：61 层，d = 7168，128 头，384 个路由专家（宽 3072），激活约 49B", size=12)
svg.text(258, 616, "V4-Flash：43 层，d = 4096，64 头，256 个路由专家（宽 2048），激活约 13B", size=12)
svg.text(258, 636, "图中数字按 V4-Pro 标注", size=11, color="muted")

svg.save("l18-architecture.svg")
