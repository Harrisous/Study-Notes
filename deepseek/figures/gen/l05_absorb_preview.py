"""Figure 5.15: why RoPE breaks MLA's weight absorption, and the decoupled-RoPE fix (preview of lesson 9).

Schematic. Dimensions from DeepSeek-V2 (paper and config): d_c = 512, d_h = 128, d_h^R = 64,
cache per token per layer d_c + d_h^R = 576 elements (research notes R21, section 2.7).
Writes deepseek/figures/l05-absorb-preview.svg
"""

from l05_svg import SVG

W, H = 760, 520
svg = SVG(W, H, "RoPE 为什么挡住矩阵吸收，以及解耦 RoPE")
CW, CH = 44, 26          # cache cell


def cache_row(x, y, labels, col="amber", sub=None):
    for n, lab in enumerate(labels):
        svg.rect(x + n * (CW + 6), y, CW, CH, fill=col, fill_opacity=0.22, stroke=col, width=1.2, rx=3)
        svg.text(x + n * (CW + 6) + CW / 2, y + 17, lab, size=12, anchor="middle")
    if sub:
        svg.text(x, y + CH + 16, sub, size=11.5, color="muted")


# ---------------- row A
ya = 30
svg.text(20, ya, "① 没有 RoPE：吸收成立", size=13.5, weight="700", color="green")
svg.box(20, ya + 16, 92, 34, ["*q*_{t}（128）"], color="blue", size=13)
svg.arrow(112, ya + 33, 150, ya + 33, color="ink", width=1.4)
svg.box(152, ya + 12, 150, 42, ["乘 W^{UK} 的转置", "固定矩阵，可预先并进 W^{Q}"], color="blue", fill_opacity=0.06, size=12.5)
svg.arrow(302, ya + 33, 340, ya + 33, color="ink", width=1.4)
svg.box(342, ya + 16, 96, 34, ["q̃_{t}（512）"], color="blue", size=13)
svg.arrow(438, ya + 33, 470, ya + 33, color="ink", width=1.4)
cache_row(474, ya + 20, ["*c*_{0}", "*c*_{1}", "*c*_{2}", "*c*_{3}", "…"], sub="缓存：每个 token 一个 *c*_{j}（512 维）")
svg.text(20, ya + 86, "一个 q̃_{t} 对缓存里所有位置通用：打分 = q̃_{t}·*c*_{j}", size=12, color="green")

# ---------------- row B
yb = 150
svg.text(20, yb, "② 键上加了 RoPE：中间夹着随位置变化的旋转，吸收失败", size=13.5, weight="700", color="red")
svg.box(20, yb + 40, 92, 34, ["*q*'_{t}（128）"], color="blue", size=13)   # q'_t: before rotation, as in (5.24)
xs = 474
for n in range(4):
    cx = xs + n * (CW + 6) + CW / 2
    svg.rect(cx - CW / 2, yb + 14, CW, 24, fill="red", fill_opacity=0.10, stroke="red", width=1.0, rx=4)
    svg.text(cx, yb + 30, "R_{%d−t}" % n, size=11.5, anchor="middle", color="red")
    svg.arrow(cx, yb + 38, cx, yb + 58, color="red", width=1.1)
svg.polyline([(112, yb + 57), (440, yb + 57), (440, yb + 26)], stroke="blue", width=1.3)
svg.arrow(440, yb + 26, xs - 4, yb + 26, color="blue", width=1.3)
cache_row(xs, yb + 60, ["*c*_{0}", "*c*_{1}", "*c*_{2}", "*c*_{3}", "…"])
# (5.24): q'^T R_{j-t} W^UK c_j -- the matrix between q' and c_j is R_{j-t} W^UK; the old wording
# "W^UK transposed, then times R_{j-t}" had the rotation on the wrong side (it would be R_{t-j})
svg.text(130, yb + 84, "每个位置 j 夹着一个不同的矩阵 R_{j−t}W^{UK}（式 5.24）：", size=12, color="red")
# kept short so W^Q ends left of the R_{j-t} subscript on the line above (they overlapped)
svg.text(130, yb + 103, "没有固定矩阵能预先并进 W^{Q}，", size=12, color="red")
svg.text(130, yb + 120, "只能把所有缓存的键重新展开再旋转。", size=12, color="red")

# ---------------- row C
yc = 320
svg.text(20, yc, "③ 解耦 RoPE（DeepSeek-V2/V3）：内容部分不转、可吸收；另开 64 维专门承载位置", size=13.5, weight="700",
         color="blue")
svg.box(20, yc + 18, 110, 34, ["*q*^{C}_{t}（128）"], color="blue", size=13)
svg.box(20, yc + 62, 110, 34, ["*q*^{R}_{t}（64，RoPE）"], color="red", fill_opacity=0.10, size=12.5)
svg.arrow(130, yc + 35, 300, yc + 35, color="ink", width=1.3)
svg.text(215, yc + 28, "吸收，同 ①", size=12, anchor="middle", color="green")
svg.arrow(130, yc + 79, 300, yc + 79, color="ink", width=1.3)
svg.text(215, yc + 96, "直接与 *k*^{R}_{j} 点积", size=12, anchor="middle", color="red")
# cache per token: 512 + 64
svg.rect(306, yc + 22, 320, 26, fill="amber", fill_opacity=0.22, stroke="amber", width=1.2, rx=3)
svg.text(466, yc + 40, "*c*_{j}：512 维（内容，所有头共用）", size=12, anchor="middle")
svg.rect(306, yc + 66, 40, 26, fill="red", fill_opacity=0.12, stroke="red", width=1.2, rx=3)
svg.text(356, yc + 84, "*k*^{R}_{j}：64 维，已按位置 j 旋转，所有头共享", size=12, color="red")
svg.path("M 632 %d h 10 v %d h -10" % (yc + 22, 70), stroke="amber", width=1.3)
svg.text(648, yc + 52, "每 token 每层", size=12, color="amber")
svg.text(648, yc + 70, "缓存 576 个数", size=12, color="amber", weight="700")
svg.text(20, yc + 128, "打分 = *q*^{C}·*k*^{C} + *q*^{R}·*k*^{R}：前一项靠吸收只读 *c*_{j}，后一项占缓存的 64/576 = 11.1%。", size=12.5)
svg.text(20, yc + 148, "YaRN 只作用在 *q*^{R}、*k*^{R} 这 64 维上（§7）；详细推导在第 9 课。", size=12.5, color="muted")
svg.text(20, yc + 168, "维度取自 DeepSeek-V2：d_{c} = 512，d_{h} = 128，d_{h}^{R} = 64（V3 相同）。", size=12, color="muted")
svg.save("l05-absorb-preview.svg")
