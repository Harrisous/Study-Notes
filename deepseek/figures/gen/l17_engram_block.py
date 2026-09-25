"""Figure 17.5: one Engram layer from the memory vector to the residual streams
(redrawn after Engram fig. 1 right half, bottom part, plus the multi-branch integration of §2.4).

Shapes are the Engram-27B ones (notes §3.1): d = 2560, d_mem = 1280, mHC with 4 branches.
The fused projection stacks W_V and the four W_K^(m): (5 * 2560) x 1280 = 16,384,000 weights.
|s| <= sqrt(2560) = 50.6 after RMSNorm (learnable scales ignored); the signed square root is in the
released code only (gh-engram, V4.1 inference code), not in the paper's eq. (4).
Writes deepseek/figures/l17-engram-block.svg
"""

import math

from l17_svg import SVG

D, DMEM, BR = 2560, 1280, 4
FUSED_ROWS = (BR + 1) * D
assert FUSED_ROWS == 12_800 and FUSED_ROWS * DMEM == 16_384_000
assert round(math.sqrt(D), 1) == 50.6

W, H = 760, 470
svg = SVG(W, H, "Engram 的一层：键值投影、上下文门控、短卷积与多分支集成（Engram-27B 的形状）")

# ---------------------------------------------------------------- residual streams (right)
SX = [636, 660, 684, 708]
svg.text(672, 22, "mHC 残差流 ×4", size=12.5, anchor="middle", color="blue", weight="700")
for i, x in enumerate(SX):
    svg.line(x, 34, x, 440, color="blue", width=2.4 if i == 0 else 1.6, opacity=1 if i == 0 else 0.35)
for i, x in enumerate(SX):
    svg.circle(x, 414, 8, fill="surface", stroke="blue", width=1.6 if i == 0 else 1.1)
    svg.text(x, 418.5, "+", size=13, anchor="middle", color="blue")
svg.text(672, 460, "每条流各得一个 α^{(m)}·*v*", size=12, anchor="middle", color="blue")

# ---------------------------------------------------------------- memory vector and fused projection (left)
svg.box(20, 52, 176, 36, ["*e*_{t} ∈ ℝ^{1280}"], color="orange", fill_opacity=0.14, size=13.5)
svg.text(116, 112, "16 行 × 80 维拼成（图 17.2）", size=11.5, color="muted")
svg.arrow(108, 90, 108, 124, color="orange", width=1.6)
svg.rect(20, 126, 176, 84, fill="ink", fill_opacity=0.04, stroke="ink", width=1.2, rx=6)
svg.text(108, 148, "一次 FP8 矩阵乘法", size=12.5, anchor="middle", weight="700")
svg.text(108, 168, "[W^{V}; W^{K,(1)}; …; W^{K,(4)}]", size=12.5, anchor="middle")
svg.text(108, 188, "12,800 × 1,280", size=12, anchor="middle", color="muted")
svg.text(108, 204, "16.4M 权重 / 层", size=12, anchor="middle", color="muted")

svg.box(20, 262, 176, 36, ["*v*_{t} ∈ ℝ^{2560}（4 条流共享）"], color="green", fill_opacity=0.12, size=12.5)
svg.arrow(108, 212, 108, 258, color="green", width=1.6)
svg.text(116, 240, "值", size=12, color="green")

svg.box(232, 150, 168, 36, ["*k*_{t}^{(m)} ∈ ℝ^{2560}，m = 1…4"], color="violet", fill_opacity=0.12, size=12.5)
svg.arrow(198, 168, 228, 168, color="violet", width=1.6)
svg.text(214, 160, "键", size=12, anchor="middle", color="violet")

# ---------------------------------------------------------------- the gate of branch m (middle)
svg.box(440, 52, 150, 36, ["*h*_{t}^{(m)} ∈ ℝ^{2560}"], color="blue", fill_opacity=0.12, size=13)
svg.arrow(SX[0], 70, 594, 70, color="blue", width=1.6)
svg.text(612, 62, "读", size=12, anchor="middle", color="blue")
svg.text(515, 106, "Query：看过前文的隐藏状态", size=11.5, anchor="middle", color="muted")

svg.box(232, 216, 168, 30, ["RMSNorm"], color="violet", fill_opacity=0.08, size=12.5)
svg.box(440, 216, 150, 30, ["RMSNorm"], color="blue", fill_opacity=0.08, size=12.5)
svg.arrow(316, 188, 316, 212, color="violet", width=1.5)
svg.arrow(515, 110, 515, 212, color="blue", width=1.5)

DX, DY = 418, 280
svg.circle(DX, DY, 15, fill="surface", stroke="ink", width=1.4)
svg.text(DX, DY + 5, "·", size=20, anchor="middle", weight="700")
svg.arrow(316, 248, DX - 11, DY - 11, color="violet", width=1.4)
svg.arrow(515, 248, DX + 11, DY - 11, color="blue", width=1.4)
svg.text(DX + 24, DY + 5, "点积 ÷ √2560 = s_{t}，|s_{t}| ≤ 50.6", size=12)

svg.rect(DX - 84, 312, 168, 30, fill="violet", fill_opacity=0.06, stroke="violet", width=1.2, rx=5, dash="5 3")
svg.text(DX, 332, "sign(s)·√|s|（仅开源代码）", size=12, anchor="middle", color="violet")
svg.arrow(DX, DY + 16, DX, 308, color="ink", width=1.4)
svg.box(DX - 60, 356, 120, 30, ["σ → α_{t}^{(m)} ∈ (0,1)"], color="ink", fill_opacity=0.05, size=12.5)
svg.arrow(DX, 344, DX, 352, color="ink", width=1.4)

# multiply alpha * v
MX, MY = 290, 414
svg.circle(MX, MY, 14, fill="surface", stroke="green", width=1.5)
svg.text(MX, MY + 5, "×", size=15, anchor="middle", color="green", weight="700")
svg.path("M196 280 H%s V%s" % (MX, MY - 18), stroke="green", width=1.6, arrow=True)
svg.arrow(DX - 60, 380, MX + 12, MY - 8, color="ink", width=1.4)
svg.text(MX - 20, MY + 32, "*ṽ*_{t}^{(m)} = α_{t}^{(m)} *v*_{t}", size=12.5, anchor="middle", color="green")

# short conv (dashed: removed in V4.1)
CX0, CW = 348, 236
svg.rect(CX0, MY - 17, CW, 34, fill="amber", fill_opacity=0.08, stroke="amber", width=1.2, rx=5, dash="5 3")
svg.text(CX0 + CW / 2, MY - 1, "短卷积 w = 4，δ = 3，SiLU，+ 残差", size=12, anchor="middle", color="amber")
svg.text(CX0 + CW / 2, MY + 13, "27B 有；V4.1-Flash 去掉", size=11.5, anchor="middle", color="muted")
svg.arrow(MX + 15, MY, CX0 - 3, MY, color="green", width=1.6)
svg.arrow(CX0 + CW + 1, MY, SX[0] - 11, MY, color="green", width=1.6)
svg.text((CX0 + CW + SX[0]) / 2 - 2, MY - 8, "*Y*", size=13, anchor="middle", color="green", weight="700")

svg.save("l17-engram-block.svg")
