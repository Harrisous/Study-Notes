"""Figure 14.1: per-token FLOPs of dense attention versus position T, DeepSeek-V3 dimensions.

Model (lesson 14 research notes, section 2.3): one query at position T attends to T keys.
Dense MLA in MHA mode costs 2 * n_h * T * [(d_h + d_h^R) + d_h] = 81,920 T FLOPs per layer
(QK^T plus PV, one multiply-add = 2 FLOPs). 61 layers. Everything else (all weight matrices)
is approximated by 2 N_active = 2 * 37e9 = 74 GFLOPs per token.
Writes deepseek/figures/l14-quadratic-bill.svg
"""

import math

from l14_svg import SVG

N_H, D_H, D_R, LAYERS = 128, 128, 64, 61
PER_KEY = 2 * N_H * ((D_H + D_R) + D_H)          # 81,920
assert PER_KEY == 81920
PARAM = 2 * 37e9                                  # 74 GFLOPs


def attn(T):
    return LAYERS * PER_KEY * T


CROSS = PARAM / (LAYERS * PER_KEY)                # 14,808.5
assert abs(CROSS - 14808.5) < 1

W, H = 720, 420
X0, X1, Y0, Y1 = 92, 560, 40, 340
LX0, LX1 = math.log2(512), math.log2(131072)
LY0, LY1 = 0, 3                                   # 1 .. 1000 GFLOPs


def px(T):
    return X0 + (math.log2(T) - LX0) / (LX1 - LX0) * (X1 - X0)


def py(F):
    return Y1 - (math.log10(F / 1e9) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "稠密注意力每 token FLOPs 随位置 T 的增长（DeepSeek-V3 维度，61 层，双对数坐标）")

for e in range(LY0, LY1 + 1):
    y = py(10 ** e * 1e9)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, ["1", "10", "100", "1000"][e], size=11, anchor="end", color="muted")
for T, lab in ((512, "512"), (2048, "2K"), (8192, "8K"), (32768, "32K"), (131072, "128K")):
    x = px(T)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, lab, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 42, "位置 T（这个 token 能看到的 token 数，对数刻度，1K = 1024）", size=12,
         anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "每 token、全部 61 层的 GFLOPs（对数刻度）", size=12, anchor="middle",
         color="muted", rotate=-90)

# parameter line
yp = py(PARAM)
svg.line(X0, yp, X1, yp, color="muted", width=2.2, dash="7 4")
svg.text(X1 + 10, yp - 4, "其余全部矩阵乘", size=13, color="muted", weight="600")
svg.text(X1 + 10, yp + 13, "2N = 74 GFLOPs", size=12, color="muted")
svg.text(X1 + 10, yp + 29, "（与 T 无关）", size=12, color="muted")

# attention line (straight on log-log: slope 1)
svg.line(px(512), py(attn(512)), px(131072), py(attn(131072)), color="blue", width=2.8)
svg.text(X1 + 10, py(attn(131072)) + 4, "稠密注意力", size=13, color="blue", weight="600")
svg.text(X1 + 10, py(attn(131072)) + 21, "61 × 81,920 × T", size=12, color="blue")

# crossover
xc = px(CROSS)
svg.circle(xc, yp, 4.5, fill="ink")
svg.line(xc, yp + 5, xc, Y1, color="ink", width=1, dash="3 3")
svg.text(xc + 8, yp + 44, "T ≈ 14,808：注意力 = 其余之和", size=12, weight="600")

# shares
for T, lab in ((4096, "4K"), (32768, "32K"), (131072, "128K")):
    a = attn(T)
    share = a / (a + PARAM)
    x, y = px(T), py(a)
    svg.circle(x, y, 4, fill="blue")
    label = "%s：%.1f GF，占 %.1f%%" % (lab, a / 1e9, share * 100)
    if T < 8192:
        svg.text(x - 12, y - 10, label, size=12, anchor="end", color="ink")
    else:
        svg.text(x - 8, y - 12, label, size=12, anchor="end", color="ink")

# slope note: lower right, below the blue line and clear of the share labels along the top
svg.text(xc + 12, Y1 - 44, "斜率 1：位置每翻一倍，", size=12, color="muted")
svg.text(xc + 12, Y1 - 26, "这个 token 的注意力账单翻一倍", size=12, color="muted")

svg.save("l14-quadratic-bill.svg")
