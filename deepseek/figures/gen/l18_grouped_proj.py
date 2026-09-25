"""Figure 18.4: grouped output projection of V4-Pro (sec. 2.3.1 of the V4 report; research notes C25).

n_h = 128 heads x c = 512 -> 65,536 dims; g = 16 groups of 8 heads (4096 dims) -> d_g = 1024 each;
concatenated 16 x 1024 = 16,384 -> W^{O2} -> d = 7168.
Panel (a) is the data flow; panel (b) draws the three weight matrices to scale
(one pixel = SCALE dimensions, rows = output, columns = input) so the areas are the parameter counts.
Writes deepseek/figures/l18-grouped-proj.svg
"""

from l18_svg import SVG, fmt

N_H, C, G, D_G, D = 128, 512, 16, 1024, 7168
HEADS_PER_G = N_H // G                      # 8
IN_G = HEADS_PER_G * C                      # 4096
CAT = N_H * C                               # 65,536
MID = G * D_G                               # 16,384

P_DIRECT = D * CAT                          # 469,762,048
P_O1 = G * D_G * IN_G                       # 67,108,864
P_O2 = D * MID                              # 117,440,512

W, H = 760, 452
svg = SVG(W, H, "分组输出投影（V4-Pro）：先组内、后全局")

# ---------------------------------------------------------------- panel (a): data flow
svg.text(24, 22, "(a) 一个 token 的注意力输出怎样回到 7168 维", size=13, weight="700")

X0, GW, GAP = 186, 30, 5                    # 16 groups across 186 .. 741
LBL = X0 - 12                               # right edge of the row labels
right = X0 + G * (GW + GAP) - GAP
cx = (X0 + right) / 2

svg.text(X0, 42, "拼接的注意力输出 *o*_{t} = 65,536 维；每根细条 1 个头（512 维），每 8 个头一组",
         size=11, color="muted")
YH = 52
for k in range(G):
    x = X0 + k * (GW + GAP)
    svg.rect(x, YH, GW, 26, fill="green", fill_opacity=0.06, stroke="green", width=1, rx=3)
    for h in range(HEADS_PER_G):
        svg.rect(x + 2.6 + h * 3.3, YH + 4, 2.2, 18, fill="green", fill_opacity=0.6)
    svg.arrow(x + GW / 2, YH + 26, x + GW / 2, 94, width=1.1)
    svg.box(x, 94, GW, 22, [str(k + 1)], color="blue", fill_opacity=0.12, size=11, rx=3)
    svg.arrow(x + GW / 2, 116, x + GW / 2, 131, width=1.1)
    svg.rect(x + 4, 131, GW - 8, 12, fill="green", fill_opacity=0.35, stroke="green", width=0.8, rx=2)

svg.text(LBL, YH + 17, "128 个头 × 512 维", size=12, anchor="end")
svg.text(LBL, 103, "第一步：16 个", size=12, anchor="end", color="blue")
svg.text(LBL, 118, "1024 × 4096 的 *W*^{O1}_{κ}", size=12, anchor="end", color="blue")
svg.text(LBL, 141, "16 组 × 1024 维", size=12, anchor="end")

svg.path("M%s 146 v6 H%s v-6" % (fmt(X0 + 4), fmt(right - 4)), stroke="green", width=1.1)
svg.arrow(cx, 152, cx, 180, width=1.4)
svg.text(cx + 10, 170, "拼接：16 × 1024 = 16,384 维", size=11, color="muted")
svg.box(cx - 215, 180, 430, 28, ["第二步：*W*^{O2}（7168 × 16,384），各组之间在这里才混合"],
        color="blue", fill_opacity=0.10, size=12)
svg.arrow(cx, 208, cx, 224, width=1.4)
svg.box(cx - 130, 224, 260, 26, ["输出 7168 维 → 交给 mHC 的 *H*^{post}"],
        color="green", fill_opacity=0.10, size=12)
svg.text(LBL, 192, "组与组之间", size=11, anchor="end", color="muted")
svg.text(LBL, 207, "要到第二步才混合", size=11, anchor="end", color="muted")

# ---------------------------------------------------------------- panel (b): parameters to scale
svg.line(24, 266, W - 24, 266, color="rule", width=1)
svg.text(24, 288, "(b) 三个权重矩阵按同一比例尺画出（行 = 输出维，列 = 输入维），面积就是参数量",
         size=13, weight="700")

SCALE = 204.8                                # dimensions per pixel: 65,536 -> 320 px


def px(n):
    return n / SCALE


BX = 40
TX = BX + px(CAT) + 14                       # text column right of the 320-px-wide matrices

yd = 304                                     # direct projection: 7168 x 65,536
svg.rect(BX, yd, px(CAT), px(D), fill="muted", fill_opacity=0.30, stroke="muted", width=1)
svg.text(TX, yd + 14, "直接投影 *W*^{O}：7168 × 65,536", size=12)
svg.text(TX, yd + 30, "= %.1fM 个参数" % (P_DIRECT / 1e6), size=12, color="muted")

yo = 356                                     # block-diagonal W^{O1}: 16,384 x 65,536 frame
svg.rect(BX, yo, px(CAT), px(MID), fill=None, stroke="blue", width=1, dash="3 3")
bw, bh = px(IN_G), px(D_G)
for k in range(G):
    svg.rect(BX + k * bw, yo + k * bh, bw, bh, fill="blue", fill_opacity=0.85)
svg.text(TX, yo + 14, "*W*^{O1}：16 个 1024 × 4096 的块排在对角线上", size=12, color="blue")
svg.text(TX, yo + 30, "= %.1fM（虚线框里其余位置恒为 0，不存）" % (P_O1 / 1e6), size=12, color="muted")

x2, y2 = TX, yo + 44                         # W^{O2}: 7168 x 16,384
svg.rect(x2, y2, px(MID), px(D), fill="blue", fill_opacity=0.30, stroke="blue", width=1)
svg.text(x2 + px(MID) + 10, y2 + 14, "*W*^{O2}：7168 × 16,384 = %.1fM" % (P_O2 / 1e6), size=12, color="blue")
svg.text(x2 + px(MID) + 10, y2 + 31, "两步合计 %.1fM，是直接投影的 %.1f%%"
         % ((P_O1 + P_O2) / 1e6, 100 * (P_O1 + P_O2) / P_DIRECT), size=12, weight="700")

svg.save("l18-grouped-proj.svg")
