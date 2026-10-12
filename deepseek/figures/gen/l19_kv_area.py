"""Figure 19.6: global KV bytes per token as an area.

Each rectangle is one stored copy of global KV (one KV source layer):
  width  = entries written per token by that copy = 1/m      (sequence dimension)
  height = bytes per entry (main KV + indexer key if any)    (entry-size dimension)
  count  = number of copies                                   (layer dimension)
Total area = global KV bytes per token: V4-Flash 3,514.25 B, V4.1-Flash 890 B.
Writes deepseek/figures/l19-kv-area.svg
"""

from l19_svg import SVG
from l19_num import MAIN_V4, MAIN_V41, IDX, G4F, G41, N_CSA_4F, N_HCA_4F

W, H = 760, 482
svg = SVG(W, H, "每 token 全局 KV 的面积图：V4-Flash 3,514 字节，V4.1-Flash 890 字节")

X0 = 120
SX = 92.0             # px per entry-per-token
SY = 0.25             # px per byte


def xs(e):
    return X0 + e * SX


def draw_axis(base, top_b, label_y):
    # y ticks (bytes per entry)
    for b in range(0, top_b + 1, 200):
        y = base - b * SY
        svg.line(X0 - 4, y, X0, y, color="muted", width=1)
        svg.text(X0 - 8, y + 4, "%d" % b, size=11, anchor="end", color="muted")
    svg.line(X0, base, X0, base - top_b * SY - 6, color="muted", width=1)
    svg.line(X0, base, xs(5.75), base, color="muted", width=1)
    for e in range(0, 6):
        svg.line(xs(e), base, xs(e), base + 4, color="muted", width=1)
        svg.text(xs(e), base + 17, str(e), size=11, anchor="middle", color="muted")
    svg.text(X0 - 50, base - top_b * SY / 2, "字节/条目", size=12, anchor="middle", color="muted",
             rotate=-90)


# ---------------------------------------------------------------- row 1: V4-Flash
B1 = 222
svg.text(20, 26, "(a) DeepSeek-V4-Flash：41 份全局 KV（21 份 CSA，m = 4；20 份 HCA，m = 128）", size=13, weight="700")
draw_axis(B1, 700, 0)
x = 0.0
for i in range(N_CSA_4F):
    w = 0.25
    svg.rect(xs(x), B1 - MAIN_V4 * SY, w * SX, MAIN_V4 * SY, fill="amber", fill_opacity=0.45,
             stroke="surface", width=1)
    svg.rect(xs(x), B1 - (MAIN_V4 + IDX) * SY, w * SX, IDX * SY, fill="violet", fill_opacity=0.6,
             stroke="surface", width=1)
    x += w
hca0 = x
for i in range(N_HCA_4F):
    w = 1 / 128
    svg.rect(xs(x), B1 - MAIN_V4 * SY, w * SX, MAIN_V4 * SY, fill="orange", fill_opacity=0.55)
    x += w
assert abs(x - (N_CSA_4F / 4 + N_HCA_4F / 128)) < 1e-9
# annotations
svg.text(xs(21 * 0.25 / 2), B1 - (MAIN_V4 + IDX) * SY - 10, "21 块，每块宽 1/4、高 584 + 68 = 652", size=12,
         anchor="middle")
HX = xs(hca0 + 20 / 128) + 8
svg.line(HX - 6, B1 - MAIN_V4 * SY * 0.6, HX + 14, B1 - MAIN_V4 * SY * 0.6, color="orange", width=1.2)
svg.text(HX + 18, B1 - MAIN_V4 * SY * 0.6 - 14, "20 块 HCA，", size=12, color="orange")
svg.text(HX + 18, B1 - MAIN_V4 * SY * 0.6 + 4, "每块宽 1/128，", size=12, color="orange")
svg.text(HX + 18, B1 - MAIN_V4 * SY * 0.6 + 22, "合计 91.25 字节", size=12, color="orange")
svg.text(xs(21 * 0.25 / 2), B1 - 60, "面积 = %s 字节/token" % "{:,.2f}".format(G4F), size=14, anchor="middle", weight="700")

# ---------------------------------------------------------------- row 2: V4.1-Flash
B2 = 390
svg.text(20, 268, "(b) DeepSeek-V4.1-Flash：4 份全局 KV（层 2、8、14 的 m = 2；层 20 的 m = 1）", size=13, weight="700")
draw_axis(B2, 400, 0)
x = 0.0
for layer, m in ((2, 2), (8, 2), (14, 2), (20, 1)):
    w = 1 / m
    svg.rect(xs(x), B2 - MAIN_V41 * SY, w * SX, MAIN_V41 * SY, fill="amber", fill_opacity=0.45,
             stroke="surface", width=1)
    svg.rect(xs(x), B2 - (MAIN_V41 + IDX) * SY, w * SX, IDX * SY, fill="violet", fill_opacity=0.6,
             stroke="surface", width=1)
    svg.text(xs(x + w / 2), B2 - MAIN_V41 * SY / 2 + 4, "层 %d" % layer, size=12, anchor="middle")
    x += w
assert x == 2.5
svg.text(xs(2.5) + 12, B2 - (MAIN_V41 + IDX) * SY + 8, "高 288 + 68 = 356（主 KV 改用 FP4）", size=12)
svg.text(xs(2.5) + 12, B2 - (MAIN_V41 + IDX) * SY + 26, "总宽 3 × 1/2 + 1 × 1 = 2.5 条目", size=12)
svg.text(xs(2.5) + 12, B2 - (MAIN_V41 + IDX) * SY + 50, "面积 = %d 字节/token" % G41, size=14, weight="700")

# ---------------------------------------------------------------- legend
LY = 440
svg.text(X0 + 2.9 * SX, B2 + 36, "每 token 条目数（各份之和）", size=12, anchor="middle", color="muted")
svg.rect(20, LY, 14, 12, fill="amber", fill_opacity=0.45)
svg.text(40, LY + 11, "主 KV（main KV）", size=12)
svg.rect(170, LY, 14, 12, fill="violet", fill_opacity=0.6)
svg.text(190, LY + 11, "索引键（indexer K）", size=12)
svg.rect(330, LY, 14, 12, fill="orange", fill_opacity=0.55)
svg.text(350, LY + 11, "HCA 主 KV（无索引键）", size=12)
svg.text(20, LY + 31, "高 = 每条目字节（条目大小）；宽 = 1/m（序列维）；块数 = 份数（层维）；面积之和 = 每 token 字节。",
         size=12, color="muted")

svg.save("l19-kv-area.svg")
