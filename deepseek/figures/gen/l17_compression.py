"""Figure 17.13: compression ratio vs OCR precision on Fox (redrawn after DeepSeek-OCR fig. 1(a),
using the numbers of its table 2; notes §3.5).

English Fox pages with 600-1300 ground-truth text tokens, 100 pages in 7 bins; Tiny (64 vision tokens)
and Small (100 vision tokens). Marker area proportional to the number of pages in the bin.
Page-weighted precision over the four cells below 10x: 97.14 % (notes §4.5).
Writes deepseek/figures/l17-compression.svg
"""

import math

from l17_svg import SVG
from l17_num import FOX, W_UNDER10

W, H = 760, 430
svg = SVG(W, H, "Fox 基准上的压缩比与 OCR 精度（据 DeepSeek-OCR 论文表 2 绘制）")

X0, X1, Y0, Y1 = 70, 600, 30, 340
RA, RB = 6, 21
PA, PB = 55, 100


def x(r):
    return X0 + (r - RA) / (RB - RA) * (X1 - X0)


def y(p):
    return Y1 - (p - PA) / (PB - PA) * (Y1 - Y0)


svg.rect(X0, Y0, x(10) - X0, Y1 - Y0, fill="green", fill_opacity=0.05)
for p in range(60, 101, 10):
    svg.line(X0, y(p), X1, y(p), color="grid", width=1)
    svg.text(X0 - 8, y(p) + 4, "%d%%" % p, size=11, anchor="end", color="muted")
for r in range(6, 22, 2):
    svg.line(x(r), Y0, x(r), Y1, color="grid", width=1)
    svg.text(x(r), Y1 + 17, "%d×" % r, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.line(x(10), Y0, x(10), Y1, color="ink", width=1.2, dash="5 4")
svg.text(x(10) + 5, Y0 + 14, "10×", size=12, weight="700")
svg.text((X0 + X1) / 2, Y1 + 38, "压缩比 r = 文本 token 数 / 视觉 token 数", size=12, anchor="middle", color="muted")
svg.text(X0 - 50, (Y0 + Y1) / 2, "OCR 精度（Precision）", size=12, anchor="middle", color="muted", rotate=-90)

series = [("64 个视觉 token（Tiny）", "orange", 1), ("100 个视觉 token（Small）", "blue", 2)]
for lab, col, idx in series:
    pts = [(row[idx][1], row[idx][0], row[3]) for row in FOX]
    svg.path("M" + " L".join("%.1f %.1f" % (x(r), y(p)) for r, p, _ in pts), stroke=col, width=1.6)
    for r, p, n in pts:
        rad = 2.5 * math.sqrt(n)        # area = pi * 6.25 * n: exactly proportional to the page count
        svg.circle(x(r), y(p), rad, fill=col, stroke=col, width=1.2, fill_opacity=0.28)

# labels on selected points
svg.text(x(19.7) - 16, y(59.1) + 4, "19.7×，59.1%", size=11.5, anchor="end", color="orange", weight="700")
svg.text(x(19.7) - 16, y(59.1) + 20, "这一格只有 4 页", size=11.5, anchor="end", color="orange")
svg.text(x(10.5) + 12, y(96.5) + 4, "10.5×，96.5%", size=11.5, color="orange")
svg.text(x(13.2) - 12, y(83.8) + 18, "13.2×，83.8%", size=11.5, anchor="end", color="orange")
svg.text(x(15.1) + 14, y(85.9) - 6, "15.1×，85.9%：比 13.2× 还高", size=11.5, color="orange")
svg.text(x(12.6) - 8, y(87.1) + 20, "12.6×，87.1%", size=11.5, anchor="end", color="blue")
svg.text(x(6.7) + 2, y(98.5) - 14, "6.7×，98.5%", size=11.5, color="blue")
svg.text(x(6.2), y(91), "< 10× 的四格", size=12, color="green", weight="700")
svg.text(x(6.2), y(91) + 17, "按页加权 %.2f%%" % W_UNDER10, size=12, color="green")
svg.text(x(6.2), y(91) + 34, "（77 页）", size=11.5, color="muted")

# legend
LX = 614
for k, (lab, col, _) in enumerate(series):
    yy = 60 + k * 44
    svg.circle(LX + 8, yy, 7, fill=col, stroke=col, width=1.2, fill_opacity=0.28)
    parts = lab.split("（")
    svg.text(LX + 22, yy + 4, parts[0], size=12, color=col, weight="700")
    svg.text(LX + 22, yy + 21, "（" + parts[1], size=11.5, color="muted")
svg.text(LX, 160, "圆面积 ∝ 该档页数", size=11.5, color="muted")
svg.text(LX, 178, "（7, 28, 28, 14,", size=11.5, color="muted")
svg.text(LX, 195, "  11, 8, 4 页）", size=11.5, color="muted")
svg.text(LX, 232, "Fox 英文部分，", size=11.5, color="muted")
svg.text(LX, 249, "原文 600–1300 token，", size=11.5, color="muted")
svg.text(LX, 266, "共 100 页，提示", size=11.5, color="muted")
svg.text(LX, 283, "“Free OCR.”", size=11.5, color="muted")
svg.text(20, 412, "每一格的压缩比 = 该档平均文本 token 数 ÷ 64 或 100；精度的计算方式论文没有定义。", size=11.5, color="muted")
svg.save("l17-compression.svg")
