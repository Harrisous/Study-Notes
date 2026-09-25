"""Figure 9.12: roofline of the decode attention kernel on H800, MHA vs GQA vs MLA (absorbed).

Hardware numbers are DeepSeek's (FlashMLA deep-dive blog, 2025-04-22): 3.35 TB/s HBM bandwidth,
990 TFLOPS nominal BF16, ~865 TFLOPS after throttling.  Arithmetic intensity per cached token,
one decoded token, bf16 (lesson-09-doc sec. 4.5):
  MHA cache (per-head K, V)         (2*128*128*2)/(2*128*128*2)          = 1.0 FLOP/B
  GQA-8 (DeepSeek 67B: 64 q heads)  (2*64*128*2)/(2*8*128*2)             = 8.0
  MLA absorbed, s_q = 1             (2*128*576 + 2*128*512)/(2*576)      = 241.8
  MLA absorbed, s_q = 2 (MTP)                                            = 483.6
Attainable = min(bandwidth * intensity, peak).
Writes deepseek/figures/l09-roofline.svg
"""

import math

from l09_svg import SVG, LogAxis

BWD = 3.35            # TB/s -> TFLOP/s per (FLOP/B)
PEAK, PEAK_T = 990.0, 865.0
pts = [
    ("MHA 式缓存", 1.0, "muted"),
    ("GQA-8（DeepSeek 67B）", 8.0, "muted"),
    ("MLA 吸收，s_{q} = 1", (2 * 128 * 576 + 2 * 128 * 512) / (2 * 576), "orange"),
    ("MLA 吸收，s_{q} = 2（MTP）", 2 * (2 * 128 * 576 + 2 * 128 * 512) / (2 * 576), "orange"),
]
assert abs(pts[2][1] - 241.78) < 0.01
ridge, ridge_t = PEAK / BWD, PEAK_T / BWD
print(ridge, ridge_t)

W, H = 720, 400
svg = SVG(W, H, "H800 上解码注意力的屋顶线：MHA、GQA 与吸收后的 MLA")
X0, X1, Y0, Y1 = 74, 690, 26, 330
xs = LogAxis(0.5, 2000, X0, X1)
ys = LogAxis(1, 2000, Y1, Y0)
for v, lab in [(1, "1"), (10, "10"), (100, "100"), (1000, "1000")]:
    svg.line(X0, ys(v), X1, ys(v), color="grid", width=1)
    svg.text(X0 - 6, ys(v) + 4, lab, size=11, anchor="end", color="muted")
for v, lab in [(1, "1"), (10, "10"), (100, "100"), (1000, "1000")]:
    svg.line(xs(v), Y0, xs(v), Y1, color="grid", width=1)
    svg.text(xs(v), Y1 + 16, lab, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="rule", width=1.2)
svg.line(X0, Y0, X0, Y1, color="rule", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "算术强度（每读 1 字节缓存做多少次运算，FLOP/B，对数刻度）", size=12, anchor="middle",
         color="muted")
svg.text(20, (Y0 + Y1) / 2, "可达算力（TFLOP/s，对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

# roofs
x_lo = 0.5
svg.polyline([(xs(x_lo), ys(BWD * x_lo)), (xs(ridge), ys(PEAK)), (xs(2000), ys(PEAK))], stroke="ink", width=2.2)
svg.polyline([(xs(ridge_t), ys(PEAK_T)), (xs(2000), ys(PEAK_T))], stroke="ink", width=1.4, dash="5 4")
# rotate the label to the slope the bandwidth line actually has on screen (log-log, unequal decades)
slope_deg = math.degrees(math.atan2(ys(BWD * 1) - ys(BWD * 10), xs(10) - xs(1)))
svg.text(xs(1.5), ys(BWD * 1.5) - 12, "带宽斜线 3.35 TB/s", size=12, rotate=-slope_deg, color="ink")
svg.text(xs(2000) - 4, ys(PEAK) - 7, "名义峰值 990", size=11.5, anchor="end")
svg.text(xs(2000) - 4, ys(PEAK_T) + 16, "降频后约 865", size=11.5, anchor="end", color="muted")
for r, lab, dy in ((ridge, "屋脊 295.5", 14), (ridge_t, "258.2", 30)):
    svg.line(xs(r), ys(1), xs(r), ys(PEAK if r == ridge else PEAK_T), color="rule", width=1, dash="2 3")
svg.text(xs(ridge) + 5, Y1 - 8, "295.5", size=11, color="muted")
svg.text(xs(ridge_t) - 5, Y1 - 8, "258.2", size=11, color="muted", anchor="end")
svg.text(xs(ridge) + 5, Y1 - 22, "屋脊点", size=11, color="muted")

for name, I, col in pts:
    att = min(BWD * I, PEAK)
    x, y = xs(I), ys(att)
    svg.circle(x, y, 6, fill=col, stroke=col)
    if I < 100:
        svg.text(x + 10, y + 4, "%s：%.0f FLOP/B → %.1f TFLOP/s" % (name, I, att), size=12,
                 color="ink" if col == "muted" else col, weight="600")
    elif I < 300:
        svg.text(x - 12, y + 2, "%s：%.1f FLOP/B，紧贴屋脊点左侧" % (name, I), size=12, color=col,
                 weight="600", anchor="end")
    else:
        svg.text(x - 6, y - 14, "%s：%.1f FLOP/B" % (name, I), size=12, color=col, weight="600", anchor="end")
svg.save("l09-roofline.svg")
