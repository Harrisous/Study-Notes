"""Figure M10.5: roofline model for A100 80GB SXM and H800 SXM5 (log-log), with DeepSeek-V3 kernels.

Roofs:
  A100 80GB SXM: F = 312 TFLOP/s dense BF16, BW = 2.039 TB/s (NVIDIA data sheet) -> ridge 153.
  H800 SXM5:     F = 990 TFLOP/s, BW = 3.35 TB/s; throttled ~865 TFLOP/s (DeepSeek FlashMLA blog).
                 H100 SXM (989, 3.35) is indistinguishable at this scale.
Points (BF16, s = 2, placed on the H800 roof):
  residual add y = x + z: 1 FLOP per 6 bytes; decode GEMV / MHA decode attention: 1;
  GQA-8 decode attention: 8; expert GEMM 7168 -> 2048 with m = 64 and m = 256 tokens;
  MLA decode attention in MQA mode (V3: 128 heads, d_k = 576, d_v = 512): 128 * 1088 / 576;
  prefill GEMM with a square 7168 x 7168 weight (V3 hidden size) and m = 4096 tokens.
Writes math/figures/m10-roofline.svg
"""

import math

import numpy as np

from m10_svg import SVG


def gemm_I(m, n, k, s=2):
    return 2 * m * n * k / (s * (m * k + k * n + m * n))


roofs = [("A100 80GB SXM：312 TFLOP/s，2.039 TB/s", 312, 2.039, "muted"),
         ("H800 SXM5：990 TFLOP/s，3.35 TB/s", 990, 3.35, "blue")]
THROTTLED = 865

points = [
    # name, I, label dx, label dy (pixels from the point), anchor
    ("残差相加", 1 / 6, 18, 28, "start"),
    ("decode 的 GEMV、MHA 注意力", 1.0, 18, 46, "start"),
    ("GQA-8 注意力", 8.0, 18, 46, "start"),
    ("专家 GEMM，m = 64", gemm_I(64, 2048, 7168), -24, -26, "end"),
    ("专家 GEMM，m = 256", gemm_I(256, 2048, 7168), -40, -40, "end"),
    ("MLA 解码注意力（V3）", 128 * (576 + 512) / 576, 60, 130, "start"),
    ("prefill GEMM，m = 4096", gemm_I(4096, 7168, 7168), 0, -30, "middle"),
]

W, H = 720, 476
X0, X1, Y0, Y1 = 84, 600, 36, 370
LX0, LX1 = -1.0, 4.0            # I from 0.1 to 10^4 FLOP/B
LY0, LY1 = -1.0, 3.5            # 0.1 .. ~3162 TFLOP/s


def px(v):
    return X0 + (math.log10(v) - LX0) / (LX1 - LX0) * (X1 - X0)


def py(v):
    return Y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "屋顶线模型：A100 与 H800，以及 DeepSeek-V3 里几种计算的位置（双对数坐标）")

for e in range(-1, 4):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, ("0.1", "1", "10", "100", "1000")[e + 1], size=11, anchor="end", color="muted")
for e in range(-1, 5):
    x = px(10 ** e)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, ("0.1", "1", "10", "100", "1000", "10⁴")[e + 1], size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "算术强度 I（FLOP/字节，对数刻度）", size=12, anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "可达算力（TFLOP/s，对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

Is = np.logspace(LX0, LX1, 400)
for name, F, BW, col in roofs:
    ys = [min(F, BW * i) for i in Is]
    d = "M" + " L".join("%.1f %.1f" % (px(i), py(y)) for i, y in zip(Is, ys) if y >= 10 ** LY0)
    svg.path(d, stroke=col, width=2.8 if col == "blue" else 2.4, join="round")
    ridge = F / BW
    svg.line(px(ridge), py(F), px(ridge), Y1, color=col, width=1, dash="3 3")
    if col == "blue":
        svg.text(px(ridge) + 4, Y1 - 8, "脊点 %.1f" % ridge, size=12, color=col, weight="600")
    else:
        svg.text(px(ridge) - 4, Y1 - 8, "脊点 %.0f" % ridge, size=12, color=col, weight="600", anchor="end")
# throttled roof
r2 = THROTTLED / 3.35
svg.line(px(r2), py(THROTTLED), X1, py(THROTTLED), color="blue", width=1.2, dash="5 4")
svg.text(X1 + 6, py(THROTTLED) + 14, "降频后约 865", size=11, color="blue")
svg.text(X1 + 6, py(990) - 2, "990", size=12, color="blue", weight="600")
svg.text(X1 + 6, py(312) + 4, "312", size=12, color="muted", weight="600")

# roof legend below the x-axis title
LGX, LGY = X0, Y1 + 70
svg.line(LGX, LGY, LGX + 26, LGY, color="blue", width=2.8)
svg.text(LGX + 32, LGY + 4, roofs[1][0] + "（H100 SXM 为 989、3.35，几乎重合）", size=12, color="blue")
svg.line(LGX, LGY + 22, LGX + 26, LGY + 22, color="muted", width=2.4)
svg.text(LGX + 32, LGY + 26, roofs[0][0] + "；橙点受带宽限制，绿点受算力限制", size=12, color="muted")

for name, I, dx, dy, anchor in points:
    y = min(990, 3.35 * I)
    x, yy = px(I), py(y)
    lx, ly = x + dx, yy + dy
    if abs(dx) + abs(dy) > 40:
        # leader line ends at the label corner nearest the point: the top of a label below the
        # point, the bottom of the second line for a label above it, so it never crosses the text
        ex = lx - (4 if anchor == "start" else -4 if anchor == "end" else 0)
        ey = ly - 12 if dy > 0 else ly + 19
        svg.line(x, yy, ex, ey, color="muted", width=0.8)
    svg.circle(x, yy, 4.4, fill="orange" if I < 990 / 3.35 else "green", stroke="surface", width=1)
    ival = ("%.2f" % I) if I < 1 else ("%.0f" % I if I >= 10 else "%.1f" % I)
    yval = "%.0f" % y if y >= 10 else "%.2g" % y
    svg.text(lx, ly, name, size=12, anchor=anchor, weight="600")
    svg.text(lx, ly + 15, "I = %s，%s TFLOP/s" % (ival, yval), size=11, anchor=anchor, color="muted")
    print(name, round(I, 2), "H800 attainable", round(y, 1), "A100", round(min(312, 2.039 * I), 1))

svg.save("m10-roofline.svg")
