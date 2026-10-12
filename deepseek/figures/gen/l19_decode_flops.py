"""Figure 19.8: single-token decode FLOPs vs context length, redrawn after V4.1 paper Fig. 2.

Data: the four curves digitised from the paper's PNG (notes 4-F, about +-3%; weighted FLOPs with
BF16 = 1, FP8 = 0.5, FP4 = 0.25). Thin dashed lines: the author's analytic growth terms,
F(T) = F(4K) + g (T - 4096) with g = 5,120 (V4.1) and 41,984 (V4-Flash) weighted FLOPs per context token.
Writes deepseek/figures/l19-decode-flops.svg
"""

import math

from l19_svg import SVG
from l19_num import CTX_K, FIG2, GROW_41, GROW_4F, CROSS

W, H = 740, 400
svg = SVG(W, H, "单 token 解码 FLOPs 随上下文长度的变化（据 V4.1 论文图 2 重绘）")

PX0, PX1, PY0, PY1 = 76, 560, 30, 330
X_MIN, X_MAX = math.log2(4), math.log2(1024)
Y_MIN, Y_MAX = math.log10(10), math.log10(4000)


def xs(k):
    return PX0 + (math.log2(k) - X_MIN) / (X_MAX - X_MIN) * (PX1 - PX0)


def ys(g):
    return PY1 - (math.log10(g) - Y_MIN) / (Y_MAX - Y_MIN) * (PY1 - PY0)


for g in (10, 30, 100, 300, 1000, 3000):
    svg.line(PX0, ys(g), PX1, ys(g), color="grid", width=1)
    svg.text(PX0 - 8, ys(g) + 4, "{:,}".format(g), size=11, anchor="end", color="muted")
for k in (4, 16, 64, 256, 1024):
    svg.line(xs(k), PY0, xs(k), PY1, color="grid", width=1)
    svg.text(xs(k), PY1 + 17, "%dK" % k if k < 1024 else "1M", size=11, anchor="middle", color="muted")
svg.line(PX0, PY1, PX1, PY1, color="muted", width=1)
svg.line(PX0, PY0, PX0, PY1, color="muted", width=1)
svg.text((PX0 + PX1) / 2, PY1 + 38, "上下文长度（token，对数刻度）", size=12, anchor="middle", color="muted")
svg.text(22, (PY0 + PY1) / 2, "单 token 解码 GFLOPs（对数刻度，加权）", size=12, anchor="middle", color="muted",
         rotate=-90)

STYLE = {
    "DeepSeek-V1 (67B)": ("muted", 2.0, "7 5", None),
    "DeepSeek-V3.2": ("ink", 2.0, "9 4 2 4", 0.75),
    "DeepSeek-V4-Flash": ("blue", 2.4, None, 0.45),
    "DeepSeek-V4.1-Flash": ("blue", 3.0, None, None),
}
for name, vals in FIG2.items():
    col, w, dash, op = STYLE[name]
    d = "M" + " L".join("%.1f %.1f" % (xs(k), ys(v)) for k, v in zip(CTX_K, vals))
    svg.path(d, stroke=col, width=w, dash=dash, opacity=op)
    for k, v in zip(CTX_K, vals):
        svg.circle(xs(k), ys(v), 2.2, fill=col, fill_opacity=op if op else 1)

# analytic growth terms (author's model)
for name, g in (("DeepSeek-V4.1-Flash", GROW_41), ("DeepSeek-V4-Flash", GROW_4F)):
    base = FIG2[name][0]
    pts = []
    for i in range(0, 101):
        k = 2 ** (X_MIN + i / 100 * (X_MAX - X_MIN))
        pts.append((xs(k), ys(base + g * (k * 1024 - 4096) / 1e9)))
    svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke="orange", width=1.2, dash="3 3")

# crossover
kx = CROSS / 1024
svg.line(xs(kx), ys(12), xs(kx), ys(40), color="red", width=1, dash="2 3")
svg.text(xs(kx), ys(44), "交叉点 ≈ 100K", size=11, anchor="middle", color="red")

# right-hand labels
LX = PX1 + 12
labels = [("DeepSeek-V1 (67B)", 3359, "V1（67B）：3,359"), ("DeepSeek-V3.2", 591, "V3.2：591"),
          ("DeepSeek-V4-Flash", 60.1, "V4-Flash：60.1"), ("DeepSeek-V4.1-Flash", 25.6, "V4.1-Flash：25.6")]
for name, v, lab in labels:
    col, w, dash, op = STYLE[name]
    svg.text(LX, ys(v) + 4, lab, size=12, color=col, weight="700" if "4.1" in name else None)
svg.text(LX, ys(60.1) + 22, "（1M 处，GFLOPs）", size=11, color="muted")
svg.text(xs(4) + 6, ys(20.1) - 9, "20.1", size=11, color="blue")
svg.text(xs(4) + 6, ys(16.4) + 16, "16.4", size=11, color="blue")

# legend for the analytic lines
svg.line(PX0 + 12, PY0 + 16, PX0 + 42, PY0 + 16, color="orange", width=1.2, dash="3 3")
svg.text(PX0 + 50, PY0 + 20, "笔者的解析增长项：V4.1 +5,120·T，V4-Flash +41,984·T", size=12, color="orange")

svg.save("l19-decode-flops.svg")
