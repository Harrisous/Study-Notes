"""Figure 17.12: how many vision tokens a page costs.

DeepSeek-OCR native modes: n_v = (R / 64)^2 -> 64 / 100 / 256 / 400 (OCR table 1); the released code adds
one newline token per row and one view separator: (s + 1) * s + 1 -> 73 / 111 / 273 / 421 (notes §4.5).
Gundam: n x 100 + 256 with n = 2..9 tiles (456 .. 1156); Gundam-M: n x 256 + 400 (912 .. 2704; not in the
released model). DeepSeek-OCR 2: k x 144 + 256, k = 0..6 (256 .. 1120).
Averages on OmniDocBench (OCR table 3): Base 182 valid, Large 285 valid, Gundam 795, Gundam-M 1853;
GOT-OCR2.0 256, Qwen2.5-VL-72B 3949, dots.ocr 5545, MinerU2.0 6790.
Right column: compression ratio for a page with 1,000 text tokens, 1000 / n_v.
Writes deepseek/figures/l17-modes.svg
"""

import math

from l17_svg import SVG
from l17_num import ocr_tokens

native = [("Tiny", 512), ("Small", 640), ("Base", 1024), ("Large", 1280)]
NV = [ocr_tokens(r) for _, r in native]
assert NV == [64, 100, 256, 400]
CODE = [(math.isqrt(n) + 1) * math.isqrt(n) + 1 for n in NV]
assert CODE == [73, 111, 273, 421]
GUNDAM = (2 * 100 + 256, 9 * 100 + 256)
GUNDAM_M = (2 * 256 + 400, 9 * 256 + 400)
OCR2 = (0 * 144 + 256, 6 * 144 + 256)
assert GUNDAM == (456, 1156) and GUNDAM_M == (912, 2704) and OCR2 == (256, 1120)

W, H = 760, 452
svg = SVG(W, H, "一页文档要多少视觉 token：DeepSeek-OCR 各模式与其他系统")

X0, X1 = 238, 640
L0, L1 = math.log10(40), math.log10(9000)


def x(v):
    return X0 + (math.log10(v) - L0) / (L1 - L0) * (X1 - X0)


TOP, RH = 58, 30
rows = [
    ("Tiny", "512²", ("pt", 64, 73, None), "blue"),
    ("Small", "640²", ("pt", 100, 111, None), "blue"),
    ("Base", "1024²，补边", ("pt", 256, 273, 182), "blue"),
    ("Large", "1280²，补边", ("pt", 400, 421, 285), "blue"),
    ("Gundam", "n 块 640² + 1024²", ("rg", 456, 1156, 795), "blue"),
    ("Gundam-M", "n 块 1024² + 1280²", ("rg", 912, 2704, 1853), "muted"),
    ("OCR 2", "k 块 768² + 1024²", ("rg", 256, 1120, None), "violet"),
    ("GOT-OCR2.0", "", ("avg", 256), "ink"),
    ("Qwen2.5-VL-72B", "", ("avg", 3949), "ink"),
    ("dots.ocr", "200dpi", ("avg", 5545), "ink"),
    ("MinerU2.0", "", ("avg", 6790), "ink"),
]
NROW = len(rows)
svg.text(20, 24, "每页视觉 token 数（对数轴）", size=13, weight="700")
svg.text(650, 24, "1000 文本 token", size=12, weight="700")
svg.text(650, 40, "的压缩比", size=12, weight="700")
for v in (50, 100, 200, 500, 1000, 2000, 5000):
    svg.line(x(v), TOP - 10, x(v), TOP + NROW * RH - 8, color="grid", width=1)
    svg.text(x(v), TOP + NROW * RH + 8, "{:,}".format(v), size=11, anchor="middle", color="muted")
svg.line(X0, TOP + NROW * RH - 8, X1, TOP + NROW * RH - 8, color="muted", width=1.2)
svg.line(20, TOP + 7 * RH - 19, 740, TOP + 7 * RH - 19, color="rule", width=1, dash="4 3")
svg.text(20, TOP - 14, "DeepSeek-OCR", size=12, color="blue", weight="700")
svg.text(20, TOP + 7 * RH - 4 - 20 + 16, "", size=11)
svg.text(22, TOP + 7 * RH - 2, "其他系统（OmniDocBench 每页平均）", size=11.5, color="muted")

for i, (name, sub, spec, col) in enumerate(rows):
    yy = TOP + i * RH + (8 if i >= 7 else 0)
    svg.text(20, yy + 5, name, size=12.5, weight="700" if i < 7 else None, color=col if col != "muted" else "ink")
    if sub:
        svg.text(118, yy + 5, sub, size=11.5, color="muted")
    kind = spec[0]
    if kind == "pt":
        _, n, code, avg = spec
        svg.line(x(n), yy, x(code), yy, color=col, width=5, opacity=0.35)
        svg.circle(x(n), yy, 5, fill=col, stroke="surface", width=1.2)
        svg.text(x(n), yy - 9, str(n), size=11.5, anchor="middle", color=col, weight="700")
        svg.text(x(code) + 7, yy + 4.5, "代码占 %d" % code, size=11, color="muted")
        if avg:
            svg.circle(x(avg), yy, 3.6, fill="surface", stroke=col, width=1.4)
            svg.text(x(avg) - 7, yy + 4.5, "均 %d" % avg, size=11, anchor="end", color=col)
        svg.text(704, yy + 4.5, "%.1f×" % (1000 / n), size=12, anchor="middle", color="orange", weight="700")
    elif kind == "rg":
        _, a, b, avg = spec
        svg.line(x(a), yy, x(b), yy, color=col, width=3)
        svg.line(x(a), yy - 6, x(a), yy + 6, color=col, width=2)
        svg.line(x(b), yy - 6, x(b), yy + 6, color=col, width=2)
        svg.text(x(a) - 7, yy + 4.5, str(a), size=11.5, anchor="end", color=col)
        svg.text(x(b) + 7, yy + 4.5, "{:,}".format(b), size=11.5, color=col)
        if avg:
            svg.circle(x(avg), yy, 4.2, fill="surface", stroke=col, width=1.6)
            svg.text(x(avg), yy - 8, "均 {:,}".format(avg), size=11, anchor="middle", color=col)
    else:
        _, v = spec
        svg.circle(x(v), yy, 4.6, fill=col, stroke="surface", width=1.2)
        svg.text(x(v) + 9, yy + 4.5, "{:,}".format(v), size=11.5, color=col)
svg.text(118, TOP + 5 * RH + 18, "开源版未提供", size=11, color="muted")
svg.text(20, TOP + NROW * RH + 34, "实心点：编码器输出的视觉 token（论文口径）；浅色延长段：开源代码实际占的位置（每行多一个换行 token，末尾一个分隔 token）；",
         size=11.5, color="muted")
svg.text(20, TOP + NROW * RH + 52, "空心点：OmniDocBench 上的平均（Base、Large 为补边后的平均有效 token）。Gundam 瓦片数 n = 2–9，OCR 2 局部视图 k = 0–6。",
         size=11.5, color="muted")
svg.save("l17-modes.svg")
