"""Figure 12.2: bit layouts of the floating-point formats used in DeepSeek-V3 training.

One cell per bit, drawn to scale: sign / exponent / mantissa. The right-hand columns are
computed from the format definitions (IEEE-style: max = (2 - 2^-m) * 2^(2^(e-1)-1) except
E4M3, which reclaims the top exponent and keeps only S.1111.111 as NaN, giving 448;
FP8 values as in "FP8 Formats for Deep Learning", Table 1). E5M6 is DeepSeek's custom
format; the V3 report gives no bias or special values, so only its mantissa spacing is shown.
Writes deepseek/figures/l12-float-bits.svg
"""

from l12_svg import SVG

rows = [
    # name, e, m, note
    ("FP32", 8, 23, "主权重、权重梯度、FP32 累加"),
    ("BF16", 8, 7, "Adam 的两个矩、核心注意力、combine 通信"),
    ("FP16", 5, 10, "（对照用）"),
    ("E5M6", 5, 6, "V3 自定义：只存注意力后那个 Linear 的输入"),
    ("E5M2", 5, 2, "FP8 论文建议用于梯度（V3 没有用）"),
    ("E4M3", 4, 3, "V3：所有 FP8 GEMM 的输入都用它"),
]


def props(name, e, m):
    bias = 2 ** (e - 1) - 1
    eps = 2.0 ** -m
    if name == "E4M3":
        mx = 1.75 * 2 ** 8
        assert mx == 448
    elif name == "E5M6":
        return None, None, eps
    else:
        mx = (2 - 2.0 ** -m) * 2.0 ** bias
    return mx, 2.0 ** (1 - bias), eps


def fmt_max(v):
    if v is None:
        return "未给"
    if v > 1e6:
        s = "%.2e" % v
        a, b = s.split("e+")
        return "%s×10^{%d}" % (a, int(b))
    return "{:,}".format(int(v))


def fmt_pow(v):
    if v is None:
        return "未给"
    import math
    k = round(math.log2(v))
    return "2^{%d}" % k


W = 760
ROW = 50
TOP = 58
H = TOP + ROW * len(rows) + 30
CELL = 11
BX = 96
CX = [566, 640, 712]

svg = SVG(W, H, "浮点格式的位布局：符号、指数、尾数，以及由此决定的范围与精度")
# header
svg.text(BX, 22, "每格 1 位（按比例）", size=12, color="muted")
lx = BX + 150
svg.rect(lx, 12, 12, 12, fill="muted", fill_opacity=0.35, stroke="muted", width=0.8)
svg.text(lx + 17, 22, "符号", size=12)
svg.rect(lx + 56, 12, 12, 12, fill="violet", fill_opacity=0.28, stroke="violet", width=0.8)
svg.text(lx + 73, 22, "指数：管范围", size=12)
svg.rect(lx + 162, 12, 12, 12, fill="blue", fill_opacity=0.22, stroke="blue", width=0.8)
svg.text(lx + 179, 22, "尾数：管精度", size=12)
for x, t in zip(CX, ("最大有限值", "最小正规数", "1 附近间距")):
    svg.text(x, 48, t, size=12, anchor="middle", color="muted", weight="600")
svg.line(16, 56, W - 12, 56, color="rule", width=1)

for i, (name, e, m, note) in enumerate(rows):
    y = TOP + i * ROW + 8
    low = name in ("E4M3", "E5M2")
    custom = name == "E5M6"
    svg.text(16, y + 13, name, size=14, weight="700", color="orange" if (low or custom) else "ink")
    svg.text(16, y + 30, "1 / %d / %d" % (e, m), size=11, color="muted")
    bits = [("muted", 0.35)] + [("violet", 0.28)] * e + [("blue", 0.22)] * m
    for j, (col, op) in enumerate(bits):
        svg.rect(BX + j * CELL, y, CELL - 1, 18, fill=col, fill_opacity=op, stroke=col, width=0.6)
    if low or custom:
        svg.rect(BX - 3, y - 3, len(bits) * CELL + 5, 24, stroke="orange", width=1.3, rx=3,
                 dash="4 3" if custom else None)
    svg.text(BX, y + 36, note, size=11, color="muted")
    mx, mn, eps = props(name, e, m)
    svg.text(CX[0], y + 14, fmt_max(mx), size=12, anchor="middle")
    svg.text(CX[1], y + 14, fmt_pow(mn), size=12, anchor="middle")
    svg.text(CX[2], y + 14, fmt_pow(eps), size=12, anchor="middle")
    if i:
        svg.line(16, y - 8, W - 12, y - 8, color="grid", width=1)

svg.text(16, H - 8, "橙实框 = FP8；橙虚框 = V3 自定义的 12 位格式。E4M3 把指数全 1 的码字也用作有限数，所以最大值是 448 而不是 240。",
         size=11, color="muted")
svg.save("l12-float-bits.svg")
