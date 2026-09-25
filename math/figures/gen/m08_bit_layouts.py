"""Figure M8.2: bit layouts of FP32, BF16, FP16, FP8 (E5M2, E4M3), FP4 (E2M1) and E8M0,
drawn to scale (one cell = one bit), with bias and largest finite value computed from
the layout by m08_fp.params (E8M0: 2^(254-127) = 2^127).
Writes math/figures/m08-bit-layouts.svg
"""

from m08_fp import params
from m08_svg import SVG

ROWS = [
    # name, sub, sign bits, exponent bits, mantissa bits, key in m08_fp (or None)
    ("FP32", "单精度", 1, 8, 23, "FP32"),
    ("BF16", "bfloat16", 1, 8, 7, "BF16"),
    ("FP16", "半精度", 1, 5, 10, "FP16"),
    ("FP8 E5M2", "", 1, 5, 2, "E5M2"),
    ("FP8 E4M3", "", 1, 4, 3, "E4M3"),
    ("FP4 E2M1", "", 1, 2, 1, "E2M1"),
    ("E8M0", "只作缩放因子", 0, 8, 0, None),
]

CELL = 14
X0 = 132
RH = 50
Y0 = 58
W, H = 760, Y0 + RH * len(ROWS) + 34
svg = SVG(W, H, "七种数值格式的位布局（按位数等比例）")


def sci(v):
    if v >= 1e5:
        e = len(str(int(v))) - 1
        e = int(__import__("math").floor(__import__("math").log10(v)))
        return "%.2f×10^{%d}" % (v / 10 ** e, e)
    if v == int(v):
        return "{:,}".format(int(v))
    return "%g" % v


# legend
lx = X0
for label, col in (("符号 S", "muted"), ("指数 e", "blue"), ("尾数 m", "green")):
    svg.rect(lx, 18, 16, 16, fill=col, fill_opacity=0.25, stroke=col, width=1)
    svg.text(lx + 22, 31, label, size=12)
    lx += 92
svg.text(lx + 10, 31, "每一小格 = 1 位", size=12, color="muted")
svg.text(W - 16, 31, "偏置 · 最大有限值", size=12, anchor="end", color="muted")

for r, (name, sub, S, E, M, key) in enumerate(ROWS):
    y = Y0 + r * RH
    total = S + E + M
    svg.text(X0 - 12, y + 16, name, size=13, anchor="end", weight="600")
    svg.text(X0 - 12, y + 31, "%d 位%s" % (total, (" · " + sub) if sub else ""), size=11,
             anchor="end", color="muted")
    x = X0
    for n, col, lab in ((S, "muted", ""), (E, "blue", "指数 %d" % E), (M, "green", "尾数 %d" % M)):
        if n == 0:
            continue
        svg.rect(x, y + 4, n * CELL, 24, fill=col, fill_opacity=0.22, stroke=col, width=1.2)
        for k in range(1, n):
            svg.line(x + k * CELL, y + 4, x + k * CELL, y + 28, color=col, width=0.6, opacity=0.6)
        if lab and n * CELL >= 60:
            svg.text(x + n * CELL / 2, y + 20.5, lab, size=12, anchor="middle")
        elif lab:
            svg.text(x + n * CELL / 2, y + 42, str(n), size=11, anchor="middle", color=col)
        x += n * CELL
    # right-hand facts
    if key:
        p = params(key)
        facts = "偏置 %d · 最大 %s" % (p["bias"], sci(p["max"]))
    else:
        facts = "偏置 127 · 只表示 2 的幂，2^{-127} 到 2^{127}"
    svg.text(W - 16, y + 21, facts, size=12, anchor="end")

svg.text(X0, H - 12, "BF16 = FP32 的前 16 位；E5M2 = FP16 的前 8 位（同一偏置 15）；E8M0 没有符号位和尾数位",
         size=12, color="muted")
svg.save("m08-bit-layouts.svg")
