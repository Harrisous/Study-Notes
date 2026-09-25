"""Figure 18.15: MRCR 8-needle (average MMR) vs context length, redrawn from Fig. 9 of the V4 report.
Data labels read from the paper's SVG and re-checked in a browser (research notes C74 and sec. 10.1
item 22): Pro-Max and Flash-Max at 8K .. 1024K.
Writes deepseek/figures/l18-mrcr.svg
"""

import math

from l18_svg import SVG

LENS = [8, 16, 32, 64, 128, 256, 512, 1024]
PRO = [0.90, 0.85, 0.94, 0.90, 0.92, 0.82, 0.66, 0.59]
FLASH = [0.91, 0.84, 0.87, 0.85, 0.87, 0.76, 0.60, 0.49]
assert round(sum(PRO) / 8, 4) == 0.8225 and round(sum(FLASH) / 8, 3) == 0.774

W, H = 700, 360
svg = SVG(W, H, "MRCR 8-needle 得分随上下文长度的变化")
X0, X1, Y0, Y1 = 70, 600, 30, 290


def px(k):
    return X0 + (math.log2(k) - 3) / 7 * (X1 - X0)


def py(v):
    return Y1 - (v - 0.4) / 0.6 * (Y1 - Y0)


svg.rect(px(128), Y0, X1 - px(128), Y1 - Y0, fill="red", fill_opacity=0.05)
svg.text(px(128) + 6, Y0 + 14, "128K 以外：逐段下降", size=11, color="red")
for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    svg.line(X0, py(v), X1, py(v), color="grid", width=1)
    svg.text(X0 - 8, py(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
for k in LENS:
    svg.line(px(k), Y0, px(k), Y1, color="grid", width=1)
    svg.text(px(k), Y1 + 18, "%dK" % k, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "上下文长度（对数刻度，每格 2 倍）", size=12, anchor="middle", color="muted")
svg.text(X0 - 56, Y0 - 12, "平均 MMR", size=12, color="muted")
for vals, col, name, dy in ((PRO, "blue", "V4-Pro-Max", -8), (FLASH, "violet", "V4-Flash-Max", 16)):
    svg.path("M" + " L".join("%.1f %.1f" % (px(k), py(v)) for k, v in zip(LENS, vals)), stroke=col, width=2.4)
    for k, v in zip(LENS, vals):
        svg.circle(px(k), py(v), 3.6, fill=col)
    svg.text(px(1024) + 8, py(vals[-1]) + 4, name, size=12, color=col, weight="700")
for n, (k, v) in enumerate(zip(LENS, PRO)):
    above = PRO[n] >= FLASH[n]            # label the higher curve above, the lower one below
    svg.text(px(k), py(v) - 9 if above else py(v) + 18, "%.2f" % v, size=11, anchor="middle", color="blue")
for n, (k, v) in enumerate(zip(LENS, FLASH)):
    above = FLASH[n] > PRO[n]
    svg.text(px(k), py(v) - 9 if above else py(v) + 18, "%.2f" % v, size=11, anchor="middle", color="violet")
svg.text(20, 348, "表格里的“MRCR 1M = 83.5”（Pro-Max）是另一种未定义的汇总口径，不是 1M 处的 0.59，也不是这 8 个点的平均 0.82。", size=11,
         color="muted")

svg.save("l18-mrcr.svg")
