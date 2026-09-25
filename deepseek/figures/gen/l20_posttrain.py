"""Figure 20.5 (summary of thread D): (a) model-state memory of four post-training methods for a 7B
policy, by network, with (3.26) 16 B per trained and 2 B per frozen parameter (lesson 3 p6);
(b) post-training GPU hours as a share of DeepSeek-V3's 2,664K pre-training hours (lesson 12 p1,
lesson 13 p8, lesson 15 p1), log axis.
Writes deepseek/figures/l20-posttrain.svg
"""

import math

from l20_svg import SVG
from l20_num import MEM, N7, POST, PRE_V3

W, H = 760, 440
svg = SVG(W, H, "主线 D 的总账：后训练的显存与算力")

svg.text(10, 22, "(a) 7B 策略的模型状态显存（GB，按 16 字节/训练参数、2 字节/冻结参数）", size=13, weight="700")
X0, X1 = 110, 610


def xm(v):
    return X0 + v / 260 * (X1 - X0)


TOP, RH = 58, 36
for v in (0, 50, 100, 150, 200, 250):
    svg.line(xm(v), TOP - 8, xm(v), TOP + 4 * RH - 6, color="grid", width=1)
    svg.text(xm(v), TOP - 12, str(v), size=11, anchor="middle", color="muted")
PARTS = [("策略（训练）", "blue", 0.6), ("价值网络（训练）", "red", 0.35), ("参考模型（冻结）", "muted", 0.35),
         ("奖励模型（冻结）", "muted", 0.18)]
for i, (name, pol, frozen, val) in enumerate(MEM):
    y = TOP + i * RH
    svg.text(X0 - 10, y + 16, name, size=12, anchor="end", weight="700")
    segs = [(16 * N7 * pol, PARTS[0]), (16 * N7 * val, PARTS[1]), (2 * N7 * min(frozen, 1), PARTS[2]),
            (2 * N7 * max(frozen - 1, 0), PARTS[3])]
    x = 0
    for v, (lab, col, op) in segs:
        if v <= 0:
            continue
        svg.rect(xm(x), y + 4, xm(x + v) - xm(x), 18, fill=col, fill_opacity=op, stroke=col, width=0.8)
        x += v
    svg.text(xm(x) + 6, y + 17, "%d GB" % x, size=12, weight="700")
lx = X0
for lab, col, op in PARTS:
    svg.rect(lx, TOP + 4 * RH + 2, 12, 12, fill=col, fill_opacity=op, stroke=col, width=0.8)
    svg.text(lx + 17, TOP + 4 * RH + 13, lab, size=12)
    lx += 128

# (b)
Y2 = TOP + 4 * RH + 50
svg.text(10, Y2, "(b) 后训练 GPU 小时占 V3 预训练（2,664K H800 小时）的比例（对数刻度）", size=13, weight="700")
XB0, XB1 = 330, 700


def xl(p):          # p in percent, 0.1% .. 30%
    return XB0 + (math.log10(p) + 1) / (math.log10(30) + 1) * (XB1 - XB0)


T2 = Y2 + 34
for p in (0.1, 1, 10):
    svg.line(xl(p), T2 - 8, xl(p), T2 + 3 * 40 - 10, color="grid", width=1)
    svg.text(xl(p), T2 - 12, ("%g%%" % p), size=11, anchor="middle", color="muted")
for i, (name, hours, where) in enumerate(POST):
    y = T2 + i * 40
    p = hours / PRE_V3 * 100
    lower = "下限" in name
    svg.text(XB0 - 10, y + 12, name, size=12, anchor="end")
    amount = ("%gK GPU 小时" % round(hours, 1)) if not lower else ("> %gK GPU 小时（按 V3 基数）" % round(hours, 1))
    svg.text(XB0 - 10, y + 27, "%s · %s" % (amount, where), size=11, anchor="end", color="muted")
    svg.rect(XB0, y + 4, xl(p) - XB0, 18, fill="green", fill_opacity=0.25 if lower else 0.5, stroke="green", width=0.8,
             dash="4 3" if lower else None)
    if lower:
        svg.arrow(xl(p), y + 13, xl(p) + 36, y + 13, color="green", width=1.6)
    svg.text(xl(p) + (42 if lower else 6), y + 17, ("> 10%" if lower else ("%.3g%%" % p if p < 1 else "%.1f%%" % p)), size=12, weight="700")

svg.text(10, H - 8, "(b) 三行的口径不同：V3 与 R1 是论文列出的 GPU 小时；V3.2 只说“超过预训练成本的 10%”，基数与分子都没有披露。",
         size=12, color="muted")

svg.save("l20-posttrain.svg")
