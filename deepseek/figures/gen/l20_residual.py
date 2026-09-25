"""Figure 20.6 (summary of thread E): (a) composite gain of the residual shortcut after 60 sub-layers,
log axis centred on 1; (b) residual-stream memory traffic per sub-layer in multiples of d.

(a) from l20_num.GAIN: scalar products are computed; the HC (2825) and mHC (1.65) values are the
digitised readings of the mHC paper figures quoted in lesson 16 p3 / p7; the bound n = 4 is proved in
lesson 16 p4. (b) from l20_num.TRAFFIC (lesson 16 p7 table; V4 / V4.1 include the pre-norm).
Writes deepseek/figures/l20-residual.svg
"""

import math

from l20_svg import SVG
from l20_num import GAIN, TRAFFIC

W, H = 760, 540
svg = SVG(W, H, "主线 E 的总账：捷径上的“1”，以及让多条残差流跑得动的访存账")

svg.text(10, 22, "(a) 60 个子层之后捷径的复合增益（对数刻度，1 在中间）", size=13, weight="700")
X0, X1 = 300, 720
LMIN, LMAX = -2, 4


def xs(v):
    return X0 + (math.log10(v) - LMIN) / (LMAX - LMIN) * (X1 - X0)


TOP, RH = 58, 32
for e in range(LMIN, LMAX + 1):
    x = xs(10 ** e)
    svg.line(x, TOP - 8, x, TOP + len(GAIN) * RH - 6, color="ink" if e == 0 else "grid", width=1.2 if e == 0 else 1)
    svg.text(x, TOP - 12, ("%g" % 10 ** e) if e < 4 else "10,000", size=11, anchor="middle", color="muted")
for i, (name, g, where) in enumerate(GAIN):
    y = TOP + i * RH
    col = "muted"
    if name.startswith("HC"):
        col = "red"
    elif name.startswith("mHC"):
        col = "green"
    elif "上界" in name:
        col = "violet"
    svg.text(X0 - 10, y + 16, name, size=12, anchor="end")
    x1 = xs(1)
    x2 = xs(g)
    if abs(x2 - x1) < 1:
        svg.circle(x1, y + 12, 4, fill=col, stroke=col)
    else:
        svg.rect(min(x1, x2), y + 4, abs(x2 - x1), 16, fill=col, fill_opacity=0.5, stroke=col, width=0.8,
                 dash="4 3" if "上界" in name else None)
    txt = ("%.3f" % g) if g < 0.1 else ("{:,.0f}".format(g) if g >= 100 else ("%.3g" % g))
    if g < 1:
        svg.text(x2 - 6, y + 16, txt, size=12, anchor="end", weight="700")
    else:
        svg.text(max(x1, x2) + 6, y + 16, txt, size=12, weight="700")

Y2 = TOP + len(GAIN) * RH + 30
svg.text(10, Y2, "(b) 每个 token、每个子层的残差流访存（d 的倍数，n = 4）", size=13, weight="700")
XB0, XB1 = 300, 660


def xt(v):
    return XB0 + v / 36 * (XB1 - XB0)


T2 = Y2 + 32
for v in (0, 10, 20, 30):
    svg.line(xt(v), T2 - 8, xt(v), T2 + len(TRAFFIC) * 30 - 8, color="grid", width=1)
    svg.text(xt(v), T2 - 12, "%dd" % v if v else "0", size=11, anchor="middle", color="muted")
for i, (name, v) in enumerate(TRAFFIC):
    y = T2 + i * 30
    col = "green" if "V4.1" in name else ("red" if "未融合" in name else "amber")
    svg.text(XB0 - 10, y + 14, name, size=12, anchor="end")
    svg.rect(XB0, y + 2, xt(v) - XB0, 16, fill=col, fill_opacity=0.5, stroke=col, width=0.8)
    svg.text(xt(v) + 6, y + 15, "%dd" % v, size=12, weight="700")

svg.text(10, H - 26, "(a) HC 与 mHC 两行是 27B 模型的反向复合增益，取自对论文图的数字化读数（第 16 课 §3、§7）；其余由公式计算。",
         size=12, color="muted")
svg.text(10, H - 8, "(b) 前三行不含子层自己的 pre-norm，V4 与 V4.1 两行含；mHC 的 18d 是第 16 课的推算（§7 总表）。", size=12,
         color="muted")

svg.save("l20-residual.svg")
