"""Figure 12.14: pipeline bubbles of 1F1B, ZB1P and DualPipe from the V3 report's Table 2.

Formulas (course notation; V3 report Table 2, B = full backward):
    1F1B      (P - 1)(T_F + T_B)
    ZB1P      (P - 1)(T_F + T_B - 2 T_W)
    DualPipe  (P/2 - 1)(T_F&B + T_B - 3 T_W)
Unit costs: T_F = 1, T_B = 2, T_W = 1, communication ignored; T_F&B = 3 (overlap does not speed up
compute) or 2.5 (overlap saves half a unit). Left: bubble per rank for P = 8 and 16. Right:
bubble share of the step, bubble / (3C + bubble), for P = 16 as the number of micro-batches C
grows (each rank computes C forwards and C backwards in all three schemes).
The measured V3 figure (Insights Table 4: bubble 2.06 s of a 19.926 s step) is drawn only as a
reference line; its C and communication costs are not public, so it is not directly comparable.
Writes deepseek/figures/l12-bubble-compare.svg
"""

import math

from l12_svg import SVG

TF, TB, TW = 1, 2, 1


def b1f1b(P):
    return (P - 1) * (TF + TB)


def bzb(P):
    return (P - 1) * (TF + TB - 2 * TW)


def bdual(P, tfb=3.0):
    return (P / 2 - 1) * (tfb + TB - 3 * TW)


assert [b1f1b(16), bzb(16), bdual(16), bdual(16, 2.5)] == [45, 15, 14, 10.5]
assert [b1f1b(8), bzb(8), bdual(8), bdual(8, 2.5)] == [21, 7, 6, 4.5]
measured = 2.06 / 19.926

W, H = 760, 380
svg = SVG(W, H, "三种流水线调度的气泡：绝对格数（左）与占一步时间的比例（右）")

# ---- left panel: grouped bars
LX0, LX1, LY0, LY1 = 60, 360, 40, 290
ymax = 50


def ly(v):
    return LY1 - v / ymax * (LY1 - LY0)


for v in range(0, ymax + 1, 10):
    svg.line(LX0, ly(v), LX1, ly(v), color="grid", width=1)
    svg.text(LX0 - 6, ly(v) + 4, str(v), size=11, anchor="end", color="muted")
svg.line(LX0, LY1, LX1, LY1, color="muted", width=1.2)
series = [("1F1B", "red", b1f1b), ("ZB1P", "violet", bzb), ("DualPipe", "blue", bdual),
          ("DualPipe*", "green", lambda P: bdual(P, 2.5))]
bw = 26
for gi, P in enumerate((8, 16)):
    gx = LX0 + 24 + gi * 150
    for si, (name, col, f) in enumerate(series):
        v = f(P)
        x = gx + si * (bw + 4)
        svg.rect(x, ly(v), bw, LY1 - ly(v), fill=col, fill_opacity=0.28, stroke=col, width=1.1)
        svg.text(x + bw / 2, ly(v) - 5, ("%g" % v), size=11, anchor="middle", color=col, weight="600")
    svg.text(gx + 2 * (bw + 4) - 2, LY1 + 18, "P = %d" % P, size=12, anchor="middle")
# legend
lgy = 330
for si, (name, col, f) in enumerate(series):
    x = LX0 + (si % 2) * 150
    y = lgy + (si // 2) * 20
    svg.rect(x, y - 10, 12, 12, fill=col, fill_opacity=0.28, stroke=col, width=1)
    lab = {"1F1B": "1F1B", "ZB1P": "ZB1P（= ZB-H1 的公式）", "DualPipe": "DualPipe，T_{F&B} = 3",
           "DualPipe*": "DualPipe，T_{F&B} = 2.5"}[name]
    svg.text(x + 17, y, lab, size=11)

# ---- right panel: share of the step vs C, P = 16
RX0, RX1, RY0, RY1 = 450, 730, 40, 290
Cs = [32, 48, 64, 96, 128, 192, 256]
ymax2 = 0.35


def rx(C):
    return RX0 + (math.log2(C) - 5) / (8 - 5) * (RX1 - RX0)


def ry(v):
    return RY1 - v / ymax2 * (RY1 - RY0)


for v in (0, 0.1, 0.2, 0.3):
    svg.line(RX0, ry(v), RX1, ry(v), color="grid", width=1)
    svg.text(RX0 - 6, ry(v) + 4, "%d%%" % round(100 * v), size=11, anchor="end", color="muted")
for C in (32, 64, 128, 256):
    svg.text(rx(C), RY1 + 18, str(C), size=11, anchor="middle", color="muted")
    svg.line(rx(C), RY0, rx(C), RY1, color="grid", width=1)
svg.line(RX0, RY1, RX1, RY1, color="muted", width=1.2)
svg.text((RX0 + RX1) / 2, RY1 + 36, "micro-batch 数 C（对数刻度），P = 16", size=12, anchor="middle", color="muted")
for name, col, f in series[:3]:
    pts = [(C, f(16) / (3 * C + f(16))) for C in Cs]
    svg.path("M" + " L".join("%.1f %.1f" % (rx(C), ry(v)) for C, v in pts), stroke=col, width=2.2)
    svg.circle(rx(32), ry(pts[0][1]), 3.2, fill=col)
    dy = {"1F1B": -6, "ZB1P": -8, "DualPipe": 64}[name]
    svg.text(rx(32) + 8, ry(pts[0][1]) + dy, "%s %.1f%%" % (name, 100 * pts[0][1]), size=11, color=col, weight="600")
svg.line(RX0, ry(measured), RX1, ry(measured), color="ink", width=1.2, dash="5 4")
svg.text((RX0 + RX1) / 2, RY1 + 56, "虚线：由 Insights 表 4 算出的 %.1f%%" % (100 * measured), size=11, anchor="middle")
svg.text((RX0 + RX1) / 2, RY1 + 72, "（2.06 s ÷ 19.926 s；含通信，C 未公开）", size=11, anchor="middle", color="muted")
svg.text(RX0 - 40, 24, "气泡占一步时间的比例", size=12, color="muted")
svg.text(LX0 - 40, 24, "每个 rank 的气泡（单位 T_{F}）", size=12, color="muted")
svg.save("l12-bubble-compare.svg")
