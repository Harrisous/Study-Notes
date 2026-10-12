"""Figure 16.5: propagation instability of HC on the 27B model (redrawn after mHC Fig. 3).

Data: research notes lesson-16-doc.md section 5.3 item 5 -- values DIGITISED from the vector paths of
mHC (arXiv 2512.24880 v2) Fig. 3 by the notes' author (grid-calibrated; not numbers printed in the paper).
x axis: sublayer l = 1..60 (30 blocks x {attention, FFN}).
(a) single-layer H^res_l: forward gain = max_i |row sum|, backward gain = max_j |column sum|.
(b) composite: forward = H_l ... H_1 (layers 1..l), backward = H_60 ... H_l (layers l..60).
Peak of the composite backward gain: 2825 at l = 52 (the paper's text says "peaks of 3000").
Writes deepseek/figures/l16-hc-gain.svg
"""

import math

from l16_svg import SVG

SF = [18.7, 1.02, 0.911, 0.954, 0.927, 0.644, 0.785, 0.68, 1.03, 0.809, 1.17, 0.838, 1.19, 1.12, 1.32, 0.819,
      1.01, 1.04, 1.39, 0.89, 1.38, 0.677, 1.34, 0.957, 1.16, 0.769, 1.07, 0.848, 1.16, 0.955, 1.37, 0.992, 1.59,
      1.21, 1.16, 1.43, 1.18, 0.839, 1.11, 1.02, 1.03, 1.1, 1.64, 1.14, 1.24, 1.06, 1.19, 1.2, 1.31, 0.905, 1.07,
      1.55, 1.87, 1.63, 1.47, 1.64, 2.81, 2.68, 9.02, 22.5]
SB = [6.81, 0.996, 0.871, 0.84, 1.01, 0.741, 0.937, 0.916, 1.02, 1.19, 0.94, 0.91, 0.985, 0.945, 1, 0.963, 0.785,
      0.843, 1.28, 0.925, 1.28, 0.702, 1.14, 1.23, 1.19, 0.943, 1.42, 0.95, 1.21, 0.828, 1.3, 0.734, 1.25, 0.974,
      1.31, 1.12, 1.02, 0.921, 1.19, 1.24, 1.04, 0.944, 1.77, 1.19, 1.25, 0.968, 1.22, 1.13, 1.55, 0.796, 1.09,
      1.68, 1.53, 1.4, 1.42, 1.74, 2.64, 2.83, 6.56, 12]
CF = [18.7, 26.7, 27.6, 28.8, 27.8, 26.6, 25, 22.7, 19.7, 20, 18.1, 15.2, 13.6, 9.83, 8.99, 8.99, 8.82, 9.25, 8.97,
      9.64, 9.29, 8.34, 7.7, 8.2, 8.38, 7.25, 7.78, 7.48, 7.19, 6.47, 6.4, 6.87, 6.46, 6.34, 6.72, 6.69, 6.71, 6.53,
      6.68, 5.64, 4.95, 4.93, 4.7, 5.06, 4.83, 5.24, 5.64, 5.65, 5.63, 5.23, 5.15, 6.11, 7.19, 8.4, 9.15, 11.5,
      12.2, 15.2, 34.7, 509]
CB = [259, 27.6, 24.7, 23.9, 25.6, 27, 29.9, 32, 35.9, 39.8, 46.3, 52.8, 62.7, 70.4, 82.9, 90.9, 106, 115, 118,
      94.2, 141, 190, 199, 205, 244, 263, 288, 298, 420, 390, 489, 550, 584, 720, 838, 909, 1200, 1270, 1440, 1330,
      1570, 1460, 1800, 2740, 2640, 2420, 2650, 2490, 2790, 2510, 2780, 2820, 2040, 1480, 1090, 804, 486, 216,
      79.4, 12]
assert len(SF) == len(SB) == len(CF) == len(CB) == 60
peak = max(range(60), key=lambda i: CB[i])
print("composite backward peak %.0f at l=%d; composite forward at l=60: %.0f" % (CB[peak], peak + 1, CF[-1]))
print("single-layer forward range excluding l=1,59,60: %.3f..%.3f" % (min(SF[1:58]), max(SF[1:58])))

W, H = 760, 440
svg = SVG(W, H, "HC 在 27B 模型上的增益：单层与复合（据 mHC 论文图 3 重绘，数字化值）")


def panel(x0, x1, y0, y1, e0, e1, series, title, ticks, notes):
    def px(l):
        return x0 + (l - 1) / 59 * (x1 - x0)

    def py(v):
        return y1 - (math.log10(v) - e0) / (e1 - e0) * (y1 - y0)

    svg.text(x0 - 40, 26, title, size=14, weight="700")
    for v, lab in ticks:
        y = py(v)
        svg.line(x0, y, x1, y, color="grid", width=1)
        svg.text(x0 - 7, y + 4, lab, size=11, anchor="end", color="muted")
    for l in (1, 10, 20, 30, 40, 50, 60):
        svg.line(px(l), y0, px(l), y1, color="grid", width=1)
        svg.text(px(l), y1 + 17, str(l), size=11, anchor="middle", color="muted")
    svg.line(x0, y1, x1, y1, color="muted", width=1.2)
    svg.line(x0, y0, x0, y1, color="muted", width=1.2)
    svg.text((x0 + x1) / 2, y1 + 38, "子层编号 ~l~（30 个块 × 注意力 / FFN）", size=12, anchor="middle", color="muted")
    svg.line(x0, py(1), x1, py(1), color="green", width=1.4, dash="5 4")
    for vals, color, width, dash in series:
        d = "M" + " L".join("%.1f %.1f" % (px(i + 1), py(v)) for i, v in enumerate(vals))
        svg.path(d, stroke=color, width=width, dash=dash, join="round")
    notes(px, py)


def notes_a(px, py):
    svg.text(px(2.5), py(18.7) + 4, "~l~ = 1：前向 18.7，反向 6.81", size=12, color="ink")
    svg.text(px(60) - 4, py(22.5) - 8, "~l~ = 60：22.5", size=12, anchor="end", color="ink")
    svg.text(px(30), py(0.5) + 4, "绿虚线：增益 = 1", size=12, anchor="middle", color="green")


def notes_b(px, py):
    svg.circle(px(peak + 1), py(CB[peak]), 4, fill="red")
    # notes: SVG re-digitisation gives 2824.7 at l = 52 (the list above is rounded to 3 digits)
    tx, ty = px(4), py(6500)
    svg.text(tx, ty, "反向峰值 ≈ 2825（~l~ = %d，数字化）" % (peak + 1), size=12, color="red", weight="600")
    svg.text(tx, ty + 17, "论文文字：峰值约 3000", size=12, color="red")
    svg.line(px(34), ty + 12, px(peak + 1) - 5, py(CB[peak]) - 3, color="red", width=1, dash="3 3")
    svg.circle(px(60), py(CF[-1]), 3.5, fill="ink")
    # label kept clear of the falling red curve (red is >= 216 for l <= 58, black <= 15.2 there)
    svg.text(px(57.6), py(40), "~l~ = 60 的圆点：509", size=12, anchor="end", color="ink")
    svg.text(px(2), py(259) - 8, "259", size=12, color="red")


LOG_A = [(0.5, "0.5"), (1, "1"), (2, "2"), (5, "5"), (10, "10"), (20, "20")]
panel(60, 350, 50, 345, math.log10(0.4), math.log10(30),
      [(SF, "ink", 1.6, None), (SB, "red", 1.9, None)], "(a) 单层 H^{res}_{l}（对数纵轴）", LOG_A, notes_a)
LOG_B = [(1, "1"), (10, "10"), (100, "100"), (1000, "1000"), (10000, "10^{4}")]
panel(450, 740, 50, 345, 0, 4, [(CF, "ink", 1.6, None), (CB, "red", 2.2, None)],
      "(b) 复合映射（对数纵轴）", LOG_B, notes_b)

# legend
LY = 402
svg.line(170, LY, 200, LY, color="ink", width=1.8)
svg.text(206, LY + 4, "前向增益：最大 |行和|", size=12)
svg.line(420, LY, 450, LY, color="red", width=2.2)
svg.text(456, LY + 4, "反向增益：最大 |列和|", size=12, color="red")
svg.text(380, LY + 24, "(b) 中前向 = 第 1..~l~ 个子层之积，反向 = 第 ~l~..60 个子层之积", size=12, anchor="middle",
         color="muted")

svg.save("l16-hc-gain.svg")
