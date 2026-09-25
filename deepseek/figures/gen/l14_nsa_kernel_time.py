"""Figure 14.7: NSA kernel time vs FlashAttention-2 (redrawn from NSA Fig. 6, 8xA100, Triton).

Bar heights were recovered from the geometry of the paper's vector figure (lesson-14 research
notes, section 4.1); speedups FA2 / (selection + compression-and-window) are recomputed here and
match the labels printed in the paper (2.1/3.8/6.3/9.0 forward, 1.1/2.0/3.4/6.0 backward).
Writes deepseek/figures/l14-nsa-kernel-time.svg
"""

import math

from l14_svg import SVG

LABELS = ["8K", "16K", "32K", "64K"]
FWD = {"fa2": [16.3, 63.8, 249.5, 998.1], "sel": [5.5, 11.1, 23.5, 47.0], "cw": [2.3, 5.8, 15.8, 63.3]}
BWD = {"fa2": [42.5, 165.6, 647.8, 2591.5], "sel": [28.3, 59.8, 124.4, 250.9], "cw": [10.3, 24.2, 64.1, 180.5]}
PRINTED_F = [2.1, 3.8, 6.3, 9.0]
PRINTED_B = [1.1, 2.0, 3.4, 6.0]
for data, printed in ((FWD, PRINTED_F), (BWD, PRINTED_B)):
    for i in range(4):
        sp = data["fa2"][i] / (data["sel"][i] + data["cw"][i])
        assert abs(sp - printed[i]) < 0.06, (sp, printed[i])

W, H = 760, 400
svg = SVG(W, H, "NSA 注意力核与 FlashAttention-2 的耗时（据 NSA 图 6 重绘，对数刻度）")
LY0, LY1 = 0, 4            # 1 ms .. 10,000 ms


def panel(x0, x1, title, data, printed):
    y0, y1 = 60, 320

    def py(v):
        return y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (y1 - y0)

    svg.text((x0 + x1) / 2, 30, title, size=13, anchor="middle", weight="700")
    for e in range(LY0, LY1 + 1):
        y = py(10 ** e)
        svg.line(x0, y, x1, y, color="grid", width=1)
        svg.text(x0 - 8, y + 4, ["1", "10", "100", "1000", "10000"][e], size=11, anchor="end", color="muted")
    svg.line(x0, y1, x1, y1, color="muted", width=1.2)
    gw = (x1 - x0) / 4
    for i in range(4):
        gx = x0 + i * gw + 10
        bw = (gw - 26) / 3
        for k, (key, col) in enumerate((("fa2", "blue"), ("sel", "violet"), ("cw", "orange"))):
            v = data[key][i]
            svg.rect(gx + k * (bw + 3), py(v), bw, y1 - py(v), fill=col, fill_opacity=0.65, stroke=col, width=0.8)
        top = py(data["fa2"][i])
        svg.text(gx + 1.5 * bw + 3, top - 8, "%.1f×" % printed[i], size=12, anchor="middle", weight="700")
        svg.text(gx + 1.5 * bw + 3, y1 + 18, LABELS[i], size=12, anchor="middle")
    svg.text(x0 - 44, (y0 + y1) / 2, "耗时 ms（对数）", size=12, anchor="middle", color="muted", rotate=-90)


panel(80, 380, "前向", FWD, PRINTED_F)
panel(450, 750, "反向", BWD, PRINTED_B)

ly = 360
for k, (lab, col) in enumerate((("FlashAttention-2（全注意力）", "blue"), ("NSA 选择分支核", "violet"),
                                 ("NSA 压缩 + 窗口分支", "orange"))):
    x = 80 + k * 230
    svg.rect(x, ly - 10, 12, 12, fill=col, fill_opacity=0.65)
    svg.text(x + 18, ly, lab, size=12)
svg.text(80, 386, "柱顶数字 = FA2 ÷（选择 + 压缩窗口）。FA2 每翻倍 ×3.9–4.0（平方），选择核每翻倍约 ×2（线性）。",
         size=12, color="muted")

svg.save("l14-nsa-kernel-time.svg")
