"""Figure 19.14: 3,514 -> 1,630 -> 890 bytes/token, split into a structure step and a precision step.

Bars (bytes of global KV per token, stacked by component), all from l19_num:
  V4-Flash                 CSA main 21/4*584, HCA main 20/128*584, CSA indexer 21/4*68
  V4.1 structure, V4 fmt   2.5*584 main + 2.5*68 indexer = 1,630
  V4.1                     2.5*288 main + 2.5*68 indexer = 890
Writes deepseek/figures/l19-waterfall.svg
"""

import math

from l19_svg import SVG
from l19_num import MAIN_V4, MAIN_V41, IDX, G4F, G41, G41_STRUCT_V4PREC, N_CSA_4F, N_HCA_4F

W, H = 720, 384
svg = SVG(W, H, "从 3,514 到 890：结构省 2.16 倍，精度再省 1.83 倍")

PX0, PY0, PY1 = 90, 40, 300
YMAX = 3800


def ys(v):
    return PY1 - v / YMAX * (PY1 - PY0)


for v in range(0, 3801, 500):
    svg.line(PX0, ys(v), 700, ys(v), color="grid", width=1)
    svg.text(PX0 - 8, ys(v) + 4, "{:,}".format(v), size=11, anchor="end", color="muted")
svg.line(PX0, PY1, 700, PY1, color="muted", width=1)
svg.text(24, (PY0 + PY1) / 2, "每 token 全局 KV（字节）", size=12, anchor="middle", color="muted", rotate=-90)

bars = [
    ("V4-Flash", [(N_CSA_4F / 4 * MAIN_V4, "amber", 0.45, "CSA 主 KV"), (N_HCA_4F / 128 * MAIN_V4, "orange", 0.55, "HCA"),
                  (N_CSA_4F / 4 * IDX, "violet", 0.55, "索引键")]),
    ("V4.1 的结构\n+ V4 的格式", [(2.5 * MAIN_V4, "amber", 0.45, "主 KV（584 B/条）"), (2.5 * IDX, "violet", 0.55, "索引键")]),
    ("V4.1-Flash", [(2.5 * MAIN_V41, "orange", 0.75, "主 KV（FP4，288 B/条）"), (2.5 * IDX, "violet", 0.55, "索引键")]),
]
totals = [G4F, G41_STRUCT_V4PREC, G41]
BW = 150
xs = [130, 335, 540]
for (name, parts), tot, x in zip(bars, totals, xs):
    assert abs(sum(p[0] for p in parts) - tot) < 1e-6
    y = PY1
    for v, col, op, lab in parts:
        h = v / YMAX * (PY1 - PY0)
        svg.rect(x, y - h, BW, h, fill=col, fill_opacity=op, stroke=col, width=0.8)
        if h > 40:
            svg.text(x + BW / 2, y - h / 2 - 3, lab, size=11, anchor="middle")
            svg.text(x + BW / 2, y - h / 2 + 13, "{:,.2f}".format(v).rstrip("0").rstrip("."), size=11, anchor="middle")
        elif h > 16:
            svg.text(x + BW / 2, y - h / 2 + 4, "%s %s" % (lab, "{:,.2f}".format(v).rstrip("0").rstrip(".")), size=11,
                     anchor="middle")
        y -= h
    svg.text(x + BW / 2, y - 8, "{:,.2f}".format(tot).rstrip("0").rstrip("."), size=13, anchor="middle", weight="700")
    for k, line in enumerate(name.split("\n")):
        svg.text(x + BW / 2, PY1 + 18 + 15 * k, line, size=12, anchor="middle")
# HCA sliver label
svg.line(xs[0] + BW, ys(N_CSA_4F / 4 * MAIN_V4 + 45), xs[0] + BW + 14, ys(N_CSA_4F / 4 * MAIN_V4 + 45) - 1,
         color="orange", width=1)
svg.text(xs[0] + BW + 16, ys(N_CSA_4F / 4 * MAIN_V4 + 45) + 3, "HCA 91.25", size=11, color="orange")

# step arrows
f1, f2 = G4F / G41_STRUCT_V4PREC, G41_STRUCT_V4PREC / G41
share = math.log(f1) / math.log(G4F / G41)
svg.arrow(xs[0] + BW + 6, ys(2000), xs[1] - 6, ys(1760), color="ink", width=1.4)
# the gap between bars is narrower than the label: start it just right of bar 1, above bar 2's top
lx1 = xs[0] + BW + 10
svg.text(lx1, ys(2650), "结构 ÷%.2f" % f1, size=13, weight="700")
svg.text(lx1, ys(2650) + 16, "41 份 → 4 份", size=11, color="muted")
svg.arrow(xs[1] + BW + 6, ys(1400), xs[2] - 6, ys(1050), color="ink", width=1.4)
mx2 = (xs[1] + BW + xs[2]) / 2
svg.text(mx2, ys(2000), "精度 ÷%.2f" % f2, size=13, anchor="middle", weight="700")
svg.text(mx2, ys(2000) + 16, "584 B → 288 B", size=11, anchor="middle", color="muted")

svg.text(PX0, PY1 + 58, "合计 ÷%.2f（论文“约 1/4”）。按对数分摊：结构占 %.0f%%，精度占 %.0f%%。索引键两边都是 68 B/条，只随份数变化。"
         % (G4F / G41, share * 100, (1 - share) * 100), size=12, color="muted")

svg.save("l19-waterfall.svg")
