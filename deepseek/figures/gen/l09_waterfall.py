"""Figure 9.14: where DeepSeek-V2's "KV cache reduced by 93.3%" comes from (step chart).

Headline (arXiv:2405.04434v5 abstract, Fig. 1(b)): 389.1 KB (DeepSeek 67B) -> 25.9 KB (V2).
  The two KB values are bar lengths reverse-computed from the arXiv HTML vector figure
  (lesson-09-doc sec. 3.5), not numbers printed in the paper's text.
Decomposition (lesson-09-doc sec. 4.2; the 6-bit reading is an inference from the exact match):
  389,120 B = 2*8*128 * 95 layers * 2 B    (67B, GQA-8, bf16)
  109,440 B = 576 * 95 * 2                 (MLA per layer, same 95 layers)      x 576/2048
   69,120 B = 576 * 60 * 2                 (V2's 60 layers)                     x 60/95
   25,920 B = 576 * 60 * 6/8               (6-bit KV quantisation, deployment)  x 6/16
Log shares: ln(389120/69120)/ln(389120/25920) = 0.638.
Writes deepseek/figures/l09-waterfall.svg
"""

import math

from l09_svg import SVG, LinAxis

steps = [
    ("DeepSeek 67B", "GQA-8 · 95 层 · bf16", 2 * 8 * 128 * 95 * 2, "muted"),
    ("换成 MLA", "576 / 2048 = ×0.281", 576 * 95 * 2, "orange"),
    ("层数 95 → 60", "60 / 95 = ×0.632", 576 * 60 * 2, "orange"),
    ("KV 量化到 6 bit", "6 / 16 = ×0.375", 576 * 60 * 6 // 8, "orange"),
]
assert [s[2] for s in steps] == [389_120, 109_440, 69_120, 25_920]
share = math.log(389120 / 69120) / math.log(389120 / 25920)
print(1 - 69120 / 389120, 1 - 25920 / 389120, share)

W, H = 720, 380
svg = SVG(W, H, "“KV cache 降低 93.3%”的分解")
X0, Y0, Y1 = 70, 34, 300
ys = LinAxis(0, 400_000, Y1, Y0)
for v in range(0, 400_001, 100_000):
    svg.line(X0, ys(v), 690, ys(v), color="grid", width=1)
    svg.text(X0 - 6, ys(v) + 4, "%d" % (v // 1000), size=11, anchor="end", color="muted")
svg.text(20, (Y0 + Y1) / 2, "每 token 的 KV cache（KB）", size=12, anchor="middle", color="muted", rotate=-90)
BW, GAP = 86, 50
prev = None
for i, (name, sub, b, col) in enumerate(steps):
    x = X0 + 30 + i * (BW + GAP)
    if prev is None:
        svg.rect(x, ys(b), BW, Y1 - ys(b), fill=col, fill_opacity=0.5, rx=2)
    else:
        # removed part (ghost) + remaining bar
        svg.rect(x, ys(prev), BW, ys(b) - ys(prev), fill="red", fill_opacity=0.07, stroke="red", width=1,
                 dash="4 3", rx=2)
        fill_op = 0.55 if i < 3 else 0.35
        svg.rect(x, ys(b), BW, Y1 - ys(b), fill=col, fill_opacity=fill_op, rx=2)
        svg.line(x - GAP, ys(prev), x, ys(prev), color="rule", width=1, dash="3 3")
        if i < 3:
            svg.text(x + BW / 2, ys(prev) - 8, sub, size=11.5, anchor="middle", color="red")
        else:
            svg.text(x + BW + 6, (ys(prev) + ys(b)) / 2 + 4, sub, size=11.5, color="red")
    if i < 3:
        svg.text(x + BW / 2, ys(b) - 8 if prev is None else ys(b) + 16, "%.2f KB" % (b / 1000), size=12.5,
                 anchor="middle", weight="700", color="ink")
    else:
        svg.text(x + BW + 6, ys(b) + 14, "%.2f KB" % (b / 1000), size=12.5, weight="700", color="ink")
    svg.text(x + BW / 2, Y1 + 18, name, size=12.5, anchor="middle", weight="600",
             color="ink" if col == "muted" else "orange")
    if prev is None:
        svg.text(x + BW / 2, Y1 + 34, sub, size=11, anchor="middle", color="muted")
    prev = b
svg.line(X0, Y1, 690, Y1, color="rule", width=1.2)
# brackets
xa = X0 + 30 + 2 * (BW + GAP) + BW + 30
svg.text(xa - 60, 92, "到这一步（纯 MLA + 层数，bf16）：", size=12, color="ink")
svg.text(xa - 60, 110, "降低 %.1f%%" % (100 * (1 - 69120 / 389120)), size=13, color="orange", weight="700")
svg.text(xa - 60, 150, "再加 6 bit 量化：降低 %.1f%%" % (100 * (1 - 25920 / 389120)), size=13, color="orange",
         weight="700")
svg.text(xa - 60, 168, "= 论文图 1(b) 两根柱子的长度", size=12, color="muted")
svg.text(xa - 60, 185, "389.1 → 25.9 KB（按矢量图坐标反算）", size=12, color="muted")
svg.text(X0, Y1 + 62, "按对数份额：MLA 与层数贡献 %.0f%% 的降幅，量化贡献 %.0f%%。" % (100 * share, 100 * (1 - share)),
         size=12.5, color="ink")
svg.save("l09-waterfall.svg")
