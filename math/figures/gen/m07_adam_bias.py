"""Figure M7.5: Adam's bias-correction factors and what skipping the correction would do.

Left:  1 − β^t for β = 0.9 (first moment), 0.95 and 0.999 (second moment), t = 1 .. 10^4 on a log axis.
Right: the uncorrected step m_t/√v_t divided by the corrected step m̂_t/√v̂_t, i.e. (1 − β1^t)/√(1 − β2^t),
       for β1 = 0.9 with β2 = 0.999 (Adam default) and β2 = 0.95 (GPT-3, DeepSeek). Values:
       β2 = 0.999: t=1 3.162, peak at t=12 (1/0.15224 = 6.569), t=1000 1.2576;  β2 = 0.95: t=1 0.4472.
Every point is computed here. Writes math/figures/m07-adam-bias.svg
"""

import numpy as np

from m07_svg import SVG, Axes, clipped_path

t = np.unique(np.round(np.logspace(0, 4, 400)).astype(int))
b1 = 0.9
W, H = 740, 372
svg = SVG(W, H, "Adam 的偏差修正因子，以及不修正时步长被放大或缩小的倍数")

# ---- left
ax = Axes(62, 44, 340, 290, (1, 1e4), (0, 1.0), xlog=True)
svg.text((ax.x0 + ax.x1) / 2, 22, "修正因子 1 − β^{t}", size=14, anchor="middle", weight="600")
ax.ygrid(svg, [0, 0.25, 0.5, 0.75, 1.0])
ax.xgrid(svg, [1, 10, 100, 1000, 10000], labels=["1", "10", "100", "1000", "10^{4}"])
ax.frame(svg)
svg.text((ax.x0 + ax.x1) / 2, ax.y1 + 36, "步数 t（对数刻度）", size=12, anchor="middle", color="muted")
box = (ax.x0, ax.y0, ax.x1, ax.y1)
curves = [(0.9, "blue", None, "β_{1} = 0.9"), (0.95, "violet", "6 4", "β_{2} = 0.95"),
          (0.999, "violet", None, "β_{2} = 0.999")]
for b, col, dash, lab in curves:
    y = 1 - b ** t
    svg.path(clipped_path([ax(a, v) for a, v in zip(t, y)], box), stroke=col, width=2.3, dash=dash)
# β1 label in the empty upper-left corner (the blue curve is still low there), clear of the y axis
svg.text(ax.x0 + 6, ax.Y(0.92), "β_{1} = 0.9", size=12, color="blue", weight="600")
svg.text(ax.X(45), ax.Y(0.78), "β_{2} = 0.95", size=12, color="violet", weight="600")
svg.text(ax.X(1200), ax.Y(0.33), "β_{2} = 0.999", size=12, color="violet", weight="600")
# marks
for tt, b, col in ((1000, 0.999, "violet"), (10, 0.9, "blue")):
    v = 1 - b ** tt
    svg.circle(ax.X(tt), ax.Y(v), 3.5, fill=col)
svg.text(ax.X(1000) - 8, ax.Y(1 - 0.999 ** 1000) - 8, "t = 1000：0.632", size=11, color="violet", anchor="end")
# two short lines above-left of the point: the rising blue curve is lower to the left of t = 10
svg.text(ax.X(10) - 6, ax.Y(1 - 0.9 ** 10) - 20, "t = 10", size=11, color="blue", anchor="end")
svg.text(ax.X(10) - 6, ax.Y(1 - 0.9 ** 10) - 7, "0.651", size=11, color="blue", anchor="end")

# ---- right
ax2 = Axes(440, 44, 720, 290, (1, 1e4), (0, 7), xlog=True)
svg.text((ax2.x0 + ax2.x1) / 2, 22, "不修正的步长 ÷ 修正后的步长", size=14, anchor="middle", weight="600")
ax2.ygrid(svg, [0, 1, 2, 3, 4, 5, 6, 7])
ax2.xgrid(svg, [1, 10, 100, 1000, 10000], labels=["1", "10", "100", "1000", "10^{4}"])
ax2.frame(svg)
svg.text((ax2.x0 + ax2.x1) / 2, ax2.y1 + 36, "步数 t（对数刻度）", size=12, anchor="middle", color="muted")
box2 = (ax2.x0, ax2.y0, ax2.x1, ax2.y1)
svg.line(ax2.x0, ax2.Y(1), ax2.x1, ax2.Y(1), color="green", width=1.4, dash="5 4")
# above the dashed line at the left, where the red curve is still high (≥ 3.16) and orange stays ≤ 1.1
svg.text(ax2.X(1) + 10, ax2.Y(1) - 9, "比值 = 1：修正与否无差别", size=11, color="green")
for b2, col in ((0.999, "red"), (0.95, "orange")):
    r = (1 - b1 ** t) / np.sqrt(1 - b2 ** t)
    svg.path(clipped_path([ax2(a, v) for a, v in zip(t, r)], box2), stroke=col, width=2.4)
tt = np.arange(1, 20001)
r = (1 - b1 ** tt) / np.sqrt(1 - 0.999 ** tt)
tp = int(tt[np.argmax(r)])
svg.circle(ax2.X(tp), ax2.Y(r.max()), 3.8, fill="red")
svg.text(ax2.X(tp) + 8, ax2.Y(r.max()) - 4, "t = %d：%.2f 倍" % (tp, r.max()), size=12, color="red", weight="600")
svg.circle(ax2.X(1), ax2.Y(3.1623), 3.5, fill="red")
svg.text(ax2.X(1) + 7, ax2.Y(3.1623) + 4, "3.16", size=11, color="red")
svg.text(ax2.X(200), ax2.Y(4.2), "β_{2} = 0.999", size=12, color="red", weight="600")
r95_1 = (1 - b1) / np.sqrt(1 - 0.95)
svg.circle(ax2.X(1), ax2.Y(r95_1), 3.5, fill="orange")
# under the orange curve, which has already risen above 0.8 to the right of t = 4
svg.text(ax2.X(4), ax2.Y(0.3), "β_{2} = 0.95：起点 0.447", size=11, color="orange")
svg.text(66, 364, "两图都取 β_{1} = 0.9。左图：滑动平均从 0 起步时“攒到”的总权重；右图：若省掉偏差修正，前若干步会走多大的步子。",
         size=11, color="muted")
svg.save("m07-adam-bias.svg")
print("peak", tp, r.max(), "t=1000", r[999])
