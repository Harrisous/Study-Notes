"""Figure 19.3: how much of a full 40-layer prefill CED actually does, as a function of prompt length T.

Work is counted in layer-tokens (one token through one layer), the paper's O(NL) accounting:
  full model                      40 T
  CED + Decoder SWA Bounded Replay 20 T + 20 min(T, 128)          -> 1/2 + 64/T
  CED with an exact decoder SWA    20 T + 20 min(T, 20 x 128)     -> 1/2 + 1280/T
Curves come from l19_num.prefill_ratio_bounded / prefill_ratio_exact.
Writes deepseek/figures/l19-prefill-ratio.svg
"""

import math

from l19_svg import SVG
from l19_num import prefill_ratio_bounded, prefill_ratio_exact

W, H = 720, 350
svg = SVG(W, H, "CED 的 prefill 工作量占完整 40 层的比例，随提示长度变化")

PX0, PX1, PY0, PY1 = 78, 690, 40, 300      # plot box
T_MIN, T_MAX = 2 ** 7, 2 ** 20
Y_MIN, Y_MAX = 0.4, 1.05


def xs(T):
    return PX0 + (math.log2(T) - math.log2(T_MIN)) / (math.log2(T_MAX) - math.log2(T_MIN)) * (PX1 - PX0)


def ys(v):
    return PY1 - (v - Y_MIN) / (Y_MAX - Y_MIN) * (PY1 - PY0)


# grid and axes
for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    svg.line(PX0, ys(v), PX1, ys(v), color="grid", width=1)
    svg.text(PX0 - 8, ys(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
ticks = [(2 ** 7, "128"), (2 ** 10, "1K"), (2 ** 12, "4K"), (2 ** 14, "16K"), (2 ** 16, "64K"),
         (2 ** 17, "128K"), (2 ** 18, "256K"), (2 ** 20, "1M")]
for T, lab in ticks:
    svg.line(xs(T), PY0, xs(T), PY1, color="grid", width=1)
    svg.text(xs(T), PY1 + 17, lab, size=11, anchor="middle", color="muted")
svg.line(PX0, PY1, PX1, PY1, color="muted", width=1)
svg.line(PX0, PY0, PX0, PY1, color="muted", width=1)
svg.text((PX0 + PX1) / 2, PY1 + 38, "提示长度 T（token，对数刻度，底 2）", size=12, anchor="middle", color="muted")
svg.text(24, (PY0 + PY1) / 2, "占完整 prefill 的比例", size=12, anchor="middle", color="muted", rotate=-90)

# reference lines
svg.line(PX0, ys(1.0), PX1, ys(1.0), color="muted", width=1.6, dash="6 4")
svg.text(PX1 - 4, ys(1.0) - 7, "不用 CED：40 层全跑", size=12, anchor="end", color="muted")
svg.line(PX0, ys(0.5), PX1, ys(0.5), color="muted", width=1, dash="2 3")
svg.text(PX1 - 4, ys(0.5) + 16, "下限 1/2：只剩编码器", size=12, anchor="end", color="muted")

# curves: sample densely in log2 T
N = 400
pts_b, pts_e = [], []
for i in range(N + 1):
    T = 2 ** (math.log2(T_MIN) + i / N * (math.log2(T_MAX) - math.log2(T_MIN)))
    pts_b.append((xs(T), ys(prefill_ratio_bounded(T))))
    pts_e.append((xs(T), ys(prefill_ratio_exact(T))))


def poly(pts):
    return "M" + " L".join("%.1f %.1f" % p for p in pts)


svg.path(poly(pts_e), stroke="red", width=2, dash="7 4")
svg.path(poly(pts_b), stroke="blue", width=2.6)

# labelled points
for T, lab in ((4096, "4K"), (1024, "1K")):
    b, e = prefill_ratio_bounded(T), prefill_ratio_exact(T)
    svg.circle(xs(T), ys(b), 4, fill="blue")
    svg.circle(xs(T), ys(e), 4, fill="red")
svg.text(xs(4096) + 8, ys(prefill_ratio_bounded(4096)) - 8, "T = 4K：0.516", size=12, color="blue")
svg.text(xs(4096) + 8, ys(prefill_ratio_exact(4096)) - 6, "T = 4K：0.8125", size=12, color="red")
svg.text(xs(1024) - 8, ys(prefill_ratio_bounded(1024)) + 20, "T = 1K：0.5625", size=12, anchor="end", color="blue")

# legend
LX, LY = 436, 96
svg.line(LX, LY, LX + 30, LY, color="blue", width=2.6)
svg.text(LX + 38, LY + 4, "V4.1：解码器只回放最后 128 个", size=12)
svg.text(LX + 38, LY + 21, "（1/2 + 64/T）", size=12, color="muted")
svg.line(LX, LY + 44, LX + 30, LY + 44, color="red", width=2, dash="7 4")
svg.text(LX + 38, LY + 48, "若要精确重建解码器 SWA", size=12)
svg.text(LX + 38, LY + 65, "（回放 2,560 个；1/2 + 1280/T）", size=12, color="muted")

svg.save("l19-prefill-ratio.svg")
