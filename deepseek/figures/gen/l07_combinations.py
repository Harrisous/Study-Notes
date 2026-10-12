"""Figure 7.10: number of expert combinations C(16m, 2m) versus segmentation granularity m (log axis).

Points: log10 C(16m, 2m), m = 1..8 (math.comb). Dashed: entropy upper bound
C(n, k) <= 2^{n H(k/n)} with n = 16m, k = 2m, H(1/8) = 0.5436 bits, i.e. 2.618 decades per unit m.
Dotted: lower bound 2^{n H} / (n + 1). Right-hand markers: routed-expert combinations of real
configurations (only the unordered set of routed experts is counted).
Writes deepseek/figures/l07-combinations.svg
"""

import math
from math import comb

from l07_svg import SVG, comma

N, K = 16, 2
ms = list(range(1, 9))
vals = [comb(N * m, K * m) for m in ms]
logs = [math.log10(v) for v in vals]
p = K / N
Hbits = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
up = [N * m * Hbits * math.log10(2) for m in ms]
lo = [u - math.log10(N * m + 1) for m, u in zip(ms, up)]
assert all(l <= v <= u + 1e-9 for l, v, u in zip(lo, logs, up))

refs = [
    ("Mixtral：C(8, 2)", comb(8, 2)),
    ("GShard 2048 专家 top-2：C(2048, 2)", comb(2048, 2)),
    ("DeepSeekMoE 16B：C(64, 6)", comb(64, 6)),
    ("DeepSeekMoE 2B：C(63, 7)", comb(63, 7)),
]

W, H = 740, 430
X0, X1, Y0, Y1 = 70, 470, 34, 360
LY1 = 22


def px(m):
    return X0 + (m - 0.5) / 8.0 * (X1 - X0)


def py(lv):
    return Y1 - lv / LY1 * (Y1 - Y0)


svg = SVG(W, H, "专家组合数随切分粒度 m 爆炸式增长（对数纵轴）")
for e in range(0, LY1 + 1, 2):
    y = py(e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
for m in ms:
    svg.text(px(m), Y1 + 18, str(m), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "切分粒度 m（16 个专家各切成 m 份，激活 2m 个）", size=12, anchor="middle", color="muted")
svg.text(18, (Y0 + Y1) / 2, "组合数 C(16m, 2m)（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

svg.path("M" + " L".join("%.1f %.1f" % (px(m), py(u)) for m, u in zip(ms, up)), stroke="violet", width=1.4,
         dash="6 4")
svg.path("M" + " L".join("%.1f %.1f" % (px(m), py(l)) for m, l in zip(ms, lo)), stroke="violet", width=1.2,
         dash="2 3")
svg.path("M" + " L".join("%.1f %.1f" % (px(m), py(v)) for m, v in zip(ms, logs)), stroke="blue", width=2.2)
for m, v, lv in zip(ms, vals, logs):
    svg.circle(px(m), py(lv), 4.2, fill="blue")
svg.text(px(1) + 8, py(logs[0]) + 16, "m = 1：120", size=12, color="blue", weight="600")
svg.text(px(4) + 10, py(logs[3]) + 30, "m = 4：%s" % comma(vals[3]), size=12, color="blue", weight="600")
svg.text(px(4) + 10, py(logs[3]) + 46, "（论文的例子：C(64, 8)）", size=11, color="muted")
svg.text(px(8) + 10, py(logs[7]) + 4, "约 9.3×10^{19}", size=12, color="blue")
svg.text(px(6) - 40, py(up[5]) - 10, "上界 2^{16m·H(1/8)}", size=12, anchor="end", color="violet")
svg.text(px(6) + 10, py(lo[5]) + 30, "下界 2^{16m·H(1/8)}/(16m+1)", size=12, color="violet")

# right-hand markers for real configurations (labels spread apart, leader lines to the true height)
RX = 500
svg.text(RX - 8, Y0 - 12, "真实配置的路由组合数", size=12, color="ink", weight="600")
items = sorted(((math.log10(v), name, v) for name, v in refs), reverse=True)
last = -1e9
for lv, name, v in items:
    y_true = py(lv)
    y_lab = max(y_true, last + 34)
    last = y_lab
    svg.line(X1 - 8, y_true, X1, y_true, color="ink", width=1.2)
    svg.path("M%.1f %.1f L%.1f %.1f L%.1f %.1f" % (X1, y_true, RX - 16, y_lab - 6, RX - 8, y_lab - 6), stroke="muted",
             width=0.9)
    svg.text(RX, y_lab - 2, name, size=11, color="ink")
    svg.text(RX, y_lab + 13, "= %s" % comma(v), size=11, color="muted")

svg.text(X0, H - 8, "每增加 1，纵轴约上升 %.2f 个数量级（上界的斜率：16·H(1/8)·log_{10}2，H 为二元熵）。" % (N * Hbits * math.log10(2)),
         size=12, color="muted")

svg.save("l07-combinations.svg")
print("H", Hbits, "decades/m", N * Hbits * math.log10(2), [round(x, 3) for x in logs])
