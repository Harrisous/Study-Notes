"""Figure 10.5: how many devices does a top-6 choice touch? (160 experts = 8 devices x 20, K = 6)

Bars: share of all C(160, 6) = 21,193,254,160 expert sets that touch exactly m devices
(equivalently, the probability under a uniformly random choice of 6 experts), by inclusion-exclusion.
Line: cumulative share touching at most m devices; at m = 3 it is 2,270,854,160 / 21,193,254,160 = 10.7 %.
Expected number of devices touched: 8 (1 - C(140, 6) / C(160, 6)) = 4.46.
Writes deepseek/figures/l10-device-count.svg
"""

from math import comb

from l10_svg import SVG

D, PER, K = 8, 20, 6
N = D * PER
total = comb(N, K)


def exactly(m):
    return comb(D, m) * sum((-1) ** j * comb(m, j) * comb(PER * (m - j), K) for j in range(m + 1))


ex = {m: exactly(m) for m in range(1, K + 1)}
assert sum(ex.values()) == total == 21193254160
cum = {m: sum(ex[k] for k in range(1, m + 1)) for m in ex}
assert cum[3] == 2270854160
E = D * (1 - comb(N - PER, K) / total)
assert abs(E - 4.4586) < 1e-3

W, H = 720, 390
X0, X1, Y0, Y1 = 80, 640, 40, 300
svg = SVG(W, H, "从 160 个专家（8 台 × 20）里选 6 个，会触及几台设备")


def py(p):
    return Y1 - p * (Y1 - Y0)


for k in range(0, 11):
    y = py(k / 10)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "%d%%" % (k * 10), size=11, anchor="end", color="muted")
slot = (X1 - X0) / 6
# allowed region m <= 3
svg.rect(X0, Y0, 3 * slot, Y1 - Y0, fill="blue", fill_opacity=0.05)
svg.line(X0 + 3 * slot, Y0 - 8, X0 + 3 * slot, Y1, color="blue", width=1.4, dash="5 4")
svg.text(X0 + 3 * slot - 6, Y0 + 2, "M = 3：只允许这一侧", size=12, anchor="end", color="blue", weight="600")
pts = []
for m in range(1, 7):
    cx = X0 + (m - 0.5) * slot
    p = ex[m] / total
    bw = slot * 0.56
    svg.rect(cx - bw / 2, py(p), bw, Y1 - py(p), fill="blue" if m <= 3 else "muted",
             fill_opacity=0.75 if m <= 3 else 0.45)
    lab = ("%.1f%%" % (100 * p)) if p >= 0.001 else ("%.4f%%" % (100 * p))
    if p > 0.2:
        svg.text(cx, py(p) + 18, lab, size=12, anchor="middle", color="surface", weight="600")
    elif p > 0.05:
        svg.text(cx, Y1 - 8, lab, size=12, anchor="middle", color="surface", weight="600")
    else:
        svg.text(cx, py(p) - 8, lab, size=12, anchor="middle", color="ink")
    svg.text(cx, Y1 + 18, "%d 台" % m, size=12, anchor="middle", color="ink")
    pts.append((cx, py(cum[m] / total)))
svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke="violet", width=2.2)
for m, (x, y) in zip(range(1, 7), pts):
    svg.circle(x, y, 3.6, fill="violet")
# callout for m = 3
x3, y3 = pts[2]
svg.line(x3 - 4, y3 - 4, 250, 150, color="violet", width=1)
svg.text(246, 144, "至多 3 台：10.7%", size=13, anchor="end", color="violet", weight="600")
svg.text(246, 126, "设备受限路由允许的组合", size=12, anchor="end", color="violet")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "一个 token 的 6 个专家分布在几台设备上", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "占全部 6 专家组合的比例", size=12, anchor="middle", color="muted", rotate=-90)
# legend
svg.rect(X1 - 250, Y1 + 52, 14, 10, fill="blue", fill_opacity=0.75)
svg.text(X1 - 230, Y1 + 62, "恰好 m 台（柱）", size=12)
svg.line(X1 - 120, Y1 + 57, X1 - 100, Y1 + 57, color="violet", width=2.2)
svg.text(X1 - 94, Y1 + 62, "至多 m 台（线）", size=12)
svg.text(X0, Y1 + 62, "均匀随机时平均触及 %.2f 台" % E, size=12, color="ink", weight="600")
svg.save("l10-device-count.svg")
