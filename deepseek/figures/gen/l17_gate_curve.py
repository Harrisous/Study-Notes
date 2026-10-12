"""Figure 17.4: the Engram gate with and without the signed square root.

Paper eq. (4): alpha = sigmoid(s).  Released code (gh-engram demo, V4.1 inference code):
alpha = sigmoid(sign(s) * sqrt(max(|s|, 1e-6))).  s = RMSNorm(h)^T RMSNorm(k) / sqrt(d).
Marked points and ranges are the ones in the notes §4.4 (learnable RMSNorm scales ignored):
 s = -4, -1, -0.25, 0, 0.25, 1, 4, 9, 16;  |s| <= sqrt(d): 50.60 (d=2560), 71.55 (d=5120);
 after the square root 7.11 / 8.46;  alpha in [0.00081, 0.99919] / [0.00021, 0.99979].
Writes deepseek/figures/l17-gate-curve.svg
"""

import math

from l17_svg import SVG
from l17_num import sigmoid, ssqrt

PTS = [-4, -1, -0.25, 0, 0.25, 1, 4, 9, 16]
EXPECT = {-4: (0.0180, 0.1192), -1: (0.2689, 0.2689), -0.25: (0.4378, 0.3775), 0: (0.5, 0.5),
          0.25: (0.5622, 0.6225), 1: (0.7311, 0.7311), 4: (0.9820, 0.8808), 9: (0.9999, 0.9526),
          16: (1.0000, 0.9820)}
for s, (a, b) in EXPECT.items():
    assert round(sigmoid(s), 4) == a and round(sigmoid(ssqrt(s)), 4) == b, s
RANGES = []
for d in (2560, 5120):
    smax = math.sqrt(d)
    r = smax ** 0.5
    RANGES.append((d, smax, r, sigmoid(-r), sigmoid(r)))
assert [round(x[1], 2) for x in RANGES] == [50.6, 71.55]
assert [round(x[2], 2) for x in RANGES] == [7.11, 8.46]
assert round(RANGES[0][3], 5) == 0.00081 and round(RANGES[1][4], 5) == 0.99979

W, H = 760, 372
svg = SVG(W, H, "门控曲线：论文的 σ(s) 与代码的 σ(sign(s)·√|s|)")

X0, X1, Y0, Y1 = 64, 470, 30, 300
S0, S1 = -16, 16


def x(s):
    return X0 + (s - S0) / (S1 - S0) * (X1 - X0)


def y(p):
    return Y1 - p * (Y1 - Y0)


for p in (0, 0.25, 0.5, 0.75, 1):
    svg.line(X0, y(p), X1, y(p), color="grid", width=1)
    svg.text(X0 - 8, y(p) + 4, "%g" % p, size=11, anchor="end", color="muted")
for s in range(S0, S1 + 1, 4):
    svg.line(x(s), Y0, x(s), Y1, color="grid", width=1)
    svg.text(x(s), Y1 + 17, str(s).replace("-", "−"), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "缩放点积 s_{t}", size=12, anchor="middle", color="muted")
svg.text(X0 - 44, (Y0 + Y1) / 2, "门控 α_{t}", size=12, anchor="middle", color="muted", rotate=-90)

# |s| = 1 crossover band
svg.rect(x(-1), Y0, x(1) - x(-1), Y1 - Y0, fill="violet", fill_opacity=0.06)
svg.text(x(0), Y0 + 14, "|s| < 1", size=11, anchor="middle", color="violet")

N = 640
ss = [S0 + (S1 - S0) * i / N for i in range(N + 1)]
svg.path("M" + " L".join("%.1f %.1f" % (x(s), y(sigmoid(s))) for s in ss), stroke="ink", width=2.2)
svg.path("M" + " L".join("%.1f %.1f" % (x(s), y(sigmoid(ssqrt(s)))) for s in ss), stroke="violet", width=2.4)

for s in PTS:
    a, b = sigmoid(s), sigmoid(ssqrt(s))
    svg.circle(x(s), y(a), 3.4, fill="ink", stroke="surface", width=1)
    svg.circle(x(s), y(b), 3.6, fill="violet", stroke="surface", width=1)
svg.text(x(4) + 6, y(sigmoid(4)) - 8, "0.982", size=11, color="ink")
svg.text(x(4) + 6, y(sigmoid(2)) + 15, "0.881", size=11, color="violet")
svg.text(x(16) - 4, y(sigmoid(4)) + 16, "0.982", size=11, anchor="end", color="violet")
svg.text(x(-4) + 6, y(sigmoid(-4)) + 2, "0.018", size=11, color="ink")
svg.text(x(-4) - 6, y(sigmoid(-2)) - 6, "0.119", size=11, anchor="end", color="violet")

svg.text(x(-15.5), y(0.86), "论文式 (4)：σ(s)", size=12.5, color="ink", weight="700")
svg.text(x(-15.5), y(0.86) + 18, "代码：σ(sign(s)·√|s|)", size=12.5, color="violet", weight="700")

# ---------------------------------------------------------------- range ladder (right)
RX = 504
svg.text(RX, 38, "RMSNorm 之后 s 能取到多大", size=13, weight="700")
svg.text(RX, 58, "（忽略可学习缩放）", size=11.5, color="muted")
rows = [("", "Engram-27B", "V4.1-Flash"),
        ("隐藏维 d", "2560", "5120"),
        ("|s| ≤ √d", "%.2f" % RANGES[0][1], "%.2f" % RANGES[1][1]),
        ("√|s| ≤ d^{1/4}", "%.2f" % RANGES[0][2], "%.2f" % RANGES[1][2]),
        ("代码 α 最小", "%.5f" % RANGES[0][3], "%.5f" % RANGES[1][3]),
        ("代码 α 最大", "%.5f" % RANGES[0][4], "%.5f" % RANGES[1][4])]
for i, (k, a, b) in enumerate(rows):
    yy = 92 + i * 26
    if i == 0:
        svg.text(RX + 150, yy, a, size=12, anchor="middle", color="muted", weight="700")
        svg.text(RX + 222, yy, b, size=12, anchor="middle", color="muted", weight="700")
        svg.line(RX, yy + 8, RX + 250, yy + 8, color="rule", width=1)
        continue
    svg.text(RX, yy, k, size=12)
    svg.text(RX + 150, yy, a, size=12, anchor="middle")
    svg.text(RX + 222, yy, b, size=12, anchor="middle")
svg.text(RX, 262, "平方根在 |s| < 1 处放大，|s| > 1 处压缩，", size=12, color="violet")
svg.text(RX, 280, "交点在 |s| = 1。s = 4 时门控为 0.881", size=12, color="violet")
svg.text(RX, 298, "而非 0.982：大点积时更晚饱和（笔者解读）。", size=12, color="violet")
svg.text(RX, 330, "平方根一步只见于开源代码与 V4.1 推理代码", size=11.5, color="muted")
svg.text(RX, 347, "（注释：与训练内核一致），论文式 (4) 没有。", size=11.5, color="muted")

svg.save("l17-gate-curve.svg")
