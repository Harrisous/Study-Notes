"""Figure M10.4: arithmetic intensity of Y = X W as the number of tokens m grows (log-log).

I(m, n, k) = 2 m n k / (s (m k + k n + m n)), s = 2 bytes (BF16), counting one read of X and W
and one write of Y.  Two shapes built from DeepSeek-V3's released config (hidden size 7168,
expert intermediate size 2048): a square 7168 x 7168 weight (for clean arithmetic; not an actual
V3 matrix), and one expert's up-projection 7168 -> 2048.
Horizontal lines: ridge points F/BW of A100 80GB SXM (312 / 2.039 = 153) and
H800 SXM5 (990 / 3.35 = 295.5, FlashMLA blog; H100 SXM gives 295.2).
Writes math/figures/m10-intensity.svg
"""

import math

import numpy as np

from m10_svg import SVG

S = 2


def I(m, n, k):
    return 2 * m * n * k / (S * (m * k + k * n + m * n))


shapes = [("7168 × 7168 投影", 7168, 7168, "ink"), ("专家 7168 → 2048", 2048, 7168, "blue")]
R_A100 = 312 / 2.039
R_H800 = 990 / 3.35

W, H = 720, 430
X0, X1, Y0, Y1 = 84, 560, 30, 360
LX0, LX1 = 0, math.log10(16384)
LY0, LY1 = -0.5, 4.0


def px(m):
    return X0 + (math.log10(m) - LX0) / (LX1 - LX0) * (X1 - X0)


def py(v):
    return Y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "矩阵乘法 Y = XW 的算术强度随 token 数 m 的变化（BF16，双对数坐标）")

for e in range(0, 5):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "10^{%d}" % e if e > 1 else ("1" if e == 0 else "10"), size=11, anchor="end", color="muted")
for m, lab in ((1, "1"), (4, "4"), (16, "16"), (64, "64"), (256, "256"), (1024, "1K"), (4096, "4K"), (16384, "16K")):
    x = px(m)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, lab, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "一次乘法里的 token 数 m（对数刻度，1K = 1024）", size=12, anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "算术强度 I（FLOP/字节，对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

# ridge lines
for R, lab, col, dash in ((R_A100, "A100 脊点 153", "muted", "6 4"), (R_H800, "H800 / H100 脊点 295", "red", "6 4")):
    y = py(R)
    svg.line(X0, y, X1, y, color=col, width=1.4, dash=dash)
    svg.text(X1 + 6, y + 4, lab, size=12, color=col)

# asymptote I = m (small m) and plateaus
ms = np.logspace(0, LX1, 300)
svg.path("M" + " L".join("%.1f %.1f" % (px(m), py(m)) for m in ms if m <= 700), stroke="muted", width=1, dash="2 3")
svg.text(px(40) - 4, py(40) - 12, "m 小时两条线都贴着 I = m", size=12, anchor="end", color="muted")
for name, n, k, col in shapes:
    plateau = 2 / S / (1 / n + 1 / k)
    svg.line(px(7000), py(plateau), X1, py(plateau), color=col, width=1, dash="4 3")
    ys = [I(m, n, k) for m in ms]
    svg.path("M" + " L".join("%.1f %.1f" % (px(m), py(v)) for m, v in zip(ms, ys)), stroke=col, width=2.6)
    svg.text(X1 + 6, py(plateau) + 4, "上限 %.0f" % plateau, size=11, color=col)

# labelled points
pts = [(1, 7168, 7168, "ink", "m = 1（decode）：I = 1.0", 10, 26),
       (256, 2048, 7168, "blue", "m = 256：%.0f" % I(256, 2048, 7168), 12, 34),
       (4096, 7168, 7168, "ink", "m = 4096：%.0f" % I(4096, 7168, 7168), -12, -14),
       (4096, 2048, 7168, "blue", "%.0f" % I(4096, 2048, 7168), 10, 18)]
for m, n, k, col, lab, dx, dy in pts:
    x, y = px(m), py(I(m, n, k))
    svg.circle(x, y, 4.2, fill=col)
    svg.text(x + dx, y + dy, lab, size=12, color=col, anchor="end" if dx < 0 else "start", weight="600")

# crossing points with the H800 ridge
cross = []
for name, n, k, col in shapes:
    lo, hi = 1, 20000
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if I(mid, n, k) >= R_H800:
            hi = mid
        else:
            lo = mid
    svg.circle(px(hi), py(R_H800), 3.2, fill="surface", stroke=col, width=1.6)
    cross.append(hi)
    print(name, "crosses H800 ridge at m =", hi)
svg.text(px(2600), py(R_A100) + 24, "空心圆：越过 295 的位置，", size=11, color="muted")
svg.text(px(2600), py(R_A100) + 39, "m = %d（黑）、%d（蓝）" % tuple(cross), size=11, color="muted")

# legend
svg.line(X0 + 16, Y0 + 14, X0 + 40, Y0 + 14, color="ink", width=2.6)
svg.text(X0 + 46, Y0 + 18, "7168 × 7168 的投影矩阵", size=12)
svg.line(X0 + 16, Y0 + 34, X0 + 40, Y0 + 34, color="blue", width=2.6)
svg.text(X0 + 46, Y0 + 38, "一个专家的上投影 7168 → 2048", size=12, color="blue")

svg.save("m10-intensity.svg")
for name, n, k, col in shapes:
    print(name, [round(I(m, n, k), 2) for m in (1, 64, 256, 4096)], "plateau", round(2 / S / (1 / n + 1 / k), 1))
