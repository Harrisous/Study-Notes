"""Figure M6.2: error of forward and centred finite differences versus the step h (log-log).

Redraw of Baydin et al. (2018), Figure 3, recomputed here in float64: the truncated logistic map
f(x) = 64x(1-x)(1-2x)^2(1-8x+8x^2)^2 at x0 = 0.2; the exact derivative comes from the polynomial
coefficients, so the plotted error is |difference quotient - f'(x0)|.
Writes math/figures/m06-finite-diff.svg
"""

import math

import numpy as np

from m06_svg import SVG, sci

P = np.polynomial.Polynomial
x = P([0.0, 1.0])
f = 64 * x * (1 - x) * (1 - 2 * x) ** 2 * (1 - 8 * x + 8 * x ** 2) ** 2
df = f.deriv()
x0 = 0.2
exact = float(df(x0))

hs = np.logspace(-16, 0, 321)
fwd = np.array([abs((f(x0 + h) - f(x0)) / h - exact) for h in hs])
ctr = np.array([abs((f(x0 + h) - f(x0 - h)) / (2 * h) - exact) for h in hs])

W, H = 720, 420
X0, X1, Y0, Y1 = 84, 560, 40, 350
LX0, LX1 = -16, 0
LY0, LY1 = -12, 2
FLOOR = 10.0 ** LY0


def px(h):
    return X0 + (math.log10(h) - LX0) / (LX1 - LX0) * (X1 - X0)


def py(e):
    e = max(e, FLOOR)
    return Y1 - (math.log10(e) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "前向差分与中心差分的误差随步长 h 变化（双对数坐标）")
for e in range(LY0, LY1 + 1, 2):
    y = py(10.0 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
for e in range(LX0, LX1 + 1, 2):
    xx = px(10.0 ** e)
    svg.line(xx, Y0, xx, Y1, color="grid", width=1)
    svg.text(xx, Y1 + 18, "10^{%d}" % e, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 42, "步长 h（对数刻度）", size=12, anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "|差分近似 − 真导数|（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)


def poly(xs, ys):
    pts = []
    for a, b in zip(xs, ys):
        if b <= 0:
            b = FLOOR
        if b > 10.0 ** LY1:
            continue
        pts.append("%.1f %.1f" % (px(a), py(b)))
    return "M" + " L".join(pts)


svg.path(poly(hs, fwd), stroke="blue", width=1.8, join="round")
svg.path(poly(hs, ctr), stroke="violet", width=1.8, join="round")

# minima of the running median of log10(error) over 21 neighbouring h values, so that a single
# lucky cancellation (a spike downwards) is not reported as "the best h"
def robust_min(arr):
    lg = np.log10(np.maximum(arr, FLOOR))
    k = 10
    med = np.array([np.median(lg[max(0, i - k):i + k + 1]) for i in range(len(lg))])
    i = int(np.argmin(med))
    return i, 10.0 ** med[i]


i_f, e_f = robust_min(fwd)
i_c, e_c = robust_min(ctr)
for i, e, col in ((i_f, e_f, "blue"), (i_c, e_c, "violet")):
    svg.circle(px(hs[i]), py(e), 4.5, fill=col)
print("fwd best", hs[i_f], e_f, "ctr best", hs[i_c], e_c)

# slope guides (computed from the error model, not eyeballed)
# truncation: forward ~ |f''|/2 * h, centred ~ |f'''|/6 * h^2 ; round-off ~ eps*|f|/h
f2 = abs(float(f.deriv(2)(x0)))
f3 = abs(float(f.deriv(3)(x0)))
eps = np.finfo(float).eps
print("f''", f2, "f'''", f3)

# annotations
# sits above the forward-difference line (at h = 3e-5 the error is ~1e-3), so the line does not cross it
svg.text(px(3e-5), py(1.5e-2), "截断误差（前向）∝ h", size=12, anchor="end", color="blue")
svg.text(px(1e-4), py(1e-8), "截断误差（中心）", size=12, color="violet")
svg.text(px(1e-4), py(1e-8) + 17, "∝ h²", size=12, color="violet")
svg.text(px(1e-16) + 8, py(1e-6), "舍入误差 ∝ 1/h", size=12, color="ink")
svg.text(px(1e-16) + 8, py(1e-6) + 17, "（h 越小反而越大）", size=12, color="muted")

# right-hand panel with the numbers
RX = X1 + 26
svg.text(RX, Y0 + 8, "x₀ = 0.2，真导数", size=12, color="muted")
svg.text(RX, Y0 + 26, "f′(x₀) = %.4f" % exact, size=12, color="ink")
svg.line(RX, Y0 + 52, RX + 24, Y0 + 52, color="blue", width=2.2)
svg.text(RX + 30, Y0 + 56, "前向差分", size=13, color="blue", weight="600")
svg.text(RX, Y0 + 76, "(f(x+h) − f(x)) / h", size=12, color="blue")
svg.text(RX, Y0 + 96, "最好约 %s" % sci(e_f, 1), size=12, color="blue")
svg.text(RX, Y0 + 114, "在 h ≈ %s" % sci(hs[i_f], 1), size=12, color="blue")
svg.line(RX, Y0 + 146, RX + 24, Y0 + 146, color="violet", width=2.2)
svg.text(RX + 30, Y0 + 150, "中心差分", size=13, color="violet", weight="600")
svg.text(RX, Y0 + 170, "(f(x+h) − f(x−h)) / 2h", size=12, color="violet")
svg.text(RX, Y0 + 190, "最好约 %s" % sci(e_c, 1), size=12, color="violet")
svg.text(RX, Y0 + 208, "在 h ≈ %s" % sci(hs[i_c], 1), size=12, color="violet")
svg.text(RX, Y0 + 244, "float64 计算；圆点取", size=12, color="muted")
svg.text(RX, Y0 + 262, "21 个相邻 h 的中位数", size=12, color="muted")
svg.text(RX, Y0 + 280, "最低处，不取偶然的", size=12, color="muted")
svg.text(RX, Y0 + 298, "向下尖刺", size=12, color="muted")

svg.save("m06-finite-diff.svg")
