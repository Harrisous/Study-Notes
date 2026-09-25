"""Figure 16.10: how many layers to recompute at a time (bill figure, computed).

mHC eq. (20): activation memory per token, in units of d (= C in the paper), when only the first input x_{l0}
of every block of L_r layers is kept (n d each) and one block's intermediates are rebuilt at a time
((n + 2) d per layer):  f(L_r) = n * ceil(L / L_r) + (n + 2) * L_r,  optimum ~ sqrt(n L / (n + 2)).
n = 4. Curves: L = 60 (mHC 27B: 30 blocks x 2 sublayers) and L = 86 (V4-Flash sublayers; whether the paper's
L counts blocks or sublayers is not stated -- research notes section 8 item 7). Notes N17.
Writes deepseek/figures/l16-recompute.svg
"""

import math

from l16_svg import SVG

n = 4


def f(L, r):
    return n * math.ceil(L / r) + (n + 2) * r


def best(L):
    r = min(range(1, L + 1), key=lambda r: f(L, r))
    return r, f(L, r)


for L in (60, 86):
    r, v = best(L)
    print("L=%d  continuous %.2f  integer best L_r=%d  -> %dd   (store all: %dd, store x_l only: %dd)"
          % (L, math.sqrt(n * L / (n + 2)), r, v, (n + 2) * L, n * L))

W, H = 760, 420
svg = SVG(W, H, "重计算块长 L_r 与激活显存（mHC 式 (20)）")
X0, X1, Y0, Y1 = 80, 560, 40, 350
RMAX, VMAX = 30, 540


def px(r):
    return X0 + (r - 1) / (RMAX - 1) * (X1 - X0)


def py(v):
    return Y1 - v / VMAX * (Y1 - Y0)


for v in range(0, VMAX + 1, 90):
    svg.line(X0, py(v), X1, py(v), color="grid", width=1)
    svg.text(X0 - 8, py(v) + 4, "%dd" % v if v else "0", size=11, anchor="end", color="muted")
for r in (1, 5, 10, 15, 20, 25, 30):
    svg.line(px(r), Y0, px(r), Y1, color="grid", width=1)
    svg.text(px(r), Y1 + 17, str(r), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "重计算块长 L_{r}（每隔多少个子层存一次 *X*）", size=12, anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "每 token 的激活显存（单位 d）", size=12, anchor="middle", color="muted", rotate=-90)

for L, col, width in ((86, "blue", 1.8), (60, "amber", 2.4)):
    pts = ["%.1f %.1f" % (px(r), py(f(L, r))) for r in range(1, RMAX + 1)]
    svg.path("M" + " L".join(pts), stroke=col, width=width, join="round")
    r, v = best(L)
    svg.circle(px(r), py(v), 4.5, fill=col)
    lx = px(r) + 10
    svg.text(lx, py(v) + (22 if L == 60 else -34), "L = %d：L_{r} = %d 时最小，%dd" % (L, r, v), size=12, color=col,
             weight="600")
    # store-all reference
    ref = (n + 2) * L
    if ref <= VMAX:
        svg.line(X0, py(ref), X1, py(ref), color=col, width=1.2, dash="6 4")
        svg.text(X1 + 8, py(ref) + (-1 if L == 60 else 4), "L = %d 不重算：%dd" % (L, ref), size=12, color=col)
    svg.line(X0, py(n * L), X1, py(n * L), color=col, width=1, dash="2 4")
    svg.text(X1 + 8, py(n * L) + (12 if L == 86 else 4), "只存每层 *X*_{l}：%dd" % (n * L), size=12, color=col)

# line gaps of 22/40: the subscripts of the formula must not touch the line below
svg.text(X0 + 150, py(470) - 2, "f(L_{r}) = 4⌈L/L_{r}⌉ + 6L_{r}", size=13, color="ink")
svg.text(X0 + 150, py(470) + 20, "块长太短：存的块首输入多；块长太长：一次要重建的中间量多", size=12, color="muted")
svg.text(X0 + 150, py(470) + 38, "连续近似的最优点 √(4L/6)：L = 60 时 6.32，L = 86 时 7.57", size=12, color="muted")

svg.save("l16-recompute.svg")
