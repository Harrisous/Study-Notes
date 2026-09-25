"""Figure 8.10: the KL part of GRPO's gradient coefficient, original k3 versus DeepSeek-V3.2's fix.

With w = pi_ref / pi_theta and beta = 0.04 (DeepSeekMath):
  original k3 in the loss      -> coefficient beta * (w - 1)     (DeepSeekMath eq. 21)
  V3.2 k3 times ratio (rho = 1) -> coefficient beta * log w       (research notes 2.5(g))
Values at w = 0.01, 0.5, 2, 10, 100, 1000 are printed (research notes 4.3). Curves are clipped
to the plotting window here. Writes deepseek/figures/l08-kl-coef.svg
"""

import math

from l08_svg import SVG, polyline

beta = 0.04
for w in (0.01, 0.1, 0.5, 0.9, 1, 2, 10, 100, 1000):
    print("w=%g  beta(w-1)=%+.4f  beta*log w=%+.4f" % (w, beta * (w - 1), beta * math.log(w)))

W, H = 740, 350
svg = SVG(W, H, "KL 项在梯度系数里的贡献：原始 k3 与 V3.2 的修正")
PX0, PX1, PY0, PY1 = 70, 610, 30, 280
LX0, LX1 = -2.0, 3.0            # log10 w
YL, YH = -0.4, 4.4


def px(w):
    return PX0 + (math.log10(w) - LX0) / (LX1 - LX0) * (PX1 - PX0)


def py(v):
    return PY1 - (v - YL) / (YH - YL) * (PY1 - PY0)


for v in (0, 1, 2, 3, 4):
    svg.line(PX0, py(v), PX1, py(v), color="grid", width=1)
    svg.text(PX0 - 6, py(v) + 4, "%d" % v, size=11, anchor="end", color="muted")
for e in range(-2, 4):
    svg.line(px(10 ** e), PY0, px(10 ** e), PY1, color="grid", width=1)
    svg.text(px(10 ** e), PY1 + 16, ("%g" % (10 ** e)), size=11, anchor="middle", color="muted")
svg.line(PX0, py(0), PX1, py(0), color="muted", width=1.1)
svg.line(px(1), PY0, px(1), PY1, color="muted", width=1, dash="3 3")
svg.text((PX0 + PX1) / 2, PY1 + 36, "w = π_{ref} / π_{θ}（对数轴）", size=12, anchor="middle", color="muted")
svg.text(24, (PY0 + PY1) / 2, "梯度系数里的 KL 部分", size=12, anchor="middle", color="muted", rotate=-90)

# typical advantage magnitude
svg.line(PX0, py(1), PX1, py(1), color="green", width=1.4, dash="6 4")
svg.text(PX0 + 6, py(1) - 6, "二值奖励、两对两错时的优势 |Â| = 1", size=12, color="green", halo=True)

N = 800
ws = [10 ** (LX0 + (LX1 - LX0) * k / N) for k in range(N + 1)]
orig = [(px(w), py(beta * (w - 1))) for w in ws if beta * (w - 1) <= YH]
fix = [(px(w), py(beta * math.log(w))) for w in ws]
polyline(svg, orig, color="red", width=2.6)
polyline(svg, fix, color="blue", width=2.6)

# end of the original curve leaving the window
w_out = 1 + YH / beta
svg.arrow(px(w_out) - 2, py(YH) + 10, px(w_out) + 10, py(YH) - 4, color="red", width=1.6)
svg.text(px(w_out) + 14, py(YH) + 8, "w = 1000 时 39.96", size=12, color="red", halo=True)

for w in (100,):
    a, b = beta * (w - 1), beta * math.log(w)
    svg.circle(px(w), py(a), 4, fill="red")
    svg.circle(px(w), py(b), 4, fill="blue")
    svg.text(px(w) - 8, py(a) + 4, "w = 100：3.96", size=12, anchor="end", color="red", weight="600", halo=True)
    svg.text(px(w) + 6, py(b) - 8, "0.184", size=12, color="blue", weight="600", halo=True)
w = 0.01
svg.circle(px(w), py(beta * (w - 1)), 3.5, fill="red")
svg.circle(px(w), py(beta * math.log(w)), 3.5, fill="blue")
svg.text(px(w) + 8, py(beta * math.log(w)) + 16, "w = 0.01：−0.040 与 −0.184", size=11.5, color="ink", halo=True)

# legend at the right
lx = 624
svg.line(lx, 60, lx + 22, 60, color="red", width=2.6)
svg.text(lx, 82, "原始 k3", size=12.5, color="red", weight="600")
svg.text(lx, 100, "β(w − 1)", size=12.5, color="red")
svg.text(lx, 118, "线性、无界", size=12, color="muted")
svg.line(lx, 150, lx + 22, 150, color="blue", width=2.6)
svg.text(lx, 172, "V3.2 修正", size=12.5, color="blue", weight="600")
svg.text(lx, 190, "β log w", size=12.5, color="blue")
svg.text(lx, 208, "只按对数增长", size=12, color="muted")
svg.text(lx, 246, "β = 0.04", size=12, color="ink")

svg.text(16, H - 8, "右半边（w > 1）：策略写出这个 token 的概率比参考模型低。w 越大，原始 k3 把它往回拉的力度越大，而且没有上限。",
         size=12, color="muted")
svg.save("l08-kl-coef.svg")
