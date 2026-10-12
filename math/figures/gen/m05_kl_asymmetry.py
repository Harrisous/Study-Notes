"""Figure M5.6: KL is not symmetric. p = (a, 1-a) against the fair coin u = (1/2, 1/2).

KL(p||u) = ln 2 - H(p) never exceeds ln 2 = 0.693; KL(u||p) = -ln 2 - (ln a + ln(1-a))/2 grows without bound as
a -> 0 or 1. At a = 0.9 the two values are 0.368 and 0.511 nats (research notes N36). Computed here.
Writes math/figures/m05-kl-asymmetry.svg
"""

import math

from m05_svg import SVG

W, H = 720, 380
X0, X1, Y0, Y1 = 70, 560, 30, 320
YMAX = 1.6


def px(a):
    return X0 + a * (X1 - X0)


def py(v):
    return Y1 - v / YMAX * (Y1 - Y0)


def kl_pu(a):
    return sum(x * math.log(x / 0.5) for x in (a, 1 - a) if x > 0)


def kl_up(a):
    return 0.5 * math.log(0.5 / a) + 0.5 * math.log(0.5 / (1 - a))


svg = SVG(W, H, "KL 不对称：两点分布 p = (a, 1−a) 与公平硬币 u")
for i in range(11):
    a = i / 10
    svg.line(px(a), Y0, px(a), Y1, color="grid", width=1)
    svg.text(px(a), Y1 + 17, "%.1f" % a if i not in (0, 10) else str(i // 10), size=11, anchor="middle",
             color="muted")
for k in range(0, 9):
    v = k * 0.2
    svg.line(X0, py(v), X1, py(v), color="grid", width=1)
    svg.text(X0 - 8, py(v) + 4, "%.1f" % v if k else "0", size=11, anchor="end", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "p 的第一个概率 a", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "KL（nats）", size=12, anchor="middle", color="muted", rotate=-90)

# ln 2 ceiling
svg.line(X0, py(math.log(2)), X1, py(math.log(2)), color="blue", width=1, dash="3 4")
svg.text(px(0.5), py(math.log(2)) - 7, "ln 2 = 0.693：蓝线的上限", size=12, anchor="middle", color="blue")

n = 600
pts = [(px(k / n), py(kl_pu(k / n))) for k in range(n + 1)]
svg.path("M" + " L".join("%.1f %.1f" % xy for xy in pts), stroke="blue", width=2.6)
# KL(u||p) truncated at YMAX
seg = []
for k in range(1, n):
    a = k / n
    v = kl_up(a)
    if v <= YMAX:
        seg.append((px(a), py(v)))
    elif seg:
        svg.path("M" + " L".join("%.1f %.1f" % xy for xy in seg), stroke="red", width=2.6)
        seg = []
if seg:
    svg.path("M" + " L".join("%.1f %.1f" % xy for xy in seg), stroke="red", width=2.6)

# a = 0.9 markers
a = 0.9
for v, col in ((kl_pu(a), "blue"), (kl_up(a), "red")):
    svg.circle(px(a), py(v), 4.5, fill=col)
svg.line(px(a), py(0), px(a), py(kl_up(a)), color="muted", width=1, dash="2 3")
svg.text(px(a) - 5, Y1 - 6, "a = 0.9", size=12, anchor="end", color="muted")   # left of the dotted line

# legends at the right (the a = 0.9 values live here so they never sit on a curve)
LX = 574
svg.line(LX, 62, LX + 26, 62, color="blue", width=2.6)
svg.text(LX + 32, 66, "KL(p‖u)", size=12, color="blue", weight="600")
svg.text(LX, 84, "期望按 p 取", size=12, color="muted")
svg.text(LX, 102, "= ln 2 − H(p) ≤ ln 2", size=12, color="muted")
svg.text(LX, 120, "a = 0.9 时 %.3f" % kl_pu(a), size=12, color="blue", weight="600")
svg.line(LX, 160, LX + 26, 160, color="red", width=2.6)
svg.text(LX + 32, 164, "KL(u‖p)", size=12, color="red", weight="600")
svg.text(LX, 182, "期望按 u 取", size=12, color="muted")
svg.text(LX, 200, "a = 0.9 时 %.3f" % kl_up(a), size=12, color="red", weight="600")
svg.text(LX, 218, "a → 0 或 1 时 → ∞：", size=12, color="muted")
svg.text(LX, 236, "u 把一半质量放在", size=12, color="muted")
svg.text(LX, 254, "p 几乎为 0 的地方", size=12, color="muted")

svg.save("m05-kl-asymmetry.svg")
