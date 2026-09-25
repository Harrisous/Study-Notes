"""Figure 16.1: the product of per-layer scalar gains g over L sublayers, g^L (log axis).

Numbers (lesson-16 research notes, N18, N9): 1.05^60 = 18.68, 1.1^60 = 304.5,
0.9^60 = 0.00180, 0.95^60 = 0.0461, 3000^(1/60) = 1.1428; Post-Norm shortcut 1/sqrt(2)
per layer gives (2^-0.5)^60 = 9.31e-10 (the author's assumption sigma_i = sigma_o = 1).
Writes deepseek/figures/l16-scalar-gain.svg
"""

import math

from l16_svg import SVG

G_UP = [(1.05, "1.05"), (1.1, "1.1"), (3000 ** (1 / 60), "1.1428")]
G_DN = [(0.95, "0.95"), (0.9, "0.9")]
POST = 2 ** -0.5
LMAX = 60

assert abs(1.05 ** 60 - 18.68) < 0.01 and abs(1.1 ** 60 - 304.48) < 0.01
assert abs(3000 ** (1 / 60) - 1.14275) < 1e-4 and abs(POST ** 60 - 9.31e-10) < 1e-12

W, H = 740, 430
X0, X1, Y0, Y1 = 82, 540, 34, 360
E0, E1 = -4, 4            # 1e-4 .. 1e4


def px(l):
    return X0 + l / LMAX * (X1 - X0)


def py(v):
    e = math.log10(v)
    return Y1 - (e - E0) / (E1 - E0) * (Y1 - Y0)


svg = SVG(W, H, "每个子层把直通信号乘 g，60 个子层后是 g 的 60 次方（对数纵轴）")

for e in range(E0, E1 + 1):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    lab = {0: "1", 1: "10", 2: "100", 3: "1000", -1: "0.1", -2: "0.01", -3: "0.001"}.get(e, "10^{%d}" % e)
    svg.text(X0 - 8, y + 4, lab, size=11, anchor="end", color="muted")
for l in range(0, LMAX + 1, 10):
    x = px(l)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, str(l), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 42, "经过的子层数 L", size=12, anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "直通项的总倍数 g^{L}（对数刻度）", size=12, anchor="middle", color="muted",
         rotate=-90)


def curve(g, color, width, dash=None):
    pts = []
    for i in range(0, 121):
        l = i * LMAX / 120
        v = g ** l
        if v < 10 ** E0 or v > 10 ** E1:
            break
        pts.append((px(l), py(v)))
    d = "M" + " L".join("%.1f %.1f" % p for p in pts)
    svg.path(d, stroke=color, width=width, dash=dash)
    return pts[-1]


# reference line 3000
y3000 = py(3000)
svg.line(X0, y3000, X1, y3000, color="red", width=1, dash="2 4", opacity=0.7)
svg.text(X0 + 6, y3000 - 6, "3000：HC 在 27B 上的复合增益峰值（论文文字）", size=12, color="red")

# g = 1
curve(1.0, "green", 2.8)
svg.text(X1 + 10, py(1) + 4, "g = 1：恒等捷径，永远是 1", size=13, color="green", weight="600")

for g, lab in G_UP:
    end = curve(g, "red", 2.0 if g < 1.12 else 2.6)
    val = g ** LMAX
    txt = "g = %s → %s" % (lab, "%.1f" % val if val < 1000 else "≈ %d" % round(val))
    svg.text(X1 + 10, end[1] + 4, txt, size=12, color="red")
for g, lab in G_DN:
    end = curve(g, "red", 2.0, dash="6 4")
    val = g ** LMAX
    svg.text(X1 + 10, end[1] + 4, "g = %s → %s" % (lab, ("%.4f" % val) if val < 0.01 else ("%.3f" % val)),
             size=12, color="red")

# Post-Norm 1/sqrt 2 : leaves the chart
curve(POST, "muted", 2.0, dash="2 3")
svg.text(px(28.5), py(4e-4), "点线 g = 1/√2（Post-Norm 式缩放）：", size=12, color="muted")
svg.text(px(28.5), py(4e-4) + 17, "约 27 层就跌出图外，60 层后是 9.3 × 10^{-10}", size=12, color="muted")

svg.text(X0 + 6, py(600), "实线：g > 1，指数爆炸", size=12, color="muted")
svg.text(X0 + 6, py(600) + 17, "虚线：g < 1，指数消失", size=12, color="muted")

svg.save("l16-scalar-gain.svg")
