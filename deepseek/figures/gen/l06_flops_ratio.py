"""Figure 6.10: 6N1/M and 6N2/M versus model width d (computed; points = the seven rows of dsllm Table 3).

Closed form: 6N1/M = 6d / (6d + T).  6N2/M = (72Ld^2 + 6Vd) / (72Ld^2 + 12LdT), which also depends on L,
so it is drawn only through the Table 3 rows. V = 102400, T = 4096 unless stated.
Writes deepseek/figures/l06-flops-ratio.svg
"""

import math

import numpy as np

from l06_svg import SVG

V, T = 102400, 4096
table3 = [(8, 512), (12, 768), (24, 1024), (24, 2048), (32, 4096), (40, 5120), (80, 8192)]


def r1(L, d, T=T):
    return 72 * L * d * d / (72 * L * d * d + 12 * L * d * T)


def r2(L, d, T=T):
    return (72 * L * d * d + 6 * V * d) / (72 * L * d * d + 12 * L * d * T)


W, H = 760, 420
X0, X1, Y0, Y1 = 70, 580, 30, 350
DX0, DX1 = math.log2(512), math.log2(16384)
YLO, YHI = 0.3, 1.4


def px(d):
    return X0 + (math.log2(d) - DX0) / (DX1 - DX0) * (X1 - X0)


def py(v):
    return Y1 - (v - YLO) / (YHI - YLO) * (Y1 - Y0)


svg = SVG(W, H, "6N1/M 与 6N2/M 随隐藏维度 d 的变化")
for k in range(3, 15):
    v = k / 10
    y = py(v)
    svg.line(X0, y, X1, y, color="grid", width=1)
    if k % 2 == 1 or k == 14:
        svg.text(X0 - 8, y + 4, "%.1f" % v, size=11, anchor="end", color="muted")
for d in (512, 1024, 2048, 4096, 8192, 16384):
    x = px(d)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, str(d), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "隐藏维度 d（对数刻度）", size=12, anchor="middle", color="muted")
svg.text(20, (Y0 + Y1) / 2, "与 M 之比", size=12, anchor="middle", color="muted", rotate=-90)

# y = 1 reference
svg.line(X0, py(1), X1, py(1), color="ink", width=1.3, dash="5 3")
svg.text(X1 + 6, py(1) + 4, "= M", size=12, color="ink")


ds = np.exp2(np.linspace(DX0, DX1, 200))
for TT, dash, lab in ((2048, "2 3", "T = 2048"), (4096, None, "T = 4096"), (16384, "6 3", "T = 16384")):
    ys = [6 * d / (6 * d + TT) for d in ds]
    pts = [(px(d), py(v)) for d, v in zip(ds, ys) if YLO <= v <= YHI]
    svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke="blue", width=2 if TT == 4096 else 1.4, dash=dash,
             opacity=1 if TT == 4096 else 0.8)

# Table 3 points
p2 = [(px(d), py(r2(L, d))) for L, d in table3]
svg.path("M" + " L".join("%.1f %.1f" % p for p in p2), stroke="muted", width=1.4)
for (L, d), (x, y) in zip(table3, p2):
    svg.rect(x - 4, y - 4, 8, 8, fill="muted", stroke="muted")
for L, d in table3:
    svg.circle(px(d), py(r1(L, d)), 4.5, fill="blue")

# labels for the extremes
L, d = table3[0]
svg.text(px(d) + 10, py(r2(L, d)) - 6, "6N_{2}/M = %.2f（高估 %d%%）" % (r2(L, d), round((r2(L, d) - 1) * 100)),
         size=12, color="muted")
svg.text(px(d) + 10, py(r1(L, d)) + 18, "6N_{1}/M = %.2f（低估 %d%%）" % (r1(L, d), round((1 - r1(L, d)) * 100)),
         size=12, color="blue")
svg.text(px(512) + 10, py(r1(8, 512)) + 34, "L = 8，d = 512", size=12, color="muted")
# legend on the right
lx, ly = X1 + 34, 60
svg.text(lx, ly, "6N_{1}/M 闭式曲线：", size=12, color="blue")
for i, (dash, w, lab) in enumerate(((" 2 3", 1.4, "T = 2048"), (None, 2, "T = 4096"), ("6 3", 1.4, "T = 16384"))):
    yy = ly + 22 + i * 20
    svg.line(lx, yy, lx + 28, yy, color="blue", width=w, dash=dash.strip() if dash else None)
    svg.text(lx + 36, yy + 4, lab, size=12, color="blue")
svg.circle(lx + 14, ly + 100, 4.5, fill="blue")
svg.text(lx + 36, ly + 104, "表 3：6N_{1}/M", size=12, color="blue")
svg.rect(lx + 10, ly + 118, 8, 8, fill="muted", stroke="muted")
svg.text(lx + 36, ly + 126, "表 3：6N_{2}/M", size=12, color="muted")
svg.text(lx, ly + 160, "d = 8192 处：", size=12, color="ink")
svg.text(lx, ly + 178, "0.92 与 0.94", size=12, color="ink")

svg.save("l06-flops-ratio.svg")
