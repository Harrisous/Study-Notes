"""Figure 0.6: why the residual shortcut must carry exactly "1" (thread E).

If every one of L sub-layers multiplies the shortcut by a scalar lambda, the direct path carries lambda^L
(lesson 16, eq. 16.5): 1.05^60 = 18.7, 0.95^60 = 0.046, and 60 layers reach ~3000 with lambda = 1.1428.
The two markers on the right are the measured composite backward gains on the 27B model of the mHC paper,
as digitised in lesson 16 (HC ~3000, digitised 2825; mHC ~1.6, digitised 1.65).

    python deepseek/figures/gen/l00_residual_gain.py
"""

from __future__ import annotations

import math

from l00_svg import SVG, fmt

CURVES = [(1.1428, "red", "每层 ×1.1428"), (1.05, "orange", "每层 ×1.05"), (1.0, "green", "每层 ×1（恒等映射）"),
          (0.95, "blue", "每层 ×0.95")]
L_MAX = 60

W, H = 720, 400
L, R, T, B = 70, 560, 20, 350
Y0, Y1 = -2, 4          # log10 gain


def xl(n):
    return L + (R - L) * n / L_MAX


def yg(g):
    return B - (B - T) * (math.log10(g) - Y0) / (Y1 - Y0)


def main():
    s = SVG(W, H, title="捷径系数连乘：偏离 1 的代价随深度指数增长")
    for e in range(Y0, Y1 + 1):
        y = yg(10 ** e)
        s.line(L, y, R, y, color="rule" if e == 0 else "grid", width=1)
        s.text(L - 8, y + 4, "1" if e == 0 else "10^{%d}" % e, size=12, anchor="end", color="muted")
    for n in range(0, L_MAX + 1, 10):
        x = xl(n)
        s.line(x, B, x, B + 5, color="muted", width=1)
        s.text(x, B + 20, str(n), size=12, anchor="middle", color="muted")
    s.line(L, B, R, B, color="muted", width=1)
    s.text((L + R) / 2, B + 42, "经过的子层数", size=12.5, anchor="middle")
    s.text(L - 46, (T + B) / 2, "直通项的增益（对数轴）", size=12.5, anchor="middle", rotate=-90)
    for lam, col, lab in CURVES:
        pts = [(xl(n), yg(lam ** n)) for n in range(0, L_MAX + 1)]
        s.path("M" + " L".join("%s %s" % (fmt(x), fmt(y)) for x, y in pts), stroke=col, width=2.2)
        g = lam ** L_MAX
        s.text(R + 8, yg(g) + 4, "%s：%s" % (lab, ("%.3g" % g) if g < 100 else f"{round(g):,}"), size=12,
               color=col, weight=600)
    # measured composite gains (27B, mHC paper, digitised in lesson 16)
    for g, col, lab in ((2825, "red", "HC 实测约 3000"), (1.65, "green", "mHC 实测约 1.6")):
        x, y = xl(L_MAX), yg(g)
        s.polygon([(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)], fill="surface", stroke=col, width=1.8)
        s.text(x - 10, y - 10, lab + "（27B，矩阵连乘）", size=11.5, anchor="end",
               color=col)
    s.save("l00-residual-gain.svg")
    for lam, *_ in CURVES:
        print(lam, lam ** L_MAX)


if __name__ == "__main__":
    main()
