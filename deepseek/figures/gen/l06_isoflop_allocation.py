"""Figure 6.13: IsoFLOP curves (schematic) and the optimal model/data allocation (computed), after dsllm Fig. 4.

(a) Eight budgets 1e17 .. 3e20. Valley positions M_opt = 0.1715 C^0.5243 (eq. 4, computed here).
    Valley heights are EYEBALLED from the paper's figure (research notes F5): 1.38, 1.29, 1.20, 1.125,
    1.05, 0.99, 0.93, 0.88 bits-per-byte. Curve widths/curvature are illustrative only.
(b) M_opt(C) = 0.1715 C^0.5243 and (c) D_opt(C) = 5.8316 C^0.4757, with the 8 budget points (notes R17/R18),
    67B: M = 4.3273e11 -> C_opt = 4.51e23, D_opt = 1.04e12; actual run C = 8.65e23, D = 2e12.
Writes deepseek/figures/l06-isoflop-allocation.svg
"""

import math

import numpy as np

from l06_svg import SVG

budgets = [1e17, 3e17, 1e18, 3e18, 1e19, 3e19, 1e20, 3e20]
valley = [1.38, 1.29, 1.20, 1.125, 1.05, 0.99, 0.93, 0.88]      # eyeballed (notes F5)


def Mopt(C):
    return 0.1715 * C ** 0.5243


def Dopt(C):
    return 5.8316 * C ** 0.4757


M67 = 4.3273e11
C67opt = (M67 / 0.1715) ** (1 / 0.5243)
assert abs(C67opt / 4.5113e23 - 1) < 1e-3
C67 = M67 * 2e12

W, H = 760, 660
svg = SVG(W, H, "IsoFLOP 曲线（示意）与最优模型/数据分配（按公式计算）")

# ---------------- (a) IsoFLOP
AX0, AX1, AY0, AY1 = 80, 560, 44, 262
LM0, LM1 = math.log10(3e7), math.log10(1e11)
BP0, BP1 = 0.8, 1.6


def ax(M):
    return AX0 + (math.log10(M) - LM0) / (LM1 - LM0) * (AX1 - AX0)


def ay(v):
    return AY1 - (v - BP0) / (BP1 - BP0) * (AY1 - AY0)


svg.text(AX0, AY0 - 20, "(a) IsoFLOP：每个预算内改变模型/数据的分配（示意）", size=13, weight="600")
for v in (0.8, 1.0, 1.2, 1.4, 1.6):
    svg.line(AX0, ay(v), AX1, ay(v), color="grid", width=1)
    svg.text(AX0 - 6, ay(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
for e in range(8, 12):
    x = ax(10 ** e)
    svg.line(x, AY0, x, AY1, color="grid", width=1)
    svg.text(x, AY1 + 16, "10^{%d}" % e, size=11, anchor="middle", color="muted")
svg.line(AX0, AY1, AX1, AY1, color="muted", width=1.2)
svg.line(AX0, AY0, AX0, AY1, color="muted", width=1.2)
svg.text((AX0 + AX1) / 2, AY1 + 36, "模型规模 M（FLOPs/token，对数刻度）", size=12, anchor="middle", color="muted")
svg.text(24, (AY0 + AY1) / 2, "验证集 BPB", size=12, anchor="middle", color="muted", rotate=-90)

K = 0.28   # illustrative curvature, bits-per-byte per decade^2
opts = []
for i, (C, v0) in enumerate(zip(budgets, valley)):
    m0 = math.log10(Mopt(C))
    xs = np.linspace(m0 - 0.75, m0 + 0.75, 60)
    pts = [(ax(10 ** m), ay(v0 + K * (m - m0) ** 2)) for m in xs]
    pts = [p for p in pts if AX0 <= p[0] <= AX1 and AY0 <= p[1] <= AY1]
    shade = 0.35 + 0.65 * i / 7
    svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke="violet", width=1.6, opacity=shade)
    opts.append((ax(10 ** m0), ay(v0)))
    svg.circle(ax(10 ** m0), ay(v0), 3.6, fill="ink")
svg.path("M" + " L".join("%.1f %.1f" % p for p in opts), stroke="ink", width=1, dash="3 3")
svg.text(opts[0][0] - 6, opts[0][1] - 12, "10^{17}", size=11, anchor="middle", color="violet")
svg.text(opts[-1][0] + 10, opts[-1][1] + 18, "3×10^{20}", size=11, color="violet")
nx = AX1 + 18
svg.text(nx, 70, "每条 U 形曲线：", size=12)
svg.text(nx, 88, "同一个算力预算，", size=12)
svg.text(nx, 106, "约 10 种模型/数据分配", size=12)
svg.circle(nx + 4, 128, 3.6, fill="ink")
svg.text(nx + 14, 132, "谷底 = 该预算的", size=12)
svg.text(nx + 14, 150, "最优模型规模 M_{opt}", size=12)
svg.text(nx, 184, "谷底横坐标按 (6.14) 计算；", size=12, color="muted")
svg.text(nx, 202, "谷底高度为对原图的目测；", size=12, color="muted")
svg.text(nx, 220, "曲线宽窄为示意", size=12, color="muted")

# ---------------- (b) and (c)
PY0, PY1 = 352, 590
LC0, LC1 = 16, 25


def panel(x0, x1, title, f, ylo, yhi, ylab, col, pts, mark_opt, mark_actual, opt_label, act_label):
    def px(C):
        return x0 + (math.log10(C) - LC0) / (LC1 - LC0) * (x1 - x0)

    def py(v):
        return PY1 - (math.log10(v) - ylo) / (yhi - ylo) * (PY1 - PY0)

    svg.text(x0 - 40, PY0 - 18, title, size=13, weight="600")
    for e in range(ylo, yhi + 1):
        y = py(10 ** e)
        svg.line(x0, y, x1, y, color="grid", width=1)
        if (e - ylo) % 2 == 0:
            svg.text(x0 - 6, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
    for e in range(LC0, LC1 + 1):
        x = px(10 ** e)
        svg.line(x, PY0, x, PY1, color="grid", width=1)
        if e % 2 == 0:
            svg.text(x, PY1 + 16, "10^{%d}" % e, size=11, anchor="middle", color="muted")
    svg.line(x0, PY1, x1, PY1, color="muted", width=1.2)
    svg.line(x0, PY0, x0, PY1, color="muted", width=1.2)
    svg.text((x0 + x1) / 2, PY1 + 36, "训练算力 C（FLOPs）", size=12, anchor="middle", color="muted")
    svg.text(x0 - 46, (PY0 + PY1) / 2, ylab, size=12, anchor="middle", color="muted", rotate=-90)
    Cs = np.logspace(LC0, LC1, 400)
    line_pts = [(px(C), py(f(C))) for C in Cs if 10 ** ylo <= f(C) <= 10 ** yhi]   # clip to the plot box
    svg.path("M" + " L".join("%.1f %.1f" % p for p in line_pts), stroke=col, width=2, dash="6 3")
    for C in budgets:
        svg.circle(px(C), py(f(C)), 3.6, fill="muted")
    Co, vo = mark_opt
    svg.line(px(Co), py(vo), px(Co), PY1, color="ink", width=1, dash="2 3")
    svg.line(x0, py(vo), px(Co), py(vo), color="ink", width=1, dash="2 3")
    svg.circle(px(Co), py(vo), 5, fill=col)
    svg.text(px(Co) - 8, py(vo) - 10, opt_label, size=12, anchor="end", color=col)
    if mark_actual:
        Ca, va = mark_actual
        pts_star = []
        for k in range(10):
            r = 8 if k % 2 == 0 else 3.5
            a = math.pi / 2 + k * math.pi / 5
            pts_star.append((px(Ca) + r * math.cos(a), py(va) - r * math.sin(a)))
        svg.polygon(pts_star, fill="green", stroke="green", width=1)
        svg.text(px(Ca) + 4, py(va) - 14, act_label, size=12, anchor="end", color="green")
    return px, py


pxb, pyb = panel(98, 346, "(b) 最优模型规模", Mopt, 8, 13, "M_{opt}（FLOPs/token）", "blue",
                 budgets, (C67opt, M67), None, "67B：4.3×10^{11}", "")
svg.text(pxb(C67opt), PY1 - 6, "4.5×10^{23}", size=11, anchor="end", color="ink")
pxc, pyc = panel(476, 724, "(c) 最优数据量", Dopt, 8, 13, "D_{opt}（token）", "green",
                 budgets, (C67opt, Dopt(C67opt)), (C67, 2e12), "1.04×10^{12}", "67B 实际 2T")
svg.text(pxc(C67opt), PY1 - 6, "4.5×10^{23}", size=11, anchor="end", color="ink")
svg.text(476, PY1 + 58, "星：67B 实际 2T，是 1.04×10^{12} 的 %.2f 倍" % (2e12 / Dopt(C67opt)), size=12,
         color="green")
svg.text(98, PY1 + 58, "灰点：8 个 IsoFLOP 预算的最优点", size=12, color="muted")

svg.save("l06-isoflop-allocation.svg")
