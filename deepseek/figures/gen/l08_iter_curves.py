"""Figure 8.12: iterative RL on DeepSeekMath-Instruct 7B (redraw of DeepSeekMath v3 Figure 6).

Points are the research notes' samples every 500 steps (section 3.6), digitised from the
matplotlib vector paths of the v3 PDF (about +/- 0.1 point). The x position is the step value on
the original linear axis (ticks labelled 0, 1300, 2300, ... in the paper); the paper does not
explain why later iterations start at about 1,100 and 2,700 steps.
Writes deepseek/figures/l08-iter-curves.svg
"""

from l08_svg import SVG, polyline

gsm = {
    0: [(0, 82.9), (500, 85.1), (1000, 86.1), (1500, 85.7), (2000, 85.5), (2500, 86.7), (2800, 86.1)],
    1: [(1100, 87.0), (1500, 86.8), (2000, 86.9), (2500, 88.2), (3000, 87.3), (3500, 86.9), (4000, 87.9),
        (4500, 88.1), (5000, 88.2), (5400, 87.9)],
    2: [(2700, 87.8), (3000, 87.5), (3500, 88.0), (4000, 88.6), (4500, 87.9), (5000, 88.0), (5300, 88.8)],
}
math_ = {
    0: [(0, 46.8), (500, 48.0), (1000, 49.2), (1500, 49.8), (2000, 49.2), (2500, 49.0), (2800, 48.7)],
    1: [(1100, 49.3), (1500, 50.1), (2000, 50.4), (2500, 50.6), (3000, 50.9), (3500, 50.8), (4000, 51.1),
        (4500, 51.3), (5000, 51.6), (5400, 51.6)],
    2: [(2700, 51.1), (3000, 51.0), (3500, 51.0), (4000, 51.1), (4500, 51.6), (5000, 51.3), (5300, 51.3)],
}
COL = {0: "muted", 1: "blue", 2: "violet"}
W, H = 740, 330
svg = SVG(W, H, "迭代 RL：三轮迭代的 GSM8K 与 MATH 成绩（论文图 6 重绘）")


def panel(x0, data, title, ylo, yhi, ticks):
    PX0, PX1, PY0, PY1 = x0 + 40, x0 + 330, 44, 262

    def px(s):
        return PX0 + s / 5500 * (PX1 - PX0)

    def py(v):
        return PY1 - (v - ylo) / (yhi - ylo) * (PY1 - PY0)

    for v in ticks:
        svg.line(PX0, py(v), PX1, py(v), color="grid", width=1)
        svg.text(PX0 - 6, py(v) + 4, "%g" % v, size=11, anchor="end", color="muted")
    for s in (0, 1000, 2000, 3000, 4000, 5000):
        svg.line(px(s), PY0, px(s), PY1, color="grid", width=1)
        svg.text(px(s), PY1 + 16, "%d" % s, size=11, anchor="middle", color="muted")
    svg.line(PX0, PY1, PX1, PY1, color="muted", width=1.1)
    svg.line(PX0, PY0, PX0, PY1, color="muted", width=1.1)
    svg.text(x0 + 10, 26, title, size=13, weight="700")
    svg.text((PX0 + PX1) / 2, PY1 + 36, "训练步数", size=12, anchor="middle", color="muted")
    for it, pts in data.items():
        polyline(svg, [(px(s), py(v)) for s, v in pts], color=COL[it], width=2.2)
        for s, v in pts:
            svg.circle(px(s), py(v), 2.6, fill=COL[it])
    return px, py


px, py = panel(10, gsm, "GSM8K（Top1，%）", 82, 90, (82, 84, 86, 88, 90))
svg.text(px(60), py(82.9) + 16, "起点 82.9（= Instruct）", size=11.5, color="muted", halo=True)
svg.text(px(1500), py(89.4), "第 1 轮开始 87.0", size=11.5, color="blue", halo=True)
svg.text(px(3700), py(89.3), "第 2 轮 → 88.8", size=11.5, color="violet", halo=True)
px, py = panel(380, math_, "MATH（Top1，%）", 46, 53, (46, 48, 50, 52))
svg.text(px(60), py(46.8) + 16, "起点 46.8", size=11.5, color="muted", halo=True)
svg.text(px(1900), py(52.4), "第 1 轮 → 51.6", size=11.5, color="blue", halo=True)
svg.text(px(3100), py(50.3), "第 2 轮起点 51.1", size=11.5, color="violet", halo=True)

ly = H - 12
for it, lab, x in ((0, "第 0 轮（初始奖励模型）", 16), (1, "第 1 轮", 230), (2, "第 2 轮", 330)):
    svg.line(x, ly - 4, x + 22, ly - 4, color=COL[it], width=2.4)
    svg.text(x + 28, ly, lab, size=12, color="ink")
svg.text(430, ly, "每 500 步取样；数字化误差约 ±0.1", size=12, color="muted")
svg.save("l08-iter-curves.svg")
