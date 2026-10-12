"""Figure 8.14: RFT vs Online RFT vs GRPO (outcome / process) on DeepSeekMath-Instruct 1.3B.

Redraw of DeepSeekMath v3 Figure 5 from the research notes' samples every 1,000 steps (section 3.6),
digitised from the vector paths of the v3 PDF (about +/- 0.1 point). The original has one point
every 200 steps and is noisy; the last-5-point means (right labels) come from the full 45 points.
Writes deepseek/figures/l08-methods-curves.svg
"""

from l08_svg import SVG, polyline

steps = [1000 * k for k in range(9)]
gsm = {
    "RFT": [56.4, 59.7, 59.6, 61.3, 60.6, 59.0, 59.9, 60.2, 60.3],
    "Online RFT": [56.4, 60.5, 62.3, 59.8, 59.9, 61.6, 60.7, 62.7, 62.2],
    "GRPO+OS": [56.4, 60.3, 61.5, 61.9, 64.1, 63.4, 64.0, 62.7, 62.6],
    "GRPO+PS": [56.4, 60.5, 61.5, 61.7, 63.8, 63.3, 64.7, 64.8, 64.9],
}
mth = {
    "RFT": [26.5, 28.1, 28.0, 27.5, 27.8, 28.0, 27.9, 27.6, 27.4],
    "Online RFT": [26.5, 27.6, 28.2, 28.6, 30.2, 29.4, 29.0, 28.9, 29.1],
    "GRPO+OS": [26.5, 28.1, 29.0, 29.2, 30.0, 29.8, 30.1, 29.5, 29.3],
    "GRPO+PS": [26.5, 28.8, 29.6, 29.8, 30.3, 29.7, 30.4, 29.6, 29.7],
}
last5 = {"GSM8K": {"RFT": 60.1, "Online RFT": 61.9, "GRPO+OS": 63.2, "GRPO+PS": 64.9},
         "MATH": {"RFT": 27.9, "Online RFT": 29.0, "GRPO+OS": 29.3, "GRPO+PS": 29.7}}
COL = {"RFT": "muted", "Online RFT": "amber", "GRPO+OS": "blue", "GRPO+PS": "violet"}
DASH = {"RFT": "5 4", "Online RFT": None, "GRPO+OS": None, "GRPO+PS": None}

W, H = 740, 340
svg = SVG(W, H, "四种方法在 DeepSeekMath-Instruct 1.3B 上的训练曲线（论文图 5 重绘）")


def panel(x0, data, key, ylo, yhi, ticks):
    PX0, PX1, PY0, PY1 = x0 + 40, x0 + 262, 44, 262

    def px(s):
        return PX0 + s / 8000 * (PX1 - PX0)

    def py(v):
        return PY1 - (v - ylo) / (yhi - ylo) * (PY1 - PY0)

    for v in ticks:
        svg.line(PX0, py(v), PX1, py(v), color="grid", width=1)
        svg.text(PX0 - 6, py(v) + 4, "%g" % v, size=11, anchor="end", color="muted")
    for s in (0, 2000, 4000, 6000, 8000):
        svg.line(px(s), PY0, px(s), PY1, color="grid", width=1)
        svg.text(px(s), PY1 + 16, "%d" % s, size=11, anchor="middle", color="muted")
    svg.line(PX0, PY1, PX1, PY1, color="muted", width=1.1)
    svg.line(PX0, PY0, PX0, PY1, color="muted", width=1.1)
    svg.text(x0 + 10, 26, "%s（%%）" % key, size=13, weight="700")
    svg.text((PX0 + PX1) / 2, PY1 + 36, "训练步数", size=12, anchor="middle", color="muted")
    for name, vals in data.items():
        polyline(svg, [(px(s), py(v)) for s, v in zip(steps, vals)], color=COL[name],
                 width=2.6 if name.startswith("GRPO") else 2.0, dash=DASH[name])
    # right labels: last-5 means, spread to avoid overlap
    items = sorted(last5[key].items(), key=lambda kv: -kv[1])
    ys, prev = [], None
    for name, v in items:
        y = py(v) + 4
        if prev is not None and y - prev < 15:
            y = prev + 15
        ys.append((name, v, y))
        prev = y
    for name, v, y in ys:
        svg.text(PX1 + 6, y, "%s %.1f" % (name, v), size=11.5, color=COL[name] if name != "RFT" else "ink",
                 weight="600")


panel(10, gsm, "GSM8K", 56, 66, (56, 58, 60, 62, 64, 66))
panel(380, mth, "MATH", 26, 31, (26, 27, 28, 29, 30, 31))
svg.text(16, H - 10, "右侧数字：最后 5 个点（每 200 步一点）的平均。曲线为每 1,000 步取样；RFT 用虚线，因为它的数据来自冻结的 SFT 模型（离线）。",
         size=11.5, color="muted")
svg.save("l08-methods-curves.svg")
