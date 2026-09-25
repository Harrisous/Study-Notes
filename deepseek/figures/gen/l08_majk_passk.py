"""Figure 8.15: Maj@K and Pass@K of DeepSeekMath-Instruct 7B and -RL 7B (redraw of v3 Figure 7).

Data: research notes section 3.6 (48 points parsed from the vector paths of the v3 PDF,
about +/- 0.1 point; temperature 0.7). K = 1, 4, 8, 16, 32, 64 are equally spaced as in the paper.
The script also prints RL - Instruct differences (research notes 4.7) and the crossing of the
Pass@K curves. Writes deepseek/figures/l08-majk-passk.svg
"""

from l08_svg import SVG, polyline

Ks = [1, 4, 8, 16, 32, 64]
data = {
    "GSM8K": {
        ("maj", "inst"): [81.5, 85.8, 87.6, 89.3, 89.2, 89.5],
        ("maj", "rl"): [88.2, 89.3, 90.1, 90.5, 90.7, 91.0],
        ("pass", "inst"): [81.5, 92.3, 95.1, 96.7, 98.1, 99.1],
        ("pass", "rl"): [88.2, 93.0, 94.5, 95.8, 96.4, 97.3],
    },
    "MATH": {
        ("maj", "inst"): [44.6, 50.2, 55.1, 58.0, 59.2, 59.9],
        ("maj", "rl"): [50.9, 55.3, 57.6, 59.3, 60.4, 60.9],
        ("pass", "inst"): [44.6, 64.3, 71.7, 78.2, 83.4, 87.5],
        ("pass", "rl"): [50.9, 66.8, 72.8, 78.4, 83.1, 86.8],
    },
}
for b, d in data.items():
    for kind in ("maj", "pass"):
        diff = [round(r - i, 1) for r, i in zip(d[(kind, "rl")], d[(kind, "inst")])]
        print(b, kind, "RL - Instruct:", diff)

W, H = 740, 360
svg = SVG(W, H, "RL 前后的 Maj@K 与 Pass@K（论文图 7 重绘）")
COL = {"inst": "muted", "rl": "blue"}
DASH = {"maj": None, "pass": "6 4"}


def panel(x0, bench, ylo, yhi, ticks):
    PX0, PX1, PY0, PY1 = x0 + 40, x0 + 300, 44, 262

    def px(i):
        return PX0 + i / 5 * (PX1 - PX0)

    def py(v):
        return PY1 - (v - ylo) / (yhi - ylo) * (PY1 - PY0)

    for v in ticks:
        svg.line(PX0, py(v), PX1, py(v), color="grid", width=1)
        svg.text(PX0 - 6, py(v) + 4, "%g" % v, size=11, anchor="end", color="muted")
    for i, k in enumerate(Ks):
        svg.line(px(i), PY0, px(i), PY1, color="grid", width=1)
        svg.text(px(i), PY1 + 16, "%d" % k, size=11, anchor="middle", color="muted")
    svg.line(PX0, PY1, PX1, PY1, color="muted", width=1.1)
    svg.line(PX0, PY0, PX0, PY1, color="muted", width=1.1)
    svg.text(x0 + 10, 26, "%s（%%，温度 0.7）" % bench, size=13, weight="700")
    svg.text((PX0 + PX1) / 2, PY1 + 36, "K（采样个数，按原图等距排列）", size=12, anchor="middle", color="muted")
    d = data[bench]
    for (kind, model), vals in d.items():
        polyline(svg, [(px(i), py(v)) for i, v in enumerate(vals)], color=COL[model],
                 width=2.6 if model == "rl" else 2.0, dash=DASH[kind])
        for i, v in enumerate(vals):
            svg.circle(px(i), py(v), 2.6, fill=COL[model])
    return px, py, d


def crossing(d):
    """Linear interpolation of where Pass@K(RL) - Pass@K(Instruct) changes sign (index units)."""
    a, b = d[("pass", "rl")], d[("pass", "inst")]
    for i in range(1, len(a) - 1):
        d0, d1 = a[i] - b[i], a[i + 1] - b[i + 1]
        if d0 > 0 >= d1:
            f = d0 / (d0 - d1)
            print("Pass@K crossing between K=%d and K=%d (fraction %.2f), value %.1f"
                  % (Ks[i], Ks[i + 1], f, b[i] + f * (b[i + 1] - b[i])))
            return i + f, b[i] + f * (b[i + 1] - b[i])


px, py, d = panel(10, "GSM8K", 80, 100, (80, 85, 90, 95, 100))
svg.text(px(5) + 4, py(99.1) - 6, "Pass 99.1", size=11.5, anchor="end", color="muted", halo=True)
svg.text(px(5) + 4, py(97.3) + 15, "97.3", size=11.5, anchor="end", color="blue", halo=True)
svg.text(px(5) + 4, py(91.0) - 6, "Maj 91.0", size=11.5, anchor="end", color="blue", halo=True)
svg.text(px(5) + 4, py(89.5) + 15, "89.5", size=11.5, anchor="end", color="muted", halo=True)
xi, yi = crossing(d)
svg.circle(px(xi), py(yi), 9, stroke="red", width=1.5)
svg.text(px(xi) + 12, py(yi) + 22, "Pass 曲线在此交叉", size=11.5, color="red", halo=True)

px, py, d = panel(380, "MATH", 40, 90, (40, 50, 60, 70, 80, 90))
svg.text(px(5) + 4, py(87.5) - 6, "Pass 87.5", size=11.5, anchor="end", color="muted", halo=True)
svg.text(px(5) + 4, py(86.8) + 15, "86.8", size=11.5, anchor="end", color="blue", halo=True)
svg.text(px(5) + 4, py(60.9) - 6, "Maj 60.9", size=11.5, anchor="end", color="blue", halo=True)
svg.text(px(5) + 4, py(59.9) + 15, "59.9", size=11.5, anchor="end", color="muted", halo=True)
xi, yi = crossing(d)
svg.circle(px(xi), py(yi), 9, stroke="red", width=1.5)
svg.text(px(xi) - 12, py(yi) + 4, "交叉", size=11.5, anchor="end", color="red", halo=True)

ly = H - 12
items = [("maj", "inst", "Maj@K · Instruct"), ("maj", "rl", "Maj@K · RL"),
         ("pass", "inst", "Pass@K · Instruct"), ("pass", "rl", "Pass@K · RL")]
x = 16
for kind, model, lab in items:
    svg.line(x, ly - 4, x + 26, ly - 4, color=COL[model], width=2.4, dash=DASH[kind])
    svg.text(x + 32, ly, lab, size=12, color="ink")
    x += 176
svg.save("l08-majk-passk.svg")
