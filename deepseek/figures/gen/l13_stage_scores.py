"""Figure 13.12: benchmark scores of the intermediate checkpoints R1-Zero -> Dev1 -> Dev2 -> Dev3 -> R1
(numbers from R1 paper arXiv v2 Table 3, as recorded in reference-docs/lesson-13-doc.md section 3.15).
Writes deepseek/figures/l13-stage-scores.svg
"""

from l13_svg import SVG, Axes

STAGES = ["R1-Zero", "Dev1", "Dev2", "Dev3", "R1"]
REASON = [("AIME 2024", "blue", [77.9, 59.0, 74.0, 78.1, 79.8]),
          ("Codeforces 百分位", "violet", [80.4, 84.5, 90.5, 92.1, 96.3]),
          ("LiveCodeBench", "green", [50.0, 57.5, 63.5, 64.6, 65.9])]
GENERAL = [("ArenaHard", "orange", [53.6, 77.0, 73.2, 75.6, 92.3]),
           ("IF-Eval", "amber", [46.6, 71.7, 72.0, 78.1, 83.3]),
           ("AlpacaEval 2.0", "green", [24.7, 50.1, 55.8, 62.1, 87.6])]

START_DY = {"AIME 2024": 4.5, "Codeforces 百分位": -8, "LiveCodeBench": 17,
            "ArenaHard": None, "IF-Eval": 17, "AlpacaEval 2.0": None}

W, H = 760, 360
svg = SVG(W, H, "R1 流水线各检查点的成绩：推理类与通用偏好类基准")


def panel(x0, x1, title, series, label_dy):
    ax = Axes(x0, x1, 44, 300, (0, 4), (20, 100))
    svg.text(x0 - 36, 22, title, size=13, weight="600")
    for v in (20, 40, 60, 80, 100):
        y = ax.py(v)
        svg.line(ax.X0, y, ax.X1, y, color="grid", width=1)
        svg.text(ax.X0 - 6, y + 4, str(v), size=11, anchor="end", color="muted")
    for i, s in enumerate(STAGES):
        x = ax.px(i)
        svg.line(x, ax.Y0, x, ax.Y1, color="grid", width=1)
        svg.text(x, ax.Y1 + 18, s, size=12, anchor="middle", color="ink", weight="600" if s == "R1" else None)
    # Dev1 band: cold-start SFT
    for name, col, vals in series:
        svg.polyline([(ax.px(i), ax.py(v)) for i, v in enumerate(vals)], stroke=col, width=2.2)
        for i, v in enumerate(vals):
            svg.circle(ax.px(i), ax.py(v), 3.2, fill=col)
        if START_DY[name] in (None, "L"):
            dy = 4 if START_DY[name] is None else 10
            svg.text(ax.px(0) - 8, ax.py(vals[0]) + dy, "%.1f" % vals[0], size=11, anchor="end", color=col)
        else:
            svg.text(ax.px(0) + 7, ax.py(vals[0]) + START_DY[name], "%.1f" % vals[0], size=11, color=col)
        svg.text(ax.px(4) + 7, ax.py(vals[4]) + 4 + label_dy.get(name, 0), "%.1f" % vals[4], size=11, color=col,
                 weight="600")
    return ax


a = panel(60, 320, "(a) 推理类（%）", REASON, {"AIME 2024": 4, "Codeforces 百分位": -4})
b = panel(448, 708, "(b) 指令遵循与通用偏好类（%）", GENERAL, {"ArenaHard": -2, "IF-Eval": 4})
# legends under each panel
for ax, series in ((a, REASON), (b, GENERAL)):
    x = ax.X0 - 30
    for name, col, _ in series:
        svg.line(x, 346, x + 18, 346, color=col, width=2.4)
        svg.text(x + 22, 350, name, size=12, color="ink")
        x += 30 + len(name) * 7.8
# annotation: Dev1 drop on AIME
svg.text(a.px(1) + 6, a.py(59.0) + 18, "Dev1：59.0", size=12, color="blue", weight="600")
svg.text(b.px(4) - 8, b.py(87.6) - 10, "+25.5 点", size=12, anchor="end", color="green", weight="600")
svg.save("l13-stage-scores.svg")
