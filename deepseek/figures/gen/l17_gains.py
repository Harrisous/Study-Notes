"""Figure 17.9: what Engram buys, and what is lost when it is switched off.

Left: Engram-27B minus MoE-27B on all 27 benchmarks of Engram table 1 (notes §4.1; every difference
is positive).  Right: share of performance retained when the Engram output is zeroed at inference
(Engram fig. 6; the percentages are the labels printed on the figure, whose pairing with the
benchmarks was confirmed from the vector file, notes §4.9). Paper text: factual 29-44 %,
reading comprehension 81-93 %.
Writes deepseek/figures/l17-gains.svg
"""

from l17_svg import SVG

DELTA = {
    "MMLU": 3.0, "MMLU-Redux": 3.4, "MMLU-Pro": 1.8, "CMMLU": 4.0, "C-Eval": 4.7, "AGIEval": 3.2,
    "ARC-Easy": 2.5, "ARC-Challenge": 3.7, "TriviaQA": 1.9, "TriviaQA-ZH": 1.5, "PopQA": 0.2, "CCPM": 7.5,
    "BBH": 5.0, "HellaSwag": 0.9, "PIQA": 1.6, "WinoGrande": 0.2, "DROP": 3.3, "RACE-M": 1.9, "RACE-H": 2.8,
    "C3": 3.5, "HumanEval": 3.0, "MBPP": 1.6, "CruxEval-i": 1.5, "CruxEval-o": 0.9, "GSM8K": 2.2,
    "MGSM": 2.6, "MATH": 2.4,
}
assert len(DELTA) == 27 and min(DELTA.values()) > 0
RETAIN = [
    ("阅读理解", "green", [("C3", 93), ("RACE-Middle", 89), ("RACE-High", 84), ("DROP", 81)]),
    ("常识推理", "muted", [("HellaSwag", 85), ("ARC-Challenge", 81), ("PIQA", 81)]),
    ("知识密集推理", "muted", [("CMMLU", 78), ("MMLU", 75), ("MMLU-Pro", 72)]),
    ("代码", "muted", [("CruxEval", 76), ("MBPP", 68), ("HumanEval", 58)]),
    ("算法推理", "muted", [("BBH", 67), ("GSM8K", 62), ("MGSM", 44), ("MATH", 36)]),
    ("事实知识", "red", [("TriviaQA-ZH", 44), ("PopQA", 44), ("TriviaQA", 29)]),
]
assert sum(len(g[2]) for g in RETAIN) == 20

W, H = 760, 512
svg = SVG(W, H, "Engram-27B 相对 MoE-27B 的提升，以及推理时关掉 Engram 后各基准保留的性能")

# ---------------------------------------------------------------- left
svg.text(20, 22, "(a) Engram-27B − MoE-27B（分）", size=13, weight="700")
svg.text(20, 40, "表 1 全部 27 项，按提升排序", size=11.5, color="muted")
LX0, SC, RH, TOP = 118, 27, 15.2, 58
items = sorted(DELTA.items(), key=lambda kv: -kv[1])
for v in (0, 2, 4, 6, 8):
    svg.line(LX0 + v * SC, TOP - 6, LX0 + v * SC, TOP + 27 * RH, color="grid", width=1)
    svg.text(LX0 + v * SC, TOP + 27 * RH + 14, "+%d" % v if v else "0", size=11, anchor="middle", color="muted")
HL = {"CCPM", "BBH", "C-Eval", "CMMLU", "ARC-Challenge", "PopQA", "WinoGrande", "TriviaQA", "MMLU"}
for i, (name, v) in enumerate(items):
    yy = TOP + i * RH
    svg.rect(LX0, yy, v * SC, RH - 3.5, fill="green", fill_opacity=0.55 if name in HL else 0.28)
    svg.text(LX0 - 6, yy + 10.5, name, size=11.5, anchor="end", color="ink" if name in HL else "muted")
    svg.text(LX0 + v * SC + 5, yy + 10.5, "+%.1f" % v, size=11, color="green" if name in HL else "muted")

# ---------------------------------------------------------------- right
RX0 = 416
svg.text(RX0 - 20, 22, "(b) 推理时把 Engram 输出置零，性能保留的比例", size=13, weight="700")
svg.text(RX0 - 20, 40, "据论文图 6 的标注；横线 = 100%", size=11.5, color="muted")
BX0, BSC = RX0 + 92, 2.0
yy = TOP - 2
rows = []
for gname, col, members in RETAIN:
    svg.text(RX0 - 20, yy + 10, gname, size=11.5, color=col if col != "muted" else "ink", weight="700")
    yy += 14.5
    for name, pct in members:
        svg.rect(BX0, yy, 100 * BSC, RH - 4, fill="grid", fill_opacity=0.9)
        svg.rect(BX0, yy, pct * BSC, RH - 4, fill=col, fill_opacity=0.6 if col != "muted" else 0.45)
        svg.text(BX0 - 6, yy + 10, name, size=11.5, anchor="end", color="muted")
        svg.text(BX0 + 100 * BSC + 30, yy + 10, "%d%%" % pct, size=11, anchor="end", color=col if col != "muted" else "ink")
        yy += RH
    yy += 2
svg.line(BX0 + 100 * BSC, TOP - 6, BX0 + 100 * BSC, yy, color="ink", width=1, dash="4 3")
svg.line(BX0, TOP - 6, BX0, yy, color="muted", width=1)

svg.text(20, 505, "注意：(b) 是训练后硬关掉一个一直在用的模块（训练-推理不一致），不能读成“知识有多少比例存在 Engram 里”。",
         size=11.5, color="muted")
svg.save("l17-gains.svg")
