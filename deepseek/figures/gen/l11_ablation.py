"""Figure 11.15: V3's two ablation tables as differences (computed from the raw cells).

Table 5 (arXiv:2412.19437 p.27): aux-loss-free minus aux-loss-based; small MoE 15.7B/2.4B act/1.33T tokens,
large MoE 228.7B/20.9B act/578B tokens.
Table 4 (p.26): with MTP minus baseline; small 15.7B/1.33T, large 228.7B/540B tokens. MTP is discarded at
inference, so inference cost is identical; training cost is not (research notes E19).
The two tables use different baselines (E21) -- never add them. Pile-test BPB (lower is better) is listed
as text, not plotted, because its scale differs.
Writes deepseek/figures/l11-ablation.svg
"""

from l11_svg import SVG

bench = ["BBH", "MMLU", "DROP", "TriviaQA", "NaturalQuestions", "HumanEval", "MBPP", "GSM8K", "MATH"]
# raw cells: (small_a, small_b, large_a, large_b) with a = baseline, b = new method
t5 = {"BPB": (0.727, 0.724, 0.656, 0.652), "BBH": (37.3, 39.3, 66.7, 67.9), "MMLU": (51.0, 51.8, 68.3, 67.2),
      "DROP": (38.1, 39.0, 67.1, 67.1), "TriviaQA": (58.3, 58.5, 66.7, 67.7), "NaturalQuestions": (23.2, 23.4, 27.1, 28.1),
      "HumanEval": (22.0, 22.6, 40.2, 46.3), "MBPP": (36.6, 35.8, 59.2, 61.2), "GSM8K": (27.1, 29.6, 70.7, 74.5),
      "MATH": (10.9, 11.1, 37.2, 39.6)}
t4 = {"BPB": (0.729, 0.729, 0.658, 0.657), "BBH": (39.0, 41.4, 70.0, 70.7), "MMLU": (50.0, 53.3, 67.5, 66.6),
      "DROP": (39.2, 41.3, 68.5, 70.6), "TriviaQA": (56.9, 57.7, 67.0, 67.3), "NaturalQuestions": (22.7, 22.3, 27.2, 28.5),
      "HumanEval": (20.7, 26.8, 44.5, 53.7), "MBPP": (35.8, 36.8, 61.6, 62.2), "GSM8K": (25.4, 31.4, 72.3, 74.0),
      "MATH": (10.7, 12.6, 38.6, 39.8)}


def diffs(t):
    return {k: (round(v[1] - v[0], 3), round(v[3] - v[2], 3)) for k, v in t.items()}


d5, d4 = diffs(t5), diffs(t4)
assert d5["HumanEval"] == (0.6, 6.1) and d5["MMLU"] == (0.8, -1.1) and d4["HumanEval"] == (6.1, 9.2)
for name, d in (("table5", d5), ("table4", d4)):
    small_better = sum(1 for b in bench if d[b][0] > 0)
    large_better = sum(1 for b in bench if d[b][1] > 0)
    print(name, {b: d[b] for b in bench}, "BPB", d["BPB"], "better small/large (9 benchmarks):", small_better, large_better)

W, H = 720, 470
svg = SVG(W, H, "V3 的两组消融：新方法减基线的差值")
LBL = 118
XMIN, XMAX = -2.0, 10.0


def panel(x0, x1, title, sub, d, bpb_note):
    def px(v):
        return x0 + (v - XMIN) / (XMAX - XMIN) * (x1 - x0)

    svg.text(x0 - LBL + 10, 24, title, size=13, weight="600")
    svg.text(x0 - LBL + 10, 42, sub, size=11, color="muted")
    y0 = 66
    rh = 34
    for v in (-2, 0, 2, 4, 6, 8, 10):
        svg.line(px(v), y0 - 6, px(v), y0 + rh * 9, color="grid" if v else "ink", width=1 if v else 1.3)
        svg.text(px(v), y0 + rh * 9 + 16, "%+d" % v if v else "0", size=11, anchor="middle", color="muted")
    for r, b in enumerate(bench):
        y = y0 + r * rh + rh / 2
        svg.text(x0 - 8, y + 4, b, size=12, anchor="end")
        s, l = d[b]
        svg.line(px(s), y, px(l), y, color="rule", width=2)
        svg.circle(px(s), y, 5, fill="surface", stroke="blue", width=1.6)
        svg.circle(px(l), y, 5, fill="blue", stroke="blue", width=1.2)
        lab_l = ("%+.1f" % l) if l else "0.0"
        svg.text(px(max(s, l)) + 9, y + 4, lab_l, size=11, color="red" if l < 0 else "blue")
    svg.text((x0 + x1) / 2, y0 + rh * 9 + 34, "差值（分数点；> 0 表示新方法更好）", size=12, anchor="middle")
    svg.text(x0 - LBL + 10, y0 + rh * 9 + 56, bpb_note, size=11, color="muted")


panel(150, 330, "Ⓐ 无辅助损失 − 纯辅助损失（表 5）", "小：15.7B / 1.33T tokens；大：228.7B / 578B", d5,
      "Pile BPB（越低越好）：小 −0.003，大 −0.004")
panel(500, 680, "Ⓑ 加 MTP − 基线（表 4）", "小：15.7B / 1.33T tokens；大：228.7B / 540B", d4,
      "Pile BPB：小 0.000，大 −0.001")
svg.circle(24, 458, 5, fill="surface", stroke="blue", width=1.6)
svg.text(34, 462, "小模型", size=12)
svg.circle(100, 458, 5, fill="blue", stroke="blue", width=1.2)
svg.text(110, 462, "大模型（数字标的是大模型的差值；负值为红色）", size=12)

svg.save("l11-ablation.svg")
