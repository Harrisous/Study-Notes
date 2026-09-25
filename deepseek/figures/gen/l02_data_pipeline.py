"""Figure 2.5: GPT-3's data pipeline (Brown et al. 2020, section 2.2 and appendices A, C) plus the
retention curve of its Pareto-based quality filter.

Appendix A: a logistic-regression classifier scores each Common Crawl document (s in [0, 1]); the
document is kept if np.random.pareto(alpha) > 1 - s with alpha = 9. numpy's pareto() draws from the
Lomax (Pareto II) distribution, P(X > x) = (1 + x)^(-alpha), so P(keep | s) = (2 - s)^(-9).
Values recomputed in lesson-02-doc.md section 4.4 (and checked there by 2 million simulations):
  s = 1: 1.000, 0.9: 0.424, 0.5: 0.0260, 0.1: 0.00310, 0: 0.00195
Writes deepseek/figures/l02-data-pipeline.svg
"""

import numpy as np

from l02_svg import SVG, box, LogAxis

W, H = 720, 520
svg = SVG(W, H, "GPT-3 的数据管线与质量过滤的保留概率")

BW, GAP, X0 = 163, 16, 10
BH = 88
xs = [X0 + i * (BW + GAP) for i in range(4)]
Y1, Y2 = 30, 150
row1 = [
    ("Common Crawl", ["41 个月度分片", "2016–2019", "45 TB（压缩）"], "amber"),
    ("质量分类器", ["逻辑回归，打分 s ∈ [0, 1]", "正例：WebText 等", "负例：未过滤的 CC"], "violet"),
    ("Pareto 重采样", ["X ~ Pareto(α = 9)", "X > 1 − s 就保留", "（见下方曲线）"], "violet"),
    ("MinHashLSH 去重", ["10 个哈希，模糊匹配", "数据集内 + CC 去掉 WebText", "总量平均减少 10%"], "red"),
]
row2 = [
    ("CC 过滤后 570 GB", ["约 400B 个 BPE token", "（论文 §2.2）"], "amber"),
    ("混入精选语料", ["WebText2、Books1/2、", "英文维基", "按权重采样（图 2.6）"], "green"),
    ("13-gram 去污染", ["删去与评测集重叠的文档", "有 bug：部分重叠漏网", "因成本未重训"], "red"),
    ("训练 300B token", ["CC 只看了 0.44 遍", "维基看了 3.4 遍"], "blue"),
]
for i, (t, ls, col) in enumerate(row1):
    box(svg, xs[i], Y1, BW, BH, t, ls, color=col, fill_opacity=0.08, line_size=12)
    if i:
        svg.arrow(xs[i - 1] + BW + 1, Y1 + BH / 2, xs[i] - 2, Y1 + BH / 2, color="ink", width=1.4)
for i, (t, ls, col) in enumerate(row2):
    box(svg, xs[i], Y2, BW, BH, t, ls, color=col, fill_opacity=0.08, line_size=12)
    if i:
        svg.arrow(xs[i - 1] + BW + 1, Y2 + BH / 2, xs[i] - 2, Y2 + BH / 2, color="ink", width=1.4)
# snake connector from the end of row 1 to the start of row 2
ym = (Y1 + BH + Y2) / 2
svg.path("M%.1f %.1f V%.1f H%.1f V%.1f" % (xs[3] + BW / 2, Y1 + BH + 1, ym, xs[0] + BW / 2, Y2 - 2),
         stroke="ink", width=1.4, arrow=True)

# ---------------- retention curve
PX0, PX1, PY0, PY1 = 90, 420, 300, 478
ax = LogAxis(0, 1, PX0, PX1, log=False)
ay = LogAxis(1e-4, 1, PY1, PY0, log=True)
svg.text(PX0 - 60, 272, "质量过滤：分类器分数 s 与文档被保留的概率 P = (2 − s)^{−9}", size=13, color="violet",
         weight="600")
for e, lab in ((-4, "0.01%"), (-3, "0.1%"), (-2, "1%"), (-1, "10%"), (0, "100%")):
    y = ay(10 ** e)
    svg.line(PX0, y, PX1, y, color="grid", width=1)
    svg.text(PX0 - 8, y + 4, lab, size=11, anchor="end", color="muted")
for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
    x = ax(v)
    svg.line(x, PY0, x, PY1, color="grid", width=1)
    svg.text(x, PY1 + 16, "%.1f" % v, size=11, anchor="middle", color="muted")
svg.line(PX0, PY1, PX1, PY1, color="muted", width=1.2)
svg.line(PX0, PY0, PX0, PY1, color="muted", width=1.2)
svg.text((PX0 + PX1) / 2, PY1 + 36, "分类器给文档的分数 s（越像精选语料越接近 1）", size=12, anchor="middle", color="muted")
svg.text(PX0 - 52, (PY0 + PY1) / 2, "保留概率（对数）", size=12, anchor="middle", color="muted", rotate=-90)

s = np.linspace(0, 1, 120)
P = (2 - s) ** -9.0
svg.path("M" + " L".join("%.1f %.1f" % (ax(a), ay(b)) for a, b in zip(s, P)), stroke="violet", width=2.6)
for sv, lab, dx, dy, anc in ((0.0, "s = 0：0.2%", 10, 18, "start"), (0.5, "s = 0.5：2.6%", 8, 18, "start"),
                              (0.9, "0.9：42%", 6, 20, "start"), (1.0, "s = 1：100%", -8, -6, "end")):
    x, y = ax(sv), ay((2 - sv) ** -9.0)
    svg.circle(x, y, 4, fill="violet")
    svg.text(x + dx, y + dy, lab, size=12, anchor=anc, color="ink")

# explanation on the right
TX = 482
lines = [
    ("和“设一个阈值”有什么不同？", "ink", "600"),
    ("阈值把低于它的文档全部删掉；", "muted", None),
    ("按这条曲线抽样，高分文档几乎全留，", "muted", None),
    ("低分文档仍有很小的概率留下，", "muted", None),
    ("而不是一刀切（由公式算出的效果）。", "muted", None),
    ("", "muted", None),
    ("numpy 的 pareto(α) 是 Lomax 分布：", "ink", "600"),
    ("P(X > x) = (1 + x)^{−α}，", "muted", None),
    ("所以 P(X > 1 − s) = (2 − s)^{−α}。", "muted", None),
]
for i, (t, col, wt) in enumerate(lines):
    svg.text(TX, 312 + i * 19, t, size=12, color=col, weight=wt)

svg.save("l02-data-pipeline.svg")
