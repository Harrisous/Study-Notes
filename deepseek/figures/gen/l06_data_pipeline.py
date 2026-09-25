"""Figure 6.3: the three-stage data pipeline (top) and the deduplication rate versus the number of
Common Crawl dumps deduplicated together (bottom, DeepSeek LLM Table 1, log x-axis).

Table 1 (dsllm §2.1): dumps 1, 2, 6, 12, 16, 22, 41, 91 -> 22.2, 46.7, 55.7, 69.9, 75.7, 76.3, 81.6, 89.8 %.
Writes deepseek/figures/l06-data-pipeline.svg
"""

import math

from l06_svg import SVG, box

dumps = [1, 2, 6, 12, 16, 22, 41, 91]
rate = [22.2, 46.7, 55.7, 69.9, 75.7, 76.3, 81.6, 89.8]

W, H = 760, 520
svg = SVG(W, H, "DeepSeek LLM 的三阶段数据管线与跨 dump 去重率")

# ---- top: pipeline
ty, th = 36, 112
bw, gap = 190, 34
x0 = 20
stages = [
    (["① 去重 deduplication", "目标：同一内容只留一份", "跨全部 91 个 dump 一起做", "（README：MinHash LSH）"], "red"),
    (["② 过滤 filtering", "目标：提高信息密度", "语言学 + 语义两类评估", "（阈值与模型未公开）"], "ink"),
    (["③ 重混 remixing", "目标：纠正领域失衡", "提高少数领域的占比", "（配比未公开）"], "ink"),
]
for i, (lines, col) in enumerate(stages):
    x = x0 + i * (bw + gap)
    box(svg, x, ty, bw, th, lines, color=col, fill_opacity=0.07, size=12, line_gap=19)
    svg.arrow(x + bw + 3, ty + th / 2, x + bw + gap - 3, ty + th / 2, color="ink", width=1.6)
fx = x0 + 3 * (bw + gap)
box(svg, fx, ty + 14, W - 20 - fx, th - 28, ["2T", "tokens"], color="green", fill_opacity=0.10, size=13)
svg.text(x0, ty - 14, "论文 §2.1：去重与重混保证多样性，过滤提高信息密度", size=12, color="muted")

# ---- bottom: dedup rate chart
X0, X1, Y0, Y1 = 92, 600, 206, 470


def px(n):
    return X0 + math.log10(n) / math.log10(100) * (X1 - X0)


def py(r):
    return Y1 - r / 100 * (Y1 - Y0)


for r in range(0, 101, 20):
    y = py(r)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "%d%%" % r, size=11, anchor="end", color="muted")
for n in (1, 2, 5, 10, 20, 50, 100):
    x = px(n)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, str(n), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "一起去重的 Common Crawl dump 个数（对数刻度）", size=12, anchor="middle", color="muted")
svg.text(28, (Y0 + Y1) / 2, "去重率（被删文档占比）", size=12, anchor="middle", color="muted", rotate=-90)

# removed share as red area under the curve, kept share implied above
d = "M%.1f %.1f " % (px(dumps[0]), py(0))
d += " ".join("L%.1f %.1f" % (px(n), py(r)) for n, r in zip(dumps, rate))
d += " L%.1f %.1f Z" % (px(dumps[-1]), py(0))
svg.path(d, stroke=None, fill="red", fill_opacity=0.08)
svg.path("M" + " L".join("%.1f %.1f" % (px(n), py(r)) for n, r in zip(dumps, rate)), stroke="red", width=2.2)
for n, r in zip(dumps, rate):
    svg.circle(px(n), py(r), 4, fill="red")
    dy = -10 if n not in (1, 16, 22) else (-10 if n == 16 else (18 if n == 22 else 4))
    dx = 0 if n not in (1, 16, 22) else (-6 if n == 16 else 8)
    anchor = "middle" if n not in (1, 16, 22) else ("end" if n == 16 else "start")
    svg.text(px(n) + dx, py(r) + dy, "%.1f" % r, size=11, anchor=anchor, color="red")

# annotations on the right
ax = 620
svg.text(ax, 250, "单个 dump 内去重：", size=12, color="ink")
svg.text(ax, 268, "删 22.2%，留 77.8%", size=12, color="ink", weight="600")
svg.text(ax, 306, "91 个 dump 一起去重：", size=12, color="ink")
svg.text(ax, 324, "删 89.8%，留 10.2%", size=12, color="ink", weight="600")
svg.text(ax, 362, "删掉的文档数是", size=12, color="muted")
svg.text(ax, 380, "单 dump 的 89.8/22.2", size=12, color="muted")
svg.text(ax, 398, "≈ 4.05 倍", size=12, color="muted")

svg.save("l06-data-pipeline.svg")
