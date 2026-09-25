"""Figure 11.8: balance scope -- sequence-wise vs batch-wise (schematic + the V3 sec. 4.5.3 numbers).

Left (SCHEMATIC, illustrative numbers generated here, not paper data): 3 sequences from 3 domains x 8
experts. A sequence-wise constraint forces every row to be flat; a batch-wise constraint only needs the
column sums to be flat, so each domain may concentrate on "its" experts.
Right (paper data, arXiv:2412.19437 sec. 4.5.3): validation loss of MoE models trained with a
sequence-wise auxiliary loss, the auxiliary-loss-free method, and a batch-wise auxiliary loss:
1B: 2.258 / 2.253 / 2.253; 3B: 2.085 / 2.080 / 2.080.
Writes deepseek/figures/l11-batch-vs-seq.svg
"""

import numpy as np

from l11_svg import SVG

W, H = 720, 400
svg = SVG(W, H, "均衡的范围：序列级约束与 batch 级约束")

# ---------------------------------------------------------------- schematic heat grids
doms = ["代码", "数学", "百科"]
flat = np.full((3, 8), 1.0)
spec = np.array([
    [2.4, 2.2, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6],
    [0.4, 0.5, 2.3, 2.2, 0.6, 0.6, 0.7, 0.7],
    [0.2, 0.3, 0.2, 0.3, 1.8, 1.8, 1.7, 1.7],
])
assert np.allclose(spec.sum(0), 3.0) and np.allclose(spec.sum(1), 8.0)   # columns flat, rows = 8 each
CELL = 24


def grid(x0, y0, M, title, sub):
    svg.text(x0, y0 - 30, title, size=13, weight="600")
    svg.text(x0, y0 - 12, sub, size=11, color="muted")
    for r in range(3):
        svg.text(x0 - 8, y0 + r * CELL + 16, doms[r], size=12, anchor="end")
        for c in range(8):
            v = M[r, c]
            svg.rect(x0 + c * CELL, y0 + r * CELL, CELL - 2, CELL - 2, fill="blue", fill_opacity=min(0.9, 0.08 + 0.3 * v),
                     rx=2)
    for c in range(8):
        svg.text(x0 + c * CELL + CELL / 2 - 1, y0 + 3 * CELL + 14, str(c + 1), size=11, anchor="middle", color="muted")
    svg.text(x0 + 4 * CELL, y0 + 3 * CELL + 30, "专家编号", size=11, anchor="middle", color="muted")


grid(66, 70, flat, "Ⓐ 序列级约束", "每一行（每个序列）都要摊平")
grid(66, 250, spec, "Ⓑ batch 级约束", "只要求各列之和（整个 batch）摊平")
svg.text(66 + 8 * CELL + 10, 250 + 1.5 * CELL + 4, "各列之和", size=11, color="green")
svg.text(66 + 8 * CELL + 10, 250 + 1.5 * CELL + 20, "≈ 相等", size=11, color="green")
svg.text(16, 392, "左侧两格为示意（颜色深 = 该领域的 token 更多地去这个专家），不是论文数据。", size=11, color="muted")

svg.line(330, 30, 330, 370, color="rule", width=1)

# ---------------------------------------------------------------- bars (paper numbers)
data = {"1B MoE": [2.258, 2.253, 2.253], "3B MoE": [2.085, 2.080, 2.080]}
labs = ["序列级辅助损失", "无辅助损失（偏置）", "batch 级辅助损失"]
cols = ["muted", "green", "violet"]
svg.text(350, 40, "Ⓒ 验证损失（V3 §4.5.3，越低越好）", size=13, weight="600")
svg.text(350, 58, "两个纵轴都不从 0 开始；差值只有 0.005", size=11, color="muted")
px0 = 400
for k, (name, vals) in enumerate(data.items()):
    y0, y1 = 96 + k * 138, 96 + k * 138 + 100
    lo = round(min(vals) - 0.004, 3)
    hi = round(max(vals) + 0.002, 3)

    def py(v):
        return y1 - (v - lo) / (hi - lo) * (y1 - y0)

    svg.text(px0 + 290, y0 + 20, name, size=12, weight="600", anchor="end")
    for tv in np.arange(lo, hi + 1e-9, 0.002):
        svg.line(px0, py(tv), px0 + 280, py(tv), color="grid", width=1)
        svg.text(px0 - 6, py(tv) + 4, "%.3f" % tv, size=11, anchor="end", color="muted")
    for j, v in enumerate(vals):
        x = px0 + 20 + j * 90
        svg.rect(x, py(v), 56, y1 - py(v), fill=cols[j], fill_opacity=0.6, stroke=cols[j], width=1)
        svg.text(x + 28, py(v) - 5, "%.3f" % v, size=12, anchor="middle", color="ink", weight="600")
    svg.line(px0, y1, px0 + 280, y1, color="ink", width=1.2)
for j, lab in enumerate(labs):
    x = px0 + 20 + j * 90
    svg.text(x + 28, 348, lab.split("（")[0] if j != 1 else "无辅助损失", size=11, anchor="middle", color=cols[j] if cols[j] != "muted" else "ink")
svg.text(px0 + 20 + 90 + 28, 364, "（偏置）", size=11, anchor="middle", color="green")
svg.text(px0 + 20 + 28, 364, "（每个序列）", size=11, anchor="middle")
svg.text(px0 + 20 + 180 + 28, 364, "（整个 batch）", size=11, anchor="middle", color="violet")

svg.save("l11-batch-vs-seq.svg")
