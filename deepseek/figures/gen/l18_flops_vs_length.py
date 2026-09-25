"""Figure 18.14: per-token FLOPs vs position (redraw of the upper-right panel of Fig. 1 of the V4
report) and the breakdown at t = 2^20. Model: eq. (18.2) / (18.26) with "equivalent FP8" weights
BF16 x2, FP8 x1, FP4 x0.5 -- the research notes' inference from Fig. 1, not stated by the paper.
Writes deepseek/figures/l18-flops-vs-length.svg
"""

from l18_svg import SVG
from l18_data import K, M1, flops_v32, flops_v4, total

TKS = [k / 4 for k in range(0, 4 * 1024 + 1, 2)]          # positions in K, fine grid
curves = {
    "V3.2": ([total(flops_v32(k * K)) / 1e12 for k in TKS], "amber"),
    "V4-Pro": ([total(flops_v4(k * K, "pro")) / 1e12 for k in TKS], "orange"),
    "V4-Flash": ([total(flops_v4(k * K, "flash")) / 1e12 for k in TKS], "red"),
}
assert round(curves["V3.2"][0][-1], 4) == 1.1916 and round(curves["V4-Pro"][0][-1], 4) == 0.3158

W, H = 760, 452
svg = SVG(W, H, "单 token FLOPs 随位置的变化与 1M 处的构成")

# ---------------------------------------------------------------- left: curves
X0, X1, Y0, Y1, YMAX = 64, 440, 40, 330, 1.3


def px(k):
    return X0 + k / 1024 * (X1 - X0)


def py(v):
    return Y1 - v / YMAX * (Y1 - Y0)


svg.text(20, 22, "(a) 单 token FLOPs（等效 FP8，TFLOPs）", size=13, weight="700")
for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
    svg.line(X0, py(v), X1, py(v), color="grid", width=1)
    svg.text(X0 - 8, py(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
for k in (0, 256, 512, 768, 1024):
    svg.line(px(k), Y0, px(k), Y1, color="grid", width=1)
    svg.text(px(k), Y1 + 18, "%dK" % k, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "查询位置 t（K = 1024）", size=12, anchor="middle", color="muted")
svg.line(px(128), Y0, px(128), Y1, color="red", width=1, dash="3 3")
svg.text(px(128) + 4, Y0 + 12, "128K", size=11, color="red")
LAB = {"V3.2": (-6, "V3.2：1.19"), "V4-Pro": (-8, "Pro：0.316"), "V4-Flash": (14, "Flash：0.118")}
for name, (vals, col) in curves.items():
    pts = " L".join("%.1f %.1f" % (px(k), py(v)) for k, v in zip(TKS, vals))
    svg.path("M" + pts, stroke=col, width=2.4)
    dy, lab = LAB[name]
    svg.text(px(1024) - 4, py(vals[-1]) + dy, lab, size=12, anchor="end", color=col, weight="700")
# ratio notes
svg.text(px(150), py(1.12), "1M 处：Pro 为 V3.2 的 26.5%，Flash 为 9.9%", size=11)
svg.text(px(150), py(1.12) + 16, "128K 处：Pro 为 52.0%，Flash 为 15.0%", size=11)

# ---------------------------------------------------------------- right: stacked breakdown at 1M
BX0, BY0, BY1, BMAX = 520, 40, 330, 1.3


def by(v):
    return BY1 - v / BMAX * (BY1 - BY0)


svg.text(480, 22, "(b) t = 2^{20} 处的构成（TFLOPs）", size=13, weight="700")
parts = [
    ("V3.2", flops_v32(M1), [("proj", "muted", "投影"), ("core", "blue", "核心注意力"), ("idx", "violet", "indexer")]),
    ("Pro", flops_v4(M1, "pro"), [("proj", "muted", ""), ("csa", "blue", ""), ("hca", "orange", ""), ("swa", "green", ""),
                                   ("idx", "violet", "")]),
    ("Flash", flops_v4(M1, "flash"), [("proj", "muted", ""), ("csa", "blue", ""), ("hca", "orange", ""),
                                       ("swa", "green", ""), ("idx", "violet", "")]),
]
for v in (0, 0.4, 0.8, 1.2):
    svg.line(BX0 - 10, by(v), 740, by(v), color="grid", width=1)
    svg.text(BX0 - 14, by(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
svg.line(BX0 - 10, BY1, 740, BY1, color="muted", width=1.2)
for n, (name, d, segs) in enumerate(parts):
    x = BX0 + n * 72
    y = BY1
    for key, col, _ in segs:
        h = d[key] / 1e12 / BMAX * (BY1 - BY0)
        if h > 0.2:
            svg.rect(x, y - h, 40, h, fill=col, fill_opacity=0.75, stroke=col, width=0.6)
        y -= h
    svg.text(x + 20, y - 6, "%.3f" % (total(d) / 1e12), size=11, anchor="middle", weight="700")
    svg.text(x + 20, BY1 + 18, name, size=12, anchor="middle")
# legend under the bars
leg = [("muted", "投影与专家 2N"), ("blue", "CSA / DSA 核心注意力"), ("orange", "HCA 稠密注意力"),
       ("violet", "indexer 打分"), ("green", "纯窗口层（极小）")]
for n, (col, lab) in enumerate(leg):
    yy = 364 + (n // 2) * 18
    xx = 470 + (n % 2) * 140
    svg.rect(xx, yy - 10, 12, 12, fill=col, fill_opacity=0.75, stroke=None)
    svg.text(xx + 18, yy, lab, size=11)
pro = flops_v4(M1, "pro")
svg.text(20, 428, "Pro 在 1M 处：投影 0.098 + CSA 0.018 + HCA %.3f + indexer %.3f（等效 FP8）。" % (
    pro["hca"] / 1e12, pro["idx"] / 1e12), size=11, color="muted")
svg.text(20, 446, "折算权重 BF16×2、FP8×1、FP4×0.5 是研究笔记从论文图 1 反推的，论文没写；这是按公开架构的估算，不是实测。", size=11,
         color="muted")

svg.save("l18-flops-vs-length.svg")
