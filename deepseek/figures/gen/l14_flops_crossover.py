"""Figure 14.13: per-query attention FLOPs per layer vs position T, dense vs DSA (DeepSeek-V3 dims).

Model (lesson-14 notes, section 2.3; one multiply-add = 2 FLOPs, only QK^T and PV):
  dense, MHA mode  F = 81,920 T         dense, MQA mode  F = 278,528 T
  DSA core (MQA)   F = 278,528 min(T, 2048)
  indexer          F = 16,384 T         DSA total = core + indexer
Crossovers: 2,176 (DSA total = dense MQA), 6,963 (DSA core = dense MHA),
8,704 (DSA total = dense MHA), 34,816 (indexer = DSA core). All asserted below.
Writes deepseek/figures/l14-flops-crossover.svg
"""

import math

import numpy as np

from l14_svg import SVG

K = 2048
MHA, MQA, IDX = 81920, 278528, 16384


def core(T):
    return MQA * np.minimum(T, K)


def total(T):
    return core(T) + IDX * T


X_MQA = MQA * K / (MQA - IDX)
X_CORE = MQA * K / MHA
X_TOT = MQA * K / (MHA - IDX)
X_IDX = MQA * K / IDX
assert (round(X_MQA), round(X_CORE), round(X_TOT), round(X_IDX)) == (2176, 6963, 8704, 34816)

Ts = np.logspace(math.log10(512), math.log10(131072), 300)
Ts = np.unique(np.concatenate([Ts, [K]]))

W, H = 760, 460
X0, X1, Y0, Y1 = 90, 590, 30, 360
LX0, LX1 = math.log2(512), math.log2(131072)
LY0, LY1 = 7, 11           # 10^7 .. 10^11 FLOPs


def px(T):
    return X0 + (math.log2(T) - LX0) / (LX1 - LX0) * (X1 - X0)


def py(F):
    return Y1 - (math.log10(F) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "每层、每个查询的注意力 FLOPs：稠密与 DSA（V3 维度，双对数坐标）")
for e in range(LY0, LY1 + 1):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
for T, lab in ((512, "512"), (2048, "2K"), (8192, "8K"), (32768, "32K"), (131072, "128K")):
    x = px(T)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, lab, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "位置 T（对数刻度，1K = 1024）", size=12, anchor="middle", color="muted")
svg.text(26, (Y0 + Y1) / 2, "每层、每个查询的 FLOPs（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)


def poly(fn):
    pts = []
    for T in Ts:
        F = fn(T)
        if 10 ** LY0 <= F <= 10 ** LY1:
            pts.append("%.1f %.1f" % (px(T), py(F)))
    return "M" + " L".join(pts)


svg.path(poly(lambda T: MQA * T), stroke="blue", width=2, dash="6 4")
svg.path(poly(lambda T: MHA * T), stroke="blue", width=2.6)
svg.path(poly(lambda T: IDX * T), stroke="violet", width=2.2)
svg.path(poly(core), stroke="green", width=2.2, dash="3 3")
svg.path(poly(total), stroke="green", width=3)

# direct labels at the right edge
Te = 131072
labels = [
    (MQA * Te, "稠密 MQA 模式 278,528·T", "blue", 0),
    (MHA * Te, "稠密 MHA 模式 81,920·T", "blue", 0),
    (total(Te), "DSA 合计", "green", -4),
    (IDX * Te, "indexer 16,384·T", "violet", 12),
    (core(Te), "DSA 核心 278,528·2048", "green", 0),
]
for F, lab, col, dy in labels:
    svg.text(X1 + 8, py(F) + 4 + dy, lab, size=12, color=col, weight="600")

# crossover markers
marks = [
    (X_MQA, MQA * X_MQA, "2,176", "DSA 合计 = 稠密 MQA（解码口径）"),
    (X_CORE, MHA * X_CORE, "6,963", "DSA 核心 = 稠密 MHA"),
    (X_TOT, MHA * X_TOT, "8,704", "DSA 合计 = 稠密 MHA（预填充口径）"),
    (X_IDX, IDX * X_IDX, "34,816", "indexer = DSA 核心"),
]
for i, (T, F, lab, desc) in enumerate(marks):
    x, y = px(T), py(F)
    svg.circle(x, y, 4.2, fill="ink")
    svg.line(x, y + 5, x, Y1, color="ink", width=0.8, dash="2 3")
    svg.text(x + 4, Y1 - 8 - i * 16, lab, size=11, color="ink", weight="700")

# key box
bx, by = 100, 48
svg.rect(bx - 6, by - 14, 280, 92, fill="surface", fill_opacity=0.9, stroke="rule", width=1, rx=6)
for i, (T, F, lab, desc) in enumerate(marks):
    svg.text(bx, by + 4 + i * 20, "%s：%s" % (lab, desc), size=12)

svg.text(X0, H - 16, "只数 QKᵀ 与 PV（每次乘加 = 2 FLOPs）；投影、softmax、top-k 不计。DSA 核心在 T = 2048 处折平。",
         size=11, color="muted")

svg.save("l14-flops-crossover.svg")
