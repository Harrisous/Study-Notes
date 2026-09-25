"""Figure 18.10: which V4 parameters use Muon vs AdamW (V4 report sec. 2.4; research notes sec. 3.4)
and how the sqrt(max(p, q)) * 0.18 rescaling puts orthogonalised updates of different shapes on one
scale (Liu et al. Lemma 1: RMS(U V^T) = 1 / sqrt(max(p, q)) for a full-rank p x q matrix).
Shapes are V4-Pro matrices: W^DQ 1536 x 7168, W^UQ 65536 x 1536, one output group 1024 x 4096.
Writes deepseek/figures/l18-muon-split.svg
"""

import math

import numpy as np

from l18_svg import SVG

SHAPES = [("W^{DQ}", 1536, 7168), ("W^{UQ}", 65536, 1536), ("W^{O1} 的一组", 1024, 4096)]
GAMMA = 0.18
# check Lemma 1 numerically on a small random matrix
rng = np.random.default_rng(0)
A = rng.normal(size=(64, 16))
U, _, Vt = np.linalg.svd(A, full_matrices=False)
O = U @ Vt
assert abs(np.sqrt((O ** 2).mean()) - 1 / math.sqrt(64)) < 1e-9

W, H = 760, 446
svg = SVG(W, H, "Muon 与 AdamW 的分工，以及更新 RMS 的缩放")

# ---------------------------------------------------------------- the two columns
cols = [
    (24, "AdamW", "muted", ["embedding（129,280 × d）", "预测头（d × 129,280）", "所有 RMSNorm 的权重",
                            "mHC 的静态偏置 S 与门控因子 α"]),
    (392, "Muon", "blue", ["注意力的全部投影：W^{DQ}、W^{UQ}、窗口投影、", "压缩器、indexer 投影、分组输出投影",
                           "每个路由专家与共享专家的三个矩阵", "mHC 的动态投影 W^{pre}、W^{res}、W^{post}",
                           "路由器、MTP 模块里的矩阵（推测）"]),
]
for x, title, col, items in cols:
    svg.rect(x, 20, 344, 178, fill=col, fill_opacity=0.08, stroke=col, width=1.3, rx=8)
    svg.text(x + 16, 46, title, size=15, weight="700", color=col)
    for n, it in enumerate(items):
        svg.text(x + 16, 76 + n * 24, "· " + it if not it.startswith("压缩器") else "  " + it, size=12)
svg.text(24, 218, "AdamW：β₁ = 0.9，β₂ = 0.95，ε = 10⁻²⁰，权重衰减 0.1", size=12, color="muted")
svg.text(392, 218, "Muon：动量 0.95，权重衰减 0.1，更新 RMS = 0.18", size=12, color="muted")

# ---------------------------------------------------------------- RMS bars (log scale)
BX0, BY0, BY1 = 150, 272, 396
LOG0, LOG1 = -3.0, 0.0            # 0.001 .. 1


def by(v):
    return BY1 - (math.log10(v) - LOG0) / (LOG1 - LOG0) * (BY1 - BY0)


svg.text(24, 252, "正交化后的更新 RMS（对数轴）", size=13, weight="700")
for e in (-3, -2, -1, 0):
    y = by(10 ** e)
    svg.line(BX0, y, 740, y, color="grid", width=1)
    svg.text(BX0 - 8, y + 4, "10^{%d}" % e if e else "1", size=11, anchor="end", color="muted")
svg.line(BX0, BY1, 740, BY1, color="muted", width=1.2)
for n, (name, p, q) in enumerate(SHAPES):
    before = 1 / math.sqrt(max(p, q))
    after = before * math.sqrt(max(p, q)) * GAMMA
    assert abs(after - GAMMA) < 1e-12
    cx = BX0 + 100 + n * 200
    for k, (v, col, lab) in enumerate(((before, "muted", "%.4f" % before), (after, "blue", "0.18"))):
        x = cx - 34 + k * 40
        svg.rect(x, by(v), 30, BY1 - by(v), fill=col, fill_opacity=0.55 if col == "muted" else 0.75,
                 stroke=col, width=0.8)
        svg.text(x + 15, by(v) - 6, lab, size=11, anchor="middle", color=col)
    svg.text(cx - 4, BY1 + 18, "%s：%d × %d" % (name, p, q), size=12, anchor="middle")
svg.rect(360, 242, 14, 12, fill="muted", fill_opacity=0.55, stroke=None)
svg.text(380, 252, "正交因子 UV^{⊤}：1/√max(p,q)", size=11)
svg.rect(560, 242, 14, 12, fill="blue", fill_opacity=0.75, stroke=None)
svg.text(580, 252, "再乘 √max(p,q) × 0.18", size=11)
svg.text(24, 440, "形状按 V4-Pro 的真实矩阵取（输出 × 输入）；缩放后与形状无关，这就是能共用一个学习率的原因。", size=11,
         color="muted")

svg.save("l18-muon-split.svg")
