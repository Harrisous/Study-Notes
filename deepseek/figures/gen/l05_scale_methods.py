"""Figure 5.13: per-pair frequency scale theta'_i / theta_i under four context-extension methods.

DeepSeek setting: d_h^R = 64, L = 4096, s = 40 (alpha = 1, beta = 32). PI: 1/40 for every pair;
NTK-aware: s^(-2i/(d-2)); YaRN as written in the paper (ramp linear in r); YaRN reference code as used by
DeepSeek (ramp linear in the pair index, floor/ceil boundaries). Log y axis. Research notes R6-R8.
Writes deepseek/figures/l05-scale-methods.svg
"""

import math

from l05_svg import SVG
from l05_yarn_common import (I_ALPHA, I_BETA, N, S, scale_code, scale_ntk, scale_paper, scale_pi)

W, H = 760, 430
X0, X1, Y0, Y1 = 80, 600, 40, 350
LY0, LY1 = math.log10(0.02), math.log10(1.25)


def X(i):
    return X0 + i / (N - 1) * (X1 - X0)


def Y(v):
    return Y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "四种上下文扩展方法对每一对频率的缩放（DeepSeek 设置：64 维，s = 40）")
for v in (0.025, 0.05, 0.1, 0.2, 0.5, 1.0):
    svg.line(X0, Y(v), X1, Y(v), color="grid", width=1)
    svg.text(X0 - 8, Y(v) + 4, ("%g" % v), size=11, anchor="end", color="muted")
for i in (0, 4, 8, 12, 16, 20, 24, 28, 31):
    svg.text(X(i), Y1 + 17, str(i), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 36, "第 i 对（0 = 最快的针，31 = 最慢的针）", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "频率缩放 θ'_{i} / θ_{i}（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

# region where NTK-aware is above 1/40 although the pair is a "slow hand" (lambda > L): i >= 23
svg.rect(X(22.5), Y0, X(31) - X(22.5) + 6, Y1 - Y0, fill="red", fill_opacity=0.06)
svg.text(X(26.8), Y0 + 16, "慢针（λ > 4096）", size=12, anchor="middle", color="red")
# YaRN boundaries
for xi, lab, anc in ((I_BETA, "r = β = 32", "start"), (I_ALPHA, "r = α = 1", "end")):
    svg.line(X(xi), Y0, X(xi), Y1, color="ink", width=1, dash="4 3")
    svg.text(X(xi) + (4 if anc == "start" else -4), Y0 + 12, lab, size=11.5, color="ink", anchor=anc)

series = [
    ("PI", scale_pi, "muted", "6 4", None),
    ("NTK-aware", scale_ntk, "violet", None, "dot"),
    ("YaRN 论文式（按 r 线性）", scale_paper, "amber", None, "ring"),
    ("YaRN 代码（DeepSeek 所用）", scale_code, "blue", None, "dot"),
]
for name, f, col, dash, mark in series:
    pts = [(X(i), Y(f(i))) for i in range(N)]
    svg.polyline(pts, stroke=col, width=2.0 if col != "muted" else 1.6, dash=dash)
    if mark == "dot":
        for x, y in pts:
            svg.circle(x, y, 2.6, fill=col)
    elif mark == "ring":
        for x, y in pts:
            svg.circle(x, y, 3.6, stroke=col, width=1.3)
# NTK out-of-bound marker at i = 23
svg.circle(X(23), Y(scale_ntk(23)), 6, stroke="red", width=1.6)
svg.text(X(23.4), Y(0.16), "i = 23：NTK 缩放 %.4f" % scale_ntk(23), size=11.5, color="red", weight="600")
svg.text(X(23.4), Y(0.16) + 16, "> 1/40，越界外推", size=11.5, color="red", weight="600")
# i = 16 comparison
svg.line(X(16), Y(scale_paper(16)), X(16), Y(scale_code(16)), color="ink", width=1)
svg.text(X(16) + 6, Y(scale_code(16)) - 6, "i = 16：代码 %.2f" % scale_code(16), size=11.5, color="blue")
svg.text(X(16) - 6, Y(scale_paper(16)) + 4, "论文 %.4f" % scale_paper(16), size=11.5, color="amber", anchor="end",
         halo=True)  # the NTK-aware line passes behind this label

# legend
lx, ly = X1 + 14, Y0 + 20
for k, (name, f, col, dash, mark) in enumerate(series):
    y = ly + k * 42
    svg.line(lx, y, lx + 26, y, color=col, width=2.2, dash=dash)
    words = name.split("（")
    svg.text(lx + 32, y + 4, words[0], size=12.5, color=col, weight="600")
    if len(words) > 1:
        svg.text(lx + 32, y + 21, "（" + words[1], size=11.5, color=col)
svg.text(lx, ly + 190, "1 = 保持原频率（外推）", size=12, color="ink")
svg.text(lx, ly + 208, "1/%d = 0.025 = 完全插值" % S, size=12, color="ink")
svg.save("l05-scale-methods.svg")
