"""Figure M7.7: learning-rate schedules on one axis (fraction of the training tokens), plus MiniCPM's decay.

Left, η/η_max against progress p = τ/D (warmup omitted on the main axes):
  cosine to 10%          0.1 + 0.45 (1 + cos πp)
  DeepSeek LLM multi-step 1 (p < 0.8), 10^{-1/2} = 0.316 (0.8 ≤ p < 0.9), 0.1 (p ≥ 0.9)
  DeepSeek-V4.1-Flash     2.6e-4 until 28T of 45T tokens, cosine to 2.6e-5 by 40T, then flat to 45T
                          (stable 62.2 %, decay 26.7 %, final hold 11.1 %)
  Inset: V4.1's 2000 warmup steps at a fixed 100.6M-token batch = 201.2B tokens = 0.447 % of 45T
         (drawn linear, as V4.1 §4.2.2 states: "linearly warmed up over the first 2000 steps").
Right, MiniCPM's exponential decay read as a half-life: η = 0.01 · 0.5^{u/5000}, u = steps into the decay.
Every point is computed here. Writes math/figures/m07-lr-schedules.svg
"""

import math

import numpy as np

from m07_svg import SVG, Axes, clipped_path

p = np.linspace(0, 1, 1201)
cosine = 0.1 + 0.45 * (1 + np.cos(math.pi * p))
multi = np.where(p < 0.8, 1.0, np.where(p < 0.9, 10 ** -0.5, 0.1))
a, b = 28 / 45, 40 / 45


def v41(x):
    if x < a:
        return 1.0
    if x < b:
        tau = (x - a) / (b - a)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * tau))
    return 0.1


v41c = np.array([v41(x) for x in p])

W, H = 740, 372
svg = SVG(W, H, "学习率调度：cosine、DeepSeek LLM 多步、DeepSeek-V4.1-Flash，以及 MiniCPM 的指数衰减")
ax = Axes(66, 44, 470, 300, (0, 1), (0, 1.08))
svg.text((ax.x0 + ax.x1) / 2, 22, "η / η_{max} 随训练进度（按 token 计）", size=14, anchor="middle", weight="600")
# V4.1 phase shading
for x0, x1, lab in ((0, a, "稳定 62.2%"), (a, b, "cosine 衰减 26.7%"), (b, 1, "保持 11.1%")):
    svg.rect(ax.X(x0), ax.y0, ax.X(x1) - ax.X(x0), ax.y1 - ax.y0, fill="green",
             fill_opacity=0.05 if lab.startswith("cos") else 0.0)
ax.ygrid(svg, [0, 0.1, 0.316, 0.5, 1.0], labels=["0", "0.1", "0.316", "0.5", "1"])
ax.xgrid(svg, [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0", "20%", "40%", "60%", "80%", "100%"])
ax.frame(svg)
svg.text((ax.x0 + ax.x1) / 2, ax.y1 + 36, "已处理的训练 token 占计划总量的比例", size=12, anchor="middle", color="muted")
box = (ax.x0, ax.y0 - 2, ax.x1, ax.y1)
svg.path(clipped_path([ax(x, y) for x, y in zip(p, cosine)], box), stroke="muted", width=2.0, dash="6 4")
svg.path(clipped_path([ax(x, y) for x, y in zip(p, multi)], box), stroke="blue", width=2.4)
svg.path(clipped_path([ax(x, y) for x, y in zip(p, v41c)], box), stroke="green", width=2.4)
svg.line(ax.X(a), ax.y0, ax.X(a), ax.y1, color="green", width=0.9, dash="2 3")
# the 40T guide line is interrupted where the DeepSeek LLM label (three lines at Y(0.80) .. +29) sits
lab_top, lab_bot = ax.Y(0.80) - 14, ax.Y(0.80) + 34
svg.line(ax.X(b), ax.y0, ax.X(b), lab_top, color="green", width=0.9, dash="2 3")
svg.line(ax.X(b), lab_bot, ax.X(b), ax.y1, color="green", width=0.9, dash="2 3")
svg.text(ax.X(a) - 3, ax.y0 + 12, "28T", size=11, color="green", anchor="end")
svg.text(ax.X(b) + 3, ax.y0 + 12, "40T", size=11, color="green")
svg.text(ax.X(0.39), ax.Y(0.45), "cosine", size=12, color="muted", weight="600")
svg.text(ax.X(0.39), ax.Y(0.45) + 15, "（降到 10%）", size=11, color="muted")
svg.text(ax.X(0.02), ax.Y(1.0) - 8, "V4.1-Flash：峰值 2.6×10^{−4}", size=12, color="green", weight="600")
# the 40T guide line runs behind this label: a surface-coloured halo keeps the text legible
svg.text(ax.X(0.81), ax.Y(0.80), "DeepSeek LLM", size=12, color="blue", weight="600", halo=True)
svg.text(ax.X(0.81), ax.Y(0.80) + 15, "80%：×0.316", size=11, color="blue", halo=True)
svg.text(ax.X(0.81), ax.Y(0.80) + 29, "90%：×0.1", size=11, color="blue", halo=True)
svg.text(ax.X(0.99), ax.Y(0.1) + 17, "V4.1 末段：峰值的 10%", size=11, color="green", anchor="end")

# inset: warmup of V4.1
ix = Axes(108, 196, 196, 262, (0, 0.012), (0, 1.0))
svg.rect(ix.x0 - 30, ix.y0 - 20, ix.x1 - ix.x0 + 40, ix.y1 - ix.y0 + 44, fill="surface", stroke="rule", width=1)
svg.text(ix.x0 - 24, ix.y0 - 7, "放大：前 1.2% 的 warmup", size=11, color="ink")
ix.frame(svg)
wu = 2000 * 100.6e6 / 45e12
svg.path("M%.1f %.1f L%.1f %.1f L%.1f %.1f" % (ix.X(0), ix.Y(0), ix.X(wu), ix.Y(1), ix.X(0.012), ix.Y(1)),
         stroke="green", width=2)
svg.text(ix.X(wu), ix.y1 + 14, "0.447%", size=11, anchor="middle", color="green")
svg.text(ix.x0 - 4, ix.Y(1) + 4, "1", size=11, anchor="end", color="muted")
svg.text(ix.x0 - 4, ix.Y(0) + 4, "0", size=11, anchor="end", color="muted")

# right: MiniCPM decay
ax2 = Axes(560, 44, 720, 300, (0, 15000), (0, 0.0105))
svg.text((ax2.x0 + ax2.x1) / 2 - 10, 22, "MiniCPM 的衰减段", size=14, anchor="middle", weight="600")
ax2.ygrid(svg, [0, 0.0025, 0.005, 0.0075, 0.01], labels=["0", "0.0025", "0.005", "0.0075", "0.01"])
ax2.xgrid(svg, [0, 5000, 10000, 15000], labels=["0", "5k", "10k", "15k"])
ax2.frame(svg)
svg.text((ax2.x0 + ax2.x1) / 2, ax2.y1 + 36, "衰减段内的步数 u", size=12, anchor="middle", color="muted")
us = np.linspace(0, 15000, 301)
svg.path(clipped_path([ax2(u, 0.01 * 0.5 ** (u / 5000)) for u in us], (ax2.x0, ax2.y0, ax2.x1, ax2.y1)),
         stroke="violet", width=2.4)
for u in (0, 5000, 10000):
    v = 0.01 * 0.5 ** (u / 5000)
    svg.circle(ax2.X(u), ax2.Y(v), 3.4, fill="violet")
    svg.text(ax2.X(u) + 6, ax2.Y(v) - 6, "%g" % v, size=11, color="violet")
svg.text(ax2.x1 - 8, ax2.Y(0.0085), "0.01 × 0.5^{u/5000}", size=12, anchor="end", color="violet", weight="600")
svg.text(ax2.x1 - 8, ax2.Y(0.0085) + 16, "半衰期 5000 步", size=11, anchor="end", color="muted")
svg.text(66, H - 10, "左图忽略 warmup（见插图）；cosine 为通用形状。V4.1 的阶段按 token 划分：0–28T、28–40T、40–45T。",
         size=11, color="muted")
svg.save("m07-lr-schedules.svg")
print("warmup fraction", wu, "multistep 0.316:", 10 ** -0.5)
