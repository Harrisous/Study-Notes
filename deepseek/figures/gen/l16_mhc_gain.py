"""Figure 16.12: propagation stability of mHC on the 27B model (redrawn after mHC Fig. 7), next to HC.

Data: research notes section 5.3 item 6 -- DIGITISED from the vector paths of mHC (arXiv 2512.24880 v2) Fig. 7
(not numbers printed in the paper). Single-layer forward gains lie in 1.0000-1.0005 and composite forward gains in
1.0001-1.0066 (only ranges are recorded, drawn here as bands). Backward gains are the 60-point lists below.
Composite backward maximum 1.649 (text of the paper: "approximately 1.6"); HC's composite backward peak
(Fig. 3, digitised) 2825 -> ratio 1713, i.e. 3.2 orders of magnitude (notes N19).
Upper bound n = 4 for the backward gain of products of row-stochastic nonnegative matrices (notes 2.7 item 6).
Writes deepseek/figures/l16-mhc-gain.svg
"""

import math

from l16_svg import SVG

SB = [1.037, 1.027, 1.027, 1.033, 1.032, 1.054, 1.061, 1.054, 1.041, 1.045, 1.036, 1.032, 1.044, 1.035, 1.03,
      1.041, 1.044, 1.032, 1.045, 1.025, 1.018, 1.021, 1.021, 1.012, 1.011, 1.013, 1.023, 1.022, 1.032, 1.04, 1.039,
      1.049, 1.039, 1.044, 1.04, 1.038, 1.034, 1.022, 1.03, 1.03, 1.03, 1.021, 1.036, 1.034, 1.034, 1.023, 1.025,
      1.026, 1.021, 1.025, 1.028, 1.027, 1.024, 1.034, 1.019, 1.033, 1.031, 1.051, 1.022, 1.006]
CB = [1.11, 1.071, 1.233, 1.211, 1.182, 1.325, 1.378, 1.451, 1.507, 1.528, 1.555, 1.553, 1.554, 1.563, 1.566,
      1.553, 1.599, 1.623, 1.624, 1.649, 1.645, 1.646, 1.646, 1.641, 1.636, 1.631, 1.623, 1.599, 1.58, 1.555, 1.504,
      1.501, 1.52, 1.513, 1.484, 1.46, 1.434, 1.41, 1.395, 1.376, 1.353, 1.331, 1.315, 1.284, 1.279, 1.27, 1.258,
      1.255, 1.247, 1.232, 1.22, 1.211, 1.2, 1.177, 1.141, 1.126, 1.091, 1.062, 1.027, 1.006]
assert len(SB) == len(CB) == 60
imax = max(range(60), key=lambda i: CB[i])
print("single-layer backward range %.3f..%.3f; composite backward max %.3f at l=%d" % (min(SB), max(SB), CB[imax],
                                                                                    imax + 1))
print("HC/mHC peak ratio: %.0f  (log10 %.2f)" % (2825 / CB[imax], math.log10(2825 / CB[imax])))

W, H = 760, 420
svg = SVG(W, H, "mHC 在 27B 模型上的增益：单层与复合都在 1 附近（据 mHC 论文图 7 重绘，数字化值）")
V0, V1 = 0.9, 1.8


def panel(x0, x1, y0, y1, title, fwd_band, back, notes):
    def px(l):
        return x0 + (l - 1) / 59 * (x1 - x0)

    def py(v):
        return y1 - (v - V0) / (V1 - V0) * (y1 - y0)

    svg.text(x0 - 40, 26, title, size=14, weight="700")
    for v in (0.9, 1.0, 1.2, 1.4, 1.6, 1.8):
        svg.line(x0, py(v), x1, py(v), color="grid", width=1)
        svg.text(x0 - 7, py(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
    for l in (1, 10, 20, 30, 40, 50, 60):
        svg.line(px(l), y0, px(l), y1, color="grid", width=1)
        svg.text(px(l), y1 + 17, str(l), size=11, anchor="middle", color="muted")
    svg.line(x0, y1, x1, y1, color="muted", width=1.2)
    svg.line(x0, y0, x0, y1, color="muted", width=1.2)
    svg.text((x0 + x1) / 2, y1 + 38, "子层编号 ~l~", size=12, anchor="middle", color="muted")
    lo, hi = fwd_band
    svg.rect(x0, py(hi) - 1.2, x1 - x0, max(2.4, py(lo) - py(hi) + 2.4), fill="ink", fill_opacity=0.55)
    d = "M" + " L".join("%.1f %.1f" % (px(i + 1), py(v)) for i, v in enumerate(back))
    svg.path(d, stroke="green", width=2.2, join="round")
    notes(px, py)


def notes_a(px, py):
    svg.text(px(30), py(1.0) + 20, "前向：1.0000–1.0005（灰色细带）", size=12, anchor="middle")
    svg.text(px(30), py(1.12), "反向：1.006–1.061（绿线）", size=12, anchor="middle", color="green")


def notes_b(px, py):
    svg.circle(px(imax + 1), py(CB[imax]), 4, fill="green")
    svg.text(px(imax + 1) - 4, py(1.765), "反向最大 ≈ %.2f（~l~ = %d）" % (CB[imax], imax + 1), size=12,
             color="green", weight="600")
    svg.text(px(imax + 1) - 4, py(1.765) + 17, "论文文字：约 1.6", size=12, color="green")
    svg.text(px(30), py(1.0) + 20, "前向：1.0001–1.0066（灰色细带）", size=12, anchor="middle")


panel(60, 350, 50, 330, "(a) 单层 H^{res}_{l}（线性纵轴）", (1.0000, 1.0005), SB, notes_a)
panel(450, 740, 50, 330, "(b) 复合映射（线性纵轴）", (1.0001, 1.0066), CB, notes_b)

svg.text(20, 396, "对照 HC（图 16.5）：复合反向增益峰值约 2825，是这里 1.65 的约 1700 倍，相差三个多数量级。", size=12)
svg.text(20, 414, "理论上界：行和精确为 1 的非负矩阵之积，反向增益不超过 n = 4（§4）；实测远低于上界。", size=12,
         color="muted")

svg.save("l16-mhc-gain.svg")
