"""Figure 17.8: where the parameters of the four 27B-scale models go, and V4.1-Flash for scale.

Segments (notes §4.2, recomputed in l17_num.py):
  token embedding 129,280 x 2560 = 0.331B;  activated (non-embedding) 3.8B (Engram table 1);
  inactive routed experts (72-6 or 55-6) x 29 MoE layers x 3*2560*1536 (expert width inferred from the
  DeepSeek-VL2 27B config -- the Engram paper does not give it);  Engram table 5.79B / 18.53B
  (prime layout of the demo code).  Totals 4.13 / 26.71 / 26.69 / 39.43B vs paper 4.1 / 26.7 / 26.7 / 39.5.
V4.1-Flash: 552B backbone + 196.6B Engram (dsv41 §2.1, config).
Writes deepseek/figures/l17-ledger.svg
"""

from l17_svg import SVG
from l17_num import ACT, EMB, P_MOE_27, P_MOE_BASE, E27_PARAMS, E40_PARAMS, V41_PARAMS, RHO_27, TOTALS

G = 1e9
MODELS = [
    ("Dense-4B", [EMB, ACT, 0, 0], "稠密"),
    ("MoE-27B", [EMB, ACT, P_MOE_BASE, 0], "ρ = 100%"),
    ("Engram-27B", [EMB, ACT, P_MOE_27, E27_PARAMS], "ρ = %.1f%%" % (RHO_27 * 100)),
    ("Engram-40B", [EMB, ACT, P_MOE_27, E40_PARAMS], "表更大"),
]
for name, seg, _ in MODELS:
    assert abs(sum(seg) - TOTALS[name]) < 1
SEGS = [("token 嵌入 0.33B", "muted", 0.35), ("激活参数 3.8B", "blue", 0.55), ("不激活的路由专家", "violet", 0.22),
        ("Engram 表", "amber", 0.6)]

W, H = 760, 400
svg = SVG(W, H, "四个 27B 规模模型的参数账本，以及 V4.1-Flash")

X0, Y0, Y1 = 70, 40, 350
VMAX = 42


def y(v):
    return Y1 - v / VMAX * (Y1 - Y0)


for v in range(0, 43, 10):
    svg.line(X0, y(v), 470, y(v), color="grid", width=1)
    svg.text(X0 - 8, y(v) + 4, "%dB" % v, size=11, anchor="end", color="muted")
svg.line(X0, Y1, 470, Y1, color="muted", width=1.2)
svg.text(20, 22, "总参数分解（十亿）", size=13, weight="700")

BW = 64
for i, (name, seg, tag) in enumerate(MODELS):
    bx = X0 + 24 + i * 100
    acc = 0.0
    for (lab, col, op), v in zip(SEGS, seg):
        if v <= 0:
            continue
        v /= G
        svg.rect(bx, y(acc + v), BW, y(acc) - y(acc + v), fill=col, fill_opacity=op, stroke=col, width=1)
        if v > 2.5:
            svg.text(bx + BW / 2, (y(acc) + y(acc + v)) / 2 + 4, "%.2f" % v, size=12, anchor="middle",
                     color="ink" if col != "amber" else "ink", weight="700" if col == "amber" else None)
        acc += v
    svg.text(bx + BW / 2, y(acc) - 22, "%.2fB" % acc, size=12.5, anchor="middle", weight="700")
    svg.text(bx + BW / 2, y(acc) - 7, tag, size=11.5, anchor="middle", color="amber" if "ρ = 7" in tag else "muted")
    svg.text(bx + BW / 2, Y1 + 18, name, size=12, anchor="middle")
svg.text(X0 + 24 + 150 + BW / 2, Y1 + 36, "↑ 等总参数、等激活参数 ↑", size=11.5, anchor="middle", color="muted")

# legend
LX, LY = 488, 52
for k, (lab, col, op) in enumerate(reversed(SEGS)):
    yy = LY + k * 24
    svg.rect(LX, yy - 11, 16, 14, fill=col, fill_opacity=op, stroke=col, width=1)
    svg.text(LX + 24, yy, lab, size=12)
svg.text(LX, LY + 108, "专家宽度 1536 取自 DeepSeek-VL2 的", size=11.5, color="muted")
svg.text(LX, LY + 125, "27B 配置（笔者推断，论文未给）；", size=11.5, color="muted")
svg.text(LX, LY + 142, "Engram 表按素数布局复算。", size=11.5, color="muted")

# V4.1 on its own scale
VY = 262
svg.text(LX, VY - 22, "V4.1-Flash（另一尺度）", size=13, weight="700")
tot = 552 + V41_PARAMS / G
bw = 250
w1 = 552 / tot * bw
svg.rect(LX, VY, w1, 30, fill="violet", fill_opacity=0.18, stroke="violet", width=1)
svg.rect(LX + w1, VY, bw - w1, 30, fill="amber", fill_opacity=0.6, stroke="amber", width=1)
svg.text(LX + w1 / 2, VY + 20, "骨干 552B", size=12, anchor="middle")
svg.text(LX + w1 + (bw - w1) / 2, VY + 20, "196.6B", size=12, anchor="middle", weight="700")
svg.text(LX, VY + 50, "合计 %.1fB；Engram 占 %.1f%%" % (tot, V41_PARAMS / G / tot * 100), size=12)
svg.text(LX, VY + 68, "官方“552B”不含 Engram", size=11.5, color="muted")
svg.text(LX, VY + 86, "（骨干里激活的只有 8B / 16B）", size=11.5, color="muted")

assert round(V41_PARAMS / G / tot * 100, 1) == 26.3
svg.save("l17-ledger.svg")
