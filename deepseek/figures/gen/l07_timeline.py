"""Figure 7.2: MoE from 1991 to DeepSeekMoE, total parameters (log axis) and activated params.

All numbers from the lesson-07 research notes (section 3.3 / 5.2, figure 7.1 there):
Shazeer 2017 MoE layer 137B; GShard 600B; Switch-C 1571B; ST-MoE-32B 269B;
Mixtral 46.7B / 12.9B (recomputed from config); DeepSeekMoE 16B 16.4B / 2.8B;
DeepSeekMoE 145B 144.6B / 22.2B (arXiv v1 only, preliminary); DeepSeek-V2-Lite 15.7B / 2.4B.
Writes deepseek/figures/l07-timeline.svg
"""

import math

from l07_svg import SVG

rows = [
    # date, name, expert text, total (B) or None, activated (B) or None, highlight
    ("1991", "Jacobs 等", "4 或 8 个专家，全部参与计算", None, None, False),
    ("2017-01", "Shazeer 等", "LSTM 层之间；最多 131,072 专家", 137, None, False),
    ("2020-06", "GShard", "每隔一层；每层 2048 专家，top-2", 600, None, False),
    ("2021-01", "Switch-C", "每层；2048 专家，top-1", 1571, None, False),
    ("2022-02", "ST-MoE-32B", "每 4 层一个；32 专家，top-2", 269, None, False),
    ("2024-01", "Mixtral 8x7B", "每层；8 专家，top-2", 46.7, 12.9, False),
    ("2024-01", "DeepSeekMoE 16B", "2 共享 + 64 路由选 6", 16.4, 2.8, True),
    ("2024-01", "DeepSeekMoE 145B", "4 共享 + 128 路由选 12（初步）", 144.6, 22.2, True),
    ("2024-05", "DeepSeek-V2-Lite", "2 共享 + 64 路由选 6", 15.7, 2.4, True),
]

W = 740
RH = 34
Y0 = 64
H = Y0 + RH * len(rows) + 56
svg = SVG(W, H, "MoE 的演进：1991 年至 2024 年各代模型的总参数与激活参数")

C_DATE, C_NAME, C_EXP = 16, 84, 228
BX0, BX1 = 468, 690
L0, L1 = 9.0, 12.3          # log10 range: 1B .. 2T


def bx(v_billion):
    return BX0 + (math.log10(v_billion * 1e9) - L0) / (L1 - L0) * (BX1 - BX0)


# header
svg.text(C_DATE, 24, "时间", size=12, color="muted", weight="600")
svg.text(C_NAME, 24, "模型", size=12, color="muted", weight="600")
svg.text(C_EXP, 24, "MoE 层的位置与专家配置", size=12, color="muted", weight="600")
svg.text(BX0, 24, "参数量（对数刻度）", size=12, color="muted", weight="600")
for v, lab in ((1, "1B"), (10, "10B"), (100, "100B"), (1000, "1T")):
    x = bx(v)
    svg.line(x, Y0 - 16, x, Y0 + RH * len(rows) - 6, color="grid", width=1)
    svg.text(x, 44, lab, size=11, anchor="middle", color="muted")

for i, (date, name, exp, tot, act, hl) in enumerate(rows):
    yc = Y0 + i * RH
    if hl:
        svg.rect(8, yc - 16, W - 16, RH - 3, fill="blue", fill_opacity=0.06, rx=4)
    svg.text(C_DATE, yc + 4, date, size=12, color="muted")
    svg.text(C_NAME, yc + 4, name, size=12, color="ink", weight="600" if hl else None)
    svg.text(C_EXP, yc + 4, exp, size=12, color="ink")
    if tot is None:
        svg.text(BX0, yc + 4, "玩具任务，参数量可忽略", size=12, color="muted")
        continue
    xt = bx(tot)
    svg.line(BX0, yc, xt, yc, color="blue", width=2, opacity=0.45)
    svg.circle(xt, yc, 5, fill="surface", stroke="blue", width=2)
    svg.text(xt + 9, yc + 4, ("%gB" % tot), size=11, color="ink")
    if act is not None:
        xa = bx(act)
        svg.circle(xa, yc, 4.5, fill="blue")
        svg.text(xa, yc - 8, "%gB" % act, size=11, anchor="middle", color="blue")

ly = Y0 + RH * len(rows) + 10
svg.circle(BX0 - 212, ly, 5, fill="surface", stroke="blue", width=2)
svg.text(BX0 - 202, ly + 4, "总参数（Shazeer 一行只计 MoE 层）", size=12, color="ink")
svg.circle(BX0 + 36, ly, 4.5, fill="blue")
svg.text(BX0 + 46, ly + 4, "每 token 激活参数", size=12, color="ink")
svg.text(16, ly + 30, "145B 只见于 DeepSeekMoE 的 arXiv v1（训练 245B token 的初步结果），正式发表版已删去。", size=12,
         color="muted")

svg.save("l07-timeline.svg")
