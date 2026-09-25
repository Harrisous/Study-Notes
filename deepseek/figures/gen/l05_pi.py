"""Figure 5.10: extrapolation vs position interpolation (after PI paper Fig. 1), with the two RoPE hands.

LLaMA: pre-trained on 2048 positions, extended to 4096 (s = 2), d_h = 128, b = 10000.
Left: three number lines (dots every 128 positions). Right: the slowest hand theta_63 and the fastest hand
theta_0, showing the angle range seen in pre-training, reached by extrapolation, and kept by PI.
All angles are computed here. Writes deepseek/figures/l05-pi.svg
"""

import math

from l05_svg import SVG

L, LP, S = 2048, 4096, 2
th_slow = 10000.0 ** (-126 / 128)
a_train = math.degrees(L * th_slow)        # 13.55 deg
a_extra = math.degrees(LP * th_slow)       # 27.10 deg
step_fast = math.degrees(1.0)              # 57.3 deg per token
step_fast_pi = math.degrees(1.0 / S)       # 28.6 deg per token

W, H = 760, 400
svg = SVG(W, H, "外推与位置插值：LLaMA 从 2048 扩到 4096")
X0, X1 = 150, 470                          # 0 .. 4096 maps to X0 .. X1


def X(p):
    return X0 + p / LP * (X1 - X0)


rows = [
    ("预训练", "位置 0 … 2048", 60),
    ("直接外推", "位置 0 … 4096", 160),
    ("位置插值 PI", "位置 t 当作 t/2 用", 260),
]
for lab, sub, y in rows:
    svg.text(20, y + 5, lab, size=13, weight="700")
    svg.text(20, y + 23, sub, size=12, color="muted")
    svg.line(X(0), y, X(LP), y, color="rule", width=1.2)
    for p in (0, 1024, 2048, 3072, 4096):
        svg.line(X(p), y - 4, X(p), y + 4, color="muted", width=1)
        svg.text(X(p), y + 20, str(p), size=11, anchor="middle", color="muted")
# pre-training dots
for p in range(0, L + 1, 128):
    svg.circle(X(p), 60, 3.6, fill="blue")
# extrapolation: seen part blue, unseen part red
for p in range(0, LP + 1, 128):
    svg.circle(X(p), 160, 3.6, fill="blue" if p <= L else "red")
svg.rect(X(L), 140, X(LP) - X(L), 12, fill="red", fill_opacity=0.10)
svg.text((X(L) + X(LP)) / 2, 134, "没见过的位置", size=12, anchor="middle", color="red", weight="600")
# PI: 33 positions squeezed into [0, 2048]
for p in range(0, LP + 1, 128):
    svg.circle(X(p / S), 260, 3.0, fill="green")
    if p <= 1024:
        pass
svg.text(X(L) + 10, 264, "4096 个位置全部落在 [0, 2048] 内", size=12, color="green")
# arrows from line 2 to line 3 for a few positions
for p in (1024, 2048, 3072, 4096):
    svg.arrow(X(p), 168, X(p / S), 252, color="green", width=1.0, dash="3 3")

# ---- right side: two clock faces per row (slowest hand, fastest hand)
CX1, CX2, R = 560, 680, 30
svg.text(CX1, 22, "最慢的针 i = 63", size=12.5, anchor="middle", weight="600")
svg.text(CX2, 22, "最快的针 i = 0", size=12.5, anchor="middle", weight="600")


def dial(cx, cy):
    svg.circle(cx, cy, R, fill="surface", stroke="rule", width=1.3)
    svg.line(cx, cy, cx + R, cy, color="muted", width=0.9, dash="2 2")


def wedge(cx, cy, a0, a1, col, op=0.30):
    steps = max(4, int(abs(a1 - a0) / 2))
    pts = [(cx, cy)] + [(cx + (R - 1) * math.cos(math.radians(a0 + (a1 - a0) * k / steps)),
                         cy - (R - 1) * math.sin(math.radians(a0 + (a1 - a0) * k / steps))) for k in range(steps + 1)]
    svg.polygon(pts, fill=col, fill_opacity=op)


def hand(cx, cy, a, col):
    svg.line(cx, cy, cx + (R - 4) * math.cos(math.radians(a)), cy - (R - 4) * math.sin(math.radians(a)),
             color=col, width=2.2, cap="round")


for lab, y in (("pre", 60), ("ext", 160), ("pi", 260)):
    cy = y + 8
    dial(CX1, cy)
    dial(CX2, cy)
    if lab == "pre":
        wedge(CX1, cy, 0, a_train, "blue")
        hand(CX1, cy, a_train, "red")
        svg.text(CX1, cy + R + 16, "0°–%.1f°" % a_train, size=11.5, anchor="middle")
        hand(CX2, cy, 0, "red")
        hand(CX2, cy, step_fast, "red")
        svg.text(CX2, cy + R + 16, "每 token %.1f°" % step_fast, size=11.5, anchor="middle")
    elif lab == "ext":
        wedge(CX1, cy, 0, a_train, "blue")
        wedge(CX1, cy, a_train, a_extra, "red", 0.35)
        hand(CX1, cy, a_extra, "red")
        svg.text(CX1, cy + R + 16, "到 %.1f°：越界" % a_extra, size=11.5, anchor="middle", color="red", weight="600")
        hand(CX2, cy, 0, "red")
        hand(CX2, cy, step_fast, "red")
        svg.text(CX2, cy + R + 16, "每 token %.1f°" % step_fast, size=11.5, anchor="middle")
    else:
        wedge(CX1, cy, 0, a_train, "green")
        hand(CX1, cy, a_train, "red")
        svg.text(CX1, cy + R + 16, "仍是 0°–%.1f°" % a_train, size=11.5, anchor="middle", color="green",
                 weight="600")
        hand(CX2, cy, 0, "red")
        hand(CX2, cy, step_fast_pi, "red")
        svg.text(CX2, cy + R + 16, "每 token 只剩 %.1f°" % step_fast_pi, size=11.5, anchor="middle", color="amber",
                 weight="600")

svg.text(20, 348, "外推：慢针转到训练中没见过的角度。插值：所有针都拨慢一半，慢针不越界，", size=12.5)
svg.text(20, 368, "但快针相邻两个 token 的角度差也减半，近邻变得更难区分（YaRN 对 PI 的批评）。", size=12.5)
svg.text(20, 390, "改绘自 PI 论文图 1 的思路；角度按 d_{h} = 128、b = 10000 计算。", size=12, color="muted")

svg.save("l05-pi.svg")
