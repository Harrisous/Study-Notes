"""Figure 11.14: the V3 pre-training schedule on one token axis (0-14.8T), all values from V3 sec. 4.2.

learning rate: linear warm-up to 2.2e-4 in the first 2K steps; constant until 10T tokens; cosine decay to
2.2e-5 over 4.3T; 2.2e-5 for 333B; 7.3e-6 for the final 167B.
batch size: 3072 -> 15360 sequences over the first 469B tokens ("gradually"; the shape is not published,
drawn linear and dashed), then 15360.
MTP weight lambda: 0.3 for the first 10T, 0.1 for the remaining 4.8T.
bias update speed gamma: 0.001 for the first 14.3T, 0.0 for the remaining 500B.
sequence-wise balance factor alpha: 1e-4 throughout.
Writes deepseek/figures/l11-schedule.svg
"""

import math

import numpy as np

from l11_svg import SVG, polyline

TMAX = 14.8
assert abs(10 + 4.3 + 0.333 + 0.167 - TMAX) < 1e-9 and abs(14.3 + 0.5 - TMAX) < 1e-9

W, H = 720, 520
X0, X1 = 110, 700
svg = SVG(W, H, "V3 预训练日程：学习率、batch、λ、γ 放在同一条 token 轴上")


def px(t):
    return X0 + t / TMAX * (X1 - X0)


def lr(t):
    if t < 10:
        return 2.2e-4
    if t < 14.3:
        return 2.2e-5 + 0.5 * (2.2e-4 - 2.2e-5) * (1 + math.cos(math.pi * (t - 10) / 4.3))
    if t < 14.633:
        return 2.2e-5
    return 7.3e-6


rows = [("学习率", 40, 150), ("batch（序列数）", 190, 270), ("MTP 权重 λ", 310, 370), ("偏置速度 γ", 410, 470)]
for name, y0, y1 in rows:
    svg.text(X0 - 10, (y0 + y1) / 2 + 4, name, size=12, anchor="end", weight="600")
    svg.line(X0, y1, X1, y1, color="ink", width=1)
    svg.line(X0, y0, X0, y1, color="rule", width=1)
# vertical guides
for t, lab in ((10, "10T"), (14.3, "14.3T")):
    svg.line(px(t), 30, px(t), 480, color="muted", width=1, dash="4 4")
    svg.text(px(t), 24, lab, size=11, anchor="middle", color="muted")
for t in (0, 2, 4, 6, 8, 10, 12, 14.8):
    svg.text(px(t), 496, ("%g" % t) + ("T" if t else ""), size=11, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, 514, "已训练的 token 数（线性刻度，单位 T = 10^{12}）", size=12, anchor="middle")

# learning rate
y0, y1 = 40, 150


def ly(v):
    return y1 - v / 2.4e-4 * (y1 - y0)


pts = [(px(0), ly(0)), (px(0.02), ly(2.2e-4))] + [(px(t), ly(lr(t))) for t in np.linspace(0.02, 14.79, 400)]
polyline(svg, pts, color="blue", width=2)
svg.text(px(0.25), ly(2.2e-4) - 6, "2.2×10^{−4}（前 2K 步线性预热）", size=11, color="blue")
svg.text(px(10.2), ly(1.4e-4), "余弦衰减", size=11, color="blue")
# tail of the schedule, written between the two dashed guides so it crosses neither
svg.text(px(10) + 6, 167, "末段：2.2×10^{−5}（333B）", size=11, color="blue")
svg.text(px(10) + 6, 186, "→ 7.3×10^{−6}（最后 167B）", size=11, color="blue")

# batch
y0, y1 = 190, 270


def by(v):
    return y1 - v / 16000 * (y1 - y0)


svg.path("M %s %s L %s %s" % (px(0), by(3072), px(0.469), by(15360)), stroke="green", width=2, dash="5 3")
svg.line(px(0.469), by(15360), px(TMAX), by(15360), color="green", width=2)
svg.text(px(0.6), by(15360) - 6, "前 469B tokens 从 3072 条增到 15360 条（形状未公开，虚线示意）", size=11,
         color="green")
svg.text(px(0.05), by(3072) + 14, "3072", size=11, color="green")

# lambda
y0, y1 = 310, 370


def lamy(v):
    return y1 - v / 0.35 * (y1 - y0)


svg.path("M %s %s H %s V %s H %s" % (px(0), lamy(0.3), px(10), lamy(0.1), px(TMAX)), stroke="violet", width=2)
svg.text(px(4.5), lamy(0.3) - 6, "λ = 0.3", size=12, color="violet", weight="600")
svg.text(px(11.8), lamy(0.1) - 6, "λ = 0.1", size=12, color="violet", weight="600")

# gamma
y0, y1 = 410, 470


def gy(v):
    return y1 - v / 0.0012 * (y1 - y0)


svg.path("M %s %s H %s V %s H %s" % (px(0), gy(0.001), px(14.3), gy(0), px(TMAX)), stroke="orange", width=2)
svg.text(px(4.5), gy(0.001) - 6, "γ = 0.001", size=12, color="orange", weight="600")
svg.text(px(14.3) - 4, gy(0.0003), "最后 500B：γ = 0（偏置冻结）", size=11, color="orange", anchor="end")
svg.text(X0, 24, "另：序列级护栏系数 α = 10^{−4} 全程不变", size=11, color="muted")

svg.save("l11-schedule.svg")
