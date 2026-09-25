"""Figure 15.1: the V3 -> V3.2 release timeline (Dec 2024 - Dec 2025) and the post-training share.

Dates: HF API createdAt / DeepSeek API news pages (lesson-15-doc.md section 0.3).
Post-training share: V3 report section 1 gives 2664K pre-training and 5K post-training H800 GPU hours
(5/2664 = 0.188%); V3.2 section 1 says the post-training budget exceeds 10% of the pre-training cost.
The x axis is linear in days; the share panel uses a log axis.
Writes deepseek/figures/l15-timeline.svg
"""

import datetime as dt
import math

from l15_svg import SVG

D0, D1 = dt.date(2024, 12, 1), dt.date(2026, 1, 1)
X0, X1 = 36, 736
AY = 214                      # axis y


def dx(d):
    return X0 + (d - D0).days / (D1 - D0).days * (X1 - X0)


W, H = 760, 534
svg = SVG(W, H, "DeepSeek-V3 到 V3.2 的发布时间线，以及后训练算力占比")

# ---------------------------------------------------------------- axis and month ticks
svg.line(X0, AY, X1, AY, color="ink", width=1.6)
for y, m in [(2024, 12)] + [(2025, k) for k in range(1, 13)] + [(2026, 1)]:
    d = dt.date(y, m, 1)
    x = dx(d)
    svg.line(x, AY - 4, x, AY + 4, color="ink", width=1)
    if (y, m) in ((2024, 12), (2026, 1)):
        lab = "%d.%d" % (y, m)
    else:
        lab = str(m)
    if (y, m) != (2026, 1):
        svg.text(x + 2, AY + 18, lab, size=11, color="muted")
svg.text(dx(dt.date(2025, 1, 1)) + 2, AY + 32, "2025 年（月份）", size=11, color="muted")

TAG = {"arch": ("架构", "blue"), "train": ("训练", "green"), "num": ("数值", "orange"), "rel": ("发布", "muted")}

# (date, box x, box y, box w, [lines], [tags], dot colour)
EVENTS = [
    (dt.date(2024, 12, 25), 36, 118, 144, ["DeepSeek-V3", "671B / 37B，FP8 训练"], ["rel"], "ink"),
    (dt.date(2025, 3, 24), 106, 262, 170, ["V3-0324", "RL 中开始用 Keep Routing"], ["train"], "green"),
    (dt.date(2025, 5, 28), 238, 118, 132, ["R1-0528", "V3.1 的对比对象"], ["rel"], "ink"),
    (dt.date(2025, 8, 21), 290, 262, 196, ["V3.1（08-21；Base 08-19）", "混合思考 · UE8M0 缩放", "839B token 长上下文续训"],
     ["train", "num"], "orange"),
    (dt.date(2025, 9, 22), 362, 28, 194, ["V3.1-Terminus（09-22）", "语言一致性、代码/搜索智能体"], ["train"], "green"),
    (dt.date(2025, 9, 29), 564, 28, 188, ["V3.2-Exp（09-29）", "DSA 稀疏注意力（第 14 课）", "API 降价 50%+"], ["arch"], "blue"),
    (dt.date(2025, 11, 27), 584, 146, 168, ["DeepSeekMath-V2", "11-27 · 自我验证的证明器"], ["train"], "green"),
    (dt.date(2025, 12, 1), 560, 262, 192, ["V3.2 与 V3.2-Speciale", "12-01 发布；论文 12-02", "规模化 RL · 智能体"],
     ["train"], "green"),
]

for d, bx, by, bw, lines, tags, dcol in EVENTS:
    x = dx(d)
    bh = 22 + 18 * len(lines) + 4
    above = by < AY
    # leader line: from the dot to the nearest horizontal edge of the box
    ey = by + bh if above else by
    ex = min(max(x, bx + 10), bx + bw - 10)
    svg.line(x, AY + (-5 if above else 5), ex, ey, color="muted", width=1, dash="3 3")
    svg.circle(x, AY, 5, fill=dcol, stroke="surface", width=1.5)
    svg.rect(bx, by, bw, bh, fill="surface", stroke="rule", width=1.2, rx=6)
    tx = bx + 8
    for t in tags:
        lab, col = TAG[t]
        svg.rect(tx, by + 5, 34, 16, fill=col, fill_opacity=0.14, stroke=col, width=0.8, rx=3)
        svg.text(tx + 17, by + 17.5, lab, size=11, anchor="middle", color=col)
        tx += 40
    for i, ln in enumerate(lines):
        svg.text(bx + 8, by + 38 + i * 18, ln, size=12 if i else 13, weight="700" if i == 0 else None,
                 color="ink" if i == 0 else "muted")

# ---------------------------------------------------------------- post-training share panel (log axis)
PY = 398
svg.line(X0, PY - 30, X1, PY - 30, color="rule", width=1)
svg.text(X0, PY - 8, "后训练算力占预训练的比例（对数刻度）", size=13, weight="700")
LX0, LX1 = 168, 700
LO, HI = -1, 2            # 0.1% .. 100%


def lx(pct):
    return LX0 + (math.log10(pct) - LO) / (HI - LO) * (LX1 - LX0)


for e in range(LO, HI + 1):
    x = lx(10 ** e)
    svg.line(x, PY + 6, x, PY + 88, color="grid", width=1)
    svg.text(x, PY + 104, ["0.1%", "1%", "10%", "100%"][e - LO], size=11, anchor="middle", color="muted")

V3_SHARE = 5 / 2664 * 100          # 0.1877 %
assert abs(V3_SHARE - 0.1877) < 1e-3
rows = [("DeepSeek-V3", V3_SHARE, "muted", "5K / 2664K GPU 小时 = 0.188%"),
        ("DeepSeek-V3.2", 10.0, "blue", "“超过预训练成本的 10%”（无绝对值）")]
for i, (lab, v, col, note) in enumerate(rows):
    y = PY + 14 + i * 36
    svg.text(LX0 - 10, y + 16, lab, size=12, anchor="end", weight="700")
    svg.rect(LX0, y, lx(v) - LX0, 22, fill=col, fill_opacity=0.35, stroke=col, width=1)
    svg.text(lx(v) + (8 if i == 0 else 16), y + 16, note, size=12, color="ink")
# the ">" marker for V3.2
x10 = lx(10.0)
svg.path("M%.1f %.1f l7 4 l-7 4" % (x10 + 2, PY + 50 + 3), stroke="blue", width=1.6)
svg.text(X0, PY + 124, "两者之比约 53 倍（10 / 0.188）。V3.2 没有说明“预训练成本”以哪一次预训练为基数。",
         size=12, color="muted")

svg.save("l15-timeline.svg")
