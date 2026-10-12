"""Figure 8.8: GRPO advantages for 0/1 rewards as a function of the group's accuracy p.

With population std: A+ = sqrt((1-p)/p) for correct answers, A- = -sqrt(p/(1-p)) for wrong ones.
Without std (Dr. GRPO, DeepSeek-V3.2): A+ = 1 - p, A- = -p. Right panel: the question's total
weight (1/G) * sum |A_i| = 2 sqrt(p(1-p)) with std, 2 p (1-p) without (research notes 4.1).
Curves are clipped to the plotting window in this script (no clipPath).
Writes deepseek/figures/l08-adv-vs-p.svg
"""

import math

from l08_svg import SVG, polyline

for c, G in ((1, 64), (16, 64), (32, 64), (63, 64), (1, 4), (2, 4)):
    p = c / G
    print("G=%d c=%d p=%.4f A+=%.4f A-=%.4f 1/std=%.4f sum|A|=%.3f sum|r-mean|=%.3f" % (
        G, c, p, math.sqrt((1 - p) / p), -math.sqrt(p / (1 - p)), 1 / math.sqrt(p * (1 - p)),
        2 * G * math.sqrt(p * (1 - p)), 2 * G * p * (1 - p)))

W, H = 740, 360
svg = SVG(W, H, "二值奖励下 GRPO 的优势与题目正确率的关系")
ps = [0.01 + 0.98 * k / 600 for k in range(601)]


def clipped(pts, ylo, yhi):
    """Split a polyline into pieces that stay inside [ylo, yhi] (values), dropping the rest."""
    out, cur = [], []
    for x, y in pts:
        if ylo <= y <= yhi:
            cur.append((x, y))
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


# ---------- left panel
PX0, PX1, PY0, PY1 = 60, 360, 40, 290
YL, YH = -4.0, 4.0


def px(p):
    return PX0 + p * (PX1 - PX0)


def py(v):
    return PY1 - (v - YL) / (YH - YL) * (PY1 - PY0)


for v in range(-4, 5):
    svg.line(PX0, py(v), PX1, py(v), color="grid", width=1)
    svg.text(PX0 - 6, py(v) + 4, ("%d" % v).replace("-", "−"), size=11, anchor="end", color="muted")
for p in (0, 0.25, 0.5, 0.75, 1.0):
    svg.line(px(p), PY0, px(p), PY1, color="grid", width=1)
    svg.text(px(p), PY1 + 16, "%g" % p, size=11, anchor="middle", color="muted")
svg.line(PX0, py(0), PX1, py(0), color="muted", width=1.1)
svg.text((PX0 + PX1) / 2, PY1 + 36, "组内正确率 p = c / G", size=12, anchor="middle", color="muted")
svg.text(PX0, 24, "每个回答的优势", size=13, weight="700")

curves = [
    ([(p, math.sqrt((1 - p) / p)) for p in ps], "green", None),
    ([(p, -math.sqrt(p / (1 - p))) for p in ps], "red", None),
    ([(p, 1 - p) for p in ps], "green", "5 4"),
    ([(p, -p) for p in ps], "red", "5 4"),
]
for pts, col, dash in curves:
    for piece in clipped(pts, YL, YH):
        polyline(svg, [(px(x), py(y)) for x, y in piece], color=col, width=2.4, dash=dash)
# annotate the extremes at G = 64
for c in (1, 63):
    p = c / 64
    a_plus, a_minus = math.sqrt((1 - p) / p), -math.sqrt(p / (1 - p))
    if c == 1:
        svg.text(px(0.06), py(3.7), "p = 1/64：答对的那一个 +7.94", size=11.5, color="green", halo=True)
    else:
        svg.text(px(0.37), py(-3.7), "p = 63/64：答错的那一个 −7.94", size=11.5, color="red", halo=True)
svg.circle(px(0.5), py(1), 3.5, fill="green")
svg.circle(px(0.5), py(-1), 3.5, fill="red")
svg.text(px(0.48), py(-1.5), "p = ½ 时 Â = ±1", size=11.5, anchor="end", color="ink", halo=True)
polyline(svg, [(px(0.03), py(-2.55)), (px(0.11), py(-2.55))], color="ink", width=2.4)
svg.text(px(0.13), py(-2.55) + 4, "除以 std（DeepSeekMath）", size=12, color="ink", halo=True)
polyline(svg, [(px(0.03), py(-3.1)), (px(0.11), py(-3.1))], color="ink", width=2.4, dash="5 4")
svg.text(px(0.13), py(-3.1) + 4, "不除 std（Dr. GRPO、V3.2）", size=12, color="ink", halo=True)

# ---------- right panel
QX0, QX1 = 440, 725


def qx(p):
    return QX0 + p * (QX1 - QX0)


def qy(v):
    return PY1 - v / 1.1 * (PY1 - PY0)


for v in (0, 0.25, 0.5, 0.75, 1.0):
    svg.line(QX0, qy(v), QX1, qy(v), color="grid", width=1)
    svg.text(QX0 - 6, qy(v) + 4, "%g" % v, size=11, anchor="end", color="muted")
for p in (0, 0.25, 0.5, 0.75, 1.0):
    svg.line(qx(p), PY0, qx(p), PY1, color="grid", width=1)
    svg.text(qx(p), PY1 + 16, "%g" % p, size=11, anchor="middle", color="muted")
svg.line(QX0, PY1, QX1, PY1, color="muted", width=1.1)
svg.text((QX0 + QX1) / 2, PY1 + 36, "组内正确率 p", size=12, anchor="middle", color="muted")
svg.text(QX0 - 30, 24, "每道题的总权重 (1/G)·Σ|Â_{i}|", size=13, weight="700")
polyline(svg, [(qx(p), qy(2 * math.sqrt(p * (1 - p)))) for p in ps], color="violet", width=2.4)
polyline(svg, [(qx(p), qy(2 * p * (1 - p))) for p in ps], color="violet", width=2.4, dash="5 4")
svg.text(qx(0.5), qy(1.0) - 8, "除以 std：2√(p(1−p))", size=12, anchor="middle", color="violet", halo=True)
svg.text(qx(0.5), qy(0.5) - 8, "不除 std：2p(1−p)", size=12, anchor="middle", color="violet", halo=True)
p = 1 / 64
svg.circle(qx(p), qy(2 * math.sqrt(p * (1 - p))), 3.5, fill="violet")
svg.text(qx(p) + 8, qy(2 * math.sqrt(p * (1 - p))) + 4, "0.248", size=11.5, color="violet", halo=True)
svg.circle(qx(p), qy(2 * p * (1 - p)), 3.5, fill="violet")
svg.text(qx(p) + 8, qy(2 * p * (1 - p)) - 2, "0.031", size=11.5, color="violet", halo=True)

# Dividing by std multiplies a question's total weight by 1/sqrt(p(1-p)): x2 at p = 1/2, x8.06 at
# p = 1/64. Relative to a medium question the hard one therefore gains 8.06 / 2 = 4.03x, not 8x.
amp_mid, amp_hard = 1 / math.sqrt(0.25), 1 / math.sqrt((1 / 64) * (63 / 64))
assert abs(amp_mid - 2) < 1e-12 and abs(amp_hard - 8.0632) < 1e-4
svg.text(16, H - 8, "除以 std 把每道题的总权重乘上 1/√(p(1−p))：p = ½ 时 ×2，p = 1/64 时 ×%.2f，"
         "所以相对中等题，极难题被放大约 %.0f 倍。" % (amp_hard, amp_hard / amp_mid),
         size=12, color="muted")
svg.save("l08-adv-vs-p.svg")
