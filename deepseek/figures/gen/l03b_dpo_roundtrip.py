"""Figure 3.15 of the blackboard version of lesson 3: reading the reward back from the optimal policy.

The three-answer example of section 3: pi_ref = [0.6, 0.3, 0.1], r = [0, 1, 2], beta = 1, so
pi* = pi_ref * exp(r / beta) / Z with Z = 2.1544. Reading back beta * ln(pi* / pi_ref) gives r - beta ln Z:
every reward moves down by the same 0.7675, the gaps stay 1 and 1, and so every Bradley-Terry probability
is unchanged. All numbers are computed here.
Writes deepseek/figures/l03b-dpo-roundtrip.svg
"""

import math

from l03_svg import SVG, Axis

PREF = [0.6, 0.3, 0.1]
R = [0.0, 1.0, 2.0]
BETA = 1.0
w = [p * math.exp(r / BETA) for p, r in zip(PREF, R)]
Z = sum(w)
PSTAR = [x / Z for x in w]
RATIO = [ps / pr for ps, pr in zip(PSTAR, PREF)]
BACK = [BETA * math.log(x) for x in RATIO]
assert round(Z, 4) == 2.1544 and [round(p, 4) for p in PSTAR] == [0.2785, 0.3785, 0.3430]
assert [round(x, 4) for x in RATIO] == [0.4642, 1.2617, 3.4298]
assert [round(x, 4) for x in BACK] == [-0.7675, 0.2325, 1.2325]
assert all(abs((b - r) + math.log(Z)) < 1e-12 for b, r in zip(BACK, R))
SIG1 = 1 / (1 + math.exp(-1))
assert round(SIG1, 4) == 0.7311


def f4(x):
    return ("%.4f" % x).replace("-", "−")


W, H = 740, 330
svg = SVG(W, H, "黑板上倒着读：从最优策略读回的奖励只差一个常数")

# ---------------- left: pi_ref and pi* as paired bars
svg.text(16, 26, "§3 的三个回答（β = 1）", size=13.5, weight="700")
base = 214
scale = 200.0          # px per unit probability
BW = 30
for k in range(3):
    gx = 40 + k * 100
    for j, (p, col, op) in enumerate(((PREF[k], "muted", 0.2), (PSTAR[k], "green", 0.3))):
        h = p * scale
        x = gx + j * (BW + 4)
        svg.rect(x, base - h, BW, h, fill=col, fill_opacity=op, stroke=col, width=1.1)
        svg.text(x + BW / 2, base - h - 6, ("%.4f" % p).rstrip("0").rstrip(".") if j == 0 else "%.4f" % p,
                 size=11, anchor="middle", color=col if j == 1 else "muted")
    cx = gx + BW + 2
    svg.text(cx, base + 18, "回答 %d（r = %d）" % (k + 1, int(R[k])), size=12, anchor="middle")
    svg.text(cx, base + 36, "放大 ×%s" % f4(RATIO[k]), size=12, anchor="middle", color="blue", weight="600")
svg.line(30, base, 330, base, color="muted", width=1.1)
svg.rect(40, 262, 12, 12, fill="muted", fill_opacity=0.2, stroke="muted", width=1)
svg.text(58, 272, "参考 π_{ref}", size=12)
svg.rect(150, 262, 12, 12, fill="green", fill_opacity=0.3, stroke="green", width=1)
svg.text(168, 272, "最优 π^{*}", size=12)
svg.text(40, 298, "放大倍数 = π^{*} ÷ π_{ref} = e^{r} ÷ Z", size=12, color="muted")

# ---------------- right: true rewards vs rewards read back
svg.line(350, 40, 350, 312, color="rule", width=1)
ax = Axis(-1.0, 2.3, 480, 716)
svg.text(366, 26, "两排奖励：整排平移，间隔不变", size=13.5, weight="700")
ROWS = [(96, ("真正的奖励", "r"), R, "green"), (206, ("从 π^{*} 读回", "β·ln(π^{*}/π_{ref})"), BACK, "blue")]
for y, label, vals, col in ROWS:
    svg.text(366, y - 4, label[0], size=12.5, weight="600", color=col)
    svg.text(366, y + 14, label[1], size=12.5, weight="600", color=col)
    svg.line(ax(-1.0), y, ax(2.3), y, color="muted", width=1)
    for t in (-1, 0, 1, 2):
        svg.line(ax(t), y - 4, ax(t), y + 4, color="muted", width=1)
        svg.text(ax(t), y + 18, str(t).replace("-", "−"), size=11, anchor="middle", color="muted")
    for v in vals:
        svg.circle(ax(v), y, 6, fill=col, fill_opacity=0.35, stroke=col, width=1.4)
    for a, b in ((vals[0], vals[1]), (vals[1], vals[2])):
        xa, xb = ax(a) + 8, ax(b) - 8
        svg.path("M%.1f %.1f v-6 H%.1f v6" % (xa, y - 12, xb), stroke=col, width=1)
        svg.text((xa + xb) / 2, y - 22, "差 %d" % round(b - a), size=11.5, anchor="middle", color=col)
    if col == "blue":
        for v in vals:
            svg.text(ax(v), y + 34, f4(v), size=11.5, anchor="middle", color=col, weight="600")
# shift arrows
for r, b in zip(R, BACK):
    svg.arrow(ax(r), 96 + 8, ax(b), 206 - 8, color="muted", width=1.2, dash="4 3", shorten=2)
mid = (ax(1.0) + ax(BACK[1]) + ax(2.0) + ax(BACK[2])) / 4      # between the 2nd and 3rd arrow
svg.text(mid, 146, "下移 ln Z", size=11.5, anchor="middle", color="muted")
svg.text(mid, 162, "= %s" % f4(math.log(Z)), size=11.5, anchor="middle", color="muted", weight="600")
svg.text(366, 272, "差值一样，所以 Bradley–Terry 的胜率一样：", size=12.5)
svg.text(366, 292, "回答 2 胜回答 1 的概率都是 σ(1) = %s" % f4(SIG1), size=12.5, weight="600")
svg.save("l03b-dpo-roundtrip.svg")
