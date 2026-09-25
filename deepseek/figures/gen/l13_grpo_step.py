"""Figure 13.3: one GRPO group, computed. G = 16 answers to one question, 4 correct, all well formatted.

Rewards r = acc + format = 2 (correct) or 1 (wrong); mean 1.25, population std 0.4330;
advantages +1.7321 (correct) and -0.5774 (wrong). Every number below is computed here.
Writes deepseek/figures/l13-grpo-step.svg
"""

import numpy as np

from l13_svg import SVG

G = 16
correct = {3, 7, 8, 13}                       # 1-based positions of the correct answers
acc = np.array([1.0 if i + 1 in correct else 0.0 for i in range(G)])
fmt_r = np.ones(G)
r = acc + fmt_r
mean, std = r.mean(), r.std()                  # population std (ddof = 0)
adv = (r - mean) / std
std1 = r.std(ddof=1)
adv1 = (r - mean) / std1
assert abs(adv.sum()) < 1e-12 and abs((adv ** 2).sum() - G) < 1e-9

W, H = 760, 440
svg = SVG(W, H, "一组 GRPO 的逐项计算：16 个回答、4 个答对，从奖励到组内优势")

X0, CW, GAP = 104, 36, 4.5
def cx(i):
    return X0 + i * (CW + GAP) + CW / 2

svg.text(16, 22, "一道题、16 个回答：4 个答对，全部把思考放进了 <think> 标签", size=13, weight="600")
# row labels
svg.text(16, 56, "回答 o_{i}", size=12, color="muted")
svg.text(16, 108, "r_{acc} + r_{format}", size=12, color="muted")
svg.text(16, 134, "= 奖励 r_{i}", size=12, color="muted")
svg.text(16, 288, "优势 Â_{i}", size=12, color="muted")

for i in range(G):
    x = cx(i)
    ok = acc[i] == 1
    col = "green" if ok else "red"
    svg.rect(x - CW / 2, 36, CW, 30, fill=col, fill_opacity=0.10, stroke=col, width=1.2, rx=6)
    svg.text(x, 56, str(i + 1), size=12, anchor="middle", color="ink")
    if ok:
        svg.check(x + 1, 80, s=5.5, color="green", width=2)
    else:
        svg.cross(x, 80, s=4.5, color="red", width=1.8)
    svg.text(x, 108, "%d+1" % int(acc[i]), size=12, anchor="middle", color="muted")
    svg.text(x, 132, "%d" % int(r[i]), size=14, anchor="middle", color=col, weight="700")

# statistics band
svg.rect(96, 146, 648, 44, fill="amber", fill_opacity=0.08, stroke="amber", width=1, rx=8)
svg.text(420, 165, "组内均值 = (4 × 2 + 12 × 1) / 16 = %.2f；总体标准差 = √(0.25 × 0.75) = %.3f" % (mean, std),
         size=12.5, anchor="middle")
svg.text(420, 182, "Â_{i} = (r_{i} − %.2f) / %.3f ：答对的 (2 − 1.25)/0.433，答错的 (1 − 1.25)/0.433" % (mean, std),
         size=12, anchor="middle", color="muted")

# advantage bars
Y0 = 288
S = 34.0                                       # pixels per unit of advantage
svg.line(96, Y0, 744, Y0, color="muted", width=1.1)
for v in (-1, 1, 2):
    y = Y0 - v * S
    svg.line(96, y, 744, y, color="grid", width=1)
    svg.text(92, y + 4, ("%+d" % v).replace("-", "−") if v else "0", size=11, anchor="end", color="muted")
svg.text(92, Y0 + 4, "0", size=11, anchor="end", color="muted")
for i in range(G):
    x = cx(i)
    v = adv[i]
    col = "green" if v > 0 else "red"
    top = Y0 - max(v, 0) * S
    h = abs(v) * S
    svg.rect(x - 13, top, 26, h, fill=col, fill_opacity=0.75)
first_ok = min(correct) - 1
first_bad = 0
svg.text(cx(first_ok), Y0 - adv[first_ok] * S - 6, "+%.3f" % adv[first_ok], size=12, anchor="middle",
         color="green", weight="700")
svg.text(cx(first_bad), Y0 - adv[first_bad] * S + 16, "−%.3f" % -adv[first_bad], size=12, anchor="middle",
         color="red", weight="700")

# checks
svg.rect(16, 338, 728, 90, fill="surface", stroke="rule", width=1, rx=8)
svg.text(30, 360, "检查 ①  Σ Â_{i} = 4 × (+%.3f) + 12 × (−%.3f) = %.0f：比平均好的与比平均差的正好抵消"
         % (adv[first_ok], -adv[first_bad], abs(adv.sum())), size=12.5)
svg.text(30, 383, "检查 ②  Σ Â_{i}^{2} = 4 × 3 + 12 × (1/3) = %.0f = G：标准化让一组的“总信号强度”固定为 G"
         % (adv ** 2).sum(), size=12.5)
svg.text(30, 406, "若改用样本标准差（分母 G − 1 = 15）：σ = %.3f，Â 变成 +%.3f 与 −%.3f，比例不变"
         % (std1, adv1[first_ok], -adv1[first_bad]), size=12.5, color="muted")
svg.save("l13-grpo-step.svg")
