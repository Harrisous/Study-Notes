"""Figure 8.11: process-supervision advantages are suffix sums of normalised step rewards.

Author-constructed example (research notes 4.2): output o1 has three steps with process rewards
[0.9, 0.8, 0.2], output o2 has two steps [0.6, 0.9]. All five step rewards are normalised together
(population std), and every token's advantage is the sum of the normalised rewards of the steps
that end at or after it (DeepSeekMath section 4.1.3). For contrast: outcome supervision with o1
wrong (0) and o2 right (1). Token counts per step are illustrative.
Writes deepseek/figures/l08-process-adv.svg and prints every number used in the lesson.
"""

import numpy as np

from l08_svg import SVG

steps = {"o₁": [0.9, 0.8, 0.2], "o₂": [0.6, 0.9]}
ntok = {"o₁": [5, 4, 3], "o₂": [4, 6]}
allr = np.array([r for v in steps.values() for r in v])
mu, sd = allr.mean(), allr.std()
print("mean %.4f pop std %.4f sample std %.4f" % (mu, sd, allr.std(ddof=1)))
norm = {k: [(r - mu) / sd for r in v] for k, v in steps.items()}
adv = {k: [sum(v[j:]) for j in range(len(v))] for k, v in norm.items()}
for k in steps:
    print(k, "norm", np.round(norm[k], 4), "adv", np.round(adv[k], 4))
outc = {"o₁": -1.0, "o₂": 1.0}     # outcome: rewards (0, 1), G = 2, population std -> -1, +1

W, H = 740, 400
svg = SVG(W, H, "过程监督下每个 token 的优势：后面各步归一化奖励之和")
CW, GAP = 36, 2
X0 = 110
STEP_COL = ["blue", "violet", "amber"]


def signed(v):
    return ("%+.2f" % v).replace("-", "−")


def row(y, key):
    svg.text(16, y + 44, key, size=15, weight="700")
    x = X0
    for j, (r, n) in enumerate(zip(steps[key], ntok[key])):
        w = n * (CW + GAP) - GAP
        # step header
        svg.text(x + w / 2, y - 22, "第 %d 步：PRM 分 %.1f" % (j + 1, r), size=12, anchor="middle", color="ink")
        svg.text(x + w / 2, y - 6, "归一化 %s" % signed(norm[key][j]), size=12, anchor="middle", color="muted")
        # token cells (process advantage)
        a = adv[key][j]
        col = "green" if a > 0 else "red"
        for t in range(n):
            xx = x + t * (CW + GAP)
            svg.rect(xx, y + 2, CW, 30, fill=col, fill_opacity=min(0.7, 0.12 + 0.3 * abs(a)), rx=3)
            svg.rect(xx, y + 2, CW, 30, stroke=STEP_COL[j], width=1.2, rx=3)
        svg.text(x + w / 2, y + 22, signed(a), size=13, anchor="middle", color="ink", weight="700",
                 halo=True)
        # outcome row
        oc = outc[key]
        ocol = "green" if oc > 0 else "red"
        for t in range(n):
            xx = x + t * (CW + GAP)
            svg.rect(xx, y + 40, CW, 22, fill=ocol, fill_opacity=0.25, rx=3)
        x += w + 14
    svg.text(x + 4, y + 22, "← 过程优势", size=12, color="ink")
    svg.text(x + 4, y + 56, "← 结果优势 %s" % signed(outc[key]), size=12, color="muted")


row(70, "o₁")
row(210, "o₂")

svg.text(16, 330, "整组 5 个步奖励一起归一化：均值 %.2f，总体标准差 %.4f。" % (mu, sd), size=12, color="ink")
svg.text(16, 350, "o₁ 第 1 步本身得分高（0.9），它的 token 优势却是负的：后缀和把后面那个坏步（0.2）也算了进来。",
         size=12, color="ink")
svg.text(16, 370, "结果监督（设 o₁ 答错、o₂ 答对，G = 2）：每条回答的所有 token 共用 −1 或 +1，分不出哪一步出了错。",
         size=12, color="muted")
svg.text(16, 390, "每步的 token 个数只是示意。", size=12, color="muted")
svg.save("l08-process-adv.svg")
