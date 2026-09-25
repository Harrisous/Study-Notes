"""Figure M5.8: sequence KL = per-token KL along the policy's own trajectory, and what a single-sample estimate uses.

A toy 5-token vocabulary {我, 你, 很, 好, 。}. The current policy q (= pi_theta) samples the answer 你 很 好 。 one token
at a time. At each position the figure draws the full next-token distributions of q (blue) and the reference p
(= pi_ref, grey), the exact per-position KL(q||p) (a sum over the whole vocabulary), and the single-sample values
k1 = -ln w, k3 = w - 1 - ln w that use only the sampled token (w = p/q of that token). The bottom band adds them up
along this one answer. Every number is computed here from the distributions below (lesson (M5.16), (M5.18)).
Writes math/figures/m05-seq-kl.svg
"""

import math

from m05_svg import SVG

VOCAB = ["我", "你", "很", "好", "。"]
# (prefix shown, q = current policy, p = reference, index of the sampled token)
STEPS = [
    ("（空）", [0.15, 0.60, 0.05, 0.10, 0.10], [0.40, 0.40, 0.05, 0.10, 0.05], 1),
    ("你", [0.05, 0.05, 0.50, 0.35, 0.05], [0.05, 0.05, 0.65, 0.20, 0.05], 2),
    ("你很", [0.02, 0.03, 0.05, 0.80, 0.10], [0.10, 0.05, 0.02, 0.80, 0.03], 3),
    ("你很好", [0.03, 0.02, 0.05, 0.05, 0.85], [0.05, 0.05, 0.10, 0.10, 0.70], 4),
]


def kl(q, p):
    return sum(a * math.log(a / b) for a, b in zip(q, p) if a > 0)


W, H = 760, 450
svg = SVG(W, H, "序列 KL 的逐 token 分解：精确计算要整张分布，单样本估计只看采样到的词")

# legend strip
svg.rect(40, 12, 14, 12, fill="blue", fill_opacity=0.75)
svg.text(60, 23, "当前策略 q = π_{θ}（采样用）", size=12, color="blue")
svg.rect(250, 12, 14, 12, fill="muted", fill_opacity=0.35, stroke="muted", width=1)
svg.text(270, 23, "参考模型 p = π_{ref}", size=12, color="muted")
svg.text(720, 23, "每一列：给定前缀时，下一个 token 的分布", size=12, anchor="end", color="muted")

BASE, TOP = 196, 86          # bar baseline and the y of probability 1.0
SCALE = BASE - TOP
COLW, X0 = 176, 44

tot_kl = tot_k1 = tot_k3 = 0.0
for i, (prefix, q, p, s) in enumerate(STEPS):
    cx = X0 + i * COLW
    k = kl(q, p)
    w = p[s] / q[s]
    k1 = -math.log(w)
    k3 = w - 1 - math.log(w)
    tot_kl += k
    tot_k1 += k1
    tot_k3 += k3

    # column frame and header
    svg.rect(cx, 38, COLW - 12, 300, fill="surface", stroke="rule", width=1, rx=8)
    svg.text(cx + 10, 58, "t = %d" % (i + 1), size=13, weight="600")
    svg.text(cx + COLW - 22, 58, "前缀：" + prefix, size=12, anchor="end", color="muted")

    # probability axis (ticks on every column so each chart reads alone)
    svg.line(cx + 10, BASE, cx + COLW - 22, BASE, color="muted", width=1)
    for v in (0.5, 1.0):
        y = BASE - v * SCALE
        svg.line(cx + 10, y, cx + COLW - 22, y, color="grid", width=1)
    if i == 0:
        svg.text(cx - 4, BASE + 4, "0", size=11, anchor="end", color="muted")
        svg.text(cx - 4, BASE - 0.5 * SCALE + 4, "0.5", size=11, anchor="end", color="muted")
        svg.text(cx - 4, TOP + 4, "1", size=11, anchor="end", color="muted")

    slot = (COLW - 32) / len(VOCAB)
    for j, tok in enumerate(VOCAB):
        sx = cx + 12 + j * slot
        if j == s:
            svg.rect(sx - 1, TOP - 14, slot - 2, BASE - TOP + 14 + 26, fill="blue", fill_opacity=0.08,
                     stroke="blue", width=1, rx=4)
            svg.text(sx + slot / 2 - 2, TOP - 3, "采样", size=12, anchor="middle", color="blue", weight="600")
        bw = (slot - 8) / 2
        hq, hp = q[j] * SCALE, p[j] * SCALE
        svg.rect(sx + 2, BASE - hq, bw, hq, fill="blue", fill_opacity=0.75)
        svg.rect(sx + 2 + bw + 1, BASE - hp, bw, hp, fill="muted", fill_opacity=0.35, stroke="muted", width=1)
        svg.text(sx + slot / 2 - 2, BASE + 18, tok, size=13, anchor="middle",
                 color="blue" if j == s else "ink", weight="600" if j == s else None)

    # numbers
    y = 244
    svg.text(cx + 10, y, "精确 KL = %.3f" % k, size=13, weight="600")
    svg.text(cx + 10, y + 17, "对 5 个词求和", size=12, color="muted")
    svg.text(cx + 10, y + 41, "w = p/q = %.2f/%.2f = %.2f" % (p[s], q[s], w), size=12)
    k1s = "%.3f" % (0.0 if abs(k1) < 5e-4 else k1)
    svg.text(cx + 10, y + 60, "k1 = %s" % k1s.replace("-", "−"), size=12, color="red", weight="600")
    svg.text(cx + 88, y + 60, "k3 = %.3f" % k3, size=12, color="blue", weight="600")
    svg.text(cx + 10, y + 78, "只用采样到的那一个词", size=12, color="muted")

    # arrow: the sampled token becomes part of the next prefix
    if i < len(STEPS) - 1:
        svg.arrow(cx + COLW - 12, 52, cx + COLW + 2, 52, color="muted", width=1.3)

# bottom band: add up along this one answer
BY = 352
svg.rect(X0, BY, 4 * COLW - 12, 84, fill="muted", fill_opacity=0.06, stroke="rule", width=1, rx=8)
svg.text(X0 + 14, BY + 24, "沿这条回答“你 很 好 。”相加：", size=13, weight="600")
svg.text(X0 + 14, BY + 48, "Σ 精确 KL = %.3f" % tot_kl, size=13, weight="600")
svg.text(X0 + 14, BY + 68, "每步要参考模型的整张分布", size=12, color="muted")
svg.text(X0 + 250, BY + 48, "Σ k1 = %.3f" % tot_k1, size=13, color="red", weight="600")
svg.text(X0 + 250, BY + 68, "可正可负，第 2 步是负的", size=12, color="muted")
svg.text(X0 + 452, BY + 48, "Σ k3 = %.3f" % tot_k3, size=13, color="blue", weight="600")
svg.text(X0 + 452, BY + 68, "每步只要两个概率；第 3 步为 0", size=12, color="muted")

svg.save("m05-seq-kl.svg")
