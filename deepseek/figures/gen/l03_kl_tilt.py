"""Figure 3.4: the KL-regularised optimum pi* = pi_ref * exp(r/beta) / Z on a 3-answer toy problem.

pi_ref = [0.6, 0.3, 0.1], rewards r = [0, 1, 2] (toy numbers from the lesson-03 research notes, sec. 4.5).
For each beta we print E[r], KL(pi*||pi_ref) and the optimal objective, which equals beta*log Z.
Writes deepseek/figures/l03-kl-tilt.svg
"""
import math
from l03_svg import SVG

pref = [0.6, 0.3, 0.1]
r = [0.0, 1.0, 2.0]


def opt(beta):
    w = [p * math.exp(x / beta) for p, x in zip(pref, r)]
    Z = sum(w)
    pi = [v / Z for v in w]
    Er = sum(p * x for p, x in zip(pi, r))
    KL = sum(p * math.log(p / q) for p, q in zip(pi, pref))
    return pi, Er, KL, beta * math.log(Z), Z


rows = [("π_{ref}", pref, sum(p * x for p, x in zip(pref, r)), 0.0, None)]
for b in (2, 1, 0.5, 0.25):
    pi, Er, KL, obj, Z = opt(b)
    assert abs(obj - (Er - b * KL)) < 1e-9
    rows.append(("β = %s" % b, pi, Er, KL, obj))
# numbers quoted in the lesson text
assert [round(v, 4) for v in opt(1)[0]] == [0.2785, 0.3785, 0.343]

W, H = 740, 330
svg = SVG(W, H, "KL 约束下的最优策略：参考分布按 exp(r/β) 倾斜")
x0, bw = 96, 400
op = [0.10, 0.30, 0.62]
svg.text(16, 24, "三个回答的奖励 r = 0、1、2；每一行是一个概率分布（总长 = 1）", size=13.5, weight="600")
for k, lab in enumerate(["回答 1（r = 0）", "回答 2（r = 1）", "回答 3（r = 2）"]):
    lx = x0 + k * 140
    svg.rect(lx, 38, 16, 12, fill="green", fill_opacity=op[k], stroke="green", width=1)
    svg.text(lx + 22, 49, lab, size=12)
hx = [548, 616, 690]
for t, x in zip(["E[r]", "KL", "目标值"], hx):
    svg.text(x, 80, t, size=12.5, anchor="middle", weight="600", color="muted")
y = 92
for lab, pi, Er, KL, obj in rows:
    svg.text(x0 - 10, y + 22, lab, size=13, anchor="end", weight="600")
    cx = x0
    for j, p in enumerate(pi):
        w = bw * p
        svg.rect(cx, y, w, 32, fill="green", fill_opacity=op[j], stroke="green", width=1)
        if w > 40:
            svg.text(cx + w / 2, y + 21, "%.3f" % p, size=12, anchor="middle")
        elif w > 3:
            svg.text(cx + w / 2 + 14, y - 3, "%.3f" % p, size=11, anchor="middle", color="muted")
        cx += w
    svg.text(hx[0], y + 21, "%.3f" % Er, size=12.5, anchor="middle")
    svg.text(hx[1], y + 21, "%.3f" % KL, size=12.5, anchor="middle")
    svg.text(hx[2], y + 21, "—" if obj is None else "%.3f" % obj, size=12.5, anchor="middle")
    y += 44
svg.text(16, y + 12, "β 越小，概率越往高奖励的回答倾斜。目标值 = E[r] − β·KL，恰好等于 β·log Z（Z 是归一化常数）。", size=12, color="muted")
svg.save("l03-kl-tilt.svg")
