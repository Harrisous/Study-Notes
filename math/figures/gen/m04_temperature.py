"""Figure M4.7: temperature reshapes a distribution without changing the ranking.

Toy logits z = (2, 1, 0.5, 0, -1) (the same as lesson 4). Left: p_tau for tau = 0.5, 1, 2.
Right: entropy in bits against tau (log axis), from 0 (greedy) to log2(5) (uniform).
Writes math/figures/m04-temperature.svg
"""

import math

import numpy as np

from m04_svg import SVG, Axes

z = np.array([2.0, 1.0, 0.5, 0.0, -1.0])


def soft(z, tau):
    a = z / tau
    e = np.exp(a - a.max())
    return e / e.sum()


def H(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


TAUS = (0.5, 1.0, 2.0)
OPA = {0.5: 0.85, 1.0: 0.45, 2.0: 0.16}
P = {t: soft(z, t) for t in TAUS}
for t in TAUS:
    print(t, np.round(P[t], 4), round(H(P[t]), 3), round(2 ** H(P[t]), 2))

W, H_ = 720, 350
svg = SVG(W, H_, "温度改变分布的形状：同一组 logits 在三种温度下的分布与熵")

A = Axes(svg, 60, 58, 350, 210, (-0.5, 4.5), (0, 0.9))
A.frame(xticks=[], yticks=[0, 0.2, 0.4, 0.6, 0.8], ylabel="概率", ylabel_dx=-40)
svg.text(A.x0, 26, "logits z = (2, 1, 0.5, 0, −1) 在三种温度下", size=13, weight="600")
bw = 20
for i in range(5):
    for j, t in enumerate(TAUS):
        x = A.X(i) + (j - 1) * (bw + 3)
        v = float(P[t][i])
        svg.rect(x - bw / 2, A.Y(v), bw, A.Y(0) - A.Y(v), fill="blue", fill_opacity=OPA[t],
                 stroke="blue", width=1)
        if i == 0 or i == 4:
            svg.text(x, A.Y(v) - 5, ("%.2f" % v).lstrip("0") if v < 1 else "1", size=11,
                     anchor="middle", color="blue")
    svg.text(A.X(i), A.Y(0) + 16, "z = %g" % z[i], size=11, anchor="middle", color="muted")
svg.text(A.x0 + A.w / 2, A.Y(0) + 34, "token（按 logit 从大到小）", size=12, anchor="middle")
# legend
lx, ly = A.x0 + 190, 70
for j, t in enumerate(TAUS):
    svg.rect(lx, ly + j * 20 - 10, 14, 12, fill="blue", fill_opacity=OPA[t], stroke="blue", width=1)
    svg.text(lx + 20, ly + j * 20, "τ = %g，熵 %.3f bit" % (t, H(P[t])), size=12)

# right: entropy vs tau
B = Axes(svg, 490, 58, 210, 210, (0.05, 100), (0, 2.6), logx=True)
B.frame(xticks=[0.1, 1, 10, 100], yticks=[0, 0.5, 1, 1.5, 2, 2.5], xlabel="温度 τ（对数轴）",
        ylabel="熵（bit）", ylabel_dx=-36)
svg.text(B.x0 - 20, 26, "熵随温度的变化", size=13, weight="600")
ts = np.logspace(math.log10(0.05), 2, 300)
B.curve(ts, [H(soft(z, t)) for t in ts], stroke="blue", width=2.2)
hmax = math.log2(5)
svg.line(B.x0, B.Y(hmax), B.x0 + B.w, B.Y(hmax), color="green", width=1.4, dash="5 4")
svg.text(B.X(0.07), B.Y(hmax) - 6, "均匀：log₂5 = 2.32", size=12, color="green")
for t in TAUS:
    svg.circle(*B.P(t, H(P[t])), 4, fill="blue")
svg.text(B.X(0.5) + 6, B.Y(H(P[0.5])) + 14, "0.88", size=11, color="blue")
svg.text(B.X(1.0) + 6, B.Y(H(P[1.0])) + 14, "1.74", size=11, color="blue")
svg.text(B.X(2.0) + 6, B.Y(H(P[2.0])) + 14, "2.15", size=11, color="blue")
svg.text(B.X(0.06), B.Y(1.95), "τ → 0：熵 → 0", size=12, color="red")
svg.text(B.X(0.06), B.Y(1.95) + 16, "（贪心）", size=12, color="red")

svg.text(60, 330, "温度不改变排名：每种温度下都是第一个 token 最大；只改变“有多确定”。", size=12, color="muted")
svg.save("m04-temperature.svg")
