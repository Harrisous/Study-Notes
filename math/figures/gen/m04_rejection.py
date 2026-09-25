"""Figure M4.10: rejection sampling vs speculative sampling with the same proposal q.

Target p = (0.5, 0.3, 0.2), proposal q = (0.2, 0.5, 0.3) (the toy pair of lesson 4).
Left: rejection sampling with envelope M q, M = max p/q = 2.5, acceptance 1/M = 0.4.
Right: speculative sampling (Leviathan et al. 2023): accept min(1, p/q), acceptance
beta = sum min(p, q) = 0.7; on rejection draw from norm(max(0, p - q)) = (1, 0, 0).
Writes math/figures/m04-rejection.svg
"""

import numpy as np

from m04_svg import SVG, Axes

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])
M = float((p / q).max())
acc_rej = p / (M * q)
beta = float(np.minimum(p, q).sum())
resid = np.maximum(p - q, 0)
resid_n = resid / resid.sum()
acc_spec = np.minimum(1, p / q)
out = np.minimum(p, q) + (1 - beta) * resid_n
assert np.allclose(out, p)
print("M", M, "acc", acc_rej, "P(acc)", 1 / M, "beta", beta, "resid", resid_n, "acc spec", acc_spec)

W, H = 720, 380
svg = SVG(W, H, "拒绝采样与投机采样：同一个提议分布 q，两种接受规则")

# ---- left: rejection sampling
A = Axes(svg, 60, 58, 250, 200, (-0.6, 2.6), (0, 1.3))
A.frame(xticks=[], yticks=[0, 0.25, 0.5, 0.75, 1.0, 1.25], ylabel="概率 / 包络高度", ylabel_dx=-40)
svg.text(A.x0 - 20, 24, "拒绝采样：包络 M·q，M = 2.5", size=13, weight="600")
bw = 44
for i in range(3):
    x = A.X(i)
    env = M * q[i]
    svg.rect(x - bw / 2, A.Y(env), bw, A.Y(p[i]) - A.Y(env), fill="red", fill_opacity=0.14,
             stroke="red", width=1.2, dash="4 3")
    svg.rect(x - bw / 2, A.Y(p[i]), bw, A.Y(0) - A.Y(p[i]), fill="green", fill_opacity=0.45,
             stroke="green", width=1.2)
    svg.text(x, A.Y(env) - 6, "Mq=%.2f" % env, size=11, anchor="middle", color="red")
    svg.text(x, A.Y(p[i] / 2) + 4, "p=%.1f" % p[i], size=11, anchor="middle", color="ink")
    svg.text(x, A.Y(0) + 16, "x = %d" % i, size=12, anchor="middle", color="muted")
    svg.text(x, A.Y(0) + 32, "接受 %.2f" % acc_rej[i], size=11, anchor="middle", color="green")
svg.text(A.x0 - 20, A.Y(0) + 56, "绿 = 接受的面积（恰为 p），红虚框 = 被拒的面积", size=12, color="ink")
svg.text(A.x0 - 20, A.Y(0) + 74, "P(接受) = 1/M = 0.4，平均要试 M = 2.5 次", size=12, color="red")

# ---- right: speculative sampling
B = Axes(svg, 420, 58, 280, 200, (-0.6, 2.6), (0, 0.6))
B.frame(xticks=[], yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
svg.text(B.x0 - 20, 24, "投机采样：接受 min(1, p/q)", size=13, weight="600")
for i in range(3):
    x = B.X(i)
    lo = min(p[i], q[i])
    # overlap min(p, q): accepted directly
    svg.rect(x - bw / 2, B.Y(lo), bw, B.Y(0) - B.Y(lo), fill="green", fill_opacity=0.45,
             stroke="green", width=1.2)
    if p[i] > q[i]:   # residual mass, supplied by the resample
        svg.rect(x - bw / 2, B.Y(p[i]), bw, B.Y(lo) - B.Y(p[i]), fill="orange", fill_opacity=0.35,
                 stroke="orange", width=1.2)
        svg.text(x + bw / 2 + 4, B.Y((p[i] + lo) / 2) + 4, "p−q", size=11, color="orange")
    if q[i] > p[i]:   # proposals that get rejected
        svg.rect(x - bw / 2, B.Y(q[i]), bw, B.Y(lo) - B.Y(q[i]), fill="red", fill_opacity=0.14,
                 stroke="red", width=1.2, dash="4 3")
        svg.text(x + bw / 2 + 4, B.Y((q[i] + lo) / 2) + 4, "q−p", size=11, color="red")
    svg.text(x, B.Y(max(p[i], q[i])) - 6, "p=%.1f q=%.1f" % (p[i], q[i]), size=11, anchor="middle")
    svg.text(x, B.Y(0) + 16, "x = %d" % i, size=12, anchor="middle", color="muted")
    svg.text(x, B.Y(0) + 32, "接受 %s" % ("1" if acc_spec[i] >= 1 else "%.3g" % acc_spec[i]), size=11,
             anchor="middle", color="green")
rej = q - np.minimum(p, q)
svg.text(B.x0 - 20, B.Y(0) + 56, "绿 = min(p, q)，直接接受，合计 β = %.1f" % beta, size=12, color="green")
svg.text(B.x0 - 20, B.Y(0) + 74, "红 = 被拒的提议，合计 %.1f + %.1f = %.1f = 1 − β" % (rej[1], rej[2], rej.sum()),
         size=12, color="red")
svg.text(B.x0 - 20, B.Y(0) + 92, "橙 = 拒绝后从 max(0, p−q) 补采，合计也是 %.1f" % resid.sum(), size=12,
         color="orange")
svg.text(B.x0 - 20, B.Y(0) + 110, "红色的质量整块搬到橙色：绿 + 橙 = p", size=12, color="ink")

svg.save("m04-rejection.svg")
