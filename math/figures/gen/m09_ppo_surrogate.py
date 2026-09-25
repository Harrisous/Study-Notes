"""Figure M9.9 (first figure of §6): why PPO clips. Toy model, a batch of only two answers, surrogate objectives vs the truth.

Toy model of M09 (prompt "2+3=", tokens 5/6 then ！/吧; pi = 0.6, 0.7, 0.5; returns 1, 0.5, 0, 0.2).
Batch: "5吧" (R = 0.5) and "6吧" (R = 0.2); baseline = batch mean 0.35, so every token of the first
answer has advantage +0.15 and every token of the second -0.15. We move the six logits along the
(unit) gradient of the empirical surrogate L_CPI at theta_old, step alpha, and evaluate exactly:
  true improvement J(theta) - J(theta_old), empirical L_CPI, empirical L_CLIP (eps = 0.2).
All numbers: math/reference-docs/m09-notes.md §2.5. Writes math/figures/m09-ppo-surrogate.svg
"""

import math

import numpy as np

from m09_svg import SVG

EPS = 0.2
z0 = np.array([math.log(0.6), math.log(0.4), math.log(0.7), math.log(0.3), math.log(0.5), math.log(0.5)])


def sm(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def probs(z):
    return {"q": sm(z[0:2]), "q5": sm(z[2:4]), "q6": sm(z[4:6])}


def J(z):
    p = probs(z)
    return p["q"][0] * (p["q5"][0] * 1 + p["q5"][1] * 0.5) + p["q"][1] * (p["q6"][0] * 0 + p["q6"][1] * 0.2)


batch = [(0, 1, 0.5), (1, 1, 0.2)]         # (first token, second token, return)
base = sum(b[2] for b in batch) / len(batch)
toks = []
for a1, a2, R in batch:
    toks += [("q", a1, R - base), ("q5" if a1 == 0 else "q6", a2, R - base)]
P0 = probs(z0)


def L(z, clip):
    p = probs(z)
    tot = 0.0
    for s, a, A in toks:
        rho = p[s][a] / P0[s][a]
        tot += min(rho * A, min(max(rho, 1 - EPS), 1 + EPS) * A) if clip else rho * A
    return tot / len(toks)


def ngrad(f, z, h=1e-6):
    return np.array([(f(z + h * e) - f(z - h * e)) / (2 * h) for e in np.eye(6)])


u = ngrad(lambda z: L(z, False), z0)
u /= np.linalg.norm(u)
J0 = J(z0)
alphas = np.linspace(0, 6, 601)
dJ = np.array([J(z0 + a * u) - J0 for a in alphas])
Lc = np.array([L(z0 + a * u, False) for a in alphas])
Lk = np.array([L(z0 + a * u, True) for a in alphas])
i_plat = int(np.argmax(Lk > Lk.max() - 1e-9))
a_plat = alphas[i_plat]
i_best = int(np.argmax(dJ))
i_cross = int(np.argmax((alphas > 1) & (dJ < 0)))
sup_cpi = L(z0 + 60 * u, False)
lim_dJ = J(z0 + 60 * u) - J0
assert abs(a_plat - 0.76) < 0.02 and abs(dJ[i_plat] - 0.0737) < 0.002
assert abs(sup_cpi - 0.1875) < 1e-4 and abs(lim_dJ + 0.05) < 1e-4

W, H = 720, 380
svg = SVG(W, H, "只有两条回答的批次上：替代目标、裁剪目标与真实目标")
X0, X1, Y0, Y1 = 70, 500, 30, 300
AMIN, AMAX, VMIN, VMAX = 0.0, 6.0, -0.06, 0.20


def px(a):
    return X0 + (a - AMIN) / (AMAX - AMIN) * (X1 - X0)


def py(v):
    return Y1 - (v - VMIN) / (VMAX - VMIN) * (Y1 - Y0)


for v in (-0.05, 0, 0.05, 0.10, 0.15, 0.20):
    y = py(v)
    svg.line(X0, y, X1, y, color="grid" if v != 0 else "muted", width=1 if v != 0 else 1.3)
    svg.text(X0 - 8, y + 4, ("%.2f" % v).replace("-", "−"), size=11, anchor="end", color="muted")
for a in range(0, 7):
    x = px(a)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, str(a), size=11, anchor="middle", color="muted")
svg.rect(X0, Y0, X1 - X0, Y1 - Y0, stroke="muted", width=1.1)
svg.text((X0 + X1) / 2, Y1 + 40, "沿批次替代目标的梯度方向走的步长 α（6 个 logit 一起动）", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "相对起点的变化", size=12, anchor="middle", color="muted", rotate=-90)


def poly(ys):
    return "M" + " L".join("%.1f %.1f" % (px(a), py(v)) for a, v in zip(alphas, ys))


svg.path(poly(Lc), stroke="blue", width=2.4, dash="7 4")
svg.path(poly(Lk), stroke="green", width=2.6)
svg.path(poly(dJ), stroke="ink", width=2.6)

# annotations
xp, yp = px(a_plat), py(Lk[i_plat])
svg.circle(xp, yp, 4.5, fill="green")
svg.line(xp, yp, xp, py(dJ[i_plat]), color="green", width=1, dash="3 3")
svg.circle(xp, py(dJ[i_plat]), 4.5, fill="ink")
# both labels sit under the zero line, where no curve passes for alpha < 3.7
svg.line(xp, py(0) + 2, xp, py(-0.012), color="green", width=1, dash="2 2")
svg.text(xp + 4, py(-0.021), "α ≈ %.2f：裁剪目标变平，梯度从此为 0" % a_plat, size=12, color="green", weight="600")
svg.text(xp + 4, py(-0.021) + 17, "此时真实改进 +%.3f" % dJ[i_plat], size=12, weight="600")
# the peak label sits above-left, clear of both rising curves, with a leader line
svg.circle(px(alphas[i_best]), py(dJ[i_best]), 3.5, fill="ink")
svg.line(px(alphas[i_best]), py(0.140), px(alphas[i_best]), py(dJ[i_best]) - 5, color="ink", width=0.9)
svg.text(px(0.12), py(0.146), "真实改进最高 +%.3f（α ≈ %.2f）" % (dJ[i_best], alphas[i_best]), size=12)
svg.circle(px(alphas[i_cross]), py(0), 3.5, fill="red")
svg.text(px(alphas[i_cross]) + 6, py(0) - 8, "α ≈ %.1f 起比原来还差" % alphas[i_cross], size=12, color="red")
svg.text(px(5.95), py(dJ[-1]) + 18, "α→∞ 时趋于 −0.050", size=12, anchor="end", color="red", weight="600")
svg.text(px(4.6), py(sup_cpi) + 2, "α→∞：0.1875", size=12, anchor="end", color="blue", weight="600")

LX = 516
svg.text(LX, Y0 + 8, "批次只有两条回答：", size=12, weight="600")
svg.text(LX, Y0 + 26, "“5吧” R = 0.5，“6吧” R = 0.2", size=12, weight="600")
svg.text(LX, Y0 + 44, "基线 = 批次均值 0.35，优势 ±0.15", size=12, color="muted")
y = Y0 + 74
for col, dash, name, sub in (("blue", "7 4", "批次替代目标 L^{CPI}", "（不裁剪，一路上涨）"),
                              ("green", None, "批次裁剪目标 L^{CLIP}", "（ε = 0.2，涨到 0.030 就平）"),
                              ("ink", None, "真实改进 J(θ) − J(θ_{old})", "（用精确期望计算）")):
    svg.line(LX, y - 4, LX + 26, y - 4, color=col, width=2.4, dash=dash)
    svg.text(LX + 32, y, name, size=12.5, color=col, weight="600")
    svg.text(LX + 32, y + 18, sub, size=12, color="muted")
    y += 48
svg.text(LX, y + 4, "两条样本的噪声让方向偏了：它把", size=12, color="muted")
svg.text(LX, y + 22, "“5”之后的“吧”也往上推。一直优化", size=12, color="muted")
svg.text(LX, y + 40, "L^{CPI}，真实目标先升后降；裁剪让", size=12, color="muted")
svg.text(LX, y + 58, "优化在还在变好的地方停下。", size=12, color="muted")

svg.save("m09-ppo-surrogate.svg")
