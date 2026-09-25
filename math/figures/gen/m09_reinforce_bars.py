"""Figure M9.4: single-sample REINFORCE estimates of dJ/dz_{q,5} in the toy model, with three baselines.

For each of the four possible answers the estimate is (R - b) * (1[a0=5] - 0.6). Baselines:
b = 0, b = V(q) = 0.55, and the variance-optimal constant b* = E[R u^2] / E[u^2] = 0.40 (M9.13; u = score of the logit, the local variable s below).
Every bar, mean and variance is computed here. Writes math/figures/m09-reinforce-bars.svg
"""

from m09_svg import SVG

answers = [("5！", 0.42, 1.0, 0.4), ("5吧", 0.18, 0.5, 0.4), ("6！", 0.20, 0.0, -0.6), ("6吧", 0.20, 0.2, -0.6)]
# (label, probability, return R, score 1[a0=5]-pi(5|q))
num = sum(p * s * s * R for _, p, R, s in answers)
den = sum(p * s * s for _, p, R, s in answers)
bstar = num / den
V = 0.55
bases = [("b = 0（不减基线）", 0.0, "muted"), ("b = V(q) = 0.55", V, "blue"), ("b* = %.2f（方差最优）" % bstar, bstar, "green")]
stats = []
for name, b, col in bases:
    g = [(R - b) * s for _, p, R, s in answers]
    m = sum(p * x for (_, p, _, _), x in zip(answers, g))
    v = sum(p * x * x for (_, p, _, _), x in zip(answers, g)) - m * m
    stats.append((g, m, v))
    assert abs(m - 0.18) < 1e-12
assert abs(bstar - 0.40) < 1e-12

W, H = 720, 360
svg = SVG(W, H, "玩具模型里对第一个 logit 的单样本梯度估计：三种基线")
X0, X1, Y0, Y1 = 64, 520, 30, 290
YMIN, YMAX = -0.2, 0.5


def py(v):
    return Y1 - (v - YMIN) / (YMAX - YMIN) * (Y1 - Y0)


for t in (-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5):
    y = py(t)
    svg.line(X0, y, X1, y, color="grid" if t != 0 else "muted", width=1 if t != 0 else 1.2)
    svg.text(X0 - 8, y + 4, ("%.1f" % t).replace("-", "−"), size=11, anchor="end", color="muted")
svg.text(18, (Y0 + Y1) / 2, "单样本估计值 g", size=12, anchor="middle", color="muted", rotate=-90)

gw = (X1 - X0) / 4
bw = 30
for i, (lab, p, R, s) in enumerate(answers):
    cx = X0 + gw * (i + 0.5)
    for j, (name, b, col) in enumerate(bases):
        g = stats[j][0][i]
        x = cx + (j - 1) * (bw + 6) - bw / 2
        ytop, ybot = (py(g), py(0)) if g >= 0 else (py(0), py(g))
        svg.rect(x, ytop, bw, max(ybot - ytop, 0.8), fill=col, fill_opacity=0.85 if col != "muted" else 0.55, rx=2)
        ty = ytop - 5 if g >= 0 else ybot + 14
        lab_g = "0" if abs(g) < 1e-9 else ("%.2f" % g).replace("-", "−")
        svg.text(x + bw / 2, ty, lab_g, size=11, anchor="middle", color=col if col != "muted" else "ink")
    svg.text(cx, Y1 + 20, "“%s”" % lab, size=13, anchor="middle", weight="600")
    svg.text(cx, Y1 + 38, "概率 %.2f，R = %.1f" % (p, R), size=12, anchor="middle", color="muted")

# true gradient
yt = py(0.18)
svg.line(X0, yt, X1, yt, color="ink", width=1.4, dash="6 4")

# legend with variances
LX = 540
svg.text(LX, Y0 + 6, "三种基线（期望都是 0.18）", size=12, weight="600")
for j, (name, b, col) in enumerate(bases):
    y = Y0 + 34 + j * 58
    svg.rect(LX, y - 11, 14, 14, fill=col, fill_opacity=0.85 if col != "muted" else 0.55, rx=2)
    svg.text(LX + 20, y + 1, name, size=12)
    svg.text(LX + 20, y + 20, "方差 %.4f" % stats[j][2], size=12, color="muted")
    svg.text(LX + 20, y + 37, "标准差 %.3f" % (stats[j][2] ** 0.5), size=12, color="muted")
svg.line(LX, Y0 + 205, LX + 26, Y0 + 205, color="ink", width=1.4, dash="6 4")
svg.text(LX + 32, Y0 + 209, "真值 ∂J/∂z = 0.18", size=12, weight="600")
svg.text(LX, Y0 + 234, "b = 0 时有一条回答给出", size=12, color="muted")
svg.text(LX, Y0 + 251, "负的估计（方向错）；", size=12, color="muted")
svg.text(LX, Y0 + 268, "用 b* 时四条全为正。", size=12, color="muted")

svg.save("m09-reinforce-bars.svg")
