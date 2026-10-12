"""Figure 4.14: the geometry of the speculative-sampling acceptance rule on a three-token toy example.

Target p = [0.5, 0.3, 0.2], draft q = [0.2, 0.5, 0.3] (lesson-04-doc §4.8).
  accept a draft x with probability min(1, p(x)/q(x))       -> [1, 0.6, 0.667]
  mass accepted: min(p, q) = [0.2, 0.3, 0.2], alpha = 0.7 = 1 - TV(p, q)
  rejected draft mass: (q - p)+ = [0, 0.2, 0.1]; residual (p - q)+ = [0.3, 0, 0] -> normalised [1, 0, 0]
  output law: min(p, q) + (1 - alpha) * residual = p exactly. A Monte Carlo run (numpy, seed 0,
  200,000 draws) is printed on the figure as a check.
Writes deepseek/figures/l04-spec-accept.svg
"""

import numpy as np

from l04_svg import SVG

WORDS = ["山", "海", "河"]
p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])
acc = np.minimum(1, p / q)
mn = np.minimum(p, q)
alpha = mn.sum()
tv = 0.5 * np.abs(p - q).sum()
rej = np.maximum(0, q - p)
res = np.maximum(0, p - q)
res_n = res / res.sum()
out = mn + (1 - alpha) * res_n
assert np.allclose(acc, [1, 0.6, 2 / 3]) and abs(alpha - 0.7) < 1e-12 and abs(tv - 0.3) < 1e-12
assert np.allclose(out, p)

rng = np.random.default_rng(0)
N = 200_000
x = rng.choice(3, size=N, p=q)
u = rng.random(N)
ok = u < acc[x]
y = np.where(ok, x, rng.choice(3, size=N, p=res_n))
freq = np.bincount(y, minlength=3) / N
rate = ok.mean()
assert np.allclose(freq, p, atol=0.005) and abs(rate - 0.7) < 0.005

W, H = 720, 412
svg = SVG(W, H, "投机采样的接受规则：接受 min(p, q)，拒绝的质量由残差 (p − q)+ 补上")
X0, BW, GAPX = 60, 64, 46
Y1, SC = 300, 420     # baseline y, px per unit probability


def ypx(v):
    return Y1 - v * SC


svg.text(16, 24, "Ⓐ 每个候选词：绿色 = 接受的质量 min(p, q)，红色 = 被拒的草稿质量，虚线 = 补采的质量", size=14, weight="600")
for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
    y = ypx(v)
    svg.line(X0 - 6, y, X0 + 3 * (BW + GAPX) - GAPX + 6, y, color="grid", width=1)
    svg.text(X0 - 10, y + 4, "%.1f" % v, size=11, anchor="end", color="muted")
for i in range(3):
    x = X0 + i * (BW + GAPX)
    # accepted mass
    svg.rect(x, ypx(mn[i]), BW, mn[i] * SC, fill="green", fill_opacity=0.55)
    if rej[i] > 0:
        svg.rect(x, ypx(q[i]), BW, rej[i] * SC, fill="red", fill_opacity=0.35)
        svg.text(x + BW / 2, ypx(q[i]) + rej[i] * SC / 2 + 4, "−%.1f" % rej[i], size=12, anchor="middle", color="red",
                 weight="600")
    if res[i] > 0:
        svg.rect(x, ypx(p[i]), BW, res[i] * SC, fill="green", fill_opacity=0.08, stroke="green", width=1.5, dash="5 3")
        svg.text(x + BW / 2, ypx(p[i]) + res[i] * SC / 2 + 4, "+%.1f" % res[i], size=12, anchor="middle", color="green",
                 weight="600")
    # outlines of p (target) and q (draft)
    svg.line(x - 4, ypx(p[i]), x + BW + 4, ypx(p[i]), color="green", width=2.4)
    svg.line(x - 4, ypx(q[i]), x + BW + 4, ypx(q[i]), color="blue", width=2.4, dash="6 3")
    svg.text(x + BW / 2, ypx(mn[i]) + mn[i] * SC / 2 + 4, "%.1f" % mn[i], size=12, anchor="middle", weight="600")
    svg.text(x + BW / 2, Y1 + 20, WORDS[i], size=13, anchor="middle", weight="600")
    svg.text(x + BW / 2, Y1 + 40, "p = %.1f，q = %.1f" % (p[i], q[i]), size=11, anchor="middle", color="muted")
    svg.text(x + BW / 2, Y1 + 58, "接受概率 %s" % ("1" if acc[i] == 1 else "%.3f" % acc[i]), size=12, anchor="middle",
             color="green")
svg.line(X0 - 6, Y1, X0 + 3 * (BW + GAPX) - GAPX + 6, Y1, color="muted", width=1.2)
lx, ly = X0, Y1 + 90
svg.line(lx, ly - 4, lx + 26, ly - 4, color="green", width=2.4)
svg.text(lx + 32, ly, "目标分布 p（大模型）", size=12)
svg.line(lx + 180, ly - 4, lx + 206, ly - 4, color="blue", width=2.4, dash="6 3")
svg.text(lx + 212, ly, "草稿分布 q（小模型）", size=12)

# right column: the bookkeeping
tx = 400
svg.text(tx, 78, "Ⓑ 记账", size=14, weight="600")
lines = [
    ("接受率 β = Σ min(p, q) = 0.2 + 0.3 + 0.2 = %.1f" % alpha, "ink"),
    ("全变差 TV(p, q) = ½ Σ |p − q| = %.1f = 1 − β" % tv, "ink"),
    ("被拒的草稿质量 Σ(q − p)_{+} = 0.2 + 0.1 = %.1f" % rej.sum(), "red"),
    ("残差 (p − q)_{+} = [0.3, 0, 0] → 归一化 [1, 0, 0]", "green"),
    ("输出 = min(p, q) + (1 − β) · 残差", "ink"),
    ("     = [0.2 + 0.3·1, 0.3 + 0, 0.2 + 0] = [0.5, 0.3, 0.2] = p", "ink"),
]
for k, (s, col) in enumerate(lines):
    svg.text(tx, 108 + k * 24, s, size=12, color=col, weight="600" if k in (0, 5) else None)
svg.text(tx, 108 + 6 * 24 + 16, "模拟 %s 次（numpy，种子 0）：" % format(N, ","), size=12, color="muted")
svg.text(tx, 108 + 7 * 24 + 16, "输出频率 [%.3f, %.3f, %.3f]，接受率 %.3f" % (freq[0], freq[1], freq[2], rate), size=12,
         color="muted")
svg.text(tx, 108 + 9 * 24, "被拒的红色质量（0.3）恰好由绿色虚线", size=12)
svg.text(tx, 108 + 10 * 24, "（0.3）补回：两块面积相等，因为", size=12)
svg.text(tx, 108 + 11 * 24, "p 与 q 的总和都是 1。", size=12)
svg.save("l04-spec-accept.svg")
print("MC freq", freq, "rate", rate)
