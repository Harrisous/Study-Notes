"""Figure 14.11: what the indexer's KL loss compares (hand-sized example).

Two main-attention heads over four candidate tokens: a1 = (.10,.70,.15,.05), a2 = (.20,.40,.35,.05).
Target p = head average = (.15,.55,.25,.05). Indexer scores I = (1, 3, 1.5, 0) (Fig. 14.8),
q = softmax(I). Dense stage: KL(p || q) = 0.0576, gradient dL/dI = q - p.
Sparse stage on S = {2, 3}: the main attention now runs only on S, so each head's weights become
a_i / m_i (m_1 = 0.85, m_2 = 0.75 are the dense weights each head had on S); their average is
p_S = (0.6784, 0.3216). This is NOT the dense average p cut to S and renormalised, (0.6875, 0.3125),
which would weight the heads by m_i. q_S = softmax(I_S); KL(p_S || q_S) = 0.0557. All computed here.
Writes deepseek/figures/l14-kl-align.svg
"""

import math

from l14_svg import SVG

a1 = [0.10, 0.70, 0.15, 0.05]
a2 = [0.20, 0.40, 0.35, 0.05]
p = [(x + y) / 2 for x, y in zip(a1, a2)]
I = [1.0, 3.0, 1.5, 0.0]
Z = sum(math.exp(v) for v in I)
q = [math.exp(v) / Z for v in I]
kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
grad = [qi - pi for pi, qi in zip(p, q)]
assert abs(kl - 0.0576) < 5e-4
S = [1, 2]
m = [sum(a[s] for s in S) for a in (a1, a2)]              # dense weight each head had on S
assert [round(v, 2) for v in m] == [0.85, 0.75]
pS = [(a1[s] / m[0] + a2[s] / m[1]) / 2 for s in S]         # each head renormalised on S, then averaged
assert [round(v, 4) for v in pS] == [0.6784, 0.3216]
p_cut = [p[s] / sum(p[t] for t in S) for s in S]            # the other reading: average first, then cut
assert [round(v, 4) for v in p_cut] == [0.6875, 0.3125]
ZS = sum(math.exp(I[s]) for s in S)
qS = [math.exp(I[s]) / ZS for s in S]
klS = sum(a * math.log(a / b) for a, b in zip(pS, qS))
assert abs(klS - 0.0557) < 5e-4

W, H = 760, 470
svg = SVG(W, H, "indexer 的 KL 对齐：目标分布是主注意力各头的平均（手算例）")

# ---------------------------------------------------------------- panel (a)
X0, X1, Y0, Y1 = 60, 440, 64, 290


def py(v):
    return Y1 - v / 0.8 * (Y1 - Y0)


svg.text(20, 24, "(a) 稠密预热：在全部 4 个候选 token 上对齐", size=13, weight="700")
for v in (0, 0.2, 0.4, 0.6, 0.8):
    y = py(v)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "%.1f" % v, size=11, anchor="end", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
gw = (X1 - X0) / 4
series = [(a1, "muted", 0.25, "头 1 注意力"), (a2, "muted", 0.45, "头 2 注意力"), (p, "green", 0.7, "目标 p（两头平均）"),
          (q, "violet", 0.7, "softmax(I)（indexer）")]
bw = (gw - 20) / 4
for s in range(4):
    gx = X0 + s * gw + 10
    for k, (vals, col, op, _) in enumerate(series):
        v = vals[s]
        svg.rect(gx + k * bw, py(v), bw - 2, Y1 - py(v), fill=col, fill_opacity=op, stroke=col, width=0.8)
    svg.text(gx + 2 * bw, Y1 + 18, "s = %d" % (s + 1), size=12, anchor="middle")
    g = grad[s]
    svg.text(gx + 2 * bw, Y1 + 38, ("+%.3f" % g) if g > 0 else ("−%.3f" % -g), size=12, anchor="middle",
             color="red" if g > 0 else "green")
svg.text(X0 - 50, Y1 + 38, "∂L/∂I", size=12, color="muted")
for k, (_, col, op, lab) in enumerate(series):
    svg.rect(X0 + 8 + (k % 2) * 190, Y0 - 34 + (k // 2) * 18, 12, 12, fill=col, fill_opacity=op)
    svg.text(X0 + 26 + (k % 2) * 190, Y0 - 24 + (k // 2) * 18, lab, size=12)
svg.text(X0, Y1 + 64, "KL(p ‖ softmax(I)) = %.4f" % kl, size=13, weight="700", color="violet")
svg.text(X0, Y1 + 84, "梯度 = softmax(I) − p：s = 2 被 indexer 高估（+0.160），", size=12, color="muted")
svg.text(X0, Y1 + 102, "梯度下降会压低 s = 2 的分数、抬高其余三个。", size=12, color="muted")

# ---------------------------------------------------------------- panel (b)
BX0, BX1 = 520, 740
svg.text(470, 24, "(b) 稀疏训练：只在 S_{t} = {2, 3} 上", size=13, weight="700")
for v in (0, 0.2, 0.4, 0.6, 0.8):
    y = py(v)
    svg.line(BX0, y, BX1, y, color="grid", width=1)
    svg.text(BX0 - 8, y + 4, "%.1f" % v, size=11, anchor="end", color="muted")
svg.line(BX0, Y1, BX1, Y1, color="muted", width=1.2)
gw2 = (BX1 - BX0) / 2
for i, s in enumerate(S):
    gx = BX0 + i * gw2 + 18
    for k, (vals, col) in enumerate(((pS, "green"), (qS, "violet"))):
        v = vals[i]
        svg.rect(gx + k * 34, py(v), 30, Y1 - py(v), fill=col, fill_opacity=0.7, stroke=col, width=0.8)
        svg.text(gx + k * 34 + 15, py(v) - 5, "%.3f" % v, size=11, anchor="middle", color=col)
    svg.text(gx + 32, Y1 + 18, "s = %d" % (s + 1), size=12, anchor="middle")
svg.text(470, Y1 + 64, "KL(p_{S} ‖ softmax(I_{S})) = %.4f" % klS, size=13, weight="700", color="violet")
svg.text(470, Y1 + 84, "主注意力每个头只在 S_{t} 内做 softmax，", size=12, color="muted")
svg.text(470, Y1 + 102, "再取两头平均（笔者的理解，论文未写明）；", size=12, color="muted")
svg.text(470, Y1 + 120, "S_{t} 外的 token 没有梯度。", size=12, color="muted")

# stop-gradient note
svg.rect(20, 446 - 18, 720, 30, fill="red", fill_opacity=0.05, stroke="red", width=1, rx=6, dash="5 3")
svg.text(32, 448, "⊘ 梯度分离：indexer 的输入从计算图上 detach。indexer 只从这个 KL 损失学习，主模型只从语言模型损失学习。",
         size=12, color="red")

svg.save("l14-kl-align.svg")
