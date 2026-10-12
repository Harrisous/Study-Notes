"""Figure 2.4: the gradient of the cross-entropy loss with respect to the logits is p - y.

A toy four-token vocabulary after the context "The unbelievable"; the true next token is "Ġtoken".
Logits z = (2, 1, 0, -1). Everything below is computed here:
  p = softmax(z) = (0.644, 0.237, 0.087, 0.032);  y = one-hot(Ġtoken)
  dL/dz = p - y = (0.644, -0.763, 0.087, 0.032)   (sums to zero)
  one gradient step on the logits with step size 1: z' = z - (p - y), p' = softmax(z')
  loss -log p(Ġtoken): 1.440 -> 0.633
Writes deepseek/figures/l02-logit-grad.svg
"""

import math

import numpy as np

from l02_svg import SVG

words = ["Ġstory", "Ġtoken", "Ġnews", "Ġcat"]
z = np.array([2.0, 1.0, 0.0, -1.0])
p = np.exp(z) / np.exp(z).sum()
y = np.array([0, 1, 0, 0.0])
g = p - y
z2 = z - g
p2 = np.exp(z2) / np.exp(z2).sum()
assert abs(g.sum()) < 1e-12
L1, L2 = -math.log(p[1]), -math.log(p2[1])

W, H = 720, 366
svg = SVG(W, H, "交叉熵对 logits 的梯度 = p − y，以及沿负梯度走一步的效果")

ROW0, RH = 92, 50
cols = {"word": 20, "z": 116, "p": 170, "g": 370, "p2": 572}
BARW = 100           # width for probability 1.0
GW = 70              # half-width for |gradient| = 1

svg.text(20, 22, "语境 “The unbelievable” 之后，真实的下一个 token 是 Ġtoken（y 在这一格为 1，其余为 0）", size=12,
         color="ink")
heads = [("word", "候选 token", "start"), ("z", "logit z", "middle"), ("p", "p = softmax(z)", "start"),
         ("g", "梯度 ∂ℓ/∂z = p − y", "middle"), ("p2", "走一步后 p′", "start")]
for key, lab, anc in heads:
    x = cols[key] + (0 if key != "g" else GW + 10) + (0 if key != "z" else 14)
    svg.text(x, 64, lab, size=12, anchor=anc, color="muted", weight="600")

gx0 = cols["g"] + GW + 10        # zero line of the diverging bars
svg.line(gx0, 74, gx0, ROW0 + 4 * RH - 6, color="muted", width=1.2)
for i, w in enumerate(words):
    yy = ROW0 + i * RH
    tgt = i == 1
    if tgt:
        svg.rect(12, yy - 20, W - 24, RH - 6, fill="green", fill_opacity=0.07, rx=6)
    svg.text(cols["word"], yy + 4, w, size=13, color="green" if tgt else "ink", weight="600" if tgt else None)
    svg.text(cols["z"] + 14, yy + 4, ("%+.0f" % z[i]).replace("-", "−") if z[i] else "0", size=13, anchor="middle", color="ink")
    # p bar
    svg.rect(cols["p"], yy - 8, BARW, 14, fill="grid", fill_opacity=1)
    svg.rect(cols["p"], yy - 8, BARW * p[i], 14, fill="blue", fill_opacity=0.85)
    svg.text(cols["p"] + BARW + 6, yy + 4, "%.3f" % p[i], size=12, color="blue")
    # gradient bar (diverging)
    gw = GW * abs(g[i])
    if g[i] >= 0:
        svg.rect(gx0, yy - 8, gw, 14, fill="ink", fill_opacity=0.55)
        svg.text(gx0 + gw + 6, yy + 4, "+%.3f" % g[i], size=12, color="ink")
    else:
        svg.rect(gx0 - gw, yy - 8, gw, 14, fill="red", fill_opacity=0.85)
        svg.text(gx0 - gw - 6, yy + 4, ("%.3f" % g[i]).replace("-", "−"), size=12, color="red", anchor="end")
    # after one step
    svg.rect(cols["p2"], yy - 8, BARW, 14, fill="grid", fill_opacity=1)
    svg.rect(cols["p2"], yy - 8, BARW * p2[i], 14, fill="green" if tgt else "blue", fill_opacity=0.85)
    svg.text(cols["p2"] + BARW + 6, yy + 4, "%.3f" % p2[i], size=12, color="green" if tgt else "blue")

yb = ROW0 + 4 * RH - 4
svg.text(gx0 - 60, yb + 12, "← 把 z 往上推", size=12, color="red", anchor="middle")
svg.text(gx0 + 60, yb + 12, "把 z 往下压 →", size=12, color="ink", anchor="middle")
svg.text(20, yb + 44, ("四个梯度之和 = %.3f + (%.3f) + %.3f + %.3f = 0：概率总量守恒，一格涨、其余按 p 的比例跌。"
          % (g[0], g[1], g[2], g[3])).replace("-", "−"), size=12, color="ink")
svg.text(20, yb + 64, "沿负梯度改 logits 一步（z′ = z − 1·(p − y)）：p(Ġtoken) 从 %.3f 升到 %.3f，损失从 %.3f 降到 %.3f nats。"
         % (p[1], p2[1], L1, L2), size=12, color="ink")

svg.save("l02-logit-grad.svg")
