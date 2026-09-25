"""Figure 7.4: three gating conventions applied to the same 8 router logits, k = 2.

logits z = (0.2, 2, 0, -0.5, 1, 0.5, -0.3, -1) (the research notes' R13 example, permuted so
that experts 2 and 5 win, matching figure 7.3).
  (1) softmax over all 8, keep top-2, no renormalisation   (Switch, DeepSeekMoE, V2)
  (2) keep top-2 logits, softmax over them                  (Shazeer 2017, Mixtral)
  (3) softmax over all 8, keep top-2, renormalise           (GShard)
Every bar height is computed here.
Writes deepseek/figures/l07-gating-orders.svg
"""

import math

from l07_svg import SVG

z = [0.2, 2.0, 0.0, -0.5, 1.0, 0.5, -0.3, -1.0]
n = len(z)
Z = sum(math.exp(v) for v in z)
p = [math.exp(v) / Z for v in z]
top = sorted(range(n), key=lambda i: -z[i])[:2]
g1 = [p[i] if i in top else 0.0 for i in range(n)]
Zt = sum(math.exp(z[i]) for i in top)
g2 = [math.exp(z[i]) / Zt if i in top else 0.0 for i in range(n)]
st = sum(p[i] for i in top)
g3 = [p[i] / st if i in top else 0.0 for i in range(n)]
assert all(abs(a - b) < 1e-12 for a, b in zip(g2, g3))

W, H = 740, 452
svg = SVG(W, H, "三种门控顺序作用在同一组路由 logits 上（8 个专家，k = 2）")

# ---- left: logits
LX0, LBW, LG = 30, 14, 6
lz0 = 150            # y of z = 0
lscale = 30          # px per unit logit
svg.text(20, 26, "路由 logits z_{i} = *u*_{t}^{⊤}*e*_{i}", size=13, weight="600")
svg.line(LX0 - 6, lz0, LX0 + n * (LBW + LG), lz0, color="muted", width=1)
for i, v in enumerate(z):
    x = LX0 + i * (LBW + LG)
    y = lz0 - v * lscale if v > 0 else lz0
    h = abs(v) * lscale
    chosen = i in top
    svg.rect(x, y, LBW, max(h, 1), fill="violet", fill_opacity=0.8 if chosen else 0.3)
    svg.text(x + LBW / 2, 224, str(i + 1), size=11, anchor="middle", color="ink" if chosen else "muted")
    lab = ("%g" % v)
    ty = (y - 5) if v >= 0 else (lz0 + h + 13)
    svg.text(x + LBW / 2, ty, lab, size=11, anchor="middle", color="violet" if chosen else "muted")
svg.text(20, 246, "专家编号", size=11, color="muted")
svg.text(20, 272, "最大的两个：专家 2（z = 2）", size=12, color="ink")
svg.text(20, 290, "和专家 5（z = 1）", size=12, color="ink")

# ---- right: three pipelines
RX = 238
rows = [
    ("① softmax（全部 8 个）→ 取 top-2，不再归一化", "Switch、DeepSeekMoE、DeepSeek-V2", g1, True),
    ("② 先取 top-2 → 只对这 2 个做 softmax", "Shazeer 2017、Mixtral", g2, False),
    ("③ softmax（全部 8 个）→ 取 top-2 → 重新归一化", "GShard", g3, False),
]
BW, BG = 16, 5
bh = 56              # px for gate value 0.8
for r, (title, who, g, ghost) in enumerate(rows):
    y0 = 34 + r * 132
    svg.text(RX, y0, title, size=13, weight="600")
    svg.text(RX, y0 + 18, who, size=12, color="muted")
    base = y0 + 100
    bx0 = RX
    svg.line(bx0 - 4, base, bx0 + n * (BW + BG), base, color="muted", width=1)
    for i in range(n):
        x = bx0 + i * (BW + BG)
        if ghost and i not in top:
            h = bh * p[i] / 0.8
            svg.rect(x, base - h, BW, h, fill="muted", fill_opacity=0.12, stroke="muted", width=0.8, dash="2 2")
        if g[i] > 0:
            h = bh * g[i] / 0.8
            svg.rect(x, base - h, BW, h, fill="blue", fill_opacity=0.85)
            svg.text(x + BW / 2, base - h - 5, "%.2f" % g[i], size=11,
                     anchor="middle", color="blue")
    tx = bx0 + n * (BW + BG) + 20
    gs = [g[i] for i in top]
    svg.text(tx, base - 44, "g_{2,t} = %.4f，g_{5,t} = %.4f" % (gs[0], gs[1]), size=12, color="blue")
    svg.text(tx, base - 24, "门控值之和 = %.4f" % sum(gs), size=12, color="ink")
    if ghost:
        svg.text(tx, base - 4, "虚线框：被丢掉的 6 个 softmax 概率（和 %.4f）" % (1 - sum(gs)), size=11, color="muted")
    elif r == 1:
        svg.text(tx, base - 4, "e^{2}/(e^{2}+e^{1}) = 1/(1+e^{-1})", size=12, color="muted")
    else:
        svg.text(tx, base - 4, "%.4f/%.4f = %.4f：与②完全相同" % (p[top[0]], st, gs[0]), size=11, color="muted")

svg.line(16, 416, W - 16, 416, color="rule", width=1)
svg.text(20, 438, "k = 1 时：①的门值是 %.4f，随路由器参数连续变化，路由器有梯度；②的门值恒为 1，路由器拿不到梯度。" % p[top[0]],
         size=12, color="ink")

svg.save("l07-gating-orders.svg")
print("p", [round(v, 4) for v in p], "g1", [round(v, 4) for v in g1 if v], "g2", [round(v, 4) for v in g2 if v])
