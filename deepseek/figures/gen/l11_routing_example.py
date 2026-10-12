"""Figure 11.10: one complete routing decision, computed the way the released V3 gate code does it.

Toy size (research notes sec. 5.2 fig. 11.3 / exercise 4; constructed numbers): 16 routed experts on 4 nodes
(4 per node, 0-based ids as in the code), K_r = 4, M = 2 nodes, so each node is scored by the sum of its
top K_r / M = 2 values of s + b.  Steps (HF MoEGate / GitHub Gate):
  s = sigmoid(logits); s~ = s + b; node score = sum of the 2 largest s~ in the node; keep the top-2 nodes;
  mask the other experts with -inf; Top-4 of s~; gate = s / sum(selected s) (raw s, no bias); x 2.5.
Result: nodes {0, 1}, experts {0, 2, 4, 5}, g = [0.2776, 0.2163, 0.2568, 0.2493], x2.5 -> [0.6941, ...].
Expert 10 has the highest affinity (0.8176) but is excluded by the node limit alone.
Writes deepseek/figures/l11-routing-example.svg
"""

import numpy as np

from l11_svg import SVG, box

logits = np.array([1.2, -0.3, 0.4, -1.0, 0.9, 0.8, -0.5, -0.2, -0.7, 0.1, 1.5, -1.2, 0.0, -0.4, 0.3, 0.6])
bias = np.array([-0.05, 0, 0.02, 0, 0.03, 0, 0, 0, 0, 0, -0.10, 0, 0.04, 0, 0, 0.06])
s = 1 / (1 + np.exp(-logits))
st = s + bias
node_score = np.array([np.sort(st[4 * n:4 * n + 4])[-2:].sum() for n in range(4)])
nodes = sorted(np.argsort(-node_score)[:2].tolist())
masked = np.full(16, -np.inf)
for n in nodes:
    masked[4 * n:4 * n + 4] = st[4 * n:4 * n + 4]
sel = sorted(np.argsort(-masked)[:4].tolist())
g = s[sel] / s[sel].sum()
assert nodes == [0, 1] and sel == [0, 2, 4, 5]
assert np.allclose(np.round(node_score, 4), [1.3372, 1.4309, 1.2426, 1.2801])
assert np.allclose(np.round(g, 4), [0.2776, 0.2163, 0.2568, 0.2493])
# the two counterfactuals quoted in the text
assert sorted(np.argsort(-st)[:4].tolist()) == [0, 4, 10, 15]      # bias, no node limit
ns0 = np.array([np.sort(s[4 * n:4 * n + 4])[-2:].sum() for n in range(4)])
assert sorted(np.argsort(-ns0)[:2].tolist()) == [0, 1]              # node limit, no bias

W, H = 720, 530
svg = SVG(W, H, "一次完整的路由：16 个专家、4 个节点、选 2 个节点内的 4 个专家")

PX0, PX1, PY0, PY1 = 60, 704, 96, 270
YMAX = 1.0


def py(v):
    return PY1 - v / YMAX * (PY1 - PY0)


svg.text(16, 22, "Ⓐ 亲和度 s（柱）与加偏置后的 s + b（点）；编号从 0 开始，与代码一致", size=13, weight="600")
for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
    svg.line(PX0, py(v), PX1, py(v), color="grid", width=1)
    svg.text(PX0 - 6, py(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
gw = (PX1 - PX0) / 4
for n in range(4):
    gx = PX0 + n * gw
    chosen = n in nodes
    svg.rect(gx + 4, PY0 - 44, gw - 8, PY1 - PY0 + 68, fill="blue" if chosen else "muted",
             fill_opacity=0.05 if chosen else 0.03, stroke="blue" if chosen else "rule", width=1.4 if chosen else 1,
             rx=6, dash=None if chosen else "4 3")
    svg.text(gx + gw / 2, PY0 - 26, "节点 %d%s" % (n, "（入选）" if chosen else "（落选）"),
             size=12, anchor="middle", color="blue" if chosen else "muted", weight="600" if chosen else None)
    svg.text(gx + gw / 2, PY0 - 10, "节点分 %.4f" % node_score[n], size=12, anchor="middle",
             color="blue" if chosen else "muted")
    for j in range(4):
        i = 4 * n + j
        x = gx + 16 + j * (gw - 24) / 4
        bw = (gw - 24) / 4 - 10
        on = i in sel
        col = "blue" if on else "muted"
        svg.rect(x, py(s[i]), bw, PY1 - py(s[i]), fill=col, fill_opacity=0.7 if on else 0.3, rx=2)
        svg.text(x + bw / 2, PY1 + 15, str(i), size=12, anchor="middle", color="blue" if on else "ink",
                 weight="600" if on else None)
        cx = x + bw / 2
        if bias[i] != 0:
            svg.line(cx, py(s[i]), cx, py(st[i]), color="red" if bias[i] < 0 else "green", width=2)
        svg.circle(cx, py(st[i]), 3.6, fill="violet", stroke="violet", width=1)
        svg.text(cx, py(max(s[i], st[i])) - 7, "%.2f" % s[i], size=11, anchor="middle", color="ink")
svg.line(PX0, PY1, PX1, PY1, color="ink", width=1.2)
svg.text(PX0, PY1 + 44, "横轴：专家编号（蓝 = 选中）", size=11, color="muted")
# flag expert 10
x10 = PX0 + 2 * gw + 16 + 2 * (gw - 24) / 4 + ((gw - 24) / 4 - 10) / 2
svg.text(x10 - 4, PY1 + 44, "↑ 专家 10：亲和度最高（0.82），", size=11, color="red")
svg.text(x10 + 8, PY1 + 58, "但节点 2 落选，它也就出局", size=11, color="red")

# ---------------------------------------------------------------- steps and gates
y = 350
svg.text(16, y, "Ⓑ 按代码的顺序", size=13, weight="600")
steps = [
    "① s = σ(logit)；s̃ = s + b（紫点；红/绿竖线是偏置）",
    "② 节点分 = 节点内最大的 K_{r}/M = 2 个 s̃ 之和 → 选分最高的 M = 2 个节点：0、1",
    "③ 其余节点的专家分数置为 −∞，在剩下 8 个里按 s̃ 取 Top-4：专家 0、2、4、5",
    "④ 门控值只用原始 s：g = s / Σs，再乘 routed_scaling_factor = 2.5",
]
for k, t in enumerate(steps):
    svg.text(24, y + 22 + k * 20, t, size=12)
ty = y + 112
heads = ["专家", "0", "2", "4", "5", "合计"]
rows = [
    ("s", ["%.4f" % v for v in s[sel]] + ["%.4f" % s[sel].sum()]),
    ("g = s / Σs", ["%.4f" % v for v in g] + ["1"]),
    ("× 2.5", ["%.4f" % (2.5 * v) for v in g] + ["2.5"]),
]
cxs = [24, 150, 230, 310, 390, 470]
for x, h in zip(cxs, heads):
    svg.text(x, ty, h, size=12, weight="600", color="muted")
for r, (lab, vals) in enumerate(rows):
    yy = ty + 18 + r * 18
    svg.text(cxs[0], yy, lab, size=12, color="green" if r else "ink")
    for x, v in zip(cxs[1:], vals):
        svg.text(x, yy, v, size=12, color="green" if r else "ink")

svg.save("l11-routing-example.svg")
