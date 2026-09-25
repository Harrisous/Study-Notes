"""Figure 11.9: node-limited routing (schematic) and how many nodes 8 random experts would touch (computed).

Left: 8 nodes x 8 GPUs, 4 routed experts per GPU (256 / 64). A token on node 1 selects 8 experts on nodes
{1, 3, 4, 6}; each remote target node receives ONE copy over InfiniBand (50 GB/s), which is then forwarded
inside the node over NVLink (160 GB/s) (V3 sec. 3.2.2; Insights sec. 4.3).
Right: if 8 of the 256 experts were chosen uniformly at random, the number of distinct nodes they span
(exact inclusion-exclusion, 32 experts per node). P(<= 4 nodes) = 17.48 %, E[nodes] = 5.29.
Writes deepseek/figures/l11-node-limited.svg
"""

from math import comb

from l11_svg import SVG, box

# ---------------------------------------------------------------- exact distribution
TOT = comb(256, 8)


def exactly(j):  # choose j of 8 nodes, all 8 experts inside them, every chosen node used
    return comb(8, j) * sum((-1) ** i * comb(j, i) * comb(32 * (j - i), 8) for i in range(j + 1))


dist = [exactly(j) / TOT for j in range(1, 9)]
assert abs(sum(dist) - 1) < 1e-12
p_le4 = sum(dist[:4])
mean_nodes = sum((j + 1) * p for j, p in enumerate(dist))
assert round(p_le4, 4) == 0.1748 and round(mean_nodes, 2) == 5.29
print(["%.5f%%" % (100 * p) for p in dist], p_le4, mean_nodes)

W, H = 720, 470
svg = SVG(W, H, "节点受限路由：IB 只到 4 个节点，节点内再用 NVLink 转发")

# ---------------------------------------------------------------- left: the cluster
svg.text(16, 22, "Ⓐ 一个 token 的派发路径（每 token 至多 M = 4 个节点）", size=13, weight="600")
NW, NH = 84, 64
pos = {}
for n in range(8):
    r, c = divmod(n, 4)
    x, y = 24 + c * 98, 92 + r * 146
    pos[n] = (x, y)
targets = {0: [(0, 1), (1, 2)], 2: [(0, 0), (1, 3)], 3: [(0, 2), (1, 1)], 5: [(0, 3), (1, 0)]}
for n, (x, y) in pos.items():
    hit = n in targets
    box(svg, x, y, NW, NH, "", color="blue" if hit else "muted", fill_opacity=0.08 if hit else 0.05)
    svg.text(x + NW / 2, y - 6, "节点 %d" % (n + 1), size=12, anchor="middle", color="blue" if hit else "muted",
             weight="600" if hit else None)
    for g in range(8):
        gr, gc = divmod(g, 4)
        gx, gy = x + 8 + gc * 18, y + 12 + gr * 24
        on = hit and (gr, gc) in targets[n]
        svg.rect(gx, gy, 14, 16, fill="blue" if on else "muted", fill_opacity=0.85 if on else 0.18, rx=2)
# source token on node 1, GPU (0,0)
sx, sy = pos[0][0] + 8, pos[0][1] + 12
svg.rect(sx - 2, sy - 2, 18, 20, fill=None, stroke="orange", width=2, rx=3)
svg.text(pos[0][0] - 2, pos[0][1] + NH + 16, "token 在这里", size=11, color="orange")
# IB arrows from node 1 to nodes 3, 4, 6
x1, y1 = pos[0]
for n in (2, 3):
    x2, y2 = pos[n]
    svg.path("M %s %s Q %s %s %s %s" % (x1 + NW / 2, y1 - 22, (x1 + x2 + NW) / 2, y1 - 62, x2 + NW / 2, y2 - 24),
             stroke="ink", width=1.8, arrow=True)
x2, y2 = pos[5]
svg.path("M %s %s L %s %s" % (x1 + NW - 10, y1 + NH, x2 + 12, y2 - 3), stroke="ink", width=1.8, arrow=True)
svg.text(290, 196, "IB：每个远端节点只发一份", size=12, color="ink", anchor="middle", weight="600")
svg.text(290, 212, "（50 GB/s）", size=11, color="muted", anchor="middle")
svg.text(16, 340, "到达后在节点内用 NVLink（160 GB/s）转发到各 GPU；", size=12)
svg.text(16, 358, "蓝色格 = 放着被选中专家的 GPU（每张 GPU 放 4 个专家）。", size=12)
svg.text(16, 386, "IB 流量 ∝ 远端节点数：不受限时最多 7 份，M = 4 时至多 4 份。", size=12, color="blue")
svg.text(16, 404, "NVLink / IB = 160 / 50 = 3.2：节点内转发 3.2 份的时间", size=12)
svg.text(16, 422, "= IB 发 1 份的时间，故 4 × 3.2 = 12.8（论文写“至多 13 个”）。", size=12)

svg.line(404, 36, 404, 440, color="rule", width=1)

# ---------------------------------------------------------------- right: distribution
svg.text(420, 22, "Ⓑ 若 8 个专家随机选，会跨几个节点？", size=13, weight="600")
BX0, BX1, BY0, BY1 = 452, 704, 60, 330
PMAX = 0.45


def by(p):
    return BY1 - p / PMAX * (BY1 - BY0)


for v in (0, 0.1, 0.2, 0.3, 0.4):
    svg.line(BX0, by(v), BX1, by(v), color="grid", width=1)
    svg.text(BX0 - 6, by(v) + 4, "%d%%" % round(100 * v), size=11, anchor="end", color="muted")
bw = (BX1 - BX0) / 8
for j, p in enumerate(dist):
    k = j + 1
    col = "green" if k <= 4 else "red"
    x = BX0 + j * bw + 4
    h = BY1 - by(p)
    svg.rect(x, BY1 - max(h, 0.8), bw - 8, max(h, 0.8), fill=col, fill_opacity=0.55, stroke=col, width=0.8)
    lab = ("%.1f%%" % (100 * p)) if p >= 0.001 else ("%.2f%%" % (100 * p) if p >= 0.0001 else "≈0")
    svg.text(x + (bw - 8) / 2, BY1 - max(h, 0.8) - 5, lab, size=11, anchor="middle", color="ink")
    svg.text(x + (bw - 8) / 2, BY1 + 16, str(k), size=11, anchor="middle", color="muted")
svg.line(BX0, BY1, BX1, BY1, color="ink", width=1.2)
svg.text((BX0 + BX1) / 2, BY1 + 34, "8 个专家分布在几个节点上", size=12, anchor="middle")
svg.text(420, 390, "绿色（≤ 4 个节点，节点受限允许）合计 %.2f%%" % (100 * p_le4), size=12, color="green", weight="600")
svg.text(420, 408, "红色（5–8 个节点，被禁止）合计 %.2f%%" % (100 * (1 - p_le4)), size=12, color="red")
svg.text(420, 426, "随机选时平均跨 %.2f 个节点" % mean_nodes, size=12)
svg.text(420, 452, "这只是组合计数；真实路由远非随机（见正文）。", size=11, color="muted")

svg.save("l11-node-limited.svg")
