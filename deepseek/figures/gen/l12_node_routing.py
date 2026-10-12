"""Figure 12.15: node-limited dispatch across H800 nodes (V3 report 3.2.2, Insights 4.3).

Training layout: EP64 over 8 nodes x 8 GPUs, 4 routed experts per GPU (256 / 64). A token that
lives on GPU 3 of node A has picked 8 experts on at most M = 4 nodes. It is sent once per target
node over InfiniBand (50 GB/s per the V3 report) to the GPU with the same in-node index (3), then
forwarded over NVLink (160 GB/s) to the GPUs that hold its experts. The expert placement in the
picture is an illustrative example (2 + 2 + 1 + 3 = 8 experts).
Writes deepseek/figures/l12-node-routing.svg
"""

from l12_svg import SVG

assert 256 // 64 == 4 and abs(160 / 50 - 3.2) < 1e-12 and abs(4 * 3.2 - 12.8) < 1e-12

W, H = 760, 456
svg = SVG(W, H, "节点受限路由：每个 token 至多发往 4 个节点，先走 IB 到同编号 GPU，再走 NVLink 扇出")
G = 24          # GPU box size
GAP = 4


def node(x, y, name, targets=(), src=None, side=False):
    w = 8 * (G + GAP) + 16
    svg.rect(x, y, w, 58, fill="muted", fill_opacity=0.05, stroke="muted", width=1.2, rx=8)
    if side:
        svg.text(x - 8, y + 26, name, size=13, weight="700", anchor="end")
    else:
        svg.text(x + 8, y - 6, name, size=12, weight="600")
    svg.rect(x + 8, y + 42, w - 16, 10, fill="muted", fill_opacity=0.22, rx=3)
    svg.text(x + w - 12, y + 51, "NVSwitch", size=11, anchor="end", color="muted")
    pos = {}
    for g in range(8):
        gx = x + 8 + g * (G + GAP)
        col, op = ("blue", 0.35) if g in targets else (("orange", 0.35) if g == src else ("muted", 0.08))
        svg.rect(gx, y + 8, G, G, fill=col, fill_opacity=op, stroke=col if col != "muted" else "rule", width=1)
        svg.text(gx + G / 2, y + 25, str(g), size=11, anchor="middle")
        pos[g] = (gx + G / 2, y + 8, y + 8 + G)
    return pos, w


# source node
A, wA = node(30, 170, "节点 A（token 在 GPU 3）", src=3)
targets = {"B": (1, 6), "C": (3, 5), "D": (0,), "E": (2, 4, 7)}
assert sum(len(v) for v in targets.values()) == 8
ys = [40, 136, 232, 328]
TX = 470
for (nm, tg), y in zip(targets.items(), ys):
    nm = "节点 " + nm
    pos, w = node(TX, y, nm, targets=tg, side=True)
    svg.text(TX + w + 6, y + 26, "%d 个" % len(tg), size=12, color="blue")
    # IB: A.GPU3 -> nm.GPU3. The curve leaves A to the right, stays left of the node column
    # until it reaches the gap just above its own target node, then drops onto GPU 3 from above,
    # so no curve crosses another node's box.
    sx, sy0, sy1 = A[3]
    tx, ty0, ty1 = pos[3]
    gap_y = y - 16
    svg.path("M%.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f"
             % (sx, sy0, sx + 150, sy0 - 12, TX - 70, gap_y, TX - 20, gap_y,
                tx - 40, gap_y, tx, gap_y + 2, tx, ty0 - 2),
             stroke="orange", width=2.2, arrow=True)
    # NVLink fan-out inside the node, through the NVSwitch below the GPUs
    for g in tg:
        if g == 3:
            continue
        gx = pos[g][0]
        svg.path("M%.1f %.1f Q %.1f %.1f %.1f %.1f" % (tx, ty1, (tx + gx) / 2, ty1 + 22, gx, ty1 + 1),
                 stroke="blue", width=1.5, arrow=True)
svg.text(30, 422, "橙线 IB：每个目标节点只发一次", size=12, color="orange", weight="600")
svg.text(242, 422, "（50 GB/s；dispatch 的数据是 FP8）", size=12, color="orange")
svg.text(30, 444, "蓝弧 NVLink：到达后在节点内转发（160 GB/s）", size=12, color="blue", weight="600")

# accounting box: below node A and left of x = 360, so that the IB curves pass above it
bx, by = 30, 300
lines = [
    ("不限节点：最多要经 IB 发 8 份（8t）", "ink"),
    ("限制 M = 4：IB 至多发 4 份（4t），通信减半", "ink"),
    ("每次 IB 到达后，节点内顺带扇出 160/50 = 3.2 个专家", "ink"),
    ("最多可选 4 × 3.2 ≈ 13 个而通信开销不变（实际 8 个）", "ink"),
    ("通信内核只用 132 个 SM 中的 20 个（10 个通道）", "muted"),
]
for i, (t, col) in enumerate(lines):
    svg.text(bx, by + i * 22, t, size=12, color=col)
svg.save("l12-node-routing.svg")
