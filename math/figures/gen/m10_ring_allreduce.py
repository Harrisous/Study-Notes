"""Figure M10.9: ring all-reduce on p = 4 GPUs = reduce-scatter (3 steps) + all-gather (3 steps).

Every GPU's vector is cut into p chunks.  Reduce-scatter step s (s = 0..p-2): GPU i sends chunk
(i - s) mod p to GPU i+1, which adds it to its own copy.  After p-1 steps GPU i owns the full sum
of chunk (i+1) mod p.  All-gather step s: GPU i forwards the finished chunk (i + 1 - s) mod p to
GPU i+1, which overwrites its copy.  Each cell shows which GPUs' contributions it already holds.
The grids are produced by simulating exactly these rules.
Writes math/figures/m10-ring-allreduce.svg
"""

from m10_svg import SVG

p = 4
state = [[{i} for c in range(p)] for i in range(p)]      # state[gpu][chunk] = set of contributors
snap_rs = [[[set(x) for x in row] for row in state]]
sent_rs = []
for s in range(p - 1):
    sends = [(i, (i - s) % p) for i in range(p)]
    sent_rs.append(sends)
    new = [[set(x) for x in row] for row in state]
    for i, c in sends:
        new[(i + 1) % p][c] |= state[i][c]
    state = new
    snap_rs.append([[set(x) for x in row] for row in state])
for i in range(p):
    assert state[i][(i + 1) % p] == set(range(p))
snap_ag = []
sent_ag = []
for s in range(p - 1):
    sends = [(i, (i + 1 - s) % p) for i in range(p)]
    sent_ag.append(sends)
    new = [[set(x) for x in row] for row in state]
    for i, c in sends:
        assert state[i][c] == set(range(p))
        new[(i + 1) % p][c] = set(state[i][c])
    state = new
    snap_ag.append([[set(x) for x in row] for row in state])
assert all(state[i][c] == set(range(p)) for i in range(p) for c in range(p))

W, H = 740, 432
svg = SVG(W, H, "环形 all-reduce：p = 4 时的 reduce-scatter 与 all-gather")
CELL = 28
SLOT = 168
GX = 70


def label(cs):
    if cs == set(range(p)):
        return "Σ"
    return "".join(str(x) for x in sorted(cs))


def grid(x, y, snap, title, outgoing=None, first=False):
    svg.text(x + 2 * CELL, y - 10, title, size=12, anchor="middle", weight="600")
    for i in range(p):
        if first:
            svg.text(x - 8, y + i * CELL + CELL / 2 + 4, "GPU %d" % i, size=11, anchor="end", color="muted")
        for c in range(p):
            cs = snap[i][c]
            full = cs == set(range(p))
            fill = "green" if full else "blue"
            op = 0.55 if full else 0.08 + 0.14 * (len(cs) - 1)
            svg.rect(x + c * CELL, y + i * CELL, CELL - 2, CELL - 2, fill=fill, fill_opacity=op, rx=3)
            if outgoing and (i, c) in outgoing:
                svg.rect(x + c * CELL - 1, y + i * CELL - 1, CELL, CELL, stroke="ink", width=1.8, rx=4)
            svg.text(x + c * CELL + (CELL - 2) / 2, y + i * CELL + CELL / 2 + 3, label(cs), size=11,
                     anchor="middle", color="surface" if full else "ink", weight="600" if full else None)
    for c in range(p):
        svg.text(x + c * CELL + (CELL - 2) / 2, y + p * CELL + 13, "块%d" % c, size=11, anchor="middle", color="muted")


svg.text(16, 22, "① reduce-scatter：每步把一块发给右邻，右邻把它加到自己那块上（黑框 = 下一步要发出的块）",
         size=13, weight="600")
Y1 = 62
titles = ["开始：各有一份", "第 1 步后", "第 2 步后", "第 3 步后"]
for k in range(p):
    outgoing = set(sent_rs[k]) if k < p - 1 else set(sent_ag[0])
    grid(GX + k * SLOT, Y1, snap_rs[k], titles[k], outgoing=outgoing, first=(k == 0))
svg.text(GX + 3 * SLOT + 2 * CELL, Y1 + p * CELL + 32, "每卡各有一块“全和”Σ", size=12, anchor="middle", color="green",
         weight="600")

svg.text(16, 250, "② all-gather：把加好的 Σ 块沿环再传一圈，收到的直接覆盖", size=13, weight="600")
Y2 = 290
for k in range(p - 1):
    outgoing = set(sent_ag[k + 1]) if k + 1 < p - 1 else None
    grid(GX + k * SLOT, Y2, snap_ag[k], "第 %d 步后" % (p + k), outgoing=outgoing, first=(k == 0))

# summary box
bx, by = GX + 3 * SLOT - 16, Y2 - 18
svg.rect(bx, by, 172, 150, fill="grid", fill_opacity=0.5, stroke="rule", rx=6)
lines = ["消息共 S 字节", "每步每卡发 S/p", "共 2(p−1) = 6 步", "每卡共发 2(p−1)S/p",
         "= 1.5S（p = 4）", "p 很大时趋于 2S"]
for j, t in enumerate(lines):
    svg.text(bx + 12, by + 24 + j * 22, t, size=12, weight="600" if j == 3 else None)

svg.save("m10-ring-allreduce.svg")
print("ok: every GPU ends with the full sum of every chunk")
