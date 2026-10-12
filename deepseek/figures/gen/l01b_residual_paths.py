"""Figure 1.14 of the blackboard version of lesson 1: the 8 gradient paths through 3 residual blocks.

Each block is h <- h + f(h) with the derivative of f taken as the number J = 0.1. Backwards,
the gradient at every block either takes the shortcut (x1) or goes through the layer (x0.1).
The 2^3 = 8 paths are enumerated here, grouped by how many layers they cross, and their
products summed: 1 + 3(0.1) + 3(0.01) + 0.001 = 1.331 = 1.1^3. Without residuals only the
path through all three layers remains (0.001).
Writes deepseek/figures/l01b-residual-paths.svg
"""

import itertools
import math

from l01_svg import SVG

J = 0.1
NL = 3
paths = list(itertools.product((0, 1), repeat=NL))       # 1 = goes through that layer
contrib = {p: J ** sum(p) for p in paths}
total = sum(contrib.values())
assert len(paths) == 8 and abs(total - (1 + J) ** NL) < 1e-12 and abs(total - 1.331) < 1e-12
groups = {k: [p for p in paths if sum(p) == k] for k in range(NL + 1)}
assert [len(groups[k]) for k in range(NL + 1)] == [math.comb(NL, k) for k in range(NL + 1)]

W, H = 760, 400
svg = SVG(W, H, "黑板上的 3 层网络：梯度的 8 条路")

# ---------------------------------------------------------------- top: the unrolled chain
svg.text(10, 22, "反向传播：梯度从输出端（右）进来，往输入端（左）走；每一层都有两条路", size=13, weight="700")
Y = 120                      # the shortcut line
NX = [70, 270, 470, 670]     # nodes h(0) .. h(3)
for k, x in enumerate(NX):
    svg.circle(x, Y, 17, fill="surface", stroke="ink", width=1.6)
    svg.text(x, Y + 4.5, "h", size=13, anchor="middle", weight="600")
    svg.text(x, Y + 36, "第 %d 层输出" % k if k else "输入", size=11, anchor="middle", color="muted")
svg.text(NX[-1] + 30, Y - 22, "梯度 1", size=12, color="ink", weight="600")
for k in range(NL):
    xr, xl = NX[k + 1], NX[k]
    # shortcut: straight green line, arrow pointing left
    svg.arrow(xr - 19, Y, xl + 21, Y, color="green", width=3)
    svg.text((xl + xr) / 2, Y + 18, "捷径 ×1", size=12, anchor="middle", color="green", weight="600")
    # through the layer: an arc above with a box
    bx, by, bw, bh = (xl + xr) / 2 - 46, Y - 78, 92, 30
    svg.path("M%d %d C %d %d, %d %d, %d %d" % (xr - 12, Y - 12, xr - 30, Y - 60, bx + bw + 20, by + bh / 2,
                                                bx + bw, by + bh / 2), stroke="muted", width=1.5)
    svg.path("M%d %d C %d %d, %d %d, %d %d" % (bx, by + bh / 2, bx - 20, by + bh / 2, xl + 30, Y - 60,
                                                xl + 12, Y - 12), stroke="muted", width=1.5, arrow=True)
    svg.box(bx, by, bw, bh, ["第 %d 层 ×0.1" % (k + 1)], color="muted", fill_opacity=0.14, size=12)

# ---------------------------------------------------------------- bottom: the 8 paths
svg.line(10, 178, W - 10, 178, color="rule", width=1)
svg.text(10, 200, "把 8 条路按“穿过几层”分组（小方格从左到右是第 1、2、3 层；实心 = 穿过，×0.1；空心 = 走捷径，×1）",
         size=12, color="muted")
GX = [20, 205, 390, 575]
names = ["穿过 0 层", "穿过 1 层", "穿过 2 层", "穿过 3 层"]
for k in range(NL + 1):
    gx = GX[k]
    gp = groups[k]
    s = sum(contrib[p] for p in gp)
    svg.text(gx, 226, "%s：%d 条" % (names[k], len(gp)), size=12.5, weight="700",
             color="green" if k == 0 else "ink")
    for r, p in enumerate(gp):
        ry = 240 + r * 24
        for c in range(NL):
            filled = p[c] == 1
            svg.rect(gx + c * 20, ry, 16, 16, fill="muted" if filled else None,
                     fill_opacity=0.55 if filled else None, stroke="green" if k == 0 else "muted", width=1.2, rx=2)
        svg.text(gx + 70, ry + 12.5, "= %s" % ("1" if k == 0 else "%g" % contrib[p]), size=12.5,
                 color="green" if k == 0 else "ink")
    svg.text(gx, 240 + 3 * 24 + 22, "小计 %s" % ("1" if k == 0 else "%g" % round(s, 6)), size=13,
             weight="700", color="green" if k == 0 else "ink")
# the no-residual case
gx = GX[3]
svg.rect(gx - 8, 208, 172, 136, stroke="muted", width=1.2, dash="5 4", rx=6)
svg.text(gx, 286, "没有残差的网络，", size=12, color="muted")
svg.text(gx, 304, "只剩这一条路：0.001", size=12, color="muted")

# total
ty = H - 18
svg.text(20, ty, "合计：1 + 0.3 + 0.03 + 0.001 = %.3f = 1.1³。绿色那条路一层也不穿过，贡献恒为 1。" % total,
         size=13, weight="600")
svg.save("l01b-residual-paths.svg")
