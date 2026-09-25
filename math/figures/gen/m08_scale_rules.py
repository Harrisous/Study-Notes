"""Figure M8.11: where each scale rule puts a block's largest value on the FP4 grid.

For a block with absolute maximum amax (log axis, 1/2 .. 32) the curves show amax / s, i.e. the
position of the block maximum on the E2M1 axis before rounding, for three rules implemented in
m08_block_scaling.scale():
    OCP MX example algorithm 1:  s = 2^(floor(log2 amax) - 2)      -> amax/s in [4, 8)
    power of two, rounded up:    s = 2^ceil(log2(amax / 6))        -> amax/s in (3, 6]
    E4M3 scale (NVFP4 style):    s = E4M3(amax / 6)                -> amax/s close to 6
The y axis is labelled with the E2M1 grid values; anything above 6 saturates to 6.
Writes math/figures/m08-scale-rules.svg
"""

import math

import numpy as np

from m08_block_scaling import scale
from m08_svg import SVG

L0, L1 = -1, 5
amax = 2.0 ** np.linspace(L0, L1, 3000)
rules = [("pow2_up", "blue", None, ("2 的幂，向上取整", "（UE8M0 常用做法）")),
         ("ocp_floor", "violet", "6 3", ("2 的幂，向下取整", "（OCP MX 示例算法 1）")),
         ("e4m3", "green", None, ("E4M3 缩放因子", "（NVFP4 与 V4.1）"))]

W, H = 760, 400
X0, X1, Y0, Y1 = 70, 600, 40, 330
YMAX = 8.4


def px(a):
    return X0 + (math.log2(a) - L0) / (L1 - L0) * (X1 - X0)


def py(v):
    return Y1 - v / YMAX * (Y1 - Y0)


svg = SVG(W, H, "三种缩放规则把块内最大值放在 FP4 网格的什么位置")
svg.rect(X0, py(YMAX), X1 - X0, py(6) - py(YMAX), fill="red", fill_opacity=0.07)
# the note for the red band sits above the plot, where no curve crosses it
svg.rect(X0, Y0 - 22, 14, 11, fill="red", fill_opacity=0.12)
svg.text(X0 + 20, Y0 - 12, "红色区域（高于 6）：超出 E2M1 的范围，被钳成 6，块内最大值变小", size=12, color="red")
for g in (0.5, 1, 1.5, 2, 3, 4, 6):
    svg.line(X0, py(g), X1, py(g), color="grid" if g != 6 else "red", width=1 if g != 6 else 1.4,
             dash=None if g != 6 else "5 3")
    svg.text(X0 - 8, py(g) + 4, "%g" % g, size=11, anchor="end", color="muted")
for e in range(L0, L1 + 1):
    x = px(2.0 ** e)
    svg.line(x, Y1, x, Y1 + 4, color="muted", width=1)
    lab = "1/2" if e == -1 else str(2 ** e)
    svg.text(x, Y1 + 18, lab, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "块内最大绝对值 amax（对数刻度）", size=12, anchor="middle", color="muted")
svg.text(20, (Y0 + Y1) / 2, "amax / s：块内最大值落在 FP4 轴上的位置", size=12, anchor="middle",
         color="muted", rotate=-90)

for rule, col, dash, lab in rules:
    ys = np.array([a / scale(a, "E2M1", rule) for a in amax])
    # break the path at the jumps so the sawtooth has no vertical connectors
    segs, cur = [], []
    for a, y in zip(amax, ys):
        if cur and abs(y - cur[-1][1]) > 0.8:
            segs.append(cur)
            cur = []
        cur.append((a, min(y, YMAX)))
    segs.append(cur)
    for sgm in segs:
        svg.path("M" + " L".join("%.1f %.1f" % (px(a), py(y)) for a, y in sgm), stroke=col,
                 width=3.2 if rule == "pow2_up" else 1.8, dash=dash)

# legend on the right
ly = 70
for rule, col, dash, lab in rules:
    svg.line(X1 + 14, ly - 4, X1 + 38, ly - 4, color=col, width=2.4, dash=dash)
    svg.text(X1 + 44, ly, lab[0], size=12, color=col)
    svg.text(X1 + 44, ly + 16, lab[1], size=11, color="muted")
    ly += 54
svg.text(X1 + 14, ly + 4, "蓝线与紫色虚线在 4 到 6", size=11, color="muted")
svg.text(X1 + 14, ly + 20, "之间重合：块内最大值的", size=11, color="muted")
svg.text(X1 + 14, ly + 36, "尾数不超过 1.5 时两种", size=11, color="muted")
svg.text(X1 + 14, ly + 52, "取整给出同一个缩放因子", size=11, color="muted")

# worked points
pts = [(7.0, "ocp_floor", "amax = 7：7 → 钳成 6"), (0.9, "ocp_floor", "amax = 0.9：7.2 → 钳成 6"),
       (3.01, "pow2_up", "amax = 3.01：只用到 3，4 和 6 空着")]
for a, rule, lab in pts:
    y = a / scale(a, "E2M1", rule)
    svg.circle(px(a), py(min(y, YMAX)), 4.2, fill="ink")


def text_width(s, size=12):
    # rough advance widths: CJK and full-width punctuation ~1 em, everything else ~0.56 em
    return sum(size if ord(ch) > 0x2E80 else size * 0.56 for ch in s)


for (a, rule, lab), (dx, dy, anc) in zip(pts, ((6, -8, "start"), (8, 16, "start"), (0, 24, "middle"))):
    y = a / scale(a, "E2M1", rule)
    tx, ty = px(a) + dx, py(min(y, YMAX)) + dy
    w = text_width(lab)
    left = tx - w / 2 if anc == "middle" else tx
    # an opaque label plate (surface colour, follows the theme) so the dashed curves that
    # pass behind a label do not show through the gaps between its characters
    svg.rect(left - 3, ty - 12, w + 6, 16, fill="surface", fill_opacity=0.92, rx=3)
    svg.text(tx, ty, lab, size=12, anchor=anc)

svg.save("m08-scale-rules.svg")
