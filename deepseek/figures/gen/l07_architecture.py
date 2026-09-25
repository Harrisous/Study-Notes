"""Figure 7.8: dense FFN vs top-2 MoE vs fine-grained segmentation vs + shared expert isolation.

Redrawn after DeepSeekMoE (arXiv v1 fig. 2 / ACL fig. 1), with an extra panel (0) for the dense FFN.
Block widths are proportional to parameters: one "unit" = one standard FFN. Panels (a)-(c) hold
8 units of expert parameters and activate 2 units per token; the dense FFN in (0) is a single FFN
of 2 units, i.e. the same per-token compute. Combination counts are computed with math.comb.
Writes deepseek/figures/l07-architecture.svg
"""

from math import comb

from l07_svg import SVG, comma

W = 740
PANEL_H = 128
TOP = 40
H = TOP + 4 * PANEL_H + 44
svg = SVG(W, H, "稠密 FFN、传统 top-2 MoE、细粒度专家切分、再加共享专家隔离的结构对比")

RX0, RW = 256, 376          # expert row: x range, 376 px = 8 standard FFNs
UNIT = RW / 8.0

svg.text(16, 22, "每块的宽度 ∝ 参数量（1 份 = 一个标准 FFN）。每一行蓝色与紫色的宽度之和都等于 2 份。", size=12,
         color="muted")

panels = [
    dict(title="(0) 稠密 FFN", cfg=["一个中间维为 2d_{ff} 的 FFN", "每个 token 全部计算"],
         blocks=[("dense", 2.0)], chosen=set(), shared=0,
         note="总参数 2 份 = 激活参数 2 份；没有路由"),
    dict(title="(a) 传统 top-2 路由", cfg=["N = 8 个专家，每个 1 份", "每 token 激活 K = 2 个"],
         blocks=[("r", 1.0)] * 8, chosen={2, 5}, shared=0,
         note="总 8 份，激活 2 份；可选的专家组合 C(8, 2) = %s 种" % comma(comb(8, 2))),
    dict(title="(b) + 细粒度专家切分", cfg=["切成 mN = 16 个，每个 1/2 份", "激活 mK = 4 个（m = 2）"],
         blocks=[("r", 0.5)] * 16, chosen={1, 4, 10, 13}, shared=0,
         note="总 8 份，激活仍是 2 份；组合 C(16, 4) = %s 种" % comma(comb(16, 4))),
    dict(title="(c) + 共享专家隔离", cfg=["1 个共享 + 15 个路由，各 1/2 份", "共享总是激活，路由再选 3 个"],
         blocks=[("s", 0.5)] + [("r", 0.5)] * 15, chosen={4, 8, 12}, shared=1,
         note="总 8 份，激活仍是 2 份；路由组合 C(15, 3) = %s 种" % comma(comb(15, 3))),
]

for p_i, P in enumerate(panels):
    y = TOP + p_i * PANEL_H
    if p_i:
        svg.line(16, y - 10, W - 16, y - 10, color="rule", width=1)
    svg.text(16, y + 22, P["title"], size=13, weight="600")
    svg.text(16, y + 42, P["cfg"][0], size=12, color="ink")
    svg.text(16, y + 60, P["cfg"][1], size=12, color="ink")
    row_y = y + 34
    bh = 36
    # router bar (MoE panels only), spanning the routed experts
    x = RX0
    xs = []
    for kind, w in P["blocks"]:
        xs.append((x, w * UNIT))
        x += w * UNIT
    if P["title"].startswith("(0)"):
        pass
    else:
        rs = xs[P["shared"]][0]
        re = xs[-1][0] + xs[-1][1]
        svg.rect(rs + 2, y + 6, re - rs - 4, 16, fill="violet", fill_opacity=0.14, stroke="violet", width=1, rx=3)
        svg.text((rs + re) / 2, y + 18, "路由器：打分后取 top-%d" % len(P["chosen"]), size=11, anchor="middle",
                 color="violet")
    # blocks
    for k, ((kind, w), (bx, bw)) in enumerate(zip(P["blocks"], xs)):
        pad = 2
        if kind == "dense":
            svg.rect(bx + pad, row_y, bw - 2 * pad, bh, fill="blue", fill_opacity=0.75, rx=4)
            svg.text(bx + bw / 2, row_y + 23, "FFN", size=12, anchor="middle", color="surface", weight="600")
            continue
        if kind == "s":
            svg.rect(bx + pad, row_y, bw - 2 * pad, bh, fill="violet", fill_opacity=0.7, rx=4)
            svg.text(bx + bw / 2, row_y + 23, "S", size=12, anchor="middle", color="surface", weight="600")
            continue
        on = k in P["chosen"]
        svg.rect(bx + pad, row_y, bw - 2 * pad, bh, fill="blue" if on else "muted",
                 fill_opacity=0.75 if on else 0.12, stroke=None if on else "rule", width=1, rx=4)
        idx = k + 1
        svg.text(bx + bw / 2, row_y + 23, str(idx), size=11 if bw < 40 else 12, anchor="middle",
                 color="surface" if on else "muted", weight="600" if on else None)
        if on:
            svg.line(bx + bw / 2, y + 22, bx + bw / 2, row_y - 1, color="violet", width=1.2)
    # input and output
    svg.arrow(RX0 - 28, row_y + bh / 2, RX0 - 4, row_y + bh / 2, color="ink", width=1.4)
    svg.text(RX0 - 42, row_y + bh / 2 + 5, "*u*_{t}", size=13, anchor="middle")
    end = xs[-1][0] + xs[-1][1]
    ox = max(end, RX0 + RW) + 8
    if end < ox - 12:
        svg.line(end + 2, row_y + bh / 2, ox, row_y + bh / 2, color="green", width=1.4)
    svg.arrow(ox, row_y + bh / 2, ox + 26, row_y + bh / 2, color="green", width=1.4)
    svg.text(ox + 30, row_y + bh / 2 - 2, "Σ+残差", size=11, color="muted")
    svg.text(ox + 30, row_y + bh / 2 + 14, "→ *h*_{t}′", size=12, color="green")
    svg.text(RX0, row_y + bh + 20, P["note"], size=12, color="muted")

# legend
ly = H - 18
items = [("blue", 0.75, "被选中（计算）的专家"), ("muted", 0.12, "未被选中（跳过）"), ("violet", 0.7, "共享专家（总是计算）")]
x = 16
for col, op, lab in items:
    svg.rect(x, ly - 11, 14, 14, fill=col, fill_opacity=op, stroke="rule" if col == "muted" else None, width=1, rx=2)
    svg.text(x + 20, ly, lab, size=12)
    x += 20 + len(lab) * 12 + 24

svg.save("l07-architecture.svg")
print(comb(8, 2), comb(16, 4), comb(15, 3))
