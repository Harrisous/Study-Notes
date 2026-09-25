"""Figure 4.10: MHA / GQA / MQA head sharing (after Ainslie et al. 2023, Fig. 2) and the mean-pooling
conversion used for uptraining (after their Fig. 1). n_h = 8 query heads; GQA with n_g = 2 groups.
Cache per token per layer: 2 n_g d_h elements -> 16 d_h, 4 d_h, 2 d_h. Schematic.
Writes deepseek/figures/l04-mqa-gqa.svg
"""

from l04_svg import SVG

NH = 8
W, H = 720, 436
svg = SVG(W, H, "MHA、GQA、MQA 的头共享方式，以及把多头检查点转成 GQA 的均值池化")

PW = 236
QW, QH, QG = 20, 22, 6


def panel(ox, oy, ng, title, sub):
    svg.text(ox + PW / 2 - 8, oy, title, size=14, anchor="middle", weight="600")
    svg.text(ox + PW / 2 - 8, oy + 18, sub, size=12, anchor="middle", color="muted")
    qx0 = ox + (PW - 16 - (NH * QW + (NH - 1) * QG)) / 2
    qy = oy + 40
    qs = []
    for i in range(NH):
        x = qx0 + i * (QW + QG)
        svg.rect(x, qy, QW, QH, fill="blue", fill_opacity=0.35, stroke="blue", width=1, rx=3)
        qs.append(x + QW / 2)
    svg.text(qx0 - 6, qy + 15, "Q", size=12, anchor="end", color="blue", weight="600")
    # kv heads
    per = NH // ng
    ky = qy + 110
    span = NH * QW + (NH - 1) * QG
    kvw = 2 * QW + 4 if ng > 1 else 2 * QW + 4
    kxs = []
    for g in range(ng):
        centre = qx0 + span * (g + 0.5) / ng
        kx = centre - kvw / 2
        kxs.append(kx)
        for i in range(g * per, (g + 1) * per):
            svg.line(qs[i], qy + QH, centre, ky - 2, color="muted", width=1.1)
        if ng == NH:
            svg.rect(centre - QW / 2, ky, QW, QH, fill="violet", fill_opacity=0.45, stroke="violet", width=1, rx=3)
            svg.rect(centre - QW / 2, ky + QH + 4, QW, QH, fill="green", fill_opacity=0.45, stroke="green", width=1, rx=3)
        else:
            svg.rect(kx, ky, kvw, QH, fill="violet", fill_opacity=0.45, stroke="violet", width=1.2, rx=3)
            svg.rect(kx, ky + QH + 4, kvw, QH, fill="green", fill_opacity=0.45, stroke="green", width=1.2, rx=3)
    svg.text(qx0 - 6, ky + 15, "K", size=12, anchor="end", color="violet", weight="600")
    svg.text(qx0 - 6, ky + QH + 19, "V", size=12, anchor="end", color="green", weight="600")
    # cache bill
    yb = ky + 2 * QH + 30
    svg.rect(ox + 6, yb - 16, PW - 28, 26, fill="amber", fill_opacity=0.10, rx=5)
    svg.text(ox + PW / 2 - 8, yb + 2, "缓存 2n_{g}d_{h} = %d d_{h} / token / 层" % (2 * ng), size=12, anchor="middle",
             color="amber", weight="600")
    return yb


svg.text(16, 22, "Ⓐ 8 个查询头如何共用键和值（每条连线：这个查询头读哪一对 K、V）", size=14, weight="600")
yb = panel(16, 56, 8, "MHA", "n_{g} = n_{h} = 8：各用各的")
panel(16 + PW, 56, 2, "GQA", "n_{g} = 2：每 4 个查询头一组")
panel(16 + 2 * PW, 56, 1, "MQA", "n_{g} = 1：全部共用一对")

# ---- uptraining strip
y0 = yb + 44
svg.line(16, y0 - 22, W - 16, y0 - 22, color="rule", width=1)
svg.text(16, y0, "Ⓑ 从 MHA 检查点改成 GQA：组内 K、V 投影矩阵取平均，再续训一小段（uptraining）", size=14, weight="600")
mx, my = 40, y0 + 24
for i in range(4):
    x = mx + i * 62
    svg.rect(x, my, 44, 58, fill="violet", fill_opacity=0.14 + 0.08 * i, stroke="violet", width=1.1, rx=4)
    svg.text(x + 22, my + 34, "W^{K}_{%d}" % (i + 1), size=12, anchor="middle")
svg.text(mx + 4 * 62 - 6, my + 34, "→", size=18, color="muted")
svg.text(mx + 4 * 62 + 42, my + 20, "取平均", size=12, anchor="middle", color="muted")
svg.text(mx + 4 * 62 + 42, my + 38, "(1/4)Σ", size=12, anchor="middle", color="muted")
svg.rect(mx + 4 * 62 + 80, my, 44, 58, fill="violet", fill_opacity=0.35, stroke="violet", width=1.6, rx=4)
svg.text(mx + 4 * 62 + 102, my + 34, "W^{K}_{g}", size=12, anchor="middle", weight="600")
svg.text(mx + 4 * 62 + 140, my + 20, "V 的投影同样处理。之后按原预训练配方", size=12)
svg.text(mx + 4 * 62 + 140, my + 38, "再训练原步数的 5%，约 600 TPUv3 芯片·天。", size=12)
svg.text(mx + 4 * 62 + 140, my + 56, "消融：取平均 > 只取第一个头 > 随机初始化。", size=12, color="muted")
svg.save("l04-mqa-gqa.svg")
