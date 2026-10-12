"""Figure 7.9: fine-grained expert segmentation = slicing the FFN's intermediate neurons.

A standard SwiGLU FFN with d = 2048 and d_ff = 5632 (DeepSeekMoE 16B: expert 1408 = 0.25 x 5632)
is a sum over its 5632 intermediate neurons; grouping the neurons into m = 4 slices of 1408 gives
4 small experts whose outputs add up to the original FFN. The table (m = 1, 2, 4, 8) is computed:
per-expert params 3*d*(d_ff/m), total expert params for 16 standard experts, router params 16*m*d.
Writes deepseek/figures/l07-segmentation.svg
"""

from l07_svg import SVG, comma

d, dff, N = 2048, 5632, 16

W, H = 740, 504
svg = SVG(W, H, "细粒度专家切分：把一个 FFN 的中间神经元分组，就得到几个小专家")

X0, X1 = 150, 710
BW = X1 - X0

# ---- row 1: the standard FFN as a strip of neurons
y1 = 58
svg.text(16, y1 - 32, "一个标准 FFN（d = 2048，中间维 d_{ff} = 5632）", size=13, weight="600")
svg.text(16, y1 + 12, "W_{1}, W_{3} 的行", size=12, color="muted")
svg.text(16, y1 + 44, "W_{2} 的列", size=12, color="muted")
svg.rect(X0, y1, BW, 22, fill="blue", fill_opacity=0.18, stroke="blue", width=1.2)
svg.rect(X0, y1 + 30, BW, 22, fill="blue", fill_opacity=0.10, stroke="blue", width=1.2)
for k in range(1, 56):
    x = X0 + BW * k / 56
    svg.line(x, y1, x, y1 + 22, color="blue", width=0.5, opacity=0.35)
    svg.line(x, y1 + 30, x, y1 + 52, color="blue", width=0.5, opacity=0.35)
svg.text(16, y1 - 10, "每一竖条代表约 100 个中间神经元；神经元 j 拥有 W_{1}、W_{3} 的第 j 行和 W_{2} 的第 j 列，各 2048 个数",
         size=11, color="muted")

# ---- row 2: sliced into 4 experts
y2 = 160
svg.text(16, y2 - 12, "按神经元切成 m = 4 段", size=13, weight="600")
gap = 10
sw = (BW - 3 * gap) / 4
for k in range(4):
    x = X0 + k * (sw + gap)
    svg.rect(x, y2, sw, 22, fill="blue", fill_opacity=0.35, stroke="blue", width=1.2, rx=2)
    svg.rect(x, y2 + 30, sw, 22, fill="blue", fill_opacity=0.22, stroke="blue", width=1.2, rx=2)
    svg.text(x + sw / 2, y2 + 16, "小专家 %d · 1408 个" % (k + 1), size=12, anchor="middle")
    svg.text(x + sw / 2, y2 + 46, "%s 参数" % comma(3 * d * dff // 4), size=12, anchor="middle", color="ink")
    # dashed guides from the strip above
    xa = X0 + BW * k / 4
    svg.line(xa + 2, y1 + 54, x + 2, y2 - 2, color="muted", width=0.8, dash="3 3")
    svg.line(xa + BW / 4 - 2, y1 + 54, x + sw - 2, y2 - 2, color="muted", width=0.8, dash="3 3")

# ---- row 3: the identity
y3 = 246
svg.text(16, y3, "为什么切得开：", size=13, weight="600")
svg.text(X0, y3, "FFN(*u*) = Σ_{j=1}^{5632} *w*_{2,j} · SiLU(*w*_{1,j}^{⊤}*u*) · (*w*_{3,j}^{⊤}*u*)", size=13)
svg.text(X0 + 62, y3 + 24, "= FFN_{1}(*u*) + FFN_{2}(*u*) + FFN_{3}(*u*) + FFN_{4}(*u*)", size=13)
svg.text(X0, y3 + 46, "每个神经元的贡献只依赖它自己的三组权重，分组求和不改变总和。", size=12, color="muted")

# ---- table: m = 1, 2, 4, 8
y4 = 326
svg.text(16, y4, "参数账（16 个标准专家，按 m 切分）", size=13, weight="600")
cols = [("m", 16, "start"), ("专家数 mN", 70, "start"), ("每专家中间维", 180, "start"),
        ("每专家参数", 300, "start"), ("专家参数合计", 420, "start"), ("路由器参数 mN·d", 560, "start")]
hy = y4 + 26
for name, x, a in cols:
    svg.text(x, hy, name, size=12, color="muted", weight="600")
svg.line(16, hy + 7, W - 16, hy + 7, color="rule", width=1)
for r, m in enumerate((1, 2, 4, 8)):
    yy = hy + 28 + r * 26
    inter = dff // m
    pe = 3 * d * inter
    tot = N * m * pe
    router = N * m * d
    hl = (m == 4)
    if hl:
        svg.rect(10, yy - 17, W - 20, 24, fill="blue", fill_opacity=0.07, rx=3)
    vals = [str(m), str(N * m), str(inter), comma(pe), comma(tot), comma(router)]
    for (name, x, a), v in zip(cols, vals):
        svg.text(x, yy, v, size=12, color="blue" if (hl and name != "m") else "ink",
                 weight="600" if name == "专家参数合计" else None)
    # router bar, to scale within its column
    svg.rect(640, yy - 11, 60 * router / (N * 8 * d), 10, fill="violet", fill_opacity=0.6)
svg.text(16, H - 10, "专家参数合计始终是 553,648,128，不随 m 变；路由器参数随 m 线性增长（紫色短条），但仍不到专家参数的 0.05%。",
         size=12, color="muted")

svg.save("l07-segmentation.svg")
