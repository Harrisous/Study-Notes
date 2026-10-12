"""Figure 9.3: MHA / GQA / MQA / MLA, redrawn after DeepSeek-V2 (arXiv:2405.04434v5) Figure 3.

Each panel draws 8 query heads, as the paper does.  What is cached at inference gets a thick amber
outline.  The paper's MLA panel is a "simplified illustration" and omits the decoupled RoPE key
k^R; it is added here (red) because it is cached too (paper sec. 2.1.3, appendix C).
Per-token-per-layer cache counts below each panel use V2's n_h = d_h = 128, d_c = 512,
d_h^R = 64; GQA uses n_g = 8 (the DeepSeek 67B setting) although 4 groups are drawn.
Writes deepseek/figures/l09-four-attention.svg
"""

from l09_svg import SVG

n_h, d_h, d_c, d_r = 128, 128, 512, 64
counts = {
    "MHA": ("2n_{h}d_{h}", 2 * n_h * d_h),
    "GQA": ("2n_{g}d_{h}，n_{g}=8", 2 * 8 * d_h),
    "MQA": ("2d_{h}", 2 * d_h),
    "MLA": ("d_{c}+d_{h}^{R}", d_c + d_r),
}
assert counts["MHA"][1] == 32768 and counts["MLA"][1] == 576

W, H = 764, 356
svg = SVG(W, H, "MHA、GQA、MQA、MLA 四种注意力：推理时各缓存什么")
PX0, PW, GAP = 46, 174, 6
BW, BH, BG = 16, 18, 4           # head box
YQ, YK, YV = 58, 120, 146
titles = {
    "MHA": ("多头注意力 MHA", "每个头一对 K、V"),
    "GQA": ("分组查询 GQA", "每组头共用一对 K、V"),
    "MQA": ("多查询 MQA", "所有头共用一对 K、V"),
    "MLA": ("多头潜在注意力 MLA", "缓存一个潜向量 + 一个小键"),
}
capab = {"MHA": "Strong（强）", "GQA": "Moderate（中）", "MQA": "Weak（弱）", "MLA": "Stronger（更强）"}

# row labels
svg.text(PX0 - 8, YQ + 13, "Q", size=13, anchor="end", color="blue", weight="700")
svg.text(PX0 - 8, YK + 13, "K", size=13, anchor="end", color="violet", weight="700")
svg.text(PX0 - 8, YV + 13, "V", size=13, anchor="end", color="green", weight="700")

for p, kind in enumerate(("MHA", "GQA", "MQA", "MLA")):
    x0 = PX0 + p * (PW + GAP)
    svg.rect(x0, 6, PW, 300, fill="grid", fill_opacity=0.35, rx=8)
    t, sub = titles[kind]
    svg.text(x0 + PW / 2, 26, t, size=13, anchor="middle", weight="700")
    svg.text(x0 + PW / 2, 44, sub, size=11.5, anchor="middle", color="muted")
    inner = 8 * BW + 7 * BG
    qx = [x0 + (PW - inner) / 2 + i * (BW + BG) for i in range(8)]
    for x in qx:
        svg.rect(x, YQ, BW, BH, fill="blue", fill_opacity=0.25, stroke="blue", width=1.1, rx=3)
    if kind in ("MHA", "MLA"):
        groups = [[i] for i in range(8)]
    elif kind == "GQA":
        groups = [[2 * g, 2 * g + 1] for g in range(4)]
    else:
        groups = [list(range(8))]
    cached = kind != "MLA"
    for g in groups:
        if len(g) == 8:
            kx, kw = x0 + PW / 2 - 18, 36
        else:
            kx = qx[g[0]]
            kw = qx[g[-1]] + BW - kx
        # attention lines query -> key
        for i in g:
            svg.line(qx[i] + BW / 2, YQ + BH, kx + kw / 2, YK, color="ink", width=0.9, opacity=0.55)
        if cached:
            svg.rect(kx, YK, kw, BH, fill="violet", fill_opacity=0.3, stroke="amber", width=2.4, rx=3)
            svg.rect(kx, YV, kw, BH, fill="green", fill_opacity=0.3, stroke="amber", width=2.4, rx=3)
        else:
            svg.rect(kx, YK, kw, BH, fill="violet", fill_opacity=0.08, stroke="violet", width=1, rx=3, dash="3 2")
            svg.rect(kx, YV, kw, BH, fill="green", fill_opacity=0.08, stroke="green", width=1, rx=3, dash="3 2")
    if kind == "MLA":
        lx, ly, lw, lh = x0 + PW / 2 - 44, 212, 64, 22
        left, right = qx[0], qx[-1] + BW
        svg.polygon([(left, YV + BH + 3), (right, YV + BH + 3), (lx + lw, ly - 2), (lx, ly - 2)],
                    fill="orange", fill_opacity=0.10)
        svg.text(x0 + PW / 2 - 12, 188, "投影 W^{UK}、W^{UV}", size=11, anchor="middle", color="orange")
        svg.rect(lx, ly, lw, lh, fill="orange", fill_opacity=0.28, stroke="amber", width=2.4, rx=4)
        svg.text(lx + lw / 2, ly + 16, "*c*^{KV}", size=13, anchor="middle", color="orange", weight="700")
        svg.text(lx + lw / 2, ly + lh + 16, "潜向量", size=11, anchor="middle", color="orange")
        rx_ = lx + lw + 12
        svg.rect(rx_, ly, 26, lh, fill="red", fill_opacity=0.18, stroke="amber", width=2.4, rx=4)
        svg.text(rx_ + 13, ly + 16, "*k*^{R}", size=12, anchor="middle", color="red", weight="700")
        svg.text(rx_ + 13, ly + lh + 16, "原图未画", size=11, anchor="middle", color="red")
    else:
        svg.text(x0 + PW / 2, 200, "K、V 原样写进缓存", size=11.5, anchor="middle", color="amber")
    f, n = counts[kind]
    svg.text(x0 + PW / 2, 268, "每 token 每层缓存", size=11.5, anchor="middle", color="muted")
    svg.text(x0 + PW / 2, 286, "%s = %s" % (f, format(n, ",")), size=12.5, anchor="middle", weight="700",
             color="orange" if kind == "MLA" else "ink")
    svg.text(x0 + PW / 2, 302, "论文表 1：%s" % capab[kind], size=11, anchor="middle", color="muted")

# legend
ly = 334
svg.rect(PX0, ly - 12, 22, 14, fill="violet", fill_opacity=0.3, stroke="amber", width=2.4, rx=3)
svg.text(PX0 + 30, ly, "推理时缓存（琥珀色粗框）", size=12)
svg.rect(PX0 + 220, ly - 12, 22, 14, fill="violet", fill_opacity=0.08, stroke="violet", width=1, rx=3, dash="3 2")
svg.text(PX0 + 250, ly, "不缓存：需要时从潜向量投影出来，或被吸收掉（§3）", size=12)
svg.save("l09-four-attention.svg")
