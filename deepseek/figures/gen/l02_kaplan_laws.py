"""Figure 2.11: Kaplan et al. (2020) single-variable power laws, computed from eqs. (1.1)-(1.3).

  L(N)     = (N_c / N)^0.076,      N_c = 8.8e13 non-embedding parameters
  L(D)     = (D_c / D)^0.095,      D_c = 5.4e13 tokens
  L(C_min) = (C_c / C_min)^0.050,  C_c = 3.1e8 PF-days
Loss in nats/token on WebText2, tokenized with byte-pair encoding, n_vocab = 50257 (section 2; the paper does
not call it "GPT-2's tokenizer", so the figure does not either); the constants have no absolute meaning.
Solid = roughly the range the paper's experiments covered, dashed = extrapolation. N: 768 to 1.5e9
non-embedding parameters and D: 22M to 23B tokens are the paper's model and data ranges (section 3). For C_min
the paper gives no fit range: section 1.2 only says the relation holds "across eight orders of magnitude in
C_min", and the figure 13 caption says 1-layer networks (the lump near 1e-5 PF-days) were excluded from the
fit. So the solid C_min segment is drawn as an illustration of eight decades, 1e-7 to 1e1 PF-days, whose top
end is about the 6ND of Kaplan's largest configuration (1.5e9 x 2.5e5 steps x 2^19 tokens = 1.2e21 FLOPs,
about 14 PF-days). Marked values (lesson-02-doc.md section 4.2):
  L(85M) = 2.865, L(1.5B) = 2.304, L(175B) = 1.604; L(23B) = 2.090, L(300B) = 1.638; L(1) = 2.658, L(3640) = 1.764
Writes deepseek/figures/l02-kaplan-laws.svg
"""

import math

import numpy as np

from l02_svg import SVG, LogAxis

W, H = 720, 362
svg = SVG(W, H, "Kaplan 的三条幂律：损失对参数量、数据量、算力")

panels = [
    ("L(N) = (8.8×10^{13} / N)^{0.076}", "非嵌入参数 N", 8.8e13, 0.076, (1e3, 1e12), (768, 1.5e9),
     [(85e6, "85M：2.87"), (1.5e9, "1.5B：2.30"), (175e9, "175B：1.60")], "blue", (3, 6, 9, 12)),
    ("L(D) = (5.4×10^{13} / D)^{0.095}", "训练 token 数 D", 5.4e13, 0.095, (1e7, 1e12), (22e6, 23e9),
     [(23e9, "23B：2.09"), (300e9, "300B：1.64")], "green", (7, 9, 11)),
    ("L(C) = (3.1×10^{8} / C)^{0.050}", "算力 C_{min}（PF-days）", 3.1e8, 0.050, (1e-9, 1e4), (1e-7, 1e1),
     [(1.0, "1：2.66"), (3640, "3,640：1.76")], "violet", (-9, -6, -3, 0, 3)),
]
PWID, GAP, X0 = 196, 36, 54
Y0, Y1 = 58, 282
ay = LogAxis(1.4, 8.0, Y1, Y0)
for i, (title, xlab, Xc, a, (lo, hi), (flo, fhi), marks, col, ticks) in enumerate(panels):
    px0 = X0 + i * (PWID + GAP)
    px1 = px0 + PWID
    ax = LogAxis(lo, hi, px0, px1)
    svg.text((px0 + px1) / 2, 22, title, size=12, anchor="middle", color=col, weight="600")
    for v in (1.5, 2, 3, 4, 6, 8):
        y = ay(v)
        svg.line(px0, y, px1, y, color="grid", width=1)
        if i == 0:
            svg.text(px0 - 6, y + 4, ("%g" % v), size=11, anchor="end", color="muted")
    for e in ticks:
        x = ax(10.0 ** e)
        svg.line(x, Y0, x, Y1, color="grid", width=1)
        svg.text(x, Y1 + 16, "10^{%d}" % e, size=11, anchor="middle", color="muted")
    svg.line(px0, Y1, px1, Y1, color="muted", width=1.2)
    svg.line(px0, Y0, px0, Y1, color="muted", width=1.2)
    svg.text((px0 + px1) / 2, Y1 + 38, xlab, size=12, anchor="middle", color="muted")
    xs = np.logspace(math.log10(lo), math.log10(hi), 120)
    ys = (Xc / xs) ** a
    ok = (ys <= 8.0) & (ys >= 1.4)
    fit = ok & (xs >= flo) & (xs <= fhi)
    ext = ok & ~((xs > flo) & (xs < fhi))
    svg.path("M" + " L".join("%.1f %.1f" % (ax(x), ay(y)) for x, y in zip(xs[fit], ys[fit])), stroke=col, width=2.6)
    for seg in (xs < flo, xs > fhi):
        m = seg & ok
        if m.sum() > 1:
            svg.path("M" + " L".join("%.1f %.1f" % (ax(x), ay(y)) for x, y in zip(xs[m], ys[m])), stroke=col,
                     width=1.6, dash="5 4")
    for j, (xv, lab) in enumerate(marks):
        x, y = ax(xv), ay((Xc / xv) ** a)
        svg.circle(x, y, 3.8, fill=col)
        if x > px1 - 70:
            svg.text(x - 6, y + 17, lab, size=11, color="ink", anchor="end")
        else:
            svg.text(x + 6, y - 8 if j % 2 == 0 else y + 16, lab, size=11, color="ink")
svg.text(X0 - 40, (Y0 + Y1) / 2, "损失（nats/token，对数）", size=12, anchor="middle", color="muted", rotate=-90)
svg.text(X0, H - 8, "实线：论文实验大致覆盖的范围；虚线：外推。常数依赖 WebText2 与它的 BPE 分词，没有绝对意义。", size=12,
         color="ink")

svg.save("l02-kaplan-laws.svg")
