"""Figure 4.11: quality versus decoding cost for MHA / MQA / GQA, from the papers' tables.

Left  (Shazeer 2019, WMT14 EN-DE, Tables 1-2): decoder amortized time per token (us) vs dev BLEU.
      MHA 46 us / 26.7, MQA 3.8 us / 26.5. The four MHA variants (h, d_k=d_v) = (1,128), (2,64),
      (4,32), (8,16) shrink the per-layer K width to h*d_k = 128, the same KV size as MQA, and reach
      25.8 / 26.2 / 26.1 / 25.8 dev BLEU (no timing reported): drawn as a band.
Right (Ainslie et al. 2023, T5-XXL, Table 1): inference time per sample (s) vs average score.
      MHA-Large 0.37 / 46.0, MHA-XXL 1.51 / 47.2, MQA-XXL 0.24 / 46.6, GQA-8-XXL 0.28 / 47.1.
All values from lesson-04-doc C31, C32, C41. Writes deepseek/figures/l04-mqa-gqa-quality.svg
"""

from l04_svg import SVG

SHZ = [("MHA", 46.0, 26.7, "violet"), ("MQA", 3.8, 26.5, "blue")]
REDUCED = (25.8, 26.2, 26.1, 25.8)
GQA = [("MHA-Large", 0.37, 46.0, "muted"), ("MHA-XXL", 1.51, 47.2, "violet"),
       ("MQA-XXL", 0.24, 46.6, "blue"), ("GQA-8-XXL", 0.28, 47.1, "green")]
assert round(46 / 3.8, 1) == 12.1 and round(1.51 / 0.28, 2) == 5.39

W, H = 720, 400
svg = SVG(W, H, "MHA、MQA、GQA 的质量与解码成本（据 Shazeer 2019 表 1–2 与 Ainslie 等人 2023 表 1）")


def axes(x0, x1, y0, y1, xt, yt, fx, fy, xlab, ylab):
    for v in xt:
        x = fx(v)
        svg.line(x, y0, x, y1, color="grid", width=1)
        svg.text(x, y1 + 16, ("%g" % v), size=11, anchor="middle", color="muted")
    for v in yt:
        y = fy(v)
        svg.line(x0, y, x1, y, color="grid", width=1)
        svg.text(x0 - 6, y + 4, ("%.1f" % v), size=11, anchor="end", color="muted")
    svg.line(x0, y1, x1, y1, color="muted", width=1.2)
    svg.line(x0, y0, x0, y1, color="muted", width=1.2)
    svg.text((x0 + x1) / 2, y1 + 38, xlab, size=12, anchor="middle", color="muted")
    svg.text(x0 - 40, (y0 + y1) / 2, ylab, size=12, anchor="middle", color="muted", rotate=-90)


# ---- left panel
L0, L1, T0, T1 = 64, 330, 50, 290
fxL = lambda v: L0 + v / 50 * (L1 - L0)
fyL = lambda v: T1 - (v - 25.6) / (27.0 - 25.6) * (T1 - T0)
svg.text(16, 24, "Ⓐ Shazeer：WMT14 英→德翻译", size=14, weight="600")
axes(L0, L1, T0, T1, (0, 10, 20, 30, 40, 50), (25.6, 26.0, 26.4, 26.8), fxL, fyL,
     "解码器每 token 摊销时间（μs）", "dev BLEU")
svg.rect(L0 + 1, fyL(max(REDUCED)), L1 - L0 - 2, fyL(min(REDUCED)) - fyL(max(REDUCED)), fill="red", fill_opacity=0.10)
# two lines, both inside the plot: one long line started left of the y-axis
svg.text(L1 - 8, fyL(min(REDUCED)) - 26, "KV 同样缩小 8 倍的 MHA", size=12, anchor="end", color="red")
svg.text(L1 - 8, fyL(min(REDUCED)) - 9, "（减 h 或 d_{k}）：BLEU 25.8–26.2", size=12, anchor="end", color="red")
for name, x, y, col in SHZ:
    svg.circle(fxL(x), fyL(y), 6, fill=col)
svg.text(fxL(46) - 10, fyL(26.7) + 4, "MHA 46 μs，26.7", size=12, anchor="end", color="violet", weight="600")
svg.text(fxL(3.8) + 10, fyL(26.5) + 4, "MQA 3.8 μs，26.5", size=12, color="blue", weight="600")
svg.text(fxL(25), fyL(26.5) + 36, "解码快 12 倍，BLEU 只低 0.2", size=12, anchor="middle", weight="600")

# ---- right panel
R0, R1 = 430, 700
fxR = lambda v: R0 + v / 1.6 * (R1 - R0)
fyR = lambda v: T1 - (v - 45.8) / (47.4 - 45.8) * (T1 - T0)
svg.text(376, 24, "Ⓑ Ainslie 等人：T5 摘要/翻译/问答平均", size=14, weight="600")
axes(R0, R1, T0, T1, (0, 0.4, 0.8, 1.2, 1.6), (46.0, 46.4, 46.8, 47.2), fxR, fyR,
     "每个样本的推理时间（s）", "平均分")
for name, x, y, col in GQA:
    svg.circle(fxR(x), fyR(y), 6, fill=col)
svg.text(fxR(1.51) - 10, fyR(47.2) + 4, "MHA-XXL 1.51 s，47.2", size=12, anchor="end", color="violet", weight="600")
svg.text(fxR(0.28) + 10, fyR(47.1) + 18, "GQA-8-XXL 0.28 s，47.1", size=12, color="green", weight="600")
svg.text(fxR(0.24) + 10, fyR(46.6) + 4, "MQA-XXL 0.24 s，46.6", size=12, color="blue", weight="600")
svg.text(fxR(0.37) + 10, fyR(46.0) + 4, "MHA-Large 0.37 s，46.0", size=12, color="muted", weight="600")

svg.text(16, H - 36, "两张图各自的数据来自不同的论文、任务与硬件，不能横向比较；只比较同一张图里的点。",
         size=12, color="muted")
svg.text(16, H - 16, "GQA-8 用 8 组 KV 头；XXL 与 Large 是 T5 的两个规模，MQA/GQA 版本由 MHA-XXL 转换后续训 5% 得到。",
         size=12, color="muted")
svg.save("l04-mqa-gqa-quality.svg")
