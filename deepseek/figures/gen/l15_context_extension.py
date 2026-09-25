"""Figure 15.3: continued-training tokens after the V3 base checkpoint.

V3 report section 4.3: two extension phases of 1000 steps each, 32K x batch 1920 and 128K x batch 480
(K = 1024 tokens), i.e. 62.91B tokens each. V3.1 model card: 32K phase "increased 10-fold to 630B",
128K phase "extended by 3.3x to 209B". V3.2 section 2.1.1 (DSA, lesson 14): warm-up 1000 x 16 x 128K = 2.1B,
sparse stage 15000 x 480 x 128K = 943.7B. Linear axis.
Writes deepseek/figures/l15-context-extension.svg
"""

from l15_svg import SVG, fmt

V3_32K = 1000 * 1920 * 32 * 1024
V3_128K = 1000 * 480 * 128 * 1024
assert V3_32K == V3_128K == 62_914_560_000
V31_32K, V31_128K = 630e9, 209e9
DSA_WARM = 1000 * 16 * 128 * 1024
DSA_SPARSE = 15000 * 480 * 128 * 1024
assert round(DSA_WARM / 1e9, 1) == 2.1 and round(DSA_SPARSE / 1e9, 1) == 943.7

W, H = 760, 336
svg = SVG(W, H, "V3 之后的续训 token 数：V3.1 的两阶段长上下文扩展与 V3.2 的 DSA 续训")

X0, X1 = 212, 700
VMAX = 1000e9


def bx(v):
    return X0 + v / VMAX * (X1 - X0)


TOP = 30
svg.text(16, 20, "续训 token 数（十亿，线性刻度）", size=13, weight="700")
for t in range(0, 1001, 200):
    x = bx(t * 1e9)
    svg.line(x, TOP, x, 256, color="grid", width=1)
    svg.text(x, 272, str(t), size=11, anchor="middle", color="muted")
svg.text(X1, 286, "B token", size=11, anchor="end", color="muted")

rows = [
    ("32K 阶段", [("V3（2024-12）", V3_32K, "muted", "1000 步 × 1920 条 × 32K = 62.9B"),
                 ("V3.1（2025-08）", V31_32K, "blue", "630B，×%s" % fmt(V31_32K / V3_32K))]),
    ("128K 阶段", [("V3（2024-12）", V3_128K, "muted", "1000 步 × 480 条 × 128K = 62.9B"),
                  ("V3.1（2025-08）", V31_128K, "blue", "209B，×%s" % fmt(V31_128K / V3_128K))]),
    ("DSA 续训", [("V3.2-Exp 预热", DSA_WARM, "violet", "只训 indexer：2.1B"),
                 ("V3.2-Exp 稀疏", DSA_SPARSE, "violet", "943.7B（第 14 课）")]),
]
y = TOP + 10
for gi, (glab, bars) in enumerate(rows):
    svg.text(16, y + 30, glab, size=13, weight="700")
    for lab, v, col, note in bars:
        svg.text(X0 - 8, y + 15, lab, size=12, anchor="end", color="ink")
        svg.rect(X0, y + 2, max(bx(v) - X0, 1.5), 18, fill=col, fill_opacity=0.35, stroke=col, width=1)
        tx = bx(v) + 8
        if tx > X1 - 170:
            tx = bx(v) - 8
            svg.text(tx, y + 15, note, size=12, anchor="end", color="ink", weight="700")
        else:
            svg.text(tx, y + 15, note, size=12, color="ink")
        y += 26
    y += 26

svg.text(16, 308, "V3.1 两阶段合计 630 + 209 = 839B（新闻页写“840B”）；再加 DSA 的 945.8B，V3 基座之后共续训约 1.78T token。",
         size=12, color="muted")
svg.text(16, 326, "对照：V3 的预训练是 14.8T token。“10 倍 / 3.3 倍”的基数是 V3 报告里各 1000 步的 62.9B。", size=12, color="muted")

svg.save("l15-context-extension.svg")
