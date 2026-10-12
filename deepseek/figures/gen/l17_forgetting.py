"""Figure 17.14: the "optical forgetting" idea of DeepSeek-OCR §5 / fig. 13, made computable.

A 10-round conversation, 1,000 text tokens per round (hypothetical). Recent 2 rounds stay as text;
older rounds are rendered to images that get smaller the older they are: Large (400) -> Base (256)
-> Small (100) -> Tiny (64), two rounds each. KV bytes at V4.1-Flash's 890 B/token (dsv41 abstract),
assuming a vision token costs the same KV as a text token (author's assumption).
Fox table 2 only measured Tiny/Small, so the precision notes exist only for those two:
Small at ~10x: 96.8 % (9.7x) / 91.5 % (10.6x); Tiny at ~15.6x: 85.9 % (15.1x) / 79.3 % (16.5x).
The paper never ran such a multi-round experiment; this is its proposal.
Writes deepseek/figures/l17-forgetting.svg
"""

from l17_svg import SVG

ROUND = 1000
PLAN = [("Tiny", 64), ("Tiny", 64), ("Small", 100), ("Small", 100), ("Base", 256), ("Base", 256),
        ("Large", 400), ("Large", 400), ("文本", 1000), ("文本", 1000)]
TOTAL = sum(n for _, n in PLAN)
assert TOTAL == 3640
KV = 890
A_BYTES, B_BYTES = 10 * ROUND * KV, TOTAL * KV
assert round(A_BYTES / 1e6, 2) == 8.9 and round(B_BYTES / 1e6, 2) == 3.24

W, H = 760, 400
svg = SVG(W, H, "光学遗忘：越久远的对话渲染得越小、越模糊（据 DeepSeek-OCR 论文图 13 的设想重绘，数字为笔者算例）")

X0, CW, GAP = 30, 48, 7
BASE_Y, HMAX = 250, 170
svg.text(20, 22, "一段 10 轮的对话，每轮 1,000 个文本 token（假设）", size=13, weight="700")
svg.arrow(X0 + 10 * (CW + GAP) - 20, 44, X0, 44, color="muted", width=1.3)
svg.text(X0 + 14, 38, "越往左越久远", size=12, color="muted")

for i, (mode, n) in enumerate(PLAN):
    x = X0 + i * (CW + GAP)
    h = n / ROUND * HMAX
    is_text = mode == "文本"
    col = "green" if is_text else "orange"
    age = 9 - i
    op = 0.5 if is_text else 0.12 + 0.05 * (8 - age)
    svg.rect(x, BASE_Y - h, CW, h, fill=col, fill_opacity=op, stroke=col, width=1.2, rx=3)
    # page glyph inside: fewer, fainter lines for older rounds
    if not is_text:
        k = max(1, int(h // 9))
        for j in range(k):
            svg.line(x + 8, BASE_Y - h + 6 + j * 8, x + CW - 8 - (j % 2) * 10, BASE_Y - h + 6 + j * 8,
                     color="orange", width=1.6, opacity=max(0.15, 0.85 - 0.12 * age))
    svg.text(x + CW / 2, BASE_Y - h - 8, str(n), size=12, anchor="middle", color=col, weight="700")
    svg.text(x + CW / 2, BASE_Y + 18, "第%d轮" % (i + 1), size=11.5, anchor="middle", color="muted")
    svg.text(x + CW / 2, BASE_Y + 36, mode, size=12, anchor="middle", color=col)
    if not is_text:
        svg.text(x + CW / 2, BASE_Y + 54, "%.1f×" % (ROUND / n), size=11.5, anchor="middle", color="orange")
svg.line(X0 - 6, BASE_Y, X0 + 10 * (CW + GAP) - 2, BASE_Y, color="ink", width=1.2)

# precision notes (only Tiny/Small were measured on Fox)
svg.text(X0, BASE_Y + 84, "Fox 表 2 的对应格：Tiny ≈ 15.6× 时 85.9%（15.1×）/ 79.3%（16.5×）；Small ≈ 10× 时 96.8%（9.7×）/ 91.5%（10.6×）；",
         size=11.5, color="orange")
svg.text(X0, BASE_Y + 102, "Base、Large 在 Fox 上没有测过。这是逐字 OCR 的精度，不是“理解这段历史”的保真度。", size=11.5, color="orange")

# KV ledger card
CX = 590
svg.rect(CX, 60, 160, 176, fill="amber", fill_opacity=0.06, stroke="amber", width=1.2, rx=6)
svg.text(CX + 10, 80, "KV 账（890 B/token）", size=12.5, weight="700", color="amber")
svg.text(CX + 10, 104, "全保留为文本", size=12)
svg.text(CX + 10, 122, "10,000 token", size=12, color="muted")
svg.text(CX + 150, 122, "%.2f MB" % (A_BYTES / 1e6), size=12, anchor="end", weight="700")
svg.text(CX + 10, 150, "按左图遗忘", size=12)
svg.text(CX + 10, 168, "{:,} token".format(TOTAL), size=12, color="muted")
svg.text(CX + 150, 168, "%.2f MB" % (B_BYTES / 1e6), size=12, anchor="end", weight="700")
svg.line(CX + 10, 182, CX + 150, 182, color="rule", width=1)
svg.text(CX + 10, 202, "压缩比 %.2f（剩 %.1f%%）" % (A_BYTES / B_BYTES, B_BYTES / A_BYTES * 100), size=12.5,
         color="amber", weight="700")
svg.text(CX + 10, 222, "（设视觉 token 与文本同价）", size=11, color="muted")
svg.text(20, 386, "论文只提出了这个设想（§5 与图 13），没有做过多轮对话的实验；轮次、模式与精度对应都是笔者为说明而设的算例。",
         size=11.5, color="muted")
svg.save("l17-forgetting.svg")
