"""Figure 6.16: one FIM training example in PSM order, with DeepSeek-Coder's real sentinel tokens.

Document: 'def add(a, b):\\n    return a + b\\n', cut at character level into
prefix 'def add(a, b):\\n    ret', middle 'urn a ', suffix '+ b\\n' (research notes E11).
Sentinels from the released tokenizer (notes §2.2, §3.3): <｜fim▁begin｜> 32016, <｜fim▁hole｜> 32015,
<｜fim▁end｜> 32017, <｜end▁of▁sentence｜> 32014.
Writes deepseek/figures/l06-fim-psm.svg
"""

from l06_svg import SVG

doc = "def add(a, b):\n    return a + b\n"
prefix, middle, suffix = "def add(a, b):\n    ret", "urn a ", "+ b\n"
assert prefix + middle + suffix == doc

W, H = 760, 420
svg = SVG(W, H, "FIM 的 PSM 格式：把中段挪到末尾")


def show(s):
    return s.replace("\n", "↵")


MONO = "ui-monospace, 'Cascadia Mono', Consolas, monospace"


def chip(x, y, text, col, fill_op=0.12, mono=True, w=None, size=13, weight=None):
    width = w or (sum(12.5 if ord(ch) > 0x2000 else 7.6 for ch in text) * size / 13 + 14)
    svg.rect(x, y, width, 30, fill=col, fill_opacity=fill_op, stroke=col, width=1.3, rx=5)
    fam = ' font-family="%s"' % MONO if mono else ""
    svg.raw('<text x="%.1f" y="%.1f" font-size="%d"%s fill="var(--fig-%s, %s)"%s>%s</text>' % (
        x + width / 2, y + 20, size, fam, "ink", "#1C1B19", ' text-anchor="middle"' +
        (' font-weight="%s"' % weight if weight else ""),
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")))
    return x + width


# row 1: the original document, colour-coded
y1 = 44
svg.text(16, y1 - 12, "① 原始文档，按字符随机切成三段（↵ 表示换行）", size=13, weight="600")
x = 16
x = chip(x, y1, show(prefix), "blue", 0.10)
x = chip(x, y1, show(middle), "green", 0.22)
x = chip(x, y1, show(suffix), "blue", 0.10)
svg.text(16, y1 + 50, "前缀 prefix", size=12, color="blue")
svg.text(16 + (len(show(prefix)) * 8.2 + 16) + 4, y1 + 50, "中段 middle", size=12, color="green")

# row 2: the PSM training sequence
y2 = 150
svg.text(16, y2 - 14, "② PSM 训练序列：前缀、后缀在前，中段挪到最后（以 0.5 的概率对每个文档这样做）", size=13,
         weight="600")
x = 16
x = chip(x, y2, "<｜fim▁begin｜>", "violet", 0.14, size=12)
x = chip(x + 3, y2, show(prefix), "blue", 0.10)
x = chip(x + 3, y2, "<｜fim▁hole｜>", "violet", 0.14, size=12)
x = chip(x + 3, y2, show(suffix), "blue", 0.10)
x2 = 16
y2b = y2 + 40
x2 = chip(x2, y2b, "<｜fim▁end｜>", "violet", 0.14, size=12)
x2 = chip(x2 + 3, y2b, show(middle), "green", 0.22)
x2 = chip(x2 + 3, y2b, "<｜end▁of▁sentence｜>", "violet", 0.14, size=12)
svg.text(x2 + 12, y2b + 20, "（接上一行，是同一条序列）", size=12, color="muted")
svg.text(16, y2b + 52, "发布的分词器中的 token id：fim▁begin 32016，fim▁hole 32015，fim▁end 32017，",
         size=12, color="muted")
svg.text(16, y2b + 70, "end▁of▁sentence 32014。报告正文写作 fim_start / fim_hole / fim_end。", size=12, color="muted")

# row 3: inference
y3 = 318
svg.text(16, y3 - 14, "③ 推理：光标停在代码中间时，只给到 <｜fim▁end｜>，让模型写出中段", size=13,
         weight="600")
x = 16
x = chip(x, y3, "<｜fim▁begin｜>", "violet", 0.14, size=12)
x = chip(x + 3, y3, show(prefix), "blue", 0.10)
x = chip(x + 3, y3, "<｜fim▁hole｜>", "violet", 0.14, size=12)
x = chip(x + 3, y3, show(suffix), "blue", 0.10)
x = chip(x + 3, y3, "<｜fim▁end｜>", "violet", 0.14, size=12)
svg.arrow(x + 6, y3 + 15, x + 40, y3 + 15, color="green", width=1.8)
svg.text(x + 46, y3 + 20, "生成：urn a", size=13, color="green", weight="600")
svg.text(16, y3 + 56, "读到 <｜fim▁end｜> 时，模型已经看过前缀和后缀，接下来要预测的恰好是中段；",
         size=12, color="muted")
svg.text(16, y3 + 74, "所以训练不需要新的损失函数，重排后的序列照常做下一词预测。", size=12, color="muted")

svg.save("l06-fim-psm.svg")
