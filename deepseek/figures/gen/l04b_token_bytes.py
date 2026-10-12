"""Figure 4.7 of the blackboard version of lesson 4: one token's footprint in the KV cache.

Top: the toy model of the blackboard example (2 layers, 2 KV heads, d_h = 4, bf16) drawn cell by
cell: one cell = one number = 2 bytes. Bottom: the same product of factors, once for the toy model and
once for Llama-2-7B (32 layers, 32 KV heads, d_h = 128), with the products computed here.
Writes deepseek/figures/l04b-token-bytes.svg
"""

from l04_svg import SVG, esc, fmt
from l04_svg import c as colour

TOY = dict(kv=2, ng=2, dh=4, l=2, s=2)
LLAMA = dict(kv=2, ng=32, dh=128, l=32, s=2)


def prod(m):
    return m["kv"] * m["ng"] * m["dh"] * m["l"] * m["s"]


W, H = 740, 372
svg = SVG(W, H, "一个 token 在缓存里长什么样：先数格子，再乘起来")

CELL, CH, GAP = 24, 20, 14
for layer in (1, 2):
    x0 = 16 + (layer - 1) * 362
    y0 = 14
    svg.rect(x0, y0, 346, 150, fill="amber", fill_opacity=0.05, stroke="amber", width=1.3, rx=8)
    svg.text(x0 + 12, y0 + 22, "第 %d 层" % layer, size=13, weight="700")
    svg.text(x0 + 334, y0 + 22, "16 个数 = 32 字节", size=12, anchor="end", color="amber", weight="600")
    for r, (name, col) in enumerate((("K", "violet"), ("V", "green"))):
        ry = y0 + 44 + r * 40
        svg.text(x0 + 22, ry + 15, name, size=13, anchor="middle", color=col, weight="700")
        for head in range(TOY["ng"]):
            hx = x0 + 40 + head * (TOY["dh"] * CELL + GAP)
            for c in range(TOY["dh"]):
                svg.rect(hx + c * CELL, ry, CELL - 2, CH, fill=col, fill_opacity=0.22, stroke=col, width=0.8, rx=2)
        svg.text(x0 + 40 + 2 * (TOY["dh"] * CELL + GAP) + 4, ry + 15, "8 个数", size=12, color="muted")
    for head in range(TOY["ng"]):
        hx = x0 + 40 + head * (TOY["dh"] * CELL + GAP)
        svg.text(hx + TOY["dh"] * CELL / 2, y0 + 138, "KV 头 %d（d_{h} = 4）" % (head + 1), size=12,
                 anchor="middle", color="muted")


def chips(y, label, m, result):
    svg.text(16, y - 10, label, size=13, weight="700")
    # (value, words, symbol, subscript): the symbol is set in italic serif like the formula (4.13)
    items = [(m["kv"], "K 与 V", None, None), (m["ng"], "KV 头 ", "n", "g"), (m["dh"], "每头 ", "d", "h"),
             (m["l"], "层数 ", "l", None), (m["s"], "字节 ", "s", None)]
    cw, gap = 86, 22
    x = 16
    for i, (v, words, sym, sub) in enumerate(items):
        svg.rect(x, y, cw, 50, fill="amber", fill_opacity=0.08, stroke="amber", width=1.1, rx=6)
        svg.text(x + cw / 2, y + 20, "{:,}".format(v), size=15, anchor="middle", weight="700")
        lab = esc(words)
        if sym:
            lab += ('<tspan font-style="italic" font-family="Georgia, \'Times New Roman\', serif">%s</tspan>'
                    % sym)
        if sub:
            lab += '<tspan baseline-shift="sub" font-size="11">%s</tspan>' % sub
        svg.raw('<text x="%s" y="%s" font-size="12" text-anchor="middle" fill="%s">%s</text>'
                % (fmt(x + cw / 2), fmt(y + 37), colour("muted"), lab))
        x += cw
        if i < len(items) - 1:
            svg.text(x + gap / 2, y + 27, "×", size=15, anchor="middle")
            x += gap
    svg.text(x + 12, y + 28, result, size=14, color="amber", weight="700")


t = prod(TOY)
chips(214, "玩具模型（黑板上的例子）", TOY, "= %d 字节" % t)
b = prod(LLAMA)
chips(306, "Llama-2-7B（MHA）", LLAMA, "= {:,} 字节 = {} KiB".format(b, b // 1024))
svg.save("l04b-token-bytes.svg")
