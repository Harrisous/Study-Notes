"""Figure 1.14: where the parameters of GPT-2 live.

(a) GPT-2 small, exact count 124,439,808, split into wte / wpe / attention / FFN / LayerNorm.
(b) the four GPT-2 sizes: embedding share of the exact total, and non-embedding count vs 12 L d^2.
Formula (reference-docs/lesson-01-doc.md E14): (V + T_max) d + L (12 d^2 + 13 d) + 2 d, tied output.
Writes deepseek/figures/l01-params.svg
"""

from l01_svg import SVG

V, T = 50257, 1024
sizes = [("small", 12, 768), ("medium", 24, 1024), ("large", 36, 1280), ("xl", 48, 1600)]


def parts(L, d):
    return {
        "wte": V * d,
        "wpe": T * d,
        "attn": L * (4 * d * d + 4 * d),
        "ffn": L * (8 * d * d + 5 * d),
        "ln": L * 4 * d + 2 * d,
    }


totals = {}
for name, L, d in sizes:
    p = parts(L, d)
    totals[name] = sum(p.values())
assert totals["small"] == 124_439_808 and totals["medium"] == 354_823_168
assert totals["large"] == 774_030_080 and totals["xl"] == 1_557_611_200

W, H = 760, 400
s = SVG(W, H, "GPT-2 的参数都在哪里")

# ---------------------------------------------------------------- (a) GPT-2 small
p = parts(12, 768)
tot = totals["small"]
s.text(10, 22, "(a) GPT-2 small 的 124,439,808 个参数", size=13, weight="700")
X0, X1, Y = 20, 740, 44
segs = [("wte 词嵌入", p["wte"], "amber", 0.55), ("wpe", p["wpe"], "amber", 0.25),
        ("注意力 ×12", p["attn"], "blue", 0.45), ("FFN ×12", p["ffn"], "muted", 0.45), ("LN", p["ln"], "ink", 0.6)]
x = X0
for k, (name, v, col, op) in enumerate(segs):
    w = v / tot * (X1 - X0)
    s.rect(x, Y, max(w, 1.2), 34, fill=col, fill_opacity=op, stroke=col, width=0.8)
    if w > 60:
        s.text(x + w / 2, Y + 22, name, size=12.5, anchor="middle", weight="600")
        s.text(x + w / 2, Y + 54, "%.2fM · %.1f%%" % (v / 1e6, 100 * v / tot), size=12, anchor="middle", color="muted")
    x += w
xw = X0 + (p["wte"]) / tot * (X1 - X0)
s.line(xw + 2, Y + 36, xw + 20, Y + 76, color="muted", width=0.8)
s.text(xw + 22, Y + 88, "wpe 0.79M（0.6%）", size=11.5, color="muted")
s.line(X1 - 1, Y + 36, X1 - 30, Y + 76, color="muted", width=0.8)
s.text(X1, Y + 88, "LayerNorm 共 25 个：0.04M（0.03%）", size=11.5, color="muted", anchor="end")
s.text(X0, Y + 116, "每块 7,087,872 = 注意力 2,362,368（4d²+4d）+ FFN 4,722,432（8d²+5d）+ 2 个 LN 3,072（4d）。"
       "12 块合计 85.05M，占 68.4%。", size=12)

# ---------------------------------------------------------------- (b) four sizes
s.line(10, 190, 750, 190, color="rule", width=1)
s.text(10, 212, "(b) 四个规模：嵌入表的占比随模型变大迅速下降；12Ld² 与非嵌入参数的差别不到 0.2%", size=13, weight="700")
BX0, BX1 = 178, 556
for k, (name, L, d) in enumerate(sizes):
    y = 232 + k * 38
    pp = parts(L, d)
    t = totals[name]
    emb = pp["wte"] + pp["wpe"]
    non = t - emb
    s.text(BX0 - 10, y + 18, "%s（L=%d, d=%d）" % (name, L, d), size=12, anchor="end")
    we = emb / t * (BX1 - BX0)
    s.rect(BX0, y + 4, we, 22, fill="amber", fill_opacity=0.55, stroke="amber", width=0.8)
    s.rect(BX0 + we, y + 4, (BX1 - BX0) - we, 22, fill="ink", fill_opacity=0.10, stroke="muted", width=0.8)
    s.text(BX0 + we + 6, y + 20, "嵌入 %.1f%%" % (100 * emb / t), size=12, color="amber", weight="600")
    s.text(BX1 + 10, y + 12, "总 %s" % "{:,}".format(t), size=11.5, color="ink")
    s.text(BX1 + 10, y + 27, "非嵌入 %.2fM，12Ld² %.2fM" % (non / 1e6, 12 * L * d * d / 1e6), size=11, color="muted")
s.text(BX0, 392, "每条横条都按 100% 画：金色 = wte + wpe，灰色 = 12 个（或更多）块与 LayerNorm", size=11.5, color="muted")

s.save("l01-params.svg")
