"""Figure 1.16: Post-LN vs Pre-LN, redrawn from Xiong et al. (2020) Fig. 1 in the course colours,
plus the hidden-state norms of their Lemma 2 (at initialisation, under the paper's simplified
assumptions): Post-LN E||x^{post,5}_l||^2 = 1.5 d for every layer l > 0 (the sum that enters the
second LayerNorm; after that LayerNorm the norm^2 is exactly d); Pre-LN
(1 + l/2) d <= E||x^{pre}_l||^2 <= (1 + 3l/2) d, drawn for l = 0..12 with d normalised to 1.
Writes deepseek/figures/l01-layernorm.svg
"""

import numpy as np

from l01_svg import SVG

W, H = 760, 420
s = SVG(W, H, "Post-LN 与 Pre-LN：LayerNorm 放在残差主干上还是分支里")

BW, BH = 150, 30


def column(x0, title, sub, rows, bypass_pairs, color_final=None):
    s.text(x0, 22, title, size=13.5, weight="700")
    s.text(x0, 40, sub, size=12, color="muted")
    ys = []
    y = 56
    cx = x0 + 30 + BW / 2
    for lab, col, kind in rows:
        if kind == "io":
            s.text(cx, y + 18, lab, size=13, anchor="middle", weight="600")
            h = 26
        elif kind == "add":
            s.plus(cx, y + 12, r=10)
            h = 24
        else:
            s.box(x0 + 30, y, BW, BH, [(lab, {"size": 12.5})], color=col, fill_opacity=0.12)
            h = BH
        ys.append((y, h))
        y += h + 14
    for k in range(1, len(rows)):
        y_prev, h_prev = ys[k - 1]
        s.arrow(cx, y_prev + h_prev, cx, ys[k][0], width=1.3)
    for a, b in bypass_pairs:
        ya = ys[a][0] + ys[a][1] + 6
        yb = ys[b][0] + 12
        s.path("M%.1f %.1f H%.1f V%.1f H%.1f" % (cx, ya, x0 + 12, yb, cx - 11), stroke="ink", width=1.2, arrow=True)
    return ys


# Post-LN: x -> attn -> add -> LN -> FFN -> add -> LN
post = [("x_{l}", None, "io"), ("多头注意力", "blue", "box"), ("+", None, "add"), ("LayerNorm", "muted", "box"),
        ("FFN", "muted", "box"), ("+", None, "add"), ("LayerNorm", "muted", "box"), ("x_{l+1}", None, "io")]
column(8, "(a) Post-LN（原 Transformer）", "LN 在残差主干上，每个子层之后", post, [(0, 2), (3, 5)])
# Pre-LN: x -> LN -> attn -> add -> LN -> FFN -> add
pre = [("x_{l}", None, "io"), ("LayerNorm", "muted", "box"), ("多头注意力", "blue", "box"), ("+", None, "add"),
       ("LayerNorm", "muted", "box"), ("FFN", "muted", "box"), ("+", None, "add"), ("x_{l+1}", None, "io")]
column(236, "(b) Pre-LN（GPT-2 采用）", "LN 在分支里，主干上只有加法", pre, [(0, 3), (3, 6)])
s.text(236 + 30 + BW / 2, 400, "全部 L 层之后再加一个最终 LayerNorm", size=11.5, anchor="middle", color="muted")
s.text(8 + 30 + BW / 2, 400, "主干上的每一步都要经过 LN", size=11.5, anchor="middle", color="muted")

# ---------------------------------------------------------------- right: Lemma 2 norms
X0, X1, Y0, Y1 = 520, 740, 70, 330
LMAX, YMAX = 12, 20
ls = np.arange(0, LMAX + 1)
lo = 1 + ls / 2
hi = 1 + 3 * ls / 2
post_n = np.full_like(ls, 1.5, dtype=float)


def px(l):
    return X0 + l / LMAX * (X1 - X0)


def py(v):
    return Y1 - v / YMAX * (Y1 - Y0)


s.line(478, 12, 478, 410, color="rule", width=1)
s.text(492, 22, "(c) 初始化时隐藏状态的范数平方 / d", size=13, weight="700")
s.text(492, 40, "Xiong 等（2020）引理 2，简化假设下", size=12, color="muted")
for l in (0, 3, 6, 9, 12):
    s.line(px(l), Y0, px(l), Y1, color="grid", width=1)
    s.text(px(l), Y1 + 16, str(l), size=11, anchor="middle", color="muted")
for v in (0, 5, 10, 15, 20):
    s.line(X0, py(v), X1, py(v), color="grid", width=1)
    s.text(X0 - 6, py(v) + 4, str(v), size=11, anchor="end", color="muted")
s.line(X0, Y1, X1, Y1, color="muted", width=1.1)
s.text((X0 + X1) / 2, Y1 + 34, "层号 l", size=12, anchor="middle", color="muted")
band = [(px(l), py(v)) for l, v in zip(ls, hi)] + [(px(l), py(v)) for l, v in zip(ls[::-1], lo[::-1])]
s.polygon(band, fill="orange", fill_opacity=0.18, stroke="orange", width=1.2)
# Lemma 2's Post-LN statement is for l > 0 (x^{post,5} only exists inside a layer), so start at l = 1
s.path("M" + " L".join("%.1f %.1f" % (px(l), py(v)) for l, v in zip(ls[1:], post_n[1:])), stroke="blue", width=2.2)
s.text(px(12) - 4, py(19) - 8, "Pre-LN：(1 + l/2) 到 (1 + 3l/2)", size=11.5, anchor="end", color="orange",
       weight="600")
s.text(px(12) - 4, py(12.5) + 30, "随层数线性增长", size=11.5, anchor="end", color="orange")
s.text(px(4.8), py(1.5) - 6, "Post-LN（LN 前的和）：1.5", size=11.5, color="blue", weight="600")

s.save("l01-layernorm.svg")
