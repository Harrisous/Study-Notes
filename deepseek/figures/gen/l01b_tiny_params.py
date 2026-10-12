"""Figure 1.17 of the blackboard version of lesson 1: a GPT-2-style toy with 308 parameters.

V = 10, T_max = 4, d = 4, L = 1, FFN 4d wide, biases, Pre-LN, tied output layer. Every
parameter is one small square, drawn in the real shape of its matrix (rows x columns); weights
are dark, biases light, LayerNorm gamma/beta black. The counts are computed here and checked
against formula (1.30): (V + T_max) d + L (12 d^2 + 13 d) + 2 d = 308.
Writes deepseek/figures/l01b-tiny-params.svg
"""

from l01_svg import SVG

V, TM, D, L = 10, 4, 4, 1
F = 4 * D
PARTS = [  # (label, rows, cols, colour, kind)
    ("wte 词嵌入", V, D, "amber", "w"),
    ("wpe 位置表", TM, D, "amber", "w"),
    ("ln_1", 2, D, "ink", "ln"),
    ("c_attn：Q、K、V", D, 3 * D, "blue", "w"),
    ("", 1, 3 * D, "blue", "b"),
    ("W^O", D, D, "blue", "w"),
    ("", 1, D, "blue", "b"),
    ("ln_2", 2, D, "ink", "ln"),
    ("W_1 升维", D, F, "muted", "w"),
    ("", 1, F, "muted", "b"),
    ("W_2 降维", F, D, "muted", "w"),
    ("", 1, D, "muted", "b"),
    ("ln_f", 2, D, "ink", "ln"),
]
total = sum(r * c for _, r, c, _, _ in PARTS)
formula = (V + TM) * D + L * (12 * D * D + 13 * D) + 2 * D
assert total == formula == 308, (total, formula)
by_colour = {}
for _, r, c, col, _ in PARTS:
    by_colour[col] = by_colour.get(col, 0) + r * c
assert by_colour == {"amber": 56, "blue": 80, "muted": 148, "ink": 24}, by_colour

W, H = 760, 510
svg = SVG(W, H, "黑板上的玩具 GPT：308 个参数，一格一个")
P = 12            # cell pitch
S = 10            # cell size


def grid(x, y, rows, cols, colour, kind):
    op = {"w": 0.6, "b": 0.22, "ln": 0.7}[kind]
    for i in range(rows):
        for j in range(cols):
            svg.rect(x + j * P, y + i * P, S, S, fill=colour, fill_opacity=op, rx=1.5)
    return cols * P, rows * P


def block(x, y, items, caption, count):
    """items: list of (rows, cols, colour, kind); stacked vertically. Returns width."""
    wmax = 0
    yy = y
    for rows, cols, colour, kind in items:
        w, h = grid(x, yy, rows, cols, colour, kind)
        wmax = max(wmax, w)
        yy += h + 4
    svg.text(x, yy + 12, caption, size=12, weight="600")
    svg.text(x, yy + 28, count, size=12, color="muted")
    return wmax


svg.text(10, 22, "V = 10、T_max = 4、d = 4、L = 1 的 GPT-2 式模型：每个小方格是一个参数，按矩阵的真实形状摆放",
         size=12.5, weight="700")

# row 1: embeddings and attention
Y1 = 44
x = 16
w = block(x, Y1, [(V, D, "amber", "w")], "wte 词嵌入", "10×4 = 40")
x += w + 44
w = block(x, Y1, [(TM, D, "amber", "w")], "wpe 位置表", "4×4 = 16")
x += w + 44
w = block(x, Y1, [(2, D, "ink", "ln")], "ln_1", "γ, β = 8")
x += w + 44
w = block(x, Y1, [(D, 3 * D, "blue", "w"), (1, 3 * D, "blue", "b")], "c_attn：Q、K、V", "4×12 + 12 = 60")
x += w + 44
w = block(x, Y1, [(D, D, "blue", "w"), (1, D, "blue", "b")], "W^{O}", "4×4 + 4 = 20")
assert x + w < W

# row 2: FFN, final LN, tied output
Y2 = 222
svg.line(10, Y2 - 18, W - 10, Y2 - 18, color="rule", width=1)
x = 16
w = block(x, Y2, [(2, D, "ink", "ln")], "ln_2", "γ, β = 8")
x += w + 44
w = block(x, Y2, [(D, F, "muted", "w"), (1, F, "muted", "b")], "W_{1} 升维", "4×16 + 16 = 80")
x += w + 44
w = block(x, Y2, [(F, D, "muted", "w"), (1, D, "muted", "b")], "W_{2} 降维", "16×4 + 4 = 68")
x += w + 44
w = block(x, Y2, [(2, D, "ink", "ln")], "ln_f", "γ, β = 8")
x += w + 44
svg.rect(x, Y2, D * P - 2, V * P - 2, stroke="amber", width=1.3, dash="4 3", rx=2)
svg.text(x, Y2 + V * P + 16, "输出层", size=12, weight="600")
svg.text(x, Y2 + V * P + 32, "与 wte 共享 = 0", size=12, color="muted")
sx = x + D * P + 30
svg.text(sx, Y2 + 14, "合计", size=13, weight="700")
lines = [("嵌入 56（18%）", "amber"), ("注意力 80", "blue"), ("FFN 148", "muted"), ("LayerNorm 24", "ink")]
for k, (t, col) in enumerate(lines):
    svg.rect(sx, Y2 + 28 + k * 22, 12, 12, fill=col, fill_opacity=0.6, rx=2)
    svg.text(sx + 18, Y2 + 38 + k * 22, t, size=12)
svg.text(sx, Y2 + 38 + 4 * 22 + 6, "= %d" % total, size=14, weight="700")
assert sx + 120 < W, sx

# legend
ly = H - 16
svg.rect(16, ly - 11, 10, 10, fill="muted", fill_opacity=0.6, rx=1.5)
svg.text(32, ly - 2, "权重矩阵", size=12)
svg.rect(106, ly - 11, 10, 10, fill="muted", fill_opacity=0.22, rx=1.5)
svg.text(122, ly - 2, "偏置", size=12)
svg.rect(170, ly - 11, 10, 10, fill="ink", fill_opacity=0.7, rx=1.5)
svg.text(186, ly - 2, "LayerNorm 的 γ、β", size=12)
svg.text(W - 16, ly - 2, "一个块 = 8 + 60 + 20 + 8 + 80 + 68 = 244 = 12d² + 13d", size=12, anchor="end", color="muted")
svg.save("l01b-tiny-params.svg")
