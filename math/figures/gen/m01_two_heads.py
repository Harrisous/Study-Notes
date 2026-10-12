"""Figure M1.7: two small heads can look at two places at once; one big head cannot.

The worked example of section 4: d = 4, n_h = 2, d_h = 2 (head 1 = dims 1-2, head 2 = dims 3-4).
q = (1, 0, 0, 2), k_1 = (2, 1, 0, 0), k_2 = (0, 0, 1, 1).
Per-head scores are the dot products of the two halves; the big head's score is their sum (M1.15).
Each set of scores goes through its own softmax. Every number below is computed here.
Writes math/figures/m01-two-heads.svg
"""

import numpy as np

from m01_svg import SVG, fmt

q = np.array([1.0, 0.0, 0.0, 2.0])
keys = [np.array([2.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0])]
halves = (slice(0, 2), slice(2, 4))

s1 = np.array([q[halves[0]] @ k[halves[0]] for k in keys])     # head 1 scores
s2 = np.array([q[halves[1]] @ k[halves[1]] for k in keys])     # head 2 scores
sb = np.array([q @ k for k in keys])                           # one big head
assert np.allclose(sb, s1 + s2)                                # (M1.15)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


w1, w2, wb = softmax(s1), softmax(s2), softmax(sb)
assert abs(w1[0] - 0.8808) < 1e-4 and abs(wb[0] - 0.5) < 1e-12

W, H = 720, 408
svg = SVG(W, H, "两个小头各自做 softmax，能分别盯住两个 token；一个大头把分数加在一起，只能对半平均")

# ---------------------------------------------------------------- top left: the vectors
svg.text(16, 24, "① 查询与两个键（d = 4，按头切成两段）", size=14, weight="600")
CELL, GAP, CH = 34, 12, 26
X0, Y0 = 70, 62


def cell_x(j):
    return X0 + j * CELL + (GAP if j >= 2 else 0)


# head brackets above the columns
for h, (a, b) in enumerate(((0, 1), (2, 3))):
    xa, xb = cell_x(a) + 2, cell_x(b) + CELL - 2
    svg.path("M%s %s v-6 h%s v6" % (fmt(xa), fmt(Y0 - 6), fmt(xb - xa)), stroke="muted", width=1.1)
    svg.text((xa + xb) / 2, Y0 - 16, "头 %d" % (h + 1), size=12, anchor="middle", color="muted")

rows = (("*q*", q, "blue"), ("*k*_{1}", keys[0], "violet"), ("*k*_{2}", keys[1], "violet"))
for r, (name, vec, col) in enumerate(rows):
    y = Y0 + r * (CH + 10)
    svg.text(X0 - 12, y + CH / 2 + 5, name, size=14, anchor="end", color=col, weight="600")
    for j, v in enumerate(vec):
        x = cell_x(j)
        svg.rect(x + 1, y, CELL - 2, CH, fill=col, fill_opacity=0.12, stroke=col, width=0.9, rx=3)
        svg.text(x + CELL / 2, y + CH / 2 + 5, "%d" % v, size=13, anchor="middle", color="ink")

# ---------------------------------------------------------------- top right: the score table
svg.line(300, 40, 300, 172, color="rule", width=1)
svg.text(318, 24, "② 分数：各头只点乘自己那一段", size=14, weight="600")
cols_x = (340, 440, 540, 648)
heads = ("键", "头 1 分数", "头 2 分数", "大头 = 两段之和")
for x, lab in zip(cols_x, heads):
    svg.text(x, 62, lab, size=12, anchor="middle", color="muted", weight="600")
svg.line(318, 72, W - 12, 72, color="rule", width=1)
for r in range(2):
    y = 100 + r * 34
    svg.text(cols_x[0], y, "*k*_{%d}" % (r + 1), size=14, anchor="middle", color="violet", weight="600")
    svg.text(cols_x[1], y, "%d" % s1[r], size=14, anchor="middle", color="blue")
    svg.text(cols_x[2], y, "%d" % s2[r], size=14, anchor="middle", color="blue")
    svg.text(cols_x[3], y, "%d + %d = %d" % (s1[r], s2[r], sb[r]), size=14, anchor="middle")
svg.text(318, 166, "头 1 只看前两维，头 2 只看后两维；大头看全部四维。", size=12, color="muted")

svg.line(16, 190, W - 16, 190, color="rule", width=1)

# ---------------------------------------------------------------- bottom: three softmaxes
svg.text(16, 216, "③ 各自做 softmax 得到的注意力权重", size=14, weight="600")
BASE, HMAX, BW = 360, 92, 36
charts = (
    ("头 1：softmax(%d, %d)" % tuple(s1), w1, "主要看 token 1"),
    ("头 2：softmax(%d, %d)" % tuple(s2), w2, "主要看 token 2"),
    ("一个大头：softmax(%d, %d)" % tuple(sb), wb, "平局：对半平均"),
)
for c, (title, w, note) in enumerate(charts):
    cx = 130 + c * 230                     # centre of this chart
    svg.text(cx, 240, title, size=13, anchor="middle", weight="600", color="violet" if c < 2 else "ink")
    # reference line at weight 0.5
    y05 = BASE - 0.5 * HMAX
    svg.line(cx - 70, y05, cx + 70, y05, color="muted", width=0.8, dash="3 3")
    if c == 0:
        svg.text(cx - 74, y05 + 4, "0.5", size=11, anchor="end", color="muted")
    svg.line(cx - 70, BASE, cx + 70, BASE, color="ink", width=1.2)
    for t in range(2):
        x = cx - 50 + t * 64
        h = w[t] * HMAX
        strong = w[t] > 0.55
        svg.rect(x, BASE - h, BW, h, fill="violet", fill_opacity=0.62 if strong else 0.28, stroke="violet", width=1)
        svg.text(x + BW / 2, BASE - h - 6, "%.3f" % w[t], size=12, anchor="middle")
        svg.text(x + BW / 2, BASE + 16, "token %d" % (t + 1), size=12, anchor="middle", color="muted")
    svg.text(cx, BASE + 38, note, size=13, anchor="middle", color="green" if c < 2 else "red")

svg.save("m01-two-heads.svg")
