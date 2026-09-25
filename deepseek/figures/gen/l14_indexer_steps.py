"""Figure 14.8: the lightning indexer on a hand-sized example (H^I = 2 heads, d^I = 2, 4 keys).

q^I_1 = (1, 0), q^I_2 = (0, 1), w = (0.5, 1.0); keys k_1 = (2, -1), k_2 = (-1, 3), k_3 = (1, 1),
k_4 = (-2, -2). I_{t,s} = sum_j w_j * ReLU(q_j . k_s) = (1.0, 3.0, 1.5, 0); top-2 = {2, 3}.
Every number is computed below.
Writes deepseek/figures/l14-indexer-steps.svg
"""

from l14_svg import SVG

Q = [(1, 0), (0, 1)]
WTS = [0.5, 1.0]
K = [(2, -1), (-1, 3), (1, 1), (-2, -2)]
DOTS = [[q[0] * k[0] + q[1] * k[1] for k in K] for q in Q]
RELU = [[max(0, v) for v in row] for row in DOTS]
WEIGHTED = [[w * v for v in row] for w, row in zip(WTS, RELU)]
I = [sum(col) for col in zip(*WEIGHTED)]
assert I == [1.0, 3.0, 1.5, 0.0]
TOP2 = sorted(range(4), key=lambda s: -I[s])[:2]
assert sorted(TOP2) == [1, 2]


def num(v):
    return ("%g" % v).replace("-", "−")


W, H = 740, 360
svg = SVG(W, H, "lightning indexer 的一次打分：两个头、四个键（手算例）")

CX0, CW = 210, 74          # value columns
ROWH = 30
Y0 = 58

svg.text(20, 26, "查询 token t 的 indexer：*q*_{1}^{I} = (1, 0)，*q*_{2}^{I} = (0, 1)；头权重 w = (0.5, 1.0)", size=13,
         weight="600")
for s in range(4):
    x = CX0 + s * CW + CW / 2
    svg.text(x, Y0, "s = %d" % (s + 1), size=12, anchor="middle", weight="700")

rows = [
    ("键 *k*_{s}^{I}", ["(%s, %s)" % (num(k[0]), num(k[1])) for k in K], "ink", None),
    ("头 1：*q*_{1}^{I} · *k*_{s}^{I}", DOTS[0], "ink", "dot"),
    ("ReLU", RELU[0], "ink", "relu0"),
    ("× w_{1} = 0.5", WEIGHTED[0], "violet", None),
    ("头 2：*q*_{2}^{I} · *k*_{s}^{I}", DOTS[1], "ink", "dot"),
    ("ReLU", RELU[1], "ink", "relu1"),
    ("× w_{2} = 1.0", WEIGHTED[1], "violet", None),
]
y = Y0 + 12
for r, (label, vals, col, kind) in enumerate(rows):
    y = Y0 + 12 + r * ROWH
    if r in (1, 4):
        svg.line(20, y, CX0 + 4 * CW, y, color="rule", width=1)
    svg.text(24, y + 20, label, size=12, color="muted" if kind in ("relu0", "relu1") else "ink")
    for s, v in enumerate(vals):
        x = CX0 + s * CW + CW / 2
        if isinstance(v, str):
            svg.text(x, y + 20, v, size=12, anchor="middle")
            continue
        c = col
        if kind == "dot" and v < 0:
            c = "red"
        if kind in ("relu0", "relu1"):
            row = 0 if kind == "relu0" else 1
            if DOTS[row][s] < 0:
                c = "red"
                svg.rect(x - 14, y + 5, 28, 21, fill="red", fill_opacity=0.08, rx=4)
        svg.text(x, y + 20, num(v), size=13, anchor="middle", color=c)

# sum row
ys = Y0 + 12 + len(rows) * ROWH + 6
svg.line(20, ys, CX0 + 4 * CW, ys, color="ink", width=1.4)
svg.text(24, ys + 24, "I_{t,s} = 两行相加", size=13, weight="700", color="violet")
for s in range(4):
    x = CX0 + s * CW + CW / 2
    top = s in TOP2
    if top:
        svg.rect(x - 28, ys + 6, 56, 28, fill="violet", fill_opacity=0.18, stroke="violet", width=1.4, rx=5)
    svg.text(x, ys + 25, num(I[s]), size=14, anchor="middle", weight="700", color="violet" if top else "ink")
svg.text(24, ys + 58, "top-2（k = 2）", size=13, weight="700")
for s in range(4):
    x = CX0 + s * CW + CW / 2
    svg.text(x, ys + 58, "选中" if s in TOP2 else "—", size=13, anchor="middle",
             color="violet" if s in TOP2 else "muted", weight="700" if s in TOP2 else None)

# side notes
NX = CX0 + 4 * CW + 22
notes = [
    ("ReLU：负的点积一律记 0", "ink"),
    ("（红底的 0 就是被截断的）。", "ink"),
    ("s = 1 在头 1 上得 2 分，", "muted"),
    ("在头 2 上是 −1，被截成 0；", "muted"),
    ("“无关”和“强烈相反”", "muted"),
    ("在这个头上分不出来。", "muted"),
    ("", "muted"),
    ("w 是由 *h*_{t} 算出的实数，", "ink"),
    ("可正可负：一个头可以投", "muted"),
    ("“反对票”（本例两者都为正）。", "muted"),
    ("", "muted"),
    ("没有 softmax：I 只用来排序，", "ink"),
    ("绝对大小不影响选谁。", "muted"),
]
for i, (t, c) in enumerate(notes):
    if t:
        svg.text(NX, Y0 + 12 + i * 22, t, size=12, color=c)

svg.save("l14-indexer-steps.svg")
