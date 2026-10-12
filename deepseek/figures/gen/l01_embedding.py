"""Figure 1.4: the same matrix E used twice. (a) input: one-hot row times E selects a row;
(b) output: logits z = E h~ (one dot product per vocabulary entry), then softmax.

Toy vocabulary of 5 tokens, d = 3. All numbers are computed here; the lesson text quotes them.
Writes deepseek/figures/l01-embedding.svg
"""

import numpy as np

from l01_svg import SVG

words = ["the", "cat", "sat", "on", "mat"]
E = np.array([[0.2, -0.1, 0.4],
              [0.9, 0.3, -0.2],
              [-0.3, 0.8, 0.1],
              [0.1, 0.0, 0.6],
              [0.7, 0.4, -0.5]])
v = 1                                   # input token "cat"
onehot = np.eye(5)[v]
row = onehot @ E
assert np.allclose(row, E[v])
h = np.array([-0.4, 2.0, 0.6])          # a final hidden state (after ln_f)
z = E @ h
p = np.exp(z - z.max())
p /= p.sum()
print("row", row, "z", np.round(z, 3), "p", np.round(p, 3))


def f(x, nd=1):
    s_ = ("%%.%df" % nd) % x
    if s_.startswith("-"):
        s_ = "−" + s_[1:]
    return "0" if s_ in ("0.0", "−0.0", "0.00", "−0.00") and nd else s_


W, H = 760, 424
s = SVG(W, H, "同一个矩阵 E 用两次：输入查表与输出打分")
CW, CH = 34, 26
OC = 26          # one-hot cell width

# ---------------------------------------------------------------- (a) input lookup
s.text(10, 22, "(a) 输入：独热行向量 × E = 取出 E 的一行", size=13, weight="700")
EY = 48
yr = EY + v * CH                       # the row being selected
x0 = 20
for j, w in enumerate(words):
    s.text(x0 + j * OC + OC / 2, yr - 8, w, size=11, anchor="middle", color="blue" if j == v else "muted")
s.matrix(x0, yr, [[int(o) for o in onehot]], cell=OC, cell_h=CH, highlight={(0, v): "blue"}, size=12)
s.text(x0 + 2.5 * OC, yr + CH + 20, "独热向量  [1, 5]", size=12, anchor="middle", color="muted")
s.text(x0 + 5 * OC + 22, yr + 18, "×", size=16, anchor="middle")
ex = 232
for i, w in enumerate(words):
    s.text(ex - 9, EY + i * CH + 17, w, size=12, anchor="end", color="blue" if i == v else "muted")
s.matrix(ex, EY, [[f(x) for x in r] for r in E], cell=CW, cell_h=CH,
         highlight={(v, j): "blue" for j in range(3)}, size=12)
s.text(ex + 1.5 * CW, EY + 5 * CH + 18, "*E*  [5, 3]", size=12, anchor="middle", color="muted")
s.text(ex + 3 * CW + 20, yr + 18, "=", size=16, anchor="middle")
rx = ex + 3 * CW + 40
s.matrix(rx, yr, [[f(x) for x in row]], cell=CW, cell_h=CH, highlight={(0, j): "blue" for j in range(3)}, size=12)
s.text(rx + 1.5 * CW, yr + CH + 20, "cat 的词向量  [1, 3]", size=12, anchor="middle", color="muted")
for k, t in enumerate(["乘法里有 4 行被乘以 0，", "结果恰好是 E 的第 v 行。", "实现里直接按下标取行，",
                       "每个 token 省下 2Vd 次运算", "（GPT-2 small：2×50257×768", "≈ 7.7×10^{7}）。"]):
    s.text(520, EY + 12 + k * 20, t, size=12, color="muted")

# ---------------------------------------------------------------- (b) output scoring
s.line(10, 214, 750, 214, color="rule", width=1)
s.text(10, 238, "(b) 输出：logits *z* = *E* h̃（每个词一次点积），再做 softmax", size=13, weight="700")
BY = 264
bx = 60
top = int(np.argmax(z))
for i, w in enumerate(words):
    s.text(bx - 9, BY + i * CH + 17, w, size=12, anchor="end", color="green" if i == top else "muted")
s.matrix(bx, BY, [[f(x) for x in r] for r in E], cell=CW, cell_h=CH, size=12,
         highlight={(top, j): "green" for j in range(3)})
s.text(bx + 1.5 * CW, BY + 5 * CH + 18, "*E*  [5, 3]", size=12, anchor="middle", color="muted")
s.text(bx + 3 * CW + 18, BY + 2.5 * CH + 5, "×", size=16, anchor="middle")
hx = bx + 3 * CW + 36
s.matrix(hx, BY + CH, [[f(x)] for x in h], cell=CW, cell_h=CH, highlight={(i, 0): "blue" for i in range(3)}, size=12)
s.text(hx + CW / 2, BY + 4 * CH + 18, "h̃  [3]", size=12, anchor="middle", color="muted")
s.text(hx + CW + 18, BY + 2.5 * CH + 5, "=", size=16, anchor="middle")
zx = hx + CW + 36
s.matrix(zx, BY, [[f(x, 2)] for x in z], cell=46, cell_h=CH, highlight={(top, 0): "green"}, size=12)
s.text(zx + 23, BY + 5 * CH + 18, "*z*  [5]", size=12, anchor="middle", color="muted")
px_ = zx + 70
s.text(px_, BY - 6, "softmax →", size=12, color="green", weight="600")
for i in range(5):
    y = BY + i * CH
    wbar = p[i] * 120
    s.rect(px_, y + 6, wbar, CH - 12, fill="green", fill_opacity=0.6 if i == top else 0.22)
    s.text(px_ + wbar + 5, y + 17, f(p[i], 3), size=11, color="green" if i == top else "muted")
for k, t in enumerate(["每个 logit 是 h̃ 与一行词向量的点积：", "z_{sat} = −0.4×(−0.3) + 2.0×0.8 + 0.6×0.1",
                       "     = 1.78，五个里最大，", "所以 sat 的概率最高（0.557）。",
                       "h̃ 越长，logit 差距越大，分布越尖。"]):
    s.text(520, BY + 12 + k * 20, t, size=12, color="muted")

s.save("l01-embedding.svg")
