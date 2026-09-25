"""Figure M1.2: one query, three keys. Dot product and cosine rank them differently.

q = (2, 1); k1 = (1, 1), k2 = (4, -1), k3 = (0, 2). Scores are computed here.
Writes math/figures/m01-dot-vs-cosine.svg
"""

import numpy as np

from m01_svg import SVG

q = np.array([2.0, 1.0])
keys = {"*k*_{1}": np.array([1.0, 1.0]), "*k*_{2}": np.array([4.0, -1.0]), "*k*_{3}": np.array([0.0, 2.0])}
names = list(keys)
dots = {n: float(q @ k) for n, k in keys.items()}
coss = {n: dots[n] / (np.linalg.norm(q) * np.linalg.norm(k)) for n, k in keys.items()}
rank_dot = {n: i + 1 for i, n in enumerate(sorted(names, key=lambda n: -dots[n]))}
rank_cos = {n: i + 1 for i, n in enumerate(sorted(names, key=lambda n: -coss[n]))}

W, H = 720, 322
S = 50
O = (78, 196)


def P(v):
    return O[0] + S * v[0], O[1] - S * v[1]


svg = SVG(W, H, "同一个查询 q 与三个键：点积和余弦相似度给出不同的排名")

# grid
for i in range(-1, 5):
    x, _ = P((i, 0))
    svg.line(x, P((0, 2.6))[1], x, P((0, -1.6))[1], color="grid", width=1)
for j in range(-1, 3):
    _, y = P((0, j))
    svg.line(P((-1.2, 0))[0], y, P((4.6, 0))[0], y, color="grid", width=1)
svg.line(P((-1.2, 0))[0], O[1], P((4.6, 0))[0], O[1], color="muted", width=1.1)
svg.line(O[0], P((0, 2.6))[1], O[0], P((0, -1.6))[1], color="muted", width=1.1)
for i in (1, 2, 3, 4):
    x, _ = P((i, 0))
    svg.text(x, O[1] + 16, str(i), size=11, anchor="middle", color="muted")

# unit circle: cosine only sees where each arrow crosses it
svg.circle(O[0], O[1], S, stroke="muted", width=1.1, dash="3 3")
for n, k in keys.items():
    u = k / np.linalg.norm(k)
    svg.circle(*P(u), 3.2, fill="violet")
uq = q / np.linalg.norm(q)
svg.circle(*P(uq), 3.2, fill="blue")

# arrows
for n, k in keys.items():
    svg.arrow(*P((0, 0)), *P(k), color="violet", width=2.1)
svg.arrow(*P((0, 0)), *P(q), color="blue", width=2.8)
svg.text(*(np.array(P(q)) + (8, -6)), "*q* = (2, 1)", size=14, color="blue")
svg.text(*(np.array(P(keys["*k*_{1}"])) + (4, -12)), "*k*_{1} = (1, 1)", size=13, color="violet", anchor="middle")
svg.text(*(np.array(P(keys["*k*_{2}"])) + (-4, 20)), "*k*_{2} = (4, −1)", size=13, color="violet", anchor="middle")
svg.text(*(np.array(P(keys["*k*_{3}"])) + (8, 4)), "*k*_{3} = (0, 2)", size=13, color="violet")
svg.text(P((-1.2, 0))[0], P((0, -1.6))[1] + 22, "虚线圆半径为 1；圆上的点是各向量的方向（余弦只看它）", size=12, color="muted")

# table
X = [360, 430, 520, 600, 668]
Y0 = 70
svg.rect(340, 34, 368, 194, fill="surface", stroke="rule", rx=10)
heads = ["键", "*q*·*k*", "|*k*|", "cos", ""]
for x, h in zip(X, heads):
    svg.text(x, Y0 - 12, h, size=13, color="muted", anchor="start" if x == X[0] else "middle", weight="600")
svg.text(X[4], Y0 - 12, "排名", size=13, color="muted", anchor="middle", weight="600")
svg.line(350, Y0 - 3, 698, Y0 - 3, color="rule", width=1)
for i, n in enumerate(names):
    y = Y0 + 22 + i * 36
    svg.text(X[0], y, n, size=14, color="violet")
    svg.text(X[1], y, "%g" % dots[n], size=14, anchor="middle",
             weight="700" if rank_dot[n] == 1 else None, color="ink")
    svg.text(X[2], y, "%.2f" % np.linalg.norm(keys[n]), size=14, anchor="middle", color="ink")
    svg.text(X[3], y, "%.3f" % coss[n], size=14, anchor="middle",
             weight="700" if rank_cos[n] == 1 else None, color="ink")
    svg.text(X[4], y, "%d / %d" % (rank_dot[n], rank_cos[n]), size=13, anchor="middle", color="ink")
svg.text(360, 202, "排名写成“按点积 / 按余弦”。", size=12, color="muted")
svg.text(360, 219, "最长的 *k*_{2} 点积最高，最同向的 *k*_{1} 余弦最高。", size=12, color="muted")

# softmax note under the table
e = np.exp(np.array([dots[n] for n in names]))
w = e / e.sum()
svg.text(340, 262, "若直接对点积做 softmax（第 1 课），三个权重约为", size=13, color="ink")
svg.text(340, 286, "  %s" % "，".join("%s %.3f" % (n, v) for n, v in zip(names, w)), size=13, color="ink")
svg.text(340, 308, "  *k*_{2} 几乎拿走全部注意力。", size=13, color="ink")

svg.save("m01-dot-vs-cosine.svg")
