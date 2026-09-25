"""Figure 9.8: why RoPE blocks absorption -- the multiplication chain and a 2x2 numeric example.

Toy setting (lesson-09-doc sec. 4.6, constructed by the author and checked in Python):
d_h = 2, d_c = 1, W^UK = (3, 1)^T, query before rotation q = (1, 2)^T at position t = 3,
cached latents c_1 = 1, c_2 = 2, c_3 = -1, RoPE angle theta = 0.5 rad per position.
The effective absorbed query for history position j is  q^T R((j - t) theta) W^UK,
which is -1.5058, 1.9908, 5.0000 for j = 1, 2, 3: it changes with j, so no single q~ exists.
Writes deepseek/figures/l09-rope-breaks.svg
"""

import math

from l09_svg import SVG, fmt

q = (1.0, 2.0)
w = (3.0, 1.0)
t, theta = 3, 0.5
cs = {1: 1.0, 2: 2.0, 3: -1.0}


def rot(phi, v):
    return (math.cos(phi) * v[0] - math.sin(phi) * v[1], math.sin(phi) * v[0] + math.cos(phi) * v[1])


rows = []
for j in (1, 2, 3):
    phi = (j - t) * theta
    rv = rot(phi, w)
    qt = q[0] * rv[0] + q[1] * rv[1]
    rows.append((j, phi, rv, qt, qt * cs[j]))
assert abs(rows[0][3] + 1.5058) < 1e-3 and abs(rows[1][3] - 1.9908) < 1e-3 and abs(rows[2][3] - 5) < 1e-9
for r in rows:
    print(r)

W, H = 740, 480
svg = SVG(W, H, "RoPE 为什么挡住矩阵吸收：乘法链与一个二维数值例子")

# ------------------------------------------------ top: the chain
svg.text(16, 24, "给键施加 RoPE 之后，第 t 个查询对第 j 个历史 token 的分数：", size=13.5, weight="700")
x = 40
y = 64
svg.text(x, y, "*q*^{⊤} *R*_{t}^{⊤} *R*_{j} *W*^{UK} *c*_{j}", size=17, color="ink")
svg.text(x + 170, y, "=", size=17, color="ink")
svg.text(x + 196, y, "*q*^{⊤}", size=17, color="blue")
svg.rect(x + 226, y - 22, 116, 32, fill="red", fill_opacity=0.10, stroke="red", width=1.6, rx=5)
svg.text(x + 234, y, "*R*_{j−t} *W*^{UK}", size=17, color="red")
svg.text(x + 350, y, "*c*_{j}", size=17, color="orange")
svg.text(x + 226, y + 30, "↑ 夹在中间的矩阵随 j 变：每个历史位置一个，找不到对所有 j 通用的 *q̃*。", size=12.5, color="red")
svg.text(x, y + 62, "对照：没有 RoPE 时  *q*^{⊤}*W*^{UK}*c*_{j} = *q̃*^{⊤}*c*_{j}，其中 *q̃* = (*W*^{UK})^{⊤}*q* 与 j 无关，每步只算一次。",
         size=12.5, color="blue")
svg.line(12, 146, W - 12, 146, color="rule", width=1)

# ------------------------------------------------ bottom-left: the plane
S = 50.0
OX, OY = 70, 320
svg.text(16, 172, "二维例子（d_{h} = 2，θ = 0.5 rad/位置，t = 3）", size=13, weight="700")
for gx in range(-1, 5):
    svg.line(OX + gx * S, OY - 2.6 * S, OX + gx * S, OY + 2.6 * S, color="grid", width=1)
for gy in range(-2, 3):
    svg.line(OX - 1.1 * S, OY - gy * S, OX + 4.2 * S, OY - gy * S, color="grid", width=1)
svg.line(OX - 1.1 * S, OY, OX + 4.2 * S, OY, color="rule", width=1.3)
svg.line(OX, OY - 2.6 * S, OX, OY + 2.6 * S, color="rule", width=1.3)
for gx in (1, 2, 3, 4):
    svg.text(OX + gx * S, OY + 15, str(gx), size=11, anchor="middle", color="muted")
for gy in (-2, -1, 1, 2):
    svg.text(OX - 8, OY - gy * S + 4, str(gy), size=11, anchor="end", color="muted")


def P(v):
    return OX + v[0] * S, OY - v[1] * S


# rotation arc for |W^UK| from angle of w down to angle-1
rr = math.hypot(*w) * S
a0 = math.atan2(w[1], w[0])
arc = []
for k in range(0, 21):
    a = a0 - 1.0 * k / 20
    arc.append((OX + rr * math.cos(a), OY - rr * math.sin(a)))
svg.polyline(arc, stroke="red", width=1.2, dash="3 3")
ops = {1: 0.55, 2: 0.75, 3: 1.0}
for j, phi, rv, qt, sc in rows:
    x2, y2 = P(rv)
    svg.arrow(OX, OY, x2, y2, color="violet", width=2.0 if j == 3 else 1.6, shorten=1)
    lab = {1: "j = 1：转 −1 rad", 2: "j = 2：转 −0.5", 3: "j = 3：不转 = *W*^{UK}"}[j]
    dx, dy = {1: (8, 14), 2: (8, 6), 3: (8, -4)}[j]
    svg.text(x2 + dx, y2 + dy, lab, size=11.5, color="violet")
qx, qy = P(q)
svg.arrow(OX, OY, qx, qy, color="blue", width=2.4, shorten=1)
svg.text(qx - 6, qy - 8, "*q* = (1, 2)", size=12.5, color="blue", weight="700", anchor="end")

# ------------------------------------------------ bottom-right: the table
TX = 382
svg.text(TX, 172, "有效“吸收查询” *q̃*_{3,j} = *q*^{⊤}*R*((j−3)θ)*W*^{UK}", size=13, weight="700")
cols = [TX, TX + 30, TX + 104, TX + 230, TX + 300]
heads = ["j", "旋转角", "*R* *W*^{UK}", "*q̃*_{3,j}", "分数"]
for cx, hd in zip(cols, heads):
    svg.text(cx, 204, hd, size=12, color="muted", weight="600")
svg.line(TX, 212, W - 16, 212, color="rule", width=1)
for r, (j, phi, rv, qt, sc) in enumerate(rows):
    y = 234 + r * 26
    svg.text(cols[0], y, str(j), size=12.5)
    svg.text(cols[1], y, ("%s rad" % fmt(phi)).replace("-", "−"), size=12.5)
    svg.text(cols[2], y, ("(%.4f, %.4f)" % rv).replace("-", "−"), size=12.5, color="violet")
    svg.text(cols[3], y, ("%.4f" % qt).replace("-", "−"), size=12.5, color="red", weight="700")
    svg.text(cols[4], y, ("%.3f" % sc).replace("-", "−"), size=12.5, color="ink")
svg.line(TX, 302, W - 16, 302, color="rule", width=1)
svg.text(TX, 326, "三个 *q̃* 各不相同（红色）。要算分数，只能对每个 j", size=12.5)
svg.text(TX, 346, "重新造出 *R*_{j−t}*W*^{UK}*c*_{j}，也就是把前缀的键全部重算；", size=12.5)
svg.text(TX, 366, "或者把旋转后的键存进缓存，那又回到了 MHA 的缓存。", size=12.5)
svg.text(TX, 398, "没有 RoPE 时：*q̃* = 1·3 + 2·1 = 5，对所有 j 都一样，", size=12.5, color="blue")
svg.text(TX, 418, "分数 5、10、−5 可以直接用缓存里的 c_{j} 算。", size=12.5, color="blue")
svg.text(TX, 450, "c_{1} = 1，c_{2} = 2，c_{3} = −1（缓存的一维潜向量）", size=11.5, color="orange")
svg.save("l09-rope-breaks.svg")
