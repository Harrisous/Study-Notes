"""Figure 4.3 of the blackboard version of lesson 4: a three-step decode trace with real numbers.

One attention head, d_h = 2, tokens 从 前 有. The q/k/v values are the toy numbers used in the
lesson text (step 3 is the worked example of the original lesson). Every score, softmax weight and
output is computed here, not typed in. At step t the new row (blue frame) is computed, the older rows
(amber frame) are read from the cache, and the older queries are shown greyed out as discarded.
Writes deepseek/figures/l04b-board-trace.svg
"""

import math

from l04_svg import SVG

TOK = ["从", "前", "有"]
Q = [(1, 0), (0, 1), (2, 0)]
K = [(1, 0), (0, 1), (1, 1)]
V = [(1, 0), (0, 2), (2, 2)]
DH = 2


def vec(v, nd=0):
    if nd == 0:
        return "(%d, %d)" % v
    return "(" + ", ".join(("%." + str(nd) + "f") % x for x in v) + ")"


def step(t):
    q = Q[t - 1]
    raw = [q[0] * K[j][0] + q[1] * K[j][1] for j in range(t)]
    sc = [r / math.sqrt(DH) for r in raw]
    e = [math.exp(x) for x in sc]
    s = sum(e)
    a = [x / s for x in e]
    o = tuple(sum(a[j] * V[j][d] for j in range(t)) for d in range(2))
    return raw, sc, a, o


W = 760
ROW = 27
X_LAB, X_Q, X_K, X_V = 14, 110, 222, 324
BW = 88              # width of a K or V row box
X_RAW, X_SC, X_A, X_O = 480, 538, 600, 646

panels = []
y = 58
for t in (1, 2, 3):
    h = 34 + t * ROW + 10
    panels.append((t, y, h))
    y += h + 10
H = y + 34

svg = SVG(W, H, "黑板上的三步解码：每一步新算哪一行、从缓存读哪几行")

# column headers
hy = 30
svg.text(X_Q, hy, "当前查询", size=12, color="blue", weight="600")
svg.text(X_K + BW / 2, hy, "键 K", size=12, anchor="middle", color="violet", weight="600")
svg.text(X_V + BW / 2, hy, "值 V", size=12, anchor="middle", color="green", weight="600")
svg.text(X_RAW, hy, "q·k", size=12, anchor="middle", weight="600")
svg.text(X_SC, hy, "÷√2", size=12, anchor="middle", weight="600")
svg.text(X_A, hy, "权重 a", size=12, anchor="middle", weight="600")
svg.text(X_O, hy, "输出（Σ a·v）", size=12, color="green", weight="600")
svg.line(10, hy + 10, W - 10, hy + 10, color="rule", width=1)

for t, oy, h in panels:
    raw, sc, a, o = step(t)
    svg.rect(8, oy, W - 16, h, fill=None, stroke="rule", width=1, rx=8)
    svg.text(X_LAB + 4, oy + 24, "第 %d 步" % t, size=13, weight="700")
    svg.text(X_LAB + 4, oy + 44, "新 token「%s」" % TOK[t - 1], size=12, color="muted")
    # current query
    qy = oy + 34 + (t - 1) * ROW
    svg.rect(X_Q - 4, qy - 15, 94, 22, fill="blue", fill_opacity=0.12, stroke="blue", width=1.6, rx=4)
    svg.text(X_Q + 2, qy + 1, "*q*_{%d} = %s" % (t, vec(Q[t - 1])), size=13, color="blue")
    # discarded older queries
    if t > 1:
        old = "、".join("*q*_{%d}" % j for j in range(1, t))
        svg.text(X_Q, oy + 24, old + " 已扔", size=12, color="muted")
    for j in range(1, t + 1):
        ry = oy + 34 + (j - 1) * ROW
        new = j == t
        st = "blue" if new else "amber"
        sw = 1.8 if new else 1.1
        svg.rect(X_K, ry - 15, BW, 22, fill="violet", fill_opacity=0.14 if new else 0.06, stroke=st, width=sw, rx=4)
        svg.text(X_K + 6, ry + 1, "*k*_{%d} %s" % (j, vec(K[j - 1])), size=13)
        svg.rect(X_V, ry - 15, BW, 22, fill="green", fill_opacity=0.14 if new else 0.06, stroke=st, width=sw, rx=4)
        svg.text(X_V + 6, ry + 1, "*v*_{%d} %s" % (j, vec(V[j - 1])), size=13)
        svg.text(X_RAW, ry + 1, "%d" % raw[j - 1], size=13, anchor="middle")
        svg.text(X_SC, ry + 1, "%.3f" % sc[j - 1], size=13, anchor="middle")
        svg.text(X_A, ry + 1, "%.3f" % a[j - 1] if t > 1 else "1", size=13, anchor="middle", color="blue")
        tag = "新算" if new else "读缓存"
        svg.text(X_V + BW + 8, ry + 1, tag, size=11, color=st)
    # output
    oy2 = oy + 34 + (t - 1) * ROW
    if t > 1:
        svg.text(X_O, oy2 - 22, "*o*_{%d} =" % t, size=13, color="green", weight="600")
        svg.text(X_O, oy2 + 1, vec(o, 3), size=13, color="green", weight="600")
    else:
        svg.text(X_O, oy2 + 1, "*o*_{1} = %s" % vec((round(o[0]), round(o[1]))), size=13, color="green",
                 weight="600")

# legend
ly = H - 14
svg.rect(16, ly - 12, 30, 16, fill="violet", fill_opacity=0.14, stroke="blue", width=1.8, rx=3)
svg.text(54, ly, "本步新算", size=12)
svg.rect(136, ly - 12, 30, 16, fill="violet", fill_opacity=0.06, stroke="amber", width=1.1, rx=3)
svg.text(174, ly, "从缓存读出（上一步写进去的）", size=12)
svg.text(420, ly, "不缓存时 K 要算 1 + 2 + 3 = 6 行", size=12, color="muted")
svg.save("l04b-board-trace.svg")
