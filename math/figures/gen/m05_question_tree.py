"""Figure M5.3: guessing a symbol with yes/no questions.

True distribution p = (A 1/2, B 1/4, C 1/8, D 1/8). Three questioning strategies, each the optimal one for some
belief q: q = p (lengths 1,2,3,3), q = uniform (2,2,2,2), q = reversed (1/8,1/8,1/4,1/2) (lengths 3,3,2,1).
Average number of questions under the TRUE p: 1.75 = H(p), 2 = H(p,u), 2.625 = H(p,q_rev).
The averages are computed here from the lengths. Writes math/figures/m05-question-tree.svg
"""

from fractions import Fraction as F

from m05_svg import SVG

P = {"A": F(1, 2), "B": F(1, 4), "C": F(1, 8), "D": F(1, 8)}
COL = {"A": "blue", "B": "violet", "C": "green", "D": "amber"}

W, H = 760, 430
svg = SVG(W, H, "用是/否问题猜一个符号：按哪个分布设计问题，平均就要问多少次")


def qnode(x, y, s, w=86):
    svg.rect(x - w / 2, y - 14, w, 28, fill="surface", stroke="ink", width=1.2, rx=6)
    svg.text(x, y + 5, s, size=12, anchor="middle")


def leaf(x, y, sym, depth):
    svg.circle(x, y, 15, fill=COL[sym], fill_opacity=0.16, stroke=COL[sym], width=1.5)
    svg.text(x, y + 5, sym, size=14, anchor="middle", color=COL[sym], weight="700")
    svg.text(x, y + 33, "%d 问" % depth, size=12, anchor="middle", color="muted")


def edge(x1, y1, x2, y2, lab, left):
    svg.line(x1, y1 + 14, x2, y2 - 15, color="muted", width=1.2)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    svg.text(mx + (-9 if left else 9), my + 2, lab, size=11, anchor="end" if left else "start", color="muted")


def chain(cx, order, qs):
    """Caterpillar tree: ask about order[0], then order[1], then split order[2]/order[3]."""
    y0, dy = 92, 62
    x = [cx - 62, cx - 20, cx + 22]
    qnode(x[0], y0, qs[0])
    leaf(x[0] - 44, y0 + dy, order[0], 1)
    edge(x[0], y0, x[0] - 44, y0 + dy, "是", True)
    qnode(x[1] + 22, y0 + dy, qs[1])
    edge(x[0], y0, x[1] + 22, y0 + dy, "否", False)
    leaf(x[1] - 22, y0 + 2 * dy, order[1], 2)
    edge(x[1] + 22, y0 + dy, x[1] - 22, y0 + 2 * dy, "是", True)
    qnode(x[2] + 44, y0 + 2 * dy, qs[2])
    edge(x[1] + 22, y0 + dy, x[2] + 44, y0 + 2 * dy, "否", False)
    leaf(x[2] + 4, y0 + 3 * dy, order[2], 3)
    edge(x[2] + 44, y0 + 2 * dy, x[2] + 4, y0 + 3 * dy, "是", True)
    leaf(x[2] + 84, y0 + 3 * dy, order[3], 3)
    edge(x[2] + 44, y0 + 2 * dy, x[2] + 84, y0 + 3 * dy, "否", False)
    return {order[0]: 1, order[1]: 2, order[2]: 3, order[3]: 3}


def balanced(cx):
    y0, dy = 92, 62
    qnode(cx, y0, "是 A 或 B？", w=96)
    qnode(cx - 58, y0 + dy + 30, "是 A？", w=62)
    qnode(cx + 58, y0 + dy + 30, "是 C？", w=62)
    edge(cx, y0, cx - 58, y0 + dy + 30, "是", True)
    edge(cx, y0, cx + 58, y0 + dy + 30, "否", False)
    for (qx, a, b) in ((cx - 58, "A", "B"), (cx + 58, "C", "D")):
        leaf(qx - 26, y0 + 3 * dy, a, 2)
        leaf(qx + 26, y0 + 3 * dy, b, 2)
        edge(qx, y0 + dy + 30, qx - 26, y0 + 3 * dy, "", True)
        edge(qx, y0 + dy + 30, qx + 26, y0 + 3 * dy, "", False)
    return {"A": 2, "B": 2, "C": 2, "D": 2}


panels = [
    (130, "(a) 按真实分布 p 设计", "blue"),
    (380, "(b) 按均匀分布设计", "violet"),
    (630, "(c) 按“反了”的分布设计", "red"),
]
lens = []
for cx, title, col in panels:
    svg.rect(cx - 122, 18, 244, 400, fill=col, fill_opacity=0.04, stroke="rule", width=1, rx=8)
    svg.text(cx, 42, title, size=13, anchor="middle", weight="600")
lens.append(chain(130, "ABCD", ["是 A？", "是 B？", "是 C？"]))
svg.text(130, 62, "q = (1/2, 1/4, 1/8, 1/8)", size=12, anchor="middle", color="muted")
lens.append(balanced(380))
svg.text(380, 62, "q = (1/4, 1/4, 1/4, 1/4)", size=12, anchor="middle", color="muted")
lens.append(chain(630, "DCAB", ["是 D？", "是 C？", "是 A？"]))
svg.text(630, 62, "q = (1/8, 1/8, 1/4, 1/2)", size=12, anchor="middle", color="muted")

names = ["= H(p)", "= H(p, q)", "= H(p, q)"]
extra = ["多问 0", "多问 0.25 = KL(p‖q)", "多问 0.875 = KL(p‖q)"]
for (cx, _, col), L, nm, ex in zip(panels, lens, names, extra):
    avg = sum(P[s] * L[s] for s in "ABCD")
    terms = " + ".join("%s·%d" % ({F(1, 2): "½", F(1, 4): "¼", F(1, 8): "⅛"}[P[s]], L[s]) for s in "ABCD")
    svg.text(cx, 354, terms, size=12, anchor="middle", color="muted")
    svg.text(cx, 376, "平均 %s 问 %s" % (("%.3f" % float(avg)).rstrip("0").rstrip("."), nm), size=13,
             anchor="middle", color=col, weight="600")
    svg.text(cx, 398, ex, size=12, anchor="middle", color=col)

svg.save("m05-question-tree.svg")
