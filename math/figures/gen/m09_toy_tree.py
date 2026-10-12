"""Figure M9.2: the two-token toy MDP as a tree, with every V, Q and A computed here.

Prompt "2+3="; first token 5 (right) or 6 (wrong); second token ！ (assertive) or 吧 (hedged).
Returns at the end: 5！=1, 5吧=0.5, 6！=0, 6吧=0.2. gamma = 1. Numbers are the author's
construction, used throughout M09 (see math/reference-docs/m09-notes.md §2.1).
Writes math/figures/m09-toy-tree.svg
"""

from m09_svg import SVG

pi = {"q": {"5": 0.6, "6": 0.4}, "q5": {"！": 0.7, "吧": 0.3}, "q6": {"！": 0.5, "吧": 0.5}}
R = {("5", "！"): 1.0, ("5", "吧"): 0.5, ("6", "！"): 0.0, ("6", "吧"): 0.2}
V = {s: sum(pi[s][a] * R[(s[1], a)] for a in pi[s]) for s in ("q5", "q6")}
V["q"] = sum(pi["q"][a] * V["q" + a] for a in pi["q"])
Q = {("q", a): V["q" + a] for a in pi["q"]}
Q.update({("q" + a1, a2): R[(a1, a2)] for (a1, a2) in R})
A = {k: Q[k] - V[k[0]] for k in Q}
assert abs(V["q"] - 0.55) < 1e-12

W, H = 720, 436
svg = SVG(W, H, "玩具模型的全部状态、价值与优势")

NW, NH, LW = 124, 56, 132
root = (12, 164)
mids = {"q5": (258, 78), "q6": (258, 250)}
leaves = {("5", "！"): (576, 20), ("5", "吧"): (576, 110), ("6", "！"): (576, 196), ("6", "吧"): (576, 286)}


def node(x, y, text, line2, color="ink", fill=None, w=None):
    w = w or NW
    svg.rect(x, y, w, NH, fill=fill or "blue", fill_opacity=0.07, stroke=fill or "blue", width=1.3, rx=8)
    svg.text(x + w / 2, y + 23, text, size=15, anchor="middle", weight="600")
    svg.text(x + w / 2, y + 44, line2, size=12.5, anchor="middle", color=color)


def fmt_signed(v):
    return ("+%.2f" % v) if v > 0 else ("−%.2f" % abs(v)) if v < 0 else "0.00"


node(root[0], root[1], "“2+3=”", "V = %.2f（= J）" % V["q"])
node(*mids["q5"], "“2+3=5”", "V = %.2f" % V["q5"])
node(*mids["q6"], "“2+3=6”", "V = %.2f" % V["q6"])
for (a1, a2), (x, y) in leaves.items():
    p = pi["q"][a1] * pi["q" + a1][a2]
    node(x, y, "“2+3=%s%s”" % (a1, a2), "R = %.1f，概率 %.2f" % (R[(a1, a2)], p), fill="green", w=LW)


def edge(x1, y1, x2, y2, tok, prob, adv, label_dx=0):
    svg.arrow(x1, y1, x2, y2, color="muted", width=1.4, shorten=2)
    mx, my = (x1 + x2) / 2 + label_dx, (y1 + y2) / 2
    col = "green" if adv > 0 else "red"
    svg.rect(mx - 46, my - 21, 92, 38, fill="surface", fill_opacity=1.0, stroke="rule", width=1, rx=5)
    svg.text(mx, my - 5, "“%s” π = %.1f" % (tok, prob), size=12.5, anchor="middle")
    svg.text(mx, my + 12, "A = %s" % fmt_signed(adv), size=12.5, anchor="middle", color=col, weight="600")


rx, ry = root[0] + NW, root[1] + NH / 2
for a in ("5", "6"):
    mx, my = mids["q" + a]
    edge(rx, ry, mx, my + NH / 2, a, pi["q"][a], A[("q", a)])
for (a1, a2), (x, y) in leaves.items():
    mx, my = mids["q" + a1]
    edge(mx + NW, my + NH / 2, x, y + NH / 2, a2, pi["q" + a1][a2], A[("q" + a1, a2)])

# worked arithmetic under the tree
y0 = 374
svg.line(16, y0 - 8, 704, y0 - 8, color="rule", width=1)
svg.text(16, y0 + 10, "V(“2+3=5”) = 0.7×1 + 0.3×0.5 = %.2f；V(“2+3=6”) = 0.5×0 + 0.5×0.2 = %.2f；"
         "V(“2+3=”) = 0.6×%.2f + 0.4×%.2f = %.2f" % (V["q5"], V["q6"], V["q5"], V["q6"], V["q"]), size=12, color="muted")
svg.text(16, y0 + 30, "Q(s, a) = 走这一步之后的价值；A(s, a) = Q(s, a) − V(s)。例：A(“2+3=”, “5”) = %.2f − %.2f = %s。"
         % (Q[("q", "5")], V["q"], fmt_signed(A[("q", "5")])), size=12, color="muted")
svg.text(16, y0 + 50, "每个状态上 Σ π(a|s)·A(s, a) = 0：0.6×0.30 + 0.4×(−0.45) = 0，0.7×0.15 + 0.3×(−0.35) = 0。",
         size=12, color="muted")
svg.save("m09-toy-tree.svg")
