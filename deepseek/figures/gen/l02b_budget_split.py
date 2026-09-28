"""Figure 2.14 of the blackboard version of lesson 2: splitting a fixed budget between N and D.

Toy version of the Chinchilla parametric loss (2.26) with E = 0, A = B = 1, alpha = beta = 1:
    L(N, D) = 1/N + 1/D   subject to   N * D = budget   (budget plays the role of C / 6).
Each bar is one split, stacked as the "model-limited" term 1/N (blue, bottom) and the "data-limited"
term 1/D (violet, top). Panel (a) budget 100: optimum N = D = 10, L = 0.20. Panel (b) budget 400:
optimum N = D = 20, L = 0.10 -- four times the budget, both factors doubled.
The closed form (2.27) gives N_opt = G (C/6)^a with a = beta/(alpha+beta) = 1/2 and G = 1.
All bar heights are computed here.
Writes deepseek/figures/l02b-budget-split.svg
"""

import math

from l02_svg import SVG

ALPHA = BETA = 1.0
A = B = 1.0
A_EXP = BETA / (ALPHA + BETA)
G = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))

PANELS = [
    ("(a) 预算 ND = 100", 100, [2, 4, 5, 10, 20, 25, 50]),
    ("(b) 预算翻 4 倍：ND = 400", 400, [5, 8, 10, 20, 40, 50, 80]),
]

W, H = 720, 340
svg = SVG(W, H, "黑板上的预算分配：两项损失此消彼长")
YMAX = 0.55
PY0, PY1 = 58, 250


def fmt(v):
    s = "%.3f" % v
    return s.rstrip("0").rstrip(".")


for pi, (title, budget, Ns) in enumerate(PANELS):
    X0 = 58 + pi * 345
    X1 = X0 + 300
    n_opt = G * budget ** A_EXP
    assert abs(n_opt - round(n_opt)) < 1e-9 and round(n_opt) in Ns
    svg.text(X0 - 30, 26, title, size=13, weight="600")
    # y grid
    for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        y = PY1 - v / YMAX * (PY1 - PY0)
        svg.line(X0, y, X1, y, color="grid", width=1)
        svg.text(X0 - 7, y + 4, fmt(v), size=11, anchor="end", color="muted")
    svg.line(X0, PY1, X1, PY1, color="muted", width=1.2)
    slot = (X1 - X0) / len(Ns)
    bw = slot * 0.62
    for i, n in enumerate(Ns):
        d = budget / n
        tn, td = A / n ** ALPHA, B / d ** BETA
        tot = tn + td
        cx = X0 + slot * (i + 0.5)
        yb = PY1
        hn = tn / YMAX * (PY1 - PY0)
        hd = td / YMAX * (PY1 - PY0)
        best = abs(n - n_opt) < 1e-9
        svg.rect(cx - bw / 2, yb - hn, bw, hn, fill="blue", fill_opacity=0.55 if best else 0.30,
                 stroke="blue", width=1.8 if best else 1.0)
        svg.rect(cx - bw / 2, yb - hn - hd, bw, hd, fill="violet", fill_opacity=0.55 if best else 0.30,
                 stroke="violet", width=1.8 if best else 1.0)
        svg.text(cx, yb - hn - hd - 6, fmt(tot), size=11 if not best else 12, anchor="middle",
                 color="ink", weight="700" if best else None)
        svg.text(cx, PY1 + 16, "N=%d" % n, size=11, anchor="middle", color="blue" if best else "muted",
                 weight="700" if best else None)
        svg.text(cx, PY1 + 31, "D=%s" % fmt(d), size=11, anchor="middle", color="violet" if best else "muted",
                 weight="700" if best else None)
        if best:
            svg.text(cx, PY1 + 50, "谷底：两段一样长", size=12, anchor="middle", color="ink", weight="600")

# legend
ly = H - 14
svg.rect(60, ly - 12, 26, 14, fill="blue", fill_opacity=0.45, stroke="blue", width=1)
svg.text(94, ly, "模型有限 1/N", size=12)
svg.rect(210, ly - 12, 26, 14, fill="violet", fill_opacity=0.45, stroke="violet", width=1)
svg.text(244, ly, "数据有限 1/D", size=12)
svg.text(372, ly, "柱顶数字 = 玩具损失 1/N + 1/D", size=12, color="muted")
svg.save("l02b-budget-split.svg")
