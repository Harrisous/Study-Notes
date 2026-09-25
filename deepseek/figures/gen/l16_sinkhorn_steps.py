"""Figure 16.8: Sinkhorn-Knopp step by step on a 3x3 example, and how fast it converges.

Worked example (research notes N23, the notes author's choice): A = [[4,1,1],[1,4,1],[2,2,2]].
mHC eq. (9): M^(0) = A (= exp of the logits), M^(t) = T_r(T_c(M^(t-1))) -- column-normalise, then row-normalise.
Exact fractions via fractions.Fraction; the limit via 200 float iterations.
Bottom panel: max |column sum - 1| after t iterations (rows are exact after every T_r), log axis, for
  (i) the 3x3 example (linear convergence, ratio 0.0436 per step),
  (ii) [[1,1],[0,1]] (no full support: error exactly 1/(2t+1), notes N25),
  (iii) [[1,1],[e^-10,1]] (positive but nearly degenerate: error 0.0232 at t = 20, notes N25).
Writes deepseek/figures/l16-sinkhorn-steps.svg and prints the numbers.
"""

import math
from fractions import Fraction as Fr

import numpy as np

from l16_svg import SVG

A = [[Fr(4), Fr(1), Fr(1)], [Fr(1), Fr(4), Fr(1)], [Fr(2), Fr(2), Fr(2)]]


def Tc(M):
    cs = [sum(M[i][j] for i in range(len(M))) for j in range(len(M[0]))]
    return [[M[i][j] / cs[j] for j in range(len(M[0]))] for i in range(len(M))]


def Tr(M):
    return [[x / sum(r) for x in r] for r in M]


def rows(M):
    return [sum(r) for r in M]


def cols(M):
    return [sum(M[i][j] for i in range(len(M))) for j in range(len(M[0]))]


C1 = Tc(A)
M1 = Tr(C1)
L = np.array(A, float)
for _ in range(200):
    L = L / L.sum(0, keepdims=True)
    L = L / L.sum(1, keepdims=True)


def errs(M0, T=20):
    M = np.array(M0, float)
    out = []
    for _ in range(T):
        M = M / M.sum(0, keepdims=True)
        M = M / M.sum(1, keepdims=True)
        out.append(abs(M.sum(0) - 1).max())
    return out


E3 = errs([[4, 1, 1], [1, 4, 1], [2, 2, 2]])
E2 = [1 / (2 * t + 1) for t in range(1, 21)]
E2num = errs([[1, 1], [0, 1]])
assert all(abs(a - b) < 1e-12 for a, b in zip(E2, E2num))
Ek = errs([[1, 1], [math.exp(-10), 1]])
print("3x3 errors:", ["%.3g" % e for e in E3[:10]])
print("ratio:", E3[2] / E3[1])
print("2x2 exact 1/(2t+1) at t=20:", E2[-1], " near-degenerate e^-10 at t=20: %.4f" % Ek[-1])
print("limit:\n", np.round(L, 5))

W, H = 760, 716
svg = SVG(W, H, "Sinkhorn–Knopp 的一个 3×3 算例，以及它收敛得有多快")


def fr(x):
    return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)


def block(x, y, M, title, sub, row_ok, col_ok, fmt=fr, cw=52, ch=30, rs=None, cs=None):
    svg.text(x, y, title, size=14, weight="700")
    svg.text(x, y + 22, sub, size=12, color="muted")   # 22: clear of subscripts in the title
    y0 = y + 44
    for i in range(3):
        for j in range(3):
            svg.rect(x + j * cw + 1, y0 + i * ch + 1, cw - 2, ch - 2, fill="blue", fill_opacity=0.07, rx=4)
            svg.text(x + j * cw + cw / 2, y0 + i * ch + ch / 2 + 5, fmt(M[i][j]), size=13, anchor="middle")
    rs = rs or [fmt(v) for v in rows(M)]
    cs = cs or [fmt(v) for v in cols(M)]
    rc = "green" if row_ok else "red"
    cc = "green" if col_ok else "red"
    svg.text(x + 3 * cw + 8, y0 - 6, "行和", size=11, color=rc)
    for i in range(3):
        svg.text(x + 3 * cw + 8, y0 + i * ch + ch / 2 + 5, rs[i], size=12, color=rc, weight="600")
    svg.text(x - 4, y0 + 3 * ch + 18, "列和", size=11, anchor="end", color=cc)
    for j in range(3):
        svg.text(x + j * cw + cw / 2, y0 + 3 * ch + 18, cs[j], size=12, anchor="middle", color=cc, weight="600")


block(56, 30, A, "① M⁽⁰⁾ = A", "（mHC 里是 exp(logits)）", False, False)
svg.arrow(290, 120, 368, 120, color="ink", width=1.8)
svg.text(329, 108, "𝓣_{c}", size=13, anchor="middle")
svg.text(329, 140, "每列除以列和", size=11, anchor="middle", color="muted")
block(430, 30, C1, "② 𝓣_{c}(A)", "列和都变成 1，行和被打乱", False, True)

block(56, 250, M1, "③ M⁽¹⁾ = 𝓣_{r}(𝓣_{c}(A))", "行和变成 1，列和又偏了", True, False,
      cs=["136/135", "136/135", "133/135"])
svg.path("M600 202 V224 H26 V309 H50", stroke="ink", width=1.6, arrow=True)
svg.text(420, 214, "𝓣_{r}：每行除以行和", size=12, anchor="middle")
svg.arrow(290, 340, 368, 340, color="ink", width=1.8)
svg.text(329, 328, "重复 t 次", size=12, anchor="middle")
svg.text(329, 360, "t → ∞", size=11, anchor="middle", color="muted")
block(430, 250, L, "④ 极限（五位小数）", "行和、列和都为 1：双随机", True, True, fmt=lambda v: "%.5f" % v, cw=66,
      rs=["1", "1", "1"], cs=["1", "1", "1"])
svg.text(56, 440, "③ 的列和 ≈ 1.0074, 1.0074, 0.9852：误差 0.0148；每多一轮，误差约乘 0.0436", size=12,
         color="muted")

# ------------------------------------------------------------------ convergence plot
X0, X1, Y0, Y1 = 90, 540, 484, 670
E_LO, E_HI = -15, 0


def px(t):
    return X0 + (t - 1) / 19 * (X1 - X0)


def py(v):
    return Y1 - (math.log10(v) - E_LO) / (E_HI - E_LO) * (Y1 - Y0)


svg.text(40, 470, "⑤ 最大列和误差 |列和 − 1| 随迭代轮数 t 的变化（对数纵轴）", size=14, weight="700")
for e in range(E_LO, E_HI + 1, 3):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 7, y + 4, "10^{%d}" % e if e else "1", size=11, anchor="end", color="muted")
for t in (1, 5, 10, 15, 20):
    svg.line(px(t), Y0, px(t), Y1, color="grid", width=1)
    svg.text(px(t), Y1 + 16, str(t), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 34, "迭代轮数 t（mHC 取 t_{max} = 20）", size=12, anchor="middle", color="muted")


def series(vals, color, width, dash=None):
    pts = [(px(i + 1), py(max(v, 10 ** E_LO))) for i, v in enumerate(vals) if v >= 10 ** E_LO]
    svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke=color, width=width, dash=dash)
    for p in pts:
        svg.circle(p[0], p[1], 2.2, fill=color)
    return pts


p3 = series(E3, "green", 2.2)
p2 = series(E2, "red", 2.2)
pk = series(Ek, "orange", 2.0, dash="5 3")
LX = X1 + 14
svg.text(LX, Y0 + 10, "红：[[1,1],[0,1]]", size=12, color="red", weight="600")
svg.text(LX, Y0 + 27, "误差恰为 1/(2t+1)", size=12, color="red")
svg.text(LX, Y0 + 44, "t = 20 时 1/41 = 0.024", size=12, color="red")
svg.text(LX, Y0 + 72, "橙虚：[[1,1],[e^{-10},1]]", size=12, color="orange", weight="600")
svg.text(LX, Y0 + 89, "t = 20 时 %.4f" % Ek[-1], size=12, color="orange")
svg.text(LX, Y0 + 106, "与红线几乎重合", size=12, color="orange")
svg.text(LX, Y0 + 134, "绿：本页 3×3 算例", size=12, color="green", weight="600")
svg.text(LX, Y0 + 151, "每轮 × 0.0436（直线）", size=12, color="green")
svg.text(LX, Y0 + 173, "t = 10 时 8.4 × 10^{-15}", size=12, color="green")

svg.save("l16-sinkhorn-steps.svg")
