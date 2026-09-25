"""Figure M4.2: E[N] = sum_n n P(N=n) = sum_m P(N>=m), drawn as one area cut two ways.

Example: one round of speculative decoding with acceptance alpha = 0.8 and draft length
gamma = 3 (Leviathan et al. 2023, eq. 1): P(N >= m) = alpha^(m-1) for m = 1..gamma+1.
Writes math/figures/m04-tail-sum.svg
"""

from m04_svg import SVG, fmt

alpha, gamma = 0.8, 3
nmax = gamma + 1
p = {n: (alpha ** (n - 1)) * (1 - alpha) for n in range(1, nmax)}
p[nmax] = alpha ** gamma
tail = {m: sum(v for n, v in p.items() if n >= m) for m in range(1, nmax + 1)}
E_vert = sum(n * v for n, v in p.items())
E_horz = sum(tail.values())
E_formula = (1 - alpha ** (gamma + 1)) / (1 - alpha)
assert abs(E_vert - E_horz) < 1e-12 and abs(E_vert - E_formula) < 1e-12
print("P(N=n)", p, "tail", tail, "E", E_vert)

W, H = 720, 392
svg = SVG(W, H, "尾和公式：同一块面积竖着切与横着切")

X0, XW = 70, 440          # probability 0..1 -> X0..X0+XW
YB, UH = 300, 52          # baseline and height of one unit of N


def X(v):
    return X0 + XW * v


def Y(v):
    return YB - UH * v


svg.text(X0, 26, "投机解码的一轮：接受率 α = 0.8，草稿长 γ = 3，产出 N ∈ {1, 2, 3, 4} 个 token",
         size=13, weight="600")

# columns, tallest first so every horizontal layer starts at the left edge
order = sorted(p, reverse=True)
x = 0.0
cols = []
for n in order:
    x1 = x + p[n]
    cols.append((n, x, x1))
    svg.rect(X(x), Y(n), X(x1) - X(x), Y(0) - Y(n), fill="blue", fill_opacity=0.10 + 0.05 * n,
             stroke="blue", width=1.3)
    x = x1

# horizontal layers
for m in range(1, nmax + 1):
    w = tail[m]
    y_mid = (Y(m - 1) + Y(m)) / 2
    svg.line(X(0), Y(m), X(w), Y(m), color="ink", width=1.2, dash="4 3")
    # a bracket showing the width of layer m
    svg.line(X(0) + 2, y_mid, X(w) - 2, y_mid, color="orange", width=1.6)
    svg.line(X(0) + 2, y_mid - 5, X(0) + 2, y_mid + 5, color="orange", width=1.6)
    svg.line(X(w) - 2, y_mid - 5, X(w) - 2, y_mid + 5, color="orange", width=1.6)
    svg.text(X(1) + 18, y_mid + 5,
             "第 %d 层宽 P(N ≥ %d) = 0.8^{%d} = %s" % (m, m, m - 1, ("%.3f" % w).rstrip("0").rstrip(".")),
             size=12, color="orange")

# axes
svg.line(X(0), Y(0), X(1) + 6, Y(0), color="muted", width=1.1)
svg.line(X(0), Y(0), X(0), Y(nmax) - 8, color="muted", width=1.1)
for m in range(0, nmax + 1):
    svg.line(X(0) - 4, Y(m), X(0), Y(m), color="muted", width=1)
    svg.text(X(0) - 8, Y(m) + 4, str(m), size=11, anchor="end", color="muted")
svg.text(X(0) - 36, (Y(0) + Y(nmax)) / 2, "高度 = N 的取值", size=12, anchor="middle", rotate=-90)
for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
    svg.line(X(t), Y(0), X(t), Y(0) + 4, color="muted", width=1)
    svg.text(X(t), Y(0) + 16, fmt(t), size=11, anchor="middle", color="muted")

# column labels under the axis
for n, a, b in cols:
    xm = (X(a) + X(b)) / 2
    svg.text(xm, Y(0) + 34, "N=%d" % n, size=12, anchor="middle", color="blue")
    svg.text(xm, Y(0) + 50, ("%.3f" % p[n]).rstrip("0").rstrip("."), size=11, anchor="middle", color="blue")
svg.text(X(0), Y(0) + 72, "柱宽 = P(N = n)，柱高 = n，柱面积 = n·P(N = n)", size=12, color="blue")

# the two sums
terms = " + ".join("%d×%s" % (n, ("%.3f" % p[n]).rstrip("0").rstrip(".")) for n in order)
svg.text(X(0), 52, "竖着切：%s = %.3f" % (terms, E_vert), size=12, color="blue")
svg.text(X(0), 70, "横着切：1 + 0.8 + 0.64 + 0.512 = %.3f = (1 − α^{4}) / (1 − α)" % E_horz,
         size=12, color="orange")

svg.save("m04-tail-sum.svg")
