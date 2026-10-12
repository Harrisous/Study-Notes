"""Figure M4.4: the variance of a sum is the sum of every cell of the n x n covariance table.

n = 8 variables, each with variance sigma^2 = 1, pairwise correlation rho = 0, 0.3, 1.
Diagonal cells hold Var(X_i) = 1, the n(n-1) = 56 off-diagonal cells hold Cov(X_i, X_j) = rho.
Var(sum) = n + n(n-1) rho: 8, 24.8, 64; sd = 2.83, 4.98, 8.
Writes math/figures/m04-cov-grid.svg
"""

import math

from m04_svg import SVG

n = 8
RHOS = (0.0, 0.3, 1.0)
# the table only sees correlations, so the left panel is "uncorrelated"; independence is one way to get it
TITLES = ("不相关（ρ = 0）", "两两相关，ρ = 0.3", "完全相关（ρ = 1）")

W, H = 740, 370
svg = SVG(W, H, "和的方差 = n×n 协方差表里所有格子之和")
svg.text(24, 24, "Var(X_{1} + … + X_{8}) = 把 8×8 表里的 Cov(X_{i}, X_{j}) 全部加起来（每个 Var(X_{i}) = σ² = 1）",
         size=13, weight="600")

cs = 22                       # cell size
for k, (rho, title) in enumerate(zip(RHOS, TITLES)):
    px = 34 + k * 244
    gx, gy = px + 22, 88
    svg.text(gx + n * cs / 2, 56, title, size=13, anchor="middle", weight="600",
             color=("blue" if rho == 0 else "red"))
    svg.text(gx + n * cs / 2, 73, "j = 1 … 8", size=11, anchor="middle", color="muted")
    svg.text(gx - 8, gy + n * cs / 2 + 4, "i", size=12, anchor="end", color="muted", italic=True)
    for i in range(n):
        for j in range(n):
            x, y = gx + j * cs, gy + i * cs
            if i == j:
                svg.rect(x, y, cs, cs, fill="blue", fill_opacity=0.55, stroke="blue", width=1)
                svg.text(x + cs / 2, y + cs / 2 + 4, "1", size=11, anchor="middle", color="ink")
            else:
                if rho > 0:
                    svg.rect(x, y, cs, cs, fill="red", fill_opacity=0.5 * rho + 0.08, stroke="rule", width=1)
                else:
                    svg.rect(x, y, cs, cs, fill="surface", stroke="rule", width=1)
                lab = "0" if rho == 0 else ("1" if rho == 1 else ".3")
                svg.text(x + cs / 2, y + cs / 2 + 4, lab, size=11, anchor="middle",
                         color=("muted" if rho == 0 else "ink"))
    var = n + n * (n - 1) * rho
    sd = math.sqrt(var)
    yb = gy + n * cs
    svg.text(px, yb + 24, "对角 %d 格 × 1 = %d" % (n, n), size=12, color="blue")
    svg.text(px, yb + 43, "交叉 %d 格 × %g = %g" % (n * (n - 1), rho, round(n * (n - 1) * rho, 6)), size=12,
             color=("muted" if rho == 0 else "red"))
    svg.text(px, yb + 64, "和的方差 %g，标准差 %.2f" % (round(var, 6), sd), size=12, color="ink", weight="600")
    print("rho", rho, "var", var, "sd", round(sd, 4))

svg.text(24, H - 14, "n 翻倍：对角格数翻倍（方差 ∝ n），全部格数翻 4 倍（∝ n²）。交叉格只要不是 0，n 大时就占主导。",
         size=12, color="muted")
svg.save("m04-cov-grid.svg")
