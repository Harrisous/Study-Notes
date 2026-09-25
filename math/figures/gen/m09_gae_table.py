"""Figure M9.6: GAE as a weighted average of k-step estimators, read by rows and by columns.

Three-step example from the shared notes (N70): r = (0, 0, 1), V = (0.5, 0.6, 0.8, 0),
gamma = 0.99, lambda = 0.95, at t = 0. Row k is the k-step estimator (sum of the first k terms
gamma^l delta_l); its weight is (1-lambda) lambda^(k-1), the last row takes all remaining weight.
Reading down a column gives lambda^l, so delta_l carries (gamma lambda)^l in total.
Writes math/figures/m09-gae-table.svg
"""

from m09_svg import SVG

r = [0.0, 0.0, 1.0]
V = [0.5, 0.6, 0.8, 0.0]
g, lam = 0.99, 0.95
delta = [r[t] + g * V[t + 1] - V[t] for t in range(3)]
terms = [g ** l * delta[l] for l in range(3)]
rows = [sum(terms[:k]) for k in (1, 2, 3)]
wts = [(1 - lam), (1 - lam) * lam, lam ** 2]
colw = [sum(wts[l:]) for l in range(3)]
gae = sum(w * a for w, a in zip(wts, rows))
gae2 = sum((g * lam) ** l * delta[l] for l in range(3))
assert abs(gae - gae2) < 1e-12 and abs(gae - 0.451484) < 1e-6

W, H = 720, 392
svg = SVG(W, H, "GAE = k 步估计的加权平均：按行读与按列读")

GX, CW, RH, GY = 176, 124, 54, 58
svg.text(16, 22, "例：r = (0, 0, 1)，V = (0.5, 0.6, 0.8, 0)，γ = 0.99，λ = 0.95，站在 t = 0", size=13, weight="600")
labels = ["δ_{0} = %.3f" % delta[0], "γδ_{1} = %.3f" % terms[1], "γ^{2}δ_{2} = %.3f" % terms[2]]
for l in range(3):
    svg.text(GX + CW * l + CW / 2, GY - 10, "l = %d" % l, size=12, anchor="middle", color="muted")

for k in range(3):
    y = GY + k * RH
    # subscript first, so the "0" sits right under the hat instead of after the "(k)"
    svg.text(16, y + 21, "Â_{0}^{(%d)}（%d 步估计）" % (k + 1, k + 1), size=13, weight="600")
    svg.text(16, y + 44, "= %.4f" % rows[k], size=12, color="muted")
    for l in range(3):
        x = GX + CW * l
        if l <= k:
            svg.rect(x + 3, y + 3, CW - 6, RH - 6, fill="blue", fill_opacity=0.14, stroke="blue", width=1.1, rx=6)
            svg.text(x + CW / 2, y + RH / 2 + 5, labels[l], size=13, anchor="middle")
        else:
            svg.rect(x + 3, y + 3, CW - 6, RH - 6, stroke="rule", width=1, rx=6, dash="3 3")
    wx = GX + 3 * CW + 12
    wl = ["(1−λ) = %.4f" % wts[0], "(1−λ)λ = %.4f" % wts[1], "λ^{2} = %.4f（余下全部）" % wts[2]][k]
    svg.text(wx, y + 32, "× " + wl, size=12.5, color="violet")

svg.text(GX + 3 * CW + 12, GY - 10, "行权重", size=12, color="violet", weight="600")

# column sums
yc = GY + 3 * RH + 10
svg.line(GX, yc - 4, GX + 3 * CW, yc - 4, color="rule", width=1)
svg.text(16, yc + 18, "按列加起行权重", size=12.5, color="violet")
svg.text(16, yc + 44, "再乘 γ^{l}：δ_{l} 的总权重", size=12.5, color="green")
for l in range(3):
    cx = GX + CW * l + CW / 2
    lab = ["1", "λ = %.2f" % colw[1], "λ^{2} = %.4f" % colw[2]][l]
    svg.text(cx, yc + 18, lab, size=12.5, anchor="middle", color="violet")
    lab2 = ["(γλ)^{0} = 1", "(γλ)^{1} = %.4f" % (g * lam), "(γλ)^{2} = %.4f" % ((g * lam) ** 2)][l]
    svg.text(cx, yc + 44, lab2, size=12.5, anchor="middle", color="green", weight="600")

yb = yc + 80
svg.rect(12, yb - 20, 696, 64, fill="green", fill_opacity=0.08, stroke="green", width=1.1, rx=8)
svg.text(24, yb, "按行：GAE = 0.05×%.4f + 0.0475×%.4f + 0.9025×%.4f = %.4f" % (rows[0], rows[1], rows[2], gae), size=13)
svg.text(24, yb + 26, "按列：GAE = %.3f + %.4f×%.3f + %.4f×%.3f = %.4f  （同一个数，求和换了顺序）"
         % (delta[0], g * lam, delta[1], (g * lam) ** 2, delta[2], gae2), size=13)

svg.save("m09-gae-table.svg")
