"""Figure M2.5: singular-value spectrum and the best rank-k error (Eckart-Young-Mirsky), with competitors.

Matrix: G = np.random.default_rng(1).standard_normal((6, 4))  (math-refs-doc N24).
Top-left: the singular values sigma_i as bars.
Bottom-left: for k = 0..4, the error of the truncated SVD G_k in both norms:
    spectral  ||G - G_k||_2 = sigma_{k+1}          (computed with np.linalg.norm(., 2))
    Frobenius ||G - G_k||_F = sqrt(sum_{i>k} sigma_i^2)
Right: 20,000 random rank-2 competitors X Y (X random 6x2, Y the least-squares best given X),
histogram of their Frobenius errors, against the optimum 1.5321; also the best pair of actual columns.
Writes math/figures/m02-ey-spectrum.svg
"""

import itertools

import numpy as np

from m02_svg import SVG, polyline_d

G = np.random.default_rng(1).standard_normal((6, 4))
U, s, Vt = np.linalg.svd(G)
err2, errF = [], []
for k in range(5):
    Gk = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
    err2.append(np.linalg.norm(G - Gk, 2) if k < 4 else 0.0)
    errF.append(np.linalg.norm(G - Gk, "fro"))
    if k < 4:
        assert abs(err2[k] - s[k]) < 1e-12
    assert abs(errF[k] - np.sqrt((s[k:] ** 2).sum())) < 1e-12
ey2 = errF[2]

rng = np.random.default_rng(0)
comp = []
for _ in range(20000):
    X = rng.standard_normal((6, 2))
    Y, *_ = np.linalg.lstsq(X, G, rcond=None)
    comp.append(np.linalg.norm(G - X @ Y, "fro"))
comp = np.array(comp)
best_cols = min(np.linalg.norm(G - G[:, c] @ np.linalg.lstsq(G[:, c], G, rcond=None)[0], "fro")
                for c in map(list, itertools.combinations(range(4), 2)))
assert comp.min() > ey2 and best_cols > ey2

W, H = 760, 372
svg = SVG(W, H, "6×4 矩阵的奇异值谱、截断 SVD 在两种范数下的误差，以及 2 万个随机秩 2 近似的误差分布")

# ---------- top-left: spectrum
lx0, lx1 = 70, 360
ty0, ty1 = 44, 140           # chart area (top)
svg.text(20, 22, "① 奇异值谱 σ_{i}", size=14, weight="600")
smax = 2.5


def TY(v):
    return ty1 - (ty1 - ty0) * v / smax


for v in (0, 1, 2):
    svg.line(lx0, TY(v), lx1, TY(v), color="grid", width=1)
    svg.text(lx0 - 8, TY(v) + 4, str(v), size=11, color="muted", anchor="end")
slot = (lx1 - lx0) / 5.0


def KX(k):                   # x position for k = 0..4 (also used for i = 1..4 bars: bar i sits at k = i)
    return lx0 + slot * (k + 0.5)


for i in range(4):
    x = KX(i)                # bar sigma_{k+1} sits right above k in the lower chart
    svg.rect(x - 14, TY(s[i]), 28, TY(0) - TY(s[i]), fill="blue", fill_opacity=0.25, stroke="blue", width=1.2)
    svg.text(x, TY(s[i]) - 5, "%.4f" % s[i], size=11, anchor="middle", color="blue")
    svg.text(x, ty1 + 16, "σ_{%d}" % (i + 1), size=12, anchor="middle", color="blue")

# ---------- bottom-left: error staircase
by0, by1 = 196, 318
svg.text(20, 178, "② 保留前 k 项（截断 SVD）后的误差", size=14, weight="600")
emax = 3.5


def BY(v):
    return by1 - (by1 - by0) * v / emax


for v in (0, 1, 2, 3):
    svg.line(lx0, BY(v), lx1, BY(v), color="grid", width=1)
    svg.text(lx0 - 8, BY(v) + 4, str(v), size=11, color="muted", anchor="end")
for k in range(5):
    svg.text(KX(k), by1 + 16, "k = %d" % k, size=11, anchor="middle", color="muted")
svg.path(polyline_d([(KX(k), BY(errF[k])) for k in range(5)]), stroke="ink", width=2)
svg.path(polyline_d([(KX(k), BY(err2[k])) for k in range(5)]), stroke="blue", width=2, dash="6 3")
for k in range(5):
    svg.circle(KX(k), BY(errF[k]), 4, fill="ink")
    svg.circle(KX(k), BY(err2[k]), 3.6, fill="surface", stroke="blue", width=1.8)
    if k < 4:
        svg.text(KX(k) + 7, BY(errF[k]) - 7, "%.4f" % errF[k], size=11)
svg.line(250, BY(2.75), 272, BY(2.75), color="ink", width=2)
svg.text(278, BY(2.75) + 4, "F 范数误差", size=12)
svg.line(250, BY(2.35), 272, BY(2.35), color="blue", width=2, dash="6 3")
svg.text(278, BY(2.35) + 4, "谱范数误差", size=12, color="blue")
svg.text(20, H - 14, "① 里 σ_{k+1} 的柱子正好在 ② 里 k 的正上方：谱范数误差 ‖*G* − *G*_{k}‖_{2} 就等于它。", size=12,
         color="muted")

# ---------- right: histogram of competitors
rx0, rx1, ry0, ry1 = 430, 740, 44, 300
svg.line(400, 16, 400, H - 30, color="rule", width=1)
svg.text(410, 22, "③ 2 万个随机秩 2 近似的 F 误差", size=14, weight="600")
lo, hi = 1.4, 3.2
nb = 36
hist, edges = np.histogram(comp, bins=nb, range=(lo, hi))
hmax = hist.max()


def HX(v):
    return rx0 + (rx1 - rx0) * (v - lo) / (hi - lo)


def HY(c):
    return ry1 - (ry1 - ry0 - 40) * c / hmax


for i in range(nb):
    if hist[i]:
        svg.rect(HX(edges[i]) + 0.5, HY(hist[i]), HX(edges[i + 1]) - HX(edges[i]) - 1, ry1 - HY(hist[i]),
                 fill="muted", fill_opacity=0.35)
svg.line(rx0, ry1, rx1, ry1, color="rule", width=1)
for v in (1.5, 2.0, 2.5, 3.0):
    svg.line(HX(v), ry1, HX(v), ry1 + 5, color="muted", width=1)
    svg.text(HX(v), ry1 + 18, "%.1f" % v, size=11, color="muted", anchor="middle")
svg.text((rx0 + rx1) / 2, ry1 + 36, "‖*G* − *XY*‖_{F}", size=12, color="muted", anchor="middle")
svg.line(HX(ey2), ry0 + 6, HX(ey2), ry1, color="green", width=2.4)
svg.text(HX(ey2) + 4, ry0 + 18, "截断 SVD：%.4f" % ey2, size=12, color="green")
svg.line(HX(best_cols), ry0 + 44, HX(best_cols), ry1, color="orange", width=1.6, dash="5 3")
svg.text(HX(best_cols) + 4, ry0 + 56, "最好的两列：%.4f" % best_cols, size=12, color="orange")
svg.text(HX(comp.min()) + 4, ry0 + 76, "随机里最好：%.4f" % comp.min(), size=12, color="ink")
svg.polygon([(HX(comp.min()), ry1 - 1), (HX(comp.min()) - 5, ry1 - 10), (HX(comp.min()) + 5, ry1 - 10)], fill="ink")
# leader from the label down to its marker (the label sits 300 units above it)
svg.line(HX(comp.min()), ry0 + 82, HX(comp.min()), ry1 - 12, color="ink", width=1, dash="2 3", opacity=0.7)
# label above the tallest bar (HY(hmax) = ry0 + 40), not on top of the grey bars where it was hard to read
svg.text(HX(np.median(comp)) + 5, ry0 + 28, "中位数 %.2f" % np.median(comp), size=12, color="muted")
svg.line(HX(np.median(comp)), ry0 + 20, HX(np.median(comp)), ry1, color="muted", width=1, dash="2 3")

svg.save("m02-ey-spectrum.svg")
print("s", np.round(s, 4), "errF", np.round(errF, 4), "err2", np.round(err2, 4))
print("competitors min %.4f median %.4f, best cols %.4f" % (comp.min(), np.median(comp), best_cols))
