"""Figure 2.12: Chinchilla's parametric loss L(N, D) = E + A/N^alpha + B/D^beta (method 3), computed with the
unrounded fit from the arXiv TeX source (lesson-02-doc.md D34):
  E = e^0.5267228 = 1.6934, A = e^6.0073404 = 406.40, B = e^6.0179186 = 410.72, alpha = 0.33917, beta = 0.28491
Left: iso-loss contours in the (N, D) plane, iso-FLOP lines C = 6ND, and the efficient frontier
      N_opt = G (C/6)^a, D_opt = G^-1 (C/6)^b, G = 1.300, a = 0.4565, b = 0.5435.
Right: iso-FLOP slices L(N, C/(6N)) for C = 1e19, 1e21, 1e23 with their minima (C = 1e21: N_opt = 2.22B).
Redrawn after Hoffmann et al. (2022) figure 4 (left) and the idea of figure 3 (right); these curves are the
fitted formula, not the paper's data points. The paper never states the unit of its losses (no "nats" or
"bits" in the arXiv text), so the right panel's axis is labelled only "predicted loss".
Writes deepseek/figures/l02-isoflop.svg
"""

import math

import numpy as np

from l02_svg import SVG, LogAxis

E, A, B = math.exp(0.5267228), math.exp(6.0073404), math.exp(6.0179186)
al, be = 0.33917084, 0.2849083
G = (al * A / (be * B)) ** (1 / (al + be))
a, b = be / (al + be), al / (al + be)


def Lhat(N, D):
    return E + A / N ** al + B / D ** be


def nopt(C):
    return G * (C / 6) ** a


assert abs(nopt(1e21) - 2.217e9) < 5e6

W, H = 720, 390
svg = SVG(W, H, "Chinchilla 的参数化损失：等损失线、等算力线与最优前沿")

# ---------------- left panel
X0, X1, Y0, Y1 = 70, 350, 40, 320
ax = LogAxis(1e8, 1e12, X0, X1)
ay = LogAxis(1e9, 1e13, Y1, Y0)
svg.text(X0 - 50, 20, "(a) 等损失线（细）与等算力线（虚）", size=13, color="ink", weight="600")
for e in range(8, 13):
    x = ax(10.0 ** e)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 16, {8: "100M", 9: "1B", 10: "10B", 11: "100B", 12: "1T"}[e], size=11, anchor="middle",
             color="muted")
for e in range(9, 14):
    y = ay(10.0 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 6, y + 4, {9: "1B", 10: "10B", 11: "100B", 12: "1T", 13: "10T"}[e], size=11, anchor="end",
             color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 36, "参数量 N", size=12, anchor="middle", color="muted")
svg.text(18, (Y0 + Y1) / 2, "训练 token 数 D", size=12, anchor="middle", color="muted", rotate=-90)

Ns = np.logspace(8, 12, 300)
for lev in (1.9, 2.0, 2.2, 2.5, 3.0):
    rest = lev - E - A / Ns ** al
    m = rest > 0
    Ds = np.where(m, (B / np.where(m, rest, 1)) ** (1 / be), np.nan)
    ok = m & (Ds >= 1e9) & (Ds <= 1e13)
    if ok.sum() < 2:
        continue
    svg.path("M" + " L".join("%.1f %.1f" % (ax(n), ay(d)) for n, d in zip(Ns[ok], Ds[ok])), stroke="muted",
             width=1.2)
    i = np.where(ok)[0][-1]
    svg.text(ax(Ns[i]) - 2, ay(Ds[i]) - 5, "L=%.1f" % lev, size=11, anchor="end", color="muted")
for C, lab in ((1e19, "10^{19}"), (1e21, "10^{21}"), (1e23, "10^{23}"), (1e25, "10^{25}")):
    Dl = C / (6 * Ns)
    ok = (Dl >= 1e9) & (Dl <= 1e13)
    svg.path("M" + " L".join("%.1f %.1f" % (ax(n), ay(d)) for n, d in zip(Ns[ok], Dl[ok])), stroke="orange",
             width=1.3, dash="5 4")
    idx = np.where(ok)[0]
    j = idx[int(len(idx) * {1e19: 0.6, 1e21: 0.6, 1e23: 0.42, 1e25: 0.72}[C])]
    svg.rect(ax(Ns[j]) + 2, ay(Dl[j]) - 2, 52, 16, fill="surface", fill_opacity=0.9, rx=3)
    svg.text(ax(Ns[j]) + 5, ay(Dl[j]) + 11, "C=" + lab, size=11, color="orange")
Cf = np.logspace(19, 26, 100)
Nf, Df = nopt(Cf), Cf / (6 * nopt(Cf))
ok = (Nf >= 1e8) & (Nf <= 1e12) & (Df >= 1e9) & (Df <= 1e13)
svg.path("M" + " L".join("%.1f %.1f" % (ax(n), ay(d)) for n, d in zip(Nf[ok], Df[ok])), stroke="blue", width=2.6)
xt, yt = ax(Nf[ok][-1]), ay(Df[ok][-1])
svg.rect(xt - 118, yt + 2, 96, 36, fill="surface", fill_opacity=0.9, rx=4)
svg.text(xt - 26, yt + 16, "最优前沿", size=12, color="blue", weight="600", anchor="end")
svg.text(xt - 26, yt + 33, "N ∝ C^{0.46}", size=11, color="blue", anchor="end")

# ---------------- right panel
RX0, RX1, RY0, RY1 = 440, 704, 40, 320
rx = LogAxis(1e8, 1e12, RX0, RX1)
ry = LogAxis(1.8, 3.2, RY1, RY0, log=False)
svg.text(RX0 - 40, 20, "(b) 固定算力，改变 N 与 D 的分法", size=13, color="ink", weight="600")
for e in range(8, 13):
    x = rx(10.0 ** e)
    svg.line(x, RY0, x, RY1, color="grid", width=1)
    svg.text(x, RY1 + 16, {8: "100M", 9: "1B", 10: "10B", 11: "100B", 12: "1T"}[e], size=11, anchor="middle",
             color="muted")
for v in (1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2):
    y = ry(v)
    svg.line(RX0, y, RX1, y, color="grid", width=1)
    svg.text(RX0 - 6, y + 4, "%.1f" % v, size=11, anchor="end", color="muted")
svg.line(RX0, RY1, RX1, RY1, color="muted", width=1.2)
svg.line(RX0, RY0, RX0, RY1, color="muted", width=1.2)
svg.text((RX0 + RX1) / 2, RY1 + 36, "参数量 N（D = C / 6N）", size=12, anchor="middle", color="muted")
svg.text(RX0 + 6, RY0 + 14, "预测损失", size=11, color="muted")
for C, col, lab in ((1e19, "red", "C = 10^{19}"), (1e21, "orange", "C = 10^{21}"), (1e23, "green", "C = 10^{23}")):
    L = Lhat(Ns, C / (6 * Ns))
    ok = (L <= 3.2) & (L >= 1.8)
    svg.path("M" + " L".join("%.1f %.1f" % (rx(n), ry(l)) for n, l in zip(Ns[ok], L[ok])), stroke=col, width=2.2)
    n0 = nopt(C)
    l0 = Lhat(n0, C / (6 * n0))
    svg.circle(rx(n0), ry(l0), 4.5, fill=col)
    lab2 = "%s：谷底 N = %s" % (lab, ("%.0fM" % (n0 / 1e6)) if n0 < 1e9 else ("%.2fB" % (n0 / 1e9)))
    if C < 1e20:
        svg.text(rx(n0) + 8, ry(l0) + 18, lab2, size=11, color=col, weight="600")
    elif C < 1e22:
        # below-right of the valley: centred below it, the C = 10^23 curve would cut through the label
        svg.text(rx(n0) + 8, ry(l0) + 26, lab2, size=11, color=col, weight="600")
    else:
        svg.text(rx(n0), ry(l0) + 20, lab2, size=11, anchor="middle", color=col, weight="600")

svg.text(16, H - 8, "由 Chinchilla 方法 3 的拟合式（未四舍五入参数）计算；左图据论文图 4 重绘，右图示意方法 2 的 IsoFLOP 谷底。",
         size=12, color="ink")

svg.save("l02-isoflop.svg")
