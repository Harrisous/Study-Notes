"""Figure M11.10: products of 60 mixing matrices. (a) spectral norm of the product vs depth; (b) how fast a
product of doubly stochastic matrices forgets everything but the average (distance to J_n).

(a) deterministic: H = [[1.2, 0.3], [0.1, 0.9]] (M06 / research notes N30), ||H^L||_2.
    simulation (numpy, seed 0, 1000 trials, n = 4, L = 60), a fresh matrix per layer:
      unconstrained  I + 0.2 G,             G_ij ~ N(0, 1)
      row-stochastic  softmax of each row of the logits  3 I + Z,  Z_ij ~ N(0, 1)
      doubly stoch.   20 Sinkhorn-Knopp rounds (column then row) on exp(3 I + Z), same logits
    curves = median over trials; band = 10th-90th percentile (unconstrained), max (row-stochastic).
(b) ||D^L - J_3||_2 for the circulant D = 0.5 I + 0.3 C + 0.2 C^2 and for the running example A of M11,
    and the median of ||P_L - J_4||_2 for products of the fully converged (200-round) Sinkhorn projections.
Writes math/figures/m11-products.svg
"""

import math

import numpy as np

from m11_svg import SVG, fmt

rng = np.random.default_rng(0)
n, L, T = 4, 60, 1000
SIG, BETA = 0.2, 3.0


def sk(M, t):
    for _ in range(t):
        M = M / M.sum(0, keepdims=True)
        M = M / M.sum(1, keepdims=True)
    return M


un = np.zeros((T, L))
rs = np.zeros((T, L))
ds = np.zeros((T, L))
dev = np.zeros((T, L))
J4 = np.ones((n, n)) / n
for tr in range(T):
    Pu, Pr, Pd, Pe = np.eye(n), np.eye(n), np.eye(n), np.eye(n)
    for l in range(L):
        Hu = np.eye(n) + SIG * rng.standard_normal((n, n))
        E = np.exp(BETA * np.eye(n) + rng.standard_normal((n, n)))
        Pu = Hu @ Pu
        Pr = (E / E.sum(1, keepdims=True)) @ Pr
        Pd = sk(E, 20) @ Pd
        Pe = sk(E, 200) @ Pe
        un[tr, l] = np.linalg.norm(Pu, 2)
        rs[tr, l] = np.linalg.norm(Pr, 2)
        ds[tr, l] = np.linalg.norm(Pd, 2)
        dev[tr, l] = np.linalg.norm(Pe - J4, 2)

H = np.array([[1.2, 0.3], [0.1, 0.9]])
hL = [np.linalg.norm(np.linalg.matrix_power(H, k), 2) for k in range(1, L + 1)]
C = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
D = 0.5 * np.eye(3) + 0.3 * C + 0.2 * C @ C
A = np.array([[0.6, 0.4, 0], [0.1, 0.3, 0.6], [0.3, 0.3, 0.4]])
J3 = np.ones((3, 3)) / 3
dD = [np.linalg.norm(np.linalg.matrix_power(D, k) - J3, 2) for k in range(1, L + 1)]
dA = [np.linalg.norm(np.linalg.matrix_power(A, k) - J3, 2) for k in range(1, L + 1)]

med = lambda a: np.median(a, axis=0)
pct = lambda a, p: np.percentile(a, p, axis=0)
for k in (1, 10, 20, 30, 60):
    i = k - 1
    print("L=%d  H %.4g | un med %.3g p10 %.3g p90 %.3g | row med %.4f p90 %.4f max %.4f | ds med %.7f max %.7f"
          " | devSK med %.3g | D %.3g A %.3g" % (k, hL[i], med(un)[i], pct(un, 10)[i], pct(un, 90)[i], med(rs)[i],
                                                pct(rs, 90)[i], rs[:, i].max(), med(ds)[i], ds[:, i].max(),
                                                med(dev)[i], dD[i], dA[i]))
print("overall max ds", ds.max(), "overall max row", rs.max())

W, H_ = 760, 566
svg = SVG(W, H_, "60 个混合矩阵连乘：无约束的会爆炸，只约束行和的有界但会超过 1，双随机的恒为 1；代价是乘积趋向“全部取平均”")

# ---------------- panel (a)
X0, X1, Y0, Y1 = 70, 380, 56, 330
LY0, LY1 = -0.35, 6.7          # log10 of norm


def px(l):
    return X0 + (l - 1) / (L - 1) * (X1 - X0)


def py(v):
    return Y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg.text(X0 - 50, 28, "(a) 乘积的谱范数 ‖H_{L}⋯H_{1}‖_{2}（对数纵轴）", size=13, weight="600")
for e in range(0, 7):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "1" if e == 0 else "10" if e == 1 else "10^{%d}" % e, size=11, anchor="end", color="muted")
for l in (1, 10, 20, 30, 40, 50, 60):
    svg.line(px(l), Y1, px(l), Y1 + 4, color="muted", width=1)
    svg.text(px(l), Y1 + 18, str(l), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "层数 L", size=12, anchor="middle", color="muted")


def poly(vals, col, width=2.2, dash=None):
    d = "M" + " L".join("%.1f %.1f" % (px(k + 1), py(v)) for k, v in enumerate(vals))
    svg.path(d, stroke=col, width=width, dash=dash, join="round")


lo, hi = pct(un, 10), pct(un, 90)
band = [(px(k + 1), py(v)) for k, v in enumerate(hi)] + [(px(k + 1), py(v)) for k, v in reversed(list(enumerate(lo)))]
svg.polygon(band, fill="orange", fill_opacity=0.14)
poly(med(un), "orange")
poly(hL, "red", width=2.4)
svg.line(X0, py(2), X1, py(2), color="blue", width=1.2, dash="5 4")
poly(rs.max(axis=0), "blue", width=1.2, dash="2 3")
poly(med(rs), "blue", width=2.2)
poly(med(ds), "green", width=2.6)

# inset: on the log axis the two constrained families are squeezed into the band 1..2 (about 12 units tall);
# redraw that band on a linear axis. The box sits above the red line (checked: red is below y=176 for x<=236).
BX, BY, BW, BH = 84, 58, 152, 114
svg.rect(BX, BY, BW, BH, fill="surface", stroke="rule", width=1, rx=4)
svg.text(BX + 8, BY + 14, "放大 1–2（线性纵轴）", size=11, color="muted")
IX0, IX1, IY0, IY1 = BX + 32, BX + BW - 8, BY + 24, BY + BH - 22
ZL, ZH = 0.9, 2.1


def ipx(l):
    return IX0 + (l - 1) / (L - 1) * (IX1 - IX0)


def ipy(v):
    return IY1 - (v - ZL) / (ZH - ZL) * (IY1 - IY0)


for v in (1, 1.5, 2):
    svg.line(IX0, ipy(v), IX1, ipy(v), color="grid", width=1)
    svg.text(IX0 - 4, ipy(v) + 4, fmt(v), size=11, anchor="end", color="muted")
for l in (1, 30, 60):
    svg.text(ipx(l), IY1 + 14, str(l), size=11, anchor="middle", color="muted")
svg.line(IX0, IY1, IX1, IY1, color="muted", width=1)
svg.line(IX0, IY0 - 4, IX0, IY1, color="muted", width=1)


def ipoly(vals, col, width=2.0, dash=None):
    d = "M" + " L".join("%.1f %.1f" % (ipx(k + 1), ipy(v)) for k, v in enumerate(vals))
    svg.path(d, stroke=col, width=width, dash=dash, join="round")


svg.line(IX0, ipy(2), IX1, ipy(2), color="blue", width=1.2, dash="5 4")
svg.text(IX1, ipy(2) + 13, "上界 √4", size=11, anchor="end", color="blue")
ipoly(rs.max(axis=0), "blue", width=1.4, dash="2 3")
ipoly(med(rs), "blue", width=2.0)
ipoly(med(ds), "green", width=2.2)

# ---------------- panel (b)
X0b, X1b = 470, 730
LYb0, LYb1 = -16.3, 0.3


def pxb(l):
    return X0b + (l - 1) / (L - 1) * (X1b - X0b)


def pyb(v):
    return Y1 - (math.log10(v) - LYb0) / (LYb1 - LYb0) * (Y1 - Y0)


svg.text(X0b - 50, 28, "(b) 代价：乘积离 J 有多远 ‖P_{L} − J‖_{2}", size=13, weight="600")
for e in range(0, -17, -4):
    y = pyb(10 ** e)
    svg.line(X0b, y, X1b, y, color="grid", width=1)
    svg.text(X0b - 8, y + 4, "1" if e == 0 else "10^{%d}" % e, size=11, anchor="end", color="muted")
for l in (1, 20, 40, 60):
    svg.line(pxb(l), Y1, pxb(l), Y1 + 4, color="muted", width=1)
    svg.text(pxb(l), Y1 + 18, str(l), size=11, anchor="middle", color="muted")
svg.line(X0b, Y1, X1b, Y1, color="muted", width=1.2)
svg.line(X0b, Y0, X0b, Y1, color="muted", width=1.2)
svg.text((X0b + X1b) / 2, Y1 + 38, "层数 L", size=12, anchor="middle", color="muted")


def polyb(vals, col, width=2.2, dash=None, floor=1e-15):
    pts = []
    for k, v in enumerate(vals):
        if v < floor:
            break
        pts.append((pxb(k + 1), pyb(v)))
    d = "M" + " L".join("%.1f %.1f" % p for p in pts)
    svg.path(d, stroke=col, width=width, dash=dash, join="round")
    return pts[-1]


polyb(med(dev), "green", width=2.4)
eD = polyb(dD, "violet", width=2.2)
eA = polyb(dA, "blue", width=2.2, dash="5 3")
svg.line(X0b, pyb(1e-15), X1b, pyb(1e-15), color="muted", width=1, dash="2 3")
svg.text(X1b, pyb(1e-15) + 14, "双精度的分辨率附近，曲线在此截断", size=11, anchor="end", color="muted")


# ---------------- legends
def entry(x, y, col, text, dash=None, width=2.4, band=False):
    if band:
        svg.rect(x, y - 6, 26, 12, fill=col, fill_opacity=0.14)
    svg.line(x, y, x + 26, y, color=col, width=width, dash=dash)
    svg.text(x + 34, y + 4, text, size=12, color=col if col != "muted" else "muted")


LX, LYL = 20, Y1 + 72
entry(LX, LYL, "red", "固定的 H = [1.2, 0.3; 0.1, 0.9]：60 层后 2.84×10^{6}")
entry(LX, LYL + 22, "orange", "随机 I + 0.2G：中位 %s（色带：10%%–90%%）" % fmt(med(un)[-1]), band=True)
entry(LX, LYL + 44, "blue", "行随机（逐行 softmax）：中位 %.3f" % med(rs)[-1])
entry(LX, LYL + 66, "blue", "行随机：各层最大值，最高 %.2f" % rs.max(), dash="2 3", width=1.2)
entry(LX, LYL + 88, "blue", "行随机的上界 √4 = 2", dash="5 4", width=1.2)
entry(LX, LYL + 110, "green", "双随机（Sinkhorn 20 次）：中位 %.6f，最高 %.4f" % (med(ds)[-1], ds.max()), width=2.6)

LXb = 440
entry(LXb, LYL, "violet", "循环矩阵 D：恰为 0.2646^{L}")
entry(LXb, LYL + 22, "blue", "主算例 A（第二特征值 0.2）", dash="5 3")
entry(LXb, LYL + 44, "green", "随机双随机（充分迭代的投影）：中位")
svg.text(LXb + 34, LYL + 68, "L = 60 时 %s；各流都趋向同一个平均值" % ("%.1e" % med(dev)[-1]).replace("e-0", "×10^{-").replace("e-", "×10^{-") + "}", size=12, color="green")
svg.text(LX, H_ - 12, "模拟：n = 4，每层独立抽一个矩阵，1000 次试验，numpy 种子 0。(a) 的行随机与双随机用同一组 logit 3I + Z，Z 为标准正态。",
         size=11, color="muted")
svg.save("m11-products.svg")
