"""Figure M2.8: learning the two thin factors directly vs truncating the SVD of a fixed matrix (toy experiment).

Setting (M02 §6): m = n = 32. Target map W = randn(32, 32) / sqrt(32) (seed 0).
Inputs x ~ N(0, S_x), S_x = Q diag(lam) Q^T with lam_i = 10^(-3 i / 31) (1 ... 0.001) and a random orthonormal Q
(same seed, drawn before W, so its eigenbasis is unrelated to W's singular vectors).
Loss of an approximation B: E ||(W - B) x||^2 = ||(W - B) S_x^{1/2}||_F^2, reported relative to E ||W x||^2.
  plain   : truncated SVD of W itself (ignores the inputs)
  optimal : truncated SVD of W S_x^{1/2}, mapped back with S_x^{-1/2}  (derived in §6)
  learned : U (32 x k), V (k x 32) trained by minibatch SGD on samples (x, W x) only:
            batch 64, lr 0.05, 60,000 steps, init N(0, 0.05^2), seed 2
Left: relative error vs k (log scale). Right: normalised singular values of W and of W S_x^{1/2} (log scale).
Runs in about 15 s. Writes math/figures/m02-learned-vs-svd.svg
"""

import math

import numpy as np

from m02_svg import SVG, polyline_d

m = n = 32
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
lam = 10.0 ** (-3 * np.arange(n) / (n - 1))
Sx = Q @ np.diag(lam) @ Q.T
Sh = Q @ np.diag(np.sqrt(lam)) @ Q.T
Shi = Q @ np.diag(1 / np.sqrt(lam)) @ Q.T
W = rng.standard_normal((m, n)) / np.sqrt(n)


def loss(B):
    D = W - B
    return float(np.trace(D @ Sx @ D.T))


tot = loss(np.zeros_like(W))


def trunc(A, k):
    U, s, Vt = np.linalg.svd(A)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k]


ks = list(range(0, n + 1))
plain = [loss(trunc(W, k)) / tot for k in ks]
opt = [loss(trunc(W @ Sh, k) @ Shi) / tot for k in ks]
# the optimum equals the discarded energy of W S^{1/2}
sWS = np.linalg.svd(W @ Sh, compute_uv=False)
assert all(abs(opt[k] - (sWS[k:] ** 2).sum() / (sWS ** 2).sum()) < 1e-9 for k in ks)
sW = np.linalg.svd(W, compute_uv=False)


def train_sgd(k, steps=60000, lr=0.05, bs=64, seed=2):
    r = np.random.default_rng(seed)
    U = r.standard_normal((m, k)) * 0.05
    V = r.standard_normal((k, n)) * 0.05
    for _ in range(steps):
        X = Sh @ r.standard_normal((n, bs))
        Z = V @ X
        E = U @ Z - W @ X
        gU = 2 * E @ Z.T / bs
        gV = 2 * U.T @ E @ X.T / bs
        U -= lr * gU
        V -= lr * gV
    return loss(U @ V) / tot


learned = {k: train_sgd(k) for k in (1, 2, 4, 8, 12, 16, 24)}

W_, H = 760, 330
svg = SVG(W_, H, "玩具实验：直接学习两块瘦矩阵，与截断 SVD 的误差对比")

# ---------- left: relative error vs k (log)
x0, x1, y0, y1 = 70, 400, 50, 262
lo, hi = -4.0, 0.0            # log10 range


def X_(k):
    return x0 + (x1 - x0) * k / 32.0


def Y_(v):
    v = max(v, 10 ** lo)
    return y1 - (y1 - y0) * (math.log10(v) - lo) / (hi - lo)


svg.text(20, 24, "① 相对误差 E‖(*W* − *B*)*x*‖² / E‖*Wx*‖²", size=14, weight="600")
for e in range(int(lo), 1):
    svg.line(x0, Y_(10 ** e), x1, Y_(10 ** e), color="grid", width=1)
    lab = "1" if e == 0 else "10^{%d}" % e
    svg.text(x0 - 8, Y_(10 ** e) + 4, lab, size=11, color="muted", anchor="end")
for k in (0, 8, 16, 24, 32):
    svg.line(X_(k), y1, X_(k), y1 + 5, color="muted", width=1)
    svg.text(X_(k), y1 + 19, str(k), size=11, color="muted", anchor="middle")
svg.text((x0 + x1) / 2, y1 + 38, "秩预算 k（中间维）", size=12, color="muted", anchor="middle")
pk = [k for k in ks if k <= 30]
svg.path(polyline_d([(X_(k), Y_(plain[k])) for k in pk]), stroke="orange", width=2.2)
svg.path(polyline_d([(X_(k), Y_(opt[k])) for k in pk if opt[k] > 10 ** lo]), stroke="green", width=2.2)
for k, v in learned.items():
    svg.circle(X_(k), Y_(v), 4.6, fill="blue")
svg.text(X_(9), Y_(plain[8]) - 10, "截断 *W* 自己的 SVD", size=12, color="orange")
svg.text(X_(0.8), Y_(10 ** -2.35), "绿线：按数据加权的最优解", size=12, color="green")
svg.circle(X_(21.5), Y_(0.4), 4.6, fill="blue")
svg.text(X_(22.5), Y_(0.4) + 4, "SGD 直接学 *U*、*V*", size=12, color="blue")
# annotate k = 8
svg.line(X_(8), Y_(plain[8]), X_(8), Y_(opt[8]), color="ink", width=1, dash="2 3")
# two short lines: on one line the text ran into the green curve near k = 25
svg.text(X_(0.8), Y_(10 ** -3.3), "k = 8：截断 %.1f%%" % (100 * plain[8]), size=12)
svg.text(X_(0.8), Y_(10 ** -3.3) + 18, "最优 %.2f%%，SGD %.2f%%" % (100 * opt[8], 100 * learned[8]), size=12)

# ---------- right: spectra
svg.line(430, 16, 430, H - 16, color="rule", width=1)
rx0, rx1 = 480, 740
svg.text(440, 24, "② 两个矩阵的奇异值（各自除以最大值）", size=14, weight="600")
lo2 = -4.0


def RX(i):
    return rx0 + (rx1 - rx0) * (i - 1) / 31.0


def RY(v):
    return y1 - (y1 - y0) * (math.log10(max(v, 10 ** lo2)) - lo2) / (0 - lo2)


for e in range(int(lo2), 1):
    svg.line(rx0, RY(10 ** e), rx1, RY(10 ** e), color="grid", width=1)
    svg.text(rx0 - 8, RY(10 ** e) + 4, "1" if e == 0 else "10^{%d}" % e, size=11, color="muted", anchor="end")
for i in (1, 8, 16, 24, 32):
    svg.line(RX(i), y1, RX(i), y1 + 5, color="muted", width=1)
    svg.text(RX(i), y1 + 19, str(i), size=11, color="muted", anchor="middle")
svg.text((rx0 + rx1) / 2, y1 + 38, "第 i 个奇异值", size=12, color="muted", anchor="middle")
svg.path(polyline_d([(RX(i + 1), RY(sW[i] / sW[0])) for i in range(32)]), stroke="orange", width=2)
svg.path(polyline_d([(RX(i + 1), RY(sWS[i] / sWS[0])) for i in range(32)]), stroke="green", width=2)
for i in range(32):
    svg.circle(RX(i + 1), RY(sW[i] / sW[0]), 2.4, fill="orange")
    svg.circle(RX(i + 1), RY(sWS[i] / sWS[0]), 2.4, fill="green")
svg.text(RX(9), RY(sW[8] / sW[0]) - 10, "*W*", size=12, color="orange")
svg.text(RX(2), RY(10 ** -2.6), "*WS*_{x}^{1/2}（绿）：衰减更快", size=12, color="green")
svg.text(20, H - 12, "输入方差从 1 到 0.001 按对数均匀分布、方向与 *W* 无关；m = n = 32。橙线与绿线逐 k 精确计算，蓝点是训练 6 万步后的结果。",
         size=12, color="muted")

svg.save("m02-learned-vs-svd.svg")
print("plain", [round(plain[k], 4) for k in (1, 2, 4, 8, 12, 16, 24)])
print("opt  ", [round(opt[k], 4) for k in (1, 2, 4, 8, 12, 16, 24)])
print("learn", {k: round(v, 4) for k, v in learned.items()})
print("sW", round(sW[0], 4), round(sW[7], 4), round(sW[-1], 4), "sWS", round(sWS[0], 4), round(sWS[7], 4), round(sWS[-1], 5))
