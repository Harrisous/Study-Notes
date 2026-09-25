"""Figure M11.9: how fast Sinkhorn-Knopp converges, and what 20 rounds leave behind.

Error after round t = max_j |column sum_j - 1| (each round ends with the row step, so row sums are exactly 1).
(a) error vs t (log axis) for
    - the 3x3 worked example A = [[2,1,1],[1,3,1],[1,4,6]] of M11 §5,
    - the 4x4 "mHC-like" matrix exp(H~) with H~ ~ N(0,1) from the research notes (N33), and the same logits
      scaled by s = 3, 5, 10 (N34),
    - [[1,1],[0,1]] (support but not total support; error exactly 1/(2t+1), N35),
    - [[1,1,1],[1,0,0],[1,0,0]] (no positive diagonal: the column sums tend to (2, 1/2, 1/2)).
(b) after exactly 20 rounds (t_max of mHC), for exp(s * H~) with s from 0.25 to 12: column error and
    ||M^(20)||_2 - 1 (the spectral norm can exceed 1 when the columns have not converged).
Writes math/figures/m11-sinkhorn-convergence.svg
"""

import math

import numpy as np

from m11_svg import SVG

Ht = np.array([[-0.88445, -0.25963, -0.23522, -0.00985],
               [0.42466, -0.11329, -0.84359, -0.63865],
               [-0.62088, -0.48862, -1.06303, 0.55300],
               [0.19527, 0.90549, 3.10495, 2.35742]])
TMAX = 40


def errors(M, T=TMAX):
    M = np.array(M, float)
    out = []
    for _ in range(T):
        M = M / M.sum(0, keepdims=True)
        M = M / M.sum(1, keepdims=True)
        out.append(np.abs(M.sum(0) - 1).max())
    return out, M


def after20(M):
    M = np.array(M, float)
    for _ in range(20):
        M = M / M.sum(0, keepdims=True)
        M = M / M.sum(1, keepdims=True)
    return np.abs(M.sum(0) - 1).max(), np.linalg.norm(M, 2) - 1


curves = [
    ("3×3 算例（§5）", errors([[2, 1, 1], [1, 3, 1], [1, 4, 6]])[0], "blue", None, 2.4),
    ("4×4，exp(H̃)", errors(np.exp(Ht))[0], "green", None, 2.4),
    ("4×4，exp(3H̃)", errors(np.exp(3 * Ht))[0], "amber", None, 2.0),
    ("4×4，exp(5H̃)", errors(np.exp(5 * Ht))[0], "orange", None, 2.0),
    ("4×4，exp(10H̃)", errors(np.exp(10 * Ht))[0], "red", None, 2.0),
    ("[1, 1; 0, 1]：1/(2t+1)", errors([[1, 1], [0, 1]])[0], "violet", "6 4", 2.0),
]
with np.errstate(divide="ignore", invalid="ignore"):
    nosup = errors([[1, 1, 1], [1, 0, 0], [1, 0, 0]])[0]
curves.append(("无正对角线：误差停在 1", nosup, "muted", "2 3", 2.0))
for k, (name, e, *_r) in enumerate(curves):
    print("curve", k, " ".join("t%d=%.3g" % (t, e[t - 1]) for t in (1, 2, 3, 5, 10, 20, 40)))

W, H = 760, 470
svg = SVG(W, H, "Sinkhorn–Knopp 的收敛：正矩阵按几何级数收敛，logit 越悬殊越慢；有支撑无完全支撑时只有 1/t；没有正对角线时不收敛")

X0, X1, Y0, Y1 = 70, 400, 56, 330
LY0, LY1 = -16.3, 0.4


def px(t):
    return X0 + (t - 1) / (TMAX - 1) * (X1 - X0)


def py(v):
    return Y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg.text(X0 - 50, 28, "(a) 第 t 轮后的最大列和误差（对数纵轴）", size=13, weight="600")
for e in range(0, -17, -2):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "1" if e == 0 else "10^{%d}" % e, size=11, anchor="end", color="muted")
for t in (1, 10, 20, 30, 40):
    svg.line(px(t), Y1, px(t), Y1 + 4, color="muted", width=1)
    svg.text(px(t), Y1 + 18, str(t), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "迭代轮数 t（每轮先列后行）", size=12, anchor="middle", color="muted")
svg.line(px(20), Y0, px(20), Y1, color="ink", width=1.2, dash="4 3")
svg.text(px(20) + 6, py(1e-7), "mHC 取 20 轮", size=12, color="ink", weight="600")

for name, e, col, dash, wdt in curves:
    pts = []
    for t, v in enumerate(e, 1):
        if not np.isfinite(v) or v < 2e-16:
            break
        pts.append((px(t), py(v)))
    svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke=col, width=wdt, dash=dash, join="round")

# legend below panel (a)
LX, LY = 24, Y1 + 66
for k, (name, e, col, dash, wdt) in enumerate(curves):
    x = LX + (k % 2) * 200
    y = LY + (k // 2) * 20
    svg.line(x, y, x + 24, y, color=col, width=wdt, dash=dash)
    svg.text(x + 30, y + 4, name, size=12, color=col if col != "muted" else "muted")

# ---- panel (b): after 20 rounds vs logit scale
X0b, X1b = 500, 730
LYb0, LYb1 = -13, 0


def pxb(s):
    return X0b + (s - 0) / 12 * (X1b - X0b)


def pyb(v):
    v = max(v, 10 ** LYb0)
    return Y1 - (math.log10(v) - LYb0) / (LYb1 - LYb0) * (Y1 - Y0)


svg.text(X0b - 56, 28, "(b) 只做 20 轮：logit 放大 s 倍", size=13, weight="600")
for e in range(0, -14, -2):
    y = pyb(10 ** e)
    svg.line(X0b, y, X1b, y, color="grid", width=1)
    svg.text(X0b - 8, y + 4, "1" if e == 0 else "10^{%d}" % e, size=11, anchor="end", color="muted")
for s in (0, 3, 6, 9, 12):
    svg.line(pxb(s), Y1, pxb(s), Y1 + 4, color="muted", width=1)
    svg.text(pxb(s), Y1 + 18, str(s), size=11, anchor="middle", color="muted")
svg.line(X0b, Y1, X1b, Y1, color="muted", width=1.2)
svg.line(X0b, Y0, X0b, Y1, color="muted", width=1.2)
svg.text((X0b + X1b) / 2, Y1 + 38, "logit 缩放 s：M⁽⁰⁾ = exp(s·H̃)", size=12, anchor="middle", color="muted")
ss = np.linspace(0.25, 12, 95)
ce, ne = [], []
for s in ss:
    a, b = after20(np.exp(s * Ht))
    ce.append(a)
    ne.append(b)
pts = [(pxb(s), pyb(v)) for s, v in zip(ss, ce) if v > 10 ** LYb0]
svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke="orange", width=2.4)
pts = [(pxb(s), pyb(v)) for s, v in zip(ss, ne) if v > 10 ** LYb0]
svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke="red", width=2.0, dash="5 3")
for s in (1, 3, 5, 10):
    a, b = after20(np.exp(s * Ht))
    svg.circle(pxb(s), pyb(a), 3.4, fill="orange")
    print("s=%g after 20: colerr %.4g  norm-1 %.4g" % (s, a, b))
svg.text(X0b + 6, Y1 + 66, "橙线：20 轮后的最大列和误差", size=12, color="orange")
svg.text(X0b + 6, Y1 + 86, "红虚线：‖M⁽²⁰⁾‖₂ − 1（大于 0 即", size=12, color="red")
svg.text(X0b + 6, Y1 + 104, "谱范数超过 1）", size=12, color="red")
svg.text(X0b + 6, Y1 + 124, "圆点：s = 1、3、5、10", size=12, color="muted")
svg.save("m11-sinkhorn-convergence.svg")
