"""Figure 16.6: simulated composite gain of 60 random 4x4 residual matrices, unconstrained vs constrained.

Author's illustration (not a reproduction of HC training dynamics; lesson-16 research notes N20/N21 ran a
similar experiment). n = 4 streams, 60 i.i.d. factors, 2000 trials, numpy default_rng(0):
  unconstrained      H = I + s G,                  G_ij ~ N(0,1), s in {0.1, 0.2, 0.3}
                     (the lesson writes the perturbation scale as s, not sigma: sigma(.) is the sigmoid)
  row-stochastic     H = rowsoftmax(5 I + 2 G)      (rows sum to 1, columns do not)
  doubly stochastic  H = SK_20(exp(5 I + 2 G))      (mHC eq. (9): 20 x [column-normalise, then row-normalise])
Composite P_L = H_L ... H_1. Forward gain = max_i |row sum_i(P)|, backward gain = max_j |col sum_j(P)|
(mHC section 3.1 "Amax Gain Magnitude").
Also prints (separate generator default_rng(1), 20000 trials, does not change the figure) the median length of
P_60 v for a unit vector v, to compare with the second-order estimate e^{60 s^2} derived in the lesson (section 3).
Writes deepseek/figures/l16-sim-products.svg and prints the numbers quoted in the lesson.
"""

import math

import numpy as np

from l16_svg import SVG

n, L, N = 4, 60, 2000
rng = np.random.default_rng(0)


def sk(M, t=20):
    for _ in range(t):
        M = M / M.sum(-2, keepdims=True)   # T_c: every column sums to 1
        M = M / M.sum(-1, keepdims=True)   # T_r: every row sums to 1
    return M


def rowsoftmax(Z):
    E = np.exp(Z - Z.max(-1, keepdims=True))
    return E / E.sum(-1, keepdims=True)


def run(make):
    P = np.broadcast_to(np.eye(n), (N, n, n)).copy()
    fw, bw = [np.ones(N)], [np.ones(N)]
    for _ in range(L):
        P = make() @ P
        fw.append(np.abs(P.sum(-1)).max(-1))
        bw.append(np.abs(P.sum(-2)).max(-1))
    return np.array(fw), np.array(bw)


def q(a, p):
    return np.percentile(a, p, axis=1)


free = {}
for s in (0.1, 0.2, 0.3):
    free[s] = run(lambda: np.eye(n) + s * rng.standard_normal((N, n, n)))
row_fw, row_bw = run(lambda: rowsoftmax(5 * np.eye(n) + 2 * rng.standard_normal((N, n, n))))
ds_fw, ds_bw = run(lambda: sk(np.exp(5 * np.eye(n) + 2 * rng.standard_normal((N, n, n)))))

for s, (fw, bw) in free.items():
    print("s=%.1f  forward median @10,20..60:" % s, np.round(q(fw, 50)[10::10], 2),
          " p90 @60: %.0f  max @60: %.0f" % (q(fw, 90)[-1], fw[-1].max()),
          " backward median @60: %.2f" % q(bw, 50)[-1])
print("row-stochastic: forward max %.12f; backward median @60 %.3f, p90 %.3f, max %.3f"
      % (row_fw.max(), q(row_bw, 50)[-1], q(row_bw, 90)[-1], row_bw.max()))
print("doubly stochastic: forward max %.12f; backward median @60 %.4f, p90 %.4f, max %.4f"
      % (ds_fw.max(), q(ds_bw, 50)[-1], q(ds_bw, 90)[-1], ds_bw.max()))

# length of P_60 v for a unit vector v (lesson section 3: typical growth ~ e^{60 s^2} for n = 4)
rng_v = np.random.default_rng(1)
NV = 20000
for s in (0.1, 0.2, 0.3):
    v = np.zeros((NV, n))
    v[:, 0] = 1.0                      # G is rotation invariant, so any fixed unit vector gives the same law
    logn = np.zeros(NV)
    for _ in range(L):
        v = v + s * np.einsum("nij,nj->ni", rng_v.standard_normal((NV, n, n)), v)
        r = np.linalg.norm(v, axis=1)
        logn += np.log(r)
        v /= r[:, None]
    print("s=%.1f  median |P_60 v| = %.2f   e^{60 s^2} = %.2f   Amax/|P v| (medians) = %.2f"
          % (s, np.median(np.exp(logn)), math.exp(L * s * s), q(free[s][0], 50)[-1] / np.median(np.exp(logn))))

# ------------------------------------------------------------------ drawing
W, H = 760, 440
svg = SVG(W, H, "60 个随机 4×4 残差矩阵连乘后的增益：无约束 vs 行随机 vs 双随机（模拟）")

# panel (a): log axis
AX0, AX1, AY0, AY1 = 70, 400, 48, 360
E0, E1 = -0.5, 4.0


def ax(l):
    return AX0 + l / L * (AX1 - AX0)


def ay(v):
    return AY1 - (math.log10(v) - E0) / (E1 - E0) * (AY1 - AY0)


svg.text(AX0, 26, "(a) 对数纵轴：无约束的连乘", size=14, weight="700")
for e in range(0, 5):
    y = ay(10 ** e)
    svg.line(AX0, y, AX1, y, color="grid", width=1)
    svg.text(AX0 - 8, y + 4, ["1", "10", "100", "1000", "10^{4}"][e], size=11, anchor="end", color="muted")
for l in range(0, L + 1, 10):
    svg.line(ax(l), AY0, ax(l), AY1, color="grid", width=1)
    svg.text(ax(l), AY1 + 18, str(l), size=11, anchor="middle", color="muted")
svg.line(AX0, AY1, AX1, AY1, color="muted", width=1.2)
svg.line(AX0, AY0, AX0, AY1, color="muted", width=1.2)
svg.text((AX0 + AX1) / 2, AY1 + 40, "连乘的矩阵个数（深度）", size=12, anchor="middle", color="muted")
svg.text(22, (AY0 + AY1) / 2, "复合增益（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)


def clipv(v):
    return min(max(v, 10 ** E0), 10 ** E1)


def band(lo, hi, color, op):
    top = ["%.1f %.1f" % (ax(i), ay(clipv(hi[i]))) for i in range(L + 1)]
    bot = ["%.1f %.1f" % (ax(i), ay(clipv(lo[i]))) for i in reversed(range(L + 1))]
    svg.path("M" + " L".join(top + bot) + " Z", stroke=None, fill=color, fill_opacity=op)


def line(vals, color, width, dash=None, xf=ax, yf=ay):
    d = "M" + " L".join("%.1f %.1f" % (xf(i), yf(v)) for i, v in enumerate(vals))
    svg.path(d, stroke=color, width=width, dash=dash)


styles = {0.1: (1.6, 0.10), 0.2: (2.2, 0.10), 0.3: (2.8, 0.10)}
for s in (0.3, 0.2, 0.1):
    fw = free[s][0]
    band(q(fw, 10), q(fw, 90), "red", styles[s][1])
for s in (0.1, 0.2, 0.3):
    fw = free[s][0]
    med = q(fw, 50)
    line(med, "red", styles[s][0])
    svg.text(AX1 + 8, ay(med[-1]) + 4, "s = %.1f：%s" % (s, ("%.1f" % med[-1]) if med[-1] < 100 else "%d" % round(med[-1])),
             size=12, color="red")
line(ds_fw[:, 0] * 0 + 1, "green", 2.6)
svg.text(AX1 + 8, ay(1) + 16, "双随机：1", size=12, color="green", weight="600")
svg.text(AX0 + 8, AY0 + 16, "实线：中位数；浅色带：第 10–90 百分位", size=12, color="muted")
svg.text(AX0 + 8, AY0 + 33, "H = I + sG，G 的元素 ~ N(0, 1)", size=12, color="muted")

# panel (b): linear zoom near 1
BX0, BX1, BY0, BY1 = 560, 736, 48, 360
V0, V1 = 0.9, 2.5


def bx(l):
    return BX0 + l / L * (BX1 - BX0)


def by(v):
    return BY1 - (v - V0) / (V1 - V0) * (BY1 - BY0)


svg.text(BX0 - 40, 26, "(b) 线性纵轴：两种约束", size=14, weight="700")
for v in (1.0, 1.5, 2.0, 2.5):
    svg.line(BX0, by(v), BX1, by(v), color="grid", width=1)
    svg.text(BX0 - 8, by(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
for l in range(0, L + 1, 20):
    svg.line(bx(l), BY0, bx(l), BY1, color="grid", width=1)
    svg.text(bx(l), BY1 + 18, str(l), size=11, anchor="middle", color="muted")
svg.line(BX0, BY1, BX1, BY1, color="muted", width=1.2)
svg.line(BX0, BY0, BX0, BY1, color="muted", width=1.2)
svg.text((BX0 + BX1) / 2, BY1 + 40, "深度", size=12, anchor="middle", color="muted")

rb_lo, rb_hi = q(row_bw, 10), q(row_bw, 90)
top = ["%.1f %.1f" % (bx(i), by(rb_hi[i])) for i in range(L + 1)]
bot = ["%.1f %.1f" % (bx(i), by(rb_lo[i])) for i in reversed(range(L + 1))]
svg.path("M" + " L".join(top + bot) + " Z", stroke=None, fill="orange", fill_opacity=0.14)
line(q(row_bw, 50), "orange", 2.2, xf=bx, yf=by)
db_lo, db_hi = q(ds_bw, 10), q(ds_bw, 90)
top = ["%.1f %.1f" % (bx(i), by(db_hi[i])) for i in range(L + 1)]
bot = ["%.1f %.1f" % (bx(i), by(db_lo[i])) for i in reversed(range(L + 1))]
svg.path("M" + " L".join(top + bot) + " Z", stroke=None, fill="green", fill_opacity=0.18)
line(q(ds_bw, 50), "green", 2.2, dash="6 3", xf=bx, yf=by)
line([1.0] * (L + 1), "green", 2.6, xf=bx, yf=by)

svg.text(BX0 + 6, by(2.42), "行随机的反向增益", size=12, color="orange", weight="600")
svg.text(BX0 + 6, by(2.42) + 16, "中位 %.2f，p90 %.2f" % (q(row_bw, 50)[-1], q(row_bw, 90)[-1]), size=12,
         color="orange")
svg.text(BX0 + 6, by(1.27), "双随机的反向增益", size=12, color="green", weight="600")
svg.text(BX0 + 6, by(1.27) + 16, "中位 %.3f（虚线）" % q(ds_bw, 50)[-1], size=12, color="green")
svg.text(BX0 + 6, by(1.27) + 32, "两者前向增益 = 1（实线）", size=12, color="green")

svg.save("l16-sim-products.svg")
