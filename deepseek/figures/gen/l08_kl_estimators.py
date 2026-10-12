"""Figure 8.9: the three single-sample KL estimators k1, k2, k3 (Schulman 2020).

Notation of the course: w = pi_ref / pi_theta (Schulman's r = p/q with q = pi_theta, p = pi_ref).
Left: the value of each estimator as a function of log w.
Right: relative standard deviation std/true for Schulman's Gaussian test, samples x ~ q = N(0,1),
p = N(mu,1), log w = mu*x - mu^2/2, true KL[q||p] = mu^2/2. Closed forms (derived in the lesson):
  k1: std/true = 2/mu;  k2: std/true = sqrt(2 + mu^2), bias/true = mu^2/4;
  k3: std/true = 2 sqrt(exp(mu^2) - 1 - mu^2) / mu^2, bias 0.
The closed forms are checked against 200-point Gauss-Hermite quadrature below.
Writes deepseek/figures/l08-kl-estimators.svg
"""

import math

import numpy as np

from l08_svg import SVG, polyline

# ---------------- closed forms and a quadrature check
xg, wg = np.polynomial.hermite_e.hermegauss(200)      # probabilists' Hermite: weight exp(-x^2/2)
wg = wg / wg.sum()


def moments(mu):
    lw = mu * xg - mu * mu / 2
    ks = {"k1": -lw, "k2": 0.5 * lw ** 2, "k3": np.exp(lw) - 1 - lw}
    true = mu * mu / 2
    out = {}
    for k, v in ks.items():
        m = (wg * v).sum()
        sd = math.sqrt(max((wg * (v - m) ** 2).sum(), 0))
        out[k] = ((m - true) / true, sd / true)
    return out


def closed(mu):
    t = mu * mu / 2
    return {"k1": (0.0, 2 / mu), "k2": (mu * mu / 4, math.sqrt(2 + mu * mu)),
            "k3": (0.0, math.sqrt(math.exp(mu * mu) - 1 - mu * mu) / t)}


for mu in (0.1, 0.5, 1.0, 1.2, 1.5, 2.0):
    q, c = moments(mu), closed(mu)
    for k in ("k1", "k2", "k3"):
        assert abs(q[k][0] - c[k][0]) < 1e-6 and abs(q[k][1] - c[k][1]) < 1e-4 * max(1, c[k][1]), (mu, k, q[k], c[k])
    print("mu=%.1f KL=%.4f  " % (mu, mu * mu / 2) +
          "  ".join("%s bias/true %.4f std/true %.4f" % (k, *c[k]) for k in ("k1", "k2", "k3")))
# crossing point of k2 and k3 relative std
lo, hi = 0.5, 2.0
for _ in range(60):
    mid = (lo + hi) / 2
    c = closed(mid)
    lo, hi = (mid, hi) if c["k3"][1] < c["k2"][1] else (lo, mid)
print("k3 std exceeds k2 std for mu > %.3f (KL > %.3f)" % (lo, lo * lo / 2))
MU_X = lo

W, H = 740, 380
svg = SVG(W, H, "k1、k2、k3 三个 KL 估计量的取值与相对标准差")
COL = {"k1": "red", "k2": "amber", "k3": "blue"}

# ---------------- left: values vs log w
PX0, PX1, PY0, PY1 = 52, 340, 44, 300
XL, XH, YL, YH = -3.0, 3.0, -3.0, 6.0


def px(v):
    return PX0 + (v - XL) / (XH - XL) * (PX1 - PX0)


def py(v):
    return PY1 - (v - YL) / (YH - YL) * (PY1 - PY0)


for v in range(-3, 7):
    svg.line(PX0, py(v), PX1, py(v), color="grid", width=1)
    svg.text(PX0 - 6, py(v) + 4, ("%d" % v).replace("-", "−"), size=11, anchor="end", color="muted")
for v in range(-3, 4):
    svg.line(px(v), PY0, px(v), PY1, color="grid", width=1)
    svg.text(px(v), PY1 + 16, ("%d" % v).replace("-", "−"), size=11, anchor="middle", color="muted")
svg.line(PX0, py(0), PX1, py(0), color="muted", width=1.1)
svg.line(px(0), PY0, px(0), PY1, color="muted", width=1.1)
svg.text((PX0 + PX1) / 2, PY1 + 36, "log w = log(π_{ref} / π_{θ})", size=12, anchor="middle", color="muted")
svg.text(PX0 - 36, 26, "单个样本上的取值", size=13, weight="700")

xs = np.linspace(XL, XH, 601)
fns = {"k1": lambda t: -t, "k2": lambda t: 0.5 * t * t, "k3": lambda t: math.exp(t) - 1 - t}
for k, f in fns.items():
    seg, pieces = [], []
    for t in xs:
        v = f(t)
        if YL <= v <= YH:
            seg.append((px(t), py(v)))
        elif seg:
            pieces.append(seg)
            seg = []
    if seg:
        pieces.append(seg)
    for piece in pieces:
        polyline(svg, piece, color=COL[k], width=2.6 if k == "k3" else 2.0)
svg.text(px(-2.95), py(5.1), "k2 = ½(log w)²", size=12.5, color="amber", weight="600", halo=True)
# a surface-coloured plate so the vertical axis does not show between the glyphs
svg.rect(px(1.88) - 120, py(5.4) - 13, 122, 18, fill="surface", fill_opacity=1)
svg.text(px(1.88), py(5.4), "k3 = w − 1 − log w", size=12.5, anchor="end", color="blue", weight="600", halo=True)
svg.text(px(-2.9), py(-2.2), "k1 = −log w：", size=12.5, color="red", weight="600", halo=True)
svg.text(px(-2.9), py(-2.75), "w > 1 时为负", size=12.5, color="red", weight="600", halo=True)
svg.text(px(-2.9), py(-0.75), "k3 ≥ 0，在 w = 1 处", size=12, color="blue", halo=True)
svg.text(px(-2.9), py(-1.35), "与 k2 相切（二阶相同）", size=12, color="blue", halo=True)

# ---------------- right: std/true vs mu (log y)
QX0, QX1 = 430, 700
MX0, MX1 = 0.05, 2.0
LY0, LY1 = 0.0, math.log10(50)


def qx(m):
    return QX0 + (m - MX0) / (MX1 - MX0) * (QX1 - QX0)


def qy(v):
    return PY1 - (math.log10(v) - LY0) / (LY1 - LY0) * (PY1 - PY0)


for v in (1, 2, 5, 10, 20, 50):
    svg.line(QX0, qy(v), QX1, qy(v), color="grid", width=1)
    svg.text(QX0 - 6, qy(v) + 4, "%g" % v, size=11, anchor="end", color="muted")
for m in (0.1, 0.5, 1.0, 1.5, 2.0):
    svg.line(qx(m), PY0, qx(m), PY1, color="grid", width=1)
    svg.text(qx(m), PY1 + 16, "%g" % m, size=11, anchor="middle", color="muted")
svg.line(QX0, PY1, QX1, PY1, color="muted", width=1.1)
svg.text((QX0 + QX1) / 2, PY1 + 36, "μ（真 KL = μ²/2）", size=12, anchor="middle", color="muted")
svg.text(QX0 - 36, 26, "相对标准差 std / 真 KL（对数轴）", size=13, weight="700")
ms = np.linspace(MX0, MX1, 400)
for k in ("k1", "k2", "k3"):
    pts = [(qx(m), qy(closed(m)[k][1])) for m in ms if 1 <= closed(m)[k][1] <= 50]
    polyline(svg, pts, color=COL[k], width=2.6 if k == "k3" else 2.0)
for m in (0.1, 1.0):
    for k in ("k1", "k2", "k3"):
        v = closed(m)[k][1]
        svg.circle(qx(m), qy(v), 3.3, fill=COL[k])
# end-of-curve labels and a small value table (instead of crowded point labels)
for k, dy in (("k1", 4), ("k2", 12), ("k3", -2)):
    svg.text(QX1 + 4, qy(closed(MX1)[k][1]) + dy, k, size=12.5, color=COL[k], weight="700")
tx, ty = qx(0.62), 70
svg.rect(tx - 8, ty - 16, 196, 70, fill="surface", stroke="rule", width=1, rx=6)
for j, (h, x) in enumerate((("", 0), ("k1", 62), ("k2", 104), ("k3", 150))):
    if h:
        svg.text(tx + x, ty, h, size=12, color=COL[h], weight="700", anchor="middle")
for i, m in enumerate((0.1, 1.0)):
    yy = ty + 20 + i * 20
    svg.text(tx, yy, "μ = %g" % m, size=12, color="ink")
    for k, x in (("k1", 62), ("k2", 104), ("k3", 150)):
        v = closed(m)[k][1]
        svg.text(tx + x, yy, ("%.0f" % v) if k == "k1" else ("%.3f" % v), size=12, anchor="middle", color="ink")
svg.line(qx(MU_X), qy(1.0), qx(MU_X), qy(3.3), color="muted", width=1, dash="3 3")
svg.text(qx(MU_X) + 4, qy(3.8), "μ ≈ %.2f 以后" % MU_X, size=11.5, color="muted", halo=True)
svg.text(qx(MU_X) + 4, qy(3.8) + 15, "k3 的波动反超 k2", size=11.5, color="muted", halo=True)

svg.text(16, H - 8, "右图：样本 x ~ N(0,1)，参考分布 N(μ,1)。k2 另有偏差：偏差/真值 = μ²/4（μ = 1 时 25%）；k1、k3 无偏。",
         size=12, color="muted")
svg.save("l08-kl-estimators.svg")
