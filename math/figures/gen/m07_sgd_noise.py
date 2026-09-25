"""Figure M7.2: SGD on a noisy quadratic -- the noise floor is proportional to η, and a learning-rate drop
removes the floor of the steep direction within a few dozen steps.

Model (per coordinate i, independent): f(θ) = ½ Σ h_i θ_i², stochastic gradient h_i θ_i + ξ_i with
ξ_i ~ N(0, σ²/B), σ² = 1, B = 1. Curvatures h = (0.02, 1) (flat and steep), start θ0 = (10, 3), f(θ0) = 5.5.
Exact expectation recursions (no sampling needed):
    mean  m_{t+1} = (1 − η h) m_t
    var   v_{t+1} = (1 − η h)² v_t + η² σ²/B
    E f_t = ½ Σ h_i (m_i² + v_i)
Stationary floor per coordinate: ½ h v∞ = η σ² / (2B(2 − η h)).
Three schedules over 3000 steps: η = 0.1; η = 0.01; η = 0.1 then 0.01 from step 2000.
One sampled path (seed 0) of the η = 0.1 run is drawn faintly to show the fluctuation.
Writes math/figures/m07-sgd-noise.svg
"""

import numpy as np

from m07_svg import SVG, Axes, clipped_path, sci

h = np.array([0.02, 1.0])
th0 = np.array([10.0, 3.0])
sig2, Bsz = 1.0, 1
S = 3000
drop = 2000


def expect(etas):
    m, v = th0.copy(), np.zeros(2)
    out = [0.5 * np.sum(h * (m ** 2 + v))]
    for eta in etas:
        m = (1 - eta * h) * m
        v = (1 - eta * h) ** 2 * v + eta ** 2 * sig2 / Bsz
        out.append(0.5 * np.sum(h * (m ** 2 + v)))
    return np.array(out)


def floor(eta):
    return float(np.sum(eta * sig2 / (2 * Bsz * (2 - eta * h))))


hi = np.full(S, 0.1)
lo = np.full(S, 0.01)
dr = np.where(np.arange(S) < drop, 0.1, 0.01)
E_hi, E_lo, E_dr = expect(hi), expect(lo), expect(dr)

rng = np.random.default_rng(0)
th = th0.copy()
path = [0.5 * np.sum(h * th ** 2)]
for t in range(S):
    th = th - 0.1 * (h * th + rng.standard_normal(2) * np.sqrt(sig2 / Bsz))
    path.append(0.5 * np.sum(h * th ** 2))
path = np.array(path)

W, H = 740, 380
svg = SVG(W, H, "带噪声的二次函数上的 SGD：噪声地板与学习率降档")
ax = Axes(78, 40, 600, 318, (0, S), (1e-3, 10), ylog=True)
ax.ygrid(svg, [1e-3, 1e-2, 1e-1, 1, 10], labels=["10^{−3}", "10^{−2}", "10^{−1}", "1", "10"])
ax.xgrid(svg, [0, 500, 1000, 1500, 2000, 2500, 3000])
ax.frame(svg)
svg.text((ax.x0 + ax.x1) / 2, ax.y1 + 36, "步数 t", size=12, anchor="middle", color="muted")
svg.text(20, (ax.y0 + ax.y1) / 2, "损失 f（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)
box = (ax.x0, ax.y0, ax.x1, ax.y1)

step = 3
idx = np.arange(0, S + 1, step)
svg.path(clipped_path([ax(t, max(path[t], 1e-6)) for t in idx], box), stroke="blue", width=0.8, opacity=0.22)

# green (drop) first and wide, so the identical blue curve before step 2000 shows a green halo
for E, col, wid in ((E_dr, "green", 5.0), (E_lo, "violet", 2.4), (E_hi, "blue", 2.0)):
    pts = [ax(t, E[t]) for t in range(0, S + 1, 5)]
    svg.path(clipped_path(pts, box), stroke=col, width=wid)

f_hi, f_lo = floor(0.1), floor(0.01)
for fv in (f_hi, f_lo):
    svg.line(ax.x0, ax.Y(fv), ax.x1, ax.Y(fv), color="muted", width=1.1, dash="5 4")
svg.line(ax.X(drop), ax.y0, ax.X(drop), ax.y1, color="muted", width=1, dash="2 3")
svg.text(ax.X(drop) + 4, ax.y0 + 14, "第 2000 步：η 从 0.1 降到 0.01", size=12, color="green")

# right-hand labels
lx = ax.x1 + 10
svg.text(lx, ax.Y(E_hi[-1]) - 2, "η = 0.1 恒定", size=12, color="blue", weight="600")
svg.text(lx, ax.Y(E_hi[-1]) + 14, "地板 %.4f" % f_hi, size=11, color="muted")
svg.text(lx, ax.Y(E_dr[-1]) + 2, "先 0.1 后 0.01", size=12, color="green", weight="600")
svg.text(lx, ax.Y(E_dr[-1]) + 18, "末值 %.4f" % E_dr[-1], size=11, color="muted")
svg.text(lx, ax.Y(E_lo[-1]) + 4, "η = 0.01 恒定", size=12, color="violet", weight="600")
svg.text(lx, ax.Y(E_lo[-1]) + 20, "末值 %.3f" % E_lo[-1], size=11, color="muted")
svg.text(lx, ax.Y(f_lo) + 4, "地板 %.4f" % f_lo, size=11, color="muted")
svg.text(ax.X(180), ax.Y(2.2e-3), "浅蓝细线：η = 0.1 的一条实际样本轨迹（期望值是粗蓝线）", size=11, color="blue")

svg.save("m07-sgd-noise.svg")
print("floors", f_hi, f_lo)
print("E_hi[end]", E_hi[-1], "E_lo[end]", E_lo[-1], "E_dr[1999]", E_dr[drop], "E_dr[2100]", E_dr[drop + 100],
      "E_dr[end]", E_dr[-1])
print("E_lo at 2000", E_lo[drop])
