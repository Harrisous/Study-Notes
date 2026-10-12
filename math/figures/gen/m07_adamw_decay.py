"""Figure M7.6: inside Adam, L2 regularisation shrinks a weight by an amount that depends on its gradient
scale; decoupled weight decay (AdamW) shrinks every weight by the same η·λ·w.

Simulation (numpy, seed 0): one scalar weight w0 = 1 whose loss gradient is pure noise s·N(0, 1);
learning rate η = 1e-3, coefficient λ = 0.1, β1 = 0.9, β2 = 0.999, ε = 1e-8, 200 steps, R = 100000 runs.
  Adam + L2 : g_t = s ξ_t + λ w_{t−1}, then the Adam step.
  AdamW     : Adam step on g_t = s ξ_t, plus w ← w − η λ w_{t−1}.
Plotted: mean per-step decrease (w0 − E[w_200]) / 200, for s from 1e-5 to 10^1.5 (log-log).
Dashed: the rough prediction η·λw / sqrt(s² + (λw)²) for Adam + L2 (w = 1), and η·λ = 1e-4 for AdamW.
Writes math/figures/m07-adamw-decay.svg
"""

import numpy as np

from m07_svg import SVG, Axes, clipped_path

eta, lam, b1, b2, eps = 1e-3, 0.1, 0.9, 0.999, 1e-8
steps, R = 200, 100000
svals = np.logspace(-5, 1.5, 27)
rng = np.random.default_rng(0)


def simulate(s, mode):
    w = np.ones(R)
    m = np.zeros(R)
    v = np.zeros(R)
    for t in range(1, steps + 1):
        g = s * rng.standard_normal(R)
        if mode == "l2":
            g = g + lam * w
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        upd = mh / (np.sqrt(vh) + eps)
        if mode == "adamw":
            w = w - eta * (upd + lam * w)
        else:
            w = w - eta * upd
    return (1.0 - w.mean()) / steps, w.std() / np.sqrt(R) / steps


res = {"l2": [], "adamw": []}
for s in svals:
    for mode in res:
        res[mode].append(simulate(s, mode))
l2 = np.array([r[0] for r in res["l2"]])
aw = np.array([r[0] for r in res["adamw"]])
l2se = np.array([r[1] for r in res["l2"]])

W, H = 740, 374
svg = SVG(W, H, "Adam 里 L2 正则与解耦权重衰减的每步收缩量随梯度尺度的变化（模拟）")
ax = Axes(90, 40, 560, 300, (1e-5, 10 ** 1.5), (1e-6, 3e-3), xlog=True, ylog=True)
ax.ygrid(svg, [1e-6, 1e-5, 1e-4, 1e-3], labels=["10^{−6}", "10^{−5}", "10^{−4}", "10^{−3}"])
ax.xgrid(svg, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
         labels=["10^{−5}", "10^{−4}", "10^{−3}", "10^{−2}", "10^{−1}", "1", "10"])
ax.frame(svg)
svg.text((ax.x0 + ax.x1) / 2, ax.y1 + 38, "损失梯度的典型大小 s（对数刻度）", size=12, anchor="middle", color="muted")
svg.text(24, (ax.y0 + ax.y1) / 2, "平均每步 w 减少多少", size=12, anchor="middle", color="muted", rotate=-90)
box = (ax.x0, ax.y0, ax.x1, ax.y1)

xs = np.logspace(-5, 1.5, 300)
pred = eta * lam / np.sqrt(xs ** 2 + lam ** 2)
# simulation = dots; the rough prediction (M7.14) = dashed line drawn on top, so both stay visible
for a, b in zip(svals, l2):
    svg.circle(ax.X(a), ax.Y(b), 3.4, fill="red")
for a, b in zip(svals, aw):
    svg.circle(ax.X(a), ax.Y(b), 3.4, fill="green")
svg.path(clipped_path([ax(a, b) for a, b in zip(xs, pred)], box), stroke="red", width=1.6, dash="6 4")
svg.path(clipped_path([ax(a, eta * lam) for a in (1e-5, 10 ** 1.5)], box), stroke="green", width=1.6,
         dash="6 4")
svg.line(ax.X(lam), ax.y0, ax.X(lam), ax.y1, color="muted", width=1, dash="2 3")
svg.text(ax.X(lam) + 4, ax.y1 - 8, "s = λw = 0.1", size=11, color="muted")

lx = ax.x1 + 14
# second lines sit 19 units lower so a superscript never touches the line above it
svg.text(lx, ax.Y(aw[-1]) - 6, "AdamW（解耦）", size=13, color="green", weight="600")
svg.text(lx, ax.Y(aw[-1]) + 15, "恒为 ηλw ≈ 10^{−4}", size=12, color="green")
svg.text(lx, ax.Y(l2[-1]) - 6, "Adam + L2", size=13, color="red", weight="600")
svg.text(lx, ax.Y(l2[-1]) + 13, "梯度越大，被拉得越少", size=12, color="red")
svg.text(ax.X(1.5e-5), ax.Y(4.2e-4) - 2, "梯度很小时，L2 项主导 Adam 的归一化，", size=11, color="red")
svg.text(ax.X(1.5e-5), ax.Y(4.2e-4) + 17, "每步几乎走满 η = 10^{−3}（比 AdamW 大 10 倍）", size=11, color="red")
svg.text(ax.x0 + 6, H - 8, "点：10 万次模拟的平均（seed 0）。虚线：粗略预测 ηλw/√(s² + (λw)²) 与 ηλw。",
         size=11, color="muted")
svg.save("m07-adamw-decay.svg")
for s, a, b, se in zip(svals, l2, aw, l2se):
    print("s=%.3g  L2 %.3e (se %.1e)  AdamW %.3e  ratio %.2f" % (s, a, se, b, a / b))
