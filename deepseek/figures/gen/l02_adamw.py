"""Figure 2.7: one AdamW step as a data flow (left), and why L2 regularisation inside Adam is not
weight decay (right, computed).

Left: Loshchilov & Hutter (2019) algorithm 2, in the course notation, with GPT-3's settings
(beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weight decay 0.1; Brown et al. 2020 appendix B).
Right: fraction of a weight removed by the regulariser in one step, as a function of that weight's
gradient RMS sqrt(v_hat), for eta = 6e-5 (GPT-3 175B peak) and lambda = 0.1:
  AdamW (decoupled, PyTorch convention): eta * lambda = 6e-6, the same for every weight
  Adam + L2 (coefficient lambda'):       approx. eta * lambda' / sqrt(v_hat)
lambda' is chosen so that the two lines cross at sqrt(v_hat) = 1e-3 (an illustration of the shape,
assuming the L2 term's own contribution to v_hat is negligible).
Writes deepseek/figures/l02-adamw.svg
"""

import numpy as np

from l02_svg import SVG, box, LogAxis

W, H = 720, 430
svg = SVG(W, H, "AdamW 的一步，以及 L2 正则与解耦权重衰减的区别")

# ---------------- left: data flow
svg.text(16, 22, "AdamW 的一步（对每个参数逐元素进行）", size=13, color="ink", weight="600")
bw = 170
box(svg, 110, 36, bw, 44, "梯度 g_{t}", ["（已裁剪）"], color="blue", fill_opacity=0.10, line_gap=15)
box(svg, 16, 106, bw, 76, "一阶矩 m_{t}", ["β_{1}m_{t−1} + (1−β_{1})g_{t}", "梯度的滑动平均"], color="amber",
    line_gap=23)
box(svg, 200, 106, bw, 76, "二阶矩 v_{t}", ["β_{2}v_{t−1} + (1−β_{2})g_{t}^{2}", "梯度平方的滑动平均"],
    color="amber", line_gap=23)
box(svg, 16, 200, bw, 44, "m̂_{t} = m_{t} / (1 − β_{1}^{t})", [], color="muted", fill_opacity=0.06)
box(svg, 200, 200, bw, 44, "v̂_{t} = v_{t} / (1 − β_{2}^{t})", [], color="muted", fill_opacity=0.06)
box(svg, 70, 270, 250, 54, "u_{t} = m̂_{t} / (√v̂_{t} + ε)", ["每个参数的步长被自己的梯度尺度归一化"],
    color="blue", fill_opacity=0.08, line_gap=20)
box(svg, 30, 352, 330, 56, "θ_{t} = θ_{t−1} − η_{t} u_{t} − η_{t} λ θ_{t−1}", ["学习率 η_{t} 来自调度（图 2.8）"],
    color="green", fill_opacity=0.10, line_gap=21)

svg.arrow(160, 81, 101, 106, color="ink", width=1.4)
svg.arrow(230, 81, 285, 106, color="ink", width=1.4)
svg.arrow(101, 183, 101, 198, color="ink", width=1.4)
svg.arrow(285, 183, 285, 198, color="ink", width=1.4)
svg.arrow(101, 245, 150, 270, color="ink", width=1.4)
svg.arrow(285, 245, 245, 270, color="ink", width=1.4)
svg.arrow(195, 325, 195, 350, color="ink", width=1.4)

# decoupled decay note (green) and the L2 alternative (red dashed)
for i, (t, wt) in enumerate((("解耦权重衰减：", "600"), ("−η_{t}λθ_{t−1} 直接", None), ("作用在 θ 上，", None),
                             ("不经过 m、v", None))):
    svg.text(372, 338 + i * 20, t, size=12, color="green", weight=wt)
svg.text(16, 48, "Adam+L2：", size=12, color="red", weight="600")
svg.text(16, 66, "把 λ′θ 加进 g", size=12, color="red")
svg.path("M92 72 H108", stroke="red", width=1.5, dash="3 3", arrow=True)

# ---------------- right: computed shrink fraction
PX0, PX1, PY0, PY1 = 542, 708, 70, 300
ax = LogAxis(1e-5, 1e-1, PX0, PX1)
ay = LogAxis(1e-8, 1e-3, PY1, PY0)
eta, lam, sref = 6e-5, 0.1, 1e-3
svg.text(PX0 - 50, 22, "每步被正则“收走”的比例", size=13, color="ink", weight="600")
svg.text(PX0 - 50, 42, "η = 6×10^{−5}，λ = 0.1", size=12, color="muted")
for e in range(-8, -2):
    y = ay(10.0 ** e)
    svg.line(PX0, y, PX1, y, color="grid", width=1)
    svg.text(PX0 - 6, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
for e in range(-5, 0):
    x = ax(10.0 ** e)
    svg.line(x, PY0, x, PY1, color="grid", width=1)
    if True:
        svg.text(x, PY1 + 16, "10^{%d}" % e, size=11, anchor="middle", color="muted")
svg.line(PX0, PY1, PX1, PY1, color="muted", width=1.2)
svg.line(PX0, PY0, PX0, PY1, color="muted", width=1.2)
svg.text((PX0 + PX1) / 2, PY1 + 38, "该参数的梯度尺度 √v̂（对数）", size=12, anchor="middle", color="muted")
svg.text((PX0 + PX1) / 2, PY1 + 54, "越往右，L2 的正则越弱", size=12, anchor="middle", color="red")

sig = np.logspace(-5, -1, 60)
f_l2 = eta * lam * sref / sig
f_w = np.full_like(sig, eta * lam)
svg.path("M" + " L".join("%.1f %.1f" % (ax(a), ay(b)) for a, b in zip(sig, f_w)), stroke="green", width=2.6)
svg.path("M" + " L".join("%.1f %.1f" % (ax(a), ay(b)) for a, b in zip(sig, f_l2)), stroke="red", width=2.4,
         dash="7 4")
# labels sit in the two empty triangles: AdamW below its line on the left (the red line is above it
# there), Adam+L2 above the green line on the right (the red line has dropped below it there)
svg.text(ax(1.3e-5), ay(eta * lam) + 45, "AdamW：6×10^{−6}", size=12, color="green", weight="600")
svg.text(ax(1.3e-5), ay(eta * lam) + 63, "对所有参数一样", size=12, color="green")
svg.text(PX1 - 4, ay(3e-4), "Adam+L2", size=12, anchor="end", color="red", weight="600")
svg.text(PX1 - 4, ay(3e-4) + 18, "∝ 1/√v̂", size=12, anchor="end", color="red")

svg.save("l02-adamw.svg")
