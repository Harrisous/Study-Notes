"""Figure M7.3: the per-direction convergence factor as a function of curvature h, for the κ = 10 quadratic
of Figures M7.1 and M7.4 (h_min = 1, h_max = 10).

Left, gradient descent: the factor is |1 − η h| (eq. M7.2), a V with its tip at h = 1/η.
  η = 0.05  worst at h_min: 0.95           (too small)
  η = 2/11  both ends equal: 0.818         (optimal, eq. M7.4)
  η = 0.21  worst at h_max: 1.1 > 1        (diverges)
Right, heavy ball: the factor is the larger root modulus of r² − (1 + μ − ηh) r + μ = 0 (eq. M7.9).
  GD (μ = 0), η = 2/11                     the same V as on the left (reference)
  μ* = 0.2699, η* = 0.2309                 complex roots on exactly [1, 10]: flat at √μ* = 0.5195
  μ = 0.9, η = 2/11                        complex roots on [0.0145, 20.9]: flat at √0.9 = 0.949
Outside the complex-root region the two roots are real with product μ, so the larger one exceeds √μ
and the curve bends up. Every point is computed here. Writes math/figures/m07-rate-curves.svg
"""

import math

import numpy as np

from m07_svg import SVG, Axes, clipped_path

H_MIN, H_MAX = 1.0, 10.0
kappa = H_MAX / H_MIN
rho_hb = (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)
mu_opt = rho_hb ** 2
eta_opt_hb = 4 / (math.sqrt(H_MAX) + math.sqrt(H_MIN)) ** 2
eta_gd = 2 / (H_MIN + H_MAX)


def gd_rate(eta, h):
    return abs(1 - eta * h)


def hb_rate(mu, eta, h):
    tr = 1 + mu - eta * h
    disc = tr * tr - 4 * mu
    if disc < 0:
        return math.sqrt(mu)
    s = math.sqrt(disc)
    return max(abs(tr + s), abs(tr - s)) / 2


# checks against the numbers quoted in the lesson
assert abs(gd_rate(eta_gd, 1) - 9 / 11) < 1e-12 and abs(gd_rate(eta_gd, 10) - 9 / 11) < 1e-12
assert abs(hb_rate(mu_opt, eta_opt_hb, 1) - rho_hb) < 1e-6 and abs(hb_rate(mu_opt, eta_opt_hb, 10) - rho_hb) < 1e-6
assert abs(hb_rate(mu_opt, eta_opt_hb, 5) - rho_hb) < 1e-12
assert abs(hb_rate(0.9, eta_gd, 5) - math.sqrt(0.9)) < 1e-12

W, Hh = 740, 392
svg = SVG(W, Hh, "每个特征方向的收敛因子随曲率 h 的变化：梯度下降的 V 形与重球的平台")
hs = np.linspace(0, 12, 1201)


def panel(ax, title):
    svg.text((ax.x0 + ax.x1) / 2, 22, title, size=14, anchor="middle", weight="600")
    svg.rect(ax.X(H_MIN), ax.y0, ax.X(H_MAX) - ax.X(H_MIN), ax.y1 - ax.y0, fill="muted", fill_opacity=0.08)
    ax.ygrid(svg, [0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax.xgrid(svg, [0, 1, 2, 4, 6, 8, 10, 12])
    ax.frame(svg)
    svg.line(ax.x0, ax.Y(1), ax.x1, ax.Y(1), color="red", width=1, dash="4 4")
    svg.text((ax.x0 + ax.x1) / 2, ax.y1 + 36, "曲率 h（灰带：本题的 [h_{min}, h_{max}] = [1, 10]）", size=12,
             anchor="middle", color="muted")
    return (ax.x0, ax.y0, ax.x1, ax.y1)


# ---------- left: gradient descent
ax = Axes(62, 44, 336, 300, (0, 12), (0, 1.3))
box = panel(ax, "梯度下降：|1 − ηh|")
svg.text(ax.X(4.0), ax.Y(1) - 6, "≥ 1 即发散", size=11, color="red")
gd_runs = [(0.05, "blue"), (eta_gd, "ink"), (0.21, "red")]
for eta, col in gd_runs:
    svg.path(clipped_path([ax(h, gd_rate(eta, h)) for h in hs], box), stroke=col, width=2.3)
# the worst direction inside [1, 10] is always an end of the band: mark it and write its value.
# Curve names are in the legend strip; value labels sit in gaps between the curves (halo for guides).
svg.circle(ax.X(1), ax.Y(0.95), 4, fill="blue")
svg.text(ax.X(1) + 7, ax.Y(0.95) - 5, "0.95", size=11, color="blue", weight="600", halo=True)
svg.circle(ax.X(1), ax.Y(9 / 11), 4, fill="ink")
svg.text(ax.X(1) + 8, ax.Y(9 / 11) + 4, "0.818", size=11, color="ink", weight="600", halo=True)
svg.circle(ax.X(10), ax.Y(9 / 11), 4, fill="ink")
# right of the dot and below the black line, which rises to the right; the red line is far to the left here
svg.text(ax.X(10) + 7, ax.Y(9 / 11) + 15, "0.818", size=11, color="ink", weight="600", halo=True)
svg.circle(ax.X(10), ax.Y(1.1), 4, fill="red")
svg.text(ax.X(10) - 7, ax.Y(1.1) + 4, "1.1", size=11, color="red", weight="600", anchor="end", halo=True)

# ---------- right: heavy ball
ax2 = Axes(436, 44, 710, 300, (0, 12), (0, 1.3))
box2 = panel(ax2, "重球：两个特征根中较大的模")
hb_runs = [(0.0, eta_gd, "ink", 1.6), (mu_opt, eta_opt_hb, "green", 2.6), (0.9, eta_gd, "orange", 2.3)]
for mu, eta, col, wid in hb_runs:
    svg.path(clipped_path([ax2(h, hb_rate(mu, eta, h)) for h in hs], box2), stroke=col, width=wid)
svg.circle(ax2.X(1), ax2.Y(rho_hb), 4, fill="green")
svg.circle(ax2.X(10), ax2.Y(rho_hb), 4, fill="green")
# the black V crosses the plateau at h = 2.64 and 8.36, so this label is kept inside that span
svg.text(ax2.X(5.5), ax2.Y(rho_hb) - 9, "复根区：√μ∗ = %.4f" % rho_hb, size=11, color="green", weight="600",
         anchor="middle", halo=True)
svg.text(ax2.X(5.5), ax2.Y(math.sqrt(0.9)) + 17, "μ = 0.9：√0.9 = 0.949", size=11, color="orange",
         weight="600", anchor="middle", halo=True)
svg.text(ax2.X(11.8), ax2.Y(0.30), "区间外根变实数，", size=11, color="green", anchor="end", halo=True)
svg.text(ax2.X(11.8), ax2.Y(0.30) + 14, "较大的模 > √μ∗", size=11, color="green", anchor="end", halo=True)

# legend: one row per panel
rows = [
    (364, "左图：", [("blue", "η = 0.05（太小）", 132), ("ink", "η = 2/11（最优）", 136),
                   ("red", "η = 0.21（> 2/h_{max}）", 0)]),
    (384, "右图：", [("ink", "梯度下降 η = 2/11（μ = 0）", 186),
                   ("green", "重球 μ∗ = %.3f，η∗ = %.3f（最优）" % (mu_opt, eta_opt_hb), 232),
                   ("orange", "重球 μ = 0.9，η = 2/11", 0)]),
]
for ly, head, items in rows:
    svg.text(62, ly, head, size=11, color="muted")
    lx = 100
    for col, lab, step in items:
        svg.line(lx, ly - 4, lx + 18, ly - 4, color=col, width=2.4)
        svg.text(lx + 23, ly, lab, size=11)
        lx += step
svg.save("m07-rate-curves.svg")

print("GD 2/11 ends", gd_rate(eta_gd, 1), gd_rate(eta_gd, 10), "HB* plateau", rho_hb, "mu*", mu_opt, "eta*", eta_opt_hb)
print("HB* at h=0.5, 11, 12:", hb_rate(mu_opt, eta_opt_hb, 0.5), hb_rate(mu_opt, eta_opt_hb, 11),
      hb_rate(mu_opt, eta_opt_hb, 12))
print("mu=0.9 complex region: h in [%.4f, %.2f]" % ((1 - math.sqrt(0.9)) ** 2 / eta_gd, (1 + math.sqrt(0.9)) ** 2 / eta_gd))
