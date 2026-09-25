"""Figure M7.1: gradient descent on the quadratic f(θ) = ½(θ1² + 10 θ2²) with three step sizes.

H = diag(1, 10), so h_min = 1, h_max = 10, κ = 10; start θ0 = (−4, 1.5), f(θ0) = 19.25.
Per coordinate GD multiplies θi by (1 − η h_i):
  η = 0.05  -> (0.95, 0.5)       too small: the flat direction crawls
  η = 2/11  -> (0.818, −0.818)   optimal: both directions shrink by (κ−1)/(κ+1)
  η = 0.21  -> (0.79, −1.1)      > 2/h_max: the steep direction oscillates and grows
Left: contours and the first iterates; right: f(θ_t) on a log axis for t = 0..40.
Every point is computed here. Writes math/figures/m07-gd-quadratic.svg
"""

import math

import numpy as np

from m07_svg import SVG, Axes, clipped_path, fmt

Hd = np.array([1.0, 10.0])
th0 = np.array([-4.0, 1.5])
runs = [
    (0.05, "blue", "η = 0.05（太小）"),
    (2 / 11, "green", "η = 2/11 ≈ 0.182（最优）"),
    (0.21, "red", "η = 0.21（> 2/h_{max}，发散）"),
]


def traj(eta, n):
    th = th0.copy()
    out = [th.copy()]
    for _ in range(n):
        th = th - eta * Hd * th
        out.append(th.copy())
    return np.array(out)


def f(th):
    return 0.5 * np.sum(Hd * th ** 2, axis=-1)


assert abs(f(th0) - 19.25) < 1e-12
W, H = 740, 400
svg = SVG(W, H, "梯度下降在病态二次函数上的轨迹：步长太小、最优、太大")

# ---------- left panel: contours + trajectories (equal aspect)
s = 54.0  # px per unit
ox, oy = 300.0, 204.0  # screen position of the origin
xlo, xhi, ylo, yhi = -4.8, 1.0, -2.55, 3.0
box = (ox + xlo * s, oy - yhi * s, ox + xhi * s, oy - ylo * s)


def P(x, y):
    return ox + x * s, oy - y * s


svg.text((box[0] + box[2]) / 2, 22, "等高线与前几步的位置", size=14, anchor="middle", weight="600")
svg.rect(box[0], box[1], box[2] - box[0], box[3] - box[1], stroke="rule", width=1)
# axes through the minimum
svg.line(box[0], oy, box[2], oy, color="rule", width=1)
svg.line(ox, box[1], ox, box[3], color="rule", width=1)
phi = np.linspace(0, 2 * math.pi, 361)
for lev in (0.5, 2.0, 4.5, 8.0, 12.5, 19.25):
    r1, r2 = math.sqrt(2 * lev / Hd[0]), math.sqrt(2 * lev / Hd[1])
    pts = [P(r1 * math.cos(a), r2 * math.sin(a)) for a in phi]
    d = clipped_path(pts, box)
    svg.path(d, stroke="muted", width=1.0, opacity=0.55)
svg.text(P(-4.0, -2.05)[0], P(-4.0, -2.05)[1], "f = 19.25", size=11, color="muted")
svg.text(ox + 4, oy + 26, "极小点", size=11, color="muted")
svg.text((box[0] + box[2]) / 2, box[3] + 18, "横轴 θ_{1}：平缓方向（h = 1）；纵轴 θ_{2}：陡峭方向（h = 10）", size=11,
         anchor="middle", color="muted")

nsteps = {0.05: 12, 2 / 11: 12, 0.21: 9}
for eta, col, lab in runs:
    tr = traj(eta, nsteps[eta])
    pts = [P(x, y) for x, y in tr]
    svg.path(clipped_path(pts, box), stroke=col, width=1.8, opacity=0.9)
    for (x, y) in tr:
        sx, sy = P(x, y)
        if box[0] <= sx <= box[2] and box[1] <= sy <= box[3]:
            svg.circle(sx, sy, 2.8, fill=col)
sx, sy = P(*th0)
svg.circle(sx, sy, 5, fill="ink")
svg.line(sx - 2, sy - 6, box[0] + 30, box[1] + 22, color="ink", width=0.8)
svg.text(box[0] + 8, box[1] + 17, "起点 θ_{0} = (−4, 1.5)", size=12, weight="600")

# ---------- right panel: f(θ_t) on a log axis
ax = Axes(470, 44, 720, 330, (0, 40), (1e-6, 1e5), ylog=True)
svg.text((ax.x0 + ax.x1) / 2, 22, "损失 f(θ_{t})（对数纵轴）", size=14, anchor="middle", weight="600")
ax.ygrid(svg, [10.0 ** e for e in range(-6, 6, 2)],
         labels=["10^{%s}" % str(e).replace("-", "−") for e in range(-6, 6, 2)])
ax.xgrid(svg, [0, 10, 20, 30, 40])
ax.frame(svg)
svg.text((ax.x0 + ax.x1) / 2, ax.y1 + 36, "步数 t", size=12, anchor="middle", color="muted")
cbox = (ax.x0, ax.y0, ax.x1, ax.y1)
ends = {}
for eta, col, lab in runs:
    tr = traj(eta, 40)
    fs = f(tr)
    pts = [ax(t, max(v, 1e-12)) for t, v in enumerate(fs)]
    svg.path(clipped_path(pts, cbox), stroke=col, width=2.2)
    ends[eta] = fs
# direct labels. The two falling lines are labelled below-left of a point on the line: a falling line
# is higher everywhere to the left of that point, so the text never crosses it.
svg.text(ax.X(40) - 4, ax.Y(ends[0.21][40]) - 8 if ends[0.21][40] < 1e5 else ax.y0 + 14,
         "η = 0.21：×1.21 / 步", size=12, anchor="end", color="red", weight="600")
svg.text(ax.X(40) - 4, ax.Y(ends[0.05][40]) + 18, "η = 0.05", size=12, anchor="end", color="blue", weight="600")
tg = 25
svg.text(ax.X(tg), ax.Y(ends[2 / 11][tg]) + 20, "η = 2/11：×0.669 / 步", size=12, anchor="end",
         color="green", weight="600")

# legend under the left panel
ly = 388
lx = 60
for eta, col, lab in runs:
    svg.line(lx, ly - 4, lx + 22, ly - 4, color=col, width=2.4)
    svg.text(lx + 28, ly, lab, size=12, color="ink")
    lx += 232
svg.save("m07-gd-quadratic.svg")

for eta, col, lab in runs:
    fs = f(traj(eta, 40))
    print(lab, "f_1=%.4g f_10=%.4g f_40=%.4g" % (fs[1], fs[10], fs[40]))
