"""Figure M7.4: heavy-ball momentum versus plain gradient descent on the same quadratic as Figure M7.1.

f(θ) = ½(θ1² + 10 θ2²), κ = 10, start θ0 = (−4, 1.5), velocity v0 = 0.
Heavy ball:  v_{t+1} = μ v_t − η ∇f(θ_t),  θ_{t+1} = θ_t + v_{t+1}.
  GD (μ = 0), η = 2/11                         rate (κ−1)/(κ+1) = 0.818
  heavy ball, μ* = ((√κ−1)/(√κ+1))² = 0.2699, η* = 4/(√h_max+√h_min)² = 0.2309   rate 0.5195
  heavy ball, μ = 0.9, η = 2/11                rate √0.9 = 0.949 (both eigen-directions spiral)
Left: trajectories; right: distance ‖θ_t‖ to the minimum on a log axis. Every point is computed here.
Writes math/figures/m07-momentum.svg
"""

import math

import numpy as np

from m07_svg import SVG, Axes, clipped_path

Hd = np.array([1.0, 10.0])
th0 = np.array([-4.0, 1.5])
k = Hd[1] / Hd[0]
rho = (math.sqrt(k) - 1) / (math.sqrt(k) + 1)
mu_opt = rho ** 2
eta_opt = 4 / (math.sqrt(Hd[1]) + math.sqrt(Hd[0])) ** 2
runs = [
    (0.0, 2 / 11, "ink", "梯度下降 η = 2/11（μ = 0）"),
    (mu_opt, eta_opt, "green", "重球 μ = %.3f，η = %.3f（最优）" % (mu_opt, eta_opt)),
    (0.9, 2 / 11, "orange", "重球 μ = 0.9，η = 2/11（动量过大）"),
]


def run(mu, eta, n):
    th, v = th0.copy(), np.zeros(2)
    out = [th.copy()]
    for _ in range(n):
        v = mu * v - eta * Hd * th
        th = th + v
        out.append(th.copy())
    return np.array(out)


W, H = 740, 400
svg = SVG(W, H, "重球动量与梯度下降在同一个二次函数上的比较")
s = 54.0
ox, oy = 295.0, 204.0
xlo, xhi, ylo, yhi = -4.8, 2.4, -2.55, 3.0
box = (ox + xlo * s, oy - yhi * s, ox + xhi * s, oy - ylo * s)


def P(x, y):
    return ox + x * s, oy - y * s


svg.text((box[0] + box[2]) / 2, 22, "前 25 步的位置", size=14, anchor="middle", weight="600")
svg.rect(box[0], box[1], box[2] - box[0], box[3] - box[1], stroke="rule", width=1)
svg.line(box[0], oy, box[2], oy, color="rule", width=1)
svg.line(ox, box[1], ox, box[3], color="rule", width=1)
phi = np.linspace(0, 2 * math.pi, 361)
for lev in (0.5, 2.0, 4.5, 8.0, 12.5, 19.25):
    r1, r2 = math.sqrt(2 * lev / Hd[0]), math.sqrt(2 * lev / Hd[1])
    svg.path(clipped_path([P(r1 * math.cos(a), r2 * math.sin(a)) for a in phi], box), stroke="muted",
             width=1.0, opacity=0.55)
for mu, eta, col, lab in runs[::-1]:
    tr = run(mu, eta, 25)
    wid, rad = (1.3, 2.0) if col == "orange" else (1.9, 2.7)
    svg.path(clipped_path([P(x, y) for x, y in tr], box), stroke=col, width=wid, opacity=0.9)
    for x, y in tr:
        sx, sy = P(x, y)
        if box[0] <= sx <= box[2] and box[1] <= sy <= box[3]:
            svg.circle(sx, sy, rad, fill=col)
sx, sy = P(*th0)
svg.circle(sx, sy, 5, fill="ink")
svg.line(sx - 2, sy - 6, box[0] + 30, box[1] + 22, color="ink", width=0.8)
svg.text(box[0] + 8, box[1] + 17, "起点 θ_{0} = (−4, 1.5)，*m*_{0} = 0", size=12, weight="600")
svg.text((box[0] + box[2]) / 2, box[3] + 18, "等高线与图 M7.1 相同（h = 1 与 10，κ = 10）", size=11,
         anchor="middle", color="muted")

ax = Axes(470, 44, 720, 330, (0, 60), (1e-9, 10), ylog=True)
svg.text((ax.x0 + ax.x1) / 2, 22, "到极小点的距离 ‖θ_{t}‖（对数纵轴）", size=14, anchor="middle", weight="600")
ax.ygrid(svg, [1e-9, 1e-6, 1e-3, 1, 10], labels=["10^{−9}", "10^{−6}", "10^{−3}", "1", "10"])
ax.xgrid(svg, [0, 20, 40, 60])
ax.frame(svg)
svg.text((ax.x0 + ax.x1) / 2, ax.y1 + 36, "步数 t", size=12, anchor="middle", color="muted")
cbox = (ax.x0, ax.y0, ax.x1, ax.y1)
dist = {}
for mu, eta, col, lab in runs:
    tr = run(mu, eta, 60)
    d = np.linalg.norm(tr, axis=1)
    dist[col] = d
    svg.path(clipped_path([ax(t, max(v, 1e-15)) for t, v in enumerate(d)], cbox), stroke=col, width=2.2)
svg.text(ax.X(60) - 4, ax.y0 + 16, "μ = 0.9：×0.949 / 步", size=12, anchor="end",
         color="orange", weight="600")
# below the falling GD line (a falling line is higher to the left, so text below-left never crosses it)
svg.text(ax.X(60) - 4, ax.Y(dist["ink"][60]) + 20, "GD：×0.818 / 步", size=12, anchor="end", color="ink",
         weight="600")
tg = 30
svg.text(ax.X(tg) + 8, ax.Y(dist["green"][tg]) + 4, "最优重球：×0.520 / 步", size=12, color="green", weight="600")

ly, lx = 388, 24
for mu, eta, col, lab in runs:
    svg.line(lx, ly - 4, lx + 22, ly - 4, color=col, width=2.4)
    svg.text(lx + 28, ly, lab, size=12)
    lx += 238
svg.save("m07-momentum.svg")
for mu, eta, col, lab in runs:
    d = np.linalg.norm(run(mu, eta, 60), axis=1)
    print(lab, "d10=%.3g d30=%.3g d60=%.3g" % (d[10], d[30], d[60]))
print("rho", rho, "mu_opt", mu_opt, "eta_opt", eta_opt)
