"""Figure M2.7: where to spend a rank-1 budget depends on where the inputs are (M02 §6, exercise 8).

W = diag(3, 1); inputs x ~ N(0, S_x) with S_x = diag(0.01, 1): almost no spread along e1, normal spread along e2.
  (1) the input cloud (160 samples, seed 5);
  (2) the full output W x, with W's image of the unit circle (semi-axes sigma_1 = 3 along e1, sigma_2 = 1 along e2);
  (3) truncated SVD of W itself, W_1 = diag(3, 0): keeps e1, the direction W stretches most but the inputs barely use;
  (4) data-weighted optimum (M2.20), B* = (W S^{1/2})_1 S^{-1/2} = diag(0, 1): keeps e2.
In (3) and (4) the full outputs W x are drawn faintly behind the kept part, so the lost part is visible.
Errors are exact, from (M2.19): E||W x||^2 = tr(W S W^T) = 1.09; E||(W - W_1) x||^2 = 1 (91.7 %);
E||(W - B*) x||^2 = 0.09 (8.3 %).
Writes math/figures/m02-data-weighted.svg
"""

import numpy as np

from m02_svg import SVG, Plane, c, fmt, polyline_d

W = np.diag([3.0, 1.0])
S = np.diag([0.01, 1.0])
Sh, Shi = np.sqrt(S), np.diag(1 / np.sqrt(np.diag(S)))
tot = float(np.trace(W @ S @ W.T))
U, s, Vt = np.linalg.svd(W)
W1 = s[0] * np.outer(U[:, 0], Vt[0])
U2, s2, Vt2 = np.linalg.svd(W @ Sh)
Bstar = s2[0] * np.outer(U2[:, 0], Vt2[0]) @ Shi


def err(B):
    D = W - B
    return float(np.trace(D @ S @ D.T))


assert np.allclose(W1, np.diag([3.0, 0.0])) and np.allclose(Bstar, np.diag([0.0, 1.0]))
assert abs(tot - 1.09) < 1e-12 and abs(err(W1) - 1.0) < 1e-12 and abs(err(Bstar) - 0.09) < 1e-12

rng = np.random.default_rng(5)
X = rng.standard_normal((2, 160)) * np.sqrt(np.diag(S))[:, None]

WIDTH, H = 760, 282
svg = SVG(WIDTH, H, "秩 1 预算该留给哪个方向：截断 W 自己的 SVD 留下输入几乎不去的方向，按数据加权的最优解留下数据所在的方向")
R = 3.3                      # half-range of every panel, in data units
BOX = 160
SC = BOX / (2 * R)
TOP = 60
cy = TOP + BOX / 2
xs = [12, 204, 396, 588]


def panel(x0, title, sub, line1, line2, col2="muted"):
    P = Plane(x0 + BOX / 2, cy, SC)
    svg.rect(x0, TOP, BOX, BOX, stroke="rule", width=1, rx=6)
    svg.text(x0, 22, title, size=13, weight="600")
    svg.text(x0, 40, sub, size=12, color="muted")
    svg.line(*P(-R + 0.1, 0), *P(R - 0.1, 0), color="grid", width=1.1)
    svg.line(*P(0, -R + 0.1), *P(0, R - 0.1), color="grid", width=1.1)
    svg.text(x0, TOP + BOX + 22, line1, size=12)
    svg.text(x0, TOP + BOX + 44, line2, size=12, color=col2)   # 44: clear of line 1's subscripts
    return P


def dots(P, Y, col, r=1.9, op=0.55):
    # one <g> carries the style so each dot is only cx/cy/r (keeps the inlined page small)
    pts = []
    for x, y in Y.T:
        if abs(x) < R - 0.05 and abs(y) < R - 0.05:      # drop the rare sample outside the box (none at seed 5)
            sx, sy = P(x, y)
            pts.append('<circle cx="%s" cy="%s" r="%s"/>' % (fmt(sx), fmt(sy), fmt(r)))
    svg.raw('<g fill="%s" fill-opacity="%s">%s</g>' % (c(col), fmt(op), "".join(pts)))


assert np.abs(W @ X).max() < R - 0.05

# (1) inputs
P = panel(xs[0], "① 输入 *x*", "160 个样本", "*S*_{x} = diag(0.01, 1)", "横向几乎不动，纵向正常")
dots(P, X, "blue")

# (2) full output, with W's image of the unit circle
P = panel(xs[1], "② 完整输出 *Wx*", "虚线：*W* 把单位圆拉成的椭圆", "*W* = diag(3, 1)",
          "E‖*Wx*‖² = 0.09 + 1 = 1.09")
t = np.linspace(0, 2 * np.pi, 181)
svg.path(polyline_d([P(*(W @ np.array([np.cos(a), np.sin(a)]))) for a in t]) + " Z", stroke="ink", width=1.2,
         dash="4 3")
dots(P, W @ X, "ink", op=0.45)
# semi-axis labels kept outside both the ellipse and the point cloud (|x| < 0.7)
svg.text(*P(3.15, -1.6), "σ_{1} = 3", size=12, anchor="end")
svg.text(*P(0.9, 1.45), "σ_{2} = 1", size=12)

# divider: (3) and (4) are the two rank-1 alternatives
svg.line(xs[2] - 16, 14, xs[2] - 16, H - 12, color="rule", width=1)

# (3) truncated SVD of W itself
P = panel(xs[2], "③ 截断 *W* 的 SVD", "留下 σ_{1} = 3 的横向", "*W*_{1} = diag(3, 0)",
          "丢掉纵向：误差 %.1f%%" % (100 * err(W1) / tot), col2="orange")
dots(P, W @ X, "muted", r=1.7, op=0.3)
dots(P, W1 @ X, "orange", r=2.1, op=0.7)

# (4) data-weighted optimum
P = panel(xs[3], "④ 按数据加权的最优", "留下数据所在的纵向", "*B*^{*} = diag(0, 1)",
          "丢掉横向：误差 %.1f%%" % (100 * err(Bstar) / tot), col2="green")
dots(P, W @ X, "muted", r=1.7, op=0.3)
dots(P, Bstar @ X, "green", r=2.1, op=0.7)

svg.save("m02-data-weighted.svg")
print("total %.4f  err W1 %.4f (%.2f%%)  err B* %.4f (%.2f%%)" % (tot, err(W1), 100 * err(W1) / tot, err(Bstar),
                                                               100 * err(Bstar) / tot))
