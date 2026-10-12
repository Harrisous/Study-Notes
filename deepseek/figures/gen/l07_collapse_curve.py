"""Figure 7.12: the expert-level balance loss as routing collapses.

A fraction lam of the tokens collapses onto a fixed set of K_r experts (selects them, and puts
probability 1/K_r on each); the rest route uniformly (each expert selected with frequency K_r/N_r,
probability 1/N_r each). Then (research notes R10)
  L/alpha_1 = K_r[(1-lam) + lam N_r/K_r][(1-lam)/N_r + lam/K_r] + (N_r-K_r)(1-lam)^2/N_r,
from 1 (uniform) to N_r/K_r (full collapse). Curves for (64, 6), (63, 7), (16, 2).
The expression simplifies to the parabola 1 + (N_r/K_r - 1) * lam^2 (lesson exercise 9);
both forms are checked below.
Writes deepseek/figures/l07-collapse-curve.svg
"""

from l07_svg import SVG


def loss(lam, Nr, Kr):
    return Kr * ((1 - lam) + lam * Nr / Kr) * ((1 - lam) / Nr + lam / Kr) + (Nr - Kr) * (1 - lam) ** 2 / Nr


for (Nr, Kr, ref) in ((64, 6, [1.0, 1.0967, 1.6042, 3.4167, 6.4375, 8.8300, 10.6667]),):
    for lam, v in zip((0, 0.1, 0.25, 0.5, 0.75, 0.9, 1), ref):
        assert abs(loss(lam, Nr, Kr) - v) < 1e-4
for (Nr, Kr) in ((64, 6), (63, 7), (16, 2)):
    for k in range(101):
        lam = k / 100
        assert abs(loss(lam, Nr, Kr) - (1 + (Nr / Kr - 1) * lam ** 2)) < 1e-12

W, H = 740, 380
X0, X1, Y0, Y1 = 70, 520, 30, 320
YMAX = 11.5


def px(l):
    return X0 + l * (X1 - X0)


def py(v):
    return Y1 - v / YMAX * (Y1 - Y0)


svg = SVG(W, H, "专家级均衡损失随路由坍缩程度的变化")
for v in range(0, 12, 2):
    svg.line(X0, py(v), X1, py(v), color="grid", width=1)
    svg.text(X0 - 8, py(v) + 4, str(v), size=11, anchor="end", color="muted")
for l in (0, 0.25, 0.5, 0.75, 1.0):
    svg.text(px(l), Y1 + 18, "%g" % l, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "坍缩比例 λ（这部分 token 全部挤进同一组 K_{r} 个专家，其余均匀路由）", size=12, anchor="middle",
         color="muted")
svg.text(20, (Y0 + Y1) / 2, "L_{ExpBal} / α_{1}", size=12, anchor="middle", color="muted", rotate=-90)

curves = [(64, 6, "blue", "16B：N_{r} = 64，K_{r} = 6", 2.6), (63, 7, "violet", "2B：N_{r} = 63，K_{r} = 7", 2.0),
          (16, 2, "muted", "N_{r} = 16，K_{r} = 2", 1.8)]
n = 100
for Nr, Kr, col, lab, wdt in curves:
    pts = [(i / n, loss(i / n, Nr, Kr)) for i in range(n + 1)]
    svg.path("M" + " L".join("%.1f %.1f" % (px(l), py(v)) for l, v in pts), stroke=col, width=wdt)
    end = Nr / Kr
    svg.circle(px(1), py(end), 4, fill=col)
    svg.text(px(1) + 10, py(end) + 4, "%s：终点 %.3g" % (lab, end), size=12, color=col if col != "muted" else "ink")
svg.circle(px(0), py(1), 4.5, fill="ink")
svg.text(px(0) + 12, py(1) + 18, "均匀路由：三条曲线都从 1 出发", size=12, color="ink")

# worked points on the 16B curve
for lam in (0.25, 0.5, 0.75):
    v = loss(lam, 64, 6)
    svg.circle(px(lam), py(v), 3.2, fill="surface", stroke="blue", width=1.6)
    svg.text(px(lam) - 8, py(v) - 8, "%.2f" % v, size=11, anchor="end", color="blue")

svg.text(X1 + 10, py(3.4) + 4, "16B 训练用 α_{1} = 0.001：", size=12, color="ink")
svg.text(X1 + 10, py(3.4) + 22, "损失至多 0.0107（全部坍缩），", size=12, color="ink")
svg.text(X1 + 10, py(3.4) + 40, "数值本身很小，", size=12, color="ink")
svg.text(X1 + 10, py(3.4) + 58, "起作用的是它的梯度。", size=12, color="ink")

svg.save("l07-collapse-curve.svg")
