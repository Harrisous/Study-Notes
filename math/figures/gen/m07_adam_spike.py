"""Figure M7.9: the aftermath of one gradient spike inside Adam, and what clipping first changes.

Toy model (the author's illustration, not from a paper): one coordinate whose gradient is normally 1, with
Adam's moving averages in steady state m = v = 1 (long after the start, so bias correction ≈ 1),
β1 = 0.9, β2 = 0.95 (DeepSeek's values), ε = 0. At step k = 0 the gradient is 100 (same sign) or −100
(opposite sign); from k = 1 on it is 1 again. Plotted: the step m_k/√v_k in units of η.
  Same sign:  m_k = 1 + 9.9·0.9^k, v_k = 1 + 499.95·0.95^k.  Split into the normal share 1/√v_k and the
              share that is still the spike, 9.9·0.9^k/√v_k.  k=0 0.487, k=10 0.257 (spike 0.199),
              k=20 normal 0.0745, k=100 normal 0.5025.  Σ_{k=0..150} = 62.4 (no spike: 151).
  Opposite:   m_k = 1 − 10.1·0.9^k: the coordinate moves the wrong way for k = 0..21 (22 steps),
              k=0 −0.407;  Σ_{k=0..50} = −1.2, Σ_{k=0..150} = 50.8.
  Clipped first (threshold c = 1, the normal gradient size): the spike enters Adam as ±1;
              +1 changes nothing, −1 gives 0.8 at k = 0 and is back above 0.97 by k = 20;  Σ_{k=0..150} = 149.0.
Every point is computed here. Writes math/figures/m07-adam-spike.svg
"""

import numpy as np

from m07_svg import SVG, Axes, clipped_path

b1, b2 = 0.9, 0.95
K0, K1 = -20, 150
ks = np.arange(K0, K1 + 1)


def run(spike, clip=None):
    """Return (step, normal share) for k = K0..K1; m = v = 1 before the spike."""
    m, v = 1.0, 1.0
    step, normal = [], []
    for k in ks:
        g = spike if k == 0 else 1.0
        if clip is not None and abs(g) > clip:
            g = g * clip / abs(g)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step.append(m / np.sqrt(v))
        normal.append(1 / np.sqrt(v))
    return np.array(step), np.array(normal)


up, un = run(100.0)
dn, _ = run(-100.0)
dc, _ = run(-100.0, clip=1.0)
after = ks >= 0
cum_up, cum_dn, cum_dc = up[after].sum(), dn[after].sum(), dc[after].sum()
neg = ks[(dn < 0)]
assert abs(up[ks == 0][0] - 0.487) < 1e-3 and neg.min() == 0 and neg.max() == 21

W, H = 740, 404
svg = SVG(W, H, "一次梯度尖峰在 Adam 里的余波：同号时步子被压小上百步，反号时先倒退二十多步")


def panel(ax, title, ylo, yticks):
    svg.text((ax.x0 + ax.x1) / 2, 22, title, size=14, anchor="middle", weight="600")
    ax.ygrid(svg, yticks)
    ax.xgrid(svg, [0, 25, 50, 75, 100, 125, 150])
    ax.frame(svg)
    svg.text((ax.x0 + ax.x1) / 2, ax.y1 + 36, "尖峰之后的步数 k", size=12, anchor="middle", color="muted")
    # the no-spike step, also what clipping restores
    svg.line(ax.x0, ax.Y(1), ax.x1, ax.Y(1), color="ink", width=1.3, dash="5 4")
    # the spike step
    svg.line(ax.X(0), ax.y0, ax.X(0), ax.y1, color="muted", width=1, dash="2 3")
    svg.text(ax.X(0) + 4, ax.y0 + 12, "尖峰", size=11, color="muted")
    return (ax.x0, ax.y0, ax.x1, ax.y1)


# ---------- left: same-sign spike, stacked shares
ax = Axes(62, 44, 342, 300, (K0, K1), (0, 1.15))
box = panel(ax, "同号尖峰（g = +100）", 0, [0, 0.25, 0.5, 0.75, 1.0])
k_after = ks[after]
top = [ax(k, s) for k, s in zip(k_after, up[after])]
mid = [ax(k, s) for k, s in zip(k_after, un[after])]
base = [ax(k, 0) for k in k_after]
poly_spike = top + mid[::-1]
poly_normal = mid + base[::-1]
svg.polygon(poly_normal, fill="green", fill_opacity=0.28)
svg.polygon(poly_spike, fill="amber", fill_opacity=0.30)
pre = [ax(k, s) for k, s in zip(ks[~after], up[~after])] + [ax(0, up[after][0])]
svg.path(clipped_path(pre, box), stroke="orange", width=2.4)
svg.path(clipped_path(top, box), stroke="orange", width=2.4)
svg.path(clipped_path(mid, box), stroke="green", width=1.2)
svg.text(ax.X(150) - 4, ax.Y(1) - 8, "平时（或先裁剪）：每步 1η", size=11, color="ink", anchor="end", halo=True)
svg.circle(ax.X(0), ax.Y(up[ks == 0][0]), 3.6, fill="orange")
# value label left of the dot: only the vertical drop at k = 0 is there, and it ends at this dot
svg.text(ax.X(0) - 6, ax.Y(up[ks == 0][0]) + 4, "0.487", size=11, color="orange", weight="600", anchor="end",
         halo=True)
# labels for the two shaded shares, both in the empty band between the curve and the dashed 1η line,
# each with a thin leader line down into its own area
svg.text(ax.X(6), ax.Y(0.66), "滑动平均里残留的尖峰", size=11, color="amber", weight="600", halo=True)
svg.line(ax.X(14), ax.Y(0.63), ax.X(12), ax.Y(0.28), color="amber", width=0.9)
svg.text(ax.X(14), ax.Y(0.86), "正常梯度的份额：v 被抬到 501，", size=11, color="green", weight="600", halo=True)
svg.text(ax.X(14), ax.Y(0.86) + 15, "第 20 步只剩 7%，第 100 步回到一半", size=11, color="green", halo=True)
svg.line(ax.X(40), ax.Y(0.86) + 20, ax.X(58), ax.Y(0.08), color="green", width=0.9)
svg.circle(ax.X(100), ax.Y(un[ks == 100][0]), 3.2, fill="green")
svg.text(ax.X(100) + 6, ax.Y(un[ks == 100][0]) + 16, "0.50", size=11, color="green", weight="600", halo=True)
svg.text(ax.x0 + 4, ax.y1 + 54, "k = 0…150 共走 %.0fη（平时 151η）" % cum_up, size=12, color="orange", weight="600")

# ---------- right: opposite-sign spike, with and without clipping
ax2 = Axes(440, 44, 720, 300, (K0, K1), (-0.5, 1.15))
box2 = panel(ax2, "反号尖峰（g = −100）", -0.5, [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
svg.line(ax2.x0, ax2.Y(0), ax2.x1, ax2.Y(0), color="muted", width=1.1)
kneg = ks[(ks >= 0) & (ks <= 22)]
dneg = np.minimum(dn[(ks >= 0) & (ks <= 22)], 0)
poly = [ax2(k, s) for k, s in zip(kneg, dneg)] + [ax2(k, 0) for k in kneg[::-1]]
svg.polygon(poly, fill="red", fill_opacity=0.16)
svg.path(clipped_path([ax2(k, s) for k, s in zip(ks, dc)], box2), stroke="green", width=2.4)
svg.path(clipped_path([ax2(k, s) for k, s in zip(ks, dn)], box2), stroke="red", width=2.4)
svg.circle(ax2.X(0), ax2.Y(dn[ks == 0][0]), 3.6, fill="red")
svg.text(ax2.X(0) + 7, ax2.Y(dn[ks == 0][0]) + 4, "−0.41", size=11, color="red", weight="600", halo=True)
svg.circle(ax2.X(22), ax2.Y(0), 3.2, fill="red")
svg.text(ax2.X(26), ax2.Y(-0.17), "k = 0…21 往反方向走", size=11, color="red", weight="600", halo=True)
# above the red curve, which is still below 0.36 up to k = 82
svg.text(ax2.X(22), ax2.Y(0.55), "未裁剪：第 100 步", size=11, color="red", halo=True)
svg.text(ax2.X(22), ax2.Y(0.55) + 15, "才恢复到 0.5η", size=11, color="red", halo=True)
svg.circle(ax2.X(0), ax2.Y(dc[ks == 0][0]), 3.2, fill="green")
svg.text(ax2.X(8), ax2.Y(0.80), "先裁剪（c = 1）：共只少走 2η", size=11, color="green", weight="600", halo=True)
svg.text(ax2.x0 + 4, ax2.y1 + 54, "未裁剪共走 %.0fη，先裁剪 %.0fη" % (cum_dn, cum_dc), size=12, color="red",
         weight="600")
svg.text(62, H - 6, "单坐标玩具模型：平时梯度恒为 1，β_{1} = 0.9、β_{2} = 0.95，m = v = 1 的稳态；纵轴是每步位移 m/√v（单位 η）。",
         size=11, color="muted")
svg.save("m07-adam-spike.svg")
print("cum up %.2f  cum down %.2f  cum clipped %.2f" % (cum_up, cum_dn, cum_dc))
for k in (0, 10, 20, 50, 100):
    i = int(np.where(ks == k)[0][0])
    print("k=%d up %.4f normal %.4f  down %.4f  clipped %.4f" % (k, up[i], un[i], dn[i], dc[i]))
