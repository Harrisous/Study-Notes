"""Figure 3.14: what the DPO loss rewards.

Left : per-pair gradient weight sigma(-m) against the implicit-reward margin m = r_w - r_l,
       with the worked examples (beta = 0.1: m = 0.15; beta = 0.5: m = 0.75) and m = +-1.
Right: the plane of log-ratio changes d_w = log pi(o_w) - log pi_ref(o_w), d_l = same for o_l.
       The loss depends only on m = beta (d_w - d_l), so iso-loss lines are diagonals.
       Shaded: loss already below ln 2 although o_w itself became LESS likely (d_w < 0 < d_w - d_l).
Worked points (lesson-03 research notes sec. 4.7): A = (+1.0, -0.5), B = (-1.0, -3.0), beta = 0.1.
Writes deepseek/figures/l03-dpo-plane.svg
"""
import math
from l03_svg import SVG, Axis, polyline_d, fmt

sig = lambda z: 1 / (1 + math.exp(-z))
loss = lambda m: -math.log(sig(m))
BETA = 0.1
assert round(loss(BETA * 1.5), 4) == 0.6210 and round(loss(BETA * 2.0), 4) == 0.5981

W, H = 756, 384          # 756 wide so the label of point A (right of the plane) is not cut off
svg = SVG(W, H, "DPO 损失奖励什么：梯度权重与对数比平面")


def tw(t, size):
    """Rough text width: CJK glyphs ~1 em, others ~0.6 em (markup characters ignored)."""
    t = t.replace("_{", "").replace("}", "").replace("*", "")
    return sum(size if ord(ch) > 0x2E80 else 0.6 * size for ch in t)


def plate(x, y, t, size, anchor="start", **kw):
    w = tw(t, size) + 6
    x0 = x - 3 if anchor == "start" else (x - w + 3 if anchor == "end" else x - w / 2)
    svg.rect(x0, y - size, w, size + 5, fill="surface", fill_opacity=0.9, rx=3)
    svg.text(x, y, t, size=size, anchor=anchor, **kw)

# ---------------- left: weight curve
ax = Axis(-4, 4, 56, 316)
ay = Axis(0, 1, 280, 60)
svg.text(16, 26, "每对样本的梯度权重 σ(−m)", size=13.5, weight="600")
for v in (-4, -2, 0, 2, 4):
    svg.line(ax(v), ay(0), ax(v), ay(1), color="grid", width=1)
    svg.text(ax(v), ay(0) + 16, fmt(v).replace("-", "−"), size=11, anchor="middle", color="muted")
for v in (0, 0.25, 0.5, 0.75, 1):
    svg.line(ax(-4), ay(v), ax(4), ay(v), color="grid", width=1)
    svg.text(ax(-4) - 6, ay(v) + 4, fmt(v), size=11, anchor="end", color="muted")
svg.line(ax(-4), ay(0), ax(4), ay(0), color="muted", width=1.1)
svg.line(ax(-4), ay(0), ax(-4), ay(1), color="muted", width=1.1)
xs = [-4 + 8 * i / 200 for i in range(201)]
svg.path(polyline_d([(ax(x), ay(sig(-x))) for x in xs]), stroke="blue", width=2.6)
svg.text(ax(0), ay(0) + 36, "隐式奖励间隔 m = r̂_{w} − r̂_{l}", size=12.5, anchor="middle")
pts = [(-1, "m=−1：0.731", -10, 16, "end"), (0, "m=0（初始）：0.5", 8, -8, "start"),
       (0.15, "β=0.1 例：0.463", 8, 12, "start"), (0.75, "β=0.5 例：0.321", 8, 14, "start"),
       (1, "m=1：0.269", 16, 12, "start")]
for m, lab, dx, dy, anc in pts:
    svg.circle(ax(m), ay(sig(-m)), 3.6, fill="ink")
for m, lab, dx, dy, anc in [pts[0], pts[1], pts[4]]:
    svg.text(ax(m) + dx, ay(sig(-m)) + dy, lab, size=11.5, anchor=anc)
svg.text(ax(-3.9), ay(0.18), "排反的对：权重大", size=12, color="red")
svg.text(ax(3.9), ay(0.92), "已排对的对：权重小", size=12, color="green", anchor="end")

# ---------------- right: log-ratio plane
bx = Axis(-4, 2, 430, 710)
by = Axis(-4, 2, 318, 80)
svg.text(410, 26, "对数比平面（β = 0.1）", size=13.5, weight="600")
# shaded failure region: d_w < 0 and d_l < d_w
poly = [(bx(-4), by(-4)), (bx(0), by(-4)), (bx(0), by(0))]
svg.polygon(poly, fill="red", fill_opacity=0.15)   # 0.15 so the region still shows on the dark theme
for v in (-4, -3, -2, -1, 0, 1, 2):
    svg.line(bx(v), by(-4), bx(v), by(2), color="grid", width=1)
    svg.line(bx(-4), by(v), bx(2), by(v), color="grid", width=1)
    svg.text(bx(v), by(-4) + 16, fmt(v).replace("-", "−"), size=11, anchor="middle", color="muted")
    svg.text(bx(-4) - 6, by(v) + 4, fmt(v).replace("-", "−"), size=11, anchor="end", color="muted")
svg.line(bx(0), by(-4), bx(0), by(2), color="muted", width=1.2)
svg.line(bx(-4), by(0), bx(2), by(0), color="muted", width=1.2)
svg.text(bx(-1), by(-4) + 36, "Δ_{w} = log π_{θ}(o_{w}) − log π_{ref}(o_{w})", size=12, anchor="middle")
svg.text(bx(-4) - 36, by(2) - 16, "Δ_{l}：o_{l} 的同一个量", size=12)
# iso-loss diagonals d_w - d_l = c, clipped to the square [-4,2]^2
for c in (-2, 0, 2, 4):
    lo, hi = max(-4, -4 + c), min(2, 2 + c)          # d_w range where d_l = d_w - c stays in [-4, 2]
    svg.line(bx(lo), by(lo - c), bx(hi), by(hi - c), color="blue" if c else "ink", width=1.3 if c else 1.6,
             dash="5 4" if c else None)
    lab = "损失 %.3f" % loss(BETA * c)
    xl = min(hi, 2) - 0.1
    plate(bx(xl) - 2, by(xl - c) - 6, lab, 11, anchor="end", color="blue" if c else "ink")
svg.arrow(bx(-2.6), by(-0.6), bx(-1.7), by(-1.5), color="green", width=1.8)
svg.text(bx(-2.65), by(-0.6) - 6, "损失下降的方向", size=11.5, color="green", anchor="middle")
# worked points
for (dw, dl), lab, dx, dy, anc in [((0, 0), "起点 π_{θ} = π_{ref}", -8, -10, "end"),
                                   ((1.0, -0.5), "A：损失 0.621", 6, 14, "start"),
                                   ((-1.0, -3.0), "B：两边都降，损失 0.598", 6, 16, "start")]:
    svg.circle(bx(dw), by(dl), 4.2, fill="ink")
    plate(bx(dw) + dx, by(dl) + dy, lab, 11.5, anchor=anc)
svg.text(bx(-2.1), by(-3.75), "红色区", size=12, color="red", weight="600")
svg.text(16, 376, "红色区：Δ_{w} < 0 却已在对角线下方，损失低于 ln 2，而偏好回答 o_{w} 的概率比参考模型还低。", size=12, color="red")
svg.save("l03-dpo-plane.svg")
