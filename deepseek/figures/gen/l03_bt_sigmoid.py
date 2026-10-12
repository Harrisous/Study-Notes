"""Figure 3.2: Bradley-Terry preference probability and the reward-model loss.

Left : sigma(D) = 1/(1+e^-D) and Christiano et al.'s 10%-noise version 0.05 + 0.9 sigma(D).
Right: loss -log sigma(D) and its gradient magnitude |dL/dD| = sigma(-D).
D = r_w - r_l is the reward gap between the preferred and the rejected answer.
Writes deepseek/figures/l03-bt-sigmoid.svg
"""
import math
from l03_svg import SVG, Axis, polyline_d, fmt

sig = lambda z: 1 / (1 + math.exp(-z))
W, H = 720, 330
svg = SVG(W, H, "Bradley–Terry 偏好概率与奖励模型损失")


def frame(ax, ay, xt, yt, xlab, ylab, x0, x1, y0, y1):
    for v in xt:
        svg.line(ax(v), ay(y0), ax(v), ay(y1), color="grid", width=1)
        svg.text(ax(v), ay(y0) + 16, fmt(v).replace("-", "−"), size=11, anchor="middle", color="muted")
    for v in yt:
        svg.line(ax(x0), ay(v), ax(x1), ay(v), color="grid", width=1)
        svg.text(ax(x0) - 6, ay(v) + 4, fmt(v), size=11, anchor="end", color="muted")
    svg.line(ax(x0), ay(y0), ax(x1), ay(y0), color="muted", width=1.1)
    svg.line(ax(x0), ay(y0), ax(x0), ay(y1), color="muted", width=1.1)
    svg.text((ax(x0) + ax(x1)) / 2, ay(y0) + 34, xlab, size=12, anchor="middle", color="ink")
    svg.text(ax(x0), ay(y1) - 10, ylab, size=12, color="ink")


# ---------------- left panel
ax = Axis(-6, 6, 52, 340)
ay = Axis(0, 1, 268, 48)
svg.rect(ax(-6), ay(1), ax(0) - ax(-6), ay(0) - ay(1), fill="red", fill_opacity=0.05)
svg.rect(ax(0), ay(1), ax(6) - ax(0), ay(0) - ay(1), fill="green", fill_opacity=0.06)
frame(ax, ay, [-6, -4, -2, 0, 2, 4, 6], [0, 0.25, 0.5, 0.75, 1], "奖励差 Δ = r_{w} − r_{l}", "P(o_{w} ≻ o_{l})", -6, 6, 0, 1)
xs = [-6 + 12 * i / 240 for i in range(241)]
svg.path(polyline_d([(ax(x), ay(0.05 + 0.9 * sig(x))) for x in xs]), stroke="muted", width=1.8, dash="5 4")
svg.path(polyline_d([(ax(x), ay(sig(x))) for x in xs]), stroke="ink", width=2.4)
for x, lab, dx, dy, anc in [(-1, "(−1, 0.269)", 6, 16, "start"), (0, "(0, 0.5)", -8, -6, "end"),
                            (1, "(1, 0.731)", 8, 14, "start"),
                            (math.log(9), "(ln 9 ≈ 2.20, 0.9)", -9, -3, "end")]:
    svg.circle(ax(x), ay(sig(x)), 3.6, fill="ink")
    svg.text(ax(x) + dx, ay(sig(x)) + dy, lab, size=11.5, anchor=anc)
svg.text(ax(-5.7), ay(0.93), "排反了", size=12, color="red")
svg.text(ax(5.7), ay(0.10), "排对了", size=12, color="green", anchor="end")
svg.line(ax(1.4), ay(0.44), ax(2.4), ay(0.44), color="ink", width=2.4)
svg.text(ax(2.6), ay(0.44) + 4, "σ(Δ)", size=12)
svg.line(ax(1.4), ay(0.34), ax(2.4), ay(0.34), color="muted", width=1.8, dash="5 4")
svg.text(ax(2.6), ay(0.34) + 4, "0.05 + 0.9σ(Δ)", size=12, color="muted")

# ---------------- right panel
bx = Axis(-4, 4, 430, 700)
by = Axis(0, 4.2, 268, 48)
svg.rect(bx(-4), by(4.2), bx(0) - bx(-4), by(0) - by(4.2), fill="red", fill_opacity=0.05)
svg.rect(bx(0), by(4.2), bx(4) - bx(0), by(0) - by(4.2), fill="green", fill_opacity=0.06)
frame(bx, by, [-4, -2, 0, 2, 4], [0, 1, 2, 3, 4], "奖励差 Δ", "损失与梯度大小", -4, 4, 0, 4.2)
xs = [-4 + 8 * i / 200 for i in range(201)]
svg.path(polyline_d([(bx(x), by(-math.log(sig(x)))) for x in xs]), stroke="ink", width=2.4)
svg.path(polyline_d([(bx(x), by(sig(-x))) for x in xs]), stroke="blue", width=2, dash="6 4")
for x in (-1, 0, 1.5):
    L, g = -math.log(sig(x)), sig(-x)
    svg.circle(bx(x), by(L), 3.6, fill="ink")
    svg.circle(bx(x), by(g), 3.4, fill="blue")
svg.text(bx(-1) + 8, by(1.3133) - 4, "Δ=−1：损失 1.313", size=11.5)
svg.text(bx(-1) - 6, by(0.7311) + 16, "|梯度| 0.731", size=11.5, color="blue", anchor="end")
svg.text(bx(0) + 8, by(0.6931) - 6, "Δ=0：ln 2 = 0.693", size=11.5)
# leader from the Delta = 1.5 points to their label, routed right of the "Delta=0" label
svg.path(polyline_d([(bx(1.58), by(0.26)), (bx(3.4), by(0.5)), (bx(3.4), by(2.02))]), stroke="muted", width=1)
svg.text(bx(4), by(2.45), "Δ=1.5：损失 0.201", size=11.5, anchor="end")
svg.text(bx(4), by(2.45) + 16, "|梯度| 0.182", size=11.5, color="blue", anchor="end")
svg.line(bx(1.2), by(3.75), bx(2.0), by(3.75), color="ink", width=2.4)
svg.text(bx(2.15), by(3.75) + 4, "−log σ(Δ)", size=12)
svg.line(bx(1.2), by(3.25), bx(2.0), by(3.25), color="blue", width=2, dash="6 4")
svg.text(bx(2.15), by(3.25) + 4, "σ(−Δ)", size=12, color="blue")
svg.save("l03-bt-sigmoid.svg")
