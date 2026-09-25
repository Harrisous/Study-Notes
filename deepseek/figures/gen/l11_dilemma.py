"""Figure 11.2: the load-balance / quality dilemma of auxiliary losses (after LossFree fig. 2).

1B-parameter MoE (Wang et al. 2024, arXiv:2408.15664). x = MaxVio_global, y = validation perplexity.
Exact values (table 2): alpha = 0.001 -> (0.72, 9.56); Loss-Free -> (0.04, 9.50).
The other three points are READ OFF the original figure and are approximate
(research notes sec. 5.2, fig. 11.5): alpha = 0.01 -> (0.10, 9.69), alpha = 1e-4 -> (2.22, 9.527),
alpha = 0 -> (5.07, 9.54).
Writes deepseek/figures/l11-dilemma.svg
"""

from l11_svg import SVG, polyline

aux = [  # (label, maxvio, ppl, exact?)
    ("α = 0.01", 0.10, 9.69, False),
    ("α = 0.001", 0.72, 9.56, True),
    ("α = 0.0001", 2.22, 9.527, False),
    ("α = 0（无辅助损失，也无偏置）", 5.07, 9.54, False),
]
free = ("无辅助损失（偏置）", 0.04, 9.50)

W, H = 720, 400
X0, X1, Y0, Y1 = 78, 690, 30, 330
XMIN, XMAX, YMIN, YMAX = 0.0, 5.5, 9.48, 9.71


def px(v):
    return X0 + (v - XMIN) / (XMAX - XMIN) * (X1 - X0)


def py(v):
    return Y1 - (v - YMIN) / (YMAX - YMIN) * (Y1 - Y0)


svg = SVG(W, H, "辅助损失的两难：均衡度与困惑度（1B MoE，据 LossFree 图 2 重绘）")
for t in range(0, 6):
    svg.line(px(t), Y0, px(t), Y1, color="grid", width=1)
    svg.text(px(t), Y1 + 18, str(t), size=11, anchor="middle", color="muted")
for k in range(0, 5):
    v = 9.50 + 0.05 * k
    svg.line(X0, py(v), X1, py(v), color="grid", width=1)
    svg.text(X0 - 8, py(v) + 4, "%.2f" % v, size=11, anchor="end", color="muted")
svg.line(X0, Y1, X1, Y1, color="ink", width=1.2)
svg.line(X0, Y0, X0, Y1, color="ink", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "MaxVio_{global}（越小越均衡；0.72 表示最忙的专家是平均负载的 1.72 倍）", size=12,
         anchor="middle")
svg.text(22, (Y0 + Y1) / 2, "验证集困惑度（越低越好）", size=12, anchor="middle", rotate=-90)

polyline(svg, [(px(m), py(p)) for _, m, p, _ in aux], color="violet", width=1.6, dash="6 4")
offsets = [(14, 4, "start"), (14, -8, "start"), (0, 26, "middle"), (6, -14, "end")]
for (lab, m, p, exact), (dx, dy, anc) in zip(aux, offsets):
    svg.circle(px(m), py(p), 6, fill="violet", fill_opacity=0.85 if exact else 0.25, stroke="violet", width=1.6)
    ps = ("%.3f" % p).rstrip("0")
    txt = "%s：(%s, %s)" % (lab, ("%.2f" % m) if exact else "≈%.2f" % m, ps if exact else "≈" + ps)
    svg.text(px(m) + dx, py(p) + dy, txt, size=12, anchor=anc, color="violet")

# loss-free star
cx, cy = px(free[1]), py(free[2])
import math
pts = []
for k in range(10):
    r = 9 if k % 2 == 0 else 4
    a = -math.pi / 2 + k * math.pi / 5
    pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
svg.polygon(pts, fill="green", fill_opacity=0.9, stroke="green", width=1)
svg.text(cx + 16, cy + 4, "%s：(0.04, 9.50)" % free[0], size=12, color="green", weight="600")

svg.text(px(1.25), py(9.665), "α 大：均衡好，但困惑度高", size=12, color="muted")
svg.text(px(3.0), py(9.595), "α 太小：负载严重失衡，困惑度也回升", size=12, color="muted", anchor="middle")
svg.text(X1, Y0 - 10, "实心点为论文表 2 的精确值；空心点读自原图，是约数", size=11, anchor="end", color="muted")

svg.save("l11-dilemma.svg")
