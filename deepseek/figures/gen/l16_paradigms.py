"""Figure 16.3: residual connection vs Hyper-Connections vs mHC (redrawn after mHC Fig. 1).

Mechanism figure. Shapes follow the lesson notation (research notes section 2.0):
X_l in R^{n x d} with n = 4 streams (rows); H^pre, H^post in R^{1 x n}; H^res in R^{n x n}.
mHC constraints (notes C29-C33): H^pre = sigmoid(.) in (0,1), H^post = 2 sigmoid(.) in (0,2),
H^res = Sinkhorn-Knopp with t_max = 20 -> doubly stochastic.
Writes deepseek/figures/l16-paradigms.svg
"""

from l16_svg import SVG

W, H = 760, 470
svg = SVG(W, H, "残差连接、Hyper-Connections 与 mHC 的一个子层")

TOP, BOT = 58, 372
TAP, PLUS = 76, 318


def panel_residual(x0):
    x = x0 + 70
    svg.text(x0 + 120, 24, "(a) 残差连接", size=14, anchor="middle", weight="700")
    svg.text(x, TOP - 12, "*x*_{l} ∈ ℝ^{d}", size=13, anchor="middle")
    svg.line(x, TOP, x, PLUS - 9, color="ink", width=3)
    svg.arrow(x, PLUS + 9, x, BOT, color="ink", width=3)
    svg.text(x, BOT + 18, "*x*_{l+1}", size=13, anchor="middle")
    # branch
    bx = x0 + 170
    svg.circle(x, TAP, 3.5, fill="ink")
    svg.path("M%d %d H%d V%d" % (x, TAP, bx, 150), stroke="ink", width=1.6, arrow=True)
    svg.box(bx - 42, 152, 84, 44, ["𝓕_{l}", ("含 pre-norm", {"size": 12, "color": "muted"})], color="blue",
            size=14)
    svg.path("M%d %d V%d H%d" % (bx, 196, PLUS, x + 11), stroke="ink", width=1.6, arrow=True)
    svg.circle(x, PLUS, 9, fill="surface", stroke="ink", width=1.6)
    svg.text(x, PLUS + 5, "+", size=15, anchor="middle", weight="700")
    svg.text(x + 10, 262, "恒等捷径：", size=12, color="green", weight="600")
    svg.text(x + 10, 279, "系数固定为 1", size=12, color="green")
    svg.text(x0 + 120, 420, "*x*_{l+1} = *x*_{l} + 𝓕_{l}(*x*_{l})", size=13, anchor="middle")
    svg.text(x0 + 120, 440, "一条带子，捷径系数恒为 1", size=12, anchor="middle", color="muted")


def panel_hc(x0, mhc):
    col = "green" if mhc else "red"
    xs = [x0 + 30 + 24 * i for i in range(4)]
    title = "(c) mHC：三个映射都加约束" if mhc else "(b) HC：三个映射都可学"
    svg.text(x0 + 125, 24, title, size=14, anchor="middle", weight="700")
    svg.text((xs[0] + xs[3]) / 2, TOP - 12, "*X*_{l} ∈ ℝ^{4×d}", size=13, anchor="middle")
    # streams: top -> res box -> plus -> bottom
    RES0, RES1 = 198, 240
    for x in xs:
        svg.line(x, TOP, x, RES0, color="ink", width=1.6)
        svg.line(x, RES1, x, PLUS - 7, color="ink", width=1.6)
        svg.arrow(x, PLUS + 7, x, BOT, color="ink", width=1.6)
        svg.circle(x, PLUS, 7, fill="surface", stroke="ink", width=1.3)
        svg.text(x, PLUS + 4.5, "+", size=12, anchor="middle", weight="700")
    svg.text((xs[0] + xs[3]) / 2, BOT + 18, "*X*_{l+1}", size=13, anchor="middle")
    # res box
    svg.rect(x0 + 12, RES0, 116, RES1 - RES0, fill=col, fill_opacity=0.10, stroke=col, width=1.4, rx=6)
    svg.text(x0 + 70, RES0 + 17, "H^{res}_{l}  4×4", size=13, anchor="middle")
    svg.text(x0 + 70, RES0 + 34, "SK×20 → 双随机" if mhc else "任意实数矩阵", size=12,
             anchor="middle", color=col)
    # branch column
    bx = x0 + 192
    svg.rect(bx - 48, 86, 96, 42, fill=col, fill_opacity=0.10, stroke=col, width=1.4, rx=6)
    svg.text(bx, 103, "H^{pre}_{l}  1×4", size=13, anchor="middle")
    svg.text(bx, 120, "σ(·) ∈ (0,1)" if mhc else "读出：加权合流", size=12, anchor="middle", color=col)
    for x in xs:
        svg.circle(x, TAP, 2.6, fill="ink")
        svg.line(x, TAP, bx - 48, 107, color="muted", width=1)
    svg.arrow(bx, 128, bx, 146, color="ink", width=1.6)
    svg.box(bx - 40, 148, 80, 40, ["𝓕_{l}", ("1×d → 1×d", {"size": 12, "color": "muted"})],
            color="blue", size=14)
    svg.arrow(bx, 188, bx, 204, color="ink", width=1.6)
    svg.rect(bx - 48, 206, 96, 42, fill=col, fill_opacity=0.10, stroke=col, width=1.4, rx=6)
    svg.text(bx, 223, "H^{post}_{l}  1×4", size=13, anchor="middle")
    svg.text(bx, 240, "2σ(·) ∈ (0,2)" if mhc else "写入：分发", size=12, anchor="middle", color=col)
    for x in xs:
        svg.line(bx - 48, 246, x + 5, PLUS - 5, color="muted", width=1)
    if mhc:
        svg.text(x0 + 125, 420, "三个映射都由 *X*_{l} 算出，再投影到约束集", size=12, anchor="middle")
        svg.text(x0 + 125, 444, "复合映射仍双随机：增益 ≈ 1", size=12, anchor="middle", color="green")
    else:
        svg.text(x0 + 125, 420, "*X*_{l+1} = H^{res}*X*_{l} + (H^{post})^{⊤}𝓕(H^{pre}*X*_{l})",
                 size=12, anchor="middle")
        svg.text(x0 + 125, 444, "多层连乘 ∏ H^{res} 可以爆炸", size=12, anchor="middle", color="red")


panel_residual(8)
svg.line(250, 40, 250, 450, color="rule", width=1)
panel_hc(256, False)
svg.line(506, 40, 506, 450, color="rule", width=1)
panel_hc(510, True)

svg.save("l16-paradigms.svg")
