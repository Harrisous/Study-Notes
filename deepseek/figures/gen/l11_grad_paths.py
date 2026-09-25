"""Figure 11.5: where the gradients go -- auxiliary balance loss vs routing bias (schematic).

Left: an f_i P_i loss is differentiable in the router scores s_i = sigma(u_t . e_i) (or softmax), so its
gradient reaches both the expert centroids e_k and the token representation u_t, and from u_t back into
attention and every earlier layer (derivation in lesson 11 sec. 3, eqs. 11.9-11.10).
Right: the bias b_i is updated from load counts only; it produces no gradient, so the only gradient in
the network is the language-model one.
Writes deepseek/figures/l11-grad-paths.svg
"""

from l11_svg import SVG, box

W, H = 720, 470
svg = SVG(W, H, "梯度往哪里流：辅助损失与偏置的对照")

BW, BH = 164, 40
rows = {"lm": 50, "later": 112, "moe": 174, "router": 236, "u": 298, "early": 360}


def panel(x0, title, title_col):
    xc = x0 + 20
    svg.text(x0, 24, title, size=13, weight="600", color=title_col)
    box(svg, xc, rows["lm"], BW, BH, "语言模型损失 ℒ_{LM}", color="green", fill_opacity=0.12, size=12)
    box(svg, xc, rows["later"], BW, BH, "后面的层 …", color="muted", fill_opacity=0.08, size=12)
    box(svg, xc, rows["moe"], BW, BH, "MoE 层：Σ g_{i}·FFN_{i}", color="blue", fill_opacity=0.10, size=12)
    box(svg, xc, rows["router"], BW, BH, "路由器：s_{i} = σ(*u*_{t}·*e*_{i})", color="violet", fill_opacity=0.12, size=12)
    box(svg, xc, rows["u"], BW, BH, "隐藏状态 *u*_{t}", color="ink", fill_opacity=0.06, size=12)
    box(svg, xc, rows["early"], BW, BH, "注意力与更早的层", color="muted", fill_opacity=0.08, size=12)
    # forward arrows (up), drawn on the centre line
    cx = xc + BW / 2
    for a, b in (("early", "u"), ("u", "router"), ("router", "moe"), ("moe", "later"), ("later", "lm")):
        svg.arrow(cx, rows[a], cx, rows[b] + BH + 1, color="muted", width=1.2)
    # LM gradient (down), on the left edge
    gx = xc + 14
    svg.path("M%s %s V %s" % (gx, rows["lm"] + BH, rows["early"] + BH + 22), stroke="green", width=2.2,
             dash="7 4", arrow=True)
    svg.text(gx - 6, rows["early"] + BH + 38, "ℒ_{LM} 的梯度流经每一层", size=11, color="green")
    return xc, cx


# ---------------------------------------------------------------- left: auxiliary loss
xc, cx = panel(16, "Ⓐ 辅助损失：α Σ f_{i}P_{i} 可导，梯度进入 *u*_{t}", "red")
ax = xc + BW + 22
box(svg, ax, rows["router"] - 4, 104, 48, "辅助损失", color="red", fill_opacity=0.10, size=12,
    sub="α Σ f_{i}P_{i}", sub_size=11)
svg.arrow(xc + BW + 1, rows["router"] + BH / 2 - 8, ax - 1, rows["router"] + BH / 2 - 8, color="muted", width=1.1)
# aux gradient back into router and into u_t
svg.arrow(ax - 1, rows["router"] + BH / 2 + 8, xc + BW + 1, rows["router"] + BH / 2 + 8, color="red", width=2)
rx = xc + BW - 30
svg.path("M%s %s H %s V %s" % (ax + 20, rows["router"] + 44, rx, rows["router"] + BH + 1), stroke="red", width=2,
         arrow=False)
svg.path("M%s %s V %s" % (rx, rows["router"] + BH + 2, rows["u"] - 1), stroke="red", width=2.2, arrow=True)
svg.path("M%s %s V %s" % (rx, rows["u"] + BH, rows["early"] - 1), stroke="red", width=2.2, arrow=True)
svg.text(ax + 2, rows["router"] + 66, "∂ℒ/∂*e*_{k}：改路由器", size=11, color="red")
svg.text(ax + 2, rows["u"] + 26, "∂ℒ/∂*u*_{t}：改表示", size=11, color="red", weight="600")
svg.text(ax + 2, rows["early"] + 22, "再传进注意力和", size=11, color="red")
svg.text(ax + 2, rows["early"] + 38, "更早的层 = “干扰梯度”", size=11, color="red", weight="600")

svg.line(360, 36, 360, 440, color="rule", width=1)

# ---------------------------------------------------------------- right: bias
xc2, cx2 = panel(376, "Ⓑ 偏置：只按计数更新，不产生梯度", "violet")
bx = xc2 + BW + 22
box(svg, bx, rows["moe"] - 4, 112, 48, "负载计数", color="violet", fill_opacity=0.08, size=12,
    sub="每专家几个 token", sub_size=11, dash="4 3")
box(svg, bx, rows["router"] + 36, 112, 48, "偏置 b_{i}", color="violet", fill_opacity=0.14, size=12,
    sub="b_{i} ∓ γ", sub_size=11)
svg.arrow(xc2 + BW + 1, rows["moe"] + BH / 2, bx - 1, rows["moe"] + BH / 2, color="violet", width=1.2, dash="4 3")
svg.arrow(bx + 56, rows["moe"] + 44, bx + 56, rows["router"] + 35, color="violet", width=1.4)
svg.arrow(bx - 1, rows["router"] + 60, xc2 + BW + 1, rows["router"] + BH / 2 + 6, color="violet", width=1.4)
svg.text(bx + 2, rows["router"] + 104, "只改“选谁”", size=11, color="violet")
svg.text(bx + 2, rows["router"] + 120, "不改权重，无梯度", size=11, color="violet", weight="600")

svg.text(16, 462, "绿色虚线：语言模型损失的梯度；红色实线：辅助损失的梯度；紫色：偏置的更新只用计数（下一步才生效）。", size=12,
         color="ink")

svg.save("l11-grad-paths.svg")
