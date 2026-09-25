"""Figure M1.6: multi-head attention as a block structure (column convention, one token).

Left: W^Q in R^{n_h d_h x d} is n_h row blocks W^Q_i in R^{d_h x d}; q_t = W^Q h_t stacks
q_{t,i} = W^Q_i h_t. Right: W^O in R^{d x n_h d_h} is n_h column blocks W^O_i in R^{d x d_h};
u_t = W^O [o_{t,1}; ...; o_{t,n_h}] = sum_i W^O_i o_{t,i}.
Drawn to scale for GPT-2 small (d = 768, n_h = 12, d_h = 64); head i = 3 is highlighted.
Writes math/figures/m01-multihead-blocks.svg
"""

from m01_svg import SVG

d, n_h = 768, 12
d_h = d // n_h
k = 0.22                         # pixels per dimension
D, DH = d * k, d_h * k           # 169, 14
HL = 2                           # 0-based index of the highlighted head (the 3rd)
VEC = 12                         # drawn width of a vector

W, H = 720, 366
svg = SVG(W, H, "多头注意力的分块结构：拆头是按行分块，合头是按列分块再相加")
top = 94

# ---------------------------------------------------------------- left: split heads
svg.text(20, 26, "① 拆头：*W*^{Q} 按行切成 12 块", size=14, weight="600", color="blue")
svg.text(20, 46, "每块 [d_{h}, d] = [64, 768]，一块对应一个头", size=12, color="muted")
x0 = 58
for i in range(n_h):
    y = top + i * DH
    strong = i == HL
    svg.rect(x0, y, D, DH, fill="blue", fill_opacity=0.34 if strong else (0.10 if i % 2 else 0.05),
             stroke="blue", width=0.8, stroke_opacity=0.5)
svg.rect(x0, top, D, D, stroke="blue", width=1.4)
svg.text(x0 + D / 2, top + D + 20, "*W*^{Q}  [n_{h}d_{h}, d] = [768, 768]", size=12, anchor="middle", color="blue")
# leader to the highlighted block
yh = top + HL * DH + DH / 2
svg.text(x0 - 8, yh + 4, "*W*^{Q}_{3}", size=12, anchor="end", color="blue", weight="700")

svg.text(x0 + D + 16, top + D / 2 + 6, "×", size=18, anchor="middle", color="muted")
xh = x0 + D + 30
svg.rect(xh, top, VEC, D, fill="ink", fill_opacity=0.08, stroke="ink", width=1.2)
svg.text(xh + VEC / 2, top + D + 20, "*h*_{t}", size=13, anchor="middle")
svg.text(xh + VEC / 2, top + D + 36, "[768]", size=12, anchor="middle", color="muted")

svg.text(xh + VEC + 16, top + D / 2 + 6, "=", size=18, anchor="middle", color="muted")
xq = xh + VEC + 30
for i in range(n_h):
    y = top + i * DH
    strong = i == HL
    svg.rect(xq, y, VEC, DH, fill="blue", fill_opacity=0.34 if strong else (0.10 if i % 2 else 0.05),
             stroke="blue", width=0.8, stroke_opacity=0.5)
svg.rect(xq, top, VEC, D, stroke="blue", width=1.4)
svg.text(xq + VEC / 2, top + D + 20, "*q*_{t}", size=13, anchor="middle", color="blue")
svg.text(xq + VEC / 2, top + D + 36, "[768]", size=12, anchor="middle", color="muted")
svg.line(xq + VEC + 4, yh, xq + VEC + 14, yh, color="blue", width=1)
svg.text(xq + VEC + 17, yh + 4, "*q*_{t,3}", size=12, color="blue", weight="700")
svg.text(xq + VEC + 17, yh + 21, "[64]", size=12, color="muted")

svg.text(20, top + D + 70, "*q*_{t,3} = *W*^{Q}_{3}*h*_{t}：拆头不做任何运算，", size=13, color="blue")
svg.text(20, top + D + 92, "第 3 个头的查询就是大向量 *q*_{t} 的第 129–192 个分量。", size=13)

# ---------------------------------------------------------------- right: merge heads
svg.line(380, 16, 380, H - 16, color="rule", width=1)
svg.text(396, 26, "② 合头：*W*^{O} 按列切成 12 块", size=14, weight="600", color="green")
svg.text(396, 46, "每块 [d, d_{h}] = [768, 64]", size=12, color="muted")
x1 = 400
for i in range(n_h):
    x = x1 + i * DH
    strong = i == HL
    svg.rect(x, top, DH, D, fill="ink", fill_opacity=0.22 if strong else (0.08 if i % 2 else 0.03),
             stroke="ink", width=0.8, stroke_opacity=0.4)
svg.rect(x1, top, D, D, stroke="ink", width=1.4)
xc = x1 + HL * DH + DH / 2
svg.line(xc, top - 4, xc, top - 14, color="ink", width=1)
svg.text(xc, top - 18, "*W*^{O}_{3}", size=12, anchor="middle", weight="700")
svg.text(x1 + D / 2, top + D + 20, "*W*^{O}  [d, n_{h}d_{h}] = [768, 768]", size=12, anchor="middle")

svg.text(x1 + D + 16, top + D / 2 + 6, "×", size=18, anchor="middle", color="muted")
xo = x1 + D + 30
for i in range(n_h):
    y = top + i * DH
    strong = i == HL
    svg.rect(xo, y, VEC, DH, fill="green", fill_opacity=0.34 if strong else (0.10 if i % 2 else 0.05),
             stroke="green", width=0.8, stroke_opacity=0.5)
svg.rect(xo, top, VEC, D, stroke="green", width=1.4)
svg.text(xo + VEC / 2, top + D + 20, "拼接", size=12, anchor="middle", color="green")
svg.text(xo + VEC / 2, top + D + 36, "[768]", size=12, anchor="middle", color="muted")
yo = top + HL * DH + DH / 2
svg.line(xo + VEC + 4, yo, xo + VEC + 16, yo, color="green", width=1)
svg.text(xo + VEC + 20, yo + 4, "*o*_{t,3}", size=12, color="green", weight="700")

xu = xo + VEC + 78
svg.text(xu - 16, top + D / 2 + 6, "=", size=18, anchor="middle", color="muted")
svg.rect(xu, top, VEC, D, fill="green", fill_opacity=0.14, stroke="green", width=1.4)
svg.text(xu + VEC / 2, top + D + 20, "*u*_{t}", size=13, anchor="middle", color="green")
svg.text(xu + VEC / 2, top + D + 36, "[768]", size=12, anchor="middle", color="muted")

svg.text(396, top + D + 70, "*u*_{t} = Σ_{i} *W*^{O}_{i} *o*_{t,i}：拼接后乘 *W*^{O}，", size=13, color="green")
svg.text(396, top + D + 92, "等于每个头各乘自己那一块，再相加。", size=13)

svg.save("m01-multihead-blocks.svg")
