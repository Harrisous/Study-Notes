"""Figure 6.9: what 6N1, 6N2 and M each count, for three rows of DeepSeek LLM Table 3 (computed).

Per token, training (forward + backward):
  parameter matmuls 72 L d^2, attention scores/weighted sum 12 L d T, output head 6 V d.
  M = 72Ld^2 + 12LdT;  6N1 = 72Ld^2;  6N2 = 72Ld^2 + 6Vd.     V = 102400, T = 4096 (dsllm eq. 2).
Bars are normalised so that M = 100 % for each model.
Writes deepseek/figures/l06-flops-breakdown.svg
"""

from l06_svg import SVG, sci

V, T = 102400, 4096
rows = [(8, 512), (24, 2048), (80, 8192)]

W, H = 760, 470
X0, X1 = 110, 610
XMAX = 1.4


def px(f):
    return X0 + f / XMAX * (X1 - X0)


svg = SVG(W, H, "三种记账方式各自数了哪些 FLOPs（以 M 为 100%）")


def label_x(f):
    """Value label just right of the bar end; a label that would straddle the dashed M = 100% line
    (bar ends within ~9% below it) moves to the right of that line instead."""
    return px(1) + 6 if 0.91 <= f < 1 else px(f) + 6

top = 64
for k in range(0, 15, 2):
    f = k / 10
    x = px(f)
    svg.line(x, top - 24, x, 408, color="grid", width=1)
    svg.text(x, 426, "%d%%" % round(f * 100), size=11, anchor="middle", color="muted")
svg.line(px(1), top - 30, px(1), 408, color="ink", width=1.2, dash="4 3")
svg.text(px(1), top - 36, "M = 100%", size=12, anchor="middle", color="ink")
svg.text((X0 + X1) / 2, 450, "占 M 的比例（每个模型各自归一）", size=12, anchor="middle", color="muted")

y = top
for L, d in rows:
    a = 72 * L * d * d
    att = 12 * L * d * T
    voc = 6 * V * d
    M = a + att
    fa, fatt, fv = a / M, att / M, voc / M
    svg.text(X0 - 40, y - 8, "L = %d，d = %d" % (L, d), size=13, weight="600")
    svg.text(X0 + 110, y - 8, "M = %s FLOPs/token" % sci(M), size=12, color="muted")
    bh = 20
    # M row
    yy = y
    svg.text(X0 - 8, yy + 15, "M", size=12, anchor="end", color="ink", weight="600")
    svg.rect(px(0), yy, px(fa) - px(0), bh, fill="blue", fill_opacity=0.45, stroke="blue", width=1)
    svg.rect(px(fa), yy, px(fa + fatt) - px(fa), bh, fill="violet", fill_opacity=0.45, stroke="violet", width=1)
    # 6N1 row
    yy = y + 28
    svg.text(X0 - 8, yy + 15, "6N_{1}", size=12, anchor="end", color="ink", weight="600")
    svg.rect(px(0), yy, px(fa) - px(0), bh, fill="blue", fill_opacity=0.45, stroke="blue", width=1)
    svg.text(label_x(fa), yy + 15, "%.2f" % fa, size=12, color="blue", weight="600")
    # 6N2 row
    yy = y + 56
    svg.text(X0 - 8, yy + 15, "6N_{2}", size=12, anchor="end", color="ink", weight="600")
    svg.rect(px(0), yy, px(fa) - px(0), bh, fill="blue", fill_opacity=0.45, stroke="blue", width=1)
    svg.rect(px(fa), yy, px(fa + fv) - px(fa), bh, fill="muted", fill_opacity=0.35, stroke="muted", width=1)
    svg.text(label_x(fa + fv), yy + 15, "%.2f" % (fa + fv), size=12, color="muted", weight="600")
    y += 116

# legend
ly = 408 - 2
lx = X1 + 20
items = [("blue", "参数矩阵乘", "72Ld^{2}"), ("violet", "注意力打分与", "加权求和 12LdT"), ("muted", "输出头（词表）", "6Vd")]
for i, (col, s1, s2) in enumerate(items):
    yy = 70 + i * 58
    svg.rect(lx, yy, 16, 16, fill=col, fill_opacity=0.45, stroke=col, width=1)
    svg.text(lx + 22, yy + 12, s1, size=12, color=col)
    svg.text(lx + 22, yy + 29, s2, size=12, color=col)

svg.save("l06-flops-breakdown.svg")
