"""Figure M6.4: the same chain of Jacobians multiplied in two orders.

Chain x -> h1 -> h2 -> h3 -> L with d = 768 everywhere (GPT-2 small width). The gradient of the
scalar loss w.r.t. x is g^T J3 J2 J1, where g^T = dL/dh3 is a [1, d] row. Reverse mode multiplies
from the left, so every intermediate result stays a [1, d] row (vector-Jacobian products);
forward mode multiplies from the right and carries full [d, d] matrices. FLOP counts use 2mnk
(M01) and are computed here. Block sizes are drawn to scale in the d direction.
Writes math/figures/m06-jacobian-order.svg
"""

from m06_svg import SVG, sci

d = 768
vjp = 2 * 1 * d * d                  # [1,d] x [d,d]
mm = 2 * d * d * d                   # [d,d] x [d,d]
rev_total = 3 * vjp
fwd_total = 2 * mm + vjp
print("vjp", vjp, "mm", mm, "rev", rev_total, "fwd", fwd_total, "ratio", fwd_total / rev_total)

W, H = 720, 470
svg = SVG(W, H, "同一串雅可比，两种乘法顺序：反向模式与前向模式")
S = 64          # px for 768
T = 7           # px thickness drawn for the "1" side of a row vector (not to scale, see caption)

# chain on top
y0 = 36
labels = ["x", "h_{1}", "h_{2}", "h_{3}", "𝓛"]
xs = [70, 210, 350, 490, 630]
for i, (x, lab) in enumerate(zip(xs, labels)):
    svg.circle(x, y0 + 10, 17, fill="blue" if i < 4 else "red", fill_opacity=0.12,
               stroke="blue" if i < 4 else "red", width=1.3)
    svg.text(x, y0 + 15, lab, size=14, anchor="middle", weight="600")
    if i < 4:
        svg.arrow(x + 19, y0 + 10, xs[i + 1] - 19, y0 + 10, color="ink", width=1.4)
        tag = "J_{%d}" % (i + 1) if i < 3 else "标量损失"
        svg.text((x + xs[i + 1]) / 2, y0 + 1, tag, size=13, anchor="middle", color="muted")
svg.text(70, y0 + 44, "[768]", size=11, anchor="middle", color="muted")
for x in xs[1:4]:
    svg.text(x, y0 + 44, "[768]", size=11, anchor="middle", color="muted")
svg.text(630, y0 + 44, "[1]", size=11, anchor="middle", color="muted")
svg.text(W / 2, y0 + 70, "(∇_{x}𝓛)ᵀ = gᵀ J_{3} J_{2} J_{1}，其中 gᵀ = ∂𝓛/∂h_{3} 是 [1, 768] 的行向量；每个 J 是 [768, 768]",
         size=12, anchor="middle", color="ink")


def rowvec(x, y, lab, col="red"):
    svg.rect(x, y + (S - T) / 2, S, T, fill=col, fill_opacity=0.55, stroke=col, width=1)
    svg.text(x + S / 2, y + (S - T) / 2 - 6, lab, size=12, anchor="middle", color=col)


def square(x, y, lab, col="blue"):
    svg.rect(x, y, S, S, fill=col, fill_opacity=0.12, stroke=col, width=1.2)
    svg.text(x + S / 2, y + S / 2 + 5, lab, size=13, anchor="middle", color=col, weight="600")


def op(x, y, s):
    svg.text(x, y + S / 2 + 5, s, size=16, anchor="middle", color="muted")


# reverse mode row
yr = 150
svg.text(24, yr - 14, "反向模式：从左往右乘，中间结果始终是一行（向量–雅可比积）", size=13, weight="600", color="red")
x = 24
rowvec(x, yr, "gᵀ [1,768]"); x += S + 14
op(x - 7, yr, "×"); square(x, yr, "J_{3}"); x += S + 22
op(x - 11, yr, "→"); rowvec(x, yr, "[1,768]"); x += S + 14
op(x - 7, yr, "×"); square(x, yr, "J_{2}"); x += S + 22
op(x - 11, yr, "→"); rowvec(x, yr, "[1,768]"); x += S + 14
op(x - 7, yr, "×"); square(x, yr, "J_{1}"); x += S + 22
op(x - 11, yr, "→"); rowvec(x, yr, "(∇_{x}𝓛)ᵀ [1,768]")
svg.text(24, yr + S + 22, "每一步 [1,768]×[768,768]：2·768² = %s 次运算；三步合计 %s" % (sci(vjp), sci(rev_total)),
         size=12, color="red")

# forward mode row
yf = 300
svg.text(24, yf - 14, "前向模式：从右往左乘，要搬运整块 [768,768] 的矩阵（雅可比–矩阵积）", size=13, weight="600", color="blue")
x = 24
square(x, yf, "J_{2}"); x += S + 14
op(x - 7, yf, "×"); square(x, yf, "J_{1}"); x += S + 22
op(x - 11, yf, "→"); square(x, yf, "J_{2}J_{1}"); x += S + 30
op(x - 15, yf, "；")
square(x, yf, "J_{3}"); x += S + 14
op(x - 7, yf, "×"); square(x, yf, "J_{2}J_{1}"); x += S + 22
op(x - 11, yf, "→"); square(x, yf, "J_{3}J_{2}J_{1}")
svg.text(x + S + 10, yf + S / 2 - 4, "最后才乘", size=12, color="muted")
svg.text(x + S + 10, yf + S / 2 + 14, "gᵀ", size=12, color="muted")
svg.text(24, yf + S + 22, "每步 [768,768]×[768,768]：2·768³ = %s 次运算；合计 %s，是反向模式的 %d 倍"
         % (sci(mm), sci(fwd_total), round(fwd_total / rev_total)), size=12, color="blue")

svg.text(24, H - 32, "方块边长按 768 等比例画；行向量的“1”那一边为了看得见画粗了。", size=12, color="muted")
svg.text(24, H - 12, "输入换成 1.24 亿个参数时，差距变成“一次反向传播 vs 约 1.24 亿次前向”。", size=12, color="muted")

svg.save("m06-jacobian-order.svg")
