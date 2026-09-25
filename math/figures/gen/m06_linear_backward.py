"""Figure M6.8: the three matrix products of a linear layer Y = XW, drawn to scale.

Shapes from GPT-2 small's FFN up-projection: n = 768 inputs, m = 3072 outputs, a batch of
B = 1024 tokens. Forward Y = XW; backward dX = dY W^T and dW = X^T dY. Rectangles are drawn with
side lengths proportional to the dimensions (1024 -> 40 px). FLOPs = 2 * (product of the three
dimensions) are computed here.
Writes math/figures/m06-linear-backward.svg
"""

from m06_svg import SVG, sci

B, n, m = 1024, 768, 3072
k = 40 / 1024.0
fl = 2 * B * n * m
print("flops per product", fl)

W_, H_ = 720, 480
svg = SVG(W_, H_, "线性层的三次矩阵乘法（GPT-2 small 的 FFN 上投影，按比例）")


def block(x, y, rows, cols, lab, shape, col, fill_op=0.14, shape_below=True):
    w, h = cols * k, rows * k
    svg.rect(x, y, w, h, fill=col, fill_opacity=fill_op, stroke=col, width=1.3)
    svg.text(x + w / 2, y + h / 2 + 5, lab, size=13, anchor="middle", color=col, weight="600")
    if shape_below:
        svg.text(x + w / 2, y + h + 15, shape, size=11, anchor="middle", color="muted")
    return w, h


def op(x, y, s):
    svg.text(x, y, s, size=16, anchor="middle", color="muted")


rows_y = [58, 196, 392]
titles = [("前向：Y = X W", "blue"), ("反向 ①：dX = dY Wᵀ（继续往回传）", "red"),
          ("反向 ②：dW = Xᵀ dY（用来更新 W）", "red")]
for (t, c), yy in zip(titles, rows_y):
    svg.text(20, yy - 16, t, size=13, weight="600", color=c)

# row 1: X [B,n] x W [n,m] = Y [B,m]
y0 = rows_y[0]
x = 20
w, h = block(x, y0, B, n, "X", "[1024, 768]", "blue"); x += w + 24
op(x - 12, y0 + h / 2 + 5, "×")
w2, h2 = block(x, y0, n, m, "W", "[768, 3072]", "ink", 0.06); x += w2 + 24
op(x - 12, y0 + h / 2 + 5, "=")
block(x, y0, B, m, "Y", "[1024, 3072]", "blue")

# row 2: dY [B,m] x W^T [m,n] = dX [B,n]
y1 = rows_y[1]
x = 20
w, h = block(x, y1, B, m, "dY", "[1024, 3072]", "red"); x += w + 24
op(x - 12, y1 + h / 2 + 5, "×")
wt, ht = block(x, y1, m, n, "Wᵀ", "[3072, 768]", "ink", 0.06); x += wt + 24
op(x - 12, y1 + h / 2 + 5, "=")
block(x, y1, B, n, "dX", "[1024, 768]", "red")

# row 3: X^T [n,B] x dY [B,m] = dW [n,m]
y2 = rows_y[2]
x = 20
w, h = block(x, y2, n, B, "Xᵀ", "[768, 1024]", "blue", 0.08); x += w + 24
op(x - 12, y2 + 20, "×")
w3, h3 = block(x, y2, B, m, "dY", "[1024, 3072]", "red"); x += w3 + 24
op(x - 12, y2 + 20, "=")
block(x, y2, n, m, "dW", "[768, 3072]", "red")

# right-hand ledger
RX = 470
yy = 60
svg.text(RX, yy, "每次乘法的运算量", size=13, weight="600"); yy += 22
svg.text(RX, yy, "2 × 1024 × 768 × 3072 = %s" % sci(fl), size=12); yy += 28
svg.text(RX, yy, "前向 1 次：%s FLOPs" % sci(fl), size=12, color="blue"); yy += 20
svg.text(RX, yy, "反向 2 次：%s FLOPs" % sci(2 * fl), size=12, color="red"); yy += 20
svg.text(RX, yy, "合计 %s = 6 × 参数量 × token 数" % sci(3 * fl), size=12); yy += 34
svg.text(RX, yy, "形状自检：dW 必须与 W 同为", size=12, color="muted"); yy += 18
svg.text(RX, yy, "[768, 3072]；手里只有 X [1024, 768]", size=12, color="muted"); yy += 18
svg.text(RX, yy, "与 dY [1024, 3072]，唯一对得上的", size=12, color="muted"); yy += 18
svg.text(RX, yy, "乘法是 Xᵀ dY", size=12, color="muted"); yy += 34
svg.text(RX, yy, "dW 对 1024 个 token 求和：", size=12, color="muted"); yy += 18
svg.text(RX, yy, "每个 token 贡献一个外积 x·(dy)ᵀ", size=12, color="muted"); yy += 18
svg.text(RX, yy, "（同一个 W 被每个 token 用了一次）", size=12, color="muted")
svg.text(20, H_ - 10, "方块边长按维数等比例画（1024 ↔ 40 像素）。", size=12, color="muted")

svg.save("m06-linear-backward.svg")
