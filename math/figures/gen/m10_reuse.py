"""Figure M10.3: why the intensity of Y = X W grows with the number of tokens m (data reuse).

Schematic of one weight matrix W (k x n, BF16) that is read from HBM once and used by
m tokens.  Left: m = 1 (matrix-vector, decode).  Right: a batch of m tokens (prefill).
One weight w_ij is highlighted with the column of X it multiplies and the column of Y it
feeds.  The numbers under each panel are computed here for a square k = n = 7168 weight
(7168 = DeepSeek-V3's hidden size from the released config; V3 itself has no square 7168 x 7168
projection, the square is used only to keep the arithmetic clean), with one read of X and W and
one write of Y, as in (M10.9).
Writes math/figures/m10-reuse.svg
"""

from m10_svg import SVG

S = 2          # bytes per BF16 number
K = N = 7168   # V3 hidden size


def bill(m):
    flops = 2 * m * K * N
    nbytes = S * (m * K + K * N + m * N)
    return flops, nbytes, flops / nbytes


W, H = 720, 388
svg = SVG(W, H, "矩阵乘法 Y = XW 的数据复用：同一份权重被 m 个 token 共用")

CELL = 11
NC = 7                 # cells drawn per matrix side (k, n)
HI_I, HI_J = 2, 4      # highlighted weight w_ij

panels = [
    (16, 1, "① m = 1：矩阵乘向量（decode）", "一个 token 乘一个 k × n 的权重", 1),
    (368, 4096, "② 一批 m 个 token 乘同一个权重（prefill）", "示意：画 6 行代表 m 行", 6),
]

for x0, m, title, sub, rows in panels:
    pw = 336
    svg.rect(x0, 8, pw, 324, stroke="rule", width=1.2, rx=8)
    svg.text(x0 + 14, 32, title, size=13, weight="600")
    svg.text(x0 + 14, 50, sub, size=12, color="muted")

    # HBM band with W stored in it
    svg.rect(x0 + 14, 62, pw - 28, 34, fill="amber", fill_opacity=0.10, stroke="amber", width=1, rx=4)
    svg.text(x0 + 24, 84, "显存 HBM", size=12, color="amber", weight="600")
    svg.text(x0 + pw - 24, 84, "W：k × n 个数，共 2kn 字节", size=12, anchor="end")

    # on-chip area
    svg.rect(x0 + 14, 122, pw - 28, 132, stroke="muted", width=1, rx=4, dash="4 3")
    svg.text(x0 + 24, 140, "片上 SRAM", size=12, color="muted")

    xX = x0 + 30
    xW = xX + NC * CELL + 22
    xY = xW + NC * CELL + 22
    top = 150

    # arrow: W is read from HBM exactly once
    wcx = xW + NC * CELL / 2
    svg.arrow(wcx, 97, wcx, top - 2, color="amber", width=1.8)
    svg.text(wcx + 8, 115, "只读一次", size=12, color="amber")

    # X (m x k): blue, highlighted column i
    for r in range(rows):
        for q in range(NC):
            hi = q == HI_I
            svg.rect(xX + q * CELL, top + r * CELL, CELL, CELL, fill="blue",
                     fill_opacity=0.75 if hi else 0.2, stroke="surface", width=1)
    # W (k x n): amber tint, highlighted w_ij
    for r in range(NC):
        for q in range(NC):
            hi = (r, q) == (HI_I, HI_J)
            svg.rect(xW + q * CELL, top + r * CELL, CELL, CELL, fill="amber",
                     fill_opacity=0.85 if hi else 0.2, stroke="surface", width=1)
    # Y (m x n): green, highlighted column j
    for r in range(rows):
        for q in range(NC):
            hi = q == HI_J
            svg.rect(xY + q * CELL, top + r * CELL, CELL, CELL, fill="green",
                     fill_opacity=0.75 if hi else 0.2, stroke="surface", width=1)

    xh = rows * CELL
    opy = top + min(xh, NC * CELL) / 2 + 5
    svg.text(xW - 11, opy, "×", size=15, anchor="middle")
    svg.text(xY - 11, opy, "=", size=15, anchor="middle")
    svg.text(xX + NC * CELL / 2, top + xh + 16, "X：%s × k" % ("1" if m == 1 else "m"), size=12,
             anchor="middle", color="blue")
    svg.text(wcx, top + NC * CELL + 16, "W：k × n", size=12, anchor="middle", color="amber")
    svg.text(xY + NC * CELL / 2, top + xh + 16, "Y：%s × n" % ("1" if m == 1 else "m"), size=12,
             anchor="middle", color="green")

    # the bill
    flops, nbytes, inten = bill(m)
    if m == 1:
        l1 = "运算 2kn 次；字节 ≈ 2kn（几乎全是 W）"
        l2 = "I ≈ 1：每个权重读进来只用 1 次"
        l3 = "k = n = 7168：读 %.1f MB，做 %.2f × 10^{8} 次，I = %.1f" % (nbytes / 1e6, flops / 1e8, inten)
    else:
        l1 = "运算 2mkn 次；字节 2kn + 2mk + 2mn"
        l2 = "I ≈ m：每个权重被 m 个 token 各用 1 次"
        l3 = "k = n = 7168，m = %d：读 %.1f MB，I = %.0f" % (m, nbytes / 1e6, inten)
    svg.text(x0 + 14, 276, l1, size=12)
    svg.text(x0 + 14, 298, l2, size=12, weight="600")
    svg.text(x0 + 14, 320, l3, size=12, color="muted")
    print(m, "flops %.4g bytes %.4g I %.2f" % (flops, nbytes, inten))

svg.text(16, 356, "高亮：W 里的一个权重 w_{ij}（琥珀）与 X 的第 i 列（蓝）逐个相乘，结果累加进 Y 的第 j 列（绿）。",
         size=12, color="muted")
svg.text(16, 376, "m 大到与 k、n 同量级后，X 与 Y 的字节不能再忽略，强度被最短的一维封顶，见 (M10.10) 与图 M10.4。",
         size=12, color="muted")

svg.save("m10-reuse.svg")
