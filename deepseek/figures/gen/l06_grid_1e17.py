"""Figure 6.11: the 1e17-FLOPs batch-size x learning-rate grid (redrawn from dsllm Fig. 2a, values as printed).

Model: 177M FLOPs/token (so D = 1e17 / 177e6 = 5.65e8 tokens). Near-optimal = loss within 0.25 % of the
minimum (dsllm §3.1), i.e. <= 3.685 * 1.0025 = 3.6942 -> 5 cells. The cross marks what eq. (1) predicts
for C = 1e17: log2 B = 16.70, log2 eta = -8.74.
Writes deepseek/figures/l06-grid-1e17.svg
"""

import math

from l06_svg import SVG

rows_log2B = [15, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20]
cols_log2lr = [-10.5 + 0.25 * k for k in range(12)]
data = """
3.872 3.872 3.886 3.893 3.900 3.877 3.891 3.890 3.878 3.902 3.912 3.918
3.787 3.781 3.794 3.775 3.782 3.759 3.755 3.784 3.750 3.747 3.761 3.805
3.757 3.740 3.711 3.712 3.741 3.733 3.720 3.715 3.708 3.707 3.693 3.781
3.747 3.744 3.734 3.724 3.712 3.695 3.698 3.693 3.690 3.753 3.685 3.744
3.779 3.740 3.741 3.726 3.701 3.709 3.709 3.704 3.746 3.754 3.743 3.764
3.805 3.770 3.769 3.739 3.734 3.730 3.718 3.691 3.785 3.775 3.788 3.809
3.880 3.867 3.839 3.800 3.796 3.786 3.910 3.852 3.835 3.845 3.862 3.808
4.136 4.071 3.999 3.964 3.957 4.047 3.960 3.970 3.976 3.920 3.956 3.960
4.653 4.456 4.764 4.513 4.347 4.197 4.281 4.222 4.156 4.111 4.078 4.166
4.958 5.152 5.098 5.013 4.964 5.001 4.965 4.770 4.834 4.715 4.741 4.614
"""
grid = [[float(v) for v in line.split()] for line in data.strip().splitlines()]
assert len(grid) == 10 and all(len(r) == 12 for r in grid)
lo = min(min(r) for r in grid)
thr = lo * 1.0025
near = [(i, j) for i in range(10) for j in range(12) if grid[i][j] <= thr]
assert abs(lo - 3.685) < 1e-9 and len(near) == 5, near

W, H = 760, 360
CX0, CY0 = 96, 46
CW, CH = 40, 28
svg = SVG(W, H, "1e17 FLOPs 预算下 batch size × 学习率的训练损失网格")

for i, row in enumerate(grid):
    for j, v in enumerate(row):
        x, y = CX0 + j * CW, CY0 + i * CH
        if v <= thr:
            svg.rect(x + 1, y + 1, CW - 2, CH - 2, fill="green", fill_opacity=0.55, stroke="green", width=2)
            col, wt = "ink", "700"
        elif v <= 3.80:
            t = (v - thr) / (3.80 - thr)
            svg.rect(x + 1, y + 1, CW - 2, CH - 2, fill="green", fill_opacity=0.32 * (1 - t) + 0.04)
            col, wt = "ink", None
        else:
            t = min(1.0, (v - 3.80) / (5.0 - 3.80))
            svg.rect(x + 1, y + 1, CW - 2, CH - 2, fill="red", fill_opacity=0.06 + 0.30 * t)
            col, wt = "ink", None
        svg.text(x + CW / 2, y + CH / 2 + 4, "%.3f" % v, size=11, anchor="middle", color=col, weight=wt)

# axes labels
for i, b in enumerate(rows_log2B):
    lab = "2^{%s}" % (("%g" % b))
    svg.text(CX0 - 8, CY0 + i * CH + CH / 2 + 4, lab, size=11, anchor="end", color="muted")
for j, lr in enumerate(cols_log2lr):
    svg.text(CX0 + j * CW + CW / 2, CY0 - 8, "%g" % lr, size=11, anchor="middle", color="muted")
svg.text(CX0 + 6 * CW, CY0 - 28, "学习率 η = 2^{k}，表头为 k", size=12, anchor="middle", color="muted")
svg.text(28, CY0 + 5 * CH, "batch size（token）", size=12, anchor="middle", color="muted", rotate=-90)

# predicted point from eq. (1) at C = 1e17
C = 1e17
lb = math.log2(0.2920 * C ** 0.3271)
le = math.log2(0.3118 * C ** -0.1250)


def row_y(l2b):
    for k in range(len(rows_log2B) - 1):
        a, b = rows_log2B[k], rows_log2B[k + 1]
        if a <= l2b <= b:
            return CY0 + (k + 0.5 + (l2b - a) / (b - a)) * CH
    raise ValueError


def col_x(l2e):
    return CX0 + ((l2e - cols_log2lr[0]) / 0.25 + 0.5) * CW


xp, yp = col_x(le), row_y(lb)
svg.line(xp - 9, yp - 9, xp + 9, yp + 9, color="blue", width=2.6)
svg.line(xp - 9, yp + 9, xp + 9, yp - 9, color="blue", width=2.6)

# legend / notes on the right
nx = CX0 + 12 * CW + 16
svg.rect(nx, 50, 18, 14, fill="green", fill_opacity=0.55, stroke="green", width=2)
svg.text(nx + 26, 62, "近优格：", size=12)
svg.text(nx + 26, 80, "≤ 最小值 × 1.0025", size=12)
svg.text(nx + 26, 98, "（阈值 %.4f，%d 格）" % (thr, len(near)), size=12, color="muted")
svg.rect(nx, 114, 18, 14, fill="green", fill_opacity=0.2)
svg.text(nx + 26, 126, "3.80 以下，越深越低", size=12)
svg.rect(nx, 142, 18, 14, fill="red", fill_opacity=0.3)
svg.text(nx + 26, 154, "3.80 以上，越红越高", size=12)
svg.line(nx + 2, 174, nx + 16, 188, color="blue", width=2.6)
svg.line(nx + 2, 188, nx + 16, 174, color="blue", width=2.6)
svg.text(nx + 26, 186, "式 (6.12) 的预测：", size=12, color="blue")
svg.text(nx + 26, 204, "B = 2^{%.2f}" % lb, size=12, color="blue")
svg.text(nx + 26, 222, "η = 2^{%.2f}" % le, size=12, color="blue")
svg.text(nx, 256, "最小值 3.685 在", size=12)
svg.text(nx, 274, "B = 2^{17}，η = 2^{-8}", size=12)
svg.text(nx, 306, "每格训练 5.65×10^{8}", size=12, color="muted")
svg.text(nx, 324, "个 token；B = 2^{20}", size=12, color="muted")
svg.text(nx, 342, "时只有约 539 步", size=12, color="muted")

svg.save("l06-grid-1e17.svg")
