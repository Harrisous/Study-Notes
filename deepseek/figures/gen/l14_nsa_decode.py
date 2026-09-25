"""Figure 14.6: tokens loaded per decoding step, full attention vs NSA (NSA Table 4).

NSA loads at most floor((t - l_c)/delta_c) compressed tokens, n_s * l_s selected tokens and w
window tokens per step (t = cached length; the paper writes s); Table 4 writes the compressed
term as t/16. Expected speedup = ratio
of memory access (the paper's "Expected Speedup" row). Values recomputed here.
Writes deepseek/figures/l14-nsa-decode.svg
"""

from l14_svg import SVG

LENS = [8192, 16384, 32768, 65536]
LABELS = ["8K", "16K", "32K", "64K"]
CMP = [s // 16 for s in LENS]
SEL = [16 * 64] * 4
WIN = [512] * 4
NSA = [c + a + b for c, a, b in zip(CMP, SEL, WIN)]
assert NSA == [2048, 2560, 3584, 5632]
SPEED = [s / n for s, n in zip(LENS, NSA)]
PRINTED = ["4×", "6.4×", "9.1×", "11.6×"]      # as printed in NSA Table 4
assert [round(x, 1) for x in SPEED] == [4.0, 6.4, 9.1, 11.6]

W, H = 760, 400
svg = SVG(W, H, "NSA 每步解码要装入的 token 数与期望加速（据 NSA 表 4）")

# ---------------------------------------------------------------- left panel: full vs NSA
X0, X1, Y0, Y1 = 70, 390, 50, 320
YMAX = 70000


def py(v):
    return Y1 - v / YMAX * (Y1 - Y0)


svg.text(X0, 28, "每步装入的 token 数（线性刻度）", size=13, weight="700")
for v in range(0, 70001, 10000):
    y = py(v)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "%dK" % (v // 1000) if v else "0", size=11, anchor="end", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
gw = (X1 - X0) / 4
for i, (s, n) in enumerate(zip(LENS, NSA)):
    gx = X0 + i * gw + 12
    bw = 28
    svg.rect(gx, py(s), bw, Y1 - py(s), fill="muted", fill_opacity=0.35, stroke="muted", width=1)
    svg.rect(gx + bw + 6, py(n), bw, Y1 - py(n), fill="green", fill_opacity=0.7, stroke="green", width=1)
    svg.text(gx + bw / 2, py(s) - 6, "%d" % s, size=11, anchor="middle", color="muted")
    svg.text(gx + bw * 1.5 + 6, py(n) - 22, PRINTED[i], size=13, anchor="middle", weight="700", color="green")
    svg.text(gx + bw * 1.5 + 6, py(n) - 7, "%d" % n, size=11, anchor="middle", color="green")
    svg.text(gx + bw + 3, Y1 + 18, LABELS[i], size=12, anchor="middle")
svg.text((X0 + X1) / 2, Y1 + 40, "上下文长度（1K = 1024）", size=12, anchor="middle", color="muted")
svg.rect(X0 + 6, Y0 - 2, 12, 12, fill="muted", fill_opacity=0.35, stroke="muted", width=1)
svg.text(X0 + 24, Y0 + 8, "全注意力：装入全部 t 个", size=12, color="muted")
svg.rect(X0 + 6, Y0 + 18, 12, 12, fill="green", fill_opacity=0.7)
svg.text(X0 + 24, Y0 + 28, "NSA（绿色柱顶 = 期望加速）", size=12, color="green")

# ---------------------------------------------------------------- right panel: NSA composition
RX0, RX1, RY0, RY1 = 480, 730, 50, 320
RMAX = 6000


def ry(v):
    return RY1 - v / RMAX * (RY1 - RY0)


svg.text(RX0 - 30, 28, "NSA 的 token 从哪里来（线性刻度）", size=13, weight="700")
for v in range(0, 6001, 1000):
    y = ry(v)
    svg.line(RX0, y, RX1, y, color="grid", width=1)
    svg.text(RX0 - 8, y + 4, str(v), size=11, anchor="end", color="muted")
svg.line(RX0, RY1, RX1, RY1, color="muted", width=1.2)
rw = (RX1 - RX0) / 4
for i in range(4):
    x = RX0 + i * rw + 14
    bw = rw - 28
    base = 0
    for val, col in ((WIN[i], "green"), (SEL[i], "violet"), (CMP[i], "orange")):
        svg.rect(x, ry(base + val), bw, ry(base) - ry(base + val), fill=col, fill_opacity=0.6, stroke=col,
                 width=0.8)
        if val >= 900:
            svg.text(x + bw / 2, (ry(base) + ry(base + val)) / 2 + 4, str(val), size=11, anchor="middle")
        base += val
    svg.text(x + bw / 2, RY1 + 18, LABELS[i], size=12, anchor="middle")
# legend
for k, (lab, col) in enumerate((("压缩 t/16", "orange"), ("选择 16 × 64", "violet"), ("窗口 512", "green"))):
    svg.rect(RX0 + 4, RY0 + 2 + k * 18, 12, 12, fill=col, fill_opacity=0.6)
    svg.text(RX0 + 22, RY0 + 12 + k * 18, lab, size=12)
svg.text(RX0 - 30, RY1 + 40, "只有压缩项随长度增长：每 16 个 token 多 1 个", size=12, color="muted")

svg.text(20, 392, "期望加速 = 全注意力装入量 ÷ NSA 装入量，是按访存量推算的估计，不是实测延迟（论文表 4 行名即 Expected Speedup）。",
         size=11, color="muted")

svg.save("l14-nsa-decode.svg")
