"""Figure 9.1: per-sequence KV cache (bf16) vs context length, four attention variants of V2's shape.

All four variants use DeepSeek-V2's attention shape: l = 60 layers, n_h = 128 heads, d_h = 128
(arXiv:2405.04434v5 sec. 3.1.2); MLA adds d_c = 512, d_h^R = 64.  Elements per token:
  MHA   2 n_h d_h l      = 1,966,080
  GQA-8 2 * 8 * d_h l    =   122,880
  MQA   2 d_h l          =    15,360
  MLA   (d_c + d_h^R) l  =    34,560   (Table 1 of the paper)
bf16 = 2 bytes per element.  GB = 1e9 bytes.  The 80 GB reference line is the HBM of an
A100/H100 80GB card (lesson-04-doc sec. 3.2).
Writes deepseek/figures/l09-cache-growth.svg
"""

from l09_svg import SVG, LogAxis, fmt

l, n_h, d_h, d_c, d_r = 60, 128, 128, 512, 64
variants = [
    ("MHA（每头独立 K、V）", 2 * n_h * d_h * l, "ink", None, 2.2),
    ("GQA，8 组", 2 * 8 * d_h * l, "muted", None, 2.0),
    ("MLA（本课）", (d_c + d_r) * l, "orange", None, 3.0),
    ("MQA，1 组", 2 * d_h * l, "muted", "6 4", 2.0),
]
assert variants[0][1] == 1_966_080 and variants[2][1] == 34_560

W, H = 720, 400
svg = SVG(W, H, "单条序列的 KV cache 随上下文长度增长：DeepSeek-V2 的形状，四种注意力")
X0, X1, Y0, Y1 = 78, 540, 24, 336
xs = LogAxis(1024, 131072, X0, X1)
ys = LogAxis(0.01, 1000, Y1, Y0)

# grid + ticks
for e in range(-2, 4):
    v = 10 ** e
    y = ys(v)
    svg.line(X0, y, X1, y, color="grid", width=1)
    lab = {-2: "0.01", -1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1000"}[e]
    svg.text(X0 - 8, y + 4, lab, size=11, anchor="end", color="muted")
for k in range(10, 18):
    v = 2 ** k
    x = xs(v)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, "%dK" % (v // 1024), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="rule", width=1.2)
svg.line(X0, Y0, X0, Y1, color="rule", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "上下文长度 T（token，对数刻度）", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "每条序列的 KV cache（GB，bf16，对数刻度）", size=12, anchor="middle",
         color="muted", rotate=-90)

# 80 GB reference
y80 = ys(80)
svg.line(X0, y80, X1, y80, color="red", width=1.3, dash="5 4")
svg.text(X0 + 8, y80 - 7, "一张 80 GB 显存 GPU 的容量", size=12, color="red")

# lines
L128 = 131072
for name, e, col, dash, wdt in variants:
    pts = []
    for k in range(10, 18):
        L = 2 ** k
        pts.append((xs(L), ys(e * 2 * L / 1e9)))
    svg.polyline(pts, stroke=col, width=wdt, dash=dash)
    gb = e * 2 * L128 / 1e9
    svg.circle(xs(L128), ys(gb), 3.5, fill=col)

yl = []
for name, e, col, dash, wdt in variants:
    gb = e * 2 * L128 / 1e9
    txt = "%.1f GB" % gb if gb >= 10 else "%.2f GB" % gb
    yl.append((ys(gb), name, txt, col, e))
# spread labels vertically if closer than 26 px
yl.sort()
for i in range(1, len(yl)):
    if yl[i][0] - yl[i - 1][0] < 34:
        yl[i] = (yl[i - 1][0] + 34,) + yl[i][1:]
for y, name, txt, col, e in yl:
    tc = "ink" if col == "muted" else col
    svg.text(X1 + 12, y - 2, name, size=12, color=tc, weight="600")
    svg.text(X1 + 12, y + 13, "128K 时 %s" % txt, size=11, color="muted")

svg.save("l09-cache-growth.svg")
for name, e, *_ in variants:
    print(name, e, e * 2, round(e * 2 * L128 / 1e9, 3))
