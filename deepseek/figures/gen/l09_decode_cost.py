"""Figure 9.7: one decoding step of one MLA layer, three ways to compute it, vs cached length L.

DeepSeek-V2 dims n_h = d_h = 128, d_h^R = 64, d_c = 512 (arXiv:2405.04434v5 sec. 3.1.2).
FLOPs per layer per decoded token (lesson-09-doc sec. 4.4):
  (1) re-expand every step : 33,636,352 L
  (2) cache expanded K, V  : 33,554,432 + 81,920 L   (MHA-style cache: 128 heads x (192 + 128) numbers;
                             the fixed term expands the new token's own k, v once, 2*512*2*128*128)
  (3) absorb (MQA mode)    : 33,554,432 + 278,528 L
Cache bytes read per layer per decoded token (bf16): (1) and (3) read the 576-dim latent row,
1,152 B per token; (2) reads 128 x (192 + 128) numbers, 81,920 B per token.
Writes deepseek/figures/l09-decode-cost.svg
"""

from l09_svg import SVG, LogAxis

Ls = [2 ** k for k in range(10, 18)]
f1 = lambda L: 33_636_352 * L
f2 = lambda L: 33_554_432 + 81_920 * L
f3 = lambda L: 33_554_432 + 278_528 * L
b13 = lambda L: 1_152 * L
b2 = lambda L: 81_920 * L
assert abs(f1(4096) / 1e9 - 137.77) < 0.01 and abs(f3(4096) / 1e9 - 1.1744) < 1e-3
assert abs(f2(4096) / 1e9 - 0.3691) < 1e-3 and f3(4096) - f2(4096) == 196_608 * 4096

W, H = 760, 380
svg = SVG(W, H, "解码一步的算力与读缓存字节，三种算法")


def panel(x0, title, ylab, ymin, ymax, yticks, series, unit_div, notes):
    X0, X1, Y0, Y1 = x0 + 56, x0 + 330, 44, 300
    xs = LogAxis(1024, 131072, X0, X1)
    ys = LogAxis(ymin, ymax, Y1, Y0)
    svg.text(x0 + 8, 20, title, size=13.5, weight="700")
    for v, lab in yticks:
        y = ys(v)
        svg.line(X0, y, X1, y, color="grid", width=1)
        svg.text(X0 - 6, y + 4, lab, size=11, anchor="end", color="muted")
    for k in range(10, 18):
        L = 2 ** k
        x = xs(L)
        svg.line(x, Y0, x, Y1, color="grid", width=1)
        if k % 1 == 0:
            svg.text(x, Y1 + 16, "%dK" % (L // 1024), size=11, anchor="middle", color="muted")
    svg.line(X0, Y1, X1, Y1, color="rule", width=1.2)
    svg.line(X0, Y0, X0, Y1, color="rule", width=1.2)
    svg.text((X0 + X1) / 2, Y1 + 36, "缓存里的 token 数 T（对数刻度）", size=12, anchor="middle", color="muted")
    svg.text(x0 + 14, (Y0 + Y1) / 2, ylab, size=12, anchor="middle", color="muted", rotate=-90)
    for fn, col, dash, wdt, lab, dy in series:
        pts = [(xs(L), ys(fn(L) / unit_div)) for L in Ls]
        svg.polyline(pts, stroke=col, width=wdt, dash=dash)
        xe, ye = pts[-1]
        svg.circle(xe, ye, 3.2, fill=col)
        idx, off, anc = dy
        svg.text(pts[idx][0] + (-4 if anc == "end" else 6), pts[idx][1] + off, lab, size=12, color=col,
                 weight="600", anchor=anc)
    for i, (txt, col) in enumerate(notes):
        svg.text(X0 + 4, Y0 + 14 + i * 17, txt, size=11.5, color=col)


panel(0, "每层每步的运算量（GFLOPs）", "GFLOPs（对数刻度）", 0.01, 10000,
      [(0.01, "0.01"), (0.1, "0.1"), (1, "1"), (10, "10"), (100, "100"), (1000, "1000"), (10000, "10⁴")],
      [(f1, "red", None, 2.2, "① 每步重新展开", (1, 20, "start")),
       (f3, "blue", None, 2.6, "③ 吸收", (4, -12, "end")),
       (f2, "muted", "6 4", 2.0, "② 缓存展开后的 K、V", (0, 17, "start"))],
      1e9,
      [("T = 4096：① 137.8，② 0.37，③ 1.17", "ink"),
       ("③ / ② → 3.4：吸收用算力换字节", "blue")])
panel(380, "每层每步要读的缓存（MB，bf16）", "MB（对数刻度）", 0.1, 100000,
      [(0.1, "0.1"), (1, "1"), (10, "10"), (100, "100"), (1000, "1000"), (10000, "10⁴"), (100000, "10⁵")],
      [(b2, "muted", "6 4", 2.0, "② 81,920 B / token", (0, 18, "start")),
       (b13, "orange", None, 2.6, "①、③ 1,152 B / token", (0, 17, "start"))],
      1e6,
      [("① 和 ③ 都只读 576 维的潜向量行", "orange"),
       ("② 要读的是它的 71 倍", "ink")])
svg.save("l09-decode-cost.svg")
