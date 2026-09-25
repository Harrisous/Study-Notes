"""Figure M2.10: what the parenthesis costs. One MLA layer of DeepSeek-V2, one decoded token, T cached tokens
(the new token included; T rather than L because the course reserves L for the number of layers).

Per-layer FLOPs (formulas from deepseek/reference-docs/lesson-09-doc.md §4.4 with its §11.7 correction, recomputed
here; n_h = d_h = 128, d_h^R = 64, d_c = 512):
  (a) expand every cached c_j into keys and values, then attend:  2 T 512 (2 128 128) + 2 128 192 T + 2 128 128 T
  (b) keys/values of the old tokens already expanded and cached (MHA-style cache); only the new token's c_t is
      expanded once:                                               2 512 (2 128 128) + 2 128 192 T + 2 128 128 T
  (c) absorbed (MQA mode): 2 128 128 512 + 2 128 512 128 + (2 128 576 + 2 128 512) T
At T = 1 the cache holds only the new token, so (a) and (b) coincide (33,636,352).
Right: bytes read per cached token per layer in bf16: (b) 2 * 128 * (192 + 128) = 81,920 (40,960 numbers: every head
stores a full 192-dim key, as the HF reference code does); (c) 2 * 576 = 1,152.
Writes math/figures/m02-absorb-flops.svg
"""

import math

import numpy as np

from m02_svg import SVG, polyline_d


def fl(Tv):
    a = 2 * Tv * 512 * (2 * 128 * 128) + 2 * 128 * 192 * Tv + 2 * 128 * 128 * Tv
    b = 2 * 512 * (2 * 128 * 128) + 2 * 128 * 192 * Tv + 2 * 128 * 128 * Tv
    c = 2 * 128 * 128 * 512 + 2 * 128 * 512 * 128 + (2 * 128 * 576 + 2 * 128 * 512) * Tv
    return a, b, c


T = np.logspace(0, math.log10(131072), 200)
a, b, c = fl(T)
a4, b4, c4 = fl(4096)
assert fl(1)[0] == fl(1)[1] == 33_636_352
assert abs(a4 / 1e9 - 137.77) < 0.01 and abs(c4 / 1e9 - 1.1744) < 1e-3 and abs(b4 / 1e9 - 0.3691) < 1e-3
assert round(c4 / b4, 2) == 3.18

W, H = 760, 330
svg = SVG(W, H, "MLA 一层解码一步的 FLOPs 随缓存长度变化：先展开、缓存展开结果、吸收三种算法")
x0, x1, y0, y1 = 76, 470, 44, 262
lx0, lx1 = 0.0, math.log10(131072)
ly0, ly1 = 5.0, 13.0


def X_(v):
    return x0 + (x1 - x0) * (math.log10(v) - lx0) / (lx1 - lx0)


def Y_(v):
    return y1 - (y1 - y0) * (math.log10(v) - ly0) / (ly1 - ly0)


svg.text(20, 24, "① 一层、解码一个 token 的 FLOPs（对数坐标）", size=14, weight="600")
for e in range(5, 14):
    svg.line(x0, Y_(10 ** e), x1, Y_(10 ** e), color="grid", width=1)
    if e % 2 == 0:
        svg.text(x0 - 8, Y_(10 ** e) + 4, "10^{%d}" % e, size=11, color="muted", anchor="end")
for v, lab in ((1, "1"), (16, "16"), (256, "256"), (4096, "4K"), (131072, "128K")):
    svg.line(X_(v), y1, X_(v), y1 + 5, color="muted", width=1)
    svg.text(X_(v), y1 + 19, lab, size=11, color="muted", anchor="middle")
svg.text((x0 + x1) / 2, y1 + 38, "缓存长度 T（token，含新 token）", size=12, color="muted", anchor="middle")
svg.path(polyline_d([(X_(p), Y_(q)) for p, q in zip(T, a)]), stroke="red", width=2.2)
svg.path(polyline_d([(X_(p), Y_(q)) for p, q in zip(T, c)]), stroke="orange", width=2.4)
svg.path(polyline_d([(X_(p), Y_(q)) for p, q in zip(T, b)]), stroke="violet", width=2.2, dash="6 4")
svg.line(X_(4096), y0, X_(4096), y1, color="muted", width=1, dash="2 3")
for v, col, dx, dy, anc in ((a4, "red", -6, -8, "end"), (c4, "orange", -6, -8, "end"), (b4, "violet", 6, 18, "start")):
    svg.circle(X_(4096), Y_(v), 4, fill=col)
    svg.text(X_(4096) + dx, Y_(v) + dy, "%.3g GFLOPs" % (v / 1e9), size=12, color=col, anchor=anc)
# legend in the empty band under the curves (every curve is >= 3.36e7 FLOPs, i.e. above 10^7.5)
lg = [("red", None, "(a) 每步先把 T 个 *c*_{j} 展开"), ("violet", "6 4", "(b) 缓存展开后的 K、V"),
      ("orange", None, "(c) 吸收")]
for i, (col, dash, lab) in enumerate(lg):
    yy = Y_(10 ** 6.75) + 19 * i
    svg.line(x0 + 10, yy - 4, x0 + 34, yy - 4, color=col, width=2.2, dash=dash)
    svg.text(x0 + 40, yy, lab, size=12, color=col)
svg.text(x1, y0 - 6, "T = 4096：(a)/(c) = %.0f 倍" % (a4 / c4), size=12, anchor="end")

# right: bytes per cached token
svg.line(500, 16, 500, H - 16, color="rule", width=1)
svg.text(512, 24, "② 每个缓存 token 读几个字节", size=14, weight="600")
svg.text(512, 44, "（每层，bf16）", size=12, color="muted")
bx0, bx1 = 540, 740
by = [120, 190]
vals = [(81920, "(b) 展开的 K、V", "violet"), (1152, "(c) 吸收：只读 [*c*_{j}; *k*^{R}_{j}]", "orange")]
for (v, lab, col), yy in zip(vals, by):
    w = (bx1 - bx0) * v / 81920
    svg.rect(bx0, yy, max(w, 2), 26, fill=col, fill_opacity=0.3, stroke=col, width=1.2)
    svg.text(bx0, yy - 8, lab, size=12, color=col)
    svg.text(bx0 + max(w, 2) + 6 if v < 50000 else bx1 - 6, yy + 18, "{:,} B".format(v), size=12,
             color="ink", anchor="start" if v < 50000 else "end")
for i, (t, col) in enumerate((("吸收后，每个缓存 token 的 FLOPs", "ink"),
                              ("是 (b) 的 %.1f 倍，要读的字节却只有" % (278528 / 81920), "ink"),
                              ("(b) 的 1/%.0f：用算力换带宽。" % (81920 / 1152), "ink"),
                              ("(b) 要为每层每个 token 存 40,960", "muted"),
                              ("个数，MLA 只存 576 个（第 9 课）。", "muted"))):
    svg.text(512, 238 + 18 * i, t, size=12, color=col)

svg.save("m02-absorb-flops.svg")
print("T=4096: a %.4g b %.4g c %.4g, a/c %.1f, c/b %.2f" % (a4, b4, c4, a4 / c4, c4 / b4))
