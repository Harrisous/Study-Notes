"""Figure M1.10: two bracketings of the same score q^T (W^UK c_j), one attention head.

Cached latents C = [c_1; ...; c_T] in R^{T x d_c}; per-head key up-projection W^UK_i in R^{d_h x d_c}.
  expand first : K_i = C (W^UK_i)^T, then scores K_i q_i        -> 2 T d_c d_h + 2 T d_h FLOPs
  absorb first : q~_i = (W^UK_i)^T q_i, then scores C q~_i      -> 2 d_h d_c + 2 T d_c FLOPs
DeepSeek-V2 dimensions d_h = 128, d_c = 512 (arXiv:2405.04434, sec. 3.1.2); T = 4096 is an example.
Blocks: widths/heights for d_c and d_h are to scale; T is drawn schematically. RoPE is ignored here.
Writes math/figures/m01-absorption.svg
"""

from m01_svg import SVG

d_h, d_c, T = 128, 512, 4096
expand = 2 * T * d_c * d_h + 2 * T * d_h
absorb = 2 * d_h * d_c + 2 * T * d_c
assert expand == 537_919_488 and absorb == 4_325_376
ratio = expand / absorb

k = 0.1875                     # px per dimension: d_c -> 96, d_h -> 24
DC, DH = d_c * k, d_h * k
TT = 112                       # schematic height for T rows
V = 12                         # drawn width of a vector

W, H = 720, 534
svg = SVG(W, H, "同一个注意力分数的两种算法：先展开键，或先把 W^UK 吸收进查询")


def cache_block(x, y):
    svg.rect(x, y, DC, TT, fill="orange", fill_opacity=0.14, stroke="amber", width=1.8)
    svg.text(x + DC / 2, y + TT + 18, "缓存 *C*  [T, d_{c}]", size=12, anchor="middle", color="amber", weight="600")


def op(x, y, s):
    svg.text(x, y, s, size=18, anchor="middle", color="muted")


# ---------------------------------------------------------------- A: expand first
ya = 52
svg.text(16, 24, "Ⓐ 先展开：用 *W*^{UK}_{i} 把缓存里每个 *c*_{j} 变回键，再与查询点积", size=14, weight="600", color="violet")
x = 30
cache_block(x, ya)
op(x + DC + 16, ya + TT / 2 + 6, "×")
x2 = x + DC + 32
svg.rect(x2, ya + (TT - DC) / 2, DH, DC, fill="violet", fill_opacity=0.16, stroke="violet", width=1.3)
svg.text(x2 + DH / 2, ya + (TT - DC) / 2 - 13, "(*W*^{UK}_{i})^{⊤}", size=12, anchor="middle", color="violet", weight="600")
svg.text(x2 + DH / 2, ya + TT + 18, "[d_{c}, d_{h}]", size=12, anchor="middle", color="muted")
op(x2 + DH + 16, ya + TT / 2 + 6, "=")
x3 = x2 + DH + 32
svg.rect(x3, ya, DH, TT, fill="violet", fill_opacity=0.16, stroke="violet", width=1.3)
svg.text(x3 + DH / 2, ya + TT + 18, "*K*_{i}  [T, d_{h}]", size=12, anchor="middle", color="violet", weight="600")
op(x3 + DH + 16, ya + TT / 2 + 6, "×")
x4 = x3 + DH + 32
svg.rect(x4, ya + TT / 2 - DH / 2, V, DH, fill="blue", fill_opacity=0.2, stroke="blue", width=1.3)
svg.text(x4 + V / 2, ya + TT / 2 - DH / 2 - 11, "*q*_{i}", size=12, anchor="middle", color="blue", weight="600")
svg.text(x4 + V / 2, ya + TT + 18, "[d_{h}]", size=12, anchor="middle", color="muted")
op(x4 + V + 16, ya + TT / 2 + 6, "=")
x5 = x4 + V + 32
svg.rect(x5, ya, V, TT, fill="ink", fill_opacity=0.10, stroke="ink", width=1.2)
svg.text(x5 + V / 2, ya + TT + 18, "分数 [T]", size=12, anchor="middle")
tx = 400
svg.text(tx, ya + 18, "每一步都要把 T 个 *c*_{j} 展开：", size=13)
svg.text(tx, ya + 40, "2Td_{c}d_{h} = %s" % format(2 * T * d_c * d_h, ","), size=13, color="violet")
svg.text(tx, ya + 62, "再做 T 个 d_{h} 维点积：", size=13)
svg.text(tx, ya + 84, "2Td_{h} = %s" % format(2 * T * d_h, ","), size=13, color="violet")
svg.text(tx, ya + 108, "合计 %s（每头每步）" % format(expand, ","), size=13, weight="700")
svg.line(16, 196, W - 16, 196, color="rule", width=1)

# ---------------------------------------------------------------- B: absorb first
yb = 254                        # leaves room for the (W^UK_i)^T label under the section title
svg.text(16, 220, "Ⓑ 先吸收：把查询变成 d_{c} 维，直接与缓存点积", size=14, weight="600", color="blue")
x = 30
svg.rect(x, yb + (TT - DC) / 2, DH, DC, fill="violet", fill_opacity=0.16, stroke="violet", width=1.3)
svg.text(x + DH / 2, yb + (TT - DC) / 2 - 13, "(*W*^{UK}_{i})^{⊤}", size=12, anchor="start", color="violet", weight="600")
svg.text(x + DH / 2, yb + TT + 18, "[d_{c}, d_{h}]", size=12, anchor="middle", color="muted")
op(x + DH + 16, yb + TT / 2 + 6, "×")
x2 = x + DH + 32
svg.rect(x2, yb + TT / 2 - DH / 2, V, DH, fill="blue", fill_opacity=0.2, stroke="blue", width=1.3)
svg.text(x2 + V / 2, yb + TT / 2 - DH / 2 - 11, "*q*_{i}", size=12, anchor="middle", color="blue", weight="600")
op(x2 + V + 16, yb + TT / 2 + 6, "=")
x3 = x2 + V + 32
svg.rect(x3, yb + (TT - DC) / 2, V, DC, fill="blue", fill_opacity=0.2, stroke="blue", width=1.3)
svg.text(x3 + V / 2, yb + TT + 18, "*q̃*_{i}  [d_{c}]", size=12, anchor="middle", color="blue", weight="600")
svg.text(x3 + V + 22, yb + TT / 2 + 5, "然后", size=12, anchor="middle", color="muted")
x4 = x3 + V + 44
cache_block(x4, yb)
op(x4 + DC + 16, yb + TT / 2 + 6, "×")
x5 = x4 + DC + 32
svg.rect(x5, yb + (TT - DC) / 2, V, DC, fill="blue", fill_opacity=0.2, stroke="blue", width=1.3)
svg.text(x5 + V / 2, yb + (TT - DC) / 2 - 11, "*q̃*_{i}", size=12, anchor="middle", color="blue", weight="600")
op(x5 + V + 16, yb + TT / 2 + 6, "=")
x6 = x5 + V + 32
svg.rect(x6, yb, V, TT, fill="ink", fill_opacity=0.10, stroke="ink", width=1.2)
svg.text(x6 + V / 2, yb + TT + 18, "分数 [T]", size=12, anchor="middle")
svg.text(tx, yb + 18, "查询只变一次：", size=13)
svg.text(tx, yb + 40, "2d_{h}d_{c} = %s" % format(2 * d_h * d_c, ","), size=13, color="blue")
svg.text(tx, yb + 62, "再做 T 个 d_{c} 维点积：", size=13)
svg.text(tx, yb + 84, "2Td_{c} = %s" % format(2 * T * d_c, ","), size=13, color="blue")
svg.text(tx, yb + 108, "合计 %s（每头每步）" % format(absorb, ","), size=13, weight="700")
svg.line(16, 398, W - 16, 398, color="rule", width=1)

# ---------------------------------------------------------------- bill
yc = 422
svg.text(16, yc, "账单：d_{h} = 128，d_{c} = 512（DeepSeek-V2），T = 4096，一个头、生成一个 token", size=13, weight="600")
BX, BW = 150, 440
for i, (lab, val, col) in enumerate((("Ⓐ 先展开", expand, "violet"), ("Ⓑ 先吸收", absorb, "blue"))):
    y = yc + 20 + i * 30
    svg.text(BX - 10, y + 14, lab, size=13, anchor="end", color=col, weight="600")
    w = max(2.0, BW * val / expand)
    svg.rect(BX, y, w, 18, fill=col, fill_opacity=0.55, rx=2)
    svg.text(BX + w + 8, y + 14, "%.2f×10^{%d}" % (val / 10 ** (len(str(val)) - 1), len(str(val)) - 1), size=12, color="ink")
svg.text(BX, yc + 100, "约 %.0f 倍。T 越大，差距越接近 d_{h}(1 + 1/d_{c}) = %.2f；T = 1 时两者几乎相等。"
         % (ratio, d_h * (1 + 1 / d_c)),
         size=12, color="muted")

svg.save("m01-absorption.svg")
