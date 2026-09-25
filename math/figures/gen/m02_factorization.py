"""Figure M2.2: a low-rank factorisation W = U V drawn to scale with DeepSeek-V2's MLA key path.

Dimensions (DeepSeek-V2 §3.1.2, cross-referenced from deepseek/reference-docs/lesson-09-doc.md):
d = 5120, d_c = 512, n_h d_h = 128 * 128 = 16384.
Left: the effective key map W^K_eff = W^UK W^DKV (16384 x 5120) stored in full vs as two thin factors,
areas proportional to parameter counts (1 px = 80 rows/cols).
Right: one token's path h (5120) -> c (512) -> k (16384), bar heights proportional to vector length,
with the FLOPs of each matrix-vector product (2 m n).
Writes math/figures/m02-factorization.svg
"""

from m02_svg import SVG

d, dc, dk = 5120, 512, 128 * 128
P_full = dk * d
P_up, P_down = dk * dc, dc * d
F_down, F_up = 2 * dc * d, 2 * dk * dc
F_full = 2 * dk * d
F_build = 2 * dk * dc * d
assert P_full == 83_886_080 and P_up + P_down == 11_010_048

K = 80.0                      # rows / cols per pixel
W, H = 760, 392
svg = SVG(W, H, "低秩分解 W = UV 的形状与参数账（按 DeepSeek-V2 的 MLA 维度等比例绘制）")


def n(x):
    return "{:,}".format(x)


top = 78
hk, wd, wc = dk / K, d / K, dc / K       # 204.8, 64, 6.4 px

# ---- left: full matrix
x0 = 40
svg.text(20, 22, "① 存成一整张表", size=14, weight="600")
svg.rect(x0, top, wd, hk, fill="violet", fill_opacity=0.16, stroke="violet", width=1.4)
svg.text(x0 + wd / 2, top - 8, "5120 列", size=12, color="muted", anchor="middle")
svg.text(x0 - 6, top + hk / 2, "16384 行", size=12, color="muted", anchor="middle", rotate=-90)
svg.text(x0 + wd / 2, top + hk / 2 - 6, "*W*^{K}", size=15, anchor="middle", color="violet")
svg.text(x0 + wd / 2, top + hk / 2 + 14, "满秩存储", size=12, anchor="middle", color="violet")
svg.text(20, top + hk + 26, n(P_full) + " 个参数", size=13)

svg.text(142, top + hk / 2 + 6, "对比", size=13, color="muted", anchor="middle")

# ---- middle: factors in "outer product" layout
ux = 200
vx = ux + wc + 8
svg.text(178, 22, "② 存成两块瘦矩阵：*W*^{UK} *W*^{DKV}", size=14, weight="600")
svg.rect(vx, top - wc - 8, wd, wc, fill="orange", fill_opacity=0.35, stroke="orange", width=1.2)
svg.text(vx, top - wc - 16, "*W*^{DKV}：512 × 5120", size=12, color="orange")
svg.rect(ux, top, wc, hk, fill="violet", fill_opacity=0.35, stroke="violet", width=1.2)
svg.text(ux - 6, top + hk / 2, "*W*^{UK}：16384 × 512", size=12, color="violet", anchor="middle", rotate=-90)
svg.rect(vx, top, wd, hk, stroke="violet", width=1.2, dash="5 4", fill="violet", fill_opacity=0.04)
svg.text(vx + wd / 2, top + hk / 2 - 6, "乘积", size=12, color="violet", anchor="middle")
svg.text(vx + wd / 2, top + hk / 2 + 12, "秩 ≤ 512", size=12, color="violet", anchor="middle")
svg.text(vx + wd / 2, top + hk / 2 + 28, "不存", size=12, color="muted", anchor="middle")
svg.text(178, top + hk + 26, n(P_up) + " + " + n(P_down), size=13)
svg.text(178, top + hk + 44, "= " + n(P_up + P_down) + "（%.1f%%）" % (100 * (P_up + P_down) / P_full), size=13,
         weight="600")

# ---- right: one token's data flow, bar heights proportional to vector length
svg.text(400, 22, "③ 一个 token：*W*^{UK}(*W*^{DKV}*h*_{t})，从右往左", size=14, weight="600")
base = top + hk               # bars stand on this baseline
bw = 22
bx = [420, 560, 700]
labels = [("*h*_{t}", "5120", "ink", d), ("*c*_{t}^{KV}", "512", "orange", dc), ("*k*_{t}", "16384", "violet", dk)]
for x, (name, dim, col, length) in zip(bx, labels):
    h = length / K
    svg.rect(x - bw / 2, base - h, bw, h, fill=col, fill_opacity=0.22 if col != "ink" else 0.10, stroke=col,
             width=1.3)
    svg.text(x, base + 18, name, size=14, anchor="middle", color=col)
    svg.text(x, base + 36, dim + " 维", size=12, anchor="middle", color="muted")
# arrows
ya = base - 30
svg.arrow(bx[0] + bw / 2 + 6, ya, bx[1] - bw / 2 - 6, ya, color="orange", width=1.8)
svg.text((bx[0] + bx[1]) / 2, ya - 44, "乘 *W*^{DKV}", size=13, anchor="middle", color="orange")
svg.text((bx[0] + bx[1]) / 2, ya - 26, "2·512·5120", size=12, anchor="middle", color="muted")
svg.text((bx[0] + bx[1]) / 2, ya - 10, "= %.2fM FLOPs" % (F_down / 1e6), size=12, anchor="middle", color="muted")
svg.arrow(bx[1] + bw / 2 + 6, ya, bx[2] - bw / 2 - 6, ya, color="violet", width=1.8)
svg.text((bx[1] + bx[2]) / 2, ya - 44, "乘 *W*^{UK}", size=13, anchor="middle", color="violet")
svg.text((bx[1] + bx[2]) / 2, ya - 26, "2·16384·512", size=12, anchor="middle", color="muted")
svg.text((bx[1] + bx[2]) / 2, ya - 10, "= %.2fM FLOPs" % (F_up / 1e6), size=12, anchor="middle", color="muted")
svg.text(400, base + 62, "两步合计 %.1fM FLOPs" % ((F_down + F_up) / 1e6), size=13, weight="600")
svg.text(400, base + 82, "直接乘满秩 *W*^{K}：%.1fM FLOPs（%.1f 倍）" % (F_full / 1e6, F_full / (F_down + F_up)),
         size=12)
svg.text(400, base + 100, "先把乘积 *W*^{UK}*W*^{DKV} 造出来：%.1f×10^{10} FLOPs" % (F_build / 1e10),
         size=12, color="red")
svg.line(380, 40, 380, H - 12, color="rule", width=1)

svg.save("m02-factorization.svg")
