"""Figure 9.4: low-rank joint KV compression, vector lengths drawn to scale (DeepSeek-V2 dims).

h_t in R^5120 --W^DKV--> c_t^KV in R^512 --W^UK--> k_t^C in R^16384 (128 heads x 128)
                                        --W^UV--> v_t^C in R^16384
Dims from arXiv:2405.04434v5 sec. 3.1.2 (d = 5120, n_h = d_h = 128, d_c = 512).
Bar heights are proportional to the vector length (16384 -> 150 px).  Parameter counts:
W^DKV 512*5120, W^UK = W^UV 16384*512; MHA's W^K, W^V are 16384*5120 each.
Writes deepseek/figures/l09-joint-compression.svg
"""

from l09_svg import SVG

d, n_h, d_h, d_c = 5120, 128, 128, 512
nd = n_h * d_h
p_dkv, p_uk = d_c * d, nd * d_c
p_mha = 2 * nd * d
assert p_dkv == 2_621_440 and p_uk == 8_388_608 and p_mha == 167_772_160

W, H = 740, 446
svg = SVG(W, H, "低秩 KV 联合压缩：5120 → 512 → 2 × 16384，按比例画")
K = 150 / nd                     # px per dimension
YC = 206                         # vertical centre of h and c
BW = 26


def vbar(x, y, dim, col, cached=False, segs=0):
    h = dim * K                  # exactly to scale: 512 -> 4.7 px, 5120 -> 46.9 px, 16384 -> 150 px
    svg.rect(x, y, BW, h, fill=col, fill_opacity=0.28, stroke="amber" if cached else col,
             width=2.6 if cached else 1.2, rx=3)
    for s in range(1, segs):
        yy = y + h * s / segs
        svg.line(x, yy, x + BW, yy, color=col, width=0.8, opacity=0.6)
    return h


# h_t
xh = 60
hh = vbar(xh, YC - d * K / 2, d, "blue")
svg.text(xh + BW / 2, YC - hh / 2 - 24, "*h*_{t}", size=14, anchor="middle", color="blue", weight="700")
svg.text(xh + BW / 2, YC - hh / 2 - 8, "d = 5120", size=11.5, anchor="middle", color="muted")
# c
xc = 262
HC2 = d_c * K / 2               # half-height of the c^KV bar
hc = vbar(xc, YC - HC2, d_c, "orange", cached=True)
svg.text(xc + BW / 2, YC - 30, "*c*^{KV}_{t}", size=14, anchor="middle", color="orange", weight="700")
svg.text(xc + BW / 2, YC - 13, "d_{c} = 512", size=11.5, anchor="middle", color="muted")
svg.text(xc + BW / 2, YC + 26, "推理时缓存", size=12, anchor="middle", color="amber", weight="600")
svg.text(xc + BW / 2, YC + 42, "只有这 512 个数", size=11.5, anchor="middle", color="amber")
# down projection wedge
svg.polygon([(xh + BW, YC - hh / 2), (xh + BW, YC + hh / 2), (xc, YC + HC2), (xc, YC - HC2)],
            fill="orange", fill_opacity=0.12)
svg.text((xh + xc + BW) / 2 + 4, YC - 64, "下投影 *W*^{DKV}", size=13, anchor="middle", weight="600")
svg.text((xh + xc + BW) / 2 + 4, YC - 47, "[512 × 5120]", size=11.5, anchor="middle", color="muted")
svg.text((xh + xc + BW) / 2 + 4, YC + 56, "%s 个参数" % format(p_dkv, ","), size=11.5, anchor="middle",
         color="muted")
# up projections: k on top, v below
xk = 520
yk_top, yv_top = YC - 10 - nd * K, YC + 10
hk = vbar(xk, yk_top, nd, "violet", segs=8)
hv = vbar(xk, yv_top, nd, "green", segs=8)
svg.polygon([(xc + BW, YC - HC2), (xc + BW, YC), (xk, yk_top + hk), (xk, yk_top)], fill="violet", fill_opacity=0.10)
svg.polygon([(xc + BW, YC), (xc + BW, YC + HC2), (xk, yv_top + hv), (xk, yv_top)], fill="green", fill_opacity=0.10)
svg.text(420, YC - 92, "上投影 *W*^{UK}", size=13, anchor="middle", weight="600", color="violet")
svg.text(420, YC - 75, "[16384 × 512]", size=11.5, anchor="middle", color="muted")
svg.text(420, YC + 84, "上投影 *W*^{UV}", size=13, anchor="middle", weight="600", color="green")
svg.text(420, YC + 101, "[16384 × 512]", size=11.5, anchor="middle", color="muted")
svg.text(420, YC + 6, "各 %s 个参数" % format(p_uk, ","), size=11.5, anchor="middle", color="muted")
tx = xk + BW + 14
svg.text(tx, yk_top + 16, "*k*^{C}_{t}：16384 维", size=13, color="violet", weight="700")
svg.text(tx, yk_top + 34, "= 128 个头 × 每头 128 维", size=12)
svg.text(tx, yk_top + 52, "（横线每 16 个头一条）", size=11.5, color="muted")
svg.text(tx, yv_top + 16, "*v*^{C}_{t}：16384 维", size=13, color="green", weight="700")
svg.text(tx, yv_top + 34, "= 128 个头 × 每头 128 维", size=12)
svg.text(tx, YC + 4, "推理时不缓存", size=12, color="muted", weight="600")

# bottom comparison
yb = 400
svg.line(20, yb - 22, W - 20, yb - 22, color="rule", width=1)
svg.text(24, yb, "MHA：每 token 每层缓存 *k*、*v* 共 32,768 个数；*W*^{K}、*W*^{V} 共 %s 个参数。" % format(p_mha, ","),
         size=12.5)
svg.text(24, yb + 24, "MLA：缓存 512 个数（另有 64 维的 *k*^{R}，见 §5）；*W*^{DKV}、*W*^{UK}、*W*^{UV} 共 %s 个参数。"
         % format(p_dkv + 2 * p_uk, ","), size=12.5, color="orange")
svg.save("l09-joint-compression.svg")
