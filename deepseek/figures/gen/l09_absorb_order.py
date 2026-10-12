"""Figure 9.5: two ways to "absorb" W^UK into the query side, drawn to scale for one head.

(a) precompute M_i = (W^UK_i)^T W^UQ_i once, a dense [512 x 1536] block;
(b) keep the two thin factors and change the order of multiplication:
    q^C = W^UQ_i c^Q  ([128 x 1536] x [1536]), then q~ = (W^UK_i)^T q^C  ([512 x 128] x [128]).
V2 dims (arXiv:2405.04434v5 sec. 3.1.2): d_c = 512, d_c' = 1536, d_h = 128, n_h = 128.
Per head: (a) 786,432 parameters, 2*512*1536 = 1,572,864 FLOPs per token;
          (b) 196,608 + 65,536 = 262,144 parameters, 393,216 + 131,072 = 524,288 FLOPs.
Per layer (x128): 100,663,296 vs 33,554,432 parameters; 201,326,592 vs 67,108,864 FLOPs
(the same numbers as the worked example in sec. 3 of the lesson).
Rectangles are to scale (1 dimension = 0.28 px); the vectors are not.
Writes deepseek/figures/l09-absorb-order.svg
"""

from l09_svg import SVG, comma

d_c, d_q, d_h, n_h = 512, 1536, 128, 128
pre_par, pre_flop = d_c * d_q, 2 * d_c * d_q
ord_par = d_h * d_q + d_c * d_h
ord_flop = 2 * d_h * d_q + 2 * d_c * d_h
assert (pre_par, pre_flop, ord_par, ord_flop) == (786_432, 1_572_864, 262_144, 524_288)
assert pre_par * n_h == 100_663_296 and ord_par * n_h == 33_554_432
assert pre_flop * n_h == 201_326_592 and ord_flop * n_h == 67_108_864
print(pre_par / ord_par, pre_flop / ord_flop)

S = 0.28                       # px per dimension
WQ, HK, TH = d_q * S, d_c * S, d_h * S   # 430.1, 143.4, 35.8
W, H = 760, 500
svg = SVG(W, H, "吸收的两种做法：预乘成一块，还是改变乘法顺序")
X0 = 96                        # left edge of every [512 x 1536]-shaped region
RX = X0 + WQ + 22              # right-hand text column

# ---------------------------------------------------------------- (a) precompute
svg.text(16, 22, "预乘：离线把 (*W*^{UK}_{i})^{⊤}*W*^{UQ}_{i} 乘成一整块", size=14, weight="700")
ya = 36
svg.rect(X0, ya, WQ, HK, fill="muted", fill_opacity=0.16, stroke="ink", width=1.4, rx=2)
svg.text(X0 + WQ / 2, ya + HK / 2 - 12, "(*W*^{UK}_{i})^{⊤}*W*^{UQ}_{i}", size=14, anchor="middle", weight="700")
svg.text(X0 + WQ / 2, ya + HK / 2 + 8, "[512 × 1536]，稠密，每个数都要存", size=12, anchor="middle", color="muted")
svg.text(X0 + WQ / 2, ya + HK / 2 + 28, "%s 个参数" % comma(pre_par), size=12.5, anchor="middle", weight="700")
svg.text(X0 - 8, ya + HK / 2 + 4, "512", size=11.5, anchor="end", color="muted")
svg.text(X0 + WQ / 2, ya + HK + 14, "1536", size=11.5, anchor="middle", color="muted")
svg.text(RX, ya + 22, "每头、每 token：", size=12)
svg.text(RX, ya + 42, "× *c*^{Q}_{t} [1536] → *q̃*_{t,i} [512]", size=12, color="blue")
svg.text(RX, ya + 66, "2 × 512 × 1536", size=12)
svg.text(RX, ya + 84, "= %s 次运算" % comma(pre_flop), size=12.5, weight="700")
svg.line(12, 204, W - 12, 204, color="rule", width=1)

# ---------------------------------------------------------------- (b) change the order
svg.text(16, 228, "改变顺序：两块瘦矩阵分开存，先乘 *W*^{UQ}_{i}，再乘 (*W*^{UK}_{i})^{⊤}", size=14,
         weight="700", color="blue")
yb = 242
# W^UQ_i: [128 x 1536], drawn on top of the would-be product
svg.rect(X0, yb, WQ, TH, fill="blue", fill_opacity=0.18, stroke="blue", width=1.4, rx=2)
svg.text(X0 + WQ / 2, yb + TH / 2 + 5, "*W*^{UQ}_{i}  [128 × 1536]  %s" % comma(d_h * d_q), size=12.5,
         anchor="middle", color="blue", weight="700")
# (W^UK_i)^T: [512 x 128], drawn to the left of it
yk = yb + TH + 4
svg.rect(X0 - TH - 4, yk, TH, HK, fill="violet", fill_opacity=0.18, stroke="violet", width=1.4, rx=2)
svg.text(X0 - TH / 2 - 4, yk + HK / 2 + 4, "512", size=11.5, anchor="middle", color="violet")
svg.text(X0 - TH / 2 - 4, yk + HK + 14, "128", size=11.5, anchor="middle", color="violet")
# the product region that is never materialised
svg.rect(X0, yk, WQ, HK, fill=None, stroke="muted", width=1.2, rx=2, dash="5 4")
svg.text(X0 + 10, yk + 22, "← (*W*^{UK}_{i})^{⊤}  [512 × 128]  %s" % comma(d_c * d_h), size=12.5,
         color="violet", weight="700")
svg.text(X0 + WQ / 2, yk + HK / 2 + 8, "虚线框 = 两块相乘会得到的 [512 × 1536]", size=12, anchor="middle",
         color="muted")
svg.text(X0 + WQ / 2, yk + HK / 2 + 26, "这里从不把它算出来，也不存", size=12, anchor="middle", color="muted")
svg.text(X0 + WQ / 2, yk + HK - 12, "两条色块合计 %s 个参数，是整块的 1/3" % comma(ord_par), size=12.5,
         anchor="middle", weight="700")
svg.text(RX, yb + 22, "每头、每 token：", size=12)
svg.text(RX, yb + 42, "*W*^{UQ}_{i}*c*^{Q}_{t} → *q*^{C}_{t,i} [128]", size=12, color="blue")
svg.text(RX, yb + 60, "2 × 128 × 1536 = %s" % comma(2 * d_h * d_q), size=12)
svg.text(RX, yb + 84, "(*W*^{UK}_{i})^{⊤}*q*^{C}_{t,i} → *q̃*_{t,i} [512]", size=12, color="violet")
svg.text(RX, yb + 102, "2 × 512 × 128 = %s" % comma(2 * d_c * d_h), size=12)
svg.text(RX, yb + 126, "合计 %s 次运算" % comma(ord_flop), size=12.5, weight="700")
svg.text(RX, yb + 144, "（预乘的 1/3）", size=12, color="blue")

# ---------------------------------------------------------------- summary
svg.text(16, H - 30, "每层 128 个头：预乘 %s 个参数、%s 次运算；改变顺序 %s 个参数、%s 次运算。"
         % (comma(pre_par * n_h), comma(pre_flop * n_h), comma(ord_par * n_h), comma(ord_flop * n_h)),
         size=12)
svg.text(16, H - 12, "差 3 倍的原因：中间维 d_{h} = 128 远小于两端的 512、1536，512 × 1536 ÷ (128 × (512 + 1536)) = 3。",
         size=12, color="muted")
svg.save("l09-absorb-order.svg")
