"""Figure 12.5: fine-grained quantization groups of DeepSeek-V3, drawn to scale along K.

Activations X (T x 7168) are scaled per 1x128 tile (one token, 128 channels); a weight
W (2048 x 7168, e.g. one expert's gate projection: moe_intermediate_size 2048, hidden 7168)
is scaled per 128x128 block. Both are drawn with the inner (reduction) dimension K = 7168
horizontal at the same scale, 10 px per 128 channels, so the g-th tile column lines up with
the g-th block column: the scale factors change along K, which is why a standard FP8 GEMM
(one scale per tensor) cannot apply them after the full K-sum.
Writes deepseek/figures/l12-tile-block.svg
"""

from l12_svg import SVG

K, NOUT, G = 7168, 2048, 128
ncol = K // G            # 56 groups along K
nblk = NOUT // G         # 16 block rows
assert ncol == 56 and nblk == 16 and ncol * nblk == 896
T_SHOW = 6
g_hi, t_hi, a_hi = 20, 2, 5

W_, H_ = 760, 462
X0 = 150
U = 10                   # px per 128 channels
XY, RH = 58, 14          # activation block: top and row height
WY = XY + T_SHOW * RH + 44

svg = SVG(W_, H_, "细粒度量化：激活按 1×128 tile、权重按 128×128 block 各配一个缩放因子（沿内维 K 对齐）")

# K axis
svg.line(X0, 30, X0 + ncol * U, 30, color="muted", width=1.2)
for g in range(0, ncol + 1, 8):
    svg.line(X0 + g * U, 26, X0 + g * U, 34, color="muted", width=1)
    svg.text(X0 + g * U, 20, str(g * G), size=11, anchor="middle", color="muted")
svg.text(X0 + ncol * U + 8, 34, "K", size=13, weight="700")

# highlighted column band across both matrices
svg.rect(X0 + g_hi * U, XY - 6, U, WY + nblk * U - XY + 12, fill="amber", fill_opacity=0.28)

# activation
for t in range(T_SHOW):
    for g in range(ncol):
        hi = (t == t_hi and g == g_hi)
        svg.rect(X0 + g * U, XY + t * RH, U, RH, fill="orange" if hi else "blue",
                 fill_opacity=0.85 if hi else 0.10, stroke="blue", width=0.4)
svg.text(X0 - 10, XY + 18, "激活 X", size=13, anchor="end", weight="600", color="blue")
svg.text(X0 - 10, XY + 36, "T × 7168", size=12, anchor="end", color="muted")
svg.text(X0 - 10, XY + 54, "（画 6 行，高度示意）", size=11, anchor="end", color="muted")

# weight
for a in range(nblk):
    for g in range(ncol):
        hi = (a == a_hi and g == g_hi)
        svg.rect(X0 + g * U, WY + a * U, U, U, fill="orange" if hi else "violet",
                 fill_opacity=0.85 if hi else 0.08, stroke="violet", width=0.4)
svg.text(X0 - 10, WY + 40, "权重 W", size=13, anchor="end", weight="600", color="violet")
svg.text(X0 - 10, WY + 58, "2048 × 7168", size=12, anchor="end", color="muted")
svg.text(X0 - 10, WY + 76, "（一个专家的", size=11, anchor="end", color="muted")
svg.text(X0 - 10, WY + 92, "一个投影）", size=11, anchor="end", color="muted")

# callouts
BY = WY + nblk * U + 30
svg.text(X0, BY, "一个 1×128 tile（橙，上）：第 t 个 token 的第 g 段 128 个通道，共用 1 个缩放因子 s_{t,g}", size=12)
svg.text(X0, BY + 23, "一个 128×128 block（橙，下）：128 个输出行 × 同一段 128 个输入通道，共用 1 个缩放因子 S_{a,g}", size=12)
svg.text(X0, BY + 46, "每个 token 56 个激活缩放因子；这个权重 16 × 56 = 896 个。黄带 = 同一个 g：两边的缩放因子都随 g 变。",
         size=12, color="muted")
svg.text(X0, BY + 68, "反向算 Wgrad 时内维变成 token 方向：缓存的激活要反量化、转置，再按 128×1 tile 重新量化。", size=11, color="muted")
svg.save("l12-tile-block.svg")
