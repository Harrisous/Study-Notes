"""Figure 16.9: how mHC computes its three mappings from the residual stream (mechanism figure).

mHC eq. (7)-(9) (research notes 2.4) in the lesson's notation, dimensions of DeepSeek-V4-Flash
(n = 4, d = 4096; notes 3.2):
  x = vec(X_l) in R^{1 x nd} (1 x 16384) -> RMSNorm -> x W_l with W_l = [W^pre | W^post | W^res] in R^{nd x 24}
  -> 24 numbers = 4 (pre) + 4 (post) + 16 (res);
  H~pre = a_pre * (.) + S_pre -> H^pre = sigmoid        in (0,1)^{1x4}
  H~post = a_post * (.) + S_post -> H^post = 2 sigmoid   in (0,2)^{1x4}
  H~res = a_res * mat(.) + S_res -> exp -> 20 x [T_c then T_r] -> H^res ~ doubly stochastic 4x4
Parameter count per sublayer: 16384*24 + 24 + 3 = 393,243 (notes N14).
Writes deepseek/figures/l16-mhc-param.svg
"""

from l16_svg import SVG

n, d = 4, 4096
params = n * d * (n * n + 2 * n) + (n * n + 2 * n) + 3
assert params == 393243
W, H = 760, 440
svg = SVG(W, H, "mHC 怎样从残差流算出三个映射（V4-Flash 的维度）")

# ---------------- tier 1
Y1, H1 = 30, 52
svg.box(14, Y1, 120, H1, ["*X*_{l}", ("4 × 4096", {"size": 12, "color": "muted"})], color="ink",
        fill_opacity=0.06, size=14)
svg.arrow(134, Y1 + 26, 156, Y1 + 26)
svg.box(158, Y1, 150, H1, ["展平 vec(·)", ("1 × 16384", {"size": 12, "color": "muted"})], color="ink",
        fill_opacity=0.06, size=13)
svg.arrow(308, Y1 + 26, 330, Y1 + 26)
svg.box(332, Y1, 140, H1, ["RMSNorm", ("整条 4d 向量一起", {"size": 12, "color": "muted"})], color="muted",
        fill_opacity=0.12, size=13)
svg.arrow(472, Y1 + 26, 494, Y1 + 26)
svg.box(496, Y1, 250, H1, ["× *W*_{l}（16384 × 24）", ("一次小矩阵乘：输出只有 24 维", {"size": 12, "color": "muted"})],
        color="blue", fill_opacity=0.10, size=13)
svg.arrow(621, Y1 + H1, 621, 112)

# ---------------- tier 2: the 24 numbers
Y2 = 114
cols = [(14, 240, "blue", "前 4 个 → 读出 pre"), (262, 240, "green", "中 4 个 → 写入 post"),
        (510, 236, "violet", "后 16 个 → 混合 res")]
counts = [4, 4, 16]
for (x, w, col, lab), k in zip(cols, counts):
    cw = w / k
    for i in range(k):
        svg.rect(x + i * cw + 0.5, Y2, cw - 1, 20, fill=col, fill_opacity=0.35, rx=2)
    svg.text(x + w / 2, Y2 + 38, lab, size=12, anchor="middle", color=col, weight="600")
svg.text(380, Y2 - 6, "24 个数（每个 token 一组）", size=12, anchor="middle", color="muted")

# ---------------- tiers 3-5 per column
Y3, Y4, Y5 = 172, 244, 318
steps = [
    ("× α^{pre} + *S*^{pre}", "σ(·)", ["*H*^{pre}_{l} ∈ (0, 1)^{1×4}", "非负：防止正负抵消"]),
    ("× α^{post} + *S*^{post}", "2σ(·)", ["*H*^{post}_{l} ∈ (0, 2)^{1×4}", "logit 为 0 时恰为 1"]),
    ("× α^{res}，排成 4×4，+ *S*^{res}", "exp → [𝓣_{c}, 𝓣_{r}] × 20", ["*H*^{res}_{l} ≈ 双随机 4×4",
                                                                    "Sinkhorn–Knopp（§5）"]),
]
for (x, w, col, _), (s1, s2, res) in zip(cols, steps):
    cx = x + w / 2
    svg.arrow(cx, Y2 + 46, cx, Y3 - 2)
    svg.box(x + 6, Y3, w - 12, 40, [s1], color=col, fill_opacity=0.08, size=13)
    svg.arrow(cx, Y3 + 40, cx, Y4 - 2)
    svg.box(x + 6, Y4, w - 12, 40, [s2], color=col, fill_opacity=0.16, size=13)
    svg.arrow(cx, Y4 + 40, cx, Y5 - 2)
    svg.box(x + 6, Y5, w - 12, 52, [(res[0], {"weight": "600"}), (res[1], {"size": 12, "color": "muted"})],
            color="green", fill_opacity=0.10, size=13)

svg.text(14, 400, "α：3 个门控标量，初值 0.01；*S*：24 个静态偏置；*W*_{l}：动态投影。三者都是可学习参数。", size=12)
svg.text(14, 422, "每个子层的参数：16384 × 24 + 24 + 3 = %s 个（V4-Flash）；mHC 论文用 φ_{l}、*b*_{l} 表示 *W*_{l}、*S*。"
         % format(params, ","), size=12, color="muted")

svg.save("l16-mhc-param.svg")
