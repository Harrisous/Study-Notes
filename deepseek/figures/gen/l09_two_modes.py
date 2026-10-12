"""Figure 9.11: the two ways to run the same MLA layer -- MHA mode vs MQA mode.

After DeepSeek-V3.2 (arXiv:2512.02556v1) appendix A: "the MHA mode is used for training and
prefilling, while the MQA mode is used for decoding".  FlashMLA README: MQA mode means
head_dim_k = 576, head_dim_v = 512; MHA mode head_dim_k = 192, head_dim_v = 128.
FLOPs per layer (lesson-09-doc sec. 4.8), V2 dims n_h = d_h = 128, d_h^R = 64, d_c = 512:
  per (query, key) pair: MHA mode 2*128*(192+128) = 81,920; MQA mode 2*128*(576+512) = 278,528
  per token fixed cost (expand, or absorb + W^UV): 2*512*128*128*2 = 33,554,432 in both modes
  prefill T tokens, causal ~ T(T+1)/2 pairs; decode one token against L cached tokens.
Writes deepseek/figures/l09-two-modes.svg
"""

from l09_svg import SVG

FIX = 2 * 512 * 128 * 128 * 2
PAIR_MHA, PAIR_MQA = 2 * 128 * (192 + 128), 2 * 128 * (576 + 512)
assert FIX == 33_554_432 and PAIR_MHA == 81_920 and PAIR_MQA == 278_528
T = L = 4096
pairs = T * (T + 1) // 2
pre_mha = PAIR_MHA * pairs + FIX * T
pre_mqa = PAIR_MQA * pairs + FIX * T
dec_mha = (FIX + PAIR_MHA) * L          # must re-expand all L cached latents
dec_mqa = FIX + PAIR_MQA * L
print(pre_mha / 1e9, pre_mqa / 1e9, dec_mha / 1e9, dec_mqa / 1e9)
assert round(pre_mha / 1e9, 1) == 824.8 and round(pre_mqa / 1e9, 1) == 2474.5

W, H = 760, 470
svg = SVG(W, H, "同一层 MLA 的两种算法：MHA 模式与 MQA 模式")
BW, BH, BG = 16, 18, 4


def heads_row(x0, y, n=8, col="blue"):
    xs = []
    for i in range(n):
        x = x0 + i * (BW + BG)
        svg.rect(x, y, BW, BH, fill=col, fill_opacity=0.25, stroke=col, width=1.1, rx=3)
        xs.append(x)
    return xs


# ---------------- left: MHA mode
LX = 20
svg.rect(LX, 8, 350, 300, fill="grid", fill_opacity=0.35, rx=8)
svg.text(LX + 175, 30, "MHA 模式：先把缓存展开", size=14, anchor="middle", weight="700")
svg.text(LX + 175, 48, "训练与预填充用它", size=12, anchor="middle", color="muted")
qx = heads_row(LX + 176, 62)
svg.text(LX + 168, 76, "128 个查询头，每头 192 维", size=11.5, anchor="end", color="blue")
for x in qx:
    svg.line(x + BW / 2, 80, x + BW / 2, 118, color="ink", width=0.9, opacity=0.55)
for x in qx:
    svg.rect(x, 118, BW, BH, fill="violet", fill_opacity=0.3, stroke="violet", width=1.1, rx=3)
    svg.rect(x, 142, BW, BH, fill="green", fill_opacity=0.3, stroke="green", width=1.1, rx=3)
svg.text(LX + 168, 132, "K：每头 192 维", size=11.5, anchor="end", color="violet")
svg.text(LX + 168, 156, "V：每头 128 维", size=11.5, anchor="end", color="green")
svg.polygon([(qx[0], 164), (qx[-1] + BW, 164), (LX + 274, 196), (LX + 234, 196)], fill="orange", fill_opacity=0.12)
svg.text(LX + 120, 190, "每个 token 展开一次", size=11.5, anchor="middle", color="orange")
svg.text(LX + 120, 206, "×*W*^{UK}、×*W*^{UV}", size=11.5, anchor="middle", color="orange")
svg.rect(LX + 209, 198, 90, 22, fill="orange", fill_opacity=0.25, stroke="amber", width=2.4, rx=4)
svg.text(LX + 254, 214, "缓存 576", size=12, anchor="middle", color="orange", weight="700")
svg.text(LX + 16, 246, "每个（查询, 键）对：2 · 128 · (192 + 128)", size=12)
svg.text(LX + 16, 264, "= %s 次运算" % format(PAIR_MHA, ","), size=12.5, weight="700")
svg.text(LX + 16, 286, "每个 token 展开一次：%s 次" % format(FIX, ","), size=12, color="muted")

# ---------------- right: MQA mode
RX = 390
svg.rect(RX, 8, 350, 300, fill="grid", fill_opacity=0.35, rx=8)
svg.text(RX + 175, 30, "MQA 模式：吸收，缓存原样当作一个 KV 头", size=14, anchor="middle", weight="700")
svg.text(RX + 175, 48, "解码用它", size=12, anchor="middle", color="muted")
qx = heads_row(RX + 176, 62)
svg.text(RX + 168, 76, "128 个查询头，每头 576 维", size=11.5, anchor="end", color="blue")
kx, kw = RX + 176 + 78 - 46, 92
for x in qx:
    svg.line(x + BW / 2, 80, kx + kw / 2, 115, color="ink", width=0.9, opacity=0.55)
svg.rect(kx, 115, kw, 22, fill="orange", fill_opacity=0.28, stroke="amber", width=2.4, rx=3)
svg.text(kx + kw / 2, 131, "[*c* ; *k*^{R} ]  576", size=11.5, anchor="middle", color="orange", weight="700")
vw = kw * 512 / 576          # V = the first 512 of the 576 cached numbers, drawn left-aligned under K
svg.rect(kx, 142, vw, BH, fill="orange", fill_opacity=0.28, stroke="amber", width=2.4, rx=3)
svg.text(kx + vw / 2, 156, "*c*  512", size=11.5, anchor="middle", color="orange", weight="700")
svg.text(RX + 168, 132, "唯一的 K", size=11.5, anchor="end", color="violet")
svg.text(RX + 168, 156, "唯一的 V", size=11.5, anchor="end", color="green")
svg.text(RX + 175, 190, "缓存里的 576 维直接充当 K，其中前 512 维兼作 V；", size=11.5, anchor="middle", color="orange")
svg.text(RX + 175, 206, "*W*^{UK} 并进查询，*W*^{UV} 挪到输出（§3）", size=11.5, anchor="middle", color="orange")
svg.text(RX + 16, 246, "每个（查询, 键）对：2 · 128 · (576 + 512)", size=12)
svg.text(RX + 16, 264, "= %s 次运算（MHA 模式的 3.4 倍）" % format(PAIR_MQA, ","), size=12.5, weight="700")
svg.text(RX + 16, 286, "每个 token 吸收、输出各一次：%s 次" % format(FIX, ","), size=12, color="muted")

# ---------------- bottom table
TY = 334
svg.text(20, TY, "一层的账单（DeepSeek-V2 维度）", size=13.5, weight="700")
cx = [20, 330, 490, 640]
for x, h in zip(cx, ["场景", "MHA 模式", "MQA 模式", "选哪个"]):
    svg.text(x, TY + 26, h, size=12, color="muted", weight="600")
svg.line(20, TY + 34, W - 20, TY + 34, color="rule", width=1)
rows = [
    ("预填充 T = 4096 个 token（因果，约 T²/2 对）", pre_mha, pre_mqa, "MHA 模式", "violet"),
    ("解码 1 个 token，缓存 T = 4096", dec_mha, dec_mqa, "MQA 模式", "blue"),
]
for r, (lab, a, b, win, col) in enumerate(rows):
    y = TY + 58 + r * 28
    svg.text(cx[0], y, lab, size=12.5)
    svg.text(cx[1], y, "%s GFLOPs" % format(round(a / 1e9, 1), ",") if a > 1e10 else "%.2f GFLOPs" % (a / 1e9),
             size=12.5, weight="700" if win == "MHA 模式" else None)
    svg.text(cx[2], y, "%s GFLOPs" % format(round(b / 1e9, 1), ",") if b > 1e10 else "%.3f GFLOPs" % (b / 1e9),
             size=12.5, weight="700" if win == "MQA 模式" else None)
    svg.text(cx[3], y, win, size=12.5, color=col, weight="700")
svg.line(20, TY + 100, W - 20, TY + 100, color="rule", width=1)
svg.text(20, TY + 122, "解码时 MHA 模式若不存展开后的 K、V，就得把 T 个潜向量全部重新展开；若存下来，缓存变成 71 倍。",
         size=11.5, color="muted")
svg.save("l09-two-modes.svg")
