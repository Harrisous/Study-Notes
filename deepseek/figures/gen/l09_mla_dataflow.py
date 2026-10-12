"""Figure 9.10: the complete MLA layer of DeepSeek-V2 with real dimensions.

Redrawn after the MLA part of Figure 2 of arXiv:2405.04434v5 (bottom-up there, top-down here),
with the shapes of appendix C, eqs. 37-47, and sec. 3.1.2: d = 5120, n_h = d_h = 128,
d_c = 512, d_c' = 1536, d_h^R = 64.  The cached latent is the RMS-normalised one: DeepSeek-V3's
inference/model.py writes kv_norm(kv) to kv_cache, which is what absorption needs (the norm divides
each token by its own RMS, so it cannot be folded into W^UK).  RMSNorm after both latents (sec. 3.1.2; in the HF code the
KV norm covers only the 512-dim c^KV, not k^R).  Parameter counts per matrix:
W^DQ 5120*1536, [W^UQ;W^QR] 1536*128*192, W^DKV 5120*512, W^KR 5120*64,
[W^UK;W^UV] 512*128*256, W^O 16384*5120  -> 149.2M with the two RMSNorm weight vectors.
Writes deepseek/figures/l09-mla-dataflow.svg
"""

from l09_svg import SVG

P = {
    "DQ": 5120 * 1536, "UQ": 1536 * 128 * 128, "QR": 1536 * 128 * 64, "DKV": 5120 * 512,
    "KR": 5120 * 64, "UK": 512 * 128 * 128, "UV": 512 * 128 * 128, "O": 128 * 128 * 5120,
}
total = sum(P.values()) + 1536 + 512
assert round(total / 1e6, 1) == 149.2, total
print(total)

W, H = 760, 590
svg = SVG(W, H, "DeepSeek-V2 的一层 MLA：完整数据流与真实维度")


def mp(n):
    return "%.1fM 参数" % (n / 1e6) if n >= 1e6 else "%.2fM 参数" % (n / 1e6)


def node(cx, y, w, h, title, sub, col, cached=False, fill_op=0.13, title_size=12.5):
    x = cx - w / 2
    svg.rect(x, y, w, h, fill=col, fill_opacity=fill_op, stroke="amber" if cached else col,
             width=2.6 if cached else 1.2, rx=5)
    if sub:
        svg.text(cx, y + h / 2 - 1, title, size=title_size, anchor="middle", color=col, weight="700")
        svg.text(cx, y + h / 2 + 13, sub, size=11, anchor="middle", color="muted")
    else:
        svg.text(cx, y + h / 2 + 4.5, title, size=title_size, anchor="middle", color=col, weight="700")
    return x, y, w, h


def down(x, y1, y2, col="ink", dash=None):
    svg.arrow(x, y1, x, y2, color=col, width=1.4, dash=dash)


# column centres
QC, QR_ = 84, 224          # query content / query rope
VC, KC = 376, 514          # value / key content
RC = 670                   # rope key
QM, KVM = 154, 445         # middle of query branch / kv branch

# row 0: input
node(380, 12, 230, 36, "输入 *h*_{t}  [5120]", None, "ink", fill_op=0.06, title_size=13)
svg.path("M380 48 L380 58 L%d 58 L%d 74" % (QM, QM), stroke="ink", width=1.3, arrow=True)
svg.path("M380 58 L%d 58 L%d 74" % (KVM, KVM), stroke="ink", width=1.3, arrow=True)
svg.path("M380 58 L%d 58 L%d 74" % (RC, RC), stroke="ink", width=1.3, arrow=True)

# row 1: down projections
node(QM, 76, 200, 40, "*W*^{DQ}：5120 → 1536", mp(P["DQ"]), "blue")
node(KVM, 76, 200, 40, "*W*^{DKV}：5120 → 512", mp(P["DKV"]), "orange")
node(RC, 76, 140, 40, "*W*^{KR}：5120 → 64", mp(P["KR"]), "red")
down(QM, 116, 132, "blue")
down(KVM, 116, 132, "orange")
down(RC, 116, 132, "red")

# row 2: latents / rope
node(QM, 134, 200, 34, "*c*^{Q}_{t}  [1536]", None, "blue", fill_op=0.2)
node(KVM, 134, 200, 34, "*c*^{KV}_{t}  [512]", None, "orange", fill_op=0.24)
node(RC, 134, 140, 34, "RoPE（位置 t）", None, "red", fill_op=0.08)
down(QM, 168, 182, "blue")
down(KVM, 168, 182, "orange")
down(RC, 168, 182, "red")

# row 3: norms / cached rope key
node(QM, 184, 200, 28, "RMSNorm", None, "muted", fill_op=0.08, title_size=12)
node(KVM, 184, 200, 28, "RMSNorm 之后缓存（512 维）", None, "orange", cached=True, fill_op=0.16, title_size=12)
node(RC, 184, 140, 34, "*k*^{R}_{t}  [64]  缓存", None, "red", cached=True, fill_op=0.16)
svg.path("M%d 212 L%d 222 L%d 222 L%d 236" % (QM, QM, QC, QC), stroke="blue", width=1.3, arrow=True)
svg.path("M%d 212 L%d 222 L%d 222 L%d 236" % (QM, QM, QR_, QR_), stroke="blue", width=1.3, arrow=True)
svg.path("M%d 212 L%d 222 L%d 222 L%d 236" % (KVM, KVM, VC, VC), stroke="orange", width=1.3, arrow=True)
svg.path("M%d 212 L%d 222 L%d 222 L%d 236" % (KVM, KVM, KC, KC), stroke="orange", width=1.3, arrow=True)

# row 4: up projections
node(QC, 238, 136, 42, "*W*^{UQ}：1536→128×128", mp(P["UQ"]), "blue", title_size=11.5)
node(QR_, 238, 136, 42, "*W*^{QR}：1536→128×64", mp(P["QR"]), "blue", title_size=11.5)
node(VC, 238, 132, 42, "*W*^{UV}：512→128×128", mp(P["UV"]), "green", title_size=11.5)
node(KC, 238, 132, 42, "*W*^{UK}：512→128×128", mp(P["UK"]), "violet", title_size=11.5)
down(QC, 280, 296, "blue")
down(QR_, 280, 296, "blue")
down(VC, 280, 296, "green")
down(KC, 280, 296, "violet")

# row 5: per-head vectors
node(QC, 298, 136, 34, "*q*^{C}_{t,i}  [128] ×128 头", None, "blue", fill_op=0.2, title_size=12)
node(QR_, 298, 136, 34, "RoPE→*q*^{R}_{t,i} [64] ×128", None, "red", fill_op=0.08, title_size=11.5)
node(VC, 298, 132, 34, "*v*^{C}_{t,i}  [128] ×128", None, "green", fill_op=0.2, title_size=11.5)
node(KC, 298, 132, 34, "*k*^{C}_{t,i}  [128] ×128", None, "violet", fill_op=0.2, title_size=11.5)
# k^R broadcast
svg.path("M%d 218 L%d 350 L%d 350 L%d 362" % (RC, RC, KC + 70, KC + 70), stroke="red", width=1.4, dash="5 3",
         arrow=True)
svg.text(RC + 6, 270, "同一个 *k*^{R}_{t}", size=11.5, color="red")
svg.text(RC + 6, 286, "复制给 128 个头", size=11.5, color="red")
svg.text(RC + 6, 302, "（MQA 式共享）", size=11.5, color="red")

# row 6: concat
svg.path("M%d 332 L%d 350 L%d 350 L%d 362" % (QC, QC, QM, QM), stroke="blue", width=1.3, arrow=True)
svg.path("M%d 332 L%d 350 L%d 350" % (QR_, QR_, QM), stroke="red", width=1.3)
svg.path("M%d 332 L%d 362" % (KC, KC), stroke="violet", width=1.3, arrow=True)
node(QM, 364, 250, 36, "*q*_{t,i} = [*q*^{C}_{t,i}; *q*^{R}_{t,i}]  [192]", None, "blue", fill_op=0.12)
node(KC + 42, 364, 240, 36, "*k*_{t,i} = [*k*^{C}_{t,i}; *k*^{R}_{t}]  [192]", None, "violet", fill_op=0.12)
svg.path("M%d 332 L%d 426" % (VC, VC), stroke="green", width=1.3, arrow=True)
down(QM, 400, 426, "blue")
down(KC + 42, 400, 426, "violet")

# row 7: attention
svg.rect(16, 428, W - 32, 58, fill="ink", fill_opacity=0.05, stroke="ink", width=1.3, rx=6)
svg.text(W / 2, 452, "注意力（每个头 i 各做一次，128 个头）：*o*_{t,i} = Σ_{j ≤ t} softmax_{j}( *q*_{t,i}^{⊤}*k*_{j,i} / √192 ) *v*^{C}_{j,i}",
         size=13, anchor="middle")
svg.text(W / 2, 474, "历史位置 j 的 *k*_{j,i}、*v*^{C}_{j,i} 由缓存里的 *c*^{KV}_{j} 和 *k*^{R}_{j} 得到（或者按 §3 吸收，根本不展开）",
         size=11.5, anchor="middle", color="muted")
down(380, 486, 502)
# row 8: output
node(380, 504, 300, 36, "*W*^{O}：128×128 = 16384 → 5120", mp(P["O"]), "ink", fill_op=0.06)
down(380, 540, 552)
node(380, 554, 220, 30, "输出 *u*_{t}  [5120]，加回残差流", None, "green", fill_op=0.12)
# legend
svg.rect(22, 558, 20, 14, fill="orange", fill_opacity=0.24, stroke="amber", width=2.6, rx=3)
svg.text(48, 570, "推理时缓存", size=11.5, color="amber")
svg.text(W - 20, 570, "整层 %.1fM 参数" % (total / 1e6), size=11.5, anchor="end", color="muted")
svg.save("l09-mla-dataflow.svg")
