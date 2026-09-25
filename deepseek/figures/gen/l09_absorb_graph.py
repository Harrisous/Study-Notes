"""Figure 9.6: compute graph of one head at one decoding step, before and after weight absorption.

Shapes use DeepSeek-V2 dims (d = 5120, d_h = 128, d_c = 512; arXiv:2405.04434v5 sec. 3.1.2).
FLOPs per layer (all 128 heads) per decoded token with L cached tokens (lesson-09-doc sec. 4.4):
  (a) expand:  2*L*512*(2*128*128) + 2*128*192*L + 2*128*128*L = 33,636,352 * L
  (c) absorb:  2*128*128*512 + 2*128*512*128 + (2*128*576 + 2*128*512) * L = 33,554,432 + 278,528 * L
The decoupled-RoPE part (64 dims) is included in the FLOP totals but not drawn.
Writes deepseek/figures/l09-absorb-graph.svg
"""

from l09_svg import SVG

L = 4096
fa = 2 * L * 512 * (2 * 128 * 128) + 2 * 128 * 192 * L + 2 * 128 * 128 * L
fc = 2 * 128 * 128 * 512 + 2 * 128 * 512 * 128 + (2 * 128 * 576 + 2 * 128 * 512) * L
assert fa == 33_636_352 * L and fc == 33_554_432 + 278_528 * L
print(fa / 1e9, fc / 1e9, fa / fc)

W, H = 760, 520
svg = SVG(W, H, "矩阵吸收前后，一个头在一次解码步里的计算图")


def node(x, y, w, h, top, bottom, col, cached=False):
    svg.rect(x, y, w, h, fill=col, fill_opacity=0.14, stroke="amber" if cached else col,
             width=2.4 if cached else 1.3, rx=6)
    svg.text(x + w / 2, y + h / 2 - 2, top, size=13, anchor="middle", color=col if col != "ink" else "ink",
             weight="700")
    svg.text(x + w / 2, y + h / 2 + 14, bottom, size=11.5, anchor="middle", color="muted")


# ------------------------------------------------------------------ (a) expand first
svg.text(16, 22, "① 每步重新展开：每解码一步，都把缓存里 T 个潜向量重新变回键和值", size=14, weight="700",
         color="red")
node(400, 36, 120, 40, "*q*^{C}_{t,i}", "[128]", "blue")
node(210, 92, 110, 42, "*K*_{i}", "[T, 128]", "violet")
node(400, 92, 120, 42, "分数 *K*_{i}*q*", "[T]", "ink")
node(20, 126, 116, 48, "缓存 *C*", "[T, 512]", "orange", cached=True)
node(210, 170, 110, 42, "*V*_{i}", "[T, 128]", "green")
node(400, 170, 120, 42, "*o*_{t,i} = *V*_{i}^{⊤}*p*", "[128]", "green")
node(590, 170, 150, 42, "× *W*^{O}_{i}", "[5120 × 128] → [5120]", "ink")
svg.arrow(460, 76, 460, 92, color="blue", shorten=0)
svg.arrow(320, 113, 400, 113, color="violet")
svg.arrow(136, 140, 210, 113, color="red", width=2.2)
svg.arrow(136, 160, 210, 191, color="red", width=2.2)
svg.text(150, 104, "×(*W*^{UK}_{i})^{⊤}", size=12, color="red")
svg.text(150, 214, "×(*W*^{UV}_{i})^{⊤}", size=12, color="red")
svg.arrow(460, 134, 460, 170, color="ink")
svg.text(468, 157, "softmax → *p*  [T]", size=11.5, color="muted")
svg.arrow(320, 191, 400, 191, color="green")
svg.arrow(520, 191, 590, 191, color="ink")
svg.text(20, 240, "红色两步每一步都要做 T 次：2 · T · 512 · 128 × 2 次运算，缓存越长越贵。", size=12.5, color="red")
svg.text(548, 58, "每层（128 头）每解码 1 个 token：", size=12, color="ink")
svg.text(548, 76, "33,636,352 × T 次运算", size=12, color="ink", weight="700")
svg.text(548, 100, "T = 4096：%.1f GFLOPs" % (fa / 1e9), size=13, color="red", weight="700")
svg.line(12, 262, W - 12, 262, color="rule", width=1)

# ------------------------------------------------------------------ (b) absorbed
y0 = 270
svg.text(16, y0 + 22, "③ 吸收：*W*^{UK}_{i} 挪到查询一侧，*W*^{UV}_{i} 挪到输出一侧，缓存原样使用", size=14,
         weight="700", color="blue")
node(20, y0 + 44, 116, 42, "*q*^{C}_{t,i}", "[128]", "blue")
node(210, y0 + 44, 110, 42, "*q̃*_{t,i}", "[512]", "blue")
node(400, y0 + 44, 120, 42, "分数 *C**q̃*", "[T]", "ink")
node(210, y0 + 112, 110, 48, "缓存 *C*", "[T, 512]", "orange", cached=True)
node(400, y0 + 176, 120, 42, "*õ*_{t,i} = *C*^{⊤}*p*", "[512]", "orange")
node(590, y0 + 176, 150, 42, "*W*^{UV}_{i}*õ* → × *W*^{O}_{i}", "[128] → [5120]", "ink")
svg.arrow(136, y0 + 65, 210, y0 + 65, color="blue", width=2.2)
svg.text(140, y0 + 56, "×(*W*^{UK}_{i})^{⊤}", size=12, color="blue")
svg.text(142, y0 + 84, "每步 1 次", size=12, color="blue")
svg.arrow(320, y0 + 65, 400, y0 + 65, color="blue")
svg.arrow(320, y0 + 126, 400, y0 + 80, color="orange")
svg.arrow(320, y0 + 146, 400, y0 + 190, color="orange")
svg.arrow(460, y0 + 86, 460, y0 + 176, color="ink")
svg.text(468, y0 + 136, "softmax → *p*  [T]", size=11.5, color="muted")
svg.arrow(520, y0 + 197, 590, y0 + 197, color="ink", width=2.2)
svg.text(530, y0 + 238, "×*W*^{UV}_{i}：每步 1 次", size=12, color="blue")
svg.text(20, y0 + 238, "缓存 *C* 被直接读两遍（打分、加权），从不展开。", size=12.5, color="orange")
svg.text(560, y0 + 72, "每层每解码 1 个 token：", size=12, color="ink")
svg.text(560, y0 + 90, "33,554,432 + 278,528 × T", size=12, color="ink", weight="700")
svg.text(560, y0 + 114, "T = 4096：%.3f GFLOPs" % (fc / 1e9), size=13, color="blue", weight="700")
svg.text(560, y0 + 134, "是 ① 的 1/%d" % round(fa / fc), size=12, color="blue")
svg.save("l09-absorb-graph.svg")
