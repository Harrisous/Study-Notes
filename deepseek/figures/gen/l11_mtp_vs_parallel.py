"""Figure 11.12: two kinds of multi-token prediction (schematic).

Left: Gloeckle et al. 2024 (arXiv:2404.19737): shared trunk f_s -> z_t, n independent heads f_{h_i}
(one transformer layer each), shared unembedding f_u; head i predicts x_{t+i} from z_t only (a marginal).
Parameter-matched: adding n-1 head layers removes n-1 trunk layers. Appendix B tried a causal variant
(heads stacked, no real tokens fed in): table S4, MBPP@1 31.9 vs parallel 33.8 (n = 4, 7B, 200B tokens).
Right: DeepSeek-V3 (arXiv:2412.19437 sec. 2.2): sequential modules; depth k also receives the embedding of
the TRUE token x_{t+k}, so it predicts x_{t+k+1} conditioned on x_{<=t+k} (complete causal chain).
V3 uses D_MTP = 1 (depth 2 drawn dashed). Extra parameters 11.6B (research notes sec. 4.1).
Writes deepseek/figures/l11-mtp-vs-parallel.svg
"""

from l11_svg import SVG, box

W, H = 720, 500
svg = SVG(W, H, "两种多 token 预测：并行独立头 与 顺序因果链")

# ---------------------------------------------------------------- left: parallel heads
svg.text(16, 24, "Ⓐ Gloeckle 等：共享主干 + n 个并行头", size=13, weight="600")
box(svg, 30, 360, 300, 48, "共享主干 f_{s}（读入 x_{1}…x_{t}）", color="blue", fill_opacity=0.08, size=12)
svg.arrow(180, 358, 180, 334, color="blue", width=1.5)
svg.text(188, 346, "*z*_{t}", size=12, color="blue")
hx = [34, 110, 186, 262]
svg.line(34 + 32, 330, 262 + 32, 330, color="blue", width=1.4)
for k, x in enumerate(hx):
    svg.arrow(x + 32, 330, x + 32, 305, color="blue", width=1.3)
    box(svg, x, 256, 64, 48, "头 %d" % (k + 1), color="violet", fill_opacity=0.12, size=12, sub="1 层", sub_size=11)
    svg.arrow(x + 32, 254, x + 32, 230, color="muted", width=1.3)
box(svg, 30, 196, 300, 32, "共享反嵌入 f_{u} → softmax", color="violet", fill_opacity=0.08, size=12)
for k, x in enumerate(hx):
    svg.arrow(x + 32, 194, x + 32, 170, color="muted", width=1.3)
    svg.rect(x + 6, 140, 52, 28, fill="green", fill_opacity=0.14, stroke="green", width=1.1, rx=4)
    svg.text(x + 32, 159, "x_{t+%d}" % (k + 1), size=12, anchor="middle")
svg.text(30, 94, "第 i 个头只看 *z*_{t}：学的是 p(x_{t+i} | x_{≤t})", size=12, color="violet")
svg.text(30, 112, "各头给定 *z*_{t} 后相互独立，彼此看不到对方的预测", size=12, color="muted")
svg.text(30, 432, "参数匹配：头多加 n − 1 层，主干就减 n − 1 层", size=12)
svg.text(30, 452, "附录 B 的“因果”变体（头串联、但不喂真实 token）：", size=12, color="muted")
svg.text(30, 470, "MBPP@1 31.9，低于并行头的 33.8（表 S4）", size=12, color="muted")

svg.line(360, 36, 360, 480, color="rule", width=1)

# ---------------------------------------------------------------- right: sequential causal chain
svg.text(376, 24, "Ⓑ DeepSeek-V3：顺序模块 + 完整因果链", size=13, weight="600")
box(svg, 380, 380, 116, 48, "主模型", color="blue", fill_opacity=0.08, size=12, sub="61 层", sub_size=11)
svg.text(380, 448, "读入 x_{1}…x_{t}", size=11, color="muted")
box(svg, 510, 290, 90, 48, "深度 1", color="green", fill_opacity=0.10, size=12, sub="TRM_{1}", sub_size=11)
box(svg, 614, 200, 90, 48, "深度 2", color="green", fill_opacity=0.04, size=12, sub="V3 未用", sub_size=11,
    dash="5 4")
# outputs
for (x, y, lab) in ((408, 300, "x_{t+1}"), (529, 210, "x_{t+2}"), (633, 120, "x_{t+3}")):
    svg.rect(x, y, 52, 28, fill="green", fill_opacity=0.14, stroke="green", width=1.1, rx=4)
    svg.text(x + 26, y + 19, lab, size=12, anchor="middle")
svg.arrow(434, 378, 434, 330, color="muted", width=1.3)
svg.arrow(555, 288, 555, 240, color="muted", width=1.3)
svg.arrow(659, 198, 659, 150, color="muted", width=1.3, dash="4 3")
# hidden-state chain
svg.path("M 496 404 H 540 V 339", stroke="blue", width=1.6, arrow=True)
svg.text(500, 394, "*h*^{(0)}_{t}", size=11, color="blue")
svg.path("M 600 314 H 640 V 249", stroke="blue", width=1.6, arrow=True, dash="4 3")
svg.text(604, 304, "*h*^{(1)}_{t}", size=11, color="blue")
# true-token embeddings
svg.rect(560, 404, 96, 28, fill="orange", fill_opacity=0.14, stroke="orange", width=1.2, rx=4)
svg.text(608, 420, "Emb(x_{t+1})", size=11, anchor="middle", color="orange")
svg.arrow(580, 403, 580, 339, color="orange", width=1.6)
svg.rect(620, 330, 92, 28, fill="orange", fill_opacity=0.08, stroke="orange", width=1.2, rx=4, dash="4 3")
svg.text(666, 346, "Emb(x_{t+2})", size=11, anchor="middle", color="orange")
svg.arrow(676, 329, 676, 249, color="orange", width=1.4, dash="4 3")
svg.text(376, 94, "深度 k 额外拿到*真实*的 x_{t+k} 的嵌入（橙色）：", size=12, color="orange")
svg.text(376, 112, "学的是 p(x_{t+k+1} | x_{≤t+k})，因果链不断", size=12, color="green")
svg.text(376, 470, "额外参数 11.6B；推理时丢掉，或当投机解码的草稿", size=12)

svg.save("l11-mtp-vs-parallel.svg")
