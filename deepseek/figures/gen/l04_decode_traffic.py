"""Figure 4.7: what one decode step must move from HBM to the compute units (batch of B = 3 sequences).

Weights are read once per step and shared by all B sequences; each sequence reads only its own KV cache.
So the weight part has intensity 2 N_mm B / (2 N_mm) = B FLOP/byte (bf16) and the attention part has
4 l n_h d_h t B / (2 * 2 n_g d_h l t B) = n_h / n_g FLOP/byte, independent of B and t. Schematic; the
example numbers are Llama-2-7B on an A100 80GB SXM (lesson-04-doc §3.2, §4.4, §4.6).
Writes deepseek/figures/l04-decode-traffic.svg
"""

from l04_data import HW, MODELS, kv_bytes, n_mm_llama
from l04_svg import SVG

m = MODELS["Llama-2-7B"]
wgb = 2 * n_mm_llama(m) / 1e9
kvgb = kv_bytes(m) * 4096 / 1e9
assert abs(wgb - 13.21) < 0.01 and abs(kvgb - 2.147) < 0.001

W, H = 720, 470
svg = SVG(W, H, "decode 一步要从显存搬什么：权重被 batch 共用，KV cache 各读各的")

# HBM box
hx, hy, hw, hh = 16, 40, 270, 290
svg.rect(hx, hy, hw, hh, fill="amber", fill_opacity=0.05, stroke="muted", width=1.2, rx=10)
svg.text(hx + 12, hy - 10, "显存 HBM（片外，容量大，带宽有限）", size=13, weight="600")
# weights
svg.rect(hx + 14, hy + 16, hw - 28, 62, fill="muted", fill_opacity=0.25, stroke="muted", width=1.2, rx=6)
svg.text(hx + 26, hy + 40, "权重：s · N_{mm} 字节", size=13, weight="600")
svg.text(hx + 26, hy + 60, "Llama-2-7B：%.1f GB" % wgb, size=12, color="muted")
# KV caches of three sequences, different lengths
lens = [1.0, 0.62, 0.38]
for b, frac in enumerate(lens):
    y = hy + 104 + b * 46
    svg.text(hx + 14, y - 4, "序列 %d 的 KV cache" % (b + 1), size=12, color="amber")
    svg.rect(hx + 14, y, (hw - 28) * frac, 18, fill="amber", fill_opacity=0.45, stroke="amber", width=1, rx=3)
svg.text(hx + 14, hy + hh - 12, "每条 2 n_{g} d_{h} l s · t 字节（4096 token：%.2f GB）" % kvgb, size=11, color="muted")

# compute box
cx, cy, cw, ch = 470, 40, 234, 290
svg.rect(cx, cy, cw, ch, fill="blue", fill_opacity=0.05, stroke="blue", width=1.2, rx=10)
svg.text(cx + 12, cy - 10, "计算单元（片上，算得快）", size=13, weight="600", color="blue")
svg.text(cx + 14, cy + 30, "权重 × B 个新 token 的向量", size=12, weight="600")
svg.text(cx + 14, cy + 48, "FLOPs = B · 2N_{mm}", size=12)
for b in range(3):
    svg.rect(cx + 20 + b * 18, cy + 60, 12, 26, fill="blue", fill_opacity=0.5, rx=2)
svg.text(cx + 82, cy + 78, "← B = 3 个新 token", size=11, color="blue")
for b in range(3):
    y = cy + 112 + b * 46
    svg.text(cx + 14, y + 13, "q^{(%d)} 与序列 %d 自己的 K、V" % (b + 1, b + 1), size=12)
svg.text(cx + 14, cy + ch - 30, "每条 FLOPs = 4 l n_{h} d_{h} t", size=12, weight="600")
svg.text(cx + 14, cy + ch - 10, "（打分、加权求和各 2 l n_{h} d_{h} t）", size=11, color="muted")

# arrows through the bandwidth pipe
svg.rect(hx + hw + 6, cy + 100, cx - hx - hw - 12, 36, fill="ink", fill_opacity=0.05, rx=6)
svg.text((hx + hw + cx) / 2, cy + 94, "带宽 BW：A100 2.039 TB/s", size=12, anchor="middle", weight="600")
svg.arrow(hx + hw - 14, hy + 46, cx - 4, cy + 38, color="ink", width=3.2)
svg.text((hx + hw + cx) / 2 + 4, hy + 34, "读 1 次，B 条序列共用", size=12, anchor="middle")
for b in range(3):
    y = hy + 113 + b * 46
    svg.arrow(hx + 14 + (hw - 28) * lens[b] + 4, y, cx - 4, cy + 116 + b * 46, color="amber", width=1.8)
svg.text((hx + hw + cx) / 2 + 12, hy + 262, "各读各的，不能共用", size=12, anchor="middle", color="amber", weight="600")

# intensity summary
yb = 356
svg.rect(16, yb, 336, 96, fill="ink", fill_opacity=0.04, stroke="rule", width=1, rx=8)
svg.text(28, yb + 24, "权重部分的算术强度（bf16，s = 2）", size=13, weight="600")
svg.text(28, yb + 48, "B · 2 N_{mm} FLOPs ÷ 2 N_{mm} 字节 = B", size=13)
svg.text(28, yb + 72, "加大 batch 就能抬高：B 条序列分摊一次读取", size=12, color="muted")
svg.rect(368, yb, 336, 96, fill="amber", fill_opacity=0.06, stroke="amber", width=1, rx=8)
svg.text(380, yb + 24, "注意力部分的算术强度（bf16，s = 2）", size=13, weight="600", color="amber")
svg.text(380, yb + 48, "B · 4 l n_{h} d_{h} t ÷ B · 4 l n_{g} d_{h} t = n_{h} / n_{g}", size=13)
svg.text(380, yb + 72, "与 B、t 都无关：batch 救不了，只能减少 KV", size=12, color="muted")
svg.save("l04-decode-traffic.svg")
