"""Figure 10.15: where "KV cache -93.3 %" and "5.76x generation throughput" come from.

Left: per-token KV cache, DeepSeek 67B (GQA-8, 95 layers, bf16): 2*8*128*95*2 = 389,120 B;
V2 MLA in bf16: (512+64)*60*2 = 69,120 B; V2 as deployed (KV quantised to 6 bits on average):
34,560*6/8 = 25,920 B. 1 - 25920/389120 = 93.3 % (lesson-09-doc sec. 4.2; the attribution of 93.3 % to the
6-bit deployed cache is the author's inference from matching numbers).
Right (author's illustration): tokens of KV that fit in 8 x 80 GB after the weights
(67B bf16 weights 134 GB; V2 FP8 weights 236 GB or bf16 weights 472 GB), ignoring activations.
Writes deepseek/figures/l10-headline.svg
"""

import math

from l10_svg import SVG

kv = [("DeepSeek 67B", "GQA-8，bf16", 2 * 8 * 128 * 95 * 2, "muted"),
      ("V2：MLA", "bf16", (512 + 64) * 60 * 2, "orange"),
      ("V2：部署态", "平均 6 bit", (512 + 64) * 60 * 6 // 8, "orange")]
assert [k[2] for k in kv] == [389120, 69120, 25920]
mem = 8 * 80e9
cap = [("67B：bf16 权重 + bf16 KV", (mem - 134e9) / 389120, "muted"),
       ("V2：bf16 权重 + bf16 KV", (mem - 472e9) / 69120, "orange"),
       ("V2：FP8 权重 + 6 bit KV", (mem - 236e9) / 25920, "orange")]
assert abs(cap[0][1] / 1e6 - 1.30) < 0.01 and abs(cap[2][1] / 1e6 - 15.59) < 0.01

W, H = 760, 432
svg = SVG(W, H, "93.3% 与 5.76 倍从哪里来：每 token 的 KV 字节，以及 8 张 80 GB 显卡能装下多少 token")

# ---- left: waterfall of bytes per token
X0, Y0, Y1 = 40, 70, 330
svg.text(20, 26, "① 每个 token 的 KV cache（KB）", size=13, weight="600")
svg.text(20, 44, "按元素数 × 每元素位数计算", size=12, color="muted")
vmax = 400e3
for k, (name, sub, b, col) in enumerate(kv):
    cx = X0 + 50 + k * 110
    h = (Y1 - Y0) * b / vmax
    svg.rect(cx - 32, Y1 - h, 64, h, fill=col, fill_opacity=0.75 if k else 0.45)
    svg.text(cx, Y1 - h - 8, "%.1f" % (b / 1000), size=13, anchor="middle", weight="600")
    svg.text(cx, Y1 + 18, name, size=12, anchor="middle")
    svg.text(cx, Y1 + 34, sub, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X0 + 340, Y1, color="muted", width=1.2)
# step annotations
svg.text(X0 + 150, 150, "−82.2%", size=13, anchor="middle", color="red", weight="600")
svg.text(X0 + 150, 168, "MLA + 60 层", size=11, anchor="middle", color="muted")
svg.text(X0 + 262, 230, "−62.5%", size=13, anchor="middle", color="red", weight="600")
svg.text(X0 + 262, 248, "6 / 16 bit", size=11, anchor="middle", color="muted")
svg.text(X0 + 170, 396, "合计 1 − 25.92 / 389.12 = 93.3%", size=13, anchor="middle", weight="600")
lg = math.log(389120 / 69120) / math.log(389120 / 25920)
svg.text(X0 + 170, 416, "对数份额：MLA+层数 %.0f%%，量化 %.0f%%（笔者计算）" % (100 * lg, 100 * (1 - lg)), size=12,
         anchor="middle", color="muted")

# ---- right: capacity in 8 x 80 GB
RX0, RX1 = 420, 740
svg.text(RX0 - 10, 26, "② 8 × 80 GB 装下权重后，还能放多少 token 的 KV", size=13, weight="600")
svg.text(RX0 - 10, 44, "笔者示意：忽略激活与碎片，单位：百万 token", size=12, color="muted")
cmax = 20e6
for k, (name, v, col) in enumerate(cap):
    y = 90 + k * 78
    w = (RX1 - RX0) * v / cmax
    svg.text(RX0 - 10, y - 8, name, size=12)
    svg.rect(RX0 - 10, y, w, 30, fill=col, fill_opacity=0.75 if k == 2 else (0.45 if k == 0 else 0.4))
    svg.text(RX0 - 10 + w + 8, y + 20, "%.2fM" % (v / 1e6), size=13, weight="600")
svg.text(RX0 - 10, 330, "V2 部署态可缓存的 token 约为 67B 的 %.0f 倍" % (cap[2][1] / cap[0][1]), size=13,
         weight="600")
svg.text(RX0 - 10, 352, "能同时服务的序列更多，batch 更大；", size=12, color="muted")
svg.text(RX0 - 10, 370, "论文：单节点 8×H800 生成吞吐超过 50K token/s，", size=12, color="muted")
svg.text(RX0 - 10, 388, "是 67B 最大生成吞吐的 5.76 倍", size=12, color="muted")
svg.save("l10-headline.svg")
