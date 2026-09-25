"""Figure 10.9: DeepSeek-V2 vs DeepSeek 67B training cost -- unit price, total bill, and efficiency.

Paper (V2 sec. 3.2.3): per trillion tokens on the H800 cluster, 67B needs 300.6K GPU hours, V2 172.8K
("saves 42.5%"). Totals are the author's estimates: unit price x tokens (V2 8.1T, 67B 2.0T), assuming a
constant rate. MFU estimates: tokens/s/GPU x FLOPs/token / 989 TFLOPS (H800 dense BF16, commonly quoted),
FLOPs/token = 6N (N = activated params: 21B vs 67B) plus attention (average context 2048):
V2 3*2*128*(192+128)*2048*60, 67B 3*2*64*(128+128)*2048*95  (lesson-10-doc sec. 4.4).
Writes deepseek/figures/l10-training-cost.svg
"""

from l10_svg import SVG

unit = {"67B": 300.6e3, "V2": 172.8e3}
tokens = {"67B": 2.0, "V2": 8.1}
total = {k: unit[k] * tokens[k] for k in unit}
save = 1 - unit["V2"] / unit["67B"]
assert abs(save - 0.4251) < 1e-4
tps = {k: 1e12 / (unit[k] * 3600) for k in unit}
six_n = {"67B": 6 * 67e9, "V2": 6 * 21e9}
attn = {"V2": 3 * 2 * 128 * (192 + 128) * 2048 * 60, "67B": 3 * 2 * 64 * (128 + 128) * 2048 * 95}
PEAK = 989e12
mfu_lo = {k: tps[k] * six_n[k] / PEAK for k in unit}
mfu_hi = {k: tps[k] * (six_n[k] + attn[k]) / PEAK for k in unit}
assert abs(mfu_lo["V2"] - 0.2048) < 1e-3 and abs(mfu_hi["67B"] - 0.3935) < 1e-3

W, H = 760, 400
svg = SVG(W, H, "DeepSeek-V2 与 DeepSeek 67B 的训练账：单价、总价与算力利用率")
COL = {"67B": "muted", "V2": "blue"}


def panel(x0, title, sub, vals, vmax, fmt_v, note=None, note_col="ink"):
    w, y0, y1 = 200, 70, 300
    svg.text(x0, 26, title, size=13, weight="600")
    svg.text(x0, 44, sub, size=12, color="muted")
    svg.line(x0, y1, x0 + w, y1, color="muted", width=1.2)
    for k, (name, v) in enumerate(vals):
        bw = 58
        cx = x0 + 50 + k * 100
        h = (y1 - y0) * v / vmax
        svg.rect(cx - bw / 2, y1 - h, bw, h, fill=COL[name], fill_opacity=0.75 if name == "V2" else 0.45)
        svg.text(cx, y1 - h - 7, fmt_v(v), size=12, anchor="middle", weight="600")
        svg.text(cx, y1 + 18, "DeepSeek 67B" if name == "67B" else "DeepSeek-V2", size=12, anchor="middle")
    if note:
        svg.text(x0 + w / 2, 350, note, size=13, anchor="middle", weight="600", color=note_col)


panel(20, "① 单价：每万亿 token 的 GPU 小时", "论文 §3.2.3（H800 集群实测）",
      [("67B", unit["67B"]), ("V2", unit["V2"])], 340e3, lambda v: "%.1fK" % (v / 1e3),
      "V2 省 %.1f%%" % (100 * save), "green")
panel(275, "② 总价：预训练全程的 GPU 小时", "笔者估算：单价 × token 数（2.0T / 8.1T）",
      [("67B", total["67B"]), ("V2", total["V2"])], 1.6e6, lambda v: "%.2fM" % (v / 1e6),
      "V2 是 67B 的 %.2f 倍" % (total["V2"] / total["67B"]), "red")
panel(530, "③ 算力利用率 MFU（笔者估算）", "6N 口径（括号内含注意力）",
      [("67B", mfu_lo["67B"]), ("V2", mfu_lo["V2"])], 0.48,
      lambda v: "%.1f%%" % (100 * v))
for k, name in enumerate(("67B", "V2")):
    cx = 530 + 50 + k * 100
    h = 230 * mfu_lo[name] / 0.48
    svg.text(cx, 300 - h - 24, "（%.1f%%）" % (100 * mfu_hi[name]), size=11, anchor="middle", color="muted")
svg.text(630, 350, "激活参数只有 31%，GPU 小时却是 57%", size=12, anchor="middle", color="ink", weight="600")
svg.text(630, 368, "MoE 的通信、不均衡与小矩阵吃掉了一部分", size=12, anchor="middle", color="muted")
svg.text(20, 392, "MFU = 每卡每秒 token 数 × 每 token FLOPs ÷ 989 TFLOPS（H800 稠密 BF16 的常用峰值）；只是量级估算。", size=12,
         color="muted")
svg.save("l10-training-cost.svg")
