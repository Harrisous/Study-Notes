"""Figure 19.15 (closing figure of thread A): KV bytes per token from GPT-3 to DeepSeek-V4.1-Flash.

Each bar is computed in l19_num from the model's published shape at its deployed precision:
  GPT-3 175B        2 x 96 layers x 96 heads x 128 x 2 B (FP16/BF16 assumed; paper gives no inference precision)
  Llama-3-70B       2 x 80 x 8 KV heads x 128 x 2 B (BF16, GQA-8)
  DeepSeek LLM 67B  2 x 95 x 8 x 128 x 2 B (BF16, GQA-8)
  DeepSeek-V2       60 x (512 + 64) x 2 B (BF16); lighter sub-bar: 6 bits/element as deployed
  DeepSeek-V3       61 x 576 x 2 B (BF16)
  DeepSeek-V3.2     61 x (656 + 132) B (FP8 MLA row + FP8 indexer key)
  DeepSeek-V4-Pro   (30/4 + 31/128) x 584 + 30/4 x 68 B
  DeepSeek-V4-Flash (21/4 + 20/128) x 584 + 21/4 x 68 B
  DeepSeek-V4.1     2.5 x (288 + 68) B
Log-scale x axis; the dimension each step saved on is written next to the bar.
Writes deepseek/figures/l19-kv-staircase.svg
"""

import math

from l19_svg import SVG
from l19_num import GPT3, LLAMA3_70B, DS67B, V2_BF16, V2_6BIT, V3_BF16, V32, G4P, G4F, G41

W, H = 760, 540
svg = SVG(W, H, "主线 A 的总结：每 token KV 字节，从 GPT-3 到 DeepSeek-V4.1-Flash")

ROWS = [
    ("GPT-3 175B", "MHA · 96 层 × 96 头", "按 2 字节", GPT3, None, "起点"),
    ("Llama-3-70B", "GQA-8 · 80 层", "BF16", LLAMA3_70B, None, "头数：96 → 8 组"),
    ("DeepSeek LLM 67B", "GQA-8 · 95 层", "BF16", DS67B, None, "头数"),
    ("DeepSeek-V2", "MLA · 60 层 × 576", "BF16", V2_BF16, V2_6BIT, "条目大小（MLA）"),
    ("DeepSeek-V3", "MLA · 61 层 × 576", "BF16", V3_BF16, None, "条目大小"),
    ("DeepSeek-V3.2", "MLA + 索引键 · 61 层", "FP8+BF16", V32, None, "精度（FP8）"),
    ("DeepSeek-V4-Pro", "30 CSA + 31 HCA", "FP8+BF16", G4P, None, "序列维（压缩）"),
    ("DeepSeek-V4-Flash", "21 CSA + 20 HCA", "FP8+BF16", G4F, None, "序列维"),
    ("DeepSeek-V4.1-Flash", "4 份 CSA2", "FP4", G41, None, "层维 + 精度（FP4）"),
]
PX0, PX1 = 200, 470
LMIN, LMAX = 2, 7


def xs(b):
    return PX0 + (math.log10(b) - LMIN) / (LMAX - LMIN) * (PX1 - PX0)


TOP, RH = 58, 44
for e in range(LMIN, LMAX + 1):
    svg.line(xs(10 ** e), TOP - 10, xs(10 ** e), TOP + len(ROWS) * RH, color="grid", width=1)
    svg.text(xs(10 ** e), TOP - 16, "10^{%d}" % e if e > 3 else "{:,}".format(10 ** e), size=11, anchor="middle",
             color="muted")
svg.text(PX0 + 2, 22, "每 token 的 KV 字节（对数刻度；按各模型的部署精度）· 是 V4.1 的几倍", size=12, color="muted")

for i, (name, shape, prec, b, sub, dim) in enumerate(ROWS):
    y = TOP + i * RH
    last = name.endswith("V4.1-Flash")
    svg.text(PX0 - 10, y + 13, name, size=12, anchor="end", weight="700" if last else None,
             color="orange" if last else "ink")
    svg.text(PX0 - 10, y + 29, "%s · %s" % (shape, prec), size=11, anchor="end", color="muted")
    col = "orange" if last else "amber"
    svg.rect(xs(100), y + 4, xs(b) - xs(100), 20, fill=col, fill_opacity=0.75 if last else 0.4, stroke=col, width=0.8)
    if sub:
        svg.rect(xs(100), y + 26, xs(sub) - xs(100), 8, fill="amber", fill_opacity=0.25, stroke="amber", width=0.6)
        svg.text(xs(sub) + 5, y + 34, "部署时 6 比特：%s" % "{:,}".format(round(sub)), size=11, color="muted")
    val = "{:,.0f}".format(b) if b >= 1000 else "%d" % b
    ratio = b / G41
    rtxt = "×%s" % ("{:,.0f}".format(ratio) if ratio >= 100 else "%.2f" % ratio if ratio < 10 else "%.1f" % ratio)
    svg.text(xs(b) + 6, y + 19, "%s B · %s" % (val, rtxt if not last else "×1"), size=12,
             weight="700" if last else None)
    svg.text(596, y + 19, dim, size=12, color="blue" if i else "muted")

svg.text(596, TOP - 16, "这一级省在哪个维度", size=12, color="blue")
FY = TOP + len(ROWS) * RH + 22
svg.text(20, FY, "V4 系列与 V4.1 只计全局 KV（主 KV + 索引键），不含每请求常数的 SWA 状态；V3.2 起的数字含索引器键。",
         size=12, color="muted")
svg.text(20, FY + 20, "1M token 的一个请求：GPT-3 4,608 GiB → DeepSeek 67B 380 GiB → V3 68.6 GiB → V3.2 46.9 GiB → V4-Flash 3.43 GiB → V4.1 0.87 GiB。",
         size=12, color="muted")
svg.text(20, FY + 40, "GPT-3 按全部 96 层全量缓存、2 字节计，是上界（论文未写推理精度；一半层是局部带状稀疏注意力）。",
         size=12, color="muted")

svg.save("l19-kv-staircase.svg")
