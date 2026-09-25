"""Figure 9.13: KV cache bytes per token across models (computed, log axis).

Elements per token = (cached numbers per layer) x layers; bytes = elements x bytes/element.
Sources (lesson-09-doc sec. 3.1, 3.3, 4.1):
  DeepSeek-V2 MLA     (512+64)*60,  bf16                      = 69,120 B
  DeepSeek-V2 deploy  (512+64)*60,  6 bit on average (sec. 3.2.3) = 25,920 B  (Fig. 1(b): 25.9 KB)
  DeepSeek-V2-Lite    (512+64)*27,  bf16                      = 31,104 B
  DeepSeek-V3         (512+64)*61,  bf16                      = 70,272 B  (ISCA'25 Table 1)
  DeepSeek-V3.2 FP8 format (FlashMLA README): 656 B per layer * 61  = 40,016 B
  DeepSeek LLM 67B    2*8*128*95,   bf16 (GQA-8)              = 389,120 B (Fig. 1(b): 389.1 KB)
  DeepSeek LLM 7B     2*32*128*30,  bf16 (MHA)                = 491,520 B
  DeepSeekMoE 16B     2*16*128*28,  bf16 (MHA)                = 229,376 B
  Qwen-2.5 72B        2*8*128*80,   bf16                      = 327,680 B (ISCA'25 Table 1)
  LLaMA-3.1 405B      2*8*128*126,  bf16                      = 516,096 B (ISCA'25 Table 1)
  V2 shape as MHA     2*128*128*60, bf16 (hypothetical)       = 3,932,160 B
KB = 1000 bytes.
Writes deepseek/figures/l09-kv-bytes.svg
"""

from l09_svg import SVG, LogAxis

rows = [
    ("DeepSeek-V2 的形状若用 MHA（假想）", 2 * 128 * 128 * 60 * 2, "hypo"),
    ("LLaMA-3.1 405B · GQA-8 · 126 层", 2 * 8 * 128 * 126 * 2, "gqa"),
    ("DeepSeek LLM 7B · MHA · 30 层", 2 * 32 * 128 * 30 * 2, "gqa"),
    ("DeepSeek LLM 67B · GQA-8 · 95 层", 2 * 8 * 128 * 95 * 2, "gqa"),
    ("Qwen-2.5 72B · GQA-8 · 80 层", 2 * 8 * 128 * 80 * 2, "gqa"),
    ("DeepSeekMoE 16B · MHA · 28 层", 2 * 16 * 128 * 28 * 2, "gqa"),
    ("DeepSeek-V3 · MLA · 61 层", (512 + 64) * 61 * 2, "mla"),
    ("DeepSeek-V2 · MLA · 60 层", (512 + 64) * 60 * 2, "mla"),
    ("DeepSeek-V3.2 · FlashMLA 的 FP8 格式", 656 * 61, "quant"),
    ("DeepSeek-V2-Lite · MLA · 27 层", (512 + 64) * 27 * 2, "mla"),
    ("DeepSeek-V2 部署态 · 平均 6 bit", (512 + 64) * 60 * 6 // 8, "quant"),
]
expect = [3_932_160, 516_096, 491_520, 389_120, 327_680, 229_376, 70_272, 69_120, 40_016, 31_104, 25_920]
assert [r[1] for r in rows] == expect

W = 740
ROW = 27
Y0 = 40
H = Y0 + ROW * len(rows) + 62
svg = SVG(W, H, "每 token 的 KV cache 字节数：各模型对比（对数横轴）")
svg.hatch("hq", "orange")
X0, X1 = 262, 650
xs = LogAxis(10_000, 10_000_000, X0, X1)
for v, lab in [(10_000, "10 KB"), (100_000, "100 KB"), (1_000_000, "1 MB"), (10_000_000, "10 MB")]:
    svg.line(xs(v), Y0 - 10, xs(v), Y0 + ROW * len(rows), color="grid", width=1)
    svg.text(xs(v), Y0 + ROW * len(rows) + 16, lab, size=11, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, Y0 + ROW * len(rows) + 38, "每 token 的 KV cache（全部层合计，字节，bf16 除非注明；对数刻度）", size=12,
         anchor="middle", color="muted")
svg.text(16, 22, "灰色 = MHA 或 GQA；橙色 = MLA；橙色斜纹 = MLA 再量化；虚框 = 假想配置", size=12, color="muted")


def human(b):
    if b >= 1e6:
        return "%.2f MB" % (b / 1e6)
    return "%.1f KB" % (b / 1e3)


for i, (name, b, kind) in enumerate(rows):
    y = Y0 + i * ROW
    col = {"hypo": "muted", "gqa": "muted", "mla": "orange", "quant": "orange"}[kind]
    svg.text(X0 - 10, y + 13, name, size=12, anchor="end",
             color="orange" if kind in ("mla", "quant") else "ink",
             weight="600" if "V2 · MLA" in name else None)
    w = xs(b) - X0
    if kind == "hypo":
        svg.rect(X0, y, w, 18, fill="muted", fill_opacity=0.08, stroke="muted", width=1.3, dash="5 3", rx=2)
    elif kind == "quant":
        svg.rect(X0, y, w, 18, fill="orange", fill_url="hq", stroke="orange", width=1.2, rx=2)
    else:
        svg.rect(X0, y, w, 18, fill=col, fill_opacity=0.55 if kind == "mla" else 0.45, rx=2)
    svg.text(X0 + w + 6, y + 13, human(b), size=11.5, color="ink")
svg.save("l09-kv-bytes.svg")
