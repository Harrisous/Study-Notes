"""Figure 8.3: the corpus comparison of DeepSeekMath v3 Table 1.

DeepSeek-LLM 1.3B trained for 150B tokens on each math corpus, few-shot CoT.
Sizes use the DeepSeek tokenizer (vocabulary ~100K). Numbers transcribed in the
research notes section 3.2. Bars are computed to scale per panel.
Writes deepseek/figures/l08-corpus-bench.svg
"""

from l08_svg import SVG

corpora = [  # name, size (B tokens), GSM8K, MATH, CMATH, MMLU-STEM
    ("MathPile", 8.9, 2.7, 3.3, 1.2, 15.7),
    ("OpenWebMath", 13.6, 11.5, 8.9, 16.8, 29.6),
    ("Proof-Pile-2", 51.9, 14.3, 11.2, 19.9, 29.2),
    ("DeepSeekMath 语料", 120.2, 23.8, 13.6, 41.5, 33.1),
]
baseline = (None, 2.9, 3.0, 12.3, 19.5)   # "No Math Training"
panels = [("语料规模（B tokens）", 130.0, 0), ("GSM8K（%）", 30.0, 1), ("MATH（%）", 15.0, 2),
          ("CMATH（%）", 45.0, 3), ("MMLU-STEM（%）", 36.0, 4)]

W, H = 740, 300
LX = 136                 # label column
PW, GAP = 110, 10        # panel width, gap
Y0, RH, BHt = 64, 44, 24  # first row y, row height, bar thickness
svg = SVG(W, H, "不同数学语料训练 1.3B 模型的效果（论文表 1）")

for k, (name, *_rest) in enumerate(corpora):
    y = Y0 + k * RH
    col = "green" if k == 3 else "ink"
    svg.text(LX - 10, y + BHt / 2 + 5, name, size=12.5, anchor="end", color=col,
             weight="600" if k == 3 else None)

for p, (title, vmax, idx) in enumerate(panels):
    x0 = LX + p * (PW + GAP)
    svg.text(x0, Y0 - 28, title, size=12, color="ink", weight="600")
    svg.line(x0, Y0 - 12, x0, Y0 + len(corpora) * RH - 10, color="muted", width=1.1)
    if idx:
        b = baseline[idx]
        xb = x0 + b / vmax * (PW - 30)
        svg.line(xb, Y0 - 8, xb, Y0 + len(corpora) * RH - 10, color="red", width=1.3, dash="4 3")
    for k, row in enumerate(corpora):
        v = row[1 + idx]
        y = Y0 + k * RH
        w = v / vmax * (PW - 30)
        col = "amber" if idx == 0 else ("green" if k == 3 else "muted")
        op = 0.85 if (k == 3 or idx == 0) else 0.55
        svg.rect(x0, y, w, BHt, fill=col, fill_opacity=op, rx=2)
        svg.text(x0 + w + 4, y + BHt / 2 + 4.5, ("%.1f" % v), size=11.5,
                 color="green" if (k == 3 and idx) else "ink", weight="600" if k == 3 else None, halo=True)

yl = Y0 + len(corpora) * RH + 14
svg.line(LX, yl, LX + 22, yl, color="red", width=1.3, dash="4 3")
svg.text(LX + 28, yl + 4, "不做数学训练的 1.3B 基线（GSM8K 2.9、MATH 3.0、CMATH 12.3、MMLU-STEM 19.5）",
         size=12, color="muted")
svg.text(LX, yl + 26, "每个语料都训 150B tokens：OpenWebMath 要重复约 11 轮，DeepSeekMath 语料约 1.25 轮。", size=12, color="muted")

svg.save("l08-corpus-bench.svg")
