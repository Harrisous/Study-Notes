"""Figure 2.6: GPT-3 training mixture (Brown et al. 2020, table 2.2), size share versus sampling weight.

Table 2.2 as printed: tokens (B), weight in training mix, epochs elapsed when training for 300B tokens.
  Common Crawl (filtered) 410 60% 0.44 | WebText2 19 22% 2.9 | Books1 12 8% 1.9 | Books2 55 8% 0.43 | Wikipedia 3 3% 3.4
Computed here: share of the 499B tokens, and weight * 300 / tokens (the epochs implied by the weight column).
The printed epoch column and the recomputed one disagree for WebText2 (2.9 vs 3.47) and Wikipedia (3.4 vs 3.00);
the printed weights sum to 101% (lesson-02-doc.md section 7, F9).
Writes deepseek/figures/l02-data-mix.svg
"""

from l02_svg import SVG

rows = [("Common Crawl（过滤后）", 410, 0.60, 0.44), ("WebText2", 19, 0.22, 2.9), ("Books1", 12, 0.08, 1.9),
        ("Books2", 55, 0.08, 0.43), ("Wikipedia", 3, 0.03, 3.4)]
tot = sum(r[1] for r in rows)
assert tot == 499

W, H = 720, 372   # last caption line sits at y = 362
svg = SVG(W, H, "GPT-3 的数据配比：数据量占比与训练采样权重")
LX, BX0, BX1 = 16, 186, 486          # label column, bar area
scale = (BX1 - BX0) / 0.85
ROW0, RH = 72, 46

svg.text(BX0, 24, "每个数据集：", size=12, color="ink", weight="600")
svg.rect(BX0 + 84, 14, 14, 12, fill="muted", fill_opacity=0.55)
svg.text(BX0 + 104, 24, "占总 token 数的比例", size=12, color="ink")
svg.rect(BX0 + 234, 14, 14, 12, fill="blue", fill_opacity=0.85)
svg.text(BX0 + 254, 24, "训练时的采样权重", size=12, color="ink")
svg.text(632, 50, "300B 时看了几遍", size=12, anchor="middle", color="muted", weight="600")
svg.text(596, 66, "论文印", size=11, anchor="middle", color="muted")
svg.text(672, 66, "按权重复算", size=11, anchor="middle", color="muted")

for v in (0, 0.2, 0.4, 0.6, 0.8):
    x = BX0 + v * scale
    svg.line(x, ROW0 - 16, x, ROW0 + len(rows) * RH - 12, color="grid", width=1)
    svg.text(x, ROW0 + len(rows) * RH + 4, "%d%%" % round(v * 100), size=11, anchor="middle", color="muted")

for i, (name, q, w, ep) in enumerate(rows):
    y = ROW0 + i * RH
    share = q / tot
    ep2 = w * 300 / q
    svg.text(LX, y + 8, name, size=13, color="ink", weight="600")
    svg.text(LX, y + 24, "%dB token" % q, size=12, color="muted")
    svg.rect(BX0, y - 8, share * scale, 13, fill="muted", fill_opacity=0.55)
    svg.text(BX0 + share * scale + 5, y + 3, "%.1f%%" % (share * 100), size=11, color="muted")
    svg.rect(BX0, y + 8, w * scale, 13, fill="blue", fill_opacity=0.85)
    svg.text(BX0 + w * scale + 5, y + 19, "%d%%" % round(w * 100), size=12, color="blue", weight="600")
    over = w > share
    svg.text(596, y + 12, "%.2g" % ep if ep < 1 else "%.1f" % ep, size=13, anchor="middle",
             color="green" if ep > 1 else "ink", weight="600")
    mism = abs(ep2 - ep) / ep > 0.08
    svg.text(672, y + 12, "%.2f" % ep2, size=13, anchor="middle", color="red" if mism else "muted",
             weight="600" if mism else None)

yb = ROW0 + len(rows) * RH + 24
svg.text(LX, yb, "蓝条长于灰条 = 被“过采样”：WebText2 只占 3.8% 的数据，却占 22% 的训练，看了约 3 遍；", size=12, color="ink")
svg.text(LX, yb + 18, "Common Crawl 占 82% 的数据，只占 60% 的训练，连一遍都没看完（0.44 遍）。", size=12, color="ink")
svg.text(LX, yb + 36, "红字：论文印的遍数与“权重 × 300B ÷ 数据量”复算不一致；印刷的五个权重加起来是 101%。", size=12, color="red")

svg.save("l02-data-mix.svg")
