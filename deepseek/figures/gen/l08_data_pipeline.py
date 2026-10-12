"""Figure 8.2: the iterative pipeline that recalls math web pages from Common Crawl.

Redrawn after DeepSeekMath v3 Figure 2 in the course colours (corpus/storage amber, model blue,
human step violet). All numbers from DeepSeekMath v3 section 2.1 (research notes C1-C5):
seed OpenWebMath, 500K positives + 500K Common Crawl negatives, fastText (dim 256, lr 0.1,
word n-gram <= 3, min count 3, 3 epochs), deduplicated Common Crawl of 40B HTML pages,
top 40B tokens kept in round 1, >10 % collected => math domain, 4 rounds, 35.5M pages,
120B tokens, ~98 % already collected by round 3, 10-gram decontamination.
Writes deepseek/figures/l08-data-pipeline.svg
"""

from l08_svg import SVG, box

W, H = 740, 440
svg = SVG(W, H, "从 Common Crawl 迭代召回数学网页的管线")

# pool of deduplicated Common Crawl pages (top)
svg.rect(470, 14, 254, 50, fill="amber", fill_opacity=0.16, stroke="amber", width=1.4, rx=8, dash="5 3")
svg.text(597, 35, "去重后的 Common Crawl", size=13, anchor="middle", color="amber", weight="600")
svg.text(597, 54, "40B 个 HTML 网页（单位：网页）", size=12, anchor="middle", color="ink")

R1, R2 = 96, 262          # top row y, bottom row y
BH = 98
xs = [16, 258, 500]
BW = 224

box(svg, xs[0], R1, BW, BH, ["种子语料", "第 1 轮：OpenWebMath", "取 500K 条作正例", "另取 CC 网页 500K 作负例"],
    color="amber", size=13, fill_opacity=0.12)
box(svg, xs[1], R1, BW, BH, ["① 训练 fastText 分类器", "向量维度 256，学习率 0.1", "词 n-gram ≤ 3，最小词频 3", "训练 3 个 epoch"],
    color="blue", size=13, fill_opacity=0.10)
box(svg, xs[2], R1, BW, BH, ["② 给每个网页打分、召回", "按分数排序，只留高分页", "第 1 轮保留 top 40B tokens", "（试过 40/80/120/160B）"],
    color="blue", size=13, fill_opacity=0.10)
box(svg, xs[2], R2, BW, BH, ["数学语料", "本轮召回的网页", "4 轮后：35.5M 个网页", "共 120B tokens"],
    color="amber", size=13, fill_opacity=0.12)
box(svg, xs[1], R2, BW, BH, ["③ 发现数学相关域名", "按域名（base URL）分组", "已被收集 > 10% 的域名", "例：mathoverflow.net"],
    color="ink", size=13, fill_opacity=0.05)
box(svg, xs[0], R2, BW, BH, ["④ 人工标注 URL 路径", "例：mathoverflow.net/questions", "路径下尚未被收集的网页", "作为新正例加入种子"],
    color="violet", size=13, fill_opacity=0.10)

mid1, mid2 = R1 + BH / 2, R2 + BH / 2
svg.arrow(xs[0] + BW, mid1, xs[1], mid1, color="ink", width=1.8, shorten=2)
svg.arrow(xs[1] + BW, mid1, xs[2], mid1, color="ink", width=1.8, shorten=2)
svg.arrow(597, 64, 597, R1, color="amber", width=1.8, shorten=2)
svg.arrow(xs[2] + BW / 2, R1 + BH, xs[2] + BW / 2, R2, color="ink", width=1.8, shorten=2)
svg.arrow(xs[2], mid2, xs[1] + BW, mid2, color="ink", width=1.8, shorten=2)
svg.arrow(xs[1], mid2, xs[0] + BW, mid2, color="ink", width=1.8, shorten=2)
svg.arrow(xs[0] + BW / 2, R2, xs[0] + BW / 2, R1 + BH, color="violet", width=2.0, shorten=2)
svg.text(xs[0] + BW / 2 + 10, (R1 + BH + R2) / 2 + 4, "下一轮种子更大、更好", size=12, color="violet")
svg.text(xs[2] + BW / 2 + 10, (R1 + BH + R2) / 2 + 4, "召回结果", size=12, color="muted")

# footer: stopping rule and decontamination
fy = R2 + BH + 26
svg.line(16, fy - 12, 724, fy - 12, color="rule", width=1)
svg.text(16, fy + 6, "停止条件：第 4 轮发现约 98% 的数据在第 3 轮已被收集，于是停在 4 轮。", size=12.5, color="ink")
svg.text(16, fy + 28, "去污染：凡与 GSM8K、MATH 等评测集有完全相同 10-gram 的文本段一律删除（短于 10-gram 的评测文本用精确匹配）。",
         size=12, color="muted")
svg.text(16, fy + 50, "注意：同一段原文里有两个“40B”：40B 个网页（候选池）与 40B tokens（第 1 轮保留量），单位不同。",
         size=12, color="red")

svg.save("l08-data-pipeline.svg")
