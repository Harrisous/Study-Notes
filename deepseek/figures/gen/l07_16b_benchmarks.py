"""Figure 7.15: DeepSeekMoE 16B vs DeepSeek 7B vs LLaMA2 7B on ten benchmarks (dot plot).

DeepSeekMoE arXiv v1 Tables 3 and 4 (research notes section 5.2, figure 7.11 there).
Order of values: (LLaMA2 7B, DeepSeek 7B, DeepSeekMoE 16B). FLOPs per 4K tokens:
187.9T, 183.5T, 74.4T. Rows where 16B is below DeepSeek 7B are flagged.
Writes deepseek/figures/l07-16b-benchmarks.svg
"""

from l07_svg import SVG

rows = [
    ("HellaSwag", 75.6, 75.4, 77.1),
    ("PIQA", 78.0, 79.2, 80.2),
    ("ARC-challenge", 49.0, 48.1, 49.8),
    ("GSM8K", 15.5, 17.4, 18.8),
    ("HumanEval", 14.6, 26.2, 26.8),
    ("MBPP", 21.8, 39.0, 39.2),
    ("TriviaQA", 63.8, 59.7, 64.8),
    ("NaturalQuestions", 25.5, 22.2, 25.5),
    ("MMLU", 45.8, 48.2, 45.0),
    ("CEval", 33.9, 45.0, 40.6),
]

W = 740
RH = 30
Y0 = 92
H = Y0 + RH * len(rows) + 76
X0, X1 = 170, 640
VMAX = 90.0


def px(v):
    return X0 + v / VMAX * (X1 - X0)


svg = SVG(W, H, "DeepSeekMoE 16B 与两个 7B 稠密模型在十项基准上的得分")
svg.text(16, 22, "每 4K token 的计算量：LLaMA2 7B 187.9T · DeepSeek 7B 183.5T · DeepSeekMoE 16B 74.4T（约 40%）", size=13,
         weight="600")
# legend
lx = 16
for lab, kind in (("LLaMA2 7B", "sq"), ("DeepSeek 7B", "ring"), ("DeepSeekMoE 16B", "dot")):
    if kind == "sq":
        svg.rect(lx, 40, 11, 11, fill="muted", fill_opacity=0.7)
    elif kind == "ring":
        svg.circle(lx + 6, 46, 5.5, fill="surface", stroke="ink", width=1.8)
    else:
        svg.circle(lx + 6, 46, 6, fill="blue")
    svg.text(lx + 18, 50, lab, size=12)
    lx += 18 + len(lab) * 8 + 30
svg.text(W - 16, 50, "分数越高越好", size=12, anchor="end", color="muted")

for v in range(0, 91, 10):
    x = px(v)
    svg.line(x, Y0 - 16, x, Y0 + RH * len(rows) - 12, color="grid", width=1)
    svg.text(x, Y0 + RH * len(rows) + 4, str(v), size=11, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, Y0 + RH * len(rows) + 24, "得分", size=12, anchor="middle", color="muted")

for i, (name, ll, ds, moe) in enumerate(rows):
    y = Y0 + i * RH
    behind = moe < ds
    if behind:
        svg.rect(10, y - 14, W - 20, RH - 4, fill="red", fill_opacity=0.07, rx=3)
    svg.text(X0 - 12, y + 4, name, size=12, anchor="end", color="red" if behind else "ink")
    lo, hi = min(ll, ds, moe), max(ll, ds, moe)
    svg.line(px(lo), y, px(hi), y, color="rule", width=2)
    svg.rect(px(ll) - 5.5, y - 5.5, 11, 11, fill="muted", fill_opacity=0.7)
    svg.circle(px(ds), y, 5.5, fill="surface", stroke="ink", width=1.8)
    svg.circle(px(moe), y, 6, fill="blue")
    diff = moe - ds
    svg.text(X1 + 12, y + 4, "16B %.1f（%+.1f）" % (moe, diff), size=12, color="red" if behind else "blue")

svg.text(16, H - 28, "右列括号内是 DeepSeekMoE 16B 减 DeepSeek 7B 的差值。", size=12, color="muted")
svg.text(16, H - 10, "红底：16B 落后于 DeepSeek 7B 的两项（都是多项选择题式的考试），论文归因于注意力参数少。", size=12,
         color="muted")

svg.save("l07-16b-benchmarks.svg")
