"""Figure 0.5: where the published GPU hours went (thread D: how small post-training was, then was not).

Numbers (H800 GPU hours, thousands) from lesson-00-doc.md N8, N10 and lesson 15:
  DeepSeek-V3 (Table 1 of the V3 report): pre-training 2,664K, context extension 119K, post-training 5K
  DeepSeek-R1 (Table 7 of arXiv v2):      R1-Zero 101K, SFT data 5K, R1 41K
  DeepSeek-V3.2: post-training budget "exceeding 10% of the pre-training cost" (ratio only; the base is not
  published). The bar shows 10% of V3's 2,664K as an illustration, drawn open-ended.
Dollar figures use the reports' own assumption of $2 per GPU hour.

    python deepseek/figures/gen/l00_gpu_hours.py
"""

from __future__ import annotations

from l00_svg import SVG

ROWS = [
    ("DeepSeek-V3 的正式训练（2024-12）", [(2664, "ink", 0.45, "预训练 2,664K"), (119, "blue", 0.75, ""),
                                      (5, "green", 0.9, "")]),
    ("DeepSeek-R1（2025-01，建在 V3-Base 上）", [(101, "green", 0.9, ""), (5, "green", 0.5, ""),
                                         (41, "green", 0.7, "")]),
    ("DeepSeek-V3.2 的后训练（2025-12）", [(266.4, "green", 0.25, "")]),
]

W, H = 740, 360
L, R = 20, 700
X_MAX = 2900.0
TOP = 40
PITCH = 88
BAR_H = 30


def xb(v):
    return L + (R - L) * v / X_MAX


def main():
    s = SVG(W, H, title="公开的 GPU 小时：预训练与后训练")
    base = TOP + PITCH * len(ROWS) - 18
    for v in range(0, 2901, 500):
        x = xb(v)
        s.line(x, TOP - 6, x, base, color="grid", width=1)
        s.text(x, base + 16, "%s K" % f"{v:,}" if v else "0", size=12, anchor="middle", color="muted")
    s.text((L + R) / 2, base + 38, "H800 GPU 小时（千）", size=12.5, anchor="middle")
    for i, (title, segs) in enumerate(ROWS):
        y = TOP + i * PITCH
        s.text(L, y, title, size=13, weight=700)
        x = xb(0)
        for b, col, op, lab in segs:
            w = xb(b) - xb(0)
            dashed = i == 2
            s.rect(x, y + 8, w, BAR_H, fill=col, fill_opacity=op, stroke=col if dashed else "surface",
                   width=1.4 if dashed else 1, dash="5 3" if dashed else None)
            if lab:
                s.text(x + 10, y + 8 + BAR_H / 2 + 4.5, lab, size=12.5, color="surface", weight=700)
            x += w
        ty = y + 8 + BAR_H + 18
        if i == 0:
            s.text(x - 4, y - 1, "上下文扩展 119K（蓝）+ 后训练 5K（绿，只有预训练的 0.19%）", size=12, anchor="end",
                   color="green", weight=600)
            s.text(L, ty, "合计 2,788K 小时 × 2 美元 = 557.6 万美元；不含此前的研究与消融实验", size=12, color="muted")
        elif i == 1:
            s.text(x + 8, y + 8 + BAR_H / 2 + 4.5, "R1-Zero 101K + SFT 数据 5K + R1 41K = 147K 小时 = 29.4 万美元",
                   size=12, color="green", weight=600)
            s.text(L, ty, "约为 V3 全部正式训练的 5.3%；不含 V3-Base 本身、准备实验与蒸馏", size=12, color="muted")
        else:
            s.path("M%.1f %.1f l10 0 m-4 -4 l4 4 l-4 4" % (x + 2, y + 8 + BAR_H / 2), stroke="green", width=1.4)
            s.text(x + 18, y + 8 + BAR_H / 2 + 4.5, "“超过预训练成本的 10%”：按 V3 的 2,664K 折算至少 266K 小时",
                   size=12, color="green", weight=600)
            s.text(L, ty, "论文只给比例，没有给基数与口径；这根柱子是示意，右端开放", size=12, color="muted")
    s.save("l00-gpu-hours.svg")


if __name__ == "__main__":
    main()
