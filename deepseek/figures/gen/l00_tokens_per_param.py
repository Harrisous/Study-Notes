"""Figure 0.7: training tokens per parameter, per TOTAL and per ACTIVATED parameter (thread F).

Tokens and parameters are the papers' own numbers (lesson-00-doc.md section 3.3; ratios recomputed as in N20):
  DeepSeek LLM 67B 2.0T / 67B; DeepSeekMoE 16B 2T / 16.4B, 2.8B; V2 8.1T / 236B, 21B; V2-Lite 5.7T / 15.7B, 2.4B;
  V3 14.8T / 671B, 37B; V4-Flash 32T / 284B, 13B; V4-Pro 33T / 1.6T, 49B; V4.1-Flash 45T (multimodal) / 552B, 16B
  (decode count). Chinchilla's "about 20 tokens per parameter" rule of thumb (lesson 2) is drawn for reference.

    python deepseek/figures/gen/l00_tokens_per_param.py
"""

from __future__ import annotations

import math

from l00_svg import SVG

ROWS = [
    ("DeepSeek LLM 67B（稠密）", 2.0e12, 67e9, 67e9),
    ("DeepSeekMoE 16B", 2.0e12, 16.4e9, 2.8e9),
    ("DeepSeek-V2", 8.1e12, 236e9, 21e9),
    ("DeepSeek-V2-Lite", 5.7e12, 15.7e9, 2.4e9),
    ("DeepSeek-V3", 14.8e12, 671e9, 37e9),
    ("DeepSeek-V4-Flash", 32e12, 284e9, 13e9),
    ("DeepSeek-V4-Pro", 33e12, 1600e9, 49e9),
    ("DeepSeek-V4.1-Flash", 45e12, 552e9, 16e9),
]

W, H = 740, 470
L, R = 190, 700
TOP = 30
PITCH = 46
BAR = 15
X0, X1 = 1, 4           # log10 tokens per parameter


def xv(v):
    return L + (R - L) * (math.log10(v) - X0) / (X1 - X0)


def main():
    s = SVG(W, H, title="每个参数分到多少训练 token")
    bottom = TOP + PITCH * len(ROWS)
    for e in range(X0, X1 + 1):
        for k in range(1, 10):
            v = k * 10 ** e
            if v <= 10 ** X1:
                x = xv(v)
                s.line(x, TOP - 6, x, bottom - 6, color="rule" if k == 1 else "grid", width=1 if k == 1 else 0.8)
        s.text(xv(10 ** e), bottom + 12, f"{10 ** e:,}", size=12, anchor="middle", color="muted")
    s.text((L + R) / 2, bottom + 34, "训练 token 数 ÷ 参数量（对数轴）", size=12.5, anchor="middle")
    x20 = xv(20)
    s.line(x20, TOP - 14, x20, bottom - 6, color="red", width=1.5, dash="5 4")
    s.text(x20 + 5, TOP - 12, "Chinchilla 经验值：约 20", size=12, color="red")
    for i, (name, d, n_tot, n_act) in enumerate(ROWS):
        y = TOP + i * PITCH
        s.text(L - 10, y + BAR + 4, name, size=12.5, anchor="end", weight=600)
        r_tot, r_act = d / n_tot, d / n_act
        s.rect(xv(10), y, xv(r_tot) - xv(10), BAR, fill="ink", fill_opacity=0.25, stroke="ink", width=0.8)
        s.text(xv(r_tot) + 5, y + BAR - 3, "%.1f（按总参数）" % r_tot, size=11.5, color="muted")
        if abs(r_act - r_tot) > 1e-9:
            s.rect(xv(10), y + BAR + 3, xv(r_act) - xv(10), BAR, fill="ink", fill_opacity=0.8)
            s.text(xv(r_act) + 5, y + 2 * BAR, f"{r_act:,.1f}（按激活参数）", size=11.5, color="ink", weight=600)
        print("%-24s %8.1f %9.1f" % (name, r_tot, r_act))
    s.save("l00-tokens-per-param.svg")


if __name__ == "__main__":
    main()
