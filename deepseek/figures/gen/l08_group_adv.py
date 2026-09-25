"""Figure 8.7: GRPO outcome-supervision advantages for four groups of G = 4 answers.

A_i = (r_i - mean(r)) / std(r). Bars use the population std (divide by G); the numbers under
each bar use the sample std (divide by G - 1), which is what TRL implements. The two differ by
the constant sqrt((G-1)/G) = 0.866 for G = 4 (research notes 2.3 and 4.1).
Writes deepseek/figures/l08-group-adv.svg and prints every value used in the lesson text.
"""

import numpy as np

from l08_svg import SVG

groups = [
    ("(a) 两对两错", [1, 0, 0, 1]),
    ("(b) 一对三错", [1, 0, 0, 0]),
    ("(c) 连续奖励", [0.9, 0.2, 0.4, 0.7]),
    ("(d) 全对", [1, 1, 1, 1]),
]

W, H = 740, 622
svg = SVG(W, H, "GRPO 组内优势的四个例子（G = 4）")
PW, PH = 344, 282


def fmtv(v):
    s = "%+.3f" % v
    return "0" if abs(v) < 5e-4 else s.replace("-", "−")


for k, (title, r) in enumerate(groups):
    r = np.array(r, dtype=float)
    mu, sp, ss = r.mean(), r.std(), r.std(ddof=1)
    Ap = (r - mu) / sp if sp > 0 else np.zeros_like(r)
    As = (r - mu) / ss if ss > 0 else np.zeros_like(r)
    print(title, "mean %.4f  pop std %.4f  sample std %.4f" % (mu, sp, ss),
          "A_pop", np.round(Ap, 4), "A_sample", np.round(As, 4))
    x0 = 16 + (k % 2) * (PW + 20)
    y0 = 14 + (k // 2) * (PH + 14)
    svg.rect(x0, y0, PW, PH, fill="surface", stroke="rule", width=1, rx=8)
    svg.text(x0 + 12, y0 + 22, title, size=13, weight="700")
    stat = "均值 %.2f · 总体 std %.3f · 样本 std %.3f" % (mu, sp, ss) if sp > 0 else "均值 1 · std = 0"
    svg.text(x0 + 12, y0 + 41, stat, size=11.5, color="muted")
    # rewards row
    cx = [x0 + 98 + j * 68 for j in range(4)]
    svg.text(x0 + 12, y0 + 72, "奖励 r", size=12, color="ink")
    for j, v in enumerate(r):
        col = "green" if v > 0 else "red"
        svg.rect(cx[j] - 26, y0 + 54, 52, 26, fill=col, fill_opacity=0.10 + 0.25 * v, stroke=col, width=1, rx=5)
        svg.text(cx[j], y0 + 72, ("%g" % v), size=13, anchor="middle", weight="600")
    # advantage bars around a zero line
    zy = y0 + 172
    scale = 34.0          # px per unit of advantage
    svg.text(x0 + 12, zy + 4, "优势 Â", size=12, color="ink")
    svg.line(x0 + 50, zy, x0 + PW - 12, zy, color="muted", width=1.1)
    for j, v in enumerate(Ap):
        h = v * scale
        col = "green" if v > 0 else "red"
        if abs(v) > 1e-9:
            svg.rect(cx[j] - 18, zy - max(h, 0), 36, abs(h), fill=col, fill_opacity=0.75, rx=2)
        ty = zy - h - 7 if v >= 0 else zy - h + 16
        if abs(v) < 1e-9:
            ty = zy - 8
        svg.text(cx[j], ty, fmtv(v), size=12, anchor="middle", color="ink", weight="600")
        svg.text(cx[j], y0 + PH - 12, "样本 " + fmtv(As[j]), size=11, anchor="middle", color="muted")
    if sp == 0:
        svg.text(x0 + PW / 2, zy + 40, "全部为 0：这道题不提供任何信号", size=12, anchor="middle", color="red")
        svg.text(x0 + PW / 2, zy + 58, "（公式里是 0/0，实现中分母加一个小数）", size=11, anchor="middle", color="muted")

svg.text(16, H - 8, "条高按总体标准差（除以 G）；每格底部的灰字是按样本标准差（除以 G − 1）算的值，两者只差常数 0.866。",
         size=12, color="muted")
svg.save("l08-group-adv.svg")
