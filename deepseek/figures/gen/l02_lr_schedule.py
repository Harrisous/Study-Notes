"""Figure 2.8: learning-rate schedules, computed from their formulas.

(a) GPT-3 175B (Brown et al. 2020, appendix B): peak 0.6e-4 (table 2.1); linear warmup over the first
    375M tokens; cosine decay to 10% of the peak over 260B tokens; then constant at 10% until 300B.
    The paper gives no formula; this is the course formula (2.16) with tau measured from the end of the
    warmup, plotted against tokens and ignoring the batch-size ramp (so it is an illustration of the
    written description). Values: 50B 5.53e-5, 100B 4.26e-5, 130B 3.31e-5, 200B 1.28e-5, 260B 6.0e-6.
(b) Vaswani et al. (2017) eq. (3): d_model^-0.5 * min(step^-0.5, step * warmup^-1.5), d = 512,
    warmup 4000: peak 6.99e-4 at step 4000, 1.40e-4 at step 100,000.
Writes deepseek/figures/l02-lr-schedule.svg
"""

import math

import numpy as np

from l02_svg import SVG, LogAxis

EMAX, WARM, COS_END, TOTAL = 0.6e-4, 0.375e9, 260e9, 300e9


def gpt3(D):
    if D <= WARM:
        return EMAX * D / WARM
    if D <= COS_END:
        tau = (D - WARM) / (COS_END - WARM)
        return EMAX * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * tau)))
    return 0.1 * EMAX


def vaswani(step, d=512, w=4000):
    return d ** -0.5 * min(step ** -0.5, step * w ** -1.5)


assert abs(gpt3(130e9) - 3.306e-5) < 1e-8 and abs(vaswani(4000) - 6.988e-4) < 1e-7

W, H = 720, 350
svg = SVG(W, H, "学习率调度：GPT-3 的 warmup + 余弦，以及 Transformer 原论文的 warmup + 平方根倒数")

# ---------------- panel (a)
AX0, AX1, AY0, AY1 = 70, 410, 52, 300
ax = LogAxis(0, 300, AX0, AX1, log=False)
ay = LogAxis(0, 7e-5, AY1, AY0, log=False)
svg.text(AX0 - 50, 22, "(a) GPT-3 175B：warmup 375M token → 余弦降到 10% → 保持", size=13, color="ink", weight="600")
for v in range(0, 8):
    y = ay(v * 1e-5)
    svg.line(AX0, y, AX1, y, color="grid", width=1)
    svg.text(AX0 - 6, y + 4, "%d×10^{−5}" % v if v else "0", size=11, anchor="end", color="muted")
for v in range(0, 301, 50):
    x = ax(v)
    svg.line(x, AY0, x, AY1, color="grid", width=1)
    svg.text(x, AY1 + 16, "%dB" % v, size=11, anchor="middle", color="muted")
svg.line(AX0, AY1, AX1, AY1, color="muted", width=1.2)
svg.line(AX0, AY0, AX0, AY1, color="muted", width=1.2)
svg.text((AX0 + AX1) / 2, AY1 + 36, "已训练的 token 数（忽略 batch 渐增）", size=12, anchor="middle", color="muted")

Ds = np.concatenate([np.linspace(0, 2e9, 80), np.linspace(2e9, TOTAL, 300)])
svg.path("M" + " L".join("%.1f %.1f" % (ax(D / 1e9), ay(gpt3(D))) for D in Ds), stroke="blue", width=2.6)
svg.line(ax(260), 196, ax(260), AY1, color="muted", width=1, dash="3 3")
svg.text(ax(260) - 4, 214, "260B：余弦结束", size=11, color="muted", anchor="end")
for D, dx, dy, anc in ((50e9, 6, -8, "start"), (130e9, -8, 18, "end"), (200e9, 6, -8, "start")):
    x, y = ax(D / 1e9), ay(gpt3(D))
    svg.circle(x, y, 3.5, fill="blue")
    svg.text(x + dx, y + dy, "%.2f×10^{−5}" % (gpt3(D) / 1e-5), size=11, anchor=anc, color="blue")
svg.text(ax(282), ay(0.6e-5) - 8, "6×10^{−6}", size=11, anchor="middle", color="blue")

# inset: the first 1B tokens
IX0, IX1, IY0, IY1 = 262, 394, 84, 146
ix = LogAxis(0, 1.0, IX0, IX1, log=False)
iy = LogAxis(0, 7e-5, IY1, IY0, log=False)
svg.rect(IX0 - 30, IY0 - 22, IX1 - IX0 + 40, IY1 - IY0 + 44, fill="surface", fill_opacity=1, stroke="rule", rx=6)
svg.text(IX0 - 30, IY0 - 8, "放大前 1B token", size=11, color="muted")
svg.line(IX0, IY1, IX1, IY1, color="muted", width=1)
svg.line(IX0, IY0, IX0, IY1, color="muted", width=1)
xs = np.linspace(0, 1e9, 100)
svg.path("M" + " L".join("%.1f %.1f" % (ix(D / 1e9), iy(gpt3(D))) for D in xs), stroke="blue", width=2)
svg.line(ix(0.375), IY0, ix(0.375), IY1, color="amber", width=1, dash="3 3")
svg.text(ix(0.375), IY1 + 14, "0.375B", size=11, anchor="middle", color="amber")
svg.text(IX1, IY1 + 14, "1B", size=11, anchor="end", color="muted")

# ---------------- panel (b)
BX0, BX1, BY0, BY1 = 500, 704, 52, 300
bx = LogAxis(0, 100000, BX0, BX1, log=False)
by = LogAxis(0, 8e-4, BY1, BY0, log=False)
svg.text(BX0 - 44, 22, "(b) Transformer 原论文 (d = 512)", size=13, color="ink", weight="600")
for v in range(0, 9, 2):
    y = by(v * 1e-4)
    svg.line(BX0, y, BX1, y, color="grid", width=1)
    svg.text(BX0 - 6, y + 4, "%d×10^{−4}" % v if v else "0", size=11, anchor="end", color="muted")
for v in (0, 25000, 50000, 75000, 100000):
    x = bx(v)
    svg.line(x, BY0, x, BY1, color="grid", width=1)
    svg.text(x, BY1 + 16, "%dk" % (v // 1000), size=11, anchor="middle", color="muted")
svg.line(BX0, BY1, BX1, BY1, color="muted", width=1.2)
svg.line(BX0, BY0, BX0, BY1, color="muted", width=1.2)
svg.text((BX0 + BX1) / 2, BY1 + 36, "训练步数", size=12, anchor="middle", color="muted")
steps = np.concatenate([np.linspace(1, 4000, 60), np.linspace(4000, 100000, 200)])
svg.path("M" + " L".join("%.1f %.1f" % (bx(s), by(vaswani(s))) for s in steps), stroke="violet", width=2.6)
svg.circle(bx(4000), by(vaswani(4000)), 3.5, fill="violet")
svg.text(bx(4000) + 8, by(vaswani(4000)) + 4, "第 4000 步：6.99×10^{−4}", size=11, color="violet")
svg.circle(bx(100000), by(vaswani(100000)), 3.5, fill="violet")
svg.text(bx(100000), by(vaswani(100000)) - 10, "1.40×10^{−4}", size=11, anchor="end", color="violet")
svg.text(bx(52000), by(4.2e-4), "之后按 1/√step 衰减", size=12, anchor="middle", color="violet")

svg.save("l02-lr-schedule.svg")
