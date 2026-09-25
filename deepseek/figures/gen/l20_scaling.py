"""Figure 20.7 (summary of thread F): (a) training tokens per parameter, per total and per activated
parameter, against the Chinchilla rule of thumb of 20; (b) training compute estimated as 6 N_act D.

Data from l20_num.TOK (lesson-00-doc N20; GPT-3 from lesson 2 p8, whose 3.14e23 FLOPs is quoted
instead of the rounded 6 x 175B x 300B). Both panels are order-of-magnitude estimates: DeepSeek LLM
itself argues for M (non-embedding FLOPs per token) instead of 6N (lesson 6 p5).
Writes deepseek/figures/l20-scaling.svg
"""

import math

from l20_svg import SVG
from l20_num import TOK, DN

W, H = 760, 600
svg = SVG(W, H, "主线 F 的总账：每个参数喂多少 token，以及按 6ND 估的训练算力")

svg.text(10, 22, "(a) 训练 token 数 ÷ 参数量（对数刻度）", size=13, weight="700")
X0, X1 = 200, 720
LMIN, LMAX = 0, 3.6


def xs(v):
    return X0 + (math.log10(v) - LMIN) / (LMAX - LMIN) * (X1 - X0)


TOP, RH = 74, 30
for e in range(0, 4):
    x = xs(10 ** e)
    svg.line(x, TOP - 8, x, TOP + len(TOK) * RH - 8, color="grid", width=1)
    svg.text(x, TOP - 12, "{:,}".format(10 ** e), size=11, anchor="middle", color="muted")
svg.line(xs(20), TOP - 8, xs(20), TOP + len(TOK) * RH - 4, color="green", width=1.4, dash="5 3")
svg.text(xs(20), TOP + len(TOK) * RH + 8, "Chinchilla 经验值：每参数 20 token", size=11, anchor="middle", color="green")
svg.circle(X0 + 4, 38, 5, fill=None, stroke="violet", width=1.6)
svg.text(X0 + 14, 42, "÷ 总参数", size=12)
svg.circle(X0 + 94, 38, 5, fill="blue", stroke="blue")
svg.text(X0 + 104, 42, "÷ 激活参数（MoE）", size=12)
for i, t in enumerate(TOK):
    y = TOP + i * RH
    name = t[0]
    dn_tot, dn_act, _ = DN[name]
    svg.text(X0 - 10, y + 12, name, size=12, anchor="end")
    svg.line(xs(dn_tot), y + 8, xs(dn_act), y + 8, color="rule", width=2)
    svg.circle(xs(dn_tot), y + 8, 5, fill="surface", stroke="violet", width=1.6)
    if dn_act > dn_tot * 1.01:
        svg.circle(xs(dn_act), y + 8, 5, fill="blue", stroke="blue")
        svg.text(xs(dn_act) + 9, y + 12, "{:,.0f}".format(dn_act), size=11, color="blue")
        svg.text(xs(dn_tot) - 9, y + 12, "%.1f" % dn_tot, size=11, anchor="end", color="violet")
    else:
        svg.text(xs(dn_tot) + 9, y + 12, "%.1f" % dn_tot, size=11, color="violet")

Y2 = TOP + len(TOK) * RH + 40
svg.text(10, Y2, "(b) 训练算力 6 N_{act} D（FLOPs，对数刻度；只作量级比较）", size=13, weight="700")
LB0, LB1 = 22.5, 25.2


def xc(v):
    return X0 + (math.log10(v) - LB0) / (LB1 - LB0) * (X1 - X0)


T2 = Y2 + 30
for e in (23, 24, 25):
    x = xc(10 ** e)
    svg.line(x, T2 - 8, x, T2 + len(TOK) * 24 - 6, color="grid", width=1)
    svg.text(x, T2 - 12, "10^{%d}" % e, size=11, anchor="middle", color="muted")
for i, t in enumerate(TOK):
    y = T2 + i * 24
    name = t[0]
    c6 = 3.14e23 if name.startswith("GPT-3") else DN[name][2]
    if c6 < 10 ** LB0:
        svg.text(X0 - 10, y + 12, name, size=12, anchor="end")
        svg.text(X0 + 4, y + 12, "%.2f × 10^{%d}（在刻度以左）" % (c6 / 10 ** math.floor(math.log10(c6)),
                                                       math.floor(math.log10(c6))), size=11, color="muted")
        continue
    col = "muted" if name.startswith("GPT-3") or name.endswith("67B") else "blue"
    svg.text(X0 - 10, y + 12, name, size=12, anchor="end")
    svg.rect(X0, y + 2, xc(c6) - X0, 14, fill=col, fill_opacity=0.5, stroke=col, width=0.8)
    ex = math.floor(math.log10(c6))
    svg.text(xc(c6) + 6, y + 13, "%.2f × 10^{%d}" % (c6 / 10 ** ex, ex), size=11)

svg.text(10, H - 8, "数据：研究笔记（lesson-00-doc N20）；GPT-3 取第 2 课的 3.14 × 10^{23}。V4.1 的激活参数按 decode 的 16B。", size=12,
         color="muted")

svg.save("l20-scaling.svg")
