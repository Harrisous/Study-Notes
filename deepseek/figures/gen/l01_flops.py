"""Figure 1.19: forward FLOPs per token of GPT-2 small.

(a) T = 1024: weight matmuls 2 * 12 L d^2 = 169.87M, unembedding 2 d V = 77.19M, attention scores
    and weighted sums 2 L T d = 18.87M (Kaplan et al. 2020, Table 1 counting) or 4 L T d = 37.75M
    (PaLM Appendix B / Chinchilla counting). Values asserted against lesson-01-doc sec. 4.3.
(b) share of the attention term in (weight matmuls + attention) as a function of T (log axis):
    Kaplan T / (12 d + T), crossing 50% at T = 12 d = 9216; PaLM T / (6 d + T), crossing at 6 d = 4608.
Writes deepseek/figures/l01-flops.svg
"""

import math

import numpy as np

from l01_svg import SVG

L, d, V, T = 12, 768, 50257, 1024
w = 2 * 12 * L * d * d
unemb = 2 * d * V
att_k = 2 * L * T * d
att_p = 4 * L * T * d
assert (w, unemb, att_k, att_p) == (169_869_312, 77_194_752, 18_874_368, 37_748_736)
assert w + att_k + unemb == 265_938_432

W_, H_ = 760, 350
s = SVG(W_, H_, "GPT-2 small 每个 token 的前向运算量")

# ---------------------------------------------------------------- (a) stacked bar
s.text(10, 22, "(a) T = 1024 时每个 token 的前向 FLOPs", size=13, weight="700")
X0, X1 = 30, 330
scale = (X1 - X0) / 300e6
bars = [("权重矩阵乘 2·12Ld²", w, "blue", 0.45), ("反嵌入 2dV", unemb, "green", 0.45),
        ("注意力 2LTd（Kaplan 口径）", att_k, "violet", 0.55)]
y = 50
for name, v, col, op in bars:
    s.text(X0, y + 2, name, size=12, color="ink")
    s.rect(X0, y + 8, v * scale, 24, fill=col, fill_opacity=op, stroke=col, width=0.8)
    xe = X0 + (att_p if col == "violet" else v) * scale
    s.text(xe + 6, y + 25, "%.2fM" % (v / 1e6), size=12, color=col, weight="600")
    y += 56
# PaLM counting as dashed extension of the attention bar
s.rect(X0 + att_k * scale, y - 56 + 8, (att_p - att_k) * scale, 24, stroke="violet", width=1, dash="4 3")
s.text(X0, y + 4, "虚线：按 PaLM / Chinchilla 口径注意力为 4LTd = %.2fM" % (att_p / 1e6), size=11.5, color="violet")
# total
tot = w + unemb + att_k
s.line(X0, y + 22, X1 + 60, y + 22, color="rule", width=1)
s.text(X0, y + 44, "合计 %sM（Kaplan 口径，含反嵌入）" % ("%.2f" % (tot / 1e6)), size=12.5, weight="700")
s.text(X0, y + 64, "其中反嵌入占 %.0f%%；Kaplan 的 C_{forward} 不计这一项" % (100 * unemb / tot), size=12, color="muted")

# ---------------------------------------------------------------- (b) attention share vs T
R0, R1, Y0, Y1 = 452, 740, 58, 282
LX0, LX1 = 7, 17            # log2 T from 128 to 131072


def rx(t):
    return R0 + (math.log2(t) - LX0) / (LX1 - LX0) * (R1 - R0)


def ry(v):
    return Y1 - v * (Y1 - Y0)


s.line(410, 12, 410, 340, color="rule", width=1)
s.text(424, 22, "(b) 注意力项占“权重 + 注意力”的比例（d = 768）", size=13, weight="700")
for e in range(LX0, LX1 + 1, 2):
    t = 2 ** e
    s.line(rx(t), Y0, rx(t), Y1, color="grid", width=1)
    lab = "%dK" % (t // 1024) if t >= 1024 else str(t)
    s.text(rx(t), Y1 + 16, lab, size=11, anchor="middle", color="muted")
for v in (0, 0.25, 0.5, 0.75, 1):
    s.line(R0, ry(v), R1, ry(v), color="grid", width=1)
    s.text(R0 - 6, ry(v) + 4, "%d%%" % (100 * v), size=11, anchor="end", color="muted")
s.line(R0, Y1, R1, Y1, color="muted", width=1.1)
s.text((R0 + R1) / 2, Y1 + 34, "上下文长度 T（对数横轴，1K = 1024）", size=12, anchor="middle", color="muted")
ts = 2 ** np.linspace(LX0, LX1, 200)
kap = ts / (12 * d + ts)
palm = ts / (6 * d + ts)
s.path("M" + " L".join("%.1f %.1f" % (rx(a), ry(b)) for a, b in zip(ts, kap)), stroke="violet", width=2.2)
s.path("M" + " L".join("%.1f %.1f" % (rx(a), ry(b)) for a, b in zip(ts, palm)), stroke="violet", width=1.6,
       dash="5 4")
for tc, lab, dy in ((12 * d, "12d = 9,216", 16), (6 * d, "6d = 4,608", -14)):
    s.circle(rx(tc), ry(0.5), 3.6, fill="violet")
    s.text(rx(tc) + (8 if dy > 0 else -14), ry(0.5) + dy, lab, size=11.5, color="violet",
           anchor="start" if dy > 0 else "end")
s.line(rx(1024), Y0, rx(1024), Y1, color="ink", width=1, dash="3 3")
s.text(rx(1024) + 4, Y0 + 12, "GPT-2：T = 1024", size=11.5, color="ink")
s.circle(rx(1024), ry(1024 / (12 * d + 1024)), 3.2, fill="ink")
s.text(rx(1024) + 6, ry(0.10) + 14, "10.0%", size=11.5, color="ink")
s.text(R1 - 4, ry(0.115), "实线：Kaplan 口径 T/(12d+T)", size=11.5, color="violet", anchor="end")
s.text(R1 - 4, ry(0.045), "虚线：PaLM 口径 T/(6d+T)", size=11.5, color="violet", anchor="end")

s.save("l01-flops.svg")
