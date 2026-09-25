"""Figure M5.4: perplexity = exp(average per-token surprise).

(a) Lesson 2's four-token example, p = 0.5, 0.25, 0.1, 0.8: mean 1.151 nats, PPL 3.16.
(b) Ten tokens, nine at p = 0.5 and one at p = 0.001: mean 1.315 nats, PPL 3.72 (all ten at 0.5 would give 2.00);
    the reciprocal of the arithmetic mean probability, 1/0.4501 = 2.22, hides the bad token.
Bar heights and all numbers computed here. Writes math/figures/m05-perplexity.svg
"""

import math

from m05_svg import SVG

A = [0.5, 0.25, 0.1, 0.8]
B = [0.5] * 9 + [0.001]

W, H = 740, 380
Y0, Y1 = 40, 300
YMAX = 7.5


def py(v):
    return Y1 - v / YMAX * (Y1 - Y0)


svg = SVG(W, H, "困惑度 = 平均意外的指数")


def panel(x0, x1, probs, title, bad=None):
    for v in range(0, 8):
        svg.line(x0, py(v), x1, py(v), color="grid", width=1)
        svg.text(x0 - 6, py(v) + 4, str(v), size=11, anchor="end", color="muted")
    svg.line(x0, Y1, x1, Y1, color="muted", width=1.2)
    svg.line(x0, Y0, x0, Y1, color="muted", width=1.2)
    svg.text((x0 + x1) / 2, 22, title, size=13, anchor="middle", weight="600")
    n = len(probs)
    slot = (x1 - x0) / n
    bw = min(34, slot * 0.66)
    s = [-math.log(p) for p in probs]
    for i, (p, v) in enumerate(zip(probs, s)):
        cx = x0 + slot * (i + 0.5)
        col = "red" if (bad is not None and i == bad) else "blue"
        svg.rect(cx - bw / 2, py(v), bw, Y1 - py(v), fill=col, fill_opacity=0.28, stroke=col, width=1.2)
        svg.text(cx, py(v) - 5, "%.2f" % v, size=11, anchor="middle", color=col)
        lab = ("%g" % p)
        svg.text(cx, Y1 + 16, lab, size=11, anchor="middle", color="muted")
    mean = sum(s) / n
    svg.line(x0, py(mean), x1, py(mean), color="ink", width=1.6, dash="6 4")
    return mean


ma = panel(62, 262, A, "(a) 第 2 课的 4 个 token")
svg.text(162, py(4.4), "平均 %.3f nats（虚线）" % ma, size=12, anchor="middle")
svg.text(162, py(4.4) + 18, "PPL = exp(%.3f) = %.2f" % (ma, math.exp(ma)), size=12, anchor="middle", weight="600")
svg.text(162, Y1 + 34, "模型给真实 token 的概率", size=12, anchor="middle", color="muted")

mb = panel(330, 700, B, "(b) 9 个 0.5，1 个 0.001", bad=9)
svg.text(480, py(mb) - 30, "平均 %.3f nats（虚线）→ PPL = exp(%.3f) = %.2f" % (mb, mb, math.exp(mb)), size=12,
         anchor="middle", weight="600")
svg.text(480, py(mb) - 12, "若第 10 个也是 0.5：PPL = 2.00", size=12, anchor="middle", color="muted")
svg.text(505, Y1 + 34, "模型给真实 token 的概率", size=12, anchor="middle", color="muted")
am = sum(B) / len(B)
svg.text(640, py(6.4), "一个 token 贡献了", size=12, anchor="end", color="red")
svg.text(640, py(6.4) + 16, "%.0f%% 的总意外" % (100 * -math.log(0.001) / sum(-math.log(p) for p in B)), size=12,
         anchor="end", color="red")
svg.text(480, py(4.7), "对照：1 ÷ 概率的算术平均 = 1/%.4f = %.2f" % (am, 1 / am), size=12,
         anchor="middle", color="muted")
svg.text(480, py(4.7) + 17, "算术平均几乎看不出那个坏 token", size=12, anchor="middle", color="muted")
svg.text(20, (Y0 + Y1) / 2, "每个 token 的意外 −ln p（nats）", size=12, anchor="middle", color="muted", rotate=-90)

svg.save("m05-perplexity.svg")
