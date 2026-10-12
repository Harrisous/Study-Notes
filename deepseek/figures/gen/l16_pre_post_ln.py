"""Figure 16.2: Post-LN vs Pre-LN seen as a conveyor belt (mechanism figure).

Post-LN (Vaswani et al. 2017, sec. 3.1): x_{l+1} = LN(x_l + F(x_l)) -- the LayerNorm sits ON the belt,
so the direct path is multiplied by c_l = 1/std(x_l + F(x_l)) at every sublayer (HC paper eq. (16) writes
this as the n = 1 HC matrix [[0, c], [1, c]]). With unit, independent input and increment, c = 1/sqrt 2 and
60 sublayers give 2^-30 = 9.3e-10 (research notes N9, author's assumption).
Pre-LN: x_{l+1} = x_l + F(LN(x_l)) -- the belt only adds; its norm grows with depth (Xiong et al. 2020,
lemma 2: squared norm linear in l). Belt thickness below is schematic, proportional to sqrt(1 + l).
Writes deepseek/figures/l16-pre-post-ln.svg
"""

import math

from l16_svg import SVG

W, H = 760, 530
svg = SVG(W, H, "Post-LN 与 Pre-LN：LayerNorm 在传送带上，还是在工位里")

TOP = 62
STATIONS = [0, 1, 2]
DY = 100


def belt(x, y0, y1, w, color="ink"):
    svg.line(x, y0, x, y1, color=color, width=w, cap="butt")


def panel(x0, post):
    bx = x0 + 96          # belt x
    fx = x0 + 250         # branch column x
    title = "(a) Post-LN：LN 在传送带上" if post else "(b) Pre-LN：LN 在工位里"
    svg.text(x0 + 170, 28, title, size=14, anchor="middle", weight="700")
    svg.text(bx, TOP - 12, "*x*_{0}", size=13, anchor="middle")
    y = TOP
    for k in STATIONS:
        w = 4.5 if post else 4.0 * math.sqrt(1 + k)
        tap = TOP + 20 + DY * k
        plus = TOP + 74 + DY * k
        # belt segment down to the plus
        belt(bx, y, plus - 9, w)
        svg.circle(bx, tap, 3.4, fill="ink")
        if post:
            svg.path("M%d %d H%d V%d" % (bx, tap, fx, tap + 10), stroke="ink", width=1.5, arrow=True)
            svg.box(fx - 38, tap + 12, 76, 30, ["𝓕_{%d}" % (k + 1)], color="blue", size=14)
            svg.path("M%d %d V%d H%d" % (fx, tap + 42, plus, bx + 11), stroke="ink", width=1.5, arrow=True)
        else:
            svg.path("M%d %d H%d V%d" % (bx, tap, fx - 58, tap + 10), stroke="ink", width=1.5, arrow=True)
            svg.box(fx - 80, tap + 12, 44, 30, ["LN"], color="muted", size=13, fill_opacity=0.14)
            svg.arrow(fx - 36, tap + 27, fx - 20, tap + 27, color="ink", width=1.5)
            svg.box(fx - 18, tap + 12, 60, 30, ["𝓕_{%d}" % (k + 1)], color="blue", size=14)
            svg.path("M%d %d V%d H%d" % (fx + 12, tap + 42, plus, bx + 11), stroke="ink", width=1.5,
                     arrow=True)
        svg.circle(bx, plus, 9, fill="surface", stroke="ink", width=1.6)
        svg.text(bx, plus + 5, "+", size=15, anchor="middle", weight="700")
        if post:
            ln_y = plus + 14
            belt(bx, plus + 9, ln_y, w)
            svg.rect(bx - 24, ln_y, 48, 24, fill="red", fill_opacity=0.12, stroke="red", width=1.4, rx=5)
            svg.text(bx, ln_y + 17, "LN", size=13, anchor="middle", color="red", weight="600")
            svg.text(bx - 32, ln_y + 17, "× c_{%d}" % (k + 1), size=13, anchor="end", color="red")
            y = ln_y + 24
        else:
            y = plus + 9
            svg.text(bx - 16, plus + 5, "× 1", size=13, anchor="end", color="green")
    wlast = 4.5 if post else 4.0 * math.sqrt(1 + len(STATIONS))
    end = TOP + DY * len(STATIONS) + 22
    svg.line(bx, y, bx, end, color="ink", width=wlast, arrow=False)
    svg.arrow(bx, end - 2, bx, end + 14, color="ink", width=1.6)
    svg.text(bx + 14, end + 12, "……（共 L 个子层）", size=12, color="muted")
    # summary lines
    sy = 452
    if post:
        svg.text(x0 + 8, sy, "直通项：c_{1}c_{2}⋯c_{L}", size=13, color="red", weight="600")
        svg.text(x0 + 8, sy + 22, "例：每层 c = 1/√2，60 层后 ≈ 9.3 × 10^{-10}", size=12, color="red")
        svg.text(x0 + 8, sy + 44, "⇒ 底层梯度消失；训练易发散，要 warm-up", size=12)
        svg.text(x0 + 8, sy + 66, "带子宽度不变：每一站都被重新归一化", size=12, color="muted")
    else:
        svg.text(x0 + 8, sy, "直通项：1 × 1 × ⋯ × 1 = 1", size=13, color="green", weight="600")
        svg.text(x0 + 8, sy + 22, "梯度沿带子原样直达底层", size=12, color="green")
        svg.text(x0 + 8, sy + 44, "⇒ 但带子越来越粗（范数平方 ∝ 层数）", size=12)
        svg.text(x0 + 8, sy + 66, "深层工位放上去的增量占比越来越小", size=12, color="muted")


panel(10, True)
svg.line(385, 44, 385, 520, color="rule", width=1)
panel(392, False)

svg.save("l16-pre-post-ln.svg")
