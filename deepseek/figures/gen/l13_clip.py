"""Figure 13.5: where the GRPO/PPO clip stops a token from learning.

For each sign of the advantage and each clip ratio (the common PPO value 0.2 and R1's stage-1 value 10),
the ratio axis rho > 0 is split into the part where the clipped term l_clip = min(rho*A, clip(rho)*A) still
depends on rho (gradient flows) and the part where l_clip is constant (gradient 0). The boundaries follow
eq. (13.19) of the lesson: A > 0 is clipped for rho > 1 + eps, A < 0 for rho < 1 - eps. Tokens A and B
are the two tokens of the worked example in section 3 (rho = 1.3 and 0.7).
Writes deepseek/figures/l13-clip-zones.svg
"""

from l13_svg import SVG, Axes

LO, HI = 0.05, 20.0          # plotted range of rho (log axis)


def zones(sign, eps):
    """Return [(rho_from, rho_to, live)] covering [LO, HI] for one advantage sign and clip ratio."""
    if sign > 0:
        b = 1 + eps           # clipped above 1 + eps
        return [(LO, min(b, HI), True)] + ([(b, HI, False)] if b < HI else [])
    b = 1 - eps               # clipped below 1 - eps; for eps >= 1 the bound is <= 0 and never reached
    if b <= LO:
        return [(LO, HI, True)]
    return [(LO, b, False), (b, HI, True)]


# sanity checks against the worked example: token A (A>0, rho=1.3), token B (A<0, rho=0.7)
def live(sign, eps, rho):
    return any(a <= rho <= b and ok for a, b, ok in zones(sign, eps))


assert not live(+1, 0.2, 1.3) and live(+1, 10, 1.3)
assert not live(-1, 0.2, 0.7) and live(-1, 10, 0.7)
assert not live(+1, 10, 11.5) and live(-1, 10, 0.05)      # exercise 7

W, H = 760, 306
svg = SVG(W, H, "裁剪让哪些 token 停止学习：按优势的符号与裁剪系数，把重要性比率分成照常学习与梯度为零两段")
ax = Axes(270, 740, 0, 1, (LO, HI), (0, 1), xlog=True)

ROWS = [  # header, then (sign, eps, left label, y of bar top, token)
    (22, "Â > 0（答对回答里的 token）：更新想把 ρ 往上推 →"),
    (+1, 0.2, "ε = 0.2（PPO 常用）", 38, ("A", 1.3)),
    (+1, 10, "ε = 10（R1 第一阶段）", 80, ("A", 1.3)),
    (134, "Â < 0（答错回答里的 token）：更新想把 ρ 往下推 ←"),
    (-1, 0.2, "ε = 0.2（PPO 常用）", 150, ("B", 0.7)),
    (-1, 10, "ε = 10（R1 第一阶段）", 192, ("B", 0.7)),
]
BAR = 22
TOP, BOTTOM = 28, 222

# rho = 1 reference line (parameters not yet updated)
svg.line(ax.px(1), TOP, ax.px(1), BOTTOM, color="ink", width=1.1, dash="4 3")
svg.text(ax.px(1) + 7, 122, "ρ = 1：参数还没更新", size=12, color="ink", weight="600")

for row in ROWS:
    if len(row) == 2:
        svg.text(16, row[0], row[1], size=13, weight="600")
        continue
    sign, eps, label, y, (tok, rho) = row
    svg.text(256, y + 16, label, size=12.5, anchor="end")
    segs = zones(sign, eps)
    for a, b, ok in segs:
        col = "green" if ok else "red"
        x0, x1 = ax.px(a), ax.px(b)
        svg.rect(x0, y, x1 - x0, BAR, fill=col, fill_opacity=0.16, stroke=col, width=1.1)
    # clip boundary (1 + eps or 1 - eps) labelled above the bar, if it falls inside the plotted range
    for edge in sorted({e for a, b, _ in segs for e in (a, b) if LO < e < HI}):
        svg.text(ax.px(edge), y - 4, "%g" % edge, size=11.5, anchor="middle", color="red", weight="600")
    # text inside the segments; positions chosen to avoid the rho = 1 line and the token dots
    for a, b, ok in segs:
        if ok:
            if len(segs) == 1:
                txt, x = "梯度照常：下界 1 − ε = −9，碰不到", ax.px(4.0)
            elif a == LO:
                txt, x = "梯度照常", ax.px(0.2)
            else:
                txt, x = "梯度照常", ax.px(4.0)
            svg.text(x, y + 16, txt, size=12, anchor="middle", color="green", weight="600")
        else:
            w = ax.px(b) - ax.px(a)
            if w > 120:
                x = ax.px(0.2) if a == LO else (ax.px(a) + ax.px(b)) / 2 + 8
                svg.text(x, y + 16, "被裁剪：梯度为 0", size=12, anchor="middle", color="red", weight="600")
            else:
                svg.text((ax.px(a) + ax.px(b)) / 2, y + 16, "梯度 0", size=11, anchor="middle", color="red",
                         weight="600")
    # the worked-example token
    x = ax.px(rho)
    svg.circle(x, y + BAR / 2, 8, fill="ink")
    svg.text(x, y + BAR / 2 + 4, tok, size=11, anchor="middle", color="surface", weight="700")

# axis
svg.line(ax.px(LO), BOTTOM + 4, ax.px(HI), BOTTOM + 4, color="muted", width=1)
for v in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20):
    x = ax.px(v)
    svg.line(x, BOTTOM + 4, x, BOTTOM + 9, color="muted", width=1)
    svg.text(x, BOTTOM + 22, "%g" % v, size=11, anchor="middle", color="muted")
svg.text((ax.px(LO) + ax.px(HI)) / 2, BOTTOM + 42, "重要性比率 ρ = 新策略概率 / 旧策略概率（对数刻度）", size=12,
         anchor="middle", color="muted")

# legend
ly = 294
svg.rect(16, ly - 11, 18, 12, fill="green", fill_opacity=0.16, stroke="green", width=1.1)
svg.text(40, ly, "梯度照常：这个 token 继续学习", size=12)
svg.rect(250, ly - 11, 18, 12, fill="red", fill_opacity=0.16, stroke="red", width=1.1)
svg.text(274, ly, "被裁剪：裁剪项是常数，梯度为 0", size=12)
svg.circle(512, ly - 5, 7, fill="ink")
svg.text(526, ly, "A、B：上文算例里的两个 token", size=12)
svg.save("l13-clip-zones.svg")
