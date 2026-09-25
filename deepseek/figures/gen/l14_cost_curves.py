"""Figure 14.15: estimated cost per million tokens vs token position, V3.1-Terminus vs V3.2(-Exp).

Redrawn from V3.2-Exp Fig. 3 / V3.2 Fig. 3. The paper's curves are piecewise-linear; the
vertices below were parsed from the PDF vector data (lesson-14 notes, section 4.5). Costs are
estimated from benchmarking the deployed service on H800 at 2 USD per GPU-hour; they are not
API prices. Crossovers and ratios are computed here.
Writes deepseek/figures/l14-cost-curves.svg
"""

from l14_svg import SVG

PREFILL = {
    "dense": [(0, 0.0529), (128, 0.6658)],
    "dsa": [(0, 0.0611), (6.589, 0.0999), (128, 0.1871)],
}
DECODE = {
    "dense": [(0, 0.0705), (128, 2.1543)],
    "dsa": [(0, 0.0952), (2.056, 0.1457), (101.607, 0.2274), (128, 0.2491)],
}


def interp(pts, x):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise ValueError(x)


def crossover(a, b):
    lo, hi = 0.0, 128.0
    f = lambda x: interp(a, x) - interp(b, x)
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


XP = crossover(PREFILL["dense"], PREFILL["dsa"])
XD = crossover(DECODE["dense"], DECODE["dsa"])
RP = interp(PREFILL["dense"], 128) / interp(PREFILL["dsa"], 128)
RD = interp(DECODE["dense"], 128) / interp(DECODE["dsa"], 128)
assert abs(XP - 10.4) < 0.05 and abs(XD - 4.76) < 0.02
assert round(RP, 2) == 3.56 and round(RD, 2) == 8.65

W, H = 760, 420
svg = SVG(W, H, "每百万 token 的估计成本随 token 位置的变化（据 V3.2 论文图 3 重绘）")


def panel(x0, x1, title, data, ymax, ystep, cross, ratio, kink, kink_lab, kink_dy=18, cross_lab_at=None):
    y0, y1 = 56, 330

    def px(x):
        return x0 + x / 128 * (x1 - x0)

    def py(v):
        return y1 - v / ymax * (y1 - y0)

    svg.text((x0 + x1) / 2, 24, title, size=13, anchor="middle", weight="700")
    v = 0.0
    while v <= ymax + 1e-9:
        y = py(v)
        svg.line(x0, y, x1, y, color="grid", width=1)
        svg.text(x0 - 8, y + 4, "$%.1f" % v if ystep >= 0.1 else "%.2f" % v, size=11, anchor="end", color="muted")
        v += ystep
    for xk in (0, 32, 64, 96, 128):
        svg.line(px(xk), y1, px(xk), y1 + 4, color="muted", width=1)
        svg.text(px(xk), y1 + 18, "%dK" % xk, size=11, anchor="middle", color="muted")
    svg.line(x0, y1, x1, y1, color="muted", width=1.2)
    svg.line(x0, y0, x0, y1, color="muted", width=1.2)
    for key, col, wdt in (("dense", "blue", 2.4), ("dsa", "green", 2.8)):
        pts = data[key]
        svg.path("M" + " L".join("%.1f %.1f" % (px(a), py(b)) for a, b in pts), stroke=col, width=wdt,
                 join="round")
        for a, b in pts[1:-1]:
            svg.circle(px(a), py(b), 3, fill=col)
    # crossover
    yc = interp(data["dense"], cross)
    svg.circle(px(cross), py(yc), 4.5, fill="ink")
    # the label sits in the wedge between the two lines, right of the crossing, with a leader line,
    # so neither line runs through the text (cross_lab_at is in data coordinates: K, dollars)
    lx, ly = px(cross_lab_at[0]), py(cross_lab_at[1])
    lab_w = 72 * 128 / (x1 - x0)                       # label width (~72 units) in K
    for xk in (cross_lab_at[0], cross_lab_at[0] + lab_w):
        assert py(interp(data["dense"], xk)) < ly - 13, "label touches the dense line"
        assert py(interp(data["dsa"], xk)) > ly + 3, "label touches the DSA line"
    svg.line(px(cross) + 4, py(yc) - 2, lx - 3, ly - 4, color="ink", width=0.8)
    svg.text(lx, ly, "交叉 %.1fK" % cross if cross >= 10 else "交叉 %.2fK" % cross, size=12, weight="700")
    # kink
    kx, ky = kink
    if kink_dy < 0:
        svg.line(px(kx), py(ky) - 4, px(kx), py(ky) + kink_dy + 4, color="green", width=0.8)
        svg.text(px(kx) - 4, py(ky) + kink_dy, kink_lab, size=11, color="green")
    else:
        svg.text(px(kx) + 8, py(ky) + kink_dy, kink_lab, size=11, color="green")
    # ratio at 128K
    yd, ys = interp(data["dense"], 128), interp(data["dsa"], 128)
    svg.line(px(128) - 6, py(yd), px(128) - 6, py(ys), color="ink", width=1.2, dash="3 2")
    svg.text(px(128) - 12, (py(yd) + py(ys)) / 2, "%.2f×" % ratio, size=13, anchor="end", weight="700")
    svg.text(px(128) - 4, py(yd) - 8, "$%.4f" % yd, size=11, anchor="end", color="blue")
    svg.text(px(128) - 12, py(ys) - 8, "$%.4f" % ys, size=11, anchor="end", color="green")   # left of the dashed line
    svg.text((x0 + x1) / 2, y1 + 38, "token 在序列中的位置", size=12, anchor="middle", color="muted")


panel(70, 360, "(a) 预填充（prefill）", PREFILL, 0.7, 0.1, XP, RP, (6.589, 0.0999), "拐点 6.589K",
      cross_lab_at=(36, 0.162))
panel(440, 730, "(b) 解码（decode）", DECODE, 2.4, 0.4, XD, RD, (2.056, 0.1457), "拐点 2.056K", kink_dy=-44,
      cross_lab_at=(27, 0.30))

ly = 386
svg.rect(70, ly - 10, 12, 12, fill="blue")
svg.text(88, ly, "V3.1-Terminus（稠密 MLA）", size=12)
svg.rect(300, ly - 10, 12, 12, fill="green")
svg.text(318, ly, "V3.2-Exp / V3.2（DSA）", size=12)
svg.text(70, ly + 22, "纵轴：每百万 token 的估计成本（美元），按 H800 租价 2 美元 / GPU·时 折算实际服务的基准测试；不是 API 价格。",
         size=11, color="muted")

svg.save("l14-cost-curves.svg")
