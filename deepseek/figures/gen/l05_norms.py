"""Figure 5.7: LayerNorm vs RMSNorm on x = (1, 2, 3, 4), x + 10 and 3x (research notes R16).

gain g = 1, bias 0, epsilon = 0. Every bar height and label is computed here.
Writes deepseek/figures/l05-norms.svg
"""

import math

from l05_svg import SVG


def layernorm(x):
    m = sum(x) / len(x)
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / len(x))
    return [(v - m) / sd for v in x], m, sd


def rmsnorm(x):
    r = math.sqrt(sum(v * v for v in x) / len(x))
    return [v / r for v in x], r


rows = [("x = (1, 2, 3, 4)", [1, 2, 3, 4]), ("x + 10 = (11, 12, 13, 14)", [11, 12, 13, 14]),
        ("3x = (3, 6, 9, 12)", [3, 6, 9, 12])]
ln0, _, _ = layernorm(rows[0][1])
rn0, _ = rmsnorm(rows[0][1])

W, H = 760, 560
svg = SVG(W, H, "LayerNorm 与 RMSNorm：平移与缩放下谁变谁不变")
COLX = [150, 380, 590]
CW = 150
heads = [("输入", "muted"), ("LayerNorm：减均值，再除以标准差", "violet"), ("RMSNorm：只除以均方根", "blue")]
for (h, col), x0 in zip(heads, COLX):
    svg.text(x0 + CW / 2, 26, h, size=13, anchor="middle", weight="700", color=col)


def bars(x0, ybase, vals, vmin, vmax, col, h=92, fmt="%.4f", highlight=None):
    """bars centred in a CW-wide cell; zero line at value 0."""
    def Y(v):
        return ybase + h - (v - vmin) / (vmax - vmin) * h
    y0 = Y(0)
    svg.line(x0, y0, x0 + CW, y0, color="muted", width=1)
    bw = 22
    gap = (CW - 4 * bw) / 5
    prev_ty, prev_lifted = None, False
    for n, v in enumerate(vals):
        bx = x0 + gap + n * (bw + gap)
        top, bot = (Y(v), y0) if v >= 0 else (y0, Y(v))
        svg.rect(bx, top, bw, max(bot - top, 0.8), fill=col, fill_opacity=0.75)
        ty = top - 5 if v >= 0 else bot + 13
        # labels (about 38 units wide) are wider than the bar pitch: when two neighbours sit at
        # nearly the same height, lift every other one by a line so the numbers do not run together
        lifted = False
        if (v >= 0 and len(fmt % v) > 3 and prev_ty is not None and not prev_lifted
                and abs(ty - prev_ty) < 9):
            ty, lifted = ty - 12, True
        prev_ty, prev_lifted = (top - 5 if v >= 0 else bot + 13), lifted
        svg.text(bx + bw / 2, ty, (fmt % v).replace("-", "−"), size=11, anchor="middle",
                 color=col if col != "muted" else "ink")
    if highlight:
        svg.rect(x0 - 6, ybase - 18, CW + 12, h + 24, stroke=highlight, width=1.6, rx=6, dash="5 3")


for r, (lab, x) in enumerate(rows):
    yb = 58 + r * 150
    svg.text(16, yb + 50, lab.split(" = ")[0], size=13, weight="700")
    svg.text(16, yb + 68, "= " + lab.split(" = ")[1], size=12, color="muted")
    bars(COLX[0], yb, x, 0, 15, "muted", fmt="%g")
    ln, m, sd = layernorm(x)
    rn, rms = rmsnorm(x)
    same_ln = max(abs(a - b) for a, b in zip(ln, ln0)) < 1e-12
    same_rn = max(abs(a - b) for a, b in zip(rn, rn0)) < 1e-12
    bars(COLX[1], yb, ln, -1.6, 1.6, "violet", highlight=None)
    bars(COLX[2], yb, rn, -1.6, 1.6, "blue", highlight=None if same_rn else "red")
    svg.text(COLX[1] + CW / 2, yb + 118, "均值 %g，标准差 %.4f" % (m, sd), size=11.5, anchor="middle", color="muted")
    svg.text(COLX[2] + CW / 2, yb + 118, "均方根 RMS = %.4f" % rms, size=11.5, anchor="middle", color="muted")
    if r > 0:
        svg.text(COLX[1] + CW / 2, yb + 134, "与第一行相同" if same_ln else "变了", size=12, anchor="middle",
                 color="green" if same_ln else "red", weight="600")
        svg.text(COLX[2] + CW / 2, yb + 134, "与第一行相同" if same_rn else "变了：平移没有被消掉", size=12,
                 anchor="middle", color="green" if same_rn else "red", weight="600")

svg.text(16, H - 14, "三列的纵轴刻度不同：输入列 0–15，两列输出 −1.6–1.6。增益 g = 1、偏置 0、ε = 0。", size=12, color="muted")
svg.save("l05-norms.svg")
