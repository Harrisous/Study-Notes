"""Figure 15.15: accuracy versus output tokens (V3.2 paper Table 3).

Each panel is one benchmark; each point is (average output tokens in thousands, score). Order of models:
GPT-5 High, Gemini-3.0 Pro, Kimi-K2 Thinking, DeepSeek-V3.2 Thinking, DeepSeek-V3.2-Speciale.
Codeforces is a rating, and Kimi-K2 has no Codeforces entry. Arrow: V3.2 -> Speciale.
Speciale / V3.2 token ratios average 1.67 (arithmetic mean of the 8 ratios).
Writes deepseek/figures/l15-speciale-tokens.svg
"""

from l15_svg import SVG, fmt

DATA = {  # benchmark: [(tokens_k, score) or None] in model order
    "AIME 2025": [(13, 94.6), (15, 95.0), (24, 94.5), (16, 93.1), (23, 96.0)],
    "HMMT Feb 2025": [(16, 88.3), (16, 97.5), (31, 89.4), (19, 92.5), (27, 99.2)],
    "HMMT Nov 2025": [(20, 89.2), (15, 93.3), (29, 89.2), (18, 90.2), (25, 94.4)],
    "IMOAnswerBench": [(31, 76.0), (18, 83.3), (37, 78.6), (27, 78.3), (45, 84.5)],
    "LiveCodeBench": [(13, 84.5), (13, 90.7), (29, 82.6), (16, 83.3), (27, 88.7)],
    "Codeforces（等级分）": [(29, 2537), (22, 2708), None, (42, 2386), (77, 2701)],
    "GPQA Diamond": [(8, 85.7), (8, 91.9), (12, 84.5), (7, 82.4), (16, 85.7)],
    "HLE（无工具）": [(15, 26.3), (15, 37.7), (24, 23.9), (21, 25.1), (35, 30.6)],
}
ratios = [v[4][0] / v[3][0] for v in DATA.values()]
MEAN_RATIO = sum(ratios) / len(ratios)
assert abs(MEAN_RATIO - 1.673) < 1e-3

MODELS = [("GPT-5 High", "muted", "G"), ("Gemini-3.0 Pro", "violet", "Gm"), ("Kimi-K2 Thinking", "muted", "K"),
          ("V3.2 Thinking", "blue", None), ("V3.2-Speciale", "blue", None)]

W, H = 760, 440
svg = SVG(W, H, "准确率与输出 token 数：Speciale 用更多 token 换分")

PW, PH = 150, 128
X0, Y0 = 46, 34
GX, GY = 182, 186
for k, (name, pts) in enumerate(DATA.items()):
    col, row = k % 4, k // 4
    ox, oy = X0 + col * GX, Y0 + row * GY
    xs = [p[0] for p in pts if p]
    ys = [p[1] for p in pts if p]
    xmax = max(xs) * 1.12
    span = max(ys) - min(ys)
    ylo, yhi = min(ys) - 0.18 * span - 0.5, max(ys) + 0.18 * span + 0.5

    def sx(v):
        return ox + v / xmax * PW

    def sy(v):
        return oy + PH - (v - ylo) / (yhi - ylo) * PH

    svg.rect(ox, oy, PW, PH, fill="surface", stroke="rule", width=1)
    svg.text(ox + PW / 2, oy - 8, name, size=12, anchor="middle", weight="700")
    for t in range(0, int(xmax) + 1, 10 if xmax < 30 else 20):
        if True:
            svg.line(sx(t), oy + PH, sx(t), oy + PH + 4, color="muted", width=1)
            svg.text(sx(t), oy + PH + 16, str(t), size=11, anchor="middle", color="muted")
    step = 100 if yhi > 1000 else 1
    for v in (ylo + 0.12 * (yhi - ylo), yhi - 0.12 * (yhi - ylo)):
        tv = round(v / step) * step
        svg.line(ox, sy(tv), ox + 4, sy(tv), color="muted", width=1)
        svg.text(ox - 4, sy(tv) + 4, "%d" % tv, size=11, anchor="end", color="muted")
    (x3, y3), (xs_, ys_) = pts[3], pts[4]
    svg.arrow(sx(x3), sy(y3), sx(xs_), sy(ys_), color="blue", width=1.4, shorten=6)
    for m, p in enumerate(pts):
        if not p:
            continue
        mname, mcol, tag = MODELS[m]
        if m == 3:
            svg.circle(sx(p[0]), sy(p[1]), 5, fill="surface", stroke="blue", width=2)
        elif m == 4:
            svg.circle(sx(p[0]), sy(p[1]), 5.5, fill="blue", stroke="surface", width=1)
        else:
            svg.rect(sx(p[0]) - 4, sy(p[1]) - 4, 8, 8, fill=mcol, fill_opacity=0.7, stroke=mcol, width=1)
            if name.startswith("GPQA") and tag == "K":
                # the V3.2 -> Speciale arrow passes right of this marker: put the tag underneath
                svg.text(sx(p[0]), sy(p[1]) + 17, tag, size=11, anchor="middle", color=mcol)
            else:
                svg.text(sx(p[0]) + 7, sy(p[1]) + 4, tag, size=11, color=mcol)
    svg.text(ox + PW - 4, oy + 14, "×%s" % fmt(pts[4][0] / pts[3][0]), size=11, anchor="end", color="blue")

svg.text(W / 2, 404, "横轴：平均输出 token 数（千）；纵轴：准确率（%%）或 Codeforces 等级分。右上角是 Speciale 与 V3.2 的 token 之比，8 项平均 ×%.2f。"
         % MEAN_RATIO, size=11, anchor="middle", color="muted")
LY = 428
x = 60
for i, (mname, mcol, tag) in enumerate(MODELS):
    if i == 3:
        svg.circle(x + 6, LY - 4, 5, fill="surface", stroke="blue", width=2)
    elif i == 4:
        svg.circle(x + 6, LY - 4, 5.5, fill="blue", stroke="surface", width=1)
    else:
        svg.rect(x + 2, LY - 8, 8, 8, fill=mcol, fill_opacity=0.7, stroke=mcol, width=1)
    lab = mname + ("（%s）" % tag if tag else "")
    svg.text(x + 16, LY, lab, size=12)
    x += 30 + 7.4 * len(lab)

svg.save("l15-speciale-tokens.svg")
