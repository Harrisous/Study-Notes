"""Figure 4.15: expected tokens produced per target-model call in speculative decoding.

E[# tokens] = (1 - alpha^(gamma+1)) / (1 - alpha)   (Leviathan et al. 2023, eq. 1; i.i.d. acceptance assumption)
Curves for gamma = 1, 2, 3, 5, 7, 10 and the gamma -> infinity limit 1 / (1 - alpha), clipped in the script.
Marks: DeepSeek-V3 MTP as a draft of length 1 with acceptance 0.85-0.90 (V3 report sec. 5.4.3),
and the worked example alpha = 0.8, gamma = 5 -> 3.69. Grid values match lesson-04-doc §4.8.
Writes deepseek/figures/l04-spec-expected.svg
"""

from l04_svg import SVG


def expected(a, g):
    return (1 - a ** (g + 1)) / (1 - a)


# a few entries of the notes' grid (§4.8)
assert round(expected(0.8, 5), 2) == 3.69 and round(expected(0.9, 10), 2) == 6.86
assert round(expected(0.95, 10), 2) == 8.62 and round(expected(0.7, 3), 2) == 2.53

W, H = 720, 470
X0, X1, Y0, Y1 = 70, 560, 30, 390
YMAX = 11.0


def px(a):
    return X0 + a / 1.0 * (X1 - X0)


def py(e):
    return Y1 - (e - 0) / YMAX * (Y1 - Y0)


svg = SVG(W, H, "投机解码：每次调用目标模型期望产出的 token 数")
for e in range(0, 12):
    y = py(e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    if e % 2 == 1 or e == 0:
        svg.text(X0 - 8, y + 4, str(e), size=11, anchor="end", color="muted")
for a in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
    x = px(a)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 17, "%.1f" % a, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "平均接受率 α", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "每轮期望产出的 token 数", size=12, anchor="middle", color="muted", rotate=-90)

alphas = [i / 400 for i in range(0, 390)]
# limit gamma -> infinity, clipped to the plot
lim = [(a, 1 / (1 - a)) for a in alphas if 1 / (1 - a) <= YMAX]
svg.path("M" + " L".join("%.1f %.1f" % (px(a), py(e)) for a, e in lim), stroke="muted", width=1.6, dash="6 4")
a_end, e_end = lim[-1]
svg.text(px(a_end) - 6, py(e_end) + 4, "γ → ∞：1/(1 − α)", size=12, anchor="end", color="muted")

shades = {1: 0.45, 2: 0.55, 3: 0.65, 5: 0.78, 7: 0.9, 10: 1.0}
for g, op in shades.items():
    pts = [(a, expected(a, g)) for a in alphas + [0.975]]
    svg.path("M" + " L".join("%.1f %.1f" % (px(a), py(e)) for a, e in pts), stroke="blue", width=2.2, opacity=op)
    a_lab = 0.975
    svg.text(px(a_lab) + 8, py(expected(a_lab, g)) + 4, "γ = %d" % g, size=12, color="blue", weight="600")

# DeepSeek-V3 MTP: gamma = 1, alpha 0.85..0.90
seg = [(a / 1000, expected(a / 1000, 1)) for a in range(850, 901)]
svg.path("M" + " L".join("%.1f %.1f" % (px(a), py(e)) for a, e in seg), stroke="green", width=7, opacity=0.8)
xm, ym = px(0.875), py(expected(0.875, 1))
# short leader that stops above the label (the label's text occupies roughly y = py(0.5)-11 .. py(0.5)+2)
svg.line(xm, ym + 6, xm - 6, py(0.5) - 15, color="green", width=1)
svg.text(px(0.02), py(0.5), "DeepSeek-V3 的 MTP 当草稿（γ = 1）：α = 0.85–0.90 → 每步 1.85–1.90 个 token", size=12,
         color="green", weight="600")

# worked example alpha = 0.8, gamma = 5
xe, ye = px(0.8), py(expected(0.8, 5))
svg.circle(xe, ye, 5, fill="ink")
svg.line(xe - 4, ye - 4, px(0.56) + 2, py(5.4), color="ink", width=0.9)
svg.text(px(0.56), py(5.6), "例：α = 0.8，γ = 5 → 3.69", size=12, anchor="end", weight="600")

svg.text(16, H - 14, "按式 (1)（Leviathan 等人）逐点计算；假设各位置的接受事件独立同分布。γ 是每轮草稿长度。", size=12, color="muted")
svg.save("l04-spec-expected.svg")
