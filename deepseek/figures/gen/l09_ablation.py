"""Figures 9.2 and 9.15: the two attention ablations of DeepSeek-V2 (arXiv:2405.04434v5, appendix D).

Table 8 (D.1): 7B dense models, 1.33T tokens, MQA / GQA (8 groups) / MHA.
Table 9 (D.2): small MoE (~16B total, 1.33T tokens) and large MoE (~250B total, 420B tokens),
               MHA vs MLA, with KV cache per token in elements.
Numbers copied from lesson-09-doc sec. 3.4.
Writes deepseek/figures/l09-ablation-gqa.svg and deepseek/figures/l09-ablation-mla.svg
"""

from l09_svg import SVG, LinAxis, fmt

BENCH = ["BBH", "MMLU", "C-Eval", "CMMLU"]

# ---------------------------------------------------------------- Table 8
T8 = {
    "MQA": [33.2, 37.9, 30.0, 34.6],
    "GQA": [35.6, 41.2, 37.7, 38.4],
    "MHA": [37.0, 45.2, 42.9, 43.5],
}
W, H = 720, 300
svg = SVG(W, H, "DeepSeek-V2 附录 D.1：7B 稠密模型上 MQA、GQA、MHA 的得分")
X0, X1 = 120, 690
xs = LinAxis(28, 48, X0, X1)
Y0 = 58
ROW = 48
style = {"MQA": ("violet", 0.35, "MQA（1 组 K/V，7.1B 参数）"),
         "GQA": ("violet", 0.65, "GQA（8 组 K/V，6.9B）"),
         "MHA": ("violet", 1.0, "MHA（每头独立 K/V，6.9B）")}
# legend
lx = X0
for k in ("MQA", "GQA", "MHA"):
    col, op, lab = style[k]
    svg.circle(lx + 6, 24, 6, fill=col, fill_opacity=op, stroke=col, width=1.2)
    svg.text(lx + 17, 28, lab, size=12)
    lx += 196
for v in range(28, 49, 4):
    x = xs(v)
    svg.line(x, Y0 - 16, x, Y0 + ROW * 4 - 16, color="grid", width=1)
    svg.text(x, Y0 + ROW * 4 + 2, str(v), size=11, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, Y0 + ROW * 4 + 22, "得分（EM / 准确率，%）", size=12, anchor="middle", color="muted")
for r, b in enumerate(BENCH):
    y = Y0 + r * ROW + 8
    svg.text(X0 - 14, y + 4, b, size=13, anchor="end", weight="600")
    lo, hi = T8["MQA"][r], T8["MHA"][r]
    svg.line(xs(lo), y, xs(hi), y, color="violet", width=2, opacity=0.35)
    for k in ("MQA", "GQA", "MHA"):
        col, op, _ = style[k]
        v = T8[k][r]
        svg.circle(xs(v), y, 7, fill=col, fill_opacity=op, stroke=col, width=1.2)
        svg.text(xs(v), y - 12, "%.1f" % v, size=11, anchor="middle", color="ink")
    svg.text(xs(hi) + 14, y + 4, "差 %.1f" % (hi - lo), size=11, color="muted")
svg.save("l09-ablation-gqa.svg")

# ---------------------------------------------------------------- Table 9
T9 = {
    ("小", "MHA"): [37.9, 48.7, 51.6, 52.3],
    ("小", "MLA"): [39.0, 50.0, 50.9, 53.4],
    ("大", "MHA"): [46.6, 57.5, 57.9, 60.7],
    ("大", "MLA"): [50.7, 59.0, 59.2, 62.5],
}
KV = {("小", "MHA"): "110.6K", ("小", "MLA"): "15.6K", ("大", "MHA"): "860.2K", ("大", "MLA"): "34.6K"}
W, H = 720, 440
svg = SVG(W, H, "DeepSeek-V2 附录 D.2：MoE 模型上 MHA 与 MLA 的得分")
X0, X1 = 150, 690
xs = LinAxis(36, 64, X0, X1)
Y0 = 92
ROW = 38
svg.circle(X0 + 6, 22, 6, fill="muted", stroke="muted")
svg.text(X0 + 17, 26, "MHA", size=12, weight="600")
svg.circle(X0 + 76, 22, 6, fill="orange", stroke="orange")
svg.text(X0 + 87, 26, "MLA", size=12, weight="600", color="orange")
svg.text(X0 + 140, 26, "箭头：同规模下从 MHA 换成 MLA", size=12, color="muted")
svg.text(X0, 50, "小 MoE（约 16B，1.33T token）：每 token KV %s → %s 个元素" % (KV[("小", "MHA")], KV[("小", "MLA")]),
         size=12, color="muted")
svg.text(X0, 68, "大 MoE（约 250B，420B token）：每 token KV %s → %s 个元素" % (KV[("大", "MHA")], KV[("大", "MLA")]),
         size=12, color="muted")
nrows = 8
for v in range(36, 65, 4):
    x = xs(v)
    svg.line(x, Y0 - 14, x, Y0 + ROW * nrows - 14, color="grid", width=1)
    svg.text(x, Y0 + ROW * nrows + 2, str(v), size=11, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, Y0 + ROW * nrows + 22, "得分（EM / 准确率，%）", size=12, anchor="middle", color="muted")
r = 0
for b_i, b in enumerate(BENCH):
    for scale in ("小", "大"):
        y = Y0 + r * ROW + 4
        if scale == "小":
            svg.text(X0 - 50, y + 4 + ROW / 2, b, size=13, anchor="end", weight="600")
        svg.text(X0 - 14, y + 4, scale, size=12, anchor="end", color="muted")
        a, m = T9[(scale, "MHA")][b_i], T9[(scale, "MLA")][b_i]
        worse = m < a
        col = "red" if worse else "orange"
        svg.arrow(xs(a), y, xs(m), y, color=col, width=1.8, shorten=7)
        svg.circle(xs(a), y, 6, fill="muted", stroke="muted")
        svg.circle(xs(m), y, 6, fill="orange", stroke="orange")
        left, right = (m, a) if worse else (a, m)
        svg.text(xs(left) - 12, y + 4, "%.1f" % left, size=11, anchor="end",
                 color="orange" if left == m else "ink")
        svg.text(xs(right) + 12, y + 4, "%.1f" % right, size=11,
                 color="orange" if right == m else "ink")
        if worse:
            svg.text(xs(right) + 46, y + 4, "8 项里仅此项 MLA 低于 MHA", size=11, color="red")
        r += 1
    if b_i < 3:
        yy = Y0 + r * ROW - 14
        svg.line(X0 - 110, yy, X1, yy, color="rule", width=0.8)
svg.save("l09-ablation-mla.svg")
