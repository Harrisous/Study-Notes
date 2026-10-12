"""Figure 7.14: DeepSeekMoE 16B parameter budget, total vs activated, recomputed from config.json.

config: 28 layers, d = 2048, vocab 102400, MHA, first layer dense FFN (intermediate 10944),
27 MoE layers with 2 shared + 64 routed experts (intermediate 1408), top-6, untied embeddings.
Total 16,375,728,128; activated 2,828,650,496 (incl. input embedding and output head).
Writes deepseek/figures/l07-16b-budget.svg
"""

from l07_svg import SVG

d, V, L = 2048, 102400, 28
e = 3 * d * 1408
comp_total = [
    ("路由专家", 27 * 64 * e, "blue", 0.8),
    ("共享专家", 27 * 2 * e, "violet", 0.75),
    ("注意力", L * 4 * d * d, "ink", 0.3),
    ("输入嵌入", V * d, "muted", 0.3),
    ("输出头", V * d, "muted", 0.18),
    ("首层稠密 FFN", 3 * d * 10944, "blue", 0.3),
    ("路由器", 27 * 64 * d, "violet", 0.3),
]
norms = d + L * 2 * d
total = sum(v for _, v, _, _ in comp_total) + norms
act_parts = [(n, (27 * 6 * e if n == "路由专家" else v), c, o) for n, v, c, o in comp_total]
act = sum(v for _, v, _, _ in act_parts) + norms
assert total == 16_375_728_128 and act == 2_828_650_496

W, H = 740, 400
svg = SVG(W, H, "DeepSeekMoE 16B 的参数预算：总参数与每 token 激活参数")
X0, X1 = 110, 710
BW = X1 - X0


def bar(y, parts, scale_total, label, sub, labels_below=False, min_label=46):
    x = X0
    svg.text(X0 - 10, y + 15, label, size=13, anchor="end", weight="600")
    svg.text(X0 - 10, y + 31, sub, size=11, anchor="end", color="muted")
    for name, v, col, op in parts:
        w = BW * v / scale_total
        svg.rect(x, y, w, 36, fill=col, fill_opacity=op, stroke="surface", width=1)
        if w >= min_label:
            share = 100 * v / sum(p[1] for p in parts)
            tc = "surface" if (col in ("blue", "violet") and op > 0.6) else "ink"
            svg.text(x + w / 2, y + 15, name, size=12, anchor="middle", color=tc)
            svg.text(x + w / 2, y + 30, "%.1f%%" % share, size=11, anchor="middle", color=tc)
        x += w
    return x


svg.text(16, 22, "(a) 按同一比例尺：激活参数只有总参数的 %.1f%%" % (100 * act / total), size=13, weight="600")
end_t = bar(40, comp_total, total, "总参数", "16.376B", min_label=60)
end_a = bar(92, act_parts, total, "激活参数", "2.829B", min_label=400)
svg.text(end_a + 8, 115, "← 每个 token 实际用到的部分", size=12, color="blue")

svg.text(16, 170, "(b) 把激活参数放大到整条：路由专家只占一半", size=13, weight="600")
bar(188, act_parts, act, "激活参数", "放大 %.1f 倍" % (total / act), min_label=56)

# legend with absolute values
ly = 262
svg.text(16, ly, "构成（B = 10^{9}）", size=12, weight="600")
for k, (name, v, col, op) in enumerate(comp_total):
    r, c = divmod(k, 4)
    x = 16 + c * 180
    y = ly + 22 + r * 42
    svg.rect(x, y - 11, 14, 14, fill=col, fill_opacity=op, stroke="rule", width=0.8)
    av = dict((n, vv) for n, vv, _, _ in act_parts)[name]
    svg.text(x + 20, y, name, size=12)
    if av == v:
        svg.text(x + 20, y + 17, "%.3fB（全部激活）" % (v / 1e9), size=11, color="muted")
    else:
        svg.text(x + 20, y + 17, "总 %.3fB / 激活 %.3fB" % (v / 1e9, av / 1e9), size=11, color="muted")
svg.text(16, H - 12, "由 config.json 复算：总 16,375,728,128，激活 2,828,650,496（含输入嵌入与输出头；两者都去掉是 2.409B）。",
         size=12, color="muted")

svg.save("l07-16b-budget.svg")
