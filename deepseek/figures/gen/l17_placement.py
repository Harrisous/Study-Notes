"""Figure 17.6: where Engram goes.

Top: Engram-27B (30 layers, Engram in layers 2 and 15, counted from 1; notes §3.1) and
V4.1-Flash (40 layers, config engram_layer_ids [1, 14] counted from 0 = layers 2 and 15; notes §3.4),
and the order inside a block: Engram -> Attention -> MoE, each a residual add (paper §2.3).
Bottom: the §6.2 ablation (12-layer 3B MoE, 0.56B active, 100B tokens, +1.6B Engram).
Only three numbers are printed in the paper text: baseline 1.808, best single layer (layer 2) 1.770,
reference (layers 2 + 6) 1.768. Everything else was read back from the vector figure
arch_ablation.svg of arXiv v2 (notes §4.9, about +-0.0002) and is marked as read from the figure.
Writes deepseek/figures/l17-placement.svg
"""

from l17_svg import SVG

SWEEP = [1.7730, 1.7705, 1.7748, 1.7762, 1.7786, 1.7774, 1.7793, 1.7802, 1.7808, 1.7810, 1.7824, 1.7822]
REF, BASE = 1.7678, 1.808
COMP = [("去掉多分支集成", 1.7743), ("去掉分词器压缩", 1.7735), ("去掉上下文门控", 1.7724),
        ("加 4-gram", 1.7695), ("去掉短卷积", 1.7683)]
assert min(range(12), key=lambda i: SWEEP[i]) == 1          # layer 2 is the best single layer
assert round(SWEEP[1], 3) == 1.770 or round(SWEEP[1], 4) == 1.7705

W, H = 760, 608
svg = SVG(W, H, "Engram 放在哪一层：两个真实配置与 12 层模型上的逐层扫描")


# ---------------------------------------------------------------- top: backbones
def strip(y, n, hits, label, sub):
    x0, x1 = 196, 744
    bw = (x1 - x0) / n
    svg.text(20, y + 13, label, size=13, weight="700")
    svg.text(20, y + 31, sub, size=11.5, color="muted")
    for i in range(n):
        hit = (i + 1) in hits
        svg.rect(x0 + i * bw + 1, y, bw - 2, 26, fill="amber" if hit else "blue",
                 fill_opacity=0.55 if hit else 0.10, stroke="amber" if hit else "blue",
                 width=1.2 if hit else 0.8, rx=2)
    for i in sorted(set([1, n] + list(hits))):
        svg.text(x0 + (i - 0.5) * bw, y + 42, str(i), size=11, anchor="middle",
                 color="amber" if i in hits else "muted", weight="700" if i in hits else None)


strip(20, 30, (2, 15), "Engram-27B", "30 层，第 2、15 层")
strip(84, 40, (2, 15), "V4.1-Flash", "40 层，配置 [1, 14]（从 0 计）")

# inside one block
BY = 158
svg.text(20, BY + 22, "一个带 Engram 的块：", size=12.5)
xs = [176, 330, 484]
names = [("Engram", "amber"), ("注意力", "violet"), ("MoE", "blue")]
svg.line(150, BY + 42, 690, BY + 42, color="ink", width=1.6)
svg.arrow(680, BY + 42, 700, BY + 42, color="ink", width=1.6)
svg.text(706, BY + 46, "*H*", size=13, weight="700")
for x, (nm, col) in zip(xs, names):
    svg.box(x, BY, 104, 26, [nm], color=col, fill_opacity=0.14, size=12.5)
    svg.path("M%s %s V%s" % (x + 20, BY + 42, BY + 28), stroke="ink", width=1.2, arrow=True)
    svg.path("M%s %s V%s" % (x + 84, BY + 28, BY + 36), stroke="ink", width=1.2, arrow=True)
    svg.circle(x + 84, BY + 42, 6, fill="surface", stroke="ink", width=1.2)
    svg.text(x + 84, BY + 46.5, "+", size=11.5, anchor="middle")
svg.text(598, BY + 18, "三次残差相加", size=12, color="muted")

# ---------------------------------------------------------------- bottom: the sweep
X0, X1, Y0, Y1 = 70, 468, 262, 540
L0, L1 = 1.7665, 1.7835


def x(layer):
    return X0 + (layer - 0.5) / 12 * (X1 - X0)


def y(v):
    return Y1 - (v - L0) / (L1 - L0) * (Y1 - Y0)


svg.text(20, 238, "12 层 3B MoE 上加 1.6B Engram：验证损失（越低越好）", size=13, weight="700")
for v in (1.767, 1.770, 1.773, 1.776, 1.779, 1.782):
    svg.line(X0, y(v), 744, y(v), color="grid", width=1)
    svg.text(X0 - 8, y(v) + 4, "%.3f" % v, size=11, anchor="end", color="muted")
for i in range(1, 13):
    svg.text(x(i), Y1 + 17, str(i), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "只插一个 Engram 模块时的层号", size=12, anchor="middle", color="muted")

# baseline is far above the axis
svg.text(X0 + 8, Y0 + 14, "纯 MoE 基线 1.808：在本图上沿之外，高出约 0.025 ↑", size=12, color="red")

svg.line(X0, y(REF), 744, y(REF), color="green", width=1.6, dash="6 4")
svg.text(X1 - 4, y(REF) - 7, "参考配置：第 2 + 6 层两块，1.768", size=12, anchor="end", color="green",
         weight="700")

svg.path("M" + " L".join("%.1f %.1f" % (x(i + 1), y(v)) for i, v in enumerate(SWEEP)), stroke="amber",
         width=2)
for i, v in enumerate(SWEEP):
    if i == 1:
        svg.circle(x(i + 1), y(v), 5.2, fill="amber", stroke="surface", width=1.2)
    else:
        svg.circle(x(i + 1), y(v), 3.8, fill="surface", stroke="amber", width=1.6)
svg.text(x(2) + 9, y(SWEEP[1]) + 4, "第 2 层最好：1.770", size=12, color="amber", weight="700")
svg.text(x(1) - 2, y(SWEEP[0]) - 10, "第 1 层", size=11.5, anchor="middle", color="amber")
svg.text(x(9), y(1.7745), "越深越差", size=12, anchor="middle", color="amber")

# component ablations (right column, same y scale)
CXd = 520
svg.line(CXd - 22, Y0, CXd - 22, Y1, color="rule", width=1)
svg.text(CXd - 12, Y0 - 6, "从参考配置去掉/改一项：", size=12, color="muted")
for nm, v in COMP:
    svg.circle(CXd, y(v), 4, fill="surface", stroke="violet", width=1.6)
    svg.text(CXd + 10, y(v) + 4, "%s %.4f" % (nm, v), size=11.5, color="violet")

svg.text(20, 600, "实心点与绿线、1.808 是论文正文数字；空心点据 arXiv v2 图 5 的矢量坐标读取（约 ±0.0002），不是正文数字。",
         size=11.5, color="muted")
svg.save("l17-placement.svg")
