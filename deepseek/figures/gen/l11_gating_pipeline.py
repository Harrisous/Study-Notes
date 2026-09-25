"""Figure 11.4: the gating pipeline of V2 vs V3 (schematic).

V2 (arXiv:2405.04434 eqs. 21-22; HF config): softmax over 160 logits, device-limited (M = 3, device
score = max affinity on the device), top-6, g = s (no normalisation), code multiplies by 16.0.
V3 (arXiv:2412.19437 eqs. 13-16; HF config / MoEGate): sigmoid over 256 logits; selection uses s + b,
node-limited (8 groups, keep 4, group score = sum of the 2 largest s + b), top-8; gate uses the raw s
normalised over the 8 selected, code multiplies by routed_scaling_factor 2.5.
Writes deepseek/figures/l11-gating-pipeline.svg
"""

from l11_svg import SVG, box

W, H = 720, 400
svg = SVG(W, H, "门控流水线：V2 与 V3 对照")

BW, BH, GAP = 92, 48, 14


def row(x0, y, items, color, fo=0.12):
    xs = []
    for k, (a, b) in enumerate(items):
        x = x0 + k * (BW + GAP)
        box(svg, x, y, BW, BH, a, color=color, fill_opacity=fo, size=12, sub=b, sub_size=11)
        if k:
            svg.arrow(x - GAP, y + BH / 2, x - 1, y + BH / 2, color=color, width=1.3)
        xs.append(x)
    return xs


# ---------------------------------------------------------------- V2
svg.text(16, 24, "V2（softmax，设备受限，不归一化）", size=13, weight="600", color="muted")
v2 = [("logit", "160 个 z_{i}"), ("softmax", "s 之和 = 1"), ("设备受限", "设备内最大 s"),
      ("选 3 台设备", "M = 3"), ("Top-6", "在 3 台内选"), ("g = s", "不归一化")]
xs = row(16, 40, v2, "muted", 0.10)
svg.arrow(xs[-1] + BW, 64, xs[-1] + BW + 18, 64, color="muted", width=1.3)
svg.text(xs[-1] + BW + 22, 60, "× 16", size=12, color="muted")
svg.text(xs[-1] + BW + 22, 76, "（代码）", size=11, color="muted")

svg.line(16, 116, W - 16, 116, color="rule", width=1)

# ---------------------------------------------------------------- V3 selection branch
svg.text(16, 142, "V3（sigmoid，偏置只管选，节点受限，选中后归一化）", size=13, weight="600", color="violet")
svg.text(16, 166, "选择支路：决定“去哪几个专家”", size=12, color="violet")
v3a = [("logit", "256 个 z_{i}"), ("sigmoid", "s_{i} ∈ (0, 1)"), ("+ 偏置 b_{i}", "s_{i} + b_{i}"),
       ("节点打分", "最大 2 个之和"), ("选 4 个节点", "M = 4"), ("Top-8", "在 4 个节点内")]
xa = row(16, 176, v3a, "violet", 0.12)

# weight branch
svg.text(16, 270, "权重支路：决定“各占多少”", size=12, color="green")
wy = 280
wx = [xa[1], xa[2] + 30, xa[3] + 60, xa[4] + 90]
labels = [("取 8 个的 s", "不含 b"), ("归一化", "g_{i} = s_{i} / Σ s_{j}"), ("× 2.5", "（代码）"),
          ("加权求和", "Σ g_{i}·FFN_{i}(*u*)")]
for k, ((a, b), x) in enumerate(zip(labels, wx)):
    col = "green"
    box(svg, x, wy, BW + (14 if k == 1 else 0), BH, a, color=col, fill_opacity=0.12, size=12, sub=b, sub_size=11)
for k in range(1, 4):
    x_prev = wx[k - 1] + BW + (14 if k - 1 == 1 else 0)
    svg.arrow(x_prev, wy + BH / 2, wx[k] - 1, wy + BH / 2, color="green", width=1.3)

# sigmoid output feeds the weight branch (raw s)
sx = xa[1] + BW / 2
svg.arrow(sx, 176 + BH, sx, wy - 1, color="green", width=1.4)
svg.text(sx - 6, 240, "原始 s", size=11, color="green", anchor="end")
# selected indices from Top-8 go down to "取 8 个的 s"
tx = xa[5] + BW / 2
svg.path("M%s %s V %s H %s V %s" % (tx, 176 + BH, 246, wx[0] + BW - 16, wy - 1), stroke="violet", width=1.3,
         dash="5 3", arrow=True)
svg.text(tx - 4, 240, "只传下标（哪 8 个）", size=11, color="violet", anchor="end")
# bias is barred from the weight branch
bx = xa[2] + BW / 2
svg.line(bx, 176 + BH, bx, 262, color="red", width=1.3, dash="3 3")
svg.line(bx - 7, 255, bx + 7, 269, color="red", width=2)
svg.line(bx - 7, 269, bx + 7, 255, color="red", width=2)
svg.text(bx + 10, 266, "b 不进入门控值", size=11, color="red")

svg.text(16, 366, "要点：V3 把“选谁”和“给多少权重”拆成两条支路。偏置 b 只改变选择（紫色），", size=12)
svg.text(16, 386, "门控值只由原始亲和度决定（绿色）；V2 的 softmax 让每个门控值都依赖全部 160 个 logit。", size=12)

svg.save("l11-gating-pipeline.svg")
