"""Figure 15.9: why Keep Routing is needed (constructed toy, not model output).

16 routed experts, top-4 (DeepSeek-V3 / V3.2 actually use 256 routed experts, top-8, per MoE layer).
Row 1: affinities as computed by the inference engine during sampling. Row 2: the same token re-computed
by the training engine; the values differ by at most 0.0015 (a made-up perturbation of the size of BF16
rounding: the BF16 spacing in [0.5, 1) is 2^-8 = 0.0039), which is enough to swap experts ranked 4th and
5th. Row 3: Keep Routing reuses the set chosen at sampling time (V3.2 paper section 3.1).
Writes deepseek/figures/l15-keep-routing.svg
"""

import numpy as np

from l15_svg import SVG, fmt

S_INFER = np.array([0.12, 0.35, 0.08, 0.61, 0.22, 0.5170, 0.05, 0.30,
                    0.55, 0.18, 0.10, 0.5185, 0.27, 0.09, 0.66, 0.14])
DELTA = np.array([0.0004, -0.0006, 0.0003, -0.0008, 0.0005, 0.0012, -0.0002, 0.0007,
                  -0.0009, 0.0001, -0.0004, -0.0015, 0.0006, 0.0002, -0.0005, 0.0003])
S_TRAIN = S_INFER + DELTA
K = 4
top_inf = set(np.argsort(-S_INFER)[:K].tolist())
top_tr = set(np.argsort(-S_TRAIN)[:K].tolist())
assert np.abs(DELTA).max() <= 0.0015 < 2 ** -8
assert top_inf ^ top_tr == {5, 11}, (top_inf, top_tr)          # exactly one swap: expert 11 out, 5 in
assert abs(2 ** -8 - 0.00390625) < 1e-12

W, H = 760, 350
svg = SVG(W, H, "Keep Routing：同一个 token，推理框架与训练框架可能选出不同的专家")

BX0, BX1 = 196, 744
n = len(S_INFER)
cw = (BX1 - BX0) / n
SCALE = 64 / 0.7


def row(y_base, vals, chosen, title, sub, changed=()):
    svg.text(16, y_base - 38, title, size=13, weight="700")
    svg.text(16, y_base - 20, sub, size=12, color="muted")
    svg.line(BX0, y_base, BX1, y_base, color="rule", width=1)
    for i, v in enumerate(vals):
        x = BX0 + i * cw + 4
        h = v * SCALE
        on = i in chosen
        col = "blue" if on else "muted"
        svg.rect(x, y_base - h, cw - 8, h, fill=col, fill_opacity=0.45 if on else 0.15,
                 stroke="red" if i in changed else col, width=2.2 if i in changed else 1)
        svg.text(x + (cw - 8) / 2, y_base + 14, str(i), size=11, anchor="middle", color="blue" if on else "muted")
        if i in (5, 11):
            svg.text(x + (cw - 8) / 2, y_base - h - 5, "%.4f" % v, size=11, anchor="middle",
                     color="red" if i in changed else "ink")


row(104, S_INFER, top_inf, "① 采样时（推理框架）", "选中 {3, 8, 11, 14}")
row(214, S_TRAIN, top_tr, "② 训练时重算（训练框架）", "选中 {3, 5, 8, 14}", changed=(5, 11))
svg.text(BX0 - 8, 104 + 14, "专家", size=11, anchor="end", color="muted")
svg.text(BX0 - 8, 214 + 14, "专家", size=11, anchor="end", color="muted")

# row 3: keep routing
Y3 = 270
svg.text(16, Y3 + 4, "③ Keep Routing", size=13, weight="700")
svg.text(16, Y3 + 22, "训练时强制用 ① 的集合", size=12, color="muted")
x = BX0
for e in sorted(top_inf):
    svg.rect(x, Y3 - 12, 64, 28, fill="blue", fill_opacity=0.45, stroke="blue", width=1.2, rx=5)
    svg.text(x + 32, Y3 + 7, "专家 %d" % e, size=12, anchor="middle")
    x += 74
svg.text(x + 6, Y3 + 7, "→ 采样与训练优化的是同一组专家参数", size=12, color="blue")

svg.text(16, 326, "专家 5 与 11 的亲和度只差 0.0015（第 4、5 名），两套框架的数值差异（每项 ≤ 0.0015）足以让它们换位。",
         size=12, color="muted")
svg.text(16, 344, "数值为构造的示意，不是模型输出。真实的 V3.2 每层是 256 个路由专家选 8 个，共 58 个 MoE 层。", size=12, color="muted")

svg.save("l15-keep-routing.svg")
