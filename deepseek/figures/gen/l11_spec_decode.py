"""Figure 11.13: MTP as a speculative-decoding draft (timeline schematic + expected tokens per step).

Top: each decoding step, the main model verifies the previous draft and produces one token of its own;
the MTP module drafts the token after that. Accepted draft -> 2 tokens that step; rejected -> 1.
Bottom: expected tokens per main-model step with k draft tokens accepted independently with probability p,
E = (1 - p^{k+1}) / (1 - p)  (k = 1 gives 1 + p).  V3 sec. 5.4.3 reports second-token acceptance 85-90 %
and 1.8x TPS; Insights sec. 2.3.3 reports 80-90 % and 1.8x. Gloeckle et al. table S2 (code, 7B):
2 / 3 / 4 heads -> 1.94 / 2.78 / 3.50 tokens per forward pass.
Writes deepseek/figures/l11-spec-decode.svg
"""

import numpy as np

from l11_svg import SVG, box, polyline

W, H = 720, 560
svg = SVG(W, H, "用 MTP 做投机解码：时间线与每步期望产出")

# ---------------------------------------------------------------- timeline
svg.text(16, 22, "Ⓐ 每一步：主模型验证上一步的草稿，并自己再出 1 个 token；MTP 模块起草下一个", size=13, weight="600")
steps = [("第 1 步", True), ("第 2 步", False), ("第 3 步", True)]
x0 = 20
for k, (name, ok) in enumerate(steps):
    x = x0 + k * 232
    box(svg, x, 40, 216, 150, "", color="ink", fill_opacity=0.03)
    svg.text(x + 10, 60, name, size=12, weight="600")
    box(svg, x + 10, 72, 96, 32, "主模型前向", color="blue", fill_opacity=0.10, size=12)
    box(svg, x + 114, 72, 92, 32, "MTP 起草", color="green", fill_opacity=0.10, size=12)
    if ok:
        svg.text(x + 10, 126, "上一步的草稿 ✓ 被接受", size=12, color="green")
        svg.text(x + 10, 146, "本步产出：草稿 + 自己的 1 个", size=12)
        svg.text(x + 10, 170, "= 2 个 token", size=13, color="green", weight="700")
    else:
        svg.text(x + 10, 126, "上一步的草稿 ✗ 被拒绝", size=12, color="red")
        svg.text(x + 10, 146, "本步只产出主模型的 1 个", size=12)
        svg.text(x + 10, 170, "= 1 个 token", size=13, color="red", weight="700")
    if k < 2:
        svg.arrow(x + 216, 115, x + 231, 115, color="muted", width=1.4)
svg.text(20, 212, "接受概率为 p 时，每步期望产出 1 + p 个 token；草稿与验证本身也要算力，所以实际加速低于 1 + p。", size=12,
         color="muted")

# ---------------------------------------------------------------- expected-tokens chart
PX0, PX1, PY0, PY1 = 70, 520, 260, 510
P0, P1, E0, E1 = 0.5, 1.0, 1.0, 4.0


def px(p):
    return PX0 + (p - P0) / (P1 - P0) * (PX1 - PX0)


def py(e):
    return PY1 - (e - E0) / (E1 - E0) * (PY1 - PY0)


svg.text(16, 244, "Ⓑ 每步期望产出 E = (1 − p^{k+1}) / (1 − p)，k = 草稿数（各草稿独立、接受率同为 p）", size=13, weight="600")
for e in (1, 1.5, 2, 2.5, 3, 3.5, 4):
    svg.line(PX0, py(e), PX1, py(e), color="grid", width=1)
    svg.text(PX0 - 6, py(e) + 4, "%g" % e, size=11, anchor="end", color="muted")
for p in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    svg.line(px(p), PY1, px(p), PY1 + 4, color="ink", width=1)
    svg.text(px(p), PY1 + 17, "%.1f" % p, size=11, anchor="middle", color="muted")
svg.line(PX0, PY1, PX1, PY1, color="ink", width=1.2)
svg.line(PX0, PY0, PX0, PY1, color="ink", width=1.2)
svg.text((PX0 + PX1) / 2, PY1 + 36, "单个草稿 token 的接受率 p", size=12, anchor="middle")
svg.text(24, (PY0 + PY1) / 2, "每步期望 token 数", size=12, anchor="middle", rotate=-90)
# V3 / Insights bands
svg.rect(px(0.80), PY0, px(0.90) - px(0.80), PY1 - PY0, fill="amber", fill_opacity=0.08)
svg.rect(px(0.85), PY0, px(0.90) - px(0.85), PY1 - PY0, fill="green", fill_opacity=0.14)
ps = np.linspace(P0, 0.999, 200)
cols = {1: "green", 2: "blue", 3: "violet"}
for k in (1, 2, 3):
    E = (1 - ps ** (k + 1)) / (1 - ps)
    polyline(svg, [(px(p), py(e)) for p, e in zip(ps, E)], color=cols[k], width=2)
    pe = 0.7
    ev = (1 - pe ** (k + 1)) / (1 - pe)
    svg.text(px(pe), py(ev) + (18 if k == 1 else -8), "k = %d" % k, size=12, color=cols[k], weight="600",
             anchor="middle")
# 1.8x measured
svg.line(PX0, py(1.8), PX1, py(1.8), color="ink", width=1.2, dash="6 4")
svg.text(px(0.625), py(1.8) - 6, "实测 1.8 倍 TPS（V3 与 Insights）", size=11, color="ink")
for p in (0.85, 0.90):
    svg.circle(px(p), py(1 + p), 4, fill="green")
svg.text(px(0.85) - 4, py(1.95) - 4, "1.85", size=11, anchor="end", color="green")
svg.text(px(0.90) + 6, py(1.95) - 18, "1.90", size=11, color="green")

# notes
nx = 540
svg.text(nx, 280, "绿带：V3 报告的", size=12, color="green")
svg.text(nx, 297, "接受率 85%–90%", size=12, color="green")
svg.text(nx, 322, "黄带：Insights 论文", size=12, color="amber")
svg.text(nx, 339, "的 80%–90%", size=12, color="amber")
svg.text(nx, 372, "对照：Gloeckle 7B 代码", size=12)
svg.text(nx, 389, "模型的自投机解码", size=12)
svg.text(nx, 406, "2 头：1.94 个/次前向", size=12, color="muted")
svg.text(nx, 423, "3 头：2.78 个/次前向", size=12, color="muted")
svg.text(nx, 440, "4 头：3.50 个/次前向", size=12, color="muted")
svg.text(nx, 466, "（Gloeckle 等，表 S2）", size=11, color="muted")

svg.save("l11-spec-decode.svg")
