"""Figure 20.3 (summary of thread B): total vs activated parameters, and the activated share, from
DeepSeek LLM 67B (dense) to DeepSeek-V4.1-Flash.

Bars are on a log axis computed from l20_num.MOE (lesson 7 p8, lesson 10 p1/p8, lesson 11 p8,
lesson 18 p7, lesson 19 p1). V4.1-Flash also shows its 8B prefill activation (tick) and the 196B
Engram table that sits outside the 552B backbone (dashed extension).
Writes deepseek/figures/l20-moe-params.svg
"""

import math

from l20_svg import SVG
from l20_num import MOE, ROUTED, ACT, V41_PREFILL, V41_ENGRAM

W, H = 760, 480
svg = SVG(W, H, "主线 B 的总账：总参数与激活参数，激活占比从 100% 降到约 3%")
X0, X1 = 170, 500
LMIN, LMAX = 0, 3.5          # 1B .. ~3,162B


def xs(v):
    return X0 + (math.log10(v) - LMIN) / (LMAX - LMIN) * (X1 - X0)


TOP, RH = 74, 42
for e, labt in [(0, "1B"), (1, "10B"), (2, "100B"), (3, "1,000B")]:
    x = xs(10 ** e)
    svg.line(x, TOP - 8, x, TOP + len(MOE) * RH - 6, color="grid", width=1)
    svg.text(x, TOP - 14, labt, size=11, anchor="middle", color="muted")
svg.text(X0, 20, "参数量（对数刻度）", size=12, color="muted")
svg.rect(X0, 30, 12, 12, fill="violet", fill_opacity=0.18, stroke="violet", width=0.8)
svg.text(X0 + 17, 41, "总参数", size=12)
svg.rect(X0 + 76, 30, 12, 12, fill="blue", fill_opacity=0.7, stroke="blue", width=0.8)
svg.text(X0 + 93, 41, "每 token 激活", size=12)
svg.text(560, 41, "激活占比", size=12, anchor="middle", color="blue", weight="700")
svg.text(680, 41, "路由专家（选中 / 总数）", size=12, anchor="middle", color="violet", weight="700")

for i, (name, tot, act, note) in enumerate(MOE):
    y = TOP + i * RH
    last = name.endswith("V4.1-Flash")
    svg.text(X0 - 10, y + 17, name, size=12, anchor="end", weight="700" if last else None)
    svg.rect(X0, y + 4, xs(tot) - X0, 16, fill="violet", fill_opacity=0.18, stroke="violet", width=0.8)
    svg.rect(X0, y + 8, xs(act) - X0, 8, fill="blue", fill_opacity=0.7, stroke="blue", width=0.6)
    tl = ("%.0fB" % tot) if tot >= 100 else ("%.1fB" % tot)
    if tot == 1600:
        tl = "1.6T"
    if last:
        x8 = xs(V41_PREFILL)
        svg.line(x8, y + 2, x8, y + 22, color="blue", width=1.6)
        svg.text(x8, y + 34, "prefill 8B", size=11, anchor="middle", color="blue")
        xe = xs(tot + V41_ENGRAM)
        svg.rect(xs(tot), y + 4, xe - xs(tot), 16, fill="amber", fill_opacity=0.10, stroke="amber", width=1,
                 dash="3 2")
        svg.text(xs(tot) - 60, y + 34, "虚线：另有 196B Engram 表", size=11, color="amber")
        svg.text(xe + 5, y + 16, "%s / %sB" % (tl, ("%g" % act)), size=11, color="muted")
    else:
        svg.text(xs(tot) + 5, y + 16, "%s / %sB" % (tl, ("%g" % act)), size=11, color="muted")
    svg.text(560, y + 16, "%.1f%%" % (100 * ACT[name]) if ACT[name] < 1 else "100%", size=12, anchor="middle",
             color="blue")
    if name in ROUTED:
        k, n = ROUTED[name]
        svg.text(680, y + 16, "%s（%.1f%%）" % (note, 100 * k / n), size=12, anchor="middle", color="violet")
    else:
        svg.text(680, y + 16, "稠密", size=12, anchor="middle", color="muted")

FY = TOP + len(MOE) * RH + 30
svg.text(10, FY, "激活占比 = 激活参数 / 总参数（V4.1 用 decode 的 16B，分母是 552B 主干，不含 Engram）。", size=12, color="muted")
svg.text(10, FY + 18, "括号里是路由专家里被选中的比例 k / N_{r}；共享专家每个 token 都算，不在其中。", size=12, color="muted")
svg.text(10, FY + 36, "数据：第 7、10、11、18、19 课的规格表；由 l20_num.py 计算。", size=12, color="muted")

svg.save("l20-moe-params.svg")
