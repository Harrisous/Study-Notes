"""Figure 6.2: where the experiments were run versus where the real runs landed (log axis of C).

All positions are computed here from the research notes:
  hyper-parameter grid/fit: 1e17 .. 2e19 (dsllm §3.1 text), validation at 1e20 (Fig. 2b)
  IsoFLOP budgets: 1e17, 3e17, ..., 3e20 (dsllm §3.2)
  7B:  C = M D = 4.2467e10 * 2e12 = 8.49e22      67B: 4.3273e11 * 2e12 = 8.65e23  (notes R5)
Writes deepseek/figures/l06-extrapolation.svg
"""

import math

from l06_svg import SVG, sci

C7 = 4.2467e10 * 2e12
C67 = 4.3273e11 * 2e12
W, H = 760, 330
X0, X1 = 44, 724
L0, L1 = 16, 25
AXY = 286


def px(c):
    return X0 + (math.log10(c) - L0) / (L1 - L0) * (X1 - X0)


svg = SVG(W, H, "DeepSeek LLM 的实验预算与真实训练算力，对数坐标")

# decade grid + axis
for e in range(L0, L1 + 1):
    x = px(10 ** e)
    svg.line(x, 26, x, AXY, color="grid", width=1)
    svg.text(x, AXY + 18, "10^{%d}" % e, size=11, anchor="middle", color="muted")
svg.line(X0, AXY, X1, AXY, color="muted", width=1.2)
svg.text((X0 + X1) / 2, AXY + 38, "训练算力 C（FLOPs，对数刻度；C = M·D，见 §5）", size=12, anchor="middle",
         color="muted")

# row 1: hyper-parameter sweep
y1 = 52
svg.rect(px(1e17), y1 - 9, px(2e19) - px(1e17), 18, fill="blue", fill_opacity=0.18, stroke="blue", width=1.2, rx=4)
svg.text(px(1e17), y1 - 16, "超参数网格与拟合（正文：10^{17}–2×10^{19}）", size=12, color="blue")
# row 2: validation at 1e20
y2 = 100
svg.circle(px(1e20), y2, 6, fill="blue", stroke="blue")
svg.text(px(1e20) + 12, y2 + 4, "10^{20} 上验证公式（论文图 2b）", size=12, color="blue")
# row 3: IsoFLOP budgets
y3 = 148
budgets = [1e17, 3e17, 1e18, 3e18, 1e19, 3e19, 1e20, 3e20]
for b in budgets:
    svg.circle(px(b), y3, 5, fill="violet", stroke="violet")
svg.line(px(1e17), y3, px(3e20), y3, color="violet", width=1.2, opacity=0.6)
svg.text(px(1e17), y3 - 14, "IsoFLOP：8 个预算，10^{17}–3×10^{20}", size=12, color="violet")

# row 4: the real runs
y4 = 222
for c, lab in ((C7, "7B"), (C67, "67B")):
    x = px(c)
    pts = []
    for k in range(10):
        r = 9 if k % 2 == 0 else 4
        a = math.pi / 2 + k * math.pi / 5
        pts.append((x + r * math.cos(a), y4 - r * math.sin(a)))
    svg.polygon(pts, fill="green", stroke="green", width=1)
    if lab == "7B":
        svg.text(x + 6, y4 + 28, "%s：%s" % (lab, sci(c)), size=12, anchor="end", color="green", weight="600")
    else:
        svg.text(x - 6, y4 + 28, "%s：%s" % (lab, sci(c)), size=12, anchor="start", color="green", weight="600")

# ratio arcs from the largest experiment
xa = px(3e20)
for c, lab, lift in ((C7, "×%d" % round(C7 / 3e20), 34), (C67, "×%d" % round(C67 / 3e20), 62)):
    xb = px(c)
    ym = y4 - 12 - lift
    svg.path("M%.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" % (xa, y3 + 8, xa, ym, xb, ym, xb, y4 - 12),
             stroke="ink", width=1.2, dash="4 3", arrow=True)
    dy = -8 if lift > 40 else 26      # 67B label above its arc, 7B label below its arc
    svg.text((xa + xb) / 2 + 20, ym + dy, lab, size=12, anchor="middle", color="ink", weight="600")
svg.text(X1, 70, "虚线：从最大一组 IsoFLOP 实验", size=12, anchor="end", color="muted")
svg.text(X1, 86, "（3×10^{20}）外推到真实训练", size=12, anchor="end", color="muted")

# a note on the hyper-parameter extrapolation
svg.text(px(2e19) + 6, y1 + 5, "→ 到 67B 约 ×%s" % format(round(C67 / 2e19, -3), ",.0f"), size=12, color="blue")

svg.save("l06-extrapolation.svg")
