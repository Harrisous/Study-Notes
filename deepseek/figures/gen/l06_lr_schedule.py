"""Figure 6.7: multi-step versus cosine learning-rate schedule for DeepSeek LLM 7B (computed).

Multi-step (dsllm §2.3, README): peak 4.2e-4 after 2000 warm-up steps; x0.316 (= 10^-1/2) after 80 % of
the tokens (1.6T) and x0.1 after 90 % (1.8T). Batch 2304 x 4096 tokens, so warm-up = 18.87B tokens.
The warm-up shape (linear) and the cosine floor (10 % of peak) are NOT given in the paper; they are
assumptions for illustration and are labelled as such in the caption.
Extension: the same recipe for a 3T budget shares the first 1.6T with the 2T run.
Writes deepseek/figures/l06-lr-schedule.svg
"""

import math

import numpy as np

from l06_svg import SVG

PEAK = 4.2e-4
WARM = 2000 * 2304 * 4096 / 1e12   # in T tokens (0.0189)


def multistep(t, D):
    if t < WARM:
        return PEAK * t / WARM
    if t < 0.8 * D:
        return PEAK
    if t < 0.9 * D:
        return PEAK * 10 ** -0.5
    return PEAK * 0.1


def cosine(t, D, floor=0.1):
    if t < WARM:
        return PEAK * t / WARM
    p = (t - WARM) / (D - WARM)
    return PEAK * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


W, H = 760, 400
X0, X1, Y0, Y1 = 78, 600, 30, 330
TMAX, YMAX = 3.0, 4.6e-4


def px(t):
    return X0 + t / TMAX * (X1 - X0)


def py(v):
    return Y1 - v / YMAX * (Y1 - Y0)


svg = SVG(W, H, "DeepSeek LLM 7B 的多步学习率调度与 cosine 调度（示意）")
for k in range(0, 5):
    y = py(k * 1e-4)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "%d×10^{-4}" % k if k else "0", size=11, anchor="end", color="muted")
for t in (0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    x = px(t)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, ("%.1fT" % t) if t else "0", size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "已处理的训练 token（T = 10^{12}）", size=12, anchor="middle", color="muted")
svg.text(20, (Y0 + Y1) / 2, "学习率 η", size=12, anchor="middle", color="muted", rotate=-90)


def curve(f, D, tmax):
    ts = np.concatenate([np.linspace(0, WARM, 20), np.linspace(WARM, tmax, 900)])
    pts = []
    for t in ts:
        pts.append((px(t), py(f(t, D))))
    return "M" + " L".join("%.1f %.1f" % p for p in pts)


def step_path(D, t_start=0.0):
    """Exact vertical steps for the multi-step curve."""
    pts = []
    if t_start == 0.0:
        pts += [(px(0), py(0)), (px(WARM), py(PEAK))]
    else:
        pts += [(px(t_start), py(PEAK))]
    pts += [(px(0.8 * D), py(PEAK)), (px(0.8 * D), py(PEAK * 10 ** -0.5)),
            (px(0.9 * D), py(PEAK * 10 ** -0.5)), (px(0.9 * D), py(PEAK * 0.1)), (px(D), py(PEAK * 0.1))]
    return "M" + " L".join("%.1f %.1f" % p for p in pts)


# cosine (illustrative floor)
svg.path(curve(cosine, 2.0, 2.0), stroke="muted", width=2, dash="6 4")
# extension to 3T: shares 0..1.6T, continues at peak until 2.4T
svg.path(step_path(3.0, t_start=1.6), stroke="violet", width=2, dash="3 3")
# the real 2T schedule
svg.path(step_path(2.0), stroke="blue", width=2.6, join="round")

# level labels
svg.text(px(1.6) - 6, py(PEAK) - 8, "4.2×10^{-4}（峰值）", size=12, anchor="end", color="blue")
svg.text(px(1.83), py(PEAK * 10 ** -0.5) + 4, "← 1.33×10^{-4}", size=12, color="blue")
svg.text(px(1.83) + 14, py(PEAK * 10 ** -0.5) + 21, "（31.6%）", size=12, color="blue")
svg.text(px(2.03), py(PEAK * 0.1) + 4, "← 4.2×10^{-5}", size=12, color="blue")
svg.text(px(2.03) + 14, py(PEAK * 0.1) + 21, "（10%）", size=12, color="blue")
# step positions
for t, lab in ((1.6, "1.6T（80%）"), (1.8, "1.8T（90%）")):
    svg.line(px(t), py(0) - 2, px(t), py(0) + 6, color="blue", width=1.4)
svg.text(px(1.6), Y1 - 8, "1.6T", size=11, anchor="end", color="blue")
svg.text(px(1.8) + 3, Y1 - 8, "1.8T", size=11, color="blue")

# warm-up callout (placed in the empty lower-left region)
svg.line(px(WARM) + 2, py(PEAK * 0.45), px(0.11), py(PEAK * 0.3) - 4, color="ink", width=0.8)
svg.text(px(0.12), py(PEAK * 0.3), "warmup 2000 步 ≈ 189 亿 token", size=12, color="ink")
svg.text(px(0.12), py(PEAK * 0.3) + 17, "（占 2T 的 0.94%；", size=12, color="muted")
svg.text(px(0.12), py(PEAK * 0.3) + 34, "线性形状为假设）", size=12, color="muted")

# legend
lx, ly = X1 + 16, 60
svg.line(lx, ly, lx + 30, ly, color="blue", width=2.6)
svg.text(lx + 38, ly + 4, "多步（7B 实际）", size=12, color="blue")
svg.line(lx, ly + 26, lx + 30, ly + 26, color="muted", width=2, dash="6 4")
svg.text(lx + 38, ly + 30, "cosine（示意，", size=12, color="muted")
svg.text(lx + 38, ly + 46, "终值设为 10%）", size=12, color="muted")
svg.line(lx, ly + 72, lx + 30, ly + 72, color="violet", width=2, dash="3 3")
svg.text(lx + 38, ly + 76, "假想：续训到 3T", size=12, color="violet")
svg.text(lx + 38, ly + 92, "0–1.6T 与 2T 的", size=12, color="violet")
svg.text(lx + 38, ly + 108, "那次完全共用", size=12, color="violet")

svg.save("l06-lr-schedule.svg")
