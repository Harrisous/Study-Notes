"""Figure 19.13: the V4.1-Flash pre-training schedule on a token axis (paper §4.2.2, §2.5, §4.1).

Learning rate: linear warm-up over 2,000 steps of 100.6M tokens (0.2012T), 2.6e-4 until 28T, cosine decay
to 2.6e-5 between 28T and 40T, then constant to 45T. Sequence length 64K with sparse attention from the
start, extended to 1M at 34T. Vision encoder frozen until the decay stage (28T). Text-only : multimodal
tokens = 7 : 1. The curve is computed point by point. Writes deepseek/figures/l19-pretrain.svg
"""

import math

from l19_svg import SVG

W, H = 740, 350
svg = SVG(W, H, "V4.1-Flash 的预训练日程：45T token 上的学习率、上下文长度与视觉编码器")

LR_MAX, LR_MIN = 2.6, 0.26           # units of 1e-4
WARM = 2000 * 100.6e6 / 1e12         # tokens (T) of warm-up
T_DECAY0, T_DECAY1, T_END, T_EXT = 28.0, 40.0, 45.0, 34.0
assert abs(WARM - 0.2012) < 1e-9


def lr(t):
    if t < WARM:
        return LR_MAX * t / WARM
    if t < T_DECAY0:
        return LR_MAX
    if t < T_DECAY1:
        tau = (t - T_DECAY0) / (T_DECAY1 - T_DECAY0)
        return LR_MIN + (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * tau)) / 2
    return LR_MIN


PX0, PX1, PY0, PY1 = 80, 700, 40, 220


def xs(t):
    return PX0 + t / T_END * (PX1 - PX0)


def ys(v):
    return PY1 - v / 2.8 * (PY1 - PY0)


# vision-encoder trainable region
svg.rect(xs(T_DECAY0), PY0, xs(T_END) - xs(T_DECAY0), PY1 - PY0, fill="green", fill_opacity=0.06)
svg.text(xs(36.3), ys(2.45), "视觉编码器解冻", size=12, color="green")
svg.text(xs(36.3), ys(2.45) + 16, "（衰减开始时）", size=11, color="green")

for v in (0, 0.5, 1.0, 1.5, 2.0, 2.5):
    svg.line(PX0, ys(v), PX1, ys(v), color="grid", width=1)
    svg.text(PX0 - 8, ys(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
for t in range(0, 46, 5):
    svg.line(xs(t), PY1, xs(t), PY1 + 4, color="muted", width=1)
    svg.text(xs(t), PY1 + 17, "%dT" % t, size=11, anchor="middle", color="muted")
svg.line(PX0, PY1, PX1, PY1, color="muted", width=1)
svg.line(PX0, PY0, PX0, PY1, color="muted", width=1)
svg.text(26, (PY0 + PY1) / 2, "学习率（× 10⁻⁴）", size=12, anchor="middle", color="muted", rotate=-90)

pts = []
N = 900
for i in range(N + 1):
    t = T_END * (i / N) ** 1.0
    pts.append((xs(t), ys(lr(t))))
pts.insert(1, (xs(WARM), ys(LR_MAX)))
pts.sort()
svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke="blue", width=2.4)
svg.text(xs(12), ys(LR_MAX) - 8, "2.6 × 10⁻⁴（2,000 步预热后保持）", size=12, anchor="middle", color="blue")
# the curve passes lr = 1.75 at about 32.9T; start the label to its right so the stroke does not cross it
svg.text(xs(34.4), ys(1.75), "余弦衰减", size=12, color="blue")
svg.text(xs(40.3), ys(LR_MIN) - 8, "2.6 × 10⁻⁵", size=12, color="blue")

# context extension
svg.line(xs(T_EXT), ys(1.2), xs(T_EXT), PY1, color="orange", width=1.4, dash="5 3")
svg.text(xs(T_EXT) - 6, ys(0.9), "34T：序列长度 64K → 1M", size=12, anchor="end", color="orange")

# bottom bars: sequence length and data mix
BY = PY1 + 40
svg.rect(xs(0), BY, xs(T_EXT) - xs(0), 16, fill="orange", fill_opacity=0.18, stroke="orange", width=0.8)
svg.text(xs(T_EXT / 2), BY + 12, "64K，稀疏注意力从头训练（无稠密预热）", size=11, anchor="middle")
svg.rect(xs(T_EXT), BY, xs(T_END) - xs(T_EXT), 16, fill="orange", fill_opacity=0.40, stroke="orange", width=0.8)
svg.text(xs((T_EXT + T_END) / 2), BY + 12, "1M", size=11, anchor="middle")
MY = BY + 26
tm = T_END * 7 / 8
svg.rect(xs(0), MY, xs(tm) - xs(0), 16, fill="muted", fill_opacity=0.18, stroke="muted", width=0.8)
svg.text(xs(tm / 2), MY + 12, "纯文本 %.3fT（7 份）" % tm, size=11, anchor="middle")
svg.rect(xs(tm), MY, xs(T_END) - xs(tm), 16, fill="green", fill_opacity=0.25, stroke="green", width=0.8)
svg.text(xs((tm + T_END) / 2), MY + 12, "多模态 %.3fT" % (T_END - tm), size=11, anchor="middle")
svg.text(PX0 - 8, BY + 12, "序列", size=11, anchor="end", color="muted")
svg.text(PX0 - 8, MY + 12, "数据", size=11, anchor="end", color="muted")
svg.text(PX0, MY + 42, "batch 固定为 1.006 亿 token，共约 %s 步；数据条只表示 7 : 1 的比例，两类数据在训练中是混合的。"
         % "{:,}".format(round(T_END * 1e12 / 100.6e6)), size=12, color="muted")

svg.save("l19-pretrain.svg")
