"""Figure 10.8: DeepSeek-V2's learning-rate and batch-size schedules against training tokens.

Paper (V2 sec. 3.1.2): linear warmup over the first 2K steps to 2.4e-4; multiply by 0.316 after ~60 % of
tokens and again after ~90 %; batch grows from 2304 to 9216 sequences (4K tokens each) over the first
225B tokens, then stays at 9216; 8.1T tokens in total. The shape of the batch ramp is not given
("gradually"); here it is assumed linear in tokens, which puts the end of warmup at ~21B tokens.
Writes deepseek/figures/l10-lr-batch.svg
"""

import numpy as np

from l10_svg import SVG, fmt

TOT = 8.1e12
SEQ = 4096
B0, B1, RAMP = 2304, 9216, 225e9
LR0, DEC = 2.4e-4, 0.316
D1, D2 = 0.6 * TOT, 0.9 * TOT


def batch(x):
    return B0 + (B1 - B0) * min(x, RAMP) / RAMP


# walk the steps to find where warmup ends (in tokens); warmup is linear in STEPS (paper), so keep the
# token count reached after each step and interpolate the step index from it
x, step, warm_x = 0.0, 0, None
xs_steps = [0.0]
while step < 2000:
    x += batch(x) * SEQ
    step += 1
    xs_steps.append(x)
warm_x = x
assert 20e9 < warm_x < 23e9
assert 2950 < batch(warm_x) < 2970      # batch at the end of warmup, quoted in the lesson (~2960)


def lr(x):
    if x < warm_x:
        return LR0 * float(np.interp(x, xs_steps, np.arange(len(xs_steps)))) / 2000
    if x < D1:
        return LR0
    if x < D2:
        return LR0 * DEC
    return LR0 * DEC * DEC


W, H = 760, 384
svg = SVG(W, H, "DeepSeek-V2 的学习率与 batch 调度（横轴：已训练的 token 数）")
X0, X1, Y0, Y1 = 78, 690, 40, 330
LRMAX, BMAX = 2.8e-4, 10000


def px(t):
    return X0 + t / TOT * (X1 - X0)


def py_lr(v):
    return Y1 - v / LRMAX * (Y1 - Y0)


def py_b(b):
    return Y1 - b / BMAX * (Y1 - Y0)


for k in range(0, 8):
    v = k * 0.4e-4
    y = py_lr(v)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, fmt(v * 1e4), size=11, anchor="end", color="blue")
for b in range(0, 10001, 2000):
    svg.text(X1 + 8, py_b(b) + 4, "%d" % b, size=11, color="violet")
for t in range(0, 9):
    xx = px(t * 1e12)
    svg.line(xx, Y1, xx, Y1 + 5, color="muted", width=1)
    svg.text(xx, Y1 + 19, "%dT" % t, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="blue", width=1.2)
svg.line(X1, Y0, X1, Y1, color="violet", width=1.2)
svg.text(22, (Y0 + Y1) / 2, "学习率（×10⁻⁴）", size=12, anchor="middle", color="blue", rotate=-90)
svg.text(W - 12, (Y0 + Y1) / 2, "batch（条序列，每条 4K token）", size=12, anchor="middle", color="violet",
         rotate=90)
svg.text((X0 + X1) / 2, Y1 + 40, "已训练的 token 数（线性刻度，共 8.1T）", size=12, anchor="middle", color="muted")

# batch curve
bx = np.linspace(0, TOT, 800)
svg.path("M" + " L".join("%.1f %.1f" % (px(t), py_b(batch(t))) for t in bx), stroke="violet", width=2.2)
# lr curve (step function, draw exact corners)
pts = [(0, 0), (warm_x, LR0), (D1, LR0), (D1, LR0 * DEC), (D2, LR0 * DEC), (D2, LR0 * DEC ** 2),
       (TOT, LR0 * DEC ** 2)]
svg.path("M" + " L".join("%.1f %.1f" % (px(t), py_lr(v)) for t, v in pts), stroke="blue", width=2.6)

# annotations
svg.text(px(3.3e12), py_lr(LR0) + 20, "峰值 2.4×10⁻⁴", size=12, anchor="middle", color="blue", weight="600")
svg.text(px(D1) + 6, py_lr(LR0 * DEC) - 8, "×0.316 → 7.58×10⁻⁵", size=12, color="blue")
svg.text(px(D2) - 9, py_lr(LR0 * DEC ** 2) - 30, "×0.316 → 2.40×10⁻⁵", size=12, anchor="end",
         color="blue")
# the 90 % label goes left of its line, otherwise it runs into the right (batch) axis
for t, lab, side in ((D1, "60%：4.86T", 1), (D2, "90%：7.29T", -1)):
    svg.line(px(t), Y0, px(t), Y1, color="muted", width=1, dash="3 3")
    svg.text(px(t) + 5 * side, Y1 - 8, lab, size=11, color="muted", anchor="start" if side > 0 else "end")
svg.text(px(2.4e12), py_b(B1) - 8, "batch 9216 条 = 37.7M token/步", size=12, anchor="middle", color="violet")

# inset: first 300B tokens
IX0, IX1, IY0, IY1 = 150, 380, 196, 300
svg.rect(IX0 - 44, IY0 - 30, IX1 - IX0 + 86, IY1 - IY0 + 58, fill="surface", stroke="rule", width=1, rx=6)
svg.text(IX0 - 36, IY0 - 14, "放大：前 300B token", size=12, weight="600")
IT = 300e9


def ipx(t):
    return IX0 + t / IT * (IX1 - IX0)


def ipy_lr(v):
    return IY1 - v / LRMAX * (IY1 - IY0)


def ipy_b(b):
    return IY1 - b / BMAX * (IY1 - IY0)


svg.line(IX0, IY1, IX1, IY1, color="muted", width=1)
for t in (0, 100e9, 200e9, 300e9):
    svg.text(ipx(t), IY1 + 14, "%dB" % (t / 1e9), size=11, anchor="middle", color="muted")
ix = np.linspace(0, IT, 300)
svg.path("M" + " L".join("%.1f %.1f" % (ipx(t), ipy_b(batch(t))) for t in ix), stroke="violet", width=1.8)
svg.path("M" + " L".join("%.1f %.1f" % (ipx(t), ipy_lr(lr(t))) for t in ix), stroke="blue", width=2)
svg.text(ipx(warm_x) + 4, ipy_lr(LR0) - 6, "2K 步 warmup ≈ 21B", size=11, color="blue")
svg.text(ipx(RAMP), ipy_lr(LR0) + 18, "225B", size=11, anchor="middle", color="violet")
svg.text(ipx(40e9), ipy_b(B0) + 14, "2304", size=11, color="violet")
svg.save("l10-lr-batch.svg")
