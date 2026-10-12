"""Figure 15.4: UE8M0 scales and the E4M3 range they leave unused.

Top: the UE8M0 byte for the scale 2^-7 (exponent field 120, bias 127; CUDA Math API __nv_fp8_e8m0:
"Bias equals to 127, so numbers 0 through 254 represent 2^-127 through 2^127. Number 0xFF = 255 is
reserved for NaN.").
Bottom: with block amax a, DeepSeek's kernel (inference/kernel.py, fast_round_scale) uses
s = 2^ceil(log2(a/448)). The largest code in the block is a/s = 448*u with
u(a) = (a/448) / 2^ceil(log2(a/448)) in (1/2, 1]. Plotted against log2(a); FP32 scaling gives u = 1.
If log2 a is spread uniformly, the mean of u is 1/(2 ln 2) = 0.7213.
Writes deepseek/figures/l15-ue8m0-sawtooth.svg
"""

import math

from l15_svg import SVG, fmt

FP8_MAX = 448.0


def util(a):
    return (a / FP8_MAX) / 2.0 ** math.ceil(math.log2(a / FP8_MAX))


assert abs(util(3.0) - 0.857142857) < 1e-6 and util(224.0) == 1.0 and abs(util(224.1) - 0.50022) < 1e-4
MEAN = 1 / (2 * math.log(2))

W, H = 760, 470
svg = SVG(W, H, "UE8M0 缩放因子：只能取 2 的整数次幂，代价是 E4M3 的最上面一段有时用不上")

# ---------------------------------------------------------------- top: the byte
svg.text(16, 22, "UE8M0：8 位全部是指数（无符号位、无尾数），值 = 2^{E−127}", size=13, weight="700")
bits = format(120, "08b")
BX, BY, BW = 40, 36, 34
for i, b in enumerate(bits):
    svg.rect(BX + i * BW, BY, BW - 2, 32, fill="orange", fill_opacity=0.14 if b == "1" else 0.04,
             stroke="orange", width=1.2, rx=3)
    svg.text(BX + i * BW + (BW - 2) / 2, BY + 22, b, size=15, anchor="middle", weight="700")
svg.text(BX + 4 * BW - 1, BY + 50, "指数字段 E = 0111 1000₂ = 120", size=12, anchor="middle", color="muted")
svg.text(BX + 8 * BW + 20, BY + 14, "→  2^{120−127} = 2^{−7} = 0.0078125", size=13, color="ink")
svg.text(BX + 8 * BW + 20, BY + 34, "E = 0…254 表示 2^{−127}…2^{127}；E = 255 保留给 NaN。", size=12, color="muted")
svg.text(BX + 8 * BW + 20, BY + 52, "没有 0，也没有负数：缩放因子本来就是正数。", size=12, color="muted")

# ---------------------------------------------------------------- bottom: sawtooth
PX0, PX1, PY0, PY1 = 70, 600, 150, 400        # plot box (y grows downward)
LX0, LX1 = 0.0, 10.0
UY0, UY1 = 0.4, 1.24


def px(lx):
    return PX0 + (lx - LX0) / (LX1 - LX0) * (PX1 - PX0)


def py(u):
    return PY1 - (u - UY0) / (UY1 - UY0) * (PY1 - PY0)


svg.text(16, 130, "每个 1×128 分块的最大码值 / 448（E4M3 范围的利用率 u）", size=13, weight="700")
for u in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    svg.line(PX0, py(u), PX1, py(u), color="grid", width=1)
    svg.text(PX0 - 6, py(u) + 4, fmt(u), size=11, anchor="end", color="muted")
for k in range(0, 11, 2):
    svg.line(px(k), PY0, px(k), PY1, color="grid", width=1)
    svg.text(px(k), PY1 + 16, "2^{%d}" % k, size=11, anchor="middle", color="muted")
svg.text((PX0 + PX1) / 2, PY1 + 36, "分块最大绝对值 a（对数刻度）", size=12, anchor="middle", color="muted")
svg.line(PX0, PY1, PX1, PY1, color="ink", width=1)
svg.line(PX0, PY0, PX0, PY1, color="ink", width=1)

# FP32 scaling: always 1
svg.line(PX0, py(1.0), PX1, py(1.0), color="green", width=2.2)
svg.text(PX1 + 6, py(1.0) + 4, "FP32 缩放：u = 1", size=12, color="green")
# mean
svg.line(PX0, py(MEAN), PX1, py(MEAN), color="muted", width=1.2, dash="5 4")
svg.text(PX1 + 6, py(MEAN) + 4, "平均 1/(2 ln 2) ≈ 0.721", size=12, color="muted")
svg.text(PX1 + 6, py(0.5) + 4, "最坏 1/2", size=12, color="muted")

# sawtooth: jumps where a = 448 * 2^n, i.e. log2 a = log2(448) + n
J = math.log2(FP8_MAX)
n_lo = math.floor(LX0 - J) - 1
n_hi = math.ceil(LX1 - J) + 1
for n in range(n_lo, n_hi + 1):
    j = J + n
    a0, b0 = max(j - 1, LX0), min(j, LX1)
    if a0 >= b0:
        continue
    pts = []
    for s in range(41):
        lx = a0 + (b0 - a0) * s / 40
        u = 2.0 ** (lx - j)
        pts.append("%s %s" % (fmt(px(lx)), fmt(py(u))))
    svg.path("M" + " L".join(pts), stroke="orange", width=2.2)
    if LX0 < j < LX1:
        svg.line(px(j), py(1.0), px(j), py(0.5), color="orange", width=1, dash="2 3")

# worked points, labels on a light backing so the teeth do not cut through them


def label_w(t):
    return sum(12 if ord(ch) > 0x2E80 else 6.7 for ch in t.replace("^{", "").replace("}", ""))


for a, lab, tx, ty in [(3.0, "a = 3.0：s = 2^{−7}，u = 0.857", 178, py(1.115)),
                       (100.0, "a = 100：s = 2^{−2}，u = 0.893", 392, py(1.115)),
                       (224.0, "a = 224：s = 2^{−1}，u = 1", 520, py(1.195)),
                       (224.1, "a = 224.1：s = 2^{0}，u = 0.500", 330, py(0.445))]:
    lx, u = math.log2(a), util(a)
    w = label_w(lab)
    x0 = tx - w / 2
    svg.line(px(lx), py(u), tx, ty - 5 if ty > py(u) else ty + 5, color="ink", width=0.8)
    svg.rect(x0 - 4, ty - 14, w + 8, 19, fill="surface", fill_opacity=0.92, stroke="rule", width=0.8, rx=3)
    svg.text(tx, ty, lab, size=12, anchor="middle", color="ink")
    svg.circle(px(lx), py(u), 4.5, fill="orange", stroke="surface", width=1.2)

svg.text(16, 458, "缩放因子向上取到 2 的幂，保证 a/s ≤ 448 不溢出；代价是最大码值落在 224 到 448 之间。", size=12, color="muted")

svg.save("l15-ue8m0-sawtooth.svg")
