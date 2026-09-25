"""Figure 12.7: a toy model of limited-precision accumulation, with and without promotion.

THIS IS NOT A HARDWARE MEASUREMENT. It is the lesson's own simplified model (research notes
section 4.12), meant only to show the trend:
  * inputs X (32 x K) and W (K x 32) are N(0, 1) values rounded to E4M3, so every product is
    exact in float64;
  * the products are taken 32 at a time (one WGMMA's worth along K); inside a group they are
    aligned to the group's largest exponent and truncated toward zero to 14 significant bits,
    then added exactly (fixed point);
  * the running accumulator is truncated to 14 significant bits after every addition;
  * with promotion, every N_C = 128 elements the partial sum is added to an FP32 accumulator
    and the low-precision accumulator restarts from zero.
The error is |y_model - y_exact| / |y_exact| per output element (median over 32 x 32).
Writes deepseek/figures/l12-accum-error.svg
"""

import math

import numpy as np

from l12_num import round_to_grid
from l12_svg import SVG

BITS = 14


def exponent(v):
    a = np.abs(v)
    out = np.full(v.shape, -1000.0)
    nz = a > 0
    out[nz] = np.floor(np.log2(a[nz]))
    return out


def trunc_sig(v, bits=BITS):
    q = 2.0 ** (exponent(v) - bits + 1)
    return np.trunc(v / q) * q


def run(K, promote, seed=0, n=32):
    rng = np.random.default_rng(seed)
    X = round_to_grid(rng.standard_normal((n, K)))
    Wm = round_to_grid(rng.standard_normal((K, n)))
    P = X[:, :, None] * Wm[None, :, :]
    exact = P.sum(axis=1)
    acc = np.zeros((n, n))
    f32 = np.zeros((n, n), dtype=np.float32)
    for g0 in range(0, K, 32):
        grp = P[:, g0:g0 + 32, :]
        q = 2.0 ** (exponent(grp).max(axis=1) - BITS + 1)
        s = (np.trunc(grp / q[:, None, :]) * q[:, None, :]).sum(axis=1)
        acc = trunc_sig(acc + s)
        if promote and (g0 + 32) % 128 == 0:
            f32 = (f32 + acc.astype(np.float32)).astype(np.float32)
            acc = np.zeros((n, n))
    res = f32.astype(np.float64) + acc if promote else acc
    rel = np.abs(res - exact) / np.abs(exact)
    return float(np.median(rel))


Ks = [512, 1024, 2048, 4096, 8192]
plain = [run(k, False) for k in Ks]
promo = [run(k, True) for k in Ks]
for k, a, b in zip(Ks, plain, promo):
    print("K=%5d  no promotion %.3g   N_C=128 promotion %.3g" % (k, a, b))
# linear growth without promotion (doubling K roughly doubles the error), flat with promotion
assert 1.7 < plain[-1] / plain[-2] < 2.3 and max(promo) / min(promo) < 1.5
# numbers quoted in the lesson text: K = 4096 median 2.9e-3 without promotion (inside the
# 0.26 %-0.52 % estimate of the worked example), about 29x the promoted error; 3.4e-4 .. 5.9e-3
# over K = 512 .. 8192.
assert abs(plain[3] - 2.9e-3) < 1e-4 and 0.0026 < plain[3] < 0.0052
assert round(plain[3] / promo[3]) == 29
assert abs(plain[0] - 3.4e-4) < 1e-5 and abs(plain[-1] - 5.9e-3) < 1e-4

W, H = 720, 380
X0, X1, Y0, Y1 = 90, 560, 30, 300
LY0, LY1 = -5, -2


def px(k):
    return X0 + (math.log2(k) - 9) / (13 - 9) * (X1 - X0)


def py(v):
    return Y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "简化模型：不提升时累加误差随 K 线性增长，每 128 个元素提升到 FP32 后与 K 无关")
for e in range(LY0, LY1 + 1):
    y = py(10.0 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
for k in Ks:
    x = px(k)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 18, str(k), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "内维长度 K（对数刻度）", size=12, anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "中位相对误差（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

# reference slope 1/2 (random-walk growth) from the first no-promotion point
ref = [plain[0] * math.sqrt(k / Ks[0]) for k in Ks]
svg.path("M" + " L".join("%.1f %.1f" % (px(k), py(v)) for k, v in zip(Ks, ref)), stroke="muted", width=1.2,
         dash="5 4")
svg.text(px(Ks[-1]) + 8, py(ref[-1]) + 4, "若误差随机抵消：∝ √K", size=12, color="muted")

for data, col, lab in ((plain, "red", "不提升（14 位累加到底）"), (promo, "green", "每 N_{C} = 128 提升到 FP32")):
    svg.path("M" + " L".join("%.1f %.1f" % (px(k), py(v)) for k, v in zip(Ks, data)), stroke=col, width=2.4)
    for k, v in zip(Ks, data):
        svg.circle(px(k), py(v), 3.6, fill=col)
    svg.text(px(Ks[-1]) + 8, py(data[-1]) + 4, lab, size=12, color=col, weight="600")
for k, v in zip(Ks, plain):
    svg.text(px(k), py(v) - 9, "%.1e" % v, size=11, anchor="middle", color="red")
svg.text(px(Ks[-1]) + 8, py(plain[-1]) + 22, "∝ K：截断都朝 0 偏，", size=12, color="muted")
svg.text(px(Ks[-1]) + 8, py(plain[-1]) + 40, "误差同号，越积越多", size=12, color="muted")
svg.text(X0, H - 8, "本课自己的玩具模型，不是硬件测量：输入 N(0,1) 舍入到 E4M3，32×32 个输出；只用来看趋势。", size=11,
         color="muted")
svg.save("l12-accum-error.svg")
