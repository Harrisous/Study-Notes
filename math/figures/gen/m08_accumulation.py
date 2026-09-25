"""Figure M8.6: running sums in low precision.

Left: add 1.0 repeatedly (3000 times), rounding the running sum after every addition to
BF16 (stalls at 256 = 2^8) or FP16 (stalls at 2048 = 2^11).
Right: add 4096 numbers drawn from U(0, 0.01) (numpy default_rng(0), inputs rounded to BF16):
running sum in BF16 (stalls at 4.0), in BF16 inside chunks of 128 with the chunk sums
added in FP32 (the "promotion" idea of DeepSeek-V3), and the exact sum.
Also prints the table of the §5 example "用上面的实验数据看分段的效果": the same 4096 inputs
summed with a toy "p = 14" accumulator (the running sum is rounded to 14 significant bits
after every addition, to nearest-even or by truncation toward zero), sequentially and with
the partial sum promoted to FP32 every 128 terms.  This is the author's toy model of "about
14 bits", not a reproduction of the H800 hardware.
Writes math/figures/m08-accumulation.svg
"""

import math

import numpy as np

from m08_fp import bf16, fp16, fp32
from m08_svg import SVG


def run(xs, rnd):
    s, out = 0.0, [0.0]
    for x in xs:
        s = float(rnd(s + x))
        out.append(s)
    return np.array(out)


N1 = 3000
ones = np.ones(N1)
r_bf = run(ones, bf16)
r_fp16 = run(ones, fp16)
assert r_bf[-1] == 256 and r_fp16[-1] == 2048

xs = bf16(np.random.default_rng(0).uniform(0, 0.01, 4096))
exact = np.concatenate([[0.0], np.cumsum(xs)])
seq = run(xs, bf16)
chunked, tot = [0.0], 0.0
part = 0.0
for k, x in enumerate(xs):
    part = float(bf16(part + x))
    if (k + 1) % 128 == 0:
        tot = float(fp32(tot + part))
        part = 0.0
        chunked.append(tot)
    else:
        chunked.append(float(fp32(tot + part)))
chunked = np.array(chunked)
print("exact %.4f  bf16 %.4f  chunked %.4f" % (exact[-1], seq[-1], chunked[-1]))


def round_sig(v, p, mode):
    """Round v to p significant bits: 'near' = nearest, ties to even; 'trunc' = toward zero."""
    if v == 0:
        return 0.0
    m, e = math.frexp(v)                      # v = m 2^e, 0.5 <= |m| < 1
    t = m * 2.0 ** p
    r = round(t) if mode == "near" else math.trunc(t)   # Python round(): ties to even
    return math.ldexp(r, e - p)


def total(vals, rnd, chunk=None):
    """Running sum rounded by rnd after every addition; with chunk, the partial sum is
    added into an FP32 total every `chunk` terms and restarted from zero."""
    tot = part = 0.0
    for k, v in enumerate(vals):
        part = rnd(part + v)
        if chunk and (k + 1) % chunk == 0:
            tot, part = float(fp32(tot + part)), 0.0
    return float(fp32(tot + part)) if chunk else part


ex_sum = float(exact[-1])
print("table of the §5 example (value, relative error):")
for lab, rnd in (("BF16 (p=8), nearest", lambda v: float(bf16(v))),
                 ("p=14, nearest", lambda v: round_sig(v, 14, "near")),
                 ("p=14, truncate", lambda v: round_sig(v, 14, "trunc")),
                 ("FP32 (p=24)", lambda v: float(fp32(v)))):
    a, b = total(xs, rnd), total(xs, rnd, 128)
    print("  %-20s sequential %.5f (%.2e)   promoted every 128 %.5f (%.2e)"
          % (lab, a, abs(a - ex_sum) / ex_sum, b, abs(b - ex_sum) / ex_sum))

W, H = 760, 352
svg = SVG(W, H, "低精度累加：顺序相加会停住，分段提升到 FP32 就不会")
P = [(62, 346, 60, 300), (446, 740, 60, 300)]   # x0, x1, y0(top), y1(bottom)


def axes(i, xmax, ymax, xt, yt, title, xlabel):
    x0, x1, y0, y1 = P[i]
    fx = lambda v: x0 + v / xmax * (x1 - x0)
    fy = lambda v: y1 - v / ymax * (y1 - y0)
    for v in yt:
        svg.line(x0, fy(v), x1, fy(v), color="grid", width=1)
        svg.text(x0 - 6, fy(v) + 4, "%g" % v, size=11, anchor="end", color="muted")
    for v in xt:
        svg.line(fx(v), y1, fx(v), y1 + 4, color="muted", width=1)
        svg.text(fx(v), y1 + 18, "%g" % v, size=11, anchor="middle", color="muted")
    svg.line(x0, y1, x1, y1, color="muted", width=1.2)
    svg.line(x0, y0, x0, y1, color="muted", width=1.2)
    svg.text(x0, y0 - 22, title, size=13, weight="600")
    svg.text((x0 + x1) / 2, y1 + 38, xlabel, size=12, anchor="middle", color="muted")
    return fx, fy


def poly(fx, fy, ys, step=1):
    idx = list(range(0, len(ys), step)) + [len(ys) - 1]
    return "M" + " L".join("%.1f %.1f" % (fx(i), fy(ys[i])) for i in idx)


# left panel
fx, fy = axes(0, N1, 3000, (0, 1000, 2000, 3000), (0, 500, 1000, 1500, 2000, 2500, 3000),
              "(a) 把 1 连加 3000 次", "已加的项数")
svg.path(poly(fx, fy, np.arange(N1 + 1.0), 10), stroke="ink", width=1.3, dash="5 4")
svg.path(poly(fx, fy, r_fp16, 10), stroke="blue", width=2.4)
svg.path(poly(fx, fy, r_bf, 10), stroke="orange", width=2.4)
svg.text(fx(2350), fy(2700), "精确值", size=12, anchor="end")
svg.text(fx(2600), fy(2048) + 20, "FP16 停在 2048", size=12, anchor="middle", color="blue")
svg.text(fx(2000), fy(256) - 8, "BF16 停在 256", size=12, anchor="middle", color="orange")

# right panel
fx, fy = axes(1, 4096, 22, (0, 1024, 2048, 3072, 4096), (0, 5, 10, 15, 20),
              "(b) 4096 个 U(0, 0.01) 求和", "已加的项数")
svg.path(poly(fx, fy, exact, 8), stroke="ink", width=1.3, dash="5 4")
svg.path(poly(fx, fy, chunked, 8), stroke="green", width=2.2)
svg.path(poly(fx, fy, seq, 8), stroke="orange", width=2.4)
svg.text(fx(150), fy(20.5), "虚线 = 精确和 %.2f（与绿线几乎重合）" % exact[-1], size=12)
# the green line runs diagonally across the panel, so its label sits in the empty upper-left triangle
svg.text(fx(150), fy(17.2), "绿线：每 128 个提升到 FP32，", size=12, color="green")
svg.text(fx(150), fy(17.2) + 17, "终值 %.2f" % chunked[-1], size=12, color="green")
svg.text(fx(4096) - 4, fy(seq[-1]) - 10, "BF16 顺序累加：%.2f" % seq[-1], size=12, anchor="end",
         color="orange")
svg.text(fx(1150), fy(4) + 17, "和到 4 以后 ulp = 1/32，半个 ulp > 0.01，", size=12, color="muted")
svg.text(fx(1150), fy(4) + 33, "再加什么都被舍掉", size=12, color="muted")

svg.save("m08-accumulation.svg")
