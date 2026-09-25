"""Figure M8.7: why promoting the partial sum every N_C terms works.

Top (a): the data flow of DeepSeek-V3's FP8 GEMM accumulation (schematic, numbers from the V3
report as transcribed in deepseek/reference-docs/lesson-12-doc.md C44-C46): 128 products are
accumulated in the Tensor Core's limited-precision accumulator (about 14 bits), then the partial
sum is copied to an FP32 register on the CUDA cores, multiplied by the two scaling factors and
added there; the low-precision accumulator starts again from zero.

Bottom (b): the mechanism, computed on the experiment of §5 (4096 draws from U(0, 0.01),
numpy default_rng(0), inputs rounded to BF16, BF16 used as the low-precision accumulator).
For every term k we plot half an ulp of the number sitting in the low-precision accumulator
after k additions: (orange) the running sum of everything so far, (green) the running sum of the
current block of 128 only.  An addend smaller than this half-ulp is swamped (M8.8).  The grey
band [0.001, 0.01] holds 89.4% of the addends (printed below).
Writes math/figures/m08-promotion.svg
"""

import math

import numpy as np

from m08_fp import bf16
from m08_svg import SVG

M_BF16 = 7
NC = 128
xs = bf16(np.random.default_rng(0).uniform(0, 0.01, 4096))


def half_ulp(s):
    """Half the spacing of BF16 numbers around s (s > 0)."""
    return 2.0 ** (math.floor(math.log2(s)) - M_BF16 - 1)


seq, blk = [], []
s = p = 0.0
for k, x in enumerate(xs):
    s = float(bf16(s + x))
    p = float(bf16(p + x)) if k % NC else float(bf16(x))
    seq.append(s)
    blk.append(p)
k4 = next(i for i, v in enumerate(seq) if v >= 4) + 1           # first term at which the sum is 4
blk_ends = [blk[j + NC - 1] for j in range(0, len(xs), NC)]
lo_band, hi_band = 0.001, 0.01
frac = float(np.mean((xs >= lo_band) & (xs <= hi_band)))
print("sum reaches 4 at term %d; block sums %.4f..%.4f; %.1f%% of addends in [0.001, 0.01]"
      % (k4, min(blk_ends), max(blk_ends), 100 * frac))
assert all(0.5 <= v < 1 for v in blk_ends)                      # every block ends in [0.5, 1)

W, H = 760, 500
svg = SVG(W, H, "分段提升为什么有效：让低精度累加器里的数永远不长大")

# ------------------------------------------------------------------ (a) data flow
svg.text(16, 24, "(a) DeepSeek-V3 的 FP8 矩阵乘法里，一个输出元素怎样累加（N_{C} = 128）", size=13, weight="600")
# a strip of products
sx, sy, cw, ch = 24, 66, 11, 26
for i in range(14):
    svg.rect(sx + i * cw, sy, cw - 1.5, ch, fill="orange", fill_opacity=0.16 if i < 13 else 0.0,
             stroke="orange", width=0.8, stroke_opacity=0.7)
svg.text(sx + 13 * cw + 4, sy + 17, "…", size=12, anchor="middle", color="orange")
svg.text(sx, sy - 10, "第 j 块：128 个乘积", size=12)
svg.text(sx, sy + ch + 18, "FP8 输入逐个相乘", size=12, color="muted")
# low-precision accumulator
ax0, ay0, aw, ah = 232, 52, 190, 56
svg.arrow(sx + 14 * cw + 6, sy + ch / 2, ax0 - 4, sy + ch / 2, color="ink", width=1.5)
svg.rect(ax0, ay0, aw, ah, fill="orange", fill_opacity=0.12, stroke="orange", width=1.4, rx=6)
svg.text(ax0 + aw / 2, ay0 + 23, "Tensor Core 累加器", size=13, anchor="middle", weight="600")
svg.text(ax0 + aw / 2, ay0 + 43, "约 14 位有效数字", size=12, anchor="middle", color="orange")
svg.text(ax0 + aw / 2, ay0 + ah + 18, "只装第 j 块的部分和", size=12, anchor="middle", color="muted")
svg.text(ax0 + aw / 2, ay0 + ah + 34, "装满 128 项就清零重来", size=12, anchor="middle", color="muted")
# FP32 accumulator
fx0, fw = 540, 204
svg.arrow(ax0 + aw + 4, ay0 + ah / 2, fx0 - 4, ay0 + ah / 2, color="ink", width=1.5)
svg.text((ax0 + aw + fx0) / 2, ay0 + ah / 2 - 8, "每 128 项", size=12, anchor="middle")
svg.text((ax0 + aw + fx0) / 2, ay0 + ah / 2 + 18, "复制出去", size=12, anchor="middle")
svg.rect(fx0, ay0, fw, ah, fill="green", fill_opacity=0.10, stroke="green", width=1.4, rx=6)
svg.text(fx0 + fw / 2, ay0 + 23, "CUDA core 上的 FP32 寄存器", size=13, anchor="middle", weight="600")
svg.text(fx0 + fw / 2, ay0 + 43, "总和 += 部分和 × s_{a} s_{w}", size=12, anchor="middle", color="green")
svg.text(fx0 + fw / 2, ay0 + ah + 18, "在这里乘上两个缩放因子", size=12, anchor="middle", color="muted")
svg.text(fx0 + fw / 2, ay0 + ah + 34, "FP32 有 24 位有效数字", size=12, anchor="middle",
         color="muted")

# ------------------------------------------------------------------ (b) mechanism
X0, X1, Y0, Y1 = 82, 596, 222, 440
N_SHOW = 12 * NC
L0, L1 = -17, -5


def px(k):
    return X0 + k / N_SHOW * (X1 - X0)


def py(v):
    return Y1 - (math.log2(v) - L0) / (L1 - L0) * (Y1 - Y0)


svg.text(16, 196, "(b) 为什么有效：累加器里的数不长大，它的半个 ulp 就一直小于大多数加数"
         "（§5 的 BF16 实验）", size=13, weight="600")
# addend band and the largest addend
svg.rect(X0, py(hi_band), X1 - X0, py(lo_band) - py(hi_band), fill="muted", fill_opacity=0.13)
svg.line(X0, py(hi_band), X1, py(hi_band), color="ink", width=1.2, dash="5 4")
for e in range(L0 + 1, L1 + 1, 2):
    svg.line(X0, py(2.0 ** e), X1, py(2.0 ** e), color="grid", width=1)
    svg.text(X0 - 6, py(2.0 ** e) + 4, "2^{%d}" % e, size=11, anchor="end", color="muted")
for k in range(0, N_SHOW + 1, 256):
    svg.line(px(k), Y1, px(k), Y1 + 4, color="muted", width=1)
    svg.text(px(k), Y1 + 18, "%d" % k, size=11, anchor="middle", color="muted")
for j in range(NC, N_SHOW, NC):                                   # block boundaries
    svg.line(px(j), Y0, px(j), Y1, color="green", width=0.8, dash="2 3", opacity=0.5)
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "已加的项数（绿色细虚线：每 128 项一块）", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "累加器里那个数的半个 ulp（对数刻度）", size=12, anchor="middle", color="muted",
         rotate=-90)


def steps(vals):
    """Step path of half_ulp(vals[k]) for k = 0..N_SHOW-1, clipped to the plot's y range."""
    pts, prev = [], None
    for k in range(N_SHOW):
        y = py(max(half_ulp(vals[k]), 2.0 ** L0))
        if prev is None:
            pts.append("M%.1f %.1f" % (px(k), y))
        elif abs(y - prev) > 0.01:
            pts.append("L%.1f %.1f L%.1f %.1f" % (px(k), prev, px(k), y))
        prev = y
    pts.append("L%.1f %.1f" % (px(N_SHOW), prev))
    return " ".join(pts)


svg.path(steps(blk), stroke="green", width=1.8, join="round")
svg.path(steps(seq), stroke="orange", width=2.4, join="round")

# annotations
svg.circle(px(k4), py(half_ulp(4.0)), 4, fill="orange")
svg.text(px(k4) - 10, py(half_ulp(4.0)) - 12, "第 %d 项，和到 4：半个 ulp = 1/64 > 0.01，" % k4, size=12,
         anchor="end", color="orange", halo=True)
svg.text(px(k4) - 10, py(half_ulp(4.0)) + 4, "此后每一项都被吞没", size=12, anchor="end",
         color="orange", halo=True)
svg.text(X1 - 6, py(hi_band) + 16, "最大的加数 0.01", size=12, anchor="end", halo=True)

# legend
lx, ly = X1 + 16, Y0 + 14
svg.line(lx, ly - 4, lx + 24, ly - 4, color="orange", width=2.4)
svg.text(lx + 30, ly, "顺序累加", size=12, color="orange")
svg.text(lx + 30, ly + 16, "和一直长大", size=12, color="muted")
ly += 44
svg.line(lx, ly - 4, lx + 24, ly - 4, color="green", width=1.8)
svg.text(lx + 30, ly, "分段累加", size=12, color="green")
svg.text(lx + 30, ly + 16, "每 128 项清零，", size=12, color="muted")
svg.text(lx + 30, ly + 32, "半个 ulp ≤ 1/512", size=12, color="muted")
ly += 60
svg.rect(lx, ly - 12, 24, 14, fill="muted", fill_opacity=0.13)
svg.text(lx + 30, ly, "%.0f%% 的加数" % (100 * frac), size=12, color="muted")
svg.text(lx + 30, ly + 16, "在 0.001–0.01", size=12, color="muted")
ly += 48
svg.text(lx, ly, "加数小于曲线的", size=12, color="muted")
svg.text(lx, ly + 17, "高度就被吞没", size=12, color="muted")
svg.text(lx, ly + 34, "（M8.8）", size=12, color="muted")

svg.save("m08-promotion.svg")
