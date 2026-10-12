"""Figure 17.3: hash collisions are the normal case, but a collision in two heads at once is rare.

(a) Load lambda = U/p (U distinct N-grams hashed uniformly into one table of p rows).
    One head: P(a given N-gram shares its row with someone) = 1 - exp(-lambda);
    all 8 heads shared (independence approximation) = (1 - exp(-lambda))^8.
(b) Expected number of *other* N-grams that collide with a given one in >= 2 of the 8 heads:
    U * C(8,2) / p^2, for p = 2,262,409 (Engram-27B layout) and 16,000,057 (V4.1).
Key values (notes §4.7): p=2.26M, U=1e7 -> 0.988 / 0.908; U=1e10 -> 0.055 (27B) / 0.0011 (V4.1).
The U values are hypothetical: the paper does not report how many distinct N-grams exist.
Writes deepseek/figures/l17-collision.svg
"""

import math

from l17_svg import SVG
from l17_num import E27_PRIMES, V41_PRIMES

P27, P41 = E27_PRIMES[0][0][0], V41_PRIMES[0][0][0]
assert (P27, P41) == (2_262_409, 16_000_057)


def one(lam):
    return 1 - math.exp(-lam)


def partners(U, p):
    return U * 28 / p ** 2


assert round(one(1e7 / P27), 3) == 0.988 and round(one(1e7 / P27) ** 8, 3) == 0.908
assert round(partners(1e10, P27), 3) == 0.055 and round(partners(1e10, P41), 4) == 0.0011

SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def p10(e):
    return {0: "1", 1: "10"}.get(e, "10" + str(e).translate(SUP))


W, H = 760, 410
svg = SVG(W, H, "哈希碰撞：单个头里几乎必撞，8 个头里同时撞两次却极少")

# ---------------------------------------------------------------- (a)
AX0, AX1, AY0, AY1 = 70, 350, 48, 300
L0, L1 = -2, 2  # log10 lambda


def ax(lam):
    return AX0 + (math.log10(lam) - L0) / (L1 - L0) * (AX1 - AX0)


def ay(p):
    return AY1 - p * (AY1 - AY0)


svg.text(20, 22, "(a) 一个 N-gram 的行被别人占用的概率", size=13, weight="700")
for p in (0, 0.25, 0.5, 0.75, 1.0):
    svg.line(AX0, ay(p), AX1, ay(p), color="grid", width=1)
    svg.text(AX0 - 8, ay(p) + 4, "%g" % p, size=11, anchor="end", color="muted")
for e in range(L0, L1 + 1):
    svg.line(ax(10 ** e), AY0, ax(10 ** e), AY1, color="grid", width=1)
    svg.text(ax(10 ** e), AY1 + 17, p10(e), size=11, anchor="middle", color="muted")
svg.line(AX0, AY1, AX1, AY1, color="muted", width=1.2)
svg.line(AX0, AY0, AX0, AY1, color="muted", width=1.2)
svg.text((AX0 + AX1) / 2, AY1 + 38, "负载 λ = U / p（对数轴）", size=12, anchor="middle", color="muted")

N = 160
lams = [10 ** (L0 + (L1 - L0) * i / N) for i in range(N + 1)]
svg.path("M" + " L".join("%.1f %.1f" % (ax(l), ay(one(l))) for l in lams), stroke="amber", width=2.4)
svg.path("M" + " L".join("%.1f %.1f" % (ax(l), ay(one(l) ** 8)) for l in lams), stroke="red", width=2.4)
svg.text(ax(1.25), ay(0.85), "一个头：1 − exp(−λ)", size=12, anchor="end", color="amber", weight="700")
# right of the dashed U = 1e7 line (lambda = 4.42) and right of the red curve at this height
svg.text(ax(6.0), ay(0.30), "8 个头", size=12, color="red", weight="700")
svg.text(ax(6.0), ay(0.30) + 16, "全被占", size=12, color="red", weight="700")

for lam in (1e7 / P27, 1e7 / P41):
    for f, col in ((one(lam), "amber"), (one(lam) ** 8, "red")):
        svg.circle(ax(lam), ay(f), 4.2, fill=col, stroke="surface", width=1.2)
    svg.line(ax(lam), AY1, ax(lam), ay(one(lam)), color="muted", width=1, dash="3 3")
svg.text(ax(1e7 / P27) - 6, ay(one(1e7 / P27)) - 10, "0.988", size=11, anchor="end", color="amber")
svg.text(ax(1e7 / P27) + 7, ay(one(1e7 / P27) ** 8) + 4, "0.908", size=11, color="red")
svg.text(ax(1e7 / P41) - 6, ay(one(1e7 / P41)) - 8, "0.465", size=11, anchor="end", color="amber")
svg.text(ax(1e7 / P41) - 7, ay(one(1e7 / P41) ** 8) - 7, "0.002", size=11, anchor="end", color="red")
svg.text(AX0 + 8, ay(0.74), "竖虚线：U = 10⁷ 时", size=11.5, color="muted")
svg.text(AX0 + 8, ay(0.74) + 16, "左 V4.1（λ = 0.625）", size=11.5, color="muted")
svg.text(AX0 + 8, ay(0.74) + 32, "右 27B（λ = 4.42）", size=11.5, color="muted")

# ---------------------------------------------------------------- (b)
BX0, BX1, BY0, BY1 = 450, 740, 48, 300
U0, U1 = 6, 11      # log10 U
V0, V1 = -8, 1      # log10 expected partners


def bx(U):
    return BX0 + (math.log10(U) - U0) / (U1 - U0) * (BX1 - BX0)


def by(v):
    return BY1 - (math.log10(v) - V0) / (V1 - V0) * (BY1 - BY0)


svg.text(400, 22, "(b) 与它在 ≥ 2 个头上同时相撞的期望伙伴数", size=13, weight="700")
for e in range(V0, V1 + 1):
    svg.line(BX0, by(10 ** e), BX1, by(10 ** e), color="grid", width=1)
    if e % 2 == 0 or e == 1:
        svg.text(BX0 - 8, by(10 ** e) + 4, p10(e), size=11, anchor="end", color="muted")
for e in range(U0, U1 + 1):
    svg.line(bx(10 ** e), BY0, bx(10 ** e), BY1, color="grid", width=1)
    svg.text(bx(10 ** e), BY1 + 17, p10(e), size=11, anchor="middle", color="muted")
svg.line(BX0, BY1, BX1, BY1, color="muted", width=1.2)
svg.line(BX0, BY0, BX0, BY1, color="muted", width=1.2)
svg.text((BX0 + BX1) / 2, BY1 + 38, "不同 N-gram 的个数 U（对数轴，假设值）", size=12, anchor="middle", color="muted")
svg.line(BX0, by(1), BX1, by(1), color="ink", width=1, dash="5 4")
svg.text(BX1 - 4, by(1) - 6, "平均 1 个伙伴", size=11, anchor="end")

for p, col, lab in ((P27, "amber", "p = 2,262,409（27B）"), (P41, "violet", "p = 16,000,057（V4.1）")):
    x0, x1 = bx(10 ** U0), bx(10 ** U1)
    svg.line(x0, by(partners(10 ** U0, p)), x1, by(partners(10 ** U1, p)), color=col, width=2.4)
# placed above the amber line over its whole width (the line rises 28 units per 58 across)
svg.text(bx(10 ** 6.05), by(10 ** -2.5), "p = 2,262,409（27B）", size=12, color="amber",
         weight="700")
svg.text(bx(10 ** 7.4), by(partners(10 ** 7.4, P41)) + 20, "p = 16,000,057（V4.1）", size=12, color="violet",
         weight="700")
for p, col, t in ((P27, "amber", "0.055"), (P41, "violet", "0.0011")):
    v = partners(1e10, p)
    svg.circle(bx(1e10), by(v), 4.2, fill=col, stroke="surface", width=1.2)
    svg.text(bx(1e10) - 8, by(v) - 8, t, size=11, anchor="end", color=col)

svg.text(20, 370, "(b) 的公式：U × C(8,2) / p²。同一阶 8 个头共用一个 mix < 2⁶³，而任意 3 个表大小之积 ≥ 1.158×10¹⁹ > 2⁶³，",
         size=12, color="muted")
svg.text(20, 390, "所以两个不同的 mix 最多只能在 2 个头上同时相撞（前提：哈希结构同开源代码）。U 为假设值，论文未给出。",
         size=12, color="muted")

svg.save("l17-collision.svg")
