"""Figure 5.6: RoPE "long-term decay" (research notes R9, R10; RoFormer Fig. 2 redrawn and extended).

(a) RoFormer's quantity U(D) = (1/(d/2)) * sum_{i=1}^{d/2} |S_i|, S_i = sum_{k<i} exp(1j*D*theta_k),
    theta_k = 10000^(-2k/d), d = 128, for D = 0..256 (the range of RoFormer Fig. 2).
(b) the same quantity for D = 1..65536 on a log axis: min-max band per pixel plus a centred 512-point
    moving average, showing that the decline stops near D ~ 1000 while the pointwise value keeps
    swinging between roughly 2 and 11.
(c) the expected score of a query with itself at distance D, E[q^T R_D q]/d_h = (2/d_h) sum_i cos(D theta_i)
    for q with identity covariance (R10); an independent pair (q, k) has expected score 0 at every distance.
Writes deepseek/figures/l05-decay.svg
"""

import math

import numpy as np

from l05_svg import SVG

D = 128
th = 10000.0 ** (-2 * np.arange(D // 2) / D)
MAXD = 65536
deltas = np.arange(0, MAXD + 1)
U = np.empty(MAXD + 1)
for s in range(0, MAXD + 1, 4096):
    blk = deltas[s:s + 4096]
    Sc = np.cumsum(np.exp(1j * np.outer(blk, th)), axis=1)
    U[s:s + 4096] = np.abs(Sc).mean(axis=1)
assert abs(U[0] - 32.5) < 1e-9 and abs(U[5] - 20.845) < 1e-3 and abs(U[50] - 12.629) < 1e-3
same = np.array([2 * np.cos(dl * th).sum() / D for dl in deltas])
assert abs(same[10] - 0.6691) < 1e-4 and abs(same[1000] - 0.1590) < 1e-4
win = 512
csum = np.concatenate([[0.0], np.cumsum(U)])


def movavg(i):
    a, b = max(0, i - win // 2), min(MAXD, i + win // 2)
    return (csum[b + 1] - csum[a]) / (b + 1 - a)


def envelope(vals, xf, yf):
    """Polyline through each pixel column's first, min and max points (exact where sparse, a dense band where not)."""
    cols = {}
    for dl in range(1, MAXD + 1):
        c = int(xf(dl) * 2) / 2
        cols.setdefault(c, []).append(vals[dl])
    pts = []
    for c in sorted(cols):
        v = cols[c]
        if len(v) == 1:
            pts.append((c, yf(v[0])))
        else:
            pts += [(c, yf(min(v))), (c, yf(max(v)))]
    return pts


seg = U[1000:4097]
print("U on [1000,4096]: min %.2f max %.2f" % (seg.min(), seg.max()))

W, H = 760, 600
svg = SVG(W, H, "RoPE 的远程衰减：RoFormer 的上界量与同向向量的平均得分")

# ---------------- panel (a)
AX0, AX1, AY0, AY1 = 62, 340, 56, 262
YMAX = 35


def ax(dl):
    return AX0 + dl / 256 * (AX1 - AX0)


def ay(v):
    return AY1 - v / YMAX * (AY1 - AY0)


svg.text((AX0 + AX1) / 2, 24, "(a) RoFormer 图 2 的范围：Δ = 0 … 256", size=13, anchor="middle", weight="600")
for v in range(0, 36, 5):
    svg.line(AX0, ay(v), AX1, ay(v), color="grid", width=1)
    svg.text(AX0 - 6, ay(v) + 4, str(v), size=11, anchor="end", color="muted")
for dl in (0, 64, 128, 192, 256):
    svg.text(ax(dl), AY1 + 16, str(dl), size=11, anchor="middle", color="muted")
svg.line(AX0, AY1, AX1, AY1, color="muted", width=1.2)
svg.line(AX0, AY0, AX0, AY1, color="muted", width=1.2)
svg.text((AX0 + AX1) / 2, AY1 + 34, "相对距离 Δ", size=12, anchor="middle", color="muted")
svg.text(20, (AY0 + AY1) / 2, "上界里的平均量 U(Δ)", size=12, anchor="middle", color="muted", rotate=-90)
svg.polyline([(ax(dl), ay(U[dl])) for dl in range(257)], stroke="blue", width=1.8)
for dl in (5, 50, 256):
    svg.circle(ax(dl), ay(U[dl]), 3.2, fill="blue")
svg.text(ax(5) + 7, ay(U[5]) - 4, "Δ=5：%.2f" % U[5], size=11.5, color="blue")
svg.text(ax(50) + 7, ay(U[50]) - 7, "Δ=50：%.2f" % U[50], size=11.5, color="blue")
svg.text(ax(256) - 4, ay(U[256]) + 22, "Δ=256：%.2f" % U[256], size=11.5, anchor="end", color="blue")
svg.text(ax(0) + 6, ay(U[0]) + 2, "Δ=0：32.5", size=11.5, color="blue")

# ---------------- panel (b)
BX0, BX1, BY0, BY1 = 420, 740, 56, 262
LX0, LX1 = 0, math.log2(MAXD)


def bx(dl):
    return BX0 + (math.log2(dl) - LX0) / (LX1 - LX0) * (BX1 - BX0)


def by(v):
    return BY1 - v / YMAX * (BY1 - BY0)


svg.text((BX0 + BX1) / 2, 24, "(b) 延长到 Δ = 65536（对数横轴）", size=13, anchor="middle", weight="600")
for v in range(0, 36, 5):
    svg.line(BX0, by(v), BX1, by(v), color="grid", width=1)
    svg.text(BX0 - 6, by(v) + 4, str(v), size=11, anchor="end", color="muted")
for dl, lab in ((1, "1"), (16, "16"), (256, "256"), (4096, "4096"), (65536, "65536")):
    svg.text(bx(dl), BY1 + 16, lab, size=11, anchor="middle", color="muted")
    svg.line(bx(dl), BY0, bx(dl), BY1, color="grid", width=1)
svg.line(BX0, BY1, BX1, BY1, color="muted", width=1.2)
svg.line(BX0, BY0, BX0, BY1, color="muted", width=1.2)
svg.text((BX0 + BX1) / 2, BY1 + 34, "相对距离 Δ（对数刻度）", size=12, anchor="middle", color="muted")
svg.polyline(envelope(U, bx, by), stroke="blue", width=0.8, opacity=0.45)
svg.polyline([(bx(dl), by(movavg(dl))) for dl in np.unique(np.logspace(math.log10(256), LX1 * math.log10(2), 200).astype(int))],
             stroke="blue", width=2.4)
# start the marker below the three legend lines (by(31) … by(31)+32) so it does not cut through them
svg.line(bx(1000), by(31) + 42, bx(1000), BY1, color="amber", width=1.3, dash="4 3")
svg.text(bx(1000) - 4, BY1 - 8, "Δ≈1000 之后基本持平", size=11.5, anchor="end", color="amber")
svg.text(BX1 - 4, by(31), "细线：逐点的值（密处成带）", size=11.5, anchor="end", color="blue")
svg.text(BX1 - 4, by(31) + 16, "粗线：512 点滑动平均（Δ ≥ 256）", size=11.5, anchor="end", color="blue")
svg.text(BX1 - 4, by(31) + 32, "[1000, 4096] 内逐点 %.2f – %.2f" % (seg.min(), seg.max()), size=11.5, anchor="end",
         color="ink")

# ---------------- panel (c)
CX0, CX1, CY0, CY1 = 62, 740, 350, 530
CYMIN, CYMAX = -0.3, 1.05


def cx(dl):
    return CX0 + math.log2(dl) / LX1 * (CX1 - CX0)


def cy(v):
    return CY1 - (v - CYMIN) / (CYMAX - CYMIN) * (CY1 - CY0)


svg.text((CX0 + CX1) / 2, 318, "(c) 同向的 q、k：平均得分 E[*q*^{⊤}R_{Δ}*q*] / d_{h} = (2/d_{h}) Σ cos(Δθ_{i})（d_{h} = 128）",
         size=13, anchor="middle", weight="600")
for v in (-0.25, 0, 0.25, 0.5, 0.75, 1.0):
    svg.line(CX0, cy(v), CX1, cy(v), color="grid" if v else "muted", width=1 if v else 1.2,
             dash=None if v else "5 4")
    svg.text(CX0 - 6, cy(v) + 4, ("%.2f" % v).rstrip("0").rstrip(".") if v else "0", size=11, anchor="end",
             color="muted")
for dl, lab in ((1, "1"), (16, "16"), (256, "256"), (4096, "4096"), (65536, "65536")):
    svg.text(cx(dl), CY1 + 16, lab, size=11, anchor="middle", color="muted")
    svg.line(cx(dl), CY0, cx(dl), CY1, color="grid", width=1)
svg.line(CX0, CY1, CX1, CY1, color="muted", width=1.2)
svg.text((CX0 + CX1) / 2, CY1 + 34, "相对距离 Δ（对数刻度）", size=12, anchor="middle", color="muted")
svg.polyline(envelope(same, cx, cy), stroke="green", width=0.8, opacity=0.55)
for dl in (10, 100, 1000, 4096):
    svg.circle(cx(dl), cy(same[dl]), 3.4, fill="green")
    if dl < 500:
        svg.text(cx(dl) + 6, cy(same[dl]) - 7, "Δ=%d：%.3f" % (dl, same[dl]), size=11.5, color="green")
    else:
        yl = 0.66 if dl == 1000 else 0.52
        svg.line(cx(dl), cy(same[dl]), cx(dl), cy(yl), color="green", width=0.9)
        svg.text(cx(dl) + (10 if dl == 1000 else -10), cy(yl) - 5, "Δ=%d：%.3f" % (dl, same[dl]), size=11.5,
                 anchor="end" if dl == 1000 else "start", color="green", weight="600")
svg.text(cx(1) + 8, cy(0) + 18, "虚线 0：独立随机的 q、k 在任何距离的期望得分", size=11.5, color="muted")

svg.text(20, H - 14, "U(Δ) 只是 |q·k| 上界里的一个因子；(c) 是 q 与 k 同向时的平均趋势。两者都不保证某个具体的注意力权重随距离单调变小。",
         size=12, color="ink")

svg.save("l05-decay.svg")
