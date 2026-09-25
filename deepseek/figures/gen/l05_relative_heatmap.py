"""Figure 5.5: numerical check of RoPE's relative-position property.

d_h = 128, b = 10000, q and k drawn once from numpy.random.default_rng(0).standard_normal(128)
(the same draw reproduces the numbers 1.692714... and 10.78394... in research notes R11).
Left: score S[t, j] = (R_t q)^T (R_j k) for t, j = 0..47 (causal: j <= t); every diagonal is constant.
Right: the scores of rows t = 47 and t = 30 plotted against distance t - j fall on one curve.
Writes deepseek/figures/l05-relative-heatmap.svg
"""

import numpy as np

from l05_svg import SVG

D, B, N = 128, 10000.0, 48
th = B ** (-2 * np.arange(D // 2) / D)


def rope(x, t):
    x = x.reshape(-1, 2)
    c, s = np.cos(t * th), np.sin(t * th)
    return np.stack([x[:, 0] * c - x[:, 1] * s, x[:, 0] * s + x[:, 1] * c], 1).reshape(-1)


rng = np.random.default_rng(0)
q = rng.standard_normal(D)
k = rng.standard_normal(D)
S = np.array([[rope(q, t) @ rope(k, j) for j in range(N)] for t in range(N)])
# Toeplitz check on the causal half
for dlt in range(N):
    diag = np.array([S[t, t - dlt] for t in range(dlt, N)])
    assert diag.max() - diag.min() < 1e-9
vmax = np.abs(np.tril(S)).max()

W, H = 760, 452
svg = SVG(W, H, "RoPE 得分矩阵沿对角线不变（d_h = 128，随机 q、k）")
C = 6.4
X0, Y0 = 70, 58


def color_cell(v):
    if v >= 0:
        return "blue", min(1.0, v / vmax) * 0.85 + 0.04
    return "red", min(1.0, -v / vmax) * 0.85 + 0.04


svg.text(X0 + N * C / 2, 26, "得分 S[t, j] = (R_{t}*q*)·(R_{j}*k*)", size=13, anchor="middle", weight="600")
svg.text(X0 + N * C / 2, 44, "键位置 j →", size=12, anchor="middle", color="muted")
svg.text(X0 - 30, Y0 + N * C / 2, "查询位置 t →", size=12, anchor="middle", color="muted", rotate=-90)
for t in range(N):
    for j in range(N):
        x, y = X0 + j * C, Y0 + t * C
        if j > t:
            continue
        col, op = color_cell(S[t, j])
        svg.rect(x, y, C, C, fill=col, fill_opacity=op)
svg.rect(X0, Y0, N * C, N * C, stroke="rule", width=1)
# upper triangle label
svg.text(X0 + N * C * 0.72, Y0 + N * C * 0.28, "j > t：因果掩码", size=12, anchor="middle", color="muted")
for v in (0, 16, 32, 47):
    svg.text(X0 + v * C + C / 2, Y0 + N * C + 15, str(v), size=11, anchor="middle", color="muted")
    svg.text(X0 - 4, Y0 + v * C + C / 2 + 4, str(v), size=11, anchor="end", color="muted")
# mark one diagonal: distance 10
d10 = [(X0 + (t - 10) * C + C / 2, Y0 + t * C + C / 2) for t in (10, N - 1)]
svg.line(d10[0][0], d10[0][1], d10[1][0], d10[1][1], color="ink", width=1.2, dash="3 2")
svg.text(X0 + 21 * C, Y0 + 19 * C, "← Δ = 10 的对角线", size=12, color="ink")

# colour legend
lx, ly = X0, Y0 + N * C + 34
for n in range(11):
    v = -vmax + 2 * vmax * n / 10
    col, op = color_cell(v)
    svg.rect(lx + n * 14, ly, 14, 10, fill=col, fill_opacity=op)
svg.text(lx, ly + 24, "%.0f" % -vmax, size=11, color="muted")
svg.text(lx + 77, ly + 24, "0", size=11, anchor="middle", color="muted")
svg.text(lx + 154, ly + 24, "+%.0f" % vmax, size=11, anchor="end", color="muted")

# right panel: score vs distance for two rows
PX0, PX1, PY0, PY1 = 440, 730, 70, 330
ymin, ymax = -vmax * 1.05, vmax * 1.05


def ppx(dl):
    return PX0 + dl / (N - 1) * (PX1 - PX0)


def ppy(v):
    return PY1 - (v - ymin) / (ymax - ymin) * (PY1 - PY0)


svg.text((PX0 + PX1) / 2, 26, "同一行（固定 t）按距离 Δ = t − j 画", size=13, anchor="middle", weight="600")
for v in (-8, -4, 0, 4, 8):
    if ymin < v < ymax:
        y = ppy(v)
        svg.line(PX0, y, PX1, y, color="grid", width=1)
        svg.text(PX0 - 6, y + 4, str(v), size=11, anchor="end", color="muted")
svg.line(PX0, PY1, PX1, PY1, color="muted", width=1.2)
svg.line(PX0, PY0, PX0, PY1, color="muted", width=1.2)
for dl in (0, 10, 20, 30, 40):
    svg.text(ppx(dl), PY1 + 16, str(dl), size=11, anchor="middle", color="muted")
svg.text((PX0 + PX1) / 2, PY1 + 34, "距离 Δ = t − j", size=12, anchor="middle", color="muted")
row47 = [(ppx(dl), ppy(S[47, 47 - dl])) for dl in range(48)]
svg.polyline(row47, stroke="blue", width=2)
for dl in range(31):
    svg.circle(ppx(dl), ppy(S[30, 30 - dl]), 3.2, stroke="violet", width=1.4)
svg.text(PX1 - 4, PY0 + 4, "线：t = 47 那一行", size=12, anchor="end", color="blue")
svg.text(PX1 - 4, PY0 + 22, "圈：t = 30 那一行", size=12, anchor="end", color="violet")
svg.text(PX0 + 8, PY1 + 60, "两行的点落在同一条曲线上：得分只由 Δ 决定。", size=12, color="ink")
svg.text(PX0 + 8, PY1 + 78, "Δ = 0 时是 *q*·*k* = %.4f；Δ = 26 时是 %.6f。" % (S[0, 0], S[26, 0]), size=12,
         color="ink")

svg.save("l05-relative-heatmap.svg")
