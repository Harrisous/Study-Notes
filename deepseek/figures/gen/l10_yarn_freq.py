"""Figure 10.11: what V2's YaRN setting does to each of the 32 RoPE frequency pairs of the 64-dim branch.

Setting (V2 sec. 3.1.4; HF config rope_scaling): original context L = 4096, RoPE base b = 10000,
d_h^R = 64 (32 pairs, index i = 0..31), scale s = 40, ramp thresholds alpha = 1, beta = 32.
  theta_i = b^(-2i/64), lambda_i = 2 pi / theta_i, r_i = L / lambda_i        (YaRN eq. 10)
  paper ramp gamma(r): 0 if r < alpha, 1 if r > beta, (r - alpha)/(beta - alpha) otherwise   (YaRN eq. 11)
  code ramp (HF yarn_linear_ramp_mask): linear in i between floor(i(beta)) = 10 and ceil(i(alpha)) = 23
  new frequency theta'_i = [(1 - gamma)/s + gamma] theta_i                     (YaRN eq. 13)
Top panel: gamma (share of the original frequency kept). Bottom: theta'/theta on a log axis.
Writes deepseek/figures/l10-yarn-freq.svg
"""

import math

from l10_svg import SVG

L, B, DH, S, ALPHA, BETA = 4096, 10000, 64, 40, 1, 32
idx = list(range(DH // 2))
theta = [B ** (-2 * i / DH) for i in idx]
lam = [2 * math.pi / t for t in theta]
r = [L / x for x in lam]


def gamma(rr):
    if rr < ALPHA:
        return 0.0
    if rr > BETA:
        return 1.0
    return (rr - ALPHA) / (BETA - ALPHA)


def i_of_r(rr):
    return DH * math.log(L / (2 * math.pi * rr)) / (2 * math.log(B))


low, high = math.floor(i_of_r(BETA)), math.ceil(i_of_r(ALPHA))
assert (low, high) == (10, 23) and abs(i_of_r(BETA) - 10.47) < 0.01 and abs(i_of_r(ALPHA) - 22.51) < 0.01
g_paper = [gamma(x) for x in r]
g_code = [1 - min(max((i - low) / (high - low), 0), 1) for i in idx]
mult_paper = [(1 - g) / S + g for g in g_paper]
mult_code = [(1 - g) / S + g for g in g_code]
assert abs(mult_paper[16] - 0.1986) < 1e-3 and abs(mult_code[16] - 0.5500) < 1e-3

W, H = 760, 470
svg = SVG(W, H, "V2 的 YaRN 对 64 维 RoPE 的 32 对频率各做了什么")
X0, X1 = 86, 700


def px(i):
    return X0 + (i + 0.5) / 32 * (X1 - X0)


# zone shading (by the paper's r thresholds)
zb1 = X0 + (i_of_r(BETA) + 0.5) / 32 * (X1 - X0)
zb2 = X0 + (i_of_r(ALPHA) + 0.5) / 32 * (X1 - X0)
for (xa, xb, col, lab) in ((X0, zb1, "green", "r > 32：保留原频率"), (zb1, zb2, "amber", "过渡"),
                           (zb2, X1, "orange", "r < 1：频率 ÷ 40")):
    svg.rect(xa, 18, xb - xa, 420, fill=col, fill_opacity=0.06)
    svg.text((xa + xb) / 2, 34, lab, size=12, anchor="middle", color=col, weight="600")

# ---- top panel: gamma
TY0, TY1 = 56, 182


def ty(v):
    return TY1 - v * (TY1 - TY0)


for v in (0, 0.5, 1):
    svg.line(X0, ty(v), X1, ty(v), color="grid", width=1)
    svg.text(X0 - 8, ty(v) + 4, "%g" % v, size=11, anchor="end", color="muted")
svg.text(24, (TY0 + TY1) / 2, "γ：保留原频率的份额", size=12, anchor="middle", color="muted", rotate=-90)
svg.path("M" + " L".join("%.1f %.1f" % (px(i), ty(g)) for i, g in zip(idx, g_code)), stroke="violet", width=2)
svg.path("M" + " L".join("%.1f %.1f" % (px(i), ty(g)) for i, g in zip(idx, g_paper)), stroke="blue", width=2.4)
for i, g in zip(idx, g_paper):
    svg.circle(px(i), ty(g), 2.8, fill="blue")
svg.text(px(15.5) + 10, ty(0.62), "HF 代码：对下标 i 线性（10 → 23）", size=12, color="violet")
svg.text(px(11.2), ty(0.035), "论文式：对 r 线性", size=12, color="blue")

# ---- bottom panel: theta'/theta, log scale from 1/40 to 1
BY0, BY1 = 246, 400


def by(v):
    return BY1 - (math.log10(v) - math.log10(1 / S)) / (0 - math.log10(1 / S)) * (BY1 - BY0)


for v, lab in ((1, "1"), (0.5, "1/2"), (0.1, "1/10"), (1 / S, "1/40")):
    svg.line(X0, by(v), X1, by(v), color="grid", width=1)
    svg.text(X0 - 8, by(v) + 4, lab, size=11, anchor="end", color="muted")
svg.text(24, (BY0 + BY1) / 2, "θ′/θ（对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)
svg.path("M" + " L".join("%.1f %.1f" % (px(i), by(m)) for i, m in zip(idx, mult_code)), stroke="violet", width=2)
svg.path("M" + " L".join("%.1f %.1f" % (px(i), by(m)) for i, m in zip(idx, mult_paper)), stroke="blue", width=2.4)
for i, m in zip(idx, mult_paper):
    svg.circle(px(i), by(m), 2.8, fill="blue")
svg.circle(px(16), by(mult_paper[16]), 5, stroke="blue", width=1.4)
svg.circle(px(16), by(mult_code[16]), 5, stroke="violet", width=1.4)
svg.text(px(23.4), by(0.62), "圈出的是 i = 16：", size=12, color="ink")
svg.text(px(23.4), by(0.62) + 17, "论文 0.199，代码 0.550", size=12, color="ink")

# x axis with wavelength labels
svg.line(X0, BY1, X1, BY1, color="muted", width=1.2)
for i in (0, 5, 10, 15, 20, 25, 31):
    svg.text(px(i), BY1 + 16, "%d" % i, size=11, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, BY1 + 34, "频率下标 i（0 = 转得最快）", size=12, anchor="middle", color="muted")
for i in (0, 10, 23, 31):
    lab = "λ = %s" % (("%.0f" % lam[i]) if lam[i] >= 10 else "%.1f" % lam[i]) + (" token" if i == 31 else "")
    svg.text(px(i) + (12 if i == 31 else 0), BY1 + 56, lab, size=11, anchor="end" if i == 31 else "middle",
             color="muted")
svg.save("l10-yarn-freq.svg")
