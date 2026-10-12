"""Figure 5.4: RoPE wavelengths for d_h = 128 and for DeepSeek's d_h^R = 64 (research notes R1, R2).

lambda_i = 2*pi / theta_i = 2*pi * b^(2i/d), b = 10000. The x axis is the fraction 2i/d, on which both
head sizes fall on the same curve: pair i of d = 64 equals pair 2i of d = 128.
Counts of pairs with lambda_i > L_ctx (they do not complete one turn inside the pre-training window)
are computed here. Writes deepseek/figures/l05-wavelengths.svg
"""

import math

from l05_svg import SVG

B = 10000.0


def lam(i, d):
    return 2 * math.pi * B ** (2 * i / d)


W, H = 760, 430
X0, X1, Y0, Y1 = 92, 600, 50, 360
LY0, LY1 = 0, 5          # 10^0 .. 10^5 tokens

def px(f):               # f = 2i/d in [0, 1)
    return X0 + f * (X1 - X0)

def py(v):
    return Y1 - (math.log10(v) - LY0) / (LY1 - LY0) * (Y1 - Y0)

svg = SVG(W, H, "RoPE 各对维度的波长（128 维与 64 维，底数 10000，对数纵轴）")

# grid
for e in range(LY0, LY1 + 1):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
for f in (0, 0.25, 0.5, 0.75, 1.0):
    x = px(f)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 17, "%d" % round(f * 64), size=11, anchor="middle", color="muted")
    svg.text(x, Y0 - 8, "%d" % round(f * 32), size=11, anchor="middle", color="violet")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 36, "第 i 对（d_{h} = 128，共 64 对）", size=12, anchor="middle", color="muted")
svg.text((X0 + X1) / 2, Y0 - 26, "第 i 对（DeepSeek 的 64 维 RoPE，共 32 对）", size=12, anchor="middle", color="violet")
svg.text(24, (Y0 + Y1) / 2, "波长 λ_{i}（token，对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)

# context lines
for L, lab, col, dy in ((2048, "LLaMA 预训练长度 2048", "muted", 14), (4096, "DeepSeek 预训练长度 4096", "amber", -6)):
    y = py(L)
    svg.line(X0, y, X1, y, color=col, width=1.4, dash="6 4")
    svg.text(X1 + 6, y + 4 + (8 if L == 2048 else -4), lab, size=12, color=col)

# shade above 4096: pairs that never finish a turn inside 4K
svg.rect(X0, Y0, X1 - X0, py(4096) - Y0, fill="amber", fill_opacity=0.08)

# curve d = 128 (filled dots) and d = 64 (hollow rings)
pts = [(px(2 * i / 128), py(lam(i, 128))) for i in range(64)]
svg.polyline(pts, stroke="blue", width=1.3)
for x, y in pts:
    svg.circle(x, y, 2.6, fill="blue")
for i in range(32):
    svg.circle(px(2 * i / 64), py(lam(i, 64)), 5.2, stroke="violet", width=1.6)

# counts
def count(d, L):
    return sum(1 for i in range(d // 2) if lam(i, d) > L)

c128_4k, c128_2k, c64_4k = count(128, 4096), count(128, 2048), count(64, 4096)
assert (c128_4k, c128_2k, c64_4k) == (18, 23, 9)
first64 = min(i for i in range(32) if lam(i, 64) > 4096)
TX, TY = px(0.30), py(22000)
svg.text(TX, TY - 6, "λ > 4096：在 4K 内转不满一圈", size=12.5, color="amber", weight="600", anchor="middle")
svg.text(TX, TY + 12, "128 维：%d 对（2048 内：%d 对）" % (c128_4k, c128_2k), size=12, color="blue",
         anchor="middle")
svg.text(TX, TY + 29, "64 维：%d 对（i = %d … 31）" % (c64_4k, first64), size=12, color="violet",
         anchor="middle")

# end labels
svg.text(px(0) + 12, py(6.2832) + 18, "λ_{0} = 2π ≈ 6.28", size=12, color="ink")
e128 = lam(63, 128)
e64 = lam(31, 64)
svg.text(px(126 / 128) - 6, py(e128) - 12, "{:,.0f}".format(e128), size=12, anchor="end", color="blue")
svg.text(px(62 / 64) + 2, py(e64) + 22, "{:,.0f}".format(e64), size=12, anchor="middle", color="violet")

# legend
lx, ly = X1 + 8, Y1 - 110   # keep the last legend line clear of the x-tick row (tick "64" sits at X1)
svg.circle(lx + 6, ly, 2.6, fill="blue")
svg.text(lx + 16, ly + 4, "d_{h} = 128", size=12, color="blue")
svg.circle(lx + 6, ly + 20, 5.2, stroke="violet", width=1.6)
svg.text(lx + 16, ly + 24, "64 维（DeepSeek）", size=12, color="violet")
svg.text(lx, ly + 44, "同一条曲线：", size=11.5, color="muted")
svg.text(lx, ly + 60, "64 维的第 i 对", size=11.5, color="muted")
svg.text(lx, ly + 76, "= 128 维的第 2i 对", size=11.5, color="muted")

svg.save("l05-wavelengths.svg")
