"""Figure 18.7: what a V4-Pro query at position t = 1000 (0-based) sees.

Sliding window: tokens t-127 .. t. CSA: entries s < floor((t+1)/4) (reference-code rule, eq. 18.7),
each covering 4 tokens; top-k 1024 > 250 visible entries, so all are selected. HCA: blocks
s < floor((t+1)/128), each covering 128 tokens. Research notes sec. 5.2 F6.
Writes deepseek/figures/l18-visibility.svg
"""

from l18_svg import SVG

T = 1000
WIN0 = T - 128 + 1
CSA_N = (T + 1) // 4
HCA_N = (T + 1) // 128
assert (WIN0, CSA_N, HCA_N) == (873, 250, 7)
assert HCA_N * 128 >= WIN0             # the window covers HCA's incomplete tail

W, H = 760, 492
svg = SVG(W, H, "t = 1000 的查询在三种分支里看见的 token")

# ---------------------------------------------------------------- global view 0 .. 1100
GX0, GX1, TMAX = 110, 740, 1100


def gx(tok):
    return GX0 + tok / TMAX * (GX1 - GX0)


bands = [("原始 token", 40), ("滑动窗口", 76), ("CSA 条目", 112), ("HCA 块", 148)]
svg.text(20, 22, "全局：token 0 – 1100", size=13, weight="700")
for lab, y in bands:
    svg.text(GX0 - 10, y + 15, lab, size=12, anchor="end")
# raw tokens
svg.rect(gx(0), 40, gx(T + 1) - gx(0), 20, fill="muted", fill_opacity=0.18, stroke="muted", width=0.8)
svg.rect(gx(T + 1), 40, gx(TMAX) - gx(T + 1), 20, fill="red", fill_opacity=0.12, stroke="red", width=0.8, dash="3 3")
svg.text(gx(1050), 55, "未来", size=11, anchor="middle", color="red")
# window
svg.rect(gx(WIN0), 76, gx(T + 1) - gx(WIN0), 20, fill="green", fill_opacity=0.35, stroke="green", width=1)
# CSA: one band with ticks every 10 entries
svg.rect(gx(0), 112, gx(4 * CSA_N) - gx(0), 20, fill="blue", fill_opacity=0.25, stroke="blue", width=1)
for s in range(0, CSA_N + 1, 10):
    svg.line(gx(4 * s), 112, gx(4 * s), 132, color="blue", width=0.5, opacity=0.6)
# HCA blocks
for s in range(HCA_N + 1):
    done = s < HCA_N
    svg.rect(gx(128 * s) + 1, 148, gx(128) - gx(0) - 2, 20, fill="orange" if done else None,
             fill_opacity=0.35 if done else None, stroke="orange", width=1.2, dash=None if done else "4 3")
    svg.text(gx(128 * s + 64), 163, str(s), size=11, anchor="middle", color="orange" if done else "muted")
# query marker
svg.line(gx(T + 0.5), 30, gx(T + 0.5), 176, color="blue", width=1.6)
svg.text(gx(T + 0.5), 192, "t = 1000", size=12, anchor="middle", color="blue", weight="700")
for tok in (0, 256, 512, 768, 1024):
    svg.text(gx(tok), 192 if tok < 900 else 206, str(tok), size=11, anchor="middle", color="muted")

# zoom rectangle
ZL, ZR = 860, 1008
svg.rect(gx(ZL), 34, gx(ZR) - gx(ZL), 140, stroke="ink", width=1, dash="2 3")

# ---------------------------------------------------------------- zoom 860 .. 1008
ZX0, ZX1 = 110, 740


def zx(tok):
    return ZX0 + (tok - ZL) / (ZR - ZL) * (ZX1 - ZX0)


svg.text(20, 232, "放大：上图虚线框内的 token 860 – 1007", size=13, weight="700")
zb = [("原始 token", 250), ("滑动窗口", 286), ("CSA 条目", 322), ("HCA 块", 358)]
for lab, y in zb:
    svg.text(ZX0 - 10, y + 15, lab, size=12, anchor="end")
for tok in range(ZL, ZR):
    fut = tok > T
    svg.rect(zx(tok) + 0.3, 250, zx(tok + 1) - zx(tok) - 0.6, 20, fill="red" if fut else "muted",
             fill_opacity=0.14 if fut else 0.18, stroke=None)
# window
svg.rect(zx(WIN0), 286, zx(T + 1) - zx(WIN0), 20, fill="green", fill_opacity=0.35, stroke="green", width=1)
svg.text(zx(WIN0) + 4, 300, "873", size=11, color="green")
# CSA entries in zoom
for s in range(ZL // 4, (ZR + 3) // 4):
    a, b = max(4 * s, ZL), min(4 * s + 4, ZR)
    vis = s < CSA_N
    svg.rect(zx(a) + 0.8, 322, zx(b) - zx(a) - 1.6, 20, fill="blue" if vis else None,
             fill_opacity=0.28 if vis else None, stroke="blue", width=0.8, dash=None if vis else "3 2")
svg.text(zx(1000) - 2, 318, "条目 249", size=11, anchor="end", color="blue")
# HCA in zoom
svg.rect(zx(ZL), 358, zx(896) - zx(ZL) - 1, 20, fill="orange", fill_opacity=0.35, stroke="orange", width=1.2)
svg.text(zx(ZL) + 6, 373, "块 6（768–895）", size=11)
svg.rect(zx(896) + 1, 358, zx(ZR) - zx(896) - 1, 20, stroke="orange", width=1.2, dash="4 3")
svg.text(zx(896) + 8, 373, "块 7（896–1023）未凑满，不可见", size=11, color="muted")
# key boundary lines
svg.line(zx(896), 244, zx(896), 384, color="orange", width=1.4, dash="4 3")
svg.text(zx(896), 398, "896", size=11, anchor="middle", color="orange")
svg.line(zx(T + 1), 244, zx(T + 1), 384, color="blue", width=1.6)
svg.text(zx(T + 1), 398, "t", size=12, anchor="middle", color="blue", weight="700")
for tok in range(864, ZR, 16):
    svg.text(zx(tok + 0.5), 414, str(tok), size=11, anchor="middle", color="muted")

svg.text(20, 442, "虚线框 = 尚未凑满、不可见。Pro 的 top-k = 1024 > 250，所以此时 CSA 条目全部被选中。", size=12)
svg.text(20, 461, "HCA 尚未压缩的 token 数 = (t+1) mod 128 = 105（896–1000），全部落在窗口 873–1000 内；", size=12)
svg.text(20, 480, "一般地它最多 127 个，而窗口有 128 个，所以窗口总能盖住。", size=12)

svg.save("l18-visibility.svg")
