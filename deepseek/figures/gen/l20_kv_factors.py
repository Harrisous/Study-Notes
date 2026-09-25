"""Figure 20.2 (summary of thread A): the 437x drop in KV bytes per token from DeepSeek LLM 67B to
DeepSeek-V4.1-Flash, split into five multiplicative factors per generation step.

Geometry is computed from l20_num.STEPS: the x axis is log10 of the cumulative reduction, every
segment has length log10(factor); a factor below 1 (the cache grew along that dimension) is drawn
leftwards with a dashed red outline. Factor definitions are (20.2) in the lesson:
  layers holding global KV (depth), cross-layer sharing, entries per token per copy (sequence),
  numbers per entry, bytes per number (scales amortised).
Writes deepseek/figures/l20-kv-factors.svg
"""

import math

from l20_svg import SVG, fmt
from l20_num import STEPS, TOTAL, SHARE, DIMS, BYTES

COL = {"层数": "muted", "跨层共享": "blue", "序列维": "violet", "条目维": "green", "精度": "orange"}
SHORT = {"DeepSeek LLM 67B": "67B", "DeepSeek-V2": "V2", "DeepSeek-V3": "V3", "DeepSeek-V3.2": "V3.2",
         "DeepSeek-V4-Flash": "V4-Flash", "DeepSeek-V4.1-Flash": "V4.1"}

W, H = 760, 520
svg = SVG(W, H, "主线 A 的总账：每 token KV 从 389,120 字节到 890 字节，437 倍拆成五个因子")
X0, X1 = 150, 740
LMAX = 3.4


def xs(v):
    return X0 + v / LMAX * (X1 - X0)


TOP = 70
for e in range(0, 4):
    x = xs(e)
    svg.line(x, TOP - 8, x, TOP + 6 * 64 - 4, color="grid", width=1)
    svg.text(x, TOP - 14, "×" + "{:,}".format(10 ** e), size=11, anchor="middle", color="muted")
svg.text(X0, 20, "累计缩小倍数（对数刻度，从 DeepSeek LLM 67B 算起）", size=12, color="muted")

# legend
lx = X0
for d in DIMS:
    svg.rect(lx, 30, 12, 12, fill=COL[d], fill_opacity=0.55, stroke=COL[d], width=0.8)
    svg.text(lx + 17, 41, d, size=12)
    lx += 86
svg.rect(lx + 6, 30, 12, 12, fill="red", fill_opacity=0.08, stroke="red", width=1, dash="3 2")
svg.text(lx + 23, 41, "反向（缓存变大）", size=12, color="red")

def lab(v):
    return "×%s" % (("%.2f" % v) if v < 10 else ("%.1f" % v))


cum = 0.0
RH = 64
for i, (a, b, f, r) in enumerate(STEPS):
    y = TOP + i * RH
    svg.text(X0 - 10, y + 16, "%s → %s" % (SHORT[a], SHORT[b]), size=12, anchor="end", weight="700")
    svg.text(X0 - 10, y + 33, lab(r), size=12, anchor="end", color="muted")
    # factors that shrink the cache go right first, then the ones that grow it come back left
    x = cum
    small = []
    spans = []          # x extents of the in-bar labels, so the landing line can step around them
    for d in DIMS:
        v = math.log10(f[d])
        if v <= 1e-9:
            continue
        svg.rect(xs(x), y + 6, xs(x + v) - xs(x), 20, fill=COL[d], fill_opacity=0.55, stroke=COL[d], width=0.8)
        if v > 0.07:
            mid = (xs(x) + xs(x + v)) / 2
            svg.text(mid, y + 20, lab(f[d]), size=11, anchor="middle")
            half = 3.3 * len(lab(f[d])) + 2
            spans.append((mid - half, mid + half))
        else:
            small.append("%s %s" % (d, lab(f[d])))
        x += v
    grows = [d for d in DIMS if math.log10(f[d]) < -1e-9]
    if small:
        if grows:       # the lower lane is taken: write the small factors left of the bar
            svg.text(xs(cum) - 6, y + 20, " · ".join(small), size=11, anchor="end", color="muted")
        else:
            svg.text(xs(x) - 4, y + 40, " · ".join(small), size=11, anchor="end", color="muted")
    right = x
    red = []
    for d in grows:
        v = math.log10(f[d])
        svg.rect(xs(x + v), y + 30, xs(x) - xs(x + v), 12, fill="red", fill_opacity=0.08, stroke="red", width=1,
                 dash="3 2")
        red.append("%s（%s）" % (lab(f[d]), d))
        x += v
    if red:
        if right - x > 0.07:
            svg.text((xs(x) + xs(right)) / 2, y + 56, " · ".join(red), size=11, anchor="middle", color="red")
        else:
            svg.text(xs(right) + 6, y + 40, " · ".join(red), size=11, color="red")
    # where the model lands
    end = cum + math.log10(r)
    if any(lo <= xs(end) <= hi for lo, hi in spans):     # do not strike through a factor label
        svg.line(xs(end), y + 2, xs(end), y + 9, color="ink", width=1.4)
        svg.line(xs(end), y + 23, xs(end), y + 46, color="ink", width=1.4)
    else:
        svg.line(xs(end), y + 2, xs(end), y + 46, color="ink", width=1.4)
    svg.text(xs(end) + 4, y + 2, "{:,.0f} B".format(BYTES[b]) if BYTES[b] >= 1000 else "%d B" % BYTES[b], size=11,
             color="ink")
    cum = end

# total row
y = TOP + 5 * RH + 6
svg.line(X0 - 140, y - 6, X1, y - 6, color="rule", width=1)
svg.text(X0 - 10, y + 16, "合计 67B → V4.1", size=12, anchor="end", weight="700")
svg.text(X0 - 10, y + 33, "×437", size=12, anchor="end", color="muted")
x = 0.0
for d in DIMS:
    v = math.log10(TOTAL[d])
    svg.rect(xs(x), y + 6, xs(x + v) - xs(x), 20, fill=COL[d], fill_opacity=0.55, stroke=COL[d], width=0.8)
    svg.text((xs(x) + xs(x + v)) / 2, y + 20, "×%s" % ("%.2f" % TOTAL[d] if TOTAL[d] < 9.9 else "%.1f" % TOTAL[d]),
             size=11, anchor="middle")
    svg.text((xs(x) + xs(x + v)) / 2, y + 42, "%d%%" % round(100 * SHARE[d]), size=11, anchor="middle",
             color="muted")
    x += v

svg.text(10, H - 26, "竖线是每一代落到的位置（每 token 全局 KV 字节）。百分比是各因子在对数上的份额（与乘的先后无关）。",
         size=12, color="muted")
svg.text(10, H - 8, "数据：第 9、11、14、18、19 课的配置与字节格式；由 l20_num.py 计算。", size=12, color="muted")

svg.save("l20-kv-factors.svg")
