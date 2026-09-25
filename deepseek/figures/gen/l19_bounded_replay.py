"""Figure 19.12: what exact SWA reconstruction needs vs what SWA Bounded Replay does (toy scale).

Toy: n_win = 8 positions, 4 layers with SWA, cache hit at position h. To rebuild the SWA KV of the last
n_win cached positions exactly, layer l (counted from the top) needs l * n_win positions of history, so
the bottom layer needs L * n_win (the red staircase). Bounded Replay recomputes only [h - n_win, h) in
every layer and truncates each window at the replay start s = h - n_win: a query at i sees
[max(s, i - n_win + 1), i] (paper §3.2.2). The staircase and the truncated window are computed below.
Writes deepseek/figures/l19-bounded-replay.svg
"""

from l19_svg import SVG

W, H = 760, 332
svg = SVG(W, H, "SWA Bounded Replay：精确重建要的历史 vs 只回放一个窗口")

NW, NL = 8, 4
P0, P1 = -NL * NW - 2, 10            # positions shown, relative to the hit position h (= 0)
X0, CWD = 92, 14.6
Y0, RH, RG = 64, 24, 18


def xs(p):
    return X0 + (p - P0) * CWD


def row_y(level):          # level 0 = bottom layer
    return Y0 + (NL - 1 - level) * (RH + RG)


s = -NW                     # replay start
# global KV bar (cached, exact)
svg.rect(xs(P0), Y0 - 34, xs(0) - xs(P0), 12, fill="amber", fill_opacity=0.45, stroke="amber", width=0.8)
svg.text(xs(P0), Y0 - 40, "全局 KV：前缀的全部位置都从 SSD 读回，精确、不重算", size=12, color="amber")
svg.rect(xs(0), Y0 - 34, xs(P1) - xs(0), 12, fill="blue", fill_opacity=0.25, stroke="blue", width=0.8)

for level in range(NL):
    y = row_y(level)
    svg.text(X0 - 10, y + RH / 2 + 4, "层 %d" % (level + 1), size=12, anchor="end", color="muted")
    # exact reconstruction staircase: from the top layer (level NL-1) needing NW, down to NL*NW
    need = (NL - level) * NW
    svg.rect(xs(-need), y, xs(s) - xs(-need), RH, fill="red", fill_opacity=0.10, stroke="red", width=0.8,
             dash="3 2")
    # replay window
    svg.rect(xs(s), y, xs(0) - xs(s), RH, fill="blue", fill_opacity=0.18, stroke="blue", width=1.2)
    # new suffix
    svg.rect(xs(0), y, xs(P1) - xs(0), RH, fill="green", fill_opacity=0.14, stroke="green", width=1)
    for p in range(P0, P1 + 1):
        svg.line(xs(p), y, xs(p), y + RH, color="surface", width=0.6)

# boundary lines
for p, lab, col in ((s, "s = h − n_{win}：回放起点", "blue"), (0, "h：缓存命中处", "ink")):
    svg.line(xs(p), Y0 - 18, xs(p), row_y(0) + RH + 8, color=col, width=1.4)
yb = row_y(0) + RH + 22
svg.text(xs(s), yb, "s", size=12, anchor="middle", color="blue")
svg.text(xs(0), yb, "h", size=12, anchor="middle")

# a truncated window in layer 2 (level 1)
lv = 1
y = row_y(lv)
i = s + 2
lo_full, lo_cut = i - NW + 1, max(s, i - NW + 1)
svg.circle(xs(i) + CWD / 2, y + RH / 2, 4, fill="blue")
yy = y - RG / 2          # the window line runs in the gap between rows
svg.line(xs(lo_cut), yy, xs(i + 1), yy, color="blue", width=2.2)
svg.line(xs(lo_full), yy, xs(lo_cut), yy, color="red", width=2.2, dash="3 2")
svg.text(xs(lo_full) - 4, yy + 4, "窗口被截掉的部分", size=11, anchor="end", color="red")

# annotations
svg.text(xs(-NL * NW), row_y(0) + RH + 40, "精确重建：每往下一层多要 n_{win} 个位置，最底层要 L × n_{win} 个（红色虚线）",
         size=12, color="red")
svg.text(xs(-NL * NW), row_y(0) + RH + 60, "Bounded Replay：每层只重算 [s, h) 这 n_{win} 个位置（蓝），窗口在 s 处截断",
         size=12, color="blue")
svg.text(xs(-NL * NW), row_y(0) + RH + 80, "新输入（绿）从 h 开始正常 prefill：同时写全局 KV 与 SWA KV",
         size=12, color="green")
svg.text(xs(-NL * NW), row_y(0) + RH + 104,
         "示意：n_{win} = 8、4 层。V4.1：n_{win} = 128；按论文的口径，精确重建要 L × n_{win} 个 token，有界回放只要 128 个。",
         size=12, color="muted")

svg.save("l19-bounded-replay.svg")
