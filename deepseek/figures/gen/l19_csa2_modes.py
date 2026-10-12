"""Figure 19.4: the three operating modes of CSA2, redrawn after V4.1 paper Fig. 4 (course colours).

Blue   = computed in the current layer (main Q and SWA KV are always computed here)
Amber  = main KV + indexer K reused from the most recent Full-mode layer (a cache read)
Violet = Top-K indices reused from the most recent index-producing (Full or Reindex) layer
Schematic, hand-placed; no data. Writes deepseek/figures/l19-csa2-modes.svg
"""

from l19_svg import SVG

W, H = 760, 474
svg = SVG(W, H, "CSA2 的三种模式：Full、Reindex、Reuse 各自在本层算什么、借什么")

PW, GAP, M = 240, 12, 8
PT, PB = 34, 336          # panel top / bottom
IY = 288                  # input row top
BH = 26
EXT_Y = 352               # external (reused) inputs


def op(x, y, w, label, color="muted", op_=0.08, size=12):
    svg.rect(x, y, w, BH, fill=color, fill_opacity=op_, stroke=color, width=1.1, rx=5)
    svg.text(x + w / 2, y + BH / 2 + 4.3, label, size=size, anchor="middle")


def up(x1, y1, x2, y2, color="ink"):
    svg.arrow(x1, y1, x2, y2, color=color, width=1.3)


def panel(i, title, mode):
    x0 = M + i * (PW + GAP)
    svg.rect(x0, PT, PW, PB - PT, fill="grid", fill_opacity=0.35, stroke="rule", width=1.2, rx=8)
    svg.text(x0 + PW - 10, PT + 20, title, size=13, anchor="end", weight="700")
    X = lambda v: x0 + v

    # top part, identical in all modes
    op(X(8), PT + 26, 92, "注意力输出", color="green", op_=0.12)
    op(X(8), PT + 72, 136, "核心注意力")
    op(X(56), PT + 120, 84, "拼接")
    op(X(150), PT + 120, 84, "选中的主 KV", color="amber", op_=0.12, size=11)
    op(X(84), PT + 168, 60, "选择")
    up(X(54), PT + 72, X(54), PT + 52)
    up(X(98), PT + 120, X(98), PT + 98)
    up(X(150), PT + 133, X(142), PT + 133)
    svg.path("M%.1f %.1f L%.1f %.1f L%.1f %.1f" % (X(130), PT + 168, X(130), PT + 157, X(190), PT + 157),
             stroke="ink", width=1.3)
    up(X(190), PT + 157, X(190), PT + 148)

    # always-local inputs
    op(X(6), IY, 40, "主 Q", color="blue", op_=0.16)
    op(X(50), IY, 50, "SWA KV", color="blue", op_=0.16, size=11)
    up(X(26), IY, X(26), PT + 100)
    up(X(75), IY, X(75), PT + 148)

    if mode == "full":
        op(X(104), IY, 42, "主 KV", color="blue", op_=0.16)
        op(X(150), IY, 42, "索引键", color="blue", op_=0.16, size=11)
        op(X(196), IY, 40, "索引 Q", color="blue", op_=0.16, size=11)
        svg.text(X(171), IY + 38, "索引键由主 KV 投影", size=11, anchor="middle", color="muted")
    elif mode == "reindex":
        op(X(196), IY, 40, "索引 Q", color="blue", op_=0.16, size=11)
        op(X(104), EXT_Y, 42, "主 KV", color="amber", op_=0.28)
        op(X(150), EXT_Y, 42, "索引键", color="amber", op_=0.28, size=11)
    else:
        op(X(104), EXT_Y, 42, "主 KV", color="amber", op_=0.28)
        op(X(170), EXT_Y, 60, "top-k 索引", color="violet", op_=0.18, size=11)

    kv_y = IY if mode == "full" else EXT_Y
    up(X(125), kv_y, X(125), PT + 194)                      # main KV -> selection
    if mode in ("full", "reindex"):
        op(X(156), PT + 214, 80, "索引器", color="blue", op_=0.10)
        op(X(176), PT + 168, 60, "top-k", color="blue", op_=0.16)
        up(X(171), kv_y, X(171), PT + 240)                  # indexer K -> indexer
        up(X(217), IY, X(217), PT + 240)                    # indexer Q -> indexer
        up(X(206), PT + 214, X(206), PT + 194)              # indexer -> top-k
        up(X(176), PT + 181, X(146), PT + 181)              # top-k -> selection
    else:
        svg.path("M%.1f %.1f L%.1f %.1f L%.1f %.1f" % (X(200), EXT_Y, X(200), PT + 181, X(150), PT + 181),
                 stroke="violet", width=1.3)
        svg.arrow(X(152), PT + 181, X(146), PT + 181, color="violet", width=1.3)
        svg.text(X(165), PT + 236, "不跑索引器", size=11, anchor="middle", color="muted")


panel(0, "Full 模式", "full")
panel(1, "Reindex 模式", "reindex")
panel(2, "Reuse 模式", "reuse")

# annotations for the borrowed inputs
svg.text(M + PW + GAP + PW / 2, EXT_Y + 44, "主 KV、索引键借自最近的 Full 层", size=11, anchor="middle", color="amber")
svg.text(M + 2 * (PW + GAP) + PW / 2, EXT_Y + 44, "主 KV 借自最近的 Full 层", size=11, anchor="middle",
         color="amber")
svg.text(M + 2 * (PW + GAP) + PW / 2, EXT_Y + 59, "top-k 借自最近的 Full / Reindex 层", size=11, anchor="middle",
         color="violet")

LY = 440
svg.rect(M, LY - 10, 14, 12, fill="blue", fill_opacity=0.16, stroke="blue", width=1)
svg.text(M + 20, LY, "本层计算", size=12)
svg.rect(M + 98, LY - 10, 14, 12, fill="amber", fill_opacity=0.28, stroke="amber", width=1)
svg.text(M + 118, LY, "复用：主 KV 与索引键（读缓存）", size=12)
svg.rect(M + 330, LY - 10, 14, 12, fill="violet", fill_opacity=0.18, stroke="violet", width=1)
svg.text(M + 350, LY, "复用：top-k 索引（不再打分）", size=12)
svg.text(M, LY + 24, "三种模式都在本层算自己的主 Q 与 SWA KV；区别只在主 KV、索引键、top-k 索引三样东西是自己算还是借来的。",
         size=12, color="muted")

svg.save("l19-csa2-modes.svg")
