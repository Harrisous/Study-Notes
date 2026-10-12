"""Figure 19.5: layer-by-layer map of DeepSeek-V4.1-Flash, computed from config.json (via l19_num).

Cells: layers 0-39 of the backbone plus the 3 DSpark draft blocks (checkpoint mtp.0-2).
Colour = CSA2 mode (pure SWA / Full / Reindex / Reuse); the row under the cells is the compression
ratio m; the two bracket rows show which layer each layer borrows its main KV (+ indexer K) and its
top-k indices from. Writes deepseek/figures/l19-layer-map.svg
"""

from l19_svg import SVG
from l19_num import MODES, COMPRESS_V41, KV_SRC, IDX_SRC, ENGRAM, DSPARK_TARGET, L41

W, H = 760, 330
svg = SVG(W, H, "V4.1-Flash 逐层配置：40 层只有 4 层写全局 KV，8 层跑索引器")

X0, CW = 84, 15.6
NC = L41 + 3
CY, CH = 70, 30


def cx(i):
    return X0 + i * CW


FILL = {"swa": ("muted", 0.30), "full": ("orange", 0.55), "reindex": ("violet", 0.45),
        "reuse": ("blue", 0.10), "dspark": ("muted", 0.15)}

# section brackets
def bracket(i0, i1, y, label, color="ink", size=12, below=False):
    x0, x1 = cx(i0) + 1, cx(i1 + 1) - 1
    d = 5 if below else -5
    svg.path("M%.1f %.1f L%.1f %.1f L%.1f %.1f L%.1f %.1f" % (x0, y + d, x0, y, x1, y, x1, y + d),
             stroke=color, width=1.2)
    ty = y + 16 if below else y - 6
    svg.text((x0 + x1) / 2, ty, label, size=size, anchor="middle", color=color)


bracket(0, 19, 40, "编码器（层 0–19）")
bracket(20, 39, 40, "解码器（层 20–39）")
bracket(40, 42, 40, "DSpark", color="muted")

for i in range(NC):
    mode = MODES[i] if i < L41 else "dspark"
    col, op = FILL[mode]
    svg.rect(cx(i) + 0.8, CY, CW - 1.6, CH, fill=col, fill_opacity=op, stroke=col, width=0.8, rx=2)
for l in ENGRAM:
    x = cx(l) + CW / 2
    svg.polygon([(x - 5, CY - 9), (x, CY - 14), (x + 5, CY - 9), (x, CY - 4)], fill="green", fill_opacity=0.85)
for l in DSPARK_TARGET:
    svg.circle(cx(l) + CW / 2, CY + CH + 6, 2.4, fill="muted")

svg.text(X0 - 8, CY + CH / 2 + 4, "模式", size=12, anchor="end", color="muted")
for l in (0, 2, 8, 14, 20, 24, 28, 32, 36, 39):
    svg.text(cx(l) + CW / 2, CY + CH + 22, str(l), size=11, anchor="middle", color="muted")
svg.text(X0 - 8, CY + CH + 22, "层号", size=12, anchor="end", color="muted")

MY = CY + CH + 42
svg.text(X0 - 8, MY, "压缩比 m", size=12, anchor="end", color="muted")
for i in range(L41):
    m = COMPRESS_V41[i]
    svg.text(cx(i) + CW / 2, MY, "–" if m == 0 else str(m), size=11, anchor="middle",
             color="ink" if m else "muted")

# which layer supplies the main KV / indexer K
KY = MY + 22
svg.text(X0 - 8, KY + 12, "主 KV 来自", size=12, anchor="end", color="amber")
for j, src in enumerate(KV_SRC):
    end = (KV_SRC[j + 1] - 1) if j + 1 < len(KV_SRC) else L41 - 1
    bracket(src, end, KY, "层 %d（m = %d）" % (src, COMPRESS_V41[src]), color="amber", size=11, below=True)

# which layer supplies the top-k indices
TY = KY + 44
svg.text(X0 - 8, TY + 12, "top-k 来自", size=12, anchor="end", color="violet")
for j, src in enumerate(IDX_SRC):
    end = (IDX_SRC[j + 1] - 1) if j + 1 < len(IDX_SRC) else L41 - 1
    bracket(src, end, TY, "层 %d" % src, color="violet", size=11, below=True)

# legend
LY = TY + 58
items = [("full", "Full：写全局 KV、跑索引器（4 层）", 20, LY), ("reindex", "Reindex：借 KV、自己打分（4 层）", 270, LY),
         ("reuse", "Reuse：KV 与 top-k 都借（30 层）", 510, LY), ("swa", "纯 SWA（2 层）", 20, LY + 22),
         ("dspark", "DSpark 草稿块（纯 SWA）", 150, LY + 22)]
for mode, lab, x, y in items:
    col, op = FILL[mode]
    svg.rect(x, y - 10, 14, 12, fill=col, fill_opacity=op, stroke=col, width=0.8)
    svg.text(x + 20, y, lab, size=12)
svg.polygon([(340, LY + 16), (345, LY + 11), (350, LY + 16), (345, LY + 21)], fill="green", fill_opacity=0.85)
svg.text(357, LY + 22, "Engram（层 1、14）", size=12)
svg.circle(496, LY + 17, 2.4, fill="muted")
svg.text(504, LY + 22, "DSpark 取特征的层（37–39）", size=12)
svg.text(20, LY + 46, "每一层都另有自己的 SWA（窗口 128），不跨层共享；DSpark 块不属于 40 层主干。", size=12, color="muted")

assert sum(1 for m in MODES if m == "reuse") == 30
svg.save("l19-layer-map.svg")
