"""Figure 18.13: KV cache of one request vs context length (redraw of the lower-right panel of
Fig. 1 of the V4 report, computed from eq. 18.1 and eq. 18.25). x in K = 1024 tokens, y in GB = 1e9
bytes (units inferred by the research notes from the paper's figure). Growing part only.
Writes deepseek/figures/l18-kv-vs-length.svg
"""

from l18_svg import SVG
from l18_data import K, kv_v32, kv_v4

LINES = [("V3.2", kv_v32(), "amber"), ("V4-Pro", kv_v4("pro")["total"], "orange"),
         ("V4-Flash", kv_v4("flash")["total"], "red")]
GQA = 2 * 8 * 128 * 2 * 61
end = {n: b * 1024 * K / 1e9 for n, b, _ in LINES}
assert round(end["V3.2"], 2) == 50.40 and round(end["V4-Pro"], 3) == 5.276 and round(end["V4-Flash"], 3) == 3.685
assert round(GQA * 1024 * K / 1e9, 1) == 262.0

W, H = 720, 400
svg = SVG(W, H, "一个请求的 KV cache 随长度增长")
X0, X1, Y0, Y1, YMAX = 70, 560, 30, 340, 60


def px(k):
    return X0 + k / 1024 * (X1 - X0)


def py(gb):
    return Y1 - gb / YMAX * (Y1 - Y0)


for g in range(0, 61, 10):
    svg.line(X0, py(g), X1, py(g), color="grid", width=1)
    svg.text(X0 - 8, py(g) + 4, str(g), size=11, anchor="end", color="muted")
for k in (0, 128, 256, 512, 768, 1024):
    svg.line(px(k), Y0, px(k), Y1, color="grid", width=1)
    svg.text(px(k), Y1 + 18, "%dK" % k, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text(X0 - 50, Y0 - 12, "KV cache（GB）", size=12, color="muted")
svg.text((X0 + X1) / 2, Y1 + 38, "上下文长度（K = 1024 个 token）", size=12, anchor="middle", color="muted")

# GQA8 reference: clip at the top edge of the plot
kx = YMAX / (GQA * K / 1e9)          # K-tokens at which GQA8 reaches YMAX
svg.line(px(0), py(0), px(kx), py(YMAX), color="muted", width=1.6, dash="6 4")
svg.text(px(kx) + 6, py(YMAX) + 12, "BF16 GQA8（61 层）：1M 时 262 GB", size=11, color="muted")
DY = {"V3.2": 4, "V4-Pro": -4, "V4-Flash": 14}
for name, b, col in LINES:
    svg.line(px(0), py(0), px(1024), py(end[name]), color=col, width=2.6)
    svg.circle(px(1024), py(end[name]), 3.5, fill=col)
    svg.text(px(1024) + 8, py(end[name]) + DY[name], "%s：%s GB" % (name, "%.2f" % end[name]), size=12, color=col,
             weight="700")
# 128K marker
svg.line(px(128), Y0, px(128), Y1, color="red", width=1, dash="3 3")
svg.text(px(128) - 5, Y1 - 44, "V3.2 上限 128K", size=11, color="red", rotate=-90)
svg.text(px(640), py(24), "比值处处相同：", size=12)
svg.text(px(640), py(24) + 18, "Pro 为 V3.2 的 10.5%（9.55 倍），", size=12)
svg.text(px(640), py(24) + 36, "Flash 为 7.3%（13.68 倍）", size=12)
svg.text(20, 392, "只计随长度增长的部分；另有每请求固定的滑动窗口状态（Pro 4.56 MB，Flash 3.21 MB）未画出。", size=11,
         color="muted")

svg.save("l18-kv-vs-length.svg")
