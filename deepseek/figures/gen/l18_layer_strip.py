"""Figure 18.6: per-layer attention type of V4-Pro and V4-Flash, drawn from the compress_ratios
lists in config.json (0 = pure sliding window, 4 = CSA, 128 = HCA; the last element is the MTP
layer, which is pure sliding window). Research notes C33.
Writes deepseek/figures/l18-layer-strip.svg
"""

from l18_svg import SVG
from l18_data import FLASH_RATIOS, PRO_RATIOS

ROWS = [("V4-Pro", PRO_RATIOS + [0]), ("V4-Flash", FLASH_RATIOS + [0])]
assert len(ROWS[0][1]) == 62 and len(ROWS[1][1]) == 44
COLOR = {4: "blue", 128: "orange", 0: "muted"}

W, H = 760, 300
svg = SVG(W, H, "逐层的注意力类型（config.json 的 compress_ratios）")
X0, CW, CH = 96, 10, 30

for n, (name, ratios) in enumerate(ROWS):
    y = 40 + n * 110
    svg.text(X0 - 12, y + 20, name, size=13, anchor="end", weight="700")
    for l, r in enumerate(ratios):
        mtp = l == len(ratios) - 1
        x = X0 + l * CW + (10 if mtp else 0)
        svg.rect(x + 0.5, y, CW - 1, CH, fill=COLOR[r], fill_opacity=0.75 if r else 0.35, stroke=None, rx=1)
        if mtp:
            svg.rect(x - 1, y - 2, CW + 1, CH + 4, stroke="ink", width=1, rx=2)
            svg.text(x + CW / 2, y - 8, "MTP", size=11, anchor="middle", color="muted")
    for l in range(0, len(ratios) - 1, 10):
        svg.line(X0 + l * CW + CW / 2, y + CH, X0 + l * CW + CW / 2, y + CH + 5, color="muted", width=1)
        svg.text(X0 + l * CW + CW / 2, y + CH + 18, str(l), size=11, anchor="middle", color="muted")
    main = ratios[:-1]
    counts = "%d 层 CSA + %d 层 HCA" % (main.count(4), main.count(128))
    if main.count(0):
        counts += " + %d 层纯滑动窗口" % main.count(0)
    svg.text(X0, y + CH + 40, "主干 %d 层：%s；另有 1 个 MTP 层（纯滑动窗口）" % (len(main), counts), size=12)

# legend
LY = 270
for n, (r, lab) in enumerate([(4, "CSA（m = 4，挑 top-k）"), (128, "HCA（m′ = 128，全读）"), (0, "纯滑动窗口（无压缩分支）")]):
    x = 96 + n * 220
    svg.rect(x, LY - 12, 16, 16, fill=COLOR[r], fill_opacity=0.75 if r else 0.35, stroke=None, rx=2)
    svg.text(x + 24, LY + 1, lab, size=12)
svg.text(X0, 20, "每格一层，自左向右为第 0 层到最后一层；层号为 0 起算", size=11, color="muted")

svg.save("l18-layer-strip.svg")
