"""Figure M8.10: how much a single outlier costs, by format and scaling granularity.

For each configuration in m08_block_scaling.CONFIGS: 1024 draws from N(0,1) (default_rng(0))
with element 500 replaced by the outlier (50, or 10^5), quantised and dequantised; the two
bars are the fraction of the other 1023 values flushed to zero and their median relative
error.  All numbers are computed here.
Writes math/figures/m08-block-error.svg
"""

from m08_block_scaling import CONFIGS, experiment
from m08_svg import SVG

rows = []
for lab, fmt, o, b, r in CONFIGS:
    z, m = experiment(fmt, o, b, r)
    rows.append((lab, fmt, o, z, m))
    print(lab, round(100 * z, 1), round(100 * m, 1))

W, H = 760, 346
LX = 250                     # label column right edge
PANELS = [(270, 490, "被冲成 0 的比例", 3), (530, 750, "中位相对误差", 4)]
Y0, RH = 64, 38
svg = SVG(W, H, "一个离群值的代价：格式 × 缩放粒度")

for x0, x1, title, k in PANELS:
    svg.text(x0, 34, title, size=13, weight="600")
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + t * (x1 - x0 - 50)
        svg.line(x, Y0 - 12, x, Y0 + RH * len(rows) - 8, color="grid", width=1)
        svg.text(x, Y0 + RH * len(rows) + 8, "%d%%" % round(t * 100), size=11, anchor="middle",
                 color="muted")

for i, (lab, fmt, o, z, m) in enumerate(rows):
    y = Y0 + i * RH
    col = "orange" if fmt == "E2M1" else "blue"
    svg.text(LX, y + 9, lab, size=12, anchor="end")
    for (x0, x1, title, k), v in zip(PANELS, (z, m)):
        w = v * (x1 - x0 - 50)
        svg.rect(x0, y - 4, max(w, 1.5), 18, fill=col, fill_opacity=0.75)
        svg.text(x0 + w + 6, y + 9, "%.1f%%" % (100 * v), size=12, color=col)
    if i == 2:
        svg.line(16, y + RH / 2 + 2, W - 10, y + RH / 2 + 2, color="rule", width=1)

svg.text(16, H - 12, "设置：1024 个 N(0,1) 随机数，其中一个换成离群值（前四行 50，后两行 10^{5}）；"
         "统计其余 1023 个数。缩放因子 s = 块内最大绝对值 / 格式最大值",
         size=11, color="muted")
svg.save("m08-block-error.svg")
