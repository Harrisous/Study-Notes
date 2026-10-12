"""Figure 18.12: KV bytes per token from BF16 GQA8 to V4 (log axis).

GQA8, BF16, head dim 128, 61 layers: 2*8*128*2*61 = 249,856 B; V3 MLA in BF16: 576*2*61 = 70,272 B;
V3.2 (FP8 MLA entry 656 B + FP8 indexer key 132 B, 61 layers): 48,068 B; V4-Pro / V4-Flash by
eq. (18.25): 5,031.44 / 3,514.25 B (growing part only). Research notes sec. 4.2.
Writes deepseek/figures/l18-kv-ladder.svg
"""

import math

from l18_svg import SVG
from l18_data import V32_ENTRY, V32_IDX, kv_v32, kv_v4

pro, fla = kv_v4("pro"), kv_v4("flash")
ROWS = [
    ("GQA8，BF16（61 层）", 2 * 8 * 128 * 2 * 61, "muted", "K、V 各 8 组 × 128 维"),
    ("V3 的 MLA，BF16", 576 * 2 * 61, "muted", "576 维 × 2 字节 × 61 层"),
    ("V3.2：MLA（FP8）+ indexer", kv_v32(), "amber", "条目 %s + indexer 键 %s" % (format(61 * V32_ENTRY, ","), format(61 * V32_IDX, ","))),
    ("V4-Pro", pro["total"], "orange", "CSA 条目 4,380 + indexer 510 + HCA 141"),
    ("V4-Flash", fla["total"], "orange", "CSA 条目 3,066 + indexer 357 + HCA 91"),
]
assert ROWS[0][1] == 249856 and ROWS[1][1] == 70272 and ROWS[2][1] == 48068
assert (pro["csa_main"], pro["csa_idx"]) == (4380, 510) and (fla["csa_main"], fla["csa_idx"]) == (3066, 357)

W, H = 740, 330
svg = SVG(W, H, "每 token 的 KV 字节（对数轴）")
X0, X1 = 200, 560
L0, L1 = 3.0, 5.5                # 1e3 .. ~3.2e5 bytes


def px(v):
    return X0 + (math.log10(v) - L0) / (L1 - L0) * (X1 - X0)


for e in (3, 4, 5):
    for mult in (1, 2, 5):
        v = mult * 10 ** e
        if L0 <= math.log10(v) <= L1:
            svg.line(px(v), 30, px(v), 262, color="grid" if mult > 1 else "rule", width=1)
    svg.text(px(10 ** e), 280, "10^{%d} B" % e, size=11, anchor="middle", color="muted")
svg.text(px(2e3), 280, "2K", size=11, anchor="middle", color="muted")
svg.text(px(2e4), 280, "20K", size=11, anchor="middle", color="muted")
svg.text(px(2e5), 280, "200K", size=11, anchor="middle", color="muted")
for n, (lab, v, col, sub) in enumerate(ROWS):
    y = 36 + n * 46
    svg.text(X0 - 10, y + 17, lab, size=12, anchor="end")
    svg.rect(X0, y + 4, px(v) - X0, 22, fill=col, fill_opacity=0.55 if col != "muted" else 0.35, stroke=col, width=0.8)
    svg.text(px(v) + 6, y + 14, "{:,.0f} B".format(v) if v >= 10000 else "{:,.2f} B".format(v).replace(".00", ""),
             size=12, weight="700")
    svg.text(px(v) + 6, y + 29, sub, size=11, color="muted")
# ratios
r1, r2 = kv_v32() / pro["total"], kv_v32() / fla["total"]
assert round(r1, 2) == 9.55 and round(r2, 2) == 13.68
svg.text(20, 306, "相对 V3.2：Pro 小 %.2f 倍（%.1f%%），Flash 小 %.2f 倍（%.1f%%）；相对 GQA8：约 2%%" % (
    r1, 100 / r1, r2, 100 / r2), size=12, color="orange")
svg.text(20, 326, "只计随长度增长的部分（不含滑动窗口与未凑满的尾部状态）；V4 的 indexer 键按参考代码的 68 B 计。", size=11,
         color="muted")

svg.save("l18-kv-ladder.svg")
