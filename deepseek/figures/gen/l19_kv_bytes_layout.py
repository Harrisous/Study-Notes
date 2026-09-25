"""Figure 19.10: byte layout of one cached entry, V3.2 -> V4 -> V4.1 (FlashMLA README formats; notes 3-D).

Each bar is one entry drawn to scale (0.8 px per byte). Segment sizes come from l19_num where they
exist and are asserted to add up to the row totals quoted in the lesson.
Writes deepseek/figures/l19-kv-bytes-layout.svg
"""

from l19_svg import SVG
from l19_num import MAIN_V41, IDX, MAIN_V4, SWA_V41, V32_MLA, V32_IDX

W, H = 760, 448
svg = SVG(W, H, "一个缓存条目的字节布局：V3.2、V4、V4.1")

BX, PXB = 222, 0.8
BH = 22
ROWS = [
    ("V3.2 主条目（MLA）", "576 维", [(512, "fp8", "512 × FP8"), (16, "scale", "4 × FP32 缩放"),
                                       (128, "bf16", "64 × BF16 RoPE")], V32_MLA),
    ("V3.2 索引器键", "128 维", [(128, "fp8", "128 × FP8"), (4, "scale", "1 × FP32")], V32_IDX),
    ("V4 主 KV / SWA 条目", "512 维", [(448, "fp8", "448 × FP8"), (128, "bf16", "64 × BF16 RoPE"),
                                        (7, "scale", "7 × E8M0"), (1, "pad", "填充")], MAIN_V4),
    ("V4 / V4.1 索引器键", "128 维", [(64, "fp4", "128 × E2M1"), (4, "scale", "4 × E8M0")], IDX),
    ("V4.1 SWA 条目", "512 维", [(512, "fp8", "512 × FP8（RoPE 维也量化）"), (16, "scale", "16 × E8M0")],
     SWA_V41),
    ("V4.1 主 KV 条目", "512 维", [(256, "fp4", "512 × E2M1（每字节两个）"), (32, "scale", "32 × E4M3")],
     MAIN_V41),
]
STY = {"fp8": ("orange", 0.30), "fp4": ("orange", 0.70), "bf16": ("blue", 0.30), "scale": ("violet", 0.45),
       "pad": ("muted", 0.25)}

svg.text(20, 24, "每格按字节数等比例画（0.8 像素 / 字节）", size=12, color="muted")
for i, (name, dims, segs, total) in enumerate(ROWS):
    assert sum(b for b, _, _ in segs) == total, name
    y = 48 + i * 58
    svg.text(20, y + 10, name, size=12, weight="700" if "V4.1 主" in name else None)
    bits = total * 8 / (576 if "MLA" in name else int(dims.split()[0]))
    svg.text(20, y + 27, "%s · %d 字节 · %.2f 比特/维" % (dims, total, bits), size=11, color="muted")
    x = BX
    small = []
    for b, kind, lab in segs:
        col, op = STY[kind]
        w = b * PXB
        svg.rect(x, y, w, BH, fill=col, fill_opacity=op, stroke=col, width=0.8)
        if w > 90:
            svg.text(x + w / 2, y + BH / 2 + 4, lab, size=11, anchor="middle")
        else:
            small.append((x + w / 2, lab, col))
        x += w
    if total * PXB < 200 and small:
        svg.text(x + 8, y + BH / 2 + 4, " + ".join(lab for _, lab, _ in small), size=11, color="ink")
        small = []
    for j, (cx, lab, col) in enumerate(small):
        svg.line(cx, y + BH, cx, y + BH + 6 + 12 * j, color=col, width=1)
        if cx > 600:
            svg.text(cx - 3, y + BH + 16 + 12 * j, lab, size=11, anchor="end", color=col)
        else:
            svg.text(cx + 3, y + BH + 16 + 12 * j, lab, size=11, color=col)

# legend
LY = 408
x = 20
for kind, lab in (("fp8", "FP8（E4M3）"), ("fp4", "FP4（E2M1）"), ("bf16", "BF16"), ("scale", "缩放因子"),
                  ("pad", "填充")):
    col, op = STY[kind]
    svg.rect(x, LY - 10, 14, 12, fill=col, fill_opacity=op, stroke=col, width=0.8)
    svg.text(x + 20, LY, lab, size=12)
    x += 20 + 12 * len(lab) * 0.62 + 30
svg.text(20, LY + 24, "V4.1 主 KV 条目对 V4：288 / 584 = 0.493，近乎减半；每维从 9.12 比特降到 4.50 比特。", size=12, color="orange")

svg.save("l19-kv-bytes-layout.svg")
