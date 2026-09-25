"""Figure M8.12: block formats drawn to scale (0.5 unit per bit), 128 values per row.

Rows: DeepSeek-V3 FP8 activation tile (128 x E4M3 + one FP32 scale); MXFP4 (4 blocks of
32 x E2M1 + one E8M0 each); NVFP4 / DeepSeek-V4.1 FP4 KV (8 blocks of 16 x E2M1 + one E4M3
each; NVFP4 adds one FP32 scale per tensor, V4.1 omits it).  Bottom: bytes of one 512-dim
V4.1 latent KV entry stored as BF16, as FP8 (E4M3 + one E8M0 per 32, FlashMLA's V4.1 FP8
layout) and as FP4 (E2M1 + one E4M3 per 16).  Bits per value are computed here.
Writes math/figures/m08-block-formats.svg
"""

from m08_svg import SVG

PX = 0.5                      # units per bit
X0 = 200
W, H = 760, 470
svg = SVG(W, H, "块格式按位数等比例画出：每行都是 128 个数")

ROWS = [
    ("DeepSeek-V3 FP8 激活", "1×128 一组", 128, 8, 32, "FP32", "E4M3"),
    ("MXFP4", "每 32 个一组", 32, 4, 8, "E8M0", "E2M1"),
    ("NVFP4 / V4.1 KV", "每 16 个一组", 16, 4, 8, "E4M3", "E2M1"),
]
y = 64
svg.text(X0, 30, "橙色 = 低精度的数本身，紫色 = 缩放因子；长度按位数等比例（每行 128 个数）", size=12,
         color="muted")
for name, sub, k, be, bs, sname, ename in ROWS:
    svg.text(X0 - 12, y + 12, name, size=13, anchor="end", weight="600")
    svg.text(X0 - 12, y + 28, sub, size=11, anchor="end", color="muted")
    x = X0
    nblk = 128 // k
    for j in range(nblk):
        w_e, w_s = k * be * PX, bs * PX
        svg.rect(x, y, w_e, 22, fill="orange", fill_opacity=0.35, stroke="orange", width=0.8)
        svg.rect(x + w_e, y, w_s, 22, fill="violet", fill_opacity=0.55, stroke="violet", width=0.8)
        if j == 0:
            svg.text(x + w_e / 2, y - 6, "%d 个 %s" % (k, ename), size=11, anchor="middle", color="orange")
            svg.text(x + w_e + w_s / 2, y + 38, sname, size=11, anchor="middle", color="violet")
        x += w_e + w_s
    bits = be + bs / k
    svg.text(x, y - 6, "平均 %g 位/个" % bits, size=12, anchor="end", weight="600")
    if k == 16:
        gx = x + 24
        svg.rect(gx, y, 32 * PX * 1.0, 22, fill="violet", fill_opacity=0.15, stroke="violet", width=1,
                 dash="3 2")
        svg.text(gx + 24, y + 12, "+ 每张量 1 个 FP32", size=11, color="violet")
        svg.text(gx + 24, y + 27, "（NVFP4 有，V4.1 去掉了）", size=11, color="violet")
    y += 84

# bytes of one V4.1 latent entry
y += 6
svg.text(16, y, "一个 512 维的 KV 条目（DeepSeek-V4.1 的潜向量）占多少字节", size=13, weight="600")
y += 18
BPX = 0.47                    # units per byte
entries = [("BF16", 512 * 2, 0, "blue"),
           ("FP8：E4M3 + 每 32 个 1 个 E8M0", 512, 512 // 32, "orange"),
           ("FP4：E2M1 + 每 16 个 1 个 E4M3", 512 // 2, 512 // 16, "orange")]
for lab, data_b, scale_b, col in entries:
    svg.text(X0 - 12, y + 14, lab.split("：")[0], size=12, anchor="end", weight="600")
    svg.rect(X0, y, data_b * BPX, 20, fill=col, fill_opacity=0.35, stroke=col, width=0.8)
    if scale_b:
        svg.rect(X0 + data_b * BPX, y, scale_b * BPX, 20, fill="violet", fill_opacity=0.55, stroke="violet",
                 width=0.8)
    tot = data_b + scale_b
    txt = "%d B" % tot
    if scale_b:
        txt = "%d + %d = %d B" % (data_b, scale_b, tot)
    if "：" in lab:
        txt += "（%s）" % lab.split("：")[1]
    svg.text(X0 + tot * BPX + 8, y + 14, txt, size=12)
    y += 34
svg.text(X0, y + 6, "FP4 条目是 BF16 的 1/%.2f；本图的字节数按格式计算，与 FlashMLA 文档给出的 V4.1 行格式一致"
         % (1024 / 288), size=11, color="muted")
svg.save("m08-block-formats.svg")
