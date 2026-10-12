"""Figure 17.11: DeepEncoder data flow (redrawn after DeepSeek-OCR fig. 3, plus the concatenation
found in the released code).

Numbers (notes §1.4, §1.5, §3.5): 1024x1024 input, patch 16 -> 64 x 64 = 4096 tokens; SAM-base (80M,
sam_vit_b: 12 layers, width 768, global attention only in layers 2/5/8/11, window attention elsewhere);
two conv layers k3 s2 p1, channels 256 -> 512 -> 1024: 64x64 -> 32x32 -> 16x16 = 256 tokens (16x fewer);
CLIP-large (300M, 24 layers, width 1024, global attention, first patch-embedding layer removed);
code: CLIP output (CLS dropped) concatenated with the SAM features along channels -> 2048 -> linear -> 1280;
decoder DeepSeek-3B-MoE: 12 layers, hidden 1280, 64 routed experts pick 6 + 2 shared, ~570M active.
Attention-score matrix ratio if CLIP had to see all 4096 tokens: 4096^2 / 256^2 = 256.
Writes deepseek/figures/l17-deepencoder.svg
"""

from l17_svg import SVG

R = 1024
grid = R // 16
assert grid * grid == 4096 and (grid // 4) ** 2 == 256 and 4096 ** 2 // 256 ** 2 == 256

W, H = 760, 452
svg = SVG(W, H, "DeepEncoder：SAM（窗口注意力）→ 16 倍卷积压缩 → CLIP（全局注意力）→ 3B MoE 解码器")

# ---------------------------------------------------------------- row 1
Y = 56
# page
px, py, pw = 20, Y, 96
svg.rect(px, py, pw, pw * 1.18, fill="surface", stroke="ink", width=1.3, rx=3)
for i in range(9):
    ln = pw - 22 - (i % 3) * 14
    svg.line(px + 11, py + 14 + i * 12, px + 11 + ln, py + 14 + i * 12, color="muted", width=2.2)
svg.text(px + pw / 2, py + pw * 1.18 + 18, "一页文档", size=12, anchor="middle")
svg.text(px + pw / 2, py + pw * 1.18 + 34, "1024 × 1024 像素", size=11.5, anchor="middle", color="muted")

# patch grid
gx, gy, gs = 148, Y + 4, 100
for i in range(9):
    svg.line(gx + i * gs / 8, gy, gx + i * gs / 8, gy + gs, color="blue", width=0.8, opacity=0.6)
    svg.line(gx, gy + i * gs / 8, gx + gs, gy + i * gs / 8, color="blue", width=0.8, opacity=0.6)
svg.rect(gx, gy, gs, gs, fill="blue", fill_opacity=0.06, stroke="blue", width=1.3)
svg.text(gx + gs / 2, gy + gs + 18, "16×16 切块", size=12, anchor="middle")
svg.text(gx + gs / 2, gy + gs + 34, "64 × 64 = 4096 个", size=11.5, anchor="middle", color="blue")
svg.arrow(px + pw + 4, Y + 54, gx - 4, Y + 54, color="ink", width=1.5)

# SAM
sx, sw = 282, 176
svg.rect(sx, Y - 8, sw, 124, fill="blue", fill_opacity=0.08, stroke="blue", width=1.4, rx=8)
svg.text(sx + sw / 2, Y + 14, "SAM-base（80M）", size=13, anchor="middle", color="blue", weight="700")
svg.text(sx + sw / 2, Y + 34, "12 层，宽 768", size=12, anchor="middle")
svg.text(sx + sw / 2, Y + 52, "以窗口注意力为主", size=12, anchor="middle")
svg.text(sx + sw / 2, Y + 70, "（从 0 计第 2/5/8/11 层全局）", size=11.5, anchor="middle", color="muted")
# little window-attention glyph
for k in range(4):
    svg.rect(sx + 40 + k * 26, Y + 82, 22, 22, fill="blue", fill_opacity=0.18 if k % 2 == 0 else 0.08,
             stroke="blue", width=0.8)
svg.text(sx + sw / 2, Y + 134, "4096 个 token，多数层只在窗口内互相看", size=11.5, anchor="middle", color="muted")
svg.arrow(gx + gs + 4, Y + 54, sx - 4, Y + 54, color="ink", width=1.5)

# conv compressor
cx, cw = 496, 244
svg.rect(cx, Y - 8, cw, 124, fill="orange", fill_opacity=0.08, stroke="orange", width=1.4, rx=8)
svg.text(cx + cw / 2, Y + 14, "卷积压缩器：2 层，k3 s2 p1", size=13, anchor="middle", color="orange", weight="700")
stages = [("64×64×256", 58), ("32×32×512", 40), ("16×16×1024", 24)]
sxx = cx + 16
for i, (lab, s) in enumerate(stages):
    svg.rect(sxx, Y + 30 + (58 - s) / 2, s, s, fill="orange", fill_opacity=0.12 + 0.12 * i, stroke="orange", width=1)
    svg.text(sxx + s / 2, Y + 104, lab, size=11, anchor="middle", color="orange")
    if i < 2:
        svg.arrow(sxx + s + 4, Y + 59, sxx + s + 30, Y + 59, color="orange", width=1.3)
    sxx += s + 34
svg.text(cx + cw / 2, Y + 134, "每一维缩 4 倍：token 数 ÷ 16", size=11.5, anchor="middle", color="orange")
svg.arrow(sx + sw + 4, Y + 54, cx - 4, Y + 54, color="ink", width=1.5)

# ---------------------------------------------------------------- row 2
Y2 = 268
# CLIP
kx, kw = 20, 196
svg.rect(kx, Y2, kw, 110, fill="violet", fill_opacity=0.08, stroke="violet", width=1.4, rx=8)
svg.text(kx + kw / 2, Y2 + 22, "CLIP-large（300M）", size=13, anchor="middle", color="violet", weight="700")
svg.text(kx + kw / 2, Y2 + 42, "24 层，宽 1024", size=12, anchor="middle")
svg.text(kx + kw / 2, Y2 + 60, "全局注意力", size=12, anchor="middle")
svg.text(kx + kw / 2, Y2 + 78, "去掉首层 patch embedding", size=11.5, anchor="middle", color="muted")
svg.text(kx + kw / 2, Y2 + 98, "256 × 256 的注意力矩阵", size=11.5, anchor="middle", color="violet")
# route from conv output down-left to CLIP
svg.path("M%s %s V%s H%s V%s" % (cx + cw - 30, Y + 118, 222, kx + kw / 2, Y2 - 4), stroke="orange", width=1.6,
         arrow=True)
svg.text(cx + cw - 38, 214, "256 个 token × 1024", size=11.5, anchor="end", color="orange")

# concat
ccx, ccw = 256, 134
svg.rect(ccx, Y2 + 20, ccw, 70, fill="ink", fill_opacity=0.04, stroke="ink", width=1.3, rx=8)
svg.text(ccx + ccw / 2, Y2 + 44, "通道拼接", size=12.5, anchor="middle", weight="700")
svg.text(ccx + ccw / 2, Y2 + 62, "1024 ⊕ 1024 = 2048", size=11.5, anchor="middle")
svg.text(ccx + ccw / 2, Y2 + 80, "（开源代码）", size=11.5, anchor="middle", color="muted")
svg.arrow(kx + kw + 4, Y2 + 55, ccx - 4, Y2 + 55, color="violet", width=1.5)
# SAM bypass (dashed)
svg.path("M%s %s V%s" % (ccx + ccw / 2, 222, Y2 + 16), stroke="orange", width=1.5, dash="5 3", arrow=True)
svg.text(ccx + ccw / 2 + 8, 246, "SAM 特征绕过 CLIP", size=11.5, color="orange")

# linear
lx, lw = 424, 104
svg.box(lx, Y2 + 30, lw, 50, ["线性投影", "2048 → 1280"], color="ink", fill_opacity=0.04, size=12)
svg.arrow(ccx + ccw + 4, Y2 + 55, lx - 4, Y2 + 55, color="ink", width=1.5)

# decoder
dx, dw = 562, 178
svg.rect(dx, Y2, dw, 110, fill="green", fill_opacity=0.08, stroke="green", width=1.4, rx=8)
svg.text(dx + dw / 2, Y2 + 22, "DeepSeek-3B-MoE", size=13, anchor="middle", color="green", weight="700")
svg.text(dx + dw / 2, Y2 + 42, "12 层，隐藏维 1280", size=12, anchor="middle")
svg.text(dx + dw / 2, Y2 + 60, "64 选 6 路由 + 2 共享", size=12, anchor="middle")
svg.text(dx + dw / 2, Y2 + 78, "约 570M 激活", size=12, anchor="middle")
svg.text(dx + dw / 2, Y2 + 98, "输出文本 token", size=11.5, anchor="middle", color="green")
svg.arrow(lx + lw + 4, Y2 + 55, dx - 4, Y2 + 55, color="ink", width=1.5)
svg.text(lx + lw / 2, Y2 + 98, "256 个视觉 token", size=11.5, anchor="middle", color="muted")

svg.text(20, 412, "编码器约 380M 参数（80M + 300M）。若让 CLIP 直接吃 4096 个 token，全局注意力的分数矩阵会大 4096² / 256² = 256 倍；",
         size=12)
svg.text(20, 432, "先用便宜的窗口注意力看细节、再压 16 倍，是 DeepEncoder 控制激活显存的办法。拼接与投影维度取自开源代码与 config.json。",
         size=12)
svg.save("l17-deepencoder.svg")
