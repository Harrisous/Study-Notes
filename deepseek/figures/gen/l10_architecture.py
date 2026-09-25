"""Figure 10.1: DeepSeek-V2 overall architecture, redrawn after Fig. 2 of the V2 paper with real sizes.

Left: one Transformer block (x60). Right: the DeepSeekMoE layer used in layers 2-60:
2 shared experts + 160 routed experts laid out as 8 devices x 20 experts, 6 routed experts
selected on at most 3 devices (device-limited routing, D = 8, M = 3).
Numbers: d = 5120, vocab 102400, expert intermediate 1536, first-layer dense FFN 12288,
MLA d_c = 512, d_h^R = 64, n_h = 128 (V2 paper sec. 3.1.2 and HF config.json).
Writes deepseek/figures/l10-v2-architecture.svg
"""

from l10_svg import SVG, box

W, H = 760, 540
svg = SVG(W, H, "DeepSeek-V2 整体结构：60 个 Transformer 块，第 2–60 层的 FFN 是 DeepSeekMoE")

# ------------------------------------------------------------------ left: one block
CX = 128
svg.text(CX, 22, "Transformer 块（共 60 层）", size=14, anchor="middle", weight="600")

# output head
box(svg, CX - 92, 36, 184, 30, "输出头 5120 → 102400", size=12)
# block frame
svg.rect(22, 84, 212, 356, stroke="muted", width=1.2, rx=10, dash="5 4")
svg.text(226, 104, "×60", size=14, anchor="end", color="muted", weight="600")

# second residual add (top)
y_add2 = 118
svg.circle(CX, y_add2, 10, fill="surface", stroke="ink", width=1.4)
svg.text(CX, y_add2 + 5, "+", size=15, anchor="middle")
svg.arrow(CX, y_add2 - 10, CX, 68, shorten=0)

# FFN box
box(svg, CX - 86, 150, 172, 74, None, color="blue", fill="blue", fill_opacity=0.10)
svg.text(CX, 172, "FFN", size=14, anchor="middle", weight="600")
svg.text(CX, 192, "第 1 层：稠密，中间维 12288", size=12, anchor="middle", color="muted")
svg.text(CX, 210, "第 2–60 层：DeepSeekMoE", size=12, anchor="middle", color="blue", weight="600")
svg.arrow(CX, 150, CX, y_add2 + 11)

# RMSNorm 2
box(svg, CX - 56, 244, 112, 24, "RMSNorm", size=12)
svg.arrow(CX, 244, CX, 225)

# first residual add
y_add1 = 292
svg.circle(CX, y_add1, 10, fill="surface", stroke="ink", width=1.4)
svg.text(CX, y_add1 + 5, "+", size=15, anchor="middle")
svg.arrow(CX, y_add1 - 10, CX, 269)

# MLA box
box(svg, CX - 86, 316, 172, 58, None, color="orange", fill="orange", fill_opacity=0.10)
svg.text(CX, 338, "MLA 注意力（第 9 课）", size=13, anchor="middle", weight="600")
svg.text(CX, 358, "n_{h}=128，d_{c}=512，d_{h}^{R}=64", size=12, anchor="middle", color="muted")
svg.arrow(CX, 316, CX, y_add1 + 11)

# RMSNorm 1
box(svg, CX - 56, 394, 112, 24, "RMSNorm", size=12)
svg.arrow(CX, 394, CX, 375)

# input
box(svg, CX - 92, 474, 184, 30, "嵌入 102400 × 5120", size=12)
svg.arrow(CX, 474, CX, 419)
svg.text(CX, 526, "每个 token 进入时是 d = 5120 维", size=12, anchor="middle", color="muted")

# residual paths: they run outside the FFN / MLA boxes (x 42..214) and inside the x60 frame (x 22..234)
RX, LX = 223, 32
svg.line(CX, 452, RX, 452, color="muted", width=1.4)
svg.line(RX, 452, RX, y_add1, color="muted", width=1.4)
svg.arrow(RX, y_add1, CX + 11, y_add1, color="muted", width=1.4)
svg.line(CX, 280, LX, 280, color="muted", width=1.4)
svg.line(LX, 280, LX, y_add2, color="muted", width=1.4)
svg.arrow(LX, y_add2, CX - 11, y_add2, color="muted", width=1.4)
svg.text(RX - 6, 468, "残差", size=11, anchor="middle", color="muted")

# connector to the right panel
svg.line(CX + 86, 204, 282, 204, color="blue", width=1.3, dash="4 3")
svg.arrow(282, 204, 300, 204, color="blue", width=1.3)

# ------------------------------------------------------------------ right: DeepSeekMoE layer
X0 = 304
svg.text(530, 22, "DeepSeekMoE 层（第 2–60 层，每层都一样）", size=14, anchor="middle", weight="600")

# output formula
box(svg, 360, 36, 340, 34, None, color="green", fill="green", fill_opacity=0.10)
svg.text(530, 58, "输出 h′_{t} = u_{t} + Σ 共享专家(u_{t}) + Σ g_{i,t}·路由专家_{i}(u_{t})", size=12,
         anchor="middle")
# sum node
svg.circle(530, 96, 10, fill="surface", stroke="ink", width=1.4)
svg.text(530, 101, "+", size=15, anchor="middle")
svg.arrow(530, 86, 530, 71)

# shared experts
SX, SY = X0, 150
svg.text(SX + 36, SY - 10, "共享专家 ×2", size=12, anchor="middle", color="violet", weight="600")
for k in range(2):
    box(svg, SX + 4, SY + k * 50, 64, 40, "共享 %d" % (k + 1), color="violet", fill="violet",
        fill_opacity=0.14, size=12, sub="1536")
svg.text(SX + 36, SY + 112, "每个 token", size=11, anchor="middle", color="muted")
svg.text(SX + 36, SY + 126, "都经过", size=11, anchor="middle", color="muted")
svg.line(SX + 36, SY, SX + 36, 110, color="violet", width=1.4)
svg.arrow(SX + 36, 110, 520, 100, color="violet", width=1.4)

# routed experts: 8 devices x 20 experts (2 columns x 10 rows)
selected = {1: [3, 12], 4: [0, 7, 15], 6: [9]}   # device index -> expert slots (0..19)
GX0, GY0 = 392, 150
pitch, sq, gap = 45, 14, 3
for dev in range(8):
    gx = GX0 + dev * pitch
    chosen_dev = dev in selected
    if chosen_dev:
        svg.rect(gx - 5, GY0 - 6, 2 * sq + gap + 10, 10 * (sq + gap) + 9, fill="blue",
                 fill_opacity=0.07, stroke="blue", width=1.1, rx=5, dash="3 2")
    for slot in range(20):
        col, row = slot % 2, slot // 2
        x = gx + col * (sq + gap)
        y = GY0 + row * (sq + gap)
        if chosen_dev and slot in selected[dev]:
            svg.rect(x, y, sq, sq, fill="blue", fill_opacity=0.85, stroke="blue", width=1, rx=2)
        else:
            svg.rect(x, y, sq, sq, fill="muted", fill_opacity=0.10, stroke="muted", width=0.8, rx=2,
                     stroke_opacity=0.6)
    svg.text(gx + sq + 1, GY0 + 10 * (sq + gap) + 18, "设备 %d" % (dev + 1), size=11, anchor="middle",
             color="blue" if chosen_dev else "muted")
svg.text(600, 362, "路由专家：8 台设备 × 每台 20 个 = 160 个", size=12, anchor="middle", color="ink")
# arrows from the selected devices up to the sum
for dev in selected:
    gx = GX0 + dev * pitch + sq
    svg.line(gx, GY0 - 6, gx, 124, color="blue", width=1.3)
    svg.arrow(gx, 124, 530 + (gx - 530) * 0.08, 104, color="blue", width=1.3)
svg.text(598, 134, "× g_{i,t}", size=12, color="blue")

# router
RY = 380
box(svg, 392, RY, 346, 50, None, color="violet", fill="violet", fill_opacity=0.08)
svg.text(565, RY + 20, "路由器：s_{i,t} = Softmax_{i}(u_{t}^{⊤} e_{i})，i = 1…160", size=12, anchor="middle")
svg.text(565, RY + 39, "设备受限 Top-6：先选 M = 3 台设备，再在其中取 6 个", size=12, anchor="middle",
         color="violet", weight="600")
svg.arrow(420, RY, 420, GY0 + 10 * (sq + gap) + 26, color="violet", width=1.4)

# input u_t
box(svg, 440, 470, 180, 32, "输入 u_{t}（5120 维）", size=12)
svg.arrow(530, 470, 530, RY + 51)
svg.line(440, 486, SX + 36, 486, color="violet", width=1.4)
svg.arrow(SX + 36, 486, SX + 36, SY + 132, color="violet", width=1.4)
# residual: right of the device grid (last column ends at x 738) and of the router box (ends at x 738)
svg.line(620, 486, 752, 486, color="muted", width=1.2, dash="4 3")
svg.line(752, 486, 752, 110, color="muted", width=1.2, dash="4 3")
svg.arrow(752, 110, 541, 97, color="muted", width=1.2)
svg.text(690, 480, "残差", size=11, anchor="middle", color="muted")

# footnote on widths
svg.text(522, 530, "每 token 激活 (6+2)×1536 = 12288，等于第 1 层稠密 FFN 的中间维", size=12,
         anchor="middle", color="muted")

svg.save("l10-v2-architecture.svg")
