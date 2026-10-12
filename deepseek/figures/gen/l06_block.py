"""Figure 6.5: one DeepSeek LLM 67B layer with its real dimensions (schematic, hand-placed).

Dimensions from dsllm Table 2 and the Hugging Face config (research notes §3.1):
d = 8192, 64 query heads, 8 KV heads (GQA-8), d_h = 128, FFN 22016, 95 layers, V = 102400,
untied embedding / output head. 7B: d = 4096, 32 heads, MHA, FFN 11008, 30 layers.
Writes deepseek/figures/l06-block.svg
"""

from l06_svg import SVG, box

W, H = 760, 560
svg = SVG(W, H, "DeepSeek LLM 67B 的一层：Pre-RMSNorm、GQA-8 注意力、SwiGLU 前馈，按真实维度标注")

# ---------------- left: the stack (top = input, bottom = output)
lx, lw = 24, 220
cx = lx + lw / 2
items = [
    ("tok", 18, 34, ["输入 token id"], "muted"),
    ("emb", 66, 44, ["词嵌入 102400×8192", "（与输出头不共享）"], "ink"),
    ("n1", 142, 30, ["RMSNorm"], "ink"),
    ("att", 186, 50, ["GQA 注意力", "64 个查询头 / 8 个 KV 头"], "blue"),
    ("add1", 250, 0, None, None),
    ("n2", 280, 30, ["RMSNorm"], "ink"),
    ("ffn", 324, 50, ["SwiGLU 前馈", "8192 → 22016 → 8192"], "green"),
    ("add2", 388, 0, None, None),
    ("nf", 440, 30, ["最终 RMSNorm"], "ink"),
    ("out", 484, 44, ["输出头 8192×102400", "→ logits"], "ink"),
]
ys = {}
for key, y, h, lines, col in items:
    ys[key] = (y, h)
    if lines is None:
        continue
    box(svg, lx + 20, y, lw - 40, h, lines, color=col, fill_opacity=0.08, title_size=13, size=12)

# residual adds
for key in ("add1", "add2"):
    y = ys[key][0]
    svg.circle(cx, y + 10, 10, fill="surface", stroke="ink", width=1.4)
    svg.text(cx, y + 15, "+", size=15, anchor="middle", weight="600")

# main arrows
def v(y1, y2):
    svg.arrow(cx, y1, cx, y2, color="ink", width=1.4)

v(52, 64)
v(110, 140)
v(172, 184)
v(236, 248)
v(270, 278)
v(310, 322)
v(374, 386)
v(408, 438)
v(470, 482)
# residual skip lines (left side of the block)
rx = lx + 8
svg.path("M%.1f 126 H %.1f V 260 H %.1f" % (cx, rx, cx - 12), stroke="muted", width=1.4, arrow=True)
svg.path("M%.1f 274 H %.1f V 398 H %.1f" % (cx, rx, cx - 12), stroke="muted", width=1.4, arrow=True)
svg.text(rx - 10, 196, "残差", size=12, anchor="middle", color="muted", rotate=-90)

# layer bracket
bx = lx + lw - 6
svg.path("M%.1f 136 h 8 V 414 h -8" % bx, stroke="ink", width=1.4)
svg.text(bx + 14, 262, "×95 层", size=13, weight="600")
svg.text(bx + 14, 280, "（7B：×30）", size=12, color="muted")

# ---------------- right top: GQA detail
px0 = 318
svg.text(px0, 24, "GQA-8：每 8 个查询头共享一对 K、V 头（每个头 d_{h} = 128）", size=13, weight="600")
gx, gy = px0, 44
bar_w, bar_gap, grp_gap = 3.4, 1.2, 10
for g in range(8):
    x_start = gx + g * (8 * (bar_w + bar_gap) + grp_gap)
    for k in range(8):
        x = x_start + k * (bar_w + bar_gap)
        svg.rect(x, gy, bar_w, 34, fill="blue", fill_opacity=0.55)
    grp_w = 8 * (bar_w + bar_gap) - bar_gap
    mid = x_start + grp_w / 2
    svg.path("M%.1f %.1f L %.1f %.1f" % (x_start, gy + 38, mid, gy + 52), stroke="muted", width=0.9)
    svg.path("M%.1f %.1f L %.1f %.1f" % (x_start + grp_w, gy + 38, mid, gy + 52), stroke="muted", width=0.9)
    svg.rect(mid - 11, gy + 54, 10, 26, fill="violet", fill_opacity=0.6)
    svg.rect(mid + 1, gy + 54, 10, 26, fill="green", fill_opacity=0.6)
last = gx + 8 * (8 * (bar_w + bar_gap) + grp_gap) - grp_gap
svg.text(last + 8, gy + 22, "Q：64 头", size=12, color="blue")
svg.text(last + 8, gy + 72, "K、V：8 头", size=12, color="violet")

rows = [
    ("W^{Q}：8192 → 64×128 = 8192", "blue"),
    ("W^{K}、W^{V}：8192 → 8×128 = 1024（各一个）", "violet"),
    ("W^{O}：8192 → 8192", "ink"),
    ("每 token 每层缓存 2×8×128 = 2048 个数（MHA 为 16384）", "amber"),
]
for i, (s, col) in enumerate(rows):
    svg.text(px0, 160 + i * 21, s, size=12, color=col)
svg.text(px0, 250, "注意力参数/层：2d^{2} + 2d·1024 = 1.51 亿（MHA 4d^{2} = 2.68 亿）", size=12, color="muted")

# ---------------- right bottom: SwiGLU detail
sy = 292
svg.text(px0, sy, "SwiGLU：门控支路 × 线性支路，再投回 d", size=13, weight="600")
hx, hy = px0 + 8, sy + 44
box(svg, hx, hy, 70, 34, ["*h*", ], color="ink", fill_opacity=0.06, title_size=13)
svg.text(hx + 35, hy + 50, "8192", size=12, anchor="middle", color="muted")
# two branches
b1y, b2y = sy + 18, sy + 82
bx1 = hx + 96
box(svg, bx1, b1y, 104, 34, ["W_{gate}：→ 22016"], color="green", fill_opacity=0.08, title_size=12)
box(svg, bx1, b2y, 104, 34, ["W_{up}：→ 22016"], color="green", fill_opacity=0.08, title_size=12)
svg.arrow(hx + 70, hy + 12, bx1 - 2, b1y + 17, color="ink", width=1.3)
svg.arrow(hx + 70, hy + 22, bx1 - 2, b2y + 17, color="ink", width=1.3)
sx = bx1 + 124
box(svg, sx, b1y, 50, 34, ["SiLU"], color="ink", fill_opacity=0.05, title_size=12)
svg.arrow(bx1 + 104, b1y + 17, sx - 2, b1y + 17, color="ink", width=1.3)
mx, my = sx + 80, hy + 17
svg.circle(mx, my, 12, fill="surface", stroke="ink", width=1.4)
svg.text(mx, my + 5, "⊙", size=15, anchor="middle")
svg.arrow(sx + 50, b1y + 17, mx - 9, my - 8, color="ink", width=1.3)
svg.arrow(bx1 + 104, b2y + 17, mx - 9, my + 8, color="ink", width=1.3)
dx = mx + 24
box(svg, dx, hy, 96, 34, ["W_{down}：→ 8192"], color="green", fill_opacity=0.08, title_size=12)
svg.arrow(mx + 12, my, dx - 2, my, color="ink", width=1.3)
svg.text(px0, sy + 150, "FFN 参数/层：3 × 8192 × 22016 = 5.41 亿，约占一层的 78%", size=12, color="muted")
svg.text(px0, sy + 171, "22016/8192 = 2.6875，略大于 8/3（7B 的 11008/4096 相同）", size=12,
         color="muted")

# ---------------- right footer: 7B differences
svg.text(px0, 500, "7B 的不同之处：d = 4096，32 个查询头、32 个 KV 头（即 MHA），", size=12)
svg.text(px0, 520, "FFN 11008，30 层。两者都用 RoPE（θ = 10000）、上下文 4096。", size=12)

svg.save("l06-block.svg")
