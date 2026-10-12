"""Figure 7.3: data flow of one top-2 MoE layer, drawn with Mixtral 8x7B's real sizes.

d = 4096, 8 SwiGLU experts with intermediate size 14336, top-2 (Mixtral config).
The router scores are the lesson's worked example: logits (0.2, 2, 0, -0.5, 1, 0.5, -0.3, -1)
-> softmax; experts 2 and 5 win. Per-expert parameters 3*4096*14336 are computed here.
Writes deepseek/figures/l07-moe-layer.svg
"""

import math

from l07_svg import SVG, comma

z = [0.2, 2.0, 0.0, -0.5, 1.0, 0.5, -0.3, -1.0]
Z = sum(math.exp(v) for v in z)
s = [math.exp(v) / Z for v in z]
top = sorted(range(8), key=lambda i: -s[i])[:2]
assert top == [1, 4]
d, dff = 4096, 14336
per_expert = 3 * d * dff

W, H = 740, 400
svg = SVG(W, H, "一个 top-2 MoE 层的数据流（按 Mixtral 8x7B 的一层标注尺寸）")

# ---- input vector u_t
ux, uy, uw, uh = 24, 150, 30, 120
svg.rect(ux, uy, uw, uh, fill="blue", fill_opacity=0.15, stroke="blue", width=1.4, rx=3)
svg.text(ux + uw / 2, uy - 26, "*u*_{t}", size=15, anchor="middle")
svg.text(ux + uw / 2, uy - 10, "4096 维", size=12, anchor="middle", color="muted")
svg.text(ux + uw / 2, uy + uh + 18, "注意力+残差", size=11, anchor="middle", color="muted")
svg.text(ux + uw / 2, uy + uh + 32, "之后的状态", size=11, anchor="middle", color="muted")

# ---- router
rx, ry, rw, rh = 96, 32, 150, 70
svg.rect(rx, ry, rw, rh, fill="violet", fill_opacity=0.12, stroke="violet", width=1.4, rx=6)
svg.text(rx + rw / 2, ry + 22, "路由器（分诊台）", size=13, anchor="middle", color="violet", weight="600")
svg.text(rx + rw / 2, ry + 42, "打分 *u*_{t}^{⊤}*e*_{i}，i = 1…8", size=12, anchor="middle")
svg.text(rx + rw / 2, ry + 60, "8 个 *e*_{i} 共 8×4096 参数", size=12, anchor="middle", color="muted")
svg.path("M%d %d L%d %d L%d %d" % (ux + uw, uy + 20, 70, uy + 20, 70, ry + rh / 2), stroke="ink", width=1.4)
svg.arrow(70, ry + rh / 2, rx, ry + rh / 2, color="ink", width=1.4)

# ---- score bars
bx0, by_base, bw, bgap = 290, 112, 16, 6
svg.text(bx0 - 4, 18, "softmax 后的亲和度 s_{i,t}", size=12, color="ink")
smax = max(s)
for i in range(8):
    x = bx0 + i * (bw + bgap)
    h = 62 * s[i] / smax
    chosen = i in top
    svg.rect(x, by_base - h, bw, h, fill="blue" if chosen else "muted", fill_opacity=0.85 if chosen else 0.35)
    svg.text(x + bw / 2, by_base + 14, str(i + 1), size=11, anchor="middle", color="ink" if chosen else "muted")
    if chosen:
        svg.text(x + bw / 2, by_base - h - 5, "%.2f" % s[i], size=11, anchor="middle", color="blue")
svg.arrow(rx + rw, ry + rh / 2, bx0 - 8, ry + rh / 2, color="violet", width=1.4)
bx_end = bx0 + 8 * (bw + bgap)
svg.text(bx_end + 8, 64, "取最大的 2 个：", size=12, color="ink")
svg.text(bx_end + 8, 82, "专家 2 和 5", size=12, color="blue", weight="600")
svg.text(bx_end + 8, 100, "其余 6 个门控值 = 0", size=12, color="muted")

# ---- experts row
ex0, ey, ew, eh, egap = 104, 184, 52, 56, 6
centers = []
for i in range(8):
    x = ex0 + i * (ew + egap)
    chosen = i in top
    svg.rect(x, ey, ew, eh, fill="blue" if chosen else "muted", fill_opacity=0.16 if chosen else 0.07,
             stroke="blue" if chosen else "rule", width=1.6 if chosen else 1.2, rx=5)
    svg.text(x + ew / 2, ey + 25, "FFN_{%d}" % (i + 1), size=13, anchor="middle", color="blue" if chosen else "muted",
             weight="600" if chosen else None)
    svg.text(x + ew / 2, ey + 44, "计算" if chosen else "跳过", size=12, anchor="middle",
             color="blue" if chosen else "muted")
    centers.append(x + ew / 2)
# bus from u_t into the experts
bus_y = ey - 16
svg.path("M%d %d L%d %d" % (ux + uw, uy + 60, 80, uy + 60), stroke="ink", width=1.4)
svg.line(80, uy + 60, 80, bus_y, color="ink", width=1.4)
svg.line(80, bus_y, centers[-1], bus_y, color="ink", width=1.4)
for i, cx in enumerate(centers):
    if i in top:
        svg.arrow(cx, bus_y, cx, ey, color="blue", width=1.8)
    else:
        svg.line(cx, bus_y, cx, ey - 2, color="rule", width=1.2, dash="3 3")
# selection from bars down to the chosen experts (dashed violet)
for i in top:
    x = bx0 + i * (bw + bgap) + bw / 2
    svg.path("M%.1f %d C %.1f %d, %.1f %d, %.1f %d" % (x, by_base + 20, x, 146, centers[i], 146, centers[i], bus_y - 4),
             stroke="violet", width=1.2, dash="4 3")

# ---- gates, sum, residual
sy = 318
sx = (centers[top[0]] + centers[top[1]]) / 2
for i in top:
    svg.arrow(centers[i], ey + eh, sx + (-10 if i == top[0] else 10), sy - 16, color="blue", width=1.6)
    svg.text(centers[i] + (-8 if i == top[0] else 8), ey + eh + 30, "× g_{%d,t} = %.2f" % (i + 1, s[i]), size=12,
             anchor="end" if i == top[0] else "start", color="blue")
svg.circle(sx, sy, 15, fill="surface", stroke="ink", width=1.5)
svg.text(sx, sy + 6, "Σ", size=16, anchor="middle")
px = 612
svg.circle(px, sy, 13, fill="surface", stroke="ink", width=1.5)
svg.text(px, sy + 5, "+", size=16, anchor="middle")
svg.arrow(sx + 15, sy, px - 13, sy, color="ink", width=1.5)
svg.path("M%d %d L%d %d L%d %d" % (ux + uw / 2, uy + uh + 40, ux + uw / 2, 372, px, 372), stroke="ink", width=1.3,
         dash="5 4")
svg.arrow(px, 372, px, sy + 13, color="ink", width=1.3)
svg.text(ux + uw / 2 + 150, 366, "残差：*u*_{t} 原样加回", size=12, color="muted")
svg.arrow(px + 13, sy, px + 50, sy, color="green", width=1.6)
svg.text(px + 54, sy - 6, "*h*_{t}′", size=15, color="green")
svg.text(px + 54, sy + 12, "4096 维", size=12, color="muted")

# ---- size notes
nx = 584
svg.text(nx, 150, "每个专家（SwiGLU）：", size=12, weight="600")
svg.text(nx, 168, "4096 → 14336 → 4096", size=12)
svg.text(nx, 186, "3×4096×14336", size=12, color="muted")
svg.text(nx, 204, "= %s 参数" % comma(per_expert), size=12, color="muted")
svg.text(nx, 226, "本层 8 个共 %.2fB，" % (8 * per_expert / 1e9), size=12)
svg.text(nx, 244, "每 token 只算 2 个", size=12, color="blue")
svg.text(nx, 262, "= %.2fB" % (2 * per_expert / 1e9), size=12, color="blue")

svg.save("l07-moe-layer.svg")
