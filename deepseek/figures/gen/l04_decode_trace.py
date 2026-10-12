"""Figure 4.3: a four-token decode trace through one attention head, one panel per step.

Tokens x_1..x_4 = 从 前 有 座. At step t the model computes q_t, k_t, v_t for the new token only,
reads k_1..k_{t-1}, v_1..v_{t-1} from the cache, scores q_t against all t keys, mixes the t values,
and appends (k_t, v_t) to the cache. q_t is used once and thrown away. Schematic (d_h drawn as 4 cells).
Writes deepseek/figures/l04-decode-trace.svg
"""

from l04_svg import SVG

TOK = ["从", "前", "有", "座"]
W, H = 720, 600
PW, PH = 352, 280
svg = SVG(W, H, "一个注意力头里的 4 步解码：每步新算哪几行、复用哪几行")

CELL = 11
RH = 17  # row pitch


def rowvec(x, y, col, op, stroke=None, sw=1.0, n=4):
    for i in range(n):
        svg.rect(x + i * CELL, y, CELL - 1, CELL + 1, fill=col, fill_opacity=op, rx=1)
    if stroke:
        svg.rect(x - 2, y - 2, n * CELL + 2, CELL + 5, fill=None, stroke=stroke, width=sw, rx=3)


def panel(ox, oy, t):
    svg.rect(ox, oy, PW - 8, PH - 10, fill=None, stroke="rule", width=1, rx=8)
    svg.text(ox + 12, oy + 24, "第 %d 步：新 token x_{%d} =「%s」" % (t, t, TOK[t - 1]), size=13, weight="600")
    # token strip (grey = already processed, blue = this step, dashed = does not exist yet)
    for j in range(1, 5):
        x = ox + 226 + (j - 1) * 28
        y = oy + 9
        if j < t:
            svg.rect(x, y, 24, 22, fill="muted", fill_opacity=0.12, stroke="muted", width=0.8, rx=4)
            svg.text(x + 12, y + 16, TOK[j - 1], size=12, anchor="middle", color="muted")
        elif j == t:
            svg.rect(x, y, 24, 22, fill="blue", fill_opacity=0.18, stroke="blue", width=1.6, rx=4)
            svg.text(x + 12, y + 16, TOK[j - 1], size=12, anchor="middle", weight="600")
        else:
            svg.rect(x, y, 24, 22, fill=None, stroke="grid", width=1, rx=4, dash="3 2")
    # q_t
    kx, y0 = ox + 44, oy + 88
    rowvec(kx, oy + 52, "blue", 0.6, stroke="blue", sw=1.6)
    svg.text(kx - 8, oy + 62, "q_{%d}" % t, size=12, anchor="end", color="blue", weight="600")
    svg.text(kx + 4 * CELL + 8, oy + 62, "用完即弃", size=11, color="blue")
    svg.arrow(kx + 2 * CELL, oy + 68, kx + 2 * CELL, y0 - 4, color="blue", width=1.1)
    # K rows, score column, V rows
    sx = kx + 4 * CELL + 12
    vx = sx + 28
    for j in range(1, t + 1):
        y = y0 + (j - 1) * RH
        new = j == t
        svg.text(kx - 8, y + 10, TOK[j - 1], size=11, anchor="end", color="ink" if new else "muted")
        rowvec(kx, y, "violet", 0.6 if new else 0.2, stroke="blue" if new else "amber", sw=1.8 if new else 1.0)
        svg.rect(sx, y, CELL + 3, CELL + 1, fill="blue", fill_opacity=0.25, rx=2)
        rowvec(vx, y, "green", 0.6 if new else 0.2, stroke="blue" if new else "amber", sw=1.8 if new else 1.0)
    yb = y0 + t * RH
    svg.text(kx + 2 * CELL, yb + 12, "K", size=12, anchor="middle", color="violet", weight="600")
    svg.text(sx + 7, yb + 12, "a", size=12, anchor="middle", color="blue", weight="600")
    svg.text(vx + 2 * CELL, yb + 12, "V", size=12, anchor="middle", color="green", weight="600")
    # output
    oy_out = y0 + 4 * RH + 34
    rowvec(vx, oy_out, "green", 0.8)
    svg.text(vx - 8, oy_out + 10, "o_{%d}" % t, size=12, anchor="end", color="green", weight="600")
    svg.arrow(vx + 2 * CELL, yb + 18, vx + 2 * CELL, oy_out - 3, color="green", width=1.1)
    svg.text(kx - 20, oy_out + 34, "o_{%d} = Σ_{j≤%d} a_{j} v_{j}" % (t, t), size=12, color="green")
    # right column: the bill of this step
    tx = ox + 204
    ty = oy + 66
    svg.text(tx, ty, "新算：q_{%d}、k_{%d}、v_{%d}" % (t, t, t), size=12, color="blue", weight="600")
    svg.text(tx, ty + 24, ("读缓存：k、v 各 %d 行" % (t - 1)) if t > 1 else "读缓存：（还是空的）", size=12,
             color="amber", weight="600")
    svg.text(tx, ty + 48, "写回缓存：k_{%d}、v_{%d}" % (t, t), size=12, color="amber")
    svg.text(tx, ty + 72, "打分：%d 个点积" % t, size=12)
    svg.text(tx, ty + 96, "混合：%d 个值向量" % t, size=12)
    if t == 4:
        svg.text(tx, ty + 128, "→ 预测下一个：「山」", size=12, color="green", weight="600")


for t in range(1, 5):
    ox = 8 + ((t - 1) % 2) * PW
    oy = 6 + ((t - 1) // 2) * PH
    panel(ox, oy, t)

yl = H - 16   # legend boxes start ~14 units below the lower panels (which end at y = 556)
items = [("violet", 0.6, "blue", 1.8, "本步新算的行（蓝框）"), ("violet", 0.2, "amber", 1.0, "缓存里读出的行（黄框）")]
lx = 16
for col, op, st, sw, lab in items:
    svg.rect(lx, yl - 10, 30, 12, fill=col, fill_opacity=op, rx=1)
    svg.rect(lx - 2, yl - 12, 34, 16, fill=None, stroke=st, width=sw, rx=3)
    svg.text(lx + 40, yl, lab, size=12)
    lx += 196
svg.text(lx, yl, "紫 K、绿 V、蓝 q；一行 d_{h} 个数，画成 4 格", size=12, color="muted")
svg.save("l04-decode-trace.svg")
