"""Figure 18.8: partial RoPE in V4 and the inverse rotation of the output.

Top: the 512-dim entry / query = 448 dims without position (entry: FP8) + last 64 dims with RoPE
(entry: BF16). Bottom: one rotation plane with theta = 0.1 rad per position, query at t = 10,
entry at p = 4 (the worked example in the lesson): score depends on (p - t) * theta = -0.6,
and after rotating the output by -t the entry's contribution sits at -0.6 rad.
Writes deepseek/figures/l18-partial-rope.svg
"""

import math

from l18_svg import SVG, fmt

W, H = 760, 412
svg = SVG(W, H, "部分 RoPE 与输出逆旋转")

# ---------------------------------------------------------------- top: layout of the 512 dims
BX0, BW = 150, 540
sc = BW / 512
for n, (lab, col, sub) in enumerate([("条目 *e*_{s}（K = V）", "orange", "448 维，FP8，不带位置"),
                                      ("查询 *q*_{t,i}", "blue", "448 维，不带位置")]):
    y = 30 + n * 44
    svg.text(BX0 - 10, y + 17, lab, size=12, anchor="end")
    svg.rect(BX0, y, 448 * sc, 24, fill=col, fill_opacity=0.16, stroke=col, width=1)
    svg.text(BX0 + 448 * sc / 2, y + 17, sub, size=12, anchor="middle")
    svg.rect(BX0 + 448 * sc, y, 64 * sc, 24, fill="red", fill_opacity=0.16, stroke="red", width=1)
    svg.text(BX0 + 480 * sc, y + 17, "64 维 RoPE", size=11, anchor="middle", color="red")
svg.text(BX0 + 480 * sc, 124, "条目：BF16", size=11, anchor="middle", color="red")
svg.text(BX0, 124, "只有最后 64 维按位置旋转；打分时两段各自点积再相加", size=11, color="muted")

# ---------------------------------------------------------------- bottom: one rotation plane
TH, T, P = 0.1, 10, 4
R = 62
panels = [(130, "① 条目在位置 p = 4", "转 pθ = 0.4"),
          (380, "② 查询在位置 t = 10", "转 tθ = 1.0，夹角 (p−t)θ = −0.6"),
          (630, "③ 输出再转 −t", "0.4 − 1.0 = −0.6，只剩相对位置")]
CY = 272


def pt(cx, a, r=R):
    return cx + r * math.cos(a), CY - r * math.sin(a)


def vec(cx, a, col, width=2.4, dash=None, label=None, lr=None):
    x, y = pt(cx, a)
    svg.line(cx, CY, x, y, color=col, width=width, dash=dash, arrow=dash is None)
    if label:
        lx, ly = pt(cx, a, lr or R + 16)
        svg.text(lx, ly + 4, label, size=12, anchor="middle", color=col)


def arc(cx, a0, a1, r, col):
    x0, y0 = pt(cx, a0, r)
    x1, y1 = pt(cx, a1, r)
    sweep = 0 if a1 > a0 else 1
    svg.path("M%s %s A%s %s 0 0 %d %s %s" % (fmt(x0), fmt(y0), r, r, sweep, fmt(x1), fmt(y1)),
             stroke=col, width=1.4, arrow=True)


for cx, title, sub in panels:
    svg.circle(cx, CY, R, stroke="rule", width=1.2)
    svg.line(cx - R - 8, CY, cx + R + 8, CY, color="rule", width=1)
    svg.line(cx, CY - R - 8, cx, CY + R + 8, color="rule", width=1)
    svg.text(cx, 160, title, size=12, anchor="middle", weight="700")
    svg.text(cx, 178, sub, size=11, anchor="middle", color="muted")

# panel 1: entry rotated to 0.4
vec(130, 0, "muted", width=1.4, dash="3 3")
vec(130, P * TH, "orange", label="R_{p}*e*")
arc(130, 0, P * TH, 30, "orange")
# panel 2: entry at 0.4, query at 1.0
vec(380, P * TH, "orange", label="R_{p}*e*")
vec(380, T * TH, "blue", label="R_{t}*q*")
arc(380, T * TH, P * TH, 34, "red")
svg.text(380, CY + R + 26, "打分 = cos(−0.6) = 0.825", size=12, anchor="middle", color="red")
# panel 3: output contribution rotated back by -t
vec(630, P * TH, "orange", width=1.4, dash="4 3", label="转前 0.4", lr=R + 22)
vec(630, (P - T) * TH, "green", label="R_{p−t}*e*")
arc(630, P * TH, (P - T) * TH, 40, "green")
svg.text(630, CY + R + 26, "(0.825, −0.565)", size=12, anchor="middle", color="green")

svg.text(20, 384, "情形 B（t = 110，p = 104）：条目转 10.4、查询转 11.0，打分与逆旋转后的输出都与上图完全相同；",
         size=11, color="muted")
svg.text(20, 401, "若不做逆旋转，输出里这一项是 R_{10.4}*e* 而不是 R_{0.4}*e*：同一段话换个位置，输出就变了（泄露绝对位置）。",
         size=11, color="muted")

svg.save("l18-partial-rope.svg")
