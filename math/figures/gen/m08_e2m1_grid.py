"""Figure M8.3: every FP4 E2M1 code and where its value sits.

Top: the non-negative half of the number line 0..6 with the 8 values, the rounding
midpoints between neighbours (ties go to the even code) and the step size per power of two.
Bottom: the 8 non-negative codes decoded field by field (sign bit 1 gives the negatives).
Values come from m08_fp.values("E2M1"), the decode formula from m08_fp.decode.
Writes math/figures/m08-e2m1-grid.svg
"""

from m08_fp import decode, values
from m08_svg import SVG

vals = values("E2M1")
assert list(vals) == [0, 0.5, 1, 1.5, 2, 3, 4, 6]

W, H = 760, 376
X0, X1 = 70, 700
LY = 84


def px(v):
    return X0 + v / 6.0 * (X1 - X0)


svg = SVG(W, H, "FP4 E2M1 的全部取值：16 个码、15 个不同的数")
svg.text(20, 26, "正半轴（负半轴是镜像：符号位为 1 的 8 个码给出 −0、−0.5、…、−6）", size=13, weight="600")

# number line
svg.line(X0 - 10, LY, X1 + 10, LY, color="ink", width=1.4)
mids = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
for m in mids:
    svg.line(px(m), LY - 6, px(m), LY + 22, color="muted", width=1, dash="2 2")
    svg.text(px(m), LY + 35, "%g" % m, size=11, anchor="middle", color="muted")
for i, v in enumerate(vals):
    sub = 0 < v < 1
    svg.circle(px(v), LY, 6.5, fill=None if sub else "orange", stroke="orange", width=2)
    svg.text(px(v), LY - 14, "%g" % v, size=14, anchor="middle", weight="600", color="orange")
svg.text(X1 + 10, 52, "空心 = 次正规数；虚线 = 相邻两值的中点（舍入分界点）", size=11, color="muted", anchor="end")

# step-size brackets
BY = LY + 56
for a, b, lab in ((0, 2, "[0,2)：步长 0.5（0.5 是次正规数）"), (2, 4, "[2,4)：步长 1"),
                  (4, 6, "[4,6]：步长 2")):
    xa, xb = px(a) + 3, px(b) - 3
    svg.path("M%.1f %.1f v6 H%.1f v-6" % (xa, BY - 6, xb), stroke="blue", width=1.3)
    svg.text((xa + xb) / 2, BY + 16, lab, size=12, anchor="middle", color="blue")

# table of codes
TY = 196
COLW = 78
TX = 124
labels = ("位型 s e m", "指数 e / 尾数 m", "类型", "按公式", "值")
for r, lab in enumerate(labels):
    svg.text(TX - 10, TY + 18 + r * 28, lab, size=12, anchor="end", color="muted")
    if r:
        svg.line(TX - 4, TY + r * 28, TX + 8 * COLW, TY + r * 28, color="rule", width=0.8)
code = 0
for e in range(4):
    for m in range(2):
        x = TX + code * COLW + COLW / 2
        v = decode(2, 1, 1, e, m)
        assert v == vals[code]
        svg.text(x, TY + 18, "0 %d%d %d" % (e >> 1, e & 1, m), size=13, anchor="middle",
                 weight="600")
        svg.text(x, TY + 46, "%d / %d" % (e, m), size=12, anchor="middle")
        kind = "零" if (e == 0 and m == 0) else ("次正规" if e == 0 else "正规")
        svg.text(x, TY + 74, kind, size=12, anchor="middle", color="blue" if e else "muted")
        if e == 0:
            form = "0.%d × 2^{0}" % (5 * m) if m else "0"
        else:
            form = "1.%d × 2^{%d}" % (5 * m, e - 1)
        svg.text(x, TY + 102, form, size=12, anchor="middle")
        svg.text(x, TY + 130, "%g" % v, size=14, anchor="middle", weight="600", color="orange")
        code += 1
svg.text(TX, H - 10, "偏置 1；次正规数 (m/2)·2^{1−1}，正规数 (1 + m/2)·2^{e−1}；没有 Inf，也没有 NaN",
         size=12, color="muted")
svg.save("m08-e2m1-grid.svg")
