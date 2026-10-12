"""Figure 6.4: MinHash on a toy pair of sentences, and the banded-LSH switch (computed).

Left: two sentences cut into 2-character shingles. Union = 8 shingles, intersection = 4, so J = 0.5.
Four permutations h1..h4 (hand-picked for the illustration, not random draws) put the union in a new
order; mh(S) = the rank of the first shingle of S. mh(S_A) = mh(S_B) exactly when the union's first
shingle lies in the intersection (lesson derivation after eq. 6.1). Two of the four rows match.
Right: P(candidate | J) = 1 - (1 - J^r)^b (eq. 6.2) for the lesson's illustrative b = 20, r = 5, the
exercise-10 variant b = 25, r = 4, and a single MinHash value (P = J). Neither parameter set is
DeepSeek's (the report and README do not publish them).
Writes deepseek/figures/l06-minhash-lsh.svg
"""

import numpy as np

from l06_svg import SVG

A = "小猫坐在垫子上"
B = "小猫坐在毯子上"


def shingles(s, w=2):
    return [s[i:i + w] for i in range(len(s) - w + 1)]


SA, SB = shingles(A), shingles(B)
union = list(dict.fromkeys(SA + SB))
inter = [s for s in union if s in SA and s in SB]
J = len(inter) / len(union)
assert len(union) == 8 and len(inter) == 4 and J == 0.5

perms = [
    ["坐在", "在毯", "小猫", "垫子", "子上", "在垫", "毯子", "猫坐"],
    ["毯子", "子上", "在垫", "猫坐", "小猫", "在毯", "坐在", "垫子"],
    ["小猫", "垫子", "在毯", "子上", "毯子", "坐在", "猫坐", "在垫"],
    ["垫子", "毯子", "猫坐", "在垫", "坐在", "小猫", "子上", "在毯"],
]
for p in perms:
    assert sorted(p) == sorted(union)


def kind(s):
    if s in SA and s in SB:
        return "green"
    return "blue" if s in SA else "violet"


W, H = 760, 412
svg = SVG(W, H, "MinHash 的机制与分带 LSH 的概率开关")

# ---------------- left panel: MinHash on a toy pair
LX = 16
svg.text(LX, 26, "(a) MinHash：随机排一次序，看并集里排第一的片段", size=13, weight="600")
svg.text(LX, 50, "A：" + A, size=12, color="blue")
svg.text(LX + 140, 50, "B：" + B, size=12, color="violet")
svg.text(LX, 70, "以相邻 2 个字为一个片段，并集共 %d 个：" % len(union), size=12, color="muted")

CW, GAP, CH = 34, 4, 22
CX0 = LX + 26


def chip(x, y, s, strong=False):
    col = kind(s)
    svg.rect(x, y, CW, CH, fill=col, fill_opacity=0.28 if strong else 0.12, stroke=col,
             width=2.2 if strong else 1, rx=4)
    svg.text(x + CW / 2, y + 15.5, s, size=12, anchor="middle", color="ink")


uy = 80
for i, s in enumerate(sorted(union, key=lambda s: ("green", "blue", "violet").index(kind(s)))):
    chip(CX0 + i * (CW + GAP), uy, s)
lx = CX0 + 8 * (CW + GAP) + 8
svg.text(lx, uy + 15.5, "J = %d/%d = %.1f" % (len(inter), len(union), J), size=12, weight="600")
# colour key
ky = uy + 42
for i, (col, lab) in enumerate((("green", "两篇都有（交集）"), ("blue", "只在 A"), ("violet", "只在 B"))):
    kx = CX0 + (0, 128, 214)[i]
    svg.rect(kx, ky - 10, 12, 12, fill=col, fill_opacity=0.25, stroke=col, width=1, rx=2)
    svg.text(kx + 17, ky, lab, size=12, color=col)

svg.text(LX, ky + 30, "四个彼此独立的排列（从左到右是名次 1…8）：", size=12, color="muted")
ry0 = ky + 42
ROW = 50
for r, p in enumerate(perms):
    y = ry0 + r * ROW
    svg.text(LX, y + 15.5, "h_{%d}" % (r + 1), size=13, color="ink", weight="600")
    ia = next(i for i, s in enumerate(p) if s in SA)
    ib = next(i for i, s in enumerate(p) if s in SB)
    for i, s in enumerate(p):
        chip(CX0 + i * (CW + GAP), y, s, strong=(i in (ia, ib)))
    # markers under the first shingle of each set: that rank is mh(S_A) / mh(S_B)
    for idx, lab, col, dx in ((ia, "A", "blue", -6 if ia == ib else 0), (ib, "B", "violet", 6 if ia == ib else 0)):
        mx = CX0 + idx * (CW + GAP) + CW / 2 + dx
        svg.polygon([(mx, y + CH + 2), (mx - 4, y + CH + 8), (mx + 4, y + CH + 8)], fill=col, stroke=col, width=1)
        svg.text(mx, y + CH + 21, lab, size=12, anchor="middle", color=col, weight="600")
    same = ia == ib
    svg.text(lx, y + 15.5, "相等" if same else "不等", size=12, color="green" if same else "red", weight="600")
    svg.text(lx, y + 31, "A 第 %d，B 第 %d" % (ia + 1, ib + 1), size=11, color="muted")

fy = ry0 + 4 * ROW + 12
svg.text(LX, fy, "排第一的片段是绿色，两边的最小名次才相同；", size=12, color="ink")
svg.text(LX, fy + 18, "随机排列下它是绿色的概率 = 4/8 = J（这四行恰好两行相等）。", size=12, color="ink")

# ---------------- right panel: banded LSH
RX = 462
PX0, PX1, PY0, PY1 = 510, 740, 96, 304


def px(j):
    return PX0 + j * (PX1 - PX0)


def py(p):
    return PY1 - p * (PY1 - PY0)


svg.text(RX, 26, "(b) 分带 LSH：成为候选对的概率 (6.2)", size=13, weight="600")
# legend above the plot (the plot interior is crowded)
svg.line(RX, 46, RX + 22, 46, color="blue", width=2.4)
svg.text(RX + 28, 50, "b = 20，r = 5（正文示例）", size=12, color="blue")
svg.line(RX, 66, RX + 22, 66, color="blue", width=1.5, dash="6 3")
svg.text(RX + 28, 70, "b = 25，r = 4（练习 10）", size=12, color="blue")
svg.line(RX + 176, 66, RX + 198, 66, color="muted", width=1.3, dash="2 3")
svg.text(RX + 204, 70, "只比 1 个值", size=12, color="muted")

for k in range(0, 5):
    v = k / 4
    svg.line(PX0, py(v), PX1, py(v), color="grid", width=1)
    lab = "%d" % v if v in (0, 1) else ("%.2f" % v).rstrip("0")
    svg.text(PX0 - 6, py(v) + 4, lab, size=11, anchor="end", color="muted")
for k in range(0, 6):
    v = k / 5
    svg.line(px(v), PY0, px(v), PY1, color="grid", width=1)
    svg.text(px(v), PY1 + 16, "%.1f" % v if 0 < v < 1 else "%d" % v, size=11, anchor="middle", color="muted")
svg.line(PX0, PY1, PX1, PY1, color="muted", width=1.2)
svg.line(PX0, PY0, PX0, PY1, color="muted", width=1.2)
svg.text((PX0 + PX1) / 2, PY1 + 36, "Jaccard 相似度 J", size=12, anchor="middle", color="muted")
svg.text(RX + 8, (PY0 + PY1) / 2, "成为候选对的概率 P", size=12, anchor="middle", color="muted", rotate=-90)

js = np.linspace(0, 1, 201)


def curve(b, r):
    return 1 - (1 - js ** r) ** b


svg.path("M%.1f %.1f L%.1f %.1f" % (px(0), py(0), px(1), py(1)), stroke="muted", width=1.3, dash="2 3")
svg.path("M" + " L".join("%.1f %.1f" % (px(j), py(p)) for j, p in zip(js, curve(25, 4))),
         stroke="blue", width=1.5, dash="6 3", opacity=0.85)
svg.path("M" + " L".join("%.1f %.1f" % (px(j), py(p)) for j, p in zip(js, curve(20, 5))),
         stroke="blue", width=2.4)

# the switch: J^r = 1/b; drawn only in the empty lower-right strip so it crosses no label
thr = (1 / 20) ** (1 / 5)
svg.line(px(thr), py(0.22), px(thr), PY1, color="ink", width=1.1, dash="3 3")
svg.text(px(thr) + 5, PY1 - 22, "开关", size=12, color="ink")
svg.text(px(thr) + 5, PY1 - 7, "J ≈ %.2f" % thr, size=12, color="ink")


def p20(j):
    return 1 - (1 - j ** 5) ** 20


# the three worked values of the lesson's example
for j in (0.2, 0.5, 0.8):
    svg.circle(px(j), py(p20(j)), 4, fill="blue")
# J = 0.2: label in the empty upper-left pocket, with a leader line
lx2, ly2 = px(0.02), py(0.62)
svg.text(lx2, ly2, "J = 0.2：%.4f" % p20(0.2), size=12, color="blue", weight="600")
svg.line(px(0.2), py(p20(0.2)) - 5, px(0.16), ly2 + 5, color="blue", width=1)
# J = 0.5 (the left example): right of the point, above the dashed switch tick
svg.text(px(0.5) + 8, py(p20(0.5)) + 10, "%.2f（左例 J = 0.5）" % p20(0.5), size=12, color="blue", weight="600")
# J = 0.8: above the plateau
svg.text(px(0.8), PY0 - 6, "%.4f" % p20(0.8), size=12, anchor="middle", color="blue", weight="600")

svg.save("l06-minhash-lsh.svg")
