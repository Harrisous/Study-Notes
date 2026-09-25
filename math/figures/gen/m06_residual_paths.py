"""Figure M6.9: why a residual stack keeps an identity path for the gradient.

Top: a plain stack of three blocks; the Jacobian from h0 to h3 is J3 J2 J1 (one path).
Middle: a residual stack h_l = h_{l-1} + f_l(h_{l-1}) (the lesson's notation, M6.21); the Jacobian of each step is I + J_l, and
the product (I + J3)(I + J2)(I + J1) expands into 2^3 = 8 paths, one of which is the identity.
Bottom: the expansion with the path counts C(3, k). Schematic; the counts are computed.
Writes math/figures/m06-residual-paths.svg
"""

from itertools import combinations
from math import comb

from m06_svg import SVG

L = 3
paths = [c for k in range(L + 1) for c in combinations(range(1, L + 1), k)]
assert len(paths) == 2 ** L
print(paths)

W, H = 720, 446
svg = SVG(W, H, "残差把雅可比的连乘变成 2^L 条路径之和，其中一条是恒等")

# ---- plain stack
svg.text(20, 26, "无残差：*h*_{l} = f_{l}(*h*_{l−1})，从 *h*_{0} 到 *h*_{3} 只有一条路", size=13, weight="600")
y = 74
xs = [40, 190, 340, 490, 640]
svg.text(xs[0], y + 5, "*h*₀", size=14, anchor="middle", weight="600")
svg.text(xs[4], y + 5, "*h*₃", size=14, anchor="middle", weight="600")
centres = [150, 300, 450]
for i, bx in enumerate(centres):
    svg.rect(bx - 40, y - 18, 80, 36, fill="blue", fill_opacity=0.12, stroke="blue", width=1.2, rx=6)
    svg.text(bx, y + 5, "f" + "₁₂₃"[i], size=14, anchor="middle", color="blue", weight="600")
    svg.text(bx, y - 28, "J_{%d}" % (i + 1), size=13, anchor="middle", color="muted")
chain = [(56, 108), (192, 258), (342, 408), (492, 622)]
for a, b in chain:
    svg.arrow(a, y, b, y, color="ink", width=1.4)
svg.text(W / 2, y + 42, "∂*h*₃/∂*h*₀ = J₃ J₂ J₁：每个因子都缩一点，乘起来就消失；都放大一点，就爆炸", size=12,
         anchor="middle", color="red")

# ---- residual stack
top = 150
svg.text(20, top, "有残差：*h*_{l} = *h*_{l−1} + f_{l}(*h*_{l−1})，每一步的雅可比是 I + J_{l}", size=13, weight="600")
belt = top + 110
svg.line(56, belt, 624, belt, color="green", width=3.2)
svg.text(40, belt + 5, "*h*₀", size=14, anchor="middle", weight="600")
svg.text(640, belt + 5, "*h*₃", size=14, anchor="middle", weight="600")
plus_x = [230, 400, 570]
for i, px_ in enumerate(plus_x):
    tap = px_ - 110
    bx = px_ - 55
    # branch up, box, back down into the plus
    svg.line(tap, belt, tap, belt - 58, color="blue", width=1.4)
    svg.rect(bx - 34, belt - 76, 68, 34, fill="blue", fill_opacity=0.12, stroke="blue", width=1.2, rx=6)
    svg.text(bx, belt - 54, "f" + "₁₂₃"[i], size=14, anchor="middle", color="blue", weight="600")
    svg.line(tap, belt - 58, bx - 34, belt - 58, color="blue", width=1.4)
    svg.line(bx + 34, belt - 58, px_, belt - 58, color="blue", width=1.4)
    svg.arrow(px_, belt - 58, px_, belt - 12, color="blue", width=1.4)
    svg.circle(px_, belt, 10, fill="surface", stroke="ink", width=1.4)
    svg.text(px_, belt + 5, "+", size=15, anchor="middle", weight="700")
    svg.text(bx, belt - 88, "J_{%d}" % (i + 1), size=13, anchor="middle", color="muted")
# backward along the belt
svg.arrow(610, belt + 26, 70, belt + 26, color="red", width=1.6)
svg.text(340, belt + 44, "反向：梯度沿绿色传送带原样传回（恒等通路，局部导数 = I），", size=12, anchor="middle", color="red")
svg.text(340, belt + 61, "每经过一个 ⊕，再从分支加回一份 J_{l}^{⊤} 的修正", size=12, anchor="middle", color="red")

# ---- expansion
ey = belt + 104
svg.text(20, ey, "展开：", size=13, weight="600")
terms = [("I", "green", 1), ("J₁ + J₂ + J₃", "blue", 3), ("J₂J₁ + J₃J₁ + J₃J₂", "blue", 3), ("J₃J₂J₁", "red", 1)]
x = 70
svg.text(x, ey, "(I + J₃)(I + J₂)(I + J₁) =", size=13)
x = 236
xs_terms = []
for k, (t, col, cnt) in enumerate(terms):
    if k:
        svg.text(x, ey, "+", size=13, color="muted")
        x += 16
    svg.text(x, ey, t, size=13, color=col, weight="600")
    w = 7.0 * len(t) + (6 if k == 0 else 0)
    xs_terms.append((x + w / 2, cnt, col))
    x += w + 10
for (cx_, cnt, col), k in zip(xs_terms, range(4)):
    assert cnt == comb(L, k)
    svg.text(cx_, ey + 22, "%d 条" % cnt, size=12, anchor="middle", color=col)
    lab = ["只走捷径", "经过 1 个分支", "经过 2 个分支", "经过全部分支"][k]
    svg.text(cx_, ey + 39, lab, size=11, anchor="middle", color="muted")
svg.text(20, H - 10, "L 层残差共有 2^{L} 条路径，经过 k 个分支的有 C(L, k) 条；无残差时只剩最后一项。", size=12, color="ink")

svg.save("m06-residual-paths.svg")
