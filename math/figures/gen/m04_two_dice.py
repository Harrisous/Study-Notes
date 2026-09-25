"""Figure M4.1: two dice. Joint table (6x6), distribution of the sum, and the S = 8 slice.

All probabilities are counted here from the 36 equally likely pairs.
Writes math/figures/m04-two-dice.svg
"""

from fractions import Fraction

from m04_svg import SVG, Axes

pairs = [(a, b) for a in range(1, 7) for b in range(1, 7)]
count = {s: sum(1 for a, b in pairs if a + b == s) for s in range(2, 13)}
assert sum(count.values()) == 36
cond = [p for p in pairs if sum(p) == 8]
p_cond = Fraction(sum(1 for a, b in cond if a == 3), len(cond))
assert p_cond == Fraction(1, 5)

W, H = 720, 368
svg = SVG(W, H, "两颗骰子：联合分布表、点数和的分布与已知和为 8 的条件分布")

# ---- left: joint table
cell = 34
gx, gy = 78, 70
svg.text(gx + 3 * cell, 24, "联合分布：36 格，每格 1/36", size=13, anchor="middle", weight="600")
svg.text(gx + 3 * cell, 44, "蓝骰子 B", size=12, anchor="middle", color="muted")
svg.text(gx - 40, gy + 3 * cell, "红骰子 A", size=12, anchor="middle", color="muted", rotate=-90)
for j in range(6):
    svg.text(gx + j * cell + cell / 2, gy - 6, str(j + 1), size=11, anchor="middle", color="muted")
for i in range(6):
    svg.text(gx - 8, gy + i * cell + cell / 2 + 4, str(i + 1), size=11, anchor="end", color="muted")
for i in range(6):
    for j in range(6):
        a, b = i + 1, j + 1
        s = a + b
        x, y = gx + j * cell, gy + i * cell
        if s == 8:
            svg.rect(x + 1, y + 1, cell - 2, cell - 2, fill="blue", fill_opacity=0.28, rx=3)
        else:
            # grey tint proportional to how many cells share this sum (the anti-diagonal's length),
            # so the diagonals read as bands and the darkest band (s = 7) is the tallest bar on the right
            svg.rect(x + 1, y + 1, cell - 2, cell - 2, fill="muted",
                     fill_opacity=0.03 + 0.035 * (6 - abs(s - 7)), rx=3)
        svg.text(x + cell / 2, y + cell / 2 + 4, str(s), size=12, anchor="middle",
                 color="blue" if s == 8 else "ink", weight="700" if (a, b) == (3, 5) else None)
# outline the (A=3, B=5) cell
svg.rect(gx + 4 * cell + 1, gy + 2 * cell + 1, cell - 2, cell - 2, stroke="ink", width=2.2, rx=3)
svg.text(gx, gy + 6 * cell + 22, "格子里的数是点数和 S = A + B", size=12, color="muted")
svg.text(gx, gy + 6 * cell + 40, "蓝格：S = 8，共 5 格", size=12, color="blue")
svg.text(gx, gy + 6 * cell + 58, "粗框：(A, B) = (3, 5)，5 格里的 1 格", size=12, color="ink")

# ---- right: distribution of S
ax = Axes(svg, 400, 70, 290, 204, (1.5, 12.5), (0, 6.6 / 36))
yt = [0, 2 / 36, 4 / 36, 6 / 36]
ax.frame(xticks=[], yticks=yt, yfmt=lambda v: "%d/36" % round(v * 36) if v else "0", grid=True)
svg.text(ax.x0 + ax.w / 2, 24, "点数和 S 的分布：P(S = s) = (6 − |s − 7|)/36", size=13,
         anchor="middle", weight="600")
bw = 20
for s, k in count.items():
    x = ax.X(s)
    y = ax.Y(k / 36)
    hi = s == 8
    svg.rect(x - bw / 2, y, bw, ax.Y(0) - y, fill="blue" if hi else "muted",
             fill_opacity=0.55 if hi else 0.28, stroke="blue" if hi else None, width=1.2)
    svg.text(x, y - 5, str(k), size=11, anchor="middle", color="blue" if hi else "muted")
    svg.text(x, ax.Y(0) + 16, str(s), size=11, anchor="middle", color="blue" if hi else "muted")
svg.text(ax.x0 + ax.w / 2, ax.Y(0) + 34, "s（柱顶数字 = 格子数）", size=12, anchor="middle")

# ---- bottom: the conditional computation
yb = 346
svg.text(400, yb - 18, "P(S = 8) = 5/36", size=13, color="blue")
svg.text(400, yb, "P(A = 3 | S = 8) = (1/36) ÷ (5/36) = 1/5", size=13, color="ink")

svg.save("m04-two-dice.svg")
