"""Figure M5.5: Jensen's inequality for the concave function ln.

X takes 1 or 4 with probability 1/2 each: E[ln X] = (ln1 + ln4)/2 = 0.693 lies on the chord, ln E[X] = ln 2.5 = 0.916
lies on the curve; the gap is 0.223. The tangent at mu = 2.5, y = ln mu + (x - mu)/mu, lies above the curve
everywhere: that supporting line is the whole proof. All geometry computed here.
Writes math/figures/m05-jensen.svg
"""

import math

from m05_svg import SVG

W, H = 720, 390
X0, X1, Y0, Y1 = 70, 520, 30, 330
XMIN, XMAX, YMIN, YMAX = 0.0, 5.0, -1.0, 2.0


def px(x):
    return X0 + (x - XMIN) / (XMAX - XMIN) * (X1 - X0)


def py(y):
    return Y1 - (y - YMIN) / (YMAX - YMIN) * (Y1 - Y0)


svg = SVG(W, H, "Jensen 不等式：对凹函数 ln，函数的平均 ≤ 平均的函数")
for x in range(0, 6):
    svg.line(px(x), Y0, px(x), Y1, color="grid", width=1)
    svg.text(px(x), Y1 + 17, str(x), size=11, anchor="middle", color="muted")
for y in (-1, -0.5, 0, 0.5, 1, 1.5, 2):
    svg.line(X0, py(y), X1, py(y), color="grid", width=1)
    svg.text(X0 - 8, py(y) + 4, ("%g" % y).replace("-", "−"), size=11, anchor="end", color="muted")
svg.line(X0, py(0), X1, py(0), color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "x", size=12, anchor="middle", color="muted")

# curve ln x, clipped to y >= YMIN
xs = [math.exp(YMIN) + k * (XMAX - math.exp(YMIN)) / 400 for k in range(401)]
svg.path("M" + " L".join("%.1f %.1f" % (px(x), py(math.log(x))) for x in xs), stroke="blue", width=2.6)
# lower left, right of the steep part of the curve: the tangent crowds the curve on the right
svg.text(px(0.72), py(-0.6), "y = ln x", size=13, color="blue", weight="600")

mu = 2.5
# tangent at mu, clipped to the plot box
def tan(x):
    return math.log(mu) + (x - mu) / mu
xa = max(XMIN, mu + (YMIN - math.log(mu)) * mu)
xb = min(XMAX, mu + (YMAX - math.log(mu)) * mu)
svg.line(px(xa), py(tan(xa)), px(xb), py(tan(xb)), color="orange", width=1.6, dash="7 4")
svg.text(px(0.12), py(1.85), "橙色虚线：在 μ = E[X] = 2.5 处的切线", size=12, color="orange")
svg.text(px(0.12), py(1.85) + 17, "ln x ≤ ln μ + (x − μ)/μ（处处成立）", size=12, color="orange")

# chord between the two outcomes
a, b = 1.0, 4.0
svg.line(px(a), py(math.log(a)), px(b), py(math.log(b)), color="violet", width=1.8)
svg.circle(px(a), py(0), 4.5, fill="ink")
svg.text(px(a) + 8, py(0) + 18, "X = 1：ln 1 = 0", size=12)
svg.circle(px(b), py(math.log(b)), 4.5, fill="ink")
svg.text(px(b) + 8, py(math.log(b)) + 20, "X = 4", size=12)
svg.text(px(b) + 8, py(math.log(b)) + 37, "ln 4 = 1.386", size=12)

m_chord = (math.log(a) + math.log(b)) / 2
m_curve = math.log(mu)
svg.line(px(mu), py(0), px(mu), py(m_curve), color="muted", width=1, dash="2 3")
svg.circle(px(mu), py(m_chord), 5, fill="violet")
svg.circle(px(mu), py(m_curve), 5, fill="blue")
svg.text(px(mu) + 10, py(m_chord) + 16, "E[ln X] = %.3f" % m_chord, size=12, color="violet", weight="600")
svg.text(px(mu) - 10, py(m_curve) - 10, "ln E[X] = ln 2.5 = %.3f" % m_curve, size=12, anchor="end", color="blue",
         weight="600")
# gap bracket
gx = px(mu) + 4
svg.path("M%.1f %.1f h6 V%.1f h-6" % (gx, py(m_curve), py(m_chord)), stroke="ink", width=1.2)
svg.text(gx + 14, (py(m_curve) + py(m_chord)) / 2 + 4, "差 %.3f" % (m_curve - m_chord), size=12)

# side notes
NX = 548
svg.rect(NX - 8, 40, 180, 236, fill="blue", fill_opacity=0.05, stroke="rule", width=1, rx=8)
lines = [("X 等可能取 1 或 4", "ink", None),
         ("E[X] = 2.5", "ink", None),
         ("", None, None),
         ("弦的中点：函数的平均", "violet", "600"),
         ("E[ln X] = 0.693", "violet", None),
         ("", None, None),
         ("曲线上的点：平均的函数", "blue", "600"),
         ("ln E[X] = 0.916", "blue", None),
         ("", None, None),
         ("ln 是凹函数，弦在曲线下方", "ink", None),
         ("所以 E[ln X] ≤ ln E[X]", "ink", "600"),
         ("证明只需要那条切线", "orange", None)]
y = 64
for s, col, wt in lines:
    if s:
        svg.text(NX, y, s, size=12, color=col, weight=wt)
    y += 18

svg.save("m05-jensen.svg")
