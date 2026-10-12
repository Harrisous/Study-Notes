"""Figure M9.10: where PPO's clipped objective has zero gradient, drawn on the (rho, A) plane.

One token's term is l(rho, A) = min(rho*A, clip(rho, 1-eps, 1+eps)*A) with eps = 0.2.
Its derivative in rho is 0 when (A > 0 and rho > 1+eps) or (A < 0 and rho < 1-eps), and A otherwise.
The four marked points are the shared notes' N73 cases. Writes math/figures/m09-ppo-regions.svg
"""

from m09_svg import SVG

EPS = 0.2


def ell(rho, A):
    c = min(max(rho, 1 - EPS), 1 + EPS)
    return min(rho * A, c * A)


def grad_zero(rho, A):
    return (A > 0 and rho > 1 + EPS) or (A < 0 and rho < 1 - EPS)


cases = [(1.3, 1.0), (1.3, -1.0), (0.7, 1.0), (0.7, -1.0)]
vals = [ell(r, a) for r, a in cases]
assert [round(v, 6) for v in vals] == [1.2, -1.3, 0.7, -0.8]

W, H = 720, 400
svg = SVG(W, H, "PPO 裁剪目标：在 (ρ, A) 平面上哪里梯度为零")
X0, X1, Y0, Y1 = 70, 470, 36, 336
RMIN, RMAX, AMIN, AMAX = 0.4, 1.6, -1.25, 1.25


def px(r):
    return X0 + (r - RMIN) / (RMAX - RMIN) * (X1 - X0)


def py(a):
    return Y1 - (a - AMIN) / (AMAX - AMIN) * (Y1 - Y0)


# zero-gradient regions
svg.rect(px(1 + EPS), py(AMAX), px(RMAX) - px(1 + EPS), py(0) - py(AMAX), fill="red", fill_opacity=0.10)
svg.rect(px(RMIN), py(0), px(1 - EPS) - px(RMIN), py(AMIN) - py(0), fill="red", fill_opacity=0.10)
# grid and axes
for r in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6):
    x = px(r)
    col, w, dash = ("muted", 1.3, "5 4") if r in (0.8, 1.2) else ("grid", 1, None)
    if r == 1.0:
        col, w, dash = "ink", 1.2, "2 3"
    svg.line(x, Y0, x, Y1, color=col, width=w, dash=dash)
    svg.text(x, Y1 + 18, "%.1f" % r, size=11, anchor="middle", color="muted")
for a in (-1, -0.5, 0, 0.5, 1):
    y = py(a)
    svg.line(X0, y, X1, y, color="grid" if a != 0 else "ink", width=1 if a != 0 else 1.3)
    svg.text(X0 - 8, y + 4, ("%+.1f" % a).replace("+0.0", "0").replace("-", "−"), size=11, anchor="end", color="muted")
svg.rect(X0, Y0, X1 - X0, Y1 - Y0, stroke="muted", width=1.1)
svg.text((X0 + X1) / 2, Y1 + 40, "概率比 ρ = π_{θ} / π_{θold}", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "优势 A", size=12, anchor="middle", color="muted", rotate=-90)
svg.text(px(0.8), Y0 - 8, "1 − ε", size=12, anchor="middle", color="muted")
svg.text(px(1.2), Y0 - 8, "1 + ε", size=12, anchor="middle", color="muted")
svg.text(px(1.0), Y0 - 8, "起点 ρ = 1", size=12, anchor="middle")

# arrows: direction in which gradient ascent pushes rho (sign of A), in the live regions
for a in (0.55, -0.55):
    y = py(a)
    if a > 0:
        svg.arrow(px(0.86), y, px(1.16), y, color="green", width=2)
        svg.arrow(px(0.46), y, px(0.74), y, color="green", width=2)
    else:
        svg.arrow(px(1.14), y, px(0.84), y, color="green", width=2)
        svg.arrow(px(1.54), y, px(1.26), y, color="green", width=2)
# the two middle labels sit across the dotted rho = 1 line: give them a surface-coloured backing
for lab, yy in (("好动作：推高概率", py(0.55) - 10), ("坏动作：压低概率", py(-0.55) + 22)):
    svg.rect(px(1.0) - 52, yy - 13, 104, 18, fill="surface", rx=3)
    svg.text(px(1.0), yy, lab, size=12, anchor="middle", color="green")
svg.text(px(0.6), py(0.55) - 10, "反而降了：照常拉回", size=12, anchor="middle", color="green")
svg.text(px(1.4), py(-0.55) + 22, "反而涨了：照常压回", size=12, anchor="middle", color="green")
svg.text(px(1.4), py(0.25), "已涨够：梯度 0", size=12.5, anchor="middle", color="red", weight="600")
svg.text(px(0.6), py(-0.25) + 4, "已压够：梯度 0", size=12.5, anchor="middle", color="red", weight="600")

# the four cases
for (r, a), v in zip(cases, vals):
    z = grad_zero(r, a)
    svg.circle(px(r), py(a), 5.5, fill="red" if z else "blue")
    lab = "ℓ = %s，%s" % (("%.1f" % v).replace("-", "−"), "梯度 0" if z else "梯度 = A")
    dy = -12 if a > 0 else 22
    # keep each label inside its own column, clear of the dashed 1 ± ε lines
    if r < 1:
        svg.text(px(1 - EPS) - 6, py(a) + dy, lab, size=12, anchor="end", color="red" if z else "blue", weight="600")
    else:
        svg.text(px(1 + EPS) + 6, py(a) + dy, lab, size=12, color="red" if z else "blue", weight="600")

# right-hand explanation
LX = 492
lines = [
    ("单个 token 的目标", "600", "ink"),
    ("ℓ(ρ, A) = min(ρA, clip(ρ, 1−ε, 1+ε)·A)", None, "ink"),
    ("ε = 0.2", None, "muted"),
    ("", None, "ink"),
    ("红色区域：ℓ 对 ρ 是常数，", None, "red"),
    ("这个 token 不再提供梯度。", None, "red"),
    ("", None, "ink"),
    ("白色区域：∂ℓ/∂ρ = A，", None, "ink"),
    ("和未裁剪的 ρA 一样。", None, "ink"),
    ("", None, "ink"),
    ("绿色箭头：梯度上升把 ρ", None, "green"),
    ("往哪边推。只在“往好的", None, "green"),
    ("方向已经走够”时停下；", None, "green"),
    ("往坏的方向越界照常惩罚。", None, "green"),
    ("", None, "ink"),
    ("四个点：正文例题的四种情形", None, "muted"),
]
y = Y0 + 6
for text, w, col in lines:
    if text:
        svg.text(LX, y, text, size=12, weight=w, color=col)
    y += 19 if text else 9

svg.save("m09-ppo-regions.svg")
