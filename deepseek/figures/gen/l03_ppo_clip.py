"""Figure 3.7: one term of PPO's clipped surrogate as a function of the ratio rho (redrawn after
Schulman et al. 2017, Figure 1), epsilon = 0.2.

  L(rho) = min(rho*A, clip(rho, 1-eps, 1+eps)*A)
  A > 0 :  L = A*min(rho, 1+eps)      gradient 0 for rho > 1+eps
  A < 0 :  L = A*max(rho, 1-eps)      gradient 0 for rho < 1-eps
Drawn with A = +1 and A = -1. Writes deepseek/figures/l03-ppo-clip.svg
"""
from l03_svg import SVG, Axis, polyline_d, fmt

EPS = 0.2


def L(rho, A):
    c = min(max(rho, 1 - EPS), 1 + EPS)
    return min(rho * A, c * A)


# sanity checks against the worked table in the lesson
assert abs(L(1.5, 1) - 1.2) < 1e-12 and abs(L(0.7, 1) - 0.7) < 1e-12
assert abs(L(0.7, -1) + 0.8) < 1e-12 and abs(L(1.5, -1) + 1.5) < 1e-12

W, H = 730, 372
svg = SVG(W, H, "PPO 裁剪目标的单项：A > 0 与 A < 0 两种情形（ε = 0.2）")


def tw(t, size):
    """Rough text width: CJK glyphs ~1 em, others ~0.6 em (markup characters ignored)."""
    t = t.replace("_{", "").replace("}", "").replace("*", "")
    return sum(size if ord(ch) > 0x2E80 else 0.6 * size for ch in t)


def plate(x, y, t, size, anchor="start", color="ink"):
    """Text on a surface-coloured plate, so the dashed 1±ε guides never run through a note."""
    w = tw(t, size) + 6
    x0 = x - 3 if anchor == "start" else (x - w + 3 if anchor == "end" else x - w / 2)
    svg.rect(x0, y - size, w, size + 5, fill="surface", fill_opacity=0.92, rx=3)
    svg.text(x, y, t, size=size, anchor=anchor, color=color)


def panel(x0, A, title, ylo, yhi, yticks, notes):
    ax = Axis(0, 2, x0 + 44, x0 + 324)
    ay = Axis(ylo, yhi, 290, 58)
    svg.text(x0 + 184, 26, title, size=14, anchor="middle", weight="700", color="green" if A > 0 else "red")
    # zero-gradient region
    if A > 0:
        svg.rect(ax(1 + EPS), ay(yhi), ax(2) - ax(1 + EPS), ay(ylo) - ay(yhi), fill="muted", fill_opacity=0.12)
    else:
        svg.rect(ax(0), ay(yhi), ax(1 - EPS) - ax(0), ay(ylo) - ay(yhi), fill="muted", fill_opacity=0.12)
    for v in (0, 0.5, 1, 1.5, 2):
        svg.line(ax(v), ay(ylo), ax(v), ay(yhi), color="grid", width=1)
        svg.text(ax(v), ay(ylo) + 16, fmt(v), size=11, anchor="middle", color="muted")
    for v in yticks:
        svg.line(ax(0), ay(v), ax(2), ay(v), color="grid", width=1)
        svg.text(ax(0) - 6, ay(v) + 4, fmt(v).replace("-", "−"), size=11, anchor="end", color="muted")
    svg.line(ax(0), ay(ylo), ax(2), ay(ylo), color="muted", width=1.1)
    svg.line(ax(0), ay(ylo), ax(0), ay(yhi), color="muted", width=1.1)
    for v, lab in ((1 - EPS, "1−ε"), (1 + EPS, "1+ε")):
        svg.line(ax(v), ay(ylo), ax(v), ay(yhi), color="muted", width=1, dash="3 3")
        svg.text(ax(v), ay(yhi) - 6, lab, size=11.5, anchor="middle", color="muted")
    svg.text(ax(1), ay(ylo) + 34, "概率比 ρ = π_{θ} / π_{old}", size=12.5, anchor="middle")
    xs = [2 * i / 400 for i in range(401)]
    svg.path(polyline_d([(ax(x), ay(x * A)) for x in xs if ylo <= x * A <= yhi]), stroke="muted", width=1.5, dash="5 4")
    svg.path(polyline_d([(ax(x), ay(L(x, A))) for x in xs]), stroke="blue", width=2.8)
    svg.circle(ax(1), ay(A), 5, fill="red")
    for rho, text, dx, dy, anc, col in notes:
        svg.circle(ax(rho), ay(L(rho, A)), 3.6, fill="ink")
        for k, t in enumerate(text):
            plate(ax(rho) + dx, ay(L(rho, A)) + dy + 16 * k, t, 11.5, anchor=anc, color=col)
    return ax, ay


panel(0, 1, "A > 0（好动作）", 0, 1.4, [0, 0.5, 1],
      [(1.5, ["ρ=1.5：已涨够", "L=1.2，梯度 0"], 0, 22, "middle", "ink"),
       (0.7, ["ρ=0.7：好动作反而变少", "L=0.7，照常纠正"], 6, 22, "start", "ink")])
panel(368, -1, "A < 0（坏动作）", -2, 0, [-2, -1.5, -1, -0.5, 0],
      [(0.7, ["ρ=0.7：已压够", "L=−0.8，梯度 0"], -24, 26, "middle", "ink"),
       (1.5, ["ρ=1.5：坏动作反而变多", "L=−1.5，照常纠正"], -10, 20, "end", "ink")])
svg.line(20, 356, 44, 356, color="blue", width=2.8)
svg.text(50, 360, "裁剪目标 L", size=12)
svg.line(140, 356, 164, 356, color="muted", width=1.5, dash="5 4")
svg.text(170, 360, "未裁剪的 ρA", size=12)
svg.rect(270, 349, 20, 12, fill="muted", fill_opacity=0.12)
svg.text(296, 360, "梯度为 0 的区域", size=12)
svg.circle(420, 355, 5, fill="red")
svg.text(430, 360, "起点 ρ = 1（θ = θ_{old}）", size=12)
svg.save("l03-ppo-clip.svg")
