"""Figure 8.5: the PPO clipped surrogate for one token, as a function of the ratio rho.

L(rho) = min(rho * A, clip(rho, 1 - eps, 1 + eps) * A), eps = 0.2 (the example value in the
PPO paper). Left A = +1, right A = -1. Flat pieces have zero gradient. The point rho = 1 is
where DeepSeekMath's single update per exploration stage sits (research notes 2.5(b)).
Sample values (research notes 4.10) are asserted below. Writes deepseek/figures/l08-clip.svg
"""

from l08_svg import SVG, polyline

eps = 0.2


def L(rho, A):
    cl = min(max(rho, 1 - eps), 1 + eps)
    return min(rho * A, cl * A)


assert [round(L(x, 1), 3) for x in (0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)] == [0, 0.5, 0.8, 1.0, 1.2, 1.2, 1.2]
assert [round(L(x, -1), 3) for x in (0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)] == [-0.8, -0.8, -0.8, -1.0, -1.2, -1.5, -2.0]

W, H = 740, 352
svg = SVG(W, H, "PPO 裁剪目标随比值 ρ 的变化（ε = 0.2）")
N = 401
xs = [2.0 * k / (N - 1) for k in range(N)]


def panel(x0, A, title):
    PX0, PX1 = x0 + 44, x0 + 330
    PY0, PY1 = 46, 262
    ylo, yhi = (-0.2, 2.2) if A > 0 else (-2.2, 0.2)

    def px(v):
        return PX0 + v / 2.0 * (PX1 - PX0)

    def py(v):
        return PY1 - (v - ylo) / (yhi - ylo) * (PY1 - PY0)

    # trust region band
    svg.rect(px(1 - eps), PY0, px(1 + eps) - px(1 - eps), PY1 - PY0, fill="blue", fill_opacity=0.07)
    for v in (0, 0.5, 1.0, 1.5, 2.0):
        svg.line(px(v), PY0, px(v), PY1, color="grid", width=1)
        svg.text(px(v), PY1 + 16, ("%g" % v), size=11, anchor="middle", color="muted")
    ticks = (0, 0.5, 1.0, 1.5, 2.0) if A > 0 else (0, -0.5, -1.0, -1.5, -2.0)
    for v in ticks:
        svg.line(PX0, py(v), PX1, py(v), color="grid", width=1)
        svg.text(PX0 - 6, py(v) + 4, ("%g" % v).replace("-", "−"), size=11, anchor="end", color="muted")
    svg.line(PX0, py(0), PX1, py(0), color="muted", width=1.1)
    svg.line(PX0, PY0, PX0, PY1, color="muted", width=1.1)
    # unclipped rho*A
    polyline(svg, [(px(x), py(x * A)) for x in xs], color="muted", width=1.5, dash="5 4")
    # clipped objective
    polyline(svg, [(px(x), py(L(x, A))) for x in xs], color="ink", width=2.6)
    # flat (zero-gradient) piece in red
    flat = [x for x in xs if (A > 0 and x > 1 + eps) or (A < 0 and x < 1 - eps)]
    polyline(svg, [(px(x), py(L(x, A))) for x in flat], color="red", width=3.2)
    # rho = 1 point
    svg.circle(px(1.0), py(A), 5, fill="blue")
    svg.text(x0 + 187, PY0 - 22, title, size=13, anchor="middle", color="ink", weight="600")
    svg.text((PX0 + PX1) / 2, PY1 + 36, "比值 ρ = π_{θ} / π_{θold}", size=12, anchor="middle", color="muted")
    return px, py


px, py = panel(10, 1.0, "A > 0（好 token）：取 A = +1")
svg.text(px(1.26), py(1.2) - 34, "ρ > 1.2 后变平：", size=12, color="red", halo=True)
svg.text(px(1.26), py(1.2) - 15, "再推也不再加分，梯度为 0", size=12, color="red", halo=True)
svg.text(px(1.0) + 8, py(1.0) + 22, "ρ = 1，梯度 = A∇log π", size=12, color="blue")
svg.text(px(0.62), py(0.34), "虚线：不裁剪的 ρA", size=12, color="muted")

px, py = panel(380, -1.0, "A < 0（坏 token）：取 A = −1")
svg.text(px(0.03), py(-0.52), "ρ < 0.8 后变平：", size=12, color="red", halo=True)
svg.text(px(0.03), py(-0.52) + 17, "再压也不再加分", size=12, color="red", halo=True)
svg.text(px(1.0) + 8, py(-1.0) - 12, "ρ = 1", size=12, color="blue")
# the note about the unclipped side sits above the rho > 1.2 segment, with an arrow down to it
svg.text(px(1.97), py(-0.22), "ρ > 1.2 的一侧不裁剪：", size=12, anchor="end", color="ink", halo=True)
svg.text(px(1.97), py(-0.22) + 17, "坏 token 被抬高，要全额受罚", size=12, anchor="end", color="ink", halo=True)
svg.arrow(px(1.72), py(-0.22) + 24, px(1.72), py(-1.72), color="ink", width=1.3, shorten=3)

svg.text(16, H - 10, "浅蓝带是信任区间 [1 − ε, 1 + ε] = [0.8, 1.2]。取 min 让目标永远不高于虚线：它是未裁剪目标的“悲观下界”。",
         size=12, color="muted")
svg.save("l08-clip.svg")
