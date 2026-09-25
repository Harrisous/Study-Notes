"""Figure M6.1: a smooth curve looks like its tangent line once you zoom in.

f(z) = sigmoid(z) around z0 = 1. Three panels with half-widths 2, 0.2, 0.02. In each panel the
vertical axis spans the same multiple of the half-width, so the curve and the tangent are drawn
at the same visual scale; the gap between them (the error of the linear approximation) shrinks
relative to the window. The maximum error in each window is computed here and printed on the
panel. Writes math/figures/m06-tangent-zoom.svg
"""

import math

from m06_svg import SVG, fmt, sci

z0 = 1.0
sig = lambda z: 1.0 / (1.0 + math.exp(-z))
f0 = sig(z0)
d0 = f0 * (1 - f0)                       # sigma'(1) = 0.196612
tan = lambda z: f0 + d0 * (z - z0)

W, H = 720, 372
svg = SVG(W, H, "放大之后，sigmoid 曲线与它在 z = 1 处的切线越来越分不开")
panels = [2.0, 0.2, 0.02]
PW, PH = 196, 196
gap = 42
x_start = 40
top = 62

svg.text(W / 2, 22, "f(z) = σ(z)，在 z₀ = 1 处：f(z₀) = %.4f，f′(z₀) = σ(1)(1 − σ(1)) = %.4f" % (f0, d0),
         size=13, anchor="middle", color="ink")
svg.text(W / 2, 42, "三个窗口依次放大 10 倍（横轴、纵轴一起放大）", size=12, anchor="middle", color="muted")

for k, hw in enumerate(panels):
    X0 = x_start + k * (PW + gap)
    X1 = X0 + PW
    Y0, Y1 = top, top + PH
    # vertical window: centred on f0, same aspect as the horizontal (height = 2*hw*0.5 of f units)
    vh = hw * 0.5                        # f-range half height; slope 0.197 fits comfortably
    px = lambda z: X0 + (z - (z0 - hw)) / (2 * hw) * PW
    py = lambda v: Y1 - (v - (f0 - vh)) / (2 * vh) * PH
    svg.rect(X0, Y0, PW, PH, fill="surface", stroke="rule", width=1)
    # grid
    for t in (-0.5, 0, 0.5):
        svg.line(px(z0 + t * hw), Y0, px(z0 + t * hw), Y1, color="grid", width=1)
        svg.line(X0, py(f0 + t * vh), X1, py(f0 + t * vh), color="grid", width=1)
    # sample curve, clip to the panel by construction (skip points outside)
    n = 160
    zs = [z0 - hw + 2 * hw * i / n for i in range(n + 1)]
    pts = [(px(z), py(sig(z))) for z in zs if f0 - vh <= sig(z) <= f0 + vh]
    # error band between curve and tangent
    band = []
    for z in zs:
        a, b = sig(z), tan(z)
        if f0 - vh <= a <= f0 + vh and f0 - vh <= b <= f0 + vh:
            band.append((z, a, b))
    poly = [(px(z), py(a)) for z, a, b in band] + [(px(z), py(b)) for z, a, b in reversed(band)]
    svg.polygon(poly, fill="red", fill_opacity=0.16)
    # tangent (clip analytically to the panel)
    zl = max(z0 - hw, z0 + (-vh) / d0)
    zr = min(z0 + hw, z0 + vh / d0)
    svg.line(px(zl), py(tan(zl)), px(zr), py(tan(zr)), color="blue", width=2, dash="6 4")
    svg.path("M" + " L".join("%.1f %.1f" % p for p in pts), stroke="ink", width=2.2)
    svg.circle(px(z0), py(f0), 4, fill="blue")
    # ticks at the window edges
    svg.text(X0, Y1 + 16, fmt(z0 - hw) if hw >= 0.2 else "%.2f" % (z0 - hw), size=11, anchor="start", color="muted")
    svg.text(X1, Y1 + 16, fmt(z0 + hw) if hw >= 0.2 else "%.2f" % (z0 + hw), size=11, anchor="end", color="muted")
    svg.text((X0 + X1) / 2, Y1 + 16, "z", size=11, anchor="middle", color="muted")
    err = max(abs(sig(z) - tan(z)) for z in zs)
    svg.text((X0 + X1) / 2, Y1 + 38, "窗口 z₀ ± %s" % (fmt(hw) if hw >= 0.2 else "0.02"), size=13, anchor="middle", weight="600")
    svg.text((X0 + X1) / 2, Y1 + 57, "最大误差 %s" % sci(err), size=12, anchor="middle", color="red")
    svg.text((X0 + X1) / 2, Y1 + 74, "= 窗口宽度的 %s%%" % ("%.2g" % (100 * err / (2 * hw))), size=12, anchor="middle", color="red")
    print(hw, err)
    if k < 2:
        # zoom arrow between panels
        ax = X1 + 6
        svg.arrow(ax, (Y0 + Y1) / 2, ax + gap - 12, (Y0 + Y1) / 2, color="muted", width=1.4)
        svg.text(ax + (gap - 6) / 2, (Y0 + Y1) / 2 - 8, "×10", size=11, anchor="middle", color="muted")

# legend
lx = 40
svg.line(lx, H - 14, lx + 26, H - 14, color="ink", width=2.2)
svg.text(lx + 32, H - 10, "曲线 σ(z)", size=12)
svg.line(lx + 120, H - 14, lx + 146, H - 14, color="blue", width=2, dash="6 4")
svg.text(lx + 152, H - 10, "切线 f(z₀) + f′(z₀)(z − z₀)", size=12, color="blue")
svg.rect(lx + 362, H - 21, 22, 12, fill="red", fill_opacity=0.16)
svg.text(lx + 390, H - 10, "两者之差 = 线性近似的误差", size=12, color="red")

svg.save("m06-tangent-zoom.svg")
