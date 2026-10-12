"""Figure M6.5: forward values and adjoints on the computational graph of
y = ln(x1) + x1*x2 - sin(x2) at (x1, x2) = (2, 5).

Redrawn from Baydin et al. (2018), Figure 4 (graph) and Table 3 (primal and adjoint traces), in
the course colours: forward values blue, adjoints (dy/dv) red, local derivatives on the edges.
Every number is computed here and checked against the analytic gradient.
Writes math/figures/m06-backprop-graph.svg
"""

import math

from m06_svg import SVG

x1, x2 = 2.0, 5.0
v = {"x1": x1, "x2": x2}
v["v1"] = math.log(x1)
v["v2"] = x1 * x2
v["v3"] = math.sin(x2)
v["v4"] = v["v1"] + v["v2"]
v["v5"] = v["v4"] - v["v3"]

# local derivatives d(child)/d(parent)
local = {
    ("x1", "v1"): 1 / x1, ("x1", "v2"): x2, ("x2", "v2"): x1, ("x2", "v3"): math.cos(x2),
    ("v1", "v4"): 1.0, ("v2", "v4"): 1.0, ("v4", "v5"): 1.0, ("v3", "v5"): -1.0,
}
order = ["v5", "v4", "v3", "v2", "v1", "x2", "x1"]
adj = {k: 0.0 for k in v}
adj["v5"] = 1.0
for node in order:
    for (p, ch), dl in local.items():
        if p == node:
            adj[node] += adj[ch] * dl
assert abs(adj["x1"] - (1 / x1 + x2)) < 1e-12
assert abs(adj["x2"] - (x1 - math.cos(x2))) < 1e-12
print({k: round(val, 4) for k, val in adj.items()}, {k: round(val, 4) for k, val in v.items()})

W, H = 720, 504
svg = SVG(W, H, "计算图上的前向值（蓝）与伴随 ∂y/∂v（红）")
BW, BH = 124, 66
pos = {"x1": (100, 118), "x2": (100, 318), "v1": (314, 52), "v2": (314, 212), "v3": (314, 372),
       "v4": (488, 132), "v5": (642, 248)}
title = {"x1": "x₁ = v₋₁", "x2": "x₂ = v₀", "v1": "v₁ = ln v₋₁", "v2": "v₂ = v₋₁ × v₀",
         "v3": "v₃ = sin v₀", "v4": "v₄ = v₁ + v₂", "v5": "y = v₅ = v₄ − v₃"}
adj_name = {"x1": "x̄₁", "x2": "x̄₂", "v1": "v̄₁", "v2": "v̄₂", "v3": "v̄₃", "v4": "v̄₄", "v5": "v̄₅"}


def fmt3(a):
    s = "%.3f" % a
    s = s.rstrip("0").rstrip(".") if "." in s else s
    return s.replace("-", "−")


# edges first
for (p, ch), dl in local.items():
    (xa, ya), (xb, yb) = pos[p], pos[ch]
    xa2, xb2 = xa + BW / 2, xb - BW / 2
    svg.arrow(xa2, ya, xb2, yb, color="muted", width=1.4, shorten=2)
    mx, my = (xa2 + xb2) / 2, (ya + yb) / 2
    lab = {("x1", "v1"): "1/v₋₁ = 0.5", ("x1", "v2"): "v₀ = 5", ("x2", "v2"): "v₋₁ = 2",
           ("x2", "v3"): "cos v₀=%.3f" % math.cos(x2), ("v1", "v4"): "1", ("v2", "v4"): "1",
           ("v4", "v5"): "1", ("v3", "v5"): "−1"}[(p, ch)]
    dy = -6 if yb <= ya else 14
    if (p, ch) == ("x1", "v2"):
        dy = -8
    if (p, ch) == ("x2", "v2"):
        dy = 16
    svg.rect(mx - len(lab) * 3.6 - 4, my + dy - 12, len(lab) * 7.2 + 8, 16, fill="surface", rx=3)
    svg.text(mx, my + dy, lab, size=11, anchor="middle", color="ink")

for k, (cx, cy) in pos.items():
    svg.rect(cx - BW / 2, cy - BH / 2, BW, BH, fill="surface", stroke="ink", width=1.2, rx=8)
    svg.text(cx, cy - 12, title[k], size=12, anchor="middle", weight="600")
    svg.text(cx, cy + 6, "= %s" % fmt3(v[k]), size=12, anchor="middle", color="blue")
    svg.text(cx, cy + 24, "%s = %s" % (adj_name[k], fmt3(adj[k])), size=12, anchor="middle", color="red")

# accumulation notes at the inputs
svg.text(100, 118 + BH / 2 + 20, "5 × 1 + 0.5 × 1 = 5.5", size=12, anchor="middle", color="red")
svg.text(100, 118 + BH / 2 + 37, "（两条路径相加）", size=11, anchor="middle", color="red")
svg.text(100, 318 + BH / 2 + 20, "2 × 1 + 0.284 × (−1)", size=12, anchor="middle", color="red")
svg.text(100, 318 + BH / 2 + 37, "= 1.716（两条路径相加）", size=11, anchor="middle", color="red")

# legend
lx, ly = 452, 380
svg.text(lx, ly, "蓝：前向值（从左往右算）", size=12, color="blue")
svg.text(lx, ly + 20, "红：伴随 v̄ = ∂y/∂v（从右往左算，", size=12, color="red")
svg.text(lx, ly + 38, "从 v̄₅ = 1 出发）", size=12, color="red")
svg.text(lx, ly + 58, "箭头上：局部导数 ∂(箭头终点)/∂(起点)", size=12, color="ink")
svg.text(lx, ly + 78, "规则：v̄ = Σ 子节点的伴随 × 局部导数", size=12, color="ink")
svg.text(lx, ly + 98, "核对：∂y/∂x₁ = 1/x₁ + x₂ = 5.5，", size=12, color="muted")
svg.text(lx, ly + 116, "∂y/∂x₂ = x₁ − cos x₂ = 1.716", size=12, color="muted")

svg.save("m06-backprop-graph.svg")
