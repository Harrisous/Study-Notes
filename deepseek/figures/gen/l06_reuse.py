"""Figure 6.8: why a multi-step schedule can be reused (bars drawn to scale in units of D).

Three budgets D, 2D, 4D. Cosine: three independent runs, 1 + 2 + 4 = 7 D.
Multi-step 80/10/10: one peak-LR trunk to 0.8 * 4D = 3.2D, branching at 0.8D, 1.6D, 3.2D into decay tails
of 0.2D, 0.4D, 0.8D (each tail = 10 % at 31.6 % LR + 10 % at 10 % LR). Total 4.6 D, 34.3 % less.
Writes deepseek/figures/l06-reuse.svg
"""

from l06_svg import SVG

budgets = [1, 2, 4]
p = 0.8
cos_total = sum(budgets)
ms_total = p * max(budgets) + (1 - p) * sum(budgets)
assert abs(ms_total - 4.6) < 1e-9

W, H = 760, 400
X0 = 110
U = 130            # px per D
BH = 16            # bar height

svg = SVG(W, H, "三个预算 D、2D、4D：cosine 各训一次 vs 多步调度共用第一阶段")

# axis
for k in range(0, 5):
    x = X0 + k * U
    svg.line(x, 30, x, 372, color="grid", width=1)
    svg.text(x, 388, "%dD" % k if k else "0", size=11, anchor="middle", color="muted")

# ---- cosine: three separate runs
svg.text(20, 44, "cosine", size=13, weight="600", color="muted")
svg.text(20, 60, "各训一次", size=12, color="muted")
for i, b in enumerate(budgets):
    y = 40 + i * 26
    svg.rect(X0, y, b * U, BH, fill="muted", fill_opacity=0.22, stroke="muted", width=1)
    svg.text(X0 + b * U + 8, y + 12, "%dD" % b, size=12, color="muted")
svg.text(X0 + 4 * U + 40, 70, "合计 %dD" % cos_total, size=13, weight="600", color="muted")

# ---- multi-step: shared trunk + tails
ty = 190
svg.text(20, ty + 4, "多步", size=13, weight="600", color="blue")
svg.text(20, ty + 20, "80/10/10", size=12, color="blue")
trunk = p * max(budgets)
svg.rect(X0, ty - 8, trunk * U, BH, fill="blue", fill_opacity=0.35, stroke="blue", width=1.2)
svg.text(X0 + 8, ty - 14, "第一阶段（峰值学习率），只训一次：0 → %.1fD" % trunk, size=12, color="blue")

rows = []
for i, b in enumerate(budgets):
    xb = X0 + p * b * U                 # branch point
    y = ty + 44 + i * 40
    t1 = 0.1 * b * U
    lx = xb + 2 * t1 + 8                # tail label starts here, about 122 px wide
    rows.append((b, xb, y, t1, lx))
# fork lines: the 1.6D line would run through the first tail's label, so it is cut around every
# label box it crosses (the course rule: clip geometry in the script, no clipPath, no backing)
for b, xb, y, t1, lx in rows:
    y0 = ty + BH - 8
    for _, _, yr, _, lxr in rows:
        if yr < y and lxr - 3 <= xb <= lxr + 125:
            svg.path("M%.1f %.1f V %.1f" % (xb, y0, yr - 12), stroke="blue", width=1.2, dash="3 2")
            y0 = yr + 8
    svg.path("M%.1f %.1f V %.1f" % (xb, y0, y), stroke="blue", width=1.2, dash="3 2")
    svg.circle(xb, ty, 3.5, fill="blue")
for b, xb, y, t1, lx in rows:
    svg.rect(xb, y - BH / 2, t1, BH, fill="blue", fill_opacity=0.18, stroke="blue", width=1)
    svg.rect(xb + t1, y - BH / 2, t1, BH, fill="blue", fill_opacity=0.08, stroke="blue", width=1)
    svg.text(lx, y + 4, "预算 %dD：尾巴 %.1fD" % (b, 0.2 * b), size=12, color="blue")
    svg.text(xb - 6, y + 4, "%.1fD 处分叉" % (p * b), size=12, anchor="end", color="muted")

svg.text(X0, 360, "尾巴里浅蓝两段：前 10% 用峰值的 31.6%，后 10% 用 10%。", size=12, color="muted")
svg.text(X0 + 4 * U + 40, ty + 4, "合计 %.1fD" % ms_total, size=13, weight="600", color="blue")
svg.text(X0 + 4 * U + 40, ty + 22, "省 %.1f%%" % (100 * (1 - ms_total / cos_total)), size=13, weight="600",
         color="green")

svg.save("l06-reuse.svg")
