"""Figure 7.5: expert capacity and token dropping under three capacity factors.

1024 tokens, 8 experts, top-1 loads (200, 150, 130, 120, 120, 110, 100, 94) (research notes R13).
capacity = tokens / experts * CF (Switch Transformer eq. 3): 128, 160, 256.
Tokens beyond capacity are dropped (they skip the layer through the residual connection);
slots below capacity are padding (wasted compute). All geometry computed here.
Writes deepseek/figures/l07-capacity.svg
"""

from l07_svg import SVG

loads = [200, 150, 130, 120, 120, 110, 100, 94]
T, E = 1024, 8
assert sum(loads) == T
cfs = [1.0, 1.25, 2.0]

W, H = 740, 350
svg = SVG(W, H, "容量因子与 token 丢弃：同一批 1024 个 token 在三种容量因子下的命运")

PW = 232
Y_BASE = 250
YS = 170 / 260.0           # px per token
BW, BG = 18, 6
for k, cf in enumerate(cfs):
    cap = T / E * cf
    x0 = 20 + k * (PW + 12)
    svg.text(x0, 24, "容量因子 %s：每个专家 %d 个槽" % ({1.0: "1.0", 1.25: "1.25", 2.0: "2.0"}[cf], cap), size=13, weight="600")
    # axis
    svg.line(x0 + 26, Y_BASE, x0 + 26 + E * (BW + BG), Y_BASE, color="muted", width=1)
    if k == 0:
        for t in (0, 100, 200):
            y = Y_BASE - t * YS
            svg.text(x0 + 20, y + 4, str(t), size=11, anchor="end", color="muted")
            svg.line(x0 + 24, y, x0 + 26, y, color="muted", width=1)
    dropped = empty = 0
    for i, ld in enumerate(loads):
        bx = x0 + 30 + i * (BW + BG)
        kept = min(ld, cap)
        over = max(0, ld - cap)
        pad = max(0, cap - ld)
        dropped += over
        empty += pad
        svg.rect(bx, Y_BASE - kept * YS, BW, kept * YS, fill="blue", fill_opacity=0.75)
        if over:
            svg.rect(bx, Y_BASE - ld * YS, BW, over * YS, fill="red", fill_opacity=0.75)
        if pad:
            svg.rect(bx, Y_BASE - cap * YS, BW, pad * YS, fill="muted", fill_opacity=0.08, stroke="muted", width=0.8,
                     dash="2 2")
        svg.text(bx + BW / 2, Y_BASE + 14, str(i + 1), size=11, anchor="middle", color="muted")
    yc = Y_BASE - cap * YS
    svg.line(x0 + 26, yc, x0 + 26 + E * (BW + BG), yc, color="ink", width=1.3, dash="5 3")
    svg.text(x0 + 26 + E * (BW + BG), yc - 5, "容量 %d" % cap, size=11, anchor="end", color="ink")
    svg.text(x0 + 30, Y_BASE + 36, "丢弃 %d 个 token（%.1f%%）" % (dropped, 100 * dropped / T), size=12,
             color="red" if dropped else "ink", weight="600")
    svg.text(x0 + 30, Y_BASE + 54, "空槽 %d 个（算了也白算）" % empty, size=12, color="muted")
svg.text(20, Y_BASE - 200 * YS - 16, "token 数", size=11, color="muted")

# legend
ly = H - 16
items = [("blue", 0.75, None, "被专家处理"), ("red", 0.75, None, "超出容量被丢弃，只走残差"),
         ("muted", 0.08, "2 2", "空槽（按容量预留却没用上）")]
x = 20
for col, op, dash, lab in items:
    svg.rect(x, ly - 11, 14, 14, fill=col, fill_opacity=op, stroke="muted" if dash else None, width=0.8, dash=dash)
    svg.text(x + 20, ly, lab, size=12, color="ink")
    x += 20 + len(lab) * 12 + 26
svg.text(W - 20, ly, "专家编号 1…8，负载和 = 1024", size=11, anchor="end", color="muted")

svg.save("l07-capacity.svg")
