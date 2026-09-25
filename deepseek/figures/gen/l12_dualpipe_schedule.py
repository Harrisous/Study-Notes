"""Figure 12.13: a DualPipe schedule, 8 PP ranks and 20 micro-batches in two directions
(redrawn after DeepSeek-V3 report Figure 5).

The per-rank operation sequences below are the event-driven simulation in the lesson-12 research
notes (section 5.2, produced by following the 8 phases of DualPipe's open-source step(); T_F = 1,
full backward T_B = 2, input-only backward b = 1, weight backward W = 1, an overlapped
forward+backward pair F&B = 3, communication ignored). Tokens:
    F0.3        forward of micro-batch 3 of direction 0 (1 unit)
    B1.2        full backward of micro-batch 2 of direction 1 (2 units)
    b0.6        backward for input only (1 unit)          W  backward for weights (1 unit)
    F0.7&B1.3   a forward chunk and a backward chunk overlapped (3 units)
    idleN       N idle units
This script re-checks the data: every rank is 66 units long, every rank idles exactly
(P/2 - 1)(T_F&B + T_B - 3 T_W) = 3 * 2 = 6 units, and every forward starts after the same
micro-batch's forward finished on the previous stage (direction 0 flows rank 0 -> 7,
direction 1 flows rank 7 -> 0).
Writes deepseek/figures/l12-dualpipe-schedule.svg
"""


from l12_svg import SVG

RAW = """
rank0: F0.0 F0.1 F0.2 F0.3 F0.4 F0.5 F0.6 F1.0 b1.0 W F1.1 b1.1 W F1.2 b1.2 W F1.3 F0.7&B1.3 idle3 F1.4&B0.0 F0.8&B1.4 F1.5&B0.1 F0.9&B1.5 F1.6&B0.2 B1.6 idle1 F1.7&B0.3 B1.7 idle1 F1.8&B0.4 B1.8 idle1 F1.9&B0.5 B1.9 b0.6 W b0.7 W b0.8 W b0.9 W
rank1: idle1 F0.0 F0.1 F0.2 F0.3 F0.4 F1.0 F0.5 F1.1 b1.0 W F1.2 b1.1 W F1.3 F0.6&B1.2 idle2 F1.4&B0.0 F0.7&B1.3 F1.5&B0.1 F0.8&B1.4 F1.6&B0.2 F0.9&B1.5 F1.7&B0.3 B1.6 idle1 F1.8&B0.4 B1.7 idle1 F1.9&B0.5 B1.8 B0.6 idle1 b1.9 b0.7 W b0.8 W b0.9 W W
rank2: idle2 F0.0 F0.1 F0.2 F1.0 F0.3 F1.1 F0.4 F1.2 b1.0 W F1.3 F0.5&B1.1 idle1 F1.4&B0.0 F0.6&B1.2 F1.5&B0.1 F0.7&B1.3 F1.6&B0.2 F0.8&B1.4 F1.7&B0.3 F0.9&B1.5 F1.8&B0.4 B1.6 idle1 F1.9&B0.5 B1.7 B0.6 idle1 B1.8 b0.7 idle1 b1.9 b0.8 W b0.9 W W W
rank3: idle3 F0.0 F1.0 F0.1 F1.1 F0.2 F1.2 F0.3 F1.3 F0.4 B1.0 F1.4&B0.0 F0.5&B1.1 F1.5&B0.1 F0.6&B1.2 F1.6&B0.2 F0.7&B1.3 F1.7&B0.3 F0.8&B1.4 F1.8&B0.4 F0.9&B1.5 F1.9&B0.5 B1.6 B0.6 idle1 B1.7 B0.7 idle1 b1.8 b0.8 idle1 b1.9 b0.9 W W W W
rank4: idle3 F1.0 F0.0 F1.1 F0.1 F1.2 F0.2 F1.3 F0.3 F1.4 B0.0 F0.4&B1.0 F1.5&B0.1 F0.5&B1.1 F1.6&B0.2 F0.6&B1.2 F1.7&B0.3 F0.7&B1.3 F1.8&B0.4 F0.8&B1.4 F1.9&B0.5 F0.9&B1.5 B0.6 B1.6 idle1 B0.7 B1.7 idle1 b0.8 b1.8 idle1 b0.9 b1.9 W W W W
rank5: idle2 F1.0 F1.1 F1.2 F0.0 F1.3 F0.1 F1.4 F0.2 b0.0 W F0.3 F1.5&B0.1 idle1 F0.4&B1.0 F1.6&B0.2 F0.5&B1.1 F1.7&B0.3 F0.6&B1.2 F1.8&B0.4 F0.7&B1.3 F1.9&B0.5 F0.8&B1.4 B0.6 idle1 F0.9&B1.5 B0.7 B1.6 idle1 B0.8 b1.7 idle1 b0.9 b1.8 W b1.9 W W W
rank6: idle1 F1.0 F1.1 F1.2 F1.3 F1.4 F0.0 F1.5 F0.1 b0.0 W F0.2 b0.1 W F0.3 F1.6&B0.2 idle2 F0.4&B1.0 F1.7&B0.3 F0.5&B1.1 F1.8&B0.4 F0.6&B1.2 F1.9&B0.5 F0.7&B1.3 B0.6 idle1 F0.8&B1.4 B0.7 idle1 F0.9&B1.5 B0.8 B1.6 idle1 b0.9 b1.7 W b1.8 W b1.9 W W
rank7: F1.0 F1.1 F1.2 F1.3 F1.4 F1.5 F1.6 F0.0 b0.0 W F0.1 b0.1 W F0.2 b0.2 W F0.3 F1.7&B0.3 idle3 F0.4&B1.0 F1.8&B0.4 F0.5&B1.1 F1.9&B0.5 F0.6&B1.2 B0.6 idle1 F0.7&B1.3 B0.7 idle1 F0.8&B1.4 B0.8 idle1 F0.9&B1.5 B0.9 b1.6 W b1.7 W b1.8 W b1.9 W
"""

P = 8
DUR = {"F": 1, "B": 2, "b": 1, "W": 1, "FB": 3}


def parse(tok):
    if tok.startswith("idle"):
        return ("idle", None, None, int(tok[4:]))
    if "&" in tok:
        f, b = tok.split("&")
        return ("FB", (int(f[1]), int(f[3:])), (int(b[1]), int(b[3:])), 3)
    if tok == "W":
        return ("W", None, None, 1)
    return (tok[0], (int(tok[1]), int(tok[3:])), None, DUR[tok[0]])


ranks = []
for line in RAW.strip().splitlines():
    toks = line.split(":")[1].split()
    t = 0
    cells = []
    for tok in toks:
        kind, a, b, d = parse(tok)
        cells.append((kind, a, b, t, t + d))
        t += d
    ranks.append(cells)

for r, cells in enumerate(ranks):
    assert cells[-1][4] == 66, (r, cells[-1][4])
    idle = sum(e - s for k, *_, s, e in cells if k == "idle")
    assert idle == (P // 2 - 1) * (3 + 2 - 3 * 1) == 6, (r, idle)

# forward dependency check
fend = {}
for r, cells in enumerate(ranks):
    for kind, a, b, s, e in cells:
        if kind in ("F", "FB"):
            fend[(r, a)] = (s, e)
viol = 0
for (r, (d, i)), (s, e) in fend.items():
    prev = r - 1 if d == 0 else r + 1
    if 0 <= prev < P:
        if s < fend[(prev, (d, i))][1]:
            viol += 1
assert viol == 0, viol
print("DualPipe schedule OK: 8 ranks x 66 units, idle 6 each, forward order consistent")

W, H = 760, 372
X0, U, RH = 64, 10.1, 28
Y0 = 40
svg = SVG(W, H, "DualPipe 调度：8 个 rank，两个方向各 10 个 micro-batch")
svg.text(X0, 22, "方向 0 从 rank 0 进入（深色），方向 1 从 rank 7 进入（浅色）；数字是 micro-batch 编号", size=12, color="muted")
for r, cells in enumerate(ranks):
    y = Y0 + r * RH
    svg.text(X0 - 8, y + 17, "rank %d" % r, size=12, anchor="end")
    for kind, a, b, s, e in cells:
        x, w = X0 + s * U, (e - s) * U
        if kind == "idle":
            svg.rect(x, y, w, RH - 4, fill="red", fill_opacity=0.10)
            continue
        if kind == "W":
            svg.rect(x + 0.5, y, w - 1, RH - 4, fill="violet", fill_opacity=0.28, stroke="violet", width=0.7)
            continue
        if kind == "FB":
            (fd, fi), (bd, bi) = a, b
            h2 = (RH - 4) / 2
            svg.rect(x + 0.5, y, w - 1, h2, fill="blue", fill_opacity=0.34 if fd == 0 else 0.13)
            svg.rect(x + 0.5, y + h2, w - 1, h2, fill="green", fill_opacity=0.34 if bd == 0 else 0.13)
            svg.rect(x + 0.5, y, w - 1, RH - 4, stroke="ink", width=1.6)
            svg.text(x + w / 2, y + h2 - 1.5, str(fi), size=11, anchor="middle", color="blue")
            svg.text(x + w / 2, y + 2 * h2 - 1.5, str(bi), size=11, anchor="middle", color="green")
            continue
        d, i = a
        col = "blue" if kind == "F" else "green"
        op = 0.34 if d == 0 else 0.13
        svg.rect(x + 0.5, y, w - 1, RH - 4, fill=col, fill_opacity=op, stroke=col, width=0.7,
                 dash="2 2" if kind == "b" else None)
        svg.text(x + w / 2, y + 16, str(i), size=11, anchor="middle", color=col)
# time axis
ay = Y0 + P * RH + 6
for t in range(0, 67, 6):
    svg.line(X0 + t * U, ay, X0 + t * U, ay + 5, color="muted", width=1)
    svg.text(X0 + t * U, ay + 18, str(t), size=11, anchor="middle", color="muted")
svg.text(X0 + 66 * U, ay + 34, "时间（单位 T_{F}）", size=11, anchor="end", color="muted")
# legend
ly = ay + 52
leg = [("blue", 0.34, "前向"), ("green", 0.34, "完整反向（2 格）"), ("green", 0.2, "只对输入的反向（虚线框，1 格）"),
       ("violet", 0.28, "对权重的反向 W")]
x = 16
for col, op, lab in leg:
    svg.rect(x, ly - 10, 14, 13, fill=col, fill_opacity=op, stroke=col, width=0.8, dash="2 2" if "虚线" in lab else None)
    svg.text(x + 19, ly + 1, lab, size=12)
    x += 34 + 12.2 * len(lab)
svg.rect(16, ly + 10, 28, 14, stroke="ink", width=1.6)
svg.text(50, ly + 21, "黑框 = 一个前向 chunk（上）与一个反向 chunk（下）互相重叠，3 格；红底 = 气泡，每个 rank 恰好 6 格", size=12)
svg.save("l12-dualpipe-schedule.svg")
