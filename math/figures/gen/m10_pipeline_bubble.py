"""Figure M10.8: where the pipeline bubble comes from, and how big it is.

Left: a GPipe-style schedule (all forwards, then all backwards) for P = 4 stages and m = 4
micro-batches with T_F = 1 and T_B = 2, simulated from the dependencies:
  forward j on stage p waits for forward j on stage p-1 and for the stage's previous task;
  backward j on stage p waits for backward j on stage p+1 (or its own forward j on the last stage).
Right: the bubble fraction (P-1)/(m+P-1) against m for P = 4, 8, 16 (log x axis), with
GPipe's rule of thumb m >= 4P and the 10.3% bubble measured for DeepSeek-V3's DualPipe
(P = 16; Insights table 4: 2.06 s of a 19.926 s step).
Writes math/figures/m10-pipeline-bubble.svg
"""

import math

from m10_svg import SVG

P, M, TF, TB = 4, 4, 1, 2

# ---- simulate
fwd_end = [[0.0] * M for _ in range(P)]
fwd_start = [[0.0] * M for _ in range(P)]
free = [0.0] * P
for j in range(M):
    for p in range(P):
        st = max(free[p], fwd_end[p - 1][j] if p else 0.0)
        fwd_start[p][j] = st
        fwd_end[p][j] = st + TF
        free[p] = st + TF
bwd_start = [[0.0] * M for _ in range(P)]
bwd_end = [[0.0] * M for _ in range(P)]
for j in range(M):
    for p in reversed(range(P)):
        dep = fwd_end[p][j] if p == P - 1 else bwd_end[p + 1][j]
        st = max(free[p], dep)
        bwd_start[p][j] = st
        bwd_end[p][j] = st + TB
        free[p] = st + TB
total = max(free)
busy = M * (TF + TB)
idle = [total - busy for _ in range(P)]
assert abs(total - (M + P - 1) * (TF + TB)) < 1e-9
assert all(abs(i - (P - 1) * (TF + TB)) < 1e-9 for i in idle)

W, H = 740, 330
svg = SVG(W, H, "流水线气泡：P = 4、m = 4 的 GPipe 调度，以及气泡占比随 m 的变化")
GX0, GY0 = 78, 58
U = 17.0                     # pixels per time unit
RH = 30
svg.text(20, 26, "P = 4 段、m = 4 个 micro-batch（前向 1、反向 2 个时间单位）", size=13, weight="600")
for p in range(P):
    y = GY0 + p * (RH + 8)
    svg.text(GX0 - 10, y + RH / 2 + 5, "第 %d 段" % p, size=12, anchor="end")
    svg.rect(GX0, y, total * U, RH, fill="red", fill_opacity=0.10)
    for j in range(M):
        x = GX0 + fwd_start[p][j] * U
        svg.rect(x, y, TF * U, RH, fill="blue", fill_opacity=0.75, stroke="surface", width=1)
        svg.text(x + TF * U / 2, y + RH / 2 + 5, "%d" % j, size=12, anchor="middle", color="surface", weight="600")
        x = GX0 + bwd_start[p][j] * U
        svg.rect(x, y, TB * U, RH, fill="orange", fill_opacity=0.75, stroke="surface", width=1)
        svg.text(x + TB * U / 2, y + RH / 2 + 5, "%d" % j, size=12, anchor="middle", color="surface", weight="600")
# time axis
ay = GY0 + P * (RH + 8) + 4
svg.line(GX0, ay, GX0 + total * U, ay, color="muted", width=1)
for t in range(0, int(total) + 1, 3):
    svg.line(GX0 + t * U, ay, GX0 + t * U, ay + 4, color="muted", width=1)
    svg.text(GX0 + t * U, ay + 17, str(t), size=11, anchor="middle", color="muted")
svg.text(GX0 + total * U / 2, ay + 34, "时间（单位：一次前向）", size=12, anchor="middle", color="muted")
ly = ay + 56
svg.rect(GX0, ly - 11, 14, 14, fill="blue", fill_opacity=0.75)
svg.text(GX0 + 20, ly, "前向", size=12)
svg.rect(GX0 + 66, ly - 11, 14, 14, fill="orange", fill_opacity=0.75)
svg.text(GX0 + 86, ly, "反向", size=12)
svg.rect(GX0 + 132, ly - 11, 14, 14, fill="red", fill_opacity=0.15)
svg.text(GX0 + 152, ly, "空等（气泡）", size=12)
svg.text(GX0, ly + 24, "每段忙 %d、空 %d，总长 %d：空闲占 %d/%d = %.1f%%"
         % (busy, idle[0], total, idle[0], total, 100 * idle[0] / total), size=12, weight="600")

# ---- right: bubble fraction vs m
RX0, RX1, RY0, RY1 = 470, 700, 50, 246
LM0, LM1 = 0, math.log10(512)


def rx(m):
    return RX0 + (math.log10(m) - LM0) / (LM1 - LM0) * (RX1 - RX0)


def ry(f):
    return RY1 - f * (RY1 - RY0)


svg.text(RX0 - 30, 26, "气泡占比 (P−1)/(m+P−1)", size=13, weight="600")
for f in (0, 0.25, 0.5, 0.75, 1.0):
    svg.line(RX0, ry(f), RX1, ry(f), color="grid", width=1)
    svg.text(RX0 - 6, ry(f) + 4, "%d%%" % round(100 * f), size=11, anchor="end", color="muted")
for m in (1, 4, 16, 64, 256):
    svg.line(rx(m), RY0, rx(m), RY1, color="grid", width=1)
    svg.text(rx(m), RY1 + 16, str(m), size=11, anchor="middle", color="muted")
svg.text((RX0 + RX1) / 2, RY1 + 34, "micro-batch 数 m（对数刻度）", size=12, anchor="middle", color="muted")
cols = {4: "ink", 8: "violet", 16: "blue"}
for Pp, col in cols.items():
    pts = []
    k = 0
    while True:
        m = 10 ** (LM0 + k * (LM1 - LM0) / 200)
        if m > 512:
            break
        pts.append((rx(m), ry((Pp - 1) / (m + Pp - 1))))
        k += 1
    svg.path("M" + " L".join("%.1f %.1f" % xy for xy in pts), stroke=col, width=2.2)
    m4 = 4 * Pp
    f4 = (Pp - 1) / (m4 + Pp - 1)
    svg.circle(rx(m4), ry(f4), 3.6, fill=col)
    print("P", Pp, "m=4P fraction %.3f" % f4)
for i, (Pp, col) in enumerate(((16, "blue"), (8, "violet"), (4, "ink"))):
    yy = ry(0.95) + 8 + i * 18
    svg.line(rx(60), yy - 4, rx(110), yy - 4, color=col, width=2.2)
    svg.text(rx(120), yy, "P = %d" % Pp, size=12, color=col, weight="600")
# V3 measured
fv3 = 2.06 / 19.926
svg.line(RX0, ry(fv3), RX1, ry(fv3), color="green", width=1.4, dash="5 3")
svg.text(RX0 + 4, ry(fv3) + 15, "V3 实测 %.1f%%" % (100 * fv3), size=11, color="green", weight="600")
svg.text(RX0 - 30, RY1 + 56, "圆点：m = 4P（GPipe 的经验法则）", size=11, color="muted")

svg.save("m10-pipeline-bubble.svg")
print("total", total, "idle", idle[0], "fraction", idle[0] / total, "V3", fv3)
