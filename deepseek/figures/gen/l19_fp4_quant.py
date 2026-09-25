"""Figure 19.9: the E2M1 grid and one 16-channel block quantised two ways (computed by l19_fp4).

(a) the 15 values an E2M1 number can take, on a number line (uneven spacing).
(b) the worked block X2: original values (black), dequantised with an E4M3 scale 0.375 (orange, the V4.1
    main-KV format) and with an E8M0 power-of-two scale 0.5 (red crosses, the MXFP4 indexer format).
    The two vertical rulers on the right are the representable values of each scaled grid.
Writes deepseek/figures/l19-fp4-quant.svg
"""

from l19_svg import SVG, fmt
from l19_fp4 import E2M1, X2, S_A, S_B, D_A, D_B, ERR_A, ERR_B

W, H = 740, 452
svg = SVG(W, H, "E2M1 的 15 个取值，以及一个 16 维块在两种缩放下的量化结果")

# ---------------------------------------------------------------- (a) number line
svg.text(20, 24, "(a) E2M1（1 位符号、2 位指数、1 位尾数）只能表示 15 个值", size=13, weight="700")
NX0, NX1, NY = 70, 690, 74


def nx(v):
    return NX0 + (v + 6) / 12 * (NX1 - NX0)


svg.line(NX0 - 10, NY, NX1 + 10, NY, color="muted", width=1.2)
vals = sorted({s * v for v in E2M1 for s in (1, -1)})
assert len(vals) == 15
for v in vals:
    svg.circle(nx(v), NY, 4.2, fill="orange")
    lab = fmt(v) if v >= 0 else "−" + fmt(-v)
    svg.text(nx(v), NY - 11, lab, size=11, anchor="middle")
for a, b, lab in ((0, 2, "间距 0.5"), (2, 4, "间距 1"), (4, 6, "间距 2")):
    svg.path("M%.1f %.1f L%.1f %.1f" % (nx(a) + 2, NY + 12, nx(b) - 2, NY + 12), stroke="muted", width=1)
    svg.text((nx(a) + nx(b)) / 2, NY + 27, lab, size=11, anchor="middle", color="muted")
svg.text(NX0 - 10, NY + 27, "越靠近 0 越密、越远越稀：这就是“浮点”", size=11, color="muted")

# ---------------------------------------------------------------- (b) the block
svg.text(20, 138, "(b) 一个 16 维块：原值 x（黑）与两种缩放下的反量化值", size=13, weight="700")
PX0, PX1 = 70, 540
V_MAX, V_MIN = 1.8, -2.4
PY0, PY1 = 160, 372


def px(i):
    return PX0 + 14 + i * (PX1 - PX0 - 28) / 15


def py(v):
    return PY0 + (V_MAX - v) / (V_MAX - V_MIN) * (PY1 - PY0)


for v in (-2, -1, 0, 1):
    svg.line(PX0, py(v), PX1, py(v), color="grid", width=1)
    svg.text(PX0 - 8, py(v) + 4, fmt(v) if v >= 0 else "−" + fmt(-v), size=11, anchor="end", color="muted")
svg.line(PX0, py(0), PX1, py(0), color="rule", width=1.2)
for i in range(16):
    svg.text(px(i), PY1 + 16, str(i + 1), size=11, anchor="middle", color="muted")
svg.text((PX0 + PX1) / 2, PY1 + 34, "通道（块内第几个数）", size=12, anchor="middle", color="muted")

for i, (x, a, b) in enumerate(zip(X2, D_A, D_B)):
    lo, hi = min(x, a, b), max(x, a, b)
    svg.line(px(i), py(hi), px(i), py(lo), color="rule", width=1)
    svg.circle(px(i), py(a), 5.2, stroke="orange", width=2)
    d = 4.2
    svg.line(px(i) - d, py(b) - d, px(i) + d, py(b) + d, color="red", width=1.8)
    svg.line(px(i) - d, py(b) + d, px(i) + d, py(b) - d, color="red", width=1.8)
    svg.circle(px(i), py(x), 2.8, fill="ink")

# the two rulers
for rx, s, col, name in ((590, S_A, "orange", "E4M3 缩放"), (670, S_B, "red", "E8M0 缩放")):
    svg.line(rx, PY0, rx, PY1, color="muted", width=1)
    grid = sorted({sg * v * s for v in E2M1 for sg in (1, -1)})
    for g in grid:
        if V_MIN - 1e-9 <= g <= V_MAX + 1e-9:
            svg.line(rx - 7, py(g), rx + 7, py(g), color=col, width=2)
    svg.text(rx, PY0 - 16, name, size=12, anchor="middle", color=col)
    svg.text(rx, PY0 - 3, "s = %g" % s, size=12, anchor="middle", color=col)
svg.text(630, PY1 + 16, "两把“尺子”的刻度", size=11, anchor="middle", color="muted")

# legend and errors
LY = PY1 + 60
svg.circle(76, LY - 4, 2.8, fill="ink")
svg.text(86, LY, "原值", size=12)
svg.circle(140, LY - 4, 5.2, stroke="orange", width=2)
svg.text(152, LY, "E4M3 缩放 0.375（主 KV 的格式）：相对 RMSE %.2f%%" % (ERR_A * 100), size=12, color="orange")
svg.line(470 - 4, LY - 8, 470 + 4, LY, color="red", width=1.8)
svg.line(470 - 4, LY, 470 + 4, LY - 8, color="red", width=1.8)
svg.text(482, LY, "E8M0 缩放 0.5（MXFP4）：%.2f%%" % (ERR_B * 100), size=12, color="red")

svg.save("l19-fp4-quant.svg")
