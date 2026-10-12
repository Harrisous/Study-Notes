"""Figure 2.11 of the blackboard version of lesson 2: why a small update vanishes on a coarse ruler.

(a) The blackboard's decimal calculator with three significant digits: ticks every 0.01 near 1.00;
    adding 0.001 lands at 1.001, short of the midpoint 1.005, so the display rounds back to 1.00.
(b) BF16 near 0.02: the representable numbers are k * 2^-13 for k = 162..166 (7 stored mantissa bits
    in [2^-6, 2^-5)); 0.02 itself rounds to 164 * 2^-13 = 0.02001953125. An Adam step of about
    eta = 6e-5 moves 0.4915 of a tick, short of the midpoint, so round-to-nearest returns the old value.
    FP32 in the same binade has spacing 2^-29, i.e. 2^16 = 65,536 times finer.
Every tick value and ratio is computed here (the BF16 rounding by bit manipulation), not typed in.
Writes deepseek/figures/l02b-bf16-ruler.svg
"""

import struct

from l02_svg import SVG


def bf16(x: float) -> float:
    """Round a float32 to bfloat16 (round to nearest even) and return it as a Python float."""
    (u,) = struct.unpack("<I", struct.pack("<f", x))
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    (y,) = struct.unpack("<f", struct.pack("<I", u))
    return y


SP = 2.0 ** -13            # BF16 spacing in [2^-6, 2^-5)
theta = bf16(0.02)          # 0.02001953125
k0 = round(theta / SP)      # 164
STEP = 6e-5
frac = STEP / SP            # 0.4915 of a tick
after = bf16(theta + STEP)  # rounds back to theta
assert abs(after - theta) < 1e-12 and k0 == 164
fp32_sp = 2.0 ** -29
fp32_ticks = STEP / fp32_sp

W, H = 720, 318
svg = SVG(W, H, "刻度有多粗：三位计算器与 BF16 上被舍掉的一步")

X0, X1 = 70, 650


def ruler(y, lo, hi, ticks, labels, cur, step, mid, color, step_label, mid_label, cur_label):
    def mx(v):
        return X0 + (v - lo) / (hi - lo) * (X1 - X0)
    svg.line(X0 - 10, y, X1 + 10, y, color="ink", width=1.4)
    for v, lab in zip(ticks, labels):
        x = mx(v)
        is_cur = abs(v - cur) < 1e-15
        svg.line(x, y - 11, x, y + 11, color=color, width=2.6 if is_cur else 1.6)
        svg.text(x, y + 28, lab, size=12, anchor="middle", color=color, weight="600" if is_cur else None)
    # midpoint
    xm = mx(mid)
    svg.line(xm, y - 30, xm, y + 8, color="muted", width=1.2, dash="4 3")
    svg.text(xm + 5, y - 22, mid_label, size=12, color="muted")
    # the update arrow
    xa, xb = mx(cur), mx(cur + step)
    svg.arrow(xa, y - 18, xb, y - 18, color="red", width=2.2)
    svg.text(xa - 8, y - 16, step_label, size=12, anchor="end", color="red", weight="600")
    # curved return arrow: rounds back
    d = "M%.1f %.1f Q %.1f %.1f %.1f %.1f" % (xb, y - 12, (xa + xb) / 2, y + 2, xa + 4, y - 5)
    svg.path(d, stroke="red", width=1.3, dash="3 2", arrow=True)
    svg.text(mx(cur), y + 46, cur_label, size=12, anchor="middle", color="ink")
    return mx


# ---- (a) decimal calculator
svg.text(16, 26, "(a) 黑板上的三位有效数字计算器：刻度间隔 0.01", size=13, weight="600")
ya = 96
ticks_a = [0.99, 1.00, 1.01, 1.02]
ruler(ya, 0.985, 1.025, ticks_a, ["0.99", "1.00", "1.01", "1.02"], 1.00, 0.001, 1.005, "orange",
      "+0.001", "中点 1.005", "1.001 不到中点 → 仍显示 1.00")

# ---- (b) BF16 near 0.02
svg.text(16, 176, "(b) BF16 在 0.02 附近：刻度间隔 2^{-13} ≈ %.2f×10^{-4}" % (SP * 1e4), size=13, weight="600")
yb = 236
ks = [162, 163, 164, 165, 166]
ticks_b = [k * SP for k in ks]
labs_b = ["%.7f" % v for v in ticks_b]
ruler(yb, 161.5 * SP, 166.5 * SP, ticks_b, labs_b, theta, STEP, (k0 + 0.5) * SP, "orange",
      "+6×10^{-5}（%.2f 格）" % frac, "中点（0.5 格）", "舍入后仍是 %.7f，一点没动" % after)

svg.text(16, H - 12, "FP32 在同一段的间隔是 2^{-29}，细 %s 倍：这一步在 FP32 里跨过约 %s 格，记得下来。"
         % ("{:,}".format(int(SP / fp32_sp)), "%.1f 万" % (fp32_ticks / 1e4)), size=12, color="blue")
svg.save("l02b-bf16-ruler.svg")
print("theta", theta, "k0", k0, "frac", frac, "fp32 ticks", fp32_ticks)
