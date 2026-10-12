"""Figure 5.3: RoPE as a set of clocks (research notes R1, R23).

d_h^R = 64 (DeepSeek-V2/V3 decoupled RoPE), base b = 10000, pairs i = 0, 4, ..., 28.
Hand angle of pair i at position t is t * theta_i (mod 360 deg), theta_i = b^(-2i/d).
Rows: t = 100, t = 101, t = 4096. Every angle is computed here.
Writes deepseek/figures/l05-clocks.svg
"""

import math

from l05_svg import SVG

D, B = 64, 10000.0
PAIRS = list(range(0, 32, 4))
ROWS = [100, 101, 4096]


def theta(i):
    return B ** (-2 * i / D)


W, H = 760, 470
svg = SVG(W, H, "RoPE 的多频率时钟：d_h^R = 64 的 8 对频率在 t = 100、101、4096 时的指针")
X0, DX = 118, 80
R = 27

# column headers
svg.text(20, 40, "第 i 对", size=12, color="muted")
svg.text(20, 58, "波长 λ_{i}", size=12, color="muted")
for n, i in enumerate(PAIRS):
    cx = X0 + n * DX
    lam = 2 * math.pi / theta(i)
    svg.text(cx, 40, "i = %d" % i, size=12.5, anchor="middle", weight="600")
    lab = ("%.2f" % lam) if lam < 100 else (("%.1f" % lam) if lam < 1000 else "{:,.0f}".format(lam))
    svg.text(cx, 58, lab, size=12, anchor="middle", color="muted")

for r, t in enumerate(ROWS):
    cy = 118 + r * 112
    svg.text(20, cy + 4, "t = %d" % t, size=13, weight="700")
    for n, i in enumerate(PAIRS):
        cx = X0 + n * DX
        a = t * theta(i)
        deg = math.degrees(a) % 360
        svg.circle(cx, cy, R, fill="surface", stroke="rule", width=1.3)
        for tick in range(12):
            ta = tick * math.pi / 6
            svg.line(cx + (R - 4) * math.cos(ta), cy - (R - 4) * math.sin(ta), cx + R * math.cos(ta),
                     cy - R * math.sin(ta), color="rule", width=1)
        # reference direction (angle 0)
        svg.line(cx, cy, cx + R, cy, color="muted", width=0.9, dash="2 2")
        if t == 4096 and a < 2 * math.pi:
            # a slow hand that has not yet completed one turn: shade the swept angle
            steps = max(4, int(a / 0.06))
            pts = [(cx, cy)] + [(cx + (R - 1) * math.cos(a * s / steps), cy - (R - 1) * math.sin(a * s / steps))
                                for s in range(steps + 1)]
            svg.polygon(pts, fill="amber", fill_opacity=0.22)
        rad = math.radians(deg)
        svg.line(cx, cy, cx + (R - 5) * math.cos(rad), cy - (R - 5) * math.sin(rad), color="red", width=2.2,
                 cap="round")
        svg.circle(cx, cy, 2.4, fill="red")
        svg.text(cx, cy + R + 16, "%.1f°" % deg, size=12, anchor="middle")
        if t == 4096:
            turns = a / (2 * math.pi)
            svg.text(cx, cy + R + 32, ("%.1f 圈" % turns) if turns >= 1 else ("%.2f 圈" % turns), size=11.5,
                     anchor="middle", color="amber" if turns < 1 else "muted",
                     weight="600" if turns < 1 else None)

# annotations
svg.text(X0 - 20, 188, "↑ 从 t = 100 到 101：最快的针走 57.3°，最慢的针几乎不动", size=12, color="blue")
svg.text(X0 - 20, 420, "↑ 到 t = 4096，i = 24、28 两根慢针仍不满一圈（琥珀色扇形 = 从 0 起累计转过的角度）", size=12,
         color="amber")
svg.text(20, 452, "指针角度 = t·θ_{i}（对 360° 取余），θ_{i} = 10000^{−2i/64}；每格刻度 30°。", size=12, color="muted")

svg.save("l05-clocks.svg")
