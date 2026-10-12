"""Figure 0.3: total parameters vs parameters activated per token (log-log), DeepSeek main line.

All parameter counts are the papers' own sentences as collected in lesson-00-doc.md section 3.3:
DeepSeek LLM 7B / 67B (dense), DeepSeekMoE 16B 16.4B/2.8B, V2-Lite 15.7B/2.4B, V2 236B/21B, V3 671B/37B,
V4-Flash 284B/13B, V4-Pro 1.6T/49B, V4.1-Flash 552B backbone with 8B (prefill) / 16B (decode) activated,
plus 196B Engram parameters outside the backbone. Diagonal reference lines mark activated/total ratios.

    python deepseek/figures/gen/l00_params.py
"""

from __future__ import annotations

import math

from l00_svg import SVG

# label, total (B), activated (B), kind, label (dx, dy, anchor)
POINTS = [
    ("DeepSeek LLM 7B", 7, 7, "dense", (8, 16, "start")),
    ("DeepSeek LLM 67B", 67, 67, "dense", (8, 16, "start")),
    ("DeepSeekMoE 16B", 16.4, 2.8, "moe", (-8, -7, "end")),
    ("V2-Lite", 15.7, 2.4, "moe", (-8, 12, "end")),
    ("V2", 236, 21, "moe", (-8, -6, "end")),
    ("V3", 671, 37, "moe", (-8, -7, "end")),
    ("V4-Flash", 284, 13, "moe", (-8, 5, "end")),
    ("V4-Pro", 1600, 49, "moe", (-8, -8, "end")),
    ("V4.1-Flash 解码", 552, 16, "v41", (8, -9, "start")),
    ("V4.1-Flash 预填充", 552, 8, "v41", (8, 5, "start")),
]
LABEL_AT = {1: 18, 1 / 10: 34, 1 / 20: 66, 1 / 40: 150, 1 / 80: 330}
RATIOS = [(1, "1（稠密）"), (1 / 10, "1/10"), (1 / 20, "1/20"), (1 / 40, "1/40"), (1 / 80, "1/80")]

W, H = 720, 470
L, R, T, B = 70, 690, 24, 410
X0, X1 = math.log10(4), math.log10(3000)      # total, B
Y0, Y1 = math.log10(1), math.log10(120)       # activated, B


def xp(v):
    return L + (R - L) * (math.log10(v) - X0) / (X1 - X0)


def yp(v):
    return B - (B - T) * (math.log10(v) - Y0) / (Y1 - Y0)


def main():
    s = SVG(W, H, title="总参数与每 token 激活参数")
    for v in (5, 10, 20, 50, 100, 200, 500, 1000, 2000):
        x = xp(v)
        s.line(x, T, x, B, color="rule" if v in (10, 100, 1000) else "grid", width=0.9)
        s.text(x, B + 17, ("%g" % v) + ("B" if v < 1000 else "") if v < 1000 else "%gT" % (v / 1000), size=12,
               anchor="middle", color="muted")
    for v in (1, 2, 5, 10, 20, 50, 100):
        y = yp(v)
        s.line(L, y, R, y, color="rule" if v in (1, 10, 100) else "grid", width=0.9)
        s.text(L - 7, y + 4, "%gB" % v, size=12, anchor="end", color="muted")
    s.text((L + R) / 2, B + 42, "总参数（对数轴）", size=12.5, anchor="middle")
    s.text(L - 48, (T + B) / 2, "每 token 激活参数（对数轴）", size=12.5, anchor="middle", rotate=-90)
    # ratio lines activated = r * total, clipped to the box in the script (no clipPath)
    for r, lab in RATIOS:
        xa, xb = 10 ** X0, 10 ** X1
        ya, yb_ = r * xa, r * xb
        if ya < 10 ** Y0:
            xa, ya = 10 ** Y0 / r, 10 ** Y0
        if yb_ > 10 ** Y1:
            xb, yb_ = 10 ** Y1 / r, 10 ** Y1
        s.line(xp(xa), yp(ya), xp(xb), yp(yb_), color="muted", width=1, dash="4 4")
        # label rotated along the line, at a total size where the line runs through empty space
        tl = LABEL_AT[r]
        ang = -math.degrees(math.atan(((B - T) / (Y1 - Y0)) / ((R - L) / (X1 - X0))))
        s.text(xp(tl), yp(r * tl) - 5, "激活比 " + lab, size=11.5, color="muted", rotate=ang)
    # Engram: V4.1 total including the 196B table
    s.line(xp(552), yp(16), xp(552 + 196), yp(16), color="amber", width=2, dash="3 3")
    s.circle(xp(748), yp(16), 4, fill="surface", stroke="amber", width=1.6)
    s.text(xp(552) + 9, yp(16) + 19, "虚线：再加 196B Engram 表", size=11.5, color="amber", halo=True)
    s.text(xp(552) + 9, yp(16) + 34, "合计约 748B", size=11.5, color="amber", halo=True)
    s.line(xp(552), yp(8), xp(552), yp(16), color="blue", width=2)
    for lab, tot, act, kind, (dx, dy, anc) in POINTS:
        x, y = xp(tot), yp(act)
        if kind == "dense":
            s.circle(x, y, 5.5, fill="ink")
        elif kind == "moe":
            s.circle(x, y, 5.5, fill="violet")
        else:
            s.circle(x, y, 5.5, fill="blue")
        s.text(x + dx, y + dy, "%s  %g / %g" % (lab, tot, act), size=12, anchor=anc,
               color="ink", weight=600 if kind != "dense" else None, halo=True)
        print("%-18s %7.3f%%" % (lab, 100 * act / tot))
    s.save("l00-params.svg")


if __name__ == "__main__":
    main()
