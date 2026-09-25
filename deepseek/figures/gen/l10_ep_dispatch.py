"""Figure 10.3: expert parallelism and why device-limited routing bounds the traffic.

Eight devices hold 20 routed experts each (V2: 160 routed experts, D = 8). One token picks 6 experts.
(a) Unrestricted top-6: the 6 experts happen to sit on 5 devices, so the hidden state is sent 5 times.
(b) Device-limited routing with M = 3: the 6 experts lie on at most 3 devices, so at most 3 sends.
The expert placements are illustrative; the counts 20 / 6 / 8 / 3 are V2's.
Writes deepseek/figures/l10-ep-dispatch.svg
"""

from l10_svg import SVG, box

W, H = 760, 494
svg = SVG(W, H, "专家并行下一个 token 要把隐藏状态发给几台设备：不受限 vs 设备受限（M = 3）")

DX0, PITCH, BW, BH = 184, 71, 62, 80
SQ, GP = 11, 3


def devices(y, chosen):
    """chosen: dict device index -> list of expert slots (0..19)."""
    for dev in range(8):
        x = DX0 + dev * PITCH
        hit = dev in chosen
        svg.rect(x, y, BW, BH, fill="blue" if hit else None, fill_opacity=0.07,
                 stroke="blue" if hit else "muted", width=1.3 if hit else 1, rx=6)
        for slot in range(20):
            cx = x + 5 + (slot % 4) * (SQ + GP)
            cy = y + 7 + (slot // 4) * (SQ + GP)
            if hit and slot in chosen[dev]:
                svg.rect(cx, cy, SQ, SQ, fill="blue", fill_opacity=0.9, rx=2)
            else:
                svg.rect(cx, cy, SQ, SQ, fill="muted", fill_opacity=0.12, stroke="muted", width=0.6,
                         rx=2, stroke_opacity=0.5)
        svg.text(x + BW / 2, y + BH + 17, "设备 %d" % (dev + 1), size=11, anchor="middle",
                 color="blue" if hit else "muted")


def fan(y, chosen, label_main, label_sub):
    tok_y = y + BH / 2
    box(svg, 18, tok_y - 26, 134, 52, "token t", color="ink", size=13, weight="600",
        sub="u_{t}：5120 维")
    for k, dev in enumerate(sorted(chosen)):
        x = DX0 + dev * PITCH + BW / 2
        top = y - 2
        cy = y - 24 - 7 * k
        svg.path("M152 %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" % (tok_y - 8, 170, cy, x - 30, cy, x, top),
                 stroke="blue", width=1.6, arrow=True)
    svg.text(DX0, y + BH + 42, label_main, size=13, weight="600")
    svg.text(DX0, y + BH + 60, label_sub, size=12, color="muted")


A = {1: [6], 2: [13], 4: [2, 17], 5: [9], 7: [4]}
B = {1: [6, 11], 4: [2, 17], 5: [9, 14]}
svg.text(18, 22, "(a) 不受限 Top-6：6 个专家恰好分散在 5 台设备上", size=14, weight="600")
devices(90, A)
fan(90, A, "发出 5 份隐藏状态（dispatch），专家算完再收回 5 份（combine）",
    "V2 的 160 个专家均匀随机取 6 个时，平均会触及 4.46 台设备（见图 10.5）")
svg.text(18, 286, "(b) 设备受限路由（M = 3）：先选 3 台设备，只在这 3 台的 60 个专家里取 Top-6", size=14,
         weight="600")
devices(326, B)
fan(326, B, "最多发出 3 份：每台设备每步最多向外发送 M·T = 3T 份隐藏状态",
    "代价：从 160 个专家中任选 6 个的自由度，缩小为“最多跨 3 台设备”的组合")
svg.save("l10-ep-dispatch.svg")
