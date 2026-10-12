"""Figure M10.10: all-reduce time against the number of GPUs p, in the alpha-beta model (log-log).

Models (Thakur et al. 2005; latency alpha per message, beta = 1/BW per byte, reduction cost ignored):
  naive "reduce to one GPU, then broadcast from it", done one message at a time by that GPU:
        2 (p-1) (alpha + S beta)
  ring (reduce-scatter + all-gather):       2 (p-1) alpha + 2 (p-1)/p S beta
  recursive halving + doubling (p = 2^k):   2 lg p alpha + 2 (p-1)/p S beta
Parameters: BW = 50 GB/s (one InfiniBand NIC in the DeepSeek-V3 cluster), alpha = 5 us (an
assumption of the research notes, N77; Insights table 5 measures 2.8-3.7 us for 64-byte messages).
Left: S = 14 GB (BF16 gradients of a 7B-parameter model).  Right: S = 1 MB.
Writes math/figures/m10-allreduce-time.svg
"""

import math

from m10_svg import SVG

ALPHA = 5e-6
BETA = 1 / 50e9
PS = [2 ** k for k in range(1, 13)]


def naive(p, S):
    return 2 * (p - 1) * (ALPHA + S * BETA)


def ring(p, S):
    return 2 * (p - 1) * ALPHA + 2 * (p - 1) / p * S * BETA


def hd(p, S):
    return 2 * math.log2(p) * ALPHA + 2 * (p - 1) / p * S * BETA


W, H = 740, 380
svg = SVG(W, H, "all-reduce 的时间随卡数 p 的变化：大消息看带宽，小消息看延迟")
panels = [(70, "S = 14 GB（7B 模型的 BF16 梯度）", 14e9, -1, 3), (440, "S = 1 MB", 1e6, -5, -1)]
PW, TOP, BOT = 260, 50, 300


def draw(ox, title, S, ly0, ly1):
    def px(p):
        return ox + (math.log2(p) - 1) / (12 - 1) * PW

    def py(t):
        return BOT - (math.log10(t) - ly0) / (ly1 - ly0) * (BOT - TOP)

    svg.text(ox, 30, title, size=13, weight="600")
    for e in range(ly0, ly1 + 1):
        y = py(10 ** e)
        svg.line(ox, y, ox + PW, y, color="grid", width=1)
        lab = {-5: "10 µs", -4: "0.1 ms", -3: "1 ms", -2: "10 ms", -1: "0.1 s", 0: "1 s", 1: "10 s", 2: "100 s",
               3: "1000 s"}[e]
        svg.text(ox - 6, y + 4, lab, size=11, anchor="end", color="muted")
    for k in (1, 3, 5, 7, 9, 11):
        x = px(2 ** k)
        svg.line(x, TOP, x, BOT, color="grid", width=1)
        svg.text(x, BOT + 16, str(2 ** k), size=11, anchor="middle", color="muted")
    svg.line(ox, BOT, ox + PW, BOT, color="muted", width=1.2)
    svg.line(ox, TOP, ox, BOT, color="muted", width=1.2)
    svg.text(ox + PW / 2, BOT + 34, "卡数 p（对数刻度）", size=12, anchor="middle", color="muted")
    for f, col, width, dash in ((naive, "red", 2.0, "6 3"), (ring, "blue", 2.6, None), (hd, "green", 2.2, None)):
        pts = []
        for i in range(0, 221):
            p = 2 ** (1 + 11 * i / 220)
            t = f(p, S)
            if 10 ** ly0 <= t <= 10 ** ly1:
                pts.append("%.1f %.1f" % (px(p), py(t)))
        svg.path("M" + " L".join(pts), stroke=col, width=width, dash=dash)
    return px, py


px1, py1 = draw(*panels[0])
px2, py2 = draw(*panels[1])

# annotations, left panel
S = 14e9
svg.text(px1(2) + 6, py1(300), "朴素：归约到一张卡再广播", size=11, color="red")
svg.circle(px1(8), py1(ring(8, S)), 3.5, fill="blue")
svg.circle(px1(1024), py1(ring(1024, S)), 3.5, fill="blue")
svg.text(px1(8), py1(ring(8, S)) - 10, "p=8：%.2f s" % ring(8, S), size=11, anchor="middle", color="blue")
svg.text(px1(1024), py1(ring(1024, S)) - 10, "p=1024：%.2f s" % ring(1024, S), size=11, anchor="middle", color="blue")
svg.text(px1(24), py1(0.2), "环与减半倍增几乎重合，", size=11, color="muted")
svg.text(px1(24), py1(0.2) + 15, "都趋于 2S/BW = 0.56 s", size=11, color="muted")

# annotations, right panel
S = 1e6
svg.circle(px2(1024), py2(ring(1024, S)), 3.5, fill="blue")
svg.text(px2(1024) + 6, py2(ring(1024, S)) + 18, "环：%.1f ms" % (ring(1024, S) * 1e3), size=11, color="blue",
         weight="600")
svg.circle(px2(1024), py2(hd(1024, S)), 3.5, fill="green")
# below the curve with enough clearance: the curve dips ~7 px further down at p = 128
svg.text(px2(1024) - 4, py2(hd(1024, S)) + 26, "减半倍增：%.2f ms" % (hd(1024, S) * 1e3), size=11, anchor="end",
         color="green", weight="600")
svg.text(px2(12), py2(2.6e-5), "环的延迟项 2(p−1)α 随 p 线性增长", size=11, color="muted")
svg.text(px2(12), py2(2.6e-5) + 15, "（α = 5 µs 为假设值）", size=11, color="muted")

# legend
LY = H - 22
items = [("red", "朴素：2(p−1)(α+Sβ)", "6 3"), ("blue", "环：2(p−1)α + 2(p−1)/p·Sβ", None),
         ("green", "减半倍增：2 lg p·α + 2(p−1)/p·Sβ", None)]
x = 40
for col, lab, dash in items:
    svg.line(x, LY - 4, x + 24, LY - 4, color=col, width=2.4, dash=dash)
    svg.text(x + 30, LY, lab, size=12, color=col)
    x += 30 + len(lab) * 7.6 + 26

svg.save("m10-allreduce-time.svg")
for p in (2, 8, 64, 1024):
    print(p, "14GB ring %.4f s" % ring(p, 14e9), "hd %.4f" % hd(p, 14e9), "naive %.1f" % naive(p, 14e9),
          "| 1MB ring %.3f ms hd %.3f ms naive %.2f ms" % (ring(p, 1e6) * 1e3, hd(p, 1e6) * 1e3, naive(p, 1e6) * 1e3))
