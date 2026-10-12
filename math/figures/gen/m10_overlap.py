"""Figure M10.6: overlapping data movement with compute, on the MLA decode example of section 4.

One decode step of a 32K-token sequence in DeepSeek-V3, attention part of all 61 layers
(numbers from the lesson's worked example; a and b named as in derivation (M10.13)):
  FLOPs  = 278,528 x 32,768 x 61 = 5.567e11          -> a = FLOPs / F  = 5.567e11 / 990e12 s = 0.562 ms
  bytes  = 70,272 B/token x 32,768 tokens = 2.303e9 B  -> b = bytes / BW = 2.303e9 / 3.35e12 s = 0.687 ms
Row 1: no overlap (move everything, then compute): a + b.
Row 2: the data is cut into N_TILES equal tiles; tile k is computed while tile k+1 is being moved
(double buffering). Compute of tile k starts when tile k has arrived and tile k-1 is done.
The total is b + a / N_TILES here (b > a), i.e. the lower bound max(a, b) plus one tile of the
shorter side. N_TILES = 8 is only illustrative; real kernels use many more, smaller tiles.
Writes math/figures/m10-overlap.svg
"""

from m10_svg import SVG

BYTES = 70272 * 32768
FLOPS = 278528 * 32768 * 61
A = FLOPS / 990e12 * 1e3       # ms, computing (a in the lesson)
B = BYTES / 3.35e12 * 1e3      # ms, moving    (b in the lesson)
N_TILES = 8

W, H = 720, 312
svg = SVG(W, H, "重叠与不重叠：V3 一条 32K 序列 decode 一步的注意力，搬运与计算的两种排法")
X0, X1 = 110, 650
TMAX = 1.30
scale = (X1 - X0) / TMAX


def tx(t):
    return X0 + t * scale


svg.text(16, 24, "V3 一条 32K token 的序列，decode 一步、61 层的注意力：算 a = %.3f ms，搬 b = %.3f ms"
         % (A, B), size=13, weight="600")

LANE_H = 18


def lanes(y, label, sub, moves, comps):
    svg.text(16, y - 10, label, size=13, weight="600")
    svg.text(16 + 14 * len(label) + 4, y - 10, sub, size=12, color="muted")
    svg.text(X0 - 8, y + 13, "搬（HBM）", size=11, anchor="end", color="amber")
    svg.text(X0 - 8, y + LANE_H + 19, "算（SM）", size=11, anchor="end", color="blue")
    for s, e in moves:
        svg.rect(tx(s), y, (e - s) * scale, LANE_H, fill="amber", fill_opacity=0.55, stroke="surface", width=1)
    for s, e in comps:
        svg.rect(tx(s), y + LANE_H + 6, (e - s) * scale, LANE_H, fill="blue", fill_opacity=0.6,
                 stroke="surface", width=1)
    end = max(e for _, e in moves + comps)
    svg.line(tx(end), y - 4, tx(end), y + 2 * LANE_H + 10, color="ink", width=1.4)
    svg.text(tx(end) + 6, y + LANE_H + 19, "%.3f ms" % end, size=12, weight="600")
    return end


dm = B / N_TILES      # moving one tile
dc = A / N_TILES      # computing one tile

# row 1: no overlap
moves1 = [(k * dm, (k + 1) * dm) for k in range(N_TILES)]
comps1 = [(B + k * dc, B + (k + 1) * dc) for k in range(N_TILES)]
end1 = lanes(66, "① 不重叠", "搬完全部，再开算", moves1, comps1)

# row 2: double buffering
moves2 = [(k * dm, (k + 1) * dm) for k in range(N_TILES)]
comps2 = []
t_free = 0.0
for k in range(N_TILES):
    s = max(moves2[k][1], t_free)
    comps2.append((s, s + dc))
    t_free = s + dc
end2 = lanes(150, "② 重叠", "搬第 k+1 块时算第 k 块", moves2, comps2)

# the two ends of (M10.13)
lo, hi = max(A, B), A + B
svg.line(tx(lo), 44, tx(lo), 206, color="green", width=1.2, dash="4 3")
svg.line(tx(hi), 44, tx(hi), 206, color="red", width=1.2, dash="4 3")
svg.text(tx(lo) - 4, 222, "下界 max(a, b) = %.3f ms" % lo, size=11, anchor="end", color="green")
svg.text(tx(hi), 222, "上界 a + b = %.3f ms" % hi, size=11, anchor="middle", color="red")

# time axis
YA = 238
svg.line(X0, YA, X1, YA, color="muted", width=1.1)
for i in range(0, 7):
    t = 0.2 * i
    svg.line(tx(t), YA, tx(t), YA + 4, color="muted", width=1)
    svg.text(tx(t), YA + 17, ("%.1f" % t), size=11, anchor="middle", color="muted")
svg.text(X1, YA + 17, "ms", size=11, anchor="end", color="muted")

svg.text(16, 280, "② 的总长 = b + a/%d：每块的计算都藏在下一块的搬运后面，只剩最后一块的计算露在外面（琥珀 = 搬运，蓝 = 计算）。"
         % N_TILES, size=11, color="muted")
svg.text(16, 298, "切成 %d 块只是示意：块越多，露在外面的尾巴越短，总时间越接近下界；这条尾巴和 §5 流水线的“填充与排空”是同一回事。"
         % N_TILES, size=11, color="muted")

svg.save("m10-overlap.svg")
print("a (compute) = %.4f ms, b (move) = %.4f ms, no overlap %.4f ms, overlapped (%d tiles) %.4f ms, bound %.4f ms"
      % (A, B, end1, N_TILES, end2, lo))
