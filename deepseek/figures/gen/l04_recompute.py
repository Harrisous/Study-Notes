"""Figure 4.2: what each decode step computes, without and with a KV cache (T = 8 steps).

Cell (t, j): at generation step t, position j (no prompt, so step t sees exactly t positions; the
lesson reserves s for bytes per number). Without a cache step t re-runs positions 1..t
(sum = T(T+1)/2 = 36 cells); with a cache it computes only position t (T = 8 cells) and reads
positions 1..t-1 from the cache (T(T-1)/2 = 28 reads). Counts computed below.
Writes deepseek/figures/l04-recompute.svg
"""

from l04_svg import SVG

T = 8
n_nocache = sum(s for s in range(1, T + 1))
n_cache = T
n_read = sum(s - 1 for s in range(1, T + 1))
assert n_nocache == T * (T + 1) // 2 == 36 and n_read == 28

W, H = 720, 400
C = 25
svg = SVG(W, H, "每个 decode 步算了什么：不缓存与缓存（8 步）")


def grid(x0, y0, mode):
    for s in range(1, T + 1):
        y = y0 + (s - 1) * C
        svg.text(x0 - 8, y + C / 2 + 4, "%d" % s, size=11, anchor="end", color="muted")
        for j in range(1, T + 1):
            x = x0 + (j - 1) * C
            if j > s:
                svg.rect(x + 1, y + 1, C - 2, C - 2, fill=None, stroke="grid", width=1, rx=3)
            elif mode == "nocache" or j == s:
                svg.rect(x + 1, y + 1, C - 2, C - 2, fill="blue", fill_opacity=0.55, rx=3)
            else:
                svg.rect(x + 1, y + 1, C - 2, C - 2, fill="amber", fill_opacity=0.30, stroke="amber", width=0.8, rx=3)
    for j in range(1, T + 1):
        svg.text(x0 + (j - 0.5) * C, y0 - 8, "%d" % j, size=11, anchor="middle", color="muted")
    svg.text(x0 + T * C / 2, y0 - 26, "位置 j", size=12, anchor="middle", color="muted")
    svg.text(x0 - 30, y0 + T * C / 2, "第 t 步", size=12, anchor="middle", color="muted", rotate=-90)


gx1, gx2, gy = 80, 440, 88
svg.text(16, 24, "Ⓐ 不缓存：第 t 步把前 t 个位置整段重算", size=14, weight="600")
svg.text(376, 24, "Ⓑ 缓存：每步只算新位置，旧位置读缓存", size=14, weight="600")
grid(gx1, gy, "nocache")
grid(gx2, gy, "cache")

yb = gy + T * C + 30
svg.rect(gx1, yb - 11, 14, 14, fill="blue", fill_opacity=0.55, rx=3)
svg.text(gx1 + 20, yb, "计算 %d 格 = T(T+1)/2" % n_nocache, size=13, weight="600")
svg.text(gx1, yb + 22, "T 翻倍，格子数约翻 4 倍", size=12, color="muted")
svg.rect(gx2, yb - 11, 14, 14, fill="blue", fill_opacity=0.55, rx=3)
svg.text(gx2 + 20, yb, "计算 %d 格 = T" % n_cache, size=13, weight="600")
svg.rect(gx2 + 140, yb - 11, 14, 14, fill="amber", fill_opacity=0.30, stroke="amber", width=0.8, rx=3)
svg.text(gx2 + 160, yb, "读缓存 %d 格" % n_read, size=13, weight="600", color="amber")
svg.text(gx2, yb + 22, "黄色格子不花算力，但每步都要从显存读出来", size=12, color="muted")
svg.text(16, H - 14, "每一格代表“某一步里某一个位置”在所有层上的整套计算：投影、注意力、FFN。", size=12, color="muted")
svg.save("l04-recompute.svg")
