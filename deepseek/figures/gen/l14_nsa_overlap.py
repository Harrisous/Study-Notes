"""Figure 14.4: how one selection block collects scores from overlapping compression blocks.

Compression block b covers keys [16b+1, 16b+32] (l_c = 32, delta_c = 16); selection block j
covers keys [64j+1, 64j+64] (l_s = 64), the index conventions of NSA eqs. (7) and (12).
The overlap of each compression block with selection block j = 1, in units of 16 keys,
gives the weights 1, 2, 2, 2, 1 of NSA eq. (9). All overlaps are computed below.
Writes deepseek/figures/l14-nsa-overlap.svg
"""

from l14_svg import SVG

LC, DC, LS = 32, 16, 64
KEYS = 192
J = 1                                  # the selection block we look at
X0, SCALE = 70, 3.2                    # key k (1-based) -> x = X0 + (k - 1) * SCALE


def kx(k):
    return X0 + (k - 1) * SCALE


sel_lo, sel_hi = LS * J + 1, LS * (J + 1)
overlaps = {}
for b in range((KEYS - LC) // DC + 1):
    lo, hi = DC * b + 1, DC * b + LC
    ov = max(0, min(hi, sel_hi) - max(lo, sel_lo) + 1)
    if ov:
        overlaps[b] = ov // DC
assert overlaps == {3: 1, 4: 2, 5: 2, 6: 2, 7: 1}, overlaps

W, H = 740, 330
svg = SVG(W, H, "一个选择块从与之重叠的压缩块收集分数（NSA 式 9 的权重 1, 2, 2, 2, 1）")

# key axis
AY = 58
svg.line(kx(1), AY, kx(KEYS) + SCALE, AY, color="muted", width=1.2)
for k in (1, 33, 65, 97, 129, 161, 192):
    svg.line(kx(k), AY - 4, kx(k), AY + 4, color="muted", width=1)
    svg.text(kx(k), AY - 9, str(k), size=11, anchor="middle", color="muted")
svg.text(X0 - 12, AY + 4, "键位置", size=12, anchor="end", color="muted")

# selection blocks
SY = 76
CY = 118                               # compression rows
for j in range(KEYS // LS):
    lo, hi = LS * j + 1, LS * (j + 1)
    hl = j == J
    svg.rect(kx(lo), SY, (hi - lo + 1) * SCALE, 24, fill="violet" if hl else "muted",
             fill_opacity=0.22 if hl else 0.06, stroke="violet" if hl else "rule", width=1.4 if hl else 1)
    svg.text((kx(lo) + kx(hi) + SCALE) / 2, SY + 16, "选择块 %d（键 %d–%d）" % (j, lo, hi), size=12,
             anchor="middle", color="violet" if hl else "muted", weight="600" if hl else None)
svg.text(X0 - 12, SY + 16, "ℓ_{s} = 64", size=12, anchor="end", color="violet")

# boundaries of the highlighted selection block
for k in (sel_lo, sel_hi + 1):
    svg.line(kx(k), SY + 26, kx(k), CY + 52, color="violet", width=1.2, dash="4 3")

# compression blocks in two staggered rows
n_blocks = (KEYS - LC) // DC + 1
for b in range(n_blocks):
    lo, hi = DC * b + 1, DC * b + LC
    row = b % 2
    y = CY + row * 30
    w = overlaps.get(b, 0)
    col = "orange" if w else "muted"
    svg.rect(kx(lo) + 1, y, LC * SCALE - 2, 20, fill=col, fill_opacity=0.28 if w else 0.08,
             stroke=col if w else "rule", width=1.2 if w else 0.8, rx=3)
    cx = (kx(lo) + kx(hi) + SCALE) / 2
    if w == 1:                     # half inside: put the label on the inside half, clear of the boundary line
        cx += 22 if lo < sel_lo else -22
    svg.text(cx, y + 14, "b=%d" % b, size=11, anchor="middle", color="ink" if w else "muted")
svg.text(X0 - 12, CY + 14, "ℓ_{c} = 32", size=12, anchor="end", color="orange")
svg.text(X0 - 12, CY + 44, "δ_{c} = 16", size=12, anchor="end", color="orange")

# weights under the overlapping blocks
WY = 206
svg.text(X0 - 12, WY + 4, "重叠 / 16", size=12, anchor="end", color="muted")
for b, w in overlaps.items():
    lo, hi = DC * b + 1, DC * b + LC
    cx = (kx(lo) + kx(hi) + SCALE) / 2
    svg.circle(cx, WY, 11, fill="orange", fill_opacity=0.15, stroke="orange", width=1.2)
    svg.text(cx, WY + 5, str(w), size=13, anchor="middle", weight="700", color="orange")
    svg.text(cx, WY + 30, "一半" if w == 1 else "完整", size=11, anchor="middle", color="muted")

svg.text(X0, 272, "选择块 1 的分数 = 1·p^{cmp}[3] + 2·p^{cmp}[4] + 2·p^{cmp}[5] + 2·p^{cmp}[6] + 1·p^{cmp}[7]",
         size=13, weight="600")
svg.text(X0, 294, "权重 = 压缩块落在选择块内的键数 ÷ 16：中间 3 块完整在内，两端各有一半在内。", size=12,
         color="muted")
svg.text(X0, 314, "块下标按式 (14.6) 与 (14.12) 的约定计算；NSA 原文式 (9) 的下标与此对不齐（见正文），权重模式相同。",
         size=11, color="muted")

svg.save("l14-nsa-overlap.svg")
