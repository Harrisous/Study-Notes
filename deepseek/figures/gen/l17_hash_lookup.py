"""Figure 17.2: from three tokens to one memory vector (redrawn after Engram fig. 1 right).

Tokenizer compression -> 2-gram and 3-gram -> one multiplicative-XOR mix per order -> the same
mix taken modulo 8 distinct primes (the real first-layer table sizes of the Engram-27B layout,
recomputed in l17_num.py) -> 16 rows of 80 dims -> e_t in R^1280.
The token IDs and multipliers are made up; only the primes and dimensions are real. The row
positions drawn in each table are mix mod p / p for that made-up mix, so they are computed.
Writes deepseek/figures/l17-hash-lookup.svg
"""

from l17_svg import SVG
from l17_num import E27_PRIMES

P2, P3 = E27_PRIMES[0][0], E27_PRIMES[0][1]

# made-up compressed IDs and odd multipliers (illustration only)
ids = {"t-2": 48_117, "t-1": 279, "t": 23_455}
c = [0x5DEECE66D | 1, 0x2545F491 | 1, 0x9E3779B1 | 1]
mix2 = (ids["t"] * c[0]) ^ (ids["t-1"] * c[1])
mix3 = mix2 ^ (ids["t-2"] * c[2])
rows2 = [mix2 % p for p in P2]
rows3 = [mix3 % p for p in P3]

W, H = 760, 562
svg = SVG(W, H, "Engram 的查表：分词器压缩、N-gram、乘法-异或哈希、按 8 个素数取模、拼接")

# ---------------------------------------------------------------- row A: raw tokens
TX = [40, 170, 300]
toks = [("Alexander", "x_{t−2}", "ink"), (" the", "x_{t−1}", "ink"), (" Great", "x_{t}", "blue")]
for x, (s, lab, col) in zip(TX, toks):
    svg.box(x, 18, 116, 34, [s], color=col, fill_opacity=0.10, size=13)
    # raw ID label to the left of its arrow, so the arrow never crosses the subscript
    svg.text(x + 51, 86, lab, size=12, anchor="end", color="muted")
svg.text(20, 12, "", size=12)
# tokenizer compression note
NX = 450
svg.text(NX, 28, "分词器压缩 𝒫（满射，V → V′）", size=13, weight="700", color="violet")
svg.text(NX, 48, "NFKC → NFD → 去重音 → 小写 → 空白折叠", size=12)
svg.text(NX, 66, "例：“ Great”与“ great”→ 同一个 ID", size=12)
svg.text(NX, 84, "128K 词表的有效大小减少约 23%", size=12, color="muted")

for x in TX:
    svg.arrow(x + 58, 56, x + 58, 98, color="violet", width=1.5)
svg.text(TX[2] + 66, 86, "𝒫", size=13, color="violet", weight="700")
for x, lab in zip(TX, ["x′_{t−2}", "x′_{t−1}", "x′_{t}"]):
    svg.box(x + 18, 100, 80, 30, [lab], color="violet", fill_opacity=0.10, size=13)

# ---------------------------------------------------------------- row C: two mixes
GX = [20, 392]
GW = 348
# two text lines with subscripts: a 21-unit line gap keeps the first line's subscripts clear of the second line
svg.box(GX[0], 157, GW, 57, ["2-gram (x′_{t−1}, x′_{t})", "mix_{2} = x′_{t}·c_{0} ⊕ x′_{t−1}·c_{1}"],
        color="violet", fill_opacity=0.08, size=12.5, line_gap=21)
svg.box(GX[1], 157, GW, 57, ["3-gram (x′_{t−2}, x′_{t−1}, x′_{t})", "mix_{3} = mix_{2} ⊕ x′_{t−2}·c_{2}"],
        color="violet", fill_opacity=0.08, size=12.5, line_gap=21)
# a bracket under the compressed IDs, then one arrow into each N-gram box
BL, BR = TX[0] + 18, TX[2] + 98
svg.path("M%s 134 v6 H%s v-6" % (BL, BR), stroke="violet", width=1.3)
MID = (BL + BR) / 2
for g in range(2):
    svg.path("M%s 140 C %s 148, %s 144, %s 155" % (MID, MID, GX[g] + GW / 2, GX[g] + GW / 2),
             stroke="violet", width=1.3, arrow=True)

# ---------------------------------------------------------------- row D: 8 tables per order
TY, TH = 262, 112
TW, GAP = 30, 13.5
seg_centres = []
for g, (gx, primes, rws) in enumerate(((GX[0], P2, rows2), (GX[1], P3, rows3))):
    x0 = gx + (GW - (8 * TW + 7 * GAP)) / 2
    svg.text(gx + GW / 2, TY - 30, "同一个 mix 对 8 个不同素数取模：8 个头 = 8 张表",
             size=12, anchor="middle", color="muted")
    svg.arrow(gx + GW / 2, 214, gx + GW / 2, TY - 42, color="violet", width=1.4)
    for k in range(8):
        x = x0 + k * (TW + GAP)
        svg.rect(x, TY, TW, TH, fill="amber", fill_opacity=0.10, stroke="amber", width=1.1, rx=3)
        yy = TY + 4 + (TH - 12) * rws[k] / primes[k]
        svg.rect(x + 2, yy, TW - 4, 5, fill="orange", fill_opacity=0.9)
        svg.text(x + TW / 2, TY - 8, "h%d" % (k + 1), size=11, anchor="middle", color="muted")
        seg_centres.append((x + TW / 2, TY + TH))
    lab = "p = %s, %s, …, %s" % tuple("{:,}".format(v) for v in (primes[0], primes[1], primes[7]))
    svg.text(gx + GW / 2, TY + TH + 20, lab, size=12, anchor="middle")
    svg.text(gx + GW / 2, TY + TH + 38, "每张表约 226 万行 × 80 维（行号 = mix mod p）", size=12, anchor="middle",
             color="muted")

# ---------------------------------------------------------------- row E: concatenation
BY, BH = 468, 26
BX0, BW = 60, 640
sw = BW / 16
# two halves (2-gram rows, 3-gram rows); the 16 segments are marked by short ticks at the top and
# bottom edges only, so the dividers do not run through the labels in the middle of the bar
for g in range(2):
    svg.rect(BX0 + g * BW / 2, BY, BW / 2, BH, fill="orange", fill_opacity=0.22 if g == 0 else 0.42,
             stroke="orange", width=1)
for i in range(1, 16):
    if i == 8:
        continue
    xx = BX0 + i * sw
    svg.line(xx, BY, xx, BY + 6, color="orange", width=1)
    svg.line(xx, BY + BH - 6, xx, BY + BH, color="orange", width=1)
for g in range(2):
    svg.arrow(GX[g] + GW / 2, TY + TH + 48, BX0 + (2 * g + 1) * BW / 4, BY - 4, color="orange", width=1.6)
svg.text(W / 2, TY + TH + 66, "每个头取回一行，共 16 行", size=12, anchor="middle", color="orange")
svg.text(BX0 + BW / 4, BY + 18, "8 段来自 2-gram", size=12, anchor="middle")
svg.text(BX0 + 3 * BW / 4, BY + 18, "8 段来自 3-gram", size=12, anchor="middle")
svg.text(BX0 - 8, BY + 18, "*e*_{t}", size=14, anchor="end", weight="700", color="orange")
svg.text(W / 2, BY + 48, "拼接：*e*_{t} ∈ ℝ^{1280} = 2 阶 × 8 头 × 80 维（Engram-27B）；V4.1-Flash：3 阶 × 8 头 × 256 维 = 6144 维",
         size=12.5, anchor="middle")

svg.text(W / 2, BY + 70, "示意：token ID 与乘子 c 为虚构；素数是按演示代码布局复算的 Engram-27B 第一层真实表大小。",
         size=11.5, anchor="middle", color="muted")
svg.save("l17-hash-lookup.svg")
print("rows2", rows2[:3], "rows3", rows3[:3])
