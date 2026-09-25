"""Figure 14.2: four attention patterns on the same 32-token causal grid (schematic).

(a) dense causal; (b) sliding window w = 8; (c) NSA-style block selection with block size 4:
1 initial block + 2 local blocks fixed, plus 1 block picked by score; (d) DSA-style token
selection, top-k with k = 8. Which dynamic block / tokens get picked is drawn from a seeded
random generator (an illustration, not model output). Cell counts are computed here.
Writes deepseek/figures/l14-sparse-patterns.svg
"""

import random

from l14_svg import SVG

T = 32
CELL = 5
PANEL = T * CELL            # 160
GAP = 24
X_START = 20
Y_TOP = 62
BLOCK = 4
W_WIN = 8
K_SEL = 8

rng = random.Random(14)


def dense(t):
    return {j: "fixed" for j in range(t + 1)}


def window(t):
    return {j: "fixed" for j in range(max(0, t - W_WIN + 1), t + 1)}


def nsa(t):
    b = t // BLOCK
    fixed = {0, b, b - 1} - {-1}
    out = {}
    for blk in fixed:
        for j in range(blk * BLOCK, min(t, blk * BLOCK + BLOCK - 1) + 1):
            out[j] = "fixed"
    cand = [blk for blk in range(b) if blk not in fixed]
    if cand:
        # one earlier block "picked by score": seeded random, an illustration only
        pick = rng.choice(cand)
        for j in range(pick * BLOCK, pick * BLOCK + BLOCK):
            out[j] = "picked"
    return out


def dsa(t):
    if t + 1 <= K_SEL:
        return {j: "picked" for j in range(t + 1)}
    # a few recent tokens plus some far ones, all by "score" (illustration)
    pool = list(range(t + 1))
    weights = [1.0 + 3.0 * (j > t - 5) + 1.5 * (j in (0, 1, 9, 10, 17)) for j in pool]
    chosen = set()
    while len(chosen) < K_SEL:
        chosen.add(rng.choices(pool, weights)[0])
    return {j: "picked" for j in chosen}


PANELS = [
    ("(a) 稠密因果", "每个查询看全部前文", dense),
    ("(b) 滑动窗口 w = 8", "只看最近 8 个", window),
    ("(c) 块选择（NSA 式）", "块长 4：开头 1 + 本地 2 + 选 1 块", nsa),
    ("(d) token 选择（DSA 式）", "打分选 top-8 个 token", dsa),
]

W = X_START * 2 + 4 * PANEL + 3 * GAP
H = Y_TOP + PANEL + 92
svg = SVG(W, H, "同一段 32 个 token 上的四种注意力模式（示意）")

dense_total = T * (T + 1) // 2
for p, (title, sub, fn) in enumerate(PANELS):
    x0 = X_START + p * (PANEL + GAP)
    y0 = Y_TOP
    svg.text(x0 + PANEL / 2, 22, title, size=13, anchor="middle", weight="600")
    svg.text(x0 + PANEL / 2, 40, sub, size=11, anchor="middle", color="muted")
    # masked upper triangle
    svg.polygon([(x0 + CELL, y0), (x0 + PANEL, y0), (x0 + PANEL, y0 + PANEL - CELL)],
                fill="red", fill_opacity=0.07)
    count = 0
    for t in range(T):
        sel = fn(t)
        count += len(sel)
        for j, kind in sel.items():
            col = "green" if kind == "fixed" else "violet"
            svg.rect(x0 + j * CELL + 0.4, y0 + t * CELL + 0.4, CELL - 0.8, CELL - 0.8, fill=col,
                     fill_opacity=0.78)
    svg.rect(x0, y0, PANEL, PANEL, stroke="rule", width=1)
    if p == 2:
        for b in range(1, T // BLOCK):
            svg.line(x0 + b * BLOCK * CELL, y0, x0 + b * BLOCK * CELL, y0 + PANEL, color="rule", width=0.6)
    svg.text(x0 + PANEL / 2, y0 + PANEL + 20, "算 %d 格（稠密的 %.0f%%）" % (count, 100 * count / dense_total),
             size=12, anchor="middle")
    if p == 0:
        svg.text(x0 - 6, y0 + 8, "q", size=11, anchor="end", color="muted")
        svg.text(x0 + 2, y0 - 4, "k →", size=11, color="muted")

# legend
ly = Y_TOP + PANEL + 50
lx = X_START
svg.rect(lx, ly - 10, 12, 12, fill="green", fill_opacity=0.78)
svg.text(lx + 18, ly, "按固定规则保留（位置决定）", size=12)
svg.rect(lx + 210, ly - 10, 12, 12, fill="violet", fill_opacity=0.78)
svg.text(lx + 228, ly, "按打分选中（内容决定）", size=12)
svg.rect(lx + 410, ly - 10, 12, 12, fill="red", fill_opacity=0.12)
svg.text(lx + 428, ly, "因果掩码去掉的未来位置", size=12)
svg.text(lx, ly + 24, "行 = 查询位置 t（从上往下 0…31），列 = 键位置 s（从左往右 0…31）。(c)(d) 中被选中的位置由随机数示意，不是模型输出。",
         size=11, color="muted")

svg.save("l14-sparse-patterns.svg")
