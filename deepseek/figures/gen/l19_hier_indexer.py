"""Figure 19.7: the Hierarchical Sparse Indexer, redrawn after V4.1 paper Fig. 5 with a toy instance.

Toy scale: 80 positions in 10 blocks of 8. Layer 20 (Full) scores all positions, keeps its own top-3,
and builds the candidate pool from the 3 blocks with the highest block-max score plus the block that
holds the newest position (always pinned, as in the reference code). Because the pool has at least as
many blocks as the layer keeps positions, it always contains layer 20's own top-k (2,048 >= 512 at real
scale). Reindex layers score only the pool and keep their own top-4. Scores are pseudo-random with a fixed seed; every selection is computed.
Real scale (config.json): blocks of 8, up to 2,048 blocks -> 16,384 candidates; top-512 per layer.
Writes deepseek/figures/l19-hier-indexer.svg
"""

import random

from l19_svg import SVG

W, H = 760, 420
svg = SVG(W, H, "分层稀疏索引器：层 20 全扫并圈出候选块，后面的 Reindex 层只在候选池里打分")

NB, BS = 10, 8
K_FULL, N_BLOCK, K_RE = 3, 3, 4
rng = random.Random(19)
score20 = [[0.6 * rng.random() ** 2 for _ in range(BS)] for _ in range(NB)]
# make a few positions clearly relevant so the picture reads well
for b, j, v in ((1, 3, 0.95), (4, 6, 0.9), (6, 1, 0.97), (6, 5, 0.8), (2, 0, 0.7)):
    score20[b][j] = v
flat = sorted(((score20[b][j], b, j) for b in range(NB) for j in range(BS)), reverse=True)
top20 = {(b, j) for _, b, j in flat[:K_FULL]}
bmax = sorted(((max(score20[b]), b) for b in range(NB - 1)), reverse=True)
pool = sorted({b for _, b in bmax[:N_BLOCK]} | {NB - 1})      # newest block is pinned
assert len(pool) == N_BLOCK + 1
assert all(b in pool for b, _ in top20)


def reindex_scores(seed):
    r = random.Random(seed)
    sc = {(b, j): r.random() for b in pool for j in range(BS)}
    return {p for p, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:K_RE]}


re24, re28 = reindex_scores(24), reindex_scores(28)

C, G = 15, 3          # cell size, gap


def grid(x0, y0, blocks, green, frame=None, pinned=None, dim=None):
    for r, b in enumerate(blocks):
        y = y0 + r * (C + G + 3)
        for j in range(BS):
            x = x0 + j * (C + G)
            if (b, j) in green:
                svg.rect(x, y, C, C, fill="green", fill_opacity=0.75, stroke="green", width=0.8, rx=2)
            else:
                svg.rect(x, y, C, C, fill="surface", stroke="rule", width=1, rx=2)
        if frame and b in frame:
            svg.rect(x0 - 3, y - 3, BS * (C + G) + 3, C + 6, stroke="blue", width=1.6, rx=3,
                     dash="4 2" if (pinned is not None and b == pinned) else None)
        svg.text(x0 - 8, y + C - 3, "块 %d" % b, size=11, anchor="end", color="muted")


# panel A: layer 20 scores everything
AX, AY = 58, 74
svg.text(20, 26, "层 20（Full）：对全部位置打分", size=13, weight="700")
svg.text(20, 44, "绿：本层 top-k；蓝框：块内最高分排前的块", size=12, color="muted")
grid(AX, AY, list(range(NB)), top20, frame=set(pool), pinned=NB - 1)
svg.text(AX + BS * (C + G) + 6, AY + (NB - 1) * (C + G + 3) + C - 3, "← 最新块，强制入池", size=11, color="blue")

# panel B: the shared pool
BX = 318
svg.arrow(AX + BS * (C + G) + 8, AY + 60, BX - 12, AY + 60, color="blue", width=1.6)
svg.text(BX - 6, 26, "共享候选池", size=13, weight="700")
svg.text(BX - 6, 44, "只保留蓝框里的块", size=12, color="muted")
grid(BX + 34, AY, pool, set(), frame=set(pool), pinned=NB - 1)

# panels C/D: reindex layers
CX = 552
for k, (lay, sel) in enumerate(((24, re24), (28, re28))):
    y0 = AY + k * 112
    svg.text(CX - 30, y0 - 10, "层 %d（Reindex）：只在池内打分" % lay, size=12, weight="700")
    grid(CX + 10, y0 + 2, pool, sel)
svg.arrow(BX + 34 + BS * (C + G) + 4, AY + 40, CX - 30, AY + 40, color="blue", width=1.4)
svg.arrow(BX + 34 + BS * (C + G) + 4, AY + 60, CX - 30, AY + 152, color="blue", width=1.4)
svg.text(CX + 80, AY + 228, "层 32、36 同理", size=12, anchor="middle", color="muted")

# real-scale footnote
FY = 344
svg.line(20, FY - 14, W - 20, FY - 14, color="rule", width=1)
svg.text(20, FY + 4, "示意：80 个位置、10 块 × 8；层 20 选 top-3，圈 3 块 + 最新块；Reindex 层各选 top-4。", size=12)
svg.text(20, FY + 26, "真实规模：块长 8，最多 2,048 块 → 16,384 个候选位置；1M 上下文时只占 1.56%。每层选 top-512。",
         size=12, color="blue")
svg.text(20, FY + 48, "只在解码器里用：层 20 仍扫描全部可见位置，层 24/28/32/36 的每查询打分量从 ∝ T 变成常数。",
         size=12, color="muted")

svg.save("l19-hier-indexer.svg")
