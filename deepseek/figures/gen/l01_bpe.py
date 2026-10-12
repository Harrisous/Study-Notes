"""Figure 1.3: BPE merge trajectories.

(a) Sennrich et al. (2016) Algorithm 1, run verbatim on its own toy dictionary
    {low:5, lower:2, newest:6, widest:3} for 10 merges (ties -> first pair seen, as Python's max).
    The merge sequence is asserted against reference-docs/lesson-01-doc.md sec. 4.7.
(b) GPT-2's byte-level BPE on " unbelievable": the 12 merges and their ranks in merges.txt,
    as recorded in lesson-01-doc sec. 4.7; applied here lowest-rank-first to rebuild each state.
Writes deepseek/figures/l01-bpe.svg
"""

import collections
import re

from l01_svg import SVG

# ---------------------------------------------------------------- (a) Sennrich Algorithm 1
vocab = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}


def get_stats(v):
    pairs = collections.defaultdict(int)
    for word, freq in v.items():
        sym = word.split()
        for i in range(len(sym) - 1):
            pairs[sym[i], sym[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    return {p.sub("".join(pair), w): f for w, f in v_in.items()}


states_a = [(None, None, list(vocab))]
for _ in range(10):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    n = pairs[best]
    vocab = merge_vocab(best, vocab)
    states_a.append((best, n, list(vocab)))
seq = [(b, n) for b, n, _ in states_a[1:]]
assert seq == [(("e", "s"), 9), (("es", "t"), 9), (("est", "</w>"), 9), (("l", "o"), 7), (("lo", "w"), 7),
               (("n", "e"), 6), (("ne", "w"), 6), (("new", "est</w>"), 6), (("low", "</w>"), 5),
               (("w", "i"), 3)], seq

# ---------------------------------------------------------------- (b) GPT-2 on " unbelievable"
merges_b = [(("l", "e"), 37), (("Ġ", "u"), 78), (("a", "b"), 141), (("e", "l"), 161), (("i", "e"), 238),
            (("ab", "le"), 284), (("Ġu", "n"), 299), (("b", "el"), 6411), (("ie", "v"), 10947),
            (("Ġun", "bel"), 17235), (("Ġunbel", "iev"), 20298), (("Ġunbeliev", "able"), 24223)]
ranks = {p: r for p, r in merges_b}
sym = list("Ġunbelievable")
states_b = [(None, None, sym[:])]
while True:
    cands = [(ranks[(sym[i], sym[i + 1])], i) for i in range(len(sym) - 1) if (sym[i], sym[i + 1]) in ranks]
    if not cands:
        break
    r, _ = min(cands)
    pair = [p for p, rr in ranks.items() if rr == r][0]
    out, i = [], 0
    while i < len(sym):
        if i < len(sym) - 1 and (sym[i], sym[i + 1]) == pair:
            out.append(sym[i] + sym[i + 1])
            i += 2
        else:
            out.append(sym[i])
            i += 1
    sym = out
    states_b.append((pair, r, sym[:]))
assert [(p, r) for p, r, _ in states_b[1:]] == merges_b
assert states_b[-1][2] == ["Ġunbelievable"]

# ---------------------------------------------------------------- drawing
W, H = 760, 658
s = SVG(W, H, "BPE 的合并轨迹：Sennrich 算法 1 的玩具词典，以及 GPT-2 对 “ unbelievable” 的 12 次合并")
CW = 7.0         # px per character in a chip at size 12
RH = 22


def chips(x, y, symbols, new=None, color="blue"):
    """Draw one segmentation as a row of chips; `new` = the symbol created by this row's merge."""
    for sy in symbols:
        w = max(14, len(sy) * CW + 8)
        hit = new is not None and sy == new
        s.rect(x, y - 13, w, 18, fill=color if hit else "ink", fill_opacity=0.16 if hit else 0.04,
               stroke=color if hit else "rule", width=1.2 if hit else 1, rx=3)
        s.text(x + w / 2, y + 1, sy, size=12, anchor="middle", color=color if hit else "ink",
               weight="600" if hit else None)
        x += w + 2
    return x


# (a)
s.text(10, 22, "(a) Sennrich 等（2016）算法 1 的玩具词典：每一步合并语料里出现最多的相邻符号对", size=13, weight="700")
cols = [(10, "步"), (40, "合并（出现次数）"), (172, "low ×5"), (308, "lower ×2"), (448, "newest ×6"), (598, "widest ×3")]
for x, t in cols:
    s.text(x, 46, t, size=12, color="muted", weight="600")
s.line(10, 53, 750, 53, color="rule", width=1)
xs = [172, 308, 448, 598]
for k, (pair, n, words) in enumerate(states_a):
    y = 72 + k * RH
    s.text(18, y, str(k), size=12, anchor="middle", color="muted")
    if pair:
        s.text(40, y, "%s + %s  (%d)" % (pair[0], pair[1], n), size=12, color="blue")
    else:
        s.text(40, y, "初始（逐字符）", size=12, color="muted")
    new = "".join(pair) if pair else None
    for x, w in zip(xs, words):
        chips(x, y, w.split(), new=new)
ya = 72 + 10 * RH + 22
s.text(10, ya, "</w> 是词尾符号。频数的来历：e + s 出现在 newest（6 次）和 widest（3 次）里，6 + 3 = 9；平局取先遇到的一对。",
       size=12, color="muted")

# (b)
YB = ya + 34
s.line(10, YB - 16, 750, YB - 16, color="rule", width=1)
s.text(10, YB + 4, "(b) GPT-2 的字节级 BPE：“ unbelievable” 从 13 个字节字符合并成 1 个 token", size=13, weight="700")
s.text(10, YB + 24, "Ġ 是空格字节 0x20 的可见替身；每步合并当前出现的、在 merges 表里排名（rank）最靠前的一对", size=12,
       color="muted")
for k, (pair, r, syms) in enumerate(states_b):
    y = YB + 50 + k * 20
    s.text(18, y, str(k), size=12, anchor="middle", color="muted")
    if pair:
        s.text(40, y, "%s + %s" % pair, size=12, color="violet")
        s.mono(226, y, "rank %s" % "{:,}".format(r), size=12, anchor="end", color="muted")
    else:
        s.text(40, y, "初始：13 个字节", size=12, color="muted")
    xe = chips(240, y, syms, new="".join(pair) if pair else None, color="violet")
s.text(xe + 10, YB + 50 + 12 * 20, "→ token id 24479", size=12, color="violet", weight="600")

s.save("l01-bpe.svg")
