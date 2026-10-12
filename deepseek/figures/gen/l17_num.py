"""Numbers shared by the lesson-17 figure scripts, recomputed from the research notes
(deepseek/reference-docs/lesson-17-doc.md, sections 3-4). Every derived value is asserted
against the value recorded there, so a figure can never drift from the text.

Run it directly to print the whole ledger:  python l17_num.py
"""

from __future__ import annotations

import math


# ------------------------------------------------------------------ primes (deterministic Miller-Rabin)
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# Table sizes as in engram_demo_v1.py / V4.1 engram.py: for every (layer, order, head) take the next
# prime at or above the running value that has not been used anywhere else in the model; each order
# restarts its search at the configured vocab size, so later orders and layers land just above.
def _layout(start, layers, orders, heads):
    seen: set[int] = set()
    res = []
    for _ in range(layers):
        lay = []
        for _ in range(orders):
            cur = start - 1
            row = []
            for _ in range(heads):
                cur += 1
                while not is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                row.append(cur)
            lay.append(row)
        res.append(lay)
    return res


E27_PRIMES = _layout(2_262_400, 2, 2, 8)          # Engram-27B: 2 layers x {2,3}-gram x 8 heads
E40_PRIMES = _layout(7_239_680, 2, 2, 8)          # Engram-40B
V41_PRIMES = _layout(16_000_000, 2, 3, 8)         # V4.1: 2 modules x {2,3,4}-gram x 8 heads

assert E27_PRIMES[0][0] == [2262409, 2262413, 2262437, 2262443, 2262451, 2262457, 2262461, 2262467]
assert E27_PRIMES[0][1] == [2262499, 2262521, 2262529, 2262541, 2262563, 2262569, 2262593, 2262619]
E27_ROWS = sum(sum(sum(r) for r in lay) for lay in E27_PRIMES)
E40_ROWS = sum(sum(sum(r) for r in lay) for lay in E40_PRIMES)
V41_ROWS_PER_MODULE = [sum(sum(r) for r in lay) for lay in V41_PRIMES]
assert E27_ROWS == 72_404_232, E27_ROWS
assert V41_ROWS_PER_MODULE == [384_006_168, 384_016_682], V41_ROWS_PER_MODULE
V41_ROWS = sum(V41_ROWS_PER_MODULE)
assert V41_PRIMES[0][0][0] == 16_000_057

E27_HEAD_DIM = 1280 // (2 * 8)                    # d_mem / (orders * heads) = 80
V41_HEAD_DIM = 2048 // 8                          # 256
E27_PARAMS = E27_ROWS * E27_HEAD_DIM
E40_PARAMS = E40_ROWS * E27_HEAD_DIM
V41_PARAMS = V41_ROWS * V41_HEAD_DIM
assert E27_PARAMS == 5_792_338_560
assert E40_PARAMS == 18_534_311_360, E40_PARAMS
assert V41_PARAMS == 196_613_849_600

# ------------------------------------------------------------------ parameter ledger (notes §4.2)
EXPERT = 3 * 2560 * 1536                          # one routed expert (inferred d_ff = 1536, VL2 config)
MOE_LAYERS = 29                                   # 30 layers, first one dense
EMB = 129_280 * 2560
ACT = 3.8e9
assert EXPERT == 11_796_480 and EMB == 330_956_800
P_MOE_27 = 49 * MOE_LAYERS * EXPERT               # inactive routed experts, Engram-27B (55 - 6)
P_MOE_BASE = 66 * MOE_LAYERS * EXPERT             # inactive routed experts, MoE-27B (72 - 6)
assert P_MOE_27 == 16_762_798_080
RHO_27 = P_MOE_27 / (P_MOE_27 + E27_PARAMS)
assert round(RHO_27, 4) == 0.7432
TOTALS = {
    "Dense-4B": ACT + EMB,
    "MoE-27B": ACT + P_MOE_BASE + EMB,
    "Engram-27B": ACT + P_MOE_27 + E27_PARAMS + EMB,
    "Engram-40B": ACT + P_MOE_27 + E40_PARAMS + EMB,
}
assert round(TOTALS["MoE-27B"] / 1e9, 2) == 26.71
assert round(TOTALS["Engram-27B"] / 1e9, 2) == 26.69
assert round(TOTALS["Engram-40B"] / 1e9, 2) == 39.43
assert round(TOTALS["Dense-4B"] / 1e9, 2) == 4.13

# ------------------------------------------------------------------ V4.1 storage and per-token fetch
V41_BYTES = V41_ROWS * (256 + 8)                  # FP8 values + one UE8M0 byte per 32 dims
assert round(V41_BYTES / 1e9, 2) == 202.76
V41_FETCH = 2 * 3 * 8 * (256 + 8)                 # rows fetched per token x bytes per row
assert V41_FETCH == 12_672

# ------------------------------------------------------------------ throughput (Engram table 4)
THR = {"Dense-4B": (9031.62, 8858.28), "Dense-8B": (6315.52, 6140.02)}
LOSS = {k: 1 - b / a for k, (a, b) in THR.items()}
assert round(LOSS["Dense-4B"] * 100, 3) == 1.919 and round(LOSS["Dense-8B"] * 100, 3) == 2.779

# ------------------------------------------------------------------ gating range
def ssqrt(x):
    return math.copysign(math.sqrt(abs(x)), x)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


assert round(math.sqrt(2560), 2) == 50.6 and round(2560 ** 0.25, 2) == 7.11
assert round(sigmoid(4), 4) == 0.9820 and round(sigmoid(ssqrt(4)), 4) == 0.8808

# ------------------------------------------------------------------ DeepSeek-OCR token arithmetic
def ocr_tokens(R):
    return (R // 16) ** 2 // 16


MODES = [("Tiny", 512), ("Small", 640), ("Base", 1024), ("Large", 1280)]
assert [ocr_tokens(r) for _, r in MODES] == [64, 100, 256, 400]
assert [(int(math.isqrt(n)) + 1) * int(math.isqrt(n)) + 1 for n in (64, 100, 256, 400)] == [73, 111, 273, 421]

FOX = [  # text-token bin, (acc, ratio) at 64 tokens, (acc, ratio) at 100 tokens, pages  (OCR table 2)
    ("600–700", (96.5, 10.5), (98.5, 6.7), 7),
    ("700–800", (93.8, 11.8), (97.3, 7.5), 28),
    ("800–900", (83.8, 13.2), (96.8, 8.5), 28),
    ("900–1000", (85.9, 15.1), (96.8, 9.7), 14),
    ("1000–1100", (79.3, 16.5), (91.5, 10.6), 11),
    ("1100–1200", (76.4, 17.7), (89.8, 11.3), 8),
    ("1200–1300", (59.1, 19.7), (87.1, 12.6), 4),
]
assert sum(r[3] for r in FOX) == 100
_under10 = [(r[2][0], r[3]) for r in FOX if r[2][1] < 10]
W_UNDER10 = sum(a * p for a, p in _under10) / sum(p for _, p in _under10)
assert round(W_UNDER10, 2) == 97.14

if __name__ == "__main__":
    print("E27 rows", E27_ROWS, "params", E27_PARAMS, "E40 params", E40_PARAMS)
    print("V4.1 rows", V41_ROWS_PER_MODULE, "params", V41_PARAMS, "bytes", V41_BYTES)
    print("rho", RHO_27, "totals", {k: round(v / 1e9, 2) for k, v in TOTALS.items()})
    print("E27 layer-2 primes", E27_PRIMES[1][0][:2], "V41 last", V41_PRIMES[1][2][-1])
    print("throughput loss", LOSS, "fox <10x weighted", W_UNDER10)
    print("three smallest primes product", E27_PRIMES[0][0][0] * E27_PRIMES[0][0][1] * E27_PRIMES[0][0][2], 2 ** 63)
