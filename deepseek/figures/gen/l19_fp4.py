"""FP4 block quantisation used in lesson 19 (standard library only).

E2M1 element grid, E4M3 and E8M0 scale grids, and the two block formats:
  main KV   : E2M1 + one E4M3 scale per 16 channels, scale = RNE_E4M3(amax / 6)   (V4.1 code)
  indexer K : E2M1 + one E8M0 scale per 32 channels, scale = 2^ceil(log2(amax/6)) (MXFP4, rounded up)
The worked block X2 is the one in the lesson (§5 example, exercise 7); numbers are asserted against
the research notes (4-G, exercise 7).
"""

from __future__ import annotations

import math

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def q_e2m1(v: float) -> float:
    """Round to nearest E2M1 value; ties go to the even mantissa (the value with even list index)."""
    a = abs(v)
    best = min(E2M1, key=lambda e: (abs(e - a), E2M1.index(e) % 2))
    return math.copysign(best, v)


def e4m3_values():
    vals = set()
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue                      # NaN
            vals.add(m * 2 ** -9 if e == 0 else 2 ** (e - 7) * (1 + m / 8))
    return sorted(vals)


E4M3 = e4m3_values()
assert max(E4M3) == 448 and min(v for v in E4M3 if v > 0) == 2 ** -9


def q_e4m3(v: float) -> float:
    return min(E4M3, key=lambda e: abs(e - v))


def quant_block(x, scale_kind: str):
    amax = max(abs(v) for v in x)
    if scale_kind == "e4m3":
        s = q_e4m3(max(amax, 6 * 2 ** -9) / 6)
    else:
        s = 2.0 ** math.ceil(math.log2(amax / 6))
    codes = [q_e2m1(v / s) for v in x]
    return s, codes, [c * s for c in codes]


def rel_rmse(x, y):
    num = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)) / len(x))
    return num / math.sqrt(sum(a * a for a in x) / len(x))


X2 = [0.089, -0.634, 1.29, 0.298, -2.2, 0.052, 0.492, -0.902, 0.246, 1.566, -0.358, 0.679, -0.015, 0.783,
      -1.223, 0.433]
S_A, C_A, D_A = quant_block(X2, "e4m3")
S_B, C_B, D_B = quant_block(X2, "e8m0")
assert S_A == 0.375 and S_B == 0.5
assert C_A == [0, -1.5, 3, 1, -6, 0, 1.5, -2, 0.5, 4, -1, 2, -0.0, 2, -3, 1]
assert C_B == [0, -1.5, 3, 0.5, -4, 0, 1, -2, 0.5, 3, -0.5, 1.5, -0.0, 1.5, -2, 1]
ERR_A, ERR_B = rel_rmse(X2, D_A), rel_rmse(X2, D_B)
assert round(ERR_A, 4) == 0.0891 and round(ERR_B, 4) == 0.1209

if __name__ == "__main__":
    print("amax/6", max(abs(v) for v in X2) / 6)
    for name, s, c, d, e in (("E4M3", S_A, C_A, D_A, ERR_A), ("E8M0", S_B, C_B, D_B, ERR_B)):
        print(name, s, c, [round(v, 4) for v in d], "relRMSE %.4f" % e,
              "maxerr %.4f" % max(abs(a - b) for a, b in zip(X2, d)))
    print("x/s A", [round(v / S_A, 3) for v in X2])
    print("x/s B", [round(v / S_B, 3) for v in X2])
    print("rms x", math.sqrt(sum(v * v for v in X2) / 16))
