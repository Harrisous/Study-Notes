"""Numerics shared by the lesson-12 figure scripts: the FP8 value grids and rounding.

E4M3 (FP8 Formats for Deep Learning, arXiv:2209.05433, Table 1): bias 7, no infinities,
only S.1111.111 is NaN, so the largest finite value is S.1111.110 = 1.75 * 2^8 = 448.
    normal     e = 1..15, m = 0..7 :  (1 + m/8) * 2^(e-7)   (e = 15, m = 7 excluded: NaN)
    subnormal  e = 0,     m = 1..7 :  (m/8) * 2^-6
E5M2: bias 15, IEEE-style specials (e = 31 reserved), max 1.75 * 2^15 = 57344.
    normal     e = 1..30, m = 0..3 :  (1 + m/4) * 2^(e-15)
    subnormal  e = 0,     m = 1..3 :  (m/4) * 2^-14

Rounding is round-to-nearest, ties to even mantissa; overflow saturates to the max finite
value (the paper: "Values that overflow are then saturated to the maximum representable
value"). Only numpy is used.
"""

from __future__ import annotations

import numpy as np


def e4m3_grid() -> np.ndarray:
    vals = [0.0]
    vals += [(m / 8) * 2.0 ** -6 for m in range(1, 8)]
    for e in range(1, 16):
        for m in range(8):
            if e == 15 and m == 7:
                continue
            vals.append((1 + m / 8) * 2.0 ** (e - 7))
    return np.array(sorted(vals))


def e5m2_grid() -> np.ndarray:
    vals = [0.0]
    vals += [(m / 4) * 2.0 ** -14 for m in range(1, 4)]
    for e in range(1, 31):
        for m in range(4):
            vals.append((1 + m / 4) * 2.0 ** (e - 15))
    return np.array(sorted(vals))


E4M3 = e4m3_grid()
E5M2 = e5m2_grid()
assert len(E4M3) == 127 and E4M3[-1] == 448.0
assert E5M2[-1] == 57344.0


def round_to_grid(x, grid=E4M3):
    """Round |x| to the nearest grid value (ties -> even index = even mantissa), keep sign.

    In both grids the index parity equals the mantissa parity (zero has m = 0, every binade
    starts at m = 0 and has an even number of mantissa codes), so "even index" is
    "ties to even".
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(x), grid[-1])            # saturate
    hi = np.searchsorted(grid, a, side="left")
    hi = np.clip(hi, 1, len(grid) - 1)
    lo = hi - 1
    dlo = a - grid[lo]
    dhi = grid[hi] - a
    pick = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    return np.sign(x) * grid[pick]


def quantize_e4m3(x, scale):
    """Scale, round to E4M3, and dequantize: returns scale * E4M3(x / scale)."""
    return scale * round_to_grid(np.asarray(x) / scale, E4M3)


if __name__ == "__main__":
    print("E4M3 non-negative finite values:", len(E4M3), "max", E4M3[-1], "min subnormal", E4M3[1])
    print("E5M2 non-negative finite values:", len(E5M2), "max", E5M2[-1], "min subnormal", E5M2[1])
    print("0.1 ->", round_to_grid(0.1), "rel err %.4f" % (abs(round_to_grid(0.1) - 0.1) / 0.1))
