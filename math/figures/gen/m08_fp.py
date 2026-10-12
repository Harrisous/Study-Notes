"""Number-format helpers shared by the M08 scripts (numpy only).

Every small format (FP8, FP6, FP4) is handled by *enumerating* its non-negative finite
values from the bit layout and rounding to the nearest one (ties to the even code,
i.e. the value whose mantissa LSB is 0).  BF16 uses the usual bit trick on float32,
FP16 and FP32 use numpy's own types.  Out-of-range inputs saturate to the largest
finite value for the small formats (the FP8 paper's convention for E4M3) and become
inf for FP16/BF16/FP32.

    python m08_fp.py      # prints the format table used in the lesson (§3)
"""

from __future__ import annotations

import math

import numpy as np

# name: (exponent bits E, mantissa bits M, bias, kind)
#   kind "ieee": top exponent code reserved for Inf/NaN (FP32, BF16, FP16, E5M2)
#   kind "fn":   only S.1111.111 is NaN, every other code is finite (E4M3)
#   kind "full": no Inf, no NaN, every code finite (FP6, FP4)
FORMATS = {
    "FP64": (11, 52, 1023, "ieee"),
    "FP32": (8, 23, 127, "ieee"),
    "BF16": (8, 7, 127, "ieee"),
    "FP16": (5, 10, 15, "ieee"),
    "E5M2": (5, 2, 15, "ieee"),
    "E4M3": (4, 3, 7, "fn"),
    "E3M2": (3, 2, 3, "full"),
    "E2M3": (2, 3, 1, "full"),
    "E2M1": (2, 1, 1, "full"),
}


def decode(E: int, M: int, bias: int, e: int, m: int) -> float:
    """Value of the non-negative code with exponent field e and mantissa field m."""
    if e == 0:                                   # subnormal (or zero)
        return (m / 2 ** M) * 2.0 ** (1 - bias)
    return (1 + m / 2 ** M) * 2.0 ** (e - bias)


def values(name: str) -> np.ndarray:
    """All non-negative finite values of a small format, ascending (index = code)."""
    E, M, bias, kind = FORMATS[name]
    out = []
    for e in range(2 ** E):
        for m in range(2 ** M):
            if kind == "ieee" and e == 2 ** E - 1:
                continue
            if kind == "fn" and e == 2 ** E - 1 and m == 2 ** M - 1:
                continue
            out.append(decode(E, M, bias, e, m))
    return np.array(out)


def params(name: str) -> dict:
    """Closed-form parameters of a format (the formulas of §2)."""
    E, M, bias, kind = FORMATS[name]
    top = 2 ** E - 1                              # largest exponent code
    if kind == "ieee":
        emax_code, mmax = top - 1, 2 ** M - 1
    elif kind == "fn":
        emax_code, mmax = top, 2 ** M - 2
    else:
        emax_code, mmax = top, 2 ** M - 1
    return {
        "E": E, "M": M, "bias": bias, "p": M + 1,
        "max": (1 + mmax / 2 ** M) * 2.0 ** (emax_code - bias),
        "min_normal": 2.0 ** (1 - bias),
        "min_sub": 2.0 ** (1 - bias - M),
        "eps": 2.0 ** -M,
        "u": 2.0 ** -(M + 1),
    }


def round_to(x, vals: np.ndarray) -> np.ndarray:
    """Round to nearest representable value, ties to even code, saturating."""
    x = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(x), vals[-1])
    i = np.searchsorted(vals, a)                  # vals[i-1] < a <= vals[i]
    i = np.clip(i, 1, len(vals) - 1)
    lo, hi = vals[i - 1], vals[i]
    pick_hi = (a - lo > hi - a) | ((a - lo == hi - a) & (i % 2 == 0))
    r = np.where(pick_hi, hi, lo)
    r = np.where(a == 0, 0.0, r)
    return np.sign(x) * r


def bf16(x) -> np.ndarray:
    """Round float values to bfloat16 (nearest, ties to even) and return float64."""
    a = np.atleast_1d(np.asarray(x, dtype=np.float32))
    b = a.view(np.uint32)
    lsb = (b >> np.uint32(16)) & np.uint32(1)
    b = (b + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    out = b.view(np.float32).astype(np.float64)
    return out.reshape(np.shape(x))


def fp16(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def fp32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(np.float64)


_CACHE: dict = {}


def quant(x, name: str) -> np.ndarray:
    """Round x to format `name` (any of FORMATS)."""
    if name == "FP32":
        return fp32(x)
    if name == "BF16":
        return bf16(x)
    if name == "FP16":
        return fp16(x)
    if name not in _CACHE:
        _CACHE[name] = values(name)
    return round_to(x, _CACHE[name])


def binades(name: str) -> int:
    """Number of powers of two from the smallest subnormal to the largest value."""
    p = params(name)
    return int(math.floor(math.log2(p["max"]))) - int(round(math.log2(p["min_sub"]))) + 1


if __name__ == "__main__":
    for n in FORMATS:
        p = params(n)
        extra = ""
        if n not in ("FP64", "FP32", "BF16", "FP16"):
            v = values(n)
            assert abs(v[-1] - p["max"]) < 1e-12 * p["max"], n
            extra = "  finite non-negative values: %d" % len(v)
        # "覆盖的段数" in the lesson's table = log2(max / min_sub)
        span = math.log2(p["max"]) - math.log2(p["min_sub"])
        print("%-5s E%-2d M%-2d bias %-4d max %-12.6g min_normal %-12.6g min_sub %-12.6g "
              "eps %-10.4g u %-10.4g log2(max/min_sub) %-7.1f%s"
              % (n, p["E"], p["M"], p["bias"], p["max"], p["min_normal"], p["min_sub"],
                 p["eps"], p["u"], span, extra))
    fi = np.finfo(np.float64)
    assert (fi.max, fi.tiny, fi.smallest_subnormal, fi.eps) == (
        params("FP64")["max"], params("FP64")["min_normal"], params("FP64")["min_sub"], params("FP64")["eps"])
    print("E2M1 values:", values("E2M1"))
    print("E4M3 top 8:", values("E4M3")[-8:])
