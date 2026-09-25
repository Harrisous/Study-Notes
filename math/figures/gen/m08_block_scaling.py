"""Shared experiment for §6 (figures M8.9 and M8.10): per-tensor vs per-block scaling.

qdq(x, fmt, block, rule): quantise-dequantise x with one scale per block of `block`
elements (block=None: one scale for the whole tensor).  The scale maps the block's
absolute maximum onto the format's largest value (6 for E2M1, 448 for E4M3):
    rule "fp32"      s = amax / qmax                       (ideal, full-precision scale)
    rule "e4m3"      s = E4M3(amax / qmax)                 (NVFP4-style block scale)
    rule "pow2_up"   s = 2^ceil(log2(amax / qmax))          (power-of-two, rounded up)
    rule "ocp_floor" s = 2^(floor(log2 amax) - emax_elem)   (OCP MX example algorithm 1)
The element values are then rounded with m08_fp.quant (nearest, ties to even, saturating).

    python m08_block_scaling.py      # prints the statistics quoted in §6 and §7
"""

import math

import numpy as np

from m08_fp import quant

QMAX = {"E2M1": 6.0, "E4M3": 448.0}
EMAX = {"E2M1": 2, "E4M3": 8}


def scale(amax, fmt, rule="fp32"):
    q = QMAX[fmt]
    if rule == "fp32":
        return amax / q
    if rule == "e4m3":
        return float(quant(amax / q, "E4M3"))
    if rule == "pow2_up":
        return 2.0 ** math.ceil(math.log2(amax / q))
    if rule == "ocp_floor":
        return 2.0 ** (math.floor(math.log2(amax)) - EMAX[fmt])
    raise ValueError(rule)


def qdq(x, fmt, block=None, rule="fp32"):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    n = len(x) if block is None else block
    scales = []
    for i in range(0, len(x), n):
        seg = x[i:i + n]
        amax = float(np.max(np.abs(seg)))
        s = scale(amax, fmt, rule) if amax > 0 else 1.0
        scales.append(s)
        out[i:i + n] = quant(seg / s, fmt) * s
    return out, scales


def experiment(fmt, outlier, block, rule="fp32", n=1024, pos=500, seed=0):
    """1024 draws from N(0,1) with element `pos` replaced by `outlier`.
    Returns (fraction flushed to zero, median relative error) over the other 1023 values."""
    x = np.random.default_rng(seed).standard_normal(n)
    x[pos] = outlier
    xh, _ = qdq(x, fmt, block, rule)
    keep = np.ones(n, bool)
    keep[pos] = False
    zero = float(np.mean(xh[keep] == 0))
    med = float(np.median(np.abs(xh[keep] - x[keep]) / np.abs(x[keep])))
    return zero, med


CONFIGS = [
    # label, fmt, outlier, block, rule
    ("FP4 · 整张量一个缩放", "E2M1", 50.0, None, "fp32"),
    ("FP4 · 每 32 个一个缩放", "E2M1", 50.0, 32, "fp32"),
    ("FP4 · 每 16 个一个缩放", "E2M1", 50.0, 16, "fp32"),
    ("FP8 · 整张量一个缩放", "E4M3", 50.0, None, "fp32"),
    ("FP8 · 整张量，离群值 10^{5}", "E4M3", 1e5, None, "fp32"),
    ("FP8 · 每 128 个，离群值 10^{5}", "E4M3", 1e5, 128, "fp32"),
]

if __name__ == "__main__":
    for lab, fmt, o, b, r in CONFIGS:
        z, m = experiment(fmt, o, b, r)
        print("%-32s zero %.1f%%  median rel err %.1f%%" % (lab, 100 * z, 100 * m))
    print("-- no outlier")
    for fmt, b in (("E2M1", None), ("E2M1", 16), ("E4M3", None)):
        x = np.random.default_rng(0).standard_normal(1024)
        xh, _ = qdq(x, fmt, b)
        print(fmt, b, "zero %.1f%%  median %.1f%%" % (100 * np.mean(xh == 0),
                                                      100 * np.median(np.abs(xh - x) / np.abs(x))))
    print("-- FP4 16-blocks, scale rules")
    for r in ("fp32", "e4m3", "pow2_up", "ocp_floor"):
        for b in (16, 32):
            z, m = experiment("E2M1", 50.0, b, r)
            print(r, b, "zero %.1f%%  median %.1f%%" % (100 * z, 100 * m))
