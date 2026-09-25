"""Shared YaRN arithmetic for the lesson-05 figures (research notes sections 2.4-2.6, R3, R6-R8).

DeepSeek-V2/V3 settings: d_h^R = 64 rotary dims, base b = 10000, pre-training length L = 4096,
scale s = 40, alpha = 1 (beta_slow), beta = 32 (beta_fast).
"""

import math

D, B, L, S, ALPHA, BETA = 64, 10000.0, 4096, 40, 1, 32
N = D // 2
theta = [B ** (-2 * i / D) for i in range(N)]
lam = [2 * math.pi / t for t in theta]
r = [L / x for x in lam]                       # turns completed inside the pre-training window


def i_of_r(rr, d=D, ctx=L, base=B):
    """Pair index at which r_i equals rr (YaRN code: find_correction_dim)."""
    return d * math.log(ctx / (2 * math.pi * rr)) / (2 * math.log(base))


I_BETA, I_ALPHA = i_of_r(BETA), i_of_r(ALPHA)          # 10.47, 22.51
LOW, HIGH = max(math.floor(I_BETA), 0), min(math.ceil(I_ALPHA), D - 1)   # 10, 23


def gamma_paper(rr):
    return min(1.0, max(0.0, (rr - ALPHA) / (BETA - ALPHA)))


def scale_paper(i):
    g = gamma_paper(r[i])
    return (1 - g) / S + g


def mask_code(i):
    return 1 - min(1.0, max(0.0, (i - LOW) / (HIGH - LOW)))


def scale_code(i):
    m = mask_code(i)
    return (1 - m) / S + m


def scale_ntk(i):
    return S ** (-2 * i / (D - 2))


def scale_pi(i):
    return 1 / S


assert (LOW, HIGH) == (10, 23)
assert abs(scale_paper(16) - 0.19858) < 1e-4 and abs(scale_code(16) - 0.55) < 1e-9
assert abs(scale_ntk(23) - 0.0648) < 1e-4
