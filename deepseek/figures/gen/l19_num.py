"""Numbers shared by the lesson-19 figure scripts (DeepSeek-V4.1-Flash).

Every constant comes from deepseek/reference-docs/lesson-19-doc.md (config.json, FlashMLA README,
the V4 / V4.1 papers); every derived number is computed here and checked against the value the
research notes (section 4) and the lesson text quote, so a figure can never drift from the prose.
"""

from __future__ import annotations

# ---------------------------------------------------------------- per-entry byte formats (notes 3-D)
MAIN_V41 = 512 // 2 + 512 // 16          # 256 B of E2M1 + 32 E4M3 scales (one per 16 channels)
IDX = 128 // 2 + 128 // 32               # 64 B of E2M1 + 4 E8M0 scales (MXFP4, one per 32)
MAIN_V4 = 448 + 64 * 2 + 8               # 448 FP8 NoPE + 64 BF16 RoPE + 7 E8M0 scales + 1 pad
SWA_V41 = 512 + 512 // 32                # 512 FP8 (RoPE dims quantised too) + 16 E8M0 scales
SWA_V4 = MAIN_V4                         # V4 stores SWA entries in the same 584 B row format
V32_MLA, V32_IDX = 512 + 4 * 4 + 64 * 2, 128 + 4     # V3.2 FlashMLA FP8 row and FP8 indexer key
assert (MAIN_V41, IDX, MAIN_V4, SWA_V41, V32_MLA, V32_IDX) == (288, 68, 584, 528, 656, 132)

# ---------------------------------------------------------------- layer layouts (config.json)
L41, NWIN = 40, 128
COMPRESS_V41 = [0, 0] + [2] * 18 + [1] * 20          # first 40 entries of compress_ratios
KV_SRC = [2, 8, 14, 20]
IDX_SRC = [2, 8, 14, 20, 24, 28, 32, 36]
ENGRAM = [1, 14]
DSPARK_TARGET = [37, 38, 39]
L4F, N_CSA_4F, N_HCA_4F = 43, 21, 20
L4P, N_CSA_4P, N_HCA_4P = 61, 30, 31


def mode(layer: int) -> str:
    if COMPRESS_V41[layer] == 0:
        return "swa"
    if layer in KV_SRC:
        return "full"
    if layer in IDX_SRC:
        return "reindex"
    return "reuse"


MODES = [mode(i) for i in range(L41)]
assert MODES.count("swa") == 2 and MODES.count("full") == 4
assert MODES.count("reindex") == 4 and MODES.count("reuse") == 30

# ---------------------------------------------------------------- global KV bytes per token (notes 4-A)
G41 = sum((MAIN_V41 + IDX) / COMPRESS_V41[l] for l in KV_SRC)          # 3 x (1/2) + 1 x 1 = 2.5 entries
G41_ENC = sum((MAIN_V41 + IDX) / COMPRESS_V41[l] for l in KV_SRC if l < 20)
G4F = (N_CSA_4F / 4 + N_HCA_4F / 128) * MAIN_V4 + (N_CSA_4F / 4) * IDX
G4P = (N_CSA_4P / 4 + N_HCA_4P / 128) * MAIN_V4 + (N_CSA_4P / 4) * IDX
G41_STRUCT_V4PREC = 2.5 * (MAIN_V4 + IDX)                                # V4.1 structure, V4 precision
G41_NOSHARE = (18 / 2 + 20 / 1) * (MAIN_V41 + IDX)                        # every CSA2 layer keeps its own
assert G41 == 890 and G41_ENC == 534
assert G4F == 3514.25 and round(G4P, 2) == 5031.44
assert G41_STRUCT_V4PREC == 1630 and G41_NOSHARE == 10324

# ---------------------------------------------------------------- the generational staircase (notes 4-B)
GPT3 = 2 * 96 * 96 * 128 * 2
LLAMA3_70B = 2 * 80 * 8 * 128 * 2
DS67B = 2 * 95 * 8 * 128 * 2
V2_BF16 = 60 * 576 * 2
V2_6BIT = 60 * 576 * 6 / 8
V3_BF16 = 61 * 576 * 2
V32 = 61 * (V32_MLA + V32_IDX)
assert (GPT3, LLAMA3_70B, DS67B, V2_BF16, V2_6BIT, V3_BF16, V32) == (
    4718592, 327680, 389120, 69120, 25920, 70272, 48068)

# ---------------------------------------------------------------- SWA state per request (notes 4-D)
SWA_STATE_4F = L4F * NWIN * SWA_V4
SWA_STATE_41 = L41 * NWIN * SWA_V41
assert (SWA_STATE_4F, SWA_STATE_41) == (3214336, 2703360)


def prefill_ratio_bounded(T: float) -> float:
    """CED prefill work / full 40-layer prefill: encoder on all T, decoder on the last n_win."""
    return (20 * T + 20 * min(T, NWIN)) / (40 * T)


def prefill_ratio_exact(T: float) -> float:
    """Same, if the decoder SWA state had to be rebuilt exactly (last 20 x 128 = 2560 tokens)."""
    return (20 * T + 20 * min(T, 20 * NWIN)) / (40 * T)


assert round(prefill_ratio_bounded(4096), 6) == 0.515625 and prefill_ratio_exact(4096) == 0.8125

# ---------------------------------------------------------------- decode FLOPs (paper Fig. 2, digitised; notes 4-F)
CTX_K = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
FIG2 = {
    "DeepSeek-V1 (67B)": [146, 159, 185, 236, 338, 542, 953, 1776, 3359],
    "DeepSeek-V3.2": [77, 79, 83, 91, 108, 138, 206, 333, 591],
    "DeepSeek-V4-Flash": [16.4, 16.6, 16.9, 17.6, 19.0, 22.0, 27.2, 38.3, 60.1],
    "DeepSeek-V4.1-Flash": [20.1, 20.2, 20.3, 20.4, 20.5, 20.8, 21.5, 22.8, 25.6],
}
GROW_41 = 0.25 * 2 * 32 * 128 * 2.5                               # FP4 indexer over 2.5 T entries
GROW_4F = 0.25 * 2 * 64 * 128 * 21 / 4 + 1.0 * 2 * 2 * 64 * 512 * 20 / 128
assert (GROW_41, GROW_4F) == (5120, 41984)
CROSS = (20.1 - 16.4) * 1e9 / (GROW_4F - GROW_41)
assert round(CROSS) == 100369

# ---------------------------------------------------------------- activated parameters (notes 3-C / 4-C, safetensors shapes)
ACT_ROUTED_PER_LAYER = 212.3e6          # 6 of 384 routed experts
ACT_SHARED_PER_LAYER = 35.4e6
ATTN_PER_LAYER = 128.2e6                # mean over the 40 layers
ACT_DECODE, ACT_DECODE_NOEMB = 16.48e9, 15.82e9
ACT_PREFILL, ACT_PREFILL_NOEMB = 8.24e9, 7.58e9

# ---------------------------------------------------------------- an illustrative agent session (lesson §1; assumed workload)
SESS_T0, SESS_OUT, SESS_TOOL, SESS_K = 20000, 400, 2000, 50
SESS_CTX = [SESS_T0 + (k - 1) * (SESS_TOOL + SESS_OUT) for k in range(1, SESS_K + 1)]
SESS_IN = sum(SESS_CTX)
SESS_OUT_TOT = SESS_K * SESS_OUT
SESS_MISS = SESS_T0 + (SESS_K - 1) * SESS_TOOL
SESS_HIT = SESS_IN - SESS_MISS
assert SESS_IN == SESS_K * SESS_T0 + SESS_K * (SESS_K - 1) // 2 * (SESS_TOOL + SESS_OUT) == 3940000
assert (SESS_MISS, SESS_HIT, SESS_CTX[-1]) == (118000, 3822000, 137600)
# API list prices, USD per million tokens, off-peak (DeepSeek pricing page, fetched 2026-09-24)
PRICE_HIT, PRICE_MISS, PRICE_OUT = 0.003, 0.15, 0.60
SESS_COST_CACHED = (SESS_HIT * PRICE_HIT + SESS_MISS * PRICE_MISS + SESS_OUT_TOT * PRICE_OUT) / 1e6
SESS_COST_NOCACHE = (SESS_IN * PRICE_MISS + SESS_OUT_TOT * PRICE_OUT) / 1e6
assert round(SESS_COST_CACHED, 4) == 0.0412 and round(SESS_COST_NOCACHE, 3) == 0.603

if __name__ == "__main__":
    print("G41", G41, "G4F", G4F, "G4P", G4P, "ratio", G4F / G41)
    print("SWA", SWA_STATE_4F, SWA_STATE_41, SWA_STATE_4F / G4F, SWA_STATE_41 / G41)
    print("cross", CROSS)
