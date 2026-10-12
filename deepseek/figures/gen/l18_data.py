"""Architecture numbers and the byte / FLOP models shared by the lesson-18 figure scripts.

Every number comes from deepseek/reference-docs/lesson-18-doc.md (sec. 3.1, 4.2, 4.3):
the V4 report sec. 4.2.1, the V4-Pro / V4-Flash config.json (compress_ratios, index_topk, ...),
the FlashMLA README (656 B V3.2 entry, 584 B V4 entry) and the DeepGEMM README (132 B V3.2
indexer key). The 68 B V4 indexer key follows the reference code (FP4 + UE8M0 scales).

The "equivalent FP8" weights (BF16 x2, FP8 x1, FP4 x0.5) are the research notes' inference
from the paper's Fig. 1; the paper does not state its conversion rule.
"""

from __future__ import annotations

K = 1024                     # the paper's Fig. 1 counts context in units of 1024 tokens
M1 = 2 ** 20                 # "1M" context = 1,048,576 tokens

# ---------------------------------------------------------------- per-layer layout (config.json)
PRO_RATIOS = [128, 128] + [4 if l % 2 == 0 else 128 for l in range(2, 61)]      # 61 layers
FLASH_RATIOS = [0, 0] + [4 if l % 2 == 0 else 128 for l in range(2, 43)]        # 43 layers
assert len(PRO_RATIOS) == 61 and PRO_RATIOS.count(4) == 30 and PRO_RATIOS.count(128) == 31
assert len(FLASH_RATIOS) == 43 and FLASH_RATIOS.count(4) == 21 and FLASH_RATIOS.count(128) == 20
assert FLASH_RATIOS.count(0) == 2

MODELS = {
    #          N_act,  n_CSA, n_HCA, n_SWA, heads, top-k
    "pro":   dict(N=49e9, nC=30, nH=31, nS=0, nh=128, k=1024, L=61),
    "flash": dict(N=13e9, nC=21, nH=20, nS=2, nh=64, k=512, L=43),
}
V32 = dict(N=37e9, L=61, nh=128, k=2048)

C_ENTRY = 512                # V4 entry dim, serves as key and value
M_CSA, M_HCA, N_WIN = 4, 128, 128
IDX_HEADS, IDX_DIM = 64, 128

# ---------------------------------------------------------------- bytes per token
V32_ENTRY = 512 + 4 * 4 + 64 * 2          # FP8 latent + 4 FP32 scales + BF16 RoPE key
V32_IDX = 128 + 4                         # FP8 indexer key + 1 FP32 scale
V4_ENTRY = 448 + 64 * 2 + 8               # 448 FP8 + 64 BF16 + 7 UE8M0 scales + 1 pad
V4_IDX = 128 // 2 + 128 // 32             # FP4 packed + 4 UE8M0 scales (reference code)
assert (V32_ENTRY, V32_IDX, V4_ENTRY, V4_IDX) == (656, 132, 584, 68)


def kv_v32() -> float:
    return V32["L"] * (V32_ENTRY + V32_IDX)


def kv_v4(model: str) -> dict:
    m = MODELS[model]
    parts = {
        "csa_main": m["nC"] * V4_ENTRY / M_CSA,
        "csa_idx": m["nC"] * V4_IDX / M_CSA,
        "hca": m["nH"] * V4_ENTRY / M_HCA,
    }
    parts["total"] = sum(parts.values())
    return parts


assert kv_v32() == 48068
assert kv_v4("pro")["total"] == 5031.4375 and kv_v4("flash")["total"] == 3514.25

# ---------------------------------------------------------------- per-token FLOPs at position t
EQ = dict(bf16=2.0, fp8=1.0, fp4=0.5)     # "equivalent FP8" weights (inferred, see docstring)
RAW = dict(bf16=1.0, fp8=1.0, fp4=1.0)


def flops_v32(t: float, w=EQ) -> dict:
    per_entry = 2 * 576 + 2 * 512          # QK^T with 576-dim keys + PV with 512-dim values
    return {
        "proj": 2 * V32["N"],
        "core": w["bf16"] * V32["L"] * V32["nh"] * per_entry * min(t, V32["k"]),
        "idx": w["fp8"] * V32["L"] * 2 * IDX_HEADS * IDX_DIM * t,
    }


def flops_v4(t: float, model: str, w=EQ) -> dict:
    m = MODELS[model]
    per_entry = 2 * C_ENTRY + 2 * C_ENTRY  # shared-KV MQA: 512-dim QK^T and PV
    return {
        "proj": 2 * m["N"],
        "csa": w["bf16"] * m["nC"] * m["nh"] * per_entry * (min(m["k"], t / M_CSA) + N_WIN),
        "hca": w["bf16"] * m["nH"] * m["nh"] * per_entry * (t / M_HCA + N_WIN),
        "swa": w["bf16"] * m["nS"] * m["nh"] * per_entry * N_WIN,
        "idx": w["fp4"] * m["nC"] * 2 * IDX_HEADS * IDX_DIM * t / M_CSA,
    }


def total(d: dict) -> float:
    return sum(v for k, v in d.items() if k != "total")


assert round(total(flops_v32(M1)) / 1e12, 4) == 1.1916
assert round(total(flops_v4(M1, "pro")) / 1e12, 4) == 0.3158
assert round(total(flops_v4(M1, "flash")) / 1e12, 4) == 0.1183
assert round(total(flops_v32(M1, RAW)) / 1e12, 4) == 1.1568
