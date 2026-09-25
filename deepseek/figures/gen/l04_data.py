"""Numbers shared by the lesson-04 figure scripts.

Every value below is copied from deepseek/reference-docs/lesson-04-doc.md (§3.1 model specs,
§3.2 hardware, §4 recomputed numbers); the asserts at the bottom re-derive the figures quoted in
the lesson so that a typo here fails loudly instead of drawing a wrong bar.
"""

# name: layers l, model width d, query heads n_h, KV heads n_g, head dim d_h, FFN width d_ff,
#       vocabulary V (for the parameter count only), native context (tokens), attention type
MODELS = {
    "GPT-3 175B": dict(l=96, d=12288, nh=96, ng=96, dh=128, dff=49152, V=50257, ctx=2048, kind="MHA"),
    "Llama-2-7B": dict(l=32, d=4096, nh=32, ng=32, dh=128, dff=11008, V=32000, ctx=4096, kind="MHA"),
    "DeepSeek LLM 7B": dict(l=30, d=4096, nh=32, ng=32, dh=128, dff=11008, V=102400, ctx=4096, kind="MHA"),
    "DeepSeek LLM 67B": dict(l=95, d=8192, nh=64, ng=8, dh=128, dff=22016, V=102400, ctx=4096, kind="GQA-8"),
    "Llama-2-70B": dict(l=80, d=8192, nh=64, ng=8, dh=128, dff=28672, V=32000, ctx=4096, kind="GQA-8"),
    "Llama-3-70B": dict(l=80, d=8192, nh=64, ng=8, dh=128, dff=28672, V=128256, ctx=8192, kind="GQA-8"),
    "Llama-3-8B": dict(l=32, d=4096, nh=32, ng=8, dh=128, dff=14336, V=128256, ctx=8192, kind="GQA-8"),
}
# DeepSeek-V3 (MLA): caches d_c + d_h^R = 512 + 64 elements per layer, 61 layers (lesson 9)
V3_MLA_BYTES = (512 + 64) * 61 * 2
# LLaMA-3.1 405B (Llama 3 paper Table 3), used in the ISCA 2025 Table 1 comparison
L405 = dict(l=126, d=16384, nh=128, ng=8, dh=128)

S_BF16 = 2  # bytes per element

HW = {
    "A100": dict(F=312e12, BW=2.039e12, HBM=80e9, label="A100 80GB SXM"),
    "H100": dict(F=989e12, BW=3.35e12, HBM=80e9, label="H100 SXM"),
}


def kv_bytes(m, s=S_BF16):
    """KV cache bytes per token: 2 (K and V) * n_g * d_h * l * s."""
    return 2 * m["ng"] * m["dh"] * m["l"] * s


def kv_bytes_mha(m, s=S_BF16):
    return 2 * m["nh"] * m["dh"] * m["l"] * s


def n_mm_llama(m):
    """Parameters that take part in matrix multiplies for one token (Llama-style SwiGLU block,
    untied lm_head, no input-embedding lookup): per layer Q, O (d x n_h d_h), K, V (d x n_g d_h),
    three FFN matrices (d x d_ff); plus the lm_head V x d."""
    per_layer = 2 * m["d"] * m["nh"] * m["dh"] + 2 * m["d"] * m["ng"] * m["dh"] + 3 * m["d"] * m["dff"]
    return m["l"] * per_layer + m["V"] * m["d"]


def params_llama(m):
    """Total parameters of a Llama-style model (embedding and lm_head untied; norms ignored)."""
    return n_mm_llama(m) + m["V"] * m["d"]


# ---- self-checks against the research notes (§4.1, §4.2, §4.4, §4.6)
assert kv_bytes(MODELS["GPT-3 175B"]) == 4_718_592
assert kv_bytes(MODELS["Llama-2-7B"]) == 524_288
assert kv_bytes(MODELS["Llama-2-70B"]) == 327_680
assert kv_bytes(MODELS["Llama-3-8B"]) == 131_072
assert kv_bytes(MODELS["DeepSeek LLM 7B"]) == 491_520
assert kv_bytes(MODELS["DeepSeek LLM 67B"]) == 389_120
assert kv_bytes_mha(MODELS["DeepSeek LLM 67B"]) == 3_112_960
assert V3_MLA_BYTES == 70_272
assert kv_bytes(L405) == 516_096
assert abs(n_mm_llama(MODELS["Llama-2-7B"]) - 6.607e9) < 1e6
assert abs(params_llama(MODELS["Llama-2-7B"]) - 6.738e9) < 1e6
assert abs(params_llama(MODELS["Llama-3-8B"]) - 8.030e9) < 1e6
