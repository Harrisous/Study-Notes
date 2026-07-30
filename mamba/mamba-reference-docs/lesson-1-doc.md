# Lesson 1 (advanced track) — source notes and errata

Companion to *Lesson 1 — The Sequence Modelling Problem and the Mamba Ecosystem*
(delivered 2026-07-29). Everything below was verified against primary sources during
preparation. This file exists so a future session does not repeat the research, and so
the corrections do not get lost.

---

## Structure delivered

Standalone HTML, internally paged by section, academic register, prerequisite explainers
inset beside the material that needs them.

| page | section | contains |
|---|---|---|
| 0 | Cover | TOC, reading guide |
| 1 | §1 Formal setup | autoregressive factorization, loss/perplexity, **the input-dependent-memory axis** |
| 2 | §2 The recurrent family | BPTT derivation, vanishing-gradient condition, the sequential bottleneck |
| 3 | §3 The attention family | √d derivation, Table 1, KV-cache formula + PaLM validation |
| 4 | §4 The hardware layer | HBM/SRAM, arithmetic intensity, FlashAttention Thm 1/2/Prop 3 |
| 5 | §5 What theory says | Jelassi, Arora, Merrill, Sarrof, Cooper, NVIDIA empirical |
| 6 | §6 The lineage | HiPPO → Mamba-3, timeline |
| 7 | §7 The ecosystem | sibling architectures, modalities, shipped models, 2026 consensus |
| 8 | §8 MoE × Mamba | routing math, MoE-Mamba / BlackMamba / Jamba, Zamba counterexample |
| 9 | §9 Synthesis | seven claims, eight exercises |
| 10 | References | 100 entries, arXiv-linked, artefacts-without-papers listed separately |

Six interactive figures: vanishing gradients, softmax saturation, cost curves, roofline,
the copying bound, MoE routing collapse.

**Remaining lessons to build in the same format:** 2–11 of the original course, expanded.

---

## Corrections made during research — do not regress these

1. **Pascanu et al. vanishing-gradient condition.** arXiv v2 says largest *eigenvalue*;
   the published ICML 2013 / PMLR version says largest *singular value*. The published
   version is correct — the proof uses ‖W_recᵀ‖₂. **Cite PMLR, not arXiv.** It is in §2.1
   and is *not* a numbered theorem.

2. **Parity is NOT explained by the TC⁰ result.** PARITY ∈ TC⁰; Merrill et al. use it as
   their own example of an *easy* word problem. Mamba's parity failure is a finite-precision
   + non-negative-eigenvalue argument (Sarrof et al. 2024, Thm 2; Grazzi et al. 2025). The
   TC⁰ result correctly rules out **S₅ / permutation composition** only. Conflating these
   is the single most common error in secondary sources.

3. **Zamba and Zamba2 are NOT MoE models.** They are the *inverse* trade: a single
   weight-tied attention+MLP block invoked every 6 Mamba blocks — "more FLOPs at the same
   memory cost". MoE = more params, same FLOPs. Zamba = more FLOPs, same params. Pairing
   them makes the design axis explicit; miscategorising Zamba destroys that.

4. **Jamba's 128 GB KV-cache row is LLaMA-2 6.7B, not 70B.** Widely misquoted. The 7B
   figure is high because LLaMA-2 7B uses full MHA (32 KV heads) while Mistral/Mixtral use
   GQA (8). Honest decomposition of Jamba's 32× advantage: ~4× from GQA (orthogonal to
   Mamba) and ~8× from having only 4 attention layers. **Against Mixtral at equal active
   params the genuine architectural gain is 8×.**

5. **"MoE fits Mamba because Mamba is memory-bandwidth-bound" is NOT the argument the
   papers make.** BlackMamba argues the opposite — MoE *increases* memory cost. The actual
   arguments are (a) orthogonal bottlenecks (context vs parameters) and (b) vanilla Mamba
   has no FFN, so it lacks a conditional-computation slot. Decisive evidence for (b): a
   **single-expert** MoE performs *worse* than vanilla Mamba (3.75 vs 3.72), so the gain is
   from sparsity/specialisation, not from adding an FFN.

6. **Pope et al. never use the phrase "arithmetic intensity."** Use Williams et al. 2009
   for the definition and **Shazeer 2019 §2.4.1** for the quantitative decode-vs-train
   claim: Θ(n/d + 1/b) decoding versus O(1/k + 1/bn) training.

7. **A100 HBM is "40–80GB"** in the FlashAttention paper, not 40GB.

8. **FlashAttention IO-optimality** is proved over the *range* of M, not for each fixed M.
   "IO-optimal" unqualified slightly overstates it.

9. **The Jelassi bound is not what binds in practice.** Mamba-2.8B-shaped state holds
   1.678×10⁸ bits ⇒ ~10.7M copyable tokens. Observed failures occur ~10³ tokens. The gap
   is optimisation/inductive-bias, not capacity. Teaching the theorem without this makes
   a fixable problem look unfixable.

10. **Benchmark caution.** Okpekpe & Orvieto 2025: recurrent models are unusually
    learning-rate sensitive on associative-recall benchmarks, which "can severely affect
    reported performance in previous works". Direction of results is corroborated; exact
    magnitudes are less settled than citation counts suggest.

---

## Numbers recomputed (all verified in Python)

| quantity | value |
|---|---|
| PaLM 540B KV cache, b=512, L=2048 | 3.04 TB (matches published figure exactly) |
| Llama-3-70B KV, 128K ctx, GQA-8 | 42.9 GB; under MHA-64 → 343.6 GB |
| Mamba-2.8B state, E·D·(N+4)·64·2 B | 13.1 MB, constant in L |
| 2.8B-class KV at 500K ctx | 327.7 GB |
| Mamba state capacity, fp32 | 1.678×10⁸ bit ⇒ 10.7M tokens at log₂(50280)=15.62 b/tok |
| A100 roofline ridge | 312 TFLOP/s ÷ 2.0 TB/s = 156 FLOP/byte |
| η=0.9 gradient after 100 steps | 2.66×10⁻⁵; below fp32 eps after 151 steps |
| unscaled logit sd at d_k=128 | 11.31 |

---

## Key architecture facts worth not re-deriving

- **MoE-Mamba** (2401.04081): interleaves; Switch top-1, capacity factor 1, α=0.01,
  42 experts @100M. "2.35× fewer training steps" is verbatim and correct. Inside-block MoE
  was tried in all 2³−1 combinations and **lost**. Reports its own caveat: beats
  Transformer-MoE on perplexity but Transformer-MoE "consistently achieves higher accuracy."
- **BlackMamba** (2402.01771): total substitution, `x_{l+1} = x_l + MoE(LN(x_l + mamba(LN(x_l))))`,
  8 experts top-1, **Sinkhorn** routing (not an aux loss), Apache 2.0.
- **Jamba** (2403.19887): l=8, a:m=1:7, e=2, n=16, K=2 → 32 layers, 52B/12B, 256K.
- **Routing Mamba** (2506.18145): overturns MoE-Mamba's negative inside-block result by
  **sharing one router matrix** across conv/gate/output projections.
- **Swimba** (2603.06938): the taxonomy — (1) MoE over separated SSMs multiplies the
  recurrence cost; (2) MoE-parameterized SSM mixes expert params *before* the scan, one
  trajectory. This is the principled answer to "why not inside the block".
- **Granite 4.0, Nemotron 3, Qwen3-Next, Codestral Mamba have NO arXiv paper.** Cite model
  cards.
- **Qwen3-Next / Kimi Linear / MiniMax use gated linear attention, not Mamba.**

---

## 2026 consensus (as researched)

Hybrids won; pure Mamba did not. Every shipped SSM model at scale interleaves attention
(Granite 4.0 at 9:1, Nemotron-H ~8% attention, Falcon-H1 parallel heads). The only notable
pure-Mamba LLMs — Falcon Mamba 7B and Codestral Mamba — both date to mid-2024 with no
successors. The Mamba-3 authors' own framing is that linear models will predominantly be
used alongside global self-attention.

What has cooled is **Mamba specifically as the linear layer of choice**: 2026 frontier
releases favour Gated DeltaNet / KDA / Lightning Attention. The winning idea is "linear
recurrence interleaved with sparse full attention", of which Mamba-2/SSD is one
instantiation.

Live counter-current: Apple (2604.14191, Apr 2026) claims a properly initialised **pure**
Mamba distilled from a Transformer matches its teacher with zero attention blocks — 1B
scale only.