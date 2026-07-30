# Lesson 01 (advanced deep dive) — source notes, computed numbers, errata

Companion to *Lesson 01 — The Problem with Sequences*
(`mamba/lesson-01-the-problem-with-sequences.html`, built 2026-07-29).

Everything below was verified against primary sources during preparation, mostly by pulling
the LaTeXML full text and grepping for the exact statement. This file exists so a future
session does not repeat the research and does not regress the corrections.

This lesson **replaces** the earlier `lesson-1-sequence-models-and-the-mamba-ecosystem.html`
as Lesson 1 of the course map in `lesson-0-doc.md`. The old file is kept as a supplementary
ecosystem survey; its MoE and shipped-model material belongs in a later lesson, not here.

---

## Structure delivered

Single self-contained HTML, paged by section. Tailwind v4 compiled by the standalone CLI and
**inlined** (no runtime CSS dependency); KaTeX from jsDelivr with SRI hashes; all figures
hand-authored inline SVG using `var(--fig-*)` tokens so they follow light and dark mode.

| page | section | contains |
|---|---|---|
| 0 | Cover | TOC with reading times, box legend, one-sentence summary |
| 1 | §1 The task | autoregressive factorization, cross-entropy, perplexity, why averages hide recall failures |
| 2 | §2 The state | state as bottleneck, sufficient statistics, notebook/transcript figure, the cost ledger |
| 3 | §3 Recurrence | Elman net, BPTT derived in 5 lines, exponential bound derived in 5 lines, Pascanu condition, gates, sequential depth, Vaswani Table 1 |
| 4 | §4 Attention | derivation from the requirement, √d_k variance derivation, FLOP share L/(L+6d), KV cache formula, MQA/GQA |
| 5 | §5 Hardware | HBM/SRAM, arithmetic intensity, roofline + ridge point, FlashAttention Thm 2 / Prop 3, decode is bandwidth-bound |
| 6 | §6 What is lost | Jelassi Thm 2.3 / 2.7 / Cor 2.8, the 4-order-of-magnitude gap, Zoology 82%/6.4%, TC⁰ results, parity |
| 7 | §7 Design space | two axes, every family placed, lineage, hybrids |
| 8 | §8 Carry forward | nine claims, eight exercises with worked answers |
| 9 | References | 33 entries, each with a Finding line and a Matters line |

Ten figures. Six computed in Python (vanishing gradients, FLOP share, KV memory, roofline,
softmax saturation, the capacity number line); four hand-authored schematics
(notebook/transcript, memory hierarchy, dependency graphs, the design-space map).

36 references, every one cited in the body and every citation resolving; enforced by a
validator rather than by eye.

**Remaining lessons to build in the same format:** 02–11 of the course map in `lesson-0-doc.md`.

---

## Build and tooling

Project skills installed at `.cursor/skills/`:

- **`academic-research`** — plan/search/verify/reconcile/aggregate loop, plus
  `scripts/arxiv.py` (stdlib only). `arxiv.py grep <id> <pattern>` pulls the LaTeXML full
  text once, caches it in the temp dir, and prints only matching lines. This is how every
  quote in the lesson was confirmed without reading whole papers into context.
- **`lesson-html-tailwind`** — the design system (`assets/lesson.css`) and
  `scripts/build_css.py`, which compiles Tailwind against one HTML file and injects the
  result into `<style id="tw">`. Needs the Tailwind standalone binary at
  `~/.local/bin/tailwindcss` (there is no Node on this machine; the binary needs none).
  **Re-run it after any change to a `class` attribute.**
- **`professor`** — the explanation contract: casual voice, academic vocabulary, the
  five-rung ladder, prerequisites at the point of need, analogies with their seam named.

Verification recipe: headless Chrome with `--virtual-time-budget=6000` so KaTeX finishes
before the screenshot, then `sips -c H W --cropOffset TOP LEFT` to zoom (note the argument
order; getting it wrong yields a black image).

---

## Corrections made during research — do not regress these

1. **Pascanu et al. vanishing-gradient condition.** The published ICML/PMLR version says
   *"it is sufficient for λ₁ < 1/γ, where λ₁ is the largest **singular value** of W_rec"*.
   The arXiv preprint says eigenvalue. PMLR is correct: the proof runs through
   `‖W_recᵀ‖₂ = σ_max` and submultiplicativity. **Cite PMLR.** It is prose in §2.1 with the
   proof in supplementary material, **not a numbered theorem**. The linear-case statement
   *is* in terms of the spectral radius ρ, which is where the confusion starts.
   γ = 1 for tanh, γ = 1/4 for sigmoid (stated explicitly in §2.1).

2. **FlashAttention is arXiv:2205.14135, not 2205.14314.** The wrong id had propagated into
   earlier notes and resolves to an unrelated PDE paper.

3. **FlashAttention IO-optimality.** Proposition 3 says no algorithm achieves
   `o(N²d²M⁻¹)` HBM accesses **for all M in the range [d, Nd]**. Unqualified "IO-optimal"
   overstates it. The paper's own phrasing is "optimal for a range of SRAM sizes".

4. **Shazeer 2019 never writes "arithmetic intensity."** Verified: zero occurrences. Use
   Williams et al. 2009 (CACM 52(4):65–76, doi 10.1145/1498765.1498785) for the definition
   and Shazeer §2.2 / §2.4.1 for the ratios `O(1/k + 1/bn)` batched versus `Θ(n/d + 1/b)`
   incremental.

5. **Parity is NOT explained by the TC⁰ result.** Merrill et al. use PARITY as their own
   example of an *easy* word problem in TC⁰ ("a world with a single light switch"). The TC⁰
   containment rules out **NC¹-complete** problems only, canonically S₅ composition. The
   correct explanation of parity failure is Sarrof et al. Thm 2: *"No SSM satisfying
   nonnegative can recognize PARITY at arbitrary input lengths with finite precision"*, where
   nonnegative means `A(x_t) ≥ 0`, which every surveyed non-time-invariant SSM satisfies
   because of exponential or sigmoid gate parameterizations. Grazzi et al. 2025 repair it by
   allowing eigenvalues in [-1, 1].

6. **The Jelassi bound is not what binds.** Mamba-2.8B's fp32 state permits ~10.7M tokens by
   Corollary 2.8; the measured failure is at ~10³. The authors say so themselves in §3:
   *"a latent state of 1000 32-bit floating point numbers has enough bits to store at least
   2000 tokens from a 50K token vocabulary. However, even though a GSSM could fit the context
   into memory, it may not learn to do so."* Teaching the theorem without the gap makes an
   optimization problem look like a law of nature.

7. **Jelassi's empirical claim, stated precisely.** *"even the smallest transformer (410M
   parameters) outperforms the largest GSSMs (2.8B parameters) when the phone-book size is
   long enough (L ≥ 70)"* — L counts phone-book **entries**, not tokens. The "order of 10³
   tokens" in the lesson is our estimate at roughly 15 tokens per entry, and is labelled as
   such. Their length-generalization setup trains on ≤50 tokens and tests up to 1000.

8. **Zoology's numbers.** *"errors on 'AR Hits' account for 82% of the perplexity gap to
   attention on average, despite only representing 6.4% of all tokens in the Pile dataset"*,
   and *"A 70M parameter Transformer can predict AR Hits better than a 1.4Bn parameter Hyena
   gated convolution model (20× larger)"*. The 6.4% is the load-bearing half of the pair and
   is usually dropped when this result is quoted.

9. **Attention is a minority of the FLOPs at ordinary context lengths.** Share is exactly
   `L/(L+6d)` for a standard block with a 4d feedforward. Crossover at `L = 6d`. This
   contradicts the common framing and is worth stating before anyone optimizes the wrong
   thing.

10. **Vaswani et al. Table 1's restricted-self-attention row** is `O(r·n·d)` / `O(1)` /
    `O(n/r)` in the paper. Some reproductions give the sequential-ops cell as `O(n)`.
    Transcribed here as printed.

### Second pass: corrections found by dedicated verification agents

Items 11–20 came from a second, independent verification sweep after the first draft was
built. Five of them were live errors in the delivered lesson and have been fixed; the rest
were added as material.

11. **Roofline does NOT define "arithmetic intensity".** This was the worst error in the
    first draft. Williams et al. coin **operational intensity** = "operations per byte of
    DRAM traffic" and then *explicitly decline* to call it arithmetic intensity, on the
    grounds that the older term (which they attribute to Harris 2005, ACM SIGGRAPH
    tutorials) measures processor-to-cache traffic, not cache-to-DRAM. The ML systems
    literature adopted the wrong name anyway. **Cite Roofline for "operational intensity"
    and FlashAttention §2.1 for "arithmetic intensity ... the number of arithmetic
    operations per byte of memory access".** §5.2's prerequisite box now teaches the
    distinction rather than propagating the mis-citation. Also note the published CACM
    title may be simply "Roofline" (Crossref and OpenAlex both record that); the long title
    on the Berkeley preprints differs from the commonly cited one.

12. **The √d_k footnote is number 4, and it is absent from arXiv v1.** v1 says only that the
    dot products grow "too large in magnitude to result in useful gradients"; the variance
    argument was added in v2, a week later. ar5iv renumbers it as 1, which is where the
    first draft's "footnote 1" came from.

13. **The forget gate is not in Hochreiter & Schmidhuber 1997.** The 1997 cell has the CEC
    (a linear unit with self-weight pinned at exactly 1.0, from the requirement
    `f'_j(net_j) w_jj = 1.0` in §3.2.1) plus input and output gates only. The forget gate is
    **Gers, Schmidhuber & Cummins 2000**, *Learning to Forget*, Neural Computation
    12(10):2451–2471, doi 10.1162/089976600300015015. Since the forget gate is the direct
    ancestor of Mamba's Δ, getting this attribution right matters for the course. The paper
    also contains no expression `∂ϑ_j/∂ϑ_j = 1.0`; do not quote one.

14. **Shazeer's batched ratio is in §2.3.1, not §2.2**, and the `k` in `O(1/k + 1/bn)` is the
    **key depth**, not the number of KV heads. The paper assumes `k = v = d/h`, so that term
    is `h/d`. Confirmed again that "arithmetic intensity" appears zero times in the paper.

15. **NVIDIA never prints a dense BF16 figure for H100.** Every Tensor Core row on the H100
    datasheet and product page is with-sparsity: BFLOAT16 = 1,979 TFLOPS*. Dense is 989.5 by
    halving. The "989" that *is* printed is the **TF32 row, also with sparsity** — a
    coincidence that has produced many wrong slides. FlashAttention-3 is the citable source
    for the dense number ("the H100 SXM5 GPU has 989 TFLOPS of FP16 matmul"). By contrast
    the A100 datasheet prints both (312 dense / 624 sparse), and 312 is identical across
    SXM and PCIe while bandwidth is not (2,039 vs 1,935 GB/s at 80GB). Also: 3.9 TB/s is
    **H100 NVL**, not H200; H200 is 4.8 TB/s with identical compute to H100 SXM.

16. **Elman 1990 contains no recurrence equation.** None. The architecture is described
    entirely in prose and figure captions. The familiar equation is a later formalization
    and must not be presented as a quotation.

17. **Bengio et al. 1994 has no "spectral radius < 1 is necessary" theorem.** The eigenvalue
    condition is **Definition 2** (hyperbolic attractor); the latching argument is spread
    over Theorems 1–3 and the vanishing-gradient result is **Theorem 4**. The quotable
    conflict sentence is from §1: *"either such a system is stable and resistant to noise
    or, alternatively, it is efficiently trainable by gradient descent, but not both."*
    Crucially the conclusion opens *"Assuming hyperbolic attractors are used to store state
    information"* — that hypothesis is exactly the loophole the LSTM's linear CEC exploits,
    and §3.3 now makes the point explicitly.

18. **The Pascanu revision is bigger than one word.** "Singular value" appears **zero** times
    in arXiv v1 and v2 and **four** times in PMLR; three sentences changed, and PMLR renamed
    the linear-case quantity to ρ to stop it colliding with the nonlinear λ₁ (which is the
    likely origin of the error). arXiv **v1 has no vanishing-gradient condition at all** —
    it discusses only exploding gradients. Also: Pascanu et al. cite Werbos **1988**
    (Neural Networks 1(4):339–356), not Werbos 1990. Werbos 1990 remains a fine BPTT
    citation on its own, but do not pair it with "as used by Pascanu et al."

19. **The general Ω(N) state-size bound is in Based, not Zoology.** Based **Theorem 3.1**:
    "Any recurrent model depending causally on input u requires Ω(N)-bits in state size to
    solve MQAR", with a Mamba-specific Corollary F.1, proved by reduction from INDEX.
    Zoology's only Ω(N) result is **Corollary H.24, RetNet-specific**. A third route exists:
    Sanford, Hsu & Telgarsky **Theorem 11** bounds *any* bounded-memory online algorithm
    (arbitrary per-step function) at m ≥ (N−1)/2 bits, via set disjointness. §6.1 now
    presents all three, since three independent techniques reaching the same wall is the
    real argument. Caution on Based: arXiv v2 and PMLR say the recall gain over Mamba is
    **10.36** points; the arXiv abs page still shows v1's **6.22** and the auto-generated
    HTML shows a stale **9.03**.

20. **Venues, corrected.** Mamba was **rejected from ICLR 2024** (the OpenReview venue id
    literally reads `Rejected_Submission`) and **published at COLM 2024**. Wen et al. is
    **ICLR 2025**, not a preprint. Pope et al. is **MLSys 2023**. Zoology is **ICLR 2024**;
    Based is **ICML 2024 main conference**, PMLR 235:1763–1840, despite the v2 PDF carrying
    a workshop template in its footer.

### New material added in the second pass

- **Mamba-3** exists: arXiv **2603.15569** v1, **ICLR 2026 Oral**, first authors Lahoti and
  Li with Gu and Dao as senior authors. Its §4.1.2 concedes the fixed-state retrieval
  weakness in the authors' own words and states: *"we predict that linear layers will be
  predominantly used in hybrid architectures that mitigate this downside with quadratic
  self-attention layers"*, with their own hybrids at a **5:1** linear-to-attention ratio.
  This is the strongest possible source for §7.4's claim and is now quoted there.
- **MiniMax-M2** (arXiv **2605.26494**) is the counter-example: it *reverses* the hybrid
  decision, using full multi-head attention in all 62 layers, because *"we found no variant
  that reliably matches full attention quality in production settings"* and hybrid deficits
  "emerged at a larger scale" on multi-hop reasoning. §7.4 now carries it, because a lesson
  that presents only the hybrid consensus sells a live argument as a settled one.
- **Okpekpe & Orvieto 2025** (arXiv 2508.19029, unrefereed) find that recurrent models solve
  MQAR only within a very narrow learning-rate window and that Arora et al.'s grid can miss
  it, so "prior empirical expressivity comparisons may have been confounded by suboptimal
  tuning". Not yet cited in the lesson; it belongs in whichever lesson reports benchmark
  numbers, as a standing caveat.
- **Granite 4.0** is 9:1 Mamba-2 to attention per IBM's announcement and `config.json`, with
  **no arXiv paper**; **Qwen3-Next** is Gated DeltaNet at 3:1 with **no arXiv paper**;
  **Kimi Linear** is KDA at 3:1 (arXiv 2510.26692). Cite model cards for the first two.

---

## Numbers recomputed (all verified in Python, none quoted)

| quantity | value |
|---|---|
| 0.9¹⁰⁰ | 2.66×10⁻⁵ |
| 0.9¹⁵¹ | 1.23×10⁻⁷, and fp32 ε = 1.19×10⁻⁷, so it is unrepresentable by step 152 |
| 0.95ⁿ below bf16 ε (7.81×10⁻³) | n = 95 |
| 1.02²⁰⁰ | 52.5 |
| attention FLOP share | L/(L+6d); 50% crossover at L = 6d → 4,608 / 15,360 / 49,152 for d = 768 / 2560 / 8192 |
| d = 2560 at L = 1k / 4k / 16k / 128k | 6.2% / 21.1% / 51.6% / 89.5% |
| KV bytes per token, bf16 | 2·n_layer·n_kv·d_head·2 |
| Pythia-2.8B (32L, MHA 32×80) | 327,680 B/tok = 320 KB → 1.34 GB @4k, 42.9 GB @128k, 343.6 GB @1M |
| Llama-3-8B (32L, GQA-8 ×128) | 131,072 B/tok = 128 KB → 17.2 GB @128k |
| Llama-3-70B (80L, GQA-8 ×128) | 327,680 B/tok = 320 KB → 42.9 GB @128k |
| same under MHA-64 | 2,621,440 B/tok = 2.5 MB → 343.6 GB @128k |
| Mamba-2.8B state, E·D·(N+d_conv)·64·2 B | 13.1 MB, constant in L |
| ratio Llama-3-70B : Mamba-2.8B | 102× @4k, 3,277× @128k, 26,214× @1M |
| A100 80GB ridge point | 312 TFLOP/s ÷ 2.039 TB/s = **153** FLOP/byte |
| H100 SXM ridge point | 989 ÷ 3.35 = **295** FLOP/byte (worse, and the point of exercise 4) |
| decode intensity ≈ 1 attains | 2.0 TFLOP/s = 0.7% of A100 peak |
| 42.9 GB at 2.0 TB/s | 21 ms per token from bandwidth alone, under 25 tok/s |
| unscaled logit sd at d_k = 128 | √128 = 11.31 |
| mean max attention weight, L = 256, 4000 draws | unscaled 0.82 at d_k = 128 and 0.91 at 512; scaled flat at 0.043 |
| Mamba-2.8B SSM state | E·D·N·64 = 5,242,880 scalars = 1.678×10⁸ bit fp32 |
| log₂ 50,280 | 15.62 bit/token → capacity 1.07×10⁷ tokens |
| gap to observed failure (~10³) | ×10,744, about four orders of magnitude |
| bits to copy 128k tokens | 2.05×10⁶ bit ≈ 256 KB, which is 82× *less* than Mamba-2.8B holds |
| exercise 2: 40L, GQA-8×128 @32k | 163,840 B/tok → 5.37 GB, 410× the Mamba state |

Monte Carlo for the softmax figure uses common random numbers across the scaled and unscaled
curves, so the two curves differ only by the scaling.

---

## Verbatim quotes used in the lesson

- **Vaswani footnote 1** — "To illustrate why the dot products get large, assume that the
  components of q and k are independent random variables with mean 0 and variance 1. Then
  their dot product, q·k = Σᵢ qᵢkᵢ, has mean 0 and variance d_k."
- **Dao et al. §2.1** — "the A100 GPU has 40-80GB of high bandwidth memory (HBM) with
  bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors
  with bandwidth estimated around 19TB/s". (192 KB × 108 = 20.7 MB.)
- **Shazeer §2.4.1** — "When n ≈ d or b ≈ 1, the ratio is close to 1, causing memory
  bandwidth to be a major performance bottleneck on modern computing hardware."
- **Merrill et al.** — "despite their recurrent formulation, the 'state' in common SSMs is an
  illusion: S4, Mamba, and related models have similar expressiveness limitations to
  non-recurrent models like transformers".
- **Jamba** — "an 8x smaller KV cache compared to a vanilla Transformer"; block structure
  l = 8, a:m = 1:7, MoE every e = 2 layers.
- **Nemotron-H** — "roughly 8% of the total layers in the model are self-attention layers";
  4 of 52 at 8B, 10 of 118 at 56B; "up to 3× faster at inference".

---

## Known soft spots, for the proof-reader

- The "≈10³ tokens" marker in Figure 6.1 converts Jelassi's 70 phone-book **entries** into
  tokens at roughly 15 tokens per entry. The conversion is ours and the figure says so, but a
  reader could still take 10³ as a published number.
- The latency floor argument in §3.5 ("even at an optimistic 2 µs per step") is an
  order-of-magnitude illustration of kernel-launch and memory latency, not a measurement. It
  is labelled as illustrative; if a measured number can be produced, prefer it.
- Roofline marker positions in Figure 5.2 illustrate three regimes; they are not profiles of
  specific kernels.
- Hardware specifications are vendor figures, not measurements. Listed separately in the
  references under "Artefacts cited without a paper".
- Exercise 4's H100 numbers (989 TFLOP/s dense bf16, 3.35 TB/s) should be re-checked against
  the current NVIDIA datasheet before being quoted elsewhere; dense-versus-sparse and
  SXM-versus-PCIe are easy to conflate.
