# Mamba-1 — verified reference

Companion notes to the interactive course *Mamba from Scratch* (Round 1, 2026-07-29).
Everything here was checked against arXiv:2312.00752**v2** (31 May 2024) and the
`state-spaces/mamba` source. Where paper and code disagree, both are recorded.

---

## Course map (Round 1 — Mamba-1 / S6 only)

| # | Lesson | Core idea |
|---|--------|-----------|
| 1 | The problem with sequences | RNN = fixed notebook (lossy, cheap). Transformer = full transcript (exact, expensive). |
| 2 | The state | `h_t = a·h_{t-1} + b·x_t`. Past is multiplied by a<1 repeatedly, not deleted. |
| 3 | Continuous time | Bucket with a hole. A = leak, B = inflow, C = measuring stick. |
| 4 | Discretization and Δ | **Δ is a focus dial.** Ā = exp(ΔA). Large Δ → record; small Δ → ignore. |
| 5 | Going wide | N=16 states per channel at N different decay rates; A is (D,N). HiPPO init. |
| 6 | S4 / convolution trick | LTI ⇒ unroll collapses to one fixed kernel K̄ ⇒ parallel training. |
| 7 | Why S4 fails | The kernel is content-blind. Selective Copying: 18.3% → 99.8%. |
| 8 | Selection (S6) | Δ, B, C become functions of x. **This is the whole contribution.** |
| 9 | Hardware-aware scan | Associative prefix scan + kernel fusion + recomputation. |
| 10 | The Mamba block | H3 block ∪ gated MLP, repeated homogeneously. |
| 11 | Cost and results | O(L) training, O(1) per-token inference, constant state. |

Next round candidates: **Mamba-2 / SSD** (duality with linear attention), then **hybrid
architectures** (few attention layers interleaved with many Mamba layers).

---

## Equations

Continuous (Eq. 1): `h'(t) = A h(t) + B x(t)`, `y(t) = C h(t)`

ZOH discretization (Eq. 4):
```
Ā = exp(ΔA)
B̄ = (ΔA)⁻¹ (exp(ΔA) − I) · ΔB
```

Discrete recurrence (Eq. 2): `h_t = Ā h_{t-1} + B̄ x_t`, `y_t = C h_t`   (C is not overbarred)

Convolution (Eq. 3, **LTI only**): `K̄ = (CB̄, CĀB̄, …, CĀᵏB̄, …)`, `y = x ∗ K̄`

Selection (§3.2, verbatim):
```
s_B(x) = Linear_N(x)
s_C(x) = Linear_N(x)
s_Δ(x) = Broadcast_D(Linear_1(x))     # low-rank in code: Linear_D(Linear_R(x)), R = ceil(D/16)
τ_Δ    = softplus
```
A stays a plain Parameter. Paper §3.5.2: selectivity in Δ already makes (Ā, B̄) selective,
so a selective A would be redundant; "left out for simplicity."

**Theorem 1** (N=1, A=−1, B=1): `g_t = σ(Linear(x_t))`, `h_t = (1−g_t)h_{t-1} + g_t x_t`.
Mechanism: `exp(−softplus(z)) = σ(−z)` — this is *why* softplus is the right τ_Δ.

---

## Shapes — Algorithm 1 (S4) vs Algorithm 2 (S6)

| | S4 | S6 |
|---|---|---|
| A | (D, N) Parameter | (D, N) Parameter |
| B | (D, N) Parameter | **(B, L, N)** ← s_B(x) |
| C | (D, N) Parameter | **(B, L, N)** ← s_C(x) |
| Δ | (D) ← τ_Δ(Param) | **(B, L, D)** ← τ_Δ(Param + s_Δ(x)) |
| Ā, B̄ | (D, N) | **(B, L, D, N)** |
| compute | recurrence *or* convolution | **scan only** |

Note the paper overloads `B` (batch size vs SSM input matrix). Note also the learnable
bias *inside* the softplus on line 4 — easy to drop when re-deriving.

---

## Defaults and rules of thumb

| | |
|---|---|
| E (expand) | 2 — "always fixed to E=2 in our experiments" |
| N (d_state) | 16 |
| d_conv | 4, depthwise, causal |
| dt_rank | `ceil(d_model/16)` — **code default, never stated in the paper** |
| Δ init | `τ_Δ⁻¹(Uniform[0.001, 0.1])` via inverse softplus on the bias |
| blocks per "layer" | 2, to match a Transformer's 12D² |
| A init | S4D-Real, `A_n = −(n+1)`; stored `A = −exp(A_log)` ⇒ A<0 always ⇒ Ā ∈ (0,1) |
| **memory horizon** | `2.303 / (Δ·\|A\|)` steps to 10%; half-life `0.693 / (Δ·\|A\|)` |
| **params/block** | `3ED²` (2ED² in-proj + ED² out-proj) — 95.3% accurate at D=2560 |
| **inference state** | `E·D·(N + d_conv)` per layer — includes the conv state, constant in L |

Block: `out = out_proj( ( SSM(u) + D⊙u ) ⊙ SiLU(z) )`, `u = SiLU(Conv1d(x))`,
`[x, z] = in_proj(RMSNorm(input))`. Gate applied **after** the D skip.

---

## Computed figures (recomputed, not quoted)

At D=2560, E=2, N=16, d_conv=4, 64 layers, bf16:

- 3ED² = 39,321,600 vs exact block = 41,241,600 → **95.3%**
- 64 blocks + tied embedding = **2.768B** ("Mamba-3B" in the abstract = Mamba-**2.8B**)
- SSM state grid = E·D·N = **81,920** numbers/layer
- Full recurrent state = E·D·(N+4)·64·2 B = **13.1 MB, constant**
- KV cache: 0.66 GB @ L=1k · 6.55 GB @ 10k · 65.5 GB @ 100k · 655 GB @ 1M
- Ratio at L=100k: **5,000×**

Impulse response, A=−1 B=1 C=1 Δ=0.5: Ā=0.6065, B̄=0.3935 (= 1−Ā), then
0.3935, 0.2387, 0.1447, 0.0878, 0.0533 — identical to CĀᵏB̄, i.e. the state *is* the kernel.

---

## Paper vs code — four discrepancies

1. **B̄ is Euler, not ZOH, in the released code.** Kernel computes `B̄ = ΔB` while keeping
   exact ZOH for Ā. Unacknowledged in the paper; confirmed by Albert Gu in
   `state-spaces/mamba` issue #114 ("a mix of ZOH for A and Euler discretization for B").
   *Consequence:* Theorem 1's clean convex combination relies on the full ZOH B̄. Under the
   shipped kernel you get `h_t = (1−g_t)h_{t-1} + softplus(·)x_t`. So "Mamba **is** a gated
   RNN" is a claim about the paper's math, not exactly about the implementation.
2. **D skip term is in the code, not the paper's equations.** `D = ones(d_inner)`,
   `y_t = C h_t + D x_t`, no weight decay, kept fp32.
3. **The "optional normalization layer" of §3.4 does not exist in `mamba_simple.py`.**
   Mamba-*2* adds a gated RMSNorm before out_proj. A block diagram with an internal norm
   is Mamba-2, not Mamba-1.
4. Minor: paper writes S4D-Lin as `−½ + ni`, dropping the π from S4D's `−½ + iπn`; and the
   fused-kernel IO is quoted `O(BLD+DN)` forward but `O(BLN+DN)` backward — a D/N swap that
   looks like a typo in one of the two.

---

## Empirical claims, with their sources

| claim | figure | where |
|---|---|---|
| Selective Copying: S4 / S4-in-Mamba-block / S6 | 18.3 / 56.4 / **99.8%** | Fig. 4 |
| Induction heads, trained at L=256 | perfect at **L = 2²⁰ = 1,048,576** ("4000× longer"); no other method beyond 2× | §4.1.2 |
| IO reduction from fusion | factor of O(N) → **20–40× speedup** | §4.5, App. D |
| Faster than FlashAttention-2 beyond | L ≈ 2,000 | §4.5 |
| Faster than attention at L=32K | up to 7× | App. D |
| Inference throughput vs Transformer | **4–5×** (§4.5); abstract/Fig. 12 say 5× | §4.5 |
| Mamba-2.8B vs Pythia-6.9B, zero-shot avg | 63.3 vs 61.7 | Table 1 |

Attention baselines on induction heads were only evaluated to L=2¹⁴ (OOM beyond).
Per-length cells of Table 11 were not reliably extractable — do not cite individual cells.

---

## The honest limitation

A Transformer never decides what to remember; it keeps everything, so exact lookup of an
arbitrary past token always works. Mamba's fixed-size state means information genuinely can
be lost. Selection makes the loss *intelligent* rather than arbitrary — it does not make it
zero. This is the reason hybrid architectures (a few attention layers among many Mamba
layers) beat either pure design.