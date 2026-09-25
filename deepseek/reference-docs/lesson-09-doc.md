# 第 09 课研究笔记：DeepSeek-V2（上）——多头潜在注意力 MLA

> 本文件是 `deepseek/src/lesson-09-v2-mla.md` 的研究底稿，只收“核对过的引文、复算过的数字、勘误、未能核实清单”，
> 不是讲义正文。建立于 2026-09-24。所有引文保留英文原文，并注明节号 / 表号 / 式号 / 所读版本。
> 所有派生数字都在 Python 里复算过（第 4 节给出表达式）。凡是笔者自己的推理，一律标注“笔者推测”或“笔者推导”。
> 与第 10 课笔记 `lesson-10-doc.md` 共用同一批一手资料；MoE、训练、YaRN、对齐放在第 10 课笔记里。

---

## 0. 覆盖范围与所读版本

### 0.1 论文

| 短键 | 标题 | 作者 | arXiv / 版本 | 发表状态 | 日期 | 本课用途 |
|---|---|---|---|---|---|---|
| `dsv2` | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek-AI（arXiv 元数据：“DeepSeek-AI, Aixin Liu, Bei Feng, …”，另加 151 人） | 2405.04434，**读 v5**，并对 v1–v4 全文做了 diff | 预印本（技术报告） | v1 2024-05-07；v5 2024-06-19 | MLA 全部公式、表 1、图 3、附录 C、附录 D |
| `dsv3` | DeepSeek-V3 Technical Report | DeepSeek-AI | 2412.19437v2 | 预印本 | 2024-12-27（v1） | V3 的 MLA 超参数（61 层、d=7168，其余同 V2）；缓存哪两个向量 |
| `dsv32` | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models | DeepSeek-AI | 2512.02556v1 | 预印本 | 2025-12-02 | 附录 A：MLA 的 MHA 模式与 MQA 模式 |
| `isca25` | Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures | Chenggang Zhao 等 | 2505.09343v2 | ISCA '25（第 52 届 ISCA，2025-06） | 2025-05-14 | 表 1：V3 / Qwen-2.5 72B / LLaMA-3.1 405B 每 token KV 字节 |
| `dsllm` | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism | DeepSeek-AI | 2401.02954 | 预印本 | 2024-01 | 表 2：67B = 95 层、d=8192、64 头、8 个 KV 头（GQA） |
| `roformer` | RoFormer: Enhanced Transformer with Rotary Position Embedding | Jianlin Su 等 | 2104.09864v5 | Neurocomputing（期刊版未在本次核对） | 2021-04-20（v1） | 式 (16)：$\mathbf R_{\Theta,n-m}=\mathbf R_{\Theta,m}^\top\mathbf R_{\Theta,n}$ |
| `mqa` | Fast Transformer Decoding: One Write-Head is All You Need | Noam Shazeer | 1911.02150v1 | 预印本 | 2019-11-06 | MQA 出处（V2 引用） |
| `gqa` | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Joshua Ainslie 等 | 2305.13245v3 | EMNLP 2023（未在本次核对出版页） | 2023-05-22 | GQA 出处（V2 引用） |

### 0.2 制品（无论文，`kind: artefact`）

| 短键 | 内容 | 读取方式 / 时间 |
|---|---|---|
| `hf-v2-config` | `huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json` | 2026-09-24 原样下载 |
| `hf-v2-code` | 同仓库 `modeling_deepseek.py`（HF 参考实现） | 2026-09-24 原样下载，逐行读了 `DeepseekV2Attention` |
| `hf-v2lite-config` / `hf-v3-config` / `hf-67b-config` / `hf-moe16b-config` / `hf-7b-config` | 各模型 config.json | 同上 |
| `gh-v2` | `github.com/deepseek-ai/DeepSeek-V2` README | 同上 |
| `flashmla` | `github.com/deepseek-ai/FlashMLA` README（当前版，含 2026-09-10 的 V4.1 内核新闻）+ 首次提交 `414a2f3eed`（2025-02-21）的 README + `docs/20250422-new-kernel-deep-dive.md` | GitHub API 取提交历史，raw 取文件 |
| `openinfra` | `github.com/deepseek-ai/open-infra-index` README（Open-Source Week，Day 1 = FlashMLA） | 同上 |

### 0.3 DeepSeek-V2 论文的版本史（全文 diff 结果）

| 版本 | 日期（HTML 页脚） | 与前一版的差异（本课相关的部分用粗体） |
|---|---|---|
| v1 | 2024-05-07 | 初版。附录 B = MLA 全公式，C = 注意力消融（C.1、C.2） |
| v2 | 2024-05-08 | 正文与 v1 逐字相同（只有日期变） |
| v3 | 2024-05-16 | **新增附录 B“DeepSeek-V2-Lite”**，原附录整体后移一位：MLA 全公式变为**附录 C**，消融变为**附录 D（D.1、D.2）**；引言加一段 V2-Lite 介绍；表号相应后移（原表 6/7 → 表 8/9）；个别措辞（“delve deeper into” → “explore”） |
| v4 | 2024-05-24 | V2-Lite Chat 的描述从 “long-context extension, SFT, and RL” 改为 “long context extension and SFT”（即 **Lite Chat 没有 RL**） |
| v5 | 2024-06-19 | **附录 C 最后一句被改写**：v1–v4 为 “Since this optimization is related to only model parameters, it can be completed offline at once.”；v5 为 “Therefore, we do not need to compute keys and values out for each query.” |

- **写课时请引用 v5 的节号**：MLA 全公式 = 附录 C（式 37–47），消融 = 附录 D.1（表 8）/ D.2（表 9）。
  很多二手资料引用“附录 B / 附录 C.2 / 表 6”，那是 v1/v2 的编号。
- v5 改写附录 C 那句话的原因论文没说。**笔者推测**：旧句暗示“把 $W^{UK}$ 与 $W^{UQ}$ 事先乘成一个矩阵、离线完成”，
  但第 4.4 节的复算显示，事先乘好的矩阵比分开存大 3 倍、每 token FLOPs 也多 3 倍；真正的实现是“改变乘法顺序”
  （结合律），而不是“离线预乘”。新句只说“不必为每个查询把 K、V 算出来”，更准确。

---

## 1. 核心论断与原文引文

以下均引自 `dsv2` v5，除非另注。

**C1. MHA 的缓存是推理瓶颈；MQA/GQA 省缓存但掉点。**（§2.1）
> "Conventional Transformer models usually adopts Multi-Head Attention (MHA) (Vaswani et al., 2017), but during generation, its heavy Key-Value (KV) cache will become the bottleneck that limit the inference efficiency."

> "They require a smaller magnitude of KV cache, but their performance does not match MHA (we provide the ablation of MHA, GQA and MQA in Appendix D.1)."

**C2. MHA 每 token 缓存 $2n_hd_hl$ 个元素。**（§2.1.1，式 (8) 之后）
> "During inference, all keys and values need to be cached to accelerate inference, so MHA needs to cache $2n_{h}d_{h}l$ elements for each token. In model deployment, this heavy KV cache is a large bottleneck that limits the maximum batch size and sequence length."

**C3. MLA 的核心是 K、V 的低秩联合压缩；推理时只缓存 $\mathbf c_t^{KV}$。**（§2.1.2）
> "The core of MLA is the low-rank joint compression for keys and values to reduce KV cache"

> "During inference, MLA only needs to cache $\mathbf{c}_{t}^{KV}$, so its KV cache has only $d_{c}l$ elements, where $l$ denotes the number of layers."

**C4. 吸收论证（正文版本：吸收进 $W^Q$）。**（§2.1.2）
> "In addition, during inference, since $W^{UK}$ can be absorbed into $W^{Q}$, and $W^{UV}$ can be absorbed into $W^{O}$, we even do not need to compute keys and values out for attention."

**C5. 吸收论证（附录版本：吸收进 $W^{UQ}$，依据是矩阵乘法结合律）。**（附录 C，式 (47) 之后，v5 原文）
> "During inference, the naive formula needs to recover $\mathbf{k}_{t}^{C}$ and $\mathbf{v}_{t}^{C}$ from $\mathbf{c}_{t}^{KV}$ for attention. Fortunately, due to the associative law of matrix multiplication, we can absorb $W^{UK}$ into $W^{UQ}$, and $W^{UV}$ into $W^{O}$. Therefore, we do not need to compute keys and values out for each query. Through this optimization, we avoid the computational overhead for recomputing $\mathbf{k}_{t}^{C}$ and $\mathbf{v}_{t}^{C}$ during inference."

  注意正文说 “into $W^{Q}$”，附录说 “into $W^{UQ}$”：V2 有查询压缩，所以 $W^Q$ 实际不存在，正确的是 $W^{UQ}$。
  （V2-Lite 没有查询压缩，对它来说“吸收进 $W^Q$”才字面成立。）

**C6. 查询也做低秩压缩，目的是训练时省激活显存，不省 KV cache。**（§2.1.2）
> "Moreover, in order to reduce the activation memory during training, we also perform low-rank compression for the queries, even if it cannot reduce the KV cache"

**C7. RoPE 与低秩压缩不兼容（论文原话，必须整段引）。**（§2.1.3）
> "However, RoPE is incompatible with low-rank KV compression. To be specific, RoPE is position-sensitive for both keys and queries. If we apply RoPE for the keys $\mathbf{k}_{t}^{C}$, $W^{UK}$ in Equation 10 will be coupled with a position-sensitive RoPE matrix. In this way, $W^{UK}$ cannot be absorbed into $W^{Q}$ any more during inference, since a RoPE matrix related to the currently generating token will lie between $W^{Q}$ and $W^{UK}$ and matrix multiplication does not obey a commutative law. As a result, we must recompute the keys for all the prefix tokens during inference, which will significantly hinder the inference efficiency."

**C8. 解耦 RoPE：多头的 $\mathbf q_{t,i}^R$ + 一个所有头共享的 $\mathbf k_t^R$。**（§2.1.3）
> "As a solution, we propose the decoupled RoPE strategy that uses additional multi-head queries $\mathbf{q}_{t,i}^{R}\in\mathbb{R}^{d_{h}^{R}}$ and a shared key $\mathbf{k}_{t}^{R}\in\mathbb{R}^{d_{h}^{R}}$ to carry RoPE, where $d_{h}^{R}$ denotes the per-head dimension of the decoupled queries and key."

**C9. 解耦键也要缓存，总缓存 $(d_c+d_h^R)l$。**（§2.1.3，式 (19) 之后）
> "During inference, the decoupled key should also be cached. Therefore, DeepSeek-V2 requires a total KV cache containing $(d_{c}+d_{h}^{R})l$ elements."

**C10. 表 1 与“相当于 2.25 组的 GQA”。**（表 1 标题 + §2.1.4）
> Table 1 caption: "Comparison of the KV cache per token among different attention mechanisms. [...] The amount of KV cache is measured by the number of elements, regardless of the storage precision. For DeepSeek-V2, $d_{c}$ is set to $4d_{h}$ and $d_{h}^{R}$ is set to $\frac{d_{h}}{2}$. So, its KV cache is equal to GQA with only 2.25 groups, but its performance is stronger than MHA."

> §2.1.4: "MLA requires only a small amount of KV cache, equal to GQA with only 2.25 groups, but can achieve stronger performance than MHA."

  表 1 中“按元素计数、与存储精度无关”这半句常被漏掉，而它正是第 7 节勘误 E1（93.3%）的关键。

**C11. 图 3 的标题。**（图 3）
> "Simplified illustration of Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and Multi-head Latent Attention (MLA). Through jointly compressing the keys and values into a latent vector, MLA significantly reduces the KV cache during inference."

**C12. V2 的 MLA 超参数。**（§3.1.2 “Model Hyper-Parameters”）
> "We set the number of Transformer layers to 60 and the hidden dimension to 5120. All learnable parameters are randomly initialized with a standard deviation of 0.006. In MLA, we set the number of attention heads $n_{h}$ to 128 and the per-head dimension $d_{h}$ to 128. The KV compression dimension $d_{c}$ is set to 512, and the query compression dimension $d_{c}^{\prime}$ is set to 1536. For the decoupled queries and key, we set the per-head dimension $d_{h}^{R}$ to 64."

**C13. 压缩向量后加 RMSNorm，并在“宽度瓶颈”处乘额外缩放因子。**（§3.1.2）
> "In addition, the low-rank compression and fine-grained expert segmentation will impact the output scale of a layer. Therefore, in practice, we employ additional RMS Norm layers after the compressed latent vectors, and multiply additional scaling factors at the width bottlenecks (i.e., the compressed latent vectors and the intermediate hidden states of routed experts) to ensure stable training."

**C14. MLA 内核基于改进版 FlashAttention-2。**（§3.1.3）
> "In addition, MLA is also optimized based on an improved version of FlashAttention-2 (Dao, 2023)."

**C15. 部署：FP8 权重 + 平均 6 bit 的 KV cache 量化。**（§3.2.3 “Inference Efficiency”）
> "In order to efficiently deploy DeepSeek-V2 for service, we first convert its parameters into the precision of FP8. In addition, we also perform KV cache quantization (Hooper et al., 2024; Zhao et al., 2023) for DeepSeek-V2 to further compress each element in its KV cache into 6 bits on average. Benefiting from MLA and these optimizations, actually deployed DeepSeek-V2 requires significantly less KV cache than DeepSeek 67B, and thus can serve a much larger batch size."

**C16. 头条数字（摘要原话；引言写作 “as shown in Figure 1, compared with DeepSeek 67B, …”，第 5 节写作 “In practice, compared with DeepSeek 67B, …”，数字相同）。**
> "Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times."

**C17. 吞吐测量条件。**（§3.2.3）
> "On a single node with 8 H800 GPUs, DeepSeek-V2 achieves a generation throughput exceeding 50K tokens per second, which is 5.76 times the maximum generation throughput of DeepSeek 67B. In addition, the prompt input throughput of DeepSeek-V2 exceeds 100K tokens per second."

**C18. 附录 D.1：7B 稠密模型上 MHA 显著优于 GQA、MQA。**（附录 D.1）
> "All of these three models are trained on 1.33T tokens, and share the same architecture except for the attention mechanisms. In addition, for a fair comparison, we align the number of parameters of them to around 7B by adjusting the number of layers. From the table, we can find that MHA demonstrates significant advantages over GQA and MQA on these benchmarks."

**C19. 附录 D.2：两个规模上 MLA 优于 MHA，缓存只有 14% / 4%。**（附录 D.2）
> "Two small MoE models comprise about 16B total parameters, and we train them on 1.33T tokens. Two large MoE models comprise about 250B total parameters, and we train them on 420B tokens. [...] From the table, we can observe that MLA shows better performance than MHA. More importantly, MLA requires a significantly smaller amount of KV cache (14% for small MoE models and 4% for large MoE models) than MHA."

**C20. MLA 的署名贡献者。**（附录 A 末尾）
> "Especially, Huazuo Gao and Wangding Zeng have made key innovations in the research of the MLA architecture. Furthermore, we'd like to thank Jianlin Su for his helpful discussion on position embedding."

  （Jianlin Su 即 RoFormer / RoPE 的第一作者；这是“解耦 RoPE”设计背景的一条旁证。）

**C21. V3 沿用同一 MLA，只缓存 $\mathbf c_t^{KV}$ 与 $\mathbf k_t^R$。**（`dsv3` §2.1.1）
> "Note that for MLA, only the blue-boxed vectors (i.e., $\mathbf{c}_{t}^{KV}$ and $\mathbf{k}_{t}^{R}$) need to be cached during generation, which results in significantly reduced KV cache while maintaining performance comparable to standard Multi-Head Attention (MHA) (Vaswani et al., 2017)."

  注意措辞变化：V2 说 MLA “stronger than MHA”，V3 说 “comparable to standard MHA”。

**C22. V3 的 MLA 超参数。**（`dsv3` §4.2）
> "We set the number of Transformer layers to 61 and the hidden dimension to 7168. All learnable parameters are randomly initialized with a standard deviation of 0.006. In MLA, we set the number of attention heads $n_{h}$ to 128 and the per-head dimension $d_{h}$ to 128. The KV compression dimension $d_{c}$ is set to 512, and the query compression dimension $d_{c}^{\prime}$ is set to 1536. For the decoupled queries and key, we set the per-head dimension $d_{h}^{R}$ to 64."

**C23. MLA 的两种计算模式：训练/预填充用 MHA 模式，解码用 MQA 模式。**（`dsv32` 附录 A，图 7 标题；§2 “Instantiate DSA Under MLA”）
> "For DeepSeek-V3.1-Terminus, the MHA mode is used for training and prefilling, while the MQA mode is used for decoding."

> "At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency (Yuan et al., 2025). Therefore, we implement DSA based on the MQA (Shazeer, 2019) mode of MLA"

**C24. 潜向量的投影矩阵与模型一起训练（不是事后对 MHA 做 SVD）。**（`isca25` §2.1.2）
> "we employ Multi-head Latent Attention (MLA) (DeepSeek-AI, 2024c) that compresses the KV representations of all attention heads into a smaller latent vector using a projection matrix, which is jointly trained with the model. During inference, only the latent vector needs to be cached"

**C25. ISCA 论文表 1（BF16）。**（`isca25` 表 1）
> "KV cache size comparison (BF16 precision): DeepSeek-V3 (MLA) largely reduces KV cache size compared to other models using GQA."
> 表内：DeepSeek-V3 (MLA) 70.272 KB 1x；Qwen-2.5 72B (GQA) 327.680 KB 4.66x；LLaMA-3.1 405B (GQA) 516.096 KB 7.28x。
> （**7.28x 是算错的**，应为 7.34x，见勘误 E7。）

**C26. FlashMLA：解码内核为什么是算力受限。**（`flashmla` deep-dive 博客 2025-04-22，制品）
> "Thus, the compute-memory ratio is $h_q s_q \cdot \frac{d_k+d_v}{d_k} \approx 2 h_q s_q$."
> "An NVIDIA H800 SXM5 GPU has a peak memory bandwidth of 3.35 TB/s and peak FLOPs of 990 TFlops. However, due to throttling (reducing to ~1600 MHz in our case), the practical peak FLOPs drops to ~865 TFlops. Therefore, when $h_qs_q \ge \frac{1}{2} \cdot \frac{865}{3.35} = 128$, the kernel is compute-bound; otherwise, it's memory-bound."
> "we don't use Tensor Parallel for decoding instances, meaning $h_q$ is 128 and the kernel is compute-bound."

**C27. FlashMLA 首发性能与“MQA 模式”的形状。**（`flashmla`，制品）
- 首次提交 README（`414a2f3eed`，2025-02-21）：“Achieving up to 3000 GB/s in memory-bound configuration and 580 TFLOPS in computation-bound configuration on H800 SXM5, using CUDA 12.6.”；“BF16”；“Paged kvcache with block size of 64”。
- `openinfra`：Open-Source Week “Starting this week, Feb 24, 2025”，Day 1 = FlashMLA。
- 当前 README：“The dense MLA decoding kernel achieves up to 3000 GB/s in memory-bound configuration and 660 TFLOPS in computation-bound configuration on H800 SXM5 with CUDA 12.8.”
- 当前 README 注 [1]：“MQA stands for Multi-Query Attention mode (i.e. `head_dim_k` = 576 (for DeepSeek V3/V3.1/V3.2) or 512 (for DeepSeek V4/V4.1) with `head_dim_v` = 512), while MHA stands for Multi-Head Attention mode (i.e. `head_dim_k` = 192 / 128 with `head_dim_v` = 128).”
- V3.2 的 FP8 KV 格式：每 token 一行 656 字节 = 512 字节 `float8_e4m3`（NoPE 部分）+ 16 字节（4 个 `float32` 缩放因子，每个管 128 个值）+ 128 字节（64 个 `bfloat16` 的 RoPE 部分，“not quantized for accuracy”）。
- “the dense decoding kernel reads bf16 / fp16 KV caches”（FP8 KV 只和稀疏解码一起支持）。

**C28. HF 参考实现比内部实现慢（官方自己承认）。**（`gh-v2` README §3）
> "Due to the constraints of HuggingFace, the open-source code currently experiences slower performance than our internal codebase when running on GPUs with Huggingface. To facilitate the efficient execution of our model, we offer a dedicated vllm solution that optimizes performance for running our model effectively."

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 背景：标准 MHA，式 (1)–(8)（§2.1.1，照录）

$$\mathbf{q}_{t}=W^{Q}\mathbf{h}_{t},\quad(1)\qquad \mathbf{k}_{t}=W^{K}\mathbf{h}_{t},\quad(2)\qquad \mathbf{v}_{t}=W^{V}\mathbf{h}_{t},\quad(3)$$

$$[\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}]=\mathbf{q}_{t},\quad(4)\quad
[\mathbf{k}_{t,1};\mathbf{k}_{t,2};...;\mathbf{k}_{t,n_{h}}]=\mathbf{k}_{t},\quad(5)\quad
[\mathbf{v}_{t,1};\mathbf{v}_{t,2};...;\mathbf{v}_{t,n_{h}}]=\mathbf{v}_{t},\quad(6)$$

$$\mathbf{o}_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_{j}(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}}})\mathbf{v}_{j,i},\quad(7)$$

$$\mathbf{u}_{t}=W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};...;\mathbf{o}_{t,n_{h}}],\quad(8)$$

形状（原文）：$\mathbf h_t\in\mathbb R^d$；$\mathbf q_t,\mathbf k_t,\mathbf v_t\in\mathbb R^{d_hn_h}$；$W^Q,W^K,W^V\in\mathbb R^{d_hn_h\times d}$；
$\mathbf q_{t,i},\mathbf k_{t,i},\mathbf v_{t,i}\in\mathbb R^{d_h}$；$W^O\in\mathbb R^{d\times d_hn_h}$。

### 2.2 低秩 KV 联合压缩，式 (9)–(11)（§2.1.2，照录）

$$\mathbf{c}_{t}^{KV}=W^{DKV}\mathbf{h}_{t},\quad(9)$$
$$\mathbf{k}_{t}^{C}=W^{UK}\mathbf{c}_{t}^{KV},\quad(10)$$
$$\mathbf{v}_{t}^{C}=W^{UV}\mathbf{c}_{t}^{KV},\quad(11)$$

原文形状：$\mathbf c_t^{KV}\in\mathbb R^{d_c}$，“$d_{c}(\ll d_{h}n_{h})$ denotes the KV compression dimension”；
$W^{DKV}\in\mathbb R^{d_c\times d}$（down-projection）；$W^{UK},W^{UV}\in\mathbb R^{d_hn_h\times d_c}$（up-projection）。
V2 取值：$d_c=512$，$d_hn_h=16384$，所以 $d_c$ 是 $d_hn_h$ 的 1/32。

### 2.3 查询的低秩压缩，式 (12)–(13)（§2.1.2，照录）

$$\mathbf{c}_{t}^{Q}=W^{DQ}\mathbf{h}_{t},\quad(12)$$
$$\mathbf{q}_{t}^{C}=W^{UQ}\mathbf{c}_{t}^{Q},\quad(13)$$

原文形状：$\mathbf c_t^Q\in\mathbb R^{d_c'}$；“$d_{c}^{\prime}(\ll d_{h}n_{h})$ denotes the query compression dimension”；
$W^{DQ}\in\mathbb R^{d_c'\times d}$，$W^{UQ}\in\mathbb R^{d_hn_h\times d_c'}$。V2：$d_c'=1536$。V2-Lite 不压缩查询。

### 2.4 解耦 RoPE 与完整注意力，式 (14)–(19)（§2.1.3，照录）

$$[\mathbf{q}_{t,1}^{R};\mathbf{q}_{t,2}^{R};...;\mathbf{q}_{t,n_{h}}^{R}]=\mathbf{q}_{t}^{R}=\operatorname{RoPE}({W^{QR}}\mathbf{c}_{t}^{Q}),\quad(14)$$
$$\mathbf{k}_{t}^{R}=\operatorname{RoPE}({W^{KR}}\mathbf{h}_{t}),\quad(15)$$
$$\mathbf{q}_{t,i}=[\mathbf{q}_{t,i}^{C};\mathbf{q}_{t,i}^{R}],\quad(16)$$
$$\mathbf{k}_{t,i}=[\mathbf{k}_{t,i}^{C};\mathbf{k}_{t}^{R}],\quad(17)$$
$$\mathbf{o}_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_{j}(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}+d_{h}^{R}}})\mathbf{v}_{j,i}^{C},\quad(18)$$
$$\mathbf{u}_{t}=W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};...;\mathbf{o}_{t,n_{h}}],\quad(19)$$

原文形状：$W^{QR}\in\mathbb R^{d_h^Rn_h\times d_c'}$，$W^{KR}\in\mathbb R^{d_h^R\times d}$；“$\operatorname{RoPE}(\cdot)$ denotes the operation that applies RoPE matrices; and $[\cdot;\cdot]$ denotes the concatenation operation.”

要点（写课时必须讲清）：
- 式 (15) 的 $\mathbf k_t^R$ **直接从 $\mathbf h_t$ 算**，不经过 $\mathbf c_t^{KV}$；**只有一个**，被所有头共享（式 (17) 里没有下标 $i$）。
- 式 (14) 的 $\mathbf q_t^R$ 从 $\mathbf c_t^Q$ 算，**每头一个**。
- 每头的查询/键维数 = $d_h+d_h^R=128+64=192$；值的维数仍是 $d_h=128$。所以式 (18) 的缩放是 $1/\sqrt{192}\approx0.07217$。
- 式 (17) 里的 $\mathbf k_{t,i}^C$、式 (18) 里的 $\mathbf v_{j,i}^C$ 是 $\mathbf k_t^C,\mathbf v_t^C$ 按头切片，正文没有单独写切片式，附录 C 的式 (42)(45) 写了。

### 2.5 附录 C 全公式，式 (37)–(47)（照录；方框 = 缓存）

$$\mathbf{c}_{t}^{Q}=W^{DQ}\mathbf{h}_{t},\quad(37)$$
$$[\mathbf{q}_{t,1}^{C};\mathbf{q}_{t,2}^{C};...;\mathbf{q}_{t,n_{h}}^{C}]=\mathbf{q}_{t}^{C}=W^{UQ}\mathbf{c}_{t}^{Q},\quad(38)$$
$$[\mathbf{q}_{t,1}^{R};\mathbf{q}_{t,2}^{R};...;\mathbf{q}_{t,n_{h}}^{R}]=\mathbf{q}_{t}^{R}=\operatorname{RoPE}({W^{QR}}\mathbf{c}_{t}^{Q}),\quad(39)$$
$$\mathbf{q}_{t,i}=[\mathbf{q}_{t,i}^{C};\mathbf{q}_{t,i}^{R}],\quad(40)$$
$$\boxed{\color{blue}\mathbf{c}_{t}^{KV}}=W^{DKV}\mathbf{h}_{t},\quad(41)$$
$$[\mathbf{k}_{t,1}^{C};\mathbf{k}_{t,2}^{C};...;\mathbf{k}_{t,n_{h}}^{C}]=\mathbf{k}_{t}^{C}=W^{UK}\mathbf{c}_{t}^{KV},\quad(42)$$
$$\boxed{\color{blue}\mathbf{k}_{t}^{R}}=\operatorname{RoPE}({W^{KR}}\mathbf{h}_{t}),\quad(43)$$
$$\mathbf{k}_{t,i}=[\mathbf{k}_{t,i}^{C};\mathbf{k}_{t}^{R}],\quad(44)$$
$$[\mathbf{v}_{t,1}^{C};\mathbf{v}_{t,2}^{C};...;\mathbf{v}_{t,n_{h}}^{C}]=\mathbf{v}_{t}^{C}=W^{UV}\mathbf{c}_{t}^{KV},\quad(45)$$
$$\mathbf{o}_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_{j}(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}+d_{h}^{R}}})\mathbf{v}_{j,i}^{C},\quad(46)$$
$$\mathbf{u}_{t}=W^{O}[\mathbf{o}_{t,1};\mathbf{o}_{t,2};...;\mathbf{o}_{t,n_{h}}],\quad(47)$$

原文：“where the boxed vectors in blue need to be cached for generation.”（原文用 `\boxed{\color[rgb]{0,0,1}...}`。）

### 2.6 表 1 的四行（照录）

| Attention Mechanism | KV Cache per Token (# Element) | Capability |
|---|---|---|
| Multi-Head Attention (MHA) | $2n_{h}d_{h}l$ | Strong |
| Grouped-Query Attention (GQA) | $2n_{g}d_{h}l$ | Moderate |
| Multi-Query Attention (MQA) | $2d_{h}l$ | Weak |
| MLA (Ours) | $(d_{c}+d_{h}^{R})l\approx\frac{9}{2}d_{h}l$ | Stronger |

推导（笔者补全）：$d_c=4d_h$、$d_h^R=d_h/2$ ⇒ $d_c+d_h^R=\tfrac92 d_h$；令 $2n_gd_hl=\tfrac92d_hl$ ⇒ $n_g=2.25$。

### 2.7 吸收的逐行推导（笔者推导，供讲义 `::: deriv` 使用）

记号（课程补充，论文没有）：把 $W^{UK}$ 按行切成 $n_h$ 块，$W^{UK}_i\in\R^{d_h\times d_c}$ 是第 $i$ 头的那 $d_h$ 行；
$W^{UV}_i$ 同理；$W^{UQ}_i\in\R^{d_h\times d_c'}$；$W^{O}$ 按列切成 $W^O_i\in\R^{d\times d_h}$。

1. 内容部分的打分：
$$(\mathbf q^C_{t,i})^\top\mathbf k^C_{j,i}=(\mathbf q^C_{t,i})^\top W^{UK}_i\mathbf c_j^{KV}=\big((W^{UK}_i)^\top\mathbf q^C_{t,i}\big)^\top\mathbf c_j^{KV}\equiv\tilde{\mathbf q}_{t,i}^\top\mathbf c_j^{KV},\qquad \tilde{\mathbf q}_{t,i}\in\R^{d_c}.$$
   理由：矩阵乘法结合律；$\tilde{\mathbf q}_{t,i}$ 只依赖当前 token，与 $j$ 无关，每步只算一次。
2. 进一步 $\tilde{\mathbf q}_{t,i}=(W^{UK}_i)^\top W^{UQ}_i\mathbf c_t^Q$。“吸收进 $W^{UQ}$”的字面意思是把 $(W^{UK}_i)^\top W^{UQ}_i\in\R^{d_c\times d_c'}$ 预乘好；实践中是**先算 $\mathbf q^C_{t,i}$ 再左乘 $(W^{UK}_i)^\top$**（见 4.4 节：预乘反而更贵）。
3. 完整打分（加上解耦 RoPE 部分）：
$$\mathbf q_{t,i}^\top\mathbf k_{j,i}=\tilde{\mathbf q}_{t,i}^\top\mathbf c_j^{KV}+(\mathbf q^R_{t,i})^\top\mathbf k^R_j.$$
4. 值与输出：
$$\mathbf o_{t,i}=\sum_j p_{t,j,i}W^{UV}_i\mathbf c_j^{KV}=W^{UV}_i\Big(\sum_j p_{t,j,i}\mathbf c_j^{KV}\Big)\equiv W^{UV}_i\tilde{\mathbf o}_{t,i},\qquad\tilde{\mathbf o}_{t,i}\in\R^{d_c}.$$
   理由：$W^{UV}_i$ 与 $j$ 无关，可提到求和号外（线性）。
5. $\mathbf u_t=\sum_iW^O_iW^{UV}_i\tilde{\mathbf o}_{t,i}$。“吸收进 $W^O$”= 把 $W^O_iW^{UV}_i\in\R^{d\times d_c}$ 当作一个矩阵；同样，实践中常分两步乘（4.4 节）。
6. **MQA 形式**：定义共享的键 $\hat{\mathbf k}_j=[\mathbf c_j^{KV};\mathbf k_j^R]\in\R^{576}$、共享的值 $\hat{\mathbf v}_j=\mathbf c_j^{KV}\in\R^{512}$、
   每头查询 $\hat{\mathbf q}_{t,i}=[\tilde{\mathbf q}_{t,i};\mathbf q^R_{t,i}]\in\R^{576}$，则
   $$p_{t,j,i}=\softmax_j\Big(\frac{\hat{\mathbf q}_{t,i}^\top\hat{\mathbf k}_j}{\sqrt{d_h+d_h^R}}\Big),\qquad \tilde{\mathbf o}_{t,i}=\sum_jp_{t,j,i}\hat{\mathbf v}_j.$$
   这正是 FlashMLA README 所说 `head_dim_k = 576, head_dim_v = 512` 的 “MQA mode”（C27），也是 `dsv32` 附录 A 的 MQA 模式（C23）。
   **缩放仍是 $1/\sqrt{192}$，不是 $1/\sqrt{576}$**：吸收只是重排乘法，点积的数值一个不变。

### 2.8 为什么 RoPE 破坏吸收（笔者推导 + RoFormer 式 (16)）

RoFormer 式 (16)（照录）：
$$\boldsymbol{q}_{m}^{\intercal}\boldsymbol{k}_{n}=(\boldsymbol{R}^{d}_{\Theta,m}\boldsymbol{W}_{q}\boldsymbol{x}_{m})^{\intercal}(\boldsymbol{R}^{d}_{\Theta,n}\boldsymbol{W}_{k}\boldsymbol{x}_{n})=\boldsymbol{x}^{\intercal}\boldsymbol{W}_{q}R^{d}_{\Theta,n-m}\boldsymbol{W}_{k}\boldsymbol{x}_{n}\quad(16)$$
“where $\boldsymbol{R}^{d}_{\Theta,n-m}=(\boldsymbol{R}^{d}_{\Theta,m})^{\intercal}\boldsymbol{R}^{d}_{\Theta,n}$.”

若对 $\mathbf k^C$ 直接施加 RoPE：$\mathbf k_{j,i}=\mathbf R_j W^{UK}_i\mathbf c_j^{KV}$，$\mathbf q_{t,i}=\mathbf R_t\mathbf q^C_{t,i}$，则
$$\mathbf q_{t,i}^\top\mathbf k_{j,i}=(\mathbf q^C_{t,i})^\top\,\underbrace{\mathbf R_{j-t}}_{\text{随 }j\text{ 变}}\,W^{UK}_i\,\mathbf c^{KV}_j .$$
夹在中间的 $\mathbf R_{j-t}W^{UK}_i$ 对每个历史位置 $j$ 都不同，不存在一个与 $j$ 无关的 $\tilde{\mathbf q}_{t,i}$。
要么对每个 $j$ 重新算 $\mathbf R_jW^{UK}_i\mathbf c_j$（= 重算所有前缀的键，C7 原话），要么把旋转后的键缓存下来（= 回到 MHA 的缓存）。
论文原话是 “a RoPE matrix related to the currently generating token will lie between $W^{Q}$ and $W^{UK}$”；
更精确地说，夹在中间的是**相对**旋转 $\mathbf R_{j-t}$，它同时依赖当前位置 $t$ 与被查询的位置 $j$。
（数值反例见 4.6 节与练习 E4。）

解耦的效果：位置信息只放在一个很小的共享键 $\mathbf k^R_j\in\R^{64}$ 上，它可以旋转后直接缓存（缓存 64 个数，而不是每头 192 个），
内容部分 $\mathbf k^C$ 不旋转，于是吸收成立。

### 2.9 课程记号对照（style-guide §2）

| 论文 | 课程写法 | 说明 |
|---|---|---|
| $\mathbf q^T$（上标 T） | $\mathbf q^\top$ 或 `\T` | 统一转置写法 |
| $\operatorname{Softmax}_j$ | $\softmax_j$ | 宏 `\softmax` |
| $l$（层数） | $l$ | style-guide 明确“KV cache 公式里沿用 DeepSeek 论文的 $l$” |
| $d_h,n_h,d_c,d_c',d_h^R$ | 同 | style-guide 已收录 |
| $W^{DKV},W^{UK},W^{UV},W^{KR},W^{DQ},W^{UQ},W^{QR}$ | 同 | D = down，U = up，R = RoPE，C = compressed（内容） |
| $\operatorname{RoPE}(\cdot)$、旋转矩阵 | 旋转矩阵写 $\mathbf R_{\Theta,m}$（RoFormer 记号）或 $\mathbf R_m$ | 避免与上标 $R$（RoPE 分支）和 $\R$（实数集）混淆，讲义首次出现时说明 |
| 每头切片（论文无） | $W^{UK}_i\in\R^{d_h\times d_c}$、$W^O_i\in\R^{d\times d_h}$ | 课程补充记号，首次出现要交代“第 $i$ 头对应的行/列块” |
| 吸收后的查询/输出（论文无） | $\tilde{\mathbf q}_{t,i},\tilde{\mathbf o}_{t,i}\in\R^{d_c}$ | 课程补充 |
| $\mathbf u_t$（式 (19) 注意力输出） | $\mathbf u_t$ | 与 style-guide 一致；注意第 10 课里论文又把 $\mathbf u_t$ 当 FFN 输入（见 lesson-10-doc §2.8） |
| $[\cdot;\cdot]$ | 同 | 纵向拼接（列向量上下叠） |

---

## 3. 规格与超参数表

### 3.1 注意力相关超参数（论文 vs config.json）

| 量 | 论文符号 | V2（论文 §3.1.2） | V2 config.json | V2-Lite（论文附录 B.1） | V2-Lite config | V3（`dsv3` §4.2） | V3 config |
|---|---|---|---|---|---|---|---|
| 层数 | $l$ | 60 | `num_hidden_layers` 60 | 27 | 27 | 61 | 61 |
| 隐藏维 | $d$ | 5120 | `hidden_size` 5120 | 2048 | 2048 | 7168 | 7168 |
| 头数 | $n_h$ | 128 | `num_attention_heads` 128 | 16 | 16 | 128 | 128 |
| 每头维 | $d_h$ | 128 | `qk_nope_head_dim` 128；`v_head_dim` 128 | 128 | 128 / 128 | 128 | 128 / 128 |
| KV 压缩维 | $d_c$ | 512 | `kv_lora_rank` 512 | 512 | 512 | 512 | 512 |
| 查询压缩维 | $d_c'$ | 1536 | `q_lora_rank` 1536 | 不压缩 | `q_lora_rank` null | 1536 | 1536 |
| 解耦 RoPE 每头维 | $d_h^R$ | 64 | `qk_rope_head_dim` 64 | 64 | 64 | 64 | 64 |
| 初始化标准差 | — | 0.006 | `initializer_range` **0.02** | 论文未单列 | 0.02 | 0.006 | 0.02 |
| RoPE 底数 | — | 论文未写 | `rope_theta` 10000 | — | 10000 | — | 10000 |
| KV 头数字段 | — | 无此概念 | `num_key_value_heads` 128 | — | 16 | — | 128 |

- `initializer_range` 与论文不符（0.02 vs 0.006）：该字段只在从零初始化时起作用，发布的权重不受影响；记录为“config 与论文不一致”。
- `num_key_value_heads` 是 HF 框架的通用字段，MLA 并没有 128 个独立缓存的 KV 头；不要据此算缓存（勘误 E6）。

### 3.2 HF 参考实现的模块 ↔ 论文矩阵（`hf-v2-code`，`DeepseekV2Attention.__init__`）

| HF 模块 | 形状（输入 → 输出） | 对应论文矩阵 |
|---|---|---|
| `q_a_proj` + `q_a_layernorm` | 5120 → 1536，随后 RMSNorm | $W^{DQ}$，式 (12)；RMSNorm 即 C13 所说“压缩向量后的 RMS Norm” |
| `q_b_proj` | 1536 → 128×192 | $[W^{UQ};W^{QR}]$ 按头交错拼接，式 (13)(14)；随后 `split([128, 64])` |
| `kv_a_proj_with_mqa` | 5120 → 512+64 | $[W^{DKV};W^{KR}]$，式 (9)(15)；名字里的 “with_mqa” 指那 64 维是 MQA 式共享键 |
| `kv_a_layernorm` | 512 → 512 | 只作用在 $\mathbf c^{KV}$ 上，**不作用在 $\mathbf k^R$ 上** |
| `kv_b_proj` | 512 → 128×(128+128) | $[W^{UK};W^{UV}]$，式 (10)(11) |
| `o_proj` | 128×128 → 5120 | $W^O$，式 (19) |
| `softmax_scale` | `q_head_dim ** (-0.5)` = $192^{-1/2}$，再乘 `mscale**2` | 式 (18) 的 $1/\sqrt{d_h+d_h^R}$；`mscale` 属于 YaRN（见 lesson-10-doc） |

V2-Lite 的代码在 `q_lora_rank is None` 时用 `q_proj`（2048 → 16×192）直接出查询。

### 3.3 其它模型的 KV 相关规格（用于对比）

| 模型 | 层 | $n_h$ | KV 头 | $d_h$ | 来源 |
|---|---|---|---|---|---|
| DeepSeek LLM 67B | 95 | 64 | 8（GQA） | 128（=8192/64） | `dsllm` 表 2（“95 / 8192 / 64 / 8”）与 `hf-67b-config` 一致 |
| DeepSeek LLM 7B | 30 | 32 | 32（MHA） | 128 | `dsllm` 表 2；`hf-7b-config` |
| DeepSeekMoE 16B | 28 | 16 | 16（MHA） | 128 | `hf-moe16b-config`（论文表未在本次核对） |
| Qwen-2.5 72B / LLaMA-3.1 405B | — | — | 8 | 128 | 只核对了 `isca25` 表 1 的字节数；层数 80 / 126 是笔者用来复算的假设，复算结果与表 1 逐字节吻合 |

### 3.4 附录 D 的消融表（照录）

表 8（附录 D.1，7B 稠密，1.33T tokens）：

| Benchmark (Metric) | # Shots | w/ MQA | w/ GQA (8 Groups) | w/ MHA |
|---|---|---|---|---|
| # Params | - | 7.1B | 6.9B | 6.9B |
| BBH (EM) | 3-shot | 33.2 | 35.6 | 37.0 |
| MMLU (Acc.) | 5-shot | 37.9 | 41.2 | 45.2 |
| C-Eval (Acc.) | 5-shot | 30.0 | 37.7 | 42.9 |
| CMMLU (Acc.) | 5-shot | 34.6 | 38.4 | 43.5 |

表 9（附录 D.2）：

| Benchmark (Metric) | # Shots | Small MoE w/ MHA | Small MoE w/ MLA | Large MoE w/ MHA | Large MoE w/ MLA |
|---|---|---|---|---|---|
| # Activated Params | - | 2.5B | 2.4B | 25.0B | 21.5B |
| # Total Params | - | 15.8B | 15.7B | 250.8B | 247.4B |
| KV Cache per Token (# Element) | - | 110.6K | 15.6K | 860.2K | 34.6K |
| BBH (EM) | 3-shot | 37.9 | 39.0 | 46.6 | 50.7 |
| MMLU (Acc.) | 5-shot | 48.7 | 50.0 | 57.5 | 59.0 |
| C-Eval (Acc.) | 5-shot | 51.6 | **50.9** | 57.9 | 59.2 |
| CMMLU (Acc.) | 5-shot | 52.3 | 53.4 | 60.7 | 62.5 |

注意：小模型 C-Eval 上 MLA（50.9）低于 MHA（51.6）——“MLA 全面优于 MHA”是过度概括（勘误 E8）。
小规模训练 1.33T tokens，大规模只训了 420B tokens。

### 3.5 图 1(b) 的数值（从 arXiv HTML 的矢量图 `efficiency.svg` 几何坐标反算）

| 面板 | DeepSeek 67B | DeepSeek-V2 | 图内标注 |
|---|---|---|---|
| Training Costs (K GPU Hours/T Tokens) | 300.6 | 172.8 | “saving 42.5% of training costs” |
| KV Cache for Generation (KB/Token) | 389.1 | 25.9 | “reducing KV cache by 93.3%” |
| Maximum Generation Throughput (Tokens/Sec) | ≈9,608 | ≈55,374 | “576% of maximum throughput” |

反算方法：矢量图中 x 轴刻度的像素间距给出“像素/单位”，柱子右端的 x 坐标减去原点再除以它。
训练成本两根柱反算为 300.6000 与 172.8000，与正文 §3.2.3 的 “300.6K GPU hours” “172.8K GPU hours” 完全一致，说明方法可靠。
吞吐两根柱的比值 = 5.7633，与“5.76 times”一致；绝对值（9,608 / 55,374）是反算值，正文只说 “exceeding 50K”。
（核查补注：SVG 里刻度数字是字形路径，读不出文字；反算假设三个面板的刻度间隔分别为 50、100、10,000。
前两个面板反算结果与正文 300.6/172.8 及 “389.1/25.9 = `2*8*128*95*2` / `34560*6/8`” 逐位吻合，第三个面板的 10,000 间隔仍属假设，比值 5.763 不依赖该假设。）

---

## 4. 复算的数字

所有数字由 `python` 复算；1 次乘加 = 2 FLOPs；KB = 1000 字节，KiB = 1024 字节（ISCA 论文与图 1(b) 用 KB = 1000）。

### 4.1 每 token KV 元素数与字节数

| 量 | 表达式 | 值 |
|---|---|---|
| V2 MLA 元素/token | `(512+64)*60` | 34,560 |
| 与 $\tfrac92d_hl$ 核对 | `4.5*128*60` | 34,560（完全相等） |
| 等效 GQA 组数 | `(512+64)/(2*128)` | 2.25 |
| V3 MLA 元素/token | `(512+64)*61` | 35,136 |
| V2-Lite MLA 元素/token | `(512+64)*27` | 15,552（表 9 的 “15.6K”） |
| DeepSeek 67B GQA 元素/token | `2*8*128*95` | 194,560 |
| V2 若用 MHA（同 $n_h,d_h,l$） | `2*128*128*60` | 1,966,080 |
| DeepSeekMoE 16B（MHA） | `2*2048*28` | 114,688 |
| DeepSeek 7B（MHA） | `2*4096*30` | 245,760 |
| V2 bf16 字节/token | `34560*2` | 69,120 B = 69.12 KB = 67.50 KiB |
| V2 fp8 | `34560*1` | 34,560 B |
| V2 平均 6 bit（部署） | `34560*6/8` | 25,920 B = **25.9 KB**（= 图 1(b) 的柱长） |
| V3 bf16 | `35136*2` | 70,272 B = **70.272 KB**（= `isca25` 表 1） |
| V3 纯 fp8（576 维全 fp8） | `35136*1` | 35,136 B |
| V3.2 FlashMLA FP8 格式（656 B/层） | `656*61` | 40,016 B（bf16 的 56.9%） |
| 67B bf16 | `194560*2` | 389,120 B = **389.1 KB**（= 图 1(b) 的柱长）= 380.0 KiB |
| V2-Lite bf16 | `15552*2` | 31,104 B |
| DeepSeekMoE 16B bf16 | `114688*2` | 229,376 B |
| DeepSeek 7B bf16 | `245760*2` | 491,520 B |
| Qwen-2.5 72B bf16（GQA-8, 80 层） | `2*8*128*80*2` | 327,680 B（= `isca25` 表 1） |
| LLaMA-3.1 405B bf16（GQA-8, 126 层） | `2*8*128*126*2` | 516,096 B（= `isca25` 表 1） |

### 4.2 “93.3%”到底是什么比出来的

| 比较 | 表达式 | 下降 |
|---|---|---|
| V2 6-bit vs 67B bf16（部署态） | `1-25920/389120` | **93.34%** ← 论文的 93.3% |
| V2 bf16 vs 67B bf16（只算 MLA+层数） | `1-69120/389120` | 82.24% |
| V2 fp8 vs 67B bf16 | `1-34560/389120` | 91.12% |
| 按元素（表 1 的口径） | `1-34560/194560` | 82.24% |
| V2 vs 同尺寸 MHA（按元素） | `1-34560/1966080` | 98.24%（缓存只剩 1.758%，缩小 56.9 倍） |

分解（对数份额，笔者计算）：389,120 → 69,120（×0.1776 = 每层 576/2048 × 层数 60/95）→ 25,920（×0.375 = 6/16 bit）。
`log(389120/69120)/log(389120/25920)` = 0.638：**约 64% 的对数降幅来自 MLA + 层数，约 36% 来自 6 bit 量化。**

### 4.3 整段上下文的缓存总量（每条序列）

| $L_{ctx}$ | V2 bf16 | V2 6-bit | 67B bf16 | V2-as-MHA bf16 | V3 bf16 |
|---|---|---|---|---|---|
| 4,096 | 0.283 GB | 0.106 GB | 1.594 GB | 16.11 GB | 0.288 GB |
| 32,768 | 2.265 GB | 0.849 GB | 12.75 GB | 128.8 GB | 2.303 GB |
| 131,072（128K） | 9.060 GB（8.44 GiB） | 3.397 GB | 51.00 GB | 515.4 GB | 9.211 GB |
| 163,840（160K） | 11.33 GB | 4.247 GB | 63.75 GB | 644.2 GB | 11.51 GB |

表达式：`e*bytes*L/1e9`，`e` 取 4.1 节的元素数。（67B 实际只支持 4K 上下文，长上下文列只作对比。）

### 4.4 解码一步的注意力 FLOPs（每层，缓存里有 $L$ 个 token）

V2 维度：$n_h=128,d_h=128,d_h^R=64,d_c=512$。

| 方案 | 每步 FLOPs 表达式（每层） | $L=4096$ | $L=131072$ |
|---|---|---|---|
| (a) 朴素解压：每步把所有 $\mathbf c_j$ 用 $W^{UK},W^{UV}$ 展开再做注意力 | `2*L*512*(2*128*128) + 2*128*192*L + 2*128*128*L` = 33,636,352·L | 137.8 GFLOPs | 4,409 GFLOPs |
| (b) 缓存展开后的 K/V（MHA 式缓存） | `2*512*(2*128*128)`（展开新 token 自己的 k、v 一次）+ `2*128*192*L + 2*128*128*L` = 33,554,432 + 81,920·L（11.7 节更正：原表漏了固定项） | 0.369 GFLOPs | 10.77 GFLOPs |
| (c) 吸收（MQA 模式） | `2*128*128*512`（吸 $W^{UK}$）+ `2*128*512*128`（乘 $W^{UV}$）+ `(2*128*576 + 2*128*512)*L` = 33,554,432 + 278,528·L | 1.174 GFLOPs | 36.54 GFLOPs |

- (a)/(c) = 117.3（L=4096），120.7（L=128K）：吸收把解码注意力的算力降了两个数量级。
- 乘以 60 层（整个模型，每解码 1 个 token，$L=4096$）：(a) `137.77*60` ≈ 8,266 GFLOPs；(b) `0.3691*60` ≈ 22.1 GFLOPs（含展开新 token 的固定项，见 11.7）；(c) `1.1744*60` ≈ 70.5 GFLOPs。
  对照：按 $2N$（$N$ = 21B 激活参数，已含各投影矩阵）估的前向矩阵乘算力约 `2*21e9` = 42 GFLOPs/token——$L=4096$ 时吸收后只随 $L$ 增长的那部分（打分 + 加权求和，`278528*4096*60` = 68.5 GF）已超过它（笔者计算；吸收时乘 $W^{UK},W^{UV}$ 的固定部分已含在 $2N$ 里）。
- (c)/(b) = 3.18（L=4096；11.7 节更正前误为 3.50），随 L 增大趋于 3.4：**吸收后的 MLA 每个缓存 token 要做的 FLOPs 是 MHA 式缓存的 3.4 倍**（278,528 vs 81,920），
  因为每头要和 512+64 维的潜向量做点积、对 512 维潜向量加权求和。MLA 用算力换带宽。
- 每个缓存 token 每层要读的字节（bf16）：(b) `2*128*(192+128)` = 81,920 B；(c) `2*576` = 1,152 B；比值 71.1。

预乘矩阵 vs 重排顺序（解释“吸收”在工程上怎么做，笔者计算）：

| 量 | 表达式 | 值 |
|---|---|---|
| $W^{UQ}$ 参数 | `128*128*1536` | 25,165,824 |
| $W^{UK}$ 参数 | `128*128*512` | 8,388,608 |
| 预乘 $(W^{UK}_i)^\top W^{UQ}_i$（128 个 512×1536） | `128*512*1536` | 100,663,296（是分开存的 3.0 倍） |
| 每 token 查询侧 FLOPs：预乘路线 | `2*1536*128*512` | 201,326,592 |
| 每 token 查询侧 FLOPs：先 $W^{UQ}$ 再 $(W^{UK}_i)^\top$ | `2*1536*128*128 + 2*128*128*512` | 67,108,864 |
| 预乘 $W^O_iW^{UV}_i$（5120 × 128·512） | `5120*128*512` | 335,544,320（分开存 $W^O$+$W^{UV}$ 为 92,274,688） |

结论（笔者推导）：因为 $d_c=512>d_h=128$，把矩阵预乘成一个会让参数与 FLOPs 都变大；
“吸收”的实质是**改变乘法的结合顺序**，使 $\mathbf c_j$ 不必展开成 $\mathbf k_j,\mathbf v_j$。

### 4.5 算术强度与屋顶线（每个缓存 token，单个解码 token，bf16）

| 方案 | 表达式（FLOPs / 字节） | 值 |
|---|---|---|
| MHA 式缓存（每头独立 K/V） | `(2*128*128*2)/(2*128*128*2)` | 1.0 FLOP/B |
| GQA-8（67B：64 查询头共享 8 组 KV） | `(2*64*128*2)/(2*8*128*2)` | 8.0 FLOP/B |
| MLA 吸收（$h_q=128$，$s_q=1$） | `(2*128*576+2*128*512)/(2*576)` | 241.8 FLOP/B |
| FlashMLA 博客近似式 $2h_qs_q$ | `2*128*1` | 256（精确式 $h_qs_q(d_k+d_v)/d_k$ = `128*(576+512)/576` = 241.8） |
| MLA，$s_q=2$（MTP 投机解码） | `2*241.8` | 483.6 FLOP/B |
| H800 屋脊点（名义 990 TFLOPS / 3.35 TB/s） | `990/3.35` | 295.5 FLOP/B |
| H800 屋脊点（降频后 865 TFLOPS） | `865/3.35` | 258.2 FLOP/B |
| FlashMLA 博客的阈值 $\tfrac12\cdot865/3.35$ | `0.5*865/3.35` | 129.1（博客写作 “= 128”，是取整） |

读法：MHA/GQA 的解码注意力远在屋脊点左边（带宽受限），MLA 吸收后紧贴屋脊点。注意按精确比值 241.8 仍**略低于**降频屋脊点 258.2
（名义 295.5 更高），纯按屋顶线并不能断言“已算力受限”；FlashMLA 博客用近似式 $2h_qs_q=256$ 并把阈值 129.1 取整为 128，
才得出 $h_q=128$ 时 “the kernel is compute-bound” 的判断（C26）。讲义应写“接近屋脊点，DeepSeek 官方判为算力受限”，
而不是写成屋顶线算出来的结论。这就是博客开头所说 “why the MLA kernel is compute-bound despite being a decoding-stage attention kernel”。
$s_q=2$（MTP）时 483.6，才明确越过屋脊点。

### 4.6 RoPE 破坏吸收：2×2 数值反例（笔者构造，Python 验证）

设定：每头 $d_h=2$，$d_c=1$；$W^{UK}=\begin{pmatrix}3\\1\end{pmatrix}$；当前查询（旋转前）$\mathbf q=(1,2)^\top$，位置 $t=3$；
历史潜向量 $c_1=1,c_2=2,c_3=-1$；$\mathbf R(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\ \sin\phi&\cos\phi\end{pmatrix}$。

- **无 RoPE**：$\tilde q=\mathbf q^\top W^{UK}=1\cdot3+2\cdot1=5$，一个数通吃所有 $j$：打分 $=5c_j$ → 5, 10, −5。
- **有 RoPE，$\theta=0.5$ rad/位置**：有效的“吸收查询” $\tilde q_{t,j}=\mathbf q^\top\mathbf R((j-t)\theta)W^{UK}$：
  - $j=1$（$j-t=-2$，旋转 −1 rad）：$\tilde q=-1.5058$
  - $j=2$（旋转 −0.5 rad）：$\tilde q=1.9908$
  - $j=3$（旋转 0）：$\tilde q=5.0000$
  三个值互不相同 ⇒ 不存在对所有 $j$ 通用的 $\tilde q$，吸收失败。
- **有 RoPE，$\theta=\pi/2$**（便于手算）：$\mathbf R(-\pi)=-I$，$\mathbf R(-\pi/2)=\begin{pmatrix}0&1\\-1&0\end{pmatrix}$；
  $\tilde q_{3,1}=-5$，$\tilde q_{3,2}=\mathbf q^\top\begin{pmatrix}1\\-3\end{pmatrix}=-5$，$\tilde q_{3,3}=5$；打分 −5, −10, −5。
- 核对：`np.allclose(R(t*θ).T @ R(j*θ), R((j-t)*θ))` 为 True（即 RoFormer 式 (16) 的 $\mathbf R_{\Theta,n-m}=\mathbf R_{\Theta,m}^\top\mathbf R_{\Theta,n}$）。

### 4.7 参数量对比（每层，V2）

| 量 | 表达式 | 值 |
|---|---|---|
| 不压缩查询时的 $[W^Q;W^{QR}]$（5120 → 128×192） | `5120*128*192` | 125.8M |
| 压缩后 $W^{DQ}+[W^{UQ};W^{QR}]$ | `5120*1536 + 1536*128*192` | 45.6M |
| MHA 的 $W^K+W^V$（5120 → 2×16384） | `2*5120*128*128` | 167.8M |
| MLA 的 $W^{DKV}+W^{KR}+W^{UK}+W^{UV}$ | `5120*576 + 512*128*256` | 19.7M |
| 整个 MLA 注意力块（含 $W^O$、两个 RMSNorm） | 见 lesson-10-doc 4.1 | 149.2M |

### 4.8 MHA 模式 vs MQA 模式（每层 FLOPs，笔者计算；对应 `dsv32` 附录 A）

- 两种模式每个 token 的“固定成本”相同：MHA 模式解压 $\mathbf k^C,\mathbf v^C$ = `2*512*128*128*2` = 33,554,432；
  MQA 模式吸收 $W^{UK}$ 与乘 $W^{UV}$ 也是 33,554,432。
- 每个（查询, 键）对：MHA 模式 81,920；MQA 模式 278,528。
- **预填充** $T$ 个 token（因果，约 $T(T+1)/2$ 对）：$T=1024$：MHA 模式 77.4 GF，MQA 模式 180.5 GF；$T=4096$：824.8 GF vs 2,474.5 GF → 预填充用 MHA 模式。
- **解码**一步（缓存 $L$，不存展开的 K/V）：$L=4096$：MHA 模式需重新解压全部 $L$ 个 → 137.8 GF；MQA 模式 1.175 GF → 解码用 MQA 模式。
- 这与 C23 “MHA mode is used for training and prefilling, while the MQA mode is used for decoding” 完全吻合。

### 4.9 HF 参考实现的实际缓存（`hf-v2-code`）

`DeepseekV2Attention.forward` 把展开后的 `key_states`（128 头 × 192）与 `value_states`（128 头 × 128）交给 `past_key_value.update`：
每 token 每层 `128*(192+128)` = 40,960 元素；60 层 = 2,457,600 元素 = 4,915,200 B（bf16），是 MLA 潜向量缓存的 **71.1 倍**，
甚至比同尺寸 MHA（`2*128*128*60*2` = 3,932,160 B）还大 25%（因为 RoPE 键被复制到了每个头）。

---

## 5. 图表清单与重绘建议

### 5.1 论文中的图（`dsv2` v5）

| 图 | 原标题 | 内容 | 本课是否重绘 |
|---|---|---|---|
| 图 1(a) | “MMLU accuracy vs. activated parameters, among different open-source models.” | 散点：横轴激活参数（B），纵轴 MMLU；V2 用红星标在约 (21, 78.5)；其余为 LLaMA 1/2/3、Mixtral、Command R、Qwen1.5 家族连线，另有 DBRX、Grok-1、DeepSeek 67B、Mistral 7B | 不重绘（第 10 课可用），V2 的 78.5 来自表 2 |
| 图 1(b) | “Training costs and inference efficiency of DeepSeek 67B (Dense) and DeepSeek-V2.” | 三组横条，数值见 3.5 节 | **重绘**（见建议图 9-7） |
| 图 2 | “Illustration of the architecture of DeepSeek-V2. [...]” | 左：Transformer 块 ×L（RMSNorm → Attention → 残差；RMSNorm → FFN → 残差）。右下：MLA 数据流，$\mathbf c_t^{KV}$ 与 $\mathbf k_t^R$ 用阴影标为 “Cached During Inference”；右上：DeepSeekMoE | **MLA 部分重绘**（建议图 9-2）；MoE 部分在第 10 课 |
| 图 3 | 见 C11 | 四个面板，各画 8 个查询头：MHA 8 对 K/V 全部阴影（缓存）；GQA 4 组 K/V（每组 2 个查询）；MQA 1 对 K/V；MLA 8 对 K/V 为浅色（不缓存），右侧一个阴影的 “Compressed Latent KV” 通过 “projection” 箭头展开 | **重绘**（建议图 9-1）。注意原图是“simplified”，**没有画出解耦键 $\mathbf k^R$**，重绘时加上并注明“原图省略” |
| 图 4 | NIAH 热力图 | 见第 10 课 | 第 10 课 |
| 图 5 | HumanEval / LiveCodeBench | 附录 F | 不用 |

### 5.2 建议本课绘制的图（12 张，附作图数据）

**图 9-1 四联：MHA / GQA / MQA / MLA 缓存对比**（据 `dsv2` 图 3 重绘）
- 每面板 8 个查询头（`--fig-blue`）；键 `--fig-violet`、值 `--fig-green`；被缓存的东西加 `--fig-amber` 描边或阴影。
- MHA：8K + 8V 全缓存；GQA：4 组，每组 1K+1V 服务 2 个查询；MQA：1K+1V 服务 8 个查询；MLA：1 个潜向量 $\mathbf c^{KV}$（`--fig-orange`，缓存）
  → 虚线“projection”展开成 8K/8V（不缓存，浅色），**另加一个小的共享 $\mathbf k^R$（`--fig-red` 细线，缓存）**。
- 每个面板下方标“每 token 每层缓存元素数”（用 V2 的 $n_h=128,d_h=128$）：MHA 32,768；GQA-8 2,048（67B 的配置）；MQA 256；MLA 576。

**图 9-2 MLA 完整数据流（带形状）**（据 `dsv2` 图 2 下半部分重绘，数值来自 §3.1.2）
- $\mathbf h_t\in\R^{5120}$ 三路分支：
  - 查询：$W^{DQ}$ → $\mathbf c_t^Q\in\R^{1536}$ → RMSNorm → $W^{UQ}$ → $\mathbf q^C_{t,i}\in\R^{128}$ ×128 头；$W^{QR}$ → RoPE → $\mathbf q^R_{t,i}\in\R^{64}$ ×128 头；拼成 192 维。
  - 键值：$W^{DKV}$ → $\mathbf c_t^{KV}\in\R^{512}$（**缓存**）→ RMSNorm → $W^{UK}$ → $\mathbf k^C_{t,i}\in\R^{128}$；$W^{UV}$ → $\mathbf v^C_{t,i}\in\R^{128}$。
  - 位置：$W^{KR}$ → RoPE → $\mathbf k^R_t\in\R^{64}$（**缓存**，共享）→ 复制给 128 个头，与 $\mathbf k^C_{t,i}$ 拼成 192 维。
- 注意力：缩放 $1/\sqrt{192}$ → $\mathbf o_{t,i}\in\R^{128}$ ×128 → $W^O$ → $\mathbf u_t\in\R^{5120}$。

**图 9-3 KV 字节/token 柱状图（对数纵轴，bf16，计算得出）**
- 数据（字节）：DeepSeek 7B MHA 491,520；DeepSeek 67B GQA-8 389,120；DeepSeekMoE 16B MHA 229,376；V2-Lite MLA 31,104；
  V2 若用 MHA 3,932,160（虚框，假想）；V2 MLA bf16 69,120；V2 MLA 6-bit 25,920；V3 MLA bf16 70,272；V3.2 FP8 格式 40,016；
  Qwen-2.5 72B 327,680；LLaMA-3.1 405B 516,096。
- 颜色：MHA/GQA 用 `--fig-muted`，MLA 用 `--fig-orange`，量化后用 `--fig-orange` 斜线填充。

**图 9-4 “93.3%”的瀑布分解**（计算得出）
- 三级台阶：389.12 KB（67B，bf16）→ 69.12 KB（MLA+60 层，bf16）→ 25.92 KB（6-bit）；
  标注 “×0.2813（每层 576/2048）×0.6316（层数 60/95）= ×0.1776” 与 “×0.375（6/16 bit）”；右侧标对数份额 63.8% / 36.2%。

**图 9-5 吸收前后的计算图**
- 左（朴素）：每个缓存 $\mathbf c_j$ 进入 $W^{UK}$ 得 $\mathbf k_j$（每步对 $L$ 个 token 各做一次，红色高亮“每步 $L$ 次”）→ 与 $\mathbf q$ 点积。
- 右（吸收）：$\mathbf q^C_{t,i}$ 先过 $(W^{UK}_i)^\top$ 得 $\tilde{\mathbf q}_{t,i}\in\R^{512}$（每步 1 次）→ 直接与 $\mathbf c_j$ 点积；
  加权和 $\tilde{\mathbf o}_{t,i}=\sum_jp_j\mathbf c_j$ → $W^{UV}_i$ → $\mathbf o_{t,i}$。
- 角注 FLOPs（每层，$L=4096$）：137.8 GF vs 1.174 GF。

**图 9-6 RoPE 为什么挡住吸收**
- 上：$\mathbf q^\top\,\mathbf R_t^\top\,\mathbf R_j\,W^{UK}\,\mathbf c_j$ 的乘法链，$\mathbf R_t^\top\mathbf R_j=\mathbf R_{j-t}$ 用 `--fig-red` 框出“随 $j$ 变”。
- 下：4.6 节的 2×2 例子，三条向量 $\mathbf R((j-t)\theta)W^{UK}$（$\theta=0.5$，$j=1,2,3$）画在平面上：
  $W^{UK}=(3,1)$ 旋转 −1、−0.5、0 rad 得 $(2.4624,-1.9841)$、$(3.1122,-0.5607)$、$(3,1)$，与 $\mathbf q=(1,2)$ 的点积 −1.506、1.991、5.000。
  （由 $\mathbf R(\phi)(3,1)^\top=(3\cos\phi-\sin\phi,\ 3\sin\phi+\cos\phi)$ 用 Python 算得并核对过点积。）

**图 9-7 图 1(b) 重绘 + 一根补充柱**（数据见 3.5 节）
- 三组横条：训练 300.6 vs 172.8；KV 389.1 vs 25.9；吞吐 9,608 vs 55,374。
- 在 KV 面板加一根虚线柱“V2 bf16（未量化）69.1 KB”，让读者看到 93.3% 里有一部分来自量化。

**图 9-8 解码 FLOPs 与读取字节随上下文增长（双对数，每层，计算得出）**
- $L\in[2^{10},2^{17}]$；FLOPs：(a) 33,636,352·L；(b) 33,554,432 + 81,920·L（11.7 节更正）；(c) 33,554,432 + 278,528·L。
- 字节：(b) 81,920·L；(c) 1,152·L。建议左右两个小图并排。

**图 9-9 屋顶线：MHA / GQA / MLA 的解码注意力**
- 横轴算术强度（对数），纵轴 TFLOPS；屋顶：带宽 3.35 TB/s 的斜线 + 平顶 990（实线）与 865（虚线）。
- 点：MHA 1.0；GQA-8 8.0；MLA 241.8（$s_q=1$）；MLA 483.6（$s_q=2$）。屋脊 295.5 / 258.2。

**图 9-10 解耦 RoPE 的“键向量解剖”**
- 一个头的 192 维键：前 128 维 $\mathbf k^C_{j,i}=W^{UK}_i\mathbf c_j$（每头不同，`--fig-violet`，不旋转）| 后 64 维 $\mathbf k^R_j$（所有头相同，`--fig-red`，旋转）。
- 对应查询 192 维：128 维内容 + 64 维旋转。打分 = 内容点积 + 位置点积，两部分用不同颜色的括号标出。

**图 9-11 附录 D 消融柱状图**（数据见 3.4 节）
- 左：表 8 四个基准 × {MQA, GQA-8, MHA}；右：表 9 四个基准 × {Small MHA, Small MLA, Large MHA, Large MLA}，并在图例上写缓存元素数 110.6K / 15.6K / 860.2K / 34.6K。
- 标出小模型 C-Eval 上 MLA 略低（50.9 < 51.6）。

**图 9-12 MHA 模式 vs MQA 模式**（据 `dsv32` 附录 A 的思路，数值为笔者计算）
- 两个流程并排：MHA 模式（先解压 $\mathbf c\to\mathbf k,\mathbf v$，128 头各 192/128 维）用于预填充；MQA 模式（576/512 维单 KV 头）用于解码。
- 附小表：每对 FLOPs 81,920 vs 278,528；每 token 固定 33.55 MF（两者相同）；预填充 T=4096：824.8 vs 2,474.5 GF；解码 L=4096：137.8 vs 1.175 GF。

（可选）**图 9-13 MLA 演化时间线**：2024-05-06 V2 发布（`gh-v2` News）→ 2024-05-16 V2-Lite → 2024-06-17 Coder-V2（arXiv v1）→
2024-12 V3（同一 MLA，61 层）→ 2025-02-24 FlashMLA（Open-Source Week Day 1）→ V3.2 基于 MLA 的 MQA 模式实现 DSA（`dsv32`）。

---

## 6. 算例与练习素材（附答案）

**E1（计算）每 token 缓存。** 用 V2 的 $l=60,d_c=512,d_h^R=64$ 与 67B 的 $l=95$、8 个 KV 头、$d_h=128$，算 bf16 下每 token 缓存字节。
答：V2 `(512+64)*60*2` = 69,120 B；67B `2*8*128*95*2` = 389,120 B；V2 是 67B 的 17.8%。128K 上下文：V2 9.06 GB，67B 51.0 GB。

**E2（概念+计算）“2.25 组”。** 证明 $d_c=4d_h$、$d_h^R=d_h/2$ 时 MLA 缓存等于 $n_g=2.25$ 的 GQA；再算它相当于 V2 尺寸 MHA 的百分之几。
答：$(4+\tfrac12)d_hl=2n_gd_hl\Rightarrow n_g=2.25$；$2.25/128=1.758\%$，即缩小 56.9 倍。

**E3（计算）拆解 93.3%。** 答：见 4.2 节；关键是 25,920 = 34,560 × 6/8，只有“6 bit 平均量化 + MLA”一起才得到 93.3%，纯 bf16 的 MLA 只降 82.2%。

**E4（推导+计算）RoPE 反例。** 用 4.6 节的设定（$\theta=0.5$）算 $\tilde q_{3,j}$，并解释为什么这说明吸收失败。
答：−1.506、1.991、5.000，随 $j$ 变化；吸收要求一个与 $j$ 无关的 $\tilde q$。

**E5（计算）吸收省了多少算力。** $L=4096$，求每层解码一步的 FLOPs：朴素解压 vs 吸收。
答：137.8 GF vs 1.174 GF，117 倍。追问：与 MHA 式缓存比呢？吸收是其 3.5 倍 FLOPs，但读的字节只有 1/71。

**E6（计算）屋顶线。** H800：3.35 TB/s，990 TFLOPS（名义）。求 MLA 吸收后解码注意力的算术强度（$h_q=128$，$s_q=1$）并判断受限类型；再求 GQA-8 的。
答：241.8 FLOP/B，接近但略低于屋脊点（名义 295.5；按 865 TFLOPS 为 258.2），处在带宽与算力的交界；FlashMLA 博客用近似 $2h_qs_q$ 与取整后的阈值 128 判为算力受限。GQA-8 仅 8 FLOP/B，严重带宽受限。开 MTP（$s_q=2$）后 483.6，明确算力受限。

**E7（概念）缩放因子。** 吸收后查询/键都是 576 维，为什么 softmax 仍除以 $\sqrt{192}$？
答：吸收只改变乘法顺序，$\tilde{\mathbf q}^\top\mathbf c+\mathbf q^{R\top}\mathbf k^R$ 与原来 192 维点积数值完全相等；缩放必须与训练时一致（$1/\sqrt{d_h+d_h^R}=0.07217$），若误用 $1/\sqrt{576}=0.04167$，logits 被整体缩小到 57.7%（= $\sqrt{192/576}$），softmax 变平。

**E8（计算）两种模式。** 用 4.8 节的数，说明为什么预填充用 MHA 模式、解码用 MQA 模式。
答：两模式每 token 固定成本相同（33.55 MF），MHA 模式每对便宜 3.4 倍 → 预填充（每个 token 的解压被 $T$ 个查询摊薄）选 MHA；解码每步只有 1 个查询，MHA 模式要重新解压全部历史（或缓存展开的 K/V，缓存变大 71 倍）→ 选 MQA。

**E9（计算）HF 参考实现的缓存。** 答：每 token 4,915,200 B（bf16），是 MLA 的 71.1 倍、同尺寸 MHA 的 1.25 倍。所以 README 推荐 vLLM/SGLang。

**E10（计算）为什么不“离线预乘”。** 答：$(W^{UK}_i)^\top W^{UQ}_i$ 共 100.7M 参数（分开 33.6M），查询侧每 token 201M FLOPs（分开 67M）；因为 $d_c=512>d_h=128$，预乘反而膨胀。

**E11（计算）小模型对比。** V2-Lite 与 DeepSeekMoE 16B 的每 token bf16 缓存各多少？比值？与表 9 的 14% 对照。
答：31,104 B vs 229,376 B → 13.6%；表 9 小模型 15.6K/110.6K = 14.1%（笔者由数值推断表 9 的小 MHA 基线是 27 层：`2*2048*27` = 110,592，而 DeepSeekMoE 16B 是 28 层；论文未给该基线配置，见第 8 节第 8 条）。

**E12（概念）查询压缩省了什么。** 答：不省 KV cache（C6）；每层查询投影参数从 125.8M 降到 45.6M；论文说目的是“reduce the activation memory during training”，具体机制论文未展开（见第 8 节）。

---

## 7. 勘误与常见误解

**E1. “MLA 让 KV cache 降了 93.3%。”** 不准确。93.3% 是“**部署态** V2（平均 6 bit 量化的 KV）vs DeepSeek 67B（按 bf16 计）”，
图 1(b) 的 389.1 KB 与 25.9 KB 与 `2*8*128*95*2` 和 `34560*6/8` 逐位吻合。纯 MLA（bf16 对 bf16）是 82.2%；
而对同尺寸 MHA，MLA 本身把元素数降了 98.2%。论文表 1 明说“measured by the number of elements, regardless of the storage precision”。
（论文没有明写 93.3% 的计算口径；“6 bit 部署缓存”这一解释是笔者由数值吻合推出，置信度高但属推断。）

**E2. “吸收 = 把 $W^{UK}$ 事先乘进 $W^Q$。”** 字面上论文正文如此说（C4），但 V2 有查询压缩，附录 C 改为 “into $W^{UQ}$”；
而且预乘会让参数和 FLOPs 各膨胀约 3 倍（4.4 节）。工程上是“改变结合顺序”。v5 删掉了 “it can be completed offline at once” 一句。

**E3. “MLA 省算力。”** 不对。解码时吸收后的 MLA 每缓存 token 的注意力 FLOPs 是 MHA 式缓存的 3.4 倍；它省的是**显存容量和带宽**，
代价是算力，使解码内核的算术强度升到屋脊点附近，DeepSeek 在 FlashMLA 博客里判为算力受限（C26；精确比值 241.8 略低于 258.2，见 4.5 节）。

**E4. “解耦 RoPE 的键每个头一个。”** 不对。$\mathbf k_t^R$ 只有一个（“a shared key”），所有头共享，MQA 式；每头独立的只有 $\mathbf q^R_{t,i}$。

**E5. “MLA 是对 MHA 做 SVD / 低秩近似得到的。”** 不对。压缩与展开矩阵从头随模型一起训练（C24 “jointly trained with the model”）；
V2 从零预训练。（事后把 MHA 模型转换成 MLA 的工作存在，但不是 DeepSeek-V2 的做法；本次未核对那类论文。）

**E6. “config 里 `num_key_value_heads=128`，所以 MLA 缓存 128 个 KV 头。”** 不对。那是 HF 通用字段；真正缓存的是 `kv_lora_rank + qk_rope_head_dim` = 576 维。
另外 HF 参考代码确实把展开后的 K/V 放进缓存（4.9 节），用它测显存会得出“MLA 比 MHA 还费缓存”的错误结论。

**E7. ISCA '25 论文表 1 的 “7.28x” 算错了。** 516.096 / 70.272 = 7.344；Qwen-2.5 72B 的 4.66x 正确（327.680/70.272 = 4.663）。
正文说 “LLaMA-3.1 405B's 516 KB” 本身没错，只是倍数错。（引用时写 7.3 倍或 7.34 倍，并注明原表笔误。arXiv v1 与 v2 的表 1 都印作 7.28x；ACM 出版版未核对。）

**E8. “MLA 在所有基准上都优于 MHA。”** 表 9 小模型 C-Eval：MHA 51.6 > MLA 50.9。另外表 9 的 MLA 模型激活参数更少（2.4B vs 2.5B；21.5B vs 25.0B），
大模型只训 420B tokens；V3 报告已改口为 “comparable to standard MHA”（C21）。“MLA 至少不输 MHA、且缓存小一个数量级以上”是更稳妥的表述。

**E9. 附录编号版本漂移。** v1/v2 中 MLA 全公式在附录 B、消融在附录 C（C.1/C.2，表 6/7）；v3 起为附录 C、D（D.1/D.2，表 8/9）。按 v5 引用。

**E10. 缩放因子。** 式 (18) 除以 $\sqrt{d_h+d_h^R}=\sqrt{192}$，不是 $\sqrt{d_h}$，也不是吸收后形式里的 $\sqrt{576}$。HF 代码：`softmax_scale = q_head_dim ** (-0.5)`，`q_head_dim = 128 + 64`。

**E11. 图 3 没画解耦键。** 原图是 “Simplified illustration”，MLA 面板只把潜向量标为缓存；图 2 与附录 C 才把 $\mathbf k_t^R$ 也标为缓存。重绘时要补上。

**E12. “RMSNorm 作用在整个 576 维上。”** HF 代码中 `kv_a_layernorm` 只作用于 512 维的 $\mathbf c^{KV}$，64 维 $\mathbf k^R$ 不归一化；
查询侧 `q_a_layernorm` 作用于 $\mathbf c^Q$。论文只说 “additional RMS Norm layers after the compressed latent vectors”。

**E13. FlashMLA 的首发数字与现在的数字不同。** 首发（2025-02，CUDA 12.6）：3000 GB/s、580 TFLOPS；2025-04-22 更新后：660 TFLOPS（CUDA 12.8）。
引用时注明日期。FlashMLA 博客里的阈值 “$\tfrac12\cdot\tfrac{865}{3.35}=128$” 实为 129.1，是取整。

**E14. RoPE 维度排列（代码细节）。** HF 代码 `apply_rotary_pos_emb` 先把 64 维按相邻两两成对（interleaved）重排为前后两半，再用 `rotate_half`；
即权重里 RoPE 维是 GPT-J 式相邻配对。自己实现时若直接套 LLaMA 的 `rotate_half`，结果会错。（FlashMLA 当前 README 对 V4.1 也写 “non-neox RoPE”。）

---

## 8. 未能核实 / UNVERIFIED

1. **93.3% 与 5.76× 的精确计算口径**：论文没写 67B 在对比中用什么精度的 KV/权重，也没写 V2 那根柱是不是正好“6 bit × 34,560”。数值吻合是强证据，但仍属推断。
2. **KV cache 量化的具体方案**（哪几维量化、是否分组、RoPE 部分是否量化）：论文只说 “6 bits on average”，引用 Hooper et al. 2024（KVQuant）与 Zhao et al. 2023（Atom），未给细节。
3. **查询压缩如何减少训练激活显存**：论文一句话带过，未说明是否配合重计算。
4. **“additional scaling factors at the width bottlenecks”对潜向量的具体数值**：HF 代码里看不到对 $\mathbf c^{KV}$ 的显式缩放常数（可能已并入 RMSNorm 权重）；路由专家侧的 `routed_scaling_factor=16.0` 见第 10 课。
5. **“improved version of FlashAttention-2”** 的改动内容：论文未说明；训练框架 HAI-LLM 未开源。
6. **V2 时期（2024）线上服务的 MLA 内核是否就是 2025-02 开源的 FlashMLA**：FlashMLA README 只说支持 V3/V3.1（稠密解码），未提 V2。
7. **v5 删除 “offline at once” 一句的原因**：只能推测（见 0.3 节）。
8. **表 9 两个 MHA 基线的精确配置**（头数、层数）：论文未给。110.6K = `2*2048*27` 精确吻合；860.2K 与 `2*7168*60` = 860,160 吻合，但也可能是其它组合。
9. **表 8 三个 7B 模型各自的层数**：论文只说“通过调整层数对齐到约 7B”。
10. **GQA 论文（2305.13245）的出版信息（EMNLP 2023）与 RoFormer 期刊版（Neurocomputing 2024）**：本次只核对了 arXiv 元数据。
11. **vLLM PR #4650 / SGLang 的 MLA 实现细节**：只读了 V2 README 对它们的描述，未读代码。
12. **H800 的 NVLink 带宽等硬件参数**：本课未用到，未核对；FlashMLA 博客给出的 3.35 TB/s 与 990 TFLOPS 是 DeepSeek 的说法，未与 NVIDIA 官方规格表交叉核对。
13. **MLA 的思想来源 / 是否有更早的同类工作**：论文只把 MLA 归功于 Huazuo Gao、Wangding Zeng（C20），未讨论相关工作；本次没有检索是否存在更早的“低秩 KV 联合压缩 + 吸收”论文，讲义不要写“首创”。

---

## 9. 参考文献卡片草稿

```
[dsv2] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
    a: DeepSeek-AI (Aixin Liu et al.) (2024), preprint. arXiv:2405.04434v5
    url: https://arxiv.org/abs/2405.04434v5
    finding: 提出 MLA：把 K、V 联合压缩成每层 512 维潜向量并用 64 维共享键承载 RoPE，每 token 缓存 (d_c+d_h^R)l 个元素，约等于 2.25 组 GQA。
    matters: 本课全部 MLA 公式、表 1、图 3、附录 C 的吸收论证和附录 D 消融都以它 v5 的编号为准。

[dsv3] DeepSeek-V3 Technical Report
    a: DeepSeek-AI (2024), preprint. arXiv:2412.19437v2
    url: https://arxiv.org/abs/2412.19437
    finding: V3 沿用与 V2 完全相同的 MLA 维度（128 头、d_h=128、d_c=512、d_c'=1536、d_h^R=64），层数 61、隐藏维 7168。
    matters: 用来算 V3 的每 token 缓存（35,136 元素 = 70,272 B），并说明 MLA 在下一代被原样保留。

[dsv32] DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
    a: DeepSeek-AI (2025), preprint. arXiv:2512.02556v1
    url: https://arxiv.org/abs/2512.02556
    finding: 附录 A 区分 MLA 的 MHA 模式与 MQA 模式，并说明训练和预填充用前者、解码用后者，DSA 建在 MQA 模式上。
    matters: 给“吸收 = MQA 模式”的课堂推导提供官方出处，并为第 14 课 DSA 铺路。

[isca25] Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures
    a: Chenggang Zhao et al. (2025), ISCA '25. arXiv:2505.09343v2
    url: https://arxiv.org/abs/2505.09343
    finding: 表 1 以 BF16 比较每 token KV：V3 70.272 KB、Qwen-2.5 72B 327.680 KB、LLaMA-3.1 405B 516.096 KB，并强调 MLA 的投影矩阵随模型联合训练。
    matters: 为 V3 的缓存字节提供同行评审出处，同时它的 7.28x 是本课勘误 E7 的对象。

[dsllm] DeepSeek LLM: Scaling Open-Source Language Models with Longtermism
    a: DeepSeek-AI (2024), preprint. arXiv:2401.02954
    url: https://arxiv.org/abs/2401.02954
    finding: 表 2 给出 67B 为 95 层、d_model=8192、64 头、8 个 KV 头，并说明 67B 用 GQA。
    matters: 67B 是 V2 所有“对比 DeepSeek 67B”头条数字的基线，它的 389,120 B/token 是 93.3% 的分母。

[roformer] RoFormer: Enhanced Transformer with Rotary Position Embedding
    a: Jianlin Su et al. (2021), preprint (later Neurocomputing). arXiv:2104.09864v5
    url: https://arxiv.org/abs/2104.09864
    finding: 式 (16) 表明加 RoPE 后打分为 x^T W_q R_{n-m} W_k x_n，其中 R_{n-m} = R_m^T R_n。
    matters: 正是这个夹在中间、随位置变化的旋转矩阵让 W^UK 无法被吸收，本课用它写严格推导。

[mqa] Fast Transformer Decoding: One Write-Head is All You Need
    a: Noam Shazeer (2019), preprint. arXiv:1911.02150v1
    url: https://arxiv.org/abs/1911.02150
    finding: 提出所有查询头共享一组键和值的多查询注意力（MQA）。
    matters: 表 1 与图 3 的 MQA 行出自这里，吸收后的 MLA 在计算形式上正是一种 MQA。

[gqa] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
    a: Joshua Ainslie et al. (2023), EMNLP 2023 (venue unverified here). arXiv:2305.13245v3
    url: https://arxiv.org/abs/2305.13245
    finding: 提出分组查询注意力：n_g 组键值各服务一组查询头，介于 MHA 与 MQA 之间。
    matters: “MLA 相当于 2.25 组 GQA”这一比较的参照物，也是 67B 采用的注意力。

[flashmla] FlashMLA (GitHub repository, deepseek-ai)
    a: Jiashi Li, Shengyu Liu (2025–2026), artefact.
    url: https://github.com/deepseek-ai/FlashMLA
    kind: artefact
    finding: MLA 解码内核首发（2025-02-24，Open-Source Week Day 1）达 3000 GB/s、580 TFLOPS，后更新到 660 TFLOPS；README 定义 MQA 模式为 head_dim_k=576、head_dim_v=512。
    matters: 用它说明吸收后的 MLA 在工程上就是一个 576/512 维的单 KV 头注意力内核，并引用其博客解释为何解码变成算力受限。

[flashmla-blog] A Deep-Dive Into the New Flash MLA Kernel (FlashMLA docs/20250422-new-kernel-deep-dive.md)
    a: DeepSeek (2025-04-22), artefact.
    url: https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md
    kind: artefact
    finding: 推导 MLA 解码的计算/访存比约为 2·h_q·s_q，H800 上 h_q·s_q ≥ 约 128 即算力受限。
    matters: 本课屋顶线算例（练习 E6、图 9-9）的公式与硬件参数出处。

[hf-v2] DeepSeek-V2 model repository on Hugging Face (config.json, modeling_deepseek.py)
    a: DeepSeek-AI (2024), artefact.
    url: https://huggingface.co/deepseek-ai/DeepSeek-V2
    kind: artefact
    finding: config 的 kv_lora_rank=512、q_lora_rank=1536、qk_rope_head_dim=64 与论文一致，参考代码却把展开后的 K/V 整个写进缓存。
    matters: 用来给论文矩阵与代码模块逐一对号，并提醒读者 HF 参考实现测不出 MLA 的省显存效果。

[gh-v2] DeepSeek-V2 GitHub README
    a: DeepSeek-AI (2024), artefact.
    url: https://github.com/deepseek-ai/DeepSeek-V2
    kind: artefact
    finding: 记录 V2 于 2024-05-06 发布、V2-Lite 于 2024-05-16 发布，并说明 HF 代码慢于内部实现、推荐 vLLM/SGLang。
    matters: 提供发布日期与“参考实现 ≠ 生产实现”的官方说法，支撑勘误 E6。
```

---

## 10. 核查记录（对抗式复核，2026-09-24）

复核方法：对下列高风险论断逐条回到一手来源重查，不采信本文件原有结论。论文用 `arxiv.py text/grep` 取 LaTeXML 全文
（`dsv2` 取了 v1–v5 五个版本并逐版 diff）；制品用 raw URL 原样下载（HF `config.json` / `modeling_deepseek.py`、
FlashMLA README 与博客、首次提交 `414a2f3eed`、open-infra-index README）；图 1(b) 从 arXiv HTML 的 `efficiency.svg`
几何坐标反算，图 3 直接看 `dsattn.png`；所有派生数字在 Python 里重算。

### 10.1 核对无误（22 项）

| # | 论断 | 来源与方法 | 结果 |
|---|---|---|---|
| 1 | V2 MLA 超参数 60 / 5120 / 128 / 128 / 512 / 1536 / 64，std 0.006（C12） | `dsv2` v5 §3.1.2 grep | 逐字一致 |
| 2 | 式 (1)–(19) 与附录 C 式 (37)–(47) | v5 LaTeX 源逐符号比对 | 一致（含式 (17) 的 $\mathbf k_t^R$ 无头下标、式 (18) 分母 $\sqrt{d_h+d_h^R}$） |
| 3 | 表 1 四行与标题（“regardless of the storage precision”“2.25 groups”） | v5 表 1 | 一致 |
| 4 | 正文 “absorbed into $W^Q$”、附录 “absorb $W^{UK}$ into $W^{UQ}$” | v5 第 2.1.2 节与附录 C | 一致 |
| 5 | “offline at once” 在 v1–v4，v5 改为 “Therefore, we do not need to compute keys and values out for each query.” | v1–v5 全文 diff | 一致；v4→v5 全文只有这一句与日期不同 |
| 6 | 版本日期 v1 05-07、v2 05-08、v3 05-16、v4 05-24、v5 06-19；v1/v2 附录 B/C、表 6/7 → v3 起附录 C/D、表 8/9；v3 起引言加 V2-Lite 一段、“delve deeper” 消失 | arXiv abs 页 submission history + 全文 diff | 一致；v1 与 v2 正文只差日期 |
| 7 | 42.5% / 93.3% / 5.76×（摘要）；300.6K / 172.8K GPU·h 每 T tokens；FP8 权重、6 bits on average；>50K、>100K tok/s | v5 §3.2.3 与摘要 | 一致 |
| 8 | 图 1(b) 柱长 300.6 / 172.8、389.1 / 25.9、比值 5.763 | `efficiency.svg` 反算：300.6000、172.8000、389.0999、25.9000、5.7633 | 一致（吞吐绝对值依赖刻度假设，已在 3.5 节补注） |
| 9 | 93.34%、82.24%、91.12%、98.24%、对数份额 0.638、×0.1776、×0.375 | Python | 一致 |
| 10 | 表 8、表 9 全部数值（含小模型 C-Eval MHA 51.6 > MLA 50.9） | v5 附录 D | 一致 |
| 11 | ISCA 表 1：70.272 / 327.680 / 516.096 KB，4.66x，7.28x；“jointly trained with the model”（§2.1.2） | `isca25` v1 与 v2 | 一致；`516.096/70.272` = 7.344，确认 7.28x 为原表错误 |
| 12 | 67B = 95 层、d=8192、64 头、8 KV 头、2.0T tokens，GQA | `dsllm` 表 2 + `hf-67b-config` | 一致 |
| 13 | V3 = 61 层、d=7168、MLA 维度同 V2；只缓存蓝框两向量；“comparable to standard MHA” | `dsv3` §4.2、§2.1.1 | 一致 |
| 14 | V3.2 附录 A “MHA mode … training and prefilling, … MQA mode … decoding”；DSA 基于 MQA 模式 | `dsv32` | 一致 |
| 15 | FlashMLA 博客：990 / ~865 TFLOPS、3.35 TB/s、$h_qs_q(d_k+d_v)/d_k\approx2h_qs_q$、阈值 “= 128”、解码不用 TP | `docs/20250422-new-kernel-deep-dive.md` | 一致 |
| 16 | 首发 README：3000 GB/s、580 TFLOPS、CUDA 12.6、BF16、block 64；当前 README：660 TFLOPS、CUDA 12.8、MQA 模式 576/512、V3.2 FP8 行 656 B、稠密解码只读 bf16/fp16 | 提交 `414a2f3eed`（author date 2025-02-21，commit date 2025-02-24）与当前 README | 一致 |
| 17 | Open-Source Week 自 2025-02-24 起，Day 1 = FlashMLA | open-infra-index README | 一致 |
| 18 | config：`kv_lora_rank` 512、`q_lora_rank` 1536、`qk_rope_head_dim` 64、`num_key_value_heads` 128、`initializer_range` 0.02、`rope_theta` 10000 | HF `config.json` | 一致 |
| 19 | HF 代码：`softmax_scale = q_head_dim**-0.5 * mscale**2`；`kv_a_layernorm` 只作用于 512 维；展开后 128×192 的 `key_states` 写入 `past_key_value`；RoPE 先 `view(…, d//2, 2).transpose` 再 `rotate_half` | `modeling_deepseek.py` | 一致 |
| 20 | 解码 FLOPs / 字节 / 算术强度：137.8、0.336、1.174 GF；278,528 vs 81,920；1,152 vs 81,920 B；241.8、295.5、258.2、129.1；HF 缓存 4,915,200 B（71.1×、1.25×） | Python | 数字一致；**对“已算力受限”的解读有误**，见 10.2 |
| 21 | RoFormer 式 (16) 与 $\mathbf R_{\Theta,n-m}=(\mathbf R_{\Theta,m})^\top\mathbf R_{\Theta,n}$；4.6 节反例 −1.5058 / 1.9908 / 5 与 θ=π/2 情形 | `roformer` v5 LaTeX；Python | 一致 |
| 22 | 图 3 的 MLA 面板没有画解耦键；附录 A 的 MLA 署名（Huazuo Gao、Wangding Zeng）与致谢 Jianlin Su | `dsattn.png`；v5 附录 A | 一致 |

另核对了第 0 节全部 arXiv 元数据（编号、版本、首发日期、作者数 “+151 more”）与 `hf-7b-config`、`hf-moe16b-config` 的层数与头数，均一致。

### 10.2 修改（5 处）

1. **4.5 节“读法”**：原文 “$h_q=128$ 时已算力受限”。精确比值 241.8 FLOP/B 略低于降频屋脊点 258.2（名义值 295.5 更高），
   纯按屋顶线得不出“算力受限”；这个判断来自 FlashMLA 博客的近似 $2h_qs_q=256$ 与取整后的阈值 128。已改为
   “紧贴屋脊点，DeepSeek 官方判为算力受限”，并注明 $s_q=2$ 时才明确越过屋脊点。来源：FlashMLA 博客原句（C26）+ Python。
   同段原写 “FlashMLA 博客标题所说 why the MLA kernel is compute-bound …”，博客标题实为 “A Deep-Dive Into the New Flash MLA Kernel”，
   该句出自博客第二段，已改为“博客开头所说”。
2. **练习 E6 答案**：同上，“基本进入算力受限”改为“接近但略低于屋脊点，博客判为算力受限”。
3. **勘误 E3**：“使解码内核变成算力受限”改为“升到屋脊点附近，DeepSeek 在 FlashMLA 博客里判为算力受限”。
4. **练习 E11 答案**：“表 9 的 MHA 基线是 27 层”原写成事实，论文并未给出该基线配置；改为“笔者由 `2*2048*27` = 110,592 推断”，与第 8 节第 8 条一致。
5. **C16 标题与 E7、3.5 节补注**：C16 所引是摘要原话，引言与第 5 节措辞不同（数字相同），已在标题注明；E7 补“arXiv v1、v2 都印作 7.28x，ACM 版未核对”；
   3.5 节补注图 1(b) 反算依赖的刻度间隔假设。

### 10.3 核查后仍存疑

- 图 1(b) 吞吐面板的刻度间隔（10,000 tok/s）是假设；刻度数字是字形路径，没有解码。比值 5.763 不受影响。
- ISCA '25 的 ACM 出版版表 1 是否已改正 7.28x：未核对。
- H800 的 990 / 865 TFLOPS 与 3.35 TB/s 只来自 DeepSeek 博客，未与 NVIDIA 规格表交叉核对（沿用第 8 节第 12 条）。

---

## 11. 补充核实（写课时，2026-09-24）

写 `lesson-09-v2-mla.md` 时额外核对或复算的内容。引文用 arXiv 2405.04434 v5 的 LaTeXML 全文（写课环境的临时副本）逐句 grep；
数字全部在 Python 里复算。凡属笔者构造或推断，均注明。

### 11.1 引文复核（讲义中引用的英文短句，均在 v5 全文中逐字找到）

| 讲义引用 | 位置 | 结果 |
|---|---|---|
| “needs to cache $2n_hd_hl$ elements for each token” | §2.1.1 | 一致（原句前半为 “so MHA needs to cache …”） |
| “performance does not match MHA” | §2.1 | 一致（原文 “their performance does not match MHA”） |
| “we even do not need to compute keys and values out for attention” | §2.1.2 | 一致。**更正**：讲义初稿误写成 “so we even do not need …”，原句没有 “so”，前面是 “since … can be absorbed into $W^O$,”；已改 |
| “only needs to cache $\mathbf c_t^{KV}$” | §2.1.2 | 一致 |
| “lie between $W^Q$ and $W^{UK}$”，“matrix multiplication does not obey a commutative law”，“significantly hinder the inference efficiency” | §2.1.3 | 一致 |
| “additional multi-head queries”，“a shared key” | §2.1.3 | 一致 |
| “even if it cannot reduce the KV cache” | §2.1.2 | 一致 |
| “equal to GQA with only 2.25 groups”，“regardless of the storage precision” | 表 1 标题、§2.1.4 | 一致 |
| “improved version of FlashAttention-2”，“6 bits on average” | §3.1.3、§3.2.3 | 一致 |
| “we do not need to compute keys and values out for each query” | 附录 C（v5） | 一致；“it can be completed offline at once” 在 v5 中不存在，只在 v1–v4（与 0.3 节一致） |
| “jointly trained with the model” | `isca25` §2.1.2 | 一致 |
| “comparable to standard Multi-Head Attention” | `dsv3` §2.1.1 | 一致 |
| “the MHA mode is used for training and prefilling” | `dsv32` 附录 A | 一致 |

### 11.2 讲义中新增的派生数字（笔者计算，Python 复算）

| 量 | 表达式 | 值 | 讲义位置 |
|---|---|---|---|
| V2 形状用 GQA-8 的每 token 缓存（bf16） | `2*8*128*60*2` | 245,760 B；128K 时 32.21 GB | 图 9.1、§8 表 |
| V2 形状用 MQA 的每 token 缓存（bf16） | `2*128*60*2` | 30,720 B；128K 时 4.03 GB | 图 9.1、§8 表 |
| 图 9.1 的 80 GB 参考线 | A100 80GB / H100 80GB 的显存 | 80 GB | 取自 `lesson-04-doc.md` 3.2 节（NVIDIA 规格页），H800 的显存本课未核实，故不写 H800 |
| 吸收的交叉点（每步重新展开 vs 吸收） | `33554432/(33636352-278528)` | $T^*=1.0059$；$F_③(1)=33{,}832{,}960 > F_①(1)=33{,}636{,}352$ | §3 警告框、练习 12 |
| $T=32{,}768$ 的解码账单（一层） | `33636352*T`、`81920*T`、`33554432+278528*T` | 1,102.2 / 2.68 / 9.16 GFLOPs；读缓存 37.7 MB vs 2.68 GB | 练习 6 |
| 128K 读一遍缓存的时间下界（3.35 TB/s） | `bytes*131072/3.35e12` | V2 bf16 2.70 ms；6 bit 1.01 ms；67B 15.2 ms；V2 形状 MHA 153.8 ms | 练习 7 |
| 8 路张量并行时的算术强度 | `16*(576+512)/576` | 30.2 FLOP/B | §7 小练习 |
| 达到降频屋脊点所需头数（$s_q=1$） | `258.2*576/1088` | 136.7 | §7 小练习 |
| V2-Lite 预乘 $(W^{UK}_i)^\top W^Q_i$ vs 改变顺序 | `16*512*2048` vs `16*128*2048+16*128*512`；运算 `2*512*2048*16` vs `2*2048*128*16+2*128*512*16` | 参数 16,777,216 vs 5,242,880；运算 33,554,432 vs 10,485,760（都是 3.2 倍） | 练习 8 |
| 位置键不共享时的每层缓存 | `512+128*64` | 8,704（576 的 15.1 倍） | §5 算例、练习 3 |
| MLA 的 Q/K/V 投影参数（每层） vs MHA 的 $W^Q,W^K,W^V$ | 45,613,056 + 19,726,336；`3*16384*5120` | 65.34M vs 251.66M；$W^O$ 占整层 56.2% | §6 算例 |
| 两种模式预填充 $T=1024$ | `81920*T(T+1)/2+33554432*T` 等 | 77.4 / 180.5 GFLOPs | §7 算例 |
| FlashMLA 相对博客硬件数字 | `3000/3350`、`660/865` | 89.6%、76.3% | §7 旁注（粗算，口径不同） |

### 11.3 讲义里为教学构造的数值例子（笔者构造，Python 验证）

- **吸收的小例子**（§3）：$W^{UK}_i=\begin{pmatrix}1&0&2\\0&1&-1\end{pmatrix}$，$W^{UV}_i=\begin{pmatrix}1&1&0\\0&0&1\end{pmatrix}$，$\mathbf c_1=(1,1,0)$，$\mathbf c_2=(0,2,1)$，$\mathbf q^C=(1,2)$，$p=(0.25,0.75)$。
  展开：$\mathbf k_1=(1,1)$、$\mathbf k_2=(2,1)$，分数 3、4；吸收：$\tilde{\mathbf q}=(1,2,0)$，分数 3、4。值：两条路都得 $(2,0.75)$。
- **RoPE 反例的补充小练习**（§4）：同 4.6 节设定、$\theta=\pi/2$，再加 $j=0$、$c_0=3$：$\mathbf R(-3\pi/2)(3,1)^\top=(-1,3)$，吸收查询 5（与 $j=3$ 相同纯属巧合），分数 15。
- **解耦 RoPE 的位置段例子**（§5）：在 4.6 节的内容段上加 2 维位置段，$\theta=\pi/2$，$t=3$；旋转前位置键都是 $(1,0)$，缓存转好的 $(0,1),(-1,0),(0,-1)$；
  位置查询旋转前 $(1,1)$，按 $t=3$ 转 $3\pi/2$ 得 $(1,-1)$；位置分数 −1、−1、1；总分 4、9、−4。与相对位置公式 $(1,1)^\top\mathbf R(-\pi)(1,0)^\top=-1$ 一致。

### 11.4 取自其它课研究笔记的事实

- V2 每 token 激活约 21B 参数（`lesson-10-doc.md`，论文摘要），用于 §3 “$2N\approx42$ GFLOPs”的对照（该对照本身见 4.4 节）。
- 一层 MLA 149.2M 参数、60 层约 8.95B（`lesson-10-doc.md` 4.1 节），与本课图 9.9 的逐矩阵计数 149,227,520 一致。
- 第 8 课总账里“GRPO 每道题 64 个回答、1024 token 的 KV cache 30 GiB”（`lesson-08-deepseekmath-grpo.md` 的表），讲义封面引用它作承接，数字以第 8 课为准。
- V3 反向传播时重算 RMSNorm 与 MLA 上投影：“We recompute all RMSNorm operations and MLA up-projections during back-propagation”（`dsv3` §3.2.3，转引自 `lesson-12-doc.md` C31）。讲义 §6 用它作“查询压缩为什么能省激活显存”的旁证；V2 是否同样重算，V2 论文未写，讲义标为笔者推测。

### 11.5 教学审阅时新增（2026-09-24）

| 量 | 表达式 | 值 | 讲义位置 |
|---|---|---|---|
| 一个头预乘 $(W^{UK}_i)^\top W^{UQ}_i$ 的参数与每 token 运算 | `512*1536`、`2*512*1536` | 786,432；1,572,864 | 图 9.5 |
| 一个头改变顺序的参数与运算 | `128*1536+512*128`、`2*128*1536+2*512*128` | 262,144；524,288（都是预乘的 1/3；×128 头即 §3 算例的 33,554,432 与 67,108,864） | 图 9.5 |
| 重算展开后的查询（24,576 维）的运算：从 $\mathbf h_t$ vs 从 $\mathbf c^Q_t$ | `2*5120*24576`、`2*1536*24576` | 251,658,240 ≈ 2.52e8；75,497,472 ≈ 7.55e7（比值 0.30） | §6 正文（笔者推测的机制） |
| 位置键不共享时的缓存 vs 8 组 GQA | `8704/2048` | 4.25 倍 | §5 算例 |

图号：新增图 9.5（`l09-absorb-order.svg`，生成脚本 `gen/l09_absorb_order.py`）后，原图 9.5–9.14 顺延为 9.6–9.15；上面 11.2、11.4 节里的旧图号（如“图 9.9”）指顺延前的编号。

### 11.6 数学审校（2026-09-24）

讲义里所有数字已用 Python 逐项复算；另用随机小矩阵（numpy）验证了：键、值两侧的吸收，MQA 模式与 MHA 模式的输出一致，$\mathbf R_t^\top\mathbf R_j=\mathbf R_{j-t}$。下表是审校中改动的地方。

| 问题 | 原写法 | 改后 | 讲义位置 |
|---|---|---|---|
| “旋转挪不到潜向量一侧”的理由 | 对一般的 $W^{UK}_i$，不存在 $d_c$ 维旋转 $\mathbf R'$ 使 $\mathbf R_{j-t}W^{UK}_i=W^{UK}_i\mathbf R'$ | 单个头时这种矩阵其实存在：$W^{UK}_i$ 是 $128\times512$、行满秩，$\mathbf S_j=(W^{UK}_i)^+\mathbf R_jW^{UK}_i$ 满足 $\mathbf R_jW^{UK}_i=W^{UK}_i\mathbf S_j$。真正的障碍是缓存被 128 个头共用：需要同一个 $\mathbf S_j$ 对所有头成立，等价于 $W^{UK}$ 的列空间在 $\diag(\mathbf R_j,\dots,\mathbf R_j)$ 下不变，一般的 $W^{UK}$ 不满足（随机矩阵的最小二乘残差约 0.66） | §4 正文、ana、要点 |
| 退路一的代价 | “就是算法 ①，137.8 GFLOPs” | 只需重算键，值不旋转、仍可吸收：单是展开键 `2*4096*512*16384` ≈ 68.7 GFLOPs（连打分与值一侧约 69.4），吸收的约 59 倍 | §4 “两条退路” |
| 算法 ② 的缓存口径 | 81,920 B/token，未说明口径 | 注明按每头存完整 192 维键（HF 参考实现的做法）；共享的 $\mathbf k^R$ 只存一份时为 `2*(128*256+64)` = 65,664 B，约 57 倍 | §3 算例 |
| “缓存小一个数量级以上”（本笔记 E8 的措辞同此） | §8 结论、要点 | 表 9 小模型是 14%（约 7 倍），不到一个数量级；改为“只有 MHA 的 14%（小）到 4%（大），V2 完整形状下 1.76%” | §8 |
| 键值投影参数“少一个数量级” | | `167772160/19398656` = 8.6 倍，改为“将近一个数量级（约 8.6 倍）” | §2 算例 |
| “256 个键值头向量落在同一个 512 维子空间” | 维度说不通（每个头向量只有 128 维） | 改为“每个 token 所有头的键值拼成的 32,768 维向量落在同一个至多 512 维子空间（$[W^{UK};W^{UV}]$ 的列空间）”；练习 2、10 同改 | §2、练习 2、10 |
| 吸收比 ② 多算的倍数 | 要点写 $T=4096$ 时“3.4 倍” | $T=4096$ 时 1.1744/0.3355 = 3.5，$T\to\infty$ 趋于 3.4（11.7 节补上 ② 的固定项后为 1.1744/0.3691 = 3.18） | §3 要点、图 9.7 注记 |
| 图 9.4 的比例 | $\mathbf c^{KV}$ 竖条被放宽到 6 px | 按比例画成 4.7 px，与说明文字“高度与向量长度成正比”一致 | `gen/l09_joint_compression.py` |

### 11.7 数学审校（第二轮，2026-09-24）

全部数字再次用 Python 复算（脚本在审校者的临时目录，未入库）；十五张图的生成脚本全部重跑，未改动的十一张与仓库里的 SVG 逐字节相同。改动如下。

| 问题 | 原写法 | 改后 | 讲义位置 |
|---|---|---|---|
| 算法 ② 漏了固定成本 | $F_②=81{,}920\,T$；$T=4096$ 时 0.336 GFLOPs，③/② = 3.5 | ② 每步要把新 token 自己的潜向量展开一次才能存进缓存：`2*512*2*128*128` = 33,554,432，与 ③ 的固定成本相同。$F_②=33{,}554{,}432+81{,}920\,T$：$T=4096$ 时 0.369 GFLOPs，③/② = 3.18；$T=32{,}768$ 时 2.72（③/② = 3.37）；$T=131{,}072$ 时 10.77（3.39）；极限仍是 3.4。检验：$T=1$ 时 $F_①(1)=F_②(1)=33{,}636{,}352$，原式给出 81,920，自相矛盾。§7 的 MHA 模式解码本来就按 $F_0+aT$ 计，与更正后一致 | (9.11)、§3 算例与要点、图 9.7 与脚本、练习 6、本笔记 4.4 |
| “MHA 的键值只有 512 个自由度” | 主语写成了 MHA | 改为 MLA 的键值随 token 变化的自由度至多 512（MHA 上限 5120）；并说明子空间结论在加回 RMSNorm 后仍成立（RMSNorm 只逐分量缩放 $\mathbf c^{KV}_t$） | §2 |
| 预乘何时更贵 | “因为 $d_c>d_h$” / “中间维比两边都小” | 精确条件：预乘更贵 ⇔ $d_cd_c'>d_h(d_c+d_c')$ ⇔ $\frac1{d_h}>\frac1{d_c}+\frac1{d_c'}$；V2 两边之比正好是 3，V2-Lite（512、2048）是 3.2。反例：两端都是 200 时预乘更省（`200*200` = 40,000 < `128*400` = 51,200），所以“中间维比两端小”不充分 | §3 算例与要点、练习 8、图 9.5 脚注 |
| 缓存的是 RMSNorm 之前还是之后的潜向量 | §6 表“$\mathbf c^{KV}_t=W^{DKV}\mathbf h_t$，缓存，再 RMSNorm”；图 9.10 把缓存框画在 RMSNorm 之前 | 吸收与 MQA 模式要求缓存行就是 $W^{UK}$、$W^{UV}$ 所乘的向量，即归一化之后的潜向量（RMSNorm 每个 token 除以自己的 RMS，不能并进 $W^{UK}$）。核实：DeepSeek-V3 仓库 `inference/model.py` 的 MLA 类吸收分支写 `self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)`、`self.pe_cache[...] = k_pe.squeeze(2)`（`k_pe` 已 `apply_rotary_emb`），打分为两段 einsum 之和乘 `softmax_scale`（WebFetch 读取 raw.githubusercontent.com 上 main 分支，2026-09-24）。新增参考文献 `gh-v3` | §6 表与正文、图 9.10 与脚本 |
| RoPE 作用在 $\mathbf q^R_t$ 上的方式 | 只写了 $64\times64$ 的 $\mathbf R_{\Theta,t}$，而 $\mathbf q^R_t$ 是 8,192 维 | 注明逐头施加：$\mathbf q^R_{t,i}=\mathbf R_{\Theta,t}W^{QR}_i\mathbf c^Q_t$ | (9.13) 注 |
| “93.3%”的对数份额 | 只给结论 64% / 36% | 写出公式 `ln(389120/69120)/ln(389120/25920)` = 1.728/2.709 = 0.638，并说明为什么用对数（乘法因子按顺序算百分点会依赖顺序） | (9.24) 注 |
| 图 9.12 带宽斜线的文字角度 | 固定旋转 33.5° | 按两轴每十倍的像素数算出实际斜率（约 28.3°），文字与斜线平行 | `gen/l09_roofline.py` |

### 11.8 事实与引文核查（2026-09-24）

逐条列出讲义里的事实性论断（数字、规格、日期、引文、归属），与本笔记对照；本笔记没有覆盖的，回到一手来源核实。
论文用 `arxiv.py grep`（LaTeXML 全文）；arXiv 元数据因 API 限流（HTTP 429）改用 WebFetch 读 abs 页；模型配置用 WebFetch 读 Hugging Face 上的 raw `config.json`。

**新核实的事实（讲义用到、本笔记此前没有记录的）**

| 讲义中的论断 | 来源与原文 | 结果 | 讲义位置 |
|---|---|---|---|
| V2 沿用 DeepSeek 67B 的做法用 RoPE | `dsv2` §2.1.3：“Following DeepSeek 67B (DeepSeek-AI, 2024), we intend to use the Rotary Position Embedding (RoPE) (Su et al., 2024) for DeepSeek-V2.” | 一致 | §4 开头 |
| 5.76 倍吞吐的测法：按线上 67B 服务的提示与生成长度分布 | `dsv2` §3.2.3：“We evaluate the generation throughput of DeepSeek-V2 based on the prompt and generation length distribution from the actually deployed DeepSeek 67B service.” 紧接 “On a single node with 8 H800 GPUs, …” | 一致；该句位于 FP8 权重与 6 bit KV 的部署说明之后 | §8 警告框 |
| V2-Lite：27 层、$d=2048$、16 头、$d_h=128$、$d_c=512$、不压缩查询、$d_h^R=64$；15.7B 总参数、2.4B 激活 | `dsv2` 附录 B：“DeepSeek-V2-Lite has 27 layers and a hidden dimension of 2048. It also employs MLA and has 16 attention heads, where each head has a dimension of 128. Its KV compression dimension is 512, but slightly different from DeepSeek-V2, it does not compress the queries. For the decoupled queries and key, it has a per-head dimension of 64.”；“comprises 15.7B total parameters, of which 2.4B are activated for each token.”（引言也有同样的 15.7B / 2.4B） | 一致；表 9 小 MLA 模型也是 15.7B / 2.4B | §2 小练习、§6、练习 5 |
| DeepSeekMoE 16B：28 层、隐藏维 2048、16 头、每头 128 维的 MHA | `dsmoe`（arXiv 2401.06066）§5.1.2 Hyper-Parameters：“For DeepSeekMoE 16B, we set the number of Transformer layers to 28 and the hidden dimension to 2048. We employ the multi-head attention mechanism with a total of 16 attention heads, where each head has a dimension of 128.” | 一致（第 3.3 节原注“论文表未在本次核对”，现已核对正文） | 图 9.13 及其参考文献卡 |
| Qwen2.5-72B：80 层、64 个查询头 / 8 个键值头 | `qwen25`（arXiv 2412.15115）表 1 “Model architecture and license of Qwen2.5 open-weight models”：72B 行 “80 / 64 / 8”。每头维度表里没有；Hugging Face `Qwen/Qwen2.5-72B` 的 `config.json`：`hidden_size` 8192、`num_attention_heads` 64、`num_key_value_heads` 8、`num_hidden_layers` 80，故 $d_h=128$ | 一致；`2*8*128*80*2` = 327,680 B | 图 9.13、参考文献卡 |
| Llama 3 405B：126 层、模型维 16,384、128 头、8 个键值头 | `llama3`（arXiv 2407.21783）表 3（Key/Value Heads 一行 8 / 8 / 8）与正文 “Llama 3 405B uses an architecture with 126 layers, a token representation dimension of 16,384, and 128 attention heads” | 一致；$d_h=16384/128=128$，`2*8*128*126*2` = 516,096 B | 图 9.13、参考文献卡 |
| V3 反向时重算 RMSNorm 与 MLA 上投影 | `dsv3` §3.2.3：“We recompute all RMSNorm operations and MLA up-projections during back-propagation, thereby eliminating the need to persistently store their output activations.” | 一致（11.4 节原为转引，现直接核对） | §6 |
| V3.2 的层数（FP8 格式 656 B × 61 层） | `dsv32` §2：“DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp. Compared with DeepSeek-V3.1-Terminus, … the only architectural modification of DeepSeek-V3.2 is the introduction of DeepSeek Sparse Attention (DSA) through continued training.”；Hugging Face `deepseek-ai/DeepSeek-V3.2` 的 `config.json`：`num_hidden_layers` 61、`kv_lora_rank` 512、`qk_rope_head_dim` 64、`hidden_size` 7168、`num_attention_heads` 128 | 一致 | §7 FP8 算例、图 9.13 |
| ISCA '25 属 Industry Track | arXiv 2505.09343 abs 页 Comments：“The definitive version appeared as part of the Industry Track in Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA '25)”；Journal-ref “ISCA '25, June 2025”；v1 2025-05-14，v2 2025-12-23；15 位作者 | 一致 | 参考文献卡 |
| GQA 发表于 EMNLP 2023 | arXiv 2305.13245 abs 页 Comments：“Accepted at EMNLP 2023. Added to related work”（v3 2023-12-23） | 一致（第 8 节第 10 条中 GQA 部分由此解决；出版页仍未读） | 参考文献卡 |
| 参考文献卡作者 | Qwen2.5：abs 页作者以 “Qwen” 开头，首位个人作者 An Yang；Llama 3：首位作者 Aaron Grattafiori（559 人） | 一致 | 参考文献卡 |
| 第 8 课的 “30 GiB” | `lesson-08-deepseekmath-grpo.md`：每题 64 个回答、“每个最长 1,024 个 token（论文的最大长度设置）”，$64\times1024\times480$ KiB = 30 GiB | 30 GiB 是按满长度算的上界 | 封面 |
| 67B 原生上下文 4K | `lesson-06-doc.md` 表：DeepSeek LLM Table 2 与 config `max_position_embeddings` 均为 4096 | 一致 | §1 算例、练习 4 |

**改动**

| 问题 | 原写法 | 改后 |
|---|---|---|
| 第 8 课数字的口径 | “每个 1024 个 token，光是 KV cache 就要 30 GiB” | “每个最长 1024 个 token，按满长度算……30 GiB”（与第 8 课一致） |
| 引用挂错（相邻论断替代） | 图 9.13 说明里 “V3.2 与 V3 同为 61 层” 挂在 `[@dsv2; @flashmla]` 上，两者都不支持层数；§7 算例同句无出处 | 图注只写 “每层 656 字节 × 61 层 [@flashmla]”，6 bit 单独挂 `[@dsv2]`；§7 算例改为 “V3.2 相对 V3.1-Terminus 唯一的架构改动是加上 DSA [@dsv32]……配置文件 `num_hidden_layers = 61`” |
| 质量结论过强 | 练习 10：“在 DeepSeek 的实验里它没有带来可见的质量损失” | “总体上没有带来质量损失（8 项对比里 7 项 MLA 更高，小模型的 C-Eval 略低）”，与勘误 E8 一致 |
| `qwen25` 卡片的复算依据不全 | “用它的层数与键值头数复算” | 补 “每头 128 维，报告未列，按其 Hugging Face 配置 8192/64 算得” |
| 数学课链接不指向具体页 | M02、M03、M06、M08、M10 只链到整课 | M02 §7（结合律与吸收）、M02 §5（最优低秩近似）、M03 §5（相对角恒等式）、M06 §4（计算图与反向传播：存激活或重算）、M08 §5（累加误差）、M10 §3–§4（算术强度、屋顶线） |

**核对无误、未改动的部分（摘要）**：摘要与 §3.2.3 的 42.5% / 93.3% / 5.76× / 50K / 100K；表 1、表 8、表 9 全部数字；附录 A 署名；v1–v5 版本史；FlashMLA 首发与当前 README 的数字、MQA/MHA 模式维度、656 字节格式；FlashMLA 博客的 990 / 865 TFLOPS、3.35 TB/s、阈值 128；HF 代码的 `softmax_scale`、`kv_a_layernorm`、展开后缓存、RoPE 交错配对；gh-v3 的 `kv_norm(kv)`；十五张图里的全部数字与文字标注（逐项与本笔记第 3、4 节及 11.x 节对照，并用 Python 复算了图 9.14 的三段差值 279.68 / 40.32 / 43.2 KB）。所有跨课链接都用了 writer-brief 里的真实文件名，#pN 与目标课的页序一致（第 5 课 §2 RoPE、§4 RMSNorm、§8 总账；M01 §2–§6）。
