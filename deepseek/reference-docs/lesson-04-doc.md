# 第 04 课研究笔记：推理——自回归解码、KV cache 与带宽墙

> 服务于 `deepseek/src/lesson-04-inference-kv-cache.md`（short：KV cache 与解码）。建立于 2026-09-24。
> 所有引文都是从原文（LaTeXML 全文、出版社页面、厂商数据表、config.json）逐字核对过的，按版权要求只摘取
> 不超过约 15 个英文词的关键短语，其余用中文转述；公式按原样转录。所有派生数字都在 Python 里复算过，
> 表达式见 §4。凡是没能在一手来源里看到的东西，一律放进 §8。
>
> 读法：写课文时先看 §1（论断）与 §4（数字），画图看 §5，出题看 §6，**动笔前务必看 §7 勘误**。

---

## 0. 覆盖范围与所读版本

### 0.1 论文

| key | 标题 | 作者 | arXiv（所读版本） | 发表状态 | 日期 |
|---|---|---|---|---|---|
| shazeer2019 | Fast Transformer Decoding: One Write-Head is All You Need | Noam Shazeer | 1911.02150 **v1**（唯一版本） | arXiv 预印本（未见正式发表） | 2019-11-06 |
| ainslie2023 | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai | 2305.13245 **v3**（v1 2023-05-22，v2 2023-10-24，v3 2023-12-23） | EMNLP 2023，pp. 4895–4901，doi 10.18653/v1/2023.emnlp-main.298（ACL Anthology 核对） | 2023 |
| pope2023 | Efficiently Scaling Transformer Inference | Pope, Douglas, Chowdhery, Devlin, Bradbury, Levskaya, Heek, Xiao, Agrawal, Dean | 2211.05102 **v1** | MLSys 2023（Proceedings of MLSys 5，proceedings.mlsys.org 核对） | 2022-11-09 |
| dao2022 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao, Fu, Ermon, Rudra, Ré | **2205.14135** **v2**（v1 2022-05-27，v2 2022-06-23） | NeurIPS 2022（本次未去会议页核对，见 §8） | 2022 |
| shah2024 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao | 2407.08608 **v2** | 预印本（只用来引 H100 稠密峰值） | 2024-07-11 |
| kwon2023 | Efficient Memory Management for Large Language Model Serving with PagedAttention | Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica | 2309.06180 **v1** | SOSP 2023，doi 10.1145/3600006.3613165（论文首页脚注核对） | 2023-09-12 |
| yu2022 | Orca: A Distributed Serving System for Transformer-Based Generative Models | Yu, Jeong, Kim, Kim, Chun | 无 arXiv | OSDI 2022（USENIX 页面核对；**只读了摘要页**） | 2022 |
| leviathan2023 | Fast Inference from Transformers via Speculative Decoding | Leviathan, Kalman, Matias | 2211.17192 **v2**（v1 2022-11-30，v2 2023-05-18） | ICML 2023（Oral），PMLR 202:19274–19286（PMLR 页面核对） | 2023 |
| chen2023 | Accelerating Large Language Model Decoding with Speculative Sampling | Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper（DeepMind） | 2302.01318 **v1** | arXiv 预印本（未查到正式发表版本） | 2023-02-02 |
| holtzman2020 | The Curious Case of Neural Text Degeneration | Holtzman, Buys, Du, Forbes, Choi | 1904.09751 **v2**（v1 2019-04-22，v2 2020-02-14） | ICLR 2020（arXiv comment 字段） | 2019/2020 |
| fan2018 | Hierarchical Neural Story Generation | Fan, Lewis, Dauphin | 1805.04833 **v1** | ACL 2018（Long Papers），pp. 889–898，doi 10.18653/v1/P18-1082（ACL Anthology 核对） | 2018 |
| williams2009 | Roofline: An Insightful Visual Performance Model for (Floating-Point Programs and) Multicore Architectures | Williams, Waterman, Patterson | 无 arXiv | CACM 52(4):65–76, 2009（引用信息取自 FlashAttention 参考文献 [85]）；**实际读的是技术报告 UCB/EECS-2008-134（2008-10-17）** | 2008/2009 |
| kaplan2020 | Scaling Laws for Neural Language Models | Kaplan et al. | 2001.08361 **v1** | 预印本 | 2020-01-23 |
| brown2020 | Language Models are Few-Shot Learners | Brown et al. | 2005.14165 **v4** | NeurIPS 2020（未到会议页核对，见 §8） | 2020-05-28（v1） |
| touvron2023a | LLaMA: Open and Efficient Foundation Language Models | Touvron et al. | 2302.13971 **v1** | 预印本 | 2023-02-27 |
| touvron2023b | Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al. | 2307.09288 **v2** | 预印本 | 2023-07 |
| grattafiori2024 | The Llama 3 Herd of Models | Grattafiori, Dubey, et al. | 2407.21783 **v3** | 预印本 | 2024-07-31 |
| deepseek2024llm | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism | DeepSeek-AI | 2401.02954 **v1** | 预印本 | 2024-01-05 |
| deepseek2024v2 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek-AI | 2405.04434 **v5** | 预印本 | 2024-05-07（v5 2024-06-19） |
| deepseek2024v3 | DeepSeek-V3 Technical Report | DeepSeek-AI | 2412.19437 **v2**（v1 2024-12-27，v2 2025-02-18） | 预印本 | 2024-12 |
| zhao2025 | Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures | Zhao, Deng, Ruan, Dai, Gao, Li, et al. | 2505.09343 **v2**（v1 2025-05-14，v2 2025-12-23；v1 也抽查过 Table 1 与 H800 段落，与 v2 相同） | ISCA 2025 Industry Track，doi 10.1145/3695053.3731412 | 2025 |

### 0.2 制品（artefact，无论文）

| key | 内容 | 来源 | 读取日期 |
|---|---|---|---|
| nvidia-a100-ds | A100 数据表（JUN21 版） | nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf | 2026-09-24 |
| nvidia-h100-page | H100 产品页规格表（SXM / NVL 两栏） | nvidia.com/en-us/data-center/h100/ | 2026-09-24 |
| hf-ds-llm-7b / 67b | `config.json`（base 与 67b-chat 相同） | huggingface.co/deepseek-ai/deepseek-llm-{7b,67b}-base/raw/main/config.json | 2026-09-24 |
| hf-llama-mirrors | Llama-2-7B/70B、Llama-3-8B/70B `config.json` | **第三方镜像** huggingface.co/NousResearch/…（meta-llama 官方 repo 返回 HTTP 401，需要登录） | 2026-09-24 |
| hf-ds-v3 | DeepSeek-V3 `config.json` | huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json | 2026-09-24 |
| hf-qwen25-72b | Qwen2.5-72B `config.json` | huggingface.co/Qwen/Qwen2.5-72B/raw/main/config.json | 2026-09-24 |
| hf-opt-13b | OPT-13B `config.json` | huggingface.co/facebook/opt-13b/raw/main/config.json | 2026-09-24 |

### 0.3 本课覆盖的主题 → 来源

| 主题 | 主要来源 | 旁证 |
|---|---|---|
| prefill / decode 的定义与术语 | pope2023 §2、§2.2 | kwon2023 §2.2（“prompt phase / autoregressive generation phase”）；deepseek2024v3 §3.4（“prefilling / decoding stage”） |
| KV cache 的定义、正确性 | kwon2023 §2.2；pope2023 §1 | zhao2025 §2.1.2 |
| 内存公式 $2n_hd_hl$ | deepseek2024v2 §2.1.1、Table 1 | kwon2023 §3（OPT-13B 800 KB 例子）；zhao2025 Table 1 |
| 带宽墙 | shazeer2019 §1、§2.4.1；pope2023 §2 | chen2023 §3；deepseek2024v3 §3.4.2；zhao2025 §2.1.2 |
| 算术强度 / roofline | williams2009 §3；dao2022 §2.1 | 厂商数据表 |
| MQA | shazeer2019 §3、§3.1、§4 | pope2023 §3.3、§4.2、Table 1 |
| GQA | ainslie2023 §2、§3 | touvron2023b A.2.1；grattafiori2024 §3.2；deepseek2024llm §2.2 |
| 采样：温度、top-k、top-p | holtzman2020 §3（式 2–4）；fan2018 §5.4 | leviathan2023 §2.2（各种采样都可看成“调整后的分布”） |
| 投机解码 | leviathan2023 §2–§4、附录 A | chen2023 §3–§6、Theorem 1；deepseek2024v3 §5.4.3（MTP 接受率）；zhao2025 §2.3.3 |
| PagedAttention / 连续批处理 | kwon2023 §1–§4、§7.2 | yu2022（摘要） |
| 硬件 | nvidia-a100-ds；nvidia-h100-page；shah2024 §2.2、§3.1 | zhao2025 §4.1、§4.3（H800）；deepseek2024v3 §3.2.2 |

---

## 1. 核心论断与原文引文

编号 C1…；每条给出：论断（中文）→ 逐字短引（英文）→ 位置。

### A. prefill 与 decode

**C1**（术语来源）Pope 等人把推理延迟拆成两段：处理一开始就在手的输入 token（他们称为 prefill），和逐个自回归生成输出 token（他们称为 decode）。
> “(which we call “prefill”)” … “(which we term “decode”)” — pope2023 §2 第一段。

**C2**（为什么两段性能不同）prefill 可以在 $L_{input}$ 上并行，一次前向处理全部 $B\times L_{input}$ 个 token；decode 必须对 $L_{gen}$ 串行循环，每步对 batch 里每个样本采一个新 token。
> “Since prefill can run in parallel over L_input, but decode must run sequentially” — pope2023 §2.2。

**C3**（同一概念在 vLLM 论文里的叫法）PagedAttention 论文把两段叫 prompt phase 与 autoregressive generation phase；后者“多用矩阵-向量乘法”，严重吃不满 GPU 算力、变成访存受限。
> “this phase severely underutilizes GPU computation and becomes memory-bound” — kwon2023 §2.2。

**C4**（DeepSeek 的叫法与实践）DeepSeek-V3 线上部署把 prefilling 与 decoding 两个阶段分开部署（PD 分离）；prefilling 最小部署单元 4 节点 32 GPU，decoding 最小部署单元 40 节点 320 GPU。
> “separates the prefilling and decoding stages” — deepseek2024v3 §3.4；“4 nodes with 32 GPUs” §3.4.1；“40 nodes with 320 GPUs” §3.4.2。

**C5**（PD 分离的早期雏形）Pope 等人在低延迟场景里把 batch-1 的 prefill 和 batch-32~64 的 decode 组合，并指出可以“把一个 batch-1 的 prefill 服务器流水接到 batch-64 的 decode 服务器”。
> “pipelining a batch-1 prefill server into a batch-64 decoding server” — pope2023 §4.4（Tables 2–3 之后）。

### B. KV cache：是什么、为什么是精确的

**C6**（术语）每层注意力的键、值张量在整个解码期间都得留在显存里，Pope 等人称之为 KV cache。
> “which we refer to as the KV cache” — pope2023 §1。

**C7**（只算新 token 的 K、V）decode 第 $t$ 次迭代只为新 token 计算 $k_{n+t}, v_{n+t}$，更早位置的 K、V 都是前几次迭代缓存下来的。
> “only the new key and value vector k_{n+t} and v_{n+t} are computed at this iteration” — kwon2023 §2.2。

**C8**（KV cache 不是“按 token id 查表”）同一个 token 出现在序列的不同位置时，它的 KV cache 不同，因为每个 token 的 KV 依赖它之前的所有 token（经过前面各层）。
> “the KV cache of one token depends on all its previous tokens” — kwon2023 §2.2。
> 课文用法：这正是“纪要本”类比的边界之一——纪要记的是“这个人在这场会的这个时刻说的话”，不是“这个人的名片”。

**C9**（DeepSeek 对 KV cache 的表述）缓存历史 token 的 K、V，每步只算当前 token 的 K、V 再与缓存做注意力，使每个 token 的生成复杂度降到 $O(N)$，但计算从 GEMM 变成 GEMV，算/存比低得多，于是变成访存瓶颈。
> “the computation shifts from GEMM to GEMV, which has a much lower compute-to-memory ratio” — zhao2025 §2.1.2。

### C. 内存公式

**C10**（MHA 每 token 缓存元素数）MHA 每个 token 要缓存 $2n_hd_hl$ 个元素。
> “MHA needs to cache 2n_hd_hl elements for each token” — deepseek2024v2 §2.1.1（式 (8) 之后）。

**C11**（四种注意力的对照表，按元素数计、与精度无关）MHA $2n_hd_hl$（Strong）、GQA $2n_gd_hl$（Moderate）、MQA $2d_hl$（Weak）、MLA $(d_c+d_h^R)l\approx\frac92 d_hl$（Stronger）。
> “measured by the number of elements, regardless of the storage precision” — deepseek2024v2 Table 1 注。
> 注意“Capability”一列（Strong/Moderate/Weak/Stronger）是 DeepSeek 的定性判断，不是测量值。

**C12**（按字节算的一个真实例子）OPT-13B 每个 token 的 KV cache 要 800 KB：$2\times5120\times40\times2$；序列最长 2048 时单个请求可达 1.6 GB。
> “the KV cache of a single token demands 800 KB of space” — kwon2023 §3。复算见 §4（其实是 800 KiB）。

**C13**（KV cache 可以比权重还大）Pope 等人：对 500B+ 的 MHA 模型，batch 512、上下文 2048 时 KV cache 共 3 TB，是参数的 3 倍；每生成一个 token 都要把它从片外内存读一遍，期间计算核心基本空闲。
> “the KV cache totals 3TB, which is 3 times the size of the model’s parameters” — pope2023 §2.1。复算见 §4.3。

**C14**（什么时候权重主导、什么时候 KV 主导）小 batch、短序列时加载权重的时间占主导；batch 与序列变大（例如 2048+ token、batch 512+）时加载 KV cache 的时间占主导。
> “At small batch sizes and sequence lengths, the time to load weights dominates.” — pope2023 §2 “Memory costs”。

**C15**（DeepSeek 自己的对照）DeepSeek-V3（MLA）每 token KV cache 70.272 KB，Qwen-2.5 72B（GQA）327.680 KB，LLaMA-3.1 405B（GQA）516.096 KB（BF16）。
> 表题：“KV cache size comparison (BF16 precision)” — zhao2025 Table 1（§2.1.2）。**表中 7.28x 是算错的，应为 7.34x，见 §7 E13。**

### D. 带宽墙、算术强度与 roofline

**C16**（问题的原始陈述）增量推理的速度受限于反复重新加载巨大 K、V 张量所需的显存带宽。
> “limited by the memory bandwidth necessary to reload the large "keys" and "values" tensors” — shazeer2019 §1。

**C17**（Shazeer 的比值，批量训练）批量多头注意力：算术运算 $\Theta(bnd^2)$，访存 $O(bnd+bhn^2+d^2)$，访存/运算之比 $O(\frac1k+\frac1{bn})$。
> “the ratio of memory access to arithmetic operations is O(1/k + 1/bn)” — shazeer2019 §2.3.1。

**C18**（Shazeer 的比值，增量解码）$n$ 次调用合计：运算 $\Theta(bnd^2)$，访存 $\Theta(bn^2d+nd^2)$（前项来自 K、V，后项来自投影矩阵），比值 $\Theta(\frac nd+\frac1b)$；当 $n\approx d$ 或 $b\approx1$ 时比值接近 1，带宽成为主要瓶颈。
> “When n≈d or b≈1, the ratio is close to 1” — shazeer2019 §2.4.1。

**C19**（两项各自的解法）$\frac1b$ 项容易：只要显存允许就加大 batch；$\frac nd$ 项难，它来自每步重新加载 K、V。
> “The 1/b term is the easier one - we can just use a larger batch size” — shazeer2019 §2.4.1。

**C20**（roofline 的原始定义用的是 operational intensity）Williams 等人用“operational intensity”表示每字节 DRAM 流量对应的运算数，并**明确拒绝**用 arithmetic intensity 这个词：后者（及 machine balance）量的是处理器与 cache 之间的流量，而他们要量 cache 与 DRAM 之间的流量；另外他们希望模型也适用于非算术运算。
> “we use the term operational intensity to mean operations per byte of DRAM traffic” — williams2009（TR 版）§3。
> arithmetic intensity 的出处是其参考文献 [16]：M. Harris, “Mapping Computational Concepts to GPUs,” ACM SIGGRAPH Tutorials, Ch. 31, 2005。

**C21**（roofline 公式与脊点）可达性能 = min(峰值浮点性能, 峰值带宽 × operational intensity)；两条线的交点叫 ridge point，其横坐标是达到峰值性能所需的最小 operational intensity。
> “Attainable GFlops/sec = Min(Peak Floating Point Performance, Peak Memory Bandwidth x Operational Intensity)” — williams2009 §3。
> “The x-coordinate of the ridge point is the minimum operational intensity required” — williams2009 §3。

**C22**（ML 系统文献里的用法）FlashAttention 把“每字节访存对应的算术运算数”叫 arithmetic intensity，并把 roofline 称作“arithmetic intensity 的 Roofline 模型”——与 Williams 的原始命名不符。
> “the number of arithmetic operations per byte of memory access” — dao2022 §2.1；“the Roofline model of arithmetic intensity [85]” — dao2022 附录 A。
> 课文做法：沿用 style-guide 的“算术强度（arithmetic intensity）”，但在 `::: pre` 里交代 Williams 原本叫 operational intensity 以及原因（见 §7 E2）。

**C23**（compute-bound 与 memory-bound 的分类）运算按计算与访存的平衡分为 compute-bound 与 memory-bound；大内维的矩阵乘是前者的典型，逐元素运算与归约（softmax、layer norm 等）是后者。
> “Compute-bound: the time taken by the operation is determined by how many arithmetic operations” — dao2022 §2.1。

**C24**（A100 的内存层级，FlashAttention 的说法）A100 有 40–80GB HBM，带宽 1.5–2.0TB/s；108 个 SM 每个 192KB 片上 SRAM，带宽约 19TB/s。
> “40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s” — dao2022 §2.1。

**C25**（Pope 的两个时间）把权重与 KV cache 从 HBM 搬进计算核心所需的时间叫“memory time”；一个 $N$ 参数的纯解码器模型每见一个 token 前向要 $2N$ 次矩阵乘 FLOPs，峰值算力下完成它们的时间叫“compute time”。
> “requires 2N matmul FLOPs in the forward pass per token seen” — pope2023 §2 “Compute costs”。
> 本课的“延迟下界 = max(memory time, compute time)”是把这两个定义与 roofline 合起来的**推导**，Pope 论文本身没有写出这个 max 公式（见 §2.6）。

**C26**（注意力 FLOPs 小，但带宽不小）注意力里的矩阵乘对大模型每 token 增加的 FLOPs 小得多、常被忽略；但因为 KV cache 每条序列各一份（不像权重在 batch 内共享），它仍占显存容量与带宽的大头。
> “the KV cache is unique for each sequence in the batch” — pope2023 §2 “Compute costs”。

**C27**（DeepSeek 自己的带宽墙表述）V3 的 decoding 阶段每个专家的 batch 较小（通常 256 token 以内），瓶颈是访存而非计算。
> “the bottleneck is memory access rather than computation” — deepseek2024v3 §3.4.2。

**C28**（硬上限）所有参数至少要流过一个加速器一次，所以“模型大小 ÷ 所有芯片的总带宽”给出自回归采样速度的硬上限。
> “gives us a hard ceiling on the maximum auto-regressive sampling speed” — chen2023 §3。

### E. MQA

**C29**（定义）MQA 与 MHA 相同，只是各个头共享同一组键和值。
> “the different heads share a single set of keys and values” — shazeer2019 §3。

**C30**（MQA 的比值）增量 MQA：访存 $\Theta(bnd+bn^2k+nd^2)$，比值 $\Theta(\frac1d+\frac n{dh}+\frac1b)$，把“讨厌的” $\frac nd$ 项缩小了 $h$ 倍。
> “We have reduced the offensive n/d by a factor of h.” — shazeer2019 §3.1。

**C31**（质量：略差，但比“减头数/减维度”好得多）WMT14 EN-DE 上 MQA 比基线略差，但远比减少 $h$、$d_k$、$d_v$ 的各种替代方案接近基线；beam-4 测试集 BLEU 反而最高（28.5）。
> “slightly worse than the baseline, but much closer than any of the alternatives” — shazeer2019 §4.2。

**C32**（速度）TPUv2（8 核）、batch 1024、源/目标长度 128：MHA 解码器每 token 摊销 46 μs，MQA 3.8 μs；编码器 1.7 vs 1.5 μs；训练 13.2 vs 13.0 μs/token（Table 2）。

**C33**（MQA 在张量并行下的陷阱）按头切分时，共享的 K、V 必须在每块芯片上复制一份，MQA 省下的显存就没了；Pope 等人改为在 decode 时按 batch 切分注意力，于是每芯片 KV 加载量降为 $1/n_{chips}$。
> “must be replicated on each chip and the memory cost savings … are lost” — pope2023 §3.3。

**C34**（按 batch 切分只在 decode 划算）decode 时 Q/K/V 每条样本只有 1 个 token 而 KV cache 有很多个，花 all-to-all 通信换 KV 的访存时间很划算；prefill 时 K、V 的加载被 Q 里的大量 token 摊薄，不是瓶颈，所以 prefill 仍按头切分。
> “it is typically not profitable to shard attention over batch” — pope2023 §3.3（指 prefill）。

**C35**（上下文长度提升）PaLM 540B、64 芯片、30% 显存留给 KV cache：MHA（$d_{head}$=128）最大上下文 1320/330（batch 128/512），基线 MQA 660/165，优化 MQA 43,000/10,700。
> “Optimized multiquery attention enables up to 32x larger context lengths.” — pope2023 Table 1 表题。复算见 §4.3。

**C36**（长上下文下 MQA 的收益）优化布局的 MQA 能扩展到 8192–32,768 的序列长度（batch 512 与 128），注意力只占总运行时间的 8–31%。
> “with attention taking only 8–31% of total runtime” — pope2023 §4.2。

**C37**（MQA 何时最有用）Chen 等人认为 MQA 这类缩小 cache 的方法主要提升大 batch 的吞吐，而非延迟；大模型的带宽预算大部分被参数消耗。
> “most effective at maximising throughout (at larger batch sizes) instead of latency” — chen2023 §2（原文拼作 “throughout”）。

### F. GQA

**C38**（定义）GQA 把查询头分成 $G$ 组，每组共享一个键头和一个值头；GQA-1 等价于 MQA，GQA-H 等价于 MHA。
> “GQA-1, with a single group and therefore single key and value head, is equivalent to MQA” — ainslie2023 §2.2（截短）。

**C39**（uptraining 第一步：均值池化）把多头 checkpoint 转成 MQA/GQA 时，各头的 K、V 投影矩阵被均值池化成一个；这比只选一个头或随机初始化好（Figure 4 消融：Mean > First > Random）。
> “mean pooled into single projection matrices” — ainslie2023 §2.1。

**C40**（uptraining 第二步：少量续训）转换后按原预训练配方再训练原步数的比例 $\alpha$；摘要说用原预训练算力的 5%；$\alpha=0.05$ 约 600 TPUv3 chip-days。
> “using 5% of original pre-training compute” — ainslie2023 摘要；“approximately 600 TPUv3 chip-days” — §3.1 “Uptraining”。

**C41**（结果）T5-XXL 上：MHA-XXL 1.51 s/样本、平均分 47.2；MQA-XXL 0.24 s、46.6；GQA-8-XXL 0.28 s、47.1；MHA-Large 0.37 s、46.0（Table 1）。
> 结论句：“GQA achieves significant additional quality gains, achieving performance close to MHA-XXL” — ainslie2023 §3.2（截短）。

**C42**（为什么选 8 组）从 MQA 增加组数起初只带来小幅变慢，越接近 MHA 代价越大；选 8 组作为折中。
> “We selected 8 groups as a favorable middle ground.” — ainslie2023 §3.3 “Number of groups”。

**C43**（为什么大模型更适合 GQA）KV cache 随模型维度线性增长，而 FLOPs 与参数随维度平方增长；大模型头数多，MQA 砍得更狠；GQA 保持相同比例的缩减；标准分片会按分区数复制单个 KV 头，GQA 消除了这种浪费。
> “the KV-cache scales with model dimension while model FLOPs and parameters scale with the square” — ainslie2023 §2.2。

**C44**（局限）只在编码器-解码器（T5）上评估；GQA 不用于编码器自注意力；没有和从头训练的 GQA 对比。
> “we evaluate the impact of uptraining and GQA only on encoder-decoder models” — ainslie2023 Limitations。

**C45**（Llama 2 的选择与消融）Llama 2 的 34B 与 70B 用 GQA；30B 规模、150B token 的消融里 GQA 与 MHA 大多持平、平均优于 MQA；MQA 的 FFN 放大 1.33 倍、GQA 放大 1.3 倍以对齐参数量。8×80GiB A100 张量并行下，MHA 在上下文 256 时 batch 1024、上下文 2k 时 batch 128 就 OOM。
> “for the 34B and 70B Llama 2 models we chose to use GQA instead of MQA” — touvron2023b A.2.1（截短）；OOM 细节见 Figure 24 图注。

**C46**（Llama 3）所有 Llama 3 模型都用 8 个 KV 头的 GQA，以提升推理速度、缩小解码时的 KV cache。
> “We use grouped query attention (GQA; Ainslie et al. (2023)) with 8 key-value heads” — grattafiori2024 §3.2。

**C47**（DeepSeek LLM 67B 用 GQA）为优化推理成本，67B 用 GQA 而非 MHA；与多数 GQA 模型加宽 FFN 不同，67B 把参数加在深度上（95 层；7B 为 30 层）。
> “To optimize inference cost, the 67B model uses Grouped-Query Attention (GQA)” — deepseek2024llm §2.2。

### G. PagedAttention 与连续批处理（只点到为止）

**C48**（显存构成）13B 模型跑在 40GB A100 上：约 65% 显存是权重，近 30% 是请求的动态状态（KV cache）。
> “Approximately 65% of the memory is allocated for the model weights” — kwon2023 §1。

**C49**（碎片浪费）现有系统里只有 20.4%–38.2% 的 KV cache 显存真正存了 token 状态（预留、内部碎片、外部碎片三种浪费，Figure 3）。
> “only 20.4% - 38.2% of the KV cache memory is used to store the actual token states” — kwon2023 §1。

**C50**（做法）借鉴操作系统虚拟内存与分页，把每条序列的 KV cache 切成固定大小的 KV block，逻辑块经 block table 映射到不连续的物理块；浪费被限制在每个请求的最后一块之内；默认 block size 16。
> “vLLM sets its default block size as 16” — kwon2023 §7.2。

**C51**（效果）同等延迟下吞吐提高 2–4 倍（对比 FasterTransformer 与 Orca）。
> “improves the throughput of popular LLMs by 2-4× with the same level of latency” — kwon2023 摘要。

**C52**（连续批处理的出处）iteration-level scheduling：每次迭代结束就移除完成的请求、加入新请求，而不是等整个 batch 结束；来源 Orca（OSDI 2022）。
> “iteration-level scheduling” — yu2022 摘要（USENIX 页面）；kwon2023 §2.3 转述了其机制。

### H. 采样

**C53**（温度）给定 logits $u_{1:|V|}$ 与温度 $t$，softmax 重新估计为式 (4)；$t\in[0,1)$ 把分布推向高概率事件、降低尾部质量；温度采样的出处被 Holtzman 等人归于 Ackley et al. 1985。
> “shape a probability distribution through temperature (Ackley et al., 1985)” — holtzman2020 §3.3。

**C54**（top-k）Fan 等人生成故事时“从最可能的 $k=10$ 个候选里随机采样”，认为比 beam search 有效，也避免完全随机采样采到极低概率词。
> “We randomly sample from the k=10 most likely candidates” — fan2018 §5.4。

**C55**（top-p / nucleus 的定义）top-$p$ 词表 $V^{(p)}$ 是满足累计概率 $\ge p$ 的最小集合（式 2），再按 $p'$ 重新归一化（式 3）；候选集大小随分布形状动态伸缩。
> “the smallest set such that” — holtzman2020 §3.1（式 (2) 前）。

**C56**（为什么固定 $k$ 不好）有的上下文分布很平（几十上百个合理选项），有的很尖（质量集中在一两个 token）；小 $k$ 在平分布里太保守，大 $k$ 在尖分布里会把不合适的候选经重新归一化抬高。
> “the use of a constant k is sub-optimal across varying contexts” — holtzman2020 §3.2。

**C57**（常用取值）实践中 $k$ 很少超过 100，$p$ 通常在 $[0.9,1)$。
> “values of p are usually in [0.9,1)” — holtzman2020 §5。

**C58**（数据）GPT-2 Large、5000 段、最长 200 token：人类文本困惑度 12.38；Greedy 1.50（重复 73.66%）；Beam b=16 1.48；Pure Sampling 22.73；Top-k=40 6.88；Top-k=40,t=0.7 3.48（重复 8.86%）；Nucleus p=0.95 13.13、HUSE 0.97 最高（Table 1）。

**C59**（所有采样都能化成“调整后的分布”）argmax、top-k、nucleus、温度都可以写成从一个调整后的概率分布里标准采样；例如 argmax 等价于把非最大元素置零再归一化。投机解码因此只需处理“从分布里采样”这一种情况。
> “argmax sampling is equivalent to zeroing out non-max elements of the distribution and normalizing” — leviathan2023 §2.2。

### I. 投机解码

**C60**（动机：算力有富余）大模型推理的瓶颈常常不是算术而是带宽与通信，所以额外的计算资源可能是空闲的；用“投机执行”增加并发。
> “often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication” — leviathan2023 §1。

**C61**（接受-拒绝规则）要从 $p(x)$ 采样，改为从 $q(x)$ 采样：若 $q(x)\le p(x)$ 保留；若 $q(x)>p(x)$ 以概率 $1-\frac{p(x)}{q(x)}$ 拒绝，并改从 $p'(x)=norm(max(0,p(x)-q(x)))$ 重新采样；这样得到的 $x$ 恰好服从 $p$。
> “It’s easy to show (see Section A.1) that for any distributions p(x) and q(x) … indeed x∼p(x).” — leviathan2023 §2.3（截短；证明在附录 A.1）。

**C62**（每轮至少产出一个 token）目标模型每次并行运行至少产生一个新 token，所以串行调用次数最坏也不超过普通自回归，最多 $\gamma+1$ 个。
> “the number of serial runs of the target model can never, even in the worst case, be larger” — leviathan2023 §2.1（截短）。

**C63**（期望产出 token 数）在“各步接受率 $\beta$ 独立同分布、$\alpha=E(\beta)$”的简化假设下，一轮产出的 token 数是成功概率 $1-\alpha$、上限 $\gamma+1$ 的截断几何变量，期望为式 (1) $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$。
> “a capped geometric variable, with success probability 1−α and cap γ+1” — leviathan2023 §3.1。

**C64**（$\alpha$ 的闭式）$\beta=\sum_x\min(p(x),q(x))=1-D_{LK}(p,q)$（Theorem 3.5），$\alpha=E(\min(p,q))$（Corollary 3.6）；$D_{LK}$ 就是全变差距离（Lemma 3.3 的恒等式；“就是 TV 距离”是笔者的说法，论文称其为“a natural divergence”）。

**C65**（墙钟时间）期望加速比 $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$（Theorem 3.8），$c$ 为草稿模型单步耗时与目标模型之比；若 $\alpha>c$ 必存在有益的 $\gamma$，加速至少 $\frac{1+\alpha}{1+c}$（Corollary 3.9）。

**C66**（算术运算反而增加，访存减少）总运算量期望增加倍数 $\frac{(1-\alpha)(\gamma\hat c+\gamma+1)}{1-\alpha^{\gamma+1}}$（Theorem 3.11）；但目标模型的权重与 KV cache 每轮只读一次，访存按式 (1) 的倍数减少。
> “the total number of memory accesses can go down with our method” — leviathan2023 §3.4。

**C67**（实测）T5-XXL（11B）、batch 1、单个 TPU-v4：T5-small 做草稿时 EnDe 翻译 3.4X（temp=0）/2.6X（temp=1），CNN/DM 摘要 3.1X/2.3X（Table 2）；草稿比目标小约两个数量级通常最好；二元语法（bigram）草稿 $\alpha\approx0.2$，$\gamma=3$ 得 1.25X。
> “an out-of-the-box latency improvement of 2X-3X, without any change to the outputs” — leviathan2023 §1。
> “around two orders of magnitude smaller than M_p usually performed best” — leviathan2023 §3.6。

**C68**（Chen 等人独立的同一方法）DeepMind 的 speculative sampling（SpS）与 Leviathan 的工作同时、独立完成，核心思想相同；他们更关注分布式服务场景。
> “undertaken concurrently and independently of the work on speculative decoding” — chen2023 §2。

**C69**（为什么并行打分几乎不贵）大模型 Megatron 式分片下，采样时间主要来自三块：小 batch 下访存受限的线性层、访存受限的注意力（KV cache 大小不随 $K$ 变）、延迟受限的 all-reduce；对小 $K$，并行给 $K$ 个 token 打分与采一个 token 的时间相近。
> “computing the logits of a short continuation of K tokens in parallel has a very similar latency” — chen2023 §4.1（截短）。

**C70**（Chen 的结果）Chinchilla 70B、16 个 TPU v4、batch 1、$K=4$：自回归 14.1 ms/token；XSum nucleus 7.52 ms（1.92×）、greedy 7.00 ms（2.01×）、HumanEval nucleus 5.73 ms（2.46×）；草稿模型 4B、8 层、$d_{model}$ 6144、48 头，1.8 ms/token。HumanEval 与 greedy XSum 的速度超过了自回归采样的理论带宽上限。
> “this speedup exceeded the theoretical memory bandwidth limit of the hardware” — chen2023 §6.1。

**C71**（DeepSeek 的投机解码：MTP 模块做草稿）V3 的 MTP 模块预测第二个后续 token 的接受率在 85%–90%，解码速度达 1.8 倍 TPS。ISCA 论文写作 80%–90%、1.8x。
> “ranges between 85% and 90% across various generation topics” — deepseek2024v3 §5.4.3。
> “an acceptance rate of 80% to 90% for predicting the second subsequent token” — zhao2025 §2.3.3。

### J. 硬件

**C72**（A100）A100 数据表：BF16 Tensor Core 312 TFLOPS（稠密）/ 624 TFLOPS*（*= With sparsity），SXM 与 PCIe 相同；80GB SXM 显存带宽 2,039GB/s，80GB PCIe 1,935GB/s，40GB 两者 1,555GB/s；NVLink 600GB/s。
> 数据表原文：“BFLOAT16 Tensor Core 312 TFLOPS | 624 TFLOPS*”，脚注 “* With sparsity” — nvidia-a100-ds。

**C73**（H100，NVIDIA 只印稀疏值）H100 产品页 SXM 栏：BFLOAT16 Tensor Core 1,979 teraFLOPS*，TF32 Tensor Core 989 teraFLOPS*（星号均为稀疏），显存 80GB，带宽 3.35TB/s，NVLink 900GB/s；NVL 栏 94GB、3.9TB/s、BF16 1,671*。稠密 BF16 值 NVIDIA 页面不印。
> 可引用的稠密值来自 FlashAttention-3：“the H100 SXM5 GPU has 989 TFLOPS of FP16 matmul” — shah2024 §3.1；时钟按 1830 MHz 计（附录 C.1：“clock speed used to calculate the 989 TFLOPS FP16 theoretical max throughput”）。FA3 Table 1（§2.2）写 GMEM “80 GiB @ 3.35 TB/s”。

**C74**（H800 的 NVLink 被砍）H800 SXM 基于 Hopper，与 H100 类似，但出于合规降低了 FP64 性能与 NVLink 带宽：节点内 NVLink 从 900 GB/s 降到 400 GB/s；为补偿，每节点配 8 张 400G IB CX7 网卡。
> “the NVLink bandwidth in H800 SXM nodes is reduced from 900 GB/s to 400 GB/s” — zhao2025 §4.1。

**C75**（H800 的有效带宽）节点内外带宽比约 4:1：NVLink 提供 200GB/s（实际可达约 160GB/s），每张 400Gbps IB 网卡 50GB/s（算有效带宽时取 40GB/s）。V3 报告的说法是 NVLink 160 GB/s，约为 IB（50 GB/s）的 3.2 倍。
> “NVLink provides 200GB/s bandwidth (of which about 160GB/s can actually be achieved)” — zhao2025 §4.3。
> “NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)” — deepseek2024v3 §3.2.2。
> **注意**：“400 GB/s”出自 ISCA 论文，**不在** V3 技术报告里（见 §7 E12）。

**C76**（TPU v4，Pope 的实验平台）每芯片 bfloat16 矩阵运算 275 TFLOPS、32 GiB HBM、带宽 1200 GB/s、互联 270 GB/s（3D torus）。
> “has 32 GiB of High Bandwidth Memory (HBM) at 1200 GB/s of bandwidth” — pope2023 §4 “Methodology”。

### K. 规模化与 FLOPs 计数

**C77**（前向 FLOPs）Kaplan：前向传播约 $C_{forward}\approx2N+2n_{layer}n_{ctx}d_{model}$ 次加乘（式 2.2），系数 2 来自乘加；Table 1 给出逐项分解，注意力“Mask”项为 $2n_{layer}n_{ctx}d_{attn}$。当 $d_{model}>n_{ctx}/12$ 时与上下文相关的项占比较小。
> “the factor of two comes from the multiply-accumulate operation” — kaplan2020 §2.1。

**C78**（Pope 的延迟数据）PaLM 540B、64 芯片：int8 权重、batch 64 时 28.5 ms/token，bf16 为 36.9 ms/token；低延迟配置 prefill 2048 token 0.29 s（MFU 43%），decode 64 token 1.82 s（MFU 14%）；高吞吐配置 batch 512 prefill 85.2 s（MFU 76%）、decode 6.0 s（MFU 33%）（Table 2）。
> “we achieve 28.5ms/token with int8 weights at batch size 64 on PaLM 540B” — pope2023 §4.4。

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 课程记号总表（本课新增或需要特别说明的）

| 课程记号 | 含义 | 各论文里的写法 |
|---|---|---|
| $T$ / $L_{ctx}$ | 序列长度 / 上下文长度 | Shazeer $n$（也用 $m$ 表示被注意的记忆位置数）；Kaplan $n_{ctx}$；Pope $L_{input}$、$L_{gen}$；FlashAttention $N$ |
| $l$ | 层数 | Kaplan/Pope $n_{layer}$/$n_{layers}$；V2 $l$ |
| $d$ | 模型维度 | Shazeer $d$；Pope $d_{model}$ 或 $E$；Kaplan $d_{model}$ |
| $n_h$ | 查询头数 | Shazeer $h$；GQA $H$；Pope $n_{heads}$ 或 $H$ |
| $n_g$ | KV 头数（GQA 组数） | GQA $G$；HF config `num_key_value_heads`；DeepSeek LLM 论文 $n_{kv\_heads}$ |
| $d_h$ | 每头维度 | Shazeer $k$（键）、$v$（值），假设 $k=v=d/h$；Pope $d_{head}$；FlashAttention $d$（注意！FA 的 $d$ 是头维度） |
| $n_hd_h$ | 注意力总宽度 | Kaplan $d_{attn}$ |
| $B$ | batch（按序列计） | Shazeer $b$；Pope $B$ |
| $s$ | 每个元素的字节数（bf16 为 2） | 论文里通常不写；V2 明说按元素数计、与精度无关 |
| $F$, $W$ | 峰值算力（FLOP/s）、显存带宽（byte/s） | Williams “Peak Floating Point Performance”、“Peak Memory Bandwidth” |
| $I$ | 算术强度（FLOP/byte） | Williams operational intensity；FlashAttention arithmetic intensity |
| $\tau$ | 采样温度 | **Holtzman 写 $t$**，与课程的位置下标 $t$ 冲突，建议课文改用 $\tau$ |
| $p_\star$ | top-$p$ 阈值 | **Holtzman 写 $p$**，与“分布 $p$”冲突，建议写 $p_\star$ 或文字“阈值 $p$” |
| $p$, $q$ | 目标模型分布、草稿模型分布 | Leviathan：$p$=目标、$q$=草稿（本课采用）；**Chen 相反：$q$=目标、$p$=草稿** |
| $\gamma$ | 草稿长度 | Leviathan $\gamma$；Chen $K$（课程里 $K_r$ 已用于 MoE，故用 $\gamma$） |
| $\alpha$, $\beta$ | 平均接受率、某前缀下的接受率 | Leviathan 同；第 11 课的均衡损失系数也叫 $\alpha$，只在局部使用即可 |
| $c$, $\hat c$ | 草稿/目标单步耗时比、每 token 运算量比 | Leviathan Def. 3.7、3.10 |

注意：课程里 $\mathbf q$（粗体）是查询向量，$q$（细体）在第 8 课是 GRPO 的问题 prompt，本课的 $q(x)$ 是草稿分布。课文首次出现时要交代三者不同。

### 2.2 多头注意力与 KV cache 公式（DeepSeek-V2 §2.1.1，式 (1)–(8)）

原文（$\mathbf h_t\in\R^d$ 为第 $t$ 个 token 进入注意力层的输入）：

$$\mathbf q_t=W^Q\mathbf h_t,\qquad(1)\qquad \mathbf k_t=W^K\mathbf h_t,\qquad(2)\qquad \mathbf v_t=W^V\mathbf h_t,\qquad(3)$$
$$[\mathbf q_{t,1};\mathbf q_{t,2};\dots;\mathbf q_{t,n_h}]=\mathbf q_t,\ (4)\quad [\mathbf k_{t,1};\dots;\mathbf k_{t,n_h}]=\mathbf k_t,\ (5)\quad [\mathbf v_{t,1};\dots;\mathbf v_{t,n_h}]=\mathbf v_t,\ (6)$$
$$\mathbf o_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_j\!\Big(\frac{\mathbf q_{t,i}^T\mathbf k_{j,i}}{\sqrt{d_h}}\Big)\mathbf v_{j,i},\qquad(7)$$
$$\mathbf u_t=W^O[\mathbf o_{t,1};\mathbf o_{t,2};\dots;\mathbf o_{t,n_h}],\qquad(8)$$

其中 $W^O\in\R^{d\times d_hn_h}$；“MHA needs to cache $2n_hd_hl$ elements for each token”。与课程记号完全一致（style-guide 本来就沿用 DeepSeek 记号），转置写 `^\top`。

**式 (7) 是 KV cache 的全部理由**（本笔记推导，写课文时逐行展开）：
1. 第 $t$ 步的输出只用到 $\mathbf q_{t,i}$（当前位置的查询）和 $\{\mathbf k_{j,i},\mathbf v_{j,i}\}_{j\le t}$。过去的查询 $\mathbf q_{j<t}$ 在第 $t$ 步不出现 → 不需要缓存 Q。
2. $\mathbf k_{j}=W^K\mathbf h_j$，而 $\mathbf h_j$ 由因果掩码保证只依赖 $x_{\le j}$；新 token 到来不改变 $x_{\le j}$ → $\mathbf k_j,\mathbf v_j$ 在 $j<t$ 后永远不变 → 缓存是**精确**的（在浮点意义上，重算与缓存可能有舍入级差异，见 chen2023 “within hardware numerics”）。
3. 这个论证在每一层都成立（第 $l$ 层的 $\mathbf h_j^{(l)}$ 同样只依赖 $x_{\le j}$），所以要缓存每层的 K、V：每 token $2\cdot n_h d_h\cdot l$ 个元素。
4. 条件：因果（单向）注意力、位置编码与绝对位置绑定且不随后续 token 改变。若是双向注意力（编码器）或位置被重新编号，缓存会失效（笔者推理，课文可作“失败模式”）。

### 2.3 KV cache 的内存公式（课程写法）

V2 Table 1 原文（元素数）：MHA $2n_hd_hl$；GQA $2n_gd_hl$；MQA $2d_hl$；MLA $(d_c+d_h^R)l\approx\frac92d_hl$（V2 取 $d_c=4d_h$、$d_h^R=\frac{d_h}{2}$）。

课程把字节数写成（本笔记推导）：
$$\text{bytes/token}=2\,n_g\,d_h\,l\cdot s,\qquad M_{KV}=2\,n_g\,d_h\,l\cdot s\cdot B\cdot L_{ctx}.$$
MHA 取 $n_g=n_h$，MQA 取 $n_g=1$。GQA 相对 MHA 的缩小倍数是 $n_h/n_g$。

### 2.4 Shazeer 的访存/运算比（shazeer2019）

假设（§2.3.1）：$m=n$；$k=v=\frac dh$；$n\le d$。

- §2.3.1 批量 MHA：运算 $\Theta(bnd^2)$；访存 $O(bnd+bhn^2+d^2)$；比值
  $$O\!\left(\frac1k+\frac1{bn}\right).$$
- §2.4.1 增量 MHA（$n$ 次调用合计）：运算 $\Theta(bnd^2)$；访存 $\Theta(bn^2d+nd^2)$；比值
  $$\Theta\!\left(\frac nd+\frac1b\right).$$
- §3.1 增量 MQA：运算 $\Theta(bnd^2)$；访存 $\Theta(bnd+bn^2k+nd^2)$；比值
  $$\Theta\!\left(\frac1d+\frac n{dh}+\frac1b\right).$$

课程记号：$b\to B$，$n\to T$，$h\to n_h$，$k=v\to d_h$，$d\to d$。于是增量 MHA 的比值写作 $\Theta(\frac Td+\frac1B)$，MQA 写作 $\Theta(\frac1d+\frac{T}{d\,n_h}+\frac1B)$。
**注意三点**：(a) 这是“访存 / 运算”，即算术强度的**倒数**；Shazeer 全文没有 “arithmetic intensity” 一词（grep 计数 0）。(b) 批量式里的 $k$ 是键维度 $=d/h$，不是 KV 头数。(c) 这些是渐近阶，不含常数；拿来比“哪一项主导”，不要代入算具体秒数。

### 2.5 FLOPs：Kaplan 表 1 与式 (2.2)

原文 Table 1（每 token 前向 FLOPs）：Embed $4d_{model}$；Attention: QKV $2n_{layer}d_{model}3d_{attn}$；Attention: Mask $2n_{layer}n_{ctx}d_{attn}$；Attention: Project $2n_{layer}d_{attn}d_{embd}$；Feedforward $2n_{layer}2d_{model}d_{ff}$；De-embed $2d_{model}n_{vocab}$；合计 $C_{forward}=2N+2n_{layer}n_{ctx}d_{attn}$，其中 $N=2d_{model}n_{layer}(2d_{attn}+d_{ff})$（式 2.1，标准配置下 $=12n_{layer}d_{model}^2$）。式 (2.2)：$C_{forward}\approx2N+2n_{layer}n_{ctx}d_{model}$。

**本课的逐步推导（课程记号，本笔记推导）**。第 $t$ 个位置（能看到 $t$ 个 token，含自己）的一次前向：
- 所有矩阵乘（QKVO 投影、FFN、反嵌入）：$2N_{mm}$，$N_{mm}$ 为参与矩阵乘的参数量（含 lm_head，不含输入嵌入查表）。
- $\mathbf q^\top\mathbf k$：每层每头 $t$ 个长 $d_h$ 的点积 → $2n_hd_ht$；$\sum_j a_j\mathbf v_j$：同样 $2n_hd_ht$。
$$C_{step}(t)\approx 2N_{mm}+4\,l\,n_hd_h\,t.$$
与 Kaplan 的关系：因果掩码下对 $t=1..n_{ctx}$ 取平均，$\frac1{n_{ctx}}\sum_t4ln_hd_ht\approx2ln_{ctx}n_hd_h$，恰好等于 Kaplan 的“Mask”项。Kaplan 没有说明这个项是怎么来的，“它等于因果平均”是笔者的解读（§8）。

**有缓存 vs 无缓存（本笔记推导）**。提示长 $P$，再生成 $G$ 个 token：
$$C_{\text{cache}}=\underbrace{2N_{mm}P+2ln_hd_hP(P+1)}_{\text{prefill}}+\underbrace{\sum_{t=P+1}^{P+G}\big(2N_{mm}+4ln_hd_ht\big)}_{\text{decode}}=\dots+2N_{mm}G+2ln_hd_h\,G(2P+G+1),$$
$$C_{\text{no cache}}=\sum_{t=P}^{P+G-1}\Big(2N_{mm}t+2ln_hd_h\,t(t+1)\Big).$$
从零生成 $T$ 个 token（$P\approx0$，$G=T$）的主项：
$$C_{\text{cache}}\approx2N_{mm}T+2ln_hd_hT^2,\qquad C_{\text{no cache}}\approx N_{mm}T^2+\tfrac23ln_hd_hT^3,$$
权重项主导时比值 $\approx T/2$。**一个值得讲的结论**：有缓存时逐 token 生成 $T$ 个 token 的总 FLOPs，与对这 $T$ 个 token 做一次并行（teacher-forcing / prefill）前向完全相同——KV cache 把重复计算全部消除了，剩下的只是“同样的计算被迫串行地做”。数字见 §4.4。

### 2.6 roofline 与 decode 的算术强度

Williams §3 原文：
$$\text{Attainable GFlops/sec}=\text{Min}(\text{Peak Floating Point Performance},\ \text{Peak Memory Bandwidth}\times\text{Operational Intensity})$$
课程写法：$P_{att}(I)=\min(F,\ W\cdot I)$；脊点 $I^{\star}=F/W$。

**decode 一步的访存与运算（本笔记推导，bf16，$s=2$）**：batch $B$、每条序列上下文 $t$：
$$\text{FLOPs}\approx B\,(2N_{mm}+4ln_hd_ht),\qquad \text{bytes}\approx s\,N_{mm}+B\cdot 2n_gd_hl\cdot s\cdot t\ \ \xrightarrow{\ s=2\ }\ 2N_{mm}+4B\,ln_gd_h\,t.$$
（权重每步只读一遍、被 batch 内所有序列共用；KV 每条序列各读自己的。输入嵌入只查一行，忽略；激活读写忽略。）
分开看两部分的强度：
- **权重部分**：$\frac{2N_{mm}B}{2N_{mm}}=B$ FLOP/byte。batch 就是强度；要到 A100 脊点需 $B\gtrsim153$。
- **注意力部分**：$\frac{4ln_hd_htB}{4ln_gd_htB}=\frac{n_h}{n_g}$ FLOP/byte，**与 $B$、$t$ 都无关**。MHA 恒为 1，GQA-8（64 头）为 8，MQA 为 $n_h$。加大 batch 救不了注意力，只有减少 KV 头（或压缩 KV，第 9 课的 MLA）才能提高它。这是 Shazeer $\frac nd$ 项“难以消除”的 roofline 版本。

**延迟下界（本笔记推导，把 Pope 的 memory time / compute time 与 roofline 合起来）**：
$$t_{step}\ \ge\ \max\!\Big(\frac{\text{bytes}}{W},\ \frac{\text{FLOPs}}{F}\Big).$$
这是理想下界（忽略通信、kernel 启动、激活读写、利用率 < 100%）。

FlashAttention Theorem 2（dao2022 §3.2，原文）：设 $N$ 为序列长度，$d$ 为头维度，$M$ 为 SRAM 大小，$d\le M\le Nd$。标准注意力（Algorithm 0）需要 $\Theta(Nd+N^2)$ 次 HBM 访问，FlashAttention（Algorithm 1）需要 $\Theta(N^2d^2M^{-1})$ 次。Proposition 3：不存在对 $[d,Nd]$ 内**所有** $M$ 都达到 $o(N^2d^2M^{-1})$ 的精确注意力算法。——本课只用它说明“FlashAttention 省的是 $N\times N$ 中间矩阵的 HBM 往返，主要惠及训练与 prefill；它不缩小 KV cache”。

### 2.7 采样公式（holtzman2020 §3）

式 (1)：$P(x_{1:m+n})=\prod_{i=1}^{m+n}P(x_i\mid x_1\dots x_{i-1})$。

式 (2)（top-$p$ 词表）：$V^{(p)}\subset V$ 是满足
$$\sum_{x\in V^{(p)}}P(x\mid x_{1:i-1})\ \ge\ p$$
的最小集合。

式 (3)：令 $p'=\sum_{x\in V^{(p)}}P(x\mid x_{1:i-1})$，
$$P'(x\mid x_{1:i-1})=\begin{cases}P(x\mid x_{1:i-1})/p' & x\in V^{(p)}\\ 0 & \text{otherwise.}\end{cases}$$
top-$k$（§3.2，无编号）：$V^{(k)}$ 为使 $\sum_{x\in V^{(k)}}P$ 最大的大小为 $k$ 的集合，再按式 (3) 重新归一化。

式 (4)（温度）：原文印作 $p(x=V_l\mid x_{1:i-1})=\frac{\exp(u_l/t)}{\sum_{l'}\exp(u_l'/t)}$——分母的 $u_l'$ 是排版错误，应为 $u_{l'}$。
课程写法：$p_\tau(x)=\dfrac{\exp(u_x/\tau)}{\sum_{x'}\exp(u_{x'}/\tau)}$；$\tau\to0$ 退化为 argmax，$\tau=1$ 为原分布，$\tau>1$ 更平。

### 2.8 投机解码（leviathan2023）

**接受规则**（§2.3）：$x\sim q$；若 $q(x)\le p(x)$ 保留；否则以概率 $1-\frac{p(x)}{q(x)}$ 拒绝并从
$$p'(x)=norm(max(0,p(x)-q(x)))$$
重采样。等价写法：以概率 $\min\!\big(1,\frac{p(x)}{q(x)}\big)$ 接受。

**Algorithm 1（SpeculativeDecodingStep）**关键行（原文）：
$$n\leftarrow\min\big(\{\,i-1\mid 1\le i\le\gamma,\ r_i>\tfrac{p_i(x)}{q_i(x)}\,\}\cup\{\gamma\}\big)$$
若 $n<\gamma$：$p'(x)\leftarrow norm(max(0,p_{n+1}(x)-q_{n+1}(x)))$；否则 $p'=p_{\gamma+1}$；返回 $prefix+[x_1,\dots,x_n,t]$，$t\sim p'$。（$\frac{p_i(x)}{q_i(x)}$ 在 $x=x_i$ 处取值。）

**式 (1)**：$E(\#\,generated\,tokens)=\dfrac{1-\alpha^{\gamma+1}}{1-\alpha}$。
推导（写课文用，本笔记补全）：在 i.i.d. 假设下，一轮至少产出 $k+1$ 个 token 当且仅当前 $k$ 个草稿都被接受，概率 $\alpha^k$（$k=0..\gamma$）；$E[N]=\sum_{k=0}^{\gamma}P(N\ge k+1)=\sum_{k=0}^{\gamma}\alpha^k=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$。$\gamma\to\infty$ 时上界 $\frac1{1-\alpha}$（§3.5 “oracle” 讨论同此式）。

**Definition 3.2 / Lemma 3.3 / Theorem 3.5 / Corollary 3.6**：
$$D_{LK}(p,q)=\sum_x|p(x)-M(x)|=\sum_x|q(x)-M(x)|,\quad M(x)=\tfrac{p(x)+q(x)}2;$$
$$D_{LK}(p,q)=1-\sum_x\min(p(x),q(x));\qquad \beta=E_{x\sim q}\min\!\Big(1,\frac{p(x)}{q(x)}\Big)=\sum_x\min(p(x),q(x));\qquad \alpha=1-E(D_{LK}(p,q))=E(\min(p,q)).$$
（$D_{LK}=\frac12\sum_x|p-q|$，即全变差距离 TV。）

**正确性证明（附录 A.1，原文结构）**：
$$P(guess\ accepted,x=x')=q(x')\min\!\Big(1,\tfrac{p(x')}{q(x')}\Big)=\min(q(x'),p(x')),$$
$$P(guess\ rejected,x=x')=(1-\beta)\,p'(x')=p(x')-\min(q(x'),p(x')),$$
二者相加得 $p(x')$。其中 $p'$ 的归一化常数 $\sum_{x'}(p(x')-\min(q,p))=1-\beta$。

**Theorem 3.8**（墙钟加速）：$\dfrac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$。**Corollary 3.9**：$\alpha>c$ 时存在有益 $\gamma$，$\gamma=1$ 给出 $\frac{1+\alpha}{1+c}$。**Theorem 3.11**（总运算量增加倍数）：$\dfrac{(1-\alpha)(\gamma\hat c+\gamma+1)}{1-\alpha^{\gamma+1}}$。

**Chen 等人的写法（chen2023 §4.2 与 Theorem 1），注意记号相反**：草稿 $\tilde x_{n+1}\sim p$，以概率 $\min\!\big(1,\frac{q(\tilde x_{n+1}\mid x_{1..n})}{p(\tilde x_{n+1}\mid x_{1..n})}\big)$ 接受；拒绝则 $x_{n+1}\sim(q(x\mid\cdot)-p(x\mid\cdot))_+$，其中
$$(f(x))_+=\frac{\max(0,f(x))}{\sum_x\max(0,f(x))}.$$
把 Chen 的 $q\leftrightarrow p$ 对调即得 Leviathan 的式子。课程统一用 Leviathan 记号。

### 2.9 PagedAttention 的注意力写法（kwon2023 §2.1）

式 (1)：$P(x)=P(x_1)\cdot P(x_2\mid x_1)\cdots P(x_n\mid x_1,\dots,x_{n-1})$；式 (2)：$q_i=W_qx_i,\ k_i=W_kx_i,\ v_i=W_vx_i$；式 (3)：
$$a_{ij}=\frac{\exp(q_i^\top k_j/\sqrt d)}{\sum_{t=1}^{i}\exp(q_i^\top k_t/\sqrt d)},\qquad o_i=\sum_{j=1}^{i}a_{ij}v_j.$$
（论文紧接着写 “Besides the computation in Eq. 4”，应为 Eq. 3；其分块形式在 §4.1 编号为 (4)。$\sqrt d$ 里的 $d$ 是单头维度的简化写法。）

---

## 3. 规格与超参数表

### 3.1 模型规格（注意力相关）

| 模型 | $l$ | $d$ | $n_h$ | $n_g$（KV 头） | $d_h$ | $d_{ff}$ | 词表 | 原生上下文 | 来源 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-3 175B | 96 | 12288 | 96 | 96（MHA） | 128 | $4d$=49152 | **假设** 50257（GPT-3 论文未单列，只说沿用 GPT-2 的可逆分词；50257 取自 Kaplan §2 的 GPT-2 BPE 词表，仅用于参数量复算） | 2048 | brown2020 Table 2.1；$d_{ff}=4d$ 与 $n_{ctx}=2048$ 见 §2.1 正文 |
| Llama-2-7B | 32 | 4096 | 32 | 32（MHA） | 128 | 11008 | 32000 | 4096 | 维度：touvron2023a Table 2（6.7B 行）+ touvron2023b Table 1（7B 无 GQA、4k）+ 镜像 config |
| Llama-2-70B | 80 | 8192 | 64 | 8 | 128 | 28672 | 32000 | 4096 | touvron2023b Table 1（70B ✓GQA）；层/宽/头**只来自镜像 config**；A.2.1 “8 KV projections” |
| Llama-3-8B | 32 | 4096 | 32 | 8 | 128 | 14336 | 128,000（论文）/ 128256（config） | 8192（config） | grattafiori2024 Table 3 + 镜像 config |
| Llama-3-70B | 80 | 8192 | 64 | 8 | 128 | 28672 | 同上 | 8192（config） | grattafiori2024 Table 3 + 镜像 config |
| Llama-3(.1)-405B | 126 | 16384 | 128 | 8 | 128 | 53248 | 同上 | — | grattafiori2024 Table 3（ISCA 表 1 用作 LLaMA-3.1 405B） |
| DeepSeek LLM 7B | 30 | 4096 | 32 | 32（MHA） | 128 | 11008 | 102400 | 4096 | deepseek2024llm Table 2 + 官方 config（一致） |
| DeepSeek LLM 67B | 95 | 8192 | 64 | 8 | 128 | 22016 | 102400 | 4096 | deepseek2024llm Table 2 + 官方 config（base 与 chat 一致） |
| DeepSeek-V3（MLA） | 61 | 7168 | 128 | — | 128（$d_h^R$=64，$d_c$=512） | MoE | — | 163840（config） | deepseek2024v3 §4.2；config `kv_lora_rank`=512、`qk_rope_head_dim`=64；**config 的 `num_key_value_heads`=128 不代表缓存头数** |
| Qwen2.5-72B | 80 | 8192 | 64 | 8 | 128 | — | — | 131072（config） | 官方 config（仅用于复算 ISCA 表 1） |
| OPT-13B | 40 | 5120 | 40 | 40（MHA） | 128 | 20480 | — | 2048 | 官方 config + kwon2023 §3 |
| PaLM 540B | 118 | 18432 | 48（推理时 pad 到 64） | 1（MQA） | 256 | 73728 | — | 2048（Pope 实验） | pope2023 Table D.1、§4“Methodology”；Pope 的 MHA 变体 $d_{head}$=128（§4.2） |
| Chinchilla 70B（目标） | 80 | 8192 | 64 | — | — | — | — | — | chen2023 Table 2 |
| Chen 草稿模型 4B | 8 | 6144 | 48 | — | — | — | — | — | chen2023 Table 2 |
| Shazeer WMT 基线 | 6（enc-dec） | 1024 | 8 | 8 / 1 | 128 | 4096（MQA 加宽到 5440） | — | 256 | shazeer2019 §4.1；211M 参数 |
| Shazeer LM 基线 | 6 | 1024 | 8 | 8 / 1 | 128 | 8192（MQA 9088） | — | — | shazeer2019 §4.1、Table 3；192M 参数 |

Paper-vs-config 核对结论：DeepSeek LLM 7B/67B 论文 Table 2 与官方 config 完全一致（67B：95 层、64 头、`num_key_value_heads`=8 ✓）。Llama 3 论文与 config 一致，唯独词表 128,000（论文 Table 3）vs 128256（config，含特殊 token）。

### 3.2 硬件

| 硬件 | 稠密 BF16（TFLOP/s） | 显存 | 显存带宽 | 节点内互联 | 脊点 $F/W$ | 来源 |
|---|---|---|---|---|---|---|
| A100 80GB SXM | 312（稀疏 624*） | 80GB HBM2e | 2,039 GB/s | NVLink 600 GB/s | 153 | nvidia-a100-ds |
| A100 80GB PCIe | 312（同上） | 80GB | 1,935 GB/s | NVLink Bridge 600 GB/s（2 卡） | 161 | nvidia-a100-ds |
| A100 40GB（SXM/PCIe） | 312 | 40GB HBM2 | 1,555 GB/s | — | 201 | nvidia-a100-ds |
| H100 SXM | 989（NVIDIA 只印稀疏 1,979*；稠密值取自 FA3，按 1830 MHz） | 80GB（FA3 写 80 GiB） | 3.35 TB/s | NVLink 900 GB/s | 295 | nvidia-h100-page；shah2024 §3.1、Table 1、附录 C.1 |
| H100 NVL | 稀疏 1,671*（稠密未核） | 94GB | 3.9 TB/s | NVLink 600 GB/s | — | nvidia-h100-page |
| H800 SXM | **未核实**（ISCA 只说 FP64 与 NVLink 被降，暗示其余同 H100） | **未核实** | **未核实** | NVLink 400 GB/s（H100 为 900）；有效 200 GB/s，可达约 160 GB/s | — | zhao2025 §4.1、§4.3；deepseek2024v3 §3.2.2（160 GB/s） |
| TPU v4 | 275 | 32 GiB | 1200 GB/s | 270 GB/s（3D torus） | 229 | pope2023 §4 |

### 3.3 投机解码实验设置（便于出题）

| 论文 | 目标模型 | 草稿模型 | 硬件 | batch | 草稿长度 | $\alpha$ | 加速 |
|---|---|---|---|---|---|---|---|
| leviathan2023 Table 2 | T5-XXL 11B | T5-small 77M / base 250M / large 800M | 单 TPU-v4 | 1 | $\gamma$=3–7 | 0.53–0.82 | 1.4X–3.4X |
| leviathan2023 Table 4 | 同上 | 同上 | 同上 | 1 | — | — | $c$=0.02（small）、0.04（base）、0.11（large） |
| chen2023 Table 1 | Chinchilla 70B | 4B（8 层） | 16 TPU v4 | 1 | $K$=4 | 未直接报告 | 1.92×–2.46× |
| deepseek2024v3 §5.4.3 | DeepSeek-V3 | 自带 MTP 模块（1 层） | — | — | 1 | 0.85–0.90 | 1.8× TPS |

---

## 4. 复算的数字

所有数都用 Python 计算（脚本在 scratch 目录，未入库）。“GiB”=$2^{30}$ 字节，“GB”=$10^9$ 字节，两者都给。

### 4.1 KV 字节/token（bf16，$s=2$）与 4K/32K/128K 下单条序列的缓存

表达式：`bytes = 2 * n_g * d_h * l * 2`，`GiB = bytes * L / 2**30`。

| 模型 | 表达式 | bytes/token | KiB/token | 4K | 32K | 128K | MHA 等价 bytes/token（倍数） |
|---|---|---|---|---|---|---|---|
| GPT-3 175B | 2·96·128·96·2 | 4,718,592 | 4608（4.5 MiB） | 18.0 GiB（19.33 GB） | 144 GiB（154.6 GB） | 576 GiB（618.5 GB） | — |
| Llama-2-7B | 2·32·128·32·2 | 524,288 | 512 | 2.0 GiB（2.15 GB） | 16 GiB（17.18 GB） | 64 GiB（68.72 GB） | — |
| Llama-2-70B | 2·8·128·80·2 | 327,680 | 320 | 1.25 GiB（1.34 GB） | 10 GiB（10.74 GB） | 40 GiB（42.95 GB） | 2,621,440（8×） |
| Llama-3-8B | 2·8·128·32·2 | 131,072 | 128 | 0.5 GiB（0.54 GB） | 4 GiB（4.29 GB） | 16 GiB（17.18 GB） | 524,288（4×） |
| Llama-3-70B | 2·8·128·80·2 | 327,680 | 320 | 1.25 GiB（1.34 GB） | 10 GiB（10.74 GB） | 40 GiB（42.95 GB） | 2,621,440（8×） |
| DeepSeek LLM 7B | 2·32·128·30·2 | 491,520 | 480 | 1.875 GiB（2.01 GB） | 15 GiB（16.11 GB） | 60 GiB（64.42 GB） | — |
| DeepSeek LLM 67B | 2·8·128·95·2 | 389,120 | 380 | 1.484 GiB（1.59 GB） | 11.875 GiB（12.75 GB） | 47.5 GiB（51.0 GB） | 3,112,960（8×） |
| （预告）DeepSeek-V3 MLA | (512+64)·61·2 | 70,272 | 68.6 | 0.268 GiB | 2.14 GiB | 8.58 GiB | 若 MHA 128×128：3,997,696（56.9×） |

注意：128K 超出上述所有非 V3 模型的原生上下文（GPT-3 2048、Llama-2 4096、Llama-3 8192、DeepSeek LLM 4096），这一列是“假如能跑这么长”的账单，课文里必须标明是假设。GPT-3 在原生 2048 上：9.0 GiB（9.66 GB）/序列。

### 4.2 权重字节与“KV 追上权重”的上下文长度

参数量由 config 维度算出（不含 norm，全部为 bf16）：`N = l*(d*n_h*d_h + 2*d*n_g*d_h + n_h*d_h*d + 3*d*d_ff) + V*d*(1 or 2)`（GPT-3 用 $2d\,d_{ff}$ 的 GELU FFN、权重共享）。

| 模型 | 参数量（算） | 权重 bf16 | KV = 权重时的上下文 `W/bytes_per_token` |
|---|---|---|---|
| GPT-3 175B | 174.56B（论文 175.0B） | 349.1 GB | 73,990 |
| Llama-2-7B | 6.738B | 13.48 GB | 25,704 |
| Llama-2-70B | 68.98B | 137.95 GB | 420,992 |
| Llama-3-8B | 8.030B | 16.06 GB | 122,528 |
| Llama-3-70B | 70.55B | 141.1 GB | 430,618 |
| DeepSeek LLM 7B | 6.910B | 13.82 GB | 28,117 |
| DeepSeek LLM 67B | 67.42B | 134.85 GB | 346,543 |

（核查注：最后一列按未四舍五入的参数量重算；原稿用四舍五入后的 GB 数去除，个别值差 2–15，量级结论不变。）

读法：MHA 的 7B 模型一条 2.6 万 token 的序列，KV 就和整个模型一样大；GQA-8 的 70B 要到 42 万 token。

### 4.3 复算论文里的数字

| 数字 | 表达式 | 结果 | 论文值 | 结论 |
|---|---|---|---|---|
| OPT-13B 每 token KV | `2*5120*40*2` | 819,200 B = 800 KiB | “800 KB” | 论文的 KB 实为 KiB |
| OPT-13B 2048 token | `819200*2048` | 1.678 GB = 1.5625 GiB | “1.6 GB” | 一致（四舍五入） |
| OPT-13B 权重占 40GB 的比例 | 12.85B·2/40e9 | ≈ 64% | “Approximately 65%” | 一致 |
| Pope：PaLM 540B MHA 变体 KV（48 头 × 128） | `2*118*48*128*2*512*2048` | 3.04 TB | “3TB” | 一致 |
| 同上，若按 pad 后 64 头 | `2*118*64*128*2*512*2048` | 4.05 TB | — | 说明“3TB”用的是 48 头 |
| 3TB / 参数字节 | 3.04e12 / (540e9·2) | 2.82 | “3 times” | 一致（取整） |
| Pope Table 1：MHA 最大上下文 | 0.3·64·32 GiB / (3,866,624·B)，$B$=128/512 | 1333 / 333 | 1320 / 330 | 差 <1%（假设 pad 到 64 头） |
| Pope Table 1：基线 MQA | 同上，每 token 120,832 B × 64 份复制 | 666 / 167 | 660 / 165 | 差 <1% |
| Pope Table 1：优化 MQA | 同上，每 token 120,832 B 不复制 | 42,654 / 10,663 | 43,000 / 10,700 | 差 <1% |
| Pope 低延迟 decode 每 token | 1.82 s / 64 | 28.4 ms | 28.5 ms/token（§4.4） | 一致 |
| Pope prefill 每 token vs decode 每 token | (0.29/2048) vs (1.82/64) | 0.142 ms vs 28.4 ms，差 201 倍 | — | 用于图 4.1。**口径注意**：28.4 ms 是 batch 64 下“每步”（=每条序列每 token）的延迟，Pope 也把它叫 ms/token；按吞吐算，每步产出 64 个 token，1.82/(64·64) = 0.444 ms/token，只比 prefill（batch 1）慢约 3.1 倍。课文说“差 201 倍”时必须写明是单序列延迟口径 |
| Pope 高吞吐 decode 每步 | 6.0 s / 64 | 93.8 ms/步（batch 512，摊到每 token 0.183 ms） | — | — |
| PaLM 540B int8 权重读一遍的下界（64 芯片） | 540e9 / (64·1200e9) | 7.03 ms（bf16：14.06 ms） | 实测 28.5 / 36.9 ms | 实测约为纯权重下界的 4 倍与 2.6 倍（还有 KV、通信） |
| TPU v4 64 芯片的 30% HBM | 0.3·64·32 | 614.4 GiB | “reserve 30%” | — |
| Chen：Chinchilla 带宽上限 | 70e9·2 / (16·1200e9) | 7.29 ms/token | 自回归实测 14.1 ms；SpS HumanEval 5.73 ms | SpS 低于上限（论文语义一致）；**1200 GB/s 取自 Pope，不是 Chen** |
| Chen 加速比 | 14.1/7.52，14.1/7.00，14.1/5.73 | 1.88，2.01，2.46 | 1.92，2.01，2.46 | 第一个差 0.04（论文按逐循环日志统计，非简单相除） |
| Chen 草稿/目标耗时比 | 1.8/14.1 | 0.128 | — | 对应 Leviathan 的 $c$ |
| Shazeer 解码器加速 | 46/3.8 | 12.1× | Table 2 | 贪心；beam-4：203/32 = 6.3× |
| Shazeer 训练每 token | 433 ms / 32,768 | 13.2 μs | 13.2 | 一致 |
| Shazeer 解码器每 token | 47 ms / 1024 | 45.9 μs | 46 | 一致 |
| GQA 推理时间比 | 1.51/0.24；1.51/0.28；0.28/0.24 | 6.29×；5.39×；GQA 比 MQA 慢 17% | Table 1 | — |
| ISCA 表 1：DeepSeek-V3 | (512+64)·61·2 | 70,272 B | 70.272 KB | 一致（十进制 KB） |
| ISCA 表 1：Qwen-2.5 72B | 2·80·8·128·2 | 327,680 B；倍数 4.663 | 327.680 KB；4.66x | 一致 |
| ISCA 表 1：LLaMA-3.1 405B | 2·126·8·128·2 | 516,096 B；倍数 **7.344** | 516.096 KB；**7.28x** | **论文倍数算错**（v1、v2 均如此） |
| V2：MLA ≈ $\frac92d_hl$ | $4d_h+\frac12d_h$ | $4.5\,d_hl$ = GQA 的 $n_g$=2.25 | Table 1 | 一致 |
| NVLink H100/H800 | 900/400 | 2.25× | — | — |
| V3 报告 NVLink/IB | 160/50 | 3.2 | “roughly 3.2 times” | 一致 |

### 4.4 decode 的 FLOPs：有缓存 vs 无缓存（Llama-2-7B）

参数：$l=32$，$n_hd_h=4096$，$N_{mm}=6.476\text{B（层）}+32000\cdot4096=6.607\text{B}$。

| 场景 | 有缓存 | 无缓存 | 比值 |
|---|---|---|---|
| 单步，$t=4096$ | $2N_{mm}+4\cdot32\cdot4096\cdot4096$ = 1.536e10（15.4 GFLOP） | $2N_{mm}\cdot4096+2\cdot32\cdot4096\cdot4096\cdot4097$ = 5.852e13（58.5 TFLOP） | 3810 |
| 有缓存单步里注意力占比（$t$=4096） | $4ln_hd_ht/C_{step}$ | — | 14.0% |
| 从零生成 $T$=4096 | 5.852e13 | 1.169e17 | 1997（≈ $T/2$=2048） |
| 提示 1 + 生成 1024 | 1.382e13 | 7.029e15 | 509 |
| 提示 1000 + 生成 1000 | prefill 1.348e13 + decode 1.400e13 | 2.045e16 | 744 |
| 提示 3072 + 生成 1024 | 4.307e13 + 1.546e13 = 5.852e13 | 5.202e16 | 889 |

核对：“从零生成 4096 个 token（有缓存）”= 5.852e13 与“一次 prefill 4096 个 token”= 5.852e13 完全相同（§2.5 的结论）。

（数学核查注，2026-09-24：原稿后两行的“无缓存”列是 $\sum_{t=T_p}^{T_p+T_g-1}C_{fwd}(t)$ = 2.043e16 / 5.196e16（朴素循环只跑 $T_g$ 次前向，最长 $T_p+T_g-1$ 个位置），而“有缓存”列是 prefill + $T_g$ 个 decode 步（共处理 $T_p+T_g$ 个位置），两列口径差一步。统一为课文 (4.12) 的口径后，无缓存 = $\sum_{t=T_p}^{T_p+T_g}C_{fwd}(t)$ = 2.0454e16 / 5.2019e16，比值 744.4 / 888.9。若两列都少算最后一步，比值为 743.8 / 888.1，差别 < 0.1%。“提示 1 + 生成 1024”一行同理，统一口径后比值为 510（原稿 509），课文未用。）

### 4.5 roofline 与 decode 强度

| 数 | 表达式 | 值 |
|---|---|---|
| A100 80GB SXM 脊点 | 312e12/2.039e12 | 153.0 FLOP/B |
| H100 SXM 脊点 | 989e12/3.35e12 | 295.2 FLOP/B |
| TPU v4 脊点 | 275e12/1.2e12 | 229.2 FLOP/B |
| H100/A100 算力、带宽增长 | 989/312；3.35/2.039 | 3.17×；1.64×（算力涨得比带宽快，脊点右移） |
| decode 注意力强度（bf16） | $n_h/n_g$ | MHA 1；Llama-3-8B 4；Llama-2/3-70B、DeepSeek 67B 8；MQA-64 64 |
| A100 上可达算力 | min(312, $I$·2.039) | $I$=1：2.0 TFLOP/s（0.65%）；4：8.2（2.6%）；8：16.3（5.2%）；64：130.5（41.8%） |
| 权重部分强度 | $B$ | $B$=1：2.0 TFLOP/s；16：32.6；64：130.5；153 及以上：312 |
| Llama-2-7B prefill 2048 的强度 | FLOPs / (权重 + KV 写) | ≈1971 FLOP/B → compute-bound |

### 4.6 decode 延迟下界随 batch 变化（A100 80GB SXM，上下文 4096，bf16，理想化）

表达式：`bytes = 2*N_mm + B*kv*t`，`flops = B*(2*N_mm + 4*l*n_h*d_h*t)`，`step = max(bytes/2.039e12, flops/312e12)`；最大 batch = `(80e9 - 权重)/(kv*4096)`。

| 模型 | $B$ | 访存时间 | 计算时间 | 每步下界 | 总吞吐上界 | 每序列 tok/s | 强度 | KV 占访存 |
|---|---|---|---|---|---|---|---|---|
| Llama-2-7B（MHA，最大 $B$=30） | 1 | 7.53 ms | 0.049 ms | 7.53 ms | 133 tok/s | 132.7 | 1.0 | 14% |
| | 8 | 14.91 | 0.394 | 14.91 | 537 | 67.1 | 4.0 | 57% |
| | 16 | 23.33 | 0.788 | 23.33 | 686 | 42.9 | 5.2 | 72% |
| | 30 | 38.08 | 1.477 | 38.08 | 788 | 26.3 | 5.9 | 83% |
| Llama-3-8B（GQA-8，最大 $B$=119） | 1 | 7.62 | 0.055 | 7.62 | 131 | 131.2 | 1.1 | 3% |
| | 8 | 9.47 | 0.440 | 9.47 | 845 | 105.6 | 7.1 | 22% |
| | 16 | 11.57 | 0.880 | 11.57 | 1382 | 86.4 | 11.6 | 36% |
| | 32 | 15.79 | 1.760 | 15.79 | 2027 | 63.3 | 17.1 | 53% |
| | 64 | 24.21 | 3.519 | 24.21 | 2643 | 41.3 | 22.2 | 70% |
| | 119 | 38.69 | 6.544 | 38.69 | 3075 | 25.8 | 25.9 | 81% |

读法：(1) batch 1 时两者都是约 7.5 ms 的“读一遍权重”；(2) GQA 让同一张卡能装下约 4 倍的并发序列，总吞吐上界从 788 到 3075 tok/s；(3) 即使塞满，强度也只有 6–26，离脊点 153 很远——decode 从头到尾都在带宽墙下。这些是忽略激活、碎片、通信的理想值。

### 4.7 其他练习用数字

| 数 | 表达式 | 值 |
|---|---|---|
| GPT-3 175B 在 8×A100 80GB 上 batch 1 下界 | 349.1e9/(8·2.039e12) | 21.4 ms/token |
| 同上，剩余显存能放几条 2048 序列 | (640−349.1) GB / 9.66 GB | 30 条 |
| Llama-3-70B 在 8×H100 上 batch 1 下界 | 141.1e9/(8·3.35e12) | 5.26 ms（≤190 tok/s，忽略通信） |
| DeepSeek LLM 67B 在 2×A100 80GB 上 | 余 25.15 GB；每序列 4K 1.59 GB | 最大 $B$=15；若 MHA-64（12.75 GB/序列）只能放 1 条 |
| DeepSeek LLM 67B 2×A100 batch 1 下界 | 134.85e9/(2·2.039e12) | 33.1 ms |
| DeepSeek LLM 67B，batch 8、32K | 8·32768·389,120 vs ·3,112,960 | 102.0 GB（GQA）vs 816.0 GB（假想 MHA） |

### 4.8 投机解码数字

| 数 | 表达式 | 值 | 论文值 |
|---|---|---|---|
| Leviathan Table 1（$c=\hat c=0$） | speed=$\frac{1-\alpha^{\gamma+1}}{1-\alpha}$；ops=$\frac{(1-\alpha)(\gamma+1)}{1-\alpha^{\gamma+1}}$ | (0.6,2) 1.53/1.96；(0.7,3) 1.58/2.53；(0.8,2) 1.23/2.44；(0.8,5) 1.63/3.69；(0.9,2) 1.11/2.71；(0.9,10) 1.60/6.86 | 全部一致 |
| Leviathan Table 4 “Exp”（共 12 行） | Theorem 3.8 代入表中 $\alpha,\gamma,c$ | 3.157, 3.251, 2.497, 2.258, 2.347, 1.934, 2.402, 2.619, 2.025, 1.891, 1.803, 1.541 | 3.2, 3.3, 2.5, 2.3, 2.4, 2.0, 2.4, 2.6, 2.0, 1.9, 1.8, 1.6。**9 行在一位小数上一致；第 5 行（EnDe/base/temp1：2.35 vs 2.4）、第 6 行（EnDe/large/temp1：1.93 vs 2.0）、第 12 行（CNNDM/large/temp1：1.54 vs 1.6）差 0.05–0.07**，与表中 $\alpha$、$c$ 只印两位小数的舍入误差量级相符（例如第 12 行取 $c$=0.10 得 1.58）。不能写“完全复现” |
| bigram 草稿 | $\alpha$=0.2，$\gamma$=3，$c$=0 | 1.248 | 1.25X |
| $\alpha$=0.75、$c$=0.02 的最优 $\gamma$ | 遍历 1..29 | $\gamma$=9，3.20× | Fig. 3 定性 |
| oracle-$\gamma$ 上界 | $1/(1-0.75)$ | 4.0 | §3.5 |
| Corollary 3.9 | $(1+0.75)/(1+0.02)$ | 1.716 | — |
| MTP（$\gamma$=1）每步期望 token | $1+\alpha$ | $\alpha$=0.80 → 1.80；0.85 → 1.85；0.90 → 1.90 | V3 报告 1.8× TPS（一致，且说明 TPS 增益 ≈ $1+\alpha$ 减去开销） |
| $E[\#]$ 网格（行 $\gamma$，列 $\alpha$=0.1,0.2,…,0.9,0.95） | 式 (1) | γ=1：1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 1.95<br>γ=2：1.11 1.24 1.39 1.56 1.75 1.96 2.19 2.44 2.71 2.85<br>γ=3：1.11 1.25 1.42 1.62 1.88 2.18 2.53 2.95 3.44 3.71<br>γ=5：1.11 1.25 1.43 1.66 1.97 2.38 2.94 3.69 4.69 5.30<br>γ=7：1.11 1.25 1.43 1.67 1.99 2.46 3.14 4.16 5.70 6.73<br>γ=10：1.11 1.25 1.43 1.67 2.00 2.49 3.27 4.57 6.86 8.62 | Fig. 2 的数据 |
| 玩具分布 $p$=[0.5,0.3,0.2]，$q$=[0.2,0.5,0.3] | $\alpha=\sum\min$；残差 | $\alpha$=0.70，TV=0.30；逐 token 接受概率 [1, 0.6, 0.667]；残差 [0.3,0,0] → 归一化 [1,0,0] | 蒙特卡洛 20 万次：频率 [0.502, 0.300, 0.198]，接受率 0.698 |
| $\gamma$=4、$\alpha$=0.7 的产出分布 | $P(N=k+1)=\alpha^k(1-\alpha)$，$P(N=5)=\alpha^4$ | [0.300, 0.210, 0.147, 0.103, 0.240]，期望 2.773 | — |

### 4.9 采样玩具例子（logits $u$=[2.0, 1.0, 0.5, 0.0, −1.0]）

| 操作 | 结果 |
|---|---|
| $\tau$=0.5 | [0.8292, 0.1122, 0.0413, 0.0152, 0.0021]，熵 0.878 bit |
| $\tau$=1.0 | [0.5630, 0.2071, 0.1256, 0.0762, 0.0280]，熵 1.741 bit |
| $\tau$=2.0 | [0.3745, 0.2272, 0.1769, 0.1378, 0.0836]，熵 2.152 bit |
| top-$k$=2（$\tau$=1） | 集合 {0,1}，$p'$=0.7701，重归一化 [0.7311, 0.2689] |
| top-$p$=0.9（$\tau$=1） | 集合 {0,1,2,3}，$p'$=0.9720，[0.5793, 0.2131, 0.1293, 0.0784] |
| top-$p$=0.8 | 集合 {0,1,2}，$p'$=0.8958 |

---

## 5. 图表清单与重绘建议

### 5.1 论文图表清单（值得重绘或引用的）

| 来源 | 图/表 | 图注要点 | 本课用途 |
|---|---|---|---|
| shazeer2019 | **无图**；Table 1（WMT14 BLEU/ln PPL）、Table 2（摊销 μs/token）、Table 3（LM dev-PPL） | MQA 质量略降、解码快 12× | 图 4.9 的数据 |
| ainslie2023 | Figure 1 | 各头 K、V 投影均值池化成一个 | 图 4.7 右半 |
| ainslie2023 | Figure 2 | MHA / GQA / MQA 三联：H 个 Q 头，GQA 每组共享一对 K、V | 图 4.7 主图（重绘） |
| ainslie2023 | Figure 3 | 平均性能 vs 每样本推理时间（MHA-Large、MHA-XXL、MQA-XXL、GQA-8-XXL） | 可用 Table 1 数据重绘为散点 |
| ainslie2023 | Figure 4 | 转换方法消融：Mean > First > Random（T5-Large→MQA，α=0.05） | 文字引用即可 |
| ainslie2023 | Figure 5 | 性能 vs uptraining 比例（0 到 10%）；GQA 转换后即可用，MQA 需要续训 | 文字引用 |
| ainslie2023 | Figure 6 | GQA-XXL 每样本时间 vs 组数（输入 2048、输出 512） | 文字引用（无数据表） |
| pope2023 | Figure 1 | PaLM 各尺寸 cost vs latency 帕累托前沿；左 decode 64 token，右 prefill 2048 | 图 4.1 的背景 |
| pope2023 | Figure 4/5 | MQA 按 batch 切分 vs 按头切分的布局 | 课文点到为止 |
| pope2023 | Figure 8 | 8 层 PaLM、64 芯片、batch 256：每 token 延迟 vs 序列长度（MHA / 基线 MQA / 优化 MQA） | 可定性引用 |
| pope2023 | Table 1、Table 2、Table 3 | 最大上下文；低延迟/高吞吐配置 | 图 4.8、图 4.1 |
| dao2022 | Figure 1 左 | GPU 内存层级金字塔（SRAM / HBM / CPU DRAM） | 图 4.5 旁的小插图；**图中数值本次未读取**，只能用正文 §2.1 已核对的 A100 数字（HBM 40–80GB、1.5–2.0TB/s；每 SM 192KB SRAM×108、约 19TB/s） |
| kwon2023 | Figure 1 | 13B/A100-40GB 显存布局（权重 65%、KV ~30%、激活少量）；右：吞吐与 KV 显存随 batch 增长 | 图 4.14 |
| kwon2023 | Figure 2 | 各系统 KV 显存利用率（正文给区间 20.4%–38.2%，逐系统数值**未能读取**） | 只用正文区间 |
| kwon2023 | Figure 3 | 三种浪费：预留、内部碎片、外部碎片 | 图 4.14 |
| kwon2023 | Figure 5/6/7 | PagedAttention 分块计算；block table；两个请求共享物理块 | 图 4.14 简化版 |
| holtzman2020 | Figure 1 | beam search 退化重复 vs 纯采样胡言乱语 | 文字引用 |
| holtzman2020 | Figure 5 | 平分布 vs 尖分布：固定 $k$ 在两种情况下都出错 | 图 4.10 的思想 |
| holtzman2020 | Table 1 | 各解码方法的困惑度、Self-BLEU、Zipf、重复率、HUSE | 可做小表 |
| leviathan2023 | Figure 1 | 38 个 token 只用 9 次目标模型串行调用（97M 目标 / 6M 草稿，lm1b）；绿=被接受，红=被拒，蓝=修正 | 图 4.11 旁的例子 |
| leviathan2023 | Figure 2 | $E[\#tokens]$ vs $\alpha$，多条 $\gamma$ 曲线 | 图 4.12（按式 (1) 计算） |
| leviathan2023 | Figure 3 | 最优 $\gamma$ vs $\alpha$，多条 $c$ | 可选 |
| leviathan2023 | Figure 4 | 加速 vs 运算量增加 | 图 4.13 |
| leviathan2023 | Figure 5 | 执行轨迹：$\gamma$=7、3 与普通解码 | 图 4.11 下半 |
| chen2023 | Figure 1 | 左：生成 128 token 的时间 vs $K$（XSum 在 $K$=3 最优）；中：接受数/(K+1)；右：每循环时间 ≈ 线性于 $K$ | 文字引用 |
| deepseek2024v2 | Figure 3 | “Simplified illustration of” MHA、GQA、MQA 与 MLA 四联（图注已核） | **留给第 9 课**；本课只画前三联 |
| deepseek2024v2 | Table 1 | 四种注意力的每 token 缓存元素数 | 本课结尾的“下一课预告”表 |
| zhao2025 | Table 1 | V3 / Qwen-2.5 72B / LLaMA-3.1 405B 每 token KV | 图 4.3 的预告条（倍数用复算值） |
| zhao2025 | Figure 2 | H800 节点互联 | 可选，第 12 课更合适 |

### 5.2 本课建议的图（14 张，全部可用上面的数字直接算几何）

**图 4.1 prefill 与 decode 的时间线**。横轴时间（对数或折断轴）。上行：PaLM 540B 低延迟配置（pope2023 Table 2）：prefill 2048 token 用 0.29 s（一整块，蓝色），随后 64 个 decode 小块各 28.4 ms（共 1.82 s）。标注“每 token：prefill 0.142 ms vs decode 28.4 ms，差 201 倍”，并在图注写明 decode 是 batch 64 下每条序列的每步延迟（按吞吐算只差约 3.1 倍，见 §4.3），prefill 是 batch 1。下行：高吞吐配置 batch 512：prefill 85.2 s，decode 64 步各 93.8 ms（6.0 s）。MFU 标签：43%/14%/76%/33%。颜色：计算块 `--fig-blue`，读 KV `--fig-amber`。

**图 4.2 KV cache 随上下文增长**。x：$L_{ctx}$ 从 1K 到 128K（对数），y：单条序列 KV（GiB，对数）。七条直线，斜率 1，截距 = bytes/token（§4.1 表）。水平虚线：A100/H100 80 GB（=74.5 GiB）。在每条线与各自原生上下文处打点（GPT-3 2048、Llama-2 4096、Llama-3 8192、DeepSeek LLM 4096），超出原生长度的部分画虚线并注明“假想”。

**图 4.3 每 token KV 字节柱状图**。按 §4.1 表的 bytes/token 排序：GPT-3 4,718,592；Llama-2-7B 524,288；DeepSeek 7B 491,520；DeepSeek 67B 389,120；Llama-2-70B = Llama-3-70B 327,680；Llama-3-8B 131,072；（预告，灰色）DeepSeek-V3 70,272。GQA 模型旁用浅色叠加其 MHA 等价值（2,621,440、524,288、3,112,960）。可用对数纵轴。

**图 4.4 缓存 vs 重算的 FLOPs**。Llama-2-7B。x：当前位置 $t$（1 到 4096），y：单步 FLOPs（对数）。有缓存：$2N_{mm}+4\cdot32\cdot4096\cdot t$，$N_{mm}$=6.607e9（在 $t$=4096 为 1.536e10）；无缓存：$2N_{mm}t+2\cdot32\cdot4096\cdot t(t+1)$（$t$=4096 为 5.852e13）。标注比值 3810。可加第二面板：累计 FLOPs，从零生成 $T$ 个 token（$T$=4096 时 5.852e13 vs 1.169e17）。

**图 4.5 roofline（A100 与 H100）**。双对数。A100：屋顶 312 TFLOP/s，斜线 $2.039\cdot I$，脊点 153。H100：989、$3.35\cdot I$，脊点 295。打点（A100 线上）：decode 注意力 MHA $I$=1（2.0 TFLOP/s）、GQA-4 $I$=4、GQA-8 $I$=8（16.3）、MQA-64 $I$=64（130.5）；decode 权重 $B$=1、16、64、256；Llama-2-7B prefill 2048 $I\approx1971$（在屋顶上）。说明文字注明“点的位置是按公式算的强度，不是实测 kernel profile”。

**图 4.6 decode 每步时间与 batch**。数据用 §4.6 表。两个面板（Llama-2-7B、Llama-3-8B），x：batch，y：每步下界（ms）拆成“读权重”（常数 ≈7.5 ms，`--fig-muted`）+“读 KV”（线性增长，`--fig-amber`）；计算时间画成细线（远在下方）。竖线标出显存装满时的最大 batch（30 与 119）。第二个 y 轴或另一面板：总吞吐上界（133→788；131→3075 tok/s）。

**图 4.7 MHA / GQA / MQA 的头共享**（据 ainslie2023 Figure 2 重绘）。$n_h$=8 个查询头（`--fig-blue`）；MHA 8 对 K（`--fig-violet`）V（`--fig-green`）；GQA $n_g$=2（每 4 个 Q 头连一对 KV）；MQA 1 对。每联下方标“每 token 每层缓存 $2n_gd_h$ 个元素：16$d_h$ / 4$d_h$ / 2$d_h$”。右侧小图（据 Figure 1）：4 个 K 头投影矩阵取平均 → 1 个。

**图 4.8 MQA 能装下多长的上下文**（据 pope2023 Table 1）。分组柱状（batch 128 / 512）：MHA 1320/330；基线 MQA 660/165；优化 MQA 43,000/10,700。对数纵轴。标注“30% 显存留给 KV，64 × TPU v4”，并在基线 MQA 旁注“KV 在 64 块芯片上复制”。

**图 4.9 MQA 的质量与速度账**（shazeer2019 Table 1+2）。x：解码器每 token 摊销时间（μs）：MHA 46、MQA 3.8、MHA local 23、MQA local 3.3；y：dev BLEU：26.7、26.5、26.6、26.5。另标出“减头/减维”的几个 MHA 变体的 BLEU（25.8、26.2、26.1、25.8），它们没有给出速度，只画在右侧一栏作对比。

**图 4.10 温度、top-$k$、top-$p$**。用 §4.9 的玩具 logits。三行：$\tau$=0.5/1/2 的柱状分布（附熵）；top-2 截断（灰掉 2,3,4，重归一化后 0.7311/0.2689）；top-$p$=0.9（保留 0–3，累计 0.972）。再加 holtzman2020 Figure 5 思想的两个示意分布（一平一尖）说明固定 $k$ 的问题（示意，非数据）。

**图 4.11 投机采样的几何**。玩具分布 $p$=[0.5,0.3,0.2]（目标，`--fig-green`）与 $q$=[0.2,0.5,0.3]（草稿，`--fig-blue`）并排柱状；重叠部分 $\min(p,q)$=[0.2,0.3,0.2]（和 = $\alpha$=0.7）加深；$p$ 超出 $q$ 的部分 [0.3,0,0] 标为“残差 $\max(0,p-q)$”（`--fig-orange`），$q$ 超出 $p$ 的部分 [0,0.2,0.1] 标为“被拒掉的质量”（`--fig-red`）。下方一行文字：接受概率 [1, 0.6, 0.667]。下半可画 Leviathan Figure 5 式的轨迹：$\gamma$=3，草稿 3 小块 + 目标 1 大块。

**图 4.12 每轮期望产出 token 数**（Leviathan Fig. 2，按式 (1) 计算）。x：$\alpha\in[0,1)$，y：$E[\#]$；曲线 $\gamma$=1,2,3,5,7,10（§4.8 网格）；虚线 $1/(1-\alpha)$。在 $\alpha$=0.85、$\gamma$=1 处打点标“DeepSeek-V3 MTP：1.85”。

**图 4.13 加速 vs 额外运算**（Leviathan Table 1 / Fig. 4）。x：总运算量倍数，y：加速倍数；六个点（1.53,1.96）（1.58,2.53）（1.23,2.44）（1.63,3.69）（1.11,2.71）（1.60,6.86），标注 $(\alpha,\gamma)$。说明“投机解码用更多算术换更少的串行步与访存”。

**图 4.14 显存里的 KV cache：碎片与分页**（据 kwon2023 Figure 1、3、6 简化重绘）。左：40GB A100 上 13B 模型的显存饼/条：权重 65%、KV ~30%、其他。中：连续预分配的三种浪费（预留、内部碎片、外部碎片）示意。右：block size=4 的 block table（逻辑块 0,1,2 → 物理块 7,1,3）。正文标注“现有系统只有 20.4%–38.2% 的 KV 显存存了真 token”。

**（可选）图 4.15 GPU 规格卡**：A100 / H100 / H800 三栏：稠密 BF16、带宽、NVLink、脊点；H800 的算力与带宽格子写“未公开核实”。

---

## 6. 算例与练习素材

**练习 1（计算）DeepSeek LLM 67B 的 KV 账单。** 已知 95 层、64 个查询头、8 个 KV 头、$d_h$=128、bf16。求每 token KV 字节；4K 与 32K 下单条序列的缓存；若改成 MHA 是多少。
答：$2\cdot8\cdot128\cdot95\cdot2$ = 389,120 B = 380 KiB；4K：1.484 GiB（1.59 GB）；32K：11.875 GiB（12.75 GB）；MHA：×8 = 3,112,960 B（2.97 MiB），32K 时 95 GiB（102 GB）——一条序列就装不进一张 80 GB 卡。追问：为什么它的每 token KV 比 Llama-2-70B（320 KiB）还大？（95 层 vs 80 层，$95/80$=1.1875 = 380/320。）

**练习 2（概念 + 推导）为什么缓存 K、V 而不缓存 Q？为什么缓存是精确的？** 答：见 §2.2 四步：式 (7) 只用当前 $\mathbf q_t$；因果掩码使 $\mathbf h_j^{(l)}$ 只依赖 $x_{\le j}$；于是 $\mathbf k_j,\mathbf v_j$ 一经算出永不改变。反例：双向注意力（编码器）里 $\mathbf h_j$ 依赖未来 token，新 token 到来会改变旧位置的 K、V，缓存失效。

**练习 3（计算）缓存省下多少 FLOPs。** Llama-2-7B（$N_{mm}$≈6.607e9，$l$=32，$n_hd_h$=4096），在 $t$=4096 生成下一个 token。答：有缓存 1.536e10（15.4 GFLOP），其中注意力 14%；无缓存 5.852e13（58.5 TFLOP）；比值约 3810。再问：从零生成 4096 个 token 的总 FLOPs 比值？答：1.169e17 / 5.852e13 ≈ 1997 ≈ $T/2$；并指出有缓存的总 FLOPs 恰等于一次 4096-token prefill。

**练习 4（计算）roofline。** A100 80GB SXM：312 TFLOP/s、2.039 TB/s。(a) 脊点？(b) decode 时注意力部分的强度与 batch 无关，为什么？MHA、GQA-8（64 头）、MQA（64 头）分别能达到多少 TFLOP/s？(c) 权重部分要多大 batch 才到脊点？(d) 换 H100 SXM（989、3.35）呢？
答：(a) 153 FLOP/B；(b) 每条序列的 KV 只被这条序列的查询用，FLOPs 与字节都随 $B\cdot t$ 线性增长，比值 $n_h/n_g$；2.0 / 16.3 / 130.5 TFLOP/s；(c) $B\approx153$；(d) 脊点 295，H100 算力涨 3.17 倍、带宽只涨 1.64 倍，decode 更“饿”。

**练习 5（计算）GQA 让同一张卡服务更多用户。** A100 80GB，bf16，上下文 4096。Llama-2-7B（权重 13.48 GB，KV 512 KiB/token）与 Llama-3-8B（16.06 GB，128 KiB/token）各能同时放多少条序列？各自 batch 1 与满 batch 的每步延迟下界与总吞吐上界？
答：30 条 vs 119 条；batch 1：7.53 ms vs 7.62 ms（都 ≈ 读一遍权重）；满 batch：38.1 ms / 788 tok/s vs 38.7 ms / 3075 tok/s（§4.6）。

**练习 6（复算论文）Pope 的“3 TB”。** PaLM 540B 的 MHA 变体：118 层、48 头、$d_{head}$=128，bf16；batch 512、上下文 2048。答：$2\cdot118\cdot48\cdot128\cdot2$ = 2,899,968 B/token；×1,048,576 token = 3.04 TB；参数 540e9×2 B = 1.08 TB，比值 2.8 ≈ “3 倍”。附加：用 Table 1 的设定（64 芯片×32 GiB，30% 给 KV），复算优化 MQA（$d_{head}$=256，每 token 120,832 B，不复制）在 batch 128 时的最大上下文：614.4 GiB / (120,832·128) ≈ 42,654（论文 43,000）。

**练习 7（推导）投机采样为什么不改变分布。** 给出 $p$、$q$，按“$q(x)\le p(x)$ 接受，否则以 $p/q$ 接受；拒绝后从 $norm(\max(0,p-q))$ 采”的规则，证明输出服从 $p$，并证明接受率 $\beta=\sum_x\min(p,q)=1-\mathrm{TV}(p,q)$。答：附录 A.1 两行（§2.8）。数值验证：$p$=[0.5,0.3,0.2]、$q$=[0.2,0.5,0.3]，$\alpha$=0.7，残差归一化 [1,0,0]；$P(x=0)=0.2+0.3\cdot1=0.5$，$P(x=1)=0.3+0$，$P(x=2)=0.2+0$。

**练习 8（计算）期望产出与加速。** $\alpha$=0.8、$\gamma$=5：(a) 每轮期望 token 数；(b) $c$=0.05 时的墙钟加速；(c) $\hat c$=0 时总运算量倍数；(d) 若 $\gamma$ 无限大且 $c$=0，上界？
答：(a) $(1-0.8^6)/0.2$ = 3.69；(b) 3.69/(5·0.05+1) = 2.95；(c) $0.2\cdot6/(1-0.8^6)$ = 1.63；(d) $1/(1-0.8)$ = 5。

**练习 9（联系 DeepSeek）MTP 的 1.8 倍。** DeepSeek-V3 用 1 个 MTP 模块做草稿（$\gamma$=1），第二个 token 的接受率 85%–90%。按 Leviathan 式 (1)，每次目标模型前向期望产出几个 token？与报告的 1.8× TPS 比较。答：$1+\alpha$ = 1.85–1.90；报告 1.8×，差值可归于 MTP 模块自身开销（$c>0$）与实现开销（笔者推测）。若按 ISCA 的 80%，下限是 1.80。

**练习 10（计算）采样截断。** logits [2.0, 1.0, 0.5, 0.0, −1.0]。求 $\tau$=1 的 softmax、top-2 重归一化分布、top-$p$=0.9 的候选集；$\tau$=0.5 与 2 时熵如何变化？答：见 §4.9（0.563/0.207/0.126/0.076/0.028；[0.731,0.269]；{0,1,2,3}；熵 0.878 / 1.741 / 2.152 bit）。

**练习 11（复算论文）为什么投机采样能“超过带宽上限”。** Chinchilla 70B、bf16、16 个 TPU v4（每芯片 1200 GB/s，数取自 Pope）。自回归的带宽下界是多少 ms/token？Chen 的 HumanEval 结果 5.73 ms/token 为什么不矛盾？答：140 GB / 19.2 TB/s = 7.29 ms；投机采样每读一遍权重产出不止一个 token（每轮期望 >1），所以“每 token”时间可以低于“读一遍权重”的时间；它突破的是“每 token 读一遍权重”这个前提，不是带宽本身。

**练习 12（概念）哪些办法对 decode 的哪一部分有效？** 列出：加大 batch、GQA/MQA、int8 权重、投机解码、FlashAttention、PagedAttention，分别作用于“权重访存”“KV 访存”“KV 容量”“串行步数”中的哪一项。参考答案：加大 batch → 摊薄权重访存（强度 = $B$），对 KV 访存无效；GQA/MQA → KV 访存与容量都降 $n_h/n_g$ 倍，注意力强度升到 $n_h/n_g$；int8 权重 → 权重访存减半（Pope：低 batch 成本改善“just over a factor of 2”）；投机解码 → 串行步数与每 token 的权重/KV 读取次数降为 $1/E[\#]$，总运算量增加；FlashAttention → 主要减少训练/prefill 中 $N\times N$ 矩阵的 HBM 往返，不缩小 KV cache；PagedAttention → 减少 KV 容量的碎片浪费（从 20–38% 利用率到接近满），间接增大可用 batch。

**练习 13（小推导，可选）Shazeer 的比值。** 从 §2.4.1 的“运算 $\Theta(bnd^2)$、访存 $\Theta(bn^2d+nd^2)$”推出 $\Theta(\frac nd+\frac1b)$；再对 MQA 把 K、V 的访存从 $bn^2d$ 改为 $bn^2k=bn^2d/h$，推出 $\Theta(\frac1d+\frac n{dh}+\frac1b)$。追问：这个比值与“算术强度”是什么关系？（倒数。）

---

## 7. 勘误与常见误解

**E1 FlashAttention 的 arXiv 号是 2205.14135，不是 2205.14314。** 后者是无关论文（本仓库 mamba 笔记勘误 #2 已记）。本次再次核对：2205.14135v2。

**E2 roofline 原文不叫 arithmetic intensity。** Williams 等人（TR §3）定义的是 operational intensity（每字节 **DRAM** 流量的运算数），并明确解释不用 arithmetic intensity 的两个理由（后者量的是处理器与 cache 间的流量；模型也适用于非算术运算）。“arithmetic intensity”一词他们归于 Harris 2005。FlashAttention §2.1 与附录 A（“the Roofline model of arithmetic intensity”）把两者混用，ML 系统文献普遍跟随。课文沿用“算术强度”，但须在 `::: pre` 中交代原名与区别；引 roofline 时引 Williams 的“operational intensity”，引“arithmetic intensity”的定义时引 FlashAttention §2.1。（与 mamba 笔记勘误 #11 一致。）另：Williams 的 TR 标题是“…for Floating-Point Programs and Multicore Architectures”，FlashAttention 所引 CACM 标题为“…for multicore architectures”，两者不同。

**E3 Shazeer 从未写 “arithmetic intensity”，而且他写的是它的倒数。** grep 计数 0；他的量是“ratio of memory access to arithmetic operations”。课文若写“Shazeer 证明 decode 的算术强度约为 1”，应改为“Shazeer 推出访存/运算之比为 $\Theta(\frac nd+\frac1b)$，在 $n\approx d$ 或 $b\approx1$ 时接近 1”。批量式 $O(\frac1k+\frac1{bn})$ 在 §2.3.1（不是 §2.2），其中 $k$ 是键维度（$=d/h$），不是 KV 头数（mamba 笔记勘误 #14 再次核实）。

**E4 Shazeer 论文里的笔误（引用代码与公式时不要照抄）。** (a) §2.4.1 写 K、V 张量“size $bhmk=bn^2$”——按其自身假设 $k=d/h$、$m=n$，应为 $bhmk=bnd$（笔者判断为笔误）。(b) `MultiheadSelfAttentionIncremental` 里用了未定义的 `M`（应为 `x`）和 `O`（应为 `o`）；MQA 增量代码同样如此，且对形状 `[b,k]` 的张量 `expand_dims(axis=2)` 后与 `[b,m,k]` 在 axis=2 拼接，维度对不上（应为 axis=1）。(c) `MultiqueryAttentionBatched` 文档串里 “P_v: a tensor with shape d, v]” 缺括号。(d) §2.2 “seuqence”、§4.1 “preformed” 等拼写错误。课文若展示 einsum 代码，应给修正版并注明。

**E5 投机解码的记号：Chen 与 Leviathan 的 $p$、$q$ 正好相反。** Leviathan：$p$=目标、$q$=草稿，接受概率 $\min(1,p/q)$，残差 $norm(\max(0,p-q))$。Chen：$q$=目标、$p$=草稿，接受概率 $\min(1,q/p)$，残差 $(q-p)_+$。两篇并排引用时最容易抄错。课程统一用 Leviathan 记号。

**E6 “speculative sampling”这个名字两篇用法不同。** Leviathan：speculative sampling 是其中的**采样规则**，speculative decoding 是整个**解码机制**；Chen：speculative sampling（SpS）指整个算法。

**E7 投机解码“不改变输出”的准确含义。** 保证的是输出**分布**与目标模型相同（对任意草稿模型成立，Leviathan §3.6、附录 A.1）；Chen 更谨慎地说“within hardware numerics”，并指出即使 greedy，数值差异也可能让两条序列分叉，所以只能在分布/基准分数层面验证（§6.1）。常见误解：“投机解码减少计算量”——错，Theorem 3.11 说明总运算量**增加**；减少的是串行调用次数和权重/KV 的读取次数。

**E8 MQA 的 “KV 缩小 $n_h$ 倍”在张量并行下会失效。** 按头切分到多于 KV 头数的芯片上时，共享的 K、V 必须复制到每块芯片，节省消失（Pope §3.3；Llama 2 A.2.1 也指出 8 卡 TP 时 MQA 复制后 KV 与 GQA 一样大，并因此选 GQA）。Pope 的解法是 decode 时按 batch 切注意力，但它有 batch 下限：Llama 2 A.2.1 说这种切法只在 batch 大于分片数时可用，且会让推理服务变复杂；Pope 附录 D 说 batch 小于 4（TPU v4 torus 一个轴的最小长度）时没有加速。

**E9 GQA 的“5%”与适用范围。** 摘要说“5% of original pre-training compute”，正文说续训“原预训练**步数**的 $\alpha$ 比例”；约 600 TPUv3 chip-days。实验只在 T5（编码器-解码器）上做；GQA 不用于编码器自注意力；没有与从头训练的 GQA 比较（Limitations）。GQA 论文自称 Rabe（2023）独立实现过 GQA（§4）——“GQA 由 Ainslie 等人首创”应写成“由 Ainslie 等人提出并系统评估”。另外，Llama 2/3、DeepSeek LLM 67B 都是**从头训练**的 GQA，并非 uptraining。

**E10 H100 的稠密 BF16 不是 NVIDIA 印出来的数。** 产品页 BF16 为 1,979 TFLOPS*（稀疏）；页面上的“989”是 **TF32 稀疏**那一行。稠密 989 的可引用来源是 FlashAttention-3（原文说“FP16 matmul”，1830 MHz）。A100 数据表则同时印稠密 312 与稀疏 624。3.9 TB/s 是 H100 NVL，不是 H100 SXM（3.35 TB/s）。（与 mamba 勘误 #15 一致，本次重新从产品页抓取核对。）

**E11 A100 带宽要分型号。** 2,039 GB/s 只属于 80GB SXM；80GB PCIe 1,935；40GB 1,555。FlashAttention 笼统写“1.5-2.0TB/s”。算脊点时说明用的是哪一型。

**E12 “H800 NVLink 400 GB/s”出自 ISCA 2025 论文，不在 V3 技术报告里。** V3 报告（§3.2.2）只写“NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)”。ISCA 论文 §4.1 写 900→400 GB/s，§4.3 写“NVLink provides 200GB/s（约 160GB/s 可达）”。400 与 200 的关系（双向 vs 单向）论文没有明说，是笔者推断（NVIDIA 的 900 GB/s 是双向总带宽的惯例）；放入 §8。课文写 H800 时引 zhao2025，不要引 V3 报告。

**E13 ISCA 论文 Table 1 的 LLaMA-3.1 405B 倍数算错。** 516.096 / 70.272 = 7.344，表中印 7.28x（v1、v2 相同）；Qwen 的 4.66x 正确。课文若画此图，用 7.34 并加脚注。

**E14 两份 DeepSeek 文档对 MTP 接受率的说法不同。** V3 报告 §5.4.3：85%–90%；ISCA §2.3.3：80%–90%。都说 1.8× TPS。引用时注明出处，不要混写成“80%–90%（V3 报告）”。

**E15 Leviathan “$c$ 总小于 0.05”有限定语。** §3.3 原话限定于“草稿比目标小几个数量级”的情形；附录 Table 4 里 T5-large 做草稿时 $c$=0.11（它只比 T5-XXL 小约 14 倍）。不要把“$c<0.05$”当成普遍事实。

**E16 KV cache 不是“token 的缓存”，而是“某位置、某上下文下的 token 的 K、V”。** 同一 token 出现在不同位置 KV 不同（kwon2023 §2.2）。常见误解：以为可以跨请求按 token id 复用 KV。能复用的只有**完全相同的前缀**（prefix caching，PagedAttention §4.4 的共享前缀即此，本课不展开）。

**E17 KV cache 省的是计算，不省空间；FlashAttention 省的是 HBM 往返，不省 KV。** “纪要本”类比的边界：纪要是逐字副本，空间随发言人数线性增长。FlashAttention（Theorem 2）减少的是注意力矩阵 $N\times N$ 中间结果的 HBM 读写，对 decode 阶段的 KV 容量没有影响。

**E18 DeepSeek-V3 的 config 会误导 KV 计算器。** `config.json` 里 `num_key_value_heads`=128、`num_attention_heads`=128，但 MLA 实际缓存的是 `kv_lora_rank`+`qk_rope_head_dim` = 512+64 = 576 个元素/层。按 `2*num_key_value_heads*head_dim*layers` 公式会得出 3,997,696 B/token，比真实值 70,272 B 大 56.9 倍。第 9 课要专门讲；本课只需提醒“公式要看缓存的是什么”。

**E19 DeepSeek LLM 67B 的 GQA 缓存比 Llama-2-70B 大。** 同为 8 个 KV 头、$d_h$=128，但 95 层 vs 80 层，所以 380 KiB vs 320 KiB/token。“同为 GQA-8 所以 KV 一样”是错的。

**E20 论文内部的小错与版本差异（引用时注意）。**
- GPT-3 Table 2.1 的 13B 行 $d_{model}$=5140，但 $n_{heads}\cdot d_{head}$=40×128=5120（论文笔误）。
- Llama 3 词表：论文 Table 3 “128,000”，config `vocab_size`=128256（含特殊 token）。
- Holtzman：§4.1 写 GPT-2 Large “762M parameters”，Figure 1 图注写“774M”，自相矛盾；式 (4) 分母 $u_l'$ 应为 $u_{l'}$；温度记号 $t$ 与课程位置下标冲突。
- PagedAttention §2.1 “Besides the computation in Eq. 4” 应为 Eq. 3；“800 KB”实为 800 KiB；“1.6 GB” = 1.68 GB = 1.56 GiB。
- Chen §2 把 throughput 拼成 “throughout”；Algorithm 2 条件式里多处 “$x|,x_1$” 的多余逗号，以及“sample extra token $x_{n+K+1}\sim q(x|,x_1,\dots,x_n,x_{n+K})$”下标不连贯。
- Kaplan Table 1 “Attention: Mask” 的 $2n_{layer}n_{ctx}d_{attn}$ 未说明来源；按 QK 与 AV 两次矩阵乘、因果平均正好得到这个数（笔者解读）。decode 单步在位置 $t$ 的注意力 FLOPs 应是 $4ln_hd_ht$，是 Kaplan 那一项在 $n_{ctx}=t$ 时的两倍——不要直接套 Kaplan 的式子算 decode 单步。

**E21 “prefill”一词的来源。** 至少 Pope 等人是以“which we call ‘prefill’”自己命名的；vLLM 论文叫 prompt phase，DeepSeek 叫 prefilling stage。不要写“prefill 是 Pope 等人首创”——未追溯最早用法（§8）。

**E22 top-$k$ 的归属。** 常被单独归于 Fan et al. 2018；Holtzman 同时引了 Fan 2018、Holtzman 2018、Radford 2019。Fan 等人在 §5.4 把它当作生成细节（$k$=10），不是标题级贡献。课文写“top-$k$ 采样（Fan 等人 2018 年的故事生成工作中使用 $k$=10，随后被 GPT-2 等广泛采用）”较稳妥；GPT-2 部分未核（§8）。

**E23 Pope Table 1 的复现需要两个假设。** 复算与论文差 <1% 的前提是：MHA 变体按 pad 后的 64 头计（§4 “padded the number of attention heads up from 48 to 64”），基线 MQA 的单个 KV 头在 64 块芯片上各复制一份。“3TB”那句（§2.1）却对应未 pad 的 48 头。两处用的头数不同，写课文时不要混用。

**E24 “decode 受带宽限制”的准确说法。** 受限的是 decode 中的**矩阵-向量**部分（小 batch 的线性层、全部注意力）；batch 足够大时线性层可以接近 compute-bound（Pope 高吞吐 decode MFU 33%，V3 decode 每专家 batch 通常 ≤256，所以“瓶颈是访存”）。注意力部分无论 batch 多大强度都只有 $n_h/n_g$（§2.6）。

---

## 8. 未能核实 / UNVERIFIED

1. **H800 的显存带宽与稠密 BF16 峰值。** 没读到 NVIDIA 官方 H800 规格页；ISCA 论文只说 FP64 与 NVLink 被降、其余“similar to the H100”。常见说法“与 H100 SXM 相同（3.35 TB/s、989 TFLOPS）”未核实。
2. **H800 的 400 GB/s 与 200 GB/s 是否分别是双向与单向。** 论文未说明，是笔者推断。
3. **PagedAttention Figure 2 的逐系统数值**（常被引用的 “vLLM 96.3%” 等）。图是矢量图，本机无渲染工具，未能读出；只核实了正文的 20.4%–38.2% 区间。
4. **Llama-2-7B/70B、Llama-3-8B/70B 的官方 config。** meta-llama 官方 repo 需要登录（401），用的是 NousResearch 镜像；Llama-2-70B 的层数/宽度/头数在论文里没有表，**只有**镜像 config 为证（论文只确认 70B 用 GQA、消融语境里“8 KV projections”）。
5. **Llama-3.1-405B 的 config**（镜像也 401）。用的是 Llama 3 论文 Table 3（405B：126 层、16384、128 头、8 KV 头），并假设 3.1 与 3 的架构相同。
6. **Chen et al. 2023 是否有同行评审版本。** 只查了 arXiv，未系统检索会议。
7. **温度采样最早出自 Ackley et al. 1985。** 只来自 Holtzman §3.3 的引用，没有读原文。
8. **“prefill”“KV cache”两个术语的最早使用。** 未追溯；Pope 用“which we refer to / which we call”自称命名。
9. **Orca 的机制细节。** 只读了 USENIX 摘要页（WebFetch 摘要工具返回的句子），没读全文；iteration-level scheduling 的描述主要依赖 kwon2023 §2.3 的转述。
10. **Rabe（2023）独立实现 GQA。** 来自 GQA 论文 §4 的一句话，未查看 Rabe 的实现。
11. **Williams et al. CACM 正式版的文字。** 实际读的是 2008 年技术报告 UCB/EECS-2008-134；CACM 预印本 PDF 文本层乱码无法提取。定义句与公式在 TR 中核实，CACM 版是否逐字相同未核。
12. **FlashAttention 的 NeurIPS 2022 发表信息**未到会议页核对（仅凭一般认知，需补核）。
13. **GQA v1 与 v3、Leviathan v1 与 v2 的差异。** 只读了最新版；v3 的 comment 只写“Added to related work”。
14. **Kaplan 表 1 “Mask” 项的来源解释**（因果平均）是笔者解读，论文未说明。
15. **Shazeer “$bhmk=bn^2$” 是笔误**这一判断是笔者依其假设推出的，没有作者勘误可引。
16. **Chen 的 1200 GB/s 带宽**取自 Pope 对 TPU v4 的描述，Chen 论文本身没给带宽数；另假设 Chinchilla 权重为 bf16。
17. **DeepSeek MTP 接受率的测量条件**（采样温度、数据集、batch）两份文档都没给，未能核实“85%–90%”与“80%–90%”哪个更新。
18. **GPT-2 使用 top-$k$=40**（Holtzman Table 1 与 Figure 6 图注暗示“Top-k=40, t=0.7 as in Radford et al. 2019”），未读 GPT-2 报告核实。
19. **DeepSeek 线上 decode 的实际 TPOT、batch、KV 占用**：本课不引用具体生产数字；ISCA §2.3.2 的 14.76 ms 是其理论上限推算（EP 通信受限），不是实测。
20. **FlashAttention Figure 1 左的内存层级数值**（CPU DRAM 等）未从图中读取，只核对了正文 §2.1 的 A100 数字；画图时不要凭记忆填图中数值。
21. **GPT-3 的词表大小 50257。** GPT-3 论文没有单列词表大小；§4.2 参数量复算里按 GPT-2 BPE 的 50257 计（嵌入只占 0.62B，对 KV 与权重字节的结论无影响）。
22. **H100 的 989 是 FP16 还是 BF16。** FA3 原文说“FP16 matmul”；H100 产品页上 BF16 与 FP16 的稀疏值相同（均 1,979*），据此把 989 也用于 BF16，这一步是推断。
23. **GPT-3 与 FlashAttention 的会议信息**（NeurIPS 2020 / NeurIPS 2022）未到会议页核对；Llama 2、Llama 3、Kaplan、LLaMA 按 arXiv 预印本引用。

---

## 9. 参考文献卡片草稿

> 格式：id；title；a（作者，年，venue，arXiv id）；url；finding；matters；kind（仅制品标 artefact）。

- **id**: shazeer2019
  **title**: Fast Transformer Decoding: One Write-Head is All You Need
  **a**: Noam Shazeer, 2019, arXiv preprint, arXiv:1911.02150v1
  **url**: https://arxiv.org/abs/1911.02150
  **finding**: 推出增量解码的访存/运算之比为 $\Theta(\frac nd+\frac1b)$，提出让所有头共享一组 K、V 的多查询注意力，把 $\frac nd$ 项缩小 $h$ 倍，WMT14 上解码器每 token 从 46 μs 降到 3.8 μs、BLEU 仅略降。
  **matters**: 它是本课“带宽墙”论证的原始出处，也是 MQA→GQA→MLA 这条 KV 压缩主线的起点。

- **id**: ainslie2023
  **title**: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
  **a**: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai, 2023, EMNLP 2023, arXiv:2305.13245v3
  **url**: https://arxiv.org/abs/2305.13245
  **finding**: 把查询头分成 $G$ 组、每组共享一对 K、V；用均值池化转换 MHA checkpoint 再续训 5% 即可，GQA-8 在 T5-XXL 上质量接近 MHA、速度接近 MQA。
  **matters**: 它给出了 Llama 2/3 与 DeepSeek LLM 67B 所用注意力的定义和“中间头数”这一折中的依据。

- **id**: pope2023
  **title**: Efficiently Scaling Transformer Inference
  **a**: Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean, 2022/2023, MLSys 2023, arXiv:2211.05102v1
  **url**: https://arxiv.org/abs/2211.05102
  **finding**: 在 PaLM 540B 上系统分析 prefill 与 decode 的 memory time / compute time 与分片策略，指出 MHA 的 KV cache 可达参数的 3 倍，按 batch 切分的 MQA 能把最大上下文提高约 32 倍，并实现 28.5 ms/token 的低延迟解码。
  **matters**: 本课 prefill/decode 术语、“权重主导 vs KV 主导”的判据以及 MQA 在张量并行下需要复制 KV 这一陷阱都引自它。

- **id**: dao2022
  **title**: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
  **a**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, 2022, NeurIPS 2022（会议信息待核）, arXiv:2205.14135v2
  **url**: https://arxiv.org/abs/2205.14135
  **finding**: 用分块与重算把注意力的 HBM 访问从 $\Theta(Nd+N^2)$ 降到 $\Theta(N^2d^2M^{-1})$，并给出 A100 内存层级与 compute-/memory-bound 的分类。
  **matters**: 本课用它给出 ML 文献中“算术强度”的定义与 A100 数字，同时用它澄清“FlashAttention 不缩小 KV cache”。

- **id**: shah2024
  **title**: FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
  **a**: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao, 2024, arXiv preprint, arXiv:2407.08608v2
  **url**: https://arxiv.org/abs/2407.08608
  **finding**: 在 H100 上利用异步与 FP8 把注意力做到 FP16 740 TFLOPs/s（75% 利用率），并写明 H100 SXM5 的 FP16 矩阵乘峰值为 989 TFLOPS、显存 3.35 TB/s。
  **matters**: 它是 H100 稠密峰值 989 的可引用来源，NVIDIA 官方页面只给稀疏值。

- **id**: kwon2023
  **title**: Efficient Memory Management for Large Language Model Serving with PagedAttention
  **a**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica, 2023, SOSP 2023, arXiv:2309.06180v1
  **url**: https://arxiv.org/abs/2309.06180
  **finding**: 把 KV cache 按固定大小的块分页管理，消除预分配造成的碎片（现有系统仅 20.4%–38.2% 的 KV 显存存了真实 token），吞吐提高 2–4 倍。
  **matters**: 本课用它给出 KV cache 的逐字节例子（OPT-13B 800 KiB/token）和“KV 依赖全部前文、不能按 token id 复用”的表述，并点到分页与连续批处理。

- **id**: yu2022
  **title**: Orca: A Distributed Serving System for Transformer-Based Generative Models
  **a**: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun, 2022, OSDI 2022, 无 arXiv
  **url**: https://www.usenix.org/conference/osdi22/presentation/yu
  **finding**: 提出 iteration-level scheduling，按每次迭代而非整个请求来调度批处理。
  **matters**: 它是“连续批处理”一词背后机制的原始出处，本课只需一句话提到。

- **id**: leviathan2023
  **title**: Fast Inference from Transformers via Speculative Decoding
  **a**: Yaniv Leviathan, Matan Kalman, Yossi Matias, 2023, ICML 2023 (PMLR 202:19274–19286), arXiv:2211.17192v2
  **url**: https://arxiv.org/abs/2211.17192
  **finding**: 用小模型起草、大模型并行验证，以 $\min(1,p/q)$ 接受并从 $norm(\max(0,p-q))$ 补采，严格保持目标分布；每轮期望产出 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 个 token，T5-XXL 上加速 2–3 倍。
  **matters**: 本课投机解码的全部公式与证明都按它转录，也是第 11 课 MTP 用作草稿的理论基础。

- **id**: chen2023
  **title**: Accelerating Large Language Model Decoding with Speculative Sampling
  **a**: Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper, 2023, arXiv preprint, arXiv:2302.01318v1
  **url**: https://arxiv.org/abs/2302.01318
  **finding**: 与 Leviathan 独立地提出同一接受-拒绝方案，用 4B 草稿模型在 16 个 TPU v4 上给 Chinchilla 70B 带来 1.9–2.5 倍解码加速，部分任务突破了自回归的带宽上限。
  **matters**: 它提供了“并行打分几乎不比采一个 token 贵”的硬件层解释，并且是记号反转这一易错点的来源。

- **id**: holtzman2020
  **title**: The Curious Case of Neural Text Degeneration
  **a**: Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi, 2020, ICLR 2020, arXiv:1904.09751v2
  **url**: https://arxiv.org/abs/1904.09751
  **finding**: 定义 nucleus（top-$p$）采样：取累计概率达到 $p$ 的最小词集并重归一化，在 GPT-2 Large 上比 beam search、纯采样、固定 top-$k$ 更接近人类文本统计。
  **matters**: 本课温度、top-$k$、top-$p$ 的公式（式 2–4）都从这里转录。

- **id**: fan2018
  **title**: Hierarchical Neural Story Generation
  **a**: Angela Fan, Mike Lewis, Yann Dauphin, 2018, ACL 2018 (pp. 889–898), arXiv:1805.04833v1
  **url**: https://aclanthology.org/P18-1082/
  **finding**: 故事生成时从最可能的 $k=10$ 个候选中随机采样，认为比 beam search 更少重复、比纯随机采样更少低概率错误。
  **matters**: 它是 top-$k$ 采样最常被引用的早期出处，本课据此交代 top-$k$ 的来历与归属的谨慎写法。

- **id**: williams2009
  **title**: Roofline: An Insightful Visual Performance Model for Multicore Architectures（TR 版标题含 “Floating-Point Programs and”）
  **a**: Samuel Williams, Andrew Waterman, David Patterson, 2009, Communications of the ACM 52(4):65–76；技术报告 UCB/EECS-2008-134（2008）
  **url**: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html
  **finding**: 用 min(峰值算力, 带宽 × operational intensity) 画出可达性能的“屋顶”，脊点横坐标是达到峰值所需的最小强度。
  **matters**: 本课的 roofline 图与“decode 在脊点左侧很远”的论证都以它为框架，且须按它纠正“arithmetic intensity”的命名。

- **id**: kaplan2020
  **title**: Scaling Laws for Neural Language Models
  **a**: Jared Kaplan et al., 2020, arXiv preprint, arXiv:2001.08361v1
  **url**: https://arxiv.org/abs/2001.08361
  **finding**: 给出 Transformer 每 token 前向 FLOPs 的逐项分解，$C_{forward}\approx2N+2n_{layer}n_{ctx}d_{model}$。
  **matters**: 本课“有缓存 vs 无缓存”的 FLOPs 推导以它的 $2N$ 计数为起点，并需说明它的注意力项与 decode 单步不同。

- **id**: brown2020
  **title**: Language Models are Few-Shot Learners
  **a**: Tom B. Brown et al., 2020, NeurIPS 2020（待核）, arXiv:2005.14165v4
  **url**: https://arxiv.org/abs/2005.14165
  **finding**: GPT-3 175B 为 96 层、$d_{model}$=12288、96 头、$d_{head}$=128，上下文 2048。
  **matters**: 它是本课 KV 账单里“最大的 MHA 例子”（4.5 MiB/token）的规格来源。

- **id**: touvron2023a
  **title**: LLaMA: Open and Efficient Foundation Language Models
  **a**: Hugo Touvron et al., 2023, arXiv preprint, arXiv:2302.13971v1
  **url**: https://arxiv.org/abs/2302.13971
  **finding**: Table 2 给出 LLaMA 7B（6.7B）为 4096 维、32 头、32 层。
  **matters**: Llama 2 沿用 Llama 1 架构，本课用它交叉核对 Llama-2-7B 的维度。

- **id**: touvron2023b
  **title**: Llama 2: Open Foundation and Fine-Tuned Chat Models
  **a**: Hugo Touvron et al., 2023, arXiv preprint, arXiv:2307.09288v2
  **url**: https://arxiv.org/abs/2307.09288
  **finding**: 34B 与 70B 采用 GQA；30B 规模消融中 GQA 与 MHA 大体持平、优于 MQA，且 MHA 在大 batch/长上下文时 OOM。
  **matters**: 它给了本课一个工业界从 MHA 换到 GQA 的真实理由（KV 容量与张量并行下 MQA 的复制问题）。

- **id**: grattafiori2024
  **title**: The Llama 3 Herd of Models
  **a**: Aaron Grattafiori, Abhimanyu Dubey, et al., 2024, arXiv preprint, arXiv:2407.21783v3
  **url**: https://arxiv.org/abs/2407.21783
  **finding**: Llama 3 的 8B/70B/405B 全部使用 8 个 KV 头的 GQA，并给出层数、宽度、头数表。
  **matters**: 本课 Llama-3 与 LLaMA-3.1 405B 的 KV 字节复算都以其 Table 3 为准。

- **id**: deepseek2024llm
  **title**: DeepSeek LLM: Scaling Open-Source Language Models with Longtermism
  **a**: DeepSeek-AI, 2024, arXiv preprint, arXiv:2401.02954v1
  **url**: https://arxiv.org/abs/2401.02954
  **finding**: 7B 为 30 层 MHA，67B 为 95 层、64 个查询头、8 个 KV 头的 GQA，为降低推理成本而选 GQA 并把参数加在深度上。
  **matters**: 它是 DeepSeek 自己在 KV 问题上的第一步，本课用它的规格算出 380 KiB/token 并引向第 9 课的 MLA。

- **id**: deepseek2024v2
  **title**: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
  **a**: DeepSeek-AI, 2024, arXiv preprint, arXiv:2405.04434v5
  **url**: https://arxiv.org/abs/2405.04434
  **finding**: Table 1 以元素数比较 MHA $2n_hd_hl$、GQA $2n_gd_hl$、MQA $2d_hl$ 与 MLA $(d_c+d_h^R)l\approx\frac92d_hl$ 的每 token 缓存。
  **matters**: 本课的内存公式直接采用它的记号，结尾的“能不能不减头数也把缓存压小”由这张表引出第 9 课。

- **id**: deepseek2024v3
  **title**: DeepSeek-V3 Technical Report
  **a**: DeepSeek-AI, 2024/2025, arXiv preprint, arXiv:2412.19437v2
  **url**: https://arxiv.org/abs/2412.19437
  **finding**: 线上把 prefilling 与 decoding 分开部署，decoding 阶段每专家 batch 通常 ≤256 且瓶颈在访存；MTP 预测第二个 token 的接受率 85%–90%，带来 1.8 倍 TPS。
  **matters**: 它把本课的 prefill/decode 分析、带宽墙和投机解码直接连到 DeepSeek 的真实部署，并为第 11、12 课铺路。

- **id**: zhao2025
  **title**: Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures
  **a**: Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Huazuo Gao, Jiashi Li, et al., 2025, ISCA 2025 (Industry Track), arXiv:2505.09343v2
  **url**: https://arxiv.org/abs/2505.09343
  **finding**: 说明 H800 的 NVLink 从 900 GB/s 降到 400 GB/s（有效约 160 GB/s），并比较 DeepSeek-V3（70.272 KB）、Qwen-2.5 72B、LLaMA-3.1 405B 的每 token KV cache。
  **matters**: 它是 DeepSeek 所用 GPU 互联限制的唯一一手来源，也提供了“GEMM 变 GEMV”这一 DeepSeek 自己对带宽墙的表述（其 7.28x 倍数须按勘误修正）。

**制品（kind: artefact）**

- **id**: nvidia-a100-ds；**title**: NVIDIA A100 Tensor Core GPU Data Sheet (JUN21)；**a**: NVIDIA, 2021；**url**: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf；**finding**: BF16 稠密 312 / 稀疏 624 TFLOPS，80GB SXM 带宽 2,039 GB/s，PCIe 1,935 GB/s，NVLink 600 GB/s。**matters**: 本课 A100 脊点 153 FLOP/B 与所有 A100 延迟下界的数据来源。**kind**: artefact
- **id**: nvidia-h100-page；**title**: NVIDIA H100 Tensor Core GPU（产品页规格表）；**a**: NVIDIA，读取于 2026-09-24；**url**: https://www.nvidia.com/en-us/data-center/h100/；**finding**: H100 SXM：BF16 1,979 TFLOPS（稀疏），TF32 989（稀疏），80GB、3.35 TB/s、NVLink 900 GB/s；H100 NVL：94GB、3.9 TB/s。**matters**: 用来说明 NVIDIA 只印稀疏值、以及 SXM 与 NVL 的带宽区别，避免把 989 当作官方稠密 BF16。**kind**: artefact
- **id**: hf-deepseek-llm-configs；**title**: deepseek-ai/deepseek-llm-7b-base 与 deepseek-llm-67b-base（及 67b-chat）config.json；**a**: DeepSeek-AI, Hugging Face；**url**: https://huggingface.co/deepseek-ai/deepseek-llm-67b-base/raw/main/config.json；**finding**: 67B：`num_hidden_layers`=95、`hidden_size`=8192、`num_attention_heads`=64、`num_key_value_heads`=8；7B：30 层、32/32 头。**matters**: 与论文 Table 2 逐项一致，确认 67B 为 GQA-8。**kind**: artefact
- **id**: hf-llama-mirrors；**title**: NousResearch/Llama-2-7b-hf、Llama-2-70b-hf、Meta-Llama-3-8B、Meta-Llama-3-70B config.json（第三方镜像）；**a**: NousResearch, Hugging Face；**url**: https://huggingface.co/NousResearch/Llama-2-70b-hf/raw/main/config.json；**finding**: Llama-2-70B 为 80 层、8192 维、64 头、8 KV 头；Llama-3-8B 为 32 层、32 头、8 KV 头。**matters**: 官方 repo 需登录，这是 Llama-2-70B 维度的唯一可读来源，所以相关数字在 §8 标为部分未核。**kind**: artefact
- **id**: hf-deepseek-v3-config；**title**: deepseek-ai/DeepSeek-V3 config.json；**a**: DeepSeek-AI, Hugging Face；**url**: https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json；**finding**: `kv_lora_rank`=512、`qk_rope_head_dim`=64、61 层，但 `num_key_value_heads` 也写作 128。**matters**: 用来演示“按 config 的 KV 头数套公式”会把 MLA 的缓存高估 56.9 倍。**kind**: artefact
- **id**: hf-qwen25-72b-config；**title**: Qwen/Qwen2.5-72B config.json；**a**: Qwen Team, Hugging Face；**url**: https://huggingface.co/Qwen/Qwen2.5-72B/raw/main/config.json；**finding**: 80 层、64 头、8 KV 头。**matters**: 只用于复算 ISCA 表 1 的 327,680 B/token。**kind**: artefact
- **id**: hf-opt-13b-config；**title**: facebook/opt-13b config.json；**a**: Meta AI, Hugging Face；**url**: https://huggingface.co/facebook/opt-13b/raw/main/config.json；**finding**: 40 层、5120 维、40 头、`ffn_dim` 20480、上下文 2048。**matters**: 核对 PagedAttention 的 800 KiB/token 例子。**kind**: artefact

---

## 10. 核查记录（对抗性复核，2026-09-24）

复核方法：对下列 24 组高风险论断，逐条回到一手来源重新取证。论文用 `arxiv.py grep/text` 读 LaTeXML 全文，Roofline 技术报告与 A100 数据表下载 PDF 后用 `pdftotext` 抽取，HF `config.json` 用 curl 直接读原始 JSON，H100 产品页用 WebFetch（经摘要模型转述，见下文“仍存疑”），会议页码用 curl 读 PMLR / ACL Anthology 页面。所有派生数字都在 Python 里重算（§4.1、§4.2、§4.4–§4.9 全表）。

### 10.1 核实无误（保留原文）

1. **Pope 命名 prefill / decode / KV cache**（C1、C6）：§2 第一段原句与 §1 原句逐字一致。
2. **Pope “3TB = 3 倍参数”**（C13）：§2.1 原句一致；MHA 变体 $d_{head}$=128（§4.2 “shrink d_head from 256 … to 128”）、48 头、118 层（Table D.1），重算 3.04 TB，540e9×2 B 的 2.82 倍。
3. **Pope Table 1**（C35）：1320/330、660/165、43,000/10,700 与表头 $d_{head}$ 128/256/256 一致；按 pad 到 64 头（§4 “padded … from 48 to 64”）重算 1333/333、666/167、42,654/10,663，差 <1%。E23 的“两处头数不同”成立。
4. **Pope Table 2 与 28.5 ms**（C78）：0.29 s/1.82 s/85.2 s/6.0 s、MFU 43/14/76/33%、int8/int8/bf16/bf16、batch 1/64/512/512 全部一致；§4.4 “28.5ms/token … batch size 64”“36.9ms/token”一致。附录 D 的 batch < 4 无加速一致。
5. **Shazeer 三个比值**（C17、C18、C30，§2.4 转录）：§2.3.1 $O(\frac1k+\frac1{bn})$、访存 $O(bnd+bhn^2+d^2)$；§2.4.1 $\Theta(\frac nd+\frac1b)$、访存 $\Theta(bn^2d+nd^2)$；§3.1 $\Theta(\frac1d+\frac n{dh}+\frac1b)$、访存 $\Theta(bnd+bn^2k+nd^2)$，逐符号一致；三条简化假设 $m=n$、$k=v=d/h$、$n\le d$ 一致。“$bhmk=bn^2$”的笔误确实印在 §2.4.1。
6. **Shazeer Table 1–3**（C31、C32、图 4.9）：46→3.8 μs、编码器 1.7/1.5、训练 13.2/13.0、beam-4 203/32、BLEU 26.7/26.5/26.6/26.5、减头变体 25.8/26.2/26.1/25.8、FFN 5440/9088、211M/192M 参数，全部一致。增量 MQA 代码 `expand_dims(..., axis=2)` 与 `concat(axis=2)` 的形状错误确认存在。
7. **GQA**（C38–C43、C41）：“5% of original pre-training compute”（摘要）、“approximately 600 TPUv3 chip-days”、“8 groups as a favorable middle ground”、均值池化、Rabe (2023) 一句（§4 Related Work）均一致；Table 1 四行（0.37/46.0、1.51/47.2、0.24/46.6、0.28/47.1）一致。
8. **Leviathan 接受规则与 Algorithm 1**（C61、§2.8）：§2.3 原句与 Algorithm 1 中 $n\leftarrow\min(\{i-1\mid1\le i\le\gamma,r_i>\frac{p_i(x)}{q_i(x)}\}\cup\{\gamma\})$、$p'\leftarrow norm(max(0,p_{n+1}-q_{n+1}))$ 逐符号一致；附录 A.1 两行证明一致。
9. **Leviathan 式 (1)、Def 3.2、Lemma 3.3、Thm 3.5、Cor 3.6、Thm 3.8、Cor 3.9、Thm 3.11**：逐符号一致；Lemma 3.3 的证明中间步 $\sum_x\frac{|p-q|}2$ 支持“$D_{LK}$ = TV”的说法。§3.3 “always less than 0.05” 的限定语（“a couple of orders of magnitude smaller”）一致；Table 4 的 $c$=0.02/0.04/0.11 一致；Table 2 的 3.4X/2.6X/3.1X/2.3X 一致；bigram $\alpha\approx0.2$、1.25X 一致；Table 1 六组重算全部一致。
10. **Chen 的记号与结果**（C68–C70、E5）：§4.2 原文确为草稿 $\sim p$、以 $\min(1,q/p)$ 接受、残差 $(q-p)_+$，且明说 $q$ 为目标、$p$ 为草稿；Table 1 的 14.1/7.52/7.00/5.73 ms 与 1.92×/2.01×/2.46× 一致；Table 2 草稿 6144 维、48 头、8 层、4B 一致；“within hardware numerics”与 §6.1 “a single token deviating due to numerics could result in two sequences diverging” 一致；加速比按逐循环日志统计（§6.1 “we log the amount of tokens generated for each speculative loop”）一致。
11. **DeepSeek-V3 报告**（C4、C27、C71、C75）：§3.4 “separates the prefilling and decoding stages”、§3.4.1 “4 nodes with 32 GPUs”、§3.4.2 “40 nodes with 320 GPUs”“usually within 256 tokens”、§3.2.2 “NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)”、§5.4.3 “between 85% and 90%”“1.8 times TPS” 一致；报告全文确无 “400 GB/s” 或 “900 GB/s”（E12 成立）；§4.2 的 61 层、$n_h$=128、$d_h$=128、$d_c$=512、$d_h^R$=64、MTP 深度 $D$=1 一致。
12. **ISCA 2025（zhao2025）**（C9、C15、C74、C75、E13、E14）：§4.1 “reduced from 900 GB/s to 400 GB/s”、§4.3 “200GB/s … about 160GB/s”、§2.3.3 “80% to 90%”“1.8x”、§2.1.2 “GEMM to GEMV” 一致；Table 1 印 70.272/327.680/516.096 KB 与 4.66x/7.28x，**v1 与 v2 都印 7.28x**，重算 516,096/70,272 = 7.344（E13 成立）。v2 日期 2025-12-23 经 arXiv 提交历史核实。
13. **DeepSeek-V2 Table 1 与式 (7)(8)**（C10、C11、§2.2、§2.3）：$2n_hd_hl$、$2n_gd_hl$、$2d_hl$、$(d_c+d_h^R)l\approx\frac92d_hl$、Strong/Moderate/Weak/Stronger、“regardless of the storage precision”、$d_c=4d_h$、$d_h^R=d_h/2$、“GQA with only 2.25 groups”、$W^O\in\R^{d\times d_hn_h}$ 逐符号一致。
14. **DeepSeek LLM Table 2 与 config**（C47、E19）：7B 30/4096/32/32，67B 95/8192/64/8，4096 上下文；官方 config 一致（67B `intermediate_size` 22016、词表 102400）。
15. **config.json 全部重读**（§3.1、E18）：DeepSeek-V3 `num_key_value_heads`=128、`kv_lora_rank`=512、`qk_rope_head_dim`=64、61 层；Qwen2.5-72B 80/64/8；OPT-13B 40/5120/40/20480/2048；NousResearch 镜像 Llama-2-7B/70B、Llama-3-8B/70B 各项与表一致。
16. **Llama 3 Table 3 与 §3.2**（C46、§3.1）：405B 126 层、16,384、FFN 53,248、128 头、8 KV 头，词表 128,000；GQA 8 KV 头原句一致。
17. **Llama 2 A.2.1**（C45、E8）：150B token、30B、FFN ×1.33（MQA）/×1.3（GQA）、OOM 条件（256 上下文 batch 1024、2k 上下文 batch 128）、选 GQA 的原句一致；并确认“复制 KV 使其与 GQA 一样大”的说法出自此处。
18. **PagedAttention**（C3、C7、C8、C12、C48–C51、§2.9）：“800 KB”、“1.6 GB”、“Approximately 65%”、“20.4% - 38.2%”、默认 block size 16、2-4×、“Besides the computation in Eq. 4”（式 (3) 才是注意力，(4) 在 §4.1）一致。
19. **Williams Roofline TR**（C20、C21、E2）：下载 UCB/EECS-2008-134 PDF 核对：§3 “operational intensity to mean operations per byte of DRAM traffic”、拒用 arithmetic intensity / machine balance 的两个理由、[16] Harris 2005、min(…) 公式、ridge point 定义，逐字一致；TR 标题含 “Floating-Point Programs and”，日期 2008-10-17。
20. **FlashAttention**（C22、C24、§2.6 Theorem 2 / Prop. 3）：“arithmetic intensity [85] … arithmetic operations per byte of memory access”、附录 “Roofline model of arithmetic intensity [85]”、A100 40-80GB / 1.5-2.0TB/s / 192KB×108 / 约 19TB/s、Theorem 2 的 $\Theta(Nd+N^2)$ 与 $\Theta(N^2d^2M^{-1})$、Proposition 3 逐符号一致；[85] 为 CACM 52(4):65–76。
21. **FA3 的 989 TFLOPS**（C73、E10）：“989 TFLOPS of FP16 matmul”、1830 MHz、“80 GiB @ 3.35 TB/s”、740 TFLOPs/s（75%）一致。
22. **A100 数据表**（C72、E11）：下载 PDF 核对，BF16 312 | 624*（* With sparsity）、带宽 1,555/1,935/1,555/2,039 GB/s（PCIe 40/80、SXM 40/80）、NVLink 600GB/s、Jun21 版一致。
23. **Holtzman**（C53–C58、§2.7、E20）：式 (2)(3) 逐符号一致；式 (4) 分母确印作 $u_l^\prime$；§4.1 “762M”、Figure 1 图注 “774M” 矛盾确认；5,000 段、最长 200 token；Table 1 全部 11 行数值一致；“values of p are usually in [0.9,1)” 在 §5.2。
24. **其他出处与日期**：GPT-3 Table 2.1 13B 行 5140、175B 行 96/12288/96/128、$d_{ff}=4d$、$n_{ctx}$=2048；Kaplan Table 1 “Attention: Mask” $2n_{layer}n_{ctx}d_{attn}$、乘加系数 2、$n_{vocab}$=50257、$d_{model}>n_{ctx}/12$；Fan §5.4 $k$=10；PMLR 202:19274–19286、arXiv comment “ICML 2023 Oral”；ACL Anthology GQA 起始页 4895、Fan 起始页 889；各论文 arXiv v1 日期（FA3 除外，见下）。

### 10.2 本次修改

| 位置 | 原文 | 改为 | 理由与来源 |
|---|---|---|---|
| §4.8 “Leviathan Table 4” 行 | 列 10 个值并称“一致” | 列全部 12 行；注明 9 行在一位小数上一致，第 5、6、12 行差 0.05–0.07 | 原稿漏了 CNNDM 最后两行，且 2.35→2.4、1.93→2.0 按常规舍入并不一致；第 12 行 1.54 vs 1.6。差异量级与表中 $\alpha$、$c$ 只印两位有关。来源：leviathan2023 附录 Table 4（v2），Python 重算 Theorem 3.8。研究代理报告中“Table 4 reproduces exactly”属夸大 |
| §4.3 “Pope prefill 每 token vs decode 每 token” 行；图 4.1 说明 | 只写“差 201 倍” | 加口径说明：28.4 ms 是 batch 64 下每步（每序列每 token）延迟；按吞吐算 0.444 ms/token，只差约 3.1 倍 | 防止相邻论断替换：Pope Table 2 的 decode batch 是 64、prefill batch 是 1，“201 倍”只在单序列延迟口径下成立。来源：pope2023 Table 2 |
| §4.2 最后一列 | 25,703 / 420,990 / 430,603 / 346,551 | 25,704 / 420,992 / 430,618 / 346,543 | 原稿用四舍五入后的 GB 数相除；按精确参数量重算。差异 <0.01%，结论不变 |
| E8 | “batch 小于分片数时无效（附录 D：batch < 4 无加速）” | 分别归属：Llama 2 A.2.1 说只在 batch 大于分片数时可用；Pope 附录 D 说 batch < 4（TPU v4 torus 一个轴的最小长度）无加速 | 原句把 Llama 2 的说法挂在 Pope 名下。来源：touvron2023b A.2.1；pope2023 附录 D |
| E4(d) | “§4.2 ‘preformed’” | “§4.1 ‘preformed’” | 该拼写错误出现在 §4.1 Experimental Setup 的 LM 实验段落。来源：shazeer2019 全文 |
| §0.1 shah2024 日期 | 2024-07-12 | 2024-07-11 | arXiv 提交历史：v1 Thu, 11 Jul 2024 15:44:48 UTC |

### 10.3 复核后仍存疑（补充 §8）

- **H100 产品页**这次只经 WebFetch 的摘要模型读到（BF16 1,979、TF32 989 带 “*With sparsity” 脚注、3.35TB/s、900GB/s；NVL 1,671、94GB、3.9TB/s、600GB/s），没有拿到原始 HTML 逐字核对每一格的星号。数值与原稿一致，但“BF16 那一格带星号”这一点仍依赖原稿的读取。
- **FlashAttention、GPT-3 的 NeurIPS 会议信息**仍未到会议页核对（§8 第 12、23 条不变）。
- 研究代理列出的其余 UNVERIFIED 条目（H800 带宽/算力、PagedAttention Fig. 2 逐系统数值、Llama 官方 config、Orca 全文等）本次未能补核，维持 §8 原状。

---

## 补充核实（写课时追加，2026-09-24）

写第 4 课正文时，为了几处研究笔记没有覆盖的细节，回到本机已下载的论文全文（arXiv LaTeXML 文本）逐字核对。引文均不超过约 15 个英文词。

### S1 GQA 论文（ainslie2023，2305.13245v3）

1. **评测任务**：摘要 5 个（CNN/Daily Mail、arXiv、PubMed、MediaSum、Multi-News）、翻译 WMT 2014 英→德、问答 TriviaQA，共 7 个数据集；Table 1 与 Figure 3 的“平均分”是在这 7 个上的平均。
   > “We evaluate on summarization datasets CNN/Daily Mail …” — §3.1 Experimental setup（截短）。课文用于图 4.11 的说明文字。
2. **GQA/MQA 用在哪些注意力层**：只用于解码器的自注意力与交叉注意力，不用于编码器自注意力。
   > “We apply MQA and GQA to decoder self-attention and cross-attention” — §3.1。
3. **转换后不续训的表现（Figure 5 正文）**：GQA 转换后即有不错表现，MQA 需要续训才可用；两者从 5% 续训获益，10% 收益递减。
   > “GQA already achieves reasonable performance after conversion while MQA requires uptraining” — §3.3。
4. **为什么大模型更适合 GQA（Figure 6 正文）**：大模型里 KV cache 的带宽开销相对不那么紧，而头数多使 MQA 的缩减更猛；组数从 1 增加起初只带来小幅变慢。
   > “For larger models the memory bandwidth overhead from the KV cache is less constraining” — §3.3。
5. **MQA 训练不稳定（附录）**：MQA 可能导致微调不稳定，长输入任务尤甚；从头训练的 T5-Large MQA 预训练频繁损失尖峰、微调长输入任务时立刻发散；uptraining 的 MQA 较稳但方差大；GQA 看起来稳定。
   > “multi-query attention can lead to training instability during fine-tuning” — 附录 A。
   课文 §5 的“失败模式：MQA 的训练不稳定”依据此条。

### S2 Shazeer 2019（1911.02150v1）Table 1 的对照组

减头 / 减维的四个 MHA 变体为 $(h,\ d_k=d_v)=(1,128),(2,64),(4,32),(8,16)$，FFN 都加宽到 6784 以对齐参数，dev BLEU 分别为 25.8、26.2、26.1、25.8；四者的 $h\,d_k$ 都等于 128，即每层 K（及 V）的宽度与 MQA（$1\times128$）相同。MQA 的 FFN 为 5440。
> 表格原样：“multi-head 1 128 6784 1.518 25.8 … multi-head 8 16 6784 1.513 25.8” — Table 1。
课文把这四个变体称为“KV 同样缩小 8 倍的其它做法”，是笔者按 $h\,d_k=128$ 得出的解读。

### S3 Fan 等人 2018（1805.04833v1）§5.4 对 top-k 的理由

从最可能的 10 个候选采样“比 beam search 有效得多”，后者倾向产生常见短语与重复文本；完全随机采样会引入极低概率的词。
> “substantially more effective than beam search, which tends to produce common phrases” — §5.4（截短）。

### S4 Holtzman 等人 2020（1904.09751v2）HUSE 的含义

HUSE 是 Human Unified with Statistical Evaluation（Hashimoto et al., 2019），论文 Table 1 注称其为“人工与统计结合的评估”，数值越高越好。
> “a combined human and statistical evaluation” — Table 1 表注。

### S5 本课自己复算、研究笔记里没有的数（Python 复算，脚本在 `deepseek/figures/gen/`）

| 数 | 表达式 | 值 | 用在 |
|---|---|---|---|
| Llama-3-8B 的 $N_{mm}$ | $32\times(2\cdot4096^2+2\cdot4096\cdot1024+3\cdot4096\cdot14336)+128256\cdot4096$ | $7.505\times10^9$ | 练习 6 |
| Llama-2-7B 注意力项追上权重项的位置 | $N_{mm}/(2ln_hd_h)=6.607\times10^9/262144$ | 约 25,200 | §3 警告框（超出原生上下文，外推） |
| 缓存 $\mathbf h$ 而非 K、V 时每步重算 K、V 的代价（Llama-2-7B，$t=4096$） | $4d^2t\,l=4\cdot4096^2\cdot4096\cdot32$ | $8.80\times10^{12}$，约为有缓存一步的 570 倍 | §2 旁注（笔者算账） |
| LLaMA-3.1 405B 若用 MHA 的每 token KV | $2\cdot128\cdot128\cdot126\cdot2$ | 8,257,536 B（GQA 的 16 倍） | 图 4.6 |
| 方阵 4096 矩阵乘的算术强度（bf16） | $2\cdot4096^3/(3\cdot4096^2\cdot2)$ | 约 1365 FLOP/B | §4 例题 |
| 投机采样玩具例子的蒙特卡洛 | numpy `default_rng(0)`，200,000 次 | 频率 (0.501, 0.300, 0.198)，接受率 0.699 | 图 4.14、§7 例题（研究笔记 §4.8 的另一次模拟为 0.502/0.300/0.198、0.698，量级一致） |
| Pope 等人 PaLM 540B prefill 与 decode 的时间占比 | 0.29/(0.29+1.82)；85.2/(85.2+6.0) | 14% / 86%；93% / 7% | 图 4.1 |
| GPT-3 参数量的拆分（数学复核，2026-09-24） | $96\cdot12\cdot12288^2+50257\cdot12288$ | 174,563,733,504 ≈ 174.56B（bf16 349.13 GB）；另有位置嵌入 $2048\cdot12288$ 与偏置、LN 合计约 0.04B，含上它们为 174.60B | §1 例题、§3 表（原稿写“174.6B”却配 349.1 GB，改印 174.56B 使 $2\times$ 可复核） |
| 输出层只算最后一个位置时少掉的运算（Llama-2-7B，$t$=4096） | $2Vd(t-1)=2\cdot32000\cdot4096\cdot4095$ | $1.07\times10^{12}$，占 $C_{fwd}(4096)=5.852\times10^{13}$ 的 1.8% | (4.9)、(4.11) 的口径说明 |
| 从头生成 4096 个 token 的比值 1997 为何小于 $T/2$ | 权重部分 $(T+1)/2$=2048.5；注意力部分 $(T+2)/3$=1366；按有缓存时份额 92.48% / 7.52% 加权 | 1997.2 | 练习 4(b) |
| (4.9) 二次项在 $t\le4096$ 的最大占比 | $ln_hd_h(t+1)/(N_{mm}+ln_hd_h(t+1))$ 在 $t$=4096 | 7.5% | 图 4.5 说明文字 |
| Pope 表 1 复算的每 token 字节 | MHA $2\cdot64\cdot128\cdot118\cdot2$；复制的 MQA $64\cdot(2\cdot256\cdot118\cdot2)$；按 batch 的 MQA $2\cdot256\cdot118\cdot2$ | 3,866,624 / 7,733,248 / 120,832 B | §5 正文（补上 MQA 的 256 维这一前提） |

### S6 课文里标为“笔者推测 / 笔者推理”的论断

- 投机解码一次验证 $\gamma+1$ 个位置，相当于把权重部分的强度从 1 提到约 $\gamma+1$（按 (4.18) 的类推）。
- batch 越大，decode 越接近脊点，投机解码可利用的闲置算力越少。
- 位置被重新编号会让缓存静默出错（依“位置编码只看自己的位置”这一条件推出，同研究笔记 §2.2 第 4 条）。
- MTP 的 1.8 倍 TPS 低于 $1+\alpha$ 的差额来自 MTP 模块开销与实现开销（同研究笔记练习 9）。
- （事实审校时补标）uptraining 的 MQA 训练不稳定给了“选 GQA 而不选 MQA”又一个理由——Ainslie 等人只报告现象，没有下这个结论。
- （事实审校时补标）prefill 受算力限制、decode 受带宽限制，所以 PD 分离能让两者各按自己的瓶颈配置——V3 报告自己给的理由是 SLO 与吞吐，见 S7 第 3 条。

### S7 事实审校（第二轮，2026-09-24）

逐条回到一手来源核对了课文里研究笔记没有逐字覆盖、或者措辞比笔记更强的几处。引文不超过约 15 个英文词。

1. **Llama 2 选 GQA 的理由有两个，不只是张量并行。** touvron2023b 附录 A.2.1：“based on the ablation results and ease of scaling inference, for the 34B and 70B Llama 2 models we chose to use GQA”（截短）。张量并行设定：“we host our largest models using 8 A100s in a single node with tensor parallelism”；MQA 的两种出路：复制 KV（“making the KV cache size equal to GQA”）或按 batch 切分，后者“works only when batch sizes are larger than the number of shards”并会让推理服务变复杂。课文原写“正是因此放弃 MQA”，已改为“理由之一 + 消融结果”，并补上按 batch 切分的前提。
2. **Pope Table 2 的芯片数。** 全文表格：低延迟与高吞吐两种配置的 prefill、decode 都是 64 chips；batch 1/64/512/512、int8/int8/bfloat16/bfloat16、MFU 43%/14%/76%/33%、0.29 s/1.82 s/85.2 s/6.0 s，与 C78 一致。表题：“Prefill latency is for processing 2048 tokens; decode latency is for generating 64 tokens.”（pope2023，Table 2）
3. **V3 报告对 PD 分离给出的理由。** “To simultaneously ensure both the Service-Level Objective (SLO) for online services and high throughput”（deepseek2024v3 §3.4，截短）。报告没有用“prefill 受算力限制、decode 受带宽限制”来解释分离；课文把这层联系改标为笔者的解读。
4. **Orca 的摘要（USENIX 页面，WebFetch 读取）。** “schedules execution at the granularity of iteration (instead of request)”；问题陈述：批内先结束的请求不能返回，新到的请求要等整批结束。**逐次迭代移出已完成请求、加入新请求**这一句在 PagedAttention §2.3 有原文：“After each iteration, completed requests are removed from the batch, and new ones are added.”——课文这一句现同时引 yu2022 与 kwon2023。
5. **PagedAttention 对“连续预留”的描述。** “they pre-allocate a contiguous chunk of memory with the request’s maximum length (e.g., 2048 tokens)”（kwon2023 §1），支持课文“按最大长度为每个请求预留连续空间”。
6. **Leviathan 的墙钟分析假设。** “we can run γ+1 concurrent evaluations of M_p in parallel without increasing the walltime”（leviathan2023 §3.3），支持课文“Leviathan 等人的定理也建立在这个假设上”。$D_{LK}$ 的原话是“We start by defining a natural divergence D_LK”（§3.2）——论文定义的是 $D_{LK}$，不是“把 $1-\beta$ 称为”；课文措辞已相应调整。
7. **Ainslie 对转换方法消融的解释。** “Intuitively, results are ordered by the degree to which information is preserved from the pre-trained model.”（ainslie2023 §3.3），支持课文“顺序正好是保留原模型信息的多少”。
8. **课文里对第 18、19 课的预告**（“V4 系列把缓存沿序列压缩、用 FP4 存储并在层间复用”“第 19 课的 FP4 KV”）与 `lesson-18-doc.md` C21（CSA/HCA 沿序列压缩 KV）、`lesson-19-doc.md` C27（FP4 主 KV 缓存）与 §1C（CSA2 跨层共享 KV）一致。
9. **仍未核实**：FlashAttention（NeurIPS 2022）与 GPT-3（NeurIPS 2020）的会议信息——本次 `arxiv.py meta` 访问 export.arxiv.org 超时，未能读取 journal_ref；课文的参考文献卡片本来就没有写会议，不受影响。
