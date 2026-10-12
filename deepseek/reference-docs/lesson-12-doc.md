# 第 12 课研究笔记：DeepSeek-V3（下）——FP8 训练、DualPipe、跨节点通信、推理部署与“557.6 万美元”

> 研究日期 2026-09-24。本文件是 `lesson-12-v3-infrastructure` 的唯一数字来源。正文里任何数字、公式、超参数都必须能在这里找到出处。
> 核对方法：arXiv LaTeXML 全文（`arxiv.py text/grep`）逐句检索；v1/v2 逐句 diff；GitHub README 与源码用 raw 地址下载后 grep；
> Hugging Face 的 `config.json`、`model.safetensors.index.json` 与 HF API 的参数统计；所有派生数字用 Python 复算（脚本见 §4 末尾）。
> 引文保持英文原文，尽量只截取承载事实的短句；其余用中文转述。标 **【笔者推断】** 的是本笔记自己的推理，不是论文原话。

---

## 0. 覆盖范围与所读版本

### 0.1 论文

| 短键 | 标题 | 作者 | arXiv / 版本 | 状态 / 出处 | 日期 |
|---|---|---|---|---|---|
| `v3` | DeepSeek-V3 Technical Report | DeepSeek-AI（Aixin Liu 等约 200 人） | 2412.19437，**读 v2**（2025-02-18），并与 **v1**（2024-12-27）逐句 diff | 技术报告（arXiv 预印本，未经同行评审） | v1 2024-12-27 / v2 2025-02-18 |
| `insights` | Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures | Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Huazuo Gao, Jiashi Li, Liyue Zhang, Panpan Huang, Shangyan Zhou, Shirong Ma, Wenfeng Liang, Ying He, Yuqing Wang, Yuxuan Liu, Y.X. Wei（按名字字母序） | 2505.09343，**读 v2**（2025-12-23），并与 **v1**（2025-05-14）逐句 diff | **ISCA '25 Industry Track**（第 52 届 ISCA，2025-06-21–25，东京），DOI 10.1145/3695053.3731412 | v1 2025-05-14 / v2 2025-12-23 |
| `fp8fmt` | FP8 Formats for Deep Learning | Micikevicius, Stosic, Burgess, Cornea, Dubey, Grisenthwaite 等 15 人 | 2209.05433，**读 v2**（2022-09-29） | 预印本 | v1 2022-09-12 |
| `zb` | Zero Bubble Pipeline Parallelism | Penghui Qi, Xinyi Wan, Guangxing Huang, Min Lin | 2401.10241，**读 v1**（2023-11-30） | arXiv 预印本；会议出处本次未核实（见 §8） | 2023-11-30 |
| `mx` | Microscaling Data Formats for Deep Learning | Rouhani, Zhao, More, Hall, Khodamoradi, Deng 等 | 2310.10537，读 v3（仅 grep 块大小与缩放格式） | 预印本 | 2023-10-16 |
| `zero` | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari, Rasley, Ruwase, He | 1910.02054 v3（仅 grep 显存记账 16Ψ 与 P_os） | 预印本（SC20 版本未核） | 2019-10-04 |
| `chimera` | Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines | Shigang Li, Torsten Hoefler | 2107.06925 v5（只读元数据） | SC'21（arXiv comment 所述，Best Paper Finalist） | 2021-07-14 |
| `pipedream` | PipeDream: Fast and Efficient Pipeline Parallel DNN Training | Harlap, Narayanan, Phanishayee 等 | 1806.03377 v1（只读元数据） | 预印本 | 2018-06-08 |
| `fireflyer` | Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning | Wei An 等 52 人 | 2408.14158 v2（仅 grep “HAI”） | SC24（arXiv comment 所述） | 2024-08-26 |

**版本漂移摘要（`v3`）**：第 3 节在 v1 与 v2 之间**只有 §3.5.2 第一段（Tensor Core 累加精度）被改写**，外加 §3.5.1 一个错字（v1 “entirely -utilized” → v2 “entirely under-utilized”）。表 1、表 2、§3.2–§3.4 的所有数字两版一致。
**版本漂移摘要（`insights`）**：v2 改动集中在措辞与引用：§2.3.2 增加 “Network latency is not included in this calculation.”，NVL72 那段把 v1 的 “theoretical upper limit of over 0.82 ms TPOT” 改成 “0.82 ms TPOT”，前提从 v1 的 “Assuming the computation time is equal to the communication time” 改成 “Assuming perfect overlap between computation and communication”，并补充“小 batch 时 GPU 效率大幅下降”的免责；§3.1.1 给 FP22 补了引用 (Zhang et al., 2025) 与说明句；§4.2 “reduce latency and improve TPOT” → “improve TTFT and TPOT”；§6.4 RAR 机制重写。**表 1–5 的数字两版一致**（包括下文 §7 指出的 7.28x 错误）。

### 0.2 制品（artefact，无论文；2026-09-24 从 `main` 分支下载）

| 短键 | 内容 | 读了什么 |
|---|---|---|
| `gh-dualpipe` | github.com/deepseek-ai/DualPipe | README（含 DualPipeV 与气泡/显存表）、`dualpipe/dualpipe.py` 的 `step()` 全部 8 个阶段 |
| `gh-deepep` | github.com/deepseek-ai/DeepEP | 现行 README（**V2**，NCCL Gin 后端）与 `docs/legacy.md`（**V1**，2025-02 开源时的性能表） |
| `gh-deepgemm` | github.com/deepseek-ai/DeepGEMM | 现行 README（已扩展到 FP4、Mega MoE、mHC 等，News 到 2026.09.10） |
| `gh-flashmla` | github.com/deepseek-ai/FlashMLA | 现行 README（News 到 2026.09.10 的 V4.1 内核） |
| `gh-eplb` | github.com/deepseek-ai/EPLB | README |
| `gh-3fs` | github.com/deepseek-ai/3FS | README 性能节 |
| `gh-profile` | github.com/deepseek-ai/profile-data | README |
| `gh-infra` | github.com/deepseek-ai/open-infra-index | README、`202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md`、`OpenSourcing_DeepSeek_Inference_Engine/README.md` |
| `gh-v3` | github.com/deepseek-ai/DeepSeek-V3 | README、README_WEIGHTS.md、`inference/kernel.py`、`inference/fp8_cast_bf16.py` |
| `hf-v3` | huggingface.co/deepseek-ai/DeepSeek-V3 | `config.json`、`model.safetensors.index.json`、HF API `?blobs=true`（文件大小与按 dtype 的参数计数） |
| `hf-cmp` | Qwen/Qwen2.5-72B `config.json`；Llama-3.1-405B `config.json`（meta-llama 原库需登录，401；改读 RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8 与 unsloth 镜像，两者一致） | 层数、KV 头数、hidden size |

### 0.3 看过的图

`v3` 图 4（overlap.svg，用无头 Chrome 渲染）、图 5（dualpipe.svg）、图 6（fp8-frameworkv3.png）、图 7（fp8-128accumulatorv4.png）、图 10（fp8-v.s.-bf16.png）；`insights` 图 1（basic_arch.png，带精度标注）、图 2（h800_arch.svg）、图 3（mpft.png）、图 7（ep_kernel.svg）。图片只存在于 scratch 目录，没有放进仓库。

---

## 1. 核心论断与原文引文

编号 C1–C60。位置写成“论文短键 §节 / 表 / 图”，版本默认 `v3`=v2、`insights`=v2。

### 1.1 训练成本（`v3` §1 表 1 及其后段落）

- **C1 表 1 的拆分。** 预训练 2664K、上下文扩展 119K、后训练 5K，共 2788K H800 GPU 小时；美元列 $5.328M / $0.238M / $0.01M / $5.576M。表题：“Training costs of DeepSeek-V3, assuming the rental price of H800 is $2 per GPU hour.”（`v3` 表 1）
- **C2 每万亿 token 的算力。** “each trillion tokens requires only 180K H800 GPU hours, i.e., 3.7 days on our cluster with 2048 H800 GPUs”（`v3` §1，表 1 下第一段）。
- **C3 预训练时长。** “our pre-training stage is completed in less than two months and costs 2664K GPU hours.”（同段）
- **C4 这个数字不包括什么。** “excluding the costs associated with prior research and ablation experiments on architectures, algorithms, or data.”（同段末句）。论文**只**明确排除了“先前研究与消融实验”；人员、集群购置、数据采集、R1 系列模型本身的训练等**论文没有提**（见 §7 E9）。
- **C5 预训练之后。** “The subsequent training stages after pre-training require only 0.1M GPU hours.”（`v3` §1 贡献列表）——119K+5K=124K，四舍五入为 0.1M。
- **C6 训练稳定性。** “we did not encounter any irrecoverable loss spikes or have to roll back.”（`v3` §1）

### 1.2 集群与训练框架（`v3` §3.1–§3.2）

- **C7 集群。** “DeepSeek-V3 is trained on a cluster equipped with 2048 NVIDIA H800 GPUs.” 节点内 “8 GPUs connected by NVLink and NVSwitch within nodes”，节点间 “InfiniBand (IB) interconnects are utilized”（`v3` §3.1）。
- **C8 框架。** HAI-LLM：“an efficient and lightweight training framework crafted by our engineers from the ground up.”（`v3` §3.2）。公开论文/代码未找到（§8）。另：`gh-infra` 的推理引擎说明写 “Our training framework relies on PyTorch, and our inference engine is built upon vLLM”，且推理引擎是 “an early fork of vLLM”。
- **C9 并行方式。** “16-way Pipeline Parallelism (PP) …, 64-way Expert Parallelism (EP) … spanning 8 nodes, and ZeRO-1 Data Parallelism (DP)”（`v3` §3.2；省略号处是引用）。
- **C10 不用 TP。** “enabling us to train DeepSeek-V3 without using costly Tensor Parallelism (TP).”（`v3` §3.2）；原因见 `insights` §4.2：“Tensor Parallelism is avoided during training due to its inefficiency under limited NVLink bandwidth.”
- **C11 专家摆放。** “the routed experts will be uniformly deployed on 64 GPUs belonging to 8 nodes.”（`v3` §4.2）——256/64 = 每 GPU 每层 4 个路由专家。
- **C12 批大小调度。** “the batch size is gradually increased from 3072 to 15360 in the training of the first 469B tokens, and then keeps 15360”（`v3` §4.2），序列长 4K（“maximum sequence length to 4K during pre-training”）。

### 1.3 DualPipe（`v3` §3.2.1，图 4、图 5、表 2）

- **C13 问题。** 跨节点专家并行带来 “an inefficient computation-to-communication ratio of approximately 1:1.”（`v3` §3.2.1）
- **C14 核心想法。** “overlap the computation and communication within a pair of individual forward and backward chunks.” 每个 chunk 分成四块：“attention, all-to-all dispatch, MLP, and all-to-all combine.” 反向 chunk 的 attention 与 MLP 再拆成 “backward for input and backward for weights, like in ZeroBubble”，另有一个 PP 通信组件。
- **C15 手动调 SM 比例。** “manually adjust the ratio of GPU SMs dedicated to communication versus computation.”；效果 “both all-to-all and PP communication can be fully hidden”。
- **C16 双向调度。** “feeds micro-batches from both ends of the pipeline simultaneously”（`v3` §3.2.1，图 5 前）。
- **C17 规模化论断。** 只要保持计算/通信比恒定，就能 “employ fine-grained experts across nodes while achieving a near-zero all-to-all communication overhead.”（`v3` §3.2.1；§1 也重复一次）
- **C18 表 2 的显存代价。** “only increasing the peak activation memory by $\frac{1}{PP}$ times.” 以及 “DualPipe requires keeping two copies of the model parameters, this does not significantly increase the memory consumption since we use a large EP size”（`v3` §3.2.1）。
- **C19 与 Chimera 比。** “only requires that the pipeline stages and micro-batches be divisible by 2, without requiring micro-batches to be divisible by pipeline stages.”（同段）。**代码更严**：`dualpipe.py` 断言 `num_chunks >= num_ranks * 2`（§7 E7）。
- **C20 不随 micro-batch 数增长。** “neither the bubbles nor activation memory will increase as the number of micro-batches grows.”（同段）——本笔记的仿真验证了这一点（§4.6）。
- **C21 图 4 的组件顺序**（读自渲染后的 overlap.svg）：计算行 `MLP(B)▲ → MLP(W)▲ → MLP(F)△ → |屏障| → ATTN(B)▲ → ATTN(W)▲ → ATTN(F)△ → |屏障|`；通信行 `DISPATCH(F)△ → DISPATCH(B)▲ → |屏障| → COMBINE(F)△ → PP → COMBINE(B)▲ → |屏障|`。△=前向 chunk，▲=反向 chunk。图题：“Both all-to-all and PP communication can be fully hidden.”
- **C22 图 5。** “Example DualPipe scheduling for 8 PP ranks and 20 micro-batches in two directions.” 反向那一路的 micro-batch 不标号；“Two cells enclosed by a shared black border have mutually overlapped computation and communication.”
- **C23 训练 profile 的切分。** `gh-profile`：“Each chunk contains 4 MoE (Mixture of Experts) layers.” 配置 “EP64, TP1 with 4K sequence length”，且 “the PP communication is not included during profiling”，并且 “we simulate an absolutely balanced MoE routing strategy for profiling.”

### 1.4 跨节点 all-to-all（`v3` §3.2.2）

- **C24 带宽。** “NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s).”
- **C25 节点受限。** “we limit each token to be dispatched to at most 4 nodes, thereby reducing IB traffic.” 路径：先经 IB 发到目标节点上 “the GPUs with the same in-node index”，再经 NVLink 转发到持有目标专家的 GPU。
- **C26 3.2 与 13。** 每个 token “can efficiently select an average of 3.2 experts per node without incurring additional overhead from NVLink”，因此可扩到 “a maximum of 13 experts (4 nodes × 3.2 experts/node)”（实际只选 8 个）。
- **C27 20 个 SM。** “only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink.” 并且 “partition 20 SMs into 10 communication channels”。
- **C28 warp 专门化。** dispatch 时 “(1) IB sending, (2) IB-to-NVLink forwarding, and (3) NVLink receiving are handled by respective warps”；combine 时对应 NVLink 发送、NVLink→IB 转发与累加、IB 接收与累加。warp 数按负载动态调整；用自定义 PTX 并自动调通信块大小以减少 L2 占用。
- **C29 占总 SM 的比例。** “we allocate 20 out of the 132 SMs available in the H800 GPU for this purpose”（`v3` §3.5.1）。
- **C30 `insights` 的补充说法。** 分组：“we group 256 routed experts into 8 groups, with 32 experts per group, and deploy each group on a single node.”；IB 去重：目标专家分布在 $M$ 个节点时 IB 代价从 $8t$ 降到 $Mt$（`insights` §4.3）。带宽比 “approximately 4:1”：“NVLink provides 200GB/s bandwidth (of which about 160GB/s can actually be achieved)”，IB 按 “40GB/s for effective bandwidth” 计（`insights` §4.3）。

### 1.5 省显存技巧（`v3` §3.2.3）

- **C31 重算。** “We recompute all RMSNorm operations and MLA up-projections during back-propagation”。
- **C32 EMA 放 CPU。** “The EMA parameters are stored in CPU memory and are updated asynchronously after each training step.”
- **C33 MTP 共享嵌入/输出头。** “we deploy the shallowest layers (including the embedding layer) and deepest layers (including the output head) of the model on the same PP rank.”——这是 DualPipe 双向摆放的副产品：rank 0 同时持有第一段与最后一段（`dualpipe.py` 中每个 rank 有 `module[0]`、`module[1]` 两段）。

### 1.6 FP8 混合精度框架（`v3` §3.3、§3.3.1，图 6）

- **C34 首次验证。** “for the first time, validate its effectiveness on an extremely large-scale model.”（`v3` §1）；`insights` §3.1：“prior to DeepSeek-V3, there were no open-source large models leveraging FP8 for training.”
- **C35 细粒度分组。** “tile-wise grouping with $1\times N_c$ elements or block-wise grouping with $N_c\times N_c$ elements.”（`v3` §3.3 引言；注意此处印的是小写 $N_c$，§3.3.2 与图 7 是 $N_C$）。
- **C36 精度结论。** “the relative loss error of our FP8-training model remains consistently below 0.25%”，验证规模 “two model scales similar to DeepSeek-V2-Lite and DeepSeek-V2, training for approximately 1 trillion tokens”（`v3` §3.3）。附录 B.1 具体为 “approximately 16B total parameters on 1.33T tokens” 与 “approximately 230B total parameters on around 0.9T tokens”。
- **C37 三个 GEMM 全 FP8。** “all three GEMMs associated with the Linear operator, namely Fprop (forward pass), Dgrad (activation backward pass), and Wgrad (weight backward pass), are executed in FP8.” GEMM 输入 FP8，输出 “in BF16 or FP32”。速度：“theoretically doubles the computational speed compared with the original BF16 method.”
- **C38 保持高精度的组件。** “the embedding module, the output head, MoE gating modules, normalization operators, and attention operators.” 另外 “we store the master weights, weight gradients, and optimizer states in higher precision.”（`v3` §3.3.1）
- **C39 图 6 的数据类型标注**（读图）：Input BF16 →“To FP8”→ Fprop（Σ 在 FP32）→“To BF16”→ Output；Output Gradient BF16 →“To FP8”→ Dgrad（Σ FP32）→“To BF16”→ Input Gradient；Wgrad 的两个输入都 “To FP8”，Σ FP32 → Weight Gradient **FP32** →“To BF16”→ Optimizer States →“To FP32”→ Master Weight →“To FP8”→ Weight。图题：“For clarification, only the Linear operator is illustrated.”
- **C40 `insights` 图 1 的精度地图**（读图）：Embedding、Output Head 为 BF16；Cross-Entropy Loss、RMSNorm、Router、RoPE 为 FP32；Transformer Block / FFN / 专家 “FP8 Mixed Precision”；Multi-Head Attention（核心注意力）BF16；MTP 的 Linear Projection BF16；Dispatch all-to-all 标 FP8，Combine 标 “BF16/LogFMT”。图题：“All components take inputs and outputs in BF16.”

### 1.7 量化与乘法精度（`v3` §3.3.2，图 7）

- **C41 细粒度量化。** “for activations, we group and scale elements on a 1x128 tile basis (i.e., per token per 128 channels)”；“for weights, we group and scale elements on a 128x128 block basis (i.e., per 128 input channels per 128 output channels).”
- **C42 沿内维的缩放因子。** “per-group scaling factors along the inner dimension of GEMM operations. This functionality is not directly supported in the standard FP8 GEMM.”
- **C43 与 microscaling 的关系。** “highly consistent with the idea of microscaling formats”，并提到 Blackwell 宣布支持更小粒度的 microscaling（`v3` §3.3.2）。`mx` 的具体定义：“All concrete MX formats use E8M0 (an 8-bit exponent) as the format for the shared scale.”，块大小 $k=32$（`mx` 表 1）。
- **C44 累加精度问题。** “the accumulation precision of FP8 GEMM on NVIDIA H800 GPUs is limited to retaining around 14 bits”；实验：“with K = 4096 … the limited accumulation precision in Tensor Cores results in a maximum relative error of nearly 2%.”（`v3` §3.3.2）
- **C45 提升到 CUDA core。** “Once an interval of $N_C$ is reached, these partial results will be copied to FP32 registers on CUDA Cores, where full-precision FP32 accumulation is performed.” 缩放因子也在这里乘上（反量化）。两个 warpgroup 交替：一个做 promotion，另一个做 MMA。
- **C46 $N_C=128$。** “setting $N_C=128$ elements, equivalent to 4 WGMMAs, represents the minimal accumulation interval that can significantly improve precision without introducing substantial overhead.” 图 7(b) 画的是 WGMMA 1 … WGMMA 4 在 Tensor Core 里低精度累加，然后按 “$N_C$ Interval” 送到 CUDA core 的 FP32 寄存器。
- **C47 全用 E4M3。** 与 “hybrid FP8 format … which uses E4M3 … in Fprop and E5M2 … in Dgrad and Wgrad” 不同，“we adopt the E4M3 format on all tensors for higher precision.” 可行性归因于细粒度分组：“effectively shares exponent bits among these grouped elements”。
- **C48 在线量化。** 张量级框架用 delayed quantization（维护历史 amax）；DeepSeek “calculate the maximum absolute value online for each 1x128 activation tile or 128x128 weight block.”
- **C49 块量化激活梯度会发散。** 附录 B.2：“block-wise quantization of activation gradients leads to model divergence on an MoE model comprising approximately 16B total parameters, trained for around 300B tokens.” 作者的解释是 “token-correlated outliers”（标注为 “We hypothesize”）。

### 1.8 低精度存储与通信（`v3` §3.3.3）

- **C50 优化器状态。** “We adopt the BF16 data format instead of FP32 to track the first and second moments in the AdamW … optimizer”；但 “the master weights (stored by the optimizer) and gradients (used for batch size accumulation) are still retained in FP32”。
- **C51 两个特殊激活。** (1) 注意力之后那个 Linear 的输入：“We adopt a customized E5M6 data format exclusively for these activations.”，反向时要从 1x128 tile 转成 128x1 tile，为避免额外误差 “all the scaling factors are round scaled, i.e., integral power of 2.” (2) MoE 中 SwiGLU 的输入：“we cache the inputs of the SwiGLU operator and recompute its output in the backward pass”，以 FP8 细粒度量化存储。
- **C52 低精度通信。** MoE up-projection 之前的激活先量化成 FP8 再 dispatch（缩放因子同样取 2 的整数幂）；MoE down-projection 之前的激活梯度同样处理；“For both the forward and backward combine components, we retain them in BF16”。`insights` §3.2：dispatch 用 FP8 “reducing communication volume by 50% compared to BF16”。

### 1.9 推理与部署（`v3` §3.4；`gh-infra` Day 6；`gh-profile`）

- **C53 prefill 单元。** “The minimum deployment unit of the prefilling stage consists of 4 nodes with 32 GPUs.” 注意力 “4-way Tensor Parallelism (TP4) with Sequence Parallelism (SP), combined with 8-way Data Parallelism (DP8)”；MoE “32-way Expert Parallelism (EP32)”；浅层稠密 MLP 用 “1-way Tensor Parallelism”。
- **C54 冗余专家。** 高负载专家按线上统计周期性调整 “(e.g., every 10 minutes)”；“we set 32 redundant experts for the prefilling stage.” 每 GPU “besides the original 8 experts it hosts, it will also host one additional redundant expert.” 节点内重排、不增加跨节点流量。两 micro-batch 重叠：一个的 attention+MoE 与另一个的 dispatch+combine 重叠。探索中：每 GPU 放 16 个专家、“only 9 will be activated”，每层 all-to-all 前在线求全局最优路由。
- **C55 decode 单元。** “we treat the shared expert as a routed one”（每 token 选 9 个，共享专家视为必选的高负载专家）。“The minimum deployment unit of the decoding stage consists of 40 nodes with 320 GPUs.” 注意力 “TP4 with SP, combined with DP80”，MoE “EP320”；“each GPU hosts only one expert, and 64 GPUs are responsible for hosting redundant experts and shared experts.” all-to-all 用 IB 点对点并用 IBGDA。
- **C56 decode 的瓶颈。** “the batch size per expert is relatively small (usually within 256 tokens), and the bottleneck is memory access rather than computation.” 因此给 dispatch+MoE+combine 只分少量 SM；两 micro-batch 时把一个的 attention 与另一个的 dispatch+MoE+combine 重叠。
- **C57 局限。** “the recommended deployment unit for DeepSeek-V3 is relatively large, which might pose a burden for small-sized teams.”；端到端生成速度 “more than two times that of DeepSeek-V2”（`v3` §6）。
- **C58 Day 6（2025-02，公司博文）。** “Prefilling Phase [Routed Expert EP32, MLA/Shared Expert DP32]: Each deployment unit spans 4 nodes with 32 redundant routed experts, where each GPU handles 9 routed experts and 1 shared expert.”；“Decoding Phase [Routed Expert EP144, MLA/Shared Expert DP144]: Each deployment unit spans 18 nodes with 32 redundant routed experts, where each GPU manages 2 routed experts and 1 shared expert.” decode 用 “a 5-stage pipeline”。精度：“matrix multiplications and dispatch transmissions adopt the FP8 format aligned with training, while core MLA computations and combine transmissions use the BF16 format”。24 小时统计（UTC+8 2025-02-27 12:00 至 02-28 12:00）：峰值 278 节点，平均 226.75 节点，日成本 $87,072；输入 608B（其中 342B，56.3% 命中磁盘 KV cache），输出 168B；每路输出 20–22 tok/s，平均每个输出 token 的 KV 长度 4,989；每节点 prefill ~73.7k tok/s、decode ~14.8k tok/s；若全部按 R1 价格计费 “the total daily revenue would be $562,027, with a cost profit margin of 545%.” 并自述实际收入 “substantially lower”（V3 更便宜、网页/APP 免费、夜间折扣）。**这是公司自报、按理论定价推算的数字。**
- **C59 profile-data 的推理配置。** prefill：“EP32 and TP1 (in line with DeepSeek V3/R1’s actual online deployment)”，prompt 4K，每 GPU 16K tokens；decode：“EP128, TP1, and a prompt length of 4K (closely matching the actual online deployment configuration)”，每 GPU 128 个请求；decode 时 all-to-all “does not occupy GPU SMs”。

### 1.10 硬件建议（`v3` §3.5；`insights` §3–§6）

- **C60 汇总。** `v3` §3.5.1：把通信任务（IB↔NVLink 转发与聚合、RDMA 缓冲搬运、combine 的 reduce、细粒度内存布局管理）从 SM 卸载到协处理器（类似 NVIDIA SHARP），并统一 IB（scale-out）与 NVLink（scale-up）。§3.5.2：(a) Tensor Core 提高 FP8 累加精度；(b) 原生支持 tile/block 量化（Tensor Core 接收缩放因子做 group scaling）——“Current GPUs only support per-tensor quantization”；(c) 支持在线量化：把 FP8 cast 与 TMA 融合，或近存计算，后者 “reducing off-chip memory access by roughly 50%”；(d) 支持转置 GEMM：目前反向要 “read out, dequantized, transposed, re-quantized into 128x1 tiles, and stored in HBM.”
  `insights` 的新增内容：§3.2 LogFMT（见 §2.7），“we do not employ it eventually”；§4.4 统一网卡/通信协处理器/硬件同步原语；§4.5 NVLink/PCIe 流量优先级、NIC 进 I/O die、CPU–GPU 走 NVLink；§5.1 多平面两层胖树（MPFT）；§5.2 IB vs RoCE、IBGDA；§6 鲁棒性（静默数据损坏）、CPU 瓶颈、内存语义通信的顺序问题、网内计算与压缩、DRAM 堆叠与晶圆级集成。

### 1.11 `insights` 中与本课直接相关的数字与引文

- **I1 表 1（KV cache，BF16）。** DeepSeek-V3 (MLA) 70.272 KB，1x；Qwen-2.5 72B (GQA) 327.680 KB，4.66x；LLaMA-3.1 405B (GQA) 516.096 KB，7.28x。正文：“requiring only 70 KB per token, substantially less than LLaMA-3.1 405B’s 516 KB and Qwen-2.5 72B’s 327 KB.”（**7.28x 是错的，应为 7.34x；见 §7 E1**）
- **I2 表 2（训练每 token 算力，序列长 4096）。** V2 236B：155 GFLOPS/Token；V3 671B：250；Qwen-72B Dense：394；LLaMa-405B Dense：2448。（表中单位写作 “GFLOPS/Token”，实为每 token 的 GFLOPs。）
- **I3 §2.3.2 TPOT 上界**（公式见 §2.8）：120.96 µs 是**一次 dispatch（FP8）加一次 combine（BF16）两次 all-to-all 的合计**（原文 “the time required for the two all-to-all communications in EP”）；每层再乘 2 是因为**双 micro-batch 重叠**（“an idealized scenario under dual micro-batch overlap”），不是“每层两次 all-to-all 各 120.96 µs”；61 层 → 14.76 ms，“equivalent to 67 tokens per second”；GB200 NVL72（“900GB/s unidirectional bandwidth across 72 GPUs”）→ 6.72 µs → 0.82 ms，“approximately 1200 tokens per second”。
- **I4 §2.3.3 MTP。** “an acceptance rate of 80% to 90% for predicting the second subsequent token, which increases the generation TPS by 1.8x”。（`v3` §5.4.3 写的是 “between 85% and 90%”，见 §7 E15。）
- **I5 §2.4。** FP8 消融在 “16B and 230B DeepSeek-V2 models”，“the relative accuracy loss compared to BF16 remains below 0.25%”。
- **I6 §3.1.1。** “the Tensor Core only maintains their highest 13 fraction bits for addition”；“Addition results are accumulated to FP22 registers (1 sign bit, 8 exponent bits, and 13 mantissa bits)”。v2 补充：“The term ‘FP22’ follows the naming used in the cited work”。
- **I7 §4.1。** H800 “features reduced FP64 computational performance and NVLink bandwidth for regulatory compliance”；“the NVLink bandwidth in H800 SXM nodes is reduced from 900 GB/s to 400 GB/s”；每节点 “eight 400G Infiniband (IB) CX7 NICs”。
- **I8 §4.2。** 八张 400Gbps IB 网卡使 all-to-all “at speeds exceeding 40GB/s”。
- **I9 §4.4.1。** 训练时 “up to 20 of the SMs on the H800 GPU are allocated for communication-related operations”；线上推理 “we perform EP all-to-all communication entirely through NIC RDMA, avoiding SM resource contention”。
- **I10 §5.1。** 每节点 8 GPU + 8 IB NIC，“each GPU–NIC pair assigned to a distinct network plane”；另有一块 400 Gbps RoCE NIC 接存储网（3FS）；“64-port 400G IB switches”，理论上 “up to 16,384 GPUs”；“due to policy and regulatory constraints, just over two thousand GPUs were ultimately deployed.”
- **I11 表 3（网络拓扑成本）。** FT2 / MPFT / FT3 / SF / DF：端点 2,048 / 16,384 / 65,536 / 32,928 / 261,632；交换机 96 / 768 / 5,120 / 1,568 / 16,352；链路 2,048 / 16,384 / 131,072 / 32,928 / 384,272；成本 [M$] 9 / 72 / 491 / 146 / 1,522；每端点 [k$] 4.39 / 4.39 / 7.5 / 4.4 / 5.8。
- **I12 表 4（2048 GPU 上训练 V3，MPFT vs MRFT）。** tokens/day (B) 272.80 / 272.52；time/step (s) 19.926 / 19.946；1F 1.13 / 1.13；bubble 2.06 / 2.03；1B 1.99 / 1.99；1W 0.48 / 0.48；1F1B 13.95 / 14.00；opt 0.29 / 0.31；TFLOPS (non-causal) 432 / 432；TFLOPS (causal) 385 / 385；MFU (non-causal) 43.73% / 43.68%；MFU (causal) 38.94% / 38.90%。说明：“MFU (Model Flops Utilization) is calculated based on BF16 peak performance.” 行名的定义（同段末句）：“1F, 1B, and 1W denote forward time, input backward time, and weight backward time, respectively.”——**1B 只是对输入的反向**（ZB 论文的 $T_B$、本课的 $T_{B_x}$），不是 `v3` 表 2 的完整反向 $B$。bubble、1F1B、opt 三行论文没有另行定义。
- **I13 表 5（64B 数据的 CPU 端到端延迟）。** RoCE 同 leaf 3.6us / 跨 leaf 5.6us；InfiniBand 2.8us / 3.7us；NVLink 3.33us / —。
- **I14 图 7（DeepEP on MPFT，每 GPU 4096 tokens）** 数据标签：16 GPU 43.05 与 42.47；32 GPU 58.02（dispatch）/ 56.96（combine）；64 GPU 50.58 / 48.54；128 GPU 45.34 / 41.6 GB/s。16 GPU 两个标签分别属于哪条线从图上无法判定。
- **I15 §2.2.2。** KTransformers 让完整 V3 在 “a low-cost server equipped with a consumer GPU (costing approximately $10,000)” 上 “nearly 20 TPS”。

### 1.12 FP8 格式（`fp8fmt`）

- **F1 两种编码。** “E4M3 (4-bit exponent and 3-bit mantissa) and E5M2 (5-bit exponent and 2-bit mantissa)”；“E4M3’s dynamic range is extended by not representing infinities and having only one mantissa bit-pattern for NaNs.”（摘要）
- **F2 表 1 数值**（§3，表 1）：见 §3.5。
- **F3 扩展的收益。** “extents the dynamic range by one extra power of 2, from 17 to 18 binades”；新增 7 个幅值 256…448；“The maximum representable magnitude without this modification would be 240.”（§3.1，原文拼写 “extents”）
- **F4 E5M2 的范围。** “5 bits of exponent provide sufficient per tensor dynamic range (32 binades, including the subnormal values)”（§3.1）。
- **F5 推荐用法。** “The recommended use of FP8 encodings is E4M3 for weight and activation tensors, and E5M2 for gradient tensors.”（§3）——DeepSeek-V3 偏离了这条推荐（C47）。
- **F6 溢出处理。** “Values that overflow are then saturated to the maximum representable value.”（§2）
- **F7 指数偏置。** “7 and 15 for E4M3 and E5M2, respectively.”（§3.2）

### 1.13 流水线文献（`zb`）

- **Z1** 1F1B 的来历：“It was first proposed in PipeDream (Harlap et al., 2018) under the asynchronous setting, and later introduced under synchronous settings”（`zb` §1）。
- **Z2** `zb` 表 1 下：“$T_W<T_F<T_B$ and $T_B+T_W=2T_F$”（这里 $T_B$ 只是**对输入的反向**）。
- **Z3** `zb` 表 2（transcribed）：1F1B $(p-1)(T_F+T_B+T_W)$、峰值激活 $pM_B$；ZB-H1 $(p-1)(T_F+T_B-T_W)$、$pM_B$；ZB-H2 $(p-1)(T_F+T_B-2T_W)$、$(2p-1)M_B$。
- **Z4** “ZB-1p: Automatically searched schedule with the activation memory limited to $pM_B$, which theoretically has the same peak memory as 1F1B.”（`zb` §5）

### 1.14 开源周制品（`gh-*`）

- **A1 DualPipe README 表**（“based on the same number of PP stages”）：1F1B / ZB1P / DualPipe / DualPipeV 的气泡、每设备参数、每设备激活、设备数，见 §2.1。DualPipeV：“a concise V-shape schedule derived from DualPipe using a ‘cut-in-half’ procedure, introduced by Sea AI Lab”，示例 “4 PP ranks (8 PP stages) and 10 micro-batches”。开发者：Jiashi Li、Chengqi Deng、Wenfeng Liang。
- **A2 DeepEP V1（2025-02，`docs/legacy.md`）。** 常规内核在 H800（“~160 GB/s NVLink maximum bandwidth”）+ CX7 400Gb/s（“~50 GB/s maximum bandwidth”）上、按 V3/R1 预训练设置（4096 tokens/batch、7168 hidden、top-4 groups、top-8 experts、FP8 dispatch、BF16 combine）：节点内 EP8 dispatch 153 GB/s、combine 158 GB/s（NVLink）；跨节点 EP16 43/43、EP32 58/57、EP64 51/50 GB/s（RDMA）。低延迟内核（128 tokens/batch）：dispatch EP8 77 us … EP256 194 us；combine EP8 114 us … EP256 360 us（完整表见 §3.8）。声明：“the implementation in this library may have some slight differences from the DeepSeek-V3 paper.” 示例代码 `Buffer.set_num_sms(24)`。另有一条“未定义行为 PTX”：用 `ld.global.nc.L1::no_allocate.L2::256B` 读 volatile 数据。
- **A3 DeepEP V2（现行 README）。** 改用 NCCL Gin 后端，“For V3-like legacy training, SM usage reduced from 24 to 4 - 6 while maintaining equivalent or better performance”；支持到 EP2048；还加了 “0 SM Engram”“0 SM PP”“0 SM CP” 等实验特性。
- **A4 DeepGEMM。** 开源周：“Up to 1350+ FP8 TFLOPS on Hopper GPUs”“Core logic at ~300 lines”（`gh-infra`）；2025.04.18 News：“up to **1550 TFLOPS** on H800”；SM90 缩放因子为 FP32，SM100 为打包的 UE8M0。
- **A5 FlashMLA。** 开源周：“BF16 support”“Paged KV cache (block size 64)”“3000 GB/s memory-bound | BF16 580 TFLOPS compute-bound on H800”（`gh-infra`）；2025.04.22 更新到 “up to 660 TFlops”。
- **A6 EPLB。** 冗余专家 + 启发式装箱；利用 group-limited routing 尽量把同组专家放同一节点；两种策略：层次式（节点数整除专家组数时，适合较小 EP 的 prefill）与全局式（适合较大 EP 的 decode）；“the exact method to predict the loads of experts is out of this repo's scope”。
- **A7 3FS。** 180 个存储节点（每节点 2×200Gbps IB、16 块 14TiB NVMe）读压测 “approximately 6.6 TiB/s”；GraySort 110.5 TiB、8,192 分区、30 分 14 秒、3.66 TiB/min（25 存储节点 + 50 计算节点）；KVCache 读 “peak throughput reaching up to 40 GiB/s”。
- **A8 V3 权重文档。** “DeepSeek-V3 natively supports FP8 weight format with 128x128 block scaling.”；`weight_scale_inv` 为 FP32，“(128x128 weight block) * weight_scale_inv”；不足 128 时 zero-pad 再算 scale；运行时 “online quantization at a granularity of `per-token-per-128-channel`”。README：“The total size of DeepSeek-V3 models on Hugging Face is 685B, which includes 671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights.”；“Since FP8 training is natively adopted in our framework, we only provide FP8 weights.”

---

## 2. 公式（原文转录 + 课程记号对照）

> `v3` 第 3 节**没有编号公式**。本课所有“公式”来自：`v3` 表 2（流水线气泡）、`fp8fmt` 表 1（格式定义）、`insights` §2.3.2 与 §3.2（无编号的行间式）、以及开源代码。凡是本笔记从图或代码归纳成公式的，都标“【由图/代码归纳】”。

### 2.1 流水线气泡与显存（`v3` 表 2，原文转录）

| Method | Bubble | Parameter | Activation |
|---|---|---|---|
| 1F1B | $(PP-1)(F+B)$ | $1\times$ | $PP$ |
| ZB1P | $(PP-1)(F+B-2W)$ | $1\times$ | $PP$ |
| DualPipe (Ours) | $(\frac{PP}{2}-1)(F\&B+B-3W)$ | $2\times$ | $PP+1$ |

表题原文：“$F$ denotes the execution time of a forward chunk, $B$ denotes the execution time of a full backward chunk, $W$ denotes the execution time of a "backward for weights" chunk, and $F\&B$ denotes the execution time of two mutually overlapped forward and backward chunks.”

`gh-dualpipe` README 的同一张表多了两列和一行（按“相同 PP stage 数”比较；“*PP* denotes the number of pp stages (even)”）：

| Method | Bubble | Parameter Per Device | Activation Per Device | #Devices |
|---|---|---|---|---|
| 1F1B | $(PP-1)(F+B)$ | $1\times$ | $PP$ | $PP$ |
| ZB1P | $(PP-1)(F+B-2W)$ | $1\times$ | $PP$ | $PP$ |
| DualPipe | $(PP/2-1)(F\&B+B-3W)$ | $2\times$ | $PP+1$ | $PP$ |
| DualPipeV | $(PP/2-1)(F\&B+B-3W)$ | $2\times$ | $PP+1$ | $PP/2$ |

**课程记号对照（必须改）**：`style-guide.md` §2 里 $B$ 是 batch size、$W$ 常指权重矩阵、$D$ 是训练 token 数（而 V3 的 MTP 深度也叫 $D$）。本课建议：

| 论文 | 课程 | 含义 |
|---|---|---|
| $PP$ | $P$（正文首次写“流水线级数 $P$（论文记作 $PP$）”） | 流水线 stage 数，偶数 |
| $F$ | $T_F$ | 一个前向 chunk 的时间 |
| $B$ | $T_B$ | 一个**完整**反向 chunk 的时间 $=T_{B_x}+T_W$ |
| — | $T_{B_x}$ | 只对输入求梯度的反向（ZB 论文里的 $T_B$） |
| $W$ | $T_W$ | 只对权重求梯度的反向 |
| $F\&B$ | $T_{F\&B}$ | 一对互相重叠的前向+反向 chunk 的时间 |

于是 $\text{bubble}_{\text{1F1B}}=(P-1)(T_F+T_B)$，$\text{bubble}_{\text{ZB1P}}=(P-1)(T_F+T_B-2T_W)$，$\text{bubble}_{\text{DualPipe}}=(\tfrac P2-1)(T_{F\&B}+T_B-3T_W)$。

**与 `zb` 表 2 的换算【笔者推断，代数可验】**：`zb` 的 $T_B$ 是 $T_{B_x}$。代入 $T_{B_x}=T_B-T_W$：
- `zb` 的 1F1B $(p-1)(T_F+T_{B_x}+T_W)=(p-1)(T_F+T_B)$ —— 与 `v3` 一致；
- `zb` 的 ZB-H1 $(p-1)(T_F+T_{B_x}-T_W)=(p-1)(T_F+T_B-2T_W)$ —— **正是 `v3` 表 2 的 “ZB1P” 行**；
- `zb` 的 ZB-1p 是**自动搜索**的调度，没有闭式气泡公式（Z4）。所以 V3 表中的 “ZB1P” 公式对应的是 ZB-H1 的手工调度公式（二者峰值激活都是 $pM_B$）。

### 2.2 DualPipe 调度的阶段数（`gh-dualpipe` `dualpipe.py` `step()`，代码转录）

记 $P$ = `num_ranks`（偶数），$C$ = `num_chunks`（两个方向合计的 micro-batch 数，偶数且 $C\ge 2P$），$H=P/2$，rank $r$ 的 $h=\min(r,P-1-r)$，$C/2$ 为每个方向的 micro-batch 数。每个 rank 顺序执行：

| 步 | 代码注释 | 次数 |
|---|---|---|
| 1 | `nF0` | $2(H-h-1)$ |
| 2 | `nF0F1` | $h+1$ |
| 3 | `nB1W1F1 (Use zero bubble)` | $H-h-1$ |
| 4 | `Main step: nF0B1F1B0` | $C/2-P+h+1$ |
| 5 | `nB1F1B0` | $H-h-1$ |
| 6 | `nB1B0 (The second half of the chunks use zero bubble)` | $h+1$ |
| 7 | `nWB0 (Use zero bubble)` | $H-h-1$ |
| 8 | `nW` | $h+1$ |

phase 的约定（代码注释原文）：“For the first half of the ranks: phase 0 means forward direction, phase 1 means reverse direction. For the second half of the ranks: phase 0 means reverse direction, phase 1 means forward direction.” 中间两个 rank（$r=H-1,H$）在第 4 步第一次迭代**不重叠**：“We don't overlap these two chunks to further reduce bubble size.”

### 2.3 细粒度量化（【由代码归纳】；论文只用文字和图 7(a) 描述）

`gh-v3` `inference/kernel.py` 的 `act_quant_kernel`：`amax = max(|x|)`，`amax = max(amax, 1e-4)`，`s = amax / 448.`，`y = x / s` 转成 `float8_e4m3fn`。据此，课程记号：

激活（第 $t$ 个 token 的行向量 $\mathbf x_t\in\R^{d}$，按内维切成 $d/128$ 段，第 $g$ 段 $\mathbf x_{t,[g]}\in\R^{128}$）：
$$
s_{t,g}=\frac{\max\big(\lVert \mathbf x_{t,[g]}\rVert_\infty,\,10^{-4}\big)}{448},\qquad
\hat{\mathbf x}_{t,[g]}=\mathrm{E4M3}\!\left(\frac{\mathbf x_{t,[g]}}{s_{t,g}}\right),\qquad
\mathbf x_{t,[g]}\approx s_{t,g}\,\hat{\mathbf x}_{t,[g]} .
$$
权重 $W\in\R^{d_{out}\times d_{in}}$ 按 $128\times128$ 块 $(a,g)$：$S_{a,g}=\max|W_{[a,g]}|/448$，存储的 `weight_scale_inv` 就是这个反量化乘子 $S_{a,g}$（名字里的 inv 指“量化缩放的倒数”）。448 是 E4M3 最大有限值（§2.6）。
`scale_fmt == "ue8m0"` 分支把 $s$ 上取整到 2 的幂：$s\leftarrow 2^{\lceil\log_2 s\rceil}$（这是后来 V3.1 的 UE8M0 格式，属于第 15 课；本笔记未核实该分支加入的日期）。

### 2.4 带分组缩放的 FP8 GEMM（【由图 7(a) 归纳】）

$y_{t,j}=\sum_k x_{t,k}W_{j,k}$，内维 $k$ 按 128 分组 $g$：
$$
y_{t,j}\;\approx\;\sum_{g=1}^{d_{in}/128} \underbrace{s_{t,g}\,S_{\lceil j/128\rceil,g}}_{\text{在 CUDA core 上乘}}\;\underbrace{\sum_{k\in g}\hat x_{t,k}\hat W_{j,k}}_{\text{Tensor Core 上 FP8 相乘、有限精度累加}} .
$$
这就是论文说的 “per-group scaling factors along the inner dimension”（C42）：缩放因子随 $g$ 变，因此不能在整个 $K$ 维做完再统一乘一次——这也是它“标准 FP8 GEMM 不直接支持”的原因。

### 2.5 累加精度提升（【由图 7(b) 与 §3.3.2 文字归纳】）

WGMMA 每条指令沿 $K$ 处理 32 个元素（$N_C=128$ “equivalent to 4 WGMMAs” ⇒ 128/4=32）。记 $P_g$ 为 Tensor Core 内 128 个乘积的有限精度部分和，则
$$
y\;=\;\mathrm{FP32}\text{-}\!\sum_{g}\,(s_g S_g)\cdot P_g,\qquad
P_g=\mathrm{acc}_{\approx 14\text{ bit}}\Big(\sum_{k\in g}\hat x_k\hat w_k\Big),\quad |g|=N_C=128 .
$$
硬件细节两种说法：`v3` v2 §3.5.2 “only uses the highest 14 bits of each mantissa product for addition … The accumulation of addition results into registers also employs 14-bit precision.”；`insights` §3.1.1 “highest 13 fraction bits … FP22 registers (1 sign bit, 8 exponent bits, and 13 mantissa bits)”。**【笔者推断】** 二者可以调和：13 位小数 + 1 位前导位 = 14 位有效位。`v3` v1 的原句 “employs fixed-point accumulation” 与 “at least 34-bit precision is required” 在 v2 被删掉。

### 2.6 FP8 与其它浮点格式的数值（`fp8fmt` 表 1 转录；BF16/FP16/FP32 由格式定义计算）

`fp8fmt` 表 1（原样）：

| | E4M3 | E5M2 |
|---|---|---|
| Exponent bias | 7 | 15 |
| Infinities | N/A | $S.11111.00_2$ |
| NaN | $S.1111.111_2$ | $S.11111.\{01,10,11\}_2$ |
| Zeros | $S.0000.000_2$ | $S.00000.00_2$ |
| Max normal | $S.1111.110_2=1.75*2^{8}=448$ | $S.11110.11_2=1.75*2^{15}=57{,}344$ |
| Min normal | $S.0001.000_2=2^{-6}$ | $S.00001.00_2=2^{-14}$ |
| Max subnorm | $S.0000.111_2=0.875*2^{-6}$ | $S.00000.11_2=0.75*2^{-14}$ |
| Min subnorm | $S.0000.001_2=2^{-9}$ | $S.00000.01_2=2^{-16}$ |

值的公式（E4M3，$e$ 为 4 位指数字段、$m$ 为 3 位尾数字段）：$e\ge1$ 时 $v=(-1)^S(1+m/8)\,2^{e-7}$；$e=0$ 时 $v=(-1)^S(m/8)\,2^{-6}$；唯一被拿走的码字是 $e=15,m=7$（NaN）。E5M2：$v=(-1)^S(1+m/4)\,2^{e-15}$，$e=31$ 保留给 Inf/NaN。

### 2.7 LogFMT（`insights` §3.2，原文转录）

给定一个 tile $[x_1,\cdots,x_m]$（实现中为 1x128），取 $\log|x_i|$ 的最小值 $min$ 与最大值 $max$；$min$ 编码为 $S.00\cdots01$，$max$ 编码为 $S.11\cdots11$，
$$
Step=\frac{max-min}{2^{n-1}-2},
$$
零用 $S.00\cdots00$ 表示；其余值取为 $Step$ 的最近整数倍 $K$；解码为符号位与 $exp^{\,min+Step\times(K-1)}$ 的组合；约束 $min > max-\log(2^{32})$（“the max representation range is similar to E5”）。论文强调 “it is important to round in the original Linear space, instead of the Log space, for the unbiased activation quantization.” 结果：$n=8$ 时 “LogFMT-8Bit shows superior training accuracy compared to E4M3 or E5M2”；$n=10$ 时与 BF16 combine 相当；但与 all-to-all 融合的编解码开销 “(50%∼100%)”，最终未采用。
**课程记号**：把 $min,max$ 改写成 $\ell_{\min},\ell_{\max}$，$Step$ 写作 $\Delta_\ell$，避免与 min/max 算子混淆。

### 2.8 MoE 推理速度上界（`insights` §2.3.2，原文转录）

$$
\text{Comm. Time}=(1\,\text{Byte}+2\,\text{Bytes})\times 32\times 9\times 7\text{K}/50\,\text{GB/s}=120.96\,\mu s
$$
$$
\text{Total Time Per Layer}=2\times 120.96\,\mu s=241.92\,\mu s,\qquad
\text{Total Inference Time}=61\times 241.92\,\mu s=14.76\,\text{ms}
$$
$$
\text{Comm. Time}=(1\,\text{Byte}+2\,\text{Bytes})\times 32\times 9\times 7\text{K}/900\,\text{GB/s}=6.72\,\mu s
$$
含义（原文）：dispatch 用 FP8（1 byte）、combine 用 BF16（2 bytes），hidden “approximately 7K”，9 = 8 路由 + 1 共享，每设备约 32 个 token，“Network latency is not included”。**注意 “7K” 按 7000 计算**才得到 120.96（用 7168 为 123.86 µs，§4.8）。
**两个“2”不要混**：第一式里的 $(1+2)$ Bytes 已经把 dispatch 与 combine 两次 all-to-all 都算进去了（原文引入此式时说 “the time required for the two all-to-all communications in EP is calculated as follows”）；第二式的 $2\times$ 来自双 micro-batch 重叠——原文：“this analysis represents an idealized scenario under dual micro-batch overlap. Under this assumption, the total time per layer can be formulated as”。即每层时间 = 两个 micro-batch 各自一轮（dispatch+combine）通信，计算被假设完全藏在通信后面。
NVL72 那一式的前提在 v1 与 v2 不同：v1 “Assuming the computation time is equal to the communication time … over 0.82 ms TPOT”；v2 “Assuming perfect overlap between computation and communication, this would yield a theoretical upper limit of 0.82 ms TPOT”。

### 2.9 节点受限路由的 IB 代价（`insights` §4.3，文字转公式）

8 个目标专家分散在 8 个节点：IB 时间 $8t$；落在 $M$ 个节点并经 NVLink 去重：$Mt$（$M<8$），V3 取 $M=4$。`v3` 的等效说法：一次 IB 发送后在节点内可免费扇出平均 $160/50=3.2$ 个专家，于是 $4\times3.2=12.8\approx13$。

### 2.10 成本（`v3` §1，文字转公式）

$$
\text{GPU·h}=2664\text{K}+119\text{K}+5\text{K}=2788\text{K},\qquad \text{USD}=2788\text{K}\times\$2=\$5.576\text{M},
$$
$$
\text{预训练}=14.8\ (\text{T tokens})\times 180\text{K}\ (\text{GPU·h/T})=2664\text{K}.
$$

---

## 3. 规格与超参数表

### 3.1 训练集群与硬件

| 项 | 值 | 来源 |
|---|---|---|
| GPU 数 | 2048 × H800 | `v3` §3.1；`insights` 摘要 “2,048 NVIDIA H800 GPUs” |
| 每节点 GPU | 8，NVLink + NVSwitch | `v3` §3.1 |
| 节点内 NVSwitch 芯片 | 4 个 “NVLink Switch Chip”，与 8 块 “H800 SXM5” 全连接 | `insights` 图 2（读图） |
| 每节点 PCIe 交换 / IB 网卡 | 8 个 PCIe Switch，每个挂 1 块 CX7 NIC + 1 块 H800；另 1 块 Storage NIC 挂在 CPU 2 | `insights` 图 2（读图）；§5.1 |
| SM 数 | 132 | `v3` §3.5.1 |
| NVLink（H800） | §4.1：400 GB/s（H100 为 900 GB/s）；§4.3：“NVLink provides 200GB/s bandwidth (of which about 160GB/s can actually be achieved)”；两个数的关系见 §8 第 2 条 | `insights` §4.1、§4.3；`v3` §3.2.2 写 160 GB/s |
| IB | 400 Gbps CX7 = 50 GB/s；有效按 40 GB/s | `v3` §3.2.2；`insights` §4.3 |
| 节点内 : 节点间 | ≈3.2（160/50，`v3`）或 ≈4:1（200/50 或 160/40，`insights`） | 同上 |
| 存储网 | 每节点 1 块 400 Gbps RoCE NIC，接 3FS | `insights` §5.1 |
| 网络拓扑 | 8 平面两层胖树（MPFT），64 口 400G IB 交换机，理论可达 16,384 GPU | `insights` §5.1、图 3 |
| BF16 稠密峰值（推得） | ≈ 988 TFLOPS | 由 `insights` 表 4：385/0.3894 = 988.7，432/0.4373 = 987.9（§4.3）；NVIDIA 原始数据表未读到（§8） |

### 3.2 训练并行与通信配置

| 项 | 值 | 来源 |
|---|---|---|
| PP | 16 | `v3` §3.2 |
| EP | 64，跨 8 节点；每层 256 路由专家均匀放在 64 GPU（每 GPU 4 个） | `v3` §3.2、§4.2 |
| DP | ZeRO-1 | `v3` §3.2 |
| TP | 不用（训练） | `v3` §3.2；`insights` §4.2 |
| 每 token 最多节点数 | $M=4$ | `v3` §3.2.2、§4.2；`config.json` `topk_group: 4`、`n_group: 8` |
| 通信 SM | 20 个，分成 10 个通道；warp 专门化 | `v3` §3.2.2 |
| 序列长 / batch | 4K；batch 3072 → 15360（前 469B token 内增长） | `v3` §4.2 |
| 每 chunk 层数（profile） | 4 个 MoE 层；EP64、TP1、4K | `gh-profile` |
| 训练 profile 中 PP 通信 | 未计入 | `gh-profile` |
| DualPipe 约束（论文） | stage 数与 micro-batch 数都能被 2 整除 | `v3` §3.2.1 |
| DualPipe 约束（代码） | `num_ranks % 2 == 0`，`num_chunks % 2 == 0 and num_chunks >= num_ranks * 2` | `gh-dualpipe` `dualpipe.py` |

### 3.3 FP8 配方：每个张量/算子用什么精度

| 张量 / 算子 | 精度 | 来源 |
|---|---|---|
| Linear 的 Fprop / Dgrad / Wgrad 三个 GEMM 的输入 | FP8 E4M3 | `v3` §3.3.1、§3.3.2 “E4M3 format on all tensors” |
| GEMM 输出 | BF16 或 FP32 | `v3` §3.3.1 |
| GEMM 内部累加 | Tensor Core 约 14 bit，每 $N_C=128$ 提升到 CUDA core FP32 | `v3` §3.3.2 |
| 激活量化粒度 | 1×128（每 token 每 128 通道）；反向转置为 128×1 | `v3` §3.3.2、§3.5.2、附录 B.2 |
| 权重量化粒度 | 128×128 | `v3` §3.3.2；`config.json` `weight_block_size: [128, 128]` |
| 缩放因子 | 在线计算（非 delayed）；FP32（`weight_scale_inv` 为 FP32）；个别位置取 2 的整数幂 | `v3` §3.3.2、§3.3.3；`gh-v3` README_WEIGHTS |
| 嵌入、输出头、MoE 门控、归一化、注意力算子 | 原精度（BF16 或 FP32） | `v3` §3.3.1 |
| RMSNorm、Router、RoPE、交叉熵 | FP32（图示） | `insights` 图 1 |
| 核心注意力 | BF16（图示；Day 6 “core MLA computations … BF16”） | `insights` 图 1；`gh-infra` Day 6 |
| 主权重（master weights） | FP32 | `v3` §3.3.3 |
| 权重梯度（跨 micro-batch 累加） | FP32 | `v3` §3.3.3；图 6 “Weight Gradient FP32” |
| AdamW 一阶、二阶矩 | BF16 | `v3` §3.3.3 |
| 缓存给反向用的 Linear 输入激活 | FP8 | `v3` §3.3.3 |
| 注意力之后 Linear 的输入激活 | 自定义 E5M6；缩放因子为 2 的幂 | `v3` §3.3.3 |
| MoE SwiGLU 的输入 | FP8 细粒度缓存，反向重算输出 | `v3` §3.3.3 |
| dispatch（前向 up-proj 前激活 / 反向 down-proj 前激活梯度） | FP8，缩放因子为 2 的幂 | `v3` §3.3.3 |
| combine（前向与反向） | BF16 | `v3` §3.3.3 |
| 优化器超参（顺带） | AdamW β1=0.9、β2=0.95、wd=0.1；梯度裁剪 1.0 | `v3` §4.2 |

### 3.4 推理部署：三个来源并列

| 项 | `v3` §3.4（2024-12） | Day 6（统计窗口截至 2025-02-28 12:00 UTC+8，`gh-infra`） | profile-data（2025-02 开源周，`gh-profile`） |
|---|---|---|---|
| prefill 最小单元 | 4 节点 32 GPU | 4 节点 | — |
| prefill 注意力并行 | TP4 + SP，DP8 | “MLA/Shared Expert DP32” | TP1 |
| prefill MoE | EP32；32 个冗余专家；每 GPU 8+1 个专家 | EP32；32 冗余；每 GPU 9 路由 + 1 共享 | EP32 |
| prefill 其它 | 浅层稠密 MLP 用 TP1；双 micro-batch 重叠 | 双 micro-batch 交替 | prompt 4K，每 GPU 16K tokens |
| decode 最小单元 | 40 节点 320 GPU | 18 节点 | — |
| decode 注意力 | TP4 + SP，DP80 | “MLA/Shared Expert DP144” | TP1 |
| decode MoE | EP320；每 GPU 1 个专家；64 GPU 放冗余与共享专家 | EP144；32 冗余；每 GPU 2 路由 + 1 共享 | EP128 |
| decode 通信 | IB 点对点 + IBGDA | 5 级流水重叠 | all-to-all 不占 SM |
| 冗余专家更新 | 周期性（如每 10 分钟） | — | — |

### 3.5 数值格式（`fp8fmt` 表 1；BF16/FP16/FP32 为格式定义下的计算值）

| 格式 | 位（S/E/M） | 偏置 | 最大有限值 | 最小正规数 | 最小次正规数 | 机器 ε（1 处的间距） | 非负有限值个数 |
|---|---|---|---|---|---|---|---|
| E4M3 | 1/4/3 | 7 | 448 | $2^{-6}=0.015625$ | $2^{-9}\approx0.001953$ | $2^{-3}=0.125$ | 127（含 0） |
| E5M2 | 1/5/2 | 15 | 57,344 | $2^{-14}\approx6.10\times10^{-5}$ | $2^{-16}\approx1.53\times10^{-5}$ | $2^{-2}=0.25$ | 124（含 0） |
| FP16 | 1/5/10 | 15 | 65,504 | $2^{-14}$ | $2^{-24}\approx5.96\times10^{-8}$ | $2^{-10}$ | — |
| BF16 | 1/8/7 | 127 | $\approx3.39\times10^{38}$ | $2^{-126}\approx1.18\times10^{-38}$ | $2^{-133}\approx9.18\times10^{-41}$ | $2^{-7}=0.0078125$ | — |
| FP32 | 1/8/23 | 127 | $\approx3.40\times10^{38}$ | $2^{-126}$ | $2^{-149}\approx1.40\times10^{-45}$ | $2^{-23}\approx1.19\times10^{-7}$ | — |
| E5M6（V3 自定义） | 1/5/6 | 未给 | 未给 | 未给 | 未给 | $2^{-6}$ | — |
| UE8M0（MX 缩放，预告第 15 课） | 0/8/0 | — | — | — | — | 只表示 2 的幂 | — |

E4M3 的动态范围：最大/最小次正规 = 448/2⁻⁹ = 229,376 ≈ 2^17.8（`fp8fmt` 说 “18 binades”）；最大/最小正规 = 28,672 ≈ 2^14.8。E5M2：最大/最小次正规 = 3.76×10⁹ ≈ 2^31.8（“32 binades, including the subnormal values”）。

### 3.6 发布的权重（`hf-v3`、`gh-v3`）

| 项 | 值 | 来源 |
|---|---|---|
| `quantization_config` | `quant_method: fp8`，`fmt: e4m3`，`activation_scheme: dynamic`，`weight_block_size: [128, 128]` | `config.json` |
| `torch_dtype` | `bfloat16`（尽管权重以 FP8 存） | `config.json` |
| 层数 / hidden / 头数 | 61 / 7168 / 128 | `config.json`；`v3` §4.2 |
| `kv_lora_rank` / `q_lora_rank` / `qk_rope_head_dim` / `qk_nope_head_dim` / `v_head_dim` | 512 / 1536 / 64 / 128 / 128 | `config.json` |
| `first_k_dense_replace` / `intermediate_size` / `moe_intermediate_size` | 3 / 18432 / 2048 | `config.json` |
| 路由专家 / 共享 / 激活 / 组 / 组 top-k | 256 / 1 / 8 / 8 / 4 | `config.json` |
| `num_nextn_predict_layers` | 1 | `config.json` |
| `vocab_size` | 129280 | `config.json` |
| `initializer_range` | 0.02（论文 §4.2 写 std 0.006） | `config.json` vs `v3` |
| HF API 按 dtype 的元素数 | F8_E4M3 680,571,043,840；BF16 3,918,786,560；F32 41,555,600；合计 684,531,386,000 | HF API `safetensors` 字段 |
| 163 个 safetensors 文件实际大小 | 688,586,727,753 B = 688.59 GB = 641.30 GiB | HF API `?blobs=true` 求和 |
| index.json 的 `total_size` | 1,369,062,772,000（= 2 × 参数数，按 BF16 记账，**不是**下载大小） | `model.safetensors.index.json` |
| 官方参数说法 | 671B 主模型 + 14B MTP = 685B；MTP “11.5B unique parameters”；主模型激活 “36.7B”（含 0.9B 嵌入与 0.9B 输出头）；MTP 激活 “2.4B” | `gh-v3` README、README_WEIGHTS |

### 3.7 实测训练指标（`insights` 表 4，MPFT 列）

| 指标 | 值 |
|---|---|
| tokens/day | 272.80B |
| time/step | 19.926 s |
| 1F / bubble / 1B / 1W / 1F1B / opt | 1.13 / 2.06 / 1.99 / 0.48 / 13.95 / 0.29 s（论文：1F=forward time，1B=**input** backward time，1W=weight backward time；其余三行无定义） |
| TFLOPS（non-causal / causal） | 432 / 385 |
| MFU（non-causal / causal，按 BF16 峰值） | 43.73% / 38.94% |

### 3.8 制品性能数字（均为公司自测）

| 制品 | 数字 | 来源 |
|---|---|---|
| FlashMLA（2025-02） | 3000 GB/s（访存受限），BF16 580 TFLOPS（计算受限），H800；分页 KV 块大小 64 | `gh-infra` Day 1 |
| FlashMLA（2025-04 起） | 最高 660 TFLOPS（H800 SXM5，CUDA 12.8） | `gh-flashmla` News 2025.04.22 |
| DeepGEMM | 开源时 “1350+ FP8 TFLOPS”；2025-04 起 “up to 1550 TFLOPS on H800” | `gh-infra` Day 3；`gh-deepgemm` News |
| DeepEP V1 常规内核 | 节点内 EP8：153 / 158 GB/s；跨节点 EP16：43 / 43；EP32：58 / 57；EP64：51 / 50（dispatch / combine） | `gh-deepep` `docs/legacy.md` |
| DeepEP V1 低延迟内核（128 tok/batch） | dispatch：EP8 77us(98GB/s)、16 118us(63)、32 155us(48)、64 173us(43)、128 192us(39)、256 194us(39)；combine：EP8 114us(127)、16 195us(74)、32 273us(53)、64 314us(46)、128 369us(39)、256 360us(40) | 同上 |
| DeepEP V2 | V3 式训练的 SM 从 24 降到 4–6；SM90+CX7 EP 8×2 dispatch 90 / combine 81 GB/s（12 SM），EP 8×4 61/61（6 SM） | `gh-deepep` README |
| 3FS | 6.6 TiB/s 聚合读（180 节点）；GraySort 3.66 TiB/min；KVCache 客户端峰值 40 GiB/s | `gh-3fs` |

---

## 4. 复算的数字

复算脚本：scratch 目录下 `nums.py`、`params.py`、`dpsim.py`、`fp8sim.py`、`accsim.py`（未放入仓库；关键表达式都抄在下面，可独立重跑）。

### 4.1 成本账

| 表达式 | 值 |
|---|---|
| `2664e3+119e3+5e3` | 2,788,000 GPU·h |
| `2*2788e3` | \$5,576,000；分项 \$5,328,000 / \$238,000 / \$10,000 |
| 占比 `2664/2788, 119/2788, 5/2788` | 95.55% / 4.27% / 0.18% |
| `2664e3/2048/24` | 54.20 天（“less than two months”） |
| `119e3/2048/24`；`5e3/2048/24` | 2.42 天；0.10 天 |
| `2788e3/2048/24` | 56.72 天 |
| `180e3/2048/24` | 3.662 天（论文写 3.7） |
| `14.8*180e3` | 2,664,000 —— 与表 1 预训练**精确相等** |
| `119e3+5e3` | 124,000（论文说 “0.1M”） |

### 4.2 吞吐自洽性（`insights` 表 4 ↔ `v3` 表 1）

| 表达式 | 值 |
|---|---|
| 每步 token `15360*4096` | 62,914,560 |
| 每天步数 `86400/19.926` | 4,336.0 |
| 每天 token `62914560*86400/19.926` | 272.800B（表 4 印 272.80B） |
| 每万亿 token GPU·h `1e12/272.80e9*2048*24` | 180,176 ≈ 180K（与 `v3` §1 一致） |
| 气泡占比 `2.06/19.926` | 10.34%（MRFT 2.03/19.946 = 10.18%） |
| 阶段和 `1.13+2.06+1.99+0.48+13.95+0.29` | 19.90 s（≈ time/step 19.926） |

### 4.3 每 token FLOPs 与 MFU（从 `config.json` 重建）

每层注意力参数 `qa+qb+kva+kvb+o` = 1536·7168 + 128·192·1536 + 576·7168 + 128·256·512 + 7168·16384 = **187,105,280**；稠密 FFN 3·18432·7168 = 396,361,728；单个专家 3·2048·7168 = 44,040,192；门控 256·7168 = 1,835,008；嵌入 = 输出头 = 129280·7168 = 926,679,040。

| 表达式 | 值 |
|---|---|
| 主模型每 token 激活的矩阵参数（含输出头、不含嵌入）`61*attn + 3*dense + 58*(9*expert+gate) + emb` | 36.625B（含嵌入 37.551B） |
| `6*36.625e9` | 219.75 GFLOPs |
| 注意力 FLOPs（causal，平均上下文 2048）`3*61*2*128*(192+128)*4096/2` | 30.70 GFLOPs |
| 合计（causal） | **250.45 GFLOPs/token**（`insights` 表 2 印 250） |
| 合计（non-causal，注意力翻倍） | 281.15 GFLOPs/token |
| non-causal / causal | 1.1226；表 4 的 432/385 = 1.1221 —— 吻合 |
| `272.80e9*250.45e9/86400/2048` | 386.1 TFLOPS（表 4 causal 385） |
| `272.80e9*281.15e9/86400/2048` | 433.4 TFLOPS（表 4 non-causal 432） |
| 推得的 BF16 峰值 `385/0.3894`；`432/0.4373` | 988.7；987.9 TFLOPS |
| 预训练总算力 `250.45e9*14.8e12` | 3.71×10²⁴ FLOPs（对照 `6*37e9*14.8e12` = 3.29×10²⁴） |
| 若把 MTP 模块也计入（causal） | 260.6 GFLOPs/token —— 与 250 不符，推断表 2 **不含** MTP【笔者推断】 |

### 4.4 参数与检查点（`hf-v3` 精确核对）

| 表达式 | 值 | 对照 |
|---|---|---|
| FP8 元素 `62*attn + 3*dense + 59*257*expert` | 680,571,043,840 | HF API F8_E4M3 **完全相同** |
| BF16 元素 `4*emb + 59*gate + 2*7168*7168 + 62*(7168+7168+1536+512) + 4*7168` | 3,918,786,560 | HF API BF16 **完全相同**（= 主模型嵌入+输出头、MTP 层存的嵌入与头副本、59 个门控、MTP 的 `eh_proj`、全部 RMSNorm） |
| 128×128 缩放块数 `62*Σceil + 3*… + 59*257*2688` | 41,540,496 | HF F32 41,555,600 减去 59×256 = 15,104 个 `e_score_correction_bias` 后**完全相同** |
| 主模型总参数 | 671,026,419,200 = 671.03B | 671B |
| MTP 独有参数 | 11.610B | README_WEIGHTS “11.5B” |
| MTP 含重复存储的嵌入/输出头 | 13.463B | README “14B” |
| 检查点合计 | 684,489,845,504 | “685B” |
| 实际文件 688.59 GB ≈ `680.571e9*1 + 3.9188e9*2 + 41.5556e6*4` | 688.58 GB（差额为 safetensors 头） | |

**结论**：发布的检查点里，全部线性层权重是 FP8 E4M3（含 128×128 FP32 缩放），嵌入、输出头、门控、RMSNorm、MTP 的 `eh_proj` 为 BF16，门控偏置 $b_i$ 为 FP32——与 C38 “保持高精度组件”的名单一致。

### 4.5 显存

| 表达式 | 值 |
|---|---|
| 671.03B × 1 B（FP8） | 671.0 GB = 624.9 GiB |
| × 2 B（BF16） | 1,342.1 GB |
| × 4 B（FP32） | 2,684.1 GB |
| 一个 8×H800 节点 HBM `8*80` | 640 GB（< 671 GB：FP8 权重都放不进一个节点） |
| prefill 单元 32 GPU / decode 单元 320 GPU 的 HBM | 2,560 GB / 25,600 GB |
| FP8 权重平均摊到 32 / 320 GPU | 20.97 / 2.10 GB 每 GPU |
| 经典混合精度 Adam（`zero`：16Ψ） | 10.74 TB |
| V3 配方下的训练状态（FP32 主权重 4 + FP32 梯度 4 + BF16 两个矩 2+2 = 12 B/参数）【笔者估算】 | 8.05 TB（再加 1 B FP8 权重副本为 8.72 TB） |
| 集群 HBM `2048*80` | 163.84 TB；12 B/参数完全均摊到 2048 GPU 为 3.93 GB/GPU |
| 缩放因子开销：权重 `4/(128*128)`；激活 `4/128` | 0.0244%；3.125% |
| 全模型缩放因子字节 `41540496*4` | 166.2 MB |
| DualPipe 每 GPU 参数【笔者估算：4 个 MoE 层/段，每层 4 路由专家 + 共享专家 + 注意力 + 门控】`4*(4*44040192+44040192+187105280+1835008)` | 1.637B；两段（2×）3.273B |
| decode 每 GPU 专家参数 `58*44040192` | 2.554B（FP8 约 2.55 GB） |
| decode 注意力 TP4 均摊 `61*187105280/4`【粗估，未计复制部分】 | 2.853B |
| 若每 GPU 剩约 70 GB 给 KV：`70e9/70272` | ≈ 996K token；按 Day 6 平均 KV 长 4,989 约 200 条序列【笔者估算】 |

### 4.6 流水线气泡（$T_F=1,T_B=2,T_W=1$，即 $T_{B_x}=1$；$T_{F\&B}$ 取 3 或 2.5；通信时间设为 0）

| $P$ | 1F1B $(P-1)(T_F+T_B)$ | ZB1P $(P-1)(T_F+T_B-2T_W)$ | DualPipe $(P/2-1)(T_{F\&B}+T_B-3T_W)$，$T_{F\&B}=3$ | 同左，$T_{F\&B}=2.5$ | 激活 1F1B / DualPipe |
|---|---|---|---|---|---|
| 4 | 9 | 3 | 2 | 1.5 | 4 / 5 |
| 8 | 21 | 7 | 6 | 4.5 | 8 / 9 |
| **16** | **45** | **15** | **14** | **10.5** | **16 / 17（+6.25% = 1/16）** |
| 32 | 93 | 31 | 30 | 22.5 | 32 / 33 |

气泡占全步时间的比例（每个 rank 的计算量 = $C\cdot(T_F+T_B)=3C$）：

| $P$, $C$ | 1F1B | ZB1P | DualPipe（仿真 makespan） |
|---|---|---|---|
| 8, 20 | 25.9% | 10.4% | 9.1%（66） |
| 16, 32 | 31.9% | 13.5% | 12.7%（110） |
| 16, 64 | 19.0% | 7.3% | 6.8%（206） |
| 16, 128 | 10.5% | 3.8% | 3.5%（398） |

**DualPipe 仿真（`dpsim.py`）**：按 `dualpipe.py` 的操作顺序、以“前一 stage 的 F 完成 / 后一 stage 的 B 完成”为依赖做事件驱动仿真。结果：
- $P=8,C=20$：每个 rank 空闲都恰为 6.0，等于公式 $(4-1)(3+2-3)=6$；$P=16$ 时 $C=32$ 与 $C=64$ 空闲都是 14.0 —— 验证了 C20（气泡不随 micro-batch 数增长）。
- $T_{F\&B}=2.5$（$P=8$）：最大空闲 4.5 = 公式值。
- **$T_W\ne T_F$ 时公式不精确**：$T_W=0.8$ 时仿真 7.2、公式 7.8；$T_W=0.5$ 时仿真 9.5、公式 10.5（§7 E8）。

### 4.7 FP8 数值

| 表达式 | 值 |
|---|---|
| `1.75*2**8`；`1.75*2**15` | 448；57,344 |
| 不扩展时 E4M3 最大值 `(1+7/8)*2**7` | 240（`fp8fmt` 原文同） |
| E4M3 编码数 | 256 = 254 个有限值（127 非负含 ±0 各一）+ 2 个 NaN |
| `log2(448/2**-9)`；`log2(57344/2**-16)` | 17.81；31.81 |
| 细粒度量化例（`fp8sim.py`，seed 0：128 个 N(0,1)×0.01，第 17 个元素设为 10） | 按本 tile amax 缩放：s = 0.02232，非离群值中位相对误差 2.26%，无下溢；若同一张量别处有 amax=1000 的离群值、按整张量缩放：s = 2.232，中位误差 16.4%，127 个非离群值中 21 个被量化成 0；去掉离群值的 tile（即设置 `x[17]=10` 之前的原始 128 个值）：s = 5.19×10⁻⁵，中位误差 1.65%（若改为删去第 17 个元素、只剩 127 个值，中位误差 1.66%） |
| 3 位尾数的相对间距 | 每个 binade 内 1/8 到 1/16（12.5%→6.25%），最大舍入误差为其一半 |

### 4.8 通信与路由

| 表达式 | 值 |
|---|---|
| `160/50`；`4*3.2` | 3.2；12.8（论文写 13） |
| `200/50`；`160/40`；`200/40` | 4；4；5（`insights` 说 “approximately 4:1”） |
| `20/132` | 15.2% 的 SM 用于通信 |
| `3*32*9*7000/50e9` | 120.96 µs；用 7168：123.86 µs |
| `61*2*120.96e-6` → `1/…`（×2 = 双 micro-batch，见 §2.8） | 14.757 ms → 67.8 tok/s |
| `3*32*9*7000/900e9`；`61*2*6.72e-6` → `1/…` | 6.72 µs；0.820 ms → 1,220 tok/s |
| 节点受限的 IB 代价 | 8t → 4t（减半） |

### 4.9 部署与经济账（Day 6）

| 表达式 | 值 |
|---|---|
| prefill 每 GPU 专家 `(256+32)/32` | 9（+1 共享，Day 6；论文说 8+1 冗余） |
| decode `(256+64)/320`（论文）；`(256+32)/144`（Day 6） | 1；2 |
| TP×DP：`4*8`；`4*80` | 32；320 |
| 日成本 `226.75*8*24*2` | \$87,072（与原文完全一致） |
| 按 token 数重算收入 `342e9*0.14/1e6 + 266e9*0.55/1e6 + 168e9*2.19/1e6` | \$562,100（原文 \$562,027，差异来自 token 数取整） |
| 利润率 `(562027-87072)/87072` | 545.5% |
| 缓存命中率 `342/608` | 56.25%（原文 56.3%） |
| prefill 节点数 `608e9/86400/73.7e3` | 95.5 |
| decode 节点数 `168e9/86400/14.8e3` | 131.4 |
| 合计 | 226.9 ≈ 平均 226.75 节点 —— 自洽 |

### 4.10 KV cache 每 token（`insights` 表 1 复算）

| 模型 | 表达式 | 字节 | 倍数 |
|---|---|---|---|
| DeepSeek-V3 (MLA) | `(512+64)*61*2` | 70,272 | 1 |
| Qwen2.5-72B（80 层，8 KV 头，d_h=8192/64=128） | `80*8*128*2*2` | 327,680 | **4.663** |
| Llama-3.1-405B（126 层，8 KV 头，d_h=16384/128=128） | `126*8*128*2*2` | 516,096 | **7.344**（论文印 7.28x） |
| V3 @ 4K / 32K / 128K | ×4096 / ×32768 / ×131072 | 0.288 / 2.30 / 9.21 GB | |
| Qwen @ 128K；Llama @ 128K | | 42.9 GB；67.6 GB | |

### 4.11 其它

| 表达式 | 值 |
|---|---|
| LogFMT-8：码 $K=1..127$，`Step` 分母 $2^{7}-2$ | 126 |
| 最大对数跨度 `32*ln2` → 最坏步长 `22.18/126` → 相邻比值 `exp(0.176)` | 22.18；0.176；1.192（相邻两级相差约 19%） |
| LogFMT-10 最坏相邻比值 `exp(22.18/510)` | 1.044 |
| MPFT：两层胖树 64 口 `64**2/2`；8 平面 | 2,048；16,384 端点 |
| FT2 交换机 = 64 leaf + 32 spine；8 平面 | 96；768 |
| FT3 `64**3/4`；`5*64**2/4` | 65,536 端点；5,120 交换机 |
| 每端点成本 `9e6/2048`；`491e6/65536` | \$4,395；\$7,492 |
| 3FS GraySort `110.5/(30+14/60)` | 3.655 TiB/min |

### 4.12 累加精度的简化模型（`accsim.py`，**本笔记自己的模型，不是硬件测量**）

模型：每 32 个 FP8×FP8 乘积按组内最大指数对齐、截断到 14 位有效位后定点相加；累加器每次加完截断到 14 位有效位；可选每 128 个元素把部分和加到 FP32。输入为 N(0,1) 量化到 E4M3，$32\times32$ 输出。

| $K$ | 不提升：中位逐元素相对误差 | 不提升：Frobenius 相对误差 | $N_C=128$ 提升：中位 | 提升：Frobenius |
|---|---|---|---|---|
| 512 | 3.28e-4 | 3.72e-4 | 9.74e-5 | 1.02e-4 |
| 1024 | 7.13e-4 | 7.51e-4 | 9.79e-5 | 9.83e-5 |
| 2048 | 1.44e-3 | 1.50e-3 | 9.62e-5 | 9.92e-5 |
| 4096 | 3.10e-3 | 3.10e-3 | 1.02e-4 | 1.04e-4 |
| 8192 | 6.19e-3 | 6.26e-3 | 9.83e-5 | 1.01e-4 |

要点：不提升时误差随 $K$ **线性**增长（截断是朝零的系统偏差，不会像随机误差那样按 $\sqrt K$ 抵消）；提升后与 $K$ 无关。量级与论文 “nearly 2%” 不同，因为论文没定义“最大相对误差”的口径（逐元素最大值会被接近 0 的输出放大：本模型 $48\times48$、$K=4096$ 时逐元素最大达 484%，99 分位 10.4%），而且真实硬件细节未公开。正文只能用它说明**趋势**。

---

## 5. 图表清单与重绘建议

### 5.1 论文原图清单

| 图 | 出处 | 图题要点 | 是否重绘 |
|---|---|---|---|
| 图 4 | `v3` §3.2.1 | 一对前向/反向 chunk 的重叠策略；颜色：橙=前向、绿=backward for input、蓝=backward for weights、紫=PP 通信、红=屏障 | 是（图 12.8） |
| 图 5 | `v3` §3.2.1 | 8 个 PP rank、双向 20 个 micro-batch 的 DualPipe 调度 | 是（图 12.9，用仿真数据） |
| 表 2 | `v3` §3.2.1 | 1F1B / ZB1P / DualPipe 的气泡与显存 | 是（图 12.10） |
| 图 6 | `v3` §3.3 | FP8 混合精度框架（只画 Linear） | 是（图 12.6） |
| 图 7 | `v3` §3.3.2 | (a) 细粒度量化；(b) 每 $N_C=128$ 提升到 CUDA core | 是（图 12.3、12.5） |
| 图 10 | `v3` 附录 B.1 | 16B 与 230B DeepSeek-V2 上 BF16 vs FP8 损失曲线（EMA 0.9 平滑），右上插图为相对损失差；横轴 Tokens/B（16B 约 0–1330，230B 约 0–900），插图纵轴刻度约 ±0.001 量级（目测） | 可示意，不必精确 |
| 表 1 | `v3` §1 | 成本 | 是（图 12.14） |
| 图 1 | `insights` §1 | V3 架构 + 每个组件的精度 | 是（图 12.7） |
| 图 2 | `insights` §4.1 | H800 节点互连 | 是（图 12.11） |
| 图 3 / 图 4 | `insights` §4.5、§5.1 | 8 平面两层胖树；理想多平面 NIC | 图 3 可重绘（图 12.12） |
| 图 5 / 6 / 7 / 8 | `insights` §5 | NCCL all-to-all、延迟、DeepEP on MPFT、RoCE 路由 | 可选 |
| 表 1–5 | `insights` | KV、FLOPs、拓扑成本、训练指标、延迟 | 表 1 用于图 12.16 |
| Day 6 的图 | `gh-infra` | prefill/decode 重叠、系统图、节点数曲线、成本与理论收入 | 图 12.13、12.15 参考 |

### 5.2 建议本课绘制的 15 张图（含画图所需的全部数据）

颜色遵循 `style-guide.md` §4：FP8/量化数据用 `--fig-orange`；缓存/存储/内存层级 `--fig-amber`；“正在计算”`--fig-blue`；输出/好 `--fig-green`；丢弃/错误/气泡 `--fig-red` 或 `--fig-muted`。

**图 12.1 浮点格式位布局。** 五行横条，每格一位：FP32 [1|8|23]、BF16 [1|8|7]、FP16 [1|5|10]、E5M2 [1|5|2]、E4M3 [1|4|3]；右侧列最大值 / 最小正规 / ε（§3.5 表）。可加第六行 “E5M6 [1|5|6]（V3 自定义，只用于注意力后 Linear 的输入激活）”。

**图 12.2 E4M3 数轴（对数刻度）。** 画全部 126 个正值：正规数 $e=-6..8$、$m=0..7$，$v=(1+m/8)2^{e}$，去掉 $e=8,m=7$；次正规 $v=(m/8)2^{-6}$，$m=1..7$。标出 448、240（不扩展时的最大值）、$2^{-6}$、$2^{-9}$。旁边画 E5M2 对照（$e=-14..15$、$m=0..3$，$v=(1+m/4)2^e$；次正规 $(m/4)2^{-14}$）。要看的：每个 binade 8 个点、间距每过一个 binade 翻倍。

**图 12.3 细粒度量化的 tile 与 block。** 左：激活矩阵 $T\times7168$，每行切成 56 个 1×128 tile，每 tile 一个缩放因子（橙色小方块）；右：一个专家的 gate 权重 $2048\times7168$，切成 $16\times56=896$ 个 128×128 block。下方公式 §2.4。可附一行“反向时激活转成 128×1 tile”。

**图 12.4 离群值的代价（柱状）。** 三根柱，数据来自 §4.7：“tile 内无离群值”中位误差 1.65%；“tile 内有离群值 10”2.26%；“整张量 amax=1000”16.4%（标注 21/127 个值变成 0）。复现配方：`np.random.default_rng(0)`，`x = N(0,1,128)*0.01`，`x[17]=10`，E4M3 最近舍入（§2.6 值表）。

**图 12.5 累加精度提升。** 上半：重绘图 7(b)——Tensor Core 里 WGMMA 1…4（每个沿 K 32 个元素）的低精度累加，满 128 个后箭头送到 CUDA core 的 FP32 寄存器，同时乘缩放因子。下半：§4.12 的误差–$K$ 折线（两条线，横轴 $K$=512…8192 对数刻度），必须标“简化模型示意，非硬件测量”。

**图 12.6 FP8 混合精度数据流（重绘 `v3` 图 6）。** 三个 GEMM 框（Fprop、Dgrad、Wgrad），框内 ⊗ 与 Σ(FP32)；箭头标签：Input BF16 →To FP8；Fprop →To BF16→ Output；Output Gradient BF16 →To FP8 → Dgrad/Wgrad；Dgrad →To BF16→ Input Gradient；Wgrad → Weight Gradient FP32 →To BF16→ Optimizer States →To FP32→ Master Weight →To FP8→ Weight。

**图 12.7 精度地图（据 `insights` 图 1）。** 一个 Transformer 块 + MTP 模块的方框图，每个组件贴精度标签：Embedding/Output Head BF16；RMSNorm、Router、RoPE、Cross-Entropy FP32；注意力与 FFN 中的线性层 FP8；核心注意力 BF16；dispatch FP8；combine BF16；Adam 矩 BF16；主权重与梯度 FP32。

**图 12.8 一对 chunk 的重叠（重绘 `v3` 图 4）。** 两行时间条。计算行依次：MLP(B)▲、MLP(W)▲、MLP(F)△、屏障、ATTN(B)▲、ATTN(W)▲、ATTN(F)△、屏障；通信行：DISPATCH(F)△、DISPATCH(B)▲、屏障、COMBINE(F)△、PP、COMBINE(B)▲、屏障。原图（按 1900px 宽渲染后量的像素宽度，仅示意、不是实测时间）：计算行 170、95、182、10、303、107、150、10；通信行 192、255、10、268、58、234、10。要看的：每一块通信都有一块别的 chunk 的计算“盖住”它。

**图 12.9 DualPipe 调度（重绘 `v3` 图 5，数据来自 `dpsim.py`）。** $P=8$，$C=20$（每方向 10 个），$T_F=1$、完整 $T_B=2$、零气泡模式下 $T_{B_x}=1$、$T_W=1$、$T_{F\&B}=3$，通信 0。记号：`F0.3` = 方向 0 的第 3 个 micro-batch 前向；`B1.2` = 方向 1 第 2 个的完整反向（2 格）；`b0.6` = 只对输入的反向（1 格）；`W` = 权重梯度（1 格）；`F0.7&B1.3` = 重叠的一对（3 格）；`idleN` = 空 N 格。每个 rank 从左到右连续排布（全长 66 格）：

```
rank0: F0.0 F0.1 F0.2 F0.3 F0.4 F0.5 F0.6 F1.0 b1.0 W F1.1 b1.1 W F1.2 b1.2 W F1.3 F0.7&B1.3 idle3 F1.4&B0.0 F0.8&B1.4 F1.5&B0.1 F0.9&B1.5 F1.6&B0.2 B1.6 idle1 F1.7&B0.3 B1.7 idle1 F1.8&B0.4 B1.8 idle1 F1.9&B0.5 B1.9 b0.6 W b0.7 W b0.8 W b0.9 W
rank1: idle1 F0.0 F0.1 F0.2 F0.3 F0.4 F1.0 F0.5 F1.1 b1.0 W F1.2 b1.1 W F1.3 F0.6&B1.2 idle2 F1.4&B0.0 F0.7&B1.3 F1.5&B0.1 F0.8&B1.4 F1.6&B0.2 F0.9&B1.5 F1.7&B0.3 B1.6 idle1 F1.8&B0.4 B1.7 idle1 F1.9&B0.5 B1.8 B0.6 idle1 b1.9 b0.7 W b0.8 W b0.9 W W
rank2: idle2 F0.0 F0.1 F0.2 F1.0 F0.3 F1.1 F0.4 F1.2 b1.0 W F1.3 F0.5&B1.1 idle1 F1.4&B0.0 F0.6&B1.2 F1.5&B0.1 F0.7&B1.3 F1.6&B0.2 F0.8&B1.4 F1.7&B0.3 F0.9&B1.5 F1.8&B0.4 B1.6 idle1 F1.9&B0.5 B1.7 B0.6 idle1 B1.8 b0.7 idle1 b1.9 b0.8 W b0.9 W W W
rank3: idle3 F0.0 F1.0 F0.1 F1.1 F0.2 F1.2 F0.3 F1.3 F0.4 B1.0 F1.4&B0.0 F0.5&B1.1 F1.5&B0.1 F0.6&B1.2 F1.6&B0.2 F0.7&B1.3 F1.7&B0.3 F0.8&B1.4 F1.8&B0.4 F0.9&B1.5 F1.9&B0.5 B1.6 B0.6 idle1 B1.7 B0.7 idle1 b1.8 b0.8 idle1 b1.9 b0.9 W W W W
rank4: idle3 F1.0 F0.0 F1.1 F0.1 F1.2 F0.2 F1.3 F0.3 F1.4 B0.0 F0.4&B1.0 F1.5&B0.1 F0.5&B1.1 F1.6&B0.2 F0.6&B1.2 F1.7&B0.3 F0.7&B1.3 F1.8&B0.4 F0.8&B1.4 F1.9&B0.5 F0.9&B1.5 B0.6 B1.6 idle1 B0.7 B1.7 idle1 b0.8 b1.8 idle1 b0.9 b1.9 W W W W
rank5: idle2 F1.0 F1.1 F1.2 F0.0 F1.3 F0.1 F1.4 F0.2 b0.0 W F0.3 F1.5&B0.1 idle1 F0.4&B1.0 F1.6&B0.2 F0.5&B1.1 F1.7&B0.3 F0.6&B1.2 F1.8&B0.4 F0.7&B1.3 F1.9&B0.5 F0.8&B1.4 B0.6 idle1 F0.9&B1.5 B0.7 B1.6 idle1 B0.8 b1.7 idle1 b0.9 b1.8 W b1.9 W W W
rank6: idle1 F1.0 F1.1 F1.2 F1.3 F1.4 F0.0 F1.5 F0.1 b0.0 W F0.2 b0.1 W F0.3 F1.6&B0.2 idle2 F0.4&B1.0 F1.7&B0.3 F0.5&B1.1 F1.8&B0.4 F0.6&B1.2 F1.9&B0.5 F0.7&B1.3 B0.6 idle1 F0.8&B1.4 B0.7 idle1 F0.9&B1.5 B0.8 B1.6 idle1 b0.9 b1.7 W b1.8 W b1.9 W W
rank7: F1.0 F1.1 F1.2 F1.3 F1.4 F1.5 F1.6 F0.0 b0.0 W F0.1 b0.1 W F0.2 b0.2 W F0.3 F1.7&B0.3 idle3 F0.4&B1.0 F1.8&B0.4 F0.5&B1.1 F1.9&B0.5 F0.6&B1.2 B0.6 idle1 F0.7&B1.3 B0.7 idle1 F0.8&B1.4 B0.8 idle1 F0.9&B1.5 B0.9 b1.6 W b1.7 W b1.8 W b1.9 W
```
每行空闲合计 6 格（= 表 2 公式）。与原图比对：rank0 开头 `0 1 2 3 4 5 6`、随后一个无标号橙格（`F1.0`）、三组绿蓝交替，然后 `7` 的橙绿重叠格、再 3 个白格——与原图一致。原图把方向 1 的 micro-batch 省略编号。

**图 12.10 气泡与显存对比（分组柱状）。** 数据 §4.6 表：$P=8$ 与 $P=16$ 两组，每组三根（1F1B、ZB1P、DualPipe），单位为“$T_F$ 格”；第二个面板画激活显存 $P$ vs $P+1$ 与参数 1× vs 2×。图注必须写清 $T_F=1,T_B=2,T_W=1,T_{F\&B}=3$ 且不计通信。

**图 12.11 H800 节点与跨节点 dispatch 路径（据 `insights` 图 2 + `v3` §3.2.2）。** 节点：2 CPU、8 PCIe Switch（每个下挂 1 CX7 NIC 与 1 H800）、4 个 NVSwitch 芯片全连 8 GPU、CPU2 上 1 块存储 NIC。路径：源 GPU（节点 A，局部编号 3）→ IB（50 GB/s）→ 目标节点 B、C、D、E 上**同为编号 3**的 GPU → NVLink（160 GB/s）扇出到持有目标专家的 GPU（平均 ≤3.2 个）。标注 “每 token ≤ 4 节点”“20/132 SM 做通信”。

**图 12.12 多平面两层胖树（据 `insights` 图 3 + 表 3）。** 8 个平面叠放，每个平面 64 leaf（下连 32 端点）+ 32 spine；节点里第 $i$ 块 GPU/NIC 属于平面 $i$。旁边一张小柱状图：每端点成本 FT2 4.39、MPFT 4.39、FT3 7.5、SF 4.4、DF 5.8 k$。

**图 12.13 推理部署：prefill / decode 单元。** 左：prefill 4 节点 32 GPU，注意力 TP4×DP8（8 个 4-GPU 组），MoE EP32（每 GPU 8 专家 + 1 冗余）。右：decode 40 节点 320 GPU，注意力 TP4×DP80，MoE EP320（256 个 GPU 各 1 路由专家 + 64 个 GPU 放冗余与共享专家）。脚注给 Day 6 的后续配置：prefill EP32 / 每 GPU 9+1；decode EP144（18 节点）/ 每 GPU 2+1。

**图 12.14 成本账（堆叠柱或瀑布图）。** 2664K / 119K / 5K GPU·h → \$5.328M / \$0.238M / \$0.01M；旁注 54.2 天 / 2.4 天 / 0.1 天（2048 GPU）；再用灰色虚框画“不在账内”：先前研究与消融（论文明确），以及人员、集群购置、数据等（论文未提，标“未计入且论文未说明”）。

**图 12.15 Day 6 的一天（柱状 + 注释）。** 成本 \$87,072 vs 理论收入 \$562,027（按 R1 价：\$0.14/M 命中、\$0.55/M 未命中、\$2.19/M 输出）；token：输入 608B（342B 命中）、输出 168B；节点：prefill ≈95.5、decode ≈131.4（§4.9 反推）。必须标“公司自报、理论收入”。

**图 12.16 KV 每 token（柱状，修正倍数）。** 70.272 / 327.680 / 516.096 KB，倍数 1 / 4.66 / **7.34**（在图注里说明原论文印作 7.28x）。

---

## 6. 算例与练习素材

**X1（计算）成本拆账。** 已知表 1 与 2048 块 GPU。问：(a) 总费用；(b) 预训练占几天；(c) 每万亿 token 的 GPU 小时乘以 14.8T 是否等于预训练 GPU 小时？
答：(a) 2788K×2 = \$5.576M；(b) 2664K/(2048×24) = 54.2 天；(c) 14.8×180K = 2664K，精确相等——说明表中预训练数字很可能是由吞吐推算的【推断】。

**X2（计算）从表 4 反推 180K。** 已知 batch 15360×4096 token、每步 19.926 s、2048 GPU。求每天 token 数与每万亿 token 的 GPU·h。
答：62,914,560 × 86400/19.926 = 272.80B/天；10¹²/272.80×10⁹ × 2048 × 24 = 180,176 GPU·h。

**X3（推导+计算）250 GFLOPs/token 从哪来。** 用 §4.3 的参数分解，按 6× 激活参数 + 注意力 $3\cdot L\cdot 2n_h(d_{qk}+d_v)\cdot\bar s$（$d_{qk}=192,d_v=128,\bar s=2048$）。再用 MFU 38.94% 反推 BF16 峰值。
答：219.75 + 30.70 = 250.45 GFLOPs；385/0.3894 = 988.7 TFLOPS。追问：为什么 FP8 训练的 MFU 按 BF16 峰值算会“偏高”？（按 FP8 峰值约 2×，MFU 约减半。）

**X4（概念+计算）E4M3 的 448 与 240。** 写出 E4M3 最大码字 $S.1111.110_2$ 的值；若按 IEEE 惯例把指数全 1 留给 Inf/NaN，最大值是多少？一共多少个有限值？
答：1.75×2⁸ = 448；(1+7/8)×2⁷ = 240；254 个有限值（±0 各算一个）+ 2 个 NaN。

**X5（计算）量化一个 tile。** tile 的 amax = 3.5，求缩放因子；元素 0.0123 量化后是多少、相对误差多少？
答：s = 3.5/448 = 0.0078125（恰为 2⁻⁷）；0.0123/s = 1.5744 → E4M3 在 [1,2) 的间距 0.125 → 1.625（1.5744 距 1.5 为 0.0744、距 1.625 为 0.0506）→ 反量化 1.625×0.0078125 = 0.012695；相对误差 3.2%。（可让学生再对比 amax = 1000 时：s = 2.232，0.0123/s = 0.00551 → 落在次正规区，间距 2⁻⁹ = 0.00195 → 量化为 3×2⁻⁹ = 0.00586 → 反量化 0.01308，误差 6.3%；更小的值直接变 0。）

**X6（计算）缩放因子的开销。** 权重 128×128 块、激活 1×128 tile、缩放因子 FP32。求两者的额外存储比例；全模型有多少个权重缩放因子？
答：4/16384 = 0.0244%；4/128 = 3.125%；41,540,496 个（166 MB），与 HF 检查点 F32 元素数减去门控偏置后完全一致。

**X7（计算）PP=16 的气泡。** 取 $T_F=1,T_B=2,T_W=1$。(a) 三种方法的气泡；(b) $T_{F\&B}=2.5$ 时 DualPipe 的气泡；(c) 32 个 micro-batch 时 1F1B 与 DualPipe 的气泡占比。
答：(a) 45、15、14；(b) 7×(2.5+2−3) = 10.5；(c) 45/(96+45) = 31.9%；14/(96+14) = 12.7%。追问：为什么在“计算不因重叠变快”（$T_{F\&B}=T_F+T_B$）时 DualPipe 只比 ZB1P 少 1 格？（公式里系数减半，但括号里多了一个 $T_B$；DualPipe 的主要收益是藏通信，见 §7 E5。）

**X8（估算）DualPipe 的 2× 参数为什么可以接受。** 已知 EP64、每 chunk 4 个 MoE 层。估算每 GPU 一段的参数量，并说明两份参数的代价。
答（笔者估算）：每层每 GPU 4 个路由专家（176.2M）+ 共享专家（44.0M）+ 注意力（187.1M）+ 门控（1.8M）≈ 409M；4 层 ≈ 1.64B；两段 ≈ 3.27B 参数，BF16 约 6.5 GB——与 80 GB HBM 相比不大，正是论文 “since we use a large EP size” 的意思。

**X9（计算）节点受限路由。** NVLink 160 GB/s、IB 50 GB/s。(a) 一次 IB 发送能“顺带”在节点内扇出几个专家？(b) 4 个节点最多多少个专家？(c) 8 个目标专家分散在 8 个节点 vs 限制在 4 个节点，IB 时间之比？
答：(a) 3.2；(b) 12.8 → 13；(c) 8t : 4t = 2:1。

**X10（估算）为什么 prefill 最小单元是 4 节点、decode 是 40 节点？**
答（部分为推理）：FP8 权重 671 GB > 一个节点 640 GB HBM，所以至少 2 节点才能放下权重；4 节点（2560 GB）才给 KV 与激活留余地；256 路由 + 32 冗余 = 288 个专家摊到 32 GPU 恰为每 GPU 9 个。decode 要让每个专家独占一块 GPU（访存受限，每 GPU 只读一个专家的权重）：256 + 64 = 320 块 GPU = 40 节点；此时每 GPU 专家权重约 2.55 GB，大部分 HBM 留给 KV cache（按 70,272 B/token，70 GB 约容纳 100 万 token）。

**X11（计算）Day 6 的 545%。** 由 226.75 节点、\$2/GPU·h 求日成本；由 token 数和 R1 价格求理论收入与利润率；再用每节点吞吐反推 prefill/decode 节点数。
答：\$87,072；\$562,100（原文 \$562,027）；545%；95.5 + 131.4 = 226.9 节点。讨论：这是“若全部按 R1 价计费”的理论值，公司自己说实际收入低得多。

**X12（计算）TPOT 上界。** 按 `insights` §2.3.2 的式子算 50 GB/s 与 900 GB/s 下的 TPOT；若 hidden 用 7168 而不是 7000 会怎样？
答：14.76 ms（67.8 tok/s）；0.82 ms（1,220 tok/s）；7168 时单次 123.86 µs、TPOT 15.11 ms（66.2 tok/s）。

**X13（纠错）KV 倍数。** 用 `config.json` 复算 `insights` 表 1 的三个 KV 数，并检查倍数列。
答：70,272 / 327,680 / 516,096 B；4.66x 正确，7.28x 应为 7.34x。

**X14（推导）LogFMT 的步长。** $n=8$ 时码字数与 $Step$ 的分母；在约束 $\ell_{\max}-\ell_{\min}<32\ln2$ 下最坏情况的相邻比值；对比 E4M3 在一个 binade 内的相对间距。
答：$K=1..127$，分母 126；最坏步长 0.176，相邻比 1.19；E4M3 为 1/16–1/8。说明：LogFMT 的优势来自 tile 自适应的范围，范围越窄越精细。

---

## 7. 勘误与常见误解

**E1（论文勘误）`insights` 表 1 的 7.28x 算错了。** 516.096/70.272 = **7.344**，不是 7.28（v1、v2 都印 7.28x）。4.66x 正确。三个 KV 数本身都能由 config 精确复算（§4.10）。
**E2（任务说明中的易错点）** 516 KB 是 **LLaMA-3.1 405B**，327 KB 是 **Qwen-2.5 72B**，不要对调。原文：“LLaMA-3.1 405B’s 516 KB and Qwen-2.5 72B’s 327 KB”。
**E3（版本漂移）Tensor Core 累加位数的三种写法。** `v3` v1：“fixed-point accumulation … highest 14 bits … after sign-fill right shifting”，并称累加 32 个 FP8 乘积要精确到 FP32 “at least 34-bit precision is required”；`v3` v2 删去了 34 位那句，改为“对齐 32 个乘积后取最高 14 位，寄存器累加也是 14 位”；`insights` 写 “13 fraction bits” 与 “FP22 (1+8+13)”。调和（笔者推断）：14 位有效位 = 1 位前导 + 13 位小数。正文宜写“约 14 位有效位（13 位小数）”，并注明来源分歧。
**E4（常见误解）“FP8 训练 = 所有计算都是 FP8”。** 实际上只有线性层（及 MoE 专家）的三个 GEMM 输入是 FP8；核心注意力 BF16，嵌入、输出头、门控、归一化保持原精度，主权重与梯度 FP32，Adam 两个矩 BF16，combine 通信 BF16（§3.3）。
**E5（常见误解）“DualPipe 把气泡减半”。** 公式里只是系数从 $P-1$ 变成 $P/2-1$，括号却变成 $T_{F\&B}+T_B-3T_W$。在“重叠不让计算变快”的模型里（$T_{F\&B}=T_F+T_B$，$T_F=T_W=1,T_B=2$），$P=16$ 时 DualPipe 14 格 vs ZB1P 15 格，几乎一样；DualPipe 的主要价值是把约 1:1 的 all-to-all 通信藏在另一个 chunk 的计算后面（C13–C15），以及通信开销在这个框架下才被计进 $T_{F\&B}$。论文“significantly reduces the pipeline bubbles”的比较对象是 1F1B 与 ZB1P 的**同一张表**，正文应让读者自己看到这层含义。【笔者分析】
**E6（记号陷阱）`v3` 的 $B$ ≠ `zb` 的 $T_B$。** V3 的 $B$ 是完整反向；ZB 论文的 $T_B$ 只是对输入的反向。换算后 V3 表里的 “ZB1P” 公式正是 ZB 论文 **ZB-H1**（手工调度）的公式；ZB 论文里真正叫 ZB-1p 的是自动搜索调度，没有闭式公式（§2.1）。第三套记号：`insights` 表 4 的 “1B” 定义为 “input backward time”，与 ZB 的 $T_B$ 同义、与 `v3` 的 $B$ 不同。
**E7（论文 vs 代码）DualPipe 的整除条件。** 论文只要求 stage 数与 micro-batch 数能被 2 整除；代码还断言 `num_chunks >= num_ranks * 2`（例如 8 个 rank 至少 16 个 micro-batch；论文图 5 用 20 个）。
**E8（公式的适用范围）表 2 的 DualPipe 公式在 $T_W\neq T_F$ 时不精确。** 本笔记按开源代码的操作顺序做无通信仿真：$T_F=T_W=1$ 时与公式完全吻合；$T_W=0.5$ 时仿真 9.5 格、公式 10.5 格；$T_W=0.8$ 时 7.2 vs 7.8。正文引用表 2 时写“在各 chunk 时间满足一定比例时”的估计式，不要当定理。
**E9（常见误解）“DeepSeek-V3 只花了 557.6 万美元”。** (1) 论文明说只含正式训练，不含先前研究与消融（C4）；(2) \$2/GPU·h 是**假设的租价**，不是实际支出；(3) 集群购置、人员、数据、R1 系列模型的训练（V3 后训练从 R1 蒸馏）论文都没有计入也没有讨论；(4) 预训练 GPU 小时恰等于 14.8×180K，像是由吞吐推算而非计量（推断）。正文写“正式训练一次的 GPU 租用等价成本”。
**E10（常见误解）“H800 是算力被砍的 H100”。** `insights` §4.1 只说 H800 降低了 FP64 性能和 NVLink 带宽（900→400 GB/s）；由表 4 的 MFU 反推的 BF16 稠密峰值约 988 TFLOPS，与 H100 SXM 常见数值一致（H100 数值本身未在此核实）。V3 的许多设计（不用 TP、节点受限路由、20 SM 通信内核）都是冲着**互连**瓶颈去的。
**E11（口径）MFU 按 BF16 峰值算。** `insights` 表 4 明写 “calculated based on BF16 peak performance”；FP8 GEMM 的理论峰值约为其 2 倍，所以 38.94% 不是“FP8 利用率”。
**E12（部署数字漂移）** 同一个“V3 推理部署”，论文（2024-12）是 decode EP320 / 40 节点、注意力 TP4+SP；Day 6（2025-02）是 decode EP144 / 18 节点、注意力 DP144；profile-data 是 decode EP128、TP1。正文必须写明时间点，不要混用。prefill 论文是 TP4+SP×DP8，Day 6 与 profile 都是 “DP32 / TP1”。
**E13（论文 vs 代码）通信 SM 数。** 论文 20 个 SM；DeepEP V1 文档示例 `Buffer.set_num_sms(24)`；DeepEP V2 README 说 V3 式训练 “SM usage reduced from 24 to 4 - 6”。DeepEP V1 还自述 “may have some slight differences from the DeepSeek-V3 paper”。
**E14（常见误报）“DeepSeek 利润率 545%”。** 这是公司在 Day 6 文中按“所有 token 都按 R1 价计费”的**理论**利润率，同文承认实际收入 “substantially lower”。只能写成“DeepSeek 自称的理论成本利润率”。
**E15（两处论文不一致）MTP 接受率。** `v3` §5.4.3：“between 85% and 90%”；`insights` §2.3.3：“80% to 90%”。两者都说 1.8× TPS。
**E16（措辞差异）0.25%。** `v3`：“relative loss error … below 0.25%”；`insights` §2.4：“relative accuracy loss … below 0.25%”。前者是训练损失的相对误差，后者措辞更模糊；引用时用 `v3` 的说法。另外 `v3` §3.3 说“约 1 万亿 token”，附录 B.1 细到 16B 模型 1.33T、230B 模型约 0.9T。
**E17（制品内部不一致）685B 的拆分。** README：671B + 14B MTP；README_WEIGHTS：MTP “11.5B unique parameters”。本笔记从 config 精确重建：MTP 独有 11.61B，加上检查点里重复存储的嵌入与输出头共 13.46B，合计 684.49B。另外 README_WEIGHTS 的 “激活 36.7B（含 0.9B 嵌入 + 0.9B 输出头）” 无法精确复现：不含嵌入为 36.62B，含嵌入为 37.55B；MTP 激活 “2.4B” 也无法复现（1.61B / 2.54B）。
**E18（易误读）index.json 的 `total_size` 不是下载大小。** 1,369,062,772,000 恰为参数数的 2 倍；真实文件合计 688.59 GB（FP8 权重 1 B/个）。
**E19（近似）`insights` 的 “7K”。** 公式按 7000 算出 120.96 µs；V3 hidden 实为 7168（123.86 µs）。v1 还把 NVL72 结论写成 “over 0.82 ms”，v2 改正为 0.82 ms 并加上“小 batch 效率下降”的免责。
**E20（图文不一致）** `insights` 图 1 在 combine 处标 “BF16/LogFMT”，但 §3.2.1 说 LogFMT “we do not employ it eventually”。
**E21（小写/大写）** `v3` §3.3 引言印 $N_c$，§3.3.2 与图 7 印 $N_C$；是同一个量（=128）。正文统一用 $N_C$。
**E22（附录里的负面结果，常被忽略）** 激活梯度按 128×128 块量化会让 16B MoE 在约 300B token 时发散（附录 B.2），这正是激活必须用 1×128 / 128×1 tile 的原因；原因解释是作者的假说（“We hypothesize”）。
**E23（config 与论文）** `initializer_range` 0.02 vs 论文 std 0.006（与本课无关，但记录在案以免第 11 课误用）。

---

## 8. 未能核实 / UNVERIFIED

1. **H800 官方规格**：NVIDIA 原始数据表（FP8/BF16 稠密 TFLOPS、HBM 容量 80 GB 与带宽 3.35 TB/s）本次没读到一手来源（经销商托管的 PDF 返回 404，只看到二手网页）。正文里的“约 988 TFLOPS BF16”只能写成“由 `insights` 表 4 的 MFU 反推”。80 GB 是本笔记计算显存时的假设。
2. **400 GB/s 与 200 GB/s 的关系**：`insights` §4.1 说 H800 NVLink 400 GB/s，§4.3 又说 200 GB/s（实测约 160）。“400 是双向合计、200 是单向”是笔者推断，论文没有明说。H100 的 “900 GB/s” 也只引自 `insights`。
3. **`insights` 表 4 各行的统计口径**：论文只定义了 “1F, 1B, and 1W denote forward time, input backward time, and weight backward time, respectively”（§5.1.2 表 4 前一段）；**没有**说明它们是一步里各阶段的累计耗时还是单个 chunk 的耗时，bubble、1F1B、opt 三行也没有定义。六行相加 = 19.90 s ≈ 每步时间，支持“阶段耗时”的解读，但未确认。（原稿此条写成“1F/1B/1W 论文没有定义”，与原文不符，已改。）
4. **V3 训练的 micro-batch 数与 micro-batch 大小**：未公开，所以无法用真实 $C$ 计算 V3 的气泡占比；只有表 4 的实测 10.3%。
5. **HAI-LLM**：没有找到公开论文或代码。Fire-Flyer 论文里只有 HAI Platform（调度平台），不是 HAI-LLM。
6. **“maximum relative error of nearly 2%”的度量口径**（逐元素最大？范数？）与测试矩阵分布未公开；§4.12 只是笔者的简化模型。FP22 的原始出处 (Zhang et al., 2025) 本次未阅读。
7. **`v3` v1 的“至少 34 位”**：未独立推导核对，v2 已删除。
8. **Zero Bubble 论文的会议出处**（常见说法为 ICLR 2024）未在本次核实；arXiv 全文中无 “ICLR” 字样。
9. **DualPipeV** 的数字只来自 README，没有论文；Sea AI Lab 的 “Cut-in-half” 博文未读。
10. **GPU 小时是计量值还是推算值**：14.8×180K 恰等于 2664K 只是强烈暗示，论文没说。
11. **DeepSeek 集群的所有权与购置成本**：论文只给“假设租价”，任何资本开支数字（媒体估计等）都不在一手来源中，不得写入。
12. **EPLB 示例输出**：README 给出的 `phy2log` 没有在本机重跑（无 torch）。
13. **README_WEIGHTS 的 36.7B / 2.4B 激活参数**：无法由 config 精确复现（§7 E17），口径不明。
14. **Day 6 的经济数据**：公司自报、未经审计；理论收入按 R1 价计算。
15. **2026 年的线上部署现状**：本笔记的部署数字只到 2025-02（Day 6）与 2025-02 的 profile-data；之后是否变化未知。
16. **DeepGEMM 1350+/1550 TFLOPS、FlashMLA 580/660 TFLOPS、3000 GB/s、DeepEP 带宽**：均为公司自测，未复现；相对“理论峰值”的百分比取决于未核实的 H800 峰值。
17. **`insights` 图 7 在 16 GPU 处 43.05 与 42.47 分别属于 dispatch 还是 combine**：图上无法判定。
18. **`v3` 图 10 插图的纵轴数值**：只能目测（约 ±0.1% 量级），不要写成精确数字。
19. **`kernel.py` 的 `ue8m0` 分支何时加入**：未查 git 历史；只用于第 15 课的预告。
20. **E5M6 的具体编码细节（偏置、有无次正规、特殊值）**：论文未给。
21. **Chimera、PipeDream、ZeRO 的正文**：本次只读了元数据或 grep 片段，没有逐节核对；它们只作为“被 V3 引用的背景”出现。
22. **H800 为何只有“2048 张”**：`insights` §5.1 说 “due to policy and regulatory constraints, just over two thousand GPUs were ultimately deployed”；更多背景（出口管制细节）不在一手来源中。

---

## 9. 参考文献卡片草稿

```
- id: v3-report
  title: DeepSeek-V3 Technical Report
  a: DeepSeek-AI (2024), technical report (arXiv preprint), arXiv:2412.19437 (v2 read; §3 diffed against v1)
  url: https://arxiv.org/abs/2412.19437
  finding: 671B/37B 的 MoE 模型在 2048 张 H800 上用 FP8 细粒度量化、DualPipe 与 20-SM 的跨节点 all-to-all 完成训练，正式训练共 2.788M GPU 小时（按 $2/小时合 $5.576M）。
  matters: 本课几乎所有工程数字（并行配置、FP8 配方、表 2 气泡公式、prefill/decode 部署单元、硬件建议）都以这份报告第 3 节为唯一一手来源。

- id: insights-isca25
  title: "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures"
  a: Zhao, Deng, Ruan, Dai, Gao, Li, Zhang, Huang, Zhou, Ma, Liang, He, Wang, Liu, Wei (2025), ISCA '25 Industry Track, arXiv:2505.09343 (v2 read; v1 diffed)
  url: https://arxiv.org/abs/2505.09343
  finding: 从软硬件协同的角度复盘 V3：给出 KV/FLOPs 对比表、MoE 推理速度的通信上界、FP22 累加精度、LogFMT 试验、H800 节点结构、多平面两层胖树及其训练吞吐实测。
  matters: 它提供了 V3 报告里没有的实测吞吐（272.8B token/天、MFU）与网络拓扑细节，使本课能把 180K GPU·h/万亿 token 与 250 GFLOPs/token 互相对账；同时它的表 1 倍数有错，需在课中更正。

- id: fp8-formats
  title: FP8 Formats for Deep Learning
  a: Micikevicius, Stosic, Burgess, Cornea, Dubey, Grisenthwaite et al. (2022), arXiv preprint, arXiv:2209.05433 (v2 read)
  url: https://arxiv.org/abs/2209.05433
  finding: 定义 E4M3（偏置 7，最大 448，不表示无穷）与 E5M2（偏置 15，最大 57,344，遵循 IEEE 特殊值）两种 FP8 编码，并建议权重/激活用 E4M3、梯度用 E5M2。
  matters: 本课的位布局图、448 这个缩放常数的来历，以及“V3 全部张量都用 E4M3、偏离了 E5M2 给梯度的推荐”这一对比都依赖它。

- id: zero-bubble
  title: Zero Bubble Pipeline Parallelism
  a: Qi, Wan, Huang, Lin (2023), arXiv preprint, arXiv:2401.10241 (v1 read)
  url: https://arxiv.org/abs/2401.10241
  finding: 把反向拆成对输入的 B 与对权重的 W，W 可延后填补气泡；给出 1F1B、ZB-H1、ZB-H2 的气泡与峰值激活闭式表，以及自动搜索的 ZB-1p/ZB-2p。
  matters: DualPipe 的“backward for input / backward for weights”拆分直接来自它；换算记号后可证明 V3 表 2 的 “ZB1P” 行就是 ZB-H1 的公式。

- id: microscaling
  title: Microscaling Data Formats for Deep Learning
  a: Rouhani, Zhao, More, Hall, Khodamoradi, Deng et al. (2023), arXiv preprint, arXiv:2310.10537 (v3, grep only)
  url: https://arxiv.org/abs/2310.10537
  finding: MX 格式让每 32 个元素共享一个 E8M0（纯指数）缩放因子。
  matters: V3 自称其 1×128/128×128 细粒度量化与 microscaling “highly consistent”，而 E8M0 缩放正是第 15 课 UE8M0 的前身，本课用它做前后衔接。

- id: zero
  title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
  a: Rajbhandari, Rasley, Ruwase, He (2019), arXiv preprint, arXiv:1910.02054 (v3, grep only)
  url: https://arxiv.org/abs/1910.02054
  finding: 混合精度 Adam 每参数需 16 字节（2+2+12），ZeRO-1（P_os）把优化器状态按数据并行切分，带来约 4 倍显存节省。
  matters: 本课用 16 字节/参数作为基线，对比 V3 用 BF16 存 Adam 矩、FP32 存主权重与梯度后的约 12 字节/参数。

- id: chimera
  title: "Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines"
  a: Li, Hoefler (2021), SC'21, arXiv:2107.06925 (metadata only)
  url: https://arxiv.org/abs/2107.06925
  finding: 提出双向流水线：两条方向相反的流水线共用同一组设备以减少气泡。
  matters: V3 明确把 DualPipe 与 Chimera 比较，称其只要求 stage 数与 micro-batch 数被 2 整除，本课用它说明双向调度并非 DualPipe 首创。

- id: pipedream
  title: "PipeDream: Fast and Efficient Pipeline Parallel DNN Training"
  a: Harlap, Narayanan, Phanishayee, Seshadri, Devanur, Ganger et al. (2018), arXiv preprint, arXiv:1806.03377 (metadata only)
  url: https://arxiv.org/abs/1806.03377
  finding: 在异步设定下首次提出 1F1B（一前一后交替）流水线调度。
  matters: V3 表 2 的 1F1B 行引用它作为基线，本课的气泡对比从这条基线讲起。

- id: fire-flyer
  title: "Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning"
  a: An et al. (2024), SC24, arXiv:2408.14158 (v2, grep only)
  url: https://arxiv.org/abs/2408.14158
  finding: 描述 DeepSeek 早期基于 PCIe A100 的 Fire-Flyer 2 集群、HFReduce、HAI Platform 与 3FS 的协同设计。
  matters: 只用来说明 3FS 与集群协同设计的前史，并澄清 HAI Platform 不是 V3 报告里的 HAI-LLM 训练框架。

- id: gh-dualpipe
  title: DualPipe (GitHub repository)
  a: DeepSeek-AI (2025), open-source code, no paper; README + dualpipe/dualpipe.py read 2026-09-24
  url: https://github.com/deepseek-ai/DualPipe
  finding: 开源 DualPipe 与 DualPipeV 的 PyTorch 实现，README 给出含每设备参数/激活与设备数的气泡对比表。
  matters: 本课的调度图数据由其 `step()` 八个阶段的操作顺序仿真得到，并用它发现“论文只要求整除 2、代码要求 micro-batch ≥ 2×rank”这一差异。
  kind: artefact

- id: gh-deepep
  title: DeepEP (GitHub repository)
  a: DeepSeek-AI (2025), open-source code, no paper; README (V2) + docs/legacy.md (V1) read 2026-09-24
  url: https://github.com/deepseek-ai/DeepEP
  finding: MoE dispatch/combine 通信库；V1 在 H800 上跨节点 EP16–64 实测 43–58 GB/s，低延迟内核 EP8–256 dispatch 77–194 µs；V2 把 V3 式训练的通信 SM 从 24 降到 4–6。
  matters: 它是 V3 §3.2.2 通信内核的开源对应物，本课用它的数字说明“50 GB/s 的 IB 基本跑满”，并记录 20 vs 24 SM 的出入。
  kind: artefact

- id: gh-deepgemm
  title: DeepGEMM (GitHub repository)
  a: DeepSeek-AI (2025), open-source code, no paper; README read 2026-09-24
  url: https://github.com/deepseek-ai/DeepGEMM
  finding: 细粒度缩放的 FP8 GEMM 库，开源时宣称 Hopper 上 1350+ FP8 TFLOPS，2025-04 更新为 H800 上最高 1550 TFLOPS；SM90 用 FP32 缩放因子。
  matters: 它把 V3 §3.3.2 的“分组缩放 + CUDA core 提升累加”落成可读代码，是学生动手查看 FP8 GEMM 的入口。
  kind: artefact

- id: gh-flashmla
  title: FlashMLA (GitHub repository)
  a: DeepSeek-AI (2025), open-source code, no paper; README read 2026-09-24
  url: https://github.com/deepseek-ai/FlashMLA
  finding: MLA 解码内核，开源时在 H800 上 3000 GB/s（访存受限）与 580 TFLOPS（计算受限），分页 KV 块大小 64，后更新到 660 TFLOPS。
  matters: 本课用它说明 decode 是访存受限、以及 MLA 的小 KV（70 KB/token）如何落到内核层面；它也衔接第 9 课与第 14 课。
  kind: artefact

- id: gh-eplb
  title: EPLB (GitHub repository)
  a: DeepSeek-AI (2025), open-source code, no paper; README read 2026-09-24
  url: https://github.com/deepseek-ai/EPLB
  finding: 冗余专家复制与装箱算法，分层式（prefill、较小 EP）与全局式（decode、较大 EP）两种策略，负载预测不在库内。
  matters: 它是 V3 §3.4 “redundant experts” 的开源实现，本课用它讲清冗余专家如何在不增加跨节点流量的前提下均衡 GPU 负载。
  kind: artefact

- id: gh-3fs
  title: Fire-Flyer File System (3FS) (GitHub repository)
  a: DeepSeek-AI (2025), open-source code, no paper; README read 2026-09-24
  url: https://github.com/deepseek-ai/3FS
  finding: 基于 SSD 与 RDMA 的分布式文件系统，180 节点聚合读 6.6 TiB/s，KVCache 客户端峰值 40 GiB/s。
  matters: Day 6 中 56.3% 的输入 token 命中“磁盘 KV cache”，3FS 就是这层存储；它也为第 19 课的 KV 存储层级铺路。
  kind: artefact

- id: gh-profile
  title: profile-data (GitHub repository)
  a: DeepSeek-AI (2025), profiling traces, no paper; README read 2026-09-24
  url: https://github.com/deepseek-ai/profile-data
  finding: 公开训练（EP64、TP1、每 chunk 4 个 MoE 层）、prefill（EP32、TP1）与 decode（EP128、TP1）的 PyTorch profiler 轨迹，且路由被模拟为绝对均衡。
  matters: 它是 DualPipe 重叠与推理双 micro-batch 重叠的可视证据，同时暴露了与论文部署数字（EP320）的漂移。
  kind: artefact

- id: gh-infra-day6
  title: "Day 6 - One More Thing: DeepSeek-V3/R1 Inference System Overview (open-infra-index)"
  a: DeepSeek-AI (2025, Open-Source Week day 6; statistics window ends 2025-02-28 12:00 UTC+8; exact publication date not verified), company blog in GitHub repo, no paper; read 2026-09-24
  url: https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
  finding: 线上 V3/R1 采用 PD 分离，prefill EP32、decode EP144，平均 226.75 个 H800 节点、日成本 $87,072，按 R1 价计算的理论日收入 $562,027、成本利润率 545%。
  matters: 本课用它算一笔完整的推理经济账，并示范如何把公司自报的理论数字与真实收入区分开。
  kind: artefact

- id: gh-v3-repo
  title: DeepSeek-V3 (GitHub repository: README, README_WEIGHTS.md, inference/kernel.py)
  a: DeepSeek-AI (2024-12), model release, no separate paper; read 2026-09-24
  url: https://github.com/deepseek-ai/DeepSeek-V3
  finding: 只发布 FP8 权重（E4M3、128×128 块、FP32 `weight_scale_inv`），HF 总量 685B = 671B 主模型 + 14B MTP；参考推理代码以 amax/448 做每 128 通道的在线激活量化。
  matters: 本课的量化公式（§2.3）取自其代码而非论文，并用它与 HF 检查点逐元素对账，证明“哪些组件保持高精度”在发布的权重里确实如此。
  kind: artefact

- id: hf-v3
  title: deepseek-ai/DeepSeek-V3 on Hugging Face (config.json, model.safetensors.index.json, API)
  a: DeepSeek-AI (2024-12-25 created, 2025-03-27 last modified), model card and files; read 2026-09-24
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3
  finding: FP8 E4M3 元素 680,571,043,840 个、BF16 3,918,786,560 个、F32 41,555,600 个，163 个文件共 688.59 GB。
  matters: 这些精确计数让本课能用架构超参数逐项复算出检查点的每一类张量，是“先算账”方法的最佳示范。
  kind: artefact

- id: hf-cmp-configs
  title: Qwen2.5-72B and Llama-3.1-405B config.json
  a: Qwen team / Meta (2024), model configs (Llama read via RedHatAI and unsloth mirrors because the meta-llama repo is gated); read 2026-09-24
  url: https://huggingface.co/Qwen/Qwen2.5-72B ; https://huggingface.co/RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8
  finding: Qwen2.5-72B 为 80 层、64 头、8 个 KV 头、hidden 8192；Llama-3.1-405B 为 126 层、128 头、8 个 KV 头、hidden 16384。
  matters: 它们是复算 `insights` 表 1（327,680 与 516,096 字节/token）并发现 7.28x 应为 7.34x 的依据。
  kind: artefact
```

---

## 10. 核查记录（2026-09-24，独立复核）

复核方法：自己重新下载一手来源，不沿用研究稿的缓存。`arxiv.py text` 取 `v3` v2 与 v1、`insights` v2 与 v1、`zb` v1、`fp8fmt` v2 的 LaTeXML 全文后 grep；`mx`、`zero` 用 `arxiv.py grep`；GitHub README/源码与 HF `config.json`、API 用 raw 地址重新下载；`v3` 图 4（overlap.svg，用无头 Chrome 渲染）、图 6，以及 `insights` 图 1 都重新看了原图；数字用 Python 复算；DualPipe 调度用**自己另写的**仿真器（按 `dualpipe.py` `step()` 的 8 步操作顺序，不计通信）重跑。

**核对后确认无误（24 项）**
1. `v3` 表 1：2664K / 119K / 5K / 2788K；\$5.328M / \$0.238M / \$0.01M / \$5.576M；表题原文。
2. C2–C5：180K GPU·h/万亿 token、“3.7 days”、“less than two months”、只排除“prior research and ablation experiments”、“0.1M GPU hours”。复算 180e3/2048/24 = 3.662 天，14.8×180K = 2664K。
3. C9–C12：16 路 PP、64 路 EP 跨 8 节点、ZeRO-1、不用 TP；batch 3072→15360（前 469B token）；64 GPU/8 节点摆放路由专家。
4. `v3` 表 2 与 DualPipe README 表，逐符号：$(PP-1)(F+B)$、$(PP-1)(F+B-2W)$、$(\frac{PP}{2}-1)(F\&B+B-3W)$、$1\times/2\times$、$PP/PP+1$，DualPipeV 的 $PP/2$ 设备数；表题定义（$B$ 为 full backward）。
5. `zb` 表 2 逐符号（1F1B $(p-1)(T_F+T_B+T_W)$；ZB-H1 $(p-1)(T_F+T_B-T_W)$，$pM_B$；ZB-H2 $(p-1)(T_F+T_B-2T_W)$，$(2p-1)M_B$）、“$T_W<T_F<T_B$ and $T_B+T_W=2T_F$”、ZB-1p 为自动搜索；换算 $T_{B_x}=B-W$ 后 V3 的 “ZB1P” 式 = ZB-H1 式，代数成立。
6. C24–C29：160 vs 50 GB/s、“roughly 3.2 times”、≤4 节点、3.2 专家/节点、13 专家、20 SM / 10 通道、20/132。
7. C35、C41、C44、C46、C47：$1\times N_c$（小写）、1x128 / 128x128、“around 14 bits”、K=4096 “nearly 2%”、$N_C=128$ = 4 WGMMAs、E4M3 用于全部张量。
8. `v3` 第 3 节 v1→v2 漂移：自己 diff 两版全文，第 3 节确实只有 §3.5.2 首段被改写（v1 的 “fixed-point accumulation”“at least 34-bit precision is required” 被删）与 “entirely -utilized” 错字。
9. C50–C52：BF16 存 AdamW 两个矩、FP32 主权重与梯度、E5M6、2 的整数幂缩放、combine 保持 BF16。
10. C36、C49：<0.25%；附录 B.1 的 16B/1.33T 与 230B/0.9T；B.2 的 16B/300B 发散与 “We hypothesize”。
11. C53–C57：prefill 4 节点 32 GPU、TP4+SP×DP8、EP32、32 冗余、每 GPU 8+1；decode 40 节点 320 GPU、DP80、EP320、64 GPU 放冗余与共享、IBGDA、“usually within 256 tokens”；“more than two times that of DeepSeek-V2”。
12. C58 Day 6：EP32/4 节点、EP144/18 节点、9+1 与 2+1、5 级流水、226.75 节点、278 峰值、\$87,072、608B/342B(56.3%)/168B、20–22 tok/s、4,989、73.7k/14.8k、\$562,027、545%、“substantially lower”、R1 价格。复算 \$87,072、\$562,100、545.5%、95.5+131.4 节点均无误。
13. `insights` 表 1：70.272 / 327.680 / 516.096 KB，4.66x / 7.28x（v1、v2 相同）；复算 516,096/70,272 = 7.344，**确认 E1 勘误成立**；用 Qwen2.5-72B（80 层、8 KV 头、hidden 8192、64 头）与 Llama-3.1-405B（126 层、8 KV 头、hidden 16384、128 头）的 `config.json` 复算三个 KV 数全部精确吻合。
14. `insights` §3.1.1：“highest 13 fraction bits”、FP22 = 1+8+13；v2 新增 (Zhang et al., 2025) 与 “follows the naming used in the cited work”。
15. `insights` 表 4 全部 24 个数、“calculated based on BF16 peak performance”；复算 272.80B token/天、180,176 GPU·h/万亿 token、386.1 / 433.4 TFLOPS、推得 BF16 峰值 988.7 / 987.9。
16. `insights` 表 2（155 / 250 / 394 / 2448 “GFLOPS/Token”，序列长 4096）、表 3、表 5、§4.1（900→400 GB/s、八张 400G CX7）、§4.3（200 / 160 / 40 GB/s、“approximately 4:1”、8 组×32 专家）、§5.1（64 口交换机、16,384、“just over two thousand”）。
17. `fp8fmt` 表 1 逐项（偏置 7/15、Inf/NaN 码字、448、57,344、$2^{-6}$、$2^{-14}$、$0.875\cdot2^{-6}$、$0.75\cdot2^{-14}$、$2^{-9}$、$2^{-16}$）、“240”、“17 to 18 binades”、“32 binades”、推荐用法、饱和处理；E4M3 值表复算为 127 个非负有限值、最大 448。
18. HF 检查点：API 给出 F8_E4M3 680,571,043,840、BF16 3,918,786,560、F32 41,555,600、163 个文件 688,586,727,753 B、index `total_size` 1,369,062,772,000 = 2×684,531,386,000；用 config 自己重建 FP8 / BF16 / 缩放块数（41,540,496 + 15,104 门控偏置）全部精确吻合；主模型 671,026,419,200，MTP 独有 11.610B、含重复嵌入/输出头 13.463B。README “685B = 671B + 14B”与 README_WEIGHTS “11.5B”“36.7B”“2.4B”原文确认。
19. `dualpipe.py`：`assert num_ranks % 2 == 0` 与 `num_chunks >= num_ranks * 2`；§2.2 八步的次数公式与 phase 约定注释、中间 rank 第一轮不重叠的注释，全部与源码一致。
20. DualPipe 仿真（本次独立重写）：P=8,C=20 每个 rank 空闲 6.0（makespan 66）；$T_{F\&B}=2.5$ 时最大空闲 4.5；P=16 时 C=32 与 C=64 空闲都是 14.0（makespan 110 / 206）；$T_W=0.8$ 为 7.2（公式 7.8），$T_W=0.5$ 为 9.5（公式 10.5）；P=8,C=16（代码允许的最小值）也是 6.0。与 §4.6、E8 完全一致。
21. MTP 接受率：`v3` §5.4.3 “between 85% and 90%” 与 `insights` §2.3.3 “80% to 90%”，两处都写 1.8×——E15 成立。
22. DeepEP：V1 `docs/legacy.md` 的两张性能表全部数字、`Buffer.set_num_sms(24)`、“slight differences” 声明；V2 README “SM usage reduced from 24 to 4 - 6”、EP2048、8×2 为 90/81 GB/s（12 SM）、8×4 为 61/61（6 SM）。
23. 版本与出处：`v3` v1 2024-12-27 / v2 2025-02-18；`insights` v1 2025-05-14 / v2 2025-12-23，ISCA '25 Industry Track、DOI 10.1145/3695053.3731412（arXiv 元数据），2025-06-21–25 东京（iscaconf.org/isca2025）；`fp8fmt` v1 2022-09-12 / v2 2022-09-29；`insights` 作者 15 人名单。
24. 图：`insights` 图 1 确有 combine “BF16/LogFMT”、dispatch “FP8”、RMSNorm/Router/RoPE/交叉熵 FP32、嵌入/输出头/MHA/MTP Linear Projection BF16（E20 成立）；`v3` 图 6 的全部 “To FP8 / To BF16 / To FP32” 标注与 C39 一致；`v3` 图 4 的组件顺序与 C21 一致。另复算 X5（3.2% 与 6.3%）、fp8sim（2.26%、16.4%、21/127 个变 0）、§4.3 每 token FLOPs（250.45 / 281.15）、§4.5 显存、§4.10 KV、§4.11 LogFMT 步长，均无误。

**改动（3 处实质性、2 处措辞）**
- **I3 与 §2.8：TPOT 公式里的 “×2” 解释错了。** 原稿写“每层两次 all-to-all 各 120.96 µs”。原文里 120.96 µs 已是 dispatch（1 B）+ combine（2 B）两次 all-to-all 的合计（“the time required for the two all-to-all communications in EP”），“Total Time Per Layer = 2×120.96” 的 2 来自双 micro-batch 重叠（“an idealized scenario under dual micro-batch overlap”）。已改 I3、在 §2.8 加一段说明，§4.8 表格加注。来源：`insights` v2 §2.3.2（v1 同）。
- **§8 第 3 条与 I12 / §3.7 / E6：表 4 的 1F、1B、1W 论文其实有定义。** 原文：“1F, 1B, and 1W denote forward time, input backward time, and weight backward time, respectively.”（`insights` §5.1.2，v1、v2 相同）。原稿把它列为“论文没有定义”，已改成只保留真正未定义的部分（统计口径；bubble/1F1B/opt）。同时在 I12、§3.7、E6 注明 1B 是**对输入的反向**，属于第三套记号，与 `v3` 表 2 的 $B$ 不同。
- **§0.1 `insights` 版本漂移**：补上 NVL72 那段前提的改写（v1 “Assuming the computation time is equal to the communication time” → v2 “Assuming perfect overlap between computation and communication”）。
- **§3.1 NVLink 一行**：原稿“可用 200 GB/s”是转述，已换成 §4.3 原句，避免读者以为 200 是“可用值”、160 是“实测值”的两级说法出自论文。
- **§4.7 fp8sim**：注明“去掉离群值的 tile”指设置离群值**之前**的原始 128 个值（这样复算为 1.65%；若删掉第 17 个元素只剩 127 个值则为 1.66%）。

**本次仍未能核实**（补充到 §8 的判断，不改变其结论）：H800 官方数据表仍然取不到（NVIDIA PDF 返回 404，nvidia.cn 页面返回 500）；Zero Bubble 的会议出处在 arXiv 元数据中没有 comment/journal 字段，仍未核；`insights` 表 4 各行是否为阶段累计时间仍无原文说明；`accsim.py` 的累加误差模型本次没有重跑。

---

## 11. 补充核实（写课时追加，2026-09-24）

写 `lesson-12-v3-infrastructure.md` 时，用 `arxiv.py grep 2412.19437`（V3 报告 v2 的 LaTeXML 全文）补查了下面几句原文，正文据此落笔。行号是 `arxiv.py` 缓存文本中的行号，仅供复查。

| # | 事实 | 出处与原文 |
|---|---|---|
| S1 | EMA 的用途与存放位置 | `v3` §3.2.3（约第 840–842 行）：“we preserve the Exponential Moving Average (EMA) of the model parameters for early estimation of the model performance after learning rate decay.”；“stored in CPU memory and are updated asynchronously after each training step”；“without incurring additional memory or time overhead.” |
| S2 | 不用 TP 的原因之一是显存优化 | `v3` §3.2（约第 752 行）：“we meticulously optimize the memory footprint during training, thereby enabling us to train DeepSeek-V3 without using costly Tensor Parallelism (TP).” |
| S3 | 整张量缩放对激活离群值敏感 | `v3` §3.3.2（约第 895–899 行）：“scaling the maximum absolute value of the input tensor to the maximum representable value of FP8”；“This method makes low-precision training highly sensitive to activation outliers”；细粒度分组 “can better accommodate outliers by adapting the scale according to smaller groups of elements.” |
| S4 | 线性层反向用的缓存激活存 FP8 | `v3` §3.3.3（约第 951 行）：“it is a natural choice to cache activations in FP8 format for the backward pass of the Linear operator.” |
| S5 | PD 分离的目的 | `v3` §3.4（约第 968 行）：“To simultaneously ensure both the Service-Level Objective (SLO) for online services and high throughput, we employ the following deployment strategy that separates the prefilling and decoding stages.” |
| S6 | decode 只给 MoE 分少量 SM 的理由 | `v3` §3.4.2（约第 1010–1011 行）：“Since the MoE part only needs to load the parameters of one expert, the memory access overhead is minimal, so using fewer SMs will not significantly affect the overall performance. Therefore, to avoid impacting the computation speed of the attention part, we can allocate only a small portion of SMs to dispatch+MoE+combine.” |

**从相邻课的研究笔记交叉引用（未重新下载原文）**
- 上下文扩展 4K → 32K → 128K（YaRN 两阶段各 1000 步）：`lesson-11-doc.md` C6（`v3` §4.3）。本课表 1 的“上下文扩展”一行用它作注。
- 节点受限路由的选节点规则（节点内最高 $K_r/M=2$ 个亲和度之和）：`lesson-11-doc.md` C22（`v3` §2.1.2）。

**写课时自己复算的数（脚本在 `deepseek/figures/gen/`，或正文算例里可逐步验算）**
- 0.1 存成 E4M3：$0.1=1.6\times2^{-4}$ → 舍入到 $1.625\times2^{-4}=0.1015625$，相对误差 1.5625%，码字 $e=3,m=5$（`l12_num.py` 复算）。
- 练习 6：amax 7.0 时 $s=2^{-6}$，0.05 → 3.25 → 0.05078（1.56%）；amax 7000 时 $s=15.625$，0.05 → 次正规 $2^{-8}$ → 0.0610（22.1%），0.01 → 0。
- 累加玩具模型（`l12_accum_error.py`，按 §4.12 的描述独立重写，seed 0，32×32 输出，中位逐元素相对误差）：不提升 K=512/1024/2048/4096/8192 为 3.40e-4 / 7.25e-4 / 1.41e-3 / 2.94e-3 / 5.88e-3；每 128 提升为 1.00e-4 / 9.25e-5 / 1.10e-4 / 1.02e-4 / 1.07e-4。与 §4.12 表同一量级、趋势相同（种子与细节不同，数值略异）；正文与图 12.7 用本表的数。
- 1F1B 模拟（`l12_pipeline_1f1b.py`，P=4、m=8、$T_F=1$、$T_B=2$）：朴素调度与 1F1B 的总时长都是 33，每段空闲都是 9；峰值激活朴素 8/8/8/8，1F1B 4/3/2/1。
- DualPipe 调度数据（`l12_dualpipe_schedule.py`）：用 §5.2 的 8 行序列重新核对：每行 66 格、空闲 6 格，前向先后次序无违例。
- 在 $T_{F\&B}=T_F+T_B$、$T_F=T_W=1$、$T_B=2$ 时，DualPipe 与 ZB1P 的气泡之差恒为 $(P-2)-(P-1)=-1$（练习 11）。
- 节点受限时一个 token 的跨节点流量上限（不计缩放因子）：$4\times(7168+14{,}336)=86{,}016$ 字节；dispatch 若也用 BF16 为 114,688 字节（多 33%）。【笔者估算】
- MTP 共用嵌入与输出头省下的训练状态：$2\times926{,}679{,}040\times12\approx22.2$ GB（按 12 字节/参数）。【笔者估算】
- 按 FP8 峰值（约 BF16 的 2 倍）换算的 MFU：$0.3894/2\approx19.5\%$。【笔者估算】
- ZB-H1 式调度模拟（`l12_zero_bubble.py`，教学审阅时追加，图 12.11）：P=4、m=8、$T_F=T_{B_x}=T_W=1$；F/B 保持 1F1B 顺序，下一个 F/B 未就绪时做积压的 W，第 $s$ 段最多积压 $s$ 个 W。结果：总时长 33 → 27，每段空闲 9 → 3，恰为 (12.17) 的 $(P-1)(T_F+T_B-2T_W)$；每段激活峰值 4/4/4/4（1F1B 为 4/3/2/1，最大值相同）。同一规则在 $T_W=0.5$ 时也吻合公式（P=4、8、16），但 $T_W=0.8$ 时空闲比公式多（P=4 时 4.8 对 4.2），所以这只是说明机制的简单规则，不是 ZB 论文 ZB-H1 的逐格复现。【笔者的模拟】
- 累加截断的量级估算（正文 §4 的算例）：14 位累加器每次截断平均丢掉累加和的 $2^{-15}\sim2^{-14}$；K=4096 时 128 次同号累积为 0.4%–0.8%，与玩具模型中位数 5.9e-3 同量级；提升后每段 4 次，$(1.2\sim2.4)\times10^{-4}$。【笔者估算】
  - **更正（数学审阅）**：5.9e-3 是玩具模型 **K=8192** 的中位数；K=4096 是 2.94e-3（见上面 `l12_accum_error.py` 一行），低于 0.4%–0.8% 的粗估。粗估把每次截断时的累加和都当成最终和那么大；乘积正负随机时部分和约按 $\sqrt i$ 增长，$\sqrt{i/128}$ 的平均为 0.670，修正后为 0.26%–0.52%，包含 0.29%。正文 §4 的算例已按此改写。【笔者估算】
- 冗余专家的负载算例（正文 §8）：数字是笔者构造的示例（平均每专家 100 次、热门专家 400 次），不是 DeepSeek 公布的负载数据。【笔者构造】
  - **更正（第二轮数学审阅）**：原文先说“共 25,600 次、平均每个 100 次”，又让一个专家取 400 次、其余仍取 100 次，总数实为 25,900，两处自相矛盾。改为“255 个专家各 100 次，1 个热门专家 400 次”，无热点时每块 GPU 为 800 次；1100/800 = 1.375、1000/800 = 1.25 不变；8 倍热点复制一份后最慢 GPU 为 800 + 400 = 1200（1.5 倍，不复制时 1500，1.875 倍）。
- 图 12.4 的 A、B 两行（第二轮数学审阅，`l12_outlier_error.py` 追加的断言）：A 的中位 1.65% 与 B 的 2.26% 之差**不是**离群值造成的。正规数范围内每个 binade 的相对精度相同；430 不是 2 的幂，各数在 binade 内的位置随之改变。把 A、B 的 $s$ 各乘 $2^u$（$u\in[0,1)$，100 个取值），中位误差分别在 1.65%–2.71% 与 1.71%–2.77% 之间，A 的 1.65% 恰是低端；对数均匀分布下的理论中位约 2.17%。图注已改写。【笔者计算】
- MTP 不在 250 GFLOPs 里（正文 §1 追加一段）：MTP 模块每 token 参数 = `eh_proj` 102,760,448 + 一层 MLA 187,105,280 + 9 个专家 396,361,728 + 门控 1,835,008 + 共用输出头 926,679,040 = 1,614,741,504；×6 = 9.69 GFLOPs，加一层注意力 30.70/61 = 0.50，合计 10.19，计入后 260.6（与 §4.3 表一致）。【笔者推断】
- LogFMT 的“无偏”（正文 §8 追加的解释）：相邻两级 $a<b$ 之间，线性最近舍入以 $(a+b)/2$ 为界，$\abs x$ 在 $[a,b]$ 上均匀时平均误差为 0；对数最近舍入以 $\sqrt{ab}$ 为界，平均误差为正（例：$a=1,b=1.19$ 时均值偏差约 $+0.004$）。【笔者推导】
- 截断误差在累加和变号时仍按 $n$ 线性增长（正文 §4 追加一段）：截断朝 0，第 $i$ 次误差约为 $-c_iS_i$（$c_i\sim2^{-14}$），总误差 $\approx-\bar c\sum_iS_i$；随机游走 $\operatorname{Var}(\sum_{i\le n}S_i)=\sigma^2\sum_{k=1}^{n}k^2\approx\sigma^2n^3/3$，而 $\abs{S_n}\sim\sigma\sqrt n$，故相对误差 $\propto n$。【笔者推导】

**写课时的处理**：研究笔记 §8 列为未核实的内容（H800 官方规格、`ue8m0` 分支的加入时间、表 4 各行口径、HAI-LLM 等），正文一律标为“笔者假设 / 未核实 / 笔者解读”。每卡 80 GB 显存在正文中全部标注为假设。

### 11.1 事实审阅补充（2026-09-25，逐条对照原文）

审阅方法：`arxiv.py grep`（`v3` v2、`insights` v2 的缓存全文，行号仅供复查）；GitHub raw 文件与 GitHub API（提交记录）；arXiv 摘要页。数字用 Python 重算（主模型激活参数 36.6246B、250.45 / 260.64 GFLOPs、272.800B token/天、180,176 GPU·h、386.1 TFLOP/s、TPOT 14.757 / 15.111 / 0.820 ms、Day 6 的 \$87,072 / \$562.1K / 545.5% / 95.5 + 131.4 节点、7.344×，全部与正文一致）。

| # | 事实 | 出处与原文 | 正文的处理 |
|---|---|---|---|
| S7 | V3 说 DualPipe 相对 **ZB1P 与 1F1B 两者**都“显著减少气泡” | `v3` §3.2.1（第 781 行）：“compared with ZB1P (Qi et al., 2023b) and 1F1B (Harlap et al., 2018), DualPipe significantly reduces the pipeline bubbles while only increasing the peak activation memory by $\frac{1}{PP}$ times.” 前一行（第 779 行）：“even in more general scenarios without a heavy communication burden, DualPipe still exhibits efficiency advantages.” | §6 的“DualPipe 把气泡砍掉一半”纠错框原写“比较的是同一张表里的 1F1B”，与原文不符，已改为照录原话，并按表 2 公式说明它对 ZB1P 成立到什么程度（与 §7 E5 一致）。 |
| S8 | Tensor Core 超出 14 位的部分被**截掉**，但方向未说明 | `v3` v2 §3.5.2（第 1050 行）：“the Tensor Core only uses the highest 14 bits of each mantissa product for addition, and truncates bits exceeding this range.” | §4 原写“截断的误差总朝同一个方向（朝 0）偏”，像是硬件事实；已改为“报告说是截断；方向报告没说，玩具模型假设朝 0”。 |
| S9 | “混合 FP8”是此前工作的做法 | `v3` §3.3.2（第 929 行）：“the hybrid FP8 format adopted by prior work (NVIDIA, 2024b; Peng et al., 2023b; Sun et al., 2019b), which uses E4M3 … in Fprop and E5M2 … in Dgrad and Wgrad” | §2 该句补上 `[@v3-report]`。 |
| S10 | LogFMT 的实验设定与开销原因 | `insights` §3.2（第 508 行）：“We validate our LogFMT-nBit on dense language models with around 7 billion parameters, by quantifying the output of the residual branch to simulate the combine stage in MoE models.”；§3.2.1（第 512 行）：“Due to insufficient GPU bandwidth for log/exp operations and excessive register pressure during encode/decode, if encode/decode operations are fused with all-to-all communication, the overhead can be substantial (50%∼100%).” | §8 补上“约 7B 稠密模型、模拟 combine”，并把“多花 50%–100% 的时间”改成原文的“开销可达 50%–100%”及其原因。 |
| S11 | DeepGEMM 首版 README 的定位 | 初始提交 `a6d97a1c`（2025-02-25T14:52:41Z）的 README：“FP8 General Matrix Multiplications (GEMMs) with fine-grained scaling, as proposed in DeepSeek-V3”；“To address the imprecise FP8 tensor core accumulation, it employs CUDA-core two-level accumulation (promotion).” **现行 README（News 到 2026.09.10）已不再有这两句**，只保留 SM90 FP32 缩放因子、1550 TFLOPS 等。 | §3 原写“开源的 DeepGEMM 就是它”，改为“按这种做法写的 FP8 GEMM 库”，引用首版 README；refs 卡片注明读的是首版与现行两版。 |
| S12 | Day 6 文章的提交日期 | GitHub API `repos/deepseek-ai/open-infra-index/commits?path=202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md`：只有一次提交，2025-03-01T04:03:12Z（北京时间 12:03），提交信息 “day6: one more thing”。 | 解决 §8、§9 里“确切发布日期未核”。正文与图 12.16 原写“2025 年 2 月公布”，改为“Day 6 文章（2025-03-01）”；refs 卡片同步。 |
| S13 | Zero Bubble 论文的日期与出处 | arXiv 摘要页 2401.10241v1：“Submitted on 30 Nov 2023”，无 Comments / Journal-ref 字段。 | refs 卡片的 2023 年无误；会议出处（ICLR 2024）仍未核实，正文与卡片都没有写。 |
| S14 | `insights` 表 4 的 10.3% 是推算值 | 表 4 只印 bubble 2.06 s 与 time/step 19.926 s；10.3% = 2.06/19.926，依赖“各行为阶段累计耗时”的读法（§8 第 3 条）。 | 图 12.14 的虚线标签原写“V3 实测 10.3%”，改为“由 Insights 表 4 算出的 10.3%（2.06 s ÷ 19.926 s）”；§6 小结与总账表同步。 |

**其它小改**：§1“每层 256 个路由专家”改为“前 3 层稠密、其余 58 层每层 256 个路由专家”（`config.json` `first_k_dense_replace: 3`）；§8 “一块卡放的专家越少、延迟越低、省显存给 KV cache”标为笔者的理解（报告原话只到“访存受限、只需读一个专家的参数”，S6）；正文到 M08、M10 的链接补上页码（M08 §2 位布局、§4 舍入误差、§5 累加误差；M10 §5 四种并行、§6 集合通信）。
