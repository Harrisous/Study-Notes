# 第 10 课研究笔记：DeepSeek-V2（下）——规模化 MoE、训练、长上下文与对齐

> 本文件是 `deepseek/src/lesson-10-v2-moe-training.md` 的研究底稿：只收核对过的引文、复算过的数字、勘误、未能核实清单，
> 不是讲义正文。建立于 2026-09-24。引文保留英文原文并注明节号 / 表号 / 式号 / 所读版本；派生数字全部在 Python 里复算（第 4 节）。
> 凡笔者自己的推理均标注“笔者推测 / 笔者估算”。MLA 相关内容见 `lesson-09-doc.md`，本文件不重复。

---

## 0. 覆盖范围与所读版本

### 0.1 论文

| 短键 | 标题 | 作者 | arXiv / 版本 | 状态 | 日期 | 本课用途 |
|---|---|---|---|---|---|---|
| `dsv2` | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek-AI | 2405.04434，**读 v5**，v1–v4 全文 diff（版本史见 `lesson-09-doc.md` §0.3） | 预印本 | v1 2024-05-07；v5 2024-06-19 | §2.2（MoE、设备受限路由、三种均衡损失、token 丢弃）、§3（预训练、基础设施、YaRN、效率）、§4（SFT、GRPO 两阶段 RL）、附录 B（V2-Lite） |
| `coderv2` | DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence | DeepSeek-AI（Qihao Zhu, Daya Guo, Zhihong Shao 等） | 2406.11931**v1**（唯一版本） | 预印本 | 2024-06-17 | 从 V2 中间检查点续训 6T、338 种语言、16K→128K |
| `dsmoe` | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | Damai Dai 等 | 2401.06066v1 | 预印本（后 ACL 2024，本次未核对出版页） | 2024-01-11 | 设备级均衡损失的出处（V2 沿用） |
| `dsmath` | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Zhihong Shao 等 | 2402.03300v3 | 预印本 | 2024-02-05（v1） | GRPO 原始（逐 token）目标函数，与 V2 式 (32) 对照 |
| `yarn` | YaRN: Efficient Context Window Extension of Large Language Models | Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole | 2309.00071（arXiv API 最新 v3） | 预印本（ICLR 2024，本次未核对出版页） | 2023-08-31（v1） | 式 (10)(11)(13)(14)(15)；α=1、β=32 的推荐值 |
| `zb` | Zero Bubble Pipeline Parallelism | Penghui Qi, Xinyi Wan, Guangxing Huang, Min Lin | 2401.10241v1 | 预印本 | 2023-11-30（arXiv 元数据） | V2 的 16 路零气泡流水线并行 |
| `vmoe` | Scaling Vision with Sparse Mixture of Experts | Carlos Riquelme 等 | 2106.05974v1 | 预印本 | 2021-06-10 | V2 的“按亲和度丢 token”灵感来源（仅核对了元数据） |
| `dsllm` | DeepSeek LLM | DeepSeek-AI | 2401.02954 | 预印本 | 2024-01 | 67B（95 层、GQA-8）与 7B 的规格；warmup-and-step-decay 调度出处 |

### 0.2 制品（`kind: artefact`）

| 短键 | 内容 |
|---|---|
| `hf-v2-config` / `hf-v2-code` | `deepseek-ai/DeepSeek-V2` 的 `config.json` 与 `modeling_deepseek.py`（`MoEGate`、`DeepseekV2MoE`、`DeepseekV2YarnRotaryEmbedding`） |
| `hf-v2lite-*` | `deepseek-ai/DeepSeek-V2-Lite` 的 config、modeling、README（模型卡） |
| `hf-coderv2-config` | `DeepSeek-Coder-V2-Instruct`、`DeepSeek-Coder-V2-Lite-Base` 的 config |
| `hf-v25` | `DeepSeek-V2.5` 模型卡 + DeepSeek API 新闻页 `api-docs.deepseek.com/news/news0905` |
| `hf-v2chat0628` | `DeepSeek-V2-Chat-0628` 模型卡 |
| `gh-v2` / `gh-coderv2` | 两个 GitHub README |

所有制品于 2026-09-24 原样下载阅读。

---

## 1. 核心论断与原文引文

以下均引自 `dsv2` v5，除非另注。

### 1.1 规格

**C1. 236B / 21B / 128K。**（摘要）
> "It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens."

**C2. MoE 结构与规模。**（§3.1.2）
> "Following Dai et al. (2024), we substitute all FFNs except for the first layer with MoE layers. Each MoE layer consists of 2 shared experts and 160 routed experts, where the intermediate hidden dimension of each expert is 1536. Among the routed experts, 6 experts will be activated for each token."

**C3. DeepSeekMoE 的两个思想。**（§2.2.1）
> "DeepSeekMoE has two key ideas: segmenting experts into finer granularity for higher expert specialization and more accurate knowledge acquisition, and isolating some shared experts for mitigating knowledge redundancy among routed experts."

**C4. 其余细节沿用 67B。**（§2 开头）
> "For other tiny details (e.g., layer normalization and the activation function in FFNs), unless specifically stated, DeepSeek-V2 follows the settings of DeepSeek 67B (DeepSeek-AI, 2024)."

### 1.2 设备受限路由

**C5. 动机：细粒度专家让通信变贵。**（§2.2.2）
> "For each token, its MoE-related communication frequency is proportional to the number of devices covered by its target experts. Due to the fine-grained expert segmentation in DeepSeekMoE, the number of activated experts can be large, so the MoE-related communication will be more costly if we apply expert parallelism."

**C6. 机制：先选 M 个设备，再在其中做 top-K；M≥3 时效果与不受限大致相当。**（§2.2.2）
> "For DeepSeek-V2, beyond the naive top-K selection of routed experts, we additionally ensure that the target experts of each token will be distributed on at most $M$ devices. To be specific, for each token, we first select $M$ devices that have experts with the highest affinity scores in them. Then, we perform top-K selection among experts on these $M$ devices. In practice, we find that when $M\geqslant 3$, the device-limited routing can achieve a good performance roughly aligned with the unrestricted top-K routing."

**C7. 取值：D=8、M=3。**（§3.1.2）
> "We leverage pipeline parallelism to deploy different layers of a model on different devices, and for each layer, the routed experts will be uniformly deployed on 8 devices ($D=8$). As for the device-limited routing, each token will be sent to at most 3 devices ($M=3$)."

### 1.3 三种均衡损失

**C8. 为什么要均衡。**（§2.2.3）
> "Firstly, unbalanced load will raise the risk of routing collapse (Shazeer et al., 2017), preventing some experts being fully trained and utilized. Secondly, when expert parallelism is employed, unbalanced load will diminish computation efficiency."

**C9. 三种损失的名字。**（§2.2.3）
> "During the training of DeepSeek-V2, we design three kinds of auxiliary losses, for controlling expert-level load balance ($\mathcal{L}_{\mathrm{ExpBal}}$), device-level load balance ($\mathcal{L}_{\mathrm{DevBal}}$), and communication balance ($\mathcal{L}_{\mathrm{CommBal}}$), respectively."

**C10. 通信均衡损失的动机：发送有上界，接收没有。**（§2.2.3）
> "Although the device-limited routing mechanism guarantees that the sending communication of each device is bounded, if a certain device receives more tokens than other devices, the practical communication efficiency will also be affected."

> "The device-limited routing mechanism operates on the principle of ensuring that each device transmits at most $MT$ hidden states to other devices. Simultaneously, the communication balance loss is employed to encourage each device to receive around $MT$ hidden states from other devices."

**C11. 系数。**（§3.1.2）
> "As for balance losses, we set $\alpha_{1}$ to 0.003, $\alpha_{2}$ to 0.05, and $\alpha_{3}$ to 0.02."

**C12. 设备级均衡损失并非 V2 新创，出自 DeepSeekMoE。**（`dsmoe` §3.3）
> "In addition to the expert-level balance loss, we introduce a device-level balance loss. When aiming to alleviate computation bottlenecks, it becomes unnecessary to enforce strict balance constraints at the expert level, because excessive constraints on load balance will compromise model performance."
> "In practice, we set a small expert-level balance factor to mitigate the risk of routing collapse, and meanwhile set a larger device-level balance factor to promote balanced computation across the devices."

  V2 的新东西是：设备受限路由（§2.2.2）、通信均衡损失（式 29–31）、设备级 token 丢弃（§2.2.4）。

### 1.4 token 丢弃

**C13.**（§2.2.4，摘关键句）
> "This approach first computes the average computational budget for each device, which means that the capacity factor for each device is equivalent to 1.0. Then, inspired by Riquelme et al. (2021), we drop tokens with the lowest affinity scores on each device until reaching the computational budget. In addition, we ensure that the tokens belonging to approximately 10% of the training sequences will never be dropped."

  同段其余内容（转述）：均衡损失不能保证严格均衡，所以训练时加设备级 token 丢弃；保留约 10% 永不丢弃的序列，是为了推理时可以按效率需要自由选择丢或不丢，并保持训练与推理一致。

**C14. 评测时不丢。**（§3.1.2）
> "We employ the token-dropping strategy during training for acceleration, but do not drop any tokens for evaluation."

### 1.5 数据与训练超参数

**C15. 数据。**（§3.1.1）
> "We adopt the same tokenizer as used in DeepSeek 67B, which is built based on the Byte-level Byte-Pair Encoding (BBPE) algorithm and has a vocabulary size of 100K. Our tokenized pre-training corpus contains 8.1T tokens, where Chinese tokens are approximately 12% more than English ones."

**C16. 去偏过滤。**（§3.1.1）
> "In addition, we filter out the contentious content from our pre-training corpus to mitigate the data bias introduced from specific regional cultures."
  附录 E 表 10（420 个道德情境）：三名标注者与 MMLU Humanity-Moral 子集标准答案的一致率为 66.7%、59.8%、42.1%，标注者两两之间 57.9%–69.0%。

**C17. 优化器与学习率。**（§3.1.2 “Training Hyper-Parameters”）
> "We employ the AdamW optimizer (Loshchilov and Hutter, 2017) with hyper-parameters set to $\beta_{1}=0.9$, $\beta_{2}=0.95$, and $\mathrm{weight\_decay}=0.1$. The learning rate is scheduled using a warmup-and-step-decay strategy (DeepSeek-AI, 2024). Initially, the learning rate linearly increases from 0 to the maximum value during the first 2K steps. Subsequently, the learning rate is multiplied by 0.316 after training about 60% of tokens, and again by 0.316 after training about 90% of tokens. The maximum learning rate is set to $2.4\times 10^{-4}$, and the gradient clipping norm is set to 1.0."

**C18. batch 调度与序列长度。**（§3.1.2）
> "We also use a batch size scheduling strategy, where the batch size is gradually increased from 2304 to 9216 in the training of the first 225B tokens, and then keeps 9216 in the remaining training. We set the maximum sequence length to 4K, and train DeepSeek-V2 on 8.1T tokens."

**C19. 基础设施。**（§3.1.3）
> "It employs a 16-way zero-bubble pipeline parallelism (Qi et al., 2023), an 8-way expert parallelism (Lepikhin et al., 2021), and ZeRO-1 data parallelism (Rajbhandari et al., 2020)."
> "[...] it can be trained without the necessity of tensor parallelism, thereby decreasing the communication overhead."
> "Each node in the H800 cluster contains 8 GPUs connected using NVLink and NVSwitch within nodes. Across nodes, InfiniBand interconnects are utilized to facilitate communications."

  同段其余内容（转述）：不用张量并行的理由是激活参数较少、部分算子重计算以省激活显存；共享专家的计算与专家并行的 all-to-all 通信重叠；为通信、路由和跨专家的融合线性计算定制了 CUDA 内核。

### 1.6 长上下文扩展（YaRN）

**C20.**（§3.1.4，整段关键句）
> "After the initial pre-training of DeepSeek-V2, we employ YaRN (Peng et al., 2023) to extend the default context window length from 4K to 128K. YaRN was specifically applied to the decoupled shared key $\mathbf{k}^{R}_{t}$ as it is responsible for carrying RoPE (Su et al., 2024). For YaRN, we set the scale $s$ to 40, $\alpha$ to 1, $\beta$ to 32, and the target maximum context length to 160K. Under these settings, we can expect the model to respond well for a context length of 128K. Slightly diverging from original YaRN, due to our distinct attention mechanism, we adjust the length scaling factor to modulate the attention entropy. The factor $\sqrt{t}$ is computed as $\sqrt{t}=0.0707\ln{s}+1$, aiming at minimizing the perplexity."

> "We additionally train the model for 1000 steps, with a sequence length of 32K and a batch size of 576 sequences. Although the training is conducted solely at the sequence length of 32K, the model still demonstrates robust performance when being evaluated at a context length of 128K."

**C21. 图 4 标题。**
> "Evaluation results on the “Needle In A Haystack” (NIAH) tests. DeepSeek-V2 performs well across all context window lengths up to 128K."

**C22. YaRN 原文：α、β 按模型调，LLaMA 上取 α=1、β=32。**（`yarn` §3.2）
> "The values of $\alpha$ and $\beta$ should be tuned on a case-by-case basis. For example, we have found experimentally that for the Llama family of models, good values for $\alpha$ and $\beta$ are $\alpha=1$ and $\beta=32$."

**C23. YaRN 原文的长度缩放技巧只改 RoPE 嵌入。**（`yarn` v3 §3.3 “YaRN”，式 (14) 之后；§3.4 是 Dynamic NTK，核查时更正）
> "we can instead use a “length scaling” trick which scales both $\bm{q}_m$ and $\bm{k}_n$ by a constant factor $\sqrt{1/t}$ by simply scaling the complex rotary position embeddings by the same amount."

  笔者推测：MLA 里只有 64 维 RoPE 分支，按原 YaRN 缩放 RoPE 嵌入只会放大 192 维点积中的 64 维部分；
  这正是 V2 所说 “due to our distinct attention mechanism, we adjust the length scaling factor” 的原因。
  代码佐证见 §1.10 与 §2.5。

### 1.7 训练与推理效率

**C24. 42.5% 训练成本是“每万亿 token 的 GPU 小时”。**（§3.2.3）
> "During our practical training on the H800 cluster, for training on each trillion tokens, DeepSeek 67B requires 300.6K GPU hours, while DeepSeek-V2 needs only 172.8K GPU hours, i.e., sparse DeepSeek-V2 can save 42.5% training costs compared with dense DeepSeek 67B."

**C25. 吞吐。**（§3.2.3）
> "On a single node with 8 H800 GPUs, DeepSeek-V2 achieves a generation throughput exceeding 50K tokens per second, which is 5.76 times the maximum generation throughput of DeepSeek 67B. In addition, the prompt input throughput of DeepSeek-V2 exceeds 100K tokens per second."
  （部署条件：FP8 权重、平均 6 bit KV 量化，见 `lesson-09-doc.md` C15。）

### 1.8 对齐

**C26. SFT 数据与超参数。**（§4.1）
> "Building upon our prior research (DeepSeek-AI, 2024), we curate our instruction tuning datasets to include 1.5M instances, comprising 1.2M instances for helpfulness and 0.3M instances for safety. [...] We fine-tune DeepSeek-V2 with 2 epochs, and the learning rate is set to $5\times 10^{-6}$."
  （引言里同一批数据写作 “1.5M conversational sessions”。）

**C27. 用 GRPO 省掉 critic。**（§4.2）
> "In order to save the training costs of RL, we adopt Group Relative Policy Optimization (GRPO) (Shao et al., 2024), which foregoes the critic model that is typically with the same size as the policy model, and estimates the baseline from group scores instead."

**C28. 两阶段 RL。**（§4.2 “Training Strategy”）
> "In our preliminary experiments, we find that the RL training on reasoning data, such as code and math prompts, exhibits unique characteristics that are distinct from the training on general data. For example, the mathematical and coding abilities of our model can keep improving over a longer period of training steps. Therefore, we employ a two-stage RL training strategy, which first performs reasoning alignment, and then performs human preference alignment."

**C29. 第二阶段三个奖励模型。**（§4.2）
> "In the second human preference alignment stage, we adopt a multi-reward framework, which acquires rewards from a helpful reward model $RM_{helpful}$, a safety reward model $RM_{safety}$, and a rule-based reward model $RM_{rule}$."

**C30. 奖励模型的数据与初始化。**（§4.2）
> "We obtain code preference data based on compiler-feedback, and mathematical preference data based on the ground-truth labels. For reward model training, we initialize the reward models with DeepSeek-V2 Chat (SFT) and train them with either a point-wise or a pair-wise loss."

**C31. RL 工程优化。**（§4.2 “Optimizations for Training Efficiency”）
> "(1) Firstly, we propose a hybrid engine that adopts different parallel strategies for training and inference respectively to achieve higher GPU utilization. (2) Secondly, we leverage vLLM (Kwon et al., 2023) with large batch sizes as our inference backend to accelerate the inference speed. (3) Thirdly, we carefully design a scheduling strategy for offloading models to CPUs and loading models back to GPUs, which achieves a near-optimal balance between the training speed and memory consumption."

**C32. 讨论：SFT 数据量、对齐税、在线 RL。**（§4.4）
> "However, in our experiments, we observe a significant performance decline on the IFEval benchmark if we use fewer than 10K instances."
> "However, we also notice a phenomenon of “alignment tax” (Ouyang et al., 2022), i.e., the alignment process can negatively impact the performance on some standard benchmarks such as BBH."
> "In our preference alignment experiments, we find that the online approach significantly outperforms the offline approach."

**C33. 头条对话成绩。**（§1）
> "Notably, DeepSeek-V2 Chat (RL) achieves 38.9 length-controlled win rate on AlpacaEval 2.0 (Dubois et al., 2024), 8.97 overall score on MT-Bench (Zheng et al., 2023), and 7.91 overall score on AlignBench (Liu et al., 2023)."

### 1.9 V2-Lite 与 Coder-V2

**C34. V2-Lite 结构。**（附录 B.1）
> "DeepSeek-V2-Lite has 27 layers and a hidden dimension of 2048. It also employs MLA and has 16 attention heads, where each head has a dimension of 128. Its KV compression dimension is 512, but slightly different from DeepSeek-V2, it does not compress the queries. For the decoupled queries and key, it has a per-head dimension of 64. DeepSeek-V2-Lite also employs DeepSeekMoE, and all FFNs except for the first layer are replaced with MoE layers. Each MoE layer consists of 2 shared experts and 64 routed experts, where the intermediate hidden dimension of each expert is 1408. Among the routed experts, 6 experts will be activated for each token. Under this configuration, DeepSeek-V2-Lite comprises 15.7B total parameters, of which 2.4B are activated for each token."

**C35. V2-Lite 训练。**（附录 B.1）
> "DeepSeek-V2-Lite is also trained from scratch on the same pre-training corpus of DeepSeek-V2, which is not polluted by any SFT data."
> "The maximum learning rate is set to $4.2\times 10^{-4}$ [...] it is trained with a constant batch size of 4608 sequences. During pre-training, we set the maximum sequence length to 4K, and train DeepSeek-V2-Lite on 5.7T tokens."
> "Therefore, we only employ a small expert-level balance loss with $\alpha_{1}=0.001$, and do not employ device-level balance loss and communication balance loss for it."
> "After pre-training, we also perform long context extension and SFT for DeepSeek-V2-Lite and get a chat model called DeepSeek-V2-Lite Chat."（v4 起；v3 写作 “long-context extension, SFT, and RL”）

**C36. Coder-V2：从 V2 中间检查点续训 6T。**（`coderv2` 摘要、§1、§3.3）
> "Specifically, DeepSeek-Coder-V2 is further pre-trained from an intermediate checkpoint of DeepSeek-V2 with additional 6 trillion tokens."
> "Additionally, DeepSeek-Coder-V2 expands its support for programming languages from 86 to 338, while extending the context length from 16K to 128K."
> "The intermediate checkpoint was initially trained on 4.2T tokens. Consequently, DeepSeek-Coder-V2 has been exposed to a total of 10.2T high-quality tokens during the pre-training phase."

**C37. Coder-V2 语料配比。**（`coderv2` §1）
> "In the pre-training phase, the dataset of DeepSeek-Coder-V2 is created with a composition of 60% source code, 10% math corpus, and 30% natural language corpus. The source code consists of 1,170B code-related tokens sourced from GitHub and CommonCrawl, using the same pipeline as DeepSeekMath (Shao et al., 2024)."
> "For the math corpus, we collect 221B math-related tokens sourced from CommonCrawl using the same pipeline, which approximately doubles the size of the 120B DeepSeekMath corpus"

**C38. Coder-V2 架构与一处训练不稳定。**（`coderv2` §3.2）
> "Our architecture aligns with that of DeepSeekV2 (DeepSeek-AI, 2024). The hyperparameters settings, 16B and 236B, correspond to those used in DeepSeek-V2-Lite and DeepSeek-V2, respectively. Notably, we encountered instability during training and spikes in gradient values, which we attributed to the exponential normalization technique. To address this, we reverted to the conventional normalization method."

**C39. Coder-V2 超参数与 FIM。**（`coderv2` §3.1、§3.3、表 2）
> "For DeepSeek-Coder-v2 236B, we only utilize the Next-Token-Prediction objective."；“The FIM is utilized at a rate of 0.5”（仅 16B）。
> "For learning rate scheduling, we employ a cosine decay strategy, starting with 2000 warm-up steps and gradually reducing the learning rate to 10% of its initial value."
  表 2：Lite 与 236B 的 Pre-training Tokens 都是 “4.2T+6T”，LR Scheduler 都是 “Cosine”，FIM：Lite “Enable”、236B “Disable”。

**C40. Coder-V2 长上下文：两阶段。**（`coderv2` §3.4）
> "The hyper-parameters of YARN are the same as DeepSeek-V2: the scale $s$ to 40, $\alpha$ to 1, $\beta$ to 32. We further continue training the model using two stages to enhance its capability for handling long contexts. In the first stage, we utilize a sequence length of 32K and a batch size of 1152 for 1000 steps. In the second stage, we train the model for an additional 1000 steps, employing a sequence length of 128K and a batch size of 288 sequences."

**C41. Coder-V2 对齐：奖励模型胜过直接用编译器信号。**（`coderv2` §3.5）
> SFT："Finally, we use a instruction dataset of 300M tokens. For training, we use a cosine schedule with 100 warm-up steps and an initial learning rate $5e^{-6}$."
> RL 提示："After filtering the prompts, there are approximately 40k data in total."
> "Therefore, we still decide to train a reward model on the data provided by the compiler, and use the reward model to provide signal during RL training, which is more robust and has better generalization ability, in comparison with raw compiler signal."
> "We employ Group Relative Policy Optimization (GRPO) Shao et al. (2024) as our RL algorithm, which is the same as what DeepSeek-V2 uses."

**C42. V2.5 = V2-Chat 与 Coder-V2-Instruct 合并。**（`hf-v25` 模型卡；API 新闻页 2024/09/05）
> "DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct."
  API 新闻页（2024/09/05）写作 “a powerful combination of DeepSeek-V2-0628 and DeepSeek-Coder-V2-0724”。

### 1.10 代码（`hf-v2-code`）里与本课有关的事实

- `MoEGate.forward`：打分 `scores = logits.softmax(dim=-1)`（式 (22)）；`topk_method == "group_limited_greedy"` 时
  `group_scores = scores.view(..., n_group, -1).max(dim=-1).values`，取 `topk_group` 个组，再在这些组里 top-k（即设备受限路由，**设备得分 = 组内最大亲和度**）。
- `norm_topk_prob = False` ⇒ 门控值不重新归一化，而是 `topk_weight * routed_scaling_factor`（V2 为 **16.0**）。
- `seq_aux = True`：专家级均衡损失按**每条序列**计算后对 batch 求平均，`ce ... .div_(seq_len * aux_topk / self.n_routed_experts)` 正是式 (24) 的 $\tfrac{N_r}{K_rT}$；
  代码只实现了专家级损失，**没有设备级、通信级损失和 token 丢弃**（这是推理代码，训练框架 HAI-LLM 未开源）。
- 共享专家实现为**一个** MLP，`intermediate_size = moe_intermediate_size * n_shared_experts` = 3072，数学上等于两个 1536 的专家相加。
- YaRN：`yarn_get_mscale(scale, mscale) = 0.1 * mscale * ln(scale) + 1`；V2 的 `mscale = mscale_all_dim = 0.707`，
  于是 cos/sin 缓存上的缩放 `_mscale = get_mscale(40, 0.707)/get_mscale(40, 0.707) = 1`（RoPE 嵌入不缩放），
  而 `softmax_scale = 192**-0.5 * mscale**2`（**整个 192 维点积**都乘 $(0.0707\ln 40+1)^2$）。
- V2 仓库代码里贪心分支写成 `if self.topk_method == "gready"`（拼写错误；V2 用的是 `group_limited_greedy`，不受影响）；V2-Lite 仓库代码已改为 `"greedy"`。

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 DeepSeekMoE 层，式 (20)–(22)（§2.2.1，照录）

$$\mathbf{h}_{t}^{\prime}=\mathbf{u}_{t}+\sum_{i=1}^{N_{s}}{\operatorname{FFN}^{(s)}_{i}\left(\mathbf{u}_{t}\right)}+\sum_{i=1}^{N_{r}}{g_{i,t}\operatorname{FFN}^{(r)}_{i}\left(\mathbf{u}_{t}\right)},\quad(20)$$

$$g_{i,t}=\begin{cases}s_{i,t},&s_{i,t}\in\operatorname{Topk}(\{s_{j,t}|1\leqslant j\leqslant N_{r}\},K_{r}),\\ 0,&\text{otherwise},\end{cases}\quad(21)$$

$$s_{i,t}=\operatorname{Softmax}_{i}\left({\mathbf{u}_{t}}^{T}\mathbf{e}_{i}\right),\quad(22)$$

原文符号说明：“$N_{s}$ and $N_{r}$ denote the numbers of shared experts and routed experts, respectively; [...] $K_{r}$ denotes the number of activated routed experts; $g_{i,t}$ is the gate value for the $i$-th expert; $s_{i,t}$ is the token-to-expert affinity; $\mathbf{e}_{i}$ is the centroid of the $i$-th routed expert in this layer; and $\operatorname{Topk}(\cdot,K)$ denotes the set comprising $K$ highest scores among the affinity scores calculated for the $t$-th token and all routed experts.”

V2 取值：$N_s=2$，$N_r=160$，$K_r=6$。代码里 $g_{i,t}$ 实际为 $16\,s_{i,t}$（§1.10）。
注意式 (21) 的 Topk 在设备受限路由下要换成“先选 $M$ 个设备再 Topk”，论文没有写成公式（§2.4 给出笔者的形式化）。

### 2.2 专家级均衡损失，式 (23)–(25)（照录）

$$\mathcal{L}_{\mathrm{ExpBal}}=\alpha_{1}\sum_{i=1}^{N_{r}}{f_{i}P_{i}},\quad(23)$$
$$f_{i}=\frac{N_{r}}{K_{r}T}\sum_{t=1}^{T}{\mathds{1}(\text{Token $t$ selects Expert $i$})},\quad(24)$$
$$P_{i}=\frac{1}{T}\sum_{t=1}^{T}{s_{i,t}},\quad(25)$$

原文：“$\alpha_{1}$ is a hyper-parameter called expert-level balance factor; $\mathds{1}(\cdot)$ denotes the indicator function; and $T$ denotes the number of tokens in a sequence.”

### 2.3 设备级与通信均衡损失，式 (26)–(31)（照录）

$$\mathcal{L}_{\mathrm{DevBal}}=\alpha_{2}\sum_{i=1}^{D}{f_{i}^{\prime}P_{i}^{\prime}},\quad(26)\qquad
f_{i}^{\prime}=\frac{1}{|\mathcal{E}_{i}|}\sum_{j\in\mathcal{E}_{i}}{f_{j}},\quad(27)\qquad
P_{i}^{\prime}=\sum_{j\in\mathcal{E}_{i}}{P_{j}},\quad(28)$$

$$\mathcal{L}_{\mathrm{CommBal}}=\alpha_{3}\sum_{i=1}^{D}{f_{i}^{\prime\prime}P_{i}^{\prime\prime}},\quad(29)\qquad
f_{i}^{\prime\prime}=\frac{D}{MT}\sum_{t=1}^{T}{\mathds{1}(\text{Token $t$ is sent to Device $i$})},\quad(30)\qquad
P_{i}^{\prime\prime}=\sum_{j\in\mathcal{E}_{i}}{P_{j}},\quad(31)$$

原文：“we partition all routed experts into $D$ groups $\{\mathcal{E}_{1},\mathcal{E}_{2},...,\mathcal{E}_{D}\}$, and deploy each group on a single device.” V2：$D=8$，$|\mathcal E_i|=20$，$M=3$。

归一化检查（笔者推导，用于讲义）：完全均匀时 $f_i=1$、$P_i=1/N_r$ ⇒ $\sum_if_iP_i=1$；$f'_i=1$、$P'_i=|\mathcal E_i|/N_r=1/D$ ⇒ $\sum f'P'=1$；
每个 token 恰好发往 $M$ 个设备且均匀时每设备收到 $MT/D$ 个 ⇒ $f''_i=1$，$\sum f''P''=1$。所以三项损失在均匀时分别等于 $\alpha_1,\alpha_2,\alpha_3$。
注意 $P''_i$ 与 $P'_i$ 定义完全相同（式 28 = 式 31），三种损失的差别全在 $f$ 上。

### 2.4 设备受限路由（论文无公式；笔者按 §2.2.2 与 HF 代码形式化）

记设备 $m$ 上的专家集合为 $\mathcal E_m$。
$$\text{devscore}_{m,t}=\max_{j\in\mathcal E_m}s_{j,t},\qquad \mathcal M_t=\operatorname{TopM}\big(\{\text{devscore}_{m,t}\}_{m=1}^{D},M\big),$$
$$g_{i,t}=\begin{cases}s_{i,t},& i\in\bigcup_{m\in\mathcal M_t}\mathcal E_m\ \text{且}\ s_{i,t}\in\operatorname{Topk}\big(\{s_{j,t}:j\in\bigcup_{m\in\mathcal M_t}\mathcal E_m\},K_r\big)\\0,&\text{otherwise}\end{cases}$$
“设备得分取组内最大值”来自代码（`.max(dim=-1)`），论文原话 “select $M$ devices that have experts with the highest affinity scores in them” 与之一致。
对照：V3 的 HF 代码（`topk_method="noaux_tc"`）改为 `topk(2).sum()`，即组内前 2 名（加偏置后）亲和度之和，`topk_group`=4；
V3 论文里“节点受限路由”的原文属第 11 课，本次只核对了代码。

### 2.5 YaRN 与 V2 的长度缩放

`yarn`（照录，式号为原文）：
$$\lambda_{d}=\dfrac{2\pi}{\theta_{d}}=2\pi b^{\frac{2d}{|D|}}.\quad(8)$$
$$r(d)=\dfrac{L}{\lambda_{d}}=\dfrac{L}{2\pi b^{\frac{2d}{|D|}}}.\quad(10)$$
$$\gamma(r)=\begin{cases}0,&\text{if }r<\alpha\\ 1,&\text{if }r>\beta\\ \dfrac{r-\alpha}{\beta-\alpha},&\text{otherwise}.\end{cases}\quad(11)$$
$$g(m)=m\quad(12)\qquad h(\theta_{d})=\Big(1-\gamma\big(r(d)\big)\Big)\frac{\theta_{d}}{s}+\gamma\big(r(d)\big)\theta_{d}.\quad(13)$$
$$\text{softmax}\left(\dfrac{\bm{q}_{m}^{T}\bm{k}_{n}}{t\sqrt{|D|}}\right).\quad(14)\qquad\qquad \sqrt{\frac{1}{t}}=0.1\ln({s})+1.\quad(15)$$
其中 $s=L'/L$（“scale factor”），$b$ 为 RoPE 底数，$|D|$ 为头维。

`dsv2` §3.1.4：$s=40$，$\alpha=1$，$\beta=32$，目标长度 160K，且
$$\sqrt{t}=0.0707\ln{s}+1 .$$
- 代码（§1.10）：logits 乘 $(0.1\cdot0.707\cdot\ln 40+1)^2=1.2608^2=1.5896$；YaRN 原式在 $s=40$ 时是 $(0.1\ln40+1)^2=1.3689^2=1.8739$。
- **记号反转**：YaRN 的 $t$ 出现在分母（式 14），$\sqrt{1/t}>1$ 是放大系数；V2 写 “$\sqrt t=0.0707\ln s+1$”，按代码它也是放大系数，
  即 V2 的 $\sqrt t$ 对应 YaRN 的 $\sqrt{1/t}$。讲义统一写成“logit 乘以 $\big(0.0707\ln s+1\big)^2$”，避免 $t$ 的方向之争。
- **代码的斜坡与论文不同**：YaRN 式 (11) 的 $\gamma$ 对 $r$ 线性；HF 代码 `yarn_linear_ramp_mask` 对维度下标 $i$ 线性，
  在 $i\in[\lfloor10.47\rfloor,\lceil22.51\rceil]=[10,23]$ 之间过渡（数值见 4.8 节）。两者端点一致、中间形状不同。

### 2.6 GRPO，式 (32)–(34)（§4.2，照录）

$$\begin{split}\mathcal{J}_{GRPO}(\theta)&=\mathbb{E}{[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O|q)]}\\
&\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)}A_{i},\text{clip}\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)},1-\varepsilon,1+\varepsilon\right)A_{i}\right)-\beta\mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)\right),\end{split}\quad(32)$$

$$\mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)=\frac{\pi_{ref}(o_{i}|q)}{\pi_{\theta}(o_{i}|q)}-\log\frac{\pi_{ref}(o_{i}|q)}{\pi_{\theta}(o_{i}|q)}-1,\quad(33)$$

$$A_{i}=\frac{r_{i}-{\mathrm{m}ean(\{r_{1},r_{2},\cdots,r_{G}\})}}{{\mathrm{s}td(\{r_{1},r_{2},\cdots,r_{G}\})}}.\quad(34)$$

对照 `dsmath` 式 (3)（GRPO 原文，§4.1.1 “From PPO to GRPO”；§4.1.2 是结果监督，核查时更正）：DeepSeekMath 写的是**逐 token** 形式
$\frac{1}{G}\sum_{i}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\{\min[\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}\hat A_{i,t},\ldots]-\beta\mathbb D_{KL}[\pi_\theta||\pi_{ref}]\}$；
V2 的式 (32) 把比值写在**整条回答** $o_i$ 上、没有 $\frac{1}{|o_i|}\sum_t$。V2 未说明实现是否真的按序列级比值计算
（笔者推测只是记号简化；见第 8 节）。GRPO 的详细讲解在第 8 课。

### 2.7 奖励，式 (35)–(36)（照录）

$$r_{i}=RM_{reasoning}(o_{i}).\quad(35)$$
$$r_{i}=c_{1}\cdot RM_{helpful}(o_{i})+c_{2}\cdot RM_{safety}(o_{i})+c_{3}\cdot RM_{rule}(o_{i}),\quad(36)$$
“where $c_{1}$, $c_{2}$, and $c_{3}$ are corresponding coefficients.”（系数值未公布。）

### 2.8 课程记号对照与**符号冲突**（必须在讲义里处理）

| 论文符号（本课） | 冲突对象（style-guide §2） | 建议讲义写法 |
|---|---|---|
| $D$ = 设备组数（式 26–31） | $D$ = 训练 token 数 | 写 $D_{\text{dev}}$（=8），首次出现说明“论文里就是 $D$” |
| $M$ = 每 token 最多去的设备数 | $M$ = 每 token 非嵌入 FLOPs（DeepSeek LLM 记号） | 写 $M_{\text{dev}}$（=3） |
| $T$ = “the number of tokens in a sequence”（式 24） | $T$ = 序列长度 | 一致，无冲突 |
| $\mathbf u_t$ = FFN 输入（§2.2.1） | $\mathbf u_t$ = 注意力层输出（style-guide；也是 `dsv2` 式 19 的用法） | 论文同一符号两用：式 (19) 是注意力输出，式 (20) 又当 FFN 输入且 $\mathbf h'_t=\mathbf u_t+\cdots$ 含残差、省略了 RMSNorm。讲义可沿用 $\mathbf u_t$，但要写明“这里的 $\mathbf u_t$ 已是加过注意力残差、再经 RMSNorm 的 FFN 输入”（笔者按图 2 与代码理解） |
| YaRN 的 $s$（缩放比） | $s_{i,t}$（亲和度） | 写 $s_{\text{YaRN}}$ 或 $\kappa$ |
| YaRN 的 $\alpha,\beta$ | $\alpha_{1,2,3}$（均衡系数）、$\beta$（KL 系数）、AdamW 的 $\beta_1,\beta_2$ | 写 $\alpha_{\text{Y}},\beta_{\text{Y}}$ |
| YaRN 的 $t$ | $t$（token 位置） | 写 $\tau$ 或“温度系数”，并说明 V2 与 YaRN 的方向相反（§2.5） |
| YaRN 的 $\theta_d$、$d$、$|D|$ | $\theta$（参数）、$d$（隐藏维）、$D$ | 频率写 $\theta_i$，下标用 $i$，头维写 $d_h^R$ |
| $\mathds 1(\cdot)$ | 宏表里无 `\mathds` | 写 $\mathbb 1(\cdot)$ 或 $\mathbf 1[\cdot]$ |
| $A_i$ | $\hat A$ | 写 $\hat A_i$ |
| $\operatorname{Topk}$ | 宏 `\TopK` | 写 $\TopK(\cdot,K_r)$ |
| $\varepsilon,\beta$（GRPO） | 同 style-guide | 一致 |

---

## 3. 规格与超参数表

### 3.1 DeepSeek-V2（每格注明来源）

| 项目 | 值 | 来源 |
|---|---|---|
| 总参数 / 激活参数 | 236B / 21B | 摘要；笔者按 config 复算 235.74B / 20.85–21.38B（4.1 节） |
| 层数 / 隐藏维 | 60 / 5120 | §3.1.2；config `num_hidden_layers`、`hidden_size` |
| 词表 | “vocabulary size of 100K”（BBPE） | §3.1.1；config `vocab_size` = **102400** |
| 第 1 层 | 稠密 FFN，中间维 12288 | §3.1.2 只说“except for the first layer”；12288 仅见 config `intermediate_size`，`first_k_dense_replace` = 1 |
| MoE | 2 共享 + 160 路由，激活 6，专家中间维 1536 | §3.1.2；config `n_shared_experts` 2、`n_routed_experts` 160、`num_experts_per_tok` 6、`moe_intermediate_size` 1536 |
| 门控打分 | softmax（式 22） | §2.2.1；config `scoring_func` "softmax" |
| 门控缩放 | 不归一化，乘 16 | config `norm_topk_prob` false、`routed_scaling_factor` 16.0；论文 §3.1.2 只说 “multiply additional scaling factors at the width bottlenecks” |
| 设备受限路由 | $D=8$，$M=3$ | §3.1.2；config `n_group` 8、`topk_group` 3、`topk_method` "group_limited_greedy" |
| 均衡系数 | $\alpha_1=0.003$，$\alpha_2=0.05$，$\alpha_3=0.02$ | §3.1.2；config `aux_loss_alpha` = **0.001**（与论文 0.003 不符），`seq_aux` true |
| token 丢弃 | 设备级，容量因子 1.0，约 10% 序列永不丢；评测不丢 | §2.2.4、§3.1.2 |
| 初始化 | std 0.006 | §3.1.2；config `initializer_range` = **0.02** |
| 优化器 | AdamW，β1 0.9，β2 0.95，wd 0.1 | §3.1.2 |
| 学习率 | 峰值 2.4e-4；前 2K 步线性 warmup；约 60%、90% tokens 处各 ×0.316 | §3.1.2 |
| 梯度裁剪 | 1.0 | §3.1.2 |
| batch | 前 225B tokens 从 2304 增到 9216（序列），之后 9216 | §3.1.2 |
| 预训练长度 / 数据 | 4K / 8.1T tokens（中文比英文多约 12%） | §3.1.1–3.1.2 |
| 并行 | 16 路零气泡 PP，8 路 EP，ZeRO-1 DP，无 TP | §3.1.3 |
| 硬件 | H800，每节点 8 卡（NVLink+NVSwitch），节点间 InfiniBand | §3.1.3 |
| 长上下文 | YaRN：$s=40$、$\alpha=1$、$\beta=32$，目标 160K；$\sqrt t=0.0707\ln s+1$；32K × 576 序列 × 1000 步 | §3.1.4；config `rope_scaling`：`factor` 40、`beta_slow` 1、`beta_fast` 32、`mscale`/`mscale_all_dim` 0.707、`original_max_position_embeddings` 4096；`max_position_embeddings` 163840（= 160K） |
| SFT | 1.5M 条（1.2M 有用性 + 0.3M 安全），2 epochs，lr 5e-6 | §4.1 |
| RL | GRPO；阶段 1：$RM_{reasoning}$；阶段 2：$c_1RM_{helpful}+c_2RM_{safety}+c_3RM_{rule}$；在线 RL | §4.2、§4.4 |
| 训练成本 | 172.8K GPU·h / T tokens（67B：300.6K） | §3.2.3 |
| 部署 | FP8 权重，KV 平均 6 bit；8×H800 单节点生成 >50K tok/s，预填充 >100K tok/s | §3.2.3 |
| 发布日期 | 2024-05-06（GitHub）；arXiv v1 2024-05-07 | `gh-v2` News；arXiv |

### 3.2 V2 家族对比

| 模型 | 总/激活 | 层 | $d$ | 头 | 查询压缩 | 路由/共享/激活 | 专家中间维 | 训练 tokens | 上下文 | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2 | 236B / 21B | 60 | 5120 | 128 | 1536 | 160 / 2 / 6 | 1536 | 8.1T | 128K | `dsv2` §3.1.2、表 3 |
| DeepSeek-V2-Lite | 15.7B / 2.4B | 27 | 2048 | 16 | 无 | 64 / 2 / 6 | 1408 | 5.7T（从零） | 32K | `dsv2` 附录 B、表 6；README 写“16B” |
| DeepSeek-Coder-V2 | 236B / 21B | 同 V2 | | | | | | 4.2T + 6T = 10.2T | 128K | `coderv2` §3.2–3.3、表 2 |
| DeepSeek-Coder-V2-Lite | 16B / 2.4B | 同 V2-Lite | | | | | | 4.2T + 6T | 128K | `coderv2` 表 2；`gh-coderv2` |
| DeepSeekMoE 16B（参照） | 16.4B / 2.8B | 28 | 2048 | 16（MHA） | — | 64 / 2 / 6 | 1408 | 2T | 4K | `dsv2` 表 6；`hf-moe16b-config` |

V2-Lite 其它训练超参数（附录 B.1）：峰值 lr 4.2e-4，约 80% 与 90% tokens 处各 ×0.316，固定 batch 4608 序列，只用 $\alpha_1=0.001$（所有专家同卡）；config `n_group`=1、`topk_method`="greedy"、`routed_scaling_factor`=1.0、`aux_loss_alpha`=0.001（与论文一致）。
Lite Chat 只有 SFT（v4 起的表述；README 表格也写 “DeepSeek-V2-Lite-Chat (SFT)”）。

### 3.3 Coder-V2 的 YaRN 设置：论文 vs config

| 模型 | 论文 | config `rope_scaling.mscale` / `mscale_all_dim` | 对应 logit 放大 |
|---|---|---|---|
| DeepSeek-V2 | $\sqrt t=0.0707\ln s+1$ | 0.707 / 0.707 | $1.2608^2=1.5896$ |
| DeepSeek-V2-Lite | 未单列 | 0.707 / 0.707 | 1.5896 |
| DeepSeek-Coder-V2-Instruct | “same as DeepSeek-V2: s=40, α=1, β=32”（未提缩放因子） | **1.0 / 1.0** | $1.3689^2=1.8739$ |
| DeepSeek-Coder-V2-Lite-Base | 同上 | 0.707 / 0.707 | 1.5896 |
| DeepSeek-V3（参照） | — | 1.0 / 1.0 | 1.8739 |

### 3.4 对话模型成绩（仅列讲义可能用到的；表 3、4、5 照录）

| 基准 | 67B Chat | V2 Chat (SFT) | V2 Chat (RL) | 来源 |
|---|---|---|---|---|
| MT-Bench | 8.35 | 8.62 | 8.97 | 表 4 |
| AlpacaEval 2.0（LC win rate） | 16.6 | 30.0 | 38.9 | 表 4 |
| AlignBench 总分 | 6.43 | 7.74 | 7.91 | 表 5 |
| GSM8K | 84.1 | 90.8 | 92.2 | 表 3 |
| MATH | 32.6 | 52.7 | 53.9 | 表 3 |
| HumanEval | 73.8 | 76.8 | 81.1 | 表 3 |
| LiveCodeBench（2023-09-01 至 2024-04-01） | 18.3 | 28.7 | 32.5 | 表 3 |
| BBH | 71.7 | 81.3 | **79.7** | 表 3（RL 后下降 = “alignment tax”，C32） |
| IFEval | 55.5 | 64.1 | 63.8 | 表 3 |

基础模型（表 2）：V2 MMLU 78.5、BBH 78.9、MATH 43.6、C-Eval 81.7、CMMLU 84.0；67B 分别为 71.3、68.7、18.7、66.1、70.8。

### 3.5 图 4（NIAH）读数（从 arXiv HTML 的 `needle_in_a_haystack.png` 目测，**近似值**）

- 横轴 12 个上下文长度：1K、12K、24K、35K、47K、58K、70K、81K、93K、104K、116K、128K。
- 纵轴 12 个深度（%）：0、9、18、27、36、45、55、64、73、82、91、100。色标 1–10。
- 144 格中绝大多数为 10（绿色）；例外：35K 列的 27%、36%、64% 三格呈橙红（约 3），104K 列的 45% 一格呈黄绿（约 7）。
- 图题原文：“Pressure Testing DeepSeek-V2 Base 128K Context via "Needle In A HayStack"”——测的是 **Base** 模型。

---

## 4. 复算的数字

### 4.1 参数量（按 `hf-v2-config` 逐项计数，Python）

| 部件（V2） | 表达式 | 值 |
|---|---|---|
| 每层 MLA（含 $W^O$、两个 RMSNorm） | `5120*1536+1536 + 1536*128*192 + 5120*576+512 + 512*128*256 + 16384*5120` | 149.2M |
| 每个专家（SwiGLU 三矩阵） | `3*5120*1536` | 23.59M |
| 每个 MoE 层（162 专家 + 门控 160×5120） | `162*23.59M + 160*5120` | 3.823B |
| 第 1 层稠密 FFN | `3*5120*12288` | 188.7M |
| 60 层注意力 | `60*149.2M` | 8.95B |
| 路由专家合计 | `59*160*23.59M` | 222.72B |
| 共享专家合计 | `59*2*23.59M` | 2.78B |
| 嵌入 + 输出头（不共享） | `2*102400*5120` | 1.05B |
| **总参数** | 求和 | **235.74B**（论文 236B） |
| 激活：不含嵌入与输出头 | 8.95 + 0.19 + 59×8×23.59M + 门控 | 20.33B |
| 激活：含输出头、不含输入嵌入 | +0.52B | **20.85B** |
| 激活：含嵌入与输出头 | +0.52B | 21.38B |
| V2-Lite 总参数 | 同法 | **15.71B**（论文 15.7B） |
| V2-Lite 激活（含输出头、不含输入嵌入） | 同法 | **2.45B**（论文 2.4B；含嵌入则 2.66B） |
| V3（参照）总参数 / 激活（含头不含嵌入） | 同法（不含 MTP 模块） | 671.03B / 36.63B（论文 671B / 37B） |

笔者推测：V2-Lite 的 “2.4B” 只有在“不计输入嵌入查表”时才对得上（2.45B），V2 的 “21B” 两种口径都四舍五入到 21B。讲义里说明“激活参数口径论文未定义”。

另一个整齐的事实：每 token 激活的专家总宽度 = (6+2)×1536 = **12,288** = 第 1 层稠密 FFN 的中间维（config）。

### 4.2 路由组合数（Python `math.comb`）

| 量 | 表达式 | 值 |
|---|---|---|
| 不受限：从 160 个里选 6 个 | `comb(160,6)` | 21,193,254,160 |
| 选设备：8 选 3 | `comb(8,3)` | 56 |
| 选定 3 台设备后：60 个里选 6 个 | `comb(60,6)` | 50,063,860 |
| 所有“最多跨 3 台设备”的 6 专家组合 | 容斥：$\sum_{m=1}^{3}\binom8m\sum_{j=0}^{m}(-1)^j\binom mj\binom{20(m-j)}{6}$ | 2,270,854,160（占全部的 10.7%） |
| 每台设备的专家数 | `160/8` | 20 |

### 4.3 均衡损失的数值直觉

| 情形 | 值 |
|---|---|
| 完全均匀：$\sum f_iP_i$ | 1（损失 = $\alpha_1$ = 0.003） |
| 完全均匀：$\sum f'_iP'_i$、$\sum f''_iP''_i$ | 1、1（损失 = 0.05、0.02） |
| 完全坍缩：所有 token 都选同样 6 个专家、且亲和度平均分在这 6 个上：$f=N_r/K_r$ | `160/6` = 26.67 ⇒ $\sum fP$ = 26.67（均匀时的 26.7 倍） |
| 每台设备平均接收 | $MT/D$ = 3T/8 |

### 4.4 训练成本、MFU 估算

| 量 | 表达式 | 值 |
|---|---|---|
| 每 T tokens 节省 | `1-172.8/300.6` | 42.51% |
| V2 预训练总 GPU 小时（笔者估算，假设整个 8.1T 速率恒定） | `172.8e3*8.1` | 1.40M H800·h |
| 67B 预训练总 GPU 小时（67B 训了 2.0T，`dsllm` 表 2） | `300.6e3*2` | 0.60M H800·h |
| 每卡每秒 token（V2 / 67B） | `1e12/(172.8e3*3600)`，`1e12/(300.6e3*3600)` | 1,607.5 / 924.1 |
| 按 6N 估的每 token 训练 FLOPs（V2 用 21B，67B 用 67B） | `6*21e9`，`6*67e9` | 126 GF / 402 GF |
| 注意力额外 FLOPs（fwd+bwd，平均上下文 2048，笔者估算） | V2 `3*2*128*(192+128)*2048*60`；67B `3*2*64*(128+128)*2048*95` | 30.2 GF / 19.1 GF |
| 隐含算力（仅 6N / 含注意力） | `tps*flops` | V2 203 / 251 TFLOPS；67B 371 / 389 TFLOPS |
| 相对 H800 稠密 BF16 峰值 989 TFLOPS 的 MFU（笔者估算） | 上行 / 989 | V2 20.5%–25.4%；67B 37.6%–39.3% |
| 6N FLOPs 比 vs GPU 小时比 | `21/67`，`172.8/300.6` | 0.313 vs 0.575 |

解读（笔者）：激活参数只有 67B 的 31%，GPU 小时却是 57.5%——MoE 的通信与不均衡吃掉了一部分理论收益；这正是设备受限路由、三种均衡损失、token 丢弃要对付的问题。
（989 TFLOPS 为 H100/H800 SXM 稠密 BF16 的常用数；FlashMLA 博客写 990。MFU 口径与论文无关，是笔者的量级估算。）

### 4.5 学习率与 batch 调度

| 量 | 表达式 | 值 |
|---|---|---|
| 两次衰减的位置 | `0.6*8.1`，`0.9*8.1` | 4.86T、7.29T tokens |
| 三段学习率 | `2.4e-4`，`2.4e-4*0.316`，`2.4e-4*0.316**2` | 2.4e-4 → 7.58e-5 → 2.40e-5 |
| 0.316 的来历 | `1/sqrt(10)` | 0.3162（两次后约为 1/10） |
| batch（token 数，4K 序列） | `2304*4096`，`9216*4096` | 9.44M → 37.75M tokens |
| 总步数（笔者估算，假设 225B 内 batch 线性增长） | `225e9/(5760*4096) + (8.1e12-225e9)/(9216*4096)` | ≈ 9.5K + 208.6K ≈ 218K 步 |
| V2-Lite 三段学习率 | `4.2e-4`，×0.316，×0.316² | 4.2e-4 → 1.33e-4 → 4.19e-5（在 4.56T、5.13T tokens 处） |

### 4.6 长上下文

| 量 | 表达式 | 值 |
|---|---|---|
| 目标长度 | `40*4096` | 163,840 = 160K = config `max_position_embeddings` |
| 128K | `128*1024` | 131,072 |
| V2 长上下文阶段 token 数 | `1000*576*32768` | 18.87B（预训练的 0.233%） |
| Coder-V2 两阶段 | `1000*1152*32768`，`1000*288*131072` | 37.75B + 37.75B |
| V2 的 logit 放大 | `(0.1*0.707*log(40)+1)**2` | $\sqrt t$ = 1.2608，放大 1.5896 |
| YaRN 原式在 s=40 | `(0.1*log(40)+1)**2` | 1.3689，放大 1.8739 |

### 4.7 推理吞吐与显存预算

| 量 | 表达式 | 值 |
|---|---|---|
| 每卡生成吞吐（按 50K） | `50000/8` | 6,250 tok/s |
| 按图 1(b) 反算 | `55374/8` | 6,922 tok/s |
| 67B 的最大生成吞吐（按 50K/5.76） | `50000/5.76` | 8,681 tok/s（图 1(b) 反算 ≈ 9,608） |
| 8×80GB 显存里放 KV 的 token 容量（笔者示意，忽略激活与碎片） | V2：`(640e9-236e9)/25920`；67B：`(640e9-134e9)/389120` | V2（FP8 权重 + 6-bit KV）15.6M tokens；67B（bf16）1.30M tokens，约 12 倍 |
| 若 V2 用 bf16 权重 + bf16 KV | `(640e9-472e9)/69120` | 2.43M tokens |

### 4.8 YaRN 在 64 维 RoPE 上的逐频率分配（$L=4096$，$b=10000$，$|D|=d_h^R=64$，$\alpha=1$，$\beta=32$）

| 频率下标 $i$ | 波长 $\lambda_i$（token） | $r=L/\lambda$ | YaRN $\gamma(r)$（1=不插值） | HF 代码的“保持原频率”权重 |
|---|---|---|---|---|
| 0 | 6.3 | 651.9 | 1 | 1 |
| 5 | 26.5 | 154.6 | 1 | 1 |
| 10 | 111.7 | 36.66 | 1 | 1 |
| 11 | 149.0 | 27.49 | 0.855 | 0.923 |
| 12 | 198.7 | 20.62 | 0.633 | 0.846 |
| 14 | 353.3 | 11.59 | 0.342 | 0.692 |
| 16 | 628.3 | 6.52 | 0.178 | 0.538 |
| 18 | 1117.3 | 3.67 | 0.086 | 0.385 |
| 20 | 1986.9 | 2.06 | 0.034 | 0.231 |
| 22 | 3533.3 | 1.16 | 0.005 | 0.077 |
| 23 | 4711.7 | 0.87 | 0 | 0 |
| 31 | 47117.2 | 0.087 | 0 | 0 |

表达式：`theta=10000**(-2*i/64)`，`lam=2*pi/theta`，`r=4096/lam`；代码区间 `low=floor(64*ln(4096/(32*2π))/(2 ln 10000))=10`，`high=ceil(64*ln(4096/(1*2π))/(2 ln 10000))=23`。
读法：前 11 对（$i\le10$，波长 ≤112 token）完全保留原频率；$i\ge23$（波长 >4096）全部按 $s=40$ 压慢；中间 12 对过渡。

### 4.9 GRPO 优势示例（式 34）

一组 $G=8$ 个回答，奖励 $[1,0,0,1,1,0,0,0]$：mean = 0.375；总体标准差 0.4841，样本标准差 0.5175。
- 用总体 std：正确回答 $A=+1.291$，错误回答 $A=-0.775$；
- 用样本 std：$+1.208$ / $-0.725$。
论文与 DeepSeekMath 都没说 std 用哪种（见第 8 节）。

### 4.10 Coder-V2 token 账（笔者估算）

| 量 | 表达式 | 值 |
|---|---|---|
| 总曝光 | `4.2+6` | 10.2T（与原文一致） |
| 中间检查点占 V2 全程 | `4.2/8.1` | 51.9% |
| 6T 中代码/数学/自然语言（若 60/10/30 指 6T） | `0.6*6, 0.1*6, 0.3*6` | 3.6T / 0.6T / 1.8T |
| 代码语料重复次数 | `3.6e12/1170e9` | ≈3.1 遍 |
| 数学语料重复次数 | `0.6e12/221e9` | ≈2.7 遍 |

---

## 5. 图表清单与重绘建议

### 5.1 论文中的图（本课相关）

| 图 | 内容 | 建议 |
|---|---|---|
| `dsv2` 图 1(a) | MMLU vs 激活参数散点（V2 红星约 (21, 78.5)） | 可重绘为引子图；数据点只用论文表 2 能核对的：V2 21B/78.5、67B 67B/71.3、Qwen1.5 72B 72B/77.2、Mixtral 8x22B 39B/77.6、LLaMA3 70B 70B/78.9 |
| `dsv2` 图 1(b) | 训练成本 / KV / 吞吐三组横条 | 第 9 课已重绘；本课可只用训练成本那一组 |
| `dsv2` 图 2 | V2 整体架构：Transformer 块 ×L；右上 DeepSeekMoE：输入 $\mathbf u_t$ → 路由器 → Top-$K_r$ → 路由专家（蓝）与共享专家（绿）→ 相加得 $\mathbf h'_t$ | **重绘** |
| `dsv2` 图 4 | NIAH 热力图 | **重绘**（数据见 3.5 节） |
| `coderv2` 图 2 | Coder-V2 的 NIAH | 不重绘 |
| `coderv2` 图 3 | 奖励模型 vs 编译器信号 | 可选（本次未读图中数值） |

### 5.2 建议本课绘制的图（12 张）

**图 10-1 V2 整体架构**（据 `dsv2` 图 2 重绘）
- 左：60 个块；每块 RMSNorm → MLA → ⊕；RMSNorm → FFN → ⊕。块 1 的 FFN 标“稠密，中间维 12288”，块 2–60 标“MoE”。
- 右：MoE 层：2 个共享专家（`--fig-violet`，全部激活）+ 160 个路由专家分 8 列（每列 20 个，代表 8 台设备，`--fig-muted`），
  其中 6 个被选中（`--fig-blue`），且只落在 3 列里。标注“专家中间维 1536；激活宽度 8×1536 = 12288”。

**图 10-2 设备受限路由的一次决策**（笔者构造的 4 设备 × 4 专家玩具例，Python 验证）
- 亲和度（和为 1）：A: 0.20, 0.03, 0.025, 0.05；B: 0.04, 0.12, 0.10, 0.015；C: 0.15, 0.01, 0.02, 0.035；D: 0.08, 0.055, 0.03, 0.04。
- 设备得分（组内最大）：A 0.20、B 0.12、C 0.15、D 0.08 ⇒ $M=2$ 选 A、C。
- 不受限 top-4：e1(A) 0.20、e9(C) 0.15、e6(B) 0.12、e7(B) 0.10，跨 3 台设备，门控和 0.57。
- 受限 top-4：e1 0.20、e9 0.15、e4(A) 0.05、e12(C) 0.035，门控和 0.435（V2 代码再 ×16：3.2、2.4、0.8、0.56）。
- 看点：B 有两个很强的专家（0.12、0.10，组内和 0.275 排第二），但因为按“组内最大值”打分被整台排除——按最大值还是按求和选设备是一个设计选择
  （V3 的 HF 代码改为组内前 2 名之和；若按前 2 名之和，本例设备得分 A 0.25、B 0.22、C 0.185、D 0.135，会选 A、B，
  得 {e1, e6, e7, e4}，门控和 0.47，高于按最大值选设备的 0.435，但仍丢掉了 C 上的 e9（0.15）——两种规则各有取舍，Python 已验证）。

**图 10-3 三种均衡损失的“统计对象”**
- 三行并排：专家级（160 个柱，$f_i$ 与 $P_i$）；设备级（8 个柱，$f'_i$ = 组内 $f$ 平均，$P'_i$ = 组内 $P$ 之和）；通信级（8 个柱，$f''_i$ = 被送到设备 $i$ 的 token 比例 × $D/M$）。
- 每行右侧写系数 0.003 / 0.05 / 0.02 与“均匀时 $\sum fP=1$”。

**图 10-4 token 丢弃示意**（笔者玩具数）
- 4 台设备收到的（token, 专家）分配数：14、10、9、7，平均 10 ⇒ 容量 10（容量因子 1.0）。设备 1 按亲和度从低到高丢掉 4 个（`--fig-red`），总丢弃率 4/40 = 10%。
- 旁注“约 10% 的训练序列永不丢弃，使推理时丢或不丢都与训练一致”；“评测时不丢”。

**图 10-5 NIAH 热力图**（据 `dsv2` 图 4 重绘，近似读数见 3.5 节）
- 12×12 网格，全 10 分，例外格：(35K, 27%)≈3、(35K, 36%)≈3、(35K, 64%)≈3、(104K, 45%)≈7。说明文字写明“近似读数，取自论文图 4”。

**图 10-6 训练流水线时间线**
- 预训练 8.1T @4K（batch 2304→9216 于前 225B；lr 2K 步 warmup，4.86T 与 7.29T 处 ×0.316）
  → YaRN 1000 步 @32K×576（18.9B tokens）→ SFT 1.5M（2 epochs，5e-6）→ RL 阶段 1（$RM_{reasoning}$）→ RL 阶段 2（helpful / safety / rule）。
- 分支：在 4.2T 处引出 Coder-V2：+6T（代码 60%、数学 10%、自然语言 30%）→ 32K×1152×1000 步 → 128K×288×1000 步 → SFT（300M tokens）→ GRPO（约 40k 提示）。
- 汇合：2024-09-05 V2.5 = V2-0628 + Coder-V2-0724。

**图 10-7 学习率与 batch 调度曲线**（计算得出）
- 横轴 tokens（0–8.1T）；左纵轴 lr：0→2.4e-4（前 2K 步；按起始 batch 9.44M tokens/步估约 19B tokens ≈ 0.02T，在 8.1T 的横轴上几乎是竖直上升，画成竖线并注明）、
  4.86T 处降到 7.58e-5、7.29T 处降到 2.40e-5；右纵轴 batch：前 225B 从 2304 线性升到 9216（“gradually”，线性是假设），之后平。

**图 10-8 YaRN 频率分配**（计算得出，数据见 4.8 节）
- 横轴频率下标 $i$=0…31；纵轴“保留原频率的权重”；两条线：YaRN 式 (11)（对 $r$ 线性）与 HF 代码（对 $i$ 线性）；
  背景标出三个区：外推（$r>32$）、过渡、插值（$r<1$）。

**图 10-9 GRPO 两阶段奖励结构**
- 阶段 1：同一题采样 $G$ 个回答 → $RM_{reasoning}$ 打分 → 式 (34) 组内标准化 → 式 (32) 更新。
- 阶段 2：三个奖励模型并联，按 $c_1,c_2,c_3$ 加权（式 36）。奖励模型都从 V2 Chat (SFT) 初始化；代码偏好来自编译器反馈、数学偏好来自标准答案。

**图 10-10 训练成本：每 T 与总量**（计算得出）
- 两组柱：每 T tokens GPU·h（300.6K vs 172.8K）与总 GPU·h（67B 2.0T → 0.60M；V2 8.1T → 1.40M，估算）。强调“省 42.5% 是单价，不是总价”。

**图 10-11 V2 参数构成**（计算得出，4.1 节）
- 堆叠柱“总参数 235.7B”：路由专家 222.7B、注意力 8.95B、共享专家 2.78B、嵌入+输出头 1.05B、稠密层 0.19B；
  旁边“激活 20.9B”：注意力 8.95B、激活路由专家 8.35B、共享 2.78B、输出头 0.52B、稠密 0.19B。

**图 10-12 V2 家族关系图**
- 节点：V2（2024-05-06，236B/21B）、V2-Lite（2024-05-16，15.7B/2.4B，从零 5.7T）、Coder-V2 / Coder-V2-Lite（arXiv 2024-06-17，4.2T+6T）、V2-Chat-0628、Coder-V2-0724、V2.5（2024-09-05）。
- 边：“中间检查点续训”“合并”。

---

## 6. 算例与练习素材（附答案）

**E1（计算）数参数。** 用 config（$d=5120$，专家中间维 1536，160+2 专家，59 个 MoE 层，第 1 层稠密 12288，MLA 每层 149.2M，词表 102400）算总参数与激活参数。
答：总 235.7B；激活 20.9B（含输出头不含嵌入）或 21.4B（都含）；论文 236B / 21B。

**E2（观察）激活宽度。** 每 token 激活的专家（6 路由 + 2 共享）中间维之和是多少？和第 1 层稠密 FFN 比？
答：8×1536 = 12,288，恰好等于稠密层的中间维 12,288：V2 每 token 的 FFN 算力与一个 12288 宽的稠密 FFN 相当，但可选的“知识”分布在 162 个专家里。

**E3（计算）设备受限路由。** 用图 10-2 的亲和度，$K=4$，分别求不受限与 $M=2$ 受限时选中的专家和门控和。
答：不受限 {e1, e9, e6, e7}，和 0.57，跨 A、B、C 三台；受限选设备 A、C，得 {e1, e9, e4, e12}，和 0.435。

**E4（计算）组合数。** V2 不受限有多少种 6 专家组合？受限（最多 3 台）后呢？
答：21,193,254,160；2,270,854,160（10.7%）。组合空间只缩小一个数量级，灵活性仍然很大。

**E5（推导）均衡损失的尺度。** 证明完全均匀时式 (23)、(26)、(29) 的求和项都等于 1；再算“所有 token 都去同样 6 个专家”时专家级求和项。
答：见 2.3 节；坍缩时 $f=N_r/K_r=26.67$，$\sum fP=26.67$。

**E6（计算）训练成本。** 每 T tokens 省了多少？V2 全程与 67B 全程各花多少 GPU 小时？
答：42.5%；V2 ≈ 1.40M，67B ≈ 0.60M——V2 的总账反而更高，因为它多训了 4 倍数据。

**E7（估算）MFU。** 用 6N（V2 取 21B）与 H800 稠密 BF16 峰值 989 TFLOPS 估 V2 与 67B 的 MFU。
答：V2 约 20.5%（含注意力约 25%），67B 约 37.6%（含注意力约 39%）。讨论：为什么 MoE 的 MFU 低（all-to-all 通信、不均衡、小矩阵）。

**E8（计算）学习率。** 峰值 2.4e-4，两次 ×0.316，最终学习率？发生在多少 tokens？
答：2.40e-5；衰减点 4.86T 与 7.29T。0.316 ≈ 1/√10。

**E9（计算）YaRN。** V2 的 logit 放大系数是多少？与 YaRN 原式在 $s=40$ 时相比？目标长度为何是 160K？
答：$1.2608^2=1.5896$ vs $1.3689^2=1.8739$；$40\times4096=163{,}840$ = 160K，留出 128K 以上的余量。

**E10（计算）哪些 RoPE 频率被插值。** $L=4096$，$b=10000$，64 维 RoPE，$\alpha=1$，$\beta=32$：哪些频率完全不动、哪些完全按 40 倍压慢？
答：$r>32$ 的 $i\le10$（波长 ≤112）不动；$r<1$ 的 $i\ge23$（波长 ≥4712）压慢；$i=11$–22 过渡。

**E11（计算）GRPO 优势。** 奖励 $[1,0,0,1,1,0,0,0]$，求 $A_i$；若一组全对呢？
答：+1.291 / −0.775（总体 std）；全对时分子全 0、分母 0——实现里要加 ε 或跳过，这组题不产生学习信号（style-guide 类比 5 的“边界”）。

**E12（计算）长上下文训练量。** V2 的 32K 阶段共多少 tokens？占预训练多少？Coder-V2 两阶段呢？
答：18.87B，0.233%；Coder-V2 各 37.75B。

**E13（计算）显存预算。** 8×80GB，V2 FP8 权重 236GB、KV 25,920 B/token；67B bf16 权重 134GB、KV 389,120 B/token。各能放多少 token 的 KV？
答：15.6M vs 1.30M（约 12 倍），这是“最大生成吞吐 5.76 倍”的主要来源之一（笔者推测；吞吐还受算力与调度限制）。

**E14（计算）Coder-V2 的 token 账。** 答：见 4.10 节：10.2T；代码语料约重复 3.1 遍，数学约 2.7 遍（假设 60/10/30 指新增的 6T）。

---

## 7. 勘误与常见误解

**E1. “V2 训练成本比 67B 省 42.5%。”** 只对“每万亿 token 的 GPU 小时”成立（C24）。V2 训了 8.1T、67B 训了 2T，总 GPU 小时 V2 约 1.40M，是 67B（约 0.60M）的 2.3 倍（笔者估算）。

**E2. “93.3% / 5.76× 是 MLA 的功劳。”** 5.76× 明确是部署态（FP8 权重 + 平均 6 bit KV）的 V2 对 67B（§3.2.3 先讲部署优化、再报吞吐）；93.3% 也对应部署态，但这一点是笔者由数值吻合（`34560*6/8` = 25,920 B = 图 1(b) 的 25.9 KB）推断，论文未明说。纯 MLA（bf16）的 KV 降幅是 82.2%（`lesson-09-doc.md` E1）。42.5% 的训练节省论文明确归因于稀疏（“sparse DeepSeek-V2”），不是 MLA。

**E3. “V2 在 128K 上训练过。”** 没有。长上下文阶段只在 32K 上训 1000 步（batch 576），128K 靠 YaRN 外推，目标上限设为 160K（C20）。Coder-V2 才另有 128K 阶段（C40）。

**E4. “NIAH 全绿。”** 论文说 “performs well across all context window lengths up to 128K”，但图 4 在 35K 处有三格明显失分、104K 有一格部分失分（3.5 节）；而且测的是 Base 模型。讲义写“基本全部通过”。

**E5. “YaRN 作用在所有注意力维度上。”** 论文原话是“specifically applied to the decoupled shared key $\mathbf k^R_t$”；代码里 RoPE 频率修改作用在 64 维 $\mathbf q^R$ 与 $\mathbf k^R$ 上（查询也必须同样旋转），而温度（长度缩放）改为乘在整个 192 维 logit 上（`softmax_scale *= mscale**2`，cos/sin 缩放为 1）。

**E6. “V2 的 $\sqrt t$ 就是 YaRN 的 $\sqrt t$。”** 方向相反：YaRN 式 (14) 除以 $t$，式 (15) 给的是 $\sqrt{1/t}$；V2 写 “$\sqrt t=0.0707\ln s+1$”，按代码是乘在 logit 上的放大系数。数值上 0.0707 = 0.1 × 0.707（config `mscale`）。

**E7. “V2 的 RL 用的是规则奖励（像 R1 那样）。”** 不对。阶段 1 用学出来的 $RM_{reasoning}$（偏好数据来自编译器反馈与标准答案）；阶段 2 三个奖励模型中只有一个是 rule-based。Coder-V2 还专门比较了“奖励模型信号 vs 原始编译器信号”，选了前者（C41）。

**E8. “V2 第二阶段有 3 个奖励模型，所以 V2 一共 3 个 RM。”** 一共 4 个：$RM_{reasoning}$、$RM_{helpful}$、$RM_{safety}$、$RM_{rule}$（其中 $RM_{rule}$ 是基于规则的“奖励模型”）。

**E9. “GRPO 在 V2 里是逐 token 的。”** V2 的式 (32) 把重要性比写在整条回答上；DeepSeekMath 的原式是逐 token 并除以 $|o_i|$。两者不是同一个式子，讲义以 DeepSeekMath 为准讲 GRPO，并指出 V2 写法是简化（实现细节未公开）。

**E10. “Coder-V2 是在 V2 最终模型上续训的。”** 不对：从“intermediate checkpoint”（已训 4.2T）续训 6T，总 10.2T；Lite 版表 2 同样写 “4.2T+6T”。

**E11. “V2-Lite 是 V2 蒸馏或剪枝来的。”** 不对：“trained from scratch”，5.7T tokens；且不压缩查询、只用专家级均衡损失（$\alpha_1=0.001$）、专家都在同一张卡上。

**E12. “V2-Lite Chat 做过 RL。”** v3 说做过，v4/v5 改为只有 SFT；README 表格也写 “(SFT)”。按 v5。

**E13. config 与论文不一致的字段。** `aux_loss_alpha`=0.001（论文 α1=0.003）；`initializer_range`=0.02（论文 0.006）；`vocab_size`=102400（论文 “100K”）。这些字段对推理无影响，引用超参数以论文为准，并注明。

**E14. “设备级均衡损失是 V2 的新发明。”** 出自 DeepSeekMoE（`dsmoe` §3.3，C12）；V2 新增的是设备受限路由、通信均衡损失和设备级 token 丢弃。

**E15. README 与论文数字不一致。** `gh-v2` README 的基础模型表里 Mixtral 8x22B 的 C-Eval 写 58.6，论文表 2 为 59.6。以论文为准。

**E16. “V2 用 top-k 后重新归一化的门控。”** 不：`norm_topk_prob=False`，门控值是原始 softmax 亲和度（式 21），代码再乘常数 16。V3 才改成 sigmoid + 归一化（第 11 课）。

**E17. “V2 训练用了张量并行。”** 论文明说 “it can be trained without the necessity of tensor parallelism”（C19）。

---

## 8. 未能核实 / UNVERIFIED

1. **GRPO 的超参数**（组大小 $G$、$\varepsilon$、$\beta$、学习率、步数）与奖励系数 $c_1,c_2,c_3$：V2 论文均未公布。
2. **V2 的 GRPO 是否真的用序列级重要性比**（式 32 的写法）：未公开实现。
3. **$\mathrm{std}$ 用总体还是样本标准差**（式 34）：论文与 DeepSeekMath 均未说明。
4. **设备受限路由的设备打分方式**：论文文字与 HF 推理代码（组内最大）一致，但训练代码未开源，无法确认训练时完全相同。
5. **设备级、通信级损失与 token 丢弃的实现**：HF 代码不含（推理代码）；`seq_aux=True` 表明专家级损失按序列计算，与式 (24) “tokens in a sequence” 一致。
6. **“additional scaling factors” 的全部取值**：只在 config 里看到 `routed_scaling_factor=16.0`；潜向量处的缩放未见显式常数。
7. **42.5% 与 172.8K GPU·h 是否包含 MoE 的 token 丢弃带来的加速**：论文未拆分。V2 全程总 GPU 小时是笔者按单价 × 8.1T 估算，论文未给总数。
8. **67B 在吞吐对比中的部署精度与 batch**：未说明；显存预算算例（E13）是笔者示意。
9. **batch 从 2304 增到 9216 的具体曲线**（线性或阶梯）：论文只说 “gradually increased”；总步数 ≈218K 为笔者估算。
10. **NIAH 图 4 的精确分数**：只能从 PNG 目测色块，近似。
11. **Coder-V2 的 “exponential normalization technique” 指什么**：论文未解释；config 层面看到 Coder-V2-Instruct 的 `mscale`=1.0（V2 为 0.707），但无法确认两者有关。
12. **Coder-V2-Lite 的“中间检查点”是 V2-Lite 的还是 V2 的**：表 2 只写 “4.2T+6T”，§3.3 说“intermediate checkpoint of DeepSeek-V2”；V2-Lite 全程只训 5.7T，4.2T 处检查点来自 V2-Lite 的可能性较大（笔者推测）。
13. **60/10/30 配比是指新增 6T 还是全部 10.2T**：原文语境为“pre-training phase”的数据集，笔者按 6T 计算。
14. **DeepSeek-V2 API 定价与 2024 年中国大模型“价格战”**：常见说法（每百万输入 token 1 元等），本次未找到可引用的一手页面，勿写入讲义或写为“据报道”。
15. **YaRN、DeepSeekMoE、GQA 的正式出版信息**（ICLR 2024、ACL 2024、EMNLP 2023）：本次只核对了 arXiv 元数据。
16. **“Riquelme et al. (2021)” 的按亲和度丢 token（Batch Prioritized Routing）的具体规则**：只核对了该论文元数据，没读正文。
17. **H800 的 BF16 稠密峰值**：用的是 989/990 TFLOPS 这一常用数（FlashMLA 博客写 990），未在 NVIDIA 官方规格页核对 H800。

---

## 9. 参考文献卡片草稿

```
[dsv2] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
    a: DeepSeek-AI (2024), preprint. arXiv:2405.04434v5
    url: https://arxiv.org/abs/2405.04434v5
    finding: 236B/21B 的 MoE 模型在 8.1T tokens 上预训练，用设备受限路由（M=3/D=8）、三种均衡损失和设备级 token 丢弃训练，再经 YaRN 扩到 128K、SFT 与两阶段 GRPO。
    matters: 本课所有规格、超参数、训练与对齐流程的原始出处，头条数字的口径以它 §3.2.3 的原话为准。

[coderv2] DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence
    a: DeepSeek-AI (Qihao Zhu et al.) (2024), preprint. arXiv:2406.11931v1
    url: https://arxiv.org/abs/2406.11931
    finding: 从已训 4.2T 的 V2 中间检查点再续训 6T（代码 60%/数学 10%/自然语言 30%），支持 338 种语言与 128K 上下文，并用两阶段长上下文训练与 GRPO。
    matters: 说明同一套 V2 架构如何通过“中间检查点续训”派生出专用模型，也是 V2.5 合并的一半来源。

[dsmoe] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
    a: Damai Dai et al. (2024), preprint (ACL 2024, unverified here). arXiv:2401.06066v1
    url: https://arxiv.org/abs/2401.06066
    finding: 提出细粒度专家切分与共享专家隔离，并引入设备级均衡损失（小专家级系数 + 大设备级系数）。
    matters: 用来划清 V2 的“继承”与“新增”：设备级损失是继承，设备受限路由与通信均衡损失才是 V2 新增。

[dsmath] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
    a: Zhihong Shao et al. (2024), preprint. arXiv:2402.03300v3
    url: https://arxiv.org/abs/2402.03300
    finding: GRPO 原始目标是逐 token、按 1/|o_i| 平均的裁剪目标加 k3 形式的 KL 惩罚，用组内奖励标准化代替价值网络。
    matters: V2 的式 (32) 是它的序列级简写，讲义讲 GRPO 公式时以此为准并交叉引用第 8 课。

[yarn] YaRN: Efficient Context Window Extension of Large Language Models
    a: Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole (2023), preprint (ICLR 2024, unverified here). arXiv:2309.00071
    url: https://arxiv.org/abs/2309.00071
    finding: 按每个 RoPE 维度在原上下文内转过的圈数 r 分段插值（α、β 为边界，LLaMA 上取 1 与 32），并用 sqrt(1/t)=0.1 ln s + 1 的温度修正注意力熵。
    matters: V2 的 s=40、α=1、β=32 与 0.0707 ln s + 1 都要对照它的式 (10)(11)(13)(14)(15) 才能讲清。

[zb] Zero Bubble Pipeline Parallelism
    a: Penghui Qi, Xinyi Wan, Guangxing Huang, Min Lin (2023), preprint. arXiv:2401.10241v1
    url: https://arxiv.org/abs/2401.10241
    finding: 提出把反向传播拆分以消除流水线气泡的调度方法。
    matters: V2 训练采用 16 路零气泡流水线并行，是第 12 课 DualPipe 之前的那一步。

[vmoe] Scaling Vision with Sparse Mixture of Experts
    a: Carlos Riquelme et al. (2021), preprint. arXiv:2106.05974v1
    url: https://arxiv.org/abs/2106.05974
    finding: V-MoE 论文，V2 称其按优先级丢弃 token 的做法启发了“丢弃亲和度最低的 token”。
    matters: 给 V2 的 token 丢弃策略一个来源；本次只核对了元数据，讲义引用时只说“受其启发”。

[dsllm] DeepSeek LLM: Scaling Open-Source Language Models with Longtermism
    a: DeepSeek-AI (2024), preprint. arXiv:2401.02954
    url: https://arxiv.org/abs/2401.02954
    finding: 67B 为 95 层 GQA-8 稠密模型、训 2.0T tokens，并提出 V2 沿用的 warmup-and-step-decay 学习率调度。
    matters: V2 所有“相对 67B”的成本、缓存、吞吐比较的基线，也是本课估算 67B 总 GPU 小时的依据。

[hf-v2] DeepSeek-V2 model repository on Hugging Face (config.json, modeling_deepseek.py)
    a: DeepSeek-AI (2024), artefact.
    url: https://huggingface.co/deepseek-ai/DeepSeek-V2
    kind: artefact
    finding: config 给出 n_group=8、topk_group=3、routed_scaling_factor=16、rope_scaling(factor 40, beta 32/1, mscale 0.707)，代码用组内最大亲和度选设备、把 YaRN 温度乘在整个 logit 上。
    matters: 用来核对论文超参数、揭示论文没写的实现细节，并记录 aux_loss_alpha 等字段与论文不一致。

[hf-v2lite] DeepSeek-V2-Lite model card and config
    a: DeepSeek-AI (2024), artefact.
    url: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
    kind: artefact
    finding: 15.7B/2.4B、从零训 5.7T、单张 40GB GPU 可推理、8×80GB 可微调，config 无查询压缩、所有专家一组。
    matters: 给学生一个能在单卡上跑的 MLA+MoE 实物，并作为 V2 规格的缩小对照。

[hf-v25] DeepSeek-V2.5 model card and API news (2024-09-05)
    a: DeepSeek-AI (2024), artefact.
    url: https://huggingface.co/deepseek-ai/DeepSeek-V2.5 ; https://api-docs.deepseek.com/news/news0905
    kind: artefact
    finding: V2.5 把 DeepSeek-V2-0628 与 DeepSeek-Coder-V2-0724 合并为一个模型，于 2024-09-05 发布。
    matters: 作为 V2 系列的收尾节点，连接到第 11 课的 V3。

[gh-coderv2] DeepSeek-Coder-V2 GitHub README
    a: DeepSeek-AI (2024), artefact.
    url: https://github.com/deepseek-ai/DeepSeek-Coder-V2
    kind: artefact
    finding: 列出 Lite（16B/2.4B）与完整版（236B/21B）的 Base/Instruct 四个模型，上下文均为 128K。
    matters: 证明 Coder-V2-Lite 的上下文（128K）与 V2-Lite（32K）不同，避免混用规格。
```

---

## 10. 核查记录（对抗式复核，2026-09-24）

复核方法：对下列高风险论断逐条回到一手来源重查。论文用 `arxiv.py text/grep` 取 LaTeXML 全文（`dsv2` v1–v5 逐版 diff；
`yarn` 取 v3；`coderv2` v1；`dsmath` v3；`dsmoe` v1；`dsllm` v1）；制品用 raw URL 原样下载（V2 / V2-Lite / Coder-V2-Instruct /
Coder-V2-Lite-Base / V3 的 `config.json`，V2 / V2-Lite / V3 的 `modeling_deepseek.py`，V2 与 Coder-V2 的 GitHub README，
V2-Lite 与 V2.5 模型卡，API 新闻页 news0905）；图 4 的色块用 PIL 按色标反查分值；派生数字全部在 Python 里重算。

### 10.1 核对无误（20 项）

| # | 论断 | 来源与方法 | 结果 |
|---|---|---|---|
| 1 | 2 共享 + 160 路由、激活 6、专家中间维 1536、除第 1 层外全换 MoE；D=8、M=3；α1/α2/α3 = 0.003/0.05/0.02；评测不丢 token | `dsv2` v5 §3.1.2 | 逐字一致 |
| 2 | 式 (20)–(31) | v5 LaTeX 逐符号比对 | 一致（含 $\frac{N_r}{K_rT}$、$\frac{D}{MT}$、式 (28) = 式 (31)） |
| 3 | 设备受限路由动机、“$M\geqslant3$ … roughly aligned”、通信均衡 “transmits at most $MT$” | v5 §2.2.2–2.2.3 | 一致 |
| 4 | token 丢弃：容量因子 1.0、受 Riquelme et al. 启发、约 10% 序列永不丢 | v5 §2.2.4 | 一致 |
| 5 | 设备级均衡损失出自 DeepSeekMoE（小专家级系数 + 大设备级系数） | `dsmoe` §3.3 | 一致 |
| 6 | 训练超参数：AdamW 0.9/0.95/0.1、2K 步 warmup、约 60%/90% 处 ×0.316、峰值 2.4e-4、裁剪 1.0、batch 2304→9216（前 225B）、4K、8.1T、词表 “100K” | v5 §3.1.1–3.1.2 | 一致 |
| 7 | 16 路零气泡 PP、8 路 EP、ZeRO-1、不需 TP；H800 节点 8 卡 NVLink+NVSwitch、节点间 IB | v5 §3.1.3 | 一致 |
| 8 | YaRN 设置：s=40、α=1、β=32、目标 160K、$\sqrt t=0.0707\ln s+1$、只作用于 $\mathbf k^R_t$、1000 步 × 32K × 576 | v5 §3.1.4 | 一致 |
| 9 | YaRN 原文式 (8)(10)(11)(13)(14)(15) 与 “α=1、β=32” 推荐值（§3.2） | `yarn` v3 LaTeX | 一致；**长度缩放引文的节号有误**，见 10.2 |
| 10 | 图 4 标题 “Pressure Testing DeepSeek-V2 Base 128K Context”；35K 列 27%/36%/64% 约 3 分，104K 列 45% 约 7 分 | `needle_in_a_haystack.png` 像素按色标反查：2.94、2.94、2.94、7.04（其余格 9.98） | 一致 |
| 11 | config：`aux_loss_alpha` 0.001、`initializer_range` 0.02、`vocab_size` 102400、`n_group` 8、`topk_group` 3、`routed_scaling_factor` 16.0、`norm_topk_prob` false、`seq_aux` true、`rope_scaling`（40 / 1 / 32 / 0.707 / 0.707 / 4096）、`max_position_embeddings` 163840、`first_k_dense_replace` 1、`intermediate_size` 12288 | HF `config.json` | 一致 |
| 12 | V2-Lite config（`n_group` 1、`greedy`、缩放 1.0、`q_lora_rank` null、mscale 0.707）、Coder-V2-Instruct（mscale 1.0）、Coder-V2-Lite-Base（0.707）、V3（1.0、`noaux_tc`、`topk_group` 4） | 各 `config.json` | 一致 |
| 13 | HF 代码：组内 `.max(dim=-1)` 选设备；`"gready"` 拼写（V2）与 `"greedy"`（V2-Lite）；`.div_(seq_len * aux_topk / n_routed_experts)`；共享专家合成一个 3072 宽 MLP；`yarn_get_mscale` 与 cos/sin 缩放 = 1；ramp 对下标线性；V3 用 `topk(2)…sum` | 三份 `modeling_deepseek.py` | 一致 |
| 14 | 式 (32)–(36) GRPO 与两阶段奖励（$RM_{reasoning}$；$c_1RM_{helpful}+c_2RM_{safety}+c_3RM_{rule}$）；RM 从 V2 Chat (SFT) 初始化；编译器反馈与标准答案 | v5 §4.2 LaTeX | 一致；DeepSeekMath 原式确为逐 token + $1/|o_i|$（**节号有误**，见 10.2） |
| 15 | SFT 1.5M（1.2M + 0.3M）、2 epochs、5e-6；引言 “1.5M conversational sessions”；§4.4 的 10K、alignment tax、online 优于 offline | v5 §1、§4.1、§4.4 | 一致 |
| 16 | 表 2（基础）、表 3、表 4、表 5 中本文件引用的全部数值（含 BBH 81.3→79.7、AlignBench 6.43/7.74/7.91）；README 的 Mixtral C-Eval 58.6 vs 论文 59.6 | v5 表格（列序 67B / Qwen1.5 72B / Mixtral 8x22B / LLaMA3 70B / V2）+ `gh-v2` README | 一致 |
| 17 | V2-Lite 附录 B：27 层、2048、16 头、不压缩查询、64 路由 + 2 共享、1408、15.7B/2.4B、从零 5.7T、4.2e-4、80%/90% 处 ×0.316、batch 4608、只用 α1=0.001；表 6 上下文 32K；v3 “SFT, and RL” → v4 “and SFT” | v5 附录 B + v3/v4 diff；V2-Lite 模型卡 | 一致 |
| 18 | Coder-V2：中间检查点 4.2T + 6T = 10.2T；86→338；16K→128K；60/10/30；1,170B / 221B；exponential normalization；FIM 0.5 仅 16B；cosine、2000 步 warmup、降到 10%；32K×1152 与 128K×288 各 1000 步；300M tokens SFT；约 40k 提示；RM 优于原始编译器信号；GRPO；表 2 “4.2T+6T” | `coderv2` v1 §1、§3.1–3.5、表 2 + README | 一致 |
| 19 | V2.5 = V2-0628 + Coder-V2-0724（2024/09/05）；模型卡 “combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct”；V2 发布 2024-05-06、V2-Lite 2024-05-16 | API 新闻页、V2.5 模型卡、`gh-v2` News | 一致 |
| 20 | 派生数字：235.74B / 20.85B / 21.38B；V2-Lite 15.71B / 2.45B；V3 671.03B / 36.63B；`comb` 各值与 10.7%；42.51%、1.40M vs 0.60M（2.33×）；1.5896 / 1.8739；18.87B、37.75B；4.8 节频率表；4.9 节优势值；图 10-2 玩具路由 | Python | 一致 |

### 10.2 修改（3 处）

1. **C23 节号**：YaRN 的 “length scaling” 引文在 v3 的 §3.3 “YaRN”（紧随式 (14)），原写 §3.4；§3.4 是 “Dynamic Scaling – Dynamic NTK”。已改。来源：`yarn` v3 章节目录与正文。
2. **2.6 节 DeepSeekMath 出处**：GRPO 目标函数是 `dsmath` 式 (3)，位于 §4.1.1 “From PPO to GRPO”，原写 §4.1.2（那是 “Outcome Supervision RL with GRPO”）。已改。
3. **勘误 E2**：原文把 93.3% 与 5.76× 都写成确定的“部署态”对比。5.76× 的部署条件由 §3.2.3 的上下文直接支持；93.3% 对应部署态只是由
   25,920 B = 图 1(b) 25.9 KB 的数值吻合推断（第 8 节与 `lesson-09-doc.md` E1 都已标为推断）。已补“笔者推断、论文未明说”。

### 10.3 核查后仍存疑

- 第 8 节所列各项未因本次复核而消除；特别是 GRPO 超参数、奖励系数、训练时的设备打分与均衡损失实现，公开材料里都查不到。
- 图 4 的分值是按色标反查的像素值（约 2.9 与 7.0），不是论文给出的数字。


---

## 11. 补充核实（写课时追加，2026-09-24）

写 `lesson-10-v2-moe-training.md` 时，笔者为讲义另外核对或计算了下列内容。一手来源用的是研究阶段下载到 scratch 目录的
`dsv2` v5 全文文本与 HF `config.json`（与第 0.2 节所列制品同一来源）；派生数字都在 Python 里复算，生成脚本在
`deepseek/figures/gen/l10_*.py`，其中有 `assert` 锁住讲义引用的数值。

### 11.1 引文与配置

| # | 内容 | 来源 | 原文 / 值 |
|---|---|---|---|
| S1 | V2 的训练框架名叫 HAI-LLM | `dsv2` v5 §3.1.3 | "DeepSeek-V2 is trained based on the HAI-LLM framework (High-flyer, 2023), an efficient and light-weight training framework developed internally by our engineers." |
| S2 | V2 的 RoPE 底数 $b=10000$（讲义 §6 与图 10.10、10.11 用它） | HF `deepseek-ai/DeepSeek-V2` `config.json` | `"rope_theta": 10000` |
| S3 | `max_position_embeddings` = 163840（= $40\times4096$） | 同上 | `"max_position_embeddings": 163840`（与 §3.1 表一致） |
| S4 | “中文 token 比英文多约 12%”再次核对 | `dsv2` v5 §3.1.1 | "...where Chinese tokens are approximately 12% more than English ones." |

### 11.2 笔者计算（讲义里均标注“笔者计算 / 估算”）

| # | 量 | 表达式 | 值 |
|---|---|---|---|
| S5 | 均匀随机从 8 台 × 20 个专家中选 6 个，平均触及的设备数 | `8*(1-comb(140,6)/comb(160,6))` | 4.4586（`comb(140,6)/comb(160,6)` = 0.44268） |
| S6 | 恰好触及 1…6 台设备的比例 | 容斥 `comb(8,m)*sum((-1)**j*comb(m,j)*comb(20*(m-j),6))/comb(160,6)` | 0.0015%、0.497%、10.22%、40.67%、40.16%、8.46%；累计 ≤3 台 = 10.715%（与 4.2 节一致） |
| S7 | 容斥分项（讲义 §2 算例） | `comb(40,6)`、`comb(20,6)`、`comb(60,6)` | 3,838,380；38,760；50,063,860；恰好 2 台 105,304,080，恰好 3 台 2,165,240,000 |
| S8 | DeepSeekMoE 16B 从 64 个路由专家里选 6 个的组合数 | `comb(64,6)` | 74,974,368；V2 受限后的 2,270,854,160 是它的 30.3 倍 |
| S9 | 假想：160 个专家分到 16 台（每台 10 个）时的平均触及设备数 | `16*(1-comb(150,6)/comb(160,6))` | 5.206；此时 $M=1$ 只剩 `16*comb(10,6)` = 3,360 种组合（练习 6） |
| S10 | 专家级均衡损失对路由 logit 的梯度 | 链式法则 + softmax 雅可比（讲义式 10.15） | $\partial\mathcal L/\partial z_{k,t}=\frac{\alpha_1}{T}s_{k,t}(f_k-\sum_if_is_{i,t})$；对 $k$ 求和为 0。三种损失同形，只是 $f$ 换成 $f'_{m(k)}$、$f''_{m(k)}$。这是笔者的推导，论文未给出 |
| S11 | warmup 覆盖的 token 数与总步数（假设 batch 随 token 线性增长） | 逐步累加 `batch(x)*4096` | 2000 步 ≈ 21.46B tokens；前 225B tokens 约 11,018 步；全程约 219,635 步（4.5 节按算术平均 batch 估为 ≈218K，两者口径不同、量级一致） |
| S12 | 若 batch 始终为 9216 条 | `8.1e12/(9216*4096)` | 214,577 步 |
| S13 | “假想账单”：稠密 67B 的单价训 8.1T | `300.6e3*8.1` | 2.43M GPU·h（比 V2 估算的 1.40M 多约 1.03M） |
| S14 | YaRN 过渡区两个频率（练习 9） | 4.8 节公式 | $i=12$：$\lambda$=198.7，$r$=20.61，$\gamma$=0.633，论文 $\theta'/\theta$=0.642，代码 0.850；$i=20$：$\lambda$=1986.9，$r$=2.061，$\gamma$=0.0342，论文 0.0584，代码 0.250 |
| S15 | 假想 $s=32$ 时的温度 | `0.0707*log(32)+1` | $\mu$=1.2450，$\mu^2$=1.5500；目标长度 131,072 |
| S16 | GRPO 优势，奖励 `[1,1,1,0]` | 总体 / 样本标准差 | 0.4330 → +0.577 / −1.732；0.5 → +0.5 / −1.5 |
| S17 | k3 估计 $x-\ln x-1$ | `x-log(x)-1` | $x$=0.5：0.1931；2：0.3069；1.1：0.00469；0.9：0.00536 |
| S18 | 玩具路由与均衡损失（图 10.4、10.6、练习 5、7） | 笔者构造的数字，脚本内 `assert` | 图 10.6：$\sum fP$=1.19、$\sum f'P'$=1、$\sum f''P''$=0.8125；练习 7：1.125、1.075、0.9125 |
| S19 | 设备级丢弃的玩具（图 10.7） | 笔者构造 | 负载 14/10/9/7，容量 10，丢 4/40 = 10%；丢弃后利用率 $9/10$ = 90%（不是 100%） |
| S20 | 一层 MLA 的参数分项（讲义 §1 算例；教学审阅时补写） | 按 `config.json`：`q_lora_rank` 1536、`kv_lora_rank` 512、128 头、`qk_nope_head_dim` 128、`qk_rope_head_dim` 64、`v_head_dim` 128 | $W^{DQ}$ 7,864,320；$W^{UQ}$（含 RoPE 部分）37,748,736；$W^{DKV}$+$W^{KR}$ 2,949,120；$W^{UK}$+$W^{UV}$ 16,777,216；$W^O$ 83,886,080；两个 RMSNorm 2,048；合计 149,227,520 |
| S21 | 最慢的一对 RoPE 维度（$i=31$）在各长度上的相对角度（图 10.10） | $L\cdot10000^{-62/64}$ | 4096：0.5462 rad = 31.30°；131072：17.479 rad = 2.782 圈；YaRN 除以 40 后 131072：25.04°，163840：31.30°（与 4096 时完全相同）；$i=0$ 在 4096 内转 651.9 圈 |
| S22 | 温度的玩具模型（讲义 §6，笔者推导） | $w=e^{\Delta}/(e^{\Delta}+N-1)$，$\Delta=8$ | $N$=4096：0.4213；$N$=131072：0.0222；恢复所需 $c=1+\ln32/8=1.4332$；V2 的 $\mu^2$ 一次项系数 $2\times0.0707=0.1414$ ↔ $\Delta\approx7.07$，YaRN 0.2 ↔ $\Delta=5$ |
| S23 | V2 尺度的丢弃示意（讲义 §4 算例，假设的数） | $T_g$=4096，最忙设备 = 1.2 × 容量 | 容量 3072；最忙 3686；丢 614 份 = 全组 24,576 份的 2.5% |
| S24 | 负载 14/10/9/7 时各设备的空等比例（讲义 §4） | $(14-\ell_m)/14$ | 设备 2、3、4：29%、36%、50% |

### 11.3 讲义中的笔者推断（再次列出，便于审稿）

- 设备受限路由的训练实现是否与 HF 推理代码一样按“组内最大值”选设备：训练代码未公开（第 8 节第 4 条）。
- 8 路专家并行是否限制在单个节点内：论文未说明；讲义标为笔者推测。
- token 丢弃的计数粒度（按（token, 专家）份数）与被丢部分等价于把 $g_{i,t}$ 置 0：论文未写成公式，讲义标为笔者的形式化。
- “5.76 倍主要来自显存能放下更多 KV cache”：笔者推测；8 × 80 GB 容量示意忽略激活与碎片。

### 11.4 数学复核的更正（2026-09-24，逐项在 Python 里重算）

| # | 原写法 | 更正 | 依据 |
|---|---|---|---|
| M1 | V2 激活参数三种口径“四舍五入都是 21B” | 20.33B 四舍五入是 20B；只有“只算输出头”（20.85B）和“两者都算”（21.38B）四舍五入为 21B | 算术 |
| M2 | V2-Lite 的 2.4B“只有在不计输入嵌入时才对得上”（2.45B） | 精确值 2.4513B，严格四舍五入是 2.5B；三种口径 2.24B / 2.45B / 2.66B 中它离 2.4B 最近（差 0.05B），论文的 2.4B 像是向下取的数 | `cfg_v2lite.json` 逐项计数 |
| M3 | 式 (10.23) 的预热写成对 token 数线性 | 论文是“前 2K 步线性”，预热段改用步数 $n/n_w$；预热期间 batch 从 2304 增到约 2963 条（线性爬升假设），按 token 数并非直线（例如 $x_w/2$ 处是第 1063 步而非第 1000 步）。`l10_lr_batch.py` 已改为按步数插值 | C17；S11 |
| M4 | S13“多约 1.03M” | $2{,}434.86\text{K}-1{,}399.68\text{K}=1{,}035.18\text{K}\approx1.04$M（= 2,434.86K 的 42.5%） | 算术 |
| M5 | S15“$\mu^2=1.5500$” | $\mu=1.245028$，$\mu^2=1.550094\approx1.550$ | 算术 |
| M6 | 通信均衡：论文“每台设备收到约 $MT$”与讲义公平份额 $MT/D$ 看似矛盾 | 两处 $T$ 所指不同：前者（与发送上界同）是每台设备上的 token 数，后者（式 30）是一条序列的 token 数；讲义已加说明 | C10；式 (30) |
| M7 | k3 估计无偏的推导 | 补上条件：凡 $\pi_{ref}(o\mid q)>0$ 处须 $\pi_\theta(o\mid q)>0$（softmax 语言模型自动满足） | 期望的定义 |

### 11.5 第二轮数学复核（2026-09-24）

讲义中全部数值（参数账、组合数与分布、均衡损失玩具例与梯度、丢弃例、学习率/步数、MFU、YaRN 频率表与温度、KV 字节、显存容量、GRPO 优势、k3、V2-Lite 口径、Coder-V2 账）在 Python 中逐项重算，均与正文一致；15 个生成脚本重跑后 SVG 逐字节不变。修改如下：

| # | 原写法 | 更正 | 依据 |
|---|---|---|---|
| M8 | 通信均衡的求和项“均匀时等于 1” | 只有每个 token 都恰好发往 $M$ 台设备时才等于 1。设备受限路由先圈定 $M$ 台，Top-$K$ 却可能只落在其中更少的设备上；$\sum_mf''_m=\frac{D}{MT}\times$（总发送次数）$\le D$，所以设备间均匀时 $\sum f''P''\le1$ | 式 (30) 对 $m$ 求和 |
| M9 | 专家级求和项的坍缩值 $N_r/K_r$ 未说明是否为极值 | 补证明：每个 $f_i\le N_r/K_r$（一个专家每 token 至多被选一次），$\sum P_i=1$，故 $\sum f_iP_i\le N_r/K_r$，坍缩时取等 | 式 (24)(25) |
| M10 | 设备级 $f'_i$ 只写成“组内 $f$ 的平均” | 补推导：$f'_m=\frac{D}{K_rT}\times$（该设备上的（token, 专家）计算份数），即计算量除以公平份额 $K_rT/D$，与 $f''$ 的归一化平行 | 式 (24)(27)，$\lvert\mathcal E_m\rvert=N_r/D$ |
| M11 | “纯 MLA（bf16 对 bf16）降 82.2%” | 82.2% 含层数 60 对 95 的贡献；每层 MLA 本身 $1-576/2048=71.9\%$，乘层数比得 $576\times60/(2048\times95)=0.178$ | 算术 |
| M12 | 记号：设备下标在式 (26)–(31) 用 $i$、别处用 $m$；YaRN 原训练长度写 $L$；容量写 $C$ | 设备统一用 $m$（图 10.6 标签同步）；$L\to L_{ctx}$（style-guide §2、第 5 课）；在记号说明里注明 $C$ 是容量、非训练算力 | style-guide §2 |
| M13 | GRPO 式 (32) 的期望下标写成 $q,\{o_i\}\sim\pi_{\theta_{old}}$ | 改为 $q\sim P(Q)$、$\{o_i\}\sim\pi_{\theta_{old}}(\cdot\mid q)$，与论文原式及第 8 课一致 | `dsv2` 式 (32) |

### 11.6 事实与引用复核（2026-09-24，Facts 轮）

逐条核对讲义里的数字、引文与引用归属。论文用 `arxiv.py grep/text` 取 LaTeXML 全文（`dsv2` v5、`dsmoe` v1、`dsllm`）；制品与 arXiv 摘要页用 WebFetch 重新读取。

| # | 内容 | 来源 | 原文 / 值 | 讲义处理 |
|---|---|---|---|---|
| F1 | “宽度瓶颈”的完整定义：包括路由专家的中间隐藏状态 | `dsv2` v5 §3.1.2 | "we employ additional RMS Norm layers after the compressed latent vectors, and multiply additional scaling factors at the width bottlenecks (i.e., the compressed latent vectors and the intermediate hidden states of routed experts) to ensure stable training." | 第 3.1 节表“门控缩放”一行只摘了半句。讲义 §1 的 warn 改为给出括号里的完整定义；“config 的 16 就是路由专家中间隐藏状态处的因子”标为笔者推测（专家输出投影是线性的，输出乘 16 与中间隐藏状态乘 16 等价） |
| F2 | DeepSeekMoE 16B：2 共享 + 64 路由、每 token 激活 6 个、2T tokens | `dsmoe` v1 §5 | "Each MoE layer consists of 2 shared experts and 64 routed experts [...] Each token will be routed to these 2 shared experts and 6 out of 64 routed experts."；"train DeepSeekMoE 16B on a large-scale corpus with 2T tokens" | 讲义 §1 原引 `[@dsv2]`，但 `dsv2` 表 6 只给出 16.4B / 2.8B、4K、2T，没有 64/6；改引 `[@dsmoe]` |
| F3 | DeepSeek LLM 的多步调度是“取代余弦”，不是“提出” | `dsllm` §2.3 与引言 | "replaced the cosine learning rate scheduler with a multi-step learning rate scheduler, maintaining performance while facilitating continual training"；"the multi-step learning rate scheduler allows for the reuse of training from the first phase" | 讲义 §5 与参考文献卡由“提出”改为“沿用 / 取代余弦调度” |
| F4 | V2.5 的两种表述 | API 新闻页 news0905（2024-09-05）；V2.5 模型卡 | 新闻页 "a powerful combination of DeepSeek-V2-0628 and DeepSeek-Coder-V2-0724"；模型卡 "combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct"（模型卡的评测表列出 V2-0628、Coder-V2-0724、V2.5 三列） | 参考文献卡 `hf-v25` 补上新闻页网址，finding 分别写明两处说法 |
| F5 | V2-Lite 模型卡的部署说法与发布日期 | HF `DeepSeek-V2-Lite` 模型卡 | "Deployable on single 40G GPU, fine-tunable on 8x80G GPUs"；"2024.05.16: We released the DeepSeek-V2-Lite." | 与讲义一致 |
| F6 | V2-Lite 只用专家级损失的原因 | `dsv2` v5 附录 B.1 | "for each layer, all experts will be deployed on the same device. Therefore, we only employ a small expert-level balance loss with $\alpha_{1}=0.001$ [...]" | 与讲义一致 |
| F7 | 摘要中的三个头条数字 | `dsv2` v5 摘要 | "saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times." | 与讲义一致 |
| F8 | 初始化、HAI-LLM、共享专家与 all-to-all 重叠、定制 CUDA 内核、均衡损失不能保证严格均衡、推理时可自由决定是否丢弃 | `dsv2` v5 §2.2.4、§3.1.2–3.1.3 | "randomly initialized with a standard deviation of 0.006"；"we overlap the computation of shared experts with the expert parallel all-to-all communication"；"customize faster CUDA kernels for communications, routing algorithms, and fused linear computations across different experts"；"they cannot guarantee a strict load balance"；"we can flexibly decide whether to drop tokens during inference according to the efficiency requirements, and always ensure consistency between training and inference" | 与讲义一致 |
| F9 | $M\geqslant3$ 的结论没有附消融数字；NIAH 失分正文未讨论 | `dsv2` v5 §2.2.2、§3.1.4 | 全文只有 "In practice, we find that when $M\geqslant 3$ [...] roughly aligned with the unrestricted top-K routing"；正文对图 4 只说 "performs well across all context window lengths up to 128K" | 与讲义一致 |
| F10 | 参考文献元数据 | arXiv 摘要页 | 2401.10241：Penghui Qi, Xinyi Wan, Guangxing Huang, Min Lin，v1 提交于 2023-11-30；2106.05974：Carlos Riquelme 等 8 位作者，v1 提交于 2021-06-10 | 与参考文献卡一致 |
