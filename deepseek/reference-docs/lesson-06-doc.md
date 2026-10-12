# 第 06 课研究笔记：DeepSeek LLM 7B/67B（超参数 scaling law、M 表示法、数据质量）与 DeepSeek-Coder

> 研究日期 2026-09-24。本文件只保存核对过的引文、公式、规格、复算数字、图表数据与勘误，**不是课文**。
> 所有引文均为原文英文，逐字摘自所注明的版本与位置；所有派生数字都在 Python 中重算过（表达式见 §4）。
> 读图得到的数字一律标“目测”；从 SVG 矢量坐标反解的数字标“SVG 反解”（精度到绘图坐标，不是论文正文数字）。
> 临时计算脚本放在会话 scratchpad，不入库。DeepSeek-Coder-V2 属于第 10 课，本课不展开。

---

## 0. 覆盖范围与所读版本

| key | 标题 | 作者 | arXiv / 版本 | 状态 | 日期 |
|---|---|---|---|---|---|
| `dsllm` | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism | DeepSeek-AI（Xiao Bi 等 86 人，按姓氏字母序） | 2401.02954 **v1**（唯一版本；读 arXiv HTML v1 全文 + 全部 14 张图 PNG） | 预印本（技术报告），未见正式发表 | v1：2024-01-05 18:59 UTC |
| `dscoder` | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence | Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y.K. Li, Fuli Luo, Yingfei Xiong, Wenfeng Liang（13 人） | 2401.14196 **v2**（读 v2 全文；与 v1 做了逐行 diff，见 §7） | 预印本（cs.SE） | v1 2024-01-25，v2 2024-01-26 |
| `kaplan` | Scaling Laws for Neural Language Models | Kaplan et al. | 2001.08361（grep 核对 Table 1、式 2.1/2.2、0.73/0.27） | 预印本 | 2020 |
| `chinchilla` | Training Compute-Optimal Large Language Models | Hoffmann et al. | 2203.15556 v1（grep 核对三种方法的 a、b） | arXiv v1；正式发表场所本次未核实 | 2022-03-29 |
| `fim` | Efficient Training of Language Models to Fill in the Middle | Bavarian et al. | 2207.14255（grep 核对 FIM rate、PSM、推荐区间） | 预印本 | 2022 |
| `llama1` | LLaMA: Open and Efficient Foundation Language Models | Touvron et al. | 2302.13971（grep 核对 Table 2 架构） | 预印本 | 2023 |
| `llama2` | Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al. | 2307.09288（grep 核对 A.2.1 GQA 加宽 FFN） | 预印本 | 2023 |
| `gqa` | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Ainslie et al. | 2305.13245 v3（grep 定义句） | EMNLP 2023（arXiv comment: "Accepted at EMNLP 2023"） | v1 2023-05-22 |
| `dpo` | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafailov et al. | 2305.18290 v3（grep 定义句） | arXiv v3；正式发表场所本次未核实 | v1 2023-05-29 |
| `pile` | The Pile: An 800GB Dataset of Diverse Text for Language Modeling | Gao et al. | 2101.00027（grep：OpenWebText2 出处） | 预印本 | 2020 |

制品（artefact，无论文）：

| key | 内容 | 获取方式 |
|---|---|---|
| `hf-llm7b-cfg` / `hf-llm67b-cfg` | `deepseek-ai/deepseek-llm-{7b,67b}-{base,chat}` 的 `config.json`、`generation_config.json`、`tokenizer_config.json`、`tokenizer.json`、`pytorch_model.bin.index.json` | `https://huggingface.co/<repo>/raw/main/...`，2026-09-24 抓取 |
| `hf-coder-cfg` | `deepseek-ai/deepseek-coder-{1.3b,6.7b,33b}-base`、`deepseek-coder-5.7bmqa-base`、`deepseek-coder-7b-base-v1.5` 的 `config.json`；6.7b 的 `tokenizer.json`；HF API 的 `safetensors.total` | 同上 + `https://huggingface.co/api/models/<repo>` |
| `gh-llm` | `github.com/deepseek-ai/DeepSeek-LLM` 的 README、`evaluation/more_results.md`；GitHub API 的仓库创建时间与提交历史 | raw.githubusercontent.com / api.github.com |
| `gh-coder` | `github.com/deepseek-ai/DeepSeek-Coder` 的 README；提交历史 | 同上 |

**发布时间（制品证据，非论文）**：DeepSeek-LLM GitHub 仓库创建于 2023-11-29T11:07:49Z，HF `deepseek-llm-67b-base` 创建于 2023-11-29T03:23:47Z，`deepseek-llm-7b-base` 创建于 2023-11-29T03:08:22Z；论文晚了 5 周多（2024-01-05）。DeepSeek-Coder GitHub 首个提交 `init project` 为 2023-11-02T14:07:09Z（仓库 2023-10-20 创建），HF `deepseek-coder-6.7b-base` 创建于 2023-10-23（可能先以私有仓库存在）；论文 2024-01-25。`deepseek-coder-7b-base-v1.5` HF 创建于 2024-01-25T15:39:50Z，与 Coder 论文同日。

---

## 1. 核心论断与原文引文

以下均出自 `dsllm` v1，除非另注。

### 1.1 数据

1. **三阶段数据管线。** §2.1："we have organized our approach into three essential stages: deduplication, filtering, and remixing. The deduplication and remixing stages ensure a diverse representation of the data by sampling unique instances. The filtering stage enhances the density of information"
2. **跨 91 个 Common Crawl dump 去重。** §2.1："Our analysis revealed that deduplicating the entire Common Crawl corpus results in higher removal of duplicate instances compared to deduplicating within a single dump. Table 1 illustrates that deduplicating across 91 dumps eliminates four times more documents than a single dump method." Table 1 数据见 §3.4。
3. **remix 的目的。** §2.1："In the remixing phase, we adjust our approach to address data imbalances, focusing on increasing the presence of underrepresented domains."
4. **分词器。** §2.1："For our tokenizer, we implemented the Byte-level Byte-Pair Encoding (BBPE) algorithm based on the tokenizers library (Huggingface Team, 2019). Pre-tokenization was employed to prevent the merging of tokens from different character categories such as new lines, punctuation, and Chinese-Japanese-Korean (CJK) symbols, similar to GPT-2" ；"We also chose to split numbers into individual digits"
5. **词表 100000 → 100015 → 102400。** §2.1："Based on our prior experience, we set the number of conventional tokens in the vocabulary at 100000. The tokenizer was trained on a multilingual corpus of approximately 24 GB, and we augmented the final vocabulary with 15 special tokens, bringing the total size to 100015. To ensure computational efficiency during training and to reserve space for any additional special tokens that might be needed in the future, we configured the model’s vocabulary size to 102400 for training." （`tokenizer.json` 的实际构成见 §3.3 与 §7-E2。）
6. **2T tokens，中英为主。** §1："We collect 2 trillion tokens for pre-training, primarily in Chinese and English." 摘要："we have developed a dataset that currently consists of 2 trillion tokens and is continuously expanding."

### 1.2 架构

7. **微观设计沿用 LLaMA。** §2.2："The micro design of DeepSeek LLM largely follows the design of LLaMA ..., adopting a Pre-Norm structure with RMSNorm ... function and using SwiGLU ... as the activation function for the Feed-Forward Network (FFN), with an intermediate layer dimension of $\frac{8}{3}d_{model}$. It also incorporates Rotary Embedding ... for positional encoding. To optimize inference cost, the 67B model uses Grouped-Query Attention (GQA) ... instead of the traditional Multi-Head Attention (MHA)."
8. **宏观设计：层数 30 / 95，便于流水线切分。** §2.2："DeepSeek LLM 7B is a 30-layer network, while DeepSeek LLM 67B has 95 layers. These layer adjustments, while maintaining parameter consistency with other open-source models, also facilitate model pipeline partitioning to optimize training and inference."
9. **“加深而非加宽”。** §2.2："Unlike most works using Grouped-Query Attention (GQA), we expanded the 67B model’s parameters in network depth rather than the common practice of widening the intermediate width of FFN layers, aiming for better performance." 对照的“常见做法”在 `llama2` 附录 A.2.1 可找到原文："To keep a similar overall parameter count across GQA and MQA, we increase the dimension of the feed-forward layers to compensate for the reduction in the attention layers. For the MQA variant, we increase the FFN dimension by a factor of 1.33, and for the GQA variant, we increase it by a factor of 1.3."
10. **GQA 定义（`gqa` §2.2）**："Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head. ... GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number of heads, is equivalent to MHA."

### 1.3 优化与学习率调度

11. **初始化与 AdamW。** §2.3："DeepSeek LLM is initialized with a standard deviation of 0.006 and trained using the AdamW optimizer (Loshchilov and Hutter, 2017), with the following hyperparameters: $\beta_{1}=0.9$, $\beta_{2}=0.95$, and $\mathrm{weight\_decay}=0.1$."
12. **多步学习率调度（按 token 比例，不是按步数比例）。** §2.3："A multi-step learning rate scheduler is employed during pre-training instead of the typical cosine scheduler. Specifically, the learning rate of the model reaches its maximum value after 2000 warmup steps, and then decreases to 31.6% of the maximum value after processing 80% of the training tokens. It further reduces to 10% of the maximum value after 90% of the tokens. The gradient clipping during the training phase is set to 1.0." GitHub README 给出绝对位置："stepped to 31.6% of the maximum at 1.6 trillion tokens and 10% of the maximum at 1.8 trillion tokens."（`gh-llm`）
13. **多步 ≈ cosine，且能复用第一阶段。** §2.3："despite differences in the loss reduction trend during training, the final performance using a multi-step learning rate scheduler is essentially consistent with that of a cosine scheduler, as shown in Figure 1(a). When adjusting the training scale while keeping the model size fixed, the multi-step learning rate scheduler allows for the reuse of training from the first phase, offering a unique convenience for continual training."
14. **80/10/10 是折中而非最优。** §2.3："We also demonstrate in Figure 1(b) that adjusting the proportions of different stages in the multi-step learning rate scheduler can yield slightly better performance. However, for the sake of balancing reuse ratios in continual training and model performance, we opted for the aforementioned distribution of 80%, 10%, and 10% for the three stages respectively." 图 1 实验规模（图 1 标题）："The model size is 1.6 billion parameters, trained on a dataset of 100 billion tokens."
15. **训练基建要点。** §2.4："Data parallelism, tensor parallelism, sequence parallelism, and 1F1B pipeline parallelism are integrated into this framework as done in Megatron"；"ZeRO-1 ... is exploited to partition optimizer states over data parallel ranks"；"we train the model in bf16 precision but accumulate gradients in fp32 precision"；"In-place cross-entropy is performed to reduce GPU memory consumption, i.e.: we convert bf16 logits to fp32 precision on the fly in the cross-entropy CUDA kernel"；"Model weights and optimizer states are saved every 5 minutes asynchronously"。

### 1.4 超参数 scaling law（§3.1）

16. **动机：只重调 batch size 与学习率。** §3："Empirically, it has been observed that the optimal values of most parameters during training do not change when varying compute budgets. ... However, the hyperparameters that have the most significant impact on performance, namely batch size and learning rate, were re-examined."
17. **1e17 网格搜索，最优区域很宽。** §3.1："We initially conducted a grid search for batch size and learning rate on small-scale experiments with a compute budget of 1e17, and the results of a specific model size (177M FLOPs/token) are illustrated in Figure 2(a). The results demonstrate that the generalization error remains stable across a wide range of choices of batch sizes and learning rates."
18. **“近优”的定义：比最小值高不超过 0.25%。** §3.1："we utilized the aforementioned multi-step learning rate scheduler to effectively train multiple models with different batch sizes, learning rates, and compute budgets ranging from 1e17 to 2e19 by reusing the first stage. Considering the redundancy in the parameter space, we regarded the parameters used by models whose generalization error exceeded the minimum by no more than 0.25% as near-optimal hyperparameters."
19. **趋势：B 随 C 增大，η 随 C 减小。** §3.1："the optimal batch size $B$ gradually increases with the increase in compute budget $C$, while the optimal learning rate $\eta$ gradually decreases."
20. **在 1e20 上验证。** §3.1："We validated our formulae on a series of models with a 1e20 compute budget, and the results of a specific model size (2.94B FLOPs per token) are shown in Figure 2(b). The results indicate that the fitted parameters are centered in the optimal parameter space."
21. **作者自己承认的局限。** §3.1："we have not yet considered the impact of factors beyond the compute budget $C$ on the optimal hyperparameters. This is inconsistent with some earlier works (McCandlish et al., 2018; Kaplan et al., 2020) which suggested that the optimal batch size can be modeled as being solely related to the generalization error $L$. Furthermore, we observed that in models with the same compute budget but different model/data allocations, the optimal parameter space varies slightly."

### 1.5 用 M 表示模型规模（§3.2 与附录 A.2）

22. **为什么不用 6N。** §3.2："since both $6N_{1}$ and $6N_{2}$ do not account for the computational overhead of attention operation, and $6N_{2}$ also includes the vocabulary computation, which contributes less to the model’s capacity, they both have significant approximation errors under certain settings."
23. **M 的定义。** §3.2："we introduced a new model scale representation: non-embedding FLOPs/token $M$. $M$ includes the computational overhead of attention operation but does not take into account the vocabulary computation. With the model scale represented by $M$, the compute budget $C$ can be simply expressed as $C=MD$."
24. **误差在小模型上最大。** §3.2："both $6N_{1}$ and $6N_{2}$ either overestimate or underestimate the computational cost in models of different scales. This discrepancy is particularly pronounced in small-scale models, with differences reaching up to 50%. Such inaccuracies can introduce substantial statistical errors when fitting the scaling curve."
25. **附录 A.2 的结论。** "the deviation of optimal model/data allocation among these three representations is not significant at higher compute budgets, but there are noticeable differences at lower budgets." 以及 "When using $6N_{1}$ as the model scale representation, the fitted performance scaling curve tends to overestimate the performance of large-scale models. Conversely, when using $6N_{2}$, the curve tends to underestimate their performance. Using $M$ as the model scale representation, however, achieves the most accurate predictions."

### 1.6 最优模型/数据分配与 IsoFLOP（§3.2）

26. **IsoFLOP 实验规模。** §3.2："We selected 8 different compute budgets ranging from 1e17 to 3e20, and designed around 10 different model/data scale allocations for each budget. The hyperparameters for each budget were determined by Formula(1), and the generalization error was calculated on an independent validation set, distributed similarly to the training set and containing 100M tokens."
27. **指标是 bits-per-byte。** 图 4 标题："The metric in IsoFLOP curve is bits-per-byte on the validation set."
28. **1000 倍外推。** §3.2："The results indicate that using small-scale experiments can accurately predict the performance of models with 1000$\times$ compute budget." （实际外推倍数见 §4-R20：7B 约 283 倍、67B 约 2885 倍。）

### 1.7 数据质量改变最优分配（§3.3）

29. **三个数据集的质量排序。** §3.3："Our internal data assessment revealed that current in-house data has higher data quality than early in-house data. Furthermore, the quality of OpenWebText2 even surpasses the current in-house data, due to its smaller scale which allows for more meticulous processing."
30. **核心结论。** §3.3："as data quality improves, the model scaling exponent $a$ gradually increases, while the data scaling exponent $b$ decreases, which suggests that the increased compute budget should be allocated more to the model instead of the data. This finding might also explain the significant differences in optimal model/data scaling-up allocation observed in earlier studies of scaling laws."
31. **作者的解释明确标为推测。** §3.3："An intuitive speculation for this finding is that high-quality data usually implies logical clarity and less predictive difficulty after sufficient training. Therefore, it’s more advantageous to scale up the model size when increasing compute budget." §3 引言另有："The differences in the optimal model/data scaling-up allocation strategy may also serve as an indirect approach to assess the quality of data."

### 1.8 对齐（§4、§5.5）

32. **SFT 数据 150 万条。** §4："We collect around 1.5 million instruction data instances in English and Chinese, covering a wide range of helpfulness and harmlessness topics. Our helpful data contains 1.2 million instances, with a distribution of 31.2% for general language tasks, 46.6% for mathematical problems, and 22.2% for coding exercises. The safety data consists of 300K instances, covering various sensitive topics." （引言写的是 "over 1 million instances"，两处不矛盾。）
33. **SFT 轮数与学习率。** §4："We fine-tuned our 7B model with 4 epochs, but only 2 epochs for the 67B model, since we observed the overfitting problem is serious on the 67B model. ... The learning rate is 1e-5 and 5e-6 for 7B and 67B models, respectively."
34. **数学 SFT 数据导致复读。** §4："We gathered a total of 3868 Chinese and English prompts and determined the proportion of generated responses that fail to terminate and instead endlessly repeat a sequence of text. We observed that the repetition ratio tends to rise as the quantity of math SFT data increases."
35. **两阶段 SFT（仅 7B）。** §5.1.2："the stage-1 model exhibits a repetition ratio of 2.0%, which is reduced to 1.4% after stage-2 tuning, while maintaining the benchmark score. In the case of the 67B model, the repetition ratio is already below 1% following the first stage fine-tuning, and the second stage hurts the model score on the benchmark. Therefore, only one stage of SFT is done for the 67B model."
36. **DPO 设置。** §4："We trained an epoch for DPO, with a learning rate of 5e-6 and batch size of 512, and we used a learning rate warmup and cosine learning rate scheduler. We found out that DPO can strengthen the model’s open-ended generation skill, while engendering little difference in performance among standard benchmarks." 偏好数据的回答候选："we generated responses using our DeepSeek Chat models as response candidates."（论文没说偏好标签如何产生，见 §8。）
37. **DPO 的定义（`dpo`）**："Our main contribution is Direct Preference Optimization (DPO), a simple RL-free algorithm for training language models from preferences."
38. **SFT 学的是格式。** §5.1.2："we believe that the SFT stage does not learn reasoning capabilities but rather the correct format for reasoning paths."
39. **不刷选择题。** §5.5：加入 2000 万中文选择题后 MMLU/C-Eval/CMMLU 大涨但 TriviaQA 与 ChineseQA 不涨，"Therefore, we have chosen to exclude MC data from both the pre-training and fine-tuning stages, as including it would result in overfitting to benchmarks and would not contribute to achieving true intelligence in the model."
40. **预训练末期加指令数据：等价于 SFT 时加。** §5.5："we integrated 5 million instruction data, primarily consisting of multi-choice questions, during the final 10% of the pre-training stage. ... However, the final outcomes were nearly identical to those achieved by adding the same data during the SFT stage."
41. **为后续工作埋伏笔。** §6："Soon, we will release our technique reports in code intelligence and Mixture-of-Experts (MoE), respectively." 以及 "Our initial experiments prove that reinforcement learning could boost model complex reasoning capability."

### 1.9 评测要点

42. **67B 对 LLaMA-2 70B 的优势集中在代码/数学/推理/中文。** §5.1.1："DeepSeek 67B achieves considerably better performance on MATH, GSM8K, HumanEval, MBPP, BBH, and Chinese benchmarks compared to LLaMA2 70B."
43. **小模型受“语言冲突”影响更大。** §5.1.1："the advantage of DeepSeek 67B over LLaMA2 70B is larger than that of DeepSeek 7B over LLaMA2 7B. This phenomenon highlights the greater influence of language conflict on smaller models."
44. **开放式评测。** §5.2.2："our DeepSeek LLM 67B Chat outperforms other open-source models ... and achieves 8.35 score comparable with GPT-3.5-turbo. Besides, after the DPO stage, our DeepSeek LLM 67B Chat DPO further improves the average score to 8.76, which is only behind GPT-4" （注意 8.35 < GPT-3.5-turbo-0613 的 8.39，见 §7。）
45. **held-out 评测。** §5.3：LeetCode Weekly Contest 351–372 与 Bi-Weekly 108–117（2023-07 至 2023-11），"126 problems with over 20 test cases for each"；匈牙利高考 "33 problems"；IFEval "25 types of verifiable instructions and constructed around 500 prompts"，用 prompt-level loose。

### 1.10 DeepSeek-Coder（`dscoder` v2）

46. **规模与数据。** 摘要："a range of open-source code models with sizes from 1.3B to 33B, trained from scratch on 2 trillion tokens. These models are pre-trained on a high-quality project-level code corpus and employ a fill-in-the-blank task with a 16K window"；§1："trained from scratch on 2 trillion tokens sourced from 87 programming languages"。
47. **数据构成。** §2："composed of 87% source code, 10% English code-related natural language corpus, and 3% code-unrelated Chinese natural language corpus."
48. **爬取与规则过滤。** §2.1："We collect public repositories created before February 2023 on GitHub and retain only 87 programming languages ... By applying these filtering rules, we reduce the total amount of data to only 32.8% of its original size." 规则：平均行长 > 100 或最大行长 > 1000 字符的文件删除；字母字符 < 25% 删除；HTML 可见文本 ≥ 20% 且 ≥ 100 字符才保留；JSON/YAML 只保留 50–5000 字符。
49. **仓库级依赖排序。** §2.2："we first parse the dependencies between files and then arrange these files in an order that ensures the context each file relies on is placed before that file in the input sequence." 依赖只用正则抽取："such as "import" in Python, "using" in C#, and "include" in C."
50. **改造的拓扑排序可处理环。** Algorithm 1 说明："Unlike the standard approach that selects nodes with zero in-degrees, this algorithm selects nodes with minimal in-degrees, which allows it to handle cycles within the graph." 以及 "each sequence’s files are concatenated to form a single training sample. To incorporate file path information, a comment indicating the file’s path is added at the beginning of each file."
51. **仓库级去重。** §2.3："We perform deduplication at the repository level of code, rather than at the file level, as the latter approach may filter out certain files within a repository, potentially disrupting the structure of the repository."
52. **规模与去污染。** §2.4："The total data volume is 798 GB with 603 million files." 去污染："if a piece of code includes a 10-gram string identical to any in the test data, it is excluded from our training data. In cases where the test data comprises strings that are shorter than 10-grams but no less than 3-grams, we use an exact match approach for filtering."
53. **FIM 消融结论。** §3.1.2："While the model demonstrates peak performance on the HumanEval-FIM with a 100% FIM rate, this configuration also results in the weakest code completion capability. This indicates a trade-off between FIM and code completion abilities. Moreover, we observe that with a 50% PSM rate, the model outperforms the MSP strategy. To achieve a balance between FIM efficiency and code completion proficiency, we ultimately choose the 50% PSM rate as our preferred training policy." 消融设置："we employ DeepSeek-Coder-Base 1.3B as our model architecture. We focused on a Python subset from our training dataset"。
54. **FIM 实现位置。** §3.1.2："We implement the Fill-in-the-Middle (FIM) method at the document level before the packing process, as proposed in the original work by Bavarian et al. (2022). This is done with an FIM rate of 0.5, following the PSM mode."
55. **分词器。** §3.2："we utilize a tokenizer configured with a vocabulary size of 32,000."（config 是 32256，见 §7。）
56. **33B 用 GQA-8。** §3.3："the DeepSeek 33B model integrates Grouped-Query-Attention (GQA) with a group size of 8"；并用 FlashAttention v2。
57. **沿用 DeepSeek LLM 的超参数规律。** §3.4："We adapt batch sizes and learning rates by the scaling laws suggested in DeepSeek LLM. For the learning rate scheduling, we implement a three-stage policy, which includes 2000 warm-up steps, and set the final learning rate to 10% of the initial rate. Notably, the learning rate at each stage is scaled down to $\sqrt{\frac{1}{10}}$ of the preceding stage’s rate, following the guidelines established in DeepSeek LLM"
58. **长上下文：线性 RoPE 插值 ×4。** §3.6："we employed a linear scaling strategy, increasing the scaling factor from 1 to 4 and altering the base frequency from 10000 to 100000. The model underwent an additional 1000 steps of training, using a batch size of 512 and a sequence length of 16K. ... Theoretically, these modifications enable our model to process up to 64K tokens in context. However, empirical observations suggest that the model delivers its most reliable outputs within a 16K token range."
59. **指令微调。** §3.7："we employed a unique delimiter token <|EOT|> ... we use a cosine schedule with 100 warm-up steps and an initial learning rate 1e-5. We also use a batch size of 4M tokens and 2B tokens in total."
60. **仓库级预训练的效果（消融）。** §4.3："When only utilizing file-level code corpus (w/o Repo Pre-training) to pre-train DeepSeek-Coder, we observe a decrease in performance in the Java, TypeScript, and C# languages, indicating the effectiveness of the repository-level pre-training." （注意 Python 列 EM 反而 16.02→16.14 几乎不变、ES 66.65→66.51 略降，所以只说 Java/TS/C#。）
61. **v1.5：从通用 LLM 续训。** §5："we perform additional pre-training from the general language model DeepSeek-LLM-7B Base ... on 2 trillion tokens, resulting in DeepSeek-Coder-v1.5 7B. ... Unlike DeepSeek-Coder, DeepSeek-Coder-v1.5 employs solely a next token prediction objective with a 4K context length during its pre-training phase." 结论 §6："the most effective code-focused Large Language Models (LLMs) are those built upon robust general LLMs."

### 1.11 旁证文献

62. **Kaplan 的 6N 省略了上下文项（`kaplan` §2.1）**："Since we primarily study models where $d_{\rm model}\gg n_{\rm ctx}/12$, we do not include context-dependent terms in our training compute estimate. Accounting for the backwards pass (approximately twice the compute as the forwards pass), we then define the estimated non-embedding compute as $C\approx 6N$ floating point operators per training token."
63. **Kaplan 的数据集是 WebText2**（`kaplan` §2.2）："We train language models on WebText2, an extended version of the WebText [RWC+19] dataset"。OpenWebText2 是 The Pile 新引入的（`pile` §1）："We also introduce OpenWebText2 and BookCorpus2, which are extensions of the original OpenWebText ... and BookCorpus ... datasets, respectively."
64. **Kaplan 的分配指数**："which closely matches the empirically optimal results $N\propto C_{\rm min}^{0.73}$, $B\propto C_{\rm min}^{0.24}$, and $S\propto C_{\rm min}^{0.03}$"；以及 "data requirements growing very slowly as $D\sim C^{0.27}$"。
65. **Chinchilla 的 0.49/0.51 是方法 2（IsoFLOP）**（`chinchilla` §3.2）："we find that $a=0.49$ and $b=0.51$—as summarized in Table 2." 方法 1 为 0.50/0.50，方法 3 为 0.46/0.54（Table 2）。
66. **FIM-for-free 与推荐 FIM rate（`fim`）**："models trained jointly on a mixture of FIM transformed data and ordinary left-to-right data achieve the same left-to-right capability while learning how to fill-in-the-middle. We call this the FIM-for-free property." §8.1："In practice, any value in the range between 50% and 90% is a reasonable choice." 文档级 FIM 的定义（§3）："with a certain probability $p$ called the FIM rate (we use $p=0.5$ for our main suite of models), we cut each document into three parts: prefix, middle, and suffix. We perform this split prior to tokenization"。

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 原文转录（`dsllm` v1）

算力近似（§3，无编号）：
$$C=6ND \qquad\text{与}\qquad C=MD$$

**式 (1)** 超参数 scaling law（HTML 中 (1) 标在这一组两行上）：
$$\eta_{\mathrm{opt}}=0.3118\cdot C^{\,-0.1250},\qquad B_{\mathrm{opt}}=0.2920\cdot C^{\,0.3271}\tag{1}$$

**式 (2)** 三种模型规模表示（同一编号下三行）：
$$6N_{1}=72\,n_{\mathrm{layer}}\,d_{\mathrm{model}}^{2},\qquad 6N_{2}=72\,n_{\mathrm{layer}}\,d_{\mathrm{model}}^{2}+6\,n_{\mathrm{vocab}}\,d_{\mathrm{model}},\qquad M=72\,n_{\mathrm{layer}}\,d_{\mathrm{model}}^{2}+12\,n_{\mathrm{layer}}\,d_{\mathrm{model}}\,l_{\mathrm{seq}}\tag{2}$$
原文符号说明："where $n_{\mathrm{layer}}$ represents the number of layers, $d_{\mathrm{model}}$ represents the model width, $n_{\mathrm{vocab}}$ is the vocabulary size, and $l_{\mathrm{seq}}$ is the sequence length."

**式 (3)** 优化目标（照印刷原样，注意右边写的是 $L(N,D)$）：
$$M_{\mathrm{opt}}(C),D_{\mathrm{opt}}(C)=\underset{M,D\,\mathrm{s.t.}\,C=MD}{\mathrm{argmin}}L(N,D)\tag{3}$$

**式 (4)** 最优分配：
$$M_{\mathrm{opt}}=M_{\mathrm{base}}\cdot C^{a},\ M_{\mathrm{base}}=0.1715,\ a=0.5243;\qquad D_{\mathrm{opt}}=D_{\mathrm{base}}\cdot C^{b},\ D_{\mathrm{base}}=5.8316,\ b=0.4757\tag{4}$$

§3.2 的定义句（无编号）："finding model scaling exponent $a$ and data scaling exponent $b$ that satisfy $N_{\mathrm{opt}}\propto C^{a}$ and $D_{\mathrm{opt}}\propto C^{b}$"。

单位：$C$ 为 FLOPs；$B$ 为 **token 数**（图 3 纵轴 "Optimal Batch Size (Tokens)"）；$\eta$ 为峰值学习率；$M$ 为 FLOPs/token；$D$ 为 token 数。

### 2.2 `dscoder` v2 的 PSM 模板（§3.1.2，无编号）

$$\texttt{<｜fim\_start｜>}f_{pre}\texttt{<｜fim\_hole｜>}f_{suf}\texttt{<｜fim\_end｜>}f_{middle}\texttt{<|eos\_token|>}$$

v1 印成 `<｜fim_hole｜>f_pre<｜fim_hole｜>f_suf...`（两个 hole），v2 改正为 `fim_start`。**发布的 tokenizer 中真实 token 名是 `<｜fim▁begin｜>`（id 32016）、`<｜fim▁hole｜>`（32015）、`<｜fim▁end｜>`（32017）**，README 的插入示例也用 `fim▁begin/hole/end`。写课文时的推理格式应以 tokenizer 为准：`<｜fim▁begin｜>前缀<｜fim▁hole｜>后缀<｜fim▁end｜>` → 模型生成中段。

`fim`（Bavarian）原文的 PSM 编码（§3）：$\textsc{<pre>}\circ\text{Enc}(\text{prefix})\circ\textsc{<suf>}\circ\text{Enc}(\text{suffix})\circ\textsc{<mid>}\circ\text{Enc}(\text{middle})$。

### 2.3 旁证公式（`kaplan`）

式 (2.1)：$N\approx 2d_{\rm model}n_{\rm layer}(2d_{\rm attn}+d_{\rm ff})=12n_{\rm layer}d_{\rm model}^{2}$（"with the standard $d_{\rm attn}=d_{\rm ff}/4=d_{\rm model}$"）。
式 (2.2)：$C_{\rm forward}\approx 2N+2n_{\rm layer}n_{\rm ctx}d_{\rm model}$。
Kaplan Table 1 的注意力项只有一行 "Attention: Mask" $2n_{\rm layer}n_{\rm ctx}d_{\rm attn}$。

### 2.4 课程记号对照（style-guide §2）

| 论文 | 课程 | 说明 |
|---|---|---|
| $n_{\mathrm{layer}}$ / $n_{\mathrm{layers}}$ | $L$（或 $l$） | **陷阱**：论文的 $l_{\mathrm{seq}}$ 是序列长度，不是层数；课程里 $M$ 式建议写成 $L$ 层、$T$ 长度 |
| $d_{\mathrm{model}}$ | $d$ | |
| $n_{\mathrm{heads}}$ | $n_h$ | |
| $n_{\mathrm{kv\_heads}}$ | $n_g$ | GQA 组数；67B 为 8 |
| $l_{\mathrm{seq}}$ | $T$ | 预训练时 $T=4096$ |
| $n_{\mathrm{vocab}}$ | $V$ | 102400 |
| $N_1$, $N_2$ | $N_1$（非嵌入参数）, $N_2$（“完整”参数） | $N_2$ 只加了一份 $Vd$（见 §7-E5） |
| $M$ | $M$ | 每 token 非嵌入 FLOPs（训练，前向+反向） |
| $B$, $\eta$ | $B$（按 token）, $\eta$ | |
| $a$, $b$ | $a$, $b$ | |

课程版本：
$$6N_1=72\,L\,d^{2},\qquad M=72\,L\,d^{2}+12\,L\,d\,T,\qquad \frac{6N_1}{M}=\frac{6d}{6d+T}$$

### 2.5 推导要点（供课文逐行展开；每一步都已在 §4 用数字核对）

**(a) 72 从哪来。** 每层参数（MHA，$d_{ff}=\tfrac83 d$）：注意力 $W^Q,W^K,W^V,W^O$ 共 $4d^2$；SwiGLU 三个矩阵 $3\cdot d\cdot\tfrac83 d=8d^2$；合计 $12d^2$ → $N_1=12Ld^2$。训练每参数每 token 约 6 FLOPs（前向 2、反向 4）→ $6N_1=72Ld^2$。

**(b) 12 L d T 从哪来。** 每层每 token 前向：$\mathbf q_t^\top K$ 与 $T$ 个键点积 $2Td$ FLOPs；注意力权重乘 $V$ 再 $2Td$；合计 $4Td$。乘 3（前向+反向）→ $12Td$；乘 $L$ 层 → $12LdT$。**注意：这里每个 token 都按与全部 $T$ 个位置计算，没有因果掩码的“平均一半”折扣**；Kaplan 的训练算力若保留上下文项则为 $3\times 2Ln_{ctx}d=6Ln_{ctx}d$，只有 DeepSeek 的一半（见 §7-E4）。

**(c) 注意力占比。** $\dfrac{12LdT}{72Ld^2+12LdT}=\dfrac{T}{6d+T}$，与 mamba 第 1 课研究笔记中的 $L/(L+6d)$ 同式。

**(d) 一致性检验。** 由 $C=MD$ 与式 (4)：$M_{\mathrm{opt}}D_{\mathrm{opt}}=M_{\mathrm{base}}D_{\mathrm{base}}C^{a+b}$，必须恒等于 $C$，所以应有 $a+b=1$、$M_{\mathrm{base}}D_{\mathrm{base}}=1$。实测：$0.5243+0.4757=1.0000$，$0.1715\times5.8316=1.00012$。

**(e) “数据越好越该加大模型”的读法。** 算力扩大 10 倍：模型扩大 $10^a$ 倍，数据扩大 $10^b$ 倍。早期数据 $a=0.450$ → 模型 ×2.82、数据 ×3.55；当前数据 $a=0.524$ → ×3.34 / ×2.99；OpenWebText2 $a=0.578$ → ×3.78 / ×2.64（§4-R14）。

**(f) 多步调度的数学形式（课文用）。** 设总 token 数 $D$，已处理 $\tau$ 个 token：
$$\eta(\tau)=\eta_{\max}\times\begin{cases}\text{warmup（2000 步，形状论文未写）}\\ 1, & \tau<0.8D\\ 10^{-1/2}\approx0.316, & 0.8D\le\tau<0.9D\\ 10^{-1}=0.1, & \tau\ge0.9D\end{cases}$$
$31.6\%=\sqrt{0.1}$，即两次各乘 $\sqrt{1/10}$；DeepSeek-Coder §3.4 明文写了 "scaled down to $\sqrt{\frac{1}{10}}$ of the preceding stage’s rate"。

---

## 3. 规格与超参数表

### 3.1 DeepSeek LLM 7B / 67B

| 项 | 7B | 67B | 来源 |
|---|---|---|---|
| 层数 $L$ | 30 | 95 | `dsllm` Table 2；config `num_hidden_layers` 同 |
| $d$ | 4096 | 8192 | Table 2；config `hidden_size` 同 |
| $n_h$ | 32 | 64 | Table 2；config 同 |
| $n_{kv}$ | 32（MHA） | 8（GQA-8） | Table 2；config `num_key_value_heads` 同 |
| $d_h$ | 128 | 128 | 由 $d/n_h$ 计算 |
| FFN 中间维 | 11008 | 22016 | **只在 config**（`intermediate_size`）；论文只写 $\tfrac83 d$（=10922.7 / 21845.3），config 是向上取整后的值，两者比值都是 $43/16=2.6875$ |
| 上下文 | 4096 | 4096 | Table 2；config `max_position_embeddings` 4096 |
| batch size | 2304（序列） | 4608（序列） | Table 2（表头 "Sequence Batch Size"）；README 同。= 9,437,184 / 18,874,368 token |
| 峰值学习率 | 4.2e-4 | 3.2e-4 | Table 2；README 同 |
| 训练 token | 2.0T | 2.0T | Table 2 |
| 词表（嵌入行数） | 102400 | 102400 | §2.1；config `vocab_size` |
| RoPE $\theta$ | 10000 | 10000 | config `rope_theta`（论文未写） |
| RMSNorm eps | 1e-6 | 1e-6 | config（论文未写） |
| 嵌入共享 | 否 | 否 | config `tie_word_embeddings: false` |
| 初始化 std | 0.006 | 0.006 | §2.3；**config `initializer_range` 为 0.02**（HF 默认字段，仅影响重新初始化，见 §7） |
| AdamW | β1 0.9，β2 0.95，wd 0.1 | 同 | §2.3 |
| 调度 | 2000 步 warmup；80% token 处 ×0.316；90% 处 ×0.1 | 同 | §2.3；README：1.6T / 1.8T |
| 梯度裁剪 | 1.0 | 1.0 | §2.3 |
| 精度 | bf16 训练，fp32 累积梯度 | 同 | §2.4 |
| 精确参数量 | 6,910,365,696 | 67,425,001,472 | 按 config 计算，与 `pytorch_model.bin.index.json` 的 `total_size`/2（13,820,731,392 / 134,850,002,944 字节，bf16）逐位一致 |
| 非嵌入参数（含 norm） | 6,071,504,896 | 65,747,279,872 | 计算 |
| 嵌入+输出头 | 838,860,800 | 1,677,721,600 | 计算：$2Vd$ |
| bos/eos id | config.json 写 1/2；generation_config 与 tokenizer 为 100000/100001 | 同 | 制品不一致（§7-E15）；chat 版 config 为 100000/100001 |

### 3.2 对齐超参数

| 项 | 7B Chat | 67B Chat | 来源 |
|---|---|---|---|
| SFT 数据 | 150 万（有用 120 万：通用 31.2% / 数学 46.6% / 代码 22.2%；安全 30 万） | 同 | §4 |
| SFT epoch | 4 | 2 | §4 |
| SFT 学习率 | 1e-5 | 5e-6 | §4 |
| SFT 阶段 | 两阶段（第二阶段去掉数学和代码数据） | 一阶段 | §5.1.2、§5.5 |
| 复读率 | 阶段 1：2.0%（0.020）→ 阶段 2：1.4%（0.014） | 阶段 1 后已 < 1% | §5.1.2、Table 12 |
| DPO | 1 epoch，lr 5e-6，batch 512，warmup + cosine | 同 | §4 |

### 3.3 分词器（`tokenizer.json` 实测）

| 项 | DeepSeek LLM | DeepSeek-Coder |
|---|---|---|
| 模型类型 | BPE（byte-level） | BPE（byte-level） |
| BPE 词表条目 | 100,000（id 0–99999），merges 99,757 | 32,000（id 0–31999），merges 31,757 |
| added tokens | 15 个（id 100000–100014）：`<｜begin▁of▁sentence｜>`、`<｜end▁of▁sentence｜>` 为 special；其余 13 个是单个 Latin-1 字符（ø ö ú ÿ õ ÷ û ý À ù Á þ ü），`special: false` | 22 个（id 32000–32021）：同样 13 个单字符 + BOS/EOS（32013/32014）+ `<｜fim▁hole｜>`32015、`<｜fim▁begin｜>`32016、`<｜fim▁end｜>`32017、`<pad>`、`<\|User\|>`、`<\|Assistant\|>`、`<\|EOT\|>` |
| 已用 id 总数 | 100,015 | 32,022 |
| 嵌入行数（config） | 102,400（预留 2,385 行） | 32,256（预留 234 行） |
| 预分词 | 换行单独切；字母类 Unicode 区段；标点；CJK 区段 `[一-龥ࠀ-一가-퟿]+`；`Digits(individual_digits=true)`；ByteLevel | 类似，字母用 `\p{L}`、标点用 `\p{P}`；同样逐位切数字 |
| 论文写法 | 100000 + 15 → 100015，训练用 102400 | "vocabulary size of 32,000" |

### 3.4 数据：Table 1（`dsllm`，去重率随 dump 数）

| 使用的 dump 数 | 1 | 2 | 6 | 12 | 16 | 22 | 41 | 91 |
|---|---|---|---|---|---|---|---|---|
| 去重率 (%) | 22.2 | 46.7 | 55.7 | 69.9 | 75.7 | 76.3 | 81.6 | 89.8 |

GitHub README 补充（制品）："Our advanced deduplication system, using MinhashLSH, strictly removes duplicates both at document and string levels."；"Our minimal viable solution departs from RefinedWeb + CCNet."

### 3.5 scaling law 系数（`dsllm` Table 4）

| 方法 | $a$（$N_{opt}$ 或 $M_{opt}\propto C^a$） | $b$（$D_{opt}\propto C^b$） | 备注 |
|---|---|---|---|
| OpenAI (OpenWebText2) | 0.73 | 0.27 | 表中标签原样；Kaplan 实际数据是 WebText2（§7-E10） |
| Chinchilla (MassiveText) | 0.49 | 0.51 | = Chinchilla 方法 2（IsoFLOP）；方法 1 为 0.50/0.50，方法 3 为 0.46/0.54 |
| Ours (Early Data) | 0.450 | 0.550 | |
| Ours (Current Data) | 0.524 | 0.476 | 与式 (4) 的 0.5243/0.4757 一致 |
| Ours (OpenWebText2) | 0.578 | 0.422 | |

### 3.6 Table 3（`dsllm`，三种规模表示，$V=102400$、$T=4096$ 对所有行成立）

| $L$ | $d$ | $N_1$ | $N_2$ | $M$ | $6N_1/M$ | $6N_2/M$ |
|---|---|---|---|---|---|---|
| 8 | 512 | 25.2M | 77.6M | 352M | 0.43 | 1.32 |
| 12 | 768 | 84.9M | 164M | 963M | 0.53 | 1.02 |
| 24 | 1024 | 302M | 407M | 3.02B | 0.60 | 0.81 |
| 24 | 2048 | 1.21B | 1.42B | 9.66B | 0.75 | 0.88 |
| 32 | 4096 | 6.44B | 6.86B | 45.1B | 0.85 | 0.91 |
| 40 | 5120 | 12.6B | 13.1B | 85.6B | 0.88 | 0.92 |
| 80 | 8192 | 64.4B | 65.3B | 419B | 0.92 | 0.94 |

全部可由式 (2) 复现（§4-R8），唯一出入：第 5 行 $6N_1/M$ 算得 0.857，印作 0.85。

### 3.7 DeepSeek-Coder（`dscoder` Table 2 vs config）

| 项 | 1.3B | 6.7B | 33B | 来源 |
|---|---|---|---|---|
| $d$ | 2048 | 4096 | 7168 | Table 2 = config |
| FFN 中间维 | 5504 | 11008 | 19200 | Table 2 = config |
| 层数 | 24 | 32 | 62 | Table 2 = config |
| 头数 | 16 | 32 | 56 | Table 2 = config |
| 注意力 | MHA | MHA | GQA（8）；config `num_key_value_heads: 8` | Table 2 = config |
| batch size | 1024 | 2304 | 3840 | Table 2（单位未写，按 4096 序列 = 4.19M / 9.44M / 15.73M token，见 §4-R24） |
| 峰值学习率 | 5.3e-4 | 4.2e-4 | 3.5e-4 | Table 2 |
| 词表 | 32,000（论文）/ 32,256（config） | 同 | 同 | §3.2 vs config |
| 上下文 | 16384 | 16384 | 16384 | config `max_position_embeddings`；论文 "16K" |
| RoPE | linear scaling factor 4，$\theta=100000$ | 同 | 同 | §3.6 = config |
| 精确参数量 | 1,346,471,936（计算） | 6,740,512,768 | 33,342,991,360 | 计算；6.7B、33B 与 HF API `safetensors.total` 逐位一致 |
| 训练数据 | 2T token；87% 代码 / 10% 英文代码相关 / 3% 中文 | 同 | 同 | §2 |
| FIM | 文档级，PSM，rate 0.5 | 同 | 同 | §3.1.2、§4.2 |
| 优化器 | AdamW β1 0.9 β2 0.95；2000 步 warmup；三阶段，每阶段 ×$\sqrt{1/10}$，终值 10% | 同 | 同 | §3.4 |
| 指令微调 | cosine，100 步 warmup，lr 1e-5，batch 4M token，共 2B token | 同 | 同 | §3.7 |

其它变体（制品）：`deepseek-coder-5.7bmqa-base` config：32 层、$d$ 4096、32 头、`num_key_value_heads: 1`（MQA）、FFN 11008，README 列为 "1B, 5.7B, 6.7B and 33B"，**论文未提**。`deepseek-coder-7b-base-v1.5` config：30 层、$d$ 4096、词表 102400、4096 上下文——即 DeepSeek LLM 7B 的架构，参数 6,910,365,696（HF API），论文 Table 10 写 6.9B。

### 3.8 Coder 训练数据语言分布（`dscoder` Table 1 前十，按 GB）

| 语言 | GB | 文件（千） | 占比 % |
|---|---|---|---|
| Java | 148.66 | 134,367 | 18.63 |
| Python | 120.68 | 75,188 | 15.12 |
| C++ | 90.87 | 36,006 | 11.39 |
| TypeScript | 60.62 | 62,432 | 7.60 |
| PHP | 58.92 | 40,627 | 7.38 |
| C# | 58.56 | 53,739 | 7.34 |
| JavaScript | 53.84 | 71,895 | 6.75 |
| HTML | 30.05 | 14,998 | 3.77 |
| C | 28.64 | 27,111 | 3.59 |
| Tex | 20.46 | 2,867 | 2.56 |
| 合计（印刷） | 797.92 | 603,173 | 100.00 |

87 行逐项相加：797.88 GB、占比 99.93%（四舍五入误差）。

### 3.9 评测要点表（`dsllm` Table 5/6/9/8，均为论文内部框架）

| 基准 | LLaMA2 7B | DS 7B Base | LLaMA2 70B | DS 67B Base | DS 7B Chat | DS 67B Chat |
|---|---|---|---|---|---|---|
| MMLU（5-shot；chat 0-shot） | 45.8 | 48.2 | 69.0 | 71.3 | 49.4 | 71.1 |
| GSM8K（8-shot；chat 0-shot） | 15.5 | 17.4 | 58.4 | 63.4 | 63.0 | 84.1 |
| MATH（4-shot；chat 0-shot） | 2.5 | 6.0 | 13.5 | 18.7 | 15.8 | 32.6 |
| HumanEval（0-shot） | 14.6 | 26.2 | 28.7 | 42.7 | 48.2 | 73.8 |
| BBH（3-shot） | 38.5 | 39.5 | 62.9 | 68.7 | 42.3 | 71.7 |
| C-Eval（5-shot；chat 0-shot） | 33.9 | 45.0 | 51.4 | 66.1 | 47.0 | 65.2 |
| CHID（0-shot） | 37.9 | 89.3 | 55.5 | 92.1 | 64.9 | 72.6 |
| TriviaQA（5-shot） | 63.8 | 59.7 | 79.5 | 78.9 | 57.9 | 81.5 |
| Pile-test BPB（越低越好） | 0.741 | 0.725 | 0.649 | 0.642 | – | – |

开放式：MT-Bench（Table 8）67B Chat 8.35、67B Chat DPO 8.76、GPT-3.5-turbo-0613 8.39、GPT-4-1106-preview 9.26；AlignBench（Table 7，gpt-4-0613 打分）67B-Chat 6.43、67B-Chat-DPO 6.69、gpt-3.5-turbo-0613 6.08。held-out（Table 9）67B Chat：LeetCode 17.5、匈牙利 58、IFEval 55.5；7B Chat：4.7 / 28.5 / 41.2；GPT-4：48.4 / 68 / 79.3。安全（Table 10）：2400 题，各类安全数 486/500、473/500、281/300、290/300、767/800；Do-Not-Answer（Table 11）67B Chat 97.8。

---

## 4. 复算的数字

全部在 Python 3.11 中计算。记号：$D=2\times10^{12}$，$T=4096$。

| # | 量 | 表达式 | 值 |
|---|---|---|---|
| R1 | 7B 的 $M$（论文式 2） | `72*30*4096**2 + 12*30*4096*4096` | 4.2279e10 |
| R2 | 7B 的 $M$（按真实结构：MHA，FFN 11008） | `6*30*(2*d*d+2*d*d+3*d*11008) + 12*30*d*4096`，d=4096 | 4.2467e10 |
| R3 | 67B 的 $M$（论文式 2，假设 MHA 与 $\tfrac83d$） | `72*95*8192**2 + 12*95*8192*4096` | 4.9728e11 |
| R4 | 67B 的 $M$（按真实结构：GQA-8，FFN 22016） | `6*95*(2*d*d + 2*d*8*128 + 3*d*22016) + 12*95*d*4096` | **4.3273e11**（图 4(b) 标注 "4.3e11 FLOPs/Token"，吻合） |
| R5 | 训练算力 $C=MD$ | 7B：`4.2467e10*2e12`；67B：`4.3273e11*2e12` | 8.4935e22；8.6545e23 |
| R6 | 式 (1) 对 7B 的预测 | `0.3118*8.4935e22**-0.125`；`0.2920*8.4935e22**0.3271` | $\eta$=4.244e-4（论文 4.2e-4）；$B$=9.24M token = 2255 条 4096 序列（图 3 标 9.2M；实用 2304 条 = 9.44M） |
| R7 | 式 (1) 对 67B 的预测（用 R4） | 同上，C=8.6545e23 | $\eta$=3.175e-4（论文 3.2e-4）；$B$=19.74M token = 4818 条（图 3 标 **19.7M**；实用 4608 条 = 18.87M） |
| R7' | 若误用 R3 的 $M$ | C=9.9455e23 | $\eta$=3.12e-4；$B$=20.65M——与图 3 标注 19.7M 不符，说明论文用的是真实结构的 $M$ |
| R8 | Table 3 七行全部复现 | $N_1=12Ld^2$，$N_2=N_1+Vd$，$M$ 式 (2) | 25.17M/77.60M/352M/0.429/1.321；84.93M/163.6M/963M/0.529/1.020；302.0M/406.8M/3.020B/0.600/0.808；1.208B/1.418B/9.664B/0.750/0.880；6.443B/6.862B/45.10B/**0.857**/0.913；12.58B/13.11B/85.56B/0.882/0.919；64.42B/65.26B/418.8B/0.923/0.935 |
| R9 | $6N_1/M$ 的闭式 | $6d/(6d+T)$ | 7B：0.857；67B（d=8192）：0.923 |
| R10 | 注意力 FLOPs 占比 $T/(6d+T)$ | `4096/(6*4096+4096)`；`4096/(6*8192+4096)` | 7B 14.3%；67B 7.7%；Coder 6.7B 在 16K：`16384/(24576+16384)` = 40.0% |
| R11 | 一致性：$a+b$、$M_{base}D_{base}$ | `0.5243+0.4757`；`0.1715*5.8316` | 1.0000；1.00012 |
| R12 | 67B 的算力最优点（反解式 4） | $C=(M/0.1715)^{1/0.5243}$，M=4.3273e11 | $C_{opt}$=4.51e23（图 4 标 "4.5e23"）；$D_{opt}$=1.043e12（图 4(c) 标 "1.04e12 Tokens"）；实际 2T = **1.92×** $D_{opt}$ |
| R13 | 7B 的算力最优点 | 同上，M=4.2467e10 | $C_{opt}$=5.39e21，$D_{opt}$=1.27e11；实际 2T = **15.8×** |
| R14 | 算力 ×10 时模型/数据倍数 | `10**a`, `10**b` | Kaplan ×5.37/×1.86；Chinchilla ×3.09/×3.24；早期 ×2.82/×3.55；当前 ×3.34/×2.99；OWT2 ×3.78/×2.64 |
| R15 | 式 (4) 在各 $C$ 的 $D/(M/6)$（“每个 6N 等效参数的 token 数”） | `6*Dopt(C)/Mopt(C)` | 1e17：30.4；1e19：24.3；1e21：19.5；1e23：15.6；1e24：13.9；1e25：12.4（随 C 缓慢下降，因 $b<a$；Chinchilla 为常数约 20，属第 2 课） |
| R16 | 实际 tokens/参数 | `2e12/6.910e9`；`2e12/67.425e9` | 7B 289；67B 29.7 |
| R17 | IsoFLOP 各预算的 $M_{opt}$（式 4） | `0.1715*C**0.5243` | 1e17 1.40e8；3e17 2.50e8；1e18 4.70e8；3e18 8.35e8；1e19 1.57e9；3e19 2.79e9；1e20 5.25e9；3e20 9.34e9 |
| R18 | 同上 $D_{opt}$ | `5.8316*C**0.4757` | 7.12e8；1.20e9；2.13e9；3.59e9；6.37e9；1.07e10；1.91e10；3.21e10 |
| R19 | 式 (1) 在 1e17 与 1e20 | log2 值 | 1e17：$\eta$=2.338e-3（$2^{-8.74}$），$B$=1.062e5（$2^{16.70}$）；1e20：$\eta$=9.86e-4（$2^{-9.99}$），$B$=1.017e6（$2^{19.96}$）——正是图 2(b) 星号所在格 $(2^{20},2^{-10})$ |
| R20 | 外推倍数 | `8.4935e22/3e20`；`8.6545e23/3e20`；`8.6545e23/2e19` | 7B 283×；67B 2885×（论文说 "1000×"）；超参数拟合（按正文最大 2e19）到 67B：43,275×（若按图 3(a) 灰点最右约 1.6e20 算，`8.6545e23/1.6e20` ≈ 5,400×，见 §8-23） |
| R21 | 图 2 的训练 token 数 | `1e17/177e6`；`1e20/2.94e9` | 5.65e8；3.40e10 |
| R22 | 训练步数与 warmup | `2e12/(2304*4096)`；`2000*2304*4096` | 7B：211,928 步，warmup 18.9B token，降档步约 169,542 与 190,735；67B：105,964 步，warmup 37.7B token，降档约 84,771 与 95,367 |
| R23 | 学习率三档 | `lr*[1, 10**-0.5, 0.1]` | 7B：4.2e-4 / 1.328e-4 / 4.2e-5；67B：3.2e-4 / 1.012e-4 / 3.2e-5 |
| R24 | Coder 用式 (1) 的预测（C 按 2T token、$T$=4096、真实结构 M） | 同 R6 | 1.3B：M=9.70e9，$\eta$ 5.10e-4（论文 5.3e-4），$B$ 1391 条（论文 1024）；6.7B：M=4.53e10，$\eta$ 4.21e-4（论文 4.2e-4），$B$ 2303 条（论文 2304）；33B：M=2.19e11，$\eta$ 3.46e-4（论文 3.5e-4），$B$ 3857 条（论文 3840） |
| R25 | KV cache bf16 每 token $2\,L\,n_{kv}\,d_h\times2$ 字节 | | 7B：491,520 B = 480 KiB；67B（GQA-8）：389,120 B = 380 KiB；67B 若为 MHA：3,112,960 B ≈ 2.97 MiB（×8）；Coder 1.3B 192 KiB；6.7B 512 KiB；33B（GQA-8）248 KiB；5.7B-MQA 16 KiB |
| R26 | 单序列满上下文 KV | ×4096 或 ×16384 | 7B@4K 1.875 GiB；67B@4K 1.484 GiB（MHA 版 11.875 GiB）；Coder 6.7B@16K 8.0 GiB；33B@16K 3.875 GiB；1.3B@16K 3.0 GiB |
| R27 | 嵌入参数占比 | `2*102400*4096/6.910e9`；`2*102400*8192/67.425e9` | 7B 12.1%；67B 2.5% |
| R28 | 预留词表行 | `102400-100015`；`32256-32022` | 2,385；234 |
| R29 | 去重倍数 | `89.8/22.2`；剩余 `100-89.8`、`100-22.2` | 4.05×；91 dump 剩 10.2%，单 dump 剩 77.8% |
| R30 | SFT 有用数据拆分 | `1.2e6*[0.312,0.466,0.222]` | 374,400 / 559,200 / 266,400 |
| R31 | 安全测试通过率 | `(486+473+281+290+767)/2400` | 2297/2400 = 95.7% |
| R32 | MT-Bench 差值 | `8.35-8.39`；`8.76-8.39` | −0.04（未 DPO 略低于 GPT-3.5）；+0.37 |
| R33 | Coder 长上下文续训 token | `1000*512*16384` | 8.39e9（≈8.4B；README 说 16K 阶段 200B，矛盾见 §7） |
| R34 | Coder RoPE 理论长度 | `16384*4` | 65,536（论文 "up to 64K"） |
| R35 | Coder HumanEval 提升 | `50.3-41.0`；`66.0-55.2` | 9.3 / 10.8 个百分点（论文 "9% and 11%" 为百分点） |
| R36 | Coder LeetCode 33B Instruct 解出题数 | `0.578*45+0.22*91+0.091*44` | 26+20+4 = 50 = 27.8%×180 |
| R37 | 多步调度复用节省（例题） | 预算 D、2D、4D：cosine 7D；多步 `3.2D+0.2D+0.4D+0.8D` | 4.6D，节省 34.3% |
| R38 | Coder Table 6 "Mean" 对简单平均 | `(a+b+c)/3` | SantaCoder 60.0（印 69.0）；CodeLlama-7B 74.0（印 69.7）；DS 7B 78.1（印 80.7）；DS 33B 78.2（印 81.2）；其余三行吻合 |
| R39 | 图 5 目测点的饱和幂律拟合（笔者拟合，非论文） | 网格搜索 $E$，$L=E+A C^{-\alpha}$ | $L\approx0.383+30.1\,C^{-0.0870}$；1e16→1.60，7B→0.687，67B→0.631，1e25→0.584 |

---

## 5. 图表清单与重绘建议

### 5.1 论文原图清单

`dsllm` v1（HTML 中 14 张 PNG，文件名在括号内）：

| 图 | 标题（原文） | 内容 | 是否重绘 |
|---|---|---|---|
| 1(a) | Multi-step v.s. cosine learning rate decay（loss_step_cosine） | 1.6B/100B token 训练损失；多步曲线在 80B 前高于 cosine、80B 处陡降，100B 时两者都约 2.33（目测） | 是（改成学习率曲线 + 示意损失） |
| 1(b) | Different proportions of multi-step stages（loss_diff_step） | 80/10/10、70/15/15、60/20/20 三条，终点几乎重合（目测约 2.33） | 可选 |
| 2(a) | 1e17 FLOPs (177M FLOPs/token)（loss_bs_lr_1e17） | batch $2^{15}$–$2^{20}$ × lr $2^{-10.5}$–$2^{-7.75}$ 的训练损失热力图，格内印有数值 | 是（数据见 §6-E7） |
| 2(b) | 1e20 FLOPs (2.94B FLOPs/token)（loss_bs_lr_1e20） | batch $2^{19}$–$2^{23.5}$ × lr $2^{-11.25}$–$2^{-8.5}$；星号 "Optimal Hyperparameters Fitted" 在 $(2^{20},2^{-10})$，值 2.474 | 可选 |
| 3(a) | Batch size scaling curve（flops_bsz_fitting） | 横轴 "Non-Embedding Training FLOPs" 1e16–1e25，纵轴 "Optimal Batch Size (Tokens)" $2^{14}$–$2^{27}$（像素反解：底轴在 $2^{14}$，刻度标签 $2^{16}$–$2^{26}$）；图例 "7B MHA 2T Token" / "67B GQA 2T Token"；灰点 = 近优；虚线 = 拟合；星 7B "9.2M"、67B "19.7M"。**注意**：像素反解显示本面板灰点横跨约 8e16–1.6e20，超出正文 "1e17 to 2e19"；3(b) 灰点只到约 2.4e19（见 §8-23） | 是 |
| 3(b) | Learning rate scaling curve（flops_lr_fitting） | 纵轴 1.25e-4–8e-3；灰点 lr 取值 1e-3、1.4e-3、2e-3、2.8e-3（目测）；星 "4.2e-04"、"3.2e-04" | 是 |
| 4(a) | IsoFLOP curve（nosafe_flops_per_token_bpb） | 8 条 U 形曲线（1e17…3e20），横轴 M（1e8–3e10），纵轴 BPB 0.8–1.8 | 是（示意，数据见 §5.2-F5） |
| 4(b) | Optimal model scaling（nosafe_flops_flops_per_token） | $M_{opt}$ vs C，标注 "DeepSeek LLM 67B / 4.3e11 FLOPs/Token"、横轴 "4.5e23" | 是 |
| 4(c) | Optimal data scaling（nosafe_flops_tokens） | $D_{opt}$ vs C，标注 "1.04e12 Tokens"、"4.5e23" | 与 4(b) 合并 |
| 5 | Performance scaling curve（flops_bpb） | BPB vs C=MD；8 个灰点 + 两颗星（7B 约 0.715，67B 约 0.63，目测） | 是 |
| 6(a)(b)(c) | Performance scaling curves using different model scale representations（flops_bpb_6n1d/6n2d/md） | 同一批实验分别用 6N1D、6N2D、MD 作横轴；6N1 时 7B 星在曲线上方（曲线高估性能），6N2 时 67B 星在曲线下方（曲线低估性能） | 可选（概念上重要，可画示意） |
| 7 | Benchmark metrics curves of DeepSeek LLM Base（pretrain_metric） | 6 个基准随 token（0–2000B）变化；7B 在 1600B 处有明显跳升（TriviaQA 约 51→57、GSM8K 约 10→15，目测） | 可选（用于说明 LR 降档效应） |
| §5.3 无编号图 | LeetCode vs HumanEval 散点 | 正文说 "depicted in the Figure below"，HTML v1 中无图片 | 否 |

`dscoder` v2：

| 图 | 标题 | 内容 | 是否重绘 |
|---|---|---|---|
| 1 | The Performance of DeepSeek-Coder（PLT.png） | 基准雷达/柱状总览 | 否 |
| 2 | The Procedure of Dataset Creation（data_clean_pipeline.png） | 五步：Data Crawling → Rule Filtering → Dependency Parsing → Repo-level Deduplication → Quality Screening | 是 |
| 3 | The effectiveness of using FIM objective（fim.svg） | 三面板：HumanEval-Pass@1、HumanFIM-Pass@1、MBPP-Pass@1 vs 步数 0–9000；图例 fim_0（绿）、fim_0.5（红）、fim_1.0（蓝）、msp_0.5（青）；HumanFIM 面板没有 fim_0 曲线 | 是（数据为 SVG 反解，见 F10） |
| 4、5、6 | 对话示例截图 | | 否 |
| 7 | Benchmark curves during training（curve.svg） | 训练中基准曲线 | 否 |

### 5.2 本课建议图（15 张，F1–F15），附绘图所需数据

**F1 多步 vs cosine 学习率曲线（7B 真实步数）。** 横轴：已处理 token 0–2T（或步数 0–211,928）。多步：warmup 2000 步（≈18.9B token，形状论文未写，建议画线性并在说明里标注“假设”）；到 1.6T 前为 4.2e-4；1.6T–1.8T 为 1.328e-4；1.8T–2T 为 4.2e-5。对照 cosine：论文未给 cosine 的终值，建议画“终值 10% 峰值”并标注为示意。可叠加第二条“延长到 3T 的续训”虚线：第一阶段持续到 2.4T 再降档，展示“1.6T 之前完全复用”。

**F2 多步调度的“复用”示意（例题 R37 的图）。** 一根 3.2D 长的主干（峰值 LR），在 0.8D、1.6D、3.2D 三处分叉，各接一段长 0.2D、0.4D、0.8D 的“降档尾巴”。总长 4.6D vs cosine 三条独立 D、2D、4D 共 7D。

**F3 超参数 scaling law 双面板（按式 1 计算）。** $C\in[10^{16},10^{25}]$ 对数轴。左：$B_{opt}=0.2920\,C^{0.3271}$（纵轴以 $2^k$ token 标刻度）；右：$\eta_{opt}=0.3118\,C^{-0.125}$。星：7B $(8.49\times10^{22},\ 9.2\text{M},\ 4.2\times10^{-4})$，67B $(8.65\times10^{23},\ 19.7\text{M},\ 3.2\times10^{-4})$。实际使用值可另画空心点：7B $B$=9.44M，67B $B$=18.87M。可标出正文所说的拟合范围 1e17–2e19（阴影），直观显示外推 4 个数量级以上；但原图 3(a) 的灰点像素反解延伸到约 1.6e20（§8-23），阴影说明里应写“按正文”，并避免与灰点位置矛盾。灰色带宽论文未给数值，不要硬画，或标“示意”。

**F4 1e17 热力图（重绘图 2(a)）。** 数据完整见 §6-E7；以 3.6942（最小值 3.685 × 1.0025）为阈值高亮 5 个近优格；式 (1) 预测点 $(\log_2B,\log_2\eta)=(16.70,-8.74)$ 画一个十字。

**F5 IsoFLOP 示意（重绘图 4(a)）。** 8 条抛物线（在 $\log M$ 上），谷底位置用 R17 的 $M_{opt}$，谷底高度用目测 BPB：1e17→1.38、3e17→1.29、1e18→1.20、3e18→1.125、1e19→1.05、3e19→0.99、1e20→0.93、3e20→0.88（**目测，须在说明中标注**）。曲线宽度：论文图中每条大致覆盖谷底两侧各约 0.6–0.8 个 decade（目测）。

**F6 最优分配（重绘图 4(b)(c)）。** 双对数：$M_{opt}(C)$ 与 $D_{opt}(C)$ 两条直线（式 4），1e16–1e25。8 个圆点取 R17/R18。标注 67B：$M$=4.3e11、$C_{opt}$=4.5e23、$D_{opt}$=1.04e12；再加一个“实际训练点” $(8.65\times10^{23}, 2\times10^{12})$，显示 67B 实际数据量是最优的 1.92 倍（R12）。

**F7 性能 scaling 曲线（重绘图 5，示意）。** 8 个目测点（F5 的谷底 BPB）+ 星（7B 约 0.715 @ 8.49e22，67B 约 0.63 @ 8.65e23，目测）。论文**没有给出损失拟合公式**；若要画虚线，只能用笔者拟合 $L\approx0.383+30.1C^{-0.087}$（R39）并注明“笔者对目测点的拟合”。

**F8 $6N_1$、$6N_2$ 与 $M$ 的比较。** 横轴 $d$（512–8192），画 Table 3 七行的 $6N_1/M$ 与 $6N_2/M$ 两组点；叠加闭式曲线 $6d/(6d+T)$（$T$=4096 与 2048 两条），以及 $6N_2/M=(12Ld+V)/(12Ld+2LT)$（取 Table 3 各行的 $L$）。在 $y=1$ 画参考线，说明小模型 $6N_1$ 低估 57%、$6N_2$ 高估 32%。

**F9 数据质量 → 分配指数（Table 4）。** 五组 $(a,b)$ 的并列条形或“算力 ×10 时模型×/数据×”双条（R14）。三组 DeepSeek 数据按质量从左到右：早期 0.450 → 当前 0.524 → OWT2 0.578。

**F10 FIM 消融三面板（重绘 Coder 图 3，SVG 反解数据）。** 每 1000 步：

| 曲线 | 0 | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k |
|---|---|---|---|---|---|---|---|---|---|---|
| HumanEval fim_0 | 0 | .045 | .086 | .127 | .147 | .153 | .161 | .169 | .179 | .191 |
| HumanEval fim_0.5 | 0 | .050 | .088 | .118 | .133 | .142 | .155 | .166 | .164 | .160 |
| HumanEval fim_1.0 | 0 | .035 | .063 | .090 | .110 | .120 | .127 | .135 | .144 | .149 |
| HumanEval msp_0.5 | 0 | .038 | .072 | .102 | .118 | .133 | .149 | .161 | .165 | .164 |
| HumanFIM fim_0.5 | 0 | .124 | .281 | .440 | .535 | .575 | .590 | .608 | .622 | .626 |
| HumanFIM fim_1.0 | 0 | .311 | .504 | .610 | .647 | .659 | .649 | .622 | .608 | .604 |
| HumanFIM msp_0.5 | 0 | .138 | .229 | .305 | .351 | .380 | .395 | .405 | .412 | .416 |
| MBPP fim_0 | 0 | .046 | .097 | .146 | .178 | .199 | .217 | .232 | .245 | .254 |
| MBPP fim_0.5 | 0 | .037 | .080 | .129 | .163 | .189 | .211 | .222 | .229 | .233 |
| MBPP fim_1.0 | 0 | .020 | .048 | .084 | .117 | .141 | .155 | .165 | .174 | .178 |
| MBPP msp_0.5 | 0 | .036 | .072 | .110 | .143 | .169 | .187 | .203 | .218 | .229 |

（原图每 500 步一个点；SVG 坐标映射：HumanEval 面板 y 每 0.025 = 33.058 pt，HumanFIM 每 0.1 = 38.227 pt，MBPP 每 0.05 = 49.559 pt，x 每 2000 步 = 63.584 pt。）

**F11 PSM 数据布局图。** 一段代码切成 prefix / middle / suffix（字符级、切点均匀随机，`fim` 原文 "the lengths of prefix, middle, and suffix are each 1/3 of the full document in expectation"）；重排成 `<｜fim▁begin｜> prefix <｜fim▁hole｜> suffix <｜fim▁end｜> middle <eos>`（token id 32016 / 32015 / 32017 / 32014）。颜色：前缀、后缀用 `--fig-blue` 细描边（上下文），中段用 `--fig-green`（要生成的东西），哨兵 token 用 `--fig-violet`。旁注：训练语料中每个文档以概率 0.5 做这种变换，其余保持左到右。

**F12 仓库级依赖排序示意。** 一个 4 文件的小仓库：`utils.py`（无依赖）、`model.py` import utils、`train.py` import model 与 utils、`a.py`↔`b.py` 互相 import（环）。画依赖图（边从被依赖者指向依赖者，与 Algorithm 1 第 12 行一致），入度表，按“最小入度”逐步取出的顺序：utils(0) → model(1→0) → train；环内 a、b 入度都为 1，取其一（最小入度）后另一个入度降为 0。最终拼接顺序 + 每个文件开头加路径注释。

**F13 数据管线双联图。** 左：DeepSeek LLM 的 deduplication → filtering → remixing（去重率曲线 Table 1 作为插图：dump 数 1/2/6/12/16/22/41/91 对去重率 22.2/46.7/55.7/69.9/75.7/76.3/81.6/89.8，横轴建议对数）。右：Coder 的五步（图 2）+ 关键数字：规则过滤后剩 32.8%、最终 798 GB / 603M 文件 / 87 语言、10-gram 去污染。

**F14 KV cache 柱状图（接第 4 课）。** 每 token bf16 字节：7B MHA 480 KiB；67B GQA-8 380 KiB；67B 假想 MHA 2.97 MiB；Coder 6.7B 512 KiB；Coder 33B GQA-8 248 KiB；Coder 5.7B MQA 16 KiB（R25）。

**F15 架构“深 vs 宽”对比。** 三个模型并排的长方形（高 = 层数，宽 = $d$）：LLaMA-1 65B（80 层 × 8192，`llama1` Table 2）、DeepSeek 67B（95 层 × 8192）、DeepSeek 7B（30 层 × 4096）对 LLaMA-1 6.7B（32 层 × 4096）。FFN 比值标注：DeepSeek 67B 22016/8192 = 2.6875；Llama 2 做法是 GQA 时 FFN ×1.3（`llama2` A.2.1）。LLaMA-2 70B 的具体层数/FFN 未能从一手来源核实（§8），不要画进图里。

---

## 6. 算例与练习素材

**E1（计算）7B 的 M 与 6N 差多少。** 已知 $L$=30，$d$=4096，$T$=4096。求 $6N_1$、$M$、$6N_1/M$，并解释差额来自哪里。
答：$6N_1=72\cdot30\cdot4096^2=3.624\times10^{10}$；$M=4.228\times10^{10}$；比值 $=6d/(6d+T)=24576/28672=0.857$；差额 14.3% 就是注意力的 $QK^\top$ 与 $AV$ 两次矩阵乘（每层每 token $12dT$）。

**E2（计算）用式 (1) 为 7B 和 67B 选超参数。** 给定 $M_{7B}$=4.2467e10、$M_{67B}$=4.3273e11、$D$=2e12。
答：$C$=8.49e22 与 8.65e23；$\eta$=4.24e-4 与 3.17e-4（论文用 4.2e-4 与 3.2e-4）；$B$=9.24M 与 19.74M token（即 2255 与 4818 条 4096 序列；论文用 2304 与 4608 条）。追问：若把 67B 的 M 按式 (2) 算成 4.97e11，$B$ 会变成 20.65M——与论文图 3 的 “19.7M” 不符，说明论文按真实 GQA 结构计 M。

**E3（推导）证明 $a+b=1$ 且 $M_{base}D_{base}=1$。** 由 $C=M_{opt}D_{opt}$ 对所有 $C$ 成立直接得出；数值核对 0.5243+0.4757=1、0.1715×5.8316=1.0001。

**E4（计算）67B 是不是“算力最优”？** 用式 (4) 反解 67B 的 $C_{opt}$ 与 $D_{opt}$。
答：$C_{opt}=(4.327\times10^{11}/0.1715)^{1/0.5243}=4.51\times10^{23}$，$D_{opt}=1.04\times10^{12}$；实际 2T，是最优的 1.92 倍。7B 同法得 $D_{opt}$=1.27e11，实际是 15.8 倍。讨论：为什么仍然值得（推理成本、同一份数据两个模型共用）——此处属“笔者推测”，论文没有给这个理由。

**E5（计算）数据质量改变分配。** 算力从 1e22 增到 1e23，按早期数据、当前数据、OpenWebText2 的指数，模型与数据各扩大多少倍？
答：×2.82/×3.55；×3.34/×2.99；×3.78/×2.64（R14）。对照 Kaplan ×5.37/×1.86、Chinchilla ×3.09/×3.24。

**E6（计算）多步调度的复用收益。** 同一模型要在 D、2D、4D 三个数据量上各训一次。cosine 各自从头训；多步（80/10/10）可以共享第一阶段。求总训练量。
答：cosine 7D；多步 $3.2D+0.2D+0.4D+0.8D=4.6D$，省 34.3%。追问：若比例改成 60/20/20，共享主干 $0.6\times4D=2.4D$，尾巴 0.4D+0.8D+1.6D=2.8D，总 5.2D，只省 25.7%——这正是论文 "balancing reuse ratios in continual training and model performance" 的权衡。

**E7（读数）1e17 网格里有几个“近优”格？** 数据（行 = batch，列 = lr；lr 列依次为 $2^{-10.5},2^{-10.25},2^{-10},2^{-9.75},2^{-9.5},2^{-9.25},2^{-9},2^{-8.75},2^{-8.5},2^{-8.25},2^{-8},2^{-7.75}$；从论文图 2(a) 格内印刷数值转录）：

```
2^15   3.872 3.872 3.886 3.893 3.900 3.877 3.891 3.890 3.878 3.902 3.912 3.918
2^16   3.787 3.781 3.794 3.775 3.782 3.759 3.755 3.784 3.750 3.747 3.761 3.805
2^16.5 3.757 3.740 3.711 3.712 3.741 3.733 3.720 3.715 3.708 3.707 3.693 3.781
2^17   3.747 3.744 3.734 3.724 3.712 3.695 3.698 3.693 3.690 3.753 3.685 3.744
2^17.5 3.779 3.740 3.741 3.726 3.701 3.709 3.709 3.704 3.746 3.754 3.743 3.764
2^18   3.805 3.770 3.769 3.739 3.734 3.730 3.718 3.691 3.785 3.775 3.788 3.809
2^18.5 3.880 3.867 3.839 3.800 3.796 3.786 3.910 3.852 3.835 3.845 3.862 3.808
2^19   4.136 4.071 3.999 3.964 3.957 4.047 3.960 3.970 3.976 3.920 3.956 3.960
2^19.5 4.653 4.456 4.764 4.513 4.347 4.197 4.281 4.222 4.156 4.111 4.078 4.166
2^20   4.958 5.152 5.098 5.013 4.964 5.001 4.965 4.770 4.834 4.715 4.741 4.614
```
答：最小 3.685（$2^{17}$，$2^{-8}$）；阈值 3.685×1.0025=3.6942；近优格 5 个：$(2^{16.5},2^{-8})$ 3.693、$(2^{17},2^{-8.75})$ 3.693、$(2^{17},2^{-8.5})$ 3.690、$(2^{17},2^{-8})$ 3.685、$(2^{18},2^{-8.75})$ 3.691。式 (1) 预测 $(2^{16.70},2^{-8.74})$ 落在这片区域中间。注意此图标题写的是 "Training loss"，正文称 "generalization error"。1e20 网格（图 2(b)）同一阈值下有 20 个近优格，最小 2.474。

**E8（计算）KV cache：GQA 帮 67B 省了多少？**
答：67B GQA-8 每 token $2\cdot95\cdot8\cdot128\cdot2$=389,120 B=380 KiB；若 MHA 则 ×8=2.97 MiB；4096 上下文单序列 1.48 GiB vs 11.9 GiB。7B（MHA，30 层）反而是 480 KiB/token，比 67B 还大。

**E9（计算）逐位复现参数量。** 按 config 算 7B 与 67B 的精确参数量。
答：每层 = $2d^2+2d\,n_{kv}d_h+3d\,d_{ff}+2d$；总 = $L\times$每层 $+d+2Vd$。7B：6,910,365,696；67B：67,425,001,472；与官方 checkpoint 的 `total_size`/2 完全一致。可顺带问：“7B”其实是 6.91B，其中 12.1% 是嵌入与输出头。

**E10（概念+计算）词表为什么是 102400？** 答：BPE 条目 100000 + 15 个追加 token = 100015；训练时取 102400（= 100 × 1024 = 400 × 256，笔者注：是 64/128/256 的倍数，便于矩阵分块），预留 2385 行。追问：15 个追加 token 中只有 2 个是 BOS/EOS，其余 13 个是单个 Latin-1 字符（`tokenizer.json` 实测）。

**E11（构造）FIM 训练样本。** 文档 `def add(a, b):\n    return a + b\n`，在字符位置切成 prefix=`def add(a, b):\n    ret`、middle=`urn a `、suffix=`+ b\n`，写出 PSM 训练序列与推理提示。
答：训练：`<｜fim▁begin｜>def add(a, b):\n    ret<｜fim▁hole｜>+ b\n<｜fim▁end｜>urn a <｜end▁of▁sentence｜>`；推理时只给到 `<｜fim▁end｜>`，模型续写中段。说明为什么切在词中间也要训练（`fim` §8.1 建议字符级切分以获得 "mid-token robustness"）。

**E12（读图）FIM 率的权衡。** 用 F10 数据：9000 步时 HumanEval fim_0 / fim_0.5 / fim_1.0 = 0.191 / 0.160 / 0.149；HumanFIM fim_0.5 / fim_1.0 = 0.626 / 0.604（但 fim_1.0 在 5000 步达峰 0.659）。问：论文说 "peak performance on the HumanEval-FIM with a 100% FIM rate" 严格吗？为什么 0.5 是折中？与 Bavarian 的 "FIM-for-free" 是否矛盾？
答要点：“峰值”只对训练中途成立，终点 0.5 反超；0.5 在 HumanEval 上比 fim_0 低 3.1 个点，说明在这个 1.3B/Python 子集的设定里 FIM 并非完全免费。Bavarian 也用 HumanEval 通过率检验过，结论是各 FIM rate 表现相近（§7-E24），所以两篇在同一指标上结论相反；差异来自模型规模、数据与训练长度，不能直接判定谁对。

**E13（计算）Coder 的长上下文续训有多少 token？** 答：$1000\times512\times16384=8.39\times10^9$；RoPE 线性插值因子 4 → 理论 $16384\times4=65536$（64K）；论文说可靠范围仍是 16K。追问：README 说 16K 阶段是 200B token，与 8.4B 相差约 24 倍，如何对待？（答：两处来源矛盾，课文只能并列引用。）

**E14（计算）注意力在 16K 上下文的占比。** Coder 6.7B（$d$=4096）在 $T$=16384 时，按 $M$ 的记法注意力占比多少？答：$16384/(6\cdot4096+16384)=40\%$；对照 4K 时 14.3%。

---

## 7. 勘误与常见误解

**E1 发布日期 ≠ 论文日期。** 课程地图写“2024 年 1 月：7B / 67B”。模型（HF 与 GitHub）在 **2023-11-29** 公开（证据是 HF 仓库 createdAt 与 GitHub 仓库创建时间、当天的 README 提交；createdAt 本身只证明创建，不证明当时已公开），论文 2024-01-05。DeepSeek-Coder 模型 2023-11-02 前后公开，论文 2024-01-25。课文应写“2023 年 11 月发布，2024 年 1 月发表技术报告”。

**E2 “15 个 special token”并不都是特殊符号。** `tokenizer.json` 中 id 100000–100014 共 15 个追加 token，只有 BOS/EOS 标 `special: true`；另外 13 个是单个 Latin-1 字符（ø ö ú ÿ õ ÷ û ý À ù Á þ ü），标 `special: false`。Coder 的 tokenizer 里也有同样 13 个字符。它们为什么以追加 token 形式存在，官方没有说明（§8）。

**E3 式 (2) 的 M 只对 MHA + $d_{ff}=\tfrac83d$ 精确。** 67B 用 GQA-8 且 FFN 22016，按式 (2) 算得 4.97e11，按真实结构算得 4.33e11；论文图 4(b) 标注的是 **4.3e11**，图 3 的批大小 19.7M 也只有用 4.33e11 才能复现（R7 vs R7'）。核查补充：图 3(a) PNG 像素反解，67B 星的横坐标 $\log_{10}C=23.937$，正好等于 $\log_{10}(8.6545\times10^{23})$；若用式 (2) 的 M 应在 23.998（约差 28 像素，可分辨）。7B 星在 22.930（预测 22.929）。图 3 图例原文就写 "67B GQA 2T Token" / "7B MHA 2T Token"。结论：论文实际使用的是“真实结构的非嵌入 FLOPs/token”，式 (2) 是 MHA 下的简化写法。课文若直接把 67B 代进式 (2) 会得到与论文图不一致的数字。

**E4 M 的注意力项不打因果折扣，是 Kaplan 的 2 倍。** DeepSeek 计 $12LdT$（每个 token 都与全部 $T$ 个位置做 $QK^\top$ 与 $AV$）；Kaplan 式 (2.2) 前向上下文项为 $2Ln_{ctx}d$，×3 后为 $6Ln_{ctx}d$。两者都是约定，不是精确 FLOPs：因果掩码下平均每个 token 只看约 $T/2$ 个位置（笔者注：因果注意力核是否真的省下这一半取决于实现，本课来源未核对）。不要把 $M$ 说成“精确 FLOPs”。

**E5 $N_2$ 只加了一份词表矩阵。** Table 3 的 $N_2=N_1+Vd$（例：25.17M + 52.43M = 77.6M），而 DeepSeek LLM 的输入嵌入与输出头不共享（`tie_word_embeddings: false`），真实“完整参数”是 $N_1+2Vd$。$6N_2$ 里的 $6Vd$ 对应的是输出头的前向+反向 FLOPs（嵌入查表几乎不花算力），这正是论文说的 "vocabulary computation"。

**E6 batch size 的单位。** Table 2 的 2304 / 4608 是**序列数**（×4096 = 9.44M / 18.87M token；Table 2 表头原文就是 "Sequence Batch Size"）；式 (1) 与图 3 的 $B$ 是 **token 数**。混用会差 4096 倍。

**E7 多步调度的切点按 token 比例定义。** 原文 "after processing 80% of the training tokens"；README 给出 1.6T / 1.8T。31.6% 不是随便取的，是 $10^{-1/2}$：两次各乘 $\sqrt{1/10}$，最终 10%（Coder §3.4 明写）。另外，论文没说 warmup 的形状（线性与否）、也没说图 1 cosine 对照的终值。

**E8 Table 3 第 5 行舍入不一致。** $L$=32、$d$=4096：$6N_1/M=24576/28672=0.857$，印作 0.85（其余行都是四舍五入）。

**E9 式 (3) 记号笔误。** 优化变量是 $M,D$，目标却写成 $L(N,D)$；应理解为 $L(M,D)$。

**E10 “OpenAI (OpenWebText2)”标签不准确。** Kaplan 等人用的是 OpenAI 内部的 **WebText2**（`kaplan` §2.2）；**OpenWebText2** 是 The Pile（Gao et al. 2020）新引入的开源复刻（`pile`）。DeepSeek §3.3 写 "OpenWebText2, which was utilized in the previous study of scaling laws (Kaplan et al., 2020)" 把两者混了。Table 4 第一行的 0.73/0.27 是 Kaplan 在 WebText2 上的结果；第五行 "Ours (OpenWebText2)" 才是 DeepSeek 自己在 OpenWebText2 上拟合的。

**E11 Table 4 不同行的 $a$ 不是同一个量。** Kaplan/Chinchilla 的 $a$ 是 $N_{opt}\propto C^a$；DeepSeek 的 $a$ 是 $M_{opt}\propto C^a$。由于 $M/N$ 随形状（$T/d$）变化，两者只能近似比较；表头写的 "$N_{\mathrm{opt}}(M_{\mathrm{opt}})\propto C^{a}$" 已暗示这一点。另外 Kaplan 的 0.73 是对 $C_{\min}$ 的指数（PF-day 单位，且数据侧 $D\sim C^{0.27}$ 由 $B\propto C^{0.24}$ 与 $S\propto C^{0.03}$ 合成）。

**E12 Chinchilla 的 0.49/0.51 是三种方法之一。** Chinchilla Table 2：方法 1 0.50/0.50，方法 2（IsoFLOP）0.49/0.51，方法 3（参数化损失）0.46/0.54。常见说法“Chinchilla 是 0.5/0.5”与 DeepSeek 引的 0.49/0.51 都对，但指的方法不同。

**E13 “1000× 外推”是数量级说法。** IsoFLOP 最大预算 3e20；7B 训练算力 8.49e22（283×），67B 8.65e23（2885×）。按正文，超参数公式的拟合范围只到 2e19，到 67B 是 4.3 万倍外推（R20）；但图 3(a) 的 batch 灰点像素反解延伸到约 1.6e20（图 3(b) 的 lr 灰点只到约 2.4e19），正文与图不完全一致（§8-23）。

**E14 “67B Chat 超过 GPT-3.5” 要看是哪个版本。** MT-Bench：67B Chat 8.35 < GPT-3.5-turbo-0613 8.39（论文措辞 "comparable"），只有 DPO 版 8.76 超过；AlignBench 上非 DPO 版 6.43 已超过 gpt-3.5-turbo-0613 的 6.08。摘要的 "superior performance compared to GPT-3.5" 在英文 MT-Bench 上只对 DPO 版成立；引言其实写明了前提："Following SFT and DPO, the DeepSeek 67B chat model outperforms GPT-3.5 in both Chinese and English open-ended evaluations."

**E15 GitHub README 与论文的数字漂移。** 匈牙利高考：README 介绍段写 67B Chat "exceptional score of 65"，论文 Table 9 与同仓库 `evaluation/more_results.md` 都是 **58**（README 另有 "We have rectified an error from our initial evaluation" 的说明，但介绍段未改）。IFEval：`evaluation/more_results.md` 表格为 59.1（README 正文没有这个数字的文字版，只给图片 `if_eval.png` 和说明句 "Here, we used the first version released by Google for the evaluation. For the Google revised test set evaluation results, please refer to the number in our paper."），论文 55.5（修订版测试集）。7B Chat GSM8K：README 62.6，论文 Table 6 63.0。课文一律引用论文。

**E16 系统提示：论文 vs README。** 论文 Table 14：加系统提示后 7B Chat MT-Bench 7.15→7.11，67B Chat 8.35→8.58；README 却说 "since the system prompt is not compatible with this version of our models, we DO NOT RECOMMEND including the system prompt in your input."

**E17 base 模型 config 的 bos/eos 与分词器不一致。** `deepseek-llm-{7b,67b}-base/config.json` 写 `bos_token_id: 1`、`eos_token_id: 2`；`generation_config.json`、tokenizer 与 chat 版 config 都是 100000/100001。实际推理由 tokenizer 与 generation_config 决定。`initializer_range: 0.02` 与论文的 0.006 不同，这是 HF 字段默认值，不代表训练时的初始化。

**E18 Coder 词表 32,000 vs 32,256。** 论文说 32,000；实际 BPE 条目 32,000 + 22 个追加 token（含 FIM 哨兵与对话标记）= 32,022，嵌入行 32,256。

**E19 Coder FIM 哨兵名称与 v1 笔误。** 论文 v2 写 `<｜fim_start｜>`、`<｜fim_hole｜>`、`<｜fim_end｜>`；tokenizer 实际为 `<｜fim▁begin｜>`、`<｜fim▁hole｜>`、`<｜fim▁end｜>`。论文 v1 的模板把开头也印成 `<｜fim_hole｜>`，v2 修正（这是 v1→v2 的主要实质改动之一；另一处是补上 LeetCode 基准的 GitHub 链接、Table 7 补 % 号）。

**E20 Coder 长上下文阶段的 token 数自相矛盾。** 论文 §3.6：额外 1000 步、batch 512、16K 序列 = 8.4B token；GitHub README："Further Pre-training using an extended 16K window size on an additional 200B tokens"，并称第一步 "1.8T tokens and a 4K window size"。两者差约 24 倍，无法由公开材料调和（§8）。

**E21 Coder v1.5 的续训量：论文内部矛盾。** §5 写 "on 2 trillion tokens"，结论 §6 写 "This additional training involved processing a diverse dataset comprising 2 billion tokens"。以 §5 的 2T 为准更合理（笔者判断，未核实）。

**E22 Coder Table 6 的 Mean 列多处不是简单平均。** SantaCoder 三语 44.0/62.0/74.0 平均 60.0，印 69.0；CodeLlama-7B 74.0 印 69.7；DS 7B 78.1 印 80.7；DS 33B 78.2 印 81.2；另三行吻合（R38）。可能是按题数加权或录入错误，论文未说明。同表把模型写成 "1B/7B"，与正文 1.3B/6.7B 不一致。Table 3 的 Avg 列则都是简单平均（已核对）。

**E23 “100% FIM 率时 HumanFIM 最好”只对训练中途成立。** SVG 反解：fim_1.0 在 5000 步达 0.659 后回落到 0.604，fim_0.5 在 9000 步为 0.626，反超（F10）。

**E24 FIM 并非在所有设定下“免费”。** Bavarian 等人报告 FIM-for-free（左到右损失不变，FIM rate 50%–90% 都合理）；DeepSeek-Coder 在 1.3B + Python 子集上看到 HumanEval/MBPP 的下降（fim_0.5 比 fim_0 低约 3 个点），因而称之为 "trade-off"。**核查更正**：两者的分歧不在指标上——Bavarian 同样测了 HumanEval 通过率（`fim` §2.1："For code, we measure the pass rates on HumanEval"），而且在 FIM-rate 消融（6 个大模型、50B token，rate 0–1.0）中写道 "For HumanEval, the left plot in Figure 5 shows all models irrespective of FIM rate have a similar performance"。所以同一指标（HumanEval pass@1）上两篇的结论确实相反；可列举的设定差异是模型规模、数据（DeepSeek 只用 Python 子集）、训练长度（DeepSeek 消融只到 9000 步，步长对应的 token 数未给出）。课文应并列呈现，不下“谁错”的结论。

**E25 Coder 的“87 种语言”。** 论文 Table 1 恰好 87 行；README 的支持语言列表有 89 项（多了 markdown 与 restructuredtext，它们属于 10% 的自然语言部分）。

**E26 Coder 的模型规模命名。** “6.7B”实为 6.74B，“33B”实为 33.34B，“1.3B”实为 1.35B；README 还列出论文没有的 5.7B（MQA 版）。§3.7 正文写 "An example of using DeepSeek-Coder-Instruct 34B"，而图 4 标题写 33B——“34B”是笔误。

**E27 仓库级预训练的收益不是全语言的。** Table 7 中去掉仓库级预训练后，Python 的 EM 从 16.14% 变为 16.02%、ES 从 66.51% 变为 66.65%（几乎无差别，ES 反而略高），下降只出现在 Java/TypeScript/C#，论文原句也只点了这三种。

**E28 “加深而非加宽”的对照对象。** 67B 为 95 层 × 8192；LLaMA-1 65B 为 80 层 × 8192（`llama1` Table 2，已核实）。常被引用的“LLaMA-2 70B 为 80 层、FFN 28672”在 Llama 2 论文中没有印出（论文只在附录 A.2.1 说 GQA 时 FFN ×1.3），官方 config 受访问限制，本次未能核实（§8）。

**E29 SFT 数据量的两种说法。** 引言 "over 1 million instances"，§4 "around 1.5 million"——不矛盾，但课文用 §4 的 150 万并给出拆分。

**E30 “7B”的层数不是 32。** DeepSeek LLM 7B 是 30 层（LLaMA 7B 为 32 层）；因为词表 102400（LLaMA 32000）多出约 5.7 亿嵌入参数，少两层后总参数仍约 6.9B（笔者解释，论文只说 "maintaining parameter consistency with other open-source models"）。

---

## 8. 未能核实 / UNVERIFIED

1. **总训练 GPU 小时与集群规模**：`dsllm` 未报告 7B/67B 的 GPU 小时数、GPU 型号或数量。`dscoder` §3.5 只说使用 A100 与 H800 集群，也没有 GPU 小时。
2. **图 5 的损失拟合公式**：论文没有给出 BPB 对 $C$ 的拟合函数与系数；§4-R39 是笔者对目测点的拟合，只能用于重绘示意。
3. **图 3、图 5 的灰色带宽**：论文没有给出数值定义（是否为 0.25% 近优区间的包络）。
4. **IsoFLOP 各点的精确数值**：图 4(a) 只有图，未列数据表；§5.2-F5 的谷底 BPB 为目测。
5. **Warmup 形状与 cosine 对照终值**：论文只说 "2000 warmup steps"，未说线性；图 1(a) 的 cosine 终值未给。
6. **数据管线细节**：filtering 用了哪些“linguistic and semantic evaluations”、remix 的领域配比、2T 中中英文与代码/数学的比例，论文均未给数值。README 提到 MinhashLSH 与 RefinedWeb + CCNet 起点，属制品说法。
7. **13 个单字符追加 token 的用途**：为什么以追加 token 而不是 BPE 词表条目存在，官方未说明。
8. **67B 的 $M$ 具体如何计算**：§7-E3 的结论（论文按真实 GQA 结构计 $M$）是由图 3、图 4 的标注数字反推得到的，**论文正文没有写明**；吻合到 3 位有效数字，核查时又用图 3(a) 星号的像素横坐标独立确认（$\log_{10}C$=23.937，式 (2) 版应为 23.998），但“怎么算的”仍属推断。
9. **DPO 偏好对的标注方式**：论文只说用 DeepSeek Chat 生成候选，没说谁/什么来判定 chosen/rejected。
10. **SFT 的 batch size、序列长度**：未报告。
11. **Coder 长上下文阶段的真实 token 数**：8.4B（论文）vs 200B（README），无法调和（§7-E20）。
12. **Coder Table 2 batch size 的单位**：未写；“按 4096 序列”的解读与 DeepSeek LLM 式 (1) 的预测吻合（6.7B、33B），但 1.3B 的 1024 与预测 1391 差距较大。
13. **Coder 的质量模型（quality model）与编译器过滤细节**：§2.4 只一句带过。
14. **Coder Table 6 Mean 列的计算方式**（§7-E22）。
15. **Coder v1.5 的续训 token 数**：2T（§5）vs 2B（§6）。
16. **LLaMA-2 70B 的层数与 FFN 维度**（80 层、28672）：Llama 2 论文未印；`meta-llama/Llama-2-70b-hf` 的 config 需登录（HTTP 401），本次未核实。
17. **论文 §5.3 的 LeetCode 散点图**：HTML v1 中没有这张图片，PDF 未单独核对。
18. **匈牙利高考 58 vs 65 哪个是最终值**：论文与 more_results.md 为 58，README 介绍段为 65；README 的“rectified”说明所指的修订方向不明。
19. **Coder 图 3 各曲线的精确值**：F10 由 SVG 路径坐标反解，精度受绘图坐标限制（约 ±0.001），不是论文正文数字。
20. **FIM 率 0.5 在 Coder 全量模型训练中的 PSM/SPM 比例**：论文说 "following the PSM mode"，是否完全不含 SPM 未明确；Bavarian 推荐 PSM+SPM 联合训练。
21. **Chinchilla 与 DPO 的正式发表场所**：arXiv 元数据未写，本次未核对会议版本（GQA 的 EMNLP 2023 来自其 arXiv comment）。
22. **DeepSeek LLM 与 Coder 的正式发表状态**：截至 2026-09-24 只核对了 arXiv 版本，未检索是否有期刊/会议版本。
23. **超参数拟合实际用了哪些预算**（核查时新增）：正文说 "compute budgets ranging from 1e17 to 2e19"；对 arXiv HTML v1 的 `flops_bsz_fitting.png` 做像素反解（竖网格线定标 462.75 px/decade，灰点填充色 RGB 185），图 3(a) 灰点中心横跨 $\log_{10}C\approx16.92$–$20.20$（约 8.3e16–1.6e20）；`flops_lr_fitting.png` 的图 3(b) 灰点只到约 19.39（约 2.4e19）。两张图的最左点都在约 8.3e16 而不是 1e17。可能 3(a) 另外画进了 1e20 验证实验（图 2(b)）的近优点，也可能横轴口径与名义预算不同，论文没有说明。课文引用拟合范围时应写“正文称 1e17–2e19”。

---

## 9. 参考文献卡片草稿

```
[dsllm] DeepSeek LLM: Scaling Open-Source Language Models with Longtermism
    DeepSeek-AI (Bi, X. et al., 86 authors) (2024), arXiv preprint. arXiv:2401.02954v1
    https://arxiv.org/abs/2401.02954
    finding: 用非嵌入 FLOPs/token M 表示模型规模，拟合出 η_opt、B_opt 随算力的幂律与 M_opt、D_opt 的分配指数（a=0.5243），并发现数据质量越高越应把算力分给模型。
    matters: 本课主体；7B/67B 的全部规格、多步调度、超参数 scaling law 与对齐流程都以它为准，也是第 7–12 课所有模型沿用的训练配方的起点。

[dscoder] DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence
    Guo, D., Zhu, Q., Yang, D. et al. (2024), arXiv preprint (cs.SE). arXiv:2401.14196v2
    https://arxiv.org/abs/2401.14196
    finding: 1.3B/6.7B/33B 代码模型在 2T token（87% 代码）上从头训练，采用仓库级依赖排序与去重、PSM 格式 50% FIM 率和 16K 上下文。
    matters: 作为同期旁支，展示 DeepSeek LLM 的超参数规律如何被直接复用，并引入 FIM 与仓库级数据这两种后续 Coder-V2 继承的做法。

[kaplan] Scaling Laws for Neural Language Models
    Kaplan, J. et al. (2020), arXiv preprint. arXiv:2001.08361
    https://arxiv.org/abs/2001.08361
    finding: 在 WebText2 上得到 N_opt ∝ C^0.73、D ∼ C^0.27，并在 d_model ≫ n_ctx/12 的假设下把训练算力近似为 6N（忽略上下文项）。
    matters: 本课用它说明 6N1 的来历与局限，以及 DeepSeek Table 4 第一行数字的真实数据集（WebText2 而非 OpenWebText2）。

[chinchilla] Training Compute-Optimal Large Language Models
    Hoffmann, J. et al. (2022), arXiv preprint (venue not re-verified here). arXiv:2203.15556
    https://arxiv.org/abs/2203.15556
    finding: 三种方法估计的最优分配指数分别为 0.50/0.50、0.49/0.51（IsoFLOP）、0.46/0.54，结论是参数与数据应大致同比例增长。
    matters: DeepSeek 直接借用它的 IsoFLOP 方法并引用其方法 2 的 0.49/0.51 作对照，课文需要说明引的是哪种方法。

[fim] Efficient Training of Language Models to Fill in the Middle
    Bavarian, M. et al. (2022), arXiv preprint. arXiv:2207.14255
    https://arxiv.org/abs/2207.14255
    finding: 以概率 p（主实验 0.5）在字符级把文档切成前缀/中段/后缀并重排为 PSM/SPM 格式，左到右能力不受损（FIM-for-free），推荐 FIM 率 50%–90%。
    matters: DeepSeek-Coder 的 FIM 实现直接照此执行，而 Coder 的消融观察到了与“免费”不同的权衡，课文需要并列两者。

[llama1] LLaMA: Open and Efficient Foundation Language Models
    Touvron, H. et al. (2023), arXiv preprint. arXiv:2302.13971
    https://arxiv.org/abs/2302.13971
    finding: Table 2 给出 6.7B（32 层 × 4096）到 65.2B（80 层 × 8192、64 头）的架构与 4M batch 等设置；6.7B/13B 训练 1.0T token，32.5B/65.2B 训练 1.4T token。
    matters: 作为 DeepSeek LLM “微观沿用 LLaMA、宏观改层数”的对照基线，用于画“深 vs 宽”对比图。

[llama2] Llama 2: Open Foundation and Fine-Tuned Chat Models
    Touvron, H. et al. (2023), arXiv preprint. arXiv:2307.09288
    https://arxiv.org/abs/2307.09288
    finding: 附录 A.2.1 在 30B 规模上比较 MHA/MQA/GQA，为保持参数量把 GQA 的 FFN 维度乘以 1.3、MQA 乘以 1.33。
    matters: 正是 DeepSeek 所说“加宽 FFN 的常见做法”的一手出处，让“67B 选择加深”这一决定有具体对照。

[gqa] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
    Ainslie, J. et al. (2023), EMNLP 2023. arXiv:2305.13245
    https://arxiv.org/abs/2305.13245
    finding: 把查询头分成 G 组、每组共享一对 K/V 头，GQA-1 即 MQA、GQA-H 即 MHA，质量接近 MHA 而速度接近 MQA。
    matters: 67B 与 Coder 33B 都用 GQA-8，本课的 KV cache 账（380 KiB vs 2.97 MiB）以此为定义基础。

[dpo] Direct Preference Optimization: Your Language Model is Secretly a Reward Model
    Rafailov, R. et al. (2023), arXiv preprint (venue not re-verified here). arXiv:2305.18290
    https://arxiv.org/abs/2305.18290
    finding: 通过奖励的重参数化把 RLHF 目标化为对偏好对的简单分类损失，无需强化学习采样。
    matters: DeepSeek LLM Chat 的第二阶段对齐直接用 DPO（1 epoch、lr 5e-6、batch 512），课文在此回扣第 3 课。

[pile] The Pile: An 800GB Dataset of Diverse Text for Language Modeling
    Gao, L. et al. (2020), arXiv preprint. arXiv:2101.00027
    https://arxiv.org/abs/2101.00027
    finding: 引入 OpenWebText2 与 BookCorpus2，作为 OpenWebText 与 BookCorpus 的扩展版本。
    matters: 用于纠正 DeepSeek 把 Kaplan 的 WebText2 写成 OpenWebText2 的混淆，并说明 DeepSeek 第三个数据集的来源。

[hf-llm-cfg] deepseek-ai/deepseek-llm-{7b,67b}-{base,chat}: config.json, tokenizer.json, pytorch_model.bin.index.json
    DeepSeek-AI (2023), Hugging Face model repository. kind: artefact
    https://huggingface.co/deepseek-ai/deepseek-llm-67b-base
    finding: 给出论文未印的 FFN 维度 11008/22016、rope_theta 10000、非共享嵌入，以及 100000+15 的分词器实际构成与逐位的参数总量。
    matters: 本课所有精确参数量、KV cache 与“M 按真实结构计算”的推断都依赖这些文件，也暴露了 bos/eos、initializer_range 等制品漂移。

[hf-coder-cfg] deepseek-ai/deepseek-coder-{1.3b,5.7bmqa,6.7b,33b}-base, deepseek-coder-7b-base-v1.5: config.json, tokenizer.json
    DeepSeek-AI (2023–2024), Hugging Face model repository. kind: artefact
    https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
    finding: 证实 Coder 的线性 RoPE 因子 4、θ=100000、16384 上下文、词表 32256，以及 FIM 哨兵的真实名称 fim▁begin/hole/end。
    matters: 课文画 FIM 布局图与写推理示例时必须用 tokenizer 的真实 token，而不是论文里的 fim_start 写法。

[gh-llm] deepseek-ai/DeepSeek-LLM GitHub README and evaluation/more_results.md
    DeepSeek-AI (2023), GitHub repository. kind: artefact
    https://github.com/deepseek-ai/DeepSeek-LLM
    finding: 以绝对 token 数（1.6T、1.8T）说明多步调度切点，并给出中间 checkpoint 下载方式与部分与论文不同的评测数字。
    matters: 用来核对调度切点和发布日期，同时是“README 与论文数字漂移”（匈牙利 65 vs 58、系统提示建议）的证据来源。

[gh-coder] deepseek-ai/DeepSeek-Coder GitHub README
    DeepSeek-AI (2023), GitHub repository. kind: artefact
    https://github.com/deepseek-ai/DeepSeek-Coder
    finding: 描述两段预训练（4K 窗口 1.8T token + 16K 窗口 200B token）与 2B token 指令微调，列出 89 项支持语言与 5.7B 版本。
    matters: 与论文 §3.6 的 1000 步续训相矛盾，是本课“论文与代码仓库不一致”一节的主要例证。
```

---

## 10. 核查记录

> 2026-09-24，对抗式复核（独立于研究代理）。所有论文引文用 `arxiv.py text/grep` 重新抓取 arXiv HTML 后核对；HF/GitHub 制品在浏览器里直接读 raw 文件与 API（`huggingface.co/<repo>/raw/main/...`、`huggingface.co/api/models/<repo>`、`api.github.com/repos/...`、`raw.githubusercontent.com/...`）；论文图用 arXiv HTML v1 的原始 PNG/SVG 做像素或矢量坐标反解；所有算术在 Python 3.11 中重算。

**逐项核对且确认无误（未改动）**

| # | 论断 | 核对方式与来源 |
|---|---|---|
| 1 | Table 2：7B 30 层/4096/32 头/32 KV/2304/4.2e-4/2.0T；67B 95/8192/64/8/4608/3.2e-4/2.0T | `dsllm` v1 Table 2 原文；两份 base config 同 |
| 2 | 式 (1) 0.3118·C^-0.1250、0.2920·C^0.3271 | `dsllm` §3.1 逐符号对照 |
| 3 | 式 (2) 三行（72、6、12 系数与 $n_{\rm layer},d_{\rm model},n_{\rm vocab},l_{\rm seq}$） | `dsllm` §3.2 逐符号对照 |
| 4 | 式 (3) 印作 $L(N,D)$；式 (4) 0.1715/0.5243/5.8316/0.4757 | `dsllm` §3.2 |
| 5 | Table 3 七行全部复现，只有第 5 行 0.857 印作 0.85 | Python 重算（§4-R8 数字逐项一致） |
| 6 | Table 4 五行 (a,b) 与 "OpenAI (OpenWebText2)" 标签；§3.3 把 OpenWebText2 说成 Kaplan 所用 | `dsllm` Table 4、§3.3；`kaplan` §2.1 "We train language models on WebText2"；`pile` "We also introduce OpenWebText2" |
| 7 | Chinchilla 方法 1/2/3 = 0.50/0.50、0.49/0.51、0.46/0.54 | `chinchilla` Table 2 与 §3.1–3.3 |
| 8 | Kaplan 式 (2.2)、Table 1 "Attention: Mask" 项、$C\approx6N$ 的省略条件 | `kaplan` §2.1 |
| 9 | 多步调度 2000 步 warmup、80%→31.6%、90%→10%；README 1.6T/1.8T | `dsllm` §2.3；DeepSeek-LLM README 原句 |
| 10 | 分词器 100000 + 15 = 100015，训练 102400；tokenizer.json 实测 15 个追加 token 中仅 BOS/EOS 为 special，其余 13 个 Latin-1 单字符 | `dsllm` §2.1；`deepseek-llm-7b-base/tokenizer.json`（BPE 词表 100000、merges 99757） |
| 11 | 精确参数量 6,910,365,696 / 67,425,001,472 | 按 config 重算，与 `pytorch_model.bin.index.json` 的 total_size 13,820,731,392 / 134,850,002,944 各除以 2 逐位一致 |
| 12 | M、C、式 (1)(4) 的全部派生数（R1–R7'、R11–R13、R15–R19、R21–R23） | Python 重算，全部吻合 |
| 13 | 图 3 标注 9.2M / 19.7M、4.2e-04 / 3.2e-04；图 4(b)(c) 标注 "4.3e11 FLOPs/Token"、"4.5e23"、"1.04e12 Tokens" | 直接查看 arXiv HTML v1 的 PNG |
| 14 | 图 2(a) 120 个格值（§6-E7）逐格一致；1e17 近优格 5 个、1e20 近优格 20 个、最小 2.474 | PNG 目读 + Python 阈值计算（3.6942、2.4802） |
| 15 | SFT 150 万（120 万 = 31.2/46.6/22.2%，安全 30 万）、4/2 epoch、1e-5/5e-6；DPO 1 epoch、5e-6、512；复读 2.0%→1.4% | `dsllm` §4、§5.1.2、Table 12 |
| 16 | MT-Bench 8.35 / 8.76 / GPT-3.5 8.39 / GPT-4 9.26；AlignBench 6.43 / 6.69 / 6.08；Table 9 17.5/58/55.5 | `dsllm` Table 7/8/9 |
| 17 | README 匈牙利 "exceptional score of 65"、more_results.md 为 58；7B Chat GSM8K 62.6；系统提示 "DO NOT RECOMMEND" | README 与 `evaluation/more_results.md` 原文 |
| 18 | 发布时间：HF 7B/67B base createdAt 2023-11-29T03:08:22Z / 03:23:47Z；DeepSeek-LLM 仓库 2023-11-29T11:07:49Z；Coder 仓库 2023-10-20，首提交 `init project` 2023-11-02T14:07:09Z；coder-6.7b-base HF 2023-10-23；v1.5 HF 2024-01-25T15:39:50Z；论文 v1 2024-01-05 18:59:13 UTC，Coder v1 2024-01-25、v2 2024-01-26 | HF API、GitHub API、arXiv abs 页 Submission history |
| 19 | base config bos/eos 1/2、generation_config 与 chat config 100000/100001、initializer_range 0.02、FFN 11008/22016、rope_theta 10000、tie=false | 各 raw 文件 |
| 20 | Coder：87%/10%/3%、32.8%、798 GB/603M、Table 1 恰 87 种语言（逐行求和 797.88 GB、99.93%）、README 列 89 种 | `dscoder` v2 §2、Table 1；README 语言列表计数 |
| 21 | Coder 词表 32,000（论文）vs 32256（config）；tokenizer 22 个追加 token，fim▁hole/begin/end = 32015/32016/32017 | `dscoder` §3.2；`deepseek-coder-6.7b-base/tokenizer.json` 与 config |
| 22 | Coder v1 模板两个 `fim_hole`、v2 改为 `fim_start`；v1→v2 另外只改了 LeetCode 链接与 Table 7 的 % 号 | v1 与 v2 全文 diff（13 行差异） |
| 23 | Coder §3.6 1000 步 × 512 × 16K（= 8.39e9）vs README "additional 200B tokens"、"1.8T tokens and a 4K window"；v1.5 §5 "2 trillion" vs §6 "2 billion" | `dscoder` v2 原文；Coder README |
| 24 | Coder Table 6 Mean 列 4 行不是简单平均（SantaCoder 60.0 印 69.0 等） | Python 重算 |
| 25 | Coder FIM 图 3 的 SVG 数据（F10）、颜色图例（fim_0 绿、fim_0.5 红、fim_1.0 蓝、msp_0.5 青）、fim_1.0 在 5000 步 0.659 后降到 0.604 | 读 `fim.svg` 的路径坐标与网格线定标（与 §5.2 所记定标常数相同）；图例字形解码；重新反解的值与 F10 相差至多 0.001（舍入） |
| 26 | Coder 精确参数量 1,346,471,936 / 6,740,512,768 / 33,342,991,360；KV cache 192/512/248/16 KiB；Table 2 超参数；R24、R35、R36 | Python 重算；HF API safetensors.total；`dscoder` Table 2/3/5 |

**改动（均已在正文相应位置落实）**

1. **§7-E24 与 §6-E12 答案：更正“指标不同”的说法。** 原文说 Bavarian 与 DeepSeek-Coder 的分歧部分来自“指标（损失 vs pass@1）不同”。实际上 Bavarian 也测了 HumanEval 通过率（`fim` §2.1 "For code, we measure the pass rates on HumanEval"），FIM-rate 消融写道 "For HumanEval, the left plot in Figure 5 shows all models irrespective of FIM rate have a similar performance"。两篇在同一指标上结论相反，剩下的差异是模型规模、数据与训练长度。
2. **§5.1 图 3(a) 纵轴范围** 由 $2^{15}$–$2^{27}$ 改为 $2^{14}$–$2^{27}$（像素定标：$2^{24}$ 网格线 y=695、每个 $\log_2$ 单位 188 px，底轴 y=2571 对应 $2^{14.0}$），并补上图例原文。
3. **新增 §8-23，并在 §5.1、F3、R20、E13 加注：图 3(a) 灰点超出正文说的拟合范围。** 像素反解显示图 3(a) 灰点延伸到约 1.6e20，而正文与图 3(b) 都在约 2e19 以内。这不推翻正文引文，但“拟合只到 2e19、外推 4.3 万倍”需要注明是“按正文”。
4. **§7-E3 与 §8-8 加入独立证据。** 67B 星号的像素横坐标 23.937 与真实结构 M 的预测 23.937 一致，与式 (2) 版 23.998 不符；图例原文 "67B GQA 2T Token"。
5. **§7-E6 与 §3.1 补直接来源。** batch size 是序列数这一点原本只是推算，现在补上 Table 2 表头原文 "Sequence Batch Size"。
6. **§7-E15 精确化 IFEval 59.1 的出处。** README 正文里没有 59.1 这个数字（只有图片和说明句），文字版在 `evaluation/more_results.md`。
7. **§7-E14 补引言原句** "Following SFT and DPO, the DeepSeek 67B chat model outperforms GPT-3.5 ..."，表明论文在引言里自己加了 DPO 前提。
8. **§7-E1 措辞。** 注明 HF createdAt 只证明仓库创建，“公开”由 GitHub 仓库与当天 README 提交佐证。
9. **§9 `llama1` 卡片。** 原写“4M batch、1.4T token”，更正为 6.7B/13B 训练 1.0T、32.5B/65.2B 训练 1.4T（`llama1` Table 2）。

**本轮未能独立复核（保持原标注）**：图 1、图 5、图 7 的目测数值；R39 笔者拟合；LLaMA-2 70B 的层数/FFN；Coder 16K 阶段真实 token 数；GQA 引文里 "GQA-H" 的大小写（arXiv HTML 渲染为 "GQA-h"，PDF 未核）。

---

## 11. 补充核实（写课时追加，2026-09-24）

> 写第 6 课时为课文新用到、而上文没有逐字记录的事实。引文均用 `.cursor/skills/academic-research/scripts/arxiv.py grep/text` 从 arXiv HTML 重新抓取，逐字摘录；派生数字在 Python 中重算。

### 11.1 新核实的引文

| # | 用在课文哪里 | 来源与位置 | 原文 |
|---|---|---|---|
| S1 | §1 动机 | `dsllm` §3 引言 | "early works (Kaplan et al., 2020; Hoffmann et al., 2022) on the optimal model/data scaling-up allocation strategy have shown varying conclusions, raising doubts about the general applicability of scaling laws. Moreover, these studies often lacked a complete description of hyperparameter settings, leaving it uncertain whether models under different compute budgets reached optimal performance." |
| S2 | §1 “长期主义” | `dsllm` 摘要 | "Guided by the scaling laws, we introduce DeepSeek LLM, a project dedicated to advancing open-source language models with a long-term perspective." |
| S3 | §2 过滤 | `dsllm` §2.1 | "In the filtering stage, we focus on developing robust criteria for document quality assessment. This involves a detailed analysis incorporating both linguistic and semantic evaluations, providing a view of data quality from individual and global perspectives." |
| S4 | §1、§7 外推被证实 | `dsllm` 图 5 标题 | "The blue stars represent DeepSeek LLM 7B and 67B. Their performance is well-predicted by the scaling curve." 图 6 标题："N1, N2, and M represent the non-embedding parameters, complete parameters, and non-embedding FLOPs/token of the model, respectively." |
| S5 | §4 训练基建 | `dsllm` §2.4 | "We use an efficient and light-weight training framework named HAI-LLM (High-flyer, 2023)"；"we convert bf16 logits to fp32 precision on the fly in the cross-entropy CUDA kernel (instead of converting it beforehand in HBM), calculate the corresponding bf16 gradient, and overwrite logits with its gradient." |
| S6 | §8 复读与两阶段 SFT | `dsllm` §4、§5.1.2、表 12 标题 | "This can be attributed to the fact that math SFT data occasionally includes similar patterns in reasoning. Consequently, weaker models struggle to grasp such reasoning patterns, resulting in repetitive responses. To tackle the problem, we tried two-stage fine-tuning and DPO ..., both of which could almost keep the benchmark score and reduce the repetition significantly."；"In the 7B model fine-tuning, we initially fine-tune the model using all data. Subsequently, a second stage is introduced, which excludes math and code data."；表 12："The repetition ratio is computed when the temperature is 0." |
| S7 | §8 DPO 公式 | `dpo` 式 (7) | $\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}})=-\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_{\theta}(y_{w}\mid x)}{\pi_{\text{ref}}(y_{w}\mid x)}-\beta\log\frac{\pi_{\theta}(y_{l}\mid x)}{\pi_{\text{ref}}(y_{l}\mid x)}\right)\right]$。DeepSeek LLM 没有报告 $\beta$。 |
| S8 | §7 参数化损失推导 | `chinchilla` 式 (2)、(4)（§3.3） | $\hat{L}(N,D)\triangleq E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$；"$N_{opt}(C)=G(C/6)^{a}$, $D_{opt}(C)=G^{-1}(C/6)^{b}$, where $G=(\alpha A/\beta B)^{1/(\alpha+\beta)}$, $a=\frac{\beta}{\alpha+\beta}$, and $b=\frac{\alpha}{\alpha+\beta}$." 课文用它做“数据质量 → $\beta/\alpha$ → $a$”的**笔者解读**，DeepSeek 没有拟合这种形式。 |
| S9 | §8 仓库级排序 | `dscoder` 算法 1 及其说明 | 算法第 18 行 `subgraphs ← getDisconnectedSubgraphs(graphs)`；第 23 行 `file ← argmin({inDegree[file] | file ∈ subgraph and file ∉ results})`；说明："The algorithm concludes by returning a list of these sorted sequences, and each sequence’s files are concatenated to form a single training sample." 即每个连通子图单独成一条样本（图 6.14 按此绘制）。 |
| S10 | §8 FIM 消融里的 MSP | `dscoder` §3.1.2 | "The Masked Span Prediction (MSP) strategy, initially introduced in T5 (Raffel et al., 2023), conceals multiple text spans and trains the model to reconstruct these segments."；PSM/SPM："In the PSM mode, the training corpus is organized in the sequence of Prefix, Suffix, Middle ... the SPM mode arranges the segments as Suffix, Prefix, Middle" |
| S11 | §8 FIM 是数据变换 | `fim` §1 | "autoregressive language models can learn to infill text after we apply a straightforward transformation to the dataset, which simply moves a span of text from the middle of a document to its end." |
| S12 | §8 Coder-V2 的路线 | 转引 `lesson-10-doc.md` C36（`coderv2` 摘要） | "Specifically, DeepSeek-Coder-V2 is further pre-trained from an intermediate checkpoint of DeepSeek-V2 with additional 6 trillion tokens." |

### 11.2 课文新增的派生数字（笔者计算，Python 复算）

| 量 | 表达式 | 值 | 课文位置 |
|---|---|---|---|
| 67B 每层注意力参数 | MHA $4d^2$；GQA-8 $2d^2+2d\cdot1024$ | 268,435,456；150,994,944；每层省 117,440,512（43.75%）；95 层省 11,156,848,640 | §3 |
| 每层参数（$d_{ff}$=22016，含 2 个 RMSNorm） | MHA / GQA-8 | 809,517,056 / 692,076,544 | §3、图 6.5 |
| 80 层 MHA 的参数量折合 GQA-8 层数 | `80*809517056/692076544` | 93.58 | §3、图 6.5 |
| 80 层 GQA-8 要达到 67B 非嵌入参数所需 FFN 中间维 | 解 $80(2d^2+2d\cdot1024+3d\,d_{ff}+2d)=65{,}747{,}271{,}680$ | 27,296（×1.24） | §3（假想方案） |
| 7B 词表多出的参数折合层数 | `(102400-32000)*4096*2/202383360` | 576,716,800 → 2.85 层 | §3（笔者解释） |
| cosine（终值 10%）在 1T 处 | $0.1+0.45(1+\cos(\pi\tau/D))$ | $D$=2T：0.55；$D$=4T：0.868 | §4 例题 |
| 复用节省的一般式 | $p(1-D_{\max}/\sum D_i)$ | $D,2D,4D$：34.3%（p=0.8）/ 25.7%（0.6）；$D..4D$：48% / 54%（p=0.9） | §4、练习 8 |
| 一条 4096 序列的 logits | `4096*102400*2`、`*4` 字节 | 0.78 GiB（bf16）、1.5625 GiB（fp32） | §4 例题 |
| 1e17 网格各 batch 的步数 | `5.65e8/2**k` | $2^{15}$ 17,242；$2^{17}$ 4,310；$2^{20}$ 539 | §6 例题、图 6.10 |
| 1e17 网格在 1%、2% 阈值内的格数 | 阈值 3.685×1.01、×1.02 | 19、41 | §6 |
| 指数误差 ±0.02 外推 4.64 个数量级 | `10**(0.02*4.64)` | 1.24 | §6 失败模式 |
| 给定 $M$ 的最优数据量 | $D_{opt}\propto M^{1/0.5243-1}$ | $M^{0.907}$；模型 ×10 → 数据 ×8.1 | §7 |
| 由 $a$ 反推 $\beta/\alpha$（玩具模型） | $a/(1-a)$ | 0.82、1.10、1.37 | §7 |
| $C=10^{22}$ 的超参数 | 式 (1) | $\eta=5.54\times10^{-4}$；$B=4.59\times10^6$ token ≈ 1120 条 | 练习 6 |
| $C=10^{24}$ 的最优分配 | 式 (4) | $M=6.57\times10^{11}$，$D=1.52\times10^{12}$，$D/(M/6)=13.9$ | 练习 7 |
| LSH 示例（$b$=20,$r$=5 与 $b$=25,$r$=4） | $1-(1-J^r)^b$ | 前者 J=0.8/0.5/0.2：0.9996/0.47/0.0064，开关约 0.55；后者 J=0.8/0.5/0.3：1.000/0.80/0.18，开关约 0.45 | §2、练习 10（示意参数，非 DeepSeek 的） |
| $6N_1$ 误差 ≤ 10% 的条件 | $6d/(6d+T)\ge0.9$ | $d\ge1.5T$；Kaplan 记法下 $d\ge0.75n_{ctx}$ | 练习 11 |
| MinHash 玩具例（图 6.4，Teach 复审追加） | “小猫坐在垫子上”/“小猫坐在毯子上”按 2 字切片 | 交集 4、并集 8，$J=0.5$；$b$=20、$r$=5 时 $P=0.47$ | §2、图 6.4（笔者构造的示意） |
| MinHash 估计的标准差 | $\sqrt{J(1-J)/K}$，$J$=0.5、$K$=100 | 0.05（±2σ 约 0.4–0.6） | §2 |
| 带噪声二次函数的噪声地板（玩具模型） | 引 `math/src/m07-optimization.md` §2 的 (M7.6)：$\approx\eta\sigma^2/(4B)$，弛豫约 $1/(2\eta h)$ 步 | 7B 两次降档后地板为峰值时的 31.6%、10%；M07 数值例 $h$=1、$\eta$=0.01 时 50 步 | §4（笔者解读，非论文分析） |
| 按 $M/6$ 口径的实际 token/等效参数 | $2\times10^{12}/(M/6)$ | 7B 283、67B 27.7（按总参数：289、29.7） | §7 |
| 67B 算力下的最优模型 | $C=8.65\times10^{23}$ 代入式 (4) | $M_{opt}=6.09\times10^{11}$（67B 的 1.41 倍），$D_{opt}=1.42\times10^{12}$ | 练习 7 |
| 指数误差 ±0.02 的不对称 | $10^{\pm0.0928}$ | +24% / −19% | §6 失败模式 |
| $10^{17}$ 网格近优 batch 的跨度 | $2^{18}/2^{16.5}$ | 2.83 倍 | §6 失败模式 |
| 67B 按真实结构的注意力占比（Math 复审追加） | $12LdT/M$，$M$=4.3273e11 | 3.825e10/4.3273e11 = 8.8%（按式 (2) 的 MHA 口径是 1/13 = 7.7%）；$6N_1/M$ 真实结构 0.912 | §5 例题、练习 11 |
| 指数误差的杠杆臂取拟合中心（Math 复审追加） | 设 1e17–2e19 在对数轴均匀，中心 $10^{18.15}\approx1.4\times10^{18}$；到 67B 5.79 个数量级 | $10^{0.02\times5.79}=1.31$（以右端 2e19 为支点是 1.24） | §6 失败模式 |
| 1e17 网格最下面两行的损失范围（Math 复审追加） | §6-E7 数据 | $2^{19.5}$ 行 4.078–4.764；$2^{20}$ 行 4.614–5.152 | §6 例题 |
| 去重率逐段增量（Math 复审追加） | 表 1 相邻差 | 24.5、9.0、14.2、5.8、0.6、5.3、8.2 个百分点（总体放缓，但不逐段单调） | §2 例题 |
| 80 层 MHA 换成 GQA-8 省下的参数（Math 复审 2） | `80*117440512`；`80*809517056+8192`；`/692076544` | 93.95 亿 ≈ 13.6 层 GQA-8；LLaMA-1 65B 形状的非嵌入参数 647.6 亿；95 层 GQA-8（657.5 亿）比它多 9.9 亿 ≈ 1.4 层。**更正**：课文原把“95 层共省 111.6 亿”与“≈ 14 层”并列，两者口径不同（111.6 亿 ≈ 16.1 层） | §3、§8 总账 |
| 因果折扣下 $M$ 的多算比例（Math 复审 2） | 实际 $72Ld^2+6LdT$，多算 $T/(12d+T)$，$T$=4096 | $d$=512：40%；$d$=8192：4%。**更正**：课文原说 $M$ 的误差比例“在小模型和大模型上大致相同”，不成立 | §5 warn、练习 1 |
| 表 3 第一行 $N_2$（Math 复审 2） | `25165824+102400*512` | 77,594,624 = 77.59M（印 77.6M；课文原写 77.60M） | §5 |
| 3e20 最优点的 $M/6$（Math 复审 2） | `9.3415e9/6` | 1.56e9 ≈ 16 亿（课文原写 15 亿） | §7 例题 |
| 67B 实际点相对同算力最优数据量（Math 复审 2） | $D_{opt}(8.6545\times10^{23})$ 由式 (4) | 1.421e12；$2/1.421=1.41$（相对自身 $M$ 的最优量是 1.92） | 图 6.13 说明 |

### 11.3 事实核查复审（Facts 复审，2026-09-24）

> 逐条核对课文中的数字、引文与出处。引文用 `arxiv.py grep/text` 从 arXiv HTML 重新抓取（`dsllm` v1 全文存于会话 scratchpad），制品用 WebFetch 读 raw 文件。

| # | 用在课文哪里 | 来源与位置 | 原文 / 结果 |
|---|---|---|---|
| F1 | §8 “两个不做的决定” | `dsllm` §5.5 Multi-Choice Question 段与表 13 | "During the alignment stage, we tested adding 20 million Chinese multi-choice questions and obtained the performance as shown in Table 13."表 13（7B Chat → 7B Chat + MC）：MMLU 49.4→60.9，C-Eval 47.0→71.3，CMMLU 49.7→73.8，TriviaQA 57.9→57.9，ChineseQA 75.0→74.4。**更正**：课文原写“在预训练和微调中加入 2000 万道中文选择题”，实际只在对齐阶段、只对 7B Chat 做了这个实验；“两个阶段都不用”是据此做出的决定。 |
| F2 | §8 两阶段 SFT | `dsllm` §5.5 Staged Fine-Tuning 段 | "the first stage involves fine-tuning with all available data, while the second stage focuses specifically on fine-tuning with conversational data." 课文“第二阶段……只用对话数据”有此出处（§5.1.2 只说 "excludes math and code data"）。 |
| F3 | §8 对齐概述 | `dsllm` 全文 grep "reward" | 对齐部分（§4、§5）没有出现奖励模型；"reward modeling" 只在引言泛泛提到。**更正**：课文原写“没有用奖励模型和 PPO”，偏好标签怎样产生论文未说（§8-9），不能断言没有用奖励模型；改为“没有用 PPO 一类的强化学习（报告也没有提到奖励模型）”。 |
| F4 | §8 DPO 数据 | `dsllm` §4 | "We constructed the preference data for DPO training in terms of helpfulness and harmlessness. For helpfulness data, we collected multilingual prompts, which cover categories including creative writing, question answering, instruction following, and so on. Then we generated responses using our DeepSeek Chat models as response candidates. Similar operations are applied to harmlessness preference data construction." |
| F5 | §2 分词器 | `dsllm` §2.1 | "We also chose to split numbers into individual digits following the approach used in (Touvron et al., 2023a; Touvron et al., 2023b)." **更正**：课文原写“报告没有给理由”，改为“只说沿用 LLaMA 的做法，没有给理由”。 |
| F6 | §1 例题、§7 | `dsllm` §3.3 | 原句 "This finding might also explain the significant differences ..."（"might"）。课文把“数据质量不同，最优分配就不同”写成“报告给出的解释”，改为“报告提出的一个可能的解释”。 |
| F7 | 图 6.13 说明、§7 | `dsllm` 图 4、图 5 标题 | 图 4："The dotted lines in optimal model/data scaling curves represent the power law fitting the smaller model (grey circles)."；图 5："The dotted line represents the power law fitting the smaller model (grey circles)."；§3.2："we fitted the loss scaling curve according to compute budget C and optimal generalization error"。课文原说原图实测点“几乎完美地排在直线上”，这是读图判断、无记录，改为按标题描述（灰点 = 实测最优点，虚线 = 幂律拟合）。 |
| F8 | §8 FIM 图说明、练习 4 | `fim` §4.5 与结论 | "we recommend applying FIM transformation at the character level and always including some character-level random spans as it allows the model to generate sensible completion even when the prefix and suffix end in the middle of a token."；§3："We perform this split prior to tokenization, when the document is still a sequence of characters."（核实课文“按字符切以获得 token 被切断时的鲁棒性”。） |
| F9 | §3 加深而非加宽、`llama2` 卡片 | `llama2` 附录 A.2.1 | "We train all models with 150B tokens while keeping a fixed 30B model size."（卡片“在 30B 规模上比较”有出处。） |
| F10 | `gqa` 卡片 | `gqa` §1 | "We show that uptrained GQA achieves quality close to multi-head attention while being almost as fast as multi-query attention." |
| F11 | §3 “80 层 MHA 网络（d=8192、FFN 22016）”、图 6.6 | `huggyllama/llama-65b` 的 `config.json`（Hugging Face 社区转换版，**非 Meta 官方仓库**），2026-09-24 WebFetch | `hidden_size` 8192、`intermediate_size` 22016、`num_hidden_layers` 80、`num_attention_heads` 64、`vocab_size` 32000。按 (6.3)(6.4) 算得总参数 65,285,660,672（论文表 2 印 65.2B，属截断）；6.7B/13B/32.5B 同法得 6.74B/13.0B/32.5B。LLaMA-1 65B 的 FFN 与 DeepSeek 67B 同为 22016，课文把 80 层 MHA 网络称作“LLaMA-1 65B 那样的”成立。 |
| F12 | 课文内链接 | 各课源文件的 `# ` 页标题（排除代码块） | 第 1 课 §2 分词；第 2 课 §5 优化/调度、§6 C≈6ND、§7 Kaplan 与 Chinchilla；第 3 课 §7 DPO；第 4 课 §2 KV cache、§6 MQA/GQA；第 10 课 §8 Coder-V2；M01 §5 数 FLOPs；M04 §3 独立和的方差；M05 §2 交叉熵；M07 §2 噪声地板（数值例 $h$=1、$\eta$ 0.1→0.01、弛豫 50 步）、§4 AdamW、§5 调度；M10 §1 数 FLOPs。课文链接已补上相应 #pN。 |
