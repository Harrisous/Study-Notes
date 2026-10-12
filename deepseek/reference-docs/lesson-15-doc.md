# 第 15 课研究笔记：DeepSeek-V3.1 / V3.2——混合思考、规模化 RL 与智能体

> 本文件是第 15 课 `lesson-15-v3-2-agents` 的核对底稿，不是课文。建立于 2026-09-24。
> 所有引文均为原文英文，逐字照抄（保留原文的大小写、粗体标记与空格；引文中夹带的数学符号是从 arXiv HTML
> 文本层转写的纯文本形式，如 `π_old`、`o_{i,t}`，公式本身以 §2 的 LaTeX 转录为准）。112 条带引号的引文已用脚本
> 对照源文本逐段核对，除数学符号外逐字一致；所有派生数字都在 Python 中复算过，
> 表达式见 §4。凡是没能在一手来源里找到的说法，一律放进 §8，不进课文。
> 写课文时请同时对照 `course-map.md` 第 15 课条目与 `style-guide.md` §2 符号表（本笔记 §2 给出了逐项映射）。
> DSA（稀疏注意力）本身属于第 14 课，这里只在需要时引用其公式编号，不重复核对。

---

## 0. 覆盖范围与所读版本

### 0.1 论文（有 arXiv 编号）

| 键 | 标题 | 作者 | arXiv 编号与所读版本 | 状态 | 日期 |
|---|---|---|---|---|---|
| **V32** | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models | DeepSeek-AI（Aixin Liu 等 263 人：arXiv 元数据 264 个作者条目，其中第一条是机构名 “DeepSeek-AI”；附录 E 作者表 Research & Engineering 212 + Data Annotation 27 + Business & Compliance 24 = 263） | 2512.02556 **v1**（读了 LaTeXML HTML v1、arXiv PDF v1，并与 Hugging Face 仓库内 `assets/paper.pdf` 做了逐句 diff，见 §7 E17） | 预印本，截至 2026-09-24 arXiv 只有 v1 | arXiv v1 2025-12-02；HF PDF 2025-12-01 |
| **MV2** | DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning | Zhihong Shao\*, Yuxiang Luo\*, Chengda Lu\*†, Z.Z. Ren\*, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, Xiaokang Zhang（\* 核心贡献者，† 实习期间完成） | 2511.22570 **v1**（HTML + PDF v1；图中数值从 PDF 文本层提取） | 预印本，截至 2026-09-24 只有 v1 | 2025-11-27 |
| **V3** | DeepSeek-V3 Technical Report | DeepSeek-AI | 2412.19437 **v2** | 预印本 | 2024-12-27（v1 日期） |
| **DSM** | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Shao, Wang, Zhu, Xu, Song, Bi 等 | 2402.03300 **v3** | 预印本 | 2024-02-05（v1 日期） |
| **TM** | On a few pitfalls in KL divergence gradient estimation for RL | Yunhao Tang, Rémi Munos | 2506.09477 **v1** | 预印本 | 2025-06-11 |
| **GSPO** | Group Sequence Policy Optimization | Chujie Zheng 等（Qwen 团队） | 2507.18071 **v2** | 预印本 | 2025-07-24（v1 日期） |
| **MX** | Microscaling Data Formats for Deep Learning | Bita Darvish Rouhani 等 | 2310.10537 **v3** | 预印本 | 2023-10-16（v1 日期） |

### 0.2 制品（无论文，`kind: artefact`）

| 键 | 内容 | URL | 读取方式 / 日期 |
|---|---|---|---|
| **HF-V31** | DeepSeek-V3.1 模型卡 + `config.json` + `assets/chat_template.jinja` + `tokenizer.json` + `generation_config.json` | https://huggingface.co/deepseek-ai/DeepSeek-V3.1 | raw 文件，2026-09-24；HF API：createdAt 2025-08-21，V3.1-Base createdAt 2025-08-19 |
| **HF-V31T** | DeepSeek-V3.1-Terminus 模型卡 + `config.json` + `inference/config_671B_v3.1.json` + `inference/kernel.py` | https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus | raw 文件；createdAt 2025-09-22 |
| **EXP** | DeepSeek-V3.2-Exp 技术报告 PDF（6 页，GitHub，无 arXiv 编号） | https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf | pypdf 抽文本；HF createdAt 2025-09-29 |
| **HF-V32** | DeepSeek-V3.2 模型卡 + `encoding/encoding_dsv32.py` + 测试输入输出 + `config.json` + `generation_config.json` + `inference/kernel.py`、`inference/model.py` | https://huggingface.co/deepseek-ai/DeepSeek-V3.2 | raw 文件；createdAt 2025-12-01 |
| **HF-V32S** | DeepSeek-V3.2-Speciale 模型卡（README 与 HF-V32 **逐字节相同**，config 也相同） | https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale | createdAt 2025-11-28 |
| **HF-MV2** | DeepSeek-Math-V2 模型卡（Apache-2.0 许可） | https://huggingface.co/deepseek-ai/DeepSeek-Math-V2 | createdAt 2025-11-27 |
| **NEWS** | DeepSeek API 新闻页：news250821（V3.1）、news250922（Terminus）、news250929（V3.2-Exp）、news251201（V3.2） | https://api-docs.deepseek.com/news/news250821 等 | HTML 抽文本 |
| **API-TM** | API 指南 “Thinking Mode”：Wayback 快照 2025-12-09（V3.2 时代）与 2026-09 现行版（V4 时代） | https://api-docs.deepseek.com/guides/thinking_mode ；快照 http://web.archive.org/web/20251209045516/https://api-docs.deepseek.com/guides/thinking_mode | 两版对比见 §7 E15 |
| **DG** | DeepGEMM README（UE8M0 缩放因子格式） | https://github.com/deepseek-ai/DeepGEMM | raw README |
| **PTX** | NVIDIA PTX ISA 文档 “Alternate Floating-Point Data Formats”（ue8m0 定义） | https://docs.nvidia.com/cuda/parallel-thread-execution/ | HTML 抽文本 |
| **JS** | John Schulman, “Approximating KL Divergence”（博客，2020-03-07） | http://joschu.net/blog/kl-approx.html | HTML |
| **CKPT** | 笔者对 V3 / V3.1 / V3.1-Terminus / V3.2 检查点 safetensors 头部与缩放因子张量的实测（HTTP Range 读取，只读几 KB） | HF 各仓库 `model-000NN-of-000163.safetensors` | 2026-09-24，方法见 §4 N32–N33 |
| **IMO25** | IMO 2025 官方个人成绩页（计算奖牌线） | https://www.imo-official.org/results/individual/year/2025/ | 页面内嵌 JSON |
| **IOI25** | IOI 2025 官方统计页与成绩页 | https://stats.ioinformatics.org/olympiads/2025 ，https://stats.ioinformatics.org/results/2025 | HTML |
| **ICPC25** | 第 49 届 ICPC 世界总决赛（巴库）记分牌 | https://worldfinals.icpc.global/scoreboard/2025/finals/ | WebFetch 摘要 |
| **QQ** | 腾讯新闻（转扬子晚报）2025-08-21 报道 UE8M0 “国产芯片”表述（**二手来源**） | https://news.qq.com/rain/a/20250821A07GD100 | WebFetch 摘要 |

### 0.3 发布时间线（全部来自 NEWS 页标题日期或 HF API `createdAt`）

| 日期 | 事件 | 来源 |
|---|---|---|
| 2024-12-25 | DeepSeek-V3 权重上传 HF | HF API createdAt |
| 2025-03-24 | DeepSeek-V3-0324 上传 HF（V32 称 Keep Routing 自此版本起使用）；API 新闻页目录把 “DeepSeek-V3-0324 Release” 标为 2025/03/25 | HF API；V32 §3.1；NEWS 侧栏 |
| 2025-05-28 | DeepSeek-R1-0528 上传 HF（V3.1 模型卡的对比对象） | HF API |
| 2025-08-19 / 08-21 | V3.1-Base 上传 / V3.1 发布（新闻页 “DeepSeek V3.1 Release 2025/08/21”） | HF API；NEWS 250821 |
| 2025-09-22 | V3.1-Terminus（新闻页 “DeepSeek V3.1 Update 2025/09/22”） | NEWS 250922 |
| 2025-09-29 | V3.2-Exp，DSA 首发，API 降价 “50%+” | NEWS 250929 |
| 2025-11-27 | DeepSeekMath-V2（arXiv 与 HF 同日） | MV2；HF API |
| 2025-12-01 | V3.2 与 V3.2-Speciale 发布；Speciale 临时端点至 2025-12-15 15:59 UTC | NEWS 251201 |
| 2025-12-02 | V3.2 论文 arXiv v1 | V32 |

---

## 1. 核心论断与原文引文

编号 C1–C60。每条：一句中文概括 → 原文引文 → 位置。

### A. DeepSeek-V3.1（2025-08）

**C1 一个模型，两种模式，靠聊天模板切换。**
> "DeepSeek-V3.1 is a hybrid model that supports both thinking mode and non-thinking mode."
> "**Hybrid thinking mode**: One model supports both thinking mode and non-thinking mode by changing the chat template."

位置：HF-V31 模型卡 “Introduction”。

**C2 V3.1 的思考模式比 R1-0528 更快、质量相当（厂商自述）。**
> "**Higher thinking efficiency**: DeepSeek-V3.1-Think achieves comparable answer quality to DeepSeek-R1-0528, while responding more quickly."

位置：HF-V31 “Introduction”。

**C3 V3.1-Base 是在 V3 基座上做的两阶段长上下文续训，32K 阶段 630B、128K 阶段 209B token。**
> "DeepSeek-V3.1 is post-trained on the top of DeepSeek-V3.1-Base, which is built upon the original V3 base checkpoint through a two-phase long context extension approach, following the methodology outlined in the original DeepSeek-V3 report. We have expanded our dataset by collecting additional long documents and substantially extending both training phases. The 32K extension phase has been increased 10-fold to 630B tokens, while the 128K extension phase has been extended by 3.3x to 209B tokens."

位置：HF-V31 “Introduction”。新闻页的合计说法：
> "V3.1 Base: 840B tokens continued pretraining for long context extension on top of V3"

位置：NEWS 250821 “Model Update”。（630 + 209 = 839，“840B” 是四舍五入，见 §4 N1–N3。）

**C4 “10 倍”与“3.3 倍”的基数是 V3 原报告的两阶段各 1000 步。**
> "After the pre-training stage, we apply YaRN (Peng et al., 2023a) for context extension and perform two additional training phases, each comprising 1000 steps, to progressively expand the context window from 4K to 32K and then to 128K."
> "In the first phase, the sequence length is set to 32K, and the batch size is 1920."
> "During the second phase, the sequence length is increased to 128K, and the batch size is reduced to 480."

位置：V3 v2 §4.3 “Long Context Extension”。复算：两阶段各 62.91B token（§4 N1）。

**C5 V3.1 在权重和激活上都使用 UE8M0 缩放格式。**
> "Additionally, DeepSeek-V3.1 is trained using the **UE8M0 FP8 scale data format on both model weights and activations** to ensure compatibility with microscaling data formats. Please refer to [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) for more details."

> "**Ensure that FP8 model weights and activations are formatted using the UE8M0 scale format.**"

位置：HF-V31 “Introduction” 与 “How to Run Locally / Usage Recommendations” 第 2 条。

**C6 无辅助损失均衡的偏置必须用 FP32。**（与第 11 课的 $b_i$ 直接对应）
> "**The `mlp.gate.e_score_correction_bias `parameters should be loaded and computed in FP32 precision.**"

位置：HF-V31 Usage Recommendations 第 1 条（原文反引号内在 `bias` 后有一个空格，照抄）。

**C7 V3.1 结构与 V3 相同。**
> "The model structure of DeepSeek-V3.1 is the same as DeepSeek-V3."

位置：HF-V31 “How to Run Locally”。config.json 对照见 §3.2。

**C8 非思考模式的前缀以 `</think>` 结尾；思考模式以 `<think>` 结尾；历史轮次的思考内容被丢弃但保留 `</think>`。**
> "With the given prefix, DeepSeek V3.1 generates responses to queries in non-thinking mode. Unlike DeepSeek V3,  it introduces an additional token `</think>`."

> "The multi-turn template is the same with non-thinking multi-turn chat template. It means the thinking token in the last turn will be dropped but the `</think>` is retained in every turn of context."

位置：HF-V31 “Chat Template”（第一句原文 “V3,” 后有两个空格，照抄）。模板字符串见 §3.8。

**C9 V3.1 的工具调用在非思考模式下进行；搜索智能体另有思考模式专用格式。**
> "Toolcall is supported in non-thinking mode."
> "We design a specific format for searching toolcall in thinking mode, to support search agent."

位置：HF-V31 “ToolCall”、“Search-Agent”。**这是 V3.1 与 V3.2 的关键差别**（对照 C30）。

**C10 API 映射：`deepseek-chat` = 非思考，`deepseek-reasoner` = 思考，均为 128K。**
> "deepseek-chat → non-thinking mode"
> "deepseek-reasoner → thinking mode"
> "128K context for both"

位置：NEWS 250821 “API Update”（原文每行前有 emoji，省略）。

### B. V3.1-Terminus 与 V3.2-Exp（2025-09）

**C11 Terminus 修语言一致性与智能体能力。**
> "Language consistency: Reducing instances of mixed Chinese-English text and occasional abnormal characters;"
> "Agent capabilities: Further optimizing the performance of the Code Agent and Search Agent."

位置：HF-V31T “Introduction”。

**C12 Terminus 自承 `o_proj` 的缩放因子不符合 UE8M0。**
> "**NOTE: In the current model checkpoint, the parameters of `self_attn.o_proj` do not conform to the UE8M0 FP8 scale data format. This is a known issue and will be corrected in future model releases.**"

位置：HF-V31T “How to Run Locally”。笔者实测：V3.1（原版）与 V3.2 的 `o_proj` 同样不是 2 的幂（§4 N32–N33，§7 E13）。

**C13 V3.2-Exp 的后训练沿用 V3.1-Terminus 的流水线、算法和数据（为了公平评估 DSA）。**
> "In pursuit of a rigorous assessment of the impact of introducing DSA, for DeepSeek-V3.2-Exp, we maintain the same post-training pipeline, algorithm, and data as used for DeepSeek-V3.1-Terminus, which are detailed as follows."

位置：EXP §2.2。推论（笔者）：专家蒸馏 + 混合 RL 的流水线至迟在 V3.1-Terminus 就已使用。

**C14 与 V3.1-Terminus 相比，V3.2 唯一的架构改动是 DSA。**
> "DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp. Compared with DeepSeek-V3.1-Terminus, the last version of DeepSeek-V3.1, the only architectural modification of DeepSeek-V3.2 is the introduction of DeepSeek Sparse Attention (DSA) through continued training."

位置：V32 §2.1 首段。

### C. V3.2 的动机与后训练总体

**C15 三个短板：vanilla attention 的效率、后训练算力不足、智能体泛化与指令遵循。**
> "First, architecturally, the predominant reliance on vanilla attention (Vaswani et al., 2017) mechanisms severely constrains efficiency for long sequences."
> "Second, regarding resource allocation, open-source models suffer from insufficient computational investment during the post-training phase, limiting their performance on hard tasks."
> "Finally, in the context of AI agents, open-source models demonstrate a marked lag in generalization and instruction-following capabilities compared to their proprietary counterparts"

位置：V32 §1 第 2 段。

**C16 后训练算力预算超过预训练成本的 10%（两处措辞不同，都要照抄）。**
> "Notably, this framework allocates a post-training computational budget exceeding 10% of the pre-training cost, unlocking advanced capabilities."

位置：V32 §1 第 3 段。

> "Over recent months, we have observed consistent performance improvements correlating with extended RL training budget, which already exceeds 10% of the pre-training cost. We hypothesize that reasoning capabilities could be further enhanced with additional computational budget allocation."

位置：V32 §4.1 表 2 之后第 1 段。注意：是 “cost”，不是 “FLOPs”，也没有给出绝对值。

**C17 专家蒸馏：同一基座分别训练 6 个领域专家（外加写作与通用问答），每个专家都用大规模 RL 训练。**
> "For each task, we initially develop a specialized model dedicated exclusively to that particular domain, with all specialist models being fine-tuned from the same pre-trained DeepSeek-V3.2 base checkpoint. In addition to writing tasks and general question-answering, our framework encompasses six specialized domains: mathematics, programming, general logical reasoning, general agentic tasks, agentic coding, and agentic search, with all the domains supporting both thinking and non-thinking modes. Each specialist is trained with large-scale Reinforcement Learning (RL) computing."

位置：V32 §3 “Specialist Distillation”。**版本漂移**：EXP §2.2 写的是 “five specialized domains: mathematics, competitive programming, general logical reasoning, agentic coding, and agentic search”（见 §7 E9）。

**C18 蒸馏数据训练出的模型只比专家略差，差距被后续 RL 抹平；思考/非思考数据由不同模型生成。**
> "Furthermore, we employ different models to generate training data for long chain-of-thought reasoning (thinking mode) and direct response generation (non-thinking mode)."
> "Experimental results demonstrate that models trained on the distilled data achieve performance levels only marginally below those of domain-specific specialists, with the performance gap being effectively eliminated through subsequent RL training."

位置：V32 §3 “Specialist Distillation”（EXP §2.2 措辞相同）。

**C19 混合 RL：推理、智能体、人类对齐合并为一个 RL 阶段；奖励构成。**
> "For DeepSeek-V3.2, we still adopt Group Relative Policy Optimization (GRPO) (Shao et al., 2024; DeepSeek-AI, 2025) as the RL training algorithm. As DeepSeek-V3.2-Exp, we merge reasoning, agent, and human alignment training into one RL stage. This approach effectively balances performance across diverse domains while circumventing the catastrophic forgetting issues commonly associated with multi-stage training paradigms. For reasoning and agent tasks, we employ rule-based outcome reward, length penalty, and language consistency reward. For general tasks, we employ a generative reward model where each prompt has its own rubrics for evaluation."

位置：V32 §3 “Mixed RL Training”。EXP 的对应句更明确地说出了与过去的区别：
> "Unlike in previous DeepSeek models, which are trained with multi-stage reinforcement learning, we merge reasoning, agent, and human alignment training into one RL stage."
> "Our reward design carefully balances two key trade-offs: (1) length versus accuracy and (2) language consistency versus accuracy."

位置：EXP §2.2。

**C20 V3.2 与 Speciale 的区别：Speciale 只用推理数据、长度惩罚更小、并入 DeepSeekMath-V2 的数据与奖励。**
> "DeepSeek-V3.2 integrates reasoning, agent, and human alignment data distilled from specialists, undergoing thousands of steps of continued RL training to reach the final checkpoints. To investigate the potential of extended thinking, we also developed an experimental variant, DeepSeek-V3.2-Speciale. This model was trained exclusively on reasoning data with a reduced length penalty during RL. Additionally, we incorporated the dataset and reward method from DeepSeekMath-V2 (Shao et al., 2025) to enhance capabilities in mathematical proofs."

位置：V32 §3 “DeepSeek-V3.2 and DeepSeek-V3.2-Speciale”。

### D. 规模化 GRPO（V32 §3.1，本课重点）

**C21 GRPO 目标与优势定义（V3.2 版本的优势只减均值，不除标准差）。**
> "Â_{i,t} is the advantage of o_{i,t} which is estimated by normalizing the outcome reward within each group. Specifically, a set of reward models are used to score an outcome reward R_i for each output o_i in the group, yielding G rewards R={R_1,⋯,R_G} respectively. The advantage of o_{i,t} is calculated by subtracting the average reward of the group from the reward of output o_i, i.e., Â_{i,t}=R_i−mean(R)."

位置：V32 §3.1，式 (5)(6) 之后（引文中的数学符号按 HTML 文本层转写）。对照 DSM §4.1.2：
> "Subsequently, these rewards are normalized by subtracting the group average and dividing by the group standard deviation."

位置：DSM §4.1.2。差异见 §2.1 与 §7 E2。

**C22 无偏 KL 估计：用重要性比率修正 K3。**
> "Given o_{i,t} is sampled from the old policy π_old(⋅|q,o_{i,<t}), we correct the K3 estimator (Schulman, 2020) to obtain an unbiased KL estimate using the importance-sampling ratio between the current policy π_θ and the old policy π_old ."

位置：V32 §3.1 “Unbiased KL Estimate”，紧接式 (7)。

**C23 原始 K3 的梯度在 $\pi_\theta\ll\pi_{ref}$ 时给出无界的大权重。**
> "As a direct result of this adjustment, the gradient of this KL estimator becomes unbiased, which eliminates systematic estimation errors, thereby facilitating stable convergence. This contrasts sharply with the original K3 estimator, particularly when the sampled tokens have substantially lower probabilities under the current policy than the reference policy, i.e., π_θ≪π_ref. In such cases, the gradient of the K3 estimator assigns disproportionately large, unbounded weights to maximize the likelihood of these tokens, resulting in noisy gradient updates that accumulate to degrade sample quality in subsequent iterations and lead to unstable training dynamics."

位置：V32 §3.1，式 (7) 之后。独立佐证：TM 式 (3) 证明对 k3 求导得到的是 $\nabla\mathrm{KL}(\pi_{ref},\pi)$ 而非 $\nabla\mathrm{KL}(\pi,\pi_{ref})$（见 C24、§2.2）。

**C24（旁证）对 K3 直接求导，得到的是方向反过来的 KL 的梯度。**
> "Differentiating the variance reduced KL estimate, we obtain a gradient estimate with non-zero expectation."
> "Very importantly, the positions of the two distributions are reversed compared to the target KL divergence"

位置：TM §3 “Variance reduced estimate”，式 (3)：$\mathbb{E}[\widehat{\nabla\mathbb{KL}}_{\text{var-reduced}}]=\nabla\mathbb{KL}(\pi_{\text{ref}},\pi)\neq\nabla\mathbb{KL}(\pi,\pi_{\text{ref}})$。TM 的 “variance reduced estimate” 就是 k3（TM §2.1 定义为 $\log\frac{\pi}{\pi_{ref}}+\frac{\pi_{ref}}{\pi}-1$，以控制变量解释；注意 TM 把它引作 “Schulman, 2017”，而 DSM、V32 引作 “Schulman, 2020”，博客页面本身标注 “Posted on 2020/03/07”，课文按 2020 引）。笔者在 5 元 softmax 上数值验证了两个方向（§4 N23）。

**C25 不同领域需要不同强度的 KL；数学可以弱化甚至去掉。**
> "In practice, we find that different domains benefit from varying strengths of KL regularization. For certain domains, such as mathematics, applying a relatively weak KL penalty or even omitting it entirely can yield improved performance."

位置：V32 §3.1 “Unbiased KL Estimate” 段末。

**C26 Off-policy 序列掩码：两种 off-policy 来源、掩码对象、阈值 δ。**
> "To improve the efficiency of RL systems, we typically generate a large batch of rollout data, which is subsequently split into multiple mini-batches for several gradient update steps. This practice inherently introduces off-policy behavior. Additionally, inference frameworks used for efficient data generation are often highly optimized, which may differ in implementation details from training frameworks. Such training-inference inconsistency further exacerbates the degree of off-policyness. To stabilize training and improve tolerance for off-policy updates, we mask negative sequences that introduce significant policy divergence, as measured by the KL divergence between the data-sampling policy π_old and the current policy π_θ."

> "δ is a hyper-parameter that controls the threshold of policy divergence. Note that π_old here denotes the sampling probability directly returned by the inference framework, thus the KL divergence between the old and current policy accounts for both sources of off-policyness mentioned above. It is also worth noting that we only mask sequences with negative advantages."

> "Intuitively, the model benefits the most by learning from its own mistakes, whereas highly off-policy negative samples can be detrimental, potentially misleading or destabilizing the optimization process. We empirically observe that this Off-Policy Sequence Masking operation improves stability in certain training scenarios that would otherwise exhibit instability."

位置：V32 §3.1 “Off-Policy Sequence Masking”，式 (8)(9) 前后。δ 的数值**未给出**（§8 U1）。

**C27 Keep Routing：推理时的专家路由路径在训练时强制复用；自 V3-0324 起使用。**
> "However, discrepancies between inference and training frameworks, compounded by policy updates, can result in inconsistent expert routing during inference and training even for identical inputs. Such inconsistency induces abrupt shifts in the active parameter subspace, which destabilizes optimization and exacerbates off-policy issues. To mitigate this, we preserve the expert routing paths used during sampling in the inference framework and enforce the same routing paths during training, ensuring that identical expert parameters are optimized. This Keep Routing operation was found crucial for RL training stability of MoE models, and has been adopted in our RL training pipeline since DeepSeek-V3-0324."

位置：V32 §3.1 “Keep Routing”。同类技术在 GSPO 中称 “Routing Replay”：
> "Specifically, we cache the activated experts in π_θold and “replay” these routing modes in π_θ when computing the importance ratios"
> "For example, with the 48-layer Qwen3-30B-A3B-Base model, after each RL gradient update and for the same rollout sample, there are roughly 10% of the experts activated under the new policy π_θ that are different from those under the old policy π_θold."
> "its practice of reusing routing modes incurs additional memory and communication overhead and can also limit the actual capacity of the MoE model."

位置：GSPO v2 §5.3。

**C28 Keep Sampling Mask：保存 top-p/top-k 截断掩码并施加到 $\pi_\theta$，使两者动作空间一致。**
> "Top-p and top-k sampling are widely used sampling strategies to enhance the quality of responses generated by LLMs. Employing these strategies in RL training is also advantageous, as it avoids sampling extremely low-probability tokens that would be used as optimization targets. While such truncation preserves sample quality, it introduces a mismatch between the action spaces of π_old and π_θ, which violates the principles of importance sampling and destabilizes training. To address this, we preserve the truncation masks during sampling from π_old and apply them to π_θ during training, ensuring both policies share identical action subspaces. Empirically, we find that combining top-p sampling with the Keep Sampling Mask strategy effectively preserves language consistency during RL training."

位置：V32 §3.1 “Keep Sampling Mask”。是否对 $\pi_\theta$ 在保留集合上重新归一化，原文未说（§8 U4）。

### E. 工具使用中的思考（V32 §3.2）

**C29 R1 式“新一轮就丢弃思考”的做法在工具调用中浪费 token。**
> "We observed that replicating DeepSeek-R1’s strategy—discarding reasoning content upon the arrival of the second round of messages—results in significant token inefficiency. This approach forces the model to redundantly re-reason through the entire problem for each subsequent tool call."

位置：V32 §3.2.1。

**C30 思考上下文管理的两条规则。**
> "Historical reasoning content is discarded only when a new user message is introduced to the conversation. If only tool-related messages (e.g., tool outputs) are appended, the reasoning content is retained throughout the interaction."
> "When reasoning traces are removed, the history of tool calls and their results remains preserved in the context."

位置：V32 §3.2.1 两个要点；图 4 “Thinking retention mechanism in tool-calling scenarios.”。代码实现：HF-V32 `encoding_dsv32.py` 的 `drop_thinking_messages()` 只删除**最后一条 user/developer 消息之前**的 assistant 消息的 `reasoning_content`（§2.5）。新闻页的说法：
> "DeepSeek-V3.2 is our first model to integrate thinking directly into tool-use, and also supports tool-use in both thinking and non-thinking modes."

位置：NEWS 251201 “Thinking in Tool-Use”。

**C31 用 user 消息模拟工具返回的框架（Roo Code、Terminus）享受不到这一机制，推荐用非思考模式。**
> "Notably, certain agent frameworks, such as Roo Code or Terminus, simulate tool interactions via user messages. These frameworks may not fully benefit from our enhanced reasoning persistence due to the context management rules outlined above. Therefore, we recommend utilizing non-thinking models for optimal performance with such architectures."

位置：V32 §3.2.1 末段。

**C32 冷启动：靠系统提示把“推理数据”和“非推理的智能体数据”拼起来。**
> "Given the availability of reasoning data (non-agentic) and non-reasoning agentic data, a straightforward strategy for integrating these two capabilities is through carefully designed prompting."
> "In this manner, although the reasoning in tool-use patterns may lack robustness, the model is occasionally able to generate the desired trajectories, thereby providing a basis for subsequent reinforcement learning stages."

位置：V32 §3.2.2；三种系统提示模板见附录 B 表 6–8（表 8 规定 “with a maximum of 20 code executions”）。注意：arXiv 的 HTML（LaTeXML）渲染把这三处交叉引用都显示成 “Tables 8–8 / Table 8”，PDF 正文是正确的 “Tables 6–8 / Table 6 / Table 7 / Table 8”；引用时以 PDF 为准（§7 E18）。

**C33 V3.2 聊天模板改用新的工具调用格式并引入 `developer` 角色（仅限搜索智能体）。**
> "DeepSeek-V3.2 introduces significant updates to its chat template compared to prior versions. The primary changes involve a revised format for tool calling and the introduction of a "thinking with tools" capability."
> "A new role named `developer` has been introduced in the chat template. This role is dedicated exclusively to search agent scenarios and is designated for no other tasks. The official API does not accept messages assigned to `developer`."
> "This release does not include a Jinja-format chat template."

位置：HF-V32 “Chat Template” 及 “Important Notes”。

### F. 大规模智能体任务合成（V32 §3.2.3，§4.3）

**C34 引言中的总数：超过 1,800 个环境与 85,000 条复杂提示。**
> "Subsequently, we advance to large-scale agentic task synthesis, where we generate over 1,800 distinct environments and 85,000 complex prompts."

位置：V32 §1 第 3 段。新闻页：“1,800+ environments & 85k+ complex instructions”（NEWS 251201）。拆分见 C35 与 §4 N8–N9。

**C35 表 1 四类任务的数量、环境真实与否、提示来源。**
> "Table 1: The description of different agent tasks, including the number of tasks, environment type (real or synthesized), and prompt source (extracted or synthesized)."

| 任务 | number of tasks | environment | prompt |
|---|---|---|---|
| code agent | 24667 | real | extracted |
| search agent | 50275 | real | synthesized |
| general agent | 4417 | synthesized | synthesized |
| code interpreter | 5908 | real | extracted |

位置：V32 表 1（§3.2.3）。

**C36 真实环境 + 非真实用户提示。**
> "For tasks such as search, code engineering, and code interpretation, we employ real-world tools, including actual web search APIs, coding tools, and Jupyter Notebooks. While these RL environments are real, the prompts employed are either extracted from Internet sources or synthetically generated, rather than obtained from actual user interactions."

位置：V32 §3.2.3 首段。

**C37 搜索智能体数据：多智能体流水线，只保留“标准答案对、候选全错”的样本。**
> "A verification agent with search capabilities validates all answers through multiple passes, retaining only samples where the ground-truth is correct and all candidates are verifiably incorrect."

位置：V32 §3.2.3 “Search Agent”。

**C38 代码智能体环境：挖掘数百万 issue–PR 对；成功标准 F2P > 0 且 P2F = 0；覆盖 8 种语言。**
> "An environment is deemed successfully built only when applying the gold patch results in a non-zero count of false-to-positive (F2P) test cases (indicating the issue is fixed) and a zero count of pass-to-fail (P2F) test cases (indicating no regressions)."
> "Using this pipeline, we successfully built tens of thousands of reproducible issue resolution environments spanning multiple programming languages, including Python, Java, JavaScript, TypeScript, C, C++, Go, and PHP."

位置：V32 §3.2.3 “Code Agent”。（原文写作 “false-to-positive”，业界通常叫 fail-to-pass，照抄原文。）

**C39 通用智能体：自动合成 1,827 个环境、4,417 个任务；“难解易验”；pass@100 非零过滤。**
> "To scale up agent environments and tasks in RL, we employ an automatic environment-synthesis agent that synthesizes 1,827 task-oriented environments. These tasks are hard to solve but easy to verify."
> "Following this workflow, we obtain thousands of ⟨environment,tools,task,verifier⟩ tuples. We then perform RL on this dataset using DeepSeek-V3.2 and retain only instances with non-zero pass@100, resulting in 1,827 environments and their corresponding tasks (4,417 in total)."

位置：V32 §3.2.3 “General Agent”。三步流程（数据入库 → 合成工具函数 → 由易到难地提出任务 + Python 解法 + 验证函数，解法只能调用工具）也在该节。

**C40 合成任务够难，且只在合成任务上做 RL 也能迁移到 τ²-bench、MCP-Mark、MCP-Universe。**
> "As shown in Table 5, DeepSeek-V3.2-Exp attains an accuracy of only 12%, while frontier closed-source models achieve at most 62%."
> "As shown in Figure 5, large-scale RL on synthetic data yields substantial improvements over DeepSeek-V3.2-SFT on Tau2Bench, MCP-Mark, and MCP-Universe benchmarks. In contrast, restricting RL to code and search scenarios does not improve performance on these benchmarks, further highlighting the potential of synthetic data."

位置：V32 §4.3，表 5、图 5（表 5 随机抽 50 个合成任务）。

### G. 评测与 Speciale

**C41 评测设置：温度 1.0，上下文 128K，工具评测用思考模式与标准 function call 格式。**
> "Tool-use benchmarks are evaluated using the standard function call format, wherein models are configured to thinking mode."
> "We set the temperature to 1.0, and the context window to 128K tokens."

位置：V32 §4.1。

**C42 V3.2 与 GPT-5-high 在推理上相当、略逊 Gemini-3.0-Pro；成绩受长度约束奖励模型限制。**
> "DeepSeek-V3.2 achieves similar performance with GPT-5-high on reasoning tasks, but is slightly worse than Gemini-3.0-Pro."
> "Notably, the performance of DeepSeek-V3.2 presented herein is constrained by a length constraint reward model; upon removal of the restriction, we observe further improvement in model performance, as detailed in Section 4.2."

位置：V32 §4.1。

**C43 Terminal Bench 2.0 的 46.4 是在 Claude Code 框架下得到的；Terminus 非思考模式 39.3。**
> "Regarding Terminal Bench 2.0, as previously noted, our context management strategy for the ’thinking mode’ is currently incompatible with Terminus; consequently, the reported score of 46.4 was achieved using the Claude Code framework. We also evaluated DeepSeek-V3.2 with Terminus in non-thinking mode, yielding a score of 39.3."

位置：V32 §4.1。

**C44 BrowseComp：约 20%+ 用例超过 128K，所以报告的 67.6 用了上下文管理；不用时 51.4。**
> "Since DeepSeek-V3.2 supports a maximum context length of only 128K, approximately 20%+ of the test cases exceed this limit. To address this, we employ a context management method to derive the final score. For reference, the score is 51.4 without context management."

位置：V32 §4.1。三种策略（Summary / Discard-75% / Discard-all）与触发条件（token 使用量超过上下文窗口的 80%）见 §4.4：
> "we introduce context management employing simple strategies to extend token budgets at test time， when the token usage exceeds 80% of the context window length."
> "Despite its simplicity, Discard-all performs well in both efficiency and scalability, achieving a score of 67.6, comparable to parallel scaling while using significantly fewer steps."

**C45 V3.2 经常冗余自验证，导致轨迹超过 128K（MCP-Mark GitHub、Playwright）。**
> "During our testing, we observed that DeepSeek-V3.2 frequently engages in redundant self-verification, generating excessively long trajectories. This tendency often causes the context length to exceed the 128K limit, particularly in tasks such as MCP-Mark GitHub and Playwright evaluation."

位置：V32 §4.1 工具使用段。

**C46 Speciale 的金牌说法（摘要、引言、§4.2 三处措辞不同，逐一照抄）。**
- 摘要：
> "Notably, our high-compute variant, DeepSeek-V3.2-Speciale, surpasses GPT-5 and exhibits reasoning proficiency on par with Gemini-3.0-Pro, achieving gold-medal performance in both the 2025 International Mathematical Olympiad (IMO) and the International Olympiad in Informatics (IOI)."
- 引言：
> "Notably, with the aim of pushing the boundaries of open models in the reasoning domain, we relaxed the length constraints to develop DeepSeek-V3.2-Speciale. As a result, DeepSeek-V3.2-Speciale achieves performance parity with the leading closed-source system, Gemini-3.0-Pro (DeepMind, 2025a). It shows gold-medal performance in the IOI 2025, ICPC World Final 2025, IMO 2025, and CMO 2025."
- §4.2：
> "Remarkably, as shown in Table 4, this general-purpose model attains gold-medal level performance in the 2025 International Olympiad in Informatics (IOI) and the ICPC World Finals (ICPC WF) without targeted training. Furthermore, by incorporating techniques from Shao et al. (2025), the model excels in complex proof tasks, reaching gold-medal thresholds in the 2025 International Mathematical Olympiad (IMO) and China Mathematical Olympiad (CMO)"
- 表 4 标题：
> "Table 4: Performance of DeepSeek-V3.2-Speciale in top-tier mathematics and coding competitions. For ICPC WF 2025, we report the number of submissions for each successfully solved problem. DeepSeek-V3.2-Speciale ranked 2nd in ICPC WF 2025 and 10th in IOI 2025."

**C47 竞赛评测方法（附录 D）：无工具、无联网、最大生成 128k；IOI 采样 500 个、选思考最长的 50 个提交；IMO/CMO 用生成—验证—修正循环。**
> "For all competitions, the model’s maximum generation length is set to 128k. No tools or internet access are used, and testing strictly adheres to the contest’s time and attempt limits."
> "Specifically, we first sampled 500 candidate solutions for each problem, then applied a multi-stage filtering pipeline."
> "From the remaining valid candidates, we selected the 50 samples with the longest thinking traces for final submission."
> "For the ICPC evaluation, we adapted the same filtering methodology but with a smaller initial sample size. We generated 32 candidate solutions per problem and applied the identical filtering criteria to select submissions."
> "In the IMO and CMO tasks, we employ a generate-verify-refine loop. The model iteratively improves its solution until it achieves a perfect self-evaluation or hits the maximum revision cap, identical to the process in Shao et al. (2025)."

位置：V32 附录 D。

**C48 Speciale 的代价：token 效率明显差于 Gemini-3.0-Pro；官方 V3.2 因此加了更严格的 token 约束。**
> "However, the token efficiency of DeepSeek-V3.2-Speciale remains significantly inferior to that of Gemini-3.0-Pro. To mitigate deployment costs and latency, we imposed stricter token constraints during the training of the official DeepSeek-V3.2, aiming to optimize the trade-off between performance and cost."

位置：V32 §4.2。另：
> "Please note that the DeepSeek-V3.2-Speciale variant is designed exclusively for deep reasoning tasks and does not support the tool-calling functionality."

位置：HF-V32S Usage Recommendations 第 2 条。
> "V3.2-Speciale: Served via a temporary endpoint: base_url="https://api.deepseek.com/v3.2_speciale_expires_on_20251215". Same pricing as V3.2, no tool calls, available until Dec 15th, 2025, 15:59 (UTC Time)."

位置：NEWS 251201 “API Update”。

**C49 局限性（作者自述）：世界知识广度、token 效率、复杂任务。**
> "First, due to fewer total training FLOPs, the breadth of world knowledge in DeepSeek-V3.2 still lags behind that of leading proprietary models."
> "Second, token efficiency remains a challenge; DeepSeek-V3.2 typically requires longer generation trajectories (i.e., more tokens) to match the output quality of models like Gemini-3.0-Pro."

位置：V32 §5 第 2 段。

**C50 非思考模式在智能体任务上“稍差但仍有竞争力”。**
> "The performance of non-thinking mode is slightly worse than the thinking mode, but still competitive."

位置：V32 附录 C 表 9 之后。

### H. DeepSeekMath-V2（MV2）

**C51 动机：答案对不等于推理对；定理证明没有最终答案可比。**
> "First, it serves as an unreliable proxy for reasoning correctness – a model can arrive at the correct answer through flawed logic or fortunate errors. Second, it is inapplicable to theorem proving tasks, where problems may not require producing numerical final answers and rigorous derivation is the primary objective."

位置：MV2 §1 第 1 段。

**C52 验证器输出“问题清单 + 三档分数”（1 / 0.5 / 0）。**
> "Specifically, given a problem X and a proof Y, the verifier π_φ(⋅|X,Y,ℐ_v) is designed to produce a proof analysis that first summarizes identified issues (if any) and then assigns a score based on three levels: 1 for complete and rigorous proofs with all logical steps clearly justified; 0.5 for proofs with sound overall logic but minor errors or omitted details; and 0 for fundamentally flawed proofs containing fatal logical errors or critical gaps."

位置：MV2 §2.1.1。冷启动题库 17,503 道（AoPS，优先奥赛、选拔考试、2010 年后需要证明的题）。

**C53 元验证：防止验证器“分数对、理由是编的”。**
> "This creates a critical vulnerability: when evaluating flawed proofs (where s_i<1) during training, the verifier can receive full reward by predicting the correct scores while hallucinating non-existent issues, undermining its trustworthiness."
> "On a validation split of 𝒟_v, the average quality score of the verifier’s proof analyses – as evaluated by the meta-verifier – improved from 0.85 to 0.96, while maintaining the same accuracy in proof score prediction."

位置：MV2 §2.1.2。

**C54 生成器必须先自评；奖励 = 证明分与自评分的加权和，α=0.76，β=0.24。**
> "However, we observed a critical limitation: when prompted to both generate and analyze its own proof in one shot, the generator tends to claim correctness even when the external verifier easily identify flaws."
> "We set α=0.76 and β=0.24."
> "A good strategy to obtain high rewards for the proof generator is to identify and resolve as many issues as possible before finalizing the response."

位置：MV2 §2.2.2，式 (5)(6)。

**C55 生成—验证差距靠“放大验证算力自动标注”维持；最后两轮完全取代人工标注。**
> "To maintain the generation-verification gap as the generator becomes stronger, we propose to scale verification compute to automatically label new hard-to-verify proofs, creating training data to further improve the verifier."
> "In our last two training iterations, this fully automated pipeline replaced human annotation entirely."

位置：MV2 摘要；§2.3 末。$n,m,k$ 的数值未给出（§8 U14）。

**C56 训练：GRPO；每轮先训验证、再以验证器检查点初始化生成器；从第二轮起用拒绝微调合并两种能力。**
> "In each iteration, we first optimized proof verification. The proof generator was then initialized from the verifier checkpoint and optimized for proof generation. Starting from the second iteration, the proof verifier was initialized with a checkpoint that consolidated both verification and generation capabilities from the previous iteration through rejection fine-tuning."

位置：MV2 §3.1。基座：
> "Built on DeepSeek-V3.2-Exp-Base (DeepSeek-AI, 2025), we developed DeepSeekMath-V2"

位置：MV2 §1。

**C57 高算力搜索：64 个候选 × 64 次验证，每轮取最高 64 个、各配 8 条分析，最多 16 轮或通过全部 64 次验证。**
> "Our approach maintains a pool of candidate proofs for each problem, initialized with 64 proof samples with 64 verification analyses generated for each. In each refinement iteration, we select the 64 highest-scoring proofs based on average verification scores and pair each with 8 randomly selected analyses, prioritizing those identifying issues (scores 0 or 0.5)."
> "This process continues for up to 16 iterations or until a proof successfully passes all 64 verification attempts, indicating high confidence in correctness."

位置：MV2 §3.3.3。

**C58 成绩：IMO 2025 解出 5/6，CMO 2024 解出 4 题加 1 题部分分，Putnam 2024 118/120（人类最高 90）。**
> "As shown in Table 1, our approach solved 5 of 6 problems from IMO 2025 and 4 problems plus partial credit on another from CMO 2024, achieving gold medal performance in both pinnacle high-school competitions. On Putnam 2024, the preeminent undergraduate mathematics competition, our model solved 11 of 12 problems completely and the remaining problem with minor errors, scoring 118/120 and surpassing the highest human score of 90."

位置：MV2 §3.3.3，表 1（IMO 2025 83.3%，CMO 2024 73.8%，Putnam 2024 98.3%）。评分者：
> "To validate our results, mathematical experts assessed the highest-scoring proofs."

**C59 IMO-ProofBench：基础集超过 DeepThink（IMO Gold），进阶集接近。**
> "Our approach outperforms DeepMind’s DeepThink (IMO Gold) on the basic set and remains competitive on the advanced set, while substantially outperforming all other baselines."

位置：MV2 §3.3.3，图 3（除 DeepSeekMath-V2 外所有数字取自 Luong et al. 2025；DeepSeekMath-V2 由其自家专家按评分指南评）。

**C60 作者承认最难的 IMO 级题仍然困难；未完全解出的题，生成器通常能指出真实问题。**
> "We observe that the hardest IMO-level problems remain challenging for our model. Notably, for problems not fully solved, our generator typically identifies the genuine issues in its proofs, while fully solved problems pass all 64 verification attempts."

位置：MV2 §3.3.3。

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.0 记号冲突总表（写课文前必读）

| 原文记号 | 出处 | 课程记号（style-guide §2） | 处理建议 |
|---|---|---|---|
| $\pi_{\mathrm{old}}$ | V32 式 (5)–(9) | $\pi_{\theta_{old}}$ | **注意语义不同**：V32 明说它是“推理框架直接返回的采样概率”（C26），不是训练框架用旧参数重算的概率。课文建议写 $\pi_{\theta_{old}}^{\text{infer}}$ 或在首次出现时专门说明。 |
| $r_{i,t}(\theta)$（重要性比率） | V32 式 (6) | 课程 $r$ = 奖励 | **冲突**。课文改写为 $\rho_{i,t}(\theta)$。 |
| $R_i$（结果奖励） | V32 §3.1 | $r_i$ | 课文统一用 $r_i$，并在旁注 “论文记作 $R_i$”。 |
| $M_{i,t}$（二值掩码） | V32 式 (8)(9) | 课程 $M$ = 每 token 非嵌入 FLOPs（DeepSeek LLM） | **冲突**。课文改写为 $m_{i,t}\in\{0,1\}$。 |
| $\delta$ | V32 式 (9) | 无 | 新符号，“off-policy 阈值”。 |
| $\varepsilon,\beta$ | V32 式 (5) | 同 | 无冲突。 |
| $\mathbb{D}_{\mathrm{KL}}(\cdot\Vert\cdot)$ | V32 式 (5)(7) | `\KL` 宏 | 课文用 $\KL(\pi_\theta\,\Vert\,\pi_{ref})$。 |
| $\alpha=0.76,\ \beta=0.24$（奖励权重） | MV2 式 (5) | 课程 $\beta$ = KL 系数 | **冲突**。课文改写为 $w_Y=0.76,\ w_Z=0.24$。 |
| $\pi_\varphi,\pi_\eta,\pi_\theta$ | MV2 | 同 | 分别是验证器、元验证器、生成器；课文首次出现时交代。 |
| $X,Y,Z,s,s',ms$ | MV2 | 无 | 题目、证明、自评、证明分、自评分、元验证分。 |
| $k$（top-k） | V32 §2.1 | 第 14 课已用 | 与 “k3 估计器” 的 k 无关，课文注意区分。 |

### 2.1 GRPO 目标（V32 式 (5)(6)）

原文转录（V32 v1 §3.1）：

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)=
\mathbb{E}_{q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\mathrm{old}}(\cdot|q)}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}
\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\text{clip}\left(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right)-\beta\mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}(o_{i,t})\,\middle\|\,\pi_{\mathrm{ref}}(o_{i,t})\right)\Bigg],
\tag{5}
$$

$$
r_{i,t}(\theta)=\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\mathrm{old}}(o_{i,t}|q,o_{i,<t})}
\tag{6}
$$

优势（正文，无编号）：$\hat{A}_{i,t}=R_{i}-\text{mean}(\boldsymbol{R})$。

课程记号改写：

$$
\mathcal J(\theta)=\E_{q,\{o_i\}\sim\pi_{\theta_{old}}}\Bigg[\frac1G\sum_{i=1}^G\frac1{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(\rho_{i,t}\hat A_{i,t},\ \clip(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big)-\beta\,\KL\big(\pi_\theta\,\|\,\pi_{ref}\big)_{i,t}\Big)\Bigg],\quad \hat A_{i,t}=r_i-\tfrac1G\textstyle\sum_j r_j .
$$

与 DSM（2402.03300 v3）的对照：DSM 式 (3) 结构相同（同样有 $\frac1G\sum_i\frac1{|o_i|}\sum_t$ 与 KL 项），但 §4.1.2 的优势是
$\hat{A}_{i,t}=\widetilde{r}_{i}=\frac{r_{i}-{\rm mean}(\mathbf{r})}{{\rm std}(\mathbf{r})}$。**V32 印出的优势只减均值、不除标准差**。这可能是有意的改动，也可能是正文省略；论文没有解释（§7 E2，§8 U3）。

### 2.2 无偏 KL 估计（V32 式 (7)）

原文转录：

$$
\mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}(o_{i,t})\,\middle\|\,\pi_{\mathrm{ref}}(o_{i,t})\right)=\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\mathrm{old}}(o_{i,t}|q,o_{i,<t})}\left(\frac{\pi_{\mathrm{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-\log\frac{\pi_{\mathrm{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-1\right).
\tag{7}
$$

对照 DSM 式 (4)（原始 k3）：

$$
\mathbb{D}_{KL}\left[\pi_{\theta}||\pi_{ref}\right]=\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-\log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-1,
\tag{DSM 4}
$$
DSM 原文紧接着说 “which is guaranteed to be positive.”；Schulman 博客（JS）的写法是 $k_3=(r-1)-\log r$，其中 $r=p(x)/q(x)$、$x\sim q$。

课程记号：记 $u=\pi_{ref}/\pi_\theta$（单个 token 上的比值），$k_3(u)=u-\log u-1\ge 0$。V32 的修正版是 $\hat{\KL}^{\text{corr}}=\rho\cdot k_3(u)$，$\rho=\pi_\theta/\pi_{\theta_{old}}$。

**性质推导（笔者推导，论文只给结论；供课文 `::: deriv` 使用，已数值验证，§4 N21–N23）**：

1. 值的无偏性。若 $x\sim\pi_\theta$：$\E_{\pi_\theta}[u]=\sum_x\pi_{ref}(x)=1$，$\E_{\pi_\theta}[-\log u]=\KL(\pi_\theta\|\pi_{ref})$，故 $\E_{\pi_\theta}[k_3]=\KL(\pi_\theta\|\pi_{ref})$。
   但样本来自 $\pi_{\theta_{old}}$ 时，需要重要性权重：$\E_{\pi_{\theta_{old}}}[\rho\,k_3]=\sum_x\pi_\theta(x)k_3(x)=\KL(\pi_\theta\|\pi_{ref})$。这就是 “correct the K3 estimator ... using the importance-sampling ratio” 的含义。
2. 梯度。把 $k_3$ 当损失直接求导（样本固定）：$\nabla_\theta k_3=(1-u)\nabla\log\pi_\theta$；在 $\pi_\theta$ 下取期望 $=-\sum_x\pi_{ref}(x)\nabla\log\pi_\theta(x)=\nabla\KL(\pi_{ref}\|\pi_\theta)$——**方向反了**（与 TM 式 (3) 一致）。
   对修正版求导：$\nabla(\rho\,k_3)=\rho\,\nabla\log\pi_\theta\cdot\big[k_3+(1-u)\big]=\rho\,\nabla\log\pi_\theta\cdot\log\frac{\pi_\theta}{\pi_{ref}}$；期望 $=\E_{\pi_\theta}\big[\nabla\log\pi_\theta\cdot\log\frac{\pi_\theta}{\pi_{ref}}\big]=\nabla\KL(\pi_\theta\|\pi_{ref})$（单步分布情形；序列情形的额外项见 TM §4，本课不展开）。
3. 权重对比（对数导数前的系数，梯度下降方向）：原始 k3 是 $1-\pi_{ref}/\pi_\theta$，当 $\pi_\theta\ll\pi_{ref}$ 时约为 $-\pi_{ref}/\pi_\theta$，**无界**；修正版是 $\rho\log(\pi_\theta/\pi_{ref})$，只按对数增长。这正是 C23 所说的 “disproportionately large, unbounded weights”。
4. 一个易错点：在第一个 mini-batch 且训练/推理完全一致时 $\rho=1$，修正版的**数值**与 k3 相同，但 $\nabla\rho=\nabla\log\pi_\theta\neq0$，所以**梯度**不同。实现时 $\pi_{\theta_{old}}$ 必须 detach，$\pi_\theta$ 不能 detach。

### 2.3 Off-policy 序列掩码（V32 式 (8)(9)）

原文转录：

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)=
\mathbb{E}_{q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\mathrm{old}}(\cdot|q)}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}
\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\text{clip}\left(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right)M_{i,t}-\beta\mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}(o_{i,t})\,\middle\|\,\pi_{\mathrm{ref}}(o_{i,t})\right)\Bigg],
\tag{8}
$$

$$
M_{i,t}=\begin{cases}0&{\hat{A}_{i,t}<0,\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\log\frac{\pi_{\mathrm{old}}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}>\delta}\\[4.30554pt]
1&{\text{otherwise},}\end{cases}
\tag{9}
$$

课程记号改写（把求和下标换成 $s$，避免与外层 $t$ 冲突）：

$$
m_{i}=\begin{cases}0,& \hat A_{i}<0\ \text{且}\ \bar D_i>\delta\\ 1,&\text{否则}\end{cases},\qquad
\bar D_i=\frac1{|o_i|}\sum_{s=1}^{|o_i|}\log\frac{\pi_{\theta_{old}}^{\text{infer}}(o_{i,s}\mid q,o_{i,<s})}{\pi_\theta(o_{i,s}\mid q,o_{i,<s})}.
$$

要点（全部可由式 (8)(9) 直接读出）：
- 条件里两项是“且”：优势为负 **且** 平均对数比超过 $\delta$。正优势序列永远不掩（C26 “we only mask sequences with negative advantages”）。
- $\hat A_{i,t}$ 是结果奖励减组均值，同一序列所有 $t$ 相同，$\bar D_i$ 也与 $t$ 无关，所以 $M_{i,t}$ 实际上是**序列级**的（名字里的 “Sequence” 由此而来）。原文式 (9) 用 $t$ 同时作外层下标与求和变量，是记号重载。
- 掩码只乘在裁剪代理项上；式 (8) 中 KL 项在掩码之外，被掩序列的 KL 正则仍然起作用。
- $\bar D_i$ 是 $\KL(\pi_{\theta_{old}}\|\pi_\theta)$ 的单样本（逐 token 平均）蒙特卡洛估计：$\E_{o\sim\pi_{\theta_{old}}}[\log(\pi_{\theta_{old}}/\pi_\theta)]=\KL(\pi_{\theta_{old}}\|\pi_\theta)$。单条序列上它可以是负数（当前策略更偏爱这些 token）。
- 为什么 PPO 裁剪不够（笔者分析，课文标“笔者推测/分析”）：对 $\hat A<0$，$\min(\rho\hat A,\clip(\rho)\hat A)$ 在 $\rho<1-\varepsilon$ 时取常数 $(1-\varepsilon)\hat A$，梯度为零；但 $\rho\in[1-\varepsilon,\infty)$ 的 token 仍有梯度（$\rho>1+\varepsilon$ 时甚至不受上界保护）。序列级掩码一次性去掉整条“离策略太远的负样本”，包括那些 $\rho$ 仍在裁剪区间内的 token。数值例见 §6 X3。

### 2.4 Keep Routing 与 Keep Sampling Mask（笔者形式化，论文无公式）

- Keep Routing：设推理时第 $l$ 层对 token $t$ 选出的专家集合为 $\mathcal E^{\text{infer}}_{l,t}$（top-$K_r$）。训练时计算 $\pi_\theta(o_{i,t}\mid\cdot)$ 用
  $\mathcal E^{\text{train}}_{l,t}:=\mathcal E^{\text{infer}}_{l,t}$，而不是重新做 $\TopK$。门控值 $g_{i,t}$ 是否仍用当前参数重算，原文未说（§8 U5）。
- Keep Sampling Mask：设采样时 top-p 截断后的 token 集合为 $\mathcal S_{i,t}$，则
  $\pi_{\theta_{old}}^{\text{trunc}}(a)=\pi_{\theta_{old}}(a)\mathbb 1[a\in\mathcal S]/\sum_{b\in\mathcal S}\pi_{\theta_{old}}(b)$；训练时对 $\pi_\theta$ 施加同一 $\mathcal S$。是否重新归一化，原文未说（§8 U4）；§6 X6 给出两种算法的数值。

### 2.5 思考上下文管理（代码即规范）

HF-V32 `encoding/encoding_dsv32.py`（2025-12-01 版）的 `drop_thinking_messages()` 规则，用伪公式表示：
对消息序列 $(\mathrm{msg}_1,\dots,\mathrm{msg}_n)$，令 $u^*=\max\{j: \text{role}(\mathrm{msg}_j)\in\{\text{user},\text{developer}\}\}$。对每条 assistant 消息 $\mathrm{msg}_j$：若 $j<u^*$，删去其 `reasoning_content`；若 $j>u^*$，保留。user/system/tool 消息全部保留。（这里不用 $m$，避免与掩码 $m_i$ 混淆。）
核查补充（代码原样）：① 只有 `encode_messages(..., thinking_mode == "thinking", drop_thinking=True)`（`drop_thinking` 默认 True）时才调用 `drop_thinking_messages`，非思考模式不做这一步；② 函数的保留条件是 `role in ["user", "system", "tool"] or idx >= last_user_idx`，`elif role == "assistant"` 才去掉思考——所以位于 $u^*$ **之前**的 `developer` 消息两条分支都不命中，会被**整条丢弃**，而不是保留。这是代码行为，论文与模型卡都没有提到。
渲染时：最后一条 user 消息之后若是思考模式，追加 `<think>`；每组工具结果 `</function_results>` 之后，若 $j\ge u^*$ 且思考模式，追加 `\n\n<think>`，否则追加 `\n\n</think>`。
这与 V32 §3.2.1 的两条规则（C30）逐条一致；V3.1 的 Jinja 模板则是“遇到 `</think>` 就只保留其后内容”，即每一轮都丢弃思考（C8）。

### 2.6 UE8M0 缩放（代码转录）

HF-V32 / HF-V31T `inference/kernel.py`（TileLang）中的激活量化：

```python
def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))

def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)

def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))
```
（`fp8_max = 448.0`，`group_size = 128`；`round_scale=scale_fmt is not None`。）

数学形式（课程记号，$a$ = 一个 1×128 分块内的最大绝对值）：

$$
s_{\text{FP32}}=\frac{a}{448},\qquad s_{\text{UE8M0}}=2^{\lceil\log_2(a/448)\rceil},\qquad q_j=\clip\!\big(x_j/s,\,-448,\,448\big)\ \text{（再舍入到 E4M3）}.
$$

- 因为 $s_{\text{UE8M0}}\ge s_{\text{FP32}}$，量化后最大码值 $a/s_{\text{UE8M0}}\in(224,448]$：E4M3 的动态范围只用到 50%–100%（§4 N29–N30）。
- MX 规范的转换算法（MX 算法 1）是 $X\leftarrow 2^{\lfloor\log_2\max|V_i|\rfloor-emax_{elem}}$，块大小 $k=32$；DeepSeek 用的是**向上取整到 2 的幂、块大小 128**。两者都产生 2 的幂缩放，但块粒度与取整方向不同（§7 E14）。
- 定义层面：PTX 文档称 ue8m0 为 “an 8-bit unsigned floating-point format with 8 bits for exponent and 0 bits for mantissa. The ue8m0 encoding does not support infinity. NaN value is limited to 0xff.”；MX 论文称 “All concrete MX formats use E8M0 (an 8-bit exponent) as the format for the shared scale. The representable exponents of these formats is a superset of the representable exponents of FP32.”

### 2.7 DeepSeekMath-V2 的奖励（MV2 式 (1)–(6)）

原文转录：

$$R_{\text{score}}(s_{i}^{\prime},s_{i})=1-|s_{i}^{\prime}-s_{i}|\tag{1}$$

$$\max_{\pi_{\varphi}}\mathbb{E}_{(X_{i},Y_{i},s_{i})\sim\mathcal{D}_{v},(V_{i}^{\prime},s_{i}^{\prime})\sim\pi_{\varphi}(\cdot|X_{i},Y_{i})}\left[R_{\text{format}}(V_{i}^{\prime})\cdot R_{\text{score}}(s_{i}^{\prime},s_{i})\right]\tag{2}$$

$$R_{V}=R_{\text{format}}\cdot R_{\text{score}}\cdot R_{\text{meta}}\tag{3}$$

$$\max_{\pi_{\theta}}\mathbb{E}_{X_{i}\sim\mathcal{D}_{p},Y_{i}\sim\pi_{\theta}(\cdot|X_{i})}[R_{Y}]\tag{4}$$

$$R=R_{\text{format}}(Y,Z)\cdot(\alpha\cdot R_{Y}+\beta\cdot R_{Z})\tag{5}$$

$$R_{Z}=R_{\text{score}}(s^{\prime},s)\cdot R_{\text{meta}}(Z)\tag{6}$$

其中 $\alpha=0.76,\ \beta=0.24$；$R_Y=s$ 是验证器给证明 $Y$ 的分数；$s'$ 是生成器在自评 $Z$ 中给自己的分数；$R_{\text{meta}}(Z)=ms$ 是验证器对自评的元验证分。课程记号：$R=R_{\text{fmt}}\cdot(w_Y R_Y+w_Z R_Z)$，$w_Y=0.76,\ w_Z=0.24$。算例见 §6 X10。

### 2.8 DSA（仅指针，属第 14 课）

V32 式 (1) lightning indexer $I_{t,s}=\sum_{j=1}^{H^{I}}w_{t,j}^{I}\cdot\text{ReLU}(\mathbf{q}^{I}_{t,j}\cdot\mathbf{k}^{I}_{s})$，式 (2) top-k 选择，式 (3)(4) indexer 的 KL 对齐损失。本课只需要两条：DSA 是 V3.2 相对 V3.1-Terminus 的唯一架构改动（C14）；V3.2 的后训练同样使用稀疏注意力（V32 §3 首段 “The post-training of DeepSeek-V3.2 also employs sparse attention in the same way as the sparse continued pre-training stage.”）。

---

## 3. 规格与超参数表

### 3.1 版本与模式

| 模型 | 发布 | 总参 / 激活 | 上下文 | 模式 | 工具调用 | 来源 |
|---|---|---|---|---|---|---|
| DeepSeek-V3.1-Base | 2025-08-19（HF） | 671B / 37B | 128K | 基座 | — | HF-V31 下载表；HF API |
| DeepSeek-V3.1 | 2025-08-21 | 671B / 37B | 128K | thinking + non-thinking（模板切换） | 非思考模式；搜索智能体为思考模式专用格式 | HF-V31；NEWS 250821 |
| DeepSeek-V3.1-Terminus | 2025-09-22 | 同 V3.1（结构同 V3） | 128K | 同上 | 同上，搜索模板更新 | HF-V31T；NEWS 250922 |
| DeepSeek-V3.2-Exp | 2025-09-29 | 同 V3.2 | 128K | 同上 | — | NEWS 250929；EXP |
| DeepSeek-V3.2 | 2025-12-01 | 结构同 V3.2-Exp | 128K（V32 §4.1 “supports a maximum context length of only 128K”） | 同上 | 思考与非思考模式都支持；首次“思考中调用工具” | HF-V32；NEWS 251201；V32 |
| DeepSeek-V3.2-Speciale | 2025-12-01（HF createdAt 11-28） | 同 V3.2 | 竞赛评测最大生成 128k（附录 D） | 仅深度推理 | **不支持** | HF-V32S；NEWS 251201 |
| DeepSeekMath-V2 | 2025-11-27 | 基于 V3.2-Exp-Base | 证明生成 “within the 128K token limit”（MV2 §3.3.2） | 证明生成 + 自验证 | — | MV2；HF-MV2 |

HF safetensors 元数据中的参数总数（含 MTP 层与 FP32 缩放因子，**不是**论文口径的 671B）：V3 / V3.1 / V3-0324 / R1-0528 均为 684,531,386,000；V3.2-Exp / V3.2-Speciale / Math-V2 均为 685,396,921,376。差值 865,535,376 恰好等于 62 层（61 层 + 1 个 MTP 层）× 每层 indexer 参数 13,960,248（§4 N34）。HF 页面上 V3.2 本体显示 685,355,329,792，与 Speciale 不一致，但笔者实测 V3.2 各分片缩放因子 dtype 仍为 F32，推测是 HF 元数据统计问题（§8 U18）。

### 3.2 架构配置（config.json，V3.1 与 V3.2 对照）

| 字段 | V3.1 | V3.2 | 备注 / 来源 |
|---|---|---|---|
| `model_type` / `architectures` | `deepseek_v3` / `DeepseekV3ForCausalLM` | `deepseek_v32` / `DeepseekV32ForCausalLM` | HF-V31、HF-V32 config.json |
| `hidden_size` | 7168 | 7168 | |
| `num_hidden_layers` | 61 | 61 | |
| `first_k_dense_replace` | 3 | 3 | 前 3 层稠密 FFN |
| `num_attention_heads` / `num_key_value_heads` | 128 / 128 | 128 / 128 | MLA |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | 1536 / 512 | $d_c'=1536$，$d_c=512$ |
| `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` | 128 / 64 / 128 | 128 / 64 / 128 | |
| `n_routed_experts` / `n_shared_experts` / `num_experts_per_tok` | 256 / 1 / 8 | 256 / 1 / 8 | $N_r=256,N_s=1,K_r=8$ |
| `moe_intermediate_size` / `intermediate_size` | 2048 / 18432 | 2048 / 18432 | |
| `n_group` / `topk_group` / `topk_method` | 8 / 4 / `noaux_tc` | 8 / 4 / `noaux_tc` | 节点受限 + 无辅助损失 |
| `scoring_func` / `routed_scaling_factor` | sigmoid / 2.5 | sigmoid / 2.5 | |
| `num_nextn_predict_layers` | 1 | 1 | MTP |
| `index_n_heads` / `index_head_dim` / `index_topk` | — | 64 / 128 / 2048 | DSA（第 14 课） |
| `rope_scaling` | yarn, factor 40, original 4096, beta_fast 32, beta_slow 1 | 同 | |
| `max_position_embeddings` | 163840 | 163840 | $=40\times4096$（§4 N35） |
| `vocab_size` | 129280 | 129280 | |
| `quantization_config` | fp8, e4m3, dynamic, block [128,128], `scale_fmt: ue8m0` | 同 | V3 原版无 `scale_fmt` 字段 |
| V3.1-Terminus `config.json` | **无** `scale_fmt`，多一个 `"attn_module_list_cfg": []` | — | 但其 `inference/config_671B_v3.1.json` 写 `"scale_fmt": "ue8m0"`（§7 E13） |
| `generation_config.json` 采样 | temperature 0.6, top_p 0.95 | temperature 1.0, top_p 0.95 | HF-V32 另建议本地部署用 “temperature = 1.0, top_p = 0.95” |

### 3.3 训练 token 账

| 阶段 | 步数 × 批 × 长度 | token | 来源 |
|---|---|---|---|
| V3 原 32K 扩展 | 1000 × 1920 × 32K | 62.91B（复算） | V3 §4.3 |
| V3 原 128K 扩展 | 1000 × 480 × 128K | 62.91B（复算） | V3 §4.3 |
| V3.1 32K 扩展 | “increased 10-fold” | 630B | HF-V31 |
| V3.1 128K 扩展 | “extended by 3.3x” | 209B | HF-V31 |
| V3.1 合计 | — | “840B”（复算 839B） | NEWS 250821 |
| V3.2 DSA 稠密预热 | 1000 × 16 × 128K，lr $10^{-3}$，只训 indexer | 2.1B | V32 §2.1.1 |
| V3.2 DSA 稀疏训练 | 15000 × 480 × 128K，lr $7.3\times10^{-6}$，每 query 选 2048 KV | 943.7B | V32 §2.1.1 |
| V3.2 后训练 RL | “thousands of steps” | 未披露 | V32 §3 |

### 3.4 RL 超参数：披露与未披露

| 项 | 值 | 来源 |
|---|---|---|
| 算法 | GRPO | V32 §3 |
| 优势 | $R_i-\text{mean}(\boldsymbol R)$ | V32 §3.1 |
| 损失归一化 | $\frac1G\sum_i\frac1{\lvert o_i\rvert}\sum_t$（先序列内平均） | V32 式 (5) |
| KL 估计 | 重要性修正 k3；数学领域可弱化或去掉 | V32 式 (7)，C25 |
| off-policy 掩码 | 仅负优势序列；阈值 $\delta$ | V32 式 (9) |
| Keep Routing | 自 V3-0324 起 | V32 §3.1 |
| Keep Sampling Mask | 与 top-p 联用 | V32 §3.1 |
| 奖励（推理/智能体） | 规则结果奖励 + 长度惩罚 + 语言一致性奖励 | V32 §3 |
| 奖励（通用） | 生成式奖励模型 + 每题 rubric | V32 §3 |
| $\varepsilon,\beta,\delta,G$、学习率、batch、mini-batch 数、top-p 值、长度惩罚形式、RL 总步数 | **均未披露** | §8 U1–U8 |
| 后训练算力 | “exceeding / already exceeds 10% of the pre-training cost” | V32 §1、§4.1 |

### 3.5 智能体任务（V32 表 1 + §3.2.3）

| 类别 | 任务数 | 环境 | 提示 | 占总数 | 额外事实 |
|---|---|---|---|---|---|
| code agent | 24,667 | real | extracted | 28.9% | 环境来自数百万 GitHub issue–PR 对，“tens of thousands” 可复现环境，8 种语言 |
| search agent | 50,275 | real | synthesized | 59.0% | 多智能体生成 + 验证；另加有益性 RL 数据与 rubric 奖励 |
| general agent | 4,417 | synthesized | synthesized | 5.2% | 1,827 个合成环境；pass@100 非零过滤 |
| code interpreter | 5,908 | real | extracted | 6.9% | Jupyter Notebook；数学、逻辑、数据科学 |
| 合计 | 85,267 | | | 100% | 引言说 “85,000 complex prompts” |

### 3.6 评测设置（V32）

温度 1.0；上下文 128K；数学题模板 `"{question}\nPlease reason step by step, and put your final answer within \boxed{}."`；HLE 另用官方模板得 23.9；τ²-bench 由模型自己扮演用户；MCP 工具输出放在 `tool` 角色；MCP-Universe / MCP-Mark 用内部环境；搜索用商业搜索 API。来源：V32 §4.1。

### 3.7 DeepSeekMath-V2 超参数

| 项 | 值 | 来源 |
|---|---|---|
| 冷启动题库 $\mathcal D_p$ | 17,503 题（AoPS） | MV2 §2.1.1 |
| 证明分档 | $\{0,0.5,1\}$ | MV2 §2.1.1 |
| 奖励权重 | $\alpha=0.76,\ \beta=0.24$ | MV2 §2.2.2 |
| 元验证效果 | 0.85 → 0.96 | MV2 §2.1.2 |
| 自动标注 $n,m,k$ | 未披露 | MV2 §2.3 |
| 一次生成评测 | 每题 8 个证明、8 次验证多数投票 | MV2 §3.3.1 |
| 顺序精修评测 | ISL 2024，每题 32 条线程，32 次验证多数投票，最多 8 次迭代 | MV2 §3.3.2 |
| 高算力搜索 | 池初始化 64 证明 × 64 分析；每轮取前 64、各配 8 条分析；≤16 轮或通过全部 64 次验证 | MV2 §3.3.3 |
| 基座 | DeepSeek-V3.2-Exp-Base（验证器 RL 从 V3.2-Exp-SFT 的一个版本开始） | MV2 §1，§2.1.1 |
| 许可 | Apache-2.0（V3-0324、R1-0528、V3.1 至 V3.2-Speciale 的 HF 标签均为 MIT；**V3 原版**不是 MIT：代码 MIT、权重另受 DeepSeek Model License 约束，HF API 无 license 标签） | HF-MV2；HF API；DeepSeek-V3 README §7 |
| 内部 CNML 级题 | 91 题：代数 13、几何 24、数论 19、组合 24、不等式 11 | MV2 §3.2 |

### 3.8 聊天模板与特殊 token

V3.1（HF-V31 模型卡，原样）：
- 非思考首轮：`<｜begin▁of▁sentence｜>{system prompt}<｜User｜>{query}<｜Assistant｜></think>`
- 思考首轮：`<｜begin▁of▁sentence｜>{system prompt}<｜User｜>{query}<｜Assistant｜><think>`
- 多轮上下文（两种模式相同）：`...<｜User｜>{query}<｜Assistant｜></think>{response}<｜end▁of▁sentence｜>...`
- 工具调用格式：`<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{additional_tool_calls}<｜tool▁calls▁end｜>`

特殊 token id（笔者从 HF `tokenizer.json` 读出）：

| id | V3 | V3.1 | 备注 |
|---|---|---|---|
| 0 / 1 | `<｜begin▁of▁sentence｜>` / `<｜end▁of▁sentence｜>` | 同 | |
| 128796 | `<｜place▁holder▁no▁796｜>` | `<｜search▁begin｜>` | V3.1 复用占位 id |
| 128797 | `<｜place▁holder▁no▁797｜>` | `<｜search▁end｜>` | |
| 128798 | `<｜place▁holder▁no▁798｜>` | `<think>` | `special: false` |
| 128799 | `<｜place▁holder▁no▁799｜>` | `</think>` | `special: false` |
| 128803 / 128804 | `<｜User｜>` / `<｜Assistant｜>` | 同 | |
| 128806–128814 | tool calls / outputs / sep | 同 | |

两个词表都是 128,000 个 BPE + 818 个 added token；只有上表 4 个 id 在 V3→V3.1 之间改变了内容。HF-V32 README 的示例 `tokenizer.encode(prompt)` 输出里 128798 与 128799 分别出现在 `<think>` 与 `</think>` 的位置，与此一致。

V3.2（HF-V32 `encoding_dsv32.py`）：工具调用改为 `<｜DSML｜function_calls>` / `<｜DSML｜invoke name="...">` / `<｜DSML｜parameter name="..." string="true|false">` 的 XML 式格式；工具结果包在 `<function_results><result>...</result></function_results>` 中；思考模式下工具结果之后追加 `<think>`。无 Jinja 模板。

---

## 4. 复算的数字

全部在 Python 3.11（numpy 1.26）中复算，脚本在会话临时目录（`nums.py`、`rl.py`、`mask.py`、`ue8m0.py`），不入库。格式：表达式 → 值。

### 4.1 token 与算力账

- **N1** V3 原 32K 阶段：`1000*1920*32768` → 62,914,560,000 = **62.91B**（V3 §4.3）
- **N2** V3 原 128K 阶段：`1000*480*131072` → 62,914,560,000 = **62.91B**（两阶段 token 数相同）
- **N3** V3.1 倍数：`630e9/62.91456e9` → **10.01**；`209e9/62.91456e9` → **3.32**；`630+209` → **839B**（新闻页写 “840B”）
- **N4** 反推：`10*62.91` → 629.1B（≈630B）；`3.3*62.91` → 207.6B（公布 209B，说明 “3.3x” 是四舍五入）
- **N5** DSA 稠密预热：`1000*16*131072` → 2,097,152,000 = **2.1B**（V32 原文 “2.1B”）
- **N6** DSA 稀疏训练：`15000*480*131072` → 943,718,400,000 = **943.7B**（V32 原文 “943.7B”）
- **N7** V3 基座之后的续训合计（V3.1 扩展 + DSA 两阶段）：`839 + 2.1 + 943.7` → **≈1,784.8B**（笔者汇总，论文未给此合计）
- **N8** 表 1 任务总数：`24667+50275+4417+5908` → **85,267**（引言 “85,000”）
- **N9** 各类占比：code 28.9%、search 59.0%、general 5.2%、code interpreter 6.9%（`v/85267`）
- **N10** 通用智能体每环境任务数：`4417/1827` → **2.42**
- **N11** V3 的后训练占比：`5/2664` → **0.1877%**（V3 §1：2664K 预训练、119K 上下文扩展、5K 后训练 GPU 小时）；若以 V3 预训练为基数，“10%” 相当于 `0.1*2664K` → **266.4K H800 GPU 小时**，按 V3 报告的 2 美元/小时为 **53.28 万美元**；与 V3 相比占比放大 `10/0.1877` → **53.3 倍**。（笔者示意；V32 没有说 “pre-training cost” 的基数是哪一次预训练，§8 U9）
- **N12** τ²-bench 类别平均：`(63.8+81.1+96.2)/3` → **80.3667**，四舍五入应为 80.4，表 2 印 80.3（差 0.07，可能是未四舍五入的类别分或截断，§7 E20）

### 4.2 竞赛与基准

- **N13** Speciale IMO 2025：`7*5+0` → 35；`35/42` → **83.33%**（等于 MV2 表 1 的 “83.3%”）
- **N14** Speciale CMO 2025：`18+18+9+21+18+18` → **102**；`102/126` → 80.95%（CMO 每题 21 分，6 题满分 126）
- **N15** Speciale IOI 2025：`100+82+72+100+55+83` → **492**；`492/600` → 82.0%；IOI25 官方金牌线 438.30，超出 `492-438.30` → **53.70**；人类第 9 名 502.78、第 10 名 491.65，所以 492 插在第 9 与第 10 之间 → “10th” 与官方成绩表一致（IOI25 成绩页；共 330 名选手，金牌 28 枚）
- **N16** Speciale ICPC WF 2025：解出 A C D E F H I J K L 共 **10** 题，提交次数合计 `3+1+1+2+2+1+1+1+1+1` → **14**；ICPC25 记分牌：第 1 名解 11 题，第 2、3 名各解 10 题，12 题（A–L），金牌 4 队
- **N17** 表 3 token 倍数（Speciale / V3.2-Thinking）：AIME 23k/16k=1.44，HMMT Feb 27/19=1.42，HMMT Nov 25/18=1.39，IMOAnswerBench 45/27=1.67，LiveCodeBench 27/16=1.69，Codeforces 77/42=1.83，GPQA 16/7=2.29，HLE 35/21=1.67；**平均 1.67 倍**（8 个比值的算术平均 1.673；若用 token 总和之比 275k/166k 则为 1.66，课文写 “约 1.7 倍” 两种算法都成立）。Speciale / Gemini-3.0-Pro：1.53、1.69、1.67、2.50、2.08、3.50、2.00、2.33
- **N18** 表 5 自洽性（笔者示意）：若各次尝试独立、pass@1 = 12%，则 `1-0.88**2` → 22.6%、`1-0.88**4` → 40.0%；实测 pass@2 = 18%、pass@4 = 26%，明显低于独立假设——说明失败主要是“题目本身难”而不是“运气差”
- **N19** IMO 2025 官方奖牌线（IMO25 页面内嵌 JSON 统计）：金牌 72 人，最低 **35** 分；银牌 104 人 28–34；铜牌 145 人 19–27。Speciale 与 DeepSeekMath-V2 的 35/42 **恰好等于**金牌线

### 4.3 RL 机制算例

- **N20** GRPO 优势，$G=4$，奖励 [1,0,0,1]：V32 口径 `R-mean` → [0.5, −0.5, −0.5, 0.5]；DSM 口径（总体标准差 0.5）→ [1, −1, −1, 1]；若用样本标准差（ddof=1）→ [0.866, −0.866, −0.866, 0.866]。奖励 [1,0,0,0]：V32 → [0.75, −0.25, −0.25, −0.25]；DSM → [1.732, −0.577, −0.577, −0.577]。奖励全 1：两种口径优势都是 0（DSM 口径分母为 0，需要特判）
- **N21** 单 token KL 权重（$\pi_{ref}=0.1,\ \pi_\theta=\pi_{\theta_{old}}=0.001$）：k3 值 `100-ln100-1` → **94.39**；真实对数比 `ln(0.001/0.1)` → −4.605；k3 对 $\log\pi_\theta$ 的导数 `1-100` → **−99**；修正版导数 `1*ln(0.01)` → **−4.605**（有限差分复核：−99.0000 与 −4.6052）；两者相差 **21.5 倍**
- **N22** 权重随 $x=\pi_\theta/\pi_{ref}$ 的变化（原始 $1-1/x$ vs 修正 $\log x$）：x=0.001 → −999 vs −6.91；0.01 → −99 vs −4.61；0.1 → −9 vs −2.30；0.5 → −1 vs −0.69；1 → 0 vs 0；2 → 0.5 vs 0.69；10 → 0.9 vs 2.30。k3 的值本身：x=0.001 → 992.09；0.01 → 94.39；0.1 → 6.70；1 → 0；10 → 1.40
- **N23** 全分布梯度核对（5 元 softmax，logits 与参考分布由 `numpy.random.default_rng(0)` 生成，有限差分）：
  $\nabla\KL(\pi\|\pi_{ref})$ = [−0.0195, −0.2027, −0.0565, 0.1880, 0.0908]，$\E[\nabla(\rho k_3)]$ = **完全相同**；
  $\nabla\KL(\pi_{ref}\|\pi)$ = [0.0330, −0.2783, 0.0344, 0.1398, 0.0712]，$\E[\nabla k_3]$ = **完全相同**。
  结论：修正版的期望梯度是 $\KL(\pi_\theta\|\pi_{ref})$ 的梯度，原始 k3 的期望梯度是 $\KL(\pi_{ref}\|\pi_\theta)$ 的梯度（与 TM 式 (3) 一致）
- **N24** Off-policy 掩码算例（§6 X3 的数据）：公共 $\pi_{\theta_{old}}$ = [0.9, 0.6, 0.8, 0.5, 0.7, 0.95]；
  o1（$\hat A=+0.5$）$\pi_\theta$=[0.91,0.62,0.80,0.52,0.70,0.95] → 逐 token [−0.0110, −0.0328, 0, −0.0392, 0, 0]，$\bar D$ = **−0.0138**；
  o2（$\hat A=-0.5$）$\pi_\theta$=[0.8,0.3,0.7,0.2,0.6,0.9] → 逐 token [0.1178, 0.6931, 0.1335, 0.9163, 0.1542, 0.0541]，$\bar D$ = **0.3448**；
  o3（$\hat A=-0.5$）$\pi_\theta$=[0.85,0.5,0.75,0.4,0.65,0.93] → 逐 token [0.0572, 0.1823, 0.0645, 0.2231, 0.0741, 0.0213]，和 0.6225，$\bar D$ = **0.1038**；
  o4（$\hat A=+0.5$）$\pi_\theta$ 同 o2 → $\bar D$ = 0.3448。
  $\delta=0.05$：$m$ = (1, 0, 0, 1)；$\delta=0.2$：$m$ = (1, 0, 1, 1)。（$\delta$ 为笔者假设值，论文未给）
- **N25** o2 的重要性比率 $\rho$ = [0.8889, 0.5, 0.875, 0.4, 0.8571, 0.9474]；$\varepsilon=0.2$ 时逐 token 代理项 `min(ρA, clip(ρ)A)` = [−0.4444, −0.4, −0.4375, −0.4, −0.4286, −0.4737]，其中 ρ=0.5 与 0.4 两个 token 取到裁剪常数 −0.4（无梯度），其余 **4 个 token 仍有梯度**；掩码后 6 个全为 0。o3 的 $\rho$ = [0.944, 0.833, 0.938, 0.800, 0.929, 0.979]，全部在裁剪下界上或以上，6 个都有梯度——所以 δ 的取值直接决定这 6 个梯度留不留
- **N26** Keep Sampling Mask 算例：$\pi_{\theta_{old}}$ = [0.5, 0.3, 0.15, 0.04, 0.01]，top-p = 0.95 → 核 {a,b,c}（累计 0.95）；截断归一化后 [0.5263, 0.3158, 0.1579, 0, 0]。$\pi_\theta$ = [0.45, 0.33, 0.12, 0.07, 0.03]，其自身 top-p 核是 {a,b,c,d}（0.90 不够，加 d 到 0.97）。token c 的比率：全词表 $\pi_\theta$ / 截断 $\pi_{\theta_{old}}$ = `0.12/0.1579` → **0.760**；同一掩码并重归一化后 `0.1333/0.1579` → **0.844**；$\pi_\theta$ 落在旧核之外的质量 = **0.10**
- **N27** 上下文管理触发点：`0.8*131072` → **104,857.6** token；BrowseComp 的 Summary 策略平均步数 `364/140` → **2.6 倍**（140 与 53.4 只出现在 HF PDF，见 §7 E17）；Discard-all 相对无上下文管理 `67.6-51.4` → **+16.2** 分

### 4.4 UE8M0 与检查点实测

- **N28** E8M0 的指数跨度：$2^{-127}$ 到 $2^{127}$，`254*log10(2)` → **76.46** 个十进制数量级（二手报道所说 “76 个数量级” 与此一致；偏置 127 的规范出处未能直接读取，§8 U12）
- **N29** 缩放因子取整（E4M3 最大 448）：
  - a=3.0：FP32 缩放 0.006696；UE8M0 `2**ceil(log2(3/448))` = $2^{-7}$ = 0.0078125 → 最大码值 **384**（448 的 85.7%）
  - a=100：$2^{-2}$=0.25 → 400（89.3%）
  - a=0.02：$2^{-14}$ → 327.68（73.1%）
  - a=224：$2^{-1}$ → 448（100%）；a=224.1：$2^{0}$ → 224.1（**50.0%**，最坏情形）
  - 用 `struct` 复现 `fast_log2_ceil`/`fast_pow2` 的位运算，结果与 `math.ceil(math.log2())` 完全一致
- **N30** 若分块最大值在对数尺度上均匀分布，E4M3 范围的平均利用率 = $\int_0^1 2^{-u}du=\frac{1}{2\ln2}$ → **0.7213**（蒙特卡洛 20 万样本：0.7210，最小 0.5000）
- **N31** 缩放因子存储开销（笔者示意）：每 1×128 激活分块存 FP32 缩放 = `4/128` = 0.03125 B/元素；存 1 字节 UE8M0 = `1/128` = 0.0078 B/元素；MX 的 k=32 块存 1 字节 = `1/32` = 0.03125 B/元素。注意 HF 检查点与 `inference/kernel.py` 中缩放因子**实际仍以 FP32 存储**（N32），打包成 1 字节 UE8M0 的是 DeepGEMM 在 SM100 上的内核（DG）
- **N32** 检查点缩放因子实测（CKPT；HTTP Range 读取 safetensors 头与 `weight_scale_inv` 张量，判断 `np.frexp` 尾数是否全为 0.5）：
  - V3（2024-12）：第 0 层 `q_a_proj`、`o_proj`、`mlp.down_proj` 的 2 的幂比例 **0.000**
  - V3.1：`q_a_proj` 1.000，`mlp.down_proj` 1.000，**`o_proj` 0.000**；第 30、60 层 `kv_b_proj`、`experts.7.down_proj`、`shared_experts.up_proj` 均 1.000，`o_proj` 均 0.000
  - V3.1-Terminus：与 V3.1 相同（只测第 0 层）
  - V3.2：与 V3.1 相同；另 `indexer.wq_b`、`indexer.wk` 均 1.000（第 30、60 层）
  - 所有版本缩放因子 dtype 均为 **F32**，形状如 `q_a_proj.weight_scale_inv` = [12, 56]（`1536/128=12`，`7168/128=56`）；V3.2 抽查的 6 个分片（1、2、50、100、162、163）中全部缩放张量均为 F32
- **N33** 由 N32 得出：V3.1 模型卡说 “on both model weights and activations”，但 `o_proj` 权重从 V3.1 起到 V3.2 都不是 2 的幂缩放；Terminus 模型卡说 “will be corrected in future model releases”，在 V3.2 检查点中（抽查第 0/30/60 层）并未修正（§7 E13）
- **N34** DSA indexer 参数量（每层）：`1536*64*128`（wq_b）+ `7168*128`（wk）+ `2*128`（k_norm 的 weight+bias）+ `7168*64`（weights_proj）+ `64*12`（wq_b 缩放）+ `1*56`（wk 缩放）= **13,960,248**；× 62 层（索引文件中第 0–61 层都有 indexer）= **865,535,376**，与 HF 元数据差值 `685396921376-684531386000` **逐位相等**；占总量 **0.126%**。按 dtype 拆开（V3.2-Exp 对 V3.1-Terminus）也逐项吻合：F8_E4M3 差 837,025,792 = 62×(wq_b+wk)；BF16 差 28,442,624 = 62×weights_proj；F32 差 66,960 = 62×(k_norm 256 + 两组缩放 824)（核查时复算）
- **N35** YaRN：`40*4096` → **163,840** = `max_position_embeddings`

### 4.5 DeepSeekMath-V2

- **N36** 奖励 $R=0.76R_Y+0.24R_Z$（设 $R_{\text{fmt}}=1$，$R_{\text{meta}}=1$）：

  | 真实分 s | 自评 s' | $R_Z=1-\lvert s'-s\rvert$ | R |
  |---|---|---|---|
  | 1 | 1 | 1 | **1.00** |
  | 1 | 0.5 | 0.5 | 0.88 |
  | 0.5 | 1（夸大） | 0.5 | **0.50** |
  | 0.5 | 0.5（诚实） | 1 | **0.62** |
  | 0 | 1（夸大） | 0 | **0.00** |
  | 0 | 0（诚实） | 1 | **0.24** |

- **N37** CMO 2024：`0.738*126` → 92.99 → 93 分 = 4 题满分 84 + 另一题 **9 分**部分分（笔者由百分比反推；MV2 未给逐题分）
- **N38** Putnam 2024：`118/120` → 98.33%（MV2 表 1 “98.3%”）；人类最高 90，领先 28 分
- **N39** MV2 图 2（ISL 2024，数值从 PDF 文本层提取，Pass@1/Best@32 的配对顺序为笔者据图例与 “Best@32 高于 Pass@1” 推断）：迭代 1→8：Pass@1 0.15, 0.18, 0.21, 0.22, 0.24, 0.24, 0.26, 0.27；Best@32 0.26, 0.29, 0.33, 0.39, 0.39, 0.40, 0.40, 0.42。增益：Pass@1 **+0.12**，Best@32 **+0.16**
- **N40** MV2 图 3（IMO-ProofBench 专家评分，Basic / Advanced，%）：Claude Sonnet 4 27.1/4.8；DeepSeek R1 29.0/3.8；Qwen3-235B 33.3/5.2；Grok 4 46.7/18.6；Gemini 2.5 Pro 55.2/17.6；GPT-5 59.0/20.0；Gemini 2.5 Pro + (Huang & Yang, 2025) 69.5/24.8；Gemini Deep Think (IMO lite) 83.8/37.6；Gemini Deep Think (IMO Gold) 89.0/65.7；**DeepSeekMath-V2 (Heavy) 99.0/61.9**。与 DeepThink IMO Gold 相比：Basic **+10.0**，Advanced **−3.8**（与 C59 “outperforms ... on the basic set and remains competitive on the advanced set” 一致）
- **N41** MV2 图 1（CNML 级 91 题平均证明分，按类别 代数/几何/数论/组合/不等式；分组顺序按图例推断，且满足 “consistently outperforms”）：Gemini2.5-Pro 0.35/0.17/0.32/0.32/0.45；GPT5-Thinking-High 0.54/0.15/0.36/0.35/0.38；DeepSeekMath-V2 0.60/0.52/0.54/0.47/0.59；DeepSeekMath-V2 相对次优者的最小领先 0.06（代数）

### 4.6 版本间进步（同一基准、同一口径才可比）

| 基准 | V3.1（Thinking 或 Agent mode） | V3.1-Terminus | V3.2-Exp | V3.2（Thinking） | 来源 |
|---|---|---|---|---|---|
| MMLU-Pro | 84.8 | 85.0 | 85.0 | 85.0 | HF-V31、HF-V31T、EXP 表 1、V32 表 2 |
| GPQA-Diamond | 80.1 | 80.7 | 79.9 | 82.4 | 同上 |
| HLE（无工具） | 15.9 | 21.7 | 19.8 | 25.1 | 同上 |
| SWE Verified | 66.0（**非思考**，Agent mode） | 68.4 | 67.8 | 73.1（思考；非思考 72.1，表 9） | 同上（V3.2 为内部框架） |
| BrowseComp | 30.0 | 38.5 | 40.1 | 51.4（无上下文管理）/ 67.6* | 同上 |
| AIME 2025 | 88.4 | 88.4 | 89.3 | 93.1 | HF-V31、EXP 表 1（Terminus 列）、V32 |

注意 SWE Verified 一行的模式不一致：V3.1 模型卡里 66.0 在 “DeepSeek V3.1-NonThinking” 列（其 Thinking 列为 “-”，因为 V3.1 只在非思考模式调用工具，C9），V3.2 的 73.1 是思考模式。同模式比较应为 V3.1 非思考 66.0 → V3.2 非思考 72.1（V32 表 9）。

注意 Terminal-bench：V3.1/Terminus 报的是 “Terminal-bench (Terminus 1 framework)” 31.3/36.7（V3.1 的 31.3 同样在非思考列；V3.2-Exp 为 37.7，EXP 表 1），V3.2 报的是 “Terminal Bench 2.0” 46.4（Claude Code 框架），**不同版本、不同框架，不可直接比较**。EXP 表 1 自己也说明 V3.2-Exp 在 GPQA、HLE、HMMT 上略低是因为 “DeepSeek-V3.2-Exp generates fewer reasoning tokens”。

---

## 5. 图表清单与重绘建议

### 5.1 原论文图表清单

**V32（2512.02556v1）**

| 图/表 | 原标题（照抄） | 内容 | 本课是否重绘 |
|---|---|---|---|
| 图 1 | “Benchmark of DeepSeek-V3.2 and its counterparts. For HMMT 2025, we report the February competition, consistent with the baselines. For HLE, we report the text-only subset.” | 8 组柱状图（AIME 2025、HMMT 2025、HLE、Codeforces、SWE Verified、Terminal Bench 2.0、τ² Bench、Tool Decathlon），5 个模型 | 可选；数据见 F14。注意 HMMT 组 V3.2 的柱画的是 90.2（Nov），不是图注说的 Feb 92.5（§7 E19，矢量几何已确认） |
| 图 2 | “Attention architecture of DeepSeek-V3.2, where DSA is instantiated under MLA.” | DSA 结构 | 属第 14 课 |
| 图 3 | “Inference costs of DeepSeek-V3.1-Terminus and DeepSeek-V3.2 on H800 clusters.” | (a) prefilling、(b) decoding 每百万 token 成本 vs token 位置 0–128K；纵轴 (a) 0–0.7 美元、(b) 0–2.4 美元；按 2 美元/GPU 小时 | 属第 14 课；曲线数值无法从矢量图提取（§8 U17） |
| 图 4 | “Thinking retention mechanism in tool-calling scenarios.” | Turn 1.1–1.3 保留 Thinking 1.1/1.2，Turn 2.1 输入里只剩工具调用与结果、Answer 1、User message 2 | **重绘**（F8） |
| 图 5 | “RL training of DeepSeek-V3.2-SFT using exclusively synthetic general agent data.” | 3×3 子图：τ²-bench Airline/Retail/Telecom/Overall，MCP-Mark Filesystem/PostgreSQL，MCP-Universe Financial-Analysis/Location-Navigation/3D-Designing；横轴步数 0–2000；蓝线 RL，绿虚线 V32-SFT，黑虚线 V32-Exp | **简化重绘**（F11） |
| 图 6 | “Accuracy of Browsecomp with different test-time compute expansion strategies.” | 横轴 Real Steps 100–900，纵轴 52.5–70.0；Summary、Discard-75%、Discard-all、Parallel-fewest-step | 只有端点数据（F12） |
| 图 7 | “Illustration of the MHA and MQA modes of MLA. For DeepSeek-V3.1-Terminus, the MHA mode is used for training and prefilling, while the MQA mode is used for decoding.” | MLA 两种模式 | 属第 9/14 课 |
| 表 1 | 智能体任务 | 见 §3.5 | **重绘**（F10） |
| 表 2 | 与闭源/开源模型对比 | 20 行基准 | 摘录（F14） |
| 表 3 | “Benchmark performance and efficiency of reasoning models.” | 准确率 + 输出 token（千） | **重绘为散点**（F13） |
| 表 4 | Speciale 竞赛成绩 | IMO/CMO/IOI 逐题 + ICPC 提交次数 | **重绘**（F15） |
| 表 5 | 合成任务 pass@K | K=1,2,4 | 进 F10 旁注 |
| 表 6–8 | 冷启动系统提示 | 推理 / 智能体 / 思考中调用工具 | 进 F8 旁注 |
| 表 9 | 非思考 vs 思考（智能体基准） | 7 行 | 进 F14 |

**MV2（2511.22570v1）**

| 图/表 | 原标题（照抄） | 内容 |
|---|---|---|
| 图 1 | “Average proof scores on CNML-level problems by category and model, as evaluated by our verifier.” | 5 类 × 3 模型（N41） |
| 图 2 | “Proof quality improvements as the maximum sequential iterations varies from 1 (no refinement) to 8 (initial generation plus up to 7 refinements based on self verification).” | Pass@1 与 Best@32 两条线（N39） |
| 图 3 | “Expert evaluation results on the Basic and Advanced subsets of IMO-ProofBench. All results are sourced from Luong et al. (2025), with the exception of DeepSeekMath-V2, which was evaluated by our experts following the grading guidelines.” | 10 个系统（N40） |
| 表 1 | “Problems in gray are fully solved, while underlined problems received partial credit.” | IMO 2025 83.3%、CMO 2024 73.8%、Putnam 2024 98.3% |

### 5.2 本课建议的图（15 张 + 1 张备选，文件名 `deepseek/figures/l15-*.svg`）

**F1 `l15-timeline` 版本时间线。** 横轴 2024-12 至 2025-12，节点：V3（2024-12-25）、V3-0324（2025-03-24，标注 “Keep Routing 起用”）、R1-0528（2025-05-28）、V3.1-Base（08-19）/V3.1（08-21，“混合思考 + UE8M0 + 839B 续训”）、V3.1-Terminus（09-22，“语言一致性”）、V3.2-Exp（09-29，“DSA，API 降价 50%+”）、DeepSeekMath-V2（11-27）、V3.2 + Speciale（12-01），arXiv v1（12-02）。颜色：架构改动 `--fig-blue`，训练/数据改动 `--fig-green`，数值格式改动 `--fig-orange`。

**F2 `l15-hybrid-template` 混合思考模板。** 三条 token 带（每格一个片段）：
① 非思考：`<bos>` `{system}` `<｜User｜>` `{query}` `<｜Assistant｜>` `</think>` → 生成 `{response}` `<eos>`；
② 思考：… `<｜Assistant｜>` `<think>` → 生成 `{reasoning}` `</think>` `{response}` `<eos>`；
③ 多轮：历史轮次统一为 `<｜Assistant｜></think>{response}<eos>`（思考被删，`</think>` 保留），最后一轮按模式接 `<think>` 或 `</think>`。
在 `<think>`、`</think>` 下标 id 128798、128799（§3.8）。被删的思考用 `--fig-red` 删除线。

**F3 `l15-context-extension` 续训 token 对比柱状图。** 两组柱（32K 阶段、128K 阶段），每组 V3 vs V3.1：62.91B vs 630B；62.91B vs 209B；柱顶标注 “×10.0” “×3.32”。对数纵轴或断轴。

**F4 `l15-ue8m0-sawtooth` 2 的幂缩放的利用率锯齿。** 横轴 $\log_2 a$（$a$ = 分块最大绝对值）从 −6 到 12，纵轴利用率 $u(a)=\dfrac{a}{448\cdot 2^{\lceil\log_2(a/448)\rceil}}$，在 $(0.5,1]$ 之间锯齿；每当 $a=448\cdot2^n$ 时 $u=1$，刚过则跳到 0.5。水平虚线标平均值 $1/(2\ln2)=0.7213$；标注 a=3.0（u=0.857）、a=224.1（u=0.500）两个例子点。上方再画一条 FP32 缩放的水平线 u=1 作对照。

**F5 `l15-scale-conformance` 检查点缩放因子是否为 2 的幂。** 网格：行 = 模块（q_a_proj、kv_b_proj、o_proj、mlp.down_proj、experts.7.down_proj、shared_experts.up_proj、indexer.wq_b、indexer.wk），列 = V3 / V3.1 / V3.1-Terminus / V3.2；格子填 1.000（`--fig-green`）、0.000（`--fig-red`）或 “无此模块”（`--fig-muted`）。数据见 N32；V3 只测了第 0 层的 q_a_proj、o_proj、down_proj，Terminus 只测了第 0 层，其余格注明 “未测”。

**F6 `l15-posttrain-pipeline` V3.2 后训练流水线。** 左：同一 V3.2 基座 → 8 个分支（数学、编程、通用逻辑推理、通用智能体、智能体编程、智能体搜索 + 写作、通用问答），每个 “专家 = 大规模 RL”；中：专家分别生成思考/非思考数据 → 蒸馏到一个检查点；右：一个混合 RL 阶段（GRPO；推理/智能体用规则奖励 + 长度惩罚 + 语言一致性，通用任务用生成式奖励模型 + rubric）→ DeepSeek-V3.2。下方分支：只用推理数据 + 更小长度惩罚 + DeepSeekMath-V2 的数据与奖励 → V3.2-Speciale。旁注 “算力 > 预训练成本的 10%”。

**F7 `l15-kl-weights` k3 与修正估计器的梯度权重。** 横轴 $x=\pi_\theta/\pi_{ref}$，对数刻度 $10^{-3}$ 到 $10^{1}$；两条曲线：原始 $1-1/x$（`--fig-red`），修正 $\log x$（`--fig-blue`，取 $\rho=1$）。纵轴用 symlog 或截断到 [−100, 3] 并在 $x=10^{-3}$ 处标 “−999”。数据点见 N22。标注 “$\pi_\theta\ll\pi_{ref}$ 区域：原始权重无界”。

**F8 `l15-thinking-retention` 重绘 V32 图 4。** 四列：Turn 1.1（输入 Tools + User 1；输出 Thinking 1.1 + Tool call 1.1）、Turn 1.2（输入加 Thinking 1.1、Tool call 1.1、Tool result 1.1；输出 Thinking 1.2 + Tool call 1.2）、Turn 1.3（输入再加 Thinking 1.2、Tool call 1.2、Tool result 1.2；输出 Thinking 1.3 + Answer 1）、Turn 2.1（输入 Tools、User 1、Tool call 1.1、Tool result 1.1、Tool call 1.2、Tool result 1.2、Answer 1、User 2——**无任何 Thinking**；输出 Thinking 2.1 …）。Thinking 块用 `--fig-blue` 实色，被丢弃处用 `--fig-red` 虚框示意。

**F9 `l15-offpolicy-mask` Off-policy 掩码决策图。** 横轴 $\bar D_i$（−0.1 到 0.4），纵轴 $\hat A_i$（−0.75 到 0.75）；右下象限（$\hat A<0$ 且 $\bar D>\delta$）填 `--fig-red` 浅色 = 掩掉；画两条竖线 $\delta=0.05$、$\delta=0.2$。四个点：o1 (−0.0138, +0.5)、o2 (0.3448, −0.5)、o3 (0.1038, −0.5)、o4 (0.3448, +0.5)。o3 在两条 δ 线之间，说明阈值的作用；o4 与 o2 横坐标相同但在上半平面，说明“只掩负样本”。

**F10 `l15-agent-tasks` 智能体任务构成 + 通用环境合成回路。** 左：横向条形图 search 50,275 / code 24,667 / code interpreter 5,908 / general 4,417（总 85,267），用颜色区分 real/synthesized 环境。右：通用智能体合成回路：任务类别 + 沙箱（bash、search）→ 取数据入库 → 合成工具函数 → 提出简单任务 + Python 解法 + 验证函数 → 验证不通过则改解法/验证函数 → 逐步加难、必要时扩充工具 → 用 V3.2 做 RL、保留 pass@100 > 0 → 1,827 环境 / 4,417 任务。旁注表 5：pass@1 V3.2-Exp 12%、Sonnet-4.5 34%、Gemini-3.0 Pro 51%、GPT-5-Thinking 62%。

**F11 `l15-synthetic-rl` 合成任务 RL 的迁移（据 V32 图 5 简化重绘，数值为目测读数，图注须写 “约”）。** 只画 τ²-bench Overall 一个子图：SFT 基线约 0.55（绿虚线）、V3.2-Exp 约 0.46（黑虚线）、RL 曲线从约 0.67（第 ~50 步）上升到约 0.83（第 ~1900 步）。可选再加 MCP-Mark/Filesystem：两条基线都约 0.14，RL 从约 0.15 升至约 0.345。

**F12 `l15-browsecomp-cm` 测试时上下文管理。** 只画已知点：无上下文管理 51.4（arXiv 表 2）；Summary 60.2（平均 364 步）；Discard-all 67.6。HF PDF 另有 “from 140 to 364 … from 53.4 to 60.2”，与 51.4 不一致，图中若用 140 步/53.4 须注明来源（§7 E17）。不要编造曲线形状。

**F13 `l15-speciale-tokens` 准确率 vs 输出 token 散点（表 3）。** 每个基准一个小面板；点 = (输出 token 千, 准确率)，顺序 GPT-5 High、Gemini-3.0 Pro、Kimi-K2 Thinking、V3.2 Thinking、V3.2 Speciale：
AIME 2025：(13, 94.6)、(15, 95.0)、(24, 94.5)、(16, 93.1)、(23, 96.0)；
HMMT Feb 2025：(16, 88.3)、(16, 97.5)、(31, 89.4)、(19, 92.5)、(27, 99.2)；
HMMT Nov 2025：(20, 89.2)、(15, 93.3)、(29, 89.2)、(18, 90.2)、(25, 94.4)；
IMOAnswerBench：(31, 76.0)、(18, 83.3)、(37, 78.6)、(27, 78.3)、(45, 84.5)；
LiveCodeBench：(13, 84.5)、(13, 90.7)、(29, 82.6)、(16, 83.3)、(27, 88.7)；
Codeforces（rating）：(29, 2537)、(22, 2708)、K2 无、(42, 2386)、(77, 2701)；
GPQA Diamond：(8, 85.7)、(8, 91.9)、(12, 84.5)、(7, 82.4)、(16, 85.7)；
HLE：(15, 26.3)、(15, 37.7)、(24, 23.9)、(21, 25.1)、(35, 30.6)。
用箭头连 V3.2 → Speciale，显示 “多花约 1.67 倍 token 换分”（N17）。

**F14 `l15-v32-benchmarks` 表 2 摘录柱状图。** 推理：AIME 2025、HMMT Feb 2025、HLE；智能体：SWE Verified、Terminal Bench 2.0、BrowseComp（51.4 与 67.6* 分开画）、τ²-bench、MCP-Universe、Tool-Decathlon；模型：Claude-4.5-Sonnet、GPT-5 High、Gemini-3.0 Pro、Kimi-K2 Thinking、MiniMax M2、DeepSeek-V3.2 Thinking。V32 表 2 数值（按上列模型顺序）：AIME 2025 87.0/94.6/95.0/94.5/78.3/93.1；HMMT Feb 2025 79.2/88.3/97.5/89.4/–/92.5；HLE 13.7/26.3/37.7/23.9/12.5/25.1；SWE Verified 77.2/74.9/76.2/71.3/69.4/73.1；Terminal Bench 2.0 42.8/35.2/54.2/35.7/30.0/46.4；BrowseComp 24.1/54.9/–/–·60.2*/44.0/51.4·67.6*；τ²-Bench 84.7/80.2/85.4/74.3/76.9/80.3；MCP-Universe 46.5/47.9/50.7/35.6/29.4/45.9；Tool-Decathlon 38.6/29.0/36.4/17.6/16.0/35.2。可叠加表 9 非思考模式（Terminal Bench 2.0 37.1、SWE Verified 72.1、SWE Multilingual 68.9、τ² 77.2、MCP-Universe 38.6、MCP-Mark 26.5、Tool-Decathlon 25.6）。

**F15 `l15-competitions` 竞赛成绩与奖牌线。** 四个横条：IMO 2025 35/42（金牌线 35，线正好压在条端）、CMO 2025 102/126（金牌线未核实，不画线）、IOI 2025 492/600（金牌线 438.30，第 9 名 502.78、第 10 名 491.65 两条细线）、ICPC WF 2025 10/12（第 1 名 11 题）。右侧加 DeepSeekMath-V2：IMO 2025 35/42、CMO 2024 93/126（反推）、Putnam 2024 118/120（人类最高 90）。

**（备选）F16 `l15-mathv2-loop` 验证器—生成器循环 + 奖励表。** 回路：人工标注 → 训练验证器（$R_{\text{fmt}}\cdot R_{\text{score}}$）→ 元验证器 → 增强验证器（$\times R_{\text{meta}}$）→ 用验证器当奖励训练生成器（证明 + 自评，$0.76R_Y+0.24R_Z$）→ 新证明 → 放大验证算力自动标注（$n$ 次验证、$m$ 次元验证、阈值 $k$）→ 回到验证器。下方附 N36 的奖励表与 N39 的精修曲线。

---

## 6. 算例与练习素材

**X1（计算）V3.1 的续训账。** V3 两阶段扩展各 1000 步：32K×1920、128K×480。求每阶段 token 数，并验证 V3.1 “10 倍 / 3.3 倍” 与 630B / 209B 的对应。
答案：各 62.91B（N1–N2）；630/62.91 = 10.01，209/62.91 = 3.32；合计 839B，新闻页取整为 840B。

**X2（计算 + 概念）GRPO 优势的两种口径。** $G=4$，奖励 [1,0,0,0]。分别按 V3.2（减均值）和 DeepSeekMath（再除总体标准差）计算优势；再算奖励全为 1 时的情形。说明两种口径在 “难题（只有一个答对）” 上给正样本的放大倍数差多少。
答案：V3.2 [0.75, −0.25, −0.25, −0.25]；DSM [1.732, −0.577, −0.577, −0.577]；正样本放大比 1.732/0.75 = 2.31。全 1 时 V3.2 全 0；DSM 分母为 0，需要特判（N20）。

**X3（计算，本课核心算例）Off-policy 序列掩码。** 用 N24 的四条序列（$G=4$，奖励 [1,0,0,1]，$\hat A$=[+0.5, −0.5, −0.5, +0.5]）：
(a) 计算每条序列的 $\bar D_i=\frac1{|o_i|}\sum_s\log(\pi_{\theta_{old}}/\pi_\theta)$；
(b) 分别取 $\delta=0.05$ 与 $\delta=0.2$，写出 $m_i$；
(c) 对 o2，$\varepsilon=0.2$，哪些 token 在不加掩码时仍然有梯度？
(d) o4 与 o2 的 $\bar D$ 相同，为什么 o4 不被掩？
答案：(a) −0.0138、0.3448、0.1038、0.3448。(b) $\delta=0.05$：(1,0,0,1)；$\delta=0.2$：(1,0,1,1)。(c) $\rho$ = [0.889, 0.5, 0.875, 0.4, 0.857, 0.947]，$\rho<0.8$ 的两个 token 取到裁剪常数 (1−ε)Â = −0.4，无梯度；其余 4 个仍有梯度，掩码把它们一起去掉（N25）。(d) 式 (9) 只掩负优势序列：“we only mask sequences with negative advantages”；作者的直觉是 “the model benefits the most by learning from its own mistakes, whereas highly off-policy negative samples can be detrimental”。

**X4（推导）为什么要乘 $\pi_\theta/\pi_{\theta_{old}}$。** 证明 $\E_{x\sim\pi_{\theta_{old}}}\big[\tfrac{\pi_\theta(x)}{\pi_{\theta_{old}}(x)}k_3(x)\big]=\KL(\pi_\theta\|\pi_{ref})$，其中 $k_3=u-\log u-1$，$u=\pi_{ref}/\pi_\theta$。
答案：换测度得 $\sum_x\pi_\theta(x)[u-\log u-1]=\sum_x\pi_{ref}(x)-\sum_x\pi_\theta\log\frac{\pi_{ref}}{\pi_\theta}-1=1+\KL(\pi_\theta\|\pi_{ref})-1$。

**X5（推导 + 计算）k3 的梯度方向。** (a) 求 $\partial k_3/\partial\log\pi_\theta$；(b) 证明 $\E_{\pi_\theta}[\nabla k_3]=\nabla\KL(\pi_{ref}\|\pi_\theta)$；(c) 求 $\partial(\rho k_3)/\partial\log\pi_\theta$ 在 $\rho=1$ 处的值；(d) 代入 $\pi_{ref}=0.1$、$\pi_\theta=\pi_{\theta_{old}}=0.001$。
答案：(a) $1-u$；(b) $\E_{\pi_\theta}[(1-u)\nabla\log\pi_\theta]=0-\sum_x\pi_{ref}\nabla\log\pi_\theta=\nabla\KL(\pi_{ref}\|\pi_\theta)$（第一项为零是因为 $\E_{\pi_\theta}[\nabla\log\pi_\theta]=0$）；(c) $\log(\pi_\theta/\pi_{ref})$；(d) −99 vs −4.605，差 21.5 倍（N21）。参考 TM 式 (3)。

**X6（计算）Keep Sampling Mask。** 用 N26 的五元分布，top-p = 0.95。(a) 写出 $\pi_{\theta_{old}}$ 与 $\pi_\theta$ 各自的核；(b) 若不保留掩码，token d 在训练中会出现什么问题？(c) 分别计算 token c 的 “全词表 $\pi_\theta$ / 截断 $\pi_{\theta_{old}}$” 与 “同掩码重归一化后” 的比率。
答案：(a) {a,b,c} 与 {a,b,c,d}；(b) d 在 $\pi_{\theta_{old}}^{\text{trunc}}$ 下概率为 0，重要性比率无定义，动作空间不一致正是 C28 所说 “violates the principles of importance sampling”；(c) 0.760 与 0.844。注：论文没说是否重归一化，课文要把两种算法都列出并注明（§8 U4）。

**X7（计算）UE8M0 缩放。** 一个 1×128 激活分块的最大绝对值 $a=3.0$，E4M3 最大 448。(a) 按 FP32 缩放与按 `fast_round_scale` 分别求缩放因子；(b) 量化后最大码值各是多少；(c) 最坏情况下 E4M3 的范围只用到多少？(d) 为什么这样做能 “兼容 microscaling”？
答案：(a) 0.006696 与 $2^{-7}$=0.0078125；(b) 448 与 384（85.7%）；(c) 刚超过 $448\cdot2^n$ 时只用到约 50%（N29）；(d) MX 格式的共享缩放是 E8M0，即只能表示 2 的幂（MX 论文 §2.2），硬件（如 SM100，DG）按打包的 UE8M0 读缩放因子。

**X8（计算 + 概念）85,000 条提示从哪来。** 用表 1 求总数与各类占比；引言的 “1,800 个环境” 对应表 1 的哪一行？
答案：85,267；search 59.0% 最多；1,800+ 环境对应 general agent 的 1,827 个合成环境，它们只承载 4,417 个任务（5.2%）；code agent 的环境是 “tens of thousands” 个真实仓库环境，不在 “1,800” 之内（C34–C39）。

**X9（计算）金牌是怎么算的。** (a) Speciale IMO 2025 逐题 7,7,7,7,7,0，与官方金牌线比较；(b) IOI 2025 逐题 100,82,72,100,55,83，与金牌线 438.30 及人类第 9、10 名比较；(c) ICPC 10/12 与第 1 名比较。
答案：(a) 35 = 金牌线 35（N19），不多不少；(b) 492，超出金牌线 53.70，介于 502.78 与 491.65 之间 → 第 10（N15）；(c) 第 1 名 11 题，Speciale 10 题与第 2、3 名题数相同，“第 2” 依赖罚时比较，论文只给了提交次数（§8 U13）。并强调附录 D 的前提：IOI 每题采样 500 个、按 “思考最长” 选 50 个提交，无工具、无联网。

**X10（计算）DeepSeekMath-V2 的诚实奖励。** $R=0.76R_Y+0.24R_Z$，$R_Z=1-|s'-s|$（设格式与元验证分均为 1）。计算 N36 表中六种情况，并回答：一个其实只有 0.5 分的证明，自评 1 与自评 0.5 哪个奖励高？一个错误证明老实自评 0 能拿多少？
答案：自评 0.5（0.62）高于自评 1（0.50）；错误证明老实自评得 0.24，夸大自评得 0。这就是 MV2 §2.2.2 要点所说 “Faithful acknowledgment of errors is rewarded over false claims of correctness.”

**X11（计算）后训练占比。** V3 的 GPU 小时：预训练 2664K、上下文扩展 119K、后训练 5K。(a) V3 后训练占预训练的百分比；(b) 若 V3.2 的 RL 预算 “超过预训练成本的 10%”，且以 V3 预训练为基数，至少多少 GPU 小时、多少美元（2 美元/小时）？(c) 这个推算有哪些不确定性？
答案：(a) 0.188%；(b) ≥266.4K GPU 小时，≥53.3 万美元，占比约为 V3 的 53 倍（N11）；(c) V32 没说基数是哪次预训练（V3 的 14.8T？是否含 V3.1 的 839B 与 DSA 的 945.8B 续训？）、“cost” 口径、是否含专家模型的 RL，均未披露。

**X12（计算）DSA indexer 占多少参数。** 由 config（`q_lora_rank` 1536、`index_n_heads` 64、`index_head_dim` 128、`hidden_size` 7168）与 FP8 128×128 分块缩放，求每层 indexer 参数与 62 层合计，并与 HF 元数据差值比较。
答案：13,960,248/层，865,535,376 合计，与 685,396,921,376 − 684,531,386,000 逐位相等；占 0.126%（N34）。本题适合放在第 14/15 课衔接处。

**X13（概念）思考上下文管理。** 给出一段消息序列：system、user₁、assistant₁（reasoning + tool_call）、tool、assistant₂（reasoning + tool_call）、tool、assistant₃（reasoning + answer）、user₂。按 V3.2 规则，编码 user₂ 之后的提示时哪些 reasoning 被保留？如果某框架把工具结果当作 user 消息发回来会怎样？
答案：user₂ 是最后一条 user 消息，它之前的 assistant₁–₃ 的 reasoning 全部被删，工具调用与结果保留（C30，§2.5）；若工具结果以 user 身份出现，每次工具返回都会触发 “新 user 消息” 规则，思考被反复丢弃，模型每次都要重新推理——这就是 C31 建议 Roo Code / Terminus 用非思考模式的原因，也是 Terminal Bench 2.0 的 46.4 改用 Claude Code 框架测的原因（C43）。HF-V32 的 `encoding/test_output.txt` 就是这一场景的真实编码样例（第一问两次工具往返的思考在第二问时被删，第二问的思考保留）。

---

## 7. 勘误与常见误解

**E1 “V3.1 用 840B token 重新预训练”——错。** 840B 是在 V3 基座上的**长上下文续训**：32K 阶段 630B + 128K 阶段 209B = 839B（C3、N3）。“10 倍 / 3.3 倍” 的基数是 V3 报告中各 1000 步的 62.91B（C4、N1–N2）。

**E2 “V3.2 的 GRPO 优势 = (r − mean)/std”——与论文印出的不符。** V32 §3.1 明写 “subtracting the average reward of the group from the reward of output o_i, i.e., Â_{i,t}=R_i−mean(R)”，没有除以标准差；DSM §4.1.2 的原始 GRPO 则除以 std。论文前一句又说 “estimated by normalizing the outcome reward within each group”，与后文公式不完全一致。课文照印出的公式讲，并注明 “是否除 std 论文前后表述不一”（§8 U3）。

**E3 “k3 是无偏的，所以直接当 KL 损失就行”——混淆了值的无偏与梯度的无偏。** k3 的**值**在 $x\sim\pi_\theta$ 时无偏；样本来自 $\pi_{\theta_{old}}$ 时需要重要性权重。更重要的是，把 k3 当损失求导，其期望梯度是 $\nabla\KL(\pi_{ref}\|\pi_\theta)$，方向反了（TM 式 (3)；N23 数值验证）。V32 的修正让期望梯度变成 $\nabla\KL(\pi_\theta\|\pi_{ref})$（单步情形）。V32 原话是 “the gradient of this KL estimator becomes unbiased”，没有展开推导，推导是笔者补的。

**E4 “off-policy 序列掩码 = 丢掉所有离策略样本”——错。** 只掩 “$\hat A<0$ 且平均对数比 $>\delta$” 的序列；正优势序列无论多离策略都保留；掩码只作用在裁剪代理项，KL 项仍在（式 (8)(9)，C26）。

**E5 “$\pi_{old}$ 就是上一步的训练策略”——在 V3.2 中不准确。** V32 明说 “π_old here denotes the sampling probability directly returned by the inference framework”，因此比率同时包含 “多个 mini-batch 更新” 与 “训练/推理实现差异” 两种离策略来源（C26）。

**E6 “Keep Routing 是 V3.2 新发明”——错。** 论文说 “has been adopted in our RL training pipeline since DeepSeek-V3-0324”（C27）。同类做法在 Qwen 的 GSPO 论文里叫 Routing Replay，并被指出有显存/通信开销、会限制 MoE 容量；GSPO 声称序列级比率可以免去这一技巧（GSPO §5.3）。不要把首创归给任何一方，谁先谁后无法从这两篇确认（§8 U6）。

**E7 “1,800+ 环境里有 85,000 个任务”——错把两个数字绑在一起。** 1,827 个合成环境只对应 4,417 个通用智能体任务；85,267 个任务中 59% 是真实环境的搜索任务、29% 是真实仓库的代码任务（C34–C35、N8–N10）。

**E8 “后训练算力是预训练的 10%”——措辞不准。** 原文两处：“exceeding 10% of the pre-training cost” 与 “already exceeds 10% of the pre-training cost”。是 “超过”，单位是 “cost”，基数未定义，也没有绝对 GPU 小时（C16、X11）。

**E9 专家蒸馏的领域数：V3.2-Exp 报告说 5 个，V3.2 论文说 6 个。** EXP：“mathematics, competitive programming, general logical reasoning, agentic coding, and agentic search”；V32：“mathematics, programming, general logical reasoning, general agentic tasks, agentic coding, and agentic search”。新增的 “general agentic tasks” 与 §3.2.3 的合成通用智能体任务对应。课文按 V32 写 6 个，并注明 V3.2-Exp 时是 5 个（C17）。

**E10 “V3.1 已经能在思考中调用工具”——错。** V3.1 模型卡：“Toolcall is supported in non-thinking mode.”（搜索智能体有思考模式专用格式）。NEWS 251201 称 V3.2 是 “our first model to integrate thinking directly into tool-use”（C9、C30）。

**E11 “Speciale 参加了 IMO/IOI 并拿到金牌”——措辞需谨慎。** 论文说的是 “gold-medal performance / gold-medal level performance / reaching gold-medal thresholds”，是 DeepSeek 自己按赛制评测、事后对照奖牌线（附录 D）。IOI 每题采样 500 个、按思考最长选 50 个提交；IMO/CMO 用生成—验证—修正循环；无工具、无联网、最大 128k 生成。IMO 的 35/42 恰好等于 2025 年金牌线 35（N19），P6 得 0。评分者是谁，V32 没写（§8 U11）。另外 V32 §4.2 脚注 5 写明 “We evaluated the English version of CMO 2025.”，并称 IMO 2025 与 CMO 2025 的题目和推理代码放在 DeepSeek-Math-V2 的 GitHub 仓库——CMO 成绩是英文译本上的成绩。

**E12 DeepSeekMath-V2 的 CMO 是 2024 年，V3.2-Speciale 的 CMO 是 2025 年。** MV2 摘要：“gold-level scores on IMO 2025 and CMO 2024”；V32 表 4：CMO 2025 102/126。两者常被混写。MV2 的 Putnam 118/120 也是 “scaled test-time compute” 下的结果，且 “the remaining problem with minor errors”，由其自家数学专家评分（C58）。

**E13 UE8M0 在检查点里并非 “全部权重”。** 实测（N32–N33）：从 V3.1 到 V3.2，除 `self_attn.o_proj` 外的 FP8 权重缩放因子都是 2 的幂；`o_proj` 的缩放因子一个都不是。Terminus 模型卡把这当作 “known issue ... will be corrected in future model releases”，但 V3.1 原版模型卡没有提，V3.2 检查点（抽查 0/30/60 层）也未修正。另外，Terminus 的 `config.json` 没有 `scale_fmt` 字段，而它的 `inference/config_671B_v3.1.json` 有 `"scale_fmt": "ue8m0"`——依赖 `config.json` 的推理框架可能按普通 FP32 缩放处理（笔者推测）。

**E14 “UE8M0 = MXFP8”——不完全对。** MX 规范的具体格式块大小 $k=32$、共享缩放 E8M0（MX 表 1）；DeepSeek 沿用 V3 的 1×128 激活分块与 128×128 权重分块（`group_size = 128`，`weight_block_size [128,128]`），只是把缩放因子限制为 2 的幂（向上取整）。检查点里缩放因子仍以 F32 存储（N32）；打包成 UE8M0 字节是 DeepGEMM 在 SM100 上的要求（DG：“SM100 requires scaling factors in packed UE8M0 format, which packs 4 UE8M0 into a single `torch.int`.”）。另外，V3 报告里已有两类激活用了 “integral power of 2” 的缩放（V3 §3.3：“all the scaling factors are round scaled, i.e., integral power of 2.”），V3.1 是把这一做法推广到全部权重与激活；V3 报告也已写到 “our fine-grained quantization strategy is highly consistent with the idea of microscaling formats (Rouhani et al., 2023b)”。

**E15 API 文档漂移：思考内容要不要回传。** 2025-12-09 快照（V3.2 时代）：“When the next user question begins (Turn 2.1), the previous reasoning_content should be removed, while keeping other elements to send to the API. If reasoning_content is retained and sent to the API, the API will ignore it.”——与 V32 §3.2.1 一致。2026-09 现行版（V4 时代）：“If the request carries the tools parameter: the reasoning_content of all previous turns should be passed back to the API and will be concatenated into the context.”——规则已变。课文讲 V3.2 时必须引快照版，并注明现行 API 行为已不同。快照版还写明思考模式 “max_tokens：The maximum output length (including the COT part). Default to 32K, maximum to 64K.”

**E16 “V3.2 的非思考模式不能用工具”——错。** NEWS 251201：“also supports tool-use in both thinking and non-thinking modes”；附录 C 表 9 给出非思考模式的智能体成绩。

**E17 HF 仓库里的 paper.pdf 与 arXiv v1 有差异。** 逐句 diff 只发现两处：HF PDF §4.4 写 “Summary extends the average steps from 140 to 364, improving performance from 53.4 to 60.2.”，arXiv v1 改为 “Summary extends the average steps to 364, achieving a performance improvement of up to 60.2.”；arXiv v1 在引言多引了 Qwen3 技术报告。53.4 与表 2 的 “51.4 without context management” 不一致，这很可能是 arXiv 版删去这对数字的原因（笔者推测）。课文引用以 arXiv v1 为准。

**E18 arXiv HTML 渲染错误。** HTML 版把 §3.2.2 的 “Tables 6–8 / Table 6 / Table 7” 显示成 “Tables 8–8 / Table 8”；PDF 正确。引用表号以 PDF 为准。

**E19 图 1 的 HMMT 柱与图注不符（已由矢量几何确认）。** 图注说 “For HMMT 2025, we report the February competition”，表 2/表 3 中 V3.2-Thinking 的 HMMT Feb 2025 为 92.5、HMMT Nov 2025 为 90.2；但 arXiv PDF 与 HF PDF 的文本层中，图 1 HMMT 组的 5 个标签都是 99.2、90.2、88.3、79.2、97.5，没有 92.5。其余 7 组标签都严格按图例顺序（Speciale、V3.2、GPT-5、Claude、Gemini）排列。核查时进一步读取了 arXiv v1 PDF 第 1 页图 1 表单对象（`/Im1`）里的填充路径：纵轴零线 y=95.4、每 1% = 4.896 单位（由 0 与 100 两条刻度得出）；HMMT 组第 2 根柱（V3.2-Thinking 槽位，x=215.6–237.3）顶端 y=537.0，折合 **90.19**；同法算得同组其余柱为 99.18、88.30、79.19、97.49，AIME 组 V3.2 柱为 93.10，都与表值吻合。所以**柱子本身**（不只是标签）画的就是 Nov 的 90.2。重绘时用表 2 的 92.5 并加注。

**E20 表 2 τ²-bench 80.3 与三类平均的四舍五入差 0.07。** (63.8+81.1+96.2)/3 = 80.367（N12）。影响可忽略，但若课文让学生 “验算 80.3”，要预先说明。

**E21 “V3.2 可以直接用 Terminus 等框架跑思考模式”——论文建议相反。** C31、C43：思考模式的上下文管理与用 user 消息模拟工具的框架不兼容；Terminal Bench 2.0 的 46.4 是 Claude Code 框架下的成绩，Terminus 框架只测了非思考模式（39.3）。

**E22 “DeepSeekMath-V2 在 IMO 2025 拿了满分”——错。** 5/6 解出，35/42（83.3%），恰好压金牌线；P6 未解出（C58、N13、N19）。

**E23 “Speciale 支持工具调用 / 官方 API 长期可用”——错；但 “Speciale 只能通过 API 用” 也不对。** 模型卡：“does not support the tool-calling functionality”；官方 API 只通过临时端点提供，到 2025-12-15 15:59 UTC 为止（C48）。新闻页的 “API-only for now” 指官方服务只走 API（不进 App/Web）；同一新闻页 “Open Source Release” 一节就给出了 Speciale 的 HF 权重链接（HF-V32S，createdAt 2025-11-28，MIT 许可），所以端点到期后仍可自行部署。

**E24 R1 式的丢思考并非 “bug”，是设计取舍。** V32 的说法是 R1 策略在工具调用里 “results in significant token inefficiency”（C29）；V3.1 的模板（C8）也是每轮丢弃思考。V3.2 的改变只针对 “同一 user 轮内的工具往返”，新 user 消息到来时仍然丢弃（C30）。

**E25 V3.1 与 V3.2 的推荐温度不同。** `generation_config.json`：V3.1 为 temperature 0.6 / top_p 0.95，V3.2 为 temperature 1.0 / top_p 0.95；V32 的所有评测也用温度 1.0（C41）。课文若给 “推荐采样参数”，须按版本区分。

---

## 8. 未能核实 / UNVERIFIED

- **U1** Off-policy 掩码阈值 $\delta$ 的取值（及是否随领域变化）。V32 只说 “δ is a hyper-parameter”。本笔记算例中的 0.05 / 0.2 是笔者假设。
- **U2** $\varepsilon$、$\beta$（各领域的 KL 强度）、组大小 $G$、学习率、rollout batch 与 mini-batch 划分、RL 总步数（只说 “thousands of steps”）。均未披露。
- **U3** V3.2 的优势是否真的不除标准差：§3.1 前一句说 “normalizing the outcome reward within each group”，后一句公式只减均值。未见训练代码可核对。
- **U4** Keep Sampling Mask 是否对 $\pi_\theta$ 在保留集合上重新归一化；RL 采样用的 top-p / top-k 具体取值（部署推荐 top_p 0.95 不等于 RL 用值）。
- **U5** Keep Routing 时门控值 $g_{i,t}$ 是否用当前参数重算，还是连门控权重一起回放；无辅助损失均衡的偏置 $b_i$ 在 RL 中是否更新。
- **U6** Keep Routing（V3-0324，2025-03）与 GSPO 的 Routing Replay（2025-07 论文）谁先提出；二者实现是否相同。
- **U7** 长度惩罚的函数形式；“length constraint reward model” 是规则还是学出来的模型；Speciale 的 “reduced length penalty” 减了多少。
- **U8** 生成式奖励模型与 rubric 的细节（模型规模、rubric 如何生成）。
- **U9** “pre-training cost” 的基数（V3 的 14.8T 预训练？是否含 V3.1 续训 839B 与 DSA 续训 945.8B？）以及 RL 的绝对 GPU 小时。
- **U10** V3.1 自己的后训练配方。V3.1 没有论文；EXP 只说 V3.2-Exp 沿用 V3.1-Terminus 的流水线（C13），V3.1（08-21）是否已采用专家蒸馏 + 单阶段混合 RL，没有一手来源。
- **U11** Speciale 的 IMO/CMO 证明由谁评分（MV2 说由其数学专家评 MV2 的证明；V32 对 Speciale 没写）；CMO 2025 与 CMO 2024 的官方金牌线未能找到一手来源，故 “102/126 = 金牌”、“93/126 = 金牌” 只能按论文自述。
- **U12** E8M0 的指数偏置 127 与 “无零值” 等编码细节：OCP MX v1.0 规范 PDF 访问返回 403，未能直接读取。偏置 127 目前的依据是 DeepSeek 代码 `fast_pow2` 的位运算（与 FP32 指数偏置相同）和 MX 论文 “superset of the representable exponents of FP32” 的表述；PTX 文档只说明 8 位指数、0 位尾数、无无穷、NaN 为 0xff。
- **U13** ICPC WF 2025 “第 2 名”：官方记分牌第 2、3 名都是 10 题，Speciale 的名次取决于罚时，而论文只给提交次数、未给解题时间。
- **U14** DeepSeekMath-V2 自动标注的 $n,m,k$ 取值、总迭代轮数、每轮数据量。
- **U15** DeepSeek 官方微信公众号关于 “UE8M0 FP8是针对即将发布的下一代国产芯片设计” 的原话：只从腾讯新闻（转载扬子晚报，2025-08-21）等二手报道读到，未访问到微信原文。课文如引用须标 “据媒体报道”。
- **U16** ~~V32 图 1 HMMT 组 V3.2 的柱标签是否确为 90.2~~ 已解决：由 PDF 矢量路径算得柱高 90.19（E19）。仍未做的是渲染后目视；另外，作者是有意画 Nov 还是笔误，无从得知。
- **U17** V32 图 3 两条成本曲线的具体数值（矢量图，文本层只有坐标轴刻度）。
- **U18** HF 页面上 V3.2 本体的参数统计（685,355,329,792，F32 仅 1,075,456）与 Speciale 不一致的原因；实测分片中缩放因子仍是 F32，推测是 HF 元数据问题。核查补充：两者总数之差 41,591,584 恰好等于 F32 计数之差（42,667,040 − 1,075,456），BF16 与 F8 计数完全相同，支持 “HF 对 V3.2 本体漏计了大部分 F32 张量” 这一解释，但根因仍未确认。
- **U19** V32 图 5、图 6 的曲线数值：图 5 只能目测（F11 的 “约” 值），图 6 只有正文给出的端点。
- **U20** V3.2 / DeepSeekMath-V2 是否有期刊或会议版本（截至 2026-09-24 arXiv 只有 v1；未检索期刊数据库）。
- **U21** V3.2 的 DSML 格式与 API 的 OpenAI 兼容格式之间，服务端如何转换；V3.1 新闻页 “Strict Function Calling supported in Beta API” 的实现细节。
- **U22** 表 5 “GPT-5-Thinking” 与表 2/表 3 “GPT-5 High” 是否同一配置。
- **U23** V3.1 模型卡 “V3.1-Think achieves comparable answer quality to DeepSeek-R1-0528, while responding more quickly” 中 “更快” 的量化数据（模型卡只给了准确率，没有给 token 数或时延）。

---

## 9. 参考文献卡片草稿

格式与 `lesson-markdown` SKILL 的 ```refs 块一致，可直接粘贴；`id` 即正文 `[@id]` 的键。论文 7 条，制品 13 条。

```refs
- id: dsv32
  title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
  a: DeepSeek-AI (2025), preprint. arXiv:2512.02556v1
  url: https://arxiv.org/abs/2512.02556
  finding: 给出 V3.2 的 DSA 续训、专家蒸馏 + 单阶段混合 GRPO，以及无偏 KL、off-policy 序列掩码、Keep Routing、Keep Sampling Mask 四项稳定化改动和大规模智能体任务合成。
  matters: 本课几乎所有 RL 公式（式 5–9）、智能体任务数字和 Speciale 竞赛成绩都以它为唯一一手来源。
- id: dsmathv2
  title: "DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning"
  a: Shao, Luo, Lu, Ren, Hu, Ye, Gou, Ma, Zhang (2025), preprint. arXiv:2511.22570v1
  url: https://arxiv.org/abs/2511.22570
  finding: 训练能给出问题清单与三档分数的证明验证器，加上元验证与 “证明 + 自评” 加权奖励，得到 IMO 2025 5/6、Putnam 2024 118/120 的证明生成器。
  matters: 它是 Speciale 证明能力的来源（V32 称并入其数据与奖励），也是本课 “结果奖励之外如何给奖励” 的例子。
- id: dsv3
  title: "DeepSeek-V3 Technical Report"
  a: DeepSeek-AI (2024), preprint. arXiv:2412.19437v2
  url: https://arxiv.org/abs/2412.19437
  finding: V3 的两阶段上下文扩展各 1000 步（32K×1920、128K×480），后训练仅 5K GPU 小时。
  matters: 是 V3.1 “10 倍 / 3.3 倍” 续训和 V3.2 “后训练超过 10%” 的比较基线。
- id: dsmath
  title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
  a: Shao et al. (2024), preprint. arXiv:2402.03300v3
  url: https://arxiv.org/abs/2402.03300
  finding: 提出 GRPO，优势取组内标准化奖励，并用 Schulman 的 k3 估计 KL。
  matters: 本课逐项对比 V3.2 对 GRPO 的改动（优势不除 std、k3 加重要性修正）时的原始版本。
- id: tang2025kl
  title: "On a few pitfalls in KL divergence gradient estimation for RL"
  a: Tang, Munos (2025), preprint. arXiv:2506.09477v1
  url: https://arxiv.org/abs/2506.09477
  finding: 证明把 k3 当损失直接求导得到的是 ∇KL(π_ref, π) 而非 ∇KL(π, π_ref)。
  matters: 为 V3.2 “无偏 KL 估计” 的动机提供独立的数学依据，课文推导梯度方向时引它。
- id: gspo
  title: "Group Sequence Policy Optimization"
  a: Zheng et al. (2025), preprint. arXiv:2507.18071v2
  url: https://arxiv.org/abs/2507.18071
  finding: 报告 MoE 在一次 RL 更新后约 10% 激活专家发生变化，GRPO 需要 Routing Replay，GSPO 用序列级比率免去它。
  matters: 说明 Keep Routing 解决的问题有多大，以及它不是唯一解法。
- id: mx2023
  title: "Microscaling Data Formats for Deep Learning"
  a: Rouhani et al. (2023), preprint. arXiv:2310.10537v3
  url: https://arxiv.org/abs/2310.10537
  finding: 定义 MX 块格式：k=32 个元素共享一个 E8M0（纯指数）缩放。
  matters: 解释 V3.1 “兼容 microscaling” 指什么，也是指出 DeepSeek 块大小仍为 128 的对照依据。
- id: hf-v31
  kind: artefact
  title: "DeepSeek-V3.1 model card, config.json, chat_template.jinja, tokenizer.json"
  a: DeepSeek-AI (2025), Hugging Face
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
  finding: 一个模型靠聊天模板切换思考/非思考，630B + 209B 续训，权重与激活用 UE8M0 缩放，工具调用在非思考模式。
  matters: V3.1 没有论文，这是其混合思考模板、续训 token 数和 UE8M0 说法的唯一一手来源。
- id: hf-v31t
  kind: artefact
  title: "DeepSeek-V3.1-Terminus model card and inference code"
  a: DeepSeek-AI (2025), Hugging Face
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus
  finding: 修复语言混杂并加强代码/搜索智能体，并承认 o_proj 缩放不符合 UE8M0。
  matters: 是 V3.2 的直接前身（V32 称唯一架构差别是 DSA），其 o_proj 声明是本课检查点实测的起点。
- id: dsv32exp
  kind: artefact
  title: "DeepSeek-V3.2-Exp technical report (DeepSeek_V3_2.pdf)"
  a: DeepSeek-AI (2025), GitHub
  url: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
  finding: V3.2-Exp 沿用 V3.1-Terminus 的后训练流水线，专家领域为五个，并说明单阶段混合 RL 取代多阶段 RL。
  matters: 用来标出 V3.2 论文与其前身在专家领域数量上的版本漂移。
- id: hf-v32
  kind: artefact
  title: "DeepSeek-V3.2 / V3.2-Speciale model cards, encoding_dsv32.py, inference kernels"
  a: DeepSeek-AI (2025), Hugging Face
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
  finding: 新的 DSML 工具调用格式、developer 角色、按最后一条 user 消息丢弃思考的编码代码，以及 2 的幂缩放的量化内核。
  matters: 让课文能用代码而不只是论文文字来讲思考上下文管理和 UE8M0 取整。
- id: ds-news
  kind: artefact
  title: "DeepSeek API news: V3.1 (250821), V3.1-Terminus (250922), V3.2-Exp (250929), V3.2 (251201)"
  a: DeepSeek (2025), API docs
  url: https://api-docs.deepseek.com/news/news250821
  finding: 给出发布日期、840B 续训合计、1,800+ 环境与 85k+ 指令，以及 Speciale 在官方服务中只走 API、无工具调用、至 2025-12-15 的临时端点（权重另在 HF 开源）。
  matters: 时间线与 API 层面行为（deepseek-chat/reasoner 映射、降价、端点期限）只在这里有一手记录。
- id: ds-thinking-guide
  kind: artefact
  title: "DeepSeek API guide \"Thinking Mode\" (Wayback 2025-12-09 snapshot vs live 2026-09)"
  a: DeepSeek (2025–2026), API docs
  url: http://web.archive.org/web/20251209045516/https://api-docs.deepseek.com/guides/thinking_mode
  finding: V3.2 时代要求同一问题内回传 reasoning_content、新问题开始时删除；现行版改为带 tools 时回传全部历史思考。
  matters: 用来防止课文把 V4 时代的 API 行为错套到 V3.2 上。
- id: deepgemm
  kind: artefact
  title: "DeepGEMM README"
  a: DeepSeek-AI (2025), GitHub
  url: https://github.com/deepseek-ai/DeepGEMM
  finding: SM90 用 FP32 缩放因子，SM100 要求 4 个 UE8M0 打包进一个 int。
  matters: 解释为什么 UE8M0 是面向新硬件的格式选择，而检查点中缩放仍存为 FP32。
- id: ptx-ue8m0
  kind: artefact
  title: "NVIDIA PTX ISA, \"Alternate Floating-Point Data Formats\""
  a: NVIDIA (2025), documentation
  url: https://docs.nvidia.com/cuda/parallel-thread-execution/
  finding: ue8m0 是 8 位无符号、8 位指数 0 位尾数、无无穷、NaN 为 0xff 的格式。
  matters: 课文定义 UE8M0 时可引的一手硬件文档（OCP 规范原文未能访问）。
- id: schulman-kl
  kind: artefact
  title: "Approximating KL Divergence"
  a: Schulman (2020), blog
  url: http://joschu.net/blog/kl-approx.html
  finding: 提出 k1、k2、k3 三种单样本 KL 估计，其中 k3=(r−1)−log r 无偏且恒非负。
  matters: DSM 与 V32 都以 “Schulman, 2020” 引用它，是 k3 的出处。
- id: ckpt-scan
  kind: artefact
  title: "Author's scan of DeepSeek V3 / V3.1 / V3.1-Terminus / V3.2 safetensors scale tensors"
  a: 本课研究笔记 (2026-09-24)
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main
  finding: V3.1 起除 o_proj 外的 FP8 权重缩放均为 2 的幂，o_proj 始终不是；V3 全部不是。
  matters: 把模型卡的 UE8M0 说法落到可复现的检查点事实上，并暴露 o_proj 例外持续到 V3.2。
- id: imo2025
  kind: artefact
  title: "66th IMO 2025 – Individual Results"
  a: International Mathematical Olympiad (2025)
  url: https://www.imo-official.org/results/individual/year/2025/
  finding: 2025 年金牌 72 人，最低 35 分。
  matters: 说明 Speciale 与 DeepSeekMath-V2 的 35/42 恰好压在金牌线上。
- id: ioi2025
  kind: artefact
  title: "IOI 2025 Statistics and Results"
  a: International Olympiad in Informatics (2025)
  url: https://stats.ioinformatics.org/olympiads/2025
  finding: 金牌线 438.30 分（28 枚），人类第 9、10 名为 502.78 与 491.65。
  matters: 用来复核 V32 表 4 “492/600、第 10 名” 的说法。
- id: icpc2025
  kind: artefact
  title: "49th ICPC World Finals Baku scoreboard"
  a: ICPC (2025)
  url: https://worldfinals.icpc.global/scoreboard/2025/finals/
  finding: 12 题，第 1 名解 11 题，第 2、3 名各解 10 题，金牌 4 队。
  matters: 说明 Speciale 的 10/12 与 “第 2 名” 之间还隔着一个未公开的罚时比较。
```

---

## 10. 补充核实（写课时追加）

（写课作者自行核实的新事实，按“来源 + 原文引文 + 位置”追加。以下由第 15 课作者于 2026-09-24 核实。）

**S1 E8M0 的指数偏置 127（部分解决 §8 U12）。** 来源：NVIDIA CUDA Math API Reference，`__nv_fp8_e8m0` 结构说明页 https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e8m0.html （WebFetch 读取）。原文：
> "This structure implements the datatype for handling 8-bit scale factors of `e8m0` kind: interpreted as powers of two with biased exponent. Bias equals to 127, so numbers 0 through 254 represent 2^-127 through 2^127. Number `0xFF` = 255 is reserved for NaN."

用于课文 (15.5) 与参考卡 `cuda-e8m0`。OCP MX v1.0 规范原文仍未读到；“偏置 127”现在有 NVIDIA 一手文档为据。

**S2 V3.2 §3.2.3 通用智能体合成流程的原文（补充 C39）。** 来源：arXiv 2512.02556v1 LaTeXML HTML（curl 下载后去标签抽文本，逐字）。原文：
> "The synthesis workflow primarily consists of environment and toolset construction, task synthesis, and solution generation. Specifically, the workflow proceeds as follows. 1. Given a task category (e.g., planning a travel itinerary) and a sandbox equipped with a bash and a search tool, the agent first uses these tools to generate or retrieve relevant data from the Internet and store them in the sandbox database. 2. The agent then synthesizes a set of task-specific tools, each implemented as a function. 3. To create tasks that are both challenging and automatically verifiable, the agent initially proposes a simple task based on the current database, along with its solution and verification functions implemented in Python. The solution function is restricted to invoking tool functions or performing logical computations, and cannot call other functions or directly access the database, ensuring the task can only be solved through the tool interface. Additionally, the results produced by the solution function must be validated by the verification function. If the solution is not validated, the agent will modify the solution or verification functions until the solution’s output passes the verification. The agent then iteratively increases the difficulty of the task and updates the corresponding solution and verification functions. During this iterative process, if the current toolset is not sufficient to solve the task, the agent will augment the toolset."

用于课文 §7 与图 15.12。

**S3 V3.2 §3.2.3 搜索智能体数据流水线的原文（补充 C37）。** 同上来源。原文（节选）：
> "We employ a multi-agent pipeline based on DeepSeek-V3.2 to generate diverse, high-quality training data. We first sample informative long-tail entities across diverse domains from large-scale web corpora. A question-construction agent then explores each entity using search tools with configurable depth and breadth parameters, consolidating the discovered information into question-answer pairs. Multiple answer-generation agents with heterogeneous configurations (different checkpoints, system prompts, etc.) produce diverse candidate responses for each proposed QA pair."

> "To complement these verifiable samples and better reflect real-world usage, we also augment the dataset with filtered instances from our existing helpful RL datasets, for which the search tool provides measurable benefits. We then develop detailed evaluation rubrics across multiple quality dimensions and employ a generative reward model to score responses based on these rubrics."

代码解释器一段：
> "We utilize Jupyter Notebook as a code interpreter to address complex reasoning tasks. To facilitate this, we curate a diverse set of problems spanning mathematics, logic, and data science, each requiring the model to leverage code execution capabilities to arrive at a solution."

**S4 V3.2 §4.4 三种上下文管理策略的定义（补充 C44、F12）。** 同上来源。原文：
> "(2) Discard-75% , which discards the first 75% tool call history in the trajectory to free up spaces; (3) Discard-all , which resets the context by discarding all previous tool call history (similar to the new context tool ( Anthropic, 2025b ) ). For comparison, we also implement a parallel scaling baseline, Parallel-fewest-step , which samples N independent trajectories and selects the trajectory with the fewest steps."

Summary 的定义（同段，前半句）：“summarizes the overflowed trajectory and re-initiates the rollout”。arXiv v1 的结果句：“For example, Summary extends the average steps to 364, achieving a performance improvement of up to 60.2. However, its overall efficiency is relatively low.”

**S5 V3.2 表 5 的完整数字（补充 F10、N18）。** 同上来源（WebFetch 读取表格，与 N18 的 V3.2-Exp 数值一致）：

| Pass@K | DeepSeek-V3.2-Exp | Sonnet-4.5 | Gemini-3.0 Pro | GPT-5-Thinking |
|---|---|---|---|---|
| 1 | 12% | 34% | 51% | 62% |
| 2 | 18% | 47% | 65% | 75% |
| 4 | 26% | 62% | 74% | 82% |

**S6 课文与图中笔者构造或推导的内容（不是事实，供审稿区分）。**
- 图 15.9（Keep Routing）的 16 个亲和度与扰动是笔者构造的示意；图中 “BF16 在 [0.5, 1) 的间距 2^-8” 是 BF16 格式（7 位尾数）的直接推论。
- §5 的三 token 分布算例（π_θ = (0.5, 0.3, 0.2)、π_ref = (0.4, 0.4, 0.2)、π_old = (0.2, 0.7, 0.1)）与练习 5–10 的数字是笔者设计的，全部在 Python 中复算（原始 k3 在旧样本上高估 45%：0.03658 对 0.02527）。
- §3 “UE8M0 最坏使正规数门槛升高一倍（少一个 binade）”与练习 12 “取最近的 2 的幂会让一半分块溢出”是笔者由 (15.6)(15.7) 推出的分析，V3.1 资料没有讨论。
- 数学复核（2026-09-24）改动了两处构造数据，使算例不再落在裁剪折点 $\rho=1-\varepsilon=0.8$ 上（该点左右导数不同，“有梯度”无定义）：N24 的 o3 第 4 个 token 的 $\pi_\theta$ 由 0.40 改为 0.42（$\rho=0.84$；$\bar D_{o3}$ 由 0.1038 变为 **0.0956**，两个 $\delta$ 下的掩码不变）；课文练习 8 的 o_a 第 2 个 token 由 0.4 改为 0.42（$\rho=0.84$；$\bar D_a$ 由 0.1348 变为 **0.1226**，原值按四位小数应为 0.1347）。§6 X3 的答案 (a) 里 o3 的 0.1038 相应作废。
- 同一次复核补充的推论（笔者推导）：MX 取 $2^{\lfloor\log_2a\rfloor-8}$ 与 DeepSeek 取 $2^{\lceil\log_2(a/448)\rceil}$ 只在 $a=m\cdot2^k$、$m>1.75$ 时相差一档（此时 MX 的最大码值 $256m>448$ 需饱和）；Python 在 $m\in[1,2)$ 上逐点核对无例外。
- 教学复核（2026-09-24）新增的笔者算例，均非论文内容：§3 “UE8M0 ≠ MXFP8” 框里 $a=3.96$ 的对照（MX 码值 506.9 饱和到 448、解码 3.5、误差 11.6%；DeepSeek $s=2^{-6}$、码值 253.4 舍入到 256、解码 4.0、误差 1.0%、$u=0.566$）；§6 Keep Sampling Mask 算例里“全词表比率 0.760 < 0.8 被裁、核内比率 0.844 > 0.8 不被裁”的对照（以 V3.2 在核内重新归一化为前提，论文未说明）；§7 的旅行行程玩具四元组（示意代码）；§7 用表 5 的 pass@1/pass@2 拟合“两类题”模型（$p=0.5$、$q=0.24$，预测 pass@4 = 0.225，实测 0.26）。

以下 S7–S11 由事实复核（2026-09-25）补充，均为复核者重新读取一手来源所得。

**S7 检查点缩放因子的完整补测（补全 N32、图 15.5，解决 §11.3 第一条）。** 方法同 N32：取各仓库 `model.safetensors.index.json` 定位分片，HTTP Range 读 safetensors 头，再只读目标 `weight_scale_inv` 张量的字节，用 `np.frexp` 判断尾数是否全为 0.5。脚本在复核者会话临时目录（`scan_scales.py`），不入库。结果（2 的幂所占比例；每个张量要么 1.000 要么 0.000，没有中间值）：

| 张量（层） | V3 | V3.1 | V3.1-Terminus | V3.2 |
|---|---|---|---|---|
| `self_attn.q_a_proj`（0、30、60） | 0.000 | 1.000 | 1.000 | 1.000 |
| `self_attn.kv_b_proj`（0、30、60） | 0.000 | 1.000 | 1.000 | 1.000 |
| `self_attn.o_proj`（0、30、60） | 0.000 | 0.000 | 0.000 | 0.000 |
| `mlp.down_proj`（0，稠密层） | 0.000 | 1.000 | 1.000 | 1.000 |
| `mlp.experts.7.down_proj`（30、60） | 0.000 | 1.000 | 1.000 | 1.000 |
| `mlp.shared_experts.up_proj`（30、60） | 0.000 | 1.000 | 1.000 | 1.000 |
| `self_attn.indexer.wq_b` / `wk`（30、60） | 无 | 无 | 无 | 1.000 / 1.000 |

全部张量 dtype 为 F32；形状：`q_a_proj` [12, 56]、`kv_b_proj` [256, 4]、`o_proj` [56, 128]、稠密 `down_proj` [56, 144]、`experts.7.down_proj` [56, 16]、`shared_experts.up_proj` [16, 56]、`indexer.wq_b` [64, 12]、`indexer.wk` [1, 56]。与 N32 的已测格子逐一一致；原先 “未测” 的 V3 与 Terminus 格子现已补齐，图 15.5 据此重画（不再有 “未测” 格）。

**S8 推理内核对 amax 的下限钳位（补充 §2.6）。** 来源：HF-V32 `inference/kernel.py`（`act_quant_kernel`，raw 文件 2026-09-25 读取；HF-V31T 同名文件同一行一致）。原文：

```python
T.reduce_absmax(x_local, amax_local, dim=1)
for i in T.Parallel(blk_m):
    amax_local[i] = T.max(amax_local[i], 1e-4)
    if round_scale:
        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
    else:
        s_local[i] = amax_local[i] * fp8_max_inv
```

量化步是 `T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)`。推论（复算）：$a\ge10^{-4}$ 时 $a/448\ge2.23\times10^{-7}$，$\lceil\log_2\rceil\ge-22$，全 0 分块得 $s=2^{-22}$；`fast_log2_ceil` 的输入永远是 FP32 正规数。课文 §3 “逐行读”一段据此改写（原稿写“未见特殊处理”）。

**S9 V3 报告中已有的 2 的幂缩放（精确化 E14）。** 来源：arXiv 2412.19437v2 §3.3（`arxiv.py grep`）。原文：
> "(1) Inputs of the Linear after the attention operator. ... To avoid introducing extra quantization error, all the scaling factors are round scaled, i.e., integral power of 2."

> "we quantize the activation before MoE up-projections into FP8 and then apply dispatch components, which is compatible with FP8 Fprop in MoE up-projections. Like the inputs of the Linear after the attention operator, scaling factors for this activation are integral power of 2. A similar strategy is applied to the activation gradient before MoE down-projections."

即两类激活加一类激活梯度；课文 §3 原写 “两类激活的缩放 round scaled”，已按原文改为三处并分别引出原话。

**S10 V3.2 §3.2.3 与 §4.3 的两处原文（纠正课文 §7 一处表述）。** 来源：arXiv 2512.02556v1（`arxiv.py grep`）。
- §3.2.3 首段紧接 C36 的下一句：“For other tasks, the environment and prompts are both synthetically constructed.”——所以 “真实环境、非真实用户” 只适用于搜索、代码、代码解释器三类；课文原写 “这四类都是”，已改为 “前三类”，并补引此句。
- §4.3 图 5 的实验设置：“To exclude the effects of long CoT and other RL data, we conduct RL only on synthetic agentic tasks in non-thinking mode. We then compare the model with DeepSeek-V3.2-SFT and DeepSeek-V3.2-Exp, where DeepSeek-V3.2-Exp is trained with RL only in search and code environments.”——“只用代码与搜索做 RL” 的对照组是另一个模型 V3.2-Exp，合成任务 RL 在非思考模式下进行。课文 §7 已补这两点。
- 附录 B 中含 “with a maximum of 20 code executions” 的模板标题为 “Reasoning Required Agent System Prompt”（即思考中调用工具的那一种）。

**S11 DeepSeekMath-V2 的三处原文（补充 C52、C56、C58）。** 来源：arXiv 2511.22570v1（`arxiv.py grep`）。
- §2.1.1：“…prioritizing math olympiads, team selection tests, and post-2010 problems explicitly requiring proofs, totaling 17,503 problems. This problem set is denoted as 𝒟_p.”（课文原写 “17,503 道证明题”，已改为按原文描述题库构成。）
- §2.2.2 要点原句：“Faithful acknowledgment of errors is rewarded over false claims of correctness.”
- §3.1：“We employed Group Relative Policy Optimization (GRPO) (Shao et al., 2024) for reinforcement learning, iteratively optimizing proof verification and generation capabilities as described in Section 2.”
- 表 1 文本层：IMO 2025 “P1, P2, P3, P4, P5 / 83.3%”；CMO 2024 “P1, P2, P4, P5, P6 / 73.8%”；Putnam 2024 “A1 ∼ B4, B5, B6 / 98.3%”。表中只标完全解出与部分分（灰底 / 下划线，文本层看不出哪一题是下划线），**没有逐题分数**：图 15.14 中 DeepSeekMath-V2 的每题 7 分是由 “完全解出” 折算，CMO 2024 的 93 分由 73.8% 反推，图注已写明。


---

## 11. 核查记录（对抗式复核，2026-09-24）

复核者独立重取一手来源，没有沿用研究笔记的缓存：arXiv LaTeXML 全文（`arxiv.py text`）、arXiv v1 PDF 与 HF `assets/paper.pdf`（pypdf 抽文本层与矢量路径）、HF raw README/config/generation_config/tokenizer/`inference/*.py`/`encoding/*`、HF API、DeepSeek 新闻页与 Thinking Mode 指南（Wayback 2025-12-09 快照与现行版）、IMO/IOI/ICPC 官方成绩页，以及对 V3、V3.1、V3.2 检查点缩放张量的 HTTP Range 读取。所有派生数字在 Python 中重算。

### 11.1 逐项核对（均为 “确认”，除非另注）

| # | 论断 | 核对方式与来源 | 结果 |
|---|---|---|---|
| 1 | V32 式 (5)(6)(7)(8)(9) 的 LaTeX 转录（§2.1–2.3） | 与 2512.02556v1 LaTeXML 源逐符号对比 | 逐符号一致 |
| 2 | V32 优势 $\hat A_{i,t}=R_i-\text{mean}(\boldsymbol R)$，前句却说 “normalizing”（C21、E2） | V32 §3.1 原文 | 确认；DSM §4.1.2 确为除 std（2402.03300v3） |
| 3 | DSM 式 (4) 原始 k3 与 “guaranteed to be positive” | 2402.03300v3 §4.1.1 | 一致 |
| 4 | TM 式 (3)：对 k3 求导的期望为 $\nabla\KL(\pi_{ref},\pi)$ | 2506.09477v1 §3 “Variance reduced estimate”；§2.1 定义 | 一致；补注 TM 引 “Schulman, 2017”（C24） |
| 5 | N21–N23 梯度权重与 5 元 softmax 数值 | Python 重算；用 `default_rng(0)` 连抽两组 5 维正态作 logits 与参考 logits，四个梯度向量逐位复现 | 一致 |
| 6 | Keep Routing “since DeepSeek-V3-0324”；GSPO Routing Replay 约 10% 专家变化 | V32 §3.1；2507.18071v2 §5.3 | 一致 |
| 7 | 表 1 四类任务数 24,667 / 50,275 / 4,417 / 5,908，合计 85,267 与占比 | V32 表 1；Python | 一致 |
| 8 | 1,827 个合成环境、4,417 个任务、pass@100 非零过滤 | V32 §3.2.3 | 一致 |
| 9 | “exceeding / already exceeds 10% of the pre-training cost” | V32 §1 第 3 段、§4.1 | 一致 |
| 10 | 专家领域 V32 为 6、EXP 为 5（E9） | V32 §3；EXP PDF §2.2（pypdf） | 一致 |
| 11 | 表 4：IMO 7,7,7,7,7,0=35；CMO 18,18,9,21,18,18=102；IOI 100,82,72,100,55,83=492；ICPC 10 题、提交 14 次 | V32 表 4；Python | 一致 |
| 12 | IMO 2025 金牌线 35（72 人）、银 28–34（104）、铜 19–27（145） | imo-official.org 个人成绩页内嵌 JSON，只做计数聚合 | 一致 |
| 13 | IOI 2025：330 人，金牌 28 枚、线 438.30，第 9/10 名 502.78/491.65 | stats.ioinformatics.org 统计页与成绩页 | 一致 |
| 14 | ICPC WF 2025：12 题，第 1 名 11 题，第 2、3 名各 10 题，4 枚金牌 | worldfinals.icpc.global 记分牌 | 一致 |
| 15 | 表 3 token 与 “平均 1.67 倍”；表 2、表 5、表 9 摘录（F13、F14） | V32 各表 | 一致；N17 补注两种平均法 |
| 16 | BrowseComp 51.4 / 67.6、20%+ 超长、80% 触发；Terminal Bench 46.4（Claude Code）/ 39.3（Terminus 非思考） | V32 §4.1、§4.4、表 9 标题 | 一致 |
| 17 | HF PDF 与 arXiv v1 的差异（E17） | 两份 PDF 逐句 diff | 恰为两处（§4.4 那句 + 引言多引 Qwen3），一致 |
| 18 | 图 1 HMMT 组 V3.2 柱（E19） | 读取 PDF 表单对象填充路径 | **由 “疑似” 升级为 “确认”**，柱高 90.19 |
| 19 | V3.1 模型卡：630B / 209B、“10-fold”“3.3x”、UE8M0、FP32 偏置、工具调用仅非思考、模板字符串 | HF raw README | 一致；V3 §4.3 的 1000 步、1920/480 批也一致，N1–N4 复算一致 |
| 20 | config.json / generation_config.json 字段（§3.2） | HF raw JSON（V3、V3.1、Terminus、V3.2、Speciale、Math-V2） | 一致（Terminus 确无 `scale_fmt`，有 `attn_module_list_cfg`） |
| 21 | 特殊 token 128796–128799 在 V3→V3.1 间改变，均为 128,000 + 818 | 两份 tokenizer.json 逐 id 比较 | 一致，且只有这 4 个 id 变化 |
| 22 | HF 参数计数 684,531,386,000 / 685,396,921,376 / 685,355,329,792 与 indexer 13,960,248×62 | HF API `safetensors`；Python | 一致；N34 补 dtype 拆分核对 |
| 23 | 检查点缩放：V3 全不是 2 的幂；V3.1、V3.2 的 `q_a_proj`、`kv_b_proj` 全是，`o_proj` 全不是；dtype F32 | 独立 Range 读取第 0、30 层 | 一致（Terminus、第 60 层、专家与 indexer 张量未重测） |
| 24 | `fast_log2_ceil`/`fast_pow2`/`fast_round_scale` 转录、`fp8_max=448`、`group_size=128` | HF-V32 `inference/kernel.py` | 逐行一致；N29–N30 复算一致 |
| 25 | MV2 式 (1)–(6)、α=0.76/β=0.24、0.85→0.96、17,503 题、表 1 “Points” 83.3/73.8/98.3%、Putnam 118/120 vs 90 | 2511.22570v1 | 一致；N36–N41 复算或文本层复核一致 |
| 26 | Thinking Mode 指南两版引文（E15） | Wayback 快照与现行页 | 一致 |
| 27 | 新闻页：840B、1,800+ 与 85k+、“first model to integrate thinking directly into tool-use”、Speciale 端点至 2025-12-15 15:59 UTC | api-docs.deepseek.com/news/* | 一致 |
| 28 | 全部 112 条英文引文 | 脚本把规范化后的引文与上述全部源文本比对 | 106 条逐字命中；其余 6 条差异只在数学符号转写或 HTML 链接包裹处（文件头已声明），文字部分命中 |

### 11.2 本次修改

1. **§0.1 作者数 264 → 263。** arXiv 元数据有 264 个作者条目，其中第一条是机构 “DeepSeek-AI”；附录 E 列出 212 + 27 + 24 = 263 个名字。来源：arXiv API `id_list=2512.02556`；V32 附录 E。
2. **§4.6 V3.1-Terminus 的 AIME 2025 由 “—” 补为 88.4。** 来源：EXP 表 1 Terminus 列。
3. **§4.6 SWE Verified 行标出模式。** V3.1 的 66.0 在模型卡 “V3.1-NonThinking” 列（Thinking 列为 “-”），V3.2 的 73.1 是思考模式；同模式比较应为 66.0 → 72.1（V32 表 9）。Terminal-bench 31.3 也补注为非思考列，并补 V3.2-Exp 的 37.7。来源：HF-V31 Evaluation 表；EXP 表 1；V32 表 9。原表把不同模式放在同一行，是 “相邻论断替换” 型缺陷。
4. **E19 / U16 / §5.1 图 1 行：由 “疑似” 改为 “确认”。** 依据是 PDF 矢量路径算出的柱高（见 11.1 #18）。
5. **E23 与 `ds-news` 参考卡：更正 “Speciale 长期可用——错” 的过度表述。** 到期的只是官方 API 临时端点；新闻页的 “API-only for now” 指官方服务只走 API，同一页 “Open Source Release” 给出了 HF 权重（MIT）。来源：NEWS 251201；HF API（DeepSeek-V3.2-Speciale，license:mit）。
6. **§3.7 许可：“V3.x 系列为 MIT” 改得更精确。** V3 原版权重受 DeepSeek Model License 约束，代码为 MIT。来源：DeepSeek-V3 README §7；HF API license 标签。
7. **§2.5 补代码细节。** `drop_thinking` 只在思考模式下执行；位于最后一条 user/developer 消息之前的 `developer` 消息会被整条丢弃。来源：HF-V32 `encoding/encoding_dsv32.py` 中的 `encode_messages`、`drop_thinking_messages`、`find_last_user_index`。
8. **§0.3 补 V3-0324 的新闻页日期。** 新闻页标为 2025/03/25，HF 上传时间为 03-24。
9. **补注（不改结论）。** E11 补 CMO 2025 为英文译本评测（V32 §4.2 脚注 5）；N17 补两种平均法；C24 补 TM 与 DSM/V32 对 Schulman 博客年份的不同引法；N34 补 dtype 拆分；U18 补 “总数差 = F32 差”。

### 11.3 本次未能独立复核（沿用研究笔记，未改动）

- V3.1-Terminus 的检查点缩放，以及第 60 层、`experts.7.down_proj`、`shared_experts.up_proj`、indexer 张量的 2 的幂比例：只重测了 V3/V3.1/V3.2 第 0、30 层的 `q_a_proj`、`o_proj`、`kv_b_proj`。
- QQ 二手报道（U15）、OCP MX 规范中的偏置 127（U12）：本次也没能取到一手原文。
- V32 图 3、图 5、图 6 的曲线读数（U17、U19）。
