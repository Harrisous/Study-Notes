# 第 13 课研究笔记：DeepSeek-R1——用强化学习激发推理

> 供 `deepseek/src/lesson-13-r1-reasoning.md` 的作者使用。建立于 2026-09-24。
> 方法：按 `.cursor/skills/academic-research/SKILL.md` 的 plan → search → verify → reconcile → aggregate 流程。
> arXiv 全文用 `arxiv.py text` 分别拉取 **v1 与 v2** 的 LaTeXML 渲染后逐段阅读；Nature 正文取自 nature.com 文章页 HTML；
> Nature 补充信息（Supplementary Information, SI）与同行评审文件（Peer Review File）取自 Springer 的 PDF，经 `pdftotext` 转文本后检索。
> 论文图的数据点是**从 arXiv v2 的 SVG 源文件里直接解析出来的**（matplotlib → PDF → SVG 的路径坐标，按网格线标定），
> 不是目测；个别图（第 6、7 节标注“目测”的）是从渲染后的 PNG 读的近似值。
> 所有派生数字都在 Python 里重算过，表达式见第 4 节。引文保持英文原文。

---

## 0. 覆盖范围与所读版本

### 0.1 一手来源

| 键 | 标题 | 作者 | 编号 / 版本 | 发表状态 | 日期 |
|---|---|---|---|---|---|
| `r1-arxiv` | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI（arXiv 元数据列 Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang 等共 199 位） | arXiv:2501.12948，**读了 v1 和 v2**（v2 为截至 2026-09-24 的最新版） | 预印本；arXiv 元数据 journal-ref 已填 Nature 645, 633–638 (2025) | v1：2025-01-22 15:19 UTC；v2：2026-01-04 03:57 UTC（arXiv Submission history） |
| `r1-nature` | DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning | Guo, D., Yang, D., Zhang, H. et al.（通讯作者 Wenfeng Liang） | **Nature 645 (issue 8081), 633–638**，doi:10.1038/s41586-025-09422-z | 同行评审期刊论文，开放获取 CC BY 4.0 | Received 14 Feb 2025；Accepted 17 Jul 2025；**Published 17 Sep 2025**；Issue date 18 Sep 2025 |
| `r1-si` | Supplementary Information for: DeepSeek-R1 Incentivizes Reasoning in LLMs via Reinforcement Learning | DeepSeek-AI Team | Nature 文章的 SI（41586_2025_9422_MOESM1_ESM.pdf，83 页） | 随 Nature 论文发表，“In the format provided by the authors and unedited” | 2025-09-17 |
| `r1-peer-review` | Peer Review File | Nature | 41586_2025_9422_MOESM2_ESM.pdf | 公开的审稿意见与作者答复 | 2025-09-17 |
| `deepseekmath` | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Shao, Wang, Zhu, Xu, Song, Bi 等 | arXiv:2402.03300 v3 | 预印本 | 2024-02-05（v1） |
| `deepseek-v3` | DeepSeek-V3 Technical Report | DeepSeek-AI | arXiv:2412.19437 v2 | 预印本 | 2024-12-27（v1） |
| `dr-grpo` | Understanding R1-Zero-Like Training: A Critical Perspective | Liu, Chen, Li, Qi, Pang, Du 等 | arXiv:2503.20783 v2 | 预印本（发表会议未核实） | 2025-03-26（v1） |
| `schulman-kl` | Approximating KL Divergence | John Schulman | 个人博客 joschu.net/blog/kl-approx.html | 博客（artefact） | Posted 2020/03/07 |
| `gao-rm-overopt` | Scaling Laws for Reward Model Overoptimization | Gao, Schulman, Hilton | arXiv:2210.10760 v1 | 预印本（R1 用它支撑“reward hacking”） | 2022-10-19 |
| `lightman-prm` | Let's Verify Step by Step | Lightman, Kosaraju, Burda, Edwards, Baker, Lee 等 | arXiv:2305.20050 v1 | 预印本（R1 引作 PRM 代表作） | 2023-05-31 |
| `ppo` | Proximal Policy Optimization Algorithms | Schulman, Wolski, Dhariwal, Radford, Klimov | arXiv:1707.06347 v2 | 预印本 | 2017-07-20 |
| `gae` | High-Dimensional Continuous Control Using Generalized Advantage Estimation | Schulman, Moritz, Levine, Jordan, Abbeel | arXiv:1506.02438 v6 | ICLR 2016（会议信息未在本次核实） | 2015-06-08 |

### 0.2 制品（没有论文的来源，kind: artefact）

| 键 | 内容 | URL |
|---|---|---|
| `r1-card` | Hugging Face 模型卡 deepseek-ai/DeepSeek-R1（README.md，与 GitHub README 的正文相同） | https://huggingface.co/deepseek-ai/DeepSeek-R1 |
| `r1-github` | GitHub deepseek-ai/DeepSeek-R1 README（Usage Recommendations、Official Prompts、License） | https://github.com/deepseek-ai/DeepSeek-R1 |
| `r1-0528-card` | Hugging Face 模型卡 deepseek-ai/DeepSeek-R1-0528 | https://huggingface.co/deepseek-ai/DeepSeek-R1-0528 |
| `r1-config` | R1 / R1-Zero / R1-0528 的 `config.json`（三者逐字节相同，见 §3.9） | https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/config.json |
| `r1-distill-configs` | 6 个 Distill 模型与 R1-0528-Qwen3-8B 的 `config.json` | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/raw/main/config.json 等 |
| `r1-news` | DeepSeek API 文档新闻 “DeepSeek-R1 Release 2025/01/20” | https://api-docs.deepseek.com/news/news250120/ |
| `r1-0528-news` | DeepSeek API 文档新闻 “DeepSeek-R1-0528 Release 2025/05/28” | https://api-docs.deepseek.com/news/news250528 |
| `nature-nv` | Nature News & Views “AI can learn to show its workings through trial and error”（Daphne Ippolito, Yiming Zhang, 17 Sept 2025） | 只核实了标题/作者/日期（Nature 文章页“Associated content”），正文未读 |

### 0.3 三个版本的关系（写课前必须弄清）

1. **v1（2025-01-22）**：模型发布（2025-01-20，`r1-news`）两天后的技术报告，约 6 万字符正文。R1-Zero 的数字是 **AIME 2024 pass@1 71.0%**。
   没有训练超参数、没有训练成本、没有安全报告。
2. **Nature 论文（2025-09-17）**：正文完全重写（摘要、引言都换了），Methods 首次给出 GRPO 超参数、奖励模型细节；
   R1-Zero 继续训练到 10,400 步，数字变成 **77.9%**。这个数不在训练曲线上，对应哪个检查点、用哪种评测协议，见 §8 第 21 条。SI 83 页，含训练成本、数据配方、安全报告等 11 节。
3. **arXiv v2（2026-01-04）**：= Nature 正文 + SI 的合并版（22.9 万字符）。附录 A–J 与 SI 第 1–10 节一一对应，
   编号规律：**arXiv v2 附录表 n = SI 表 n−3**（如 v2 Table 7 训练成本 = SI Table 4），**arXiv v2 图 n = SI 图 n−2**
   （v2 Fig. 6 reward hacking = SI Fig. 4；v2 Fig. 9 反思词 = SI Fig. 7 = Nature Extended Data Fig. 1）。
   v2 附录有 29 张表（Table 4–32）、16 张图（Fig. 3–18），恰好对应 SI 声称的 “Supplementary Tables 1–29 and Supplementary Figs. 1–16”。

编号对照（写课引用时用哪个版本就写哪个编号）：

| 内容 | v1 | v2 | Nature 正文 / SI |
|---|---|---|---|
| R1-Zero 模板 | Table 1 | Table 1 | 正文 “DeepSeek-R1-Zero” 节内联引文（无表号） |
| aha moment | Table 3 | Table 2 | Table 1 |
| R1-Zero 与 o1 对比 | Table 2 | （删除） | — |
| AIME 训练曲线 | Figure 2 | Figure 1(a) | Fig. 1a |
| 回答长度曲线 | Figure 3 | Figure 1(b) | Fig. 1b |
| R1 多阶段流程图 | （无图） | Figure 2 | Fig. 2 |
| 各阶段结果（R1-Zero/Dev1/Dev2/Dev3/R1） | （无） | Table 3 | Table 2 |
| R1 与其他模型对比 | Table 4 | Table 8 | SI Table 5 |
| 蒸馏模型结果 | Table 5 | Table 15 | SI Table 12 |
| 蒸馏 vs RL（Qwen-32B） | Table 6 | Table 16 | SI Table 13 |
| PPO vs GRPO 结构图 | — | Figure 3 | SI Fig. 1 / Extended Data Fig. 2 |
| 训练成本 | — | Table 7（§B.4.4） | SI Table 4（§2.4.4） |
| 反思词 / “wait” 频次 | — | Figure 9 | SI Fig. 7 / Extended Data Fig. 1 |
| PRM / MCTS 失败尝试 | §4.2 | §G.2 | SI §7.2 |

### 0.4 v1 → v2 的实质差异清单（version drift）

| # | v1 | v2 / Nature | 说明 |
|---|---|---|---|
| D1 | R1-Zero AIME 2024 pass@1 “from 15.6% to 71.0%” | “from an initial 15.6% to 77.9%” | 同一次训练，v1 截在 8,400 步（v1 图 2 最后一点 0.710，与 v2 图 1a 第 8,400 步的点重合），v2 延长到 10,400 步并在 8.2k 步把最大长度从 32,768 提到 65,536。**但 77.9% 不是图 1a 上的任何一个点**：图上 pass@1 点都是 1/480 的整数倍，最高点 373/480 = 77.7%（9,200 步），最后一点 371/480 = 77.3%（10,000 步）；77.9% ≈ 374/480 没有画出，出处未说明（见 §8 第 21 条） |
| D2 | “with majority voting, the score further improves to 86.7%” / Table 2 cons@64 = 86.7 | “by leveraging the self-consistency decoding ... achieving an accuracy of 86.7%” | 数字不变；训练曲线图上画的是 cons@16 |
| D3 | Table 2：R1-Zero GPQA 73.3、MATH-500 95.9、LCB 50.0、CF 1444 | Table 3：GPQA **75.8**、MATH-500 95.9、LCB 50.0、CF 1444、AIME 77.9、CNMO 88.1 | 只有 AIME 与 GPQA 变了（论文没解释为何 MATH/LCB/CF 未重测） |
| D4 | 与 o1-0912 对比（“matching the performance of OpenAI-o1-0912”） | 删除所有 o1-0912，改与“human participants”比 | v2 图 1a 的参照线换成 0.378（人类平均） |
| D5 | “it is the first open research to validate that reasoning capabilities of LLMs can be incentivized purely through RL, without the need for SFT” | 删除 | v2 全文无 “first open research” |
| D6 | 冷启动格式 `|special_token|<reasoning_process>|special_token|<summary>` | 删除；改为 §B.3.2 的详细流程（R1-Zero 采样 T=1.0 → 过滤 → V3 改写 → 人工复核） | |
| D7 | 第二次 SFT：“fine-tune DeepSeek-V3-Base for two epochs” | §B.4.2：“for 2-3 epochs”，lr 5e-5→5e-6 cosine，32,768 上下文，batch 128 | 轮数表述不同 |
| D8 | 第二阶段 RL 无超参数 | T=0.7，共 1,700 步，一般数据与偏好奖励只在最后 400 步加入 | |
| D9 | 第一阶段 RL 无超参数 | lr 3e-6，KL 0.001，**clip ε = 10**，T=1，16 samples，最大 32,768 | |
| D10 | Qwen-32B 上做 RL 的模型叫 “DeepSeek-R1-Zero-Qwen-32B” | §F.1 / Table 16 改名 “Qwen2.5-32B-Zero”（§B.4.1 标题仍用旧名） | 数字相同 |
| D11 | Table 5 含 QwQ-32B-Preview、o1-mini 行 | Table 15 只保留 GPT-4o、Claude-3.5-Sonnet 两个基线 | QwQ 仍在 Table 16 |
| D12 | “Without safety RL, DeepSeek-R1 could achieve an accuracy of over 70%”（C-SimpleQA） | 删除 | |
| D13 | ArenaHard 摘要平均 689 tokens、AlpacaEval 2,218 字符 | 删除 | |
| D14 | 结论 “advancing beyond the boundaries of intelligence” | “advancing beyond the boundaries of **human** intelligence” | §F.1 |
| D15 | — | 新增：训练成本、RL 基础设施、RL 数据表、SFT 数据统计、reward hacking 图、语言一致性消融、安全报告、去污染、Qwen2-Math-7B-Zero 实验、Key Findings（7B/16B 失败） | 大多是回应审稿（见 §1 C36） |

---

## 1. 核心论断与原文引文

每条：论断（中文）→ 原文（英文，逐字）→ 位置。未注明版本的引文在 v1 与 v2 中逐字相同。

### 1.1 R1-Zero：直接在 base 上做 RL

**C1 R1-Zero 的定义：不经 SFT，直接在 DeepSeek-V3-Base 上用 GRPO 做大规模 RL。**
> "DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities."
— v1 Abstract。
> "Specifically, we use DeepSeek-V3-Base as the base model and employ GRPO (Shao et al., 2024) as the RL framework to improve model performance in reasoning."
— v1 §1。

**C2 不做 SFT 的理由（v2/Nature 才写明的动机）：人类定义的推理模式可能限制探索。**
> "This design choice stems from our hypothesis that human-defined reasoning patterns may limit model exploration, whereas unrestricted RL training can better incentivize the emergence of novel reasoning capabilities in LLMs."
— v2 §1（Nature Main 用 "originates from"，其余相同）。

**C3 训练模板只约束结构、不约束内容。** 模板原文（v2 Table 1；v1 Table 1 的换行不同，文字相同，只是 v1 写成 `<think> </think>`、v2 写成 `<think>...</think>`）：
> "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: prompt. Assistant:"
— v2 Table 1，caption：“prompt will be replaced with the specific reasoning question during training.”
> "We intentionally limit our constraints to this structural format, avoiding any content-specific biases—such as mandating reflective reasoning or promoting particular problem-solving strategies—to ensure that we can accurately observe the model’s natural progression during the RL process."
— v1 §2.2.3（v2 §2.3 删去了破折号中的举例）。

**C4 奖励 = 基于规则的准确性奖励 + 格式奖励，两者等权相加。**
> "Accuracy rewards: The accuracy reward model evaluates whether the response is correct. For example, in the case of math problems with deterministic results, the model is required to provide the final answer in a specified format (e.g., within a box), enabling reliable rule-based verification of correctness. Similarly, for LeetCode problems, a compiler can be used to generate feedback based on predefined test cases."
— v1 §2.2.2。v2 §2.2 与 Nature 改写了这段措辞，不只是换词：开头改成 “Accuracy rewards evaluate whether the response is correct”，“LeetCode problems” 改成 “code competition prompts”，后半句在 v2 改成 “a compiler can be utilized to evaluate the model’s responses against a suite of predefined test cases, thereby generating objective feedback on correctness”，Nature 又略有不同（“can be used to evaluate the responses of the model”）。引用时要注明版本。
> "Format rewards: In addition to the accuracy reward model, we employ a format reward model that enforces the model to put its thinking process between ‘<think>’ and ‘</think>’ tags."
— v1 §2.2.2。
> "The accuracy, reward and format reward are combined with the same weight."
— v2 §2.2（原文如此，“accuracy, reward” 中的逗号是原文笔误），Nature Methods “Reward design” 同。

**C5 不用神经奖励模型（无论结果型还是过程型）的原因：reward hacking + 重训成本 + 流程复杂。**
> "We do not apply the outcome or process neural reward model in developing DeepSeek-R1-Zero, because we find that the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process, and retraining the reward model needs additional training resources and it complicates the whole training pipeline."
— v1 §2.2.2。
> "Notably, we abstain from applying neural reward models—whether outcome-based or process-based—to reasoning tasks. This decision is predicated on our observation that neural reward models are susceptible to reward hacking during large-scale reinforcement learning."
— v2 §2.2。

**C6 数学题的准确性奖励是 0/1 二值。**
> "If the answer aligns with the reference, the reward is assigned a value of 1; otherwise, it is assigned a value of 0."
— v2 §B.3.1（Mathematics dataset）。STEM 选择题：“a binary reward is assigned based on whether the correct option is matched.”（同节）。

**C7 R1-Zero 的 AIME 2024 曲线：15.6% → 71.0%（v1）/ 77.9%（v2），多数投票 86.7%。**
> "For instance, the pass@1 score on AIME 2024 increases from 15.6% to 71.0%, and with majority voting, the score further improves to 86.7%, matching the performance of OpenAI-o1-0912."
— v1 §1。
> "Figure 1(a) depicts the performance trajectory of DeepSeek-R1-Zero on the AIME 2024 benchmark throughout the RL training process, where the average pass@1 score on AIME 2024 shows a significant increase, jumping from an initial 15.6% to 77.9%. In addition, by leveraging the self-consistency decoding (Wang et al., 2023c), the model’s performance can be further improved, achieving an accuracy of 86.7%."
— v2 §2.3；Nature Main 同（“a marked increase, jumping from an initial value of 15.6% to 77.9%”）。
v1 图 2 caption：“For each question, we sample 16 responses and calculate the overall average accuracy to ensure a stable evaluation.”

**C8 R1-Zero 训练超参数（v2/Nature 首次公开）。**
> "To train DeepSeek-R1-Zero, we set the learning rate to 3e-6, the KL coefficient to 0.001, and the sampling temperature to 1 for rollout. For each question, we sample 16 outputs with a maximum length of 32,768 tokens before the 8.2k step and 65,536 tokens afterward. As a result, both the performance and response length of DeepSeek-R1-Zero exhibit a significant jump at the 8.2k step, with training continuing for a total of 10,400 steps, corresponding to 1.6 training epochs. Each training step consists of 32 unique questions, resulting in a training batch size of 512. Every 400 steps, we replace the reference model with the latest policy model. To accelerate training, each rollout generates 8,192 outputs, which are randomly split into 16 mini-batches and trained for only a single inner epoch."
— v2 §2.1；Nature Methods “Training details of DeepSeek-R1-Zero” 同（写作 3 × 10−6、“batch size of 512 per step”）。

**C9 回答长度随训练自发增长（“thinking time”）。**
> "DeepSeek-R1-Zero naturally acquires the ability to solve increasingly complex reasoning tasks by leveraging extended test-time computation. This computation ranges from generating hundreds to thousands of reasoning tokens, allowing the model to explore and refine its thought processes in greater depth."
— v1 §2.2.4。
> "As shown in Figure 1(b), DeepSeek-R1-Zero exhibits a steady increase in thinking time throughout training, driven solely by intrinsic adaptation rather than external modifications."
— v2 §2.3。
注意：v2 自己在 C8 里承认 8.2k 步处的长度跳变是**外部改动**（最大长度 32K→64K）造成的；“solely by intrinsic adaptation” 与此并存，写课时要点明（见 §7 E11）。

**C10 aha moment（中间版本的回答，拟人化语气）。**
Table 3（v1）/ Table 2（v2）中模型输出的关键一行：
> "Wait, wait. Wait. That’s an aha moment I can flag here."
表题：
> "An interesting “aha moment” of an intermediate version of DeepSeek-R1-Zero. The model learns to rethink using an anthropomorphic tone. This is also an aha moment for us, allowing us to witness the power and beauty of reinforcement learning."
题目为：“If $a>1$, then the sum of the real solutions of $\sqrt{a-\sqrt{a+x}}=x$ is equal to”。
v2 把 aha moment 与 “wait” 词频挂钩：
> "Notably, during training, DeepSeek-R1-Zero exhibits an “aha moment” (Table 2), characterized by a sudden increase in the use of the word “wait” during reflections (see Figure 9(b) in Supplementary C.2 for details)."
— v2 §2.3。

**C11 反思词频次上升 5–7 倍；“wait” 在 8,000 步后激增。**
> "Specifically, the count of the reflective words rises 5- to 7-fold compared to the start of training, suggesting that RL plays a key role in generating long-chain intermediate tokens."
> "This reflective strategy was nearly absent during early training, showed occasional usage between steps 4000-7000, and then exhibited significant spikes after step 8000."
— v2 §C.2；SI §3.2。反思词表：“wait”, “mistake”, “however”, “but”, “retry”, “error”, “verify”, “wrong”, “evaluate”, “check”。
选词人数：v2 写 “selected by 33 human experts”，SI §3.2 写 “selected by 3 human experts”，Nature Extended Data Fig. 1 写 “a panel of three human experts”（见 §7 E5）。

**C12 R1-Zero 的缺陷：可读性差、语言混杂；v2 补充“规则奖励只覆盖推理任务”。**
> "DeepSeek-R1-Zero struggles with challenges like poor readability, and language mixing."
— v1 §2.2.4 “Drawback of DeepSeek-R1-Zero”。
> "Although DeepSeek-R1-Zero demonstrates excellent reasoning capabilities, it faces challenges such as poor readability and language mixing, occasionally combining English and Chinese within a single chain-of-thought response. Furthermore, the rule-based RL training stage of DeepSeek-R1-Zero is narrowly focused on reasoning tasks, resulting in limited performance in broader areas such as writing and open-domain question answering."
— v2 §1。模型卡（`r1-card`）另加一项：“challenges such as endless repetition, poor readability, and language mixing”。
语言混杂的成因（作者推测）：
> "The training data of DeepSeek-V3 base are mostly Chinese and English, which might be the cause for DeepSeek-R1-Zero language mixing when the language consistent reward is absent."
— v2 §A.1。

**C13 base 模型的容量是 RL-from-base 奏效的前提（7B 稠密、16B MoE 失败）。**
> "During the initial phase of our development, we experimented with smaller-scale models, specifically a 7B dense model and a 16B Mixture-of-Experts (MoE) model, as the foundational architectures for RL training. However, these configurations consistently failed to yield meaningful improvements when evaluated on the AIME benchmark, which we employed as the primary validation set."
> "To address these limitations, we transitioned to larger-scale models, including a 32B dense model (Qwen, 2024a), a 230B MoE model (DeepSeek-AI, 2024a), and a 671B MoE model (DeepSeek-AI, 2024b)."
— v2 §G.1 “The importance of base checkpoint”。

**C14 V3-Base 的预训练数据里可能间接含有其他强模型的输出。**
> "However, we have observed that some web pages contain a significant number of OpenAI-model-generated answers, which may lead the base model to acquire knowledge from other powerful models indirectly."
— v2 §A.1。
> "This extensive exposure equips the model with the capability to generate plausible solution candidates, from which reinforcement learning can effectively identify and optimize high-quality outputs."
— v2 §A.1。

### 1.2 GRPO 在 R1 中的用法与 PPO 的对比

**C15 选 GRPO 是为了省掉与策略同规模的 critic。**
> "In order to save the training costs of RL, we adopt Group Relative Policy Optimization (GRPO) (Shao et al., 2024), which foregoes the critic model that is typically the same size as the policy model, and estimates the baseline from group scores instead."
— v1 §2.2.1。

**C16 长 CoT 让价值模型更难学（部分回答的内容可能被后文推翻）。**
> "As the output length increases, the model is more likely to engage in behaviors such as reflection and revision during generation, meaning that the content initially generated may later be revised or contradicted, which makes it even less feasible to predict the final reward based on a partial response."
— v2 §A.3。

**C17 PPO 把逐 token KL 作为稠密奖励，会隐式惩罚长度；GRPO 把 KL 直接加进损失，并周期性重置参考策略。**
> "Since the optimization goal of reinforcement learning is to maximize cumulative rewards, PPO’s approach penalizes the cumulative KL divergence, which may implicitly penalize the length of the response and thereby prevent the model’s response length from increasing."
> "In order to balance the scope that the training policy can explore and the stability of the training, we periodically update the reference policy to the latest policy during the actual training process."
— v2 §A.3。

**C18 PPO 对 GAE 的 λ 敏感；λ=1.0 时接近 GRPO。**
> "When λ is set to 0.95 (the default value in most open-source PPO implementations), PPO performs considerably worse than GRPO. However, with careful tuning (setting λ to 1.0), PPO’s performance improves substantially, nearing that of GRPO."
— v2 §A.3（实验：DeepSeek-Coder-V2-Lite，“16B MoE with 2.4B active parameters”，MATH 任务，图 4）。

**C19 R1 第一阶段 RL 用了很大的裁剪系数 ε = 10，并说明裁剪系数的两难。**
> "In the first stage of RL, we set the learning rate to 3e-6, the KL coefficient to 0.001, the GRPO clip ratio ε to 10, and the sampling temperature to 1 for rollout."
> "Note that the clip ratio plays a crucial role in training. A lower value can lead to the truncation of gradients for a significant number of tokens, thereby degrading the model’s performance, while a higher value may cause instability during training."
— v2 §3.2.1；Nature Methods 同。作者贡献说明：“Zhibin Gou proposed a large PPO clipping strategy to enhance GRPO performance”（v2 §7 Author List）。

### 1.3 DeepSeek-R1 的四阶段流水线

**C20 流水线概述：两段 SFT + 两段 RL。**
> "The pipeline incorporates two RL stages aimed at discovering improved reasoning patterns and aligning with human preferences, as well as two SFT stages that serve as the seed for the model’s reasoning and non-reasoning capabilities."
— v1 §1.1。
> "To address these questions, we design a pipeline to train DeepSeek-R1. The pipeline consists of four stages, outlined as follows."
— v1 §2.3。v2 图 2 把中间检查点命名为 Dev1（冷启动 SFT 后）、Dev2（推理 RL 后）、Dev3（第二次 SFT 后）、R1（全场景 RL 后）。

**C21 冷启动：几千条长 CoT；动机是可读性（v2 说主要是产品驱动）。**
> "In this work, we collect thousands of cold-start data to fine-tune the DeepSeek-V3-Base as the starting point for RL."
— v1 §2.3.1。
> "The motivation is primarily product-driven, with a strong emphasis on enhancing user experience."
— v2 §B.3.2。
v2 给出的构造流程：
> "For each prompt, we generate multiple reasoning trajectories using DeepSeek-R1-Zero with a relatively high temperature of 1.0. Next, we filter these generations to retain only those with correct final answers and a readable format."
— v2 §B.3.2（数学答案用 sympy 比对；格式用重复检测、语言混杂过滤；再让 DeepSeek-V3 改写推理与总结；人工二次核验）。
v2 同节还有一段自我提醒（写课时适合引用来对冲拟人化）：
> "Here, we would like to emphasize that the observed vivid reasoning patterns primarily reflect DeepSeek-engineered heuristics, rather than indicating that the model has inherently acquired human-like intelligence or autonomous problem-solving capabilities."

**C22 推理导向 RL + 语言一致性奖励（= CoT 中目标语言词的比例），直接与准确性奖励相加；代价是性能略降。**
> "To mitigate the issue of language mixing, we introduce a language consistency reward during RL training, which is calculated as the proportion of target language words in the CoT. Although ablation experiments show that such alignment results in a slight degradation in the model’s performance, this reward aligns with human preferences, making it more readable. Finally, we combine the accuracy of reasoning tasks and the reward for language consistency by directly summing them to form the final reward."
— v1 §2.3.2。
消融（v2 §B.6，在 DeepSeek-R1-Distill-Qwen-7B 上做）：
> "For benchmark performance, the model maintains comparable performance on the mathematical benchmark, while a slight degradation is observed on the coding benchmark."

**C23 拒绝采样 + SFT：约 60 万推理样本 + 约 20 万非推理样本 ≈ 80 万，从 V3-Base 重新微调。**
> "For each prompt, we sample multiple responses and retain only the correct ones. In total, we collect about 600k reasoning related training samples."
> "In the end, we collected a total of approximately 200k training samples that are unrelated to reasoning."
> "We fine-tune DeepSeek-V3-Base for two epochs using the above curated dataset of about 800k samples."
— v1 §2.3.3。这一阶段开始用生成式奖励模型（DeepSeek-V3 判官）判定部分非规则可验数据：
> "some of which use a generative reward model by feeding the ground-truth and model predictions into DeepSeek-V3 for judgment."
— v1 §2.3.3。v2 Table 5 给出精确计数：总计 **804,745**（见 §3.8）。

**C24 全场景 RL：推理数据仍用规则奖励；一般数据用奖励模型；有用性只看总结，无害性看全文。**
> "For helpfulness, we focus exclusively on the final summary, ensuring that the assessment emphasizes the utility and relevance of the response to the user while minimizing interference with the underlying reasoning process. For harmlessness, we evaluate the entire response of the model, including both the reasoning process and the summary, to identify and mitigate any potential risks, biases, or harmful content that may arise during the generation process."
— v1 §2.3.4。
v2 补充超参数：
> "The second stage of RL retains most of the parameters from the first stage, with the key difference being a reduced temperature of 0.7, as we find that higher temperatures in this stage lead to incoherent generation. The stage comprises a total of 1,700 training steps, during which general instruction data and preference-based rewards are incorporated exclusively in the final 400 steps. We find that more training steps with the model based preference reward signal may lead to reward hacking, which is documented in Supplementary B.5."
— v2 §3.2.2。

**C25 reward hacking 真实发生在有用性奖励模型上（奖励涨、CodeForces 跌）。**
> "In our work, we observe such reward hacking behavior when employing the helpful reward model."
— v2 §B.5；图 6 caption：“Reward hacking: the reward exhibits an increasing trend as the performance on CodeForces decreases for training.”

**C26 各阶段的作用（Dev1 推理退步、Dev2 推理大涨、Dev3 通用提升、R1 偏好基准大涨）。**
> "However, due to the limited size of the cold-start dataset, Dev1 exhibits a partial degradation in reasoning performance compared to DeepSeek-R1-Zero, most notably on the AIME benchmark."
> "The primary advancements in the final DeepSeek-R1 were in general instruction-following and user-preference benchmarks, with AlpacaEval 2.0 improving by 25% and ArenaHard by 17%."
— v2 §4。（这里的 25% / 17% 是百分点，见 §4 N25。）

### 1.4 蒸馏

**C27 用 80 万条 R1 样本对 6 个开源稠密模型做纯 SFT，不做 RL。**
> "we directly fine-tuned open-source models like Qwen (Qwen, 2024b) and Llama (AI@Meta, 2024) using the 800k samples curated with DeepSeek-R1, as detailed in §2.3.3."
> "The base models we use here are Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, and Llama-3.3-70B-Instruct. We select Llama-3.3 because its reasoning capability is slightly better than that of Llama-3.1."
> "For distilled models, we apply only SFT and do not include an RL stage, even though incorporating RL could substantially boost model performance."
— v1 §2.4。

**C28 蒸馏 vs 在小模型上直接 RL：Qwen-32B 上 RL 1 万步只到 QwQ-32B-Preview 水平，蒸馏版全面更好。**
> "To answer this question, we conduct large-scale RL training on Qwen-32B-Base using math, code, and STEM data, training for over 10K steps, resulting in DeepSeek-R1-Zero-Qwen-32B."
> "First, distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation. Second, while distillation strategies are both economical and effective, advancing beyond the boundaries of intelligence may still require more powerful base models and larger-scale reinforcement learning."
— v1 §4.1（v2 §F.1 改名 Qwen2.5-32B-Zero，并改为 “boundaries of human intelligence”）。

**C29 回应审稿的补充实验：在 o1 发布之前的 base（Qwen2-Math-7B）上做 RL 也有效。**
> "Apart from the experiment based on Qwen-2.5-32B, we conducted experiments on Qwen2-Math-7B (released August 2024) prior to the launch of the first reasoning model, OpenAI-o1 (September 2024), to ensure the base model was not exposed to any reasoning trajectory data."
— v2 §F.1（Table 17：Qwen2-Math-7B-Zero AIME 2024 22.3%、AIME 2025 18.1%，“approximately 10,000 policy gradient update steps”）。
审稿人原话（`r1-peer-review`，Referee 意见）：“Can you please run your experiment in table D14 starting from a base model that was developed **before the public release of any reasoning trajectories**?”

### 1.5 失败尝试（PRM、MCTS）

**C30 PRM 的三条限制。**
> "First, it is challenging to explicitly define a fine-grain step in general reasoning. Second, determining whether the current intermediate step is correct is a challenging task. Automated annotation using models may not yield satisfactory results, while manual annotation is not conducive to scaling up. Third, once a model-based PRM is introduced, it inevitably leads to reward hacking (Gao et al., 2022), and retraining the reward model needs additional training resources and it complicates the whole training pipeline."
> "In conclusion, while PRM demonstrates a good ability to rerank the top-N responses generated by the model or assist in guided search (Snell et al., 2024), its advantages are limited compared to the additional computational overhead it introduces during the large-scale reinforcement learning process in our experiments."
— v1 §4.2 / v2 §G.2（逐字相同，只是 Lightman 的年份 2023→2024）。

**C31 MCTS 的两条困难：搜索空间指数级大、细粒度价值模型难训。**
> "First, unlike chess, where the search space is relatively well-defined, token generation presents an exponentially larger search space. To address this, we set a maximum extension limit for each node, but this can lead to the model getting stuck in local optima. Second, the value model directly influences the quality of generation since it guides each step of the search process. Training a fine-grained value model is inherently difficult, which makes it challenging for the model to iteratively improve."
> "In conclusion, while MCTS can improve performance during inference when paired with a pre-trained value model, iteratively boosting model performance through self-search remains a significant challenge."
— v1 §4.2 / v2 §G.2。
两处的免责声明：“We share our failure experiences here to provide insights, but this does not imply that these approaches are incapable of developing effective reasoning models.”

### 1.6 成本、测试时计算、安全

**C32 训练成本（SI §2.4.4 / v2 §B.4.4）。**
> "Regarding our research on DeepSeek-R1, we utilized the A100 GPUs to prepare for the experiments with a smaller model (30B parameters). The results from this smaller model have been promising, which has allowed us to confidently scale up to 660B R1-Zero and R1. For the training of DeepSeek-R1-Zero, we employed 64*8 H800 GPUs, and the process required approximately 198 hours. Additionally, during the training phase of DeepSeek-R1, we utilized the same 64*8 H800 GPUs, completing the process in about 4 days, or roughly 80 hours. To create the SFT datasets, we use 5K GPU hours."
表题（SI Table 4 = v2 Table 7）：“Training costs of DeepSeek-R1, assuming the rental price of H800 is $2 per GPU hour.” 数字见 §3.12。

**C33 测试时计算随难度自适应；R1 平均 8,793 思考 token，GPT-4o 711 token。**
> "DeepSeek-R1 achieves a 61.8% solve rate (Pass@1) by scaling test-time compute to an average of 8,793 thinking tokens per problem. Notably, the model adaptively adjusts its computational effort based on problem difficulty, using fewer than 7,000 thinking tokens for simple problems while dedicating more than 18,000 thinking tokens to the most challenging ones"
> "On this same set of math problems, GPT-4o 0513 achieves only a 24.7% solve rate while generating 711 output tokens on average — an order of magnitude less than DeepSeek-R1."
— v2 §E.4（题集：“366 problems drawn from 93 mathematics competitions held in 2024”，§E.3）。
> "DeepSeek-R1’s Pass@64 score on AIME 2024 is 90.0%, significantly higher than its Pass@1 score of 79.8%."
> "majority voting further improves DeepSeek-R1’s accuracy from 79.8% to 86.7%."
— v2 §E.4。

**C34 安全：模型本身“中等”，加风控系统后提升。**
> "These comprehensive safety analyses conclude that the inherent safety level of the DeepSeek-R1 model, compared to other state-of-the-art models, is generally at a moderate level (comparable to GPT-4o (2024-05-13)). Besides, when coupled with the risk control system, the model’s safety level is elevated to a superior standard."
— v2 §5 / Nature “Ethics and safety statement”。
> "For example, R1 can be subject to jailbreak attacks, leading to the generation of dangerous content such as explosive manufacturing plans, while the enhanced reasoning capabilities enable the model to provide plans with better operational feasibility and executability."
— 同上。风控系统 = 关键词匹配筛出可疑对话 → 连同风险审查提示词交给 DeepSeek-V3 判定是否撤回（v2 §D.3.1）。

**C35 去污染：10-gram 过滤，仅数学领域就删掉约 600 万条预训练文本；作者承认挡不住改写。**
> "we filtered out any text segments (including web pages and GitHub files) that contained matching 10-gram sequences from evaluation questions or reference solutions. As one example of our decontamination efforts, in the mathematics domain alone, our decontamination process identified and removed approximately six million potential pre-training texts."
> "However, we acknowledge that the n-gram based decontamination method cannot prevent the paraphrase of testset."
— v2 §D.1。

**C36 这些补充材料大多是审稿要求的。** Peer Review File 中审稿人要求：
“More information about the cost. We read that RL training incurs "high demands". This should be made explicit with actual data of time and compute...”；
“Safety: this is the most important thing that has to be addressed.”；
另一位审稿人批评拟人化：“The paper suffers from excessive anthropomorphization in my view”。
（写课时可以用来说明：Nature 版的成本、安全、去污染章节是评审推动的，不是 v1 就有的。）

**C37 作者的总结判断：关键是难题、可靠的验证器和足够的 RL 算力。**
> "We believe that the key to unlocking this potential lies not in large-scale human annotation but in the provision of hard reasoning questions, a reliable verifier, and sufficient computational resources for reinforcement learning."
— v2 §6 / Nature Conclusion。

### 1.7 R1-0528（模型卡，artefact）

**C38 R1-0528 是“小版本升级”：更多后训练算力 + 算法优化；思考更深（AIME 每题 12K → 23K tokens）。**
> "The DeepSeek R1 model has undergone a minor version upgrade, with the current version being DeepSeek-R1-0528. In the latest update, DeepSeek R1 has significantly improved its depth of reasoning and inference capabilities by leveraging increased computational resources and introducing algorithmic optimization mechanisms during post-training."
> "For instance, in the AIME 2025 test, the model’s accuracy has increased from 70% in the previous version to 87.5% in the current version. This advancement stems from enhanced thinking depth during the reasoning process: in the AIME test set, the previous model used an average of 12K tokens per question, whereas the new version averages 23K tokens per question."
> "Beyond its improved reasoning capabilities, this version also offers a reduced hallucination rate, enhanced support for function calling, and better experience for vibe coding."
— `r1-0528-card` §1。使用方式的变化（§4）：“1. System prompt is supported now. 2. It is not required to add "<think>\n" at the beginning of the output to force the model into thinking pattern.”
发布日期：`r1-0528-news` 标题 “DeepSeek-R1-0528 Release 2025/05/28”，并写 “Supports JSON output & function calling”。
**架构不变**：R1-0528 的 `config.json` 与 R1 逐字节相同（§3.9）。R1-0528 没有论文，训练细节（“algorithmic optimization mechanisms” 具体是什么）未公开。

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 GRPO 目标（R1 论文版本）

v1 式 (1)–(3)；v2 正文式 (1)–(3)，附录 A.3 重抄为 (11)–(13)；Nature Methods 式 (1)–(3)。三处的**数学内容**逐符号一致。v1 与 v2 的 LaTeX 源几乎逐字相同：v2 多一个 `\footnotesize`，对齐符 `&` 从 `=` 前移到了 $\mathcal J$ 前。Nature 的公式由出版社重排：用 `\epsilon` 而不是 `\varepsilon`，`ref`、`old`、`GRPO` 用正体，并用 `array` 环境；符号含义不变。下面是 v1 原文：

$$
\begin{split}\mathcal{J}_{GRPO}(\theta)&=\mathbb{E}{[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O|q)]}\\
&\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)}A_{i},\text{clip}\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)},1-\varepsilon,1+\varepsilon\right)A_{i}\right)-\beta\mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)\right),\end{split}
\tag{1}
$$

$$
\mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)=\frac{\pi_{ref}(o_{i}|q)}{\pi_{\theta}(o_{i}|q)}-\log\frac{\pi_{ref}(o_{i}|q)}{\pi_{\theta}(o_{i}|q)}-1,
\tag{2}
$$

$$
A_{i}=\frac{r_{i}-{\mathrm{m}ean(\{r_{1},r_{2},\cdots,r_{G}\})}}{{\mathrm{s}td(\{r_{1},r_{2},\cdots,r_{G}\})}}.
\tag{3}
$$

原文对符号的交代：“where $\pi_{ref}$ is a reference policy, $\varepsilon$ and $\beta$ are hyper-parameters, and $A_i$ is the advantage, computed using a group of rewards $\{r_1,r_2,\ldots,r_G\}$ corresponding to the outputs within each group”（v2 §2.1；v1 没有 “$\pi_{ref}$ is a reference policy” 这半句）。

**与 DeepSeekMath 原式的差别（重要）。** DeepSeekMath（arXiv:2402.03300 v3，式 (3)）写的是 **token 级**比率并对每条回答按长度平均：

$$
\begin{split}\mathcal{J}_{GRPO}(\theta)&=\mathbb{E}{[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O|q)]}\\
&\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\left\{\min\left[\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}\hat{A}_{i,t},\text{clip}\left(\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})},1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right]-\beta\mathbb{D}_{KL}\left[\pi_{\theta}||\pi_{ref}\right]\right\}\end{split}
$$

并且结果监督下 $\hat A_{i,t}=\widetilde r_i=\frac{r_i-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)}$（同一回答的所有 token 共享一个优势）。
R1 论文把比率写成整条序列的概率比 $\pi_\theta(o_i|q)/\pi_{\theta_{old}}(o_i|q)$、没有 $1/|o_i|$。论文没有说明实现用的是哪一种；
整条 32K–64K token 序列的概率比数值上几乎必然溢出，**笔者推测**实现是 token 级（列入 §8 UNVERIFIED）。
课上建议：主式用 DeepSeekMath 的 token 级写法（与第 8 课一致），旁注“R1 论文印刷为序列级的简写”。

**k3 估计器的来源。** R1 的式 (2) 在 DeepSeekMath 中是**式 (4)**，token 级写法，条件是 $q,o_{i,<t}$。DeepSeekMath 自己的式 (2) 是 PPO 的逐 token KL 惩罚，不要混淆。DeepSeekMath 把式 (4) 称为 “the following unbiased estimator (Schulman, 2020)”，并说它 “is guaranteed to be positive”。Schulman 博客（2020/03/07）中 k3 为 $(r-1)-\log r$，其中 $r=p(x)/q(x)$、$x\sim q$，估计的是 $\mathrm{KL}[q,p]$。
映射：$q=\pi_\theta$、$p=\pi_{ref}$、$r=\pi_{ref}/\pi_\theta$，于是式 (2) 就是 k3，估计 $\KL(\pi_\theta\Vert\pi_{ref})$。
无偏性要求样本来自 $\pi_\theta$；GRPO 的样本来自 $\pi_{\theta_{old}}$，严格说有 off-policy 偏差——这是第 15 课 V3.2 “无偏 KL 估计”修正的伏笔（本课不展开）。

### 2.2 课程记号对照（style-guide §2）

| 论文 | 课程 | 形状 / 类型 | 含义 |
|---|---|---|---|
| $q$ | $q$ | token 序列 | 问题（prompt） |
| $o_i$, $o_{i,t}$ | 同 | 序列 / token | 第 $i$ 条回答、其第 $t$ 个 token |
| $G$ | $G$ | 正整数 | 组大小；R1-Zero 与 R1 都是 **16** |
| $\pi_\theta,\pi_{\theta_{old}},\pi_{ref}$ | 同 | 条件分布 | 当前 / 采样用旧 / 参考策略 |
| $A_i$ | $\hat A_i$ | 标量 | 组内标准化优势（课程统一加帽子） |
| $r_i$ | $r_i$ | 标量 | 第 $i$ 条回答的总奖励 |
| $\varepsilon$ | $\varepsilon$ | 标量 | 裁剪系数（R1 第一阶段 = 10） |
| $\beta$ | $\beta$ | 标量 | KL 系数（= 0.001） |
| $\mathbb D_{KL}(\pi_\theta\Vert\pi_{ref})$ | $\KL(\pi_\theta\Vert\pi_{ref})$ | 标量 | 用 k3 逐样本估计 |
| $\mathrm{mean},\mathrm{std}$ | 同 | — | 组内均值、标准差（总体/样本标准差论文未说明） |

课程推荐写法（token 级，$\rho_{i,t}$ 为重要性比率）：

$$
\mathcal J_{GRPO}(\theta)=\E\!\left[\frac1G\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(\rho_{i,t}\hat A_i,\ \clip(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_i\big)-\beta\,\widehat{\KL}_{i,t}\Big)\right],\quad
\rho_{i,t}=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})}
$$

### 2.3 奖励公式（v2 / Nature；v1 只有文字）

$$
Reward_{\text{rule}}=Reward_{\text{acc}}+Reward_{\text{format}}\tag{4}
$$
$$
Reward_{helpful}=RM_{helpful}(Response_{A},Response_{B})\tag{5}
$$
$$
Reward_{safety}=RM_{safety}(Response)\tag{6}
$$
$$
Reward_{language}=\frac{Num(Words_{target})}{Num(Words)}\tag{7}
$$
$$
Reward=Reward_{\text{reasoning}}+Reward_{\text{general}}+Reward_{\text{language}}\tag{8}
$$
$$
\text{where, }Reward_{\text{reasoning}}=Reward_{\text{rule}}\tag{9}
$$
$$
Reward_{\text{general}}=Reward_{\text{reward\_model}}+Reward_{\text{format}}\tag{10}
$$

注：式 (5) 的有用性 RM 是**成对**训练（66,000 对），式 (6) 的安全 RM 是**逐点**训练（106,000 条）；“For general queries, each instance is categorized as belonging to either the safety dataset or the helpfulness dataset.”（v2 §3.1）。
式 (8) 是第二阶段的总奖励；第一阶段是 $Reward_{acc}(+Reward_{format}) + Reward_{language}$（“We apply the language consistency reward to both reasoning and non-reasoning data by directly adding it to the final reward.” v2 §3.2.1）。
**式 (7) 的“词”如何切分、中文如何计“词”，论文未说明**（§8）。

### 2.4 评测公式

$$
\text{pass@1}=\frac{1}{k}\sum_{i=1}^{k}p_{i}
$$
“where $p_i$ denotes the correctness of the $i$-th response.”（v1 §3 Evaluation Setup；v2 §D.1）。温度 0.6、top-p 0.95；v2 写明 “$k=64$ for AIME and GPQA, $k=16$ for MATH and CodeForces, and $k=8$ for LCB”。cons@64 = 64 个样本多数投票。

USAMO 资格指数（v2 Table 13 caption）：$\texttt{AMC score}+10\times\texttt{AIME score}$，超过 251.5 有资格参加 USAMO。

### 2.5 由式 (3) 推出的二值奖励闭式解（本笔记推导，供课上 `::: deriv` 用）

设一组 $G$ 条回答中有 $k$ 条正确（$r=1$）、其余 $r=0$，令 $p=k/G$。
组均值 $=p$；总体标准差 $=\sqrt{p(1-p)}$（样本标准差再乘 $\sqrt{G/(G-1)}$）。于是（总体标准差）

$$
\hat A_{+}=\frac{1-p}{\sqrt{p(1-p)}}=\sqrt{\frac{1-p}{p}},\qquad
\hat A_{-}=\frac{-p}{\sqrt{p(1-p)}}=-\sqrt{\frac{p}{1-p}},\qquad \hat A_{+}\cdot\hat A_{-}=-1 .
$$

性质（Python 验证，§4 N30–N33）：$\sum_i\hat A_i=0$；$\sum_i\hat A_i^2=G$（总体标准差时恒等于 16）；
对奖励做平移 $r\to r+c$ 或正缩放 $r\to a r$，优势不变——所以当组内所有回答都满足格式时，格式奖励对梯度**没有**贡献。

---

## 3. 规格与超参数表

### 3.1 R1-Zero 的 RL 超参数

| 项 | 值 | 来源 |
|---|---|---|
| 起点模型 | DeepSeek-V3-Base | v1 §1；v2 §1 |
| 算法 | GRPO | v1 §2.2.1 |
| 学习率 | 3e-6 | v2 §2.1；Nature Methods |
| KL 系数 $\beta$ | 0.001 | 同上 |
| 裁剪系数 $\varepsilon$ | **未给出** | v2 §2.1 只在 R1 第一阶段给了 ε=10 |
| rollout 温度 | 1 | 同上 |
| 每题采样数 $G$ | 16 | 同上 |
| 最大长度 | 32,768 tokens（8.2k 步前）→ 65,536 tokens（之后） | 同上 |
| 总步数 | 10,400 步 = 1.6 epochs | 同上 |
| 每步题数 / 回答数 | 32 题 / 512 条 | 同上 |
| 参考模型重置 | 每 400 步换成最新策略 | 同上 |
| 每次 rollout | 8,192 条回答，随机分 16 个 mini-batch，只训 1 个 inner epoch | 同上 |
| 奖励 | 规则：准确性 + 格式，等权 | v2 §2.2 |
| 训练设备与时间 | 64×8 H800，约 198 小时 | SI §2.4.4 |

### 3.2 R1 第一阶段 RL（冷启动之后，推理导向，得到 Dev2）

| 项 | 值 | 来源 |
|---|---|---|
| 学习率 / KL / 温度 | 3e-6 / 0.001 / 1 | v2 §3.2.1 |
| 裁剪系数 $\varepsilon$ | **10** | v2 §3.2.1；Nature Methods |
| $G$ / 最大长度 | 16 / 32,768 | 同上 |
| 每步 | 32 题，512 条 | 同上 |
| 参考模型重置 | 每 400 步 | 同上 |
| rollout | 8,192 条，16 个 mini-batch，1 inner epoch | 同上 |
| 奖励 | 规则奖励 + 语言一致性奖励（直接相加） | v1 §2.3.2；v2 §3.2.1 |
| 总步数 | **未给出**（v1：“until it achieves convergence on reasoning tasks”） | — |

### 3.3 R1 第二阶段 RL（全场景，得到 R1）

| 项 | 值 | 来源 |
|---|---|---|
| 与第一阶段的差别 | “retains most of the parameters”，温度降为 **0.7** | v2 §3.2.2 |
| 总步数 | 1,700 | 同上 |
| 一般数据与偏好奖励 | 只在最后 400 步加入 | 同上 |
| 奖励 | 式 (8)–(10) | 同上 |
| RL 数据（一般） | 66K 有用性问题 + 12,000 无害性问题 | v2 §B.3.1 |

### 3.4 SFT 超参数（冷启动 SFT 与第二次 SFT）

| 项 | 值 | 来源 |
|---|---|---|
| 起点 | DeepSeek-V3-Base（两次都是） | v2 §B.4.2；v1 §2.3.3 “retrain the DeepSeek-V3-Base model” |
| 轮数 | 2–3 epochs（v2）；第二次 SFT 在 v1 写 “two epochs” | v2 §B.4.2；v1 §2.3.3 |
| 学习率 | cosine，5×10⁻⁵ → 5×10⁻⁶ | v2 §B.4.2 |
| 最大上下文 / batch | 32,768 / 128 | 同上 |
| 冷启动数据量 | “thousands” | v1 §2.3.1 |

### 3.5 蒸馏超参数（v2 Table 6 = SI Table 3）

| 蒸馏模型 | base | 初始学习率 | 来源 |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 1×10⁻⁴ | v2 Table 6 |
| DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 8×10⁻⁵ | 同上 |
| DeepSeek-R1-Distill-Qwen-14B | Qwen2.5-14B | 7×10⁻⁵ | 同上 |
| DeepSeek-R1-Distill-Qwen-32B | Qwen2.5-32B | 6×10⁻⁵ | 同上 |
| DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B | 5×10⁻⁵ | 同上 |
| DeepSeek-R1-Distill-Llama-70B | Llama-3.3-70B-Instruct | 2×10⁻⁵ | 同上 |

共同设置：2–3 epochs，80 万数据，cosine 衰减到初始值的 1/10，最大上下文 32,768，batch 64（v2 §B.4.3）。
模型卡：“We slightly change their configs and tokenizers. Please use our setting to run these models.”（`r1-card` §3）。

Qwen-32B 上的纯 RL 对照（DeepSeek-R1-Zero-Qwen-32B / Qwen2.5-32B-Zero）：lr **2e-6**、KL 0.001、T=1、$G$=16、最大 32,768、每步 32 题、每 400 步换参考、8,192 条/rollout、16 mini-batch（v2 §B.4.1）；“training for over 10K steps”（v1 §4.1）。

### 3.6 奖励模型

| 项 | 有用性 RM | 安全 RM | 来源 |
|---|---|---|---|
| 数据 | 66,000 对偏好（DeepSeek-V3 按 Arena-Hard 格式判 4 次、随机交换 A/B，均分差 Δ>1 才保留，选/拒长度可比） | 106,000 条带 safe/unsafe 标注的提示+回答 | v2 §3.1 |
| 训练方式 | 成对（pairwise） | 逐点（point-wise） | 同上 |
| 架构 | 与 R1 相同 + 标量奖励头 | 同左 | 同上 |
| 超参数 | batch 256，lr 6e-6，1 epoch，训练时最大 8,192 tokens，推理时不限长度 | 同有用性 | 同上 |

### 3.7 RL 数据（v2 Table 4 = SI Table 1）

| 类型 | 提示数 | 题型 | 输出类型 | 备注（v2 §B.3.1） |
|---|---|---|---|---|
| Math | 26K | Quantitative Reasoning | Number/Expression/Equation | 平均提示 122 tokens；不含证明题；奖励 0/1 |
| Code | 17K | Algorithm and Bug Fixing | Code Solution | 正文：17k 算法竞赛题 “along with 8k bug fixing problems” |
| STEM | 22K | Multi-Choice | Option | 4–8 个选项；物理 15.5%、生物 30.7%、化学 46.5%、其他 7.3%；平均 161 tokens |
| Logic | 15K | Choice/Quantitative Reasoning | Option/Number | 平均约 420 tokens；含 code-IO 与谜题（如 24 点、斑马谜题） |
| General | 66K | Helpfulness/Harmlessness | Ranked Responses | 另含 12,000 条无害性问题 |

“All questions are in Chinese or English.”（v2 §B.3.1）。冷启动代码数据来源：“5151 problems from Codeforces and 2504 problems from AtCoder”（v2 §B.3.2）。

### 3.8 80 万 SFT 数据统计（v2 Table 5 = SI Table 2）

| 领域 | 样本数 | 平均轮数 | 平均 tokens |
|---|---|---|---|
| Math | 395,285 | 1.0 | 6,094.2 |
| Code | 211,129 | 1.1 | 7,435.7 |
| STEM | 10,124 | 1.0 | 4,928.8 |
| Logic | 10,395 | 1.0 | 2,739.0 |
| General | 177,812 | 1.1 | 1,419.8 |
| Total | 804,745 | 1.0 | 5,355.3 |

### 3.9 模型规格（config.json，`r1-config`；R1、R1-Zero、R1-0528 三个文件逐字节相同）

| 字段 | 值 | 解读 |
|---|---|---|
| architectures / model_type | DeepseekV3ForCausalLM / deepseek_v3 | 就是 V3 架构 |
| hidden_size | 7168 | $d$ |
| num_hidden_layers | 61 | $l$ |
| num_attention_heads / num_key_value_heads | 128 / 128 | MLA（KV 头数字段对 MLA 无意义） |
| kv_lora_rank / q_lora_rank | 512 / 1536 | $d_c$ / $d_c'$ |
| qk_nope_head_dim / qk_rope_head_dim / v_head_dim | 128 / 64 / 128 | $d_h$ / $d_h^R$ / 值头维 |
| n_routed_experts / n_shared_experts / num_experts_per_tok | 256 / 1 / 8 | $N_r$ / $N_s$ / $K_r$ |
| moe_intermediate_size / intermediate_size | 2048 / 18432 | 专家 FFN / 稠密层 FFN |
| first_k_dense_replace | 3 | 前 3 层稠密 |
| n_group / topk_group / topk_method / scoring_func | 8 / 4 / noaux_tc / sigmoid | 节点受限路由 + 无辅助损失均衡 |
| routed_scaling_factor | 2.5 | |
| num_nextn_predict_layers | 1 | MTP 1 层 |
| vocab_size | 129,280 | |
| max_position_embeddings | 163,840 | = 4096 × 40（YaRN factor 40，original 4096） |
| quantization_config | fp8, e4m3, weight_block_size [128,128], activation dynamic | FP8 权重发布 |
| 模型卡 | 671B 总参数 / 37B 激活 / 128K 上下文（R1 与 R1-Zero 同） | `r1-card` §3 |
| generation_config | temperature 0.6, top_p 0.95 | `r1-card` generation_config.json |

注意：SI §2.4.4 写 “scale up to **660B** R1-Zero and R1”，与 671B 不一致（§7 E6）。

### 3.10 蒸馏模型规格（`r1-distill-configs`）

| 模型 | 架构 | $d$ | 层 | $n_h$ | KV 头 | $d_h$ | FFN | max_pos | 词表 |
|---|---|---|---|---|---|---|---|---|---|
| Distill-Qwen-1.5B | Qwen2 | 1536 | 28 | 12 | 2 | 128 | 8960 | 131072 | 151,936 |
| Distill-Qwen-7B | Qwen2 | 3584 | 28 | 28 | 4 | 128 | 18944 | 131072 | 152,064 |
| Distill-Llama-8B | Llama | 4096 | 32 | 32 | 8 | 128 | 14336 | 131072 | 128,256 |
| Distill-Qwen-14B | Qwen2 | 5120 | 48 | 40 | 8 | 128 | 13824 | 131072 | 152,064 |
| Distill-Qwen-32B | Qwen2 | 5120 | 64 | 40 | 8 | 128 | 27648 | 131072 | 152,064 |
| Distill-Llama-70B | Llama | 8192 | 80 | 64 | 8 | 128 | 28672 | 131072 | 128,256 |
| R1-0528-Qwen3-8B | Qwen3 | 4096 | 36 | 32 | 8 | 128 | 12288 | 131072 | 151,936 |

### 3.11 评测设置

| 项 | 值 | 来源 |
|---|---|---|
| 最大生成长度 | 32,768 tokens | v1 §3；v2 §D.1 |
| 采样 | T=0.6，top-p 0.95，每题 $k$ 个（4–64；v2：AIME/GPQA 64、MATH/CF 16、LCB 8） | 同上 |
| 不用贪心解码的理由 | “higher repetition rates and significant variability across different checkpoints” | 同上 |
| LiveCodeBench 窗口 | 2024-08 – 2025-01 | 同上 |
| Codeforces | “10 Div.2 contests along with expert-crafted test cases” | 同上 |
| AlpacaEval / ArenaHard | GPT-4-Turbo-1106 当裁判，只喂 final summary | 同上 |
| 模型卡 | “generate 64 responses per query to estimate pass@1” | `r1-card` §4 |
| R1-0528 模型卡 | 最大 64K，T=0.6，top-p 0.95，每题 16 个回答 | `r1-0528-card` §2 |

### 3.12 训练成本（SI Table 4 = v2 Table 7，按 H800 租价 $2/GPU·h）

| 项 | H800 GPU 小时 | 美元 |
|---|---|---|
| DeepSeek-R1-Zero | 101K | $202K |
| SFT data creation | 5K | $10K |
| DeepSeek-R1 | 41K | $82K |
| **Total** | **147K** | **$294K** |

**覆盖范围**：R1-Zero 的 RL（512 卡 × ~198 h）、造 SFT 数据（5K GPU·h）、R1 的训练（512 卡 × ~80 h）。
**明确不在内 / 未说明**：V3-Base 的预训练（V3 报告：全部训练 2.788M H800 GPU 小时、$5.576M，`deepseek-v3` §1）；
30B 小模型在 A100 上的准备实验（文中提到但没计价）；PRM/MCTS 等失败尝试；Qwen-32B/Qwen2-Math-7B 对照实验；
蒸馏 6 个小模型；两个奖励模型的训练（未说明是否计入）；人员与数据标注。
审稿人原本要求的范围更大：“including cost of RL and all the stages of R1-zero and R1, distillation, etc.”（`r1-peer-review`）。SI Table 4 只给了三行，蒸馏成本没有列出。

### 3.13 R1 vs R1-0528（`r1-0528-card` §2 表）

| 基准 | R1 | R1-0528 |
|---|---|---|
| MMLU-Redux (EM) | 92.9 | 93.4 |
| MMLU-Pro (EM) | 84.0 | 85.0 |
| GPQA-Diamond (Pass@1) | 71.5 | 81.0 |
| SimpleQA (Correct) | 30.1 | 27.8 |
| FRAMES (Acc.) | 82.5 | 83.0 |
| Humanity's Last Exam (Pass@1) | 8.5 | 17.7 |
| LiveCodeBench (2408-2505) (Pass@1) | 63.5 | 73.3 |
| Codeforces-Div1 (Rating) | 1530 | 1930 |
| SWE Verified (Resolved) | 49.2 | 57.6 |
| Aider-Polyglot (Acc.) | 53.3 | 71.6 |
| AIME 2024 (Pass@1) | 79.8 | 91.4 |
| AIME 2025 (Pass@1) | 70.0 | 87.5 |
| HMMT 2025 (Pass@1) | 41.7 | 79.4 |
| CNMO 2024 (Pass@1) | 78.8 | 86.9 |
| BFCL_v3_MultiTurn (Acc) | - | 37.0 |
| Tau-Bench (Pass@1) | - | 53.5 (Airline) / 63.9 (Retail) |

DeepSeek-R1-0528-Qwen3-8B（把 R1-0528 的 CoT 蒸馏进 Qwen3-8B Base）：AIME 24 86.0、AIME 25 76.3、HMMT Feb 25 61.5、GPQA Diamond 61.1、LCB (2408-2505) 60.5；模型卡称 AIME 2024 上 “surpassing Qwen3 8B by +10.0%”。

### 3.14 与 DeepSeekMath GRPO 设置的对比（第 8 课 ↔ 第 13 课）

| 项 | DeepSeekMath（2402.03300 §4.2） | R1-Zero（v2 §2.1） |
|---|---|---|
| 策略学习率 | 1e-6 | 3e-6 |
| KL 系数 $\beta$ | 0.04 | 0.001 |
| 每题采样 $G$ | 64 | 16 |
| 最大长度 | 1024 | 32,768 → 65,536 |
| 训练 batch | 1024 | 512 |
| 参考模型 | 未提重置 | 每 400 步重置为当前策略 |

DeepSeekMath 原文：“For GRPO, we set the learning rate of the policy model as 1e-6. The KL coefficient is 0.04. For each question, we sample 64 outputs. The max length is set to 1024, and the training batch size is 1024.”

### 3.15 主要基准数（R1 与 R1-Zero；v2 Table 3 / Table 8 / Table 12）

| 基准 | V3-Base | V3 | R1-Zero | Dev1 | Dev2 | Dev3 | R1 | o1-1217 |
|---|---|---|---|---|---|---|---|---|
| AIME 2024 (Pass@1) | - | 39.2 | 77.9 | 59.0 | 74.0 | 78.1 | 79.8 | 79.2 |
| MATH-500 (Pass@1) | - | 90.2 | 95.9 | 94.2 | 95.9 | 95.4 | 97.3 | 96.4 |
| CNMO 2024 (Pass@1) | - | 43.2 | 88.1 | 58.0 | 73.9 | 77.3 | 78.8 | - |
| GPQA Diamond (Pass@1) | - | 59.1 | 75.8 | 66.1 | 70.7 | 71.2 | 71.5 | 75.7 |
| LiveCodeBench (Pass@1-COT) | - | 36.2 | 50.0 | 57.5 | 63.5 | 64.6 | 65.9 | 63.4 |
| Codeforces (Rating) | - | 1134 | 1444 | 1534 | 1687 | 1746 | 2029 | 2061 |
| Codeforces (Percentile) | - | 58.7 | 80.4 | 84.5 | 90.5 | 92.1 | 96.3 | 96.6 |
| IF-Eval (Prompt Strict) | 58.6 | 86.1 | 46.6 | 71.7 | 72.0 | 78.1 | 83.3 | - |
| AlpacaEval2.0 (LC-winrate) | - | 70.0 | 24.7 | 50.1 | 55.8 | 62.1 | 87.6 | - |
| ArenaHard (GPT-4-1106) | - | 85.5 | 53.6 | 77.0 | 73.2 | 75.6 | 92.3 | - |
| Aider-Polyglot (Acc.) | - | 49.6 | 12.2 | 6.7 | 25.6 | 44.8 | 53.3 | 61.7 |
| SWE Verified (Resolved) | - | 42.0 | 43.2 | 39.6 | 44.6 | 45.6 | 49.2 | 48.9 |
| MMLU-Pro (EM) | 64.4 | 75.9 | 68.9 | 74.1 | 83.8 | 83.1 | 84.0 | - |
| C-SimpleQA (Correct) | - | 68.0 | 66.4 | 58.8 | 64.2 | 66.9 | 63.7 | - |

蒸馏与对照（v1 Table 5/6 = v2 Table 15/16）：

| 模型 | AIME24 pass@1 | AIME24 cons@64 | MATH-500 | GPQA | LCB | CF rating |
|---|---|---|---|---|---|---|
| GPT-4o-0513 | 9.3 | 13.4 | 74.6 | 49.9 | 32.9 | 759 |
| Claude-3.5-Sonnet-1022 | 16.0 | 26.7 | 78.3 | 65.0 | 38.9 | 717 |
| o1-mini（仅 v1） | 63.6 | 80.0 | 90.0 | 60.0 | 53.8 | 1820 |
| QwQ-32B-Preview | 50.0（模型卡 44.0） | 60.0 | 90.6 | 54.5 | 41.9 | 1316 |
| R1-Zero-Qwen-32B / Qwen2.5-32B-Zero | 47.0 | 60.0 | 91.6 | 55.0 | 40.2 | - |
| Distill-Qwen-1.5B | 28.9 | 52.7 | 83.9 | 33.8 | 16.9 | 954 |
| Distill-Qwen-7B | 55.5 | 83.3 | 92.8 | 49.1 | 37.6 | 1189 |
| Distill-Qwen-14B | 69.7 | 80.0 | 93.9 | 59.1 | 53.1 | 1481 |
| Distill-Qwen-32B | 72.6 | 83.3 | 94.3 | 62.1 | 57.2 | 1691 |
| Distill-Llama-8B | 50.4 | 80.0 | 89.1 | 49.0 | 39.6 | 1205 |
| Distill-Llama-70B | 70.0 | 86.7 | 94.5 | 65.2 | 57.5 | 1633 |

### 3.16 安全数字（v2 §D.3）

- 标准基准（Table 9，括号内为不带风控的纯模型；只有 HarmBench 与平均分有括号）：R1 SST 97.5 / BBQ 96.6 / ART 96.2 / XSTest 95.3 / DNA 94.8 / HarmBench 89.3 (35.0) / 平均 95.0 (85.9)。HarmBench 差的主因是知识产权类（被要求生成歌词时不拒绝）。
- 内部基准：4 大类 28 子类，每子类 20 道中文题再译成英文，共 1,120 题（Table 10）。R1 不带风控 Unsafe 25.2% / Rej 5.6%；带风控 8.5% / 27.3%。
- 多语言：译成 50 种语言，共 9,330 题；评分 safe 5 / rejection 4 / unsafe 0。带风控 R1 85.9%、V3 86.5%、Claude-3.7-Sonnet 88.3%；不带风控 R1 74.2%、V3 75.3%、GPT-4o 75.2%。
- 越狱：2,232 条越狱模板。R1（不带风控）Unsafe 25.2% → 85.9%（+60.7）；带风控 8.5% → 4.3%（−4.2），但拒答 27.3% → 87.3%（Table 11）。

---

## 4. 复算的数字

所有数字在 Python 3.11 中计算（脚本在 scratchpad 的 `compute.py`、`grpo.py`、`fig1a.py`、`dig_len.py`；关键表达式抄在下面）。

### 4.1 成本

| # | 表达式 | 值 | 备注 |
|---|---|---|---|
| N1 | `64*8` | 512 张 H800 | |
| N2 | `512*198` | 101,376 GPU·h ≈ 101K ✓ | R1-Zero |
| N3 | `512*80` | 40,960 ≈ 41K ✓ | R1；若按“about 4 days”= 96 h 则 `512*96` = 49,152，与表不符 |
| N4 | `101+5+41` | 147K GPU·h ✓ | |
| N5 | `147*2` | $294K ✓；分项 `101*2, 5*2, 41*2` = 202, 10, 82 | |
| N6 | `294/5576` | 0.0527 → R1 后训练成本约为 V3 全部训练成本的 5.3% | V3：$5.576M |
| N7 | `5.576+0.294` | $5.870M（V3 + R1 两份公开账单相加；仍不含研究与失败实验） | |
| N8 | `198*3600/10400` | 68.5 秒 / 训练步（R1-Zero 墙钟时间，含 rollout） | 推算 |
| N9 | `512*198/10400` | 9.75 GPU·h / 训练步 | 推算 |

### 4.2 R1-Zero 的训练日程

| # | 表达式 | 值 | 备注 |
|---|---|---|---|
| N10 | `32*16` | 512 条回答 / 步 ✓（“batch size of 512”） | |
| N11 | `8192/512` | 16 → 每次 rollout 供 16 个训练步（16 个 mini-batch × 1 inner epoch） | |
| N12 | `10400/16` | 650 次 rollout | 推算 |
| N13 | `10400/400` | 26 次参考模型重置 | 推算 |
| N14 | `10400*32` | 332,800 次“题目出现” | |
| N15 | `332800/1.6` | 208,000 道不重复题（由“1.6 epochs”反推） | 与 Table 4 推理题 `26+17+22+15` = 80K（+8K bug fixing = 88K）**对不上**，见 §7 E14 |
| N16 | `(10400-8200)/10400` | 21.2% 的步数在 64K 长度上限下训练 | |
| N17 | `512*32768`, `512*65536` | 每步生成上限 16,777,216 / 33,554,432 tokens | |
| N18 | `10400*512` | 5,324,800 条 rollout 回答 | |
| N19 | Σ(512 × 数字化平滑长度) | ≈ 3.24×10¹⁰ tokens（前 8,200 步 1.80×10¹⁰，后 2,200 步 1.44×10¹⁰） | **估算**：假设图 1b 的“length”单位是 token（图未标单位），10,200–10,400 步按末值外推 |
| N20 | `3.24e10/(101376*3600)` | ≈ 89 tokens / GPU·秒（含训练时间的平均产出率） | 粗估，只作量级 |

### 4.3 第二阶段与数据

| # | 表达式 | 值 |
|---|---|---|
| N21 | `400/1700` | 23.5%：第二阶段中带偏好奖励的步数占比 |
| N22 | `395285+211129+10124+10395+177812` | 804,745 ✓ |
| N23 | 推理部分 `804745-177812` | 626,933（“about 600k”）；一般部分 177,812（“approximately 200k”）；推理占 77.9% |
| N24 | `Σ n_k × avg_tok_k / 804745` | 5,355.3 ✓（表中 Total 行）；总 token ≈ 4.31×10⁹；2–3 epochs → 8.6–12.9×10⁹ tokens |
| N24b | token 占比 | Math 55.9%、Code 36.4%、General 5.9%、STEM 1.2%、Logic 0.7% |
| N25 | `87.6-62.1`, `92.3-75.6` | AlpacaEval +25.5 **百分点**（相对 +41.1%），ArenaHard +16.7 百分点（相对 +22.1%）→ 论文 “25% / 17%” 指百分点 |
| N26 | `143.7+10*11.3` | 256.7 ✓ > 251.5（o1：`141+10*12`=261 ✓；人类 `61.7+10*6.2`=123.7 ✓；GPT-4o 104 ✓；V3 131.3 ✓） |
| N27 | `11.3/15`, `12/15` | 0.753（“75%”）、0.800（“80%”） ✓ |
| N28 | `8793/711` | 12.4 倍 token（“an order of magnitude”） |
| N29 | `77.9-71.0` | v1→v2 R1-Zero AIME +6.9 点 |

### 4.4 GRPO 算例（二值奖励，$G=16$）

| # | $k$ 正确 | $p$ | 总体 std | $\hat A_+$ | $\hat A_-$ | 样本 std（ddof=1） | $\hat A_+$ | $\hat A_-$ |
|---|---|---|---|---|---|---|---|---|
| N30a | 1 | 0.0625 | 0.2421 | 3.8730 | −0.2582 | 0.2500 | 3.7500 | −0.2500 |
| N30b | 2 | 0.125 | 0.3307 | 2.6458 | −0.3780 | 0.3416 | 2.5617 | −0.3660 |
| N30c | 4 | 0.25 | 0.4330 | **1.7321** | **−0.5774** | 0.4472 | 1.6771 | −0.5590 |
| N30d | 8 | 0.5 | 0.5000 | 1.0000 | −1.0000 | 0.5164 | 0.9682 | −0.9682 |
| N30e | 12 | 0.75 | 0.4330 | 0.5774 | −1.7321 | 0.4472 | 0.5590 | −1.6771 |
| N30f | 15 | 0.9375 | 0.2421 | 0.2582 | −3.8730 | 0.2500 | 0.2500 | −3.7500 |

- N31 闭式 `sqrt((1-p)/p)` 与数值完全一致；每行 `sum(A)`≈0（1e-16 量级），`sum(A**2)` = 16.00（总体 std）。
- N32 全对或全错：std = 0，式 (3) 分母为 0；常见实现加 eps 后 $\hat A$ 全为 0（论文未说明，§8）。
- N33 平移/缩放不变：`adv(r)`、`adv(r+1)`、`adv(2*r)` 对 k=4 都给 1.7321。
- N34 规则奖励混合例（acc+format）：`r=[2]*3+[1]*10+[0]*3` → 均值 1.0，std 0.6124，$\hat A$ = +1.633（对且格式对）/ **0.000**（错但格式对）/ −1.633（格式都不对）。
- N35 加语言一致性奖励例：acc 前 4 条为 1，LC 取 `[1.0,0.9,0.6,1.0, 1.0,0.95,1.0,0.5, 1×8]` → $\hat A$ = [1.923, 1.687, 0.980, 1.923, −0.435, −0.553, −0.435, −1.613, −0.435×8]（语言混杂的正确答案优势从 1.92 降到 0.98）。
- N36 组内“零信号”概率 $p^{16}+(1-p)^{16}$：p=0.5 → 3.05×10⁻⁵；p=0.7/0.3 → 0.00332；p=0.9/0.1 → 0.1853；p=0.95/0.05 → 0.4401；p=0.99 → 0.8515。对比 $G=64$：p=0.9 → 0.00118，p=0.95 → 0.0375。
- N37 k3 值 `x - ln x - 1`（$x=\pi_{ref}/\pi_\theta$）：x=0.5 → 0.19315；0.8 → 0.02314；0.9 → 0.00536；1.0 → 0；1.1 → 0.00469；1.25 → 0.02686；2.0 → 0.30685（全部 ≥ 0）。
- N38 ε=10 的裁剪区间 `[1-10, 1+10]` = [−9, 11]；比率恒正，下界失效，实际只在比率 > 11 时裁剪（对比常见 ε=0.2 的 [0.8, 1.2]）。
- N39 超参数倍数：β 小 `0.04/0.001` = 40 倍；lr 大 3 倍；$G$ 少 4 倍；最大长度大 32 倍 → 64 倍。

### 4.5 token 预算与 KV cache（BF16，每 token 字节 = 2 字节 × 缓存元素数）

| # | 模型 | 表达式 | B/token | 32K 序列 | 64K 序列 |
|---|---|---|---|---|---|
| N40 | R1（MLA，61 层） | `(512+64)*61*2` | 70,272 B = 68.6 KiB | 2.14 GiB | **4.29 GiB** |
| N41 | Distill-Qwen-1.5B（GQA 2×128，28 层） | `2*2*128*28*2` | 28,672 = 28 KiB | 0.875 GiB | — |
| N42 | Distill-Qwen-7B（4×128，28 层） | `2*4*128*28*2` | 57,344 = 56 KiB | 1.75 GiB | — |
| N43 | Distill-Llama-8B（8×128，32 层） | `2*8*128*32*2` | 131,072 = 128 KiB | 4 GiB | — |
| N44 | Distill-Qwen-14B（8×128，48 层） | `2*8*128*48*2` | 196,608 = 192 KiB | 6 GiB | — |
| N45 | Distill-Qwen-32B（8×128，64 层） | `2*8*128*64*2` | 262,144 = 256 KiB | **8 GiB** | — |
| N46 | Distill-Llama-70B（8×128，80 层） | `2*8*128*80*2` | 327,680 = 320 KiB | 10 GiB | — |
| N47 | R1-0528-Qwen3-8B（8×128，36 层） | `2*8*128*36*2` | 147,456 = 144 KiB | 4.5 GiB | — |

- N48 一次 rollout 的 KV 账（示意）：8,192 条 × 平均 1 万 token × 70,272 B = 5.76 TB；摊到 512 卡 ≈ 11.2 GB/卡（假设 BF16 缓存、全部同时在显存；论文没说 rollout 并发数与 KV 精度，只作量级说明）。
- N49 R1-0528 每题 token `23/12` = 1.92 倍 → 同样定价下单题生成成本约翻倍（按输出 token 计费的推论）。
- N50 发布定价（`r1-news`）：输入 $0.14/M（缓存命中）、$0.55/M（未命中），输出 $2.19/M。例：一次 AIME 题 23K 输出 token ≈ `23000*2.19/1e6` = $0.050（按 R1 发布价，R1-0528 价格未核实）。

### 4.6 图数据的一致性检验（数字化后）

- N51 图 1a 的 pass@1 点全部是 1/480 的整数倍（最大偏差 0.005 个单位）→ 与“30 题 × 16 个样本”一致；cons@16 点全部是 1/150 的整数倍（最大偏差 0.001）→ **笔者推测**是 30 题 × 5 次独立的 16 样本投票取平均（论文未说明）。
  核查复算（2026-09-24，重新解析 arXiv v2 的 `plot_aime_with_maj.svg`）：43 个 pass@1 点与 k/480 的最大偏差是 8.8e-5 个单位，43 个 cons@16 点与 k/150 的最大偏差是 1.7e-5 个单位，比原稿说的更精确。cons@16 点**不是** 1/30 的整数倍（最大偏差 0.40 个单位），所以不可能是“30 题各做一次 16 票投票”，上面的推测方向成立，但仍是推测。第一个点在第 200 步、值 75/480 = 0.15625（即正文的 15.6%）。
- N52 安全 Table 9 平均分复核：R1 `(97.5+96.6+96.2+95.3+94.8+89.3)/6` = 94.95 ✓（95.0）；纯模型把 HarmBench 换成 35.0 → 85.9 ✓；V3 96.3 ✓ / (91.5) ✓。
- N53 内部安全题数 `28*20*2` = 1,120 ✓。Claude-3.7 越狱后“安全回答比例”下降 `(100-10.7-3.6)-(100-26.2-21.9)` = 33.8 点 ✓（原文 “a 33.8% decrease”）。
- N54 MATH 难度 1 级只有 43 题：3%–5% 错误率对应 `43*0.03`=1.29、`43*0.05`=2.15 道，✓ 原文 “just 1-2 unsolved problems”。

---

## 5. 图表清单与重绘建议

### 5.1 论文中的图（值得重绘的标 ★）

| 编号（v2） | Nature/SI | 标题（原文） | 画的是什么 |
|---|---|---|---|
| ★ Fig. 1(a) | Fig. 1a | AIME accuracy of DeepSeek-R1-Zero during training | 步数 0–10,000；蓝 pass@1、红 cons@16、绿虚线人类平均 0.378（v1 图 2 为到 8,400 步、参照线 o1-0912 pass@1 74.4 / cons@64 83.3） |
| ★ Fig. 1(b) | Fig. 1b | The average response length of DeepSeek-R1-Zero on the training set during the RL process | 淡蓝原始曲线 + 深蓝平滑曲线，0–10,200 步；8.2k 附近跳变 |
| ★ Fig. 2 | Fig. 2 | The multi-stage pipeline of DeepSeek-R1 | 四列：R1-Zero（RL，规则奖励）→ 采样+过滤（Accuracy & Format）+改写（DeepSeek V3+Human）→ 冷启动 SFT → Dev1 → RL（Rule-based Reward & Lang. Consistency）→ Dev2 → 采样（Reasoning）+ DeepSeek V3（Non-Reasoning）→ 在 V3 Base 上 SFT → Dev3 → RL（Diverse Prompts; Rule-based Reward & Preference Reward）→ R1 |
| ★ Fig. 3 | SI Fig. 1 / ED Fig. 2 | Demonstration of PPO and our GRPO | PPO：策略→o→参考/奖励/价值模型→KL⊕r、v→GAE→A；GRPO：o₁…o_G→参考/奖励模型→r₁…r_G→Group Computation→A₁…A_G；“Trained Models” 黄、“Frozen Models” 蓝 |
| Fig. 4 | SI Fig. 2 | Performance of PPO and GRPO on the MATH task | 200–2,400 步；GRPO、PPO(λ=1.0)、PPO(λ=0.95) |
| Fig. 5 | SI Fig. 3 | Overview of our RL framework | Rollout / Inference / Rule-based Reward / Training 四模块 |
| ★ Fig. 6 | SI Fig. 4 | Reward hacking | 双轴：奖励分（左）上升，CodeForces test pass@1（右）下降 |
| Fig. 7 | SI Fig. 5 | LC Reward ablation | 三联：LC 奖励、LiveCodeBench、AIME，有/无 LC 奖励 |
| Fig. 8 | SI Fig. 6 | MATH 难度 1–5 级随训练 | 5 条线 |
| ★ Fig. 9 | SI Fig. 7 / ED Fig. 1 | (a) 反思词频次；(b) “wait” 频次 | 0–10,400 步 |
| Fig. 10 | SI Fig. 8 | R1、R1-Zero 与人类比较 | AIME 79.8/77.9/37.8；CF 百分位 96.3/80.4/50.0；GPQA 71.5/75.8/81.2 |
| Fig. 11–12 | SI Fig. 9–10 | Chatbot Arena 截图（2025-01-24） | 截图，不重绘 |
| Fig. 13–14 | SI Fig. 11–12 | 安全分类法、多语言安全 | |
| Fig. 15–17 | SI Fig. 13–15 | MMLU / MMLU-Pro 分科、数学分类 | |
| ★ Fig. 18 | SI Fig. 16 | Test-time compute scaling w.r.t. problem difficulty | x：Pass@1（1.0→0.0，易→难），y：平均思考 token（UnivariateSpline 平滑，s=5） |

### 5.2 本课建议的图（15 张，含 1 张可选；颜色按 style-guide §4）

**图 13.1 R1-Zero 的 AIME 曲线（账单图，据 v2 图 1a 数字化重绘）。** 数据（step, pass@1, cons@16），从 SVG 路径解析：
```
200 .156 .253 | 400 .225 .340 | 600 .246 .313 | 800 .287 .347 | 1000 .317 .447 | 1200 .371 .567
1400 .412 .553 | 1600 .377 .487 | 2000 .381 .547 | 2400 .485 .613 | 2800 .529 .733 | 3200 .512 .687
3600 .583 .767 | 4000 .600 .767 | 4200 .602 .733 | 4400 .558 .767 | 4600 .579 .693 | 4800 .569 .733
5000 .540 .707 | 5200 .631 .780 | 5400 .575 .727 | 5600 .590 .800 | 5800 .617 .800 | 6000 .644 .780
6200 .625 .800 | 6400 .688 .867 | 6800 .663 .767 | 7000 .685 .833 | 7200 .652 .800 | 7400 .677 .833
7600 .673 .813 | 7800 .669 .807 | 8000 .688 .820 | 8200 .677 .833 | 8400 .710 .867 | 8600 .738 .867
8800 .744 .867 | 9000 .735 .867 | 9200 .777 .867 | 9400 .767 .867 | 9600 .767 .867 | 9800 .752 .867
10000 .773 .867
```
参照线：人类平均 0.378（v2）；可选 v1 的 o1-0912 pass@1 0.744、cons@64 0.833。标注：8.2k 步竖线“最大长度 32,768 → 65,536”；
正文数字 77.9% 与图中最高点 0.777（9,200 步）差 0.002，**这不是数字化误差**：SVG 的坐标是矢量精确值，每个 pass@1 点与 k/480 的偏差都小于 1e-4 个单位，0.777 就是 373/480，而 77.9% ≈ 374/480 刚好差一个量化单位，图中没有这个点（见 E1）。图注里不要把 77.9% 标在曲线上。颜色：pass@1 用 `--fig-blue`，cons@16 用 `--fig-green`，人类基线 `--fig-muted` 虚线。

**图 13.2 回答长度随步数增长（账单图，据 v2 图 1b 数字化）。** 平滑曲线每 400 步取值（单位按 token 理解，图未标）：
```
0:550 400:950 800:1330 1200:1350 1600:1700 2000:2010 2400:2330 2800:3160 3200:3130 3600:3800
4000:3860 4400:4190 4800:4910 5200:6010 5600:5050 6000:5970 6400:5610 6800:6590 7200:6590
7600:7940 8000:9200 8400:10520 8800:13210 9200:12710 9600:14600 10000:13470 10200:13650
```
原始曲线最大 ≈ 19,980（≈9,525 步）；平滑曲线最大 ≈ 15,500（≈9,910 步）；最小 ≈ 440（≈127 步）。
标注：8.2k 竖线；32,768 与 65,536 两条“长度上限”水平线（用 `--fig-red` 细虚线），让读者看到平均长度离上限还有多远。

**图 13.3 “wait”与反思词频次（账单图，据 v2 图 9 数字化）。** 两个子图或双轴（建议两个子图，y 量级差 10 倍）。
(a) 反思词（step: count）：
```
200:1479 400:2192 600:1887 800:736 1000:635 1200:1678 1400:1243 1600:1170 1800:4744 2000:1041
2200:1616 2400:1120 2600:2173 2800:1066 3000:1019 3200:1750 3400:1604 3600:1113 3800:1118
4000:1369 4200:4026 4400:5305 4600:2900 4800:3481 5200:4528 5400:6770 5600:6520 5800:8666
6000:7018 6200:7015 6400:10183 6800:5433 7000:8891 7200:8006 7400:5352 7600:7169 7800:6781
8200:8660 8600:9720 8800:6637 9000:5343 9200:8921 9400:8408 9600:7777 9800:14289 10000:7218
10200:9811 10400:7081
```
(b) “wait”：
```
200–1800:0 2000:1 2200:1 2400:2 2600:1 2800:0 3000:0 3200:1 3400:0 3600:1 3800:6 4000:11
4200:21 4400:14 4600:39 4800:81 5200:48 5400:160 5600:87 5800:88 6000:154 6200:67 6400:51
6800:20 7000:23 7200:99 7400:53 7600:143 7800:126 8200:887 8600:1500 8800:334 9000:106
9200:344 9400:420 9600:550 9800:365 10000:401 10200:1008 10400:570
```
标注 4,000–7,000 “occasional usage”、8,000 之后 “significant spikes”，并画 8.2k 竖线（与最大长度改动重合，见 §7 E11）。

**图 13.4 R1-Zero 的模板与规则奖励（机制图）。** 左：模板文本框（`<think>…</think><answer>…</answer>`，prompt 位置高亮 `--fig-blue`）；
右：回答 → 两个检查器：①答案抽取（\boxed{}）→ 与参考答案比对 / 代码 → 编译运行测试用例 → 0/1；②标签检查 → 0/1；二者相加 = $Reward_{rule}$。
旁注“没有神经奖励模型”（红叉），理由三条（reward hacking、重训成本、流程复杂，C5）。

**图 13.5 GRPO 组内优势（账单图）。** 16 根柱子，k=4 正确：4 根 +1.732（`--fig-green`），12 根 −0.577（`--fig-red`），组均值线 0.25、标准差 0.433。
可加第二行 k=15（1 根 −3.873 很长的红柱）与 k=1（1 根 +3.873），说明“稀有事件得到大优势”。数据见 §4 N30。

**图 13.6 优势大小与零信号概率随正确率 p 变化（账单图）。** x = p ∈ (0,1)；曲线 $\sqrt{(1-p)/p}$（正确者）、$-\sqrt{p/(1-p)}$（错误者）；
第二个 y 轴或第二子图：$p^{16}+(1-p)^{16}$（G=16）与 $p^{64}+(1-p)^{64}$（G=64）。关键点：p=0.9 时 0.185 vs 0.0012（§4 N36）。

**图 13.7 PPO vs GRPO 结构对比（机制图，据 v2 图 3 重绘）。** 四模型 vs 三模型：PPO 有 Policy（训练）、Value（训练）、Reward（冻结）、Reference（冻结）；
GRPO 去掉 Value，改为 “Group Computation”。与第 3、8 课的同名图保持一致（颜色：训练中的模型 `--fig-blue`，冻结模型 `--fig-muted`）。
可加注：KL 在 PPO 中进奖励（逐 token 累加 → 隐式惩罚长度，C17），在 GRPO 中进损失。

**图 13.8 R1 四阶段流水线（机制图，据 v2 图 2 重绘）。** 节点与边见 §5.1 Fig. 2 行。注意两个易错点要画对：
①冷启动数据来自 R1-Zero 的采样 + 过滤 + V3/人工改写；②第二次 SFT 的起点是 **V3-Base**（不是 Dev2），Dev2 只提供推理样本。
每个节点旁标数量：冷启动“thousands”；推理 RL（ε=10、LC 奖励）；SFT 804,745 条（626,933 推理 + 177,812 一般）；全场景 RL 1,700 步（最后 400 步加偏好奖励，T=0.7）。

**图 13.9 各阶段能力变化（账单图，v2 Table 3）。** 分组柱状或折线，x = R1-Zero, Dev1, Dev2, Dev3, R1：
AIME 77.9/59.0/74.0/78.1/79.8；LCB 50.0/57.5/63.5/64.6/65.9；IF-Eval 46.6/71.7/72.0/78.1/83.3；AlpacaEval 24.7/50.1/55.8/62.1/87.6；ArenaHard 53.6/77.0/73.2/75.6/92.3。
看点：Dev1 推理掉点（冷启动数据太少）、Dev2 推理回升、R1 偏好类大涨。

**图 13.10 蒸馏 vs 小模型直接 RL（账单图，v1 Table 6 / v2 Table 16）。** 三组柱：QwQ-32B-Preview、Qwen2.5-32B-Zero、Distill-Qwen-32B；
指标 AIME pass@1 50.0/47.0/72.6、cons@64 60.0/60.0/83.3、MATH-500 90.6/91.6/94.3、GPQA 54.5/55.0/62.1、LCB 41.9/40.2/57.2。

**图 13.11 蒸馏模型 AIME 随参数量（账单图，Table 5/15）。** x = 参数量（对数轴）：Qwen-1.5B 28.9、Qwen-7B 55.5、Llama-8B 50.4、Qwen-14B 69.7、Qwen-32B 72.6、Llama-70B 70.0；
参照线 GPT-4o 9.3、Claude-3.5 16.0、o1-mini 63.6、R1 79.8。Qwen 与 Llama 用不同标记（base 不同，且 Qwen-1.5B/7B 用的是 Math 版 base）。

**图 13.12 训练账单（账单图）。** 条形：R1-Zero 101K、SFT 数据 5K、R1 41K GPU·h（= $202K/$10K/$82K），总计 147K / $294K；
旁边一根对数刻度的 V3 全部训练 2.788M GPU·h（$5.576M），标注“5.3%”。用虚线框写“不含：V3-Base 预训练、30B A100 准备实验、失败尝试、蒸馏、RM 训练（未说明）”。

**图 13.13 reward hacking（账单图，据 v2 图 6 数字化）。** 右轴 CodeForces test pass@1（红）：
```
50:.3351 100:.3338 150:.3432 200:.3324 250:.3338 300:.3446 350:.3459 400:.3378
450:.3351 500:.3216 550:.3176 600:.3378 700:.2892
```
左轴奖励分（蓝点，每 10 步一点，70 个）：起点 ≈3.52（第 5 步），最低 ≈3.32（第 45 步），后期 ≈4.64（第 555、655 步），末点 ≈4.32（第 695 步）。
看点：奖励一路涨、真实代码能力在 ~700 步掉到 0.289——这就是为什么第二阶段只让偏好奖励参与最后 400 步。

**图 13.14 测试时计算：R1 vs GPT-4o（账单图，v2 §E.4）。** 两点散点或双柱：R1 8,793 tokens / 61.8%；GPT-4o 711 tokens / 24.7%；
再加 AIME 2024 的四个数：R1 pass@1 79.8、cons@64 86.7、pass@64 90.0；GPT-4o pass@1 9.3 → cons@64 13.4。
（v2 图 18 的平滑曲线目测范围约 8,400–16,200 tokens，与正文“<7,000 / >18,000”不完全吻合，**不要**据图 18 重绘具体曲线，见 §7 E16。）

**图 13.15（可选）R1 → R1-0528（账单图，模型卡）。** AIME 2025 70.0 → 87.5；平均 token 12K → 23K；HLE 8.5 → 17.7；GPQA 71.5 → 81.0；Aider 53.3 → 71.6。标“同一架构（config 相同），只改后训练”。

---

## 6. 算例与练习素材

**X1（计算，GRPO 优势）** 一道题采 16 个回答，4 个对。用总体标准差算每个回答的 $\hat A$；再用样本标准差算一遍。
答：均值 0.25，σ = √(0.25×0.75) = 0.4330；对的 +0.75/0.4330 = **+1.732**，错的 −0.25/0.4330 = **−0.577**；样本 σ = 0.4472 → +1.677 / −0.559。检验：4×1.732 − 12×0.577 ≈ 0。

**X2（推导，闭式解）** 证明二值奖励下 $\hat A_+=\sqrt{(1-p)/p}$、$\hat A_-=-\sqrt{p/(1-p)}$，并说明 $\hat A_+\hat A_-=-1$、$\sum\hat A_i^2=G$。
答：见 §2.5；$\sum\hat A_i^2 = k\frac{1-p}{p}+(G-k)\frac{p}{1-p}=G(1-p)+Gp=G$。

**X3（概念+计算，格式奖励何时起作用）** 规则奖励 = 准确 + 格式。(a) 16 个回答都格式正确、4 个答对，格式奖励对优势有无影响？(b) 3 个“对且格式对”、10 个“错但格式对”、3 个“格式错也答错”，求三类的 $\hat A$。
答：(a) 无影响（全组 +1 平移，优势不变，§4 N33）。(b) 奖励 2/1/0，均值 1.0，σ=0.6124 → +1.633 / 0.000 / −1.633。“错但格式对”恰好零优势——格式奖励把“格式错”的回答推到最下面。

**X4（计算，零信号的比例）** 若某题模型正确率 p=0.9，G=16 时这题“全对”的概率是多少？整组不提供梯度的概率？若 G=64 呢？
答：0.9¹⁶ = 0.1853；再加全错 0.1¹⁶ ≈ 0 → 0.1853。G=64：0.9⁶⁴ = 0.00118。结论：G 越小，越需要把“已经会做”的题过滤掉（这是推论，论文没有描述难度过滤策略）。

**X5（计算，k3 估计器）** 某 token 上 $\pi_{ref}/\pi_\theta=0.8$ 与 1.25，分别求式 (2) 的值；为什么它永远 ≥ 0？
答：0.8 − ln0.8 − 1 = 0.02314；1.25 − ln1.25 − 1 = 0.02686。$f(x)=x-\ln x-1$，$f'(x)=1-1/x$，在 x=1 处取最小值 0。

**X6（计算，训练账单）** R1-Zero 用 64×8 张 H800 训了约 198 小时，按 $2/GPU·h 值多少钱？R1 全部后训练 $294K 占 V3 训练账单（$5.576M）的百分之几？
答：512×198 = 101,376 GPU·h ≈ $202,752（表中记 101K/$202K）；294/5576 = 5.27%。追问：这 $294K 不包括什么？（§3.12 覆盖范围）

**X7（计算，训练日程）** 每步 32 题 × 16 回答；每次 rollout 生成 8,192 条回答。10,400 步一共多少次 rollout？参考模型换了几次？“1.6 epochs”意味着训练集有多少道题？与 Table 4 比较。
答：8192/512 = 16 步/rollout → 650 次；10400/400 = 26 次；10400×32/1.6 = 208,000 道，远多于 Table 4 的 80K 推理题（+8K bug fixing）——论文没交代 R1-Zero 的训练集规模，这里的不一致值得讨论（§7 E14）。

**X8（计算，KV cache 与 token 预算）** R1（61 层 MLA，$d_c=512$，$d_h^R=64$，BF16）生成一条 65,536 token 的回答要多少 KV cache？Distill-Qwen-32B（64 层，8 个 KV 头 × 128 维）生成 32,768 token 呢？
答：(512+64)×61×2 = 70,272 B/token → ×65,536 = 4.61 GB（4.29 GiB）；Qwen-32B：2×8×128×64×2 = 262,144 B/token → ×32,768 = 8 GiB。32B 稠密模型每 token 的缓存是 671B 的 R1 的 3.7 倍（262,144/70,272），这是第 9 课 MLA 的回报。

**X9（计算，SFT 数据量）** 用 Table 5 求 80 万 SFT 数据的总 token 数；2–3 个 epoch 一共训练多少 token？数学与代码占 token 的几成？
答：Σ nᵢ×tokᵢ = 4.31×10⁹；×2–3 = 8.6–12.9×10⁹；数学 55.9% + 代码 36.4% = 92.3%（样本数上一般数据占 22.1%，但 token 上只占 5.9%）。

**X10（计算+概念，语言一致性奖励）** 一条中文题的 CoT 有 200 个“词”，其中 180 个是中文词，式 (7) 给多少？若这条回答答对了，它的总奖励是多少？为何作者说这个奖励会让性能略降还要加？
答：180/200 = 0.9；总奖励 = 1（准确）+ 0.9（语言）[+ 格式]；理由：“this reward aligns with human preferences, making it more readable”（C22），消融显示数学持平、代码略降（§B.6）。

**X11（计算，评测指标）** R1 在 AIME 2024 上 pass@1 79.8%、cons@64 86.7%、pass@64 90.0%。三个数分别回答什么问题？为什么 GPT-4o 做 64 次多数投票只从 9.3% 升到 13.4%？
答：pass@1 = 单次采样的期望正确率；cons@64 = 64 个样本多数投票的正确率；pass@64 = 64 个中至少一个对的比例。多数投票只能放大“单样本已经常对”的题；GPT-4o 单样本正确率太低，投票放不大（C33 原文：“samples are generated independently rather than building upon each other”）。

**X12（计算，USAMO 指数）** R1 在 AMC 12 2024 得 143.7、AIME 2025 得 11.3/15，USAMO 指数是多少？是否过线（251.5）？
答：143.7 + 10×11.3 = 256.7 > 251.5，过线；o1-1217 为 261.0；人类平均 123.7。

**X13（概念）** 为什么 PPO 的逐 token KL 惩罚会压制回答变长，而 GRPO 的做法不会？R1 为什么还要每 400 步重置参考模型？
答：PPO 把 $-\beta\,\mathrm{KL}_t$ 加进每个 token 的奖励，回报是累加的，回答越长累计惩罚越大；GRPO 把 KL 作为损失项并按 token 平均（DeepSeekMath 写法），不随长度累加。训练上万步后策略远离初始参考，重置参考是在“可探索范围”和“稳定性”之间折中（C17）。

**X14（概念，失败尝试）** 用论文的三条理由解释 PRM 为什么没用上；用两条理由解释 MCTS。它们各自在什么场景下仍然有用？
答：见 C30、C31；PRM 仍可“rerank the top-N responses … or assist in guided search”；MCTS 配合预训练价值模型仍能提升推理时表现，难在“iteratively boosting model performance through self-search”。

---

## 7. 勘误与常见误解

**E1 “R1-Zero 在 AIME 上从 15.6% 升到 71.0%”——只对 v1 成立。** v2 与 Nature 写 77.9%（训练延长到 10,400 步、8.2k 步后最大长度加倍）。两个数出自同一次训练：71.0% 就是训练曲线上第 8,400 步的点（v1 图 2 末点 = v2 图 1a 第 8,400 步，0.710 ≈ 341/480）。77.9% 却**不在** v2 图 1a 上：图中 pass@1 最高 0.777（9,200 步），最后一点 0.773（10,000 步），第 10,200、10,400 步没有画。它可能来自最终检查点，或来自 Table 3 的评测协议（AIME 取 k=64，而训练曲线每题 16 个样本），论文没说。写课用 Nature 版 77.9%，并脚注 v1 的 71.0%；**不要**写成“曲线终点 77.9%”。

**E2 “15.6% 是 V3-Base 的起点”——不准确。** 图 1a 的第一个点在第 200 步（数字化：x=200，0.156），第 0 步（V3-Base 本身）没有画出。原文措辞是 “initial”。

**E3 “86.7% 是 64 次多数投票”——需分版本。** v1 Table 2 的 86.7 是 cons@64；训练曲线上的红线是 cons@16，也在 8,400 步后稳定在 0.867；v2/Nature 正文改称 “self-consistency decoding”，没写投票数。

**E4 “R1 只花了 $294K”——偷换了范围。** $294K 是 R1-Zero RL + 造 SFT 数据 + R1 训练在 $2/GPU·h 假设下的租金折算（SI Table 4），不含 V3-Base 预训练（$5.576M，且那个数本身也不含研究与消融）、30B 模型的 A100 准备实验、失败尝试、蒸馏、奖励模型训练（未说明）、人员与标注。

**E5 反思词由 “33 位专家”选定——arXiv v2 的笔误。** SI §3.2 写 “3 human experts”，Nature Extended Data Fig. 1 写 “a panel of three human experts”。以 Nature/SI 为准：**3 位**。

**E6 “660B R1-Zero and R1”（SI §2.4.4 / v2 §B.4.4）与全文其他处的 671B 不一致。** 模型卡、Table 8、config 都是 671B 总参数 / 37B 激活。写课一律用 671B/37B，必要时脚注 660B 的出处。（更正：原稿写“模型卡、Table 8、config 都是 671B”。`config.json` 只给架构超参数，不写参数量；写 671B/37B 的是模型卡与 v2 Table 8 / Table 12 的表头行。config 只能证明 R1 与 V3 架构相同。）

**E7 “about 4 days, or roughly 80 hours”自相矛盾。** 4 天 = 96 小时；表中 41K GPU·h = 512 × 80 h，所以表是按 80 h 算的。写课引用 GPU·h 与美元，不要引用“4 天”。

**E8 “R1 的第二次 SFT 是在 Dev2 上继续训”——错。** v1：“retrain the DeepSeek-V3-Base model”；v2 图 2：SFT 的输入模型是 DeepSeek V3 Base。Dev2 只负责拒绝采样出推理数据。

**E9 “R1 用神经奖励模型训练推理”——错。** 按论文所述，推理数据全程用规则奖励（v2 §2.2：“we abstain from applying neural reward models—whether outcome-based or process-based—to reasoning tasks”；§3.2.2 第二阶段推理数据仍 “employs rule-based rewards”）。神经 RM（有用性、安全）只用于一般数据，且只在第二阶段最后 400 步。
DeepSeek-V3 作为 LLM 判官在论文里出现过四处：
- 拒绝采样时对照标准答案，给部分非规则可验的 SFT 数据判对错（C23）；
- 构造有用性 RM 的偏好对（Arena-Hard 提示，判 4 次，§3.6）；
- 冷启动数据改写；
- 风控系统审查（C34）。
（更正：原稿写 V3 判官“只用于”拒绝采样，范围说窄了。）另外，v2 §G.1 说 “rule-based reward models (RMs) and LLMs to assess an answer’s correctness against a predefined ground-truth—serve as robust mechanisms for mitigating issues related to reward hacking”，但没有说 LLM 判官在 R1-Zero 或 R1 的哪个 RL 阶段用过（§8 第 22 条）。所以写课时说“推理 RL 不用**训练出来的**神经 RM”最稳妥。

**E10 “蒸馏模型也做了 RL”——错。** “For distilled models, we apply only SFT and do not include an RL stage”（C27）。

**E11 “aha moment / 变长完全是模型自发的”——要加限定。** ①作者自己说 8.2k 步的性能与长度跳变来自把最大长度从 32,768 改为 65,536（C8）；数字化后 “wait” 频次的激增点（8,200: 887；8,600: 1,500）也紧挨着这次改动（**笔者观察**，论文未讨论二者关系）。
②Dr. GRPO（arXiv:2503.20783）报告 “DeepSeek-V3-Base already exhibit ''Aha moment''”，即 base 模型本来就会出现自我反思式输出，RL 放大而非无中生有。③v2 §B.3.2 作者自己提醒 “vivid reasoning patterns primarily reflect DeepSeek-engineered heuristics”（针对 R1，冷启动数据刻意偏好第一人称）。④审稿人批评 “excessive anthropomorphization”（C36）。

**E12 “R1-Zero 没有用到任何人类先验”——过度简化。** 模板（结构约束）、规则奖励、题目选择（数学、代码、STEM、逻辑）都是人为设计；V3-Base 的网页语料中含 “a significant number of OpenAI-model-generated answers”（C14）。准确说法是“没有用人工标注的推理轨迹做 SFT”。

**E13 “GRPO 的 ε 是 0.2”——R1 第一阶段是 10。** v2/Nature 写 “the GRPO clip ratio ε to 10”；R1-Zero 的 ε 未公开。ε=10 时下界 1−ε=−9 无意义，只剩上界 11（§4 N38）。不要把 PPO 常用的 0.2 当成 R1 的设置。

**E14 “1.6 epochs” 与数据表对不上（未解决）。** 10,400 步 × 32 题 / 1.6 = 208,000 道不重复题，而 Table 4 的推理类 RL 题目合计 80K（代码另有 8K bug fixing）。可能 R1-Zero 的训练集不同于 Table 4（Table 4 描述的是 R1 的 RL 数据），也可能“epoch”另有定义。论文无解释，写课时不要给出 R1-Zero 训练集的规模。

**E15 “论文的 GRPO 式子就是实现”——式子是序列级简写。** R1 论文印刷的比率是 $\pi_\theta(o_i|q)/\pi_{\theta_{old}}(o_i|q)$，没有 DeepSeekMath 的 $1/|o_i|$ 和 token 级比率（§2.1）。实现细节（token 级还是序列级、按 token 还是按样本平均）未公开；Dr. GRPO 专门讨论了 $1/|o_i|$ 与 std 归一化带来的长度偏置（“Dr. GRPO introduces simple yet significant modifications to address the biases in GRPO … by removing the length and std normalization terms”）。

**E16 测试时 token 数：正文与图 18 不完全吻合。** 正文：简单题 “fewer than 7,000”、最难题 “more than 18,000”，平均 8,793；图 18 的平滑曲线目测在约 8,400–16,200 之间（标准差带可到 18,000 以上）。图经 UnivariateSpline 平滑（s=5），不能用来复现正文数字。写课只引用正文数字。

**E17 “R1 的 Codeforces 从 2029 掉到 1530”——不是退步，是换了评测。** 论文：2029（10 场 Div.2 + 专家测试用例）；R1-0528 模型卡里 R1 的 “Codeforces-Div1 (Rating)” 为 1530、R1-0528 为 1930。同理 LiveCodeBench：论文 65.9（2408–2501 窗口）vs 模型卡 63.5（2408–2505 窗口）。跨来源比较要对齐评测协议。

**E18 QwQ-32B-Preview 的 AIME pass@1：论文 50.0，模型卡 44.0。** v1 Table 5/6、v2 Table 16 都是 50.0；`r1-card` 蒸馏表写 44.0。
另外，模型卡**主对比表**（“DeepSeek-R1 Evaluation”）中 Claude-3.5 / GPT-4o 的 LiveCodeBench 为 33.8 / 34.2，V3 为 “-”；论文 v1 Table 4 / v2 Table 8 为 38.9 / 32.9 / 36.2。模型卡自己的**蒸馏表**里这两个基线又是 38.9 / 32.9，与论文一致，所以模型卡内部就不一致（2026-09-24 重新抓取 README 核对）。以论文为准，并注明分歧。

**E19 “AlpacaEval 提升 25%”是百分点。** 62.1 → 87.6 是 +25.5 个百分点（相对 +41%）；ArenaHard 75.6 → 92.3 是 +16.7 个百分点（相对 +22%）。

**E20 “800K = 600K 推理 + 200K 非推理”只是约数。** 精确为 804,745 = 626,933（数学+代码+STEM+逻辑）+ 177,812（一般）。

**E21 “语言一致性奖励提升了能力”——相反。** 论文明确说有轻微性能下降（代码），收益是可读性（C22）。

**E22 “小模型不能从 RL 获益”——过度概括。** 论文结论是：在同等条件下，32B 上的大规模 RL 不如蒸馏。但 Qwen2-Math-7B-Zero 从 **Qwen2-Math-7B base** 出发，经约 1 万步 RL 后 AIME 2024 达 22.3%、AIME 2025 达 18.1%，高于作对照的 Qwen2-Math-7B-Instruct（7.9% / 4.6%）与 GPT-4o-0513（9.3%）（v2 Table 17）。
（更正：原稿写“从 Instruct 版 7.9% 到 22.3%”，读起来像 RL 从 Instruct 模型出发。其实 Instruct 只是并列的对照，不是起点。）
作者还说 “applying RL to these distilled models yields significant further gains”，但这句**只在 v1 §3.2**，v2 删掉了。v2 只保留 “even though incorporating RL could substantially boost model performance”（§F），引用时注明版本。另，G.1 的“7B dense、16B MoE 失败”是作者早期的 base（未具名），不是 Qwen2-Math-7B。

**E23 “R1 发布于 2025-01-22”——那是 arXiv v1 的日期。** 模型与 API 在 2025-01-20 发布（`r1-news`）；Nature 于 2025-09-17 在线发表（卷期日期 2025-09-18）。

**E24 “R1-0528 是新架构 / 新基座”——不是。** `config.json` 与 R1 逐字节相同；模型卡称 “minor version upgrade”，改进来自后训练（“increased computational resources and introducing algorithmic optimization mechanisms during post-training”）。

**E25 R1-Zero 的缺陷清单版本差异。** 论文：可读性差、语言混杂（v2 加“规则 RL 只覆盖推理任务”）；模型卡另加 “endless repetition”。

---

## 8. 未能核实 / UNVERIFIED

1. **式 (3) 的 std 是总体还是样本标准差**；std = 0（全对/全错）时如何处理（加 eps？跳过？）——论文未说明。
2. **实现是 token 级还是序列级比率、损失按 token 还是按样本平均**——论文只印了序列级简写（§2.1、E15）。
3. **R1-Zero 的裁剪系数 ε**；第二阶段 RL 的 ε、β、学习率（只说 “retains most of the parameters from the first stage”）。
4. **R1 第一阶段 RL 的步数**；冷启动数据的确切条数（只说 “thousands”）。
5. **41K GPU·h（“DeepSeek-R1”）具体覆盖哪些阶段**（两次 SFT + 两次 RL 是否全含）；奖励模型训练、蒸馏、30B A100 实验的成本是否计入——SI 未分项。
6. **“1.6 training epochs” 对应的训练集规模**（与 Table 4 的 80K/88K 不符，E14）。
7. **式 (7) 中“词”的切分方式**（中文如何计词、代码与公式是否计入）。
8. **图 1b 的纵轴单位**（“Average length per response”，未写 token 还是字符）；本笔记的 3.24×10¹⁰ token 估算依赖“单位 = token”的假设。
9. **cons@16 数据点为 1/150 的整数倍的原因**（笔者推测“30 题 × 5 次投票平均”），以及 AIME 2024 共 30 题——论文未写题数（与 pass@1 点为 1/480 整数倍一致，但仍是推断）。
10. **开源的 DeepSeek-R1-Zero 权重对应哪一步检查点**（71.0% 还是 77.9% 那一版）。
11. **R1-0528 的训练细节**（额外算力多少、“algorithmic optimization mechanisms”是什么）——只有模型卡，无论文；R1-0528 的 API 价格本次未核实。
12. **Qwen2-Math-7B-Zero 的超参数**（只有 “approximately 10,000 policy gradient update steps”）。
13. **Table 9 中 HELM 来源的四个基准（SST、BBQ、ART、XSTest）测的是带风控的服务还是纯模型**——表注只说括号内是纯模型，而这四列没有括号。
14. **Table 10 “Overall” 列的加权方式**：R1 四类 Unsafe 的算术平均为 24.5，而 Overall 为 25.2，说明按子类题数加权，但各类题数未给出。
15. **Dr. GRPO（2503.20783）的正式发表会议**（2026-09-24 查 arXiv 摘要页，没有 journal-ref 或 comments 字段）；其结论只作为“他人的批评性复现”引用，本笔记未复核其实验。
16. **Nature News & Views（Ippolito & Zhang）正文**未读；只核实了标题、作者、日期。
17. **Distill 模型“slightly change their configs and tokenizers”具体改了哪些字段**——未与原 Qwen2.5/Llama 配置逐项比对。
18. **ε=10 与 “Zhibin Gou proposed a large PPO clipping strategy” 是否指同一件事**——作者贡献栏与超参数节未互相引用，本笔记是并置而非确证。
19. **R1-Zero 训练中是否使用了第 8 课的“迭代 RL”（重训奖励模型）**——R1-Zero 用规则奖励，应不涉及；但“每 400 步重置参考模型”是否就是全部的迭代机制，论文未进一步说明。
20. **v2 图 18 的具体数值**（只能目测，且与正文数字不吻合，E16）；v2 图 7（LC 消融）与图 4（PPO vs GRPO）只做了目测读数：图 4 GRPO 在 2,400 步约 0.554、PPO(λ=1.0) 约 0.536、PPO(λ=0.95) 约 0.501；图 7 无 LC 奖励时 LC 分数从约 0.97 降到约 0.87（5,000 步），有 LC 奖励时保持约 0.98–0.99。
21. **R1-Zero “77.9%” 的出处**：不是 v2 图 1a 上的任何一点（图上最高 373/480 = 77.7%，末点 10,000 步 = 77.3%）。论文没说它是第 10,400 步的检查点，还是按 Table 3 的评测协议（AIME k=64）重测的结果（E1）。
22. **LLM 判官（对照标准答案判对错）是否用于 R1-Zero/R1 的 RL 奖励**：v2 §G.1 称这是抗 reward hacking 的可靠验证器之一，但正文只说推理 RL 用规则奖励，没有说明 LLM 判官用在哪个阶段（E9）。

---

## 9. 参考文献卡片草稿

```
id: r1-arxiv
title: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
a: DeepSeek-AI (Guo, Yang, Zhang, Song, Wang, et al.), 2025, arXiv preprint, arXiv:2501.12948 (v1 2025-01-22; v2 2026-01-04)
url: https://arxiv.org/abs/2501.12948
finding: 在 DeepSeek-V3-Base 上只用规则奖励的 GRPO 训练出 R1-Zero（AIME 2024 pass@1 15.6%→71.0%/77.9%），再用冷启动 SFT、推理 RL、拒绝采样 SFT、全场景 RL 四阶段得到与 o1-1217 相当的 R1，并把 80 万条样本蒸馏进 6 个小模型。
matters: 本课的主来源；v1 与 v2 的数字差异（71.0 vs 77.9、超参数与成本只在 v2）决定了课里每个数字该引哪个版本。
```

```
id: r1-nature
title: DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning
a: Guo, D., Yang, D., Zhang, H. et al. (DeepSeek-AI), 2025, Nature 645, 633–638, doi:10.1038/s41586-025-09422-z（arXiv:2501.12948 的期刊版）
url: https://www.nature.com/articles/s41586-025-09422-z
finding: 经同行评审的版本，Methods 首次公开 R1-Zero 与 R1 两个 RL 阶段的超参数（lr 3e-6、KL 0.001、G=16、32,768→65,536、ε=10、第二阶段 1,700 步）和奖励模型细节。
matters: 课上所有超参数与“正式”数字以它为准，也用它说明“开放权重的大模型论文第一次走完 Nature 评审”的历史位置。
```

```
id: r1-si
title: Supplementary Information for: DeepSeek-R1 Incentivizes Reasoning in LLMs via Reinforcement Learning
a: DeepSeek-AI Team, 2025, Nature 645 Supplementary Information (83 pp.)
url: https://www.nature.com/articles/s41586-025-09422-z#MOESM1
finding: 给出训练成本表（R1-Zero 101K、SFT 数据 5K、R1 41K H800 GPU·h，合计 147K GPU·h = $294K）、RL 与 SFT 数据配方、安全报告与去污染流程。
matters: “$294K”这个广为流传的数字的一手出处，课上用它讲清这笔账包含什么、不包含什么，并纠正 arXiv v2 的“33 位专家”笔误。
```

```
id: r1-peer-review
title: Peer Review File — DeepSeek-R1 Incentivizes Reasoning in LLMs via Reinforcement Learning
a: Nature referees and DeepSeek-AI authors, 2025, Nature 645 Peer Review File
url: https://www.nature.com/articles/s41586-025-09422-z#MOESM2
kind: artefact
finding: 审稿人要求补充训练成本、安全评估、去污染分析，并批评拟人化表述，还要求在 o1 发布前的 base 模型上重做 RL 实验。
matters: 解释为什么 Nature/v2 版多出了成本、安全、Qwen2-Math-7B-Zero 等章节，示范科学共同体如何校正一篇爆款论文。
```

```
id: deepseekmath
title: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
a: Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., et al., 2024, arXiv preprint, arXiv:2402.03300
url: https://arxiv.org/abs/2402.03300
finding: 提出 GRPO：以组内奖励标准化代替价值网络，token 级裁剪目标加 k3 KL 估计，原设置 lr 1e-6、KL 0.04、每题 64 个样本、最大长度 1024。
matters: R1 的 RL 算法直接取自这里；对比两篇的式子与超参数（G 64→16、β 0.04→0.001、长度 1024→65,536）能看出 R1 把 GRPO 推到了什么尺度。
```

```
id: deepseek-v3
title: DeepSeek-V3 Technical Report
a: DeepSeek-AI, 2024, arXiv preprint, arXiv:2412.19437
url: https://arxiv.org/abs/2412.19437
finding: R1 与 R1-Zero 的基座 DeepSeek-V3-Base（671B 总参数/37B 激活），全部训练 2.788M H800 GPU·h，按 $2/GPU·h 为 $5.576M。
matters: 为 R1 的 $294K 提供对照尺度（约 5.3%），并提醒读者 R1 的“便宜”建立在一个昂贵的预训练基座之上。
```

```
id: dr-grpo
title: Understanding R1-Zero-Like Training: A Critical Perspective
a: Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., et al., 2025, arXiv preprint, arXiv:2503.20783
url: https://arxiv.org/abs/2503.20783
finding: 发现 DeepSeek-V3-Base 本身已出现“aha moment”式自我反思，并指出 GRPO 的长度归一化与 std 归一化带来优化偏置，提出去掉这两项的 Dr. GRPO。
matters: 课上用它给“aha moment 是 RL 凭空涌现”和“回答变长全是能力提升”两个流行说法加上限定。
```

```
id: schulman-kl
title: Approximating KL Divergence
a: Schulman, J., 2020, blog post (joschu.net), posted 2020-03-07
url: http://joschu.net/blog/kl-approx.html
kind: artefact
finding: 给出 KL 的三个蒙特卡洛估计器 k1/k2/k3，其中 k3 = (r−1) − log r（r = p/q，样本取自 q）无偏且方差低。
matters: R1 式 (2) 的 KL 项就是 k3，课上用它说明这一项为什么恒非负、为什么需要样本来自当前策略。
```

```
id: gao-rm-overopt
title: Scaling Laws for Reward Model Overoptimization
a: Gao, L., Schulman, J., Hilton, J., 2022, arXiv preprint, arXiv:2210.10760
url: https://arxiv.org/abs/2210.10760
finding: 研究针对代理奖励模型优化时真实目标先升后降的“过度优化”现象及其规模规律。
matters: R1 放弃 PRM、并把偏好奖励限制在最后 400 步时引用它作为 reward hacking 的依据，课上与 R1 的图 6 并读。
```

```
id: lightman-prm
title: Let's Verify Step by Step
a: Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., et al., 2023, arXiv preprint, arXiv:2305.20050
url: https://arxiv.org/abs/2305.20050
finding: 提出用逐步标注训练过程奖励模型（PRM）来监督数学推理的中间步骤。
matters: R1 “失败尝试”一节把 PRM 当作被放弃的路线，课上需要它来说明过程监督原本想解决什么问题。
```

```
id: ppo
title: Proximal Policy Optimization Algorithms
a: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O., 2017, arXiv preprint, arXiv:1707.06347
url: https://arxiv.org/abs/1707.06347
finding: 提出以裁剪重要性比率的代理目标做稳定的策略更新。
matters: GRPO 的 min/clip 结构来自 PPO，课上讲 ε=10 这一非常规取值时要回到 PPO 的原始动机。
```

```
id: gae
title: High-Dimensional Continuous Control Using Generalized Advantage Estimation
a: Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P., 2015, arXiv preprint, arXiv:1506.02438
url: https://arxiv.org/abs/1506.02438
finding: 提出用 λ 加权的多步 TD 残差估计优势（GAE），在偏差与方差之间折中。
matters: R1 附录显示 PPO 在 λ=0.95（常见默认）时明显差于 GRPO、λ=1.0 时接近，课上用它解释为何去掉价值网络也省掉了一个敏感超参数。
```

```
id: r1-card
title: DeepSeek-R1 model card (Hugging Face)
a: DeepSeek-AI, 2025, Hugging Face model card deepseek-ai/DeepSeek-R1
url: https://huggingface.co/deepseek-ai/DeepSeek-R1
kind: artefact
finding: 列出 R1/R1-Zero 为 671B 总参数、37B 激活、128K 上下文，给出 6 个蒸馏模型的 base 与使用建议（温度 0.5–0.7、不用系统提示、强制以 "<think>\n" 开头）。
matters: 课上讲部署与复现时的实际用法来自这里，也用它指出模型卡与论文在部分基线数字上的分歧。
```

```
id: r1-0528-card
title: DeepSeek-R1-0528 model card (Hugging Face)
a: DeepSeek-AI, 2025, Hugging Face model card deepseek-ai/DeepSeek-R1-0528
url: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
kind: artefact
finding: R1 的小版本升级：同一架构，后训练加算力与算法优化，AIME 2025 从 70.0% 升到 87.5%、每题平均 token 从 12K 增到 23K，并支持系统提示与函数调用。
matters: 课末用它说明“更长的思考”仍是 2025 年中推理能力提升的主要来源，并衔接第 15 课的混合思考与工具调用。
```

```
id: r1-config
title: DeepSeek-R1 config.json (also identical for DeepSeek-R1-Zero and DeepSeek-R1-0528)
a: DeepSeek-AI, 2025, Hugging Face repository file
url: https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/config.json
kind: artefact
finding: R1 使用 DeepseekV3ForCausalLM：61 层、d=7168、MLA（kv_lora_rank 512、rope 维 64）、256 路由专家 + 1 共享专家、每 token 激活 8 个、FP8 128×128 块量化权重。
matters: 证明 R1 没有改架构（所有差别都在后训练），也为本课的 KV cache 算例提供参数。
```

```
id: r1-news
title: DeepSeek-R1 Release 2025/01/20
a: DeepSeek, 2025, DeepSeek API Docs news
url: https://api-docs.deepseek.com/news/news250120/
kind: artefact
finding: 2025-01-20 发布 R1 与 6 个蒸馏模型，MIT 许可，API 模型名 deepseek-reasoner，价格为输入 $0.14/$0.55 每百万 token（缓存命中/未命中）、输出 $2.19。
matters: 给出 R1 真正的发布日期（早于 arXiv v1 两天）和许可变更（允许用输出做蒸馏），课上讲时间线与开放性时引用。
```

```
id: r1-0528-news
title: DeepSeek-R1-0528 Release 2025/05/28
a: DeepSeek, 2025, DeepSeek API Docs news
url: https://api-docs.deepseek.com/news/news250528
kind: artefact
finding: 2025-05-28 发布 R1-0528，宣称基准提升、幻觉减少、支持 JSON 输出与函数调用，API 用法不变。
matters: 为 R1-0528 的日期与定位提供一手出处，避免把它误写成新模型或新架构。
```

---

## 10. 补充核实（写课作者追加）

写课作者（2026-09-24）为第 13 课正文额外核实或推导的内容。引文保持英文原文。

**S1 GAE 的定义与 λ = 1 特例（课上式 13.21、13.22）。** 来源：arXiv:1506.02438 v6 PDF（Schulman et al., ICLR 2016），§3，pdftotext 转文本后阅读（希腊字母在转文本时丢失，结构与式号可辨）。
式 (16)：$\hat A_t^{GAE(\gamma,\lambda)}:=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta^V_{t+l}$；式 (17)：$\lambda=0$ 时 $\hat A_t=\delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t)$；式 (18)：$\lambda=1$ 时 $\hat A_t=\sum_l\gamma^l\delta_{t+l}=\sum_l\gamma^lr_{t+l}-V(s_t)$。原文：
> "GAE(γ, 1) is γ-just regardless of the accuracy of V, but it has high variance due to the sum of terms."
课上据此写“λ = 1 时无论 V 准不准都没有偏差，只是方差大”，并在 γ = 1、只有终点奖励时化简为 $\hat A_t=r-V(s_t)$（望远镜求和，笔者推导）。

**S2 第二次 SFT 的非推理数据与思维链过滤（v1 §2.3.3）。** 来源：arxiv.org/html/2501.12948v1。
> "For non-reasoning data, such as writing, factual QA, self-cognition, and translation, we adopt the DeepSeek-V3 pipeline and reuse portions of the SFT dataset of DeepSeek-V3."
> "because the model output is sometimes chaotic and difficult to read, we have filtered out chain-of-thought with mixed languages, long paragraphs, and code blocks."
课上 §5 与图 13.10（原图 13.9）的“写作、问答、翻译等”即据此。

**S3 MCTS 尝试的具体做法（v1 §4.2）。** 来源同上。
> "This approach involves breaking answers into smaller parts to allow the model to explore the solution space systematically."
> "To facilitate this, we prompt the model to generate multiple tags that correspond to specific reasoning steps necessary for the search."
> "For training, we first use collected prompts to find answers via MCTS guided by a pre-trained value model. Subsequently, we use the resulting question-answer pairs to train both the actor model and the value model, iteratively refining the process."

**S4 DeepSeekMath 的 GRPO 起点模型。** 课上 §2 的对比表写“DeepSeekMath-Instruct 7B”，依据第 8 课研究笔记 `lesson-08-doc.md` C24（“在 DeepSeekMath-Instruct 7B 上做 RL”），本笔记原无此条。

**S5 本课新增的派生数字（Python 3.11 复算，表达式如下）。**
- 测试时计算：`2*37e9` = 7.4×10¹⁰ FLOPs/token；`7.4e10*8793` ≈ 6.5×10¹⁴ FLOPs/题（按每 token 2·N_act，未计注意力）。
- pass@k 均匀假设：`1-0.202**64` ≈ 1 − 3×10⁻⁴⁵。
- 玩具模型一步 GRPO（单 logit、s = 0.25、G = 16、k = 4、η = 1）：梯度 `(4*sqrt(3)*0.75 + 12*(1/sqrt(3))*0.25)/16` = 0.4330 = √(p(1−p))；z：−1.0986 → −0.6656；σ = 0.3395。只减均值的梯度 = p(1−p) = 0.1875；放大倍数 1/√(p(1−p))：p=0.5 → 2，0.25 → 2.309，0.1/0.9 → 3.333。
- 逐 token 权重：`1.732/2000` = 8.66×10⁻⁴，`-0.577/8000` = −7.2×10⁻⁵。
- ε=10 例：`1.3*1.732` ≈ 2.25，`-0.577*0.7` ≈ −0.404。
- 练习 8（G = 4，奖励 2.0/1.5/1.0/1.0）：均值 1.375，总体 std 0.4146，优势 +1.508/+0.302/−0.905/−0.905。
- 零信号：`0.8**16+0.2**16` = 0.0281；`0.8**64` = 6.3×10⁻⁷。
- 搜索空间：`log10(129280)` = 5.1115；10 个 token ≈ 10⁵¹；8,793 个 token ≈ 10^44,946。
- AIME 量化：75/480 = 0.15625，341/480 = 0.7104，373/480 = 0.7771，374/480 = 0.7792；1 个百分点 ≈ 4.8 次作答。
- R1 输出费（发布价 2.19 美元/百万 token）：`8793*2.19/1e6` ≈ 0.019 美元。
- aha 题的解（笔者推导，纯数学）：$(x^2-x-a)(x^2+x-a+1)=x^4-2ax^2-x+a^2-a$；唯一实根 $x=(\sqrt{4a-3}-1)/2$；a = 3 时四次方程的根为 1、−2、(1±√13)/2，只有 1 满足原方程。

**S6 写课时对论文措辞的处理。** 课上把 R1 流水线分成“第 ① 到第 ④ 步”（图 13.10 的四列，原图 13.9），并注明论文的“RL 第一阶段 / 第二阶段”分别对应第 ② 步里的推理 RL 与第 ④ 步的全场景 RL，避免与“四阶段”混用。

**S7 教学复核时新增的推导与图（2026-09-24）。** 都是纯数学推导，不引入新的事实数字。
- §1 新增算例“一种写法怎样被统计地强化”：基线取组均值时，一种写法得到的总系数为 $G\,p(1-p)(f_+-f_-)$（$f_\pm$ 为它在答对 / 答错答卷中的出现比例）。例：$G=16$、$p=1/4$、$f_+=3/4$、$f_-=1/4$ 得 1.5；$f_+=f_-$ 得 0。
- §3 新增图 13.5（`l13-clip-zones.svg`，生成脚本 `gen/l13_clip.py`）：按式 (13.19) 画出 $\varepsilon=0.2$ 与 $\varepsilon=10$ 下正、负优势 token 的“照常学习 / 梯度为 0”区间，端点 1.2、0.8、11 由脚本计算；原图 13.5–13.15 顺延为 13.6–13.16。
- 图 13.12 的 Codeforces 百分位（R1 96.3）取自本笔记 §3.15（v2 Table 3）。

**S8 数学复核（2026-09-24）。** 逐条重推了课上所有 `::: deriv` 与行间公式，并在 Python 中重算了正文、算例、表格、练习答案与图生成脚本里的全部数字（均一致）。新增或改动的只有纯数学内容，不引入新的事实数字：
- §4 AIME 曲线的采样噪声（笔者估算）：30 题 × 16 个独立样本时，单个检查点 pass@1 的方差 $=\frac{1}{480}\cdot\frac{1}{30}\sum_q s_q(1-s_q)\le\bar s(1-\bar s)/480$（$x(1-x)$ 凹，Jensen）；$\bar s=0.7$ 时标准误 $\le\sqrt{0.21/480}=0.0209$，两检查点之差 $\le0.0296$。
- 图 13.13 说明：蒸馏学习率只在同一系列内随模型变大而单调下降（Qwen 1e-4 → 8e-5 → 7e-5 → 6e-5；Llama 5e-5 → 2e-5），跨系列不单调（Llama-8B 5e-5 < Qwen-14B 7e-5）。原说明“模型越大学习率越小”不成立，已改。
- 裁剪项在课上改记为 $\ell^{clip}$（原记 $L$，与全局符号表中的层数 $L$ 冲突）；k3 无偏性补上“softmax 概率处处为正”与“给定前缀的逐 token KL”两处前提；aha 题补上正根 $>0$ 与两次平方可逆的条件。

**S9 事实与引用复核（2026-09-25）。** 逐条对照正文的数字、引文与 `[@key]`，重取一手来源（`arxiv.py text 2501.12948v1 / v2`、`arxiv.py grep 2503.20783 / 2409.12122`、arxiv.org 摘要页、GAE v6 PDF 首页、Hugging Face config）。新核实或据此改动的：
- **71.0% 在 v1 的位置。** v1 摘要没有任何百分比（摘要原文只说 “demonstrates remarkable reasoning capabilities”）；71.0% 出现在 v1 §1 引言（“the pass@1 score on AIME 2024 increases from 15.6% to 71.0%”）、Table 2 与 §2.2.4。课上原写“正是 v1 摘要里的数字”，已改为“v1 引言、表 2 和 §2.2.4”。
- **800K 数据里的推理样本没有“改写”。** v2 §B.3.3：“generate reasoning trajectories by performing rejection sampling from the checkpoint of the first-stage RL training … we have filtered out chain-of-thought with mixed languages, long paragraphs, and code blocks.” 让 DeepSeek-V3 改写（“prompt DeepSeek-V3 to refine both the reasoning and the summaries”）只出现在冷启动数据的构造里（v2 §B.3.2）。课上 §6 原写“经过了 §5 的过滤与改写”，已改为“拒绝采样与过滤”。
- **两次 SFT 共用一组超参数。** v2 §B.4.2：“For code-start SFT and the second-stage SFT, we fine-tune DeepSeek-V3-Base for 2-3 epochs …”（原文 “code-start” 是 “cold-start” 的笔误）。编号对应 SI §2.4.2，课上引用改为 `[@r1-arxiv; @r1-si]`。
- **换大模型之后 RL 才奏效。** v2 §G.1：“With these more capable architectures, we finally observed substantial performance gains attributable to pure RL training.” 同段还说 7B、16B 小模型 “exhibited a tendency toward repetition”。支持课上“换成 32B 稠密、230B MoE 和 671B MoE 的模型之后才奏效”。
- **R1 的 cons@64。** v2 §D.1：“For AIME 2024, we also report consensus (majority vote) results using 64 samples, denoted as cons@64”；§E.4：“majority voting further improves DeepSeek-R1’s accuracy from 79.8% to 86.7%”。课上把 86.7% 称为 cons@64 有据。
- **v2 §A.3 的四点。** 价值模型 “usually of similar size as the policy model”；长 CoT 让“从部分回答预测最终奖励”更不可行；PPO 的逐 token KL 奖励 “may implicitly penalize the length of the response”；PPO 对 GAE 的 λ “highly sensitive”，λ=0.95 明显差、λ=1.0 接近 GRPO。课上“四条理由”是这四点的归纳。
- **R1 引用 Gao et al. (2022) 只有一处**：PRM 第三条限制（“it inevitably leads to reward hacking (Gao et al., 2022)”，v2 §G.2）。§B.5 讲有用性 RM 被 hack 时没有引用它。参考文献卡 `gao-rm-overopt` 的 matters 原写“并把偏好奖励限制在最后 400 步时引用它”，已改。
- **R1 引用 Lightman et al. 的方式。** v2 §G.2：“PRM is a reasonable method … (Uesato et al., 2022; Lightman et al., 2024; Wang et al., 2023a)”，是三篇之一，不是唯一代表。课上措辞已改。
- **PPO 的 ε = 0.2。** PPO 原文 §3 “say, ϵ = 0.2”（见 `lesson-03-doc.md` C41 的核对）。课上给“常见的 PPO 设置 ε=0.2”补了 `[@ppo]`。
- **GAE 的发表信息。** arXiv:1506.02438v6 PDF 首页页眉：“Published as a conference paper at ICLR 2016”。§0.1 标的“会议信息未核实”至此核实。
- **Qwen2.5-Math 底座（新增 `qwen25-math`）。** arXiv:2409.12122，“Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement”，An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li 等 16 人，2024-09-18。原文：“we compile the Qwen Math Corpus v2 for Qwen2.5-Math-1.5B/7B/72B pre-training”；“the total token count of Qwen Math Corpus v2 escalates from 700B to over 1T”；“we leverage the Qwen2.5 series base models for parameter initialization”。
- **Distill-Qwen-32B 的 config（新增 `distill-qwen32b-config`）。** 2026-09-25 重新读取：Qwen2ForCausalLM，hidden_size 5120，num_hidden_layers 64，num_attention_heads 40，num_key_value_heads 8，没有 head_dim 字段（$5120/40=128$）。与 §3.10 一致。
- **arXiv 作者顺序。** arxiv.org 摘要页（当前版本 v2）：DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, …；journal-ref “Nature volume 645, pages 633-638 (2025)”。与参考文献卡一致。
- **Dr. GRPO 两处用法。** 摘要：“DeepSeek-V3-Base already exhibit ‘Aha moment’”；§1：GRPO 的偏差 “may lead to progressively longer incorrect responses”。与课上 §3、§4 的引用一致。
- **措辞上的精确化（不引入新数字）**：R1 的 8,793 是“思考 token”、GPT-4o 的 711 是“输出 token”，课上改为分别称呼；R1-0528 的改进改为“模型卡归于后训练”；“推理写法能蒸馏、知识蒸不进去”标为笔者的解读；封面的版本警告框把“不同截点”改为“同一次训练，77.9% 对应的检查点未说明”，与 §7 E1 及 §11.2 第 1 条一致。

---

## 11. 核查记录（对抗式复核，2026-09-24）

复核人独立重取了一手来源，没有使用研究笔记留下的文本转换：
- arXiv:2501.12948 v1 与 v2 的 LaTeXML 全文（`arxiv.py text`）；
- DeepSeekMath 2402.03300 v3；
- V3 报告 2412.19437；
- Dr. GRPO 2503.20783；
- nature.com 文章页 HTML；
- SI 与 Peer Review File 的 PDF（用 `pdftotext` 重新转换）；
- Hugging Face 上的 R1 / R1-0528 README，R1 / R1-Zero / R1-0528 及 7 个小模型的 `config.json`，以及 `generation_config.json`；
- DeepSeek API 新闻页、Schulman 博客；
- arXiv v2 HTML 的图 1a、图 9b、图 2 的 SVG 源，以及 v1 图 2 的 PNG。

算术全部在 Python 中重算。

### 11.1 逐项核对（✓ = 与来源一致）

| # | 论断 | 结果 | 来源 |
|---|---|---|---|
| 1 | R1-Zero AIME：15.6→71.0（v1）/ 77.9（v2、Nature），投票 86.7 | ✓ 数字与引文；但 77.9 不在图 1a 上（已改 D1、E1、图 13.1） | v1 §1、§2.2.4；v2 §2.3；Nature Main |
| 2 | 15.6% 是第 200 步的首个画点，第 0 步未画 | ✓（v2 SVG 首点 x=200、y=75/480；v1 PNG 同） | v2 `plot_aime_with_maj.svg`；v1 Fig. 2 |
| 3 | R1-Zero 超参数（lr 3e-6、KL 0.001、T=1、G=16、32,768→65,536 @8.2k、10,400 步 = 1.6 epochs、32 题/步、batch 512、每 400 步换参考、8,192/rollout、16 mini-batch） | ✓ 逐字 | v2 §2.1；Nature Methods |
| 4 | R1 第一阶段 ε=10 及其余超参数；裁剪系数的两难引文 | ✓ | v2 §3.2.1；Nature Methods |
| 5 | 第二阶段 T=0.7、1,700 步、偏好奖励只在最后 400 步 | ✓ | v2 §3.2.2；Nature Methods |
| 6 | 式 (1)–(3) 逐符号转录 | ✓ 与 v1 LaTeX 逐字一致；“三处 LaTeX 相同”表述不准（v2 移动了 `&`，Nature 由出版社重排），已改 §2.1 | v1 §2.2.1；v2 §2.1、A.3 (11)–(13)；Nature 式 (1)–(3) |
| 7 | DeepSeekMath 式 (3) 转录、结果监督 $\hat A_{i,t}=\widetilde r_i$、超参数（1e-6、0.04、64、1024、1024） | ✓；“DeepSeekMath 称式 (2)……”编号有歧义，KL 估计在 DeepSeekMath 是式 (4)，已改 | 2402.03300 v3 §4.1.1–4.1.2、§4.2 |
| 8 | k3 = (r−1) − log r，r=p/q，样本来自 q，估计 KL[q,p]；博客日期 2020/03/07 | ✓ | joschu.net/blog/kl-approx.html |
| 9 | 训练成本 101K/5K/41K/147K GPU·h = $202K/$10K/$82K/$294K；64×8 H800；198 h；“about 4 days, or roughly 80 hours”；“660B” | ✓ 逐字（v2 Table 7、SI Table 4 两处核对） | v2 §B.4.4；SI §2.4.4 |
| 10 | V3 全部训练 2.788M H800 GPU·h、$5.576M，不含前期研究与消融 | ✓ | 2412.19437 §1 |
| 11 | 反思词选词人数：v2 “33”、SI “3”、Nature ED Fig. 1 “a panel of three human experts” | ✓ 三处原文都已核对 | v2 §C.2；SI §3.2；Nature ED Fig. 1 |
| 12 | 80 万 SFT 数据：395,285 / 211,129 / 10,124 / 10,395 / 177,812 = 804,745；平均 5,355.3 tokens；总计 4.31B；token 占比 | ✓ 重算一致 | v2 Table 5 |
| 13 | SFT：V3-Base、2–3 epochs（v1 为 “two epochs”）、5e-5→5e-6、32,768、batch 128 | ✓ | v2 §B.4.2；v1 §2.3.3 |
| 14 | 蒸馏 base 与学习率 6 行、batch 64、衰减到 1/10 | ✓ | v2 §B.4.3 Table 6 |
| 15 | 各阶段表（R1-Zero/Dev1/Dev2/Dev3/R1）与 V3、V3-Base、o1-1217 列 | ✓ 逐格 | v2 Table 3、Table 8、Table 12 |
| 16 | “AlpacaEval 2.0 improving by 25% and ArenaHard by 17%” 是百分点（+25.5 / +16.7；相对 +41.1% / +22.1%） | ✓ | v2 §4 |
| 17 | 蒸馏 vs RL：QwQ 50.0 / Qwen2.5-32B-Zero 47.0 / Distill-32B 72.6；模型卡 QwQ 44.0 | ✓；模型卡的 LCB 基线分歧只存在于其主表，模型卡蒸馏表与论文一致（已改 E18） | v1 Table 5/6；v2 Table 15/16；HF README |
| 18 | Nature 645 (8081), 633–638；收稿 2025-02-14、接收 2025-07-17、在线 2025-09-17、卷期 2025-09-18；CC BY 4.0；ED Fig. 编号；Table 1/2 编号 | ✓ | nature.com 文章页 |
| 19 | arXiv v1 2025-01-22 15:19:35 UTC、v2 2026-01-04 03:57:36 UTC；journal-ref 填 Nature 645, 633–638 | ✓ | arxiv.org/abs/2501.12948 |
| 20 | R1、R1-Zero、R1-0528 的 config 逐字节相同，字段值；7 个小模型的 config；KV cache 算例 N40–N47 | ✓（三个文件 sha256 相同；70,272 B/token；64K 序列 4.29 GiB） | HF config.json |
| 21 | R1-0528：“minor version upgrade”、AIME 2025 70.0→87.5、12K→23K tokens、§2 对比表各行、Qwen3-8B 蒸馏表、使用方式两条变化 | ✓ | HF R1-0528 README；API 新闻 2025/05/28 |
| 22 | 越狱表：R1 25.2→85.9；风控 8.5→4.3、拒答 27.3→87.3；“moderate level (comparable to GPT-4o (2024-05-13))”；Table 9 平均 95.0 (85.9) | ✓ | v2 §5、§D.3 Table 9–11 |
| 23 | 测试时计算：61.8%、8,793 tokens、<7,000 / >18,000、GPT-4o 24.7% / 711；366 题、93 场比赛；USAMO 指数 | ✓ | v2 §E.2–E.4、Table 13 |
| 24 | 奖励式 (4)–(10) 转录；RM：66,000 对、106,000 条、batch 256、lr 6e-6、1 epoch、8,192 | ✓ 逐符号 | v2 §2.2、§3.1、§3.2 |
| 25 | R1 发布 2025-01-20、MIT、deepseek-reasoner、$0.14 / $0.55 / $2.19 | ✓ | api-docs.deepseek.com/news/news250120/ |
| 26 | Dr. GRPO 两句引文 | ✓ | 2503.20783 摘要、Fig. 1 caption |
| 27 | 审稿意见三条引文（成本、安全、拟人化）与 “table D14” 引文 | ✓ | Peer Review File |
| 28 | 数值例 N30–N39、N6、N8–N18、N23–N28、N48–N50、N52–N54 | ✓ 全部重算一致 | Python |
| 29 | “wait” 频次 8,200: 887、8,600: 1,500 | ✓（按 y 轴 0–1,400 标定重解析 SVG） | v2 `wait_count_2.svg` |
| 30 | Qwen2-Math-7B-Zero 22.3% / 18.1%、约 10,000 步 | ✓ 数字；E22 起点表述有误，已改 | v2 §F.1 Table 17 |

### 11.2 改了什么、为什么

1. **D1、E1、图 13.1：“77.9% 是同一曲线的截点 / 与最高点的差是数字化误差”→ 更正。** v2 图 1a 是矢量 SVG，pass@1 点与 k/480 的偏差都小于 1e-4 个单位。最高点 0.7771 = 373/480（9,200 步），末点 0.7729 = 371/480（10,000 步），77.9% ≈ 374/480 不在图上。71.0% 确实是第 8,400 步的点。77.9% 的出处列入 §8 第 21 条。
2. **§2.1 “三处 LaTeX 相同” → 改为“数学内容逐符号一致”。** v2 把对齐符 `&` 移到了 $\mathcal J$ 前；Nature 由出版社重排（`\epsilon`、正体下标、`array`）。
3. **§2.1 k3 段 “DeepSeekMath 称式 (2)” → 注明 DeepSeekMath 中是式 (4)。** DeepSeekMath 自己的式 (2) 是 PPO 的 KL 惩罚，原写法容易误导。
4. **C4 “v2 只把 LeetCode 改成 code competition prompts” → 改为“v2 与 Nature 改写了整段措辞”**，并给出 v2 原句。
5. **E6 “模型卡、Table 8、config 都是 671B” → config 不写参数量**，写 671B/37B 的是模型卡与 v2 Table 8/12。
6. **E9 “V3 判官只用于拒绝采样” → 范围说窄了。** V3 还用于构造有用性偏好对、冷启动改写、风控审查（均见 v2）；§G.1 提到 LLM 对照标准答案判对错，但没说用在哪一阶段（新增 §8 第 22 条）。
7. **E18 → 说明模型卡内部不一致。** 主表 LCB 基线为 33.8 / 34.2，蒸馏表为 38.9 / 32.9（与论文同）。
8. **E22 “从 Instruct 版 7.9% 到 22.3%” → RL 起点是 Qwen2-Math-7B base，Instruct 只是对照。** 同时注明 “applying RL to these distilled models yields significant further gains” 只在 v1 §3.2，v2 删除了。
9. **N51 → 补充复算精度**，并发现 cons@16 点不是 1/30 的整数倍，排除了“单次 16 票”的解释（推测仍标为推测）。
10. **§3.12 → 补充审稿人原本要求把蒸馏成本也列出**，而 SI 表没有列。

### 11.3 复核后仍存疑（已并入 §8）

- 77.9% 的检查点 / 评测协议（§8 第 21 条）。
- LLM 判官是否进入 RL 奖励（§8 第 22 条）。
- 图 9b 的 y 轴按 0–1,400（步长 200）标定。这是由网格线数量和刻度字形推断的，与研究笔记的数值吻合，但没有逐个解码刻度文字。
- Dr. GRPO 的发表会议仍未核实。
