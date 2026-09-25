# 第 03 课研究笔记：从预训练模型到助手——SFT、奖励模型、RLHF（PPO）与 DPO

> 本文件是 `deepseek/src/lesson-03-alignment-rlhf.md` 的事实底稿。所有数字、公式、超参数都在 2026-09-24
> 对照一手来源核对过（arXiv LaTeXML HTML 全文 grep + arXiv PDF 文本抽取 + Crossref/出版社页面）。
> 引文保持英文原文，尽量只摘关键短语；较长内容用中文转述并标出位置。
> 标注约定：**[原文]** = 论文原话或原式；**[推导]** = 笔者按原式逐行推出、已用 Python 数值验证；
> **[推论]** = 笔者依据原文做的工程推断，写进课文时要标“笔者推测/推断”；**[未核实]** 见第 8 节。

---

## 0. 覆盖范围与所读版本

| key | 标题 | 作者 | arXiv（所读版本） | 发表状态 | 日期 |
|---|---|---|---|---|---|
| `ouyang2022` | Training language models to follow instructions with human feedback（InstructGPT） | Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, …, Paul Christiano, Jan Leike, Ryan Lowe（共 20 人） | 2203.02155 **v1**（唯一版本；HTML 全文 + PDF）；另读 NeurIPS 正式版 PDF（15 页正文） | NeurIPS 2022（Advances in NeurIPS 35, Main Conference Track） | v1：2022-03-04 |
| `christiano2017` | Deep reinforcement learning from human preferences | Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei | 1706.03741 **v4**（HTML + PDF） | NIPS 2017（Advances in NeurIPS 30） | v1 2017-06-12；v4 2023-02-17 |
| `ziegler2019` | Fine-Tuning Language Models from Human Preferences | Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, Geoffrey Irving | 1909.08593 **v2**（HTML + PDF） | arXiv 预印本 | v1 2019-09-18；v2 2020-01-08 |
| `stiennon2020` | Learning to summarize from human feedback | Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano | 2009.01325 **v3**（HTML + PDF） | NeurIPS 2020（arXiv comments 字段） | v1 2020-09-02；v3 2022-02-15 |
| `schulman2017ppo` | Proximal Policy Optimization Algorithms | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov | 1707.06347 **v2**（无 HTML 渲染；读 PDF，pypdf 抽取以保留希腊字母；v1 PDF 做了 diff） | arXiv 预印本（未见会议版） | v1 2017-07-20；v2 2017-08-28 |
| `schulman2016gae` | High-Dimensional Continuous Control Using Generalized Advantage Estimation | John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel | 1506.02438 **v6**（HTML + PDF） | ICLR 2016（PDF 页眉 “Published as a conference paper at ICLR 2016”） | v1 2015-06-08；v6 2018-10-20 |
| `bradley1952` | Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons | Ralph Allan Bradley, Milton E. Terry | 非 arXiv | Biometrika 39(3/4):324–345，1952 年 12 月；DOI 10.2307/2334029（JSTOR）与 10.1093/biomet/39.3-4.324（OUP） | 1952-12 |
| `rafailov2023dpo` | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn | 2305.18290 **v3**（HTML + PDF）；v1 PDF 做了对照 | NeurIPS 2023（v3 PDF 页脚 “37th Conference on Neural Information Processing Systems (NeurIPS 2023)”；v1 页脚 “Preprint. Under review.”） | v1 2023-05-29；v2 2023-12-13；v3 2024-07-29 |
| `bai2022hh` | Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback | Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, …（共 31 人） | 2204.05862 **v1**（唯一版本；HTML） | arXiv 预印本；数据 github.com/anthropics/hh-rlhf | 2022-04-12 |
| `shao2024deepseekmath`（桥接第 8 课） | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Zhihong Shao 等 | 2402.03300 **v3**（HTML） | arXiv 预印本 | v1 2024-02-05；v3 2024-04-27 |
| `rajbhandari2019zero`（显存账） | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He | 1910.02054 **v3**（HTML） | SC 2020（未在本轮核对会议信息，见第 8 节） | v3 2020-05-13 |
| `brown2020gpt3`（仅核对模型尺寸） | Language Models are Few-Shot Learners | Tom B. Brown 等 | 2005.14165（HTML，Table 2.1） | NeurIPS 2020 | 2020 |
| `deepseek2024llm`（桥接第 6 课） | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism | DeepSeek-AI | 2401.02954 **v1**（HTML，§4） | arXiv 预印本 | 2024-01-05 |

读取方式说明：arxiv.py `text` 拉取的是 arxiv.org/html 的 LaTeXML 渲染（2203.02155 渲染页头写 “arXiv:2203.02155v1 [cs.CL] 04 Mar 2022”；2305.18290 渲染页头写 “arXiv:2305.18290v3 [cs.LG] 29 Jul 2024”）。PPO 没有可用的 HTML/ar5iv 渲染（脚本报 too short），所以 PPO 全部引文来自 v2 PDF；pdftotext 会丢希腊字母，改用 pypdf 抽取（“□”字符是减号）。

---

## 1. 核心论断与原文引文

### 1.1 InstructGPT（`ouyang2022`，arXiv v1）

**C1. 三步法。** §3.1 三个步骤的小标题原文：
- Step 1: “Collect demonstration data, and train a supervised policy.”
- Step 2: “Collect comparison data, and train a reward model.”
- Step 3: “Optimize a policy against the reward model using PPO.”
（§3.1；Figure 2 是这三步的示意图，图中 Step 3 写的是 “using reinforcement learning”。）

**C2. 第 2、3 步可以迭代。** “Steps 2 and 3 can be iterated continuously”（§3.1）；并说明大部分比较数据来自 SFT 策略，少部分来自 PPO 策略（§3.1 同段，转述）。

**C3. 三个数据集的规模（按 prompt 计）。** §3.2：“The SFT dataset contains about 13k training prompts”，RM 数据集 “33k training prompts”，PPO 数据集 “31k training prompts (only from the API)”。精确数见 Table 6（§A.3），第 3 节表 3.1 已转录。注意：PPO 数据集**没有人工标签**，只是 RL 阶段的输入 prompt（§3.2 原话 “without any human labels”）。

**C4. 标注员。** 约 40 名合同工（§1 “a team of 40 contractors”；§3.4 “about 40 contractors on Upwork and through ScaleAI”）。一致率：“training labelers agree with each-other 72.6±1.5% of the time”，留出标注员之间 77.3±1.3%（§3.4）。

**C5. SFT 细节。** 16 个 epoch、余弦衰减、residual dropout 0.2（§3.5）；“we find that our SFT models overfit on validation loss after 1 epoch”，但多训几个 epoch 对 RM 分和人类偏好都有帮助（§3.5，转述）；最终按 **RM 分**而不是验证损失选模型（§C.1）。

**C6. 只用 6B 奖励模型。** “In this paper we only use 6B RMs, as this saves a lot of compute”（§3.5）；理由之二是 175B RM 训练不稳定，不适合作为价值函数的初始化（§3.5、§C.2 转述）。§C.2：“We trained a single 6B reward model which we used for all PPO models of all sizes.”

**C7. K 个回答排序 → C(K,2) 对比较，一个 prompt 一个 batch 元素。** “we present labelers with anywhere between K=4 and K=9 responses to rank”（§3.5）；“we train on all $\binom K2$ comparisons from each prompt as a single batch element”（§3.5）。脚注 3 解释若把比较打散：每个 completion 会参与 $K-1$ 次梯度更新，一个 epoch 就过拟合（转述）。§C.2：“Ties were dropped.”；一个 batch 64 个 prompt，所以 “up to 64×$\binom K2$ ≤ 2,304 comparisons”。

**C8. RM 损失（式 (1)）**，见 §2.2 转录。

**C9. 奖励归一化。** RM 损失对奖励平移不变（“the RM loss is invariant to shifts in reward”），所以 RL 前加一个偏置，使 **labeler demonstrations** 的平均分为 0（§3.5）。

**C10. RL 环境是“赌博机”，但 KL 惩罚按 token 加。** “The environment is a bandit environment”（§3.5）；“we add a per-token KL penalty from the SFT model at each token”（§3.5）；“The value function is initialized from the RM.”（§3.5）

**C11. PPO 与 PPO-ptx。** 式 (2) 把预训练梯度混进 PPO；“For "PPO" models, γ is set to 0.”；“in this paper InstructGPT refers to the PPO-ptx models”（§3.5）。

**C12. RLHF 超参数（§C.4）。**
- KL 系数：“with β=0.02 (see Equation 2)”。
- “We train all the RL models for 256k episodes.”，约 31k 个唯一 prompt。
- “The batch size for each iteration is 512, with a minibatch size of 64.”；“trained on for only a single inner epoch”。
- 学习率恒定，前 10 次迭代 warmup（从峰值的 1/10 开始）；权重 EMA，“with a decay rate of 0.992”。
- “No discount is applied when estimating the generalized advantage”（即 GAE 的折扣 $\gamma=1$；**λ 没有给出**，见第 8 节）。
- “The PPO clip ratio is set to 0.2, and the sampling temperature is 1 for rollouts.”
- “for all PPO models we use a 6B RM and a 6B value function”，价值函数由 RM 初始化；价值函数学习率 9e-6（1.3B、6B 策略）与 5e-6（175B 策略）。

**C13. 参考模型其实不是正文的“SFT 模型”，而是 RL 初始化模型。** §C.3：RLHF 初始化模型是从预训练 GPT-3 出发 “apply supervised fine-tuning for 2 epochs on the demonstration dataset”，并且 “We also mix in 10% pretraining data during fine-tuning”；§C.4：“These models are also used to compute the KL reward”。也就是说式 (2) 里的 $\pi^{\mathrm{SFT}}$ 在实现中是“2 epoch + 10% 预训练混合”的那个模型，而不是 §3.5/§C.1 描述的 16 epoch SFT 基线（见第 7 节勘误 E-5）。§C.3 说混 10% 预训练数据的依据“见 Appendix E.11”，实际实验在 §E.8，且 §E.8 称 PPO 对这些选择不敏感（勘误 E-11）。

**C14. 预训练混合系数。** §C.4：“We use 8 times more pretraining examples than the number of the RL training episodes.”；预训练梯度乘以 “γ=27.8”。§E.6：γ ≥ 20 即可在 1.3B 上恢复回退，“a single value of 27.8 seems to work well across model sizes”。

**C15. 只加大 KL 系数修不好“对齐税”。** §E.6：在 1.3B、γ=0 下把 KL 系数扫到 2.0（默认值的 100 倍），“the regressions still cannot be fixed”。

**C16. 最优 KL 系数。** §E.7：“The optimal value is around 0.01 and 0.02.”（系数为 0 和 2 都差。）

**C17. 标题级结果（人类偏好）。** 摘要：1.3B InstructGPT 的输出被偏好于 175B GPT-3，“despite having 100x fewer parameters”；§1 同一句写作 “over 100x fewer parameters”。§1/§4.1：“175B InstructGPT outputs are preferred to GPT-3 outputs 85 ± 3% of the time”，对 few-shot GPT-3 为 71 ± 4%。评测集是**留出客户的 API prompt 分布**上的标注员偏好，不是公开 NLP 基准（§3.6）。

**C18. RM 对留出标注员的泛化。** §4.1：5 折交叉验证，留出组准确率 69.6 ± 0.9%，训练组 72.4 ± 0.4%。

**C19. “对齐税”（alignment tax）。** §1：RLHF 在 SQuAD、DROP、HellaSwag、WMT15 Fr→En 上回退，“This is an example of an “alignment tax””；PPO-ptx 大幅减轻回退且不损失标注员偏好（§1 转述）。

**C20. 模型与数值格式。** “We train three model sizes (1.3B, 6B, and 175B parameters)”（§1），均为 GPT-3 架构；§C：“All models use fp16 weights and activations, with fp32 master copies of weights.”；上下文 2k token，prompt 与回答各限 1k；Adam β1=0.9、β2=0.95（§C）。

**C21. 对齐的对象。** 该流程把 GPT-3 对齐到 “the stated preferences of a specific group of people”（主要是标注员与研究者），而非广义“人类价值”（§1，§5.2）。

**C22. RM 的初始化在正文、附录、NeurIPS 版三处说法不同。** arXiv §3.5：“Starting from the SFT model with the final unembedding layer removed”；§C.2：最终 RM “was initialized from a 6B GPT-3 model that was fine-tuned on a variety of public NLP datasets”，并说 “This was mostly for historical reasons”，从 GPT-3 或 SFT 初始化结果相近；NeurIPS 版正文：“We fine-tune GPT-3 to take in a prompt and response, and output a scalar reward.”（见勘误 E-6。）

### 1.2 Christiano et al. 2017（`christiano2017`，v4）

**C23. 偏好预测器 = Bradley–Terry。** §2.2.3 式 (1) 把片段内奖励求和后做 softmax（见 §2.2）；紧接着：“This follows the Bradley-Terry model (Bradley and Terry, 1952)”，并类比国际象棋 Elo 等级分（§2.2.3）。

**C24. 10% 随机作答噪声。** “we assume there is a 10% chance that the human responds uniformly at random”（§2.2.3）。[推导] 于是预测概率变为 $0.05+0.9\,\hat P$，被夹在 $[0.05,0.95]$ 内（数值见 §4）。

**C25. 这篇没有用 PPO。** “we use advantage actor-critic (A2C; Mnih et al., 2016) to play Atari games”，MuJoCo 用 TRPO（§2.2.1）。PPO 论文比它晚一个多月才上 arXiv（2017-07-20 vs 2017-06-12）。

**C26. 反馈量很小。** 摘要：反馈 “on less than one percent of our agent's interactions”；Hopper 后空翻 “This behavior was trained using 900 queries in less than an hour.”（§3.2）

### 1.3 Ziegler et al. 2019（`ziegler2019`，v2）——把“比较数据训练的 RM + PPO + KL 惩罚”用到预训练语言模型（GPT-2）

（归属说明：论文本身**没有**声称“首次”；其 §1 相关工作列出了更早的“人类反馈 + RL 用于语言”工作——Kreutzer et al. 2018（翻译，off-policy 奖励学习）、Jaques et al. 2019（对话，隐式人类偏好）、Yi et al. 2019（对话，从人学奖励但平滑后做监督学习）。课文可写“InstructGPT 所沿用方法的直接来源”，不要写“第一个”。见核查记录。）

**C27. RM 损失是四选一 softmax。** 标注员从 $(y_0,y_1,y_2,y_3)$ 中选最好的 $b$，式 (1)（§2，见 §2.2；印刷版漏了负号，见勘误 E-3）。RM 归一化到 “mean 0 and variance 1”（§2）。

**C28. KL 修正奖励（式 (2)）。** “we add a penalty with expectation β KL(π, ρ)”（§2）。该项三个作用：“it plays the role of an entropy bonus”、防止策略离开 $r$ 有效的区域、在风格任务中还是任务定义的一部分（§2 转述）。出处归属：“Following Jaques et al. [2017], Jaques et al. [2019], we use a KL constraint”（§1）。

**C29. 自适应 β 控制器。** §2.2 的对数空间比例控制器（见 §2.3 转录），“We used $K_\beta=0.1$.”

**C30. 规模与超参数。** “We use a 774M parameter version of the GPT-2 language model”（§2.1），36 层、20 头、宽 1280；风格任务 5,000 次比较，摘要任务 60,000 次比较（摘要）；PPO：2M episodes、$\gamma=1$、每批 4 个 PPO epoch（§2.2）。

**C31. 去掉 KL 会怎样。** Table 10 标题：“Without regularization towards natural language, the results are gibberish”，即使加熵奖励也一样（附录 Table 10 转述）。注意该实验是对 **mock sentiment 分类器**（程序化奖励）做微调，不是对人类偏好 RM（Table 10 标题原文 “a model fine-tuned to mock sentiment without a KL penalty”）；课文引用时要说明，别写成“对人类偏好 RM 去掉 KL 就乱码”。

### 1.4 Stiennon et al. 2020（`stiennon2020`，v3）

**C32. RM 损失（成对 BT）与归一化。** §3.4 “Reward models”：成对 logistic 损失（见 §2.2）；训练结束后归一化使 “the reference summaries from our dataset achieve a mean score of 0”。

**C33. KL 奖励与“熵奖励”解释。** §3.4：完整奖励 $R=r_\theta-\beta\log[\pi^{RL}_\phi/\pi^{SFT}]$；第一个作用 “it acts as an entropy bonus”，第二个作用防止策略产出 RM 没见过的输出（转述）。脚注 7：奖励只在整段摘要结束时给，EOS 结束一个 episode，“the discount factor γ=1”。

**C34. 价值网络独立、四个模型同尺寸。** “we use a Transformer with completely separate parameters from the policy”；价值函数用 RM 参数初始化；“the reward model, policy, and value function are the same size”（§3.4）。

**C35. PPO 超参数（附录 B.1 Hyperparameters）。** “We set γ = 1 and λ = 0.95 for the advantage estimation”，每批 rollout 做 4 个 epoch；“We used a KL coefficient of 0.05 for both of the main runs”；batch 512（1.3B）/256（6.7B）；1M episodes；学习率 1.5e-5（1.3B）/7e-6（6.7B），线性衰减（附录 B.1）。

**C36. 奖励过度优化。** §4.3 / Figure 5：优化越强，真实偏好先升后降，“eventually the reward model becomes anti-correlated with human preferences”。

**C37. 小模型 + 人类反馈 胜过 10 倍大的监督模型。** §4.1：1.3B 人类反馈模型 vs 10 倍大的监督模型，“61% versus 43% raw preference score against reference summaries”。公开数据集 “64,832 summary comparisons”（§1 贡献 4）。

**C38. RM 的规模化。** Figure 6 标题：“Doubling amount of training data leads to a ~1.1% increase in reward model validation accuracy”，“doubling the model size leads to a ~1.8% increase”。

### 1.5 PPO（`schulman2017ppo`，v2 PDF）

**C39. 朴素策略梯度不能多步复用样本。** §2.1：对 $L^{PG}$ 用同一批轨迹做多步优化 “is not well-justified”，并常导致 “destructively large policy updates”。

**C40. TRPO 用硬约束而非惩罚的理由。** §2.2：很难选一个在不同问题、甚至同一问题不同阶段都合适的 β（转述）。

**C41. 裁剪目标（式 (7)）与 ε=0.2。** §3：“where epsilon is a hyperparameter, say, ϵ = 0.2”；min 使得目标是未裁剪目标的 “lower bound (i.e., a pessimistic bound)”；“we only ignore the change in probability ratio when it would make the objective improve”；“$L^{CLIP}(\theta)=L^{CPI}(\theta)$ to first order around θold”。

**C42. 自适应 KL 惩罚变体（式 (8)）表现更差。** §4：“we found that the KL penalty performed worse than the clipped surrogate objective”；规则：$d<d_{targ}/1.5$ 则 $\beta\leftarrow\beta/2$，$d>d_{targ}\times1.5$ 则 $\beta\leftarrow\beta\times2$；1.5 和 2 是启发式选择。**这里的 KL 是对 $\pi_{\theta_{old}}$ 的，不是对参考模型的**（见勘误 M-3）。

**C43. 共享参数时的合并目标（式 (9)）与截断 GAE（式 (10)–(12)）。** §5，见 §2.4 转录（式 (10)(11) 指数有印刷错误，见勘误 E-2）。

**C44. Algorithm 1。** $N$ 个并行 actor 各采 $T$ 步 → 在 $NT$ 个样本上用 minibatch 大小 $M\le NT$ 做 $K$ 个 epoch → $\theta_{old}\leftarrow\theta$（§5）。

**C45. 消融结果（Table 1，7 个 MuJoCo 环境 × 3 种子 = 21 次）。** 无裁剪无惩罚 −0.39；裁剪 ε=0.1/0.2/0.3：0.76/**0.82**/0.70；自适应 KL $d_{targ}$=0.003/0.01/0.03：0.68/0.74/0.71；固定 KL β=0.3/1/3/10：0.62/0.71/0.72/0.69（§6.1 Table 1）。

**C46. v1→v2 的差别。** diff 两版 PDF 文本：正文唯一实质改动是引言结尾一句——v1 说 Atari 上 PPO 比 A2C **和 TRPO** 好、比 DQN 差但更省墙钟时间；v2 删去 TRPO 与 DQN 的比较，只保留 “better … than A2C and similarly to ACER”。公式、表格未变。

### 1.6 GAE（`schulman2016gae`，v6）

**C47. TD 残差是优势的无偏估计（当 $V=V^{\pi,\gamma}$）。** §3 式 (10)。否则 “it will yield biased policy gradient estimates”。

**C48. GAE 是 k 步估计的指数加权平均。** §3 “The generalized advantage estimator GAE(γ,λ) is defined as the exponentially-weighted average of these k-step estimators”，化简得式 (16)。

**C49. 偏差-方差。** GAE(γ,1) “has high variance”，GAE(γ,0) 在 $V$ 不准时有偏但方差低（§3）；“λ < 1 introduces bias only when the value function is inaccurate”；“the best value of λ is much lower than the best value of γ”（§3）。实验最佳区间：$\gamma\in[0.96,0.99]$、$\lambda\in[0.92,0.99]$（§6.3.1 Cart-pole，21 个随机种子平均），讨论部分称 λ 取 $[0.9,0.99]$ 通常最好（§7）。

### 1.7 DPO（`rafailov2023dpo`，v3）

**C50. RLHF 的三阶段与 KL 目标。** §3：SFT → 偏好采样与奖励学习 → RL 优化；式 (1) BT，式 (2) RM 负对数似然，式 (3) KL 约束奖励最大化；标准做法是构造奖励 $r_\phi-\beta(\log\pi_\theta-\log\pi_{ref})$ 再用 PPO 最大化（§3 末）。

**C51. 最优策略闭式解（式 (4)）。** 引用前人（[31, 30, 19, 15]），完整推导在附录 A.1（式 (11)–(15)）。$Z(x)$ 难估（§4）。

**C52. 配分函数在 BT 模型中抵消。** §4：BT 只依赖奖励之差，把式 (5) 代入式 (1) 后 “the partition function cancels”，得到式 (6)，进而得到 DPO 损失式 (7)。

**C53. 梯度的加权解释。** §4：样本按隐式奖励排序“错得多离谱”加权；naive 版本 “without the weighting coefficient can cause the language model to degenerate”（指 Appendix Table 3）。

**C54. 参考模型的选择。** §4 “DPO outline”：有 SFT 模型就令 $\pi_{ref}=\pi^{SFT}$；否则用在 $y_w$ 上做最大似然得到的模型（Preferred-FT）来缓解分布偏移。

**C55. 理论：等价类与重参数化。** Definition 1（奖励差只依赖 $x$ 即等价）、Lemma 1（同类奖励在 PL/BT 下诱导相同偏好分布）、Lemma 2（同类奖励诱导相同最优策略）、Theorem 1（“Under mild assumptions”，所有与 PL/BT 相容的奖励类都可写成 $\beta\log\frac{\pi}{\pi_{ref}}$；附录 A.6 给出的假设是 $\pi_{ref}(y|x)>0$ 对所有 $(x,y)$ 且 $\beta>0$）；Proposition 1（每个等价类里这种形式的奖励唯一）。

**C56. 对 PPO 不稳定性的解释。** §5.2：式 (10) 中的归一化项可解释为 “the soft value function of the reference policy”；不影响最优解，但缺了它策略梯度方差可能很大；前人用人类完成作基线相当于对该项的单样本蒙特卡洛估计（转述）。

**C57. DPO 超参数（附录 B，v3）。** “we use a β = 0.1, batch size of 64 and the RMSprop optimizer”，学习率 1e-6，150 步线性 warmup；“For TL;DR summarization, we use β = 0.5”。**v1 写的是 “Adam optimizer”**（版本漂移，见勘误 V-2）。

**C58. 实验结论（方向可信、幅度谨慎）。** 情感任务：DPO 的奖励–KL 前沿 “strictly dominates PPO”（§6.1）；TL;DR：“DPO has a win rate of approximately 61% at a temperature of 0.0”，“exceeding the performance of PPO at 57%”（GPT-4 评判，§6.2）；人评中 DPO（温度 0.25）对 PPO（温度 0）胜 58%（§6.2/§6.4）；最大模型 “up to 6B parameters”（§7 Discussion 的 “Limitations & Future Work” 段，不在 §1）。评估器是 GPT-4，Table 2 做了人–GPT-4 一致率校验。

**C59. DPO 对 RLHF 成本的定性描述。** §1：RLHF “involving training multiple LMs and sampling from the LM policy in the loop of training”。

### 1.8 HH-RLHF（`bai2022hh`，v1，选读）

**C60. KL 系数极小。** 式 (4.1) $r_{total}=r_{PM}-\lambda_{KL}D_{KL}$；“In practice we use a very small value of λ_KL = 0.001”，而且 “might actually be wholly unnecessary”（§4.1）。

**C61. √KL 与奖励近似线性。** §4.3 标题 “An Approximately Linear Relation Between √D_KL and Reward”；解释：KL 在 $\delta\pi$ 上展开从二阶开始，而奖励对 $\delta\pi$ 线性，于是奖励 ∝ √KL（§4.3 转述，作者自己标注为经验现象 + 启发式解释）。

**C62. PPO 超参数。** 附录 B.1：“PPO clipping ϵ = 0.2, discount factor γ = 1, and no entropy bonus”；同一批样本重复 $K$ 次，RLHF 扫描 $K=1$、鲁棒性研究 $K=2$、在线 RLHF $K=4$。

**C63. 数据规模。** §2.3：基础数据集 “44k helpfulness comparisons and 42k red-teaming (harmlessness) comparisons”；拒绝采样数据 52k/2k；在线迭代数据 22k 有用性、无红队。模型 13M 到 52B 共 7 个，约 4 倍几何级数（§3.1）。

**C64. Elo 与 PM 分换算。** 式 (2.1)：胜率 $=1/(1+10^{\Delta(\text{Elo})/400})$，“Δ(Elo Score) ≈ 174 ∗ Δ(PM Score)”（§2.4）。[推导] $400/\ln 10=173.7$，这就是 174 的来源（见 §4）。

### 1.9 桥接材料

**C65.（DeepSeekMath，第 8 课的伏笔）** §4.1.1：PPO 的价值函数 “typically another model of comparable size as the policy model”，带来显存与计算负担；LLM 中通常只有最后一个 token 拿到 RM 分，使得训练逐 token 准确的价值函数更难（转述）；式 (2) 是逐 token KL 奖励。

**C66.（ZeRO，显存账）** §3.1：混合精度 Adam 需要 fp16 参数 $2\Psi$ + fp16 梯度 $2\Psi$ + 优化器状态 $K\Psi$，“Mixed-precision Adam has K=12”，合计 “16Ψ bytes of memory requirement”。

**C67.（GPT-3 尺寸）** Table 2.1 中与 InstructGPT “6B” 对应的行是 “GPT-3 6.7B”（6.7B 参数），1.3B 对应 “GPT-3 XL”。InstructGPT 全文没有出现 “6.7B”（grep 0 次），把 6B 等同于 GPT-3 6.7B 是[推论]。

**C68.（DeepSeek LLM，第 6 课伏笔）** §4：SFT 后用 DPO，偏好数据按有用性与无害性构造；“We trained an epoch for DPO, with a learning rate of 5e-6”，batch 512，warmup + 余弦。**没有报告 β**（第 8 节）。

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 记号对照总表（写课文时统一用右列）

| 含义 | InstructGPT | Stiennon | Ziegler | DPO | PPO/GAE 原文 | DeepSeekMath | **课程记号（建议）** |
|---|---|---|---|---|---|---|---|
| prompt | $x$ | $x$（帖子） | $x$ | $x$ | 状态 $s_t$ | $q$ | $q$（style-guide 已定义） |
| 回答 | $y$ | $y$ | $y$ | $y$ | 动作 $a_t$ | $o$ | $o$；第 $t$ 个 token $o_t$ |
| 偏好/非偏好回答 | $y_w,y_l$ | $y_i, y_{1-i}$ | $y_b$ | $y_w,y_l$ | — | — | $o_w,o_l$（**style-guide 未定义，需补入**） |
| 被训练的策略 | $\pi_\phi^{\mathrm{RL}}$（参数 $\phi$） | $\pi_\phi^{\mathrm{RL}}$ | $\pi$ | $\pi_\theta$ | $\pi_\theta$ | $\pi_\theta$ | $\pi_\theta$ |
| 参考/初始策略 | $\pi^{\mathrm{SFT}}$ | $\pi^{\mathrm{SFT}}$ | $\rho$ | $\pi_{ref}$ | — | $\pi_{ref}$ | $\pi_{ref}$ |
| 采样用旧策略 | — | — | — | — | $\pi_{\theta_{old}}$ | $\pi_{\theta_{old}}$ | $\pi_{\theta_{old}}$ |
| 奖励模型 | $r_\theta$（参数 $\theta$！） | $r_\theta$ | $r$ | $r_\phi$ | — | $r_\varphi$ | $r_\phi$ |
| 价值函数 | 未命名 | 未命名 | — | — | $V(s)$ / $V_\theta$ | $V_\psi$ | $V_\psi$ |
| KL 系数（对参考模型） | $\beta$ | $\beta$ | $\beta$ | $\beta$ | —（PPO 的 $\beta$ 是对旧策略的，见下） | $\beta$ | $\beta$ |
| PPO 自适应 KL 系数（对旧策略） | — | — | — | — | $\beta$ | — | $\beta_{old}$（建议新符号，避免与 $\beta$ 混淆） |
| 预训练混合系数 | $\gamma$ | — | — | — | — | — | $\gamma_{ptx}$（**与折扣 $\gamma$ 冲突，必须改名**） |
| 折扣 | （GAE 不折扣） | $\gamma=1$ | $\gamma=1$ | — | $\gamma$ | — | $\gamma$ |
| GAE 参数 | — | $\lambda=0.95$ | — | — | $\lambda$ | — | $\lambda$（style-guide 未定义，需补） |
| 概率比 | — | — | — | — | $r_t(\theta)$（与奖励 $r_t$ 同字母！） | 分式直接写出 | $\rho_t(\theta)$（建议新符号） |
| 裁剪系数 | — | — | — | — | $\epsilon$ | $\varepsilon$ | $\varepsilon$ |
| 优势 | — | — | — | — | $\hat A_t$ | $A_t$ | $\hat A_t$ |
| 排序回答数 | $K$ | — | 4 | $K$（PL） | $K$ = PPO epoch 数 | — | 本课内用 $K$ 并就地说明；PPO epoch 数改写 $n_{ep}$ |
| logistic 函数 | $\sigma$ | $\sigma$ | — | $\sigma$ | — | — | $\sigma$（注意 Christiano 的 $\sigma$ 是轨迹片段） |

注意三处“同字母异义”：(i) InstructGPT 的 $\theta$ 是 **RM 参数**、$\phi$ 是**策略参数**，DPO 恰好反过来；(ii) InstructGPT 的 $\gamma$ 是预训练系数，PPO/GAE 的 $\gamma$ 是折扣；(iii) PPO 论文自己在 §3 用 $r_t(\theta)$ 表示概率比，在 §5 式 (10)(12) 又用 $r_t$ 表示奖励。Christiano 的 $\sigma^1,\sigma^2$ 是**轨迹片段**，$o_t$ 是**观测**，与课程的 $\sigma$（sigmoid）、$o_t$（回答 token）都撞车。

### 2.2 偏好模型与奖励模型损失

**Bradley–Terry 的现代写法（DPO 式 (1)，§3）[原文]：**
$$p^*(y_1\succ y_2\mid x)=\frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right)+\exp\left(r^*(x,y_2)\right)} \tag{DPO-1}$$

[推导] 分子分母同除 $\exp(r^*(x,y_1))$：$\frac{1}{1+\exp(-(r_1-r_2))}=\sigma(r_1-r_2)$。DPO §4 原文也写了 $p^*(y_1\succ y_2\mid x)=\sigma(r^*(x,y_1)-r^*(x,y_2))$。
Bradley & Terry 1952 原论文的参数化（$\pi_i/(\pi_i+\pi_j)$ 形式）**未能读到原文**，见第 8 节；令 $\pi_i=e^{r_i}$ 即与上式等价。

**Christiano 式 (1)（§2.2.3）[原文]：**
$$\hat P\left[\sigma^1\succ\sigma^2\right]=\frac{\exp\sum\hat r\left(o^1_t,a^1_t\right)}{\exp\sum\hat r\left(o^1_t,a^1_t\right)+\exp\sum\hat r\left(o^2_t,a^2_t\right)} \tag{C-1}$$
$$\mathrm{loss}(\hat r)=-\sum_{(\sigma^1,\sigma^2,\mu)\in\mathcal D}\mu(1)\log\hat P\left[\sigma^1\succ\sigma^2\right]+\mu(2)\log\hat P\left[\sigma^2\succ\sigma^1\right] \tag{C-无号}$$
（$\mu$ 是标签分布：偏好哪边就把概率质量放哪边，平局时可各 0.5——“平局可各 0.5”是笔者对 $\mu$ 的解读，原文此处未展开。）脚注 3：式 (1) 不折扣。

**Ziegler 式 (1)（§2，四选一）[原文，按印刷照录，缺负号]：**
$$\mathrm{loss}(r)=\mathbb{E}_{(x,\{y_i\}_i,b)\sim S}\left[\log\frac{e^{r(x,y_b)}}{\sum_i e^{r(x,y_i)}}\right] \tag{Z-1}$$
正确的“损失”应在前面加负号（这是对数似然，要最大化），见勘误 E-3。四选一时这是 Plackett–Luce 的“只取第一名”一项；$K=2$ 时退化为 BT。

**Stiennon（§3.4，无编号）[原文]：**
$$\text{loss}(r_\theta)=-E_{(x,y_0,y_1,i)\sim D}\left[\log\left(\sigma\left(r_\theta(x,y_i)-r_\theta(x,y_{1-i})\right)\right)\right]$$

**InstructGPT 式 (1)（§3.5）[原文]：**
$$\operatorname{loss}(\theta)=-\frac{1}{\binom{K}{2}}E_{(x,y_w,y_l)\sim D}\left[\log\left(\sigma\left(r_\theta(x,y_w)-r_\theta(x,y_l)\right)\right)\right] \tag{I-1}$$
读法：一个 prompt 的全部 $\binom K2$ 对作为一个 batch 元素，除以 $\binom K2$ 是对这一组求平均，使 $K=9$ 的 prompt 不比 $K=4$ 的 prompt 权重大 6 倍（$36/6$）——后半句是[推论]，原文只给了式子和“single batch element”。

**DPO 式 (2)（§3）[原文]：**
$$\mathcal{L}_R(r_\phi,\mathcal D)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal D}\bigl[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\bigr] \tag{DPO-2}$$

**HH-RLHF 式 (4.2)（§4.1）[原文]：** $P(A>B)=\dfrac{1}{1+e^{r_{PM}(B)-r_{PM}(A)}}$。
**HH-RLHF 式 (2.1)（§2.4）[原文]：** $\mathrm{Win\ Fraction}=\dfrac{1}{1+10^{\frac{\Delta(\mathrm{Elo\ Score})}{400}}}$，$\Delta(\mathrm{Elo\ Score})\approx174\ast\Delta(\mathrm{PM\ Score})$。
（符号约定：这里 $\Delta$ 显然是“对手减自己”，否则胜率随 Elo 差增大而下降——原文没说明方向，写课文时要明说。）

**Plackett–Luce（DPO 附录 A.3 式 (18)）[原文]：**
$$p^*(\tau\mid y_1,\ldots,y_K,x)=\prod_{k=1}^{K}\frac{\exp(r^*(x,y_{\tau(k)}))}{\sum_{j=k}^{K}\exp(r^*(x,y_{\tau(j)}))} \tag{DPO-18}$$
原文：“Notice that when K=2, Equation 18 reduces to the Bradley-Terry model.”（InstructGPT 没用 PL，而是把排序拆成 $\binom K2$ 对 BT。）

**课程记号版（建议课文采用）：**
$$P(o_w\succ o_l\mid q)=\sigma\big(r_\phi(q,o_w)-r_\phi(q,o_l)\big),\qquad
\mathcal L_{RM}(\phi)=-\E_{(q,o_w,o_l)}\big[\log\sigma\big(r_\phi(q,o_w)-r_\phi(q,o_l)\big)\big].$$
[推导] 设 $\Delta=r_w-r_l$，$\frac{\partial}{\partial\Delta}[-\log\sigma(\Delta)]=-(1-\sigma(\Delta))=-\sigma(-\Delta)$：排错得越离谱（$\Delta\ll0$），梯度越接近 1；已排对且差距大时梯度趋于 0。这是 DPO 梯度权重的前身。

### 2.3 带 KL 惩罚的 RL 目标

**Ziegler 式 (2)（§2）[原文]：**
$$R(x,y)=r(x,y)-\beta\log\frac{\pi(y|x)}{\rho(y|x)} \tag{Z-2}$$
**Ziegler 自适应控制器（§2.2，无编号）[原文]：**
$$e_t=\operatorname{clip}\left(\frac{\operatorname{KL}(\pi_t,\rho)-\operatorname{KL}_{\mathrm{target}}}{\operatorname{KL}_{\mathrm{target}}},-0.2,0.2\right),\qquad \beta_{t+1}=\beta_t(1+K_\beta e_t),\quad K_\beta=0.1$$

**Stiennon（§3.4，无编号）[原文]：** $R(x,y)=r_\theta(x,y)-\beta\log[\pi^{\text{RL}}_\phi(y|x)/\pi^{\text{SFT}}(y|x)]$

**InstructGPT 式 (2)（§3.5，arXiv v1）[原文]：**
$$\begin{split}\operatorname{objective}(\phi)=&E_{(x,y)\sim D_{\pi_{\phi}^{\mathrm{RL}}}}\left[r_{\theta}(x,y)-\beta\log\left(\pi_{\phi}^{\mathrm{RL}}(y\mid x)/\pi^{\mathrm{SFT}}(y\mid x)\right)\right]+\\&\gamma E_{x\sim D_{\textrm{pretrain}}}\left[\log(\pi_{\phi}^{\mathrm{RL}}(x))\right]\end{split} \tag{I-2}$$
（NeurIPS 正式版正文**删去了**这个式子，只说 PPO-ptx “mixing the pretraining gradients into the PPO gradients”，并指向附录 D.4；式号 (2) 只在 arXiv 版有效。）

**DPO 式 (3)（§3）[原文]：**
$$\max_{\pi_{\theta}}\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_{\theta}(y\mid x)}\bigl[r_{\phi}(x,y)\bigr]-\beta\mathbb{D}_{\textrm{KL}}\bigl[\pi_{\theta}(y\mid x)\mid\mid\pi_{\text{ref}}(y\mid x)\bigr] \tag{DPO-3}$$

**HH-RLHF 式 (4.1)（§4.1）[原文]：** $r_{\rm total}=r_{\rm PM}-\lambda_{\rm KL}D_{\rm KL}(\text{policy}\,\|\,\text{policy}_0)$，$\lambda_{KL}=0.001$。

**DeepSeekMath 式 (2)（§4.1.1）[原文]：**
$$r_{t}=r_{\varphi}(q,o_{\leq t})-\beta\log\frac{\pi_{\theta}(o_{t}|q,o_{<t})}{\pi_{ref}(o_{t}|q,o_{<t})} \tag{DSM-2}$$
（字面上 $r_\varphi(q,o_{\le t})$ 像是每个 token 都有 RM 分；同一节下文说 LLM 中 “usually only the last token is assigned a reward score”。课文里应写成：RM 分只在最后一个 token 加，KL 项每个 token 都加。）

**课程记号版 [推导]：**
$$\mathcal J(\theta)=\E_{q\sim\mathcal D,\;o\sim\pi_\theta(\cdot\mid q)}\Big[r_\phi(q,o)-\beta\log\frac{\pi_\theta(o\mid q)}{\pi_{ref}(o\mid q)}\Big]\;+\;\gamma_{ptx}\,\E_{x\sim\mathcal D_{pretrain}}\big[\log\pi_\theta(x)\big].$$

**[推导] 逐 token KL 与序列级 KL 是同一件事（折扣 $\gamma=1$ 时）。** 自回归分解（Ziegler §2 也写了 $\rho(x_0\cdots x_{n-1})=\prod_k\rho(x_k|x_0\cdots x_{k-1})$）：
$$\log\frac{\pi_\theta(o\mid q)}{\pi_{ref}(o\mid q)}=\sum_{t=1}^{|o|}\log\frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{ref}(o_t\mid q,o_{<t})}.$$
令逐 token 奖励 $r_t=-\beta\log\frac{\pi_\theta(o_t|\cdot)}{\pi_{ref}(o_t|\cdot)}+\mathbb 1[t=|o|]\,r_\phi(q,o)$，不折扣的回报 $\sum_t r_t$ 恰好等于式 (I-2) 第一项方括号里的量。这解释了 InstructGPT 为什么能同时说“bandit environment”（整段回答一个奖励）和“per-token KL penalty”，也解释了它为什么 “No discount is applied”。又因为 $o\sim\pi_\theta$，$\E_{o\sim\pi_\theta}[\log\frac{\pi_\theta}{\pi_{ref}}]=\mathrm{KL}(\pi_\theta\|\pi_{ref})$，单样本 $\log$ 比是 KL 的无偏估计（第 8 课称 k1 估计器，数学课 M05）。

**[推导] KL 项 = 熵奖励 + 参考似然奖励：**
$$-\beta\,\E_{o\sim\pi_\theta}\Big[\log\frac{\pi_\theta(o|q)}{\pi_{ref}(o|q)}\Big]=\beta\,H(\pi_\theta(\cdot|q))+\beta\,\E_{o\sim\pi_\theta}[\log\pi_{ref}(o|q)].$$
这就是 Ziegler/Stiennon 说 “entropy bonus” 的精确含义：第一项鼓励分散，第二项把输出拉回参考模型认为“像语言”的区域。（数值验证见 §4.6。）

### 2.4 PPO（`schulman2017ppo` v2，式号为原文）

$$\hat g=\hat{\mathbb E}_t\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat A_t\right] \tag{1}$$
$$L^{PG}(\theta)=\hat{\mathbb E}_t\left[\log\pi_\theta(a_t\mid s_t)\hat A_t\right] \tag{2}$$
$$\underset{\theta}{\text{maximize}}\;\hat{\mathbb E}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}\hat A_t\right] \tag{3}$$
$$\text{subject to }\hat{\mathbb E}_t\left[\mathrm{KL}[\pi_{\theta_{old}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)]\right]\le\delta \tag{4}$$
$$\underset{\theta}{\text{maximize}}\;\hat{\mathbb E}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}\hat A_t-\beta\,\mathrm{KL}[\pi_{\theta_{old}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)]\right] \tag{5}$$
$$L^{CPI}(\theta)=\hat{\mathbb E}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}\hat A_t\right]=\hat{\mathbb E}_t\left[r_t(\theta)\hat A_t\right] \tag{6}$$
$$L^{CLIP}(\theta)=\hat{\mathbb E}_t\left[\min\left(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right] \tag{7}$$
$$L^{KLPEN}(\theta)=\hat{\mathbb E}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}\hat A_t-\beta\,\mathrm{KL}[\pi_{\theta_{old}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)]\right] \tag{8}$$
$$L^{CLIP+VF+S}_t(\theta)=\hat{\mathbb E}_t\left[L^{CLIP}_t(\theta)-c_1L^{VF}_t(\theta)+c_2S[\pi_\theta](s_t)\right],\quad L^{VF}_t=(V_\theta(s_t)-V^{targ}_t)^2 \tag{9}$$
$$\hat A_t=-V(s_t)+r_t+\gamma r_{t+1}+\cdots+\gamma^{T-t+1}r_{T-1}+\gamma^{T-t}V(s_T)\quad\text{（印刷如此，指数应为 }T-t-1\text{）} \tag{10}$$
$$\hat A_t=\delta_t+(\gamma\lambda)\delta_{t+1}+\cdots+\cdots+(\gamma\lambda)^{T-t+1}\delta_{T-1}\quad\text{（同上，应为 }T-t-1\text{）} \tag{11}$$
$$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t) \tag{12}$$
其中 $r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，$r(\theta_{old})=1$（§3）。PPO 论文里 $\hat{\mathbb E}_t$ 是有限 batch 上的经验平均（§2.1）。

**课程记号版（LLM token 级 MDP：$s_t=(q,o_{<t})$，$a_t=o_t$）：**
$$\rho_t(\theta)=\frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{\theta_{old}}(o_t\mid q,o_{<t})},\qquad
\mathcal J^{CLIP}(\theta)=\hat\E_t\Big[\min\big(\rho_t\hat A_t,\ \clip(\rho_t,1-\varepsilon,1+\varepsilon)\hat A_t\big)\Big].$$

**[推导] 单项的分段形式与梯度（$\varepsilon>0$）：**
- $\hat A_t>0$：$\min(\rho\hat A,\clip(\rho)\hat A)=\hat A\cdot\min(\rho,1+\varepsilon)$。$\rho<1+\varepsilon$ 时梯度 $\hat A\,\nabla\rho$；$\rho>1+\varepsilon$ 时为常数，梯度 0。$\rho<1-\varepsilon$ 时**不裁剪**（min 选中未裁剪项）。
- $\hat A_t<0$：$=\hat A\cdot\max(\rho,1-\varepsilon)$。$\rho>1-\varepsilon$ 时梯度 $\hat A\nabla\rho$；$\rho<1-\varepsilon$ 时梯度 0。$\rho>1+\varepsilon$ 时**不裁剪**。
- 结论：只在“再往这个方向走会让目标变好”的一侧截断（原文 “we only ignore the change in probability ratio when it would make the objective improve”）。

### 2.5 GAE（`schulman2016gae` v6，式号为原文）

$$\delta^V_t=r_t+\gamma V(s_{t+1})-V(s_t)$$
$$\mathbb E_{s_{t+1}}\left[\delta^{V^{\pi,\gamma}}_t\right]=\mathbb E_{s_{t+1}}\left[Q^{\pi,\gamma}(s_t,a_t)-V^{\pi,\gamma}(s_t)\right]=A^{\pi,\gamma}(s_t,a_t) \tag{10}$$
$$\hat A^{(k)}_t:=\sum_{l=0}^{k-1}\gamma^l\delta^V_{t+l}=-V(s_t)+r_t+\gamma r_{t+1}+\dots+\gamma^{k-1}r_{t+k-1}+\gamma^kV(s_{t+k}) \tag{14}$$
$$\hat A^{(\infty)}_t=\sum_{l=0}^\infty\gamma^l\delta^V_{t+l}=-V(s_t)+\sum_{l=0}^\infty\gamma^lr_{t+l} \tag{15}$$
$$\hat A^{\mathrm{GAE}(\gamma,\lambda)}_t:=(1-\lambda)\left(\hat A^{(1)}_t+\lambda\hat A^{(2)}_t+\lambda^2\hat A^{(3)}_t+\dots\right)=\sum_{l=0}^\infty(\gamma\lambda)^l\delta^V_{t+l} \tag{16}$$
$$\mathrm{GAE}(\gamma,0):\ \hat A_t:=\delta_t=r_t+\gamma V(s_{t+1})-V(s_t) \tag{17}$$
$$\mathrm{GAE}(\gamma,1):\ \hat A_t:=\sum_{l=0}^\infty\gamma^l\delta_{t+l}=\sum_{l=0}^\infty\gamma^lr_{t+l}-V(s_t) \tag{18}$$
（式 (11)–(13) 是 $k=1,2,3$ 的特例。）课程记号只需把 $V$ 写成 $V_\psi$。
[推导] 有限长回答（长度 $|o|$，终止后 $V=0$）的递推实现：$\hat A_{|o|}=\delta_{|o|}$，$\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$；价值回归目标常取 $\hat A_t+V_\psi(s_t)$（这是常见实现，GAE 论文 §5 讨论了 TD(λ) 目标，未逐字核对到“目标 = 优势 + V”的写法，写课文时标“常见做法”）。

### 2.6 DPO 推导链（`rafailov2023dpo` v3，式号为原文）

**最优策略（§4 式 (4)(5)）[原文]：**
$$\pi_r(y\mid x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right),\qquad Z(x)=\sum_y\pi_{\text{ref}}(y\mid x)\exp\left(\frac1\beta r(x,y)\right) \tag{4}$$
$$r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x) \tag{5}$$

**附录 A.1 逐行（式 (11)–(15)）[原文，逐行照录，理由为笔者加注]：**
$$\max_\pi\mathbb E_{x\sim\mathcal D,y\sim\pi}[r(x,y)]-\beta\mathbb D_{KL}[\pi(y|x)\|\pi_{ref}(y|x)] \tag{11}$$
$$\begin{aligned}
&=\max_\pi\mathbb E_{x\sim\mathcal D}\mathbb E_{y\sim\pi(y|x)}\left[r(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{ref}(y|x)}\right] &&\text{（KL 定义展开）}\\
&=\min_\pi\mathbb E_{x\sim\mathcal D}\mathbb E_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{ref}(y|x)}-\frac1\beta r(x,y)\right] &&\text{（乘 }-1/\beta\text{，max 变 min）}\\
&=\min_\pi\mathbb E_{x\sim\mathcal D}\mathbb E_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac1\beta r(x,y)\right)}-\log Z(x)\right] &&\text{（加减 }\log Z\text{）}
\end{aligned} \tag{12}$$
$$\pi^*(y|x)=\frac1{Z(x)}\pi_{ref}(y|x)\exp\left(\frac1\beta r(x,y)\right)\quad\text{（非负且和为 1，是合法分布）}$$
$$\min_\pi\mathbb E_{x\sim\mathcal D}\left[\mathbb E_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi^*(y|x)}\right]-\log Z(x)\right]= \tag{13}$$
$$\min_\pi\mathbb E_{x\sim\mathcal D}\left[\mathbb D_{KL}(\pi(y|x)\|\pi^*(y|x))-\log Z(x)\right] \tag{14}$$
$$\pi(y|x)=\pi^*(y|x)=\frac1{Z(x)}\pi_{ref}(y|x)\exp\left(\frac1\beta r(x,y)\right) \tag{15}$$
关键理由：$Z(x)$ 不依赖 $\pi$；Gibbs 不等式（KL ≥ 0，当且仅当两分布相同取 0）。前提：“a general non-parametric policy class”（A.1 原文），即对每个 $x$ 可任意选分布；参数化策略未必能表示 $\pi^*$。
[推导] 由 (14)，最优值（按原 max 目标计）$=\beta\,\E_x[\log Z(x)]$；§4.6 用玩具例子数值验证。

**配分函数抵消（§4 式 (6)，附录 A.2 式 (16)(17)）[原文]：**
$$p^*(y_1\succ y_2\mid x)=\frac{1}{1+\exp\left(\beta\log\frac{\pi^*(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}-\beta\log\frac{\pi^*(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\right)} \tag{6}$$
附录 A.2 末行：$=\sigma\left(\beta\log\frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)}-\beta\log\frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}\right)$。
[推导要点] 代入式 (5) 后分子分母都含 $\exp(\beta\log Z(x))$ 因子，同一个 $x$ 上两个回答共享 $Z(x)$，于是约掉——这正是 BT“只看差值”的性质在起作用。

**DPO 损失（§4 式 (7)）[原文]：**
$$\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}})=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right] \tag{7}$$

**梯度（§4，无编号）[原文]：**
$$\nabla_\theta\mathcal L_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}})=-\beta\,\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\Big[\underbrace{\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))}_{\text{higher weight when reward estimate is wrong}}\big[\underbrace{\nabla_\theta\log\pi(y_w\mid x)}_{\text{increase likelihood of }y_w}-\underbrace{\nabla_\theta\log\pi(y_l\mid x)}_{\text{decrease likelihood of }y_l}\big]\Big],$$
$$\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\quad\text{（隐式奖励）}$$
（原文梯度里写的是 $\pi$ 而非 $\pi_\theta$，是笔误级省略。）

**附录 A.4 的推导有两处印刷错误（v1、v3 相同），课文必须用下面的正确版本：**
原文式 (21)：$\nabla_\theta\mathcal L_{DPO}=-\nabla_\theta\mathbb E\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}-\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}\right)\right]$——$\sigma$ 的自变量与式 (7) 相反；
原文 A.4 末式的权重写作 $\beta\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)=\beta\sigma(\hat r_w-\hat r_l)$——与 §4 的 $\sigma(\hat r_l-\hat r_w)$ 相反。
**[推导] 正确版本：** 令 $u=\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)$，则式 (7) 的单样本损失为 $-\log\sigma(-u)$。
$$\nabla_\theta[-\log\sigma(-u)]=\frac{\sigma'(-u)}{\sigma(-u)}\nabla_\theta u=(1-\sigma(-u))\nabla_\theta u=\sigma(u)\nabla_\theta u,$$
$$\nabla_\theta u=\beta\big(\nabla_\theta\log\pi_\theta(y_l|x)-\nabla_\theta\log\pi_\theta(y_w|x)\big)\ \Rightarrow\ \nabla_\theta\mathcal L=-\beta\,\E\big[\sigma(\hat r_l-\hat r_w)\,(\nabla\log\pi_\theta(y_w|x)-\nabla\log\pi_\theta(y_l|x))\big].$$
（用到 $\sigma'(z)=\sigma(z)(1-\sigma(z))$、$1-\sigma(-u)=\sigma(u)$，参考项 $\pi_{ref}$ 不含 $\theta$。）§4.8 用有限差分验证：§4 的形式与数值梯度一致，A.4 末式不一致。

**Theorem 1 的投影算子（§5.1 式 (8)(9)）[原文]：**
$$f(r;\pi_{\text{ref}},\beta)(x,y)=r(x,y)-\beta\log\sum_y\pi_{\text{ref}}(y\mid x)\exp\left(\frac1\beta r(x,y)\right) \tag{8}$$
$$\sum_y\underbrace{\pi_{\text{ref}}(y\mid x)\exp\left(\frac1\beta r(x,y)\right)}_{=\pi(y\mid x)}=1 \tag{9}$$
**§5.2 式 (10)[原文]：**
$$\max_{\pi_\theta}\mathbb E_{\pi_\theta(y\mid x)}\bigg[\underbrace{r_\phi(x,y)-\beta\log\sum_y\pi_{\text{ref}}(y\mid x)\exp\left(\frac1\beta r_\phi(x,y)\right)}_{f(r_\phi,\pi_{\text{ref}},\beta)}-\underbrace{\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}}_{\text{KL}}\bigg] \tag{10}$$

**DPO 的 PL 版本（附录 A.3 式 (20)）[原文]：**
$$\mathcal L_{DPO}(\pi_\theta,\pi_{ref})=-\mathbb E_{\tau,y_1,\ldots,y_K,x\sim\mathcal D}\left[\log\prod_{k=1}^K\frac{\exp\left(\beta\log\frac{\pi_\theta(y_{\tau(k)}|x)}{\pi_{ref}(y_{\tau(k)}|x)}\right)}{\sum_{j=k}^K\exp\left(\beta\log\frac{\pi_\theta(y_{\tau(j)}|x)}{\pi_{ref}(y_{\tau(j)}|x)}\right)}\right] \tag{20}$$
（这是把 InstructGPT 的 K 路排序直接喂给 DPO 的方式；课文可作为练习提及。）

**课程记号版：**
$$\hat r_\theta(q,o)=\beta\log\frac{\pi_\theta(o\mid q)}{\pi_{ref}(o\mid q)},\qquad
\mathcal L_{DPO}(\theta)=-\E_{(q,o_w,o_l)}\Big[\log\sigma\big(\hat r_\theta(q,o_w)-\hat r_\theta(q,o_l)\big)\Big].$$

**附录 B 的参考实现（PyTorch）[原文要点，不整段照抄]：** `losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))`，其中 `pi_logratios = pi_yw_logps - pi_yl_logps`、`ref_logratios = ref_yw_logps - ref_yl_logps`；`rewards = beta * (pi_logps - ref_logps).detach()`。[推导] $\beta[(\log\pi_w-\log\pi_l)-(\log\pi^{ref}_w-\log\pi^{ref}_l)]=\hat r_w-\hat r_l$，与式 (7) 一致（按“策略比值差减参考比值差”重排）。

---

## 3. 规格与超参数表

### 3.1 InstructGPT 数据集（Table 6，§A.3，单位：prompt 数）

| 数据集 | split | 来源 | 数量 | 小计 |
|---|---|---|---|---|
| SFT | train | labeler | 11,295 | train 合计 12,725（≈“13k”） |
| SFT | train | customer | 1,430 | |
| SFT | valid | labeler | 1,550 | valid 合计 1,653 |
| SFT | valid | customer | 103 | |
| RM | train | labeler | 6,623 | train 合计 33,207（≈“33k”） |
| RM | train | customer | 26,584 | |
| RM | valid | labeler | 3,488 | valid 合计 17,887 |
| RM | valid | customer | 14,399 | |
| PPO | train | customer | 31,144 | （≈“31k”，无人工标签） |
| PPO | valid | customer | 16,185 | |

来源：arXiv v1 Table 6（NeurIPS 版同一张表编号为 Table 3——版本漂移）。小计为笔者加总（§4.2）。§A.3 说 RM 训练的比较对数 “an order of magnitude larger than the number of prompts”。

### 3.2 InstructGPT 训练超参数

| 阶段 | 项 | 值 | 来源 |
|---|---|---|---|
| 全部 | 架构 / 精度 | GPT-3 架构；fp16 权重与激活 + fp32 主权重 | §C |
| 全部 | 上下文 | 2k token；prompt > 1k 过滤；回答 ≤ 1k | §C |
| 全部 | 优化器 | Adam，β1=0.9，β2=0.95 | §C |
| 策略尺寸 | — | 1.3B、6B、175B | §1 |
| SFT | epoch / dropout | 16 / residual dropout 0.2 | §3.5、§C.1 |
| SFT | LR / batch | 1.3B、6B：9.65e-6、32；175B：5.03e-6、8 | §C.1 |
| SFT | 调度 | 余弦降到 10%，无 warmup；按 RM 分选模型 | §C.1 |
| RM | 尺寸 | 6B（所有策略共用一个） | §3.5、§C.2 |
| RM | 每 prompt 回答数 | K = 4…9，平局丢弃 | §3.5、§C.2 |
| RM | epoch / LR / batch | 1 epoch；9e-6，余弦降到 10%；64 个 prompt（≤ 2,304 对） | §C.2 |
| RL 初始化 = 参考模型 | 训练 | 预训练 GPT-3 + 演示数据 2 epoch + 10% 预训练混合 | §C.3、§C.4 |
| RL 初始化 | LR / batch | 1.3B：5e-6；6B：1.04e-5；175B：2.45e-6；batch 32/32/8 | §C.3 |
| PPO | KL 系数 β | 0.02（E.7：最优约 0.01–0.02） | §C.4、§E.7 |
| PPO | episodes | 256k（约 31k 唯一 prompt） | §C.4 |
| PPO | batch / minibatch / inner epoch | 512 / 64 / 1 | §C.4 |
| PPO | 裁剪 ε | 0.2 | §C.4 |
| PPO | 折扣 | 不折扣（γ=1）；GAE λ **未报告** | §C.4 |
| PPO | 采样温度 | 1 | §C.4 |
| PPO | EMA | 权重 EMA，衰减 0.992 | §C.4 |
| PPO | warmup | 前 10 次迭代，从峰值 1/10 起；此后恒定 | §C.4 |
| PPO | 策略 LR 扫描 | 1.3B/6B：2.55e-6 到 2.55e-5 对数扫描；无 ptx 时 >8.05e-6 全部发散；175B：2.55e-6、3.74e-6 | §E.9 |
| 价值函数 | 尺寸 / 初始化 / LR | 6B，由 RM 初始化；9e-6（1.3B、6B 策略），5e-6（175B 策略） | §C.4 |
| PPO-ptx | 预训练系数 γ | 27.8（≥20 即可在 1.3B 恢复） | §C.4、§E.6 |
| PPO-ptx | 预训练样本量 | RL episodes 的 8 倍（比例 4 时预训练损失会上升；32 更好但慢数倍；8 使训练时间翻倍） | §C.4、§E.11 |
| 消融 | batch / minibatch 扫描 | batch 64–1024 中 512 最好；minibatch 32 略优于 64，但因 GPU 利用率用 64 | §E.11 |

### 3.3 其他 RLHF 系统的超参数对照

| 项 | Ziegler 2019 | Stiennon 2020 | InstructGPT 2022 | HH-RLHF 2022 |
|---|---|---|---|---|
| 策略尺寸 | 774M（GPT-2） | 1.3B、6.7B | 1.3B、6B、175B | 13M…52B（7 个） |
| RM 尺寸 | 与策略同源（从 $\rho$ 初始化） | 与策略同尺寸 | 固定 6B | 多为 52B PM（另有与策略同尺寸的对照组；Figure 13 标题：对“matched to policy sizes”的小 PM 训练最终表现差） |
| RM 损失 | 四选一 softmax | 成对 BT | K 路排序拆 $\binom K2$ 对 BT | 成对（PM 预训练 PMP + 微调） |
| KL 系数 | 常数或自适应（$K_\beta=0.1$） | 0.05 | 0.02 | $\lambda_{KL}=0.001$ |
| 折扣 / λ | γ=1 / 默认 | γ=1 / 0.95 | 不折扣 / 未报告 | γ=1 / 未见 |
| PPO epoch | 4（每批 1 个 minibatch） | 4 | 1（8 个 minibatch） | K=1/2/4 |
| 裁剪 ε | PPO2 默认（未写数值） | 未报告（全文无 clip） | 0.2 | 0.2 |
| 价值网络 | 未单独说明（未核实） | 独立参数、同尺寸、由 RM 初始化 | 6B、由 RM 初始化 | 未核实 |
| episodes | 2M | 1M | 256k | — |
| 来源 | §2.1–2.2、§4.2 | §3.4、附录 B.1 | §3.5、§C.4 | §3.1、§4.1、Figure 13、附录 B.1 |

### 3.4 PPO 原论文超参数

| 项 | MuJoCo（Table 3） | Atari（Table 5） |
|---|---|---|
| Horizon T | 2048 | 128 |
| Adam 步长 | 3×10⁻⁴ | 2.5×10⁻⁴×α（α 从 1 线性退火到 0） |
| epoch 数 | 10 | 3 |
| minibatch | 64 | 32×8 |
| 折扣 γ | 0.99 | 0.99 |
| GAE λ | 0.95 | 0.95 |
| actor 数 | （1，未列） | 8 |
| 裁剪 ε | 在 Table 1 中扫描，0.2 最好 | 0.1×α |
| VF 系数 c1 / 熵系数 c2 | 不共享参数，c1 无关；不用熵奖励（§6.1） | 1 / 0.01 |

（Roboschool 用 Table 4：T=512、15 epoch、minibatch 4096、actor 32/128；此处略。）

### 3.5 DPO 超参数（附录 B / C，v3）

| 项 | 值 | 来源 |
|---|---|---|
| β（默认） | 0.1 | 附录 B |
| β（TL;DR） | 0.5 | 附录 B |
| batch | 64 | 附录 B |
| 优化器 / LR | RMSprop，1e-6（v1 写 Adam），150 步线性 warmup | 附录 B |
| 情感任务扫描 | DPO β ∈ {0.05, 0.1, 1, 5}；PPO target KL ∈ {3, 6, 9, 12}；另有 unlikelihood α ∈ {0.05, 0.1, 0.5, 1} 与 Preferred-FT 的随机种子；“This sweep includes 22 runs in total”（22 是全部算法合计，不只 DPO + PPO） | §6.1 |
| 模型 | 情感：GPT-2-large；TL;DR：GPT-J SFT（CarperAI）；HH：Pythia-2.8B + Preferred-FT | §6、附录 C.1 |
| 数据 | HH：170k 对话；IMDb：25000 前缀 × 4 个样本 → 每前缀 6 对 | §6、附录 C.1 |

### 3.6 “RLHF 要在内存里放几个模型”——四模型表

| 系统 | 策略 $\pi_\theta$（训练） | 价值 $V_\psi$（训练） | 参考 $\pi_{ref}$（冻结，前向） | 奖励 $r_\phi$（冻结，前向） | 合计 | 来源 |
|---|---|---|---|---|---|---|
| InstructGPT PPO | 1.3B / 6B / 175B | 6B（由 RM 初始化） | = RL 初始化模型，与策略同尺寸 | 6B | 4 个网络（其中 2 个固定 6B） | §3.5、§C.3、§C.4 |
| Stiennon PPO | 1.3B 或 6.7B | 同尺寸，独立参数 | 同尺寸（SFT） | 同尺寸 | 4 个同尺寸网络 | §3.4 |
| DPO | 训练 | 无 | 冻结（其对数概率可预先算好，[推论]） | 无（隐式） | 2 个 | §4、附录 B |
| GRPO（第 8 课） | 训练 | 无（组内基线） | 冻结 | 冻结 | 3 个 | DeepSeekMath §4.1.1、Figure 4 |
| SFT | 训练 | — | — | — | 1 个 | — |

补充说明：
1. **$\pi_{\theta_{old}}$ 不是第五个模型。** PPO 只需要采样时那一刻的 $\log\pi_{\theta_{old}}(o_t|\cdot)$，rollout 时顺手存下即可（Algorithm 1 的 “$\theta_{old}\leftarrow\theta$” 只是快照语义）——[推论]，依据是式 (6)(7) 只在已采样的 $(s_t,a_t)$ 上用到 $\pi_{\theta_{old}}$。
2. **InstructGPT 还有一份 EMA 权重**（§C.4 “Exponential moving averages of the weights are applied”），是策略权重的又一份拷贝；其精度未报告（第 8 节）。
3. **单 inner epoch 仍需要裁剪**：InstructGPT 每批 512 分成 8 个 minibatch 顺序更新，第 1 个 minibatch 上 $\rho_t\equiv1$，之后的 minibatch 面对的是已经被前面 minibatch 改过的 $\theta$——[推论]。

---

## 4. 复算的数字

所有数字用 Python 3.11 复算（脚本在笔者 scratch 目录，未入库）。GB 指 $10^9$ 字节。

### 4.1 Bradley–Terry 概率与 RM 损失（$\sigma(\Delta)=1/(1+e^{-\Delta})$）

| $\Delta=r_w-r_l$ | $\sigma(\Delta)$ | $-\log\sigma(\Delta)$ | $\partial\mathcal L/\partial\Delta=-\sigma(-\Delta)$ |
|---|---|---|---|
| −2 | 0.1192 | 2.1269 | −0.8808 |
| −1 | 0.2689 | 1.3133 | −0.7311 |
| 0 | 0.5000 | 0.6931（= ln 2） | −0.5000 |
| 0.5 | 0.6225 | 0.4741 | −0.3775 |
| 1 | 0.7311 | 0.3133 | −0.2689 |
| 1.5 | 0.8176 | 0.2014 | −0.1824 |
| 2 | 0.8808 | 0.1269 | −0.1192 |
| 3 | 0.9526 | 0.0486 | −0.0474 |
| 4.6 | 0.9900 | 0.0100 | −0.0100 |

- 表达式：`1/(1+math.exp(-d))`，`-math.log(sig(d))`。
- 反过来：要 90% 置信度需要奖励差 $\ln 9=2.1972$；99% 需要 $\ln 99=4.5951$。
- Christiano 10% 噪声：$0.05+0.9\sigma(\Delta)$ → $\Delta$=0,1,2,5,10 时为 0.5000、0.7080、0.8427、0.9440、0.9500；对应“排错”一侧的损失 $-\log(0.05+0.9\sigma(-\Delta))$ 在 $\Delta=10$ 时封顶约 2.9949（无噪声时会无界增长）。

### 4.2 InstructGPT 计数

| 量 | 表达式 | 值 |
|---|---|---|
| $\binom K2$，K=4…9 | `math.comb(K,2)` | 6, 10, 15, 21, 28, 36 |
| 每个 completion 出现的对数 | $K-1$ | 3…8 |
| 一个 batch 最多比较对数 | `64*math.comb(9,2)` | 2,304（与 §C.2 印刷值一致） |
| 一个 batch 最少比较对数 | `64*math.comb(4,2)` | 384 |
| SFT train prompt | 11295+1430 | 12,725 |
| RM train prompt | 6623+26584 | 33,207 |
| RM valid prompt | 3488+14399 | 17,887 |
| RM 比较对数的上下界（若全部 K=4 / 全部 K=9） | 33207×6，33207×36 | 199,242 … 1,195,452（真实值未报告，只知道“数量级大于 prompt 数”） |
| 参数量比 | 175/1.3 | 134.6（摘要写 “100x”，§1 写 “over 100x”） |
| 175B 策略 / 6B RM | 175/6 | 29.2 |
| PPO 迭代次数 | 256000/512 | 500（若 256k 指 262,144 则为 512；原文未说明） |
| 每迭代 minibatch 数 | 512/64 | 8 |
| PPO-ptx 预训练样本数 | 8×256000 | 2,048,000 |
| E.6 KL 扫描上限相对默认 | 2.0/0.02 | 100 倍（与原文 “100 times” 一致） |

### 4.3 PPO 裁剪（ε = 0.2，单项，$L=\min(\rho A,\ \clip(\rho,0.8,1.2)A)$）

| $\rho$ | $A$ | $\rho A$ | $\clip(\rho)A$ | $L^{CLIP}$ | $\partial L/\partial\rho$ | 解释 |
|---|---|---|---|---|---|---|
| 1.0 | +1 | 1.0 | 1.0 | 1.0 | 1 | 起点 |
| 1.1 | +1 | 1.1 | 1.1 | 1.1 | 1 | 区间内 |
| 1.5 | +1 | 1.5 | 1.2 | 1.2 | 0 | 已涨够，停止推动 |
| 0.7 | +1 | 0.7 | 0.8 | 0.7 | 1 | 好动作概率反而降了：照常纠正 |
| 0.7 | −1 | −0.7 | −0.8 | −0.8 | 0 | 坏动作已降够，停止推动 |
| 1.5 | −1 | −1.5 | −1.2 | −1.5 | −1 | 坏动作概率反而涨了：照常纠正（悲观） |
| 1.1 | −1 | −1.1 | −1.1 | −1.1 | −1 | 区间内 |
| 0.9 | −2 | −1.8 | −1.8 | −1.8 | −2 | 区间内 |
| 1.25 | +2 | 2.5 | 2.4 | 2.4 | 0 | 已涨够 |
| $e^{0.3}=1.3499$ | +1 | 1.3499 | 1.2 | 1.2 | 0 | 对数概率差 0.3 |
| $e^{-0.3}=0.7408$ | −1 | −0.7408 | −0.8 | −0.8 | 0 | 对数概率差 −0.3 |

- 裁剪边界换成对数概率差：$\ln1.2=0.1823$，$\ln0.8=-0.2231$（不对称）。
- PPO §4 自适应 KL（$d_{targ}=0.01$）：阈值 $0.01/1.5=0.006667$ 与 $0.01\times1.5=0.015$。
- Ziegler 控制器：KL=12、目标 6 → $e=\clip(1.0,-0.2,0.2)=0.2$ → β×1.02；KL=5.4、目标 6 → $e=-0.1$ → β×0.99。

### 4.4 GAE

**有效视野 $1/(1-\gamma\lambda)$ 与权重 $(\gamma\lambda)^l$：**

| (γ, λ) | γλ | $1/(1-\gamma\lambda)$ | $l=0..4$ 权重 | 出处 |
|---|---|---|---|---|
| (0.99, 0.95) | 0.9405 | 16.81 | 1, 0.9405, 0.8845, 0.8319, 0.7824 | PPO Table 3 |
| (1.0, 0.95) | 0.95 | 20.0 | 1, 0.95, 0.9025, 0.8574, 0.8145 | Stiennon 附录 B.1 |
| (0.99, 0) | 0 | 1 | 1, 0, 0, 0, 0 | GAE(γ,0)=δ |
| (0.99, 1) | 0.99 | 100 | 1, 0.99, 0.9801, 0.9703, 0.9606 | GAE(γ,1) |
| (1, 1) | 1 | ∞ | 全为 1 | 蒙特卡洛回报减基线 |

**4-token 回答的逐 token 例子（本笔记自拟数据）：** β=0.05，RM 分 1.0 只加在最后一个 token；采样到的逐 token 对数比 $\log\frac{\pi_\theta}{\pi_{ref}}=[0.10,-0.05,0.20,0.00]$；价值 $V=[0.50,0.60,0.70,0.90]$，终止后 $V=0$；γ=1、λ=0.95。
- 奖励 $r_t=[-0.005,\ 0.0025,\ -0.01,\ 1.0]$，总和 0.9875 $=1.0-0.05\times0.25$（验证 §2.3 的逐 token/序列等价）。
- $\delta_t=r_t+V_{t+1}-V_t=[0.095,\ 0.1025,\ 0.19,\ 0.1]$。
- $\hat A^{GAE}_t=[0.4495875,\ 0.37325,\ 0.2850,\ 0.1000]$，四位小数（四舍五入）$[0.4496,\ 0.3733,\ 0.2850,\ 0.1000]$（递推 $\hat A_t=\delta_t+0.95\hat A_{t+1}$；二进制浮点的 `round` 会把 0.37325 舍成 0.3732，见文末“数学审读追加”）。
- λ=1：回报 $G_t=[0.9875,0.9925,0.99,1.0]$，$\hat A=G-V=[0.4875,0.3925,0.29,0.1]$；λ=0：$\hat A=\delta$。
- 价值回归目标 $\hat A+V=[0.9496,0.9733,0.985,1.0]$（精确值 0.9495875、0.97325）。

### 4.5 KL 正则最优策略的玩具例子（式 (4)）

$\pi_{ref}=[0.6,0.3,0.1]$，$r=[0,1,2]$，$\E_{\pi_{ref}}[r]=0.5$。表达式：`w=[p*exp(r/b)]; Z=sum(w); pi=w/Z`。

| β | $Z$ | $\pi^*$ | $\E_{\pi^*}[r]$ | $\mathrm{KL}(\pi^*\|\pi_{ref})$ | 目标值 $\E r-\beta\mathrm{KL}$ | $\beta\log Z$ |
|---|---|---|---|---|---|---|
| 10 | 1.0537 | [0.5694, 0.3147, 0.1159] | 0.5465 | 0.0023 | 0.5230 | 0.5230 |
| 2 | 1.3664 | [0.4391, 0.3620, 0.1989] | 0.7598 | 0.0677 | 0.6244 | 0.6244 |
| 1 | 2.1544 | [0.2785, 0.3785, 0.3430] | 1.0645 | 0.2970 | 0.7675 | 0.7675 |
| 0.5 | 8.2765 | [0.0725, 0.2678, 0.6597] | 1.5872 | 1.0609 | 1.0567 | 1.0567 |
| 0.25 | 315.0752 | [0.0019, 0.0520, 0.9461] | 1.9442 | 2.0240 | 1.4382 | 1.4382 |

- 每一行都用式 (5) 反算 $\beta\log\frac{\pi^*}{\pi_{ref}}+\beta\log Z$，得回 $[0,1,2]$（数值误差 < 1e-12）。
- “目标值 = $\beta\log Z$” 与 §2.6 的推导一致。
- β→∞ 时 $\pi^*\to\pi_{ref}$；β→0 时 $\pi^*$ 趋向把全部质量放在最大奖励上。

### 4.6 熵分解数值检验

β=1 行：$-\mathrm{KL}(\pi^*\|\pi_{ref})=-0.29697$，$H(\pi^*)+\E_{\pi^*}[\log\pi_{ref}]=-0.29697$（相等；全精度 −0.2969679…）。

### 4.7 DPO 隐式奖励例子（本笔记自拟对数概率）

$\log\pi_\theta(o_w)=-12.0$，$\log\pi_{ref}(o_w)=-13.0$，$\log\pi_\theta(o_l)=-15.5$，$\log\pi_{ref}(o_l)=-15.0$。

| β | $\hat r_w$ | $\hat r_l$ | 间隔 $m=\hat r_w-\hat r_l$ | 损失 $-\log\sigma(m)$ | 梯度权重 $\sigma(-m)$ | 隐含 $P(o_w\succ o_l)$ |
|---|---|---|---|---|---|---|
| 0.1（默认） | 0.1000 | −0.0500 | 0.1500 | 0.6210 | 0.4626 | 0.5374 |
| 0.5（TL;DR） | 0.5000 | −0.2500 | 0.7500 | 0.3869 | 0.3208 | 0.6792 |

- 初始化 $\pi_\theta=\pi_{ref}$：$m=0$，损失 $\ln2=0.6931$，权重 0.5。
- “两边都降”的例子：$\log\pi_\theta(o_w)=-14.0$（参考 −13.0）、$\log\pi_\theta(o_l)=-18.0$（参考 −15.0），β=0.1 → $\hat r_w=-0.1$、$\hat r_l=-0.3$、$m=0.2$、损失 0.5981：**DPO 损失下降并不要求 $o_w$ 的概率上升**（见勘误 M-9）。
- 权重对比：间隔 −1 的样本权重 $\sigma(1)=0.7311$，间隔 +1 的只有 $\sigma(-1)=0.2689$。

### 4.8 DPO 梯度的有限差分验证

三选项 softmax 策略（logits $\theta\sim\mathcal N(0,1)$，种子 0），参考 logits 另取随机数，β=0.1，$o_w$=选项 0、$o_l$=选项 2（可复现写法：`rng=numpy.random.default_rng(0)`，先 `rng.standard_normal(3)` 取策略 logits，再取一次作参考 logits；用 `np.random.seed(0)` 或 `random.gauss` 会得到别的数）：
- 数值梯度（中心差分 1e-6）：[−0.050645, 0, 0.050645]
- §4 公式（权重 $\sigma(\hat r_l-\hat r_w)=0.50645$）：[−0.050645, 0, 0.050645] ✔
- 附录 A.4 末式（权重 $\sigma(\hat r_w-\hat r_l)=0.49355$）：[−0.049355, 0, 0.049355] ✘
- 同一点上式 (7) 损失 0.70613，按原文式 (21) 写法的损失 0.68033（二者是不同函数）。
- 式 (21) 损失的数值梯度为 [+0.049355, 0, −0.049355]，恰好是 A.4 末式 [−0.049355, 0, +0.049355] 的相反数（核查时补算）。

### 4.9 其他换算

| 量 | 表达式 | 值 |
|---|---|---|
| HH Elo 系数 | 400/ln 10 | 173.72（原文取 174） |
| PM 差 1 的胜率 | σ(1) vs 1/(1+10^(−174/400)) | 0.73106 vs 0.73138 |
| HH 的 KL 惩罚上界 | 0.001×100 | < 0.1（原文：$D_{KL}<100$ 为典型） |
| PPO 与 Christiano 上 arXiv 的间隔 | 2017-06-12 → 2017-07-20 | 38 天 |

### 4.10 显存账（只算模型状态；不含激活、rollout 的 KV cache、临时缓冲）

假设（写课文时必须写明）：可训练网络按 ZeRO §3.1 的混合精度 Adam 计 16 字节/参数；冻结网络只存 16 位权重 2 字节/参数；ZeRO 分片只改变每卡份额，不改变总量。InstructGPT 的 §C 精度描述（fp16 + fp32 主权重）与这一假设相符。

| 场景 | 表达式（参数单位：十亿） | 模型状态总量 |
|---|---|---|
| SFT，7B | 16×7 | 112 GB |
| DPO，7B（策略 + 参考） | 16×7 + 2×7 | 126 GB |
| DPO，7B，参考对数概率预先算好 | 16×7 | 112 GB |
| GRPO 式，7B（策略 + 参考 + RM） | 16×7 + 2×7 + 2×7 | 140 GB |
| PPO，7B，四个同尺寸（策略、价值训练；参考、RM 冻结） | 16×(7+7) + 2×(7+7) | 252 GB（= SFT 的 2.25 倍） |
| Stiennon 式，6.7B 四个同尺寸 | 16×(6.7+6.7) + 2×(6.7+6.7) | 241.2 GB |
| InstructGPT，1.3B 策略 | 16×(1.3+6) + 2×(1.3+6) | 131.4 GB |
| InstructGPT，175B 策略 | 16×(175+6) + 2×(175+6) | 3,258 GB（策略 2,800 + 参考 350 + 价值 96 + RM 12） |
| 上一行中 6B 价值网络的占比 | 96/3258 | 2.9% |

读法：InstructGPT 把 RM 与价值网络固定为 6B，所以对 175B 策略而言“第四个模型”的代价很小；而“价值网络与策略同尺寸”的配置（Stiennon；DeepSeekMath 所说的 “comparable size”）才让价值网络成为主要负担之一——这是第 8 课 GRPO 去掉价值网络的动机。以上是**模型状态的记账估算**，不是任何论文报告的实测显存。

---

## 5. 图表清单与重绘建议

### 5.1 论文原图清单（供重绘参考，不得直接复制图片）

| 来源 | 图号 | 标题/内容 | 本课用途 |
|---|---|---|---|
| InstructGPT | Figure 1 | 各模型在 API prompt 分布上相对 175B SFT 的胜率，横轴模型尺寸，曲线 GPT、GPT(prompted)、SFT、PPO、PPO-ptx；误差线 95% CI | 标题结果；原文未给数值表，只能定性重绘 |
| InstructGPT | Figure 2 | 三步法示意（Step 1 演示→SFT；Step 2 排序 D > C > A = B→RM；Step 3 PPO 循环，奖励记为 $r_k$） | 重绘为图 3.1 |
| InstructGPT | Figure 3 | 留出/训练标注员、GPT-3/InstructGPT prompt 分布四宫格胜率 | 可选 |
| InstructGPT | Figure 4 | 元数据（遵循约束、幻觉等）| 可选 |
| InstructGPT | Figure 33/34/36 | 预训练系数扫描、KL 系数扫描、Likert 对 KL 系数 | 讲“β 与 γ_ptx 的分工”时引用 |
| Christiano | Figure 1 | 奖励预测器异步从片段比较中学习的示意 | 历史小节 |
| Ziegler | Figure 1 | 奖励模型训练与策略训练两个过程（在线时交错） | 历史小节 |
| Stiennon | Figure 2 | 人类反馈、RM 训练、策略训练流程图 | 与 InstructGPT 对照 |
| Stiennon | Figure 5 | 偏好分数 vs 奖励模型优化程度（过度优化） | 重绘为示意图 3.14 |
| Stiennon | Figure 6 | RM 准确率 vs 数据量与模型尺寸 | 可选 |
| PPO | Figure 1 | $L^{CLIP}$ 单项对 $r$ 的函数图，左 $A>0$、右 $A<0$，红点 $r=1$ | 重绘为图 3.6 |
| PPO | Figure 2 | 沿更新方向插值时 $L^{CPI}$、$L^{CLIP}$、KL 的变化（Hopper-v1，KL≈0.02 处 $L^{CLIP}$ 最大） | 可选 |
| GAE | Figure 2 | 不同 γ、λ 的学习曲线 | 不重绘，只引结论 |
| DPO | Figure 1 | RLHF（偏好数据→最大似然 RM→采样/打分的 RL 循环）vs DPO（偏好数据→最大似然→最终 LM） | 重绘为图 3.12 |
| DPO | Figure 2 | 左：情感任务奖励–KL 前沿；右：TL;DR 胜率 vs 采样温度 | 用 §4.5 玩具数据画“前沿”替代 |
| HH-RLHF | Figure 4 / 13 | PM 分 vs √KL 近似线性 | 可选（仅定性） |
| DeepSeekMath | Figure 4 | PPO vs GRPO 结构对比 | 第 8 课重绘；本课末尾预告 |

### 5.2 本课建议绘制的图（13 张，含精确数据）

**图 3.1 RLHF 三阶段流程图（据 InstructGPT Figure 2 重绘）。** 三列：
- Step 1 SFT：12,725 个训练 prompt（11,295 标注员写 + 1,430 API），标注员写示范，微调 GPT-3（16 epoch）。
- Step 2 RM：33,207 个训练 prompt（6,623 + 26,584），每个 prompt K=4…9 个回答排序 → $\binom K2$ 对；RM 6B，1 epoch。
- Step 3 PPO：31,144 个 API prompt，策略生成 → RM 打分 → 减去 $\beta\log(\pi_\theta/\pi_{ref})$（β=0.02）→ PPO 更新；PPO-ptx 另混预训练梯度（γ_ptx=27.8）。
- 蓝色箭头表示“这份数据用于训练某个模型”。第 2、3 步之间画一条虚线回环表示可迭代。

**图 3.2 Bradley–Terry sigmoid 与 RM 损失。** 横轴 $\Delta=r_w-r_l\in[-6,6]$；实线 $\sigma(\Delta)$，虚线 $-\log\sigma(\Delta)$，点线 Christiano 噪声版 $0.05+0.9\sigma(\Delta)$。标注点：(−1, 0.2689)、(0, 0.5)、(1, 0.7311)、(2, 0.8808)、(ln 9 = 2.197, 0.9)。颜色：正奖励侧 `--fig-green`，负侧 `--fig-red`。

**图 3.3 K 个回答 → $\binom K2$ 对。** 左：K=4 的完全图（4 个节点 A–D，6 条边，按 D > C > A > B 给边加箭头；注意原图有 A = B 平局，而 §C.2 说平局丢弃——重绘时用严格排序或把 A–B 边画成虚线“丢弃”）。右：柱状图 K=4…9 → 6, 10, 15, 21, 28, 36。

**图 3.4 KL 正则最优策略的“倾斜”。** 三个回答的分组柱状图：$\pi_{ref}=[0.6,0.3,0.1]$，以及 β=2、1、0.5、0.25 下的 $\pi^*$（§4.5 表）。上方标注奖励 $r=[0,1,2]$。说明：β 越小，分布越往高奖励回答倾斜。

**图 3.5 奖励–KL 前沿（玩具版，对应 DPO Figure 2 左的思想）。** 横轴 KL，纵轴 $\E[r]$，点：(0, 0.5) β=∞；(0.0023, 0.5465) β=10；(0.0677, 0.7598) β=2；(0.2970, 1.0645) β=1；(1.0609, 1.5872) β=0.5；(2.0240, 1.9442) β=0.25。连线表示“最优策略族”，任何策略都不可能在它左上方。

**图 3.6 PPO 裁剪函数（据 PPO Figure 1 重绘）。** 两个子图，ε=0.2：
- 左 $A=+1$：$L=r$（$0\le r\le1.2$），$L=1.2$（$r>1.2$）；横轴 $r\in[0,2]$。
- 右 $A=-1$：$L=-0.8$（$0\le r<0.8$），$L=-r$（$r\ge0.8$）。
- 两图在 $r=1$ 画红点；梯度为 0 的平台段加浅灰阴影；在 $r=1.5$（左）与 $r=0.7$（右）处标“停止推动”，在 $r=0.7$（左）与 $r=1.5$（右）处标“照常纠正”。

**图 3.7 裁剪的四象限表。** 2×3 网格：行 $A>0$ / $A<0$，列 $\rho<1-\varepsilon$、$1-\varepsilon\le\rho\le1+\varepsilon$、$\rho>1+\varepsilon$；格内写梯度是否为 0（数据见 §2.4 推导 + §4.3 表）。

**图 3.8 PPO-RLHF 的四个模型交互图。** 节点：策略 $\pi_\theta$（训练，`--fig-blue`）、价值 $V_\psi$（训练，`--fig-violet`）、参考 $\pi_{ref}$（冻结，`--fig-muted`）、奖励 $r_\phi$（冻结，`--fig-green`）。流程：prompt → 策略采样回答 → (a) RM 给末 token 打分；(b) 参考模型给每个 token 的 $\log\pi_{ref}$，与存下的 $\log\pi_{\theta_{old}}$ 合成逐 token KL 奖励；(c) 价值网络给每个 token 的 $V$ → GAE → $\hat A_t$ → 裁剪目标更新策略；价值网络回归 $\hat A+V$。旁注 InstructGPT 尺寸：策略 1.3B/6B/175B，参考同策略，RM 6B，价值 6B（由 RM 初始化）。

**图 3.9 显存柱状图（计算得出，7B）。** 堆叠柱：SFT 112；DPO 126（112 训练 + 14 参考）；GRPO 式 140（112 + 14 + 14）；PPO 252（策略 112 + 价值 112 + 参考 14 + RM 14）。副图：InstructGPT 175B 堆叠 2,800 / 350 / 96 / 12（总 3,258 GB）。图注写明 16/2 字节假设与“不含激活/KV cache”。

**图 3.10 GAE 权重衰减曲线。** 横轴 $l=0..40$，纵轴 $(\gamma\lambda)^l$；四条曲线 γλ = 0.95（Stiennon）、0.9405（PPO MuJoCo）、0.99、1.0；在 $l=1/(1-\gamma\lambda)$（20、16.8、100）处画竖虚线。另画 λ=0 只在 $l=0$ 有一根柱。

**图 3.11 一段 4-token 回答上的奖励与优势（§4.4 数据）。** 四个 token 方格；每格下方写 $r_t$（−0.005, 0.0025, −0.01, 1.0）、$V_t$（0.5, 0.6, 0.7, 0.9）、$\delta_t$（0.095, 0.1025, 0.19, 0.1）、$\hat A_t$（0.4496, 0.3733, 0.2850, 0.1）。最后一格用 `--fig-green` 标 “RM 分只在这里”，每格小红字标 KL 惩罚。

**图 3.12 RLHF vs DPO 流水线（据 DPO Figure 1 重绘）。** 左：偏好数据 →（最大似然）→ 奖励模型 ⇄（打分 / 采样）LM 策略，标“强化学习”；右：偏好数据 →（最大似然）→ 最终 LM。右图下方写隐式奖励 $\hat r_\theta=\beta\log(\pi_\theta/\pi_{ref})$。

**图 3.13 DPO 损失与梯度权重 vs 间隔。** 横轴 $m=\hat r_w-\hat r_l\in[-4,4]$；实线 $-\log\sigma(m)$，虚线 $\sigma(-m)$。标注：$m=0$（0.6931, 0.5）；$m=0.15$（0.6210, 0.4626）；$m=0.75$（0.3869, 0.3208）；$m=-1$ 权重 0.7311；$m=1$ 权重 0.2689。

**图 3.14 三种“KL”辨析示意（可选，强烈建议）。** 三栏：(a) RLHF 奖励里的 $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{ref})$——拉向**参考模型**，整个训练期固定锚点；(b) PPO §4 的 $\beta_{old}\,\mathrm{KL}(\pi_{\theta_{old}},\pi_\theta)$——拉向**上一轮快照**，每轮更新锚点，按 $d_{targ}$ 自适应；(c) PPO 裁剪——不是 KL，而是对 $\rho_t$ 的逐样本截断，锚点同 (b)。

**图 3.15 过度优化示意（据 Stiennon Figure 5 的定性形状，标“示意”）。** 横轴“对 RM 的优化强度（KL）”，两条曲线：RM 预测的偏好单调上升；真实人类偏好先升后降。图注引用 C36，并注明无原始数据、只画形状。

---

## 6. 算例与练习素材

**练习 1（计算，BT）。** RM 给偏好回答 2.1 分、非偏好回答 0.6 分。求 BT 下标注员偏好前者的概率与该对的 RM 损失。若两者都加 5 分呢？
答：$\Delta=1.5$，$\sigma(1.5)=0.8176$，损失 $-\ln0.8176=0.2014$。都加 5 分，$\Delta$ 不变，概率与损失不变——这是 InstructGPT 说 RM 损失 “invariant to shifts in reward”、需要额外用偏置把示范数据平均分归零的原因（C9）。

**练习 2（计算，排序→成对）。** 某标注任务 K=7。(a) 一个 prompt 产生多少对比较？(b) 每个回答参与几对？(c) 一个 batch 64 个 prompt、全部 K=7 时有多少对？(d) InstructGPT 为什么不把这些对打散后当独立样本训练？
答：(a) $\binom72=21$；(b) 6；(c) 64×21=1,344（上限是 K=9 时的 2,304）；(d) 同一回答会参与 $K-1$ 次梯度更新、相关性强，一个 epoch 就过拟合；放在一个 batch 元素里每个回答只需一次前向（C7）。

**练习 3（计算，PPO 裁剪四象限）。** ε=0.2。对 (ρ, A) = (1.5, +1)、(0.7, +1)、(0.7, −1)、(1.5, −1) 求 $L^{CLIP}$ 与 $\partial L/\partial\rho$。
答：1.2 / 0；0.7 / 1；−0.8 / 0；−1.5 / −1（§4.3）。规律：只在“已经往好的方向走够了”的一侧截断。

**练习 4（计算，从对数概率到裁剪）。** 某 token 采样时 $\log\pi_{\theta_{old}}=-2.30$，更新几步后 $\log\pi_\theta=-2.00$。求 ρ；若 $\hat A=+0.5$ 与 $\hat A=-0.5$，该项目标值与是否有梯度。
答：$\rho=e^{0.3}=1.3499$。$\hat A=+0.5$：$\min(0.675,0.6)=0.6$，被裁剪，梯度 0。$\hat A=-0.5$：$\min(-0.675,-0.6)=-0.675$，未裁剪，梯度照常（把这个 token 的概率往下压）。

**练习 5（计算，GAE）。** 用 §4.4 的数据（β=0.05，RM=1.0，逐 token 对数比 [0.10, −0.05, 0.20, 0.00]，$V$=[0.5, 0.6, 0.7, 0.9]，γ=1，λ=0.95）计算 $r_t$、$\delta_t$、$\hat A_t$；再算 λ=0 与 λ=1 的结果并比较。
答：$r=[-0.005,0.0025,-0.01,1.0]$；$\delta=[0.095,0.1025,0.19,0.1]$；$\hat A^{0.95}=[0.4496,0.3733,0.2850,0.1]$；$\hat A^{0}=\delta$；$\hat A^{1}=[0.4875,0.3925,0.29,0.1]$。λ 越大越依赖真实回报（低偏差高方差），越小越依赖 $V$。

**练习 6（计算 + 推导，KL 正则最优策略）。** $\pi_{ref}=[0.6,0.3,0.1]$，$r=[0,1,2]$，β=1。(a) 求 $Z$ 与 $\pi^*$；(b) 用式 (5) 从 $\pi^*$ 反算奖励；(c) 验证最优目标值 $=\beta\log Z$。
答：(a) $Z=0.6+0.3e+0.1e^2=2.1544$，$\pi^*=[0.2785,0.3785,0.3430]$；(b) $\log(\pi^*/\pi_{ref})+\log Z$ 得 [0, 1, 2]；(c) $\E r=1.0645$，KL=0.2970，差 0.7675 $=\ln2.1544$。

**练习 7（推导，配分函数为何抵消）。** 从式 (5) 出发，写出 $p^*(y_1\succ y_2|x)$，说明 $Z(x)$ 为何消失；如果偏好模型不是“只依赖差值”的形式，这一步还成立吗？
答：见 §2.6 与附录 A.2；关键是同一 $x$ 下两个回答共享 $\beta\log Z(x)$，而 BT 只依赖奖励差。若偏好概率依赖奖励的绝对值（例如 $\sigma(r_1)$ 这类非差分形式），$Z(x)$ 不会约掉，DPO 的重参数化就用不上。

**练习 8（计算，DPO 隐式奖励）。** β=0.1，$\log\pi_\theta(o_w)=-12.0$、$\log\pi_{ref}(o_w)=-13.0$、$\log\pi_\theta(o_l)=-15.5$、$\log\pi_{ref}(o_l)=-15.0$。求 $\hat r_w$、$\hat r_l$、损失、梯度权重。换成 β=0.5 呢？
答：β=0.1：0.1、−0.05、0.6210、0.4626；β=0.5：0.5、−0.25、0.3869、0.3208（§4.7）。

**练习 9（概念，DPO 的陷阱）。** 若训练后 $\log\pi_\theta(o_w)$ 从 −13.0 降到 −14.0，而 $\log\pi_\theta(o_l)$ 从 −15.0 降到 −18.0（β=0.1），损失上升还是下降？这说明了什么？
答：$m=-0.1-(-0.3)=0.2>0$，损失从 0.6931 降到 0.5981。DPO 只约束**相对**间隔，偏好回答的绝对概率可以下降（M-9）。

**练习 10（计算，显存账）。** 7B 策略，按 16/2 字节规则，比较 SFT、DPO、PPO（四个同尺寸）的模型状态显存；再算 InstructGPT 175B 策略配 6B RM 与 6B 价值网络的情形。
答：112、126、252 GB；3,258 GB，其中价值网络只占 96 GB（2.9%）（§4.10）。

**练习 11（计算，PPO 自适应 KL）。** $d_{targ}=0.01$。一轮更新后测得 $d=0.004$；下一轮测得 $d=0.02$。β 如何变化？这个 β 和 RLHF 奖励里的 β 是同一个东西吗？
答：0.004 < 0.00667 → β 减半；0.02 > 0.015 → β 加倍。不是同一个：PPO §4 的 KL 以 $\pi_{\theta_{old}}$ 为锚，RLHF 的 KL 以 $\pi_{ref}$ 为锚（M-3）。

**练习 12（推导，逐 token KL）。** 证明：不折扣时，逐 token 奖励 $-\beta\log\frac{\pi_\theta(o_t|\cdot)}{\pi_{ref}(o_t|\cdot)}$ 之和加上末 token 的 RM 分，等于式 (2) 第一项的序列级奖励；并说明为什么其期望是 $\beta\,\mathrm{KL}$。
答：自回归分解把序列对数比拆成逐 token 之和（§2.3）；$o\sim\pi_\theta$ 时 $\E[\log\frac{\pi_\theta}{\pi_{ref}}]$ 按 KL 定义就是 $\mathrm{KL}(\pi_\theta\|\pi_{ref})$。

**练习 13（计算，Elo）。** 为什么 HH-RLHF 说 Δ(Elo) ≈ 174×Δ(PM)？
答：令 $\sigma(\Delta_{PM})=1/(1+10^{-\Delta_{Elo}/400})$，即 $e^{-\Delta_{PM}}=10^{-\Delta_{Elo}/400}$，得 $\Delta_{Elo}=\frac{400}{\ln10}\Delta_{PM}=173.7\,\Delta_{PM}$。

---

## 7. 勘误与常见误解

### 7.1 论文自身的印刷/一致性错误（E 类）

**E-1. DPO 附录 A.4 梯度推导（v1 与 v3 PDF 均如此）。** 式 (21) 中 $\sigma$ 的自变量写成 $\beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}-\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}$，与式 (7) 相反；A.4 最后一式的权重写成 $\sigma(\hat r_w-\hat r_l)$，与 §4 正文 $\sigma(\hat r_l-\hat r_w)$ 相反。§4 正文版本正确（§4.8 有限差分验证）。补充（核查时发现）：按原文自己定义的 $u=\hat r_l-\hat r_w$ 从式 (22) 推下去，末式前面应是“+”而原文印“−”，所以 A.4 末式既不是式 (7) 的梯度（权重错），也不是式 (21) 的梯度（符号错），数值上恰为式 (21) 梯度的相反数（§4.8）。课文推导以 §2.6 的“正确版本”为准，并可在脚注提一句原文附录的笔误。

**E-2. PPO 式 (10)(11) 的指数。** 印刷为 $\gamma^{T-t+1}r_{T-1}$ 与 $(\gamma\lambda)^{T-t+1}\delta_{T-1}$；从 $r_t$（指数 0）数到 $r_{T-1}$，指数应为 $T-t-1$（例：$T=5,t=1$ 时需要 3，印刷给出 5）。与 GAE 论文式 (14) 对照可确认。

**E-3. Ziegler 式 (1) 漏了负号。** 印刷的“loss”是对数似然 $\E[\log\frac{e^{r(x,y_b)}}{\sum_ie^{r(x,y_i)}}]$，作为损失最小化会把正确答案的概率压低。HTML 与 PDF 都是这样。课文若引用应写 $-\E[\cdots]$ 并注明原文笔误。

**E-4. DPO §5.2 “π∗ is the optimal policy from Eq. 7”。** 式 (7) 是 DPO 损失，最优策略是式 (4)。v1、v3 都如此。

**E-5. InstructGPT：KL 参考模型不是正文所说的“SFT 模型”。** 正文 §3.5 与式 (2) 写 $\pi^{\mathrm{SFT}}$，而 §C.3/§C.4 表明参考模型是“预训练 GPT-3 + 演示数据 2 epoch + 10% 预训练混合”的 RL 初始化模型（C13）。§E.6 的 KL 扫描实验里又换成了 “The pretrained GPT model is used as the KL reward model”。三处参考模型不同，课文说“参考模型通常是 SFT 模型”时应加限定。

**E-6. InstructGPT：RM 从哪里初始化，三处说法不同。** arXiv §3.5（SFT 去掉 unembedding）、§C.2（在公开 NLP 数据上微调过的 6B GPT-3，“mostly for historical reasons”）、NeurIPS 正文（“We fine-tune GPT-3”）。课文只宜说“用 6B 的 GPT-3 系模型加一个标量输出头”，并引 §C.2 说初始化选择影响不大。

**E-7. InstructGPT §E.2 的 inter/intra 疑似写反。** §E.2 说 “inter- and intra-group validation accuracies … are 72.4±0.4%, and 69.6±0.9% respectively”，而 §4.1 明说 69.6% 是**留出组**、72.4% 是**训练组**。按字面，跨组（inter）应是 69.6%。以 §4.1 为准。

**E-8. InstructGPT 表号交叉引用错位。** §3.2 说 “In Table 2 we show the distribution of use-case categories”，但用例分布是 Table 1（Table 2 是示例 prompt）；§4.1 的 “see Table 2” 同理。NeurIPS 版中数据集大小表变成 Table 3（arXiv 为 Table 6）。

**E-9. InstructGPT Figure 2 画了平局（D > C > A = B），但 §C.2 说 “Ties were dropped.”** 重绘时不要把平局画成训练信号。

**E-10. “100x” 的两种写法。** 摘要 “100x fewer parameters”，§1 “over 100x fewer parameters”；实际 175/1.3 = 134.6。课文写“约 135 倍（论文称逾 100 倍）”。

**E-11. InstructGPT §C.3 的附录交叉引用指错。** §C.3 说 RL 初始化模型混 10% 预训练数据 “since we find it helpful for PPO training (see Appendix E.11 for details)”，但初始化模型的对比实验（1 或 2 epoch × 0%/10%/50% 预训练混合，Figure 37）在 **§E.8 “PPO init models”**；§E.11 讲的是 PPO-ptx 的预训练数据比例与 batch 扫描。§E.8 还补了一句 “PPOs’ performance seems not sensitive to these particular choice”——课文不宜把“10% 混合”说成关键因素。

### 7.2 版本漂移（V 类）

**V-1. InstructGPT：arXiv v1 vs NeurIPS 2022。** NeurIPS 正文删去式 (2)（PPO-ptx 目标），RM 初始化措辞不同，数据集大小表号 6→3，附录编号也变（arXiv 的 Appendix C 模型细节对应 NeurIPS 的 Appendix D）。**正文节号也移位**：NeurIPS 版没有 arXiv 的 §3.3 “Tasks”，所以 arXiv §3.4 人类数据 / §3.5 Models / §3.6 Evaluation 在 NeurIPS 是 §3.3 / §3.4 / §3.5；NeurIPS 版也没有 arXiv §5.2 “Who are we aligning to?”（C21 引的是 arXiv 节号）。本笔记所有式号、表号、附录号、节号均指 **arXiv v1**。

**V-2. DPO：v1（2023-05，预印本）vs v3（2024-07，NeurIPS 2023 相机版之后）。** 附录 B 优化器 v1 为 Adam、v3 为 RMSprop；v1 没有 §6.3 CNN/DailyMail 分布外实验（v3 Table 1：DPO 0.36/0.31 vs PPO 0.26/0.23，温度 0/0.25）；摘要措辞改写。主要公式与 61%/57%/58% 数字未变；附录 A.4 的笔误两版都在。

**V-3. PPO：v1 vs v2（2017-08-28）。** 仅引言结尾一句的比较对象变化（C46），公式、Table 1、附录超参数未变。（核查时 pypdf 重新 diff：正文文字确实只有这一句不同；另有若干**图内坐标刻度文本**不同，如 “17.5”“40000” 只出现在 v2，说明有图被重绘，未逐图核对。）

**V-4. 附录章节号。** DeepSeekMath 引 GAE 写 “(Schulman et al., 2015)”（arXiv 年份），GAE 正式发表是 ICLR 2016。

### 7.3 常见误解（M 类）

**M-1. “RLHF 是 InstructGPT/ChatGPT 发明的。”** 偏好学习奖励 + RL 的深度版本是 Christiano et al. 2017（Atari/MuJoCo，NIPS 2017）；把“比较数据训练的 RM + PPO + KL 惩罚”用到预训练语言模型（GPT-2 774M）的是 Ziegler et al. 2019（论文本身不称“首次”，且列出了 Kreutzer 2018、Jaques 2019、Yi 2019 等更早的“人类反馈 + RL 用于语言”工作，所以课文不要写“第一个”）；摘要任务是 Stiennon et al. 2020。InstructGPT 自述 “Our methodology follows that of Ziegler et al. (2019) and Stiennon et al. (2020)”（§3.1）。

**M-2. “Christiano 2017 用的是 PPO。”** 用的是 A2C（Atari）和 TRPO（MuJoCo）（C25）；PPO 晚 38 天才发布。

**M-3. 三种“KL”混为一谈。** (a) RLHF 奖励中的 $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{ref})$：锚点固定为参考模型，目的是防奖励黑客与保持语言质量（Ziegler、Stiennon、InstructGPT、DPO 式 (3)）；(b) PPO §4 的自适应 KL 惩罚：锚点是 $\pi_{\theta_{old}}$，是 PPO 的**备选**信赖域机制，论文实验里比裁剪差；(c) PPO 裁剪本身不是 KL。InstructGPT 用的是 (a) + (c)，没有用 (b)。

**M-4. “PPO 裁剪保证新旧策略比值留在 [1−ε, 1+ε] 内。”** 裁剪只是让越界一侧的**该样本项梯度为 0**（“removes the incentive”），参数更新仍可能因其他样本、其他 token 把比值推出区间；而且在“往坏方向越界”的一侧根本不裁剪（悲观下界）。

**M-5. “PPO 的四个模型一样大。”** 取决于实现：Stiennon 四个同尺寸；InstructGPT 的 RM 和价值网络固定 6B，175B 策略时它们只占模型状态的约 3%（§4.10）。DeepSeekMath 的 “comparable size” 描述的是常见做法，不是定律。

**M-6. “旧策略 $\pi_{\theta_{old}}$ 也要单独占一份显存。”** 只需要 rollout 时记录下的逐 token 对数概率（[推论]，§3.6 注 1）。真正额外的权重拷贝在 InstructGPT 里是 EMA 权重。

**M-7. “DPO 不需要参考模型 / 没有奖励模型 / 去掉了 KL 约束。”** 三句都不对：损失里显式出现 $\pi_{ref}$（但其对数概率可离线预计算）；奖励模型是隐式的 $\hat r_\theta=\beta\log(\pi_\theta/\pi_{ref})$（论文标题正是 “Your Language Model is Secretly a Reward Model”）；DPO 优化的正是式 (3) 的 KL 约束目标，β 的角色与 RLHF 中相同。

**M-8. “DPO 与 PPO-RLHF 在实践中完全等价。”** 等价性是在非参数策略类、BT 偏好模型、无限数据的前提下成立的（A.1 “general non-parametric policy class”；Theorem 1 “Under mild assumptions”）。实践中 DPO 用离线数据、不从当前策略采样；PPO 在线采样还能用无标签 prompt（DPO §6.3 自己也点出 “DPO does not use the additional unlabeled Reddit TL;DR prompts that PPO uses”）。

**M-9. “DPO 损失下降意味着偏好回答的概率上升。”** 只意味着隐式奖励**间隔**变大；偏好回答的绝对对数概率可以下降（§4.7、练习 9）。DPO 论文未讨论此现象，写课文时标“笔者推导的可能失败模式”。

**M-10. “InstructGPT 1.3B 在各项能力上胜过 175B GPT-3。”** 结论是**标注员在 API prompt 分布上的偏好**；公开 NLP 数据集上 RLHF 反而有回退（“alignment tax”），需要 PPO-ptx 弥补（C17、C19）。

**M-11. “加大 KL 系数就能保住预训练能力。”** InstructGPT §E.6：KL 系数加到默认的 100 倍仍修不好 SQuAD/DROP 回退，必须混预训练梯度（C15）。

**M-12. “KL 系数是通用超参数，0.02 就是标准值。”** 各系统相差一个数量级以上（0.001 / 0.02 / 0.05 / DPO 0.1–0.5），因为奖励尺度不同（InstructGPT 让示范均值为 0，Ziegler 归一化到均值 0 方差 1，HH 直接用 PM 分），而且 DPO 的 β 与 PPO-RLHF 的 β 虽然在式 (3) 中角色相同，但训练动力学不同，数值不可直接比较。

**M-13. “InstructGPT 用 GAE 时 γ 是 0.99。”** §C.4：“No discount is applied”，即 γ=1；式 (2) 里的 γ 是**预训练系数** 27.8，不是折扣。

**M-14. “RM 训练很多个 epoch。”** InstructGPT、Ziegler 都只训 1 个 epoch，多 epoch 迅速过拟合（§C.2 “multiple epochs quickly overfit”；Ziegler §2.2 “We use a single epoch”）。

**M-15. “Bradley–Terry 最早由 Bradley & Terry 1952 用 logistic 形式提出。”** 1952 年论文用的是强度参数比例形式（待核原文，第 8 节），logistic/Elo 形式是后来的等价重参数化；更早的 Zermelo（1929）也有等价模型——此条为记忆线索，**未核实**，课文不要写，除非另行核对。

**M-16. 记号撞车。** InstructGPT 的 $\theta$=RM、$\phi$=策略，DPO 反过来；PPO 论文 $r_t$ 一会儿是概率比一会儿是奖励；Christiano 的 $\sigma$ 是轨迹片段；$K$ 在 InstructGPT 是回答数、在 PPO 是 epoch 数、在 DeepSeek 课程里是路由专家数。课文一律用 §2.1 右列。

---

## 8. 未能核实 / UNVERIFIED

1. **Bradley & Terry 1952 原文的模型写法**（是否为 $\pi_i/(\pi_i+\pi_j)$、是否有 logistic 形式）：只核实了书目信息（Crossref：JSTOR DOI 10.2307/2334029 标 1952-12；OUP DOI 10.1093/biomet/39.3-4.324 标 39(3-4):324–345；OUP 页面标 1952-12-01）。原文在 JSTOR/OUP 付费墙后，未读到正文。另：OUP 页面标题作 “Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons”（无 “I.”），Crossref 的 JSTOR 记录与 Christiano/DPO 的参考文献有 “I.”。
2. **Zermelo 1929 更早提出等价模型**：仅凭记忆，未检索。
3. **InstructGPT 的 GAE λ**：§C.4 只说不折扣，没有给 λ。
4. **InstructGPT 的价值损失形式、是否裁剪价值、熵奖励系数、是否对优势做归一化**：论文未报告。
5. **InstructGPT 的 EMA 权重精度与是否用 EMA 权重做评测**：未报告。
6. **InstructGPT “256k episodes” 的精确含义**（256,000 还是 262,144）：未说明，影响迭代数 500 vs 512。
7. **InstructGPT 的 6B 是否就是 GPT-3 6.7B**：全文无 “6.7B” 字样；Stiennon 用 6.7B。等同是推论。
8. **InstructGPT RM 实际训练的比较对数**：只知道 “an order of magnitude larger than the number of prompts”；上下界 199,242–1,195,452 是笔者按 K 的范围算的。
9. **Ziegler 的价值网络是否与策略共享参数、PPO2 的裁剪系数默认值**：原文只说 “default values for the other parameters”，未追到 Dhariwal et al. 2017（OpenAI Baselines）代码核对。核查补充：Ziegler v2 全文 grep “value function / value head / value network” 零命中；§4.2 只说 RM 与策略 “both initialized to ρ” 但 “train them as separate networks”——这是 RM 与策略的关系，**不是**价值网络的信息。
10. **Stiennon 的 PPO 裁剪系数**：全文（HTML 版）grep “clip / ϵ / epsilon” 零命中，论文未报告；表 3.3 中写“未见”。
11. **HH-RLHF 的价值网络设置**：未检索。
12. **DPO 附录 A.4 的错误是否已在作者 GitHub 或勘误中更正**：未检索作者仓库。
13. **DeepSeek LLM 的 DPO β 值**：2401.02954 §4 只给了 epoch、学习率、batch，没有 β（grep “β/beta” 只命中 AdamW 的 β1/β2）。
14. **ZeRO 的会议信息（SC 2020）**：凭记忆，本轮未核对出版页。（已解决：见文末“事实审读追加”F12，Crossref 确认 SC20，DOI 10.1109/SC41405.2020.00024。）
15. **InstructGPT Figure 1 的具体胜率数值**：原文只有图，没有数值表；重绘只能定性。
16. **Stiennon Figure 5 的数据点**：未提取，图 3.15 只画形状。
17. **InstructGPT 的 KL 惩罚在实现中是否就是“每 token 的采样对数比”（k1）而非解析 KL**：原文只说 “per-token KL penalty … at each token”，与 DeepSeekMath 式 (2) 的写法一致，但未见代码。
18. **“DPO 的参考对数概率可以预先计算”**：由损失形式推出（参考模型冻结、数据固定），DPO 论文本身未明说；附录 B 的参考代码把 `ref_logps` 作为输入。
19. **GAE 价值目标“= 优势 + V”** 的写法：GAE §5 讨论了 TD(λ) 目标，未逐字核对这一实现式。
20. **“Ziegler 2019 是第一个把 RLHF 用于预训练语言模型的工作”**：这是常见的二手说法，一手来源里找不到——Ziegler 本身不声称“首次”，它引用的 Kreutzer 2018、Jaques 2019、Yi 2019 本轮都没读原文，无法判断它们算不算“预训练 LM + 人类偏好 RL”。课文不写“第一个”。

---

## 9. 参考文献卡片草稿

```
- id: ouyang2022
  title: Training language models to follow instructions with human feedback
  a: Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, … Christiano, Leike, Lowe (2022), NeurIPS 2022. arXiv:2203.02155v1
  url: https://arxiv.org/abs/2203.02155
  finding: 用 SFT → 6B 奖励模型（K 路排序拆成 C(K,2) 对）→ 带逐 token KL 惩罚与预训练混合的 PPO，把 GPT-3 调成指令助手，1.3B 模型在 API prompt 上被偏好于 175B GPT-3。
  matters: 本课三步法、RM 损失式 (1)、PPO-ptx 目标式 (2) 和全部超参数（β=0.02、γ=27.8、ε=0.2）都以它为准，四模型显存账也用它的尺寸配置。

- id: christiano2017
  title: Deep reinforcement learning from human preferences
  a: Christiano, Leike, Brown, Martic, Legg, Amodei (2017), NIPS 2017. arXiv:1706.03741v4
  url: https://arxiv.org/abs/1706.03741
  finding: 用轨迹片段两两比较拟合 Bradley–Terry 奖励预测器，再用 A2C/TRPO 优化，只需不到 1% 交互的人类反馈即可学会 Atari 与后空翻等行为。
  matters: 课文追溯 RLHF 源头、说明“偏好 → BT → 奖励”的思想早于语言模型，并纠正“最初就用 PPO”的误解。

- id: ziegler2019
  title: Fine-Tuning Language Models from Human Preferences
  a: Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, Irving (2019), arXiv preprint. arXiv:1909.08593v2
  url: https://arxiv.org/abs/1909.08593
  finding: 把人类比较数据训练的奖励模型 + PPO 用于预训练语言模型（GPT-2 774M），并在奖励中加入对初始模型的 KL 惩罚与自适应 β 控制器（论文不称“首次”）。
  matters: 它是 KL 修正奖励式 R = r − β log(π/ρ) 的直接来源，也提供“去掉 KL 就输出乱码”的反例（Table 10，mock sentiment 任务）。

- id: stiennon2020
  title: Learning to summarize from human feedback
  a: Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei, Christiano (2020), NeurIPS 2020. arXiv:2009.01325v3
  url: https://arxiv.org/abs/2009.01325
  finding: 在 TL;DR 摘要上，成对 BT 奖励模型 + PPO（KL 系数 0.05，γ=1，λ=0.95）训练的 1.3B 模型胜过 10 倍大的监督模型，但过度优化 RM 会与人类偏好背离。
  matters: 课文用它说明“四个同尺寸模型”的配置、KL 项的熵奖励解释，以及奖励过度优化这一失败模式。

- id: schulman2017ppo
  title: Proximal Policy Optimization Algorithms
  a: Schulman, Wolski, Dhariwal, Radford, Klimov (2017), arXiv preprint. arXiv:1707.06347v2
  url: https://arxiv.org/abs/1707.06347
  finding: 提出裁剪概率比的替代目标 L^CLIP（ε=0.2 在消融中最好），用一阶优化和多 epoch minibatch 复用样本，近似 TRPO 的信赖域效果。
  matters: 本课 PPO 小节的式 (6)–(12)、裁剪函数图和四象限分析都转录自它，也用它区分“对旧策略的 KL 惩罚”与 RLHF 的参考 KL。

- id: schulman2016gae
  title: High-Dimensional Continuous Control Using Generalized Advantage Estimation
  a: Schulman, Moritz, Levine, Jordan, Abbeel (2016), ICLR 2016. arXiv:1506.02438v6
  url: https://arxiv.org/abs/1506.02438
  finding: 把优势估计写成 TD 残差的 (γλ) 指数加权和，λ 在偏差与方差之间插值，λ<1 只在价值函数不准时引入偏差。
  matters: 课文用它解释价值网络（critic）为何存在、PPO-RLHF 中逐 token 优势怎样算，为第 8 课“GRPO 去掉 critic”铺垫。

- id: bradley1952
  title: Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons
  a: Bradley, Terry (1952), Biometrika 39(3/4):324–345. doi:10.2307/2334029 (JSTOR) / 10.1093/biomet/39.3-4.324 (OUP)
  url: https://doi.org/10.2307/2334029
  finding: 提出成对比较的统计模型，每个对象一个强度参数，对象 i 胜过 j 的概率由两者强度决定（原文参数化形式未读到，见 UNVERIFIED）。
  matters: 奖励模型损失与 DPO 都建立在这个偏好模型上，课文需要给出正确的书目出处而不是二手引用。
  kind: paper (non-arXiv)

- id: rafailov2023dpo
  title: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
  a: Rafailov, Sharma, Mitchell, Ermon, Manning, Finn (2023), NeurIPS 2023. arXiv:2305.18290v3
  url: https://arxiv.org/abs/2305.18290
  finding: 利用 KL 约束奖励最大化的闭式最优策略，把奖励重参数化为 β log(π/π_ref)，BT 模型中配分函数抵消，从而用一个二分类损失直接训练策略。
  matters: 课文 DPO 小节的逐行推导（式 (4)–(7)、附录 A.1–A.4）和隐式奖励算例都来自它，同时记录其附录 A.4 的梯度符号笔误。

- id: bai2022hh
  title: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
  a: Bai, Jones, Ndousse, Askell, Chen, DasSarma, … Kaplan (2022), arXiv preprint. arXiv:2204.05862v1
  url: https://arxiv.org/abs/2204.05862
  finding: 在有用性与无害性偏好数据上做 RLHF，KL 系数仅 0.001，观察到 PM 奖励与 √KL 近似线性，并给出 Elo 与 PM 分的换算。
  matters: 课文用它说明 KL 系数的取值因奖励尺度而差异巨大，并提供 HH 数据集（DPO 的对话实验所用）的来源。

- id: shao2024deepseekmath
  title: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
  a: Shao et al. (2024), arXiv preprint. arXiv:2402.03300v3
  url: https://arxiv.org/abs/2402.03300
  finding: 指出 PPO 的价值函数通常与策略同量级，带来显存与计算负担，并提出用组内平均奖励作基线的 GRPO。
  matters: 本课结尾用它的原话为第 8 课埋伏笔，把“四个模型的显存账”与 DeepSeek 的设计动机连起来。

- id: rajbhandari2019zero
  title: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
  a: Rajbhandari, Rasley, Ruwase, He (2020), arXiv:1910.02054v3
  url: https://arxiv.org/abs/1910.02054
  finding: 混合精度 Adam 训练每个参数需要 2+2+12=16 字节的模型状态（fp16 参数与梯度，加 fp32 主权重、动量、方差）。
  matters: 课文 PPO/DPO/GRPO 显存对比的“16 字节/可训练参数”假设以它为出处。

- id: brown2020gpt3
  title: Language Models are Few-Shot Learners
  a: Brown et al. (2020), NeurIPS 2020. arXiv:2005.14165
  url: https://arxiv.org/abs/2005.14165
  finding: Table 2.1 列出 GPT-3 八个尺寸，其中 GPT-3 XL 为 1.3B、GPT-3 6.7B 为 6.7B、最大为 175B。
  matters: 只用于核对 InstructGPT 所说 1.3B/6B/175B 与 GPT-3 家族的对应关系（6B≈6.7B 为推论）。

- id: deepseek2024llm
  title: DeepSeek LLM: Scaling Open-Source Language Models with Longtermism
  a: DeepSeek-AI (2024), arXiv preprint. arXiv:2401.02954v1
  url: https://arxiv.org/abs/2401.02954
  finding: DeepSeek Chat 在 SFT 之后用 DPO 做一轮偏好优化（1 epoch，学习率 5e-6，batch 512），偏好数据按有用性与无害性构造。
  matters: 课文用它说明 DPO 是 DeepSeek 第一代对齐方案，为第 6 课与第 8 课（改用 GRPO）做衔接。
```

### 本轮额外读过、但不单独立卡的来源

- NeurIPS 2022 论文集页面（InstructGPT 作者列表与 “Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Main Conference Track”）：proceedings.neurips.cc/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html。注意：一次 WebSearch 的摘要把作者 “Alex Ray” 误写成 “Alex Gray”，以论文集页面原文为准。
- NeurIPS 2017 论文集中 Christiano et al. 的 PDF 链接（proceedings.neurips.cc/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf），仅用于确认会议归属。
- Jaques et al. 2017（Sequence Tutor, ICML 2017）与 Jaques et al. 2019（arXiv 1907.00456）：只通过 Ziegler/Stiennon 的参考文献条目看到，未读原文；它们是 KL 约束做法的更早出处，若课文要提需另行核对。

---

## 10. 核查记录（对抗式复核，2026-09-24）

做法：不采信初稿，逐条回到一手来源重新 grep / 重新复算。来源：arXiv LaTeXML 全文（`arxiv.py text`：2203.02155v1、2305.18290v3、1706.03741v4、1909.08593v2、2009.01325v3、1506.02438v6、2204.05862v1、1910.02054v3、2401.02954v1、2402.03300v3、2005.14165）；arXiv PDF + pypdf（1707.06347 v1/v2、2305.18290 v1/v3、1909.08593v2）；NeurIPS 2022 论文集 PDF（InstructGPT，15 页）；arXiv export API（各版本日期）；Crossref API（两个 Bradley–Terry DOI）；Python 3.11 + numpy 复算。

### 10.1 核对无误（照原文逐字/逐符号比对）

| # | 论断 | 核对结果与出处 |
|---|---|---|
| 1 | InstructGPT 式 (1) RM 损失 | 与 §3.5 LaTeX 逐符号一致（$-\frac{1}{\binom K2}E[\log\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))]$）；NeurIPS 版式 (1) 相同 |
| 2 | InstructGPT 式 (2) PPO-ptx 目标 | 与 §3.5 一致；“For "PPO" models, γ is set to 0.”；NeurIPS 正文确无此式，只写 “mixing the pretraining gradients … (see Appendix D.4)” |
| 3 | §C.4 超参数 β=0.02、256k episodes、512/64、单 inner epoch、EMA 0.992、不折扣、clip 0.2、6B RM + 6B 价值、价值 LR 9e-6/5e-6、8 倍预训练样本、γ=27.8 | 全部逐字命中 §C.4 |
| 4 | §C.3 参考模型 = 2 epoch + 10% 预训练混合；“These models are also used to compute the KL reward”；§E.6 用 “pretrained GPT model … as the KL reward model” | 逐字命中（E-5 成立） |
| 5 | Table 6 数据集规模（11,295/1,430/1,550/103；6,623/26,584/3,488/14,399；31,144/16,185） | 逐项命中 §A.3 Table 6；加总 12,725 / 33,207 / 17,887 / 1,653 复算正确 |
| 6 | 标题结果 85 ± 3%、71 ± 4%、“100x” / “over 100x” | §1（摘要与正文）命中；NeurIPS 版同；175/1.3 = 134.6 复算正确 |
| 7 | §4.1 留出组 69.6 ± 0.9% vs 训练组 72.4 ± 0.4%；§E.2 把 inter/intra 写成 72.4/69.6 | 两处原文均命中，E-7 成立 |
| 8 | §E.6 KL 系数扫到 2.0 = 默认 100 倍仍修不好；§E.7 最优 0.01–0.02；γ ≥ 20 可恢复、27.8 通用 | 逐字命中 |
| 9 | PPO 式 (7)、ε=0.2、“lower bound (i.e., a pessimistic bound)”、§4 规则 1.5/2、Table 1 全部 11 个数、Table 3/5 超参数 | v2 PDF 逐项命中；7 环境 × 3 种子 = 21 次命中 §6.1 |
| 10 | PPO 式 (10)(11) 指数印成 $T-t+1$（E-2） | v2 PDF 原样为 “γT−t+1rT−1 … γT−tV(sT)” 与 “(γλ)T−t+1δT−1”；GAE 式 (14) 取 $k=T-t$ 得 $\gamma^{T-t-1}r_{T-1}+\gamma^{T-t}V(s_T)$，勘误成立 |
| 11 | GAE 式 (10)(14)–(18)、“λ<1 introduces bias only when the value function is inaccurate”、Cart-pole 区间 [0.96,0.99]/[0.92,0.99]（21 个种子）、§7 “[0.9,0.99]” | 逐字命中 v6 |
| 12 | DPO 式 (1)–(10)、(11)–(20)、§4 梯度（前面有 $-\beta$） | 与 v3 LaTeX 逐符号一致 |
| 13 | DPO 附录 A.4 两处错误（E-1） | v3 LaTeX 与 v1 PDF 均为：式 (21) 的 σ 自变量是 $\beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}-\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}$，末式权重 $\beta\sigma(\hat r_w-\hat r_l)$。另注意：若按原文的 $u$ 推，末式前面应为“+”，原文印“−”，所以 A.4 的推导即使自洽地看也有一处符号问题；§4 正文版本是对的 |
| 14 | DPO §5.2 “optimal policy from Eq. 7”（E-4） | v1 PDF 与 v3 均如此；v3 式号 (4) 是最优策略、(7) 是 DPO 损失 |
| 15 | DPO 附录 B 超参数与 v1→v3 优化器漂移（V-2） | v3 “RMSprop … 1e-6 … 150 steps … TL;DR β=0.5”；v1 “the Adam optimizer”；v1 页脚 “Preprint. Under review.”、v3 页脚 NeurIPS 2023；CNN/DM Table 1（0.36/0.31 vs 0.26/0.23）只在 v3 |
| 16 | DPO 61% / 57% / 58% | §6.2 与 Table 2（Human win % = 58）命中 |
| 17 | Christiano：式 (1) 与损失、Bradley–Terry 与 Elo 类比、10% 随机作答、A2C + TRPO、900 queries < 1 hour、“less than one percent”（arXiv 摘要；正文写 “less than 1%”） | 逐字命中 v4 |
| 18 | 日期：Christiano v1 2017-06-12、v4 2023-02-17；PPO v1 2017-07-20、v2 2017-08-28；间隔 38 天 | arXiv API 命中；天数 Python 复算 = 38 |
| 19 | Ziegler 式 (1) 无负号（E-3）、式 (2)、控制器 clip 到 [−0.2,0.2]、$K_\beta=0.1$、774M/36 层/20 头/1280、2M episodes、γ=1、4 个 PPO epoch、RM 单 epoch | HTML 与 PDF 均无负号；其余逐字命中 |
| 20 | Stiennon：损失式、KL 0.05、γ=1、λ=0.95、4 epoch、价值网络独立且由 RM 初始化、三者同尺寸、64,832、61% vs 43%、Figure 6 的 1.1%/1.8%、“anti-correlated” | 逐字命中 v3 |
| 21 | HH-RLHF：$\lambda_{KL}=0.001$ + “might actually be wholly unnecessary”、式 (2.1) 与 174、式 (4.2)、ε=0.2、γ=1、K=1/2/4、44k/42k、52k/2k、13M–52B 共 7 个 | 逐字命中；400/ln10 = 173.72 复算 |
| 22 | ZeRO $2\Psi+2\Psi+K\Psi=16\Psi$、“Mixed-precision Adam has K=12” | 逐字命中 §3.1 |
| 23 | DeepSeek LLM DPO：1 epoch、5e-6、batch 512、未给 β | 逐字命中 §4；grep β 只命中 AdamW |
| 24 | DeepSeekMath 式 (2) 与 “comparable size”、“only the last token” | 逐字命中 §4.1.1 |
| 25 | GPT-3 Table 2.1：XL = 1.3B，“GPT-3 6.7B” = 6.7B；InstructGPT 全文无 “6.7B” | 命中 |
| 26 | Bradley–Terry 书目 | Crossref：10.2307/2334029 → Biometrika 39(3/4) p.324，1952-12，标题带 “I.”；10.1093/biomet/39.3-4.324 → 324–345，1952 |
| 27 | §4 全部复算数字（BT 表、噪声版、$\binom K2$ 与 batch 上下界、裁剪表、GAE 表与 4-token 例、玩具最优策略表、DPO 隐式奖励例、Elo、显存账 112/126/140/252/241.2/131.4/3,258 GB 与 2.9%、练习 1–13 答案） | Python 逐项复算，除 10.2 第 1 条外全部一致 |
| 28 | §4.8 DPO 有限差分 | 用 `numpy.random.default_rng(0)` 复现出 0.50645 / 0.49355 / 损失 0.70613 与 0.68033 |

### 10.2 本轮改动

1. **§4.6 数值**：$-\mathrm{KL}=-0.29699$ 改为 $-0.29697$（全精度 −0.2969679；§4.5 表中 KL=0.2970 本来就对，只是 §4.6 第 5 位小数错）。
2. **C58 出处**：“up to 6B parameters” 标的是 §1，实际在 §7 Discussion 的 “Limitations & Future Work” 段（v3 全文只出现一次）。
3. **Ziegler“第一次”的归属**（§1.3 标题、M-1、参考卡片 finding）：Ziegler 原文不称“首次”，且在 §1 相关工作里列出 Kreutzer 2018、Jaques 2019、Yi 2019。改成中性表述，并在第 8 节加第 20 条 UNVERIFIED。属于“telephone game”式归属。
4. **C31 / 参考卡片**：Table 10 的“去掉 KL 变乱码”是 mock sentiment（程序化奖励）实验，补注，避免被当成“对人类偏好 RM”的结论（adjacent-claim substitution）。
5. **新增勘误 E-11**：InstructGPT §C.3 写 “see Appendix E.11”，实际初始化模型实验在 §E.8；并补 §E.8 “not sensitive” 一句。C13 同步加注。
6. **V-1 补充**：NeurIPS 版少了 §3.3 Tasks 与 §5.2 “Who are we aligning to?”，正文节号整体前移。
7. **V-3 补充**：PPO v1/v2 正文只差一句，但图内坐标刻度文本不同，说明有图重绘。
8. **§3.5 DPO 表**：“22 次运行”包含 unlikelihood 与 Preferred-FT，补全扫描范围。
9. **§3.3 表 HH 列**：“多为 52B PM”补上出处 Figure 13；Ziegler 列来源补 §4.2。
10. **§4.8**：写明随机数生成方式（换用 `np.random.seed(0)` 或 `random.gauss` 得不到文中数字）。
11. **第 8 节第 9 条**：补充 Ziegler 全文没有任何价值网络描述；§4.2 “separate networks” 讲的是 RM 与策略。
12. **E-1 与 §4.8 补充**：A.4 末式按原文自己的 $u$ 推应为“+”却印“−”，数值上是式 (21) 梯度的相反数（有限差分确认）。E-1 的结论不变，只是多记一处。

### 10.3 仍然存疑

- Bradley & Terry 1952 原文参数化与 Zermelo 1929（第 8 节 1、2 条）仍未读到原文。
- DPO 附录 A.4 的错误是否在作者仓库或 OpenReview 相机版里改过：仍未检索（v1、v3 arXiv 均有错）。
- PPO v1/v2 图的具体差异未逐图核对。
- InstructGPT “6B” 等同于 GPT-3 6.7B 仍是推论。

---

## 补充核实（写课时追加，2026-09-24）

写课时没有引入新的外部事实；下面是课文里新出现的**计算值**和**措辞决定**，全部用 Python 3.11 复算（图的数值由 `deepseek/figures/gen/l03_*.py` 里的 assert 固定）。

### 新增算例（课文 §2–§9）

| 用处 | 输入 | 结果 |
|---|---|---|
| §1 SFT 例 | 三个 token 概率 0.5、0.2、0.9 | 损失 0.6931 + 1.6094 + 0.1054 = 2.4079，均值 0.8026；第二个 token 占 66.8% |
| §2 三回答排序 | $r_A,r_B,r_C=1.2,0.4,-0.3$，A > B > C | 三对损失 0.3711 / 0.2014 / 0.4032，均值 0.3252 |
| §2 Gumbel ⇒ BT | 笔者推导（换元 $u=e^{-a}$）；蒙特卡洛 200 万样本 $P(1.5+g_A>g_B)=0.8178$ vs $\sigma(1.5)=0.8176$ | 推导成立（数学事实，非论文内容） |
| §2 Elo 400 分 | $1/(1+10^{-1})$ | 0.909 |
| §3 全押回答 3 | $\pi=[0,0,1]$，$\beta=1$ | $\E r=2$，KL $=\ln10=2.3026$，目标 −0.3026 |
| §3 小练习 | $\beta=0.5$ 时 $\pi^*_3/\pi^*_1=\frac{0.1}{0.6}e^{4}$ | 9.10（与表中 0.6597/0.0725 一致） |
| 图 3.5 | 策略 $[0.1,0.8,0.1]$ | KL 0.6055，$\E r=1.0$；同 KL 的前沿点 $\beta=0.699$，$\E r=1.3191$，差 0.319 |
| §4 四 token 裁剪例 | (logπ_old, logπ_θ, Â) = (−1.20,−1.00,+0.8), (−0.50,−0.60,+0.5), (−2.00,−1.70,−0.6), (−0.90,−1.20,−0.4) | ρ = 1.2214/0.9048/1.3499/0.7408；项值 0.9600/0.4524/−0.8099/−0.3200；均值 0.0706；对 logπ 的梯度 0/0.4524/−0.8099/0 |
| §5 小练习 | §4.4 数据中 $V_3$ 改为 0.80 | $\hat A_1=0.45434$（原 0.4496） |
| 练习 6 | β=0.1，对数比 [0.2,−0.1,0]，RM 0.5，V=[0.30,0.35,0.40]，λ=0.9 | r=[−0.02,0.01,0.5]，δ=[0.03,0.06,0.10]，Â=[0.165,0.15,0.10]；λ=1：[0.19,0.16,0.10] |
| 练习 7(c) | β=0.5 反算 $r_3$ | $0.5\ln6.597+0.5\ln8.2765=2.000$ |
| 练习 8 | 13B 四网络同尺寸；InstructGPT 6B 策略配置 | 208/234/260/468 GB；216 GB |
| 练习 9 | β=0.2，logπ_θ(o_w)=−20.0，ref −20.5；logπ_θ(o_l)=−18.0，ref −17.0 | $\hat r_w=0.1$，$\hat r_l=-0.2$，损失 0.5544，权重 0.4256 |
| §6 计算账 | 第 2 课 2N/6N 规则，四网络同尺寸 | PPO ≈ 20N/token，DPO ≈ 8N，GRPO 式 ≈ 12N，SFT 6N（**笔者估算**，忽略注意力项） |

### 课文中标为“笔者推断 / 常见做法”的措辞（未在一手来源逐字核对）

1. 奖励模型从回答末尾 token 的隐藏状态读出分数：标为“常见实现，InstructGPT 原文未写明”。
2. SFT 只对回答部分计损失：标为“常见写法，InstructGPT 原文未写到”。
3. PPO 实现中监控新旧策略 KL 并提前停止一轮更新：标为“常见做法，不是 PPO 论文的要求”；本轮未找一手出处。
4. 价值目标写成 $\hat A_t+V_{\psi_{old}}(s_t)$：沿用第 2.5 节的说明，标为常见实现。
5. “写示范比做比较费事”：标为笔者判断，InstructGPT 未给耗时对比。
6. “最优策略主要在参考模型已会写的回答之间重新分配概率”：由式 (4) 得出的量级直觉，课文明说“不是定理”（softmax 概率恒为正）。
7. DPO“两边概率都降、损失仍下降”：沿用勘误 M-9，标为笔者推导的可能失败模式。

### 教学审读追加（2026-09-24，Teach 轮）

没有引入新的外部事实。改动与新增的计算值（Python 3.11 复算）：

| 用处 | 输入 | 结果 |
|---|---|---|
| §2 噪声版 BT 的损失上界 | $\hat P=0.05+0.9\sigma(\Delta)$ | 课文原写“$\Delta=10$ 时已经是 2.995”，应为 $\Delta=-10$（排反）：$-\ln(0.05+0.9\sigma(-10))=2.9949$；$\Delta=+10$ 时损失约 0.05。已改正 |
| §4 重要性比的直观例子 | 旧概率 0.2，新概率 0.3 / 0.1 | 权重 1.5 / 0.5 |
| §4 整段比值 = 逐 token 比值之积 | 200 个 token，每个比值 1.01 或 0.99 | $1.01^{200}=7.316$，$0.99^{200}=0.1340$。“逐 token 比值忽略前缀分布变化”的讨论与 [M09 §5] 一致（M09 引 TRPO 式 (3)、Kakade & Langford 2002），课文只链接 M09，不另立参考卡片 |
| §8 价值网络的账单 | 7B、四网络同尺寸 | 训练状态 112 → 224 GB；总账 140 → 252 GB；每 token 计算 12N → 20N（原文“计算多出三分之一以上”改为具体数字） |

### 数学审读追加（2026-09-24，Math 轮）

没有引入新的外部事实。逐行重推了全部 `::: deriv`，并用 Python（`fractions` 精确算术 + numpy）复算了课文、表格、练习答案和 15 个生成脚本里的每个数；15 张图都能由脚本逐字节复现。改动：

| 位置 | 问题 | 改正 |
|---|---|---|
| §5 GAE 例、式 (3.25) 下的价值目标、图 3.10 | $\hat A_2$ 的精确值是 $0.1025+0.95\times0.285=0.37325$。课文写 0.3732，再用 0.3732 算出 $\hat A_1=0.4496$，读者按计算器复核得 $0.095+0.95\times0.3732=0.44954\to0.4495$，对不上（0.3732 来自二进制浮点把 0.37325 舍低） | 课文写精确值 0.37325、0.44959；图和价值目标按四舍五入取 0.3733、0.9733；生成脚本改用 `Fraction` + `ROUND_HALF_UP` 并加断言 |
| §6 DPO 的显存倍数、图 3.12 | $126/112=1.125$，课文与图写 1.12 | 改为 1.125 |
| §7 “β 越大……梯度越弱” | 梯度权重 $\sigma(-m)$ 确实变小（0.4626 → 0.3208），但 (3.32) 前面乘着 $\beta$，对 $\log\pi_\theta$ 的实际梯度 $\beta\sigma(-m)$ 是 0.0463 → 0.1604，反而变大 | 改成“权重变小、损失更早饱和”，并给出两个实际梯度大小 |
| §4 “用优势代替回报不改变梯度的期望” | 只对真实优势（或回报减只依赖状态的基线）成立；GAE 估计在价值网络不准时有偏 | 加上前提，并指向 §5 |
| §4 一轮 PPO 迭代 | (3.8) 的 KL 奖励在实现里用 $\pi_{\theta_{old}}$ 算、当常数，没有说明为何不丢梯度 | 补一句：直接对 (3.7) 求导多出的项是 $-\beta\E[\nabla\log\pi_\theta]=0$（得分函数期望为零） |
| §2 “为什么偏偏是 sigmoid” | 原文暗示“只依赖差值”是 sigmoid 独有的好处；正态噪声（Thurstone）模型同样只依赖差值，DPO 推导照样成立 | 改写，说明换成正态模型时损失变成 $\log\Phi$ |
| §3 Gibbs 不等式 | 求和范围与 $p(x)=0$、$q(x)=0$ 的约定没写 | 求和改为 $p(x)>0$ 的项，注明 $0\log0=0$ 与 KL $=+\infty$ 的情形 |
| §3 蒙特卡洛前置框 | “误差按 $1/\sqrt N$ 缩小”缺方差有限的前提 | 写成“方差 $\sigma_f^2$ 有限时标准差 $\sigma_f/\sqrt N$” |
| §3 小练习（1） | 引用图 3.4 的 0.0725，但图里只印三位小数 0.072 | 注明四位小数来自练习 7：$0.6597/0.0725=9.10$ |
| §2 式 (3.5) 的注 | “多出 6 倍权重”有歧义（多出 6 倍 = 7 倍？） | 改为“36 对与 6 对，权重是 6 倍” |
| §6 式 (3.27) 的注 | 多 epoch 时“按 epoch 数翻倍” | 改为训练两项乘以 $n_{ep}$ |
| 练习 11 | 缺 $\pi_{ref}(o)>0$ 的前提；“驻点即最大值”的理由不完整 | 补前提；说明严格凹 + 线性约束 ⇒ 一阶条件点是唯一全局最大值 |
| 记号 | §7 的简写 $\ell^*(o)$ 与式 (3.18) 的 $\ell(\rho,A)$ 同字母异义；§7 代码下方写 $\pi^{ref}_w$ | 改为 $z^*(o)$；写全 $\pi_{ref}(o_w)$ |
| 图 3.5 生成脚本 | SVG 标题写“两个非最优策略”，图里只有一个（“全押回答 3”是前沿在 $\beta\to0$ 的端点） | 标题改为“一个非最优策略” |

### 数学审读追加（第二轮，2026-09-24，Math 轮）

没有引入新的外部事实。重新逐行推了全部 `::: deriv` 与行间公式，用 Python（numpy + `fractions`）复算了课文、表格、练习答案里的每个数，并重跑 15 个生成脚本，15 张 SVG 与重跑前逐字节相同（数值没有错误）。改动都是精度、前提与记号：

| 位置 | 问题 | 改正 |
|---|---|---|
| §3 β = 1 例题的熵分解核对 | 写 $H=1.0908$、$\E[\log\pi_{ref}]=-1.3877$、和 $-0.2970$；按所示四位小数相加得 $-0.2969$ | 写五位 $1.09076$、$-1.38773$、$-0.29697$，并说明舍入 |
| 练习 7(b) | $1.5872-0.5\times1.0609=1.05675$，按四舍五入是 1.0568，而答案写 1.0567；用四位小数的 $\pi^*$ 算 KL 得 1.0610 | 期望奖励改用 $\frac{2.2167+2\times5.4598}{8.2765}$；给出不舍入值 $\E r=1.58718$、KL $=1.06094$、目标值 $1.05671=0.5\ln8.27653$ |
| 式 (3.5) 的注 | 论文把 $1/\binom K2$ 写在期望外面，$K$ 随提示变化时字面不成立 | 补一句：先在每个提示的 $\binom{K_q}2$ 对上平均，再对提示取期望 |
| 式 (3.20) | $Q^\pi$ 的期望缺条件 $\mid s_t,a_t$；注里说“期望对奖励取”，但给定 $(s_t,a_t)$ 时 $r_t$ 与 $s_{t+1}$ 都是确定的 | 补条件；注改为 $Q^\pi=r_t+\gamma V^\pi(s_{t+1})$；§5 TD 残差处补“此时 $\delta_t$ 就等于 $A^\pi$” |
| 式 (3.24) 的注 | $\lambda=1$ 时加权平均的权重全为 0，推导不适用；回答结束后的 $\delta$ 没说明 | 说明 $\lambda=1$ 按左边的和定义（也是 $\lambda\to1$ 的极限），结束后的 $\delta$ 记为 0 |
| §4 一轮 PPO 迭代 | “把 KL 修正奖励当常数不丢梯度”只在 $\theta=\theta_{old}$ 处精确 | 补“在 $\theta=\theta_{old}$ 处” |
| §7 定理 1 | “奖励类”没有定义 | 补：只差一个只依赖提示的函数的一族奖励 |
| §3 读法 | “$\beta\to0$ 时押在最高分回答上”缺“参考模型给了正概率” | 补上（练习 2 答案原本就有） |
| §1 SFT 例 | 只给了对概率的导数 $-1/p$ | 补对 logit 的导数 $p-1$：$-0.8$ 对 $-0.1$，结论相同 |
| §7 附录笔误框 | 有限差分“对什么求导”没写 | 写明 softmax 策略、对三个 logits、$\beta=0.1$、$o_w,o_l$ 是第 1、3 个选项 |
| 记号 | 符号表 $N$ = 参数量，$\sigma$ = sigmoid，$V=\abs{\mathcal V}$，$G$ = GRPO 组大小，$M$ = 每 token FLOPs；课内另把 $N$ 用作蒙特卡洛样本数、$\sigma_f$ 作标准差、$V^{\abs{o}}$ 作回答数（与价值 $V$ 同字母）、$G_t$ 作回报、$M_{states}$ 作显存、$K$ 作 ZeRO 的优化器系数（与排序的 $K$ 撞车）；$\mathcal D$ 同时指偏好数据集与提示分布 | 蒙特卡洛改用 $n$ 与 $\sqrt{\Var[f]/n}$；回答数写 $\abs{\mathcal V}^{\abs{o}}$；回报写 $R_t$；显存写 $\mathrm{Mem}_{states}$ 并定义 $N_{train},N_{frozen}$；ZeRO 系数直接写 12；偏好数据集写 $\mathcal D_{pref}$（式 (3.3)、(3.5)、(3.30)） |
| 图 3.4、3.6、3.14 的说明 | 3.4“最后一列每一行”（参考分布行是“—”）；3.6“虚线右边”有两条虚线；3.14“离对角线多远”应是带符号的 $\Delta_w-\Delta_l$ | 分别改为“每个 β 行”、“竖虚线（真实偏好的最高点）”、“在对角线哪一侧、离它多远” |

### 事实审读追加（2026-09-24，Facts 轮）

逐条把课文里的数字、规格、日期、超参数、引语和归属对照本笔记（含第 10 节核查记录）；笔记没有覆盖或措辞需要更精确的，回到一手来源核对。来源与取法：`arxiv.py grep`（LaTeXML 全文：2203.02155、1706.03741、1909.08593、2305.18290 v1/v2/v3、2204.05862、2402.03300、2501.12948）；PPO 无 HTML 渲染，下载 1707.06347v2 PDF 用 pypdf 抽取后 grep；Crossref API（ZeRO）。15 张 SVG 的文字标签也逐一对过笔记（日期、表 1、表 6、显存数字都一致）。

**新核实的事实**

| # | 事实 | 一手出处（原文） | 课文用处 |
|---|---|---|---|
| F1 | PPO 论文对 $L^{CPI}$ 本身有一句专门的话：“Without a constraint, maximization of $L^{CPI}$ would lead to an excessively large policy update”；而 C39 的 “not well-justified … destructively large policy updates” 说的是 $L^{PG}$ | 1707.06347v2 §3（式 (6) 之后）与 §2.1 | §4“可是离开起点越远”一段原先只引了 $L^{PG}$ 那句来支撑关于 $\mathcal J^{CPI}$ 的论断（相邻论断替换），改为两句各归其位 |
| F2 | TRPO 的约束问题 “can efficiently be approximately solved using the conjugate gradient algorithm, after making a linear approximation to the objective and a quadratic approximation to the constraint”；PPO 的目标是 “using only first-order optimization” | 1707.06347v2 §2.2、§1 | §4 原写“带约束的优化要用二阶方法”，改为线性化 + 二次近似 + 共轭梯度，并引 PPO |
| F3 | 表 1 的 −0.39：“the score is negative for the setting without clipping or penalties, because for one environment (half cheetah) it leads to a very negative score, which is worse than the initial random policy” | 1707.06347v2 §6.1 | 图 3.8 说明补一句：负分主要来自 HalfCheetah 一个环境 |
| F4 | Ziegler 的 mock sentiment 奖励 $r_s$：一个 6 层 Transformer 分类器，在 McAuley et al. 2015 亚马逊评论数据的二值化、平衡子集上训练，$r_s$ 取它判“正面”的对数几率；“人”由“总选 $r_s$ 更高者”模拟 | 1909.08593v2 §3.1.1 | §3 原写“程序化的情感分类器”，改为“在亚马逊商品评论上训练的情感分类器”；参考卡片同步 |
| F5 | Christiano 自述 “Our algorithm follows the same basic approach as Akrour et al. (2012) and Akrour et al. (2014)”，并列出 Akrour 2011、Wilson 2012、Wirth 2016 等更早的偏好 RL 工作 | 1706.03741v4 §1.1 | §1“方法从哪里来”原写“是 Christiano 等人 2017 年……做的”，暗示首创；改为“做到深度强化学习的规模”，并写明其自述沿用 Akrour 等人 |
| F6 | Christiano 摘要：不到 1% 的交互给反馈，解决 “Atari games and simulated robot locomotion”；后空翻是另一句 “novel behaviors with about an hour of human time”（Hopper，“900 queries in less than an hour”） | 1706.03741v4 摘要、§3.2 | 参考卡片 finding 原把“<1%”和“后空翻”并在一起，已拆开 |
| F7 | InstructGPT 把 KL 参考换成预训练 GPT 的是 §E.6（1.3B、预训练系数 0、扫 KL 系数到 2.0 的对齐税实验）：“The pretrained GPT model is used as the KL reward model, in these experiments”；§E.7（最优 KL 系数约 0.01–0.02）是另一组扫描 | 2203.02155v1 §E.6、§E.7 | §3 警告框原写“附录的 KL 系数扫描实验”，有两组扫描，改为指明 E.6 |
| F8 | InstructGPT §C.4 原文 “Exponential moving averages of the weights are applied, with a decay rate of 0.992”，没说是策略还是也包括价值网络 | 2203.02155v1 §C.4 | §6 原写“对策略权重做了 EMA”，改为“对权重”，并注明原文未说对象 |
| F9 | InstructGPT 把一个提示的全部比较放进一个 batch 元素 “only requires a single forward pass of the RM for each completion”；5 折交叉验证是把标注员分成 5 组、训 5 个 RM（3 个种子）、在留出组上评估 | 2203.02155v1 §3.5 脚注、§4.1 | 核对 §2 已有表述，无需改 |
| F10 | HH-RLHF 附录 B.1：“PPO clipping ϵ=0.2, discount factor γ=1, and no entropy bonus” | 2204.05862v1 附录 B.1 | §4 式 (3.19) 注“熵奖励往往不用”补引 [@bai2022hh]；“因为 KL 里已含熵奖励”改标“笔者的理解” |
| F11 | DPO 附录 B 的优化器：v1 “Adam”，v2 与 v3 都是 “RMSprop”（其余超参数三版相同） | 2305.18290v1/v2/v3 附录 B | 支持课文“arXiv 第一版写的是 Adam，后来的版本改成了 RMSprop”（原笔记只核对了 v1 与 v3） |
| F12 | ZeRO 的正式发表：SC20（International Conference for High Performance Computing, Networking, Storage and Analysis），2020-11，pp. 1–16，DOI 10.1109/SC41405.2020.00024；arXiv 页面没有 comments / journal-ref 字段 | Crossref API | 第 8 节第 14 条（凭记忆的 “SC 2020”）现已核实；参考卡片补上会议与 DOI |
| F13 | R1-Zero 对数学、代码、逻辑推理数据用 “rule-based rewards”（accuracy rewards 与 format rewards），R1 的推理数据沿用这一做法 | 2501.12948v2（arxiv.py 取到的是 2026-01-04 的 v2 渲染，节号与 v1 不同）§2.2 “Reward Design”、§3.2.2 | 支持封面与 §6 的前向指针“R1 用规则判分代替了一部分奖励模型”（本课不立 R1 卡片，第 13 课再引） |
| F14 | DeepSeekMath：GRPO “uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline”；结果监督下奖励 “normalized by subtracting the group average and dividing by the group standard deviation” | 2402.03300v3 §4.1.1、§4.1.2 | 支持 §6 对 GRPO 的一句话概括（“比这组平均分高出几个标准差”） |
| F15 | DPO 自己说它没用无标签提示的那句在 §6.3（CNN/DailyMail 分布外实验）：“even though DPO does not use the additional unlabeled Reddit TL;DR prompts that PPO uses” | 2305.18290v3 §6.3 | 核对 §7 已有表述，无需改 |

**其他改动（不涉及新事实）**

- 参考卡片：`schulman2017ppo` 的 matters 原写“四种情形的分析都转录自它”，而四种情形的表是课文自己的分析，改为“以它为准 / 据其图 1 重绘 / 取自其表 1”；`ziegler2019` 的 matters 原写“它是 KL 修正奖励的直接来源”，InstructGPT 原文说它的 KL 奖励 “in the same way as Stiennon et al., (2020)”，且 Ziegler 自述沿用 Jaques 等人，改为“取自它的式 (2)，Stiennon 与 InstructGPT 用同一形式”；`shao2024deepseekmath` 的 matters “用它的原话”改为“用它对价值网络的两点批评”（课文是转述不是引语）。
- 跨课链接：§5 开头“[M09 §3–§4]”原指向 `#p4`，改为区间起点 `#p3`。其余 16 个跨课链接的 slug 都在 writer-brief 的清单里，`#pN` 对应的页标题与课文所说的内容一致（M04 §4 有“两个数的 softmax 就是 sigmoid”算例，M06 §1 有“sigmoid 的导数：第 3、13 课要用”小节，M10 §5 讲 ZeRO 分片，M09 §5 讲逐 token 比值与乘积比的方差）。
