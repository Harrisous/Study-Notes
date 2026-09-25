# 第 08 课研究笔记：DeepSeekMath 与 GRPO

> 服务于 `deepseek/src/lesson-08-deepseekmath-grpo.md`（short：GRPO）。建立于 2026-09-24。
> 核对方法：DeepSeekMath 读 arXiv v3 的 LaTeXML 全文（`arxiv.py text 2402.03300`），并与 v1、v2 逐行 diff；
> 图 4–7 在 HTML 版里没有渲染，改读 v3 PDF，其中图 5–7 是 matplotlib 矢量图，**数据点坐标是从 PDF 内容流里直接解析
> 出来的**（不是目测），换算误差约 ±0.1 个百分点。所有派生数字都在 Python 里重算过（第 4 节给出表达式）。
>
> **引文政策**：英文原文只摘关键短语（每条少于 15 词），其余用中文转述并给出精确位置。要看整句，运行
> `python .cursor/skills/academic-research/scripts/arxiv.py grep 2402.03300 '<短语>' -C 2`。
> 数值、超参数、公式属于事实与数学，照原文转录。

---

## 0. 覆盖范围与所读版本

### 0.1 主来源

| 键 | 标题 | 作者 | arXiv / 版本 | 状态 | 日期 | 读了什么 |
|---|---|---|---|---|---|---|
| `shao2024deepseekmath` | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo（DeepSeek-AI / 清华 / 北大） | 2402.03300 **v3** | arXiv 预印本（未找到同行评审版本，见 §8） | v1 2024-02-05；v2 2024-02-06；v3 2024-04-27 | v3 全文 + v3 PDF（图 4–7）；v1/v2/v3 diff |
| `schulman2017ppo` | Proximal Policy Optimization Algorithms | Schulman, Wolski, Dhariwal, Radford, Klimov | 1707.06347 **v2** | 预印本 | 2017-07-20（v1） | PDF 文本（无 LaTeXML/ar5iv 渲染） |
| `schulman2020kl` | Approximating KL Divergence（博客） | John Schulman | —（kind: artefact） | 博客 | 页面标注 “Posted on 2020/03/07” | 全文 http://joschu.net/blog/kl-approx.html |

**版本差异（DeepSeekMath）**：v1→v2 只改了表 5：新增 SeaLLM-v2 7B 一行（CoT：GSM8K 78.2%、MATH 27.5%、MGSM-zh 64.8%），
并把 MAmmoTH 70B 的工具推理成绩从 72.4% / 21.1% 改为 76.9% / 41.8%。v2→v3 的正文与附录 LaTeXML 文本**无差异**
（逐行 diff 为空）。GRPO 的公式、超参数、结论三个版本完全一致。**本笔记一律引用 v3。**

### 0.2 用于对照、纠错和“后来怎么变了”的来源

| 键 | 标题 | arXiv / 版本 | 状态 | 用途 |
|---|---|---|---|---|
| `deepseekai2025r1` | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | 2501.12948 **v2**（页眉 04 Jan 2026） | Nature 645:633–638（2025），doi 10.1038/s41586-025-09422-z（arXiv journal-ref） | R1 版 GRPO 公式（序列级）、超参数、GRPO vs PPO 附录 A.3 |
| `deepseekai2025v32` | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models | 2512.02556 **v1**（2025-12-02） | 预印本 | §3.1 “Scaling GRPO”：去掉 std 归一化、KL 估计的重要性修正 |
| `liu2025drgrpo` | Understanding R1-Zero-Like Training: A Critical Perspective（Dr. GRPO） | 2503.20783 **v2**（06 Oct 2025） | 预印本（会议未核实） | GRPO 的长度偏差与难度偏差；与 RLOO 的等价 |
| `tang2025kl` | On a few pitfalls in KL divergence gradient estimation for RL | 2506.09477 **v1**（2025-06-11） | 预印本 | 把 k3 当损失求导得到的不是 KL 的梯度 |
| `yue2025rlvr` | Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | 2504.13837 **v5** | NeurIPS 2025 Oral（arXiv journal-ref） | 把“RL 提升 Maj@K 不提升 Pass@K”推广到大 k |
| `wang2024mathshepherd` | Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations | 2312.08935 **v3**（19 Feb 2024） | ACL 2024（长文，aclanthology 2024.acl-long.510） | 过程奖励数据的构造方法（DeepSeekMath §4.2 所依） |
| `ouyang2022instructgpt` | Training language models to follow instructions with human feedback | 2203.02155 **v1** | （会议未在此核实） | 每 token KL 惩罚的出处（DeepSeekMath 式 (2) 引用它） |
| `rafailov2023dpo` | Direct Preference Optimization | 2305.18290 **v3**（29 Jul 2024） | — | 原版 DPO 损失 (7) 与梯度，对照 DeepSeekMath 附录的改写 |
| `guo2024deepseekcoder` | DeepSeek-Coder: When the Large Language Model Meets Programming | 2401.14196 **v2** | 预印本 | Coder-v1.5 的来历（DeepSeekMath 的初始化） |
| `paster2023openwebmath` | OpenWebMath | 2310.06786 **v1** | 预印本 | 种子语料；自报 14.7B tokens |
| `rajbhandari2019zero` | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | 1910.02054 **v3** | — | 混合精度 Adam 16Ψ 字节（显存算例用） |
| `joulin2016fasttext` | Bag of Tricks for Efficient Text Classification | 1607.01759 v3 | — | 只核对了元数据（fastText 出处） |
| `yuan2023rft` | Scaling Relationship on Learning Mathematical Reasoning with LLMs（RFT） | 2308.01825 v2 | 预印本 | 只核对了元数据 |
| `schulman2015gae` | High-Dimensional Continuous Control Using GAE | 1506.02438 v6 | — | 只核对了元数据 |

### 0.3 制品（kind: artefact，无论文）

| 键 | 内容 | 位置 / 版本 |
|---|---|---|
| `hf-dsmath-config` | `deepseek-math-7b-{base,instruct,rl}` 的 `config.json` | https://huggingface.co/deepseek-ai/deepseek-math-7b-base/raw/main/config.json 等，2026-09-24 读取 |
| `gh-deepseek-math` | DeepSeek-Math GitHub README 与仓库目录 | https://github.com/deepseek-ai/DeepSeek-Math （main，2026-09-24）——**仓库只有评测代码，没有 RL/GRPO 训练代码** |
| `trl-grpo` | Hugging Face TRL 的 `GRPOTrainer` | `trl/trainer/grpo_trainer.py` @ commit `c8ef2433`（2026-09-23），`trl/trainer/utils.py` @ `6c5f1350`，`grpo_config.py`，`VERSION = 1.14.0.dev0` |

---

## 1. 核心论断与原文引文

编号 C1–C40。“位置”一律指 DeepSeekMath v3，除非另注。

### 1.1 数学语料（§2.1–§2.2）

**C1 种子与分类器。** 种子语料是 OpenWebMath；从中随机取 500,000 条作正例，从 Common Crawl 取 500,000 个网页作负例，
训练 fastText 分类器；fastText 配置：向量维度 256、学习率 0.1、词 n-gram 最大长度 3、最小词频 3、训练 3 个 epoch。
——位置：§2.1 第 2 段（脚注 1 指向 https://fasttext.cc）。

**C2 去重后的 Common Crawl 是 40B 个 HTML 网页（不是 tokens）。** 先用基于 URL 的去重和近似去重缩小 Common Crawl，
得到 “40B HTML web pages”；再用 fastText 召回，按分数排序只保留高分页。保留多少由预训练实验决定（试了 top
40B、80B、120B、160B tokens），**第一轮保留 top 40B tokens**。——§2.1 第 2 段。注意同一段里有两个“40B”，单位不同（见 §7 E7）。

**C3 按域名扩种子。** 把整个 Common Crawl 按 base URL 分成互不相交的域名；第一轮中被收集比例超过 10% 的域名判为数学相关
（例：mathoverflow.net）；人工标注这些域名里与数学相关的 URL 路径（例：mathoverflow.net/questions）；这些路径下尚未被收集的网页
加入种子，再训练更好的 fastText。——§2.1 第 3 段；图 2 把流程画成 4 步循环（见 §5）。

**C4 四轮迭代，35.5M 网页，120B tokens。** 四轮后得到 35.5M 个数学网页、共 120B tokens；第四轮发现
“nearly 98% of the data has already been collected in the third iteration”，于是停止。——§2.1 第 3 段末。
（表 1 用 DeepSeek 分词器统计的大小是 **120.2B**。）

**C5 去污染。** 任何包含与评测集子串完全相同的 10-gram 的文本段都删掉；评测文本短于 10-gram 但至少 3-gram 的，用精确匹配过滤。
覆盖 GSM8K、MATH、CMATH、AGIEval 等。——§2.1 第 4 段。

**C6 语料质量验证实验。** 用与 DeepSeek LLM 同框架的 1.3B 通用模型（DeepSeek-LLM 1.3B），在每个数学语料上各训 150B tokens；
AdamW（$\beta_1=0.9,\beta_2=0.95$，weight decay 0.1）；多步学习率：2,000 步 warmup 到峰值，训练进度 80% 时降到峰值的 31.6%，
90% 时降到 10.0%；峰值学习率 5.3e-4；batch 4M tokens；上下文 4K。——§2.2.1。结果见表 1（数字在 §3.2）。

**C7 语料对比的解读。** DeepSeekMath Corpus 在 8 个基准上全面领先；图 3 显示在 50B tokens（Proof-Pile-2 的 1 个完整 epoch）时已胜过
Proof-Pile-2；基线语料因为太小被重复多轮而很快进入平台期。——§2.2.2。

### 1.2 DeepSeekMath-Base 7B（§2.3）

**C8 初始化与训练量。** 从 DeepSeek-Coder-Base-v1.5 7B 初始化，训练 500B tokens；数据配比：56% DeepSeekMath Corpus、
4% AlgebraicStack、10% arXiv、20% GitHub 代码、10% Common Crawl 自然语言（中英）。其余设置沿用 §2.2.1，只把峰值学习率改为
4.2e-4、batch 改为 10M tokens。——§2.3 第 1 段。

**C9 起点是“学习率衰减前”的检查点。** 表 4 的脚注说明带 † 的 DeepSeek-Coder-Base-v1.5 是
“the checkpoint right before learning rate decay”，DeepSeekMath-Base 就是从它继续训练的。——表 4 注。
（这正是第 6 课“多步学习率便于续训”的一个实际用例。）

**C10 为什么从代码模型出发。** 作者说从代码训练过的模型出发比从通用 LLM 出发更好；数学训练还提高了 MMLU 与 BBH。——§1 第 3 段；
表 4：MMLU 54.9% / BBH 59.5%，高于 Coder-Base-v1.5 的 49.1% / 55.2%。

**C11 Base 模型成绩（few-shot CoT，表 2）。** GSM8K 64.2%、MATH 36.2%、OCW 15.4%、SAT 84.4%、MMLU-STEM 56.5%、CMATH 71.7%、
Gaokao-MathCloze 20.3%、Gaokao-MathQA 35.3%。Minerva 540B：GSM8K 58.8%、MATH 33.6%。作者称 Minerva 540B 是
“77 times larger”（§2.3）。

**C12 Coder-v1.5 本身的来历（DeepSeek-Coder 论文）。** DeepSeek-Coder-v1.5 7B 是从 DeepSeek-LLM-7B Base 继续预训练得到的，
只用下一词预测、4K 上下文。继续训练量在 §5 写作 2 trillion tokens，而在 §6 结论里写作 2 billion tokens——**论文自相矛盾**
（见 §7 E16）。表 10 标注其参数量 6.9B。——`guo2024deepseekcoder` v2 §5、§6、表 10。

### 1.3 SFT（§3）

**C13 SFT 数据 776K 条。** 中英文数学题，配 CoT、PoT（program-of-thought）、工具集成推理三种格式的解答，总计 776K 条训练样本。
英文部分：给 GSM8K 与 MATH 题目标注工具集成解答，加上 MathInstruct 的子集与 Lila-OOD 训练集；中文部分：K-12 题目，覆盖 76 个子主题，
用 CoT 与工具集成两种格式标注。——§3.1。

**C14 SFT 训练设置。** 样本随机拼接到最长 4K tokens；500 步，batch 256，恒定学习率 5e-5。——§3.2 第 1 段。

### 1.4 GRPO 本体（§4.1）

**C15 PPO 的做法与负担。** DeepSeekMath 把 PPO 写成带 $1/|o|$ 的 token 平均形式（式 (1)），优势 $A_t$ 由 GAE 基于奖励与学到的
价值函数 $V_\psi$ 计算；为防奖励模型被过度优化，按 InstructGPT 的做法在**每个 token 的奖励里**减去 KL 惩罚（式 (2)）。价值函数通常与策略
同等规模，带来显著的显存与计算负担；而 LLM 场景里奖励通常只给最后一个 token，使得训练“每个 token 都准确”的价值函数更困难。——§4.1.1。

**C16 GRPO 的核心一句话。** “GRPO foregoes the critic model, instead estimating the baseline from group scores”（§1 第 5 段、§1.1、
图 4 标题均有同义表述）。基线换成同一问题下多条采样输出的平均奖励。——§4.1.1。

**C17 KL 放进损失而不是奖励。** 与 PPO 把 KL 加进每 token 奖励不同，GRPO “regularizes by directly adding the KL divergence”
（到损失里），从而不把 $\hat A_{i,t}$ 的计算弄复杂。——§4.1.1，式 (3) 下方。

**C18 KL 估计器与性质。** 作者称用的是 “the following unbiased estimator (Schulman, 2020)”，即式 (4)，并称它
“which is guaranteed to be positive”。——§4.1.1，式 (4) 前后。
（严格说是“非负”：$r=1$ 时等于 0；“无偏”只对估计 KL 的**值**且样本来自 $\pi_\theta$ 时成立；见 §2.6 与 §7 E1–E2。）

**C19 组相对优势与奖励模型的“比较”本性相合。** 作者的理由：奖励模型通常就是在“同一问题的不同输出之间的比较”上训练的，所以组内
相对的优势计算与之契合。——§4.1.1，式 (3) 下方。

**C20 结果监督（outcome supervision）。** 每个问题从 $\pi_{\theta_{old}}$ 采 $G$ 个输出，奖励模型打分得 $\mathbf r=\{r_1,\dots,r_G\}$，
减去组均值再除以组标准差得到归一化奖励 $\widetilde r_i$，并把它作为该输出**所有 token** 的优势。——§4.1.2。

**C21 过程监督（process supervision）。** 按 Wang et al.（Math-Shepherd）的做法，过程奖励模型在每个推理步的末尾打分；
把整组所有步的奖励 $\mathbf R$ 一起用均值、标准差归一化；每个 token 的优势是
“the sum of the normalized rewards from the following steps”（即结束位置 $\ge t$ 的各步归一化奖励之和）。——§4.1.3。

**C22 迭代 RL。** 随着训练推进，旧奖励模型可能不足以监督当前策略；于是用策略的采样结果生成奖励模型的新训练集，用回放机制持续训练奖励模型，
回放 “incorporates 10% of historical data”；然后把参考模型设为当前策略，用新奖励模型继续训练策略。——§4.1.4 与算法 1。

**C23 算法 1 的结构。** 外层 `iteration = 1..I`：$\pi_{ref}\leftarrow\pi_\theta$；中层 `step = 1..M`：采 batch、$\pi_{\theta_{old}}\leftarrow\pi_\theta$、
每题采 $G$ 个输出、奖励模型打分、算组相对优势；内层 `GRPO iteration = 1..μ`：最大化 GRPO 目标；每个外层迭代末尾用回放继续训练 $r_\varphi$。
——算法 1（§4.1.1 末）。注意算法第 11 行写 “maximizing the GRPO objective (Equation 21)”，而式 (21) 其实是梯度系数（§7 E9）。

### 1.5 RL 训练设置与结果（§4.2）

**C24 RL 数据。** 在 DeepSeekMath-Instruct 7B 上做 RL；训练数据是 SFT 数据中与 GSM8K、MATH 相关的 CoT 格式问题，约 144K 个；
其余 SFT 问题被**有意排除**，用来观察 RL 对“RL 阶段没见过数据”的基准的影响。——§4.2 第 1 段。

**C25 奖励模型。** 奖励模型训练集按 Wang et al.（2023b，Math-Shepherd）构造；初始奖励模型基于 DeepSeekMath-Base 7B，学习率 2e-5。——§4.2。

**C26 GRPO 超参数。** 策略学习率 1e-6；KL 系数 0.04；每个问题采 64 个输出；最大长度 1024；训练 batch size 1024；
“The policy model only has a single update following each exploration stage.”——§4.2 第 1 段。
（论文**没有给出** clip 系数 $\varepsilon$、采样温度、RL 总步数；见 §8。）

**C27 主结果（表 5，CoT，不用工具）。** DeepSeekMath-RL 7B：GSM8K 88.2%、MATH 51.7%、MGSM-zh 79.6%、CMATH 88.8%；
DeepSeekMath-Instruct 7B：82.9%、46.8%、73.2%、84.6%。工具集成推理：RL 86.7% / 58.8% / 78.4% / 87.6%；Instruct 83.7% / 57.4% / 72.0% / 84.3%。
正文 §4.2 与 §1 都复述 88.2% / 51.7%。表注：“Scores in gray denote majority votes with 32 candidates”，其余为 Top1。
经检查 HTML 的颜色标记（`#808080`），表 5 中灰色的只有 Gemini Ultra 的 GSM8K 94.4% 和 Gemini Pro 的 GSM8K 86.5%。

**C28 自洽投票。** 摘要：MATH 上 51.7%（不用外部工具、不投票）；对 64 个样本做 self-consistency 达 60.9%。
（与图 7 数字化得到的 MATH Maj@64-RL ≈ 60.9 一致，见 §3.6。）

**C29 RL 只用 GSM8K/MATH 的 CoT 数据，却在所有指标上超过 Instruct。** ——§4.2 第 3 段结论 2)、表 5 标题。

### 1.6 讨论：预训练的教训（§5.1）

**C30 代码训练有益于数学推理。** 在 DeepSeek-LLM 1.3B 上：两阶段“400B 代码 → 150B 数学”得 GSM8K 21.9%、MATH 15.3%，优于
“400B 通用 → 150B 数学”的 19.1%、14.4%；一阶段“代码+数学混训”缓解了灾难性遗忘并提升工具推理（GSM8K+Python 19.7%、MATH+Python 13.5%），
但损害了无工具数学推理（17.6% / 12.1%）。——§5.1.1，表 6、表 7。§5 的实验用的是**第二轮**收集得到的 89B-token 语料（§5.1 开头）。

**C31 arXiv 论文似乎无效。** 只用 arXiv 语料（MathPile 8.9B、ArXiv-RedPajama 28.0B）训 1.3B（150B tokens）或 Coder-v1.5 7B（40B tokens），
各基准没有明显提升甚至下降（表 8、表 9）；作者列出三条局限（未研究定理非形式化等任务、与其他数据组合的效果、更大模型规模）。——§5.1.2。

### 1.7 讨论：强化学习的洞见（§5.2）

**C32 统一范式。** 各方法对参数的梯度都可写成式 (5)：数据源 $\mathcal D$、奖励函数 $\pi_{rf}$、算法 $\mathcal A$ 三个组件；算法把数据和
奖励信号加工成“梯度系数” GC，决定对数据的惩罚或强化幅度。表 10 汇总 SFT、RFT、DPO、Online RFT、PPO、GRPO。作者的结论：这些方法都可看作
“直接或简化的 RL”。——§5.2.1、§5.2.3、附录 A.1。

**C33 在线 vs 离线。** 在线采样（数据来自实时策略）与离线采样（数据来自初始 SFT 模型）；RFT、DPO 属离线，Online RFT、GRPO 属在线。
图 5（DeepSeekMath-Instruct **1.3B**）上 Online RFT 显著优于 RFT：早期相当，后期拉开。——§5.2.1 “Observation about Data Source”。

**C34 梯度系数的作用。** GRPO 依据奖励模型给出的数值调整梯度系数，可以按幅度差异化地强化与惩罚；Online RFT 则
“does not penalize incorrect responses”，并对所有正确回答一视同仁地强化。图 5 中 GRPO 胜过 Online RFT；GRPO+PS 胜过 GRPO+OS。
迭代 RL 做了两轮，第一轮提升最明显（图 6）。——§5.2.1 “Observation about Gradient Coefficient”。

**C35 为什么 RL 有效：Maj@K 升、Pass@K 不升。** 在 Instruct 与 RL 模型上测 GSM8K、MATH 的 Pass@K 与 Maj@K（温度 0.7）：
“RL enhances Maj@K's performance but not Pass@K”；作者的解释是提升来自
“boosting the correct response from TopK rather than the enhancement of fundamental capabilities”。——§5.2.2，图 7 标题同义。

**C36 作者对原因的推测与方向。** 作者认为只用指令微调阶段的问题、只用朴素的 nucleus 采样，可能是 RL 只提升 Maj@K 的原因；
提出三个方向：数据源（分布外问题、树搜索等采样）、算法（对噪声奖励鲁棒，weak-to-strong）、奖励函数（泛化、不确定性、高质量 PRM）。
引用 PRM800K 约 20% 标注错误（脚注 7 指向 GitHub issue）。——§5.2.3。

**C37 局限。** 几何与定理证明弱于闭源模型（dry run 中处理不了三角形、椭圆问题）；受规模所限 few-shot 能力不如 GPT-4。——§6。

### 1.8 对照来源的论断

**C38 PPO（`schulman2017ppo` v2）。** 裁剪目标 $L^{CLIP}$ 为式 (7)，文中举例 $\epsilon=0.2$；取 min 使最终目标是
“a lower bound (i.e., a pessimistic bound) on the unclipped objective”（§3）。§4 的自适应 KL 惩罚式 (8) 惩罚的是
$\mathrm{KL}[\pi_{\theta_{old}}(\cdot|s_t),\pi_\theta(\cdot|s_t)]$（**对旧策略**，不是对参考模型），规则：$d<d_{targ}/1.5$ 则 $\beta\leftarrow\beta/2$；
$d>d_{targ}\times1.5$ 则 $\beta\leftarrow\beta\times2$。表 1：不裁剪不惩罚 −0.39；裁剪 $\epsilon=0.1$ 0.76、**0.2 为 0.82（最好）**、0.3 为 0.70。
式 (9) 为 $L^{CLIP+VF+S}$，式 (11)(12) 为截断 GAE。

**C39 Schulman 的 KL 估计（`schulman2020kl`）。** 记号：$\mathrm{KL}[q,p]=\E_{x\sim q}[\log\tfrac{q(x)}{p(x)}]$，样本来自 $q$，$r=p(x)/q(x)$。
$k_1=-\log r$（无偏但方差大，一半样本为负）；$k_2=\tfrac12(\log r)^2$（有偏、方差低；其期望是一个 $f$-散度，$f''(1)=1$，与 KL 二阶相同）；
用控制变量 $r-1$（期望为 0）并取 $\lambda=1$，由 $\log x\le x-1$ 得恒非负的 $k_3=(r-1)-\log r$；它是 $\log$ 与其切线之间的竖直距离（Bregman 散度）。
另给出 $\mathrm{KL}[p,q]$ 的估计 $r\log r-(r-1)$。实验：$q=N(0,1),p=N(0.1,1)$，真 KL 0.005，bias/true：k1 0、k2 0.002、k3 0；
stdev/true：20、1.42、1.42；$p=N(1,1)$，真 KL 0.5：bias/true 0、0.25、0；stdev/true 2、1.73、1.7。原文结论：
“k3 has even lower standard deviation than k2 while being unbiased”。

**C40 后续修订（按时间）。**
- R1（`deepseekai2025r1` v2 §2.1 式 (1)–(3)、附录 A.3 式 (11)–(13)）把 GRPO 写成**序列级**：比值用 $\pi_\theta(o_i|q)/\pi_{\theta_{old}}(o_i|q)$，
  没有 $1/|o_i|$ 与 token 求和；优势 $A_i=(r_i-\mathrm{mean})/\mathrm{std}$。R1-Zero 的训练设置在 §2.1 末段，R1 第一阶段 RL 的设置（含
  clip 比 $\varepsilon=10$）在 §3.2.1，后者并说明 clip 比过低会截断大量 token 的梯度、过高会不稳定。附录 A.3：PPO 把 KL 作为每 token 的稠密奖励会惩罚累计 KL，
  “可能隐式地惩罚回答长度”；GRPO 训练中周期性地把参考策略更新为最新策略。用 DeepSeek-Coder-V2-Lite（16B MoE，2.4B 激活）对比：
  GAE 的 $\lambda=0.95$ 时 PPO 明显差于 GRPO，调到 $\lambda=1.0$ 后接近 GRPO。R1 作者贡献说明写明 Junxiao Song 提出了 GRPO 算法并实现初版。
- V3.2（`deepseekai2025v32` §3.1 式 (5)–(7)）：优势改为 $\hat A_{i,t}=R_i-\mathrm{mean}(\boldsymbol R)$（**不再除以 std**）；
  用重要性比 $\pi_\theta/\pi_{old}$ 乘在 k3 前面得到“无偏 KL 估计”，并指出原 k3 的梯度在 $\pi_\theta\ll\pi_{ref}$ 时
  “assigns disproportionately large, unbounded weights”；数学等领域用弱 KL 甚至不用 KL 效果更好。
- Dr. GRPO（`liu2025drgrpo` §3.1）：指出 “Response-level length bias”（来自 $1/|o_i|$）与 “Question-level difficulty bias”（来自除以 std）；
  §3.2 提议两者都去掉；附录 A 证明去掉 std 后的 $\tilde A$ 乘以 $G/(G-1)$ 等于 RLOO 优势。
- Tang & Munos（`tang2025kl`）：“Differentiating KL estimates as loss functions does not usually produce the right estimate”（§2.2 末的 Key insight 框）；
  §3 说明把 k3（他们称 variance-reduced estimate）当损失求导，期望梯度是 $\nabla\mathrm{KL}(\pi_{ref},\pi)$（他们称 reversed KL）；
  §4.1：序列场景下 token 级损失至多给出部分梯度；而对 k3 这种 token 级损失，他们进一步指出它
  “does not produce even the partial gradient of any existing sequence-level divergence”（§4.1 末）。
- Yue et al.（`yue2025rlvr` 图 1 说明右半）：随着 RLVR 训练推进，pass@1 上升，但
  “the coverage of solvable problems (i.e., pass@256) decreases”。

---

## 2. 公式（原文转录 + 课程记号对照）

转录时删去了排版用的 `\footnotesize`、`\small`、`\begin{split}` 与 `\color` 命令，其余照 LaTeXML 源码。

### 2.1 PPO（DeepSeekMath 式 (1)(2)）

$$
\mathcal{J}_{PPO}(\theta)=\mathbb{E}{[q\sim P(Q),o\sim\pi_{\theta_{old}}(O|q)]}\frac{1}{|o|}\sum_{t=1}^{|o|}\min\left[\frac{\pi_{\theta}(o_{t}|q,o_{<t})}{\pi_{\theta_{old}}(o_{t}|q,o_{<t})}A_{t},\text{clip}\left(\frac{\pi_{\theta}(o_{t}|q,o_{<t})}{\pi_{\theta_{old}}(o_{t}|q,o_{<t})},1-\varepsilon,1+\varepsilon\right)A_{t}\right]\tag{1}
$$

$$
r_{t}=r_{\varphi}(q,o_{\leq t})-\beta\log\frac{\pi_{\theta}(o_{t}|q,o_{<t})}{\pi_{ref}(o_{t}|q,o_{<t})}\tag{2}
$$

对照原版 PPO（`schulman2017ppo` v2 §3 式 (6)(7)）：$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，
$L^{CLIP}(\theta)=\hat{\mathbb E}_t\big[\min(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t)\big]$。
原版**没有** $1/|o|$；这个长度归一化是 DeepSeekMath（以及许多开源实现）写法里加进去的（Dr. GRPO §3.1 与表 2 专门指出这点）。

对照 InstructGPT（`ouyang2022instructgpt` v1 §3.5 式 (2)）：
$\mathrm{objective}(\phi)=\mathbb E_{(x,y)\sim D_{\pi^{RL}_\phi}}\big[r_\theta(x,y)-\beta\log(\pi^{RL}_\phi(y|x)/\pi^{SFT}(y|x))\big]+\gamma\,\mathbb E_{x\sim D_{pretrain}}[\log\pi^{RL}_\phi(x)]$；
公式是序列级写法，正文说实现为 “per-token KL penalty from the SFT model at each token”。DeepSeekMath 式 (2) 是它的每 token 版本。

### 2.2 GRPO 目标（式 (3)）与 KL 估计（式 (4)）

$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}{[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O|q)]}
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\left\{\min\left[\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}\hat{A}_{i,t},\text{clip}\left(\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})},1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right]-\beta\mathbb{D}_{KL}\left[\pi_{\theta}||\pi_{ref}\right]\right\}\tag{3}
$$

$$
\mathbb{D}_{KL}\left[\pi_{\theta}||\pi_{ref}\right]=\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-\log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-1\tag{4}
$$

**课程记号版**（建议写法；$\rho$ 与 $w$ 两个符号 style-guide 尚未定义，见本节末“记号冲突”）：

$$
\mathcal J_{GRPO}(\theta)=\E_{q\sim P(Q),\,\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{old}}(\cdot\mid q)}
\left[\frac1G\sum_{i=1}^{G}\frac1{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(\rho_{i,t}\hat A_{i,t},\ \clip(\rho_{i,t},1-\varepsilon,1+\varepsilon)\,\hat A_{i,t}\big)-\beta\,\widehat{\KL}_{i,t}\Big)\right],
$$
$$
\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})},\qquad
w_{i,t}=\frac{\pi_{ref}(o_{i,t}\mid q,o_{i,<t})}{\pi_\theta(o_{i,t}\mid q,o_{i,<t})},\qquad
\widehat{\KL}_{i,t}=w_{i,t}-\log w_{i,t}-1 .
$$

读法要点：
- 式 (4) 左边虽写作 $\mathbb D_{KL}[\pi_\theta\|\pi_{ref}]$，右边却是**单个 token 的估计量**，依赖 $i,t$。课程里写成 $\widehat{\KL}_{i,t}$，避免读者以为是精确 KL。
- 三重平均：组内平均 $\frac1G\sum_i$、序列内 token 平均 $\frac1{|o_i|}\sum_t$、问题分布上的期望。
- $\hat A_{i,t}$ 在结果监督下与 $t$ 无关（整条输出共享），在过程监督下随 $t$ 变化。

### 2.3 优势估计（§4.1.2、§4.1.3，无编号）

结果监督：
$$
\hat{A}_{i,t}=\widetilde{r}_{i}=\frac{r_{i}-{\rm mean}(\mathbf{r})}{{\rm std}(\mathbf{r})}
$$

过程监督：$\mathbf{R}=\{\{r_{1}^{index(1)},\cdots,r_{1}^{index(K_{1})}\},\cdots,\{r_{G}^{index(1)},\cdots,r_{G}^{index(K_{G})}\}\}$，
$index(j)$ 是第 $j$ 步的结束 token 下标，$K_i$ 是第 $i$ 个输出的步数；
$$
\widetilde{r}_{i}^{index(j)}=\frac{r_{i}^{index(j)}-{\rm mean(\mathbf{R})}}{{\rm std(\mathbf{R})}},\qquad
\hat{A}_{i,t}=\sum_{index(j)\geq t}\widetilde{r}_{i}^{index(j)}
$$

**std 的约定**：论文只写 ${\rm std}(\mathbf r)$，未说总体（除以 $G$）还是样本（除以 $G-1$）标准差；DeepSeek 没有公开 GRPO 训练代码
（`gh-deepseek-math` 仓库只有 `evaluation/`、`replicate/`）。TRL 用的是**样本标准差**（Bessel 修正）再加 $10^{-4}$：
`nanstd` 里 `correction = count / (count - 1)`，`advantages = advantages / (std_rewards + 1e-4)`（`trl/trainer/utils.py`、
`grpo_trainer.py` 约第 2823–2840 行，commit 见 §0.3）。两种约定下优势相差常数因子 $\sqrt{(G-1)/G}$：$G=4$ 时 0.866，$G=64$ 时 0.992（§4.1）。
课程写作建议：讲原理时用总体标准差（结果干净：二值奖励时 $\hat A=\pm1$），并脚注说明 TRL 用样本标准差。

### 2.4 统一范式（式 (5)）与附录 A.1 的逐项梯度

$$
\nabla_{\theta}\mathcal{J}_{\mathcal{A}}(\theta)=\mathbb{E}[\underbrace{(q,o)\sim\mathcal{D}}_{Data\ Source}]\left(\frac{1}{|o|}\sum_{t=1}^{|o|}\underbrace{GC_{\mathcal{A}}(q,o,t,\pi_{rf})}_{Gradient\ Coefficient}\nabla_{\theta}\log\pi_{\theta}(o_{t}|q,o_{<t})\right)\tag{5}
$$

**注意 $\pi_{rf}$ 是“奖励函数”（reward function），不是参考模型 $\pi_{ref}$。** 课程里建议写成 $\mathcal R$ 或文字“奖励函数”，避免混淆。

SFT：
$$
\mathcal{J}_{SFT}(\theta)=\mathbb{E}[q,o\sim P_{sft}(Q,O)]\left(\frac{1}{|o|}\sum_{t=1}^{|o|}\log\pi_{\theta}(o_{t}|q,o_{<t})\right)\tag{6}
$$
$$
\nabla_{\theta}\mathcal{J}_{SFT}=\mathbb{E}[q,o\sim P_{sft}(Q,O)]\left(\frac{1}{|o|}\sum_{t=1}^{|o|}\nabla_{\theta}\log\pi_{\theta}(o_{t}|q,o_{<t})\right)\tag{7}
$$
数据源：SFT 数据集；奖励函数：可视为人工挑选；梯度系数：恒为 1。

RFT：
$$
\mathcal{J}_{RFT}(\theta)=\mathbb{E}[q\sim P_{sft}(Q),o\sim\pi_{sft}(O|q)]\left(\frac{1}{|o|}\sum_{t=1}^{|o|}\mathbb{I}(o)\log\pi_{\theta}(o_{t}|q,o_{<t})\right)\tag{8}
$$
$$
\nabla_{\theta}\mathcal{J}_{RFT}(\theta)=\mathbb{E}[{q\sim P_{sft}(Q),o\sim\pi_{sft}(O|q)}]\left(\frac{1}{|o|}\sum_{t=1}^{|o|}{\mathbb{I}(o)}\nabla_{\theta}\log\pi_{\theta}(o_{t}|q,o_{<t})\right)\tag{9}
$$
$$
GC_{RFT}(q,o,t)=\mathbb{I}(o)=\begin{cases}1&\text{the answer of o is correct}\\0&\text{the answer of o is incorrect}\end{cases}\tag{10}
$$

Online RFT（与 RFT 唯一差别：$o\sim\pi_\theta$ 而非 $\pi_{sft}$）：
$$
\nabla_{\theta}\mathcal{J}_{OnRFT}(\theta)=\mathbb{E}[{q\sim P_{sft}(Q),o\sim\pi_{\theta}(O|q)}]\left(\frac{1}{|o|}\sum_{t=1}^{|o|}{\mathbb{I}(o)}\nabla_{\theta}\log\pi_{\theta}(o_{t}|q,o_{<t})\right)\tag{11}
$$

DPO（**照原样转录，含原文笔误**：负例项分子分母都写成了 $o^-_{<t}$，应为 $o^-_t$）：
$$
\mathcal{J}_{DPO}(\theta)=\mathbb{E}{[q\sim P_{sft}(Q),o^{+},o^{-}\sim\pi_{sft}(O|q)]}\log\sigma\left(\beta\frac{1}{|o^{+}|}\sum_{t=1}^{|o^{+}|}\log\frac{\pi_{\theta}(o^{+}_{t}|q,o^{+}_{<t})}{\pi_{\text{ref}}(o^{+}_{t}|q,o^{+}_{<t})}-\beta\frac{1}{|o^{-}|}\sum_{t=1}^{|o^{-}|}\log\frac{\pi_{\theta}(o^{-}_{<t}|q,o^{-}_{<t})}{\pi_{\text{ref}}(o^{-}_{<t}|q,o^{-}_{<t})}\right)\tag{12}
$$
$$
\nabla_{\theta}\mathcal{J}_{DPO}(\theta)=\mathbb{E}{[q\sim P_{sft}(Q),o^{+},o^{-}\sim\pi_{sft}(O|q)]}\left(\frac{1}{|o^{+}|}\sum_{t=1}^{|o^{+}|}GC_{DPO}(q,o,t)\nabla_{\theta}\log\pi_{\theta}(o^{+}_{t}|q,o^{+}_{<t})-\frac{1}{|o^{-}|}\sum_{t=1}^{|o^{-}|}GC_{DPO}(q,o,t)\nabla_{\theta}\log\pi_{\theta}(o^{-}_{t}|q,o^{-}_{<t})\right)\tag{13}
$$
$$
GC_{DPO}(q,o,t)=\sigma\left(\beta\log\frac{\pi_{\theta}(o^{-}_{t}|q,o^{-}_{<t})}{\pi_{\text{ref}}(o^{-}_{t}|q,o^{-}_{<t})}-\beta\log\frac{\pi_{\theta}(o^{+}_{t}|q,o^{+}_{<t})}{\pi_{\text{ref}}(o^{+}_{t}|q,o^{+}_{<t})}\right)\tag{14}
$$
与原版 DPO 的差别（`rafailov2023dpo` v3 §4 式 (7) 及其梯度）：原版用**整句**对数概率比（对 token 求和、不除以长度）：
$\mathcal L_{DPO}=-\mathbb E\big[\log\sigma\big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\big)\big]$，
梯度前有因子 $\beta$，权重 $\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))$ 对整句相同。DeepSeekMath 式 (12) 是**长度归一化**版本；式 (14) 的 GC 写成依赖 $t$ 的逐 token 比值，
但从式 (12) 求导得到的系数应是整句（长度平均后）比值的函数、对所有 $t$ 相同，且应带因子 $\beta$（可并入学习率）。课程里讲 DPO 的 GC 时以原版为准，
并说明 DeepSeekMath 的写法是简化示意。

PPO（附录 A.1.5）：
$$
\mathcal{J}_{PPO}(\theta)=\mathbb{E}{[q\sim P_{sft}(Q),o\sim\pi_{\theta_{old}}(O|q)]}\frac{1}{|o|}\sum_{t=1}^{|o|}\min\left[\frac{\pi_{\theta}(o_{t}|q,o_{<t})}{\pi_{\theta_{old}}(o_{t}|q,o_{<t})}A_{t},\text{clip}\left(\frac{\pi_{\theta}(o_{t}|q,o_{<t})}{\pi_{\theta_{old}}(o_{t}|q,o_{<t})},1-\varepsilon,1+\varepsilon\right)A_{t}\right]\tag{15}
$$
假设每次探索后只更新一次，从而 $\pi_{\theta_{old}}=\pi_\theta$，去掉 min 与 clip：
$$
\mathcal{J}_{PPO}(\theta)=\mathbb{E}{[q\sim P_{sft}(Q),o\sim\pi_{\theta_{old}}(O|q)]}\frac{1}{|o|}\sum_{t=1}^{|o|}\frac{\pi_{\theta}(o_{t}|q,o_{<t})}{\pi_{\theta_{old}}(o_{t}|q,o_{<t})}A_{t}\tag{16}
$$
$$
\nabla_{\theta}\mathcal{J}_{PPO}(\theta)=\mathbb{E}{[q\sim P_{sft}(Q),o\sim\pi_{\theta_{old}}(O|q)]}\frac{1}{|o|}\sum_{t=1}^{|o|}A_{t}\nabla_{\theta}\log\pi_{\theta}(o_{t}|q,o_{<t})\tag{17}
$$
$$
GC_{PPO}(q,o,t,\pi_{\theta_{rm}})=A_{t}\tag{18}
$$

GRPO（附录 A.1.6，同样假设 $\pi_{\theta_{old}}=\pi_\theta$）：
$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}{[q\sim P_{sft}(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O|q)]}\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\left[\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}\hat{A}_{i,t}-\beta(\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-\log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-1)\right]\tag{19}
$$
$$
\nabla_{\theta}\mathcal{J}_{GRPO}(\theta)=\mathbb{E}{[q\sim P_{sft}(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O|q)]}\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\left[\hat{A}_{i,t}+\beta\left(\frac{\pi_{ref}(o_{i,t}|o_{i,<t})}{\pi_{\theta}(o_{i,t}|o_{i,<t})}-1\right)\right]\nabla_{\theta}\log\pi_{\theta}(o_{i,t}|q,o_{i,<t})\tag{20}
$$
$$
GC_{GRPO}(q,o,t,\pi_{\theta_{rm}})=\hat{A}_{i,t}+\beta\left(\frac{\pi_{ref}(o_{i,t}|o_{i,<t})}{\pi_{\theta}(o_{i,t}|o_{i,<t})}-1\right)\tag{21}
$$
（式 (20)(21) 里 $\pi_{ref}(o_{i,t}|o_{i,<t})$ 省略了条件中的 $q$，是排版省略，不影响含义。）

**表 10（照录）**

| Methods | Data Source | Reward Function | Gradient Coefficient |
|---|---|---|---|
| SFT | $q,o\sim P_{sft}(Q,O)$ | - | 1 |
| RFT | $q\sim P_{sft}(Q)$, $o\sim\pi_{sft}(O\mid q)$ | Rule | Equation 10 |
| DPO | $q\sim P_{sft}(Q)$, $o^{+},o^{-}\sim\pi_{sft}(O\mid q)$ | Rule | Equation 14 |
| Online RFT | $q\sim P_{sft}(Q)$, $o\sim\pi_{\theta}(O\mid q)$ | Rule | Equation 10 |
| PPO | $q\sim P_{sft}(Q)$, $o\sim\pi_{\theta}(O\mid q)$ | Model | Equation 18 |
| GRPO | $q\sim P_{sft}(Q)$, $\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta}(O\mid q)$ | Model | Equation 21 |

### 2.5 逐项推导（供 `::: deriv` 使用；均已数值验证，见 §4.4）

**(a) 式 (16)→(17)：比值项的梯度。** 在 $\theta=\theta_{old}$ 处，$\nabla_\theta\frac{\pi_\theta}{\pi_{\theta_{old}}}=\frac{\pi_\theta}{\pi_{\theta_{old}}}\nabla_\theta\log\pi_\theta=\nabla_\theta\log\pi_\theta$
（对数导数技巧，$\nabla\pi=\pi\nabla\log\pi$；比值在该点等于 1）。所以 $\nabla(\rho A)=A\,\nabla\log\pi_\theta$。

**(b) 为什么 DeepSeekMath 的实际训练里 clip 不起作用。** C26 说每次探索后策略只更新一次。于是这唯一一次梯度步发生在 $\theta=\theta_{old}$，
$\rho_{i,t}=1\in[1-\varepsilon,1+\varepsilon]$，min 两边相等且梯度相同，**clip 对梯度没有任何影响**；GRPO 在该设置下退化为带组基线的
REINFORCE + KL 正则。$\varepsilon$ 只有在一批数据上做多步更新（$\mu>1$ 或多个 mini-batch）时才生效——R1-Zero 与 R1 就是这样（每次 rollout 的
8,192 个输出分成 16 个 minibatch，见 §3.5），R1 第一阶段 RL 并把 $\varepsilon$ 设为 10（R1 v2 §3.2.1）。
（核查补记：这条推论成立的前提是把 §4.2 的 “a single update following each exploration stage” 读成“对整批 rollout 只做一次优化器步”。
附录 A.1.5 正是这样读的（“thereby ensuring that $\pi_{\theta_{old}}=\pi_\theta$”），但 A.1.5 同时说这是 “To simplify the analysis” 的假设。
如果实际是“一个 epoch、分多个 minibatch”（R1 的做法），第二个 minibatch 起 $\rho\ne1$，clip 就会起作用。论文没给出 minibatch 数，见 §8。
另外，若采样引擎与训练框架的数值不一致，$\rho$ 在数值上也不严格等于 1。）

**(c) 式 (19)→(20)：KL 项的梯度。** 记 $w=\pi_{ref}/\pi_\theta$，则 $\nabla w=-w\,\nabla\log\pi_\theta$，$\nabla(-\log w)=\nabla\log\pi_\theta$。
$$
\nabla_\theta\big[-\beta(w-\log w-1)\big]=-\beta\big(-w+1\big)\nabla\log\pi_\theta=\beta\,(w-1)\,\nabla\log\pi_\theta .
$$
与 $\hat A_{i,t}$ 的梯度 $\hat A_{i,t}\nabla\log\pi_\theta$ 相加，得式 (21) 的 $GC=\hat A_{i,t}+\beta(w-1)$。
直觉：$w>1$（当前策略给这个 token 的概率低于参考模型）时系数为正，把它往上拉；$w<1$ 时往下压——拉回参考模型。

**(d) k3 非负。** 由 $\log x\le x-1$（$\log$ 凹，切线在上方，等号仅当 $x=1$）得 $w-1-\log w\ge0$，等号当且仅当 $w=1$。所以是“非负”，不是严格“正”。

**(e) k3 对 KL **值** 无偏（单步、样本来自 $\pi_\theta$）。** $\E_{x\sim\pi_\theta}[w]=\sum_x\pi_\theta(x)\frac{\pi_{ref}(x)}{\pi_\theta(x)}=\sum_x\pi_{ref}(x)=1$
（要求 $\pi_\theta(x)>0$ 处覆盖 $\pi_{ref}$ 的支撑；softmax 输出处处为正，满足）。所以
$\E[w-1-\log w]=\E[-\log w]=\E_{\pi_\theta}[\log\frac{\pi_\theta}{\pi_{ref}}]=\KL[\pi_\theta\|\pi_{ref}]$。
序列情形：对每个位置取条件分布上的 KL，再沿 $\pi_\theta$ 生成的前缀求和，就是序列级 KL（链式法则）；但式 (3) 里的 $1/|o_i|$ 权重、以及样本实际来自 $\pi_{\theta_{old}}$，都会让“无偏”打折扣。

**(f) k3 当损失求导，期望梯度不是 $\nabla\KL[\pi_\theta\|\pi_{ref}]$。** 单步情形：
$$
\E_{x\sim\pi_\theta}\big[\nabla_\theta(w-1-\log w)\big]=\sum_x\pi_\theta(x)(1-w)\nabla\log\pi_\theta(x)
=\underbrace{\sum_x\nabla\pi_\theta(x)}_{=0}-\sum_x\pi_{ref}(x)\nabla\log\pi_\theta(x)=\nabla_\theta\KL[\pi_{ref}\|\pi_\theta].
$$
即它是**另一个方向的 KL** 的梯度（`tang2025kl` §3 称之为 reversed KL；**命名方向因文献而异**——例如 TRL 的 `use_bias_correction_kl` 说明把修正后的 $\nabla\KL[\pi_\theta\|\pi_{ref}]$ 叫 “reverse-KL gradient”，与 Tang & Munos 的用法正好相反；课程里直接写 $\KL[\pi_{ref}\|\pi_\theta]$，不写“正向/反向”）。
在 $\pi_\theta\approx\pi_{ref}$ 时两个方向的 KL 二阶相同（Schulman 博客的 $f''(1)=1$ 论证），所以小 KL 区间差别不大（`tang2025kl` §5.1 也这样推测）。

**(g) V3.2 的修正（`deepseekai2025v32` 式 (7)）。** 把 k3 乘上 $\rho=\pi_\theta/\pi_{old}$：$\rho\,(w-\log w-1)$。在 $\theta=\theta_{old}$ 处求导：
$$
\nabla\big[\rho(w-\log w-1)\big]=\rho\big[(w-\log w-1)+(1-w)\big]\nabla\log\pi_\theta=\rho\,\log\frac{\pi_\theta}{\pi_{ref}}\,\nabla\log\pi_\theta ,
$$
其在 $\pi_{old}$ 下的期望是 $\sum_x\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}\nabla\log\pi_\theta=\nabla\KL[\pi_\theta\|\pi_{ref}]$（单步）。对应到 GC：原 k3 的 KL 系数 $\beta(w-1)$
随 $w$ **线性**增长（$\pi_\theta\ll\pi_{ref}$ 时无界），修正后是 $\beta\log w$，只按**对数**增长——这就是 V3.2 说的 “unbounded weights” 问题（数字见 §4.3）。

**(h) 组均值基线与 RLOO（`liu2025drgrpo` 附录 A）。** $\frac{G}{G-1}(r_i-\bar r)=r_i-\frac1{G-1}\sum_{j\ne i}r_j$。
右边是留一法（leave-one-out）基线，与 $o_i$ 独立，所以不引入偏差；左边多出的常数因子可并入学习率。**除以组内 std 会破坏这一点**（std 依赖 $r_i$ 本身），
Dr. GRPO 据此称 std 归一化带来难度偏差。

### 2.6 Schulman 博客的三个估计器（`schulman2020kl`）与 GRPO 的对应

博客记号：估计 $\mathrm{KL}[q,p]=\E_{x\sim q}[\log\frac{q(x)}{p(x)}]$，样本 $x\sim q$，$r=\frac{p(x)}{q(x)}$：
$$
k_1=-\log r,\qquad k_2=\tfrac12(\log r)^2,\qquad k_3=(r-1)-\log r .
$$
另：$\mathrm{KL}[p,q]$ 的非负估计 $r\log r-(r-1)$；一般 $f$-散度的非负估计 $f(r)-f'(1)(r-1)$。

**对应关系（务必在课里写明）**：博客的 $q$ ≡ 采样分布 ≡ $\pi_\theta$；博客的 $p$ ≡ $\pi_{ref}$；博客的 $r$ ≡ 课程的 $w=\pi_{ref}/\pi_\theta$。
于是 GRPO 式 (4) 恰是 $k_3$，估计的是 $\KL[\pi_\theta\|\pi_{ref}]$。而 PPO 式 (2) 每 token 奖励里的 $\log\frac{\pi_\theta}{\pi_{ref}}$ 恰是 $k_1$。
**记号冲突**：博客的 $q$（分布）与课程的 $q$（问题/prompt）撞车；博客的 $r$（比值）与课程的 $r$（奖励）、PPO 的 $r_t(\theta)$（比值）三者撞车。

### 2.7 后续论文的公式（用于“后来怎么变了”）

R1 v2 式 (1)–(3)（序列级写法）：
$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}{[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O|q)]}\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)}A_{i},\text{clip}\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)},1-\varepsilon,1+\varepsilon\right)A_{i}\right)-\beta\mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)\right)
$$
$$
\mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)=\frac{\pi_{ref}(o_{i}|q)}{\pi_{\theta}(o_{i}|q)}-\log\frac{\pi_{ref}(o_{i}|q)}{\pi_{\theta}(o_{i}|q)}-1,\qquad
A_{i}=\frac{r_{i}-{\mathrm{mean}(\{r_{1},r_{2},\cdots,r_{G}\})}}{{\mathrm{std}(\{r_{1},r_{2},\cdots,r_{G}\})}}
$$

V3.2 式 (5)–(7)（token 级、无 std、带重要性修正的 KL）：
$$
\hat{A}_{i,t}=R_{i}-\text{mean}(\boldsymbol{R}),\qquad
\mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}(o_{i,t})\,\middle\|\,\pi_{\mathrm{ref}}(o_{i,t})\right)=\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\mathrm{old}}(o_{i,t}|q,o_{i,<t})}\left(\frac{\pi_{\mathrm{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-\log\frac{\pi_{\mathrm{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-1\right)
$$

Dr. GRPO（`liu2025drgrpo` 图 1 / 附录 A）：从式 (3) 中划去 $\frac1{|o_i|}$ 与分母 ${\rm std}$；实现上把 `masked_mean` 里的 `mask.sum` 换成常数（如生成预算）。

Math-Shepherd 步骤标签（`wang2024mathshepherd` §3.3 式 (3)(4)）：用 completer 从第 $i$ 步续写 $N$ 条完整解答，
$y_{s_i}^{HE}=1$ 若存在 $a_j=a^*$，否则 0；$y_{s_i}^{SE}=\frac{\sum_{j=1}^{N}\mathbb{I}(a_j=a^*)}{N}$；论文训练 PRM 时用 HE 版本（§4 实验设置）。

### 2.8 记号对照与冲突清单

| 论文记号 | 含义 | 课程记号（style-guide §2） | 备注 |
|---|---|---|---|
| $\pi_\theta,\pi_{\theta_{old}},\pi_{ref}$ | 当前/采样用旧/参考策略 | 同 | ✓ |
| $q,o,o_{i,t},o_{i,<t}$ | 问题、回答、第 $i$ 个回答第 $t$ 个 token、其前缀 | 同 | ✓ |
| $G$ | 组大小 | $G$ | ✓ |
| $r_i$, $\mathbf r$ | 奖励、组内奖励向量 | $r$；向量用粗体小写 | ✓ |
| $\widetilde r_i$ | 归一化奖励 | 新符号，建议保留 | style-guide 未列 |
| $\hat A_{i,t}$ | 优势 | $\hat A$ | ✓ |
| $\varepsilon,\beta$ | 裁剪系数、KL 系数 | 同 | ✓ |
| $\mathbb D_{KL}[\pi_\theta\Vert \pi_{ref}]$ | 式 (4) 的逐 token 估计 | `\KL`；建议写 $\widehat{\KL}_{i,t}$ | 区分“真 KL”与“估计量” |
| 比值 $\pi_\theta/\pi_{\theta_{old}}$ | 重要性比 | **未定义**，建议 $\rho_{i,t}$ | PPO 原文用 $r_t(\theta)$，与奖励 $r$ 撞车 |
| $\pi_{ref}/\pi_\theta$ | k3 里的比值 | **未定义**，建议 $w_{i,t}$ | Schulman 用 $r$，又撞车 |
| $\pi_{rf}$（式 5） | 奖励函数 | 建议文字“奖励函数”或 $\mathcal R$ | 极易误读成 $\pi_{ref}$ |
| $\pi_{\theta_{rm}}$（式 18、21） | 奖励模型 | $r_\varphi$ | 论文内部两套写法 |
| $\mathbf R$（§4.1.3） | 整组过程奖励的集合 | 建议 $\mathcal R^{proc}$ 或文字 | style-guide 里大写粗体是矩阵 |
| $K_i$ | 第 $i$ 个输出的步数 | 本课局部 | 与 MoE 的 $K_r$ 不冲突，但需说明 |
| $I, M, \mu$（算法 1） | 外层迭代数、每迭代步数、每批内更新次数 | 建议 $I$、$S$、$\mu$ | **$M$ 与课程 $M$（每 token 非嵌入 FLOPs）冲突** |
| $\mathbb I(o)$ | 答案是否正确的指示函数 | $\mathbb I$ | ✓ |
| $GC_{\mathcal A}$ | 梯度系数 | $\mathrm{GC}$（course-map 已用） | ✓ |

---

## 3. 规格与超参数表

### 3.1 数据管线（§2.1）

| 项 | 值 | 来源 |
|---|---|---|
| 种子语料 | OpenWebMath | §2.1 |
| fastText 训练样本 | 正例 500,000（种子）+ 负例 500,000（Common Crawl） | §2.1 |
| fastText 超参 | 向量维度 256；lr 0.1；词 n-gram 最大长度 3；最小词频 3；epoch 3 | §2.1 |
| 去重后 Common Crawl | 40B 个 HTML 网页（URL 去重 + 近似去重） | §2.1、图 2 |
| 保留量试验 | top 40B / 80B / 120B / 160B tokens | §2.1 |
| 第一轮保留 | top 40B tokens | §2.1 |
| 数学域名判据 | 该域名 >10% 的网页已被收集 | §2.1 |
| 迭代轮数 | 4（第 4 轮约 98% 数据已在第 3 轮收集，停止） | §2.1 |
| 最终规模 | 35.5M 网页；120B tokens（表 1：120.2B，DeepSeek 分词器，词表 100K） | §2.1；表 1 |
| 去污染 | 10-gram 精确匹配；短文本（≥3-gram、<10-gram）精确匹配 | §2.1 |
| §5 消融所用语料 | 第 2 轮得到的 89B-token 版本 | §5.1 |

### 3.2 语料对比实验（表 1，DeepSeek-LLM 1.3B，各训 150B tokens，few-shot CoT）

| 语料 | 大小 | GSM8K | MATH | OCW | SAT | MMLU-STEM | CMATH | Gaokao MathCloze | Gaokao MathQA |
|---|---|---|---|---|---|---|---|---|---|
| No Math Training | N/A | 2.9% | 3.0% | 2.9% | 15.6% | 19.5% | 12.3% | 0.8% | 17.9% |
| MathPile | 8.9B | 2.7% | 3.3% | 2.2% | 12.5% | 15.7% | 1.2% | 0.0% | 2.8% |
| OpenWebMath | 13.6B | 11.5% | 8.9% | 3.7% | 31.3% | 29.6% | 16.8% | 0.0% | 14.2% |
| Proof-Pile-2 | 51.9B | 14.3% | 11.2% | 3.7% | 43.8% | 29.2% | 19.9% | 5.1% | 11.7% |
| DeepSeekMath Corpus | 120.2B | 23.8% | 13.6% | 4.8% | 56.3% | 33.1% | 41.5% | 5.9% | 23.6% |

训练设置（§2.2.1）：AdamW $\beta_1=0.9,\beta_2=0.95$，wd 0.1；warmup 2,000 步；80% 处降到 31.6%，90% 处降到 10.0%；峰值 lr 5.3e-4；batch 4M tokens；上下文 4K。
Proof-Pile-2 按 arXiv:Web:Code = 2:4:1 混合（§2.2）；MathPile 85% 以上来自 arXiv；Proof-Pile-2 含 AlgebraicStack 10.3B 与 arXiv 28.0B。

### 3.3 DeepSeekMath-Base 7B：架构（config.json）与训练（§2.3）

| 项 | 论文 | config.json（base / instruct / rl 三者相同） | 备注 |
|---|---|---|---|
| 架构 | “与 DeepSeek LLM 同框架”（§2.2.1 对 1.3B 的描述）；初始化自 Coder-Base-v1.5 7B | `LlamaForCausalLM`，`model_type: llama` | |
| 层数 $l$ | — | 30 | |
| $d$ | — | 4096 | |
| $n_h$ / KV 头数 | — | 32 / 32（MHA，非 GQA） | |
| $d_h$ | — | 128（=4096/32，推算） | |
| $d_{ff}$ | — | 11008（SwiGLU，`hidden_act: silu`） | |
| 词表 $V$ | “vocabulary size of 100K”（表 1 注） | 102400 | 论文取整 |
| 上下文 | 4K | `max_position_embeddings: 4096` | |
| RoPE 基数 | — | 10000 | |
| 权重共享 | — | `tie_word_embeddings: false` | |
| RMSNorm eps | — | 1e-6 | |
| 精度 | — | bfloat16 | |
| 参数量 | “7B” | 6.910B（按 config 计算，§4.2） | DeepSeek-Coder 表 10 记 Coder-v1.5 为 6.9B |
| 训练 tokens | 500B | — | §2.3 |
| 数据配比 | 56% DSM Corpus / 4% AlgebraicStack / 10% arXiv / 20% GitHub 代码 / 10% CC 中英自然语言 | — | §2.3 |
| 峰值 lr / batch | 4.2e-4 / 10M tokens | — | §2.3；其余同 §2.2.1 |
| 起点 | Coder-Base-v1.5 “学习率衰减前”检查点 | — | 表 4 注 |

（`deepseek-math-7b-rl/config.json` 多出一个无用字段 `"moe_intermediate_size": 11008`，`rope_theta` 写成整数 10000；对模型无影响。）

### 3.4 SFT 与 DeepSeekMath-Instruct 7B（§3）

| 项 | 值 | 来源 |
|---|---|---|
| 样本数 | 776K | §3.1 |
| 格式 | CoT、PoT、工具集成推理 | §3.1 |
| 英文来源 | GSM8K/MATH（新标注工具集成解答）、MathInstruct 子集、Lila-OOD 训练集 | §3.1 |
| 中文来源 | K-12，76 个子主题，CoT + 工具集成 | §3.1 |
| 打包 | 随机拼接至最长 4K tokens | §3.2 |
| 步数 / batch / lr | 500 步 / 256 / 恒定 5e-5 | §3.2 |

### 3.5 RL（GRPO）超参数：DeepSeekMath vs R1 vs V3.2 vs TRL 默认

| 项 | DeepSeekMath（§4.2） | DeepSeek-R1-Zero（R1 v2 §2.1 末段）；括号内为 R1 第一阶段 RL（§3.2.1）不同处 | DeepSeek-V3.2（§3.1） | TRL `GRPOConfig` 默认（main，1.14.0.dev0） |
|---|---|---|---|---|
| 起点模型 | DeepSeekMath-Instruct 7B | DeepSeek-V3-Base（R1 第一阶段：冷启动 SFT 后的模型，属第 13 课） | — | — |
| 训练问题 | ~144K（GSM8K、MATH 相关 CoT 问题） | — | — | — |
| 奖励 | **神经奖励模型**（初始 RM 基于 DSM-Base 7B，lr 2e-5；数据按 Math-Shepherd） | 基于规则（R1 第一阶段另加语言一致性奖励） | 一组奖励模型给结果奖励 | 用户提供 |
| 策略 lr | 1e-6 | 3e-6 | 未核实 | 1e-6 |
| KL 系数 $\beta$ | 0.04 | 0.001 | 按领域调，数学可弱或为 0 | **0.0** |
| 组大小 $G$ | 64 | 16 | 未核实 | 8 |
| 最大长度 | 1024 | 32,768（R1-Zero 在第 8.2k 步后改为 65,536） | — | 512（completion） |
| batch | 1024（单位未说明，见 §8） | 32 个问题 × 16 = 512 | — | — |
| 训练步数 | 未报告 | R1-Zero 共 10,400 步 ≈ 1.6 epoch | — | — |
| 每次探索后的更新 | 1 次 | 每次 rollout 8,192 个输出分 16 个 minibatch，1 个 inner epoch（即 16 次更新） | — | `num_iterations=1` |
| clip $\varepsilon$ | **未报告** | R1-Zero 段未写（R1 第一阶段：10） | — | 0.2 |
| 优势 | $(r_i-\mathrm{mean})/\mathrm{std}$ | $(r_i-\mathrm{mean})/\mathrm{std}$ | $R_i-\mathrm{mean}$（无 std） | 组内，样本 std + 1e-4（`scale_rewards="group"`） |
| KL 估计 | k3（式 4） | k3（式 2，序列级写法） | k3 × 重要性比 | k3 × 重要性比（`use_bias_correction_kl` **默认 True**）；但默认 `beta=0.0`，KL 项不参与、参考模型也不加载 |
| 长度聚合 | $\frac1G\sum_i\frac1{\lvert o_i\rvert}\sum_t$ | 序列级写法 | $\frac1G\sum_i\frac1{\lvert o_i\rvert}\sum_t$ | `loss_type="dapo"`（全 batch token 数归一）；`"grpo"` 为逐序列平均 |
| 参考模型更新 | 每个外层迭代 $\pi_{ref}\leftarrow\pi_\theta$ | 每 400 步 | — | — |
| 采样温度 | 未报告（§5.2.3 称“naive nucleus sampling”） | 1 | — | 1.0 |

### 3.6 结果汇总

**表 5（节选，Top1，灰色为 maj@32）**

| 模型 | 规模 | GSM8K | MATH | MGSM-zh | CMATH |
|---|---|---|---|---|---|
| *CoT* | | | | | |
| Gemini Ultra | - | 94.4%（灰，maj@32） | 53.2% | - | - |
| GPT-4 | - | 92.0% | 52.9% | - | 86.0% |
| Gemini Pro | - | 86.5%（灰，maj@32） | 32.6% | - | - |
| DeepSeek-LLM-Chat | 67B | 84.1% | 32.6% | 74.0% | 80.3% |
| Qwen | 72B | 78.9% | 35.2% | - | - |
| DeepSeekMath-Instruct | 7B | 82.9% | 46.8% | 73.2% | 84.6% |
| DeepSeekMath-RL | 7B | **88.2%** | **51.7%** | 79.6% | 88.8% |
| *工具集成* | | | | | |
| GPT-4 Code Interpreter | - | 97.0% | 69.7% | - | - |
| DeepSeekMath-Instruct | 7B | 83.7% | 57.4% | 72.0% | 84.3% |
| DeepSeekMath-RL | 7B | 86.7% | 58.8% | 78.4% | 87.6% |

**Base 模型（表 2 / 表 3 / 表 4）**：DSM-Base 7B CoT：GSM8K 64.2%、MATH 36.2%；工具（PoT）：GSM8K+Python 66.9%、MATH+Python 31.4%；
miniF2F-valid 25.8%、test 24.6%；MMLU 54.9%、BBH 59.5%、HumanEval 40.9%、MBPP 52.6%。

**图 5（DeepSeekMath-Instruct 1.3B；从 PDF 矢量路径解析，每 200 步一个点，共 45 点，0–8,800 步）**

| 方法 | GSM8K 起点 | GSM8K 末 5 点均值 | GSM8K 最大 | MATH 起点 | MATH 末 5 点均值 | MATH 最大 |
|---|---|---|---|---|---|---|
| RFT | 56.4 | 60.1 | 61.3 | 26.5 | 27.9 | 28.9 |
| Online RFT | 56.4 | 61.9 | 64.0 | 26.5 | 29.0 | 30.2 |
| GRPO+OS | 56.4 | 63.2 | 64.6 | 26.5 | 29.3 | 30.4 |
| GRPO+PS | 56.4 | 64.9 | 65.8 | 26.5 | 29.7 | 30.6 |

每 1,000 步的取样（GSM8K）：RFT 56.4, 59.7, 59.6, 61.3, 60.6, 59.0, 59.9, 60.2, 60.3（8,000 步）；Online RFT 56.4, 60.5, 62.3, 59.8, 59.9, 61.6, 60.7, 62.7, 62.2；
GRPO+OS 56.4, 60.3, 61.5, 61.9, 64.1, 63.4, 64.0, 62.7, 62.6；GRPO+PS 56.4, 60.5, 61.5, 61.7, 63.8, 63.3, 64.7, 64.8, 64.9。
（MATH）：RFT 26.5, 28.1, 28.0, 27.5, 27.8, 28.0, 27.9, 27.6, 27.4；Online RFT 26.5, 27.6, 28.2, 28.6, 30.2, 29.4, 29.0, 28.9, 29.1；
GRPO+OS 26.5, 28.1, 29.0, 29.2, 30.0, 29.8, 30.1, 29.5, 29.3；GRPO+PS 26.5, 28.8, 29.6, 29.8, 30.3, 29.7, 30.4, 29.6, 29.7。
（y 轴刻度：GSM8K 56–66，MATH 27–30；原图原始曲线噪声大，重绘时建议画全部 45 点或做滑动平均并注明。）

**图 6（迭代 RL，DeepSeekMath-Instruct 7B；x 轴刻度原图标为 0、1300、2300、3300、4300、5300，线性；点距 100 步）**

| 曲线 | 起止步（按轴换算） | GSM8K 起→末（最大） | MATH 起→末（最大） |
|---|---|---|---|
| Iteration-0 | 0 → 2,800（首个非零点在 400） | 82.9 → 86.1（87.0） | 46.8 → 48.7（50.0） |
| Iteration-1 | 1,100 → 5,400 | 87.0 → 87.9（88.2） | 49.3 → 51.6（52.2） |
| Iteration-2 | 2,700 → 5,300 | 87.8 → 88.8（89.1） | 51.1 → 51.3（51.7） |

每 500 步取样（GSM8K）：It-0 82.9, 85.1(500), 86.1(1000), 85.7, 85.5, 86.7(2500), 86.1(2800)；It-1 87.0(1100), 86.8(1500), 86.9, 88.2(2500), 87.3, 86.9, 87.9, 88.1, 88.2(5000), 87.9(5400)；
It-2 87.8(2700), 87.5(3000), 88.0, 88.6(4000), 87.9, 88.0(5000), 88.8(5300)。
（MATH）：It-0 46.8, 48.0, 49.2, 49.8, 49.2, 49.0, 48.7；It-1 49.3, 50.1, 50.4, 50.6, 50.9, 50.8, 51.1, 51.3, 51.6, 51.6；It-2 51.1, 51.0, 51.0, 51.1, 51.6, 51.3, 51.3。
起点 82.9 / 46.8 与表 5 的 Instruct Top1 完全相同，说明图 6 的评测是贪心 Top1 口径（推断）。

**图 7（Maj@K 与 Pass@K，温度 0.7；从 PDF 矢量路径解析；x 轴 K = 1, 4, 8, 16, 32, 64 等距排列）**

| 基准 | 曲线 | K=1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| GSM8K | Maj@K-Instruct | 81.5 | 85.8 | 87.6 | 89.3 | 89.2 | 89.5 |
| GSM8K | Maj@K-RL | 88.2 | 89.3 | 90.1 | 90.5 | 90.7 | 91.0 |
| GSM8K | Pass@K-Instruct | 81.5 | 92.3 | 95.1 | 96.7 | 98.1 | 99.1 |
| GSM8K | Pass@K-RL | 88.2 | 93.0 | 94.5 | 95.8 | 96.4 | 97.3 |
| MATH | Maj@K-Instruct | 44.6 | 50.2 | 55.1 | 58.0 | 59.2 | 59.9 |
| MATH | Maj@K-RL | 50.9 | 55.3 | 57.6 | 59.3 | 60.4 | 60.9 |
| MATH | Pass@K-Instruct | 44.6 | 64.3 | 71.7 | 78.2 | 83.4 | 87.5 |
| MATH | Pass@K-RL | 50.9 | 66.8 | 72.8 | 78.4 | 83.1 | 86.8 |

注意 K=1 在温度 0.7 下是 81.5 / 44.6（Instruct）与 88.2 / 50.9（RL），和表 5 的贪心 Top1（82.9 / 46.8、88.2 / 51.7）不是同一口径。
MATH Maj@64-RL = 60.9 与摘要的 “self-consistency over 64 samples … 60.9%” 吻合，可作为数字化正确性的旁证。
y 轴刻度：GSM8K 82–98，MATH 45–85。

---

## 4. 复算的数字

全部在 Python 3.11 + numpy 1.26 中计算（脚本留在本次会话 scratch，不入库）。

### 4.1 GRPO 优势（结果监督）

| 表达式 | 值 |
|---|---|
| `r=[1,0,0,1]; mean` | 0.5 |
| 总体 std `np.std(r)` | 0.5 → $\hat A=[+1,-1,-1,+1]$ |
| 样本 std `np.std(r,ddof=1)` | 0.5774 → $\hat A=[+0.866,-0.866,-0.866,+0.866]$ |
| TRL：`(r-mean)/(std_ddof1+1e-4)` | ±0.86588 |
| `r=[1,0,0,0]` 总体 std | 0.4330 → $\hat A=[+1.7321,-0.5774,-0.5774,-0.5774]$ |
| `r=[1,0,0,0]` 样本 std | 0.5 → $\hat A=[+1.5,-0.5,-0.5,-0.5]$ |
| `r=[1,1,1,0]` 总体 | $[+0.5774,+0.5774,+0.5774,-1.7321]$ |
| `r=[1,1,1,1]` | std = 0 → 论文公式 0/0 未定义；TRL 得 0/(0+1e-4)=0 → 无学习信号 |
| `r=[0.9,0.2,0.4,0.7]`（连续奖励）总体 | mean 0.55，std 0.2693，$\hat A=[+1.2999,-1.2999,-0.5571,+0.5571]$ |
| 同上，样本 std | 0.3109，$\hat A=[+1.1257,-1.1257,-0.4825,+0.4825]$ |
| 两种约定的比值 `sqrt((G-1)/G)` | G=2: 0.7071；4: 0.8660；8: 0.9354；16: 0.9682；64: 0.9922 |

**二值奖励的闭式（总体 std）**：正确率 $p=c/G$，$\mathrm{std}=\sqrt{p(1-p)}$，
正确者 $\hat A^+=\sqrt{(1-p)/p}$，错误者 $\hat A^-=-\sqrt{p/(1-p)}$；$\sum_i\hat A_i=0$，$\frac1G\sum_i\hat A_i^2=1$。

| $G$, $c$ | $p$ | $\hat A^+$ | $\hat A^-$ | $1/\mathrm{std}$ | $\sum_i\lvert\hat A_i\rvert$（有 std） | $\sum_i\lvert r_i-\bar r\rvert$（无 std） |
|---|---|---|---|---|---|---|
| 4, 1 | 0.25 | 1.7321 | −0.5774 | 2.3094 | 3.464 | 1.500 |
| 4, 2 | 0.50 | 1.0000 | −1.0000 | 2.0000 | 4.000 | 2.000 |
| 64, 1 | 0.0156 | 7.9373 | −0.1260 | 8.0632 | 15.875 | 1.969 |
| 64, 16 | 0.25 | 1.7321 | −0.5774 | 2.3094 | 55.426 | 24.000 |
| 64, 32 | 0.50 | 1.0000 | −1.0000 | 2.0000 | 64.000 | 32.000 |
| 64, 63 | 0.9844 | 0.1260 | −7.9373 | 8.0632 | 15.875 | 1.969 |

（$\sum|\hat A|=2G\sqrt{p(1-p)}$；无 std 时 $=2Gp(1-p)$。有 std 时，极易/极难题相对权重被放大 $1/\sqrt{p(1-p)}$ 倍，这就是 Dr. GRPO 的“难度偏差”。）

**RLOO 等价**：`r=[1,0,0,1]`：RLOO 优势 $r_i-\text{mean}_{j\ne i}r_j=[+0.667,-0.667,-0.667,+0.667]$；`G/(G-1)*(r-mean)` = 同值。

### 4.2 过程监督算例（笔者构造的数字，不是论文数据）

两个输出：$o_1$ 三步，步奖励 $[0.9,0.8,0.2]$；$o_2$ 两步，$[0.6,0.9]$。整组 $\mathbf R$ = [0.9, 0.8, 0.2, 0.6, 0.9]：
mean 0.68；总体 std 0.2638（样本 std 0.2950）。
- $o_1$ 归一化步奖励 [+0.8339, +0.4549, −1.8194] → 第 1/2/3 步内 token 的优势 = 后缀和 [−0.5307, −1.3646, −1.8194]。
- $o_2$ 归一化 [−0.3032, +0.8339] → 优势 [+0.5307, +0.8339]。
- 读法：$o_1$ 第一步本身得分高（0.9），但它的 token 优势为负，因为后缀和把后面的坏步（0.2）也算进来了——过程监督的优势是“从这里往后”的累计。

### 4.3 k1/k2/k3 与 KL 梯度系数（$w=\pi_{ref}/\pi_\theta$，$\beta=0.04$）

| $w$ | $\log w$ | $k_1=-\log w$ | $k_2=\frac12(\log w)^2$ | $k_3=w-1-\log w$ | 原 GC 的 KL 部分 $\beta(w-1)$ | V3.2 修正后 $\beta\log w$ |
|---|---|---|---|---|---|---|
| 0.01 | −4.6052 | +4.6052 | 10.604 | 3.6152 | −0.0396 | −0.1842 |
| 0.1 | −2.3026 | +2.3026 | 2.6509 | 1.4026 | −0.0360 | −0.0921 |
| 0.25 | −1.3863 | +1.3863 | 0.9609 | 0.6363 | −0.0300 | — |
| 0.5 | −0.6931 | +0.6931 | 0.2402 | 0.1931 | −0.0200 | −0.0277 |
| 0.9 | −0.1054 | +0.1054 | 0.0056 | 0.0054 | −0.0040 | — |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.1 | +0.0953 | −0.0953 | 0.0045 | 0.0047 | +0.0040 | — |
| 2 | +0.6931 | −0.6931 | 0.2402 | 0.3069 | +0.0400 | +0.0277 |
| 4 | +1.3863 | −1.3863 | 0.9609 | 1.6137 | +0.1200 | — |
| 10 | +2.3026 | −2.3026 | 2.6509 | 6.6974 | +0.3600 | +0.0921 |
| 100 | +4.6052 | −4.6052 | 10.604 | 94.395 | +3.9600 | +0.1842 |
| 1000 | +6.9078 | — | — | — | +39.96 | +0.2763 |

（$w=0.01$、100 两行的 $k$ 值：`0.01-1-log(0.01)=3.6152`，`100-1-log(100)=94.3948`，`0.5*log(100)**2=10.6038`。）
读法：当 $\pi_\theta$ 比 $\pi_{ref}$ 小 100 倍（$w=100$）时，原 k3 的 KL 系数 3.96 已是二值奖励优势（约 ±1）的 4 倍，而修正版只有 0.18。

**Schulman 实验复现**（$q=N(0,1)$ 采样，$p=N(\mu,1)$，$\log r=\mu x-\mu^2/2$，1,000 万样本，seed 0）：

| $\mu$ | 真 KL $=\mu^2/2$ | k1 bias/true, std/true | k2 bias/true, std/true | k3 bias/true, std/true |
|---|---|---|---|---|
| 0.1 | 0.005 | +0.006, 19.99 | +0.0015, 1.418 | −0.001, 1.416 |
| 1.0 | 0.5 | −0.001, 2.000 | +0.2496, 1.732 | −0.0001, 1.696 |

解析值：k1 的 std/true $=\mu/(\mu^2/2)=2/\mu$（20、2）；k2 的 bias/true $=\mu^2/4$（0.0025、0.25）。博客表中 $\mu=0.1$ 的 k2 bias 印作 0.002（MC 波动/舍入），解析值 0.0025；
其余与博客一致（1.42、1.42；2、1.73、1.7）。
（核查补记：上表 MC 数字的末位依赖随机数生成器与抽样方式（μ=0.1 一行用 `default_rng(0)` 可逐位复现，μ=1 一行末位复现不出），属正常 MC 波动。
用 200 点 Gauss–Hermite 求积得到的精确值：$\mu=0.1$：std/true k1 20.00、k2 1.4177、k3 1.4166；$\mu=1$：k1 2.000、k2 1.7321、k3 1.695；k3 的 bias 两处均为 0。）

### 4.4 梯度恒等式的数值验证（6 类 softmax 分布，有限差分 $h=10^{-6}$）

| 验证项 | 最大绝对误差 |
|---|---|
| (a) 单样本：$\nabla[-\beta k_3]=\beta(w-1)\nabla\log\pi_\theta$（式 20/21） | 1.5e-12 |
| (b) $\E_{\pi_\theta}[\nabla k_3]=\nabla\KL[\pi_{ref}\Vert \pi_\theta]$ | 3.7e-11（与 $\nabla\KL[\pi_\theta\Vert \pi_{ref}]$ 相差 0.084） |
| (c) V3.2：$\E_{\pi_{old}}[\nabla(\rho k_3)]=\nabla\KL[\pi_\theta\Vert \pi_{ref}]$（$\theta=\theta_{old}$） | 7.7e-11 |
| (d) $\E_{\pi_\theta}[k_3]=\KL[\pi_\theta\Vert \pi_{ref}]$ | 两者均 0.183208，差 1e-16 量级 |
| (e) $\nabla(\rho A)\big\rvert_{\theta=\theta_{old}}=A\nabla\log\pi_\theta$ | 1.3e-10 |

### 4.5 语料、训练量与模型

| 表达式 | 值 |
|---|---|
| `120e9/35.5e6` 平均每页 tokens | 3,380 |
| `35.5e6/40e9` 数学页占去重 CC 网页比例 | 0.0888%（约 1/1,127） |
| `120.2/13.6`（对 OpenWebMath，DSM 分词器） | 8.84 倍（论文说 “9 times”） |
| `120/14.7`（对 OpenWebMath 自报 14.7B） | 8.16 倍 |
| `120.2/51.9`（对 Proof-Pile-2） | 2.32 倍 |
| 第一轮保留占最终比例 `40/120` | 1/3 |
| 500B 配比：DSM Corpus `0.56*500` | 280B tokens → `280/120.2` ≈ **2.33 个 epoch** |
| AlgebraicStack `0.04*500` | 20B → `20/10.3` ≈ 1.94 epoch |
| arXiv `0.10*500` | 50B → 相对 ArXiv-RedPajama 28.0B ≈ 1.79 epoch（论文未说 arXiv 数据具体是哪份，见 §8） |
| GitHub 代码 `0.20*500` / CC 自然语言 `0.10*500` | 100B / 50B |
| （口径提醒，核查补记） | 120.2B 明确是 DeepSeek 分词器口径（表 1 注）；AlgebraicStack 10.3B、arXiv 28.0B 出自 §2.2 对 Proof-Pile-2 组成的文字描述，论文没说用哪个分词器统计（未核实）。分词器不同会差几个百分点（OWM：自报 14.7B vs 表 1 的 13.6B，差约 8%），所以 1.94、1.79 epoch 只看量级 |
| 预训练步数 `500e9/10e6` | 50,000 步；warmup 2,000；第一次衰减在第 40,000 步（400B tokens）到 1.327e-4；第二次在 45,000 步（450B）到 4.2e-5 |
| 1.3B 实验 `150e9/4e6` | 37,500 步；第一次衰减在 30,000 步 = **120B tokens** 处（到 1.675e-4），第二次在 135B（5.3e-5） |
| `1/sqrt(10)` | 0.3162 —— “31.6%”就是每次乘 $1/\sqrt{10}$ 的第一步（两步合计 1/10）（观察，论文未解释） |
| 参数量（config）：嵌入 `102400*4096` | 419,430,400（×2，不共享） |
| 每层 `4*4096**2 + 3*4096*11008 + 2*4096` | 202,383,360 |
| 总计 `2*emb + 30*per_layer + 4096` | **6,910,365,696 ≈ 6.91B**（非嵌入 6.07B） |
| `540/7`；`540/6.91` | 77.1（论文 “77 times larger”）；78.1 |
| KV cache 每 token `2*30*32*128` 元素 × 2 字节 | 245,760 元素 = **491,520 B = 480 KiB** |
| 4K 上下文单序列 KV | 1.875 GiB |
| 一个问题的 64 条 × 1024 token rollout 的 KV | 30 GiB |
| 1024 条 × 1024 token 的 KV | 480 GiB |
| 续训 FLOPs `6*6.910e9*500e9` | 2.07e22（非嵌入口径 1.82e22） |
| SFT token 上限 `500*256*4096` | 524,288,000 ≈ 0.52B tokens（打包到满 4K 时的上限） |
| RL 一轮遍历 `144e3*64` | 9.216M 条输出 |
| 若 batch 1024 指输出条数：`1024/64` | 每步 16 个问题（推断，见 §8） |

### 4.6 显存账（笔者估算，论文没有给显存数字）

按 ZeRO §3.1：混合精度 Adam 每个可训练参数 16 字节（fp16 参数 2 + 梯度 2 + fp32 参数/动量/方差 12，$K=12$），冻结模型只存 bf16 权重 2 字节/参数；$N=6.91$B；不计激活与 KV cache：

| 方案 | 可训练 | 冻结 | 表达式 | 合计 |
|---|---|---|---|---|
| PPO | 策略 + 价值 | 参考 + 奖励 | `2*16*N + 2*2*N` | 248.8 GB |
| GRPO | 策略 | 参考 + 奖励 | `16*N + 2*2*N` | 138.2 GB |
| 节省 | | | `1 - 138.2/248.8` | 44.4% |

（价值模型规模取与策略相同，符合 C15 “comparable size”；实际系统会用 ZeRO/卸载/共享骨干等手段，数字只作量级示意。）

### 4.7 结果差值

| 表达式 | 值 |
|---|---|
| 表 5 CoT，RL − Instruct | GSM8K +5.3；MATH +4.9；MGSM-zh +6.4；CMATH +4.2 |
| 表 5 工具，RL − Instruct | GSM8K +3.0；MATH +1.4；MGSM-zh +6.4；CMATH +3.3 |
| 图 7 GSM8K Maj@K 差（K=1,4,8,16,32,64） | +6.7, +3.5, +2.5, +1.2, +1.5, +1.5 |
| 图 7 GSM8K Pass@K 差 | +6.7, +0.7, −0.6, −0.9, −1.7, −1.8 |
| 图 7 MATH Maj@K 差 | +6.3, +5.1, +2.5, +1.3, +1.2, +1.0 |
| 图 7 MATH Pass@K 差 | +6.3, +2.5, +1.1, +0.2, −0.3, −0.7 |

（Pass@K 曲线在 K≈8（GSM8K）/ K≈32（MATH）附近**交叉**：大 K 时 RL 模型的 Pass@K 反而略低于 Instruct。论文正文只说“不提升”，没有提到交叉。）

### 4.8 Pass@k 无偏估计（Chen et al. 2021，经 `yue2025rlvr` 附录 A.2 转引）

$\text{pass@}k=\E_{x_i}\big[1-\binom{n-c_i}{k}/\binom{n}{k}\big]$，$n=64$：

| $c$ | pass@1 | @4 | @8 | @16 | @32 | @64 |
|---|---|---|---|---|---|---|
| 1 | 0.0156 | 0.0625 | 0.125 | 0.25 | 0.5 | 1.0 |
| 4 | 0.0625 | 0.2325 | 0.4219 | 0.6938 | 0.9434 | 1.0 |
| 16 | 0.25 | 0.6938 | 0.9147 | 0.9954 | 1.0 | 1.0 |
| 32 | 0.5 | 0.9434 | 0.9976 | 1.0 | 1.0 | 1.0 |

### 4.9 Maj@K vs Pass@K 的玩具模型（笔者构造，说明“锐化”机制）

每题 3 种答案，第一种是正确答案；Maj@K 平票时均匀随机打破；Pass@K $=1-(1-p_{correct})^K$；精确枚举多项分布。

| 题 | 答案分布 | K=1 (maj, pass) | K=4 | K=16 | K=64 |
|---|---|---|---|---|---|
| Q1 RL 前 | [0.40, 0.35, 0.25] | 0.40, 0.40 | 0.436, 0.870 | 0.524, 1.000 | 0.661, 1.000 |
| Q1 RL 后 | [0.60, 0.25, 0.15] | 0.60, 0.60 | 0.729, 0.974 | 0.937, 1.000 | 0.999, 1.000 |
| Q2 RL 前 | [0.10, 0.50, 0.40] | 0.10, 0.10 | 0.040, 0.344 | 0.002, 0.815 | 0.000, 0.999 |
| Q2 RL 后 | [0.02, 0.70, 0.28] | 0.02, 0.02 | 0.002, 0.078 | 0.000, 0.276 | 0.000, 0.726 |
| 两题平均 RL 前 | | 0.250, 0.250 | 0.238, 0.607 | 0.263, 0.907 | 0.330, 0.999 |
| 两题平均 RL 后 | | 0.310, 0.310 | 0.365, 0.526 | 0.469, 0.638 | 0.500, 0.863 |

RL 把“本来就大概率对”的题锐化（Maj@64 0.33→0.50，Pass@1 0.25→0.31），同时把“偶尔能对”的题压没（Pass@64 0.999→0.863）——与图 7 的方向一致。

### 4.10 其他

| 表达式 | 值 |
|---|---|
| PPO 裁剪 $L(r)=\min(rA,\mathrm{clip}(r,0.8,1.2)A)$，$A=+1$，r=0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0 | 0, 0.5, 0.8, 1.0, 1.2, 1.2, 1.2 |
| 同上 $A=-1$ | −0.8, −0.8, −0.8, −1.0, −1.2, −1.5, −2.0 |
| DPO（原版）权重 $\sigma(\beta\Delta^- - \beta\Delta^+)$，$\beta=0.1$：$\Delta^\pm=0$ | 0.5 |
| $\Delta^+=2,\Delta^-=-2$（已排对） | 0.4013 |
| $\Delta^+=-1,\Delta^-=+1$（排反） | 0.5498 |

---

## 5. 图表清单与重绘建议

### 5.1 论文原图/表清单（v3）

| 编号 | 标题（中文概括） | 显示什么 | 本课是否重绘 |
|---|---|---|---|
| 图 1 | 开源模型 MATH Top1 随时间 | 2023-02 至 2024-01 各模型点 + GPT-4/Gemini-Ultra 水平线（栅格 PNG） | 可选；点值多数无法追溯到表格（§8） |
| 图 2 | 从 Common Crawl 收集数学网页的迭代管线 | 种子 → ① 训 fastText → ② 从去重 CC（40B HTML 页）召回 → 数学语料 → ③ 发现数学域名 → ④ 人工标注 URL 路径 → 回到种子 | **重绘（F1）** |
| 图 3 | 1.3B 在不同语料上的基准曲线 | GSM8K/MATH/CMATH/BBH vs 训练 tokens（0–150B），四条语料曲线；~120B 处 DeepSeekMath Corpus、Proof-Pile-2、OpenWebMath 三条曲线同时跳升（MATH、BBH、CMATH 面板最明显），**MathPile 曲线没有跳升** | 可选（F2 用表 1 终点即可） |
| 图 4 | PPO 与 GRPO 示意 | 上：q→策略→o→奖励模型 r、参考模型 KL ⊕ 进 r、价值模型 v、GAE→A；下：q→策略→o₁…o_G→奖励→r₁…r_G→组计算→A₁…A_G，KL 直接进损失；区分“训练的模型/冻结的模型” | **重绘（F5）** |
| 图 5 | 1.3B 上 RFT / Online RFT / GRPO+OS / GRPO+PS | 两个面板，0–8,800 步 | **重绘（F13），数据见 §3.6** |
| 图 6 | 7B 迭代 RL | Iteration-0/1/2 | **重绘（F14）** |
| 图 7 | Maj@K 与 Pass@K | Instruct vs RL，GSM8K、MATH | **重绘（F15）** |
| 表 1 | 语料对比 | §3.2 | 用于 F2 |
| 表 2–4 | Base 模型成绩 | §3.6 | 部分引用 |
| 表 5 | Instruct / RL 成绩 | §3.6 | F4 用 |
| 表 6–9 | 代码、arXiv 消融 | C30、C31 | 文字引用 |
| 表 10 | 统一范式 | §2.4 | **重绘（F12）** |

### 5.2 本课建议的 15 张图（数据齐全，可直接用 Python 算几何）

**F1 数据管线迭代图（机制图，据图 2 重绘）。** 节点与数字：种子 OpenWebMath（正例 500K + CC 负例 500K）→ fastText（dim 256、lr 0.1、n-gram≤3、minCount 3、3 epoch）
→ 从“去重 CC：40B 个 HTML 网页”召回并按分数排序 → 第 1 轮保留 top 40B tokens → 按域名统计（>10% 页已收集 ⇒ 数学域名）→ 人工标注 URL 路径 → 新正例回种子。
右侧标“4 轮后：35.5M 网页 / 120B tokens；第 4 轮约 98% 已在第 3 轮收集”。颜色：存储/语料 `--fig-amber`，模型 `--fig-blue`，人工步骤 `--fig-violet`。
**提醒：两个 40B 用不同单位（网页 vs tokens），图上要写全单位。**

**F2 语料对比柱状图（账单图）。** 数据 = 表 1（§3.2）。建议两组：左 GSM8K / MATH / CMATH（1.3B，150B tokens），右 语料大小（8.9、13.6、51.9、120.2B）。
可加 No Math Training 基线线。

**F3 500B 训练配比（账单图）。** 56/4/10/20/10（%）→ 280/20/50/100/50（B tokens）；在 DSM Corpus 扇区标“≈2.33 epoch”。

**F4 模型谱系与各阶段成绩（机制图 + 数字）。** DeepSeek-LLM-7B Base →（+2T tokens，Coder 论文 §5）→ Coder-Base-v1.5 7B（取衰减前检查点）→（+500B tokens）→
DeepSeekMath-Base 7B（GSM8K 64.2 / MATH 36.2，few-shot CoT）→（SFT 776K 样本，500 步）→ Instruct 7B（82.9 / 46.8）→（GRPO，144K 问题，G=64）→ RL 7B（88.2 / 51.7）。
标注“RL 只用了 GSM8K/MATH 的 CoT 问题”。

**F5 PPO vs GRPO 结构对比（机制图，据图 4 重绘）。** 上排 PPO：策略（训练）、价值模型（训练）、奖励模型（冻结）、参考模型（冻结）；KL 以 ⊕ 进每 token 奖励 → GAE → $A_t$。
下排 GRPO：策略（训练）、奖励模型（冻结）、参考模型（冻结）；$o_1\dots o_G$ → $r_1\dots r_G$ → “组计算” → $\hat A_1\dots\hat A_G$；KL 直接进损失。
可在旁边放 §4.6 的显存条：PPO 248.8 GB vs GRPO 138.2 GB（标“笔者估算，混合精度 Adam 16 B/参数，不含激活”）。
颜色：训练中的模型 `--fig-blue`，冻结模型 `--fig-muted`，奖励 `--fig-green`，KL/参考 `--fig-violet`。

**F6 组内优势标准化示例（机制图 + 数字）。** 四份答卷，两组数据：
(a) $r=[1,0,0,1]$：均值 0.5；总体 std 0.5 → $\hat A=[+1,-1,-1,+1]$；样本 std 0.577 → ±0.866；
(b) $r=[1,0,0,0]$：均值 0.25；总体 std 0.433 → $[+1.732,-0.577,-0.577,-0.577]$；
(c) $r=[1,1,1,1]$：std 0 → 全 0（“这道题不提供信号”）。
柱子正值 `--fig-green`、负值 `--fig-red`；标均值虚线。对应 style-guide 类比 5（同题多答、组内排名）。

**F7 二值奖励下优势与难度（账单图）。** x 轴正确率 $p\in(0,1)$，曲线：$\hat A^+=\sqrt{(1-p)/p}$、$\hat A^-=-\sqrt{p/(1-p)}$（有 std）；$1-p$、$-p$（无 std，Dr. GRPO/V3.2）；
以及每题总权重 $2\sqrt{p(1-p)}$ vs $2p(1-p)$（除以 $G$ 后）。取样点见 §4.1 与下表：
p = 1/64（≈0.0156）, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 63/64（≈0.9844）→ $\hat A^+$ = 7.937, 4.359, 3.000, 1.732, 1.000, 0.577, 0.333, 0.229, 0.126；$1/\mathrm{std}$ = 8.063, 4.588, 3.333, 2.309, 2.000, 2.309, 3.333, 4.588, 8.063。
建议 y 轴对数或截断在 ±8。

**F8 k1 / k2 / k3 估计器曲线（账单图）。** x 轴 $\log w\in[-3,3]$（$w=\pi_{ref}/\pi_\theta$），三条曲线 $k_1=-\log w$、$k_2=\frac12(\log w)^2$、$k_3=e^{\log w}-1-\log w$；
标出 $k_1$ 可为负、$k_3\ge0$ 且在 0 处与 $k_2$ 相切（二阶一致）。取样点见 §4.3。可加一个小表：Schulman 实验的 bias/std（§4.3）。

**F9 KL 梯度系数：原 k3 vs V3.2 修正（账单图）。** x 轴 $w$（对数轴 0.01–1000），y 轴 GC 的 KL 部分：$\beta(w-1)$ 与 $\beta\log w$，$\beta=0.04$；
标注 $w=100$：3.96 vs 0.184；再画一条 $\pm1$ 的水平带表示典型优势幅度。

**F10 “只更新一次”时 clip 为何不起作用（机制图）。** 左：PPO 裁剪函数（$A>0$、$A<0$，$\varepsilon=0.2$，数据见 §4.10，可链接第 3 课的同名图）；
右：在 $\rho=1$ 处（DeepSeekMath 的唯一一步）两段重合，梯度 = $A\nabla\log\pi$。

**F11 过程监督优势的“后缀和”（机制图 + 数字）。** 用 §4.2 的例子：$o_1$ 三步 [0.9, 0.8, 0.2]、$o_2$ 两步 [0.6, 0.9]；画 token 条，按步着色；
每步上方写归一化奖励（+0.83, +0.45, −1.82 / −0.30, +0.83），下方写优势（−0.53, −1.36, −1.82 / +0.53, +0.83）。与结果监督对照（整条同一个值）。

**F12 统一范式表（机制图）。** 三列：数据源（离线 $\pi_{sft}$ / 在线 $\pi_\theta$）、奖励函数（Rule / Model）、梯度系数；六行按表 10。
再加一行数值示例：同一组 $r=[1,0,0,1]$ 下各方法对四个回答的 GC：SFT —（只用人工数据）；RFT/Online RFT $[1,0,0,1]$；GRPO（$\beta$ 项取 0）$[+1,-1,-1,+1]$。
强调 Online RFT “不惩罚错误回答”（C34）。

**F13 图 5 重绘（账单图）。** 数据 §3.6（每 1,000 步取样；如需全部 45 点，用本笔记 §3.6 的方法从 PDF 重新解析）。
标题写明“DeepSeekMath-Instruct 1.3B”。

**F14 图 6 重绘（账单图）。** 数据 §3.6；x 轴建议直接用步数（0–5,400），注明“原图刻度 0/1300/2300/…”。

**F15 图 7 重绘 + 玩具模型（账单图）。** 左：§3.6 的 Maj@K / Pass@K 数据，x 轴 K 取对数等距（1, 4, 8, 16, 32, 64）；强调 Pass@K 曲线交叉；
右（可选）：§4.9 的两题玩具模型平均值，说明“锐化”。颜色：Instruct 用 `--fig-muted`，RL 用 `--fig-blue`；Maj 实线、Pass 虚线。

---

## 6. 算例与练习素材（附答案）

**X1（计算）组内优势。** $G=4$，奖励 [1,0,0,1] 与 [1,0,0,0]。分别用总体与样本标准差算 $\hat A$。
答：[1,0,0,1]：均值 0.5，总体 std 0.5 → ±1；样本 std 0.5774 → ±0.866。[1,0,0,0]：均值 0.25，总体 std 0.4330 → [1.732, −0.577, −0.577, −0.577]；
样本 std 0.5 → [1.5, −0.5, −0.5, −0.5]。两种约定只差常数 $\sqrt{3/4}=0.866$。

**X2（推导）二值奖励的闭式与性质。** 证明 $c$ 个正确、$G-c$ 个错误时（总体 std），$\hat A^+=\sqrt{(1-p)/p}$，$\hat A^-=-\sqrt{p/(1-p)}$，且 $\sum\hat A=0$、$\frac1G\sum\hat A^2=1$；
求 $G=64,c=1$ 时的值。答：均值 $p$，方差 $p(1-p)$；代入即得。$c\hat A^+ +(G-c)\hat A^-=G[p\sqrt{(1-p)/p}-(1-p)\sqrt{p/(1-p)}]=G[\sqrt{p(1-p)}-\sqrt{p(1-p)}]=0$。
$G=64,c=1$：$\hat A^+=7.937$，$\hat A^-=-0.126$。

**X3（概念）全对或全错的组。** 为什么 $r=[1,1,1,1]$ 或 [0,0,0,0] 的题对策略梯度没有贡献？KL 项还有作用吗？
答：优势全为 0（std=0 时实现里分子也为 0；TRL 得 0/(0+1e-4)=0）；该题只剩 KL 项的梯度 $\beta(w-1)\nabla\log\pi$。所以要让题目难度落在“组内有对有错”的区间。

**X4（推导）k3 非负且（对值）无偏。** 答：见 §2.5 (d)(e)。数值：$w=2$ 时 $k_3=0.3069$，$w=0.5$ 时 $0.1931$，都 >0；$k_1$ 分别为 −0.693 与 +0.693，可为负。

**X5（推导 + 计算）KL 项的梯度系数。** 从式 (19) 推出式 (21) 的 $\beta(w-1)$；取 $\beta=0.04$，求 $w=0.5,2,10,100$ 时的值，并与二值奖励 $G=4$、$c=2$ 的优势比较。
答：推导见 §2.5 (c)。数值 −0.02、+0.04、+0.36、+3.96；优势为 ±1。$w=100$ 时 KL 项已是优势的 4 倍——V3.2 称之为“unbounded weights”。

**X6（推导，进阶）把 k3 当损失，梯度指向哪个 KL？** 证明单步情形 $\E_{\pi_\theta}[\nabla k_3]=\nabla\KL[\pi_{ref}\|\pi_\theta]$，并说明 V3.2 乘上 $\rho=\pi_\theta/\pi_{old}$ 后变成 $\nabla\KL[\pi_\theta\|\pi_{ref}]$。
答：见 §2.5 (f)(g)；数值验证见 §4.4。

**X7（概念）clip 为什么没起作用。** DeepSeekMath 每次探索后只更新一次。此时 GRPO 的梯度是什么？$\varepsilon$ 取 0.1 还是 0.3 会改变结果吗？
答：唯一一步在 $\theta=\theta_{old}$，$\rho=1$，梯度 $=\frac1G\sum_i\frac1{|o_i|}\sum_t[\hat A_{i,t}+\beta(w_{i,t}-1)]\nabla\log\pi_\theta$，与 $\varepsilon$ 无关。
（前提是“只更新一次”指对整批只做一次优化器步，这是附录 A.1.5 的读法，论文正文没说明 minibatch 数，见 §2.5 (b) 补记与 §8。
在这个读法下，论文不报告 $\varepsilon$ 也不影响复现；R1 用 minibatch 多步更新，才需要 $\varepsilon$。）

**X8（计算）统一范式下的梯度系数。** 同一组四个回答 $r=[1,0,0,1]$（1 = 答对），$\beta$ 项忽略。写出 RFT、Online RFT、GRPO（总体 std）的 GC。
答：RFT/Online RFT：[1, 0, 0, 1]（错误回答系数 0，不被惩罚；两者差别只在样本来自 $\pi_{sft}$ 还是 $\pi_\theta$）；GRPO：[+1, −1, −1, +1]。

**X9（计算）过程监督优势。** 用 §4.2 的数据求每步 token 的优势。答：$o_1$：[−0.531, −1.365, −1.819]；$o_2$：[+0.531, +0.834]。
追问：为什么 $o_1$ 第一步得分 0.9 却是负优势？答：优势是“从这一步到结束”的归一化奖励之和，后面的坏步拖累了前面。

**X10（计算）语料账。** 35.5M 网页、120B tokens：平均每页多少 tokens？500B 训练中 56% 来自该语料，相当于几个 epoch？
答：120e9/35.5e6 ≈ 3,380 tokens/页；0.56×500 = 280B，280/120.2 ≈ 2.33 epoch。

**X11（计算）显存与 KV 账。** (a) 7B 模型（6.91B 参数），用 16 字节/可训练参数、2 字节/冻结参数估算 PPO 与 GRPO 的模型状态显存。
(b) DeepSeekMath 7B 是 30 层、32 头 × 128 维的 MHA，bf16 下每 token KV 多少字节？一个问题采 64 条、每条 1024 token 需要多少 KV？
答：(a) PPO 248.8 GB，GRPO 138.2 GB，省 44.4%。(b) $2\times30\times32\times128\times2=491{,}520$ B = 480 KiB/token；$64\times1024\times480$ KiB = 30 GiB。
（衔接第 4 课：rollout 是解码密集型负载，GRPO 省掉了价值模型，但组采样把 KV 压力放大了 $G$ 倍。）

**X12（计算）Pass@k。** $n=64$ 个样本中有 $c=4$ 个正确。用无偏估计求 pass@1、pass@4、pass@16。
答：$1-\binom{60}{k}/\binom{64}{k}$：0.0625、0.2325、0.6938。

**X13（概念 + 计算）Maj@K 与 Pass@K。** 用 §4.9 的两题模型解释为什么 RL 可以提高 Maj@K、Pass@1，却降低大 K 的 Pass@K；再读 §3.6 图 7 的数据，找出 Pass@K 曲线交叉的位置。
答：锐化把 Q1 的正确答案推成众数（Maj@64 0.661→0.999），但把 Q2 的正确概率从 0.10 压到 0.02（Pass@64 0.999→0.726）。图 7：GSM8K 在 K=4 与 8 之间交叉（+0.7 → −0.6），MATH 在 16 与 32 之间（+0.2 → −0.3）。

**X14（推导）RLOO 等价。** 证明 $\frac{G}{G-1}(r_i-\bar r)=r_i-\frac1{G-1}\sum_{j\ne i}r_j$，并说明为什么这种基线不引入偏差，而除以组内 std 的版本不再严格等价。
答：展开 $\bar r=\frac1G(r_i+\sum_{j\ne i}r_j)$ 即得；留一基线不依赖 $o_i$，$\E[b\,\nabla\log\pi(o_i)]=b\,\E[\nabla\log\pi]=0$；std 含 $r_i$，与 $o_i$ 相关。数值：[1,0,0,1] → ±0.667。

---

## 7. 勘误与常见误解

**E1 “GRPO 的 KL 项是无偏的，所以它在优化 $\KL[\pi_\theta\|\pi_{ref}]$。”** 部分错误。式 (4) 对 KL 的**值**无偏，前提是 token 采样自 $\pi_\theta$；
但把它当损失自动求导，期望梯度是 $\nabla\KL[\pi_{ref}\|\pi_\theta]$（§2.5 (f)，`tang2025kl` §3）；序列上 token 级损失一般至多给出部分梯度，而 k3 形式的 token 级损失连任何现有序列级散度的部分梯度都不是（`tang2025kl` §4.1）；
而且实际样本来自 $\pi_{\theta_{old}}$。DeepSeek 自己在 V3.2 §3.1 用重要性比修正了它。小 KL 区间两方向近似相同，DeepSeekMath 的 $\beta=0.04$、单步更新下影响多大，没有实验数据（§8）。

**E2 “k3 guaranteed to be positive”。** 应为非负：$w=1$ 时 $k_3=0$。

**E3 “GRPO 的 std 是总体标准差/样本标准差”。** 论文没说。TRL 用样本 std（Bessel）+ 1e-4。两者差 $\sqrt{(G-1)/G}$，$G=64$ 时仅 0.8%，$G=4$ 的教学例子里差 13.4%，写算例时要声明约定。

**E4 “GRPO 用的是基于规则的可验证奖励。”** 在 DeepSeekMath 里不是：表 10 GRPO 的 Reward Function 是 “Model”，§4.2 训练了神经奖励模型（过程/结果），迭代 RL 还持续训练奖励模型。
基于规则的奖励是 R1 的选择。“GRPO = RLVR”是后来的用法。

**E5 “GRPO 去掉了奖励模型和参考模型。”** 只去掉了价值模型（critic）。DeepSeekMath 的 GRPO 仍有策略、参考、奖励三个模型（图 4）。显存节省来自价值模型的权重、梯度与优化器状态（§4.6 估算约 44%）。

**E6 “DeepSeekMath-RL 在 MATH 上 51.7% 是投票/工具的成绩。”** 51.7% 是 CoT、无工具、Top1；64 样本自洽投票是 60.9%；工具集成推理 RL 是 58.8%。
表 5 里灰色（maj@32）的只有 Gemini Ultra 与 Gemini Pro 的 GSM8K（94.4%、86.5%），Gemini Ultra 的 MATH 53.2% 不是灰色。

**E7 “40B”的两种单位。** §2.1 同一段先说去重后得到 40B 个 **HTML 网页**，又说第一轮保留 top 40B **tokens**。图 2 中心写的是 “40B HTML pages”。转述时必须带单位。

**E8 “120B tokens”的精度与“9 倍”。** 正文 120B，表 1 为 120.2B（DeepSeek 分词器，词表约 100K）。OpenWebMath 在表 1 中是 13.6B（同一分词器），其论文摘要自报 14.7B。
“9 times”用 DeepSeek 口径是 8.84 倍，用 OWM 自报口径是 8.16 倍。

**E9 算法 1 的“Equation 21”。** 算法第 11 行说“maximizing the GRPO objective (Equation 21)”，式 (21) 其实是梯度系数；GRPO 目标是式 (3)（或附录简化版式 (19)）。

**E10 式 (12) 的笔误与“DPO”的含义。** 式 (12) 负例项分子分母都写作 $o^-_{<t}$（应为 $o^-_t$）。更重要的是：DeepSeekMath 附录里的 DPO 是**长度归一化**版本，GC（式 14）写成逐 token 比值且没有 $\beta$ 前因子；
原版 DPO（`rafailov2023dpo` 式 (7)）用整句对数概率比。讲“统一范式”时应说明这是作者的简化写法。

**E11 式 (5) 的 $\pi_{rf}$ 不是 $\pi_{ref}$。** $\pi_{rf}$ 指奖励函数。式 (18)(21) 又写成 $\pi_{\theta_{rm}}$。

**E12 “PPO 的 KL 惩罚”指两件不同的事。** PPO 原论文 §4 的 $L^{KLPEN}$ 惩罚的是**对旧策略** $\pi_{\theta_{old}}$ 的 KL，且 $\beta$ 自适应；RLHF（InstructGPT）里的 KL 是**对 SFT/参考模型**、按 token 加进奖励。
DeepSeekMath 式 (2) 指的是后者，并正确地引用了 Ouyang et al. 2022。另外 PPO 原文目标没有 $1/|o|$，DeepSeekMath 式 (1) 的长度归一化是其写法（Dr. GRPO 表 2 指出很多开源 PPO 实现也这样做）。

**E13 “RL 提升 Maj@K 而不提升 Pass@K”的适用范围。** 这是在 Instruct→RL、GSM8K/MATH（RL 的同分布题）、温度 0.7、K≤64 下的**实测**，不是定理。
数字化的图 7 显示大 K 时 RL 的 Pass@K **略低**（GSM8K Pass@64 99.1→97.3，MATH 87.5→86.8），论文正文只说“不提升”。
方向可以说，幅度要谨慎：论文没给置信区间；若按完整测试集粗算二项标准误（GSM8K 1,319 题、MATH 5,000 题——论文实际用的题数本笔记未核实），
两模型之差的标准误约 0.55 pp（GSM8K）与 0.67 pp（MATH），所以 GSM8K 的 −1.8 约为 3 个标准误，MATH 的 −0.7 只有约 1 个标准误。作者自己把原因推测为只用了指令微调问题和朴素采样（§5.2.3）。
Yue et al. 2025 在 base→RLVR、k 到 256 以上的设置下得到了同方向且更强的结论（pass@256 覆盖率下降）。不要把两者混为一个结论。

**E14 “迭代 RL”与“GRPO iteration μ”是两回事。** 迭代 RL（§4.1.4）= 用新采样重训奖励模型（回放 10% 历史数据）+ 把 $\pi_{ref}$ 重置为当前策略；μ 是同一批数据上的内层更新次数（DeepSeekMath 实际为 1）。

**E15 图 5 是 1.3B 模型。** 图 5 的对照实验（RFT、Online RFT、GRPO+OS、GRPO+PS）用的是 DeepSeekMath-Instruct **1.3B**，不是 7B；图 6 才是 7B。表 10 列了 DPO 和 PPO，但图 5 里没有这两条曲线。

**E16 DeepSeek-Coder-v1.5 的继续训练量：2T 还是 2B？** DeepSeek-Coder 论文 §5 写 “2 trillion tokens”，§6 结论写 “2 billion tokens”。以 §5 为准的可能性大（§5 是方法描述），但本笔记未找到第三方来源裁决（§8）。

**E17 DeepSeekMath-Base 从“衰减前检查点”出发。** 常被省略。表 4 的 † 行（MMLU 42.9%、BBH 42.9%、HumanEval 40.2%、MBPP 52.6%）才是真正的起点，不是最终的 Coder-Base-v1.5（49.1%、55.2%、43.2%、60.4%）。

**E18 GRPO 的公式在 DeepSeek 自己的论文里也变过。** DeepSeekMath：token 级、$1/|o_i|$、除以 std、k3、$\beta=0.04$、$G=64$。R1-Zero/R1：序列级写法、除以 std、k3、$\beta=0.001$、$G=16$（R1 第一阶段 RL 另设 $\varepsilon=10$）。V3.2：token 级、**不除以 std**、k3×重要性比。
TRL 当前默认：`beta=0.0`（不加 KL）、`loss_type="dapo"`（按全 batch 的 token 数归一）、组内样本 std；一旦设 `beta>0`，KL 默认就是 V3.2 式的“k3 × 重要性比”（`use_bias_correction_kl=True` 为默认值），不是 DeepSeekMath 式 (4) 的原版 k3。“GRPO 的超参数”必须说清是哪篇/哪个实现。

**E19 图 1 的点值不都来自论文表格。** 按 PNG 像素质心换算（y 轴网格 10–50，15.275 px/点），图 1 中 Llemma-34B ≈ 31.4、Mistral-7B ≈ 28.2，而表 2 的 base 成绩分别是 25.3%、14.3%；WizardMath-70B ≈ 22.3，与表 5 的 WizardMath-v1.0 70B 22.7% 也不一致（差 0.4，超出像素误差约 0.1）。只有 Qwen-72B（35.2）、GPT-4 API 线（52.9）、Gemini-Ultra 线（53.2）能与表 5 对上。重绘图 1 时不要把它的点当成表 2 的数字。

**E20 “31.6%” 不是随意的数。** $1/\sqrt{10}=0.316$：两次衰减各乘约 $1/\sqrt{10}$，合计到 10%。论文没解释，课程里标“观察”。1.3B 实验的第一次衰减恰在 120B tokens 处，图 3 中除 MathPile 外的三条语料曲线在约 120B 处同时跳升（MathPile 曲线平坦，看不出跳升），笔者推测与此有关（论文未说明）。

**E21 Schulman 博客的年份。** 页面日期 2020/03/07；DeepSeekMath、R1、V3.2 都引作 “Schulman, 2020”；Tang & Munos 正文引作 “Schulman, 2017”（与博客页面日期不符）。

**E22 Schulman 的记号与课程记号相反地撞车。** 博客的 $q$ 是采样分布（对应 $\pi_\theta$），$p$ 对应 $\pi_{ref}$，$r=p/q$ 对应 $\pi_{ref}/\pi_\theta$。直接照抄会与课程的 $q$（问题）、$r$（奖励）冲突（§2.8）。

**E23 “PPO 必然差于 GRPO”。** R1 附录 A.3 的对照显示：GAE $\lambda=0.95$ 时 PPO 明显更差，调到 $\lambda=1.0$ 后接近 GRPO；GRPO 的优势主要在省掉价值模型带来的资源与调参成本。

---

## 8. 未能核实 / UNVERIFIED

1. **std 约定**：DeepSeek 内部 GRPO 实现用总体还是样本标准差、是否加 $\epsilon$——论文未写，官方仓库无训练代码。
2. **clip 系数 $\varepsilon$**、**rollout 采样温度 / top-p**、**RL 总步数**、**每个外层迭代的步数 $M$**：DeepSeekMath 均未报告（§5.2.3 只说“naive nucleus sampling”）。
3. **“training batch size is 1024”的单位**：是 1024 条输出（=16 题×64）还是 1024 个问题，论文未说明；R1 的用法（32 题×16=512）支持前者，但只是类比推断。
4. **发布的 DeepSeekMath-RL 7B 用的是哪个变体**（结果监督还是过程监督、迭代几轮）：论文未明确。
5. **奖励模型细节**：规模（初始 RM 基于 7B Base）、训练集大小、过程/结果 RM 各自的数据量、Math-Shepherd 的 HE 还是 SE 标签——未报告。
6. **表 5 的 Top1 是否贪心解码**：论文写 Top1、“without … voting”，未明说贪心；图 6 起点与表 5 Instruct 数字完全相同，间接支持贪心口径（推断）。
7. **图 7 的 Pass@K 估计方式**（每题采样数 $n$、是否用无偏估计）与 **Maj@K 的平票规则**：未说明。
8. **数字化精度**：图 5–7 的数值由 PDF 矢量坐标经刻度线线性换算得到，误差约 ±0.1 个百分点；图 6 的 x 轴刻度（0, 1300, 2300, …）与各迭代起点（约 1,100、2,700 步）的含义论文未解释。
9. **图 1 的点值**：栅格图，核查时按像素质心换算（LLaMA1-65B 10.6、WizardMath-70B 22.3、Qwen-14B 24.8、Mistral-7B 28.2、Llemma-34B 31.4、Qwen-72B 35.2、“GPT-4 early version”线 42.5），多数点无法对应论文表格；来源未知。
10. **“almost 7 times the size of the math web pages used by Minerva”**：隐含 Minerva 数学网页约 17B tokens，本笔记未核对 Minerva 论文。
11. **500B 配比中“10% from arXiv”用的是哪份 arXiv 数据**（ArXiv-RedPajama？Proof-Pile-2 的 arXiv 部分？）：未说明；§4.5 的 1.79 epoch 以 28.0B 为假设。
12. **DeepSeek-Coder-v1.5 的继续训练量**（2T vs 2B，论文自相矛盾，E16）。DeepSeek-LLM-7B Base 本身的预训练 token 数本笔记未核（属第 6 课）。
13. **DeepSeekMath 是否有同行评审版本**：检索未发现，按 arXiv 预印本处理。InstructGPT、Dr. GRPO 的会议信息本笔记未核。
14. **Math-Shepherd 的 ACL 页码**（9426–9439）来自检索摘要，未打开 aclanthology 页面核对。
15. **k3 梯度偏差在 DeepSeekMath 设置下的实际影响**：无实验数据；Tang & Munos 的推测是低 KL 区间影响小。
16. **§4.6 的显存节省 44%**：笔者按 ZeRO 16Ψ 规则估算，不是论文数字；实际系统依赖并行与卸载策略。
17. **“人工标注”的规模**（§2.1 第 3 段与 §1 所说的 human annotation）：多少 URL 路径、多少人工时，未报告。
18. **PRM800K 约 20% 错误标注**：DeepSeekMath 脚注 7 引自一个 GitHub issue 评论，本笔记未打开该链接核对。
19. **图 5 所用 DeepSeekMath-Instruct 1.3B 的来历**（如何预训练与 SFT）：论文未说明。
20. **R1-Zero 与 R1 第一阶段 RL 的超参数**只在本笔记用作对照（R1 v2 §2.1 末段、§3.2.1），完整核对与两者差异的解读属于第 13 课。
21. **“single update” 的确切含义**（核查补记）：是对整批 rollout 只做一次优化器步，还是一个 epoch 分多个 minibatch，DeepSeekMath 没说。
    §2.5 (b)“clip 不起作用”的推论依赖前一种读法（附录 A.1.5 的简化假设就是这样写的）。
22. **图 7 Pass@K “交叉”的统计显著性**（核查补记）：论文没给置信区间、每题样本数与测试题数；§7 E13 的标准误只是按完整测试集的二项近似粗算。
23. **图 1 WizardMath-70B 点（≈22.3）对应的版本**（核查补记）：与表 5 WizardMath-v1.0 70B 的 22.7% 不一致，来源未知。

---

## 9. 参考文献卡片草稿

```
id: shao2024deepseekmath
title: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
a: Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo (2024), arXiv preprint, arXiv:2402.03300v3
url: https://arxiv.org/abs/2402.03300v3
finding: 用 fastText 四轮迭代从 Common Crawl 挖出 120B tokens 数学语料，训练 7B 模型并提出去掉价值网络、以组内奖励标准化作优势的 GRPO，RL 后 MATH 达 51.7%。
matters: 本课的主来源：GRPO 目标式 (3)、k3 KL 估计式 (4)、结果/过程监督优势、统一范式表 10 和 Maj@K/Pass@K 观察都出自这里。
```

```
id: schulman2017ppo
title: Proximal Policy Optimization Algorithms
a: Schulman, Wolski, Dhariwal, Radford, Klimov (2017), arXiv preprint, arXiv:1707.06347v2
url: https://arxiv.org/abs/1707.06347
finding: 提出裁剪概率比的代理目标 L^CLIP（式 7），使目标成为未裁剪目标的悲观下界，并给出自适应 KL 惩罚的替代方案。
matters: GRPO 直接沿用 PPO 的裁剪结构；课程用它说明 GRPO 改了什么（优势来源、KL 放置）以及为何单步更新时裁剪不生效。
```

```
id: schulman2020kl
title: Approximating KL Divergence
a: John Schulman (2020), blog post
url: http://joschu.net/blog/kl-approx.html
kind: artefact
finding: 给出 KL 的三个蒙特卡洛估计 k1、k2、k3，其中 k3=(r-1)-log r 借控制变量得到，既无偏又恒非负，方差与 k2 相当。
matters: GRPO 式 (4) 的 KL 估计器就是 k3，本课推导其非负性与无偏性时以此为出处，并需说明博客记号与课程记号的对应。
```

```
id: deepseekai2025r1
title: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
a: DeepSeek-AI, Guo, Yang, Zhang, Song, Wang, et al. (2025), Nature 645:633–638, arXiv:2501.12948v2
url: https://arxiv.org/abs/2501.12948
finding: 用 GRPO 与基于规则的奖励在 V3-Base 上直接做 RL 得到 R1-Zero，论文给出序列级写法的 GRPO 目标与 GRPO/PPO 对比。
matters: 本课用它说明 GRPO 的公式和超参数在 DeepSeek 后续工作中如何变化，并为第 13 课铺路。
```

```
id: deepseekai2025v32
title: DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
a: DeepSeek-AI (2025), arXiv preprint, arXiv:2512.02556v1
url: https://arxiv.org/abs/2512.02556
finding: 规模化 GRPO 时改用不除以标准差的优势，并用重要性比修正 k3，使 KL 梯度无偏、避免 π_θ≪π_ref 时的无界权重。
matters: 这是 DeepSeek 对 DeepSeekMath 版 KL 估计的官方修正，本课“GRPO 的局限”一节据此说明 k3 当损失求导的问题。
```

```
id: liu2025drgrpo
title: Understanding R1-Zero-Like Training: A Critical Perspective
a: Liu, Chen, Li, Qi, Pang, Du, Lee, Lin (2025), arXiv preprint, arXiv:2503.20783v2
url: https://arxiv.org/abs/2503.20783
finding: 指出 GRPO 的 1/|o_i| 带来回答长度偏差、除以组内标准差带来题目难度偏差，去掉两者（Dr. GRPO）即回到无偏的策略梯度，并与 RLOO 等价。
matters: 本课讲组内标准化时用它说明这一设计是“选择”而非“推论”，以及它的代价。
```

```
id: tang2025kl
title: On a few pitfalls in KL divergence gradient estimation for RL
a: Tang, Munos (2025), arXiv preprint, arXiv:2506.09477v1
url: https://arxiv.org/abs/2506.09477
finding: 证明把无偏 KL 估计当损失自动求导一般得不到 KL 的梯度，k3 形式得到的是另一方向 KL 的梯度，token 级损失在序列上只给部分梯度。
matters: 为本课“k3 对值无偏但对梯度不无偏”的勘误提供独立来源。
```

```
id: yue2025rlvr
title: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
a: Yue, Chen, Lu, Zhao, Wang, Yue, Song, Huang (2025), NeurIPS 2025 (Oral), arXiv:2504.13837v5
url: https://arxiv.org/abs/2504.13837
finding: 在多个模型与基准上发现 RLVR 提高 pass@1 但在大 k 时降低 pass@k，基座模型的可解题覆盖面更大。
matters: 本课用它把 DeepSeekMath 的 Maj@K/Pass@K 观察放到后续研究中，并说明两者的实验设置不同。
```

```
id: wang2024mathshepherd
title: Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
a: Wang, Li, Shao, Xu, Dai, Li, Chen, Wu, Sui (2024), ACL 2024, arXiv:2312.08935v3
url: https://aclanthology.org/2024.acl-long.510/
finding: 用 completer 从中间步续写多条解答，以能否到达正确答案（HE）或到达频率（SE）自动标注步骤质量，训练过程奖励模型。
matters: DeepSeekMath 的奖励模型训练集按此构造，本课讲过程监督时需要它解释“步级奖励从哪来”。
```

```
id: ouyang2022instructgpt
title: Training language models to follow instructions with human feedback
a: Ouyang, Wu, Jiang, et al. (2022), arXiv:2203.02155v1
url: https://arxiv.org/abs/2203.02155
finding: 三阶段 RLHF 中，PPO 阶段在每个 token 的奖励上加入相对 SFT 模型的 KL 惩罚以缓解奖励模型被过度优化。
matters: DeepSeekMath 式 (2) 的每 token KL 惩罚出自这里，本课用它对比 GRPO 把 KL 移进损失的做法。
```

```
id: rafailov2023dpo
title: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
a: Rafailov, Sharma, Mitchell, Ermon, Manning, Finn (2023), arXiv:2305.18290v3
url: https://arxiv.org/abs/2305.18290
finding: 由 KL 约束下最优策略的闭式解推出只需偏好对的 DPO 损失，其梯度按隐式奖励排错的程度加权。
matters: 本课在统一范式里推导 DPO 的梯度系数时以原版式 (7) 为准，并指出 DeepSeekMath 附录用的是长度归一化的变体。
```

```
id: guo2024deepseekcoder
title: DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence
a: Guo, Zhu, Yang, Xie, Dong, Zhang, et al. (2024), arXiv preprint, arXiv:2401.14196v2
url: https://arxiv.org/abs/2401.14196
finding: 发布 DeepSeek-Coder 系列，并在 §5 描述从 DeepSeek-LLM-7B Base 继续预训练得到 Coder-v1.5 7B。
matters: DeepSeekMath-Base 从 Coder-Base-v1.5 的衰减前检查点初始化，本课的模型谱系图需要它。
```

```
id: paster2023openwebmath
title: OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text
a: Paster, Dos Santos, Azerbayev, Ba (2023), arXiv preprint, arXiv:2310.06786v1
url: https://arxiv.org/abs/2310.06786
finding: 从 Common Crawl 提取并保留 LaTeX 的 14.7B tokens 高质量数学网页数据集。
matters: 它是 DeepSeekMath 数据管线的种子语料，也是表 1 的对照语料之一。
```

```
id: rajbhandari2019zero
title: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
a: Rajbhandari, Rasley, Ruwase, He (2019), arXiv:1910.02054v3
url: https://arxiv.org/abs/1910.02054
finding: 混合精度 Adam 训练每个参数需 2+2+12=16 字节的模型状态显存。
matters: 本课估算 PPO 与 GRPO 的显存差（价值模型的代价）时用这条规则。
```

```
id: joulin2016fasttext
title: Bag of Tricks for Efficient Text Classification
a: Joulin, Grave, Bojanowski, Mikolov (2016), arXiv:1607.01759v3
url: https://arxiv.org/abs/1607.01759
finding: 提出用 n-gram 词袋与线性分类器实现的快速文本分类器 fastText。
matters: DeepSeekMath 用 fastText 从数百亿网页中召回数学页，本课的数据管线图需要交代分类器是什么。
```

```
id: yuan2023rft
title: Scaling Relationship on Learning Mathematical Reasoning with Large Language Models
a: Yuan, Yuan, Li, Dong, Lu, Tan, Zhou, Zhou (2023), arXiv preprint, arXiv:2308.01825v2
url: https://arxiv.org/abs/2308.01825
finding: 提出拒绝采样微调（RFT）：用模型自己采样出的正确解答扩充监督数据。
matters: RFT 是统一范式表 10 的一行，本课用它对照 GRPO 如何惩罚错误回答。
```

```
id: schulman2015gae
title: High-Dimensional Continuous Control Using Generalized Advantage Estimation
a: Schulman, Moritz, Levine, Jordan, Abbeel (2015), arXiv:1506.02438v6
url: https://arxiv.org/abs/1506.02438
finding: 提出用价值函数与指数加权 TD 残差构造优势估计的 GAE。
matters: PPO 的优势来自 GAE 与价值网络，本课用它说明 GRPO 省掉的是哪一块。
```

```
id: hf-dsmath-config
title: deepseek-ai/deepseek-math-7b-base (and -instruct, -rl) config.json
a: DeepSeek-AI (2024), Hugging Face model repository
url: https://huggingface.co/deepseek-ai/deepseek-math-7b-base/raw/main/config.json
kind: artefact
finding: 三个检查点均为 30 层、d=4096、32 头 MHA、FFN 11008、词表 102400、4K 上下文的 LLaMA 式结构。
matters: 论文没有给出架构表，本课的参数量、KV cache 与显存算例都以这里的数字为准。
```

```
id: gh-deepseek-math
title: deepseek-ai/DeepSeek-Math (GitHub repository)
a: DeepSeek-AI (2024), GitHub
url: https://github.com/deepseek-ai/DeepSeek-Math
kind: artefact
finding: 仓库发布了 Base/Instruct/RL 权重的使用说明与评测代码，推荐的提示后缀要求逐步推理并把答案放进 \boxed{}，但没有 RL 训练代码。
matters: 说明 GRPO 的若干实现细节（如 std 约定）无法从官方代码核对，只能参考第三方实现。
```

```
id: trl-grpo
title: Hugging Face TRL GRPOTrainer (trl/trainer/grpo_trainer.py)
a: Hugging Face (2026), GitHub, commit c8ef2433 (2026-09-23), VERSION 1.14.0.dev0
url: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
kind: artefact
finding: 组内优势用 Bessel 修正的样本标准差加 1e-4 归一化，KL 用 k3 且默认乘重要性比（use_bias_correction_kl=True），默认 beta=0、loss_type="dapo"。
matters: 本课给出 GRPO 代码级算例时以它为参照，并用它说明“论文公式”和“常用实现”的差别。
```

---

## 10. 补充核实（写课时追加）

（写课作者若核实了本笔记没有的事实，追加在此，附来源与位置。）

**写第 8 课时追加（2026-09-24）。** 一手来源都用 `arxiv.py grep` 读 LaTeXML 全文核对；派生数字在 Python 中重算，
生成图的脚本里有 `assert` 或打印（`deepseek/figures/gen/l08_*.py`）。

| # | 事实 | 来源与位置 | 引文（< 15 词）或核对方式 | 课中位置 |
|---|---|---|---|---|
| S1 | fastText：词（与 n-gram）向量查表后取平均，接线性分类器，softmax 输出；模型写作 $f(BAx_n)$，训练最小化 $-\frac1N\sum_n y_n\log f(BAx_n)$，$x_n$ 为归一化的特征词袋；n-gram 用哈希技巧存 | `joulin2016fasttext` v3 §2（式 1）、§2.2 | “word representations are then averaged into a text representation”；“fed to a linear classifier”；“using the hashing trick” | §2 式 (8.3)(8.4) |
| S2 | GAE：$\delta^V_t=r_t+\gamma V(s_{t+1})-V(s_t)$；$\hat A^{\mathrm{GAE}(\gamma,\lambda)}_t=\sum_{l\ge0}(\gamma\lambda)^l\delta^V_{t+l}$；$\lambda=1$ 无偏、$\lambda<1$ 有偏 | `schulman2015gae` v6 §3（GAE 定义，式 (14)(15) 之后）、§4 末 | “λ=1 gives an unbiased estimate of gγ, whereas λ<1 gives a biased estimate” | §3 式 (8.8)(8.9) |
| S3 | 原版 DPO 的梯度：$\nabla\mathcal L=-\beta\,\E[\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))(\nabla\log\pi(y_w\mid x)-\nabla\log\pi(y_l\mid x))]$，$\hat r_\theta=\beta\log\frac{\pi_\theta}{\pi_{ref}}$ | `rafailov2023dpo` v3 §4 | “higher weight when reward estimate is wrong”（公式下括注） | §7 式 (8.31) |
| S4 | DeepSeekMath §5.2.2 的原句带弯撇号：“RL enhances Maj@K’s performance but not Pass@K”；图 7 标题为 “RL enhances Maj@K but not Pass@K”；另有 “rendering the output distribution more robust” | `shao2024deepseekmath` v3 §5.2.2、图 7 标题 | 见左 | §8 |
| S5 | DeepSeek LLM 预训练收集 2T tokens | 第 6 课研究笔记 `lesson-06-doc.md` 第 6 条（引 `dsllm` 2401.02954 v1 §1：“We collect 2 trillion tokens for pre-training”） | 见左 | §1 |
| S6 | 高斯测试（$x\sim N(0,1)$，参考 $N(\mu,1)$）三个估计量的闭式：k1 相对标准差 $2/\mu$；k2 偏差/真值 $\mu^2/4$、相对标准差 $\sqrt{2+\mu^2}$；k3 方差 $e^{\mu^2}-1-\mu^2$ | 笔者推导（练习 8），`l08_kl_estimators.py` 用 200 点高斯–埃尔米特求积核对（$\mu$ = 0.1、0.5、1、1.2、1.5、2 全部一致） | $\mu=1$：2、1.7321、1.6950；$\mu=0.1$：20、1.4177、1.4166（与 §4.3 核查补记一致） | §5 图 8.9、练习 8 |
| S7 | k3 的相对标准差在 $\mu\approx1.205$（真 KL ≈ 0.726）以后超过 k2；$\mu=2$ 时 3.52 对 2.45 | 笔者计算（二分法，同上脚本） | — | §5 例题 |
| S8 | 图 7 Pass@K 交叉点（线性插值）：GSM8K 在 K=4 与 8 之间（RL−Instruct 从 +0.7 到 −0.6），MATH 在 K=16 与 32 之间，约 16→32 的 40% 处、80.3% | `l08_majk_passk.py`，数据为本笔记 §3.6 | — | §8 图 8.15 |
| S9 | 课中构造的算例（非论文数据）：基线方差例子 27.5625 → 0（两点分布）；GAE 玩具（$\gamma=1,\lambda=0.95$）$\hat A=(0.474,0.455,0.632,0.424,0.288)$；$G=5$、$\mathbf r=(1,0,1,1,0)$ 的优势 $+0.816/-1.225$（总体）、$+0.730/-1.095$（样本）、RLOO $+0.5/-0.75$；过程监督第二例 $o_1$ 优势 $(+0.557,+1.207)$、$o_2$ $(-0.557,-0.836,+0.743)$；两 token 词表 $\E[\partial_zk_3]=-0.30=\partial_z\KL[\pi_{ref}\Vert\pi_\theta]$，$\partial_z\KL[\pi_\theta\Vert\pi_{ref}]=-0.347$；1.3B 价值模型时 PPO 模型状态 159.0 GB（GRPO 省 13.1%）；$G=16$、长度 4096 的 KV 同为 30 GiB | Python 重算（本次会话 scratch 与 `l08_gae_credit.py`、`l08_group_adv.py`、`l08_process_adv.py`） | — | §3、§4、练习 4、6、7、10 |

| S10 | Dr. GRPO 对长度偏差的表述：去掉长度与 std 归一化；GRPO 的偏差“may lead to progressively longer incorrect responses”；§3.1：$\hat A<0$ 时长回答因 $\lvert o_i\rvert$ 大而受罚更少；表 2：所查的开源 PPO 实现都按回答长度归一化损失 | `liu2025drgrpo` v2 §1、§3.1、表 2 | “longer responses are penalized less due to their larger \|o_i\|” | §4 长度偏差 |
| S11 | DeepSeekMath 对“在线胜过离线”的解释：早期 actor 与 SFT 模型 “exhibit close resemblance”，后期实时采样优势更大；表 10 的 “Model” 指训练奖励模型打分，且 “The training data of the reward model is based on the rule judgment”；Online RFT “does not penalize incorrect responses”；各方法可看作 “direct or simplified RL techniques” | `shao2024deepseekmath` v3 §5.2.1、表 10 注、§1.2 | 见左 | §6、§7 |

**事实复核时追加（2026-09-24，Facts 审阅）。** 一手来源用 `arxiv.py grep` 读 LaTeXML 全文；Schulman 博客用 curl 取原页面后检索。

| # | 事实 | 来源与位置 | 引文（< 15 词）或核对方式 | 课中位置与处理 |
|---|---|---|---|---|
| S12 | §5.1.2 局限的第一条是定理的**非形式化**（informalization：把形式化的定理或证明改写成非形式的版本），不是“形式化” | `shao2024deepseekmath` v3 §5.1.2 | “such as informalization of theorems” | §2 原写“定理形式化”，已改 |
| S13 | R1 第一阶段 RL 与 R1-Zero 一样：每次 rollout 8,192 个输出随机分成 16 个 minibatch、只训 1 个 inner epoch；clip 比的作用：过低截断大量 token 的梯度、损害性能，过高可能不稳定 | `deepseekai2025r1` v2 §2.1 末段、§3.2.1 | “randomly split into 16 minibatches”；“the clip ratio plays a crucial role in training” | §4、练习 3（原文正确，补出处） |
| S14 | DeepSeek-Coder-v1.5 继续预训练的数据配比（表 9）：源代码 70%、Markdown 与 StackExchange 10%、与代码相关的自然语言 7%、与数学相关的自然语言 7%、中英双语自然语言 6% | `guo2024deepseekcoder` v2 §5 表 9 | 表格数值 | 图 8.1 的“代码为主”改为“源代码占 70%”，说明文字与参考文献卡片同步 |
| S15 | Schulman 博客说 k1 “negative for half of the samples”，这是一般性的描述，不是实验结果；λ=1 的理由是由 $\log x\le x-1$ 保证非负（最优 λ 依赖 p、q，难以解析求出） | `schulman2020kl` 原页面（2020/03/07） | “it’s negative for half of the samples” | §5 原把“一半为负”归到实验里，已改 |
| S16 | InstructGPT 的 KL 奖励系数固定为 β=0.02（附录 C.4），并在附录 E.7 扫描 β；PPO 每批 512 分 8 个 minibatch、单个 inner epoch；价值函数由 RM 初始化 | `ouyang2022instructgpt` v1 附录 C.4、E.7 | “with β=0.02 (see Equation 2)” | §3 警告框“系数固定”有据，未改 |
| S17 | DeepSeekMath SFT 数据：英文 GSM8K/MATH 标注工具集成解答，MathInstruct 子集与 Lila-OOD 用 CoT 或 PoT；中文 K-12 题目的解答**同时**标注 CoT 与工具集成两种格式 | `shao2024deepseekmath` v3 §3.1 | “annotated in both CoT and tool-integrated reasoning format” | §2 原写“每题配三种格式之一”，已改 |
| S18 | R1 附录 A.3：λ=0.95 是“大多数开源 PPO 实现的默认值”；结论是 PPO 调好也能接近 GRPO，但调参要额外算力，加上训练价值模型的显存与计算，GRPO 更实用 | `deepseekai2025r1` v2 附录 A.3 | “the default value in most open-source PPO implementations” | §3、§8 已补出处与措辞 |
| S19 | Math-Shepherd（2023-12）作者含 DeepSeek-AI 实习生与员工，早于 GRPO；“GRPO 是 DeepSeek 在后训练算法上的第一个原创贡献”无法成立 | `wang2024mathshepherd` v3 作者单位 | “Contribution during internship at DeepSeek-AI” | 封面“科技树位置”已改为“DeepSeek 自己提出的后训练算法” |
| S20 | DeepSeek LLM：多步学习率调度“allows for the reuse of training from the first phase”，便于续训 | 第 6 课研究笔记 `lesson-06-doc.md` 第 13 条（引 `dsllm` §2.3） | “offering a unique convenience for continual training” | §2 续训段补引 `[@dsllm]` |

**事实复核时追加（第二轮，2026-09-24，Facts 审阅）。** 方法同上（`arxiv.py grep` 读 LaTeXML 全文）。

| # | 事实 | 来源与位置 | 引文（< 15 词）或核对方式 | 课中位置与处理 |
|---|---|---|---|---|
| S21 | OpenWebMath 不是“用规则”筛出来的：五个阶段依次是数学字符串预过滤（tex class、`<math>` 标签、“mathjax”、常见 LaTeX 符号）→ 文本抽取 → 语言识别、困惑度（KenLM）过滤与 MathScore 分类器（用 fastText 训练）→ SimHash 去重 → 人工检查高频域名。DeepSeekMath 对第一轮之后的诊断是正例“lacks sufficient diversity” | `paster2023openwebmath` v1 §3（Figure 1 的五阶段、§3.5.1 MathScore）；`shao2024deepseekmath` v3 §2.1 第 3 段 | “We use FastText (Joulin et al., 2016) to train this model”；“trained on a set of positive examples that lacks sufficient diversity” | §2 “为什么要迭代”原写“OpenWebMath 本身是用规则从 Common Crawl 里筛出来的，它偏向某些网站、某些写法”（无出处），已改为有出处的描述；`paster2023openwebmath` 卡片 finding 同步 |
| S22 | Yue et al. 的最大 $k$ 因基准而异：`n` 取 pass@k 曲线最右端的 $k$，“typically 128, 256, or 1024”（MATH500、GSM8K 为 128，AMC23、AIME24 为 1024）；图 1 右的覆盖面指标是 pass@256 | `yue2025rlvr` v5 附录 A.2、图 1 说明 | “typically 128, 256, or 1024” | §8 原写“把 $K$ 推到 256 甚至更大”“$K\ge256$”，已改为“128、256 乃至 1024（视基准而定）”。本笔记 §7 E13 的“k 到 256 以上”应按此理解 |
| S23 | 表 4 † 行（衰减前检查点）MMLU 42.9%、BBH 42.9%、HumanEval 40.2%、MBPP 52.6% 两个 42.9 都是原文数字，不是转录错误 | `shao2024deepseekmath` v3 表 4 | LaTeXML 表格逐格读出 | §2 未改 |
| S24 | GAE 论文对 $\lambda=1$ 的方差：“has high variance due to the sum of terms”；$\lambda=0$ “typically has much lower variance” | `schulman2015gae` v6 §3 | 见左 | §3 式 (8.9) 说明“$\lambda=1$ 无偏但方差大”有据，未改 |
| S25 | RFT 的定义：用监督微调过的模型生成并收集正确推理路径，作为增广的微调数据 | `yuan2023rft` v2 摘要 | “generate and collect correct reasoning paths as augmented fine-tuning datasets” | §7 的 RFT 描述与参考文献卡片有据，未改 |
| S26 | 课文“三个检查点的 config.json 完全相同”与本笔记 §3.3 末注（RL 版多出 `moe_intermediate_size`、`rope_theta` 写成整数）不符 | 本笔记 §3.3 | — | §1 改为“给出的结构完全相同（RL 版只多一个不起作用的字段）” |
| S27 | PPO 每题采几个回答：论文图 4 的 PPO 一排只画了一个输出 $o$（本笔记 §5.1 对图 4 的描述）；“常见做法是 1 个”没有出处 | 本笔记 §5.1 | — | §8 总账表改为“不需要成组（论文图 4 画的是 1 个）” |
| S28 | Dr. GRPO 对长度偏差后果的措辞是 “may lead to progressively longer incorrect responses”（S10），是“可能”而非断言 | `liu2025drgrpo` v2 §1 | 见 S10 | §4 长度偏差框改为“可能让模型写出越来越长的错误答案”；参考文献卡片把“组均值基线乘 $G/(G-1)$ 就是留一基线”改为“优势乘 $G/(G-1)$ 就是 RLOO 的优势”（与 11.5 对式 (8.15) 标签的修正一致） |

**写课时发现的工具问题（不影响本笔记内容）**：写课会话的 scratch 目录与其他课的写作会话共用，同名临时文件会被互相覆盖；本课改用私有子目录。

---

## 11. 核查记录（对抗式复核，2026-09-24）

复核方法：独立重抓一手来源，不沿用研究笔记的缓存结论。DeepSeekMath 读 v3 LaTeXML 全文（`arxiv.py text 2402.03300`），公式逐符号对照其 LaTeX 源；
表 5 的灰色标记查 v3 HTML 的 `--ltx-fg-color:#808080`；v1/v2/v3 的表 5 差异查三份缓存 HTML；图 5–7 从 v3 PDF（`arxiv.org/pdf/2402.03300v3`）
第 19–21 页的 matplotlib 表单 XObject 重新解析路径坐标，并用网格线换算；图 1、图 3 从栅格 PNG 按像素读取。其余来源：R1 v2、V3.2 v1、Tang & Munos v1、
Dr. GRPO v2、Yue et al. v5、Math-Shepherd v3、DeepSeek-Coder v2、DPO、InstructGPT 的 LaTeXML 全文；PPO v2 的 PDF 文本（pypdf）；
Schulman 博客原页面；三个 `config.json`（HF raw）；TRL `c8ef2433` 的 `grpo_config.py`、`grpo_trainer.py`、`utils.py`、`VERSION`；
GitHub API 列出的 DeepSeek-Math 仓库目录；arXiv abs 页的提交历史。所有派生数字在 Python 中重算。

### 11.1 核对通过（与来源一致）

| # | 论断 | 来源与位置 | 结果 |
|---|---|---|---|
| 1 | 式 (3) GRPO 目标、式 (4) k3（含 “unbiased estimator (Schulman, 2020)”、“guaranteed to be positive”） | DSM v3 §4.1.1 LaTeX | 逐符号一致 |
| 2 | 式 (1)(2) PPO 与每 token KL 奖励 | DSM v3 §4.1.1 | 一致 |
| 3 | 结果监督 $\hat A=(r_i-\mathrm{mean})/\mathrm{std}$；过程监督后缀和 $\sum_{index(j)\ge t}$ | DSM v3 §4.1.2–4.1.3 | 一致 |
| 4 | 式 (12) 的 $o^-_{<t}$ 笔误；式 (13)(14) 无 $\beta$ 前因子；式 (19)(20)(21) | DSM v3 附录 A.1 | 一致 |
| 5 | 算法 1 第 11 行写 “Equation 21”；回放 10% 历史数据 | DSM v3 §4.1.4、算法 1 | 一致 |
| 6 | GRPO 超参：lr 1e-6、KL 0.04、64 个输出、最大长度 1024、batch 1024、single update；约 144K 问题；RM 基于 Base 7B、lr 2e-5 | DSM v3 §4.2 | 一致 |
| 7 | 表 10：GRPO/PPO 的 Reward Function 为 Model，RFT/DPO/Online RFT 为 Rule | DSM v3 表 10 | 一致 |
| 8 | 表 5 全部 DSM 行（CoT 与工具）、Gemini/GPT-4/Qwen/DeepSeek-LLM-Chat 行；灰色只有 Gemini Ultra 94.4%、Gemini Pro 86.5% | DSM v3 表 5 与 HTML 颜色 | 一致 |
| 9 | 摘要 51.7%（无工具、无投票）与 64 样本自洽 60.9% | DSM v3 摘要 | 一致 |
| 10 | §2.1：500,000 正负例、fastText 超参、40B HTML 页 vs top 40B tokens、>10% 域名判据、35.5M 页 / 120B、98%、10-gram 去污染 | DSM v3 §2.1 | 一致 |
| 11 | 表 1 全部数字（120.2B、13.6B 等）；表 2 DSM-Base 与 Minerva 540B；“77 times larger”；“9 times”；“almost 7 times” | DSM v3 表 1、表 2、§1、§2.3 | 一致；8.84×、8.16×、77.1× 重算无误 |
| 12 | §2.3：500B tokens、56/4/10/20/10 配比、峰值 lr 4.2e-4、batch 10M；§2.2.1 的 31.6% / 10.0% 调度 | DSM v3 §2.2.1、§2.3 | 一致 |
| 13 | 表 4 † 注 “the checkpoint right before learning rate decay”；† 行与最终 Coder-v1.5 行数字 | DSM v3 表 4 | 一致 |
| 14 | SFT 776K、500 步、batch 256、恒定 5e-5；76 个子主题 | DSM v3 §3 | 一致 |
| 15 | 图 5 是 DeepSeekMath-Instruct **1.3B**；图 7 标题 “RL enhances Maj@K but not Pass@K”（温度 0.7） | DSM v3 图 5、图 7 标题，§5.2.2 | 一致 |
| 16 | 图 7 的 48 个数据点；图 6 起点 82.9 / 46.8、各迭代起止步 0/1,100/2,700、x 轴刻度 0/1300/2300/…；图 5 的 8 条曲线汇总 | v3 PDF 矢量路径重解析 | 图 7 全部一致；图 6、图 5 见 11.2 |
| 17 | 版本日期 v1 2024-02-05、v2 02-06、v3 04-27；v1→v2 表 5 改动（SeaLLM-v2 新增；MAmmoTH 70B 72.4/21.1→76.9/41.8） | arXiv 提交历史；v1/v2/v3 HTML | 一致 |
| 18 | R1 v2：R1-Zero lr 3e-6、β 0.001、G 16、32,768→65,536（8.2k 步后）、10,400 步 ≈ 1.6 epoch、32 题 × 16 = 512、每 400 步换参考模型、8,192 输出分 16 个 minibatch；第一阶段 RL ε=10；序列级式 (1)–(3)；A.3 的 λ 0.95/1.0 与 Coder-V2-Lite；“Junxiao Song proposed the GRPO algorithm, implemented the initial version”；Nature 645:633–638 | R1 v2 §2.1、§3.2.1、附录 A.3、作者贡献；arXiv journal-ref | 一致 |
| 19 | V3.2 §3.1：$\hat A=R_i-\mathrm{mean}(\boldsymbol R)$、式 (7) IS 修正 k3、“unbounded weights”、数学域弱 KL 或不用 KL；目标仍有 $1/|o_i|$ | V3.2 v1 §3.1 | 一致 |
| 20 | Tang & Munos：Key insight 原句、k3 梯度对应 $\KL(\pi_{ref},\pi)$（reversed KL）、正文引作 “Schulman, 2017”、§5.1 低 KL 区推测 | 2506.09477 v1 §2.2、§3、§5.1 | 一致（另见 11.2 第 2 条） |
| 21 | Schulman 博客：发布日期 2020/03/07；两组实验表（20/1.42/1.42；0.25/1.73/1.7）；k2 的 $f''(1)=1$ 论证；$r\log r-(r-1)$；原句 “k3 has even lower standard deviation than k2 while being unbiased” | joschu.net 原页面 | 一致 |
| 22 | PPO v2：式 (7) 与 “pessimistic bound on the unclipped objective”、ε=0.2 示例；式 (8) KL 对 $\pi_{\theta_{old}}$、d_targ/1.5 与 ×1.5 规则；表 1（−0.39/0.76/0.82/0.70）；式 (9)–(12) | PPO v2 PDF | 一致 |
| 23 | config.json 三份：30 层、4096、32/32 头、11008、102400、4096 上下文、不共享嵌入、rl 版多出 `moe_intermediate_size`、`rope_theta` 为整数 | HF raw config.json | 一致；6,910,365,696 参数、480 KiB/token KV 重算无误 |
| 24 | TRL：`nanstd` 的 Bessel 修正、`/(std+1e-4)`（约第 2840 行）、默认 lr 1e-6、G 8、completion 512、温度 1.0、β 0.0、`num_iterations` 1、ε 0.2、`scale_rewards="group"`、`loss_type="dapo"`；VERSION 1.14.0.dev0 | TRL @ c8ef2433 | 一致（`use_bias_correction_kl` 见 11.2） |
| 25 | DeepSeek-Coder v2：§5 “2 trillion tokens” 与 §6 “2 billion tokens” 自相矛盾；只用下一词预测、4K 上下文；表 10 标 6.9B | 2401.14196 v2 | 一致 |
| 26 | Dr. GRPO 两种偏差原文、表 2（开源 PPO 实现的长度偏差）、附录 A 的 $G/(G-1)$ RLOO 等价；Yue et al. pass@256 原句与 NeurIPS 2025 Oral；Math-Shepherd 用 HE 训练 PRM；DPO 式 (7) 与带 $\beta$ 的梯度；InstructGPT 式 (2) 与 “per-token KL penalty from the SFT model” | 各论文全文 | 一致 |
| 27 | 数字重算：§4.1 全部优势值与闭式表、§4.2 过程监督算例、§4.3 k 表、§4.5 语料/步数/参数/KV/FLOPs、§4.6 显存、§4.7 差值、§4.8 pass@k、§4.9 玩具模型（精确多项分布枚举）、§4.10 | Python | 全部一致（仅 11.2 所列舍入） |
| 28 | k3 梯度恒等式：$\E_{\pi_\theta}[\nabla k_3]=\nabla\KL[\pi_{ref}\|\pi_\theta]$（误差 2e-11）、V3.2 修正后 $=\nabla\KL[\pi_\theta\|\pi_{ref}]$（7e-11）、$\E[k_3]=\KL[\pi_\theta\|\pi_{ref}]$ | 6 类 softmax 有限差分 | 一致 |
| 29 | DeepSeek-Math 仓库顶层只有 `evaluation/`、`replicate/` 等，没有训练代码 | GitHub API | 一致 |

### 11.2 改动（错误、夸大或需要限定的地方）

1. **TRL 的 KL 默认值（§3.5 表、E18、`trl-grpo` 卡片、§2.5 (f)）**：原文写“`use_bias_correction_kl=True` 时乘重要性比”，暗示这是可选开关。
   实际在 c8ef2433 的 `grpo_config.py` 里它**默认就是 True**，说明文字直接引用 V3.2。已改为“默认乘重要性比；但默认 β=0，KL 项不参与”。
   §2.5 (f) 补了一条命名旁证：TRL 把修正后的梯度叫 “reverse-KL gradient”，与 Tang & Munos 的叫法相反。
2. **Tang & Munos §4.1（C40、E1）**：原文只说“token 级损失至多给出部分梯度”。原论文对 k3 形式说得更重：它
   “does not produce even the partial gradient of any existing sequence-level divergence”。已补上。
3. **图 5 RFT 的 MATH 末 5 点均值**：重解析得 27.852，四舍五入应为 **27.9**，原表 27.8。已改。
4. **图 3 “~120B 处全体跳升”（§5.1、E20）**：看原图，MathPile 曲线是平的，没有跳升；跳升的是其余三条语料曲线。已改为“除 MathPile 外”。
5. **图 1 点值（E19、§8 第 9 条）**：原文说 Llemma-34B“约 31.5”、Mistral-7B“约 28”，并称是目测。按像素质心重读：31.4 与 28.2。
   另外发现 WizardMath-70B（22.3）与表 5 的 22.7% 也对不上。已改数字、补上这一条，并写明读取方法。
6. **§2.5 (b)“clip 不起作用”与 X7**：这个结论依赖把 “single update” 读成“整批只做一次优化器步”。附录 A.1.5 确实这样读，但原文注明这是
   “To simplify the analysis” 的假设，正文也没给 minibatch 数。已补上前提，并列入 §8 第 21 条。原文的“这也是论文不报告 ε 却不影响复现的原因之一”也加了同样的前提。
7. **E13 Pass@K 下降的幅度**：方向可以说，幅度要谨慎。按二项近似粗算（完整测试集，本笔记未核实实际题数），MATH 的 −0.7 pp 只有约 1 个标准误，
   GSM8K 的 −1.8 pp 约为 3 个。已补注，并列入 §8 第 22 条。
8. **§4.3 Schulman 复现表**：MC 末位依赖随机数生成器。用 `default_rng(0)` 抽 1,000 万个样本，μ=0.1 一行能逐位复现；μ=1 一行末位不同
   （笔者得 k1 +0.0006 / 1.999、k2 0.2493 / 1.732、k3 −0.0007 / 1.694）。换成 `np.random.seed(0)`，两行末位都会变。这属于 MC 波动，不影响结论。
   已补上 Gauss–Hermite 求积的精确值（例如 μ=1 时 k3 的 std/true 为 1.695）。
9. **§4.5 epoch 估算**：AlgebraicStack 10.3B、arXiv 28.0B 的分词器口径论文没说，与 120.2B（DeepSeek 分词器）未必同口径。已加口径提醒，1.94 与 1.79 epoch 只看量级。
10. **F7 取样点**：表中 7.937 / 8.063 对应的是 p=1/64，不是字面上的 0.0156（后者得 7.944 / 8.070）。已把 p 写成 1/64 与 63/64。

### 11.3 未改但需知道的

- 图 6 Iteration-0 的 GSM8K 终点重解析为 86.05，正好落在 86.0 与 86.1 之间，原表写 86.1，在 ±0.1 的精度内，未改。
- 图 6 其余起止值、最大值与原表一致（误差在 0.1 以内）；图 7 的 48 个点与原表逐一相同。
- 研究笔记 §8 原有的 20 条未核实项，本次没有找到能裁决的一手来源，全部保留。

### 11.4 数学复核补记（2026-09-24）

- **E13 / 11.2 第 7 条的标准误要改**：按完整测试集、$K=64$、两模型当独立样本，
  $\sqrt{(0.991\cdot0.009+0.973\cdot0.027)/1319}=0.517$ pp（GSM8K），不是 0.55；−1.8 pp 约为 **3.5** 个标准误（原写“约 3 个”）。
  MATH 为 $\sqrt{(0.875\cdot0.125+0.868\cdot0.132)/5000}=0.669$ pp，−0.7 约 1.05 个标准误，原结论不变。课文 §8 已改。
- **组均值基线“无偏”的前提**：只在组内各回答独立采样、且不计 $1/\lvert o_i\rvert$ 时成立。对不依赖 $o$ 的 $b$，
  $\E_o[\tfrac{b}{\lvert o\rvert}\nabla\log\pi_\theta(o)]=b\,\nabla\E_o[1/\lvert o\rvert]\ne0$，所以带长度平均时连留一基线也有偏（与 Dr. GRPO 的长度偏差一致）。课文 §4 已补。
- **R1 的 $\varepsilon=10$**：区间 $[-9,11]$ 下只有 $\hat A>0$ 且 $\rho>11$ 的 token 被截断；$\hat A<0$ 时 min 总取未裁剪项，永不裁剪。课文 §4 已补。
- **记号冲突（课文已改）**：控制变量系数 $\lambda$（Schulman 记号）与 GAE 的 $\lambda$ 冲突，改为 $\kappa$；fastText 的篇数 $N$ 与参数量 $N$ 冲突，改为 $N_{\text{doc}}$；
  Math-Shepherd (3)(4) 的 $s_i$、$N$、$j$ 与课中 (8.25)(8.26) 的“$i$ 回答、$j$ 步”相反，改为步 $j$、续写 $m$ 条、下标 $\ell$；标准化小节的 $\sigma$（标准差）与 sigmoid 冲突，改为 $\mathrm{std}$。

### 11.5 数学复核补记（第二轮，2026-09-24）

课中所有算例、表格、练习答案与 15 个生成脚本的数字在 Python 中逐一重算（含玩具模型的多项分布精确枚举、Pass@K 无偏性的二项求和验证），全部一致。改动：

- **难度偏差的“放大倍数”**（本笔记 §4.1 表下括注、图 8.8 底注）：除以 std 让**每道题自己的**总权重乘 $1/\sqrt{p(1-p)}$（$p=\tfrac12$ 时 ×2，$p=1/64$ 时 ×8.06）；
  所以极难题**相对中等题**只放大 $8.06/2\approx4.03$ 倍，不是 $1/\sqrt{p(1-p)}$ 倍。§4.1 括注“相对权重被放大 $1/\sqrt{p(1-p)}$ 倍”应按此理解。图 8.8 底注与说明文字已改，与练习 9 一致。
- **原版 DPO 的 GC 与 (8.28) 的约定**：原版 DPO 对整句求和、不除长度，所以 $\pm\beta\sigma(\hat r^--\hat r^+)$ 是乘在 $\sum_t\nabla\log\pi_\theta$ 上的系数；按 (8.28)“先乘 $1/\lvert o\rvert$”的约定应为 $\pm\beta\lvert o^\pm\rvert\sigma(\cdot)$。课文 (8.31) 说明已补。
- **PPO 重要性比只修正单个 token**：$\E_{\pi_{old}}[\rho_tA_t]$ 不修正前缀分布、$A_t$ 也按旧策略估，只在 $\theta\approx\theta_{old}$ 时可信（在 $\theta_{old}$ 处梯度恰为策略梯度）。课文 §3 已补。
- **Pass@K 估计式的“无偏”**：对 $c\sim\mathrm{Bin}(n,p_q)$，$\E[\binom{n-c}{K}/\binom{n}{K}]=(1-p_q)^K$（$n=64,K=16$ 数值核对到 $10^{-15}$）。课文 (8.34) 已补前提（$n\ge K$、样本独立同分布）。
- **记号**：显存 $M_{\text{PPO}}$ 与全课符号表 $M$（每 token 非嵌入 FLOPs）冲突，改为 $\mathrm{Mem}$；正态分布 $N(\cdot)$ 与参数量 $N$ 冲突，改为 $\mathcal N$；§3 起价值函数 $V$ 与 §1 词表大小 $V$ 同名，已加说明；fastText 的 $A,B,C$ 与优势、batch size、算力同名，已注明只在该节使用。
- **(8.15) 的标签**原写“组均值 = 留一基线 × 常数”，字面不成立（是**优势**之间差常数倍），已改。控制变量的定义改为“期望已知”（这里恰为 0）。
