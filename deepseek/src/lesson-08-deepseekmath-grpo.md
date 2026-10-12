---
title: 第 8 课 · DeepSeekMath 与 GRPO：去掉价值网络的强化学习
short: DeepSeekMath 与 GRPO
kicker: DeepSeek 科技树 · 第 8 课
description: DeepSeekMath 的两件事：用 fastText 迭代召回从 Common Crawl 挖出 120B tokens 数学语料；用 GRPO（组相对策略优化）代替 PPO，去掉价值网络，以组内奖励标准化作优势。逐项推导 GRPO 目标、k3 KL 估计器的无偏性与非负性、结果/过程监督、迭代 RL，以及 SFT、RFT、DPO、PPO、GRPO 的统一梯度系数。
lede: 2024 年 2 月，一个只有 7B 参数的开源模型在 MATH 基准上拿到 51.7%，离 GPT-4 的 52.9% 只差一个多百分点。它靠两件事：一条从网页里迭代“钓”出 120B tokens 数学文本的数据管线，和一个比 PPO 少养一个大模型的强化学习算法 **GRPO**。这一课把这两件事从头讲透，重点是 GRPO：每一项从哪来、为什么这样选、代价是什么，以及它后来在 DeepSeek 自己手里又被改了哪几处。
meta: 预计阅读 220 分钟，练习另需约 50 分钟 · 前置：第 3 课（RLHF 与 PPO），第 6 课；数学 M04、M05、M09
index: index.html
prev: lesson-07-deepseekmoe.html | 第 7 课 · DeepSeekMoE
next: lesson-09-v2-mla.html | 第 9 课 · MLA
---

## 这一课回答什么问题

[第 3 课](lesson-03-alignment-rlhf.html)讲完 RLHF 时留下了一张账单：PPO 训练时显存里要同时放四个模型，其中**价值网络**（critic）和策略模型一样大，而且要和策略一起训练，梯度和优化器状态一样不少。对一个想在数学推理上做强化学习的团队来说，这张账单很具体：7B 的策略，就要再养一个 7B 的“打分预测器”。

DeepSeekMath [@shao2024deepseekmath] 给出的答案叫 **GRPO**（Group Relative Policy Optimization，组相对策略优化）。它的核心只有一句话：对同一道题采样一组回答，用**这一组回答的平均奖励**当基线，而不是训练一个网络去预测基线。读完这一课，下面这些问题你都能自己回答：

- 价值网络在 PPO 里到底干什么？把它拿掉，损失的是什么，省下的又是多少显存？
- “组内标准化”具体怎么算？$G=4$ 份答卷、两对两错时，每份答卷的优势是多少？除以的是哪种标准差？
- GRPO 的目标函数有五六层求和与平均，每一层各管什么？
- KL 项为什么写成 $\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1$ 这个奇怪的样子？它“无偏”“非负”各是什么意思，在什么条件下成立，又在哪里悄悄失效？
- SFT、拒绝采样微调、DPO、PPO、GRPO 看上去是五种方法，为什么它们的梯度都能写成同一个形状？
- 论文说强化学习提升了 Maj@K 而没有提升 Pass@K，这说明 RL 在做什么？

## 这一课在科技树上的位置

这一课属于第 0 课的**主线 D：训练目标与后训练**：下一词预测 → SFT → RLHF/PPO（第 3 课）→ DPO（第 3 课）→ **GRPO（本课）** → R1 的纯 RL（[第 13 课](lesson-13-r1-reasoning.html)）→ V3.2 的规模化 RL（[第 15 课](lesson-15-v3-2-agents.html)）。前两课（第 6、7 课）讲的是 DeepSeek 在架构和规模化上的起点；这一课讲的 GRPO 是 DeepSeek 自己提出的**后训练算法**，此后的 R1 与 V3.2 都以它为核心算法，只是改了其中的几项（本课 [§8](#p8) 会列出改了哪几项）。数据管线部分则接着第 6 课“数据质量决定算力该怎么分”的话题。

上一课（DeepSeekMoE）回答的是“怎样用更少的激活参数得到更强的模型”；这一课换一个方向问：**模型的参数已经定了，怎样用更少的显存和更少的人工把它的推理能力再往上推一截？**

## 读完你能……

1. 画出 DeepSeekMath 的数据管线，说出每一轮迭代做了什么，并用真实数字算出语料的规模与占比。
2. 从策略梯度出发，说出基线、优势、价值网络、GAE 各自的作用，并估算 PPO 与 GRPO 的模型状态显存。
3. 给定一组奖励，手算 GRPO 的组内优势（总体与样本两种标准差），写出 GRPO 目标的每一项并解释它。
4. 推导 k3 KL 估计器的无偏性与非负性，算出它对梯度的贡献，并说明它“对值无偏、对梯度有偏”的问题与 V3.2 的修正。
5. 在统一范式下逐一推出 SFT、RFT、DPO、PPO、GRPO 的梯度系数，并用 Maj@K 与 Pass@K 的数据说明 RL 在这篇论文里改变了什么、没改变什么。

::: note 本课的记号
- $q$：一道题（prompt）；$o$：模型的一个回答；$o_{i,t}$：第 $i$ 个回答的第 $t$ 个 token；$o_{i,<t}$：它前面的所有 token；$\abs{o_i}$：第 $i$ 个回答的长度（token 数）。
- $\pi_\theta$：正在训练的策略（就是语言模型本身）；$\pi_{\theta_{old}}$：采样时用的那一版参数；$\pi_{ref}$：参考模型，训练中冻结，用来防止策略跑太远。
- $G$：每道题采样的回答个数（组大小）；$r_i$：第 $i$ 个回答的奖励；$\hat A_{i,t}$：优势估计。
- 两个比值，全课统一：**重要性比** $\rho_{i,t}=\pi_\theta(o_{i,t}\mid q,o_{i,<t})/\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})$；**参考比** $w_{i,t}=\pi_{ref}(o_{i,t}\mid q,o_{i,<t})/\pi_\theta(o_{i,t}\mid q,o_{i,<t})$。论文没有给它们起名字，PPO 原文把重要性比记作 $r_t(\theta)$，会和奖励 $r$ 撞车，所以本课改用 $\rho$ 和 $w$。
- $\varepsilon$：裁剪系数；$\beta$：KL 系数；$\widehat{\KL}_{i,t}$：单个 token 上的 KL **估计量**（不是精确的 KL）。
:::

[[legend]]

[[toc]]

# 问题：一个 7B 模型怎样逼近 GPT-4 | 两张账单：数学语料不够，PPO 太贵 | 15

先看结果，再问它是怎么来的。DeepSeekMath 论文的表 5 在“思维链、不用工具”的设定下比较了一批模型在两个数学基准上的成绩 [@shao2024deepseekmath]。**GSM8K** 是小学难度的数学应用题（名字里的 GSM 就是 grade school math），**MATH** 是高中数学竞赛题，难得多。表里的几个数：

| 模型 | 规模 | GSM8K | MATH |
|---|---|---|---|
| GPT-4 | 未公开 | 92.0% | 52.9% |
| Gemini Ultra | 未公开 | 94.4%（32 次投票） | 53.2% |
| Qwen | 72B | 78.9% | 35.2% |
| DeepSeek-LLM-Chat | 67B | 84.1% | 32.6% |
| DeepSeekMath-Instruct | 7B | 82.9% | 46.8% |
| DeepSeekMath-RL | 7B | **88.2%** | **51.7%** |

这张表里的成绩都是 Top1：模型只答一次、按这一次的答案算对错。唯一的例外是 Gemini Ultra 的 GSM8K 94.4%，表注说明它是 32 个候选答案的多数投票。DeepSeekMath-RL 的 51.7% 既没有调用代码工具，也没有投票；论文摘要另外报告，对 64 个采样做多数投票（自洽，self-consistency）能到 60.9% [@shao2024deepseekmath]。

一个 7B 模型在 MATH 上比 72B、67B 的通用开源模型分别高出 16.5 和 19.1 个百分点。这一课要拆开的就是这十几个百分点的来源。它们来自两步，每一步都对着一张具体的账单。

## 第一张账单：数学文本在网页里太少

模型的数学能力首先来自预训练时读过多少数学。问题是，公开的、成规模的数学网页语料很小。DeepSeekMath 用作种子的 **OpenWebMath** 是从 Common Crawl 里筛出来、保留了 LaTeX 公式的数学网页，它的论文自报 14.7B tokens [@paster2023openwebmath]；用 DeepSeek 的分词器重新数，是 13.6B（论文表 1）。当时更大的公开组合语料 Proof-Pile-2 也只有 51.9B（同样是论文表 1 用 DeepSeek 分词器数的）。对比一下第 6 课：DeepSeek LLM 的预训练收集了 2T tokens 的通用文本 [@dsllm]。公开数学语料比通用预训练语料小一到两个数量级：2T 是 OpenWebMath（14.7B）的约 136 倍、Proof-Pile-2 的约 39 倍。

[§2](#p2) 会看到，DeepSeekMath 用一个很便宜的线性分类器，在四轮迭代里从 Common Crawl 召回了 120B tokens 的数学网页，是 OpenWebMath 的约 9 倍。

## 第二张账单：PPO 要同时养四个模型

第二步是强化学习。回忆第 3 课的 RLHF：先用人类偏好数据训一个**奖励模型**（reward model，RM）给回答打分，再用 **PPO**（Proximal Policy Optimization，近端策略优化）[@schulman2017ppo] 让策略去追高分。PPO 训练时显存里同时有四个模型：

1. **策略模型**（policy）：正在训练的语言模型。
2. **参考模型**（reference）：训练开始时策略的一份冻结副本，用来算 KL 惩罚，防止策略为了追高分而跑偏。
3. **奖励模型**：冻结，给完整回答打分。
4. **价值模型**（value model，也叫 critic）：预测“从当前这个前缀出发，最终大概能拿多少分”，**和策略一起训练**。

前三个是“用奖励模型打分、带 KL 约束”的 RL 都需要的，第四个是 PPO 自己的选择。DeepSeekMath 指出，价值模型通常与策略模型规模相当，带来显著的显存与计算负担；而且在语言模型里奖励通常只给最后一个 token，要训练一个在**每个 token**上都准确的价值函数更难 [@shao2024deepseekmath]。

先算显存。第 1 课推导过参数量公式，这里用 DeepSeekMath 7B 的真实配置。

::: pre 混合精度 Adam 的“每参数 16 字节” | 学过 ZeRO 或第 12 课的可以跳过
用混合精度训练一个参数，显存里要存：一份 16 位的参数（2 字节）、一份 16 位的梯度（2 字节），以及 Adam 需要的三份 32 位状态：32 位的参数主副本、一阶动量、二阶动量（各 4 字节，共 12 字节）。合计 $2+2+12=16$ 字节。这是 ZeRO 论文 §3.1 的算法 [@rajbhandari2019zero]。**冻结**的模型只做前向，不需要梯度和优化器状态，用 BF16 存只要 2 字节/参数。这里算的都是“模型状态”，不含激活值和 KV cache。这条规则和 ZeRO 怎样把它切到多张卡上，见 [M10 §5](../math/m10-compute-arithmetic.html#p5)。
:::

DeepSeekMath 的三个发布检查点（Base、Instruct、RL）的 `config.json` 给出的结构完全相同（RL 版只多一个不起作用的字段）：30 层，$d=4096$，32 个注意力头、每头 $d_h=128$ 维，并且是标准多头注意力（KV 头也是 32 个），FFN 中间维 $d_{ff}=11008$（SwiGLU），词表 $V=102400$，输入嵌入与输出层不共享权重 [@hf-dsmath-config]。

::: eq DeepSeekMath 7B 的参数量 | (8.1)
$$N=\underbrace{2Vd}_{\text{嵌入与输出层}}+\;l\,\big(\underbrace{4d^2}_{\text{注意力}}+\underbrace{3\,d\,d_{ff}}_{\text{FFN}}+\underbrace{2d}_{\text{归一化}}\big)+d$$
注意力是 $W^Q,W^K,W^V,W^O$ 四个 $d\times d$ 矩阵；SwiGLU 的 FFN 有三个 $d\times d_{ff}$ 矩阵；每层两个 RMSNorm 各有 $d$ 个缩放参数；最后的 $+d$ 是输出前的最终 RMSNorm。和第 1 课的 $12ld^2$ 相比，差别在 FFN：三个矩阵、中间维 11008 而不是 $4d$。
:::

::: ex 用真实配置算出 6.91B
代入 $V=102400$、$d=4096$、$l=30$、$d_{ff}=11008$：

- 嵌入与输出层：$2\times102400\times4096=838{,}860{,}800$。
- 每层：$4\times4096^2+3\times4096\times11008+2\times4096=67{,}108{,}864+135{,}266{,}304+8{,}192=202{,}383{,}360$。
- 30 层：$6{,}071{,}500{,}800$；加上嵌入、输出层与最终归一化的 $4096$：$N=6{,}910{,}365{,}696\approx6.91\text{B}$。

论文和模型卡都只说“7B”，这个精确数是按配置算出来的。
:::

PPO 的策略和价值模型都要训练，参考模型和奖励模型冻结。这里假设四个模型一样大，与论文“规模相当”的说法一致。

::: eq 模型状态显存（笔者估算，不含激活与 KV cache） | (8.2)
$$\begin{aligned}\mathrm{Mem}_{\text{PPO}}&=\underbrace{2\times16N}_{\text{策略、价值：训练}}+\underbrace{2\times2N}_{\text{参考、奖励：冻结}}=36N\\ \mathrm{Mem}_{\text{GRPO}}&=\underbrace{16N}_{\text{策略：训练}}+\underbrace{2\times2N}_{\text{参考、奖励：冻结}}=20N\end{aligned}$$
单位是字节，$N$ 是单个模型的参数量。GRPO（[§4](#p4)）去掉了价值模型，其余三个模型不变。（显存记作 $\mathrm{Mem}$，因为全课符号表里的 $M$ 已留给第 6 课的“每 token 非嵌入 FLOPs”。）
:::

::: ex 四个模型的显存账
代入 $N=6.91\times10^9$：$\mathrm{Mem}_{\text{PPO}}=36\times6.91\times10^9$ 字节 $\approx248.8$ GB，$\mathrm{Mem}_{\text{GRPO}}=20\times6.91\times10^9$ 字节 $\approx138.2$ GB，少 $1-20/36\approx44.4\%$。[第 3 课 §6](lesson-03-alignment-rlhf.html#p6) 用同一条规则、按整数 $7\times10^9$ 记账，得到 252 GB 与 140 GB；两处的差别只在 $N$ 取名义值还是按配置算出的 6.91B，比例都是 $20/36$。

这是笔者按 16 字节规则做的量级估算，论文没有给显存数字；真实系统会用 ZeRO 切分、卸载、共享骨干等手段，绝对值会不同，但“少训练一个同规模模型，模型状态少掉近一半”这个比例关系不变。
:::

显存只是账单的一半。另一半是**训练价值模型本身有多难**。价值模型要对回答的每一个前缀 $o_{\le t}$ 预测“最终会拿多少分”，但它能拿来学习的信号只有回答结束时奖励模型给的一个数。一个 500 token 的解答，只有最后一个位置有“标准答案”，前面 499 个位置的目标都是靠自己的预测一步步传回来的（[§3](#p3) 会写出这个过程）。价值模型学不准，优势估计就不准，策略梯度的方向就歪了。

## 这一课的路线

::: fig figures/l08-lineage.svg | 图 8.1 DeepSeekMath 的模型谱系与每一步之后的成绩 | wide scroll
**看右边三组横条从上到下变长：继续预训练把 MATH 做到 36.2%，SFT 做到 46.8%，GRPO 再做到 51.7%，紫色条的末端逼近 GPT-4 的虚线。** 左列是检查点之间的继承关系，箭头旁写着每一步做了什么。Base 模型的成绩是少样本思维链（few-shot CoT），Instruct 与 RL 是不用工具、不投票的 Top1；空心菱形是 Minerva 540B 的成绩 [@shao2024deepseekmath]。DeepSeek-Coder-v1.5 继续预训练的数据 70% 是源代码（DeepSeek-Coder 论文表 9）；它的继续预训练量，该论文 §5 写作 2 trillion tokens、§6 结论写作 2 billion tokens，前后矛盾，所以图上不写这个数 [@guo2024deepseekcoder]。
:::

后面的页按图 8.1 从上往下走，但分量很不均匀：

- [§2](#p2) 数据：fastText 迭代召回、语料对比、继续预训练与 SFT。
- [§3](#p3) 从 PPO 出发：策略梯度、基线、优势、GAE、裁剪。这是第 3 课的复习，也是 GRPO 要改动的对象。
- [§4](#p4) GRPO 本体：组采样、组内标准化、目标函数逐项拆解。
- [§5](#p5) KL 项：k1、k2、k3 三个估计器，k3 的推导、性质与问题。
- [§6](#p6) 结果监督、过程监督、迭代 RL，以及训练设置与结果。
- [§7](#p7) 统一范式：五种方法的梯度系数逐一推导。
- [§8](#p8) RL 为什么有效（Maj@K 与 Pass@K），以及全课的总账。

::: key 本页要点
- DeepSeekMath-RL 7B 在 MATH 上 51.7%（Top1、无工具、不投票），GPT-4 为 52.9%；同表中的 67B、72B 开源通用模型只有 32.6%、35.2%。
- 两张账单：公开数学语料太小（OpenWebMath 13.6B tokens）；PPO 要同时放四个模型，其中价值模型与策略同规模且要训练。
- 按每可训练参数 16 字节、冻结参数 2 字节估算，7B（6.91B）规模下 PPO 的模型状态约 248.8 GB，去掉价值模型后约 138.2 GB，少 44.4%（笔者估算）。
:::

# 数据：用 fastText 从网页里召回数学 | 迭代召回管线、语料对比、继续预训练与 SFT | 25

这一页讲第一张账单怎么付。问题可以说得很具体：去重之后的 Common Crawl 有 **400 亿个网页**，其中数学网页可能不到千分之一。要把它们找出来，需要一个能给每个网页打“像不像数学”分数的打分器，而且这个打分器必须**便宜到能在几百亿个网页上各跑一遍**。用一个 7B 语言模型去读每个网页，算力上负担不起：每个网页平均几千 token，再乘以 400 亿。DeepSeekMath 的选择是一个线性分类器：fastText。

## fastText：一个只会“数词”的线性分类器

**含义。** fastText [@joulin2016fasttext] 是一个文本分类器：输入一篇文本，输出它属于各个类别的概率。这里只有两类：“数学”与“非数学”。

**理解。** 它不管词序，只看文本里出现了哪些词和短词组（n-gram），像一个只会“数词”的阅卷人：看到 `\frac`、`theorem`、`integral` 就倾向于判数学，看到 `price`、`celebrity` 就倾向于判非数学。每个词（和每个 n-gram）有一个学出来的向量；一篇文本的表示就是它所有词向量的**平均**；最后用一个线性层加 softmax 输出类别概率。整个模型没有注意力，没有深层网络，所以在普通 CPU 上每秒能处理大量文本。n-gram 很多，fastText 用哈希把它们映射到固定个数的桶里，以控制内存 [@joulin2016fasttext]。

这个“数词的阅卷人”比喻有它的边界。第一，阅卷人手里的“关键词表”不是人写的：每个词、每个 n-gram 的倾向都是从训练正例和负例里学出来的向量，所以训练数据偏向哪类网页，它就只会认哪类网页（下面“为什么要迭代”讲的正是这个问题）。第二，n-gram 让它看得到很短的词序（DeepSeekMath 用到 3 个词），但更长的结构它完全看不到：一段推导写得对不对、一页内容是不是真在讲数学而不只是提到几个术语，它分辨不了。它只负责“像不像”，不负责“好不好”。

**数学。** 设一篇文本有 $F$ 个特征（词与 n-gram），$\mathbf x\in\R^{H}$ 是它的归一化词袋向量（$H$ 是特征表的大小，第 $j$ 个分量是特征 $j$ 出现的次数除以 $F$）。fastText 论文把模型写成 $f(BA\mathbf x)$ [@joulin2016fasttext]：

::: eq fastText 分类器 | (8.3)
$$\hat{\mathbf y}=\softmax\big(B\,A\,\mathbf x\big),\qquad A\in\R^{k\times H},\quad B\in\R^{C\times k}$$
$A$ 是查找表（每一列是一个特征的 $k$ 维向量），$A\mathbf x$ 就是特征向量的平均；$B$ 是线性分类器；$C$ 是类别数（这里 $C=2$）。形状逐步是 $\mathbf x\in\R^{H}\to A\mathbf x\in\R^{k}\to BA\mathbf x\in\R^{C}$，softmax 把这 $C$ 个 logit 变成概率。DeepSeekMath 用的向量维度 $k=256$。$A$、$B$、$C$ 沿用 fastText 论文的记号，只在本节使用，与后文的优势 $A_t$、全课符号表里的 batch size $B$ 和算力 $C$ 都无关。
:::

训练就是最小化负对数似然（第 2 课的交叉熵）。对 $N_{\text{doc}}$ 篇带标签的文本（下标 doc 是为了和 [§1](#p1) 的参数量 $N$ 区分），$\mathbf y_n\in\{0,1\}^{C}$ 是第 $n$ 篇的独热标签：

::: eq fastText 的训练损失 | (8.4)
$$\mathcal L=-\frac1{N_{\text{doc}}}\sum_{n=1}^{N_{\text{doc}}}\mathbf y_n^\top\log\softmax\big(BA\mathbf x_n\big)$$
fastText 论文式 (1) 的写法（论文把篇数记作 $N$），$\log$ 逐元素作用；$\mathbf y_n^\top$（$1\times C$）乘 $C$ 维的对数概率，取出正确类别的那一个。
:::

::: ex 手算一个二维的 fastText 打分（笔者构造的数字）
取 $k=2$。假设训练后四个特征的向量是：`\frac` $(2,1)$，`theorem` $(1,2)$，`the` $(0,0)$，`price` $(-1,0)$。分类器 $B$ 的两行：“数学”$(1,1)$，“非数学”$(-1,-1)$。

网页甲只含这四个词各一次（为了能手算，这里只数单词、不数 n-gram，于是 $F=4$）。平均向量 $A\mathbf x=\big(\tfrac{2+1+0-1}{4},\tfrac{1+2+0+0}{4}\big)=(0.5,0.75)$；两个 logit 分别是 $0.5+0.75=1.25$ 与 $-1.25$；$p(\text{数学})=\dfrac{e^{1.25}}{e^{1.25}+e^{-1.25}}=\dfrac1{1+e^{-2.5}}\approx0.924$。

网页乙是“the price the price”：平均向量 $\big(\tfrac{0-1+0-1}{4},0\big)=(-0.5,0)$，logit $-0.5$ 与 $+0.5$，$p(\text{数学})=1/(1+e^{1})\approx0.269$。

一次打分只要把 $F$ 个向量加起来、做一次 $2\times k$ 的矩阵乘向量，运算量与网页长度成正比，和语言模型动辄每 token 数十亿次运算相比微不足道。
:::

DeepSeekMath 的 fastText 配置是：向量维度 256、学习率 0.1、词 n-gram 最大长度 3、最小词频 3、训练 3 个 epoch；第一轮的训练数据是从 OpenWebMath 随机取的 500,000 条正例，加上从 Common Crawl 随机取的 500,000 个网页作负例 [@shao2024deepseekmath]。

## 为什么要迭代：种子决定了能认出什么

一个分类器只认得出“长得像训练正例”的东西。OpenWebMath 本身是一串过滤器从 Common Crawl 里筛出来的：先查 LaTeX 命令、`<math>` 标签、“mathjax”这类数学字符串，再用一个同样基于 fastText 的“MathScore”分类器打分 [@paster2023openwebmath]；它收进来的网页有它自己的偏好，只用它训练的 fastText 会漏掉风格不同的数学网页。DeepSeekMath 对第一轮的诊断正是这样：仍有大量数学网页没被收集，主要原因是正例不够多样 [@shao2024deepseekmath]。它的做法是让召回结果反过来扩充种子，一共转了四圈。

::: fig figures/l08-data-pipeline.svg | 图 8.2 从 Common Crawl 迭代召回数学网页的管线 | wide scroll
**看左下角那支紫色的回边：人工只在“域名与 URL 路径”这一级做标注，就把一整批新正例送回种子，下一轮的分类器因此认得更多种数学网页。** 蓝色是模型步骤，黄色是语料（存储），紫色是人工步骤。据 DeepSeekMath 论文图 2 重绘，数字取自其 §2.1 [@shao2024deepseekmath]。
:::

每一圈的四步是：

1. **训练 fastText。** 用当前种子作正例、随机网页作负例。
2. **召回。** 给去重后的 Common Crawl 里每个网页打分，按分数排序，只保留分数最高的一部分。保留多少由预训练实验决定：论文试了保留前 40B、80B、120B、160B tokens 的效果，第一轮保留了前 40B tokens [@shao2024deepseekmath]。
3. **发现数学相关域名。** 把整个 Common Crawl 按基础 URL（域名）分组。如果某个域名下已经有超过 10% 的网页被召回，就把这个域名判为数学相关，例如 mathoverflow.net。
4. **人工标注 URL 路径。** 在这些域名里，人工标出与数学相关的 URL 路径，例如 mathoverflow.net/questions；这些路径下**尚未被召回**的网页被加进种子，作为新的正例。

第 4 步是整个管线里唯一的人工环节，而它标注的单位是“路径”而不是“网页”：标一条路径，就可能一次认领一大批网页（论文没有报告人工标了多少条路径、每条路径下有多少网页）。第 3 步则利用了召回结果本身的统计。论文只给出了 10% 这个阈值，下面是笔者对它的解读：一个域名下 10% 的网页都像数学，这个域名多半是数学网站，没被召回的那 90% 里有不少可能只是 fastText 还认不出来。

四轮之后，语料达到 **35.5M 个网页、120B tokens**。第四轮时，作者发现约 98% 的数据在第三轮就已经收集到了，于是停止迭代 [@shao2024deepseekmath]。论文表 1 用 DeepSeek 分词器统计的精确规模是 120.2B tokens。

::: warn 同一段原文里有两个“40B”
论文 §2.1 先说去重后的 Common Crawl 有 “40B HTML web pages”（400 亿个**网页**），几句之后又说第一轮保留 top 40B **tokens**。两者单位不同，大小也差得很远：前者是候选池，后者是保留量。图 8.2 上把两个单位都写全了。转述这篇论文时，“40B”后面必须带单位。
:::

::: ex 语料的几个比例（可用计算器复核）
- 平均每个网页的长度：$120\times10^9/(35.5\times10^6)\approx3{,}380$ tokens。
- 数学网页占候选池的比例：$35.5\times10^6/(40\times10^9)\approx0.0888\%$，约每 1,127 个网页里有 1 个。
- 与 OpenWebMath 相比：用同一个分词器，$120.2/13.6\approx8.84$ 倍（论文说 “9 times”）；如果用 OpenWebMath 自报的 14.7B，是 $120/14.7\approx8.16$ 倍。两个口径都“约 8 到 9 倍”，但写精确数字时要说明分母用的是哪个。
- 与 Proof-Pile-2 相比：$120.2/51.9\approx2.32$ 倍。
:::

**去污染**（decontamination）。训练语料里如果混进了评测题，基准分数就会虚高。DeepSeekMath 删掉了所有与 GSM8K、MATH、CMATH、AGIEval 等评测集有**完全相同 10-gram**（连续 10 个词）的文本段；评测文本短于 10 个词但至少 3 个词的，用精确匹配过滤 [@shao2024deepseekmath]。

## 语料好不好：让同一个小模型各读 150B tokens

“语料更大”不等于“语料更好”。为了比较质量，论文用第 6 课那个框架下的 1.3B 通用模型（DeepSeek-LLM 1.3B），在每个数学语料上各训练 150B tokens，其余设置完全相同：AdamW（$\beta_1=0.9,\beta_2=0.95$，权重衰减 0.1），多步学习率（2,000 步预热到峰值 $5.3\times10^{-4}$，训练进度 80% 时降到峰值的 31.6%，90% 时降到 10.0%），batch 4M tokens，上下文 4K [@shao2024deepseekmath]。

::: fig figures/l08-corpus-bench.svg | 图 8.3 四个数学语料各训 150B tokens 后，1.3B 模型的成绩（论文表 1） | wide scroll
**看绿色的最后一行：DeepSeekMath 语料在每个基准上都最长；再看 MathPile 那一行，它的 GSM8K、MATH、CMATH 都落在红色虚线（不做数学训练）附近甚至左边。** 最左栏是语料规模（DeepSeek 分词器）。所有成绩都是少样本思维链。数据取自 DeepSeekMath 论文表 1 [@shao2024deepseekmath]，条长由脚本按数值计算。
:::

图 8.3 有两点值得读出来。第一，DeepSeekMath 语料全面领先；论文的图 3 还显示，在训练到 50B tokens 时（相当于把 Proof-Pile-2 完整读一遍）它就已经胜过 Proof-Pile-2 [@shao2024deepseekmath]。第二，**规模本身就是质量的一部分**。每个语料都训 150B tokens，OpenWebMath 只有 13.6B，要重复约 11 轮，模型很快把能学的学完，曲线进入平台期；论文对基线语料的解释正是“太小、被重复多轮”。

MathPile（85% 以上来自 arXiv 论文）几乎没有帮助，甚至在 CMATH 上低于不做数学训练的基线。这和论文 §5.1 的另一个观察一致：只用 arXiv 语料训练，1.3B 模型和 7B 模型在各个数学基准上都没有明显提升，甚至下降 [@shao2024deepseekmath]。作者自己列出了这个结论的局限：没研究定理的非形式化（把形式化的定理或证明改写成自然语言）这类具体任务、没研究 arXiv 与其他数据组合的效果、没在更大规模上试。所以它是一个**实测**，不是“论文语料无用”的定论。

## 继续预训练：从代码模型出发，读 500B tokens

有了语料，下一步是训练 **DeepSeekMath-Base 7B**。它不是从头训练的，而是从 **DeepSeek-Coder-Base-v1.5 7B** 初始化，继续训练 500B tokens [@shao2024deepseekmath]。两个细节值得注意。

第一，起点是 Coder-Base-v1.5 **在学习率衰减之前的那个检查点**（论文表 4 的注），而不是训练完成的最终版本。这正是第 6 课讲的多步学习率调度的一个实际用处：DeepSeek LLM 论文说，多步调度可以复用第一阶段的训练，为继续训练提供了便利 [@dsllm]。笔者的理解是：在学习率还没降下来的检查点上接着训练，相当于把原来的预训练“续”下去，不必从一个已经退火（学习率降到很低）的点重新把学习率拉高；论文只交代了起点，没有解释这样选的理由。

第二，数据配比里数学只占一半多一点：

| 数据 | 占比 | tokens | 相对规模 |
|---|---|---|---|
| DeepSeekMath 语料 | 56% | 280B | $280/120.2\approx2.33$ 个 epoch |
| AlgebraicStack | 4% | 20B | 约 1.9 个 epoch（按 Proof-Pile-2 中的 10.3B 估算） |
| arXiv | 10% | 50B | 论文未说明用的是哪份 arXiv 数据 |
| GitHub 代码 | 20% | 100B | |
| Common Crawl 自然语言（中英） | 10% | 50B | |

其余设置沿用上面的 1.3B 实验，只把峰值学习率改为 $4.2\times10^{-4}$、batch 改为 10M tokens。

::: ex 预训练的步数与学习率台阶
$500\times10^9/(10\times10^6)=50{,}000$ 步。预热 2,000 步到 $4.2\times10^{-4}$；80% 处（第 40,000 步，读完 400B tokens）降到 $4.2\times10^{-4}\times0.316\approx1.33\times10^{-4}$；90% 处（第 45,000 步）降到峰值的 10%，即 $4.2\times10^{-5}$。

两个台阶的比例 31.6% 与 10% 并不随意：$1/\sqrt{10}\approx0.316$，两次各乘约 $1/\sqrt{10}$，合起来正好是 $1/10$。这是笔者的观察，论文没有解释这个数。
:::

为什么从代码模型出发？论文 §5.1 用 1.3B 模型做了对照 [@shao2024deepseekmath]：

| 训练方式（DeepSeek-LLM 1.3B） | GSM8K | MATH |
|---|---|---|
| 先 400B 通用文本，再 150B 数学 | 19.1% | 14.4% |
| 先 400B 代码，再 150B 数学 | **21.9%** | **15.3%** |
| 代码与数学一阶段混合训练 | 17.6% | 12.1% |

两阶段“先代码后数学”最好。一阶段混训缓解了{{灾难性遗忘::模型在新数据上继续训练时，把前一阶段学会的能力（这里是代码能力）忘掉的现象}}（catastrophic forgetting）、提升了带 Python 工具的推理（GSM8K+Python 19.7%，MATH+Python 13.5%），但损害了不用工具的数学推理。作者据此认为代码训练对数学推理有益。数学训练还顺带提高了通用推理：DeepSeekMath-Base 7B 的 MMLU、BBH 是 54.9%、59.5%，高于 Coder-Base-v1.5 最终版的 49.1%、55.2%，更高于真正的起点（衰减前检查点）的 42.9%、42.9% [@shao2024deepseekmath]。这些都是**实测**，只在这篇论文的规模与数据上成立。

结果：DeepSeekMath-Base 7B 在少样本思维链设定下 GSM8K 64.2%、MATH 36.2%，超过了 Minerva 540B 的 58.8%、33.6%；论文称后者“大 77 倍”，$540/7\approx77.1$ [@shao2024deepseekmath]。

## 监督微调：从 Base 到 Instruct

强化学习之前还有一步 SFT（监督微调，第 3 课）。DeepSeekMath 的 SFT 数据有 776K 条，覆盖中英文数学题，解答有三种格式：思维链（chain of thought，CoT，用文字一步步推）、思维程序（program of thought，PoT，写一段程序算出答案）、工具集成推理（文字推理中穿插代码执行）[@shao2024deepseekmath]。英文部分包括给 GSM8K、MATH 新标注的工具集成解答，以及用 CoT 或 PoT 解答的 MathInstruct 子集和 Lila-OOD 训练集；中文部分是 K-12 题目，覆盖 76 个子主题，每题的解答同时标注了 CoT 与工具集成两种格式。

训练把样本随机拼接到最长 4K tokens，训练 500 步、batch 256、恒定学习率 $5\times10^{-5}$。这是一个很轻的微调：$500\times256\times4096\approx5.2\times10^8$，也就是说最多读了约 0.52B tokens（假设每个拼接序列都填满 4K），约为继续预训练（500B）的千分之一。得到的 **DeepSeekMath-Instruct 7B** 在 GSM8K、MATH 上是 82.9%、46.8%。

下一页开始讲本课的主角：从 Instruct 到 RL 的那 5 个百分点是怎么来的。

::: key 本页要点
- fastText = 词与 n-gram 向量取平均 + 线性分类器 + softmax；便宜到能给 400 亿个网页各打一次分。
- 迭代召回：训分类器 → 召回并按分数截断 → 按“>10% 已召回”找数学域名 → 人工标 URL 路径扩种子；四轮后 35.5M 网页、120B tokens，第四轮约 98% 已在第三轮收到。
- 同样训 150B tokens，DeepSeekMath 语料在全部基准上领先；小语料被重复多轮后很快进入平台期；只用 arXiv 的语料几乎无效（实测）。
- Base 7B 从 Coder-v1.5 的衰减前检查点出发，读 500B tokens（数学 56%，约 2.33 个 epoch），MATH 36.2%；SFT 776K 条、500 步后 Instruct 为 46.8%。
:::

# 从 PPO 出发：策略梯度、优势与价值网络 | 第 3 课的复习，也是 GRPO 要改的地方 | 30

GRPO 不是一个从零发明的算法，它是在 PPO 上改了两处。要看清改了什么，得先用 DeepSeekMath 的记号把 PPO 完整写一遍。这一页是[第 3 课](lesson-03-alignment-rlhf.html)和 [M09 §1–§6](../math/m09-rl-math.html#p1) 的复习，熟悉的读者可以快速浏览，但本页的记号和几个结论（基线不改变梯度的期望、GAE、裁剪）后面每一页都要用。

## 语言模型就是一个策略

**含义。** 强化学习里的**策略**（policy）是一个“看到当前局面，给出每个动作的概率”的函数。

**理解。** 让语言模型解一道数学题，就是在玩一个一步一步落子的游戏：局面（状态）是题目加上已经写出的部分，动作是下一个 token，写出结束符时一局结束。结束后奖励模型给整篇解答打一个分。模型每一步都按自己的概率分布选下一个 token，这正是一个策略。

**数学。** 记题目为 $q$，回答为 $o=(o_1,\dots,o_{\abs{o}})$。第 $t$ 步的状态是 $(q,o_{<t})$，动作是 $o_t$，策略是 $\pi_\theta(o_t\mid q,o_{<t})$，即语言模型在这个前缀下给第 $t$ 个 token 的概率。整个回答的概率是逐个 token 概率的乘积，所以它的对数是逐个 token 对数概率之和：

$$
\pi_\theta(o\mid q)=\prod_{t=1}^{\abs{o}}\pi_\theta(o_t\mid q,o_{<t}),\qquad \log\pi_\theta(o\mid q)=\sum_{t=1}^{\abs{o}}\log\pi_\theta(o_t\mid q,o_{<t}).
$$

强化学习要最大化的是期望奖励：题目从题库分布 $P(Q)$ 里抽，回答从当前策略里采样，奖励 $r(q,o)$ 由奖励模型或规则给出。

::: eq 强化学习的目标 | (8.5)
$$\mathcal J(\theta)=\E_{q\sim P(Q),\;o\sim\pi_\theta(\cdot\mid q)}\big[\,r(q,o)\,\big]$$
难点在于采样分布本身依赖 $\theta$：不能直接把梯度符号挪进期望里。
:::

## 策略梯度：对数导数技巧

::: pre 对数导数技巧 | 在 M09 或第 3 课见过 $\nabla\pi=\pi\nabla\log\pi$ 的可以跳过
对任何取正值的可导函数 $f(\theta)$，链式法则给出 $\nabla_\theta\log f=\frac{\nabla_\theta f}{f}$，两边乘 $f$ 得 $\nabla_\theta f=f\,\nabla_\theta\log f$。用在概率上就是 $\nabla\pi_\theta(o)=\pi_\theta(o)\nabla\log\pi_\theta(o)$。它的用处是：把“对概率求导”变成“概率乘以某个量”，于是求和又能写回成期望，可以用采样来估计。完整的策略梯度推导见 [M09 §2](../math/m09-rl-math.html#p2)。
:::

固定一道题 $q$，把期望写成对所有可能回答的求和（回答的集合有限，只是非常大）：

::: deriv 策略梯度（REINFORCE）
$\nabla_\theta\mathcal J=\nabla_\theta\sum_{o}\pi_\theta(o\mid q)\,r(q,o)$ || 期望 = 概率加权求和；奖励不依赖 $\theta$
$=\sum_{o}\nabla_\theta\pi_\theta(o\mid q)\,r(q,o)$ || 有限求和，梯度可以逐项求
$=\sum_{o}\pi_\theta(o\mid q)\,\nabla_\theta\log\pi_\theta(o\mid q)\,r(q,o)$ || 对数导数技巧
$=\E_{o\sim\pi_\theta}\big[\,r(q,o)\,\nabla_\theta\log\pi_\theta(o\mid q)\,\big]$ || 又变回了期望，可以用采样估计
$=\E_{o\sim\pi_\theta}\Big[\,r(q,o)\sum_{t=1}^{\abs{o}}\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\Big]$ || 整句的对数概率是逐 token 之和
:::

::: eq 策略梯度 | (8.6)
$$\nabla_\theta\mathcal J(\theta)=\E_{q,\;o\sim\pi_\theta}\Big[\,r(q,o)\sum_{t=1}^{\abs{o}}\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\Big]$$
读法：采一个回答，把它每个 token 的对数概率都往上推，推的力度与奖励成正比。
:::

(8.6) 说的事情很朴素：奖励高的回答，让它的每个 token 以后更容易被写出来。这个估计是**无偏**的（期望正好等于真梯度），但**方差很大**：奖励只看整篇回答，一篇 500 token 的解答里，真正决定对错的可能只有几个 token，可 (8.6) 把同一个奖励乘给了全部 500 个。

## 基线：减去一个数，期望不变，方差变小

先证明一个小事实：对数概率的梯度 $\nabla_\theta\log\pi_\theta(o\mid q)$（统计里叫**得分函数**，score function）在自己的分布下期望为零。

::: deriv 得分函数的期望为零
$\E_{o\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(o\mid q)\big]=\sum_o\pi_\theta(o\mid q)\,\nabla_\theta\log\pi_\theta(o\mid q)$ || 期望的定义
$=\sum_o\nabla_\theta\pi_\theta(o\mid q)$ || 对数导数技巧反着用
$=\nabla_\theta\sum_o\pi_\theta(o\mid q)=\nabla_\theta1=0$ || 概率之和恒为 1，与 $\theta$ 无关
:::

所以对任何**不依赖于这个回答 $o$** 的数 $b$（可以依赖题目 $q$），$\E[\,b\,\nabla\log\pi_\theta(o\mid q)]=b\cdot0=0$。从奖励里减去它，梯度的期望不变：

::: eq 带基线的策略梯度 | (8.7)
$$\nabla_\theta\mathcal J(\theta)=\E_{q,\;o\sim\pi_\theta}\Big[\,\big(r(q,o)-b(q)\big)\sum_{t}\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\Big]$$
$b(q)$ 叫**基线**（baseline）。条件是它不能看这个回答本身；[§4](#p4) 会看到 GRPO 的基线在这一点上有一个小瑕疵。
:::

期望不变，方差却可以差很多。一个两种回答的例子最清楚。

::: ex 基线把方差从 27.56 降到 0（笔者构造的数字）
设某道题只有两种可能的回答：答对（概率 $p=0.5$）与答错。策略由一个参数 $z$ 控制，$p=\sigma(z)$。答对时 $\partial_z\log p=1-p=0.5$，答错时 $\partial_z\log(1-p)=-p=-0.5$。

奖励：答对 1 分、答错 0 分。不减基线，单样本梯度估计要么是 $1\times0.5=0.5$，要么是 $0\times(-0.5)=0$，各一半概率：均值 0.25，方差 $\tfrac12(0.25^2+0.25^2)=0.0625$。

现在把两个奖励都加 10（答对 11 分、答错 10 分）。这不改变“哪个更好”，真梯度仍是 0.25。但单样本估计变成 $11\times0.5=5.5$ 或 $10\times(-0.5)=-5$：均值仍是 0.25，方差却是 $\tfrac12(5.25^2+5.25^2)\approx27.56$，大了 441 倍。

减去基线 $b=10.5$（两种奖励的期望）：估计变成 $(11-10.5)\times0.5=0.25$ 或 $(10-10.5)\times(-0.5)=0.25$，每次都正好等于真梯度，方差为 0。基线的作用就是把“奖励的绝对水平”减掉，只留下“比平均好多少”。
:::

最自然的基线是“在这个局面下平均能拿多少分”。它有名字：**价值函数**（value function）$V(s)$（这是强化学习的通用记号，与 [§1](#p1) 的词表大小 $V$ 无关；本课从这里起 $V$ 只指价值函数），表示从状态 $s$ 出发、按当前策略继续写下去，期望能拿到的总奖励。“这一步实际走出来的结果”减去“这个局面的平均水平”，叫**优势**（advantage）：正的优势说明这个动作比平均好，应该加强；负的说明比平均差，应该削弱。

::: def 价值函数与优势
记第 $t$ 步的状态 $s_t=(q,o_{<t})$（题目加上前 $t-1$ 个 token），$r_t$ 是写下第 $t$ 个 token 后拿到的即时奖励（一个实数），$\gamma\in(0,1]$ 是折扣。对策略 $\pi$：

$$\begin{aligned}V^{\pi}(s_t)&=\E_{\pi}\Big[\textstyle\sum_{l\ge0}\gamma^{l}\,r_{t+l}\ \Big|\ s_t\Big]\\ Q^{\pi}(s_t,o_t)&=\E_{\pi}\Big[\textstyle\sum_{l\ge0}\gamma^{l}\,r_{t+l}\ \Big|\ s_t,\,o_t\Big]\\ A^{\pi}(s_t,o_t)&=Q^{\pi}(s_t,o_t)-V^{\pi}(s_t)\end{aligned}$$

$V^\pi$（上文的 $V(s)$）是“站在这个局面、之后按 $\pi$ 写下去”的期望回报；$Q^\pi$ 多固定了一件事：这一步写的是 $o_t$；两者之差 $A^\pi$ 就是“这一步写 $o_t$ 比这个局面下的平均写法好多少”。数学题只在末尾给 0/1 奖励、$\gamma=1$ 时，$V^\pi(s_t)$ 就是“从这个前缀写下去最终答对的概率”。$V^\pi(s_t)$ 不看 $o_t$，把上面“得分函数期望为零”的推导换成单个 token 的分布 $\pi_\theta(\cdot\mid s_t)$ 再做一遍，就知道拿它当基线同样不改变梯度的期望。详见 [M09 §3–§4](../math/m09-rl-math.html#p3)。
:::

## 价值网络与 GAE：把终点的一个奖励分到每个 token

PPO 不满足于每道题一个基线，它想给**每个 token**一个基线，这样每个 token 都有自己的优势 $\hat A_t$，能区分“这一步是好棋还是坏棋”。为此它训练一个**价值网络** $V_\psi$（$\psi$ 是它的参数），输入状态 $s_t=(q,o_{<t})$，输出一个实数，作为 $V^\pi(s_t)$ 的估计。有了它，就可以一步一步地核算“这一步让前景变好了多少”：

::: eq TD 残差 | (8.8)
$$\delta_t=r_t+\gamma\,V_\psi(s_{t+1})-V_\psi(s_t)$$
$r_t$ 是第 $t$ 步拿到的即时奖励，$\gamma\in(0,1]$ 是折扣。$\delta_t>0$：走完这一步后，前景比走之前估计得更好。最后一步之后 $V_\psi=0$。
:::

为什么说 $\delta_t$ 是“这一步的功劳”？在语言模型里，下一个状态 $s_{t+1}$ 就是 $s_t$ 接上 $o_t$，是确定的；即时奖励 $r_t$ 也由 $s_t$ 和 $o_t$ 决定。所以只要价值网络是准的（$V_\psi=V^\pi$），$r_t+\gamma V^\pi(s_{t+1})$ 就正好等于 $Q^\pi(s_t,o_t)$（先拿这一步的奖励，再从下一个局面按平均水平走下去），$\delta_t$ 也就正好等于优势 $A^\pi(s_t,o_t)$。麻烦全在“价值网络准不准”：它不准，$\delta_t$ 就把它的误差原样带进优势。

单个 $\delta_t$ 只看一步，价值网络不准它就不准；把后面所有步的 $\delta$ 都加起来，又会把后面所有步的随机性都带进来。**GAE**（generalized advantage estimation，广义优势估计）[@schulman2015gae] 用一个参数 $\lambda$ 在两者之间折中：

::: eq GAE | (8.9)
$$\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}=\delta_t+\gamma\lambda\,\hat A_{t+1}^{\text{GAE}}$$
$\lambda=0$ 时只用 $\delta_t$：方差小，但价值网络的误差全部进入优势。$\lambda=1$ 时，中间各项的 $V$ 两两抵消，等于“之后所有奖励的折扣和减去 $V_\psi(s_t)$”；GAE 论文指出这时的估计无偏，但方差大 [@schulman2015gae]。
:::

在语言模型的 RLHF 里，即时奖励 $r_t$ 在最后一个 token 是奖励模型的分数，每个 token 还要扣一点 KL 惩罚（下面的 (8.12)）。图 8.4 把这个过程摊开在一个五个 token 的回答上。

::: fig figures/l08-gae-credit.svg | 图 8.4 PPO 怎样把末尾的一个奖励分摊成每个 token 的优势（笔者构造的玩具数字） | wide scroll
**看第三行的 $\delta_2=-0.146$：价值网络认为第 2 个 token 让前景变差了，但经过 GAE 往回累加，它的优势（第四行）仍然是正的。** 第一行：前四个 token 只有很小的 KL 项 $-\beta k_1$（可正可负：第 2 个 token 的 $\log(\pi_\theta/\pi_{ref})<0$，反而得到 +0.004），最后一个 token 另外拿到奖励模型的 1.0 分。第二行是价值网络的预测。最后一行是对照：GRPO 不要价值网络，整条回答的每个 token 共用同一个优势。所有数字由脚本 `l08_gae_credit.py` 算出。
:::

::: ex 手算图 8.4 的最后三个 token
取 $\gamma=1$、$\lambda=0.95$、$\beta=0.04$。最后一个 token：$r_5=1.0-0.04\times0.3=0.988$，$V(s_6)=0$，$\delta_5=0.988+0-0.70=0.288$，所以 $\hat A_5=0.288$。倒数第二个：$r_4=-0.04\times0=0$，$\delta_4=0+0.70-0.55=0.150$，$\hat A_4=\delta_4+0.95\times\hat A_5=0.150+0.2736=0.4236$。再往前一步：$r_3=-0.04\times0.5=-0.02$，$\delta_3=-0.02+0.55-0.30=0.230$，$\hat A_3=0.230+0.95\times0.4236\approx0.632$。每一步只用到下一步的结果，所以 GAE 是从后往前一遍扫完的。
:::

这套机制的代价在 [§1](#p1) 已经算过：价值网络和策略一样大，要训练，占掉近一半的模型状态显存。它还有一个质量上的风险：**优势的好坏取决于价值网络准不准**。价值网络用平方误差去拟合回报的估计（PPO 论文的式 (9) 把这一项和裁剪目标、一个熵奖励合成一个总损失 [@schulman2017ppo]），可它的训练目标本身又部分来自自己的预测。DeepSeek-R1 论文附录 A.3 的对照实验给了一个具体例子：GAE 取 $\lambda=0.95$（论文说这是大多数开源 PPO 实现的默认值）时 PPO 明显差于 GRPO，把 $\lambda$ 调到 1.0 后才接近 GRPO [@deepseekai2025r1]。$\lambda=1$ 恰恰是最少依赖价值网络的那一端。

## 重要性比与裁剪

生成回答很贵：每道题要自回归地写上千个 token。所以 PPO 希望**一批采样用好几次**：采一次，在同一批数据上做若干次梯度更新。问题是，做完第一次更新后，参数已经从 $\theta_{old}$ 变成了 $\theta$，手里的样本却是旧策略采的，(8.7) 的期望对不上了。

::: pre 重要性采样 | 知道 $\E_{x\sim\pi_1}[f]=\E_{x\sim\pi_2}[\tfrac{\pi_1}{\pi_2}f]$ 的可以跳过
想算分布 $\pi_1$ 下的期望，手里却只有分布 $\pi_2$ 的样本，可以给每个样本乘一个权重 $\pi_1(x)/\pi_2(x)$：$\E_{x\sim\pi_1}[f(x)]=\sum_x\pi_1(x)f(x)=\sum_x\pi_2(x)\frac{\pi_1(x)}{\pi_2(x)}f(x)=\E_{x\sim\pi_2}\big[\tfrac{\pi_1(x)}{\pi_2(x)}f(x)\big]$。要求 $\pi_2(x)>0$ 的地方覆盖 $\pi_1(x)>0$ 的地方。权重偏离 1 越远，估计的方差越大。见 [M04 §7](../math/m04-probability.html#p7)；用在策略上的版本见 [M09 §5](../math/m09-rl-math.html#p5)。
:::

把重要性采样用在每个 token 上，就得到 PPO 的**重要性比**：

::: eq 重要性比 | (8.10)
$$\rho_t(\theta)=\frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{\theta_{old}}(o_t\mid q,o_{<t})}$$
$\rho_t=1.5$：当前策略写出这个 token 的概率是采样时的 1.5 倍。在 $\theta=\theta_{old}$ 处，$\rho_t=1$。
:::

用 $\rho_tA_t$ 代替 $A_t$ 做优化目标，梯度就按新旧策略在**这一个 token** 上的差别修正了回来。这只是一个近似：前缀 $o_{<t}$ 仍是旧策略写的，$A_t$ 也是按旧策略估的，这两处都没有修正，所以只有 $\theta$ 离 $\theta_{old}$ 不远时它才可信（在 $\theta=\theta_{old}$ 处 $\rho_t=1$，它的梯度正是以 $A_t$ 为权重的策略梯度 $\sum_tA_t\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})$）。而 $\rho_t$ 可能离 1 很远：一次更新走得太大，旧样本就不再能代表新策略。PPO 的办法是**裁剪**（clip）：把 $\rho_t$ 限制在 $[1-\varepsilon,1+\varepsilon]$ 内，再和不裁剪的版本取较小者。DeepSeekMath 把 PPO 写成下面的形式（论文式 (1)，这里换成本课记号）[@shao2024deepseekmath]：

::: eq PPO 的裁剪目标（DeepSeekMath 式 1） | (8.11)
$$\mathcal J_{\text{PPO}}(\theta)=\E_{q\sim P(Q),\,o\sim\pi_{\theta_{old}}(\cdot\mid q)}\frac{1}{\abs{o}}\sum_{t=1}^{\abs{o}}\min\Big[\rho_t(\theta)A_t,\;\clip\big(\rho_t(\theta),1-\varepsilon,1+\varepsilon\big)A_t\Big]$$
$A_t$ 由 GAE (8.9) 算出。$\clip(x,a,b)$ 把 $x$ 截到区间 $[a,b]$ 内。
:::

::: fig figures/l08-clip.svg | 图 8.5 一个 token 的 PPO 裁剪目标随比值 ρ 的变化（ε = 0.2） | wide scroll
**看红色的平台段：好 token 的概率涨到 1.2 倍以上、或坏 token 的概率压到 0.8 倍以下之后，目标不再变化，梯度为 0，这一步就不会再往这个方向推。** 黑线是 (8.11) 里 min 的结果，灰色虚线是不裁剪的 $\rho A$；黑线处处不高于虚线。蓝点是 $\rho=1$：[§4](#p4) 会看到，DeepSeekMath 的实际训练恰好只在这一点上求梯度。$\varepsilon=0.2$ 是 PPO 论文举例用的值 [@schulman2017ppo]。
:::

取 min 的效果，用 PPO 论文的话说，是让最终目标成为未裁剪目标的一个“悲观下界”（pessimistic bound）[@schulman2017ppo]：概率比往“有利”的方向走出区间后，收益被封顶；往“不利”的方向走（例如坏 token 的概率反而升高，$A<0$ 且 $\rho>1.2$），惩罚照单全收。PPO 论文在它的连续控制基准上比较过 $\varepsilon$ 的取值：不裁剪也不加惩罚时归一化得分为 −0.39，$\varepsilon=0.1$、0.2、0.3 分别为 0.76、0.82、0.70 [@schulman2017ppo]。这是机器人控制任务上的**实测**，不能直接搬到语言模型上。

::: note 式 (8.11) 里的 $1/\abs{o}$ 不是 PPO 原文的
PPO 原文的目标是对时间步取经验平均 $\hat{\E}_t$，没有“除以回答长度”这一步 [@schulman2017ppo]。$1/\abs{o}$（每条回答先在自己的 token 上取平均）是 DeepSeekMath 的写法；Dr. GRPO 论文指出，许多开源的 PPO 实现也是这样做的，而这会带来与长度有关的偏差 [@liu2025drgrpo]。[§4](#p4) 会回到这一点。
:::

## KL 惩罚：放进每个 token 的奖励里

最后一个部件是 KL 约束。奖励模型只是人类判断的一个近似，策略如果一味追高分，会找到奖励模型的漏洞（这叫奖励模型被“过度优化”）。InstructGPT 的做法是在每个 token 的奖励里扣掉一项：当前策略与参考模型在这个 token 上的对数概率之差，乘以系数 $\beta$ [@ouyang2022instructgpt]；DeepSeekMath 沿用了它（论文式 (2)）：

::: eq 每个 token 的奖励（DeepSeekMath 式 2） | (8.12)
$$r_t=r_\varphi(q,o_{\le t})-\beta\log\frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{ref}(o_t\mid q,o_{<t})}$$
$r_\varphi$ 是奖励模型（$\varphi$ 是它的参数）；它只在最后一个 token 给分，所以前面的 token 只剩下 KL 惩罚那一项。$\beta>0$ 是 KL 系数。图 8.4 第一行就是这样来的。
:::

$\log(\pi_\theta/\pi_{ref})$ 在一个 token 上可正可负；它在 $\pi_\theta$ 下的期望才是 KL 散度。[§5](#p5) 会把它叫作 **k1 估计量**，并解释 GRPO 为什么换了一个。

::: warn “PPO 的 KL 惩罚”指的是两件不同的事
PPO 原论文 §4 也有一个 KL 惩罚版本，但它惩罚的是新策略与**采样时的旧策略** $\pi_{\theta_{old}}$ 之间的 KL，系数 $\beta$ 还会自适应调整：KL 小于目标值的 1/1.5 就把 $\beta$ 减半，大于目标值的 1.5 倍就加倍 [@schulman2017ppo]。RLHF 里的 KL 惩罚则是对**参考模型**（SFT 模型）的，系数固定，按 token 加进奖励 [@ouyang2022instructgpt]。DeepSeekMath 式 (2) 是后者。
:::

::: key 本页要点
- 语言模型是策略：状态 = 题目 + 已写的前缀，动作 = 下一个 token；策略梯度 (8.6) 把每个 token 的对数概率按奖励推高。
- 得分函数的期望为 0，所以减去任何不依赖于回答本身的基线都不改变梯度的期望，却能大幅降低方差（例子里从 27.56 降到 0）。
- PPO 用一个和策略同样大的价值网络给每个 token 估基线，再用 GAE (8.9) 把终点奖励分摊成逐 token 优势；优势的质量取决于价值网络准不准。
- 重要性比 $\rho_t$ 让一批样本能用多次；裁剪 + min 让目标成为悲观下界。PPO-RLHF 把 KL 惩罚以 k1 的形式加进每个 token 的奖励。
:::

# GRPO：同题多答，组内排名 | 用一组回答的平均分代替价值网络 | 35

[§3](#p3) 的结论是：策略梯度需要一个基线，PPO 用一个和策略一样大的价值网络去估它，贵，而且估不准时会把优势带歪。GRPO 的问题于是很具体：**不训练任何网络，能不能得到一个像样的基线？**

DeepSeekMath 的回答是：对同一道题多采几个回答，用它们的平均奖励当基线 [@shao2024deepseekmath]。论文的原话是 GRPO “foregoes the critic model, instead estimating the baseline from group scores”：放弃价值模型，改从组内得分估计基线。

::: fig figures/l08-ppo-vs-grpo.svg | 图 8.6 PPO 与 GRPO 的结构对比，以及两者的模型状态显存 | wide scroll
**看上下两排的差别：GRPO 那一排没有“价值模型”，GAE 换成了“组计算”，KL 也不再加进奖励 r，而是直接进入目标。** 蓝框是训练中的模型，灰框是冻结的模型。据 DeepSeekMath 论文图 4 重绘 [@shao2024deepseekmath]。底部的显存条是笔者按 (8.2) 估算的，不是论文数字：价值模型的 110.6 GB 正是两条长度之差。
:::

## 图景：同一道题写 G 份答卷

::: ana 类比：GRPO = 同题多答、组内排名
**映射。** 一道题 $q$ = 一场考试的一道大题；策略 = 学生；$G$ 个回答 = 同一个学生对这道题写的 $G$ 份答卷；奖励模型 = 阅卷老师给每份答卷的分数 $r_i$；组内标准化 = 不看绝对分，而看“这份答卷比这一组的平均分高几个标准差”；优势为正的答卷，它的写法被鼓励；为负的，被抑制。PPO 的价值网络，则相当于另请一位“预测分数的老师”，在学生每写一行时就估计“照这样写下去最终能得几分”。

**收益。** 它把“基线从哪来”变成了一个统计问题：同一道题的 $G$ 份答卷，平均分就是这道题“正常发挥”的分数，不用再训练一个预测者。它也解释了为什么只在同一道题内部比较：难题上的 60 分可能已经很好，简单题上的 60 分可能很差，跨题比较绝对分没有意义。

**失效的地方。** 只有同一道题内部的比较才有意义：如果一组答卷全对或全错，每份都等于平均分，优势全为零，这道题不提供任何学习信号。类比里的“答卷”是同一个学生反复作答，答卷之间不是独立的不同学生；真实的 GRPO 还把同一个分数平摊给答卷里的每一个字（token），哪一步写得好、哪一步写得差，它分不出来（[§6](#p6) 的过程监督就是为了补这一点）。
:::

## 形式化：组采样与结果监督的优势

::: pre 标准化：比平均高出几个标准差 | 熟悉 z 分数的可以跳过
一组数 $\mathbf x=(x_1,\dots,x_G)$ 的均值 $\bar x=\frac1G\sum_jx_j$，（总体）标准差 $\mathrm{std}(\mathbf x)=\sqrt{\frac1G\sum_j(x_j-\bar x)^2}$（本课不用 $\sigma$ 记标准差，$\sigma$ 留给 sigmoid 函数）。把每个数换成 $z_j=(x_j-\bar x)/\mathrm{std}(\mathbf x)$ 叫**标准化**：新的一组数均值为 0、均方为 1，意思是“比这一组的平均高出几个标准差”，与原来的单位无关。例如四个分数 $(60,70,80,90)$：均值 75，偏差 $(-15,-5,5,15)$，标准差 $\sqrt{500/4}\approx11.18$，标准化后是 $(-1.342,-0.447,+0.447,+1.342)$；把四个分数都乘 10、或者都加 20，标准化的结果不变。见 [M04 §2](../math/m04-probability.html#p2)；样本标准差为什么除以 $G-1$，见 [M04 §6](../math/m04-probability.html#p6)。
:::

对每道题 $q$，用采样时的旧策略 $\pi_{\theta_{old}}$ 生成 $G$ 个回答 $\{o_1,\dots,o_G\}$，奖励模型给每个回答打分，得到奖励向量 $\mathbf r=(r_1,\dots,r_G)\in\R^{G}$。**结果监督**（outcome supervision）只在回答结束时打一次分，GRPO 把每个奖励在组内标准化，作为这个回答**所有 token** 的优势 [@shao2024deepseekmath]：

::: eq 结果监督下的组相对优势 | (8.13)
$$\hat A_{i,t}=\widetilde r_i=\frac{r_i-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)},\qquad t=1,\dots,\abs{o_i}$$
$\mathrm{mean}(\mathbf r)=\frac1G\sum_jr_j$ 是组均值；$\widetilde r_i$ 叫归一化奖励。优势与 $t$ 无关：同一个回答的每个 token 共用一个数。
:::

::: warn 除以的是哪种标准差？论文没有说
(8.13) 里的 $\mathrm{std}$ 可以是**总体标准差**（平方偏差之和除以 $G$）或**样本标准差**（除以 $G-1$）。DeepSeekMath 没有写明，DeepSeek 也没有公开 GRPO 的训练代码（官方仓库只有评测代码）[@gh-deepseek-math]。常用的开源实现 Hugging Face TRL 用的是样本标准差，再在分母上加 $10^{-4}$ [@trl-grpo]。两种约定下的优势只差一个常数因子：

$$\frac{\hat A^{\text{样本}}}{\hat A^{\text{总体}}}=\sqrt{\frac{G-1}{G}}:\qquad G=4\text{ 时 }0.866,\qquad G=64\text{ 时 }0.992.$$

常数因子可以并入学习率，对训练的影响很小。严格说，目标里还有不随优势缩放的 KL 项，所以换约定相当于同时把学习率乘 $\sqrt{(G-1)/G}$、把 $\beta$ 乘 $\sqrt{G/(G-1)}$（$G=64$ 时约 1.008 倍）。但手算例题时两者差 13%，必须声明。**本课讲原理时一律用总体标准差**（二值奖励、两对两错时结果恰好是 $\pm1$），同时给出样本标准差的值。
:::

::: ex G = 4 的四组答卷（对照图 8.7）
(a) $\mathbf r=(1,0,0,1)$：均值 $0.5$。总体方差 $\tfrac14(0.25\times4)=0.25$，总体 std $0.5$，优势 $(+1,-1,-1,+1)$。样本方差 $\tfrac13(0.25\times4)=0.333$，样本 std $0.5774$，优势 $(\pm0.866)$。

(b) $\mathbf r=(1,0,0,0)$：均值 $0.25$。偏差 $(0.75,-0.25,-0.25,-0.25)$，平方和 $0.75$；总体 std $\sqrt{0.75/4}=0.4330$，优势 $(+1.732,-0.577,-0.577,-0.577)$；样本 std $\sqrt{0.75/3}=0.5$，优势 $(+1.5,-0.5,-0.5,-0.5)$。唯一答对的那份得到很大的正优势，三份错的各分一小份负优势。

(c) 连续奖励 $\mathbf r=(0.9,0.2,0.4,0.7)$：均值 $0.55$，偏差 $(0.35,-0.35,-0.15,0.15)$，平方和 $0.29$；总体 std $\sqrt{0.29/4}=0.2693$，优势 $(+1.300,-1.300,-0.557,+0.557)$。注意 0.4 分的答卷优势为负：它低于这一组的平均。

(d) $\mathbf r=(1,1,1,1)$：偏差全为 0，std 也为 0，公式是 $0/0$。实现里分母加一个小数（TRL 是 $10^{-4}$），得到优势全为 0：这道题的优势项对策略梯度没有贡献，只剩 KL 项（练习 2）。

无论哪一组，优势之和都是 0：组内有正有负，总量守恒。
:::

::: fig figures/l08-group-adv.svg | 图 8.7 四组 G = 4 的答卷：奖励与组内标准化后的优势 | wide scroll
**看 (b) 与 (d)：一对三错时，唯一答对的那份拿到 +1.732 的大优势；全对时，所有优势都是 0，这道题白做了。** 每格上方是四份答卷的奖励，下方的条是按总体标准差算的优势，底部灰字是按样本标准差算的值。所有数由脚本 `l08_group_adv.py` 计算。
:::

## 二值奖励的闭式：难度偏差从哪来

数学题的奖励常常只有对错两种。设一组 $G$ 份答卷里有 $c$ 份答对（奖励 1），其余答错（奖励 0），记正确率 $p=c/G$（$0<p<1$）。这时 (8.13) 有闭式。

::: deriv 二值奖励下的组内优势（总体标准差）
$\mathrm{mean}(\mathbf r)=\frac{c\cdot1+(G-c)\cdot0}{G}=p$ || 均值就是正确率
$\mathrm{std}^2(\mathbf r)=\frac{c(1-p)^2+(G-c)p^2}{G}$ || 总体方差：平方偏差的平均
$=p(1-p)^2+(1-p)p^2=p(1-p)$ || $c/G=p$，提出公因子 $p(1-p)$
$\hat A^{+}=\frac{1-p}{\sqrt{p(1-p)}}=\sqrt{\frac{1-p}{p}}$ || 代入 (8.13)，答对者 $r_i=1$
$\hat A^{-}=\frac{0-p}{\sqrt{p(1-p)}}=-\sqrt{\frac{p}{1-p}}$ || 答错者 $r_i=0$
$c\hat A^{+}+(G-c)\hat A^{-}=G\Big[p\sqrt{\tfrac{1-p}{p}}-(1-p)\sqrt{\tfrac{p}{1-p}}\Big]$ || 代入，$c=Gp$
$=G\big[\sqrt{p(1-p)}-\sqrt{p(1-p)}\big]=0$ || 化简：优势之和为 0
$\frac1G\sum_i\hat A_i^2=p\cdot\frac{1-p}{p}+(1-p)\cdot\frac{p}{1-p}$ || 两类回答各占 $p$、$1-p$
$=(1-p)+p=1$ || 优势的均方为 1：标准化的本意
:::

::: eq 二值奖励的组内优势 | (8.14)
$$\hat A^{+}=\sqrt{\frac{1-p}{p}},\qquad \hat A^{-}=-\sqrt{\frac{p}{1-p}},\qquad \frac1G\sum_{i}\abs{\hat A_i}=2\sqrt{p(1-p)}$$
最后一式：$c$ 个 $\hat A^{+}$ 与 $G-c$ 个 $\abs{\hat A^{-}}$ 相加再除以 $G$，得 $p\,\hat A^{+}+(1-p)\abs{\hat A^{-}}$；两项各等于 $\sqrt{p(1-p)}$。
:::

::: ex DeepSeekMath 的 G = 64
- 64 份里只对 1 份（$p=1/64$）：$\hat A^+=\sqrt{63}\approx7.937$，$\hat A^-=-1/\sqrt{63}\approx-0.126$。那唯一一份正确答案被强烈鼓励，63 份错答案各被轻轻压一下。
- 对一半（$p=1/2$）：$\hat A^\pm=\pm1$。
- 对 63 份（$p=63/64$）：$\hat A^+\approx0.126$，唯一错的那份 $\hat A^-\approx-7.937$。

不除以标准差时（只减均值），优势是 $1-p$ 与 $-p$，$p=1/64$ 时为 $+0.984$ 与 $-0.016$。除以标准差把这两个数都乘上了 $1/\sqrt{p(1-p)}\approx8.06$。
:::

::: fig figures/l08-adv-vs-p.svg | 图 8.8 二值奖励下，优势与“每道题的总权重”随组内正确率 p 的变化 | wide scroll
**看右图两条弧线的形状差别：不除标准差时（虚线），极难或极易的题总权重趋于 0；除以标准差后（实线），它们的权重被抬高，$p=1/64$ 时从 0.031 抬到 0.248，是 8 倍。** 中等题（$p=\tfrac12$）也从 0.5 抬到 1.0，是 2 倍，所以极难题**相对**中等题的放大约为 $8.06/2\approx4$ 倍（练习 9）。左图是单个回答的优势，右图是一道题所有回答的优势绝对值的平均，即这道题在梯度里的“音量”。曲线由 (8.14) 计算。
:::

除以标准差是一个**选择**，不是推导的结果。它的好处是让每道题的优势有统一的尺度（均方为 1），奖励模型给分的绝对范围变了也不影响训练。代价是改变了题与题之间的相对权重：Dr. GRPO 论文把它叫作“题目难度偏差”（question-level difficulty bias），并建议去掉 [@liu2025drgrpo]；DeepSeek 自己的 V3.2 论文印出的优势公式也只减均值、不除标准差，不过同一段的前一句又说“在组内归一化”，前后表述不一，也没有训练代码可核（[第 15 课 §4](lesson-15-v3-2-agents.html#p4)）[@deepseekai2025v32]。哪种更好取决于你希望训练把力气花在哪类题上，这篇论文没有做这个对照。

## 组均值为什么是一个合法的基线

[§3](#p3) 的 (8.7) 要求基线**不依赖于回答本身**。组均值 $\mathrm{mean}(\mathbf r)=\frac1G\sum_jr_j$ 里恰好包含了 $r_i$ 自己，严格说不满足条件。好在修正只差一个常数。

::: deriv 组均值基线与留一基线
$r_i-\frac1G\sum_{j=1}^{G}r_j=r_i-\frac1G r_i-\frac1G\sum_{j\ne i}r_j$ || 把自己那一项从和里拆出来
$=\frac{G-1}{G}r_i-\frac1G\sum_{j\ne i}r_j$ || 合并 $r_i$ 的两项
$=\frac{G-1}{G}\Big(r_i-\frac1{G-1}\sum_{j\ne i}r_j\Big)$ || 提出公因子 $\frac{G-1}{G}$
:::

::: eq 减组均值的优势 = 常数 × 留一基线的优势 | (8.15)
$$\frac{G}{G-1}\big(r_i-\mathrm{mean}(\mathbf r)\big)=r_i-\frac{1}{G-1}\sum_{j\ne i}r_j$$
右边的基线只用**其他** $G-1$ 个回答的奖励；同一道题的 $G$ 个回答是各自独立采样的，所以它与 $o_i$ 独立，不引入偏差。这种基线叫留一（leave-one-out）基线，对应的算法叫 RLOO（名字里的 LOO 就是 leave-one-out）。
:::

所以“只减组均值”的 GRPO 优势，等于 RLOO 的优势乘上常数 $\frac{G-1}{G}$；把它乘在整条回答的 $\sum_t\nabla_\theta\log\pi_\theta$ 上（(8.7) 的形状），期望正好是真梯度的 $\frac{G-1}{G}$ 倍，即无偏（常数并入学习率）[@liu2025drgrpo]。这句话有两个前提：一是组内各回答独立地从求梯度处的策略采样；二是先不计 (8.16) 里的 $1/\abs{o_i}$。带上 $1/\abs{o_i}$ 后，连留一基线也不再无偏：对不依赖 $o$ 的 $b$，$\E_o\big[\tfrac{b}{\abs{o}}\nabla_\theta\log\pi_\theta(o\mid q)\big]=b\,\nabla_\theta\E_o\big[\tfrac1{\abs{o}}\big]$（与 [§3](#p3) “得分函数期望为零”同样的推导，只是多了一个随 $o$ 变化的因子 $1/\abs{o}$），一般不为 0。这正是下面“长度偏差”的来源。**除以组内标准差之后，这个等价就不再严格成立**：标准差是用包括 $r_i$ 在内的全部奖励算的，与 $o_i$ 相关。例如 $\mathbf r=(1,0,0,1)$ 时，留一基线下的优势是 $1-\tfrac13(0+0+1)=+0.667$ 与 $0-\tfrac13(1+0+1)=-0.667$，恰好是 $\frac{G}{G-1}=\frac43$ 乘以减均值后的 $\pm0.5$。

## GRPO 的目标函数，逐项读

把组相对优势放进 PPO 的裁剪目标，再把 KL 项从奖励里挪到目标里，就得到 GRPO 的目标（DeepSeekMath 式 (3)，这里换成本课的 $\rho$、$w$ 记号）[@shao2024deepseekmath]：

::: eq GRPO 的目标（DeepSeekMath 式 3） | (8.16)
$$\begin{aligned}\mathcal J_{\text{GRPO}}(\theta)=\;&\E_{q\sim P(Q),\ \{o_i\}_{i=1}^{G}\sim\pi_{\theta_{old}}(\cdot\mid q)}\\&\frac1G\sum_{i=1}^{G}\frac1{\abs{o_i}}\sum_{t=1}^{\abs{o_i}}\Big\{\min\big[\rho_{i,t}\hat A_{i,t},\ \clip(\rho_{i,t},1-\varepsilon,1+\varepsilon)\,\hat A_{i,t}\big]-\beta\,\widehat{\KL}_{i,t}\Big\}\end{aligned}$$
$\rho_{i,t}=\pi_\theta(o_{i,t}\mid q,o_{i,<t})/\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})$；$\widehat{\KL}_{i,t}=w_{i,t}-\log w_{i,t}-1$，$w_{i,t}=\pi_{ref}(o_{i,t}\mid q,o_{i,<t})/\pi_\theta(o_{i,t}\mid q,o_{i,<t})$（[§5](#p5) 专门讲）。论文把 $\widehat{\KL}_{i,t}$ 写成 $\mathbb D_{KL}[\pi_\theta\Vert\pi_{ref}]$，但它是单个 token 上的估计量，依赖 $i$、$t$。
:::

这一行公式里有六层东西，从外往里：

| 项 | 做什么 | 与 PPO (8.11) 相比 |
|---|---|---|
| $\E_{q\sim P(Q)}$ | 对题目取平均：每步从题库里抽一批题 | 相同 |
| $\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{old}}$ | 每道题用旧策略采 $G$ 个回答 | PPO 不需要成组采样 |
| $\frac1G\sum_i$ | 组内平均：一道题的 $G$ 个回答各占 $1/G$ | 新增 |
| $\frac1{\abs{o_i}}\sum_t$ | 回答内平均：先在自己的 token 上取平均 | 相同（DeepSeekMath 的写法） |
| $\min[\rho\hat A,\ \clip(\rho)\hat A]$ | PPO 的裁剪；优势换成组相对优势 (8.13) | 优势不再来自价值网络和 GAE |
| $-\beta\,\widehat{\KL}_{i,t}$ | 把策略拉向参考模型，直接作为目标的一项 | PPO 把 KL 加进每个 token 的奖励 (8.12) |

最后一行的挪动有一个直接的好处：KL 不再混进奖励，优势 $\hat A_{i,t}$ 的计算就只和奖励模型的打分有关，干净许多。论文的说法是，GRPO 通过直接在损失里加 KL 来正则化，避免了把 $\hat A_{i,t}$ 的计算弄复杂 [@shao2024deepseekmath]。论文还给了组相对优势的另一个理由：奖励模型通常就是在“同一道题的不同回答之间的比较”上训练出来的，组内比较正好和它的训练方式相合 [@shao2024deepseekmath]。

::: warn 每条回答先除以自己的长度：长度偏差
$\frac1{\abs{o_i}}$ 让每条回答的总权重相同，于是**每个 token** 的权重与回答长度成反比。设两个答错的回答，优势都是 $-1$，一个 100 个 token，一个 1,000 个 token。在 (8.16) 里，前者每个 token 的系数是 $-1/100$，后者是 $-1/1000$：长的错答案，每个 token 受的惩罚只有短的十分之一。对答对的回答则相反，短的每个 token 得到更多鼓励。Dr. GRPO 论文把这叫作“回答长度偏差”（response-level length bias），认为它可能让模型写出越来越长的错误答案 [@liu2025drgrpo]。这同样是一个写法上的选择；V3.2 的 GRPO 目标仍然保留了 $1/\abs{o_i}$ [@deepseekai2025v32]。
:::

## DeepSeekMath 的实际设置下，裁剪没有起作用

DeepSeekMath 报告的 RL 设置里有一句：策略模型在每个探索阶段之后**只更新一次**（“The policy model only has a single update following each exploration stage.”）[@shao2024deepseekmath]。如果把它读成“每采一批数据，只做一次优化器更新”（论文附录 A.1.5 在简化分析时就是这样读的），那么这唯一一次求梯度发生在 $\theta=\theta_{old}$ 处。

::: deriv 在 $\theta=\theta_{old}$ 处，GRPO 的梯度
$\rho_{i,t}\big|_{\theta=\theta_{old}}=1\in[1-\varepsilon,1+\varepsilon]$ || 分子分母是同一个分布
$\min\big[\rho\hat A,\ \clip(\rho,1-\varepsilon,1+\varepsilon)\hat A\big]=\rho\hat A\quad(\rho\text{ 在 1 附近})$ || 区间内部 clip 不改变 $\rho$
$\nabla_\theta\rho_{i,t}=\rho_{i,t}\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})=\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})$ || 对数导数技巧；$\rho=1$
$\nabla_\theta\big(\rho_{i,t}\hat A_{i,t}\big)=\hat A_{i,t}\,\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})$ || 优势是常数，不求导
:::

第二行说的是：在 $\rho=1$ 的一个小邻域里，min 的两项完全相同，所以它们的梯度也相同，取 min 不改变梯度（图 8.5 的蓝点正处在黑线与虚线重合的那一段上）。

KL 项的梯度在 [§5](#p5) 算，结果是 $\beta(w_{i,t}-1)\nabla\log\pi_\theta$。合起来：

::: eq 单次更新时 GRPO 的梯度 | (8.17)
$$\nabla_\theta\mathcal J_{\text{GRPO}}=\E\Big[\frac1G\sum_{i=1}^{G}\frac1{\abs{o_i}}\sum_{t=1}^{\abs{o_i}}\Big(\hat A_{i,t}+\beta\,(w_{i,t}-1)\Big)\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\Big]$$
即论文附录的式 (20)。它不含 $\varepsilon$：在这个读法下，GRPO 就是“带组均值基线、除以组内标准差的 REINFORCE，加上一个 KL 正则”。
:::

这或许也解释了论文为什么没有报告裁剪系数 $\varepsilon$：在这种设置下它不影响结果（这是笔者的推断，论文没有说明）。但有两点保留。第一，“只更新一次”也可能指“把一批数据分成若干个小批，各走一步”；那样从第二个小批开始 $\rho\ne1$，裁剪就会起作用，而论文没有说明小批的个数。第二，即便只走一步，如果采样用的推理引擎和训练框架算出的概率在数值上不完全一致，$\rho$ 也不严格等于 1。后来的 R1 明确采用了多个小批的做法：每次采出的 8,192 个输出分成 16 个小批依次更新 [@deepseekai2025r1]。那里 $\rho$ 不再恒等于 1，$\varepsilon$ 的取值就有了意义：R1 论文说，裁剪比例太低会截断大量 token 的梯度，太高又会让训练不稳定，它的第一阶段 RL 把 $\varepsilon$ 设成了 10 [@deepseekai2025r1]（[第 13 课](lesson-13-r1-reasoning.html)）。这个数比 PPO 常用的 0.2 大得多：区间变成 $[-9,11]$，比值 $\rho$ 总是正的，下界永远碰不到；于是只有优势为正、且 $\rho$ 超过 11 的 token 才会被截断，优势为负的 token 永远不被裁剪（$\rho>11$ 时 $\rho\hat A<11\hat A$，min 取的是未裁剪的那一项）。这是笔者按定义算的。

## 代价：G 份答卷的采样账

GRPO 省掉了价值模型，但每道题要采 $G$ 个回答，DeepSeekMath 用的是 $G=64$。采样是解码密集型的负载（[第 4 课](lesson-04-inference-kv-cache.html)），账单落在 KV cache 上。

::: ex 一道题的 64 份答卷要多少 KV cache
DeepSeekMath 7B 是 30 层、32 个 KV 头、每头 128 维的 MHA（[§1](#p1)）。BF16 下每个 token 的 KV cache：$2\times30\times32\times128\times2=491{,}520$ 字节 $=480$ KiB（依次是 K 与 V 两份、30 层、32 个头、每头 128 维、每个数 2 字节）。

一道题 64 个回答、每个最长 1,024 个 token（论文的最大长度设置）：$64\times1024\times480$ KiB $=31{,}457{,}280$ KiB $=30$ GiB，还不算题目本身的 token。价值模型省下的 110.6 GB 模型状态，一部分就这样花在了组采样的 KV cache 上。这也是为什么后来的推理系统工作（第 9 课的 MLA、第 12 课的推理部署）对 RL 同样重要。
:::

::: key 本页要点
- GRPO 对每道题采 $G$ 个回答，优势 $\hat A_{i}=(r_i-\mathrm{mean})/\mathrm{std}$，同一回答的所有 token 共用；不训练价值网络。
- 标准差的约定论文没说：本课用总体标准差（两对两错时 $\pm1$），TRL 用样本标准差（$\pm0.866$），只差常数 $\sqrt{(G-1)/G}$。
- 二值奖励下 $\hat A^\pm$ 有闭式 (8.14)；除以标准差会放大极难、极易题的权重（难度偏差），只减均值则等价于无偏的留一基线（RLOO；前提是组内独立采样、不计 $1/\abs{o_i}$）。
- 目标 (8.16) = 题目平均 × 组内平均 × 回答内 token 平均 × PPO 裁剪，再减去 KL 估计量；$1/\abs{o_i}$ 带来长度偏差。
- DeepSeekMath 每批只更新一次时 $\rho=1$，裁剪不起作用，梯度就是 (8.17)。
:::

# KL 项：k3 估计器从哪来 | k1、k2、k3，无偏与非负，以及梯度上的隐患 | 30

GRPO 目标 (8.16) 的最后一项是 $-\beta\,\widehat{\KL}_{i,t}$，论文式 (4) 把它写成

$$
\widehat{\KL}_{i,t}=\frac{\pi_{ref}(o_{i,t}\mid q,o_{i,<t})}{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}-\log\frac{\pi_{ref}(o_{i,t}\mid q,o_{i,<t})}{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}-1=w_{i,t}-\log w_{i,t}-1,
$$

并称它是一个“无偏估计”（unbiased estimator），出处是 Schulman 2020 年的一篇博客，而且“保证为正”（guaranteed to be positive）[@shao2024deepseekmath]。这一页回答三个问题：为什么要“估计” KL 而不是直接算？这个式子是怎么来的，“无偏”“为正”各在什么意义下成立？把它放进损失求梯度，得到的又是什么？

## 先回忆 KL 散度

::: pre KL 散度 | 学过 M05 的可以跳过
两个分布 $P$、$Q$ 之间的 **KL 散度**（Kullback–Leibler divergence）是 $\KL[P\Vert Q]=\E_{x\sim P}\big[\log\frac{P(x)}{Q(x)}\big]$。它衡量“用 $Q$ 去近似 $P$ 损失了多少信息”，恒 $\ge0$，当且仅当 $P=Q$ 时为 0。它**不对称**：$\KL[P\Vert Q]\ne\KL[Q\Vert P]$，期望是在第一个分布下取的。非负性的证明（Jensen 不等式）见 [M05 §4](../math/m05-information-theory.html#p4)，两个方向的差别见 [M05 §5](../math/m05-information-theory.html#p5)；本页的 k1、k2、k3 在 [M05 §6](../math/m05-information-theory.html#p6) 从信息论一侧又讲了一遍。
:::

GRPO 想约束的是 $\KL[\pi_\theta\Vert\pi_{ref}]$：期望在**当前策略**下取，也就是在策略自己会写出的那些 token 上，比较它和参考模型的概率。在某个位置（给定前缀）上，用参考比 $w=\pi_{ref}/\pi_\theta$ 改写：

::: eq 单个位置上的 KL | (8.18)
$$\KL\big[\pi_\theta\Vert\pi_{ref}\big]=\E_{x\sim\pi_\theta}\Big[\log\frac{\pi_\theta(x)}{\pi_{ref}(x)}\Big]=\E_{x\sim\pi_\theta}\big[-\log w(x)\big],\qquad w(x)=\frac{\pi_{ref}(x)}{\pi_\theta(x)}$$
$x$ 取遍词表；为了简洁，前缀 $(q,o_{<t})$ 不写出来。
:::

为什么不直接算？整条回答的 KL 要对所有可能的回答求和，根本算不出来。逐个位置算是可以的：把两个模型在这个位置上的完整分布（DeepSeekMath 的词表有 102,400 项）拿来按 (8.18) 加一遍。但这需要参考模型为每个位置输出并保留整个分布；而被采样到的那个 token 的对数概率，算策略损失时本来就有了。所以实践中常用**单样本估计**：只用被采样到的那个 token 的 $w$ 算一个数，要求它的期望等于 KL。这是笔者对“为什么要估计”的解释，论文没有讨论这一点。

## 三个估计器：k1、k2、k3

Schulman 的博客 [@schulman2020kl] 比较了三个单样本估计器。博客的记号与本课不同，对照如下：

| Schulman 博客 | 本课 | 含义 |
|---|---|---|
| $q$（采样分布） | $\pi_\theta$ | 样本从这里来 |
| $p$ | $\pi_{ref}$ | 被比较的分布 |
| $r=p(x)/q(x)$ | $w=\pi_{ref}/\pi_\theta$ | 单个样本上的比值 |
| $\mathrm{KL}[q,p]$ | $\KL[\pi_\theta\Vert\pi_{ref}]$ | 要估计的量 |

博客的 $q$、$r$ 和本课的 $q$（题目）、$r$（奖励）撞名，所以本课一律用 $w$。

**k1：直接用定义。** (8.18) 的被积函数本身就是一个无偏估计：

::: eq k1 估计器 | (8.19)
$$k_1=-\log w,\qquad \E_{\pi_\theta}[k_1]=\KL[\pi_\theta\Vert\pi_{ref}]$$
PPO-RLHF 每 token 奖励 (8.12) 里扣的 $\beta\log\frac{\pi_\theta}{\pi_{ref}}$ 正是 $\beta k_1$。
:::

k1 的毛病是**可以为负**：只要这个 token 在参考模型下的概率比在当前策略下高（$w>1$），$k_1<0$。而 KL 本身是非负的。单个样本的“KL 估计”为负，说明它的波动很大：Schulman 博客说它对一半左右的样本为负；博客的高斯实验里（真 KL 为 0.005），它的标准差是真值的 20 倍 [@schulman2020kl]。

**k2：平方一下。** $k_2=\frac12(\log w)^2$ 恒非负，方差低得多，但它是**有偏的**。博客指出它的期望是另一种散度（一个 {{f-散度::用一个凸函数 f 定义的一族“分布差异”度量，KL 是其中的一员}}），与 KL 在 $w\approx1$ 附近二阶相同 [@schulman2020kl]。“二阶相同”可以直接展开看出来（这里用 $\xi$ 记 $w$ 离 1 的偏移，免得和 [§3](#p3) 的 TD 残差 $\delta_t$ 混淆）：

::: deriv k1 与 k2 在 $w\approx1$ 附近的期望相同到二阶
$w=1+\xi,\quad \E_{\pi_\theta}[\xi]=\E_{\pi_\theta}[w]-1=0$ || 下面会证明 $\E_{\pi_\theta}[w]=1$
$k_1=-\log(1+\xi)=-\xi+\tfrac12\xi^2-\tfrac13\xi^3+\cdots$ || 对数的泰勒展开
$k_2=\tfrac12\big(\xi-\tfrac12\xi^2+\cdots\big)^2=\tfrac12\xi^2-\tfrac12\xi^3+\cdots$ || 平方后保留到三阶
$\E[k_1]=\tfrac12\E[\xi^2]+O(\E\abs{\xi}^3),\qquad \E[k_2]=\tfrac12\E[\xi^2]+O(\E\abs{\xi}^3)$ || 一阶项的期望为 0
:::

所以两个分布很接近时，k2 的偏差很小；分布离得远了，偏差就显出来。

**k3：加一个期望为零的修正项。** 想要一个既无偏、又非负、方差又低的估计器。关键观察是：$w-1$ 这个量的期望恰好为 0。

::: deriv 参考比的期望为 1
$\E_{x\sim\pi_\theta}[w(x)]=\sum_x\pi_\theta(x)\,\frac{\pi_{ref}(x)}{\pi_\theta(x)}$ || 期望的定义
$=\sum_x\pi_{ref}(x)=1$ || 约分（要求 $\pi_\theta(x)>0$，softmax 满足）
$\E_{x\sim\pi_\theta}[w-1]=0$ || 两边减 1
:::

期望已知的量叫**控制变量**（control variate），这里用的 $w-1$ 期望恰好为零：把它的任意倍数加到一个无偏估计上，期望不变，方差却可以调。于是有一整族无偏估计：

::: eq 用控制变量构造无偏估计 | (8.20)
$$\begin{aligned}\tilde k_\kappa&=k_1+\kappa\,(w-1)=-\log w+\kappa\,(w-1)\\ \E_{\pi_\theta}\big[\tilde k_\kappa\big]&=\KL[\pi_\theta\Vert\pi_{ref}]+\kappa\cdot0=\KL[\pi_\theta\Vert\pi_{ref}]\quad\text{对任意 }\kappa\end{aligned}$$
$\kappa$ 是自由参数，选它来降低方差。Schulman 博客把它记作 $\lambda$；本课的 $\lambda$ 已经留给 GAE (8.9)，所以改用 $\kappa$。
:::

Schulman 取 $\kappa=1$，理由是这时估计器恒非负 [@schulman2020kl]：

::: eq k3 估计器（GRPO 式 4） | (8.21)
$$k_3=(w-1)-\log w,\qquad \E_{\pi_\theta}[k_3]=\KL[\pi_\theta\Vert\pi_{ref}],\qquad k_3\ge0$$
GRPO 的 $\widehat{\KL}_{i,t}=w_{i,t}-\log w_{i,t}-1$ 就是 k3。
:::

::: deriv k3 非负，且只在 w = 1 时为零
$f(w)=w-1-\log w,\quad w>0$ || 把 k3 看成 $w$ 的函数
$f'(w)=1-\frac1w,\qquad f''(w)=\frac1{w^2}>0$ || 求导；二阶导恒正，$f$ 是凸函数
$f'(1)=0\ \Rightarrow\ \min_w f(w)=f(1)=0$ || 凸函数的驻点就是全局最小点
$w-1-\log w\ge0\quad(\text{等号当且仅当 }w=1)$ || 等价于熟悉的不等式 $\log w\le w-1$
:::

几何上，$w-1$ 是 $\log w$ 在 $w=1$ 处的切线；$\log$ 是凹函数，切线总在它上方，k3 就是切线与曲线之间的**竖直距离**。所以严格说 k3 是“非负”而不是论文说的“为正”：$w=1$（当前策略与参考模型在这个 token 上概率相同）时它正好为 0。

::: fig figures/l08-kl-estimators.svg | 图 8.9 三个 KL 估计量：单个样本上的取值（左），以及在一个高斯例子里的相对标准差（右） | wide scroll
**看左图 $w=1$ 附近：蓝色的 k3 与黄色的 k2 贴在一起（二阶相同），而红色的 k1 是一条穿过零点的斜线，$w>1$ 时为负。右图看红线：k1 的波动在 KL 很小时是真值的几十倍。** 右图是 Schulman 博客的测试：样本 $x\sim\mathcal N(0,1)$，另一个分布 $\mathcal N(\mu,1)$，真 KL 为 $\mu^2/2$。三条曲线是笔者推出的闭式（见下面的例题与练习 8），并用 200 点高斯–埃尔米特求积核对过；两个 $\mu$ 下的值与博客的实验表一致 [@schulman2020kl]。
:::

::: ex 高斯例子里的三个估计量（可以手算前两个）
设 $\pi_\theta=\mathcal N(0,1)$、$\pi_{ref}=\mathcal N(\mu,1)$（正态分布记作 $\mathcal N$，以免和参数量 $N$ 混淆）。对样本 $x$，$\log w=\log\frac{\pi_{ref}(x)}{\pi_\theta(x)}=\mu x-\frac{\mu^2}{2}$，是一个均值 $-\mu^2/2$、方差 $\mu^2$ 的正态变量。真 KL $=\mu^2/2$。

- **k1** $=-\log w$：期望 $\mu^2/2$，无偏；标准差 $\mu$，所以相对标准差 $=\mu/(\mu^2/2)=2/\mu$。$\mu=0.1$ 时是 20，$\mu=1$ 时是 2。
- **k2** $=\frac12(\log w)^2$：期望 $\frac12\big(\mu^2+\frac{\mu^4}{4}\big)=\frac{\mu^2}{2}+\frac{\mu^4}{8}$，偏差/真值 $=\mu^2/4$：$\mu=0.1$ 时 0.25%，$\mu=1$ 时 25%。相对标准差 $\sqrt{2+\mu^2}$：1.418 与 1.732。
- **k3** $=w-1-\log w$：无偏；相对标准差 $2\sqrt{e^{\mu^2}-1-\mu^2}/\mu^2$：1.417 与 1.695。

Schulman 博客的实验表给出的相对标准差是 20、1.42、1.42（$\mu=0.1$）与 2、1.73、1.7（$\mu=1$）[@schulman2020kl]，与闭式一致。博客据此的结论是：k3 在 $\mu=1$ 时标准差比 k2 还低，而且无偏。笔者按闭式往更大的 $\mu$ 算：$\mu$ 超过约 1.2（真 KL 约 0.73）以后，k3 的相对标准差反而超过 k2（$\mu=2$ 时 3.52 对 2.45），所以“k3 方差更低”只在两个分布足够接近时成立。带 KL 约束的 RL 微调，本意就是不让策略离参考模型那么远（这是笔者的判断，论文没有报告实际的 KL 大小）。
:::

## 无偏的前提，以及序列上发生了什么

“k3 无偏”这句话有两个前提。第一，**样本来自 $\pi_\theta$**。GRPO 的样本来自 $\pi_{\theta_{old}}$；在 DeepSeekMath 每批只更新一次的设置下，求梯度的那一刻 $\pi_{\theta_{old}}=\pi_\theta$，前提成立；如果一批数据上走多步，就不再严格成立。第二，它估计的是**单个位置**上的 KL。整条回答的 KL，按链式法则等于沿着 $\pi_\theta$ 生成的前缀把每个位置的 KL 加起来；而 (8.16) 对每条回答除以 $\abs{o_i}$，给出的是“每个 token 的平均 KL”，是另一个量。

## k3 对梯度的贡献

GRPO 把 $-\beta k_3$ 放进目标求梯度。先算单个 token 的梯度（$\pi_{ref}$ 冻结，只有 $\pi_\theta$ 依赖 $\theta$）：

::: deriv KL 项的梯度系数
$\nabla_\theta w=\pi_{ref}\,\nabla_\theta\frac1{\pi_\theta}=-\frac{\pi_{ref}}{\pi_\theta}\cdot\frac{\nabla_\theta\pi_\theta}{\pi_\theta}=-w\,\nabla_\theta\log\pi_\theta$ || 商的求导，再用对数导数技巧
$\nabla_\theta(-\log w)=\nabla_\theta\big(\log\pi_\theta-\log\pi_{ref}\big)=\nabla_\theta\log\pi_\theta$ || $\pi_{ref}$ 与 $\theta$ 无关
$\nabla_\theta k_3=\nabla_\theta w+\nabla_\theta(-\log w)=(1-w)\,\nabla_\theta\log\pi_\theta$ || 两项相加（常数 $-1$ 的梯度为 0）
$\nabla_\theta\big(-\beta k_3\big)=\beta\,(w-1)\,\nabla_\theta\log\pi_\theta$ || 乘以 $-\beta$
:::

::: eq KL 项在梯度系数里的贡献（DeepSeekMath 附录式 21 的第二项） | (8.22)
$$\nabla_\theta\big(-\beta\,\widehat{\KL}_{i,t}\big)=\beta\,\big(w_{i,t}-1\big)\,\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})$$
$w>1$（策略写这个 token 的概率低于参考模型）：系数为正，把它往上拉；$w<1$：系数为负，往下压。方向都是“回到参考模型”。
:::

::: ex KL 的拉力与优势比一比（β = 0.04）
- $w=0.5$：$0.04\times(0.5-1)=-0.02$；$w=2$：$+0.04$。两者都远小于二值奖励、两对两错时的优势 $\pm1$。
- $w=10$：$+0.36$。
- $w=100$（策略写这个 token 的概率只有参考模型的 1/100）：$+3.96$，已经是优势的约 4 倍。

另一侧却有下界：$w\to0$ 时系数趋于 $-\beta=-0.04$。也就是说，原始 k3 在 $\pi_\theta\gg\pi_{ref}$ 时拉力很弱，在 $\pi_\theta\ll\pi_{ref}$ 时拉力线性增长、没有上限。
:::

## 隐患：k3 当损失求导，梯度指向另一个 KL

k3 对 KL 的**值**无偏。但我们真正用的是它的**梯度**。把 (8.22) 在 $\pi_\theta$ 下取期望，看它等于什么：

::: deriv 把 k3 当损失，期望梯度是哪个 KL 的梯度
$\E_{x\sim\pi_\theta}\big[\nabla_\theta k_3\big]=\sum_x\pi_\theta(x)\,(1-w(x))\,\nabla_\theta\log\pi_\theta(x)$ || 见上一个推导
$=\sum_x\nabla_\theta\pi_\theta(x)-\sum_x\pi_{ref}(x)\,\nabla_\theta\log\pi_\theta(x)$ || 拆开；$\pi_\theta w=\pi_{ref}$
$=0-\sum_x\pi_{ref}(x)\,\nabla_\theta\log\pi_\theta(x)$ || 概率和恒为 1
$=\nabla_\theta\sum_x\pi_{ref}(x)\big(\log\pi_{ref}(x)-\log\pi_\theta(x)\big)=\nabla_\theta\KL[\pi_{ref}\Vert\pi_\theta]$ || 补一个与 $\theta$ 无关的项
:::

第二行的第一项用了对数导数技巧 $\pi_\theta\nabla\log\pi_\theta=\nabla\pi_\theta$；最后一行补上的 $\sum_x\pi_{ref}\log\pi_{ref}$ 不含 $\theta$，梯度为 0，补上它只是为了认出 KL 的样子。

::: eq k3 的期望梯度 | (8.23)
$$\E_{\pi_\theta}\big[\nabla_\theta k_3\big]=\nabla_\theta\KL\big[\pi_{ref}\Vert\pi_\theta\big]\ne\nabla_\theta\KL\big[\pi_\theta\Vert\pi_{ref}\big]$$
值估计的是 $\KL[\pi_\theta\Vert\pi_{ref}]$，梯度优化的却是方向相反的 $\KL[\pi_{ref}\Vert\pi_\theta]$。
:::

Tang 与 Munos 在 2025 年的论文里专门讨论了这件事，原话是“把 KL 估计量当作损失函数求导，一般得不到正确的梯度估计”（“Differentiating KL estimates as loss functions does not usually produce the right estimate”），并指出 k3 这样用，期望梯度是另一方向 KL 的梯度；在序列上，逐 token 的 k3 损失甚至不是任何现有序列级散度的（部分）梯度 [@tang2025kl]。两个方向的 KL 在两分布很接近时二阶相同（上面 k1、k2 的展开是同一个道理），所以 KL 很小时差别不大，这也是 Tang 与 Munos 的推测；DeepSeekMath 的 $\beta=0.04$、单步更新设置下实际影响有多大，没有实验数据。

::: err “GRPO 的 KL 项是无偏的，所以它在优化 $\KL[\pi_\theta\Vert\pi_{ref}]$”
前半句只对 KL 的**值**成立，而且要求样本来自 $\pi_\theta$。后半句不成立：把 k3 放进损失自动求导，期望梯度是 $\nabla\KL[\pi_{ref}\Vert\pi_\theta]$ (8.23)。“无偏估计量”和“无偏梯度”是两回事。
:::

DeepSeek 自己在 V3.2 里修正了它：把 k3 乘上重要性比 $\rho=\pi_\theta/\pi_{\theta_{old}}$，并指出原来的 k3 在 $\pi_\theta\ll\pi_{ref}$ 时会给出不成比例的、无界的权重（“disproportionately large, unbounded weights”）[@deepseekai2025v32]。在 $\theta=\theta_{old}$ 处求导看看修正后的样子：

::: deriv V3.2 的修正：乘上重要性比
$\nabla_\theta\big[\rho\,k_3\big]=k_3\,\nabla_\theta\rho+\rho\,\nabla_\theta k_3=\rho\,\big[k_3+(1-w)\big]\nabla_\theta\log\pi_\theta$ || 乘积法则
$=\rho\,\big[(w-1-\log w)+(1-w)\big]\nabla_\theta\log\pi_\theta=-\rho\log w\,\nabla_\theta\log\pi_\theta$ || 代入 k3
$\E_{\pi_{\theta_{old}}}\big[-\rho\log w\,\nabla\log\pi_\theta\big]=\E_{\pi_\theta}\big[-\log w\,\nabla\log\pi_\theta\big]$ || 重要性采样
$=\sum_x\pi_\theta\,\log\frac{\pi_\theta}{\pi_{ref}}\,\nabla\log\pi_\theta+\sum_x\pi_\theta\,\nabla\log\pi_\theta$ || 展开；第二项为 0
$=\sum_x\Big[\nabla\pi_\theta\,\log\frac{\pi_\theta}{\pi_{ref}}+\pi_\theta\,\nabla\log\frac{\pi_\theta}{\pi_{ref}}\Big]=\nabla_\theta\KL[\pi_\theta\Vert\pi_{ref}]$ || 乘积法则倒着用
:::

第一行用了 $\nabla_\theta\rho=\rho\,\nabla_\theta\log\pi_\theta$（$\pi_{\theta_{old}}$ 是常数）和上面算过的 $\nabla_\theta k_3=(1-w)\nabla_\theta\log\pi_\theta$。第三行的 $\rho=\pi_\theta/\pi_{\theta_{old}}$ 正好把“在 $\pi_{\theta_{old}}$ 下取期望”换成“在 $\pi_\theta$ 下取期望”。第四行添上的 $\sum_x\pi_\theta\nabla\log\pi_\theta=\sum_x\nabla\pi_\theta=0$；第五行把它和第一项合起来，正好是 $\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}$ 的梯度（$\pi_{ref}$ 不依赖 $\theta$）。

::: eq V3.2 修正后的 KL 梯度系数（ρ = 1 处） | (8.24)
$$\nabla_\theta\big(-\beta\,\rho\,k_3\big)=\beta\,\rho\,\log w\;\nabla_\theta\log\pi_\theta,\qquad\text{系数从 }\beta(w-1)\text{ 变为 }\beta\log w$$
期望梯度变成了正确方向的 $-\beta\nabla\KL[\pi_\theta\Vert\pi_{ref}]$；系数只按对数增长。
:::

::: fig figures/l08-kl-coef.svg | 图 8.10 KL 项在梯度系数里的贡献：原始 k3 的 β(w − 1) 与 V3.2 修正后的 β log w | wide scroll
**看右半边两条线的差距：$w=100$ 时原始 k3 的系数是 3.96，约为绿色虚线（优势 1）的 4 倍；修正后只有 0.184。** 横轴是参考比 $w=\pi_{ref}/\pi_\theta$ 的对数轴，$\beta=0.04$ 取 DeepSeekMath 的设置。两条曲线由 (8.22)(8.24) 计算。
:::

V3.2 还报告，在数学这类领域，用很弱的 KL 甚至不用 KL 效果更好 [@deepseekai2025v32]。Hugging Face TRL 当前的 GRPO 实现也默认 $\beta=0$，一旦设成正数，默认用的就是 V3.2 式的“k3 乘重要性比” [@trl-grpo]。所以今天说“GRPO 的 KL 项”，要先问清楚是哪一版。

::: key 本页要点
- 单样本 KL 估计：k1 $=-\log w$ 无偏但可负、方差大；k2 $=\frac12(\log w)^2$ 非负、方差小但有偏；k3 $=(w-1)-\log w$ 由控制变量 $w-1$（期望为 0）构造，无偏且非负（切线与 $\log$ 之间的距离）。
- “无偏”指 KL 的值、样本来自 $\pi_\theta$、单个位置；“为正”应为“非负”。
- $-\beta k_3$ 的梯度系数是 $\beta(w-1)$：把策略拉回参考模型，但在 $\pi_\theta\ll\pi_{ref}$ 时线性无界（$w=100$ 时 3.96）。
- 把 k3 当损失，期望梯度是 $\nabla\KL[\pi_{ref}\Vert\pi_\theta]$；V3.2 乘上重要性比后变成 $\nabla\KL[\pi_\theta\Vert\pi_{ref}]$，系数变为 $\beta\log w$。
:::

# 结果监督、过程监督与迭代 RL | 奖励从哪来，怎样分到每个 token，奖励模型怎样跟上策略 | 30

前两页把 GRPO 的“骨架”讲完了：组采样、组内标准化、裁剪、KL。这一页回答骨架之外的三个问题：奖励本身从哪来？只给整条回答一个分数，能不能细一点？策略越训越强之后，给它打分的奖励模型会不会跟不上？最后给出 DeepSeekMath 的训练设置和结果。

## 奖励来自一个神经网络，不是规则

今天提起 GRPO，很多人会联想到“用答案对不对当奖励”（基于规则的可验证奖励）。DeepSeekMath 里**不是**这样：它训练了神经网络奖励模型。奖励模型的训练集按 Math-Shepherd 的方法构造，初始奖励模型基于 DeepSeekMath-Base 7B，学习率 $2\times10^{-5}$；论文的统一范式表（[§7](#p7)）里，GRPO 与 PPO 的奖励函数一栏都写着 “Model”，意思是训练一个奖励模型给回答打分；论文同时说明，这个奖励模型的训练数据本身是按规则（答案是否正确）判定的 [@shao2024deepseekmath]。基于规则的奖励是后来 R1 的选择（[第 13 课](lesson-13-r1-reasoning.html)）。

## 结果监督：一个分数，平摊给每个 token

[§4](#p4) 的 (8.13) 就是**结果监督**：奖励模型只看完整回答，给一个分数；标准化后，这个回答的每个 token 共用同一个优势。它简单，但粗：一篇解答前十步都对、最后一步算错，所有 token 一起被惩罚；前面蒙对了关键一步、后面写得一塌糊涂却碰巧答案正确，所有 token 一起被鼓励。

## 过程监督：每一步打分，优势取后缀和

**过程监督**（process supervision）让奖励模型在推理的**每一步**结束时打一个分，这种奖励模型叫**过程奖励模型**（process reward model，PRM）。记第 $i$ 个回答有 $K_i$ 步，第 $j$ 步的最后一个 token 的位置是 $\tau_i(j)$（论文记作 $index(j)$），PRM 在这里给的分数是 $r_i^{(j)}$。把一组 $G$ 个回答的**所有**步奖励放在一起，记作集合 $\mathcal S_q=\{r_i^{(j)}:\ i=1..G,\ j=1..K_i\}$，用它的均值和标准差归一化 [@shao2024deepseekmath]：

::: eq 过程奖励的组内归一化 | (8.25)
$$\widetilde r_i^{(j)}=\frac{r_i^{(j)}-\mathrm{mean}(\mathcal S_q)}{\mathrm{std}(\mathcal S_q)}$$
论文把 $\mathcal S_q$ 写作 $\mathbf R$。注意归一化的范围是整组所有回答的所有步，而不是一个回答内部。
:::

然后，每个 token 的优势是**从它所在的这一步开始、到回答结束**的所有步的归一化奖励之和：

::: eq 过程监督的优势：后缀和 | (8.26)
$$\hat A_{i,t}=\sum_{j:\ \tau_i(j)\ge t}\widetilde r_i^{(j)}$$
即“结束位置不早于 $t$ 的那些步”。第 $j$ 步里的所有 token 共用同一个值，它等于第 $j$ 步及其后各步的归一化奖励之和。
:::

为什么是后缀和？回忆 [§3](#p3)：一个动作的好坏，要看它**之后**拿到的所有奖励，而不只是眼前这一步（GAE 在 $\lambda=1$、$\gamma=1$ 时就是“之后奖励之和减基线”）。(8.26) 就是这个思路，只是基线换成了组内归一化里的均值。这仍然是一个选择：也可以只用当前步的分数，或者给后面的步打折扣，论文没有比较这些变体。

::: ex 手算过程监督的优势（笔者构造的数字）
一组两个回答。$o_1$ 三步，PRM 分数 $(0.9,0.8,0.2)$；$o_2$ 两步，$(0.6,0.9)$。

- 整组 5 个步奖励：均值 $0.68$；总体方差 $\tfrac15(0.22^2+0.12^2+0.48^2+0.08^2+0.22^2)=\tfrac15(0.348)=0.0696$，标准差 $0.2638$。
- 归一化：$o_1$ 为 $(+0.834,+0.455,-1.819)$，$o_2$ 为 $(-0.303,+0.834)$。
- 后缀和：$o_1$ 第 3 步的 token：$-1.819$；第 2 步：$0.455-1.819\approx-1.365$；第 1 步：$0.834+0.455-1.819\approx-0.531$。$o_2$ 第 2 步：$+0.834$；第 1 步：$-0.303+0.834=+0.531$。（加法用未舍入的归一化值做，所以末位可能与三位小数直接相加差 1。）

$o_1$ 的第一步本身得分很高（0.9），它的 token 却拿到负优势：后缀和把后面那个坏步也算了进来。反过来，$o_2$ 第一步只有 0.6 分，归一化后为负，但因为后面一步很好，它的 token 仍被鼓励。
:::

::: fig figures/l08-process-adv.svg | 图 8.11 过程监督下每个 token 的优势（上排）与结果监督（下排）的对照 | wide scroll
**看 $o_1$ 的三段颜色由浅到深：越靠后的步，后缀和里剩下的项越少，但那个 0.2 分的坏步始终在里面，把前两步也拉成了负值。** 每个小方格代表一个 token，边框颜色区分不同的步；上方写着每步的 PRM 分数和归一化值。下排是结果监督的对照（设 $o_1$ 答错、$o_2$ 答对）：整条回答同一个优势。数字由脚本 `l08_process_adv.py` 计算，步内 token 个数只是示意。
:::

**步级分数从哪来？** 让人给每一步打标签很贵。Math-Shepherd [@wang2024mathshepherd] 的办法是自动标注：对一条解答的第 $j$ 步，从这一步之后用一个“续写模型”（completer）续写 $m$ 条完整解答，看有多少条能走到正确答案 $a^*$：

::: eq Math-Shepherd 的两种步级标签 | (8.27)
$$y^{HE}_{j}=\begin{cases}1,&\exists\,\ell:\ a_\ell=a^*\\0,&\text{否则}\end{cases}\qquad y^{SE}_{j}=\frac1m\sum_{\ell=1}^{m}\mathbb I(a_\ell=a^*)$$
HE（hard estimation）：只要有一条续写能做对，这一步就算“好”；SE（soft estimation）：做对的比例。$a_\ell$ 是第 $\ell$ 条续写的最终答案，$\mathbb I$ 是指示函数。Math-Shepherd 原文把步记作 $s_i$、续写条数记作 $N$、续写下标记作 $j$；这里改成与 (8.25)(8.26) 一致的记号（$j$ 仍是步的下标）。
:::

直觉是：如果从某一步出发，怎么续写都做不对，这一步多半已经走错了。Math-Shepherd 训练 PRM 时用的是 HE 标签 [@wang2024mathshepherd]；DeepSeekMath 说它的奖励模型训练集按 Math-Shepherd 构造，但没说用的是 HE 还是 SE，也没有报告训练集的规模。

## 迭代 RL：让奖励模型跟上策略

奖励模型是在**旧策略**写出的回答上训练的。RL 进行一段时间后，策略写出的回答越来越不像它当初见过的样子，奖励模型在这些新回答上的判断就可能不准，甚至被策略钻了空子。DeepSeekMath 的对策叫**迭代 RL**（iterative RL）：每隔一段，用当前策略的采样结果重新生成奖励模型的训练数据，用一种**回放机制**（混入 10% 的历史数据）继续训练奖励模型；同时把参考模型重置为当前策略，然后用新奖励模型继续训练策略 [@shao2024deepseekmath]。论文的算法 1 可以写成三层循环：

```text
输入：初始策略 π_init，奖励模型 r_φ，题目集 D，超参数 ε、β、μ
π_θ ← π_init
for iteration = 1 … I:                         # 外层：迭代 RL
    π_ref ← π_θ                                 # 参考模型重置为当前策略
    for step = 1 … S:                           # 中层：普通的 GRPO 训练步（论文记作 M）
        从 D 采一批题 D_b
        π_θold ← π_θ
        对每道题 q ∈ D_b，用 π_θold 采 G 个回答 {o_i}
        用 r_φ 给每个回答打分，按 (8.13) 或 (8.26) 算组相对优势
        for GRPO iteration = 1 … μ:              # 内层：同一批数据上的更新次数
            最大化 GRPO 目标 (8.16)，更新 π_θ
    用 π_θ 的采样结果 + 10% 历史数据，继续训练 r_φ（回放）
输出：π_θ
```

几点说明。第一，最内层的 $\mu$ 是“同一批数据上更新几次”（沿用论文记号，与 [§5](#p5) 高斯例子里的均值 $\mu$ 无关），DeepSeekMath 实际取 1（[§4](#p4)）；它和最外层的“迭代 RL”是两回事，别混淆。第二，每个外层迭代开始时 $\pi_{ref}\leftarrow\pi_\theta$，KL 约束的锚点随之前移：策略每一轮只需要“别离上一轮的自己太远”，而不是“别离最初的 SFT 模型太远”。第三，论文算法 1 的第 11 行写的是“maximizing the GRPO objective (Equation 21)”，但式 (21) 是梯度系数，GRPO 的目标是式 (3)（即本课 (8.16)），这是论文的一个笔误。

::: fig figures/l08-iter-curves.svg | 图 8.12 迭代 RL：DeepSeekMath-Instruct 7B 上三轮迭代的成绩（据论文图 6 重绘） | wide scroll
**看灰线和蓝线的衔接：第 0 轮在 GSM8K 上约 86 附近已经走平，换了奖励模型的第 1 轮一开始就到了 87.0，这是提升最大的一步；第 2 轮只在 88 附近再加一点。** 纵轴是 Top1 成绩；起点 82.9、46.8 与表 5 的 Instruct 完全相同，由此推断它和表 5 是同一口径（论文没有明说是否贪心解码）。数据由研究笔记从论文 PDF 的矢量路径中解析、每 500 步取一个点，误差约 ±0.1 个百分点 [@shao2024deepseekmath]。原图横轴的刻度标为 0、1300、2300……，后两轮为什么从约 1,100 步与 2,700 步开始，论文没有解释。
:::

论文的观察是：迭代 RL 做了两轮，第一轮的提升最明显 [@shao2024deepseekmath]。笔者的理解是：初始奖励模型与策略之间的分布差距，在第一轮 RL 之后最大，所以重训一次奖励模型的收益也最大；论文没有专门分析原因。

## 训练设置与结果

| 项 | DeepSeekMath 的设置 |
|---|---|
| 起点 | DeepSeekMath-Instruct 7B |
| 训练题目 | SFT 数据中与 GSM8K、MATH 相关的思维链格式题目，约 144K 道 |
| 奖励 | 神经奖励模型（按 Math-Shepherd 构造数据；初始 RM 基于 Base 7B，学习率 $2\times10^{-5}$） |
| 策略学习率 | $1\times10^{-6}$ |
| KL 系数 $\beta$ | 0.04 |
| 每题采样数 $G$ | 64 |
| 最大长度 | 1024 tokens |
| 训练 batch | 1024（论文没说单位是回答条数还是题数） |
| 每次探索后的更新 | 1 次 |
| 未报告 | 裁剪系数 $\varepsilon$、采样温度、总步数、每轮迭代的步数 |

表中数字均出自论文 §4.2 [@shao2024deepseekmath]。训练题目只取了 GSM8K、MATH 相关的思维链题目，其余 SFT 题目是**有意排除**的：作者想看 RL 对“RL 阶段没见过的数据”有没有用。

::: ex 采样量的量级
遍历一次 144K 道题，每题 64 个回答：$144{,}000\times64=9{,}216{,}000$ 条回答。如果 batch 1024 指的是回答条数，每一步就是 $1024/64=16$ 道题（这是笔者的推断，论文没有说明）。每条回答最长 1024 个 token，一次遍历最多约 $9.4\times10^9$ 个生成 token。
:::

结果（论文表 5，Top1，不投票）[@shao2024deepseekmath]：

| 设定 | 模型 | GSM8K | MATH | MGSM-zh | CMATH |
|---|---|---|---|---|---|
| 思维链 | Instruct 7B | 82.9% | 46.8% | 73.2% | 84.6% |
| 思维链 | RL 7B | 88.2% | 51.7% | 79.6% | 88.8% |
| 思维链 | 提升 | +5.3 | +4.9 | +6.4 | +4.2 |
| 工具集成 | Instruct 7B | 83.7% | 57.4% | 72.0% | 84.3% |
| 工具集成 | RL 7B | 86.7% | 58.8% | 78.4% | 87.6% |
| 工具集成 | 提升 | +3.0 | +1.4 | +6.4 | +3.3 |

RL 只用了与 GSM8K、MATH 相关的思维链题目，却在中文基准（MGSM-zh、CMATH）和工具集成推理上也都提高了，这就是作者说的“RL 在所有指标上都超过了 Instruct” [@shao2024deepseekmath]。论文没有明说最终发布的 DeepSeekMath-RL 7B 用的是结果监督还是过程监督、迭代了几轮。

::: key 本页要点
- DeepSeekMath 的 GRPO 用神经网络奖励模型（Math-Shepherd 式数据），不是规则奖励。
- 结果监督：一个分数平摊给全部 token。过程监督：PRM 给每步打分，整组所有步一起归一化 (8.25)，token 的优势是本步及之后各步归一化奖励之和 (8.26)。
- Math-Shepherd 用“从这一步续写 $m$ 次能否做对”自动给步骤打标签 (8.27)。
- 迭代 RL：每轮用新采样 + 10% 历史数据继续训练奖励模型，并把参考模型重置为当前策略；两轮中第一轮提升最大。
- 设置：$G=64$、$\beta=0.04$、学习率 $10^{-6}$、最大长度 1024、约 144K 道题；RL 后 GSM8K 88.2%、MATH 51.7%，没训练过的中文基准也提高了。
:::

# 统一范式：五种方法，一个梯度形状 | SFT、RFT、DPO、PPO、GRPO 的梯度系数逐一推导 | 30

到这里我们已经见过好几种后训练方法：SFT（第 3 课）、DPO（第 3 课）、PPO（[§3](#p3)）、GRPO（[§4](#p4)），还有一个本页才正式出场的拒绝采样微调。它们的目标函数长得完全不同，一个是交叉熵，一个是 $\log\sigma$，一个带裁剪。DeepSeekMath 在讨论部分提出了一个看它们的统一角度 [@shao2024deepseekmath]：**不看目标函数，看梯度**。所有这些方法的梯度，都是在把某些 token 的对数概率往上推或往下压，区别只在“推哪些 token”和“推多大力”。

## 统一的形状

::: eq 统一范式（DeepSeekMath 式 5） | (8.28)
$$\nabla_\theta\mathcal J_{\mathcal A}(\theta)=\E_{(q,o)\sim\mathcal D}\Big[\frac1{\abs{o}}\sum_{t=1}^{\abs{o}}\underbrace{\mathrm{GC}_{\mathcal A}(q,o,t,\mathcal R)}_{\text{梯度系数}}\,\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\Big]$$
三个组成部分：**数据源** $\mathcal D$（题目和回答从哪来）、**奖励函数** $\mathcal R$（用什么判断好坏）、**算法** $\mathcal A$（怎样把数据和奖励加工成梯度系数 GC）。论文把奖励函数写作 $\pi_{rf}$，注意它**不是**参考模型 $\pi_{ref}$。
:::

**含义。** 梯度系数（gradient coefficient，GC）是乘在每个 token 的 $\nabla\log\pi_\theta$ 前面的那个数。

**理解。** 正的 GC 让这个 token 以后更容易被写出，负的让它更难被写出，绝对值越大力度越大，为 0 就不管它。

**数学。** 下面逐一推出每种方法的 GC。有一个约定要先说清：这里的“梯度”是对**已经采好的一批数据**求的损失梯度，样本本身当作常数，这正是训练代码实际计算的东西。

## SFT：系数恒为 1

监督微调在人工标注的数据集 $P_{sft}(Q,O)$ 上做最大似然（论文式 6，按回答长度平均）：

::: deriv SFT 的梯度系数
$\mathcal J_{\text{SFT}}(\theta)=\E_{(q,o)\sim P_{sft}}\Big[\frac1{\abs{o}}\sum_{t}\log\pi_\theta(o_t\mid q,o_{<t})\Big]$ || 目标：人工解答的平均对数似然
$\nabla_\theta\mathcal J_{\text{SFT}}=\E_{(q,o)\sim P_{sft}}\Big[\frac1{\abs{o}}\sum_{t}1\cdot\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\Big]$ || 数据与 $\theta$ 无关，梯度直接进期望
:::

::: eq SFT | (8.29)
$$\mathrm{GC}_{\text{SFT}}=1,\qquad \mathcal D=P_{sft}(Q,O)$$
每个人工写的 token 都被同样力度地推高。没有奖励函数：数据本身就被当作“好的”。
:::

## RFT 与 Online RFT：只推高答对的回答

**拒绝采样微调**（rejection sampling fine-tuning，RFT）[@yuan2023rft]：让 SFT 模型对每道题采样若干回答，只保留答案正确的，再在这些回答上做 SFT。写成一个目标，就是给每个回答乘上一个“是否答对”的指示函数 $\mathbb I(o)$（答对为 1，答错为 0）（论文式 8–10）：

::: deriv RFT 的梯度系数
$\mathcal J_{\text{RFT}}(\theta)=\E_{q\sim P_{sft}(Q),\,o\sim\pi_{sft}(\cdot\mid q)}\Big[\frac1{\abs{o}}\sum_{t}\mathbb I(o)\,\log\pi_\theta(o_t\mid q,o_{<t})\Big]$ || 答错的回答乘 0，等于被丢掉
$\nabla_\theta\mathcal J_{\text{RFT}}=\E\Big[\frac1{\abs{o}}\sum_{t}\mathbb I(o)\,\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\Big]$ || $\mathbb I(o)$ 由规则判定，与 $\theta$ 无关
:::

**Online RFT** 与 RFT 只有一处不同：回答不是从冻结的 SFT 模型 $\pi_{sft}$ 采的，而是从**正在训练的策略** $\pi_\theta$ 实时采的（论文式 11）。

::: eq RFT 与 Online RFT | (8.30)
$$\mathrm{GC}_{\text{RFT}}=\mathrm{GC}_{\text{OnRFT}}=\mathbb I(o)=\begin{cases}1,&o\text{ 的答案正确}\\0,&o\text{ 的答案错误}\end{cases}$$
RFT：$o\sim\pi_{sft}$（离线）；Online RFT：$o\sim\pi_\theta$（在线）。
:::

对比 [§3](#p3) 的策略梯度 (8.6) 就会发现：Online RFT 的梯度恰好是“奖励为 $\mathbb I(o)$、**不减基线**”的 REINFORCE（除了 $1/\abs{o}$ 的长度平均）。这是笔者的读法，但它让下面的比较很清楚：撇开 KL 项和奖励的来源（规则还是奖励模型），Online RFT 与 GRPO 的差别，就是有没有基线、以及要不要除以标准差。没有基线的后果是：答错的回答系数为 0，**不被惩罚**，只是被忽略。

## DPO：成对比较，力度由“排错了多少”决定

DPO（[第 3 课 §7](lesson-03-alignment-rlhf.html#p7)）用成对的数据：同一道题的一个较好回答 $o^+$ 和一个较差回答 $o^-$（第 3 课记作 $o_w$、$o_l$；这里用上标 $\pm$，是为了让下面的 $\hat r^\pm$、$\Delta^\pm$ 能一次写出两个）。按原论文的式 (7) [@rafailov2023dpo]，最大化

$$
\mathcal J_{\text{DPO}}(\theta)=\E\Big[\log\sigma\big(\hat r^{+}-\hat r^{-}\big)\Big],\qquad \hat r^{\pm}=\beta\log\frac{\pi_\theta(o^\pm\mid q)}{\pi_{ref}(o^\pm\mid q)} .
$$

$\hat r$ 叫**隐式奖励**：策略比参考模型更偏爱这个回答多少（乘以 $\beta$）。$\sigma$ 是 sigmoid 函数 $\sigma(u)=1/(1+e^{-u})$。

记 $u=\hat r^+-\hat r^-$。求导要用到 sigmoid 的两个性质：$\sigma'(u)=\sigma(u)\big(1-\sigma(u)\big)$，以及 $1-\sigma(u)=\sigma(-u)$。合起来，$\frac{d}{du}\log\sigma(u)=\frac{\sigma'(u)}{\sigma(u)}=1-\sigma(u)=\sigma(-u)$。

::: deriv DPO 的梯度系数
$\nabla_\theta\log\sigma(u)=\sigma(-u)\,\nabla_\theta u$ || 链式法则
$\nabla_\theta u=\beta\big[\nabla_\theta\log\pi_\theta(o^+\mid q)-\nabla_\theta\log\pi_\theta(o^-\mid q)\big]$ || $\pi_{ref}$ 冻结
$\nabla_\theta\log\pi_\theta(o^\pm\mid q)=\sum_t\nabla_\theta\log\pi_\theta(o^\pm_t\mid q,o^\pm_{<t})$ || 逐 token 求和
$\nabla_\theta\log\sigma(u)=\beta\,\sigma(\hat r^--\hat r^+)\big[\nabla\log\pi_\theta(o^+\mid q)-\nabla\log\pi_\theta(o^-\mid q)\big]$ || 合并前三行
:::

最后一行用了 $-u=\hat r^--\hat r^+$；再对数据取期望，就是 $\nabla_\theta\mathcal J_{\text{DPO}}$。

::: eq DPO | (8.31)
$$\mathrm{GC}_{\text{DPO}}=\begin{cases}+\beta\,\sigma\big(\hat r^--\hat r^+\big),&o^+\text{ 的 token}\\-\beta\,\sigma\big(\hat r^--\hat r^+\big),&o^-\text{ 的 token}\end{cases}$$
与 DPO 原论文给出的梯度一致 [@rafailov2023dpo]。$\sigma(\hat r^--\hat r^+)$ 在隐式奖励“排反了”（$\hat r^->\hat r^+$）时大，排对了时小。注意这里的系数乘的是整句的 $\sum_t\nabla_\theta\log\pi_\theta$（原版 DPO 不除以长度）；按 (8.28) 的约定（先乘 $1/\abs{o}$ 再对 token 求和），要写成 $\pm\beta\abs{o^\pm}\,\sigma(\hat r^--\hat r^+)$，见下面的警告框。
:::

::: ex DPO 的系数有多大（β = 0.1）
记 $\Delta^\pm=\log\frac{\pi_\theta(o^\pm)}{\pi_{ref}(o^\pm)}$，则 $\hat r^\pm=0.1\Delta^\pm$。

- 训练开始，$\pi_\theta=\pi_{ref}$，$\Delta^\pm=0$：$\sigma(0)=0.5$，系数 $\pm0.05$。
- 已经排对：$\Delta^+=2,\ \Delta^-=-2$：$\sigma(-0.2-0.2)=\sigma(-0.4)\approx0.401$，系数 $\pm0.040$。
- 排反了：$\Delta^+=-1,\ \Delta^-=+1$：$\sigma(0.1+0.1)=\sigma(0.2)\approx0.550$，系数 $\pm0.055$。

排错越多，推得越狠；这和 [§4](#p4) 里 GRPO“按奖励差异分配力度”是同一种精神。
:::

::: warn DeepSeekMath 附录里的 DPO 和原版不完全一样
DeepSeekMath 附录的式 (12) 用的是**按长度平均**的对数概率比（每条回答除以 $\abs{o}$），这样它的 GC 才能落进 (8.28) 的 $1/\abs{o}$ 形状；式 (14) 又把 GC 写成了逐 token 比值的函数，并且省掉了前面的 $\beta$。从式 (12) 真正求导，系数应当是整句（长度平均后）比值的函数、对所有 token 相同，并带因子 $\beta$（可以并入学习率）。式 (12) 的负例项还有一个笔误（分子分母都写成了 $o^-_{<t}$）。本课以原版 DPO 为准；DeepSeekMath 的写法可以看成示意。原版 DPO 对整句的 token **求和**而不是平均，所以若要硬套进 (8.28) 的形状，系数还要再乘上 $\abs{o}$。
:::

## PPO：系数是 GAE 优势

[§4](#p4) 已经在 GRPO 上做过同样的事。PPO 的目标是 (8.11)；论文附录同样假设每次探索后只更新一次，于是求梯度时 $\theta=\theta_{old}$、$\rho_t=1$，min 与 clip 都不起作用（论文式 15–17）：

::: deriv PPO 的梯度系数
$\min\big[\rho_tA_t,\ \clip(\rho_t,1-\varepsilon,1+\varepsilon)A_t\big]=\rho_tA_t$ || $\rho_t=1$ 在区间内部，两项相同
$\nabla_\theta\big(\rho_tA_t\big)=A_t\,\rho_t\,\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})$ || 对数导数技巧，$A_t$ 不求导
$=A_t\,\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})$ || 代入 $\rho_t=1$
:::

::: eq PPO | (8.32)
$$\mathrm{GC}_{\text{PPO}}=A_t\quad(\text{由奖励模型、价值网络与 GAE 算出})$$
每个 token 有自己的系数。论文把这一栏的奖励函数记为“Model”（奖励模型）。
:::

## GRPO：组相对优势加 KL 拉力

把 (8.17) 读成 (8.28) 的形状，GC 就是括号里的量（论文式 19–21）：

::: eq GRPO | (8.33)
$$\mathrm{GC}_{\text{GRPO}}=\hat A_{i,t}+\beta\,\big(w_{i,t}-1\big),\qquad w_{i,t}=\frac{\pi_{ref}(o_{i,t}\mid q,o_{i,<t})}{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}$$
第一项来自组内标准化的奖励 (8.13) 或过程监督的后缀和 (8.26)；第二项是 k3 的梯度 (8.22)。V3.2 的修正版把第二项换成 $\beta\log w_{i,t}$ (8.24)。
:::

## 一张表看全部

::: fig figures/l08-unified.svg | 图 8.13 统一范式：六种方法的数据源、奖励函数与梯度系数（据论文表 10 扩展） | wide scroll
**看最右一栏的同一组四个回答：RFT 与 Online RFT 给两个错误回答的系数是 0（不管它们），GRPO 给的是 −1（主动压低它们）。** “数据源”一栏的黄框是离线数据（来自冻结的 SFT 模型），蓝框是在线数据（来自正在训练的策略）。奖励一栏照录论文表 10：RFT、DPO、Online RFT 是规则，PPO、GRPO 是模型 [@shao2024deepseekmath]。例子一栏的 DPO 取 $\beta=0.1$、训练开始时的值，GRPO 用总体标准差、忽略 KL 项。
:::

::: ex 同一组回答在各方法下被怎样对待
一道题采了四个回答，奖励 $(1,0,0,1)$：第 1、4 个答对。

- RFT / Online RFT：GC $=(1,0,0,1)$。两个正确回答被推高，两个错误回答被忽略。
- GRPO：均值 0.5，总体标准差 0.5，GC $=(+1,-1,-1,+1)$（外加很小的 KL 项）。正确的推高，错误的压低，而且“推”和“压”的总量相等。
- 如果奖励变成 $(1,0,0,0)$：RFT 仍然只推高第一个（系数 1）；GRPO 给 $(+1.732,-0.577,-0.577,-0.577)$，唯一答对的那个被推得更狠，因为它“比同组好得多”。
:::

## 论文的观察：数据要在线，系数要有正有负

统一范式不只是换个写法，它提出了两个可以做实验的问题：**数据源**（离线还是在线）重要吗？**梯度系数**（只有 0 和 1，还是按奖励有正有负、有大有小）重要吗？论文在 DeepSeekMath-Instruct **1.3B** 上比较了 RFT、Online RFT、GRPO+结果监督（OS）、GRPO+过程监督（PS）[@shao2024deepseekmath]（表 10 里的 DPO 与 PPO 没有出现在这张图里）。

::: fig figures/l08-methods-curves.svg | 图 8.14 四种方法在 DeepSeekMath-Instruct 1.3B 上的训练曲线（据论文图 5 重绘） | wide scroll
**看右侧从下到上的四个数：RFT < Online RFT < GRPO+OS < GRPO+PS，两个基准上顺序相同。** 灰色虚线的 RFT 在 2,000 步以后基本不再上升；其余三条都用在线数据。曲线为每 1,000 步取样，右侧数字为最后 5 个点（每 200 步一点）的平均，数据由研究笔记从论文 PDF 的矢量路径解析，误差约 ±0.1 个百分点 [@shao2024deepseekmath]。
:::

1. **在线胜过离线。** Online RFT 与 RFT 的梯度系数完全相同，只有数据源不同；训练前期两者相当，后期 Online RFT 明显领先（GSM8K 末段均值 61.9 对 60.1，MATH 29.0 对 27.9）。论文的解释是：训练早期策略和 SFT 模型很像，两种采样差别不大；越往后，策略采出的数据与 SFT 模型的差别越大，实时采样的优势也就越大 [@shao2024deepseekmath]。
2. **系数有正有负、有大有小，胜过只有 0/1。** GRPO+OS 与 Online RFT 都是在线数据，差别在梯度系数：GRPO 按奖励的数值给出不同幅度的强化和惩罚，而 Online RFT 对所有正确回答一视同仁、不惩罚错误回答。GRPO+OS 更好（63.2 对 61.9，29.3 对 29.0）。
3. **过程监督胜过结果监督。** GRPO+PS 最好（64.9，29.7）：更细的奖励让梯度系数也更细。

作者由此得出的结论是：这些方法都可以看作“直接或简化的 RL”，而要改进 RL，可以分别在数据源、奖励函数、算法三个部件上下功夫 [@shao2024deepseekmath]。这三个部件在 [§8](#p8) 还会出现一次。

::: key 本页要点
- 统一范式 (8.28)：梯度 = 数据源上的期望 × 每 token 的梯度系数 GC × $\nabla\log\pi_\theta$。
- GC：SFT 为 1；RFT / Online RFT 为 $\mathbb I(o)$（错的不罚）；DPO 为 $\pm\beta\sigma(\hat r^--\hat r^+)$（排错越多推得越狠）；PPO 为 GAE 优势 $A_t$；GRPO 为 $\hat A_{i,t}+\beta(w_{i,t}-1)$。
- Online RFT 就是奖励为 0/1、不减基线的 REINFORCE；GRPO 在它的基础上加了组基线与标准化，于是会惩罚错误回答。
- 1.3B 上的实测：在线 > 离线；有正有负的系数 > 0/1 系数；过程监督 > 结果监督。
:::

# RL 改变了什么，以及总账 | Maj@K 升、Pass@K 不升；GRPO 的收益、代价与后来的修改 | 25

[§6](#p6) 的表显示 RL 让 MATH 的 Top1 从 46.8% 升到 51.7%。一个自然的问题是：这 4.9 个百分点是模型“学会了原来不会的题”，还是“原来就会、只是不稳定的题，现在更稳了”？DeepSeekMath 用两个指标把这两件事分开了。

## 两个指标：Pass@K 与 Maj@K

**含义。** 对每道题采样 $K$ 个回答。**Pass@K**：$K$ 个里只要有一个答对，这道题就算解出。**Maj@K**：对 $K$ 个回答的最终答案做多数投票，票数最多的答案正确才算解出。

**理解。** Pass@K 量的是“能力的边界”：只要模型有一定概率写出正确答案，采得够多总能碰上。Maj@K 量的是“稳定性”：正确答案必须是模型最常给出的那个答案。一道题，模型 10% 的时候答对、90% 的时候给出同一个错误答案，Pass@64 几乎是 1，Maj@64 几乎是 0。

**数学。** 设对某题采了 $n\ge K$ 个样本，其中 $c$ 个正确。用这 $n$ 个样本估计 Pass@K，常用的无偏估计是“从 $n$ 个里不放回地随机抽 $K$ 个、至少一个正确”的概率 [@yue2025rlvr]：

::: eq Pass@K 的无偏估计 | (8.34)
$$\text{Pass@}K=\E_{q}\left[1-\frac{\binom{n-c}{K}}{\binom{n}{K}}\right]$$
$\binom{n-c}{K}/\binom{n}{K}$ 是抽到的 $K$ 个全是错误样本的概率。“无偏”的意思是：若这道题单次答对的概率是 $p_q$，且 $n$ 个样本独立同分布，则方括号里的量的期望正好是 $1-(1-p_q)^K$（$n$ 个独立样本里随机取的 $K$ 个，本身就是 $K$ 个独立样本）。这个估计式出自 Chen 等人 2021 年的代码评测工作，这里经 Yue 等人的论文转引；DeepSeekMath 没有说明它用哪种方式估计 Pass@K。
:::

::: eq Maj@K | (8.35)
$$\text{Maj@}K=\E_q\Big[\mathbb I\Big(\argmax_{a}\,\#\{j\le K:\ a_j=a\}=a^*\Big)\Big]$$
$a_j$ 是第 $j$ 个回答的最终答案，$a^*$ 是正确答案，$\#\{\cdot\}$ 是计数。平票怎么处理，论文没有说明。
:::

::: ex Pass@K 的估计（n = 64，c = 4）
Pass@1 $=1-\frac{60}{64}=0.0625$，就是正确率。Pass@4 $=1-\frac{\binom{60}{4}}{\binom{64}{4}}=1-\frac{60\cdot59\cdot58\cdot57}{64\cdot63\cdot62\cdot61}=1-\frac{11{,}703{,}240}{15{,}249{,}024}\approx0.2325$。Pass@16 $\approx0.6938$。只有 6% 的时候答对的题，给 16 次机会就有七成把握“解出”。
:::

## 观察：RL 提升了 Maj@K，没有提升 Pass@K

论文在温度 0.7 下测了 Instruct 与 RL 两个 7B 模型在 GSM8K、MATH 上的这两个指标，结论是一句话：“RL enhances Maj@K’s performance but not Pass@K”（§5.2.2）[@shao2024deepseekmath]。

::: fig figures/l08-majk-passk.svg | 图 8.15 RL 前后的 Maj@K（实线）与 Pass@K（虚线）（据论文图 7 重绘） | wide scroll
**看两对虚线：K = 1 时蓝线（RL）明显高于灰线（Instruct），往右越靠越近，最后在红圈处交叉，K = 64 时 RL 反而略低。再看两对实线：蓝线始终在灰线上方。** 灰色是 DeepSeekMath-Instruct 7B，蓝色是 DeepSeekMath-RL 7B，温度 0.7。K 按原图等距排列。数据由研究笔记从论文 PDF 的矢量路径解析，误差约 ±0.1 个百分点 [@shao2024deepseekmath]；交叉位置由线性插值算出。
:::

把图 8.15 的差值（RL 减 Instruct）列出来：

| K | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| GSM8K Maj@K | +6.7 | +3.5 | +2.5 | +1.2 | +1.5 | +1.5 |
| GSM8K Pass@K | +6.7 | +0.7 | −0.6 | −0.9 | −1.7 | −1.8 |
| MATH Maj@K | +6.3 | +5.1 | +2.5 | +1.3 | +1.2 | +1.0 |
| MATH Pass@K | +6.3 | +2.5 | +1.1 | +0.2 | −0.3 | −0.7 |

作者的解释是：RL 让输出分布更“稳健”，提升来自把原本已经在前几名里的正确答案推到前面（原文说是 “boosting the correct response from TopK”），而不是增强了模型的基本能力 [@shao2024deepseekmath]。论文正文只说 Pass@K“没有提升”；从数字看，大 $K$ 时 RL 模型的 Pass@K 还**略低**，GSM8K 在 $K=4$ 与 8 之间交叉，MATH 在 16 与 32 之间交叉。

## 一个两道题的玩具模型：“锐化”是什么意思

“提升 Maj@K、不提升 Pass@K”在分布上意味着什么？用一个极简的例子看清楚。设每道题有三种可能的最终答案，第一种是正确的；RL 前后模型在两道题上的答案分布如下（笔者构造的数字，Maj@K 用多项分布精确枚举，平票随机打破）。

::: ex RL 把分布“锐化”：会的题更稳，不太会的题更难碰上
| 题 | 答案分布（正确, 错误1, 错误2） | K=1 | Maj@4 / Pass@4 | Maj@16 / Pass@16 | Maj@64 / Pass@64 |
|---|---|---|---|---|---|
| Q1 RL 前 | (0.40, 0.35, 0.25) | 0.40 | 0.436 / 0.870 | 0.524 / 1.000 | 0.661 / 1.000 |
| Q1 RL 后 | (0.60, 0.25, 0.15) | 0.60 | 0.729 / 0.974 | 0.937 / 1.000 | 0.999 / 1.000 |
| Q2 RL 前 | (0.10, 0.50, 0.40) | 0.10 | 0.040 / 0.344 | 0.002 / 0.815 | 0.000 / 0.999 |
| Q2 RL 后 | (0.02, 0.70, 0.28) | 0.02 | 0.002 / 0.078 | 0.000 / 0.276 | 0.000 / 0.726 |
| 两题平均 RL 前 | | 0.250 | 0.238 / 0.607 | 0.263 / 0.907 | 0.330 / 0.999 |
| 两题平均 RL 后 | | 0.310 | 0.365 / 0.526 | 0.469 / 0.638 | 0.500 / 0.863 |

Pass@K 可以手算：Q1 RL 前 Pass@4 $=1-0.6^4=0.870$；Q2 RL 后 Pass@64 $=1-0.98^{64}\approx0.726$。

RL 在 Q1 上把本来就是最常见答案的正确答案推得更高（Maj@64 从 0.661 到 0.999），在 Q2 上却把本来就少见的正确答案压得更少（0.10 → 0.02）。平均下来：K=1 与 Maj@K 升了，大 K 的 Pass@K 降了（0.999 → 0.863），方向与图 8.15 一致。这个玩具模型（不是论文数据）只说明“锐化”在指标上长什么样。在 GRPO 里，Q2 这种题（64 个回答里约 6 个正确）的正确回答会拿到很大的正优势，按说应被推高；RL 之后它为什么反而变少，论文没有给出机制。笔者能想到的一种可能是：训练用的是神经奖励模型，它未必给少见的正确解法打高分；这只是推测。
:::

## 这个结论能推多远

这是一个**实测**，适用范围要说清楚：Instruct → RL，GSM8K 与 MATH（RL 训练题目的同分布基准），温度 0.7，$K\le64$。论文没有给置信区间。笔者按完整测试集（GSM8K 1,319 题、MATH 5,000 题；实际用了多少题本课未核实）粗算 $K=64$ 处的二项标准误，把两个模型当作独立样本：$\sqrt{\big(p_1(1-p_1)+p_2(1-p_2)\big)/n}$，GSM8K 为 $\sqrt{(0.991\times0.009+0.973\times0.027)/1319}\approx0.52$ 个百分点，MATH 为 $\sqrt{(0.875\times0.125+0.868\times0.132)/5000}\approx0.67$ 个百分点。GSM8K 的 −1.8 约为 3.5 个标准误，MATH 的 −0.7 只有约 1 个。（两个模型其实在同一批题上测，配对比较的标准误通常更小；这里只求量级。）所以“Pass@K 不升”可以说，“RL 让大 K 的 Pass@K 下降”在 MATH 上证据很弱。

作者自己推测了原因：RL 只用了指令微调阶段的题目，采样也只用了朴素的 {{nucleus 采样::也叫 top-p 采样：每一步只在累计概率达到 p 的那一小撮最可能的 token 里按概率抽一个}}（[M04 §5](../math/m04-probability.html#p5)）；并提出了三个改进方向，恰好对应 [§7](#p7) 统一范式的三个部件：**数据源**（分布外的题目、基于树搜索等更好的采样）、**算法**（对有噪声的奖励更鲁棒，从弱到强的对齐）、**奖励函数**（泛化能力、不确定性、更高质量的过程奖励模型）[@shao2024deepseekmath]。

一年多以后，Yue 等人在“从基座模型直接做可验证奖励 RL”的设置下，把 $K$ 推到 128、256 乃至 1024（视基准而定），得到了同方向且更强的结论：RL 让 pass@1 上升，但可解题目的覆盖面（pass@256）下降 [@yue2025rlvr]。两者的设置不同（Instruct → RL 对 Base → RLVR，$K\le64$ 对 $K$ 最大到 128–1024），不要混为同一个结论。DeepSeek 自己后来的做法在 [第 13 课](lesson-13-r1-reasoning.html)：R1-Zero 从 V3-Base 出发、改用基于规则的奖励、把回答上限放宽到 32,768 个 token（第 8,200 步起再加倍到 65,536）[@deepseekai2025r1]，结果如何留到那一课讨论。

## 总账一：PPO 与 GRPO

| | PPO（RLHF 式） | GRPO（DeepSeekMath） |
|---|---|---|
| 显存里的模型 | 策略、价值、参考、奖励 | 策略、参考、奖励 |
| 需要训练的 | 策略 + 价值 | 策略 |
| 模型状态（6.91B，笔者估算） | 248.8 GB | 138.2 GB（−44.4%） |
| 基线 | 价值网络 $V_\psi(s_t)$ | 同一题 $G$ 个回答的平均奖励 |
| 优势的粒度 | 每个 token（GAE） | 每个回答（结果监督）或每一步（过程监督） |
| KL 放在哪、用什么估计 | 每个 token 的奖励里，k1 | 损失里，k3 |
| 每道题采样 | 不需要成组（论文图 4 画的是 1 个） | $G=64$ 个 |
| 每道题采样的 KV cache（1024 token，BF16） | 480 MiB | 30 GiB |
| 需要小心的选择 | GAE 的 $\lambda$、价值网络的训练 | 除以 std（难度偏差）、$1/\abs{o_i}$（长度偏差）、k3 的梯度方向、全对/全错组无信号 |

GRPO 用“多采样”换掉了“多一个要训练的模型”。这笔交换值不值，取决于显存与采样哪个更紧：在 DeepSeekMath 的 7B 规模上，省下的 110.6 GB 模型状态，大约等于 3.4 道题（每题 64 个回答、各 1024 token）的 KV cache。R1 附录 A.3 的对照还提醒我们，PPO 并不是“必然差于 GRPO”：把 GAE 的 $\lambda$ 调到 1.0 后，两者接近。R1 论文自己的结论是：GRPO 的优势在于省掉了多训练一个价值模型的显存与计算，也省掉了为 PPO 调 $\lambda$ 所花的额外算力 [@deepseekai2025r1]。

## 总账二：GRPO 后来被改了哪几处

“GRPO 的超参数”必须说清是哪篇论文、哪个实现。下表只列研究笔记核对过的项（R1 的细节属于第 13 课，V3.2 属于第 15 课）：

| 项 | DeepSeekMath（2024-02） | DeepSeek-R1-Zero | DeepSeek-V3.2 | TRL 默认（2026-09） |
|---|---|---|---|---|
| 奖励 | 神经奖励模型 | 基于规则 | 推理与智能体：规则 + 长度惩罚 + 语言一致性；通用任务：生成式奖励模型 | 用户提供 |
| 组大小 $G$ | 64 | 16 | 未披露 | 8 |
| KL 系数 $\beta$ | 0.04 | 0.001 | 按领域调，数学可弱或为 0 | 0（不加 KL） |
| 优势 | $(r_i-\text{mean})/\text{std}$ | 同左 | 印出的公式为 $r_i-\text{mean}$（不除 std；前一句却说“归一化”） | 组内，样本 std + $10^{-4}$ |
| KL 估计 | k3 | k3（序列级写法） | k3 × 重要性比 | $\beta>0$ 时为 k3 × 重要性比 |
| 每批采样后的更新 | 1 次 | 分 16 个小批 | 分多个小批（个数未披露） | 1 次 |
| 裁剪 $\varepsilon$ | 未报告 | 未写（R1 第一阶段 RL 为 10） | 未披露 | 0.2 |
| 参考模型更新 | 每个外层迭代 | 每 400 步 | 未披露 | — |

来源：DeepSeekMath §4.2 [@shao2024deepseekmath]；R1 v2 §2.1 与 §3.2.1 [@deepseekai2025r1]；V3.2 §3（混合 RL）与 §3.1 [@deepseekai2025v32]，细节见[第 15 课 §4–§6](lesson-15-v3-2-agents.html#p4)；TRL commit c8ef2433 [@trl-grpo]。可以看到，“组采样 + 组内比较 + 去掉价值网络”这个骨架没变；变的是标准化、KL 的形式与强度、奖励的来源。

## 这一课留下了什么

把整课压成几句话：

1. **数据**：一个便宜的 fastText 分类器加上“按域名扩种子、人工只标 URL 路径”的迭代，四轮从 Common Crawl 召回了 120B tokens 数学网页；同样训 150B tokens，它全面胜过更小的公开数学语料。
2. **GRPO**：同一题采 $G$ 个回答，用组内标准化的奖励当优势，去掉了与策略同规模的价值网络；KL 从奖励挪进损失，用 k3 估计。
3. **k3**：对 KL 的值无偏且非负，但当损失求导时，期望梯度指向 $\KL[\pi_{ref}\Vert\pi_\theta]$，而且在 $\pi_\theta\ll\pi_{ref}$ 时系数线性无界；V3.2 用重要性比修正。
4. **统一范式**：SFT、RFT、DPO、PPO、GRPO 的梯度都是“某个系数 × $\nabla\log\pi$”；在线数据与有正有负的系数更好。
5. **RL 做了什么**：在这篇论文的设置下，RL 让已经会的题更稳（Maj@K、Top1 上升），没有扩大能解的题（Pass@K 不升）。

下一课（[第 9 课](lesson-09-v2-mla.html)）回到架构与 KV cache：DeepSeek-V2 用 MLA 把每个 token 的缓存压小。本页“每道题 30 GiB 的 KV cache”说明，这类压缩对 RL 的大量采样同样有用（这是笔者的联系，V2 论文讲的是推理）。GRPO 本身则会在 [第 13 课](lesson-13-r1-reasoning.html)（R1）和 [第 15 课](lesson-15-v3-2-agents.html)（V3.2）再次出场。

::: key 本页要点
- Pass@K：$K$ 次里有一次对就算对，量能力边界；Maj@K：多数投票对才算对，量稳定性。
- 论文原话：“RL enhances Maj@K’s performance but not Pass@K”；数字上大 $K$ 时 RL 的 Pass@K 还略低（GSM8K 更明显）。
- 机制是分布“锐化”：常见的正确答案被推高，罕见的正确答案被压低。这是 Instruct → RL、$K\le64$ 下的实测，不是定理。
- GRPO 用多采样（每题 30 GiB KV）换掉价值网络（110.6 GB 模型状态）；骨架在 R1、V3.2 中保留，标准化、KL 与奖励来源都被改过。
:::

# 练习与答案 | 12 题：概念、计算、推导 | 50

先自己做，再展开答案。计算题用计算器就够；想核对的话，每题都能用几行 NumPy 验证。

## 概念题

:::: quiz 练习 1 · GRPO 去掉了什么，没去掉什么
判断下面两句话，错的请改正：（a）“GRPO 去掉了奖励模型和参考模型，所以只剩一个模型。”（b）“GRPO 用的是基于规则的可验证奖励。”
::: answer 看答案
（a）错。GRPO 只去掉了**价值模型**（critic）。DeepSeekMath 的 GRPO 仍有策略、参考、奖励三个模型（图 8.6），其中只有策略要训练。显存的节省来自价值模型的权重、梯度与优化器状态：按 (8.2) 估算，6.91B 规模下约 110.6 GB，占 PPO 模型状态的 44.4%。

（b）在 DeepSeekMath 里错。它训练了神经网络奖励模型（按 Math-Shepherd 构造数据，初始化自 Base 7B），论文表 10 中 GRPO 的奖励函数一栏写的是 “Model”；迭代 RL 还会继续训练这个奖励模型。基于规则的奖励是后来 R1 的选择。“GRPO = 可验证奖励 RL”是后来的用法。
:::
::::

:::: quiz 练习 2 · 全对或全错的组
（a）一组回答的奖励是 $(1,1,1,1)$ 或 $(0,0,0,0)$ 时，这道题对策略梯度有什么贡献？KL 项还起作用吗？（b）由此，你会怎样挑选 RL 阶段的训练题目？（c）$G$ 从 4 增大到 64，对这个问题有什么影响？
::: answer 看答案
（a）偏差全为 0，标准差也为 0，(8.13) 是 $0/0$；实现里分母加一个小数（TRL 为 $10^{-4}$），优势全为 0。于是 (8.17) 里只剩 KL 项 $\beta(w_{i,t}-1)\nabla\log\pi_\theta$：这道题只把策略拉向参考模型，不提供任何“怎样做对”的信号。

（b）题目的难度要落在“组内有对有错”的区间：对当前策略来说太简单（总是全对）或太难（总是全错）的题，都是在浪费采样。

（c）若单个回答答对的概率为 $p$，且组内各回答独立采样，一组 $G$ 个回答全对的概率是 $p^G$、全错的概率是 $(1-p)^G$。$G$ 越大，这两种情况越少，例如 $p=0.9$ 时 $0.9^4\approx0.66$，$0.9^{64}\approx0.0012$。代价是每题的采样与 KV cache 增加 16 倍。
:::
::::

:::: quiz 练习 3 · 裁剪为什么没起作用
DeepSeekMath 说策略“在每个探索阶段之后只更新一次”。（a）如果这指每批数据只做一次优化器更新，GRPO 的梯度是什么？$\varepsilon$ 取 0.1 还是 0.3 会改变结果吗？（b）如果这指把一批数据分成 16 个小批、各更新一次，情况有什么不同？
::: answer 看答案
（a）唯一一次求梯度发生在 $\theta=\theta_{old}$，$\rho_{i,t}=1$，落在裁剪区间内部，min 两边相同，梯度是 (8.17)：$\frac1G\sum_i\frac1{\abs{o_i}}\sum_t[\hat A_{i,t}+\beta(w_{i,t}-1)]\nabla\log\pi_\theta$，与 $\varepsilon$ 无关。这时 GRPO 等于“带组均值基线、除以组内标准差的 REINFORCE 加 KL 正则”。

（b）第一个小批仍在 $\theta_{old}$ 处，但从第二个小批起参数已经变了，$\rho\ne1$，裁剪开始起作用，$\varepsilon$ 的取值就重要了。R1 正是这样做的（8,192 个输出分 16 个小批），并在第一阶段 RL 里把 $\varepsilon$ 设为 10。DeepSeekMath 没有说明小批的个数，所以（a）的结论依赖于论文附录的简化读法。
:::
::::

## 计算题

:::: quiz 练习 4 · 五份答卷的组内优势
$G=5$，奖励 $\mathbf r=(1,0,1,1,0)$。（a）分别用总体标准差与样本标准差算优势。（b）两者的比值是多少，与 $\sqrt{(G-1)/G}$ 一致吗？（c）只减均值、再乘 $G/(G-1)$，与留一基线的优势对照。
::: answer 看答案
（a）均值 $0.6$。偏差：答对 $+0.4$，答错 $-0.6$；平方和 $3\times0.16+2\times0.36=1.2$。总体方差 $1.2/5=0.24$，标准差 $0.4899$：答对 $+0.4/0.4899=+0.816$，答错 $-0.6/0.4899=-1.225$。样本方差 $1.2/4=0.3$，标准差 $0.5477$：答对 $+0.730$，答错 $-1.095$。

（b）$0.730/0.816=0.894$，$\sqrt{4/5}=0.894$，一致。

（c）$\frac54(r_i-0.6)$：答对 $+0.5$，答错 $-0.75$。留一基线：答对者 $1-\frac14(0+1+1+0)=1-0.5=+0.5$；答错者 $0-\frac14(1+1+1+0)=-0.75$。完全相同，正是 (8.15)。
:::
::::

:::: quiz 练习 5 · KL 的拉力
$\beta=0.04$。（a）按 (8.22) 求 $w=0.5,2,10,100$ 时 KL 项的梯度系数，并与 V3.2 修正后的 $\beta\log w$ 对照。（b）$w$ 多大时，原始 k3 的 KL 系数等于 1（与两对两错时的优势一样大）？修正版呢？
::: answer 看答案
（a）原始 $\beta(w-1)$：$-0.020$、$+0.040$、$+0.360$、$+3.960$。修正 $\beta\log w$：$-0.0277$、$+0.0277$、$+0.0921$、$+0.1842$。在 $w$ 接近 1 时两者差不多（$\log w\approx w-1$），$w$ 大时原始版线性增长，修正版只按对数增长。

（b）原始：$0.04(w-1)=1\Rightarrow w=26$，即策略写这个 token 的概率只有参考模型的 1/26 时，KL 拉力就和优势一样大了。修正：$0.04\log w=1\Rightarrow w=e^{25}\approx7.2\times10^{10}$，实际上碰不到。
:::
::::

:::: quiz 练习 6 · 显存与 KV cache 的另一种算法
（a）如果 PPO 的价值模型不和策略一样大，而是只有 1.3B 参数，按 (8.2) 的规则（训练 16 字节/参数、冻结 2 字节/参数，策略、参考、奖励模型都是 6.91B），PPO 的模型状态是多少？GRPO 相对它省多少？（b）DeepSeekMath 7B 每 token 的 KV cache 是 480 KiB。如果把组大小改成 $G=16$、最大长度改成 4096，一道题的 KV cache 是多少？
::: answer 看答案
（a）$16\times6.91+16\times1.3+2\times2\times6.91=110.56+20.8+27.64\approx159.0$ GB。GRPO 仍是 $138.2$ GB，只省 $1-138.2/159.0\approx13.1\%$。小价值模型能缩小差距，但价值模型越小，对每个 token 的价值估计通常越粗，这是另一种代价。

（b）$16\times4096\times480$ KiB $=31{,}457{,}280$ KiB $=30$ GiB。与 $G=64$、长度 1024 完全相同，因为 $16\times4096=64\times1024$：KV cache 只取决于一道题采样的总 token 数。
:::
::::

:::: quiz 练习 7 · 过程监督的优势
一组两个回答。$o_1$ 两步，PRM 分数 $(0.5,0.9)$；$o_2$ 三步，$(0.7,0.3,0.8)$。用总体标准差，按 (8.25)(8.26) 求每一步 token 的优势。哪一步“本身得分低、优势却为正”？
::: answer 看答案
整组 5 个步奖励的均值 $0.64$；偏差 $(-0.14,0.26,0.06,-0.34,0.16)$，平方和 $0.232$，总体方差 $0.0464$，标准差 $0.2154$。

归一化：$o_1$ 为 $(-0.650,+1.207)$；$o_2$ 为 $(+0.279,-1.578,+0.743)$。

后缀和：$o_1$ 第 2 步 $+1.207$，第 1 步 $-0.650+1.207=+0.557$。$o_2$ 第 3 步 $+0.743$，第 2 步 $-1.578+0.743\approx-0.836$（未舍入的值是 $-0.8356$），第 1 步 $0.279-0.836=-0.557$。

$o_1$ 的第 1 步只有 0.5 分（归一化后为负），但因为后面一步很好，它的 token 优势为 $+0.557$。$o_2$ 的第 1 步有 0.7 分，却因为后面的坏步，优势为 $-0.557$。两条回答第 1 步的优势正好互为相反数，因为同一组所有步的归一化奖励之和为 0，而两条回答第 1 步的后缀和各自覆盖了自己的全部步。
:::
::::

## 推导题

:::: quiz 练习 8 · 高斯例子里三个估计量的波动
设样本 $x\sim\mathcal N(0,1)$，参考分布 $\mathcal N(\mu,1)$，于是 $Y=\log w=\mu x-\mu^2/2\sim\mathcal N(-\mu^2/2,\ \mu^2)$，真 KL $=\mu^2/2$。已知正态变量 $Y\sim\mathcal N(m,s^2)$ 满足

$$\E[e^{aY}]=e^{am+a^2s^2/2},\qquad \Var(Y^2)=2s^4+4m^2s^2 .$$

- （a）证明 k1 的相对标准差是 $2/\mu$。
- （b）证明 k2 的偏差/真值 $=\mu^2/4$，相对标准差 $=\sqrt{2+\mu^2}$。
- （c）证明 $\Var(k_3)=e^{\mu^2}-1-\mu^2$。
- （d）$\mu=1$ 时三个相对标准差各是多少？
::: answer 看答案
记 $m=-\mu^2/2$，$s^2=\mu^2$。

（a）$k_1=-Y$，$\E[k_1]=\mu^2/2$（无偏），标准差 $s=\mu$，相对标准差 $\mu/(\mu^2/2)=2/\mu$。

（b）$k_2=Y^2/2$，$\E[k_2]=\frac12(m^2+s^2)=\frac12\big(\frac{\mu^4}{4}+\mu^2\big)=\frac{\mu^2}2+\frac{\mu^4}8$，偏差 $\mu^4/8$，偏差/真值 $=\mu^2/4$。$\Var(k_2)=\frac14\Var(Y^2)=\frac14(2\mu^4+\mu^6)$，标准差 $\frac{\mu^2}{2}\sqrt{2+\mu^2}$，除以真值得 $\sqrt{2+\mu^2}$。

（c）$k_3=e^Y-1-Y$，$\Var(k_3)=\Var(e^Y)+\Var(Y)-2\Cov(e^Y,Y)$。由给出的公式：$\E[e^Y]=e^{m+s^2/2}=1$，$\E[e^{2Y}]=e^{2m+2s^2}=e^{\mu^2}$，所以 $\Var(e^Y)=e^{\mu^2}-1$。对 $a$ 求导得 $\E[Ye^{aY}]=(m+as^2)e^{am+a^2s^2/2}$，取 $a=1$：$\E[Ye^Y]=m+s^2=\mu^2/2$；于是 $\Cov(e^Y,Y)=\mu^2/2-m\cdot1=\mu^2$。合起来 $\Var(k_3)=e^{\mu^2}-1+\mu^2-2\mu^2=e^{\mu^2}-1-\mu^2$。

（d）k1：2；k2：$\sqrt3\approx1.732$；k3：$\sqrt{e-2}/0.5=2\times0.8475\approx1.695$。与 Schulman 博客的 2、1.73、1.7 一致，也就是图 8.9 右图 $\mu=1$ 处的三个点。
:::
::::

:::: quiz 练习 9 · 二值奖励与难度偏差
（a）按 (8.14)，写出一道题所有回答的 $\sum_i\abs{\hat A_i}$（除以标准差）与 $\sum_i\abs{r_i-\bar r}$（不除）的比值，它只依赖于什么？（b）$G=64$ 时，只对 1 个的题与对一半的题，这个比值各是多少？除以标准差让前者相对后者的权重放大了几倍？
::: answer 看答案
（a）除以标准差就是每个优势都乘以 $1/\mathrm{std}=1/\sqrt{p(1-p)}$，所以比值 $=\frac{2G\sqrt{p(1-p)}}{2Gp(1-p)}=\frac1{\sqrt{p(1-p)}}$，只依赖正确率 $p$。

（b）$p=1/64$：$1/\sqrt{(1/64)(63/64)}=64/\sqrt{63}\approx8.06$；$p=1/2$：$2$。极难题相对中等题的权重被放大了 $8.06/2\approx4.0$ 倍。这就是 Dr. GRPO 所说的难度偏差；去掉标准差后，每道题的总权重是 $2Gp(1-p)$，在 $p=1/2$ 时最大。
:::
::::

:::: quiz 练习 10 · 在两个 token 的词表上检查 k3 的梯度
词表只有两个 token $a,b$。当前策略 $\pi_\theta(a)=\sigma(z)$，在 $z=0$ 处 $\pi_\theta=(0.5,0.5)$；参考模型 $\pi_{ref}=(0.8,0.2)$。（a）求 $\partial_z\log\pi_\theta(a)$ 与 $\partial_z\log\pi_\theta(b)$。（b）求 $\E_{\pi_\theta}[\partial_zk_3]$。（c）分别求 $\partial_z\KL[\pi_{ref}\Vert\pi_\theta]$ 与 $\partial_z\KL[\pi_\theta\Vert\pi_{ref}]$，与（b）对照，并说明 V3.2 的修正会给出哪一个。
::: answer 看答案
（a）$\partial_z\log\sigma(z)=1-\sigma(z)=0.5$；$\partial_z\log(1-\sigma(z))=-\sigma(z)=-0.5$。

（b）$w(a)=0.8/0.5=1.6$，$w(b)=0.2/0.5=0.4$。$\partial_zk_3=(1-w)\,\partial_z\log\pi_\theta$。期望 $=0.5\times(1-1.6)\times0.5+0.5\times(1-0.4)\times(-0.5)=-0.15-0.15=-0.30$。

（c）$\partial_z\KL[\pi_{ref}\Vert\pi_\theta]=-\sum_x\pi_{ref}(x)\partial_z\log\pi_\theta(x)=-(0.8\times0.5+0.2\times(-0.5))=-0.30$，与（b）相同，印证 (8.23)。$\partial_z\KL[\pi_\theta\Vert\pi_{ref}]=\sum_x\pi_\theta(x)\log\frac{\pi_\theta(x)}{\pi_{ref}(x)}\partial_z\log\pi_\theta(x)=0.25\log0.625-0.25\log2.5\approx-0.1175-0.2291=-0.347$，与（b）不同。两者方向相同（沿负梯度下降都会让 $z$ 增大、$\pi_\theta(a)$ 向 0.8 靠拢），大小相差 0.047。V3.2 乘上重要性比后，期望梯度是后一个，$-0.347$。
:::
::::

:::: quiz 练习 11 · 长度归一化 DPO 的梯度系数
DeepSeekMath 附录用的 DPO 把整句对数概率比换成按长度平均的版本：$\mathcal J=\E\big[\log\sigma(u)\big]$，其中

$$u=\frac{\beta}{\abs{o^+}}\sum_t\log\frac{\pi_\theta(o^+_t\mid\cdot)}{\pi_{ref}(o^+_t\mid\cdot)}-\frac{\beta}{\abs{o^-}}\sum_t\log\frac{\pi_\theta(o^-_t\mid\cdot)}{\pi_{ref}(o^-_t\mid\cdot)} .$$

推导它的梯度，写成 (8.28) 的形状，并说明它的 GC 是否依赖于 $t$。
::: answer 看答案
和 (8.31) 的推导一样：$\nabla\log\sigma(u)=\sigma(-u)\nabla u$，而

$$\nabla u=\beta\Big[\frac1{\abs{o^+}}\sum_t\nabla\log\pi_\theta(o^+_t\mid\cdot)-\frac1{\abs{o^-}}\sum_t\nabla\log\pi_\theta(o^-_t\mid\cdot)\Big].$$

所以 $o^+$ 的每个 token 的 GC 都是 $+\beta\sigma(-u)$，$o^-$ 的每个 token 都是 $-\beta\sigma(-u)$，前面恰好带着 (8.28) 需要的 $1/\abs{o}$。$u$ 是整句（长度平均后）的量，所以 GC **不依赖于 $t$**。论文的式 (14) 把它写成逐 token 比值的函数、并省掉了 $\beta$，是一种简化的写法；$\beta$ 可以并入学习率，但“依赖 $t$”与求导结果不符。
:::
::::

:::: quiz 练习 12 · Pass@K 与“锐化”
（a）某题 $n=64$ 个样本里 $c=4$ 个正确，用 (8.34) 求 Pass@1、Pass@4、Pass@16。（b）玩具模型里 Q2 在 RL 后的答案分布是 $(0.02,0.70,0.28)$（第一个为正确答案）。求 Pass@64，并解释为什么它的 Maj@64 几乎为 0。（c）读图 8.15 的差值表：GSM8K 与 MATH 的 Pass@K 曲线各在哪两个 K 之间交叉？
::: answer 看答案
（a）Pass@1 $=4/64=0.0625$；Pass@4 $=1-\binom{60}{4}/\binom{64}{4}\approx0.2325$；Pass@16 $\approx0.6938$。

（b）Pass@64 $=1-0.98^{64}\approx1-0.274=0.726$。Maj@64 要求正确答案是 64 次采样里出现最多的答案，而正确答案的期望次数只有 $64\times0.02\approx1.3$ 次，错误答案 1 的期望次数约 45 次，正确答案几乎不可能成为众数。

（c）GSM8K：$K=4$ 时差 $+0.7$，$K=8$ 时 $-0.6$，在 4 与 8 之间交叉。MATH：$K=16$ 时 $+0.2$，$K=32$ 时 $-0.3$，在 16 与 32 之间交叉。注意 MATH 的差值只有零点几个百分点，约 1 个标准误（[§8](#p8)），不宜过度解读。
:::
::::

```refs
- id: yue2025rlvr
  title: "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"
  a: Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Yue, Y., Song, S., Huang, G. (2025). NeurIPS 2025 (Oral)；arXiv:2504.13837v5
  url: https://arxiv.org/abs/2504.13837
  finding: 在多个模型与基准上发现，可验证奖励的 RL 提高 pass@1，但在大 k（如 pass@256）时降低可解题目的覆盖面；附录给出 pass@k 的无偏估计式。
  matters: 本课用它把 DeepSeekMath 的 Maj@K / Pass@K 观察放到后续研究中，并说明两者的实验设置不同；Pass@K 估计式 (8.34) 经它转引。
- id: rafailov2023dpo
  title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
  a: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., Finn, C. (2023). arXiv:2305.18290v3
  url: https://arxiv.org/abs/2305.18290
  finding: 由 KL 约束下最优策略的闭式解推出只需偏好对的 DPO 损失（式 7），其梯度按隐式奖励 $\hat r_\theta=\beta\log(\pi_\theta/\pi_{ref})$ 排错的程度加权，前面带因子 $\beta$。
  matters: 本课在统一范式里推导 DPO 的梯度系数时以原版为准，并指出 DeepSeekMath 附录用的是长度归一化的变体。
- id: yuan2023rft
  title: "Scaling Relationship on Learning Mathematical Reasoning with Large Language Models"
  a: Yuan, Z., Yuan, H., Li, C., Dong, G., Lu, K., Tan, C., Zhou, C., Zhou, J. (2023). arXiv preprint arXiv:2308.01825v2
  url: https://arxiv.org/abs/2308.01825
  finding: 提出拒绝采样微调（RFT）：用模型自己采样出的正确解答扩充监督数据。
  matters: RFT 是统一范式表 10 的一行，本课用它对照 GRPO 怎样惩罚错误回答。
- id: wang2024mathshepherd
  title: "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"
  a: Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., Sui, Z. (2024). ACL 2024；arXiv:2312.08935v3
  url: https://aclanthology.org/2024.acl-long.510/
  finding: 用续写模型从中间步骤续写多条解答，以能否到达正确答案（HE）或到达的比例（SE）自动标注步骤质量，训练过程奖励模型（论文用 HE）。
  matters: DeepSeekMath 的奖励模型训练集按此构造，本课讲过程监督时用它解释“步级奖励从哪来”。
- id: schulman2020kl
  kind: artefact
  title: Approximating KL Divergence（博客）
  a: John Schulman (2020). 个人博客，页面日期 2020/03/07
  url: http://joschu.net/blog/kl-approx.html
  finding: 给出 KL 的三个单样本估计 k1、k2、k3；k3 = (r − 1) − log r 用期望为零的控制变量 r − 1 构造，既无偏又恒非负；在高斯例子里 k3 的标准差与 k2 相当甚至更低。
  matters: GRPO 式 (4) 的 KL 估计器就是 k3；本课推导它的无偏性与非负性时以此为出处，并说明博客记号与课程记号的对应。
- id: tang2025kl
  title: "On a few pitfalls in KL divergence gradient estimation for RL"
  a: Tang, Y., Munos, R. (2025). arXiv preprint arXiv:2506.09477v1
  url: https://arxiv.org/abs/2506.09477
  finding: 指出把无偏 KL 估计量当作损失自动求导一般得不到 KL 的梯度：k3 这样用，期望梯度是另一方向 KL 的梯度；逐 token 的损失在序列上至多给出部分梯度，k3 形式的甚至不是任何现有序列级散度的部分梯度。
  matters: 为本课“k3 对值无偏、对梯度有偏”的推导 (8.23) 提供独立来源。
- id: deepseekai2025v32
  title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
  a: DeepSeek-AI (2025). arXiv preprint arXiv:2512.02556v1
  url: https://arxiv.org/abs/2512.02556
  finding: §3.1 规模化 GRPO 时印出的优势公式是 $R_i-\mathrm{mean}(\boldsymbol R)$（不除以标准差，但前一句说“在组内归一化”），并用重要性比 $\pi_\theta/\pi_{old}$ 乘在 k3 前面得到无偏的 KL 估计，指出原 k3 的梯度在 $\pi_\theta\ll\pi_{ref}$ 时给出无界的大权重。
  matters: 这是 DeepSeek 对 DeepSeekMath 版 GRPO 的官方修正，本课讲难度偏差与 k3 梯度问题时以它为落脚点；它是第 15 课的主来源之一。
- id: gh-deepseek-math
  kind: artefact
  title: deepseek-ai/DeepSeek-Math（GitHub 仓库）
  a: DeepSeek-AI (2024). GitHub，main 分支，读取于 2026-09-24
  url: https://github.com/deepseek-ai/DeepSeek-Math
  finding: 仓库发布了 Base、Instruct、RL 三个检查点的使用说明与评测代码（顶层只有 evaluation/、replicate/ 等），没有 RL/GRPO 训练代码。
  matters: 说明 GRPO 的若干实现细节（例如标准差的约定）无法从官方代码核对，只能参考第三方实现。
- id: trl-grpo
  kind: artefact
  title: Hugging Face TRL：GRPOTrainer（trl/trainer/grpo_trainer.py）
  a: Hugging Face (2026). GitHub，commit c8ef2433（2026-09-23），VERSION 1.14.0.dev0
  url: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
  finding: 组内优势用 Bessel 修正的样本标准差加 $10^{-4}$ 归一化；默认 $\beta=0$（不加 KL、不加载参考模型）、$G=8$、$\varepsilon=0.2$、loss_type="dapo"；设 $\beta>0$ 时 KL 默认是乘了重要性比的 k3。
  matters: 本课用它说明“论文公式”和“常用实现”在标准差约定、KL 形式与默认超参数上的差别。
- id: schulman2015gae
  title: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  a: Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2015). arXiv:1506.02438v6
  url: https://arxiv.org/abs/1506.02438
  finding: 定义 TD 残差 $\delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t)$ 与广义优势估计 $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$；$\lambda=1$ 时无偏，$\lambda<1$ 时有偏但方差更小。
  matters: PPO 的逐 token 优势来自 GAE 与价值网络，本课用它说明 GRPO 省掉的是哪一块。
- id: deepseekai2025r1
  title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
  a: DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., et al. (2025). Nature 645:633–638; arXiv:2501.12948v2
  url: https://arxiv.org/abs/2501.12948
  finding: 用 GRPO 与基于规则的奖励在 V3-Base 上直接做 RL；给出序列级写法的 GRPO 目标与 R1-Zero、R1 的 RL 超参数，附录 A.3 用 16B MoE 模型对比 PPO 与 GRPO。
  matters: 本课用它说明 GRPO 的公式与超参数在 DeepSeek 后续工作中怎样变化，以及 PPO 对 GAE 的 $\lambda$ 有多敏感；它是第 13 课的主来源。
- id: liu2025drgrpo
  title: "Understanding R1-Zero-Like Training: A Critical Perspective"
  a: Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., Lin, M. (2025). arXiv preprint arXiv:2503.20783v2
  url: https://arxiv.org/abs/2503.20783
  finding: 指出 GRPO 的 $1/\lvert o_i\rvert$ 带来回答长度偏差、除以组内标准差带来题目难度偏差；两者都去掉（Dr. GRPO）后，优势 $r_i-\mathrm{mean}(\mathbf r)$ 乘 $G/(G-1)$ 就是 RLOO（留一基线）的优势。
  matters: 本课讲组内标准化与长度平均时，用它说明这两处是“选择”而不是“推论”，以及各自的代价。
- id: ouyang2022instructgpt
  title: "Training language models to follow instructions with human feedback"
  a: Ouyang, L., Wu, J., Jiang, X., et al. (2022). arXiv:2203.02155v1
  url: https://arxiv.org/abs/2203.02155
  finding: 三阶段 RLHF 的 PPO 阶段在每个 token 的奖励上加入相对 SFT 模型的 KL 惩罚，以缓解奖励模型被过度优化。
  matters: DeepSeekMath 式 (2) 的每 token KL 惩罚出自这里，本课用它对比 GRPO 把 KL 移进损失的做法。
- id: shao2024deepseekmath
  title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
  a: Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., Guo, D. (2024). arXiv preprint arXiv:2402.03300（读的是 v3）
  url: https://arxiv.org/abs/2402.03300v3
  finding: 用 fastText 四轮迭代从 Common Crawl 召回 120B tokens 数学语料，训练 7B 模型；提出去掉价值网络、以组内奖励标准化作优势的 GRPO（式 3、式 4），RL 后 MATH Top1 达 51.7%；§5.2 给出统一范式（式 5、表 10）与“RL 提升 Maj@K 而非 Pass@K”的观察。
  matters: 本课的主来源：数据管线、GRPO 目标与 k3 KL 估计、结果/过程监督、迭代 RL、统一梯度系数与 Maj@K/Pass@K 的数据都出自这里。
- id: paster2023openwebmath
  title: "OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text"
  a: Paster, K., Dos Santos, M., Azerbayev, Z., Ba, J. (2023). arXiv preprint arXiv:2310.06786v1
  url: https://arxiv.org/abs/2310.06786
  finding: 从 Common Crawl 提取并保留 LaTeX 的数学网页数据集，自报 14.7B tokens；筛选依次用数学字符串预过滤、语言识别、困惑度过滤和基于 fastText 的 MathScore 分类器，再做 SimHash 去重。
  matters: 它是 DeepSeekMath 数据管线第一轮的种子语料，也是表 1 的对照语料之一。
- id: dsllm
  title: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"
  a: DeepSeek-AI (Bi, X. et al.) (2024). arXiv preprint arXiv:2401.02954v1
  url: https://arxiv.org/abs/2401.02954
  finding: 收集 2T tokens 中英双语数据训练 7B 与 67B 模型，并采用多步学习率调度。
  matters: 本课用它说明通用预训练语料与数学语料在规模上的差距，以及“从衰减前检查点续训”背后的多步调度。
- id: schulman2017ppo
  title: "Proximal Policy Optimization Algorithms"
  a: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). arXiv preprint arXiv:1707.06347v2
  url: https://arxiv.org/abs/1707.06347
  finding: 提出裁剪概率比的代理目标 $L^{CLIP}$（式 7），使目标成为未裁剪目标的悲观下界；§4 另给出自适应 KL 惩罚（惩罚对旧策略的 KL）。
  matters: GRPO 直接沿用 PPO 的裁剪结构；本课用它说明 GRPO 改了什么（优势来源、KL 放置），以及为何单步更新时裁剪不起作用。
- id: rajbhandari2019zero
  title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
  a: Rajbhandari, S., Rasley, J., Ruwase, O., He, Y. (2019). arXiv:1910.02054v3
  url: https://arxiv.org/abs/1910.02054
  finding: 混合精度 Adam 训练每个参数需要 2 + 2 + 12 = 16 字节的模型状态显存。
  matters: 本课估算 PPO 与 GRPO 的显存差（价值模型的代价）时用这条规则。
- id: hf-dsmath-config
  kind: artefact
  title: deepseek-ai/deepseek-math-7b-base（及 -instruct、-rl）config.json
  a: DeepSeek-AI (2024). Hugging Face 模型仓库，读取于 2026-09-24
  url: https://huggingface.co/deepseek-ai/deepseek-math-7b-base/raw/main/config.json
  finding: 三个检查点均为 30 层、d = 4096、32 头 MHA、FFN 11008、词表 102400、4K 上下文、不共享嵌入的 LLaMA 式结构。
  matters: 论文没有架构表，本课的参数量、显存与 KV cache 算例都以这里的数字为准。
- id: guo2024deepseekcoder
  title: "DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence"
  a: Guo, D., Zhu, Q., Yang, D., Xie, Z., Dong, K., Zhang, W., et al. (2024). arXiv preprint arXiv:2401.14196v2
  url: https://arxiv.org/abs/2401.14196
  finding: §5 描述从 DeepSeek-LLM-7B Base 继续预训练得到 Coder-v1.5 7B（只用下一词预测、4K 上下文，表 9 的数据 70% 为源代码）；续训量在 §5 写作 2 trillion、§6 写作 2 billion tokens，前后矛盾。
  matters: DeepSeekMath-Base 从 Coder-Base-v1.5 的衰减前检查点初始化，图 8.1 的谱系需要它。
- id: joulin2016fasttext
  title: "Bag of Tricks for Efficient Text Classification"
  a: Joulin, A., Grave, E., Bojanowski, P., Mikolov, T. (2016). arXiv:1607.01759v3
  url: https://arxiv.org/abs/1607.01759
  finding: 提出 fastText：词与 n-gram 特征查表后取平均，接线性分类器与 softmax（模型写作 $f(BAx)$），用哈希技巧存 n-gram，训练极快。
  matters: DeepSeekMath 用 fastText 给数百亿网页打分召回数学页，本课的式 (8.3)(8.4) 与手算例子以它为准。
```
