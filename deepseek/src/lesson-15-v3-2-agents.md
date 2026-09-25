---
title: 第 15 课 · DeepSeek-V3.1 / V3.2：混合思考、规模化 RL 与智能体
short: DeepSeek-V3.2 与智能体
kicker: DeepSeek 科技树 · 第 15 课
description: V3.1 用一个 token 切换思考与非思考、839B 长上下文续训、只能取 2 的幂的 UE8M0 缩放因子；V3.2 的专家蒸馏与单阶段混合 GRPO，四项稳定化改动（无偏 KL、off-policy 序列掩码、Keep Routing、Keep Sampling Mask）逐条推导与算例；工具调用中的思考上下文管理、85,267 个智能体任务的合成管线；V3.2-Speciale 与 DeepSeekMath-V2 的竞赛成绩及其评测条件。
lede: R1 证明了强化学习能教会模型思考。接下来的一年，DeepSeek 没有再动模型的骨架（只加了第 14 课的稀疏注意力），力气全花在三件事上：让**同一个模型**既能长考也能直接回答；把后训练的算力从预训练的 0.19% 放大到 10% 以上，并让 GRPO 在这个规模上不崩；让模型在一次任务里**边想边用工具**。这一课把这三件事的每个公式、每个数字讲透，也把“金牌”这类说法背后的评测条件摆清楚。
meta: 预计阅读 200 分钟，练习另需约 50 分钟 · 前置：第 8、11、12、13、14 课，数学 M05、M08、M09
index: index.html
prev: lesson-14-sparse-attention.html | 第 14 课 · 稀疏注意力
next: lesson-16-mhc.html | 第 16 课 · mHC
---

## 这一课回答什么问题

2025 年 1 月，DeepSeek 手里有两个 671B 的模型：V3 负责日常对话，R1 负责长篇思考（[第 13 课](lesson-13-r1-reasoning.html)）。到同年 12 月发布 V3.2 时，只剩一个模型，它能思考也能不思考，能在思考的过程中调用搜索和代码工具，在数学和编程上和当时最强的闭源模型处在同一档。这中间的一年，模型的骨架几乎没变：V3.1 的结构与 V3 相同，V3.2 相对 V3.1-Terminus 唯一的架构改动是第 14 课的稀疏注意力 DSA [@hf-v31; @dsv32]。变的是数据、数值格式和后训练。这一课回答五个问题：

- **想，还是不想？** 同一组参数怎样既能长考又能直接回答？V3.1 的答案是在提示的末尾换一个 token。多轮对话里，上一轮的思考要不要留在上下文里？
- **UE8M0 是什么？** V3.1 的模型卡说权重和激活都用“UE8M0 FP8 缩放格式”。一个只能表示 2 的整数次幂的 8 位数，替换掉第 12 课的 FP32 缩放因子，省了什么，又丢了什么？
- **RL 怎样放大而不崩？** V3.2 的后训练预算超过预训练成本的 10%，而 V3 只有 0.19%。放大之后，GRPO 里原本可以忽略的四种“采样时的模型与训练时的模型不一致”变成了训练崩溃的来源。V3.2 对每一种各有一条修补：无偏 KL 估计、off-policy 序列掩码、Keep Routing、Keep Sampling Mask。它们各修的是什么？
- **智能体的训练数据从哪来？** 1,800 多个环境、85,000 多条任务是怎样造出来、又怎样筛出来的？模型在一次任务里调用十几次工具，思考内容什么时候留、什么时候删？
- **“金牌”是什么意思？** V3.2-Speciale 与 DeepSeekMath-V2 在 IMO、IOI 上的“金牌水平”，是在什么条件下、按什么方式评出来的？

## 这一课在科技树上的位置

这一课在主线 D（训练目标与后训练）上：SFT → RLHF 与 PPO（[第 3 课](lesson-03-alignment-rlhf.html)）→ GRPO（[第 8 课](lesson-08-deepseekmath-grpo.html)）→ R1 的纯 RL（[第 13 课](lesson-13-r1-reasoning.html)）→ **V3.2 的规模化 RL 与智能体（本课）** → V4 的专家训练与合并（[第 18 课](lesson-18-v4.html)）。第 8 课在讲 k3 估计器时已经预告了 V3.2 的 KL 修正；第 13 课结尾留下了三个问题，其中“规模化 RL 的稳定性”和“想还是不想”由本课回答，“长输出的注意力账单”由第 14 课回答。

本课还有一小段在主线 C（数值与系统）上：第 12 课的 FP8 细粒度量化给每 128 个数配一个 FP32 缩放因子，V3.1 把这个缩放因子限制成 2 的整数次幂，这是第 18、19 课 FP4 的前奏：第 18 课的 MXFP4 同样只用 2 的幂缩放，第 19 课的 FP4 主 KV 缓存则反过来改用 E4M3 缩放，理由是它只存储、不做 FP4 矩阵乘。下一课 [第 16 课](lesson-16-mhc.html) 转到主线 E，讲残差流的新几何 mHC。

## 读完你能……

1. 写出 V3.1 思考模式与非思考模式的提示前缀，说出多轮对话时哪些内容被删、哪些保留，以及 V3.2 在工具调用里把这条规则改成了什么。
2. 按 DeepSeek 推理内核的规则手算 UE8M0 缩放因子，推导它让 E4M3 的利用率落在 $(\tfrac12,1]$、平均约 0.721，并说出它与 MX 格式的两处不同。
3. 写出 V3.2 版 GRPO 的目标函数，说清四项稳定化改动各修的是哪一种不一致；给定一组序列的概率，手算 off-policy 掩码与 Keep Sampling Mask 下的重要性比。
4. 推导“k3 乘以重要性比”之后，估计值无偏、期望梯度指向 $\nabla\KL[\pi_\theta\Vert\pi_{ref}]$，并算出 $\pi_\theta\ll\pi_{ref}$ 时新旧两种梯度权重相差多少倍。
5. 说出 85,267 个智能体任务的来源构成，1,827 个合成环境是怎样生成和筛选的；读懂 Speciale 与 DeepSeekMath-V2 各项竞赛成绩的评测条件，并用官方奖牌线复核。

::: note 本课的记号
- $\pi_\theta$：正在训练的策略（参数 $\theta$）；$\pi_{\theta_{old}}$：采样时的策略；$\pi_{ref}$：参考策略（冻结）。**V3.2 里的 $\pi_{\theta_{old}}$ 特指推理框架在采样时直接返回的概率**，不是训练框架用旧参数重算的概率，§6 会专门讲这件事。
- $\rho_{i,t}=\pi_\theta/\pi_{\theta_{old}}$：第 $i$ 条回答第 $t$ 个 token 的**重要性比**（importance ratio）。V3.2 论文把它记作 $r_{i,t}$，本课程的 $r$ 留给奖励，所以改写为 $\rho$（与第 13 课一致）。
- $w_{i,t}=\pi_{ref}/\pi_\theta$：**参考比**，沿用第 8 课的记号。
- $r_i$：第 $i$ 条回答的奖励（论文记作 $R_i$）；$\hat A_{i,t}$：优势；$G$：组大小；$\varepsilon,\beta$：裁剪系数与 KL 系数。
- $m_i\in\{0,1\}$：off-policy 序列掩码（论文记作 $M_{i,t}$；本课程的 $M$ 是每 token 的非嵌入 FLOPs，所以改写）；$\delta$：掩码阈值。§2 的模式 token $m$、§3 警告框里的尾数 $m$ 只在各自那一小段里用，与 $m_i$ 无关。
- §3 里的 $a$ 是一个量化分块的最大绝对值（amax），$s$ 是缩放因子，$E$ 是指数字段，$u$ 是 E4M3 范围的利用率（M08 §7 把利用率记作 $\rho$，因为 M08 的 $u$ 是单位舍入；本课的 $\rho$ 已留给重要性比）。
- §8 里 DeepSeekMath-V2 的奖励权重论文记作 $\alpha,\beta$，与 KL 系数 $\beta$ 冲突，本课改写为 $w_Y=0.76$、$w_Z=0.24$。
:::

[[legend]]

[[toc]]

# 问题：R1 之后还差什么 | 一年里的七个版本、三条诊断、放大 53 倍的后训练账单 | 20

先把这一年发生了什么摆出来，再看 DeepSeek 自己怎样诊断开源模型的短板，最后算一笔后训练的账。算完这笔账，你会看到本课后半的那些稳定化改动不是锦上添花，而是规模一大就绕不过去的问题。

## 一年里的七个版本

::: fig figures/l15-timeline.svg | 图 15.1 从 DeepSeek-V3 到 V3.2 的发布时间线，以及后训练算力占比 | wide
**看上半部分的颜色标签：除了两个标“发布”、圆点与文字同色的参照点（V3 与 R1-0528），一年里只有一个蓝色（架构）节点，其余全是绿色（训练与数据）和橙色（数值格式）。** 横轴按天线性排列，日期取自 Hugging Face 的上传时间与 DeepSeek API 新闻页 [@ds-news]。下半部分是对数刻度：V3 的后训练只占预训练 GPU 小时的 0.188%，V3.2 自称“超过预训练成本的 10%”，两者相差约 53 倍（(15.1) 与下面的算例）。
:::

按时间顺序，每个版本带来的东西是：

- **V3-0324**（2025-03）：V3.2 论文回顾说，Keep Routing（§6）从这个版本起就用在 RL 训练里 [@dsv32]。
- **R1-0528**（2025-05）：第 13 课结尾讲过，它是 V3.1 比较思考能力时的对照对象。
- **V3.1**（2025-08-21）：一个模型两种模式（§2）；在 V3 基座上续训 839B token 做长上下文扩展（§2）；权重与激活改用 UE8M0 缩放格式（§3）[@hf-v31]。
- **V3.1-Terminus**（2025-09-22）：减少中英文混杂和偶发的乱码，加强代码智能体与搜索智能体 [@hf-v31t]。
- **V3.2-Exp**（2025-09-29）：在 Terminus 上续训出 DSA 稀疏注意力（第 14 课），API 降价 50% 以上 [@ds-news]。
- **DeepSeekMath-V2**（2025-11-27）：一个会自我验证的数学证明模型（§8）[@dsmathv2]。
- **V3.2 与 V3.2-Speciale**（2025-12-01 发布，论文 12-02 上 arXiv）：本课的主角 [@dsv32]。

骨架几乎没动这件事，有两条原文为证。V3.1 的模型卡写道：“The model structure of DeepSeek-V3.1 is the same as DeepSeek-V3.” [@hf-v31] V3.2 的论文写道，相对 V3.1-Terminus，“the only architectural modification of DeepSeek-V3.2 is the introduction of DeepSeek Sparse Attention (DSA) through continued training.” [@dsv32] 所以这一年的进步，除了 DSA 带来的长上下文效率，都要到训练里去找。

## V3.2 论文的三条诊断

V3.2 论文的引言把开源模型落后于闭源模型的原因归成三条 [@dsv32]。原文逐条照录，后面是它们在本门课里的去处：

1. 架构：“the predominant reliance on vanilla attention (Vaswani et al., 2017) mechanisms severely constrains efficiency for long sequences.” 普通注意力在长序列上效率太低。这一条由 DSA 回答，见第 14 课。
2. 资源分配：“open-source models suffer from insufficient computational investment during the post-training phase, limiting their performance on hard tasks.” 后训练的算力投入不够。这一条由本课 §4–§6 回答。
3. 智能体：“open-source models demonstrate a marked lag in generalization and instruction-following capabilities compared to their proprietary counterparts”。在智能体场景里，泛化与指令遵循明显落后。这一条由本课 §7 回答。

第 2 条最值得算一算，因为它给出了一个可以量化的比较。

## 后训练的账单

**含义。** 后训练（post-training）指预训练结束以后的所有训练：SFT、强化学习、蒸馏。**后训练占比**就是它花的算力（或成本）与预训练之比。

**理解。** 预训练给模型知识，后训练教它怎样用这些知识：怎样按格式回答、怎样一步步推理、怎样调用工具。R1 之前，业界普遍把后训练当作“收尾工作”，花的钱和预训练相比是零头。V3.2 的主张是：这个比例定得太低了，难任务上的差距有一部分就来自这里。

**数学。** V3 技术报告给出了三段训练各用的 H800 GPU 小时：预训练 2664K、上下文扩展 119K、后训练 5K [@dsv3]。

::: eq 后训练占比 | (15.1)
$$\text{后训练占比}=\frac{\text{后训练的算力（或成本）}}{\text{预训练的算力（或成本）}},\qquad \text{V3}:\ \frac{5\text{K}}{2664\text{K}}=0.188\%$$
V3 的三段都按 GPU 小时计，所以这里“算力”与“成本”只差一个单价。
:::

V3.2 论文在两处说到它的后训练预算，措辞不同，都照录 [@dsv32]：

“this framework allocates a post-training computational budget exceeding 10% of the pre-training cost”（§1）

“we have observed consistent performance improvements correlating with extended RL training budget, which already exceeds 10% of the pre-training cost. We hypothesize that reasoning capabilities could be further enhanced with additional computational budget allocation.”（§4.1）

::: ex 10% 相当于多少
假设“预训练成本”指 V3 的那次预训练（2664K GPU 小时；论文没有说明基数，见下面的警告）。

- 10% 是 $0.1\times2664\text{K}=266.4\text{K}$ GPU 小时。按 V3 报告用的每 GPU 小时 2 美元，是 $266.4\text{K}\times2=53.28$ 万美元。
- 与 V3 的 0.188% 相比，占比放大了 $10/0.1877\approx53.3$ 倍。
- 对照 [第 13 课 §8](lesson-13-r1-reasoning.html#p8) 的公开账单：R1（含 R1-Zero）合计约 147K GPU 小时。V3.2 的后训练“超过” 266.4K，至少是它的 1.8 倍。

这三个数都是下限（原文是“超过”），而且都建立在上面的假设上。
:::

::: warn 这个“10%”没有说清的三件事
第一，原文是“cost”（成本），不是 FLOPs，也没有给出任何绝对的 GPU 小时。第二，没有说明分母是哪一次预训练：V3 的 14.8T token？是否包含 V3.1 的 839B 续训和 DSA 的 945.8B 续训？第三，没有说明分子是否包含 §4 那些“专家模型”各自的大规模 RL。所以本课只把它当作一个数量级：后训练从“零头”变成了“一成以上”。
:::

## 放大 RL，会在哪里坏

GRPO 的骨架在第 8 课和第 13 课已经讲完：同一道题采 $G$ 个回答，按组内相对好坏给优势，用裁剪的重要性比更新策略，再用 KL 项把策略拴在参考模型附近。这个骨架里有一个默认的前提：**生成样本的那个策略，就是正在被训练的那个策略**，至多相差一两步参数更新。R1 规模的训练里，这个前提近似成立；把 RL 的预算放大几十倍，它就在四个地方同时松动。

::: ana 类比：同题多答，但答卷是昨天写的
**映射。** 沿用第 8 课的类比：GRPO = 同一道题让学生写 $G$ 份答卷，每份按“比这组平均分高多少”给分，据此纠正学生。放大训练以后，情况变成：答卷是学生**昨天**写的（采样策略 $\pi_{\theta_{old}}$），今天被纠正的是已经改变了一些的学生（$\pi_\theta$）；而且昨天的答卷是由一位**誊写员**誊抄后交上来的（推理框架），誊写员偶尔会把一个字写得和学生原本的写法不同。

**收益。** 它让四种不一致变得具体：① 用昨天的答卷估计今天的学生离“标准写法”（参考模型）有多远，估计会偏（§5）；② 有些答卷已经完全不像今天的学生会写的了，却还拿来纠正他（§6 off-policy 掩码）；③ 学生脑子里“请教哪几位老师”（MoE 路由）昨天和今天不一样（§6 Keep Routing）；④ 誊写员只誊抄了学生最有把握的那些字（top-p 截断），今天却按学生的全部可能写法来比较（§6 Keep Sampling Mask）。

**失效的地方。** 真实的“昨天”和“今天”之间只隔几个小批的梯度步，差异通常很小；类比会让人以为差异总是很大。另外，誊写员（推理框架）与学生（训练框架）其实是**同一组参数**在两套代码里运行，差异来自数值实现（算子顺序、精度、并行方式），而不是“另一个人”。
:::

| 不一致从哪来 | 规模小时 | 规模大时的症状 | V3.2 的修补 | 见 |
|---|---|---|---|---|
| KL 估计的样本来自 $\pi_{\theta_{old}}$，不是 $\pi_\theta$；k3 当损失求导，梯度方向不对 | 差别很小 | $\pi_\theta\ll\pi_{ref}$ 的 token 得到无界的大权重，梯度噪声累积 | 无偏 KL 估计：k3 乘以 $\rho$ | §5 |
| 一大批 rollout 分成多个小批更新；推理框架与训练框架的数值实现不同 | 前几个小批几乎在策略上 | 负样本离当前策略太远，误导优化 | off-policy 序列掩码 | §6 |
| MoE 路由在推理与训练时对同一输入选出不同专家 | — | 被优化的参数子空间突然跳变 | Keep Routing | §6 |
| top-p / top-k 截断让采样分布与完整 softmax 不同 | — | 重要性采样的前提被破坏，语言一致性变差 | Keep Sampling Mask | §6 |

这张表是本课后半的路线图。第一行的问题第 8 课已经指出过；后三行都来自一个工程事实：为了让 RL 快，采样用的是高度优化的推理框架，训练用的是另一套训练框架，两者算出来的概率并不逐位相同 [@dsv32]。

::: key 本页要点
- 2024-12 到 2025-12 的七个版本里，只有 V3.2-Exp 改了架构（DSA）；V3.1 结构与 V3 相同。进步来自数据、数值格式和后训练。
- V3.2 论文的三条诊断：普通注意力太贵（第 14 课）、后训练算力不足（§4–§6）、智能体泛化差（§7）。
- 后训练占比：V3 是 5K/2664K = 0.188%；V3.2 “超过预训练成本的 10%”，约 53 倍，但基数与口径都没有披露。
- 放大 RL 时，“采样的策略 = 训练的策略”这一前提在四处松动，V3.2 各有一条修补。
:::

# V3.1：一个模型，两种模式 | 用提示的最后一个 token 切换思考、多轮时删掉旧思考、839B 长上下文续训 | 25

R1 很会想，但它对每个问题都想：问它“1 加 1 等于几”，它也要先写一段思考。用户为这段思考付出的是时间和输出 token 的费用；只想要快速回答的场景，就得换用另一个模型 V3。V3.1 的做法是把两者合成一个模型，由调用者决定这一次要不要思考。它的模型卡开门见山：

“DeepSeek-V3.1 is a hybrid model that supports both thinking mode and non-thinking mode.” [@hf-v31]

## 混合思考：换一个 token

**含义。** 混合思考（hybrid thinking）：同一组参数，靠聊天模板在提示末尾放不同的 token，来决定模型这一轮先写思考再回答，还是直接回答。模型卡的原话是“One model supports both thinking mode and non-thinking mode by changing the chat template.” [@hf-v31]

**理解。** 可以把它想成同一位老师，你在问题后面加一句“请边想边说”或者“直接告诉我答案”。老师没有换人，变的只是你给的最后一句提示。对语言模型来说，“最后一句提示”就是提示的最后一个 token：它决定了下一个 token 的条件分布，从而决定了接下来写的是思考还是回答。这个比方在一处失效：老师听得懂任何措辞，模型只认训练里反复见过的那两种格式。它并不“决定”要不要想，只是在两种条件下学到了两种不同的续写分布；训练数据没有覆盖的组合，模型卡也就不承诺支持（比如 V3.1 的工具调用只在非思考模式下支持，见下文）。

**数学。** 记提示（系统提示、历史对话和当前问题按模板拼好）为 $x$，模式 token 为 $m$，模型的输出为 $o$。

::: eq 混合思考：同一组参数、两个条件 | (15.2)
$$\begin{gathered}\pi_\theta(o\mid x,m),\qquad m\in\{\texttt{<think>},\ \texttt{</think>}\}\\[4pt] o=c\ \Vert\ \texttt{</think>}\ \Vert\ a\quad(m=\texttt{<think>}),\qquad o=a\quad(m=\texttt{</think>})\end{gathered}$$
$c$：思考内容（一串 token）；$a$：回答；$\Vert$ 表示拼接。思考模式下，结束思考的 `</think>` 是模型**自己生成**的；非思考模式下，它已经在提示里了。
:::

按自回归分解，思考模式的输出概率是 $\pi_\theta(c\mid x,\texttt{<think>})\cdot\pi_\theta(\texttt{</think>}\mid x,\texttt{<think>},c)\cdot\pi_\theta(a\mid x,\texttt{<think>},c,\texttt{</think>})$：先写思考，再自己决定何时停，最后在看过思考的条件下写回答。非思考模式只剩最后一项，条件里没有 $c$。两种模式共用同一个 $\pi_\theta$，训练时只要两种格式的数据都有，模型就学会“看到 `</think>` 就直接答”。

::: fig figures/l15-hybrid-template.svg | 图 15.2 V3.1 的混合思考模板：同一组参数，靠提示的最后一个 token 切换模式
**看紫色的格子：两种模式的提示只差这一个 token，竖虚线是提示的结尾。** 非思考模式的提示以 `</think>`（id 128799）结尾，模型接着直接写回答；思考模式以 `<think>`（id 128798）结尾，模型先写思考，自己写出 `</think>`，再写回答。第 ③ 行是多轮对话：历史轮次一律写成 `</think>回答‹eos›`，上一轮的思考被删掉。模板字符串取自 V3.1 模型卡，token id 取自其 `tokenizer.json` [@hf-v31]；图中把 `<｜begin▁of▁sentence｜>` 等特殊 token 简写为 ‹bos› 等。
:::

模型卡给出的模板字符串如下（原样照录，`{…}` 是占位符）[@hf-v31]：

```text
非思考首轮：<｜begin▁of▁sentence｜>{system prompt}<｜User｜>{query}<｜Assistant｜></think>
思考首轮：  <｜begin▁of▁sentence｜>{system prompt}<｜User｜>{query}<｜Assistant｜><think>
多轮上下文：...<｜User｜>{query}<｜Assistant｜></think>{response}<｜end▁of▁sentence｜>...
```

::: ex 两个 token 是从哪里来的
V3 与 V3.1 的词表大小相同，都是 128,000 个 BPE token 加 818 个附加 token。逐个 id 比较两份 `tokenizer.json`，只有 4 个 id 的内容变了，全是把 V3 预留的占位 token 改作他用 [@hf-v31]：

| id | V3 | V3.1 |
|---|---|---|
| 128796 | `<｜place▁holder▁no▁796｜>` | `<｜search▁begin｜>` |
| 128797 | `<｜place▁holder▁no▁797｜>` | `<｜search▁end｜>` |
| 128798 | `<｜place▁holder▁no▁798｜>` | `<think>` |
| 128799 | `<｜place▁holder▁no▁799｜>` | `</think>` |

所以 V3.1 不需要扩大嵌入矩阵：`<think>` 和 `</think>` 用的是 V3 预训练时就留好的两个空位。这两个 token 在 V3.1 的词表里标为 `special: false`，它们是普通 token，可以出现在模型的输出里；非思考模式的提示以 128799 结尾，思考模式以 128798 结尾。（逐 id 比较由研究笔记完成。）
:::

## 多轮对话：旧的思考删掉，`</think>` 留下

第二轮提问时，第一轮的思考要不要放回上下文？V3.1 的规则是不放：

“It means the thinking token in the last turn will be dropped but the `</think>` is retained in every turn of context.” [@hf-v31]

写成公式，第 $k$ 轮的提示是

::: eq V3.1 第 k 轮的提示 | (15.3)
$$\begin{aligned}x_k=\ &\texttt{‹bos›}\,\text{sys}\\ &\mathop{\Big\Vert}\limits_{j=1}^{k-1}\Big(\texttt{‹User›}\,q_j\,\texttt{‹Assistant›}\,\texttt{</think>}\,a_j\,\texttt{‹eos›}\Big)\\ &\Vert\ \texttt{‹User›}\,q_k\,\texttt{‹Assistant›}\,m_k\end{aligned}$$
$q_j,a_j$：第 $j$ 轮的问题与回答；$c_j$（第 $j$ 轮的思考）不出现在任何历史轮次里；$m_k$ 是这一轮的模式 token。
:::

这条规则有两个效果。一是省上下文：思考往往比回答长得多，不删的话，几轮之后上下文就被旧思考占满。二是让历史轮次的格式统一：不管当时是哪种模式，历史里都长得像一次非思考的回答。为什么要保留 `</think>`？模型卡没有解释。笔者的推测是：这样每一个历史回答前面都是“`‹Assistant›</think>`”，与非思考模式的首轮格式完全一致，模型在训练中见过的就是这种格式。

::: ex 删不删思考，历史差多少（假设的数字）
设每轮的问题约 200 token、思考约 6,000 token、回答约 800 token（为了算账而假设，不是 V3.1 的统计）。第 $k$ 轮提问时，上下文里有 $k-1$ 轮历史：

- 按 (15.3) 删掉思考：历史约 $(k-1)(200+800)=1{,}000\,(k-1)$ token。
- 如果保留思考：历史约 $(k-1)(200+6{,}000+800)=7{,}000\,(k-1)$ token。

128K 的窗口是 131,072 token。保留思考时，第 19 轮的历史就有 $7{,}000\times18=126{,}000$ token，窗口几乎被占满；删掉思考，同样 18 轮的历史只有 18,000 token，不到窗口的 14%。不删会坏在哪里也就清楚了：长对话要么被截断、丢掉最早的轮次，要么把大部分窗口和注意力算力花在已经用完的旧思考上。
:::

::: err “V3.1 已经能在思考的过程中调用工具”
不对。V3.1 的模型卡写的是“Toolcall is supported in non-thinking mode.”，只为搜索智能体另设了一种思考模式下的专用格式（“We design a specific format for searching toolcall in thinking mode, to support search agent.”）[@hf-v31]。DeepSeek 在 V3.2 的发布说明里称 V3.2 是“our first model to integrate thinking directly into tool-use” [@ds-news]。把思考和工具调用接起来，要改的正是 (15.3) 这条“每轮都删思考”的规则，§7 会讲 V3.2 怎样改。
:::

V3.1 的工具调用格式也写在模型卡里：`<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>工具名<｜tool▁sep｜>参数<｜tool▁call▁end｜>…<｜tool▁calls▁end｜>` [@hf-v31]。在 API 上，两种模式对应两个名字：`deepseek-chat` 是非思考模式，`deepseek-reasoner` 是思考模式，上下文都是 128K [@ds-news]。

## 思考模式的效率

模型卡对思考模式的说法是：

“DeepSeek-V3.1-Think achieves comparable answer quality to DeepSeek-R1-0528, while responding more quickly.” [@hf-v31]

这是厂商的自述。模型卡给了两者的准确率对比，但没有给“更快”的量化数据（平均 token 数或时延），所以本课无法复核“更快”快多少。能复核的是准确率：例如 AIME 2025，V3.1 思考模式 88.4（V3.1-Terminus 同为 88.4），V3.2 是 93.1 [@hf-v31; @dsv32exp; @dsv32]。

## 长上下文续训：839B token

V3.1 的另一处变化在基座。它不是重新预训练的，而是在 V3 的基座检查点上继续训练，专门扩长上下文：

“The 32K extension phase has been increased 10-fold to 630B tokens, while the 128K extension phase has been extended by 3.3x to 209B tokens.” [@hf-v31]

“10 倍”“3.3 倍”是相对谁？是相对 V3 报告里的两阶段扩展：先把上下文从 4K 扩到 32K，再扩到 128K，“each comprising 1000 steps”；第一阶段序列长 32K、batch 1920，第二阶段序列长 128K、batch 480 [@dsv3]。（YaRN 的机制见 [第 10 课 §6](lesson-10-v2-moe-training.html#p6)。）

::: deriv V3 两阶段扩展各用了多少 token
$D=\text{步数}\times\text{batch}\times\text{序列长}$ || 训练 token 数（课程记号 $D$）：每步处理 batch 条、每条“序列长”个 token
$D_{32K}=1000\times1920\times32{,}768=62{,}914{,}560{,}000$ || 32K $=2^{15}=32{,}768$
$D_{128K}=1000\times480\times131{,}072=62{,}914{,}560{,}000$ || 128K $=2^{17}$；batch 缩为 1/4、长度放大 4 倍，乘积不变
$630\text{B}/62.91\text{B}=10.01,\qquad 209\text{B}/62.91\text{B}=3.32$ || 与“10-fold”“3.3x”一致
$630\text{B}+209\text{B}=839\text{B}$ || DeepSeek 新闻页写作“840B tokens continued pretraining” [@ds-news]，是四舍五入
:::

::: eq 续训 token 数 | (15.4)
$$\begin{gathered}D=\text{步数}\times B\times T\\[4pt] D^{\text{V3}}_{32K}=D^{\text{V3}}_{128K}=62.91\text{B},\qquad D^{\text{V3.1}}=630\text{B}+209\text{B}=839\text{B}\end{gathered}$$
$D$：训练 token 数；$B$：每步的序列条数；$T$：序列长度（token）。
:::

::: fig figures/l15-context-extension.svg | 图 15.3 V3 基座之后的续训 token 数：V3.1 的两阶段长上下文扩展与 V3.2 的 DSA 续训 | wide
**看蓝色条与灰色条的长度比：32K 阶段放大了 10 倍，128K 阶段放大了 3.3 倍。** 紫色是第 14 课讲过的 DSA 续训（只训 indexer 的 2.1B 预热，加 943.7B 的稀疏训练）。线性刻度，每根条的长度由 (15.4) 与模型卡、V3 报告、V3.2 论文的数字计算 [@hf-v31; @dsv3; @dsv32]。
:::

::: ex V3 基座之后一共续训了多少
V3.1 的 839B，加上 V3.2-Exp 的 DSA 续训 $2.1\text{B}+943.7\text{B}=945.8\text{B}$，V3 基座之后累计约 $839+945.8=1{,}784.8\text{B}\approx1.78\text{T}$ token。V3 预训练是 14.8T token [@dsv3]，续训累计约为它的 12%。这是笔者的汇总，论文没有给这个合计。
:::

::: err “V3.1 用 840B token 重新预训练了一遍”
不对。840B 是在 V3 基座上的**长上下文续训**：32K 阶段 630B 加 128K 阶段 209B。模型卡说得很清楚：V3.1-Base “is built upon the original V3 base checkpoint through a two-phase long context extension approach” [@hf-v31]。
:::

## 结构不变，只改了一处数值格式

对照 V3 与 V3.1 的 `config.json`，隐藏维 7168、61 层、MLA 的 $d_c=512$、$d_c'=1536$、256 个路由专家激活 8 个、1 层 MTP、YaRN 因子 40，全部相同；唯一多出来的是量化配置里的 `"scale_fmt": "ue8m0"` [@hf-v31]。这就是下一页的主题。模型卡还有一条使用建议与第 11 课直接相关：“The `mlp.gate.e_score_correction_bias` parameters should be loaded and computed in FP32 precision.” [@hf-v31] 这个参数就是无辅助损失均衡里的专家偏置 $b_i$（[第 11 课 §4](lesson-11-v3-architecture.html#p4)）。模型卡没有说为什么；笔者的理解是：它只影响 top-$K$ 的选择，一点点舍入误差就可能让某个 token 换一个专家。

一个月后的 V3.1-Terminus 结构不变，模型卡列出的改进是语言一致性（“Reducing instances of mixed Chinese-English text and occasional abnormal characters”）与智能体能力（“Further optimizing the performance of the Code Agent and Search Agent”）[@hf-v31t]。

::: key 本页要点
- 混合思考 (15.2)：同一个 $\pi_\theta$，提示以 `<think>`（128798）结尾就先思考，以 `</think>`（128799）结尾就直接回答；两个 token 占用的是 V3 词表里预留的空位。
- 多轮 (15.3)：历史轮次删掉思考、保留 `</think>`。V3.1 的工具调用只在非思考模式下进行。
- V3.1-Base = V3 基座 + 630B（32K）+ 209B（128K）续训 = 839B，分别是 V3 原扩展的 10.01 倍与 3.32 倍。
- 结构与 V3 相同，只在量化配置里多了 `scale_fmt: ue8m0`。
:::

# UE8M0：只能取 2 的幂的缩放因子 | 8 位全是指数的格式、向上取整的规则、E4M3 的利用率与检查点实测 | 25

V3.1 的模型卡有一句加粗的话：

“DeepSeek-V3.1 is trained using the **UE8M0 FP8 scale data format on both model weights and activations** to ensure compatibility with microscaling data formats.” [@hf-v31]

要读懂它，得先回到第 12 课的 FP8 细粒度量化。

::: pre FP8 与缩放因子 | 读过第 12 课 §2–§3 的可以跳过
FP8 的 E4M3 格式有 1 位符号、4 位指数、3 位尾数，最大有限值 448，最小正规数 $2^{-6}$，正规数的相对舍入误差不超过 $2^{-4}=6.25\%$（[第 12 课 §2](lesson-12-v3-infrastructure.html#p2)）。它的范围太窄，直接存权重和激活会溢出或下溢，所以每一小组数配一个**缩放因子** $s$：存 $\hat x_i=\mathrm{E4M3}(x_i/s)$，用时乘回 $x_i\approx s\,\hat x_i$。V3 让激活每 $1\times128$ 个元素、权重每 $128\times128$ 个元素共用一个 $s$，取 $s=a/448$，$a$ 是这一组的最大绝对值（amax），这样最大的元素恰好落在 448（[第 12 课 §3](lesson-12-v3-infrastructure.html#p3)）。V3 的 $s$ 大多是任意的 FP32 数（只有少数几类激活例外，见本页“UE8M0 缩放 ≠ MXFP8”警告框的最后一段）。缩放因子与细粒度量化的一般讲法见 [M08 §6](../math/m08-floating-point.html#p6)，MX 与 UE8M0 见 [M08 §7](../math/m08-floating-point.html#p7)。
:::

## UE8M0 是什么

**含义。** UE8M0 是一种 8 位的数：U 表示无符号（unsigned），E8 表示 8 位指数，M0 表示 0 位尾数。它只能表示 2 的整数次幂。

**理解。** 普通浮点数 = 符号 × 有效数字 × 2 的幂。把符号位和尾数全拿掉，只剩“2 的几次幂”这一项，就是 UE8M0。用它当缩放因子，缩放就只剩“把二进制小数点往左或往右挪几位”这一种操作。回到第 12 课的类比：E4M3 是一把短尺子，缩放因子是给每一段数据配的放大镜；UE8M0 规定放大镜只能是 ×2、×4、×1/2 这样的整倍率，不能是 ×2.23。

::: ana 类比的延伸：只能用整倍率的放大镜
**映射。** 缩放因子 $s$ = 放大镜的倍率；FP32 缩放 = 倍率可以连续调，让这段数据里最大的数恰好顶到尺子末端 448；UE8M0 缩放 = 倍率只能取 2 的整数次幂，选“不让最大的数超出尺子”的最大那一档。

**收益。** 一眼看出代价的上限：倍率只能按 2 倍一档地调，最大的数最多离尺子末端差一档，也就是落在 224 到 448 之间；尺子的上半截有时空着。也看出好处：整倍率的换算只要挪小数点（改指数），不需要真正做一次乘法。

**失效的地方。** 真实的 E4M3 不是等刻度的尺子，而是每过一个 2 的幂刻度就变稀一倍的“对数尺”。所以“尺子上半截空着”并不等于“精度差一半”：同一个 binade 里的相对精度不变，真正损失的是尺子**最细的那一端**，下面会算出来。
:::

**数学。** 设 8 位的指数字段为 $E\in\{0,1,\dots,255\}$：

::: eq UE8M0 的值 | (15.5)
$$v_{\text{UE8M0}}(E)=2^{\,E-127},\qquad E=0,\dots,254;\qquad E=255\ \text{表示 NaN}$$
偏置 127 与 FP32 的指数偏置相同，所以 UE8M0 能表示的指数 $-127,\dots,127$ 覆盖了 FP32 正规数的全部指数 $-126,\dots,127$，还多出一个 $2^{-127}$（在 FP32 里它是次正规数）。没有 0、没有负数、没有无穷大。
:::

NVIDIA 的两份文档给出了这个定义。PTX 指令集文档说 ue8m0 是“an 8-bit unsigned floating-point format with 8 bits for exponent and 0 bits for mantissa. The ue8m0 encoding does not support infinity. NaN value is limited to 0xff.” [@ptx-ue8m0] CUDA Math API 对相应数据类型 `__nv_fp8_e8m0` 的说明给出了偏置：“Bias equals to 127, so numbers 0 through 254 represent 2^-127 through 2^127. Number 0xFF = 255 is reserved for NaN.” [@cuda-e8m0] MX 论文则说：“All concrete MX formats use E8M0 (an 8-bit exponent) as the format for the shared scale. The representable exponents of these formats is a superset of the representable exponents of FP32.” [@mx2023]

::: ex 读一个 UE8M0 字节
字节 `0111 1000`：$E=64+32+16+8=120$，值 $=2^{120-127}=2^{-7}=0.0078125$。

范围：$E=0$ 是 $2^{-127}\approx5.9\times10^{-39}$，$E=254$ 是 $2^{127}\approx1.7\times10^{38}$。从最小到最大跨 $254$ 个 2 倍，折合 $254\times\log_{10}2\approx76.5$ 个十进制数量级。缩放因子本来就是正数，所以不需要符号位；它只负责“量级”，有效数字交给 E4M3 的 3 位尾数去管，所以也不需要尾数。
:::

## 向上取整到 2 的幂

缩放因子只能是 2 的幂，那么对一组 amax 为 $a$ 的数，该取哪一档？DeepSeek 在 V3.2（以及 V3.1-Terminus）开源的推理内核里是这样写的 [@hf-v32]：

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

逐行读：`fast_log2_ceil` 把一个 FP32 数的 32 位原样当作整数，右移 23 位取出 8 位指数字段，减去偏置 127 得到 $\lfloor\log_2 x\rfloor$（这一步要求 $x$ 是正的 FP32 正规数。调用它之前，同一文件的量化内核先把 amax 下限钳到 $10^{-4}$：`amax_local[i] = T.max(amax_local[i], 1e-4)`，所以全 0 的分块也按 $a=10^{-4}$ 处理，得到 $s=2^{-22}$，输入 $a/448\ge2.2\times10^{-7}$ 永远落在 FP32 正规数范围里）；只要尾数不全为 0（$x$ 不是 2 的整幂），就再加 1，于是得到 $\lceil\log_2 x\rceil$。`fast_pow2` 反过来，把整数 $n$ 加上 127 塞进指数字段、尾数置 0，就得到 FP32 的 $2^n$（要求 $-126\le n\le127$，结果是 FP32 正规数；$n=-127$ 时指数字段为 0，得到的是 0 而不是 $2^{-127}$）。`fast_round_scale` 把两者连起来，输入是 $a\times\frac1{448}$（代码里 `fp8_max = 448.0`，激活按 `group_size = 128` 分组）。整个过程只有移位、加减和比较，没有一次浮点除法或对数运算。

::: eq 两种缩放因子 | (15.6)
$$s_{\text{FP32}}=\frac{a}{448},\qquad s_{\text{UE8M0}}=2^{\lceil\log_2(a/448)\rceil},\qquad \hat x_j=\mathrm{E4M3}\!\left(\frac{x_j}{s}\right)$$
$a=\max_j\abs{x_j}$：一个 $1\times128$ 激活分块（或 $128\times128$ 权重块）的最大绝对值；$\lceil\cdot\rceil$：向上取整。
:::

为什么向**上**取整？因为取整的方向决定了会不会溢出。

::: deriv 向上取整保证不溢出，代价是最大码值落在 (224, 448]
$\lceil y\rceil\ge y\ \Rightarrow\ s_{\text{UE8M0}}=2^{\lceil\log_2(a/448)\rceil}\ge2^{\log_2(a/448)}=\frac{a}{448}$ || 向上取整不会变小；$2^{(\cdot)}$ 单调递增
$\frac{a}{s_{\text{UE8M0}}}\le448$ || 上式两边取倒数再乘 $a$：块里最大的数缩放后不超过 E4M3 的最大值
$\lceil y\rceil<y+1\ \Rightarrow\ s_{\text{UE8M0}}<2\cdot\frac{a}{448}$ || 向上取整至多多出 1（不到 1）
$\frac{a}{s_{\text{UE8M0}}}>224$ || 同样取倒数：最大码值严格大于 448 的一半
:::

如果改成向下取整，$s$ 会小于 $a/448$，最大的元素缩放后超过 448，在 FP8 里被饱和（削平）或变成 NaN。所以“向上”是一个被“不许溢出”逼出来的选择。

::: ex 手算三个分块
**$a=3.0$。** FP32 缩放：$s=3/448=0.006696$。UE8M0：$\log_2(3/448)=\log_2 0.006696=-7.22$，向上取整得 $-7$，$s=2^{-7}=0.0078125$。最大码值 $3.0/0.0078125=384$，是 448 的 85.7%。

**$a=224$。** $224/448=0.5=2^{-1}$，本身就是 2 的幂，$\lceil-1\rceil=-1$，$s=0.5$，最大码值恰好 448，与 FP32 缩放完全相同。

**$a=224.1$。** $\log_2(224.1/448)=\log_2 0.50022=-0.99936$，向上取整得 $0$，$s=1$。最大码值 $224.1$，只用到 448 的 50.0%。这是最坏的情形：$a$ 刚刚越过 $448\times2^n$，缩放因子就跳一整档。
:::

## 利用率：锯齿与平均 0.721

**含义。** 把 $u=\dfrac{a/s}{448}$ 叫作这个分块对 E4M3 范围的**利用率**：块里最大的码值占 448 的比例。FP32 缩放时 $u\equiv1$。

**理解。** 用尺子的类比说：$u$ 就是这一段数据里最大的那个数被“放大镜”换算以后，落在尺子长度的百分之几处。$u=1$ 是恰好顶到尺子末端 448；$u=0.5$ 是只到 224，尺子最上面 224 到 448 那一截这一块数据用不上。上面三个算例的 $u$ 依次是 0.857、1、0.500。

**数学。** 由 (15.6)，$u$ 只取决于 $\log_2(a/448)$ 离它上方最近的整数有多远：

::: eq 利用率 | (15.7)
$$\begin{gathered}u(a)=\frac{a/448}{2^{\lceil\log_2(a/448)\rceil}}=2^{-v},\qquad u\in\left(\tfrac12,1\right]\\[4pt] v=\lceil\log_2(a/448)\rceil-\log_2(a/448)\in[0,1)\end{gathered}$$
:::

$a$ 每翻一倍，$u$ 就沿同一条曲线从 $\tfrac12$ 附近爬到 1，再跳回 $\tfrac12$：一把锯齿。平均下来是多少？如果各分块的 $\log_2 a$ 在一个很宽的范围里散开，$v$ 可以近似看作 $[0,1)$ 上的均匀分布（这是一个假设，真实激活的分布未必如此）：

::: deriv 对数均匀假设下的平均利用率
$\E[u]=\int_0^1 2^{-v}\,dv$ || $v\sim\mathrm{Uniform}[0,1)$，$u=2^{-v}$
$=\int_0^1 e^{-v\ln2}\,dv=\left[-\frac{e^{-v\ln2}}{\ln2}\right]_0^1$ || $2^{-v}=e^{-v\ln 2}$，指数函数的原函数
$=\frac{1-\tfrac12}{\ln2}=\frac1{2\ln2}\approx0.7213$ || 代入上下限
:::

研究笔记用 20 万个对数均匀的随机 $a$ 做了蒙特卡洛，平均 0.7210，最小 0.5000，与上式一致。

::: fig figures/l15-ue8m0-sawtooth.svg | 图 15.4 UE8M0 缩放因子：只能取 2 的整数次幂，代价是 E4M3 的最上面一段有时用不上 | wide
**看橙色锯齿：每当分块最大值 $a$ 越过 $448\times2^n$，利用率就从 1 跳回 0.5，平均约 0.721（灰色虚线）；绿色直线是 FP32 缩放，恒为 1。** 上方是一个 UE8M0 字节的例子（$E=120$ 表示 $2^{-7}$）。曲线由 (15.7) 逐点计算；四个标注点里，$a=3.0$、224、224.1 是上面算例里的分块，$a=100$ 是练习 5 的分块；横轴是 $a$ 的对数刻度。
:::

## 丢掉的是哪一段精度

利用率 0.72 听起来像“精度只剩七成”。并非如此。E4M3 是浮点格式：同一个 binade 里有 8 个格点，每过一个 2 的幂间距就翻倍，所以任何一个正规数的相对舍入误差都不超过 6.25%，与它落在哪个 binade 无关。把整块数据乘以 $2^{-n}$ 只是把每个数的指数平移 $n$，在正规数范围内，每个数的相对误差上界不变。真正受影响的是块里**最小**的那些数：整块被多缩小了一倍，它们就更早地掉进次正规区（间距固定、相对误差变大），或者更早地被舍成 0。下面用第 12 课 (12.7) 之后的那个推导来量化（这一段是笔者的分析，V3.1 的资料没有讨论）。

::: deriv 一个元素比 amax 小多少倍就开始失真（UE8M0 版）
$\frac{\abs{x_j}}{s}=448\,u\cdot\frac{\abs{x_j}}{a}$ || 由 (15.7)，$a/s=448u$
$448u\cdot\frac{\abs{x_j}}{a}\ge2^{-6}\iff\abs{x_j}\ge\frac{a}{28{,}672\,u}$ || 仍是正规数（相对误差 $\le6.25\%$）的条件；$448\times2^6=28{,}672$
$448u\cdot\frac{\abs{x_j}}{a}\le2^{-10}\iff\abs{x_j}\le\frac{a}{458{,}752\,u}$ || 不超过最小次正规数 $2^{-9}$ 的一半，舍成 0（恰好一半时按“就近取偶”也舍成 0）；$448\times2^{10}=458{,}752$
$u\in(\tfrac12,1]\ \Rightarrow\ \frac{a}{28{,}672\,u}\in\left[\frac{a}{28{,}672},\frac{a}{14{,}336}\right)$ || 最坏时阈值升高一倍：动态范围至多少一个 binade
:::

::: ex 同一个分块，两种缩放
取最坏的 $a=224.1$（$u=0.50022$）。

- FP32 缩放（$u=1$）：比 $224.1/28{,}672=0.00782$ 大的元素都是正规数；比 $224.1/458{,}752=0.000489$ 小的元素变成 0。
- UE8M0 缩放：正规数的门槛升到 $224.1/(28{,}672\times0.50022)=0.015625$，变 0 的门槛升到 $0.000977$。

所以绝对值在 0.0078 到 0.0156 之间的元素，在 FP32 缩放下是 6.25% 以内的误差，在 UE8M0 缩放下掉进了次正规区。对 $a=3.0$（$u=6/7$）这样的分块，门槛只从 $a/28{,}672$ 变成 $a/24{,}576$，升高 $1/u=7/6$ 倍，约 17%。E4M3 本身有约 18 个 binade 的范围（第 12 课），UE8M0 最坏拿走其中 1 个。
:::

## 省下了什么，为什么说“兼容 microscaling”

代价是至多一个 binade，换来的是什么？

**存储。** FP32 缩放因子每个 4 字节；UE8M0 每个 1 字节。

::: eq 缩放因子的存储开销（每元素） | (15.8)
$$\begin{aligned}\text{FP32，每 128 个元素一个}&:\ \tfrac{4}{128}=0.03125\ \text{B/元素}\\ \text{UE8M0，每 128 个元素一个}&:\ \tfrac{1}{128}\approx0.0078\ \text{B/元素}\\ \text{MX（每 32 个元素一个 E8M0）}&:\ \tfrac{1}{32}=0.03125\ \text{B/元素}\end{aligned}$$
激活按每元素 1 字节的 E4M3 计，FP32 缩放的额外开销是 3.125%（第 12 课 (12.11)），UE8M0 是 0.78%。
:::

**运算。** 乘以 $2^n$ 在二进制浮点里是精确的（不溢出、不下溢时）：只把指数加上 $n$，尾数不变，没有舍入误差，也不需要一次真正的浮点乘法。

**硬件格式。** 这才是模型卡说的“compatibility with microscaling data formats”。MX（microscaling）是 2023 年提出的一族块格式：每 $k=32$ 个元素共享一个 E8M0 缩放 [@mx2023]。DeepSeek 的 DeepGEMM 在 README 里写道：“SM100 requires scaling factors in packed UE8M0 format, which packs 4 UE8M0 into a single `torch.int`.” [@deepgemm] 也就是说，在 NVIDIA 的 SM100 这一代 GPU 上，缩放因子要以 4 个 UE8M0 字节拼成一个 32 位整数的形式交给内核。笔者的解读是：如果模型训练时用的是任意 FP32 缩放，部署到这种硬件上就得把缩放因子再取整一次，数值和训练时不一样；训练时就只用 2 的幂，部署时就没有这层误差。

::: warn UE8M0 缩放 ≠ MXFP8
两者有两处不同。

**块大小。** MX 的具体格式是每 32 个元素一个缩放 [@mx2023]；DeepSeek 沿用 V3 的 $1\times128$ 激活分块与 $128\times128$ 权重块（`group_size = 128`，`weight_block_size [128,128]`）[@hf-v32]。

**取整方向。** MX 论文给出的转换算法取 $2^{\lfloor\log_2\max_i\abs{V_i}\rfloor-emax_{elem}}$，是对 amax 的指数**向下**取整、再减去元素格式的最大指数 [@mx2023]（E4M3 的 $emax_{elem}=8$，因为 $448=1.75\times2^8$）；DeepSeek 是对 $a/448$ **向上**取整。两者都得到 2 的幂，但同一块数据可能差一档。把 amax 写成 $a=m\cdot2^k$（$1\le m<2$）就能看清差在哪里：MX 恒取 $2^{k-8}$，最大码值 $256m\in[256,512)$；DeepSeek 在 $m\le1.75$ 时同样取 $2^{k-8}$，在 $m>1.75$ 时取 $2^{k-7}$。也就是说，恰好在 MX 的最大码值 $256m$ 超过 448、只能饱和到 448 的那些块上，DeepSeek 把缩放因子取大一档（码值整体减半），从而不溢出（这一段是笔者由两个公式推出的）。

**一个数（笔者的算例）。** 取 $a=3.96=1.98\times2^1$，$m=1.98>1.75$。MX 取 $2^{1-8}=2^{-7}$，最大元素的码值 $3.96\times128=506.9>448$，被饱和成 448，解码回 $448\times2^{-7}=3.5$，误差 11.6%。DeepSeek 取 $2^{\lceil\log_2(3.96/448)\rceil}=2^{\lceil-6.82\rceil}=2^{-6}$，码值 $3.96\times64=253.4$，E4M3 在 $[128,256)$ 里的间距是 16，舍入到 256，解码回 $256\times2^{-6}=4.0$，误差 1.0%。代价是这一块的利用率只有 $253.4/448=0.566$。而 $a=3.0$（$m=1.5\le1.75$）时两者都取 $2^{-7}$，结果完全相同。

另外，V3 报告里已经有几处缩放因子只取 2 的整数次幂：注意力之后那个线性层的输入（“all the scaling factors are round scaled, i.e., integral power of 2”）、MoE 上投影之前要分发的激活（“scaling factors for this activation are integral power of 2”），以及按同样办法处理的 MoE 下投影之前的激活梯度；报告也说过它的细粒度量化“highly consistent with the idea of microscaling formats” [@dsv3]。V3.1 做的是把这个做法推广到全部权重与激活。
:::

## 检查点里的实际情况

“权重和激活都用 UE8M0”是一句可以检验的话：发布的 FP8 检查点里，每个 $128\times128$ 权重块都存有一个缩放因子（张量名 `weight_scale_inv`），看它们是不是 2 的整数次幂即可。研究笔记用 HTTP Range 请求只读取 safetensors 文件头和这些缩放张量，逐个检查（判据：`numpy.frexp` 分解出的尾数是否都恰好是 0.5）[@ckpt-scan]。

::: fig figures/l15-scale-conformance.svg | 图 15.5 各版本检查点里 FP8 权重的缩放因子是否全是 2 的幂
**看 `o_proj` 那一行：从 V3.1 到 V3.2 始终是红色的 0.000，而其他模块从 V3.1 起全部变成绿色的 1.000。** 每格是该模块缩放因子中 2 的整数次幂所占的比例，同一格里列出的各层结果完全相同；“无此模块”指 V3.2 之前没有 DSA 的 indexer。数据来自对 Hugging Face 检查点的抽查（四个版本、图中列出的层逐一读取）[@ckpt-scan]，没有覆盖其余的层。
:::

图 15.5 给出三个事实。第一，抽查到的 V3 缩放因子没有一个是 2 的幂；V3.1 起（包括 V3.1-Terminus）除 `self_attn.o_proj` 外都是，与模型卡相符。第二，`o_proj` 是例外，而且这个例外 DeepSeek 自己承认过：V3.1-Terminus 的模型卡写道“In the current model checkpoint, the parameters of `self_attn.o_proj` do not conform to the UE8M0 FP8 scale data format. This is a known issue and will be corrected in future model releases.” [@hf-v31t] 但抽查的 V3.2 检查点（第 0、30、60 层）里它仍然没有改。原因没有公开。第三，所有版本的缩放因子张量都仍以 FP32 存储（形状如 `q_a_proj.weight_scale_inv` 为 $[12,56]$，即 $1536/128\times7168/128$）；“UE8M0”约束的是它们的**取值**，打包成 1 字节是在 SM100 内核里做的。

::: warn 两处容易踩的坑
Terminus 的 `config.json` 里没有 `scale_fmt` 字段，而同一仓库的 `inference/config_671B_v3.1.json` 里有 `"scale_fmt": "ue8m0"` [@hf-v31t]。依赖 `config.json` 的推理框架可能按普通 FP32 缩放处理激活（笔者推测）。另外，网上流传“UE8M0 是为下一代国产芯片设计的”，其出处是 2025-08-21 的媒体转载 [@qq-ue8m0]；研究笔记没能找到 DeepSeek 的一手原文，本课不据此下结论。
:::

::: key 本页要点
- UE8M0 (15.5)：8 位全是指数，值 $2^{E-127}$，255 是 NaN；只能表示 2 的整数次幂。
- DeepSeek 的取法 (15.6)：$s=2^{\lceil\log_2(a/448)\rceil}$，向上取整保证不溢出，最大码值落在 $(224,448]$；利用率 (15.7) $u\in(\tfrac12,1]$，对数均匀假设下平均 $1/(2\ln2)\approx0.721$。
- 代价在块内最小的数：正规数门槛从 $a/28{,}672$ 升到 $a/(28{,}672u)$，最坏升高一倍（少一个 binade）。好处：缩放因子可存成 1 字节，乘 $2^n$ 精确且便宜，与 MX 类硬件格式一致。
- 与 MXFP8 的差别：块大小 128 对 32，取整方向向上对向下。检查点里 `o_proj` 从 V3.1 到 V3.2 都不是 2 的幂；缩放张量仍以 FP32 存储。
:::

# V3.2 的后训练：专家蒸馏与一次混合 RL | 同一基座出发的八个专家、蒸馏成一个模型、一个 RL 阶段，以及 V3.2 版的 GRPO 目标 | 25

R1 的后训练分四个阶段：冷启动 SFT、面向推理的 RL、拒绝采样加第二次 SFT、全场景 RL（[第 13 课 §5](lesson-13-r1-reasoning.html#p5)）。阶段一多，就有一个老问题：后面的阶段会冲掉前面阶段学到的东西，这叫**灾难性遗忘**（catastrophic forgetting）。V3.2 换了一种组织方式：先分头练出一批专家，把它们的本事蒸馏进一个模型，再用**一个** RL 阶段把所有领域一起练。这套流程不是 V3.2 才有的：V3.2-Exp 的报告说，为了公平地评估 DSA，它“maintain the same post-training pipeline, algorithm, and data as used for DeepSeek-V3.1-Terminus” [@dsv32exp]。所以至迟在 V3.1-Terminus，这套流程就已经在用了（这是笔者由这句话做的推论）。

::: fig figures/l15-posttrain-pipeline.svg | 图 15.6 DeepSeek-V3.2 的后训练流水线 | wide
**从左往右读：同一个基座分出 8 个专家，各自做大规模 RL；专家生成的数据蒸馏回一个检查点；最后只做一次混合 RL。** 蓝色是被训练的模型，黄色是生成出来的数据，绿色是奖励。下方虚线是 Speciale 的分支：论文只说它“只用推理数据、RL 时长度惩罚更小、并入 DeepSeekMath-V2 的数据与奖励方法”，没有说它从哪个检查点出发，图中画成从蒸馏后的检查点另做一次 RL，是笔者的简化。据 V3.2 论文第 3 节绘制 [@dsv32]。
:::

## 专家蒸馏

**含义。** 专家蒸馏（specialist distillation）：先为每个领域单独训练一个“专家”模型，再让这些专家生成训练数据，用这些数据对一个模型做监督微调。这里的“蒸馏”是**数据层面**的：学生学的是老师写出来的文字，不是老师的完整概率分布。

**理解。** 可以想象一所学校先让各科的尖子生各自刷大量的题（每个专家做大规模 RL），再把他们写的解答汇编成一本厚厚的题解，最后让一个通才学生把整本题解学一遍（SFT）。好处是每门课的难点都由最擅长的人去攻，互不干扰；通才只需要模仿。这幅图景在两处失效：第一，这里的“尖子生”都是从**同一个基座**复制出来、各自练过的，不是天生不同的人；第二，模仿题解学到的是“怎么写”，学生很难超过写题解的人，所以论文也只说学生“只比专家略差”，剩下的差距要靠后面的 RL 补上。

论文的描述是 [@dsv32]：

“For each task, we initially develop a specialized model dedicated exclusively to that particular domain, with all specialist models being fine-tuned from the same pre-trained DeepSeek-V3.2 base checkpoint. In addition to writing tasks and general question-answering, our framework encompasses six specialized domains: mathematics, programming, general logical reasoning, general agentic tasks, agentic coding, and agentic search, with all the domains supporting both thinking and non-thinking modes. Each specialist is trained with large-scale Reinforcement Learning (RL) computing.”

也就是 6 个专门领域（数学、编程、通用逻辑推理、通用智能体、智能体编程、智能体搜索）加上写作与通用问答，共 8 个专家，每个都支持思考与非思考两种模式。版本之间有一处出入：V3.2-Exp 的报告列的是“five specialized domains: mathematics, competitive programming, general logical reasoning, agentic coding, and agentic search”，没有“通用智能体” [@dsv32exp]。多出来的这一个，正对应 §7 要讲的合成通用智能体任务。

蒸馏数据怎么生成、效果如何，论文说了两句 [@dsv32]：

“Furthermore, we employ different models to generate training data for long chain-of-thought reasoning (thinking mode) and direct response generation (non-thinking mode).”

“Experimental results demonstrate that models trained on the distilled data achieve performance levels only marginally below those of domain-specific specialists, with the performance gap being effectively eliminated through subsequent RL training.”

**数学。** 论文只用文字描述了蒸馏，下面是笔者按这些文字写出的形式化。记第 $k$ 个专家为 $\pi^{\text{spec}}_k$（$k=1,\dots,8$），它负责的提示集合为 $\mathcal X_k$：

::: eq 专家蒸馏（笔者的形式化） | (15.9)
$$\begin{gathered}\mathcal D=\bigcup_{k=1}^{8}\big\{(x,o)\ :\ x\in\mathcal X_k,\ o\sim\pi^{\text{spec}}_k(\cdot\mid x)\big\}\\[4pt] \mathcal L_{\text{SFT}}(\theta)=-\E_{(x,o)\sim\mathcal D}\sum_{t=1}^{\abs{o}}\log\pi_\theta(o_t\mid x,o_{<t})\end{gathered}$$
$o\sim\pi^{\text{spec}}_k$：专家生成的回答；实际上思考模式与非思考模式的数据由不同的模型生成，这里合写成一个 $\pi^{\text{spec}}_k$。$\mathcal L_{\text{SFT}}$ 就是普通的下一词交叉熵，只是训练数据换成了专家写的。
:::

## 一次混合 RL

蒸馏之后，所有领域在同一个 RL 阶段里一起训练 [@dsv32]：

“For DeepSeek-V3.2, we still adopt Group Relative Policy Optimization (GRPO) (Shao et al., 2024; DeepSeek-AI, 2025) as the RL training algorithm. As DeepSeek-V3.2-Exp, we merge reasoning, agent, and human alignment training into one RL stage. This approach effectively balances performance across diverse domains while circumventing the catastrophic forgetting issues commonly associated with multi-stage training paradigms. For reasoning and agent tasks, we employ rule-based outcome reward, length penalty, and language consistency reward. For general tasks, we employ a generative reward model where each prompt has its own rubrics for evaluation.”

V3.2-Exp 的报告把与过去的区别说得更直接：“Unlike in previous DeepSeek models, which are trained with multi-stage reinforcement learning, we merge reasoning, agent, and human alignment training into one RL stage.”，并说奖励设计要平衡两对矛盾：“(1) length versus accuracy and (2) language consistency versus accuracy.” [@dsv32exp]

奖励分两类。**推理与智能体任务**用程序判定的结果奖励（答案对不对、测试过没过，见 [第 13 课 §2](lesson-13-r1-reasoning.html#p2)），加上长度惩罚（防止越写越长）和语言一致性奖励（思维链用题目的语言写，见 [第 13 课 §5](lesson-13-r1-reasoning.html#p5)）。**通用任务**（写作、开放问答）没有标准答案，用一个“生成式奖励模型”：它读回答，按这道题自己的评分细则（rubric）写出评价并给分。长度惩罚的函数形式、奖励模型的规模和细则的生成方式，论文都没有披露。另外，V3.2 的后训练全程使用稀疏注意力：“The post-training of DeepSeek-V3.2 also employs sparse attention in the same way as the sparse continued pre-training stage.” [@dsv32]

最后，同一套流程产出了两个模型 [@dsv32]：

“DeepSeek-V3.2 integrates reasoning, agent, and human alignment data distilled from specialists, undergoing thousands of steps of continued RL training to reach the final checkpoints. To investigate the potential of extended thinking, we also developed an experimental variant, DeepSeek-V3.2-Speciale. This model was trained exclusively on reasoning data with a reduced length penalty during RL. Additionally, we incorporated the dataset and reward method from DeepSeekMath-V2 (Shao et al., 2025) to enhance capabilities in mathematical proofs.”

## V3.2 版的 GRPO 目标

GRPO 的公式第 8 课已经推导过（[第 8 课 §4](lesson-08-deepseekmath-grpo.html#p4)）。V3.2 论文重新写了一遍，式 (5)(6)，改写成本课程的记号如下 [@dsv32]：

::: eq V3.2 的 GRPO 目标（论文式 5，课程记号） | (15.10)
$$\begin{aligned}\mathcal J(\theta)=\E_{q,\ \{o_i\}_{i=1}^{G}\sim\pi_{\theta_{old}}}\Bigg[\frac1G\sum_{i=1}^{G}\frac1{\abs{o_i}}\sum_{t=1}^{\abs{o_i}}\Big(&\min\big(\rho_{i,t}\hat A_{i,t},\ \clip(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big)\\ &-\beta\,\widehat{\KL}_{i,t}\big(\pi_\theta\,\Vert\,\pi_{ref}\big)\Big)\Bigg]\end{aligned}$$
$q$：提示；$o_i$：旧策略对 $q$ 采的第 $i$ 个回答，长 $\abs{o_i}$ 个 token；$\varepsilon$：裁剪系数；$\beta$：KL 系数；$\widehat{\KL}_{i,t}$：第 $t$ 个 token 上的 KL 估计，§5 专讲。
:::

::: eq 重要性比与优势（论文式 6 及其后一句） | (15.11)
$$\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})},\qquad \hat A_{i,t}=r_i-\frac1G\sum_{j=1}^{G}r_j$$
$r_i$：第 $i$ 个回答的结果奖励（论文记作 $R_i$）。同一个回答的所有 token 共享一个优势。
:::

骨架与第 8 课相同：组采样、组内比较、裁剪、KL、先在每个回答内部按长度平均（$\frac1{\abs{o_i}}$）。看得见的差别在优势：V3.2 印出来的是“the advantage of $o_{i,t}$ is calculated by subtracting the average reward of the group from the reward of output $o_i$” [@dsv32]，**只减组均值**；DeepSeekMath 的原版还要再除以组内标准差：“normalized by subtracting the group average and dividing by the group standard deviation” [@dsmath]。

只减均值为什么还能当优势？因为组均值可以拆成“自己”和“别人”两部分：

::: deriv 减组均值 = 留一基线 × 常数
$r_i-\frac1G\sum_{j=1}^{G}r_j=r_i-\frac1G r_i-\frac1G\sum_{j\ne i}r_j$ || 把 $j=i$ 那一项单独拿出来
$=\frac{G-1}{G}\,r_i-\frac{G-1}{G}\cdot\frac1{G-1}\sum_{j\ne i}r_j$ || 合并 $r_i$ 的系数；给第二项乘除 $G-1$
$=\frac{G-1}{G}\Big(r_i-\underbrace{\tfrac1{G-1}\textstyle\sum_{j\ne i}r_j}_{b_{-i}}\Big)$ || $b_{-i}$：其余 $G-1$ 个回答的平均奖励，与 $o_i$ 无关
:::

$b_{-i}$ 不依赖回答 $o_i$ 本身（给定 $q$ 时 $G$ 个回答是独立采样的，所以 $b_{-i}$ 与 $o_i$ 独立），是一个合法的基线：减去它不改变策略梯度的期望，只改变方差；基线接近这道题的期望奖励时，方差通常会下降（基线定理见 [M09 §3](../math/m09-rl-math.html#p3)，第 8 课 (8.15) 也推过同一件事）。前面的 $\frac{G-1}{G}$ 是常数，相当于把学习率乘上 $\frac{G-1}{G}$；严格说，(15.10) 里的 KL 项不随优势缩放，所以等价的说法是学习率乘 $\frac{G-1}{G}$、同时 $\beta$ 乘 $\frac{G}{G-1}$。除以标准差则是另一回事：它不改变同一道题内的相对大小，但改变**题与题之间**的权重。

::: ex 两种优势，同一组奖励
$G=4$。

**奖励 $[1,0,0,1]$**（两对两错）：均值 0.5。V3.2 口径：$[+0.5,-0.5,-0.5,+0.5]$。DeepSeekMath 口径：总体标准差 $\sqrt{\tfrac14(4\times0.25)}=0.5$，优势 $[+1,-1,-1,+1]$。

**奖励 $[1,0,0,0]$**（只对一个，难题）：均值 0.25。V3.2 口径：$[+0.75,-0.25,-0.25,-0.25]$。DeepSeekMath 口径：标准差 $\sqrt{\tfrac14(0.5625+3\times0.0625)}=0.433$，优势 $[+1.732,-0.577,-0.577,-0.577]$。

对难题里唯一答对的那个回答，DeepSeekMath 口径给的优势是 V3.2 口径的 $1.732/0.75=2.31$ 倍；两对两错时只是 2 倍。所以除以标准差会**额外放大**“只有少数回答答对”的题，第 8 课 §4 把这叫作题目难度偏差。**奖励全为 1**：两种口径的优势都是 0，这道题不提供学习信号；DeepSeekMath 口径还要特判分母为 0。
:::

::: warn 论文前后两句不完全一致
V3.2 在给出公式之前的那一句是“estimated by normalizing the outcome reward within each group” [@dsv32]，“normalizing”通常暗示除以标准差，而紧接着的公式只减均值。训练代码没有公开，无法核对到底是哪一种。本课按印出来的公式讲，并在这里注明。
:::

::: key 本页要点
- 流水线：同一个 V3.2 基座 → 8 个专家（6 个专门领域 + 写作 + 通用问答），各自做大规模 RL → 专家生成数据，蒸馏（SFT）进一个检查点 → 一个混合 RL 阶段（数千步）。V3.2-Exp 时专门领域是 5 个。
- 混合 RL 把推理、智能体、人类对齐合成一个阶段，以避免多阶段训练的灾难性遗忘；推理与智能体用规则结果奖励 + 长度惩罚 + 语言一致性，通用任务用带每题评分细则的生成式奖励模型。
- V3.2 版 GRPO (15.10)(15.11)：优势只减组均值（等于留一基线乘 $\frac{G-1}{G}$），不除标准差。奖励 $[1,0,0,0]$ 时，答对者的优势在 DeepSeekMath 口径下是 V3.2 口径的 2.31 倍：除以标准差会额外放大难题。
:::

# 稳定化（一）：无偏 KL 估计 | k3 乘以重要性比：值无偏、梯度指向正确的 KL、权重只按对数增长 | 25

GRPO 目标 (15.10) 的最后一项 $-\beta\,\widehat{\KL}_{i,t}$ 负责把策略拴在参考模型附近。DeepSeekMath 与 R1 用的是 Schulman 的 k3 估计器 [@dsmath; @schulman-kl]。V3.2 对它做了一处修改：

“Given $o_{i,t}$ is sampled from the old policy $\pi_{old}(\cdot\mid q,o_{i,<t})$, we correct the K3 estimator (Schulman, 2020) to obtain an unbiased KL estimate using the importance-sampling ratio between the current policy $\pi_\theta$ and the old policy $\pi_{old}$.” [@dsv32]

这一页讲清楚：原来的 k3 在什么意义上“有偏”，乘上重要性比之后修好了什么，以及这个修改对梯度意味着什么。

::: pre k3 估计器 | 读过第 8 课 §5 的可以跳过
在某个位置（给定前缀），记被采到的 token 为 $x$，**参考比** $w(x)=\pi_{ref}(x)/\pi_\theta(x)$。k3 是 $k_3=w-1-\log w$。它有两条性质：**非负**（$w-1$ 是 $\log w$ 在 $w=1$ 处的切线，凹函数在切线下方）；**若 $x\sim\pi_\theta$，则值无偏**：$\E_{\pi_\theta}[k_3]=\KL[\pi_\theta\Vert\pi_{ref}]$，因为 $\E_{\pi_\theta}[w]=\sum_x\pi_{ref}(x)=1$（这里的求和只遍及 $\pi_\theta(x)>0$ 的 token；softmax 策略给每个 token 正概率，所以就是全词表）。完整推导见 [第 8 课 §5](lesson-08-deepseekmath-grpo.html#p5)；KL 散度本身见 [M05 §4](../math/m05-information-theory.html#p4)，k1、k2、k3 三种估计器的期望、方差与梯度见 [M05 §6](../math/m05-information-theory.html#p6)。
:::

“值无偏”有两个前提：样本来自 $\pi_\theta$；说的是**值**，不是**梯度**。规模化的 RL 把这两个前提都打破了。

## 前提一被打破：样本来自旧策略

GRPO 的样本来自 $\pi_{\theta_{old}}$，而在 V3.2 里，$\pi_{\theta_{old}}$ 还是推理框架算出的概率（§6）。一大批样本分成多个小批依次更新，从第二个小批起 $\pi_\theta\ne\pi_{\theta_{old}}$；即使在第一个小批，训练与推理两套实现也不逐位相同。这时 k3 的期望是 $\sum_x\pi_{\theta_{old}}(x)\,k_3(x)$，不再等于 $\KL[\pi_\theta\Vert\pi_{ref}]$。

::: pre 重要性采样 | 学过 M09 §5 的可以跳过
**含义。** 想求“按分布 $p$ 取平均”的量，手里却只有从另一个分布 $q$ 采来的样本，就给每个样本乘上权重 $p(x)/q(x)$ 再平均。

**理解。** 像做民意调查：样本里老人占七成，全国只占两成，那么每位老人的回答按 $0.2/0.7$ 计，每位年轻人按 $0.8/0.3$ 计，加权平均就还原了全国的平均。

**数学。**

$$\E_{x\sim q}\Big[\frac{p(x)}{q(x)}\,f(x)\Big]=\sum_x q(x)\,\frac{p(x)}{q(x)}\,f(x)=\sum_x p(x)\,f(x)=\E_{x\sim p}\big[f(x)\big]$$

成立的条件：凡是 $p(x)f(x)\ne0$ 的 $x$，都要有 $q(x)>0$；$q$ 采不到的地方，再怎么加权也补不回来（§6 的 Keep Sampling Mask 处理的正是这一条）。代价是方差：权重 $p/q$ 很大的少数样本会主导平均。完整讲法见 [M09 §5](../math/m09-rl-math.html#p5)。
:::

修补的办法就是重要性采样，取 $p=\pi_\theta$、$q=\pi_{\theta_{old}}$：乘上 $\rho=\pi_\theta/\pi_{\theta_{old}}$，把“按 $\pi_{\theta_{old}}$ 取平均”换回“按 $\pi_\theta$ 取平均”。V3.2 的式 (7) 就是这样写的，改写成课程记号：

::: eq V3.2 的 KL 估计（论文式 7） | (15.12)
$$\widehat{\KL}_{i,t}=\rho_{i,t}\,k_3(w_{i,t})=\frac{\pi_\theta(o_{i,t}\mid\cdot)}{\pi_{\theta_{old}}(o_{i,t}\mid\cdot)}\left(\frac{\pi_{ref}(o_{i,t}\mid\cdot)}{\pi_\theta(o_{i,t}\mid\cdot)}-\log\frac{\pi_{ref}(o_{i,t}\mid\cdot)}{\pi_\theta(o_{i,t}\mid\cdot)}-1\right)$$
“$\mid\cdot$”是条件 $(q,o_{i,<t})$ 的简写。与第 8 课 (8.21) 的 k3 相比，只多了最前面的 $\rho_{i,t}$。
:::

::: deriv 乘上 ρ 之后，值无偏
$\E_{x\sim\pi_{\theta_{old}}}\big[\rho(x)\,k_3(x)\big]=\sum_x\pi_{\theta_{old}}(x)\,\frac{\pi_\theta(x)}{\pi_{\theta_{old}}(x)}\,k_3(x)$ || 期望的定义；要求凡 $\pi_\theta(x)>0$ 处都有 $\pi_{\theta_{old}}(x)>0$
$=\sum_x\pi_\theta(x)\big(w(x)-1-\log w(x)\big)$ || 约分：这一步就是“换测度”
$=\sum_x\pi_{ref}(x)-1-\sum_x\pi_\theta(x)\log\frac{\pi_{ref}(x)}{\pi_\theta(x)}$ || $\pi_\theta w=\pi_{ref}$；$\sum_x\pi_\theta=1$
$=1-1+\KL[\pi_\theta\Vert\pi_{ref}]=\KL[\pi_\theta\Vert\pi_{ref}]$ || $\sum_x\pi_{ref}=1$（softmax 的 $\pi_\theta$ 处处为正，求和遍及全词表）；KL 的定义
:::

::: ex 三个 token 的分布：旧样本上的 k3 偏了多少
词表只有 3 个 token。$\pi_\theta=(0.5,0.3,0.2)$，$\pi_{ref}=(0.4,0.4,0.2)$，采样用的 $\pi_{\theta_{old}}=(0.2,0.7,0.1)$（为了看清效果，故意取得离 $\pi_\theta$ 很远）。

- 真 KL：$0.5\ln\frac{0.5}{0.4}+0.3\ln\frac{0.3}{0.4}+0=0.1116-0.0863=0.02527$。
- 每个 token 的 k3：$w=(0.8,\ 1.333,\ 1)$，$k_3=(0.02314,\ 0.04565,\ 0)$。
- 按 $\pi_{\theta_{old}}$ 平均原始 k3：$0.2\times0.02314+0.7\times0.04565=0.03658$，**高估 45%**：旧策略更常采到第二个 token，而它的 k3 恰好较大。
- 乘上 $\rho=(2.5,\ 0.4286,\ 2)$ 再按 $\pi_{\theta_{old}}$ 平均：$0.2\times2.5\times0.02314+0.7\times0.4286\times0.04565=0.01157+0.01370=0.02527$，与真 KL 一致。
:::

## 前提二：值无偏 ≠ 梯度无偏

更要紧的是梯度。V3.2 对修改的效果是这样说的 [@dsv32]：

“As a direct result of this adjustment, the gradient of this KL estimator becomes unbiased, which eliminates systematic estimation errors, thereby facilitating stable convergence. This contrasts sharply with the original K3 estimator, particularly when the sampled tokens have substantially lower probabilities under the current policy than the reference policy, i.e., $\pi_\theta\ll\pi_{ref}$. In such cases, the gradient of the K3 estimator assigns disproportionately large, unbounded weights to maximize the likelihood of these tokens, resulting in noisy gradient updates that accumulate to degrade sample quality in subsequent iterations and lead to unstable training dynamics.”

论文没有展开推导。下面补上（第 8 课在 $\rho=1$ 处推过一次，这里对一般的 $\rho$ 推）。$\pi_{ref}$ 冻结，$\pi_{\theta_{old}}$ 是采样时记下的常数，只有 $\pi_\theta$ 依赖 $\theta$；记 $g=\nabla_\theta\log\pi_\theta(x)$。

::: deriv 两个估计量各自的梯度
$\nabla_\theta w=-w\,g,\qquad \nabla_\theta\rho=\rho\,g$ || $w\propto1/\pi_\theta$，$\rho\propto\pi_\theta$；对数导数技巧
$\nabla_\theta k_3=\nabla w-\nabla\log w=-w\,g+g=(1-w)\,g$ || $\nabla\log w=-g$
$\nabla_\theta(\rho\,k_3)=k_3\,\nabla\rho+\rho\,\nabla k_3=\rho\,g\,\big[(w-1-\log w)+(1-w)\big]$ || 乘积法则
$=\rho\,(-\log w)\,g=\rho\,\log\frac{\pi_\theta}{\pi_{ref}}\;g$ || $w-1$ 与 $1-w$ 抵消
:::

::: deriv 修正后的期望梯度，正是要约束的那个 KL 的梯度
$\E_{\pi_{\theta_{old}}}\Big[\rho\log\frac{\pi_\theta}{\pi_{ref}}\,g\Big]=\sum_x\pi_\theta(x)\log\frac{\pi_\theta(x)}{\pi_{ref}(x)}\,\nabla\log\pi_\theta(x)$ || 换测度，同上一个推导
$\nabla_\theta\KL[\pi_\theta\Vert\pi_{ref}]=\sum_x\nabla\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}+\sum_x\pi_\theta\,\nabla\log\pi_\theta$ || 对 $\sum_x\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}$ 用乘积法则
$=\sum_x\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}\,\nabla\log\pi_\theta+\nabla\textstyle\sum_x\pi_\theta$ || 第一项 $\nabla\pi_\theta=\pi_\theta\nabla\log\pi_\theta$；第二项同理
$=\sum_x\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}\,\nabla\log\pi_\theta$ || $\sum_x\pi_\theta=1$，梯度为 0。与第一行相同
:::

原始 k3 呢？在 $\pi_{\theta_{old}}=\pi_\theta$ 时，$\E[(1-w)g]=\nabla\KL[\pi_{ref}\Vert\pi_\theta]$，是**方向反过来**的那个 KL（第 8 课 (8.23)）；Tang 与 Munos 独立证明了同一件事：“Very importantly, the positions of the two distributions are reversed compared to the target KL divergence” [@tang2025kl]。在 $\pi_{\theta_{old}}\ne\pi_\theta$ 时，它连这个都不是。

::: eq 每个 token 的梯度权重 | (15.13)
$$\nabla_\theta k_3=\Big(1-\frac1{z}\Big)\,g,\qquad \nabla_\theta(\rho\,k_3)=\rho\,\log z\;g,\qquad z=\frac{\pi_\theta(o_{i,t}\mid\cdot)}{\pi_{ref}(o_{i,t}\mid\cdot)}=\frac1{w}$$
$z$：这个 token 上当前策略与参考策略的概率之比（上面的推导里 $x$ 表示 token，所以这里换一个字母）。“权重”指 $g$ 前面的系数。损失里的 KL 项是 $+\beta\widehat{\KL}$，学习率为 $\eta$ 的一步梯度下降是 $\Delta\theta=-\eta\beta\,(\text{权重})\,g$，一阶近似下这个 token 的对数概率变化 $g^\top\Delta\theta=-\eta\beta\,(\text{权重})\norm{g}^2$：权重为负，对数概率上升。$z<1$ 时两个权重都是负的，都把 $\pi_\theta$ 往上拉回参考模型；区别在大小。
:::

当 $\pi_\theta\ll\pi_{ref}$（$z\to0$），原始权重 $1-1/z$ 像 $-1/z$ 一样**无界**地增长；修正后的 $\rho\log z$ 在 $\rho$ 固定时（例如下面的 $\rho=1$）只按对数增长。这就是论文说的“disproportionately large, unbounded weights”。

::: ex 一个策略几乎不写、参考模型常写的 token
设 $\pi_{ref}=0.1$，$\pi_\theta=\pi_{\theta_{old}}=0.001$（所以 $\rho=1$，$z=0.01$）。

- k3 的值：$w=100$，$k_3=100-1-\ln100=94.39$。修正版的值 $\rho k_3$ 也是 94.39：$\rho=1$ 时两者的值相同。
- 原始权重：$1-1/0.01=-99$。修正权重：$1\times\ln0.01=-4.605$。原始 k3 对这个 token 的拉力是修正版的 $99/4.605=21.5$ 倍。
- 对照：第 8 课的 $\beta=0.04$ 下，原始 KL 项在这个 token 上贡献 $0.04\times99=3.96$，比“两对两错”时 V3.2 口径的优势 0.5 大近 8 倍；修正后是 $0.04\times4.605=0.18$。

更多的点（研究笔记复算）：$z=0.001$ 时 $-999$ 对 $-6.91$；$z=0.1$ 时 $-9$ 对 $-2.30$；$z=0.5$ 时 $-1$ 对 $-0.69$；$z=2$ 时 $+0.5$ 对 $+0.69$；$z=10$ 时 $+0.9$ 对 $+2.30$。注意 $z>1$ 一侧反过来了：原始权重被 1 封顶，修正权重反而更大，仍按对数增长（因为对所有 $z>0$ 都有 $\log z\ge1-1/z$）。不过这一侧有天然的上限：$\pi_\theta\le1$，所以 $z\le1/\pi_{ref}$，修正权重（$\rho=1$ 时）至多 $\log(1/\pi_{ref})$，例如 $\pi_{ref}=10^{-4}$ 时约 9.2。左侧则没有这样的上限：$\pi_\theta$ 可以任意小，$1-1/z$ 随之趋于 $-\infty$。
:::

::: fig figures/l15-kl-gradcheck.svg | 图 15.7 KL 项的梯度权重：原始 k3 与 V3.2 的重要性修正版 | wide
**左图看两条线在左端的差距：$\pi_\theta$ 比 $\pi_{ref}$ 小 1000 倍时，原始权重是 −999，修正后只有 −6.91。右图看柱子的配对：蓝色的实心与空心一样高，红色的实心与空心一样高。** 左图由 (15.13) 在 $\rho=1$ 处计算，纵轴是对称对数刻度。右图是一个 5 个 token 的 softmax（logits 与参考 logits 由 `numpy.random.default_rng(0)` 生成），在策略采样（$\pi_{\theta_{old}}=\pi_\theta$）：把修正估计量当损失求导后的期望梯度，逐分量等于 $\nabla\KL[\pi_\theta\Vert\pi_{ref}]$；把原始 k3 当损失求导，得到的是 $\nabla\KL[\pi_{ref}\Vert\pi_\theta]$。期望由对 5 个 token 精确求和算出，真梯度用有限差分核对。生成脚本还换了一个 $\pi_{\theta_{old}}\ne\pi_\theta$ 重做一遍（图中未画）：修正版的期望梯度仍等于 $\nabla\KL[\pi_\theta\Vert\pi_{ref}]$，原始 k3 则两个方向的 KL 梯度都对不上。
:::

::: warn 实现时的一个陷阱：值相同，梯度不同
每批数据的第一个小批，若训练与推理完全一致，$\rho=1$，$\rho k_3$ 与 $k_3$ 的**数值**完全相同。但它们的梯度不同，因为 $\nabla\rho=\rho\,g\ne0$。所以代码里 $\rho$ 的分母必须从计算图上断开，分子不能断开。下面是一个示意（笔者写的，不是 DeepSeek 的代码）：

```python
logp     = policy_logprobs             # log π_θ，参与求导
logp_old = sampler_logprobs.detach()   # 推理框架返回的 log π_old
logp_ref = ref_logprobs.detach()       # 参考模型
rho   = torch.exp(logp - logp_old)     # 数值可能恰好是 1，但带着梯度
log_w = logp_ref - logp
k3    = torch.exp(log_w) - log_w - 1
kl    = rho * k3                       # V3.2 式 (7)
```

如果误写成 `rho.detach() * k3`，数值不变，梯度却变成 $\rho\,(1-w)\,g$：只是原始 k3 的梯度乘上一个常数 $\rho$，$\rho=1$ 时就是原始 k3 的梯度，期望方向仍是反过来的那个 KL。
:::

## 不同领域，不同强度

论文还补了一句经验：“In practice, we find that different domains benefit from varying strengths of KL regularization. For certain domains, such as mathematics, applying a relatively weak KL penalty or even omitting it entirely can yield improved performance.” [@dsv32] 也就是说，$\beta$ 不是一个全局常数，数学这类有可靠结果奖励的领域可以把它调得很小甚至设为 0。各领域具体用了多大的 $\beta$，论文没有披露。

::: err “k3 是无偏的，所以直接把它当损失就行”
混淆了两件事。k3 的**值**只在样本来自 $\pi_\theta$ 时无偏；样本来自 $\pi_{\theta_{old}}$ 时要乘 $\rho$。更重要的是**梯度**：把 k3 当损失求导，在策略采样时期望梯度是 $\nabla\KL[\pi_{ref}\Vert\pi_\theta]$，方向反了，而且在 $\pi_\theta\ll\pi_{ref}$ 时权重无界。V3.2 的修正让期望梯度变成 $\nabla\KL[\pi_\theta\Vert\pi_{ref}]$（单个位置的情形；整条序列上的额外项见 Tang 与 Munos 的论文 [@tang2025kl]，本课不展开）。
:::

::: key 本页要点
- V3.2 的 KL 估计 (15.12) $=\rho\cdot k_3$：乘上 $\rho=\pi_\theta/\pi_{\theta_{old}}$ 后，样本来自 $\pi_{\theta_{old}}$ 时值仍无偏（换测度）。
- 梯度：$\nabla(\rho k_3)=\rho\log(\pi_\theta/\pi_{ref})\,\nabla\log\pi_\theta$，期望是 $\nabla\KL[\pi_\theta\Vert\pi_{ref}]$；原始 k3 的期望梯度是反方向的 $\nabla\KL[\pi_{ref}\Vert\pi_\theta]$。
- 权重 (15.13)，$z=\pi_\theta/\pi_{ref}$：原始 $1-1/z$ 在 $z\to0$ 时无界，修正 $\rho\log z$ 按对数增长；$\pi_{ref}=0.1$、$\pi_\theta=\pi_{\theta_{old}}=0.001$ 时相差 21.5 倍。
- $\rho=1$ 时值相同、梯度不同：分母 detach，分子不 detach。数学领域可以用很弱的 KL 甚至不用。
:::

# 稳定化（二）：序列掩码、固定路由与采样掩码 | 掩掉离策略太远的负样本（off-policy 序列掩码）、Keep Routing、Keep Sampling Mask，每一项配一个算例 | 30

上一页修的是 KL 这一项。这一页的三项改动修的都是另一件事：**采样时的策略和训练时的策略不是同一个**。先把这种不一致从哪来讲清楚，再逐项看 V3.2 怎样处理。

## 两个 off-policy 来源

**含义。** 用来训练的样本如果不是由当前正在更新的策略生成的，就叫**离策略**（off-policy）样本；两者的差距叫“离策略程度”（off-policyness）。

V3.2 论文指出了两个来源 [@dsv32]：

“To improve the efficiency of RL systems, we typically generate a large batch of rollout data, which is subsequently split into multiple mini-batches for several gradient update steps. This practice inherently introduces off-policy behavior. Additionally, inference frameworks used for efficient data generation are often highly optimized, which may differ in implementation details from training frameworks. Such training-inference inconsistency further exacerbates the degree of off-policyness.”

第一个来源是**多个小批**：一次 rollout 采一大批回答，分成几个小批依次更新，第 $j$ 个小批上场时参数已经被前面 $j-1$ 个小批改过了（第 13 课 §3 用 R1 的 16 个小批讲过这件事）。第二个来源是**两套框架**：采样用高度优化的推理框架（算子融合、不同的并行切分、不同的累加顺序），训练用另一套框架，同一组参数、同一个输入，算出的概率也不逐位相同。V3.2 特意让比率的分母包含这两种来源：

“Note that $\pi_{old}$ here denotes the sampling probability directly returned by the inference framework, thus the KL divergence between the old and current policy accounts for both sources of off-policyness mentioned above.” [@dsv32]

::: err “旧策略的概率，就是训练框架用上一步参数算出的概率”
在 V3.2 里不准确。$\pi_{\theta_{old}}$ 是推理框架在采样当时返回的概率，不是训练框架用旧参数重新算的概率。所以即使在第一个小批、参数还没动时，$\rho_{i,t}$ 也未必等于 1，它的偏离量正好量出了训练与推理的实现差异。
:::

## Off-policy 序列掩码

**含义。** 对每一条回答，量一量它“离当前策略有多远”；如果它是一条**负样本**（优势为负）而且离得太远，就把它在策略损失里的贡献整条去掉。

**理解。** 回到 §1 的类比：用学生昨天的错答卷来纠正今天的学生。如果那份答卷今天的学生根本不会再写出来（离策略太远），拿它来“惩罚”今天的学生，惩罚的是一个已经不存在的习惯，还可能误伤别的东西。正样本则不同：一份答对的答卷，不管离得多远，照着学总归是有益的。论文的原话是：“Intuitively, the model benefits the most by learning from its own mistakes, whereas highly off-policy negative samples can be detrimental, potentially misleading or destabilizing the optimization process.” [@dsv32]

这个比方在两处失效。第一，“今天的学生还会不会这样写”在真实训练里只能用下面的 $\bar D_i$ 去估计，而 $\bar D_i$ 只是一条回答上的平均，回答越短波动越大，可能误掩，也可能漏掩。第二，被掩掉的错答卷不再提供任何“别这样写”的信号：阈值 $\delta$ 取得太小，大部分负样本都会被掩掉，训练就只剩“向对的学”这一半（这一句是笔者的分析，论文没有讨论阈值过小的后果）。

**数学。** V3.2 的式 (8) 在式 (5) 的裁剪项上乘了一个 0/1 掩码，式 (9) 定义这个掩码。改写成课程记号（求和下标换成 $s$，以免与外层的 $t$ 混淆）：

::: eq 带掩码的 GRPO 目标（论文式 8） | (15.14)
$$\mathcal J(\theta)=\E\Bigg[\frac1G\sum_{i=1}^{G}\frac1{\abs{o_i}}\sum_{t=1}^{\abs{o_i}}\Big(\min\big(\rho_{i,t}\hat A_{i,t},\ \clip(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big)\,m_i-\beta\,\widehat{\KL}_{i,t}\Big)\Bigg]$$
掩码只乘在裁剪代理项上；KL 项在掩码之外，被掩掉的回答仍然受 KL 约束。
:::

::: eq 序列掩码（论文式 9） | (15.15)
$$m_i=\begin{cases}0, & \hat A_{i}<0\ \ \text{且}\ \ \bar D_i>\delta\\ 1, & \text{其他情况}\end{cases}\qquad \bar D_i=\frac1{\abs{o_i}}\sum_{s=1}^{\abs{o_i}}\log\frac{\pi_{\theta_{old}}(o_{i,s}\mid q,o_{i,<s})}{\pi_\theta(o_{i,s}\mid q,o_{i,<s})}$$
$\delta$：阈值，一个超参数（“δ is a hyper-parameter that controls the threshold of policy divergence”），论文没有给出取值 [@dsv32]。
:::

三点要看清。第一，条件是“**且**”：优势为负，同时离得太远。原文强调“we only mask sequences with negative advantages” [@dsv32]，正样本从不被掩。第二，$\hat A_{i,t}$ 对同一条回答的所有 token 相同，$\bar D_i$ 也与 $t$ 无关，所以掩码是**整条序列**一起开或关，名字里的“序列”就是这个意思（论文式 (9) 把 $t$ 同时用作外层下标和求和变量，是记号重载）。第三，$\bar D_i$ 为什么叫“KL 散度”？因为它是一个 KL 的单样本估计：

::: deriv D̄ 是“采样策略相对当前策略”的 KL 的单样本估计
$\E_{x\sim\pi_{\theta_{old}}}\Big[\log\frac{\pi_{\theta_{old}}(x)}{\pi_\theta(x)}\Big]=\KL[\pi_{\theta_{old}}\Vert\pi_\theta]$ || KL 的定义：期望在第一个分布下取
$\bar D_i=\frac1{\abs{o_i}}\sum_s\log\frac{\pi_{\theta_{old}}(o_{i,s})}{\pi_\theta(o_{i,s})}$ || 回答的 token 正是从 $\pi_{\theta_{old}}$ 采的，所以每一项都是上式被积函数的一个样本
$\bar D_i\ \text{可以为负}$ || 单个样本上，$\pi_\theta$ 可能比 $\pi_{\theta_{old}}$ 更喜欢这些 token；只有期望才非负
:::

## 为什么 PPO 的裁剪还不够

PPO 的裁剪本来就是为“新旧策略差太远”准备的，为什么还要掩码？把裁剪项按优势的符号拆开（第 13 课 §3 推导过）：

::: eq 裁剪项按优势符号拆开 | (15.16)
$$\min\big(\rho\hat A,\ \clip(\rho,1-\varepsilon,1+\varepsilon)\hat A\big)=\begin{cases}\hat A\cdot\min(\rho,\ 1+\varepsilon), & \hat A>0\\[2pt] \hat A\cdot\max(\rho,\ 1-\varepsilon), & \hat A<0\end{cases}$$
:::

对负样本，只有 $\rho<1-\varepsilon$ 的 token 被“裁掉”（取常数，梯度为 0）；$\rho>1-\varepsilon$ 的 token 照常有梯度（$\rho$ 恰好等于 $1-\varepsilon$ 是折点，左右导数不同，下面的算例都避开了它），其中 $\rho>1+\varepsilon$ 的 token 甚至没有上界保护。裁剪是**逐 token** 的判断，而一条离策略很远的回答里，往往还有许多 token 的比率落在裁剪区间内，它们照样推动梯度。序列掩码一次性去掉整条回答。（这一段是笔者的分析，论文只给出了经验结论：“We empirically observe that this Off-Policy Sequence Masking operation improves stability in certain training scenarios that would otherwise exhibit instability.” [@dsv32]）

## 算例：四条回答，两个阈值

::: ex 手算掩码（δ 取两个假设值）
同一道题 $G=4$ 条回答，奖励 $[1,0,0,1]$，按 (15.11) 优势是 $[+0.5,-0.5,-0.5,+0.5]$。每条回答 6 个 token，采样时的概率都是 $\pi_{\theta_{old}}=[0.9,0.6,0.8,0.5,0.7,0.95]$；经过几个小批之后，当前策略给这些 token 的概率是：

| 回答 | $\hat A$ | $\pi_\theta$（6 个 token） | $\log(\pi_{\theta_{old}}/\pi_\theta)$ 逐 token | $\bar D_i$ |
|---|---|---|---|---|
| o1 | +0.5 | 0.91, 0.62, 0.80, 0.52, 0.70, 0.95 | −0.0110, −0.0328, 0, −0.0392, 0, 0 | **−0.0138** |
| o2 | −0.5 | 0.80, 0.30, 0.70, 0.20, 0.60, 0.90 | 0.1178, 0.6931, 0.1335, 0.9163, 0.1542, 0.0541 | **0.3448** |
| o3 | −0.5 | 0.85, 0.50, 0.75, 0.42, 0.65, 0.93 | 0.0572, 0.1823, 0.0645, 0.1744, 0.0741, 0.0213 | **0.0956** |
| o4 | +0.5 | 与 o2 相同 | 与 o2 相同 | **0.3448** |

以 o3 为例：$\ln(0.9/0.85)=0.0572$，…，六项之和 0.5738，除以 6 得 0.0956。

- $\delta=0.05$：o2（$\hat A<0$，$0.3448>0.05$）与 o3（$0.0956>0.05$）被掩，$m=(1,0,0,1)$。
- $\delta=0.2$：只有 o2 被掩，$m=(1,0,1,1)$。
- o4 离得和 o2 一样远，但优势为正，两种阈值下都保留。

再看 o2 和 o3 如果不掩会怎样。取 $\varepsilon=0.2$，由 (15.16)，负样本的代理项是 $\hat A\cdot\max(\rho,\,0.8)$，逐 token 列出来：

| token | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| o2 的 $\rho=\pi_\theta/\pi_{\theta_{old}}$ | 0.889 | 0.500 | 0.875 | 0.400 | 0.857 | 0.947 |
| o2 的代理项 $-0.5\max(\rho,0.8)$ | −0.444 | **−0.400** | −0.4375 | **−0.400** | −0.429 | −0.474 |
| o2 这个 token 有梯度吗 | 有 | 无 | 有 | 无 | 有 | 有 |
| o3 的 $\rho$ | 0.944 | 0.833 | 0.9375 | 0.840 | 0.929 | 0.979 |
| o3 这个 token 有梯度吗 | 有 | 有 | 有 | 有 | 有 | 有 |

o2 里 $\rho=0.5$ 与 $0.4$ 的两个 token 取到常数 $-0.4$，没有梯度；**其余 4 个 token 仍有梯度**，掩码把这 4 个也一并去掉。o3 的六个比率都高于 0.8，六个 token 全有梯度：$\delta$ 取 0.05 还是 0.2，决定的就是这 6 个梯度留不留。
:::

::: fig figures/l15-offpolicy-mask.svg | 图 15.8 off-policy 序列掩码：只掩掉离策略太远的负优势序列 | wide
**左图看红色区域：只有“优势为负”且“$\bar D_i$ 超过 $\delta$”的右下角被掩；o4 与 o2 横坐标相同，却因为在上半平面而保留。右图看红色柱：o2 里比率高于 0.8 的 4 个 token，裁剪拦不住它们的梯度，只有掩码能拦。** 数据就是上面的算例，由 (15.15)(15.16) 计算；$\delta$ 的两个取值是假设，论文没有披露。
:::

## Keep Routing：采样时选了哪些专家，训练时就用哪些

**问题。** V3.2 是 MoE 模型：每层 256 个路由专家，每个 token 只激活其中 8 个（[第 11 课 §2](lesson-11-v3-architecture.html#p2)）。选谁由亲和度的 top-$K$ 决定，而 top-$K$ 是一个**不连续**的操作：第 8 名和第 9 名的分数只要差一点点，数值上一点微小的差异就能让它们换位。论文的描述是 [@dsv32]：

“However, discrepancies between inference and training frameworks, compounded by policy updates, can result in inconsistent expert routing during inference and training even for identical inputs. Such inconsistency induces abrupt shifts in the active parameter subspace, which destabilizes optimization and exacerbates off-policy issues.”

“active parameter subspace”（被激活的参数子空间）说的是：换了一个专家，就是换了一整块参数。采样时 token 经过的是专家 11，训练时算概率、求梯度却经过专家 5：比率 $\rho$ 里分子和分母根本不是同一个函数算出来的，梯度也落到了一个采样时没用过的专家身上。

::: ana 类比：照着挂号单复诊
**映射。** 沿用 [第 7 课](lesson-07-deepseekmoe.html) 的医院类比：路由器 = 分诊台，路由专家 = 专科医生，token = 病人。采样时，推理框架的分诊台把病人分到了 3、8、11、14 号诊室；训练时，训练框架的分诊台重新打分，11 号和 5 号的分数本来只差一点点，一点数值抖动就让病人改去了 5 号。Keep Routing = 训练时不再重新分诊，照着采样时的挂号单走。

**收益。** 它说清了换位的后果：病历（采样时记下的 $\pi_{\theta_{old}}$）是 11 号医生写的，复诊（训练时算 $\pi_\theta$、求梯度）却由 5 号医生来做，两份记录比较的不是同一位医生的判断，“纠正”也落到了没看过这位病人的医生身上。

**失效的地方。** 两个分诊台其实是同一组参数在两套代码里运行，差别只来自数值实现和几步参数更新，不是两位不同的分诊护士。另外，挂号单只固定了“去哪几个诊室”；每位医生的意见占多大权重（门控值）训练时是否按当前参数重算，论文没有说。
:::

**做法。**

“To mitigate this, we preserve the expert routing paths used during sampling in the inference framework and enforce the same routing paths during training, ensuring that identical expert parameters are optimized. This Keep Routing operation was found crucial for RL training stability of MoE models, and has been adopted in our RL training pipeline since DeepSeek-V3-0324.” [@dsv32]

::: eq Keep Routing（笔者的形式化，论文无公式） | (15.17)
$$\mathcal E^{\text{train}}_{l,t}:=\mathcal E^{\text{infer}}_{l,t}\qquad\text{而不是}\qquad \mathcal E^{\text{train}}_{l,t}=\TopK\big(\{s_{e,t}+b_e\}_{e=1}^{256},\ 8\big)$$
$\mathcal E_{l,t}$：第 $l$ 层对第 $t$ 个 token 选中的专家集合；$s_{e,t}$：亲和度；$b_e$：无辅助损失均衡的偏置（第 11 课）。右式把选择简写成一次 top-8，略去了 V3 的节点受限路由（每个 token 至多去 4 个节点，[第 11 课 §5](lesson-11-v3-architecture.html#p5)）；这不影响这里的论点：带这条限制的选择同样是不连续的。门控值是否仍用当前参数重算，论文没有说。
:::

::: fig figures/l15-keep-routing.svg | 图 15.9 Keep Routing：同一个 token，推理框架与训练框架可能选出不同的专家
**看第 ② 行的两根红框柱：专家 5 与 11 的亲和度只差 0.0015，训练框架重算时它们换了位，被激活的参数就换了一块。** 第 ③ 行是 Keep Routing：训练时直接沿用采样时的集合。这是笔者构造的 16 选 4 的示意，扰动大小取在 BF16 舍入的量级之下（BF16 在 $[0.5,1)$ 里的间距是 $2^{-8}\approx0.0039$）；真实模型每层 256 选 8。
:::

::: ex 这种换位有多常见
V3.2 论文没有给出统计。Qwen 团队的 GSPO 论文给了一个数：在 48 层的 Qwen3-30B-A3B 上，“after each RL gradient update and for the same rollout sample, there are roughly 10% of the experts activated under the new policy $\pi_\theta$ that are different from those under the old policy $\pi_{\theta_{old}}$” [@gspo]。这是**一次参数更新**造成的差异，还没算上训练与推理的实现差异。按 V3 的配置，每个 token 在 58 个 MoE 层（61 层减去前 3 个稠密层）里各选 8 个专家，共 $58\times8=464$ 次选择；哪怕只有其中几次换位，这个 token 的前向计算就已经不是采样时那一个了。（把 Qwen 的 10% 套用到 DeepSeek 上没有依据，这里只用它说明量级。）
:::

同类做法在 GSPO 论文里叫 Routing Replay，该论文也指出了它的代价：“its practice of reusing routing modes incurs additional memory and communication overhead and can also limit the actual capacity of the MoE model” [@gspo]，并主张用序列级的重要性比代替逐 token 的比率，从而不再需要它。V3.2 说自己从 V3-0324（2025-03）起就在用 Keep Routing，GSPO 论文发表于 2025-07；两者谁先提出、实现是否相同，从这两篇文献无法确认。

## Keep Sampling Mask：采样时截掉了哪些 token，训练时也截掉

**问题。** RL 采样时通常不直接从完整的 softmax 里抽，而是用 top-p（只在累计概率达到 $p$ 的最小集合里抽）或 top-k。论文说这样做在 RL 里也有好处：“it avoids sampling extremely low-probability tokens that would be used as optimization targets” [@dsv32]。但这样一来，实际的采样分布是**截断并重新归一化**后的分布，而训练时算 $\pi_\theta$ 用的是完整词表：

“While such truncation preserves sample quality, it introduces a mismatch between the action spaces of $\pi_{old}$ and $\pi_\theta$, which violates the principles of importance sampling and destabilizes training. To address this, we preserve the truncation masks during sampling from $\pi_{old}$ and apply them to $\pi_\theta$ during training, ensuring both policies share identical action subspaces.” [@dsv32]

**理解。** 回到 §1 的类比：誊写员只从学生最有把握的那几种写法里抄（top-p 核），所以答卷上根本不会出现核外的写法；今天比较时如果按学生的全部写法去算“这个字有多大可能”，比较的范围里就混进了昨天不可能被抄上去的那一部分。Keep Sampling Mask 就是规定今天也只在昨天那张“候选字表”里比较。类比的边界：候选字表不是一张固定的表，每个位置各有自己的核，要逐个位置记下来。

**数学。** 设第 $i$ 条回答第 $t$ 个位置上，采样时保留的 token 集合（top-p 核）为 $\mathcal S_{i,t}$（与第 14 课 DSA 的选择集合无关）：

::: eq 截断后的采样分布，与训练时的同一掩码 | (15.18)
$$\pi^{\mathcal S}_{\theta_{old}}(a)=\frac{\pi_{\theta_{old}}(a)\,\mathbb 1[a\in\mathcal S_{i,t}]}{\sum_{b\in\mathcal S_{i,t}}\pi_{\theta_{old}}(b)},\qquad \pi^{\mathcal S}_{\theta}(a)=\frac{\pi_{\theta}(a)\,\mathbb 1[a\in\mathcal S_{i,t}]}{\sum_{b\in\mathcal S_{i,t}}\pi_{\theta}(b)}$$
$\mathbb 1[\cdot]$：条件成立为 1，否则为 0。右式是“对 $\pi_\theta$ 施加同一掩码并重新归一化”的写法；V3.2 是否重新归一化，论文没有说。
:::

重要性采样要求：只要新分布给某个动作正的概率，旧分布也必须给它正的概率，否则比率 $\pi_\theta/\pi^{\mathcal S}_{\theta_{old}}$ 在那里没有定义。截断恰恰破坏了这一条：核外的 token 永远采不到，加权平均只能覆盖核内，

$$\E_{a\sim\pi^{\mathcal S}_{\theta_{old}}}\Big[\frac{\pi_\theta(a)}{\pi^{\mathcal S}_{\theta_{old}}(a)}f(a)\Big]=\sum_{a\in\mathcal S_{i,t}}\pi_\theta(a)\,f(a)\ne\E_{a\sim\pi_\theta}\big[f(a)\big],$$

漏掉的正是 $\pi_\theta$ 放在核外的那部分概率。改用施加了同一掩码的 $\pi^{\mathcal S}_\theta$ 作分子，同样的计算给出 $\sum_{a\in\mathcal S_{i,t}}\pi^{\mathcal S}_\theta(a)f(a)=\E_{a\sim\pi^{\mathcal S}_\theta}[f(a)]$，两边的动作空间一致，恒等式重新成立。

::: ex 五个 token 的例子
$\pi_{\theta_{old}}=[0.5,0.3,0.15,0.04,0.01]$（token a–e），top-p $=0.95$：$0.5+0.3+0.15=0.95$，核是 $\{a,b,c\}$，截断后的采样分布是 $[0.5263,0.3158,0.1579,0,0]$。

几个小批之后 $\pi_\theta=[0.45,0.33,0.12,0.07,0.03]$。它自己的 top-p 核是 $\{a,b,c,d\}$（$0.90<0.95\le0.97$），而且它把 $0.07+0.03=0.10$ 的概率放在了旧核之外，那里旧分布的概率是 0。

token c 的比率有两种算法：用全词表的 $\pi_\theta$，$0.12/0.1579=0.760$；施加同一掩码并在核内重新归一化，$(0.12/0.90)/0.1579=0.844$。后者是前者的 $1/0.90=1.11$ 倍：重新归一化除掉的，正是那 0.10 的“核外质量”。Keep Sampling Mask 保证训练时只在 $\{a,b,c\}$ 上比较，d、e 不参与。还要注意一点：被采到的 token 总在核内，所以如果只把核外置零、不重新归一化，被采到的 token 上比率的数值与全词表口径完全相同（c 仍是 0.760）；掩码要改变比率，靠的是重新归一化这一步（笔者的分析）。

口径不同，后果可以大到改变“这个 token 要不要更新”。设 c 所在的回答优势为负，$\varepsilon=0.2$。按 (15.16)，全词表口径的比率 $0.760<0.8$，代理项被裁成常数，c 没有梯度；核内重新归一化口径的比率 $0.844>0.8$，c 照常有梯度。前一种口径把采样时根本不可能出现的 d、e 的 0.10 概率也算进了比较，所以把 c 看得比实际更“离策略”。（这是笔者的算例；如前所述，V3.2 是否重新归一化，论文没有说。）
:::

::: fig figures/l15-sampling-mask.svg | 图 15.10 Keep Sampling Mask：训练时对当前策略施加采样时的截断掩码 | wide
**看中间一栏的红框柱 d、e：当前策略给了它们 0.10 的概率，而采样分布在那里是 0；右栏施加同一个核之后，两边的动作空间重新一致。** 下方给出 token c 的两种比率。数据即上面的算例，由 (15.18) 计算。
:::

论文还报告了一个副作用：“Empirically, we find that combining top-p sampling with the Keep Sampling Mask strategy effectively preserves language consistency during RL training.” [@dsv32] 一种可能的解释（笔者推测）：中英文混杂往往始于一个低概率的“换语言” token，top-p 把这类 token 截掉，Keep Sampling Mask 又保证训练时不会因为比较口径不一致而把概率推向它们。RL 采样用的 top-p 取值论文没有给出；HF 模型卡建议本地部署用 temperature 1.0、top_p 0.95 [@hf-v32]，那是推理建议，不等于 RL 的设置。

::: key 本页要点
- 两个 off-policy 来源：一批 rollout 分多个小批更新；推理框架与训练框架的实现差异。V3.2 的 $\pi_{\theta_{old}}$ 取推理框架返回的概率，两种都算进去。
- 序列掩码 (15.14)(15.15)：$\hat A_i<0$ 且 $\bar D_i>\delta$ 时整条回答的策略项置 0；$\bar D_i$ 是 $\KL[\pi_{\theta_{old}}\Vert\pi_\theta]$ 的单样本估计；正样本永远保留；KL 项不受影响。裁剪只拦 $\rho<1-\varepsilon$ 的 token，掩码拦整条。
- Keep Routing (15.17)：训练沿用采样时的专家集合，自 V3-0324 起使用；top-$K$ 不连续，微小数值差就能换专家。
- Keep Sampling Mask (15.18)：训练时对 $\pi_\theta$ 施加采样时的 top-p/top-k 掩码，保证重要性采样的两边是同一个动作空间。
- $\delta$、top-p、$\varepsilon$、$\beta$、$G$ 等取值均未披露。
:::

# 智能体：边想边用工具 | 思考上下文怎样管理、冷启动、85,267 个任务从哪来、上下文不够时怎么办 | 30

**智能体**（agent）在这里指这样一种用法：模型不只回答一次，而是在一次任务里反复“想一想 → 调用一个工具（搜索、执行代码、读文件）→ 看工具返回的结果 → 再想 → 再调用”，直到给出答案。V3.1 能调用工具，但只在非思考模式下（§2）。V3.2 要让思考和工具调用交替进行，先要回答一个看似琐碎、其实决定成败的问题：**两次工具调用之间，上一段思考还留不留在上下文里？**

## R1 的做法为什么在这里浪费

R1 和 V3.1 的多轮规则都是“新一轮开始，就删掉上一轮的思考”（(15.3)）。把这条规则原样搬到工具调用上，每次工具返回都算“新一轮”，思考就被反复删掉。V3.2 论文说 [@dsv32]：

“We observed that replicating DeepSeek-R1’s strategy—discarding reasoning content upon the arrival of the second round of messages—results in significant token inefficiency. This approach forces the model to redundantly re-reason through the entire problem for each subsequent tool call.”

模型刚想清楚“先查 A、再用 A 的结果算 B”，调用完查 A 的工具，这段计划就没了，只能从头再想一遍。一次任务调用十几次工具，就要把同一个问题重新想十几次。

::: ex 每步重想一遍，要多写多少 token（笔者的玩具模型）
设一次任务调用 $n$ 次工具。每一步开始时，模型要先把题目和计划想明白，这部分思考约 $P$ 个 token；然后针对刚返回的工具结果再想 $s$ 个 token。取 $P=2{,}000$、$s=300$、$n=12$（为了算账而假设的数字，论文没有给这样的统计）。

- R1 式（每次工具返回都删思考）：每一步都要重写一遍计划，思考共 $n(P+s)=12\times2{,}300=27{,}600$ token。
- V3.2 式（工具返回不删）：计划只写一次，之后每步只写增量，共 $P+ns=2{,}000+12\times300=5{,}600$ token。

前者约是后者的 $27{,}600/5{,}600=4.9$ 倍；工具调用越多，差距越大，$n\to\infty$ 时比值 $\frac{n(P+s)}{P+ns}$ 趋于 $\frac{P+s}{s}=\frac{2{,}300}{300}\approx7.7$。代价在另一侧：V3.2 式把全部思考留在上下文里，输入越来越长，本页最后一节讲窗口满了怎么办。
:::

## V3.2 的两条规则

“Historical reasoning content is discarded only when a new user message is introduced to the conversation. If only tool-related messages (e.g., tool outputs) are appended, the reasoning content is retained throughout the interaction.”

“When reasoning traces are removed, the history of tool calls and their results remains preserved in the context.” [@dsv32]

**含义。** 只有**新的用户消息**才会触发删除；工具返回不算。删除时只删思考，工具调用和工具结果保留。

**理解。** 用 §1 的类比：一位学生在做一道需要查资料的大题，每查一次资料就回到草稿纸上接着写；草稿一直留着，直到老师布置**下一道题**，才把上一题的草稿扔掉，但查到的资料（工具结果）和交上去的答案还摆在桌上。这个比方的边界在于：草稿纸可以一直续，上下文窗口有上限（V3.2 是 128K）。保留思考省掉了重复推理，却让上下文涨得更快；资料一多，窗口照样会满。

**数学。** HF 仓库里的编码脚本 `encoding_dsv32.py` 把这条规则写成了代码（函数 `drop_thinking_messages`）[@hf-v32]。用公式表示：对消息序列 $(\text{msg}_1,\dots,\text{msg}_n)$，令

::: eq V3.2 的思考保留规则（据编码代码） | (15.19)
$$\begin{gathered}u^*=\max\{\,j:\ \mathrm{role}(\text{msg}_j)\in\{\text{user},\ \text{developer}\}\,\}\\[4pt] \text{assistant 消息 }\text{msg}_j\text{ 的思考}:\ \begin{cases}\text{删除}, & j<u^*\\ \text{保留}, & j>u^*\end{cases}\end{gathered}$$
$u^*$：最后一条用户（或 developer）消息的位置。user、system、tool 消息全部保留。
:::

代码里还有两处论文没提的细节（研究笔记读代码所得）：删除只在思考模式下进行；位于 $u^*$ 之前的 `developer` 消息会被整条丢掉，而不只是删掉思考。渲染提示时，若处于思考模式，每组工具结果之后接一个 `<think>`，让模型接着思考。

::: fig figures/l15-thinking-retention.svg | 图 15.11 工具调用中的思考保留：只有新的 user 消息才会删掉之前的思考 | wide
**从左往右看第 1 轮的三步：每次工具结果回来，之前的思考都留在输入里；第 2 轮开始时来了 User 2，三段思考全部变成红色虚框（删除），而工具调用、工具结果与回答 1 都还在。** 据 V3.2 论文图 4 重绘 [@dsv32]；方块大小不代表长度。
:::

::: ex 哪些思考被保留
消息序列：system、user₁、assistant₁（思考 + 工具调用）、tool、assistant₂（思考 + 工具调用）、tool、assistant₃（思考 + 回答）、user₂。现在要编码 user₂ 之后的提示。

$u^*$ 指向 user₂（第 8 条）。assistant₁、₂、₃ 都在它之前，三段思考全部删除；两次工具调用、两次工具结果和最终回答保留。如果在 assistant₃ 之前就编码（即模型还在处理 user₁），$u^*$ 指向 user₁，assistant₁、₂ 在它之后，思考都保留。HF 仓库的 `encoding/test_output.txt` 就是这个场景的真实编码样例 [@hf-v32]。
:::

V3.2 还换了工具调用的格式：不再用 V3.1 的 `<｜tool▁calls▁begin｜>…` 那一串特殊 token，而是一种 XML 式的 `<｜DSML｜function_calls>` / `<｜DSML｜invoke name="…">` 格式，并新增了一个只用于搜索智能体的 `developer` 角色：“The official API does not accept messages assigned to `developer`.”；这一版也不再提供 Jinja 模板 [@hf-v32]。

## 与智能体框架的兼容

规则 (15.19) 依赖一个前提：工具结果以 **tool** 角色回到上下文。有些框架用**用户消息**来模拟工具返回，那么每次工具返回都会被当作“新的用户消息”，思考就被反复删掉，回到 R1 的老问题。论文明说了这一点 [@dsv32]：

“Notably, certain agent frameworks, such as Roo Code or Terminus, simulate tool interactions via user messages. These frameworks may not fully benefit from our enhanced reasoning persistence due to the context management rules outlined above. Therefore, we recommend utilizing non-thinking models for optimal performance with such architectures.”

这直接影响了评测数字：V3.2 在 Terminal Bench 2.0 上报告的 46.4 是在 Claude Code 框架里跑的；用 Terminus 框架时只测了非思考模式，得 39.3 [@dsv32]。

::: warn API 文档后来变了
DeepSeek API 的“Thinking Mode”指南在 2025-12-09 的存档版里写的是：“When the next user question begins (Turn 2.1), the previous reasoning_content should be removed, while keeping other elements to send to the API. If reasoning_content is retained and sent to the API, the API will ignore it.”，与上面的规则一致。2026 年 9 月的现行版（V4 时代）改成了：“If the request carries the tools parameter: the reasoning_content of all previous turns should be passed back to the API and will be concatenated into the context.” [@ds-thinking-guide] 讲 V3.2 时要用前一个版本；现在调用 API 的行为已经不同。
:::

## 冷启动：用提示把两种能力拼起来

RL 需要模型**偶尔**能产生正确的轨迹，才有正样本可学。一开始，DeepSeek 手里有“会推理但不调工具”的数据，也有“调工具但不推理”的智能体数据，没有两者交织的数据。论文的办法是靠系统提示把两者拼起来 [@dsv32]：

“Given the availability of reasoning data (non-agentic) and non-reasoning agentic data, a straightforward strategy for integrating these two capabilities is through carefully designed prompting.”

“In this manner, although the reasoning in tool-use patterns may lack robustness, the model is occasionally able to generate the desired trajectories, thereby providing a basis for subsequent reinforcement learning stages.”

三种系统提示的模板列在论文附录 B 的表 6–8，其中“思考中调用工具”的模板规定最多执行 20 次代码（“with a maximum of 20 code executions”）[@dsv32]。

## 85,267 个任务从哪来

冷启动只能让模型“偶尔”做对；要把能力练出来，需要大量**可以自动判分**的智能体任务。V3.2 引言的说法是：“we generate over 1,800 distinct environments and 85,000 complex prompts” [@dsv32]。表 1 给出了拆分：

::: eq 表 1 的任务总数 | (15.20)
$$\underbrace{50{,}275}_{\text{搜索}}+\underbrace{24{,}667}_{\text{代码}}+\underbrace{5{,}908}_{\text{代码解释器}}+\underbrace{4{,}417}_{\text{通用}}=85{,}267$$
占比依次为 59.0%、28.9%、6.9%、5.2%。
:::

::: fig figures/l15-agent-synthesis.svg | 图 15.12 V3.2 的智能体 RL 任务：四类来源，以及通用智能体环境的合成流程 | wide
**上半部分看条的长短与实虚：近九成任务是搜索和代码，环境都是真实的；只有 5.2% 的通用智能体任务跑在合成环境里。下半部分从“输入”沿箭头走：取数据 → 合成工具 → 出题 → 验证不过就改 → 加难 → 用 RL 试解筛选。** 据 V3.2 论文表 1、表 5 与 §3.2.3 绘制 [@dsv32]。
:::

四类任务的做法各不相同：

- **搜索智能体**（真实环境，提示合成）：一条基于 V3.2 的多智能体流水线。先从大规模网页语料里采样各领域的长尾实体；出题智能体用搜索工具围绕每个实体探索，整理成问答对；多个配置不同（不同检查点、系统提示等）的答题智能体各自作答；最后由一个带搜索能力的验证智能体多轮核对，“retaining only samples where the ground-truth is correct and all candidates are verifiably incorrect.” [@dsv32] 只留“标准答案对、候选答案全错”的题，保证题目难。此外还混入一部分已有的“有益性”RL 数据，用多维度评分细则加生成式奖励模型打分。
- **代码智能体**（真实环境，提示取自网上）：从 GitHub 挖掘数以百万计的 issue 与 PR 配对，搭建可复现的修复环境。环境算搭建成功的条件是：“applying the gold patch results in a non-zero count of false-to-positive (F2P) test cases (indicating the issue is fixed) and a zero count of pass-to-fail (P2F) test cases (indicating no regressions).”，最终得到“tens of thousands of reproducible issue resolution environments”，覆盖 Python、Java、JavaScript、TypeScript、C、C++、Go、PHP 八种语言 [@dsv32]。（“false-to-positive”照抄原文，业界通常叫 fail-to-pass。）
- **代码解释器**（真实环境，提示取自网上）：用 Jupyter Notebook 作为代码解释器，题目覆盖数学、逻辑与数据科学。
- **通用智能体**（合成环境，提示合成）：见下一小节。

代码智能体的成功条件可以写成：

::: eq 代码环境的验收条件 | (15.21)
$$\text{环境合格}\iff \mathrm{F2P}(\text{gold patch})>0\ \ \wedge\ \ \mathrm{P2F}(\text{gold patch})=0$$
F2P：打上官方补丁后由失败变为通过的测试数（说明 issue 确实被修好）；P2F：由通过变为失败的测试数（说明没有引入回归）。
:::

前三类（搜索、代码、代码解释器）都是“真实环境、非真实用户”：“While these RL environments are real, the prompts employed are either extracted from Internet sources or synthetically generated, rather than obtained from actual user interactions.” 第四类则连环境也是造的：“For other tasks, the environment and prompts are both synthetically constructed.” [@dsv32]

## 通用智能体：难解、易验

**含义。** 一个“环境”= 一个沙箱数据库 + 一组工具函数；一个“任务”= 一道要用这些工具才能完成的题 + 一个能自动判对错的验证函数。

**理解。** 设计原则是“hard to solve but easy to verify” [@dsv32]：比如“规划一份满足若干约束的旅行行程”，要找出一份行程很难，检查一份行程是否满足约束却很容易。只有“易验”，才能在没有人工的情况下给 RL 提供奖励。

合成流程由一个“环境合成智能体”完成。给定任务类别（例如“planning a travel itinerary”）和一个带 bash 与搜索工具的沙箱，论文 §3.2.3 的步骤是（笔者核对原文后概括，编号与图 15.12 下半部分一致）：

1. **取数据**：用工具从网上生成或检索相关数据，存进沙箱数据库。
2. **合成工具**：写一组任务专用的工具，每个实现为一个函数。
3. **出题**：基于当前数据库提出一个简单任务，连同用 Python 实现的**解法函数**与**验证函数**。解法函数只能调用工具或做逻辑计算：“is restricted to invoking tool functions or performing logical computations, and cannot call other functions or directly access the database, ensuring the task can only be solved through the tool interface.”解法的输出必须通过验证函数，不通过就改：“If the solution is not validated, the agent will modify the solution or verification functions until the solution’s output passes the verification.”
4. **加难**：逐步提高任务难度，同步更新解法与验证函数；工具不够时扩充工具集。
5. **筛选**：用 V3.2 在全部候选上做 RL 试解，只留下 100 次采样里至少解出一次的（pass@100 > 0，见下面的前置知识框）。

第 5 步的原文是：

“Following this workflow, we obtain thousands of ⟨environment, tools, task, verifier⟩ tuples. We then perform RL on this dataset using DeepSeek-V3.2 and retain only instances with non-zero pass@100, resulting in 1,827 environments and their corresponding tasks (4,417 in total).” [@dsv32]

::: ex 一个玩具四元组（笔者构造，不是论文里的任务）
按“旅行行程”这个类别，四元组 ⟨环境, 工具, 任务, 验证器⟩ 大致长这样：

```python
# 环境：沙箱数据库里存着若干城市的航班与酒店（第 1 步从网上取来）
# 工具（第 2 步合成）：模型只能通过它们读数据库
def search_flights(src, dst, date): ...   # 返回 [(航班号, 起飞时间, 价格), ...]
def search_hotels(city, date): ...        # 返回 [(酒店名, 每晚价格, 评分), ...]

# 任务（第 3 步提出，第 4 步加难）：
#   "6 月 3 日从上海出发，去成都住两晚，再去西安住一晚，6 月 6 日回上海；
#    总价不超过 4000 元，每家酒店评分不低于 4.5。给出行程。"

def verify(plan):                         # 验证函数：只检查，不求解
    return (plan.dates_are_consecutive()
            and plan.total_price() <= 4000
            and all(h.rating >= 4.5 for h in plan.hotels)
            and plan.all_items_exist_in_database())
```

求解要在几十个航班与酒店的组合里搜索，反复调用工具；验证只是逐条检查约束，几行代码。加难也很便宜：再加一座城市、再压低预算，验证函数只多一两行。这就是“难解、易验”，也是这类任务不需要人工就能给 RL 提供奖励的原因：奖励就是 `verify(模型给出的行程)` 的真假。
:::

::: pre pass@k | 熟悉 pass@k 的可以跳过
pass@$k$：对同一道题独立采样 $k$ 次，至少有一次做对的概率。若单次做对的概率是 $p$，且各次独立，则
$$\text{pass@}k=1-(1-p)^k.$$
pass@100 > 0 就是“100 次里至少做对 1 次”：保留的是“能做对，但不一定每次都做对”的题。一次都做不对的题提供不了正样本，GRPO 从中学不到东西（第 13 课 §3 的“零信号”）。
:::

::: ex 这些合成任务有多难
论文随机抽了 50 个合成任务测试（表 5）：pass@1 为 V3.2-Exp 12%、Sonnet-4.5 34%、Gemini-3.0 Pro 51%、GPT-5-Thinking 62% [@dsv32]。V3.2-Exp 的 pass@2 与 pass@4 是 18% 与 26%。

如果各次尝试独立、单次成功率都是 12%，那么 pass@2 应为 $1-0.88^2=22.6\%$，pass@4 应为 $1-0.88^4=40.0\%$。实测低得多，说明失败并不是“运气差、多试几次就好”，而是有一批题目对这个模型来说就是做不出来：成功率在题与题之间很不均匀。这个推断有数学依据：设第 $j$ 题的单次成功率是 $p_j$，平均 $\bar p=12\%$。$f(p)=1-(1-p)^k$ 是 $p$ 的凹函数，由詹森不等式，50 题平均的 $\text{pass@}k=\frac1{50}\sum_jf(p_j)\le f(\bar p)$，$p_j$ 越分散，差得越多；极端情形是 12% 的题每次都对、其余每次都错，那时 pass@$k$ 对所有 $k$ 都停在 12%。实测的 18%、26% 介于两个极端之间。（这是笔者按独立假设做的对照，论文没有这样分析。）

再粗算一步（笔者的玩具模型）：假设题目只有两种，占比 $q$ 的题单次成功率为 $p$，其余的题永远做不出。用实测的 pass@1 与 pass@2 解出 $p,q$，再去预测 pass@4：

$$\begin{aligned}\text{pass@}1&=qp=0.12,\qquad \text{pass@}2=q\,\big[1-(1-p)^2\big]=0.18\\[6pt] \frac{\text{pass@}2}{\text{pass@}1}&=\frac{1-(1-p)^2}{p}=2-p=1.5\ \ \Rightarrow\ \ p=0.5,\quad q=0.24\\[6pt] \text{pass@}4&=0.24\times\big(1-0.5^4\big)=0.225\qquad(\text{实测 }0.26)\end{aligned}$$

预测值比实测略低，说明“做不出”的那 76% 里还有少数题偶尔能做对。粗略地说：V3.2-Exp 对约四分之一的合成任务有一半把握，对其余大多数几乎无从下手。
:::

这些合成任务练出来的能力能不能迁移？论文的图 5 显示：为了排除长思维链和其他 RL 数据的影响，只用合成的通用智能体任务、在非思考模式下对 V3.2-SFT 做 RL，τ²-bench、MCP-Mark、MCP-Universe 上都明显提升。对照组是只在搜索与代码环境里做过 RL 的 V3.2-Exp：“In contrast, restricting RL to code and search scenarios does not improve performance on these benchmarks, further highlighting the potential of synthetic data.” [@dsv32] 通用智能体任务只占 5.2%，却承担了向陌生工具环境泛化的那一部分（这是笔者对图 5 的解读：论文的对照组是另一个模型 V3.2-Exp，也没有在完整的混合 RL 里做去掉合成任务的消融）。

## 上下文不够时：测试时的上下文管理

V3.2 的上下文是 128K。搜索智能体一次任务里要读大量网页，BrowseComp 这个基准上“approximately 20%+ of the test cases exceed this limit” [@dsv32]。论文在 token 用量超过上下文窗口的 80% 时触发上下文管理：

::: eq 触发点 | (15.22)
$$0.8\times131{,}072=104{,}857.6\ \text{token}$$
:::

三种策略的定义（原文照录）[@dsv32]：Summary“summarizes the overflowed trajectory and re-initiates the rollout”；Discard-75%“discards the first 75% tool call history in the trajectory to free up spaces”；Discard-all“resets the context by discarding all previous tool call history (similar to the new context tool)”。对照组是并行扩展：独立采 $N$ 条轨迹，选步数最少的一条。

::: fig figures/l15-context-management.svg | 图 15.13 测试时的上下文管理：用到 80% 窗口时触发，三种策略各删掉什么 | wide
**看红色虚线框：Summary 把整段历史换成一小块摘要，Discard-75% 删掉最早的四分之三工具调用历史，Discard-all 全部删掉；右栏是各自的 BrowseComp 得分，最简单的 Discard-all 最高。** 竖虚线是触发点 $0.8\times131{,}072=104{,}858$ token，窗口长度与触发点按比例画，其余方块长度是示意。Discard-75% 论文只画了曲线（其图 6），没有给单独的数字，所以不标。据 V3.2 论文 §4.1、§4.4 绘制 [@dsv32]。
:::

结果：不做上下文管理时 BrowseComp 得 51.4；Summary 把平均步数延长到 364，最高到 60.2；“Despite its simplicity, Discard-all performs well in both efficiency and scalability, achieving a score of 67.6, comparable to parallel scaling while using significantly fewer steps.” [@dsv32] 最简单的“全部扔掉”比“做摘要”还好，提高了 $67.6-51.4=16.2$ 分。表 2 里 V3.2 的 BrowseComp 67.6 就是用了这种上下文管理的成绩（带星号）。论文也承认问题的另一面：V3.2“frequently engages in redundant self-verification, generating excessively long trajectories”，在 MCP-Mark 的 GitHub 与 Playwright 任务上常常因此超过 128K [@dsv32]。

::: key 本页要点
- V3.2 的思考保留 (15.19)：只有新的 user 消息才删除之前的思考；工具返回不删；删除时保留工具调用与结果。这是“思考中调用工具”能省 token 的前提。
- 用 user 消息模拟工具返回的框架（Roo Code、Terminus）会反复触发删除，论文建议它们用非思考模式；Terminal Bench 2.0 的 46.4 是在 Claude Code 框架下测的。
- 冷启动靠系统提示把推理数据与智能体数据拼起来；之后在 85,267 个可自动判分的任务上做 RL：搜索 59.0%、代码 28.9%、代码解释器 6.9%、通用智能体 5.2%（1,827 个合成环境，pass@100 > 0 筛选）。
- 合成任务很难（V3.2-Exp pass@1 12%），且只在合成任务上做 RL 就能迁移到 τ²-bench 等基准。
- 超过 80% 上下文时做上下文管理，BrowseComp 从 51.4 升到 67.6（Discard-all）。
:::

# 推理的上限与总账 | Speciale 放宽长度约束的代价、DeepSeekMath-V2 的自我验证、竞赛“金牌”的评测条件、全课总账 | 30

正式版 V3.2 的思考长度是被约束过的。论文说它的成绩“is constrained by a length constraint reward model; upon removal of the restriction, we observe further improvement in model performance” [@dsv32]。把约束放开会怎样？这就是 V3.2-Speciale。它的证明能力又来自同期的 DeepSeekMath-V2。这一页先讲这两个模型各自的说法与评测条件，最后把全课算一遍总账。

## V3.2-Speciale：把长度约束放开

§4 引过 Speciale 的训练配方：只用推理数据、RL 时长度惩罚更小、并入 DeepSeekMath-V2 的数据与奖励方法 [@dsv32]。它的成绩，论文在三处用了三种措辞，逐一照录：

摘要：“Notably, our high-compute variant, DeepSeek-V3.2-Speciale, surpasses GPT-5 and exhibits reasoning proficiency on par with Gemini-3.0-Pro, achieving gold-medal performance in both the 2025 International Mathematical Olympiad (IMO) and the International Olympiad in Informatics (IOI).”

引言：“Notably, with the aim of pushing the boundaries of open models in the reasoning domain, we relaxed the length constraints to develop DeepSeek-V3.2-Speciale. As a result, DeepSeek-V3.2-Speciale achieves performance parity with the leading closed-source system, Gemini-3.0-Pro (DeepMind, 2025a). It shows gold-medal performance in the IOI 2025, ICPC World Final 2025, IMO 2025, and CMO 2025.”

§4.2：“Remarkably, as shown in Table 4, this general-purpose model attains gold-medal level performance in the 2025 International Olympiad in Informatics (IOI) and the ICPC World Finals (ICPC WF) without targeted training. Furthermore, by incorporating techniques from Shao et al. (2025), the model excels in complex proof tasks, reaching gold-medal thresholds in the 2025 International Mathematical Olympiad (IMO) and China Mathematical Olympiad (CMO)”

表 4 的标题还给了名次：“DeepSeek-V3.2-Speciale ranked 2nd in ICPC WF 2025 and 10th in IOI 2025.” [@dsv32]

这些是 DeepSeek **自己按赛制评测**、事后对照奖牌线得出的，不是参赛获奖。评测条件写在附录 D [@dsv32]：

“For all competitions, the model’s maximum generation length is set to 128k. No tools or internet access are used, and testing strictly adheres to the contest’s time and attempt limits.”

IOI 每题“first sampled 500 candidate solutions”，经多阶段过滤后“selected the 50 samples with the longest thinking traces for final submission”；ICPC 每题生成 32 个候选、用同样的过滤；IMO 与 CMO 用“a generate-verify-refine loop”，与 DeepSeekMath-V2 的做法相同。CMO 评的是英文版（论文脚注：“We evaluated the English version of CMO 2025.”）。

::: fig figures/l15-competitions.svg | 图 15.14 V3.2-Speciale 与 DeepSeekMath-V2 的竞赛成绩，对照官方奖牌线 | wide
**看绿色竖线落在哪里：IMO 2025 的 35 分恰好压在金牌线 35 上；IOI 的 492 分比金牌线高出 53.7 分。** 分块是逐题得分：Speciale 取自 V3.2 论文表 4 [@dsv32]；DeepSeekMath-V2 的表 1 [@dsmathv2] 只标出哪些题完全解出（IMO 2025 的 P1–P5）和总得分率，图中每题 7 分是由“完全解出”折算的，CMO 2024 的 93 分由 73.8% 反推。金牌线与名次取自 IMO、IOI 官方成绩页与 ICPC 官方记分牌 [@imo2025; @ioi2025; @icpc2025]；CMO 的金牌线未能核实，所以没有画。每根条按各自的满分画成比例。
:::

::: ex 用官方数据复核“金牌”与名次
- **IMO 2025**：逐题 7、7、7、7、7、0，合计 35/42 = 83.3%。官方成绩页：金牌 72 人，最低 35 分 [@imo2025]。35 **恰好等于**金牌线，第 6 题 0 分。
- **IOI 2025**：100 + 82 + 72 + 100 + 55 + 83 = 492/600。官方金牌线 438.30，高出 53.70；人类第 9 名 502.78、第 10 名 491.65 [@ioi2025]，492 插在两者之间，与论文的“10th”一致。
- **ICPC WF 2025**：解出 A、C、D、E、F、H、I、J、K、L 共 10 题，提交次数合计 $3+1+1+2+2+1+1+1+1+1=14$。官方记分牌：12 题，第 1 名解 11 题，第 2、3 名各解 10 题 [@icpc2025]。解 10 题的队伍按罚时排名，论文只给了提交次数、没给解题时间，所以“第 2”无法从公开数据复核。
- **CMO 2025**：18 + 18 + 9 + 21 + 18 + 18 = 102/126 = 81.0%。
:::

::: warn “金牌水平”的四个前提
一，自评：没有参加比赛，是按赛制自行评测，再对照这一届人类选手的奖牌线。二，采样量：IOI 每题先采 500 个解，再按“思考最长”挑 50 个提交，这在赛制允许的提交次数之内，但不是“一次就写对”。三，评分者：IMO、CMO 的证明由谁评分，V3.2 论文没有写。四，CMO 用的是英文译本。
:::

## 多想的代价：token

放开长度约束，Speciale 在所有推理基准上都用了更多的 token。表 3 给出了每个基准的平均输出 token 数 [@dsv32]。

::: eq Speciale 与 V3.2 的 token 之比（8 个基准的算术平均） | (15.23)
$$\begin{aligned}\bar\kappa&=\frac18\sum_{b=1}^{8}\frac{\ell^{\text{Speciale}}_b}{\ell^{\text{V3.2}}_b}\\ &=\frac18\Big(\tfrac{23}{16}+\tfrac{27}{19}+\tfrac{25}{18}+\tfrac{45}{27}+\tfrac{27}{16}+\tfrac{77}{42}+\tfrac{16}{7}+\tfrac{35}{21}\Big)=\frac{13.387}{8}\approx1.67\end{aligned}$$
$\bar\kappa$：Speciale 相对 V3.2 的 token 倍数，8 个基准取算术平均；$\ell_b$：基准 $b$ 上的平均输出长度（千 token），顺序为 AIME 2025、HMMT Feb、HMMT Nov、IMOAnswerBench、LiveCodeBench、Codeforces、GPQA、HLE。8 个比值依次约为 1.44、1.42、1.39、1.67、1.69、1.83、2.29、1.67；用这些两位小数去平均会得到 1.675，与 1.673 的差来自舍入。若改用 token 总和之比 $275/166$，是 1.66。
:::

::: fig figures/l15-speciale-tokens.svg | 图 15.15 准确率与输出 token 数：Speciale 用更多 token 换分 | wide
**看每个小图里的蓝色箭头：从空心圈（V3.2）到实心圈（Speciale），箭头都指向右上方，右上角的数字是 token 的倍数。** 紫色方块是 Gemini-3.0 Pro：在多数基准上它用更少的 token 拿到相近或更高的分。数据取自 V3.2 论文表 3 [@dsv32]。
:::

论文自己点明了这个代价：“However, the token efficiency of DeepSeek-V3.2-Speciale remains significantly inferior to that of Gemini-3.0-Pro. To mitigate deployment costs and latency, we imposed stricter token constraints during the training of the official DeepSeek-V3.2, aiming to optimize the trade-off between performance and cost.” [@dsv32] 所以 Speciale 是一个“实验性变体”：模型卡写明“designed exclusively for deep reasoning tasks and does not support the tool-calling functionality” [@hf-v32]；官方 API 只通过一个临时端点提供，到 2025-12-15 15:59（UTC）为止，但权重在 Hugging Face 上开源，端点到期后仍可自行部署 [@ds-news]。

## DeepSeekMath-V2：让模型检查自己的证明

第 13 课的 R1 靠“最终答案对不对”给奖励。DeepSeekMath-V2 指出这种奖励的两个毛病 [@dsmathv2]：

“First, it serves as an unreliable proxy for reasoning correctness – a model can arrive at the correct answer through flawed logic or fortunate errors. Second, it is inapplicable to theorem proving tasks, where problems may not require producing numerical final answers and rigorous derivation is the primary objective.”

它的办法是训练一个**验证器**来给证明打分，再用验证器的分数当奖励训练**生成器**。

::: ana 类比：阅卷老师、复核组与附自查单的考生
**映射。** 验证器 = 阅卷老师：读考生的证明，写出扣分理由，给 0、0.5 或 1 分。元验证器 = 复核组：不重新阅卷，只检查阅卷老师写的扣分理由是否站得住。生成器 = 考生，而且交卷时必须附一张“自查单”（下文的自评 $Z$），写明自己觉得哪里有问题、该得几分。

**收益。** 它让两处奖励设计变得好懂。第一，如果阅卷老师只要分数给对就拿满分，他就可能随手编一条扣分理由，复核组正是为堵这个漏洞设的。第二，考生的自查单与阅卷结果对得上才加分，所以谎称“全对”不如老实承认漏洞（下面“诚实为什么划算”的算例会算出来）。

**失效的地方。** 三个角色都是语言模型，没有一位绝对正确的最终裁判。阅卷老师判断不了的难证明，只能靠放大验证算力来自动标注（见下文“让验证器跟上”）；这一环出错，错误会顺着奖励传给考生。
:::

**验证器。** 给定题目 $X$ 和证明 $Y$，验证器 $\pi_\varphi$ 先列出发现的问题，再在三档里给分：“1 for complete and rigorous proofs with all logical steps clearly justified; 0.5 for proofs with sound overall logic but minor errors or omitted details; and 0 for fundamentally flawed proofs containing fatal logical errors or critical gaps.” [@dsmathv2] 冷启动的题库 $\mathcal D_p$ 是从 AoPS 收集的 17,503 道题（优先收奥赛、国家队选拔考试和 2010 年以后明确要求证明的题）[@dsmathv2]。验证器的奖励是它给的分数 $s'$ 与专家标注的分数 $s$ 有多接近（MV2 式 1、3）：

::: eq 验证器的奖励（MV2 式 1、3） | (15.24)
$$R_{\text{score}}(s',s)=1-\abs{s'-s},\qquad R_V=R_{\text{format}}\cdot R_{\text{score}}\cdot R_{\text{meta}}$$
$s,s'\in\{0,0.5,1\}$；$R_{\text{format}}$：输出格式是否合规（0 或 1）。
:::

$R_{\text{meta}}$ 是**元验证**（meta-verification）的分数，堵的是一个漏洞：“when evaluating flawed proofs (where $s_i<1$) during training, the verifier can receive full reward by predicting the correct scores while hallucinating non-existent issues” [@dsmathv2]。验证器可以分数给对、理由却是编的。于是再训练一个元验证器 $\pi_\eta$，检查验证器分析里指出的问题是否真实存在，把它的打分乘进奖励。效果：在验证集上，元验证器评出的分析质量“improved from 0.85 to 0.96, while maintaining the same accuracy in proof score prediction” [@dsmathv2]。

**生成器。** 生成器 $\pi_\theta$ 写出证明 $Y$ 之后，还要写一份自评 $Z$（列出自己的问题并给自己打分 $s'$）。论文观察到，不这样约束时，“the generator tends to claim correctness even when the external verifier easily identify flaws” [@dsmathv2]。奖励把两部分加权（MV2 式 5、6；论文的权重记作 $\alpha,\beta$，本课改写）：

::: eq 生成器的奖励（MV2 式 5、6） | (15.25)
$$\begin{gathered}R=R_{\text{format}}(Y,Z)\cdot\big(w_Y\,R_Y+w_Z\,R_Z\big),\qquad w_Y=0.76,\ \ w_Z=0.24\\[4pt] R_Z=R_{\text{score}}(s',s)\cdot R_{\text{meta}}(Z)\end{gathered}$$
$R_Y=s$：验证器给证明 $Y$ 的分数；$s'$：生成器在自评 $Z$ 里给自己的分数；$R_{\text{meta}}(Z)$：对自评的元验证分。注意 $s,s'$ 换了角色：训练验证器时，$s'$ 是验证器给的分、$s$ 是标注分；到了生成器，$s'$ 是自评分、$s$ 是验证器给的分，$R_{\text{score}}$ 的形式不变。
:::

::: ex 诚实为什么划算
设 $R_{\text{format}}=R_{\text{meta}}=1$，由 (15.24)(15.25)：

| 验证器给的分 $s$ | 生成器自评 $s'$ | $R_Z=1-\abs{s'-s}$ | $R=0.76s+0.24R_Z$ |
|---|---|---|---|
| 1 | 1 | 1 | 1.00 |
| 1 | 0.5 | 0.5 | 0.88 |
| 0.5 | 1（夸大） | 0.5 | 0.50 |
| 0.5 | 0.5（诚实） | 1 | 0.62 |
| 0 | 1（夸大） | 0 | 0.00 |
| 0 | 0（诚实） | 1 | 0.24 |

一个只值 0.5 分的证明，老实自评拿 0.62，夸成满分只拿 0.50；一个错误的证明，老实承认还能拿 0.24，谎称正确拿 0。论文的话是“Faithful acknowledgment of errors is rewarded over false claims of correctness.”，以及“A good strategy to obtain high rewards for the proof generator is to identify and resolve as many issues as possible before finalizing the response.” [@dsmathv2]
:::

::: fig figures/l15-mathv2-loop.svg | 图 15.16 DeepSeekMath-V2：验证器、元验证器与会自评的生成器 | wide
**左边看三个角色怎样首尾相接：生成器写证明与自评，验证器给证明打分，元验证器检查分析是否属实，绿色虚线把奖励送回生成器。右边看奖励表里红、绿两色：同一个证明，诚实自评（绿）总比夸大（红）得分高。** 据 DeepSeekMath-V2 论文第 2 节绘制，表中数值由 (15.25) 计算 [@dsmathv2]。
:::

**让验证器跟上。** 生成器越来越强，验证器就会遇到它判断不了的证明。论文的办法是：“we propose to scale verification compute to automatically label new hard-to-verify proofs, creating training data to further improve the verifier.”，并说“In our last two training iterations, this fully automated pipeline replaced human annotation entirely.” [@dsmathv2] 训练用 GRPO，每轮先训验证器，再从验证器的检查点初始化生成器；基座是 DeepSeek-V3.2-Exp-Base [@dsmathv2]。

**测试时的大规模搜索。** 最好的成绩来自一个“生成—验证—修正”的搜索：每道题维护一个证明池，“initialized with 64 proof samples with 64 verification analyses generated for each”；每轮选平均验证分最高的 64 个证明，各配 8 条分析（优先挑指出问题的）去修正，“This process continues for up to 16 iterations or until a proof successfully passes all 64 verification attempts” [@dsmathv2]。

**成绩（原文照录）。** “our approach solved 5 of 6 problems from IMO 2025 and 4 problems plus partial credit on another from CMO 2024, achieving gold medal performance in both pinnacle high-school competitions. On Putnam 2024, the preeminent undergraduate mathematics competition, our model solved 11 of 12 problems completely and the remaining problem with minor errors, scoring 118/120 and surpassing the highest human score of 90.” [@dsmathv2] 这些证明由其自家的数学专家评分（“mathematical experts assessed the highest-scoring proofs”）。在 IMO-ProofBench 上（除它自己以外的分数都取自该基准的原作者），它的 Basic / Advanced 两档得 99.0 / 61.9，对照 DeepMind 的 DeepThink（IMO Gold）89.0 / 65.7：“outperforms … on the basic set and remains competitive on the advanced set” [@dsmathv2]。作者也写明了局限：“the hardest IMO-level problems remain challenging for our model.” [@dsmathv2]

::: err 两处常见的混淆
**“DeepSeekMath-V2 在 IMO 2025 拿了满分”**：不对，是 6 题解出 5 题，35/42，恰好等于金牌线。**“两个模型都在 CMO 2025 拿了金牌”**：DeepSeekMath-V2 报告的是 **CMO 2024**（73.8%，笔者按每题 21 分反推约 93/126），V3.2-Speciale 报告的是 **CMO 2025**（102/126）。
:::

## 算总账

| 版本 | 改了什么 | 关键数字 | 本课 |
|---|---|---|---|
| V3.1 | 混合思考：提示末尾 `<think>` / `</think>` 切换 | 2 个复用的占位 token（128798、128799） | §2 |
| V3.1 | 长上下文续训 | 630B + 209B = 839B（V3 原扩展的 10.01 倍与 3.32 倍） | §2 |
| V3.1 | UE8M0 缩放：$s=2^{\lceil\log_2(a/448)\rceil}$ | E4M3 利用率 $(\tfrac12,1]$，平均 0.721；`o_proj` 例外 | §3 |
| V3.2-Exp | DSA（第 14 课） | 续训 945.8B | §2、第 14 课 |
| V3.2 | 专家蒸馏 + 单阶段混合 RL | 8 个专家；后训练 > 预训练成本的 10%（V3：0.188%） | §1、§4 |
| V3.2 | 优势只减组均值 | 奖励 [1,0,0,0] 时优势 [+0.75, −0.25, −0.25, −0.25]（DeepSeekMath 口径 [+1.732, −0.577, …]） | §4 |
| V3.2 | 无偏 KL：$\rho\cdot k_3$ | 权重 $\rho\log z$ 对 $1-1/z$；$z=0.01$ 时差 21.5 倍 | §5 |
| V3.2 | off-policy 序列掩码 | 仅掩 $\hat A<0$ 且 $\bar D>\delta$ 的整条回答 | §6 |
| V3.2 | Keep Routing / Keep Sampling Mask | 前者自 V3-0324 起用 | §6 |
| V3.2 | 思考保留规则 + 智能体任务 | 85,267 个任务，1,827 个合成环境 | §7 |
| Speciale | 放开长度约束，推理数据，并入 MathV2 奖励 | token 平均 1.67 倍；IMO 35/42、IOI 492/600 | §8 |

同一基准、同一口径的版本间进步（研究笔记汇总各模型卡与论文）[@hf-v31; @hf-v31t; @dsv32exp; @dsv32]：

| 基准 | V3.1 | V3.1-Terminus | V3.2-Exp | V3.2 |
|---|---|---|---|---|
| MMLU-Pro | 84.8 | 85.0 | 85.0 | 85.0 |
| GPQA-Diamond | 80.1 | 80.7 | 79.9 | 82.4 |
| HLE（无工具） | 15.9 | 21.7 | 19.8 | 25.1 |
| AIME 2025 | 88.4 | 88.4 | 89.3 | 93.1 |
| BrowseComp | 30.0 | 38.5 | 40.1 | 51.4（上下文管理后 67.6） |
| SWE Verified | 66.0（非思考） | 68.4 | 67.8 | 73.1（思考）/ 72.1（非思考） |

SWE Verified 一行要注意模式：V3.1 只在非思考模式调用工具，所以它的 66.0 是非思考模式的成绩；同模式比较应是 66.0 → 72.1。知识类的 MMLU-Pro 一年几乎没动，涨得多的是推理（HLE、AIME）与智能体（BrowseComp、SWE）：这正是后训练的地盘。论文自己列出的局限也指向同一处 [@dsv32]：

“First, due to fewer total training FLOPs, the breadth of world knowledge in DeepSeek-V3.2 still lags behind that of leading proprietary models.”

“Second, token efficiency remains a challenge; DeepSeek-V3.2 typically requires longer generation trajectories (i.e., more tokens) to match the output quality of models like Gemini-3.0-Pro.”

最后，本课讲到的许多超参数**都没有公开**：off-policy 阈值 $\delta$、裁剪系数 $\varepsilon$、各领域的 KL 系数 $\beta$、组大小 $G$、RL 采样的 top-p、长度惩罚的形式、RL 的总步数（只说“thousands of steps”）、后训练的绝对 GPU 小时。读到这些量的具体取值时，要先问出处。

下一课 [第 16 课](lesson-16-mhc.html) 离开后训练，回到架构：残差流从一条变成多条（Hyper-Connections），以及 DeepSeek 怎样用双随机矩阵约束它（mHC）。V4 会把 mHC、第 14 课的稀疏注意力和本课的专家训练一起放进同一个模型（[第 18 课](lesson-18-v4.html)）。

::: key 本页要点
- Speciale = 只用推理数据、减小长度惩罚、并入 DeepSeekMath-V2 的数据与奖励。“金牌水平”是自评后对照奖牌线：IMO 35/42 恰好压线，IOI 492/600 高出金牌线 53.7 分、介于人类第 9 与第 10 名之间；无工具、无联网、最长 128k，IOI 每题采 500 选 50。
- 代价是 token：8 个推理基准平均 1.67 倍，且不支持工具调用；正式版 V3.2 因此加了更严的 token 约束。
- DeepSeekMath-V2：三档分数的验证器 + 元验证器 + 自评的生成器，奖励 $0.76R_Y+0.24R_Z$ 让诚实自评比夸大得分高；放大验证算力自动标注，最后两轮完全取代人工。IMO 2025 5/6，Putnam 2024 118/120。
- 一年间知识类基准几乎不动，推理与智能体大幅上升；大量 RL 超参数未公开。
:::

# 练习与答案 | 12 题：概念、计算、推导 | 50

先自己做，再展开答案。计算题的数字都可以用计算器复核；推导题要求每一步写出理由。

## 概念题

:::: quiz 1. 一个 token 的差别
（a）写出 V3.1 非思考模式与思考模式首轮提示的最后两个 token。（b）第 3 轮提问时，第 1、2 轮在上下文里各长什么样？（c）为什么说“V3.1 不能在思考的过程中调用工具”与 (b) 的规则有关？

::: answer 看答案
（a）非思考：`<｜Assistant｜></think>`；思考：`<｜Assistant｜><think>`。两者只差最后一个 token（128799 与 128798），见 (15.2) 与图 15.2。

（b）按 (15.3)，历史轮次一律是 `<｜User｜>问题<｜Assistant｜></think>回答<｜end▁of▁sentence｜>`：思考内容被删掉，`</think>` 保留。不管当时用的是哪种模式，历史里看起来都像一次非思考回答。

（c）工具调用意味着同一个问题里有多次“模型输出 → 工具结果 → 模型再输出”。如果每次工具返回都按“新一轮”处理，(b) 的规则会把上一段思考删掉，模型每调用一次工具就得从头再想（§7 引的 V3.2 论文原话：“forces the model to redundantly re-reason through the entire problem for each subsequent tool call”）。这可以解释 V3.1 为什么只在非思考模式下支持通用的工具调用（模型卡没有说明原因，这是笔者的推测；搜索智能体另有思考模式专用格式）；V3.2 把删除的触发条件改成“只有新的 user 消息”(15.19)，才让思考与工具调用能够交替。
:::
::::

:::: quiz 2. 判断正误，并说明理由
（a）off-policy 序列掩码会丢掉所有离策略太远的回答。（b）被掩掉的回答对损失不再有任何贡献。（c）V3.2 里的 $\pi_{\theta_{old}}$ 就是训练框架用上一步参数重算的概率。（d）Keep Routing 是 V3.2 首创的技术。（e）KL 项乘上 $\rho$ 之后，在 $\rho=1$ 时它和原始 k3 完全等价。

::: answer 看答案
（a）错。只掩“$\hat A_i<0$ **且** $\bar D_i>\delta$”的回答 (15.15)；正优势的回答无论多远都保留（“we only mask sequences with negative advantages”）。

（b）错。掩码只乘在裁剪代理项上 (15.14)，KL 项在掩码之外，被掩的回答仍然受 KL 约束。

（c）错。论文明说“$\pi_{old}$ here denotes the sampling probability directly returned by the inference framework”，所以比率同时包含“多个小批”与“两套框架的实现差异”两种离策略来源（§6）。

（d）错。论文说它“has been adopted in our RL training pipeline since DeepSeek-V3-0324”，比 V3.2 早大半年；Qwen 的 GSPO 论文把同类做法叫 Routing Replay。谁最先提出，现有文献无法确认。

（e）错。**值**相同，**梯度**不同：$\nabla\rho=\rho\,\nabla\log\pi_\theta\ne0$，所以 $\nabla(\rho k_3)=\rho\log(\pi_\theta/\pi_{ref})\nabla\log\pi_\theta$，而 $\nabla k_3=(1-\pi_{ref}/\pi_\theta)\nabla\log\pi_\theta$（§5）。实现时如果把整个 $\rho$ detach，就退回了原始 k3 的梯度。
:::
::::

:::: quiz 3. 思考保留与框架
消息序列：system、user₁、assistant₁（思考 + 工具调用）、tool、assistant₂（思考 + 工具调用）、tool、assistant₃（思考 + 回答）、user₂、assistant₄（思考 + 工具调用）、tool。（a）现在编码提示让模型继续生成，按 V3.2 的规则哪几段思考留在上下文里？（b）如果某个框架把每个 tool 消息都改成 user 消息发回来，结果会怎样？论文对这类框架有什么建议，它又怎样影响了 Terminal Bench 2.0 的数字？

::: answer 看答案
（a）最后一条 user 消息是 user₂，$u^*$ 指向它 (15.19)。assistant₁、₂、₃ 在它之前，三段思考删除；它们的工具调用、工具结果与回答保留。assistant₄ 在 user₂ 之后，思考保留。所以上下文里只有 assistant₄ 的思考。

（b）每个“伪装成 user 的工具结果”都会成为新的 $u^*$，于是每次工具返回，之前的思考都被删掉，退回 R1 式的“每步重想”。论文建议这类框架（Roo Code、Terminus）使用非思考模式。V3.2 的 Terminal Bench 2.0 成绩 46.4 因此是在 Claude Code 框架里测的；用 Terminus 只测了非思考模式，得 39.3。
:::
::::

## 计算题

:::: quiz 4. 续训账
V3 报告的两阶段上下文扩展：32K 阶段 1000 步、每步 1920 条；128K 阶段 1000 步、每步 480 条（K = 1024）。V3.1 的两阶段分别是 630B 与 209B；V3.2-Exp 的 DSA 续训是预热 1000 步 × 16 条 × 128K，加稀疏阶段 15000 步 × 480 条 × 128K。（a）求 V3 每阶段的 token 数与 V3.1 的倍数。（b）求 V3 基座之后累计续训了多少 token，占 V3 预训练 14.8T 的百分之几。

::: answer 看答案
（a）$1000\times1920\times32{,}768=62{,}914{,}560{,}000$；$1000\times480\times131{,}072=62{,}914{,}560{,}000$，两阶段都是 62.91B。倍数 $630/62.91=10.01$、$209/62.91=3.32$，对应模型卡的“10-fold”“3.3x”。

（b）DSA 预热 $1000\times16\times131{,}072=2.097\text{B}$；稀疏阶段 $15{,}000\times480\times131{,}072=943.72\text{B}$。累计 $630+209+2.10+943.72=1{,}784.8\text{B}\approx1.78\text{T}$，占 $1.785/14.8=12.1\%$。
:::
::::

:::: quiz 5. UE8M0 缩放
两个 $1\times128$ 激活分块的最大绝对值分别是 $a=100$ 与 $a=0.05$。（a）分别求 FP32 缩放因子与 UE8M0 缩放因子（写成 $2^n$），以及量化后的最大码值与利用率 $u$。（b）对 $a=100$ 的分块，一个元素至少要多大才仍是 E4M3 正规数（$\ge2^{-6}$）？两种缩放各算一次。（c）按每元素 1 字节的 E4M3 计，FP32 缩放与 UE8M0 缩放各占多少额外存储？

::: answer 看答案
（a）$a=100$：$s_{\text{FP32}}=100/448=0.2232$；$\log_2(0.2232)=-2.16$，向上取整得 $-2$，$s_{\text{UE8M0}}=2^{-2}=0.25$；最大码值 $100/0.25=400$，$u=400/448=0.893$。$a=0.05$：$s_{\text{FP32}}=1.116\times10^{-4}$；$\log_2(1.116\times10^{-4})=-13.13$，向上取整得 $-13$，$s=2^{-13}=1.221\times10^{-4}$；最大码值 $0.05/2^{-13}=409.6$，$u=0.914$。

（b）元素 $x$ 缩放后是 $x/s$，要求 $x/s\ge2^{-6}$，即 $x\ge s\cdot2^{-6}$。FP32：$0.2232/64=0.00349$；UE8M0：$0.25/64=0.00391$。门槛高了 $1/u=1.12$ 倍，与 §3 的推导一致。

（c）FP32：$4/128=3.125\%$；UE8M0（1 字节）：$1/128=0.78\%$。注意发布的检查点里缩放因子仍以 FP32 存储，1 字节的打包发生在 SM100 的内核里。
:::
::::

:::: quiz 6. 优势的两种口径
$G=8$，只有第 1 个回答答对：奖励 $[1,0,0,0,0,0,0,0]$。（a）按 V3.2 (15.11) 与按 DeepSeekMath（再除以总体标准差）分别求优势。（b）答对者的优势，两种口径相差几倍？与正文 $G=4$ 的 2.31 倍比较，说明趋势。（c）用“留一基线”验证 V3.2 口径下答对者的优势。

::: answer 看答案
（a）均值 $0.125$。V3.2：答对者 $+0.875$，其余 $-0.125$。总体标准差 $\sqrt{0.125\times0.875}=0.3307$，DeepSeekMath：答对者 $0.875/0.3307=+2.646$，其余 $-0.125/0.3307=-0.378$。

（b）$2.646/0.875=3.02$ 倍。$G=4$、一对三错时是 2.31 倍。一般地，二值奖励、答对比例为 $p$ 时，总体标准差是 $\sqrt{p(1-p)}$，两种口径之比就是 $1/\sqrt{p(1-p)}$（$p=\tfrac18$ 时 3.02，$p=\tfrac14$ 时 2.31，$p=\tfrac12$ 时 2）。$\sqrt{p(1-p)}$ 在 $p<\tfrac12$ 时随 $p$ 减小而减小，所以答对的越少，除以它放大得越多（它关于 $p=\tfrac12$ 对称，“只有少数答错”的题同样被放大）：DeepSeekMath 口径会越来越重地奖励“难题里罕见的正确回答”，V3.2 口径不会。

（c）其余 7 个回答的平均奖励 $b_{-1}=0$，$\frac{G-1}{G}(r_1-b_{-1})=\frac78\times1=0.875$，与 (a) 一致。
:::
::::

:::: quiz 7. KL 项的梯度权重
某个 token：$\pi_{ref}=0.2$，$\pi_\theta=0.02$，采样时 $\pi_{\theta_{old}}=0.025$。（a）求 $w$、$k_3$、$\rho$ 与 V3.2 的估计值 $\rho k_3$。（b）求两种估计量对 $\log\pi_\theta$ 的梯度权重，相差几倍？（c）$\beta=0.04$ 时，两者在梯度系数里各贡献多少？与优势 0.5 比较。

::: answer 看答案
（a）$w=0.2/0.02=10$；$k_3=10-1-\ln10=6.697$；$\rho=0.02/0.025=0.8$；$\rho k_3=5.358$。

（b）原始：$1-w=-9$。修正：$\rho\log(\pi_\theta/\pi_{ref})=0.8\times\ln0.1=-1.842$。相差 $9/1.842=4.89$ 倍。两者都为负：梯度下降会提高这个 token 的概率，把策略拉回参考模型；原始 k3 拉得更猛。

（c）原始 $0.04\times9=0.36$，修正 $0.04\times1.842=0.074$。前者已是优势 0.5 的七成，后者约为 15%。
:::
::::

:::: quiz 8. off-policy 掩码
同一道题的两条负样本（$\hat A=-0.5$），各 4 个 token，采样概率都是 $\pi_{\theta_{old}}=[0.8,0.5,0.9,0.6]$；当前策略：o_a 为 $[0.7,0.42,0.9,0.5]$，o_b 为 $[0.8,0.45,0.85,0.6]$。（a）求 $\bar D_a$ 与 $\bar D_b$。（b）$\delta=0.1$ 时哪条被掩？（c）若不掩，o_a 在 $\varepsilon=0.2$ 下有几个 token 仍有梯度？（d）如果 o_a 的优势是 $+0.5$ 呢？

::: answer 看答案
（a）o_a：$\ln\frac{0.8}{0.7}=0.1335$，$\ln\frac{0.5}{0.42}=0.1744$，$0$，$\ln\frac{0.6}{0.5}=0.1823$，和 0.4902，$\bar D_a=0.1226$。o_b：$0$，$\ln\frac{0.5}{0.45}=0.1054$，$\ln\frac{0.9}{0.85}=0.0572$，$0$，和 0.1625，$\bar D_b=0.0406$。

（b）$0.1226>0.1$ 且优势为负，o_a 被掩；$0.0406<0.1$，o_b 保留。

（c）o_a 的 $\rho=[0.875,0.840,1.000,0.833]$，全部大于 $1-\varepsilon=0.8$，按 (15.16) 四项都取 $\rho\hat A$，**4 个 token 全有梯度**：裁剪一个也拦不住，只有掩码能拦。

（d）正优势的回答永远不掩，$m=1$。
:::
::::

:::: quiz 9. Keep Sampling Mask
$\pi_{\theta_{old}}=[0.6,0.25,0.1,0.05]$（token a–d），top-p $=0.9$。当前 $\pi_\theta=[0.55,0.3,0.08,0.07]$。（a）求采样时的核与截断后的采样分布。（b）$\pi_\theta$ 有多少概率落在核外？（c）求 token b 的重要性比：分子用全词表的 $\pi_\theta$；分子用施加同一掩码并重新归一化的 $\pi_\theta$。

::: answer 看答案
（a）按概率从大到小累加：$0.6$、$0.85<0.9$、$0.95\ge0.9$，核是 $\{a,b,c\}$。截断分布 $[0.6,0.25,0.1]/0.95=[0.6316,0.2632,0.1053,0]$。

（b）d 的 0.07。

（c）全词表：$0.3/0.2632=1.140$。同一掩码并重新归一化：$\pi^{\mathcal S}_\theta(b)=0.3/0.93=0.3226$，比率 $0.3226/0.2632=1.226$。两者差约 7.5%，来自那 0.07 的核外概率；论文没有说明 V3.2 用哪一种。
:::
::::

:::: quiz 10. 诚实的自评
DeepSeekMath-V2 的生成器奖励 $R=R_{\text{format}}\cdot(0.76R_Y+0.24R_Z)$，$R_Z=(1-\abs{s'-s})\cdot R_{\text{meta}}(Z)$，设 $R_{\text{format}}=1$。（a）证明的真实分 $s=0.5$，生成器却自评 $s'=0$（过度谦虚），$R_{\text{meta}}=1$，求 $R$。（b）$s=1$、$s'=1$，但自评里指出的问题有一半是编的，元验证分 $R_{\text{meta}}=0.5$，求 $R$。（c）对固定的 $s$ 与 $R_{\text{meta}}$，$s'$ 取什么值 $R$ 最大？

::: answer 看答案
（a）$R_Z=1-0.5=0.5$，$R=0.76\times0.5+0.24\times0.5=0.50$，与夸大成 1 分时一样低，比诚实自评的 0.62 低。

（b）$R_Z=1\times0.5=0.5$，$R=0.76+0.12=0.88$。分数自评对了，但编造问题同样扣分：这正是元验证要堵的漏洞。

（c）$R$ 中只有 $R_Z$ 依赖 $s'$，而 $1-\abs{s'-s}$ 在 $s'=s$ 时取最大值 1。所以最优策略是**如实**自评，既不夸大也不过谦。
:::
::::

## 推导题

::::: quiz 11. 重要性修正的 k3
设 $x\sim\pi_{\theta_{old}}$，$\rho=\pi_\theta/\pi_{\theta_{old}}$，$w=\pi_{ref}/\pi_\theta$，$k_3=w-1-\log w$。（a）证明 $\E_{\pi_{\theta_{old}}}[\rho\,k_3]=\KL[\pi_\theta\Vert\pi_{ref}]$，并写出这一步需要的条件。（b）证明 $\E_{\pi_{\theta_{old}}}\big[\nabla_\theta(\rho k_3)\big]=\nabla_\theta\KL[\pi_\theta\Vert\pi_{ref}]$。（c）证明在 $\pi_{\theta_{old}}=\pi_\theta$ 时 $\E[\nabla_\theta k_3]=\nabla_\theta\KL[\pi_{ref}\Vert\pi_\theta]$。

:::: answer 看答案
（a）换测度，再用两个分布各自求和为 1：

::: deriv 值无偏
$\E_{\pi_{\theta_{old}}}[\rho\,k_3]=\sum_x\pi_{\theta_{old}}(x)\,\frac{\pi_\theta(x)}{\pi_{\theta_{old}}(x)}\,k_3(x)$ || 期望的定义；要求 $\pi_\theta(x)>0$ 处都有 $\pi_{\theta_{old}}(x)>0$
$=\sum_x\pi_\theta\,(w-1-\log w)$ || 约分
$=\sum_x\pi_{ref}-\sum_x\pi_\theta+\sum_x\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}$ || $\pi_\theta w=\pi_{ref}$；$-\log w=\log\frac{\pi_\theta}{\pi_{ref}}$
$=1-1+\KL[\pi_\theta\Vert\pi_{ref}]$ || 两个分布各自求和为 1；KL 的定义
:::

条件就是第一行的那一条：凡 $\pi_\theta(x)>0$ 的 $x$，都要有 $\pi_{\theta_{old}}(x)>0$，否则那些 $x$ 根本采不到，约分不成立。top-p 截断正好会破坏这个条件，这也是 Keep Sampling Mask 的理由之一。

（b）先求单个样本的梯度，再取期望，最后与真 KL 的梯度对照。记 $g=\nabla_\theta\log\pi_\theta(x)$：

::: deriv 期望梯度指向正确的 KL
$\nabla\rho=\rho\,g$ || $\rho\propto\pi_\theta$（$\pi_{\theta_{old}}$ 是常数）；对数导数技巧
$\nabla k_3=\nabla w-\nabla\log w=-w\,g+g=(1-w)\,g$ || $w\propto1/\pi_\theta$，$\nabla\log w=-g$
$\nabla(\rho k_3)=\rho\,g\,(k_3+1-w)=\rho\,(-\log w)\,g$ || 乘积法则；$k_3+1-w=-\log w$
$\E_{\pi_{\theta_{old}}}\big[\nabla(\rho k_3)\big]=\sum_x\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}\,\nabla\log\pi_\theta$ || 换测度，同 (a) 第一行
$\nabla\KL[\pi_\theta\Vert\pi_{ref}]=\sum_x\nabla\pi_\theta\,\log\frac{\pi_\theta}{\pi_{ref}}+\sum_x\pi_\theta\,\nabla\log\pi_\theta$ || 对 $\sum_x\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}$ 用乘积法则
$=\sum_x\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}\,\nabla\log\pi_\theta+\nabla\textstyle\sum_x\pi_\theta$ || $\nabla\pi_\theta=\pi_\theta\nabla\log\pi_\theta$，两项都用
$=\sum_x\pi_\theta\log\frac{\pi_\theta}{\pi_{ref}}\,\nabla\log\pi_\theta$ || $\sum_x\pi_\theta=1$ 是常数，梯度为 0；与第三行相同
:::

（c）$\pi_{\theta_{old}}=\pi_\theta$ 时 $\rho\equiv1$，只剩原始 k3 的梯度 $(1-w)g$：

::: deriv 原始 k3 的期望梯度是反方向的 KL
$\E_{\pi_\theta}\big[(1-w)\,g\big]=\sum_x\pi_\theta\,\nabla\log\pi_\theta-\sum_x\pi_{ref}\,\nabla\log\pi_\theta$ || 展开；$\pi_\theta w=\pi_{ref}$
$=-\sum_x\pi_{ref}\,\nabla\log\pi_\theta$ || 第一项 $=\nabla\sum_x\pi_\theta=0$
$\nabla\KL[\pi_{ref}\Vert\pi_\theta]=-\sum_x\pi_{ref}\,\nabla\log\pi_\theta$ || $\KL[\pi_{ref}\Vert\pi_\theta]=\sum_x\pi_{ref}(\log\pi_{ref}-\log\pi_\theta)$，只有 $\log\pi_\theta$ 依赖 $\theta$
:::

第二行与第三行相同。方向与 (b) 相反：期望梯度约束的是 $\KL[\pi_{ref}\Vert\pi_\theta]$，这就是 Tang 与 Munos 指出的“两个分布的位置反了”。
::::
:::::

:::: quiz 12. 为什么向上取整
UE8M0 缩放因子只能是 2 的整数次幂。（a）若取 $s=2^{\lceil\log_2(a/448)\rceil}$，证明最大码值 $a/s\in(224,448]$。（b）若改为“取最近的 2 的幂”，$s=2^{\mathrm{round}(\log_2(a/448))}$，最大码值落在什么范围？什么情况下会溢出？（c）设 $\log_2(a/448)$ 的小数部分在 $[0,1)$ 上均匀分布，求 (a) 的平均利用率，以及 (b) 的溢出概率。

::: answer 看答案
（a）记 $y=\log_2(a/448)$。$\lceil y\rceil\ge y$ 给出 $s\ge a/448$，即 $a/s\le448$；$\lceil y\rceil<y+1$ 给出 $s<2a/448$，即 $a/s>224$。

（b）$\abs{\mathrm{round}(y)-y}\le\tfrac12$，所以 $a/s=448\cdot2^{\,y-\mathrm{round}(y)}\in[448/\sqrt2,\ 448\sqrt2]\approx[316.8,\ 633.6]$。只要 $y$ 被向下舍入（$\mathrm{round}(y)<y$），$a/s>448$，最大的元素就超出 E4M3 的范围，被饱和或变成 NaN。

（c）(a)：$u=2^{-v}$，$v=\lceil y\rceil-y\sim\mathrm{Uniform}[0,1)$，$\E[u]=\int_0^12^{-v}dv=\frac{1}{2\ln2}\approx0.721$。(b)：小数部分落在 $(0,\tfrac12)$ 时 $y$ 被向下舍入，概率 $\tfrac12$：**一半的分块会溢出**。所以“向上取整”不是为了好看：凡是 $\ge a/448$ 的 2 的幂都不会溢出，$2^{\lceil y\rceil}$ 是其中最小的一个，也就是不溢出前提下码值最大、精度最好的那一档；代价就是平均只用到 E4M3 范围的约 72%。
:::
::::

```refs
- id: imo2025
  kind: artefact
  title: "66th IMO 2025 – Individual Results"
  a: International Mathematical Olympiad (2025)
  url: https://www.imo-official.org/results/individual/year/2025/
  finding: 2025 年金牌 72 人，最低 35 分；银牌 28–34 分，铜牌 19–27 分。
  matters: 说明 Speciale 与 DeepSeekMath-V2 的 35/42 恰好压在金牌线上。
- id: ioi2025
  kind: artefact
  title: "IOI 2025 Statistics and Results"
  a: International Olympiad in Informatics (2025)
  url: https://stats.ioinformatics.org/olympiads/2025
  finding: 金牌线 438.30 分（28 枚），人类第 9、10 名为 502.78 与 491.65。
  matters: 用来复核 V3.2 表 4“492/600、第 10 名”的说法。
- id: icpc2025
  kind: artefact
  title: "49th ICPC World Finals Baku scoreboard"
  a: ICPC (2025)
  url: https://worldfinals.icpc.global/scoreboard/2025/finals/
  finding: 12 题，第 1 名解 11 题，第 2、3 名各解 10 题，金牌 4 队。
  matters: 说明 Speciale 的 10/12 与“第 2 名”之间还隔着一个未公开的罚时比较。
- id: ds-thinking-guide
  kind: artefact
  title: "DeepSeek API guide \"Thinking Mode\" (Wayback 2025-12-09 snapshot vs live 2026-09)"
  a: DeepSeek (2025–2026), API docs
  url: http://web.archive.org/web/20251209045516/https://api-docs.deepseek.com/guides/thinking_mode
  finding: V3.2 时代要求新问题开始时删除 reasoning_content；现行版改为带 tools 时回传全部历史思考。
  matters: 用来防止把 V4 时代的 API 行为错套到 V3.2 上。
- id: gspo
  title: "Group Sequence Policy Optimization"
  a: Zheng et al. (2025), preprint. arXiv:2507.18071v2
  url: https://arxiv.org/abs/2507.18071
  finding: 报告 MoE 在一次 RL 更新后约 10% 激活专家发生变化，GRPO 需要 Routing Replay，GSPO 用序列级比率免去它。
  matters: 说明 Keep Routing 解决的问题有多大，以及它不是唯一解法。
- id: dsmath
  title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
  a: Shao et al. (2024), preprint. arXiv:2402.03300v3
  url: https://arxiv.org/abs/2402.03300
  finding: 提出 GRPO，优势取组内标准化奖励（减均值再除标准差），并用 Schulman 的 k3 估计 KL。
  matters: 本课逐项对比 V3.2 对 GRPO 的改动（优势不除标准差、k3 加重要性修正）时的原始版本。
- id: tang2025kl
  title: "On a few pitfalls in KL divergence gradient estimation for RL"
  a: Tang, Munos (2025), preprint. arXiv:2506.09477v1
  url: https://arxiv.org/abs/2506.09477
  finding: 证明把 k3 当损失直接求导（在策略采样时），期望梯度是 $\nabla\KL[\pi_{ref}\Vert\pi_\theta]$ 而非 $\nabla\KL[\pi_\theta\Vert\pi_{ref}]$。
  matters: 为 V3.2“无偏 KL 估计”的动机提供独立的数学依据，课文推导梯度方向时引它。
- id: schulman-kl
  kind: artefact
  title: "Approximating KL Divergence"
  a: Schulman (2020), blog
  url: http://joschu.net/blog/kl-approx.html
  finding: 提出 k1、k2、k3 三种单样本 KL 估计，其中 k3 = (r − 1) − log r 无偏且恒非负。
  matters: DeepSeekMath 与 V3.2 都以“Schulman, 2020”引用它，是 k3 的出处。
- id: dsv32
  title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
  a: DeepSeek-AI (2025), preprint. arXiv:2512.02556v1
  url: https://arxiv.org/abs/2512.02556
  finding: 给出 V3.2 的 DSA 续训、专家蒸馏 + 单阶段混合 GRPO，以及无偏 KL、off-policy 序列掩码、Keep Routing、Keep Sampling Mask 四项稳定化改动和大规模智能体任务合成。
  matters: 本课几乎所有 RL 公式、智能体任务数字和 Speciale 竞赛成绩都以它为唯一一手来源。
- id: dsmathv2
  title: "DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning"
  a: Shao, Luo, Lu, Ren, Hu, Ye, Gou, Ma, Zhang (2025), preprint. arXiv:2511.22570v1
  url: https://arxiv.org/abs/2511.22570
  finding: 训练能给出问题清单与三档分数的证明验证器，加上元验证与“证明 + 自评”加权奖励，得到 IMO 2025 5/6、Putnam 2024 118/120 的证明生成器。
  matters: 它是 Speciale 证明能力的来源（V3.2 称并入其数据与奖励），也是本课“结果奖励之外如何给奖励”的例子。
- id: dsv3
  title: "DeepSeek-V3 Technical Report"
  a: DeepSeek-AI (2024), preprint. arXiv:2412.19437v2
  url: https://arxiv.org/abs/2412.19437
  finding: V3 的两阶段上下文扩展各 1000 步（32K×1920、128K×480），后训练仅 5K GPU 小时。
  matters: 是 V3.1“10 倍 / 3.3 倍”续训和 V3.2“后训练超过 10%”的比较基线。
- id: mx2023
  title: "Microscaling Data Formats for Deep Learning"
  a: Rouhani et al. (2023), preprint. arXiv:2310.10537v3
  url: https://arxiv.org/abs/2310.10537
  finding: 定义 MX 块格式：$k=32$ 个元素共享一个 E8M0（纯指数）缩放。
  matters: 解释 V3.1“兼容 microscaling”指什么，也是指出 DeepSeek 块大小仍为 128、取整方向不同的对照依据。
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
  matters: 是 V3.2 的直接前身（V3.2 称唯一架构差别是 DSA），其 o_proj 声明是本课检查点实测的起点。
- id: dsv32exp
  kind: artefact
  title: "DeepSeek-V3.2-Exp technical report (DeepSeek_V3_2.pdf)"
  a: DeepSeek-AI (2025), GitHub
  url: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
  finding: V3.2-Exp 沿用 V3.1-Terminus 的后训练流水线，专家领域为五个，并说明单阶段混合 RL 取代多阶段 RL。
  matters: 用来标出 V3.2 论文与其前身在专家领域数量上的版本漂移，并提供 Terminus 的基准分数。
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
  matters: 课文定义 UE8M0 时引用的一手硬件文档（OCP 规范原文未能访问）。
- id: cuda-e8m0
  kind: artefact
  title: "CUDA Math API Reference, __nv_fp8_e8m0 struct"
  a: NVIDIA (2025), documentation
  url: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e8m0.html
  finding: e8m0 的偏置为 127，码值 0–254 表示 $2^{-127}$ 到 $2^{127}$，0xFF 保留为 NaN。
  matters: 给出 PTX 文档没有写明的指数偏置，使 (15.5) 有一手出处。
- id: ckpt-scan
  kind: artefact
  title: "Author's scan of DeepSeek V3 / V3.1 / V3.1-Terminus / V3.2 safetensors scale tensors"
  a: 本课研究笔记 (2026-09-24，2026-09-25 补测)
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main
  finding: 在抽查的第 0、30、60 层上，V3.1、V3.1-Terminus、V3.2 除 o_proj 外的 FP8 权重缩放均为 2 的幂，o_proj 一个也不是；V3 全部不是；缩放张量一律以 F32 存储。
  matters: 把模型卡的 UE8M0 说法落到可复现的检查点事实上，并暴露 o_proj 例外持续到 V3.2。
- id: qq-ue8m0
  kind: artefact
  title: "腾讯新闻（转载扬子晚报）关于 UE8M0 FP8 的报道"
  a: 腾讯新闻 (2025-08-21)，二手报道
  url: https://news.qq.com/rain/a/20250821A07GD100
  finding: 转述 DeepSeek 称 UE8M0 FP8 针对下一代国产芯片设计。
  matters: 只作为“流传说法的出处”引用；一手原文未能读到，课文不据此下结论。
```
