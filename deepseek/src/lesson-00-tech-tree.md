---
title: 第 0 课 · 科技树总览：从 GPT 到 DeepSeek-V4.1
short: DeepSeek 科技树
kicker: DeepSeek 科技树 · 第 0 课
description: 整门课的地图：一张从 2017 年到 2026 年 9 月的科技树（六条主线，每个节点标出日期和讲它的那一课）；每条主线一页，讲它要压低的账单、一步步的改进和一张按真实数字计算的图；主线模型规格总表（每一格注明出处）；以及怎么读这门课：推荐顺序、数学补课对照表、图的颜色、八个贯穿全课的类比，然后是五个贯穿全课的问题，最后是九道带完整答案的练习。
lede: 从 2017 年 6 月的 Transformer 到 2026 年 9 月的 DeepSeek-V4.1-Flash，一共 3,377 天。这一课不推导任何机制，只做三件事：把这段历史画成一棵按时间展开的**科技树**，看清每个节点在对付哪一张账单；把 DeepSeek 主线模型的规格排成一张**总表**，每一格都注明出处；再告诉你这门课**怎么读**：先读哪一课，哪一课要补哪一节数学，图里的颜色和反复出现的类比各是什么意思。
meta: 预计阅读 155 分钟，练习另需约 40 分钟 · 前置：会矩阵乘法、softmax 和梯度下降即可；本课只给结论与链接，推导都在后面各课
index: index.html
next: lesson-01-transformer-anatomy.html | 第 1 课 · Transformer 解剖
---

## 这一课回答什么问题

这门课一共 21 课，讲的是一件事：一个标准的 GPT 怎样一步一步变成 DeepSeek-V4.1-Flash。一步一步地读，很容易只记住一串缩写：MLA、GRPO、DSA、mHC、CSA……读完这一课，你手里应该有一张地图，能回答下面这些问题：

- 从 GPT 到 DeepSeek-V4.1-Flash，中间发生了哪些事？哪些是 DeepSeek 自己提出的，哪些是它继承来的？
- 为什么说整门课只有**六条主线**？每一条线在压低哪一张账单？
- 每一代 DeepSeek 主线模型有多少层、多宽、每个 token 缓存多少字节、训练了多少 token？这些数字从哪里来，哪些是论文原话，哪些是推算？
- 21 课应该按什么顺序读？读到哪一课，需要先补哪一节数学？
- 图里的颜色、文中反复出现的“会议纪要本”“医院”“传送带”，各指什么？

## 这一课在科技树上的位置

这一课是整棵树的**俯视图**，没有上一课。后面每一课都从这里的某个节点出发，把它放大成几十页的推导；最后的[第 20 课](lesson-20-retrospective.html)再回到这张地图，按主线把全部材料重新切一遍，列出被放弃的路线和仍然开放的问题。两课共用同一份研究笔记（`reference-docs/lesson-00-doc.md`），所以这里的每个日期、每个数字，在第 20 课里都对得上。

读这一课不需要先懂任何机制。遇到 MLA、GRPO 这类名字，只要先记住“它在第几条主线上、压的是哪一笔账”，细节等读到那一课再说。

## 读完你能……

1. 说出六条主线各自要压低的是哪一种资源，并各举出两个节点和讲它们的课。
2. 在科技树上找到任何一个 DeepSeek 模型，说出它的发布日期、它在哪几条主线上改了东西。
3. 用每 token KV 字节的四因子式 (0.1) 复算 389,120、69,120、48,068、3,514、890 这五个数中的任意一个。
4. 读懂规格总表的每一格：它来自论文、配置文件还是复算，“128K”“1M”“552B”“8B / 16B”这类数字各自有什么限定。
5. 为自己选一条阅读路线，并知道每一课需要先补哪几节数学。

::: note 本课的日期、数字与记号约定
- **两种日期。** 模型写**公开发布日**（权重或服务第一次对公众可用的那天，按 UTC），论文写 arXiv **第一版（v1）的提交日**。两者常常差几天：V2 在 2024-05-06 发布，论文 v1 是 05-07；V4 在 2026-04-24 发布，论文 04-26 才提交，编号却是 2606.19348 [@dsv2; @dsv4]。Hugging Face（HF）仓库的创建时间只是发布日的下界，不当发布日用。
- **单位。** 参数量 B = $10^9$、T = $10^{12}$。字节：GB = $10^9$ 字节，GiB = $2^{30}$ 字节。“1M 上下文”指 $2^{20}=1{,}048{,}576$ 个 token。
- **记号**沿用全课的符号表：$l$ 层数，$n_h$ 注意力头数，$d_h$ 每头维度，$n_g$ GQA 的 KV 组数，$d_c$ MLA 的 KV 压缩维度，$d_h^R$ 解耦 RoPE 的每头维度，$N$、$D$、$C$ 分别是参数量、训练 token 数、训练算力。
- **事实来源。** 本课的每个日期和数字都来自研究笔记或已经写完的第 1–19 课，不凭记忆。推导留给各课，这里只给结论和链接；“笔者推算”“笔者推测”会明说。
:::

[[legend]]

[[toc]]

# 一张科技树 | 时间轴、六条主线、每个节点在哪一课 | 15

这一页先说清楚为什么要把这段历史画成“树”，再给出整张图和读图的三种方法，最后用一张表把六条主线各自对付的账单说完。后面六页（[§2](#p2)–[§7](#p7)）每页放大一条主线。

## 问题：3,377 天里发生了什么

先把时间摆出来。GPT 这一侧的六个日子都有一手来源：Transformer 的论文在 2017-06-12 提交 arXiv [@vaswani2017]；GPT-1 在 2018-06-11 公告，没有 arXiv 版 [@radford2018]；GPT-2 在 2019-02-14 公告 [@radford2019]；GPT-3 的论文在 2020-05-28 提交 [@brown2020]；InstructGPT 的公告与论文 PDF 在 2022-01-27 发出，arXiv 版晚到 2022-03-04 [@ouyang2022]；ChatGPT 在 2022-11-30 上线 [@oaiwb]。从 Transformer 到 ChatGPT 是 1,997 天。

DeepSeek 这一侧，最早公开的语言模型是代码模型 DeepSeek-Coder：它没有官方公告页，GitHub 仓库的第一次提交是 2023-11-02，本课就用这一天作代理日期 [@dscoder; @ghorg]。第一代通用模型 DeepSeek LLM 7B/67B 在 2023-11-29 发布，离 ChatGPT 上线 364 天 [@dsllm]。从它到 2026-09-10 发布的 V4.1-Flash 是 1,016 天 [@dsv41; @dsnews]。研究笔记的发布总表从 DeepSeek-Coder 一直记到 2026-09-19 的 DSec 论文，一共 50 个条目：主线模型、旁支模型、论文和开源制品。

这么多条目，按时间排成一串，看到的只是一份清单。把它们按**要压低的那一笔账**分组，清单就变成了几条线：同一条线上，后一个节点几乎总是在前一个节点的账单上再打一个折。这就是“科技树”的意思：节点是一次改进，枝干是“在它的基础上”。

主干模型的发布节奏如下（天数由日期相减得到）：

| 区间（后一个的发布日） | 天数 | 这段时间主要动了哪几条线 |
|---|---|---|
| DeepSeek LLM（<span style="white-space:nowrap">2023-11-29</span>）→ V2（<span style="white-space:nowrap">2024-05-06</span>） | 159 | A（MLA）、B（DeepSeekMoE 放大）、D（GRPO） |
| V2 → V3（<span style="white-space:nowrap">2024-12-26</span>） | 234 | B（无辅助损失均衡）、C（FP8、DualPipe）、D（MTP） |
| V3 → R1（<span style="white-space:nowrap">2025-01-20</span>） | 25 | D（纯强化学习激发推理） |
| R1 → V3.2-Exp（<span style="white-space:nowrap">2025-09-29</span>） | 252 | C（UE8M0）、D（一模两式），最后 A（DSA） |
| V3.2-Exp → V3.2（<span style="white-space:nowrap">2025-12-01</span>） | 63 | D（规模化 RL、智能体） |
| V3.2 → V4 预览版（<span style="white-space:nowrap">2026-04-24</span>） | 144 | 六条线全部 |
| V4 预览版 → V4-Pro 正式版（<span style="white-space:nowrap">2026-08-13</span>） | 111 | D（重做后训练、DSpark 投机解码） |
| V4-Pro 正式版 → V4.1-Flash（<span style="white-space:nowrap">2026-09-10</span>） | 28 | A、B、C、E |

V3 到 R1 只隔 25 天，因为 R1 没有训练新的基座，它是在 V3 的基座上做强化学习得到的 [@dsr1]。

## 整张图

::: fig figures/l00-tech-tree.svg | 图 0.1 DeepSeek 科技树：2017 年到 2026 年 9 月，六条主线上的每一个节点 | wide scroll
**先横着看每一条色带：一条色带是一条主线，卡片从左到右排成这条线上的一次次改进；再竖着看虚线：每条虚线是一个 DeepSeek 主干模型的发布日，它同时穿过好几条主线。** 卡片的**左边缘**对准日期（模型取公开发布日，论文取 arXiv v1 日期）；圆圈里的数字是讲它的那一课，0 表示只在本课提到；粗框是 DeepSeek 主干模型及其关键创新。2017 年 1 月到 2023 年 9 月的刻度是压缩的，2023 年 10 月起同样一个月画宽约 6.7 倍。模型行里的细箭头是**权重血缘**（在前者的检查点上继续训练）：InstructGPT 在预训练好的 GPT-3 上继续训练 [@ouyang2022]；DeepSeekMath 7B 从 DeepSeek-Coder-v1.5 7B 续训，后者又从 DeepSeek LLM 7B 续训（这一条图上没有画，中间隔着别的卡片）[@dsmath; @dscoder]；R1 与 V3.1 建在 V3 的基座上；V3.2-Exp 与 V3.2 都从 V3.1 的更新版 V3.1-Terminus 续训 [@dsv32]；V4.1-Flash 从头训练，所以没有箭头指向它 [@dsv41]。模型行里 V2、V3 等省略了 “DeepSeek-” 前缀。全部日期取自研究笔记，由 `l00_tech_tree.py` 从数据表生成；图很宽，电脑和手机上都可以左右滑动。
:::

## 读图的三种方法

**横着读：一条线的演进。** 以主线 A 为例：多头注意力 MHA（2017）→ MQA（2019）→ GQA（2023）→ DeepSeek LLM 67B 用 GQA-8 → V2 的 MLA → NSA → V3.2 的 DSA → V4 的 CSA/HCA → V4.1 的 CSA2。[§2](#p2) 会说明，这一串改进每一步都动了同一个乘积里的一个因子。

**竖着读：一次发布动了哪几条线。** V2 同时动了 A（MLA）、B（设备受限路由）和 D（两阶段 GRPO 对齐）；V3 动了 B（偏置均衡、节点受限路由）、C（FP8 训练、DualPipe）和 D（MTP）；V4 在六条线上都有卡片；R1 只动了 D。一个模型的技术报告读起来像一堆互不相干的改动，放回主线上，每一处改动都有自己的前因。

**看空白。** 主线 E 从 2020 年到 2024 年 9 月是空的：残差连接那条“系数恒为 1 的捷径”，从 GPT-2 把 LayerNorm 挪进分支之后，在主流模型里几年没人动过，直到 ByteDance 的 Hyper-Connections [@zhu2025hc]。主线 F 上，DeepSeek 自己的卡片在 2024 年 1 月的超参数规模化规律之后就空了两年多，中间只有别人的 Muon 博客（2024 年 12 月），直到 2026 年 4 月的 V4 才换掉优化器。

还要分清**继承的**和**自己提出的**。图里有不少卡片不是 DeepSeek 的工作：MQA、GQA、RoPE、YaRN、GShard、Switch、PPO、DPO、Hyper-Connections、并行多 token 预测、Muon 都来自别的团队。DeepSeek 自己提出的主要是 DeepSeekMoE、GRPO、MLA、无辅助损失均衡、FP8 细粒度训练与 DualPipe、NSA/DSA、mHC、Engram、CSA/HCA、CSA2。一个好的读法是：每看到一个 DeepSeek 自己的节点，就找它左边那个继承来的节点，问一句“它嫌前者哪里不够”。

## 六条主线，各自对付一张账单

| 主线（点开看本课那一页） | 要压低的账单 | 主要节点 | 主要课次 |
|---|---|---|---|
| [**A** 注意力与 KV cache](#p2) | 每个 token 缓存多少字节；解码每一步要读多少字节、算多少次打分 | MHA → MQA/GQA → MLA → NSA → DSA → CSA/HCA → CSA2 + FP4 KV | [1](lesson-01-transformer-anatomy.html)、[4](lesson-04-inference-kv-cache.html)、[5](lesson-05-modern-block.html)、[9](lesson-09-v2-mla.html)、[14](lesson-14-sparse-attention.html)、[18](lesson-18-v4.html)、[19](lesson-19-v4-1-flash.html) |
| [**B** 前馈层的稀疏化（MoE）](#p3) | 参数量与每 token 计算量绑在一起 | 稠密 FFN → GShard/Switch → DeepSeekMoE → 设备受限路由 → 无辅助损失均衡 → V4、V4.1 | [7](lesson-07-deepseekmoe.html)、[10](lesson-10-v2-moe-training.html)、[11](lesson-11-v3-architecture.html)、[18](lesson-18-v4.html) |
| [**C** 数值与系统](#p4) | 训练算力、显存、跨卡跨节点通信 | BF16 → FP8 → UE8M0 → FP4；DualPipe、专家并行、PD 分离 | [12](lesson-12-v3-infrastructure.html)、[15](lesson-15-v3-2-agents.html)、[18](lesson-18-v4.html)、[19](lesson-19-v4-1-flash.html) |
| [**D** 训练目标与后训练](#p5) | 训练信号：从“像语料”到“做得对”，以及它的算力与稳定性 | 下一词预测 → MTP；SFT → RLHF/PPO → DPO → GRPO → R1 → V3.2 → V4 | [2](lesson-02-gpt-training.html)、[3](lesson-03-alignment-rlhf.html)、[8](lesson-08-deepseekmath-grpo.html)、[11](lesson-11-v3-architecture.html)、[13](lesson-13-r1-reasoning.html)、[15](lesson-15-v3-2-agents.html)、[18](lesson-18-v4.html) |
| [**E** 残差流与记忆](#p6) | 深度方向的信号尺度；知识和上下文记在哪里 | 残差 → Pre-LN → HC → mHC；Engram；视觉压缩 | [1](lesson-01-transformer-anatomy.html)、[16](lesson-16-mhc.html)、[17](lesson-17-engram-ocr.html)、[18](lesson-18-v4.html) |
| [**F** 规模化规律与优化器](#p7) | 一次性大训练的“下注”：超参数、模型与数据的分配、优化器 | Kaplan / Chinchilla → DeepSeek LLM 的超参数规律；AdamW → Muon | [2](lesson-02-gpt-training.html)、[6](lesson-06-deepseek-llm.html)、[18](lesson-18-v4.html) |

这六条线不是互相独立的。最明显的交叉是 A 与 C：V4.1 的 FP4 KV 缓存既是“缓存更小”（A），也是“数值格式更短”（C）。B 与 E 也交叉：Engram 自称是与 MoE 互补的“新稀疏轴”[@engram]，本课把它放在 E（记忆），第 17 课两边都讲。遇到这种节点，图上只放在一条线上，正文会说明。

::: key 本页要点
- 时间：Transformer 2017-06-12 → ChatGPT 2022-11-30（1,997 天）→ DeepSeek LLM 2023-11-29（364 天）→ V4.1-Flash 2026-09-10（1,016 天）。
- 图 0.1：横读是一条主线的演进，竖读是一次发布动了哪几条线，空白处说明哪条线长期没动。
- 六条主线各压一笔账：A 缓存字节与读取量，B 参数与计算的绑定，C 算力、显存与通信，D 训练信号，E 深度方向的信号与记忆，F 大训练的下注。
:::

# 主线 A：注意力与 KV cache | 每个 token 缓存多少字节：从 389,120 到 890 | 20

这一页把主线 A 的七步改进放进一个公式里：每 token 的 KV 字节是四个因子的乘积，每一步改进都在其中一个因子上动手。然后用这个公式复算五个数，最后说清每一步省下的字节由什么来付。

## 问题：解码的每一步都要把缓存读一遍

[第 4 课](lesson-04-inference-kv-cache.html#p2)推导过：生成第 $t$ 个 token 时，每一层都要用当前的查询去和前面全部 $t-1$ 个 token 的键打分、再按权重读它们的值。这些键和值不必重算，可以缓存下来，这就是 **KV cache**；代价是它每个 token 占一份空间，而且解码的每一步都要把整份缓存从显存里读一遍。于是解码的速度主要由**读多少字节**决定，而不是算多少次乘法（[第 4 课 §4](lesson-04-inference-kv-cache.html#p4) 的“带宽墙”）。

这笔账有多大？DeepSeek LLM 67B 每个 token 缓存 389,120 字节。一个 $2^{20}$ token（1M）的请求，缓存就是 $389{,}120\times2^{20}\approx4.08\times10^{11}$ 字节，也就是 408 GB：一张 80 GB 的显卡只放得下 0.2 个这样的请求。主线 A 的全部工作，就是把这个数压下去。

::: ana KV cache = 会议纪要本
**映射。** 每个发言者（token）说完，秘书把他的“标签 + 要点”（键 K、值 V）记在本子上；后来的人只翻本子，不必重听录音。主线 A 的每一步都在让本子变薄或少翻几页：MQA/GQA 让几位发言者共用一栏；MLA 改记速记符号（潜向量），要用时再展开；DSA 先查目录，只精读得分最高的 2048 条；CSA/HCA 把每 4 句或每 128 句合写成一条；CSA2 让 40 层共用 4 本；FP4 用更短的字来写。

**收益。** 它让“省空间”和“省阅读量”分开：DSA 并没有让本子变薄（反而多了一栏目录），只是每次少翻。

**边界。** 真正的纪要是逐字副本，所以 KV cache 本身只省重复计算、不省空间，空间随发言人数线性增长；MLA 的速记、CSA 的合写却是**训练出来的有损压缩**，压缩用的矩阵是学出来的，模型必须在训练中学会读它，不能事后套在一个现成模型上。
:::

## 形式化：每 token 字节的四个因子

把每 token 的缓存拆成四个问题：一共存几份？每几个 token 存一条？每条几个数？每个数几个字节？第 19 课把这四个问题写成一个乘积（[第 19 课 §8](lesson-19-v4-1-flash.html#p8)）：

::: eq 每 token 全局 KV 字节 | (0.1)
$$B_{kv}=\underbrace{n_{\text{copy}}}_{\text{存几份}}\times\underbrace{\frac1m}_{\text{每几个 token 一条}}\times\underbrace{n_{\text{dim}}}_{\text{每条几个数}}\times\underbrace{b}_{\text{每个数几字节}}$$
$B_{kv}$ 只计随上下文线性增长的**全局** KV；滑动窗口（SWA）状态是每个请求的一个常数，不计入。$b$ 要把摊到每个数上的缩放因子也算进去。
:::

四个符号的含义：$n_{\text{copy}}$ 是整模型一共存几份缓存（普通模型每层一份，$n_{\text{copy}}=l$）；$m$ 是压缩率，每 $m$ 个 token 合成一条（不压缩时 $m=1$）；$n_{\text{dim}}$ 是一条缓存里有多少个数；$b$ 是每个数占几个字节（BF16 是 2）。把前几代模型代进去：

::: eq 稠密注意力与 MLA | (0.2)
$$\begin{aligned}\text{MHA / GQA：}\quad B_{kv}&=2\,n_g\,d_h\,l\,b\\ \text{MLA：}\quad B_{kv}&=(d_c+d_h^R)\,l\,b\end{aligned}$$
MHA 取 $n_g=n_h$，MQA 取 $n_g=1$。系数 2 是键和值各一份；MLA 每层只存一个 $d_c$ 维潜向量和一个 $d_h^R$ 维的 RoPE 键，所有头共用 [@dsv2]。
:::

V3.2 以后的缓存用低精度存储，并多了一个给 indexer 用的“索引键”；V4 起还沿序列方向压缩。它们的通式是：

::: eq V3.2：每层一条 FP8 MLA 条目加一个索引键 | (0.3)
$$B_{kv}=l\times(656+132)$$
656 字节 = 512 个 FP8 潜向量分量 + 4 个 FP32 缩放因子（16 字节）+ 64 个 BF16 的 RoPE 分量（128 字节）；132 字节 = 128 维 FP8 索引键 + 4 字节缩放 [@flashmla]。
:::

::: eq V4：CSA 层与 HCA 层分别压缩 | (0.4)
$$B_{kv}=\sum_{\ell\in\text{CSA}}\frac{584+68}{m}+\sum_{\ell\in\text{HCA}}\frac{584}{m'},\qquad m=4,\ m'=128$$
584 字节 = 448 维 FP8 + 64 维 BF16 的 RoPE + 8 字节缩放 [@flashmla]；68 字节是 FP4 的索引键（128 维 × 4 bit + 4 字节缩放），这是笔者由论文与代码推算的值，原文没有直接印出 [@dsv41; @dsv41code]。
:::

::: eq V4.1：只有 Full 层自己存 | (0.5)
$$B_{kv}=\sum_{\ell\in\text{Full}}\frac{288+68}{m_\ell}$$
288 字节 = 512 维 FP4（E2M1）+ 每 16 维一个 E4M3 缩放（32 字节）[@dsv41; @flashmla]。40 层里只有 4 层是 Full 层（三层 $m=2$、一层 $m=1$），其余层借用它们的缓存（[第 19 课 §3](lesson-19-v4-1-flash.html#p3)）。
:::

## 推导：五个数

下面五行把五代模型的架构参数代进式 (0.2)–(0.5)。每一行都可以用计算器复核。

::: deriv 从架构参数到每 token 字节
$2\times8\times128\times95\times2=389{,}120$ || DeepSeek LLM 67B：GQA-8，$d_h=128$，95 层，BF16 [@dsllm]
$(512+64)\times60\times2=69{,}120$ || V2：MLA，$d_c=512$、$d_h^R=64$，60 层，BF16 [@dsv2]
$61\times(656+132)=48{,}068$ || V3.2：61 层，每层一条 FP8 条目加一个索引键 [@dsv32]
$21\times\tfrac{584+68}{4}+20\times\tfrac{584}{128}=3{,}514.25$ || V4-Flash：21 个 CSA 层、20 个 HCA 层，前 2 层纯 SWA 不计 [@dsv4]
$\left(\tfrac12+\tfrac12+\tfrac12+\tfrac11\right)\times(288+68)=890$ || V4.1-Flash：4 个 Full 层，三个 $m=2$、一个 $m=1$ [@dsv41]
:::

最后一行值得多看一眼：四个 Full 层每 token 一共摊到 $\tfrac12\times3+1=2.5$ 条缓存，每条 $288+68=356$ 字节，所以是 $2.5\times356=890$。其余 36 层一个字节也不存。

这五个数里有四个恰好是 V4.1 论文图 1(b) 印出的柱长（389,120、48,068、3,514、890），由此也复现了图上的三段倍数 8.1×、13.7×、3.9× 和总倍数“437 倍”[@dsv41]。V4-Pro 按同一公式是 $30\times\tfrac{652}{4}+31\times\tfrac{584}{128}=5{,}031$ 字节，论文没有印这个数，只说约为 V3.2 的 10%，与之吻合（笔者推算）。

## 数字：每一步省在哪个因子上

| 这一步 | 动了哪个因子 | 每 token 字节 | 倍数 | 详见 |
|---|---|---|---|---|
| 同形状的 MHA → 67B 的 GQA-8 | $n_{\text{dim}}$：64 个 KV 头减到 8 个 | 3,112,960 → 389,120 | ÷8 | [第 6 课 §3](lesson-06-deepseek-llm.html#p3) |
| 67B → V2（MLA） | $n_{\text{dim}}$：每层 2,048 个数 → 576 个；层数 95 → 60 | 389,120 → 69,120 | ÷5.63 | [第 9 课 §8](lesson-09-v2-mla.html#p8) |
| V2 → V3 | 层数 60 → 61 | 69,120 → 70,272 | ×1.02 | [第 11 课 §8](lesson-11-v3-architecture.html#p8) |
| V3 → V3.2（FP8 + 索引键） | $b$：BF16 → FP8；另加索引键 | 70,272 → 48,068 | ÷1.46 | [第 14 课 §7](lesson-14-sparse-attention.html#p7) |
| V3.2 → V4-Flash（CSA/HCA） | $\tfrac1m$：每 4 或 128 个 token 一条；层数 43 | 48,068 → 3,514 | ÷13.7 | [第 18 课 §7](lesson-18-v4.html#p7) |
| V4-Flash → V4.1-Flash（CSA2 + FP4） | $n_{\text{copy}}$：41 份 → 4 份；$b$：FP8 → FP4 | 3,514 → 890 | ÷3.95 | [第 19 课 §8](lesson-19-v4-1-flash.html#p8) |

第一行的 3,112,960 是 $2\times64\times128\times95\times2$，即 67B 若用 64 个 KV 头的 MHA 时的缓存。最后一行的 3.95 倍可以再拆成两步：只换结构（份数与压缩率）不换格式，缓存从 3,514 降到 1,630，是 2.16 倍；再把主 KV 换成 FP4，从 1,630 降到 890，是 1.83 倍（[第 19 课 §8](lesson-19-v4-1-flash.html#p8)）。这里的“不换格式”指沿用 V4 每条 584 字节的条目；若中间那一步改用 V4.1 自己的 FP8 格式（每条 528 字节），就是 $2.5\times(528+68)=1{,}490$ 字节，拆法变成结构 2.36 倍、精度 1.67 倍（研究笔记用的是这一种）。两种拆法都是笔者的分解，结论相同：结构这一步省得比精度多。

::: fig figures/l00-kv-timeline.svg | 图 0.2 每 token 全局 KV 字节随发布时间的变化
**看那条黄褐色虚线：主干模型的每 token KV 在不到三年里降了 437 倍，但不是匀速下降：V2 一次 ÷5.6，V3 → V3.2 只 ÷1.5，V3.2 → V4-Flash 一次 ÷13.7。** 每个点都由生成脚本按式 (0.2)–(0.5) 从架构参数算出，不是从论文图上读的；右轴把字节/token 换算成一个 $2^{20}$ token 请求的 KV（GiB）。红色虚线是 80 GB：在它上方的模型（7B、67B、MoE 16B），一张卡连一个 1M token 请求的 KV 都放不下，V2、V3 刚好落在它下方（V4 之前的模型只支持 4K 到 128K，1M 对它们只是假想）。空心灰点是非主干型号或推算值：DeepSeek LLM 7B 用 MHA，每 token 反而比 67B 还大；V4-Pro 的 5,031 是笔者按同一公式推算。空心黄点是 V2 论文所说部署时平均 6 bit 的 KV [@dsv2]。GPT-3 按每个数 2 字节计 [@brown2020]。由 `l00_kv_timeline.py` 生成。
:::

::: ex 一个 1M token 的请求要多少显存
按 $T=2^{20}$ 计：DeepSeek LLM 67B 是 $389{,}120\times2^{20}$ 字节 = 408 GB（380 GiB），一张 80 GB 的卡只放得下 $80\times10^9/(389{,}120\times2^{20})=0.20$ 个请求。同样算：V3.2 是 50.4 GB，能放 1.59 个；V4-Flash 3.68 GB，21.7 个；V4.1-Flash 0.93 GB，85.7 个。这里只算全局 KV，没算权重和 SWA 状态，所以是上界。

年化看：从 DeepSeek LLM（2023-11-29）到 V4.1-Flash（2026-09-10）共 1,016 天，缩小 437.2 倍，相当于每年 $437.2^{365/1016}\approx8.9$ 倍（笔者口径）。
:::

## 代价：省下的字节由什么来付

主线 A 上没有免费的一步。把每一步的代价列在一起，才能看懂为什么它们要按这个顺序出现：

- **GQA / MQA 付质量。** 让一组查询头共用一对键值：MQA 在原论文的翻译实验里略降（WMT14 BLEU 26.7 → 26.5），还可能训练不稳；GQA-8 在 T5-XXL 上几乎追平 MHA（47.1 对 47.2）（[第 4 课 §5](lesson-04-inference-kv-cache.html#p5)）[@shazeer2019; @ainslie2023]。
- **MLA 付计算，还要绕开 RoPE。** 推理时把上投影“吸收”进查询，缓存从不展开；代价是每个缓存 token 多做约 3.4 倍运算，换来少读几十倍字节（[第 9 课 §3](lesson-09-v2-mla.html#p3)）。旋转位置编码 RoPE 夹在中间会破坏吸收，于是位置信息另存一个 64 维的共享键，这就是 $d_h^R$ [@su2021roformer]。
- **DSA 付存储，省的是读取与计算。** V3.2 在 MLA 之外多存一个 132 字节的索引键，缓存相对 FP8 的 MLA 反而多了约 20%；它省的是长上下文下每一步要读的条目数和要算的打分。在 128K 处，按 H800 租价折算，预填充约便宜 3.56 倍、解码约 8.65 倍（[第 14 课 §7](lesson-14-sparse-attention.html#p7)）[@dsv32]。DeepSeek 在它之前提出的 NSA 走的是另一条稀疏路线（压缩、选块、滑窗三个分支），获 ACL 2025 最佳论文 [@nsa; @acl2025awards]。
- **CSA/HCA 付细节。** 把 4 个或 128 个 token 合写成一条，条目就记不住“哪个数是哪个 token 的”，逐字细节要靠最近 128 个 token 的滑动窗口。1M 上下文“装得下”不等于“用得好”：多针检索 MRCR 在 1M 处明显下降（[第 18 课 §7](lesson-18-v4.html#p7)）。
- **CSA2 付近似。** 多数层借用别人的 top-k，可能选错；持久化缓存的 SWA 状态靠有界回放近似恢复；短上下文下解码反而比 V4-Flash 贵（[第 19 课 §8](lesson-19-v4-1-flash.html#p8)）。

::: warn “93.3%”和“27% / 10%”各自在比什么
- V2 论文说 KV cache 比 DeepSeek 67B 降了 93.3%，比较的是**部署态**：V2 的缓存平均量化到 6 bit，67B 按 BF16。$1-25{,}920/389{,}120=93.3\%$；两边都按 BF16 只有 $1-69{,}120/389{,}120=82.2\%$ [@dsv2]。
- V4 论文说 1M 上下文下 V4-Pro 只需 V3.2 的 27% 单 token FLOPs 和 10% KV，这只在 1M、以外推到 1M 的 V3.2 为基线、按“等效 FP8 FLOPs”计时成立 [@dsv4]。V4.1 论文的 FLOPs 图改用 BF16 当量（BF16、FP8、FP4 按 1、0.5、0.25 加权），同一个 V3.2 在两张图上相差约 2 倍，两图不能直接比 [@dsv41]。
:::

::: key 主线 A 小结
- 每 token KV = 份数 × $\tfrac1m$ × 每条维数 × 每维字节 (0.1)。GQA 动头数，MLA 动条目维，V3.2 动精度，V4 动序列维，V4.1 同时动层维与精度；DSA 不减字节，减的是每步的读取。
- 五个数 389,120 → 69,120 → 48,068 → 3,514 → 890 都能由架构复算，其中四个与 V4.1 论文图 1(b) 一致；67B 到 V4.1-Flash 共 437 倍。
- 每一步都有代价：质量（GQA）、计算（MLA）、存储反增（DSA）、细节（CSA/HCA）、近似（CSA2）。
:::

# 主线 B：前馈层的稀疏化 | 参数可以很多，每个 token 只用一小部分 | 15

这一页讲混合专家（mixture of experts，MoE）这条线：它要拆开的是“模型越大、每个 token 越贵”这个绑定。先给出 DeepSeek 用了三年的 MoE 层公式，再沿时间看它的每一次改动各自在对付什么，最后用一张图看激活比例怎样一代代变小。

## 问题：参数量和计算量绑在一起

稠密模型处理每个 token 时，每个参数都要参与一次乘加，所以每 token 的前向计算约是 $2N$ 次运算（[第 1 课 §8](lesson-01-transformer-anatomy.html#p8)）。想让模型知道更多，就要加参数；加了参数，每个 token 就更贵。前馈网络（FFN）是这笔账的大头：DeepSeek LLM 7B 有 58.7% 的参数在 FFN 里（[第 7 课 §1](lesson-07-deepseekmoe.html#p1)）。

**条件计算**（conditional computation）的想法是：把一个大 FFN 换成许多个小 FFN（“专家”），每个 token 只经过其中几个。于是**总参数** $N_{\text{total}}$ 可以很大，每个 token 真正参与计算的**激活参数** $N_{\text{act}}$ 却很小。要付的是另一笔账：全部专家都得放在显存里，而且专家分在不同的卡上，token 要在卡与卡之间来回搬。

::: ana MoE = 医院
**映射。** 路由器是分诊台；路由专家是各个专科医生；共享专家是每个病人都要看的全科医生；细粒度切分是把“内科”拆成心内、消化、呼吸……，组合更灵活；负载均衡是别让病人都挤到一个科室；V3 的偏置 $b_i$ 是分诊台看到某科排队太长，就把它的“推荐分”暂时调低。

**收益。** 它让“总参数大、每个病人只看几个医生”变得直观，也让“排队”（负载不均）成为一个自然的问题。

**边界。** 分诊按 token 进行，同一句话里的不同词会去不同科室；“专科”是训练中自发形成的，不是人为指定的，所以没有哪个专家叫“数学专家”。
:::

## 形式化：一层 DeepSeekMoE

从 DeepSeekMoE 到 V4.1，MoE 层一直是“共享专家 + 路由专家”的结构。以 V3 的写法为例（[第 11 课 §2](lesson-11-v3-architecture.html#p2)），第 $t$ 个 token 进入 FFN 位置的隐藏状态是 $\mathbf u_t\in\R^{d}$，输出为

::: eq 共享专家加路由专家 | (0.6)
$$\mathbf h'_t=\mathbf u_t+\sum_{i=1}^{N_s}\FFN^{(s)}_i(\mathbf u_t)+\sum_{i=1}^{N_r}g_{i,t}\,\FFN^{(r)}_i(\mathbf u_t)$$
$N_s$ 个共享专家每个 token 必经；$N_r$ 个路由专家里只有 $K_r$ 个的门控值 $g_{i,t}$ 不为 0。V3 取 $N_s=1$、$N_r=256$、$K_r=8$ [@dsv3]。
:::

门控值怎么来，是这条线后半段反复修改的地方。V3 先算亲和度 $s_{i,t}=\sigma(\mathbf u_t^\top\mathbf e_i)$（$\mathbf e_i\in\R^d$ 是第 $i$ 个专家的中心向量，$\sigma$ 是 sigmoid），再用一个只管“选谁”的偏置 $b_i$ 做 top-$K_r$ 选择：

::: eq 选择用 $s+b$，加权用 $s$ | (0.7)
$$g'_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}+b_i\in\TopK\big(\{s_{j,t}+b_j\}_{j=1}^{N_r},K_r\big)\\ 0, & \text{否则}\end{cases}\qquad g_{i,t}=\frac{g'_{i,t}}{\sum_{j}g'_{j,t}}$$
每一步训练之后，过载的专家 $b_i$ 减 $\gamma$，欠载的加 $\gamma$（V3 取 $\gamma=0.001$）；$b_i$ 不进入门控值，也不产生梯度 [@dsv3; @alf]。
:::

## 改进链

| 节点 | 日期 | 做了什么 | 留下的问题 | 课 |
|---|---|---|---|---|
| 稀疏门控 MoE | <span style="white-space:nowrap">2017-01-23</span> | top-$k$ 门控，把专家数放大到数千，用辅助损失防止路由坍缩 [@shazeer2017] | 负载均衡靠损失硬拉 | [7](lesson-07-deepseekmoe.html#p2) |
| GShard、Switch | <span style="white-space:nowrap">2020-06-30</span>、<span style="white-space:nowrap">2021-01-11</span> | 在 Transformer 里用 top-2 / top-1 MoE，引入容量因子与 $\alpha N\sum_if_iP_i$ 式均衡损失 [@gshard2020; @switch2021] | 专家少而大：知识混杂、冗余 | [7](lesson-07-deepseekmoe.html#p3) |
| DeepSeekMoE | <span style="white-space:nowrap">2024-01-11</span> | 细粒度切分（$N$ 个专家切成 $mN$ 个、激活 $mK$ 个）+ 共享专家隔离；16B 版 2 共享 + 64 路由、激活 6 个 [@dsmoe] | 专家变多，通信变多 | [7](lesson-07-deepseekmoe.html#p5) |
| V2 | <span style="white-space:nowrap">2024-05-06</span> | 2 共享 + 160 路由、激活 6 个；设备受限路由（每 token 至多去 3 台设备）；专家级、设备级、通信三种均衡损失；设备级 token 丢弃 [@dsv2] | 辅助损失与主目标争抢梯度 | [10](lesson-10-v2-moe-training.html#p2) |
| 无辅助损失均衡 | <span style="white-space:nowrap">2024-08-28</span> | 用按负载更新的专家偏置代替辅助损失，在至多 3B、200B tokens 的模型上验证 [@alf] | 需要在大模型上验证 | [11](lesson-11-v3-architecture.html#p4) |
| V3 | <span style="white-space:nowrap">2024-12-26</span> | 1 共享 + 256 路由、激活 8 个；sigmoid 亲和度 + 偏置均衡 + 极小的序列级损失（$\alpha=10^{-4}$）；节点受限路由（至多 4 个节点）；不丢 token [@dsv3] | 路由限制约束了专家的组合 | [11](lesson-11-v3-architecture.html#p5) |
| V4 | <span style="white-space:nowrap">2026-04-24</span> | Flash 1 + 256、Pro 1 + 384，都激活 6 个；前 3 个 MoE 层用哈希路由；取消节点数限制 [@dsv4] | —— | [18](lesson-18-v4.html#p1) |
| V4.1-Flash | <span style="white-space:nowrap">2026-09-10</span> | 1 + 384（中间维 2304）、激活 6 个；不再有哈希路由（图注说全部 FFN 都是标准 DeepSeekMoE，配置里也没有哈希层，正文未明说）；另加 196B 参数的 Engram 表作为第二条稀疏轴 [@dsv41; @hforg] | —— | [19](lesson-19-v4-1-flash.html#p7) |

表里有两处“后来被放弃”的设计，第 20 课会专门讨论：V2 的 token 丢弃到 V3 就不再用（V3 报告原话是训练中不丢任何 token）；V3 的节点受限路由到 V4 被取消。

## 数字：激活比例一代代变小

::: fig figures/l00-params.svg | 图 0.3 总参数与每 token 激活参数（双对数坐标）
**看点落在哪一条斜虚线附近：斜线是“激活参数 / 总参数”的比例，越往右下，每个 token 用到的参数份额越小。** DeepSeekMoE 16B 约 1/6，V2 约 1/11，V3 约 1/18，V4-Pro 约 1/33；V4.1-Flash 的解码与预填充分别激活 16B 与 8B（蓝色竖线的两端），是 552B 主干的 2.9% 与 1.45%。黄褐色虚线把 V4.1 的总参数加上 196B 的 Engram 表（不在主干参数里），合计约 748B。两个黑点是稠密的 DeepSeek LLM，落在“激活比 1”那条线上。参数量取各论文原句，由 `l00_params.py` 画出 [@dsllm; @dsmoe; @dsv2; @dsv3; @dsv4; @dsv41]。
:::

::: ex 细粒度切分换来了什么，又没有换来什么
**组合数。** 16 个专家里选 2 个，有 $\binom{16}{2}=120$ 种组合；每个专家切成 4 份（64 个专家选 8 个），组合数变成 $\binom{64}{8}=4{,}426{,}165{,}368$。总参数和每 token 的计算量都没有变（[第 7 课 §5](lesson-07-deepseekmoe.html#p5)）。

**激活比例。** V3 是 $37/671=5.5\%$，V4-Pro 是 $49/1600=3.1\%$，V4.1-Flash 解码 $16/552=2.9\%$、预填充 $8/552=1.45\%$。

**没有换来的。** 显存里仍要放下全部 671B 或 1.6T 参数；专家分散在许多卡上，每个 token 的 all-to-all 通信随它要去的设备数增长。这正是设备受限路由、节点受限路由和第 12 课那些通信内核要解决的问题。
:::

::: err “DeepSeekMoE 16B 只用了 40% 的计算量”
论文里的“40%”是每 4K token 的前向 FLOPs 之比（74.4T 对 183.5T），对照的是 DeepSeek LLM 7B，不是激活参数之比；16B 的激活参数 2.8B 包含嵌入与输出头（[第 7 课 §8](lesson-07-deepseekmoe.html#p8)）[@dsmoe]。引用这类比例时，先问分子分母各是什么。
:::

## 代价与失败模式

- **路由坍缩。** 被偏爱的专家学得更快、被选得更多，这是自我强化的；不加约束时最忙的专家负载可达平均的 17.8 倍（[第 7 课 §3](lesson-07-deepseekmoe.html#p3)）[@shazeer2017]。
- **辅助损失的两难。** 系数大了伤性能，小了不均衡。无辅助损失论文的 1B 实验里，偏置法同时得到更好的均衡与更低的困惑度（[第 11 课 §1](lesson-11-v3-architecture.html#p1)）；“伤性能是因为干扰梯度”是解释，不是证明。
- **通信与等待。** 专家分到多张卡后，最忙的卡决定整步的时间；token 要去的设备越多，all-to-all 越贵。

::: key 主线 B 小结
- MoE 层 = 共享专家 + 路由专家 (0.6)；DeepSeek 的两招是细粒度切分和共享专家隔离（第 7 课）。
- 均衡的办法从辅助损失（V2 三种）换成了只管选择、不产生梯度的偏置 (0.7)（V3 起）；路由的约束从设备受限（V2，至多 3 台）到节点受限（V3，至多 4 个节点）再到取消（V4）。
- 激活比例从 16B 的约 1/6 降到 V4.1-Flash 预填充的约 1/69；代价是显存要装下全部参数，以及跨卡通信。
:::

# 主线 C：数值与系统 | 用更短的数、更聪明的调度，把算力与显存省下来 | 15

这一页讲“同样的模型，怎样算得更便宜”。主线 C 有两半：数值（每个数用几位存、几位算）与系统（算、传、存怎样重叠与分工）。数值这一半有一个统一的形式：一小组数共用一个缩放因子。

## 问题：一次训练的账单

V3 的技术报告给出了全部正式训练的 GPU 小时：预训练 2,664K、上下文扩展 119K、后训练 5K，合计 2,788K H800 GPU 小时，按每小时 2 美元折算是 557.6 万美元；报告明确说这不包括此前的研究与消融实验（[第 12 课 §1](lesson-12-v3-infrastructure.html#p1)）[@dsv3]。要让这个数小下去，只有两条路：让每次矩阵乘法更便宜（更短的数），让不算的时间更少（把通信藏到计算后面）。推理时还有第三笔：显存装得下多少、每一步要读多少字节，这又回到主线 A。

::: ana 低精度数值 = 短尺子 + 分段放大镜
**映射。** FP8 是一把刻度很少的尺子（一共只有 256 个码）；缩放因子是给每一小段配一个放大镜，让这一段的数正好落在尺子最精细的区间里；细粒度量化是放大镜配得更密（每 128 个数一个，而不是整个张量一个）。

**收益。** 它解释了为什么“一个离群值”在整张量缩放下是灾难：放大镜只能照顾最大的那个数，其余的数都被挤到尺子的粗刻度上。

**边界。** 放大镜只能整体缩放，同一小段里大小悬殊的数，仍然会有一个被量得很粗；而且缩放因子本身也要存、要乘，配得越密，额外的字节和运算越多。
:::

## 形式化：一组数共用一个缩放因子

把一组数 $\{x_j\}_{j\in G}$ 存成低精度时，先按组里的最大绝对值定一个缩放因子 $s$，再把每个数除以 $s$ 后舍入到格式里最近的可表示值：

::: eq 分组缩放量化 | (0.8)
$$s=\frac{\max_{j\in G}\abs{x_j}}{q_{\max}},\qquad \hat x_j=s\cdot\operatorname{round}_{\text{fmt}}\!\Big(\frac{x_j}{s}\Big)$$
$q_{\max}$ 是格式能表示的最大值：FP8 的 E4M3 是 448，FP4 的 E2M1 是 6。V3 的训练里，激活按 $1\times128$、权重按 $128\times128$ 分组，缩放因子存成 FP32（[第 12 课 §3](lesson-12-v3-infrastructure.html#p3)）[@dsv3; @fp8formats]。
:::

V3.1 把缩放因子本身也限制成 2 的整数次幂（UE8M0 格式，8 位全是指数），并向上取整，保证不溢出（[第 15 课 §3](lesson-15-v3-2-agents.html#p3)）：

::: eq UE8M0：只取 2 的幂的缩放因子 | (0.9)
$$s=2^{\lceil\log_2(a/448)\rceil},\qquad a=\max_{j\in G}\abs{x_j}$$
好处是缩放因子只占 1 字节、乘 $2^n$ 精确且便宜；代价是组里最大的数只能用到 E4M3 范围的一半到全部，在对数均匀的假设下平均约 72%。
:::

## 改进链

| 节点 | 日期 | 做了什么 | 课 |
|---|---|---|---|
| BF16 混合精度 | <span style="white-space:nowrap">2022-03-29</span>（Chinchilla 论文） | 前向与反向用 BF16，优化器状态里另存一份 FP32 权重；这是第 2 课引用的写法出处，不是 BF16 训练的首次使用 [@hoffmann2022] | [2](lesson-02-gpt-training.html#p6) |
| FP8 格式 | <span style="white-space:nowrap">2022-09-12</span> | 定义 E4M3（最大 448）与 E5M2（最大 57,344）两种 8 位浮点 [@fp8formats] | [12](lesson-12-v3-infrastructure.html#p2) |
| Fire-Flyer 集群 | <span style="white-space:nowrap">2024-08-26</span> | 用 1 万张 PCIe A100 做软硬件协同的训练集群，配套 HFReduce 与 3FS 文件系统 [@fireflyer] | [12](lesson-12-v3-infrastructure.html#p8) |
| V3 | <span style="white-space:nowrap">2024-12-26</span> | 细粒度 FP8 训练（每 128 个元素把累加提升到 FP32）；DualPipe 让前向、反向与通信重叠；节点受限的 all-to-all 只用 20 个 SM；预填充与解码分开部署 [@dsv3] | [12](lesson-12-v3-infrastructure.html) |
| 开源周 | <span style="white-space:nowrap">2025-02-24</span> 起 | 连续开源 FlashMLA、DeepEP、DeepGEMM、DualPipe 与 EPLB、3FS [@openinfra] | [12](lesson-12-v3-infrastructure.html#p8) |
| V3 的硬件反思 | <span style="white-space:nowrap">2025-05-14</span> | ISCA '25 论文，总结 2,048 张 H800 上的软硬件协同与对硬件的期望 [@insights] | [12](lesson-12-v3-infrastructure.html#p8) |
| V3.1 | <span style="white-space:nowrap">2025-08-21</span> | UE8M0 缩放格式 [@dsv31card] | [15](lesson-15-v3-2-agents.html#p3) |
| V4 | <span style="white-space:nowrap">2026-04-24</span> | 专家权重与 indexer 的 QK 路径用 FP4 量化感知训练；KV 条目除 RoPE 维外存成 FP8 [@dsv4] | [18](lesson-18-v4.html#p6) |
| V4.1-Flash | <span style="white-space:nowrap">2026-09-10</span> | 主 KV 存成 FP4（E2M1，每 16 维一个 E4M3 缩放）；SWA 状态从 SSD 挪进主机内存，丢了只回放 128 个 token [@dsv41] | [19](lesson-19-v4-1-flash.html#p5) |
| DSec | <span style="white-space:nowrap">2026-09-19</span> | 智能体强化学习用的弹性沙箱平台，每天约 300 万个沙箱 [@dsec] | 本课 |

## 数字：一个 KV 条目里的字节

主线 C 与主线 A 在 KV 缓存上相遇：同一个“每 token 缓存多少字节”，A 管的是存几份、每几个 token 一条，C 管的是每个数几位。

::: fig figures/l00-kv-entry.svg | 图 0.4 一个 KV 条目里的字节：从 BF16 到 FP4
**看右边那一列“位/维”：同样是一条缓存，每一维从 16 位降到约 9 位，再降到 4.5 位；缩放因子（灰）一直很小，RoPE 维（红）直到 V4.1 才跟着量化。** V2/V3 的 MLA 条目是 512 维潜向量加 64 维 RoPE 键，全用 BF16；V3.2 把潜向量换成 FP8，另加一个 132 字节的索引键（紫），所以条目总字节反而比纯 FP8 的 MLA 多 20%；V4 的条目是 512 维（RoPE 包含在内），但每条管 4 个（CSA）或 128 个（HCA）token；V4.1 连 RoPE 维一起存成 FP4。字节格式取自 FlashMLA 说明与两篇论文 [@flashmla; @dsv4; @dsv41]，由 `l00_kv_entry.py` 画出。
:::

::: ex 557.6 万美元，与“每维几位”
**训练账单。** $(2{,}664+119+5)\text{K}\times\$2=\$5.576\text{M}$。报告另给出每万亿 token 需要 180K GPU 小时，在 2,048 张卡上是 $180{,}000/2{,}048/24=3.66$ 天（原文写约 3.7 天）[@dsv3]。

**每维几位。** V2/V3：$1{,}152\times8/576=16$ 位；V3.2：$656\times8/576=9.1$ 位；V4：$584\times8/512=9.1$ 位；V4.1：$288\times8/512=4.5$ 位。
:::

## 代价与失败模式

- **范围与精度。** E4M3 只有 256 个码，相对舍入误差可达 6.25%；比组内最大值小约 3 万倍的数开始失真（[第 12 课 §3](lesson-12-v3-infrastructure.html#p3)）。离群值决定整组的缩放，这就是“细粒度”存在的理由。
- **累加误差。** H800 的 FP8 张量核心累加只有约 14 位有效位，误差随内维线性变大，所以 V3 每 128 个元素把部分和搬到 FP32（[第 12 课 §4](lesson-12-v3-infrastructure.html#p4)）。
- **假设。** V4.1 省掉 FP4 的第二级全局缩放，前提是训练出的 RMSNorm 权重大约为 1，于是 512 维向量的每个分量不超过约 $\sqrt{512}\approx22.6$，远小于 $448\times6=2{,}688$；这个前提不成立时论证就不成立（[第 19 课 §5](lesson-19-v4-1-flash.html#p5)）[@dsv41]。
- **理论值与实际值。** 开源周 Day 6 公布的“成本利润率 545%”是按 R1 价格算的理论值，不是实际收入 [@openinfra]。

::: key 主线 C 小结
- 数值：BF16 → FP8（V3，分组缩放 (0.8)，每 128 个元素提升累加精度）→ UE8M0 缩放 (0.9)（V3.1）→ FP4 专家与 indexer（V4）→ FP4 主 KV（V4.1）。
- 系统：DualPipe 把通信藏进计算，节点受限路由压 IB 流量，预填充与解码分开部署；这些部件大多已经开源。
- V3 全部正式训练 2,788K GPU 小时 = 557.6 万美元，不含研究与消融。
:::

# 主线 D：训练目标与后训练 | 从“像语料”到“做得对”，以及它的算力 | 20

这一页讲模型**学什么**。预训练让模型模仿语料；后训练想让它帮得上忙、做得对。这条线上 DeepSeek 最重要的贡献是 GRPO，以及用它在 R1 上证明的一件事：只看最后答案对不对，也能教会模型一步一步地想。

## 问题：“让人满意”写不成一个可以求导的损失

预训练的目标是下一词预测：在语料上最小化交叉熵（[第 2 课 §2](lesson-02-gpt-training.html#p2)）。

::: eq 下一词预测 | (0.10)
$$\mathcal L(\theta)=-\frac1T\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$
$x_t$ 是第 $t$ 个 token，$p_\theta$ 是参数为 $\theta$ 的模型给出的条件概率。它让模型**像**语料，但语料里的“像”不等于“对”或“有用”。
:::

我们真正想要的东西（回答有没有帮上忙、数学题做没做对）只能事后打分，写不成对 $\theta$ 可导的式子。于是有了强化学习：让模型自己写回答、按奖励 $r$ 打分、把高分回答的概率推高，同时用 KL 散度把它拴在参考模型 $\pi_{ref}$ 附近，免得它跑到奖励打分不可靠的地方去（[第 3 课 §3](lesson-03-alignment-rlhf.html#p3)）：

::: eq 带 KL 约束的强化学习目标 | (0.11)
$$\max_{\pi_\theta}\ \E_{q\sim\mathcal D,\ o\sim\pi_\theta(\cdot\mid q)}\big[r(q,o)\big]-\beta\,\KL\big[\pi_\theta\,\Vert\,\pi_{ref}\big]$$
$q$ 是题目（提示），$o$ 是模型写的回答，$\beta$ 是 KL 系数。
:::

InstructGPT 用 PPO 优化它，要同时放四个模型：策略、参考模型、奖励模型、价值网络。按每个训练参数 16 字节、冻结参数 2 字节估算，7B 规模下 SFT 约 112 GB，PPO 约 252 GB（[第 3 课 §6](lesson-03-alignment-rlhf.html#p6)）[@ouyang2022; @schulman2017ppo]。

::: ana GRPO = 同题多答、组内排名
**映射。** 同一道题让学生写 $G$ 份答卷，每份按“比这组平均分高多少个标准差”给分，不用另请一位“预测分数的老师”（价值网络）。

**收益。** 它说明了 GRPO 为什么能省掉价值网络：基线直接用同题其他答卷的平均分。

**边界。** 只有同一道题内比较才有意义；如果一组全对或全错，优势全为零，这道题就不提供学习信号。
:::

## 形式化：组内标准化的优势

GRPO 对每道题 $q$ 采 $G$ 个回答，得到奖励 $r_1,\dots,r_G$，第 $i$ 个回答的每个 token 共用同一个优势（[第 8 课 §4](lesson-08-deepseekmath-grpo.html#p4)）：

::: eq GRPO 的优势 | (0.12)
$$\hat A_i=\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}$$
DeepSeekMath 与 R1 都除以标准差；V3.2 改成只减均值、不除标准差（[第 15 课 §4](lesson-15-v3-2-agents.html#p4)）[@dsmath; @dsv32]。
:::

训练目标这一侧还有一处改动：V3 在下一词预测之外加了多 token 预测（MTP），用 $D$ 个顺序模块，第 $k$ 个模块预测再往后第 $k$ 个 token，损失取平均再乘一个权重 $\lambda$（[第 11 课 §6](lesson-11-v3-architecture.html#p6)）：

::: eq 多 token 预测的附加损失 | (0.13)
$$\mathcal L_{\text{MTP}}=\frac{\lambda}{D}\sum_{k=1}^{D}\mathcal L^{(k)}$$
V3 取 $D=1$，$\lambda$ 在前 10T tokens 为 0.3、之后为 0.1 [@dsv3]；它的思路来自 Meta 的并行多 token 预测 [@gloeckle2024]。
:::

## 改进链

| 节点 | 日期 | 做了什么 | 课 |
|---|---|---|---|
| RLHF 三步法 | <span style="white-space:nowrap">2022-01-27</span> | SFT → 奖励模型 → 带 KL 的 PPO [@ouyang2022] | [3](lesson-03-alignment-rlhf.html) |
| DPO | <span style="white-space:nowrap">2023-05-29</span> | 把奖励模型藏进策略，后两步合成一个二分类损失 [@rafailov2023dpo] | [3](lesson-03-alignment-rlhf.html#p7) |
| GRPO（DeepSeekMath） | <span style="white-space:nowrap">2024-02-05</span> | 去掉价值网络，用组内标准化作优势；$G=64$、$\beta=0.04$；7B 在 MATH 上 51.7% [@dsmath] | [8](lesson-08-deepseekmath-grpo.html) |
| V2 两阶段对齐 | <span style="white-space:nowrap">2024-05-06</span> | SFT 后先做推理 RL，再做有用性与安全的偏好 RL [@dsv2] | [10](lesson-10-v2-moe-training.html#p7) |
| V3 的 MTP | <span style="white-space:nowrap">2024-12-26</span> | 训练时多预测一个 token；推理时可丢掉，也可当投机解码的草稿，接受率 85%–90% [@dsv3] | [11](lesson-11-v3-architecture.html#p6) |
| R1-Zero 与 R1 | <span style="white-space:nowrap">2025-01-20</span> | 在 V3-Base 上只用规则奖励（答案对不对 + 格式）做 GRPO；R1 用冷启动 SFT、推理 RL、拒绝采样 SFT（约 80 万条）、全场景 RL 四阶段；蒸馏出 6 个小模型 [@dsr1] | [13](lesson-13-r1-reasoning.html) |
| V3.1 一模两式 | <span style="white-space:nowrap">2025-08-21</span> | 同一个模型，提示末尾的一个特殊 token 决定先思考还是直接回答 [@dsv31card] | [15](lesson-15-v3-2-agents.html#p2) |
| V3.2 | <span style="white-space:nowrap">2025-12-01</span> | 八个专家各自 RL 再蒸馏回一个模型，一个混合 RL 阶段；后训练预算超过预训练的 10%；四项稳定化改动；85,267 个智能体任务 [@dsv32] | [15](lesson-15-v3-2-agents.html#p4) |
| V4 | <span style="white-space:nowrap">2026-04-24</span> | 先培养各领域专家，再由十个以上的教师在线蒸馏（OPD）成一个模型；三档推理强度 [@dsv4] | [18](lesson-18-v4.html#p6) |
| DSpark | <span style="white-space:nowrap">2026-06-26</span> | 事后训练的半自回归草稿模块，接在 V4 的检查点上做投机解码；同等总吞吐下，V4-Flash 的单用户生成比 MTP-1 基线快 60%–85%；V4.1 用它代替预训练里的 MTP [@dspark] | [19](lesson-19-v4-1-flash.html#p7) |
| V4.1-Flash | <span style="white-space:nowrap">2026-09-10</span> | 预训练不再带 MTP；论文自述后训练没有算法创新，改动都在数据与环境 [@dsv41] | [19](lesson-19-v4-1-flash.html#p7) |

## 数字：后训练从 0.19% 到 10% 以上

::: fig figures/l00-gpu-hours.svg | 图 0.5 公开的 GPU 小时：预训练与后训练
**看绿色部分的长短：V3 的后训练只有 5K 小时，细到在图上看不见；R1 的全部 RL 与数据构造是 147K 小时，约为 V3 全部训练的 5.3%；V3.2 的后训练“超过预训练成本的 10%”。** 前两行是报告原数（V3 技术报告表 1、R1 论文 v2 表 7）[@dsv3; @dsr1]；第三行论文只给了比例，没有给基数，这里按 V3 的 2,664K 折算成至少 266K 小时，画成右端开放的虚框，只作示意 [@dsv32]。金额都按报告自己假设的每 GPU 小时 2 美元。由 `l00_gpu_hours.py` 画出。
:::

::: ex 一组四个回答的优势
一道题采 $G=4$ 个回答，只有第一个答对：$r=(1,0,0,0)$。均值 0.25，总体标准差 $\sqrt{0.25\times0.75}=0.433$，所以

$$\hat A=\Big(\frac{0.75}{0.433},\ \frac{-0.25}{0.433},\ \frac{-0.25}{0.433},\ \frac{-0.25}{0.433}\Big)=(1.732,\ -0.577,\ -0.577,\ -0.577)$$

四个优势之和为 0：答对的那一份被推高，另外三份被压低，而且答对越稀有，推得越狠（这正是 $\sqrt{(1-p)/p}$，$p=0.25$ 时为 1.732，[第 13 课 §3](lesson-13-r1-reasoning.html#p3)）。如果四份都答对或都答错，优势全是 0，这道题这一步不提供任何信号。若按 V3.2 的写法只减均值，答对的优势是 0.75，不再除以 0.433。
:::

## 代价与失败模式

- **奖励被钻空子。** 神经网络奖励模型会被“讨好”，R1 因此在推理题上只用规则奖励，神经奖励只用在一般题目、且只在最后 400 步（[第 13 课 §5](lesson-13-r1-reasoning.html#p5)）。
- **两条没走通的路。** R1 试过过程奖励模型（PRM）和蒙特卡洛树搜索（MCTS），都没用上；论文也说明这不代表它们做不出好的推理模型（[第 13 课 §7](lesson-13-r1-reasoning.html#p7)）。
- **RL 改变了什么。** DeepSeekMath 的实测是 RL 提升 Maj@K（多数投票）而不提升 Pass@K（$K$ 次里对一次），更像把已有的正确答案“磨尖”，而不是教会新本事（[第 8 课 §8](lesson-08-deepseekmath-grpo.html#p8)）。
- **规模化的不稳定。** 后训练放大 50 多倍之后，“采样的模型”与“训练的模型”不再一致，V3.2 为此加了无偏 KL、离策略序列掩码、固定路由、固定采样掩码四项修补（[第 15 课 §5](lesson-15-v3-2-agents.html#p5)）。
- **更长的思考，更贵的推理。** R1 平均每题写 8,793 个思考 token，约为 GPT-4o 的 12 倍（[第 13 课 §1](lesson-13-r1-reasoning.html#p1)）；长输出的注意力账单又回到了主线 A。

::: key 主线 D 小结
- 目标：下一词预测 (0.10) → 带 KL 的 RL (0.11)；DeepSeek 用 GRPO 的组内优势 (0.12) 去掉了价值网络，用 MTP (0.13) 让预训练多看一步。
- R1 证明：在足够强的基座上，只用规则奖励的纯 RL 就能激发长链推理；之后的 V3.1、V3.2、V4 把力气花在“一个模型两种模式”、规模化与稳定性、智能体和专家合并上。
- 后训练的算力从 V3 的 0.19% 放大到 V3.2 的 10% 以上；V4.1 的后训练没有算法创新，MTP 被事后训练的 DSpark 取代。
:::

# 主线 E：残差流与记忆 | 深度方向的“1”，以及记在哪里 | 15

这一页有两个话题，放在同一条线上，是因为它们都在问“信息怎样在模型里保存和传递”。前一半是残差流：每一层把结果加回一条贯穿全程的向量流，这条流上的系数为什么必须是 1，又怎样在保住 1 的前提下把一条流变成四条。后一半是记忆：除了权重和 KV cache，还有没有别的地方可以记东西。

## 问题一：捷径上的系数

残差连接让每一层只学一个增量：$\mathbf x_{l+1}=\mathbf x_l+\mathcal F_l(\mathbf x_l)$，$\mathbf x_l\in\R^d$ 是第 $l$ 层之前的残差流（[第 1 课 §6](lesson-01-transformer-anatomy.html#p6)）。展开 $L$ 层后，最初的 $\mathbf x_l$ 以系数 1 出现在任意深处，梯度也原样传回来。如果捷径上乘的不是 1 而是 $\lambda$，从第 $l$ 层走到第 $L$ 层，直通项就变成 $\lambda^{L-l}$：

::: eq 捷径系数的连乘 | (0.14)
$$\mathbf x_{L}=\lambda^{L-l}\,\mathbf x_l+\cdots,\qquad 1.05^{60}=18.7,\quad 0.95^{60}=0.046$$
偏离 1 的代价随深度**指数**增长（[第 16 课 §1](lesson-16-mhc.html#p1)）。
:::

::: fig figures/l00-residual-gain.svg | 图 0.6 捷径系数连乘 60 次：只有 1 是安全的
**看四条线在第 60 层的落点：每层只多 5%，60 层后放大 18.7 倍；每层 ×1.1428，60 层后放大约 3,000 倍；每层 ×0.95，信号只剩 4.6%。** 右侧两个菱形是 mHC 论文在 27B 模型上测到的“复合反向增益”：不加约束的 Hyper-Connections 约 3000，mHC 约 1.6（第 16 课的数字化读数为 2825 与 1.65）[@mhc]。菱形测的是 4×4 矩阵的连乘，与左边的标量连乘不是同一个量，放在一起只为比较量级。由 `l00_residual_gain.py` 计算。
:::

## 图景：传送带

::: ana 残差流 = 传送带
**映射。** 每层是传送带旁的一个工位，从带上取件、加工、把增量放回带上；恒等映射就是没人动的件原样往下走。Hyper-Connections 是多条并行传送带，再加上带与带之间的调度；mHC 要求调度必须“守恒”：每条带送出去的总量和收进来的总量都是 1（双随机矩阵），所以件既不会越传越多，也不会凭空消失。

**收益。** 它让“守恒”这个约束一目了然，也让“为什么单独约束每行（只管送出）不够”变得可以想象：只管送出不管收进，某条带仍可能被挤爆。

**边界。** 真实的残差流是连续向量空间里的加法，“件”的比喻不表达方向，也不表达正负抵消。
:::

## 形式化：从一条流到四条守恒的流

Hyper-Connections 把残差流复制成 $n$ 条，$X_l\in\R^{n\times d}$；每个子层 $\mathcal F_l$ 用 $H^{\rm pre}_l\in\R^{1\times n}$ 从 $n$ 条流里读出一个输入，用 $H^{\rm post}_l\in\R^{1\times n}$ 把输出写回各条流，再用 $H^{\rm res}_l\in\R^{n\times n}$ 混合各条流（[第 16 课 §2](lesson-16-mhc.html#p2)）[@zhu2025hc]：

::: eq 超连接的一层 | (0.15)
$$X_{l+1}=H^{\rm res}_l\,X_l+H^{\rm post\,\top}_l\,\mathcal F_l\big(H^{\rm pre}_l\,X_l\big)$$
$n=1$、三个映射都取 1 时，它退回普通残差。mHC 要求 $H^{\rm res}_l$ 是**双随机矩阵**：元素非负，每行、每列的和都是 1；它的谱范数恰为 1，而且两个双随机矩阵的乘积仍是双随机矩阵，所以多少层连乘都不会放大 [@mhc]。
:::

## 问题二：记在哪里

到第 16 课为止，模型只有两种记法：训练时写进权重，要靠一层层计算才能“想起来”（付的是深度）；推理时写进 KV cache，每个 token 一份、保留到对话结束（付的是字节）。第 17 课讲两种新形态（[第 17 课 §1](lesson-17-engram-ocr.html#p1)）：

- **Engram 条件记忆。** 用最近几个 token 的编号做哈希、直接查一张大表，取回静态的局部知识。行号只由 token 编号决定，前向开始前就能算出，所以表可以放在主机内存里边算边预取 [@engram]。
- **DeepSeek-OCR 视觉压缩。** 把一页文字渲染成图片，用少得多的视觉 token 装下它：文本 token 不超过视觉 token 10 倍时，逐字读回的精度约 97% [@dsocr]。

## 改进链

| 节点 | 日期 | 做了什么 | 课 |
|---|---|---|---|
| Pre-LN | <span style="white-space:nowrap">2019-02-14</span>（GPT-2） | LayerNorm 挪进分支，主干上只剩加法 [@radford2019] | [1](lesson-01-transformer-anatomy.html#p6) |
| Pre-LN 的梯度分析 | <span style="white-space:nowrap">2020-02-12</span> | 证明初始化时 Post-LN 顶层梯度更大，解释了为什么 Post-LN 需要 warm-up [@xiong2020] | [1](lesson-01-transformer-anatomy.html#p6) |
| Hyper-Connections | <span style="white-space:nowrap">2024-09-29</span> | ByteDance 把残差流扩成 $n$ 条，$n=4$ 最好 [@zhu2025hc] | [16](lesson-16-mhc.html#p2) |
| DeepSeek-OCR | <span style="white-space:nowrap">2025-10-20</span> | DeepEncoder 把一页文字压成几十到几百个视觉 token [@dsocr] | [17](lesson-17-engram-ocr.html#p6) |
| mHC | <span style="white-space:nowrap">2025-12-31</span> | 把 $H^{\rm res}$ 投影成双随机矩阵（Sinkhorn–Knopp 迭代）；$n=4$ 时训练只多 6.7% 时间 [@mhc] | [16](lesson-16-mhc.html#p4) |
| Engram | <span style="white-space:nowrap">2026-01-12</span> | N-gram 哈希查表；MoE 与记忆的参数分配呈 U 形；Engram-27B [@engram] | [17](lesson-17-engram-ocr.html#p2) |
| DeepSeek-OCR 2 | <span style="white-space:nowrap">2026-01-27</span> | 编码器按语义重排视觉 token [@dsocr2] | [17](lesson-17-engram-ocr.html#p7) |
| V4 | <span style="white-space:nowrap">2026-04-24</span> | mHC 进入生产模型：4 条流、Sinkhorn 迭代 20 轮 [@dsv4] | [18](lesson-18-v4.html#p5) |
| V4.1-Flash | <span style="white-space:nowrap">2026-09-10</span> | 单遍 mHC，残差读写从 $20d$ 降到 $10d$；196B 参数的 Engram 表放在主机内存 [@dsv41] | [19](lesson-19-v4-1-flash.html#p7) |

::: ex 两个 2×2 的双随机矩阵
$A=\begin{pmatrix}0.7&0.3\\0.3&0.7\end{pmatrix}$ 每行、每列的和都是 1。它的平方是

$$A^2=\begin{pmatrix}0.7\cdot0.7+0.3\cdot0.3&0.7\cdot0.3+0.3\cdot0.7\\0.3\cdot0.7+0.7\cdot0.3&0.3\cdot0.3+0.7\cdot0.7\end{pmatrix}=\begin{pmatrix}0.58&0.42\\0.42&0.58\end{pmatrix}$$

仍是双随机矩阵，所以两条流的总量 $x_1+x_2$ 经过任意多层都不变。反过来，只约束行和的 $\begin{pmatrix}1&0\\1&0\end{pmatrix}$ 前向安全，反向却能把梯度放大 2 倍（列和是 2）；这就是 mHC 要同时约束行与列的原因（[第 16 课 §4](lesson-16-mhc.html#p4)）。代价也看得见：$A^2$ 比 $A$ 更接近“全取平均”的 $\begin{pmatrix}0.5&0.5\\0.5&0.5\end{pmatrix}$，流与流之间的差别只减不增。
:::

## 代价与失败模式

- **访存。** 按 V4-Flash 计，mHC 的参数只占约 0.012%，算术约 0.33%，瓶颈在读写：不做核融合时每个 token 每个子层要读写约 $34d$ 个元素，是普通残差 $3d$ 的约 11 倍（[第 16 课 §6](lesson-16-mhc.html#p6)）。
- **证据的范围。** Engram 的 U 形曲线只在两个算力点、固定稀疏度下测过；DeepSeek-OCR 测的是逐字 OCR 精度，不是在压缩后的上下文上推理的能力，V4.1-Flash 也没有采用它的编码器（[第 17 课 §7](lesson-17-engram-ocr.html#p7)）。

::: key 主线 E 小结
- 残差捷径的系数必须是 1 (0.14)；HC 把一条流变成 $n$ 条并让混合矩阵可学 (0.15)，却在 27B 上放大到约 3000 倍；mHC 用双随机矩阵把它压回约 1.6。
- 记忆的两种新形态：Engram 按 token 编号查表（与 MoE 互补，可放主机内存），DeepSeek-OCR 用视觉 token 压缩上下文（证据限于逐字 OCR）。
- V4 把 mHC 用进生产模型，V4.1 再把残差访存减半，并把 196B 的 Engram 表放进主机内存。
:::

# 主线 F：规模化规律与优化器 | 一次性的大训练，靠小实验下注 | 15

这一页讲怎样“下注”。一次大训练只能跑一遍，学习率、batch 大小、模型与数据怎么分，全要在开跑之前定下来。主线 F 的办法是：在小预算上把规律测准，拟合成幂律，再外推到大预算；优化器则决定每一步怎样走。

## 问题：一次训练，没有第二次机会

训练一个稠密 Transformer 的算力约为 $C\approx6ND$：每个 token、每个参数，前向 2 次、反向 4 次运算（[第 2 课 §6](lesson-02-gpt-training.html#p6)）。给定 $C$，模型该多大（$N$）、数据该多少（$D$）？Kaplan 2020 的拟合说主要加大模型，$N_{\text{opt}}\propto C^{0.73}$ [@kaplan2020]；Chinchilla 2022 用三种方法得到参数与数据同比例增长，由此流传出“每个参数约 20 个 token”的经验法则 [@hoffmann2022]。两家的结论互相矛盾，超参数怎么随规模变也交代不全。

## 形式化：DeepSeek LLM 的三条幂律

DeepSeek LLM 的回答分三步（[第 6 课](lesson-06-deepseek-llm.html)）。先换一个记账方式：用每 token 的非嵌入 FLOPs $M$ 代替 $6N$，训练算力写成 $C=MD$；$M=72\,l\,d^2+12\,l\,d\,T$ 把注意力的打分与加权求和也算了进去（$T$ 是序列长度）[@dsllm]。再在小预算上找出近优的学习率与 batch：

::: eq 超参数的规模化规律 | (0.16)
$$\eta_{\text{opt}}=0.3118\,C^{-0.125},\qquad B_{\text{opt}}=0.2920\,C^{0.3271}$$
$\eta$ 是学习率，$B$ 按 token 计。算力每乘 10，batch 约乘 2.12，学习率约乘 0.75（[第 6 课 §6](lesson-06-deepseek-llm.html#p6)）。
:::

最后拟合模型与数据的最优分配：

::: eq 最优分配 | (0.17)
$$M_{\text{opt}}=0.1715\,C^{0.5243},\qquad D_{\text{opt}}=5.8316\,C^{0.4757}$$
两个指数之和为 1，因为 $M_{\text{opt}}D_{\text{opt}}=C$。报告还发现，数据质量越高，模型那一侧的指数越大（0.450、0.524、0.578），并把原因标为推测（[第 6 课 §7](lesson-06-deepseek-llm.html#p7)）。
:::

优化器这一侧，DeepSeek 从 DeepSeek LLM 到 V3 一直用 AdamW [@loshchilov2019]；V4 把大部分矩阵参数换成 Muon：先按动量求出更新方向，再用牛顿–舒尔茨迭代把它近似**正交化**（所有奇异值推向 1），然后重新缩放（[第 18 课 §5](lesson-18-v4.html#p5)）：

::: eq V4 的 Muon 一步（算法 1 第 4–8 行） | (0.18)
$$\begin{aligned}M_t&=\mu M_{t-1}+G_t\\ O_t&=\operatorname{NS}(\mu M_t+G_t)\cdot\sqrt{\max(n,m)}\cdot\gamma\\ W_t&=W_{t-1}(1-\eta\lambda)-\eta\,O_t\end{aligned}$$
$G_t$ 是梯度，$W\in\R^{n\times m}$ 是一个权重矩阵，$\mu=0.95$ 是动量，$\lambda=0.1$ 是权重衰减，$\gamma$ 把更新的均方根缩放到 0.18；NS 是 8 步快速迭代加 2 步精修的牛顿–舒尔茨 [@dsv4; @jordan2024muon]。
:::

## 改进链

| 节点 | 日期 | 做了什么 | 课 |
|---|---|---|---|
| AdamW | <span style="white-space:nowrap">2017-11-14</span> | 把权重衰减从梯度里解耦出来 [@loshchilov2019] | [2](lesson-02-gpt-training.html#p5) |
| Kaplan 规模化规律 | <span style="white-space:nowrap">2020-01-23</span> | 损失随 $N$、$D$、$C$ 的幂律；主要加大模型 [@kaplan2020] | [2](lesson-02-gpt-training.html#p7) |
| Chinchilla | <span style="white-space:nowrap">2022-03-29</span> | 参数与数据同比例增长 [@hoffmann2022] | [2](lesson-02-gpt-training.html#p7) |
| DeepSeek LLM | <span style="white-space:nowrap">2024-01-05</span>（论文） | 超参数规模化规律 (0.16)、用 $M$ 记账、最优分配 (0.17) 依赖数据质量、为续训设计的多步学习率调度 [@dsllm] | [6](lesson-06-deepseek-llm.html) |
| Muon（博客） | <span style="white-space:nowrap">2024-12-08</span> | 动量 + 牛顿–舒尔茨正交化，建议嵌入与输出层仍用 AdamW [@jordan2024muon] | [18](lesson-18-v4.html#p5) |
| V4 | <span style="white-space:nowrap">2026-04-24</span> | 大部分矩阵用 Muon (0.18)；嵌入、预测头、RMSNorm 权重等仍用 AdamW [@dsv4] | [18](lesson-18-v4.html#p5) |
| V4.1-Flash | <span style="white-space:nowrap">2026-09-10</span> | 嵌入表、预测头与 Engram 表改用“动量 + Sinkhorn 平衡”的更新，只存一个动量，省下 Adam 的二阶矩状态 [@dsv41] | [17](lesson-17-engram-ocr.html#p5) |

## 数字：每个参数分到多少 token

MoE 让“每个参数分到多少 token”有了两种算法：按总参数算，还是按激活参数算。

::: fig figures/l00-tokens-per-param.svg | 图 0.7 每个参数分到多少训练 token
**看浅色与深色两根柱子差多远：按总参数算，DeepSeek 的模型大多在 20 到 120 个 token 之间，离 Chinchilla 的经验值不远；按激活参数算，MoE 模型每个激活参数见过几百到近三千个 token。** DeepSeek LLM 67B 是稠密模型，两种算法相同（29.9）。V4.1-Flash 按解码的 16B 激活参数计，训练数据是 45T 多模态 token。Chinchilla 的“约 20”是稠密模型上的经验法则，不是 MoE 的最优值，这里只作参照 [@hoffmann2022]。token 数与参数量取各论文原句，由 `l00_tokens_per_param.py` 计算。
:::

::: ex 用规律给 7B 与 67B 定超参数
把两个模型的训练算力代进 (0.16)，预测 7B 的学习率为 $4.24\times10^{-4}$、batch 为 9.24M token，67B 为 $3.17\times10^{-4}$、19.7M token（报告图 3 标作 4.2e-04、9.2M 与 3.2e-04、19.7M）；实际用的是 $4.2\times10^{-4}$ 配 2,304 条序列、$3.2\times10^{-4}$ 配 4,608 条序列（每条 4,096 个 token，即约 9.4M 与 18.9M token）（[第 6 课 §6](lesson-06-deepseek-llm.html#p6)）[@dsllm]。

再看图 0.7 的两个数：V3 是 $14.8\text{T}/671\text{B}=22.1$ 与 $14.8\text{T}/37\text{B}=400$；V4.1-Flash 是 $45\text{T}/552\text{B}=81.5$ 与 $45\text{T}/16\text{B}=2{,}812.5$。
:::

## 代价与失败模式

- **外推是下注。** 幂律是在小预算上拟合的，7B 与 67B 的训练算力分别是最大一组 IsoFLOP 实验的 283 倍与 2,885 倍（[第 6 课 §1](lesson-06-deepseek-llm.html#p1)）；拟合只依赖 $C$，同算力下不同的模型形状，最优区域会略有移动。
- **口径。** 在最小的实验模型上，只数非嵌入参数的 $6N_1$ 只有 $M$ 的 0.43，把输出头也算进去的 $6N_2$ 又高估 32%（[第 6 课 §5](lesson-06-deepseek-llm.html#p5)）；MoE 再让“$N$”分成总参数与激活参数两个数。
- **Muon 不是“全部参数都换”。** V4 的嵌入、预测头、RMSNorm 权重和 mHC 的静态偏置、门控仍用 AdamW；牛顿–舒尔茨只把奇异值**近似**推到 1（[第 18 课 §5](lesson-18-v4.html#p5)）。

::: key 主线 F 小结
- 算力 $C\approx6ND$；DeepSeek LLM 改用 $C=MD$，并拟合出学习率与 batch (0.16)、模型与数据 (0.17) 的幂律，最优分配随数据质量而变。
- 优化器从 AdamW 到 Muon (0.18)（V4），V4.1 再给嵌入表与 Engram 表换上更省状态的更新。
- MoE 让“每个参数多少 token”有两种算法：按总参数多在 20–120（V2-Lite 例外，363），按激活参数几百到近三千。
:::

# 规格总表 | 主线模型的规格、完整时间线，以及几个常见误读 | 15

这一页是全课的“参考卡”：两张规格表、一张完整时间线、一张常见误读表。每一格都来自研究笔记；“来源”一栏写的是那一格最终依据的原文位置，“笔者推算”“笔者合计”的格子会明说。读后面各课时，可以随时回到这里查一个数。

## 规格表一：规模与训练

| 模型 | 发布日 | 层数 | $d_{model}$ | 总 / 激活参数 | 训练 tokens | 上下文 | 来源 | 课 |
|---|---|---|---|---|---|---|---|---|
| DeepSeek LLM 7B | <span style="white-space:nowrap">2023-11-29</span> | 30 | 4096 | 稠密 7B | 2.0T | 4096 | 论文表 2 [@dsllm] | [6](lesson-06-deepseek-llm.html#p3) |
| DeepSeek LLM 67B | <span style="white-space:nowrap">2023-11-29</span> | 95 | 8192 | 稠密 67B | 2.0T | 4096 | 论文表 2、§2.2；config [@dsllm; @hforg] | [6](lesson-06-deepseek-llm.html#p3) |
| DeepSeekMoE 16B | <span style="white-space:nowrap">2024-01-11</span> | 28 | 2048 | 16.4B / 2.8B | 2T | 4K | 论文 §5.1.2；上下文见 V2 附录 B 与 config [@dsmoe] | [7](lesson-07-deepseekmoe.html#p8) |
| DeepSeek-V2 | <span style="white-space:nowrap">2024-05-06</span> | 60 | 5120 | 236B / 21B | 8.1T | 128K | 论文摘要、§3.1.2 [@dsv2] | [9](lesson-09-v2-mla.html)、[10](lesson-10-v2-moe-training.html#p1) |
| DeepSeek-V2-Lite | <span style="white-space:nowrap">2024-05-16</span> | 27 | 2048 | 15.7B / 2.4B | 5.7T | 32K | 论文附录 B.1 [@dsv2] | [10](lesson-10-v2-moe-training.html#p8) |
| DeepSeek-V3 | <span style="white-space:nowrap">2024-12-26</span> | 61 | 7168 | 671B / 37B | 14.8T | 128K | 论文 §4.2 [@dsv3] | [11](lesson-11-v3-architecture.html#p8)、[12](lesson-12-v3-infrastructure.html) |
| DeepSeek-R1 | <span style="white-space:nowrap">2025-01-20</span> | 同 V3 | 同 V3 | 671B / 37B | 在 V3-Base 上做 RL | 128K | 论文 v2；GitHub README 表 [@dsr1; @ghorg] | [13](lesson-13-r1-reasoning.html) |
| DeepSeek-V3.1 | <span style="white-space:nowrap">2025-08-21</span> | 同 V3 | 同 V3 | 671B / 37B | V3 基座 + 839B 续训 | 128K | 模型卡 [@dsv31card] | [15](lesson-15-v3-2-agents.html#p2) |
| DeepSeek-V3.2 | <span style="white-space:nowrap">2025-12-01</span> | 61 | 7168 | 按与 V3 相同的结构记为 671B / 37B（论文未单列） | 约 16.6T（笔者合计） | 128K | 论文 §2.1、§4；config [@dsv32; @hforg] | [14](lesson-14-sparse-attention.html)、[15](lesson-15-v3-2-agents.html) |
| DeepSeek-V4-Flash | <span style="white-space:nowrap">2026-04-24</span> | 43 | 4096 | 284B / 13B | 32T | 1M | 论文 §1、§4.2 [@dsv4] | [18](lesson-18-v4.html#p1) |
| DeepSeek-V4-Pro | <span style="white-space:nowrap">2026-04-24</span>（预览版） | 61 | 7168 | 1.6T / 49B | 33T | 1M | 论文 §1、§4.2 [@dsv4] | [18](lesson-18-v4.html#p1) |
| DeepSeek-V4.1-Flash | <span style="white-space:nowrap">2026-09-10</span> | 40（编码器 20 + 解码器 20） | 5120 | 552B 主干（另有 196B Engram）/ 预填充 8B、解码 16B | 45T 多模态 | 1M | 论文 §2.1、§4.2 [@dsv41] | [19](lesson-19-v4-1-flash.html#p2) |

V3.2 的“约 16.6T”是笔者把 V3 的 14.8T、V3.1 的 839B 续训、V3.2 的两阶段续训（2.1B + 943.7B）加在一起得到的，论文没有给总数。上下文一栏是论文或模型卡宣称支持的长度；配置文件里的 `max_position_embeddings` 另有含义（见下面的误读表）。

## 规格表二：注意力、KV 缓存与专家

| 模型 | 注意力 | 每 token 全局 KV（字节） | 专家（共享 + 路由，每 token 激活） |
|---|---|---|---|
| DeepSeek LLM 7B | MHA，32 头 × 128 | 491,520 | 稠密 |
| DeepSeek LLM 67B | GQA：64 个查询头、8 个 KV 头，$d_h=128$ | 389,120 | 稠密 |
| DeepSeekMoE 16B | MHA，16 头 × 128 | 229,376 | 2 + 64，激活 6；每个专家是标准 FFN 的 1/4 |
| DeepSeek-V2 | MLA：128 头，$d_h=128$，$d_c=512$，$d_c'=1536$，$d_h^R=64$ | 69,120（BF16）；部署时平均 6 bit 约 25,920 | 2 + 160，激活 6；第 1 层稠密 |
| DeepSeek-V2-Lite | MLA：16 头，$d_c=512$，不压缩查询，$d_h^R=64$ | 31,104 | 2 + 64，激活 6 |
| DeepSeek-V3 | MLA，维度同 V2 | 70,272（BF16） | 1 + 256，激活 8；前 3 层稠密；每 token 至多 4 个节点 |
| DeepSeek-V3.2 | MLA + DSA：indexer 64 头 × 128，top-k 2048 | 48,068 | 同 V3 |
| DeepSeek-V4-Flash | 前 2 层纯 SWA；21 层 CSA（$m=4$，top-k 512）与 20 层 HCA（$m'=128$）交替；64 头，条目维 $c=512$；SWA 窗口 128 | 3,514 | 1 + 256，激活 6；前 3 个 MoE 层哈希路由 |
| DeepSeek-V4-Pro | 61 层里 30 层 CSA（$m=4$，top-k 1024）、31 层 HCA（$m'=128$，含最前面 2 层），交替排列；128 头，$c=512$ | 5,031（笔者推算） | 1 + 384，激活 6 |
| DeepSeek-V4.1-Flash | 前 2 层纯 SWA；编码器 18 层 CSA2（$m=2$：3 Full + 15 Reuse）；解码器 20 层 CSA2（$m=1$：1 Full + 4 Reindex + 15 Reuse）；indexer 32 头 × 128，top-k 512；64 头，$c=512$；主 KV 用 FP4 | 890 | 1 + 384，激活 6；另有 196B Engram 表 |

注意力与专家两栏取自各论文的超参数小节与配置文件 [@dsllm; @dsmoe; @dsv2; @dsv3; @dsv32; @dsv4; @dsv41; @hforg]。KV 一栏由式 (0.2)–(0.5) 复算；其中 389,120、48,068、3,514、890 与 V4.1 论文图 1(b) 的柱长一致，70,272 与 ISCA 论文表 1 的 70.272 KB 一致 [@insights]。

::: ex 从配置文件复现 3,514 与 890
**V4-Flash。** 配置文件的 `compress_ratios` 有 44 项：前 43 项对应主干的 43 层，其中 21 个 4（CSA）、20 个 128（HCA）、2 个 0（纯 SWA，不存全局 KV）；第 44 项的 0 属于 MTP 层，不计。代入式 (0.4)：$21\times652/4+20\times584/128=3{,}423+91.25=3{,}514.25$。

**V4.1-Flash。** 配置文件的 `kv_source_layer_ids` 是 [2, 8, 14, 20]，只有这 4 层自己压缩 KV：前三层在编码器里（$m=2$），第 20 层是解码器的第一层（$m=1$）。代入式 (0.5)：$(3\times\tfrac12+1)\times356=890$ [@hforg; @dsv41]。
:::

## 完整时间线

主线字母沿用 [§1](#p1)；M 表示多模态旁支，P 表示形式证明旁支。“（论文）”表示日期取 arXiv v1：这一项没有模型发布，或者权重晚于论文才公开（如 DeepSeek-Prover）；“—”表示本课程没有专门讲它。日期全部取自研究笔记的发布总表 [@dsnews; @dschangelog; @ghorg; @hforg]。

| 日期 | 名称 | 一句话 | 主线 | 课 |
|---|---|---|---|---|
| <span style="white-space:nowrap">2023-11-02</span> | DeepSeek-Coder 1.3B–33B | 仓库级代码语料 + 填空（FIM）训练；日期是 GitHub 最早提交，属代理值 | D | [6](lesson-06-deepseek-llm.html#p8) |
| <span style="white-space:nowrap">2023-11-29</span> | **DeepSeek LLM 7B / 67B** | 超参数规模化规律；67B 用 GQA、加深到 95 层 | F、A | [6](lesson-06-deepseek-llm.html) |
| <span style="white-space:nowrap">2024-01-11</span> | DeepSeekMoE 16B | 细粒度专家 + 共享专家 | B | [7](lesson-07-deepseekmoe.html) |
| <span style="white-space:nowrap">2024-01-25</span> | DeepSeek-Coder-v1.5 7B | 从 LLM 7B 续训 2T tokens | D | — |
| <span style="white-space:nowrap">2024-02-05</span> | DeepSeekMath 7B | 120B 数学语料；GRPO | D | [8](lesson-08-deepseekmath-grpo.html) |
| <span style="white-space:nowrap">2024-03-11</span> | DeepSeek-VL | 混合视觉编码器 | M | — |
| <span style="white-space:nowrap">2024-05-06</span> | **DeepSeek-V2** | MLA；设备受限路由 | A、B | [9](lesson-09-v2-mla.html)、[10](lesson-10-v2-moe-training.html) |
| <span style="white-space:nowrap">2024-05-16</span> | DeepSeek-V2-Lite | 小号 MLA + MoE | A、B | [10](lesson-10-v2-moe-training.html#p8) |
| <span style="white-space:nowrap">2024-05-23</span>（论文） | DeepSeek-Prover | 800 万条合成的 Lean 4 证明 | P | — |
| <span style="white-space:nowrap">2024-06-14</span> | DeepSeek-Coder-V2 | 从 V2 中间检查点续训 6T tokens | D | [10](lesson-10-v2-moe-training.html#p8) |
| <span style="white-space:nowrap">2024-06-28</span> | DeepSeek-V2-Chat-0628 | 换 Coder-V2 基座重做对话版 | D | — |
| <span style="white-space:nowrap">2024-07-04</span> | ESFT | 只微调与任务相关的专家 | B | — |
| <span style="white-space:nowrap">2024-07-24</span> | DeepSeek-Coder-V2-0724 | 对齐优化 | D | — |
| <span style="white-space:nowrap">2024-08-15</span> | DeepSeek-Prover-V1.5 | 证明助手反馈的 RL + 蒙特卡洛树搜索变体 | P | — |
| <span style="white-space:nowrap">2024-08-26</span>（论文） | Fire-Flyer AI-HPC | 集群的软硬件协同 | C | [12](lesson-12-v3-infrastructure.html#p8) |
| <span style="white-space:nowrap">2024-08-28</span>（论文） | 无辅助损失负载均衡 | 专家偏置代替辅助损失 | B | [11](lesson-11-v3-architecture.html#p4) |
| <span style="white-space:nowrap">2024-09-05</span> | DeepSeek-V2.5 | 对话版与代码版合并 | D | [10](lesson-10-v2-moe-training.html#p8) |
| <span style="white-space:nowrap">2024-10-18</span> | Janus | 理解与生成解耦的视觉编码 | M | — |
| <span style="white-space:nowrap">2024-11-12</span> | JanusFlow | 自回归 + rectified flow | M | — |
| <span style="white-space:nowrap">2024-11-20</span> | DeepSeek-R1-Lite-Preview | 只在网页上提供的长思维链推理服务 | D | — |
| <span style="white-space:nowrap">2024-12-10</span> | DeepSeek-V2.5-1210 | V2 系列的最后一版 | D | — |
| <span style="white-space:nowrap">2024-12-13</span> | DeepSeek-VL2 | 动态切图 + MLA-MoE 语言模型 | M | — |
| <span style="white-space:nowrap">2024-12-26</span> | **DeepSeek-V3** | 无辅助损失均衡、MTP、FP8 训练、DualPipe | B、C、D | [11](lesson-11-v3-architecture.html)、[12](lesson-12-v3-infrastructure.html) |
| <span style="white-space:nowrap">2025-01-20</span> | **DeepSeek-R1**、R1-Zero 与 6 个蒸馏模型 | 纯 RL 激发推理；Nature 版 <span style="white-space:nowrap">2025-09-17</span> 在线发表 | D | [13](lesson-13-r1-reasoning.html) |
| <span style="white-space:nowrap">2025-01-27</span> | Janus-Pro | 放大数据与模型的 Janus | M | — |
| <span style="white-space:nowrap">2025-02-11</span>（论文） | CodeI/O | 用代码的输入输出预测凝练推理模式 | D | — |
| <span style="white-space:nowrap">2025-02-16</span>（论文） | NSA | 可训练的原生稀疏注意力；ACL 2025 最佳论文 | A | [14](lesson-14-sparse-attention.html#p2) |
| <span style="white-space:nowrap">2025-02-24</span> 起 | 开源周 | FlashMLA、DeepEP、DeepGEMM、DualPipe 与 EPLB、3FS | C | [12](lesson-12-v3-infrastructure.html#p8) |
| <span style="white-space:nowrap">2025-03-24</span> | DeepSeek-V3-0324 | 推理与前端能力提升，改用 MIT 许可 | D | [15](lesson-15-v3-2-agents.html#p1) |
| <span style="white-space:nowrap">2025-04-03</span>（论文） | DeepSeek-GRM | 生成式奖励模型的推理期扩展 | D | — |
| <span style="white-space:nowrap">2025-04-30</span> | DeepSeek-Prover-V2 | 子目标分解 + RL，建在 V3-Base 上 | P | — |
| <span style="white-space:nowrap">2025-05-14</span>（论文） | Insights into DeepSeek-V3 | 硬件–模型协同的反思（ISCA '25） | C | [12](lesson-12-v3-infrastructure.html#p8) |
| <span style="white-space:nowrap">2025-05-28</span> | DeepSeek-R1-0528 | 更深的推理、函数调用 | D | [13](lesson-13-r1-reasoning.html#p8) |
| <span style="white-space:nowrap">2025-08-21</span> | DeepSeek-V3.1 | 一模两式（思考 / 不思考）；UE8M0 | D、C | [15](lesson-15-v3-2-agents.html#p2) |
| <span style="white-space:nowrap">2025-09-22</span> | DeepSeek-V3.1-Terminus | 语言一致性、智能体 | D | [15](lesson-15-v3-2-agents.html#p1) |
| <span style="white-space:nowrap">2025-09-29</span> | DeepSeek-V3.2-Exp | DSA 首发 | A | [14](lesson-14-sparse-attention.html#p4) |
| <span style="white-space:nowrap">2025-10-20</span> | DeepSeek-OCR | 用视觉 token 压缩文字上下文 | E | [17](lesson-17-engram-ocr.html#p6) |
| <span style="white-space:nowrap">2025-11-19</span> | LPLB | 线性规划的专家负载均衡器（早期研究） | B | — |
| <span style="white-space:nowrap">2025-11-27</span> | DeepSeekMath-V2 | 可自我验证的证明生成 | D | [15](lesson-15-v3-2-agents.html#p8) |
| <span style="white-space:nowrap">2025-12-01</span> | **DeepSeek-V3.2**、V3.2-Speciale | 规模化 RL、思考融入工具调用 | A、D | [14](lesson-14-sparse-attention.html)、[15](lesson-15-v3-2-agents.html) |
| <span style="white-space:nowrap">2025-12-31</span>（论文） | mHC | 双随机约束的超连接 | E | [16](lesson-16-mhc.html) |
| <span style="white-space:nowrap">2026-01-12</span> | Engram | N-gram 条件记忆，新的稀疏轴 | E、B | [17](lesson-17-engram-ocr.html#p2) |
| <span style="white-space:nowrap">2026-01-27</span> | DeepSeek-OCR 2 | 按语义重排视觉 token | E | [17](lesson-17-engram-ocr.html#p7) |
| <span style="white-space:nowrap">2026-04-24</span> | **DeepSeek-V4 预览版**（Pro、Flash） | CSA/HCA、mHC、Muon、FP4 专家 | A、B、C、D、E、F | [18](lesson-18-v4.html) |
| <span style="white-space:nowrap">2026-06-26</span> | DeepSpec / DSpark | 事后训练的投机解码模块 | C、D | [19](lesson-19-v4-1-flash.html#p7) |
| <span style="white-space:nowrap">2026-07-31</span> | DeepSeek-V4-Flash-0731 | 同一结构，重做后训练（V4-Flash 正式版） | D | [18](lesson-18-v4.html#p7) |
| <span style="white-space:nowrap">2026-08-13</span> | DeepSeek-V4-Pro-0813 | V4-Pro 正式版；推理强度 low / high / max 三档 | D | [18](lesson-18-v4.html#p7) |
| <span style="white-space:nowrap">2026-08-21</span> | DeepSeek-V4-Flash-Vision-Exp | V4 的第一个多模态实验模型 | M | — |
| <span style="white-space:nowrap">2026-09-10</span> | **DeepSeek-V4.1-Flash** | 因果编码器–解码器、CSA2、FP4 KV、SWA 有界回放 | A、B、C、E | [19](lesson-19-v4-1-flash.html) |
| <span style="white-space:nowrap">2026-09-19</span>（论文） | DSec | 智能体 RL 的弹性沙箱平台 | C | — |

表里没有 DeepSeek 的 App、API 的功能更新和各类工具仓库；网上流传的“DeepSeek-R2”不在任何官方渠道里，也不收录（见下表）。

## 常见误读

| 常见说法 | 更准确的说法 | 依据 |
|---|---|---|
| “V4 论文是 2026 年 6 月发的” | 编号是 2606.19348，但 arXiv 提交于 <span style="white-space:nowrap">2026-04-26</span>；权重与技术报告 04-24 已在 HF 公开 | [@dsv4] |
| “V4.1-Flash 是 552B 模型” | 552B 是主干参数，另有 196B Engram 参数；激活参数要分预填充 8B 与解码 16B | [@dsv41] |
| “V4.1 的激活参数只有 V4-Pro 的 1/4” | 论文原话如此，但 $16/49=0.33$、$8/49=0.16$，只有两者的平均才接近 1/4；直接写 8B、16B、49B 更稳妥 | [@dsv41] |
| “DeepSeek-V1 是一次独立发布” | 这是官方后来给 DeepSeek LLM 67B 起的别名（V2 的 README、V4.1 论文图 1(b)）；最早公开的 DeepSeek 语言模型是 DeepSeek-Coder | [@ghorg; @dsv41] |
| “config 里写 163,840，说明支持 160K 上下文” | $163{,}840=4096\times40$ 是 YaRN 的位置上限；V2/V3 系列的论文与模型卡写的都是 128K | [@hforg; @dsv2] |
| “小模型的 KV cache 一定更小” | DeepSeek LLM 7B 用 MHA，每 token 491,520 字节，比用 GQA-8 的 67B（389,120）还大 | [@dsllm] |
| “V3 只花了 557.6 万美元” | 这只是正式训练的 GPU 小时按租价折算，报告明说不含此前的研究与消融；R1 的 29.4 万美元也不含 V3-Base 本身 | [@dsv3; @dsr1] |
| “V4 用 32T tokens 训练” | 32T 只是 Flash；Pro 是 33T；V4.1-Flash 是 45T 多模态 tokens | [@dsv4; @dsv41] |
| “DeepSeek-R2 已经发布” | 截至研究笔记的核查日（<span style="white-space:nowrap">2026-09-24</span>），API 新闻页、更新日志、HF 与 GitHub 组织里都没有 R2 | [@dsnews; @dschangelog] |

::: key 本页要点
- 规格表一给规模与训练，规格表二给注意力、KV 与专家；每一格都能追到论文、配置文件或复算，推算与合计的格子已标明。
- 时间线 50 行里，主干只有 7 个（DeepSeek LLM、V2、V3、R1、V3.2、V4、V4.1）；其余是同结构的更新、旁支模型、论文与开源制品。
- 引用一个数之前先问三件事：它是哪一版（预览还是正式、arXiv 哪一版），它的口径是什么（主干还是含 Engram、预填充还是解码），它是原文还是推算。
:::

# 怎么读这门课 | 推荐顺序、数学对照、图的颜色、八个类比，以及这门课是怎么做出来的 | 15

这一页是使用说明：先读什么、每一课要补哪一节数学、图里的颜色和文中的类比各指什么，最后交代这门课的材料从哪里来、你怎样在源文件里留批注。

## 推荐顺序

默认的读法是**按课号顺序**。每一课都接着上一课留下的问题开头，封面上写着“这一课在科技树上的位置”和前置课。二十课分成五个部分（加上最后的复盘），下表的时间取自各课封面的估计：

| 部分 | 课 | 讲什么 | 阅读 | 练习 |
|---|---|---|---|---|
| 一、GPT 的地基 | 1–5 | 标准 GPT 的架构、训练、对齐、推理，以及现代的 Transformer 积木 | 910 分钟 | 250 分钟 |
| 二、DeepSeek 的起点 | 6–8 | DeepSeek LLM 的规模化方法、DeepSeekMoE、DeepSeekMath 与 GRPO | 530 分钟 | 145 分钟 |
| 三、V2 与 V3 | 9–12 | MLA、规模化 MoE 的训练、无辅助损失均衡与 MTP、FP8 与 DualPipe | 835 分钟 | 200 分钟 |
| 四、推理模型与稀疏注意力 | 13–15 | R1、NSA 与 DSA、V3.1 / V3.2 与智能体 | 520 分钟 | 145 分钟 |
| 五、新架构 | 16–19 | mHC、Engram 与 OCR、V4、V4.1-Flash | 830 分钟 | 195 分钟 |

第 1–19 课合计约 3,625 分钟阅读、935 分钟练习，也就是六十多个小时。十一课数学补课另有约 1,710 分钟阅读、505 分钟练习，但不必整课读完：每一课只在用到的地方链接到对应的那一节。

**只关心一条主线的读者**可以沿图 0.1 的一条色带跳读，例如主线 A 走第 1 → 4 → 5 → 9 → 14 → 18 → 19 课，主线 D 走第 1 → 2 → 3 → 8 → 13 → 15 课。要知道这样会跳过一些前置：每一课封面写明的前置课见图 0.8，比如第 18 课还要求第 11 课（V3 的 MoE 与 MTP）和第 16 课（mHC）。跳过的部分，各课的黄色“前置知识”框会就地补一点，但不如完整读来得扎实。

::: fig figures/l00-prereq.svg | 图 0.8 每一课的前置课与数学课（由各课源文件自动读取）
**看左半边的蓝点怎样沿对角线向右下铺开：越往后的课，前置越多、跨度越大；再看右半边：每一课真正链接到的数学（空心方块）常常比封面写明的（实心方块）更多。** 左半边是封面 meta 行写明的前置课，右半边是 meta 写明的数学课与正文里链接到的数学课。图由 `l00_prereq.py` 读取第 1–19 课的源文件生成，各课改了前置之后重跑一次即可更新。
:::

## 数学补课对照表

下表左列是封面写明的数学前置，右列是正文里实际链接到的小节，点开就是那一节。已经熟悉的内容可以跳过；读课时遇到黄色的“前置知识”框，框里也有同样的链接。

| 课 | 封面写明的数学前置 | 正文链接到的小节 |
|---|---|---|
| [<span style="white-space:nowrap">第 1 课</span>](lesson-01-transformer-anatomy.html) | M01 矩阵与形状、M04 概率与方差、M06 反向传播 | M01 [§1](../math/m01-vectors-matrices.html#p1)、[§2](../math/m01-vectors-matrices.html#p2)、[§3](../math/m01-vectors-matrices.html#p3)、[§4](../math/m01-vectors-matrices.html#p4)、[§5](../math/m01-vectors-matrices.html#p5)、[§6](../math/m01-vectors-matrices.html#p6)；M04 [§2](../math/m04-probability.html#p2)、[§3](../math/m04-probability.html#p3)、[§4](../math/m04-probability.html#p4)；M06 [§3](../math/m06-calculus-backprop.html#p3)、[§5](../math/m06-calculus-backprop.html#p5)、[§7](../math/m06-calculus-backprop.html#p7) |
| [<span style="white-space:nowrap">第 2 课</span>](lesson-02-gpt-training.html) | M01 矩阵与形状、M04 概率与方差、M05 熵与 KL、M06 反向传播、M07 优化器 | M01 [§5](../math/m01-vectors-matrices.html#p5)；M05 [§1](../math/m05-information-theory.html#p1)、[§2](../math/m05-information-theory.html#p2)、[§4](../math/m05-information-theory.html#p4)；M06 [§5](../math/m06-calculus-backprop.html#p5)；M07 [§4](../math/m07-optimization.html#p4)、[§5](../math/m07-optimization.html#p5)；M08 [§2](../math/m08-floating-point.html#p2) |
| [<span style="white-space:nowrap">第 3 课</span>](lesson-03-alignment-rlhf.html) | M04 概率与方差、M05 熵与 KL、M09 策略梯度 | M04 [§4](../math/m04-probability.html#p4)、[§6](../math/m04-probability.html#p6)、[§7](../math/m04-probability.html#p7)；M05 [§4](../math/m05-information-theory.html#p4)、[§5](../math/m05-information-theory.html#p5)；M06 [§1](../math/m06-calculus-backprop.html#p1)；M09 [§2](../math/m09-rl-math.html#p2)、[§3](../math/m09-rl-math.html#p3)、[§5](../math/m09-rl-math.html#p5)；M10 [§1](../math/m10-compute-arithmetic.html#p1)、[§5](../math/m10-compute-arithmetic.html#p5) |
| [<span style="white-space:nowrap">第 4 课</span>](lesson-04-inference-kv-cache.html) | M01 矩阵与形状、M04 概率与方差、M10 算力算术 | M01 [§2](../math/m01-vectors-matrices.html#p2)、[§4](../math/m01-vectors-matrices.html#p4)、[§5](../math/m01-vectors-matrices.html#p5)、[§6](../math/m01-vectors-matrices.html#p6)；M04 [§1](../math/m04-probability.html#p1)、[§2](../math/m04-probability.html#p2)、[§4](../math/m04-probability.html#p4)、[§5](../math/m04-probability.html#p5)；M05 [§1](../math/m05-information-theory.html#p1)；M10 [§1](../math/m10-compute-arithmetic.html#p1)、[§2](../math/m10-compute-arithmetic.html#p2)、[§5](../math/m10-compute-arithmetic.html#p5) |
| [<span style="white-space:nowrap">第 5 课</span>](lesson-05-modern-block.html) | M01 矩阵与形状、M03 旋转与复数 | M01 [§3](../math/m01-vectors-matrices.html#p3)、[§5](../math/m01-vectors-matrices.html#p5)、[§6](../math/m01-vectors-matrices.html#p6)；M02 [§4](../math/m02-rank-svd-lowrank.html#p4)、[§7](../math/m02-rank-svd-lowrank.html#p7)；M03 [§2](../math/m03-rotation-complex.html#p2)、[§3](../math/m03-rotation-complex.html#p3)、[§4](../math/m03-rotation-complex.html#p4)、[§5](../math/m03-rotation-complex.html#p5)、[§6](../math/m03-rotation-complex.html#p6)、[§7](../math/m03-rotation-complex.html#p7) |
| [<span style="white-space:nowrap">第 6 课</span>](lesson-06-deepseek-llm.html) | M01 矩阵与形状、M07 优化器、M10 算力算术 | M01 [§5](../math/m01-vectors-matrices.html#p5)；M04 [§3](../math/m04-probability.html#p3)；M05 [§2](../math/m05-information-theory.html#p2)；M07 [§2](../math/m07-optimization.html#p2)、[§4](../math/m07-optimization.html#p4)、[§5](../math/m07-optimization.html#p5)；M10 [§1](../math/m10-compute-arithmetic.html#p1) |
| [<span style="white-space:nowrap">第 7 课</span>](lesson-07-deepseekmoe.html) | M04 概率与方差、M05 熵与 KL、M06 反向传播、M10 算力算术 | M01 [§2](../math/m01-vectors-matrices.html#p2)、[§7](../math/m01-vectors-matrices.html#p7)；M04 [§2](../math/m04-probability.html#p2)；M05 [§1](../math/m05-information-theory.html#p1)；M06 [§5](../math/m06-calculus-backprop.html#p5)；M10 [§3](../math/m10-compute-arithmetic.html#p3)、[§6](../math/m10-compute-arithmetic.html#p6) |
| [<span style="white-space:nowrap">第 8 课</span>](lesson-08-deepseekmath-grpo.html) | M04 概率与方差、M05 熵与 KL、M09 策略梯度 | M04 [§2](../math/m04-probability.html#p2)、[§5](../math/m04-probability.html#p5)、[§6](../math/m04-probability.html#p6)、[§7](../math/m04-probability.html#p7)；M05 [§4](../math/m05-information-theory.html#p4)、[§5](../math/m05-information-theory.html#p5)、[§6](../math/m05-information-theory.html#p6)；M09 [§1](../math/m09-rl-math.html#p1)、[§2](../math/m09-rl-math.html#p2)、[§3](../math/m09-rl-math.html#p3)、[§5](../math/m09-rl-math.html#p5)；M10 [§5](../math/m10-compute-arithmetic.html#p5) |
| [<span style="white-space:nowrap">第 9 课</span>](lesson-09-v2-mla.html) | M01 矩阵与形状、M02 低秩与 SVD | M01 [§2](../math/m01-vectors-matrices.html#p2)、[§3](../math/m01-vectors-matrices.html#p3)、[§4](../math/m01-vectors-matrices.html#p4)、[§5](../math/m01-vectors-matrices.html#p5)、[§6](../math/m01-vectors-matrices.html#p6)；M02 [§5](../math/m02-rank-svd-lowrank.html#p5)、[§7](../math/m02-rank-svd-lowrank.html#p7)；M03 [§5](../math/m03-rotation-complex.html#p5)；M06 [§4](../math/m06-calculus-backprop.html#p4)；M08 [§5](../math/m08-floating-point.html#p5)；M10 [§3](../math/m10-compute-arithmetic.html#p3) |
| [<span style="white-space:nowrap">第 10 课</span>](lesson-10-v2-moe-training.html) | M04 概率与方差、M06 反向传播、M10 算力算术 | M01 [§4](../math/m01-vectors-matrices.html#p4)、[§7](../math/m01-vectors-matrices.html#p7)；M04 [§2](../math/m04-probability.html#p2)、[§4](../math/m04-probability.html#p4)；M06 [§5](../math/m06-calculus-backprop.html#p5) |
| [<span style="white-space:nowrap">第 11 课</span>](lesson-11-v3-architecture.html) | M04 概率与方差、M05 熵与 KL、M06 反向传播、M08 浮点与量化、M10 算力算术 | M04 [§2](../math/m04-probability.html#p2)、[§4](../math/m04-probability.html#p4)、[§5](../math/m04-probability.html#p5)；M05 [§1](../math/m05-information-theory.html#p1)、[§2](../math/m05-information-theory.html#p2)、[§4](../math/m05-information-theory.html#p4)；M06 [§2](../math/m06-calculus-backprop.html#p2)、[§5](../math/m06-calculus-backprop.html#p5)；M07 [§1](../math/m07-optimization.html#p1)；M08 [§3](../math/m08-floating-point.html#p3)、[§4](../math/m08-floating-point.html#p4)；M10 [§2](../math/m10-compute-arithmetic.html#p2)、[§6](../math/m10-compute-arithmetic.html#p6) |
| [<span style="white-space:nowrap">第 12 课</span>](lesson-12-v3-infrastructure.html) | M08 浮点与量化、M10 算力算术 | M01 [§5](../math/m01-vectors-matrices.html#p5)；M08 [§2](../math/m08-floating-point.html#p2)、[§3](../math/m08-floating-point.html#p3)、[§4](../math/m08-floating-point.html#p4)、[§5](../math/m08-floating-point.html#p5)；M10 [§1](../math/m10-compute-arithmetic.html#p1)、[§5](../math/m10-compute-arithmetic.html#p5)、[§6](../math/m10-compute-arithmetic.html#p6)、[§7](../math/m10-compute-arithmetic.html#p7) |
| [<span style="white-space:nowrap">第 13 课</span>](lesson-13-r1-reasoning.html) | M04 概率与方差、M05 熵与 KL、M09 策略梯度 | M04 [§5](../math/m04-probability.html#p5)；M05 [§4](../math/m05-information-theory.html#p4)、[§5](../math/m05-information-theory.html#p5)、[§6](../math/m05-information-theory.html#p6)；M06 [§1](../math/m06-calculus-backprop.html#p1)；M09 [§2](../math/m09-rl-math.html#p2)、[§3](../math/m09-rl-math.html#p3)、[§4](../math/m09-rl-math.html#p4)、[§6](../math/m09-rl-math.html#p6) |
| [<span style="white-space:nowrap">第 14 课</span>](lesson-14-sparse-attention.html) | M01 矩阵与形状、M05 熵与 KL | M01 [§5](../math/m01-vectors-matrices.html#p5)；M05 [§4](../math/m05-information-theory.html#p4)、[§5](../math/m05-information-theory.html#p5)；M06 [§5](../math/m06-calculus-backprop.html#p5)；M08 [§6](../math/m08-floating-point.html#p6)；M10 [§3](../math/m10-compute-arithmetic.html#p3) |
| [<span style="white-space:nowrap">第 15 课</span>](lesson-15-v3-2-agents.html) | M05 熵与 KL、M08 浮点与量化、M09 策略梯度 | M05 [§4](../math/m05-information-theory.html#p4)、[§6](../math/m05-information-theory.html#p6)；M08 [§6](../math/m08-floating-point.html#p6)、[§7](../math/m08-floating-point.html#p7)；M09 [§3](../math/m09-rl-math.html#p3)、[§5](../math/m09-rl-math.html#p5) |
| [<span style="white-space:nowrap">第 16 课</span>](lesson-16-mhc.html) | M06 反向传播、M11 Sinkhorn 与双随机 | M02 [§3](../math/m02-rank-svd-lowrank.html#p3)、[§4](../math/m02-rank-svd-lowrank.html#p4)；M05 [§1](../math/m05-information-theory.html#p1)；M06 [§3](../math/m06-calculus-backprop.html#p3)、[§6](../math/m06-calculus-backprop.html#p6)、[§7](../math/m06-calculus-backprop.html#p7)；M11 [§3](../math/m11-doubly-stochastic-sinkhorn.html#p3)、[§4](../math/m11-doubly-stochastic-sinkhorn.html#p4)、[§5](../math/m11-doubly-stochastic-sinkhorn.html#p5)、[§7](../math/m11-doubly-stochastic-sinkhorn.html#p7) |
| [<span style="white-space:nowrap">第 17 课</span>](lesson-17-engram-ocr.html) | M01 矩阵与形状、M04 概率与方差 | M01 [§1](../math/m01-vectors-matrices.html#p1)；M04 [§3](../math/m04-probability.html#p3)；M07 [§3](../math/m07-optimization.html#p3)、[§4](../math/m07-optimization.html#p4)、[§7](../math/m07-optimization.html#p7)；M08 [§6](../math/m08-floating-point.html#p6)；M11 [§7](../math/m11-doubly-stochastic-sinkhorn.html#p7) |
| [<span style="white-space:nowrap">第 18 课</span>](lesson-18-v4.html) | M02 低秩与 SVD、M07 优化器 | M01 [§5](../math/m01-vectors-matrices.html#p5)；M02 [§2](../math/m02-rank-svd-lowrank.html#p2)、[§3](../math/m02-rank-svd-lowrank.html#p3)、[§4](../math/m02-rank-svd-lowrank.html#p4)；M03 [§5](../math/m03-rotation-complex.html#p5)、[§6](../math/m03-rotation-complex.html#p6)；M05 [§5](../math/m05-information-theory.html#p5)、[§6](../math/m05-information-theory.html#p6)；M07 [§4](../math/m07-optimization.html#p4)、[§6](../math/m07-optimization.html#p6)；M08 [§3](../math/m08-floating-point.html#p3)、[§7](../math/m08-floating-point.html#p7)；M10 [§5](../math/m10-compute-arithmetic.html#p5)；M11 [§2](../math/m11-doubly-stochastic-sinkhorn.html#p2) |
| [<span style="white-space:nowrap">第 19 课</span>](lesson-19-v4-1-flash.html) | M08 浮点与量化、M10 算力算术 | M01 [§1](../math/m01-vectors-matrices.html#p1)；M03 [§1](../math/m03-rotation-complex.html#p1)；M08 [§3](../math/m08-floating-point.html#p3)、[§7](../math/m08-floating-point.html#p7)；M10 [§1](../math/m10-compute-arithmetic.html#p1) |

如果只想挑最常用的几课先补：M01（矩阵与形状）几乎每一课都用，M04（概率）、M05（熵与 KL）撑起第 3、8、13、15 课的强化学习部分，M08（浮点与量化）与 M10（算力算术）撑起第 12 课之后的系统部分。

## 图的颜色

全课所有的图用同一套颜色，颜色本身就带含义：看到黄褐色就是缓存或记忆，看到橙色就是压缩或量化后的数据。

::: fig figures/l00-colors.svg | 图 0.9 全课统一的图色语义
**每一行左边是颜色，中间是它在图里代表什么，右边是它在各课图中常见的样子。** 颜色写成主题变量，所以在深色模式下会自动换成对应的浅色。图 0.1 的六条色带也按这套语义取色：A 用蓝（查询与注意力），B 用紫（路由打分），C 用橙（低精度数据），D 用绿（奖励与输出），E 用黄褐（记忆），F 用墨色。由 `l00_colors.py` 画出。
:::

## 八个贯穿全课的类比

全课只用这八个类比，每个都在几课里反复出现。每个类比都有它失效的地方，下表把“边界”一栏也列出来，免得把它推得太远。

| 类比 | 映射（一句话） | 在哪里失效 | 主要出现在 |
|---|---|---|---|
| 注意力 = 图书馆检索 | 查询是检索问题，键是书脊标签，值是书的内容，softmax 按匹配度分配阅读时间 | 真检索是挑一本，注意力是把所有书加权混合；标签和内容都是学出来的 | [1](lesson-01-transformer-anatomy.html#p3)、[14](lesson-14-sparse-attention.html)、[18](lesson-18-v4.html) |
| KV cache = 会议纪要本 | 每个发言者说完，秘书把“标签 + 要点”记下，后来的人只翻本子 | 纪要是逐字副本：省重复计算，不省空间 | [4](lesson-04-inference-kv-cache.html#p2)、[9](lesson-09-v2-mla.html)、[18](lesson-18-v4.html)、[19](lesson-19-v4-1-flash.html) |
| MLA = 速记 | 纪要改记成更短的速记符号（潜向量），要用时展开；矩阵吸收是练出直接读速记的眼光 | 速记是学出来的有损压缩；位置信息写不进速记，只能另记一小栏 | [9](lesson-09-v2-mla.html)、[18](lesson-18-v4.html)、[19](lesson-19-v4-1-flash.html) |
| MoE = 医院 | 路由器是分诊台，路由专家是专科医生，共享专家是全科医生 | 分诊按 token 进行；“专科”是训练中自发形成的 | [7](lesson-07-deepseekmoe.html)、[10](lesson-10-v2-moe-training.html)、[11](lesson-11-v3-architecture.html) |
| GRPO = 同题多答、组内排名 | 同一道题写 $G$ 份答卷，按比组平均高多少个标准差给分，不请“预测分数的老师” | 一组全对或全错时优势全为零，不提供信号 | [8](lesson-08-deepseekmath-grpo.html#p4)、[13](lesson-13-r1-reasoning.html)、[15](lesson-15-v3-2-agents.html) |
| 残差流 = 传送带 | 每层是带旁的工位，取件、加工、把增量放回带上；mHC 的调度必须守恒 | 真实残差流是向量加法，“件”不表达方向与正负抵消 | [1](lesson-01-transformer-anatomy.html#p6)、[16](lesson-16-mhc.html) |
| 稀疏注意力 = 先查目录再精读 | lightning indexer 是便宜的目录，top-$k$ 是只翻开得分最高的 $k$ 页 | 目录本身也要逐页扫一遍（仍是平方复杂度，只是很便宜） | [14](lesson-14-sparse-attention.html#p4)、[18](lesson-18-v4.html)、[19](lesson-19-v4-1-flash.html) |
| 低精度数值 = 短尺子 + 分段放大镜 | FP8 是刻度很少的尺子，缩放因子是每一小段的放大镜 | 放大镜只能整体缩放，同一段里大小悬殊的数仍有一个被量得很粗 | [12](lesson-12-v3-infrastructure.html#p3)、[15](lesson-15-v3-2-agents.html#p3)、[19](lesson-19-v4-1-flash.html#p5) |

## 这门课是怎么做出来的

每一课都先有一份**研究笔记**，再有课文。研究笔记放在 `deepseek/reference-docs/lesson-NN-doc.md`，数学课的在 `math/reference-docs/`。它记下四样东西：每个事实的一手来源与读的是哪一版（arXiv 的哪一版、哪个配置文件、哪一页模型卡）；承载事实的那一句原文（短引文）；每个派生数字的计算式（都在 Python 里复算过）；以及勘误和“未能核实”的清单。课文里的数字只能来自研究笔记，研究笔记里标“未能核实”的内容不进课文，进了也要写明。

笔记写完之后还有一轮**对抗式复核**：复核人不沿用原来的结论，而是回到一手来源逐条重查。以本课的研究笔记为例，复核记下了 25 项核对无误的论断和 8 处修改，比如 API 更新日志的条目数原先数错了、V4 技术报告在 HF 上的“更新”其实是删除、若干论文的作者人数把团体名也算成了人、DSpark 的提速区间原先把 V4-Flash 的数推广到了整个 V4。

图也遵守同样的规矩：凡是由数据决定形状的图（曲线、柱子、刻度），都由 `figures/gen/` 下的脚本从数据表计算生成，不手画；本课的九张图对应 `l00_*.py` 九个生成脚本（外加一个共用的绘图工具 `l00_svg.py`）。每一课构建后都要通过检查脚本（公式能否渲染、图的颜色是否都用主题变量、引用是否都能对上），再逐页截图检查。

**怎样留批注。** 课文的源文件是 Markdown，放在 `deepseek/src/` 和 `math/src/`。发现问题时，直接在源文件对应的位置写一条 HTML 注释，构建时它不会出现在网页上；回复写在它下面，或者直接改好并删掉批注：

```text
<!-- 批注(Harry): （示例）这里的 3,514 是只算全局 KV 吗？SWA 状态算不算？ -->
<!-- 回复(Claude): 只算全局 KV；SWA 状态是每个请求的常数，见式 (0.1) 的说明。 -->
```

在仓库根目录运行 `grep -rn "批注(" deepseek/src math/src` 可以列出所有还没处理的批注（上面这个代码块里的示例也会被列出来，认得出即可）。任何涉及事实的修改，先改对应课的研究笔记（写明来源和原文），再改课文。

::: key 本页要点
- 默认按课号顺序读；第 1–19 课约 60 小时阅读、16 小时练习；只追一条主线时，先看图 0.8 的前置。
- 数学补课不必整课读，对照表直接链接到每一课用到的那一节；M01、M04、M05、M08、M10 最常用。
- 颜色有固定含义（图 0.9），八个类比各有边界；课文的每个数字都能追到研究笔记，批注写在源文件的 HTML 注释里。
:::

# 五个贯穿全课的问题 | 读每一课时都可以问的五句话 | 10

这门课的每一课都在讲一个具体的设计，但读多了会发现，能帮你真正读懂一个设计的，总是同样几个问题。这里把它们写下来，读后面任何一课、任何一篇技术报告时都可以拿出来问。每个问题后面列几处课里的例子。

## 一、账单在哪一行？

先问这个设计省下的是什么：显存容量、每一步读的字节、每个 token 的运算、跨卡通信、训练算力，还是训练信号的质量。同一个“省”字，指的可能是完全不同的资源。

- MLA 省的是缓存字节，付的是每个缓存 token 约 3.4 倍的运算（[第 9 课 §3](lesson-09-v2-mla.html#p3)）。
- DSA 不省字节（反而多 20%），省的是长上下文下每一步要读的条目和要算的打分（[第 14 课 §7](lesson-14-sparse-attention.html#p7)）。
- GRPO 省的是一个和策略一样大的价值网络的显存（[第 8 课 §4](lesson-08-deepseekmath-grpo.html#p4)）。
- V4.1 的因果编码器–解码器省的是预填充的计算，解码反而多激活一些参数（[第 19 课 §2](lesson-19-v4-1-flash.html#p2)）。

## 二、这个数在跟谁比，按什么口径？

头条数字几乎都有一个没写在标题里的分母。读到“省了 X%”，先找到比较对象、单位和条件。

- “KV 降 93.3%”是 6 bit 部署的 V2 对 BF16 的 67B；两边同为 BF16 是 82.2%（[§2](#p2)）。
- “27% FLOPs、10% KV”只在 1M 上下文、以外推的 V3.2 为基线、按等效 FP8 计时成立（[第 18 课 §7](lesson-18-v4.html#p7)）。
- DeepSeekMoE 16B 的“40%”是 FLOPs 之比，不是参数之比（[§3](#p3)）。
- “557.6 万美元”不含研究与消融（[§4](#p4)）；“1/4 激活参数”对不上任何一个口径（[§8](#p8)）。

## 三、证据有多硬：证明、实测，还是推测？

同一篇论文里，三种强度的说法常常并排出现，课文会逐一标明。

- **证明**：双随机矩阵的谱范数恰为 1，所以 mHC 的复合映射不会放大（[第 16 课 §4](lesson-16-mhc.html#p4)）；KV cache 的精确性由因果掩码保证（[第 4 课 §2](lesson-04-inference-kv-cache.html#p2)）。
- **实测**：Engram 的 U 形曲线只在两个算力点上测过（[第 17 课 §4](lesson-17-engram-ocr.html#p4)）；RL 提升 Maj@K 而不提升 Pass@K，是 $K\le64$ 下的实测，不是定理（[第 8 课 §8](lesson-08-deepseekmath-grpo.html#p8)）。
- **推测或解释**：“辅助损失伤性能是因为干扰梯度”（[第 11 课 §3](lesson-11-v3-architecture.html#p3)）；“光学遗忘”只是设想，没有做多轮实验（[第 17 课 §7](lesson-17-engram-ocr.html#p7)）；“aha moment”要和长度设置的改动一起读（[第 13 课 §4](lesson-13-r1-reasoning.html#p4)）。

## 四、代价转移到了哪里？

主线上几乎没有免费的改进；省下的东西通常是换成了另一样东西，由另一个部件或另一个场景来付。

- MoE 用显存和通信换计算（[§3](#p3)）；mHC 的参数与算术几乎为零，代价在访存（[§6](#p6)）。
- CSA/HCA 用逐 token 的细节换缓存，1M 上下文“装得下”不等于“用得好”（[第 18 课 §7](lesson-18-v4.html#p7)）。
- V4.1-Flash 用更大的模型、短上下文下更贵的解码和两处近似，换来输入重负载下的存储与搬运（[第 19 课 §8](lesson-19-v4-1-flash.html#p8)）。
- R1 用更长的思考换正确率，推理成本随输出长度线性甚至平方增长（[第 13 课 §8](lesson-13-r1-reasoning.html#p8)）。

## 五、下一个瓶颈在哪里？

每解决一个瓶颈，账单就落到下一行。把整棵科技树连起来看，瓶颈大致是这样移动的：参数与计算绑在一起（MoE 拆开）→ 解码时读 KV 的字节（MLA）→ 长上下文的注意力计算（DSA）→ 百万 token 的缓存与计算（CSA/HCA）→ 智能体负载下缓存的存储与搬运（V4.1）。训练一侧则是：价值网络的显存（GRPO）→ 训练信号的来源（R1 的规则奖励）→ 规模化 RL 的稳定性（V3.2）。

有些设计在瓶颈移动之后被放弃了：V2 的 token 丢弃、V3 的节点受限路由、V4 的哈希路由与 HCA、预训练里的 MTP、稠密注意力热身，以及 R1 试过却没用上的过程奖励模型和蒙特卡洛树搜索。它们为什么被放弃，是[第 20 课](lesson-20-retrospective.html)的主题。

::: key 带着这五个问题去读
1. 账单在哪一行：省的是哪一种资源？
2. 这个数在跟谁比，按什么口径？
3. 证据有多硬：证明、实测，还是推测？
4. 代价转移到了哪里？
5. 下一个瓶颈在哪里？

下一课 [第 1 课](lesson-01-transformer-anatomy.html) 从科技树的根开始：一个标准 GPT 的一次前向传播，后面每一个 DeepSeek 的创新，都是把它的某一个方框换成了更省或更强的版本。
:::

# 练习与答案 | 9 题：概念、计算、推导 | 40

先自己做，再展开答案。需要的数字都在本课前几页和规格总表里；计算题用计算器即可，每一步都可以用几行 Python 核对。这些题检验的正是封面“读完你能……”里的几条：认日期、问口径、分证据、用式 (0.1) 复算。

## 概念题

:::: quiz 练习 1 · 同一个模型，该写哪一天
（a）InstructGPT 在时间线上应该标 2022 年 1 月还是 3 月？（b）V4 论文的编号是 2606.19348，为什么本课把 V4 的日期写成 2026-04-24？（c）DeepSeek-Coder 的 2023-11-02 是什么日期，为什么说它是代理值？（d）主线 F 的改进链里 DeepSeek LLM 写的是 2024-01-05（论文），时间线上却是 2023-11-29，两者矛盾吗？
::: answer 看答案
（a）标 1 月。OpenAI 的公告页与论文 PDF 在 2022-01-27 发出，arXiv v1 晚到 2022-03-04；“3 月”只对 arXiv 成立。本课的规则是模型写第一次公开的日子，所以时间线用 01-27，并注明 arXiv 日期（[§1](#p1)）[@ouyang2022]。

（b）模型写公开发布日：权重与技术报告在 2026-04-24 已在 Hugging Face 公开，arXiv v1 的提交日是 04-26。编号前缀 2606 与提交月份对不上，原因没有从 arXiv 得到确认（“编号按公告月份分配”只是研究笔记的推测），所以引用时写发布日、再注编号最稳妥 [@dsv4]。

（c）它是 GitHub 仓库最早一次提交的日期。DeepSeek-Coder 没有官方公告页，能找到的一手证据只有这次提交和 HF 仓库的创建时间；创建时间只是发布日的下界（仓库常常先私有创建、后公开），所以本课用最早提交作代理，并明确标出 [@dscoder; @ghorg]。

（d）不矛盾。2023-11-29 是 7B / 67B 权重公开的日子，2024-01-05 是报告 arXiv v1 的日子；超参数规模化规律是报告里的贡献，所以主线 F 的表按论文日期记，时间线按模型发布日记。
:::
::::

:::: quiz 练习 2 · 这个数在跟谁比
对下面四句话，各写出比较的对象、单位和成立的条件：（a）“V2 把 KV cache 降了 93.3%”；（b）“1M 上下文下 V4-Pro 只要 V3.2 的 27% FLOPs 和 10% KV”；（c）“DeepSeekMoE 16B 只用了 40% 的计算量”；（d）“V4.1-Flash 只用了 V4-Pro 1/4 的激活参数”。
::: answer 看答案
（a）对象是 DeepSeek LLM 67B；单位是每 token 的 KV 字节；条件是**部署态**：V2 的 KV 平均量化到 6 bit，67B 按 BF16。$1-25{,}920/389{,}120=93.3\%$；两边同为 BF16 只有 $1-69{,}120/389{,}120=82.2\%$（[§2](#p2)）。

（b）对象是 V3.2，可 V3.2 官方只支持 128K，1M 处的 V3.2 是按架构外推的假想点；FLOPs 按“等效 FP8 FLOPs”计，而且只在 1M 这一点上比。按式 (0.4) 推算，KV 是 $5{,}031/48{,}068=10.5\%$，与论文的 10% 吻合（[§2](#p2)、[第 18 课 §7](lesson-18-v4.html#p7)）。

（c）对象是 DeepSeek LLM 7B；单位是每 4K token 的前向 FLOPs（74.4T 对 183.5T，40.5%），不是激活参数之比（[§3](#p3)）。

（d）对象是 V4-Pro 的 49B 激活参数。V4.1 的激活参数要分口径：解码 16B 是 $16/49=0.33$，预填充 8B 是 $8/49=0.16$，都不是 1/4；只有两者的平均 12B 才有 $12/49=0.245$。直接写 8B、16B、49B 最稳妥（[§8](#p8)）。
:::
::::

:::: quiz 练习 3 · 证据有多硬
把下面六句话分成“已证明”“实测”“推算或推断”三类，并说明各自的适用范围：（a）两个双随机矩阵的乘积仍是双随机矩阵；（b）Engram 的最优分配在 75%–80%；（c）辅助损失伤性能，是因为它干扰了主目标的梯度；（d）V4 与 V4.1 的索引键每条 68 字节；（e）V4.1 取消了哈希路由；（f）在 DeepSeekMath 上，RL 提升 Maj@K 而不提升 Pass@K。
::: answer 看答案
（a）**已证明**，对任意两个同阶的双随机矩阵都成立：若 $A\mathbf 1=\mathbf 1$、$\mathbf 1^\top A=\mathbf 1^\top$，$B$ 也一样，则 $AB\mathbf 1=A\mathbf 1=\mathbf 1$，$\mathbf 1^\top AB=\mathbf 1^\top B=\mathbf 1^\top$，非负数相乘相加也仍非负（[第 16 课 §4](lesson-16-mhc.html#p4)）。

（b）**实测**，而且范围很窄：只在两个算力点、固定稀疏度下扫过。论文说 75%–80%，从图上读出的最低观测点在 85% 左右；稳妥的说法是“70%–85% 一带，纯 MoE 明显更差”（[第 17 课 §4](lesson-17-engram-ocr.html#p4)）。

（c）**推断（一种解释）**。实测到的是“系数大了困惑度变差、系数小了负载不均”；“因为干扰梯度”是对这个现象的解释，没有被单独验证（[第 11 课 §3](lesson-11-v3-architecture.html#p3)）。

（d）**推算**。依据是论文说索引器用 OCP 标准的 MXFP4、代码注释说每 32 个数一个 E8M0 缩放：$128\times4/8+128/32=64+4=68$。它让 3,514 与 890 两个论文数字同时复现，是很强的旁证，但“68”本身没有印在论文或 FlashMLA 说明里（式 (0.4)）[@dsv41code]。

（e）**推断**，证据是间接的：V4.1 论文图 3 的图注说全部前馈层都是标准 DeepSeekMoE，配置文件里也没有 V4 那样的哈希层字段；正文没有明说“取消”（[§3](#p3)）。

（f）**实测**：从 Instruct 模型做 RL，在 GSM8K 与 MATH 上、$K\le64$ 的范围内观察到；它描述的是这篇论文里的现象，不是关于 RL 的定理（[第 8 课 §8](lesson-08-deepseekmath-grpo.html#p8)）。
:::
::::

## 计算题

:::: quiz 练习 4 · 两个 16B 级 MoE 的缓存
DeepSeekMoE 16B（16.4B / 2.8B）用 MHA，28 层、16 头 × 128 维；V2-Lite（15.7B / 2.4B）用 MLA，27 层、$d_c=512$、$d_h^R=64$；都按 BF16 存。（a）用式 (0.2) 算两者每 token 的 KV 字节。（b）两者相差几倍？差在哪里？（c）和激活 21B 的 V2（69,120 字节）比，DeepSeekMoE 16B 大几倍？这说明了什么？
::: answer 看答案
（a）DeepSeekMoE 16B：$2\times16\times128\times28\times2=229{,}376$ 字节。V2-Lite：$(512+64)\times27\times2=31{,}104$ 字节。

（b）$229{,}376/31{,}104=7.37$ 倍。两个模型的参数量几乎一样，差别几乎全在注意力：MHA 每层每 token 存 $2\times16\times128=4{,}096$ 个数，MLA 只存 576 个（层数 28 对 27 只差 4%）。

（c）$229{,}376/69{,}120=3.32$ 倍。每 token 的 KV 由注意力的结构、层数和每个数的字节决定，与总参数、激活参数都没有直接关系：激活参数只有 V2 的 $2.8/21\approx13\%$ 的模型，缓存反而是它的 3.3 倍。误读表里 7B 与 67B 的例子是同一个道理（[§8](#p8)）。
:::
::::

:::: quiz 练习 5 · 一个请求要多少显存
只算全局 KV（不算权重和 SWA 状态），$1\text{M}=2^{20}$ token，$128\text{K}=2^{17}$ token，1 GB 取 $10^9$ 字节。（a）V4-Pro（每 token 约 5,031 字节）一个 1M 请求多少 GB？一张 80 GB 的卡放得下几个？（b）V3.2 一个 128K 请求呢？把它和（a）比较。（c）DeepSeek LLM 67B 的 KV 在上下文多长时就会填满 80 GB？
::: answer 看答案
（a）$5{,}031.44\times2^{20}=5.28\times10^{9}$ 字节，即 5.28 GB（4.91 GiB）；$80/5.28\approx15.2$ 个。

（b）$48{,}068\times2^{17}=6.30\times10^{9}$ 字节，即 6.30 GB（5.87 GiB），一张卡约放 12.7 个。V4-Pro 装一个 1M 请求（5.28 GB）比 V3.2 装一个 128K 请求（6.30 GB）还少：上下文长了 8 倍，缓存反而少了 16%。

（c）令 $389{,}120\times T=80\times10^{9}$，得 $T\approx205{,}592$ token，约 200K。67B 实际只支持 4,096 token，一个满长请求只要 $389{,}120\times4{,}096=1.59\times10^{9}$ 字节（1.59 GB）；[§2](#p2) 的“0.2 个 1M 请求”是把它外推到 1M 的假想。
:::
::::

:::: quiz 练习 6 · 3.95 倍的两种拆法
V4-Flash 每 token 3,514.25 字节，V4.1-Flash 890 字节。V4.1 的 4 个 Full 层每 token 共摊到 2.5 条缓存，每条再加 68 字节的索引键。（a）如果条目仍用 V4 的 584 字节格式，全局 KV 是多少？把 3.95 倍拆成“结构 × 精度”。（b）如果改用 V4.1 自己的 FP8 格式（FlashMLA 说明里每条 528 字节），又是多少、怎么拆？（c）两种拆法为什么不同？哪一个“对”？
::: answer 看答案
（a）$2.5\times(584+68)=1{,}630$ 字节；结构 $3{,}514.25/1{,}630=2.16$ 倍，精度 $1{,}630/890=1.83$ 倍，乘积 3.95（第 19 课的拆法，[第 19 课 §8](lesson-19-v4-1-flash.html#p8)）。

（b）$2.5\times(528+68)=1{,}490$ 字节；结构 $3{,}514.25/1{,}490=2.36$ 倍，精度 $1{,}490/890=1.67$ 倍，乘积仍是 3.95（研究笔记的拆法）。

（c）两种拆法用了不同的“中间模型”：（a）的中间模型保留 V4 的条目（448 维 FP8 + 64 维 BF16 的 RoPE + 缩放），（b）的中间模型已经把 RoPE 维也换成 FP8、缩放改成每 32 个数一个 E8M0。先换哪一部分，哪一步就分到那部分的功劳，所以没有唯一“对”的拆法；只要写清中间模型是什么，两种都对。它们的共同结论是：结构（份数与压缩率）这一步比格式那一步省得多。
:::
::::

:::: quiz 练习 7 · 缩小的速度
用规格总表里的发布日与每 token KV：DeepSeek LLM 67B（2023-11-29，389,120 字节）、V3（2024-12-26，70,272 字节）、V4.1-Flash（2026-09-10，890 字节）。（a）两段各隔多少天？（b）按“每年缩小几倍”$r^{365/\Delta t}$（$r$ 是这一段缩小的倍数，$\Delta t$ 是天数）算，两段各是多少？和全程的 8.9 倍 / 年比，说明了什么？
::: answer 看答案
（a）67B → V3：393 天；V3 → V4.1-Flash：623 天；合计 1,016 天，与 [§1](#p1) 一致。

（b）67B → V3：$r=389{,}120/70{,}272=5.54$，$5.54^{365/393}=4.9$ 倍 / 年。V3 → V4.1-Flash：$r=70{,}272/890=79.0$，$79.0^{365/623}=12.9$ 倍 / 年。全程的 $437.2^{365/1016}=8.9$ 倍 / 年落在两者之间：缩小在 V3 之后明显加快，主要来自 V3.2 → V4-Flash 的序列方向压缩（÷13.7）和 V4.1 的跨层共享与 FP4（÷3.95）。这只是按发布日做的年化，是笔者的口径，不是什么规律，不能拿来外推下一代。
:::
::::

## 推导题

:::: quiz 练习 8 · 从四个因子到“2.25 组 GQA”
（a）从式 (0.1) 出发，写出 MHA、GQA、MQA、MLA 四种注意力的 $n_{\text{copy}}$、$m$、$n_{\text{dim}}$，推出式 (0.2)。（b）V2 取 $d_c=4d_h$、$d_h^R=d_h/2$。证明 MLA 的缓存等于只有 2.25 组的 GQA。（c）GQA 的组数只能是整数，MLA 却“相当于” 2.25 组，这说明两者的缓存在结构上有什么不同？
::: answer 看答案
（a）四种注意力都是每层存一份、每个 token 一条：$n_{\text{copy}}=l$，$m=1$。每条的数：MHA 每个头存一个键、一个值，$n_{\text{dim}}=2n_hd_h$；GQA 每组共用一对，$n_{\text{dim}}=2n_gd_h$；MQA 是 $n_g=1$，$n_{\text{dim}}=2d_h$；MLA 存一个潜向量加一个所有头共用的 RoPE 键，$n_{\text{dim}}=d_c+d_h^R$。代入 $B_{kv}=n_{\text{copy}}\cdot\frac1m\cdot n_{\text{dim}}\cdot b$，就得到 (0.2) 的 $2n_gd_hlb$ 与 $(d_c+d_h^R)lb$。

（b）$d_c+d_h^R=4d_h+\tfrac12d_h=\tfrac92d_h=2\times2.25\times d_h$。与 GQA 的 $2n_gd_h$ 对照，就是 $n_g=2.25$。代入 V2 的 $d_h=128$：$(512+64)/(2\times128)=2.25$ [@dsv2]。

（c）GQA 的条目是若干对完整的键和值，每个查询头直接拿来用，所以只能按整组计数。MLA 的条目是一个 512 维的潜向量，128 个头的键和值都由它经上投影展开（推理时把上投影吸收进查询），它本身不是任何一个头的键或值；“2.25 组”只是等字节的换算，不代表 MLA 有 2.25 组头。这种共享的代价见 [§2](#p2)：每个缓存 token 约多 3.4 倍运算，RoPE 还得另存一栏。
:::
::::

:::: quiz 练习 9 · UE8M0 为什么不溢出，平均用到多少
一组数的最大绝对值是 $a$，E4M3 能表示的最大值是 448，UE8M0 缩放取 $s=2^{\lceil\log_2(a/448)\rceil}$（式 (0.9)）。（a）证明缩放后组里每个数的绝对值都不超过 448。（b）证明组里最大的数缩放后落在 $(224,448]$，也就是只用到 E4M3 范围的一半到全部。（c）设 $\log_2(a/448)$ 的小数部分在 $[0,1)$ 上均匀分布，求利用率 $u=a/(448s)$ 的期望。（d）$a=1{,}000$ 时 $s$ 与 $u$ 各是多少？如果改成向下取整呢？
::: answer 看答案
（a）记 $y=\log_2(a/448)$。因为 $\lceil y\rceil\ge y$，$s=2^{\lceil y\rceil}\ge2^{y}=a/448$，所以 $a/s\le448$；组里其余数的绝对值都不超过 $a$，缩放后也都不超过 448。

（b）又因为 $\lceil y\rceil<y+1$，$s<2^{y+1}=2a/448$，所以 $a/s>224$。于是 $a/s\in(224,448]$，利用率 $u\in(\tfrac12,1]$。

（c）令 $v=\lceil y\rceil-y$，则 $u=2^{-v}$，$v$ 在 $[0,1)$ 上均匀分布。$\E[u]=\int_0^12^{-v}\,dv=(1-2^{-1})/\ln2=1/(2\ln2)\approx0.721$，这就是式 (0.9) 下面说的“约 72%”。

（d）$a/448=2.232$，$\log_2 2.232=1.158$，向上取整为 2，$s=4$；$a/s=250$，$u=250/448=0.558$。若向下取整，$s=2$，$a/s=500>448$，就溢出了。所以规则规定向上取整：$2^{\lceil y\rceil}$ 是所有不溢出的 2 的幂里最小的一个，也就是精度最好的一档（[第 15 课 §3](lesson-15-v3-2-agents.html#p3)）。
:::
::::

```refs
- id: vaswani2017
  title: "Attention Is All You Need"
  a: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017), NIPS 2017, arXiv:1706.03762
  url: https://arxiv.org/abs/1706.03762
  finding: 2017-06-12 提交 arXiv，提出完全基于注意力的 Transformer（多头注意力 MHA）。
  matters: 科技树的根节点，时间轴从这一天起算（到 V4.1-Flash 共 3,377 天）。
- id: radford2018
  title: "Improving Language Understanding by Generative Pre-Training"
  a: Radford, Narasimhan, Salimans, Sutskever (2018), OpenAI technical report（无 arXiv）
  url: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
  finding: OpenAI 于 2018-06-11 公告的 GPT-1：仅解码器 Transformer 的预训练加微调。
  matters: 时间线 GPT 侧的第一个节点；它没有 arXiv 版，日期取公告页。
- id: radford2019
  title: "Language Models are Unsupervised Multitask Learners"
  a: Radford, Wu, Child, Luan, Amodei, Sutskever (2019), OpenAI technical report（无 arXiv）
  url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  finding: 2019-02-14 公告的 GPT-2（最大 1.5B），把 LayerNorm 移到子层输入处（Pre-LN）。
  matters: 时间线节点，也是主线 E 上 Pre-LN 卡片的出处。
- id: brown2020
  title: "Language Models are Few-Shot Learners"
  a: Brown et al. (2020), NeurIPS 2020, arXiv:2005.14165
  url: https://arxiv.org/abs/2005.14165
  finding: 175B 的 GPT-3，全部模型上下文 2048，2020-05-28 提交。
  matters: 图 0.2 的参照线：按 MHA、每个数 2 字节，每 token KV 为 4,718,592 字节。
- id: ouyang2022
  title: "Training language models to follow instructions with human feedback"
  a: Ouyang et al. (2022), NeurIPS 2022, arXiv:2203.02155v1
  url: https://arxiv.org/abs/2203.02155
  finding: InstructGPT 的三步法 SFT → 奖励模型 → PPO，在预训练 GPT-3 上继续训练；公告与 PDF 2022-01-27，arXiv 2022-03-04。
  matters: 主线 D 的 RLHF 节点与模型行的 GPT-3 → InstructGPT 血缘；也用来纠正“InstructGPT 是 2022 年 3 月”的说法。
- id: oaiwb
  kind: artefact
  title: "OpenAI announcements via Wayback Machine (GPT-1, GPT-2, InstructGPT, ChatGPT)"
  a: OpenAI (2018–2022)
  url: http://web.archive.org/web/20221205235144/https://openai.com/blog/chatgpt/
  finding: 四篇公告的页面日期分别为 2018-06-11、2019-02-14、2022-01-27、2022-11-30。
  matters: GPT 侧时间线可存档的一手日期来源（openai.com 直连返回 403）。
- id: dscoder
  title: "DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence"
  a: Guo, Zhu, Yang, Xie, Dong, Zhang, et al. (2024), preprint, arXiv:2401.14196v2
  url: https://arxiv.org/abs/2401.14196
  finding: 1.3B–33B 代码模型，2T tokens 从头训练，16K 窗口与 FIM 填空；§5 从 DeepSeek LLM 7B 续训出 Coder-v1.5。
  matters: DeepSeek 最早公开的语言模型，也是 DeepSeekMath 血缘的起点。
- id: ghorg
  kind: artefact
  title: "deepseek-ai organization on GitHub (39 repositories)"
  a: DeepSeek-AI (2023–2026)
  url: https://github.com/deepseek-ai
  finding: 各仓库的创建日与最早提交；V2 README 的 “2024.05.06 / 05.16” 发布条目。
  matters: 为没有新闻页的早期版本（Coder、LLM、MoE、V2）提供发布日期证据。
- id: dsllm
  title: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"
  a: DeepSeek-AI (2024), preprint, arXiv:2401.02954v1
  url: https://arxiv.org/abs/2401.02954
  finding: 7B/67B、2T tokens；67B 为 95 层、GQA-8、上下文 4096（表 2）；超参数与模型/数据分配的规模化规律。
  matters: 规格总表第一行，也是 V4.1 论文图 1(b) 中 389,120 字节那根柱子的出处。
- id: dsv41
  title: "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"
  a: DeepSeek-AI (2026), preprint, arXiv:2609.19969v1
  url: https://arxiv.org/abs/2609.19969
  finding: 552B backbone（另 196B Engram）、prefill 8B / decode 16B 激活，CED + CSA2 + FP4 KV 使全局 KV 为 890 字节/token，45T 多模态 tokens。
  matters: 课程时间线的终点；图 1(b) 的四代 KV 数字是本课式 (0.1)–(0.5) 复算的对照。
- id: dsnews
  kind: artefact
  title: "DeepSeek API News (release posts, 2024-07 to 2026-09)"
  a: DeepSeek (2024–2026)
  url: https://api-docs.deepseek.com/news/news260910
  finding: 逐篇给出 V2.5、R1-Lite、V3、R1、V3-0324、R1-0528、V3.1、Terminus、V3.2-Exp、V3.2、V4 预览、V4-Pro 正式版、V4-Flash-Vision-Exp、V4.1-Flash 的发布日与要点。
  matters: 时间线“公开发布日”的首选来源。
- id: dsv2
  title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  a: DeepSeek-AI (2024), preprint, arXiv:2405.04434v5
  url: https://arxiv.org/abs/2405.04434
  finding: MLA + DeepSeekMoE，236B/21B、8.1T、128K；表 1 给出四种注意力的 KV 通式；附录 B 给出 V2-Lite。
  matters: 式 (0.2) 的 MLA 一项、规格表 V2 与 V2-Lite 两行，以及“93.3% 需要 6 bit”的勘误依据。
- id: dsv4
  title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
  a: DeepSeek-AI (2026), preprint, arXiv:2606.19348v1
  url: https://arxiv.org/abs/2606.19348
  finding: 预览版 V4-Pro 1.6T/49B 与 V4-Flash 284B/13B，CSA/HCA 混合注意力、mHC、Muon，1M 上下文下 Pro 仅需 V3.2 的 27% FLOPs 与 10% KV。
  matters: 规格表 V4 两行与式 (0.4) 的结构依据，也是“编号 2606 却在 4 月提交”这一日期陷阱的例子。
- id: flashmla
  kind: artefact
  title: "FlashMLA README (KV cache byte formats for V3.2 / V4 / V4.1)"
  a: DeepSeek-AI (2025–2026)
  url: https://github.com/deepseek-ai/FlashMLA
  finding: V3.2 每 token 每层 656 字节；V4 584 字节；V4.1 FP4 288 字节。
  matters: 式 (0.3)–(0.5) 的字节常数来源，使 48,068、3,514、890 可以复现。
- id: dsv41code
  kind: artefact
  title: "DeepSeek-V4.1-Flash inference/model.py"
  a: DeepSeek-AI (2026)
  url: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py
  finding: 注释写明压缩 KV 用每 16 个一组的 E4M3 缩放、indexer 用每 32 个一组的 E8M0 缩放。
  matters: 推出索引键 68 字节，是 890 与 3,514 能同时复现的关键一环。
- id: dsv32
  title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
  a: DeepSeek-AI (2025), preprint, arXiv:2512.02556v1
  url: https://arxiv.org/abs/2512.02556
  finding: 以 V3.1-Terminus 为起点加入 DSA（top-2048），续训 2.1B + 943.7B tokens，上下文 128K。
  matters: 规格表 V3.2 行与 48,068 字节的结构依据。
- id: dsmath
  title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
  a: Shao, Wang, Zhu, Xu, Song, Bi, et al. (2024), preprint, arXiv:2402.03300v3
  url: https://arxiv.org/abs/2402.03300
  finding: 从 DeepSeek-Coder-Base-v1.5 7B 续训 120B 数学 token，并提出 GRPO。
  matters: 主线 D 上 GRPO 的首次出现，也是模型行 DeepSeek LLM → DeepSeekMath 血缘的依据。
- id: dsr1
  title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
  a: DeepSeek-AI (2025), Nature 645:633–638, arXiv:2501.12948v2
  url: https://arxiv.org/abs/2501.12948
  finding: 在 V3-Base 上纯 RL 得到 R1-Zero，再经多阶段得到 R1；表 7 训练成本 147K H800 小时（$294K）；附录 G.2 记录 PRM/MCTS 未成功。
  matters: 时间线 2025-01-20 主干节点，解释 V3 到 R1 只隔 25 天。
- id: zhu2025hc
  title: "Hyper-Connections"
  a: Zhu, Huang, Huang, Zeng, Mao, Wu, Min, Zhou (2025), ICLR 2025, arXiv:2409.19606
  url: https://arxiv.org/abs/2409.19606
  finding: 把残差流复制成 n 条，用可学习的矩阵同时决定层的读出、写回与流间混合；n = 4 时稳定优于 Pre-Norm 残差。
  matters: 主线 E 上 mHC 要约束的对象，也是“残差捷径几年没人动”之后的第一个改动。
- id: engram
  title: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"
  a: Cheng, Zeng, Dai, Chen, Wang, Xie, et al. (2026), ACL 2026, arXiv:2601.07372v2
  url: https://arxiv.org/abs/2601.07372
  finding: 以 N-gram 哈希查表作条件记忆，发现 MoE 与记忆的参数分配呈 U 形，放大到 Engram-27B。
  matters: 主线 E 的节点，自称与 MoE 互补的新稀疏轴；V4.1 以 196B 参数把它用进生产模型。
- id: shazeer2019
  title: "Fast Transformer Decoding: One Write-Head is All You Need"
  a: Noam Shazeer (2019), arXiv preprint, arXiv:1911.02150v1
  url: https://arxiv.org/abs/1911.02150
  finding: 提出让所有头共享一组 K、V 的多查询注意力（MQA），把解码的访存量缩小一个头数的倍数，质量仅略降。
  matters: 主线 A 上第一个“减少 KV 头”的节点。
- id: ainslie2023
  title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
  a: Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (2023), EMNLP 2023, arXiv:2305.13245v3
  url: https://arxiv.org/abs/2305.13245
  finding: 把查询头分组、每组共享一对 K、V；GQA-8 在 T5-XXL 上质量接近 MHA、速度接近 MQA。
  matters: DeepSeek LLM 67B 所用注意力的定义，主线 A 的节点。
- id: su2021roformer
  title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  a: Su, Lu, Pan, Murtadha, Wen, Liu (2021; Neurocomputing 568:127063, 2024), arXiv:2104.09864v5
  url: https://arxiv.org/abs/2104.09864
  finding: 从“q·k 只依赖相对位置”的要求出发，用复数旋转构造位置编码，频率为几何级数。
  matters: 主线 A 的 RoPE 节点；它与 MLA 矩阵吸收的冲突逼出了解耦 RoPE。
- id: nsa
  title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
  a: Yuan, Gao, Dai, Luo, Zhao, Zhang, et al. (2025), ACL 2025 (Best Paper), arXiv:2502.11089v2
  url: https://arxiv.org/abs/2502.11089
  finding: 压缩、选择、滑窗三分支的可训练稀疏注意力，在 27B/3B 骨干上不输全注意力。
  matters: 主线 A 从 MLA 走向 DSA、CSA 的中间节点，时间线 2025-02-16。
- id: acl2025awards
  kind: artefact
  title: "ACL 2025 Awards page"
  a: ACL (2025)
  url: https://2025.aclweb.org/program/awards/
  finding: NSA 列于 Best Paper 四篇之一。
  matters: 支持“NSA 获 ACL 2025 最佳论文”的表述。
- id: dsmoe
  title: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"
  a: Dai, Deng, Zhao, Xu, Gao, Chen, et al. (2024), preprint, arXiv:2401.06066v1
  url: https://arxiv.org/abs/2401.06066
  finding: 细粒度专家切分 + 共享专家隔离；16B 版 28 层、2 共享 + 64 路由激活 6 个、16.4B/2.8B、2T tokens。
  matters: 主线 B 的起点，式 (0.6) 的结构来源与规格表第二行。
- id: dsv3
  title: "DeepSeek-V3 Technical Report"
  a: DeepSeek-AI (2024), preprint, arXiv:2412.19437v2
  url: https://arxiv.org/abs/2412.19437
  finding: 671B/37B、61 层、1 + 256 专家激活 8 个、14.8T tokens，无辅助损失均衡、MTP、FP8 训练与 DualPipe；全部正式训练 2.788M H800 小时（$5.576M，不含此前的研究与消融）。
  matters: 式 (0.6)(0.7)(0.13) 的出处，也是图 0.5 第一行与规格表 V3 行的依据。
- id: alf
  title: "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"
  a: Wang, Gao, Zhao, Sun, Dai (2024), preprint, arXiv:2408.15664v1
  url: https://arxiv.org/abs/2408.15664
  finding: 用按负载更新的专家偏置代替辅助损失，在至多 3B 参数、200B tokens 的模型上验证。
  matters: V3 起无辅助损失均衡（式 (0.7)）的出处。
- id: shazeer2017
  title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
  a: Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean (2017), ICLR 2017, arXiv:1701.06538v1
  url: https://arxiv.org/abs/1701.06538
  finding: 用带噪声的 top-k 稀疏门控把专家数扩大到数千，并用辅助损失防止路由坍缩。
  matters: 主线 B 在科技树上的第一个节点，也是“路由坍缩自我强化”的出处。
- id: gshard2020
  title: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"
  a: Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen (2020/2021), ICLR 2021, arXiv:2006.16668v1
  url: https://arxiv.org/abs/2006.16668
  finding: 在 Transformer 中用 top-2 MoE，配合专家容量与 f·P 形式的辅助损失，训练 600B 参数的翻译模型。
  matters: DeepSeekMoE 对照的“传统 top-K MoE”代表，主线 B 的节点。
- id: switch2021
  title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
  a: Fedus, Zoph, Shazeer (2021/2022), JMLR 23(120):1–39, arXiv:2101.03961v3
  url: https://arxiv.org/abs/2101.03961
  finding: 每个 token 只路由到 1 个专家，给出容量因子与 α·N·Σf_iP_i 的均衡损失，训练到 1.6T 参数。
  matters: 均衡损失的标准写法，主线 B 的节点。
- id: fp8formats
  title: "FP8 Formats for Deep Learning"
  a: Micikevicius, Stosic, Burgess, Cornea, Dubey, Grisenthwaite et al. (2022), arXiv:2209.05433 (v2 read)
  url: https://arxiv.org/abs/2209.05433
  finding: 定义 E4M3（最大 448）与 E5M2（最大 57,344）两种 FP8 编码。
  matters: 式 (0.8) 中 448 这个常数的来历，主线 C 的节点。
- id: fireflyer
  title: "Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning"
  a: An, Bi, Chen, Chen, Deng, Ding, et al. (2024), SC'24, arXiv:2408.14158v2
  url: https://arxiv.org/abs/2408.14158
  finding: 1 万张 PCIe A100 的 Fire-Flyer 2 集群，配套 HFReduce 与 3FS 文件系统。
  matters: 主线 C 的基础设施起点。
- id: openinfra
  kind: artefact
  title: "open-infra-index README (DeepSeek Open-Source Week, Feb 2025)"
  a: DeepSeek-AI (2025)
  url: https://github.com/deepseek-ai/open-infra-index
  finding: 2025-02-24 起连续开源 FlashMLA、DeepEP、DeepGEMM、DualPipe/EPLB、3FS；Day 6 公布推理系统的理论成本利润率 545%。
  matters: 时间线“开源周”节点与主线 C 的制品清单。
- id: insights
  title: "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures"
  a: Zhao, Deng, Ruan, Dai, Gao, Li, et al. (2025), ISCA '25 Industry Track, arXiv:2505.09343v2
  url: https://arxiv.org/abs/2505.09343
  finding: 总结 V3 在 2,048 张 H800 上的软硬件协同，表 1 给出 V3 每 token KV 70.272 KB。
  matters: 主线 C 的节点，并交叉验证图 0.2 中 V3 的 70,272 字节。
- id: dsv31card
  kind: artefact
  title: "DeepSeek-V3.1 model card"
  a: DeepSeek-AI (2025)
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
  finding: V3.1 用一个模型提供思考与非思考两种模式，V3.1-Base 在 V3 上续训 839B tokens，并采用 UE8M0 FP8 缩放格式。
  matters: 主线 C（UE8M0）与主线 D（一模两式）两个节点的出处。
- id: dsec
  title: "DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale"
  a: Huang, Tang, Chen, Liu, Chen, Cheng, et al. (2026), preprint, arXiv:2609.22978v1
  url: https://arxiv.org/abs/2609.22978
  finding: 统一接口暴露多类沙箱后端，每天约 300 万个沙箱，服务智能体强化学习。
  matters: 时间线上最新的 DeepSeek 论文（2026-09-19），本课只在主线 C 的表里提到。
- id: schulman2017ppo
  title: "Proximal Policy Optimization Algorithms"
  a: Schulman, Wolski, Dhariwal, Radford, Klimov (2017), arXiv preprint, arXiv:1707.06347v2
  url: https://arxiv.org/abs/1707.06347
  finding: 提出裁剪概率比的替代目标，用多轮小批量复用样本，近似信赖域。
  matters: 主线 D 在科技树上的第一个节点，GRPO 就是在它上面去掉价值网络。
- id: rafailov2023dpo
  title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
  a: Rafailov, Sharma, Mitchell, Ermon, Manning, Finn (2023), NeurIPS 2023, arXiv:2305.18290v3
  url: https://arxiv.org/abs/2305.18290
  finding: 利用 KL 约束目标的闭式最优策略，把奖励重参数化进策略，用一个二分类损失直接训练策略。
  matters: 主线 D 的节点；DeepSeek LLM 的对齐用的正是 SFT + DPO。
- id: gloeckle2024
  title: "Better & Faster Large Language Models via Multi-token Prediction"
  a: Gloeckle, Youbi Idrissi, Rozière, Lopez-Paz, Synnaeve (2024), ICML 2024, arXiv:2404.19737v1
  url: https://arxiv.org/abs/2404.19737
  finding: 共享主干上接 n 个并行输出头同时预测后面 n 个 token，大模型的代码生成明显受益。
  matters: V3 MTP 的直接灵感与对照组，主线 D 的节点。
- id: dspark
  title: "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
  a: Cheng, Yu, Shao, Li, Xiong, Qian, et al. (2026), preprint, arXiv:2607.05147v1
  url: https://arxiv.org/abs/2607.05147
  finding: 半自回归草稿 + 置信度调度验证，同等总吞吐下比 MTP-1 基线把单用户生成速度提升 60%–85%（V4-Flash）与 57%–78%（V4-Pro）。
  matters: 解释 V4.1 为何在预训练中去掉 MTP。
- id: mhc
  title: "mHC: Manifold-Constrained Hyper-Connections"
  a: Xie, Wei, Cao, Zhao, Deng, Li, et al. (2025), preprint, arXiv:2512.24880v2
  url: https://arxiv.org/abs/2512.24880
  finding: 把超连接的残差矩阵约束为双随机矩阵，27B 模型上复合增益从约 3000 降到约 1.6，n = 4 时训练开销 6.7%。
  matters: 主线 E 的核心节点，图 0.6 右侧两个菱形的出处。
- id: dsocr
  title: "DeepSeek-OCR: Contexts Optical Compression"
  a: Wei, Sun, Li (2025), preprint, arXiv:2510.18234v1
  url: https://arxiv.org/abs/2510.18234
  finding: 文本 token 不超过视觉 token 10 倍时 OCR 精度约 97%，20 倍时约 60%。
  matters: 主线 E“视觉压缩上下文”的节点。
- id: dsocr2
  title: "DeepSeek-OCR 2: Visual Causal Flow"
  a: Wei, Sun, Li (2026), preprint, arXiv:2601.20552v1
  url: https://arxiv.org/abs/2601.20552
  finding: DeepEncoder V2 按图像语义动态重排视觉 token 再交给语言模型。
  matters: 视觉压缩支线的延续节点。
- id: xiong2020
  title: "On Layer Normalization in the Transformer Architecture"
  a: Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, Liu (2020), ICML 2020, arXiv:2002.04745
  url: https://arxiv.org/abs/2002.04745
  finding: 在简化假设下证明初始化时 Post-LN 顶层梯度比 Pre-LN 大，Pre-LN 可以去掉 warm-up。
  matters: 主线 E 上 Pre-LN 之后的节点。
- id: kaplan2020
  title: "Scaling Laws for Neural Language Models"
  a: Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei (2020), arXiv:2001.08361
  url: https://arxiv.org/abs/2001.08361
  finding: 损失随参数量、数据量与算力呈幂律，给定算力时主要加大模型。
  matters: 主线 F 的起点之一，DeepSeek LLM 要调和的两种结论之一。
- id: hoffmann2022
  title: "Training Compute-Optimal Large Language Models"
  a: Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai, Rutherford, et al. (2022), NeurIPS 2022, arXiv:2203.15556
  url: https://arxiv.org/abs/2203.15556
  finding: 三种方法都显示最优模型规模与训练 token 应随算力等比例增长。
  matters: “约 20 tokens/参数”经验法则的来源，图 0.7 的参照线。
- id: loshchilov2019
  title: "Decoupled Weight Decay Regularization"
  a: Loshchilov, Hutter (2019), ICLR 2019, arXiv:1711.05101
  url: https://arxiv.org/abs/1711.05101
  finding: 证明对 Adam 而言 L2 正则与权重衰减不等价，提出把衰减解耦的 AdamW。
  matters: DeepSeek 从 LLM 到 V3 一直用的优化器，主线 F 的节点。
- id: jordan2024muon
  kind: artefact
  title: "Muon: An optimizer for hidden layers in neural networks"
  a: Keller Jordan et al. (2024-12-08), blog post
  url: https://kellerjordan.github.io/posts/muon/
  finding: 定义 Muon（动量 + 牛顿–舒尔茨正交化），建议嵌入与输出层仍用 AdamW。
  matters: V4 所用优化器的来源，主线 F 的节点。
- id: hforg
  kind: artefact
  title: "deepseek-ai organization on Hugging Face (model list, configs, model cards, commit histories)"
  a: DeepSeek-AI (2023–2026)
  url: https://huggingface.co/deepseek-ai
  finding: 各仓库的建库时间，主线模型 config.json 的层数、维度、专家数、compress_ratios、kv_source_layer_ids 等字段。
  matters: 规格表里“config”一类来源，也用来从配置复现 3,514 与 890。
- id: dschangelog
  kind: artefact
  title: "DeepSeek API Change Log"
  a: DeepSeek (2024–2026)
  url: https://api-docs.deepseek.com/updates/
  finding: 2024-05-17 至 2026-09-10 共 21 个日期条目，含 V2-0517、Coder-V2-0614、V2-0628、Coder-V2-0724、V4-Flash-0731 等版本切换。
  matters: 补齐没有新闻页的版本日期，也是“没有 R2”这一判断的依据之一。
```
