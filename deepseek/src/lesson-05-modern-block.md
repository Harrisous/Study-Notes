---
title: 第 5 课 · 现代 Transformer 积木：RoPE、RMSNorm、SwiGLU 与上下文扩展
short: 现代 Transformer 积木
kicker: DeepSeek 科技树 · 第 5 课
description: 从“注意力分数只应依赖相对位置”出发用复数推导 RoPE，讲清多频率“时钟”、远程衰减的真相、RMSNorm 与 LayerNorm 的差别、SwiGLU 的 8d/3 参数匹配，以及位置插值、NTK-aware、YaRN 如何把 DeepSeek-V2/V3 从 4K 扩到 128K。
lede: 从 LLaMA（2023）起，许多开源大模型的 Transformer 块都换了四块“积木”：位置不再加在输入上，而是每层把查询和键**旋转**一个角度；LayerNorm 变成只做缩放的 RMSNorm；前馈层换成带门控的 SwiGLU；偏置全部去掉。DeepSeek 从第一个模型起就用这套积木，又用 YaRN 把只在 4K 长度上预训练的模型扩到 128K。这一课把每一块从“为什么需要”讲到“公式怎么推”，再到“DeepSeek 用的真实数字”。
meta: 预计阅读 160 分钟，练习另需约 50 分钟 · 前置：第 1、4 课，数学 M01、M03
index: index.html
prev: lesson-04-inference-kv-cache.html | 第 4 课 · KV cache 与解码
next: lesson-06-deepseek-llm.html | 第 6 课 · DeepSeek LLM
---

## 这一课回答什么问题

第 1 课拆开了 GPT-2 的一个块：词嵌入加上位置嵌入，进入 LayerNorm、多头注意力、GELU 前馈层，层层带偏置。把 2023 年以后的开源模型（LLaMA、DeepSeek LLM、DeepSeek-V2/V3）打开，会发现块的骨架没变，但四个零件都换了。DeepSeek LLM 的技术报告直接写明，它的“微观设计”基本沿用 LLaMA：Pre-Norm 加 RMSNorm、SwiGLU 前馈层、旋转位置编码 [@deepseek2024llm]。这一课逐个回答：

- 位置信息为什么要改成“旋转”？怎样从“注意力分数只应依赖两个 token 的**相对**距离”这一个要求，一步一步推出旋转矩阵？
- 旋转的频率为什么取 $\theta_i=10000^{-2i/d_h}$？“远程衰减”到底被证明了什么，没被证明什么？
- RMSNorm 比 LayerNorm 少做了什么，丢掉了什么？SwiGLU 的隐藏维为什么是 $\tfrac83d$，LLaMA-7B 的 11008 又是怎么来的？
- 只在 4K 长度上预训练的模型，直接读 100K 为什么会崩？位置插值（PI）、NTK-aware、YaRN 各自怎样“拨慢时钟”？DeepSeek-V2/V3 的 YaRN 设置里，64 维 RoPE 中哪些维被插值、哪些维原封不动？

## 这一课在科技树上的位置

上一课（[第 4 课](lesson-04-inference-kv-cache.html)）算清了解码时每个 token 要读写多少 KV cache，结论是缓存随上下文长度线性增长。它没有回答两个问题：位置信息是怎样进入缓存里那些键的？模型到底能读多长？这一课回答这两个问题，属于“注意力与 KV cache 经济学”这条主线（主线 A）。下一课（[第 6 课](lesson-06-deepseek-llm.html)）的 DeepSeek LLM 就是用本课的积木搭起来的；到[第 9 课](lesson-09-v2-mla.html)，本课的 RoPE 会给 MLA 的“矩阵吸收”制造一个麻烦，DeepSeek-V2 用“解耦 RoPE”绕开它，而 YaRN 恰好只作用在解耦出来的那一小块上。本课最后一页先把这个麻烦摆出来。

## 读完你能……

1. 从“$\langle f(\mathbf q,t),f(\mathbf k,j)\rangle$ 只依赖 $t-j$”出发，用复数写出 RoPE 的完整推导，并指出其中哪几步是**选择**、哪几步是**推论**。
2. 算出任意 $d_h$ 与底数 $b$ 下 RoPE 各对维度的频率与波长，判断哪些维度在预训练长度内“转不满一圈”，并解释远程衰减为什么只是一个上界的平均趋势。
3. 手算 RMSNorm 与 LayerNorm，说出两者的不变性差别；用参数匹配推出 SwiGLU 的 $\tfrac83d$，并复算 LLaMA-7B 的 11008 与每层参数量。
4. 写出 PI、NTK-aware、YaRN 对每一对频率的缩放公式；对 DeepSeek 的设置（$d_h^R=64$、$L_{ctx}=4096$、$s=40$、$\alpha=1$、$\beta=32$）算出三段划分（11 / 12 / 9 对）与注意力温度（V2：$\mu^2=1.5896$；V3：$1.8739$）。
5. 解释为什么 RoPE 与“缓存压缩向量再吸收矩阵”不兼容，为第 9 课的解耦 RoPE 做好准备。

::: note 本课的记号
- 位置：$t$ 是当前（查询）位置，$j$ 是被看的（键）位置；因果注意力里 $j\le t$，相对距离写 $\Delta=t-j\ge0$。位置从 0 数起。
- $d_h$：每个头的维度；DeepSeek-V2/V3 中承载 RoPE 的那一小段每头维度写 $d_h^R$（$=64$）。$d$ 是模型宽度，不要和 $d_h$ 混。
- RoPE 把 $d_h$ 维拆成 $d_h/2$ 对，第 $i$ 对（$i=0,1,\dots,d_h/2-1$，0 起）以角频率 $\theta_i=b^{-2i/d_h}$ 旋转，$b$ 是底数（配置里的 `rope_theta`，本课的模型都是 10000）；$\lambda_i=2\pi/\theta_i$ 是这一对的波长。
- 虚数单位写正体 $\mathrm i$，以免与频率下标 $i$ 混淆；$\bar z$ 是复数 $z$ 的共轭，$\operatorname{Re}z$ 是实部。
- 上下文长度：预训练长度 $L_{ctx}$，扩展目标 $L'_{ctx}$，扩展倍数 $s=L'_{ctx}/L_{ctx}$（课程里 $L$ 留给层数）。
- YaRN 的注意力“温度”因子写 $\mu$（论文写 $\sqrt{1/t}$，而 $t$ 在本课是位置），logits 乘 $\mu^2$。
:::

[[legend]]

[[toc]]

# 问题：GPT-2 的块哪里不够用 | 四处改动，以及“能读多长”的账单 | 15

先把两种块并排放在一起看。左边是第 1 课拆过的 GPT-2 small，右边是 LLaMA-7B，DeepSeek LLM 7B 与它宽度完全相同、只少两层（30 层）[@touvron2023llama; @deepseek2024llm]。骨架一模一样：残差主干，每层一个注意力子层、一个前馈子层，各自先归一化再计算、再加回主干。变的是四个零件。

::: fig figures/l05-block-compare.svg | 图 5.1 GPT-2 small 的块与 LLaMA-7B 式块 | wide
**看右边带数字的蓝框：这就是本课要讲的四处改动。** 左边的位置信息只在最底下加一次（一张可学习的表 wpe，1024 行）；右边输入处不加任何位置向量，改为在每一层的注意力里把 $\mathbf q$、$\mathbf k$ 按位置旋转（红字）。归一化从 LayerNorm 换成 RMSNorm；前馈层从 $768\to3072\to768$ 的 GELU 两层网络，换成有三个矩阵的 SwiGLU（$4096\to2\times11008\to4096$）；所有线性层和归一化层都没有偏置。维度取自 Hugging Face 上的 GPT-2 配置、LLaMA 论文表 2 与 LLaMA 官方代码（11008 与“无偏置”只见于代码）[@hf-gpt2; @touvron2023llama; @llama-v1-code]。
:::

四处改动的来源和时间线，放在一张表里：

| | GPT-2 small（2019） | LLaMA-7B（2023-02） | DeepSeek LLM 7B（2024-01） | DeepSeek-V3（2024-12） |
|---|---|---|---|---|
| <span style="white-space:nowrap">位置</span> | 可学习的绝对位置表，1024 行，在输入处相加 | RoPE，$b=10000$ | RoPE，$b=10000$ | 解耦 RoPE（$d_h^R=64$），$b=10000$，外加 YaRN |
| <span style="white-space:nowrap">归一化</span> | LayerNorm，放在子层输入 | RMSNorm，放在子层输入 | 同 LLaMA，$\epsilon=10^{-6}$ | RMSNorm，$\epsilon=10^{-6}$ |
| <span style="white-space:nowrap">前馈层</span> | GELU，$d_{ff}=4d=3072$ | SwiGLU，$d_{ff}=11008\approx2.69d$ | SwiGLU，$d_{ff}=11008$ | SwiGLU；前 3 层稠密 $d_{ff}=18432$，其余为 MoE 专家（每个 2048） |
| <span style="white-space:nowrap">偏置</span> | 线性层与 LayerNorm 都有 | 无（官方代码 `bias=False`） | 无（发布的权重里没有偏置张量） | 无（官方推理代码的线性层默认无偏置） |
| <span style="white-space:nowrap">上下文</span> | 1024 | 2048 | 4096 | 预训练 4096，之后扩到 128K |

来源：GPT-2 的论文与 Hugging Face 配置、代码 [@radford2019gpt2; @hf-gpt2]；LLaMA 论文表 2 与官方代码 [@touvron2023llama; @llama-v1-code]；DeepSeek LLM 论文 §2.2 与表 2，以及 Hugging Face 上的配置与权重索引（$\epsilon$、$b$、11008 与“无偏置”都只见于后者）[@deepseek2024llm; @ds-llm-hf]；DeepSeek-V3 的论文与配置、代码 [@deepseek2024v3; @ds-configs-code]。LLaMA 论文正文没有写上下文长度，2048 这个数来自 PI 与 YaRN 两篇论文的陈述 [@chen2023pi; @peng2023yarn]。表头括号里是论文（技术报告）首次公开的年月；DeepSeek LLM 的权重其实早在 2023-11-29 就已发布，报告晚了五周多才上 arXiv（[第 6 课 §1](lesson-06-deepseek-llm.html#p1)）。

下面三小节把每一处改动背后的“账单”摆出来。账单最大的一张是位置。

## 痛点一：位置是“绝对”的，语言关心的是“相对”

**GPT-2 怎么放位置。** GPT-2 有一张可学习的位置表 wpe，形状 $[1024,768]$，第 $t$ 行 $\mathbf p_t\in\R^{768}$ 就是位置 $t$ 的向量；进入第一层之前，它被**加**到词嵌入上：$\mathbf h_t^{(0)}=\mathbf e_{x_t}+\mathbf p_t$，其中 $x_t$ 是位置 $t$ 上那个 token 在词表里的编号，$\mathbf e_{x_t}\in\R^{768}$ 是它的词嵌入 [@hf-gpt2]。这张表只有 1024 行，因为 GPT-2 的上下文就是 1024 [@radford2019gpt2]。位置 1024 在表里没有对应的行，模型连前向传播都做不了。

**加法把位置和内容搅在了一起。** 暂时忽略 LayerNorm 和投影里的偏置（GPT-2 的 $W^Q$、$W^K$ 其实带偏置，带上它们会多出几项“偏置–内容”“偏置–位置”，结论不变），只看第一层的一个头。把 $\mathbf h^{(0)}$ 代进注意力分数 $\mathbf q_t^\top\mathbf k_j$，按乘法分配律展开：

::: eq 加性位置编码下的注意力分数 | (5.1)
$$\mathbf q_t^\top\mathbf k_j=\underbrace{\mathbf e_t^\top M\mathbf e_j}_{\text{内容–内容}}+\underbrace{\mathbf e_t^\top M\mathbf p_j}_{\text{内容–位置}}+\underbrace{\mathbf p_t^\top M\mathbf e_j}_{\text{位置–内容}}+\underbrace{\mathbf p_t^\top M\mathbf p_j}_{\text{位置–位置}},\qquad M=(W^Q)^\top W^K$$
$\mathbf e_t$ 是 $\mathbf e_{x_t}$ 的简写（位置 $t$ 上那个 token 的词嵌入）；$W^Q,W^K\in\R^{d_h\times d}$ 是这个头的查询、键投影，$M=(W^Q)^\top W^K\in\R^{d\times d}$ 由它们决定。
:::

后三项都含有**绝对**位置 $\mathbf p_t$ 或 $\mathbf p_j$。可是语言里真正要紧的大多是相对关系：“前一个词”“三个词之前的那个动词”，这些关系在句子出现在第 10 个位置还是第 1000 个位置时应该一样。用 (5.1) 的结构，模型必须**自己学会**从 $(\mathbf p_t,\mathbf p_j)$ 两个绝对向量里读出 $t-j$，而且要对每一对位置都学对；结构本身不保证这一点。[第 1 课](lesson-01-transformer-anatomy.html#p2)的正弦位置编码对此有所考虑：原论文设想，对任意固定偏移 $k$，位置 $pos+k$ 的编码可以写成位置 $pos$ 的编码的线性函数，从而让模型容易学会按相对位置注意 [@vaswani2017]；但它仍然加在输入上、和内容混在一起，“相对”只是可以学，不是结构保证。

::: ex 同样的内容、同样的距离，打分却不同（笔者构造）
取最小的情形：$d=2$，一个头的 $W^Q=W^K=I$（于是 $M=I$），两个位置上是同一个词，词嵌入都是 $\mathbf e=(1,0)$。位置向量用一个单频的正弦编码 $\mathbf p_t=(\cos0.5t,\ \sin0.5t)$。按 (5.1)：

$$\mathbf q_t^\top\mathbf k_j=\mathbf e^\top\mathbf e+\mathbf e^\top\mathbf p_j+\mathbf p_t^\top\mathbf e+\mathbf p_t^\top\mathbf p_j=1+\cos0.5j+\cos0.5t+\cos\big(0.5(t-j)\big).$$

相对距离都取 1：$(t,j)=(4,3)$ 时 $1+0.07074-0.41615+0.87758=1.5322$；$(104,103)$ 时 $1+0.32998-0.16299+0.87758=2.0446$；$(1004,1003)$ 时 $1+0.40408+0.79315+0.87758=3.0748$。只有最后一项“位置–位置”只依赖距离，中间两项“内容–位置”随绝对位置乱跳，所以同一对词、同样的距离，打分差了一倍。[§2](#p2) 的 RoPE 在同一个例子里会给出三个完全相同的数。
:::

我们想要的是一种位置编码，让打分**在结构上**只依赖相对距离：不管模型学成什么样，分数里都不可能出现 $t$ 与 $j$ 的单独影响。这就是 [§2](#p2) 的出发点。

## 痛点二：训练 4K，想用 128K

第 4 课和 M01 都算过：注意力打分与加权求和的 FLOPs 随序列长度**平方**增长（每层约 $4T^2d$，见 [M01 §5](../math/m01-vectors-matrices.html#p5)）。一条 128K 长的序列，打分部分的计算量是一条 4K 序列的 $32^2=1024$ 倍；按同样的 token 总数算（前者的条数只有后者的 1/32），也还是 32 倍：每个 token 要和 32 倍多的历史 token 打分。所以大模型的通行做法是：先在短序列上把绝大部分 token 训完，最后花一点点预算把上下文“扩展”到长序列。DeepSeek-V3 就是这样：14.8T token 在 4K 长度上预训练，然后两阶段扩到 32K、128K，扩展阶段合计约 1258 亿 token（按论文给的步数、序列长度与 batch 算出，见 [§7](#p7)），只占预训练 token 的 0.85%，花了 119K H800 GPU 小时，占总训练成本 2788K GPU 小时的 4.3% [@deepseek2024v3]。

“扩展”并不是把长序列直接喂进去这么简单。PI 论文报告，把只在 2048 长度上训练过的 LLaMA 直接用在更长的序列上（这叫**外推**，extrapolation），困惑度可以涨到 $10^3$ 以上，与没训练过的模型相当 [@chen2023pi]。原因要等 [§3](#p3) 讲完 RoPE 的“慢针”才能说清；[§6](#p6)、[§7](#p7) 讲三种补救办法，其中 YaRN 就是 DeepSeek-V2/V3 用的那一种。

## 痛点三：归一化与前馈层的小账

另外三处改动的账单小一些，但每层都要付。

- **归一化。** LayerNorm 对每个向量要算两个统计量：均值和方差。RMSNorm 的论文认为，LayerNorm 起作用靠的是“缩放”而不是“减均值”，于是只保留一个统计量；在 2019 年的实现与硬件上，它让不同模型的运行时间减少了 7%–64% [@zhang2019rmsnorm]。
- **前馈层。** 在[第 1 课](lesson-01-transformer-anatomy.html#p5)的参数账里，每层注意力约 $4d^2$ 个参数，前馈层约 $8d^2$，前馈层占了块里三分之二的参数。在参数量与计算量都不变的前提下换一种前馈层，就能让模型变好，这是白捡的收益：Shazeer 在 T5 上的实验里，SwiGLU 的对数困惑度 1.636，而 ReLU 基线是 1.677 [@shazeer2020glu]。
- **偏置。** PaLM 在所有稠密层和 LayerNorm 里都不用偏置，并报告这让大模型训练更稳定 [@chowdhery2022palm]。

::: key 本页要点
- 现代块与 GPT-2 块骨架相同，改了四处：位置（加在输入 → 每层旋转 $\mathbf q,\mathbf k$）、归一化（LayerNorm → RMSNorm）、前馈层（GELU $4d$ → SwiGLU 约 $\tfrac83d$）、偏置（有 → 无）。
- 加性位置编码让打分里出现绝对位置的项 (5.1)，相对关系只能靠模型自己学；可学习的位置表还给上下文长度设了硬上限。
- 长序列训练贵（打分随长度平方增长），所以先短后长；而直接外推会崩，需要专门的上下文扩展方法。
:::

# RoPE（上）：从“只看相对位置”推出旋转 | 复数、欧拉公式与二维情形的完整推导 | 30

这一页只做一件事：写下一个要求，然后一步一步推出满足它的位置编码。推导只需要复数乘法；最后得到的东西叫**旋转位置编码**（rotary position embedding，RoPE），由苏剑林等人在 2021 年提出，中文博客先于论文发表 [@su2021blog; @su2021roformer]。

## 要求：打分只依赖相对位置

**含义。** 我们想给查询和键各配一个“加位置”的函数 $f_q$、$f_k$，让加过位置之后的内积只依赖两件事：两个向量本身的内容，以及它们的**相对**距离。

**理解。** 设想同一句话“猫坐在垫子上”出现在一篇文章的开头（位置 3、4）和中间（位置 503、504）。“坐”对“猫”的注意力分数，在这两处应该一样，因为两处的相对关系一样。如果分数能写成“内容”与“距离”的函数，这件事就由结构保证，模型不用去学。

**数学。** 记 $\mathbf q=W^Q\mathbf h_t$、$\mathbf k=W^K\mathbf h_j$ 为还没加位置的查询与键（一个头，维度 $d_h$）：$\mathbf h_t,\mathbf h_j\in\R^{d}$ 是位置 $t$、$j$ 进入注意力层的隐藏状态，$W^Q,W^K\in\R^{d_h\times d}$ 是这个头的投影矩阵。要求是：

::: eq RoPE 的出发点 | (5.2)
$$\big\langle f_q(\mathbf q,t),\,f_k(\mathbf k,j)\big\rangle=g(\mathbf q,\mathbf k,t-j)\quad\text{对所有位置 }t,j\text{ 成立}$$
$f_q,f_k:\R^{d_h}\times\N\to\R^{d_h}$ 是待求的函数，$g$ 是某个只通过 $t-j$ 依赖位置的函数。RoFormer 论文式 (11) 就是这个要求 [@su2021roformer]。
:::

下面先在 $d_h=2$ 的情形把 $f$ 求出来，下一页再推广到任意偶数维。二维向量可以看成一个复数，而复数乘法恰好就是旋转；这是整条推导最省力的工具。

::: pre 复数、欧拉公式与“点积 = 实部” | 会用 $e^{\mathrm i\varphi}$ 表示旋转的可以跳过（系统的讲解见 [M03 §3 复数](../math/m03-rotation-complex.html#p3)、[§4 欧拉公式](../math/m03-rotation-complex.html#p4)）
**复数就是二维向量。** 把平面上的点 $(a,b)$ 写成 $z=a+\mathrm ib$，其中 $\mathrm i^2=-1$。它的**模**是长度 $\abs{z}=\sqrt{a^2+b^2}$，**辐角**是它与横轴的夹角 $\varphi$。**共轭** $\bar z=a-\mathrm ib$ 是关于横轴的镜像。

**欧拉公式。** $e^{\mathrm i\varphi}=\cos\varphi+\mathrm i\sin\varphi$ 是单位圆上辐角为 $\varphi$ 的点，于是任何复数都能写成**极坐标形式** $z=\abs{z}e^{\mathrm i\varphi}$。

**乘法 = 模相乘、辐角相加。** $\abs{z_1}e^{\mathrm i\varphi_1}\cdot\abs{z_2}e^{\mathrm i\varphi_2}=\abs{z_1}\abs{z_2}e^{\mathrm i(\varphi_1+\varphi_2)}$。特别地，乘以 $e^{\mathrm i\varphi}$ 就是**逆时针旋转** $\varphi$、长度不变。

**点积 = $z\bar w$ 的实部。** 设 $z=a+\mathrm ib$、$w=c+\mathrm id$ 分别代表二维向量 $(a,b)$、$(c,d)$：

$$z\bar w=(a+\mathrm ib)(c-\mathrm id)=(ac+bd)+\mathrm i(bc-ad),$$

实部 $ac+bd$ 正是点积 $(a,b)\cdot(c,d)$；虚部 $bc-ad$ 是两个向量围成的有向面积（取负号），它和点积一样，只由两个向量的长度和夹角决定。
:::

## 推导：二维情形

把 $\mathbf q,\mathbf k$ 和 $f_q,f_k$ 的输出都看成复数。点积是 $f_q\overline{f_k}$ 的实部，所以 (5.2) 说的是“$f_q\overline{f_k}$ 的实部只依赖 $t-j$”。RoFormer 的推导在这里多走了一步：要求**整个复数** $f_q\overline{f_k}$（实部是点积，虚部是有向面积）都只依赖 $t-j$ [@su2021roformer]。这是一个**选择**：它比 (5.2) 更强，换来的是方程可以按“模”和“辐角”分开解。

还需要一个**初始条件**，也是一个选择：位置 0 不改变向量，$f_q(\mathbf q,0)=\mathbf q$，$f_k(\mathbf k,0)=\mathbf k$。记 $\mathbf q$ 的辐角为 $\vartheta_q$、$\mathbf k$ 的辐角为 $\vartheta_k$（这两个角属于内容，和位置无关）。为了让式子短一些，下面把 $f_q(\mathbf q,t)$ 简写成 $f_q(t)$，把 $g(\mathbf q,\mathbf k,t-j)$ 简写成 $g(t-j)$；它们对内容 $\mathbf q,\mathbf k$ 的依赖一直都在，只是不写出来。

::: deriv 从要求推出旋转（二维情形）
$f_q(t)\,\overline{f_k(j)}=g(t-j)$ || 选择①：要求整个复数（不只是实部）只依赖 $t-j$
$f_q(t)=R_q(t)\,e^{\mathrm i\Phi_q(t)},\quad f_k(j)=R_k(j)\,e^{\mathrm i\Phi_k(j)}$ || 写成极坐标：$R\ge0$ 是模，$\Phi$ 是辐角；$g$ 同样写成 $R_ge^{\mathrm i\Phi_g}$
$R_q(t)\,R_k(j)=R_g(t-j)$ || 乘以共轭时模相乘；两边的模相等
$\Phi_q(t)-\Phi_k(j)=\Phi_g(t-j)$ || 乘以共轭时辐角相减；两边的辐角相等（按 $2\pi$ 取模）
$R_q(0)=\norm{\mathbf q},\ \Phi_q(0)=\vartheta_q;\ \ R_k(0)=\norm{\mathbf k},\ \Phi_k(0)=\vartheta_k$ || 选择②：初始条件 $f(\mathbf x,0)=\mathbf x$
$R_q(m)\,R_k(m)=R_g(0)=\norm{\mathbf q}\,\norm{\mathbf k}$ || 第 (3) 行取 $t=j=m$；再取 $t=j=0$ 并代入第 (5) 行，得到同一个 $R_g(0)$
$\Phi_q(m)-\vartheta_q=\Phi_k(m)-\vartheta_k$ || 第 (4) 行分别取 $t=j=m$ 与 $t=j=0$，两式相减再移项
$\Phi_q(m)=\vartheta_q+\phi(m),\quad\Phi_k(m)=\vartheta_k+\phi(m)$ || 推论：第 (7) 行左边与 $\mathbf k$ 无关、右边与 $\mathbf q$ 无关，却对所有 $\mathbf q,\mathbf k$ 相等，所以只能是 $m$ 的函数，记为 $\phi(m)$，且 $\phi(0)=0$
$R_q(m)=\norm{\mathbf q},\quad R_k(m)=\norm{\mathbf k}$ || 选择③：第 (6) 行只约束模的乘积，取最简单的解“位置不改变长度”（RoFormer 称之为 a straightforward solution）
$\phi(t)-\phi(j)=\Phi_g(t-j)-(\vartheta_q-\vartheta_k)$ || 把第 (8) 行代回第 (4) 行：左边只能通过 $t-j$ 依赖位置
$\phi(t)-\phi(t-1)=\Phi_g(1)-(\vartheta_q-\vartheta_k)=:\theta$ || 推论：取 $j=t-1$。左边只含位置、不含内容，右边只含内容、不含 $t$，两边相等只能都是同一个常数，记为 $\theta$：相邻位置的相位差处处相同
$\phi(m)=m\theta$ || 公差为 $\theta$ 的等差数列，且 $\phi(0)=0$（RoFormer 写成 $m\theta+\gamma$ 再令 $\gamma=0$；公共相位 $\gamma$ 在 $f_q\overline{f_k}$ 里本来就相消）
$f_q(t)=\norm{\mathbf q}\,e^{\mathrm i(\vartheta_q+t\theta)}=\mathbf q\,e^{\mathrm it\theta}$ || 把第 (8)(9)(12) 行代回极坐标：位置 $t$ 就是把查询旋转 $t\theta$
$f_k(j)=\norm{\mathbf k}\,e^{\mathrm i(\vartheta_k+j\theta)}=\mathbf k\,e^{\mathrm ij\theta}$ || 键同理，旋转 $j\theta$
:::

结论只有一行：**第 $t$ 个位置的查询乘以 $e^{\mathrm it\theta}$，第 $j$ 个位置的键乘以 $e^{\mathrm ij\theta}$**，也就是各自逆时针旋转 $t\theta$ 与 $j\theta$。回头验证它确实满足 (5.2)：

::: eq 二维 RoPE 与它的打分 | (5.3)
$$f_q(\mathbf q,t)\,\overline{f_k(\mathbf k,j)}=\mathbf q\,\bar{\mathbf k}\,e^{\mathrm i(t-j)\theta},\qquad \big\langle f_q(\mathbf q,t),f_k(\mathbf k,j)\big\rangle=\operatorname{Re}\!\big[\mathbf q\,\bar{\mathbf k}\,e^{\mathrm i(t-j)\theta}\big]$$
这里 $\mathbf q,\mathbf k$ 当作复数。$\mathbf q\bar{\mathbf k}$ 只含内容，$e^{\mathrm i(t-j)\theta}$ 只含相对距离，两者相乘。
:::

::: warn 哪些是选择，哪些是推论
这条推导有三处**选择**：① 要求整个复数 $f_q\overline{f_k}$ 只依赖 $t-j$（比 (5.2) 强）；② 初始条件 $f(\mathbf x,0)=\mathbf x$；③ 位置不改变向量长度。第 (8) 行与第 (11) 行是**推论**。所以 RoFormer 证明的是“这是**一个**解”，不是“唯一解”；论文的推导也只做了二维情形 [@su2021roformer]。

选择③排除了什么，可以自己算出来（笔者补充的推导，不在论文里）：如果允许 $R_q(\mathbf q,m)=\norm{\mathbf q}a(m)$、$R_k(\mathbf k,m)=\norm{\mathbf k}/a(m)$，第 (6) 行仍然成立，而 $t\ne j$ 时模的乘积是 $\norm{\mathbf q}\norm{\mathbf k}\,a(t)/a(j)$；要它只依赖 $t-j$，同样的“相邻比值是常数”论证给出 $a(m)=e^{cm}$，打分会多出一个因子 $e^{c(t-j)}$，随距离指数增长或衰减。RoPE 选了 $c=0$：**位置只转角度，不改长度**。
:::

## 写成实数矩阵：一个 2×2 旋转

复数乘法 $e^{\mathrm i\varphi}(a+\mathrm ib)=(a\cos\varphi-b\sin\varphi)+\mathrm i(a\sin\varphi+b\cos\varphi)$，换回二维实向量就是乘一个旋转矩阵：

::: eq 二维旋转矩阵 | (5.4)
$$\mathbf R(\varphi)=\begin{pmatrix}\cos\varphi&-\sin\varphi\\ \sin\varphi&\cos\varphi\end{pmatrix},\qquad f_q(\mathbf q,t)=\mathbf R(t\theta)\,\mathbf q,\qquad f_k(\mathbf k,j)=\mathbf R(j\theta)\,\mathbf k$$
$\mathbf R(\varphi)$ 是**正交矩阵**：$\mathbf R(\varphi)^\top\mathbf R(\varphi)=I$，它不改变向量长度。
:::

用矩阵也能三行验证“只剩相对位置”，每一行用到的都是旋转最基本的性质（旋转矩阵怎样从“坐标轴转到哪里”推出来、转置为什么等于反向旋转，见 [M03 §2](../math/m03-rotation-complex.html#p2) 与 [§5](../math/m03-rotation-complex.html#p5)）：

::: deriv 旋转后的内积只依赖 j − t
$\big(\mathbf R(t\theta)\mathbf q\big)^\top\big(\mathbf R(j\theta)\mathbf k\big)=\mathbf q^\top\mathbf R(t\theta)^\top\mathbf R(j\theta)\,\mathbf k$ || 乘积的转置等于转置的逆序乘积 $(AB)^\top=B^\top A^\top$（[M01 §3](../math/m01-vectors-matrices.html#p3)）
$=\mathbf q^\top\mathbf R(-t\theta)\,\mathbf R(j\theta)\,\mathbf k$ || 旋转矩阵的转置是反向旋转：把 (5.4) 转置，$\sin$ 变号，等于 $\mathbf R(-\varphi)$
$=\mathbf q^\top\mathbf R\big((j-t)\theta\big)\,\mathbf k$ || 先转 $j\theta$ 再转 $-t\theta$，等于一次转 $(j-t)\theta$（角度相加）
:::

把投影也写进去，一个头的打分就是

::: eq 带 RoPE 的打分（二维） | (5.5)
$$\mathbf q_t^\top\mathbf k_j=\mathbf h_t^\top(W^Q)^\top\,\mathbf R\big((j-t)\theta\big)\,W^K\mathbf h_j,\qquad \mathbf q_t=\mathbf R(t\theta)W^Q\mathbf h_t,\ \ \mathbf k_j=\mathbf R(j\theta)W^K\mathbf h_j$$
对照 (5.1)：夹在两个投影中间的矩阵只随 $j-t$ 变化，绝对位置从结构上消失了。
:::

注意 RoPE 与 GPT-2 的做法有三点不同。第一，它是**乘性**的（乘一个旋转矩阵），而此前的位置编码都是加性的 [@su2021roformer]。第二，它作用在**投影之后**的 $\mathbf q$、$\mathbf k$ 上，而且每一层都做，不是只在输入处做一次。第三，它**不作用于值** $\mathbf v$：位置只影响“看谁”，不直接改变“拿到什么”。第三点也让 RoPE 和[第 4 课](lesson-04-inference-kv-cache.html#p2)的 KV cache 相处融洽：$\mathbf k_j=\mathbf R(j\theta)W^K\mathbf h_j$ 只依赖键自己的位置 $j$，算出来就不再变，可以旋转之后直接写进缓存。

## 算一个数

::: fig figures/l05-rotation-2d.svg | 图 5.2 位置不同、距离相同的三对 (t, j) 得到同一个内积 | wide
**看每个小图里的黑色弧线：三对向量旋转的角度各不相同（红色弧线），但它们之间的夹角都是 92.2°，所以内积都是 −0.0349。** 取 $\theta=0.3$ rad/位置，$\mathbf q=(1,0.5)$、$\mathbf k=(0.8,-0.2)$<!-- 研究笔记 R22 的算例 -->。虚线是旋转前的方向，夹角 40.6°、内积 0.7；三对的距离都是 3，所以夹角都增加了 $3\times0.3$ rad $\approx51.6°$。每个坐标都由生成脚本算出。
:::

::: ex 用复数和用矩阵各算一遍
取 $\theta=0.3$，$\mathbf q=(1,0.5)$，$\mathbf k=(0.8,-0.2)$，查询在位置 $t=5$，键在位置 $j=2$。

**复数法。** $\mathbf q\bar{\mathbf k}=(1+0.5\mathrm i)(0.8+0.2\mathrm i)=0.8+0.2\mathrm i+0.4\mathrm i+0.1\mathrm i^2=0.7+0.6\mathrm i$。按 (5.3)，打分 $=\operatorname{Re}[(0.7+0.6\mathrm i)e^{\mathrm i\cdot0.9}]=0.7\cos0.9-0.6\sin0.9=0.7\times0.62161-0.6\times0.78333=0.43513-0.47000=-0.0349$。

**矩阵法。** $\mathbf R(1.5)\mathbf q=(\cos1.5-0.5\sin1.5,\ \sin1.5+0.5\cos1.5)=(-0.4280,\ 1.0329)$；$\mathbf R(0.6)\mathbf k=(0.8\cos0.6+0.2\sin0.6,\ 0.8\sin0.6-0.2\cos0.6)=(0.7732,\ 0.2866)$。点积 $-0.4280\times0.7732+1.0329\times0.2866=-0.3309+0.2960=-0.0349$。

两种算法一致。把位置换成 $(13,10)$ 或 $(3,0)$，旋转后的坐标完全不同（见图 5.2），打分仍是 $-0.0349$；只有 $t=j$ 时打分回到不旋转的 $\operatorname{Re}(0.7+0.6\mathrm i)=0.7$。
:::

::: ex 回到 §1 的例子：三个完全相同的数
[§1](#p1) 用加性的正弦编码算过“同一个词、距离 1”在三处的打分，得到 1.5322、2.0446、3.0748。现在不再往输入上加位置向量，改用 RoPE，频率仍取 $\theta=0.5$。词嵌入 $\mathbf e=(1,0)$，$W^Q=W^K=I$，所以 $\mathbf q=\mathbf k=(1,0)$，当作复数就是 1，$\mathbf q\bar{\mathbf k}=1$。按 (5.3)，打分 $=\operatorname{Re}\big[1\cdot e^{\mathrm i(t-j)\cdot0.5}\big]=\cos\big(0.5(t-j)\big)$。

$(t,j)=(4,3)$、$(104,103)$、$(1004,1003)$ 的距离都是 1，打分都是 $\cos0.5=0.8776$，三个数完全相同。对照 (5.1)：加性编码里随绝对位置乱跳的是两项“内容–位置”，RoPE 的打分里根本没有这样的项。
:::

::: ana 类比：钟面上的指针
**映射。** 一对维度 = 一个钟面；向量 = 钟面上的一根指针（长度是内容的“强度”，初始角度是内容的“方向”）；位置 $t$ = 时刻；$\theta$ = 指针每过一个 token 转过的角度；打分 = 两根指针的长度相乘再乘以它们夹角的余弦。

**收益。** 查询和键在各自的时刻被拨到钟面上，两根指针都按同一速度转动，所以它们之间的夹角只由“差了几个时刻”决定。这就像两个人在同一条环形跑道上以同样的速度跑步：他们在跑道上相隔多远，只取决于出发时间差，不取决于现在几点。

**失效的地方。** 模型从来不“读”角度，它只看到点积；而点积还混着内容本身的夹角 $\vartheta_q-\vartheta_k$。真实的 RoPE 也不是一个钟面，而是 $d_h/2$ 个转速不同的钟面，模型把它们的读数加在一起（下一页）。单独一个钟面还有一个致命缺陷，见下面的失败模式。
:::

::: warn 失败模式：只有一根指针会“绕圈”
取 $\theta=0.5$，$\mathbf q=\mathbf k=(1,0)$。打分是 $\cos(0.5\Delta)$：$\Delta=0$ 时 1，$\Delta=2$ 时 0.5403，$\Delta=4$ 时 $-0.4161$，到 $\Delta=13$ 又回到 $\cos6.5=0.9766$，几乎和 $\Delta=0$ 一样。指针转一整圈（$2\pi/0.5\approx12.6$ 个位置）之后，远处的 token 看起来和紧挨着的一样。一个频率分不清“近”和“恰好差一圈”，所以 RoPE 要用很多个频率，其中有的转得极慢。这正是下一页的内容。
:::

::: key 本页要点
- 要求：加位置后的打分只依赖内容与 $t-j$ (5.2)。二维情形下，RoPE 是满足它的一个解：位置 $t$ 把向量旋转 $t\theta$，即乘以 $e^{\mathrm it\theta}$ 或 $\mathbf R(t\theta)$。
- 推导里有三处选择（整个复数只依赖 $t-j$、位置 0 不变、长度不变）和两处推论（相位只依赖位置、相邻位置相位差为常数）。
- 旋转是乘性的，作用在每层投影后的 $\mathbf q,\mathbf k$ 上，不作用于 $\mathbf v$；中间矩阵 $\mathbf R((j-t)\theta)$ 只随相对距离变化 (5.5)。
:::

# RoPE（下）：多频率时钟、实现与“远程衰减” | 高维推广、频率与波长、数值验证，以及一个常被夸大的性质 | 30

上一页在二维里推出了“位置 = 旋转”，也看到了一个频率的毛病：指针转满一圈，远处和近处就分不清了。真实的注意力头有 $d_h=128$ 维（LLaMA-7B、DeepSeek LLM），DeepSeek-V2/V3 里承载 RoPE 的那一段有 $d_h^R=64$ 维 [@touvron2023llama; @ds-configs-code]。这一页把二维的结论推广到高维，同时解决“绕圈”问题：给每一对维度配一个**不同**的转速。

## 从一对到很多对

**含义。** 把 $d_h$ 维向量的分量两两分组，得到 $d_h/2$ 对；每一对当作一个二维平面，各自按 [§2](#p2) 的办法旋转，但转速 $\theta_i$ 各不相同。

**理解。** 点积是逐分量相乘再相加，所以整个向量的点积等于各对“小点积”之和。每一对的小点积已经只依赖 $t-j$，只依赖 $t-j$ 的函数相加，结果仍只依赖 $t-j$。这一步不需要新的数学，只用到点积的可加性；RoFormer 的说法是利用内积的线性，把 $d$ 维空间分成 $d/2$ 个子空间 [@su2021roformer]。

**数学。** 记第 $i$ 对为 $\mathbf q^{(i)}=(q_{2i},q_{2i+1})$（分量从 0 数起），位置 $t$ 的旋转是一个块对角矩阵（块对角旋转与多尺度频率的系统讲解见 [M03 §6](../math/m03-rotation-complex.html#p6)，[§7](../math/m03-rotation-complex.html#p7) 有一个 4 维 RoPE 的完整手算）：

::: eq 块对角旋转矩阵 | (5.6)
$$\begin{aligned}\mathbf R_{\Theta,t}&=\diag\!\big(\mathbf R(t\theta_0),\,\mathbf R(t\theta_1),\,\dots,\,\mathbf R(t\theta_{d_h/2-1})\big)\\[4pt]&=\begin{pmatrix}\cos t\theta_0&-\sin t\theta_0&&&\\ \sin t\theta_0&\cos t\theta_0&&&\\ &&\cos t\theta_1&-\sin t\theta_1&\\ &&\sin t\theta_1&\cos t\theta_1&\\ &&&&\ddots\end{pmatrix}\end{aligned}$$
$\mathbf R_{\Theta,t}\in\R^{d_h\times d_h}$，空白处都是 0。$\Theta=\{\theta_0,\dots,\theta_{d_h/2-1}\}$ 是全部频率。带 RoPE 的查询与键是 $\mathbf q_t=\mathbf R_{\Theta,t}W^Q\mathbf h_t$，$\mathbf k_j=\mathbf R_{\Theta,j}W^K\mathbf h_j$（RoFormer 式 (14)(15)）。
:::

::: deriv 高维情形仍然只依赖 j − t
$(\mathbf R_{\Theta,t}\mathbf q)^\top(\mathbf R_{\Theta,j}\mathbf k)=\sum_{i}\big(\mathbf R(t\theta_i)\mathbf q^{(i)}\big)^\top\big(\mathbf R(j\theta_i)\mathbf k^{(i)}\big)$ || 块对角矩阵乘向量 = 每一块各乘各的；点积按对拆成 $d_h/2$ 个小点积之和
$=\sum_{i}\mathbf q^{(i)\top}\,\mathbf R\big((j-t)\theta_i\big)\,\mathbf k^{(i)}$ || 每一对套用 §2 的三行推导
$=\mathbf q^\top\,\mathbf R_{\Theta,j-t}\,\mathbf k$ || 再把各块拼回一个块对角矩阵
:::

用复数写更紧凑。令第 $i$ 对对应的复数为 $\tilde q_i=q_{2i}+\mathrm iq_{2i+1}$、$\tilde k_i=k_{2i}+\mathrm ik_{2i+1}$（旋转前的分量），由 (5.3)：

::: eq RoPE 打分的复数形式 | (5.7)
$$\mathbf q_t^\top\mathbf k_j=\operatorname{Re}\sum_{i=0}^{d_h/2-1}\tilde q_i\,\overline{\tilde k_i}\;e^{\mathrm i(t-j)\theta_i}$$
每一项是“内容系数 $\tilde q_i\overline{\tilde k_i}$”乘以“按频率 $\theta_i$ 转动的相位”。PI 论文把这个和记作相对距离的函数 $a(t-j)$，[§6](#p6) 会用到 [@chen2023pi]。
:::

两点性质直接可读。第一，$\mathbf R_{\Theta,t}$ 是正交矩阵（每一块都是旋转），所以 $\norm{\mathbf R_{\Theta,t}\mathbf q}=\norm{\mathbf q}$，旋转不改变向量长度，RoFormer 认为这保证了数值稳定 [@su2021roformer]。第二，$d_h/2$ 个频率可以任意选，推导不管；怎么选是下一节的事。

## 频率怎么选：一组几何级数

**含义。** 第 $i$ 对每过一个 token 转 $\theta_i$ 弧度；$\theta_i$ 从 1 开始，按固定比例一对比一对慢下去。

**理解。** 回到钟面的图景（[第 1 课](lesson-01-transformer-anatomy.html#p2)的图 1.5 已经把正弦位置编码画成一组转速不同的时钟）：秒针、分针、时针的转速相差 60 倍、12 倍，合起来才能既分辨一秒之差、又数清一整天。RoPE 也一样，快的指针分辨相邻 token，慢的指针区分远近。RoFormer 沿用了原始 Transformer 正弦位置编码的频率设定 [@su2021roformer]，后者的波长就是一组从 $2\pi$ 到 $10000\cdot2\pi$ 的几何级数 [@vaswani2017]。

**数学。** 底数 $b=10000$：

::: eq RoPE 的频率与波长 | (5.8)
$$\theta_i=b^{-2i/d_h},\qquad \lambda_i=\frac{2\pi}{\theta_i}=2\pi\,b^{2i/d_h},\qquad i=0,1,\dots,\tfrac{d_h}{2}-1$$
相邻两对的频率之比固定为 $b^{-2/d_h}$；波长 $\lambda_i$ 是第 $i$ 对转满一圈需要的 token 数。
:::

::: ex 算几根指针的转速
**$d_h=128$。** $\theta_0=10000^0=1$ rad/token，$\lambda_0=2\pi\approx6.28$：最快的指针每个 token 转 $57.3°$，6.28 个 token 转一圈。$\theta_{32}=10000^{-64/128}=10000^{-1/2}=0.01$，$\lambda_{32}=628.3$。最慢的 $\theta_{63}=10000^{-126/128}=1.1548\times10^{-4}$，$\lambda_{63}\approx54{,}410$ token。相邻两对的比值 $10000^{-2/128}=0.86596$。

**$d_h^R=64$（DeepSeek-V2/V3 的解耦 RoPE）。** 比值 $10000^{-2/64}=0.74989$，每一对比上一对慢得更多；最慢的 $\theta_{31}=1.3335\times10^{-4}$，$\lambda_{31}\approx47{,}117$。因为 $2i/64=4i/128$，64 维的第 $i$ 对恰好就是 128 维的第 $2i$ 对：维度减半，等于隔一根取一根指针<!-- 研究笔记 R1 -->。
:::

::: fig figures/l05-clocks.svg | 图 5.3 多频率时钟：DeepSeek 64 维 RoPE 的 8 根指针在三个时刻 | wide
**看前两行：从 $t=100$ 到 $t=101$，最左边的快针转了 $57.3°$（从 $329.6°$ 到 $26.9°$），右边的慢针几乎没动。** 相邻两个 token 主要靠快针区分。再看第三行 $t=4096$：快针已经转了 651.9 圈，它的角度只告诉你“对一圈取余是多少”；而 $i=24$、$28$ 两根慢针连一圈都没转满（琥珀色扇形），它们的角度本身就单调地编码了“离开头有多远”。每一个角度都由生成脚本按 $t\theta_i$ 算出<!-- 与研究笔记 R23 一致 -->。
:::

::: ana 类比：一只有很多根指针的钟
**映射。** 每一对维度 = 一根指针；$\theta_i$ = 这根指针的转速；波长 $\lambda_i$ = 转一圈要多少个 token；位置 $t$ = 钟面上的时刻；打分 (5.7) = 把每根指针上“查询与键的夹角余弦”按内容系数加权后求和。

**收益。** 这个图景一眼说明两件后面反复要用的事：① 快针（$\lambda_i\ll L_{ctx}$）在训练中转过成百上千圈，它们只能表达**相对**的、短距离的关系；② 慢针（$\lambda_i>L_{ctx}$）在训练中连一圈都没转满，它们的角度只出现过一小段弧，而且随距离单调变化、不会“绕回来”，模型可以把它们当成“绝对位置”来用。这里的“绝对”要小心理解：打分仍然只依赖 $t-j$ (5.7)，说“绝对”，是指这把尺子在训练范围内没有重复的刻度，距离可以直接读出来、不用“对一圈取余”；YaRN 论文把这称为保留了绝对位置信息，这是它的推测，不是定理 [@peng2023yarn]。上下文扩展的全部技巧（§6、§7）都是在决定：该拨慢哪些指针、拨慢多少。

**失效的地方。** 真实时钟的转速是整数比（秒针恰好是分针的 60 倍），读数可以“进位”；RoPE 的相邻比值是 $0.866$ 这类无理数，没有进位，也没有哪根针单独代表“分钟”。模型也从不直接读角度，只读 (5.7) 这个加权和；各根指针的“权重” $\tilde q_i\overline{\tilde k_i}$ 由内容决定，是学出来的。
:::

## 慢针：在预训练长度内转不满一圈的维度

哪些指针在预训练里连一圈都没转满？条件是 $\lambda_i>L_{ctx}$。

::: ex DeepSeek 的 64 维里有几根“慢针”
$d_h^R=64$，$L_{ctx}=4096$。$\lambda_i>4096\iff2\pi\cdot10000^{i/32}>4096\iff10000^{i/32}>651.90\iff i>32\cdot\frac{\ln651.90}{\ln10000}=22.51$。所以 $i=23,\dots,31$ 共 **9 对**（18 维）转不满一圈。最慢的一对在 $t=4096$ 时只转了 $4096\times1.3335\times10^{-4}=0.5462$ rad，即 $31.3°$。

对照 LLaMA（$d_h=128$，$L_{ctx}=2048$）：有 23 对转不满一圈，最慢的一对在 $t=2048$ 只转了 $13.55°$；同样的 128 维放到 4096 长度，是 18 对<!-- 研究笔记 R2 -->。

**预告失败模式。** 把 DeepSeek 的模型直接用到 $t=163{,}840$，最慢的那根针会转到 $163840\times1.3335\times10^{-4}=21.85$ rad，约 3.5 圈，而训练中它只见过 $0°$ 到 $31.3°$ 这一小段。模型从没见过的角度，就是直接外推会崩的原因（[§6](#p6)）。
:::

::: fig figures/l05-wavelengths.svg | 图 5.4 RoPE 各对维度的波长（对数纵轴） | wide
**看琥珀色区域：波长超过 4096 的点都在里面，它们在 DeepSeek 的预训练长度内转不满一圈。** 横轴是维度在向量中的相对位置 $2i/d_h$，底下按 $d_h=128$ 标对号、顶上按 $d_h^R=64$ 标对号；两种头宽落在同一条直线上（纵轴取对数后，几何级数就是直线），64 维只是隔一个取一个。虚线是两个预训练长度：LLaMA 的 2048 与 DeepSeek 的 4096。
:::

## 实现：不做矩阵乘法

(5.6) 是一个 $d_h\times d_h$ 的矩阵，但几乎全是零，没有人真的去乘它。按对展开，旋转只需要两次逐元素乘法和一次加法（RoFormer 式 (34)）[@su2021roformer]：

::: eq RoPE 的逐元素实现 | (5.9)
$$\mathbf R_{\Theta,t}\mathbf x=\begin{pmatrix}x_0\\x_1\\x_2\\x_3\\\vdots\end{pmatrix}\odot\begin{pmatrix}\cos t\theta_0\\\cos t\theta_0\\\cos t\theta_1\\\cos t\theta_1\\\vdots\end{pmatrix}+\begin{pmatrix}-x_1\\x_0\\-x_3\\x_2\\\vdots\end{pmatrix}\odot\begin{pmatrix}\sin t\theta_0\\\sin t\theta_0\\\sin t\theta_1\\\sin t\theta_1\\\vdots\end{pmatrix}$$
$\odot$ 是逐元素乘法。$\cos t\theta_i$、$\sin t\theta_i$ 只和位置有关，可以对所有位置预先算好存成表。
:::

写成代码只有几行。下面是笔者按 (5.9) 写的 NumPy 版本，和 LLaMA 官方代码一样用“相邻两维配对”：

```python
import numpy as np

def rope(x, t, base=10000.0):
    """x: [..., d_h]；t: 位置（整数，或能与 x 的前几维广播的整数数组）。返回旋转后的 x。"""
    d_h = x.shape[-1]
    theta = base ** (-np.arange(0, d_h, 2) / d_h)      # θ_i = b^(-2i/d_h)，共 d_h/2 个
    ang = np.asarray(t)[..., None] * theta               # [..., d_h/2]
    x0, x1 = x[..., 0::2], x[..., 1::2]                  # 每一对的两个分量
    out = np.empty_like(x)
    out[..., 0::2] = x0 * np.cos(ang) - x1 * np.sin(ang)
    out[..., 1::2] = x0 * np.sin(ang) + x1 * np.cos(ang)
    return out

# 注意力里：q = rope(W_q @ h_t, t)；k = rope(W_k @ h_j, j)；v 不旋转
```

::: warn 两种“配对”写法：数学等价，权重不通用
RoFormer 论文、LLaMA 官方代码与 DeepSeek-V3 官方推理代码都把相邻两维 $(x_{2i},x_{2i+1})$ 配成一对 [@llama-v1-code; @ds-configs-code]。Hugging Face 的 LLaMA 实现却把前后两半配对，$(x_i,x_{i+d_h/2})$，用一个叫 `rotate_half` 的函数；它的权重转换脚本因此要把 $W^Q$、$W^K$ 的行重新排列（`permute`）[@hf-llama-convert]。DeepSeek 的 Hugging Face 代码则保留相邻配对的权重，在运行时先把 $\mathbf q,\mathbf k$ 重排再调用 `rotate_half`。两种写法只差一个固定的维度置换，打分完全相同；但**一种写法训练出的权重，不能直接拿到另一种写法里用**，否则每一对里配错的两个分量会被当作同一个平面来转。
:::

还有三个实现细节决定了 RoPE 在整个模型里的位置。其一，它每一层都做，作用在投影之后、点积之前。其二，只转 $\mathbf q$ 和 $\mathbf k$，$\mathbf v$ 原样不动。其三，和第 4 课的 KV cache 配合时，缓存的是**旋转之后**的键 $\mathbf k_j=\mathbf R_{\Theta,j}W^K\mathbf h_j$；它只依赖键自己的位置，写进缓存后永不改变，解码第 $t$ 步只需旋转新 token 的 $\mathbf q_t$ 与 $\mathbf k_t$。

## 数值验证：相对位置性质

推导说打分只依赖 $t-j$。用随机向量直接算一遍，是最便宜的核对方式。

::: fig figures/l05-relative-heatmap.svg | 图 5.5 RoPE 的得分矩阵沿每条对角线都是常数 | wide
**看左图的颜色：每一条从左上到右下的斜线（固定 $\Delta=t-j$）颜色都一样，这就是“只依赖相对位置”。** $d_h=128$，$\mathbf q$、$\mathbf k$ 是用固定种子（NumPy `default_rng(0)`）抽的标准正态向量，在每个位置 $t$、$j$ 上分别旋转后求点积；右上三角被因果掩码去掉。右图把 $t=47$ 与 $t=30$ 两行按距离重画，两行的点完全重合。生成脚本对 48 条对角线逐一检查了“最大值 − 最小值 $<10^{-9}$”。
:::

::: ex 同一个距离，五种位置
用图 5.5 的那对随机向量（$d_h=128$，$b=10000$）。不旋转时 $\mathbf q^\top\mathbf k=0.3463$。取距离 $\Delta=26$：

| $(t,j)$ | $(26,0)$ | $(37,11)$ | $(137,111)$ | $(1037,1011)$ | $(100037,100011)$ |
|---|---|---|---|---|---|
| 旋转后的内积 | 1.692715 | 1.692715 | 1.692715 | 1.692715 | 1.692715 |

五个数在前 11 位小数上一致，最后一列与第一列只差 $1.4\times10^{-12}$：位置到十万时，$t\theta_0$ 这样的大角度在双精度浮点里有舍入误差，这是计算机的误差，不是公式的误差。再用 (5.7) 的复数公式直接算 $\operatorname{Re}\sum_i\tilde q_i\overline{\tilde k_i}e^{\mathrm i\cdot26\theta_i}$，也得到 1.692715。长度：$\norm{\mathbf q}=10.7839$，旋转 $\mathbf R_{\Theta,12345}$ 之后仍是 $10.7839$。<!-- 这组数与研究笔记 R11 相同 -->
:::

## 远程衰减：被证明的是一个上界

RoFormer 还说，这种频率设定带来“远程衰减”（long-term decay）：相对距离越大，内积越小，这符合“离得远的词关系弱”的直觉 [@su2021roformer]。这句话常被转述成“RoPE 保证注意力随距离衰减”。先看论文实际证明了什么。

**论文的论证。** 记 $h_i=\tilde q_i\overline{\tilde k_i}$（第 $i$ 对的内容系数），$S_m=\sum_{i=0}^{m-1}e^{\mathrm i\Delta\theta_i}$（前 $m$ 个相位的部分和），约定 $S_0=0$、$h_{d_h/2}=0$。用一次“分部求和”（Abel 变换，离散版的分部积分）：

::: deriv 远程衰减的上界（RoFormer §3.4.3）
$\sum_{i}h_i\,e^{\mathrm i\Delta\theta_i}=\sum_{i}h_i\,(S_{i+1}-S_i)$ || $S_{i+1}-S_i=e^{\mathrm i\Delta\theta_i}$，按部分和的定义；$i$ 从 0 到 $d_h/2-1$
$=\sum_{i}h_iS_{i+1}-\sum_{i}h_{i+1}S_{i+1}$ || 第二个和把下标平移一位；用到 $S_0=0$ 与 $h_{d_h/2}=0$
$=-\sum_{i}S_{i+1}\,(h_{i+1}-h_i)$ || 合并同类项
$\Big\lvert\sum_{i}h_i\,e^{\mathrm i\Delta\theta_i}\Big\rvert\le\sum_{i}\abs{S_{i+1}}\,\abs{h_{i+1}-h_i}$ || 三角不等式：和的模不超过模的和
$\le\Big(\max_i\abs{h_{i+1}-h_i}\Big)\sum_{i}\abs{S_{i+1}}$ || 把每个 $\abs{h_{i+1}-h_i}$ 放大成最大的那个
:::

打分 (5.7) 是这个复数和的实部，而任何复数都满足 $\abs{\operatorname{Re}z}\le\abs{z}$，所以同一个上界也管住打分的绝对值。最后一行把上界拆成两个因子：一个只和内容有关（相邻内容系数之差的最大值），一个只和距离有关。RoFormer 的图 2 画的就是后者的平均值：

::: eq 远程衰减图画的量 | (5.10)
$$U(\Delta)=\frac{1}{d_h/2}\sum_{m=1}^{d_h/2}\abs{S_m},\qquad S_m=\sum_{i=0}^{m-1}e^{\mathrm i\Delta\theta_i}$$
$\Delta=0$ 时每个 $\abs{S_m}=m$，$U(0)=\frac{d_h/2+1}{2}$，$d_h=128$ 时为 32.5。RoFormer 图 2 没有标 $d$，按 $d=128$ 重算与原图吻合，苏剑林博客中同一张图的图注也写明 $d=128$ [@su2021blog]。
:::

::: fig figures/l05-decay.svg | 图 5.6 远程衰减：RoFormer 的量 U(Δ)，以及同向向量的平均得分 | wide
**先看 (a)，再看 (b) 的粗线在 Δ≈1000 之后变平。** (a) 是 RoFormer 图 2 的范围（$d_h=128$，$\Delta=0\dots256$），$U$ 从 32.5 降到约 6.5，看起来确实在衰减。(b) 把同一个量延长到 65536：约 1000 之后，512 点滑动平均不再系统性下降，一直在 4.0 到 6.2 之间徘徊（$\Delta=1000$ 时 5.66，4096 时 4.72，32768 时 5.08），而逐点的值在剧烈起伏（$[1000,4096]$ 内在 2.23 到 11.39 之间）。(c) 换一个问题：若 $\mathbf q$ 与 $\mathbf k$ 同向，平均打分随距离怎样变？这条曲线确实下降（$\Delta=10$ 时 0.669，1000 时 0.159），但在 $\Delta\approx1707$ 处第一次穿过 0（4096 时为 $-0.053$），此后在 0 上下摆动；独立随机的 $\mathbf q,\mathbf k$ 在任何距离的期望打分都是 0（虚线）。据 RoFormer 图 2 重绘并延长 [@su2021roformer]；所有曲线由脚本全量计算<!-- 研究笔记 R9、R10 -->。
:::

图 5.6 (c) 那条“同向平均得分”可以精确推出来，它说明了衰减在什么条件下成立。推导要用到矩阵的“迹”。

::: pre 迹（trace）与它的两条性质 | 会用 $\tr(AB)=\tr(BA)$ 的可以跳过（更完整的讲解与同一个技巧的另一次使用见 [M02 §4](../math/m02-rank-svd-lowrank.html#p4)）
**含义。** 方阵 $A\in\R^{n\times n}$ 的**迹**是对角线元素之和：$\tr A=\sum_{k=1}^{n}A_{kk}$。

**两条性质。** ① 轮换：只要 $AB$ 与 $BA$ 都是方阵，就有 $\tr(AB)=\tr(BA)$。理由是 $\tr(AB)=\sum_k\sum_lA_{kl}B_{lk}$，交换两个有限求和的次序，得到的正是 $\sum_l\sum_kB_{lk}A_{kl}=\tr(BA)$。② 一个数可以看成 $1\times1$ 的矩阵，它的迹就是它自己。

**合起来的用法。** 取 $A=\mathbf q^\top$（$1\times d_h$）、$B=R\,\mathbf q$（$d_h\times1$）：$\mathbf q^\top R\,\mathbf q=\tr(\mathbf q^\top R\,\mathbf q)=\tr(R\,\mathbf q\mathbf q^\top)$。“向量–矩阵–向量”的二次型变成了“矩阵乘外积”的迹，随机的部分全部集中到外积 $\mathbf q\mathbf q^\top$ 里，求期望时只需要知道 $\E[\mathbf q\mathbf q^\top]$。

**例子。** 二维旋转块 $\mathbf R(\varphi)$ 的两个对角元都是 $\cos\varphi$，所以 $\tr\mathbf R(\varphi)=2\cos\varphi$；$\varphi=0$ 时是单位矩阵，迹为 2。
:::

::: deriv 同向向量的平均得分
$\E\big[\mathbf q^\top\mathbf R_{\Theta,\Delta}\,\mathbf q\big]=\E\big[\tr(\mathbf R_{\Theta,\Delta}\,\mathbf q\mathbf q^\top)\big]$ || 一个数等于它自己的迹，且 $\tr(AB)=\tr(BA)$
$=\tr\big(\mathbf R_{\Theta,\Delta}\,\E[\mathbf q\mathbf q^\top]\big)=\tr(\mathbf R_{\Theta,\Delta})$ || 迹与期望都是线性的，可以交换；假设 $\E[\mathbf q\mathbf q^\top]=I$（例如各分量独立、均值 0、方差 1；只用到这个二阶矩）
$=\sum_{i=0}^{d_h/2-1}2\cos(\Delta\theta_i)$ || 每个 $2\times2$ 旋转块的迹是 $2\cos(\Delta\theta_i)$
:::

::: eq 同向与独立两种情形 | (5.11)
$$\frac{1}{d_h}\E\big[\mathbf q^\top\mathbf R_{\Theta,\Delta}\,\mathbf q\big]=\frac{2}{d_h}\sum_{i=0}^{d_h/2-1}\cos(\Delta\theta_i),\qquad \E\big[\mathbf q^\top\mathbf R_{\Theta,\Delta}\,\mathbf k\big]=\E[\mathbf q]^\top\mathbf R_{\Theta,\Delta}\,\E[\mathbf k]=0$$
左式：键与查询相同（完全同向）时的平均打分，除以 $d_h$ 使 $\Delta=0$ 时等于 1。右式：$\mathbf q,\mathbf k$ 相互独立且均值为 0。更一般地，同样的迹技巧给出 $\E[\mathbf q^\top\mathbf R_{\Theta,\Delta}\mathbf k]=\tr\big(\mathbf R_{\Theta,\Delta}\E[\mathbf k\mathbf q^\top]\big)$：当 $\E[\mathbf k\mathbf q^\top]=\kappa I$（$\kappa>0$，每一对维度上的相关程度相同）时，它是左式的 $\kappa d_h$ 倍，呈同样的下降趋势；相关性若集中在少数几对维度上，曲线的形状就由那几对的频率决定，不一定下降。
:::

::: ex 同向平均得分的几个值
$d_h=128$，按 (5.11) 左式：$\Delta=0$ 时 1；$\Delta=1$ 时 0.9702；$\Delta=10$ 时 0.6691；$\Delta=100$ 时 0.4772；$\Delta=1000$ 时 0.1590；$\Delta=4096$ 时 $-0.0529$；$\Delta=16384$ 时 $-0.1423$<!-- 研究笔记 R10 -->。趋势是下降，但不单调（$\Delta$ 从 11 到 12 就有一次小回升：0.6616 → 0.6622），在 $\Delta=1707$ 处第一次变负，之后在 0 上下摆动。用蒙特卡洛模拟核对：取 2000 个标准正态的 $\mathbf q$（NumPy `default_rng(0)`），令 $\mathbf k=\mathbf q$，逐个距离求平均，得到 $\Delta=10$ 时 0.6683、$\Delta=1000$ 时 0.1576、$\Delta=4096$ 时 $-0.0537$，与上面的精确值相差都在 0.002 以内，也就是 2000 个样本的抽样误差量级。
:::

::: err “RoPE 保证注意力随距离衰减”
不对。① RoFormer 证明的是一个**上界**，衰减靠的是图 2 的数值展示，而且只画到 256；这个上界里的平均量在约 1000 之后就不再系统性下降（图 5.6b）。② PI 论文直接批评这个界可能太松（“may be too loose”），以至于没有信息量（“vacuous”）[@chen2023pi]。③ 真正能推出来的是 (5.11)：当 $\mathbf q$ 与 $\mathbf k$ 正相关、而且各对维度上相关程度相同（$\E[\mathbf k\mathbf q^\top]=\kappa I$，$\kappa>0$）时，平均打分随距离呈下降趋势；独立的 $\mathbf q,\mathbf k$ 在任何距离的期望打分都是 0。正确的说法是：**RoPE 让（各向同性地）正相关的 $\mathbf q,\mathbf k$ 的平均打分随距离呈下降趋势，但不保证任何一个具体的注意力权重随距离递减。** 模型完全可以学出一个专门看很远处某个 token 的头。
:::

:::: quiz 小练习：LLaMA 的慢针
LLaMA-7B 的 $d_h=128$，预训练长度 2048。哪些对在 2048 以内转不满一圈？最慢的一对在 $t=2048$ 转了多少度？
::: answer 看答案
$\lambda_i>2048\iff2\pi\cdot10000^{2i/128}>2048\iff10000^{i/64}>325.95\iff i>64\times\frac{\ln325.95}{\ln10000}=64\times\frac{5.7868}{9.2103}=40.21$，所以 $i=41,\dots,63$，共 23 对。最慢的一对：$2048\times\theta_{63}=2048\times1.1548\times10^{-4}=0.2365$ rad $=13.55°$。
:::
::::

::: key 本页要点
- 高维 RoPE = $d_h/2$ 个独立的二维旋转，块对角矩阵 (5.6)；打分 (5.7) 是各对“内容系数 × 相位”的和，仍只依赖 $t-j$。
- 频率 $\theta_i=b^{-2i/d_h}$ 是几何级数 (5.8)：$d_h=128$ 时波长从 6.28 到 54,410；$d_h^R=64$ 时从 6.28 到 47,117。快针分辨近邻，慢针（$\lambda_i>L_{ctx}$）在训练中转不满一圈，DeepSeek 的 64 维里有 9 对。
- 实现是两次逐元素乘加 (5.9)；相邻配对与前后半配对等价但权重不通用；旋转后的键可以直接进 KV cache。
- “远程衰减”只是一个上界的平均趋势，约 1000 之后不再系统性下降；能推出的是 (5.11)：各对维度上同样正相关的向量，平均打分呈下降趋势（不单调）；独立向量的期望打分为 0。
:::

# RMSNorm：只缩放，不居中 | LayerNorm 的两步去掉一步，丢掉的是什么 | 20

RoPE 讲完了。块里第二处改动小得多，但每一层要做两次：归一化。这一页回答三个问题：为什么要在子层之前归一化；RMSNorm 比 LayerNorm 少做了哪一步；少做这一步，模型失去了什么。

## 问题：残差主干上的数会越来越“大小不一”

[第 1 课](lesson-01-transformer-anatomy.html#p6)把残差流比作传送带：每一层从带上取件、加工、把增量放回带上。放回去的增量有大有小，经过几十层之后，主干上的向量长度可以相差很多。而每个子层（注意力、前馈层）里的矩阵只在某个尺度范围里工作得好：输入太大，softmax 会饱和；输入太小，信号淹没在噪声里。所以每个子层在读入主干之前，先把向量“调到标准尺寸”，这就是**归一化**（normalization）。

放在哪里也有讲究。GPT-2 把 LayerNorm 移到了每个子块的**输入**处，并在最后一个注意力块之后再加一层 [@radford2019gpt2]；LLaMA 沿用这种 Pre-Norm 结构，写明“对每个子层的输入做归一化，而不是对输出”，并把 LayerNorm 换成 RMSNorm [@touvron2023llama]。图 5.1 里每个子层前面的那个框就是它。主干本身从不被归一化，只有“取件”时经过一次，这让恒等映射（原样传下去的那部分）一直畅通。

## LayerNorm 回顾：两个统计量

**含义。** LayerNorm 对一个向量做两件事：减去它自己的均值，再除以它自己的标准差；最后逐分量乘一个可学的增益、加一个可学的偏置。

**数学。** 对 $\mathbf x\in\R^{d}$（一个 token 在残差主干上的向量），记均值 $\bar x=\frac1d\sum_kx_k$：

::: eq LayerNorm | (5.12)
$$\LayerNorm(\mathbf x)=\frac{\mathbf x-\bar x\,\mathbf 1}{\sqrt{\frac1d\sum_{k=1}^{d}(x_k-\bar x)^2+\epsilon}}\odot\mathbf g+\mathbf b$$
$\mathbf 1$ 是全 1 向量；$\mathbf g,\mathbf b\in\R^d$ 是可学的增益与偏置；$\epsilon$ 是防止除以 0 的小常数，GPT-2 的配置里是 $10^{-5}$，见其 Hugging Face 配置 [@hf-gpt2]。
:::

要算两个统计量：先求均值，再求（减去均值之后的）方差。在 GPU 上这是对同一个向量的两次归约（reduction）。

## RMSNorm：只保留“除以大小”

**含义。** RMSNorm 只做一件事：把向量除以它的**均方根**（root mean square，RMS），再逐分量乘增益。不减均值，也没有偏置。

**理解。** 可以把它想成音频里的“响度归一化”：把一段录音的均方根响度调到统一的标准，不改变波形的形状。LayerNorm 则多做一步，先去掉“直流偏置”（整段波形整体抬高或压低的那一部分），再调响度。RMSNorm 论文的假设是：LayerNorm 真正起作用的是后一步“重缩放”，而不是前一步“重中心化” [@zhang2019rmsnorm]。

**数学。**

::: eq RMSNorm | (5.13)
$$\RMSNorm(\mathbf x)=\frac{\mathbf x}{\operatorname{RMS}(\mathbf x)}\odot\mathbf g,\qquad \operatorname{RMS}(\mathbf x)=\sqrt{\frac1d\sum_{k=1}^{d}x_k^2+\epsilon}$$
论文的式 (4) 里没有 $\epsilon$，它是实现加的；DeepSeek 的配置取 $\epsilon=10^{-6}$（`rms_norm_eps`）。LLaMA 官方代码与 DeepSeek 的 Hugging Face 代码都先转成 float32 再算（DeepSeek-V3 的官方推理代码则直接调用 PyTorch 的 `F.rms_norm`），而且都只有增益 $\mathbf g$、没有偏置 [@llama-v1-code; @ds-configs-code]。
:::

::: ana 类比：调音量，与先去直流再调音量
**映射。** 向量 $\mathbf x$ = 一段波形的 $d$ 个采样；$\operatorname{RMS}(\mathbf x)$ = 这段波形的均方根响度；RMSNorm = 把响度调到 1；增益 $\mathbf g$ = 一台逐频段可调的均衡器（每个分量一个旋钮，学出来的）；LayerNorm 的“减均值” = 先去掉直流偏置。

**收益。** 它说明了两者的不变性差别：把整段声音放大 3 倍，调响度之后听起来一样（两者都对缩放不变）；把整段波形整体抬高 10（加直流），先去直流的 LayerNorm 不受影响，只调响度的 RMSNorm 却会把这个“抬高”当成声音的一部分。

**失效的地方。** 音频里的直流偏置通常是噪声，该去掉；残差流里各分量的公共偏移却可能携带信息，去不去掉是设计选择。另外，残差流的 $d$ 个分量并没有“采样时间”的顺序，“波形”只是个比喻。
:::

## 推导：RMSNorm 把向量放到一个球面上

RMSNorm 的几何意义可以一行推出来。

::: deriv RMSNorm 输出的长度（取 g = 1，ε = 0）
$\operatorname{RMS}(\mathbf x)=\sqrt{\tfrac1d\sum_kx_k^2}=\tfrac{1}{\sqrt d}\norm{\mathbf x}$ || 均方根就是长度除以 $\sqrt d$
$\Big\lVert\frac{\mathbf x}{\operatorname{RMS}(\mathbf x)}\Big\rVert=\frac{\norm{\mathbf x}}{\norm{\mathbf x}/\sqrt d}=\sqrt d$ || 除以一个正数，长度也除以它
:::

所以 RMSNorm（增益为 1 时）把任何非零向量都送到半径 $\sqrt d$ 的球面上，**方向不变**；增益 $\mathbf g$ 再逐轴拉伸这个球。LayerNorm 多一步：先把向量投影到“各分量之和为 0”的超平面上（减均值），再送上同一个球面（要求 $\mathbf x$ 不是各分量全相等的常数向量，否则居中后是零向量，$\epsilon=0$ 时无定义）。两者的关系是：

::: deriv LayerNorm = 先居中，再 RMSNorm（取 g = 1，b = 0，同一个 ε）
$\frac1d\sum_k(x_k-\bar x)^2+\epsilon=\operatorname{RMS}(\mathbf x-\bar x\mathbf 1)^2$ || 按 (5.13) 的定义（根号里含 $\epsilon$）：方差就是“减去均值后的向量”的均方
$\LayerNorm(\mathbf x)=\frac{\mathbf x-\bar x\mathbf 1}{\operatorname{RMS}(\mathbf x-\bar x\mathbf 1)}=\RMSNorm(\mathbf x-\bar x\mathbf 1)$ || 代入 (5.12)：分母正好是居中之后的 RMS
:::

特别地，当 $\bar x=0$ 时两者完全相同，这正是 RMSNorm 论文的原话：均值为零时，RMSNorm 与 LayerNorm 完全相等 [@zhang2019rmsnorm]。

## 不变性：缩放保住了，平移丢掉了

**缩放不变。** 对任意 $\alpha>0$，$\operatorname{RMS}(\alpha\mathbf x)=\alpha\operatorname{RMS}(\mathbf x)$（取 $\epsilon=0$），于是 $\RMSNorm(\alpha\mathbf x)=\RMSNorm(\mathbf x)$。这一条对输入整体放大、对产生输入的整个权重矩阵放大都成立，是论文式 (6)(7) 的内容 [@zhang2019rmsnorm]。严格说要求 $\alpha>0$：$\alpha<0$ 时 $\operatorname{RMS}(\alpha\mathbf x)=\abs{\alpha}\operatorname{RMS}(\mathbf x)$，输出会整体变号，论文没写这个条件。

**平移不变只有 LayerNorm 有。** 把每个分量都加上同一个常数 $c$：LayerNorm 先减均值，$c$ 被完全消掉；RMSNorm 没有这一步，$c$ 会改变输出。论文的表 1 把这一点列得很清楚：RMSNorm 对“数据重中心化”不具不变性 [@zhang2019rmsnorm]。

| 变换 | LayerNorm | RMSNorm |
|---|---|---|
| 整体缩放 $\mathbf x\to\alpha\mathbf x$（$\alpha>0$） | 不变 | 不变 |
| 整体平移 $\mathbf x\to\mathbf x+c\mathbf 1$ | 不变 | **会变** |
| 需要的统计量 | 均值、方差 | 均方 |
| 每个归一化层的参数（宽度 $d$） | $2d$（增益 + 偏置） | $d$（只有增益） |

::: ex 手算一遍，看平移怎样“挤扁”RMSNorm 的输出
$\mathbf x=(1,2,3,4)$，$\mathbf g=\mathbf 1$，$\epsilon=0$。

**RMSNorm。** $\frac14(1+4+9+16)=7.5$，$\operatorname{RMS}=\sqrt{7.5}=2.7386$；输出 $(0.3651,0.7303,1.0954,1.4606)$。验证：输出的均方根 $=1$，长度 $=\sqrt4=2$。

**LayerNorm。** 均值 2.5，居中后 $(-1.5,-0.5,0.5,1.5)$，方差 $\frac14(2.25+0.25+0.25+2.25)=1.25$，标准差 1.1180；输出 $(-1.3416,-0.4472,0.4472,1.3416)$。

**平移 $\mathbf x+10=(11,12,13,14)$。** LayerNorm：均值 12.5，居中后还是 $(-1.5,-0.5,0.5,1.5)$，输出与上面一模一样。RMSNorm：均方 $\frac14(121+144+169+196)=157.5$，RMS $=12.5499$，输出 $(0.8765,0.9562,1.0359,1.1155)$。四个分量几乎一样大：原来 $0.37$ 到 $1.46$ 的差别，被公共的偏移“挤扁”到了 $0.88$ 到 $1.12$。

**缩放 $3\mathbf x=(3,6,9,12)$。** 两者的输出都与 $\mathbf x$ 的完全相同<!-- 研究笔记 R16 -->。
:::

::: fig figures/l05-norms.svg | 图 5.7 LayerNorm 与 RMSNorm：平移与缩放下谁变谁不变 | wide
**看右列中间那一格（红色虚线框）：输入整体加 10 之后，RMSNorm 的四个输出几乎一样高，而 LayerNorm 那一列三行完全相同。** 第三行说明两者对整体缩放都不变。每根柱子上的数由生成脚本按 (5.12)(5.13) 算出，增益为 1、偏置为 0、$\epsilon=0$。
:::

## 省了什么，丢了什么

**省下的。** 少一个统计量（均值），少一组参数（偏置）。参数上几乎可以忽略：LLaMA-7B 每层两个 RMSNorm，再加最后一个，共 $2\times32+1=65$ 个，增益参数 $65\times4096=266{,}240$ 个，只占 67 亿参数的约 0.004%；换成带偏置的 LayerNorm 也不过翻一倍。时间上，RMSNorm 论文在 2019 年的实现与硬件上测得运行时间减少 7%–64%（**实测**，因模型与实现而异）[@zhang2019rmsnorm]；在今天把归一化融合进前后算子的实现里差距还有多大，本课没有核实到数据。论文还试过只用前 $p\%$ 个分量估计 RMS 的 pRMSNorm，理论上更快，但作者没有观察到一致的实际加速 [@zhang2019rmsnorm]。

**丢掉的。** 平移不变性。如果残差主干上的向量都带着一个很大的公共分量，RMSNorm 的输出会像上面的例子一样被“挤扁”，分量之间的差别相对变小。论文的主张是“重缩放不变性才是 LayerNorm 成功的原因”，这是一个**假设**，支撑它的是多项任务上精度相当的**实测**结果，而不是定理 [@zhang2019rmsnorm]。LLaMA、DeepSeek 都选了 RMSNorm，笔者据此推断，在它们的规模上这个代价没有造成可见的损失；但两家的论文里都没有 RMSNorm 与 LayerNorm 的对照实验，反过来的证据（RMSNorm 在哪些情形下明显更差）本课也没有找到。

::: key 本页要点
- Pre-Norm：每个子层读入主干之前先归一化，主干本身不归一化（GPT-2 起如此，LLaMA 沿用并换成 RMSNorm）。
- $\RMSNorm(\mathbf x)=\mathbf x/\operatorname{RMS}(\mathbf x)\odot\mathbf g$ (5.13)：不减均值、没有偏置；增益为 1 时把向量送到半径 $\sqrt d$ 的球面上，方向不变。
- $\LayerNorm(\mathbf x)=\RMSNorm(\mathbf x-\bar x\mathbf 1)$；均值为 0 时两者相等。
- 两者都对整体缩放不变；只有 LayerNorm 对整体平移不变。RMSNorm 省下一个统计量和一组偏置，代价是平移会“挤扁”输出。
:::

# SwiGLU 与无偏置：前馈层的新配方 | 门控、SiLU、8d/3 的参数匹配，以及 LLaMA 块的参数账 | 25

第三处改动在前馈层。第 1 课的前馈层是“升维、激活、降维”：$768\to3072\to768$，激活函数 GELU。现代块把它换成了三个矩阵的 **SwiGLU**，隐藏维从 $4d$ 缩到约 $\tfrac83d$。这一页讲清门控是什么、$\tfrac83$ 怎么来、实际模型为什么又不完全是 $\tfrac83$，最后把 LLaMA-7B 一整层的参数数清楚。

## 问题：前馈层占了块里三分之二的参数

按第 1 课的账，一层的注意力有 $4d^2$ 个参数（$W^Q,W^K,W^V,W^O$），前馈层有 $2\cdot d\cdot4d=8d^2$ 个。前馈层是大头。如果能在**参数量与计算量都不变**的前提下换一种前馈层、让模型变好，这就是白捡的收益。Shazeer 2020 年的短文就在做这件事：在 T5 上把前馈层换成各种“门控线性单元”的变体，控制参数量不变，比较困惑度 [@shazeer2020glu]。

## 门控线性单元：一个支路决定另一个支路放行多少

**含义。** 门控线性单元（gated linear unit，GLU）把输入同时送进两个线性映射：一个经过激活函数，当作“门”；另一个不经激活，当作“内容”。两者**逐元素相乘**，门开得大的分量多放行一些内容。

**理解。** 普通前馈层里，一个隐藏单元的“开不开”和“输出多少”由同一个数决定（激活函数作用在同一个预激活上）。GLU 把这两件事拆开：门支路决定开度，内容支路决定放行的值，像一排水龙头（映射与它失效的地方见下面的类比框）。

**数学。** Shazeer 用行向量写（$xW$）；换成本课的列向量写法，不带偏置的两种前馈层是：

::: eq 普通前馈层与 SwiGLU 前馈层 | (5.14)
$$\FFN_{\mathrm{GELU}}(\mathbf x)=W_2\,\mathrm{GELU}(W_1\mathbf x),\qquad \FFN_{\mathrm{SwiGLU}}(\mathbf x)=W_2\big(\mathrm{SiLU}(W_1\mathbf x)\odot W_3\mathbf x\big)$$
SwiGLU 中 $W_1,W_3\in\R^{d_{ff}\times d}$，$W_2\in\R^{d\times d_{ff}}$。命名对照：$W_1$ 是门（代码 `w1`，Hugging Face 叫 `gate_proj`），$W_3$ 是内容（`w3` / `up_proj`），$W_2$ 是降维（`w2` / `down_proj`）。LLaMA 与 DeepSeek 的代码都是 `w2(F.silu(w1(x)) * w3(x))` [@llama-v1-code; @ds-configs-code]。
:::

门支路用的激活函数叫 SiLU，也叫 Swish：

::: eq SiLU（β = 1 的 Swish） | (5.15)
$$\mathrm{SiLU}(z)=z\,\sigma(z)=\frac{z}{1+e^{-z}},\qquad \mathrm{Swish}_\beta(z)=z\,\sigma(\beta z)$$
$\sigma$ 是 sigmoid。Shazeer 的 SwiGLU 定义为 $\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$，用在前馈层时去掉偏置、取 $\beta=1$，就是 (5.14) [@shazeer2020glu]。
:::

::: ex 三种激活函数的几个值，和一个两维的 SwiGLU
**激活函数**<!-- 研究笔记 R17 -->：

| $z$ | −2 | −1 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|
| ReLU | 0 | 0 | 0.5 | 1 | 2 | 4 |
| GELU | −0.0455 | −0.1587 | 0.3457 | 0.8413 | 1.9545 | 3.9999 |
| SiLU | −0.2384 | −0.2689 | 0.3112 | 0.7311 | 1.7616 | 3.9281 |

SiLU 在负半轴不是 0，而是一个小的负值，最低点在 $z=-1.2785$，值 $-0.2785$；正半轴很快接近 $z$ 本身。

**一个两维的 SwiGLU。** 设某个 token 的门支路预激活 $W_1\mathbf x=(2,-1)$，内容支路 $W_3\mathbf x=(0.5,3)$。门：$\mathrm{SiLU}(2)=1.7616$，$\mathrm{SiLU}(-1)=-0.2689$。逐元素相乘：$(1.7616\times0.5,\ -0.2689\times3)=(0.8808,\ -0.8068)$。第二个分量的内容值 3 很大，但门几乎关着（还略微反号），放行的只有 $-0.81$；第一个分量内容不大，门却开得很大。最后再乘 $W_2$ 降回 $d$ 维。
:::

::: ana 类比：一排水龙头
**映射。** 隐藏层的 $d_{ff}$ 个分量 = $d_{ff}$ 根水管；内容支路 $W_3\mathbf x$ = 每根水管里的水压；门支路 $\mathrm{SiLU}(W_1\mathbf x)$ = 这根水管上的龙头拧开多少；逐元素相乘 = 实际流出来的水；$W_2$ = 把所有水管的出水按不同比例汇进 $d$ 个出口。

**收益。** 它说明了为什么要多一个矩阵：GELU 前馈层里，一个隐藏单元“开多大”和“流多少”由同一个预激活决定；GLU 让它们由两组独立的权重 $W_1$、$W_3$ 分别学，同一个 token 可以“内容很大但门关着”（上例第二个分量），也可以“内容不大但门全开”（第一个分量）。

**失效的地方。** 真龙头的开度在 0 与 1 之间，SiLU 门却没有上限（$\mathrm{SiLU}(4)=3.93$），在负半轴还会取到小的负值（最低 $-0.2785$）。所以门既能把内容**放大**，也能让它轻微**反号**，更像一个每根水管各自可调、可正可负的增益旋钮，而不是开关。另外，两条支路是从同一个 $\mathbf x$ 同时算出来的，“龙头”并不看“水压”是多少。
:::

::: fig figures/l05-swiglu.svg | 图 5.8 SwiGLU 前馈层（LLaMA-7B 的维度）与参数匹配 | wide
**先看左边的两条支路在 ⊙ 处汇合，再看右边：三个窄方块的总面积等于两个宽方块。** 左：一个 token 的 4096 维输入同时进入门 $W_1$ 与内容 $W_3$（都升到 11008 维），门经过 SiLU 后与内容逐元素相乘，再由 $W_2$ 降回 4096 维。右：方块按比例画，高为 $d=4096$，宽为矩阵的另一维；面积就是参数量。$d_{ff}=\tfrac83d$ 时三个方块的面积之和恰好等于 GELU 前馈层的两个 $d\times4d$ 方块；LLaMA-7B 把它向上取整到 11008，多出 0.78%。
:::

## 参数匹配：8d/3 从哪里来

SwiGLU 多了一个矩阵。要和普通前馈层公平比较，就得把隐藏维缩小，让参数量相等：

::: deriv 让 SwiGLU 与 4d 的 GELU 前馈层参数相等
$2\cdot d\cdot 4d=8d^2$ || 普通前馈层：$W_1\in\R^{4d\times d}$ 与 $W_2\in\R^{d\times4d}$，不计偏置
$3\cdot d\cdot d_{ff}$ || SwiGLU：$W_1,W_3\in\R^{d_{ff}\times d}$，$W_2\in\R^{d\times d_{ff}}$
$3d\,d_{ff}=8d^2\ \Longrightarrow\ d_{ff}=\tfrac83d=\tfrac23\cdot4d$ || 令两者相等，两边除以 $3d$
:::

计算量也随之相等：每个 token 经过一个矩阵，每个参数恰好做一次乘加（2 次浮点运算），两种前馈层都是约 $16d^2$ 次；SiLU 与逐元素相乘只有 $d_{ff}$ 量级，可以忽略。Shazeer 的原话是把 $d_{ff}$ “reduce … by a factor of 2/3”，意思是**乘以** $2/3$，不是减少三分之二：在 T5-base（$d=768$）上，他把 3072 改成 2048，$2\times768\times3072=4{,}718{,}592=3\times768\times2048$ [@shazeer2020glu]。LLaMA 论文写的是“$\tfrac23\cdot4d$ 而不是 PaLM 的 $4d$” [@touvron2023llama]；PaLM 用了 SwiGLU 但隐藏维仍是 $4d$，前馈层参数因此是 $3\cdot d\cdot4d=12d^2$，比 $8d^2$ 多 50% [@chowdhery2022palm]。

::: ex LLaMA-7B 的 11008 是怎么来的
$d=4096$。$\tfrac83\times4096=10922.67$。LLaMA 官方代码先取整 `int(2 * 4d / 3)` $=10922$，再向上取到 `multiple_of = 256` 的倍数：$10922/256=42.66$，向上取 43，$43\times256=11008$ [@llama-v1-code]。

参数：$3\times4096\times11008=135{,}266{,}304$，而 $8d^2=134{,}217{,}728$，多 0.78%。其余尺寸同理：13B（$d=5120$）$13653.33\to13824$，多 1.25%；33B（$d=6656$）$17749.33\to17920$；65B（$d=8192$）$21845.33\to22016$。宽度比 $d_{ff}/d$ 分别是 2.6875、2.7000、2.6923、2.6875，都略大于 $8/3=2.6667$<!-- 研究笔记 R13 -->。代码里 `multiple_of` 的注释只说要让 SwiGLU 的隐藏维成为“2 的较大幂次”的倍数（“multiple of large power of 2”）[@llama-v1-code]；这样做通常是为了让矩阵形状对 GPU 友好，这一层理由是笔者的推测，代码没有写。
:::

::: fig figures/l05-ffn-width.svg | 图 5.9 各模型前馈层的宽度比 d_ff / d | wide
**看红色虚线附近：LLaMA 各尺寸与 DeepSeek LLM 都贴着 $8/3$，而 DeepSeek-V2/V3 的稠密层并不遵守它。** DeepSeek LLM 的 7B 与 67B 用的是与 LLaMA 相同的规则（11008/4096 = 22016/8192 = 2.6875）；V2 唯一的稠密层是 2.4，V3 前 3 个稠密层是 2.5714；MoE 里的单个专家更窄（0.3、0.2857），那是第 7、10、11 课的设计。数据来自各模型的论文、代码与配置<!-- 研究笔记 R13、R14 -->[@llama-v1-code; @deepseek2024llm; @ds-llm-hf; @ds-configs-code]。
:::

::: warn “SwiGLU 的隐藏维就是 8d/3”
只对“与 $4d$ 的 GELU 前馈层参数相等”这一设计目标成立。LLaMA 实际向上取整（11008 等）；PaLM 用 $4d$；DeepSeek-V2/V3 的稠密层是 $2.4d$ 与 $2.571d$；MoE 专家更窄。读配置时以 `intermediate_size` 为准，不要从 $d$ 反推。
:::

## 效果：实测更好，原因不明

Shazeer 的表 1 给出 T5-base 规模、预训练 524,288 步之后的留出集对数困惑度（越低越好）：ReLU 基线 1.677，SwiGLU 1.636，GEGLU 1.633 最好，ReGLU 1.645 [@shazeer2020glu]。这是**实测**：同样的参数与计算量，门控变体一致地更好。至于为什么，作者在结论里坦白没有解释，把成功归于“divine benevolence”（天意）[@shazeer2020glu]。直观的说法（门控让每个隐藏单元的开度与输出值分开学、SiLU 在负半轴保留一点梯度）都只是解释性的猜想，不是证明。

## 去掉偏置

第四处改动最简单：所有线性层、所有归一化层都不要偏置。明文出处是 PaLM：稠密层与 LayerNorm 里都不用偏置，并报告这提高了大模型训练的稳定性（**实测**）[@chowdhery2022palm]。LLaMA 论文正文没有提这一点，是官方代码里所有线性层都写了 `bias=False` [@llama-v1-code]；DeepSeek-V3 官方推理代码的线性层默认也不带偏置 [@ds-configs-code]。

偏置省下的参数微不足道（笔者算的）：如果 LLaMA-7B 每层都带偏置，注意力四个投影 $4\times4096$、SwiGLU 三个矩阵 $2\times11008+4096$、两个归一化 $2\times4096$，合计每层 50,688 个，32 层约 162 万，只占 67 亿的 0.024%。所以去掉偏置的理由是稳定性，不是省参数。

## 组装：LLaMA 块与它的参数账

四处改动合在一起，一层 LLaMA 式块写成：

::: eq LLaMA / DeepSeek LLM 式的一层 | (5.16)
$$\begin{aligned}\mathbf h&=\mathbf x+\Attn\big(\RMSNorm_{\mathrm{attn}}(\mathbf x)\big)\\ \mathbf x_{\mathrm{out}}&=\mathbf h+\FFN_{\mathrm{SwiGLU}}\big(\RMSNorm_{\mathrm{ffn}}(\mathbf h)\big)\end{aligned}$$
注意力内部用 RoPE 旋转 $\mathbf q,\mathbf k$（不转 $\mathbf v$）；所有线性层无偏置；最后一层之后再接一个 RMSNorm，然后是输出投影 [@llama-v1-code]。
:::

::: ex 数清 LLaMA-7B 的参数
$d=4096$，32 层，$d_{ff}=11008$，词表 32000，输入嵌入与输出投影不共享权重（Hugging Face 社区转存的配置：`vocab_size` 32000，`tie_word_embeddings` false）[@huggyllama-configs]。

- 每层注意力：$4d^2=4\times4096^2=67{,}108{,}864$。
- 每层 SwiGLU：$3\times4096\times11008=135{,}266{,}304$。
- 每层两个 RMSNorm：$2\times4096=8{,}192$。
- 一层合计 $202{,}383{,}360$；32 层 $6{,}476{,}267{,}520$。
- 最后的 RMSNorm $4096$；嵌入与输出投影 $2\times32000\times4096=262{,}144{,}000$。

总计 $6{,}738{,}415{,}616\approx6.74\times10^9$，与 LLaMA 论文表 2 的“6.7B”一致 [@touvron2023llama]。前馈层占每层参数的 $135.3/202.4=66.8\%$，正是本页开头说的“三分之二”。每个 token 在一层 SwiGLU 上的前向计算约 $2\times135{,}266{,}304=270{,}532{,}608$ 次浮点运算<!-- 研究笔记 R15 -->。
:::

::: key 本页要点
- SwiGLU 前馈层 $W_2(\mathrm{SiLU}(W_1\mathbf x)\odot W_3\mathbf x)$ (5.14)：门支路决定开度，内容支路决定放行的值。
- 三个矩阵与两个 $d\times4d$ 矩阵参数相等，要求 $d_{ff}=\tfrac83d$；LLaMA 向上取整到 256 的倍数（7B：11008，多 0.78%）；DeepSeek-V2/V3 的稠密层不遵守 8/3。
- 效果是实测的（1.636 对 1.677），原因没有定论；去掉偏置的依据是 PaLM 报告的训练稳定性。
- LLaMA-7B：每层 2.02 亿参数，其中前馈层占 66.8%，32 层加嵌入共 67.4 亿。
:::

# 上下文扩展（上）：外推为何崩溃，PI 与 NTK-aware | 把时钟拨慢的两种办法，以及各自的毛病 | 25

积木讲完了，回到 [§1](#p1) 的第二张账单：模型在 4K 长度上预训练，却要读 128K。这一页先说清直接外推为什么会崩，再讲两种早期的补救办法：位置插值（PI）把所有指针一样拨慢；NTK-aware 改底数，快针几乎不动、慢针拨慢最多。下一页的 YaRN 是在它们的基础上改出来的。

## 问题：慢针会转到从没见过的角度

用 (5.7) 的写法，RoPE 的打分是相对距离的函数 $a(\Delta)=\operatorname{Re}\sum_i\tilde q_i\overline{\tilde k_i}e^{\mathrm i\Delta\theta_i}$。预训练时 $\Delta$ 只在 $[0,L_{ctx})$ 里取值，模型只在这一段上见过、学过这个函数。[§3](#p3) 算过：DeepSeek 的 64 维 RoPE 里有 9 根慢针在 4096 以内连一圈都没转满，最慢的那根只见过 $0°$ 到 $31.3°$。如果直接把序列拉长到 16 万，最慢的那根会转到 $1252°$（约 3.5 圈），其余几根慢针也都转出了训练时见过的那段弧：这些角度都是模型没见过的输入组合。

没见过会怎样？PI 论文给了两方面的证据。一是**实测**：把 LLaMA 直接用于更长的序列，困惑度可以涨到 $10^3$ 以上，与没训练过的模型相当。二是一个构造出来的例子：一个在 $[0,2048]$ 内取值大约在 $\pm1$ 之间的打分函数，外推到区间之外可以大到 8000 以上 [@chen2023pi]。打分一旦出现这种量级的异常值，softmax 会把几乎全部权重压到一两个位置上，注意力就坏了。

## 外推与插值

**含义。** **外推**（extrapolation）是直接使用训练时没出现过的位置；**插值**（interpolation）是把新的、更长的位置范围**压缩回**训练时见过的范围，让每个位置都落在已知区间内。

**理解。** 想象一把只有 2048 格的尺子。外推是把尺子外面的长度也硬读出来；插值是把要量的东西缩小一半，让它整个落在尺子上，代价是每一格要分辨更细的差别。

::: fig figures/l05-pi.svg | 图 5.10 外推与位置插值：LLaMA 从 2048 扩到 4096 | wide
**看右边的两列钟面：外推时慢针越界（红色扇形），插值时慢针不越界，但快针每个 token 只转一半。** 左边三条数轴是 PI 论文图 1 的思路：预训练只见过 0 到 2048；外推把 2048 到 4096 这些“没见过的位置”直接喂给模型；位置插值把 4096 个位置按 $t/2$ 全部塞进 $[0,2048]$。右边按 LLaMA 的 $d_h=128$、$b=10000$ 算出两根针：最慢的一对在 2048 处转到 $13.6°$，外推到 4096 会到 $27.1°$；最快的一对每个 token 转 $57.3°$，插值后只剩 $28.6°$。据 PI 论文图 1 改绘 [@chen2023pi]。
:::

## 位置插值（PI）

**数学。** PI 的做法只有一行：把位置 $t$ 换成 $t\cdot L_{ctx}/L'_{ctx}=t/s$ 再做 RoPE [@chen2023pi]。因为 RoPE 的角度是 $t\theta_i$，“位置除以 $s$”与“每个频率除以 $s$”完全等价：

::: eq 位置插值 | (5.17)
$$f'(\mathbf x,t)=f\Big(\mathbf x,\ \frac{t\,L_{ctx}}{L'_{ctx}}\Big)\quad\Longleftrightarrow\quad \theta_i'=\frac{\theta_i}{s}\ \ \text{（对所有 }i\text{）},\qquad s=\frac{L'_{ctx}}{L_{ctx}}$$
$f$ 是 RoPE 旋转（对 $\mathbf q$、$\mathbf k$ 同样处理）。位置不再是整数，这没关系：RoPE 的角度对任何实数位置都有定义。
:::

PI 还给了一个理论理由，说明插值比外推“安全”得多。它比较了两件事：在两个相邻整数位置之间插值时打分函数最多偏离多少，和外推时打分函数最多能有多大。

::: thm PI 的插值误差界（PI 论文定理 2.1）
这里沿用 PI 论文的记号：$c$ 就是本课的底数 $b$，$d$ 是带 RoPE 的维度（本课的 $d_h$，不是模型宽度），$h_i=\tilde q_i\overline{\tilde k_i}$ 是 [§3](#p3) 的内容系数。

设 $a(\Delta)=\operatorname{Re}\sum_{i=0}^{d/2-1}h_ie^{\mathrm i\Delta\theta_i}$，$\theta_i=c^{-2i/d}$，$\Delta\in[\Delta_1,\Delta_2]$，$a_{\text{linear}}$ 是 $a(\Delta_1)$ 与 $a(\Delta_2)$ 之间的线性插值，则

$$\big\lvert a(\Delta)-a_{\text{linear}}(\Delta)\big\rvert\le d\Big(\max_i\abs{h_i}\Big)\frac{(\Delta-\Delta_1)(\Delta_2-\Delta)}{8\ln c}.$$

相邻整数之间 $(\Delta-\Delta_1)(\Delta_2-\Delta)\le\frac14$，取 $c=10000$ 得上界 $\frac{d}{32\ln c}\max_i\abs{h_i}\approx\frac{d\max_i\abs{h_i}}{294.73}$。论文把它与 RoFormer 式的外推界（数值上至少 $2d\max_i\abs{h_i}$）比较，说插值界“至少小 $2\cdot294.73\sim600$ 倍” [@chen2023pi]。
:::

::: warn 笔者核算：定理 2.1 的常数偏小，结论不变
定理的证明（PI 论文附录 A）先把 $\abs{a''(\Delta)}$ 放大到 $\max_i\abs{h_i}\sum_i\theta_i^2$，这一步没有问题；接着用“$x<0$、$c>1$ 时 $c^x\le1+x\ln c$”把 $\sum_i\theta_i^2\le\frac{1}{1-c^{-4/d}}$ 继续放大成 $\frac{d}{4\ln c}$。这个不等式的方向写反了：$e^y$ 是凸函数，对所有 $y$ 都有 $e^y\ge1+y$，所以 $c^{-4/d}\ge1-\frac4d\ln c$，从而 $\frac{1}{1-c^{-4/d}}\ge\frac{d}{4\ln c}$。$d=128$、$c=10^4$ 时，$\sum_i\theta_i^2=3.998$，而 $\frac{d}{4\ln c}=3.474$<!-- 研究笔记 E18 -->。

定理本身确实会被违反。取 $h_i=e^{-\mathrm i\theta_i/2}$（每个 $\abs{h_i}=1$，令 $\tilde k_i=1$、$\tilde q_i=e^{-\mathrm i\theta_i/2}$ 就能得到），在 $\Delta_1=0$、$\Delta_2=1$ 的中点，$a(\tfrac12)-a_{\text{linear}}(\tfrac12)=\sum_i\big(1-\cos\tfrac{\theta_i}{2}\big)=0.494$，比定理给的 $\frac{128}{294.73}=0.434$ 大。

改正的办法是不做最后那一步放大，直接保留 $\sum_i\theta_i^2$：

$$\big\lvert a(\Delta)-a_{\text{linear}}(\Delta)\big\rvert\le\Big(\max_i\abs{h_i}\Big)\Big(\sum_{i=0}^{d/2-1}\theta_i^2\Big)\frac{(\Delta-\Delta_1)(\Delta_2-\Delta)}{2}.$$

其中 $\sum_i\theta_i^2$ 比论文用的 $\frac{d}{4\ln c}$ 大，但不超过 $\frac{d}{4\ln c}+1$：$\sum_i\theta_i^2\le\frac{1}{1-e^{-y}}$（$y=\frac4d\ln c>0$），而 $\frac{1}{1-e^{-y}}<\frac1y+1$ 两边同乘正数 $y(1-e^{-y})$ 后等价于 $(1+y)e^{-y}<1$，即 $1+y<e^y$，这正是上面那条凸性不等式。相邻整数之间的界变成 $\frac{3.998}{8}\max_i\abs{h_i}=0.500\max_i\abs{h_i}\approx\frac{d\max_i\abs{h_i}}{256}$，与外推界之比约 $2\times256=512$ 倍，而不是 589 倍。这里的“外推界至少 $2d\max_i\abs{h_i}$”是论文的数值观察，不是定理：外推界是 $2\max_i\abs{h_i}\sum_{m}\abs{S_m}$（[§3](#p3) 的部分和），$d=128$ 时 $\sum_m\abs{S_m}$ 在 $\Delta\le4096$ 内最小约 143（$\approx1.1d$），但更远处可以降到约 77（$\approx0.6d$），那里的比值只有约 300 倍。“插值比外推安全几百倍”这个结论不受影响。
:::

::: ex 把 LLaMA 从 2048 扩到 32768
PI 论文把 LLaMA 7B 到 65B 从 2048 扩到最长 32768 的几种长度（实验里出现的是 8192、16384、32768，$s$ 最大为 16），大多数设置只微调 1000 步（7B/13B 学习率 $2\times10^{-5}$，33B/65B 为 $10^{-5}$）[@chen2023pi]。取最大的 $s=16$，用 (5.17) 看两根针：

- 最慢的一对（$d_h=128$，$i=63$）：扩展后在 $t=32768$ 处的角度 $=32768\times\theta_{63}/16=2048\times\theta_{63}$，恰好等于预训练时的最大角度 $13.55°$。**不越界**，这正是插值的目的。
- 最快的一对（$i=0$）：原来相邻两个 token 差 $57.3°$，现在只差 $57.3°/16=3.58°$。

顺带一个数：定理里的 294.73 就是 $32\ln10000=32\times9.2103=294.73$；600 倍是 $2\times294.73=589.5$ 的约数<!-- 研究笔记 R12 -->（按上面警示框改正常数后约 512 倍）。论文 §3 开头写“最多扩到 32 倍”，但全文的实验都是 16 倍<!-- 研究笔记 E10 -->。
:::

**失败模式：快针被挤在一起。** PI 对所有维度一视同仁。最快的几对原本靠“相邻 token 差一大截角度”来分辨近邻，压缩 $s$ 倍之后，这个差别也缩小 $s$ 倍。YaRN 论文对 PI 的批评正是这一点：PI 把所有 RoPE 维度同等地拉伸（“stretches all RoPE dimensions equally”）[@peng2023yarn]。对 DeepSeek 的 $s=40$，快针相邻只剩 $57.3°/40=1.43°$。这也解释了为什么 PI 需要微调：模型要重新学会在更密的刻度上分辨近邻。

## NTK-aware：不缩位置，改底数

2023 年 6 月，网友 bloc97 在 Reddit 上发帖，提出一个不同的办法：改 RoPE 的底数 $b$，而不是缩放位置（“changes the base of the RoPE instead of the scale”）。帖子正文说，LLaMA 7B 用到 4096 的上下文连微调都不需要，困惑度退化很小；标题则说能扩到 8k 以上 [@bloc97-2023ntk]。帖子的正文里没有写公式；下面的公式来自 YaRN 论文附录对它的形式化，以及 YaRN 的参考代码 [@peng2023yarn; @yarn-code]。

**思路。** 底数变大，所有频率都变慢，但慢得不一样多：$\theta_i=b^{-2i/d_h}$ 里指数 $2i/d_h$ 越大，改底数的影响越大。$i=0$ 时指数为 0，最快的针完全不动；最慢的针变化最大。只要把新底数选成“让最慢的那根针恰好慢 $s$ 倍”，最慢的那根针就像 PI 一样被插值，快针则保持原样，中间的针慢下来的倍数介于 1 与 $s$ 之间（这正是下面要说的毛病）。

::: deriv NTK-aware 的新底数
$\theta'_{d_h/2-1}=\frac{\theta_{d_h/2-1}}{s}$ || 要求：最慢的一对（$i=d_h/2-1$）恰好被插值 $s$ 倍
$(b')^{-(d_h-2)/d_h}=\frac1s\,b^{-(d_h-2)/d_h}$ || 代入 $\theta_i=b^{-2i/d_h}$，$2i=d_h-2$
$(b')^{(d_h-2)/d_h}=s\,b^{(d_h-2)/d_h}$ || 两边取倒数（YaRN 附录 A.2 式 (21)）
$b'=b\cdot s^{\,d_h/(d_h-2)}$ || 两边取 $d_h/(d_h-2)$ 次方
$\theta'_i=(b')^{-2i/d_h}=\theta_i\cdot s^{-2i/(d_h-2)}$ || 代回每一对：缩放系数随 $i$ 按几何级数从 1 降到 $1/s$
:::

::: eq NTK-aware 的频率 | (5.18)
$$b'=b\cdot s^{\frac{d_h}{d_h-2}},\qquad \theta'_i=\theta_i\cdot s^{-\frac{2i}{d_h-2}}$$
代码里就是 `base = base * alpha ** (dim / (dim-2))`，其中 `alpha` 即 $s$ [@yarn-code]。
:::

::: ex 算两个新底数，再看它的毛病
**$d_h=128$，$s=8$。** $b'=10000\times8^{128/126}=82{,}684.6$。验证：$(b'/b)^{126/128}=8$，最慢的一对恰好慢 8 倍。

**DeepSeek 的设置，$d_h^R=64$，$s=40$。** $b'=10000\times40^{64/62}=450{,}546.2$。每一对的缩放 $40^{-2i/62}$：$i=0$ 为 1，$i=8$ 为 0.386，$i=16$ 为 0.149，$i=23$ 为 0.0648，$i=31$ 为 0.025<!-- 研究笔记 R6 -->。

**毛病：中间的慢针仍然越界。** 看 $i=23$：它的波长 4711.7 已经大于 4096，是一根“慢针”，本应被完全插值（缩放 $1/40=0.025$）。NTK-aware 只把它缩到 0.0648。扩展到 163,840 时，它的最大角度 $=163840\times\theta_{23}\times0.0648$，是预训练时最大角度 $4096\times\theta_{23}$ 的 $163840\times0.0648/4096=2.59$ 倍：仍在外推<!-- 研究笔记 R7 -->。9 根慢针里，只有最后一根 $i=31$ 恰好不越界。
:::

YaRN 论文对 NTK-aware 的评价正是这一点：一部分维度被轻微外推到“越界”的值（“some dimensions are slightly extrapolated to "out-of-bound" values”），因此用它做微调的效果不如 PI；而且最合适的底数往往要靠经验搜索，比如 Code Llama 就把底数手动设成了 100 万 [@peng2023yarn]。

::: key 本页要点
- 直接外推：慢针转到训练中没见过的角度，打分函数可以出现巨大的异常值，困惑度可达 $10^3$ 以上。
- PI：所有频率除以 $s$ (5.17)。慢针不再越界，插值误差界比外推界小几百倍（论文说约 600 倍；改正其证明里一处方向写反的不等式后约 500 倍）；代价是快针相邻的角度差也缩小 $s$ 倍，需要约 1000 步微调。
- NTK-aware：改底数 $b'=b\,s^{d_h/(d_h-2)}$ (5.18)，快针不动、最慢的针恰好慢 $s$ 倍、中间几何过渡；代价是中间的慢针仍被外推（DeepSeek 设置下 $i=23$ 越界 2.59 倍）。
:::

# YaRN 与 DeepSeek 的 128K | 按波长分段插值、注意力温度，以及 V2/V3 的真实设置 | 30

YaRN（Yet another RoPE extensioN method）由 Peng 等人在 2023 年提出，收于 ICLR 2024 [@peng2023yarn]。DeepSeek-V2 与 V3 都用它把 4K 的预训练上下文扩到 128K [@deepseek2024v2; @deepseek2024v3]。这一页先讲它的两个部分：按波长分段插值（论文叫 NTK-by-parts）和注意力温度；然后把 DeepSeek 的真实设置代进去，算出 64 维 RoPE 里每一对频率的命运。

## 思路：快针不碰，慢针全插值，中间过渡

**问题。** 上一页的两个办法各有一半对：PI 保证慢针不越界，却把快针也挤扁了；NTK-aware 保住了快针，却让中间的慢针越界。能不能按每根针的具体情况分别处理？

**图景。** 回到多频率时钟（[§3](#p3)）。一根针在预训练的 $L_{ctx}$ 个位置里转了多少圈，决定了它该怎么处理：

- 转了很多圈的快针：它的角度早已“绕过”无数次，只携带短距离的**相对**信息；扩展上下文不会让它遇到新角度，所以**原封不动**。
- 连一圈都没转满的慢针：按 YaRN 论文的推测，它的角度被模型当作绝对位置用（见 [§3](#p3) 的钟面类比）；扩展后会越界，所以**完全插值**（除以 $s$）。
- 介于两者之间的：按比例混合。

YaRN 论文的原话是，不要去碰那些只编码相对位置信息的维度（“important to not touch the dimensions that only encode relative positional information”）[@peng2023yarn]。

## 定义：转圈数与斜坡函数

**含义。** 给每一对算一个“转圈数” $r_i$：它在预训练长度内转过几圈。再用一个斜坡函数 $\gamma(r)$ 把转圈数翻译成“保留原频率的比例”。

**数学。** 波长 $\lambda_i=2\pi/\theta_i$（YaRN 式 (8)），转圈数与斜坡（式 (10)(11)）：

::: eq 转圈数 | (5.19)
$$r_i=\frac{L_{ctx}}{\lambda_i}=\frac{L_{ctx}\,\theta_i}{2\pi}$$
$r_i$ 就是第 $i$ 对在 $L_{ctx}$ 个位置内转过的圈数。$r_i<1$ 即 $\lambda_i>L_{ctx}$，是 [§3](#p3) 所说的“慢针”。
:::

::: eq 斜坡函数与 NTK-by-parts | (5.20)
$$\gamma(r)=\begin{cases}0,&r<\alpha\\[2pt]1,&r>\beta\\[2pt]\dfrac{r-\alpha}{\beta-\alpha},&\text{其他}\end{cases}\qquad\qquad \theta_i'=\Big[\big(1-\gamma(r_i)\big)\frac1s+\gamma(r_i)\Big]\theta_i$$
$\gamma=1$：保留原频率；$\gamma=0$：频率除以 $s$（与 PI 相同）；中间线性混合两种频率。$\alpha,\beta$ 是两个“转圈数阈值”，YaRN 对 LLaMA 家族推荐 $\alpha=1$、$\beta=32$（这里的 $\alpha,\beta$ 只表示 YaRN 的两个转圈数阈值，与 §4 表示缩放倍数的 $\alpha$、GRPO 的 $\beta$、负载均衡的 $\alpha$ 都无关）。
:::

YaRN 的定义是“NTK-by-parts + 注意力温度”（论文定义 2）[@peng2023yarn]。注意它**不改底数**；改底数的是 NTK-aware。把 YaRN 说成“一种改底数的方法”是常见的误解<!-- 研究笔记 M1 -->。

## DeepSeek 的设置：64 维里谁不动、谁除以 40

DeepSeek-V2/V3 的设置是 $d_h^R=64$、$b=10000$、$L_{ctx}=4096$、$s=40$、$\alpha=1$、$\beta=32$ [@deepseek2024v2; @deepseek2024v3; @ds-configs-code]。要知道哪几对落在哪一段，只需把“$r_i$ 等于某个阈值”反解成下标 $i$。YaRN 的参考代码就是这么做的（函数 `find_correction_dim`）[@yarn-code]：

::: deriv 从转圈数阈值反解出下标
$r=\frac{L_{ctx}}{2\pi\,b^{2i/d_h}}$ || (5.19) 代入 $\lambda_i=2\pi b^{2i/d_h}$
$b^{2i/d_h}=\frac{L_{ctx}}{2\pi r}$ || 两边同乘 $b^{2i/d_h}/r$
$i(r)=\frac{d_h\,\ln\big(L_{ctx}/(2\pi r)\big)}{2\ln b}$ || 取对数，解出 $i$
$i(32)=\frac{64\ln(4096/(64\pi))}{2\ln10000}=10.47$ || 代入 DeepSeek 的数，取 $r=\beta=32$
$i(1)=\frac{64\ln(4096/(2\pi))}{2\ln10000}=22.51$ || 取 $r=\alpha=1$
:::

::: eq 反解下标（DeepSeek：10.47 与 22.51） | (5.21)
$$i(r)=\frac{d_h\,\ln\!\big(L_{ctx}/(2\pi r)\big)}{2\ln b}$$
$r_i$ 随 $i$ 单调递减，所以 $i<i(\beta)$ 的对 $r_i>\beta$（保留），$i>i(\alpha)$ 的对 $r_i<\alpha$（全插值）。
:::

于是 32 对分成三段：$i=0,\dots,10$（11 对、22 维）**原封不动**；$i=23,\dots,31$（9 对、18 维）**频率全部除以 40**；$i=11,\dots,22$（12 对、24 维）**过渡**。全插值的 9 对，恰好就是 [§3](#p3) 找出的那 9 根在 4K 内转不满一圈的慢针。

::: fig figures/l05-yarn-bands.svg | 图 5.11 DeepSeek 的 YaRN：32 对 RoPE 频率中哪些保留、哪些插值 | wide
**看右边红色区域：9 根慢针的新波长（蓝点）都被拉长了 40 倍，全部高过 163,840 那条蓝色虚线。** 黑色短横是原波长，蓝点是 YaRN 之后的波长（按 DeepSeek 所用的参考代码算）。左边绿色区域的 11 对完全不动；中间 12 对按斜坡部分拉长。两条黑色虚线是分界：$\lambda=4096/32=128$（$r=\beta$）和 $\lambda=4096$（$r=\alpha$）。结果是：扩展到 163,840 之后，慢针在新的范围里仍然转不满一圈，与预训练时的处境完全相同<!-- 研究笔记 R3、R8 -->。
:::

::: ex 过渡区里的一对：i = 16
$\theta_{16}=10000^{-32/64}=0.01$，$\lambda_{16}=628.32$，$r_{16}=4096/628.32=6.519$。

**按论文式 (5.20)。** $\gamma=(6.519-1)/(32-1)=0.17803$；缩放 $=(1-0.17803)/40+0.17803=0.02055+0.17803=0.19858$；新频率 $\theta'_{16}=1.986\times10^{-3}$，新波长 $628.32/0.19858=3164$。

**按 DeepSeek 实际用的参考代码。** 代码把分界取整：`low = floor(10.47) = 10`，`high = ceil(22.51) = 23`，混合权重对**下标**线性：$1-(16-10)/(23-10)=0.5385$；缩放 $=(1-0.5385)/40+0.5385=0.55$；新频率 $5.5\times10^{-3}$，新波长 1142.4<!-- 研究笔记 R8 -->。

同一对频率，两种写法的结果差了 2.8 倍。
:::

::: warn 论文的斜坡与代码的斜坡不一样
论文的 $\gamma$ 对转圈数 $r$ 线性（而 $r$ 随 $i$ 指数变化）；YaRN 参考代码与 DeepSeek 的代码对下标 $i$ 线性，外加 floor/ceil 取整 [@yarn-code; @ds-configs-code]。两者“全保留 / 过渡 / 全插值”的分界完全相同（DeepSeek 设置下都是 11 / 12 / 9 对），但过渡区的混合权重差很多，代码版保留得更多（图 5.12）。所以准确的说法是“DeepSeek 用了 YaRN 的参考实现”，而不是“用了论文式 (11)”<!-- 研究笔记 E16 -->。DeepSeek 内部训练框架是否也是这个实现，开源代码之外无从核实。
:::

::: fig figures/l05-ramp.svg | 图 5.12 YaRN 的斜坡函数：论文式与参考代码
**看竖线两侧：$r<1$ 的点两种写法都是 0，$r>32$ 的点都是 1；只有中间 12 对不同，空心蓝圈（代码）普遍高于实心点（论文）。** 横轴是转圈数 $r_i$（对数刻度），纵轴是保留原频率的权重 $\gamma$。32 个点是 DeepSeek 64 维 RoPE 的 32 对频率；$i=16$ 处论文给 0.178、代码给 0.538。
:::

::: fig figures/l05-scale-methods.svg | 图 5.13 四种方法对每一对频率的缩放（DeepSeek 设置） | wide
**看红色区域里紫线高出底部虚线的那一段：那是 NTK-aware 让慢针越界的地方；YaRN 的两条线在那里都贴着 1/40。** 纵轴是 $\theta_i'/\theta_i$（对数刻度），1 表示保持原频率，0.025 表示完全插值。PI 是一条水平线；NTK-aware 是一条从 1 到 0.025 的斜直线（几何级数在对数轴上是直线）；YaRN 在 $i\le10$ 与 PI 相反地保持 1，在 $i\ge23$ 与 PI 一致。所有数值由脚本按 (5.17)(5.18)(5.20) 与参考代码算出<!-- 研究笔记 R6、R8 -->。
:::

## 注意力温度：插值之后把 softmax 调尖一点

YaRN 的第二部分是一个标量。插值之后，YaRN 把 softmax 改为

::: eq YaRN 的注意力温度 | (5.22)
$$\softmax\!\Big(\frac{\mathbf q_t'^{\top}\mathbf k_j'}{t_{\mathrm{Y}}\sqrt{d_h}}\Big),\qquad \mu:=\sqrt{\frac1{t_{\mathrm Y}}}=0.1\ln s+1$$
$\mathbf q',\mathbf k'$ 是用 $\theta_i'$ 旋转后的查询与键；$t_{\mathrm Y}$ 是 YaRN 论文里的“温度”（论文写作 $t$，本课为避免与位置混淆加了下标）。等价写法：logits 乘以 $\mu^2$。
:::

**含义。** 所有注意力 logits 统一乘一个大于 1 的数 $\mu^2$（$s=8$ 时 1.46，DeepSeek-V3 的 $s=40$ 时 1.87）。

**理解。** logits 放大，softmax 的分布变尖，权重更集中在得分高的位置上；这相当于**降低**温度（等价温度 $1/\mu^2<1$），而不是升温<!-- 研究笔记 M2 -->。实现上不必改 softmax：把 $\mathbf q$ 和 $\mathbf k$ 各乘 $\mu$，它们的点积就乘了 $\mu^2$，而且可以直接乘进缓存的 $\cos$、$\sin$ 表里，零额外开销（论文称之为“length scaling”技巧）[@peng2023yarn]。

**数学来源：经验拟合。** $0.1\ln s+1$ 不是推导出来的，而是 YaRN 作者在 LLaMA 7B/13B/33B/65B 上、**不微调**的情况下，对不同扩展倍数 $s$ 找使困惑度最低的温度，再拟合出的规律；作者说它对 Llama 2 也拟合得不错 [@peng2023yarn]。为什么插值之后需要调尖，论文 v1 给过一个解释，但那个解释的方向和它实际做的操作相反，v2、v3 已经删掉了<!-- 研究笔记 E7、§8 第 16 条 -->。所以本课把 $\mu$ 当作**实测规律**，不当作定理。

**失败模式：调得太尖也不行。** 这个因子不是越大越好。YaRN 附录 A.3 固定 $s=8$，在 LLaMA 7B 上扫描它：图 4 的困惑度曲线有一个最低点，因子偏小或偏大困惑度都会升高（目测最低点约在 1.25–1.3，1.2 到 1.35 之间很平）；作者还报告，最合适的值在不同样本、不同位置上大体一致，而且换不同的 $s$ 时贴近 $0.1\ln s+1$ [@peng2023yarn]。$s=8$ 时公式给出 1.208，落在那段平坦区的边上。偏大为什么有害，论文没有解释；笔者的理解是：logits 乘得过大，softmax 会把权重压到得分最高的一两个位置上，和 [§6](#p6) 里异常大的打分造成的后果是同一类。

::: ex 几个 μ 值
- $s=8$：$\mu=0.1\ln8+1=0.1\times2.0794+1=1.20794$（YaRN 附录写作约 1.208）；logits 乘 $\mu^2=1.4591$。
- $s=40$（DeepSeek-V3）：$\mu=0.1\ln40+1=0.1\times3.6889+1=1.3689$，$\mu^2=1.8739$，等价温度 $1/\mu^2=0.5337$。
- $s=40$，DeepSeek-V2 的系数 0.0707：$\mu=0.0707\times3.6889+1=1.2608$，$\mu^2=1.5896$。V3 的 logits 乘数是 V2 的 $1.8739/1.5896=1.179$ 倍<!-- 研究笔记 R4 -->。

感受一下“调尖”：两个 logits $(2,0)$ 的 softmax 是 $(0.881,0.119)$；乘以 1.8739 变成 $(3.748,0)$，softmax 变成 $(0.977,0.023)$。
:::

::: fig figures/l05-temperature.svg | 图 5.14 注意力温度：μ 与 logits 乘数 μ² 随扩展倍数 s 的变化 | wide
**看 $s=40$ 那条竖线：V3 的 logits 乘 1.8739，V2 乘 1.5896。** 实线是 $\mathbf q$、$\mathbf k$ 各乘的因子 $\mu$，虚线是 logits 实际被乘的 $\mu^2$；蓝色是 YaRN 原版与 DeepSeek-V3 的 $0.1\ln s+1$，紫色是 DeepSeek-V2 的 $0.0707\ln s+1$。横轴以 2 为底取对数，所以 $\mu$ 是直线。$s=8$ 处的空心点是 YaRN 附录 A.3 的算例 $\mu\approx1.208$ [@peng2023yarn]。
:::

::: warn 同一个数，三种写法
YaRN v1 写 $\sqrt t\approx0.1\ln s+1$，把 $t$ 说成乘在 logits 上的因子；v2、v3 改写成 $\sqrt{1/t}=0.1\ln s+1$，$t$ 变成除在 logits 上的温度。两种写法数值上是同一个因子，都是“$\mathbf q$、$\mathbf k$ 各乘 $0.1\ln s+1$”。DeepSeek 的两篇论文沿用 v1 的写法 $\sqrt t=\dots$ [@peng2023yarn; @deepseek2024v2; @deepseek2024v3]。读到 $\sqrt t$ 或 $\sqrt{1/t}$ 时，只需记住：**乘在 logits 上的是 $(0.1\ln s+1)^2>1$。**
:::

## DeepSeek 怎么用：V2 与 V3

**V2。** DeepSeek-V2 §3.1.4 说，YaRN 专门作用在承载 RoPE 的解耦共享键 $\mathbf k_t^R$ 上（“specifically applied to the decoupled shared key”），取 $s=40$、$\alpha=1$、$\beta=32$，目标最大上下文 160K，期望在 128K 上表现良好；因为注意力机制不同（MLA），把温度系数改成 $\sqrt t=0.0707\ln s+1$；只用 32K 长的序列训练 1000 步，batch 576 条 [@deepseek2024v2]。“解耦键”是第 9 课的内容，[§8](#p8) 先预告；这里只需要知道：V2/V3 的每个头里，只有 64 维带 RoPE，另外 128 维不带位置。

**代码里的准确含义。** 三点要和论文措辞对照着看<!-- 研究笔记 §2.7、E12 -->[@ds-configs-code]：

1. 修改后的频率同时用于 $\mathbf q^R$ 和 $\mathbf k^R$。相对位置编码必须两边一致，“只作用于 $\mathbf k^R$”的意思是“只作用于 RoPE 那 64 维，不碰内容部分”。
2. 温度没有乘进 $\cos/\sin$（配置里 `mscale` 与 `mscale_all_dim` 相等，旋转本身的缩放因子是 1），而是乘进 softmax 的缩放系数，作用在**全部 192 维**（128 维内容 + 64 维 RoPE）的 logits 上：

::: eq DeepSeek 实现中的 softmax 缩放 | (5.23)
$$\begin{aligned}\texttt{softmax\_scale}&=\frac{\mu^2}{\sqrt{192}}=\mu^2\times0.0721688\\ \text{V2}&:\ 1.589626\times0.0721688=0.114721\\ \text{V3}&:\ 1.873854\times0.0721688=0.135234\end{aligned}$$
192 是每个头参与点积的总维度 $d_h+d_h^R=128+64$（`q_head_dim`）；$\mu$ 按 V2 / V3 各自的系数计算，这里 $\mu^2$ 多写两位，是为了让乘积的第六位有效数字也能用计算器复核<!-- 研究笔记 R5 -->。
:::

3. 是**静态**的 YaRN：$s$ 固定为 40，与实际输入长度无关，不是 YaRN 论文里按当前长度调整 $s$ 的 Dynamic 版本；配置里的最大位置 163,840 正好是 $4096\times40$<!-- 研究笔记 M8、R18 -->。

**V3。** DeepSeek-V3 §4.3 改成两个阶段、各 1000 步：先把上下文从 4K 扩到 32K（batch 1920），再扩到 128K（batch 480）；学习率都是 $7.3\times10^{-6}$，与预训练末段的学习率相同；YaRN 的 $s$、$\alpha$、$\beta$ 与 V2 相同，温度系数却是 $\sqrt t=0.1\ln s+1$ [@deepseek2024v3]。V3 论文说它的 YaRN 配置“与 V2 一致”，这句话只对 $s,\alpha,\beta$ 成立：温度系数从 0.0707 变成了 0.1，配置里的 `mscale_all_dim` 也从 0.707 变成了 1.0<!-- 研究笔记 E11 -->。

| | DeepSeek-V2 | DeepSeek-V3 |
|---|---|---|
| 作用范围 | 解耦的 64 维 RoPE | 同左 |
| $s$、$\alpha$、$\beta$ | 40、1、32 | 40、1、32 |
| 温度 $\mu$ | $0.0707\ln s+1=1.2608$ | $0.1\ln s+1=1.3689$ |
| logits 乘数 $\mu^2$ | 1.5896 | 1.8739 |
| 扩展训练 | 32K × 1000 步，batch 576 | 32K × 1000 步（batch 1920），再 128K × 1000 步（batch 480） |
| 目标 / 评测 | 目标 160K，大海捞针测到 128K | 128K |

::: ex 扩展阶段花了多少 token、多少钱
**V2。** $1000\times32768\times576=18{,}874{,}368{,}000\approx189$ 亿 token，只占 8.1T 预训练 token 的 0.233%。

**V3。** 阶段一 $1000\times32768\times1920=62{,}914{,}560{,}000$；阶段二 $1000\times131072\times480=62{,}914{,}560{,}000$。两个阶段每步的 token 数相同（6291 万），合计约 1258 亿，占 14.8T 的 0.85%<!-- 研究笔记 R19 -->。后来的 V3.1 沿用同样的两阶段方法并加了量：32K 阶段增至 6300 亿 token（“10-fold”），128K 阶段增至 2090 亿（“3.3x”）[@ds-v31-card]；用上面的数核对，$630/62.9=10.01$、$209/62.9=3.32$，对得上。

**成本。** V3 表 1 记上下文扩展用了 119K H800 GPU 小时，按每小时 2 美元是 23.8 万美元，占总训练成本 2788K GPU 小时的 4.27% [@deepseek2024v3]。若按预训练“每万亿 token 180K GPU 小时”的速率，1258 亿 token 只需约 2.26 万 GPU 小时；实际是它的约 5.25 倍。笔者的解释是长序列的注意力更贵（打分随长度平方增长），但这只是量级上的估算，没有扣除其他开销。
:::

## 失败模式：固定的 s 管不了两头

DeepSeek 用的是静态 YaRN：$s=40$ 在整个推理过程中不变。YaRN 论文 §3.4 专门讨论过这种做法的两个毛病：长度超过 $L'_{ctx}$ 时会突然变差（“an abrupt degradation”），长度小于 $L_{ctx}$ 时表现也可能打折扣（“a performance discount”）[@peng2023yarn]。用本课的数字可以把两头都看清楚。

::: ex 两头各算一个数
**超过 163,840：慢针再次越界。** 最慢的一对 $\theta_{31}=1.3335\times10^{-4}$，YaRN 把它除以 40。在 $t=163{,}840$ 处它转到 $163840\times\theta_{31}/40=4096\,\theta_{31}=0.5462$ rad $=31.3°$，恰好是预训练中见过的最大角度。再往后就越界了：$t=200{,}000$ 时是 $200000\times1.3335\times10^{-4}/40=0.6668$ rad $=38.2°$，又回到 [§6](#p6) 的外推处境。配置里的最大位置 163,840 就是这条边界。

**短输入也被改了。** 哪怕输入只有 1000 个 token，64 维 RoPE 里 $i=11,\dots,31$ 这 21 对（过渡的 12 对加全插值的 9 对）的频率已经被调慢，logits 也已经乘了 $\mu^2$（V3 为 1.8739）。模型在短文本上看到的角度与打分尺度，都和预训练时不一样。DeepSeek-V3 官方推理代码只要最大序列长度设得大于 4096（默认 16384）就启用 YaRN 的频率与温度，与实际输入多长无关 [@ds-configs-code]。这个代价在 DeepSeek 的模型上有多大，本课没有在它们的论文里找到单独的对照数据。
:::

YaRN 论文给的补救是**动态缩放**（Dynamic Scaling）：每次前向按当前序列长度 $l'$ 取 $s=\max(1,l'/L_{ctx})$，短输入时 $s=1$、什么都不改，超过训练长度时逐渐变差而不是突然崩溃 [@peng2023yarn]。代价是频率随长度变化，缓存旋转之后的键就不再正确（见练习 1）。DeepSeek 选的是静态版本；它的一个好处是旋转后的键可以直接进缓存（论文没有说明选择的理由）。

::: warn 常见误解
- “DeepSeek 的 128K 是在 160K 上训出来的。”不准确。V2 只在 32K 上训了 1000 步，按 $s=40$ 把目标设在 160K、期望 128K 可用；V3 的第二阶段才真正在 128K 长的序列上训练。
- “V2 的大海捞针全绿。”V2 论文图 4 标题写的是 Base 模型，目测热力图在约 35K、104K 处有少数不满分的格子；V3 论文图 8（SFT 之后）才全部满分。论文正文“在 128K 以内所有长度上表现良好”是概括<!-- 研究笔记 M10 -->。
:::

::: key 本页要点
- YaRN = 按转圈数 $r_i=L_{ctx}/\lambda_i$ 分段插值（$r>\beta$ 不动、$r<\alpha$ 除以 $s$、中间线性混合，(5.20)）+ 注意力温度 $\mu=0.1\ln s+1$（logits 乘 $\mu^2$，(5.22)）。它不改底数。
- DeepSeek 设置（64 维，4096，$s=40$，$\alpha=1$，$\beta=32$）：11 对不动、12 对过渡、9 对除以 40；全插值的正是那 9 根慢针。实际实现用参考代码的“对下标线性”斜坡，过渡区与论文式不同。
- 温度是经验规律：V2 系数 0.0707（$\mu^2=1.5896$），V3 系数 0.1（$\mu^2=1.8739$）；DeepSeek 把它放进 softmax 缩放，作用于全部 192 维。
- 成本极低：V3 扩展用了预训练 token 的 0.85%、总 GPU 小时的 4.27%。
- 失败模式：$s$ 固定为 40，超过 163,840 慢针重新越界；短输入也照样被改频率、乘温度。YaRN 的动态缩放能缓解两头，但缓存里就只能存旋转之前的键。
:::

# 算总账：DeepSeek 的块，以及 RoPE 留给第 9 课的难题 | 把积木放回 DeepSeek，再看清 RoPE 与缓存压缩的冲突 | 20

这一页做两件事。先把本课的五样东西（四块积木加一种扩展方法）放回 DeepSeek 的模型里，每一样配上它解决的问题和真实的数字。然后把 RoPE 留下的一个难题摆出来：它和“只缓存一个压缩向量”的想法不兼容。这个难题是[第 9 课](lesson-09-v2-mla.html#p4) MLA 的起点。

## 五样东西的账单

| 积木 | 解决什么 | 代价与数字 | DeepSeek 的用法 |
|---|---|---|---|
| RoPE（§2–§3） | 打分在结构上只依赖相对位置；旋转后的键可直接进 KV cache | 每层对 $\mathbf q,\mathbf k$ 做逐元素乘加；慢针决定了不能直接外推（64 维里 9 对、128 维里 18 对在 4K 内转不满一圈） | DeepSeek LLM：每头 128 维全转；V2/V3：只有解耦的 64 维转 |
| RMSNorm（§4） | 控制子层输入的尺度，少一个统计量、少一组偏置 | 失去平移不变性；参数可忽略（LLaMA-7B 共 26.6 万）；2019 年实测提速 7%–64% | $\epsilon=10^{-6}$，Pre-Norm |
| SwiGLU（§5） | 同参数量下更低的困惑度（T5：1.636 对 1.677） | 三个矩阵；$d_{ff}=\tfrac83d$ 才与 $4d$ 的 GELU 等参数 | DeepSeek LLM 11008 / 22016（$2.6875d$）；V2、V3 稠密层 $2.4d$、$2.571d$；MoE 专家更窄 |
| 无偏置（§5） | PaLM 报告的训练稳定性 | 几乎不省参数（LLaMA-7B 若有偏置也只占 0.024%） | 官方推理代码的线性层默认无偏置 |
| YaRN（§6–§7） | 4K 预训练、128K 使用 | 温度是经验规律；论文与代码的斜坡不同 | $s=40,\alpha=1,\beta=32$；V2 温度系数 0.0707，V3 为 0.1；V3 扩展只用了 0.85% 的 token、4.27% 的 GPU 小时 |

前四样在 DeepSeek LLM（2023 年 11 月发布权重，2024 年 1 月发表报告）里就已全部就位：论文写明了前三样，“无偏置”则可以从它发布的权重里看出来（7B 与 67B 的权重索引里没有一个偏置张量）[@deepseek2024llm; @ds-llm-hf]；第五样在 DeepSeek-V2 的技术报告里用上 [@deepseek2024v2]。表里最后一列的“只有解耦的 64 维转”，就是下面要讲的难题的解法。

## RoPE 与“缓存压缩向量”的冲突

**问题。** [第 4 课](lesson-04-inference-kv-cache.html#p3)算过，多头注意力的 KV cache 每个 token 每层要存 $2n_hd_h$ 个数。第 9 课的 MLA 想换一种存法：每个 token 只缓存一个短得多的向量 $\mathbf c_j\in\R^{d_c}$（DeepSeek-V2 取 $d_c=512$；论文和第 9 课写作 $\mathbf c_j^{KV}$，这一页只预告，省去上标），要用键时再用一个矩阵把它展开：$\mathbf k_j=W^{UK}\mathbf c_j$，只看一个头时 $W^{UK}\in\R^{d_h\times d_c}$ [@deepseek2024v2]。展开听起来要花计算，但有一个办法可以不展开：M01 最后一页讲过的**矩阵吸收**，靠的是矩阵乘法的结合律（[M01 §6](../math/m01-vectors-matrices.html#p6)；形状与 FLOPs 的完整账见 [M02 §7](../math/m02-rank-svd-lowrank.html#p7)）。

::: deriv 没有 RoPE 时的吸收（一个头）
$\mathbf q_t^\top\mathbf k_j=\mathbf q_t^\top\big(W^{UK}\mathbf c_j\big)$ || 键由缓存的 $\mathbf c_j$ 展开
$=\big((W^{UK})^\top\mathbf q_t\big)^\top\mathbf c_j=\tilde{\mathbf q}_t^\top\mathbf c_j$ || 结合律加转置：先把查询变换一次，得到 $\tilde{\mathbf q}_t\in\R^{d_c}$
:::

$\tilde{\mathbf q}_t$ 对缓存里的**每一个** $j$ 都通用。解码一步只需算一次 $\tilde{\mathbf q}_t$，再和所有缓存的 $\mathbf c_j$ 做点积；$(W^{UK})^\top$ 甚至可以预先并进 $W^Q$。

**加上 RoPE 之后。** 键要按自己的位置旋转，$\mathbf k_j=\mathbf R_{\Theta,j}W^{UK}\mathbf c_j$；查询也要旋转，$\mathbf q_t=\mathbf R_{\Theta,t}\mathbf q'_t$（$\mathbf q'_t$ 是旋转前的查询）：

::: eq RoPE 夹在中间 | (5.24)
$$\mathbf q_t^\top\mathbf k_j=\mathbf q_t'^{\top}\,\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,j}\,W^{UK}\mathbf c_j=\mathbf q_t'^{\top}\,\underbrace{\mathbf R_{\Theta,j-t}\,W^{UK}}_{\text{随 }j-t\text{ 变化}}\,\mathbf c_j$$
第二个等号就是 §3 的相对位置性质。夹在 $\mathbf q'_t$ 与 $\mathbf c_j$ 之间的矩阵对每个缓存位置 $j$ 都不同。
:::

没有一个固定的矩阵可以提前并进 $W^Q$：对缓存里的每个 $j$，“吸收后的查询” $\big(\mathbf R_{\Theta,j-t}W^{UK}\big)^\top\mathbf q'_t$ 都得重新算一遍，这和把所有键重新展开再旋转没有区别。DeepSeek-V2 的原话是：RoPE 与低秩 KV 压缩不兼容（“RoPE is incompatible with low-rank KV compression.”），因为一个与当前生成 token 位置有关的 RoPE 矩阵会夹在 $W^Q$ 与 $W^{UK}$ 之间，而矩阵乘法不满足交换律 [@deepseek2024v2]。

::: ex 一个最小的反例（笔者构造）
取二维的头，缓存向量是一个数 $c_j$，$W^{UK}=(1,0)^\top$（把 $c_j$ 展开成键 $(c_j,0)$），旋转前的查询 $\mathbf q'=(0,1)$，只有一对频率 $\theta=\pi/2$。

**没有 RoPE。** $\tilde q=(W^{UK})^\top\mathbf q'=(1,0)\cdot(0,1)=0$，一个数通吃所有位置：打分都是 $0\cdot c_j$。

**有 RoPE，当前位置 $t=3$。** 对每个 $j$，吸收后的“查询”是 $\mathbf q'^\top\mathbf R\big((j-3)\tfrac\pi2\big)(1,0)^\top=\sin\big((j-3)\tfrac\pi2\big)$：$j=0$ 时 $+1$，$j=1$ 时 $0$，$j=2$ 时 $-1$，$j=3$ 时 $0$。系数随 $j$ 变化（$j=0$ 与 $j=2$ 甚至符号相反），不存在一个能对所有 $j$ 通用的 $\tilde q$。
:::

**解耦 RoPE：把位置信息另放一小栏。** DeepSeek-V2 的解法是把每个头的查询和键拆成两部分 [@deepseek2024v2]：

- **内容部分**（每头 $d_h=128$ 维）：由压缩向量展开，**不加 RoPE**，所以可以照常吸收；
- **RoPE 部分**（每头 $d_h^R=64$ 维）：查询 $\mathbf q^R_{t,i}$ 每个头各有一份；键 $\mathbf k^R_t=\operatorname{RoPE}(W^{KR}\mathbf h_t)$（$W^{KR}\in\R^{d_h^R\times d}$）所有头**共享**一份，单独缓存。

::: eq 解耦 RoPE 的打分与缓存 | (5.25)
$$\mathbf q_{t,i}^\top\mathbf k_{j,i}=\underbrace{\mathbf q^{C\top}_{t,i}\,\mathbf k^C_{j,i}}_{\text{不带位置，可吸收}}+\underbrace{\mathbf q^{R\top}_{t,i}\,\mathbf k^R_j}_{\text{带 RoPE，直接算}}$$
$\mathbf q_{t,i}=[\mathbf q^C_{t,i};\mathbf q^R_{t,i}]$，$\mathbf k_{j,i}=[\mathbf k^C_{j,i};\mathbf k^R_j]$，拼接后每头 192 维；softmax 按 $\sqrt{d_h+d_h^R}=\sqrt{192}$ 缩放。每 token 每层只缓存 $\mathbf c_j$ 与 $\mathbf k^R_j$：$d_c+d_h^R=512+64=576$ 个数。这里只给结论，逐行推导在[第 9 课 §5](lesson-09-v2-mla.html#p5)。
:::

::: fig figures/l05-absorb-preview.svg | 图 5.15 RoPE 为什么挡住矩阵吸收，以及解耦 RoPE | wide
**看第二行的红色小方块：每个缓存位置都要一个不同的旋转，所以没有一个固定矩阵能并进 $W^Q$。** 第一行没有 RoPE，一个 $\tilde{\mathbf q}_t$ 通吃所有缓存位置。第三行是 DeepSeek-V2/V3 的做法：内容部分不转、照常吸收，缓存 512 维的 $\mathbf c_j$；位置信息放进另外 64 维、所有头共享的 $\mathbf k^R_j$，一起缓存，共 576 个数。维度取自 DeepSeek-V2 的论文与配置 [@deepseek2024v2; @ds-configs-code]。
:::

::: ex RoPE 那一小栏占多少缓存
DeepSeek-V2：60 层，每 token $576\times60=34{,}560$ 个数，用 BF16（每个数 2 字节）是 69,120 字节；DeepSeek-V3：61 层，$576\times61\times2=70{,}272$ 字节。其中 RoPE 部分占 $64/576=11.1\%$。作为对照，同样 60 层、128 头、每头 128 维的标准多头注意力，每 token 要存 $2\times128\times128\times60\times2=3{,}932{,}160$ 字节，是 V2 的约 57 倍（[第 9 课](lesson-09-v2-mla.html#p8)会把这笔账算全）。
:::

这也回答了 [§7](#p7) 留下的一个问题：YaRN 为什么“只作用在 $\mathbf k^R$ 上”。在 V2/V3 里，只有这 64 维（以及对应的 $\mathbf q^R$）带位置信息；内容部分根本没有旋转，也就没有频率可以插值。温度因子则通过 softmax 的缩放系数作用在全部 192 维上 (5.23)。

::: note 往后看
后来的模型仍在沿用这一套。按 Hugging Face 上的配置文件，DeepSeek-V3.1、V3.2 的 `rope_scaling` 与 V3 相同；V4-Flash、V4-Pro 与 V4.1-Flash 改成了 YaRN 倍数 16、原始长度 65,536（$65536\times16=1{,}048{,}576$，即百万上下文），V4 的 `qk_rope_head_dim` 仍是 64 [@ds-later-configs]。V4 的配置里还多了一个 `compress_rope_theta` = 160,000。[第 18 课 §4](lesson-18-v4.html#p4) 对照开源代码核实了它们的用法（论文没写）：带压缩的层（CSA、HCA）用底数 160,000 并加这套 YaRN；纯滑动窗口的层仍用底数 10,000，不加 YaRN。
:::

::: key 本课要点
- 现代块 = RoPE（每层旋转 $\mathbf q,\mathbf k$，打分只依赖相对位置）+ Pre-RMSNorm（只缩放、不居中）+ SwiGLU（门控，$d_{ff}\approx\tfrac83d$）+ 无偏置。DeepSeek LLM 原样采用，V2/V3 在此基础上改注意力与前馈层。
- RoPE 的频率是几何级数，快针分辨近邻、慢针在训练长度内转不满一圈；直接外推会让慢针越界。PI 全部拨慢、NTK-aware 改底数、YaRN 按转圈数分段并加温度。DeepSeek：64 维里 11 对不动、12 对过渡、9 对除以 40，$\mu^2$ 为 1.5896（V2）/ 1.8739（V3）。
- RoPE 让“缓存压缩向量 + 矩阵吸收”失效，因为 $\mathbf R_{\Theta,j-t}$ 夹在中间 (5.24)；DeepSeek 用解耦 RoPE 把位置放进单独的 64 维，每 token 每层缓存 576 个数。这是第 9 课的起点。
:::

# 练习与答案 | 12 题：概念、计算、推导 | 50

先自己做，再展开答案。计算题都能用纸笔加计算器完成；想核对的话，本课每个生成脚本（`deepseek/figures/gen/l05_*.py`）里都有可以直接改参数重算的代码。

## 概念题

:::: quiz 练习 1 · 缓存旋转之后的键，还是之前的键？
（a）DeepSeek-V3 的推理代码把**旋转之后**的 $\mathbf k^R_j$ 写进 KV cache [@ds-configs-code]。为什么这是安全的？（b）YaRN 论文的 Dynamic Scaling 每次前向按当前长度 $l'$ 取 $s=\max(1,l'/L_{ctx})$，论文特别提醒这时要缓存**旋转之前**的键。为什么？（c）DeepSeek 用的是哪一种？
::: answer 看答案
（a）旋转矩阵 $\mathbf R_{\Theta,j}$ 只依赖键自己的位置 $j$ 和固定的频率。因果注意力里 $\mathbf h_j$ 算出来就不再变（第 4 课），频率也不变，所以 $\mathbf R_{\Theta,j}W^{KR}\mathbf h_j$ 写进缓存后永远有效；解码第 $t$ 步只需要旋转新的 $\mathbf q_t$ 与 $\mathbf k_t$。

（b）Dynamic Scaling 的 $s$ 随序列变长而变，频率 $\theta_i'$ 也跟着变。按旧频率旋转好的键，到了新的 $s$ 下就是错的；缓存旋转之前的键，每一步再按当前频率旋转，才能保证查询与键用的是同一组频率 [@peng2023yarn]。

（c）静态的：$s$ 固定为 40，频率在整个推理过程中不变，所以可以缓存旋转后的键（§7）。
:::
::::

:::: quiz 练习 2 · 评一句常见的话
有人说：“RoPE 有远程衰减性，所以离得越远的 token 注意力权重越小。”指出这句话里至少两处不准确的地方，并给出正确的说法。
::: answer 看答案
① RoFormer 证明的是打分绝对值的一个**上界**，衰减是用图 2 数值展示的，而且只画到 256；这个上界里的平均量 $U(\Delta)$ 在约 1000 之后就不再系统性下降，逐点剧烈起伏（$[1000,4096]$ 内在 2.23 到 11.39 之间，图 5.6）。PI 论文还批评这个界可能太松、没有信息量。② 即使上界下降，也不能推出打分本身下降，更推不出**权重**（softmax 之后）下降：权重还取决于同一行其他位置的打分。③ 能推出来的是 (5.11)：$\mathbf q,\mathbf k$ 同向（更一般地，$\E[\mathbf k\mathbf q^\top]=\kappa I$、$\kappa>0$）时平均打分随距离呈下降趋势（且不单调），独立时期望打分在任何距离都是 0。

正确说法：RoPE 让各对维度上同样正相关的 $\mathbf q,\mathbf k$ 的**平均**打分随距离呈下降趋势，但不保证任何具体的注意力权重随距离递减。
:::
::::

:::: quiz 练习 3 · 三种扩展方法各错在哪里
用“多频率时钟”的语言回答：（a）PI 把所有指针拨慢 $s$ 倍，哪一类指针因此吃亏？（b）NTK-aware 只改底数，哪一类指针仍然越界？（c）YaRN 为什么对快针“不碰”、对慢针“全插值”？
::: answer 看答案
（a）快针。它们靠“相邻 token 角度差很大”来分辨近邻；拨慢 $s$ 倍后，相邻角度差从 $57.3°$ 变成 $57.3°/s$（DeepSeek 的 $s=40$ 时只剩 $1.43°$），近邻变得难以区分，需要微调重新学。

（b）中间的慢针。NTK-aware 只保证**最慢的一根**恰好慢 $s$ 倍；其余 $\lambda_i>L_{ctx}$ 的针缩放不足，扩展后转到训练中没见过的角度。DeepSeek 设置下 $i=23$ 越界 2.59 倍。

（c）快针在训练中转过很多圈，它们的角度早已覆盖整个圆周，只携带相对的短距离信息，扩展后不会遇到新角度，拨慢反而破坏近邻分辨；慢针在训练中没转满一圈，它们的角度被模型当作绝对位置，扩展后必然越界，所以要完全插值，让新范围里的角度仍落在训练见过的那段弧内。YaRN 论文的原话是不要去碰那些只编码相对位置信息的维度。
:::
::::

:::: quiz 练习 4 · 温度的方向与记号
（a）YaRN 的注意力“温度”让 softmax 变尖还是变平？是升温还是降温？（b）DeepSeek-V2 论文写 $\sqrt t=0.0707\ln s+1$，YaRN v3 写 $\sqrt{1/t}=0.1\ln s+1$。这两个 $t$ 是同一个意思吗？（c）“YaRN 就是改了底数的 RoPE”对吗？
::: answer 看答案
（a）logits 乘 $\mu^2>1$，softmax 变尖；等价温度 $1/\mu^2<1$，是**降温**。

（b）不是。DeepSeek 沿用 YaRN v1 的写法，$t$ 是**乘**在 logits 上的因子（DeepSeek 论文没有另外定义 $t$；按它的代码，这个 $\sqrt t$ 就是 $\mathbf q$、$\mathbf k$ 各乘的因子）；YaRN v2/v3 的 $t$ 是**除**在 logits 上的温度。两者互为倒数，但式子右边算出的都是“$\mathbf q$、$\mathbf k$ 各乘的因子 $\mu$”，所以数值上是同一件事。V2 与 V3 的差别在系数（0.0707 与 0.1），不在记号。

（c）不对。改底数的是 NTK-aware；YaRN 是“按转圈数分段插值（NTK-by-parts）+ 注意力温度”，底数不变。
:::
::::

## 计算题

:::: quiz 练习 5 · 手算一次二维 RoPE
$\theta=0.5$ rad/位置，$\mathbf q=(1,0)$，$\mathbf k=(0.6,0.8)$。（a）用复数公式 (5.3) 求 $(t,j)=(4,1)$ 的打分。（b）用矩阵分别旋转后再点积，求 $(t,j)=(10,7)$ 的打分，验证与（a）相同。（c）$(t,j)=(1,4)$（键在查询之后，非因果）的打分是多少？为什么和（a）不同？
::: answer 看答案
（a）$\mathbf q\bar{\mathbf k}=1\cdot(0.6-0.8\mathrm i)=0.6-0.8\mathrm i$；$e^{\mathrm i\cdot3\cdot0.5}=\cos1.5+\mathrm i\sin1.5=0.070737+0.997495\mathrm i$。实部：$0.6\times0.070737-(-0.8)\times0.997495=0.04244+0.79800=0.8404$。

（b）$\mathbf R(5)\mathbf q=(\cos5,\sin5)=(0.2837,-0.9589)$；$\mathbf R(3.5)\mathbf k=(0.6\cos3.5-0.8\sin3.5,\ 0.6\sin3.5+0.8\cos3.5)=(-0.2812,-0.9596)$。点积 $0.2837\times(-0.2812)+(-0.9589)\times(-0.9596)=-0.0798+0.9202=0.8404$，相同。

（c）距离是 $-3$：实部 $=0.6\cos(-1.5)-(-0.8)\sin(-1.5)=0.04244-0.79800=-0.7556$。打分只依赖**有符号**的距离 $t-j$，$+3$ 与 $-3$ 一般不同：旋转有方向，“在前面 3 个”和“在后面 3 个”是两回事。
:::
::::

:::: quiz 练习 6 · RMSNorm 与 LayerNorm
$\mathbf x=(3,-1,1,1)$，$\mathbf g=\mathbf 1$，$\mathbf b=\mathbf 0$，$\epsilon=0$。（a）求 $\LayerNorm(\mathbf x)$ 与 $\RMSNorm(\mathbf x)$。（b）对 $\mathbf x+2$ 再求一遍。（c）对 $5\mathbf x$ 再求一遍。哪些结果不变？
::: answer 看答案
（a）LayerNorm：均值 1，居中 $(2,-2,0,0)$，方差 $\frac14(4+4+0+0)=2$，标准差 1.4142；输出 $(1.4142,-1.4142,0,0)$。RMSNorm：均方 $\frac14(9+1+1+1)=3$，RMS $=1.7321$；输出 $(1.7321,-0.5774,0.5774,0.5774)$。

（b）$\mathbf x+2=(5,1,3,3)$。LayerNorm：均值 3，居中后仍是 $(2,-2,0,0)$，输出不变。RMSNorm：均方 $\frac14(25+1+9+9)=11$，RMS $=3.3166$；输出 $(1.5076,0.3015,0.9045,0.9045)$，变了，而且第二个分量从负变正。

（c）$5\mathbf x=(15,-5,5,5)$：两者输出都与（a）相同（缩放不变）。所以 LayerNorm 对平移、缩放都不变；RMSNorm 只对（正的）缩放不变。
:::
::::

:::: quiz 练习 7 · 前馈层的宽度
（a）DeepSeek LLM 67B 的 $d=8192$。按 LLaMA 的规则（$\tfrac83d$ 取整后向上取到 256 的倍数），$d_{ff}$ 是多少？SwiGLU 的参数比 $8d^2$ 多百分之几？（b）DeepSeek-V3 稠密层 $d=7168$、$d_{ff}=18432$。按同一规则应该是多少？实际参数比 $8d^2$ 多还是少、差多少？
::: answer 看答案
（a）$\tfrac83\times8192=21845.33$，取整 21845，$21845/256=85.33$，向上取 86，$86\times256=22016$，与 DeepSeek LLM 的配置一致。参数 $3\times8192\times22016=541{,}065{,}216$，$8d^2=536{,}870{,}912$，多 0.78%。

（b）$\tfrac83\times7168=19114.67$，按规则会是 $75\times256=19200$。实际 18432，参数 $3\times7168\times18432=396{,}361{,}728$，而 $8d^2=411{,}041{,}792$，**少** 3.57%；宽度比 2.571 小于 $8/3$。DeepSeek-V2 的稠密层（$d=5120$，$d_{ff}=12288$）更少，比 $8d^2$ 少 10%。V2/V3 的稠密层不遵守 8/3 规则。
:::
::::

:::: quiz 练习 8 · 如果头是 128 维：YaRN 怎么分段
设想把 DeepSeek 的 YaRN 设置（$L_{ctx}=4096$，$s=40$，$\alpha=1$，$\beta=32$，$b=10000$）用在一个 RoPE 占满 128 维的头上。（a）用 (5.21) 求两个分界下标，并给出三段各有几对。（b）第 30 对的波长、转圈数，以及按论文式与按参考代码的缩放系数。（c）把（b）的数和 64 维时第 15 对的数比较，你发现了什么？
::: answer 看答案
（a）$i(32)=\frac{128\ln(4096/(64\pi))}{2\ln10000}=20.94$，$i(1)=\frac{128\ln(4096/(2\pi))}{2\ln10000}=45.03$。保留 $i=0..20$（21 对），全插值 $i=46..63$（18 对），过渡 $i=21..45$（25 对）。代码取 `low = 20`、`high = 46`，分段相同。

（b）$\theta_{30}=10000^{-60/128}=0.013335$，$\lambda_{30}=471.17$，$r_{30}=4096/471.17=8.6932$。论文：$\gamma=(8.6932-1)/31=0.248168$，缩放 $=(1-0.248168)/40+0.248168=0.26696$。代码：权重 $1-(30-20)/(46-20)=0.6154$，缩放 $=(1-0.6154)/40+0.6154=0.625$。

（c）与 64 维的第 15 对完全相同（$\lambda=471.17$、论文 0.26696、代码 0.625）。因为 $2\cdot30/128=2\cdot15/64$，两者是同一个频率；代码的分界也恰好加倍（10→20，23→46），所以连代码的斜坡也对得上。
:::
::::

:::: quiz 练习 9 · 温度、扩展成本与越界
（a）分别按 V2 与 V3 的系数，求 $s=40$ 时的 $\mu$、$\mu^2$ 和 DeepSeek 实现里的 `softmax_scale`（每头 192 维）。（b）两个 logits $(2,0)$ 在乘 $\mu^2$ 前后，softmax 的第一个概率各是多少？（c）DeepSeek-V3 两个扩展阶段各用了多少 token？合计占 14.8T 预训练 token 的百分之几？（d）把静态 YaRN（$s=40$）用到位置 $t=327{,}680$（配置上限 163,840 的两倍），64 维 RoPE 里最慢的一对（$\theta_{31}=1.3335\times10^{-4}$）转到多少度？是预训练中见过的最大角度的几倍？这对应 §7 的哪一种失败模式？
::: answer 看答案
（a）$\ln40=3.6889$。V2：$\mu=0.0707\times3.6889+1=1.2608$，$\mu^2=1.5896$，`softmax_scale` $=\mu^2/\sqrt{192}=1.5896\times0.072169=0.11472$。V3：$\mu=0.1\ln40+1=1.368888$，$\mu^2=1.87385$，`softmax_scale` $=1.87385\times0.072169=0.13523$（$\mu^2$ 只取到 1.8739 的话会算出 0.13524，差在舍入）。

（b）乘之前：$e^2/(e^2+1)=0.8808$。V2 乘 1.5896 后 logits $(3.179,0)$，概率 0.9600；V3 乘 1.8739 后 $(3.748,0)$，概率 0.9770。都变尖了，V3 更尖。

（c）阶段一 $1000\times32768\times1920=62{,}914{,}560{,}000$；阶段二 $1000\times131072\times480=62{,}914{,}560{,}000$；合计 $1.258\times10^{11}$，占 $1.258\times10^{11}/1.48\times10^{13}=0.85\%$。

（d）YaRN 把这一对的频率除以 40：角度 $=327680\times1.3335\times10^{-4}/40=8192\,\theta_{31}=1.0924$ rad $=62.6°$。预训练中见过的最大角度是 $4096\,\theta_{31}=0.5462$ rad $=31.3°$，现在是它的 $327680/(40\times4096)=2$ 倍。这是“超过 $L'_{ctx}$ 之后慢针再次越界”：对全插值的慢针，除以 40 等于把位置除以 40，只有 163,840 以内的位置才会被压回预训练见过的 $[0,4096]$；再往后就回到直接外推的处境，YaRN 论文 §3.4 说这时表现会突然变差。
:::
::::

## 推导题

:::: quiz 练习 10 · 推出 NTK-aware 的底数
（a）从“最慢的一对恰好慢 $s$ 倍”出发，推出 $b'=b\,s^{d_h/(d_h-2)}$，并证明每一对的缩放是 $s^{-2i/(d_h-2)}$。（b）对 $d_h=128$、$s=8$ 求 $b'$。（c）对 $d_h^R=64$、$s=40$，证明第 23 对在扩展到 $163{,}840$ 时仍然越界，求越界倍数。
::: answer 看答案
（a）要求 $(b')^{-(d_h-2)/d_h}=\frac1sb^{-(d_h-2)/d_h}$。取倒数：$(b')^{(d_h-2)/d_h}=s\,b^{(d_h-2)/d_h}$；两边取 $\frac{d_h}{d_h-2}$ 次方：$b'=b\,s^{d_h/(d_h-2)}$。于是 $\theta'_i=(b')^{-2i/d_h}=b^{-2i/d_h}\,s^{-\frac{d_h}{d_h-2}\cdot\frac{2i}{d_h}}=\theta_i\,s^{-2i/(d_h-2)}$。$i=0$ 时为 1，$i=d_h/2-1$ 时为 $1/s$，中间是几何级数。

（b）$b'=10000\times8^{128/126}=82{,}684.6$。

（c）$i=23$：缩放 $40^{-46/62}=0.0648$。扩展后最大位置处的角度是 $163840\,\theta_{23}\times0.0648$；预训练中的最大角度是 $4096\,\theta_{23}$。两者之比 $=163840\times0.0648/4096=2.59$。因为 $\lambda_{23}=4711.7>4096$，这根针在预训练中没转满一圈，角度范围只到 $4096\theta_{23}$；现在要转到它的 2.59 倍，越界。
:::
::::

:::: quiz 练习 11 · 如果位置也改变长度
在 §2 的推导里，选择③要求“位置不改变长度”。现在放宽为 $R_q(\mathbf q,m)=\norm{\mathbf q}a(m)$、$R_k(\mathbf k,m)=\norm{\mathbf k}/a(m)$，其中 $a(m)>0$，$a(0)=1$，位置取整数。（a）验证第 (6) 行仍然成立。（b）证明：若要求 $t\ne j$ 时模的乘积也只依赖 $t-j$，则 $a(m)=\rho^m$（$\rho>0$ 为常数）。（c）写出这时的打分，并说明 $\rho>1$ 与 $\rho<1$ 各意味着什么。
::: answer 看答案
（a）$R_q(\mathbf q,m)R_k(\mathbf k,m)=\norm{\mathbf q}a(m)\cdot\norm{\mathbf k}/a(m)=\norm{\mathbf q}\norm{\mathbf k}$，与第 (6) 行一致。

（b）$t\ne j$ 时模的乘积是 $\norm{\mathbf q}\norm{\mathbf k}\,a(t)/a(j)$。要它只依赖 $t-j$，就要求 $a(t)/a(j)=G(t-j)$ 对某个函数 $G$ 成立。取 $j=t-1$：$a(t)/a(t-1)=G(1)=:\rho$，与 $t$ 无关。于是 $a(m)=a(0)\rho^m=\rho^m$（对负的 $m$ 同理向下递推）。

（c）$f_q(\mathbf q,t)=\mathbf q\,\rho^te^{\mathrm it\theta}$，$f_k(\mathbf k,j)=\mathbf k\,\rho^{-j}e^{\mathrm ij\theta}$，打分 $=\rho^{\,t-j}\operatorname{Re}\big[\mathbf q\bar{\mathbf k}e^{\mathrm i(t-j)\theta}\big]$。因果注意力里 $t-j\ge0$：$\rho<1$ 时打分随距离指数衰减，$\rho>1$ 时指数增长。RoPE 取 $\rho=1$，位置只转角度、不改长度，没有内置任何随距离的衰减（这是笔者补充的推导，不在 RoFormer 里）。
:::
::::

:::: quiz 练习 12 · 解耦 RoPE 为什么能吸收
（a）写出不带 RoPE 时 $\mathbf q_t^\top W^{UK}\mathbf c_j$ 的吸收形式，并说明吸收后的查询是几维的。（b）说明为什么在 (5.25) 里，第一项可以吸收，而第二项不需要吸收。（c）DeepSeek-V2 的缓存为每 token 每层 576 个数，其中多少个是为 RoPE 付出的？如果改成“每个头各有一份 64 维的 $\mathbf k^R$”（128 个头），这一项会变成多少？
::: answer 看答案
（a）$\mathbf q_t^\top W^{UK}\mathbf c_j=\big((W^{UK})^\top\mathbf q_t\big)^\top\mathbf c_j$，吸收后的查询 $(W^{UK})^\top\mathbf q_t\in\R^{d_c}$，是 512 维（每个头一个）。

（b）第一项 $\mathbf q^{C\top}_{t,i}\mathbf k^C_{j,i}=\mathbf q^{C\top}_{t,i}W^{UK}_i\mathbf c_j$，中间没有随位置变化的矩阵，结合律照常可用，所以只需缓存 $\mathbf c_j$。第二项的键 $\mathbf k^R_j$ 本身就被缓存（已按位置 $j$ 旋转），查询 $\mathbf q^R_{t,i}$ 按当前位置 $t$ 旋转后直接与它点积；它只有 64 维、所有头共享，不需要从压缩向量展开，所以也就不需要吸收。

（c）64 个，占 $64/576=11.1\%$。若每个头各存一份，这一项变成 $64\times128=8192$ 个，缓存变成 $512+8192=8704$，是现在的 15 倍多；“所有头共享一个 $\mathbf k^R$”正是为了避免这一点。
:::
::::

```refs
- id: deepseek2024v2
  title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  a: DeepSeek-AI (2024). arXiv:2405.04434（读的是 v5；§3.1.4 在 v1–v5 中逐字相同）
  url: https://arxiv.org/abs/2405.04434
  finding: §2.1.3 指出 RoPE 与低秩 KV 压缩不兼容，提出解耦 RoPE（每头 $d_h^R=64$ 维的 $\mathbf q^R$ 与所有头共享的 $\mathbf k^R$）；§3.1.4 把 YaRN 专门用在解耦的共享键上（$s=40$、$\alpha=1$、$\beta=32$，目标 160K），温度改为 $\sqrt t=0.0707\ln s+1$，只以 32K 训练 1000 步。
  matters: 本课 §7 的 DeepSeek 设置与 §8 的“RoPE 挡住吸收”都以它为原始出处，也为第 9 课的 MLA 铺路。
- id: bloc97-2023ntk
  kind: artefact
  title: "NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation"
  a: u/bloc97（Reddit r/LocalLLaMA 帖子，2023-06-29；reddit.com 直连被拒，经 Wayback 快照读取）
  url: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
  finding: 提出改 RoPE 的底数而不是缩放位置；正文称 LLaMA 7B 用到 4096 的上下文不需要微调、困惑度退化很小（标题称可到 8k 以上），并用 alpha=8 与 SuperHOT 的 scale=4 对比；帖子正文没有写出底数公式。
  matters: YaRN 的直接前身；本课用它说明“不同频率应区别对待”这一思想的来源，公式则取自 YaRN 的形式化与代码。
- id: yarn-code
  kind: artefact
  title: jquesnelle/yarn（YaRN 参考实现）
  a: Quesnelle, J., Peng, B., et al.（GitHub，master 分支）. 读取于 2026-09-24
  url: https://github.com/jquesnelle/yarn
  finding: 用 find_correction_dim 把 $\beta$、$\alpha$ 换算成频率下标，再用对下标线性、端点 floor/ceil 取整的斜坡混合 $\theta$ 与 $\theta/s$；NTK-aware 版本为 base * alpha ** (dim / (dim-2))；温度 mscale = 0.1 ln s + 1 乘进 cos/sin。
  matters: DeepSeek 代码的直接来源，也是“论文的斜坡与代码的斜坡不同”这一勘误的证据。
- id: ds-v31-card
  kind: artefact
  title: DeepSeek-V3.1-Base 模型卡（Hugging Face）
  a: DeepSeek-AI (2025). 读取于 2026-09-24
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base
  finding: V3.1 沿用 V3 的两阶段长上下文扩展，32K 阶段增至 630B token（10 倍），128K 阶段增至 209B token（3.3 倍）。
  matters: 反向印证本课算出的 V3 两阶段各约 629 亿 token，也说明这套扩展方法在后续模型中延续。
- id: ds-later-configs
  kind: artefact
  title: DeepSeek-V3.1 / V3.2 / V4-Flash / V4-Pro / V4.1-Flash 的 config.json（Hugging Face）
  a: DeepSeek-AI (2025–2026). 读取于 2026-09-24，仅作前瞻指针
  url: https://huggingface.co/deepseek-ai
  finding: V3.1、V3.2 的 rope_scaling 与 V3 相同；V4-Flash、V4-Pro 为 yarn、factor 16、original_max_position_embeddings 65536、max_position_embeddings 1,048,576、qk_rope_head_dim 64；V4.1-Flash 的 text_config 同 V4。
  matters: 让读者知道本课的 RoPE + YaRN 在后续模型里仍在使用，具体含义留给第 18、19 课对照论文核实。
- id: huggyllama-configs
  kind: artefact
  title: huggyllama/llama-7b 等的 config.json（Hugging Face 社区转存，非 Meta 官方）
  a: huggyllama（Hugging Face）. 读取于 2026-09-24
  url: https://huggingface.co/huggyllama/llama-7b/blob/main/config.json
  finding: LLaMA-7B 的 hidden_size 4096、intermediate_size 11008、num_hidden_layers 32、vocab_size 32000、tie_word_embeddings false；13B/30B/65B 的 intermediate_size 与官方代码的取整规则逐一吻合。
  matters: 只作交叉核对：本课用它补上 LLaMA 论文没写的词表大小与“嵌入不共享”，从而把 6.7B 的参数量逐项数清。
- id: vaswani2017
  title: "Attention Is All You Need"
  a: Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). arXiv:1706.03762
  url: https://arxiv.org/abs/1706.03762
  finding: §3.5 的正弦位置编码加在输入嵌入上，波长构成从 $2\pi$ 到 $10000\cdot2\pi$ 的几何级数；作者设想对任意固定偏移 $k$，$PE_{pos+k}$ 可以写成 $PE_{pos}$ 的线性函数，从而便于按相对位置注意。
  matters: RoPE 的频率设定沿用了这组几何级数；它“加在输入上”的做法也是本课 §1 对照的起点。
- id: hf-llama-convert
  kind: artefact
  title: huggingface/transformers convert_llama_weights_to_hf.py
  a: Hugging Face（GitHub，main 分支）. 读取于 2026-09-24
  url: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
  finding: 转换 LLaMA 权重时对 $W^Q$、$W^K$ 的行做 permute（注释 “permute for sliced rotary”），把相邻配对改成前后半配对，以配合 rotate_half 写法。
  matters: 解释网上两种 RoPE 代码写法为什么都“对”，以及权重与实现必须配套。
- id: su2021roformer
  title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  a: Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021; Neurocomputing 568:127063, 2024). arXiv:2104.09864（读的是 v5）
  url: https://arxiv.org/abs/2104.09864
  finding: 从“q·k 只依赖相对位置”的要求出发，在二维用复数旋转构造出一个解（§3.4.1），再推广为块对角旋转矩阵，频率 $\theta_i=10000^{-2i/d}$；§3.4.3 给出远程衰减的一个上界并用图 2 数值展示。
  matters: 本课 RoPE 推导、多频率“时钟”、远程衰减图及其局限（只是上界的平均趋势）都以它为原始出处。
- id: su2021blog
  kind: artefact
  title: 苏剑林《Transformer升级之路：2、博采众长的旋转式位置编码》
  a: 苏剑林（科学空间 kexue.fm/archives/8265，页面日期 2021-03-23；经 Wayback 快照读取）
  url: https://kexue.fm/archives/8265
  finding: RoPE 的中文首发说明，把思路概括为“通过绝对位置编码的方式实现相对位置编码”；远程衰减图的图注写明 $d=128$。
  matters: 说明 RoPE 早于论文的出处，并补上 RoFormer 图 2 没有标出的维度。
- id: zhang2019rmsnorm
  title: "Root Mean Square Layer Normalization"
  a: Zhang, B., Sennrich, R. (2019). NeurIPS 2019. arXiv:1910.07467
  url: https://arxiv.org/abs/1910.07467
  finding: 假设 LayerNorm 成功靠的是重缩放不变性而非重中心化不变性，于是去掉减均值、只按均方根缩放；在多种任务上精度相当，运行时间减少 7%–64%。
  matters: LLaMA 与 DeepSeek 用 RMSNorm 的依据，也说明“放弃平移不变性”具体丢掉了什么。
- id: shazeer2020glu
  title: "GLU Variants Improve Transformer"
  a: Shazeer, N. (2020). arXiv:2002.05202
  url: https://arxiv.org/abs/2002.05202
  finding: 在 T5 的前馈层中用 GLU 变体替换 ReLU，参数与计算量匹配（$d_{ff}$ 乘以 2/3）时 SwiGLU、GEGLU 困惑度最好；作者明言没有理论解释。
  matters: SwiGLU 的定义与“$\tfrac23\cdot4d$”参数匹配算术的原始出处。
- id: touvron2023llama
  title: "LLaMA: Open and Efficient Foundation Language Models"
  a: Touvron, H., Lavril, T., Izacard, G., et al. (2023). arXiv:2302.13971
  url: https://arxiv.org/abs/2302.13971
  finding: 把 Pre-normalization + RMSNorm、SwiGLU（隐藏维 $\tfrac23\cdot4d$）与 RoPE 组合成 7B–65B 的开源模型家族（表 2 给出各尺寸的宽度、头数、层数）。
  matters: “现代 Transformer 块”的标准模板；DeepSeek LLM 明言沿用它的微观设计。
- id: chen2023pi
  title: "Extending Context Window of Large Language Models via Positional Interpolation"
  a: Chen, S., Wong, S., Chen, L., Tian, Y. (2023). arXiv:2306.15595（读的是 v2）
  url: https://arxiv.org/abs/2306.15595
  finding: 把位置下标按 $L/L'$ 线性压缩后微调约 1000 步，可把 LLaMA 扩到 32768；直接外推时困惑度可超过 $10^3$；插值误差界比外推界小约 600 倍（定理 2.1 的常数略偏乐观，改正后约 500 倍，见 §6）。
  matters: 上下文扩展的第一个基线；它“所有维度一样压缩”的做法正是 NTK-aware 与 YaRN 要修正的地方。
- id: peng2023yarn
  title: "YaRN: Efficient Context Window Extension of Large Language Models"
  a: Peng, B., Quesnelle, J., Fan, H., Shippole, E. (2023). ICLR 2024. arXiv:2309.00071（主读 v3，对照 v1、v2）
  url: https://arxiv.org/abs/2309.00071
  finding: 按波长把 RoPE 维度分为不插值、过渡、全插值三段（斜坡 $\gamma(r)$，$\alpha=1$、$\beta=32$），再把 $\mathbf q$、$\mathbf k$ 各乘 $0.1\ln s+1$；Llama 2 以 $s=16$ 训练 400 步扩到 64K，再从该检查点以 $s=32$ 训练 200 步扩到 128K（训练数据仍是 64K 长的片段）。
  matters: DeepSeek-V2/V3 扩到 128K 用的就是它；公式、默认超参数与版本间记号的变化是本课 §6–§7 的核心。
- id: deepseek2024v3
  title: "DeepSeek-V3 Technical Report"
  a: DeepSeek-AI (2024). arXiv:2412.19437（读的是 v2）
  url: https://arxiv.org/abs/2412.19437
  finding: §4.3 用两阶段 YaRN 扩展（4K → 32K → 128K，各 1000 步，batch 1920 / 480，学习率 $7.3\times10^{-6}$，$\sqrt t=0.1\ln s+1$）；表 1 给出上下文扩展耗 119K H800 GPU 小时。
  matters: 提供本课复算的扩展 token 数与成本，并暴露“YaRN 配置与 V2 一致”一句在温度系数上的出入。
- id: deepseek2024llm
  title: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"
  a: DeepSeek-AI (2024). arXiv:2401.02954
  url: https://arxiv.org/abs/2401.02954
  finding: §2.2 写明微观设计沿用 LLaMA：Pre-Norm + RMSNorm、SwiGLU（中间维 $\tfrac83d_{model}$）、RoPE；67B 用 GQA；表 2 给出 7B 为 30 层、宽 4096、上下文 4096。
  matters: 把本课的 LLaMA 式块与 DeepSeek 科技树的起点直接连起来。
- id: chowdhery2022palm
  title: "PaLM: Scaling Language Modeling with Pathways"
  a: Chowdhery, A., et al. (2022). arXiv:2204.02311（读的是 v5）
  url: https://arxiv.org/abs/2204.02311
  finding: 使用 SwiGLU（$d_{ff}$ 仍为 $4d$）与 RoPE，并在所有稠密层与 LayerNorm 中不用偏置，报告这提高了大模型训练的稳定性。
  matters: “无偏置”设计的明文出处；也是 LLaMA 把 SwiGLU 宽度改成 $\tfrac23\cdot4d$ 时的对照。
- id: radford2019gpt2
  kind: artefact
  title: "Language Models are Unsupervised Multitask Learners"
  a: Radford, A., Wu, J., Child, R., et al. (2019). OpenAI 技术报告（无 arXiv）
  url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  finding: §2.3：把 LayerNorm 移到每个子块的输入，并在最后一个自注意力块之后再加一层 LayerNorm；上下文从 512 增加到 1024 个 token。
  matters: 说明 Pre-Norm 早在 GPT-2 就有，并给出本课对照用的 1024 上下文。
- id: hf-gpt2
  kind: artefact
  title: GPT-2 的 Hugging Face 配置与实现（config.json、modeling_gpt2.py、Conv1D）
  a: OpenAI / Hugging Face（openai-community/gpt2；transformers main 分支）. 读取于 2026-09-24
  url: https://huggingface.co/openai-community/gpt2/blob/main/config.json
  finding: n_positions = 1024、n_embd = 768、n_head = 12、activation_function = gelu_new；位置表 wpe = nn.Embedding(1024, 768) 加在词嵌入上；前馈中间维默认 4 × hidden；LayerNorm 与 Conv1D 投影都带偏置。
  matters: 图 5.1 与 §1 表格中 GPT-2 一侧的真实维度、可学习位置表与偏置都以它为准。
- id: llama-v1-code
  kind: artefact
  title: meta-llama/llama（llama_v1 分支）llama/model.py
  a: Meta AI (2023, GitHub). 读取于 2026-09-24
  url: https://github.com/meta-llama/llama/blob/llama_v1/llama/model.py
  finding: RMSNorm 先转 float32 算 x·rsqrt(mean(x²)+eps) 再乘增益；RoPE 用复数形式、相邻两维配对；前馈为 w2(silu(w1 x)·w3 x)，hidden_dim 向上取整到 multiple_of = 256 的倍数（注释说是为了成为“2 的较大幂次”的倍数）；所有线性层 bias=False。
  matters: LLaMA 论文没写的实现细节（无偏置、11008 的取整规则）以它为准。
- id: ds-llm-hf
  kind: artefact
  title: deepseek-ai/deepseek-llm-7b-base 与 67b-base 的 config.json 与 pytorch_model.bin.index.json（Hugging Face）
  a: DeepSeek-AI（Hugging Face）. 读取于 2026-09-24
  url: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base/blob/main/config.json
  finding: 两个配置都声明 LlamaForCausalLM：7B 为 30 层、宽 4096、intermediate_size 11008，67B 为 95 层、宽 8192、22016、8 个 KV 头；rms_norm_eps 1e-6、rope_theta 10000、max_position_embeddings 4096；7B（273 个张量）与 67B（858 个张量）的权重索引里没有一个 bias 张量。
  matters: 补上 DeepSeek LLM 论文没写的 $\epsilon$、RoPE 底数、$d_{ff}$ 与“无偏置”，说明它的四块积木与 LLaMA 完全相同。
- id: ds-configs-code
  kind: artefact
  title: DeepSeek-V2 / V2-Lite / V3 的 config.json 与 modeling_deepseek.py；DeepSeek-V3 inference/model.py
  a: DeepSeek-AI (2024, Hugging Face / GitHub). 读取于 2026-09-24
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
  finding: rope_scaling 为 yarn、factor 40、beta_fast 32、beta_slow 1、original_max_position_embeddings 4096；V2 的 mscale_all_dim = 0.707，V3 为 1.0；温度通过 softmax_scale 作用于全部 192 维；HF 版 RMSNorm 先转 float32，官方推理代码直接调 F.rms_norm；官方推理代码的 Linear 默认 bias=False，缓存里存的是旋转之后的 $\mathbf k^R$。
  matters: 用代码核实两篇论文的 YaRN 设置与温度系数，并说明“只作用于 $\mathbf k^R$”在实现上的准确含义。
```
