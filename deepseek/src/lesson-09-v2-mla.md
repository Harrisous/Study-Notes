---
title: 第 9 课 · DeepSeek-V2（上）：多头潜在注意力 MLA
short: MLA 潜在注意力
kicker: DeepSeek 科技树 · 第 9 课
description: DeepSeek-V2 的多头潜在注意力（MLA）逐式精讲：低秩 KV 联合压缩、矩阵吸收的逐行推导与每个解码 token 的算力账、RoPE 为什么破坏吸收（二维旋转反例）、解耦 RoPE、查询压缩、MHA 模式与 MQA 模式，以及 MHA/GQA/MQA/MLA 每 token 缓存的字节账单与“93.3%”的拆解。
lede: 第 4 课留下一个问题：KV cache 随上下文线性增长，MQA 和 GQA 靠减少键值头的个数把它压小，却要付质量的代价。DeepSeek-V2 给出了另一种答案：**头数一个不减，而是把所有头的键和值联合压成一个 512 维的潜向量**，推理时只缓存它，外加一个 64 维的位置键。这一课把 MLA 的每一个公式抄下来、讲明白，再用 V2 的真实维度把每一笔账算出来：缓存省了多少字节，算力多花了多少，RoPE 这个“小麻烦”又为什么逼出了一个解耦设计。
meta: 预计阅读 190 分钟，练习另需约 50 分钟 · 前置：第 4、5 课，数学 M01、M02
index: index.html
prev: lesson-08-deepseekmath-grpo.html | 第 8 课 · DeepSeekMath 与 GRPO
next: lesson-10-v2-moe-training.html | 第 10 课 · DeepSeek-V2 训练
---

## 这一课回答什么问题

2024 年 5 月发布的 DeepSeek-V2 有两项架构创新：注意力层换成了**多头潜在注意力**（multi-head latent attention，MLA），前馈层换成了第 7 课的 DeepSeekMoE [@dsv2]。这一课只讲前一半，后一半和训练、对齐放在第 10 课。读完这一课，下面这些问题你都应该能自己回答，并且能用计算器验算：

- 标准多头注意力（MHA）每个 token 要缓存多少个数？为什么 MQA、GQA 能省缓存，又为什么会掉点？
- MLA 缓存的“潜向量”是什么？它凭什么能代替 128 个头各自的键和值？
- 论文说推理时 $W^{UK}$ 可以被“吸收”进查询一侧、$W^{UV}$ 可以被吸收进 $W^O$，所以连键和值都不必算出来。这一步的每一行怎么推？每解码一个 token 到底省了多少运算，又多花了多少？
- 为什么 RoPE（第 5 课）会让吸收失效？“解耦 RoPE”那个额外的 64 维键为什么只有一个、所有头共享？
- 表格里“MLA 相当于 2.25 组的 GQA”“KV cache 降低 93.3%”这些数字是怎么算出来的？哪些是 MLA 的功劳，哪些不是？
- 同一层 MLA 为什么在训练、预填充时按“MHA 模式”算，在解码时按“MQA 模式”算？这和第 14 课的稀疏注意力有什么关系？

## 这一课在科技树上的位置

这一课是主线 A“注意力与 KV cache 经济学”的中心节点。[第 1 课](lesson-01-transformer-anatomy.html)讲了多头注意力；[第 4 课](lesson-04-inference-kv-cache.html)算出了它在推理时的账单：每生成一个 token，都要把之前所有 token 的键和值从显存里读一遍，缓存随上下文线性增长；MQA 和 GQA 让多个查询头共用一组键值，缓存小了，质量也降了。第 4 课的结尾问：**能不能不减头数，也把缓存压小？**[第 5 课](lesson-05-modern-block.html)讲了 RoPE，它会在本课 §4 变成一个障碍。[第 6 课](lesson-06-deepseek-llm.html)的 DeepSeek LLM 67B 用的是 GQA，它将是本课所有“对比 67B”数字的基线。

往后看：[第 11 课](lesson-11-v3-architecture.html)的 DeepSeek-V3 原样沿用本课的 MLA；[第 14 课](lesson-14-sparse-attention.html)的 DSA 稀疏注意力建立在本课 §7 的“MQA 模式”上；[第 18 课](lesson-18-v4.html)、[第 19 课](lesson-19-v4-1-flash.html)沿着“缓存更少的字节”这条路继续往下压。上一课（[第 8 课](lesson-08-deepseekmath-grpo.html)）属于后训练那条主线，但它的总账里有一个与本课直接相关的数：在 DeepSeekMath 7B 上，GRPO 每道题采 64 个回答、每个最长 1024 个 token，按满长度算，光是 KV cache（bf16）就要 30 GiB。无论是部署推理还是强化学习的采样，生成都被缓存卡着脖子。

## 读完你能……

1. 写出 MLA 的全部公式（论文附录 C 的式 37–47），说出每个矩阵的形状和作用，指出推理时缓存的是哪两个向量、每 token 每层多少个数。
2. 逐行推导矩阵吸收，写出每一步的张量形状；用 V2 的维度算出每解码一个 token 的注意力运算量（三种算法），说清“用算力换带宽”换了多少。
3. 用一个 2×2 旋转的数值例子说明 RoPE 为什么让吸收失效，并解释解耦 RoPE 怎样绕开它、代价是多少。
4. 算出 MHA、GQA、MQA、MLA 在 V2 维度下每 token 的缓存（元素数与字节数）和“等效 GQA 组数”，并把论文的“93.3%”拆成三个因子。
5. 区分 MLA 的 MHA 模式与 MQA 模式，说出训练/预填充与解码为什么各用一种，并用屋顶线模型解释解码内核为什么接近算力受限。

::: note 本课的记号
- 符号沿用 DeepSeek-V2 论文与整门课的约定：$d$ 隐藏维，$n_h$ 头数，$d_h$ 每头维度，$l$ 层数（KV cache 公式里沿用论文的小写 $l$），$T$ 序列长度或缓存里的 token 数。V2 的取值：$d=5120$，$n_h=d_h=128$，$l=60$ [@dsv2]。
- MLA 新增的维度：$d_c$ 是 KV 压缩维（V2 取 512），$d_c'$ 是查询压缩维（1536），$d_h^R$ 是解耦 RoPE 的每头维度（64）。
- 矩阵上标：D = down（下投影，压缩），U = up（上投影，展开），R = RoPE 分支，C（写在向量上）= compressed，指“由压缩向量得到的内容部分”。
- 按头切块（课程补充记号，论文没有）：$W^{UK}_i\in\R^{d_h\times d_c}$ 是 $W^{UK}$ 中属于第 $i$ 个头的那 $d_h$ 行；$W^{UV}_i$、$W^{UQ}_i$ 同理，$W^{QR}_i\in\R^{d_h^R\times d_c'}$ 是 $W^{QR}$ 中属于第 $i$ 个头的那 $d_h^R$ 行；$W^O_i\in\R^{d\times d_h}$ 是 $W^O$ 中属于第 $i$ 个头的那 $d_h$ 列。
- $[\mathbf a;\mathbf b]$ 表示把两个列向量上下拼接。RoPE 的旋转矩阵写作 $\mathbf R_{\Theta,t}$（第 5 课的记号），在不会混淆时简写成 $\mathbf R_t$，注意别把它和上标 $R$、实数集 $\R$ 混起来。
- 字节：KB = 1000 字节，KiB = 1024 字节。论文图 1(b) 与 ISCA '25 论文都用 KB，本课随它们。
:::

[[legend]]

[[toc]]

# 问题：KV cache 的账单 | MHA 的缓存有多大，MQA、GQA 省下它要付什么代价 | 20

先把第 4 课的结论摆出来。语言模型生成文本是一个循环：每一步只产生一个新 token，而为了产生它，新 token 的查询要和**之前所有 token** 的键做点积、再对它们的值加权求和。之前那些 token 的键和值不会再变（因果掩码保证了这一点），所以推理系统把它们存起来，每一步只为新 token 算一份新的，这就是 **KV cache**。它省掉了重复计算，但它本身要占显存，而且每一步都要被完整地读一遍。

::: ana 类比：KV cache = 会议纪要本
**映射。** 每个发言者（token）说完，秘书把他的“标签 + 要点”（键 $\mathbf k$、值 $\mathbf v$）记在本子上；后来的人只翻本子，不用重听录音。

**收益。** 它说明了为什么缓存是精确的：纪要是逐字抄下的副本，翻本子和重听录音得到的内容一模一样。

**失效的地方。** 纪要逐字精确，所以它省的是重复计算，**不省空间**：本子的厚度随发言人数线性增长，而且每个“头”都有自己的一套标签和要点。这一课要做的，就是换一种更短的记法。
:::

## MHA 的公式与它的缓存

DeepSeek-V2 论文先用自己的记号写下标准多头注意力（MHA），后面所有的 MLA 公式都在它上面改 [@dsv2]。记 $\mathbf h_t\in\R^{d}$ 为第 $t$ 个 token 进入注意力层时的隐藏状态，一列 $d$ 个数。三个投影矩阵把它变成查询、键、值：

::: eq MHA 的三个投影（论文式 1–3） | (9.1)
$$\mathbf q_t=W^{Q}\mathbf h_t,\qquad \mathbf k_t=W^{K}\mathbf h_t,\qquad \mathbf v_t=W^{V}\mathbf h_t,\qquad W^Q,W^K,W^V\in\R^{n_hd_h\times d}$$
$\mathbf q_t,\mathbf k_t,\mathbf v_t\in\R^{n_hd_h}$：所有头的查询（键、值）首尾相接成一个长向量。
:::

把长向量按每 $d_h$ 个数切一段，就得到每个头自己的查询、键、值（切头只是换一种方式读同一个向量，不花任何运算，见 [M01 §4](../math/m01-vectors-matrices.html#p4)）：

::: eq 切成多个头（论文式 4–6） | (9.2)
$$\mathbf q_t=[\mathbf q_{t,1};\mathbf q_{t,2};\dots;\mathbf q_{t,n_h}],\qquad \mathbf k_t=[\mathbf k_{t,1};\dots;\mathbf k_{t,n_h}],\qquad \mathbf v_t=[\mathbf v_{t,1};\dots;\mathbf v_{t,n_h}]$$
每一段 $\mathbf q_{t,i},\mathbf k_{t,i},\mathbf v_{t,i}\in\R^{d_h}$。
:::

第 $i$ 个头用自己的查询去和第 $1$ 到第 $t$ 个 token 的键打分，softmax 之后对值加权求和；各头的结果拼起来再乘输出矩阵 $W^O$：

::: eq 每个头的注意力与输出（论文式 7–8） | (9.3)
$$\mathbf o_{t,i}=\sum_{j=1}^{t}\softmax_j\!\Big(\frac{\mathbf q_{t,i}^\top\mathbf k_{j,i}}{\sqrt{d_h}}\Big)\mathbf v_{j,i},\qquad \mathbf u_t=W^O[\mathbf o_{t,1};\mathbf o_{t,2};\dots;\mathbf o_{t,n_h}]$$
$\softmax_j$ 表示对下标 $j=1,\dots,t$ 做 softmax；$W^O\in\R^{d\times n_hd_h}$；$\mathbf u_t\in\R^{d}$ 是注意力层的输出，会加回残差流。
:::

生成第 $t$ 个 token 时，式 (9.3) 用到了所有 $j\le t$ 的 $\mathbf k_{j,i}$ 和 $\mathbf v_{j,i}$。它们在更早的步骤里已经算过，存下来就不必重算。每个 token 要存多少个数？

::: deriv MHA 每个 token 的缓存
$n_hd_h$ || 一层里这个 token 所有头的键拼成的 $\mathbf k_t$，式 (9.1)
$n_hd_h+n_hd_h=2n_hd_h$ || 再加上同样长的值 $\mathbf v_t$；查询不用存，因为只有当前 token 的查询会被用到
$2n_hd_h\cdot l$ || 每一层都有自己的 $W^K$、$W^V$，所以要乘层数 $l$
:::

::: eq MHA 每 token 的 KV cache（按元素计） | (9.4)
$$\text{MHA：}\quad 2\,n_h\,d_h\,l\ \text{个元素}$$
论文原话是 MHA “needs to cache $2n_hd_hl$ elements for each token”，并指出这份沉重的缓存限制了最大 batch size 和序列长度 [@dsv2]。字节数 = 元素数 × 每个元素的字节数（bf16 为 2）。
:::

::: ex 如果 DeepSeek-V2 用 MHA，每个 token 要缓存多少
V2 的注意力形状是 $n_h=128$、$d_h=128$、$l=60$ [@dsv2]。代入 (9.4)：

$2\times128\times128\times60=1{,}966{,}080$ 个元素；用 bf16 存，每个 2 字节，就是 $3{,}932{,}160$ 字节，约 3.93 MB，**每个 token**。

一条 128K（$T=131{,}072$）的序列：$3{,}932{,}160\times131{,}072\approx5.154\times10^{11}$ 字节，约 515.4 GB。这还只是一条序列的缓存，不含模型权重。

对照 DeepSeek LLM 67B（第 6 课）：95 层、64 个查询头、8 组键值头、$d_h=128$ [@dsllm]，每 token $2\times8\times128\times95=194{,}560$ 个元素，bf16 下 $389{,}120$ 字节（本课按 KB 记作 389.1 KB；第 4 课按 KiB 记作 380 KiB，是同一个数）。67B 原生只支持 4K 上下文，这里的 128K 只是为了对比：$389{,}120\times131{,}072\approx51.0$ GB。
:::

::: fig figures/l09-cache-growth.svg | 图 9.1 单条序列的 KV cache 随上下文长度增长（DeepSeek-V2 的形状，bf16） | wide scroll
**看 128K 那一列：同样是 128 个查询头、60 层，MHA 要 515.4 GB，已经远远高出红色虚线（一张 80 GB 显存 GPU 的全部容量）。** 四条线都是直线，因为缓存与长度成正比，双对数坐标下斜率为 1；线与线之间的竖直距离处处相等，对应每 token 缓存之比（对数坐标上的距离是比值的对数）。GQA 取 8 组，MQA 取 1 组；橙色的 MLA 线是本课的目标，它落在 GQA 与 MQA 之间，§8 会说明它“相当于 2.25 组 GQA”。维度取自 DeepSeek-V2 [@dsv2]，80 GB 参考线沿用第 4 课。
:::

缓存大，伤的是两件事。一是**容量**：显存要同时放下模型权重和所有正在服务的序列的缓存，缓存越大，能同时服务的序列越少、能支持的上下文越短。二是**带宽**：第 4 课算过，解码时每生成一个 token，都要把整份缓存从显存读一遍，而对每个读进来的数只做很少的运算，所以解码的速度主要由“读得多快”决定，而不是“算得多快”。缓存小一倍，每步要读的字节也少一倍。

## MQA 与 GQA：减少键值头的个数

第 4 课讲过两种省缓存的办法。**MQA**（multi-query attention）让所有查询头共用同一个键和同一个值 [@mqa]；**GQA**（grouped-query attention）折中一下，把 $n_h$ 个查询头分成 $n_g$ 组，每组共用一对键值 [@gqa]。缓存公式里的 $n_h$ 于是换成了 $n_g$：

::: eq GQA 与 MQA 的缓存（论文表 1） | (9.5)
$$\text{GQA：}\ 2\,n_g\,d_h\,l,\qquad \text{MQA：}\ 2\,d_h\,l\quad(n_g=1)$$
$1\le n_g\le n_h$。$n_g=n_h$ 时就是 MHA。DeepSeek LLM 67B 取 $n_g=8$ [@dsllm]。
:::

它们省缓存的方式很直接：少存几份键值。代价也很直接：原本每个头可以有自己的“标签”（键）和“内容”（值），现在一组头只能共用一份。DeepSeek-V2 论文在附录 D.1 做了对照实验：三个约 7B 的稠密模型，除注意力外结构相同，都训练 1.33T token，通过调整层数把参数量对齐 [@dsv2]。

::: fig figures/l09-ablation-gqa.svg | 图 9.2 同样约 7B 参数、同样 1.33T token：MHA、GQA、MQA 的得分（据 DeepSeek-V2 附录 D.1 表 8 绘制） | wide scroll
**看每一行从左到右的三个点：键值头越少，得分越低，四个基准无一例外。** 差距最大的是 C-Eval：MHA 42.9，GQA（8 组）37.7，MQA 30.0。论文据此说 MHA 在这些基准上“有显著优势” [@dsv2]。这是在 7B 稠密模型上的实测，三个模型各自的层数论文没有给出。
:::

::: warn 这组实验能说明什么，不能说明什么
它说明在这个规模和训练量下，减少键值头会实实在在地掉点，而且 MQA 比 GQA 掉得多。它不能说明 GQA 在所有规模上都掉这么多：更大的模型、更多的训练数据下差距可能不同，论文没有做这样的扫描。所以更稳妥的说法是：**减少键值头是有质量代价的，DeepSeek 在自己的实验里看到了这个代价**。论文在正文里也只说它们“performance does not match MHA” [@dsv2]。
:::

## 这一课要回答的问题

把两件事放在一起，问题就清楚了。缓存的大小由 (9.4)、(9.5) 决定，能调的旋钮只有键值头数 $n_g$；把 $n_g$ 调小，质量就降。有没有**另一个旋钮**：让 128 个头都保留各自的键和值，但缓存里存的东西远小于 $2n_hd_h$ 个数？

换一个问法：128 个头的键和值加起来有 $2\times128\times128=32{,}768$ 个数，它们真的需要 32,768 个自由度吗？如果它们其实都能从一个短得多的向量“算出来”，那么缓存这个短向量就够了。下一页就是这个想法的精确版本。

:::: quiz 小练习：缓存公式
（1）DeepSeek LLM 7B 是 30 层、32 个头、$d_h=128$ 的 MHA [@dsllm]。每 token 的 KV cache 是多少元素、多少字节（bf16）？（2）它和 67B 相比，谁的每 token 缓存更大？为什么参数多近 10 倍的 67B 反而更小？
::: answer 看答案
（1）$2\times32\times128\times30=245{,}760$ 个元素，$491{,}520$ 字节（约 491.5 KB）。

（2）7B 更大（491.5 KB 对 389.1 KB）。67B 虽然有 95 层，但它用 GQA，只存 8 组键值（$2\times8\times128=2{,}048$ 个数每层）；7B 每层要存 $2\times32\times128=8{,}192$ 个，是 67B 每层的 4 倍，层数只少了约 3 倍。缓存由“每层存多少 × 层数”决定，和总参数量没有直接关系。
:::
::::

::: key 本页要点
- MHA 每 token 缓存 $2n_hd_hl$ 个元素。DeepSeek-V2 的形状若用 MHA，是 1,966,080 个元素、3.93 MB（bf16），128K 上下文一条序列 515.4 GB。
- 缓存同时吃掉显存容量和解码带宽：每生成一个 token 都要把它读一遍。
- MQA、GQA 把 $n_h$ 换成 $n_g$ 来省缓存，但 V2 论文在 7B 上的实测显示得分随键值头数减少而下降。MLA 要找的是另一个旋钮。
:::

# 低秩 KV 联合压缩 | 所有头的键和值都从一个 512 维潜向量展开 | 25

**问题。** 上一页问：128 个头的键和值，能不能都从一个短向量算出来？如果能，推理时只缓存这个短向量。这一页给出 MLA 的核心构造，**低秩 KV 联合压缩**（low-rank joint compression for keys and values）。论文说它是“MLA 的核心” [@dsv2]。

::: pre 秩与低秩分解 | 读过 M01 §2 的“外积之和”或 M02 的可以跳过
一个矩阵 $A\in\R^{m\times n}$ 若能写成两个瘦矩阵的乘积 $A=UV$，其中 $U\in\R^{m\times r}$、$V\in\R^{r\times n}$，那么对任何输入 $\mathbf x$，输出 $A\mathbf x=U(V\mathbf x)$ 都是 $U$ 的 $r$ 列的某个组合，也就是说所有输出都落在一个至多 $r$ 维的子空间里。这时称 $A$ 的**秩**（rank）不超过 $r$。例：$A=\begin{pmatrix}1\\2\\3\end{pmatrix}\begin{pmatrix}1&1\end{pmatrix}$ 是 $3\times2$ 的矩阵，但无论输入是什么，输出都是 $(1,2,3)$ 的倍数，秩为 1。存 $U$、$V$ 两块只需 $(m+n)r$ 个数，存整张 $A$ 需要 $mn$ 个。$r$ 远小于 $m,n$ 时这叫**低秩分解**。完整的讲法见 [M01 §2](../math/m01-vectors-matrices.html#p2) 与 [M02](../math/m02-rank-svd-lowrank.html)。
:::

::: ana 类比：MLA = 速记
**映射。** 上一页的纪要本（KV cache）原本逐字抄下每位发言者给每个头准备的标签和要点；MLA 改成记**速记符号**：每个 token 只记一个短的潜向量 $\mathbf c^{KV}$，要用的时候再按各头的“展开规则”（$W^{UK}$、$W^{UV}$）还原成标签和要点。下一页的“矩阵吸收”相当于读者练出了直接读速记的眼光，连展开都省了。

**收益。** 它说明了为什么缓存能小一个数量级：本子上只记速记，而展开规则是固定的、所有 token 共用，不必每条都抄一遍。

**失效的地方。** 速记是**学出来的线性压缩**，会丢信息，模型必须在训练中学会只用速记里剩下的东西；它不是先有逐字纪要、再事后压缩的。另外，位置信息（RoPE）没法写进这种速记，只能在旁边另记一小栏，这是 §4、§5 的内容。
:::

## 定义：先压下去，再展开

MLA 给每个 token 先算一个短向量，再从它展开出所有头的键和值 [@dsv2]：

::: eq 低秩 KV 联合压缩（论文式 9–11） | (9.6)
$$\mathbf c_t^{KV}=W^{DKV}\mathbf h_t,\qquad \mathbf k_t^{C}=W^{UK}\mathbf c_t^{KV},\qquad \mathbf v_t^{C}=W^{UV}\mathbf c_t^{KV}$$
$\mathbf c_t^{KV}\in\R^{d_c}$ 是第 $t$ 个 token 的压缩潜向量，推理时真正写进缓存的东西；$W^{DKV}\in\R^{d_c\times d}$ 是下投影；$W^{UK},W^{UV}\in\R^{d_hn_h\times d_c}$ 是上投影；$\mathbf k_t^C,\mathbf v_t^C\in\R^{d_hn_h}$ 是展开后的键和值（上标 C 表示“由压缩向量得到”）。论文要求 $d_c\ll d_hn_h$。
:::

“潜向量”（latent vector）这个名字值得拆开看。**含义**：$\mathbf c_t^{KV}$ 是一个不直接等于任何一个头的键或值的中间向量。**理解**：它的 512 个分量各代表什么，没有人事先规定，是训练中学出来的一份“公共底稿”；单看它，读不出任何一个头的键，只有乘上某个头的 $W^{UK}_i$ 或 $W^{UV}_i$，才变成那个头能用的键或值。“潜”说的就是这种藏在后面、要经过一次线性变换才显形的角色。**数学**：它就是 (9.6) 的第一式，$\mathbf c^{KV}_t=W^{DKV}\mathbf h_t\in\R^{d_c}$，一个 $d_c$ 维的列向量。

逐个交代形状。$\mathbf h_t$ 是 $[5120]$；$W^{DKV}$ 是 $[512,5120]$，把它压成 $[512]$ 的 $\mathbf c_t^{KV}$；$W^{UK}$ 是 $[16384,512]$，把 512 个数展开成 $128\times128=16{,}384$ 个数，恰好是 128 个头的键拼起来的长度，$W^{UV}$ 同理。V2 取 $d_c=512$，是 $d_hn_h=16{,}384$ 的 1/32 [@dsv2]。

按头看，展开是逐块进行的。$W^{UK}$ 按行切成 128 块，每块 $W^{UK}_i\in\R^{d_h\times d_c}$ 属于一个头（这是 [M01 §4](../math/m01-vectors-matrices.html#p4) 的“按行分块”，论文附录 C 的式 42、45 也写了按头切片）：

::: eq 每个头的键和值 | (9.7)
$$\mathbf k^C_{t,i}=W^{UK}_i\,\mathbf c_t^{KV},\qquad \mathbf v^C_{t,i}=W^{UV}_i\,\mathbf c_t^{KV},\qquad W^{UK}_i,W^{UV}_i\in\R^{d_h\times d_c}$$
128 个头的键各不相同（每头的 $W^{UK}_i$ 不同），但它们都是同一个 $\mathbf c_t^{KV}$ 的线性函数。
:::

这就是 MLA 和 MQA/GQA 的根本区别。MQA 让所有头**共用同一个键**；MLA 让所有头**共用同一个来源**，每个头从这个来源里读出自己的键。头数一个没减。

::: ex 一个能手算的小 MLA
取隐藏维 $d=3$、两个头、每头 $d_h=2$、潜向量只有 $d_c=1$ 维。下投影是一行 $W^{DKV}=(1,0,1)$；上投影各是一列 $W^{UK}=(1,2,0,1)^\top$、$W^{UV}=(1,0,1,1)^\top$，按头切开就是 $W^{UK}_1=(1,2)^\top$、$W^{UK}_2=(0,1)^\top$、$W^{UV}_1=(1,0)^\top$、$W^{UV}_2=(1,1)^\top$。

两个 token：$\mathbf h_1=(2,5,1)$，$\mathbf h_2=(1,-4,-2)$。潜向量 $c_1=2+1=3$，$c_2=1-2=-1$。展开：

- 头 1 的键：$\mathbf k_{1,1}=3\,(1,2)=(3,6)$，$\mathbf k_{2,1}=-1\,(1,2)=(-1,-2)$；头 2 的键：$(0,3)$、$(0,-1)$。
- 值：$\mathbf v^C_1=3\,(1,0,1,1)$，即头 1 的值 $(3,0)$、头 2 的值 $(3,3)$；$\mathbf v^C_2$ 同理是它的 $-\tfrac13$ 倍。

**缓存。** 每个 token 只存 1 个数（$c_1=3$、$c_2=-1$）；如果按 MHA 存两个头的键和值，是 $2\times2\times2=8$ 个数。

**代价。** 头 1 的键永远是 $(1,2)$ 的倍数，头 2 的键永远是 $(0,1)$ 的倍数：键被限制在一维子空间里，这就是“低秩”。另外，$\mathbf h$ 沿 $(0,1,0)$ 或 $(1,0,-1)$ 方向的变化完全到不了潜向量（$W^{DKV}$ 把它们乘成 0），例如 $\mathbf h_1$ 的第二个分量从 5 改成 $-4$，缓存里的 $c_1$ 纹丝不动。这是一个压缩得过头的玩具；V2 的 $d_c=512$ 要宽得多，但道理相同：训练必须让真正有用的信息落在这 512 维里。
:::

## 为什么叫“低秩”：它等价于给 MHA 的投影加了一个秩约束

把 (9.6) 的两步合成一步，看看每个 token 的键值到底是 $\mathbf h_t$ 的什么函数。下面暂时忽略 §6 要讲的一个细节（压缩向量之后还有一层 RMSNorm，它是什么见[第 5 课 §4](lesson-05-modern-block.html#p4)）：

::: deriv 联合压缩 = 一个秩不超过 512 的键值投影
$\begin{bmatrix}\mathbf k^C_t\\ \mathbf v^C_t\end{bmatrix}=\begin{bmatrix}W^{UK}\mathbf c^{KV}_t\\ W^{UV}\mathbf c^{KV}_t\end{bmatrix}$ || 式 (9.6)，把键和值上下拼成一个 $[32768]$ 的向量
$=\begin{bmatrix}W^{UK}\\ W^{UV}\end{bmatrix}\mathbf c^{KV}_t$ || 分块乘法：外侧（行）切开的矩阵乘同一个向量，见 M01 (M1.12)
$=\underbrace{\begin{bmatrix}W^{UK}\\ W^{UV}\end{bmatrix}}_{[32768,\,512]}\underbrace{W^{DKV}}_{[512,\,5120]}\mathbf h_t$ || 代入 $\mathbf c^{KV}_t=W^{DKV}\mathbf h_t$，用结合律
$\text{对照 MHA：}\begin{bmatrix}\mathbf k_t\\ \mathbf v_t\end{bmatrix}=\underbrace{\begin{bmatrix}W^{K}\\ W^{V}\end{bmatrix}}_{[32768,\,5120]}\mathbf h_t$ || 式 (9.1)：MHA 的键值投影是一整块 $[32768,5120]$ 的矩阵
:::

对照最后两行：MLA 相当于要求 MHA 的键值投影矩阵 $[W^K;W^V]$ 必须能写成 $[32768,512]$ 乘 $[512,5120]$，也就是**秩不超过 512**。换成几何的说法：每个 token 的 256 个键值头向量（128 个键、128 个值）拼成的那个 32,768 维长向量，对所有 token 都落在同一个至多 512 维的子空间里，即 $[W^{UK};W^{UV}]$ 的列空间。MLA 的键值和 MHA 一样有 32,768 个输出维度，但其中随 token 变化的“自由度”至多 512 个；MHA 的上限是 5120 个，由 $\mathbf h_t$ 的维度决定。这个子空间的说法在补回 RMSNorm 之后仍然成立：RMSNorm 只是把 $\mathbf c^{KV}_t$ 的每个分量各乘一个数，展开出来的键值仍然是 $[W^{UK};W^{UV}]$ 各列的组合；变的只是“键值是 $\mathbf h_t$ 的线性函数”这一点。

这里的“联合”（joint）指键和值共用**同一个**潜向量。另一种可以想到的做法是给键、给值各压一个潜向量，那样缓存会是两个。V2 论文附录 D 的注意力消融只比较了 MHA、GQA、MQA 与 MLA，没有报告“键值分开压缩”的对照实验，所以“联合”是一个设计选择，论文没有给出它与分开压缩孰优孰劣的实验证据。

::: fig figures/l09-four-attention.svg | 图 9.3 四种注意力在推理时各缓存什么（据 DeepSeek-V2 论文图 3 重绘） | wide scroll
**看琥珀色粗框：那是推理时真正写进缓存的东西。** MHA 每个头一对 K、V；GQA 每组头一对；MQA 全体共用一对；MLA 的 K、V 是虚线框，推理时不缓存，缓存的是下面那个橙色的潜向量 $\mathbf c^{KV}$ 和一个红色的小键 $\mathbf k^R$。原图是“简化示意”，MLA 面板只画了潜向量，**没有画 $\mathbf k^R$**，这里按论文正文与附录 C 补上（§5 讲它）。每个面板和原图一样只画 8 个查询头（GQA 面板画成 4 组），底部的数字则按 V2 的 $n_h=d_h=128$、$d_c=512$、$d_h^R=64$ 计算，GQA 取 8 组；“论文表 1”一行是论文对四者能力的定性评价 [@dsv2]。
:::

## 缓存了什么，省了多少

推理时，MLA 只需要把 $\mathbf c_t^{KV}$ 存下来；需要键和值的时候，可以用 $W^{UK}$、$W^{UV}$ 现场展开（下一页会说明甚至连展开都可以省掉）。先不算位置那一小栏，缓存就是：

::: eq 只缓存潜向量时的 KV cache | (9.8)
$$\text{MLA（不含解耦 RoPE 键）：}\quad d_c\,l\ \text{个元素}$$
论文原话：推理时 MLA “only needs to cache $\mathbf c_t^{KV}$”，缓存 $d_cl$ 个元素 [@dsv2]。§5 会加上解耦 RoPE 的键，变成 $(d_c+d_h^R)\,l$。
:::

::: fig figures/l09-joint-compression.svg | 图 9.4 低秩 KV 联合压缩，按真实维度的比例画 | wide scroll
**看中间那一条细缝：5120 维的 $\mathbf h_t$ 被压到 512 维，再展开成两根各 16,384 维的长条。** 竖条的高度与向量长度成正比。缓存的只有中间那 512 个数（琥珀色框）。右边两根长条每根是 128 个头 × 每头 128 维，推理时不缓存。底部比较了参数量：MLA 的三块矩阵合计约 1,940 万个参数，只有 MHA 的 $W^K$、$W^V$（约 1.68 亿）的 12% 左右。维度取自 DeepSeek-V2 [@dsv2]。
:::

::: ex 每层每 token 缓存多少个数、每层有多少参数
**缓存。** MHA：键 16,384 + 值 16,384 = 32,768 个数。MLA：潜向量 512 个数，是 MHA 的 $512/32{,}768=1/64$。

**参数。** MHA 的 $W^K$、$W^V$ 各是 $[16384,5120]$：$2\times16{,}384\times5{,}120=167{,}772{,}160$，约 1.68 亿。MLA 的 $W^{DKV}$ 是 $[512,5120]$，$2{,}621{,}440$；$W^{UK}$、$W^{UV}$ 各 $[16384,512]$，各 $8{,}388{,}608$。合计 $19{,}398{,}656$，约 1,940 万，是 MHA 的 11.6%。加上 §5 的 $W^{KR}$（$[64,5120]$，$327{,}680$），是 $19{,}726{,}336$。

所以 MLA 不只省缓存，键值投影的参数也少了将近一个数量级（$167{,}772{,}160/19{,}398{,}656\approx8.6$ 倍）。这两件事来自同一个原因：中间那个 512 维的瓶颈。
:::

::: note 一个名字上的巧合：kv_lora_rank
在 Hugging Face 上发布的 V2 配置文件里，$d_c$ 和 $d_c'$ 分别叫 `kv_lora_rank = 512` 和 `q_lora_rank = 1536` [@hf-v2]。名字借用了 LoRA（低秩适配）的“rank”，指的正是这里的低秩瓶颈维数。它和微调用的 LoRA 没有别的关系：MLA 的这些矩阵是从头预训练的，不是附加在某个现成模型上的适配器。
:::

::: err 常见误解：MLA 是对 MHA 的权重做 SVD 压缩得到的
不是。DeepSeek-V2 从零开始预训练，$W^{DKV}$、$W^{UK}$、$W^{UV}$ 从一开始就是模型的一部分，和其它参数一起用梯度下降训练。DeepSeek 在 ISCA '25 论文里说得很直接：潜向量的投影矩阵是“jointly trained with the model” [@isca25]。“先训一个 MHA，再用 SVD 找它的最优低秩近似”是另一回事，它的误差由被丢掉的奇异值决定（[M02 §5](../math/m02-rank-svd-lowrank.html#p5) 的最优低秩近似定理），而 MLA 根本没有一个“原版 MHA”需要去近似。
:::

::: warn 还没解决的两件事
第一，**展开要花运算**。如果每生成一个 token，都要把缓存里 $T$ 个潜向量各乘一次 $W^{UK}$、$W^{UV}$，那么省下的显存会用大量计算来还；可要是把展开后的键值存起来，缓存又回到了 MHA 的大小。下一页用矩阵乘法的结合律解决它。

第二，**位置编码**。V2 和 LLaMA 一样用 RoPE 编码位置。RoPE 要旋转每个头的键，而上面的 $\mathbf k^C_{t,i}$ 是从潜向量现场展开的。§4 会说明这两者为什么冲突，§5 给出解决办法。
:::

:::: quiz 小练习：数一数
V2-Lite 是 V2 的缩小版：27 层、16 个头、$d_h=128$、$d_c=512$ [@dsv2]。（1）若它用 MHA，每层每 token 缓存多少个数？（2）只缓存潜向量时呢？比值是多少？（3）为什么这个比值比 V2 的 1/64 大？
::: answer 看答案
（1）$2\times16\times128=4{,}096$ 个数。（2）512 个，比值 $512/4{,}096=1/8$。（3）$d_c$ 两者都是 512，但 V2-Lite 的头少，MHA 本来就只存 4,096 个数，压到 512 只省到 1/8。$d_c$ 是固定的“速记长度”，头越多、原本的纪要越厚，压缩的比例越大。
:::
::::

::: key 本页要点
- MLA 先把 $\mathbf h_t$ 压成 $d_c=512$ 维的 $\mathbf c_t^{KV}$，再用 $W^{UK}$、$W^{UV}$ 展开成 128 个头各自的键和值。头数不减，缓存只存潜向量：每层 512 个数，是 MHA 的 1/64。
- 等价地说，MLA 要求 MHA 的键值投影 $[W^K;W^V]$ 秩不超过 512（忽略 RMSNorm），而且这个低秩结构是从零训练出来的，不是事后压缩。
- 两个遗留问题：展开的运算成本（§3）和 RoPE（§4、§5）。
:::

# 矩阵吸收 | 结合律让缓存不必展开：逐行推导、形状与每 token 的算力账 | 30

**问题。** 上一页留下的第一个问题：推理时缓存里只有潜向量 $\mathbf c_j^{KV}$，而注意力要的是每个头的键 $\mathbf k^C_{j,i}$ 和值 $\mathbf v^C_{j,i}$。最直接的做法是每解码一步，都把缓存里 $T$ 个潜向量用 $W^{UK}_i$、$W^{UV}_i$ 各展开一遍。这要多少运算？每个潜向量展开成 128 个头的键和值是 $[16384,512]$ 乘 $[512]$ 两次，约 $2\times2\times16{,}384\times512\approx3.4\times10^7$ 次运算；缓存里有 4096 个 token，一层就是约 $1.4\times10^{11}$ 次，**每生成一个 token、每一层**都要来一遍。这比把展开后的键值直接存下来贵了几百倍，而后者又把缓存变回了 MHA 的大小。

论文的回答是一句话：推理时 $W^{UK}$ 可以被吸收进查询一侧，$W^{UV}$ 可以被吸收进 $W^O$，“we even do not need to compute keys and values out for attention” [@dsv2]。这一页把这句话拆开，一行一行地推。

::: pre 结合律、转置与分块 | 读过 M01 §3、§4、§6 的可以跳过
三件工具。**结合律**：$(AB)C=A(BC)$，结果相同，但先乘哪一对决定了运算量（[M01 §6](../math/m01-vectors-matrices.html#p6)）。**转置规则**：$(AB)^\top=B^\top A^\top$，一个行向量乘矩阵 $\mathbf x^\top W$ 等于 $(W^\top\mathbf x)^\top$（[M01 §3](../math/m01-vectors-matrices.html#p3)）。**分块**：$W^O$ 按列切成 $W^O_i$ 时，“拼接各头输出再乘 $W^O$”等于 $\sum_iW^O_i\mathbf o_{t,i}$（[M01 §4](../math/m01-vectors-matrices.html#p4) 的式 M1.14）。矩阵乘法 $[m,k]\times[k,n]$ 的运算量约为 $2mnk$（[M01 §5](../math/m01-vectors-matrices.html#p5)）。[M02 §7](../math/m02-rank-svd-lowrank.html#p7) 会从低秩分解的角度把“结合律与吸收”再讲一遍。
:::

## 键这一侧：把 $W^{UK}_i$ 挪到查询上

先只看打分的“内容部分”，暂时不管 RoPE（§4、§5 再处理）。第 $i$ 个头里，当前 token $t$ 的查询 $\mathbf q^C_{t,i}\in\R^{d_h}$ 与历史 token $j$ 的键打分。每一行右边写着这一行的形状和理由：

::: deriv 键一侧的吸收（一个头，一对位置 t、j）
$s_{t,j,i}=(\mathbf q^C_{t,i})^\top\mathbf k^C_{j,i}$ || 内容部分的分数；$[1,128]\times[128,1]\to$ 一个数
$=(\mathbf q^C_{t,i})^\top\big(W^{UK}_i\mathbf c^{KV}_j\big)$ || 代入式 (9.7)；括号里是 $[128,512]\times[512,1]\to[128,1]$，要对每个 $j$ 做一次
$=\big((\mathbf q^C_{t,i})^\top W^{UK}_i\big)\,\mathbf c^{KV}_j$ || 结合律；新括号是 $[1,128]\times[128,512]\to[1,512]$，里面没有 $j$
$=\big((W^{UK}_i)^\top\mathbf q^C_{t,i}\big)^\top\mathbf c^{KV}_j$ || 转置规则 $\mathbf x^\top W=(W^\top\mathbf x)^\top$；$(W^{UK}_i)^\top$ 是 $[512,128]$
$=\tilde{\mathbf q}_{t,i}^{\,\top}\,\mathbf c^{KV}_j$ || 定义 $\tilde{\mathbf q}_{t,i}:=(W^{UK}_i)^\top\mathbf q^C_{t,i}\in\R^{512}$；最后是 $[1,512]\times[512,1]\to$ 一个数
:::

::: eq 吸收后的查询与打分 | (9.9)
$$\tilde{\mathbf q}_{t,i}=(W^{UK}_i)^\top\mathbf q^C_{t,i}\in\R^{d_c},\qquad (\mathbf q^C_{t,i})^\top\mathbf k^C_{j,i}=\tilde{\mathbf q}_{t,i}^{\,\top}\,\mathbf c^{KV}_j$$
$\tilde{\mathbf q}_{t,i}$（读作“q 波浪”，课程记号）只依赖当前 token 和头 $i$，与历史位置 $j$ 无关，所以每解码一步、每个头只算一次，然后直接拿去和缓存里的 $T$ 个潜向量做点积。
:::

两条路算出的是同一个数，差别在于**哪一边被乘了 $W^{UK}_i$**。原来是 $T$ 个缓存向量各乘一次，现在是 1 个查询乘一次。这就是**矩阵吸收**（weight absorption）：把 $W^{UK}_i$ 从键那边挪到了查询这边，好像查询把这个矩阵“吸”进了自己。

## 值这一侧：把 $W^{UV}_i$ 挪到输出上

打分之后，第 $i$ 个头对值加权求和。记 softmax 之后的权重为 $p_{t,j,i}$（非负，对 $j$ 求和为 1）：

::: deriv 值一侧的吸收
$\mathbf o_{t,i}=\sum_{j=1}^{t}p_{t,j,i}\,\mathbf v^C_{j,i}$ || 式 (9.3) 的加权求和；每个 $\mathbf v^C_{j,i}$ 是 $[128]$
$=\sum_{j=1}^{t}p_{t,j,i}\,W^{UV}_i\mathbf c^{KV}_j$ || 代入式 (9.7)；每项要先做 $[128,512]\times[512]$
$=W^{UV}_i\Big(\sum_{j=1}^{t}p_{t,j,i}\,\mathbf c^{KV}_j\Big)$ || $W^{UV}_i$ 与 $j$ 无关，矩阵乘法对加法可分配，把它提到求和号外
$=W^{UV}_i\,\tilde{\mathbf o}_{t,i}$ || 定义 $\tilde{\mathbf o}_{t,i}:=\sum_jp_{t,j,i}\mathbf c^{KV}_j\in\R^{512}$，潜向量的加权平均；$[128,512]\times[512]\to[128]$
$\mathbf u_t=\sum_{i=1}^{n_h}W^O_i\mathbf o_{t,i}=\sum_{i=1}^{n_h}\big(W^O_iW^{UV}_i\big)\tilde{\mathbf o}_{t,i}$ || 式 (9.3) 的 $W^O$ 按列分块（M1.14），再用结合律；$W^O_iW^{UV}_i$ 是 $[5120,512]$
:::

::: eq 吸收后的输出 | (9.10)
$$\tilde{\mathbf o}_{t,i}=\sum_{j=1}^{t}p_{t,j,i}\,\mathbf c^{KV}_j\in\R^{d_c},\qquad \mathbf u_t=\sum_{i=1}^{n_h}W^O_i\,W^{UV}_i\,\tilde{\mathbf o}_{t,i}$$
每个头先在“潜空间”里对缓存做加权平均，得到一个 512 维的 $\tilde{\mathbf o}_{t,i}$，最后才乘一次 $W^{UV}_i$ 回到 128 维、乘 $W^O_i$ 回到 5120 维。“$W^{UV}$ 被吸收进 $W^O$”说的就是 $W^O_iW^{UV}_i$ 这一对。
:::

合起来看，吸收之后缓存里的 $\mathbf c^{KV}_j$ 被**原样读了两次**：一次和 $\tilde{\mathbf q}_{t,i}$ 点积打分，一次被加权求和。键 $\mathbf k^C_{j,i}$ 和值 $\mathbf v^C_{j,i}$ 从头到尾没有出现。

::: ex 用小数字验证两条路相等
取一个头，$d_h=2$、$d_c=3$，缓存里两个 token：

$$W^{UK}_i=\begin{pmatrix}1&0&2\\0&1&-1\end{pmatrix},\quad W^{UV}_i=\begin{pmatrix}1&1&0\\0&0&1\end{pmatrix},\quad \mathbf c_1=\begin{pmatrix}1\\1\\0\end{pmatrix},\quad \mathbf c_2=\begin{pmatrix}0\\2\\1\end{pmatrix},\quad \mathbf q^C=\begin{pmatrix}1\\2\end{pmatrix}.$$

**先展开再打分。** $\mathbf k_1=W^{UK}_i\mathbf c_1=(1,1)$，$\mathbf k_2=W^{UK}_i\mathbf c_2=(0+0+2,\ 0+2-1)=(2,1)$。分数 $\mathbf q^\top\mathbf k_1=1+2=3$，$\mathbf q^\top\mathbf k_2=2+2=4$。

**先吸收。** $\tilde{\mathbf q}=(W^{UK}_i)^\top\mathbf q=(1\cdot1+0\cdot2,\ 0\cdot1+1\cdot2,\ 2\cdot1-1\cdot2)=(1,2,0)$。分数 $\tilde{\mathbf q}^\top\mathbf c_1=1+2+0=3$，$\tilde{\mathbf q}^\top\mathbf c_2=0+4+0=4$。一致。

**值一侧。** 设 softmax 后的权重是 $p=(0.25,0.75)$。先展开：$\mathbf v_1=(2,0)$，$\mathbf v_2=(2,1)$，$\mathbf o=0.25(2,0)+0.75(2,1)=(2,\ 0.75)$。先吸收：$\tilde{\mathbf o}=0.25(1,1,0)+0.75(0,2,1)=(0.25,\ 1.75,\ 0.75)$，$W^{UV}_i\tilde{\mathbf o}=(0.25+1.75,\ 0.75)=(2,\ 0.75)$。一致。
:::

## 论文的两种说法：吸收进 $W^Q$，还是 $W^{UQ}$？

论文正文说 $W^{UK}$ 可以被吸收进 $W^Q$；附录 C 说得更准确：依据矩阵乘法的**结合律**，把 $W^{UK}$ 吸收进 $W^{UQ}$ [@dsv2]。区别在于 V2 的查询本身也经过了一次压缩（§6）：$\mathbf q^C_{t,i}=W^{UQ}_i\mathbf c^Q_t$，其中 $\mathbf c^Q_t\in\R^{1536}$，$W^{UQ}_i\in\R^{128\times1536}$，V2 里并没有一个叫 $W^Q$ 的矩阵。代进 (9.9)：

$$\tilde{\mathbf q}_{t,i}=(W^{UK}_i)^\top W^{UQ}_i\,\mathbf c^Q_t .$$

“吸收”在字面上可以理解为：把 $(W^{UK}_i)^\top W^{UQ}_i$ 预先乘成一个 $[512,1536]$ 的矩阵，加载模型时算一次。可是算一下账就会发现，这样做反而更贵。

::: ex 预先乘好，还是改变乘法顺序？
每层 128 个头，维度取 V2 的值。

**参数。** 分开存：$W^{UQ}$ 的内容部分 $128\times128\times1536=25{,}165{,}824$，加上 $W^{UK}$ 的 $8{,}388{,}608$，共 $33{,}554{,}432$。预乘：128 个 $[512,1536]$，共 $128\times512\times1536=100{,}663{,}296$，是分开存的 3 倍。

**每 token 的运算。** 预乘路线：$2\times1536\times128\times512=201{,}326{,}592$。先乘 $W^{UQ}$ 得到 $\mathbf q^C$、再乘 $(W^{UK}_i)^\top$：$2\times1536\times128\times128+2\times128\times128\times512=50{,}331{,}648+16{,}777{,}216=67{,}108{,}864$，是预乘的 1/3。

输出一侧也一样：预乘 $W^O_iW^{UV}_i$ 得到 $[5120,\,128\times512]$，共 $335{,}544{,}320$ 个参数；分开存 $W^O$ 与 $W^{UV}$ 只有 $83{,}886{,}080+8{,}388{,}608=92{,}274{,}688$ 个。

原因是中间维 $d_h=128$ 比两端的 $d_c=512$、$d_c'=1536$ 小得多：一个“中间维是 128”的乘积 $[512,128]\times[128,1536]$，显式乘出来会从秩 128 的瘦结构膨胀成一整块。精确的条件是比较一个头的参数（也就是每 token 运算的一半）：预乘是 $d_cd_c'$，分开存是 $d_h(d_c+d_c')$，所以预乘更贵当且仅当 $d_cd_c'>d_h(d_c+d_c')$，即 $\frac1{d_h}>\frac1{d_c}+\frac1{d_c'}$。V2 里左边 $\frac1{128}\approx0.0078$，右边 $\frac1{512}+\frac1{1536}\approx0.0026$，差 3 倍，正是上面的 3 倍。只说“中间维比两端小”还不够：若两端都是 200，$\frac1{128}<\frac1{200}+\frac1{200}$，预乘反而更省。所以工程上的“吸收”是**改变乘法的结合顺序**，让缓存不必展开，而不是真的把矩阵预先乘成一个。
:::

::: fig figures/l09-absorb-order.svg | 图 9.5 “吸收”的两种做法：预乘成一整块，还是改变乘法顺序（一个头，V2 维度，矩形按比例画） | wide scroll
**比较上下两张图里矩形的面积：上面预乘出来的 $[512\times1536]$ 整块，面积是下面两条瘦色块之和的 3 倍。** 面积就是参数个数，也正比于每个 token 乘一次的运算量。下半部分的虚线框是两条色块相乘会得到的那一块，改变顺序的做法从不把它算出来：先用蓝色的 $W^{UQ}_i$ 把 1536 维压成 128 维的 $\mathbf q^C_{t,i}$，再用紫色的 $(W^{UK}_i)^\top$ 展开成 512 维的 $\tilde{\mathbf q}_{t,i}$。数字由上面的例题算出，维度取自 DeepSeek-V2 [@dsv2]。
:::

::: note 论文 v5 改掉的一句话
DeepSeek-V2 论文里写 MLA 全部公式的那个附录（v1、v2 编号为附录 B，v3 起因新增 V2-Lite 附录而顺延为附录 C），在 v1 到 v4 里的最后一句说，这个优化只和模型参数有关，“it can be completed offline at once”，暗示“离线预乘”；2024 年 6 月的 v5 把它改成了“we do not need to compute keys and values out for each query” [@dsv2]。论文没有说明改动的原因。笔者推测，这正是因为上面算出来的账：预乘并不划算，真正起作用的是改变结合顺序，新的说法更准确。
:::

## 每解码一个 token 的算力账

现在可以把第一段的问题算清楚。固定一层，缓存里有 $T$ 个 token，要为 1 个新 token 算注意力。比较三种算法：① 每步把缓存重新展开；② 把展开后的键值直接存下来（缓存变回 MHA 那么大）；③ 吸收。为了和后文一致，这里把 §5 要讲的 64 维位置部分也算进去：每个头的打分向量其实是 $d_h+d_h^R=192$ 维，其中 64 维的位置键已经在缓存里，不需要展开。三种算法都要做的部分（例如算出 $\mathbf q^C_{t,i}$、$\mathbf c^{KV}_t$ 和最后乘 $W^O$）不计入。只有一项容易漏掉：② 要把**新 token 自己**的潜向量展开一次，才能把它的键值存进缓存，这一项只有 ② 要付（① 在展开全部 $T$ 个潜向量时已经包含了它，③ 根本不展开），所以算进 ②。

::: eq 一层、一个新 token、缓存 T 个 token 时的注意力运算量 | (9.11)
$$\begin{aligned}
F_{\text{①}}(T)&=\underbrace{2T\,d_c\cdot2n_hd_h}_{\text{展开 }T\text{ 个 }\mathbf k,\mathbf v}+\underbrace{2n_h(d_h+d_h^R)\,T}_{\text{打分}}+\underbrace{2n_hd_h\,T}_{\text{加权求和}}\\
F_{\text{②}}(T)&=\underbrace{2d_c\cdot2n_hd_h}_{\text{展开新 token 的 }\mathbf k,\mathbf v}+2n_h(d_h+d_h^R)\,T+2n_hd_h\,T\\
F_{\text{③}}(T)&=\underbrace{2n_hd_hd_c}_{\text{吸收 }W^{UK}}+\underbrace{2n_hd_cd_h}_{\text{乘 }W^{UV}}+\underbrace{2n_h(d_c+d_h^R)\,T}_{\text{与 576 维缓存打分}}+\underbrace{2n_hd_c\,T}_{\text{对 512 维潜向量加权}}
\end{aligned}$$
代入 V2 的 $n_h=d_h=128$、$d_h^R=64$、$d_c=512$：$F_{\text{①}}=33{,}636{,}352\,T$，$F_{\text{②}}=33{,}554{,}432+81{,}920\,T$，$F_{\text{③}}=33{,}554{,}432+278{,}528\,T$。② 与 ③ 的固定成本恰好相同（都是 $4n_hd_hd_c$），所以 $F_{\text{③}}-F_{\text{②}}=196{,}608\,T$，两者之比随 $T$ 增大而趋向 $278{,}528/81{,}920=3.4$。一个检验：$T=1$ 时缓存里只有新 token 自己，① 与 ② 做的是同一件事，公式也给出 $F_{\text{①}}(1)=F_{\text{②}}(1)=33{,}636{,}352$。
:::

::: fig figures/l09-absorb-graph.svg | 图 9.6 一个头在一次解码步里的计算图：每步重新展开（①）与吸收（③） | wide scroll
**看红色的两条箭头：它们在 ① 里每一步都要对缓存里的全部 $T$ 个潜向量各做一次，而在 ③ 里对应的乘法只对 1 个查询、1 个输出各做一次。** 方框里标的是张量形状（$T$ 为缓存长度）。③ 中琥珀色框的缓存 $C$ 被直接读两次：一次与 $\tilde{\mathbf q}_{t,i}$ 打分，一次被加权平均成 $\tilde{\mathbf o}_{t,i}$。右侧是 128 个头合计的每层运算量，$T=4096$ 时 ③ 约为 ① 的 1/117。图中略去了 64 维位置部分的连线，但运算量里算进了它。维度取自 DeepSeek-V2 [@dsv2]。
:::

::: ex 三种算法在 V2 上的账单（一层，一个新 token）
$T=4096$：

- ① 每步重新展开：$33{,}636{,}352\times4096\approx1.378\times10^{11}$，即 137.8 GFLOPs。
- ② 缓存展开后的键值：$33{,}554{,}432+81{,}920\times4096\approx3.69\times10^{8}$，即 0.369 GFLOPs（其中展开新 token 的 0.034 GFLOPs 是固定成本）。
- ③ 吸收：$33{,}554{,}432+278{,}528\times4096\approx1.174\times10^{9}$，即 1.174 GFLOPs。

③ 是 ① 的 1/117；但 ③ 是 ② 的 3.2 倍。$T=131{,}072$（128K）时三者分别是 4,409、10.77、36.54 GFLOPs，③/② 为 3.39，趋近 $278{,}528/81{,}920=3.4$。

再看每一步要从显存读的缓存。① 和 ③ 读的都是每 token 576 个数（512 维潜向量加 64 维位置键），bf16 下 1,152 字节；② 要读 128 个头的键（每头 192 维）和值（每头 128 维），$2\times128\times(192+128)=81{,}920$ 字节，是前者的 71 倍。这里按每头存完整的 192 维键计，Hugging Face 参考实现就是这样存的（§6）；若把所有头共享的 64 维位置键只存一份，是 $2\times(128\times256+64)=65{,}664$ 字节，约 57 倍，结论不变。

把 60 层加起来，$T=4096$ 时 ③ 每生成一个 token 的注意力运算约 70.5 GFLOPs，其中只随 $T$ 增长的部分（打分与加权求和）是 $278{,}528\times4096\times60\approx68.5$ GFLOPs。对照：按“每 token 前向约 $2N$”估算，V2 的 210 亿激活参数对应约 42 GFLOPs 的矩阵乘法（吸收时乘 $W^{UK}$、$W^{UV}$ 的固定部分已含在里面）。也就是说，4K 上下文时，MLA 解码的注意力运算已经超过了全部权重矩阵的运算。这是笔者按上面的公式算的，不是论文给的数字。
:::

::: fig figures/l09-decode-cost.svg | 图 9.7 解码一步的运算量与读缓存字节数，随缓存长度 T 变化（一层，V2 维度） | wide scroll
**看两张图里蓝（③）与灰（②）的相对位置：左图里吸收比缓存展开后的键值多算 3.2～3.4 倍（$T$ 越长越接近 3.4），右图里却少读 71 倍。** 红线（①）在左图里高出两个数量级，这就是不吸收的代价。三条线都由式 (9.11) 逐点计算；双对数坐标下斜率为 1 表示与 $T$ 成正比，② 和 ③ 在 $T$ 很小时略微弯曲，是同一笔固定成本 33.55M（② 展开新 token 一次，③ 吸收与乘 $W^{UV}$ 各一次）在起作用。
:::

::: key 吸收换来了什么，付出了什么
吸收不省运算，**它用运算换字节**。和“把展开后的键值存下来”（②）相比，吸收（③）每个缓存 token 多做约 3.4 倍的运算，但每个缓存 token 只需存、只需读 576 个数，而不是 40,960 个。解码受带宽限制（第 4 课），多出来的运算原本就闲着，少读的字节才是真正的收益。§7 会用屋顶线模型把这句话量化。
:::

::: warn 什么时候吸收不划算
吸收的收益来自“查询只变一次、缓存被反复使用”。缓存越长越划算；$T=1$ 时，$F_{\text{③}}(1)=33{,}832{,}960$ 反而比 $F_{\text{①}}(1)=33{,}636{,}352$ 多 0.6%。反过来，当很多个查询要读同一批键值时（例如训练和预填充，一次处理成千上万个 token），把潜向量展开一次、分摊给所有查询更便宜。§7 会把这两种情况分别称为 MLA 的“MQA 模式”和“MHA 模式”。

另外，吸收是实数运算下的恒等变形；计算机里的浮点加法不严格满足结合律，两种顺序在 bf16 下会有末位的差别（累加顺序怎样影响误差，见 [M08 §5](../math/m08-floating-point.html#p5)）。模型是按哪种顺序训练的，推理时就尽量用数值上接近的顺序，这一点在工程上需要验证，论文没有讨论。
:::

::: ana 类比的延伸：读速记的眼光
**映射。** 缓存里只有速记（$\mathbf c^{KV}_j$）。①相当于每次开会都把所有速记翻译回全文再读；吸收相当于读者事先把自己的问题也翻译成速记的“语言”（$\tilde{\mathbf q}_{t,i}$），然后直接在速记上比对，读完再把结论翻回来（$W^{UV}_i$、$W^O_i$）。

**收益。** 它说明了为什么吸收的成本与缓存长度几乎无关：翻译的是一个问题，而不是整本纪要。

**失效的地方。** 用速记的语言比对，每条比对要处理 512 个符号，比读全文里属于自己那一栏的 128 个字还多，所以每条比对更慢（3.4 倍的运算）。另外，这种“直接读速记”只在翻译规则固定时才成立；下一页会看到，一旦翻译规则随位置变化（RoPE），这套眼光就失效了。
:::

:::: quiz 小练习：吸收的形状
在 V2 里写出下列每个量的形状：$\mathbf q^C_{t,i}$、$W^{UK}_i$、$\tilde{\mathbf q}_{t,i}$、缓存矩阵 $C=[(\mathbf c_1^{KV})^\top;\dots;(\mathbf c_T^{KV})^\top]$（每个潜向量占一行）、分数向量 $C\tilde{\mathbf q}_{t,i}$、$\tilde{\mathbf o}_{t,i}$、$W^O_iW^{UV}_i$。
::: answer 看答案
$\mathbf q^C_{t,i}$：$[128]$；$W^{UK}_i$：$[128,512]$；$\tilde{\mathbf q}_{t,i}=(W^{UK}_i)^\top\mathbf q^C_{t,i}$：$[512,128]\times[128]\to[512]$；$C$：$[T,512]$；$C\tilde{\mathbf q}_{t,i}$：$[T,512]\times[512]\to[T]$；$\tilde{\mathbf o}_{t,i}=C^\top\mathbf p$：$[512,T]\times[T]\to[512]$；$W^O_iW^{UV}_i$：$[5120,128]\times[128,512]\to[5120,512]$。
:::
::::

::: key 本页要点
- 键一侧：$(\mathbf q^C_{t,i})^\top W^{UK}_i\mathbf c^{KV}_j=\tilde{\mathbf q}_{t,i}^\top\mathbf c^{KV}_j$，$\tilde{\mathbf q}_{t,i}=(W^{UK}_i)^\top\mathbf q^C_{t,i}\in\R^{512}$ 每步只算一次。值一侧：$\mathbf o_{t,i}=W^{UV}_i\sum_jp_{t,j,i}\mathbf c^{KV}_j$，先在潜空间加权平均。缓存从不展开。
- 工程上的“吸收”是改变结合顺序，不是把矩阵预先乘好；因为 $\frac1{d_h}>\frac1{d_c}+\frac1{d_c'}$（中间维 128 远小于两端的 512、1536），预乘会让参数和运算都变成 3 倍。
- V2 一层、$T=4096$：每步重新展开 137.8 GFLOPs，吸收 1.174 GFLOPs，缓存展开后的键值 0.369 GFLOPs；吸收比后者多算约 3.2 倍（$T$ 越大越接近 3.4 倍）、少读 71 倍字节。MLA 用算力换带宽。
:::

# RoPE 与吸收的冲突 | 一个随位置转动的矩阵夹在中间 | 20

**问题。** V2 沿用 DeepSeek 67B 的做法（LLaMA 也是如此），用 RoPE 给注意力注入位置信息（第 5 课）[@dsv2]。RoPE 的做法是把每个头的查询和键按各自的位置**旋转**一个角度，旋转之后的点积只依赖两个位置之差。如果照搬到上一页的 MLA 上，就是旋转从潜向量展开出来的键 $\mathbf k^C_{j,i}$。这一页说明，这样一转，上一页的吸收就做不成了。

::: pre RoPE 的三行回顾 | 学过第 5 课的可以跳过
RoPE 把 $d_h$ 维向量的分量两两配对，每一对看成平面上的一个点，按位置 $t$ 旋转角度 $t\theta_k$（不同的对用不同的频率 $\theta_k$）。整体写成一个块对角的正交矩阵 $\mathbf R_{\Theta,t}\in\R^{d_h\times d_h}$，旋转后的查询和键是 $\mathbf R_{\Theta,t}\mathbf q_t$、$\mathbf R_{\Theta,j}\mathbf k_j$。关键性质是：两次旋转的“相对量”只取决于位置差，
$$\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,j}=\mathbf R_{\Theta,j-t}.$$
RoFormer 论文的式 (16) 正是这个恒等式 [@roformer]。二维的情形就是 $\mathbf R(\alpha)^\top\mathbf R(\beta)=\mathbf R(\beta-\alpha)$：先倒转 $\alpha$ 再正转 $\beta$，等于转 $\beta-\alpha$。旋转与复数的关系见 [M03](../math/m03-rotation-complex.html)，这个相对角恒等式本身在 [M03 §5](../math/m03-rotation-complex.html#p5)，RoPE 的完整推导见[第 5 课 §2–§3](lesson-05-modern-block.html#p2)；[第 5 课 §8](lesson-05-modern-block.html#p8) 已经用一个更小的例子预告过本页的冲突。
:::

::: ana 类比的延伸：随时刻变化的翻译规则
**映射。** §3 的吸收，相当于读者把自己的问题一次性翻译成速记的“语言”（$\tilde{\mathbf q}_{t,i}$），这份译文对整本速记通用。RoPE 要求每条纪要的“标签”按它被记下的时刻 $j$ 转一个角度，读者的问题也按现在的时刻 $t$ 转一个角度；比对时真正起作用的是两者的时间差 $j-t$。于是译文不再通用：对第 $j$ 条速记，读者得先把问题按 $j-t$ 转过，再翻译一次。

**收益。** 它说明了为什么吸收会失效：翻译的次数从“每步 1 次”变回了“每步 $T$ 次”，和把整本速记展开一遍一样贵。

**失效的地方。** 类比会让人以为，只要“规则因条而异”就没救了。其实不然：假如旋转能在 512 维的速记这一侧做，每条速记在记下时按自己的时刻转好再存进缓存，读者的译文照样通用。真正的障碍是旋转发生在每个头 128 维的空间里，夹在查询和 $W^{UK}_i$ 之间，挪不到速记一侧（下面会说明原因：矩阵乘法不可交换，而 128 个头又共用同一份速记）。§5 的解耦 RoPE，正是让一小段另外的向量走通“先转好、再缓存”这条路。
:::

下面的推导要用到 RoPE 的相对位置性质。二维时它就是三角函数的和角公式，值得亲手推一遍：

::: deriv 二维旋转的相对性：R(α)ᵀR(β) = R(β − α)
$\mathbf R(\alpha)^\top=\mathbf R(-\alpha)$ || 转置交换两个非对角元 $\pm\sin\alpha$；而 $\cos$ 是偶函数、$\sin$ 是奇函数
$[\mathbf R(-\alpha)\mathbf R(\beta)]_{11}=\cos\alpha\cos\beta+\sin\alpha\sin\beta=\cos(\beta-\alpha)$ || 第一行乘第一列；和角公式
$[\mathbf R(-\alpha)\mathbf R(\beta)]_{21}=-\sin\alpha\cos\beta+\cos\alpha\sin\beta=\sin(\beta-\alpha)$ || 第二行乘第一列；和角公式
$\mathbf R(-\alpha)\mathbf R(\beta)=\mathbf R(\beta-\alpha)$ || 另外两格同理，分别是 $-\sin(\beta-\alpha)$ 与 $\cos(\beta-\alpha)$
$\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,j}=\mathbf R_{\Theta,j-t}$ || RoPE 的矩阵是若干个二维旋转沿对角线排开（第 $k$ 对的角度 $t\theta_k$），逐块套用上一行
:::

## 推一遍：旋转矩阵卡在哪里

假设对内容部分直接施加 RoPE：查询变成 $\mathbf R_t\mathbf q^C_{t,i}$，键变成 $\mathbf R_j\mathbf k^C_{j,i}=\mathbf R_jW^{UK}_i\mathbf c_j$（这里把 $\mathbf R_{\Theta,t}$ 简写成 $\mathbf R_t$，$\mathbf c^{KV}_j$ 简写成 $\mathbf c_j$）。照上一页的办法推：

::: deriv 给键加上 RoPE 之后的打分
$s_{t,j,i}=\big(\mathbf R_t\mathbf q^C_{t,i}\big)^\top\big(\mathbf R_jW^{UK}_i\mathbf c_j\big)$ || 查询和键各自按自己的位置旋转
$=(\mathbf q^C_{t,i})^\top\mathbf R_t^\top\mathbf R_j\,W^{UK}_i\,\mathbf c_j$ || 转置规则 $(\mathbf R_t\mathbf q)^\top=\mathbf q^\top\mathbf R_t^\top$，结合律
$=(\mathbf q^C_{t,i})^\top\,\mathbf R_{j-t}\,W^{UK}_i\,\mathbf c_j$ || RoPE 的相对位置性质 $\mathbf R_t^\top\mathbf R_j=\mathbf R_{j-t}$
$=\big[(\mathbf R_{j-t}W^{UK}_i)^\top\mathbf q^C_{t,i}\big]^\top\mathbf c_j$ || 想照 (9.9) 那样把前面合成一个“吸收查询”
$\tilde{\mathbf q}_{t,i}^{(j)}:=(W^{UK}_i)^\top\,\mathbf R_{j-t}^\top\,\mathbf q^C_{t,i}\quad\text{随 }j\text{ 变}$ || 方括号里含 $\mathbf R_{j-t}$，每个历史位置 $j$ 一个不同的矩阵
:::

::: eq RoPE 让“吸收查询”依赖于历史位置 | (9.12)
$$(\mathbf q^C_{t,i})^\top\,\underbrace{\mathbf R_{j-t}\,W^{UK}_i}_{\text{随 }j\text{ 变}}\,\mathbf c_j,\qquad \tilde{\mathbf q}^{(j)}_{t,i}=(W^{UK}_i)^\top\mathbf R_{j-t}^\top\,\mathbf q^C_{t,i}$$
没有 RoPE 时，(9.9) 的 $\tilde{\mathbf q}_{t,i}$ 对所有 $j$ 通用；有了 RoPE，每个历史位置都要一个自己的 $\tilde{\mathbf q}^{(j)}_{t,i}$，缓存里有 $T$ 个 token，就要算 $T$ 个。
:::

算 $T$ 个 $\tilde{\mathbf q}^{(j)}_{t,i}$，每个都是 $[512,128]$ 乘 $[128]$，这和把 $T$ 个潜向量各展开成键的运算量一样，后者每个是 $[128,512]$ 乘 $[512]$。键这一侧，吸收的好处全部消失了。值不旋转，值一侧照 (9.10) 吸收仍然成立，但每步重新展开 $T$ 个键的那一半开销是躲不掉的。

论文对这件事的解释是：RoPE 对查询和键都是位置敏感的，旋转矩阵会“lie between $W^{Q}$ and $W^{UK}$”，而“matrix multiplication does not obey a commutative law”，于是推理时必须为所有前缀 token 重新计算键，严重拖慢推理 [@dsv2]。上面的推导把这句话说得更精确一点：夹在中间的是**相对**旋转 $\mathbf R_{j-t}$，它同时依赖当前位置 $t$ 和被查询的位置 $j$。

为什么说问题出在“不满足交换律”？设想把键的旋转挪到 $W^{UK}_i$ 的右边：假如有一个作用在 $d_c$ 维潜向量上的矩阵 $\mathbf S_j$，使 $\mathbf R_jW^{UK}_i=W^{UK}_i\mathbf S_j$，那么打分就是 $(\mathbf R_t\mathbf q^C_{t,i})^\top W^{UK}_i\mathbf S_j\mathbf c_j=\big((W^{UK}_i)^\top\mathbf R_t\mathbf q^C_{t,i}\big)^\top(\mathbf S_j\mathbf c_j)$：缓存里改存“转好的”潜向量 $\mathbf S_j\mathbf c_j$，查询一侧每步仍只算一次，吸收照样成立。难处在于缓存只有一份，同一个 $\mathbf S_j$ 必须对 128 个头**同时**成立。把各头的 $W^{UK}_i$ 上下叠回 $W^{UK}$，这个条件就是：$W^{UK}$ 的列空间（16,384 维空间里的一个 512 维子空间）在每个位置的旋转 $\diag(\mathbf R_j,\dots,\mathbf R_j)$ 之下保持不变。一个不加约束、训练出来的 $W^{UK}$ 没有理由恰好具备这种特殊结构；要它具备，就得在训练时额外限制 $W^{UK}$ 的形状，论文没有走这条路。所以旋转只能留在查询和 $W^{UK}_i$ 之间。

（只看一个头时，这样的矩阵其实找得到：$W^{UK}_i$ 是 $128\times512$ 的，只要它的 128 行线性无关（训练出来的矩阵几乎总是如此），取 $\mathbf S_j=(W^{UK}_i)^{+}\mathbf R_jW^{UK}_i$ 即可，$(\cdot)^+$ 是伪逆，满足 $W^{UK}_i(W^{UK}_i)^+=I$。可它随头 $i$ 而变，等于每个头各存一份 512 维的“转好的潜向量”，缓存比 MHA 还大。所以真正卡住的不是单个头，而是“128 个头共用一份缓存”。）

::: ex 一个 2×2 的反例：同一个查询，三个历史位置，三个不同的“吸收查询”
取最小的设定：每头 $d_h=2$，潜向量只有 $d_c=1$ 维。$W^{UK}_i=\begin{pmatrix}3\\1\end{pmatrix}$，当前 token 在位置 $t=3$，旋转前的查询 $\mathbf q=\begin{pmatrix}1\\2\end{pmatrix}$；缓存里三个历史 token 的潜向量 $c_1=1$、$c_2=2$、$c_3=-1$。旋转矩阵 $\mathbf R(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\ \sin\phi&\cos\phi\end{pmatrix}$。

**没有 RoPE。** $\tilde q=\mathbf q^\top W^{UK}_i=1\times3+2\times1=5$，一个数通吃所有位置：分数 $5c_j$ 依次是 5、10、−5。

**有 RoPE，取每个位置转 $\theta=\pi/2$（方便手算）。** 按 (9.12)，位置 $j$ 的吸收查询是 $\mathbf q^\top\mathbf R\big((j-3)\theta\big)W^{UK}_i$。

- $j=1$：转 $-\pi$，$\mathbf R(-\pi)=-I$，$\mathbf R W^{UK}_i=(-3,-1)$，$\tilde q^{(1)}=-3-2=-5$。
- $j=2$：转 $-\pi/2$，$\mathbf R(-\pi/2)=\begin{pmatrix}0&1\\-1&0\end{pmatrix}$，$\mathbf R W^{UK}_i=(0\cdot3+1\cdot1,\ -1\cdot3+0\cdot1)=(1,-3)$，$\tilde q^{(2)}=1-6=-5$。
- $j=3$：不转，$\tilde q^{(3)}=5$。

分数是 $-5\times1=-5$、$-5\times2=-10$、$5\times(-1)=-5$。三个“吸收查询” $-5,-5,5$ 不全相同，不存在一个对所有 $j$ 都对的 $\tilde q$。

**换成 $\theta=0.5$ rad。** 同样的算法给出 $\tilde q^{(1)}=-1.5058$、$\tilde q^{(2)}=1.9908$、$\tilde q^{(3)}=5$，三个都不同（图 9.8 把这三个旋转后的 $W^{UK}_i$ 画在平面上）。要想得到正确的分数，只能对每个 $j$ 把 $\mathbf R_{j-t}W^{UK}_ic_j$ 重新算出来，也就是重算所有前缀的键。
:::

::: fig figures/l09-rope-breaks.svg | 图 9.8 RoPE 为什么挡住矩阵吸收：乘法链与一个二维数值例子 | wide scroll
**看左下角三支紫色箭头：它们是同一个 $W^{UK}_i=(3,1)$ 按三个不同的相对位置转过的样子，蓝色查询与它们的点积各不相同。** 上方的乘法链里，红框中的 $\mathbf R_{j-t}W^{UK}$ 对每个历史位置都不同，所以无法像没有 RoPE 时那样，事先把它并进查询。右侧表格的数由 $\mathbf R(\phi)(3,1)^\top=(3\cos\phi-\sin\phi,\ 3\sin\phi+\cos\phi)$ 逐项计算。例子是为本课构造的，参数与第 5 课的设定无关。
:::

## 两条退路，都不能接受

吸收做不成时，只剩两种办法。**退路一**：每步把所有前缀的键重新算一遍，即 $\mathbf R_jW^{UK}_i\mathbf c_j$ 对每个 $j$、每个头都算。这相当于上一页算法 ① 展开键的那一半（值不旋转，值一侧仍可吸收），一层 4K 上下文每步单是展开键就要 $2\times4096\times512\times16{,}384\approx68.7$ GFLOPs，是吸收（1.174 GFLOPs）的近 60 倍。论文的原话是“significantly hinder the inference efficiency” [@dsv2]。**退路二**：把旋转后的键存进缓存。这样打分时不用重算，但每个 token 每层要存 128 个头各 128 维的旋转键，再加上值，缓存又回到了 MHA 的大小，MLA 就白做了。

所以真正的问题是：位置信息必须进入打分，而旋转又不能碰到那条要被吸收的路径。下一页的解耦 RoPE 就是把这两件事拆开。

:::: quiz 小练习：换一个 j
在上面的 2×2 例子里（$\theta=\pi/2$），如果还有一个历史 token 在位置 $j=0$、潜向量 $c_0=3$，它的吸收查询和分数是多少？
::: answer 看答案
$j-t=-3$，转 $-3\pi/2$，等于转 $+\pi/2$：$\mathbf R(\pi/2)=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$，$\mathbf R W^{UK}_i=(0\cdot3-1\cdot1,\ 1\cdot3+0\cdot1)=(-1,3)$。吸收查询 $\tilde q^{(0)}=\mathbf q^\top(-1,3)=-1+6=5$，分数 $5\times3=15$。它碰巧与 $j=3$ 的吸收查询相同（纯属巧合：$(-1,3)$ 和 $(3,1)$ 与 $\mathbf q=(1,2)$ 的点积都是 5），但与 $j=1,2$ 的不同，所以仍然没有一个通用的 $\tilde q$。
:::
::::

::: key 本页要点
- 给从潜向量展开的键加 RoPE，打分变成 $(\mathbf q^C)^\top\mathbf R_{j-t}W^{UK}_i\mathbf c_j$：随 $j$ 变的旋转夹在查询和 $W^{UK}_i$ 之间，挪不走（矩阵乘法不可交换，而 128 个头共用一份缓存，找不到一个对所有头都成立的“潜空间旋转”），键一侧的吸收失效。
- 2×2 反例：同一个查询对三个历史位置需要三个不同的“吸收查询”（$\theta=\pi/2$ 时为 −5、−5、5；$\theta=0.5$ 时为 −1.506、1.991、5）。
- 两条退路：每步重算全部前缀的键（算力爆炸），或缓存旋转后的键（缓存回到 MHA）。都不可接受。
:::

# 解耦 RoPE | 位置信息另记一小栏：一个所有头共享的 64 维键 | 25

**问题。** 上一页的结论是：位置信息必须进入打分，而旋转不能碰到要被吸收的那条路径。DeepSeek-V2 的办法是把每个头的查询和键**拆成两段**：一段只管内容、不旋转，照 §3 吸收；另一段只管位置、要旋转，但做得很小，而且旋转后直接缓存。论文称之为**解耦 RoPE**（decoupled RoPE）[@dsv2]。

::: ana 类比的延伸：速记旁边的一小栏
**映射。** §2 说过，位置信息没法写进速记。解耦 RoPE 的做法是在每条速记旁边**另记一小栏**：这一栏用普通文字写，写上发言时刻的“时间戳”（旋转后的 $\mathbf k^R_j$），只有 64 个字，所有读者（头）共看同一栏。读者比对时分两部分：用自己的速记眼光比对内容（吸收后的 $\tilde{\mathbf q}_{t,i}$ 对 $\mathbf c_j$），再用自己的时间问题（$\mathbf q^R_{t,i}$）去看那一小栏。

**收益。** 它说明了为什么缓存只多了一点点：小栏只有 64 个数，而且全体头共享一栏，不是每头一栏。

**失效的地方。** 真实的“时间戳”不是一个数，而是一个 64 维的、由内容决定、再按位置旋转的向量（$\mathbf k^R_j=\mathbf R_jW^{KR}\mathbf h_j$），它同时携带“是什么”和“在哪里”；而且内容那一段的分数本身完全不知道位置，打分里显式的位置信息全部来自这 64 维。
:::

## 定义：多一段旋转的查询，多一个共享的键

论文用额外的多头查询 $\mathbf q^R_{t,i}\in\R^{d_h^R}$ 和一个**共享的**键 $\mathbf k^R_t\in\R^{d_h^R}$ 来承载 RoPE，$d_h^R$ 是解耦查询和键的每头维度 [@dsv2]：

::: eq 解耦的位置查询与位置键（论文式 14–15） | (9.13)
$$[\mathbf q^R_{t,1};\mathbf q^R_{t,2};\dots;\mathbf q^R_{t,n_h}]=\mathbf q^R_t=\operatorname{RoPE}\big(W^{QR}\mathbf c^Q_t\big),\qquad \mathbf k^R_t=\operatorname{RoPE}\big(W^{KR}\mathbf h_t\big)$$
$W^{QR}\in\R^{d_h^Rn_h\times d_c'}$ 从查询的压缩向量 $\mathbf c^Q_t\in\R^{d_c'}$（§6）算出 128 个头各自的 64 维位置查询；$W^{KR}\in\R^{d_h^R\times d}$ 直接从 $\mathbf h_t$ 算出**一个** 64 维位置键，不经过潜向量 $\mathbf c^{KV}_t$。$\operatorname{RoPE}(\cdot)$ 表示按位置 $t$ 施加旋转矩阵 $\mathbf R_{\Theta,t}\in\R^{d_h^R\times d_h^R}$；作用在 $n_hd_h^R$ 维的 $\mathbf q^R_t$ 上时，是对每个头的那一段 64 维各乘一次，即 $\mathbf q^R_{t,i}=\mathbf R_{\Theta,t}W^{QR}_i\mathbf c^Q_t$。
:::

然后把内容段和位置段拼起来，得到每个头 192 维的查询和键，照常做注意力：

::: eq 拼接与注意力（论文式 16–19） | (9.14)
$$\begin{aligned}\mathbf q_{t,i}&=[\mathbf q^C_{t,i};\mathbf q^R_{t,i}],\qquad \mathbf k_{t,i}=[\mathbf k^C_{t,i};\mathbf k^R_t]\\
\mathbf o_{t,i}&=\sum_{j=1}^{t}\softmax_j\!\Big(\frac{\mathbf q_{t,i}^\top\mathbf k_{j,i}}{\sqrt{d_h+d_h^R}}\Big)\mathbf v^C_{j,i},\qquad \mathbf u_t=W^O[\mathbf o_{t,1};\dots;\mathbf o_{t,n_h}]\end{aligned}$$
注意 $\mathbf k_{t,i}$ 的后半段是 $\mathbf k^R_t$，**没有头下标 $i$**：128 个头拼的是同一个位置键。值没有位置段，仍是 $d_h=128$ 维。缩放因子是 $1/\sqrt{d_h+d_h^R}=1/\sqrt{192}$。
:::

这里有三个容易看漏的细节，论文公式里都写着 [@dsv2]：$\mathbf k^R_t$ 直接从 $\mathbf h_t$ 算，不经过 $\mathbf c^{KV}_t$；$\mathbf k^R_t$ 只有一个，被所有头共享（式 17 里没有下标 $i$）；$\mathbf q^R_{t,i}$ 则每个头一个。

## 为什么这样一拆，吸收就又成立了

把打分拆成两段来看：

::: deriv 解耦之后的打分
$\mathbf q_{t,i}^\top\mathbf k_{j,i}=[\mathbf q^C_{t,i};\mathbf q^R_{t,i}]^\top[\mathbf k^C_{j,i};\mathbf k^R_j]$ || 式 (9.14)；两个 192 维向量的点积
$=(\mathbf q^C_{t,i})^\top\mathbf k^C_{j,i}+(\mathbf q^R_{t,i})^\top\mathbf k^R_j$ || 分块点积：拼接向量的点积等于各段点积之和（M01 的式 M1.15）
$=\tilde{\mathbf q}_{t,i}^{\,\top}\mathbf c^{KV}_j+(\mathbf q^R_{t,i})^\top\mathbf k^R_j$ || 内容段里没有旋转，照 (9.9) 吸收
$(\mathbf q^R_{t,i})^\top\mathbf k^R_j=\big(W^{QR}_i\mathbf c^Q_t\big)^\top\mathbf R_{j-t}\big(W^{KR}\mathbf h_j\big)$ || 位置段：$\mathbf R_t^\top\mathbf R_j=\mathbf R_{j-t}$，相对位置照常起作用
:::

::: eq 解耦之后的打分：内容 + 位置 | (9.15)
$$\mathbf q_{t,i}^\top\mathbf k_{j,i}=\underbrace{\tilde{\mathbf q}_{t,i}^{\,\top}\,\mathbf c^{KV}_j}_{\text{内容：吸收，读缓存的 512 维}}+\underbrace{(\mathbf q^R_{t,i})^\top\,\mathbf k^R_j}_{\text{位置：直接读缓存的 64 维}}$$
两段的数据都在缓存里：$\mathbf c^{KV}_j$ 与**已经旋转好的** $\mathbf k^R_j$。
:::

关键在于位置段根本不需要吸收。$\mathbf k^R_j$ 在第 $j$ 步算出来时就按位置 $j$ 旋转好了，直接存进缓存；以后每一步，新 token 的 $\mathbf q^R_{t,i}$ 按自己的位置 $t$ 旋转，再和缓存里的 $\mathbf k^R_j$ 点积，相对位置 $\mathbf R_{j-t}$ 自动出现。§4 的退路二“缓存旋转后的键”之所以不可接受，是因为那要存 128 个头各 128 维；现在要存的只有 64 个数。

::: ex 在 §4 的 2×2 例子上加一段位置
沿用 §4 的内容段：$\mathbf q=(1,2)$，$W^{UK}_i=(3,1)^\top$，$c_1=1$、$c_2=2$、$c_3=-1$，当前位置 $t=3$。内容段不旋转，吸收查询 $\tilde q=5$ 对所有 $j$ 通用，内容分数是 5、10、−5。

再加一段 2 维的位置段，每个位置转 $\theta=\pi/2$。设每个历史 token 旋转前的位置键都是 $(1,0)$，缓存里存的是**转好的**：$j=1$ 转 $\pi/2$ 得 $(0,1)$；$j=2$ 转 $\pi$ 得 $(-1,0)$；$j=3$ 转 $3\pi/2$ 得 $(0,-1)$。当前 token 的位置查询旋转前是 $(1,1)$，按 $t=3$ 转 $3\pi/2$：$\mathbf R(3\pi/2)(1,1)=(0\cdot1+1\cdot1,\ -1\cdot1+0\cdot1)=(1,-1)$。

位置分数：$(1,-1)\cdot(0,1)=-1$，$(1,-1)\cdot(-1,0)=-1$，$(1,-1)\cdot(0,-1)=1$。总分 = 内容 + 位置：$5-1=4$、$10-1=9$、$-5+1=-4$。

整个计算只用到了缓存里的 $c_j$（每个 1 个数）和转好的位置键（每个 2 个数），外加一个通用的 $\tilde q$。可以验证，位置分数只依赖相对位置：按上面推导的最后一行，$j=1$ 的位置分数是 $(1,1)^\top\mathbf R(-\pi)(1,0)^\top=(1,1)\cdot(-1,0)=-1$，与上面一致。
:::

::: fig figures/l09-decoupled-key.svg | 图 9.9 解耦 RoPE：一个头的 192 维查询与键，以及位置键如何在头之间共享 | wide scroll
**看每一行右边那段红框：查询的位置段每头各有一个，而键的位置段 $\mathbf k^R_j$ 在所有头里是同一个向量。** 上半部分：每个头的查询和键都是 128 维内容段加 64 维位置段，分数是两段点积之和，只有内容段参与吸收。下半部分：同一个 token $j$ 在第 1、2、128 个头里的键，紫色内容段各不相同（$W^{UK}_i$ 不同），红色位置段完全相同。右下角：推理时每 token 每层只缓存 $512+64=576$ 个数，同形状的 MHA 要存 32,768 个。上下两部分的色段宽度按维度比例画（右下角的缓存示意不按比例），维度取自 DeepSeek-V2 [@dsv2]。
:::

## 缓存变成 576，缩放因子是 √192

加上位置键，MLA 的缓存从 (9.8) 的 $d_cl$ 变成：

::: eq MLA 每 token 的 KV cache | (9.16)
$$\text{MLA：}\quad (d_c+d_h^R)\,l\ \text{个元素}$$
论文原话：解耦的键也要缓存，所以 DeepSeek-V2 总共需要 $(d_c+d_h^R)l$ 个元素的 KV cache [@dsv2]。V2：$(512+64)\times60=34{,}560$。
:::

::: ex 为什么位置键只要一个
**缓存。** 共享一个位置键：每层每 token $512+64=576$ 个数。如果每个头各一个 64 维位置键：$512+128\times64=8{,}704$ 个，是共享方案的 15.1 倍，是 8 组 GQA（2,048 个）的 4.25 倍。共享让位置段的缓存和 MQA 一样便宜。

**运算。** 每个（查询, 键）对，位置段的点积是 $2\times64=128$ 次运算每头，128 个头共 16,384 次；内容段在不吸收时是 $2\times128\times128=32{,}768$ 次，吸收时是 $2\times128\times512=131{,}072$ 次。

**缩放。** $\sqrt{192}\approx13.856$，$1/\sqrt{192}\approx0.07217$。Hugging Face 参考代码里写作 `softmax_scale = q_head_dim ** (-0.5)`，其中 `q_head_dim = 128 + 64` [@hf-v2]（代码还会再乘一个 YaRN 的修正项 $\mu^2$，[第 5 课 §7](lesson-05-modern-block.html#p7) 的式 (5.23) 已经算过；V2 的 YaRN 设置放回训练流程里讲，在[第 10 课 §6](lesson-10-v2-moe-training.html#p6)）。
:::

::: err 常见误解：解耦 RoPE 的键每个头一个
不是。论文写的是“additional multi-head queries”和“a shared key” [@dsv2]：查询的位置段每头一个，键的位置段只有一个，所有头共享。Hugging Face 代码里对应的投影叫 `kv_a_proj_with_mqa` [@hf-v2]：它一次输出 $512+64=576$ 维，后 64 维就是这个 MQA 式的共享键。代码没有解释这个名字，但“with_mqa”指的应该就是这个所有头共用的位置键（笔者的理解）。
:::

::: warn 这是一个设计选择，不是推论
“位置段所有头共享一个键”是为了缓存便宜而做的选择；每个头仍有自己的位置查询 $\mathbf q^R_{t,i}$，所以不同的头仍然可以对位置有不同的偏好，只是它们都对着同一组位置键提问。$d_h^R$ 取 64（$d_h$ 的一半）也是超参数选择 [@dsv2]；论文没有报告换成别的 $d_h^R$、或者位置键不共享时的消融结果。
:::

::: note 历史注记
DeepSeek-V2 论文附录 A 的贡献者名单特别写道，Huazuo Gao 和 Wangding Zeng 在 MLA 架构的研究中做出了关键创新，并感谢 Jianlin Su 在位置编码上的有益讨论 [@dsv2]。Jianlin Su 正是 RoPE 论文的第一作者 [@roformer]。
:::

:::: quiz 小练习：缩放因子错了会怎样
吸收之后，§7 会看到每个头的查询和键都变成 576 维（512 维内容加 64 维位置）。如果实现者看到 576 维，就按 $1/\sqrt{576}$ 缩放，会发生什么？
::: answer 看答案
吸收只是换了乘法的顺序，$\tilde{\mathbf q}^\top\mathbf c+(\mathbf q^R)^\top\mathbf k^R$ 与原来 192 维的点积数值完全相等，所以缩放必须和训练时一致，用 $1/\sqrt{192}\approx0.07217$。误用 $1/\sqrt{576}\approx0.04167$，所有 logits 都被乘上 $\sqrt{192/576}\approx0.577$，相当于把 softmax 的温度调高到约 1.73 倍，注意力分布变平，模型的输出会系统性地走样。
:::
::::

::: key 本页要点
- 解耦 RoPE：每个头的查询、键 = 128 维内容段（不旋转，可吸收）+ 64 维位置段（旋转）。位置查询 $\mathbf q^R_{t,i}$ 每头一个；位置键 $\mathbf k^R_t=\operatorname{RoPE}(W^{KR}\mathbf h_t)$ 只有一个，所有头共享。
- 打分 = $\tilde{\mathbf q}_{t,i}^\top\mathbf c_j$ + $(\mathbf q^R_{t,i})^\top\mathbf k^R_j$。位置键旋转后直接缓存，不需要吸收。
- 缓存 $(d_c+d_h^R)l$，V2 每层 576 个数、每 token 34,560 个元素；缩放因子 $1/\sqrt{192}$，吸收后也不变。
:::

# 查询压缩与完整的 MLA | 附录 C 的全部公式，按真实维度走一遍 | 20

前面几页已经把 MLA 的零件都讲完了，只剩最后一块：查询也被压缩了。这一页先补上它，然后把论文附录 C 的全部公式放在一起，按 V2 的维度从输入走到输出。

## 查询的低秩压缩

V2 对查询也做了同样的“先压后展”[@dsv2]：

::: eq 查询的低秩压缩（论文式 12–13） | (9.17)
$$\mathbf c^Q_t=W^{DQ}\mathbf h_t,\qquad \mathbf q^C_t=W^{UQ}\mathbf c^Q_t$$
$\mathbf c^Q_t\in\R^{d_c'}$ 是查询的压缩向量，$d_c'\ll d_hn_h$；$W^{DQ}\in\R^{d_c'\times d}$，$W^{UQ}\in\R^{d_hn_h\times d_c'}$。V2 取 $d_c'=1536$ [@dsv2]。§5 的位置查询也从同一个 $\mathbf c^Q_t$ 算出。
:::

它的目的与缓存无关。查询不进缓存（只有当前 token 的查询会被用到），压缩查询当然省不了 KV cache。论文说得很明白：压缩查询是为了减少**训练时的激活显存**，“even if it cannot reduce the KV cache” [@dsv2]。

论文没有说明它具体怎样减少激活显存。下面是一种说得通的机制，属于**笔者推测**，但有一个后来的旁证。训练时，前向传播的中间结果（激活）要留到反向传播时使用（反向传播为什么需要它们，见 [M06 §4](../math/m06-calculus-backprop.html#p4)）。每个 token 每层展开后的查询有 $128\times192=24{,}576$ 个数，要么整份存下来，要么在反向时重算。不压缩时，重算要从 5120 维的 $\mathbf h_t$ 出发：$2\times5120\times24{,}576\approx2.52\times10^8$ 次运算。压缩之后，只需保存 1536 维的 $\mathbf c^Q_t$，反向时用 $W^{UQ}$、$W^{QR}$ 重算查询，只要 $2\times1536\times24{,}576\approx7.55\times10^7$ 次，是前者的 30%。也就是说，压缩之后，“少存、多算”这笔交易里多算的那部分只剩原来的 30%。旁证来自 V3 的技术报告：它明确写着，反向传播时重算所有 RMSNorm 操作和 MLA 的上投影，以免一直保存它们的输出 [@dsv3]。V2 是否也这样做，论文没有写。

::: ex 查询压缩省了多少参数（每层）
不压缩时，查询的内容段和位置段直接从 $\mathbf h_t$ 算：$[W^Q;W^{QR}]$ 把 5120 维映射到 $128\times192$ 维，参数 $5120\times128\times192=125{,}829{,}120$，约 1.26 亿。

压缩后：$W^{DQ}$ 是 $5120\times1536=7{,}864{,}320$；$W^{UQ}$ 与 $W^{QR}$ 合起来把 1536 维映射到 $128\times192$ 维，$1536\times128\times192=37{,}748{,}736$。合计 $45{,}613{,}056$，约 4,560 万，是不压缩时的 36%。

V2 的缩小版 V2-Lite 没有压缩查询，它的查询直接由 $\mathbf h_t$ 投影得到 [@dsv2]。所以论文正文“$W^{UK}$ 被吸收进 $W^Q$”的说法，对 V2-Lite 是字面成立的；对 V2，准确的说法是附录的“吸收进 $W^{UQ}$”（§3）。
:::

## 附录 C：MLA 的全部公式

下面照录论文附录 C 的式 (37)–(47)，只把记号换成本课程的写法（转置统一写成 $\mathbf q^\top$ 的样子，Softmax 写成 $\softmax$）。方框里的两个向量是推理时要缓存的，论文原文用蓝色方框标出 [@dsv2]。

::: eq MLA 完整公式（论文附录 C，式 37–47） | (9.18)
$$\begin{aligned}
\mathbf c^Q_t&=W^{DQ}\mathbf h_t &&(37)\\
[\mathbf q^C_{t,1};\mathbf q^C_{t,2};\dots;\mathbf q^C_{t,n_h}]=\mathbf q^C_t&=W^{UQ}\mathbf c^Q_t &&(38)\\
[\mathbf q^R_{t,1};\mathbf q^R_{t,2};\dots;\mathbf q^R_{t,n_h}]=\mathbf q^R_t&=\operatorname{RoPE}(W^{QR}\mathbf c^Q_t) &&(39)\\
\mathbf q_{t,i}&=[\mathbf q^C_{t,i};\mathbf q^R_{t,i}] &&(40)\\
\boxed{\mathbf c^{KV}_t}&=W^{DKV}\mathbf h_t &&(41)\\
[\mathbf k^C_{t,1};\mathbf k^C_{t,2};\dots;\mathbf k^C_{t,n_h}]=\mathbf k^C_t&=W^{UK}\mathbf c^{KV}_t &&(42)\\
\boxed{\mathbf k^R_t}&=\operatorname{RoPE}(W^{KR}\mathbf h_t) &&(43)\\
\mathbf k_{t,i}&=[\mathbf k^C_{t,i};\mathbf k^R_t] &&(44)\\
[\mathbf v^C_{t,1};\mathbf v^C_{t,2};\dots;\mathbf v^C_{t,n_h}]=\mathbf v^C_t&=W^{UV}\mathbf c^{KV}_t &&(45)\\
\mathbf o_{t,i}&=\sum_{j=1}^{t}\softmax_j\Big(\frac{\mathbf q_{t,i}^\top\mathbf k_{j,i}}{\sqrt{d_h+d_h^R}}\Big)\mathbf v^C_{j,i} &&(46)\\
\mathbf u_t&=W^O[\mathbf o_{t,1};\mathbf o_{t,2};\dots;\mathbf o_{t,n_h}] &&(47)
\end{aligned}$$
右侧括号里是论文的式号。查询一条线（37–40），键值一条线（41–45），注意力与输出（46–47）。
:::

::: fig figures/l09-mla-dataflow.svg | 图 9.10 DeepSeek-V2 的一层 MLA：完整数据流与真实维度（据 DeepSeek-V2 论文图 2 的 MLA 部分重绘） | wide scroll
**看琥珀色粗框：整层里只有 $\mathbf c^{KV}_t$（512 维，RMSNorm 之后）和 $\mathbf k^R_t$（64 维，旋转之后）两处被缓存，其余全部是每一步现算的中间量。** 从上到下三条支路：左边是查询（先压到 1536 维，再展开成 128 个头的内容段和位置段），中间是键值（压到 512 维，再展开成 128 个头的键、值内容段），右边是那个共享的位置键。每个框里标了输出维度和这块矩阵的参数量；整层约 1.49 亿个参数。论文原图从下往上画，这里改成从上往下，并补上了每个张量的维度 [@dsv2]。
:::

## 按维度走一遍

| 步骤 | 公式 | 输出形状 | 参数（每层） |
|---|---|---|---|
| 查询下投影 | $\mathbf c^Q_t=W^{DQ}\mathbf h_t$，再 RMSNorm | $[1536]$ | 7,864,320 |
| 查询内容段 | $\mathbf q^C_t=W^{UQ}\mathbf c^Q_t$ | $[128\times128]$ | 25,165,824 |
| 查询位置段 | $\mathbf q^R_t=\operatorname{RoPE}(W^{QR}\mathbf c^Q_t)$ | $[128\times64]$ | 12,582,912 |
| 键值下投影 | $\mathbf c^{KV}_t=W^{DKV}\mathbf h_t$，再 RMSNorm，**缓存** | $[512]$ | 2,621,440 |
| 位置键 | $\mathbf k^R_t=\operatorname{RoPE}(W^{KR}\mathbf h_t)$，**缓存** | $[64]$ | 327,680 |
| 键内容段 | $\mathbf k^C_t=W^{UK}\mathbf c^{KV}_t$ | $[128\times128]$ | 8,388,608 |
| 值 | $\mathbf v^C_t=W^{UV}\mathbf c^{KV}_t$ | $[128\times128]$ | 8,388,608 |
| 注意力 | 式 (46)，缩放 $1/\sqrt{192}$ | $[128\times128]$ | 0 |
| 输出投影 | $\mathbf u_t=W^O[\dots]$ | $[5120]$ | 83,886,080 |

加上两个 RMSNorm 的权重（1536 + 512 个），一层 MLA 共 $149{,}227{,}520$ 个参数，约 1.49 亿；60 层约 89.5 亿。表中的 RMSNorm 来自论文的一句说明：低秩压缩会影响一层输出的尺度，所以在压缩后的潜向量后面加了 RMSNorm，并在“宽度瓶颈”处乘了额外的缩放因子，以保证训练稳定 [@dsv2]。缩放因子的具体数值，论文和参考代码里都没能核实到。

RMSNorm 的位置对吸收有一个要求。§3 的推导里，和 $\tilde{\mathbf q}_{t,i}$ 点积、被加权平均的 $\mathbf c^{KV}_j$，必须是 $W^{UK}$、$W^{UV}$ 真正乘上去的那个向量，也就是**归一化之后**的潜向量；RMSNorm 对每个 token 各除以一个不同的数，不是线性映射，没法并进 $W^{UK}$。所以缓存里存的应当是归一化之后的 512 维，否则每一步都要先把 $T$ 个缓存行重新归一化一遍。DeepSeek 为 V3 发布的推理代码正是这样做的：写进缓存的是 `kv_norm(kv)`，旋转好的位置键另存一份 [@gh-v3]。论文的式 (41) 不含 RMSNorm，所以只写“缓存 $\mathbf c^{KV}_t$”。

::: ex 输出投影占了一半以上
上表里最大的一块是 $W^O$：$83{,}886{,}080/149{,}227{,}520\approx56\%$。它是 MHA 原有的部件，MLA 没有动它。MLA 改动的是它前面的部分：查询、键、值三组投影合计约 6,530 万个参数；作为对照，同形状的 MHA 仅 $W^Q$、$W^K$、$W^V$ 就是 $3\times16{,}384\times5{,}120\approx2.52$ 亿。
:::

## 读代码时的对照表

Hugging Face 上发布的参考实现把论文的矩阵合并成了几个模块，名字与论文不同 [@hf-v2]：

| 代码模块 | 输入 → 输出 | 对应论文 |
|---|---|---|
| `q_a_proj` + `q_a_layernorm` | 5120 → 1536，再 RMSNorm | $W^{DQ}$，式 (37) |
| `q_b_proj` | 1536 → 128×192 | $W^{UQ}$ 与 $W^{QR}$ 按头交错拼成一块，式 (38)(39)；之后按 [128, 64] 切开 |
| `kv_a_proj_with_mqa` | 5120 → 512 + 64 | $W^{DKV}$ 与 $W^{KR}$，式 (41)(43) |
| `kv_a_layernorm` | 512 → 512 | RMSNorm，**只作用在 $\mathbf c^{KV}$ 上，不作用在 $\mathbf k^R$ 上** |
| `kv_b_proj` | 512 → 128×(128 + 128) | $W^{UK}$ 与 $W^{UV}$，式 (42)(45) |
| `o_proj` | 128×128 → 5120 | $W^O$，式 (47) |

::: warn 参考代码缓存的不是潜向量
Hugging Face 参考实现的 `forward` 把**展开后**的键（128 个头 × 192 维）和值（128 个头 × 128 维）交给缓存对象 [@hf-v2]。每 token 每层 $128\times(192+128)=40{,}960$ 个数，60 层 2,457,600 个，bf16 下 4,915,200 字节：是 MLA 潜向量缓存（69,120 字节）的 71.1 倍，甚至比同形状的 MHA（3,932,160 字节）还多 25%，因为位置键被复制到了每个头。拿这份代码测显存，会得出“MLA 比 MHA 还费缓存”的错误结论。DeepSeek 在 V2 的 GitHub 说明里也承认，受 Hugging Face 框架所限，开源代码在 GPU 上比内部代码慢，并推荐用专门优化过的 vLLM 方案运行 [@gh-v2]。
:::

::: warn 位置段的维度排列
参考代码的 `apply_rotary_pos_emb` 先把 64 维位置段按相邻两两成对的方式重排，再做 LLaMA 式的 `rotate_half` [@hf-v2]。也就是说，权重里 RoPE 维度的配对方式是“相邻两个一对”。自己实现时如果直接套用 LLaMA 的前后两半配对，分数会是错的，而且不会报任何错。
:::

:::: quiz 小练习：V3 的 MLA
DeepSeek-V3 沿用了同一套 MLA：128 个头、$d_h=128$、$d_c=512$、$d_c'=1536$、$d_h^R=64$，只是隐藏维变成 7168、层数变成 61 [@dsv3]。（1）V3 每 token 缓存多少元素、多少字节（bf16）？（2）V3 一层 MLA 里哪些矩阵的形状与 V2 不同？
::: answer 看答案
（1）$(512+64)\times61=35{,}136$ 个元素，bf16 下 $70{,}272$ 字节，与 ISCA '25 论文表 1 给出的 70.272 KB 一致 [@isca25]。缓存只和 $d_c$、$d_h^R$、层数有关，与隐藏维无关。

（2）凡是一端连着 $\mathbf h_t$ 或 $\mathbf u_t$ 的矩阵：$W^{DQ}$（7168 → 1536）、$W^{DKV}$（7168 → 512）、$W^{KR}$（7168 → 64）、$W^O$（16384 → 7168）。$W^{UQ}$、$W^{QR}$、$W^{UK}$、$W^{UV}$ 只连着压缩向量和各头，形状不变。
:::
::::

::: key 本页要点
- 查询也被压到 $d_c'=1536$ 维，目的是省训练时的激活显存，不省 KV cache；每层查询投影参数从约 1.26 亿降到约 4,560 万。V2-Lite 不压缩查询。
- 附录 C 的式 (37)–(47) 就是完整的 MLA；推理时只缓存 $\mathbf c^{KV}_t$ 和 $\mathbf k^R_t$。一层约 1.49 亿参数，其中 $W^O$ 占 56%。
- 读代码时注意：参考实现缓存的是展开后的键值（4.9 MB/token），RoPE 维度按相邻成对排列，RMSNorm 只作用在 512 维潜向量上。
:::

# 两种计算模式与解码的屋顶线 | 同一层 MLA 的 MHA 模式与 MQA 模式，以及 FlashMLA | 25

**问题。** §3 末尾的警告说：吸收在解码时划算，在预填充时不划算。那么同一套权重，能不能在不同场合用不同的算法？可以，而且 DeepSeek 正是这样做的。这一页先把吸收之后的 MLA 写成一个标准的注意力形式，你会认出它就是 MQA；再算清两种算法各自在什么场合更便宜；最后用屋顶线模型解释，为什么 DeepSeek 说 MLA 的解码内核是“算力受限”的。

## 吸收之后，MLA 就是一个 MQA

把 §3 的吸收和 §5 的位置段合在一起：

::: deriv 从解耦打分到 MQA 形式
$\mathbf q_{t,i}^\top\mathbf k_{j,i}=\tilde{\mathbf q}_{t,i}^{\,\top}\mathbf c^{KV}_j+(\mathbf q^R_{t,i})^\top\mathbf k^R_j$ || 式 (9.15)：内容段吸收，位置段直接读缓存
$=\begin{bmatrix}\tilde{\mathbf q}_{t,i}\\ \mathbf q^R_{t,i}\end{bmatrix}^\top\begin{bmatrix}\mathbf c^{KV}_j\\ \mathbf k^R_j\end{bmatrix}$ || 分块点积倒过来用：两段点积之和 = 拼接向量的点积
$=\hat{\mathbf q}_{t,i}^{\,\top}\,\hat{\mathbf k}_j,\qquad \hat{\mathbf q}_{t,i}\in\R^{576},\ \hat{\mathbf k}_j\in\R^{576}$ || 定义拼接后的查询与键；$\hat{\mathbf k}_j$ 里没有头下标 $i$
$\tilde{\mathbf o}_{t,i}=\sum_jp_{t,j,i}\,\hat{\mathbf v}_j,\qquad \hat{\mathbf v}_j:=\mathbf c^{KV}_j\in\R^{512}$ || 式 (9.10) 的值一侧：值就是缓存里的潜向量本身，也没有头下标
:::

::: eq MLA 的 MQA 形式 | (9.19)
$$\begin{aligned}\hat{\mathbf q}_{t,i}&=\begin{bmatrix}(W^{UK}_i)^\top\mathbf q^C_{t,i}\\ \mathbf q^R_{t,i}\end{bmatrix}\in\R^{576},\quad \hat{\mathbf k}_j=\begin{bmatrix}\mathbf c^{KV}_j\\ \mathbf k^R_j\end{bmatrix}\in\R^{576},\quad \hat{\mathbf v}_j=\mathbf c^{KV}_j\in\R^{512}\\
\tilde{\mathbf o}_{t,i}&=\sum_{j=1}^{t}\softmax_j\!\Big(\frac{\hat{\mathbf q}_{t,i}^{\,\top}\hat{\mathbf k}_j}{\sqrt{d_h+d_h^R}}\Big)\hat{\mathbf v}_j\end{aligned}$$
然后 $\mathbf u_t=\sum_iW^O_iW^{UV}_i\tilde{\mathbf o}_{t,i}$。128 个查询头，共享**一个**键（576 维）和**一个**值（512 维）：这正是多查询注意力。缩放仍然是 $1/\sqrt{192}$，因为点积的数值一个都没变。
:::

所以 DeepSeek 把这种算法叫作 MLA 的 **MQA 模式**；把先展开、再按普通多头注意力算的做法叫作 **MHA 模式**。FlashMLA 的说明写得很具体：MQA 模式是 `head_dim_k = 576`、`head_dim_v = 512`（对 V3 系列），MHA 模式是 `head_dim_k = 192 / 128`、`head_dim_v = 128`（V3 系列的键是 $128+64=192$ 维）[@flashmla]。缓存里的同一行 576 个数，在 MQA 模式里前 512 个数既当键的一部分又当值：一次从显存读进来，用两次。

这里值得停一下，体会一个有点反直觉的结论。MLA 的每个头都有自己的键和值（按 MHA 的方式训练），**但它在解码时的计算形式是 MQA**：128 个头对着同一个共享的键值头提问。区别在于，MQA 模型的共享键值只有 $2d_h=256$ 维，是被迫共享的；MLA 的共享键值是 576 维的潜向量，每个头通过自己的 $(W^{UK}_i)^\top$ 和 $W^{UV}_i$ 从里面读出不同的东西。

## 两种模式各算多少

两种模式的每 token 固定成本相同：MHA 模式要把每个 token 的潜向量展开一次（$2d_c\cdot2n_hd_h$），MQA 模式要为每个查询吸收一次、输出时乘一次 $W^{UV}$（$2n_hd_hd_c\times2$），都是 $33{,}554{,}432$ 次运算。区别在每一对（查询, 键）上：

::: eq 两种模式的每对运算量（一层，V2 维度） | (9.20)
$$\begin{aligned}\text{MHA 模式：}&\quad 2n_h(d_h+d_h^R)+2n_hd_h=81{,}920\\
\text{MQA 模式：}&\quad 2n_h(d_c+d_h^R)+2n_hd_c=278{,}528\end{aligned}$$
MQA 模式每对贵 3.4 倍，因为它的点积在 576 维、加权求和在 512 维上做，而不是 192 维和 128 维。
:::

::: ex 预填充与解码，各选哪个
**预填充**：一次处理 $T=4096$ 个 token，因果掩码下约有 $T(T+1)/2=8{,}390{,}656$ 对。
MHA 模式：$81{,}920\times8{,}390{,}656+33{,}554{,}432\times4096\approx824.8$ GFLOPs；MQA 模式：$278{,}528\times8{,}390{,}656+33{,}554{,}432\times4096\approx2{,}474.5$ GFLOPs。MHA 模式便宜 3 倍，因为每个 token 只展开一次，展开的成本被它后面的几千个查询分摊掉了。$T=1024$ 时两者是 77.4 与 180.5 GFLOPs。

**解码**：1 个新 token，缓存 $T=4096$。MHA 模式如果不存展开后的键值，就得把 4096 个潜向量全部重新展开：$(33{,}554{,}432+81{,}920)\times4096\approx137.8$ GFLOPs；MQA 模式 $33{,}554{,}432+278{,}528\times4096\approx1.174$ GFLOPs。MQA 模式便宜 117 倍。

所以结论和 DeepSeek 的做法一致：DeepSeek-V3.2 的论文附录说，对 V3.1-Terminus，训练和预填充用 MHA 模式，解码用 MQA 模式 [@dsv32]。
:::

::: fig figures/l09-two-modes.svg | 图 9.11 同一层 MLA 的两种算法：MHA 模式与 MQA 模式（思路据 DeepSeek-V3.2 附录 A，数字为本课计算） | wide scroll
**看两边上半部分的连线：左边 128 个查询头各连一对自己的键值（先从缓存展开），右边 128 个查询头全部连到同一对琥珀色框的键值，那就是缓存里的 576 个数本身。** 下方表格把两种模式放到预填充和解码两个场景里比较，粗体是更便宜的那一个。每对运算量与固定成本按式 (9.20) 计算，V2 维度 [@dsv2]；“训练与预填充用 MHA 模式、解码用 MQA 模式”出自 DeepSeek-V3.2 论文 [@dsv32]。
:::

MQA 模式还有一个第 14 课会用到的性质：每个缓存条目被全部查询头共享。DeepSeek-V3.2 的 DSA 稀疏注意力要为每个查询挑出一部分键值来算，论文解释说，在内核层面每个键值条目必须被多个查询共享才算得快，所以 DSA 建立在 MLA 的 MQA 模式上 [@dsv32]。

## 屋顶线：为什么解码内核接近算力受限

::: pre 算术强度与屋顶线 | 学过第 4 课的可以跳过
一段计算的**算术强度**（arithmetic intensity）= 运算次数 ÷ 从显存读写的字节数。硬件有两个上限：峰值算力（每秒多少次运算）和显存带宽（每秒读多少字节）。算术强度低于“峰值算力 ÷ 带宽”（称为**屋脊点**）时，速度由带宽决定，叫带宽受限；高于它时由算力决定，叫算力受限。第 4 课说过，普通的解码注意力每读一个数只做一两次运算，严重带宽受限。完整讲法见 [M10 §3–§4](../math/m10-compute-arithmetic.html#p3)。
:::

沿用 FlashMLA 的记号：$d_k=576$ 是 MQA 模式里键的维度，$d_v=512$ 是值的维度，$h_q$ 是共享这一个键值头的查询头数（V2 为 128），$s_q$ 是每条序列同时解码的 token 数（普通解码为 1）。用 MQA 模式解码时，缓存里每个 token 的 576 个数只读一次，却被 $h_q$ 个查询头各用一次；如果一次解码多个 token（例如第 11 课的 MTP 投机解码，每条序列同时有 $s_q$ 个查询），还要再乘 $s_q$。逐项数一遍：

::: deriv MQA 模式的算术强度（每个缓存 token）
$\text{打分：}2d_k\ \text{次运算}\times h_qs_q$ || 每个查询（$h_q$ 个头 × $s_q$ 个 token）与这个 token 的 576 维键做一次点积
$\text{加权求和：}2d_v\ \text{次运算}\times h_qs_q$ || 每个查询把这个 token 的 512 维值乘上权重加进输出
$\text{读取：}2d_k\ \text{字节}$ || bf16 每个数 2 字节；值是键的前 512 维，不另读
$I=\dfrac{2h_qs_q(d_k+d_v)}{2d_k}$ || 运算除以字节；$d_v\approx d_k$ 时约为 $2h_qs_q$
:::

::: eq MQA 模式解码的算术强度 | (9.21)
$$I=\frac{2h_qs_q(d_k+d_v)}{2d_k}=h_qs_q\cdot\frac{d_k+d_v}{d_k}\approx2h_qs_q,\qquad d_k=576,\ d_v=512$$
FlashMLA 的技术博客正是这样写的：计算与访存之比为 $h_qs_q\cdot\frac{d_k+d_v}{d_k}\approx2h_qs_q$ [@flashmla-blog]。
:::

::: ex 解码注意力的算术强度：四个算例（每个缓存 token，bf16）
- MHA 式缓存（每头独立键值）：每读一个头的 $\mathbf k$、$\mathbf v$（$2\times128\times2$ 字节）做 $2\times128\times2$ 次运算，$I=1.0$ FLOP/B。
- GQA-8（DeepSeek 67B，64 个查询头共享 8 组）：$I=\dfrac{2\times64\times128\times2}{2\times8\times128\times2}=8.0$ FLOP/B。
- MLA 的 MQA 模式，$h_q=128$、$s_q=1$：$I=128\times\dfrac{576+512}{576}\approx241.8$ FLOP/B。
- 同上，$s_q=2$：$I\approx483.6$ FLOP/B。

H800 的数字（DeepSeek 在博客里给出）：带宽 3.35 TB/s，名义峰值 990 TFLOPS，实际因降频约 865 TFLOPS [@flashmla-blog]。屋脊点分别是 $990/3.35\approx295.5$ 与 $865/3.35\approx258.2$ FLOP/B。
:::

::: fig figures/l09-roofline.svg | 图 9.12 H800 上解码注意力的屋顶线：MHA、GQA 与吸收后的 MLA | wide scroll
**看四个点离屋脊有多远：MHA 和 GQA 在斜坡的最下端，算力几乎全部闲置；MLA 的 MQA 模式（$s_q=1$）贴着屋脊点，$s_q=2$ 时越过屋脊、落在平顶上。** 横纵都是对数坐标。斜线是带宽上限（3.35 TB/s × 算术强度），实线平顶是名义峰值 990 TFLOPS，虚线是降频后约 865 TFLOPS。点的横坐标按上面的算例计算；硬件数字来自 FlashMLA 的技术博客 [@flashmla-blog]，本课未与 NVIDIA 的规格表交叉核对。
:::

FlashMLA 博客的判断是：当 $h_qs_q\ge\tfrac12\cdot\tfrac{865}{3.35}=128$ 时内核是算力受限的；而解码实例不用{{张量并行::把一层的矩阵切成几份放到几张卡上同时算；对注意力层来说，通常就是把不同的头分给不同的卡}}，所以 $h_q=128$，内核算力受限 [@flashmla-blog]。严格按屋顶线算，情况更微妙一点：$\tfrac12\cdot\tfrac{865}{3.35}$ 实际是 129.1，博客取整成了 128；按精确比值，$s_q=1$ 时的 241.8 **略低于**降频后的屋脊点 258.2。所以更准确的说法是：MLA 把解码注意力的算术强度从 GQA 的个位数抬到了屋脊点附近，DeepSeek 在工程上把它当作算力受限来优化；$s_q=2$ 时它才明确越过屋脊点。

这里也能看出“解码不用张量并行”的意义：张量并行会把 128 个头分到几张卡上，每张卡上的 $h_q$ 变小，算术强度随之按比例下降，又退回到带宽受限的区域。

::: note FlashMLA：把 MQA 模式做成内核
DeepSeek 在 2025 年 2 月 24 日开始的“开源周”第一天发布了 FlashMLA，一个为 MLA 解码写的内核 [@flashmla]。首发说明给出的是：在 H800 SXM5 上，带宽受限的配置下达到 3000 GB/s，算力受限的配置下达到 580 TFLOPS（CUDA 12.6，BF16，{{分页 KV cache::像操作系统管理内存页那样，把每条序列的缓存切成固定大小的块、按需分配，不要求整条序列连续存放}}的块大小为 64）；后来的版本更新为 660 TFLOPS（CUDA 12.8）[@flashmla]。按博客的数字粗算，3000 GB/s 约是 3.35 TB/s 的 90%，660 TFLOPS 约是 865 TFLOPS 的 76%。

V2 论文在 2024 年只说它的 MLA 是“基于改进版的 FlashAttention-2”优化的 [@dsv2]；V2 当时线上用的内核是不是后来开源的 FlashMLA，本课没能核实。FlashMLA 当前的说明里还有 V3.2 的 FP8 缓存格式：每 token 每层 656 字节，其中 512 字节是 FP8 的潜向量、16 字节是 4 个 float32 缩放因子（每个管 128 个值）、128 字节是 64 个 bf16 的位置键，位置键“不量化以保证精度” [@flashmla]。第 14 课会再回到这个格式；FlashMLA 内核本身在第 12 课讲推理部署时还会出现。
:::

::: ex FP8 缓存格式的 656 字节是怎么来的
FlashMLA 说明里的 V3.2 格式，每 token 每层一行 [@flashmla]：512 维潜向量各 1 字节（FP8）= 512 字节；每 128 个值配一个 float32 缩放因子，$512/128=4$ 个 × 4 字节 = 16 字节；64 维位置键保持 bf16，$64\times2=128$ 字节。合计 $512+16+128=656$ 字节。

V3.2 相对 V3.1-Terminus 唯一的架构改动是加上 DSA [@dsv32]，层数与 V3 相同，是 61 层（它在 Hugging Face 上的配置文件写着 `num_hidden_layers = 61`）。每 token $656\times61=40{,}016$ 字节，是 bf16 存法（70,272 字节）的 56.9%。省得不到一半，因为位置键和缩放因子没有压缩；图 9.13 里那根斜纹柱就是这个数。注意这只是 MLA 条目本身：DSA 还要为每个 token 每层另存一个 132 字节的 indexer 键（[第 14 课 §7](lesson-14-sparse-attention.html#p7)），所以 V3.2 实际的每 token 缓存是 $(656+132)\times61=48{,}068$ 字节，是 bf16 MLA 的 68.4%；第 18、19 课比较 V3.2 时用的都是这个数。
:::

:::: quiz 小练习：多少个查询头才够
（1）如果把 V2 的解码实例做 8 路张量并行、每张卡分到 16 个头，MQA 模式的算术强度是多少？（2）在降频后的屋脊点 258.2 FLOP/B 下，$s_q=1$ 时至少需要多少个查询头，算术强度才能达到屋脊点？
::: answer 看答案
（1）$16\times\dfrac{576+512}{576}\approx30.2$ FLOP/B，远在屋脊点左边，带宽受限。（2）$h_q\times\dfrac{1088}{576}\ge258.2$，$h_q\ge136.7$，即至少 137 个头；V2 只有 128 个，所以 $s_q=1$ 时略差一点，$s_q=2$ 时绰绰有余。
:::
::::

::: key 本页要点
- 吸收后的 MLA = 128 个查询头共享一个 576 维键、一个 512 维值的 MQA（MQA 模式）；先展开再算 = 普通 MHA，每头键 192 维、值 128 维（MHA 模式）。缩放都是 $1/\sqrt{192}$。
- 两种模式每 token 固定成本相同，MQA 模式每对贵 3.4 倍：预填充用 MHA 模式（$T=4096$ 时 824.8 对 2,474.5 GFLOPs），解码用 MQA 模式（137.8 对 1.174 GFLOPs）。DSA 建立在 MQA 模式上。
- MQA 模式的解码算术强度约 $2h_qs_q$：$h_q=128$ 时 241.8 FLOP/B，贴着 H800 的屋脊点（258.2～295.5），而 GQA-8 只有 8。MLA 把解码注意力从带宽受限推到了算力受限的边上。
:::

# 算总账 | 缓存表、字节、“2.25 组”、“93.3%”与质量 | 25

这一页把前面所有的数字放回到一张表、两张图里，回答开头的两个问题：MLA 到底省了多少缓存？论文那些头条数字各是怎么来的？

## 论文表 1：四种注意力的每 token 缓存

::: eq MLA 的缓存用每头维度表示 | (9.22)
$$(d_c+d_h^R)\,l=\Big(4d_h+\frac{d_h}{2}\Big)\,l=\frac92\,d_h\,l\qquad(d_c=4d_h,\ d_h^R=d_h/2)$$
V2 取 $d_c=512=4\times128$，$d_h^R=64=128/2$ [@dsv2]。论文表 1 把 MLA 这一行写作 $(d_c+d_h^R)l\approx\frac92d_hl$；在 V2 的取值下它其实是严格相等，$4.5\times128\times60=34{,}560$。
:::

::: deriv “相当于 2.25 组的 GQA”
$2n_gd_hl=\frac92d_hl$ || 让 GQA 的缓存 (9.5) 等于 MLA 的缓存 (9.22)
$2n_g=\frac92$ || 两边约去 $d_hl$
$n_g=2.25$ || 所以 MLA 的缓存等于只有 2.25 组的 GQA
:::

::: eq 等效 GQA 组数 | (9.23)
$$n_g^{\text{等效}}=\frac{d_c+d_h^R}{2d_h}=\frac{512+64}{2\times128}=2.25$$
论文表 1 的标题说：V2 的 KV cache “is equal to GQA with only 2.25 groups, but its performance is stronger than MHA” [@dsv2]。同一个标题还说，表中的缓存按元素个数计，与存储精度无关。
:::

| 注意力 | 每 token 缓存（元素） | V2 形状下每层 | 每 token（60 层） | bf16 字节 | 等效组数 | 论文表 1 的评价 |
|---|---|---|---|---|---|---|
| MHA | $2n_hd_hl$ | 32,768 | 1,966,080 | 3,932,160 | 128 | Strong |
| GQA（8 组） | $2n_gd_hl$ | 2,048 | 122,880 | 245,760 | 8 | Moderate |
| MQA | $2d_hl$ | 256 | 15,360 | 30,720 | 1 | Weak |
| MLA | $(d_c+d_h^R)l\approx\frac92d_hl$ | 576 | 34,560 | 69,120 | 2.25 | Stronger |

::: ex 和同形状的 MHA 比
MLA 每 token 34,560 个元素，同形状的 MHA 是 1,966,080 个：比值 $34{,}560/1{,}966{,}080\approx1.758\%$，缓存缩小约 56.9 倍，降幅约 98.2%。按每层看就是 $576/32{,}768$，与 $2.25/128$ 相同。

一条 128K 的序列：MLA（bf16）$69{,}120\times131{,}072\approx9.06$ GB（8.44 GiB），同形状 MHA 约 515.4 GB。第 1 页图 9.1 里那条橙线，就是这个数。
:::

::: ex 一张 80 GB 的卡能放下多少个 token 的缓存
只看缓存、不算模型权重（V2 的权重本身要多张卡才放得下，这里只比较缓存的“密度”），$80\times10^9$ 字节能放：

- V2（MLA，bf16）：$80\times10^9/69{,}120\approx1.16\times10^6$ 个 token；部署态平均 6 bit：约 $3.09\times10^6$ 个。
- DeepSeek 67B（GQA-8，bf16）：约 $2.06\times10^5$ 个。
- V2 的形状若用 MHA（bf16）：约 $2.03\times10^4$ 个，还放不下一条 32K 的序列。

同样的显存，MLA 能服务的 token 总数是同形状 MHA 的 56.9 倍：这些 token 可以分给更多并发的序列（更大的 batch），也可以留给更长的上下文。论文对部署态 V2 的解释正是这样：缓存比 67B 小得多，所以能服务大得多的 batch [@dsv2]；5.76 倍吞吐的来历放在第 10 课。
:::

::: fig figures/l09-kv-bytes.svg | 图 9.13 每 token 的 KV cache 字节数：各模型对比（按真实配置计算，对数横轴） | wide scroll
**看橙色与灰色的分界：用 MLA 的模型每 token 都在 100 KB 以下，用 MHA 或 GQA 的都在 200 KB 以上，而且与模型大小关系不大。** 横轴是对数刻度，相邻刻度差 10 倍。DeepSeek 自家：7B（MHA）491.5 KB，67B（GQA-8）389.1 KB，MoE 16B（MHA，28 层、16 头）229.4 KB；V2 69.1 KB，V2-Lite 31.1 KB，V3 70.3 KB [@dsllm; @dsmoe; @dsv2; @dsv3]。斜纹两根是在 MLA 之上再做量化：V2 部署时平均 6 bit（25.9 KB）[@dsv2] 和 V3.2 在 FlashMLA 里的 FP8 格式（40.0 KB，每层 656 字节 × 61 层；只算 MLA 条目，加上 DSA 的 indexer 键是 48.1 KB）[@flashmla]。Qwen-2.5 72B（80 层、8 个 KV 头）与 LLaMA-3.1 405B（126 层、8 个 KV 头）的规格取自各自的技术报告 [@qwen25; @llama3]，按它们算出的字节数与 ISCA '25 论文表 1 逐字节吻合 [@isca25]。
:::

::: err 一个印错的倍数
ISCA '25 论文表 1 用 BF16 比较每 token KV cache：DeepSeek-V3 70.272 KB，Qwen-2.5 72B 327.680 KB（标 4.66x），LLaMA-3.1 405B 516.096 KB（标 7.28x）[@isca25]。第一个倍数是对的，$327.680/70.272\approx4.663$；第二个算错了，$516.096/70.272\approx7.344$，应为 7.34 倍。arXiv 的 v1、v2 两个版本都印作 7.28x，ACM 出版版本课没有核对。引用时写“约 7.3 倍”并注明原表笔误。
:::

## “KV cache 降低 93.3%”是怎么来的

DeepSeek-V2 摘要里的头条数字是：与 DeepSeek 67B 相比，训练成本省 42.5%，KV cache 降低 93.3%，最大生成吞吐提升到 5.76 倍 [@dsv2]。论文图 1(b) 用两根横条画出两者每 token 的 KV cache。笔者按论文 arXiv HTML 版矢量图的坐标反算了柱长：67B 389.1 KB/token，V2 25.9 KB/token（这个反算方法用同一张图里训练成本的两根柱子校验过，得到的 300.6、172.8 与论文正文的 K GPU hours 数字完全一致）。这两个数都能用公式精确复算出来，而复算的结果说明，93.3% 不全是 MLA 的功劳。

::: deriv 把 93.3% 拆成三个因子
$389{,}120=2\times8\times128\times95\times2$ || 67B：GQA-8，每层 2,048 个数，95 层，bf16 每个 2 字节；即图 1(b) 的 389.1 KB
$\times\frac{576}{2048}=0.281$ || 每层换成 MLA 的 576 个数：得 109,440 字节
$\times\frac{60}{95}=0.632$ || 层数从 95 变成 60：得 69,120 字节，即 V2 用 bf16 存时的缓存
$\times\frac{6}{16}=0.375$ || 部署时把缓存量化到平均 6 bit：得 25,920 字节，即图 1(b) 的 25.9 KB
$1-\frac{25{,}920}{389{,}120}=93.34\%$ || 论文的 93.3%
:::

::: eq “93.3%”的因子分解 | (9.24)
$$\frac{25{,}920}{389{,}120}=\underbrace{\frac{576}{2048}}_{\text{每层：MLA 对 GQA-8}}\times\underbrace{\frac{60}{95}}_{\text{层数}}\times\underbrace{\frac{6}{16}}_{\text{6 bit 对 bf16}}\approx0.0666$$
只看 MLA 与层数（bf16 对 bf16），降幅是 $1-69{,}120/389{,}120\approx82.2\%$；再加上量化，才是 93.3%。三个因子是相乘的，按先后顺序算百分点会因顺序不同而不同，所以这里按对数分摊（与顺序无关）：$\ln(389{,}120/69{,}120)/\ln(389{,}120/25{,}920)\approx1.728/2.709\approx0.64$，即 MLA 与层数贡献了约 64% 的降幅，量化贡献约 36%。
:::

::: fig figures/l09-waterfall.svg | 图 9.14 “KV cache 降低 93.3%”的分解（首尾两根柱子对应论文图 1(b) 的两根柱子） | wide scroll
**看红色虚线框的大小：每一步砍掉的部分。第一步（每层从 2,048 个数换成 576 个）砍得最多（279.7 KB），最后一步量化砍掉的只是一小截（43.2 KB，与层数那一步的 40.3 KB 相当），却把降幅从 82.2% 推到了 93.3%。** 首尾两根柱子与论文图 1(b) 两根柱子的反算长度 389.1 KB、25.9 KB 逐位吻合 [@dsv2]；中间两根是本课插入的中间状态。每一步的因子由上面的推导计算。
:::

::: warn 这个拆解是推断，但证据很强
论文没有写明 93.3% 的计算口径。论文在同一节里说，部署时把参数转成 FP8，并把 KV cache 量化到“平均 6 bit” [@dsv2]；而 $34{,}560\times6/8=25{,}920$ 字节恰好就是图 1(b) 那根柱子反算出的 25.9 KB，$2\times8\times128\times95\times2=389{,}120$ 字节恰好就是 389.1 KB。两根柱子都逐位吻合，所以“93.3% 是部署态 V2（6 bit KV）对 bf16 的 67B”这一解读可信度很高，但它仍然是本课的推断。5.76 倍的吞吐写在同一段里，紧跟在 FP8 权重与 6 bit KV 的部署说明之后，测法是：在一台 8 张 H800 的节点上，按线上 DeepSeek 67B 服务的提示与生成长度分布来测 [@dsv2]。从上下文看它也是部署态的数字，但论文没有逐项写明测吞吐时用的精度；第 10 课再细讲。量化的具体方案（哪些维度量化、位置键是否量化），论文只引用了两篇量化论文，没有给出细节。
:::

## 质量：MLA 真的比 MHA 强吗

缓存省了，质量呢？论文附录 D.2 在两个规模上比较了 MHA 与 MLA：约 16B 总参数的小 MoE 模型训练 1.33T token，约 250B 总参数的大 MoE 模型训练 420B token [@dsv2]。

::: fig figures/l09-ablation-mla.svg | 图 9.15 MoE 模型上 MHA 与 MLA 的得分（据 DeepSeek-V2 附录 D.2 表 9 绘制） | wide scroll
**看箭头的方向：8 组对比里 7 组是橙色向右（MLA 更高），只有小模型的 C-Eval 是红色向左（50.9 对 51.6）。** 顶部两行是每 token 的缓存元素数：小模型 110.6K 对 15.6K（14%），大模型 860.2K 对 34.6K（4%）。数据照录论文表 9 [@dsv2]。
:::

论文据此说 MLA 的表现好于 MHA，而且缓存小得多 [@dsv2]。读这张表时有四点要一起记住：

1. 并非全面领先：小模型的 C-Eval 上 MLA（50.9）低于 MHA（51.6）。
2. 对照并不完全等量：MLA 模型的激活参数略少（小模型 2.4B 对 2.5B，大模型 21.5B 对 25.0B）。
3. 大模型只训练了 420B token，远少于正式模型。
4. 到了 V3 的报告里，措辞改成了 MLA 在缓存大大减少的同时，性能与标准 MHA **相当**（comparable）[@dsv3]。

所以最稳妥的表述是：**在 DeepSeek 自己的实验里，MLA 总体上不输 MHA，而缓存只有 MHA 的 14%（小模型）到 4%（大模型）**；换到 V2 的完整形状（128 个头），则是 1.76%。与图 9.2 的 GQA、MQA 放在一起看，这正是 MLA 要证明的事：省缓存不一定要以质量为代价。

## 全课总账

| 设计 | 解决什么 | 代价 | 关键数字（V2） |
|---|---|---|---|
| 低秩 KV 联合压缩（§2） | 缓存只存潜向量，头数不减 | 键值投影被限制为秩 ≤ 512 | 每层 512 个数，MHA 的 1/64 |
| 矩阵吸收（§3） | 解码时不必展开缓存 | 每个缓存 token 多做约 3.4 倍运算 | $T=4096$：1.174 对 137.8 GFLOPs |
| 解耦 RoPE（§5） | 让位置编码与吸收共存 | 多缓存 64 个数；位置键所有头共享 | 每层 576 个数 |
| 查询压缩（§6） | 省训练时的激活显存 | 不省 KV cache | 查询投影参数 1.26 亿 → 4,560 万 |
| 两种计算模式（§7） | 预填充与解码各用便宜的算法 | 需要两套内核 | 解码算术强度 241.8 FLOP/B |

::: ana 回到类比：速记纪要本
**映射。** KV cache 是会议纪要本；MLA 让秘书只记速记（512 维潜向量）外加一小栏时间戳（64 维位置键）；矩阵吸收是读者练出了直接读速记的眼光；两种计算模式是“开大会时先把速记整本翻译出来大家一起看（预填充），单独查阅时直接读速记（解码）”。

**收益。** 本子薄了 56.9 倍，而每个读者（头）仍然读出了自己想要的内容；翻译规则固定，所以可以把它“练进眼光”里。

**失效的地方。** 速记不是事后压缩出来的，而是模型从第一天就用速记思考，速记的“字典”（$W^{DKV}$、$W^{UK}$、$W^{UV}$）和模型一起学出来；直接读速记也不是免费的，每次比对要处理 512 个符号，比读全文里自己那一栏更累，只是在解码时这份累本来就闲着。
:::

## 往后看

[第 10 课](lesson-10-v2-moe-training.html)讲 V2 的另一半：DeepSeekMoE、设备受限路由、YaRN 把上下文扩展到 128K（MLA 里只有那 64 维位置段带 RoPE，这让 YaRN 的用法有一处细节要讲），以及 5.76 倍吞吐的来历。[第 11 课](lesson-11-v3-architecture.html)的 V3 原样沿用本课的 MLA。[第 14 课](lesson-14-sparse-attention.html)的 DSA 在 MQA 模式上做稀疏选择，FlashMLA 为它配了每 token 每层 656 字节的 FP8 缓存格式。FlashMLA 当前的说明里，V4 与 V4.1 的 MQA 模式键维度是 512，而 V3 系列是 576 [@flashmla]：注意力的结构又变了，[第 18 课](lesson-18-v4.html)、[第 19 课](lesson-19-v4-1-flash.html)会讲它变成了什么。

::: key 本页要点
- MLA 每 token 缓存 $(d_c+d_h^R)l=\tfrac92d_hl$，V2 为 34,560 个元素、69,120 字节（bf16），等于 2.25 组的 GQA，是同形状 MHA 的 1.76%。
- “93.3%”= 每层 576/2048 × 层数 60/95 × 量化 6/16；纯 MLA 加层数（bf16）是 82.2%。头条数字里有一部分来自部署时的量化。
- 质量：V2 的消融里 MLA 在 8 项中 7 项高于 MHA，但对照并不完全等量；V3 的说法是“与 MHA 相当”。可靠的结论是：缓存只剩 MHA 的 4%～14%（V2 的完整形状下是 1.76%），在 DeepSeek 的实验里质量总体不输 MHA。
:::

# 练习与答案 | 12 题：概念、计算、推导 | 50

先自己做，再展开答案。计算题都能用计算器完成，所需的维度都在前面各页里：V2 为 $d=5120$、$n_h=d_h=128$、$l=60$、$d_c=512$、$d_c'=1536$、$d_h^R=64$；V2-Lite 为 $d=2048$、$n_h=16$、$d_h=128$、$l=27$、$d_c=512$、$d_h^R=64$、不压缩查询；DeepSeek LLM 67B 为 95 层、8 组键值头、$d_h=128$ [@dsv2; @dsllm]。

## 概念题

:::: quiz 练习 1 · 查询压缩省了什么
（a）为什么 KV cache 里只存键和值，不存查询？（b）既然如此，V2 为什么还要压缩查询？它省下的是什么？（c）V2-Lite 不压缩查询，这对“吸收”这一步有什么影响？
::: answer 看答案
（a）生成第 $t$ 个 token 时，式 (9.3) 只用到当前 token 的查询 $\mathbf q_{t,i}$，却要用到所有 $j\le t$ 的键和值。历史 token 的查询以后再也不会被用到，所以没有必要存。

（b）压缩查询省的是**训练时的激活显存**和查询投影的参数，不是 KV cache。论文明说它这样做“even if it cannot reduce the KV cache” [@dsv2]。参数上，每层查询投影从约 1.26 亿（$5120\times128\times192$）降到约 4,560 万（$5120\times1536+1536\times128\times192$）。

（c）没有 $W^{UQ}$，查询直接是 $\mathbf q^C_{t,i}=W^Q_i\mathbf h_t$，吸收写成 $\tilde{\mathbf q}_{t,i}=(W^{UK}_i)^\top W^Q_i\mathbf h_t$，这正是论文正文“$W^{UK}$ 吸收进 $W^Q$”的字面意思。推导（结合律）和工程做法（先算 $\mathbf q^C$，再乘 $(W^{UK}_i)^\top$）都不变。
:::
::::

:::: quiz 练习 2 · MLA 解码时是 MQA，为什么不等于 MQA
§7 说，吸收后的 MLA 在解码时就是 128 个查询头共享一个键值头的 MQA。那么 MLA 和一个普通的 MQA 模型区别在哪里？请从“共享的是什么”“每个头能读出什么”两方面回答。
::: answer 看答案
**共享的是什么。** 普通 MQA 所有头共享一个 $d_h=128$ 维的键和一个 128 维的值（每层 256 个数）。MLA 共享的是一个 576 维的键（512 维潜向量 + 64 维位置键）和一个 512 维的值（就是那个潜向量）。

**每个头能读出什么。** 在 MQA 里，每个头拿到的是同一个键、同一个值，差别只在查询。在 MLA 里，每个头通过自己的 $(W^{UK}_i)^\top$ 把查询变成 512 维的 $\tilde{\mathbf q}_{t,i}$，相当于用自己的方式“读”潜向量；输出时再用自己的 $W^{UV}_i$ 把潜空间的加权平均翻译回来。所以在数学上，每个头仍然有自己的键 $W^{UK}_i\mathbf c_j$ 和值 $W^{UV}_i\mathbf c_j$，和 MHA 一样；只是所有这些键值都由同一个 512 维潜向量线性决定：每个 token 全部头的键值拼成的长向量，落在同一个至多 512 维的子空间里（秩约束，§2）。

一句话：MQA 是“少存、少用”，MLA 是“少存、照样多用”。代价是 MLA 每个缓存 token 要多做运算（§3），以及那个秩约束本身。至于这种差别在质量上意味着什么，只能靠实验回答（§8 的两组消融），不能从公式推出来。
:::
::::

:::: quiz 练习 3 · 解耦 RoPE 的三个设计点
解耦 RoPE 有三个设计点：（i）位置键 $\mathbf k^R_t$ 直接从 $\mathbf h_t$ 算，不经过 $\mathbf c^{KV}_t$；（ii）位置键所有头共享；（iii）位置键旋转之后才写进缓存。分别说明：如果不这样做，会出什么问题（或者多付什么代价）？
::: answer 看答案
（i）这一点是设计选择，不是必需：如果从 $\mathbf c^{KV}_t$ 算，$\mathbf k^R_t=\mathbf R_tW^{KR}\mathbf c^{KV}_t$，旋转后缓存照样可行。论文选择从 $\mathbf h_t$ 算，位置键因此不受 512 维瓶颈的限制（笔者的理解，论文没有说明理由）。

（ii）不共享的话，每层要多存 $128\times64=8{,}192$ 个数，缓存从 576 变成 8,704，是 15.1 倍，比 8 组的 GQA（2,048）还大，MLA 省缓存的意义就没了。

（iii）如果存的是旋转前的位置键，每一步都得按各自的位置 $j$ 把 $T$ 个历史位置键重新旋转一遍：运算不多，却完全是重复劳动。第 $j$ 个 token 旋转后的位置键以后不会再变（和第 4 课“缓存为什么是精确的”同理），所以在它自己被处理时旋转一次、存下来即可；以后每一步新查询只需旋转自己，点积里自然出现 $\mathbf R_{j-t}$。这正是 §4 的退路二“缓存旋转后的键”，只是现在它只有 64 维、全头共享，所以付得起。
:::
::::

## 计算题

:::: quiz 练习 4 · 每 token 与每条序列的缓存
（a）算 V2 与 DeepSeek 67B 在 bf16 下每 token 的 KV cache 字节数，V2 是 67B 的百分之几？（b）一条 128K（131,072 token）的序列，两者各需多少 GB？（c）V3（61 层，MLA 维度同 V2）呢？
::: answer 看答案
（a）V2：$(512+64)\times60\times2=69{,}120$ 字节。67B：$2\times8\times128\times95\times2=389{,}120$ 字节。$69{,}120/389{,}120\approx17.8\%$。

（b）V2：$69{,}120\times131{,}072\approx9.06$ GB；67B：$389{,}120\times131{,}072\approx51.0$ GB（67B 原生只支持 4K，这里只是按公式对比）。

（c）V3：$(512+64)\times61\times2=70{,}272$ 字节每 token，128K 时约 9.21 GB。与 ISCA '25 论文表 1 的 70.272 KB 一致 [@isca25]。
:::
::::

:::: quiz 练习 5 · 等效组数与表 9 的“14%”
（a）V2 与 V2-Lite 的 MLA 各相当于几组 GQA？（b）各是同形状 MHA 缓存的百分之几？（c）论文附录 D.2 表 9 里，小 MoE 模型的每 token 缓存是 MLA 15.6K、MHA 110.6K 个元素 [@dsv2]。比值是多少？和（b）的哪个数接近？为什么？
::: answer 看答案
（a）两者的 $d_c$、$d_h^R$、$d_h$ 相同：$(512+64)/(2\times128)=2.25$ 组。

（b）V2：$2.25/128\approx1.76\%$。V2-Lite 只有 16 个头：$2.25/16\approx14.1\%$，也就是 $576/(2\times16\times128)=576/4096$。

（c）$15.6/110.6\approx14.1\%$，和 V2-Lite 的比例几乎相同。$576\times27=15{,}552$ 正是 15.6K，$2\times16\times128\times27=110{,}592$ 正是 110.6K，所以表 9 的小模型很可能与 V2-Lite 一样是 16 个头、27 层（表 9 里小 MLA 模型的 15.7B 总参数、2.4B 激活参数，也与附录 B 给 V2-Lite 的数字相同）；这是笔者由数值吻合推断的，论文没有给出这两个对照模型的配置。头少，MHA 的缓存本来就不大；$d_c$ 固定为 512 时，头越多，MLA 的压缩比越大，这就是大模型那一组只剩 4% 的原因。
:::
::::

:::: quiz 练习 6 · 32K 上下文的解码账单
缓存里有 $T=32{,}768$ 个 token，V2 一层解码一个新 token。用式 (9.11) 算三种算法（① 每步重新展开，② 缓存展开后的键值，③ 吸收）的运算量，并算 ② 和 ③ 每步要读的缓存字节数（bf16）。③ 比 ② 多算几倍、少读几倍？
::: answer 看答案
运算：① $33{,}636{,}352\times32{,}768\approx1{,}102.2$ GFLOPs；② $33{,}554{,}432+81{,}920\times32{,}768\approx2.72$ GFLOPs（固定成本 0.034 GFLOPs 是展开新 token 的键值）；③ $33{,}554{,}432+278{,}528\times32{,}768\approx9.16$ GFLOPs。

读缓存：③（与 ① 相同）$1{,}152\times32{,}768\approx37.7$ MB；② $81{,}920\times32{,}768\approx2.68$ GB。

③ 比 ② 多算 $9.16/2.72\approx3.4$ 倍，少读 $81{,}920/1{,}152\approx71$ 倍。③ 比 ① 少算约 120 倍。
:::
::::

:::: quiz 练习 7 · 读一遍缓存要多久
解码时每生成一个 token，都要把这条序列的整份 KV cache 从显存读一遍。按 H800 的 3.35 TB/s 带宽 [@flashmla-blog]，一条 128K 的序列，只算读缓存的时间（忽略权重与计算），以下几种情况每生成一个 token 至少要多少毫秒？（a）V2，bf16；（b）V2，平均 6 bit；（c）67B，bf16；（d）V2 的形状若用 MHA，bf16。
::: answer 看答案
每条序列的缓存除以带宽：

（a）$9.06\times10^9/3.35\times10^{12}\approx2.70$ ms；（b）$25{,}920\times131{,}072\approx3.40$ GB，约 1.01 ms；（c）51.0 GB，约 15.2 ms；（d）515.4 GB，约 154 ms，而且 515 GB 远远超过一张 80 GB 显卡的显存。

这是理想化的下界：真实系统还要读权重、做计算，多条序列共享权重读取（batch）。但它说明了缓存字节数直接变成每个 token 的时间。
:::
::::

:::: quiz 练习 8 · 预先乘好还是改变顺序（V2-Lite）
V2-Lite 不压缩查询，查询内容段 $\mathbf q^C_{t,i}=W^Q_i\mathbf h_t$，$W^Q_i\in\R^{128\times2048}$；$W^{UK}_i\in\R^{128\times512}$；共 16 个头。比较两种做法的参数量和每 token 运算量：（a）预先乘好 $(W^{UK}_i)^\top W^Q_i$；（b）先算 $\mathbf q^C_{t,i}$，再乘 $(W^{UK}_i)^\top$。
::: answer 看答案
（a）每个头一个 $[512,2048]$：$16\times512\times2048=16{,}777{,}216$ 个参数；每 token 运算 $2\times512\times2048\times16=33{,}554{,}432$。

（b）$W^Q$ 内容段 $16\times128\times2048=4{,}194{,}304$，$W^{UK}$ $16\times128\times512=1{,}048{,}576$，共 $5{,}242{,}880$；运算 $2\times2048\times128\times16+2\times128\times512\times16=8{,}388{,}608+2{,}097{,}152=10{,}485{,}760$。

预乘在参数和运算上都是 3.2 倍。理由与 §3 相同：$\frac1{d_h}=\frac1{128}$ 大于 $\frac1{512}+\frac1{2048}$（比值正好是 $\frac{512\times2048}{128\times(512+2048)}=3.2$），中间维远小于两端，显式乘出来会把一个秩 128 的瘦结构膨胀成一整块。
:::
::::

:::: quiz 练习 9 · 参考实现的缓存
Hugging Face 参考实现把展开后的键（每头 192 维）和值（每头 128 维）存进缓存 [@hf-v2]。（a）V2 每 token 缓存多少字节（bf16）？是 MLA 潜向量缓存的几倍？是同形状 MHA 的几倍？（b）为什么比同形状的 MHA 还大？
::: answer 看答案
（a）每层每 token $128\times(192+128)=40{,}960$ 个数，60 层 $2{,}457{,}600$ 个，bf16 下 $4{,}915{,}200$ 字节。是 MLA（69,120 字节）的 71.1 倍，是同形状 MHA（3,932,160 字节）的 1.25 倍。

（b）MHA 每头的键是 128 维；参考实现存的键是 192 维，因为它把那个所有头共享的 64 维位置键复制到了每个头里。用这份代码测显存，得到的是“MLA 比 MHA 还费缓存”的错误结论。
:::
::::

## 推导题

:::: quiz 练习 10 · 所有 token 的键矩阵秩不超过 512
把 $T$ 个 token 的潜向量按行堆成 $C\in\R^{T\times d_c}$，所有头的键内容段按行堆成 $K^C\in\R^{T\times d_hn_h}$。（a）证明 $K^C=C\,(W^{UK})^\top$，并由此说明 $\rank(K^C)\le512$。（b）对照 MHA：它的 $K=H(W^K)^\top$ 的秩上限是多少（$H\in\R^{T\times d}$）？（c）这说明 MLA 对模型施加了什么约束？
::: answer 看答案
（a）第 $t$ 行：$(\mathbf k^C_t)^\top=(W^{UK}\mathbf c_t)^\top=\mathbf c_t^\top(W^{UK})^\top$（转置规则）。把 $T$ 行叠起来，左边是 $K^C$，右边是 $C(W^{UK})^\top$（这正是 M01 的式 M1.11：同一个投影的列写法与按行堆叠的写法）。$C$ 只有 $d_c=512$ 列，$K^C$ 的每一行都是 $(W^{UK})^\top$ 的 512 行的线性组合，所以 $K^C$ 的行空间至多 512 维，$\rank(K^C)\le512$。

（b）$\rank(K)\le\min(T,d,d_hn_h)=\min(T,5120,16384)$，最多 5120。

（c）$K^C$ 的每一行是一个 token 的 128 个头的键拼成的 16,384 维向量；所有 token 的这些行都挤在同一个至多 512 维的子空间里（值同理）。MHA 允许 5120 维。MLA 押注的是：注意力真正需要的键值信息远少于这个上限，而且这个 512 维子空间可以在训练中学出来（§2 的“联合训练”，不是事后 SVD）。这是一个约束，不是免费的；§8 的消融显示，在 DeepSeek 的实验里它总体上没有带来质量损失（8 项对比里 7 项 MLA 更高，小模型的 C-Eval 略低）。
:::
::::

:::: quiz 练习 11 · 什么样的中间矩阵不妨碍吸收
设打分形如 $\mathbf q^\top M_{t,j}W\mathbf c_j$，其中 $W\in\R^{d_h\times d_c}$ 是固定的权重，$M_{t,j}\in\R^{d_h\times d_h}$ 可能依赖位置。（a）证明：若 $M_{t,j}=M_t$ 只依赖当前位置 $t$，吸收仍然可行，写出吸收查询。（b）若 $M_{t,j}=\mathbf R_{j-t}$（RoPE），用 §4 的数值说明吸收失败。（c）解耦 RoPE 属于哪一种情况？
::: answer 看答案
（a）$\mathbf q^\top M_tW\mathbf c_j=\big((M_tW)^\top\mathbf q\big)^\top\mathbf c_j=\big(W^\top M_t^\top\mathbf q\big)^\top\mathbf c_j$。定义 $\tilde{\mathbf q}_t=W^\top M_t^\top\mathbf q\in\R^{d_c}$，它与 $j$ 无关，每步算一次即可（结合律 + 转置规则）。例如只旋转查询、不旋转键，就属于这种情况，只是那样打分里就没有相对位置信息了。

（b）$\tilde{\mathbf q}^{(j)}_t=W^\top\mathbf R_{j-t}^\top\mathbf q$ 随 $j$ 变。§4 的例子：$W=(3,1)^\top$，$\mathbf q=(1,2)$，$t=3$，$\theta=0.5$ 时 $j=1,2,3$ 分别得 $-1.5058$、$1.9908$、$5$，不存在通用的 $\tilde{\mathbf q}$。

（c）解耦 RoPE 把打分拆成两段：内容段的中间矩阵是单位矩阵（$M=I$，属于情况 (a) 的特例），可以吸收；位置段的中间矩阵是 $\mathbf R_{j-t}$（情况 (b)），但它根本不经过要吸收的 $W^{UK}$，而是直接用缓存里旋转好的 $\mathbf k^R_j$ 算，所以不需要吸收。
:::
::::

:::: quiz 练习 12 · 两种模式的分界
设一层 MLA 每 token 的固定成本为 $F_0=33{,}554{,}432$，每对（查询, 键）的成本在 MHA 模式为 $a=81{,}920$，在 MQA 模式为 $b=278{,}528$。（a）预填充 $T$ 个 token（因果，约 $T(T+1)/2$ 对）：证明对所有 $T\ge1$，MHA 模式都不比 MQA 模式贵。（b）解码一个 token、缓存 $T$ 个 token，且不存展开后的键值：MHA 模式成本 $(F_0+a)T$，MQA 模式 $F_0+bT$。求 MQA 模式更便宜的条件。（c）结合（a）（b），说明 DeepSeek 为什么“训练和预填充用 MHA 模式、解码用 MQA 模式” [@dsv32]。
::: answer 看答案
（a）两种模式的固定成本都是 $F_0T$，差别只在对数项：$\big(b-a\big)\cdot T(T+1)/2=196{,}608\cdot T(T+1)/2>0$。所以 MQA 模式总是更贵，MHA 模式更便宜。$T=4096$ 时差出约 1,650 GFLOPs（2,474.5 对 824.8）。

（b）$F_0+bT<(F_0+a)T\iff F_0<(F_0+a-b)T\iff T>\dfrac{F_0}{F_0+a-b}=\dfrac{33{,}554{,}432}{33{,}357{,}824}\approx1.0059$。所以只要缓存里有 2 个及以上的 token，MQA 模式就更便宜；$T=4096$ 时便宜约 117 倍。

（c）预填充和训练一次处理很多个查询，展开的成本被分摊，而 MHA 模式每对便宜 3.4 倍；解码每步只有一个查询，展开 $T$ 个潜向量的成本无处分摊，吸收（MQA 模式）只付一次固定成本。两种场合的最优算法不同，同一套权重就用两套内核来算。
:::
::::

```refs
- id: dsv2
  title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  a: DeepSeek-AI (Aixin Liu et al.) (2024), preprint. arXiv:2405.04434（读 v5，2024-06-19）
  url: https://arxiv.org/abs/2405.04434v5
  finding: 提出 MLA：把所有头的键值联合压缩成每层 512 维的潜向量，并用一个 64 维、所有头共享的解耦键承载 RoPE，每 token 缓存 $(d_c+d_h^R)l$ 个元素，论文称等于只有 2.25 组的 GQA，而能力强于 MHA。
  matters: 本课全部 MLA 公式（式 1–19 与附录 C 的式 37–47）、表 1、图 3、吸收论证、超参数与附录 D 的两组消融都以它 v5 的编号为准。
- id: dsllm
  title: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"
  a: DeepSeek-AI (2024), preprint. arXiv:2401.02954
  url: https://arxiv.org/abs/2401.02954
  finding: 表 2 给出 67B 为 95 层、隐藏维 8192、64 个查询头、8 个键值头（GQA），7B 为 30 层、32 头的 MHA。
  matters: 67B 是 V2 所有“对比 DeepSeek 67B”数字的基线，它每 token 389,120 字节的缓存是“93.3%”的分母。
- id: dsmoe
  title: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"
  a: Damai Dai et al. (2024), preprint. arXiv:2401.06066
  url: https://arxiv.org/abs/2401.06066
  finding: 第 5.1.2 节给出 DeepSeekMoE 16B 的配置：28 层、隐藏维 2048，注意力是 16 个头、每头 128 维的标准多头注意力。
  matters: 图 9.13 里 MoE 16B 每 token 229.4 KB 的出处；它是 V2 之前 DeepSeek 用 MHA 的 MoE 模型，可以和同样约 16B 的 V2-Lite（MLA）直接对照。
- id: mqa
  title: "Fast Transformer Decoding: One Write-Head is All You Need"
  a: Noam Shazeer (2019), preprint. arXiv:1911.02150
  url: https://arxiv.org/abs/1911.02150
  finding: 提出多查询注意力（MQA）：所有查询头共享同一组键和值，以减少解码时要读取的键值。
  matters: V2 表 1 与图 3 的 MQA 一行出自这里；本课 §7 会看到，吸收之后的 MLA 在计算形式上正是一种 MQA。
- id: gqa
  title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
  a: Joshua Ainslie et al. (2023), EMNLP 2023（据 arXiv 页面注记）. arXiv:2305.13245
  url: https://arxiv.org/abs/2305.13245
  finding: 提出分组查询注意力：$n_g$ 组键值各服务一组查询头，介于 MHA 与 MQA 之间。
  matters: “MLA 相当于 2.25 组 GQA”这一比较的参照物，也是 DeepSeek LLM 67B 采用的注意力。
- id: isca25
  title: "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures"
  a: Chenggang Zhao et al. (2025), ISCA '25（Industry Track）. arXiv:2505.09343
  url: https://arxiv.org/abs/2505.09343
  finding: 表 1 以 BF16 比较每 token KV：DeepSeek-V3 70.272 KB、Qwen-2.5 72B 327.680 KB、LLaMA-3.1 405B 516.096 KB；并说明 MLA 的投影矩阵与模型联合训练。
  matters: 为 V3 的缓存字节提供出处，也用来纠正“MLA 是事后 SVD 压缩”的误解；它把 7.34 倍误印为 7.28x，是本课勘误之一。
- id: qwen25
  title: "Qwen2.5 Technical Report"
  a: Qwen Team (An Yang et al.) (2024), preprint. arXiv:2412.15115
  url: https://arxiv.org/abs/2412.15115
  finding: 表 1 给出 Qwen2.5-72B 为 80 层、64 个查询头与 8 个键值头的 GQA。
  matters: 图 9.13 用它的层数与键值头数（每头 128 维，报告未列，按其 Hugging Face 配置 8192/64 算得）复算出每 token 327,680 字节，与 ISCA '25 表 1 对上，作为 DeepSeek 以外的 GQA 参照。
- id: llama3
  title: "The Llama 3 Herd of Models"
  a: Aaron Grattafiori et al. (2024), preprint. arXiv:2407.21783
  url: https://arxiv.org/abs/2407.21783
  finding: 表 3 给出 Llama 3 405B 为 126 层、模型维度 16,384、128 个注意力头、8 个键值头。
  matters: 图 9.13 用它复算出每 token 516,096 字节；这个数除以 V3 的 70,272 字节是 7.34，本课据此指出 ISCA '25 表 1 的 7.28x 是笔误。
- id: hf-v2
  kind: artefact
  title: DeepSeek-V2 在 Hugging Face 上的模型仓库（config.json 与 modeling_deepseek.py）
  a: DeepSeek-AI (2024). 读取于 2026-09-24
  url: https://huggingface.co/deepseek-ai/DeepSeek-V2
  finding: 配置中 kv_lora_rank = 512、q_lora_rank = 1536、qk_rope_head_dim = 64 与论文一致；参考代码 DeepseekV2Attention 却把展开后的整份键值写进缓存。
  matters: 用来把论文里的矩阵与代码模块逐一对号，并提醒读者：用这份参考实现测显存，测不出 MLA 省缓存的效果。
- id: roformer
  title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  a: Jianlin Su et al. (2021), preprint（后刊于 Neurocomputing，期刊版本课未核对）. arXiv:2104.09864
  url: https://arxiv.org/abs/2104.09864
  finding: 提出旋转位置编码 RoPE；其式 (16) 表明旋转后的点积里夹着 $\mathbf R_{\Theta,n-m}=\mathbf R_{\Theta,m}^\top\mathbf R_{\Theta,n}$，只依赖两个位置之差。
  matters: 正是这个夹在两个权重矩阵之间、随位置变化的旋转矩阵让 $W^{UK}$ 无法被吸收；本课 §4 的推导以这个恒等式为出发点。
- id: dsv3
  title: "DeepSeek-V3 Technical Report"
  a: DeepSeek-AI (2024), preprint. arXiv:2412.19437
  url: https://arxiv.org/abs/2412.19437
  finding: V3 沿用与 V2 维度完全相同的 MLA（128 头、$d_h=128$、$d_c=512$、$d_c'=1536$、$d_h^R=64$），层数 61、隐藏维 7168，生成时只缓存 $\mathbf c^{KV}_t$ 与 $\mathbf k^R_t$。
  matters: 说明 MLA 在下一代被原样保留，并用来算 V3 每 token 35,136 个元素的缓存；它对 MLA 质量的措辞也从“强于 MHA”改成了“与 MHA 相当”。
- id: gh-v2
  kind: artefact
  title: DeepSeek-V2 的 GitHub 说明（README）
  a: DeepSeek-AI (2024). 读取于 2026-09-24
  url: https://github.com/deepseek-ai/DeepSeek-V2
  finding: 记录 V2 于 2024-05-06 发布；说明受 Hugging Face 框架所限，开源代码在 GPU 上比内部代码慢，并推荐用专门优化的 vLLM 方案运行。
  matters: 提供“参考实现不等于生产实现”的官方说法，支撑本课对 Hugging Face 代码缓存方式的提醒。
- id: gh-v3
  kind: artefact
  title: DeepSeek-V3 的 GitHub 仓库（inference/model.py 的 MLA 类）
  a: DeepSeek-AI (2024–2025). 读取于 2026-09-24
  url: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
  finding: 官方推理代码的吸收实现把 RMSNorm 之后的潜向量 `kv_norm(kv)` 写进 kv_cache，把旋转后的 64 维位置键写进 pe_cache，打分为两段 einsum 之和再乘 softmax_scale。
  matters: 说明缓存里存的是归一化之后的潜向量，这正是 §3 的吸收与 §7 的 MQA 模式能直接拿缓存行当键值用的前提。
- id: dsv32
  title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
  a: DeepSeek-AI (2025), preprint. arXiv:2512.02556
  url: https://arxiv.org/abs/2512.02556
  finding: 附录 A 区分 MLA 的 MHA 模式与 MQA 模式，说明 V3.1-Terminus 训练和预填充用前者、解码用后者；正文说明 DSA 建立在 MLA 的 MQA 模式上，因为内核层面每个键值条目必须被多个查询共享。
  matters: 为本课 §7 “吸收 = MQA 模式”的推导提供官方出处，并把本课与第 14 课的 DSA 连起来。
- id: flashmla
  kind: artefact
  title: FlashMLA（GitHub 仓库，deepseek-ai/FlashMLA）
  a: Jiashi Li, Shengyu Liu（DeepSeek-AI）(2025–2026). 读取首次提交 414a2f3eed 与当前 README，2026-09-24
  url: https://github.com/deepseek-ai/FlashMLA
  finding: 2025-02-24 作为开源周第一天发布的 MLA 解码内核；首发在 H800 SXM5 上达到 3000 GB/s（带宽受限配置）与 580 TFLOPS（算力受限配置），后更新到 660 TFLOPS；README 把 MQA 模式定义为 head_dim_k = 576、head_dim_v = 512（V3 系列）。
  matters: 说明吸收后的 MLA 在工程上就是一个 576/512 维、单键值头的注意力内核，并给出 V3.2 的 FP8 缓存格式（每 token 每层 656 字节）。
- id: flashmla-blog
  kind: artefact
  title: "A Deep-Dive Into the New Flash MLA Kernel（FlashMLA 仓库 docs/20250422-new-kernel-deep-dive.md）"
  a: DeepSeek-AI (2025-04-22). 读取于 2026-09-24
  url: https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md
  finding: 推导 MLA 解码的计算访存比为 $h_qs_q(d_k+d_v)/d_k\approx2h_qs_q$，给出 H800 的 3.35 TB/s、990 TFLOPS 与降频后约 865 TFLOPS，并据此判断 $h_q=128$、不用张量并行时内核算力受限。
  matters: 本课屋顶线算例与图 9.12 的公式和硬件数字都出自这里；本课也指出它把阈值 129.1 取整为 128。
```
