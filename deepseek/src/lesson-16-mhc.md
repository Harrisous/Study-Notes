---
title: 第 16 课 · 残差流的新几何：Hyper-Connections 与 mHC
short: mHC 超连接
kicker: DeepSeek 科技树 · 第 16 课
description: 残差连接的恒等映射为什么必须是 1；Pre-LN 与 Post-LN 的两难；Hyper-Connections 把残差流扩成 n 条、用三个映射读写与混合；多层连乘为何在 27B 模型上把信号放大到近 3000 倍；mHC 用 Sinkhorn–Knopp 把残差映射投影到双随机矩阵（Birkhoff 多面体）上，以及让它只多花 6.7% 时间的核融合、重计算与 DualPipe 改造；DeepSeek-V4 与 V4.1-Flash 怎样使用它。
lede: 从 ResNet 到 DeepSeek-V3，主流模型里残差连接的那条“捷径”将近十年没变过：把输入原样加回去，系数恒为 1。2024 年，ByteDance 的 Hyper-Connections 动了它：把一条残差流复制成 4 条，让网络自己学流与流之间怎么混合。效果变好了，但 DeepSeek 在一个 27B 模型上发现，60 个子层的混合矩阵连乘之后，信号会被放大到近 3000 倍。mHC 的办法是只允许一种矩阵：每一行、每一列加起来都等于 1 的**双随机矩阵**。这一课讲清楚为什么是它，怎样用 20 次交替归一化把任意矩阵变成它，这件事在 GPU 上要付多少代价，以及 DeepSeek-V4 怎样把它用进 1.6T 参数的模型。
meta: 预计阅读 170 分钟，练习另需约 50 分钟 · 前置：第 1、5、12 课，数学 M06、M11
index: index.html
prev: lesson-15-v3-2-agents.html | 第 15 课 · V3.1 / V3.2
next: lesson-17-engram-ocr.html | 第 17 课 · Engram 与 OCR
---

## 这一课回答什么问题

[第 1 课](lesson-01-transformer-anatomy.html#p6)把残差流比作一条传送带：每个子层是带子旁的工位，从带上取一份、加工出一个增量、再放回带上。那一课没有追问的是：带子本身为什么能一直不出问题？这一课从这个问题出发，一路走到 DeepSeek 在 2025 年底提出、2026 年用进 V4 的 mHC。具体地说，读完之后你能回答：

- 残差连接 $\mathbf x+\mathcal F(\mathbf x)$ 里那个 $\mathbf x$ 的系数为什么必须恰好是 1？换成 1.05 或 0.95，60 层之后会怎样？Post-LN 和 Pre-LN 各自在哪里付了代价？
- Hyper-Connections（HC）把残差流扩宽成 $n$ 条之后，每个子层要做哪三件事（读、写、混合）？为什么它的初始化恰好等价于 Pre-LN？
- 一个接近单位矩阵的 $4\times4$ 混合矩阵看起来无害，为什么 60 个连乘之后能把信号放大到近 3000 倍？这个“放大倍数”怎么量？
- 双随机矩阵有哪些性质，保证连乘多少次都不会爆炸？“Birkhoff 多面体”是什么形状？只约束每行之和为什么不够？
- Sinkhorn–Knopp 迭代怎样把任意矩阵变成双随机矩阵？20 次够不够？什么时候不够？
- mHC 的参数不到全模型的千分之一、FLOPs 不到百分之一，为什么还要专门写 GPU 核？论文报告的 6.7% 额外时间从哪里来？DeepSeek-V4 与 V4.1-Flash 怎样使用它？

## 这一课在科技树上的位置

这一课在主线 E（残差流与记忆）上。这条线的前半段在第 1 课：残差连接给梯度一条直达的路，LayerNorm 从主干挪进分支（Pre-LN），让深层网络训得动（[第 1 课 §6](lesson-01-transformer-anatomy.html#p6)）；[第 5 课](lesson-05-modern-block.html#p4)把 LayerNorm 换成 RMSNorm，但“主干上只有加法”这一点没有变。从 DeepSeek LLM 到 V3.2，DeepSeek 的模型先后改了注意力（MLA、DSA）、前馈层（DeepSeekMoE）和数值格式（FP8），唯独残差流一直是那条“系数恒为 1”的单车道。

2024 年 9 月，ByteDance 的 Hyper-Connections [@zhu2025hc] 把残差流本身变成了可学习的对象：每层从哪几条流读、往哪几条流写、流与流之间怎样混合，都交给训练去定。2025 年 12 月 31 日，DeepSeek 发表 mHC [@xie2026mhc]，指出 HC 在大模型上会失稳，并给出了一个带约束的版本。它的实验模型按 DeepSeek-V3 的 MoE 架构搭建（[第 11 课](lesson-11-v3-architecture.html)），它的工程实现建立在 V3 的 DualPipe 流水线之上（[第 12 课 §6](lesson-12-v3-infrastructure.html#p6)）。之后，[第 18 课](lesson-18-v4.html)的 DeepSeek-V4 在每一个子层都使用了 mHC [@dsv4]，[第 19 课](lesson-19-v4-1-flash.html)的 V4.1-Flash 又把它的访存砍掉一半 [@dsv41]。下一课（[第 17 课](lesson-17-engram-ocr.html)）继续主线 E 的另一半：模型的“记忆”还能以什么形态存在。

## 读完你能……

1. 从残差的展开式推出“直通项”，说明恒等捷径、标量捷径（$\lambda^L$）、Post-LN（$\prod_l c_l$）、Pre-LN（系数为 1、残差流越来越粗）各自的梯度后果，并算出 $1.05^{60}$、$(1/\sqrt2)^{60}$ 这类数。
2. 写出 HC 的一个子层 $X_{l+1}=H^{\rm res}_lX_l+(H^{\rm post}_l)^\top\mathcal F_l(H^{\rm pre}_lX_l)$ 并逐项标出形状；把 HC 论文的 $(n+1)\times(n+1)$ 矩阵翻译成这三个映射；证明它的初始化等价于 Pre-LN。
3. 把 $L$ 个子层展开成复合映射 $\prod H^{\rm res}$，用行和、列和定义前向与反向增益，从论文图上印出的矩阵算出增益，并解释 3000 这个数的量级。
4. 证明双随机矩阵乘法封闭、谱范数等于 1、保持各流之和，举例说明只约束行和为什么不够，说出 $\mathcal B_4$ 的维数与顶点数。
5. 手算 Sinkhorn–Knopp 的前两步，说出收敛快慢由什么决定；算出 mHC 每个子层的参数量、访存量与重计算的最优块长，并解释 V4.1 的 Single-Pass mHC 为什么能把访存减半。

::: note 本课的记号
- $n$：**扩展率**（expansion rate），也就是残差流的条数。DeepSeek 的所有模型都取 $n=4$。
- $d$：模型隐藏维度。mHC 论文把它写成 $C$，与全课“训练算力 $C$”冲突，本课一律写 $d$。
- $X_l\in\R^{n\times d}$：第 $l$ 个子层之前的 $n$ 条残差流，**按行堆叠，每一行是一条流**（一个 $d$ 维向量）。这是全课“向量是列向量”约定的例外，出现时会再提醒。只有一条流时仍写 $\mathbf x_l\in\R^d$。
- $H^{\rm pre}_l\in\R^{1\times n}$（读出）、$H^{\rm post}_l\in\R^{1\times n}$（写入）、$H^{\rm res}_l\in\R^{n\times n}$（流间混合）：HC 与 mHC 的三个映射。mHC 论文写作 $\mathcal H^{\rm pre}_l$ 等；V4 论文写作 $A_l=H^{\rm pre}_l$、$B_l=H^{\rm res}_l$、$C_l=(H^{\rm post}_l)^\top$ [@dsv4]，其中 $B$、$C$ 与全课的 batch size、算力冲突，所以本课统一用 $H$。
- $l$：**子层**编号。每个 Transformer 块有注意力、FFN 两个子层，各算一个 $l$；$L$ 是子层总数。mHC 的 27B 模型有 30 个块，即 $L=60$ [@xie2026mhc]。
- $\mathcal F_l$：第 $l$ 个子层本身（注意力或 FFN/MoE），**包含它自己的输入归一化**（pre-norm）。
- $\mathcal B_n$：全体 $n\times n$ 双随机矩阵；$\mathbf 1_n$：$n$ 个 1 组成的列向量；$J_n=\tfrac1n\mathbf 1_n\mathbf 1_n^\top$：每个元素都是 $1/n$ 的矩阵。
- $\sigma(\cdot)$：sigmoid 函数 $1/(1+e^{-z})$，本课不用 $\sigma$ 表示标准差（标准差写 $\operatorname{std}$；随机模拟里“$s$ 倍的标准正态数”用 $s$ 表示尺度）。带下标的 $\sigma_1\ge\sigma_2\ge\cdots$ 是矩阵的奇异值（[§5](#p5)），与 sigmoid 无关。$\mathcal T_r,\mathcal T_c$：行归一化、列归一化；$t_{\max}=20$：Sinkhorn–Knopp 的迭代次数。
:::

[[legend]]

[[toc]]

# 恒等映射：残差流上的那个“1” | 问题：捷径上的系数一旦不是 1，深处就会爆炸或消失；Pre-LN 与 Post-LN 的两难 | 25

这一页只讲一件事：残差连接 $\mathbf x+\mathcal F(\mathbf x)$ 里，那个原样加回去的 $\mathbf x$ 前面的系数为什么必须是 1。这件事值得单独一页，因为后面 HC 与 mHC 做的全部工作，都是在“1”这个位置上动手：HC 把它换成一个可学习的矩阵，mHC 再把这个矩阵约束回一个“和 1 一样守恒”的集合。先把“1”的好处讲透，后面每一步改动的得失才看得清。

## 恒等捷径：展开之后的直通项

**含义。** **恒等捷径**（identity shortcut）是残差连接里那条旁路：子层的输入原封不动地加到子层的输出上，旁路本身不乘任何系数、不经过任何变换。

**理解。** 用第 1 课的传送带来想：工位只往带子上**添**东西，带子上原有的货没人碰，一路原样往下走。于是终点的货 = 起点的货 + 每个工位添上去的东西。

**数学。** 写成一个子层：

::: eq 残差连接（单条残差流） | (16.1)
$$\mathbf x_{l+1}=\mathbf x_l+\mathcal F_l(\mathbf x_l)$$
$\mathbf x_l\in\R^d$：第 $l$ 个子层之前的残差流，也就是一个 token 此刻的 $d$ 维隐藏状态；$\mathcal F_l:\R^d\to\R^d$：第 $l$ 个子层（注意力或 FFN，含自己的归一化）。He 等人与 mHC 论文都从这个式子出发 [@he2016identity; @xie2026mhc]。
:::

把 (16.1) 从第 $l$ 个子层一路用到第 $L$ 个子层：

::: deriv 把残差连接从第 l 层展开到第 L 层
$\mathbf x_{l+2}=\mathbf x_{l+1}+\mathcal F_{l+1}(\mathbf x_{l+1})=\mathbf x_l+\mathcal F_l(\mathbf x_l)+\mathcal F_{l+1}(\mathbf x_{l+1})$ || 对 $\mathbf x_{l+1}$ 再用一次 (16.1)
$\mathbf x_{L}=\mathbf x_l+\sum_{i=l}^{L-1}\mathcal F_i(\mathbf x_i)$ || 重复 $L-l$ 次，每次只多出一项增量（望远镜求和）
:::

::: eq 残差流 = 起点 + 所有增量 | (16.2)
$$\mathbf x_{L}=\mathbf x_l+\sum_{i=l}^{L-1}\mathcal F_i(\mathbf x_i)$$
任意两个子层 $l<L$ 之间，$\mathbf x_l$ 以系数 1 原样出现在 $\mathbf x_L$ 里：这就是**直通项**。
:::

反向传播时，直通项给梯度也留了一条直路。

::: pre 链式法则与雅可比矩阵 | 熟悉反向传播的可以跳过
一个函数 $\mathbf y=\mathbf f(\mathbf x)$ 把 $d$ 个数变成 $d$ 个数，它的**雅可比矩阵** $\partial\mathbf y/\partial\mathbf x$ 是一个 $d\times d$ 的表，第 $(a,b)$ 格是 $\partial y_a/\partial x_b$。两个函数复合时雅可比矩阵相乘（链式法则）；若损失 $\mathcal L$ 对 $\mathbf y$ 的梯度是列向量 $\mathbf g_{\mathbf y}$，则对 $\mathbf x$ 的梯度是 $\mathbf g_{\mathbf x}=(\partial\mathbf y/\partial\mathbf x)^\top\mathbf g_{\mathbf y}$。恒等函数 $\mathbf y=\mathbf x$ 的雅可比矩阵是单位矩阵 $I$。详见 [M06 §3](../math/m06-calculus-backprop.html#p3)；残差为什么能保住梯度，[M06 §7](../math/m06-calculus-backprop.html#p7) 有逐个数字的演示。
:::

::: deriv 直通项在反向时变成什么
$\dfrac{\partial\mathbf x_L}{\partial\mathbf x_l}=I+\dfrac{\partial}{\partial\mathbf x_l}\sum_{i=l}^{L-1}\mathcal F_i(\mathbf x_i)$ || 对 (16.2) 求雅可比：$\mathbf x_l$ 自身贡献 $I$，其余项经过各个子层
$\mathbf g_l=\Big(I+\dfrac{\partial}{\partial\mathbf x_l}\sum_{i=l}^{L-1}\mathcal F_i(\mathbf x_i)\Big)^{\!\top}\mathbf g_L$ || 链式法则；$\mathbf g_l=\partial\mathcal L/\partial\mathbf x_l\in\R^d$
$\mathbf g_l=\mathbf g_L+(\text{经过子层的项})$ || 把 $I^\top=I$ 那一项单独拿出来
:::

::: eq 梯度的直通项 | (16.3)
$$\mathbf g_l=\mathbf g_L+\Big(\frac{\partial}{\partial\mathbf x_l}\sum_{i=l}^{L-1}\mathcal F_i(\mathbf x_i)\Big)^{\!\top}\mathbf g_L$$
顶层的梯度 $\mathbf g_L$ 不经过任何权重层，原样到达第 $l$ 层。He 等人据此说：有了恒等捷径，信号可以在任意两个单元之间前向、反向直接传播 [@he2016identity]。[第 1 课 (1.27)](lesson-01-transformer-anatomy.html#p6) 的“梯度高速公路”是同一件事。
:::

## 把 1 换成 λ：标量捷径的指数律

“1”是不是必须的？He 等人在 2016 年专门做过这个实验：把捷径乘上一个标量 $\lambda_l$ [@he2016identity]。这正好是本课后面要面对的问题的最简单版本，值得完整推一遍。

::: eq 标量捷径 | (16.4)
$$\mathbf x_{l+1}=\lambda_l\,\mathbf x_l+\mathcal F_l(\mathbf x_l)$$
$\lambda_l$：第 $l$ 个子层捷径上的一个实数。$\lambda_l\equiv1$ 就回到 (16.1)。
:::

::: deriv 标量捷径的展开
$\mathbf x_{l+2}=\lambda_{l+1}\big(\lambda_l\mathbf x_l+\mathcal F_l(\mathbf x_l)\big)+\mathcal F_{l+1}(\mathbf x_{l+1})$ || 把 (16.4) 代入自己一次
$=\lambda_{l+1}\lambda_l\,\mathbf x_l+\lambda_{l+1}\mathcal F_l(\mathbf x_l)+\mathcal F_{l+1}(\mathbf x_{l+1})$ || 乘开：$\mathbf x_l$ 的系数变成两个 $\lambda$ 的乘积
$\mathbf x_L=\Big(\prod_{i=l}^{L-1}\lambda_i\Big)\mathbf x_l+\sum_{i=l}^{L-1}\Big(\prod_{k=i+1}^{L-1}\lambda_k\Big)\mathcal F_i(\mathbf x_i)$ || 重复 $L-l$ 次；每个增量也被它之后的 $\lambda$ 连乘
$\mathbf g_l=\Big(\prod_{i=l}^{L-1}\lambda_i\Big)\mathbf g_L+(\text{经过子层的项})$ || 对直通项求雅可比，得 $\big(\prod\lambda_i\big)I$
:::

::: eq 直通项变成连乘 | (16.5)
$$\mathbf x_L=\Big(\prod_{i=l}^{L-1}\lambda_i\Big)\mathbf x_l+\cdots,\qquad \mathbf g_l=\Big(\prod_{i=l}^{L-1}\lambda_i\Big)\mathbf g_L+\cdots$$
He 等人的结论：若所有 $\lambda_i>1$，这个因子会指数级变大；若所有 $\lambda_i<1$，它会指数级变小乃至消失，反向信号只能走经过权重层的路 [@he2016identity]。说“指数级”要多一个前提：各个 $\lambda_i$ 都不小于某个固定的 $\lambda>1$（或都不大于某个固定的 $\lambda<1$），这时因子至少是 $\lambda^{L-l}$（或至多是 $\lambda^{L-l}$）。只说“每个 $\lambda_i>1$”还不够，例如 $\lambda_i=1+1/i^2$ 的连乘收敛到一个有限的数。
:::

指数律的可怕之处在于“每一层看上去都很正常”。

::: ex 每层只偏 5%，60 层之后
取 $L-l=60$（mHC 的 27B 模型正好有 60 个子层）。

- $\lambda=1.05$：$1.05^{60}=18.68$。每层只放大 5%，直通信号被放大近 19 倍。
- $\lambda=1.1$：$1.1^{60}=304.5$。
- $\lambda=0.95$：$0.95^{60}=0.0461$，只剩不到 5%；$\lambda=0.9$：$0.9^{60}=0.0018$。
- 反过来问：60 层之后要放大到 3000 倍，每层需要多少？$3000^{1/60}=e^{\ln3000/60}=e^{8.006/60}=e^{0.1334}=1.1428$。每层平均只多 14.3%。

最后这个数请记住：[§3](#p3) 会看到 HC 在 27B 模型上的复合增益峰值正是“约 3000” [@xie2026mhc]。
:::

::: fig figures/l16-scalar-gain.svg | 图 16.1 每个子层把直通信号乘 g，L 个子层后是 g 的 L 次方 | scroll
**看绿线与红线的分叉：只有 $g=1$ 是水平的，其余每条线在对数坐标上都是直线，也就是指数增长或衰减。** 实线 $g>1$，虚线 $g<1$；最上面那条 $g=1.1428$ 在第 60 层到达 3000（红色点线），灰色点线是 Post-LN 式的 $g=1/\sqrt2$（下一小节），约 27 层就跌出图外。所有曲线由生成脚本按 $g^L$ 计算。
:::

He 等人的实验印证了这一点：在 110 层的 ResNet 上，把捷径乘 0.5（分支也乘 0.5），测试误差升到 12.35%；他们把缩放、门控、$1\times1$ 卷积、dropout 放在捷径上的各种做法统称为“乘性操作”，结论是这类操作会妨碍信息的传播 [@he2016identity]。

::: key 标量情形的结论
捷径上只有一个标量时，“连乘不爆炸也不消失”的唯一稳妥选择是 $\lambda\equiv1$。这不是说 $\lambda=1.01$ 一定训不动，而是说偏离 1 的代价随深度**指数**增长，没有任何余量可言。
:::

:::: quiz 小练习：多少层减半
若每个子层的捷径系数是 $\lambda=0.98$，直通信号经过多少个子层会降到原来的一半以下？
::: answer 看答案
要 $0.98^k<0.5$，即 $k>\ln0.5/\ln0.98=(-0.6931)/(-0.0202)=34.3$，所以 35 个子层。V4-Flash 有 86 个子层，走完全程只剩 $0.98^{86}=0.176$。
:::
::::

## Post-LN：传送带上的整形机

原版 Transformer 的写法是每个子层的输出为 $\LayerNorm(\mathbf x+\mathrm{Sublayer}(\mathbf x))$ [@vaswani2017]，即 LayerNorm 放在残差相加**之后**、**在主干上**（Post-LN，[第 1 课 (1.28)](lesson-01-transformer-anatomy.html#p6)）。顺带一提，mHC 论文在谈“广泛采用的 pre-norm Transformer”时把出处写成了这篇论文 [@xie2026mhc]，这是一处误引：原版是 Post-LN，Pre-LN 的分析应当看 Xiong 等人的工作 [@xiong2020preln]。从直通项的角度看，Post-LN 就是一种标量捷径。

::: deriv Post-LN 的直通系数
$\mathbf x_{l+1}=\LayerNorm\big(\mathbf x_l+\mathcal F_l(\mathbf x_l)\big)$ || Post-LN 的定义
$\approx\dfrac{\mathbf x_l+\mathcal F_l(\mathbf x_l)}{\operatorname{std}\big(\mathbf x_l+\mathcal F_l(\mathbf x_l)\big)}$ || 忽略减均值与可学习的增益、偏置，只看“除以标准差”这一步
$=c_l\,\mathbf x_l+c_l\,\mathcal F_l(\mathbf x_l)$ || 记 $c_l=1/\operatorname{std}(\mathbf x_l+\mathcal F_l)$，与 (16.4) 对比：$\lambda_l=c_l$
$c_l=\dfrac{1}{\sqrt{\Var[\mathbf x_l]+\Var[\mathcal F_l]+2\Cov[\mathbf x_l,\mathcal F_l]}}$ || 和的方差 = 方差之和 + 2 倍协方差（这里的方差指 $d$ 个分量的样本方差）
:::

::: eq Post-LN 的捷径系数 | (16.6)
$$c_l=\frac{1}{\operatorname{std}\big(\mathbf x_l+\mathcal F_l(\mathbf x_l)\big)}=\frac{1}{\sqrt{\Var[\mathbf x_l]+\Var[\mathcal F_l]+2\Cov[\mathbf x_l,\mathcal F_l]}}$$
HC 论文把 Post-Norm 写成 $n=1$ 的 HC 矩阵，其中的系数正是这个形式（原文用 $\sigma_i,\sigma_o,\sigma_{io}$ 表示输入、输出的标准差与协方差）[@zhu2025hc]。
:::

::: ex Post-LN 的直通项有多小
设输入与增量的方差都是 1、互不相关（这是笔者为算例设的数，不是论文数字），则 $c_l=1/\sqrt{1+1+0}=1/\sqrt2\approx0.707$。60 个子层之后，直通项是 $(1/\sqrt2)^{60}=2^{-30}=9.31\times10^{-10}$：从顶层直达底层的梯度信号只剩十亿分之一。

这只是“直通项”这一条路的账。LayerNorm 的雅可比不是一个标量，梯度还会经过各个子层的分支流动，所以真实网络不会这样干脆地消失；但它说明了 Post-LN 的主干在结构上**没有**一条系数为 1 的路。
:::

实测也支持这个图景。Xiong 等人证明，初始化时 Post-LN 最后一层 FFN 权重的梯度上界与层数无关、偏大，所以必须用很小的学习率热身（warm-up）[@xiong2020preln]。Liu 等人在 15 组超参数设置下比较：Pre-LN 全部收敛，Post-LN 有 7 组发散；但在 Post-LN 收敛的 8 组里，有 7 组的效果**好于** Pre-LN [@liu2020admin]。他们还指出 Post-LN 确有梯度消失，但那不是它不稳定的直接原因，Post-LN 对残差分支的依赖更强 [@liu2020admin]。

## Pre-LN：捷径干净了，流却越来越粗

Pre-LN 把 LayerNorm 挪进分支：$\mathbf x_{l+1}=\mathbf x_l+\mathcal F_l(\LayerNorm(\mathbf x_l))$。主干上只剩加法，直通系数严格为 1，(16.3) 完整成立。代价在另一头：每个子层读到的是归一化之后的输入，写回的增量尺度大致固定，而残差流是这些增量的累加，于是越来越“粗”。Xiong 等人的分析里，Pre-LN 最终那个 LayerNorm 的输入的范数平方随层数 $L$ 线性增长（他们称之为“尺度”线性增长；范数本身 $\propto\sqrt L$；这里和下表的 $L$ 沿用第 1 课与原文，是块数，即本课子层数的一半，差一个常数倍不影响这些比例关系），最后一层的梯度上界因此按 $1/\sqrt L$ 变小（[第 1 课 §6](lesson-01-transformer-anatomy.html#p6) 有定理的完整陈述）[@xiong2020preln]。

流越来越粗，意味着深处每个子层放上去的增量占整条流的比例越来越小，相邻子层看到的输入越来越像（这是笔者的直观推论）。HC 论文把这种现象称为**表示坍缩**（representation collapse），用相邻层输入的余弦相似度来展示它，并把 Pre-Norm 与 Post-Norm 的矛盾概括为梯度消失与表示坍缩之间的“跷跷板”：压下一头，另一头就翘起来 [@zhu2025hc]。

::: warn 一处引用要留心
HC 论文在谈 Pre-Norm 的表示坍缩时引用了 Liu 等人的工作，但 Liu 等人的论文里并没有“坍缩”（collapse）的说法；他们说的是 Pre-LN 对残差分支依赖较弱，常常训练出更差的模型 [@liu2020admin]。“表示坍缩”这个判断主要由 HC 自己的余弦相似度实验支撑 [@zhu2025hc]。两者方向一致，但不是同一个论断。
:::

::: fig figures/l16-pre-post-ln.svg | 图 16.2 Post-LN 与 Pre-LN：LayerNorm 在传送带上，还是在工位里 | scroll
**看两条主干上的系数：Post-LN 每一站都乘一个 $c_l$，Pre-LN 每一站都乘 1。** (a) Post-LN 把 LayerNorm（红框）装在带子上，每一站都把整条流重新归一化，带子粗细不变，但直通项是 $c_1c_2\cdots c_L$。(b) Pre-LN 把 LayerNorm 装进工位，直通项是 1，但带子越来越粗（画成越来越宽，仅为示意；Xiong 等人的引理给出的是范数平方随层数线性增长 [@xiong2020preln]）。
:::

| | Post-LN | Pre-LN |
|---|---|---|
| 主干上的直通系数 | $\prod_l c_l$，(16.6) | 恒为 1 |
| 初始化时顶层梯度（Xiong 定理 1） | $\mathcal O(d\sqrt{\ln d})$，与层数无关 | $\mathcal O(d\sqrt{\ln d/L})$ |
| 15 组设置中的收敛（Liu 等） | 8 组收敛，7 组发散 | 15 组全部收敛 |
| 收敛时谁更好（Liu 等） | 8 组中 7 组更好 | — |
| HC 所说的代价 | 梯度消失 | 表示坍缩 |

## 这一课要回答的问题

两种做法都是**人手定死**的：Pre-LN 把捷径系数定为 1，Post-LN 把它定为“除以标准差”。一个自然的想法是：能不能让网络自己学捷径该怎么走？危险也一眼可见：一旦系数可学，它就不再恰好是 1，(16.5) 的指数律立刻回来。接下来的几页按这个顺序走：先放开（HC），再看它坏在哪里，最后加上约束（mHC）。

1. **HC**（[§2](#p2)）：把残差流复制成 $n$ 条，让每个子层用可学习的系数决定从哪几条读、往哪几条写、流与流之间怎么混合。表达力大增，$n=1$ 时的 Pre-LN、Post-LN 都是它的特例。
2. **失稳**（[§3](#p3)）：可学的混合矩阵连乘 60 次，在 27B 模型上放大到近 3000 倍。
3. **mHC**（[§4](#p4)–[§6](#p6)）：只允许一类矩阵，它们像“1”一样连乘多少次都守恒，但又允许流与流之间交换信息。

::: ana 类比：残差流 = 传送带
**映射。** 残差流是一条传送带，带上的货是一个 token 的 $d$ 维隐藏状态；每个子层是带子旁的工位，读一份、加工、把增量放回带上；恒等映射 = 没人动的货原样往下走。标量捷径 = 每一站都有一台机器把带上所有的货放大 $\lambda$ 倍；Post-LN = 每一站都有一台“整形机”把整条带子上的货重新压成标准大小；Pre-LN = 整形机搬进了工位，只整形工位读到的那一份，带子上的货只增不减。

**收益。** 它让 (16.5) 一眼可见：每站都放大 5% 的带子，走 60 站就放大 19 倍；它也让 Pre-LN 的代价一眼可见：货只增不减，后来的工位放上去的那一点越来越不起眼。

**失效的地方。** 真实的“货”是一个向量，“放回”是向量加法：两个工位放上去的增量可以互相抵消，也可以叠加成一个新方向，这是实物传送带没有的。“整形机”也不是按一个固定倍数缩放，它的倍数取决于当时带上的货本身（$c_l$ 依赖 $\mathbf x_l$）。
:::

::: key 本页要点
- 恒等捷径让 $\mathbf x_l$ 以系数 1 出现在任意深处的 $\mathbf x_L$ 里 (16.2)，梯度也原样回传 (16.3)。
- 捷径乘标量 $\lambda$，直通项就变成 $\prod\lambda_i$ (16.5)：$1.05^{60}=18.7$，$0.95^{60}=0.046$；60 层放大 3000 倍只需每层 1.1428 倍。
- Post-LN 的主干等价于每层乘 $c_l=1/\operatorname{std}(\mathbf x_l+\mathcal F_l)$ (16.6)，训练易发散、要 warm-up；Pre-LN 的直通系数为 1，但残差流越来越粗，深层增量占比变小。Liu 等人的 15 组实验：Pre-LN 全部收敛，Post-LN 收敛时 8 组中 7 组更好。
- 本课的问题：能不能让捷径变得可学，又不让连乘爆炸？
:::

# Hyper-Connections：一条传送带变成 n 条 | 把残差流复制成 n 份，用读、写、混合三个映射连接子层；静态与动态、初始化与特例、收益与代价 | 30

Hyper-Connections（超连接，HC）来自 ByteDance Seed 团队，2024 年 9 月发表于 arXiv，后被 ICLR 2025 接收 [@zhu2025hc; @iclr2025-hc-page]。它想同时摆脱跷跷板的两头：不在“系数固定为 1”和“主干上做归一化”之间二选一，而是让网络学出每一层的连接强度；为了让“学连接”有足够的自由度，它把一条残差流扩宽成 $n$ 条。注意 HC 不是 DeepSeek 的发明，DeepSeek 的工作是下几页的 mHC。

## 扩宽：把隐藏状态复制成 n 份

**含义。** **扩展率** $n$ 是残差流的条数。嵌入层输出一个 $d$ 维向量 $\mathbf h^0$ 之后，HC 把它复制 $n$ 份，按行堆成一个 $n\times d$ 的矩阵，作为第一个子层之前的残差流 $X_1$。在最后一个子层之后，把 $n$ 行**加起来**，再送进最终的归一化与输出层 [@zhu2025hc]。

**理解。** 一条传送带变成 $n$ 条并排的传送带，一开始每条带上放的是同一件货的副本。每个工位不再只对着一条带子，而是可以从几条带子上各取一点、把加工结果分发到几条带子上；带子之间还有一个调度员，把货在带子之间挪来挪去。

**数学。** 每个子层要做三件事，对应三个映射：

::: eq Hyper-Connections 的一个子层 | (16.7)
$$X_{l+1}=H^{\rm res}_l\,X_l+(H^{\rm post}_l)^\top\,\mathcal F_l\big(H^{\rm pre}_l\,X_l\big)$$
$X_l\in\R^{n\times d}$：$n$ 条流，每行一条。**读出** $H^{\rm pre}_l\in\R^{1\times n}$：把 $n$ 条流加权合成子层的输入（$1\times n$ 乘 $n\times d$ 得 $1\times d$）。**写入** $H^{\rm post}_l\in\R^{1\times n}$：把子层的输出（$1\times d$）按权重分发到 $n$ 条流（$n\times1$ 乘 $1\times d$ 得 $n\times d$，一个外积）。**混合** $H^{\rm res}_l\in\R^{n\times n}$：流与流之间的线性组合。这是 mHC 论文对 HC 的写法 [@xie2026mhc]。
:::

逐条流写开，第 $j$ 条新流是：

::: eq 逐条流的写法 | (16.8)
$$\mathbf x_{l+1,j}=\sum_{i=1}^{n}(H^{\rm res}_l)_{ji}\,\mathbf x_{l,i}+(H^{\rm post}_l)_j\;\mathcal F_l\Big(\sum_{i=1}^{n}(H^{\rm pre}_l)_i\,\mathbf x_{l,i}\Big)$$
$\mathbf x_{l,i}$ 是 $X_l$ 的第 $i$ 行。$(H^{\rm res}_l)_{ji}$ 读作“旧流 $i$ 流进新流 $j$ 的比例”：$H^{\rm res}_l$ 的**第 $j$ 行**决定新流 $j$ 由哪些旧流组成，**第 $i$ 列**决定旧流 $i$ 被分到哪些新流。
:::

注意子层 $\mathcal F_l$ 本身一点没变：它仍然吃一个 $d$ 维向量、吐一个 $d$ 维向量。mHC 论文特别强调，HC 不改变单个子层的计算量 [@xie2026mhc]。多出来的只有三个很小的矩阵和一个 $n$ 倍宽的残差流。

::: ex 手算一个 n = 2 的子层
取 $n=2$、$d=3$（数字是笔者为演示选的）。两条流 $\mathbf x_{l,1}=(1,0,2)$，$\mathbf x_{l,2}=(3,1,0)$。三个映射：$H^{\rm pre}=(0.5,0.5)$，$H^{\rm post}=(1,1)$，$H^{\rm res}=\begin{pmatrix}0.9&0.1\\0.2&0.8\end{pmatrix}$。

1. **读出**：子层输入 $=0.5\times(1,0,2)+0.5\times(3,1,0)=(2,\ 0.5,\ 1)$。
2. **子层**：假设 $\mathcal F_l$ 对这个输入吐出 $(0.1,\ -0.2,\ 0.3)$。
3. **混合**：新流 1 的残差部分 $=0.9\times(1,0,2)+0.1\times(3,1,0)=(1.2,\ 0.1,\ 1.8)$；新流 2 $=0.2\times(1,0,2)+0.8\times(3,1,0)=(2.6,\ 0.8,\ 0.4)$。
4. **写入**：两条流各加上 $1\times(0.1,-0.2,0.3)$，得 $\mathbf x_{l+1,1}=(1.3,\ -0.1,\ 2.1)$，$\mathbf x_{l+1,2}=(2.7,\ 0.6,\ 0.7)$。

再看一个后面要用的量：两条流的和。混合之前是 $(1,0,2)+(3,1,0)=(4,1,2)$，只做混合之后是 $(1.2,0.1,1.8)+(2.6,0.8,0.4)=(3.8,0.9,2.2)$，**变了**。原因是 $H^{\rm res}$ 每一行的和都是 1，但两列的和是 $1.1$ 和 $0.9$：旧流 1 被“多分了”10%，旧流 2 被“少分了”10%。[§4](#p4) 会看到，要让这个和不变，列和也必须是 1。
:::

::: fig figures/l16-paradigms.svg | 图 16.3 残差连接、Hyper-Connections 与 mHC 的一个子层（据 mHC 论文图 1 重绘） | wide scroll
**看三幅图里“带子”的条数与方框的颜色。** (a) 普通残差：一条流，捷径系数固定为 1。(b) HC：四条流（$n=4$），读出 $H^{\rm pre}$（$1\times4$）把四条流合成子层输入，写入 $H^{\rm post}$（$1\times4$）把输出分发回四条流，混合 $H^{\rm res}$（$4\times4$）在流之间调度；三个映射都是可学的任意实数（红框）。(c) mHC：结构与 HC 相同，但三个映射都加了约束（绿框），$H^{\rm res}$ 被投影成双随机矩阵（[§4](#p4)、[§5](#p5)）。据 mHC 论文图 1 重绘 [@xie2026mhc]，形状标注按本课记号。
:::

## HC 论文的矩阵写法

HC 论文把一个子层的全部连接权重排成一个 $(n+1)\times(n+1)$ 的矩阵 [@zhu2025hc]：

::: eq HC 矩阵 | (16.9)
$$\mathcal{HC}=\begin{pmatrix}0&\mathbf B\\ \mathbf A_m&\mathbf A_r\end{pmatrix}=\begin{pmatrix}0&\beta_1&\beta_2&\cdots&\beta_n\\ \alpha_{1,0}&\alpha_{1,1}&\alpha_{1,2}&\cdots&\alpha_{1,n}\\ \vdots&\vdots&\vdots&\ddots&\vdots\\ \alpha_{n,0}&\alpha_{n,1}&\alpha_{n,2}&\cdots&\alpha_{n,n}\end{pmatrix}$$
$\mathbf B\in\R^{1\times n}$：写入权重；$\mathbf A_m\in\R^{n\times1}$：读出权重；$\mathbf A_r\in\R^{n\times n}$：流间权重。与 (16.7) 的对应是 $H^{\rm pre}=\mathbf A_m^\top$，$H^{\rm post}=\mathbf B$，$H^{\rm res}=\mathbf A_r^\top$（研究笔记逐项核对过形状）。这里的 $\alpha,\beta$ 是矩阵元素，和后面 mHC 的门控标量 $\alpha^{\rm pre}_l$ 不是一回事。
:::

为什么 $H^{\rm res}$ 是 $\mathbf A_r$ 的**转置**？HC 论文的更新式是 $\hat{\mathbf H}=\mathbf B^\top\mathcal T(\mathbf H^\top\mathbf A_m)^\top+\mathbf A_r^\top\mathbf H$（$\mathcal T$ 是子层，$\mathbf H$ 即本课的 $X_l$）[@zhu2025hc]，最后一项 $\mathbf A_r^\top\mathbf H$ 与 (16.7) 的 $H^{\rm res}X_l$ 对照即得。于是 $\alpha_{i,j}$（$i,j\ge1$）是“旧流 $i$ 流进新流 $j$”的权重。左上角恒为 0：子层的输出不直接喂回它自己的输入。

::: fig figures/l16-hc-matrix.svg | 图 16.4 HC 矩阵的分块读法、初始化，以及旧结构作为特例 | wide scroll
**看 (a) 的三种颜色：绿色第 0 行是写入，蓝色第 0 列是读出，紫色右下块是流间混合。** (b) 初始化时写入全为 1、只读一条流、流间为单位矩阵，结果等价于 Pre-Norm 残差（下面推导）。(c) 旧结构都可以写成固定不变的 HC 矩阵：$n=1$ 时的 Pre-Norm 与 Post-Norm（$c$ 即 (16.6)），以及 $n=2$ 时两层的“串行”与“并行”排布。据 HC 论文式 (1)、(14)–(19) 与图 2、图 4 绘制 [@zhu2025hc]。
:::

## 静态 HC 与动态 HC

**静态 HC**（SHC）：矩阵 (16.9) 的每个元素都是一个可学习的常数，对所有 token 相同。**动态 HC**（DHC）：在静态矩阵上加一个依赖当前输入的小扰动。HC 论文的做法是先把每条流归一化，再做一次线性变换，经过 tanh，最后乘一个初值很小的可学习缩放因子 [@zhu2025hc]：

::: eq 动态 HC（HC 论文式 (10)–(13)） | (16.10)
$$\begin{aligned}\overline{\mathbf H}&=\operatorname{norm}(\mathbf H),\\ \mathcal B(\mathbf H)&=s_\beta\circ\tanh(\overline{\mathbf H}\mathbf W_\beta)^\top+\mathbf B,\\ \mathcal A_m(\mathbf H)&=s_\alpha\circ\tanh(\overline{\mathbf H}\mathbf W_m)+\mathbf A_m,\\ \mathcal A_r(\mathbf H)&=s_\alpha\circ\tanh(\overline{\mathbf H}\mathbf W_r)+\mathbf A_r.\end{aligned}$$
$\overline{\mathbf H}\in\R^{n\times d}$：逐条流归一化后的残差流；$\mathbf W_\beta,\mathbf W_m\in\R^{d\times1}$，$\mathbf W_r\in\R^{d\times n}$：可学习投影，**初始化为 0**；$s_\alpha,s_\beta$：可学习标量，论文伪代码里初值 0.01。因为 $\overline{\mathbf H}\mathbf W_r$ 逐行作用，$\mathcal A_r$ 的第 $i$ 行只由第 $i$ 条流决定。这里的 $\mathcal B(\mathbf H)$、$\mathcal A_m(\mathbf H)$、$\mathcal A_r(\mathbf H)$ 沿用 HC 论文的写法，表示“依赖输入的写入、读出、流间权重”，与 [§4](#p4) 的双随机矩阵集合 $\mathcal B_n$ 无关。
:::

tanh 把扰动限制在 $(-1,1)$ 之内，再乘上 0.01 左右的 $s$，动态部分一开始只是静态矩阵上的一点涟漪。HC 论文把带 tanh 的 DHC 作为默认方法，$n=4$ 时 DHC 明显好于 SHC，$n=2$ 时二者相近 [@zhu2025hc]。

## 初始化：HC 一开始就是 Pre-LN

HC 的初始化让它在训练第一步时与 Pre-LN 残差**完全等价** [@zhu2025hc]：动态投影全为 0；第 $k$ 个子层的静态部分取 $\mathbf B=\mathbf 1_{1\times n}$（写回所有流），$\mathbf A_m=\mathbf e_{k\bmod n}$（只读第 $k\bmod n$ 条流，$\mathbf e_i$ 是第 $i$ 个位置为 1、其余为 0 的列向量；这里沿用 HC 论文的写法，流从 0 编号到 $n-1$，换成本课从 1 到 $n$ 的编号就是第 $(k\bmod n)+1$ 条流），$\mathbf A_r=I$（流之间不混合）。

::: deriv 为什么这就是 Pre-LN
$X_1=\mathbf 1_n\,\mathbf x_1^\top$ || 嵌入之后复制 $n$ 份：每一行都是同一个 $\mathbf x_1^\top$
$H^{\rm pre}X_l=\mathbf e_{k\bmod n}^\top X_l=\mathbf x_l^\top$ || 若 $X_l$ 的每一行都等于 $\mathbf x_l^\top$，读哪一条都一样
$H^{\rm res}X_l=I\,X_l=\mathbf 1_n\mathbf x_l^\top$ || 单位矩阵不改变任何一行
$(H^{\rm post})^\top\mathcal F_l(\mathbf x_l)^\top=\mathbf 1_n\,\mathcal F_l(\mathbf x_l)^\top$ || $\mathbf B=\mathbf 1_{1\times n}$：同一个输出加到每一行
$X_{l+1}=\mathbf 1_n\big(\mathbf x_l+\mathcal F_l(\mathbf x_l)\big)^\top$ || 相加：每一行仍然相同，且都等于 Pre-LN 的 $\mathbf x_{l+1}$（$\mathcal F_l$ 内含 pre-norm）
$\mathbf 1_n^\top X_L=n\,\mathbf x_L^\top$ || 归纳得 $X_L=\mathbf 1_n\mathbf x_L^\top$；最后把 $n$ 行求和，$n$ 倍的整体缩放被最终归一化抵消
:::

::: eq HC 的初始化 | (16.11)
$$\mathcal{HC}^{(k)}_{\rm init}=\begin{pmatrix}0&\mathbf 1_{1\times n}\\ \mathbf e_{k\bmod n}&I_{n}\end{pmatrix}\quad\Longrightarrow\quad X_l=\mathbf 1_n\,\mathbf x_l^\top,\ \ \mathbf x_{l+1}=\mathbf x_l+\mathcal F_l(\mathbf x_l)$$
训练从一个已知能训的起点（Pre-LN）出发，然后再去学更好的连接。
:::

这个初始化是一个**选择**，不是推论。它的好处是训练第一步没有任何风险；代价是 $n$ 条流一开始完全相同，流与流之间的差异只能靠后来学出的 $H^{\rm res}$、$H^{\rm post}$ 慢慢“掰开”。

## 旧结构都是“不可训练的 HC”

把 $n$ 取 1，(16.9) 只剩一个 $2\times2$ 矩阵，Pre-Norm 与 Post-Norm 都能写进去 [@zhu2025hc]：

::: eq Pre-Norm 与 Post-Norm 作为 n = 1 的 HC | (16.12)
$$\mathcal{HC}_{\rm Pre}=\begin{pmatrix}0&1\\1&1\end{pmatrix},\qquad \mathcal{HC}_{\rm Post}=\begin{pmatrix}0&c_l\\1&c_l\end{pmatrix}$$
Pre：读 1、写 1、捷径 1，即 $\mathbf x+\mathcal F(\mathbf x)$。Post：读 1、写 $c_l$、捷径 $c_l$，即 $c_l(\mathbf x+\mathcal F(\mathbf x))$，$c_l$ 就是 (16.6)。
:::

$n=2$ 时，HC 还能表示两种层排布：常规的“串行”（每层接着上一层）和“并行”（相邻两层读同一个输入、输出相加，类似并行 Transformer 块）。HC 论文给出了这两种排布各自对应的固定 HC 矩阵（其式 (17)–(19)），训练学到的可以是它们之间的软混合 [@zhu2025hc]。论文可视化学到的连接权重后发现，它们呈 Λ 形：既有 Post-Norm 式“离得越远权重越小”的衰减，又有 Pre-Norm 式“底层被后面各层频繁读取”的模式，论文称之为 Pre-Norm 与 Post-Norm 的自由混合 [@zhu2025hc]。

## 收益

HC 论文在 OLMo 系列上做实验，全部训练 500B tokens，默认设置为带 tanh 的 DHC、$n=4$ [@zhu2025hc]。表里的“V2”是 HC 论文表格中一组验证集的名字（论文还报告另一组，叫 V3）；loss 与 PPL（困惑度，第 2 课）都是越低越好，下游分数越高越好。

| 实验 | 基线 | HC | 说明 |
|---|---|---|---|
| OLMo-1B，$n=1$（DHC） | V2 验证 loss 2.811 | 2.819 | 只有一条流时反而更差 |
| OLMo-1B，扩展率消融 | — | $n=4$ 最好 | $n=8$ 几乎不再提升 |
| OLMo-7B，DHC×4 | V2 loss 2.581，PPL 14.316 | 2.559，14.023 | 下游平均 70.1 → 71.0；13.36 → 13.38 GFLOPs/token |
| OLMoE-1B-7B，DHC×4 | ARC-Challenge 41.8，MMLU Var 38.5 | 47.8，39.7 | 收敛快 1.8 倍；C4-en 验证 loss 低 0.028 |

两个观察对后面很重要。第一，$n=1$ 时 HC 没有收益，收益来自**多条流**。第二，HC 论文报告的是 HC **更稳**：所有 DHC 实验都没有观察到 loss 尖峰，7B 基线有频繁尖峰而 DHC 没有 [@zhu2025hc]。下一页 DeepSeek 在更大的模型上看到了相反的现象，这两件事并不矛盾，但说明“HC 稳不稳”要带上规模与出处来说。

## 代价：参数与 FLOPs 是零头，显存不是

HC 论文附录 B（v2 起才有）给出了代价的公式与实测 [@zhu2025hc]。每个 HC 模块的参数：静态 $n(n+2)$ 个；动态再加上三个投影的 $d(n+2)$ 个、两个缩放因子与归一化参数。每个 Transformer 块有两个 HC 模块。

::: ex HC 的参数账（OLMo-1B：d = 2048，16 个块）
- SHC×4：每模块 $4\times(4+2)=24$ 个，全模型 $24\times2\times16=768$ 个。
- DHC×4：每模块 $2048\times6+24+2=12{,}314$ 个，全模型 $12{,}314\times32=394{,}048$ 个（这是论文自己的算例，没有计入归一化参数）。

一个 1B 模型多出不到 40 万个参数，约 0.03%。FLOPs 同样是零头：论文表 8 的增幅在 0.038% 到 0.208% 之间。但实测显存（8 张 GPU）增加了 9.7% 到 28.28%：残差流宽了 $n$ 倍，反向传播要存的激活也跟着变多。论文给出的额外激活显存是 $2n\,T\,b\,d\,L_b$ 量级（$T$ 序列长、$b$ 微批大小、$L_b$ 块数），用重计算可以降到 $n\,T\,b\,d$ [@zhu2025hc]。
:::

::: warn 引用 HC 参数量时用公式
HC 论文表 7 的“HC Params(B)”一列有笔误：SHC 两行是公式值的 10 倍，OLMo-7B 一行也与总参数差对不上（研究笔记 E8 有逐项复算）；该表的“总参数”列与百分比列是自洽的。本课的参数数字一律按公式计算。
:::

“FLOPs 几乎为零、显存和访存却不小”这个反差，是 [§6](#p6) 基础设施一页的起点。

::: ana 把传送带类比扩成 n 条
**映射。** $n$ 条并排的传送带 = $n$ 条残差流；工位从几条带子上各取一点拼成要加工的料 = 读出 $H^{\rm pre}$；工位把成品按比例分发到几条带子上 = 写入 $H^{\rm post}$；带子之间的调度员把货从一条带挪到另一条 = 混合 $H^{\rm res}$。初始化时每条带上都是同一件货的副本，调度员什么也不做。

**收益。** 它说明了为什么 $n=1$ 时 HC 没有收益：只有一条带子，调度员无事可做，剩下的只是“调一调捷径系数”，而那正是上一页说的危险操作。

**失效的地方。** HC 的调度员可以用**负**系数，也可以把一条带子上的货“复制”到好几条带子上而不减少原带上的量（$H^{\rm res}$ 的列和可以大于 1），这在实物传送带上做不到，而且正是下一页问题的根源。带子也没有固定的“语义分工”，哪条带子装什么完全是学出来的。
:::

::: key 本页要点
- HC 把残差流复制成 $n$ 条（$X_l\in\R^{n\times d}$），每个子层做三件事：读出 $H^{\rm pre}\in\R^{1\times n}$、写入 $H^{\rm post}\in\R^{1\times n}$、流间混合 $H^{\rm res}\in\R^{n\times n}$ (16.7)；子层本身不变。
- HC 论文的 $(n+1)\times(n+1)$ 矩阵 (16.9) 与三个映射的对应：$H^{\rm pre}=\mathbf A_m^\top$，$H^{\rm post}=\mathbf B$，$H^{\rm res}=\mathbf A_r^\top$。动态 HC 用“归一化 → 线性 → tanh → 乘 0.01 起步的缩放”生成输入相关的扰动 (16.10)。
- 初始化（写全 1、读一条、混合为 $I$）使 HC 与 Pre-LN 完全等价 (16.11)；Pre-Norm、Post-Norm、串行与并行排布都是固定的 HC 矩阵 (16.12)。
- 收益来自多条流：$n=1$ 无益，$n=4$ 最好；OLMoE-1B-7B 收敛快 1.8 倍。代价：参数约 0.03%、FLOPs 约 0.2%，显存却多 10%–28%。
:::

# HC 的隐患：复合映射的增益 | 60 个 4×4 矩阵连乘：27B 模型上的 loss 突增、近 3000 倍的增益，以及为什么不能把它删掉 | 30

HC 论文在 1B、7B 的模型上没有观察到任何 loss 尖峰。DeepSeek 用 HC 训练一个 27B 的 MoE 模型（按 DeepSeek-V3 的架构搭建，$n=4$，60 个子层）时，却在第 12k 步前后看到了一次意外的 loss 突增，而且与梯度范数的不稳定高度相关 [@xie2026mhc]。这一页回答三个问题：这种不稳定从结构上从哪里来？怎样用一个数量出它有多严重？问题出在最有用的那个映射上，能不能直接删掉？

## 展开多层：连乘又回来了

把 (16.7) 像 [§1](#p1) 那样一层层展开：

::: deriv 把 HC 从第 l 个子层展开到第 L 个子层
$X_{l+1}=H^{\rm res}_lX_l+(H^{\rm post}_l)^\top\mathcal F_l(H^{\rm pre}_lX_l)$ || (16.7)
$X_{l+2}=H^{\rm res}_{l+1}H^{\rm res}_lX_l+H^{\rm res}_{l+1}(H^{\rm post}_l)^\top\mathcal F_l(\cdot)+(H^{\rm post}_{l+1})^\top\mathcal F_{l+1}(\cdot)$ || 代入下一层的 (16.7)
$X_L=\big(H^{\rm res}_{L-1}\cdots H^{\rm res}_{l}\big)X_l+\sum_{i=l}^{L-1}\big(H^{\rm res}_{L-1}\cdots H^{\rm res}_{i+1}\big)(H^{\rm post}_i)^\top\mathcal F_i(H^{\rm pre}_iX_i)$ || 重复 $L-l$ 次
:::

每一层的 $H^{\rm res}$ 都乘在左边，越新的越靠左；第 $i$ 个子层的输出写进残差流之后，还要被它之后所有的 $H^{\rm res}$ 依次混合。

::: eq HC 的多层展开（mHC 论文式 (4)） | (16.13)
$$X_L=\Big(\prod_{i=1}^{L-l}H^{\rm res}_{L-i}\Big)X_l+\sum_{i=l}^{L-1}\Big(\prod_{j=1}^{L-1-i}H^{\rm res}_{L-j}\Big)(H^{\rm post}_i)^\top\mathcal F_i\big(H^{\rm pre}_iX_i\big)$$
这是 mHC 论文式 (4) 换成本课记号的写法 [@xie2026mhc]。$\prod_{i=1}^{L-l}H^{\rm res}_{L-i}=H^{\rm res}_{L-1}H^{\rm res}_{L-2}\cdots H^{\rm res}_{l}$（$i=1$ 的因子在最左）。本课把第 $l$ 到第 $L-1$ 个子层的**复合映射**简记为 $P_{l\to L}$。
:::

对照 (16.5)：只有一条流时，$H^{\rm res}_l$ 退化成一个数 $\lambda_l$，(16.13) 就是 He 等人 2016 年的标量连乘。mHC 要解决的，正是这个十年前就被指出的问题的**矩阵版**：残差流的直通项不再是 $1$，而是 60 个 $4\times4$ 矩阵的乘积。

反向也一样。设 $G_l=\partial\mathcal L/\partial X_l\in\R^{n\times d}$ 是损失对第 $l$ 个子层之前残差流的梯度（形状与 $X_l$ 相同）。

::: pre 线性映射 Y = HX 的梯度 | 会矩阵求导的可以跳过
$Y=HX$，$H\in\R^{n\times n}$ 固定，$X\in\R^{n\times d}$。逐元素写：$Y_{jc}=\sum_iH_{ji}X_{ic}$，所以 $\partial\mathcal L/\partial X_{ic}=\sum_j(\partial\mathcal L/\partial Y_{jc})H_{ji}=(H^\top G_Y)_{ic}$。一句话：**前向左乘 $H$，反向左乘 $H^\top$**。线性层反向的一般形式见 [M06 §6](../math/m06-calculus-backprop.html#p6)。
:::

::: eq 反向的直通项 | (16.14)
$$G_l=P_{l\to L}^{\top}\,G_L+(\text{经过各子层分支的项}),\qquad P_{l\to L}^\top=(H^{\rm res}_l)^\top(H^{\rm res}_{l+1})^\top\cdots(H^{\rm res}_{L-1})^\top$$
前向，信号被 $P_{l\to L}$ 左乘；反向，梯度被 $P_{l\to L}^\top$ 左乘。一个放大了，另一个多半也好不了。
:::

## 怎样量“放大了多少”：Amax 增益

$P_{l\to L}$ 是一个 $4\times4$ 矩阵，“它把信号放大了多少”需要压成一个数。mHC 论文的选择叫 **Amax 增益**（Amax Gain Magnitude）：前向取复合映射**行和的绝对值**的最大者，反向取**列和的绝对值**的最大者，再对一条选定序列的所有 token 取平均 [@xie2026mhc]。

::: eq Amax 增益 | (16.15)
$$g_{\rm fwd}(P)=\max_{i}\Big|\sum_{j=1}^{n}P_{ij}\Big|,\qquad g_{\rm bwd}(P)=\max_{j}\Big|\sum_{i=1}^{n}P_{ij}\Big|$$
$P\in\R^{n\times n}$：单层的 $H^{\rm res}_l$，或多层的复合映射。$g=1$ 表示“不放大也不缩小”。
:::

为什么是行和与列和？看一个最典型的输入：$n$ 条流装着同一个向量。

::: deriv 行和 = 公共信号的放大倍数
$X=\mathbf 1_n\mathbf x^\top$ || 每条流都是同一个 $\mathbf x^\top$（初始化时正是如此）
$PX=(P\mathbf 1_n)\,\mathbf x^\top$ || 矩阵乘法结合律
$(PX)_{i,:}=\Big(\sum_jP_{ij}\Big)\mathbf x^\top$ || $P\mathbf 1_n$ 的第 $i$ 个元素就是第 $i$ 行的和
$G=\mathbf 1_n\mathbf g^\top\ \Rightarrow\ P^\top G=(P^\top\mathbf 1_n)\,\mathbf g^\top$ || 反向同理：每条流收到同一个梯度 $\mathbf g^\top$ 时，放大倍数是列和
:::

::: warn Amax 增益不是矩阵的 ∞-范数
矩阵的 ∞-范数是“每行**绝对值**之和”的最大者，Amax 增益是“每行之和”的**绝对值**的最大者。前者不允许正负抵消，后者允许。对元素全部非负的矩阵（例如 mHC 的双随机矩阵）两者相同；对 HC 的实数矩阵，两者可以差很多（下面的算例）。研究笔记用论文图 8 的数字核实过，论文用的是后一种 [@xie2026mhc]。
:::

::: ex 从论文图 8 读出 HC 第 1 个子层的增益
mHC 论文图 8 印出了 27B 模型 HC 第 1 个子层的 $H^{\rm res}_1$（对一条序列的 token 取平均，两位小数）[@xie2026mhc]：

$$H^{\rm res}_1=\begin{pmatrix}5.43&4.43&4.43&4.43\\-4.07&-3.07&-4.07&-4.07\\-3.95&-3.95&-2.95&-3.95\\-4.22&-4.22&-4.22&-3.22\end{pmatrix}$$

- 行和：$5.43+3\times4.43=18.72$（图上标 18.73，舍入差），其余三行 $-15.28$、$-14.80$、$-15.88$。前向增益 $=18.72$。
- 列和：第 1 列 $5.43-4.07-3.95-4.22=-6.81$，四列都是 $-6.81$。反向增益 $=6.81$。
- 两者与论文图 3 在 $l=1$ 处的前向 18.7、反向 6.81 一致。若误用“绝对值之和”，第 1 列得 $5.43+4.07+3.95+4.22=17.67$，对不上。

还有一个值得多看一眼的结构（笔者的读法，论文没有讨论）：$H^{\rm res}_1-I$ 的四列完全相同，即 $H^{\rm res}_1=I+\mathbf u\mathbf 1^\top$，$\mathbf u=(4.43,-4.07,-3.95,-4.22)^\top$。第 1 个子层之前四条流都是同一个 $\mathbf x$，于是第 $i$ 条流变成 $\mathbf x+u_i\cdot4\mathbf x=(1+4u_i)\mathbf x$，四个倍数正是四个行和 $18.72,-15.28,-14.80,-15.88$。网络学会的第一件事，是用悬殊的尺度和相反的符号把四份相同的副本“掰开”。
:::

## 27B 上看到的：单层温和，复合爆炸

::: fig figures/l16-hc-gain.svg | 图 16.5 HC 在 27B 模型上的单层增益与复合增益（据 mHC 论文图 3 重绘，数值为数字化读数） | wide scroll
**看左右两图纵轴的量级差：单层增益大多在 1 附近，复合增益却到了几千。** (a) 每个子层自己的 $H^{\rm res}_l$：除了首尾几层（$l=1$ 前向 18.7，$l=59,60$ 为 9.0 与 22.5），前向增益都在 0.64 到 2.81 之间。(b) 复合映射：与文字同色的那条线是第 1 到第 $l$ 个子层之积的前向增益，全部 60 层之积为 509；红线是第 $l$ 到第 60 个子层之积的反向增益，在 $l=52$ 附近达到约 2825（论文正文说“峰值约 3000”）。曲线数值由研究笔记从论文矢量图中按网格标定读出，是数字化读数而非论文印刷的数字 [@xie2026mhc]。
:::

对照 [§1](#p1) 的算例：60 层放大 3000 倍，平均每层只要 1.1428 倍，而左图里的单层增益确实大多就在这个范围里上下浮动。矩阵连乘比标量连乘更隐蔽：单层的增益在 1 附近有涨有落，乘积却可以一路往上走。原因是 Amax 增益只在一个方向上量：“所有流装同一个信号”。一层把信号的一部分转到别的方向（例如“流 1 与流 2 之差”）之后，下一层在那个方向上放大多少，单层的 Amax 增益完全看不到。下面的算例把这件事拆开来看。

::: ex 每一层的增益都是 1，60 层之后却是 24.8（笔者构造的算例）
取 $n=2$，两种矩阵交替出现（数字是笔者为演示选的）：

$$A=\begin{pmatrix}1.1&-0.1\\-0.1&1.1\end{pmatrix},\qquad B=\begin{pmatrix}1&0\\0&0.8\end{pmatrix}.$$

- **单层看都无害。** $A$ 的行和、列和都是 1，所以 $g_{\rm fwd}(A)=g_{\rm bwd}(A)=1$；$B$ 的行和、列和都是 $1$ 与 $0.8$，所以增益也是 1，而且 $B$ 看上去还在“缩小”。
- **把输入拆成两个方向。** 记 $\mathbf c=(1,1)^\top$（两条流相同的“公共”方向）、$\mathbf e=(1,-1)^\top$（两条流相反的“差异”方向）。$A\mathbf c=\mathbf c$，$A\mathbf e=(1.2,-1.2)^\top=1.2\,\mathbf e$：$A$ 不碰公共方向，却把差异方向放大 1.2 倍。$B$ 则把公共信号“漏”一部分到差异方向：$B\mathbf c=(1,\,0.8)^\top=0.9\,\mathbf c+0.1\,\mathbf e$。
- **跟着一个全 1 输入走两轮。** 先 $B$ 后 $A$：$(1,1)\to(1,\,0.8)\to(1.02,\,0.78)$，差异分量从 $0.1$ 变成 $0.12$；再来一轮：$\to(1.02,\,0.624)\to(1.0596,\,0.5844)$，也就是 $0.822\,\mathbf c+0.2376\,\mathbf e$。公共分量在变小，差异分量却每轮被 $B$ 补充、被 $A$ 放大。
- **乘积。** $AB=\begin{pmatrix}1.1&-0.08\\-0.1&0.88\end{pmatrix}$，行和 $1.02$ 与 $0.78$，一对矩阵的增益已经是 $1.02>1\times1$。$AB$ 的较大特征值是 $1.1318$，所以每多一对就再乘约 1.13：30 对（60 层）之后，$(AB)^{30}$ 的前向增益是 $24.8$，反向增益 $21.9$（笔者用 numpy 算出）。

这就是“每一层都把信号推向下一层最会放大的方向”的具体含义。还要注意 $A$ 的行和、列和都是 1，谱范数却是 1.2：它唯一缺的是**非负**。[§4](#p4) 会证明，非负、行和为 1、列和为 1 三个条件同时成立时，谱范数恰为 1，任何方向都不会被放大，这种“接力放大”也就无从发生。
:::

## 这不是个例：随机矩阵的模拟

27B 模型的矩阵是训练学出来的，会不会只是这一次运气不好？可以做一个与训练无关的实验：取 60 个彼此独立的随机矩阵 $H=I+sG$（$G$ 的 16 个元素独立地取标准正态分布，标量 $s$ 控制它离单位矩阵多远），把它们乘起来，看复合增益。这是笔者做的示意模拟，不是对 HC 训练过程的复现。

::: fig figures/l16-sim-products.svg | 图 16.6 60 个随机 4×4 矩阵连乘后的增益：无约束 vs 行随机 vs 双随机（模拟） | wide scroll
**看左图红线的斜率：扰动的平均值为 0，乘积的增益却一路上升。** (a) $H=I+sG$，2000 次试验，60 层后前向增益的中位数：$s=0.1$ 时 2.8，$s=0.2$ 时 16.4，$s=0.3$ 时 370（第 90 百分位已达 5118）。(b) 同样“离单位矩阵不远”的两类约束矩阵（对角 logit 为 5、扰动尺度为 2）：只保证行和为 1 的行随机矩阵，前向增益恒为 1，但反向增益的中位数稳定在 1.77；按 mHC 的方法做 20 次 Sinkhorn–Knopp 得到的双随机矩阵，前向为 1，反向中位数 1.022。由生成脚本用 numpy 模拟（种子 0）。
:::

零均值的扰动为什么会让乘积变大？下面的推导只保留到 $s^2$ 阶，是笔者为解释这个现象给出的近似，不是论文内容。

::: deriv 随机近单位矩阵每乘一次，对数范数平均涨多少（笔者推导）
$\norm{(I+sG)\mathbf v}^2=1+2sz+s^2w$ || 取单位向量 $\mathbf v\in\R^n$，展开平方；$z=\mathbf v^\top G\mathbf v$，$w=\norm{G\mathbf v}^2$
$z\sim\mathcal N(0,1),\qquad\E[w]=n$ || $z=\sum_{ij}v_iv_jG_{ij}$ 的方差为 $\sum_{ij}v_i^2v_j^2=1$；$G\mathbf v$ 的 $n$ 个分量各自是标准正态
$\log\norm{(I+sG)\mathbf v}=\tfrac12\log(1+a)\approx\tfrac12\big(a-\tfrac{a^2}{2}\big)$ || 记 $a=2sz+s^2w$，对小的 $a$ 做二阶泰勒展开
$\E[a]=ns^2,\quad\E[a^2]=4s^2+O(s^4)$ || $a^2=4s^2z^2+4s^3zw+s^4w^2$；$\E[z^2]=1$，$\E[zw]=0$（把 $G\mathbf v$ 关于 $\mathbf v$ 的垂直平面做镜像，$z$ 变号、$w$ 不变、分布不变）
$\E\log\norm{(I+sG)\mathbf v}\approx\tfrac12\big(ns^2-2s^2\big)=\tfrac{n-2}{2}\,s^2$ || 代入；$n=4$ 时恰为 $s^2$
:::

所以 $n=4$ 时，每乘一个矩阵，对数范数平均增加约 $s^2$。各层的 $G$ 互相独立，而且标准正态矩阵的分布在旋转下不变，所以不管当前向量转到了哪个方向，下一步的增量都同分布：60 步的对数范数是 60 个独立同分布增量之和，典型的放大倍数约为 $e^{60s^2}$，$s=0.1,0.2,0.3$ 分别给出 1.82、11.0、221。

这个近似可以用模拟核对。生成脚本直接模拟单位向量经过 60 个矩阵后的长度（2 万次），中位数是 1.83、11.0、231，吻合得很好。图 16.6 的 Amax 增益比这些数大约 1.5 倍，因为它量的是全 1 输入 $\mathbf 1_4$（长度是 $\sqrt4=2$，不是单位向量）的最大一个分量：$\norm{P\mathbf 1_4}$ 的典型值是上面数字的 2 倍，而一个方向随机的四维向量，最大分量的绝对值通常只有长度的 0.77 左右（中位数，笔者模拟），$2\times0.77\approx1.5$。

增长来自 $s^2w$ 这一项。$G\mathbf v$ 无论朝哪个方向，都给范数**加**上 $n$ 份能量；$2sz$ 的二阶效应只扣回 2 份。所以 $n\ge3$ 时净增为正（$n=2$ 时二阶项恰好抵消，这种各向同性的扰动几乎不再放大；本课的 $n=4$ 不在此列）。

结论是：流的条数 $n\ge3$ 时，只要每一层的混合矩阵不受“不许放大”的约束，哪怕它的扰动平均为零，深度也会把它放大成指数。

## 能不能干脆不要流间混合？

既然危险全在 $H^{\rm res}$ 的连乘上，把它固定成 $I$ 不就好了？mHC 论文的消融实验回答了这个问题（论文称之为初步实验，没有写模型规模与 token 数）：关掉某个映射时，用固定值代替它，$H^{\rm pre}$ 取均匀的 $1/n$、$H^{\rm post}$ 取全 1、$H^{\rm res}$ 取 $I$ [@xie2026mhc]。

| 打开的映射 | 相对基线的 loss 变化 |
|---|---|
| 只开 $H^{\rm res}$ | −0.022 |
| $H^{\rm res}$ + $H^{\rm pre}$ | −0.025 |
| 三个都开 | −0.027 |

HC 的收益里，八成以上来自流间混合 $H^{\rm res}$。最危险的映射恰恰是最有用的映射，不能删，只能**约束**。HC 还有别的代价：访存量大约随 $n$ 成比例增长，流水线并行时的通信量也是 $n$ 倍 [@xie2026mhc]，这些留到 [§6](#p6)。

## 约束成什么？几个候选

我们要找一个矩阵集合 $\mathcal C$，让 $H^{\rm res}_l\in\mathcal C$ 同时满足：(1) 包含（或能任意接近）$I$，这样至少能退回普通残差；(2) 允许流之间交换信息，否则就失去了 $H^{\rm res}$ 的收益；(3) 对乘法封闭，并且集合里的矩阵不放大信号，这样多少层连乘都安全。下表是笔者整理的几个候选，mHC 选的是最后一行；这是一个**选择**，不是唯一的推论。

| 候选集合 | 含 $I$ | 允许混合 | 连乘后的增益 | 备注 |
|---|---|---|---|---|
| 只有 $\{I\}$ | 是 | 否 | 恒为 1 | 退回 Pre-LN，没有收益 |
| 正交矩阵 $Q^\top Q=I$ | 是 | 是（旋转、反射） | 范数精确不变 | 元素可正可负，参数化较难；2026 年有工作用四元数参数化 $n=4$ 的旋转 [@guo2026ohc] |
| 行随机矩阵（非负、行和为 1） | 是 | 是 | 前向恒为 1，反向可达 $n$ | 逐行 softmax 即可实现；见 [§4](#p4) 的反例 |
| **双随机矩阵**（非负、行和列和都为 1） | 能任意接近 | 是 | 前向、反向都为 1 | mHC；用 Sinkhorn–Knopp 投影 |

::: key 本页要点
- HC 展开 $L-l$ 个子层后，直通项是复合映射 $P_{l\to L}=H^{\rm res}_{L-1}\cdots H^{\rm res}_l$ (16.13)，反向乘 $P_{l\to L}^\top$ (16.14)：He 2016 标量连乘的矩阵版。
- Amax 增益 (16.15)：前向取最大 |行和|、反向取最大 |列和|，即“所有流装同一个信号时被放大多少”；它允许正负抵消，不是 ∞-范数。图 8 的 $H^{\rm res}_1$：前向 18.72、反向 6.81。它只量一个方向，所以单层增益都是 1 的矩阵连乘照样能放大（2×2 算例：60 层后 24.8）。
- 27B 模型上，单层增益大多在 0.64–2.81，复合反向增益在 $l=52$ 处约 2825（论文说约 3000），第 12k 步前后出现 loss 突增。随机模拟：$n=4$ 时，尺度为 $s$ 的零均值扰动也会让乘积按约 $e^{Ls^2}$ 增长。
- 消融显示 $H^{\rm res}$ 贡献了大部分收益，所以要约束它而不是删掉它；mHC 选择双随机矩阵。
:::

# 双随机矩阵与 Birkhoff 多面体 | 约束集长什么样：定义、性质的证明、“守恒”的准确含义，以及它的代价 | 30

上一页的结论是：要给 $H^{\rm res}$ 找一个“连乘多少次都不放大”的集合。mHC 选的是双随机矩阵 [@xie2026mhc]。这一页把这个集合讲透：它是什么，为什么它能保证不放大（给出完整证明，论文只陈述没有证明），“守恒”到底守的是什么，以及它在几何上长什么样。数学补课 [M11](../math/m11-doubly-stochastic-sinkhorn.html) 有同一套内容的更慢的版本。

## 双随机矩阵：每一行、每一列都加到 1

**含义。** 一个方阵叫**双随机矩阵**（doubly stochastic matrix），如果它的元素都不小于 0，而且每一行之和、每一列之和都等于 1。

**理解。** 把它当成 $n$ 条传送带之间的调度表：第 $j$ 行说新带 $j$ 的货由各条旧带按什么比例组成（行和为 1：新带上的货是旧带的**加权平均**，不多也不少）；第 $i$ 列说旧带 $i$ 的货被怎样分给各条新带（列和为 1：旧带上的货**全部**分出去，既不凭空多出，也不凭空消失）。三个典型例子：单位矩阵 $I$（谁也不动）；置换矩阵（把带子重新编号，比如带 1 与带 2 互换）；$J_n$（把所有带子彻底搅匀，每条新带都是全部旧带的平均）。

**数学。**

::: eq 双随机矩阵集合（mHC 论文式 (6)） | (16.16)
$$\mathcal B_n=\big\{M\in\R^{n\times n}\ \big|\ M\mathbf 1_n=\mathbf 1_n,\ \ \mathbf 1_n^\top M=\mathbf 1_n^\top,\ \ M\ge0\big\}$$
$M\mathbf 1_n=\mathbf 1_n$：每行之和为 1；$\mathbf 1_n^\top M=\mathbf 1_n^\top$：每列之和为 1；$M\ge0$：逐元素非负。mHC 要求 $H^{\rm res}_l\in\mathcal B_n$。$n=1$ 时集合里只有一个数 1，恰好退回恒等映射 [@xie2026mhc]。
:::

::: ex 一个 3×3 的双随机矩阵作用在三条流上
取 $A=\begin{pmatrix}0.7&0.2&0.1\\0.2&0.5&0.3\\0.1&0.3&0.6\end{pmatrix}$，每行、每列之和都是 1。只看隐藏状态的某一个坐标，三条流在这个坐标上的值是 $\mathbf x=(6,3,0)^\top$。

- 一次混合：$A\mathbf x=(0.7\cdot6+0.2\cdot3,\ 0.2\cdot6+0.5\cdot3,\ 0.1\cdot6+0.3\cdot3)=(4.8,\ 2.7,\ 1.5)$。
- **总量不变**：$6+3+0=9$，$4.8+2.7+1.5=9$。
- **每个新值都在旧值的范围 $[0,6]$ 之内**：它们是旧值的加权平均。
- **差异变小**：三个值离平均数 3 的偏差从 $(3,0,-3)$（长度 $\sqrt{18}=4.243$）变成 $(1.8,-0.3,-1.5)$（长度 2.362）；再混合一次得 $(4.05,2.76,2.19)$，偏差长度 1.348。$A^{10}$ 的每个元素都已在 $0.3316$ 到 $0.3357$ 之间，几乎就是 $J_3$。

这个例子把后面要证明的三件事都演示了一遍：总量守恒、不会放大、反复作用会把各条流越搅越匀。最后一点既是安全的来源，也是代价的来源。
:::

## 性质一：乘法封闭

::: deriv 两个双随机矩阵的乘积仍是双随机矩阵
$A,B\in\mathcal B_n\ \Rightarrow\ AB\ge0$ || 非负数的乘积之和仍非负
$(AB)\mathbf 1_n=A(B\mathbf 1_n)=A\mathbf 1_n=\mathbf 1_n$ || 结合律；先用 $B$ 的行和为 1，再用 $A$ 的
$\mathbf 1_n^\top(AB)=(\mathbf 1_n^\top A)B=\mathbf 1_n^\top B=\mathbf 1_n^\top$ || 先用 $A$ 的列和为 1，再用 $B$ 的
:::

::: eq 乘法封闭 | (16.17)
$$A,B\in\mathcal B_n\ \Longrightarrow\ AB\in\mathcal B_n,\qquad\text{因此}\ \ P_{l\to L}=H^{\rm res}_{L-1}\cdots H^{\rm res}_l\in\mathcal B_n$$
不管连乘多少层，复合映射仍在同一个集合里。mHC 论文称之为 Compositional Closure [@xie2026mhc]。
:::

## 性质二：谱范数等于 1（不放大）

::: pre 谱范数 | 知道 $\norm{A}_2=\max\norm{A\mathbf x}/\norm{\mathbf x}$ 的可以跳过
矩阵的**谱范数** $\norm{A}_2$ 是它把一个向量“拉长”的最大倍数：$\norm{A}_2=\max_{\mathbf x\ne0}\norm{A\mathbf x}/\norm{\mathbf x}$，等于 $A$ 的最大奇异值。它满足 $\norm{AB}_2\le\norm{A}_2\norm{B}_2$：先拉长再拉长，最多是两个倍数相乘。所以只要每个因子的谱范数都不超过 1，任意多个因子的乘积也不超过 1。谱范数与奇异值见 [M02 §4](../math/m02-rank-svd-lowrank.html#p4)，本节的证明在 [M11 §3](../math/m11-doubly-stochastic-sinkhorn.html#p3) 里有两种写法。
:::

mHC 论文把这条性质写作 Norm Preservation，内容是 $\norm{H^{\rm res}_l}_2\le1$，即“非扩张”（non-expansive），没有给出证明 [@xie2026mhc]。下面是一个只用三个定义条件的证明（笔者补）。

::: deriv 双随机矩阵不会把任何向量拉长
$\norm{A\mathbf x}^2=\sum_{i}\Big(\sum_{j}a_{ij}x_j\Big)^2$ || 定义；$\mathbf x\in\R^n$ 是任意向量
$\le\sum_i\Big(\sum_ja_{ij}\Big)\Big(\sum_ja_{ij}x_j^2\Big)$ || 用到 $a_{ij}\ge0$（才能开方）：对 $(\sqrt{a_{ij}})_j$ 与 $(\sqrt{a_{ij}}\,x_j)_j$ 用柯西–施瓦茨不等式
$=\sum_i\sum_ja_{ij}x_j^2$ || 行和为 1
$=\sum_jx_j^2\sum_ia_{ij}$ || 交换求和次序
$=\sum_jx_j^2=\norm{\mathbf x}^2$ || 列和为 1
$\norm{A\mathbf 1_n}=\norm{\mathbf 1_n}$ || 行和为 1，所以全 1 向量不缩短，上界取到
:::

::: eq 谱范数等于 1 | (16.18)
$$A\in\mathcal B_n\ \Longrightarrow\ \norm{A}_2=1,\qquad \norm{AX}_F\le\norm{X}_F\ \ (X\in\R^{n\times d})$$
推导的每一步各用了一个条件：非负、行和、列和，三者缺一不可。对 $X$ 的每一列（隐藏状态的每一个坐标上的 $n$ 条流）分别用一次，就得到 Frobenius 范数的不等式。
:::

合起来：复合映射 $P_{l\to L}\in\mathcal B_n$，所以 $\norm{P_{l\to L}}_2=1$，前向不放大；$P_{l\to L}^\top$ 也是双随机矩阵（转置把行和列互换），所以反向也不放大。用 [§3](#p3) 的 Amax 增益来说：

::: eq 双随机复合映射的增益 | (16.19)
$$P\in\mathcal B_n\ \Longrightarrow\ g_{\rm fwd}(P)=g_{\rm bwd}(P)=1$$
行和、列和都恰为 1，与深度无关。这是精确算术下的结论；实际计算中 Sinkhorn–Knopp 只迭代 20 次，列和只是近似为 1（[§5](#p5)、[§7](#p7)）。
:::

## 性质三：守恒的是“各条流之和”

::: eq 各流之和不变，每条新流是旧流的凸组合 | (16.20)
$$\mathbf 1_n^\top(AX)=\mathbf 1_n^\top X,\qquad (AX)_{j,:}=\sum_i a_{ji}X_{i,:},\ \ a_{ji}\ge0,\ \sum_ia_{ji}=1$$
第一式用列和为 1：$n$ 条流逐坐标相加的总和不变，这就是“全局均值被保持”。第二式用行和为 1 与非负：每条新流都是旧流的**凸组合**（加权平均）。mHC 论文指出 HC 的复合映射做不到前者 [@xie2026mhc]；[§2](#p2) 的手算例里，列和为 1.1 和 0.9 的矩阵就改变了流的和。
:::

## 只约束行和为什么不够

只要求非负且行和为 1 的矩阵叫**行随机矩阵**（row-stochastic），用逐行 softmax 就能得到，比双随机矩阵简单得多。它够不够？

::: deriv 行随机：前向安全，反向不安全
$P=H_{L-1}\cdots H_l,\quad H_k\ge0,\quad H_k\mathbf 1_n=\mathbf 1_n$ || 每个因子都是行随机矩阵
$P\mathbf 1_n=\mathbf 1_n\ \Rightarrow\ g_{\rm fwd}(P)=1$ || 行和的乘法封闭，同 (16.17) 之前那个推导的第二行：$(AB)\mathbf 1=A(B\mathbf 1)=\mathbf 1$，逐个因子用下去
$\mathbf 1_n^\top P\mathbf 1_n=\mathbf 1_n^\top\mathbf 1_n=n$ || 全部元素之和 = 各行之和相加 = $n$
$g_{\rm bwd}(P)=\max_j\sum_iP_{ij}\le n$ || 元素非负，任一列之和不超过全部元素之和
$H=\begin{pmatrix}1&0\\1&0\end{pmatrix}:\ \text{列和}(2,0),\ \norm{H}_2=\sqrt2$ || 反例：两条新流都只从旧流 1 取货，旧流 1 被“复制”了一份
:::

所以只约束行和时，反向增益最多能到 $n$，而且上界能取到：上面的反例连乘多少次都是它自己，列和永远是 $(2,0)$。图 16.6 右图的模拟里，行随机矩阵 60 层后的反向增益中位数是 1.77。这个上界 $n=4$ 在 [§7](#p7) 还会用到：Sinkhorn–Knopp 截断在第 20 步时，**行和**精确为 1、列和只近似为 1，实际矩阵介于“行随机”与“双随机”之间。

## “混合单调增加”的准确含义

mHC 论文还说，反复作用双随机矩阵会让流之间的信息混合“单调地”增加，但没有给出定义与证明 [@xie2026mhc]。下面给出一个有明确定义、证明紧随其后的版本（笔者推导）。记 $\Pi=I-J_n$：它把每个坐标上的 $n$ 个值减去它们的平均，只留下“各条流偏离平均的部分”。

::: deriv 各流偏离平均的总量不会增加
$AJ_n=\tfrac1nA\mathbf 1_n\mathbf 1_n^\top=\tfrac1n\mathbf 1_n\mathbf 1_n^\top=J_n$ || 行和为 1
$J_nA=\tfrac1n\mathbf 1_n\mathbf 1_n^\top A=\tfrac1n\mathbf 1_n\mathbf 1_n^\top=J_n$ || 列和为 1
$\Pi A=A-J_n=A\Pi$ || 两式相减：$A$ 与 $\Pi$ 可交换
$\norm{\Pi AX}_F=\norm{A\,\Pi X}_F\le\norm{\Pi X}_F$ || 先交换，再用 (16.18)
:::

::: eq 流间差异非增 | (16.21)
$$\norm{(I-J_n)\,A X}_F\le\norm{(I-J_n)\,X}_F,\qquad A\in\mathcal B_n$$
对**所有** $X$ 都取等号，当且仅当 $A$ 是置换矩阵（包括 $I$）：这时 $A$ 只是给流重新编号。对所有 $\Pi X\ne0$ 都严格减少，当且仅当 $A$ 的第二大奇异值 $\sigma_2(A)<1$（$A$ 在“偏离平均”的方向上最多拉长 $\sigma_2(A)$ 倍，[§5](#p5) 的前置框有奇异值的定义）。所以“单调”只在“不增”的弱意义下成立。
:::

(16.21) 同时说出了双随机约束的代价：它只能把各条流**越搅越匀**，不能把它们重新拉开。能不能说得更定量一些：连乘 $L$ 层之后，复合映射离“彻底搅匀”的 $J_n$ 还有多远？下面的推导（笔者补）只用到上面已经证明的几件事。

::: deriv 连乘之后离“彻底搅匀”有多远
$P=H_L\cdots H_1\in\mathcal B_n\ \Rightarrow\ PJ_n=J_n$ || 乘法封闭 (16.17)，再用上面推导第一行的“行和为 1”
$P-J_n=P-PJ_n=P\,\Pi$ || 提出 $P$；$\Pi=I-J_n$
$P\,\Pi=(H_L\Pi)(H_{L-1}\Pi)\cdots(H_1\Pi)$ || 从等式右边往回看：每个 $H_k$ 都与 $\Pi$ 可交换，$L$ 个 $\Pi$ 都能挪到最右端；再用 $\Pi^2=\Pi$ 合并成一个，就是左边
$\norm{P-J_n}_2\le\prod_{k=1}^{L}\norm{H_k\Pi}_2$ || 谱范数的乘法不等式（本页前置框）
$\norm{H_k\Pi}_2=\sigma_2(H_k)$ || $H_k^\top H_k\mathbf 1_n=\mathbf 1_n$（先用行和、再用列和），所以 $\mathbf 1_n$ 方向的奇异值是 1；其余奇异方向都与 $\mathbf 1_n$ 垂直，$\Pi$ 只留下它们，其中最大的是 $\sigma_2(H_k)$
$\norm{H_L\cdots H_1-J_n}_2\le\prod_{k=1}^{L}\sigma_2(H_k)$ || 代入上一行
:::

::: ex 用本页开头的 3×3 例子核对
那个 $A$ 是对称矩阵，特征值是 $1$、$(4+\sqrt3)/10=0.5732$、$(4-\sqrt3)/10=0.2268$（三者之和 $1.8$ 等于对角线之和），所以 $\sigma_2(A)=0.5732$。

- 上界：$\norm{A^{10}-J_3}_2\le0.5732^{10}=0.00383$。对称矩阵的这个不等式取等号，实际值正是 0.00383（笔者用 numpy 核对）。
- 对照算例：$A^{10}$ 的元素都在 0.3316 到 0.3357 之间，离 $1/3$ 最多 0.0024，没有超过 0.00383。
- 每混合一次，偏差长度 $4.243\to2.362\to1.348$，相邻之比 0.557、0.571，都不超过 $\sigma_2=0.5732$。
:::

所以只要每一层都有一点严格的收缩，而且收缩的力度不随层数减弱（所有层都满足 $\sigma_2(H_k)\le\rho<1$），深层的复合映射就会以 $\rho^L$ 的速度趋向 $J_n$，$n$ 条流趋向同一个平均值。[§7](#p7) 会看到，27B 模型里 mHC 的 60 层复合映射与 $J_4$ 的最大差只有 0.04。2026 年 9 月的一篇后续工作（oHC，尚未经同行评审）正是从这个角度批评 mHC：双随机约束只能靠缩小流间差异来控制范数 [@guo2026ohc]。

## Birkhoff 多面体：置换矩阵的凸包

$\mathcal B_n$ 在几何上是什么形状？它由有限个线性等式和不等式定义，而且有界（每个元素都在 $[0,1]$ 之间：非负，且同一行的元素加起来才是 1），所以是一个凸多面体。它的顶点由一个经典定理给出。

::: pre 凸组合、凸包与顶点 | 学过凸集的可以跳过
几个点 $P_1,\dots,P_m$ 的**凸组合**是 $\sum_k\theta_kP_k$，其中 $\theta_k\ge0$、$\sum_k\theta_k=1$：按比例混合，比例非负、加起来是 1。平面上两个点的凸组合是连接它们的线段，三个点的凸组合是它们围成的三角形（含内部）。一组点的全部凸组合叫它们的**凸包**。**顶点**是凸集里不能写成集合中另外两点的真正混合的点，例如三角形的三个角。这里的“点”可以是矩阵：把 $n\times n$ 矩阵看成 $n^2$ 维空间里的一个点即可。慢一点的版本（含 3×3 的逐步分解）见 [M11 §4](../math/m11-doubly-stochastic-sinkhorn.html#p4)。
:::

::: thm Birkhoff–von Neumann 定理
每个 $n\times n$ 双随机矩阵都能写成置换矩阵的凸组合；$\mathcal B_n$ 的顶点恰好是 $n!$ 个置换矩阵。出处为 Birkhoff 1946 [@birkhoff1946]（原文本课未见，卷期信息来自二手索引）。

$$A\in\mathcal B_n\iff A=\sum_{k}\theta_kP_k,\qquad \theta_k\ge0,\ \ \sum_k\theta_k=1,\ \ P_k\ \text{为置换矩阵}$$
:::

mHC 论文用这个定理给出第三条性质的几何解释：残差映射是若干“置换”（把流重新编号）的凸组合 [@xie2026mhc]。用传送带的话说：每一种合法的调度，都是若干种“把带子整体换个编号”的操作按比例的混合。

::: deriv Birkhoff 多面体的维数
$n^2\ \text{个元素},\quad 2n\ \text{个等式约束}$ || $n$ 个行和、$n$ 个列和
$\sum_i(\text{第 }i\text{ 行之和})=\sum_j(\text{第 }j\text{ 列之和})$ || 两边都等于全部元素之和，所以有一个约束是多余的
$\dim\mathcal B_n=n^2-(2n-1)=(n-1)^2$ || 独立约束 $2n-1$ 个；非负约束只切出边界，不降维
:::

::: eq Birkhoff 多面体的大小 | (16.22)
$$\dim\mathcal B_n=(n-1)^2,\qquad \#\text{顶点}=n!$$
$n=2$：1 维（一条线段），2 个顶点；$n=3$：4 维，6 个顶点；$n=4$：9 维，24 个顶点。
:::

::: fig figures/l16-birkhoff.svg | 图 16.7 Birkhoff 多面体：n = 2 是线段，n = 3 画不出来，只能画一个精确的切片 | wide scroll
**看 (b) 的标题：$\mathcal B_3$ 是 4 维的，平面上只能画它的一个切片。** (a) $\mathcal B_2$ 是连接 $I$ 与交换矩阵的线段，中点是 $J_2$。(b) $\mathcal B_3$ 中所有“循环矩阵” $a I+bC+cC^2$（$C$ 是循环移位，$a+b+c=1$，非负）组成的三角形，三个顶点是三个置换矩阵，重心是 $J_3$；小方块图里紫色格是 1、灰色格是 0。(c) [§5](#p5) 算例的 Sinkhorn 极限矩阵分解成 5 个置换矩阵的凸组合（贪心分解，分解不唯一），权重之和为 1。(c) 的数字由生成脚本计算。
:::

为什么 mHC 不直接用“24 个置换矩阵的凸组合”来参数化 $H^{\rm res}$（学 24 个非负、和为 1 的权重）？这个办法对 $n=4$ 可行，但权重数随 $n!$ 增长，$n=8$ 就要 40,320 个。下一页的 Sinkhorn–Knopp 只需要 $n^2$ 个数（笔者的比较，论文没有讨论这一替代方案）。

::: err 三个常见说法，以及更准确的版本
**“mHC 让残差映射可以精确地回到恒等映射。”** $I$ 是 $\mathcal B_n$ 的一个顶点，但 mHC 的投影先取指数再归一化，输出的每个元素都严格为正，只能无限接近 $I$，永远到不了；V4 的代码还在归一化时加了 $10^{-6}$ 的 eps [@hf-v4flash-code]。mHC 恢复的是恒等映射的**性质**（均值守恒、不放大、任意深度仍守恒），不是 $H^{\rm res}=I$ 本身。

**“Norm Preservation 意味着范数不变。”** 双随机矩阵只保证范数**不增**。$J_n$ 把各流偏离平均的部分全部压成 0，范数可以大幅缩小。准确说法是“非扩张、均值守恒”。

**“mHC 把残差映射约束在一个流形上。”** 论文名字里的“流形”（manifold）是宽泛的说法。$\mathcal B_n$ 是凸多面体，有顶点、棱、面，不是光滑流形；它的相对内部（所有元素都严格为正的双随机矩阵）才是一个 $(n-1)^2$ 维的光滑流形。Sinkhorn 迭代的极限（$t\to\infty$）总落在这个内部；截断到 20 轮的实际输出元素也都严格为正，但列和只近似为 1，严格说只是贴近多面体，并不恰好在它上面（[§5](#p5)）。
:::

::: ana 传送带类比的 mHC 版本
**映射。** 双随机矩阵 = 守恒的调度规则：每条新带上的货是各条旧带的加权平均（行和为 1），每条旧带上的货全部分出去、不多不少（列和为 1），不允许负的份额（非负）。置换矩阵 = 把带子整体重新编号；$J_n$ = 把所有带子的货倒在一起搅匀再平分。

**收益。** “守恒”让 (16.19) 一眼可见：货的总量不会越传越多，也不会凭空消失，所以 60 站之后也不会爆炸。它也让代价一眼可见：只能搅匀，不能分拣，搅多了每条带子上都是一样的混合物 (16.21)。

**失效的地方。** 守恒的是“各条流逐坐标之和”，不是每件货的大小：一条带子上的向量可以因为和别的带子平均而变短。而且“货”可以是负数，两条带子的值平均后可以互相抵消，这在实物传送带上没有对应。
:::

::: key 本页要点
- 双随机矩阵：非负、行和列和都为 1 (16.16)；$n=1$ 时退回恒等映射。
- 三条性质（笔者补全了证明）：乘法封闭 (16.17)；谱范数等于 1，用柯西–施瓦茨一步一个条件地证明 (16.18)，所以复合映射的前向、反向增益都为 1 (16.19)；各流之和守恒、每条新流是旧流的凸组合 (16.20)。
- 只约束行和不够：前向安全，反向增益可到 $n$（反例 $\begin{pmatrix}1&0\\1&0\end{pmatrix}$）。
- 代价：流间差异只减不增 (16.21)，深层复合映射可以趋向 $J_n$。
- Birkhoff 多面体 = 置换矩阵的凸包，$(n-1)^2$ 维、$n!$ 个顶点 (16.22)；$\mathcal B_3$ 是 4 维，只能画切片。$I$ 是顶点，mHC 只能无限接近它。
:::

# Sinkhorn–Knopp：把任意矩阵拉进多面体 | 交替归一化、一个 3×3 的手算、收敛的快与慢、“熵投影”的准确含义，以及 mHC 的参数化 | 35

网络对每个 token、每个子层算出的是一个任意的实数矩阵 $\tilde H^{\rm res}_l\in\R^{4\times4}$（怎么算出来的，本页最后讲）。我们需要一个映射，把它变成 $\mathcal B_4$ 里的矩阵，而且要满足三个条件：处处可导（要反向传播）；在 GPU 上对每个 token 的 $4\times4$ 小矩阵都便宜；能输出任意接近 $I$ 的矩阵。几种办法：求欧氏距离最近的双随机矩阵（每个 token 解一个二次规划，贵，而且导数分段）；学 24 个置换矩阵的凸组合权重（[§4](#p4) 末尾说过，$n!$ 增长太快）；先取指数、再交替做行列归一化。mHC 选了最后一种 [@xie2026mhc]。数学补课 [M11 §5–§6](../math/m11-doubly-stochastic-sinkhorn.html#p5) 有 SK 及其收敛性的更慢的版本。

## 交替归一化

**含义。** **Sinkhorn–Knopp 迭代**（下称 SK）：先把矩阵的每一列除以这一列的和，再把每一行除以这一行的和，如此反复。

**理解。** 每一步只修一种和：列归一化之后列和全对了，但行和被打乱；行归一化之后行和全对了，列和又偏了一点。关键在于：偏差一轮比一轮小，最后两种和同时都对。整个过程只做了“整行乘一个数”和“整列乘一个数”，所以矩阵内部的“比例结构”不会被破坏（下面马上说清是什么意思）。

**数学。**

::: eq 列归一化与行归一化 | (16.23)
$$\mathcal T_c(M)_{ij}=\frac{M_{ij}}{\sum_{k}M_{kj}},\qquad \mathcal T_r(M)_{ij}=\frac{M_{ij}}{\sum_{k}M_{ik}}$$
$M$ 是元素全为正的 $n\times n$ 矩阵。$\mathcal T_c$ 之后每列和为 1，$\mathcal T_r$ 之后每行和为 1。
:::

::: eq mHC 的 Sinkhorn–Knopp 投影（mHC 论文式 (9)） | (16.24)
$$M^{(0)}=\exp\big(\tilde H^{\rm res}_l\big),\qquad M^{(t)}=\mathcal T_r\big(\mathcal T_c(M^{(t-1)})\big),\qquad H^{\rm res}_l=M^{(t_{\max})},\ \ t_{\max}=20$$
$\exp$ 逐元素作用，保证 $M^{(0)}$ 的每个元素都严格为正。每一轮**先列后行**，最后一步是行归一化。论文说 $t_{\max}\to\infty$ 时收敛到双随机矩阵，实用中取 20 [@xie2026mhc]；DeepSeek-V4 用同样的 20 次 [@dsv4]。
:::

::: ana 类比：两个调度员轮流改调度表
**映射。** 沿用传送带的说法：矩阵 $M$ 是 $n$ 条旧带与 $n$ 条新带之间的调度表，第 $i$ 列写旧带 $i$ 的货怎样分给各条新带，第 $j$ 行写新带 $j$ 从各条旧带收多少。发货调度员做列归一化：让每条旧带恰好把一份货发完，办法是把这一整列乘同一个数。收货调度员做行归一化：让每条新带恰好收到一份，办法是把这一整行乘同一个数。两人轮流改表，就是 SK。

**收益。** 它让三件事变得好懂：一个人改完，另一个人的账就被打乱，所以要来回改多轮；两人都只能整行、整列地乘，所以表里“谁更偏向谁”的相对关系（下面的交叉比）始终不变；如果表里有的格子几乎为 0，两人每轮能调动的余地都很小，账要来回很多轮才能同时对上，这就是后面会看到的“慢”。

**失效的地方。** 真实的调度员可以只改一个格子，SK 不能；所以原本为 0 的格子永远是 0，SK 无法凭空打通一条通道。类比也说明不了为什么两人最后一定能同时满意，这要靠下面的 Sinkhorn 定理。
:::

它为什么会收敛，收敛到哪里？这是 Sinkhorn 在 1964 年、Sinkhorn 与 Knopp 在 1967 年证明的定理。

::: thm Sinkhorn（1964）与 Sinkhorn–Knopp（1967）
**正矩阵**：对任意元素全为正的方阵 $A$，存在对角元素为正的对角矩阵 $D_1,D_2$，使 $D_1AD_2$ 是双随机矩阵，而且 $D_1AD_2$ 是唯一的 [@sinkhorn1964]。

**非负矩阵**：交替行列归一化的序列收敛到一个双随机极限，当且仅当 $A$ 至少含有一条“正对角线”（存在一个置换 $\pi$，使 $a_{1\pi(1)},\dots,a_{n\pi(n)}$ 全为正）；极限能写成 $D_1AD_2$ 的充要条件是 $A\ne0$ 且每个正元素都落在某条正对角线上 [@sinkhorn1967]。

mHC 先取 $\exp$，矩阵严格为正，所以真正用到的是第一种情形，条件自动满足。
:::

::: deriv SK 的结果为什么是 D₁AD₂ 的形式
$\mathcal T_c(M)=M\,\diag(1/\text{列和})$ || 每列除以一个数 = 右乘一个对角矩阵
$\mathcal T_r(M)=\diag(1/\text{行和})\,M$ || 每行除以一个数 = 左乘一个对角矩阵
$M^{(t)}=D_1^{(t)}\,M^{(0)}\,D_2^{(t)}$ || 左乘的对角矩阵连乘仍是对角矩阵，右乘同理
$M^{(t)}_{ij}=\mu_i\,a_{ij}\,\nu_j$ || 记 $M^{(0)}=A=(a_{ij})$，$D_1^{(t)}=\diag(\mu_1,\dots,\mu_n)$，$D_2^{(t)}=\diag(\nu_1,\dots,\nu_n)$（行、列缩放因子，都为正）
$\dfrac{M^{(t)}_{ij}M^{(t)}_{i'j'}}{M^{(t)}_{ij'}M^{(t)}_{i'j}}=\dfrac{a_{ij}a_{i'j'}}{a_{ij'}a_{i'j}}$ || 任取两行 $i,i'$、两列 $j,j'$，代入上一行：$\mu_i,\mu_{i'},\nu_j,\nu_{j'}$ 在分子分母里各出现一次，全部约掉
:::

最后一行左边那个量叫**交叉比**（cross ratio）。**含义**：任取两行、两列围成一个 $2\times2$ 子块，用“主对角两个元素之积”除以“副对角两个元素之积”。**理解**：它回答“第 $i$ 行比第 $i'$ 行更偏向第 $j$ 列还是第 $j'$ 列、偏多少”；把一整行乘 2，这一行对每一列的分量一起翻倍，偏好的比值不变，把一整列乘 2 也一样。例如下面 3×3 算例里，前两行、前两列的交叉比是 $\frac{4\times4}{1\times1}=16$：“行 1 偏爱列 1、行 2 偏爱列 2”的程度是 16，SK 迭代多少轮它都还是 16。

所以 SK 只调整“每行多大、每列多大”，不改变任何两行两列之间的交叉比。在所有“行列缩放后的 $A$”里，双随机的那一个是唯一的，SK 就收敛到它（“一定收敛”这件事本身由上面的 Sinkhorn 定理保证，交叉比只说明了极限只能是谁）。

## 手算一个 3×3

::: ex Sinkhorn–Knopp 的前两步（研究笔记选的算例）
取 $A=\begin{pmatrix}4&1&1\\1&4&1\\2&2&2\end{pmatrix}$（在 mHC 里它相当于 $\exp(\tilde H^{\rm res})$）。三行之和都是 6，已经相等；三列之和是 $7,7,4$，不相等。所以只把每行除以 6 不够。

**第 1 轮，列归一化**：第 1、2 列除以 7，第 3 列除以 4：

$$\mathcal T_c(A)=\begin{pmatrix}4/7&1/7&1/4\\1/7&4/7&1/4\\2/7&2/7&1/2\end{pmatrix}$$

列和都是 1。行和：$\tfrac47+\tfrac17+\tfrac14=\tfrac{16+4+7}{28}=\tfrac{27}{28}$，第 2 行同样 $\tfrac{27}{28}$，第 3 行 $\tfrac27+\tfrac27+\tfrac12=\tfrac{15}{14}$。

**第 1 轮，行归一化**：第 1、2 行乘 $\tfrac{28}{27}$，第 3 行乘 $\tfrac{14}{15}$：

$$M^{(1)}=\begin{pmatrix}16/27&4/27&7/27\\4/27&16/27&7/27\\4/15&4/15&7/15\end{pmatrix}$$

行和都是 1。列和：$\tfrac{16}{27}+\tfrac{4}{27}+\tfrac{4}{15}=\tfrac{100+36}{135}=\tfrac{136}{135}\approx1.0074$，第 2 列同样，第 3 列 $\tfrac{14}{27}+\tfrac{7}{15}=\tfrac{70+63}{135}=\tfrac{133}{135}\approx0.9852$。最大列和误差 $0.0148$。

**继续**：第 2 轮后误差 $6.46\times10^{-4}$，第 3 轮 $2.81\times10^{-5}$，第 10 轮 $8.4\times10^{-15}$。相邻两轮的误差之比稳定在 $0.0436$。极限（五位小数）是

$$\begin{pmatrix}0.58899&0.14725&0.26376\\0.14725&0.58899&0.26376\\0.26376&0.26376&0.47247\end{pmatrix}=D_1AD_2,$$

其中 $D_1\propto\diag(1.11652,\,1.11652,\,1)$，$D_2\propto\diag(0.13188,\,0.13188,\,0.23624)$（把 $D_1$ 乘以任意正数 $c$、$D_2$ 同时除以 $c$，乘积不变，所以只能写到比例）。

**两个核对**：(1) 右下角 $1\times2\times0.23624=0.47248$，与 0.47247 一致（舍入）。(2) 交叉比：$A$ 的 $\tfrac{a_{11}a_{22}}{a_{12}a_{21}}=\tfrac{4\times4}{1\times1}=16$；极限的 $(0.58899/0.14725)^2=4.0000^2=16.0$，没有变。
:::

::: fig figures/l16-sinkhorn-steps.svg | 图 16.8 Sinkhorn–Knopp 的一个 3×3 算例，以及它收敛得有多快 | wide scroll
**看每个矩阵旁边行和、列和的颜色：红色是还不等于 1 的那一组，每一步只能把一组变绿。** ①→② 列归一化，列和变 1、行和被打乱；②→③ 行归一化，行和变 1、列和又偏了一点；④ 是极限。⑤ 最大列和误差随轮数的变化（对数纵轴）：本页算例（绿）每轮乘约 0.0436，是一条直线；$\begin{pmatrix}1&1\\0&1\end{pmatrix}$（红）误差恰为 $1/(2t+1)$，20 轮后还有 0.024；元素为正但有一个极小元素 $e^{-10}$ 的矩阵（橙虚线）几乎与红线重合。全部数字由生成脚本用分数与浮点迭代算出。
:::

## 收敛的快与慢

本页算例每轮误差乘 0.0436，属于**线性收敛**（误差按几何级数下降），10 轮就到了机器精度。这个 0.0436 由什么决定？Knight 在 2008 年给出了精确的答案 [@knight2008]。

::: pre 奇异值与“第二大奇异值” | 知道 SVD 的可以跳过
任何 $n\times n$ 矩阵 $P$ 都能分解成 $P=U\Sigma V^\top$，$U,V$ 是正交矩阵，$\Sigma=\diag(\sigma_1,\dots,\sigma_n)$，$\sigma_1\ge\sigma_2\ge\dots\ge0$ 叫**奇异值**：$P$ 在 $n$ 个互相垂直的方向上各把向量拉长多少倍。$\sigma_1$ 就是 [§4](#p4) 的谱范数，双随机矩阵的 $\sigma_1=1$；$\sigma_2$ 是除去那个最大方向之后最大的拉伸倍数。对称矩阵的奇异值等于特征值的绝对值，所以对称矩阵只需求特征值。详见 [M02 §3](../math/m02-rank-svd-lowrank.html#p3)；双随机矩阵的 $\sigma_2$ 与“越搅越匀”的关系见 [M11 §7](../math/m11-doubly-stochastic-sinkhorn.html#p7)。
:::

::: thm Sinkhorn–Knopp 的收敛速率（Knight 2008，定理 4.5）
设非负方阵 $A$ **完全不可分解**（fully indecomposable）：无论怎样重排行与列，都不能把它变成 $\begin{pmatrix}A_1&0\\A_2&A_3\end{pmatrix}$（$A_1$ 为方块）这种右上角整块为 0 的形式；元素全为正的矩阵都满足。那么 SK 线性收敛到双随机矩阵 $P=D_1AD_2$，并且从某一轮起，行、列缩放向量离极限的距离每轮都缩小到原来的 $\sigma_2(P)^2$ 倍以内，$\sigma_2(P)$ 是 $P$ 的**第二大奇异值**。Knight 的数值实验显示，实际速率很快就贴近这个值 [@knight2008]。
:::

**理解。** $P$ 的最大奇异值总是 1（(16.18)），对应“各条流完全一样”的方向 $\mathbf 1_n$；第二大奇异值衡量 $P$ 把“流与流之间的差异”保留了多少。$P$ 越接近置换矩阵（几乎不混合），$\sigma_2$ 越接近 1，SK 每轮能消掉的误差就越少；$P$ 越接近 $J_n$（彻底搅匀），$\sigma_2$ 越接近 0，一两轮就收敛。

定理说的是缩放向量的误差，本课一直量的是列和误差。每轮以行归一化收尾时，列和完全由缩放向量决定，在极限附近近似是缩放向量误差的线性函数，所以列和误差渐近地按同一个比率下降（这一句是笔者的补充，下面的算例用数值核对过）。

::: ex 用定理核对本页的 0.0436
研究笔记算过本页极限矩阵的奇异值：$1,\ 0.44174,\ 0.20871$。直接套定理得 $\sigma_2^2=0.44174^2=0.1951$，比实测的 0.0436 大得多。原因在对称性（下面是笔者的解读，已用数值核对）：$A$ 的前两行、前两列同时对调之后不变，SK 的每一步都保持这个对称，所以行缩放向量总是 $(\mu,\mu,\mu_3)$ 的形状，误差永远落在“行 1 与行 2 相同”的那一部分里。$\sigma_2=0.44174$ 对应的奇异向量是 $(1,-1,0)/\sqrt2$，恰好是“行 1 与行 2 相反”的方向，误差碰不到它；真正起作用的是下一个奇异值：$0.20871^2=0.04356$，这正是实测的每轮 0.0436。

没有这种对称性时，定理的 $\sigma_2^2$ 就是实际速率。笔者用 numpy 对几个随机 $4\times4$ 正矩阵核对过：第 15 轮以后相邻两轮的误差之比与各自的 $\sigma_2(P)^2$ 吻合到三位小数（例如 0.834 对 0.834，0.458 对 0.458）。
:::

但定理有前提：完全不可分解。下面这个 $2\times2$ 的例子不满足它（先对调两行、再对调两列，就得到 $\begin{pmatrix}1&0\\1&1\end{pmatrix}$），收敛慢得多。

::: deriv 一个慢的例子：M⁽⁰⁾ = [[1, 1], [0, 1]]
$\mathcal T_c(M^{(0)})=\begin{pmatrix}1&1/2\\0&1/2\end{pmatrix}$ || 第 1 轮：列和是 1、2，第 2 列除以 2
$M^{(1)}=\begin{pmatrix}2/3&1/3\\0&1\end{pmatrix}$ || 再按行和 3/2、1/2 做行归一化
$M^{(t)}=\begin{pmatrix}1-b_t&b_t\\0&1\end{pmatrix},\ \ b_1=\tfrac13$ || 形状保持不变：左下角永远是 0
$\mathcal T_c(M^{(t)})=\begin{pmatrix}1&b_t/(1+b_t)\\0&1/(1+b_t)\end{pmatrix}$ || 第 1 列和为 $1-b_t$，第 2 列和为 $1+b_t$
$b_{t+1}=\dfrac{b_t/(1+b_t)}{1+b_t/(1+b_t)}=\dfrac{b_t}{1+2b_t}$ || 再做行归一化：第 1 行除以它的和
$\dfrac{1}{b_{t+1}}=\dfrac{1}{b_t}+2\ \Rightarrow\ \dfrac1{b_t}=2t+1$ || 取倒数得等差数列，首项 3
$\text{列和}=\Big(\dfrac{2t}{2t+1},\ \dfrac{2t+2}{2t+1}\Big)$ || 列和是 $1-b_t$ 与 $1+b_t$
$\text{误差}=b_t=\dfrac{1}{2t+1}$ || $t=20$ 时 $1/41=0.0244$
:::

这个矩阵有正对角线（主对角线），所以 SK 收敛（极限是 $I$）；但右上角那个 1 不在任何一条正对角线上，极限不能写成 $D_1AD_2$ 的形式，收敛只有 $O(1/t)$ 那么慢。mHC 的矩阵是 $\exp(\cdot)$，严格为正，不会出现精确的 0；但如果某些 logit 比同一行、同一列的其他 logit 小很多，$\exp$ 之后就会出现 $e^{-10}$ 这样“几乎为 0”的元素。研究笔记算过 $\begin{pmatrix}1&1\\e^{-k}&1\end{pmatrix}$ 迭代 20 轮后的列和误差：$k=2$ 时约为 0，$k=5$ 时 0.0004，$k=10$ 时 0.0232，$k=20$ 时 0.0244。logit 相差到 10 左右，20 轮 SK 就会留下 2% 量级的列和误差。

Knight 定理把这个“慢”算成了一个数。$\begin{pmatrix}1&1\\e^{-k}&1\end{pmatrix}$ 严格为正，定理适用。它的极限是 $\begin{pmatrix}1-\delta&\delta\\\delta&1-\delta\end{pmatrix}$，交叉比不变给出 $\frac{(1-\delta)^2}{\delta^2}=\frac{1\times1}{1\times e^{-k}}=e^{k}$，即 $\delta=\frac{1}{1+e^{k/2}}$。这个 $2\times2$ 双随机矩阵的第二个特征值（也是第二大奇异值）是 $1-2\delta$。$k=10$ 时 $\delta=0.0067$，$\sigma_2=0.9866$，$\sigma_2^2=0.973$：进入渐近阶段后每轮只能消掉不到 3% 的误差（笔者数值核对：第 150 轮前后相邻误差之比为 0.9734）。极限越接近置换矩阵，SK 越慢，这就是“慢”的准确含义。

研究笔记还做了一个随机实验（笔者模拟，不是论文数据）：logit 取 $s$ 倍的标准正态随机数，看 $t$ 轮 SK 之后单个 $4\times4$ 矩阵的反向增益（最大列和；行和已精确为 1）。

| logit 尺度 $s$ | $t=1$ 中位 / p99 | $t=5$ | $t=10$ | $t=20$ |
|---|---|---|---|---|
| 1 | 1.128 / 1.427 | 1.0003 / 1.011 | 1.0000 / 1.0004 | 1.0000 / 1.0000 |
| 2 | 1.300 / 1.924 | 1.012 / 1.105 | 1.0004 / 1.027 | 1.0000 / 1.0065 |
| 4 | 1.469 / 2.366 | 1.067 / 1.372 | 1.017 / 1.101 | 1.0021 / 1.037 |
| 8 | 1.545 / 2.703 | 1.124 / 1.978 | 1.051 / 1.402 | 1.019 / 1.076 |

logit 越悬殊，20 轮之后剩下的误差越大。用 Knight 定理的话说：logit 越悬殊，极限 $P$ 越靠近某个置换矩阵，$\sigma_2(P)$ 越接近 1，每轮缩小的比例 $\sigma_2^2$ 也越接近 1。[§7](#p7) 会看到，27B 模型里 mHC 单层的反向增益在 1.01 到 1.06 之间，与表中 $s=4$ 到 $8$ 的量级相当；这是笔者的推测性解释，论文没有分析这一偏差的来源。

::: warn 迭代顺序决定哪一组和是精确的
mHC 与 V4 的公式都是 $\mathcal T_r(\mathcal T_c(\cdot))$，每轮以行归一化收尾，所以截断后**行和精确为 1**（前向增益在精确算术下恰为 1），**列和只近似为 1**（反向增益略大于 1）。Mena 等人 2018 年的论文把 SK 写成先行后列 [@mena2018gumbel]，顺序正好相反，读文献时要留意。V4 的推理代码里存的是 $H^{\rm res}$ 的转置，代码中“先行后列”换到 $H^{\rm res}$ 的坐标下正是“先列后行”，与论文一致；代码的第一步是逐行 softmax（减去最大值再取指数，数值更稳），每个分母还加了 $10^{-6}$ [@hf-v4flash-code]。
:::

## “熵投影”：SK 到底在投影什么

mHC 论文说它把 $H^{\rm res}_l$“熵意义下地投影”（entropically project）到 Birkhoff 多面体上，但没有写出这句话对应的数学 [@xie2026mhc]。精确的含义来自 Mena 等人的定理 1 [@mena2018gumbel]：

::: eq Sinkhorn 算子是一个带熵正则的最大化问题 | (16.25)
$$\lim_{t\to\infty}\mathrm{SK}^{(t)}\big(\exp(X/\tau)\big)=\arg\max_{P\in\mathcal B_n}\ \langle P,X\rangle_F+\tau\,h(P),\qquad h(P)=-\sum_{i,j}P_{ij}\log P_{ij}$$
$X\in\R^{n\times n}$：logit 矩阵（mHC 里是 $\tilde H^{\rm res}_l$，$\tau=1$）；$\langle P,X\rangle_F=\sum_{ij}P_{ij}X_{ij}$：“$P$ 与 logit 有多一致”；$h(P)$：$P$ 的熵，$P$ 越均匀越大。Mena 等人还证明 $\tau\to0^+$ 时极限是使 $\langle P,X\rangle_F$ 最大的**置换矩阵**（线性指派问题），这一条要求 $X$ 的元素独立地取自连续分布，并且只是几乎必然成立。
:::

为什么是这样？下面用拉格朗日乘子把最优解的形状推出来（笔者补的推导，定理的严格证明见原文）。

::: pre 拉格朗日乘子 | 会用拉格朗日乘子求带等式约束的极值的可以跳过
要在等式约束 $c_1(P)=0,\dots,c_m(P)=0$ 下求 $f(P)$ 的最大值，给每个约束配一个待定的数（乘子）$\lambda_k$，写出 $\Lambda=f-\sum_k\lambda_kc_k$。在最优点（约束都是线性的，或者各约束的梯度线性无关时），存在一组乘子使 $\Lambda$ 对每个自由变量的偏导都等于 0；约束之间有冗余时，乘子不唯一，但这组方程照样成立。直观地说：在最优点，沿着任何一个不破坏约束的方向走一小步，$f$ 都不再变大，所以 $f$ 的梯度只能是各约束梯度的组合。最小的例子：在 $p_1+p_2=1$ 上最大化熵 $-p_1\log p_1-p_2\log p_2$，$\partial\Lambda/\partial p_i=-\log p_i-1-\lambda=0$ 给出 $p_1=p_2=e^{-1-\lambda}$，再由约束得 $p_1=p_2=\tfrac12$。熵的定义见 [M05 §1](../math/m05-information-theory.html#p1)。
:::

::: deriv 熵正则问题的最优解正是 D₁ exp(X) D₂
$\Lambda=\langle P,X\rangle_F+h(P)-\mathbf u^\top(P\mathbf 1-\mathbf 1)-\mathbf v^\top(P^\top\mathbf 1-\mathbf 1)$ || 取 $\tau=1$；$\Lambda$ 是拉格朗日函数（不用 $\mathcal L$，它在本课是损失）；$\mathbf u,\mathbf v\in\R^n$ 是行和、列和约束的乘子
$\dfrac{\partial\Lambda}{\partial P_{ij}}=X_{ij}-\log P_{ij}-1-u_i-v_j=0$ || 对每个 $P_{ij}$ 求导并令其为 0
$P_{ij}=e^{-1-u_i}\cdot e^{X_{ij}}\cdot e^{-v_j}$ || 解出 $P_{ij}$；它自动为正，非负约束不起作用
$P=D_1\exp(X)\,D_2$ || $D_1=\diag(e^{-1-u_i})$，$D_2=\diag(e^{-v_j})$
:::

把这几行连成一个论证，还要补三个事实。第一，$\mathcal B_n$ 是有界闭集、目标函数连续，所以最大值点存在；目标函数严格凹（线性项加上严格凹的熵），所以它唯一。第二，最大值点不会落在边界上（某个 $P_{ij}=0$）：熵对 $P_{ij}$ 的偏导 $-\log P_{ij}-1$ 在 $P_{ij}\to0^+$ 时趋于 $+\infty$，从边界点往 $J_n$ 的方向挪一小步，目标函数一定变大，所以非负约束在最优点不起作用，只需要上面两组等式约束的乘子。第三，Sinkhorn 定理说“$D_1\exp(X)D_2$ 且双随机”的矩阵也唯一。最大值点满足上面的条件，两者只能是同一个矩阵，也就是 SK 的极限。顺带一提，$2n$ 个行和、列和约束里有一个是多余的（[§4](#p4) 的维数推导），所以乘子不唯一：把每个 $u_i$ 加上同一个数 $c$、每个 $v_j$ 减去 $c$，$P_{ij}$ 不变。这正对应 3×3 算例里 $D_1$、$D_2$ 只能确定到一个比例。于是 $H^{\rm res}_l$ 的含义是：在 Birkhoff 多面体里，既尽量顺着 logit 的偏好、又尽量均匀（熵大）的那个折中点。logit 越悬殊（相当于 $\tau$ 越小），结果越靠近某个顶点（置换矩阵，包括 $I$），而上一小节说过，这时 SK 收敛也越慢。想让 $H^{\rm res}$ 接近 $I$，就得让对角 logit 远大于其他 logit，这与“20 轮就够准”之间有一点张力。下面把这点张力算成数。

::: ex 对角 logit 要多大，SK 又会慢多少（笔者算例）
设 logit 矩阵是 $cI$：对角为 $c$，其余为 0。取指数后对角是 $e^c$、其余是 $e^0=1$，每行、每列之和都是 $e^c+3$，一次归一化就得到双随机矩阵

$$H=\frac{1}{e^c+3}\begin{pmatrix}e^c&1&1&1\\1&e^c&1&1\\1&1&e^c&1\\1&1&1&e^c\end{pmatrix}.$$

它对角元素是 $e^c/(e^c+3)$；特征值是 1（方向 $\mathbf 1_4$）和三重的 $(e^c-1)/(e^c+3)$（与 $\mathbf 1_4$ 正交的方向）。$H$ 对称，所以特征值就是奇异值。

- $c=0$：$H=J_4$，每个元素 0.25。logit 全为 0 时 mHC 的 $H^{\rm res}$ 是“彻底搅匀”，不是恒等映射。
- $c=2$：对角 $7.389/10.389=0.711$，$\sigma_2=6.389/10.389=0.615$，$\sigma_2^2=0.378$。
- $c=5$：对角 $148.41/151.41=0.980$，$\sigma_2=0.974$，$\sigma_2^2=0.948$，$0.948^{20}\approx0.34$。

纯粹的 $cI$ 一次就收敛，但真实的 logit 总带着输入相关的扰动，扰动造成的列和误差按 Knight 定理每轮乘 $\sigma_2^2$。$c=5$ 时 20 轮只能把它缩小到约三分之一。笔者在 $cI$ 上加标准正态扰动各模拟 200 次，20 轮后最大列和误差的中位数：$c=0$ 约 $2\times10^{-13}$，$c=2$ 约 $4\times10^{-7}$，$c=5$ 约 $1\times10^{-2}$。也就是说，在这个设定下，想让 $H^{\rm res}$ 的对角贴近 0.98，列和就只能精确到 1% 左右；[§7](#p7) 里 27B 模型单层反向增益 1.01–1.06 是同一量级（这是笔者的推测性联系，论文没有这样分析）。
:::

## mHC 的参数化：三个映射从哪里来

最后看 $\tilde H$ 本身怎么算。mHC 的做法和 HC 的动态版本同一个思路（静态偏置 + 输入相关的动态部分），但有三处不同 [@xie2026mhc]：

::: eq mHC 的原始系数（mHC 论文式 (7)，本课记号） | (16.26)
$$\begin{aligned}\hat{\mathbf x}_l&=\RMSNorm\big(\operatorname{vec}(X_l)\big)\in\R^{1\times nd},\\ \tilde H^{\rm pre}_l&=\alpha^{\rm pre}_l\,\big(\hat{\mathbf x}_lW^{\rm pre}_l\big)+S^{\rm pre}_l,\\ \tilde H^{\rm post}_l&=\alpha^{\rm post}_l\,\big(\hat{\mathbf x}_lW^{\rm post}_l\big)+S^{\rm post}_l,\\ \tilde H^{\rm res}_l&=\alpha^{\rm res}_l\,\operatorname{mat}\big(\hat{\mathbf x}_lW^{\rm res}_l\big)+S^{\rm res}_l.\end{aligned}$$
$\operatorname{vec}(X_l)$：把 $n$ 条流首尾相接成一个 $1\times nd$ 的行向量；$W^{\rm pre}_l,W^{\rm post}_l\in\R^{nd\times n}$，$W^{\rm res}_l\in\R^{nd\times n^2}$：动态投影（论文记作 $\varphi_l$）；$\operatorname{mat}$ 把 $n^2$ 个数排成 $n\times n$；$S^{\rm pre}_l,S^{\rm post}_l\in\R^{1\times n}$，$S^{\rm res}_l\in\R^{n\times n}$：静态偏置（论文记作 $\mathbf b_l$，本课沿用 V4 论文的 $S$ [@dsv4]，以免与第 11 课的专家偏置 $b_i$ 混淆）；$\alpha$：三个可学习的门控标量，初值 0.01。
:::

::: eq mHC 的三个约束（mHC 论文式 (8)） | (16.27)
$$\begin{aligned}H^{\rm pre}_l&=\sigma\big(\tilde H^{\rm pre}_l\big)\in(0,1)^{1\times n},\\ H^{\rm post}_l&=2\,\sigma\big(\tilde H^{\rm post}_l\big)\in(0,2)^{1\times n},\\ H^{\rm res}_l&=\mathrm{SK}_{20}\big(\tilde H^{\rm res}_l\big)\in\mathcal B_n\ (\text{近似}).\end{aligned}$$
$\sigma$ 是 sigmoid；$\mathrm{SK}_{20}$ 即 (16.24)。
:::

三处不同，各有理由：

1. **系数由全部 $n$ 条流一起算。** HC 里 $\mathcal A_r$ 的第 $i$ 行只看第 $i$ 条流 (16.10)；mHC 先把 $n$ 条流展平成一个 $nd$ 维向量，每个系数都能看到所有流，论文的说法是为了保留完整的上下文信息 [@xie2026mhc]。
2. **没有 tanh。** 约束已经由 sigmoid 与 SK 施加，数值不会跑出范围。
3. **读出和写入也非负。** 论文指出非负约束可以防止正负系数互相抵消 [@xie2026mhc]。写入用 $2\sigma$：logit 为 0 时恰好是 1，和普通残差“增量原样加回去”对齐。读出只要非负就够，整体大小无所谓，因为子层 $\mathcal F_l$ 一进来先做自己的 RMSNorm，把输入的整体尺度抹掉了（$\RMSNorm(c\,\mathbf x)=\RMSNorm(\mathbf x)$，$c>0$，忽略 $\epsilon$；这是笔者的推导）。

::: fig figures/l16-mhc-param.svg | 图 16.9 mHC 怎样从残差流算出三个映射（DeepSeek-V4-Flash 的维度） | wide scroll
**看中间那一排 24 个格子：一个子层的全部 mHC 系数只有 24 个数。** 四条流（$4\times4096$）展平成 $1\times16384$，整体做 RMSNorm，乘一个 $16384\times24$ 的矩阵，得到 24 个数：前 4 个经门控、偏置与 sigmoid 成为读出 $H^{\rm pre}$，中间 4 个经 $2\sigma$ 成为写入 $H^{\rm post}$，后 16 个排成 $4\times4$、取指数、做 20 轮 SK，成为混合 $H^{\rm res}$。按 mHC 论文式 (7)–(9) 与 V4-Flash 的配置绘制 [@xie2026mhc; @hf-v4flash-code]。
:::

一个没有答案的问题：论文没有给出静态偏置 $S$ 的初值，发布的代码只加载训练好的权重。所以 mHC 在训练开始时 $H^{\rm res}$ 是否接近 $I$（这需要很大的对角 logit），本课无法回答。门控 $\alpha$ 初值只有 0.01，开训时 $\tilde H^{\rm res}_l\approx S^{\rm res}_l$，上面“对角 logit”的算例正好给出两种极端：$S^{\rm res}=0$ 时起点是彻底搅匀的 $J_4$；对角取 5 时起点接近 $I$（对角 0.98），代价是 SK 收敛变慢。

::: ex mHC 的参数与算术都是零头（DeepSeek-V4-Flash：d = 4096，43 个块 = 86 个子层）
**参数**：每个子层 $W_l$ 共 $nd\times(n^2+2n)=16384\times24=393{,}216$ 个，加 24 个静态偏置、3 个门控，合计 $393{,}243$ 个；86 个子层共 $33{,}818{,}898$ 个，约 3380 万，占总参数 284B 的 $0.012\%$。

**算术**（笔者估算，乘加各算一次）：投影 $2\times16384\times24=786{,}432$；读出 $H^{\rm pre}X$ 为 $2\times4\times4096=32{,}768$；混合 $H^{\rm res}X$ 为 $2\times16\times4096=131{,}072$；写入 $32{,}768$；SK 约 $20\times2\times16\times2=1{,}280$。合计约 $984{,}320$ 次，86 个子层约 $8.5\times10^7$ 次。V4-Flash 每 token 激活 13B 参数，前向约 $2\times13\times10^9=2.6\times10^{10}$ 次，mHC 只占约 $0.33\%$。

参数与算术都不到百分之一，那么 mHC 的代价在哪里？下一页。
:::

::: key 本页要点
- SK：$M^{(0)}=\exp(\tilde H^{\rm res})$，每轮先列归一化、再行归一化，20 轮 (16.23)(16.24)。它只做行列缩放，结果是 $D_1AD_2$，交叉比不变；Sinkhorn 定理保证正矩阵的双随机缩放存在且唯一。
- 3×3 算例：第 1 轮后列和 $136/135,136/135,133/135$，之后每轮误差约乘 0.0436。收敛速率由极限的第二大奇异值决定，渐近地每轮乘 $\sigma_2(P)^2$（Knight 2008）；本例因对称性起作用的是 $0.2087^2=0.0436$。$\begin{pmatrix}1&1\\0&1\end{pmatrix}$ 不满足定理前提，只有 $O(1/t)$，20 轮后误差 $1/41$；logit 相差 10 左右时 $\sigma_2^2=0.973$，也会剩下约 2%。
- 截断后行和精确、列和近似：前向增益为 1，反向略大于 1。
- “熵投影”：SK 的极限 = $\arg\max_{P\in\mathcal B_n}\langle P,X\rangle_F+h(P)$ (16.25)，拉格朗日条件直接给出 $D_1\exp(X)D_2$。logit 越尖锐越接近置换矩阵，也越难收敛：对角 logit 为 5 时 $H^{\rm res}$ 对角 0.98，但 $\sigma_2^2=0.948$，20 轮只把误差缩小到约三分之一。
- 参数化：展平全部流 → RMSNorm → 一次 $nd\times24$ 的矩阵乘 → 门控 + 静态偏置 → $\sigma$、$2\sigma$、SK (16.26)(16.27)。V4-Flash 每子层 393,243 个参数，全模型 0.012%，算术 0.33%。
:::

# 基础设施：让 4 倍宽的残差流跑得动 | 访存账、核融合与混合精度、重计算、DualPipe 的改造与 6.7% | 30

上一页算过，mHC 的参数与算术都是零头。可是 HC 论文实测显存多了 10%–28% [@zhu2025hc]，mHC 论文也专门用一整节写基础设施 [@xie2026mhc]。原因在 [第 4 课 §4](lesson-04-inference-kv-cache.html#p4) 讲过的“带宽墙”：GPU 算得远比搬数据快，一个操作如果每搬一个数只做一两次运算，它的耗时由搬数据决定，而不是由算决定。残差流上的这些操作，正是这一类。算力与带宽的一般算术见 [M10](../math/m10-compute-arithmetic.html)。

## 访存账：每个 token、每个子层要搬多少数

mHC 论文的表 2 逐项数了残差流上每个操作的读写量（每个 token、一个子层的前向，不含子层 $\mathcal F$ 内部）[@xie2026mhc]：

| 操作 | 读（元素） | 写（元素） |
|---|---|---|
| 普通残差：相加 | $2d$ | $d$ |
| HC：算三个映射的系数 | $nd$ | $n^2+2n$ |
| HC：读出 $H^{\rm pre}X$ | $nd+n$ | $d$ |
| HC：写入 $(H^{\rm post})^\top\mathcal F$ | $d+n$ | $nd$ |
| HC：混合 $H^{\rm res}X$ | $nd+n^2$ | $nd$ |
| HC：相加 | $2nd$ | $nd$ |
| **HC 合计** | $(5n+1)d+n^2+2n$ | $(3n+1)d+n^2+2n$ |

::: eq HC 每个子层的访存量 | (16.28)
$$\underbrace{(5n+1)d+n^2+2n}_{\text{读}}+\underbrace{(3n+1)d+n^2+2n}_{\text{写}}\ \overset{n=4}{=}\ 34d+48\qquad\text{vs.}\qquad 3d$$
逐行相加即得（研究笔记复核过）。$d$ 为几千时，比值 $(34d+48)/3d\approx11.3$，与 $d$ 几乎无关。
:::

::: ex 换成字节
按 bf16（每个元素 2 字节，这是笔者的换算假设）：27B 模型（$d=2560$）每 token 每子层，普通残差 15,360 字节，未融合的 HC 174,176 字节；V4-Flash（$d=4096$）是 24,576 字节对 278,624 字节。V4-Flash 的 86 个子层加起来，一个 token 的一次前向，残差流部分要搬约 2.1 MB，HC 要搬约 24.0 MB。这还只是前向；训练时反向要再读写一遍，还要为反向保存激活。
:::

::: ex 算术强度：mHC 离“脊点”有多远（笔者估算）
[第 4 课 (4.14)–(4.16)](lesson-04-inference-kv-cache.html#p4) 定义了**算术强度** = 运算次数 ÷ 从显存搬运的字节数（FLOP/字节；第 4 课记作 $I$，本课 $I$ 已用来表示单位矩阵，这里不写符号），以及屋顶线的**脊点**：H100 SXM 约为 295 FLOP/字节，强度低于它的操作跑不满算力，耗时由搬运决定。按 V4-Flash 的数（$d=4096$，bf16，每 token 每子层）来算 mHC：

- 运算：[§5](#p5) 末尾估算的约 984,320 次，其中约 80% 是那个 $16384\times24$ 的投影。
- 字节：未融合时 $(34d+48)\times2=278{,}624$ 字节（上面的算例）。投影矩阵本身的读取由一批 token 分摊，这里不计。
- 强度：$984{,}320/278{,}624\approx3.5$ FLOP/字节，不到 H100 脊点的 $1/80$。

即使访存压到 V4 实现的 $20d$（[§7](#p7)），强度也只有 $984{,}320/(20\times4096\times2)\approx6.0$。所以 mHC 的耗时几乎完全由字节决定：少搬一个字节就少花一份时间。这就是下面的融合、重计算，以及 [§7](#p7) 的 Single-Pass，全都在数“读写多少个 $d$”的原因。
:::

除了访存，HC 还有两笔账：反向传播要保存的残差流激活宽了 $n$ 倍；流水线并行时，相邻两个 stage 之间传的是 $n\times d$ 的残差流而不是 $d$，通信量也是 $n$ 倍 [@xie2026mhc]。

## 核融合与混合精度

**融合**（kernel fusion）的意思是把几个本来要分别从显存读、写回显存的操作放进同一个 GPU 核，中间结果留在片上，只读一次、写一次。mHC 做了三件事 [@xie2026mhc]。

**第一，把 RMSNorm 的除法挪到矩阵乘之后。** (16.26) 要先对 $16384$ 维的向量做 RMSNorm，再乘 $W_l$。这需要先读一遍 $\mathbf x$ 算范数、写出归一化后的向量、再读一遍做矩阵乘。论文注意到可以交换次序，并且数学上完全等价：

::: deriv 先乘后除
$\RMSNorm(\mathbf x)\,W=\Big(\dfrac{\mathbf x\odot\mathbf g}{r}\Big)W$ || RMSNorm 的定义：$r=\norm{\mathbf x}_2/\sqrt{nd}$ 是一个标量（实现里根号下还加一个 $\epsilon$，$r$ 仍是标量，下面的推导不变），$\mathbf g\in\R^{nd}$ 是逐元素的增益（第 5 课的记号，与 [§1](#p1) 的梯度 $\mathbf g_l$ 无关）
$=\dfrac1r\,\mathbf x\,\big(\diag(\mathbf g)\,W\big)$ || $\mathbf x\odot\mathbf g=\mathbf x\,\diag(\mathbf g)$；标量 $1/r$ 可以提到矩阵乘外面
$=\dfrac1r\,\mathbf x\,W'$ || $W'=\diag(\mathbf g)W$ 离线合并好，只存 $W'$
:::

::: eq RMSNorm 与投影融合 | (16.29)
$$\big[\tilde H^{\rm pre}_l,\tilde H^{\rm post}_l,\tilde H^{\rm res}_l\big]=\frac1r\big[\alpha^{\rm pre}_l\,\cdot,\ \alpha^{\rm post}_l\,\cdot,\ \alpha^{\rm res}_l\,\cdot\big]\big(\mathbf x\,W'_l\big)+S_l,\qquad r=\frac{\norm{\mathbf x}_2}{\sqrt{nd}}$$
同一个核里一边做 $\mathbf x W'_l$（输出只有 24 个数），一边累加 $\norm{\mathbf x}^2$，最后把 24 个数除以 $r$。V4 的推理代码正是这样写的 [@hf-v4flash-code]。
:::

**第二，混合精度。** 投影矩阵用 tfloat32，残差流 $\mathbf x$ 用 bfloat16，算出来的系数用 float32 [@xie2026mhc]。系数只有 24 个，用高精度几乎不花钱，而 SK 要对它们取指数、反复做除法，精度不够时误差会被放大（这一句是笔者的理解）。

**第三，把“写入 + 混合 + 相加”融成一个核。** 未融合时，写入读 $d+n$、写 $nd$，混合读 $nd+n^2$、写 $nd$，相加读 $2nd$、写 $nd$：共读 $(3n+1)d$、写 $3nd$（不计 $n^2+n$ 个系数的读取）。融合之后只需要读一次 $X_l$（$nd$）和子层输出（$d$），写一次 $X_{l+1}$（$nd$）：读 $(n+1)d$、写 $nd$ [@xie2026mhc]。$n=4$ 时读从 $13d$ 降到 $5d$，写从 $12d$ 降到 $4d$。这些核大多用 TileLang [@tilelang2025] 编写；SK 的 20 轮放在一个核里完成，反向用一个自定义核在片上重算中间结果，而不是把 20 轮的中间矩阵都存下来 [@xie2026mhc]。

::: ana 类比：仓库与工作台
**映射。** 显存（HBM）是远处的大仓库；GPU 核里的寄存器与片上共享内存是手边的小工作台；访存就是在仓库与工作台之间搬货。未融合的 HC，每做一道工序（读出、写入、混合、相加）都把半成品搬回仓库，下一道再搬出来；融合则是把货搬上工作台，一口气做完几道工序，只进出仓库一次。

**收益。** 它说明了为什么算术只占 0.33% 的 mHC 仍要花时间：工人手很快，时间都花在来回搬运上；也说明了融合为什么有效：工序一道没少，搬运少了。

**失效的地方。** 工作台很小，放不下整条 $n\times d$ 的残差流，只能切成小块（tile）一块一块地做；如果某道工序要先看完整条流（例如 RMSNorm 的范数、系数 $A_l$ 的归约），就得先把整条流过一遍才能开始下一道，这正是 [§7](#p7) 里 V4.1 要解决的问题。另外，GPU 上搬运与计算可以同时进行，不像一个工人只能二选一。
:::

## 重计算：每隔几层存一次

**问题。** 反向传播需要前向的中间结果。mHC 的每个子层都有 $n$ 倍宽的 $X_l$ 和几个中间量，全存下来显存吃不消。**重计算**（recomputation）的办法是：前向时丢掉中间结果，反向需要时再算一遍。mHC 只重跑自己那几个便宜的核，不重跑昂贵的子层 $\mathcal F$（$\mathcal F$ 的输出每层都保存）[@xie2026mhc]。

具体做法：把 $L$ 个子层切成长度为 $L_r$ 的块，每块只保存块首的输入 $X_{l_0}$（$nd$ 个数）；反向到这一块时，从 $X_{l_0}$ 出发把块内每层的 $X_l$（$nd$）、$H^{\rm pre}_lX_l$（$d$）、它的 RMSNorm（$d$）重算出来，共 $(n+2)d$ 每层。于是每个 token 需要的激活显存是“所有块首”加上“一个块的中间量”：

::: eq 重计算的显存与最优块长（mHC 论文式 (20)） | (16.30)
$$f(L_r)=nd\Big\lceil\frac{L}{L_r}\Big\rceil+(n+2)d\,L_r,\qquad L_r^*\approx\sqrt{\frac{nL}{n+2}}$$
块长太短，要存的块首多；块长太长，一次要重建的中间量多。
:::

::: ana 类比：打游戏存档
**映射。** 每 $L_r$ 关存一次档（块首 $X_{l_0}$）；反向传播需要某一关的中间状态时，从最近的存档读档，把这几关重打一遍（重算）。只重打便宜的 mHC 小关，昂贵的 $\mathcal F$ 大关每关都留了录像（输出已保存）。

**收益。** (16.30) 的 U 形一眼可见：存档太密，存档本身占满硬盘；存档太疏，一次要重打、要记住的中间状态太多。

**失效的地方。** 游戏重打可能打出不同结局，重计算必须得到与第一次前向相同的中间量，否则梯度就错了；而且 (16.30) 只数显存，不数重打花的时间。
:::

::: deriv 最优块长
$f(L_r)\approx\dfrac{ndL}{L_r}+(n+2)d\,L_r$ || 去掉取整
$f'(L_r)=-\dfrac{ndL}{L_r^2}+(n+2)d=0$ || 求导令其为 0
$L_r^*=\sqrt{\dfrac{nL}{n+2}}$ || 解出；$f''>0$，是最小值
:::

::: ex 27B 与 V4-Flash 的最优块长
$n=4$，$L=60$：$L_r^*=\sqrt{240/6}=\sqrt{40}=6.32$。整数里 $L_r=6$ 最好：$f=4\times10+6\times6=76$（单位 $d$）。对比：每层都存 $X_l$ 要 $4\times60=240$；三种中间量全存要 $6\times60=360$。重计算把这部分显存降到 1/3 到 1/5。

V4-Flash 若按 86 个子层算：$\sqrt{4\times86/6}=7.57$，整数最优 $L_r=8$，$f=4\times11+6\times8=92$。（论文没有说式 (20) 里的 $L$ 指块数还是子层数，这里按子层算。）

实践中，mHC 把重计算块的边界与流水线 stage 的边界对齐 [@xie2026mhc]。
:::

::: fig figures/l16-recompute.svg | 图 16.10 重计算的块长与每 token 的激活显存 | scroll
**看两条 U 形曲线的最低点：块长取 6 到 8 时显存最少，远低于上方两条参考虚线。** 纵轴是 (16.30) 的 $f(L_r)$，单位 $d$。琥珀色 $L=60$（27B），蓝色 $L=86$（V4-Flash 的子层数）；长虚线是不做重计算、三种中间量全存，点线是每层只存 $X_l$。由生成脚本按式 (20) 计算。
:::

## 与 DualPipe 的重叠

DeepSeek-V3 的训练用 DualPipe 流水线：让一个 stage 在做计算的同时，把 MoE 的 {{all-to-all::每张卡都要给其他每张卡发数据的集体通信；MoE 用它把 token 送到专家所在的卡（DISPATCH），再把专家输出收回来（COMBINE）}} 通信（DISPATCH、COMBINE）和流水线的发送接收藏在计算后面（[第 12 课 §6](lesson-12-v3-infrastructure.html#p6)）[@dsv3]。mHC 的核插进这个精心排好的时间表里，会带来两个麻烦：它们本身占时间；更糟的是，如果某个 mHC 核恰好卡在“通信结束之后、下一段计算开始之前”的关键路径上，就会把通信的等待暴露出来。

先交代两个 GPU 编程的词。**流**（stream）是一条按顺序执行的任务队列：同一条流里的核一个接一个跑，不同流里的核可以同时跑；给一条流设**高优先级**，GPU 腾出计算单元时先安排它的核。**常驻核**（persistent kernel）是一种写法：核一启动就占住一批计算单元，在核内部循环领取任务直到全部做完，省掉了反复启动核的开销，代价是它跑着的时候别的核很难插进来。有了这两个词，mHC 对 DualPipe 的三处调整就好懂了 [@xie2026mhc]：

1. 把 MLP（MoE）子层的“写入 + 混合”融合核 $\mathcal F_{\rm post,res}$ 放到一条**高优先级**的计算流上，使它不被普通计算流上的长核挡住（论文图 4 中，它位于 COMBINE 通信与流水线收发之间）。
2. 注意力子层的长时间操作**不用常驻核**（persistent kernel），使它们可以被高优先级的核抢占。
3. 重计算与流水线通信解耦：每个 stage 的首个激活已经缓存在本地，重算不必等待通信。

::: fig figures/l16-dualpipe.svg | 图 16.11 mHC 的核在 DualPipe 里怎样与通信重叠（据 mHC 论文图 4 重绘） | wide scroll
**看最下面一行的两个红块：它们夹在 COMBINE 与流水线收发之间，走的是单独的高优先级流。** 上排是普通计算流（MLP、ATTN 的前向 F、反向 B、权重梯度 W，以及 mHC 的读出核 $\mathcal F_{\rm pre}$ 与写入混合核 $\mathcal F_{\rm post,res}$，上标 A/M 表示注意力/MLP 子层），中排是通信流。各流内部的先后次序据 mHC 论文图 4；论文注明块长只是示意，本图中通信块与计算块的相对水平位置同样只是示意 [@xie2026mhc]。
:::

## 6.7%：最终的代价

做完这些，mHC 论文报告：在内部的大规模训练中，$n=4$ 的 mHC 只带来 6.7% 的额外训练时间 [@xie2026mhc]。DeepSeek-V4 的论文也给出 6.7%，但口径是“重叠后 {{1F1B::流水线并行的一种排程：每个 stage 在稳定阶段交替做一个微批的前向（1F）和另一个微批的反向（1B），第 12 课有详细讲解}} 流水线 stage 的 6.7%” [@dsv4]。两个数相同，分母未必相同；两篇论文都没有说明是在哪个模型、多少张卡上测的（研究笔记列为未核实）。

::: key 本页要点
- mHC 的瓶颈不在参数（0.012%）和算术（0.33%），而在访存：未融合的 HC 每 token 每子层读写 $34d+48$ 个元素，是普通残差 $3d$ 的约 11 倍 (16.28)；算术强度只有约 3.5 FLOP/字节，远低于 H100 的脊点 295，耗时由字节决定；另有 $n$ 倍的激活与流水线通信。
- 核融合：RMSNorm 的除法挪到矩阵乘之后，数学等价 (16.29)；系数用 fp32；“写入 + 混合 + 相加”融合后读 $13d\to5d$、写 $12d\to4d$；SK 在单个核里完成。
- 重计算：每 $L_r$ 层存一次块首，$f(L_r)=nd\lceil L/L_r\rceil+(n+2)dL_r$，最优 $L_r^*\approx\sqrt{nL/(n+2)}$ (16.30)；$L=60$ 时 $L_r=6$，76d 对 240d。
- DualPipe：MLP 子层的 post-res 核放高优先级流，注意力不用常驻核以便抢占，重计算与通信解耦。结果：$n=4$ 时额外时间 6.7%（mHC 与 V4 两处口径不同）。
:::

# 实验、V4 与 V4.1：算总账 | 27B 上稳住了什么、学到了什么矩阵、V4 怎样用它、V4.1 怎样再省一半访存 | 35

前五页讲的是机制。这一页看证据与去向：mHC 论文的实验到底显示了什么（哪些是实测、哪些只是从图上读出的数），DeepSeek-V4 怎样把它用进生产模型，V4.1-Flash 又怎样把它的访存再砍一半。最后把本课的所有设计放进一张总账。

## 实验设置

mHC 论文训练了三个规模的 MoE 模型，架构仿照 DeepSeek-V3（MLA 注意力、细粒度专家加共享专家、无辅助损失均衡），分别用普通残差（基线）、HC、mHC 各训一遍，另有一个 3B 模型训到 1T tokens 看数据规模的影响 [@xie2026mhc]：

| | 3B | 9B | 27B | 3B（1T tokens） |
|---|---|---|---|---|
| 总参数 / 激活参数 | 2.97B / 612M | 9.18B / 1.66B | 27.0B / 4.14B | 2.97B / 612M |
| 块数（子层数 $L$） | 12（24） | 18（36） | 30（60） | 12（24） |
| 隐藏维度 $d$ | 1280 | 1920 | 2560 | 1280 |
| 路由 / 激活 / 共享专家 | 64 / 6 / 2 | 64 / 6 / 2 | 72 / 6 / 2 | 64 / 6 / 2 |
| 训练 tokens | 39.3B | 105B | 262B | 1.05T |
| batch（序列）× 步数 | 320 × 30k | 512 × 50k | 1280 × 50k | 2560 × 100k |

HC 与 mHC 都取 $n=4$、门控初值 $\alpha=0.01$，mHC 的 SK 取 $t_{\max}=20$，序列长度 4096 [@xie2026mhc]。tokens 数可以复核：$1280\times4096\times50{,}000=2.62\times10^{11}$，即 262B。三个规模的 tokens 与激活参数之比分别约为 64.3、63.2、63.3（笔者的计算；论文说数据量与参数成比例，没说是哪种参数）。

## 稳定性：从近 3000 到约 1.6

::: fig figures/l16-mhc-gain.svg | 图 16.12 mHC 在 27B 模型上的单层增益与复合增益（据 mHC 论文图 7 重绘，数值为数字化读数） | wide scroll
**看纵轴：和图 16.5 不同，这里是从 0.9 到 1.8 的线性坐标。** (a) 每个子层的 $H^{\rm res}_l$：前向增益在 1.0000 到 1.0005 之间（贴着 1.0 的灰色细带），反向增益在 1.006 到 1.061 之间（绿线）。(b) 复合映射：前向在 1.0001 到 1.0066 之间，反向在第 20 个子层附近达到最大约 1.65（论文正文说约 1.6）。据 mHC 论文图 7 重绘 [@xie2026mhc]，数值由研究笔记从矢量图中读出，不是论文印刷的数字。
:::

论文的结论是：复合映射的反向增益最大约为 1.6，比 HC 的近 3000 低三个数量级 [@xie2026mhc]；按两张图的数字化读数，比值是 $2825/1.649\approx1713$，$\log_{10}1713=3.23$。三个细节值得注意：

1. **反向增益不是 1，而是略大于 1。** 原因在 [§5](#p5)：SK 只做 20 轮、以行归一化收尾，列和只近似为 1。这些偏差是**逐层相加**、而不是相乘地积累的：记 $\mathbf 1^\top H_k=\mathbf 1^\top+\boldsymbol\delta_k^\top$（第 $k$ 层列和的偏差；行和精确为 1，所以 $\boldsymbol\delta_k$ 的分量之和为 0），逐层展开得 $\mathbf 1^\top H_L\cdots H_1=\mathbf 1^\top+\sum_k\boldsymbol\delta_k^\top H_{k-1}\cdots H_1$。第 20 到 60 个子层的单层反向增益（1.006–1.051）若直接相乘是 3.1，实测的复合反向增益只有 1.65：偏差是一项项加上去的，而且每一项还会被其余各层的混合抹平一部分（笔者的推导，数字取自图 16.12 的数字化读数）。但它有上界：[§4](#p4) 证明过，行和精确为 1 的非负矩阵之积，反向增益不超过 $n=4$。
2. **前向增益也不恰好是 1。** 按 (16.24)，每个 $H^{\rm res}_l$ 的行和应当精确为 1，对 token 取平均也不改变行和，复合的前向增益应为 1；实测最高约 1.0066。论文没有解释这个偏差（可能与低精度计算、代码里的 eps 或统计方式有关，研究笔记列为未核实）。所以准确的说法是“前向增益按构造约为 1”，而不是“实测恰为 1”。
3. **训练曲线。** 27B 模型上，mHC 的最终 loss 比基线低 0.021 [@xie2026mhc]。下表是研究笔记从论文图 2、图 5 的曲线上读出的几个点（数字化读数，曲线本身噪声大，只看趋势）：

| 训练步数 | 5k | 12k | 20k | 30k | 50k |
|---|---|---|---|---|---|
| HC 相对基线的 loss 差 | −0.0443 | −0.0333 | −0.0256 | −0.0211 | −0.0166 |
| mHC 相对基线的 loss 差 | −0.0435 | −0.0342 | −0.0310 | −0.0275 | −0.0218 |
| 梯度范数：基线 / HC / mHC | 0.091 / 0.106 / 0.143 | 0.072 / 0.103 / 0.102 | 0.064 / 0.127 / 0.092 | 0.057 / 0.144 / 0.107 | 0.039 / 0.133 / 0.050 |

12k 步之前 HC 与 mHC 的领先幅度差不多；之后 HC 的领先缩得更快，梯度范数一路升到 0.14 左右，而 mHC 的梯度范数跟着基线的趋势走（40k、45k 两处的陡降是学习率按步衰减的位置，论文的学习率在 80% 和 90% 的步数处各降一次 [@xie2026mhc]）。

## 学到了什么矩阵

::: fig figures/l16-learned-maps.svg | 图 16.13 27B 模型学到的残差映射：HC 与 mHC 各 6 个 4×4 矩阵（据 mHC 论文图 8 重绘） | wide scroll
**看最右边一列：HC 的 60 层之积全是上百的大数，mHC 的 60 层之积每个元素都接近 0.25。** 每行左边三个是第 1、30、60 个子层的 $H^{\rm res}_l$，右边三个是第 1–30、31–60、1–60 个子层的复合映射；矩阵下方是论文图上印出的最大行和、最大列和（绝对值）。所有数字取自 mHC 论文图 8 印出的两位小数（研究笔记逐格转录，并用图上的行和、列和标签交叉核对过）[@xie2026mhc]。
:::

从这 12 个矩阵能读出几件事（笔者的读法）。记号：$\prod_{a..b}$ 是第 $a$ 到第 $b$ 个子层的复合映射 $H^{\rm res}_b\cdots H^{\rm res}_a$，也就是 [§3](#p3) 的 $P_{a\to b+1}$。

- **HC 的深层复合几乎是秩 1。** $\prod_{1..60}$ 的奇异值是 488、0.15、0.06、0.003：四条流被压成了同一个方向，再放大约 500 倍。这对应 [§3](#p3) 的“每一层都把信号推向它最会放大的方向”。
- **mHC 的单层大多接近单位矩阵，第 1 层看上去是例外。** mHC 的 $H^{\rm res}_{30}$ 对角元素是 0.95 到 1.00；$H^{\rm res}_1$ 却像是在大量交换：第 3 条新流的 73% 来自旧流 4，第 4 条新流的 69% 来自旧流 3。但按 [§3](#p3) 算 HC 的 $H^{\rm res}_1$ 时同样的前提（第 1 个子层之前，四条流是同一个向量 $\mathbf x$ 的四份副本），这些交换什么也没改变：行和为 1 的矩阵作用在 $\mathbf 1_4\mathbf x^\top$ 上，结果是 $(H\mathbf 1_4)\mathbf x^\top=\mathbf 1_4\mathbf x^\top$，与 $H$ 的具体数值无关。SK 以行归一化收尾，20 轮截断后行和仍精确为 1，所以在精确算术下，损失对生成 $H^{\rm res}_1$ 的那些参数（它的 logit，以及其中的 $W^{\rm res}_1$、$S^{\rm res}_1$、$\alpha^{\rm res}_1$）的梯度恰好为 0。注意不是“对 $H^{\rm res}_1$ 各元素的偏导为 0”：那个偏导是 $G_2X_1^\top=(G_2\mathbf x)\mathbf 1_4^\top$（$G_2=\partial\mathcal L/\partial X_2$，[§3](#p3) 的记号；$Y=HX$ 对 $H$ 的梯度是 $G_YX^\top$），一般不为 0，但它的每一行都是常数，与任何保持行和不变的改动 $\Delta$（$\Delta\mathbf 1_4=\mathbf 0$）的内积都是 0，而 SK 能产生的改动都是这一类。这与 HC 截然不同：HC 的 $H^{\rm res}_1$ 行和悬殊（18.72、−15.28、−14.80、−15.88），直接把副本“掰开”；mHC 的四条流只能靠写入权重 $H^{\rm post}_l$ 在各条流上的差异分开（这一条是笔者的推导，论文没有讨论）。
- **mHC 的深层复合接近 $J_4$。** $\prod_{1..60}$ 与全 0.25 的矩阵最大只差 0.04：经过 60 层，每条流都几乎是四条流的平均，这正是 (16.21) 预言的“越搅越匀”。但 $\prod_{31..60}$ 的对角线上还有 0.62、0.80：学到的矩阵不是随机的，后半段的层在刻意保留一部分流的独立性。
- **一个统计上的细节。** 图中每个矩阵都是对一条序列所有 token 的平均，而“平均之积”不等于“积之平均”。研究笔记用图上的 $\prod_{31..60}$ 乘 $\prod_{1..30}$，得到的第一行 $(-132.3,-110.5,-115.1,-111.0)$ 与图上 $\prod_{1..60}$ 的第一行 $(-132.8,-112.2,-117.0,-113.3)$ 最多相差约 2%，差别就来自这里。

## 基准结果与规模化

27B 模型在 8 个基准上的结果（mHC 论文表 4）：mHC 全部高于基线，8 项中有 7 项高于 HC [@xie2026mhc]。

| | BBH | DROP | GSM8K | HellaSwag | MATH | MMLU | PIQA | TriviaQA |
|---|---|---|---|---|---|---|---|---|
| mHC − 基线 | +7.2 | +6.9 | +7.1 | +1.0 | +4.0 | +4.4 | +2.0 | +3.3 |
| mHC − HC | +2.1 | +2.3 | +0.6 | +0.4 | **−0.4** | +0.4 | +0.6 | +1.3 |

八项平均：基线 53.13，HC 56.70，mHC 57.61。唯一的例外是 MATH（mHC 26.0，HC 26.4）。论文把 BBH、DROP 上的领先写成 “2.1%”“2.3%”，实际是分数的差（百分点），不是相对提升。

规模化方面，论文图 6 显示 mHC 的优势随算力增加只有轻微衰减 [@xie2026mhc]。研究笔记从图上读出：3B、9B、27B 三个规模 mHC 相对基线的 loss 差约为 −0.0292、−0.0233、−0.0212（数字化读数）；3B 模型训到 1T tokens 的过程中，差距从约 −0.024 缓慢收窄到约 −0.015。优势在变小，但没有消失。

## DeepSeek-V4 怎样用它

2026 年 4 月的 DeepSeek-V4 把 mHC 列为三项关键升级之一，另两项是 CSA/HCA 混合注意力与 Muon 优化器（[第 18 课](lesson-18-v4.html)）[@dsv4]。V4 论文给出的理由是：HC 在堆叠很多层时经常出现数值不稳定 [@dsv4]。它的写法换了一套字母：

::: eq V4 论文中的 mHC（V4 式 (1)(2)） | (16.31)
$$X_{l+1}=B_lX_l+C_l\,\mathcal F_l(A_lX_l),\qquad B_l\in\mathcal M=\{M\mid M\mathbf 1_n=\mathbf 1_n,\ \mathbf 1_n^\top M=\mathbf 1_n^\top,\ M\ge0\}$$
对照本课：$A_l=H^{\rm pre}_l$（$1\times n$），$B_l=H^{\rm res}_l$（$n\times n$），$C_l=(H^{\rm post}_l)^\top$（$n\times1$）；$\mathcal M$ 就是 $\mathcal B_n$。$A_l$、$C_l$ 用 sigmoid 约束为非负且有界，$B_l$ 用 20 轮 SK 投影 [@dsv4]。
:::

V4 的其他 mHC 细节 [@dsv4; @hf-v4flash-code]：

- **规模。** V4-Flash 与 V4-Pro 都取 $n_{hc}=4$、$t_{\max}=20$；发布的 config 中是 `hc_mult=4`、`hc_sinkhorn_iters=20`、`hc_eps=1e-06`。每个块的注意力与 FFN 各有一套 mHC 参数，所以 V4-Flash 的 43 个块是 86 个子层，V4-Pro 的 61 个块是 122 个子层。
- **优化器分工。** mHC 的静态偏置与门控因子用 AdamW 训练，动态投影矩阵与模型其余的矩阵一起用 Muon。
- **确定性。** mHC 的投影是一个输出维度只有 24 的矩阵乘；batch 小的时候要把求和维度切开并行（split-k），这时需要确定性的归约顺序，才能让同样的输入得到逐位相同的输出。
- **实现与开销。** 融合核、选择性重计算（重算层间的大部分隐藏状态与全部归一化后的层输入）、调整 DualPipe 的 1F1B 重叠；墙钟开销为重叠后 1F1B 流水段的 6.7%。
- **代码里的进出口。** 嵌入之后把同一个向量复制成 4 条流；最后用一组 sigmoid 权重把 4 条流加权合成一条，再做 RMSNorm 与输出层（HC 论文是直接求和）[@hf-v4flash-code]。

| 模型 | 块数 | 子层数 | $d$ | 每子层 mHC 参数 | 全模型 mHC 参数 | 占总参数 |
|---|---|---|---|---|---|---|
| mHC 27B | 30 | 60 | 2560 | 245,787 | 14.7M | 0.055% |
| DeepSeek-V4-Flash | 43 | 86 | 4096 | 393,243 | 33.8M | 0.012% |
| DeepSeek-V4-Pro | 61 | 122 | 7168 | 688,155 | 84.0M | 0.0052% |
| DeepSeek-V4.1-Flash | 40 | 80 | 5120 | 491,547 | 39.3M | 0.0071% |

表中参数按 $nd(n^2+2n)+(n^2+2n)+3$ 计算（研究笔记复核过），不含输出头与 MTP 块的合流参数。V4.1-Flash 一行的分母是报告的 552B **骨干**参数，不含另外 196B 的 Engram 表（[第 17 课 §2](lesson-17-engram-ocr.html#p2)）。

## V4.1-Flash：Single-Pass mHC

2026 年 9 月的 V4.1-Flash 关心的是部署时的访存（[第 19 课](lesson-19-v4-1-flash.html)）。V4.1 论文指出，V4 的 mHC 实现要跑三个串行的核：先更新残差流 $X_l$，再从 $X_l$ 算出本层的系数，最后用 $A_l$ 做输入混合；加上子层自己的 pre-norm，每个子层的激活访存是 $(4n+4)d$，是理想下界 $(2n+2)d$ 的两倍 [@dsv41]。

为什么不能一个核做完？因为 $A_l$ 要先对**整条** $X_l$（$nd=16{,}384$ 个数，以 V4-Flash 为例）做归约（RMSNorm 的范数和那个 $nd\times24$ 的矩阵乘）才能得到，而输入混合 $A_lX_l$ 需要 $A_l$ 才能开始：必须把 $X_l$ 从显存再读一遍。V4.1 的办法是把输入混合的系数“推迟一格”：

::: eq Single-Pass mHC（V4.1 式 (6)） | (16.32)
$$X_{l+1}=B_lX_l+C_l\,\mathcal F_l\big(A_{l-1}X_l\big),\qquad (A_l,B_l,C_l)=\mathcal H(X_l)$$
$\mathcal H$ 是 (16.26)(16.27) 那套“从残差流算出三个映射”的函数。第 $l$ 个子层的输入混合，用的是上一个子层从 $X_{l-1}$ 算出的 $A_{l-1}$。这样 $A_{l-1}$ 在读 $X_l$ 之前就已经知道，残差更新、输入混合、下一层系数的累加可以在同一次按 tile 的遍历里完成 [@dsv41]。
:::

::: fig figures/l16-single-pass.svg | 图 16.14 V4 的多趟 mHC 与 V4.1-Flash 的 Single-Pass mHC | wide scroll
**看 (a) 里那支红色箭头：多趟实现卡在“先归约出 $A_l$，才能做输入混合”这一步上。** (b) 把输入混合改用上一层预测的 $A_{l-1}$ 之后，一个核就能对 $X$ 的每一小块依次完成三件事，残差流只读一次、写一次。图中 $Y_{l-1}=\mathcal F_{l-1}(\cdot)$ 是上一个子层的输出（$1\times d$），$\hat X_l$ 是送进本子层的混合输入。据 V4.1 论文 §2.4.1 与发布的推理代码绘制 [@dsv41; @hf-v41flash-code]；“读、写各 $(n+1)d$”也出自该节：论文说 Single-Pass 使残差流只读一次、写一次，达到理想映射的读 $(n+1)d$、写 $(n+1)d$。
:::

V4.1 论文说，这个“推迟一格”在实验中带来的性能损失可以忽略，但没有给出数字 [@dsv41]。按论文的描述，预训练阶段仍用多个核实现，部署时换成单个核（Mega-mHC）：单核版的原始 mHC 是 $(3n+2)d$，单核版的 Single-Pass 达到下界 $(2n+2)d$，相对原实现访存减半 [@dsv41]。发布的代码与之一致：每个子层算出的系数留给下一个子层用，第一个子层没有“上一层”，就用只取第 0 条流的 one-hot 读出 [@hf-v41flash-code]。

::: fig figures/l16-io-bars.svg | 图 16.15 残差流每个 token、每个子层的访存量（n = 4） | scroll
**看条形的长度：未融合的 HC 是普通残差的 11 倍，V4.1 的 Single-Pass 把它压到了 10d。** 前两行来自 mHC 论文表 2（不含子层自己的 pre-norm），后三行来自 V4.1 论文 §2.4.1（含 pre-norm 的一读一写），口径不完全相同。右侧字节数按 V4.1-Flash 的 $d=5120$、bf16 换算，是笔者的假设（V4.1 提到 Mega-mHC 里有 FP8 转换，实际字节可能更少）。由生成脚本计算。
:::

::: ex V4.1-Flash 每个 token 省下多少访存
$n=4$，$d=5120$，按 bf16：多趟 $20d=102{,}400$ 个元素，即 204,800 字节；单核 mHC $14d$，143,360 字节；Single-Pass $10d$，102,400 字节。乘以 80 个子层：每个 token 分别约 16.4 MB、11.5 MB、8.2 MB。对长提示词的 prefill，这笔账要再乘以提示的 token 数。
:::

## 算总账

| | 普通残差（Pre-LN） | HC | mHC（论文） | V4 | V4.1-Flash |
|---|---|---|---|---|---|
| 残差流条数 | 1 | $n=4$ | 4 | 4 | 4 |
| $H^{\rm res}$ 的约束 | 恒为 1 | 无（任意实数） | 双随机（SK 20 轮） | 同 mHC | 同 mHC |
| 读出 / 写入 | 1 / 1 | 任意 / 任意 | $\sigma$ / $2\sigma$ | 同 mHC | 读出推迟一格 |
| 每子层额外参数 | 0 | OLMo-1B 每模块 12,314 | 27B：245,787 | Flash 393,243 | 491,547 |
| 每子层访存（$d$ 的倍数） | 3 | 34（未融合） | 18（融合后，笔者推算） | 20（多趟） | 10（下界） |
| 27B 复合增益（反向） | 1 | 约 3000 | 约 1.6 | — | — |
| 代价 | — | 显存 +9.7%–28% | 训练时间 +6.7% | 1F1B 流水段 +6.7% | 性能损失“可忽略” |

表中 mHC 的 18 是笔者按 [§6](#p6) 与图 16.14(a) 的口径推算的：算系数的核读 $nd$；读出核读 $nd$、写 $d$；融合后的“写入 + 混合 + 相加”读 $(n+1)d$、写 $nd$；合计 $(4n+2)d=18d$。再加上子层自己 pre-norm 的一读一写 $2d$，正好是 V4.1 论文给 V4 算的 $(4n+4)d=20d$，两边的账对得上。前三列的访存不含 pre-norm，V4 与 V4.1 两列含。

把本课的因果链再串一遍：残差连接的“1”保证了任意深度的直通 ([§1](#p1))；HC 用可学的矩阵换掉“1”，获得了流间交换信息的能力 ([§2](#p2))，也把 He 等人十年前指出的连乘问题带了回来，并在 27B 上变成了近 3000 倍的增益 ([§3](#p3))；mHC 把矩阵限制在双随机矩阵上，恢复了“1”的守恒性质而保留了混合能力 ([§4](#p4))，用 20 轮 Sinkhorn–Knopp 实现这个约束 ([§5](#p5))，再用融合、重计算与流水线改造把代价压到 6.7% ([§6](#p6))；V4 把它用进万亿参数的模型，V4.1 又把推理访存减半。

## 仍然开放的问题

- **多样性的耗尽。** (16.21) 说明双随机混合只会让各流越来越像，图 16.13 中 mHC 的 60 层之积已接近 $J_4$。流的条数是 4，但深处实际的“有效流数”可能少得多。2026 年 9 月的 oHC 预印本（本课只读了摘要）主张改用正交矩阵，保范数而不搅匀 [@guo2026ohc]；它的效果本课无法评价。
- **截断误差。** 20 轮 SK 的列和误差，在 logit 很尖锐时可以到百分之几 ([§5](#p5))；实测复合反向增益 1.6 是否还有更好的办法（更多轮、或以列归一化收尾交替进行），论文没有讨论。
- **初始化。** 静态偏置 $S$ 的初值没有公开，mHC 的训练是否从 $H^{\rm res}\approx I$ 出发不得而知。
- **规模。** 论文的对照实验最大到 27B；本课的研究笔记没有记录在 V4 规模上去掉 mHC 的消融。

::: key 本页要点
- 27B：复合反向增益从 HC 的约 3000（数字化 2825）降到 mHC 的约 1.6（数字化 1.65），约 1700 倍；最终 loss 比基线低 0.021；8 个基准全部高于基线，7 个高于 HC（MATH 例外）。
- 学到的矩阵：HC 的 60 层之积几乎秩 1、放大约 500 倍；mHC 单层多近似 $I$，60 层之积接近 $J_4$（最大偏差 0.04）。
- V4：$X_{l+1}=B_lX_l+C_l\mathcal F_l(A_lX_l)$ (16.31)，$n=4$、$t_{\max}=20$，每个子层一套，静态偏置与门控用 AdamW；mHC 参数只占 0.012%（Flash），开销为 1F1B 流水段的 6.7%。
- V4.1-Flash：输入混合用上一层的 $A_{l-1}$ (16.32)，一次遍历完成，访存从 $(4n+4)d=20d$ 降到下界 $(2n+2)d=10d$；质量损失据称可忽略。
:::

# 练习与答案 | 12 题：概念、计算、推导 | 50

前 4 题是概念题，第 5–9 题是计算题（需要计算器），第 10–12 题是推导题。答案都写了完整过程。

:::: quiz 练习 1 · 为什么只约束流间混合（概念）
mHC 把 $H^{\rm res}$ 约束成双随机矩阵，却只要求 $H^{\rm pre}$、$H^{\rm post}$ 非负。为什么不把三个映射都约束成“守恒”的？
::: answer 看答案
看展开式 (16.13)：跨层**连乘**的只有 $H^{\rm res}$。直通项是 $H^{\rm res}_{L-1}\cdots H^{\rm res}_l$，每多一层就多乘一个；而 $H^{\rm pre}_i$、$H^{\rm post}_i$ 在每一条路径上只出现一次（第 $i$ 个子层的输出被 $H^{\rm post}_i$ 写入一次，然后只被它之后的 $H^{\rm res}$ 混合）。所以指数律只威胁 $H^{\rm res}$。

$H^{\rm pre}$、$H^{\rm post}$ 只需要防止另一种坏事：正负系数互相抵消 [@xie2026mhc]，非负就够了。$H^{\rm pre}$ 的整体大小甚至无关紧要，因为子层 $\mathcal F_l$ 一进来先做自己的 RMSNorm，$\RMSNorm(c\,\mathbf x)=\RMSNorm(\mathbf x)$（$c>0$，忽略 $\epsilon$）。$H^{\rm post}$ 用 $2\sigma$，是为了让 logit 为 0 时写入权重恰好是 1，和普通残差对齐。
:::
::::

:::: quiz 练习 2 · mHC 能不能精确回到恒等映射（概念）
(a) $I$ 在 $\mathcal B_4$ 的什么位置？$\mathcal B_4$ 有几维、几个顶点？(b) mHC 的 $H^{\rm res}$ 能否精确等于 $I$？(c) 如果不能，“mHC 恢复了恒等映射”这句话在什么意义下成立？
::: answer 看答案
(a) $I$ 是置换矩阵，所以是 $\mathcal B_4$ 的一个顶点（Birkhoff–von Neumann 定理）。维数 $(4-1)^2=9$，顶点数 $4!=24$。

(b) 不能。$M^{(0)}=\exp(\tilde H^{\rm res})$ 的每个元素都严格为正，行列缩放不会把正数变成 0，所以 SK 的每一轮输出（以及它的极限）元素都严格为正：极限落在多面体的相对内部，截断后的输出则连列和都只近似为 1。两者都只能任意接近顶点 $I$（需要对角 logit 远大于非对角 logit），永远到不了。V4 的代码还在分母上加了 $10^{-6}$。

(c) 在“性质”的意义下成立：$n=1$ 时双随机约束退回标量 1；对任意 $n$，双随机矩阵与 $I$ 一样满足均值守恒、谱范数为 1、任意多个相乘仍在同一集合里。mHC 恢复的是恒等映射的这些**性质**，而不是 $H^{\rm res}=I$ 本身。
:::
::::

:::: quiz 练习 3 · 行随机够不够，截断在哪一步（概念）
(a) 如果只把 $H^{\rm res}$ 约束为行随机矩阵（非负、行和为 1），前向增益与反向增益各会怎样？(b) mHC 的 SK 截断在第 20 轮的行归一化之后，实际得到的矩阵更像行随机还是列随机？这对应图 16.12 中的哪个现象？
::: answer 看答案
(a) 行和在乘法下封闭（$P\mathbf 1=\mathbf 1$），所以前向增益恒为 1；列和不受约束，反向增益最多到 $n$（练习 11），$\begin{pmatrix}1&0\\1&0\end{pmatrix}$ 就能取到 2。图 16.6 的模拟里，行随机矩阵 60 层后的反向增益中位数是 1.77。

(b) 以行归一化收尾，行和精确为 1、列和近似为 1，所以实际矩阵是“行随机、近似列随机”。对应图 16.12：前向增益几乎恰为 1（单层 1.0000–1.0005），反向增益略大于 1（单层 1.006–1.061，复合最大约 1.65），而且受行随机的上界 4 约束。
:::
::::

:::: quiz 练习 4 · 0.33% 的算术，为什么花 6.7% 的时间（概念）
V4-Flash 的 mHC 只占每 token 前向算术量的约 0.33%，mHC 论文却报告 $n=4$ 时额外训练时间 6.7%，还专门写了融合核。解释这个差距，并说明 V4.1 的 Single-Pass 针对的是账单的哪一行。
::: answer 看答案
mHC 的操作几乎都是“读一遍 $n$ 倍宽的残差流、做几次乘加、写回去”，每搬一个数只做一两次运算，属于受带宽限制的操作（第 4 课的带宽墙）：耗时由搬数据的量决定，而不是由 FLOPs 决定。未融合时 HC 每个子层读写 $34d+48$ 个元素，是普通残差 $3d$ 的约 11 倍 (16.28)；此外还有 $n$ 倍宽的激活要存、流水线 stage 之间 $n$ 倍的通信。6.7% 是在融合（读 $13d\to5d$、写 $12d\to4d$）、重计算与 DualPipe 改造之后剩下的开销。

V4.1 的 Single-Pass 针对的仍是“访存”这一行：多趟实现要把残差流读三遍（$(4n+4)d$），推迟一格使用 $A_{l-1}$ 之后只需读写各一遍（$(2n+2)d$），访存减半。
:::
::::

:::: quiz 练习 5 · 每层多少才会到 3000（计算）
(a) 60 个子层的复合增益是 3000，若每层相同，单层增益是多少？(b) 单层增益为 1.05、0.95 时，60 层之后各是多少？(c) 图 16.5 (a) 中 HC 的单层前向增益大多在 0.64 到 2.81 之间，为什么复合增益不能简单地用“单层增益相乘”来估计？
::: answer 看答案
(a) $3000^{1/60}=e^{\ln3000/60}=e^{8.0064/60}=e^{0.13344}=1.1428$。

(b) $1.05^{60}=e^{60\times0.04879}=e^{2.9274}=18.68$；$0.95^{60}=e^{60\times(-0.05129)}=e^{-3.0776}=0.0461$。

(c) 矩阵乘积的行和一般不等于各因子行和的乘积；能保证相等的一种情形是每个因子自己的各行之和都相等（$H_k\mathbf 1=r_k\mathbf 1$，这时 $P\mathbf 1=H_L\cdots H_2(r_1\mathbf 1)=\cdots=\big(\prod_kr_k\big)\mathbf 1$；例如 $r_k$ 都等于 1）。对元素非负的矩阵，“单层增益相乘”至少是一个上界（这时最大行和、最大列和就是矩阵的 ∞-范数、1-范数，满足 $\norm{AB}\le\norm{A}\norm{B}$）；对有正有负的矩阵，它连上界都不是。Amax 增益只看“所有流装同一个信号”这一种输入；一层把信号推到某个方向之后，下一层面对的已经不是“各流相同”的输入，它对这个新方向的放大倍数可以远大于它的 Amax 值。[§3](#p3) 的 $2\times2$ 算例是一个极端的版本：$A$、$B$ 两种矩阵的 Amax 增益都恰为 1，交替乘 60 层之后前向增益却是 24.8，因为 $B$ 不断把公共信号漏到“差异”方向，而 $A$ 把差异方向放大 1.2 倍。随机模拟 ([§3](#p3)) 也说明，$n=4$ 时尺度为 $s$ 的零均值扰动乘多了照样会按约 $e^{Ls^2}$ 增长。
:::
::::

:::: quiz 练习 6 · 从论文图 8 读增益（计算）
HC 第 60 个子层的 $H^{\rm res}_{60}$ 在论文图 8 中印为
$$\begin{pmatrix}-5.58&-3.74&-5.71&-6.60\\-6.06&-2.27&-5.33&-6.57\\6.08&3.12&6.53&6.77\\-6.41&-3.97&-5.72&-5.49\end{pmatrix}.$$
求它的前向增益与反向增益（Amax 定义）。若误用矩阵的 ∞-范数（每行绝对值之和的最大者），前向会得到多少？
::: answer 看答案
行和：第 1 行 $-5.58-3.74-5.71-6.60=-21.63$，第 2 行 $-6.06-2.27-5.33-6.57=-20.23$，第 3 行 $6.08+3.12+6.53+6.77=22.50$，第 4 行 $-6.41-3.97-5.72-5.49=-21.59$。前向增益 $=\max|\text{行和}|=22.50$（图上标签也是 22.50；前两行的标签 −21.64、−20.22 与此处差 0.01，是转录两位小数的舍入）。

列和：$-5.58-6.06+6.08-6.41=-11.97$，$-3.74-2.27+3.12-3.97=-6.86$，$-5.71-5.33+6.53-5.72=-10.23$，$-6.60-6.57+6.77-5.49=-11.89$。反向增益 $=11.97$。

∞-范数：每行绝对值之和，第 1 行 $21.63$，第 2 行 $20.23$，第 3 行 $22.50$，第 4 行 $21.59$，最大 22.50。这一行的元素同号，所以两种定义碰巧相同；反向（列）就不同了：第 1 列绝对值之和 $5.58+6.06+6.08+6.41=24.13$，而列和的绝对值只有 11.97。
:::
::::

:::: quiz 练习 7 · 手算 Sinkhorn（计算）
对 $A=\begin{pmatrix}2&1\\1&1\end{pmatrix}$ 做一轮 mHC 顺序的 SK（先列后行），写出 $M^{(1)}$ 与它的列和。极限的交叉比 $\tfrac{m_{11}m_{22}}{m_{12}m_{21}}$ 应该是多少？由此解出极限矩阵。最后，用 Knight 定理（[§5](#p5)）预测第 2 轮的列和误差，再直接算出 $M^{(2)}$ 对照。
::: answer 看答案
**第 1 轮。** $A$ 的列和是 3 和 2，先列归一化，再按行和 $\tfrac23+\tfrac12=\tfrac76$、$\tfrac13+\tfrac12=\tfrac56$ 行归一化：

$$\mathcal T_c(A)=\begin{pmatrix}2/3&1/2\\1/3&1/2\end{pmatrix},\qquad M^{(1)}=\begin{pmatrix}4/7&3/7\\2/5&3/5\end{pmatrix}.$$

$M^{(1)}$ 的列和：$\tfrac47+\tfrac25=\tfrac{34}{35}\approx0.9714$，$\tfrac37+\tfrac35=\tfrac{36}{35}\approx1.0286$，误差 $1/35\approx0.029$。

**极限。** SK 不改变交叉比，所以极限的交叉比等于 $A$ 的 $\tfrac{2\times1}{1\times1}=2$。$2\times2$ 双随机矩阵必为 $\begin{pmatrix}a&1-a\\1-a&a\end{pmatrix}$，于是

$$\frac{a^2}{(1-a)^2}=2\ \Rightarrow\ \frac{a}{1-a}=\sqrt2\ \Rightarrow\ a=\frac{\sqrt2}{1+\sqrt2}=2-\sqrt2\approx0.5858.$$

所以极限 $P$ 的对角元素是 0.5858，非对角元素是 0.4142。$M^{(1)}$ 的第一个元素 $4/7=0.5714$ 已经离它不远。

**预测第 2 轮。** 极限 $P$ 对称，特征值是 1（向量 $(1,1)$）和 $2a-1=3-2\sqrt2\approx0.1716$（向量 $(1,-1)$），所以 $\sigma_2=0.1716$，$\sigma_2^2=(3-2\sqrt2)^2=17-12\sqrt2\approx0.02944$。预测第 2 轮误差 $\approx0.02944\times\tfrac1{35}=8.41\times10^{-4}$。

**直接算第 2 轮。** 按列和 $\tfrac{34}{35},\tfrac{36}{35}$ 列归一化，再按行和 $\tfrac{205}{204},\tfrac{203}{204}$ 行归一化：

$$\mathcal T_c(M^{(1)})=\begin{pmatrix}10/17&5/12\\7/17&7/12\end{pmatrix},\qquad M^{(2)}=\begin{pmatrix}24/41&17/41\\12/29&17/29\end{pmatrix}.$$

列和 $\tfrac{1188}{1189},\tfrac{1190}{1189}$，误差 $\tfrac1{1189}=8.41\times10^{-4}$。两轮误差之比 $\tfrac{35}{1189}=0.02944$，与 $\sigma_2^2$ 一致到四位小数。
:::
::::

:::: quiz 练习 8 · V4 系列的 mHC 账单（计算）
(a) V4-Pro（$d=7168$，61 个块）每个子层的 mHC 参数是多少？全模型多少？占 1.6T 的百分之几？(b) 按 bf16，V4-Pro 的一个子层，普通残差与未融合 HC 每个 token 各读写多少字节？
::: answer 看答案
(a) 每子层 $nd(n^2+2n)+(n^2+2n)+3=4\times7168\times24+24+3=688{,}128+27=688{,}155$。子层数 $2\times61=122$，全模型 $688{,}155\times122=83{,}954{,}910\approx8.40\times10^7$，占 $8.40\times10^7/1.6\times10^{12}=0.0052\%$。

(b) 普通残差 $3d=21{,}504$ 个元素，$\times2$ 字节 $=43{,}008$ 字节。未融合 HC $34d+48=243{,}760$ 个元素，$487{,}520$ 字节，约为前者的 11.3 倍。
:::
::::

:::: quiz 练习 9 · 重计算块长（计算）
用 (16.30)，$n=4$。(a) $L=122$（V4-Pro 的子层数）时，连续近似的最优块长是多少？在 $L_r=8,9,10$ 中哪个最好，激活显存是多少 $d$？(b) 与“每层都存 $X_l$”相比省了多少倍？
::: answer 看答案
(a) $L_r^*=\sqrt{4\times122/6}=\sqrt{81.33}=9.02$。逐个算 $f=4\lceil122/L_r\rceil+6L_r$：$L_r=8$：$4\times16+48=112$；$L_r=9$：$4\times14+54=110$；$L_r=10$：$4\times13+60=112$。$L_r=9$ 最好，$110d$。

(b) 每层都存 $X_l$ 要 $4\times122=488d$，$488/110\approx4.4$ 倍。若三种中间量都存，是 $6\times122=732d$，约 6.7 倍。
:::
::::

:::: quiz 练习 10 · 双随机矩阵的反向也守恒（推导）
设 $A\in\mathcal B_n$。(a) 证明 $A^\top\in\mathcal B_n$。(b) 残差支路上 $X_{l+1}=AX_l$，反向 $G_l=A^\top G_{l+1}$。证明各流梯度之和不变：$\mathbf 1_n^\top G_l=\mathbf 1_n^\top G_{l+1}$，且 $\norm{G_l}_F\le\norm{G_{l+1}}_F$。
::: answer 看答案
(a) $A^\top$ 的元素就是 $A$ 的元素，非负。$A^\top\mathbf 1_n=(\mathbf 1_n^\top A)^\top=(\mathbf 1_n^\top)^\top=\mathbf 1_n$：$A^\top$ 的行和是 $A$ 的列和，等于 1。$\mathbf 1_n^\top A^\top=(A\mathbf 1_n)^\top=\mathbf 1_n^\top$：$A^\top$ 的列和是 $A$ 的行和，等于 1。所以 $A^\top\in\mathcal B_n$。

(b) $\mathbf 1_n^\top G_l=\mathbf 1_n^\top A^\top G_{l+1}=(A\mathbf 1_n)^\top G_{l+1}=\mathbf 1_n^\top G_{l+1}$，用了 $A$ 的行和为 1。范数：由 (a)，$A^\top$ 也是双随机矩阵，按 (16.18) 的证明 $\norm{A^\top}_2=1$，对 $G_{l+1}$ 的每一列用一次，得 $\norm{A^\top G_{l+1}}_F\le\norm{G_{l+1}}_F$。所以反向也是“总量守恒、范数不增”，连乘任意多层仍然成立（乘法封闭）。
:::
::::

:::: quiz 练习 11 · 行随机矩阵之积的反向上界（推导）
设 $H_1,\dots,H_L$ 都是非负且行和为 1 的 $n\times n$ 矩阵，$P=H_L\cdots H_1$。证明 $P$ 的每一列之和都不超过 $n$，并构造一个例子说明 $n$ 可以取到。
::: answer 看答案
先证 $P$ 非负、行和为 1：非负矩阵之积非负；$P\mathbf 1=H_L\cdots H_2(H_1\mathbf 1)=H_L\cdots H_2\mathbf 1=\cdots=\mathbf 1$。于是全部元素之和 $\mathbf 1^\top P\mathbf 1=\mathbf 1^\top\mathbf 1=n$。任一列之和是一部分非负元素之和，不超过全部元素之和，所以 $\le n$。

取到上界：令每个 $H_k=\mathbf 1_n\mathbf e_1^\top$（每一行都是 $(1,0,\dots,0)$）。它非负、行和为 1；$H_k^2=\mathbf 1\mathbf e_1^\top\mathbf 1\mathbf e_1^\top=\mathbf 1(\mathbf e_1^\top\mathbf 1)\mathbf e_1^\top=\mathbf 1\mathbf e_1^\top$，所以 $P=\mathbf 1\mathbf e_1^\top$，第 1 列之和为 $n$。含义：所有新流都只从旧流 1 取货，反向时旧流 1 收到 $n$ 份梯度。$n=2$ 时就是正文的反例 $\begin{pmatrix}1&0\\1&0\end{pmatrix}$。
:::
::::

:::: quiz 练习 12 · 最优块长与 Sinkhorn 的慢收敛（推导）
(a) 从 $f(L_r)=ndL/L_r+(n+2)dL_r$ 推出 $L_r^*$，并说明它是最小值。(b) 对 $M^{(0)}=\begin{pmatrix}1&1\\0&1\end{pmatrix}$，证明 SK（先列后行）第 $t$ 轮后右上角元素为 $1/(2t+1)$，并求列和误差降到 0.01 以下需要多少轮。
::: answer 看答案
(a) $f'(L_r)=-ndL/L_r^2+(n+2)d$，令其为 0 得 $L_r^2=nL/(n+2)$，$L_r^*=\sqrt{nL/(n+2)}$。$f''(L_r)=2ndL/L_r^3>0$，所以是最小值。直观上，第一项（块首输入）随块长变短而增加，第二项（块内中间量）随块长变长而增加，两者相等处总和最小：$ndL/L_r=(n+2)dL_r$。

(b) 分四步。

1. **形状不变。** 设第 $t$ 轮后 $M^{(t)}=\begin{pmatrix}1-b_t&b_t\\0&1\end{pmatrix}$：行和已为 1，左下角的 0 在行列缩放下永远是 0。
2. **一轮的递推。** 列归一化（第 1 列和 $1-b_t$，第 2 列和 $1+b_t$），再行归一化（第 1 行和 $\frac{1+2b_t}{1+b_t}$，第 2 行变回 $(0,1)$）：

$$\mathcal T_c(M^{(t)})=\begin{pmatrix}1&\dfrac{b_t}{1+b_t}\\0&\dfrac1{1+b_t}\end{pmatrix}\ \Rightarrow\ b_{t+1}=\frac{b_t}{1+2b_t}\ \Rightarrow\ \frac1{b_{t+1}}=\frac1{b_t}+2.$$

3. **首项。** 第 1 轮直接算：列和 1、2，列归一化得 $\begin{pmatrix}1&1/2\\0&1/2\end{pmatrix}$，行归一化得 $\begin{pmatrix}2/3&1/3\\0&1\end{pmatrix}$，所以 $b_1=\frac13$，$\frac1{b_t}=3+2(t-1)=2t+1$。
4. **误差与轮数。** 列和为 $1-b_t$ 与 $1+b_t$，误差 $b_t=\frac1{2t+1}$。要 $\frac1{2t+1}<0.01$，需 $2t+1>100$，即 $t\ge50$ 轮。

对照 [§5](#p5) 的 3×3 算例，2 轮就到 $6.5\times10^{-4}$：这就是 $O(1/t)$ 与几何收敛的差别。
:::
::::

```refs
- id: zhu2025hc
  title: Hyper-Connections
  a: Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou (2025). ICLR 2025. arXiv:2409.19606（读的是 v3；附录 A、B 自 v2 起才有）
  url: https://arxiv.org/abs/2409.19606
  finding: 把残差流复制成 n 条，用可学习（静态或输入相关）的 (n+1)×(n+1) 矩阵同时决定子层的读出、写回与流间混合；n = 4 时在 OLMo/OLMoE 上稳定优于 Pre-Norm 残差，参数与 FLOPs 增量可忽略，显存增加 10%–28%。
  matters: 它是 mHC 要约束的对象：本课的三映射记号、初始化等价于 Pre-Norm、Pre/Post-Norm 作为特例，以及“无约束残差矩阵”都出自这里。
- id: xie2026mhc
  title: "mHC: Manifold-Constrained Hyper-Connections"
  a: Zhenda Xie, Yixuan Wei, Huanqi Cao, et al.（DeepSeek-AI，v2 共 20 位作者）(2025). 预印本. arXiv:2512.24880（读的是 v2；v1 为 2025-12-31，v2 与 v1 只差作者名单）
  url: https://arxiv.org/abs/2512.24880
  finding: 把 HC 的残差映射用 Sinkhorn–Knopp（20 次）投影到双随机矩阵，读出与写入用 sigmoid 非负化，使 27B MoE 上复合增益从近 3000 降到约 1.6，并以核融合、重计算与 DualPipe 改造把 n = 4 的额外训练时间压到 6.7%。
  matters: 本课主体：公式、稳定性分析、基础设施账和 3B/9B/27B 实验都以它为准，也是 DeepSeek-V4 采用 mHC 的直接依据。
- id: dsv4
  title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
  a: DeepSeek-AI (2026). 预印本. arXiv:2606.19348（读的是 v1）
  url: https://arxiv.org/abs/2606.19348
  finding: V4-Pro（1.6T/49B）与 V4-Flash（284B/13B）的每个子层都使用 mHC（$n_{hc}=4$，$t_{\max}=20$），mHC 的静态偏置与门控因子用 AdamW 训练，墙钟开销为重叠后 1F1B 流水段的 6.7%。
  matters: 说明 mHC 进入了生产级模型，并提供 A/B/C 记号、确定性（输出维度只有 24 的小矩阵乘）等工程细节。
- id: dsv41
  title: "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"
  a: DeepSeek-AI (2026). 预印本. arXiv:2609.19969（读的是 v1，2026-09-17）
  url: https://arxiv.org/abs/2609.19969
  finding: 提出 Single-Pass mHC：输入混合改用上一个子层预测的系数，使部署时残差流只需读写各一次，激活访存从 $(4n+4)d$ 降到下界 $(2n+2)d$，据称质量几乎无损。
  matters: 把本课的访存账延伸到时间线终点，说明 mHC 的瓶颈在带宽而不在算力，并为第 19 课铺路。
- id: he2016identity
  title: Identity Mappings in Deep Residual Networks
  a: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016). ECCV 2016. arXiv:1603.05027（读的是 v3）
  url: https://arxiv.org/abs/1603.05027
  finding: 恒等捷径使任意两层之间存在不经权重层的前向与反向直通项；若捷径乘以标量 λ，其连乘会随深度指数放大或消失，并造成优化困难。
  matters: mHC 的复合映射正是其标量连乘的矩阵版，本课用它说明“为什么复合映射必须守恒”。
- id: xiong2020preln
  title: On Layer Normalization in the Transformer Architecture
  a: Ruibin Xiong et al. (2020). ICML 2020. arXiv:2002.04745（读的是 v2）
  url: https://arxiv.org/abs/2002.04745
  finding: 初始化时 Post-LN 最后一层 FFN 的梯度量级为 $O(d\sqrt{\ln d})$、与层数无关，而 Pre-LN 为 $O(d\sqrt{\ln d/L})$，因为 Pre-LN 最终 LayerNorm 的输入规模随 L 线性增长。
  matters: 给“Pre-LN 与 Post-LN 的两难”提供有前提的定理，也替代 mHC 对 Vaswani 等人的误引。
- id: liu2020admin
  title: Understanding the Difficulty of Training Transformers
  a: Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, Jiawei Han (2020). EMNLP 2020. arXiv:2004.08249（读的是 v3）
  url: https://arxiv.org/abs/2004.08249
  finding: 15 组设置中 Pre-LN 全部收敛而 Post-LN 有 7 组发散，但 Post-LN 收敛时 8 组中有 7 组优于 Pre-LN；Post-LN 更依赖残差分支。
  matters: 用实测数字支撑“跷跷板”叙事，同时提醒 HC 对它的“表示坍缩”引用并不贴切。
- id: vaswani2017
  title: Attention Is All You Need
  a: Ashish Vaswani et al. (2017). NeurIPS 2017. arXiv:1706.03762（读的是 v7）
  url: https://arxiv.org/abs/1706.03762
  finding: 原版 Transformer 的每个子层输出为 LayerNorm(x + Sublayer(x))，即 Post-LN。
  matters: 用来确认 Post-LN 的出处，并纠正 mHC 把 pre-norm Transformer 归到本文。
- id: iclr2025-hc-page
  kind: artefact
  title: Hyper-Connections — ICLR 2025 论文集摘要页
  a: ICLR (2025). proceedings.iclr.cc
  url: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f1e8ff97057e0c3f6bcb7018c78d4df1-Abstract-Conference.html
  finding: 记载 HC 为 ICLR 2025 会议论文，8 位作者与 arXiv 一致。
  matters: 确认 HC 的发表状态（本课写 ICLR 2025 而不是“预印本”）。
- id: birkhoff1946
  title: Tres observaciones sobre el algebra lineal
  a: Garrett Birkhoff (1946). Univ. Nac. Tucumán Revista Ser. A 5:147–151（原文未见；卷期页码只来自二手索引）
  url: https://en.wikipedia.org/wiki/Birkhoff_polytope
  finding: 双随机矩阵集合是置换矩阵的凸包，其顶点恰为 n! 个置换矩阵（Birkhoff–von Neumann 定理）。
  matters: 支撑“残差映射 = 置换的凸组合”的几何解释，以及本课的 Birkhoff 多面体图与维数、顶点数的计算。
- id: guo2026ohc
  title: "oHC: Orthogonal Hyper-Connections on SO(4) via Quaternions"
  a: Haoqiang Guo et al. (2026). 预印本（未经同行评审）. arXiv:2609.02672（只读了摘要）
  url: https://arxiv.org/abs/2609.02672
  finding: 摘要称双随机约束只能靠缩小流间差异来降低范数、使各流随深度趋同，因而改把残差矩阵限制在旋转群 SO(n) 上，n = 4 时用一对单位四元数闭式参数化。
  matters: 作为开放问题的线索：mHC 的守恒以耗尽流间多样性为代价，这与本课 (16.21) 和论文图 8 的观察一致；其定理与实验本课未核。
- id: hf-v4flash-code
  kind: artefact
  title: DeepSeek-V4-Flash 推理代码（inference/model.py、inference/kernel.py）与 config.json
  a: DeepSeek-AI (2026). Hugging Face 仓库 deepseek-ai/DeepSeek-V4-Flash（2026-09-24 读取）
  url: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
  finding: 每个块有注意力与 FFN 两套 mHC 参数（24×4d 的投影、24 个静态偏置、3 个门控标量），Sinkhorn 以逐行 softmax 起步、共 20 轮，并在读出系数与各个分母中加 1e-6；config 为 hc_mult = 4、hc_sinkhorn_iters = 20。
  matters: 用来核对论文与实现是否一致（迭代顺序、系数布局、输出头的合流方式），并补上论文没有写的 eps 细节。
- id: sinkhorn1964
  title: A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices
  a: Richard Sinkhorn (1964). Annals of Mathematical Statistics 35(2):876–879. doi:10.1214/aoms/1177703591（正文未读，据题名与 Mena 等人 2018 年论文附录 A 的转述）
  url: https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-2/A-Relationship-Between-Arbitrary-Positive-Matrices-and-Doubly-Stochastic-Matrices/10.1214/aoms/1177703591.full
  finding: 任意严格正的方阵都可以左右各乘一个正对角矩阵变成双随机矩阵，且结果唯一。
  matters: mHC 的 $M^{(0)}=\exp(\tilde H)$ 严格为正，真正用到的正是这一篇覆盖的情形。
- id: sinkhorn1967
  title: Concerning nonnegative matrices and doubly stochastic matrices
  a: Richard Sinkhorn, Paul Knopp (1967). Pacific Journal of Mathematics 21(2):343–348. doi:10.2140/pjm.1967.21.343
  url: https://msp.org/pjm/1967/21-2/p14.xhtml
  finding: 非负矩阵的交替行列归一化收敛到双随机极限，当且仅当矩阵至少含一条正对角线；极限可写成 $D_1AD_2$ 当且仅当 $A\ne0$ 且每个正元素都在某条正对角线上。
  matters: mHC 所用算法的命名出处；也说明了本课慢收敛反例 $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ 为什么仍收敛、极限却写不成 $D_1AD_2$（它的 $O(1/t)$ 速度是本课自己算的）。
- id: mena2018gumbel
  title: Learning Latent Permutations with Gumbel-Sinkhorn Networks
  a: Gonzalo Mena, David Belanger, Scott Linderman, Jasper Snoek (2018). ICLR 2018. arXiv:1802.08665（读的是 v1）
  url: https://arxiv.org/abs/1802.08665
  finding: 定理 1 证明 Sinkhorn 算子 $S(X/\tau)$ 等于 Birkhoff 多面体上 $\langle P,X\rangle_F+\tau h(P)$ 的最大值点，$\tau\to0^+$ 时（元素取自连续分布，几乎必然）趋于线性指派的置换矩阵。
  matters: 给 mHC 一句话带过的“熵投影”一个精确定义，并解释 logit 越尖锐 SK 越接近置换、也越难收敛。
- id: knight2008
  title: "The Sinkhorn-Knopp Algorithm: Convergence and Applications"
  a: Philip A. Knight (2008). SIAM Journal on Matrix Analysis and Applications 30(1):261–275. doi:10.1137/060659624（读的是作者存于 Strathprints 的预印本）
  url: https://strathprints.strath.ac.uk/19685/
  finding: 定理 4.5：对完全不可分解的非负矩阵，SK 线性收敛，从某一轮起缩放向量的误差每轮至多乘双随机极限 $P$ 的第二大奇异值的平方 $\sigma_2(P)^2$；§6 指出渐近速率就等于它，数值实验中很快贴近。
  matters: 回答“SK 收敛快慢由什么决定”：本课用它解释 3×3 算例的每轮 0.0436、近退化矩阵的慢收敛，以及对角 logit 越大、20 轮越不够的张力。
- id: tilelang2025
  title: "TileLang: A Composable Tiled Programming Model for AI Systems"
  a: Lei Wang et al. (2025). 预印本. arXiv:2504.17577（只核对了题名与作者）
  url: https://arxiv.org/abs/2504.17577
  finding: 一种按分块（tile）组织 GPU 核的编程模型。
  matters: mHC 的大部分融合核与 V4 推理代码里的 Sinkhorn 核都用它编写，是基础设施一页的工具出处。
- id: dsv3
  title: DeepSeek-V3 Technical Report
  a: DeepSeek-AI (2024). 预印本. arXiv:2412.19437（本课只核对了 §3.2.1 关于 DualPipe 重叠的几句，细节见第 12 课）
  url: https://arxiv.org/abs/2412.19437
  finding: 提出 DualPipe 流水线调度，让前向、反向计算与专家并行的 all-to-all 通信、流水线收发相互重叠。
  matters: mHC 在它之上加了高优先级计算流等改造；mHC 的实验模型也按 V3 的 MoE 架构搭建。
- id: hf-v41flash-code
  kind: artefact
  title: DeepSeek-V4.1-Flash 推理代码（inference/model.py、inference/kernel.py）与 config.json
  a: DeepSeek-AI (2026). Hugging Face 仓库 deepseek-ai/DeepSeek-V4.1-Flash（2026-09-24 读取）
  url: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
  finding: 每个子层使用上一个子层算出的读出系数（第一个子层用只取第 0 条流的 one-hot），与 V4.1 论文的 Single-Pass 公式一致；config 为 hc_mult = 4、hc_sinkhorn_iters = 20、hidden_size = 5120、40 层。
  matters: 证实 Single-Pass mHC 已进入发布的推理代码，而不只是论文里的描述。
```
