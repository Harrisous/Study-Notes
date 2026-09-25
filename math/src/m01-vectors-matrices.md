---
title: M01 · 向量、矩阵与形状：注意力里的线性代数
short: 矩阵与形状思维
kicker: 数学补课 · M01
description: 点积与余弦相似度、矩阵乘法的三种读法、转置与 einsum 的形状思维、分块矩阵与多头注意力、矩阵乘法 FLOPs = 2mnk、子空间投影与结合律（MLA 矩阵吸收的数学基础），配大量可手算的例题与练习。
lede: 注意力层里的计算几乎全是矩阵乘法。这一课只讲线性代数里真正用得上的那一部分，但从头讲透：点积为什么能当“相关度”，同一个矩阵乘法可以换几种角度读，怎样只看**形状**就查出公式有没有写错，一次乘法要花多少算力，以及为什么换一个括号的位置，就能让 MLA 不必把缓存展开。
meta: 预计阅读 135 分钟，练习另需约 45 分钟 · 前置：高中向量；会一点 NumPy 更好 · 服务于第 1、4、9 课
index: index.html
next: m02-rank-svd-lowrank.html | M02 · 低秩与 SVD
---

## 这一课回答什么问题

第 1 课会写下这样一行公式：$\Attn(Q,K,V)=\softmax\!\left(QK^\top/\sqrt{d_k}\right)V$。线性代数生疏的读者看到它，往往只能认出一串字母。这一课的目标是让你看到 $QK^\top$ 时，脑子里立刻出现四件事：它是一个 $T\times T$ 的表；表里每一格是一个查询和一个键的点积；算出这张表要大约 $2T^2d_k$ 次浮点运算；它的每一行将决定一个 token 从各个 token（包括它自己）那里各取多少信息。

具体来说，读完这一课，下面这些问题你都能自己回答：

- 注意力为什么用**点积**给两个 token 打分，而不是别的“相似度”？向量的长度在打分里起什么作用？
- $QK^\top$ 为什么是 $T\times T$？论文写 $\mathbf q_t=W^Q\mathbf h_t$，代码写 `x @ W.T`，两者是一回事吗？
- 多头注意力的“切成多个头、各算各的、拼起来、再乘 $W^O$”，写成矩阵到底是什么？
- 一次矩阵乘法要多少次浮点运算？一层注意力的算力主要花在哪里？
- 为什么 DeepSeek-V2 的 MLA 只缓存一个短向量，推理时却可以不把它展开成键？

## 这一课在科技树上的位置

这是数学补课的第一课，直接服务于三课：[第 1 课](../deepseek/lesson-01-transformer-anatomy.html)（注意力的每一步都是本课的一种矩阵乘法；全流程形状追踪；“每 token 前向约 $2N$ 次运算”）、[第 4 课](../deepseek/lesson-04-inference-kv-cache.html)（KV cache 的大小与解码的算力、带宽账单）、[第 9 课](../deepseek/lesson-09-v2-mla.html)（MLA 的低秩压缩与“矩阵吸收”，后者靠的是矩阵乘法的结合律与转置规则）。下一课 [M02](m02-rank-svd-lowrank.html) 从本课的“外积之和”与“结合律”接着讲秩、SVD 与低秩近似。

## 读完你能……

1. 手算点积与余弦相似度，说出注意力为什么用点积打分，以及向量长度在其中的作用。
2. 用三种方式读同一个矩阵乘法（线性映射、批量点积、列的组合 / 外积之和），并在注意力公式里认出每一种。
3. 只看形状检查张量公式；在论文的列向量写法和代码的行向量写法之间互相翻译；写出注意力每一步的 einsum。
4. 把多头注意力写成分块矩阵，并解释“拼接后乘 $W^O$”等于“每个头分别乘一块 $W^O$ 再相加”。
5. 用 $2mnk$ 数出一层注意力的 FLOPs，算出投影与打分的交叉点；用结合律判断先乘哪一对更省，并解释 MLA 的矩阵吸收为什么成立、什么时候才划算。

::: note 本课的记号
- 向量默认是**列向量**，用粗体小写：$\mathbf x\in\R^{d}$ 表示一列 $d$ 个实数。矩阵用大写：$A\in\R^{m\times n}$ 表示 $m$ 行 $n$ 列。
- 形状用方括号写成 $[m,n]$，和 NumPy / PyTorch 的 `shape` 一致；$[T,d]$ 就是 $T$ 行 $d$ 列。
- 转置写 $A^\top$。把一段序列的向量按**行**堆成矩阵（每行一个 token）时，正文会明说。
- 符号沿用整门课的约定：$T$ 是序列长度，$d$ 是模型隐藏维度，$n_h$ 是头数，$d_h$ 是每个头的维度，$\mathbf h_t$ 是第 $t$ 个 token 进入注意力层时的隐藏状态。《Attention Is All You Need》和第 1 课公式里的 $d_k$（每个头的键向量维度）就是这里的 $d_h$。
:::

[[legend]]

[[toc]]

# 向量与点积 | 注意力的打分为什么是点积 | 20

先把问题摆出来。第 1 课会讲到，注意力层让每个 token 回头看它前面的所有 token（也包括它自己），并决定从每一个那里拿多少信息。“决定拿多少”需要先有一个数：对每一对位置 $(t,j)$，给出“第 $j$ 个 token 和第 $t$ 个 token 的当前需求有多相关”的分数。而每个 token 在模型里只是一列数。所以真正的问题是：**给你两列各 $d$ 个数，用什么办法算出一个数来表示它们有多“对得上”？**

这个办法要同时满足三个条件。第一，便宜：一层、一个头就要算 $T^2$ 次，$T$ 是几千时就是几百万次。第二，处处可导：模型靠梯度下降学习，打分函数不能有断点。第三，能批量算：所有 $(t,j)$ 的分数最好能一次性交给 GPU。点积同时满足这三条。这一页讲清楚点积是什么、它在几何上量的是什么，以及它和常见的“余弦相似度”差在哪里。

## 向量：一列有顺序的数

**含义。** 向量就是一列有顺序的数。$(3,4)$ 和 $(4,3)$ 是两个不同的向量，因为顺序不同。

**理解。** 在语言模型里，每个 token 在每一层都由一个向量表示，称为它的**隐藏状态**（hidden state）。可以把它想成一张有 $d$ 个栏目的“档案”，只是这些栏目不是人起的名字，而是训练中学出来的。GPT-2 small 的隐藏维度是 $d=768$ [@gpt2-config]，也就是说每个 token 在每一层都是一列 768 个数。向量也可以画成从原点出发的一支箭头：$d=2$ 或 $3$ 时能画在纸上，$d=768$ 时画不出来，但下面所有的代数关系照样成立。

**数学。** 记 $\mathbf x\in\R^{d}$：一个列向量，有 $d$ 个实数分量 $x_1,\dots,x_d$。它的转置 $\mathbf x^\top$ 是同样这些数排成的一行。向量的**长度**（又叫范数，norm）由勾股定理给出：

::: eq 向量的长度（范数） | (M1.1)
$$\norm{\mathbf x}=\sqrt{x_1^2+x_2^2+\cdots+x_d^2}$$
例：$\norm{(3,4)}=\sqrt{9+16}=5$；$\norm{(1,2,2)}=\sqrt{1+4+4}=3$。
:::

::: pre 求和记号 Σ | 会读 $\sum_{i=1}^{d}$ 的可以跳过
$\sum_{i=1}^{d} x_i y_i$ 读作“让 $i$ 从 1 走到 $d$，把每一个 $x_iy_i$ 加起来”，也就是 $x_1y_1+x_2y_2+\cdots+x_dy_d$。下标 $i$ 只是个计数器，换成 $k$ 或 $j$ 意思不变。本课里凡是看到同一个下标在一个乘积里出现两次、又被 $\sum$ 管着，就是“对应位置相乘再相加”，后面讲 einsum 时这个习惯会直接变成一条规则。
:::

## 点积：对应位置相乘，再全部加起来

**点积**（dot product，也叫内积 inner product）把两个同样长度的向量变成一个数：

::: eq 点积的定义 | (M1.2)
$$\mathbf a\cdot\mathbf b=\mathbf a^\top\mathbf b=\sum_{i=1}^{d}a_i b_i$$
两个向量必须一样长（都在 $\R^d$ 里）；结果是一个实数，不是向量。
:::

写成 $\mathbf a^\top\mathbf b$ 的原因下一页会讲：一行乘一列，正好是矩阵乘法最小的情形。

::: ex 手算两个点积
二维：$\mathbf a=(4,1)$，$\mathbf b=(2,3)$。$\mathbf a\cdot\mathbf b=4\times2+1\times3=11$。

三维：$\mathbf u=(1,2,3)$，$\mathbf w=(4,-5,6)$。$\mathbf u\cdot\mathbf w=4-10+18=12$。注意中间那一项是负的：两个向量在第二个分量上“意见相反”，这一项把总分往下拉。

数一数运算次数：长度为 $d$ 的点积要做 $d$ 次乘法、$d-1$ 次加法，一共约 $2d$ 次。这个 $2$ 在 [§5](#p5) 会变成矩阵乘法 FLOPs 公式里的那个 $2$。
:::

点积有三条性质，下面的推导每一步都会用到。**对称**：$\mathbf a\cdot\mathbf b=\mathbf b\cdot\mathbf a$，因为 $a_ib_i=b_ia_i$。**对每个参数线性**：$(\mathbf a+\mathbf c)\cdot\mathbf b=\mathbf a\cdot\mathbf b+\mathbf c\cdot\mathbf b$，$(\lambda\mathbf a)\cdot\mathbf b=\lambda(\mathbf a\cdot\mathbf b)$，因为求和可以拆开、常数可以提出。**自己和自己的点积是长度的平方**：$\mathbf a\cdot\mathbf a=\sum_i a_i^2=\norm{\mathbf a}^2$，对照 (M1.1)。

## 几何意义：长度 × 长度 × 夹角的余弦

定义 (M1.2) 是纯代数的：乘、加。它凭什么能量“两个向量对得上的程度”？答案是一个几何恒等式：点积等于两个长度相乘，再乘上夹角 $\theta$ 的余弦。推导只需要余弦定理。

::: pre 余弦定理 | 记得 $c^2=a^2+b^2-2ab\cos C$ 的可以跳过
三角形两边长为 $a$、$b$，夹角为 $C$，第三边长 $c$ 满足 $c^2=a^2+b^2-2ab\cos C$。$C=90^\circ$ 时 $\cos C=0$，它退化成勾股定理。把两个向量 $\mathbf a$、$\mathbf b$ 从同一点画出，它们的终点连线就是向量 $\mathbf a-\mathbf b$，三者围成一个三角形，夹角就是两个向量的夹角 $\theta$。
:::

::: deriv 点积 = 长度 × 长度 × cos θ
$\norm{\mathbf a-\mathbf b}^2=(\mathbf a-\mathbf b)\cdot(\mathbf a-\mathbf b)$ || 长度的平方等于自己和自己的点积
$=\mathbf a\cdot\mathbf a-2\,\mathbf a\cdot\mathbf b+\mathbf b\cdot\mathbf b$ || 点积对每个参数线性，且对称，所以两个交叉项相等
$=\norm{\mathbf a}^2+\norm{\mathbf b}^2-2\,\mathbf a\cdot\mathbf b$ || 再用一次“自己点自己 = 长度平方”
$\norm{\mathbf a-\mathbf b}^2=\norm{\mathbf a}^2+\norm{\mathbf b}^2-2\norm{\mathbf a}\norm{\mathbf b}\cos\theta$ || 余弦定理，三边为 $\mathbf a$、$\mathbf b$、$\mathbf a-\mathbf b$
$\mathbf a\cdot\mathbf b=\norm{\mathbf a}\norm{\mathbf b}\cos\theta$ || 第 (3) 行与第 (4) 行的左边相同，比较右边
:::

::: eq 点积的几何形式 | (M1.3)
$$\mathbf a\cdot\mathbf b=\norm{\mathbf a}\,\norm{\mathbf b}\cos\theta$$
$\norm{\mathbf b}\cos\theta$ 是 $\mathbf b$ 在 $\mathbf a$ 方向上的“影子”的有向长度，所以点积 $=\norm{\mathbf a}\times$ 影子长。
:::

::: fig figures/m01-dot-geometry.svg | 图 M1.1 点积 = 一个向量的长度 × 另一个向量在它方向上的影子长
**看紫色的粗线段：它是 $\mathbf b$ 在 $\mathbf a$ 方向上的影子。** 从 $\mathbf b$ 的箭头尖向 $\mathbf a$ 所在直线作垂线，垂足到原点的距离就是影子长 $\norm{\mathbf b}\cos\theta\approx2.67$。点积 $11$ 等于 $\mathbf a$ 的长度 $4.12$ 乘以这个影子长。右侧的每个数都可以用计算器复核。
:::

这个几何形式告诉我们点积的符号意味着什么。夹角小于 $90^\circ$ 时 $\cos\theta>0$，点积为正，两个向量大体同向；夹角等于 $90^\circ$ 时点积为 $0$，称两个向量**正交**（orthogonal），互相“看不见”；夹角大于 $90^\circ$ 时点积为负，两个向量大体反向。在 $d=768$ 维里我们画不出夹角，于是反过来用 (M1.3) **定义**夹角：$\cos\theta=\mathbf a\cdot\mathbf b/(\norm{\mathbf a}\norm{\mathbf b})$。这个比值一定落在 $[-1,1]$ 里，这就是柯西–施瓦茨不等式 $\abs{\mathbf a\cdot\mathbf b}\le\norm{\mathbf a}\norm{\mathbf b}$，证明只要几行，放在练习 7。

## 余弦相似度：只看方向，不看长短

把点积除以两个长度，就得到**余弦相似度**（cosine similarity）：

::: eq 余弦相似度 | (M1.4)
$$\cos(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\norm{\mathbf a}\,\norm{\mathbf b}}\in[-1,1]$$
把任一个向量放大 10 倍，余弦相似度不变，点积却变成 10 倍。
:::

余弦相似度只关心方向，所以在“文档检索”“句向量搜索”里很常用：一篇文档写得长、向量模长大，不应该因此更“相关”。注意力却**没有**用余弦，而是用点积本身。两者的差别用一个例子最清楚。

::: fig figures/m01-dot-vs-cosine.svg | 图 M1.2 同一个查询、三个键：点积与余弦给出不同的赢家
**看表格最后一列的排名：按点积，最长的 $\mathbf k_2$ 赢；按余弦，方向最接近 $\mathbf q$ 的 $\mathbf k_1$ 赢。** 虚线单位圆上的点是各向量的方向，余弦只看这些点之间的角度；点积还要乘上箭头的长度。右下角把三个点积直接送进 softmax（第 1 课会讲它把分数变成和为 1 的权重），$\mathbf k_2$ 拿走约 97.6% 的权重。
:::

::: ex 点积与余弦的排名为什么不同
查询 $\mathbf q=(2,1)$，$\norm{\mathbf q}=\sqrt5\approx2.236$。三个键：

- $\mathbf k_1=(1,1)$：点积 $2+1=3$；$\norm{\mathbf k_1}=\sqrt2\approx1.414$；余弦 $3/(2.236\times1.414)\approx0.949$。
- $\mathbf k_2=(4,-1)$：点积 $8-1=7$；$\norm{\mathbf k_2}=\sqrt{17}\approx4.123$；余弦 $7/(2.236\times4.123)\approx0.759$。
- $\mathbf k_3=(0,2)$：点积 $0+2=2$；$\norm{\mathbf k_3}=2$；余弦 $2/(2.236\times2)\approx0.447$。

按余弦，$\mathbf k_1$ 最相关（夹角约 $18^\circ$）；按点积，$\mathbf k_2$ 以 7 分遥遥领先，因为它的长度是 $\mathbf k_1$ 的近 3 倍，足以弥补它方向上的偏差。
:::

这个差别在注意力里是有用的，要分查询和键两边看。**查询的长度**对同一个查询的所有分数一视同仁：$\mathbf q$ 变长一倍，所有分数都翻倍，softmax 之后分布变得更尖，这相当于调低了温度（温度的数学见 [M04 §4](m04-probability.html#p4)）。**键的长度**则因键而异：一个键更长，它在每个查询面前的“嗓门”都更大。用点积打分，模型就多了一个可以学的自由度：既能表达“方向对不对”，也能表达“有多确定、多重要”。余弦把这两件事都抹掉了。这是从公式本身读出来的性质；训练好的模型实际怎样利用向量长度，要看具体模型，本课不下结论。

## 为什么注意力用点积打分

把上面的内容收拢起来，点积胜出有四个理由。第 2、3 条正是开头提出的三个条件（便宜、可导、能批量）。

1. **它量的是“同向程度”。** 由 (M1.3)，方向越一致、长度越长，分数越高；正交则为 0。
2. **便宜，而且处处可导。** 长度为 $d_h$ 的点积约 $2d_h$ 次运算，$d_h=64$ 时只有 128 次。它只由乘法和加法组成，对 $\mathbf q$ 的导数就是另一个向量本身：$\partial(\mathbf q\cdot\mathbf k)/\partial\mathbf q=\mathbf k$（导数的记号见 [M06 §2](m06-calculus-backprop.html#p2)）。
3. **能批量。** 所有查询与所有键的点积恰好是一次矩阵乘法 $QK^\top$（下一页）。《Attention Is All You Need》比较过点积注意力与“加性注意力”（用一个只有单个隐藏层的前馈网络打分）：两者理论复杂度相近，但点积注意力能用高度优化的矩阵乘法代码实现，实践中快得多、也更省内存 [@vaswani2017]。
4. **可学。** 打分的并不是隐藏状态本身，而是经过两个不同的、学出来的矩阵变换之后的 $\mathbf q_t=W^Q\mathbf h_t$ 与 $\mathbf k_j=W^K\mathbf h_j$。所以“相关”的标准不是固定的几何相似，而是模型学出来的（[§6](#p6) 会把它写成 $\mathbf h_t^\top(W^{Q})^\top W^K\mathbf h_j$ 来看）。

::: ana 类比：注意力 = 图书馆检索
**映射。** 查询 $\mathbf q_t$ = 你带着的检索问题；键 $\mathbf k_j$ = 每本书书脊上的标签；点积 = 问题与标签的匹配度；softmax = 按匹配度分配阅读时间；值 $\mathbf v_j$ = 书的内容；输出 = 按阅读时间加权的读书笔记。

**收益。** 它解释了为什么要把“标签”和“内容”分开：决定读不读一本书的依据（键），和读到的东西（值），可以是两回事。

**失效的地方。** 真实的检索是挑出一本书，注意力却把所有书按权重混在一起；标签和内容都是学出来的，同一本书在不同的头里可以贴完全不同的标签（每个头有自己的 $W^K$）。还有：点积分数会因为标签“写得大”（键长）而绝对值变大（方向大体一致时就是变高），真实的图书馆里没有这回事。
:::

::: warn 失败模式：维度越高，点积越大
点积是 $d_h$ 项之和，项数越多，和的波动越大。《Attention Is All You Need》给出一个简单的估算：若 $\mathbf q$、$\mathbf k$ 的各分量是相互独立、均值 0、方差 1 的随机变量，则 $\mathbf q\cdot\mathbf k$ 的均值为 0、方差为 $d_k$ [@vaswani2017]（这一步的推导在 [M04 §3](m04-probability.html#p3)）。$d_k=64$ 时标准差是 8：分数动辄相差十几，$e^{16}$ 约为 $8.9\times10^6$，softmax 输出几乎变成只有一个 1 的“独热”分布，梯度也随之几乎消失。要分清哪些是算出来的：方差 $d_k$ 在上述独立性假设下是证明过的；“点积变大会把 softmax 推进梯度极小的区域”，原论文写的是推测（原句以 “We suspect” 开头），方差估算是它为这个推测给的说明。第 1 课的公式里除以 $\sqrt{d_k}$，就是论文据此加上的修正。
:::

:::: quiz 小练习：点积与余弦
$\mathbf q=(1,2,2)$，$\mathbf k=(2,0,1)$。求 $\mathbf q\cdot\mathbf k$ 与余弦相似度。如果把 $\mathbf k$ 换成 $3\mathbf k$，两者各变成多少？
::: answer 看答案
$\mathbf q\cdot\mathbf k=2+0+2=4$。$\norm{\mathbf q}=\sqrt{1+4+4}=3$，$\norm{\mathbf k}=\sqrt{4+0+1}=\sqrt5\approx2.236$，余弦 $=4/(3\times2.236)\approx0.596$。换成 $3\mathbf k$ 后点积变成 $12$（线性），余弦仍是 $0.596$（分子分母都乘 3）。
:::
::::

::: note 这一页用在哪里
[第 1 课 §4](../deepseek/lesson-01-transformer-anatomy.html#p4)：缩放点积注意力、除以 $\sqrt{d_k}$ 的理由。[第 4 课 §2](../deepseek/lesson-04-inference-kv-cache.html#p2)：解码时每一步用新 token 的查询去和缓存里所有的键做点积，每个头 $T$ 个长度为 $d_h$ 的点积。[第 9 课 §3](../deepseek/lesson-09-v2-mla.html#p3)：MLA 的打分仍是点积，只是经过结合律改写后，一端变成了压缩向量（本课 [§6](#p6)）。
:::

::: key 本页要点
- 点积 $\mathbf a\cdot\mathbf b=\sum_ia_ib_i=\norm{\mathbf a}\norm{\mathbf b}\cos\theta$：同向为正，正交为零，反向为负。
- 余弦相似度只看方向；点积还看长度。注意力用点积，于是查询长度起“温度”作用，键长度起“嗓门”作用。
- 点积便宜、可导、能批量成矩阵乘法，这是注意力用它打分的工程理由。维度一高它的波动就大，所以要除以 $\sqrt{d_k}$。
:::

# 矩阵乘法的三种读法 | 线性映射、批量点积、列的组合与外积之和 | 25

“行乘列”这条规则人人会背，但只会这一种读法，读注意力公式时会处处卡住。同一个乘积 $AB$，问不同的问题时应该换不同的眼光看：问“这个矩阵把向量变成了什么”，要看成**线性映射**；问“$QK^\top$ 的每一格是什么”，要看成**批量点积**；问“注意力的输出是怎么来的”，要看成**列（行）的组合**；问“一个矩阵能不能拆成更便宜的两块”，要看成**外积之和**。这一页先给定义，再用同一组数把几种读法各走一遍。

## 定义：形状规则与逐格公式

**含义。** 矩阵是排成长方形的一张数表。$A\in\R^{m\times k}$ 有 $m$ 行、$k$ 列，第 $i$ 行第 $j$ 列的数记作 $A_{ij}$。

**理解。** 同一张表可以横着看，也可以竖着看：横着看，它是 $m$ 个长度为 $k$ 的行向量上下叠在一起；竖着看，它是 $k$ 个长度为 $m$ 的列向量左右排在一起。下面几种读法的区别，说到底就是把左右两个矩阵各自看成“一摞行”还是“一排列”。在语言模型里，矩阵几乎都是学出来的参数表：GPT-2 small 全部头的查询矩阵 $W^Q$ 是一张 $768\times768$ 的表 [@gpt2-config]，589,824 个数全是训练得到的。

**数学。** 两个矩阵能相乘，当且仅当左边的列数等于右边的行数；乘积的每一格是左边一行与右边一列的点积：

::: eq 矩阵乘法 | (M1.5)
$$C=AB,\qquad C_{ij}=\sum_{k}A_{ik}B_{kj},\qquad [m,k]\times[k,n]\to[m,n]$$
中间那个相同的 $k$ 被求和“吃掉”，两边剩下的 $m$ 和 $n$ 成为结果的形状。
:::

(M1.5) 里的 $k$ 身兼两职：在形状 $[m,k]$ 里它是中间维的**大小**，在 $\sum_k$ 里它是从 1 数到这个大小的**下标**。这是线性代数书里常见的简写，本课沿用；[§4](#p4) 需要把两者分开时，会把大小写成大写的 $K$。

向量是只有一列的矩阵：$\mathbf a\in\R^{d}$ 的形状是 $[d,1]$，$\mathbf a^\top$ 是 $[1,d]$。于是 $\mathbf a^\top\mathbf b$ 是 $[1,d]\times[d,1]\to[1,1]$，一个数，这就是 (M1.2) 写成 $\mathbf a^\top\mathbf b$ 的原因。把顺序反过来，$\mathbf a\mathbf b^\top$ 是 $[d,1]\times[1,d]\to[d,d]$，一整张表，叫**外积**（outer product），它的第 $(i,j)$ 格是 $a_ib_j$。同样两个向量，换个顺序，一个是数，一个是矩阵。本页最后一种读法就建立在外积上。

::: ex 手算一个 2×2 乘积，顺便看到“不能交换”
取两个普通的矩阵（字母 $F$、$G$ 只是名字，本课的 $P$、$Q$ 留给注意力）：

$$F=\begin{pmatrix}1&2\\3&4\end{pmatrix},\qquad G=\begin{pmatrix}5&6\\7&8\end{pmatrix}.$$

逐格算：$(FG)_{11}=1\times5+2\times7=19$，$(FG)_{12}=1\times6+2\times8=22$，$(FG)_{21}=3\times5+4\times7=43$，$(FG)_{22}=3\times6+4\times8=50$。反过来算 $GF$，例如 $(GF)_{11}=5\times1+6\times3=23$。结果：

$$FG=\begin{pmatrix}19&22\\43&50\end{pmatrix}\ \ne\ GF=\begin{pmatrix}23&34\\31&46\end{pmatrix}.$$

矩阵乘法一般**不满足交换律**。下面的读法一会说明为什么本来就不该指望它满足。
:::

::: err 把矩阵乘法当成逐格相乘
$(FG)_{11}$ 不是 $F_{11}G_{11}=5$，而是一行点一列 $=19$。逐格相乘是另一种运算，叫**逐元素乘积**（Hadamard product），要求两个矩阵形状完全相同。NumPy 和 PyTorch 里 `F * G` 是逐元素乘积，`F @ G` 才是矩阵乘法；两者对方阵都“算得出来”，写错了不会报错。
:::

## 读法一：线性映射——矩阵是一台变换向量的机器

**问题。** 为什么乘法偏偏定义成“行乘列”这种古怪的样子？因为矩阵首先是一个**函数**：喂进去一个向量 $\mathbf x\in\R^{n}$，吐出一个向量 $A\mathbf x\in\R^{m}$。定义 (M1.5) 是让“先做一个变换、再做另一个变换”仍然是一个矩阵的唯一写法。

**图景。** 把 $A\mathbf x$ 按 (M1.5) 展开，第 $i$ 个分量是 $\sum_jA_{ij}x_j$。换一种分组方式看，它等于用 $\mathbf x$ 的各个分量去给 $A$ 的各**列**加权再相加：

::: eq 矩阵乘向量 = 列的加权和 | (M1.6)
$$A\mathbf x=x_1\mathbf a_1+x_2\mathbf a_2+\cdots+x_n\mathbf a_n,\qquad \mathbf a_j=A\text{ 的第 }j\text{ 列}$$
特别地，$A\mathbf e_j=\mathbf a_j$：第 $j$ 个坐标轴方向的单位向量 $\mathbf e_j$ 被送到 $A$ 的第 $j$ 列。
:::

所以一个矩阵完全由“它把每个坐标轴送到哪里”决定，而这些去处就写在它的各列上。其余的点跟着走：$\mathbf x=(1,2)$ 等于 1 步 $\mathbf e_1$ 加 2 步 $\mathbf e_2$，变换后就是 1 步 $A\mathbf e_1$ 加 2 步 $A\mathbf e_2$。这种“先按坐标拆开、各自变换、再加起来”也行得通的性质叫**线性**（linearity）：$A(\mathbf x+\mathbf y)=A\mathbf x+A\mathbf y$，$A(\lambda\mathbf x)=\lambda A\mathbf x$。几何上，它意味着原点不动、直线仍是直线、平行等距的网格线变换后仍然平行等距（这里假定 $A$ 没有把平面压扁；若某个方向被压成零，比如 $A$ 的两列共线，整条直线可能被压成一个点，这种情形 [M02 §1](m02-rank-svd-lowrank.html#p1) 讲秩时再说）。

::: fig figures/m01-linear-map.svg | 图 M1.3 矩阵 A 把整个平面连同网格一起变换
**看右图中蓝、绿两支箭头：它们就是 $A$ 的两列 $(2,1)$ 和 $(-1,1)$，也就是两个坐标轴方向的单位向量被送到的位置。** 黑色箭头 $\mathbf x=(1,2)$ 在左图里是“1 步蓝 + 2 步绿”，变换后仍是“1 步新蓝 + 2 步新绿”，落在 $(0,3)$。黄色的单位正方形变成面积为 3 的平行四边形；这个面积倍数叫行列式，本课用不到，[M02 §3](m02-rank-svd-lowrank.html#p3) 会再遇到它。
:::

**推导：乘法为什么是行乘列。** 先用 $B$ 变换、再用 $A$ 变换，得到 $A(B\mathbf x)$。我们要求存在一个矩阵，记作 $AB$，让 $(AB)\mathbf x=A(B\mathbf x)$ 对所有 $\mathbf x$ 成立。把两步都展开：

::: deriv 复合两个线性映射
$(B\mathbf x)_k=\sum_{j}B_{kj}x_j$ || 先用 $B$：(M1.5) 应用于一个向量
$\big(A(B\mathbf x)\big)_i=\sum_{k}A_{ik}(B\mathbf x)_k$ || 再用 $A$ 作用在 $B\mathbf x$ 上
$=\sum_{k}A_{ik}\sum_{j}B_{kj}x_j=\sum_{j}\Big(\sum_{k}A_{ik}B_{kj}\Big)x_j$ || 代入第 (1) 行，交换两个有限求和的次序
$(AB)_{ij}=\sum_{k}A_{ik}B_{kj}$ || 要对所有 $\mathbf x$ 成立，$x_j$ 前面的系数就必须是 $(AB)_{ij}$
:::

这就是 (M1.5)：行乘列不是人为规定，而是“复合变换”这个要求逼出来的。顺带得到两个结论。第一，形状必须对得上：$B$ 输出的维数就是 $A$ 输入的维数，也就是 $B$ 的行数等于 $A$ 的列数。第二，**顺序有意义**：$AB$ 表示先做 $B$ 后做 $A$（从右往左读），先后调换通常得到不同的变换。

::: ex 先旋转再拉伸 ≠ 先拉伸再旋转
$$D=\begin{pmatrix}2&0\\0&1\end{pmatrix},\qquad R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.$$

$D$ 把横坐标拉长 2 倍（它是对角矩阵，diagonal，所以记作 $D$；下面的 $S$ 要留给打分表）；$R$ 逆时针旋转 $90^\circ$（它的两列说明它把 $\mathbf e_1$ 送到 $(0,1)$、把 $\mathbf e_2$ 送到 $(-1,0)$，对照 (M1.6)）。取 $\mathbf x=(1,1)$：

- 先转后拉：$R\mathbf x=(-1,1)$，再 $D$：$DR\mathbf x=(-2,1)$。
- 先拉后转：$D\mathbf x=(2,1)$，再 $R$：$RD\mathbf x=(-1,2)$。

两者不同，所以 $DR\ne RD$。这件小事在第 9 课会变成大事：DeepSeek-V2 的论文说明 RoPE 为什么破坏矩阵吸收时，理由之一正是“矩阵乘法不满足交换律”，一个随位置变化的旋转矩阵夹在两个权重矩阵之间，挪不出去 [@deepseekv2]。
:::

在注意力里，读法一对应 $\mathbf q_t=W^Q\mathbf h_t$。这里 $W^Q$ 不必是方阵：第 $i$ 个头的 $W^Q_i\in\R^{d_h\times d}$（下标 $i$ 是头的编号，[§4](#p4) 会看到它是大矩阵 $W^Q$ 的一块）把 $d$ 维的隐藏状态映射成 $d_h$ 维的查询，形状是 $[d_h,d]\times[d,1]\to[d_h,1]$。GPT-2 small 里就是从 768 维映射到 64 维 [@gpt2-config]。可以把它读成“这个头从 768 个栏目里读出 64 个它关心的组合”，[§6](#p6) 会把这句话说准确。

## 读法二：批量点积——一张打分表

(M1.5) 的每一格 $C_{ij}$ 是 $A$ 的第 $i$ 行与 $B$ 的第 $j$ 列的点积。所以一个矩阵乘积就是一张**打分表**：左边每一行和右边每一列两两配对，各自做一次点积。

这正是注意力需要的。先只看一个头（省略头的下标 $i$）：把 $T$ 个查询按行堆成 $Q\in\R^{T\times d_h}$（第 $t$ 行是 $\mathbf q_t^\top$），$T$ 个键按行堆成 $K\in\R^{T\times d_h}$。转置之后 $K^\top\in\R^{d_h\times T}$ 的第 $j$ 列就是 $\mathbf k_j$。于是

::: eq 注意力分数 = 一次矩阵乘法 | (M1.7)
$$S=QK^\top\in\R^{T\times T},\qquad S_{tj}=\mathbf q_t^\top\mathbf k_j$$
形状：$[T,d_h]\times[d_h,T]\to[T,T]$。第 $t$ 行是第 $t$ 个 token 对所有 token 的打分。
:::

::: ex 三个 token 的打分表
取 $d_h=2$，三个查询和三个键：

$$Q=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\quad K=\begin{pmatrix}1&2\\3&0\\0&1\end{pmatrix},\quad QK^\top=\begin{pmatrix}1&3&0\\2&0&1\\3&3&1\end{pmatrix}.$$

核对一格：$S_{32}=\mathbf q_3^\top\mathbf k_2=(1,1)\cdot(3,0)=3$。第 1 课会给这张表加上**因果掩码**：第 $t$ 行只保留 $j\le t$ 的格子（token 只能看自己和前面的 token），也就是只用下三角的 $1$；$2,0$；$3,3,1$。
:::

## 读法三：列的组合，以及外积之和

(M1.6) 说矩阵乘一个向量等于给各列加权求和。把 $B$ 看成并排放着的 $n$ 个列向量，$AB$ 就是把 $A$ 分别乘上每一列：$C$ 的第 $j$ 列 $=\sum_kB_{kj}\mathbf a_k$，是 $A$ 的各列按 $B$ 第 $j$ 列给出的系数加权相加。对行也有对称的说法：$C$ 的第 $i$ 行 $=\sum_kA_{ik}\,(B\text{ 的第 }k\text{ 行})$，是 $B$ 的各行按 $A$ 第 $i$ 行的系数加权相加。

行的版本就是注意力的输出。softmax 把打分表 $S$ 的每一行变成一组非负、和为 1 的权重，得到 $P\in\R^{T\times T}$；把 $T$ 个值向量按行堆成 $V\in\R^{T\times d_h}$。输出 $O=PV$ 的第 $t$ 行是

::: eq 注意力输出 = 值向量的加权平均 | (M1.8)
$$\mathbf o_t^\top=\sum_{j}P_{tj}\,\mathbf v_j^\top,\qquad P_{tj}\ge0,\ \ \sum_jP_{tj}=1$$
形状：$[T,T]\times[T,d_h]\to[T,d_h]$。每个输出都是所有值向量的凸组合（加权平均）。
:::

::: ex 用行的组合算注意力输出
$$P=\begin{pmatrix}1&0&0\\0.5&0.5&0\\0.2&0.3&0.5\end{pmatrix},\qquad V=\begin{pmatrix}1&0\\0&2\\4&4\end{pmatrix}.$$

$P$ 的每行和为 1，右上角为 0，相当于已经加了因果掩码。第 3 行：$0.2\cdot(1,0)+0.3\cdot(0,2)+0.5\cdot(4,4)=(0.2+0+2,\ 0+0.6+2)=(2.2,\ 2.6)$。第 2 行：$0.5\cdot(1,0)+0.5\cdot(0,2)=(0.5,1)$。第 1 行只能看自己：$(1,0)$。每个输出都落在三个值向量围成的三角形里（含边界），这就是“加权平均”的几何含义。
:::

把 $k$ 这个求和下标放到最外层，就得到读法三的另一面，也是最不直观的一种看法：$C$ 是 $k$ 个外积之和。

::: eq 矩阵乘积 = 外积之和 | (M1.9)
$$AB=\sum_{k}\mathbf a_k\,\mathbf b_k^{\top},\qquad \mathbf a_k=A\text{ 的第 }k\text{ 列},\ \ \mathbf b_k^\top=B\text{ 的第 }k\text{ 行}$$
理由：外积 $\mathbf a_k\mathbf b_k^\top$ 的第 $(i,j)$ 格是 $A_{ik}B_{kj}$，对 $k$ 求和正是 (M1.5)。
:::

每一个外积 $\mathbf a_k\mathbf b_k^\top$ 都是一张“一列乘一行”的表，它的每一行都是同一个行向量 $\mathbf b_k^\top$ 的倍数，只要两个向量都不是零向量，这种矩阵就叫**秩 1 矩阵**（rank-1 matrix）。(M1.9) 说：中间维为 $k$ 的乘积，至多是 $k$ 个秩 1 矩阵之和。如果中间维很小，这个乘积就是一个“低秩”矩阵，用两块瘦矩阵存它比存整张大表便宜得多。这是 [M02 §2](m02-rank-svd-lowrank.html#p2) 的主题，也是第 9 课 MLA“低秩 KV 联合压缩”的数学骨架：DeepSeek-V2 先用 $W^{DKV}\in\R^{d_c\times d}$ 把隐藏状态压成 $d_c$ 维，再用 $W^{UK}\in\R^{d_hn_h\times d_c}$ 展开成键 [@deepseekv2]，两者的乘积 $W^{UK}W^{DKV}$ 的中间维只有 $d_c$。

::: fig figures/m01-three-readings.svg | 图 M1.4 同一个乘积 C = AB，三种算法得到同一个答案
**看三行的结果（绿色）都是同一个 $C$：第一行 $(4,1)$，第二行 $(4,12)$。** ① 批量点积，也就是逐格算：一行点一列。② 按列算：$C$ 的一列是 $A$ 各列的加权和，权重来自 $B$ 的对应列。③ 按外积算：$A$ 的每一列和 $B$ 的对应行做外积，得到三张秩 1 的表，加起来。三种算法做的乘法次数相同（都是 $2\times3\times2=12$ 次），只是把同一个三重求和按不同的顺序组织。
:::

| 读法 | 公式 | 在注意力里对应 | 适合回答的问题 |
|---|---|---|---|
| 线性映射 | $A\mathbf x=\sum_jx_j\mathbf a_j$ | $\mathbf q_t=W^Q\mathbf h_t$ | 这个矩阵把向量变成了什么？ |
| 批量点积 | $C_{ij}=\text{行}_i\cdot\text{列}_j$ | $S=QK^\top$ | 表里每一格是什么？ |
| 行（列）的组合 | $C$ 的行 $=\sum_kA_{ik}\,(B\text{ 的行}_k)$ | $O=PV$ | 输出是怎么混出来的？ |
| 外积之和 | $AB=\sum_k\mathbf a_k\mathbf b_k^\top$ | $W^{UK}W^{DKV}$（第 9 课） | 能不能拆成更便宜的两块？ |

:::: quiz 小练习：认出读法
$$A=\begin{pmatrix}1&1\\0&2\end{pmatrix},\qquad \mathbf x=(3,-1).$$

（1）用“列的加权和”算 $A\mathbf x$。（2）把 $A$ 写成两个外积之和。
::: answer 看答案
（1）$A\mathbf x=3\cdot(1,0)+(-1)\cdot(1,2)=(3-1,\ 0-2)=(2,-2)$。用逐行点积核对：$(1,1)\cdot(3,-1)=2$，$(0,2)\cdot(3,-1)=-2$。

（2）把 $A$ 看成 $A\cdot I$（$I$ 是单位矩阵），按 (M1.9)：$A=\mathbf a_1\mathbf e_1^\top+\mathbf a_2\mathbf e_2^\top=\begin{pmatrix}1&0\\0&0\end{pmatrix}+\begin{pmatrix}0&1\\0&2\end{pmatrix}$。拆法不唯一，例如 $\begin{pmatrix}1\\0\end{pmatrix}(1,1)+\begin{pmatrix}0\\2\end{pmatrix}(0,1)$（按行拆）也对。
:::
::::

::: note 这一页用在哪里
[第 1 课 §3](../deepseek/lesson-01-transformer-anatomy.html#p3)：注意力的三步恰好是三种读法：投影（读法一）、打分 $QK^\top$（读法二）、加权求和 $PV$（读法三）。[第 4 课 §2](../deepseek/lesson-04-inference-kv-cache.html#p2)：解码时 $Q$ 只有一行，打分表退化成一行 $1\times T$，这就是“每步只需要新 token 的查询”的矩阵形式。[第 9 课 §2](../deepseek/lesson-09-v2-mla.html#p2)与 [M02 §2](m02-rank-svd-lowrank.html#p2)：外积之和与低秩分解。
:::

::: key 本页要点
- $[m,k]\times[k,n]\to[m,n]$：中间维被求和吃掉。乘法是“复合变换”逼出来的，所以从右往左读，而且一般不能交换。
- 同一个乘积可以读成：变换（矩阵乘向量 = 各列加权和）、打分表（行点列）、加权组合（行或列的组合）、外积之和（秩 1 的叠加）。
- 注意力 $=$ 投影（读法一）$+$ 打分 $QK^\top$（读法二）$+$ 加权平均 $PV$（读法三）。
:::

# 形状思维 | 转置、论文与代码的两种写法、einsum | 25

一个张量公式如果形状对不上，它根本算不出来；形状对得上但意思错了，就更难发现。读论文、写代码时，最有效的一个习惯是：**每写下一个乘积，就在旁边写出它的形状，并问自己结果的形状是不是预期的那个**。这一页给出做这件事需要的三样工具：转置规则、论文写法与代码写法之间的翻译，以及把下标明明白白写出来的 einsum 记号。

## 转置，以及“穿脱法则”

**转置**（transpose）把矩阵的行变成列：$(A^\top)_{ij}=A_{ji}$，形状 $[m,n]\to[n,m]$。转置两次回到原样：$(A^\top)^\top=A$。一个乘积的转置满足下面的规则，推导只用定义。

::: deriv 乘积的转置
$\big((AB)^\top\big)_{ij}=(AB)_{ji}$ || 转置的定义：交换两个下标
$=\sum_kA_{jk}B_{ki}$ || 矩阵乘法 (M1.5)
$=\sum_k(B^\top)_{ik}(A^\top)_{kj}$ || 再用一次转置的定义；两个**数**相乘可以交换次序
$=(B^\top A^\top)_{ij}$ || 这正是 $B^\top$ 乘 $A^\top$ 的第 $(i,j)$ 格
:::

::: eq 乘积的转置：顺序反过来 | (M1.10)
$$(AB)^\top=B^\top A^\top,\qquad (ABC)^\top=C^\top B^\top A^\top$$
像穿衣服和脱衣服：先穿衬衫再穿外套，脱的时候先脱外套。
:::

形状会逼你把顺序反过来。设 $A$ 是 $[m,k]$，$B$ 是 $[k,n]$，$(AB)^\top$ 是 $[n,m]$。$B^\top A^\top$ 是 $[n,k]\times[k,m]\to[n,m]$，对得上；而 $A^\top B^\top$ 是 $[k,m]\times[n,k]$，只要 $m\ne n$ 就根本乘不了。

::: ex 用 §2 的 F、G 验证
$$(FG)^\top=\begin{pmatrix}19&43\\22&50\end{pmatrix},\qquad G^\top F^\top=\begin{pmatrix}5&7\\6&8\end{pmatrix}\begin{pmatrix}1&3\\2&4\end{pmatrix}.$$

按 (M1.10) 算右边：左上 $5+14=19$，右上 $15+28=43$，左下 $6+16=22$，右下 $18+32=50$，与左边一致。如果错写成 $F^\top G^\top$，得到第一行 $(23,31)$、第二行 $(34,46)$，那是 $(GF)^\top$。这里 $F$、$G$ 都是方阵，错的顺序照样算得出来，形状检查帮不上忙，下面会专门说这个陷阱。
:::

一个数的转置是它自己，所以 $\mathbf a^\top\mathbf b=(\mathbf a^\top\mathbf b)^\top=\mathbf b^\top\mathbf a$：点积对称的另一种证明。同理，对任意矩阵 $A$ 有 $\mathbf x^\top A\mathbf y=\mathbf y^\top A^\top\mathbf x$（练习 8 会用它说明注意力为什么不对称）。

## 论文写列，代码写行：用转置来回翻译

DeepSeek 的论文用**列向量**写注意力：$\mathbf q_t=W^Q\mathbf h_t$，其中 $\mathbf h_t\in\R^{d}$，$W^Q\in\R^{d_hn_h\times d}$ [@deepseekv2]。本课程沿用这种写法。可是代码一次处理整段序列，惯例是把 $T$ 个 token **按行**堆成矩阵 $H\in\R^{T\times d}$，第 $t$ 行是 $\mathbf h_t^\top$。两种写法之间差一个转置：

::: deriv 从一个 token 的列写法到整段序列的行写法
$\mathbf q_t=W^Q\mathbf h_t$ || 论文写法，形状 $[d_hn_h,d]\times[d,1]\to[d_hn_h,1]$
$\mathbf q_t^\top=\mathbf h_t^\top(W^Q)^\top$ || 两边转置，用 (M1.10)
$Q=H(W^Q)^\top$ || 把 $T$ 行 $\mathbf q_t^\top$ 竖着叠起来，右边也逐行叠起来：读法三“$C$ 的第 $t$ 行只依赖 $A$ 的第 $t$ 行”
:::

::: eq 同一个投影的两种写法 | (M1.11)
$$\underbrace{\mathbf q_t=W^Q\mathbf h_t}_{\text{论文：一次一列}}\quad\Longleftrightarrow\quad\underbrace{Q=H\,(W^Q)^\top}_{\text{代码：}[T,d]\times[d,\,d_hn_h]}$$
行写法里出现的权重是列写法权重的转置。
:::

不同的文献和代码库在这件事上各行其是，读的时候要先确认对方用的是哪一种。

- 《Attention Is All You Need》用行写法：$\mathrm{head}_i=\Attn(QW_i^Q,KW_i^K,VW_i^V)$，其中 $W_i^Q\in\R^{d_{\text{model}}\times d_k}$ [@vaswani2017]。它的 $W_i^Q$ 就是我们的 $(W^Q_i)^\top$。
- PyTorch 的 `nn.Linear` 把权重按 `(out_features, in_features)` 存放，也就是列写法的矩阵，计算时再转置：$\mathbf y=\mathbf xA^\top+\mathbf b$ [@torch-linear]。
- Hugging Face 的 GPT-2 实现用一个叫 `Conv1D` 的层，权重形状是 `(nx, nf)`，即 `(输入维, 输出维)`，计算时直接 `x @ W`，不转置；它的文档注释写明这个层和线性层一样，只是权重转置存放 [@hf-gpt2]。所以把 GPT-2 的权重装进 `nn.Linear` 时，每个矩阵都要先转置。

::: warn 失败模式：方阵让形状检查失灵
GPT-2 small 里 $Q,K,V$ 三个投影合在一起的权重 `c_attn` 是 $[768,2304]$ [@hf-gpt2]，存反了方向，矩阵乘法会因形状不符当场报错。输出投影 `c_proj` 却是 $[768,768]$ 的方阵：存反了照样能乘，程序不报任何错，只是每一层的输出都是错的。形状检查只能抓住“乘不了”的错误；对方阵，要靠对照定义（谁是输入维、谁是输出维）或者拿已知输入核对输出。
:::

## 注意力里的形状

有了行写法，就可以把一个头的注意力逐步写出形状。下面用下标 $i$ 表示第 $i$ 个头，$H\in\R^{T\times d}$ 是整段输入。

| 步骤 | 形状 | GPT-2 small 的结果（$T=1024$） |
|---|---|---|
| 投影 $Q_i=H(W^Q_i)^\top$（$K_i$、$V_i$ 同理） | $[T,d]\times[d,d_h]\to[T,d_h]$ | $[1024,64]$ |
| 打分 $S_i=Q_iK_i^\top/\sqrt{d_h}$ | $[T,d_h]\times[d_h,T]\to[T,T]$ | $[1024,1024]$ |
| 归一化 $P_i=\softmax(S_i)$，逐行 | $[T,T]\to[T,T]$ | 每行 1024 个权重，和为 1 |
| 加权求和 $O_i=P_iV_i$ | $[T,T]\times[T,d_h]\to[T,d_h]$ | $[1024,64]$ |

::: fig figures/m01-attention-shapes.svg | 图 M1.5 一个注意力头里的三次矩阵乘法（按真实形状等比例绘制）
**看紫色的方块：打分表 $S_i$ 与权重表 $P_i$ 是 $[T,T]$，比细长的 $Q_i$、$K_i$、$V_i$ 大得多。** $T=1024$ 时，$S_i$ 有 $1024^2=1{,}048{,}576$ 格，$Q_i$ 只有 $1024\times64=65{,}536$ 格，差 16 倍；$T$ 每翻一倍，这个差距再翻一倍。红色上三角是因果掩码要丢掉的部分。每步下方的运算次数按 [§5](#p5) 的 $2mnk$ 算出。形状取自 GPT-2 small 的配置 [@gpt2-config]。
:::

图里的比例说明了一件第 4 课、第 14 课反复用到的事：注意力里真正随上下文长度平方增长的，是 $[T,T]$ 这张表，以及生成它、使用它的两次乘法。投影部分只随 $T$ 线性增长。

## einsum：把下标写出来

**问题。** 一旦加上“多个头”“一批序列”，张量就有 3 到 4 个轴，“矩阵乘法”这个词已经说不清是哪两个轴在配对、哪个轴被求和。与其靠 `transpose`、`reshape` 反复折腾，不如把下标直接写出来。

::: pre 张量的“轴”与 reshape | 用过 NumPy 数组的可以跳过
一个形状为 $[T,n_h,d_h]$ 的张量就是一个三重下标的数表，每个下标的取值范围叫一个**轴**（axis）。`reshape` 只改变“怎么数下标”，不改变数据本身：把 $[T,d]$ reshape 成 $[T,n_h,d_h]$，就是把每一行的 $d=768$ 个数按顺序每 64 个切一段。`transpose` 交换两个轴的次序。两者都不做任何乘法加法。
:::

**规则。** 爱因斯坦求和记号（Einstein summation），在 NumPy、PyTorch 里叫 `einsum`，本课用到的只有两条规则（显式写出 `->` 的写法）[@numpy-einsum]：给每个输入张量的每个轴起一个字母；**输出里没有出现的字母就求和，输出里出现的字母就保留**，按写出的顺序排列。

| 运算 | 数学 | einsum 写法 | 输出形状 |
|---|---|---|---|
| 点积 | $\sum_ia_ib_i$ | `i,i->` | 标量 |
| 外积 | $\mathbf a\mathbf b^\top$ | `i,j->ij` | $[m,n]$ |
| 矩阵乘向量 | $A\mathbf x$ | `ij,j->i` | $[m]$ |
| 矩阵乘法 | $AB$ | `ik,kj->ij` | $[m,n]$ |
| 转置 | $A^\top$ | `ij->ji` | $[n,m]$ |

注意外积和点积的差别只在输出：`i,j->ij` 保留两个不同的字母，所以什么都不求和；`i,i->` 两个输入共用字母 `i`，输出里没有它，所以对 `i` 求和。

多头注意力的每一步都可以写成一个 einsum。约定字母：`h` 头，`t` 查询位置，`s` 键位置，`d` 模型维，`e` 头内维度（大小 $d_h$）。

```python
import numpy as np

T, d, n_h = 1024, 768, 12          # GPT-2 small 的一层，一条长 1024 的序列
d_h = d // n_h                      # 64
X  = np.random.randn(T, d)          # 每行一个 token 的隐藏状态
Wq = np.random.randn(n_h, d_h, d)   # 论文写法 W^Q ∈ R^{n_h d_h × d}，按头切成 n_h 块
Wk = np.random.randn(n_h, d_h, d)
Wv = np.random.randn(n_h, d_h, d)
Wo = np.random.randn(d, n_h, d_h)   # W^O ∈ R^{d × n_h d_h}，按头切成 n_h 块

Q = np.einsum('td,hed->hte', X, Wq)            # [n_h, T, d_h]
K = np.einsum('td,hed->hte', X, Wk)
V = np.einsum('td,hed->hte', X, Wv)
S = np.einsum('hte,hse->hts', Q, K) / np.sqrt(d_h)   # [n_h, T, T]
S = S + np.triu(np.full((T, T), -np.inf), k=1)       # 因果掩码：j > t 的格子
P = np.exp(S - S.max(-1, keepdims=True))
P = P / P.sum(-1, keepdims=True)                     # 每行 softmax
O = np.einsum('hts,hse->hte', P, V)            # [n_h, T, d_h]
U = np.einsum('hte,dhe->td', O, Wo)            # [T, d]：拼接各头再乘 W^O
print(Q.shape, S.shape, O.shape, U.shape)
# (12, 1024, 64) (12, 1024, 1024) (12, 1024, 64) (1024, 768)
```

逐行读下标，形状和求和就都写在字符串里了。投影 `td,hed->hte`：`d` 不在输出里，被求和，这就是 $[T,d]\times[d,d_h]$ 在每个头上各做一次。打分 `hte,hse->hts`：`e` 被求和，`h` 两边都有、输出也有，所以它不求和，而是“逐头各算各的”，这类轴叫**批量轴**（batch axis）。最后一行 `hte,dhe->td` 同时对 `h` 和 `e` 求和，它把“拼接各头、再乘 $W^O$”一步做完，下一页会解释为什么可以这样。

实际代码里，“拆头”通常不用 einsum，而是 reshape 加 transpose。Hugging Face 的 GPT-2 先算出整块的查询（形状 $[B,T,d]$，$B$ 是一批里的序列数），再用 `view` 把最后一维拆成 $[n_h,d_h]$，然后 `transpose(1, 2)` 得到 $[B,n_h,T,d_h]$ [@hf-gpt2]。这样“头”成了和 $B$ 一样的批量轴，12 个头的 $[T,d_h]\times[d_h,T]$ 可以作为一次批量矩阵乘法一起算。

::: ex 第 200 列属于哪个头
GPT-2 small 的 $Q=H(W^Q)^\top$ 是 $[T,768]$。按 64 一段切成 12 个头，列号 $c$（从 0 数起）属于头 $\lfloor c/64\rfloor$，在头内的位置是 $c-64\lfloor c/64\rfloor$。第 200 列：$\lfloor200/64\rfloor=3$，$200-192=8$，即第 3 号头（从 0 数）的第 8 维。切头没有做任何计算，只是换了一种数下标的方式。
:::

:::: quiz 小练习：抓形状错误
$H\in\R^{T\times d}$ 按行存放 token，$W^K\in\R^{d_h\times d}$ 是论文写法的权重。下面三个写法各有什么问题？（a）$K=W^KH$；（b）$K=H W^K$；（c）打分写成 $S=Q^\top K$。
::: answer 看答案
（a）$[d_h,d]\times[T,d]$：中间维 $d\ne T$，乘不了。正确的是 $K=H(W^K)^\top$，或者等价地 $K^\top=W^KH^\top$。

（b）$[T,d]\times[d_h,d]$：中间维 $d\ne d_h$，乘不了。（若 $d_h=d$，它会“算得出来”，但用的是错方向的权重，见上面的警告框。）

（c）$Q^\top K$ 是 $[d_h,T]\times[T,d_h]\to[d_h,d_h]$，能算，形状却不是预期的 $[T,T]$。它对 token 求和，得到的是一个 $d_h\times d_h$ 的矩阵，和打分毫无关系。这就是为什么要检查**结果**的形状，而不只是检查“乘不乘得了”。
:::
::::

::: note 这一页用在哪里
[第 1 课 §7](../deepseek/lesson-01-transformer-anatomy.html#p7)的全流程形状追踪（从 token id 到 logits 的每一步）就是本页表格的延伸。[第 4 课 §3](../deepseek/lesson-04-inference-kv-cache.html#p3)：KV cache 里存的是每层所有头的 $K_i$、$V_i$，按头叠起来各是 $[n_h,T,d_h]$，它随 $T$ 线性增长。[第 9 课 §2](../deepseek/lesson-09-v2-mla.html#p2)：MLA 的每个公式都用 DeepSeek 论文的列写法，读代码时要用 (M1.11) 翻译。
:::

::: key 本页要点
- $(AB)^\top=B^\top A^\top$：转置把乘积的顺序倒过来，形状会逼你这样做。
- 论文 $\mathbf q_t=W^Q\mathbf h_t$（列）$\Leftrightarrow$ 代码 $Q=H(W^Q)^\top$（行）。不同代码库存权重的方向不同，方阵存反了不会报错。
- 注意力的形状：$[T,d]\to[T,d_h]$，$[T,d_h]\times[d_h,T]\to[T,T]$，$[T,T]\times[T,d_h]\to[T,d_h]$；只有 $[T,T]$ 那张表随 $T^2$ 增长。
- einsum：输出里没有的字母求和，有的保留；两边都有且输出也有的字母是批量轴。
:::

# 分块矩阵 | 多头注意力是一个分块结构 | 20

多头注意力通常被描述成一套流程：把查询、键、值“切成 $n_h$ 个头”，每个头各自做注意力，把结果“拼接”起来，再乘一个输出矩阵 $W^O$。读到这里自然会冒出三个问题：切头要花多少计算？“拼接再乘 $W^O$”为什么能像上一页那样写成一个对头求和的 einsum？多个小头比一个大头更贵吗？这三个问题都由同一个工具回答：**分块矩阵**（block matrix）。

## 分块乘法：把块当成数来乘

**图景。** 把一个大矩阵用横线、竖线切成若干小矩形，每个小矩形叫一个**块**。只要切法对得上，就可以把块当成“数”，按普通的行乘列规则相乘，只是“数的乘法”换成“块的矩阵乘法”。

**推导。** 先看最常用的情形：用一条竖线把 $A$ 切成左右两块 $A_1$、$A_2$，再用一条横线在**同样的位置**把 $B$ 切成上下两块 $B_1$、$B_2$。设 $A_1$ 有 $k_1$ 列，$B_1$ 就有 $k_1$ 行。

::: deriv 内侧切开 = 分段求和
$(AB)_{ij}=\sum_{k=1}^{K}A_{ik}B_{kj}$ || 矩阵乘法 (M1.5)，$K$ 是中间维
$=\sum_{k=1}^{k_1}A_{ik}B_{kj}+\sum_{k=k_1+1}^{K}A_{ik}B_{kj}$ || 把求和范围切成两段
$=(A_1B_1)_{ij}+(A_2B_2)_{ij}$ || 第一段只用到 $A_1$ 的列和 $B_1$ 的行，第二段同理
:::

::: eq 分块乘法 | (M1.12)
$$\begin{pmatrix}A_1&A_2\end{pmatrix}\begin{pmatrix}B_1\\B_2\end{pmatrix}=A_1B_1+A_2B_2,\qquad A\begin{pmatrix}B_1&B_2\end{pmatrix}=\begin{pmatrix}AB_1&AB_2\end{pmatrix}$$
左式：**内侧**（被求和的那一维）切开，结果是各块乘积之**和**。右式：**外侧**（保留的那一维）切开，结果是各块乘积**并排放**，互不相干。
:::

内侧的切法必须一致，否则块与块的形状对不上；外侧可以随便切。把两个矩阵都切成 $2\times2$ 块，就是两条规则一起用：乘积的左上块是 $A_{11}B_{11}+A_{12}B_{21}$，和 $2\times2$ 数字矩阵的公式一模一样，只是每个“数”换成了一块矩阵，乘法的先后次序不能交换。外积之和 (M1.9) 是把内侧切到每块只有一列的极端情形。

::: ex 用数字核对内侧切分
$$A=\begin{pmatrix}1&0&2&1\\0&1&1&3\end{pmatrix},\qquad B=\begin{pmatrix}1&2\\0&1\\1&0\\2&1\end{pmatrix}.$$

把中间维 4 切成 2+2：$A_1$ 是 $A$ 的前两列（恰好是单位矩阵），$A_2$ 是后两列；$B_1$ 是 $B$ 的前两行，$B_2$ 是后两行。$A_1B_1=B_1$，第一行 $(1,2)$，第二行 $(0,1)$；$A_2B_2$ 的第一行 $=(2\cdot1+1\cdot2,\ 2\cdot0+1\cdot1)=(4,1)$，第二行 $=(1\cdot1+3\cdot2,\ 1\cdot0+3\cdot1)=(7,3)$。相加得第一行 $(5,3)$、第二行 $(7,4)$。直接算：$A$ 第一行点 $B$ 第一列 $=1+0+2+2=5$，其余三格同理，一致。
:::

外侧切分在代码里很常见。GPT-2 并不分三次算 $Q$、$K$、$V$，而是把三个投影矩阵并排拼成一个 $[768,2304]$ 的大矩阵 `c_attn`，一次乘完再切开 [@hf-gpt2]。按 (M1.12) 右式，$H\begin{pmatrix}W_1&W_2&W_3\end{pmatrix}=\begin{pmatrix}HW_1&HW_2&HW_3\end{pmatrix}$，结果与分开算完全相同，运算次数也相同。合并的好处在工程上：一次大乘法通常比三次小乘法更容易把硬件用满。这一句是笔者推测（常见的工程经验，本课没有给出测量），具体快多少取决于硬件与矩阵大小。

## 拆头：按行分块，一次运算都不花

用 DeepSeek 论文的列写法，全部头的查询矩阵 $W^Q\in\R^{n_hd_h\times d}$ 可以按行切成 $n_h$ 块，每块 $W^Q_i\in\R^{d_h\times d}$ 属于一个头。这是外侧（输出维）的切分，所以

::: eq 拆头 | (M1.13)
$$\mathbf q_t=W^Q\mathbf h_t=\begin{pmatrix}W^Q_1\\W^Q_2\\\vdots\\W^Q_{n_h}\end{pmatrix}\mathbf h_t=\begin{pmatrix}W^Q_1\mathbf h_t\\W^Q_2\mathbf h_t\\\vdots\\W^Q_{n_h}\mathbf h_t\end{pmatrix}=\begin{pmatrix}\mathbf q_{t,1}\\\mathbf q_{t,2}\\\vdots\\\mathbf q_{t,n_h}\end{pmatrix}$$
第 $i$ 个头的查询 $\mathbf q_{t,i}\in\R^{d_h}$ 就是大向量 $\mathbf q_t$ 的第 $i$ 段。键、值同理。
:::

所以“切成多个头”不是一个计算步骤，只是换一种方式读同一个向量。上一页的 reshape 做的正是这件事。

## 合头：拼接再乘 $W^O$，等于各头分别乘再相加

每个头做完注意力，得到一个 $d_h$ 维的输出 $\mathbf o_{t,i}$（按 (M1.8)，是这个头的值向量的加权平均）。《Attention Is All You Need》把各头的输出拼接起来再乘 $W^O$ [@vaswani2017]。写成列形式：拼接向量是 $[\mathbf o_{t,1};\dots;\mathbf o_{t,n_h}]\in\R^{n_hd_h}$（分号表示上下叠放：$n_h$ 个 $d_h$ 维的列向量首尾相接，接成一根 $n_hd_h$ 维的长列向量），$W^O\in\R^{d\times n_hd_h}$。这次被切开的是 $W^O$ 的**列**，也就是内侧：

::: eq 合头 | (M1.14)
$$\mathbf u_t=W^O\begin{pmatrix}\mathbf o_{t,1}\\\vdots\\\mathbf o_{t,n_h}\end{pmatrix}=\begin{pmatrix}W^O_1&\cdots&W^O_{n_h}\end{pmatrix}\begin{pmatrix}\mathbf o_{t,1}\\\vdots\\\mathbf o_{t,n_h}\end{pmatrix}=\sum_{i=1}^{n_h}W^O_i\,\mathbf o_{t,i}$$
$W^O_i\in\R^{d\times d_h}$ 是 $W^O$ 的第 $i$ 个列块。$\mathbf u_t\in\R^{d}$ 是注意力层的输出，会加回残差流。
:::

(M1.14) 就是上一页 `hte,dhe->td` 对 `h` 求和的原因：拼接后乘 $W^O$，等于每个头用自己那一块 $W^O_i$ 把 $d_h$ 维的结果映射回 $d$ 维，然后 $n_h$ 份结果相加。第 1 课会从“残差流”的角度再看这件事：每个头通过自己的 $W^O_i$ 往同一条 $d$ 维的传送带上放一个增量，各头的增量叠加在一起。

::: fig figures/m01-multihead-blocks.svg | 图 M1.6 多头注意力的分块结构（GPT-2 small，按比例）
**看两处加深的条带：左边是 $W^Q$ 的第 3 个行块，右边是 $W^O$ 的第 3 个列块，它们都只和第 3 个头打交道。** 拆头（左）是外侧切分：每个行块独立地产生一段查询。合头（右）是内侧切分：每个列块乘上对应的一段拼接向量，再全部相加。两个大矩阵都是 $[768,768]$，分成 12 块，每块 64 行或 64 列 [@gpt2-config]。
:::

::: ex 合头的两种算法
取 $d=4$、$n_h=2$、$d_h=2$。$W^O$ 与两个头的输出为

$$W^O=\begin{pmatrix}1&0&0&1\\0&1&1&0\\1&1&0&0\\0&0&1&1\end{pmatrix},\qquad \mathbf o_{t,1}=(1,2),\quad \mathbf o_{t,2}=(3,-1).$$

拼接再乘，逐行点积：

$$W^O\begin{pmatrix}1\\2\\3\\-1\end{pmatrix}=\begin{pmatrix}1+0+0-1\\0+2+3+0\\1+2+0+0\\0+0+3-1\end{pmatrix}=\begin{pmatrix}0\\5\\3\\2\end{pmatrix}.$$

分别乘再相加：$W^O_1$ 是前两列，$W^O_1\mathbf o_{t,1}=1\cdot(1,0,1,0)+2\cdot(0,1,1,0)=(1,2,3,0)$；$W^O_2$ 是后两列，$W^O_2\mathbf o_{t,2}=3\cdot(0,1,0,1)-1\cdot(1,0,0,1)=(-1,3,0,2)$。相加得 $(0,5,3,2)$，一致。
:::

## 多个小头不比一个大头贵，但也不等于一个大头

**运算量相同。** 一个维度为 $d$ 的大头，打分要 $2T^2d$ 次运算（[§5](#p5) 会推导这个公式）；$n_h$ 个维度为 $d_h=d/n_h$ 的小头，每个 $2T^2d_h$，合计 $n_h\cdot2T^2d_h=2T^2d$，一样多。GPT-2 small 在 $T=1024$ 时两种算法都是 $1{,}610{,}612{,}736$ 次。原论文也指出，由于每个头的维度缩小了，多头的总计算代价与全维度的单头注意力相近 [@vaswani2017]。

**结果不同。** 用分块看点积：两个长向量的点积，等于把它们按同样方式分段后各段点积之和。

::: eq 分块点积 | (M1.15)
$$\mathbf q_t^\top\mathbf k_j=\sum_{i=1}^{n_h}\mathbf q_{t,i}^\top\mathbf k_{j,i}$$
这是 (M1.12) 左式在“一行乘一列”时的特例。
:::

一个大头只看这个总和，做一次 softmax；$n_h$ 个小头各自看自己那一项，各做一次 softmax，于是得到 $n_h$ 套不同的权重。区别用一个小例子最清楚。

::: ex 两个小头能同时看两个地方，一个大头不能
$d=4$，$n_h=2$，头 1 用前两维，头 2 用后两维。查询 $\mathbf q=(1,0,0,2)$，两个键 $\mathbf k_1=(2,1,0,0)$，$\mathbf k_2=(0,0,1,1)$。

- 头 1：$\mathbf q_1=(1,0)$，对两个键打分 $(1,0)\cdot(2,1)=2$，$(1,0)\cdot(0,0)=0$；softmax 后约 $(0.881,\ 0.119)$，主要看 token 1。
- 头 2：$\mathbf q_2=(0,2)$，打分 $0$ 与 $(0,2)\cdot(1,1)=2$；softmax 后约 $(0.119,\ 0.881)$，主要看 token 2。
- 一个大头：打分是两项之和 $2+0=2$、$0+2=2$，平局，softmax 后 $(0.5,\ 0.5)$，两个 token 的值被对半混在一起。

两个小头可以同时分别取回 token 1 和 token 2 的信息，再由 $W^O$ 分别放进输出的不同方向；一个大头只能得到一份模糊的平均。回到图书馆的类比：多头相当于同时派出 $n_h$ 个检索员，每人带着不同的问题、只看书脊标签的不同一栏。类比的边界在于：检索员之间并不商量，是 $W^O$ 事后把各人的笔记合成一份。
:::

::: fig figures/m01-two-heads.svg | 图 M1.7 两个小头各做各的 softmax，一个大头只能对半平均
**看底部三组柱子：头 1 把 0.881 的权重给 token 1，头 2 把 0.881 给 token 2；两段合成一个大头后，两个键的分数都是 2，权重打成 0.5 对 0.5。** 左上是上面例题的查询和两个键，按头切成两段；右上的表按 (M1.15) 把大头的分数写成两段点积之和。每一段单独看都有明确的赢家，相加之后两个键打平，“看谁”的信息就在求和里丢掉了。所有数都由脚本从例题的向量算出。
:::

::: note 往后看：GQA 与 MLA 也是分块结构
[第 4 课 §5](../deepseek/lesson-04-inference-kv-cache.html#p5)的 GQA 让几个查询头共用一块键和值：$W^K$ 只切成 $n_g<n_h$ 块，第 $i$ 个查询头用第 $\lceil i\,n_g/n_h\rceil$ 块。数学上仍是 (M1.13)，只是键的块比查询的块少，KV cache 也就小了 $n_h/n_g$ 倍。[第 9 课 §2](../deepseek/lesson-09-v2-mla.html#p2)的 MLA 更进一步：所有头的键都从同一个 $d_c$ 维的压缩向量展开，每个头只是一块 $W^{UK}_i$。
:::

:::: quiz 小练习：分块与运算量
（1）$d=4096$、$n_h=32$。每个头的 $d_h$ 是多少？$W^O$ 的每个列块是什么形状？（2）把头数改成 64、保持 $d$ 不变，打分的总运算量怎么变？
::: answer 看答案
（1）$d_h=4096/32=128$；$W^O\in\R^{4096\times4096}$，每个列块 $W^O_i\in\R^{4096\times128}$。（2）不变。总运算量 $n_h\cdot2T^2d_h=2T^2(n_hd_h)=2T^2d$ 只依赖 $n_hd_h=d$。变的是每个头的维度（变成 64）和 softmax 的套数（变成 64 套）。
:::
::::

::: note 这一页用在哪里
[第 1 课 §4](../deepseek/lesson-01-transformer-anatomy.html#p4)：多头注意力、$W^O$ 的作用、残差流上的增量叠加。[第 4 课 §5](../deepseek/lesson-04-inference-kv-cache.html#p5)：MQA/GQA 就是改变键、值的分块方式。[第 9 课 §3](../deepseek/lesson-09-v2-mla.html#p3)：MLA 的 $W^{UK}$、$W^{UV}$ 按头分块，吸收也是逐块进行的（本课 [§6](#p6)）。
:::

::: key 本页要点
- 分块乘法：内侧切开 → 各块乘积相加；外侧切开 → 各块乘积并排。内侧切法必须一致。
- 拆头 = 按行（外侧）分块，不花任何运算；合头 = $W^O$ 按列（内侧）分块，拼接后乘 $W^O$ 等于 $\sum_iW^O_i\mathbf o_{t,i}$。
- 多头与单头的打分运算量都是 $2T^2d$；区别在于多头有 $n_h$ 套独立的 softmax 权重，能同时看不同的地方。
:::

# 数 FLOPs | 矩阵乘法为什么是 2mnk，一层注意力的算力花在哪 | 20

第 4 课要回答“缓存 K、V 省了多少计算”，第 9 课要回答“MLA 的矩阵吸收是省了还是亏了”，后面的课还要比较各种注意力变体的成本。这些问题都要先会**数**：一个公式要做多少次运算。好在注意力里几乎所有运算都是矩阵乘法，而矩阵乘法的运算次数可以直接数出来。

## 数出 $2mnk$

**含义。** 一次**浮点运算**（floating-point operation，FLOP）指两个浮点数之间的一次加法或一次乘法。数一个算法做了多少次，得到的是它的 FLOPs（运算次数）。硬件每秒能做多少次，写作 FLOP/s，是速度。两者常被混写成 “FLOPS”，本课程统一用 FLOPs 表示次数，FLOP/s 表示速度。

**推导。** 对 $C=AB$，$A$ 是 $[m,k]$，$B$ 是 $[k,n]$：

::: deriv 矩阵乘法的运算次数
$C_{ij}=A_{i1}B_{1j}+A_{i2}B_{2j}+\cdots+A_{ik}B_{kj}$ || (M1.5)：一格是一个长度为 $k$ 的点积
$k\text{ 次乘法}+(k-1)\text{ 次加法}=2k-1$ || $k$ 个乘积，相加需要 $k-1$ 个加号
$mn\,(2k-1)\approx2mnk$ || $C$ 共有 $mn$ 格；$k$ 较大时 $-1$ 可以忽略
:::

::: eq 矩阵乘法的 FLOPs | (M1.16)
$$\mathrm{FLOPs}\big([m,k]\times[k,n]\big)\approx2mnk$$
等价的记法：“每一对被相乘的元素做一次乘法、一次加法”。
:::

业界通常用 $2mnk$ 而不是精确的 $mn(2k-1)$，一个方便的理由是硬件的基本指令是**乘加**（fused multiply-add，FMA）：一条指令完成 $a\times b+c$，记作 2 次运算。从 0 开始累加 $k$ 个乘积恰好是 $k$ 次乘加，也就是 $2k$ 次运算。有些工具报告的是 MACs（乘加次数），数值是 FLOPs 的一半，读别人的数字时先看清单位，不然会差出一个 2 倍。

::: ex 用 §2 的例子核对
$[2,3]\times[3,2]$：$m=2,k=3,n=2$。每格 3 次乘法、2 次加法，共 $4\times5=20$ 次；按约定 $2mnk=24$。两者之差正是每格省掉的那一次“加到 0 上”。

einsum 的写法给出一个方便的规则：两个张量做 einsum、且至少有一个字母被求和时，**FLOPs $\approx2\times$ 所有出现过的字母的大小之积**，因为每一种字母取值的组合恰好对应一次乘加。（没有字母被求和时，例如外积 `i,j->ij`，每个组合只有一次乘法，FLOPs 就是大小之积本身。）`ik,kj->ij` 是 $2mkn$；§3 的打分 `hte,hse->hts` 是 $2\cdot n_h\cdot T\cdot d_h\cdot T=2T^2d$（当 $n_hd_h=d$）。
:::

## 线性层：每个参数、每个 token，两次运算

一个 token 经过一个权重矩阵 $W\in\R^{m\times k}$：$\mathbf y=W\mathbf x$ 是 $[m,k]\times[k,1]$，按 (M1.16) 是 $2mk$ 次运算，而 $mk$ 正是 $W$ 的参数个数。所以**每个参数对每个 token 贡献 2 次运算**：一次乘法，一次加法。整段 $T$ 个 token 就是 $2\times(\text{参数数})\times T$。[第 1 课 §8](../deepseek/lesson-01-transformer-anatomy.html#p8) 的“每个 token 前向约 $2N$ 次运算”（$N$ 是不含嵌入表的参数量，第 1 课取 $N\approx12Ld^2$；查嵌入表只是取一行，不做乘法）就是把这件事加总到全模型；它忽略了注意力里不含参数的那部分运算，也就是下面的 $4T^2d$ 项。

::: ex GPT-2 small 的一个投影矩阵
全部头的 $W^Q$ 是 $[768,768]$，共 $589{,}824$ 个参数 [@gpt2-config]。一个 token 过这个矩阵要 $2\times589{,}824=1{,}179{,}648$ 次运算，约 $1.18\times10^6$。$T=1024$ 个 token 一起算就是 $[1024,768]\times[768,768]$，$2\times1024\times768\times768\approx1.21\times10^9$ 次。
:::

## 一层注意力的算力账单

把 §3、§4 的各步形状代进 (M1.16)。设 $n_hd_h=d$（GPT-2 就是这样），序列长 $T$，先不考虑因果掩码能省下的部分：

| 步骤 | 形状 | FLOPs |
|---|---|---|
| $Q,K,V$ 三个投影 | $3\times\big([T,d]\times[d,d]\big)$ | $6Td^2$ |
| 打分 $Q_iK_i^\top$，全部 $n_h$ 个头 | $n_h\times\big([T,d_h]\times[d_h,T]\big)$ | $2T^2d$ |
| 加权求和 $P_iV_i$，全部头 | $n_h\times\big([T,T]\times[T,d_h]\big)$ | $2T^2d$ |
| 输出投影 $W^O$ | $[T,d]\times[d,d]$ | $2Td^2$ |
| **合计** | | $8Td^2+4T^2d$ |

softmax 和除以 $\sqrt{d_h}$ 对每个分数只做几次运算，而每个分数背后有 $4d_h$ 次矩阵乘法运算（打分 $2d_h$、加权求和 $2d_h$），$d_h=64$ 时是 256 次，所以前者可以忽略。

::: eq 一层注意力的 FLOPs | (M1.17)
$$\mathrm{FLOPs}_{\text{attn}}(T)\approx\underbrace{8Td^2}_{\text{投影，随 }T\text{ 线性}}+\underbrace{4T^2d}_{\text{打分与加权求和，随 }T^2}$$
两项相等时 $8Td^2=4T^2d$，即 $T=2d$。$T<2d$ 时投影占大头，$T>2d$ 时 $T^2$ 项占大头。
:::

::: fig figures/m01-attention-flops.svg | 图 M1.8 一层注意力的 FLOPs 随序列长度变化（d = 768）
**看蓝线和紫线的交点：$T=2d=1536$。** 左边投影（蓝，斜率 1）占多数，右边打分与加权求和（紫，斜率 2）占多数，而且差距随 $T$ 线性拉大；黑色虚线是两项之和。两条坐标轴都是对数刻度，所以 $T$ 的一次方、二次方都画成直线，斜率（按两轴各自的数量级计）就是指数；两轴每个数量级画得不一样长，用尺子在画面上量出的斜率不是 1 和 2，但紫线的斜率仍是蓝线的 2 倍。GPT-2 的上下文长度 1024 落在交点左侧（竖虚线上的两个圆点，数值写在左上角）：$T=1024$ 时投影 $4.83\times10^9$，打分与加权求和 $3.22\times10^9$，比值 $2d/T=1.5$。曲线由 (M1.17) 逐点计算。
:::

::: ex 手算一个更大的配置
$d=4096$，$T=2048$。投影：$8\times2048\times4096^2=274{,}877{,}906{,}944\approx2.75\times10^{11}$。打分与加权求和：$4\times2048^2\times4096=68{,}719{,}476{,}736\approx6.87\times10^{10}$。投影是后者的 4 倍，正好等于 $2d/T=8192/2048$。上下文要到 $T=8192$，两项才持平。
:::

**解码时的账单**（第 4 课预告）。生成第 $T+1$ 个 token 时，有了 KV cache，只需要为这一个新 token 算：投影是 $[1,d]\times[d,d]$ 四次，共 $8d^2$；它的查询和缓存里 $T$ 个键打分、再对 $T$ 个值加权求和，共 $4Td$。两项之比仍是 $2d/T$，交点仍在 $T=2d$。以 GPT-2 small、$T=1024$ 为例：$8d^2=4{,}718{,}592$，$4Td=3{,}145{,}728$。

::: warn 失败模式：FLOPs 不等于时间
**因果掩码。** 每一行只需要 $j\le t$ 的约一半分数。若实现跳过被掩掉的块，$4T^2d$ 约减半、交点移到约 $T=4d$；若先算满整张表再掩掉（简单实现常这样做），就没有省。数别人的 FLOPs 时要先问清是哪一种。

**搬数据也要时间。** 一次矩阵乘法要读 $mk+kn$ 个数、写 $mn$ 个数。运算次数与搬运数量之比叫**算术强度**（arithmetic intensity）。方阵 $m=n=k=4096$：$2\cdot4096^3/(3\cdot4096^2)\approx2731$ 次运算每个数；矩阵乘向量（$n=1$）只有约 $2$ 次运算每个数，时间几乎全花在从显存读权重上。解码每步只有一个新 token，恰好是后一种情形，所以第 4 课会说解码受**带宽**限制而不是算力限制（{{屋顶线模型::roofline model：用算力上限和带宽上限两条线，判断一段计算受哪一个限制}}见 [M10 §4](m10-compute-arithmetic.html#p4)）。
:::

:::: quiz 小练习：数 FLOPs
（1）$[512,1024]\times[1024,256]$ 需要多少 FLOPs？（2）GPT-2 small 的注意力层在 $T=512$ 时，投影与打分加权求和各多少？哪个大？
::: answer 看答案
（1）$2\times512\times256\times1024=268{,}435{,}456\approx2.68\times10^8$。

（2）投影 $8\times512\times768^2=2{,}415{,}919{,}104\approx2.42\times10^9$；打分与加权求和 $4\times512^2\times768=805{,}306{,}368\approx8.05\times10^8$。投影大，是后者的 $2d/T=1536/512=3$ 倍。
:::
::::

::: note 这一页用在哪里
[第 1 课 §8](../deepseek/lesson-01-transformer-anatomy.html#p8)：每 token 前向约 $2N$ 次运算。第 2 课的训练算力 $\approx6ND$ 也从这里出发（前向 $2N$，反向约 $4N$）。[第 4 课 §3–§4](../deepseek/lesson-04-inference-kv-cache.html#p3)：缓存与重算的 FLOPs 对比、解码的带宽墙。[M10 §3–§4](m10-compute-arithmetic.html#p3)：算术强度与屋顶线模型的完整讲法。
:::

::: key 本页要点
- $[m,k]\times[k,n]$ 需要约 $2mnk$ 次运算：$mn$ 格，每格 $k$ 次乘加。einsum 的 FLOPs 约为 2 乘以所有字母大小之积。
- 线性层：每个参数、每个 token 2 次运算，于是全模型约 $2N$ 每 token。
- 一层注意力约 $8Td^2+4T^2d$，交点 $T=2d$。FLOPs 只是账单的一半，另一半是搬数据，它决定了解码为什么慢。
:::

# 投影与结合律 | 查询矩阵在做什么，以及 MLA 为什么能“吸收”矩阵 | 25

这一页回答两个问题。第一，§2 说 $W^Q_i$ 从 768 个栏目里“读出 64 个组合”，它到底保留了什么、丢掉了什么？答案是**投影到子空间**，至少在理想化的情形下是这样。第二，第 9 课的 MLA 为什么只缓存一个短向量，却能在推理时不把它展开成键？答案是矩阵乘法的**结合律**：同一个乘积，换一个括号的位置，算法和成本都变了，结果却一点不变。

## 投影到一条直线

**问题。** 给一个向量 $\mathbf h$ 和一个方向 $\mathbf u$，$\mathbf h$ 里有多少“沿着 $\mathbf u$”的成分？（本页的 $\mathbf u$、$\mathbf u_1,\mathbf u_2,\dots$ 和由它们排成的矩阵 $U$ 都表示方向，与 §4 的注意力层输出 $\mathbf u_t$ 无关。）图 M1.1 的“影子”已经给出了直觉，现在把它写成公式。

**推导。** 设投影是 $\mathbf p=\alpha\mathbf u$。“投影”的意思是剩下的部分 $\mathbf h-\alpha\mathbf u$ 与 $\mathbf u$ 垂直，也就是点积为 0：

::: deriv 投影到直线
$\mathbf u^\top(\mathbf h-\alpha\mathbf u)=0$ || 残差与方向正交：这是“投影”的定义
$\mathbf u^\top\mathbf h-\alpha\,\mathbf u^\top\mathbf u=0$ || 点积对每个参数线性
$\alpha=\dfrac{\mathbf u^\top\mathbf h}{\mathbf u^\top\mathbf u}$ || 解出 $\alpha$（$\mathbf u\ne\mathbf 0$）
$\mathbf p=\mathbf u\,(\mathbf u^\top\mathbf h)=(\mathbf u\mathbf u^\top)\,\mathbf h$ || 若 $\norm{\mathbf u}=1$，分母为 1；最后一步用了结合律（本页后半）
:::

::: eq 投影到单位方向 u | (M1.18)
$$\mathbf p=(\mathbf u^\top\mathbf h)\,\mathbf u=\mathbf u\mathbf u^\top\mathbf h,\qquad \norm{\mathbf u}=1$$
$\mathbf u^\top\mathbf h$ 是一个数（沿 $\mathbf u$ 的坐标），$\mathbf u\mathbf u^\top$ 是一个 $[d,d]$ 的外积矩阵（投影矩阵）。
:::

## 投影到子空间

**图景。** 一个**子空间**（subspace）是过原点的直线、平面，或者更高维的“平面”。取它的 $k$ 个**正交归一**（orthonormal）方向 $\mathbf u_1,\dots,\mathbf u_k$ 作坐标轴，排成矩阵 $U=[\mathbf u_1\ \cdots\ \mathbf u_k]\in\R^{d\times k}$。投影就是：分别读出 $\mathbf h$ 在每根轴上的坐标，再按这些坐标把轴加回去。

::: pre 正交归一与 $U^\top U=I$ | 知道正交矩阵的可以跳过
一组向量**正交归一**，是指每个长度为 1、两两互相垂直：$\mathbf u_a^\top\mathbf u_b=1$（$a=b$）或 $0$（$a\ne b$）。把它们排成 $U$ 的列，$U^\top U$ 的第 $(a,b)$ 格正是 $\mathbf u_a^\top\mathbf u_b$（读法二），所以“正交归一”就等于一句话：$U^\top U=I$（$k\times k$ 单位矩阵）。注意 $UU^\top$ 一般**不是**单位矩阵，它是下面的投影矩阵。
:::

::: deriv 投影到子空间
$\mathbf p=U\boldsymbol\alpha$ || 投影落在子空间里，所以是各轴的组合（读法三）；坐标 $\boldsymbol\alpha\in\R^k$ 是直线情形里那个 $\alpha$ 的 $k$ 维版本
$U^\top(\mathbf h-U\boldsymbol\alpha)=\mathbf 0$ || 残差与每一根轴都正交：$k$ 个点积同时为 0（读法二）
$U^\top\mathbf h-U^\top U\boldsymbol\alpha=\mathbf 0$ || 乘法对加减法可分配
$\boldsymbol\alpha=U^\top\mathbf h$ || $U^\top U=I$
:::

::: eq 投影到子空间 | (M1.19)
$$\boldsymbol\alpha=U^\top\mathbf h\in\R^{k}\ \ (\text{坐标}),\qquad \mathbf p=UU^\top\mathbf h\ \ (\text{投影}),\qquad \mathbf r=\mathbf h-\mathbf p\perp\text{子空间}$$
$\mathbf p$ 在子空间里、$\mathbf r$ 与子空间垂直，所以 $\mathbf p^\top\mathbf r=0$，勾股定理给出 $\norm{\mathbf h}^2=\norm{\mathbf p}^2+\norm{\mathbf r}^2$：投影永远不会比原向量长。
:::

::: ex 三维里的一次投影
$\mathbf h=(2,-1,3)$，子空间由 $\mathbf u_1=(1,0,0)$ 和 $\mathbf u_2=(0,1,1)/\sqrt2$ 张成（一个倾斜 $45^\circ$ 的“坡面”）。先核对正交归一：$\norm{\mathbf u_2}^2=(0+1+1)/2=1$，$\mathbf u_1^\top\mathbf u_2=0$。

坐标：$\boldsymbol\alpha=U^\top\mathbf h=(\mathbf u_1^\top\mathbf h,\ \mathbf u_2^\top\mathbf h)=(2,\ (-1+3)/\sqrt2)=(2,\ \sqrt2)$。

投影：$\mathbf p=2\,\mathbf u_1+\sqrt2\,\mathbf u_2=(2,0,0)+(0,1,1)=(2,1,1)$。残差：$\mathbf r=\mathbf h-\mathbf p=(0,-2,2)$。核对：$\mathbf r^\top\mathbf u_1=0$，$\mathbf r^\top\mathbf u_2=(-2+2)/\sqrt2=0$；$\norm{\mathbf h}^2=14=\norm{\mathbf p}^2+\norm{\mathbf r}^2=6+8$。
:::

::: fig figures/m01-projection.svg | 图 M1.9 投影：保留子空间里的部分，丢掉垂直于它的部分
**看红色虚线：它是被丢掉的残差 $\mathbf r$，垂直于蓝色的平面。** 蓝色箭头 $\mathbf p$ 是 $\mathbf h$ 在平面上的投影，它在平面里的坐标 $U^\top\mathbf h=(2,\sqrt2)$ 就是“只看这个平面的人”能读到的全部信息。换任何一个 $\mathbf h+\lambda\mathbf r$（沿红线滑动），读到的坐标都不变。
:::

## 这和 $W^Q$ 有什么关系

假设第 $i$ 个头的 $W^Q_i\in\R^{d_h\times d}$ 的各**行**恰好正交归一，也就是 $W^Q_i=U^\top$，$U\in\R^{d\times d_h}$ 满足 $U^\top U=I$。那么 $\mathbf q_{t,i}=W^Q_i\mathbf h_t=U^\top\mathbf h_t$ 恰好是 (M1.19) 里的坐标：这个头只读到 $\mathbf h_t$ 在一个 $d_h$ 维子空间里的部分，与这个子空间垂直的一切，它的查询都看不见。上例中若 $W^Q_i=U^\top$，查询就是 $(2,\sqrt2)$，而 $\mathbf r=(0,-2,2)$ 满足 $W^Q_i\mathbf r=\mathbf 0$。

打分时，这个头比较的是两个 token 各自经过一次“取景”之后的样子：

::: eq 一个头的打分是一个夹在中间的矩阵 | (M1.20)
$$\mathbf q_{t,i}^\top\mathbf k_{j,i}=(W^Q_i\mathbf h_t)^\top(W^K_i\mathbf h_j)=\mathbf h_t^\top\,\underbrace{(W^Q_i)^\top W^K_i}_{[d,d]}\,\mathbf h_j$$
第二个等号用 (M1.10) 和本页后半的结合律 (M1.21)。中间的 $[d,d]$ 矩阵是两块 $[d,d_h]$、$[d_h,d]$ 的乘积，按 (M1.9) 至多是 $d_h$ 个秩 1 矩阵之和。
:::

GPT-2 small 每个头 $d_h=64$，而 $d=768$ [@gpt2-config]：每个头只通过一个 64 维的窄窗口比较两个 768 维的隐藏状态。不同的头有不同的窗口，这是 §4 里“多个小头能同时看不同地方”的几何说法。

::: warn 类比的边界：学出来的 $W^Q_i$ 不是正交投影
训练出来的 $W^Q_i$ 的行一般既不等长、也不两两垂直，所以它不是纯粹的投影，而是“投影到一个 $d_h$ 维子空间，再在子空间里做拉伸和旋转”。[M02 §3](m02-rank-svd-lowrank.html#p3) 的{{奇异值分解::把任意矩阵写成“旋转、沿坐标轴拉伸、再旋转”三步的分解，M02 详讲}}（SVD）会把这句话说准确：任何矩阵都是“旋转、按轴拉伸、再旋转”（严格地说，“旋转”是正交变换，可能带一次镜面反射）。直觉里真正精确的部分是“看不见”：满足 $W^Q_i\mathbf x=\mathbf 0$ 的方向 $\mathbf x$ 组成一个至少 $d-d_h$ 维的子空间，叫它的**零空间**（null space）。理由是线性代数的秩–零化度定理：列数 $d$ = 秩 + 零空间的维数，而一个 $[d_h,d]$ 矩阵的秩至多是 $d_h$（秩的定义见 [M02 §1](m02-rank-svd-lowrank.html#p1)），GPT-2 small 里至少 704 维，这个头的查询对这些方向完全无感。另一个差别是长度：拉伸会改变查询的长度，而 §1 说过，长度决定 softmax 的尖锐程度。
:::

## 结合律：同一个乘积，不同的括号

**性质。** 三个矩阵相乘，先乘哪两个都一样：

::: deriv 结合律
$\big((AB)C\big)_{il}=\sum_j(AB)_{ij}C_{jl}=\sum_j\Big(\sum_kA_{ik}B_{kj}\Big)C_{jl}$ || 两次使用 (M1.5)
$=\sum_k\sum_jA_{ik}B_{kj}C_{jl}$ || 展开成对 $k$、$j$ 的双重求和
$=\sum_kA_{ik}\Big(\sum_jB_{kj}C_{jl}\Big)=\big(A(BC)\big)_{il}$ || 交换两个有限求和的次序，再把 $A_{ik}$ 提出来
:::

::: eq 结合律 | (M1.21)
$$(AB)C=A(BC)$$
用读法一说：两边都是“先做 $C$，再做 $B$，最后做 $A$”。用 einsum 说：两边都是 `ik,kj,jl->il`，括号只决定先对哪个字母求和。
:::

结果相同，**成本却可以差很多**。设 $A$、$B$、$C$ 的形状是 $[m,k]$、$[k,n]$、$[n,p]$，按 (M1.16)：

$$\text{先算 }AB:\ 2mkn+2mnp,\qquad \text{先算 }BC:\ 2knp+2mkp.$$

::: ex 换个括号，快 750 倍
$A$ 是 $[1000,2]$，$B$ 是 $[2,1000]$，$C$ 是 $[1000,1]$。

- $(AB)C$：先造出 $[1000,1000]$ 的 $AB$，要 $2\cdot1000\cdot2\cdot1000=4\times10^6$ 次；再乘 $C$，$2\cdot1000\cdot1000\cdot1=2\times10^6$ 次。合计 $6\times10^6$。
- $A(BC)$：$BC$ 是 $[2,1]$，只要 $2\cdot2\cdot1000\cdot1=4{,}000$ 次；再算 $A(BC)$，$2\cdot1000\cdot2\cdot1=4{,}000$ 次。合计 $8{,}000$。

相差 750 倍。$AB$ 是一个“中间维只有 2”的 $[1000,1000]$ 矩阵（(M1.9)：至多两个秩 1 矩阵之和），永远不把它显式造出来，就是低秩方法省钱的全部秘密。
:::

反过来，有时**先**把两个矩阵乘起来反而划算：如果同一个乘积 $AB$ 要作用在成千上万个不同的向量上，那就预先算好 $AB$ 一次，以后每次只乘一个矩阵。一个行向量乘两个 $[768,768]$ 方阵：$(\mathbf x^\top A)B$ 要 $4d^2=2{,}359{,}296$ 次，$\mathbf x^\top(AB)$ 现算要约 $9.07\times10^8$ 次，但若 $AB$ 已经预先算好，只剩 $2d^2$ 次。**把一个固定的矩阵乘积提前算好**，这就是“吸收”这个词的字面意思。它只在合并后的矩阵不比两块加起来更大时才省：$A\in\R^{m\times k}$、$B\in\R^{k\times n}$ 合并后，每个向量要 $2mn$ 次，分两步要 $2k(m+n)$ 次，所以要 $mn<k(m+n)$ 才划算。方阵（$m=n=k$）满足这个条件；上面快 750 倍的例子（$m=n=1000$，$k=2$）正好相反，合并是亏的。[M02 §7](m02-rank-svd-lowrank.html#p7) 会把这条规则写成“临界秩”。

## 矩阵吸收：第 9 课的 MLA 为什么不必展开缓存

**问题。** MLA 为每个 token 缓存的主体是一个压缩向量 $\mathbf c_j^{KV}\in\R^{d_c}$（下面简写成 $\mathbf c_j$），键和值都不再逐头存放。每个头的键由它展开：$\mathbf k_{j,i}=W^{UK}_i\mathbf c_j$，其中 $W^{UK}_i\in\R^{d_h\times d_c}$ 是上投影矩阵 $W^{UK}\in\R^{d_hn_h\times d_c}$ 的第 $i$ 个行块（§4 的拆头）[@deepseekv2]。如果每生成一个 token 都要把缓存里的 $T$ 个 $\mathbf c_j$ 全部展开成键，压缩省下的内存就要用大量计算来还。下面暂时忽略 RoPE 那一小部分：V2 每个 token 还要另外缓存一个所有头共用的 $d_h^R=64$ 维 RoPE 键，所以每层每 token 实际缓存 $d_c+d_h^R=576$ 个数 [@deepseekv2]，第 9 课会单独处理它。

**推导。** 换一个括号：

::: deriv 把 $W^{UK}_i$ 从键那边挪到查询这边
$\mathbf q_{t,i}^\top\mathbf k_{j,i}=\mathbf q_{t,i}^\top\big(W^{UK}_i\mathbf c_j\big)$ || 键的定义：先展开，再点积
$=\big(\mathbf q_{t,i}^\top W^{UK}_i\big)\,\mathbf c_j$ || 结合律 (M1.21)
$=\big((W^{UK}_i)^\top\mathbf q_{t,i}\big)^\top\mathbf c_j$ || 转置规则 (M1.10)：$(\mathbf q^\top W)=(W^\top\mathbf q)^\top$
$=\tilde{\mathbf q}_{t,i}^\top\,\mathbf c_j,\qquad \tilde{\mathbf q}_{t,i}:=(W^{UK}_i)^\top\mathbf q_{t,i}\in\R^{d_c}$ || 定义“吸收后的查询”，它直接和缓存里的 $\mathbf c_j$ 做点积
:::

::: eq 矩阵吸收 | (M1.22)
$$\mathbf q_{t,i}^\top\big(W^{UK}_i\mathbf c_j\big)=\big((W^{UK}_i)^\top\mathbf q_{t,i}\big)^\top\mathbf c_j$$
左边：先把缓存展开成键。右边：先把查询变成 $d_c$ 维，缓存原样使用。两边在实数运算下完全相等。
:::

再往前一步：查询本身也是某个矩阵乘出来的，比如 $\mathbf q_{t,i}=W^Q_i\mathbf h_t$。于是 $\tilde{\mathbf q}_{t,i}=\big[(W^{UK}_i)^\top W^Q_i\big]\mathbf h_t$，方括号里是两个**固定的**权重矩阵的乘积，形状 $[d_c,d_h]\times[d_h,d]\to[d_c,d]$，加载模型时算一次就够。不过“能提前算好”不等于“值得提前算好”：这里 $m=d_c$、$k=d_h$、$n=d$，代入上面的条件 $mn<k(m+n)$，V2 的 $d_h=128$、$d_c=512$、$d=5120$ 不满足。合并后每个头、每个 token 要 $2d_cd=5{,}242{,}880$ 次运算，先乘 $W^Q_i$、再乘 $(W^{UK}_i)^\top$ 只要 $2d_hd+2d_cd_h=1{,}441{,}792$ 次，合并反而贵约 3.6 倍。所以吸收真正省下的，是 (M1.22) 换括号这一步：$W^{UK}_i$ 从 $T$ 个缓存向量那一侧挪到了一个查询这一侧。两块权重要不要合并，按形状另算（[M02 §7](m02-rank-svd-lowrank.html#p7)）。DeepSeek-V2 论文说“推理时 $W^{UK}$ 可以被吸收进 $W^Q$、$W^{UV}$ 可以被吸收进 $W^O$，因此甚至不需要把键和值真正算出来” [@deepseekv2]，指的就是这件事（V2 的查询其实先压成 $d_c'=1536$ 维的 $\mathbf c^Q_t$，再由 $W^{UQ}$ 展开 [@deepseekv2]，道理相同：合并成 $(W^{UK}_i)^\top W^{UQ}_i$ 后每头每 token 要 $2d_cd_c'=1{,}572{,}864$ 次，分开乘只要 $2d_hd_c'+2d_cd_h=524{,}288$ 次，仍是分开更省；值那一侧的吸收留作练习 9）。

::: fig figures/m01-absorption.svg | 图 M1.10 同一个分数的两种括号：先展开键（Ⓐ），或先把键的上投影吸收进查询（Ⓑ）
**看底部两根柱子：同样得到 $T$ 个分数，Ⓑ 的运算量约是 Ⓐ 的 1/124。** Ⓐ 每生成一个 token 都要把缓存里的 $T$ 个 $\mathbf c_j$ 乘上 $(W^{UK}_i)^\top$；Ⓑ 只把一个查询乘一次，然后直接和缓存点积。$d_h=128$、$d_c=512$ 取自 DeepSeek-V2 [@deepseekv2]，$T=4096$ 是为算账取的例子。方块中 $d_c$、$d_h$ 两个方向按比例画，$T$ 方向是示意。
:::

::: ex 吸收前后的账单（每个头、每生成一个 token）
取 DeepSeek-V2 的 $d_h=128$、$d_c=512$ [@deepseekv2]，缓存里有 $T$ 个 token。

- Ⓐ 先展开：$[T,d_c]\times[d_c,d_h]$ 得到键，$2Td_cd_h$；再与查询点积，$2Td_h$。
- Ⓑ 先吸收：$[d_c,d_h]\times[d_h]$ 得到 $\tilde{\mathbf q}$，$2d_hd_c$；再与 $T$ 个 $\mathbf c_j$ 点积，$2Td_c$。

$T=4096$：Ⓐ $=536{,}870{,}912+1{,}048{,}576=537{,}919{,}488$；Ⓑ $=131{,}072+4{,}194{,}304=4{,}325{,}376$，约为 Ⓐ 的 $1/124$。$T$ 越大，比值 Ⓐ/Ⓑ 越接近 $d_h(1+1/d_c)=128.25$（分子分母同除以 $2T$ 再令 $T\to\infty$：$(d_cd_h+d_h)/d_c$）。

$T=1$：Ⓐ $=131{,}072+256=131{,}328$；Ⓑ $=131{,}072+1{,}024=132{,}096$，吸收反而略贵。吸收的收益来自“查询只变一次，缓存被反复使用”，缓存越长越划算。
:::

::: warn 失败模式：吸收不是免费的，也不总是可行
**每个分数变贵了。** 吸收后每个分数是一个 $d_c=512$ 维的点积，而如果键已经展开并存好，每个分数只要 $d_h=128$ 维。吸收用 4 倍的单次点积换来了“不用存、也不用现算展开后的键”。只算键的话，展开后每个 token 每层要存 $n_hd_h=128\times128=16{,}384$ 个数，而 $\mathbf c_j$ 只有 512 个 [@deepseekv2]。哪种做法更快取决于硬件与 $T$，第 9 课会比较 MLA 的两种计算模式。

**夹在中间的东西必须是固定的。** 如果 $\mathbf q$ 与 $W^{UK}\mathbf c_j$ 之间还夹着一个随位置 $(t,j)$ 变化的矩阵 $R$（RoPE 的旋转），分数变成 $\mathbf q^\top R\,W^{UK}\mathbf c_j$。结合律仍然成立，但能提前算好的只有 $R\,W^{UK}$，而它对每一对位置都不同，吸收就失去意义。DeepSeek-V2 正是以此解释 RoPE 与低秩压缩不兼容：旋转矩阵夹在 $W^Q$ 与 $W^{UK}$ 之间，而矩阵乘法不满足交换律，无法把它挪开 [@deepseekv2]。第 9 课的“解耦 RoPE”就是为此设计的。

**浮点数不严格满足结合律。** 实数运算下 (M1.22) 两边完全相等；计算机里的浮点加法有舍入误差，换一种求和顺序，末几位会不同。笔者用 NumPy 双精度按上面的维度（$d_h=128$、$d_c=512$、$T=4096$；$\mathbf q_{t,i}$ 与各 $\mathbf c_j$ 取标准正态，$W^{UK}_i$ 的元素取标准正态再除以 $\sqrt{d_c}$）试过：两种括号算出的分数最多差约 $4\times10^{-14}$，而分数本身是几十的量级，相对差约 $10^{-15}$，正好是双精度的舍入量级。低精度（BF16、FP8）下差别会大得多（见 [M08 §5](m08-floating-point.html#p5)）。所以吸收是数学上的恒等变形，实现上是“几乎相等”。
:::

## 总账：本课的每个工具用在哪里

| 工具 | 本课公式 | 用在哪里 |
|---|---|---|
| 点积、余弦、长度 | M1.1–4 | 第 1 课的缩放点积注意力与 $\sqrt{d_k}$；第 4 课解码时新查询与缓存键的打分 |
| 矩阵乘法的几种读法 | M1.5–9 | 第 1 课的投影、打分 $QK^\top$、加权求和 $PV$；外积之和通向 M02 的低秩分解 |
| 转置、两种写法、einsum | M1.10–11 | 第 1 课的形状追踪；读 MLA 等模型的代码时在论文写法与代码写法间翻译 |
| 分块矩阵 | M1.12–15 | 第 1 课的多头与 $W^O$；第 4 课的 MQA/GQA；第 9 课按头分块的 $W^{UK}$、$W^{UV}$ |
| FLOPs 计数 | M1.16–17 | 第 1 课的 $2N$；第 2 课的 $6ND$；第 4 课的缓存与重算、带宽墙；M10 |
| 投影 | M1.18–20 | 第 1 课理解 $W^Q$、$W^K$ 各头的“窗口”；M02 的 SVD |
| 结合律与吸收 | M1.21–22 | 第 9 课 MLA 的矩阵吸收与解耦 RoPE；M02 的“结合律与吸收技巧” |

::: key 本页要点
- 投影到正交归一的子空间：坐标 $U^\top\mathbf h$，投影 $UU^\top\mathbf h$，残差与子空间垂直。一个头的 $W^Q_i$ 可以看成“投影到 $d_h$ 维子空间再拉伸”，零空间里的方向它完全看不见。
- 结合律 $(AB)C=A(BC)$：结果相同，成本可以差几个数量级。不要显式造出低秩的大矩阵；固定的矩阵乘积只有在 $mn<k(m+n)$（合并后不比两块加起来大）时才值得提前算好。
- 矩阵吸收：$\mathbf q^\top(W^{UK}\mathbf c)=((W^{UK})^\top\mathbf q)^\top\mathbf c$。缓存越长越划算，$T=1$ 时不省；中间夹着随位置变化的矩阵（RoPE）时失效。
:::

# 练习与答案 | 12 题：概念、计算、推导、综合 | 45

先自己做，再展开答案。计算题都可以用纸笔或计算器完成，想核对的话，每题都能用几行 NumPy 验证。最后两道综合题要把几页的工具连起来用，一道用 DeepSeek-V2 的真实尺寸算账，一道用 einsum 写出分组查询注意力。

## 概念题

:::: quiz 练习 1 · 把键归一化
假设某个模型在打分前把每个键都缩放成长度 1。（a）对同一个查询，各个键的点积排名与余弦相似度排名是什么关系？（b）查询的长度还起作用吗？起什么作用？
::: answer 看答案
（a）$\norm{\mathbf k_j}=1$ 时 $\mathbf q\cdot\mathbf k_j=\norm{\mathbf q}\cos\theta_j$。对同一个查询，$\norm{\mathbf q}$ 是所有键共用的正数，所以点积排名与余弦排名完全相同，键的“嗓门”（§1）被取消了。

（b）起作用。$\norm{\mathbf q}$ 同时放大或缩小这个查询的所有分数，相当于 softmax 的温度：查询越长，权重越集中。排名不变，但分配的尖锐程度变了。
:::
::::

:::: quiz 练习 2 · 权重表的两种极端
在 $O=PV$ 中（(M1.8)），用“行的组合”回答：（a）若 $P$ 的第 $t$ 行是在第 $j$ 个位置为 1、其余为 0 的独热向量，输出第 $t$ 行是什么？（b）若第 $t$ 行的前 $t$ 个元素都是 $1/t$（因果掩码下的均匀分布），输出是什么？（c）多头注意力里，为什么说“拆头不花计算”却“多头不等于一个大头”？各指出所用的公式。
::: answer 看答案
（a）$\mathbf o_t^\top=\sum_jP_{tj}\mathbf v_j^\top=\mathbf v_j^\top$：原样复制第 $j$ 个值向量，注意力退化成“查表取一个”。

（b）$\mathbf o_t=\frac1t(\mathbf v_1+\cdots+\mathbf v_t)$：前 $t$ 个值向量的简单平均，所有信息被一视同仁地混在一起。真实的注意力在这两个极端之间。

（c）拆头是按行（外侧）分块 (M1.13)，每个头的查询就是大向量的一段，没有新的运算。多头不等于大头，是因为大头的分数是各头分数之和 (M1.15)，只做一次 softmax；多头各做各的 softmax，得到 $n_h$ 套不同的权重。
:::
::::

## 计算题

:::: quiz 练习 3 · 手算、转置与外积
$$A=\begin{pmatrix}2&1\\0&3\end{pmatrix},\qquad B=\begin{pmatrix}1&-1\\2&0\end{pmatrix}.$$

（a）求 $AB$ 与 $BA$。（b）求 $(AB)^\top$，并用 $B^\top A^\top$ 验证 (M1.10)。（c）把 $AB$ 写成两个外积之和 (M1.9) 并核对。
::: answer 看答案
（a）$AB$：第一行 $(2\cdot1+1\cdot2,\ 2\cdot(-1)+1\cdot0)=(4,-2)$，第二行 $(0+3\cdot2,\ 0+0)=(6,0)$。$BA$：第一行 $(1\cdot2+(-1)\cdot0,\ 1\cdot1+(-1)\cdot3)=(2,-2)$，第二行 $(2\cdot2+0,\ 2\cdot1+0)=(4,2)$。两者不同。

（b）$(AB)^\top$ 的第一行 $(4,6)$、第二行 $(-2,0)$。$B^\top$ 的两行是 $(1,2)$、$(-1,0)$，$A^\top$ 的两行是 $(2,0)$、$(1,3)$。$B^\top A^\top$：第一行 $(1\cdot2+2\cdot1,\ 1\cdot0+2\cdot3)=(4,6)$，第二行 $(-2+0,\ 0+0)=(-2,0)$，一致。

（c）$A$ 的两列是 $\mathbf a_1=(2,0)$、$\mathbf a_2=(1,3)$，$B$ 的两行是 $(1,-1)$、$(2,0)$：

$$\mathbf a_1\mathbf b_1^\top+\mathbf a_2\mathbf b_2^\top=\begin{pmatrix}2&-2\\0&0\end{pmatrix}+\begin{pmatrix}2&0\\6&0\end{pmatrix}=\begin{pmatrix}4&-2\\6&0\end{pmatrix}=AB.$$
:::
::::

:::: quiz 练习 4 · 打分表有多大
某模型 $d=4096$、$n_h=32$，序列长 $T=2048$。（a）$d_h$ 是多少？（b）一层里所有头的打分张量 $S$ 是什么形状、多少个元素？（c）若每个元素用 {{BF16::一种 16 位浮点格式，每个数占 2 字节，大模型训练与推理常用；详见 M08}}（2 字节）存下来，占多少 MiB？（d）$T$ 翻倍到 4096 时，（b）（c）变成多少？
::: answer 看答案
（a）$d_h=4096/32=128$。

（b）$[n_h,T,T]=[32,2048,2048]$，共 $32\times2048^2=134{,}217{,}728$ 个元素。

（c）$134{,}217{,}728\times2=268{,}435{,}456$ 字节 $=256$ MiB（1 MiB $=2^{20}$ 字节）。这还只是一层、一条序列。

（d）$T$ 翻倍，元素数乘 4：$536{,}870{,}912$ 个，$1{,}073{,}741{,}824$ 字节 $=1$ GiB。这种平方增长正是后来的实现尽量不把整张 $[T,T]$ 表存下来的原因之一。
:::
::::

:::: quiz 练习 5 · 写 einsum
写出下列运算的 einsum 字符串和输出形状，并用“2 乘以所有字母大小之积”估算 FLOPs。（a）批量矩阵乘法：$X$ 为 $[B,m,k]$，$Y$ 为 $[B,k,n]$，每一批各乘各的。（b）$T$ 个查询 $Q\in\R^{T\times d_h}$ 分别与**同一个**键 $\mathbf k\in\R^{d_h}$ 点积。（c）§3 的 `hte,dhe->td`：哪些字母被求和？FLOPs 是多少？与哪一步的成本相同？
::: answer 看答案
（a）`bmk,bkn->bmn`，输出 $[B,m,n]$，约 $2Bmkn$ 次。`b` 两边都有、输出也有，是批量轴。

（b）`te,e->t`，输出 $[T]$，约 $2Td_h$ 次。这就是 $Q\mathbf k$，矩阵乘向量。

（c）`h` 和 `e` 不在输出里，都被求和。FLOPs 约 $2\cdot n_h\cdot T\cdot d_h\cdot d=2Td^2$（因为 $n_hd_h=d$），和输出投影 $[T,d]\times[d,d]$ 完全相同：它本来就是“拼接后乘 $W^O$”。
:::
::::

:::: quiz 练习 6 · 括号放在哪
$A\in\R^{4096\times128}$，$B\in\R^{128\times4096}$，$\mathbf x\in\R^{4096}$。分别计算 $(AB)\mathbf x$ 与 $A(B\mathbf x)$ 的 FLOPs。
::: answer 看答案
$(AB)\mathbf x$：$AB$ 是 $[4096,4096]$，$2\cdot4096\cdot128\cdot4096=4{,}294{,}967{,}296$；再乘 $\mathbf x$，$2\cdot4096\cdot4096=33{,}554{,}432$；合计 $4{,}328{,}521{,}728\approx4.33\times10^9$。

$A(B\mathbf x)$：$B\mathbf x$ 是 $2\cdot128\cdot4096=1{,}048{,}576$；$A(B\mathbf x)$ 同样 $1{,}048{,}576$；合计 $2{,}097{,}152\approx2.10\times10^6$。

相差 2064 倍（$4{,}328{,}521{,}728\div2{,}097{,}152=2064$）。$AB$ 是中间维只有 128 的低秩矩阵，把它显式造出来是浪费；这正是 MLA 这类低秩结构在推理时要避免的事。
:::
::::

## 推导题

:::: quiz 练习 7 · 柯西–施瓦茨不等式
对任意实数 $\lambda$ 都有 $\norm{\mathbf a-\lambda\mathbf b}^2\ge0$。把它展开成 $\lambda$ 的二次式，用判别式证明 $\abs{\mathbf a\cdot\mathbf b}\le\norm{\mathbf a}\norm{\mathbf b}$，从而说明 §1 用 (M1.3) 定义的 $\cos\theta$ 一定落在 $[-1,1]$。
::: answer 看答案
若 $\mathbf b=\mathbf 0$，两边都是 0，成立。设 $\mathbf b\ne\mathbf 0$。按 §1 推导里同样的展开：

$$f(\lambda)=\norm{\mathbf a-\lambda\mathbf b}^2=\norm{\mathbf b}^2\lambda^2-2(\mathbf a\cdot\mathbf b)\lambda+\norm{\mathbf a}^2\ge0\quad\text{对所有 }\lambda.$$

这是开口向上的二次函数（$\norm{\mathbf b}^2>0$），它从不取负值，所以至多有一个实根，判别式 $\le0$：$4(\mathbf a\cdot\mathbf b)^2-4\norm{\mathbf a}^2\norm{\mathbf b}^2\le0$，即 $(\mathbf a\cdot\mathbf b)^2\le\norm{\mathbf a}^2\norm{\mathbf b}^2$，开方得证。两边同除以 $\norm{\mathbf a}\norm{\mathbf b}$（都不为零时），得 $-1\le\cos\theta\le1$。
:::
::::

:::: quiz 练习 8 · 注意力不对称
（a）证明对任意形状相容的 $\mathbf x$、$A$、$\mathbf y$，有 $\mathbf x^\top A\mathbf y=\mathbf y^\top A^\top\mathbf x$。（b）用 (M1.20) 说明：token $t$ 对 token $j$ 的分数，一般不等于 token $j$ 对 token $t$ 的分数。什么条件下两者相等？
::: answer 看答案
（a）$\mathbf x^\top A\mathbf y$ 是 $[1,1]$，一个数的转置是它自己。由 (M1.10) 的三因子形式，$(\mathbf x^\top A\mathbf y)^\top=\mathbf y^\top A^\top(\mathbf x^\top)^\top=\mathbf y^\top A^\top\mathbf x$。

（b）记 $M_i=(W^Q_i)^\top W^K_i$。$t$ 对 $j$ 的分数是 $\mathbf h_t^\top M_i\mathbf h_j$；$j$ 对 $t$ 的分数是 $\mathbf h_j^\top M_i\mathbf h_t$，由（a）等于 $\mathbf h_t^\top M_i^\top\mathbf h_j$。两者对所有 $\mathbf h$ 都相等，当且仅当 $M_i=M_i^\top$，例如 $W^Q_i=W^K_i$。一般的 $W^Q_i\ne W^K_i$ 让“我找你”和“你找我”可以不同，这是用两个不同矩阵造查询和键的一个理由。
:::
::::

:::: quiz 练习 9 · 值那一侧的吸收
在 MLA 中，第 $i$ 个头的值由缓存展开：$\mathbf v_{j,i}=W^{UV}_i\mathbf c_j$，$W^{UV}_i\in\R^{d_h\times d_c}$。头输出 $\mathbf o_{t,i}=\sum_jP^{(i)}_{tj}\mathbf v_{j,i}$，层输出 $\mathbf u_t=\sum_iW^O_i\mathbf o_{t,i}$（(M1.14)）。证明

$$\mathbf u_t=\sum_i\big(W^O_iW^{UV}_i\big)\Big(\sum_jP^{(i)}_{tj}\,\mathbf c_j\Big),$$

写出括号 $W^O_iW^{UV}_i$ 的形状，并说明这样算省掉了什么。
::: answer 看答案
第一步，线性：$\mathbf o_{t,i}=\sum_jP^{(i)}_{tj}W^{UV}_i\mathbf c_j=W^{UV}_i\big(\sum_jP^{(i)}_{tj}\mathbf c_j\big)$，因为矩阵乘向量对加法可分配、数可以提到矩阵外面。第二步，结合律：$W^O_i\mathbf o_{t,i}=W^O_i\big(W^{UV}_i\mathbf s\big)=\big(W^O_iW^{UV}_i\big)\mathbf s$，其中 $\mathbf s=\sum_jP^{(i)}_{tj}\mathbf c_j$。对 $i$ 求和即得。

形状：$[d,d_h]\times[d_h,d_c]\to[d,d_c]$。它只由权重决定，可以在加载模型时算一次。省下大头的是括号的位置：推理时不必为缓存里的 $T$ 个 token 展开值向量（每头省掉 $2Td_cd_h$ 次运算），只需先对 $\mathbf c_j$ 做加权平均（$2Td_c$），得到一个 $d_c$ 维向量 $\mathbf s$，再把它映射到 $d$ 维。这就是 DeepSeek-V2 所说的“$W^{UV}$ 被吸收进 $W^O$” [@deepseekv2]。

最后这一步和 §6 查询那一侧一样，按形状决定要不要真的合并：乘合并后的 $[d,d_c]$ 矩阵每头要 $2dd_c$ 次，先乘 $W^{UV}_i$ 再乘 $W^O_i$ 只要 $2d_hd_c+2dd_h$ 次。V2 的尺寸（$d=5120$、$d_c=512$、$d_h=128$）下是 $5{,}242{,}880$ 对 $1{,}441{,}792$，分开乘更省。
:::
::::

:::: quiz 练习 10 · 投影矩阵
设 $U\in\R^{d\times k}$ 满足 $U^\top U=I$，投影矩阵记作 $\Pi=UU^\top$（本课的 $P$ 留给注意力权重，所以这里不用它）。（a）证明 $\Pi^2=\Pi$，$\Pi^\top=\Pi$，并用一句话解释 $\Pi^2=\Pi$ 的几何意义。（b）对 §6 例题的 $U=[\mathbf u_1\ \mathbf u_2]$ 写出 $\Pi$，验证 $\Pi\mathbf h=(2,1,1)$。
::: answer 看答案
（a）$\Pi^2=(UU^\top)(UU^\top)=U(U^\top U)U^\top=UIU^\top=\Pi$，第二步用结合律。$\Pi^\top=(U^\top)^\top U^\top=UU^\top=\Pi$，用 (M1.10)。几何意义：已经在子空间里的向量，再投影一次不会变，投影两次等于投影一次。

（b）按外积之和：$\Pi=\mathbf u_1\mathbf u_1^\top+\mathbf u_2\mathbf u_2^\top$。$\mathbf u_1\mathbf u_1^\top$ 只有左上角是 1；$\mathbf u_2\mathbf u_2^\top$ 的右下 $2\times2$ 块四个元素都是 $\tfrac12$。所以

$$\Pi=\begin{pmatrix}1&0&0\\[3pt]0&\tfrac12&\tfrac12\\[3pt]0&\tfrac12&\tfrac12\end{pmatrix},\qquad \Pi\begin{pmatrix}2\\-1\\3\end{pmatrix}=\begin{pmatrix}2\\(-1+3)/2\\(-1+3)/2\end{pmatrix}=\begin{pmatrix}2\\1\\1\end{pmatrix}.$$
:::
::::

## 综合题

:::: quiz 练习 11 · DeepSeek-V2 一层注意力的打分账单
DeepSeek-V2 有 $n_h=128$ 个头，每头 $d_h=128$，隐藏维度 $d=5120$ [@deepseekv2]。先忽略 V2 每个头额外的 64 维 RoPE 部分（第 9 课再把它加上），取序列长 $T=4096$。（a）一层里所有头的打分 $Q_iK_i^\top$ 共需多少 FLOPs？加权求和 $P_iV_i$ 呢？（b）如果实现跳过因果掩码丢掉的那一半，两项合计约多少？（c）有人直接套 (M1.17) 的 $4T^2d$、代入 $d=5120$。他会算出多少？错在哪里？
::: answer 看答案
（a）每个头的打分是 $[T,d_h]\times[d_h,T]$，按 (M1.16) 要 $2T^2d_h$ 次运算；128 个头合计

$$2T^2d_hn_h=2\times4096^2\times128\times128=549{,}755{,}813{,}888\approx5.50\times10^{11}.$$

加权求和 $[T,T]\times[T,d_h]$ 每个头也是 $2T^2d_h$，总量相同。两项合计 $1{,}099{,}511{,}627{,}776\approx1.10\times10^{12}$。

（b）因果掩码下每一行只需要 $j\le t$ 的约一半格子（[§5](#p5) 的警告框），跳过被掩掉的块时两项合计约 $5.50\times10^{11}$。

（c）$4T^2d=4\times4096^2\times5120=343{,}597{,}383{,}680\approx3.44\times10^{11}$，只有（a）合计的 $1/3.2$。(M1.17) 推导时假设了 $n_hd_h=d$（GPT-2 正是这样），打分项真正的系数是 $n_hd_h$；V2 的 $n_hd_h=128\times128=16{,}384=3.2d$，前提不成立。教训是：套公式之前先核对它的前提；最稳妥的办法是回到每一步的形状，用 $2mnk$ 重新数一遍。
:::
::::

:::: quiz 练习 12 · 用 einsum 写分组查询注意力（GQA）
某模型有 $n_h=32$ 个查询头、$n_g=8$ 组键和值，每组被 $r=n_h/n_g=4$ 个查询头共用，$d_h=128$，序列长 $T$。查询按头存成 $Q\in\R^{n_h\times T\times d_h}$，键存成 $K\in\R^{n_g\times T\times d_h}$。约定第 $h$ 个查询头（从 0 数）用第 $\lfloor h/r\rfloor$ 组。（a）把 $Q$ reshape 成 $[n_g,r,T,d_h]$。说明新下标 $(g,a)$（$g$ 是组号，$a$ 是组内序号）与原头号 $h$ 的关系，以及为什么第 $g$ 块里的 $r$ 个头恰好都该用第 $g$ 组键。（b）写出打分的 einsum 字符串和输出形状，指出哪些字母被求和、哪些是批量轴。（c）与 32 组键值的普通多头注意力相比，打分的 FLOPs 和每层 KV cache 的大小各是原来的几倍？
::: answer 看答案
（a）reshape 只改变数下标的方式（[§3](#p3) 的前置知识框）：长度为 $n_h$ 的头轴被按顺序切成 $n_g$ 段、每段 $r$ 个，所以 $h=g\cdot r+a$，$0\le a<r$。反过来 $g=\lfloor h/r\rfloor$，这正是约定里头 $h$ 所用的组号。第 $g$ 块里的头是 $h=gr,\dots,gr+r-1$，全部用第 $g$ 组键。（[§4](#p4) 的注释用从 1 数的头号写成 $\lceil i\,n_g/n_h\rceil$，是同一件事。）

（b）`grte,gse->grts`，输出 $[n_g,r,T,T]=[8,4,T,T]$，再 reshape 回 $[32,T,T]$，就是每个查询头一张打分表。`e` 不在输出里，被求和；`g` 在两个输入和输出里都有，是批量轴（逐组各算各的）；`r` 只出现在 $Q$ 和输出里，不求和，意思是“同一组键被组里的 $r$ 个查询头各用一次”。

（c）FLOPs 按“2 乘以所有字母大小之积”：$2\cdot n_g\cdot r\cdot T\cdot T\cdot d_h=2\cdot32\cdot T^2\cdot128$，因为 $n_gr=n_h$，和普通多头注意力**完全相同**。KV cache 每层存 $K$、$V$ 各 $n_g\times T\times d_h$ 个数，而普通多头要存 $n_h\times T\times d_h$ 个，只剩 $n_g/n_h=1/4$。这就是 [§4](#p4) 注释里“KV cache 小了 $n_h/n_g$ 倍”的来历：GQA 省的是存储和读缓存的带宽，不是打分的运算量。
:::
::::

```refs
- id: vaswani2017
  title: "Attention Is All You Need"
  a: Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). NIPS 2017. arXiv:1706.03762（读的是 v7）
  url: https://arxiv.org/abs/1706.03762
  finding: 提出 Transformer；§3.2.1 定义缩放点积注意力 $\softmax(QK^\top/\sqrt{d_k})V$，指出点积注意力可用高度优化的矩阵乘法实现，推测大 $d_k$ 下点积过大会让 softmax 梯度极小，并在脚注里给出点积方差为 $d_k$ 的估算（该脚注 v2 起才有，v1 没有）；§3.2.2 定义多头注意力 $\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O$，其中 $W_i^Q\in\R^{d_{\text{model}}\times d_k}$。
  matters: 本课用它说明“为什么用点积打分”的工程理由，并以它的行向量写法（XW）对照 DeepSeek 论文的列向量写法（Wh）。
- id: gpt2-config
  kind: artefact
  title: GPT-2 small 模型配置（Hugging Face openai-community/gpt2, config.json）
  a: OpenAI / Hugging Face. 读取于 2026-09-24
  url: https://huggingface.co/openai-community/gpt2/blob/main/config.json
  finding: GPT-2 small 的 n_embd = 768、n_head = 12、n_layer = 12、n_positions = 1024、vocab_size = 50257。
  matters: 本课所有“真实形状”的例子（$d=768$、$n_h=12$、$d_h=64$、$T=1024$）都取自这份配置，与第 1 课的形状追踪一致。
- id: deepseekv2
  title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  a: DeepSeek-AI (2024). arXiv:2405.04434（读的是 v5）
  url: https://arxiv.org/abs/2405.04434
  finding: §2.1 用列向量写法 $\mathbf q_t=W^Q\mathbf h_t$ 定义注意力（$W^Q\in\R^{d_hn_h\times d}$）；§2.1.2 用 $W^{DKV}\in\R^{d_c\times d}$ 与 $W^{UK},W^{UV}\in\R^{d_hn_h\times d_c}$ 做低秩 KV 联合压缩，并指出推理时 $W^{UK}$ 可吸收进 $W^Q$、$W^{UV}$ 可吸收进 $W^O$；§2.1.3 说明 RoPE 让吸收失效，因为随位置变化的矩阵夹在中间，而矩阵乘法不满足交换律，并改用每 token 另存一个 $d_h^R$ 维的共享 RoPE 键（缓存 $(d_c+d_h^R)l$ 个数）；查询也做低秩压缩（$W^{DQ}\in\R^{d_c'\times d}$、$W^{UQ}\in\R^{d_hn_h\times d_c'}$）；§3.1.2 给出 $n_h=d_h=128$、$d_c=512$、$d_c'=1536$、$d_h^R=64$、$d=5120$。
  matters: 本课 §2、§6 的“不可交换”“结合律与矩阵吸收”都以它为落脚点，并用它的真实维度做吸收前后的算力对比。
- id: torch-linear
  kind: artefact
  title: PyTorch 文档：torch.nn.Linear
  a: PyTorch（文档 2.14 版）. 读取于 2026-09-24
  url: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
  finding: nn.Linear 计算 y = xAᵀ + b；可学习权重的形状是 (out_features, in_features)；输入与输出只在最后一维上分别是 in_features 与 out_features，前面的维度原样保留。
  matters: 本课用它说明主流框架按“列写法”存权重、计算时再转置，这正是 (M1.11) 在代码里的样子。
- id: numpy-einsum
  kind: artefact
  title: NumPy 文档：numpy.einsum
  a: NumPy 开发者. 读取于 2026-09-24
  url: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
  finding: einsum 按爱因斯坦求和约定计算；显式模式下 '->' 右边写出输出下标，不在输出里的下标被求和（如 'i->' 等于求和）；文档给出矩阵乘法 'ij,jk'（显式写法 'ij,jh->ih'）、转置 'ij->ji'、内积 'i,i'、外积 'i,j' 等写法。
  matters: 本课的 einsum 规则与注意力各步的写法都以它的定义为准，读者可以直接在 NumPy 里运行验证。
- id: hf-gpt2
  kind: artefact
  title: Hugging Face Transformers：GPT-2 实现（modeling_gpt2.py 与 pytorch_utils.Conv1D）
  a: Hugging Face（main 分支）. 读取于 2026-09-24
  url: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
  finding: GPT-2 的自注意力用 c_attn = Conv1D(3·embed_dim, embed_dim) 一次算出 Q、K、V 再 split；Conv1D 的权重形状为 (nx, nf) 即（输入维, 输出维），前向是 addmm(bias, x, weight)，注释写明它像线性层但权重是转置的；拆头用 view(…, −1, head_dim).transpose(1, 2)。
  matters: 本课用它展示“代码库存权重的方向各不相同”“融合的 QKV 投影就是分块矩阵”“拆头只是 reshape 加 transpose”三件事的真实样子。
```
