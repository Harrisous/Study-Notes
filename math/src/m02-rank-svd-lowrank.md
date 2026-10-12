---
title: M02 · 秩、低秩分解与 SVD：MLA 背后的线性代数
short: 低秩与 SVD
kicker: 数学补课 · M02
description: 秩与列空间、低秩分解 W = UV 的参数与 FLOPs 账、奇异值分解（手算 2×2、数值 3×3、旋转–拉伸–旋转的几何）、谱范数与 Frobenius 范数、Eckart–Young–Mirsky 最优低秩近似、学出来的压缩与 SVD 截断的区别、结合律与“吸收”的形状和 FLOPs，服务第 9 课 MLA 与第 18 课。
lede: 第 9 课的 MLA 让 128 个头的键和值都从一个 512 维的向量展开，推理时甚至不必真的展开。这一课把背后的线性代数从头讲透：一个矩阵真正能“说出”几个方向（**秩**），怎样把一张大表拆成两块瘦矩阵（**低秩分解**），为什么任何矩阵都是“旋转、拉伸、再旋转”（**SVD**），为什么丢掉最小的几个奇异值就是最好的压缩（**Eckart–Young–Mirsky**），以及 MLA 为什么不去截断一个现成的矩阵，而是把两块瘦矩阵直接**学**出来。
meta: 预计阅读 170 分钟，练习另需约 45 分钟 · 前置：M01（矩阵乘法的几种读法、外积之和、结合律） · 服务于第 9、18 课
index: index.html
prev: m01-vectors-matrices.html | M01 · 矩阵与形状
next: m03-rotation-complex.html | M03 · 旋转与复数
---

## 这一课回答什么问题

[M01](m01-vectors-matrices.html) 的最后一页留下两句话：中间维很小的矩阵乘积，是少数几个秩 1 矩阵之和；这样的大矩阵永远不要显式造出来。这一课把“少数几个方向”这件事说准确，并算清楚它值多少钱。

读完这一课，下面这些问题你都能自己回答：

- DeepSeek-V2 从隐藏状态到 128 个头的键，等效于一个 $16384\times5120$ 的矩阵。为什么说它“只有 512 个方向”？
- 把一张大表拆成两块瘦矩阵，省多少参数、多少 FLOPs？什么时候反而不省？
- SVD 是什么？怎样用纸笔算一个 $2\times2$ 的 SVD？“旋转、拉伸、再旋转”指什么？
- 最好的“秩不超过 $k$”的近似是哪一个？用哪把尺子量？误差多少？
- MLA 的压缩矩阵为什么直接训练，而不是先训练普通注意力再做 SVD 截断？
- “矩阵吸收”改变了什么？什么时候该把两个矩阵预先乘好，什么时候绝对不该？

## 这一课在科技树上的位置

本课直接服务两课。[第 9 课](../deepseek/lesson-09-v2-mla.html)（多头潜在注意力 MLA）：低秩 KV 联合压缩就是本课的低秩分解，矩阵吸收就是本课最后一页的结合律账单。[第 18 课](../deepseek/lesson-18-v4.html)（DeepSeek-V4）：注意力不再是 MLA，但保留了查询的低秩压缩，又加了一个分块低秩的“分组输出投影”；它的 Muon 优化器把更新矩阵的奇异值推向 1。[M11](m11-doubly-stochastic-sinkhorn.html) 用本课的谱范数证明双随机矩阵不放大向量。工具全部来自上一课 [M01](m01-vectors-matrices.html)（列的加权和、外积之和、结合律）；下一课 [M03](m03-rotation-complex.html) 专讲 SVD 里那两个“旋转”。

## 读完你能……

1. 用列之间的线性关系求一个小矩阵的秩和零空间，并用秩–零化度定理数维数；说出为什么两个矩阵乘积的秩不超过其中任何一个。
2. 把秩为 $r$ 的矩阵写成 $W=UV$，算出参数与 FLOPs 的账和“不省钱”的临界秩，并指出这种分解不唯一。
3. 手算一个 $2\times2$ 矩阵的 SVD，读懂数值软件给出的 $3\times3$ SVD，并画出“旋转–拉伸–旋转”的图。
4. 准确陈述 Eckart–Young–Mirsky 定理（前提、两种范数下的最优误差、唯一性），只用奇异值就写出最优秩 $k$ 近似的误差，并说对这个定理的历史归属。
5. 解释 MLA“直接学两块瘦矩阵”与“事后 SVD 截断”的区别；用结合律写出吸收前后的形状与 FLOPs，判断什么时候吸收划算。

::: note 本课的记号
- 沿用 [M01](m01-vectors-matrices.html) 的约定：列向量、粗体小写；矩阵大写；形状写 $[m,n]$。$\rank A$、$C(A)$、$N(A)$ 是秩、列空间、零空间（§1）。
- **两套 $U$、$V$。** §2 的 $W=UV$ 里它们是任意两块瘦矩阵；§3 起的 SVD $A=U\Sigma V^\top$ 里它们是正交矩阵。MLA 的 $W^{UK}$（U = up，上投影）恰好扮演 §2 的 $U$，$W^{DKV}$（D = down）扮演 $V$。
- $\sigma_i$ 是奇异值；$\norm{A}_2$、$\norm{A}_F$ 是谱范数与 Frobenius 范数（§4）。DeepSeek 维度沿用第 9 课：$d=5120$，$n_h=d_h=128$，$d_c=512$，$d_c'=1536$，$d_h^R=64$ [@dsv2]。
:::

[[legend]]

[[toc]]

# 秩与列空间 | 一个矩阵真正能“说出”几个方向 | 25

先把问题摆出来。[M01 §6](m01-vectors-matrices.html#p6) 说过，GPT-2 small 的一个查询矩阵 $W^Q_i$ 形状是 $[64,768]$，它对至少 704 个方向“完全看不见”。第 9 课则反过来：DeepSeek-V2 从隐藏状态 $\mathbf h_t\in\R^{5120}$（第 $t$ 个 token 进入注意力层的向量）算出 128 个头的键，先乘 $W^{DKV}$ 压成 512 个数，再乘 $W^{UK}$ 展开成 16384 个数 [@dsv2]。把两步合起来看，它等效于一个 $[16384,5120]$ 的矩阵，可是所有的键都只是那 512 个数的线性函数。两件事背后是同一个问题：**一个矩阵的输出，到底能朝多少个互相独立的方向伸展？** 这个数就叫**秩**（rank）。这一页从零定义它，并给出三条后面反复要用的性质。

## 线性组合与线性无关

**含义。** 用几个数给几个向量分别加权，再加起来，得到的新向量叫它们的一个**线性组合**（linear combination）。

**理解。** 像调色：三管颜料按不同比例混合，能调出很多颜色，但如果第三管本身就是前两管按 1:2 混成的，那它不会让你多调出任何一种颜色。它是“多余的”。这个类比有一处不准：颜料只能加、不能“减”，线性组合的系数却可以是负数，也可以大于 1，所以向量能“调”出的范围比颜料大得多（一整条直线、一整个平面），而不只是两管颜料之间的那一段。

**数学。** 向量 $\mathbf a_1,\dots,\mathbf a_k\in\R^m$ 的线性组合是 $c_1\mathbf a_1+\cdots+c_k\mathbf a_k$，$c_i$ 是实数。所有线性组合构成的集合叫它们**张成**（span）的空间。如果只有“全部系数为 0”这一种组合能得到零向量，就说这组向量**线性无关**（linearly independent）；否则**线性相关**。线性相关等价于“其中某一个向量是其余向量的线性组合”：若 $c_1\mathbf a_1+\cdots+c_k\mathbf a_k=\mathbf 0$ 且 $c_j\ne0$，两边除以 $c_j$ 移项，就把 $\mathbf a_j$ 写成了其余向量的组合。

::: ex 三列里有一列是多余的
本课反复使用的 $3\times3$ 例子：

$$M=\begin{pmatrix}1&0&1\\0&1&2\\1&1&3\end{pmatrix},\qquad \mathbf c_1=\begin{pmatrix}1\\0\\1\end{pmatrix},\ \mathbf c_2=\begin{pmatrix}0\\1\\1\end{pmatrix},\ \mathbf c_3=\begin{pmatrix}1\\2\\3\end{pmatrix}.$$

第 3 列等于第 1 列加 2 倍第 2 列：$\mathbf c_1+2\mathbf c_2=(1,0,1)+(0,2,2)=(1,2,3)=\mathbf c_3$。所以 $1\cdot\mathbf c_1+2\cdot\mathbf c_2-1\cdot\mathbf c_3=\mathbf 0$，三列线性相关。

前两列线性无关：$a\mathbf c_1+b\mathbf c_2=(a,\ b,\ a+b)$，它等于零向量只能 $a=b=0$。
:::

## 列空间与秩

**含义。** 一个矩阵所有可能的输出组成的集合，叫它的**列空间**（column space）；列空间有几维，矩阵的秩就是几。

**理解。** 按 [M01 (M1.6)](m01-vectors-matrices.html#p2)，$A\mathbf x=x_1\mathbf a_1+\cdots+x_n\mathbf a_n$：输出一定是各列的线性组合，反过来各列的任何组合也都能作为输出出现（把组合系数当输入）。所以“输出能去哪里”完全由列决定，多余的列不增加任何新去处。

::: def 子空间、列空间、维数与秩
设 $A\in\R^{m\times n}$，列为 $\mathbf a_1,\dots,\mathbf a_n$。

- **子空间**（subspace）：$\R^m$ 的一个子集，含零向量，而且对加法和数乘封闭（其中任意两个向量相加、任意一个向量乘一个数，结果仍在里面）。$\R^3$ 里的子空间只有四种：原点本身、过原点的直线、过原点的平面、整个 $\R^3$。不过原点的平面不是子空间：它不含零向量。
- **列空间**：$C(A)=\{A\mathbf x:\mathbf x\in\R^n\}=\operatorname{span}(\mathbf a_1,\dots,\mathbf a_n)\subseteq\R^m$。它是一个子空间：$A\mathbf 0=\mathbf 0$，$A\mathbf x+A\mathbf y=A(\mathbf x+\mathbf y)$，$cA\mathbf x=A(c\mathbf x)$，输出的和与倍数仍是输出。
- **基与维数**：一个子空间里一组线性无关、又能张成整个子空间的向量叫它的一组**基**（basis）；任意两组基所含向量个数相同（Axler 2.34 [@axler2024]），这个共同的个数叫子空间的**维数**（dimension）。
- **秩**：$\rank A=\dim C(A)$，等于从 $A$ 的列里最多能挑出多少个线性无关的列（Axler 3.52、3.58 把它称为列秩，并定义为秩）。
:::

::: eq 列空间与秩 | (M2.1)
$$C(A)=\operatorname{span}(\mathbf a_1,\dots,\mathbf a_n),\qquad \rank A=\dim C(A)\le\min(m,n)$$
上界 $\min(m,n)$：列空间由 $n$ 个列张成，所以维数不超过 $n$（Axler 2.22：线性无关组不比张成组长）；它又是 $\R^m$ 的子空间，维数不超过 $m$（同书 2.37）。
:::

::: ex M 的列空间是一个平面
$M\mathbf x=(x_1+x_3,\ x_2+2x_3,\ x_1+x_2+3x_3)$。看第三个分量：$x_1+x_2+3x_3=(x_1+x_3)+(x_2+2x_3)$，恰好是前两个分量之和。所以**无论输入是什么，输出都满足 $z=x+y$**：$C(M)$ 是 $\R^3$ 里过原点的平面 $z=x+y$，维数 2，$\rank M=2$。$\mathbf c_1$、$\mathbf c_2$ 是这个平面的一组基，$\mathbf c_3$ 落在平面里：$3=1+2$。

对比：若把右下角的 3 改成 3.1，第 3 列变成 $(1,2,3.1)$，不再满足 $z=x+y$，三列线性无关，秩变成 3，输出能铺满整个 $\R^3$。
:::

::: fig figures/m02-rank-geometry.svg | 图 M2.1 秩就是输出集合的维数
**看三个面板里绿色的部分：它是矩阵所有输出的集合。** (a) 一个秩 2 的 $2\times2$ 矩阵把单位圆（灰色虚线，24 个等距的点）变成一个椭圆，输出能到达平面上任何方向。(b) 把同一个矩阵的第二个拉伸系数 0.8 换成 0，得到秩 1 的矩阵：整个圆被压到一条线段上。红色虚线方向被送到 0（零空间），所以两个蓝色输入只差一个红色方向的分量，却落到同一个绿色输出上。(c) $3\times3$ 的 $M$：三列（蓝、绿、紫）都在同一个过原点的平面里，第 3 列是前两列的组合，没带来新方向。(a)(b) 的矩阵由脚本按 $2\,\mathbf u_1\mathbf v_1^\top+0.8\,\mathbf u_2\mathbf v_2^\top$ 算出，写法在 §3 解释。
:::

## 行秩等于列秩

上面的秩是按**列**数的。按**行**数呢？$A$ 的各行张成的空间的维数叫**行秩**。一个 $3\times5$ 的矩阵有 5 列、3 行，列秩和行秩乍看没有理由相等，但它们总是相等。证明只需要一个构造，而这个构造恰好就是下一页的主角。

::: thm 列秩等于行秩（Axler 3.56、3.57）
任意 $A\in\R^{m\times n}$ 的列秩等于行秩 [@axler2024]。因此 $\rank A=\rank A^\top$。
:::

::: deriv 用“列–行分解”证明行秩 ≤ 列秩
$\rank A=c\ge1,\qquad C=[\,\mathbf a_{j_1}\ \cdots\ \mathbf a_{j_c}\,]\in\R^{m\times c}$ || 从 $A$ 的列里挑出 $c$ 个线性无关的列，它们是列空间的一组基（Axler 2.30：张成组里总能挑出一组基）
$\mathbf a_k=C\,\mathbf r_k,\qquad \mathbf r_k\in\R^{c},\ \ k=1,\dots,n$ || 每一列都在列空间里，所以是这组基的组合；$\mathbf r_k$ 是组合系数
$A=CR,\qquad R=[\,\mathbf r_1\ \cdots\ \mathbf r_n\,]\in\R^{c\times n}$ || 把 $n$ 个等式并排写成一个矩阵等式（M01 读法三：按列）
$A_{i,:}=\textstyle\sum_{j=1}^{c}C_{ij}\,R_{j,:}$ || M01 读法三按行：乘积的第 $i$ 行是右边因子各行的组合（$R_{j,:}$ 表示 $R$ 的第 $j$ 行）
$\text{行秩}(A)\le c=\text{列秩}(A)$ || $A$ 的所有行都落在 $R$ 的 $c$ 行张成的空间里，那里的线性无关组不超过 $c$ 个（Axler 2.22）
$\text{列秩}(A)=\text{行秩}(A^\top)\le\text{列秩}(A^\top)=\text{行秩}(A)$ || 对 $A^\top$ 用同样的论证；$A^\top$ 的行就是 $A$ 的列
:::

::: ex 把 M 写成 CR
挑 $\mathbf c_1,\mathbf c_2$ 作基。三列的系数分别是 $(1,0)$、$(0,1)$、$(1,2)$（因为 $\mathbf c_3=\mathbf c_1+2\mathbf c_2$），所以

$$M=\underbrace{\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}}_{C:\ [3,2]}\underbrace{\begin{pmatrix}1&0&1\\0&1&2\end{pmatrix}}_{R:\ [2,3]}.$$

核对一格：$M_{33}=(1,1)\cdot(1,2)=3$。$M$ 的第三行 $(1,1,3)$ 等于 $R$ 的两行之和，行秩也是 2。一个 $3\times3$ 的表被写成 $[3,2]\times[2,3]$：这就是下一页的“低秩分解”。
:::

## 零空间与秩–零化度定理

**含义。** 被矩阵送到零向量的输入组成的集合，叫它的**零空间**（null space）。

**理解。** 零空间里的方向对这个矩阵“不可见”：输入沿零空间方向怎么变，输出都不变。图 M2.1(b) 里的红色虚线就是零空间。

**数学。** $N(A)=\{\mathbf x\in\R^n:A\mathbf x=\mathbf 0\}$。它是 $\R^n$ 的子空间（两个零空间里的向量相加、数乘，仍被送到 $\mathbf 0$）。输入空间的维数，恰好被“看得见的”和“看不见的”两部分分完：

::: eq 秩–零化度定理 | (M2.2)
$$n=\rank A+\dim N(A),\qquad A\in\R^{m\times n}$$
Axler 3.21，书中称“线性映射基本定理” [@axler2024]。输入有 $n$ 维；其中 $\dim N(A)$ 维被压成 0，剩下的维数恰好等于输出的维数。
:::

::: deriv 为什么恰好分完（Axler 3.21 的证明思路）
$\mathbf z_1,\dots,\mathbf z_k,\ \mathbf w_1,\dots,\mathbf w_{n-k}$ || $\mathbf z_i$ 是 $N(A)$ 的基，再扩充成 $\R^n$ 的基（Axler 2.32）
$A\mathbf x=A\big(\textstyle\sum_ia_i\mathbf z_i+\sum_jb_j\mathbf w_j\big)=\sum_jb_j\,A\mathbf w_j$ || $A\mathbf z_i=\mathbf 0$，所以 $A\mathbf w_j$ 张成 $C(A)$
$\textstyle\sum_jb_jA\mathbf w_j=\mathbf 0\ \Rightarrow\ \sum_jb_j\mathbf w_j=\sum_ic_i\mathbf z_i$ || 验证 $A\mathbf w_j$ 线性无关：$\sum_jb_j\mathbf w_j$ 被 $A$ 送到 $\mathbf 0$，属于 $N(A)$，可用基 $\mathbf z_i$ 写出
$b_1=\cdots=b_{n-k}=0$ || 移项得 $\sum_jb_j\mathbf w_j-\sum_ic_i\mathbf z_i=\mathbf 0$，而这 $n$ 个向量是一组基、线性无关，所有系数只能为 0
$\rank A=n-k=n-\dim N(A)$ || $A\mathbf w_1,\dots,A\mathbf w_{n-k}$ 张成 $C(A)$ 且线性无关，是 $C(A)$ 的一组基
:::

::: ex M 的零空间，以及“704 维看不见”
$M\mathbf x=\mathbf 0$ 即 $x_1+x_3=0$、$x_2+2x_3=0$（第三个方程是前两个之和，自动成立）。令 $x_3=-t$，得 $\mathbf x=t\,(1,2,-1)$。零空间是一条直线，维数 1；$3=2+1$，与 (M2.2) 一致。核对：$M(1,2,-1)=(1-1,\ 2-2,\ 1+2-3)=\mathbf 0$，这正是“$\mathbf c_1+2\mathbf c_2-\mathbf c_3=\mathbf 0$”换了一种写法。

M01 的说法现在可以算出来：$W^Q_i\in\R^{64\times768}$ 的秩至多 $\min(64,768)=64$，所以 $\dim N(W^Q_i)=768-\rank W^Q_i\ge704$。
:::

## 乘积的秩不超过任何一个因子

这是第 9 课最需要的一条性质：两个矩阵相乘，结果的方向数不会比其中任何一个多。

::: eq 乘积的秩 | (M2.3)
$$\rank(AB)\le\min\big(\rank A,\ \rank B\big)$$
中间维为 $r$ 的乘积 $[m,r]\times[r,n]$，秩至多是 $r$。
:::

::: deriv 为什么乘积的秩不会变大
$C(AB)\subseteq C(A)$ || $AB\mathbf x=A(B\mathbf x)$：$AB$ 的每个输出都是 $A$ 的某个输出
$\rank(AB)\le\rank A$ || 子空间的维数不超过所在空间（Axler 2.37）
$\rank(AB)=\rank\big((AB)^\top\big)=\rank(B^\top A^\top)$ || 行秩 = 列秩；转置规则 (M1.10)
$\rank(B^\top A^\top)\le\rank B^\top=\rank B$ || 对乘积 $B^\top A^\top$ 用第 (2) 行，再用一次行秩 = 列秩
:::

::: ex MLA 的键只有 512 个方向
DeepSeek-V2 中，$\mathbf k_t^C=W^{UK}(W^{DKV}\mathbf h_t)$，$W^{DKV}\in\R^{512\times5120}$，$W^{UK}\in\R^{16384\times512}$ [@dsv2]。等效矩阵 $W^{UK}W^{DKV}$ 是 $[16384,5120]$，但由 (M2.3)，它的秩至多 512。所以 128 个头、每层 16384 个键分量，全部落在一个至多 512 维的子空间里；由 (M2.2)，还有至少 $5120-512=4608$ 维的输入方向对所有键都不可见。

值也一样：把 $W^{UK}$、$W^{UV}$ 上下叠成 $[32768,512]$ 的矩阵再乘 $W^{DKV}$，秩仍至多 512。键和值共享同一个 512 维的来源，这就是论文所说的“联合”压缩（joint compression）[@dsv2]。

这里按论文的式子略去了一个细节：DeepSeek-V2 在 $\mathbf c_t^{KV}$ 之后还加了一个 RMSNorm [@dsv2]，它把 $\mathbf c_t^{KV}$ 除以自己的均方根，再逐分量乘一个学出来的系数。加上它以后，从 $\mathbf h_t$ 到键、值不再是线性映射，“等效矩阵”是略去归一化的说法；但键和值仍是归一化后那 512 个数经 $W^{UK}$、$W^{UV}$ 的线性组合，仍落在至多 512 维的列空间里，$W^{DKV}$ 零空间里的输入方向也仍然看不见。上面的结论都不变。
:::

::: warn 失败模式：秩在浮点数里是“脆”的
秩是整数，对矩阵元素的微小扰动极其敏感。把 $M$ 右下角的 3 改成 3.1，或给每个元素加上 $10^{-6}$ 量级的噪声，精确的秩都从 2 变成 3，可矩阵“几乎”仍只有两个方向（§3 会看到，改成 3.1 后第三个方向的拉伸系数只有 0.0232，前两个是 4.19 和 1.03）。所以数值软件先做 SVD，再把小于阈值的奇异值当成 0：NumPy 的 `matrix_rank` 默认阈值是“最大奇异值 $\times\max(m,n)\times$ 机器精度” [@numpy-matrix-rank]，它对精确的 $M$ 报告 2（第三个奇异值算出来是 $6.9\times10^{-17}$），对加了噪声的 $M$ 报告 3。实际要问的是“有几个方向大到值得保留”，这需要一把连续的尺子：奇异值。
:::

:::: quiz 小练习：秩与零空间
$$B=\begin{pmatrix}1&2&3\\2&4&6\end{pmatrix}.$$

（1）求 $\rank B$ 与 $N(B)$ 的维数，写出零空间的一组基。（2）$B^\top B$ 是 $3\times3$ 的，它的秩是多少？
::: answer 看答案
（1）第二行是第一行的 2 倍，行秩 1，所以 $\rank B=1$。由 (M2.2)，$\dim N(B)=3-1=2$。零空间就是 $x_1+2x_2+3x_3=0$（第二个方程是它的 2 倍），一组基：$(-2,1,0)$ 与 $(-3,0,1)$。核对：$-2+2=0$，$-3+3=0$。

（2）由 (M2.3)，$\rank(B^\top B)\le\rank B=1$；而 $B^\top B\ne0$（它的左上角是 $1^2+2^2=5$），所以秩恰好是 1。更一般地 $\rank(A^\top A)=\rank A$（Axler 7.64(d)），§3 会用到。
:::
::::

::: note 这一页用在哪里
[第 9 课 §2](../deepseek/lesson-09-v2-mla.html#p2)：“低秩 KV 联合压缩”里的“低秩”就是 (M2.3)：键和值的等效矩阵秩至多 $d_c=512$。[M01 §6](m01-vectors-matrices.html#p6)：查询矩阵的零空间有多大。
:::

::: key 本页要点
- 矩阵的输出都是列的线性组合；所有输出组成列空间 $C(A)$，它的维数就是秩。秩 $\le\min(m,n)$，行秩 = 列秩。
- 零空间 $N(A)$ 是被送到 0 的方向；秩–零化度：$n=\rank A+\dim N(A)$。
- $\rank(AB)\le\min(\rank A,\rank B)$：MLA 的键和值的等效矩阵虽有 $16384\times5120$，秩至多 512。
- 精确的秩对扰动极敏感；实际要问的是“有几个方向足够大”，这要靠奇异值（§3）。
:::

# 低秩分解 W = UV | 把一张大表拆成两块瘦矩阵：参数账、FLOPs 账与不唯一性 | 20

上一页的 $M=CR$ 已经给出了一个具体的拆法。这一页把它变成一条一般规律：**秩不超过 $r$ 的矩阵，恰好就是能写成 $[m,r]\times[r,n]$ 乘积的矩阵**。然后算账：存两块瘦矩阵比存一整张表省多少，用它乘向量又省多少；最后说明这种拆法有无穷多种，这会影响“把分解学出来”时的训练。

## 秩 ≤ r，当且仅当能写成 W = UV

**含义。** 把一个 $m\times n$ 的矩阵写成一个 $m\times r$ 的矩阵乘一个 $r\times n$ 的矩阵，叫**低秩分解**（low-rank factorization），$r$ 叫它的中间维。

**理解。** 按结合律 $W\mathbf x=U(V\mathbf x)$：输入先被 $V$ 压成 $r$ 个数，再由 $U$ 从这 $r$ 个数展开成 $m$ 个输出。所有信息都要挤过一个只有 $r$ 个数宽的“瓶颈”，所以输出最多只能朝 $r$ 个独立方向伸展。§1 的 $M=CR$ 就是 $r=2$ 的例子：3 个输入先压成 2 个数，再展开成 3 个输出。

**数学。** 下面的 (M2.4) 说这种写法与“秩不超过 $r$”完全等价。

::: eq 低秩分解 | (M2.4)
$$\rank W\le r\quad\Longleftrightarrow\quad W=UV,\qquad U\in\R^{m\times r},\ V\in\R^{r\times n}$$
$U$ 是“瘦高”的（$r$ 列），$V$ 是“矮胖”的（$r$ 行）。$r$ 远小于 $m$、$n$ 时，称 $W$ 是**低秩**（low-rank）的，$W=UV$ 是它的一个**低秩分解**。
:::

::: deriv 两个方向
$\rank W=c\le r\ \Rightarrow\ W=CR,\qquad C\in\R^{m\times c},\ R\in\R^{c\times n}$ || 上一页的列–行分解（Axler 3.56）
$U=[\,C\ \ O\,]\in\R^{m\times r},\qquad V=\begin{bmatrix}R\\ O\end{bmatrix}\in\R^{r\times n}$ || 补 $r-c$ 个零列、零行，$O$ 是全零块
$UV=CR+OO=CR=W$ || 分块乘法 (M1.12)：零块乘零块为 0
$W=UV\ \Rightarrow\ \rank W\le\rank U\le r$ || (M2.3)，以及 $U$ 只有 $r$ 列
:::

用 [M01 (M1.9)](m01-vectors-matrices.html#p2) 的“外积之和”读法，同一件事还有一种写法：

::: eq 秩 r 矩阵 = r 个秩 1 矩阵之和 | (M2.5)
$$W=UV=\sum_{i=1}^{r}\mathbf u_i\,\mathbf v_i^{\top},\qquad \mathbf u_i=U\text{ 的第 }i\text{ 列},\ \ \mathbf v_i^\top=V\text{ 的第 }i\text{ 行}$$
每个 $\mathbf u_i\mathbf v_i^\top$ 是一张“一列乘一行”的表；秩 $r$ 的矩阵至少要 $r$ 张这样的表才能叠出来，少一张都不行（否则由 (M2.3) 秩就小于 $r$ 了）。
:::

::: ana 类比：MLA = 速记
**映射。** 第 9 课把 KV cache 比作会议纪要本，把 MLA 比作速记。在本页的语言里：$V=W^{DKV}$ 是“怎样把一段发言记成速记符号”（压成 $r=512$ 个数），$U=W^{UK}$ 是“怎样把速记展开成完整的标签”；$W=UV$ 的秩至多 $r$，说的是速记符号只有 $r$ 个“笔画”，展开后的内容不可能比这 $r$ 个笔画包含更多独立的信息。

**收益。** 它解释了缓存为什么能小：本子上只记 $r$ 个数，展开规则 $U$ 对所有 token 都一样，只存一份。

**失效的地方。** 真正的速记员是先听到完整发言、再决定怎么记；MLA 的 $V$ 是一个固定的线性映射，对每个 token 用同一套规则，而且它是和模型其余部分一起**训练**出来的，不是先有完整纪要再事后压缩（§6 专门讲这一区别）。另外，速记会丢信息：秩 $r$ 的映射有一个至少 $n-r$ 维的零空间，那些方向的差别在速记里完全消失。
:::

## 参数账与 FLOPs 账

存一整张 $[m,n]$ 的表要 $mn$ 个数；存 $U$、$V$ 两块只要 $mr+rn=r(m+n)$ 个数。用它乘一个向量时，按结合律先算 $V\mathbf x$（$2rn$ 次运算）再算 $U(V\mathbf x)$（$2mr$ 次），合计 $2r(m+n)$，而直接乘整张表要 $2mn$（[M01 §5](m01-vectors-matrices.html#p5) 的 $2mnk$）。两本账的比值相同：

::: eq 低秩分解的参数与 FLOPs | (M2.6)
$$\frac{\text{分解后}}{\text{整张表}}=\frac{r(m+n)}{mn},\qquad \text{不省钱的临界秩}\ \ r^*=\frac{mn}{m+n}$$
$r<r^*$ 才省。方阵 $m=n$ 时 $r^*=n/2$：秩超过一半，拆开反而更贵。
:::

::: ex DeepSeek-V2 键投影的账
取 $m=n_hd_h=16384$、$n=d=5120$、$r=d_c=512$ [@dsv2]：

- 整张表：$16384\times5120=83{,}886{,}080$ 个参数。
- 两块瘦矩阵：$W^{UK}$ 有 $16384\times512=8{,}388{,}608$，$W^{DKV}$ 有 $512\times5120=2{,}621{,}440$，合计 $11{,}010{,}048$，是整张表的 $13.1\%$，少 7.62 倍。
- 临界秩：$r^*=83{,}886{,}080/21{,}504\approx3901$。$d_c=512$ 远在它下面。
- 键和值一起算：两张整表 $167{,}772{,}160$，而 MLA 共用一个 $W^{DKV}$，只需 $2{,}621{,}440+2\times8{,}388{,}608=19{,}398{,}656$，少 8.65 倍。“联合”压缩又省了一份下投影。
- 每个 token 的 FLOPs：$2\times83{,}886{,}080\approx1.68\times10^8$ 对比 $2\times11{,}010{,}048\approx2.20\times10^7$。
:::

::: fig figures/m02-factorization.svg | 图 M2.2 低秩分解的形状与参数账（按 DeepSeek-V2 的真实维度等比例绘制）
**看面积：左边整张表的面积与右边两块细条的面积之比，就是参数量之比 83.9M : 11.0M。** ① 把等效的键投影存成一整张 $[16384,5120]$ 的表。② 存成 $W^{UK}$（竖条，$[16384,512]$）和 $W^{DKV}$（横条，$[512,5120]$）；虚线框是它们的乘积，形状和 ① 一样，但秩至多 512，而且从来不被显式造出来。③ 一个 token 的计算，柱高正比于向量长度：5120 维的 $\mathbf h_t$ 先压成 512 维的 $\mathbf c_t^{KV}$（橙色，课程里压缩潜向量的颜色），再展开成 16384 维的键。每一步的 FLOPs 按 $2mn$ 算出；最下面一行是“先把乘积造出来”的代价，比两步本身贵三个数量级以上。维度取自 DeepSeek-V2 [@dsv2]，1 像素 = 80 行或列。
:::

::: warn MLA 真正省下的是缓存，不是参数
参数少 7.62 倍固然好，但低秩分解对 MLA 真正关键的收益在推理时：**缓存里存的是中间那 512 个数**，而不是展开后的 16384 个键分量和 16384 个值分量。按第 9 课的账，每层每个 token 缓存 $d_c+d_h^R=512+64=576$ 个数（512 维潜向量，加上一小段 64 维、单独携带位置信息的 RoPE 键，[第 9 课 §5](../deepseek/lesson-09-v2-mla.html#p5)），普通多头注意力（MHA）要存全部展开的键和值 $2n_hd_h=32{,}768$ 个，约 57 倍（[第 9 课 §8](../deepseek/lesson-09-v2-mla.html#p8)）。只有在“中间向量可以直接拿来用、不必展开”时，这笔账才成立，那是本课 §7 结合律的事。
:::

## 分解不唯一

同一个 $W$ 有无穷多种写成 $UV$ 的方法。取任何一个可逆的 $r\times r$ 矩阵 $G$：

::: eq 分解的自由度 | (M2.7)
$$W=UV=(UG)(G^{-1}V)\qquad\text{对任意可逆的 }G\in\R^{r\times r}$$
理由：结合律 (M1.21) 与 $GG^{-1}=I$。中间那 $r$ 个数（$V\mathbf x$）的“坐标系”可以任意换，只要 $U$ 反向补偿。
:::

::: ex 同一个 M，两种拆法
取 $G=\begin{pmatrix}2&0\\0&1\end{pmatrix}$，$G^{-1}=\begin{pmatrix}1/2&0\\0&1\end{pmatrix}$。上一页 $M=CR$，那么

$$M=(CG)(G^{-1}R)=\begin{pmatrix}2&0\\0&1\\2&1\end{pmatrix}\begin{pmatrix}1/2&0&1/2\\0&1&2\end{pmatrix}.$$

核对第三行第三列：$2\cdot\tfrac12+1\cdot2=3$。中间向量 $G^{-1}R\,\mathbf x$ 的第一个分量是原来的一半，展开时 $CG$ 再把它放大 2 倍，输出一模一样。
:::

这在“把分解学出来”时有实际后果：“$U$ 放大、$V$ 缩小”完全不改变乘积，损失对它毫无约束，中间向量的尺度可能随训练漂移。DeepSeek-V2 在压缩后的潜向量上额外加了 RMSNorm，并在这些“宽度瓶颈”处乘缩放因子，理由是低秩压缩（以及细粒度专家切分）会影响层输出的尺度，这样做是为了训练稳定 [@dsv2]；把它和 (M2.7) 联系起来是笔者的解读。SVD 则从无穷多种拆法里挑出一种标准的：中间坐标两两垂直、按重要性排序（下一页）。

::: warn 失败模式：秩 r 是一道硬天花板
$W=UV$ 的输出永远落在 $U$ 的 $r$ 列张成的子空间里，零空间至少 $n-r$ 维；真正需要的映射若有更多独立方向，怎样挑 $U$、$V$ 都有误差，误差最少多少由 §5 回答。另一面，$r$ 超过 $r^*$ 就连钱都省不了。DeepSeek-V2 的查询压缩 $d_c'=1536$ 只让参数少 2.54 倍（练习 5），论文说它是为了训练时省激活显存，并不减少 KV cache [@dsv2]：低秩分解在不同位置服务不同目的。
:::

:::: quiz 小练习：值不值得拆
$W\in\R^{4096\times4096}$。（1）若 $\rank W=256$，拆成 $UV$ 后参数是原来的百分之几？（2）秩多大时拆开就不省了？（3）某个 $[4096,1024]$ 的矩阵呢？
::: answer 看答案
（1）$\dfrac{256\times(4096+4096)}{4096\times4096}=\dfrac{2{,}097{,}152}{16{,}777{,}216}=12.5\%$。

（2）$r^*=\dfrac{4096\times4096}{8192}=2048$，秩到一半就不省了。

（3）$r^*=\dfrac{4096\times1024}{5120}=819.2$。一个 $[4096,1024]$ 矩阵的秩最多 1024，只有秩 $\le819$ 时拆开才省。越“扁”的矩阵，拆分的余地越小。
:::
::::

::: note 这一页用在哪里
[第 9 课 §2](../deepseek/lesson-09-v2-mla.html#p2) 的式 (9.6) 就是 (M2.4)：$W^{UK}W^{DKV}$ 的 $U=W^{UK}$、$V=W^{DKV}$、$r=d_c$。[第 9 课 §6](../deepseek/lesson-09-v2-mla.html#p6)：查询压缩 $W^{UQ}W^{DQ}$ 是另一个低秩分解，$r=d_c'=1536$。[第 18 课 §3](../deepseek/lesson-18-v4.html#p3)：V4 保留了查询的低秩压缩，本课 §7 还会讲它的分组输出投影。
:::

::: key 本页要点
- $\rank W\le r\iff W=UV$（$[m,r]\times[r,n]$）$\iff W$ 是至多 $r$ 个秩 1 矩阵之和。
- 参数与 FLOPs 都从 $mn$ 降到 $r(m+n)$；临界秩 $r^*=mn/(m+n)$。V2 的键投影：83.9M → 11.0M，少 7.62 倍。
- MLA 的主要收益是缓存只存中间的 $r$ 个数；这要求中间向量能直接用（§7）。
- 分解不唯一：$(UG)(G^{-1}V)$；学出来的分解有尺度漂移的自由度，SVD 给出一种标准拆法。
:::

# 奇异值分解（SVD） | 任何矩阵都是“旋转、拉伸、再旋转”：手算 2×2，数值 3×3 | 30

前两页留下两个问题。§1 的最后说：精确的秩太“脆”，我们需要一把连续的尺子，量出每个方向有多大。§2 的最后说：同一个低秩矩阵有无穷多种 $UV$ 拆法，最好有一种标准的。**奇异值分解**（singular value decomposition，SVD）同时回答这两个问题：它给任何矩阵找出一组输入方向、一组输出方向和一串非负的“拉伸系数”，方向两两垂直，系数从大到小排好。这一页先给图景，再给定理，然后从定理推出计算方法，手算一个 $2\times2$，读一个 $3\times3$ 的数值结果。

## 图景：圆变成椭圆

取 $A=\begin{pmatrix}3&0\\4&5\end{pmatrix}$，把单位圆上的每个点都乘上 $A$，得到一个椭圆（图 M2.3 最右）。任何一个椭圆都有一条最长的轴和一条最短的轴，两者互相垂直。记长半轴、短半轴的长度为 $\sigma_1\ge\sigma_2$，方向为单位向量 $\mathbf u_1$、$\mathbf u_2$；再记单位圆上**恰好**被送到长轴端点、短轴端点的那两个输入为 $\mathbf v_1$、$\mathbf v_2$。SVD 定理说的核心事实是：**$\mathbf v_1$ 和 $\mathbf v_2$ 也互相垂直**。于是 $A$ 对单位圆做的事可以拆成三步：先把 $\mathbf v_1,\mathbf v_2$ 转到坐标轴上，再沿两根坐标轴分别拉伸 $\sigma_1$、$\sigma_2$ 倍，最后把两根坐标轴转到 $\mathbf u_1,\mathbf u_2$ 的方向。

::: pre 正交矩阵 | 知道 $Q^\top Q=I$ 的几何意义的可以跳过
一个方阵 $Q$ 的各列长度为 1、两两垂直，就叫**正交矩阵**（orthogonal matrix），等价于 $Q^\top Q=I$（[M01 §6](m01-vectors-matrices.html#p6) 的“正交归一”，这里列数等于行数）。于是 $Q^{-1}=Q^\top$。它保持长度与夹角：$\norm{Q\mathbf x}^2=\mathbf x^\top Q^\top Q\mathbf x=\norm{\mathbf x}^2$。二维里正交矩阵只有两种：旋转，或旋转加一次镜像翻转（行列式为 $-1$）。[M03](m03-rotation-complex.html) 专讲旋转矩阵。本课说“旋转”时，都允许其中含一次翻转。
:::

::: fig figures/m02-svd-steps.svg | 图 M2.3 A = UΣVᵀ：旋转、拉伸、再旋转（A = [[3, 0], [4, 5]]）
**看黑点：同一个输入 $\mathbf e_1=(1,0)$ 依次经过三步，最后落在 $(3,4)$，正是 $A\mathbf e_1$（$A$ 的第一列）。** ① 单位圆上两个互相垂直的方向 $\mathbf v_1$（蓝）、$\mathbf v_2$（紫）。② 乘 $V^\top$：整个平面旋转 $-45^\circ$，$\mathbf v_1,\mathbf v_2$ 转到两根坐标轴上，圆还是圆。③ 乘 $\Sigma$：沿横轴拉伸 $\sigma_1=6.71$ 倍、沿纵轴拉伸 $\sigma_2=2.24$ 倍，圆变成摆正的椭圆。④ 乘 $U$：旋转 $+71.6^\circ$，椭圆的长轴转到 $\mathbf u_1$ 方向。③④ 的比例尺是 ①② 的 1/3。所有坐标由脚本按下文手算的 $U,\Sigma,V$ 算出并用 $U\Sigma V^\top=A$ 核对。
:::

**图景的边界。** 高维里椭圆变成椭球，道理相同但看不见了；“旋转”可能带一次镜像翻转；两个 $\sigma$ 相等时（圆被均匀放大）长短轴没有确定的方向，$\mathbf u_i,\mathbf v_i$ 也就不唯一。

## 定理

::: thm 奇异值分解（存在性）
设 $A\in\R^{m\times n}$，秩为 $r$。则存在正交矩阵 $U\in\R^{m\times m}$、$V\in\R^{n\times n}$，以及 $m\times n$ 的“对角”矩阵 $\Sigma$（只有 $\Sigma_{ii}=\sigma_i$ 可能非零），使得 $A=U\Sigma V^\top$，其中 $\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0$，其余 $\sigma_i=0$（MML 定理 4.22 [@mml2020]）。只保留 $r$ 个正的奇异值，就是紧凑形式 $A=U_r\Sigma_rV_r^\top$，$U_r\in\R^{m\times r}$、$V_r\in\R^{n\times r}$ 的列正交归一，$\Sigma_r$ 是 $r\times r$ 的正对角阵（Axler 7.70、7.80 [@axler2024]）。
:::

名字：$\sigma_i$ 叫**奇异值**（singular value），$U$ 的列 $\mathbf u_i$ 叫**左奇异向量**，$V$ 的列 $\mathbf v_i$ 叫**右奇异向量**（这个分解的先驱有 Beltrami、Jordan、Sylvester、Schmidt、Weyl 五位 [@stewart1993]）。按 M01 的“外积之和”读法展开，SVD 就是把 $A$ 写成 $r$ 个秩 1 矩阵之和，每一个都带着一个“分量”：

::: eq 奇异值分解 | (M2.8)
$$A=U\Sigma V^\top=\sum_{i=1}^{r}\sigma_i\,\mathbf u_i\mathbf v_i^\top,\qquad \mathbf u_i^\top\mathbf u_j=\mathbf v_i^\top\mathbf v_j=\delta_{ij}$$
$\delta_{ij}$ 在 $i=j$ 时为 1、否则为 0；$\sigma_1\ge\cdots\ge\sigma_r>0$。与 (M2.5) 的区别：这里的 $\mathbf u_i$ 两两垂直、$\mathbf v_i$ 两两垂直，每一项的“大小”由 $\sigma_i$ 单独标出、从大到小排列。
:::

把 (M2.8) 作用在 $\mathbf v_j$ 上，由于 $\mathbf v_i^\top\mathbf v_j$ 只在 $i=j$ 时为 1，求和只剩一项：

::: eq 每一对奇异向量 | (M2.9)
$$A\mathbf v_i=\sigma_i\,\mathbf u_i,\qquad A^\top\mathbf u_i=\sigma_i\,\mathbf v_i,\qquad i=1,\dots,r;\qquad A\mathbf v_i=\mathbf 0,\ \ i>r$$
第一式就是图景：输入方向 $\mathbf v_i$ 被送到输出方向 $\mathbf u_i$，长度乘 $\sigma_i$。第二式对 $A^\top=V\Sigma^\top U^\top$ 同理。$i>r$ 的 $\mathbf v_i$ 组成零空间的一组正交基。
:::

两个推论。**秩 = 正奇异值的个数**：由 (M2.8)，每个输出都是 $\mathbf u_1,\dots,\mathbf u_r$ 的组合，而每个 $\mathbf u_i=A\mathbf v_i/\sigma_i$ 本身是输出，所以 $C(A)$ 由这 $r$ 个两两垂直的向量张成；两两垂直的单位向量一定线性无关（若 $\sum_ic_i\mathbf u_i=\mathbf 0$，两边点乘 $\mathbf u_j$ 得 $c_j=0$），它们是 $C(A)$ 的一组基，维数恰好是 $r$。**SVD 是一种标准的低秩分解**：令 $U\leftarrow U_r\Sigma_r$、$V\leftarrow V_r^\top$，(M2.8) 就是 §2 的 $W=UV$，而且是 (M2.7) 无穷多种拆法里中间坐标两两垂直、按 $\sigma_i$ 排序的那一种。

::: eq SVD 作为低秩分解 | (M2.10)
$$A=\underbrace{(U_r\Sigma_r)}_{[m,r]}\ \underbrace{V_r^\top}_{[r,n]}$$
中间的 $r$ 个数 $V_r^\top\mathbf x$ 是输入在 $\mathbf v_1,\dots,\mathbf v_r$ 上的坐标（M01 (M1.19) 的投影坐标）；$U_r\Sigma_r$ 把每个坐标乘 $\sigma_i$ 后放到 $\mathbf u_i$ 方向。
:::

## 奇异值从哪里来：$A^\top A$

定理告诉我们 SVD 存在；怎么把它算出来？关键是 $A^\top A$。

::: pre 特征值与特征向量 | 会算 2×2 矩阵特征值的可以跳过
若方阵 $S$ 和非零向量 $\mathbf v$ 满足 $S\mathbf v=\lambda\mathbf v$，就说 $\mathbf v$ 是 $S$ 的**特征向量**、$\lambda$ 是对应的**特征值**（eigenvalue）：$S$ 只把这个方向拉长或缩短，不改变方向。$S\mathbf v=\lambda\mathbf v$ 即 $(S-\lambda I)\mathbf v=\mathbf 0$ 有非零解，也就是 $S-\lambda I$ 的秩小于 2（对 $2\times2$），即两列平行，等价于行列式为 0（$2\times2$ 时 $\det\begin{pmatrix}a&b\\c&d\end{pmatrix}=ad-bc$，就是 [M01 图 M1.3](m01-vectors-matrices.html#p2) 里的面积倍数）。$2\times2$ 时展开就是 $\lambda^2-(\text{迹})\,\lambda+\det S=0$，迹是对角线之和。**对称**矩阵（$S^\top=S$）有一条关键性质：可以找到由特征向量组成的一组正交归一基，特征值都是实数（实谱定理，Axler 7.29）。其中“不同特征值的特征向量互相垂直”一行就能看出：$\lambda_1\mathbf v_1^\top\mathbf v_2=(S\mathbf v_1)^\top\mathbf v_2=\mathbf v_1^\top S\mathbf v_2=\lambda_2\mathbf v_1^\top\mathbf v_2$，$\lambda_1\ne\lambda_2$ 时只能 $\mathbf v_1^\top\mathbf v_2=0$。
:::

::: deriv 从 SVD 推出 $A^\top A$ 的特征分解
$A^\top A=(U\Sigma V^\top)^\top(U\Sigma V^\top)=V\Sigma^\top U^\top U\Sigma V^\top$ || 转置规则 (M1.10)
$=V\,(\Sigma^\top\Sigma)\,V^\top$ || $U^\top U=I$
$A^\top A\,\mathbf v_i=V(\Sigma^\top\Sigma)V^\top\mathbf v_i=\sigma_i^2\,\mathbf v_i$ || $V^\top\mathbf v_i=\mathbf e_i$，$\Sigma^\top\Sigma$ 是对角阵，第 $i$ 个对角元为 $\sigma_i^2$
$\mathbf u_i=A\mathbf v_i/\sigma_i$ || (M2.9)，$\sigma_i>0$
:::

::: eq 计算 SVD 的三步 | (M2.11)
$$A^\top A\,\mathbf v_i=\sigma_i^2\,\mathbf v_i,\qquad \sigma_i=\sqrt{\lambda_i(A^\top A)},\qquad \mathbf u_i=\frac{A\mathbf v_i}{\sigma_i}$$
右奇异向量是 $A^\top A$ 的特征向量，奇异值是其特征值的平方根，左奇异向量由 $A\mathbf v_i$ 归一化得到。
:::

反过来，这三步也证明了 SVD 存在，这正是 Axler 7.70 的证明：$A^\top A$ 对称，所以由实谱定理它有一组正交归一的特征向量 $\mathbf v_1,\dots,\mathbf v_n$；特征值都 $\ge0$，因为 $\lambda_i=\mathbf v_i^\top A^\top A\mathbf v_i=\norm{A\mathbf v_i}^2$。把特征值从大到小排好，设其中有 $r$ 个为正（练习 9 说明 $r$ 恰好是 $\rank A$）。$\lambda_i=0$ 的方向满足 $\norm{A\mathbf v_i}^2=0$，即 $A\mathbf v_i=\mathbf 0$；对 $\lambda_i>0$ 的方向，令 $\sigma_i=\sqrt{\lambda_i}$、$\mathbf u_i=A\mathbf v_i/\sigma_i$，只需再验证这 $r$ 个 $\mathbf u_i$ 两两垂直：

::: deriv 左奇异向量自动两两垂直（Axler 7.70）
$\mathbf u_j^\top\mathbf u_k=\dfrac{(A\mathbf v_j)^\top(A\mathbf v_k)}{\sigma_j\sigma_k}=\dfrac{\mathbf v_j^\top(A^\top A\mathbf v_k)}{\sigma_j\sigma_k}$ || $\mathbf u$ 的定义；转置规则
$=\dfrac{\sigma_k^2\,\mathbf v_j^\top\mathbf v_k}{\sigma_j\sigma_k}=\begin{cases}1&j=k\\0&j\ne k\end{cases}$ || $\mathbf v_k$ 是特征向量；$\mathbf v_j$ 正交归一
:::

最后一步：$A$ 与 $\sum_{i\le r}\sigma_i\mathbf u_i\mathbf v_i^\top$ 在基 $\mathbf v_1,\dots,\mathbf v_n$ 的每一个向量上取值都相同（$i\le r$ 时都是 $\sigma_i\mathbf u_i$，$i>r$ 时都是 $\mathbf 0$），两个矩阵在一组基上作用相同就处处相同，所以 $A=\sum_{i\le r}\sigma_i\mathbf u_i\mathbf v_i^\top$。把 $\mathbf u_1,\dots,\mathbf u_r$ 扩充成 $\R^m$ 的一组正交归一基（Axler 6.36），就得到定理里完整的 $U$。

## 手算一个 2×2

::: ex A = [[3, 0], [4, 5]] 的 SVD，每一步都能用纸笔核对
**第 1 步：$A^\top A$。** $A^\top=\begin{pmatrix}3&4\\0&5\end{pmatrix}$，

$$A^\top A=\begin{pmatrix}3\cdot3+4\cdot4&3\cdot0+4\cdot5\\0\cdot3+5\cdot4&0\cdot0+5\cdot5\end{pmatrix}=\begin{pmatrix}25&20\\20&25\end{pmatrix}.$$

**第 2 步：特征值。** 迹 50，行列式 $25\cdot25-20\cdot20=225$，方程 $\lambda^2-50\lambda+225=0$，$\lambda=25\pm\sqrt{625-225}=25\pm20$，即 $\lambda_1=45$、$\lambda_2=5$。奇异值 $\sigma_1=\sqrt{45}=3\sqrt5\approx6.7082$，$\sigma_2=\sqrt5\approx2.2361$。

**第 3 步：右奇异向量。** $\lambda_1=45$：$(A^\top A-45I)\mathbf v=\begin{pmatrix}-20&20\\20&-20\end{pmatrix}\mathbf v=\mathbf 0$，得 $\mathbf v_1=\tfrac1{\sqrt2}(1,1)$。$\lambda_2=5$：$\begin{pmatrix}20&20\\20&20\end{pmatrix}\mathbf v=\mathbf 0$，得 $\mathbf v_2=\tfrac1{\sqrt2}(-1,1)$。两者垂直，与预备知识框里的结论一致。

**第 4 步：左奇异向量。** $A\mathbf v_1=\tfrac1{\sqrt2}(3,\,4+5)=\tfrac1{\sqrt2}(3,9)$，除以 $3\sqrt5$：$\mathbf u_1=\tfrac1{\sqrt{10}}(1,3)\approx(0.3162,0.9487)$。$A\mathbf v_2=\tfrac1{\sqrt2}(-3,\,-4+5)=\tfrac1{\sqrt2}(-3,1)$，除以 $\sqrt5$：$\mathbf u_2=\tfrac1{\sqrt{10}}(-3,1)$。核对：$\mathbf u_1^\top\mathbf u_2=(-3+3)/10=0$，长度都是 1。

**第 5 步：拼回去。** 按 (M2.8)：

$$\sigma_1\mathbf u_1\mathbf v_1^\top=\frac{3\sqrt5}{\sqrt{20}}\begin{pmatrix}1&1\\3&3\end{pmatrix}=\begin{pmatrix}1.5&1.5\\4.5&4.5\end{pmatrix},$$

$$\sigma_2\mathbf u_2\mathbf v_2^\top=\frac{\sqrt5}{\sqrt{20}}\begin{pmatrix}3&-3\\-1&1\end{pmatrix}=\begin{pmatrix}1.5&-1.5\\-0.5&0.5\end{pmatrix},$$

两者相加正好是 $\begin{pmatrix}3&0\\4&5\end{pmatrix}$。

**几何。** $\mathbf v_1$ 在 $45^\circ$，所以 $V^\top$ 是旋转 $-45^\circ$；$\mathbf u_1$ 在 $\arctan3\approx71.57^\circ$，$U=\tfrac1{\sqrt{10}}\begin{pmatrix}1&-3\\3&1\end{pmatrix}$ 是旋转 $+71.57^\circ$（行列式 $+1$）。这就是图 M2.3 的三步。
:::

::: warn 奇异值不是特征值
$A$ 是下三角矩阵，特征值是对角线上的 3 和 5；奇异值却是 6.71 和 2.24，沿 $\mathbf v_1$ 方向它把向量拉长 6.71 倍，比最大的特征值还大。特征值回答“哪个方向被映到自己身上”（$A\mathbf v=\lambda\mathbf v$），奇异值回答“最多、最少拉长多少”（$A\mathbf v_i=\sigma_i\mathbf u_i$）。两者只在对称半正定矩阵上一致；说“矩阵最多放大多少”要看 $\sigma_1$。两者的联系之一：$\det A=15=3\times5=\sigma_1\sigma_2$。
:::

## 读一个数值 SVD：3×3 的 M

::: ex numpy.linalg.svd(M) 的输出，逐项解读
§1 的 $M=\begin{pmatrix}1&0&1\\0&1&2\\1&1&3\end{pmatrix}$。NumPy 给出（保留 4 位小数；`Vt` 就是 $V^\top$，它的**行**是右奇异向量）：

$$U=\begin{pmatrix}-0.2852&0.7651&-0.5774\\-0.5199&-0.6295&-0.5774\\-0.8052&0.1355&0.5774\end{pmatrix},\qquad \boldsymbol\sigma=(4.1155,\ 1.0309,\ 0.0000),$$

$$V^\top=\begin{pmatrix}-0.2650&-0.3220&-0.9089\\0.8736&-0.4792&-0.0849\\-0.4082&-0.8165&0.4082\end{pmatrix}.$$

- **秩。** 两个正的奇异值，第三个算出来是 $6.9\times10^{-17}$（浮点误差），$\rank M=2$，与 §1 一致。
- **精确值。** $M^\top M=\begin{pmatrix}2&1&4\\1&2&5\\4&5&14\end{pmatrix}$，特征多项式 $\lambda^3-18\lambda^2+18\lambda=0$（迹 18；三个 $2\times2$ 主子式 $3+12+3=18$；行列式 $(\det M)^2=0$），所以 $\sigma_{1,2}^2=9\pm3\sqrt7$，$\sigma_1=\sqrt{9+3\sqrt7}\approx4.1155$，$\sigma_2=\sqrt{9-3\sqrt7}\approx1.0309$。
- **核对 (M2.9)。** $M\mathbf v_1=M(-0.2650,-0.3220,-0.9089)=(-1.1739,-2.1398,-3.3137)$，而 $\sigma_1\mathbf u_1=4.1155\times(-0.2852,-0.5199,-0.8052)=(-1.1737,-2.1396,-3.3138)$，第 4 位小数上 1 到 2 个单位的差别来自四舍五入（用 NumPy 的全精度数计算，两边都是 $(-1.1739,-2.1398,-3.3137)$）。
- **零空间与它的“对面”。** 第三个右奇异向量 $\mathbf v_3=(-0.4082,-0.8165,0.4082)=-\tfrac1{\sqrt6}(1,2,-1)$，正是 §1 求出的零空间方向。第三个左奇异向量 $\mathbf u_3=-\tfrac1{\sqrt3}(1,1,-1)$ 垂直于列空间 $z=x+y$（这个平面的法向量就是 $(1,1,-1)$）。所以 SVD 顺带给出了列空间（$\mathbf u_1,\mathbf u_2$）、零空间（$\mathbf v_3$）以及它们各自垂直补空间的正交基。
- **符号。** 手算时你可能得到 $+\tfrac1{\sqrt6}(1,2,-1)$。把一对 $(\mathbf u_i,\mathbf v_i)$ 同时变号，$\sigma_i\mathbf u_i\mathbf v_i^\top$ 不变，两种答案都对。
- **把 3 改成 3.1。** 奇异值变成 $(4.1892,\ 1.0298,\ 0.0232)$。精确的秩是 3，但第三个方向的拉伸只有第一个的 0.55%。§1 说“需要一把连续的尺子”，这就是它。
:::

::: warn 失败模式：不要用 $A^\top A$ 的特征值去算小奇异值
(M2.11) 是理解 SVD 的正确方式，却不是好的数值算法。$A^\top A$ 的特征值是 $\sigma_i^2$：若 $\sigma_1=1$、$\sigma_2=10^{-8}$，则 $\sigma_2^2=10^{-16}$，已经小于双精度在 1 附近的分辨率（约 $2.2\times10^{-16}$，见 [M08 §4](m08-floating-point.html#p4)），会被舍入误差吞掉。笔者用一个 $\sigma=(1,10^{-8})$ 的 $2\times2$ 矩阵实测：`np.linalg.svd` 给出 $\sigma_2=1.000000\times10^{-8}$；先算 $A^\top A$ 再求特征值开方，得到 $5.27\times10^{-9}$，差了 47%；$\sigma_2=10^{-10}$ 时后者干脆得到 0。数值库的 SVD 不经过 $A^\top A$，所以小奇异值能算准。
:::

:::: quiz 小练习：对角矩阵与秩 1 矩阵的 SVD
（1）$D=\begin{pmatrix}0&0\\0&-2\end{pmatrix}$ 的奇异值是多少？给出一组 $\mathbf u_1,\mathbf v_1$。（2）$B=\begin{pmatrix}1\\2\\2\end{pmatrix}\begin{pmatrix}3&4\end{pmatrix}$ 的非零奇异值是多少？
::: answer 看答案
（1）$D^\top D=\operatorname{diag}(0,4)$，奇异值 $(2,0)$：负号不是“拉伸系数”的一部分。取 $\mathbf v_1=(0,1)$，则 $D\mathbf v_1=(0,-2)$，$\mathbf u_1=(0,-1)$。符号被放进了奇异向量里（也可以取 $\mathbf v_1=(0,-1)$、$\mathbf u_1=(0,1)$）。

（2）$B=\mathbf a\mathbf b^\top$，$\mathbf a=(1,2,2)$，$\mathbf b=(3,4)$，$\norm{\mathbf a}=3$，$\norm{\mathbf b}=5$。写成 $B=15\cdot\dfrac{\mathbf a}{3}\Big(\dfrac{\mathbf b}{5}\Big)^\top$，这已经是 (M2.8) 的形式（只有一项），所以唯一的非零奇异值是 $\norm{\mathbf a}\norm{\mathbf b}=15$。一般地，秩 1 矩阵 $\mathbf a\mathbf b^\top$ 的奇异值是 $\norm{\mathbf a}\norm{\mathbf b}$。
:::
::::

::: note 这一页用在哪里
[第 18 课 §5](../deepseek/lesson-18-v4.html#p5)：DeepSeek-V4 的 Muon 优化器对更新矩阵做牛顿–舒尔茨迭代，按 (M2.8) 看，这种迭代只改动奇异值、不改动 $U$ 和 $V$；V4 用“8 步快速 + 2 步精修”把奇异值稳定在 1，也就是把更新变成近似的 $UV^\top$ [@dsv4]。具体的迭代见 [M07 §6](m07-optimization.html#p6)。[第 9 课](../deepseek/lesson-09-v2-mla.html)：MLA 的矩阵不是用 SVD 得来的（§6），但 SVD 是分析它们的尺子。[M03](m03-rotation-complex.html)：$U$、$V$ 这类正交矩阵的几何。
:::

::: key 本页要点
- 任意 $A=U\Sigma V^\top=\sum_{i\le r}\sigma_i\mathbf u_i\mathbf v_i^\top$：输入方向 $\mathbf v_i$、输出方向 $\mathbf u_i$ 各自两两垂直，$A\mathbf v_i=\sigma_i\mathbf u_i$；几何上是旋转、沿坐标轴拉伸、再旋转。
- 秩 = 正奇异值的个数；SVD 是低秩分解 $W=UV$ 的一种标准取法 $(U_r\Sigma_r)(V_r^\top)$。
- 计算上：$\mathbf v_i$ 是 $A^\top A$ 的特征向量，$\sigma_i=\sqrt{\lambda_i}$，$\mathbf u_i=A\mathbf v_i/\sigma_i$。手算用它，数值上不要用它算小奇异值。
- 奇异值不是特征值；奇异向量有符号自由，奇异值相等时方向也不唯一。
:::

# 两把尺子：谱范数与 Frobenius 范数 | 一个矩阵有多“大”，两个矩阵差多远 | 20

下一页要回答“最好的秩 $k$ 近似是哪一个”，先得有一把尺子量近似的误差 $E=A-B$ 有多大。常用的有两把，回答不同的问题：**Frobenius 范数**问“所有元素加起来有多大”，相当于对所有方向的误差取平均；**谱范数**问“最坏的那个方向被放大多少”。用上一页的 SVD，两者都能写成奇异值的函数。

## Frobenius 范数：把矩阵拉直成一列数

**含义。** 所有元素平方和的平方根。

**理解。** 把 $m\times n$ 个元素排成一个长度为 $mn$ 的向量，量它的长度，就是 [M01 (M1.1)](m01-vectors-matrices.html#p1) 的向量长度。它不关心元素排成什么形状。

::: pre 迹 | 知道 $\tr(XY)=\tr(YX)$ 的可以跳过
方阵的**迹**（trace）是对角线元素之和：$\tr S=\sum_iS_{ii}$。一条常用性质：只要 $XY$ 与 $YX$ 都是方阵，$\tr(XY)=\tr(YX)$。理由是 $\tr(XY)=\sum_i\sum_kX_{ik}Y_{ki}$，交换两个有限求和的次序就是 $\sum_k\sum_iY_{ki}X_{ik}=\tr(YX)$。
:::

::: deriv Frobenius 范数的三种写法
$\norm{A}_F^2=\textstyle\sum_{i,j}a_{ij}^2=\sum_{j}\norm{\mathbf a_j}^2$ || 定义；按列分组，每列贡献它的长度平方
$=\tr(A^\top A)$ || $A^\top A$ 的第 $j$ 个对角元是 $\mathbf a_j^\top\mathbf a_j=\norm{\mathbf a_j}^2$（M01 读法二）
$=\tr\big(V(\Sigma^\top\Sigma)V^\top\big)=\tr\big((\Sigma^\top\Sigma)V^\top V\big)$ || §3 的 $A^\top A=V(\Sigma^\top\Sigma)V^\top$；迹的轮换
$=\tr(\Sigma^\top\Sigma)=\textstyle\sum_i\sigma_i^2$ || $V^\top V=I$
:::

::: eq Frobenius 范数 | (M2.12)
$$\norm{A}_F=\sqrt{\sum_{i,j}a_{ij}^2}=\sqrt{\tr(A^\top A)}=\sqrt{\sigma_1^2+\sigma_2^2+\cdots+\sigma_r^2}$$
元素的平方和等于奇异值的平方和：矩阵的“总能量”按奇异方向分摊，第 $i$ 个方向分到 $\sigma_i^2$。
:::

::: ex 两次核对
$A=\begin{pmatrix}3&0\\4&5\end{pmatrix}$：元素平方和 $9+0+16+25=50$，奇异值平方和 $45+5=50$，$\norm{A}_F=\sqrt{50}\approx7.0711$。

§1 的 $M$：元素平方和 $1+0+1+0+1+4+1+1+9=18$，奇异值平方和 $(9+3\sqrt7)+(9-3\sqrt7)=18$。第一个方向独占 $16.94/18=94.1\%$ 的能量。
:::

Frobenius 范数还有一个很有用的“平均”解释。设输入 $\mathbf x$ 是随机的，各分量独立、均值 0、方差 1（例如标准正态），那么输出长度的平方的**期望**正好是 $\norm{A}_F^2$：

::: pre 期望与 $\E[\mathbf x\mathbf x^\top]=I$ | 学过 M04 的可以跳过
$\E[\cdot]$ 表示“对随机性取平均”（期望，见 [M04 §2](m04-probability.html#p2)）。若 $x_1,\dots,x_n$ 独立、均值 0、方差 1，则 $\E[x_ix_j]$ 在 $i=j$ 时是方差 1，在 $i\ne j$ 时是 $\E[x_i]\E[x_j]=0$。把这 $n^2$ 个数排成矩阵，就是 $\E[\mathbf x\mathbf x^\top]=I$。
:::

::: deriv 随机输入下的平均放大
$\E\norm{A\mathbf x}^2=\E[\mathbf x^\top A^\top A\mathbf x]$ || 长度平方 = 自己点自己
$=\E\big[\tr(A^\top A\,\mathbf x\mathbf x^\top)\big]$ || 一个数等于它自己的迹，再用迹的轮换
$=\tr\big(A^\top A\,\E[\mathbf x\mathbf x^\top]\big)=\tr(A^\top A)$ || 期望是线性的，可以移进迹与矩阵乘法里；$\E[\mathbf x\mathbf x^\top]=I$
:::

::: eq Frobenius 范数 = 各向同性输入下的均方放大 | (M2.13)
$$\E\,\norm{A\mathbf x}^2=\norm{A}_F^2\qquad\text{当 }\E[\mathbf x\mathbf x^\top]=I$$
所以用 Frobenius 范数量误差矩阵 $E$，等于问“对一个各方向同等重要的随机输入，输出平均错多少”。§6 会看到，当输入**不是**各向同性时，这正是需要修改的地方。
:::

笔者用 20 万个标准正态输入实测，$A=\begin{pmatrix}3&0\\4&5\end{pmatrix}$ 的 $\norm{A\mathbf x}^2$ 平均为 50.04（理论值 50）。输入均匀取在单位圆上时 $\E[\mathbf x\mathbf x^\top]=I/2$，平均值 25，均方根 5，即图 M2.4 里黄色虚线的高度。

## 谱范数：最坏方向的放大倍数

**含义。** 一个矩阵最多能把一个向量拉长多少倍。

**理解。** 在图 M2.3 里就是椭圆的长半轴：单位圆上所有点里，被送得最远的那个有多远。它给出一个对**每一个**输入都成立的保证：$\norm{A\mathbf x}\le\norm{A}_2\norm{\mathbf x}$。

**数学。** 定义为 $\norm{A}_2=\max_{\mathbf x\ne\mathbf 0}\norm{A\mathbf x}/\norm{\mathbf x}$。它等于最大奇异值（MML 定义 4.23、定理 4.24 [@mml2020]；Axler 7.85–7.86）。推导只用 SVD：

::: deriv 谱范数 = σ₁
$\mathbf x=\textstyle\sum_ic_i\mathbf v_i,\qquad \norm{\mathbf x}^2=\sum_ic_i^2=1$ || $\mathbf v_1,\dots,\mathbf v_n$ 是 $\R^n$ 的正交归一基；取单位长度的输入
$A\mathbf x=\textstyle\sum_{i\le r}c_i\sigma_i\,\mathbf u_i$ || (M2.9)：$A\mathbf v_i=\sigma_i\mathbf u_i$，$i>r$ 时为 $\mathbf 0$
$\norm{A\mathbf x}^2=\textstyle\sum_{i\le r}c_i^2\sigma_i^2$ || $\mathbf u_i$ 两两垂直、长度为 1：勾股定理
$\le\sigma_1^2\textstyle\sum_ic_i^2=\sigma_1^2$ || 每个 $\sigma_i\le\sigma_1$
$\mathbf x=\mathbf v_1\ \Rightarrow\ \norm{A\mathbf x}=\sigma_1$ || 上界可以取到，所以最大值恰好是 $\sigma_1$
:::

::: eq 谱范数 | (M2.14)
$$\norm{A}_2=\max_{\mathbf x\ne\mathbf 0}\frac{\norm{A\mathbf x}}{\norm{\mathbf x}}=\sigma_1,\qquad \norm{AB}_2\le\norm{A}_2\,\norm{B}_2$$
第二式：$\norm{AB\mathbf x}\le\norm{A}_2\norm{B\mathbf x}\le\norm{A}_2\norm{B}_2\norm{\mathbf x}$。一串矩阵连乘时，每一个的谱范数都 $\le1$，乘积就不会把任何向量放大。
:::

::: fig figures/m02-norms.svg | 图 M2.4 同一个矩阵，两把尺子（A = [[3, 0], [4, 5]]）
**看左图绿线的最高点和最低点：它们正是 $\sigma_1=6.708$ 与 $\sigma_2=2.236$。** 左：单位输入 $\mathbf x=(\cos\theta,\sin\theta)$ 被 $A$ 拉长的倍数，逐点计算。最高点在 $\theta=45^\circ$，就是 §3 手算出的 $\mathbf v_1$；最低点在 $135^\circ$（$\mathbf v_2$）；两个坐标轴方向 $\mathbf e_1,\mathbf e_2$ 都被拉长到 5。谱范数是这条曲线的最大值；黄色虚线是它在整个圆上的均方根 $\norm{A}_F/\sqrt2=5$。右：把 $\norm{A}_F^2=\sigma_1^2+\sigma_2^2$ 画成直角三角形，谱范数是最长的直角边，Frobenius 范数是斜边。
:::

## 两把尺子的关系

由 (M2.12) 和 (M2.14)，两个范数都只依赖奇异值：一个取最大的，一个取平方和。

::: eq 谱范数与 Frobenius 范数的关系 | (M2.15)
$$\sigma_1=\norm{A}_2\ \le\ \norm{A}_F=\sqrt{\textstyle\sum_{i\le r}\sigma_i^2}\ \le\ \sqrt r\,\sigma_1=\sqrt r\,\norm{A}_2$$
理由：$\sigma_1^2\le\sum_{i\le r}\sigma_i^2$（平方和里含 $\sigma_1^2$ 这一项），$\sum_{i\le r}\sigma_i^2\le r\,\sigma_1^2$（$r$ 项，每项 $\le\sigma_1^2$）。左边等号成立当且仅当至多一个非零奇异值（秩 $\le1$）；右边等号成立当且仅当所有 $r$ 个奇异值相等。
:::

两者还有一条共同性质：左乘、右乘正交矩阵都不改变它们（$QAR$ 与 $A$ 奇异值相同，练习 10）。这类范数叫**酉不变范数**（unitarily invariant norm；实数情形也叫正交不变），这是下一页定理的关键前提：尺子不在乎旋转，问题就能旋转到对角阵 $\Sigma$ 上来解决。

| | 谱范数 $\norm{E}_2$ | Frobenius 范数 $\norm{E}_F$ |
|---|---|---|
| 用奇异值表示 | 最大的 $\sigma_1$ | $\sqrt{\sum\sigma_i^2}$ |
| 回答的问题 | 最坏的单个输入错多少 | 各向同性随机输入平均错多少 (M2.13) |
| 典型用途 | 稳定性、连乘不爆炸（[M11](m11-doubly-stochastic-sinkhorn.html)、[第 16 课](../deepseek/lesson-16-mhc.html)） | 最小二乘、训练损失、压缩误差 |
| 计算 | 需要 SVD（或迭代法） | 元素平方和，便宜 |

::: warn 失败模式：Frobenius 一样大，最坏情况差 10 倍
两个 $100\times100$ 的误差矩阵：$E_1=\operatorname{diag}(1,0,\dots,0)$ 与 $E_2=0.1I$。$\norm{E_1}_F=1$，$\norm{E_2}_F=\sqrt{100\times0.01}=1$，Frobenius 范数完全相同。可是 $\norm{E_1}_2=1$、$\norm{E_2}_2=0.1$：$E_1$ 把误差全部集中在一个方向上，沿那个方向的输入在 $E_1$ 下的误差是 $E_2$ 下的 10 倍。只看 Frobenius 误差会掩盖“某些输入错得很厉害”；只看谱范数又分不清“处处一点小误差”和“只有一个方向有误差”。两把尺子要一起看。
:::

:::: quiz 小练习：两个范数
$B=\begin{pmatrix}2&0\\0&-3\end{pmatrix}$，$P=\dfrac12\begin{pmatrix}1&1\\1&1\end{pmatrix}$。分别求 $\norm{B}_2$、$\norm{B}_F$、$\norm{P}_2$、$\norm{P}_F$。
::: answer 看答案
$B$ 的奇异值是 $\abs{-3}=3$ 和 $2$（对角矩阵的奇异值是对角元的绝对值，符号放进奇异向量）。$\norm{B}_2=3$，$\norm{B}_F=\sqrt{4+9}=\sqrt{13}\approx3.606$。

$P=\mathbf u\mathbf u^\top$，$\mathbf u=(1,1)/\sqrt2$，是投影到对角线方向的矩阵（M01 (M1.18)）。秩 1，唯一的非零奇异值是 $\norm{\mathbf u}^2=1$，所以 $\norm{P}_2=\norm{P}_F=1$。直接核对 Frobenius：$4\times\tfrac14=1$。这是 (M2.15) 左边取等号的例子。
:::
::::

::: note 这一页用在哪里
[M11 §3](m11-doubly-stochastic-sinkhorn.html#p3) 证明双随机矩阵的谱范数恰为 1，[第 16 课](../deepseek/lesson-16-mhc.html)与[第 18 课 §5](../deepseek/lesson-18-v4.html#p5)的 mHC 靠 (M2.14) 的第二式保证多层连乘不爆炸。[M07 §6](m07-optimization.html#p6) 讲 Muon 的正交化时会再用到谱范数。下一页：两把尺子下的最优低秩近似。
:::

::: key 本页要点
- $\norm{A}_F=\sqrt{\sum a_{ij}^2}=\sqrt{\sum\sigma_i^2}$：总能量；也是各向同性随机输入下的均方放大 $\E\norm{A\mathbf x}^2$。
- $\norm{A}_2=\max\norm{A\mathbf x}/\norm{\mathbf x}=\sigma_1$：最坏方向的放大倍数；$\norm{AB}_2\le\norm{A}_2\norm{B}_2$。
- $\sigma_1\le\norm{A}_F\le\sqrt r\,\sigma_1$。两者都是酉不变范数，这是最优低秩近似定理的前提。
- Frobenius 相同的两个误差，最坏情况可以差很多；两把尺子要一起看。
:::

# 最优低秩近似 | Eckart–Young–Mirsky：截断 SVD 为什么是最好的秩 k 近似 | 30

本课的核心问题：给一个矩阵 $A$ 和一个“秩预算” $k$，在所有秩不超过 $k$ 的矩阵 $B$ 里，哪一个离 $A$ 最近？最近能近到多少？MLA 的 $d_c=512$ 就是一个秩预算；§2 说预算越小越省，这一页说对一个给定的矩阵，预算为 $k$ 时误差至少有多大。

最自然的猜测是**只保留 (M2.8) 的前 $k$ 项**，丢掉最小的奇异值，这叫**截断 SVD**（truncated SVD）。可凭什么它最好？挑 $A$ 的某 $k$ 列会不会更好？答案是：在谱范数、Frobenius 范数以及任何酉不变范数下，截断 SVD 都最优。这一页完整证明谱范数的情形，给出 Frobenius 情形的证明概要（要借一个本课不证的不等式），一般酉不变范数的情形只引用。

## 图景：留下最长的 k 根轴

回到图 M2.1：(b) 把 (a) 的第二根轴 0.8 换成 0，就是 (a) 的秩 1 截断 SVD，损失恰好是那根轴的长度 0.8。必须牺牲一个方向，就牺牲最短的轴；高维里秩 $k$ 的近似要把 $n-k$ 个方向压成零，最划算的是压掉最短的那些轴。下面把这个直觉变成定理。

## 截断 SVD 的误差

::: eq 截断 SVD 与它的误差 | (M2.16)
$$A_k=\sum_{i=1}^{k}\sigma_i\mathbf u_i\mathbf v_i^\top,\qquad \norm{A-A_k}_2=\sigma_{k+1},\qquad \norm{A-A_k}_F=\sqrt{\sigma_{k+1}^2+\cdots+\sigma_r^2}$$
理由：$A-A_k=\sum_{i=k+1}^{r}\sigma_i\mathbf u_i\mathbf v_i^\top$，这本身就是一个 SVD（奇异向量仍两两垂直），它的奇异值是 $\sigma_{k+1},\dots,\sigma_r$；再用 (M2.14) 与 (M2.12)。
:::

$A_k$ 的秩恰好是 $k$，写成 (M2.10) 的形式就是 $(U_k\Sigma_k)(V_k^\top)$：一个 $[m,k]\times[k,n]$ 的低秩分解。

## 定理

::: thm 最优低秩近似（Eckart–Young–Mirsky）
**前提。** $A\in\R^{m\times n}$，奇异值 $\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0$（$r=\rank A$），整数 $0\le k<r$，$A_k$ 如 (M2.16)。

**结论。** 在所有 $\rank B\le k$ 的 $B\in\R^{m\times n}$ 中：

(a) $\min\norm{A-B}_2=\sigma_{k+1}$（Axler 7.92 [@axler2024]；MML 定理 4.25 [@mml2020]）；

(b) $\min\norm{A-B}_F=\sqrt{\sigma_{k+1}^2+\cdots+\sigma_r^2}$（Schmidt 1907 [@schmidt1907]，经 Stewart 转述 [@stewart1993]；Eckart 与 Young 1936 以最小二乘的形式重新得到 [@eckart1936]）；

两个最小值都由 $A_k$ 取到。更一般地，$A_k$ 在**任何**酉不变范数下都是极小元（Mirsky 1960 [@mirsky1960]，经 Stewart 转述）。

**唯一性。** 谱范数下极小元一般不唯一（见下文的例子）。Frobenius 范数下，Eckart 与 Young 的摘要说解“总存在，通常唯一”；精确条件是 $\sigma_k>\sigma_{k+1}$ 时唯一（笔者整理，本课不证；Eckart–Young 原文是否写出这个条件，本课未能核实，只读到了摘要）。
:::

注意约束写的是“$\rank B\le k$”而不是“$=k$”：Schmidt 原本的问题就是“用至多 $k$ 个秩 1 项的和去逼近”[@stewart1993]。

::: eq 最优误差 | (M2.17)
$$\min_{\rank B\le k}\norm{A-B}_2=\sigma_{k+1},\qquad \min_{\rank B\le k}\norm{A-B}_F^2=\sum_{i>k}\sigma_i^2=\norm{A}_F^2-\sum_{i\le k}\sigma_i^2$$
第二式最右边的写法是 Schmidt 的形式：能保住的“能量”至多是前 $k$ 个奇异值的平方和。
:::

## 证明：谱范数版本

截断 SVD 的误差已经算出是 $\sigma_{k+1}$，只需再证明**任何**秩不超过 $k$ 的 $B$ 都不可能更好。思路：秩 $\le k$ 的 $B$ 一定会“压扁”前 $k+1$ 个右奇异向量张成的空间里的某个方向，而 $A$ 在那个空间里每个方向都至少拉长 $\sigma_{k+1}$ 倍。下面按 Axler 7.92 的证明写成逐行推导。

::: deriv 任何秩 ≤ k 的 B，谱误差都至少是第 k+1 个奇异值（Axler 7.92）
$B\mathbf v_1,\dots,B\mathbf v_{k+1}\in C(B),\quad \dim C(B)\le k$ || $k+1$ 个向量落在一个至多 $k$ 维的空间里
$\textstyle\sum_{i=1}^{k+1}a_iB\mathbf v_i=\mathbf 0,\quad a_i\text{ 不全为 }0$ || 所以它们线性相关（Axler 2.22）
$\mathbf x=\textstyle\sum_{i=1}^{k+1}a_i\mathbf v_i\ne\mathbf 0,\quad B\mathbf x=\mathbf 0$ || $\mathbf v_i$ 线性无关，系数不全为 0 就不是零向量；第二式由线性得出
$\norm{(A-B)\mathbf x}^2=\norm{A\mathbf x}^2=\textstyle\sum_{i=1}^{k+1}a_i^2\sigma_i^2$ || $B\mathbf x=\mathbf 0$；$A\mathbf v_i=\sigma_i\mathbf u_i$ 且 $\mathbf u_i$ 两两垂直（勾股）
$\ge\sigma_{k+1}^2\textstyle\sum_{i=1}^{k+1}a_i^2=\sigma_{k+1}^2\norm{\mathbf x}^2$ || $i\le k+1$ 时 $\sigma_i\ge\sigma_{k+1}$；$\mathbf v_i$ 正交归一
$\norm{A-B}_2\ge\norm{(A-B)\mathbf x}/\norm{\mathbf x}\ge\sigma_{k+1}$ || 谱范数是所有方向里的最大放大倍数 (M2.14)
:::

与 (M2.16) 合起来，谱范数下的最小误差恰好是 $\sigma_{k+1}$，$A_k$ 取到它。

::: ex 证明在一个具体的 B 上怎样运作
取 $A=\diag(3,2,1)$，奇异值 $(3,2,1)$，右奇异向量就是 $\mathbf e_1,\mathbf e_2,\mathbf e_3$。秩预算 $k=1$，所以证明要在 $\operatorname{span}(\mathbf v_1,\mathbf v_2)$，也就是 $xy$ 平面里找一个被 $B$ 压成 0 的方向。

**一个“没对准”的对手。** 任取一个秩 1 的 $B=\mathbf b\mathbf c^\top$，比如 $\mathbf b=(1,1,1)$、$\mathbf c=(2,1,0)$。$B\mathbf x=\mathbf b\,(\mathbf c^\top\mathbf x)$，只要 $\mathbf c^\top\mathbf x=0$ 就被压成 0。在 $xy$ 平面里解 $2x_1+x_2=0$，得 $\mathbf x=(1,-2,0)$，$\norm{\mathbf x}=\sqrt5$。沿这个方向 $B$ 什么也没做，误差就是 $A$ 自己：$(A-B)\mathbf x=A\mathbf x=(3,-4,0)$，长度 5。于是

$$\norm{A-B}_2\ \ge\ \frac{\norm{(A-B)\mathbf x}}{\norm{\mathbf x}}=\frac{5}{\sqrt5}=\sqrt5\approx2.236>2=\sigma_2 .$$

注意这个下界与 $\mathbf b$ 无关：$\mathbf b$ 换成什么，$B\mathbf x$ 都是 0。只要 $B$ 的每一行都平行于 $\mathbf c^\top=(2,1,0)$，谱误差就不可能低于 $\sqrt5$（笔者对 2 万个随机的 $\mathbf b$ 数值求 $\norm{A-B}_2$，最小的是 2.2362，正好贴着这个下界；上面那个 $\mathbf b=(1,1,1)$ 实际是 3.09）。

**对准了的 $B$。** 截断 SVD $A_1=3\,\mathbf e_1\mathbf e_1^\top$ 的 $\mathbf c=\mathbf e_1$，被压成 0 的平面内方向是 $\mathbf e_2$，$A\mathbf e_2$ 的长度是 2，下界恰好是 $\sigma_2=2$，而 $\norm{A-A_1}_2=\norm{\diag(0,2,1)}_2=2$ 也确实取到了它。任何秩 1 的 $B$ 都必须在 $xy$ 平面里“放弃”一个方向；$A$ 在这个平面里每个方向都至少拉长 2 倍，所以最好的做法就是放弃拉长最少的 $\mathbf e_2$。
:::

## 证明概要：Frobenius 版本

Frobenius 版本要控制的是**所有**奇异值，需要一个比上面更强的不等式。它由 Weyl 给出，Stewart 在综述里用它推出 Frobenius 版本 [@stewart1993]（§6，式 (6.2)(6.3)）：

::: eq Weyl 不等式 | (M2.18)
$$\sigma_{i+j-1}(X+Y)\le\sigma_i(X)+\sigma_j(Y),\qquad i,j\ge1$$
$\sigma_i(\cdot)$ 是第 $i$ 大奇异值（超出秩的记为 0）。$i=j=1$ 即谱范数的三角不等式。本课引用，不证。
:::

::: deriv 由 Weyl 不等式得到 Frobenius 下界
$\sigma_{i+k}(A)\le\sigma_i(A-B)+\sigma_{k+1}(B)$ || (M2.18)：$X=A-B$，$Y=B$，$j=k+1$
$\sigma_{i+k}(A)\le\sigma_i(A-B)$ || $\rank B\le k$，所以 $\sigma_{k+1}(B)=0$
$\norm{A-B}_F^2=\textstyle\sum_{i\ge1}\sigma_i(A-B)^2$ || (M2.12)
$\ge\textstyle\sum_{i\ge1}\sigma_{i+k}(A)^2=\sum_{j>k}\sigma_j(A)^2$ || 逐项用第 (2) 行
:::

$i=1$ 这一项单独拿出来，$\sigma_{k+1}(A)\le\sigma_1(A-B)=\norm{A-B}_2$，就是刚才证过的谱范数版本。两个版本其实是同一个不等式的“最大一项”和“全部平方和”。

## 数字：一个 6×4 的矩阵

::: ex 截断 SVD 的误差表，以及 2 万个对手
取 `G = np.random.default_rng(1).standard_normal((6, 4))`（读者可以复现）。奇异值 $\boldsymbol\sigma=(2.0769,\ 1.6715,\ 1.4254,\ 0.5616)$，$\norm{G}_F=\sqrt{\sum\sigma_i^2}=3.0748$。按 (M2.16)：

| 保留 $k$ | 谱范数误差 $\sigma_{k+1}$ | F 误差 $\sqrt{\sum_{i>k}\sigma_i^2}$ | 保住的能量 $\sum_{i\le k}\sigma_i^2/\sum\sigma_i^2$ |
|---|---|---|---|
| 0 | 2.0769 | 3.0748 | 0 |
| 1 | 1.6715 | 2.2674 | 45.6% |
| 2 | 1.4254 | 1.5321 | 75.2% |
| 3 | 0.5616 | 0.5616 | 96.7% |
| 4 | 0 | 0 | 100% |

核对 $k=2$：$\sqrt{1.4254^2+0.5616^2}=\sqrt{2.0318+0.3154}=\sqrt{2.3472}=1.5321$。表中每个误差都用 `np.linalg.norm(G - G_k, 2)` 与 `'fro'` 直接算过，与奇异值给出的数一致。

**对手。** 笔者随机生成 2 万个秩 2 的候选：每次随机取一个 $X\in\R^{6\times2}$，再用最小二乘挑出对这个 $X$ 最好的 $Y$，候选是 $XY$。它们的 F 误差最好的是 1.6850，中位数 2.52，没有一个低于截断 SVD 的 1.5321。直接挑 $G$ 自己的某两列当 $X$（6 种挑法里最好的一种）得 1.5854，也输给截断 SVD。
:::

::: fig figures/m02-ey-spectrum.svg | 图 M2.5 奇异值谱决定最优误差（G 是一个 6×4 的随机矩阵）
**看 ② 里的两条折线：它们完全由 ① 的柱子算出，不需要知道 $G$ 的任何其他信息。** ① $G$ 的四个奇异值；柱子 $\sigma_{k+1}$ 画在 ② 的 $k$ 正上方。② 保留前 $k$ 项之后的误差：虚线（谱范数）在每个 $k$ 处恰好等于正上方那根柱子的高度；实线（Frobenius）是被丢掉的柱子高度的平方和开方。③ 2 万个随机秩 2 近似的 F 误差分布：全部落在绿线（截断 SVD，1.5321）右边；橙色虚线是“挑 $G$ 的两列”能做到的最好结果。全部数字由脚本从 `default_rng(1)` 与 `default_rng(0)` 两个种子算出。
:::

## 谱范数下，最优解不唯一

::: ex 同样好的秩 1 近似有无穷多个
$A=\operatorname{diag}(3,2,1)$，奇异值 $(3,2,1)$，截断 SVD $A_1=\operatorname{diag}(3,0,0)$，谱误差 $\sigma_2=2$，F 误差 $\sqrt{2^2+1^2}=\sqrt5\approx2.236$。

换成 $B=\operatorname{diag}(b,0,0)$：$A-B=\operatorname{diag}(3-b,2,1)$，谱误差 $\max(\abs{3-b},2)$，F 误差 $\sqrt{(3-b)^2+5}$。只要 $1\le b\le5$，谱误差都等于 2，和截断 SVD 一样好：$b=2.5$、$3.5$ 时 F 误差是 $\sqrt{5.25}\approx2.291$，$b=4$ 时是 $\sqrt6\approx2.449$。所以“谱范数下的最优秩 1 近似”有无穷多个，而 Frobenius 范数下只有 $b=3$ 这一个。
:::

::: fig figures/m02-uniqueness.svg | 图 M2.6 谱范数的最优解是一整段平台，Frobenius 的最优解是一个点
**看蓝色虚线的平台：$b$ 从 1 到 5，谱误差一直等于最优值 2。** 黑色实线（Frobenius 误差）只在 $b=3$ 处取到最小值 $\sqrt5$。两条曲线由脚本对每个 $b$ 直接计算范数得到。
:::

这就是为什么说“截断 SVD 是**一个**最优解”要带上范数：谱范数只盯着最坏的那个方向，其余方向差一点它不在乎，所以留给了很大的自由度。

::: err 常见说法：“Eckart–Young 在 1936 年证明了截断 SVD 是谱范数下的最优近似”
两处归属都放错了。Eckart 与 Young 1936 年解的是**最小二乘**（Frobenius）问题，借助“标准形式”求解，指出解总存在、通常唯一 [@eckart1936]；而这个结论 Schmidt 1907 年研究积分方程时已经得到，Stewart 称这个定理“often (and incorrectly) called the Eckart–Young theorem” [@stewart1993]（常被、而且错误地被称为 Eckart–Young 定理）。把结论推广到一切酉不变范数（谱范数是其中之一）的是 Mirsky 1960 年的工作 [@mirsky1960]；谱范数下的那一条不等式 $\sigma_1(A-B)\ge\sigma_{k+1}(A)$，在 Stewart 转述的 Weyl 论证里已经作为第一步出现（Stewart §6 式 (6.2) [@stewart1993]，也就是本页 Weyl 不等式推导里 $i=1$ 的那一项）。有的教材（如 MML 定理 4.25 [@mml2020]）把谱范数陈述挂在“Eckart–Young (1936)”名下，不要照抄。本课称它为 Eckart–Young–Mirsky 定理，并注明 Schmidt 的优先权（Schmidt 与 Mirsky 的原文本课未读，内容来自 Stewart 的综述）。
:::

::: warn 失败模式：定理回答的是一个很窄的问题
定理说的是：**给定一个固定的矩阵，用一把各向同性的尺子量**，截断 SVD 最好。在神经网络里两个前提都可能不成立：真实的隐藏状态并不在每个方向上同样常见（尺子应当按数据加权），训练时矩阵本身和网络的其余部分都会变（矩阵不固定）。§6 专门讨论。另外，定理只保证“最好”，不保证“够好”：谱衰减得慢，截断就要丢掉可观的能量，上面的 $G$ 保留一半的秩只保住 75.2%。
:::

:::: quiz 小练习：只看奇异值就能答的题
某矩阵的奇异值为 $(10,\ 6,\ 3,\ 1,\ 0.5)$。（1）最优秩 2 近似的谱误差、F 误差各是多少？（2）要让 F 误差不超过 $\norm{A}_F$ 的 10%，至少要保留几项？
::: answer 看答案
（1）谱误差 $\sigma_3=3$。F 误差 $\sqrt{3^2+1^2+0.5^2}=\sqrt{10.25}\approx3.202$。

（2）$\norm{A}_F^2=100+36+9+1+0.25=146.25$，$\norm{A}_F\approx12.093$，10% 是 1.2093，平方后 1.4625。丢掉的平方和必须 $\le1.4625$：只丢最后一项 $0.25$ 可以；丢最后两项 $1+0.25=1.25$ 也可以；丢最后三项 $10.25$ 不行。所以至少保留 3 项，此时 F 误差 $\sqrt{1.25}\approx1.118$，是 $\norm{A}_F$ 的 9.2%。
:::
::::

::: note 这一页用在哪里
[第 9 课](../deepseek/lesson-09-v2-mla.html)：$d_c=512$ 是一个秩预算；如果要用秩 512 的矩阵去逼近一个固定的矩阵，最优误差由它第 513 个及以后的奇异值决定。不过 MLA 并不是这样得到压缩矩阵的，下一页讲区别。[第 18 课 §3](../deepseek/lesson-18-v4.html#p3)：V4 的分组输出投影与查询压缩也都是“给定秩预算”的设计（本课 §7）。
:::

::: key 本页要点
- 截断 SVD $A_k$ 的误差：谱范数 $\sigma_{k+1}$，Frobenius $\sqrt{\sum_{i>k}\sigma_i^2}$；不需要看矩阵的其他任何信息。
- 定理：在所有 $\rank B\le k$ 中，$A_k$ 在两种范数（以及一切酉不变范数）下都最优。谱范数版本的证明：秩 $\le k$ 的 $B$ 必然压扁前 $k+1$ 个奇异方向里的某个方向。
- 谱范数下最优解不唯一；Frobenius 下 $\sigma_k>\sigma_{k+1}$ 时唯一。
- 归属：Frobenius 版本最早是 Schmidt 1907，Eckart–Young 1936 重新发现；推广到一切酉不变范数（含谱范数）的是 Mirsky 1960。
- 定理的前提是“固定矩阵 + 各向同性的尺子”，训练出来的压缩两者都不满足。
:::

# 学出来的低秩 | 为什么 MLA 直接训练两块瘦矩阵，而不是截断一个现成的矩阵 | 25

得到 MLA 那两块瘦矩阵，原则上有两条路。**路线 A**：先训练一个普通的多头注意力，拿到满秩的 $W^K$、$W^V$，做 SVD，只留前 512 个奇异方向（事后截断）。**路线 B**：从一开始就把 512 维的瓶颈写进网络结构，让 $W^{DKV}$、$W^{UK}$、$W^{UV}$ 和模型其余部分一起从头训练。DeepSeek 走的是路线 B：DeepSeek-V3 的硬件论文说，MLA 用一个“与模型联合训练”的投影矩阵把所有头的 KV 表示压缩成更小的潜向量 [@isca25]。所以“MLA 就是对 MHA 的权重做了一次 SVD”是一个常见的误解（[第 9 课 §2](../deepseek/lesson-09-v2-mla.html#p2) 专门有一个框纠正它）。

可上一页刚证明截断 SVD **最优**，路线 B 凭什么行，甚至可能更好？答案在 §5 最后的失败模式里：定理的两个前提，“各向同性的尺子”和“固定的矩阵”，训练都换掉了。下面一个一个拆开。

## 第一个前提：尺子应当按数据加权

对一个线性层来说，我们真正在乎的不是矩阵 $W$ 与 $B$ 在“所有方向同等重要”的意义下差多远，而是**在真实输入上**输出差多远。设输入 $\mathbf x\in\R^n$ 的二阶矩矩阵是 $S_x=\E[\mathbf x\mathbf x^\top]\in\R^{n\times n}$（第 $(i,j)$ 个元素是 $\E[x_ix_j]$；均值为 0 时就是协方差矩阵），(M2.13) 的推导照搬过来，只是最后一步不再是单位矩阵：

::: pre 对称半正定矩阵的平方根 | 知道 $S^{1/2}$ 的可以跳过
$S_x$ 是对称的，而且对任何 $\mathbf a$，$\mathbf a^\top S_x\mathbf a=\E[(\mathbf a^\top\mathbf x)^2]\ge0$（半正定）。由实谱定理（Axler 7.29），$S_x=Q\Lambda Q^\top$，$Q$ 正交，$\Lambda=\diag(\lambda_1,\dots,\lambda_n)$ 且 $\lambda_i\ge0$：$\mathbf q_i$ 是输入的主方向，$\lambda_i=\mathbf q_i^\top S_x\mathbf q_i=\E[(\mathbf q_i^\top\mathbf x)^2]$ 是输入在该方向上的二阶矩（均值为 0 时就是方差；越大，这个方向上的输入越强、越常见）。定义 $S_x^{1/2}=Q\Lambda^{1/2}Q^\top$，则 $S_x^{1/2}S_x^{1/2}=Q\Lambda Q^\top=S_x$。若所有 $\lambda_i>0$，还有逆 $S_x^{-1/2}=Q\Lambda^{-1/2}Q^\top$。
:::

::: eq 按数据加权的误差 | (M2.19)
$$\E\,\norm{(W-B)\mathbf x}^2=\tr\big((W-B)S_x(W-B)^\top\big)=\norm{(W-B)\,S_x^{1/2}}_F^2$$
第一个等号同 (M2.13) 的推导；第二个等号用 $S_x=S_x^{1/2}S_x^{1/2}$、$S_x^{1/2}$ 对称，以及 $\norm{Y}_F^2=\tr(YY^\top)$（取 $Y=(W-B)S_x^{1/2}$）。$S_x=I$ 时退回到普通的 Frobenius 误差。
:::

这把尺子不再是酉不变的，所以截断 $W$ 自己的 SVD 不再是最优的。但只要换一个变量，问题就变回了 Eckart–Young–Mirsky：

::: deriv 按数据加权的最优秩 k 近似（笔者由定理推出）
$\min_{\rank B\le k}\norm{(W-B)S_x^{1/2}}_F=\min_{\rank B\le k}\norm{WS_x^{1/2}-BS_x^{1/2}}_F$ || 乘法对减法可分配
$B'=BS_x^{1/2},\qquad \rank B'=\rank B$ || $S_x^{1/2}$ 可逆（设所有 $\lambda_i>0$）：由 (M2.3)，乘上可逆矩阵既不增也不减秩
$\min_{\rank B'\le k}\norm{WS_x^{1/2}-B'}_F=\sqrt{\textstyle\sum_{i>k}\sigma_i^2(WS_x^{1/2})}$ || 对矩阵 $WS_x^{1/2}$ 用 Eckart–Young–Mirsky (M2.17)
$B^*=(WS_x^{1/2})_k\,S_x^{-1/2}$ || 极小元 $B'$ 是截断 SVD，再乘回 $S_x^{-1/2}$
:::

::: eq 按数据加权的最优解 | (M2.20)
$$B^*=\big(WS_x^{1/2}\big)_k\,S_x^{-1/2},\qquad \min_{\rank B\le k}\E\norm{(W-B)\mathbf x}^2=\sum_{i>k}\sigma_i^2\big(WS_x^{1/2}\big)$$
前提：$S_x$ 可逆（所有 $\lambda_i>0$）。决定压缩难易的是“透过数据看到的矩阵” $WS_x^{1/2}$ 的奇异值，而不是 $W$ 自己的。
:::

直觉：输入几乎从不出现的方向，$W$ 在那里做什么都无所谓，可以放心压扁；截断 $W$ 自己的 SVD 看不到这一点，可能为“很大但用不上”的方向浪费秩预算。图 M2.7 把这件事画在一个 $2\times2$ 的例子上。隐藏状态若集中在少数方向上（$S_x$ 只有少数几个大特征值），$WS_x^{1/2}$ 也就只有少数几个大奇异值，低秩就便宜。精确地说，$\sigma_i(WS_x^{1/2})\le\norm{W}_2\sqrt{\lambda_i}$（$\lambda_i$ 从大到小排）：取 $S_x^{1/2}$ 的秩 $i-1$ 截断 SVD $C$，由 (M2.16) $\norm{S_x^{1/2}-C}_2=\sqrt{\lambda_i}$；$WC$ 的秩 $\le i-1$ (M2.3)，所以由 (M2.17) 与 (M2.14)，$\sigma_i(WS_x^{1/2})\le\norm{WS_x^{1/2}-WC}_2\le\norm{W}_2\sqrt{\lambda_i}$。（这是条件；真实模型的隐藏状态谱本课没有测量。）

::: fig figures/m02-data-weighted.svg | 图 M2.7 秩 1 预算该留给哪个方向（W = diag(3, 1)，输入几乎只在纵向变化）
**看 ③ ④ 里的灰点：它们是完整的输出 $W\mathbf x$，彩色点是秩 1 近似留下的部分，灰点离彩色点越远，丢得越多。** ① 输入 $\mathbf x\sim N(0,S_x)$，$S_x=\diag(0.01,1)$，横向几乎不动。② $W=\diag(3,1)$ 放大最多的是横向（虚线椭圆的长半轴 $\sigma_1=3$），可输入偏偏不去那里，输出的点云仍是竖着的。③ 截断 $W$ 自己的 SVD，留下 $\sigma_1$ 所在的横向，所有输出被压到一条短横线上，按 (M2.19) 丢掉 91.7%。④ (M2.20) 的 $B^*$ 留下数据所在的纵向，只丢 8.3%。误差由 (M2.19) 精确算出，点是脚本用固定种子抽的 160 个样本；推算过程见练习 8。
:::

## 实验：SGD 自己找到了这个最优解

路线 B 不做任何 SVD，也不知道 $S_x$，只是拿样本 $(\mathbf x,W\mathbf x)$ 做梯度下降。它能找到 (M2.20) 吗？笔者用一个可控的玩具实验来看。

::: ex 直接学 U、V vs 截断 SVD（玩具实验，数字可复现）
**设置。** $m=n=32$。目标映射 $W$ 是随机高斯矩阵（除以 $\sqrt{32}$）。输入 $\mathbf x\sim N(0,S_x)$，$S_x$ 的 32 个方差从 1 到 0.001 按对数均匀分布，方向是一组随机的正交基，与 $W$ 的奇异方向无关。对每个秩预算 $k$，比较三种 $B$ 的相对误差 $\E\norm{(W-B)\mathbf x}^2/\E\norm{W\mathbf x}^2$：

- **截断**：$W$ 自己的截断 SVD $W_k$（路线 A 的做法）；
- **最优**：(M2.20) 的 $B^*$；
- **学习**：随机初始化 $U\in\R^{32\times k}$、$V\in\R^{k\times32}$，每步抽 64 个输入，用 $\norm{UV\mathbf x-W\mathbf x}^2$ 的梯度更新，学习率 0.05，训练 6 万步（路线 B 的做法）。

| $k$ | 1 | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| 截断 $W$ 的 SVD | 91.2% | 83.0% | 67.9% | 41.8% | 22.2% | 11.2% | 1.18% |
| 最优 (M2.20) | 75.6% | 58.5% | 33.7% | 11.24% | 3.69% | 1.27% | 0.08% |
| SGD 直接学 | 75.6% | 58.5% | 33.7% | 11.25% | 3.70% | 1.28% | 0.08% |

**读法。** 同样的秩预算 $k=8$，截断 $W$ 的 SVD 留下 41.8% 的误差，按数据加权的最优解只有 11.24%，差了将近 4 倍；要让截断达到 11% 左右，得把预算加到 $k=16$。而 SGD 从随机初始化出发，只看样本，学到的 $UV$ 的误差比最优解多不到 0.02 个百分点（$k=4$ 时最多，0.019；$k=24$ 时只有 0.0002）；它与 $B^*$ 本身在 (M2.19) 意义下的相对距离在 0.14%（$k=24$）到 2.1%（$k=4$）之间。它学到的 $U$、$V$ 本身并不是 SVD 的因子（(M2.7) 的自由度还在），但乘积是对的。
:::

::: fig figures/m02-learned-vs-svd.svg | 图 M2.8 直接学两块瘦矩阵 vs 截断 SVD（玩具实验）
**看左图：蓝点（SGD 直接学出的 $UV$）全部落在绿线（按数据加权的最优解）上，而橙线（截断 $W$ 自己的 SVD）在每个秩预算下都差得多。** 纵轴是对数刻度的相对误差。右图说明原因：同一个 $W$，“透过数据看”的矩阵 $WS_x^{1/2}$（绿）的奇异值比 $W$ 自己的（橙）衰减快得多，所以同样的秩能留住更多真正用得上的东西。橙线、绿线按 (M2.19)(M2.20) 逐个 $k$ 精确计算；蓝点由脚本里的 SGD 训练得到，脚本约 15 秒可复现。
:::

::: warn 这个实验说明了什么、没说明什么
**说明了：** 输入分布不均匀时，事后截断可以差很远；在这个线性、目标固定的问题里，梯度下降直接训练两个因子就找到了按数据加权的最优解。**没说明：** 非线性网络里梯度下降能否找到最好的低秩解；“SGD 到达最优”是本实验的实测，不是定理。
:::

## 第二个前提：矩阵不是固定的

真实模型里，$W^{UK}$、$W^{DKV}$ 与 $W^Q$、$W^O$ 以及前后各层**一起**训练。路线 A 先让模型在满秩条件下长成，再硬塞进瓶颈；路线 B 从第一步起就让其余部分知道“只有 512 个数能被记下来”，可以调整自己，把注意力需要的信息放进这 512 个方向。Eckart–Young–Mirsky 讨论的是固定的矩阵，不描述这种共同适应（这是笔者的推理，不是定理，也不是论文原话）。

实测的证据来自 DeepSeek-V2 附录 D.2 的消融：在两个规模的混合专家（MoE）模型上，各训练一对除注意力之外结构相同的模型，一个用 MHA、一个用 MLA [@dsv2]。论文没有写出这两个 MLA 模型的 $d_c$；表中每 token 的缓存 15.6K、34.6K 分别等于 $576\times27$、$576\times60$（四舍五入后），与每层缓存 $512+64$ 个数相符，这是笔者的推算。论文表 9 的主要数字如下：

| 模型（总参数） | 每 token KV 缓存（元素） | BBH | MMLU | C-Eval | CMMLU |
|---|---|---|---|---|---|
| 小 MoE + MHA（15.8B） | 110.6K | 37.9 | 48.7 | **51.6** | 52.3 |
| 小 MoE + MLA（15.7B） | 15.6K（14%） | **39.0** | **50.0** | 50.9 | **53.4** |
| 大 MoE + MHA（250.8B） | 860.2K | 46.6 | 57.5 | 57.9 | 60.7 |
| 大 MoE + MLA（247.4B） | 34.6K（4%） | **50.7** | **59.0** | **59.2** | **62.5** |

论文的概括是 MLA 表现好于 MHA，缓存只有 14% 和 4% [@dsv2]。两点注意：小模型的 C-Eval 上 MLA 低 0.7 分，“处处更好”是过度概括；小模型训练 1.33T token，大模型只训练 420B（第 9 课研究笔记照录了这些条件）。**论文没有解释低秩瓶颈为什么反而得分更高**；本页的玩具实验只说明“学出来的压缩远好于事后截断”，不能说明“比满秩更好”。

::: ana 回到类比：速记是学出来的
第 9 课说“速记是学出来的线性压缩”。本页给了这句话的数学内容：事后截断像先有逐字纪要，再按“哪几个字最长”删减（各向同性的尺子）；学出来的速记像速记员从第一天起就知道本子只有 512 格、知道哪些内容以后真会被查到（按数据加权），与会者也配合着把要点说得适合速记（共同适应）。**边界**：真正的速记员能临场决定怎么记；MLA 的 $W^{DKV}$ 训练完就固定了，对每个 token 用同一个线性规则。
:::

::: warn 失败模式：学出来的低秩也有天花板
瓶颈是硬的：无论怎么训练，$W^{UK}W^{DKV}$ 的秩都不超过 $d_c$，注意力若需要更多独立方向，损失就降不下去。DeepSeek-V2 取 $d_c=4d_h=512$ [@dsv2]，这是经验选择：论文没有给出“512 为什么够”的理论，也没有报告 $d_c$ 取其他值的对比实验；上面的消融只说明在这一套设定下 MLA 不输 MHA。
:::

:::: quiz 小练习：数据加权
（1）若输入各向同性，$S_x=I$，(M2.20) 的最优解是什么？与 §5 一致吗？（2）若 $S_x=\diag(1,\ 0.01)$、$W=I_2$，最优秩 1 近似保留哪个方向？相对误差多少？截断 $W$ 自己的 SVD 又会怎样？
::: answer 看答案
（1）$S_x^{1/2}=I$，$B^*=W_k$，最小误差 $\sum_{i>k}\sigma_i^2(W)=\norm{W-W_k}_F^2$，正是 §5 的 Frobenius 版本。

（2）$WS_x^{1/2}=\diag(1,\ 0.1)$，奇异值 1 与 0.1，截断保留第一个方向：$(WS_x^{1/2})_1=\diag(1,0)$，$B^*=\diag(1,0)\,\diag(1,10)=\diag(1,0)$。误差 $0.1^2=0.01$，相对误差 $0.01/(1+0.01)\approx0.99\%$。而 $W=I$ 的两个奇异值都是 1，SVD 不唯一，截断“$W$ 自己的 SVD”可能保留第二个方向 $\diag(0,1)$，此时误差是 1，相对误差 $1/1.01\approx99\%$。$W$ 自己的谱完全不告诉你该留哪个方向，数据才告诉你。
:::
::::

::: note 这一页用在哪里
[第 9 课 §2](../deepseek/lesson-09-v2-mla.html#p2) 的速记类比与“学出来的压缩”；[第 9 课 §8](../deepseek/lesson-09-v2-mla.html#p8) 的缓存与质量总账。[第 18 课 §3](../deepseek/lesson-18-v4.html#p3)：DeepSeek-V4 的注意力不再用 $W^{UK}$、$W^{UV}$ 展开，但查询仍先压到一个低维潜向量（V4-Pro 1536 维、V4-Flash 1024 维）再展开，同样是从头训练的低秩瓶颈 [@dsv4]。
:::

::: key 本页要点
- MLA 的压缩矩阵与模型一起从头训练（路线 B），不是对 MHA 权重做 SVD 截断（路线 A）。
- 真正该用的尺子是按数据加权的 $\E\norm{(W-B)\mathbf x}^2=\norm{(W-B)S_x^{1/2}}_F^2$；最优解是 $(WS_x^{1/2})_kS_x^{-1/2}$，由 $WS_x^{1/2}$ 的谱决定。
- 玩具实验：$k=8$ 时截断 $W$ 的 SVD 误差 41.8%，最优 11.24%，SGD 直接学两块因子 11.25%。
- 真实模型里其余部分还会共同适应；V2 的消融显示 MLA 在多数基准上不输甚至好于 MHA，缓存只有 14%/4%，但原因论文没有解释。
:::

# 结合律与“吸收” | 形状与 FLOPs：固定的乘积什么时候先乘，什么时候永远不乘 | 20

[M01 §6](m01-vectors-matrices.html#p6) 用结合律推出了矩阵吸收 $\mathbf q_{t,i}^\top(W^{UK}_i\mathbf c_j)=\big((W^{UK}_i)^\top\mathbf q_{t,i}\big)^\top\mathbf c_j$，[第 9 课 §3](../deepseek/lesson-09-v2-mla.html#p3) 把它推到整层。DeepSeek-V2 论文说推理时 $W^{UK}$ 可以“吸收进” $W^Q$（附录写作 $W^{UQ}$）、$W^{UV}$ 可以“吸收进” $W^O$ [@dsv2]。这一页补上秩的视角：要不要真的把这些矩阵预先乘成一个，由 §2 的临界秩决定。

## 预先乘好，还是分开乘？

设同一个固定乘积 $XY$ 要作用在很多向量上，$X\in\R^{m\times k}$，$Y\in\R^{k\times n}$。分开乘 $X(Y\mathbf z)$：每个向量 $2k(m+n)$ 次运算，存 $k(m+n)$ 个数。预先乘好 $(XY)\mathbf z$：一次性花 $2mkn$，之后每个向量 $2mn$，存 $mn$ 个数。

::: eq 预乘规则 | (M2.21)
$$\text{预先乘成一个矩阵才划算}\iff mn<k(m+n)\iff k>r^*=\frac{mn}{m+n}$$
$XY$ 的秩至多 $k$ (M2.3)。所以这条规则就是 (M2.6) 反过来读：乘积相对于它的形状是低秩的（$k<r^*$），就永远保持分开；只有中间维超过临界秩，预乘才省。
:::

把 MLA 里的三个固定乘积逐一代进去（维度取 DeepSeek-V2 [@dsv2]，数字经第 9 课研究笔记核对）：

| 固定乘积 | 形状 $[m,n]$ | 中间维 $k$（秩上界） | $r^*$ | 结论 |
|---|---|---|---|---|
| $W^{UK}W^{DKV}$（键的等效投影） | $[16384,5120]$ | 512 | 3901 | 分开：这就是压缩本身（§2） |
| $(W^{UK}_i)^\top W^{UQ}_i$（查询侧吸收，每头） | $[512,1536]$ | 128 | 384 | 分开：整存 786,432 个数，分开存 262,144，3.0 倍 |
| $W^O_iW^{UV}_i$（输出侧吸收，每头） | $[5120,512]$ | 128 | 465 | 分开：整存 2,621,440，分开存 720,896，3.6 倍 |

三行的结论都是“分开”。所以工程上的“吸收”不是把两个矩阵乘成一个，而是**改变乘法的顺序**：查询先乘 $(W^{UK}_i)^\top$ 变成 512 维，直接与缓存里的 $\mathbf c_j$ 点积；输出侧先对 $\mathbf c_j$ 加权平均，最后才乘 $W^{UV}_i$ 和 $W^O_i$（图 M2.9）。按 128 个头算，查询侧若预乘，每个 token 要 $2\times1536\times128\times512\approx2.01\times10^8$ 次运算，按顺序分开乘只要 $6.71\times10^7$。“吸收进 $W^Q$”是数学上的等价说法，不是计算的做法。

::: fig figures/m02-absorb-chain.svg | 图 M2.9 吸收后的 MLA 单头解码：顺序决定一切（DeepSeek-V2 维度）
**看两条红色虚线括号：它们框住的两个固定乘积都不能预先乘出来，因为它们的秩只有 128，远小于各自的临界秩。** ① 打分：$\mathbf h_t$（5120）经查询压缩得到 $\mathbf c_t^Q$（1536），展开成第 $i$ 头的 $\mathbf q^C_{t,i}$（128），再乘 $(W^{UK}_i)^\top$ 变成 512 维的 $\tilde{\mathbf q}_{t,i}$，直接与缓存里每个 512 维的 $\mathbf c_j$（橙色）点积。② 输出：用 softmax 权重对 $\mathbf c_j$ 加权平均，得到仍是 512 维的 $\tilde{\mathbf o}_{t,i}$，然后才展开到 128 维、再投到 5120 维，128 个头的结果相加。柱高正比于向量长度（1 像素 = 40 个数）。RoPE 那一小部分（第 9 课 §4、§5）没有画。
:::

::: ana 类比：直接读速记
**映射。** 沿用 §2 的速记类比：缓存里的 $\mathbf c_j$ 是速记符号，$W^{UK}_i$ 是“把速记展开成标签”的规则。先展开再比对，等于每来一个新问题，就把本子上所有 $T$ 条速记逐条誊写成正文再读；吸收则是把**问题**本身用同一套规则反向翻译成速记（$(W^{UK}_i)^\top\mathbf q$），然后直接拿它和速记符号比对。翻译问题每步只做 1 次，誊写记录每步要做 $T$ 次，这就是图 M2.10 里两条线越拉越开的原因。

**收益。** 它说明了吸收为什么不改变结果：翻译问题与展开记录是同一个线性规则从两侧作用，点积的值由结合律保证相等，没有任何近似。

**失效的地方。** 真实的“翻译”是精确的矩阵乘法，不需要另外学一种“读速记的眼光”，只是把括号换了位置。夹在中间的东西一旦随位置变化（RoPE 的旋转），就没有一套对所有 $j$ 都通用的翻译规则，吸收就做不成，下面的警告框会再讲。
:::

## 顺序决定每一步解码的账

第 9 课的研究笔记给出了一层 MLA 在解码一个 token、缓存里有 $T$ 个 token（含新 token 自己）时的三种算法（$n_h=d_h=128$，$d_h^R=64$，$d_c=512$；笔者复算一致）：

::: eq 一层解码一步的 FLOPs | (M2.22)
$$\begin{aligned}\text{(a) 先展开：}\quad&33{,}636{,}352\,T\\ \text{(b) 缓存展开后的 K、V：}\quad&33{,}554{,}432+81{,}920\,T\\ \text{(c) 吸收：}\quad&33{,}554{,}432+278{,}528\,T\end{aligned}$$
(a) 每步把 $T$ 个 $\mathbf c_j$ 全部展开；(b) 旧 token 展开后的 K、V 已在缓存里（缓存回到 MHA 的大小），常数项是把新 token 自己的 $\mathbf c_t$ 展开一次、存进缓存；(c) 常数项是每步各乘一次 $(W^{UK}_i)^\top$、$W^{UV}_i$。
:::

这三行系数都可以用 [M01 §5](m01-vectors-matrices.html#p5) 的规则自己数出来：$[m,n]$ 的矩阵乘一个向量是 $2mn$ 次运算（每个乘加算 2 次），两个 $n$ 维向量的点积是 $2n$ 次。每个头打分用的键是 $d_h+d_h^R=128+64=192$ 维（[第 9 课 §5](../deepseek/lesson-09-v2-mla.html#p5) 的解耦 RoPE），值是 128 维。三种算法共有的部分（由 $\mathbf h_t$ 算查询、最后乘 $W^O$、softmax）都不计入。

::: deriv (M2.22) 的三个系数从哪里来
$2\cdot512\cdot(2\cdot16384)=33{,}554{,}432$ || 展开一个 $\mathbf c_j$：$W^{UK}$、$W^{UV}$ 各 $[16384,512]$
$2\cdot192\cdot128+2\cdot128\cdot128=81{,}920$ || 一个缓存 token 的打分 + 加权求和（128 个头）
$33{,}554{,}432+81{,}920=33{,}636{,}352$ || (a) 的斜率 = 第 (1) + (2) 行；(b) 的斜率只有第 (2) 行，常数项是第 (1) 行
$2\cdot2\cdot512\cdot128\cdot128=33{,}554{,}432$ || (c) 常数项：每头各乘一次 $(W^{UK}_i)^\top$ 与 $W^{UV}_i$
$2\cdot576\cdot128+2\cdot512\cdot128=278{,}528$ || (c) 的斜率：打分 576 维，加权求和 512 维
:::

第 (2) 行里打分是 $49{,}152$、加权求和是 $32{,}768$；第 (5) 行对应的两项是 $147{,}456$ 与 $131{,}072$。第 (4) 行的 $(W^{UK}_i)^\top$ 是 $[512,128]$（查询侧，把 128 维的查询变成 512 维），$W^{UV}_i$ 是 $[128,512]$（输出侧），两者都与 $T$ 无关。

两个巧合值得点破。(a) 的斜率与 (b)、(c) 的常数项几乎相等（33,636,352 对 33,554,432），因为三者都是“128 个头 × 一次 $[128,512]$ 规模的变换，两侧各一次”；区别只在 (a) 对每个缓存 token 做一遍，(b) 每步只对新 token 做一遍，(c) 每步只对一个查询做一遍（练习 7）。$T=1$ 时缓存里只有新 token 自己，(a) 与 (b) 做的是同一件事，公式也给出相同的 33,636,352。(c) 的斜率比 (b) 大，是因为 576 维、512 维比 192 维、128 维宽：这就是下面“3.4 倍”的来源。

::: ex T = 4096 时的三本账
(a) $33{,}636{,}352\times4096\approx1.378\times10^{11}$；(b) $33{,}554{,}432+81{,}920\times4096\approx3.69\times10^{8}$；(c) $33{,}554{,}432+278{,}528\times4096\approx1.174\times10^{9}$。吸收比先展开省 117 倍；$T=128\text{K}$ 时是 121 倍，趋于 $33{,}636{,}352/278{,}528\approx120.8$。

但 (c) 是 (b) 的 3.2 倍：吸收后每个缓存 token 要和 576 维、512 维的向量打交道，比展开后的 192 维键、128 维值更贵。只看随 $T$ 增长的部分，每个缓存 token 的比值是 $278{,}528/81{,}920=3.4$（图 M2.10 右侧写的就是这个数）；两者的常数项相同，$T=4096$ 时合计 $1.174/0.369\approx3.18$ 倍，$T$ 越大越接近 3.4（$T=128\text{K}$ 时 3.39）。它换来的是字节：每层每个缓存 token，(b) 要读 $2\times128\times(192+128)=81{,}920$ 字节（bf16，按每个头存一份完整 192 维键的口径，HF 参考实现就是这样存的；若共享的 64 维 RoPE 键只存一份，是 65,664 字节），(c) 只读 $2\times576=1{,}152$ 字节，少 71 倍（按后一种口径约 57 倍）。解码受带宽限制（[M01 §5](m01-vectors-matrices.html#p5)、[第 9 课 §7](../deepseek/lesson-09-v2-mla.html#p7)），这笔交换是划算的。$T=1$ 时 (c) 比 (a) 还略贵（33,832,960 对 33,636,352），$T\ge2$ 起吸收就领先。
:::

::: fig figures/m02-absorb-flops.svg | 图 M2.10 括号放在哪里的代价（DeepSeek-V2 一层 MLA，解码一个 token）
**看左图红线与橙线的距离：在 $T=4096$ 处相差 117 倍，而且随 $T$ 继续拉开，直到约 121 倍。** 三条线都按 (M2.22) 逐点计算，两轴都是对数刻度。$T$ 很小时三条线挤在一起：三者都含一次约 $3.4\times10^7$ 的“$[128,512]$ 规模变换”（$T=1$ 时 (a) 与 (b) 相等）。紫色虚线（缓存展开后的 K、V）的 FLOPs 最低，但它要求每层每个 token 存 40,960 个数（每头 192 维键加 128 维值，与右图 81,920 字节同一口径）。右图是每个缓存 token 每层要读的字节：吸收只需读 $[\mathbf c_j;\mathbf k^R_j]$ 共 576 个数。
:::

::: warn 吸收成立的两个条件
（1）夹在查询和 $W^{UK}$ 中间的东西必须是固定的。RoPE 的旋转矩阵随位置变化，夹进去就无法吸收，这是第 9 课“解耦 RoPE”的由来（[M01 §6](m01-vectors-matrices.html#p6)、[第 9 课 §4](../deepseek/lesson-09-v2-mla.html#p4)）。（2）结合律在实数里精确成立，在浮点数里只是“几乎相等”：换一种求和顺序，末几位会变（M01 的警告框）。
:::

## 第 18 课：分组输出投影是“分块低秩”

DeepSeek-V4 的注意力不再用 $W^{UK}$、$W^{UV}$，但输出侧用了一种新的分解 [@dsv4]。以 V4-Pro 为例（维度来自第 18 课研究笔记对模型配置的核对）：128 个头、每头输出 512 维，拼起来 65,536 维，要投到 $d=7168$。直接投影要 $65{,}536\times7168\approx4.70\times10^8$ 个参数。V4 把头分成 16 组，每组 4096 维的输出先各自投到 $d_g=1024$ 维（一个分块对角的矩阵），拼成 16,384 维，再投到 7168 维：$16\times1024\times4096+16384\times7168\approx1.85\times10^8$，是直接投影的 39.3%（V4-Flash：8 组，$6.71\times10^7$ 对 $1.34\times10^8$，50%）。

用本课的语言看：等效的输出矩阵按组切成列块，每一块是 $[7168,1024]\times[1024,4096]$ 的乘积，秩至多 1024，而这一块的临界秩是 $7168\times4096/11264\approx2607$，所以分解省钱 (M2.6)。但整个等效矩阵的秩上界是 $\min(7168,\ 16384)=7168$，也就是满秩所允许的最大值：它不是一个整体的低秩约束，而是“每一组各自低秩”。

:::: quiz 小练习：该不该预乘
GPT-2 small 的一个头：$W^Q_i,W^K_i\in\R^{64\times768}$，打分可以写成 $\mathbf h_t^\top\big((W^Q_i)^\top W^K_i\big)\mathbf h_j$（M01 (M1.20)）。把 $(W^Q_i)^\top W^K_i$ 预先乘成一个 $[768,768]$ 的矩阵划算吗？
::: answer 看答案
中间维 $k=64$，临界秩 $r^*=768\times768/1536=384$。$64<384$，不划算：整存 $768^2=589{,}824$ 个数，分开存 $64\times1536=98{,}304$ 个，差 6 倍；每个向量的运算量之比也是 6 倍。$(W^Q_i)^\top W^K_i$ 的秩至多 64，这正是“每个头只通过一个 64 维的窗口比较两个 token”（M01 §6）的另一种说法。
:::
::::

::: note 这一页用在哪里
[第 9 课 §3](../deepseek/lesson-09-v2-mla.html#p3)：吸收的逐行推导与每 token 算力账；[第 9 课 §7](../deepseek/lesson-09-v2-mla.html#p7)：MHA 模式与 MQA 模式、解码的屋顶线。[第 18 课 §3](../deepseek/lesson-18-v4.html#p3)：分组输出投影。[M10 §3](m10-compute-arithmetic.html#p3)：FLOPs 与字节的交换（算术强度）。
:::

::: key 本页要点
- 固定乘积 $XY$（中间维 $k$）只有在 $k>r^*=mn/(m+n)$ 时才值得预先乘出；低秩的乘积永远分开存、分开乘。
- MLA 的“吸收”是改变乘法顺序：查询先变成 512 维与缓存点积，输出先在 512 维里加权平均。预乘 $(W^{UK}_i)^\top W^{UQ}_i$ 或 $W^O_iW^{UV}_i$ 反而要存、要算分开时的 3.0 倍和 3.6 倍。
- V2 一层解码（$T=4096$）：先展开 138 GFLOPs，吸收 1.17 GFLOPs；吸收的 FLOPs 是缓存展开 K、V 的 3.2 倍，但每个缓存 token 只读 1/71 的字节。
- V4 的分组输出投影是分块低秩：每组秩 ≤ 1024，整体仍可满秩。
:::

# 练习与答案 | 11 题：概念、计算、推导 | 45

先自己做，再展开答案。每道计算题都能用纸笔完成，也都可以用几行 NumPy 核对。

## 概念题

:::: quiz 练习 1 · MQA 与 MLA，谁的“方向”多
多查询注意力（MQA）让所有头共用一个键 $\mathbf k_t=W^K\mathbf h_t$，$W^K\in\R^{128\times5120}$；MLA 让 128 个头的键都由 $\mathbf c_t^{KV}\in\R^{512}$ 展开。（a）分别写出“从 $\mathbf h_t$ 到全部 128 个头的键”（拼成 16384 维）的等效矩阵的秩上界。（b）两者每层每个 token 缓存多少个数（MLA 算上 64 维的 RoPE 键）？（c）用这两组数说明 MLA“头数一个没减”是什么意思。
::: answer 看答案
（a）MQA：等效矩阵是 128 个 $W^K$ 上下叠起来，行与行大量重复，秩 $\le\rank W^K\le128$。MLA：$W^{UK}W^{DKV}$，秩 $\le512$ (M2.3)。

（b）MQA 存一个键和一个值，$2\times128=256$ 个数；MLA 存 $512+64=576$ 个数（第 9 课的表）。

（c）MQA 的 128 个头拿到的是**同一个** 128 维的键，彼此没有差别；MLA 的每个头用自己的 $W^{UK}_i$ 从 512 维的潜向量里读出**不同的** 128 维视图，128 个头的键合起来可以张成至多 512 维。MLA 用约 2.25 倍于 MQA 的缓存，换来 4 倍的“方向数”和各头不同的键。
:::
::::

:::: quiz 练习 2 · 定理的名字与前提
（a）用一句话说出 Frobenius 版本与谱范数版本的最优低秩近似定理各归功于谁。（b）MLA 直接训练出的低秩矩阵，在一些基准上比满秩的 MHA 得分还高。这和“截断 SVD 是最优的”矛盾吗？
::: answer 看答案
（a）Frobenius 版本最早是 Schmidt 1907，Eckart 与 Young 1936 以最小二乘形式重新得到；推广到一切酉不变范数（谱范数是其中之一）的是 Mirsky 1960（按 Stewart 1993 的考证；谱范数下界那一步在 Stewart 转述的 Weyl 论证里已经出现，见 §5 的“常见说法”框）。常说的“Eckart–Young 定理”是一个不准确的名字。

（b）不矛盾。定理回答的是“用各向同性的尺子逼近一个**固定**矩阵”；MLA 的训练目标按数据加权（§6 的 (M2.19)），而且整个网络与瓶颈共同适应，要逼近的“矩阵”并不固定。定理只说“给定 MHA 的权重，截断 SVD 是最好的事后压缩”。至于 MLA 为什么有时得分更高，论文没有解释。
:::
::::

:::: quiz 练习 3 · 奇异值与特征值
（a）一个 $2\times2$ 矩阵的特征值是 3 和 5，它的最大奇异值一定是 5 吗？举例说明。（b）什么样的矩阵，奇异值恰好等于特征值？
::: answer 看答案
（a）不一定。§3 的 $A=\begin{pmatrix}3&0\\4&5\end{pmatrix}$ 特征值是 3、5，奇异值却是 $3\sqrt5\approx6.71$ 和 $\sqrt5\approx2.24$：沿 $\mathbf v_1$ 方向它把向量拉长 6.71 倍。

（b）对称且半正定（特征值都 $\ge0$）的矩阵。这时 $A=Q\Lambda Q^\top$ 本身就是一个 SVD（$U=V=Q$，$\Sigma=\Lambda$）。对称但有负特征值时，奇异值是特征值的绝对值（负号放进奇异向量，见练习 4）。
:::
::::

## 计算题

:::: quiz 练习 4 · 手算 SVD 与最优秩 1 近似
$A=\begin{pmatrix}1&2\\2&1\end{pmatrix}$。（a）求 SVD。（b）求最优秩 1 近似 $A_1$ 及它在两种范数下的误差。（c）$\norm{A}_2$、$\norm{A}_F$ 各是多少？
::: answer 看答案
（a）$A^\top A=A^2=\begin{pmatrix}5&4\\4&5\end{pmatrix}$，迹 10，行列式 9，$\lambda^2-10\lambda+9=0$，$\lambda=9,1$，所以 $\sigma_1=3$、$\sigma_2=1$。$\lambda=9$：$\mathbf v_1=\tfrac1{\sqrt2}(1,1)$；$\lambda=1$：$\mathbf v_2=\tfrac1{\sqrt2}(1,-1)$。$\mathbf u_1=A\mathbf v_1/3=\tfrac1{\sqrt2}(1,1)$；$\mathbf u_2=A\mathbf v_2/1=\tfrac1{\sqrt2}(-1,1)=-\mathbf v_2$。$A$ 的特征值是 3 和 $-1$：负号跑进了 $\mathbf u_2=-\mathbf v_2$，奇异值取绝对值。

（b）$A_1=3\,\mathbf u_1\mathbf v_1^\top=\tfrac32\begin{pmatrix}1&1\\1&1\end{pmatrix}$。$A-A_1=\begin{pmatrix}-0.5&0.5\\0.5&-0.5\end{pmatrix}=\sigma_2\mathbf u_2\mathbf v_2^\top$，谱误差 $\sigma_2=1$，F 误差 $\sqrt{4\times0.25}=1$（只剩一个奇异值，两者相等）。

（c）$\norm{A}_2=3$；$\norm{A}_F=\sqrt{1+4+4+1}=\sqrt{10}\approx3.162=\sqrt{3^2+1^2}$。
:::
::::

:::: quiz 练习 5 · 查询压缩的账
DeepSeek-V2 的查询压缩：$W^{DQ}\in\R^{1536\times5120}$，$W^{UQ}\in\R^{16384\times1536}$。（a）与满秩的 $W^Q\in\R^{16384\times5120}$ 相比，参数少几倍？每个 token 的 FLOPs 各是多少？（b）$W^{UQ}W^{DQ}$ 的秩至多多少？若把 $d_c'$ 提高到 4096，还省参数吗？
::: answer 看答案
（a）分解：$1536\times5120+16384\times1536=7{,}864{,}320+25{,}165{,}824=33{,}030{,}144$；满秩 $83{,}886{,}080$；少 2.54 倍。FLOPs：$2\times33{,}030{,}144\approx6.61\times10^7$ 对 $2\times83{,}886{,}080\approx1.68\times10^8$。

（b）秩至多 1536 (M2.3)。临界秩 $r^*=16384\times5120/21504\approx3901$，$4096>3901$，此时分解后 $4096\times21504=88{,}080{,}384$ 个参数，比满秩还多 5%，不再省参数（论文做查询压缩的理由是训练时省激活显存，见 §2）。
:::
::::

:::: quiz 练习 6 · 幂律谱的截断
某矩阵有 10 个奇异值 $\sigma_i=1/i$（$i=1,\dots,10$）。（a）最优秩 3 近似的谱误差、F 误差各是多少？保住了多少能量？（b）要保住 95% 以上的能量，至少要几项？
::: answer 看答案
（a）谱误差 $\sigma_4=0.25$。$\sum_{i=1}^{10}1/i^2=1.54977$，前 3 项 $1+0.25+0.1111=1.36111$，保住 $87.8\%$；F 误差 $\sqrt{1.54977-1.36111}=\sqrt{0.18866}\approx0.4343$。

（b）逐项累加的能量比例：$k=4$ 为 91.9%，$k=5$ 为 94.4%，$k=6$ 为 96.2%。至少 6 项。
:::
::::

:::: quiz 练习 7 · 吸收从哪个 T 开始划算
用 (M2.22)：（a）$T=1$ 与 $T=2$ 时，(a) 与 (c) 各多少 FLOPs？（b）解出两者相等的 $T$，并解释为什么常数项几乎等于 (a) 的斜率。
::: answer 看答案
（a）$T=1$：(a) 33,636,352，(c) $33{,}554{,}432+278{,}528=33{,}832{,}960$，吸收略贵；$T=2$：(a) 67,272,704，(c) 34,111,488，吸收已经便宜近一半。

（b）$33{,}636{,}352\,T=33{,}554{,}432+278{,}528\,T$ 得 $T=33{,}554{,}432/33{,}357{,}824\approx1.006$。原因：(c) 的常数项是每步对**一个**查询做一次“128 个头 × $[128,512]$”的变换（两侧各一次），(a) 的斜率是对**每个**缓存 token 做同样规模的变换（再加一点点积），所以只要缓存里多于一个 token，吸收就赢。
:::
::::

:::: quiz 练习 8 · 秩预算该留给哪个方向
$W=\diag(3,\ 1)$，输入满足 $S_x=\E[\mathbf x\mathbf x^\top]=\diag(0.01,\ 1)$：第一个坐标方向上的输入很弱（方差 0.01），第二个方向正常。（a）截断 $W$ 自己的 SVD，最优秩 1 近似 $W_1$ 是什么？按数据加权的相对误差 $\E\norm{(W-B)\mathbf x}^2/\E\norm{W\mathbf x}^2$ 是多少？（b）用 (M2.20) 求按数据加权的最优秩 1 近似 $B^*$ 和它的相对误差。（c）改用普通的 Frobenius 范数量 $W-W_1$ 与 $W-B^*$，哪个更小？这说明了什么？
::: answer 看答案
先算分母：由 (M2.19)，$\E\norm{W\mathbf x}^2=\tr(WS_xW^\top)=3^2\times0.01+1^2\times1=1.09$。

（a）$W$ 的奇异值是 3 和 1，截断保留放大 3 倍的第一个方向：$W_1=\diag(3,0)$。误差矩阵 $W-W_1=\diag(0,1)$，$\E\norm{(W-W_1)\mathbf x}^2=1^2\times1=1$，相对误差 $1/1.09\approx91.7\%$，几乎什么都没留住。

（b）$S_x^{1/2}=\diag(0.1,\ 1)$，$WS_x^{1/2}=\diag(0.3,\ 1)$，奇异值是 1 和 0.3，大的那个在**第二个**方向。截断得 $(WS_x^{1/2})_1=\diag(0,1)$，于是 $B^*=\diag(0,1)\,S_x^{-1/2}=\diag(0,1)\,\diag(10,1)=\diag(0,1)$。误差是被丢掉的 $\sigma_2^2(WS_x^{1/2})=0.3^2=0.09$，相对误差 $0.09/1.09\approx8.3\%$，比 (a) 小约 11 倍。

（c）$\norm{W-W_1}_F=\norm{\diag(0,1)}_F=1$，$\norm{W-B^*}_F=\norm{\diag(3,0)}_F=3$：用各向同性的尺子量，$W_1$ 反而更好，两把尺子给出相反的结论。$W$ 放大得最厉害的方向恰好是输入几乎不去的方向，只看 $W$ 自己的谱，会把唯一的秩预算浪费在那里。这就是 §6 说“尺子应当按数据加权”的原因，也是图 M2.8 右图两条谱差别的来源。
:::
::::

## 推导题

:::: quiz 练习 9 · rank(AᵀA) = rank A
证明 $N(A^\top A)=N(A)$，并由秩–零化度定理得出 $\rank(A^\top A)=\rank A$。这说明 §3 用 $A^\top A$ 的非零特征值个数数奇异值为什么不会漏掉、也不会多出。
::: answer 看答案
若 $A\mathbf x=\mathbf 0$，则 $A^\top A\mathbf x=\mathbf 0$，所以 $N(A)\subseteq N(A^\top A)$。反过来，若 $A^\top A\mathbf x=\mathbf 0$，两边左乘 $\mathbf x^\top$：$\mathbf x^\top A^\top A\mathbf x=\norm{A\mathbf x}^2=0$，所以 $A\mathbf x=\mathbf 0$，$N(A^\top A)\subseteq N(A)$。两个零空间相同，维数相同；$A$ 与 $A^\top A$ 都有 $n$ 列，由 (M2.2)，$\rank(A^\top A)=n-\dim N(A^\top A)=n-\dim N(A)=\rank A$。

$A^\top A$ 是对称的，它的秩等于非零特征值的个数（按重数计），而这些非零特征值正是 $\sigma_i^2>0$。所以“正奇异值的个数 = 秩”与 §3 的结论一致。
:::
::::

:::: quiz 练习 10 · 两种范数都是酉不变的
设 $Q\in\R^{m\times m}$、$R\in\R^{n\times n}$ 正交。证明 $QAR$ 与 $A$ 有相同的奇异值，从而 $\norm{QAR}_2=\norm{A}_2$、$\norm{QAR}_F=\norm{A}_F$。
::: answer 看答案
设 $A=U\Sigma V^\top$。则 $QAR=(QU)\,\Sigma\,(R^\top V)^\top$。$QU$ 是正交矩阵：$(QU)^\top QU=U^\top Q^\top QU=U^\top U=I$；$R^\top V$ 同理。所以这就是 $QAR$ 的一个 SVD，奇异值仍是 $\Sigma$ 的对角元。由 (M2.14) 与 (M2.12)，两种范数都只由奇异值决定，所以都不变。

直观地说：旋转输入、旋转输出都不改变“椭圆”的轴长，只改变它的朝向。这正是 §5 定理对这两种范数都成立的原因。
:::
::::

:::: quiz 练习 11 · 截断误差的 Frobenius 公式
不借助 (M2.12)，直接用迹证明 $\norm{\sum_{i\in S}c_i\mathbf u_i\mathbf v_i^\top}_F^2=\sum_{i\in S}c_i^2$，其中 $\{\mathbf u_i\}$、$\{\mathbf v_i\}$ 各自正交归一，$S$ 是任意下标集合；由此得出 $\norm{A-A_k}_F^2=\sum_{i>k}\sigma_i^2$。
::: answer 看答案
记 $E=\sum_{i\in S}c_i\mathbf u_i\mathbf v_i^\top$。$\norm{E}_F^2=\tr(E^\top E)=\tr\Big(\sum_{i,j\in S}c_ic_j\,\mathbf v_i\mathbf u_i^\top\mathbf u_j\mathbf v_j^\top\Big)$。$\mathbf u_i^\top\mathbf u_j$ 只在 $i=j$ 时为 1，否则为 0，所以只剩 $\sum_{i\in S}c_i^2\tr(\mathbf v_i\mathbf v_i^\top)=\sum_{i\in S}c_i^2\,\mathbf v_i^\top\mathbf v_i=\sum_{i\in S}c_i^2$（迹的轮换）。取 $S=\{k+1,\dots,r\}$、$c_i=\sigma_i$，$E=A-A_k$，即得 $\norm{A-A_k}_F^2=\sum_{i>k}\sigma_i^2$，与 (M2.16) 一致。
:::
::::

```refs
- id: dsv2
  title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  a: DeepSeek-AI (2024). arXiv:2405.04434（读的是 v5）
  url: https://arxiv.org/abs/2405.04434
  finding: §2.1.2 用 $W^{DKV}$、$W^{UK}$、$W^{UV}$ 做低秩 KV 联合压缩并对查询做低秩压缩，推理时 $W^{UK}$、$W^{UV}$ 可被吸收；§3.1.2 给出 $d=5120$、$n_h=d_h=128$、$d_c=512$、$d_c'=1536$、$d_h^R=64$；附录 D.2 比较 MLA 与 MHA。
  matters: 本课所有真实维度的算例与 MLA/MHA 消融表都以它为准（经第 9 课研究笔记核对）。
- id: axler2024
  title: "Linear Algebra Done Right (fourth edition)"
  a: Sheldon Axler (2024). Springer；开放获取（CC BY-NC 4.0），读的是作者网站 PDF
  url: https://linear.axler.net/
  finding: 2.22、2.34 基与维数；3.21 线性映射基本定理；3.56–3.58 列–行分解与列秩等于行秩；6.36 正交归一组可扩充为正交归一基；7.29 实谱定理；7.70、7.80 SVD；7.85–7.86 范数等于最大奇异值；7.92 秩不超过 $k$ 的最佳近似误差为 $s_{k+1}$。
  matters: 本课引用的标准定理都给出这本免费教材的精确编号，§5 谱范数版本的证明按它的 7.92 写成。
- id: mml2020
  title: "Mathematics for Machine Learning"
  a: M. P. Deisenroth, A. A. Faisal, C. S. Ong (2020). Cambridge University Press（作者网站免费 PDF）
  url: https://mml-book.github.io/
  finding: 定理 4.22 陈述 SVD 存在；定义 4.23 与定理 4.24 给出谱范数等于最大奇异值；定理 4.25（§4.6）给出谱范数下最优秩 $k$ 近似的误差 $\sigma_{k+1}$。
  matters: SVD 与谱范数的免费教材出处；它把定理 4.25 归给 Eckart–Young 1936，本课按 Stewart 改正。
- id: stewart1993
  title: "On the Early History of the Singular Value Decomposition"
  a: G. W. Stewart (1993). SIAM Review 35(4):551–566, doi:10.1137/1035134
  url: https://doi.org/10.1137/1035134
  finding: 梳理 SVD 的五位先驱，指出最优低秩近似定理属于 Schmidt 1907、“Eckart–Young 定理”是误称，Mirsky 1960 推广到一切酉不变范数；§6 转述 Weyl 的论证：式 (6.2) $\sigma_1(A-B_k)\ge\sigma_{k+1}$，再由式 (6.3) 推出 Frobenius 版本。
  matters: 本课定理归属、Weyl 不等式与 Frobenius 证明概要的依据。
- id: schmidt1907
  title: "Zur Theorie der linearen und nichtlinearen Integralgleichungen. I. Teil"
  a: Erhard Schmidt (1907). Mathematische Annalen 63(4):433–476, doi:10.1007/BF01449770
  url: https://doi.org/10.1007/BF01449770
  finding: 在积分方程框架下证明，用至多 $k$ 个秩一项做最小二乘近似时最优解是前 $k$ 项截断（按 Stewart 1993 §5 转述，原文未读）。
  matters: Frobenius 版本的最早出处，用来纠正“Eckart–Young 定理”的归属。
- id: eckart1936
  title: "The Approximation of One Matrix by Another of Lower Rank"
  a: Carl Eckart, Gale Young (1936). Psychometrika 1(3):211–218, doi:10.1007/BF02288367
  url: https://doi.org/10.1007/BF02288367
  finding: 把低秩近似写成最小二乘问题，借正则形式求解，指出解总存在、通常唯一（据摘要，全文未读）。
  matters: 说明 1936 年处理的是 Frobenius 问题而不是谱范数问题。
- id: mirsky1960
  title: "Symmetric Gauge Functions and Unitarily Invariant Norms"
  a: L. Mirsky (1960). The Quarterly Journal of Mathematics 11(1):50–59, doi:10.1093/qmath/11.1.50
  url: https://doi.org/10.1093/qmath/11.1.50
  finding: 截断 SVD 在任意酉不变范数下都是最优低秩近似（按 Stewart 1993 §7 转述，原文未读）。
  matters: 谱范数版本与“一切酉不变范数”推广的正确出处。
- id: isca25
  title: "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures"
  a: Chenggang Zhao et al. (2025). ISCA '25. arXiv:2505.09343
  url: https://arxiv.org/abs/2505.09343
  finding: §2.1.2 说明 MLA 用一个与模型联合训练的投影矩阵压缩所有头的 KV 表示，推理时只缓存潜向量。
  matters: “MLA 的压缩从头学出，不是对 MHA 权重做 SVD”的出处（经第 9 课研究笔记核对）。
- id: dsv4
  title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
  a: DeepSeek-AI (2026). arXiv:2606.19348（读的是 v1）
  url: https://arxiv.org/abs/2606.19348
  finding: 注意力保留查询低秩压缩，并用分组输出投影先按组投到 $d_g$ 维再投到 $d$；Muon 的混合牛顿–舒尔茨迭代最后 2 步把更新矩阵的奇异值稳定在 1。
  matters: 第 18 课方向的落脚点：分块低秩分解与只作用于奇异值的迭代（数字经第 18 课研究笔记核对）。
- id: numpy-matrix-rank
  kind: artefact
  title: "NumPy 文档：numpy.linalg.matrix_rank"
  a: NumPy 开发者（本机 NumPy 1.26.4 的 docstring）. 读取于 2026-09-24
  url: https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html
  finding: 用 SVD 计秩：数大于阈值的奇异值；阈值缺省为 S.max() × max(M, N) × eps。
  matters: 说明数值秩是带阈值的。
```
