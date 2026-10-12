---
title: M03 · 正交矩阵、旋转与复数：RoPE 用到的全部数学
short: 旋转与复数
kicker: 数学补课 · M03
description: 正交矩阵为什么保持长度与夹角；二维旋转矩阵的推导与和角公式；从 i² = −1 讲起的复数、欧拉公式 e^{iφ} = cos φ + i sin φ 与“乘法就是旋转”；旋转的复合与相对角恒等式 R(α)ᵀR(β) = R(β−α) 的证明；旋转与一般矩阵不可交换；d 维块对角旋转、多尺度频率与波长；一个带数字的完整 RoPE 算例，以及它为什么挡住 MLA 的矩阵吸收。
lede: 第 5 课的 RoPE 只用一句话描述：把查询和键按各自的位置**旋转**一个角度。这一课把这句话背后的数学从头讲透：什么样的矩阵不会破坏向量的内容，二维旋转矩阵从哪里来，为什么一个复数乘法就是一次旋转，为什么“先倒转 α、再正转 β”只剩下 β − α，以及为什么同一个旋转夹在两个权重矩阵中间时，就再也挪不出去。
meta: 预计阅读 150 分钟，练习另需约 45 分钟 · 前置：M01（矩阵乘法、转置、分块矩阵）、高中三角函数 · 服务于第 5、9、10 课
index: index.html
prev: m02-rank-svd-lowrank.html | M02 · 低秩与 SVD
next: m04-probability.html | M04 · 概率与方差
---

## 这一课回答什么问题

[第 5 课](../deepseek/lesson-05-modern-block.html#p2)要给注意力的查询和键注入位置信息，而且希望打分只依赖两个 token 的**相对**距离。它得到的答案叫旋转位置编码（rotary position embedding，RoPE）：位置 $t$ 的查询 $\mathbf q$ 被旋转 $t\theta$，位置 $j$ 的键 $\mathbf k$ 被旋转 $j\theta$，于是两者的点积只依赖 $t-j$ [@roformer]。这一句话里藏着五个数学问题，本课逐一回答：

- 用一个矩阵去乘查询和键，怎样才能**不破坏内容**，即不改变向量的长度、也不改变向量之间的夹角？
- “旋转 $\varphi$ 角”写成矩阵是什么？为什么是 $\cos$ 和 $\sin$ 那样排列？
- 为什么说复数乘法就是旋转？欧拉公式 $e^{\mathrm i\varphi}=\cos\varphi+\mathrm i\sin\varphi$ 从哪里来，它为什么让旋转的计算变成指数相加？
- 为什么 $\mathbf R(\alpha)^\top\mathbf R(\beta)=\mathbf R(\beta-\alpha)$？它怎样推出“RoPE 的打分只依赖相对位置”？
- 一个 128 维的头怎样同时装下几十个转速不同的旋转？这些转速为什么取成几何级数？为什么 DeepSeek-V2 说 RoPE 与低秩 KV 压缩“不兼容” [@deepseekv2]？

## 这一课在科技树上的位置

这一课主要服务两课。[第 5 课](../deepseek/lesson-05-modern-block.html)用本课的全部工具推出 RoPE：复数与欧拉公式（本课 [§3](#p3)、[§4](#p4)）、相对角恒等式（[§5](#p5)）、块对角旋转与频率（[§6](#p6)）。[第 9 课](../deepseek/lesson-09-v2-mla.html#p4)解释 MLA 为什么要“解耦 RoPE”：一个随位置变化的旋转矩阵夹在两个权重矩阵之间，而旋转与一般矩阵不能交换次序（本课 [§5](#p5)、[§7](#p7)）。[第 10 课 §6](../deepseek/lesson-10-v2-moe-training.html#p6) 用 YaRN 把 DeepSeek-V2 的上下文从 4K 拉到 128K，要处理的正是本课 [§6](#p6) 的“慢针”。它也接着 [M01](m01-vectors-matrices.html) 讲：M01 的点积、转置规则、“矩阵的列是坐标轴的去处”与分块矩阵，这里都要用到；[M02](m02-rank-svd-lowrank.html) 的奇异值分解里那两个“旋转”，也正是本课的正交矩阵。

## 读完你能……

1. 说出正交矩阵的定义 $U^\top U=I$ 与它的三个后果（保持点积、保持长度、逆矩阵就是转置），并判断一个矩阵是不是正交矩阵。
2. 从“两个坐标轴转到哪里”推出二维旋转矩阵 $\mathbf R(\varphi)$，并由它读出三角函数的和角公式。
3. 做复数的加法、乘法、共轭、求模，把复数乘法翻译成“拉伸 + 旋转”，并用 $z\bar w$ 的实部算二维点积。
4. 用两种方法推出欧拉公式，用极坐标形式 $re^{\mathrm i\varphi}$ 做乘法；证明 $\mathbf R(\alpha)\mathbf R(\beta)=\mathbf R(\alpha+\beta)$ 与 $\mathbf R(\alpha)^\top\mathbf R(\beta)=\mathbf R(\beta-\alpha)$，并说明哪些矩阵能与所有旋转交换次序。
5. 写出 $d$ 维块对角旋转、频率 $\theta_i=b^{-2i/d}$ 与波长；手算一个 4 维 RoPE 打分，并解释它为什么挡住 MLA 的矩阵吸收。

::: note 本课的记号
- 向量是列向量，粗体小写；矩阵大写。$\mathbf e_1=(1,0)$、$\mathbf e_2=(0,1)$ 是二维的两个坐标向量。
- 角度一律用**弧度**（下一页的前置框会复习），逆时针为正。旋转矩阵写 $\mathbf R(\varphi)$；一般的正交矩阵写 $U$。
- 虚数单位写正体 $\mathrm i$（$\mathrm i^2=-1$），与频率下标 $i$ 区分；这与第 5 课的写法一致。复数 $z$ 的共轭写 $\bar z$，实部、虚部写 $\operatorname{Re}z$、$\operatorname{Im}z$。
- 位置沿用整门课的约定：$t$ 是当前（查询）位置，$j$ 是被看的（键）位置。RoPE 的频率写 $\theta_i$，底数写 $b$，预训练上下文长度写 $L_{ctx}$。
- [§6](#p6) 起的 $d$ 沿用 RoFormer 的记号，指**被旋转的向量的维数**：用在一个注意力头上就是每头维度 $d_h$，用在 DeepSeek 的解耦 RoPE 上就是 $d_h^R=64$；它不是全局符号表里的模型隐藏维度。推导里临时出现的 $a,b,c,d$（矩阵元素、复数的实部虚部、某个平面上的坐标）只在那一段内有效，与底数 $b$、维数 $d$ 无关。
:::

[[legend]]

[[toc]]

# 正交矩阵 | 不改变长度与夹角的变换 | 20

先把问题摆出来。第 5 课想给查询和键“加上位置”。最朴素的办法是 GPT-2 的做法（[第 5 课 §1](../deepseek/lesson-05-modern-block.html#p1)）：给每个位置学一个向量，**加**到输入上。加法的毛病是它把位置和内容搅在一起：同一个内容向量，加上不同的位置向量之后，长度和方向都变了，而且变多少取决于两个向量碰巧的相对方向。RoPE 换了一个思路：用一个随位置变化的**矩阵去乘**。那么问题就变成：什么样的矩阵乘上去，不会破坏向量携带的内容？

“不破坏内容”至少要做到两件事。第一，长度不变：[M01 §1](m01-vectors-matrices.html#p1)讲过，查询的长度起 softmax“温度”的作用，键的长度是它的“嗓门”，位置不该去改它们。第二，夹角不变：点积打分靠的就是夹角。这一页找出所有满足这两点的矩阵，它们叫**正交矩阵**。

## 要求：点积不变

**含义。** 如果一个方阵 $U$ 对任意两个向量都满足 $(U\mathbf x)\cdot(U\mathbf y)=\mathbf x\cdot\mathbf y$，就说它**保持点积**。

**理解。** 点积同时决定了长度（$\mathbf x\cdot\mathbf x=\norm{\mathbf x}^2$）和夹角（$\cos\angle(\mathbf x,\mathbf y)=\mathbf x\cdot\mathbf y/(\norm{\mathbf x}\norm{\mathbf y})$；本课的 $\theta$ 留给 RoPE 的转速，夹角写成 $\angle$）。所以“保持点积”就是：变换之后，每根箭头一样长，任意两根箭头的夹角也一样。整个图形像一张画在透明胶片上的图，被整体转动，或者翻个面，但没有被拉伸、压扁或扭歪。

**数学。** 把条件写成矩阵，看它对 $U$ 提出了什么要求。

::: deriv 保持点积 ⟺ UᵀU = I
$(U\mathbf x)^\top(U\mathbf y)=\mathbf x^\top U^\top U\,\mathbf y$ || 乘积的转置等于转置的逆序乘积 $(U\mathbf x)^\top=\mathbf x^\top U^\top$（[M01 §3](m01-vectors-matrices.html#p3)），再用结合律
$=\mathbf x^\top\mathbf y\qquad(\text{若 }U^\top U=I)$ || 充分性：中间的矩阵是单位矩阵，直接消掉
$\mathbf e_a^\top(U^\top U)\,\mathbf e_b=(U^\top U)_{ab}$ || 必要性，反过来取 $\mathbf x=\mathbf e_a$、$\mathbf y=\mathbf e_b$：夹在两个坐标向量之间，读出的正好是第 $(a,b)$ 格
$\mathbf e_a^\top\mathbf e_b=\delta_{ab}$ || 坐标向量长度为 1、两两垂直；$\delta_{ab}$ 叫克罗内克符号，$a=b$ 时为 1，否则为 0
$(U^\top U)_{ab}=\delta_{ab}\ \ \text{对所有 }a,b\iff U^\top U=I$ || 保持点积对每一对坐标向量都成立，前两行的左边必须相等
:::

::: eq 正交矩阵的定义 | (M3.1)
$$U\in\R^{n\times n},\qquad U^\top U=I$$
$U^\top U$ 的第 $(a,b)$ 格是 $U$ 的第 $a$ 列与第 $b$ 列的点积（[M01 §2](m01-vectors-matrices.html#p2) 的“批量点积”读法），所以 (M3.1) 等价于：**$U$ 的各列长度为 1、两两垂直**（正交归一）。
:::

名字里只说“正交”，定义却要求各列**正交归一**，这是历史沿用下来的叫法，不必深究。按上面的推导，正交矩阵带来下面三个后果。

::: eq 正交矩阵保持点积、长度与夹角 | (M3.2)
$$(U\mathbf x)^\top(U\mathbf y)=\mathbf x^\top\mathbf y,\qquad \norm{U\mathbf x}=\norm{\mathbf x},\qquad \cos\angle(U\mathbf x,U\mathbf y)=\cos\angle(\mathbf x,\mathbf y)$$
第二式是第一式取 $\mathbf y=\mathbf x$ 再开方；第三式是前两式代入余弦的定义。
:::

反过来也对：一个方阵只要保持**所有长度**，就自动保持点积，因而也是正交矩阵。理由是点积可以只用长度表示出来：

::: deriv 只保持长度就够了
$\norm{\mathbf x+\mathbf y}^2=\norm{\mathbf x}^2+2\,\mathbf x^\top\mathbf y+\norm{\mathbf y}^2$ || 展开 $(\mathbf x+\mathbf y)^\top(\mathbf x+\mathbf y)$，点积对称，两个交叉项相等
$\norm{\mathbf x-\mathbf y}^2=\norm{\mathbf x}^2-2\,\mathbf x^\top\mathbf y+\norm{\mathbf y}^2$ || 同理
$\mathbf x^\top\mathbf y=\tfrac14\big(\norm{\mathbf x+\mathbf y}^2-\norm{\mathbf x-\mathbf y}^2\big)$ || 两式相减再除以 4（这叫极化恒等式）
$(U\mathbf x)^\top(U\mathbf y)=\tfrac14\big(\norm{U\mathbf x+U\mathbf y}^2-\norm{U\mathbf x-U\mathbf y}^2\big)$ || 对 $U\mathbf x$、$U\mathbf y$ 用同一个恒等式
$=\tfrac14\big(\norm{\mathbf x+\mathbf y}^2-\norm{\mathbf x-\mathbf y}^2\big)=\mathbf x^\top\mathbf y$ || $U\mathbf x\pm U\mathbf y=U(\mathbf x\pm\mathbf y)$，而 $U$ 保持长度；最后一步再用第 (3) 行
:::

所以“RoPE 不改变向量长度”这一句，已经蕴含了：**同一个位置** $t$ 的旋转不改变经它变换的任意两个向量之间的夹角，位置 $t$ 上的整个向量空间只是被刚性地转了一下，没有被拉伸或扭歪。注意前提是“同一个矩阵”：位置不同的两个向量（比如位置 $t$ 的查询与位置 $j$ 的键）被转了不同的角度，它们之间的夹角会变，[§5](#p5) 正是靠这个变化编码相对位置。RoFormer 论文特意指出它的旋转矩阵是正交矩阵，并认为这保证了编码位置信息过程中的稳定 [@roformer]。

## 逆矩阵就是转置

正交矩阵还有一个非常实用的性质：**撤销**它不需要解方程，转置一下就行。

::: deriv 方阵的 UᵀU = I 推出 UUᵀ = I
$U\mathbf x=\mathbf 0\ \Rightarrow\ \norm{\mathbf x}=\norm{U\mathbf x}=0\ \Rightarrow\ \mathbf x=\mathbf 0$ || (M3.2)：长度保持不变，所以只有零向量被送到零
$U^{-1}\ \text{存在}$ || 一个方阵只把零向量送到零，它就是可逆的（各列线性无关）
$U^\top=U^\top(UU^{-1})=(U^\top U)U^{-1}=U^{-1}$ || 插入 $UU^{-1}=I$，用结合律，再用 (M3.1)
$UU^\top=UU^{-1}=I$ || 代入上一行：所以各**行**也正交归一
:::

::: eq 逆矩阵等于转置 | (M3.3)
$$U^{-1}=U^\top,\qquad UU^\top=U^\top U=I$$
求一般 $n\times n$ 矩阵的逆要约 $n^3$ 量级的运算；正交矩阵的逆只是把行列互换，不需要任何算术。
:::

::: warn 只对方阵成立：列正交归一的“瘦”矩阵
[M01 §6](m01-vectors-matrices.html#p6)讲投影时用过 $U\in\R^{d\times k}$（$k<d$），它也满足 $U^\top U=I_k$，但 $UU^\top$ **不是**单位矩阵，而是投影矩阵。上面推导失败在第二行：$U$ 仍然只把零向量送到零（第一行照样成立），但它不是方阵，谈不上逆矩阵；反过来，$U^\top$ 把 $d$ 维压到 $k$ 维，必然有非零向量 $\mathbf v$ 被 $U^\top$ 送到零，于是 $UU^\top\mathbf v=\mathbf 0\ne\mathbf v$。这种矩阵叫**半正交**（semi-orthogonal）或“列正交归一”矩阵。[M07 §6](m07-optimization.html#p6) 的 Muon 优化器想把更新矩阵换成的 $UV^\top$，一般就是长方形的半正交矩阵（矩阵是“宽”的时候，换成各**行**正交归一，即 $UV^\top(UV^\top)^\top=I$）；实际的牛顿–舒尔茨迭代只算出它的近似，奇异值落在 1 附近而不恰好是 1（同见 M07 §6）。只有方阵才享受 (M3.3)。
:::

## 两类正交矩阵：旋转与镜像

图 M3.1 把三个 $2\times2$ 矩阵作用在单位圆上。前两个是正交矩阵，第三个不是：它是**剪切**（shear）$S$，把每个点的横坐标加上它的纵坐标、纵坐标不变，像把一摞扑克牌推斜。

::: fig figures/m03-orthogonal.svg | 图 M3.1 正交矩阵把单位圆送到单位圆，剪切矩阵把它压成椭圆
**看三个面板里实线画的“像”：左、中两个仍是单位圆，右边变成了斜椭圆。** 彩色粗箭头是矩阵的两列，也就是 $\mathbf e_1$、$\mathbf e_2$ 被送到的位置（[M01 §2](m01-vectors-matrices.html#p2)）。$U$ 的两列长度为 1、互相垂直，整个平面被转了 $53.13°$；$P$ 交换两个坐标，把平面沿直线 $y=x$ 翻了个面；剪切 $S$ 的第二列 $(1,1)$ 长 $\sqrt2$，不正交，于是 $\mathbf x=(2,1)$ 被拉长到 $3.162$。每个点都由生成脚本算出。
:::

::: ex 验证正交，并看点积是否保持
$$U=\frac15\begin{pmatrix}3&-4\\4&3\end{pmatrix},\qquad S=\begin{pmatrix}1&1\\0&1\end{pmatrix},\qquad \mathbf x=\begin{pmatrix}2\\1\end{pmatrix},\ \ \mathbf y=\begin{pmatrix}1\\3\end{pmatrix}.$$

**$U$ 是正交矩阵。** 两列是 $(0.6,0.8)$ 与 $(-0.8,0.6)$：长度 $\sqrt{0.36+0.64}=1$；点积 $-0.48+0.48=0$。

**点积保持。** $\mathbf x^\top\mathbf y=2+3=5$。$U\mathbf x=\tfrac15(6-4,\ 8+3)=(0.4,\ 2.2)$，$U\mathbf y=\tfrac15(3-12,\ 4+9)=(-1.8,\ 2.6)$，$(U\mathbf x)^\top(U\mathbf y)=-0.72+5.72=5$。长度 $\norm{U\mathbf x}=\sqrt{0.16+4.84}=\sqrt5=\norm{\mathbf x}$。

**剪切不保持。** $S\mathbf x=(3,1)$，$S\mathbf y=(4,3)$，点积 $12+3=15\ne5$，$\norm{S\mathbf x}=\sqrt{10}\approx3.162\ne\sqrt5\approx2.236$。$\mathbf x$ 与 $\mathbf y$ 原本夹角 $45°$，剪切之后变成 $\arccos\big(15/(\sqrt{10}\cdot5)\big)\approx18.4°$。
:::

$U$ 与 $P$ 都是正交矩阵，差别在“有没有翻面”。衡量这件事的量是行列式。

::: pre 2×2 矩阵的行列式 | 知道“行列式 = 带符号的面积倍数”的可以跳过
**含义。** 行列式（determinant）$\det A$ 是矩阵 $A$ 把面积放大的倍数，带正负号。

**理解。** $A$ 把 $\mathbf e_1$、$\mathbf e_2$ 围成的单位正方形送成以 $A$ 的两列为邻边的平行四边形，这个平行四边形的面积就是 $\abs{\det A}$。符号记录朝向：第二列在第一列的逆时针一侧（与 $\mathbf e_2$ 在 $\mathbf e_1$ 的逆时针一侧相同）取正；在顺时针一侧，说明平面被翻了个面，取负。$\det A=0$ 表示平行四边形被压扁成一条线段。[M01 §2](m01-vectors-matrices.html#p2) 图 M1.3 里单位正方形变成面积为 3 的平行四边形，那个矩阵的行列式就是 3。

**数学。** $\det\begin{pmatrix}a&b\\c&d\end{pmatrix}=ad-bc$。[§3](#p3) 的“有向面积”定义框用和角公式验证了：$ad-bc$ 恰好等于两列长度之积乘以两列夹角的正弦，也就是这个带符号的面积。
:::

按公式，$\det U=\tfrac{9+16}{25}=1$，$\det P=0-1=-1$，$\det S=1-0=1$（剪切把正方形推成等底等高的平行四边形，面积不变，但长度和夹角都变了，所以行列式为 1 不等于正交）。下一页会在二维里直接证明：二维正交矩阵的行列式只能是 $+1$ 或 $-1$，$+1$ 的恰好是旋转，$-1$ 的恰好是镜像。RoPE 只用旋转。

## 乘积与逆仍是正交矩阵

两个正交矩阵相乘，结果还是正交矩阵：$(UV)^\top(UV)=V^\top U^\top UV=V^\top V=I$（第一步用转置规则，第二步用 $U^\top U=I$，第三步用 $V^\top V=I$）。逆矩阵 $U^\top$ 也是正交矩阵：$(U^\top)^\top U^\top=UU^\top=I$。用几何的话说：刚性地动两次还是刚性地动，撤销一次刚性运动也是刚性运动。[§5](#p5) 的“旋转接着旋转还是旋转”就是这一条在二维里的特例。

::: ana 类比：转动一张透明胶片
**映射。** 平面上的向量 = 画在透明胶片上的箭头；正交矩阵 = 把整张胶片绕原点转一下（旋转，$\det=+1$），或者把它翻过来（镜像，$\det=-1$）；非正交矩阵 = 把胶片拉伸、压扁或斜着推。

**收益。** 不用算就能读出正交矩阵的三个后果：箭头的长度不变，箭头之间的夹角不变，而且想撤销时只要反着转回去（转置）。

**失效的地方。** 胶片只有两维或三维，$d=128$ 维时“转动”的方式多得多，不能再想象成绕一根轴转；[§6](#p6) 的 RoPE 只用了其中很特殊的一类：把 128 维拆成 64 个互不干扰的平面，各转各的。另外，胶片的比喻只描述方阵；上面那个“瘦”的半正交矩阵 $U$ 虽然也保持长度，却是把 $k$ 维的胶片放进更大的 $d$ 维空间，它的转置 $U^\top$ 则把 $d$ 维压扁到 $k$ 维，丢掉的部分再也“转”不回来。
:::

::: warn 失败模式：浮点数里的“正交”会慢慢漂走
数学上 $\mathbf R(\theta)^{t}$ 就是转 $t\theta$ 的旋转，但计算机里每乘一次都有舍入误差，而且误差一步步累积。取 $\theta=0.1$，把 $\mathbf e_1$ 用 float32 的 $\mathbf R(0.1)$ 连乘 $10^5$ 次，得到的向量长度偏离 1 约 $1.2\times10^{-3}$，和精确位置相差约 $1.2\times10^{-3}$；在 float64 里是 $5\times10^{-12}$ 量级。直接用 $\cos(t\theta)$、$\sin(t\theta)$ 算第 $t$ 个位置的旋转，误差不累积：对 $t=1,\dots,10^5$，结果的长度离 1 都不到 $10^{-7}$；若角度 $t\theta$ 先用 float64 算好、再把 $\cos$、$\sin$ 存成 float32，离精确位置都不超过 $4.3\times10^{-8}$。若角度本身也在 float32 里相乘，它带着一次与角度大小成正比的舍入（float32 的相对精度约 $6\times10^{-8}$），$t\le10^5$ 时最多偏 $5.9\times10^{-4}$，但长度仍然不漂（笔者用 NumPy 实测，脚本见本课研究笔记）。所以实际代码都是先算角度 $t\theta_i$，再取 $\cos$、$\sin$，而不是把一个小旋转反复相乘：LLaMA 与 DeepSeek-V3 的官方代码用 `torch.outer` 一次算出所有位置、所有频率的角度表，再用 `torch.polar` 转成 $\cos t\theta_i+\mathrm i\sin t\theta_i$，角度是在 float32 里相乘的，属于上面的第二种情形 [@rope-code]；[第 5 课 §3](../deepseek/lesson-05-modern-block.html#p3) 的代码也是这样写的。
:::

:::: quiz 小练习：哪些是正交矩阵
判断下列矩阵是否正交；对正交的，说出它是旋转还是镜像。

$$A=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\qquad B=\frac{1}{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\qquad C=\begin{pmatrix}1&1\\-1&1\end{pmatrix}.$$

::: answer 看答案
$A$：两列 $(0,1)$、$(-1,0)$ 长度都是 1，点积 0，正交；$\det A=0\cdot0-(-1)\cdot1=1$，是旋转（逆时针 $90°$）。

$B$：两列 $(1,1)/\sqrt2$、$(1,-1)/\sqrt2$ 长度都是 1，点积 $(1-1)/2=0$，正交；$\det B=(-1-1)/2=-1$，是镜像。

$C$：两列互相垂直（点积 $1-1=0$），但长度都是 $\sqrt2$，所以 $C^\top C=2I\ne I$，**不是**正交矩阵。它等于 $\sqrt2$ 倍的一个旋转：先转 $-45°$，再整体放大 $\sqrt2$ 倍。“两列垂直”不够，还要长度为 1。
:::
::::

::: note 这一页用在哪里
[第 5 课 §3](../deepseek/lesson-05-modern-block.html#p3)：RoPE 的旋转矩阵是正交矩阵，所以不改变查询和键的长度；同一个位置的旋转也不改变经它变换的两个向量之间的夹角（位置不同，转角不同，夹角的变化正是相对位置信号）。[第 9 课 §4](../deepseek/lesson-09-v2-mla.html#p4)：推导里反复用到的 $\mathbf R^\top=\mathbf R^{-1}$ 就是 (M3.3)。[M02](m02-rank-svd-lowrank.html) 的奇异值分解 $A=U\Sigma V^\top$ 里的 $U$、$V$ 都是正交矩阵。
:::

::: key 本页要点
- 正交矩阵：$U^\top U=I$，即各列正交归一 (M3.1)。它保持点积、长度与夹角 (M3.2)；反过来，保持所有长度的线性变换一定是正交矩阵（极化恒等式）。
- 方阵的逆就是转置 (M3.3)；长方形的“半正交”矩阵只有 $U^\top U=I$，没有 $UU^\top=I$。
- 正交矩阵的乘积与逆仍是正交矩阵。行列式是带符号的面积倍数；二维正交矩阵里，行列式 $+1$ 的是旋转，$-1$ 的是镜像（翻面）。RoPE 只用旋转。
:::

# 二维旋转矩阵 | 从“坐标轴转到哪里”推出 R(φ)，顺带得到和角公式 | 20

上一页知道了旋转是行列式为 $+1$ 的正交矩阵，但还没有回答最具体的问题：**把平面逆时针转 $\varphi$ 角，是哪一个矩阵？**这一页把它推出来。推导只用 M01 的一条事实：矩阵的第 $j$ 列就是第 $j$ 个坐标向量被送到的位置。

::: pre 弧度与单位圆上的三角函数 | 熟悉弧度制与 cos、sin 的单位圆定义的可以跳过
**弧度**（radian）用弧长量角：在半径为 1 的圆上，一段圆弧有多长，它对的圆心角就是多少弧度。整圆周长 $2\pi$，所以 $360°=2\pi$，$180°=\pi$，$90°=\pi/2$，$30°=\pi/6\approx0.5236$；1 弧度 $=180°/\pi\approx57.2958°$。

**单位圆上的 cos 与 sin。** 从 $(1,0)$ 出发，沿单位圆逆时针走过角度 $\varphi$，到达的点的横坐标叫 $\cos\varphi$，纵坐标叫 $\sin\varphi$。由这个定义立刻得到三件事：① 点在单位圆上，所以 $\cos^2\varphi+\sin^2\varphi=1$；② 顺时针走同样的角度，得到关于横轴的镜像点，所以 $\cos(-\varphi)=\cos\varphi$（偶函数）、$\sin(-\varphi)=-\sin\varphi$（奇函数）；③ 多走一整圈回到原处，所以两者都以 $2\pi$ 为周期。本课只用这三条，以及下面要**推出**的和角公式。
:::

## 推导：两列就是两个坐标轴的去处

**问题。** 旋转是线性变换（它保持原点不动，把直线送到直线，把平行等距的网格送到平行等距的网格），所以它是某个 $2\times2$ 矩阵 $\mathbf R(\varphi)$。M01 的 (M1.6) 说 $A\mathbf e_j$ 等于 $A$ 的第 $j$ 列，所以只要知道两个坐标向量被转到哪里。

::: deriv 旋转矩阵的两列
$\mathbf R(\varphi)\mathbf e_1=(\cos\varphi,\ \sin\varphi)$ || $\mathbf e_1=(1,0)$ 在单位圆上角度为 0 的位置；逆时针转 $\varphi$，到达角度 $\varphi$ 的点，按定义坐标是 $(\cos\varphi,\sin\varphi)$
$(a,b)\mapsto(-b,\ a)\quad\text{是逆时针转 }90°$ || 与 $(a,b)$ 垂直（点积 $-ab+ba=0$）、等长（都是 $\sqrt{a^2+b^2}$）的向量只有 $(-b,a)$ 与 $(b,-a)$ 两个；取 $(a,b)=(1,0)$ 可见 $(-b,a)=(0,1)$ 在逆时针方向，$(b,-a)$ 在顺时针方向；$(a,b)$ 连续变动时 $(-b,a)$ 也连续变动、始终与它垂直，跳不到另一侧，所以对所有非零 $(a,b)$ 都是逆时针
$\mathbf e_2=\text{把 }\mathbf e_1\text{ 逆时针转 }90°$ || $(1,0)\mapsto(0,1)$，对照上一行
$\mathbf R(\varphi)\mathbf e_2=\text{把 }\mathbf R(\varphi)\mathbf e_1\text{ 逆时针转 }90°$ || 刚性转动不改变两根箭头之间的夹角和先后次序（转向）
$\mathbf R(\varphi)\mathbf e_2=(-\sin\varphi,\ \cos\varphi)$ || 对 $(a,b)=(\cos\varphi,\sin\varphi)$ 用第 (2) 行
:::

::: eq 二维旋转矩阵 | (M3.4)
$$\mathbf R(\varphi)=\begin{pmatrix}\cos\varphi&-\sin\varphi\\ \sin\varphi&\cos\varphi\end{pmatrix},\qquad \mathbf R(\varphi)\begin{pmatrix}x\\y\end{pmatrix}=\begin{pmatrix}x\cos\varphi-y\sin\varphi\\ x\sin\varphi+y\cos\varphi\end{pmatrix}$$
第一列是 $\mathbf e_1$ 的去处，第二列是 $\mathbf e_2$ 的去处。$\varphi>0$ 为逆时针。
:::

::: fig figures/m03-rotation-matrix.svg | 图 M3.2 旋转矩阵的两列就是两个坐标轴被转到的位置 | wide
**先看左图的两支蓝色箭头：它们就是 $\mathbf R(30°)$ 的两列 $(0.866,0.5)$ 与 $(-0.5,0.866)$，左下角把它们填进矩阵。** 虚线把两个箭头尖投到坐标轴上，读出 $\cos30°$ 与 $\sin30°$。右图把同一个旋转作用在 $\mathbf x=(2,1)$ 上：它的角度从 $26.57°$ 变成 $56.57°$，正好加了 $30°$，长度仍是 $\sqrt5$（虚线圆弧）。把“角度相加”与“矩阵乘出来的坐标”对照，就是下面的和角公式。
:::

验证它确实是正交矩阵：两列长度都是 $\sqrt{\cos^2\varphi+\sin^2\varphi}=1$，点积 $-\cos\varphi\sin\varphi+\sin\varphi\cos\varphi=0$；行列式 $\cos^2\varphi+\sin^2\varphi=1$。与上一页的结论吻合：旋转是行列式为 $+1$ 的正交矩阵。

## 顺带得到和角公式

(M3.4) 还白送一对公式。取单位圆上角度为 $\alpha$ 的点 $\mathbf u_\alpha=(\cos\alpha,\sin\alpha)$，把它再转 $\varphi$。从几何上看，结果是角度为 $\alpha+\varphi$ 的点；从代数上看，结果是 $\mathbf R(\varphi)\mathbf u_\alpha$。两种算法的答案必须相同：

::: deriv 和角公式
$\mathbf R(\varphi)\mathbf u_\alpha=\begin{pmatrix}\cos\varphi\cos\alpha-\sin\varphi\sin\alpha\\ \sin\varphi\cos\alpha+\cos\varphi\sin\alpha\end{pmatrix}$ || 代数：代入 (M3.4)，$x=\cos\alpha$，$y=\sin\alpha$
$\mathbf R(\varphi)\mathbf u_\alpha=\mathbf u_{\alpha+\varphi}=\begin{pmatrix}\cos(\alpha+\varphi)\\ \sin(\alpha+\varphi)\end{pmatrix}$ || 几何：逆时针转 $\varphi$，角度从 $\alpha$ 变成 $\alpha+\varphi$，仍在单位圆上
$\cos(\alpha+\varphi)=\cos\alpha\cos\varphi-\sin\alpha\sin\varphi$ || 比较前两行的横坐标：同一个点的坐标只有一种
$\sin(\alpha+\varphi)=\sin\alpha\cos\varphi+\cos\alpha\sin\varphi$ || 比较前两行的纵坐标
:::

::: eq 和角公式 | (M3.5)
$$\cos(\alpha+\varphi)=\cos\alpha\cos\varphi-\sin\alpha\sin\varphi,\qquad \sin(\alpha+\varphi)=\sin\alpha\cos\varphi+\cos\alpha\sin\varphi$$
中学课本里要画辅助线证明的两个公式，在这里只是“旋转矩阵作用在单位向量上”的两行。[§4](#p4) 会用欧拉公式再推一遍，只要一行。
:::

::: ex 转一个具体的向量
$\varphi=30°$：$\cos30°=0.8660$，$\sin30°=0.5$。$\mathbf x=(2,1)$ 的长度 $\sqrt5\approx2.236$，角度 $\arctan(1/2)\approx26.57°$。

按 (M3.4)：

$$\mathbf R(30°)\mathbf x=\begin{pmatrix}2\times0.8660-1\times0.5\\ 2\times0.5+1\times0.8660\end{pmatrix}=\begin{pmatrix}1.2321\\ 1.8660\end{pmatrix}.$$

核对：长度 $\sqrt{1.2321^2+1.8660^2}=\sqrt{1.5181+3.4820}=\sqrt{5.0001}\approx2.236$（末位差异来自四位小数的舍入）；角度 $\arctan(1.8660/1.2321)\approx56.57°=26.57°+30°$。图 M3.2 的右图画的就是这个例子。
:::

## 转置就是反向旋转

把 (M3.4) 转置，两个非对角元交换位置：

::: deriv R(φ)ᵀ = R(−φ)
$\mathbf R(\varphi)^\top=\begin{pmatrix}\cos\varphi&\sin\varphi\\ -\sin\varphi&\cos\varphi\end{pmatrix}$ || 转置：第 $(1,2)$ 格与第 $(2,1)$ 格交换
$=\begin{pmatrix}\cos(-\varphi)&-\sin(-\varphi)\\ \sin(-\varphi)&\cos(-\varphi)\end{pmatrix}$ || $\cos$ 是偶函数、$\sin$ 是奇函数（前置框第 ② 条）
$=\mathbf R(-\varphi)$ || 对照 (M3.4)，这正是转 $-\varphi$ 的矩阵
:::

::: eq 转置 = 逆 = 反向旋转 | (M3.6)
$$\mathbf R(\varphi)^\top=\mathbf R(\varphi)^{-1}=\mathbf R(-\varphi)$$
第一个等号是 (M3.3)。撤销一次旋转，就是往回转同样的角度。这一条是 [§5](#p5) 相对角恒等式的一半。
:::

## 二维正交矩阵只有这两类

上一页说过正交矩阵只有“旋转”和“镜像”两类。二维里可以直接证明，而且证明很短。

::: deriv 二维正交矩阵的分类
$U\mathbf e_1=(\cos\varphi,\ \sin\varphi)$ || 第一列是单位向量；单位圆上的点都有一个角度
$U\mathbf e_2=\pm(-\sin\varphi,\ \cos\varphi)$ || 第二列是与第一列垂直的单位向量，只有 $\pm(-b,a)$ 两个（“旋转矩阵的两列”第 (2) 行）
$+:\ U=\mathbf R(\varphi),\ \ \det U=+1$ || 对照 (M3.4)：旋转；$\det=\cos^2\varphi+\sin^2\varphi$
$-:\ U=\begin{pmatrix}\cos\varphi&\sin\varphi\\ \sin\varphi&-\cos\varphi\end{pmatrix}$ || $\det U=-\cos^2\varphi-\sin^2\varphi=-1$：沿角度 $\varphi/2$ 那条直线的镜像（用和角公式验算：它把 $(\cos\frac\varphi2,\sin\frac\varphi2)$ 送回自己，把与之垂直的 $(-\sin\frac\varphi2,\cos\frac\varphi2)$ 变成相反方向）；$\varphi=90°$ 时就是图 M3.1 的 $P$
:::

所以二维里“行列式为 $+1$ 的正交矩阵”与“旋转”是同一件事。RoPE 在每个二维平面里用的正是这一类。

::: warn 两个常见的方向错误
**屏幕坐标。** 数学里纵轴向上，逆时针为正；屏幕和图像坐标的纵轴通常向下，同一个 $\mathbf R(\varphi)$ 在屏幕上看起来是顺时针转。画图验证时别被它骗了。

**行向量写法。** 代码里向量常按行存放（[M01 §3](m01-vectors-matrices.html#p3)），列写法的 $\mathbf y=\mathbf R\mathbf x$ 要写成 `y = x @ R.T`。如果写成 `x @ R`，实际做的是 $\mathbf R^\top\mathbf x=\mathbf R(-\varphi)\mathbf x$，转向反了。对查询和键都犯同一个错误时，打分仍然只依赖相对位置，但夹在中间的旋转从 $\mathbf R((j-t)\theta)$ 变成了 $\mathbf R((t-j)\theta)$（[§5](#p5) 的 (M3.15) 可以验证），“前面第 3 个”和“后面第 3 个”对调，和按正确方向训练出来的权重就对不上了。
:::

:::: quiz 小练习：转 60° 与转 90°
（1）求 $\mathbf R(60°)$ 作用在 $(1,1)$ 上的结果，并核对长度。（2）写出 $\mathbf R(90°)$，说明它对任意 $(a,b)$ 做了什么。（3）$\mathbf R(180°)$ 是什么矩阵？
::: answer 看答案
（1）$\cos60°=0.5$，$\sin60°\approx0.8660$。$\mathbf R(60°)(1,1)=(0.5-0.8660,\ 0.8660+0.5)=(-0.3660,\ 1.3660)$。长度 $\sqrt{0.1340+1.8660}=\sqrt{2.0000}\approx1.414=\norm{(1,1)}$。

（2）$\mathbf R(90°)=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$，$(a,b)\mapsto(-b,a)$：交换两个坐标，再给新的第一个坐标取负号。这正是上面推导里“逆时针 $90°$ 的垂直向量”。

（3）$\cos180°=-1$，$\sin180°=0$，$\mathbf R(180°)=-I$：每个向量都变成它的反方向。它的行列式是 $(-1)(-1)-0=+1$，所以虽然每个向量都“掉了头”，它仍是旋转，不是镜像。
:::
::::

::: note 这一页用在哪里
[第 5 课 §2](../deepseek/lesson-05-modern-block.html#p2)：式 (5.4) 把复数形式的 RoPE 翻译成实数矩阵，就是本页的 $\mathbf R(t\theta)$。[第 9 课 §4](../deepseek/lesson-09-v2-mla.html#p4)：推导 $\mathbf R(\alpha)^\top\mathbf R(\beta)=\mathbf R(\beta-\alpha)$ 的第一行就是 (M3.6)，后两行用的是和角公式 (M3.5)。
:::

::: key 本页要点
- 矩阵的两列是两个坐标向量的去处，由此得到 $\mathbf R(\varphi)=\begin{pmatrix}\cos\varphi&-\sin\varphi\\\sin\varphi&\cos\varphi\end{pmatrix}$ (M3.4)。它是行列式为 $+1$ 的正交矩阵。
- 把单位向量 $(\cos\alpha,\sin\alpha)$ 转 $\varphi$，比较几何与代数两种结果，得到和角公式 (M3.5)。
- $\mathbf R(\varphi)^\top=\mathbf R(\varphi)^{-1}=\mathbf R(-\varphi)$ (M3.6)。二维正交矩阵只有两类：行列式为 $+1$ 的旋转，行列式为 $-1$ 的镜像。
:::

# 复数 | 把二维向量当成一个数：从 i² = −1 出发，乘法 = 拉伸 + 旋转 | 25

旋转矩阵好用，但有两处不方便。第一，它是一张 $2\times2$ 的表，四个数里只有一个自由参数 $\varphi$。第二，“转 $\alpha$ 再转 $\beta$ 等于转 $\alpha+\beta$”这件用手转一转就能验证的几何事实，用矩阵写出来要靠和角公式才能看见。**有没有一种“数”，它本身就是一个旋转，而它的乘法就是旋转的复合？**有，这就是复数。这一页从一条规定 $\mathrm i^2=-1$ 出发，把复数讲到能直接用于 RoPE 的程度。

## 定义：多一个 i，规定 i² = −1

**含义。** 复数（complex number）是形如 $a+b\,\mathrm i$ 的数，$a$、$b$ 是实数，$\mathrm i$ 是一个新符号，只规定一件事：$\mathrm i^2=-1$。

**理解。** 没有哪个实数的平方是 $-1$，所以 $\mathrm i$ 不在实数轴上。办法是给它另开一个方向：把 $a+b\,\mathrm i$ 画成平面上的点 $(a,b)$。横轴放实数，叫**实轴**；纵轴放 $\mathrm i$ 的倍数，叫**虚轴**。于是**一个复数就是一个二维向量**，只是它还会做乘法。“虚”字是历史上的叫法，不代表它不真实；在本课里，虚部就是纵坐标。

**数学。** 记 $z=a+b\,\mathrm i$，实部 $\operatorname{Re}z=a$，虚部 $\operatorname{Im}z=b$（虚部是实数 $b$，不含 $\mathrm i$）。加法逐分量进行，和向量加法一样。乘法按分配律展开，遇到 $\mathrm i^2$ 就换成 $-1$：

::: eq 复数乘法 | (M3.7)
$$(a+b\,\mathrm i)(c+d\,\mathrm i)=ac+ad\,\mathrm i+bc\,\mathrm i+bd\,\mathrm i^2=(ac-bd)+(ad+bc)\,\mathrm i$$
这样定义的乘法满足交换律、结合律、对加法的分配律，和实数一样：它只是“把 $\mathrm i$ 当作字母做多项式乘法，再用 $\mathrm i^2=-1$ 化简”。
:::

::: ex 算两个乘积
$(1+2\mathrm i)(3-\mathrm i)=3-\mathrm i+6\mathrm i-2\mathrm i^2=3+5\mathrm i+2=5+5\mathrm i$。

$(1+2\mathrm i)(1+\mathrm i)=1+\mathrm i+2\mathrm i+2\mathrm i^2=-1+3\mathrm i$。图 M3.3 (c) 画的就是这一个。
:::

## 模与共轭

**模**（modulus）就是这个向量的长度：$\abs{z}=\sqrt{a^2+b^2}$。**共轭**（conjugate）$\bar z=a-b\,\mathrm i$ 是关于实轴的镜像。两者由一个等式连在一起：

::: eq 模与共轭 | (M3.8)
$$z\,\bar z=\abs{z}^2,\qquad \overline{zw}=\bar z\,\bar w,\qquad \abs{zw}=\abs{z}\,\abs{w}$$
第一式：$(a+b\,\mathrm i)(a-b\,\mathrm i)=a^2-b^2\mathrm i^2=a^2+b^2$。第二式：把 (M3.7) 两边都取共轭，逐项核对（虚部整体变号）。第三式：$\abs{zw}^2=zw\,\overline{zw}=z\bar z\,w\bar w=\abs{z}^2\abs{w}^2$，用了交换律。
:::

第三式说：**乘积的长度等于长度的乘积**。验算上面的例子：$\abs{1+2\mathrm i}=\sqrt5$，$\abs{3-\mathrm i}=\sqrt{10}$，$\abs{5+5\mathrm i}=\sqrt{50}=\sqrt5\cdot\sqrt{10}\approx7.071$。长度相乘，那方向呢？下面两节回答。

## 乘以 i：逆时针转 90°

先看最简单的乘法：$\mathrm i\,(a+b\,\mathrm i)=a\,\mathrm i+b\,\mathrm i^2=-b+a\,\mathrm i$。换成坐标就是 $(a,b)\mapsto(-b,a)$，正是 [§2](#p2) 小练习里的 $\mathbf R(90°)$。所以**乘以 $\mathrm i$ 就是逆时针转 $90°$**。

这也给了那条古怪的规定一个几何意义：$\mathrm i^2=-1$ 说的是“连转两次 $90°$ 等于转 $180°$，也就是掉头”。实数乘以 $-1$ 是在数轴上掉头；复数把“掉头”拆成了两个四分之一圈。

## 乘以一个复数 = 拉伸 + 旋转

**问题。** 固定一个复数 $w=c+d\,\mathrm i$，映射 $z\mapsto wz$ 对平面做了什么？

::: deriv 乘以 w 是哪个矩阵
$wz=(c+d\,\mathrm i)(a+b\,\mathrm i)=(ca-db)+(da+cb)\,\mathrm i$ || (M3.7)，$z=a+b\,\mathrm i$
$\begin{pmatrix}a\\b\end{pmatrix}\mapsto\begin{pmatrix}ca-db\\da+cb\end{pmatrix}=\begin{pmatrix}c&-d\\d&c\end{pmatrix}\begin{pmatrix}a\\b\end{pmatrix}$ || 按坐标写出来，它是一个线性变换，矩阵记作 $M_w$
$c=r\cos\varphi,\quad d=r\sin\varphi,\qquad r=\abs{w}$ || 把 $w$ 写成极坐标：长度 $r$，与实轴正方向的夹角 $\varphi$（叫辐角，argument，记作 $\arg w$）
$M_w=r\begin{pmatrix}\cos\varphi&-\sin\varphi\\ \sin\varphi&\cos\varphi\end{pmatrix}=r\,\mathbf R(\varphi)$ || 提出公因子 $r$，剩下的正是 (M3.4)
:::

::: eq 乘以 w：按模拉伸，按辐角旋转 | (M3.9)
$$w=c+d\,\mathrm i\quad\longleftrightarrow\quad M_w=\begin{pmatrix}c&-d\\ d&c\end{pmatrix}=\abs{w}\,\mathbf R(\arg w)$$
特别地，$\abs{w}=1$ 时，$w=\cos\varphi+\mathrm i\sin\varphi$ 的乘法**就是**旋转 $\mathbf R(\varphi)$。反过来，形如 $\begin{pmatrix}c&-d\\d&c\end{pmatrix}$ 的矩阵都是“某个复数的乘法”，[§5](#p5) 会用到这一点。
:::

::: fig figures/m03-complex-mult.svg | 图 M3.3 复数是平面上的点；乘以一个复数，就是把长度乘上它的模、把角度加上它的辐角 | wide
**看 (c) 的红色弧线：$w=1+\mathrm i$ 的辐角是 $45°$，于是 $z$ 的方向从 $63.43°$ 转到 $108.43°$；同时长度从 $\sqrt5$ 变成 $\sqrt2\times\sqrt5=\sqrt{10}$。** (a) 是复数的“向量读法”：实部、虚部就是横、纵坐标，共轭是关于实轴的镜像。(b) 是最简单的一次旋转：乘以 $\mathrm i$ 逆时针转 $90°$，再乘一次就转到 $-z$，这就是 $\mathrm i^2=-1$ 的几何意义。
:::

::: ex 用复数重做图 M3.1 里的旋转
[§1](#p1) 的正交矩阵 $U=\tfrac15\begin{pmatrix}3&-4\\4&3\end{pmatrix}$ 恰好是 $\begin{pmatrix}c&-d\\d&c\end{pmatrix}$ 的样子，$c=0.6$，$d=0.8$。按 (M3.9)，它就是乘以复数 $w=0.6+0.8\mathrm i$；$\abs{w}=1$，辐角 $\arctan(0.8/0.6)=53.13°$，所以是纯旋转。

验算：图 M3.1 里的灰箭头 $\mathbf x=(2,1)$ 对应 $z=2+\mathrm i$。复数法：$wz=(0.6+0.8\mathrm i)(2+\mathrm i)=1.2+0.6\mathrm i+1.6\mathrm i+0.8\mathrm i^2=0.4+2.2\mathrm i$。矩阵法：[§1](#p1) 的例题算过 $U\mathbf x=\tfrac15(6-4,\ 8+3)=(0.4,\ 2.2)$，也就是图 M3.1 左图标着 $U\mathbf x$ 的那支箭头。两者一致，连做的 4 次实数乘法都一样；复数法省下的是记账：一个复数只要记 2 个数 $(c,d)$，矩阵要写 4 格，而这 4 格里只有 2 个独立的数。
:::

## 点积是 z 与 w 的共轭之积的实部

第 5 课推导 RoPE 时有一步“点积是 $z\bar w$ 的实部”。现在可以把它推出来：

::: deriv 二维点积的复数写法
$z=a+b\,\mathrm i\leftrightarrow(a,b),\qquad w=c+d\,\mathrm i\leftrightarrow(c,d)$ || 两个二维向量各看成一个复数
$z\,\bar w=(a+b\,\mathrm i)(c-d\,\mathrm i)=(ac+bd)+(bc-ad)\,\mathrm i$ || (M3.7)，把 $d$ 换成 $-d$
$\operatorname{Re}(z\bar w)=ac+bd=(a,b)\cdot(c,d)$ || 实部正是点积
$\operatorname{Im}(z\bar w)=bc-ad=-\det\begin{pmatrix}a&c\\b&d\end{pmatrix}$ || 虚部是两个向量围成的有向面积，带一个负号（见下面的定义框）
:::

::: def 有向面积
**含义。** 以两个向量 $\mathbf z=(a,b)$、$\mathbf w=(c,d)$ 为邻边的平行四边形的面积，再带上一个正负号：从 $\mathbf z$ 逆时针转不到 $180°$ 就能到达 $\mathbf w$ 时取正，否则取负。

**理解。** 它量的是两个向量“有多不平行”：平行时为 0，垂直时绝对值最大，等于两个长度之积。正负号只记录“谁在谁的逆时针一侧”，交换两个向量，符号就反过来。点积量的是“有多同向”，有向面积量的是“有多侧向”，两者合起来才完整描述两个向量的相对位置。

**数学。** 把两个向量并排作列，行列式就是它：$\det\begin{pmatrix}a&c\\b&d\end{pmatrix}=ad-bc$（[§1](#p1) 说过，行列式是带符号的面积放大倍数；这个矩阵把边长为 1 的正方形送成以 $\mathbf z$、$\mathbf w$ 为邻边的平行四边形）。写成极坐标 $\mathbf z=r_1(\cos\varphi_1,\sin\varphi_1)$、$\mathbf w=r_2(\cos\varphi_2,\sin\varphi_2)$，再用和角公式 (M3.5)，得 $ad-bc=r_1r_2\sin(\varphi_2-\varphi_1)$，所以 $\operatorname{Im}(z\bar w)=r_1r_2\sin(\varphi_1-\varphi_2)$，与实部的 $r_1r_2\cos(\varphi_1-\varphi_2)$ 正好配成一对。
:::

::: eq 点积与有向面积都藏在 z 乘 w 的共轭里 | (M3.10)
$$\operatorname{Re}(z\bar w)=\mathbf z\cdot\mathbf w,\qquad \abs{z\bar w}=\abs{z}\abs{w}$$
$\mathbf z$、$\mathbf w$ 是 $z$、$w$ 对应的二维向量。第二式来自 (M3.8)。下一页有了欧拉公式，会看到 $z\bar w$ 的辐角是两个向量的夹角，于是 (M3.10) 就是 M01 的 (M1.3)：$\mathbf z\cdot\mathbf w=\norm{\mathbf z}\norm{\mathbf w}\cos\angle(\mathbf z,\mathbf w)$。
:::

::: ex 一个查询、一个键
取 $\mathbf q=(1,2)$，$\mathbf k=(0.5,-1)$（[§5](#p5) 与 [§7](#p7) 还会用它们）。直接算点积：$1\times0.5+2\times(-1)=-1.5$。

复数法：$q=1+2\mathrm i$，$k=0.5-\mathrm i$，$\bar k=0.5+\mathrm i$。$q\bar k=(1+2\mathrm i)(0.5+\mathrm i)=0.5+\mathrm i+\mathrm i+2\mathrm i^2=-1.5+2\mathrm i$。实部 $-1.5$，就是点积；虚部 $2=bc-ad=2\times0.5-1\times(-1)$。

几何核对：$\mathbf q$ 的方向是 $63.43°$，$\mathbf k$ 的方向是 $-63.43°$，从 $\mathbf q$ 顺时针转 $126.87°$ 才到 $\mathbf k$。平行四边形面积 $\norm{\mathbf q}\norm{\mathbf k}\sin126.87°=\sqrt5\times\sqrt{1.25}\times0.8=2$；$\mathbf k$ 在 $\mathbf q$ 的顺时针一侧，所以有向面积 $ad-bc=1\times(-1)-2\times0.5=-2$，虚部是它的相反数 $+2$。点积也对得上：$2.5\times\cos126.87°=2.5\times(-0.6)=-1.5$。
:::

::: ana 类比：复数是“一个会做乘法的箭头”
**映射。** 复数 $a+b\,\mathrm i$ = 平面上的一支箭头 $(a,b)$；加法 = 箭头首尾相接；乘以 $w$ = 把整个平面按 $\abs{w}$ 缩放、再转 $\arg w$；共轭 = 照镜子（关于实轴）。

**收益。** 一个数同时扮演两个角色：它既是被旋转的对象（向量），又是旋转这个动作本身（乘数）。RoPE 正是利用这一点：位置 $t$ 对应一个模为 1 的复数，把它乘到查询上就完成了旋转。

**失效的地方。** 复数乘法只能表达“均匀缩放 + 旋转”，即 (M3.9) 那种 $\begin{pmatrix}c&-d\\d&c\end{pmatrix}$ 形状的矩阵。剪切、沿一个方向拉伸、镜像，都**不是**任何复数的乘法（镜像 $z\mapsto\bar z$ 甚至不是乘法）。一般的权重矩阵更不是。[§5](#p5) 会看到，这正是 RoPE 与矩阵吸收冲突的根源。复数也没有大小顺序：“$\mathrm i>0$”这样的比较没有意义。
:::

::: err 算 RoPE 打分时忘了共轭
点积是 $\operatorname{Re}(q\bar k)$，不是 $\operatorname{Re}(qk)$。上例里 $qk=(1+2\mathrm i)(0.5-\mathrm i)=0.5-\mathrm i+\mathrm i-2\mathrm i^2=2.5$，实部 $2.5\ne-1.5$。$\operatorname{Re}(qk)=ac-bd$ 是把 $\mathbf k$ 先关于实轴镜像再做点积的结果，几何上没有意义。第 5 课式 (5.3) 里键那一侧总带着一道横线，就是这个原因。
:::

:::: quiz 小练习：复数的四则运算
$u=2+\mathrm i$，$v=1-3\mathrm i$。求（1）$uv$ 与 $\abs{uv}$，并验证 $\abs{uv}=\abs u\abs v$；（2）$u\bar v$，从中读出向量 $(2,1)$ 与 $(1,-3)$ 的点积；（3）$1/u$（提示：分子分母同乘 $\bar u$）。
::: answer 看答案
（1）$uv=2-6\mathrm i+\mathrm i-3\mathrm i^2=5-5\mathrm i$，$\abs{uv}=\sqrt{50}\approx7.071$；$\abs u=\sqrt5$，$\abs v=\sqrt{10}$，乘积 $\sqrt{50}$，一致。

（2）$\bar v=1+3\mathrm i$，$u\bar v=2+6\mathrm i+\mathrm i+3\mathrm i^2=-1+7\mathrm i$。实部 $-1$ 就是点积 $2\times1+1\times(-3)=-1$。

（3）$\dfrac1u=\dfrac{\bar u}{u\bar u}=\dfrac{2-\mathrm i}{\abs u^2}=\dfrac{2-\mathrm i}{5}=0.4-0.2\mathrm i$。核对：$(2+\mathrm i)(0.4-0.2\mathrm i)=0.8-0.4\mathrm i+0.4\mathrm i-0.2\mathrm i^2=1$。模为 1 的复数，倒数就是共轭：“撤销一次旋转”就是“往回转”。
:::
::::

::: note 这一页用在哪里
[第 5 课 §2](../deepseek/lesson-05-modern-block.html#p2)：前置框“复数、欧拉公式与点积 = 实部”与式 (5.3) 的 $f_q\overline{f_k}$，用的是本页的 (M3.7)、(M3.8)、(M3.10)。[第 5 课 §3](../deepseek/lesson-05-modern-block.html#p3)：式 (5.7) 把 $d_h$ 维向量看成 $d_h/2$ 个复数 $\tilde q_i=q_{2i}+\mathrm iq_{2i+1}$，点积写成 $\operatorname{Re}\sum_i\tilde q_i\overline{\tilde k_i}\cdots$。
:::

::: key 本页要点
- 复数 $a+b\,\mathrm i$ 就是二维向量 $(a,b)$，外加一条规定 $\mathrm i^2=-1$ 带来的乘法 (M3.7)。模是长度，共轭是镜像，$z\bar z=\abs z^2$，$\abs{zw}=\abs z\abs w$ (M3.8)。
- 乘以 $\mathrm i$ 是转 $90°$；乘以 $w$ 是矩阵 $\begin{pmatrix}c&-d\\d&c\end{pmatrix}=\abs w\,\mathbf R(\arg w)$ (M3.9)。模为 1 的复数的乘法就是旋转。
- 二维点积 $=\operatorname{Re}(z\bar w)$ (M3.10)；别忘了共轭。复数乘法表达不了剪切、非均匀拉伸或一般的权重矩阵。
:::

# 欧拉公式 | 为什么指数函数会转圈，以及它怎样把旋转变成指数相加 | 25

上一页得到：模为 1 的复数 $\cos\varphi+\mathrm i\sin\varphi$ 的乘法就是旋转 $\mathbf R(\varphi)$。把两个这样的数相乘，用和角公式 (M3.5) 整理：

$$(\cos\alpha+\mathrm i\sin\alpha)(\cos\beta+\mathrm i\sin\beta)=\cos(\alpha+\beta)+\mathrm i\sin(\alpha+\beta).$$

**相乘，角度相加。**这和指数函数的性质 $e^ae^b=e^{a+b}$ 一模一样。这提示我们：$\cos\varphi+\mathrm i\sin\varphi$ 也许就是某个“与 $\varphi$ 成正比的东西”的指数。欧拉公式说，它就是 $e^{\mathrm i\varphi}$。可是“$e$ 的虚数次方”是什么意思？中学里 $e^x$ 只对实数 $x$ 有定义。这一页先把 $e^x$ 的定义推广到复数，再推出欧拉公式，最后看它怎样把 RoPE 的全部计算变成指数的加减。

::: pre 指数函数与泰勒级数 | 知道 $e^x=\sum_n x^n/n!$ 的可以跳过
$e^x$ 可以写成一个无穷级数（泰勒级数）：

$$e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots,\qquad n!=1\times2\times\cdots\times n,\ 0!=1.$$

例如 $e^1=1+1+0.5+0.1667+0.0417+0.0083+\cdots=2.71828\ldots$。余弦和正弦也有类似的级数：

$$\cos x=1-\frac{x^2}{2!}+\frac{x^4}{4!}-\cdots,\qquad \sin x=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\cdots$$

这三个级数对任何 $x$ 都收敛：第 $n$ 项的分母 $n!$ 比分子 $x^n$ 长得快得多。级数只用到加法和乘法，所以把复数代进去也完全有意义，而且照样收敛：复数级数第 $n$ 项的模 $\abs{z}^n/n!$ 正是实数级数在 $x=\abs z$ 处的第 $n$ 项，所以各项的模加起来有限（这叫**绝对收敛**，下面重新分组、重排各项时要用到它）。**这就是 $e^z$ 对复数 $z$ 的定义**。
:::

## 推导一：把 iφ 代进级数

::: deriv 欧拉公式
$e^{\mathrm i\varphi}=\sum_{n=0}^{\infty}\frac{(\mathrm i\varphi)^n}{n!}=1+\mathrm i\varphi+\frac{(\mathrm i\varphi)^2}{2!}+\frac{(\mathrm i\varphi)^3}{3!}+\frac{(\mathrm i\varphi)^4}{4!}+\cdots$ || 按定义，把 $x=\mathrm i\varphi$ 代进 $e^x$ 的级数
$\mathrm i^0=1,\ \mathrm i^1=\mathrm i,\ \mathrm i^2=-1,\ \mathrm i^3=-\mathrm i,\ \mathrm i^4=1,\ \dots$ || $\mathrm i$ 的幂每四次一循环：每乘一次 $\mathrm i$ 转 $90°$，转四次回到原处
$=1+\mathrm i\varphi-\frac{\varphi^2}{2!}-\mathrm i\frac{\varphi^3}{3!}+\frac{\varphi^4}{4!}+\mathrm i\frac{\varphi^5}{5!}-\cdots$ || 逐项代入 $\mathrm i$ 的幂
$=\Big(1-\frac{\varphi^2}{2!}+\frac{\varphi^4}{4!}-\cdots\Big)+\mathrm i\Big(\varphi-\frac{\varphi^3}{3!}+\frac{\varphi^5}{5!}-\cdots\Big)$ || 偶数次项全是实数，奇数次项全带 $\mathrm i$，分成两组（绝对收敛的级数可以重新分组）
$=\cos\varphi+\mathrm i\sin\varphi$ || 两组恰好是 $\cos\varphi$ 与 $\sin\varphi$ 的级数（前置框）
:::

::: eq 欧拉公式 | (M3.11)
$$e^{\mathrm i\varphi}=\cos\varphi+\mathrm i\sin\varphi,\qquad \abs{e^{\mathrm i\varphi}}=1,\qquad \overline{e^{\mathrm i\varphi}}=e^{-\mathrm i\varphi}=\frac{1}{e^{\mathrm i\varphi}}$$
$\varphi$ 是实数（弧度）。$e^{\mathrm i\varphi}$ 是单位圆上角度为 $\varphi$ 的点，也就是旋转 $\mathbf R(\varphi)$ 的复数形式。第二式：$\cos^2\varphi+\sin^2\varphi=1$；第三式：$\cos$ 偶、$\sin$ 奇，以及 (M3.8)。
:::

::: fig figures/m03-euler.svg | 图 M3.4 欧拉公式：单位圆上角度为 φ 的点，以及泰勒级数一项一项地转进这个点 | wide
**先看右图的折线：从 1 出发，第 $n$ 段都比上一段逆时针转了 $90°$、长度乘上 $2/n$（从 $n=3$ 起才开始缩短），折线绕着弯收进蓝点 $e^{2\mathrm i}$。** 这就是推导一的几何样子：级数第 $n$ 项是 $(2\mathrm i)^n/n!$，比第 $n-1$ 项多乘一个 $2\mathrm i/n$，即“转 $90°$、乘 $2/n$”。左图是结果：$e^{\mathrm i\varphi}$ 都在单位圆上；红色粗弧的长度是 1，它的终点 $e^{\mathrm i\cdot1}$ 的横、纵坐标就是 $\cos1=0.540$、$\sin1=0.841$。左图的三支绿箭头是下面“推导二”的速度 $\mathrm i\,e^{\mathrm i\varphi}$：每一支都是把所在位置转 $90°$ 得到的，所以都贴着圆的切线方向。所有点由生成脚本按级数与欧拉公式算出。
:::

::: ex 几个值，以及级数收敛得有多快
由 (M3.11)：$e^{\mathrm i\pi/6}=\cos30°+\mathrm i\sin30°=0.8660+0.5\mathrm i$；$e^{\mathrm i\pi/2}=\mathrm i$；$e^{\mathrm i\pi}=-1$；$e^{\mathrm i\cdot1}=0.5403+0.8415\mathrm i$；$e^{2\mathrm i}=-0.4161+0.9093\mathrm i$。

逐项累加 $\varphi=2$ 的级数（图 M3.4 右）：$1$；$1+2\mathrm i$；$-1+2\mathrm i$；$-1+0.6667\mathrm i$；$-0.3333+0.6667\mathrm i$；$-0.3333+0.9333\mathrm i$；……前 8 项（$n=0$ 到 $7$）之和 $-0.4222+0.9079\mathrm i$，离真值差 $0.0062$；前 12 项之和只差 $8.5\times10^{-6}$。$\varphi$ 越大，需要的项数越多，但总能收敛。
:::

## 推导二：一个永远垂直于半径的运动

第二种推导不再把级数分组，而是问：**$f(\varphi)=e^{\mathrm i\varphi}$ 随 $\varphi$ 怎样运动？**把 $\varphi$ 想成时间，先求它的速度 $f'(\varphi)$。实数里 $\frac{d}{dx}e^{cx}=ce^{cx}$；复数的级数逐项求导（幂级数在收敛范围内可以逐项求导）给出同样的结果：

$$f'(\varphi)=\sum_{n=1}^{\infty}\frac{\mathrm i^n\,n\,\varphi^{n-1}}{n!}=\mathrm i\sum_{n=1}^{\infty}\frac{(\mathrm i\varphi)^{n-1}}{(n-1)!}=\mathrm i\,f(\varphi).$$

第一步是 $\varphi^n$ 的导数 $n\varphi^{n-1}$（$n=0$ 的常数项导数为 0）；第二步用 $n/n!=1/(n-1)!$ 并提出一个 $\mathrm i$；第三步把求和下标 $n-1$ 改记成 $m$，正是 $f$ 自己的级数。读出 $f'=\mathrm if$ 的几何意义：

- **速度 = 位置转 $90°$。** 乘以 $\mathrm i$ 就是逆时针转 $90°$，所以任何时刻的速度都垂直于从原点指向当前位置的半径（图 M3.4 左图的绿箭头）。一个永远垂直于半径运动的点，离原点的距离不变：它在画圆。用算式说：距离的平方 $\abs f^2=f\bar f$ 的变化率是 $f'\bar f+f\overline{f'}=\mathrm i\abs f^2-\mathrm i\abs f^2=0$。
- **起点与速率。** $f(0)=e^0=1$，从 $(1,0)$ 出发，所以这个圆就是单位圆；速率 $\abs{f'}=\abs{\mathrm i}\abs{f}=1$，每单位时间走过长度 1 的弧。
- **结论。** 经过时间 $\varphi$，点沿单位圆逆时针走过弧长 $\varphi$，也就是角度 $\varphi$ 弧度，到达 $(\cos\varphi,\sin\varphi)$。这就是 (M3.11)。

这个推导也解释了为什么数学里的角度要用弧度：只有用弧度，“走过的弧长”和“转过的角度”才是同一个数，$e^{\mathrm i\varphi}$ 的转速才恰好是 1。

::: note 推导二的严格版本
上面是直观说法。严格一点：令 $g(\varphi)=(\cos\varphi+\mathrm i\sin\varphi)\,e^{-\mathrm i\varphi}$。按乘积法则，$g'=(-\sin\varphi+\mathrm i\cos\varphi)e^{-\mathrm i\varphi}+(\cos\varphi+\mathrm i\sin\varphi)(-\mathrm i)e^{-\mathrm i\varphi}=\big[-\sin\varphi+\mathrm i\cos\varphi-\mathrm i\cos\varphi+\sin\varphi\big]e^{-\mathrm i\varphi}=0$，而 $g(0)=1$，所以 $g\equiv1$，即 $\cos\varphi+\mathrm i\sin\varphi=e^{\mathrm i\varphi}$（最后一步用了 $e^{\mathrm i\varphi}e^{-\mathrm i\varphi}=e^0=1$，见下一节的指数律）。
:::

## 极坐标形式：乘法变成加法

任何复数 $z$ 都有长度 $r=\abs z$ 和辐角 $\varphi$，于是 $z=r(\cos\varphi+\mathrm i\sin\varphi)=re^{\mathrm i\varphi}$。在这种写法下，乘法变得非常简单，前提是指数律 $e^{a}e^{b}=e^{a+b}$ 对复数也成立：

::: deriv 指数律对复数成立
$e^ae^b=\sum_{m}\frac{a^m}{m!}\sum_{k}\frac{b^k}{k!}=\sum_{n=0}^\infty\sum_{m=0}^{n}\frac{a^mb^{n-m}}{m!\,(n-m)!}$ || 两个级数相乘，按总次数 $n=m+k$ 分组（绝对收敛，可以重排）
$=\sum_{n=0}^\infty\frac1{n!}\sum_{m=0}^{n}\binom nm a^mb^{n-m}$ || 二项式系数 $\binom nm=n!/\big(m!\,(n-m)!\big)$，所以每项分母里的 $m!\,(n-m)!$ 换成 $n!\big/\binom nm$
$=\sum_{n=0}^\infty\frac{(a+b)^n}{n!}=e^{a+b}$ || 二项式定理；它的证明只用到乘法的交换律、结合律与分配律，复数都满足，所以照样成立
:::

::: eq 极坐标形式与乘法 | (M3.12)
$$z=re^{\mathrm i\varphi},\qquad r_1e^{\mathrm i\varphi_1}\cdot r_2e^{\mathrm i\varphi_2}=r_1r_2\,e^{\mathrm i(\varphi_1+\varphi_2)},\qquad \big(e^{\mathrm i\theta}\big)^t=e^{\mathrm it\theta}$$
**模相乘，辐角相加。**第三式（$t$ 为正整数）是第二式反复使用 $t$ 次（棣莫弗公式）：转 $t$ 次、每次 $\theta$，等于一次转 $t\theta$。
:::

把指数律用在 $e^{\mathrm i(\alpha+\beta)}=e^{\mathrm i\alpha}e^{\mathrm i\beta}$ 上，展开右边再比较实部与虚部，就得到和角公式 (M3.5)，只要一行：$(\cos\alpha+\mathrm i\sin\alpha)(\cos\beta+\mathrm i\sin\beta)=(\cos\alpha\cos\beta-\sin\alpha\sin\beta)+\mathrm i(\sin\alpha\cos\beta+\cos\alpha\sin\beta)$。这一页开头是用和角公式“猜”指数律，现在反过来，用指数律推出和角公式：两条路都通，互相印证。

极坐标也兑现了 [§3](#p3) 的一个承诺。设 $z=r_1e^{\mathrm i\varphi_1}$、$w=r_2e^{\mathrm i\varphi_2}$，则 $\bar w=r_2e^{-\mathrm i\varphi_2}$，$z\bar w=r_1r_2\,e^{\mathrm i(\varphi_1-\varphi_2)}$：它的辐角正是两个向量的夹角，实部是 $r_1r_2\cos(\varphi_1-\varphi_2)$。把这句话和 (M3.10) 放在一起，就是 M01 的 (M1.3)：点积 $=$ 长度 $\times$ 长度 $\times$ 夹角的余弦。

(M3.12) 的第三式正是 RoPE 的出发点：第 5 课里位置 $t$ 的查询乘以 $e^{\mathrm it\theta}$。它等于“每前进一个 token 就乘一次 $e^{\mathrm i\theta}$、共乘 $t$ 次”，但实现时直接算 $\cos(t\theta)$ 与 $\sin(t\theta)$ 就好（[§1](#p1) 的失败模式说明了为什么不要真的连乘）。

::: ex 用极坐标做乘法
$z=1+2\mathrm i$ 的模是 $\sqrt5$，辐角 $\arctan2=63.43°$（$1.1071$ rad）；$w=1+\mathrm i$ 的模是 $\sqrt2$，辐角 $45°$（$0.7854$ rad）。指数里的角度本应写弧度，下面为了好读写成度数。按 (M3.12)：

$$\begin{aligned}wz&=\sqrt2\sqrt5\;e^{\mathrm i(63.43°+45°)}=\sqrt{10}\,(\cos108.43°+\mathrm i\sin108.43°)\\&\approx3.1623\times(-0.3162+0.9487\mathrm i)\approx-1.000+3.000\mathrm i.\end{aligned}$$

与 [§3](#p3) 用分配律算出的 $-1+3\mathrm i$ 一致；按弧度算，辐角是 $1.1071+0.7854=1.8925$ rad。

$(e^{\mathrm i\pi/3})^3=e^{\mathrm i\pi}=-1$：转 3 次 $60°$ 等于转 $180°$。直接用代数验算：$e^{\mathrm i\pi/3}=0.5+0.8660\mathrm i$，平方得 $0.25-0.75+0.8660\mathrm i=-0.5+0.8660\mathrm i$，再乘一次得 $-0.25-0.75+(-0.4330+0.4330)\mathrm i=-1$。
:::

::: warn 失败模式：角度只在“转几圈之后的余数”意义下确定
$e^{\mathrm i(\varphi+2\pi)}=e^{\mathrm i\varphi}$：多转一整圈，点回到原处。所以从 $e^{\mathrm i\varphi}$ 只能读出 $\varphi$ 除以 $2\pi$ 的余数，读不出转了几圈。对 RoPE 来说，这意味着单独一个频率 $\theta$ 分不清相对距离 $\Delta$ 与 $\Delta+2\pi/\theta$：取 $\theta=0.5$，$2\pi/0.5\approx12.57$，于是距离 13 的打分因子 $\cos(13\times0.5)=0.9766$ 几乎和距离 0 的 1 一样。[§6](#p6) 的多频率设计就是为了解决它。
:::

:::: quiz 小练习：欧拉公式的三个小算
（1）求 $e^{\mathrm i\pi/4}$ 与 $e^{-\mathrm i\pi/2}$。（2）用 $\varphi=\pi/2$ 的级数前 5 项（$n=0$ 到 $4$）近似 $e^{\mathrm i\pi/2}$，与真值 $\mathrm i$ 比较。（3）用 (M3.12) 说明：$\abs{z}=1$ 的复数的倒数等于它的共轭。
::: answer 看答案
（1）$e^{\mathrm i\pi/4}=\cos45°+\mathrm i\sin45°=0.7071+0.7071\mathrm i$；$e^{-\mathrm i\pi/2}=\cos(-90°)+\mathrm i\sin(-90°)=-\mathrm i$（顺时针转 $90°$）。

（2）$\pi/2\approx1.5708$。各项：$1$；$1.5708\mathrm i$；$(1.5708\mathrm i)^2/2=-1.2337$；$(1.5708\mathrm i)^3/6=-0.6460\mathrm i$；$(1.5708\mathrm i)^4/24=0.2537$。和为 $(1-1.2337+0.2537)+(1.5708-0.6460)\mathrm i=0.0200+0.9248\mathrm i$，离 $\mathrm i$ 还差约 $0.078$；再加两项（$0.0797\mathrm i$ 与 $-0.0209$）就只差约 $0.005$。

（3）$z=e^{\mathrm i\varphi}$，$\bar z=e^{-\mathrm i\varphi}$，$z\bar z=e^{\mathrm i\varphi-\mathrm i\varphi}=e^0=1$，所以 $\bar z=1/z$。几何上：“往回转”撤销“往前转”。
:::
::::

::: note 这一页用在哪里
[第 5 课 §2](../deepseek/lesson-05-modern-block.html#p2)：RoPE 的二维解 $f_q(\mathbf q,t)=\mathbf q\,e^{\mathrm it\theta}$，以及推导里把 $f$ 写成“模 × $e^{\mathrm i\cdot\text{辐角}}$”的极坐标形式。[第 5 课 §3](../deepseek/lesson-05-modern-block.html#p3)：打分公式 (5.7) 里的相位因子 $e^{\mathrm i(t-j)\theta_i}$，以及“远程衰减”推导里的部分和 $S_m=\sum_i e^{\mathrm i\Delta\theta_i}$。
:::

::: key 本页要点
- 用级数定义复数的指数，代入 $\mathrm i\varphi$，偶数项与奇数项分开，得到欧拉公式 $e^{\mathrm i\varphi}=\cos\varphi+\mathrm i\sin\varphi$ (M3.11)。另一种看法：$f'=\mathrm if$ 表示速度永远垂直于半径、速率为 1，所以 $f$ 以单位速率沿单位圆转。
- 指数律对复数成立，所以 $re^{\mathrm i\varphi}$ 的乘法是“模相乘、辐角相加”，$(e^{\mathrm i\theta})^t=e^{\mathrm it\theta}$ (M3.12)；和角公式只是它的一行推论。
- $e^{\mathrm i\varphi}$ 以 $2\pi$ 为周期：一个频率只能确定角度的余数，这就是单频率位置编码会“绕圈”的原因。
:::

# 旋转的复合与相对角 | 角度相加、相对角恒等式，以及旋转能和谁交换次序 | 25

前四页的工具在这一页汇合。RoPE 之所以能“只看相对位置”，靠的是一条恒等式：$\mathbf R(\alpha)^\top\mathbf R(\beta)=\mathbf R(\beta-\alpha)$。这一页先证明旋转的复合是角度相加，再由它推出这条恒等式和“旋转之后的点积只剩相对角”；最后回答第 9 课要用的问题：**一个旋转矩阵能和哪些矩阵交换次序？**答案很窄，这正是 RoPE 与 MLA 的矩阵吸收冲突的数学根源。

## 复合：角度相加

**问题。** 先转 $\beta$、再转 $\alpha$，总共转了多少？几何上答案是 $\alpha+\beta$；下面分别用矩阵和复数验证，两种算法各有用处。

::: deriv 两次旋转的乘积（矩阵算法）
$[\mathbf R(\alpha)\mathbf R(\beta)]_{11}=\cos\alpha\cos\beta-\sin\alpha\sin\beta=\cos(\alpha+\beta)$ || 第 1 行乘第 1 列，再用和角公式 (M3.5)
$[\mathbf R(\alpha)\mathbf R(\beta)]_{21}=\sin\alpha\cos\beta+\cos\alpha\sin\beta=\sin(\alpha+\beta)$ || 第 2 行乘第 1 列，同上
$[\mathbf R(\alpha)\mathbf R(\beta)]_{12}=-\cos\alpha\sin\beta-\sin\alpha\cos\beta=-\sin(\alpha+\beta)$ || 第 1 行乘第 2 列
$[\mathbf R(\alpha)\mathbf R(\beta)]_{22}=-\sin\alpha\sin\beta+\cos\alpha\cos\beta=\cos(\alpha+\beta)$ || 第 2 行乘第 2 列
$\mathbf R(\alpha)\mathbf R(\beta)=\mathbf R(\alpha+\beta)$ || 四格合起来，对照 (M3.4)
:::

复数算法只要一行：$e^{\mathrm i\alpha}e^{\mathrm i\beta}=e^{\mathrm i(\alpha+\beta)}$（指数律 (M3.12)）。两种算法说的是同一件事，因为 (M3.9) 把“乘以 $e^{\mathrm i\alpha}$”和“乘以矩阵 $\mathbf R(\alpha)$”对应起来了。

::: eq 旋转的复合 | (M3.13)
$$\mathbf R(\alpha)\mathbf R(\beta)=\mathbf R(\alpha+\beta)=\mathbf R(\beta)\mathbf R(\alpha),\qquad \mathbf R(\theta)^t=\mathbf R(t\theta)$$
第二个等号来自 $\alpha+\beta=\beta+\alpha$：**二维旋转两两可以交换次序**。第三式是把第一式用 $t$ 次（对 $t$ 归纳）。
:::

可交换是二维的特殊之处：所有二维旋转都在同一个平面里转，只差一个角度。三维里绕不同轴的旋转一般不能交换。拿一本书试一试：先绕竖直轴转 $90°$、再绕指向你的水平轴转 $90°$，和反过来做，书最后的朝向不同。用矩阵写，$\mathbf R_z(90°)$ 把 $\mathbf e_1$ 送到 $\mathbf e_2$、$\mathbf R_x(90°)$ 把 $\mathbf e_2$ 送到 $\mathbf e_3$：$\mathbf R_x\mathbf R_z\mathbf e_1=\mathbf e_3$，而 $\mathbf R_z\mathbf R_x\mathbf e_1=\mathbf R_z\mathbf e_1=\mathbf e_2$。[§6](#p6) 会看到，RoPE 的高维旋转之所以仍然可以交换，是因为它把空间切成若干个固定的平面，每个平面里都只是二维旋转。

::: fig figures/m03-composition.svg | 图 M3.5 旋转的复合是角度相加；两个向量各转各的，夹角只取决于转角之差 | wide
**先看左图：蓝箭头 $e^{\mathrm i30°}$ 再转 $45°$（橙色弧），落在绿箭头 $e^{\mathrm i75°}$ 上。** 虚线紫箭头是乘数 $e^{\mathrm i45°}$ 本身。右图是下一节的相对角：查询 $\mathbf q=(1,2)$ 与键 $\mathbf k=(0.5,-1)$ 按位置分别转 $t\theta$ 和 $j\theta$（$\theta=0.1$）。$(t,j)=(5,2)$ 与 $(13,10)$ 两种情形下，两支箭头各自转过的角度（红弧）完全不同，但转完之后的夹角（绿弧）都是 $144.06°$，点积都是 $-2.024045$。所有坐标由生成脚本算出。
:::

::: ex 用复数和矩阵各验一次 30° + 45°
复数：用精确值 $\cos30°=\tfrac{\sqrt3}2$、$\sin30°=\tfrac12$、$\cos45°=\sin45°=\tfrac{\sqrt2}2$，

$$\begin{aligned}\Big(\tfrac{\sqrt3}2+\tfrac12\mathrm i\Big)\Big(\tfrac{\sqrt2}2+\tfrac{\sqrt2}2\mathrm i\Big)&=\tfrac{\sqrt6}4+\tfrac{\sqrt6}4\mathrm i+\tfrac{\sqrt2}4\mathrm i+\tfrac{\sqrt2}4\mathrm i^2\\&=\tfrac{\sqrt6-\sqrt2}4+\tfrac{\sqrt6+\sqrt2}4\mathrm i=0.258819+0.965926\mathrm i,\end{aligned}$$

正是 $\cos75°+\mathrm i\sin75°$。若先把 $\cos30°$、$\sin45°$ 舍入成 $0.8660$、$0.7071$ 再乘，第 4 位小数会差 1，那是舍入造成的，不是公式的误差。

矩阵：用 NumPy 算 $\mathbf R(30°)\mathbf R(45°)-\mathbf R(75°)$，四格里最大的误差约 $1.7\times10^{-16}$。这是双精度浮点数的舍入误差（1 附近相邻两个 float64 相差 $2.2\times10^{-16}$），不是公式的误差。
:::

## 相对角恒等式

现在证明本课最重要的一条等式。它只需要两块已有的积木：转置是反向旋转 (M3.6)，复合是角度相加 (M3.13)。

::: deriv R(α)ᵀR(β) = R(β − α)
$\mathbf R(\alpha)^\top\mathbf R(\beta)=\mathbf R(-\alpha)\,\mathbf R(\beta)$ || (M3.6)：转置就是反向旋转
$=\mathbf R(-\alpha+\beta)=\mathbf R(\beta-\alpha)$ || (M3.13)：复合是角度相加
$\overline{e^{\mathrm i\alpha}}\,e^{\mathrm i\beta}=e^{-\mathrm i\alpha}e^{\mathrm i\beta}=e^{\mathrm i(\beta-\alpha)}$ || 同一件事的复数写法：共轭 = 反向旋转 (M3.11)，再用指数律
:::

::: eq 相对角恒等式 | (M3.14)
$$\mathbf R(\alpha)^\top\mathbf R(\beta)=\mathbf R(\beta-\alpha),\qquad \overline{e^{\mathrm i\alpha}}\;e^{\mathrm i\beta}=e^{\mathrm i(\beta-\alpha)}$$
读法：先倒转 $\alpha$、再正转 $\beta$，净效果是转 $\beta-\alpha$。$\alpha$ 与 $\beta$ 各自是多少都不重要，只有差值留下来。
:::

[第 9 课 §4](../deepseek/lesson-09-v2-mla.html#p4) 用另一种方法证明了同一个等式：直接逐格相乘，再用和角公式。两条路殊途同归。

## 两边各转一次，点积只剩相对角

把 (M3.14) 用到点积上，就得到 RoPE 需要的性质。

::: deriv 旋转之后的点积
$\big(\mathbf R(\alpha)\mathbf q\big)^\top\big(\mathbf R(\beta)\mathbf k\big)=\mathbf q^\top\mathbf R(\alpha)^\top\mathbf R(\beta)\,\mathbf k$ || 乘积的转置 $(\mathbf R\mathbf q)^\top=\mathbf q^\top\mathbf R^\top$，再用结合律
$=\mathbf q^\top\mathbf R(\beta-\alpha)\,\mathbf k$ || (M3.14)
$=\operatorname{Re}\big[(qe^{\mathrm i\alpha})\,\overline{(ke^{\mathrm i\beta})}\big]=\operatorname{Re}\big[q\bar k\,e^{\mathrm i(\alpha-\beta)}\big]$ || 复数写法：点积是 $\operatorname{Re}(z\bar w)$ (M3.10)；乘积的共轭是共轭的乘积 (M3.8)
:::

::: eq 旋转之后的点积只依赖转角之差 | (M3.15)
$$\big\langle\mathbf R(\alpha)\mathbf q,\ \mathbf R(\beta)\mathbf k\big\rangle=\mathbf q^\top\mathbf R(\beta-\alpha)\,\mathbf k=\operatorname{Re}\big[q\,\bar k\,e^{\mathrm i(\alpha-\beta)}\big]$$
$\mathbf q,\mathbf k\in\R^2$，$q,k$ 是它们对应的复数，$\langle\cdot,\cdot\rangle$ 是点积。几何上：点积 $=\norm{\mathbf q}\norm{\mathbf k}\cos(\text{夹角})$，两边各转一次只改变夹角，改变量是 $\alpha-\beta$。
:::

令 $\alpha=t\theta$、$\beta=j\theta$，就是 RoPE：位置 $t$ 的查询转 $t\theta$，位置 $j$ 的键转 $j\theta$，打分 $=\mathbf q^\top\mathbf R\big((j-t)\theta\big)\mathbf k$，**只依赖相对距离 $t-j$**。这就是第 5 课式 (5.5)。

::: ana 类比：同一条环形跑道上、速度相同的两个人
**映射。** 平面 = 一条环形跑道；查询和键 = 两个跑步的人，各自的起跑方向由内容决定（$\mathbf q$、$\mathbf k$ 的方向）；$\theta$ = 两人共同的速度（每个 token 跑过的角度）；位置 $t$、$j$ = 各自跑了多少个 token 的时间；点积 = 只看两人之间隔着多大的角度（再乘上两人的“个头”，即长度）。

**收益。** 速度相同，两人之间的角度差就只由起跑时的角度差和“谁多跑了几步”决定，也就是 $t-j$；两人同时再多跑 100 步，差距不变。这正是 (M3.15)：两边各转一次，只剩相对角。谁在前、谁在后也看得出来：$t-j=+3$ 让夹角增加 $3\theta$，$-3$ 让它减少 $3\theta$，两者的余弦一般不同（下面例题的最后一段）。

**失效的地方。** 必须是同一条跑道、同一个速度：查询和键的速度不同（练习 2），差距就不再只由步数差决定。而且跑完之后要直接比：如果转完之后两个向量又被一个一般矩阵 $A$ 拉伸、推斜，打分变成 $\mathbf q^\top\mathbf R(t\theta)^\top A^\top A\,\mathbf R(j\theta)\mathbf k$，中间的 $A^\top A$ 一般不能与旋转交换（本页最后一节），$t$ 和 $j$ 就又各自留在式子里了。所以 RoPE 总是先做投影、再旋转，旋转之后紧接着点积（第 5 课 §3 的代码 `q = rope(W_q @ h_t, t)`）。另外，真实的打分是 $d/2$ 条跑道（[§6](#p6)）上的结果按内容加权求和，不是一个角度。
:::

::: ex 同一个相对距离，三种算法
取 $\mathbf q=(1,2)$，$\mathbf k=(0.5,-1)$，$\theta=0.1$，查询在 $t=5$，键在 $j=2$。

**① 各转各的再点积。** 查询转 $t\theta=0.5$，键转 $j\theta=0.2$：

$$\begin{aligned}\mathbf R(0.5)\mathbf q&=(\cos0.5-2\sin0.5,\ \ \sin0.5+2\cos0.5)\\&=(0.8776-0.9589,\ \ 0.4794+1.7552)=(-0.0813,\ \ 2.2346)\\[4pt] \mathbf R(0.2)\mathbf k&=(0.5\cos0.2+\sin0.2,\ \ 0.5\sin0.2-\cos0.2)\\&=(0.4900+0.1987,\ \ 0.09933-0.98007)=(0.6887,\ \ -0.8807)\end{aligned}$$

点积 $-0.0813\times0.6887+2.2346\times(-0.8807)=-0.0560-1.9680=-2.0240$。

**② 只转一个相对角。** 按 (M3.15)，只需把键转 $(j-t)\theta=-0.3$：

$$\begin{aligned}\mathbf R(-0.3)\mathbf k&=(0.5\cos0.3-\sin0.3,\ \ -0.5\sin0.3-\cos0.3)\\&=(0.4777-0.2955,\ \ -0.1478-0.9553)=(0.1822,\ \ -1.1031)\end{aligned}$$

再和没转过的 $\mathbf q$ 点积：$\mathbf q^\top\mathbf R(-0.3)\mathbf k=0.1822-2.2062=-2.0240$。

**③ 复数。** [§3](#p3) 算过 $q\bar k=-1.5+2\mathrm i$；相位因子是 $e^{\mathrm i(\alpha-\beta)}=e^{0.3\mathrm i}=0.9553+0.2955\mathrm i$。

$$\begin{aligned}\operatorname{Re}\big[(-1.5+2\mathrm i)(0.9553+0.2955\mathrm i)\big]&=-1.5\times0.9553-2\times0.2955\\&=-1.4330-0.5910=-2.0240\end{aligned}$$

换成 $(t,j)=(13,10)$ 或 $(103,100)$，旋转后的坐标完全不同，打分仍是 $-2.024045$（图 M3.5 右）。但 $(t,j)=(2,5)$ 的打分是 $-0.841964$：相对距离从 $+3$ 变成 $-3$，$e^{\mathrm i(\alpha-\beta)}$ 从 $e^{0.3\mathrm i}$ 变成 $e^{-0.3\mathrm i}$。RoPE 区分“在我前面 3 个”与“在我后面 3 个”。
:::

## 旋转能和谁交换次序

**问题。** 第 9 课的 MLA 里，键不是直接算出来的，而是由一个缓存的短向量 $\mathbf c_j$（第 9 课的 $\mathbf c^{KV}_j$）乘上投影矩阵 $W$ 展开的：$\mathbf k_j=W\mathbf c_j$。加上 RoPE 后，打分变成 $\mathbf q^\top\mathbf R_{j-t}\,W\mathbf c_j$（这里 $\mathbf R_{j-t}$ 是按相对位置转的旋转）。[M01 §6](m01-vectors-matrices.html#p6) 的矩阵吸收要求夹在 $\mathbf q^\top$ 与 $\mathbf c_j$ 之间的矩阵对所有 $j$ 都相同，好事先把它乘进查询；可夹在中间的 $\mathbf R_{j-t}$ 随 $j$ 变。能不能把它挪到 $W$ 的另一边去？这就要问：$\mathbf RW$ 等不等于 $W\mathbf R$？

**一般不等。**用 [M01 §2](m01-vectors-matrices.html#p2) 的例子：拉伸 $\Lambda=\diag(2,1)$（M01 记作 $D$；本课的 $S$ 已是 §1 的剪切，$D$ 留给下面的镜像）、旋转 $\mathbf R(90°)$，

$$\mathbf R(90°)\Lambda=\begin{pmatrix}0&-1\\2&0\end{pmatrix}\ne \Lambda\,\mathbf R(90°)=\begin{pmatrix}0&-2\\1&0\end{pmatrix}.$$

对 $\mathbf x=(1,1)$：先拉伸后旋转得 $(-1,2)$，先旋转后拉伸得 $(-2,1)$（图 M3.6 的 (a)(b)）。那么，**哪些矩阵能和所有旋转交换？**

::: deriv 能与所有二维旋转交换的矩阵
$\mathbf R(\varphi)=\cos\varphi\,I+\sin\varphi\,J,\qquad J:=\mathbf R(90°)=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ || 把 (M3.4) 拆成对角部分与非对角部分
$A\,\mathbf R(\varphi)=\mathbf R(\varphi)\,A\ \ \text{对所有 }\varphi\iff AJ=JA$ || ⇐：$A$ 与 $I$、$J$ 都交换，就与它们的组合交换；⇒：取 $\varphi=90°$
$A=\begin{pmatrix}a&b\\c&d\end{pmatrix}:\quad AJ=\begin{pmatrix}b&-a\\d&-c\end{pmatrix},\quad JA=\begin{pmatrix}-c&-d\\a&b\end{pmatrix}$ || 行乘列
$AJ=JA\iff b=-c,\ \ a=d$ || 比较四格：$(1,1)$ 格给 $b=-c$，$(1,2)$ 格给 $a=d$，另两格重复
$A=\begin{pmatrix}a&-c\\c&a\end{pmatrix}=aI+cJ$ || 正是 (M3.9) 里“乘以复数 $a+c\,\mathrm i$”的矩阵
:::

::: eq 与所有旋转交换的，只有复数乘法 | (M3.16)
$$A\,\mathbf R(\varphi)=\mathbf R(\varphi)\,A\ \ \forall\varphi\quad\iff\quad A=aI+cJ=\begin{pmatrix}a&-c\\c&a\end{pmatrix}$$
按 (M3.9)，$aI+cJ=\sqrt{a^2+c^2}\,\mathbf R(\arg(a+c\,\mathrm i))$：能与每一个二维旋转交换次序的矩阵，恰好是“均匀缩放 + 旋转”，也就是乘以一个复数。剪切、单方向拉伸、镜像，以及几乎所有学出来的权重矩阵，都不在其中。
:::

::: fig figures/m03-noncommute.svg | 图 M3.6 旋转与拉伸交换次序后结果不同；旋转只与“复数乘法”交换 | wide
**比较 (a) 与 (b) 的红色椭圆：同样是一次拉伸加一次旋转，先后顺序一换，椭圆一个竖着、一个横着，$\mathbf x=(1,1)$ 的终点也从 $(-1,2)$ 变成 $(-2,1)$。** (c) 里的 $W=\begin{pmatrix}2&-1\\1&2\end{pmatrix}=2I+J$ 是乘以复数 $2+\mathrm i$，它把单位圆变成半径 $\sqrt5$ 的圆，所以先 $W$ 后 $\mathbf R$ 与先 $\mathbf R$ 后 $W$ 结果相同，都是 $(-3,1)$。点线是单位圆，虚线是第一步之后，实线是两步之后。
:::

回到第 9 课。MLA 的上投影 $W^{UK}_i\in\R^{128\times512}$ 是学出来的，连方阵都不是，更不会是 (M3.16) 那种形状。所以夹在中间的旋转挪不出去，“吸收”之后的查询 $(W^{UK}_i)^\top\mathbf R_{j-t}^\top\mathbf q$ 对每个历史位置 $j$ 都不同，不能事先算好。DeepSeek-V2 论文的原话译成中文是：与当前生成位置有关的 RoPE 矩阵会夹在 $W^Q$ 与 $W^{UK}$ 之间，而矩阵乘法不满足交换律 [@deepseekv2]。[§7](#p7) 用一个带数字的小例子把这句话算一遍。

::: note 笔者补充：为什么不能把旋转“搬”到缓存那一侧
一个自然的想法是：在写缓存时就给 $\mathbf c_j$ 乘一个矩阵 $\mathbf R'_j$，使 $\mathbf R_jW=W\mathbf R'_j$，这样打分变成 $(W^\top\mathbf R_t\mathbf q)^\top(\mathbf R'_j\mathbf c_j)$，吸收又可以做了。单看一个头，这样的 $\mathbf R'_j$ 是找得到的：$W$ 可逆时 $\mathbf R'_j=W^{-1}\mathbf R_jW$ 就满足这个等式（$W^{UK}_i$ 这种满行秩的“宽”矩阵也有类似的解），只不过它一般不是旋转。问题在于 MLA 的 $\mathbf c_j$ 被所有 128 个头共享，而每个头的 $W^{UK}_i$ 不同，于是需要的 $\mathbf R'_j$ 也随头而变；缓存里却只能存一份 $\mathbf c_j$。要为每个头存一份“搬过去”的版本，缓存就回到了 MHA 的大小。这段论证是笔者的分析，论文只给出“矩阵乘法不满足交换律”这一句理由 [@deepseekv2]。
:::

:::: quiz 小练习：谁和 R(90°) 交换
下列矩阵中，哪些与所有二维旋转交换？（1）$\begin{pmatrix}3&0\\0&3\end{pmatrix}$；（2）$\begin{pmatrix}1&2\\-2&1\end{pmatrix}$；（3）$\begin{pmatrix}1&0\\0&-1\end{pmatrix}$；（4）$\mathbf R(0.7)$。对不交换的，算出它与 $J$ 的两种乘积。
::: answer 看答案
按 (M3.16)，要看是否形如 $\begin{pmatrix}a&-c\\c&a\end{pmatrix}$。

（1）$a=3$，$c=0$：是（数乘，即乘以实数 3）。（2）$a=1$，$c=-2$：是（乘以复数 $1-2\mathrm i$）。（4）$\mathbf R(0.7)$：$a=\cos0.7$，$c=\sin0.7$，是；这就是 (M3.13) 说的“二维旋转两两交换”。

（3）镜像 $D=\diag(1,-1)$ 不是：$DJ=\begin{pmatrix}1&0\\0&-1\end{pmatrix}\begin{pmatrix}0&-1\\1&0\end{pmatrix}=\begin{pmatrix}0&-1\\-1&0\end{pmatrix}$，$JD=\begin{pmatrix}0&-1\\1&0\end{pmatrix}\begin{pmatrix}1&0\\0&-1\end{pmatrix}=\begin{pmatrix}0&1\\1&0\end{pmatrix}$，两者恰好差一个负号：$DJ=-JD$。镜像把逆时针变成顺时针，所以 $D\mathbf R(\varphi)=\mathbf R(-\varphi)D$。
:::
::::

::: note 这一页用在哪里
[第 5 课 §2](../deepseek/lesson-05-modern-block.html#p2)：式 (5.5) 与“旋转后的内积只依赖 $j-t$”的三行推导，就是本页的 (M3.14)、(M3.15)。[第 9 课 §4](../deepseek/lesson-09-v2-mla.html#p4)：RoPE 与矩阵吸收的冲突，用的是 (M3.14) 与“旋转与一般矩阵不可交换” (M3.16)；[第 9 课 §5](../deepseek/lesson-09-v2-mla.html#p5) 的解耦 RoPE 是对这个冲突的工程回答。
:::

::: key 本页要点
- 二维旋转的复合是角度相加，而且可以交换次序：$\mathbf R(\alpha)\mathbf R(\beta)=\mathbf R(\alpha+\beta)$，$\mathbf R(\theta)^t=\mathbf R(t\theta)$ (M3.13)。三维里绕不同轴的旋转一般不可交换。
- 相对角恒等式 $\mathbf R(\alpha)^\top\mathbf R(\beta)=\mathbf R(\beta-\alpha)$ (M3.14) 推出 $\langle\mathbf R(\alpha)\mathbf q,\mathbf R(\beta)\mathbf k\rangle=\mathbf q^\top\mathbf R(\beta-\alpha)\mathbf k$ (M3.15)：两边各转一次，只剩相对角。这就是 RoPE。
- 能与所有旋转交换的矩阵只有 $aI+cJ$（复数乘法）(M3.16)。学出来的投影矩阵不是这种形状，所以夹在中间的旋转挪不出去，MLA 的吸收因此失效。
:::

# d 维旋转与多尺度频率 | 块对角旋转矩阵、d/2 个复数、频率与波长 | 30

一个注意力头不是二维的。DeepSeek LLM 7B 的隐藏维度 4096、32 个头，每头 $d_h=128$ 维；DeepSeek-V2/V3 专门承载 RoPE 的那一段是 64 维 [@deepseek-configs]。二维旋转只能转一个平面，而且 [§4](#p4) 的失败模式说过：一个频率会“绕圈”。这一页用同一个办法同时解决这两个问题：把 $d$ 维拆成 $d/2$ 个平面，每个平面按自己的频率旋转。

## 块对角旋转矩阵

**含义。** 把 $d$ 维向量的分量两两配对，$(x_0,x_1)$、$(x_2,x_3)$、……，共 $d/2$ 对（分量从 0 数起）。第 $i$ 对看成一个二维平面，在位置 $t$ 旋转 $t\theta_i$；不同的对用不同的**频率** $\theta_i$（每前进一个 token 转过的弧度）。

**理解。** 每一对只在自己的平面里转，不碰别的分量。写成矩阵，就是把 $d/2$ 个 $2\times2$ 旋转沿对角线排开，其余位置全是 0。图 M3.7 画了 $d=8$ 的情形。

**数学。**

::: eq 块对角旋转矩阵 | (M3.17)
$$\mathbf R_{\Theta,t}=\diag\big(\mathbf R(t\theta_0),\ \mathbf R(t\theta_1),\ \dots,\ \mathbf R(t\theta_{d/2-1})\big)\in\R^{d\times d}$$
$\Theta=\{\theta_0,\dots,\theta_{d/2-1}\}$ 是全部频率。按 [M01 §4](m01-vectors-matrices.html#p4) 的分块乘法，$\mathbf R_{\Theta,t}\mathbf x$ 的第 $i$ 对等于 $\mathbf R(t\theta_i)$ 乘 $\mathbf x$ 的第 $i$ 对。RoFormer 论文的式 (15) 就是这个矩阵（下标从 1 数起）[@roformer]。
:::

::: fig figures/m03-block-rotation.svg | 图 M3.7 d = 8 的 RoPE 旋转矩阵在位置 t = 3：四个 2×2 旋转沿对角线排开，各转各的角度 | wide
**看右边四个钟面：同样是 $t=3$，第 0 对的指针转了 $171.9°$，第 3 对只转了 $0.17°$。** 取 $d=8$、底数 $b=10000$，频率 $\theta_i=10000^{-2i/8}=10^{-i}$，恰好是 1、0.1、0.01、0.001。左边的 $8\times8$ 矩阵里只有对角线上四个彩色块非零，每块就是 (M3.4) 的 $\mathbf R(3\theta_i)$；第 2、3 块转角太小，$\cos$ 已经显示成 $1.000$。每个数由生成脚本算出。
:::

块对角矩阵继承了二维旋转的全部性质，推导只是“每一块各做一遍”：

::: deriv 块对角旋转的相对位置性质
$\mathbf R_{\Theta,t}^\top=\diag\big(\mathbf R(t\theta_i)^\top\big)_{i}$ || 块对角矩阵转置时，每一块各自转置，位置不动
$\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,j}=\diag\big(\mathbf R(t\theta_i)^\top\mathbf R(j\theta_i)\big)_{i}$ || 两个同样分块的块对角矩阵相乘，对应块相乘（M01 的分块乘法）
$=\diag\big(\mathbf R((j-t)\theta_i)\big)_{i}=\mathbf R_{\Theta,j-t}$ || 每一块用相对角恒等式 (M3.14)
$\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,t}=\mathbf R_{\Theta,0}=I$ || 上一行取 $j=t$：$\mathbf R_{\Theta,t}$ 是正交矩阵
:::

::: eq 块对角旋转的三条性质 | (M3.18)
$$\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,j}=\mathbf R_{\Theta,j-t},\qquad \mathbf R_{\Theta,t}\mathbf R_{\Theta,s}=\mathbf R_{\Theta,t+s}=\mathbf R_{\Theta,s}\mathbf R_{\Theta,t},\qquad \norm{\mathbf R_{\Theta,t}\mathbf x}=\norm{\mathbf x}$$
第二式同理逐块用 (M3.13)。于是 $\langle\mathbf R_{\Theta,t}\mathbf q,\mathbf R_{\Theta,j}\mathbf k\rangle=\mathbf q^\top\mathbf R_{\Theta,j-t}\mathbf k$，这正是 RoFormer 式 (16) 里的 $R^d_{\Theta,n-m}=(R^d_{\Theta,m})^\top R^d_{\Theta,n}$ [@roformer]。
:::

::: warn 哪两维配成一对，是一个约定
本课按 RoFormer 把相邻两维 $(x_{2i},x_{2i+1})$ 配对；有的实现把前后两半配对，$(x_i,x_{i+d/2})$（[第 5 课 §3](../deepseek/lesson-05-modern-block.html#p3)）。两者只差一次固定的重排：设置换矩阵 $P$（每行每列恰好一个 1，各列是换了次序的坐标向量，所以是正交矩阵）把后一种次序排成前一种，后一种写法的旋转就是 $P^\top\mathbf R_{\Theta,t}P$。由 $PP^\top=I$，$(P^\top\mathbf R_{\Theta,t}P)^\top(P^\top\mathbf R_{\Theta,j}P)=P^\top\mathbf R_{\Theta,j-t}P$，相对位置性质照样成立。失败出在混用：按一种配对训练出的 $W^Q$、$W^K$ 放进另一种写法，被当成同一个平面来转的两个分量就配错了。
:::

## 复数写法：d/2 个复数各乘各的相位

把第 $i$ 对看成一个复数 $\tilde q_i=q_{2i}+\mathrm i\,q_{2i+1}$，一个 $d$ 维实向量就是 $d/2$ 个复数。$\mathbf R_{\Theta,t}\mathbf q$ 在复数写法下只是“第 $i$ 个复数乘以 $e^{\mathrm it\theta_i}$”，不需要矩阵。

::: deriv RoPE 打分的复数形式
$(\mathbf R_{\Theta,t}\mathbf q)^\top(\mathbf R_{\Theta,j}\mathbf k)=\sum_{i=0}^{d/2-1}\big(\mathbf R(t\theta_i)\mathbf q^{(i)}\big)^\top\big(\mathbf R(j\theta_i)\mathbf k^{(i)}\big)$ || 点积按对拆开：$d$ 项乘积之和分成 $d/2$ 组，每组是一对的二维点积；$\mathbf q^{(i)}$ 是第 $i$ 对
$=\sum_{i}\operatorname{Re}\big[\tilde q_i\,\overline{\tilde k_i}\,e^{\mathrm i(t-j)\theta_i}\big]$ || 每一对用 (M3.15)，$\alpha=t\theta_i$，$\beta=j\theta_i$
$=\operatorname{Re}\sum_{i}\tilde q_i\,\overline{\tilde k_i}\,e^{\mathrm i(t-j)\theta_i}$ || 实部对加法可以交换
:::

::: eq RoPE 打分的复数形式 | (M3.19)
$$\mathbf q_t^\top\mathbf k_j=\operatorname{Re}\sum_{i=0}^{d/2-1}h_i\,e^{\mathrm i(t-j)\theta_i},\qquad h_i=\tilde q_i\,\overline{\tilde k_i}$$
$\mathbf q_t=\mathbf R_{\Theta,t}\mathbf q$、$\mathbf k_j=\mathbf R_{\Theta,j}\mathbf k$ 是旋转后的查询与键。$h_i$ 只由内容决定（第 $i$ 对的“内容系数”），$e^{\mathrm i(t-j)\theta_i}$ 只由相对距离决定。第 5 课式 (5.7) 与“远程衰减”的讨论都从这个式子出发。
:::

## 频率怎么取：一组几何级数

(M3.18) 对任何一组频率都成立，推导不管它们取多少。RoFormer 沿用了原始 Transformer 正弦位置编码的设定，取底数 $b=10000$ [@roformer]：

::: eq 频率与波长 | (M3.20)
$$\theta_i=b^{-2i/d},\qquad \lambda_i=\frac{2\pi}{\theta_i}=2\pi\,b^{2i/d},\qquad i=0,1,\dots,\tfrac d2-1$$
$\lambda_i$ 叫第 $i$ 对的**波长**：这根指针转满一圈要走多少个 token。相邻两对的频率之比固定为 $b^{-2/d}$，所以频率（和波长）是等比数列，也叫几何级数。
:::

::: ex 算几根指针（d = 128 与 d = 64，b = 10000）
**$d=128$。** $\theta_0=1$ rad/token，$\lambda_0=2\pi\approx6.28$：最快的指针每个 token 转 $57.30°$。$\theta_1=10000^{-2/128}=0.86596$。$\theta_{32}=10000^{-64/128}=0.01$，$\lambda_{32}=628.3$。最慢的 $\theta_{63}=10000^{-126/128}=1.1548\times10^{-4}$，$\lambda_{63}\approx54{,}410$。

**每 16 对，波长乘 10。** $\log_{10}\lambda_i=\log_{10}2\pi+\frac{2i}{d}\log_{10}b=0.798+\frac{i}{16}$（$d=128$，$\log_{10}10000=4$），每增加一对，对数波长增加 $1/16$。所以 64 根指针均匀地铺满了从 6 个 token 到 5 万多个 token 的四个数量级，每个数量级 16 根。

**$d=64$（DeepSeek-V2/V3 的解耦 RoPE）。** 相邻比 $10000^{-2/64}=0.74989$，最慢的 $\theta_{31}=1.3335\times10^{-4}$，$\lambda_{31}\approx47{,}117$。因为 $2i/64=4i/128$，64 维的第 $i$ 对恰好是 128 维的第 $2i$ 对：维度减半，等于隔一根取一根指针。
:::

::: fig figures/m03-frequencies.svg | 图 M3.8 RoPE 各对维度的波长与频率（b = 10000，对数纵轴） | wide
**看这条直线：纵轴取对数之后，64 个波长排成一条斜线，从 6.28 一直升到 54,410；琥珀色区域里的点，在 4096 个 token 之内连一圈都转不满。** 蓝点是 $d=128$，橙圈是 $d=64$，橙圈恰好落在每隔一个蓝点上。灰色 × 是下一节的对照：首尾频率不变、中间改成等差排列，64 个波长几乎全挤在图的底部。左轴读波长，右轴读频率，两者是同一条曲线的两种刻度（$\lambda=2\pi/\theta$）。每个点由生成脚本按 (M3.20) 算出。
:::

## 为什么取等比数列：这是一个选择

(M3.18) 对任何一组频率都成立，所以“频率取等比数列”不是推出来的，而是设计上的**选择**。要看清它选得好在哪里，拿另一种同样自然的选法比一比：首尾两根指针不变（$d=128$，$\theta_0=1$，$\theta_{63}=1.1548\times10^{-4}$），中间改成**等差**排列，每一对比上一对少转 $0.01587$ rad。按波长把 64 根指针分成三档（笔者用 NumPy 计数）：

| 波长 $\lambda_i$ | 等比（RoPE 的选法） | 等差（对照） |
|---|---|---|
| 短于 100 个 token | 20 根 | 60 根 |
| 100 到 4096 个 token | 26 根 | 3 根 |
| 长于 4096 个 token | 18 根 | 1 根 |

等差排列的前 60 根指针频率都大于 $2\pi/100=0.0628$，几十个 token 之内就转满一圈；倒数第二根的波长也只有 393。从几百到几千个 token 的距离，几乎没有指针能“慢慢地、单调地”反映它。等比排列让对数波长等距，每个数量级分到同样多的指针（$d=128$ 时每个数量级 16 根），远近各个尺度都有指针负责。RoFormer 论文本身只说沿用原始 Transformer 的设定，并称这一设定带来“远程衰减”（long-term decay）性质 [@roformer]（这一说法证明了什么、没证明什么，见[第 5 课 §3](../deepseek/lesson-05-modern-block.html#p3)）；论文没有拿等比与别的排列作比较，这里的比较是笔者为说明这个选择而做的。


## 为什么要很多个频率：一根指针会绕圈

[§4](#p4) 的失败模式：一个频率 $\theta$ 只能确定角度除以 $2\pi$ 的余数，相对距离 $\Delta$ 与 $\Delta+2\pi/\theta$ 看起来几乎一样。多个频率之所以能解决它，是因为要让所有指针**同时**回到原位，需要每一个 $\Delta\theta_i$ 同时接近 $2\pi$ 的整数倍，频率越多、越参差，这就越不可能。

用一个能精确计算的情形来看：令 $\mathbf q=\mathbf k=(1,1,\dots,1)$（每对的复数都是 $1+\mathrm i$，$h_i=\abs{1+\mathrm i}^2=2$），按 (M3.19)，打分除以 $\Delta=0$ 时的值 $d$，得到 $\frac2d\sum_i\cos(\Delta\theta_i)$。它与第 5 课式 (5.11) 左边的“同向平均得分”是同一个式子。

::: fig figures/m03-aliasing.svg | 图 M3.9 单一频率会绕圈，多频率不会：同向查询与键的打分随相对距离的变化 | wide
**看 (a) 的红线：一个频率 $\theta=0.5$ 的打分 $\cos(0.5\Delta)$ 每隔 $2\pi/0.5\approx12.57$ 个 token 就回到 1；$\Delta=13$ 时是 0.9766，几乎和紧挨着的 token 一样“近”。** 蓝线是 $d=128$ 的 64 个频率合起来的打分，$\Delta=13$ 时是 0.6600，此后缓慢下降、不再回到 1。(b) 把距离拉长到 65536（对数横轴）：$\Delta=10$ 时 0.669，100 时 0.477，1000 时 0.159，4096 时 $-0.053$；$\Delta\ge100$ 之后的最大值只有 0.497。半透明的带是每个像素列内全部整数 $\Delta$ 的最小到最大值，实线是它们的平均。全部由生成脚本计算。
:::

::: ana 类比：一只有很多根指针的钟（沿用第 5 课）
**映射。** 每一对维度 = 一根指针；$\theta_i$ = 转速；$\lambda_i$ = 转一圈要多少个 token；位置 $t$ = 时刻；打分 (M3.19) = 把每根指针上“查询与键的夹角余弦”按内容系数 $h_i$ 加权后求和。

**收益。** 秒针、分针、时针合起来，才能既分辨一秒之差、又数清一整天。RoPE 也一样：快针（$\lambda_i$ 只有几个 token）分辨相邻的 token，慢针（$\lambda_i$ 上万）区分远近。

**失效的地方。** 真实时钟的转速是整数比，秒针转满一圈，分针恰好走一格；RoPE 的相邻比值是 0.866 这样的数，没有“进位”，也没有哪根针单独代表“第几个 token”。模型也从不直接读角度，它只看到 (M3.19) 这个加权和，权重 $h_i$ 由内容决定、是学出来的。
:::

## 慢针：在训练长度内转不满一圈

哪些指针在预训练长度 $L_{ctx}$ 之内连一圈都转不满？条件是 $\lambda_i>L_{ctx}$：

::: eq 转不满一圈的条件 | (M3.21)
$$\lambda_i>L_{ctx}\iff 2\pi\,b^{2i/d}>L_{ctx}\iff i>\frac d2\cdot\frac{\ln(L_{ctx}/2\pi)}{\ln b}$$
两边取自然对数即得（$b>1$，所以 $\ln b>0$，不等号方向不变）。$L_{ctx}$ 是预训练的上下文长度（token），也就是训练时见过的最大相对距离的量级。
:::

::: ex DeepSeek 的两种头宽（b = 10000，预训练长度 4096）
DeepSeek-V2/V3 的配置里 RoPE 底数是 10000、预训练长度 4096，解耦 RoPE 每头 64 维 [@deepseek-configs]。$\ln(4096/2\pi)=\ln651.9=6.480$，$\ln10000=9.210$。

- $d=64$：$i>32\times6.480/9.210=22.51$，所以 $i=23,\dots,31$ 共 **9 对**。最慢的一对在 $t=4096$ 转了 $4096\times1.3335\times10^{-4}=0.5462$ rad $=31.30°$。
- $d=128$：$i>64\times0.7036=45.03$，所以 $i=46,\dots,63$ 共 **18 对**。最慢的一对在 $t=4096$ 只转了 $27.10°$，约 7.5% 圈（$4096/54410$）。

这些慢针在训练中都没转满一圈：$d=64$ 时，9 对里转得最多的 $i=23$ 在 $t=4096$ 转到约 $313°$，$i=27$ 约 $99°$，最慢的 $i=31$ 只见过从 $0°$ 到约 $31°$ 的一小段角度。把模型直接用到更长的上下文，它们会转到训练中从未出现过的角度；第 5 课 §6、§7 的上下文扩展方法（PI、NTK-aware、YaRN）讨论的就是怎样把这些慢针“拨慢”。
:::

:::: quiz 小练习：快针转了几圈
$d=64$、$b=10000$，位置 $t=100$。（1）最快的一对（$\theta_0=1$）转了多少圈？对 $360°$ 取余是多少度？（2）最慢的一对转了多少度？（3）哪一根更能说明“这是第 100 个 token”？
::: answer 看答案
（1）$100$ rad $\div2\pi=15.915$ 圈，余下 $0.915$ 圈 $=329.58°$。第 5 课图 5.3 里 $t=100$ 时最左边那根快针的读数 $329.6°$ 就是它。

（2）$100\times1.3335\times10^{-4}=0.013335$ rad $=0.764°$。

（3）快针的角度只告诉你“100 除以 $2\pi$ 的余数”，$t=100$ 与 $t=100+2\pi k$ 附近的位置在它看来差不多；慢针的角度 $0.764°$ 在 $[0,31.3°]$ 的训练范围内是单调的，直接对应“离开头有多远”。区分远近靠慢针，区分紧邻的 token 靠快针。
:::
::::

::: note 这一页用在哪里
[第 5 课 §3](../deepseek/lesson-05-modern-block.html#p3)：块对角旋转 (5.6)、复数形式 (5.7)、频率与波长 (5.8) 与多频率时钟，就是本页的 (M3.17)–(M3.20)。[第 5 课 §6](../deepseek/lesson-05-modern-block.html#p6)、[§7](../deepseek/lesson-05-modern-block.html#p7)：上下文扩展从 (M3.21) 的“慢针”出发。[第 9 课 §5](../deepseek/lesson-09-v2-mla.html#p5)：解耦 RoPE 的 $d_h^R=64$ 维，就是本页 $d=64$ 的那一组频率。[第 10 课 §6](../deepseek/lesson-10-v2-moe-training.html#p6)：DeepSeek-V2 用 YaRN 把上下文拉到 128K，作用的就是这 32 对频率；每一对改多少，按它在 4096 个 token 内转了几圈分档：(M3.21) 的慢针改得最多，快针不动。
:::

::: key 本页要点
- $d$ 维 RoPE 是 $d/2$ 个互不干扰的二维旋转：块对角矩阵 $\mathbf R_{\Theta,t}$ (M3.17)。逐块套用二维结论，得到 $\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,j}=\mathbf R_{\Theta,j-t}$，且它是正交矩阵 (M3.18)。
- 复数写法：$\mathbf q_t^\top\mathbf k_j=\operatorname{Re}\sum_ih_ie^{\mathrm i(t-j)\theta_i}$，内容系数乘相位 (M3.19)。
- 频率 $\theta_i=b^{-2i/d}$ 是几何级数 (M3.20)：$d=128$、$b=10000$ 时波长从 6.28 到 54,410，每 16 对乘 10。等比是一种选择：首尾不变、改成等差的话，64 根指针里有 60 根的波长短于 100 个 token。一个频率会绕圈，多个频率合起来不会；$\lambda_i>L_{ctx}$ 的慢针在训练中转不满一圈 (M3.21)，DeepSeek 的 64 维里有 9 对。
:::

# 一个完整的 RoPE 算例 | 4 维手算、平移不变的核对，以及它为什么挡住 MLA 的矩阵吸收 | 25

前面每一页都在二维里算例子。这一页把所有步骤放进一个 4 维的例子里完整走一遍：两对维度、两个频率、真实的底数 10000。4 维小到可以手算，又大到能看清“快针”和“慢针”的差别。最后用同一套数字说明 RoPE 为什么让 MLA 的矩阵吸收失效。

## 设定

$d=4$，$b=10000$，所以只有两对维度，频率 $\theta_0=10000^{0}=1$，$\theta_1=10000^{-2/4}=0.01$。查询 $\mathbf q=(1,2,2,1)$ 在位置 $t=5$，键 $\mathbf k=(0.5,-1,1,1)$ 在位置 $j=2$。不加位置时的点积 $\mathbf q^\top\mathbf k=0.5-2+2+1=1.5$，其中第 0 对贡献 $-1.5$，第 1 对贡献 $3$。

## 第一步：每一对按自己的频率旋转

| 对 $i$ | $\theta_i$ | 查询转角 $t\theta_i$ | 键转角 $j\theta_i$ | 旋转后的查询分量 | 旋转后的键分量 |
|---|---|---|---|---|---|
| 0（快针） | 1 | 5 rad（$286.5°$） | 2 rad（$114.6°$） | $(2.2015,\ -0.3916)$ | $(0.7012,\ 0.8708)$ |
| 1（慢针） | 0.01 | 0.05 rad（$2.86°$） | 0.02 rad（$1.15°$） | $(1.9475,\ 1.0987)$ | $(0.9798,\ 1.0198)$ |

::: ex 表里的四个数是怎样算出来的
按 (M3.4)，$\mathbf R(\varphi)(x,y)=(x\cos\varphi-y\sin\varphi,\ x\sin\varphi+y\cos\varphi)$。

- 第 0 对的查询 $(1,2)$，转 5 rad：$\cos5=0.28366$，$\sin5=-0.95892$。$(0.28366+1.91785,\ -0.95892+0.56732)=(2.20151,\ -0.39160)$。
- 第 0 对的键 $(0.5,-1)$，转 2 rad：$\cos2=-0.41615$，$\sin2=0.90930$。$(-0.20807+0.90930,\ 0.45465+0.41615)=(0.70122,\ 0.87080)$。
- 第 1 对的查询 $(2,1)$，转 0.05 rad：$\cos0.05=0.99875$，$\sin0.05=0.04998$。$(1.99750-0.04998,\ 0.09996+0.99875)=(1.94752,\ 1.09871)$。
- 第 1 对的键 $(1,1)$，转 0.02 rad：$\cos0.02=0.99980$，$\sin0.02=0.02000$。$(0.97980,\ 1.01980)$。

核对长度：旋转后的查询 $\sqrt{2.2015^2+0.3916^2+1.9475^2+1.0987^2}=\sqrt{10.000}=3.1623=\norm{\mathbf q}$，正交矩阵不改变长度。
:::

## 第二步：点积

- 第 0 对：$2.20151\times0.70122+(-0.39160)\times0.87080=1.54374-0.34101=1.2027$。
- 第 1 对：$1.94752\times0.97980+1.09871\times1.01980=1.90818+1.12046=3.0286$。
- 合计：$\mathbf q_5^\top\mathbf k_2=1.2027+3.0286=4.2314$。

::: fig figures/m03-rope-example.svg | 图 M3.10 4 维 RoPE 算例：两对维度各自旋转，点积是两对小点积之和 | wide
**对比左右两个面板的红色标注：快针（左，红色弧线）把查询转了 $286.5°$、键转了 $114.6°$，两者的夹角从 $126.87°$ 变成 $61.24°$，这一对的贡献从 $-1.5$ 翻成了 $+1.2027$；慢针（右，红字）只转了 $2.9°$ 和 $1.1°$，贡献从 3 变成 3.0286。** 虚线是旋转前，实线是旋转后，绿色弧线是转完之后查询与键的夹角。每个坐标由生成脚本算出。
:::

## 第三步：用复数和相对角核对

按 (M3.19)，只需要内容系数 $h_i=\tilde q_i\overline{\tilde k_i}$ 和相对距离 $t-j=3$：

- $h_0=(1+2\mathrm i)(0.5+\mathrm i)=-1.5+2\mathrm i$；$h_1=(2+\mathrm i)(1-\mathrm i)=2-2\mathrm i+\mathrm i-\mathrm i^2=3-\mathrm i$。
- $e^{3\mathrm i}=-0.98999+0.14112\mathrm i$，$e^{0.03\mathrm i}=0.99955+0.03000\mathrm i$。
- $\operatorname{Re}(h_0e^{3\mathrm i})=(-1.5)(-0.98999)-2\times0.14112=1.48499-0.28224=1.20275$。
- $\operatorname{Re}(h_1e^{0.03\mathrm i})=3\times0.99955-(-1)\times0.03000=2.99865+0.03000=3.02865$。

合计 $4.2314$，与第二步一致；这次完全不需要知道 $t$ 和 $j$ 本身，只用到 $t-j=3$。

## 第四步：平移不变

| $(t,j)$ | $(5,2)$ | $(105,102)$ | $(1005,1002)$ | $(3,0)$ | $(2,5)$ |
|---|---|---|---|---|---|
| $\mathbf q_t^\top\mathbf k_j$ | 4.231394 | 4.231394 | 4.231394 | 4.231394 | 4.735883 |

前四列的相对距离都是 3，旋转后的坐标各不相同（例如 $(105,102)$ 时查询的第 0 对是 $(1.7001,-1.4524)$），打分完全相同。最后一列的相对距离是 $-3$，打分不同。这些数是笔者用 NumPy 按 (M3.17) 直接构造矩阵算出的。

这个例子还显示了快针与慢针的分工：相对距离 3 让快针转了 3 rad（$171.9°$），第 0 对的贡献从 $-1.5$ 变成 $1.2027$，变化很大；慢针只转了 $0.03$ rad（$1.7°$），贡献几乎不变。距离小时，位置信息主要由快针携带；距离到了几百、几千，快针的角度早已转过许多圈，能稳定反映远近的是慢针。

## RoPE 为什么挡住矩阵吸收

现在用同一个查询复现第 9 课的问题。假设键不是直接给的，而是从一个 2 维的缓存向量 $\mathbf c_j$ 展开：$\mathbf k_j=W\mathbf c_j$，

$$W=\begin{pmatrix}1&0\\0&1\\1&1\\0&1\end{pmatrix}\in\R^{4\times2}.$$

这是 MLA 的缩小版：缓存里存 2 维的 $\mathbf c_j$，而不是 4 维的键（真实尺寸见第 9 课）。

**没有 RoPE 时。** 打分 $\mathbf q^\top W\mathbf c_j=(W^\top\mathbf q)^\top\mathbf c_j$。$W^\top\mathbf q=(1+2,\ 2+2+1)=(3,5)$ 算一次，就能和缓存里每一个 $\mathbf c_j$ 直接点积。这就是矩阵吸收（[M01 §6](m01-vectors-matrices.html#p6)）。

**有 RoPE 时。** 键在展开之后按位置旋转，$\mathbf k_j=\mathbf R_{\Theta,j}W\mathbf c_j$：

::: deriv 有 RoPE 时的“吸收查询”
$(\mathbf R_{\Theta,t}\mathbf q)^\top\big(\mathbf R_{\Theta,j}W\mathbf c_j\big)=\mathbf q^\top\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,j}W\mathbf c_j$ || 转置规则与结合律
$=\mathbf q^\top\mathbf R_{\Theta,j-t}\,W\mathbf c_j$ || (M3.18)
$=\big(W^\top\mathbf R_{\Theta,j-t}^\top\mathbf q\big)^\top\mathbf c_j=\big(W^\top\mathbf R_{\Theta,t-j}\,\mathbf q\big)^\top\mathbf c_j$ || 再用转置规则；$\mathbf R_{\Theta,j-t}^\top=\mathbf R_{\Theta,t-j}$
:::

::: eq RoPE 让吸收后的查询依赖于相对距离 | (M3.22)
$$\tilde{\mathbf q}^{(j)}=W^\top\mathbf R_{\Theta,t-j}\,\mathbf q,\qquad \mathbf q_t^\top\mathbf k_j=\big(\tilde{\mathbf q}^{(j)}\big)^\top\mathbf c_j$$
没有 RoPE 时 $\tilde{\mathbf q}=W^\top\mathbf q$ 对所有 $j$ 通用；有了 RoPE，每个历史位置 $j$ 都要一个自己的 $\tilde{\mathbf q}^{(j)}$。第 9 课式 (9.12) 是同一个式子。
:::

::: ex 同一个查询，四个不同的“吸收查询”
$t=5$，缓存里有 4 个 token，$j=2,3,4,5$，它们的缓存向量取 $\mathbf c_2=(1,-1)$、$\mathbf c_3=(0.5,2)$、$\mathbf c_4=(-1,1)$、$\mathbf c_5=(2,0.5)$。按 (M3.22) 用 NumPy 计算：

| $j$ | $t-j$ | $\tilde{\mathbf q}^{(j)}$ | 正确的打分 | 硬用 $(3,5)$ 得到的打分 |
|---|---|---|---|---|
| 2 | 3 | $(0.6969,\ 1.1898)$ | $-0.4929$ | $-2$ |
| 3 | 2 | $(-0.2551,\ 3.0964)$ | $6.0652$ | $11.5$ |
| 4 | 1 | $(0.8473,\ 4.9319)$ | $4.0847$ | $2$ |
| 5 | 0 | $(3,\ 5)$ | $8.5$ | $8.5$ |

“正确的打分”一列与直接旋转、展开后再点积的结果逐位相同。只有 $j=t$ 时 $\mathbf R_{\Theta,0}=I$，吸收查询才退回 $(3,5)$。如果为了省事对所有 $j$ 都用 $(3,5)$，得到的是没有 RoPE 时的打分，位置信息全部丢失。要得到正确结果，就得为每个 $j$ 算一次 $W^\top\mathbf R_{\Theta,t-j}\mathbf q$，缓存有多长就要算多少次，这和把每个 $\mathbf c_j$ 展开成键的运算量是同一个量级。按真实尺寸（$W^{UK}_i\in\R^{128\times512}$）数一数：一个头、一个 $j$，$(W^{UK}_i)^\top$ 乘一个 128 维向量是 $512\times128=65{,}536$ 次乘加；把 $\mathbf c_j$ 展开成键 $W^{UK}_i\mathbf c_j$ 也是 $128\times512=65{,}536$ 次，一次都没省下。
:::

能不能把旋转挪到缓存那一侧，即找一个作用在 $\mathbf c_j$ 上的矩阵 $\mathbf R'_j$，使 $\mathbf R_{\Theta,j}W=W\mathbf R'_j$？在这个例子里不可能（取 $j=1$ 就够了）：$\mathbf R_{\Theta,1}$ 把 $W$ 的第一列 $(1,0,1,0)$ 转到 $(0.5403,\ 0.8415,\ 0.99995,\ 0.0100)$，而 $W\mathbf R'\mathbf c$ 永远落在 $W$ 的两列张成的平面里，那里的向量都形如 $(a,b,a+b,b)$；这里 $a+b=1.3818\ne0.99995$，所以旋转后的键已经跑出了这个平面，任何 $\mathbf R'$ 都追不回来。这条理由依赖玩具例子里的 $W$ 是“高”矩阵（2 维展开成 4 维，列空间只是 4 维里的一个平面）。真实 MLA 的单个头 $W^{UK}_i\in\R^{128\times512}$ 是“宽”矩阵，一般满行秩，列空间就是整个 $\R^{128}$，这一条障碍并不存在；那里真正的障碍是 [§5](#p5) 笔者补充说的：$\mathbf c_j$ 被所有头共享。第 9 课的解耦 RoPE 让承载位置的那一小段根本不经过 $W^{UK}$，从结构上绕开了这个问题。

## 总账：本课的每个工具用在哪里

| 工具 | 本课公式 | 用在哪里 |
|---|---|---|
| 正交矩阵：保持点积、逆 = 转置 | M3.1–3 | 第 5 课：RoPE 不改变查询和键的长度；M02 的 SVD；M07 的 Muon（半正交矩阵） |
| 二维旋转矩阵、和角公式、转置 = 反向旋转 | M3.4–6 | 第 5 课式 (5.4)；第 9 课的相对角推导 |
| 复数、共轭、点积 $=\operatorname{Re}(z\bar w)$ | M3.7–10 | 第 5 课的二维推导与式 (5.3)、(5.7) |
| 欧拉公式、极坐标、$(e^{\mathrm i\theta})^t=e^{\mathrm it\theta}$ | M3.11–12 | 第 5 课：位置 $t$ 乘以 $e^{\mathrm it\theta}$；远程衰减里的相位和 |
| 复合、相对角恒等式、旋转后的点积 | M3.13–15 | 第 5 课式 (5.5)；第 9 课 $\mathbf R_t^\top\mathbf R_j=\mathbf R_{j-t}$ |
| 与旋转交换的矩阵只有复数乘法 | M3.16 | 第 9 课：RoPE 与矩阵吸收冲突，解耦 RoPE 的动机 |
| 块对角旋转、复数形式、频率与波长、慢针 | M3.17–21 | 第 5 课 §3 的多频率时钟、§6–§7 的上下文扩展；第 9、10 课的 64 维解耦 RoPE |
| 吸收查询随相对距离变化 | M3.22 | 第 9 课式 (9.12) |

::: key 本页要点
- 4 维例子（$\theta=(1,0.01)$，$t=5$，$j=2$）：两对各转各的，快针的贡献从 $-1.5$ 变成 $1.2027$，慢针从 3 变成 $3.0286$，合计 $4.2314$；复数公式只用 $t-j=3$ 就得到同一个数。
- 平移不变：$(105,102)$、$(1005,1002)$、$(3,0)$ 的打分都是 $4.231394$；$(2,5)$ 不同，RoPE 分得清前后。
- 有 RoPE 时吸收后的查询 $W^\top\mathbf R_{\Theta,t-j}\mathbf q$ 随 $j$ 变 (M3.22)，吸收省下的运算全部还回去；这正是第 9 课要解耦 RoPE 的原因。
:::

# 练习与答案 | 12 题：概念、计算、推导 | 45

先自己做，再展开答案。计算题用计算器就够；想核对的话，每题都能用几行 NumPy 验证。

## 概念题

:::: quiz 练习 1 · 列长度为 1 就够了吗
矩阵 $M=\begin{pmatrix}1&0.6\\0&0.8\end{pmatrix}$ 的两列都是单位向量。它是正交矩阵吗？用 $\mathbf x=(1,1)$ 与 $\mathbf x=(1,-1)$ 检验它是否保持长度。
::: answer 看答案
不是。两列 $(1,0)$ 与 $(0.6,0.8)$ 长度都是 1，但点积 $0.6\ne0$，所以 $M^\top M=\begin{pmatrix}1&0.6\\0.6&1\end{pmatrix}\ne I$。

$M(1,1)=(1.6,0.8)$，长度 $\sqrt{3.2}=1.789$，比 $\sqrt2=1.414$ 长；$M(1,-1)=(0.4,-0.8)$，长度 $\sqrt{0.8}=0.894$，比 $1.414$ 短。一个方向被拉长、另一个被压短，所以它不保持长度。正交要求两条：每列长度为 1，**且**两两垂直。
:::
::::

:::: quiz 练习 2 · 查询和键能用不同的频率吗
设某个“改进版”RoPE 让查询按 $t\theta_q$ 旋转、键按 $j\theta_k$ 旋转，$\theta_q\ne\theta_k$（只看一对维度）。打分还只依赖 $t-j$ 吗？请用 (M3.15) 回答，并举出一对反例位置。
::: answer 看答案
不。按 (M3.15)，打分 $=\operatorname{Re}\big[q\bar k\,e^{\mathrm i(t\theta_q-j\theta_k)}\big]$，依赖的是 $t\theta_q-j\theta_k$，而不是 $t-j$。取 $(t,j)=(1,0)$ 与 $(2,1)$：相对距离都是 1，但相位分别是 $\theta_q$ 与 $2\theta_q-\theta_k$，两者（对 $2\pi$ 取余后）相等要求 $\theta_q-\theta_k$ 是 $2\pi$ 的整数倍；而那时 $\mathbf R(j\theta_q)=\mathbf R(j\theta_k)$ 对每个整数 $j$ 都成立，其实就是同一个频率。一般的 $\mathbf q$、$\mathbf k$ 下，两个打分因此不同。所以查询与键必须用**同一组**频率、同一种配对方式。第 5 课强调的“两种配对写法的权重不能混用”是另一层要求：配对方式还必须和训练时一致（[§6](#p6) 的注意框）。
:::
::::

:::: quiz 练习 3 · 高维旋转能交换吗
（a）RoPE 的 $\mathbf R_{\Theta,t}$ 与 $\mathbf R_{\Theta,s}$ 可以交换次序吗？（b）任意两个 $3\times3$ 旋转矩阵都能交换吗？用 $\mathbf R_z(90°)$（把 $\mathbf e_1\to\mathbf e_2$、$\mathbf e_2\to-\mathbf e_1$、$\mathbf e_3$ 不动）与 $\mathbf R_x(90°)$（$\mathbf e_1$ 不动、$\mathbf e_2\to\mathbf e_3$、$\mathbf e_3\to-\mathbf e_2$）作用在 $\mathbf e_1$ 上检验。
::: answer 看答案
（a）可以。两者分块方式相同，逐块相乘，每一块都是同一平面里的二维旋转，按 (M3.13) 可交换：$\mathbf R_{\Theta,t}\mathbf R_{\Theta,s}=\mathbf R_{\Theta,t+s}=\mathbf R_{\Theta,s}\mathbf R_{\Theta,t}$ (M3.18)。

（b）不能。$\mathbf R_x\mathbf R_z\mathbf e_1=\mathbf R_x\mathbf e_2=\mathbf e_3$；$\mathbf R_z\mathbf R_x\mathbf e_1=\mathbf R_z\mathbf e_1=\mathbf e_2$。结果不同。RoPE 能交换，是因为它只用了“固定平面里的二维旋转”这一类非常特殊的高维旋转。
:::
::::

## 计算题

:::: quiz 练习 4 · 复数的乘法与点积
（a）求 $(3+4\mathrm i)(1+2\mathrm i)$，并用“模相乘、辐角相加”核对（$\arg(3+4\mathrm i)=53.13°$，$\arg(1+2\mathrm i)=63.43°$）。（b）求 $(3+4\mathrm i)\,\overline{(1+2\mathrm i)}$，从中读出向量 $(3,4)$ 与 $(1,2)$ 的点积，以及它们的“有向面积”。
::: answer 看答案
（a）$(3+4\mathrm i)(1+2\mathrm i)=3+6\mathrm i+4\mathrm i+8\mathrm i^2=-5+10\mathrm i$。模：$\abs{-5+10\mathrm i}=\sqrt{125}=11.18=5\times\sqrt5$；辐角：$-5+10\mathrm i$ 在第二象限，$180°-\arctan2=116.57°$，与 $53.13°+63.43°=116.56°$ 相差的 $0.01°$ 只是四舍五入（精确值 $53.130°+63.435°=116.565°$）。

（b）$(3+4\mathrm i)(1-2\mathrm i)=3-6\mathrm i+4\mathrm i-8\mathrm i^2=11-2\mathrm i$。实部 11 就是点积 $3\times1+4\times2$；虚部 $-2=bc-ad=4\times1-3\times2$，按 [§3](#p3)“二维点积的复数写法”的最后一行，是有向面积 $\det\begin{pmatrix}3&1\\4&2\end{pmatrix}=2$ 取负号。
:::
::::

:::: quiz 练习 5 · 欧拉公式与棣莫弗
（a）求 $e^{\mathrm i\pi/3}$，并用 (M3.12) 求 $(e^{\mathrm i\pi/3})^6$。（b）$\theta=1$ 的快针在位置 $t=44$ 转了多少圈？它为什么“几乎”回到了起点？（c）有没有整数 $t\ne0$ 让它**精确**回到起点？
::: answer 看答案
（a）$e^{\mathrm i\pi/3}=\cos60°+\mathrm i\sin60°=0.5+0.8660\mathrm i$；$(e^{\mathrm i\pi/3})^6=e^{2\pi\mathrm i}=1$：转 6 次 $60°$ 正好一圈。

（b）$44/(2\pi)=7.0028$ 圈，只比 7 圈多 $0.0028$ 圈（约 $1.0°$），所以 $e^{44\mathrm i}\approx1$。单看这一根快针，位置 44 与位置 0 几乎分不开。

（c）没有。精确回到起点要求 $t\times1=2\pi k$，即 $\pi=t/(2k)$ 是有理数，但 $\pi$ 是无理数。所以快针永远不会精确地回到原位，只会“差一点”；[§6](#p6) 的多频率设计让所有指针同时“差一点”的情况在实际长度内不出现。
:::
::::

:::: quiz 练习 6 · DeepSeek 的 64 维 RoPE
$d=64$，$b=10000$。（a）求 $\theta_1$、$\theta_{31}$ 与 $\lambda_{31}$。（b）DeepSeek-V2 的上下文扩展目标是 $4096\times40=163{,}840$ 个 token（第 5 课 §7）。直接把原来的频率用到这个长度，最慢的一对会转多少圈？训练时它最多转过多少度？
::: answer 看答案
（a）$\theta_1=10000^{-2/64}=0.74989$；$\theta_{31}=10000^{-62/64}=1.3335\times10^{-4}$；$\lambda_{31}=2\pi/\theta_{31}\approx47{,}117$。

（b）$163840\times1.3335\times10^{-4}=21.85$ rad，约 $3.48$ 圈。训练长度 4096 内它最多转 $0.5462$ rad $=31.30°$（§6 例题）。模型从没见过的角度多达三圈多，这就是“直接外推会坏”的几何原因。
:::
::::

:::: quiz 练习 7 · 正交的两个向量，旋转后打分不为零
$d=4$，$b=10000$（$\theta=(1,0.01)$），$\mathbf q=(1,0,0,1)$ 在 $t=4$，$\mathbf k=(0,1,1,0)$ 在 $j=1$。（a）不加位置时 $\mathbf q^\top\mathbf k$ 是多少？（b）用 (M3.19) 求加 RoPE 后的打分。（c）解释为什么（a）是 0 而（b）不是。
::: answer 看答案
（a）$0+0+0+0=0$：两个向量正交。

（b）$\tilde q_0=1$，$\tilde k_0=\mathrm i$，$h_0=1\cdot\bar{\mathrm i}=-\mathrm i$；$\tilde q_1=\mathrm i$，$\tilde k_1=1$，$h_1=\mathrm i$。$t-j=3$：$\operatorname{Re}(-\mathrm i\,e^{3\mathrm i})=\operatorname{Re}(-\mathrm i\cos3+\sin3)=\sin3=0.1411200$；$\operatorname{Re}(\mathrm i\,e^{0.03\mathrm i})=-\sin0.03=-0.0299955$。合计 $0.1111245$（NumPy 直接构造矩阵计算也是这个数）。

（c）内容系数 $h_i$ 是纯虚数：实部（点积）为 0，虚部（有向面积的相反数，见 [§3](#p3)“二维点积的复数写法”）不为 0。旋转把虚部“转”进了实部：$\operatorname{Re}(h\,e^{\mathrm i\varphi})=\operatorname{Re}h\cos\varphi-\operatorname{Im}h\sin\varphi$。所以 RoPE 的打分不仅用到每对的点积，也用到每对的有向面积。
:::
::::

:::: quiz 练习 8 · 再算一个吸收查询
沿用 §7 的 $W$、$\mathbf q=(1,2,2,1)$、$t=5$。（a）写出 $\mathbf R_{\Theta,1}$ 作用在 $\mathbf q$ 上的结果（保留 4 位小数）。（b）求 $j=4$ 的吸收查询 $\tilde{\mathbf q}^{(4)}=W^\top\mathbf R_{\Theta,1}\mathbf q$，并与 §7 表格核对。
::: answer 看答案
（a）第 0 对 $(1,2)$ 转 1 rad：$\cos1=0.5403$，$\sin1=0.8415$，得 $(0.5403-1.6829,\ 0.8415+1.0806)=(-1.1426,\ 1.9221)$。第 1 对 $(2,1)$ 转 0.01 rad：$\cos0.01=0.9999500$，$\sin0.01=0.0099998$，得 $(1.9999000-0.0099998,\ 0.0199997+0.9999500)=(1.9899002,\ 1.0199497)\approx(1.9899,\ 1.0199)$。这里要多留几位：若把 $\sin0.01$ 先舍成 $0.0100$，第二个分量会算成 $1.01995$，再舍入就可能变成 $1.0200$。

（b）$W^\top=\begin{pmatrix}1&0&1&0\\0&1&1&1\end{pmatrix}$，所以 $\tilde{\mathbf q}^{(4)}=(-1.1426+1.9899,\ 1.9221+1.9899+1.0199)=(0.8473,\ 4.9319)$，与表格一致。
:::
::::

## 推导题

:::: quiz 练习 9 · 用复数证明 R(α)R(β) = R(α+β)
（a）用 (M3.9) 的对应 $w\leftrightarrow M_w$ 证明 $M_wM_z=M_{wz}$。（b）由此推出 (M3.13)，再用归纳法推出 $\mathbf R(\theta)^t=\mathbf R(t\theta)$。
::: answer 看答案
（a）对任意向量 $\mathbf x\leftrightarrow x$：$M_z\mathbf x\leftrightarrow zx$，再乘 $M_w$ 得 $w(zx)$；而 $M_{wz}\mathbf x\leftrightarrow(wz)x$。复数乘法满足结合律，$w(zx)=(wz)x$，所以 $M_wM_z\mathbf x=M_{wz}\mathbf x$ 对所有 $\mathbf x$ 成立，即 $M_wM_z=M_{wz}$。

（b）$\mathbf R(\alpha)=M_{e^{\mathrm i\alpha}}$（(M3.9) 取 $\abs w=1$）。于是 $\mathbf R(\alpha)\mathbf R(\beta)=M_{e^{\mathrm i\alpha}}M_{e^{\mathrm i\beta}}=M_{e^{\mathrm i\alpha}e^{\mathrm i\beta}}=M_{e^{\mathrm i(\alpha+\beta)}}=\mathbf R(\alpha+\beta)$（第三步是指数律 (M3.12)）。归纳：$t=1$ 成立；若 $\mathbf R(\theta)^t=\mathbf R(t\theta)$，则 $\mathbf R(\theta)^{t+1}=\mathbf R(\theta)\mathbf R(t\theta)=\mathbf R((t+1)\theta)$。
:::
::::

:::: quiz 练习 10 · 用镜像代替旋转会怎样
设 $D=\diag(1,-1)$，镜像 $F(\varphi)=\mathbf R(\varphi)D=\begin{pmatrix}\cos\varphi&\sin\varphi\\\sin\varphi&-\cos\varphi\end{pmatrix}$。（a）证明 $D\mathbf R(\psi)D=\mathbf R(-\psi)$。（b）若位置 $t$ 用 $F(t\theta)$ 代替 $\mathbf R(t\theta)$，证明打分 $\big(F(t\theta)\mathbf q\big)^\top\big(F(j\theta)\mathbf k\big)$ 也只依赖 $t-j$。（c）既然如此，为什么说这不是一种新的位置编码？
::: answer 看答案
（a）$D$ 把 $\sin$ 所在的两格变号：$D\mathbf R(\psi)D=\begin{pmatrix}\cos\psi&\sin\psi\\-\sin\psi&\cos\psi\end{pmatrix}=\mathbf R(-\psi)$（$\cos$ 偶、$\sin$ 奇）。也可以用 §5 小练习的结论 $D\mathbf R(\varphi)=\mathbf R(-\varphi)D$ 与 $D^2=I$。

（b）$F(a)^\top F(b)=D^\top\mathbf R(a)^\top\mathbf R(b)D=D\,\mathbf R(b-a)\,D=\mathbf R(a-b)$，用了 $D^\top=D$、(M3.14) 与（a）。所以打分 $=\mathbf q^\top\mathbf R\big((t-j)\theta\big)\mathbf k$，只依赖 $t-j$。

（c）$F(t\theta)\mathbf q=\mathbf R(t\theta)(D\mathbf q)$：它等于先对查询做一个**固定的**镜像 $D$，再做普通的 RoPE。固定的线性变换可以被学出来的 $W^Q$、$W^K$ 吸收，所以这只是换了个参数化。另外 $F(0)=D\ne I$，违反“位置 0 不改变向量”的初始条件；而且 $F(a)F(b)=\mathbf R(a-b)$ 是旋转而不是镜像，“再走 $s$ 步”不能写成乘以 $F(s\theta)$。旋转族满足 $\mathbf R(t\theta)\mathbf R(s\theta)=\mathbf R((t+s)\theta)$，这正是 RoPE 选旋转的好处。
:::
::::

:::: quiz 练习 11 · 块对角旋转的三条性质
（a）不借助分块，直接用 (M3.19) 证明 $\norm{\mathbf R_{\Theta,t}\mathbf x}=\norm{\mathbf x}$。（b）证明 $\mathbf R_{\Theta,t}^\top=\mathbf R_{\Theta,-t}$。（c）$d=4$、$\theta=(1,0.01)$ 时，是否存在整数 $\Delta\ne0$ 使 $\mathbf R_{\Theta,\Delta}=I$？
::: answer 看答案
（a）在 (M3.19) 中取 $\mathbf q=\mathbf k=\mathbf x$、$t=j$：$\norm{\mathbf R_{\Theta,t}\mathbf x}^2=\operatorname{Re}\sum_i\tilde x_i\overline{\tilde x_i}\,e^{0}=\sum_i\abs{\tilde x_i}^2=\sum_i(x_{2i}^2+x_{2i+1}^2)=\norm{\mathbf x}^2$。

（b）由 (M3.18)，$\mathbf R_{\Theta,t}^\top=\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,0}=\mathbf R_{\Theta,0-t}=\mathbf R_{\Theta,-t}$（$\mathbf R_{\Theta,0}=I$）。

（c）不存在。$\mathbf R_{\Theta,\Delta}=I$ 要求每一块都是单位矩阵，第 0 块要求 $\Delta\times1=2\pi k$（$k$ 为非零整数），即 $\pi=\Delta/(2k)$ 为有理数，与 $\pi$ 是无理数矛盾。所以 RoPE 的整体旋转在整数位置上永远不会精确地重复。
:::
::::

:::: quiz 练习 12 · 为什么频率取等比数列
$d=64$，$b=10000$，首尾两对的频率固定为 $\theta_0=1$、$\theta_{31}=1.3335\times10^{-4}$。（a）证明：等比频率 $\theta_i=b^{-2i/d}$ 让 $\log_{10}\lambda_i$ 随 $i$ 等距增加，并求每多一对，对数波长增加多少。（b）若改成等差排列 $\theta_i=1-i\,\delta$，$\delta$ 取让 $\theta_{31}$ 不变的值。求 $\delta$，并数出波长短于 100 个 token 的有几对、长于 4096 个 token 的有几对。（c）对等比排列做同样的计数，说明哪一种排列在“几百到几千个 token”这段距离上有指针可用。
::: answer 看答案
（a）由 (M3.20)，$\lambda_i=2\pi\,b^{2i/d}$，取对数得 $\log_{10}\lambda_i=\log_{10}2\pi+\frac{2i}{d}\log_{10}b$，是 $i$ 的一次函数，所以等距。每多一对增加 $\frac2d\log_{10}b=\frac{2}{64}\times4=0.125$，即每 8 对波长乘 10。

（b）$\delta=(1-1.3335\times10^{-4})/31=0.032254$。$\lambda_i<100\iff\theta_i>2\pi/100=0.062832\iff i<(1-0.062832)/0.032254=29.06$，所以 $i=0,\dots,29$ 共 **30 对**。$\lambda_i>4096\iff\theta_i<2\pi/4096=0.001534\iff i>(1-0.001534)/0.032254=30.96$，只有 $i=31$ 这 **1 对**。夹在中间的只有 $i=30$：$\theta_{30}=1-30\delta=0.03239$（用 $\delta$ 的更多位 $0.03225376$ 计算），$\lambda_{30}=194$。

（c）等比：$\log_{10}\lambda_i=0.798+i/8$。$\lambda_i<100\iff i<(2-0.798)\times8=9.62$，共 10 对；$\lambda_i>4096\iff i>(3.612-0.798)\times8=22.51$，共 9 对（与 §6 的例题一致）；中间 $100\le\lambda_i\le4096$ 的有 $32-10-9=13$ 对。等差排列在这段距离上只有 1 根指针（$\lambda=194$），等比排列有 13 根。所以等比排列让远近各个尺度都有指针负责；等差排列把 30 根指针浪费在几十个 token 以内。
:::
::::

```refs
- id: roformer
  title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  a: Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021; Neurocomputing 568:127063, 2024). arXiv:2104.09864（读的是 v5）
  url: https://arxiv.org/abs/2104.09864
  finding: 从“查询与键的内积只依赖相对位置”出发，在二维用复数乘以 $e^{\mathrm im\theta}$ 构造出一个解，推广为块对角旋转矩阵 $R^d_{\Theta,m}$（频率 $\theta_i=10000^{-2(i-1)/d}$），并写出 $R^d_{\Theta,n-m}=(R^d_{\Theta,m})^\top R^d_{\Theta,n}$；指出该矩阵正交、编码方式是乘性的。
  matters: 本课 §5 的相对角恒等式、§6 的块对角旋转与频率、§7 的算例都是为读懂这篇论文准备的；“正交保证稳定”的说法也出自这里。
- id: deepseekv2
  title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  a: DeepSeek-AI (2024). arXiv:2405.04434（读的是 v5）
  url: https://arxiv.org/abs/2405.04434
  finding: §2.1.3 指出 RoPE 与低秩 KV 压缩不兼容：一个与当前生成位置有关的 RoPE 矩阵会夹在 $W^Q$ 与 $W^{UK}$ 之间，而矩阵乘法不满足交换律，于是 $W^{UK}$ 无法被吸收；解法是用每头 $d_h^R=64$ 维的解耦查询与共享键单独承载 RoPE。
  matters: 本课 §5 的“旋转与一般矩阵不可交换”和 §7 的吸收失败算例，就是这句话的数学内容。
- id: deepseek-configs
  kind: artefact
  title: DeepSeek 模型配置（Hugging Face config.json：deepseek-llm-7b-base、DeepSeek-V2、DeepSeek-V3）
  a: DeepSeek-AI. 读取于 2026-09-24
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
  finding: DeepSeek LLM 7B 的 hidden_size 为 4096、num_attention_heads 为 32（每头 128 维）、rope_theta 为 10000；DeepSeek-V2 与 V3 的 qk_rope_head_dim 为 64、rope_theta 为 10000、rope_scaling 中的 original_max_position_embeddings 为 4096、kv_lora_rank 为 512。
  matters: 本课 §6 的两种头宽（128 与 64）、底数 10000 与预训练长度 4096 都取自这些配置，与第 5、9 课使用的数值一致。
- id: rope-code
  kind: artefact
  title: RoPE 的官方实现：LLaMA v1 与 DeepSeek-V3 推理代码中的 precompute_freqs_cis
  a: Meta（meta-llama/llama，llama_v1 分支 llama/model.py）；DeepSeek-AI（deepseek-ai/DeepSeek-V3，inference/model.py）。读取于 2026-09-24
  url: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
  finding: 两份代码都先用 `torch.outer(t, freqs)` 算出每个位置、每个频率的角度 $t\theta_i$（float32），再用 `torch.polar` 得到 $\cos t\theta_i+\mathrm i\sin t\theta_i$ 的表；DeepSeek-V3 的表只覆盖 64 维的 qk_rope_head_dim，并在 max_seq_len 超过 original_seq_len 时先按 YaRN 修改 freqs。
  matters: 本课 §1 的浮点失败模式说明了为什么要这样写：每个位置的角度直接算出，误差不随位置累积，没有把一个小旋转连乘 $t$ 次。
```
