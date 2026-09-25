---
title: M07 · 优化器：从 SGD 到 AdamW 再到 Muon
short: 从 SGD 到 Muon
kicker: 数学补课 · M07
description: 在二次函数上算清梯度下降的步长与条件数，推导 SGD 的噪声地板、重球动量的最优参数、Adam 的偏差修正、AdamW 与 L2 的差别，讲清 warmup/cosine/多步/WSD 调度、batch 与学习率的缩放、梯度裁剪，最后用 SVD 与极分解讲 Muon 的正交化与牛顿–舒尔茨迭代（含 DeepSeek-V4 的 8+2 步），配手算例题与练习。
lede: 大模型的训练配方里总有一串固定的名词：AdamW、$\beta_2=0.95$、权重衰减 0.1、warmup、cosine、梯度裁剪 1.0，到了 DeepSeek-V4 又多了 Muon 和“牛顿–舒尔茨迭代”。这一课从最简单的梯度下降出发，在一个能手算的二次函数上把**步长、噪声、动量、逐坐标缩放、权重衰减、调度**逐个推导清楚，最后讲 Muon：把更新矩阵“正交化”是什么意思，怎样只用矩阵乘法做到，以及它为什么可能更好：哪些是定理，哪些只是观察。
meta: 预计阅读 150 分钟，练习另需约 50 分钟 · 前置：M06（梯度）；§6 用到 M02 的 SVD，§2 用到 M04 的方差 · 服务于第 2、6、18 课
index: index.html
prev: m06-calculus-backprop.html | M06 · 反向传播
next: m08-floating-point.html | M08 · 浮点与量化
---

## 这一课回答什么问题

[第 2 课](../deepseek/lesson-02-gpt-training.html)写下 GPT 的训练配方：AdamW、warmup 加 cosine 衰减、梯度裁剪。[第 6 课](../deepseek/lesson-06-deepseek-llm.html)的 DeepSeek LLM 把 cosine 换成“多步”调度，并拟合出最优学习率与 batch size 随算力变化的幂律。[第 18 课](../deepseek/lesson-18-v4.html)的 DeepSeek-V4 把大多数权重矩阵交给了一个叫 Muon 的新优化器。这些配方背后是同一套数学。读完这一课，下面这些问题你都能自己回答：

- 学习率为什么不能随便加大？一步能走多远，由损失曲面的什么性质决定？
- 随机梯度的噪声为什么会让损失停在一个“地板”上？地板有多高？为什么学习率一降，损失就陡降？
- 动量为什么能加速？能加速多少？动量太大会发生什么？
- Adam 为什么要除以 $1-\beta^t$？AdamW 的“解耦权重衰减”和往损失里加 L2 项，差别到底在哪里？
- warmup、cosine、多步、WSD 各在解决什么问题？batch 翻倍时学习率该怎么变？梯度裁剪裁掉的是什么？
- Muon 把更新矩阵“正交化”：这是什么运算？牛顿–舒尔茨迭代怎样只靠矩阵乘法做到它？DeepSeek-V4 为什么在 8 步之后又加了 2 步？

## 这一课在科技树上的位置

这一课服务于三课，对应课程地图里的主线 F（规模化规律与优化器）。[第 2 课](../deepseek/lesson-02-gpt-training.html)用到 §1–§5：AdamW 的每一行、偏差修正、warmup + cosine、梯度裁剪。[第 6 课](../deepseek/lesson-06-deepseek-llm.html)用到 §2 与 §5：多步调度为什么一降档损失就陡降，batch size 与学习率为什么要一起随算力变。[第 18 课](../deepseek/lesson-18-v4.html)用到 §6–§7：Muon 的正交化、牛顿–舒尔茨迭代、更新 RMS 的缩放，以及 DeepSeek-V4 哪些参数用 Muon、哪些仍用 AdamW。前一课 [M06](m06-calculus-backprop.html) 讲梯度怎样算出来，这一课讲拿到梯度之后怎样走这一步。

## 读完你能……

1. 在二次函数上算出梯度下降的稳定步长上限 $2/h_{\max}$、最优步长与收敛因子 $(\kappa-1)/(\kappa+1)$，解释条件数怎样卡住训练速度。
2. 推出 SGD 的噪声地板约为 $\eta\sigma^2/(4B)$，用它解释学习率衰减、多步调度的“降档陡降”和 batch 与学习率的缩放。
3. 推导重球动量的最优参数和约 $\sqrt\kappa$ 倍的加速，并说出动量过大时轨迹长什么样。
4. 逐行推导 Adam 的偏差修正；说清 L2 正则与解耦权重衰减只在 SGD 下等价，以及 PyTorch 与 AdamW 论文在“衰减乘不乘学习率”上的差别。
5. 用 SVD 写出正交化 $UV^\top$ 与极分解，手算一步牛顿–舒尔茨迭代，解释 Jordan 系数与 DeepSeek-V4 的“8+2 步”，并分清关于 Muon 的“已证明 / 实测 / 推测”。

::: note 本课的记号
- 参数 $\theta\in\R^{N}$：模型的全部参数排成一列，$N$ 个数。损失 $\mathcal L(\theta)$，梯度 $\mathbf g=\nabla\mathcal L(\theta)\in\R^N$（和参数一样长）。
- 学习率 $\eta>0$，步数 $t$（讲调度时按惯例写 $s$）。收敛因子 $\rho$：误差每一步被乘上的数，$\rho<1$ 才收敛。
- $T$ 在这一课里只表示迭代的总步数（§2 的训练步数、§7 的牛顿–舒尔茨迭代步数），不是整门课里的序列长度。
- 二次模型的 Hessian 写 $H$，它的特征值 $h_i$ 叫“曲率”，条件数 $\kappa=h_{\max}/h_{\min}$。
- $\Sigma$ 有两个用法，按上下文区分：§2、§5 里是单个样本梯度的协方差矩阵，§6 起是 SVD 里放奇异值的对角矩阵。同样，$\sigma^2$ 在 §2 是单个样本梯度的方差，$\sigma_i$ 在 §6 起是奇异值。$\eta_{\max}$ 只指学习率调度的峰值；§5 里“batch 无限大时的最优步长”另记作 $\eta_\infty$。
- $S$ 在 §5 是总步数（调度的计划步数、McCandlish 模型里的步数），在 §6 的极分解里是对称的“伸缩”因子矩阵。
- 动量系数 $\mu$；Adam 的两个衰减率带下标写 $\beta_1,\beta_2$；Adam 的数值常数 $\epsilon$；权重衰减系数 $\lambda$。不带下标的 $\beta$ 在整门课里是 KL 系数；本课不出现 KL，只在泛指“某个滑动平均的衰减率”时（§3 的注、§4 的偏差修正 $1-\beta^t$）借用它。
- 讲 Muon 时参数是矩阵 $W\in\R^{n\times m}$，它的梯度 $\mathbf G_t$、动量 $\mathbf M_t$ 用粗体大写，和第 6 课表示“每 token 非嵌入 FLOPs”的标量 $M$ 区分。
:::

[[legend]]

[[toc]]

# 梯度下降：步长与条件数 | 在一个能手算的二次函数上，把“一步能走多远”算清楚 | 20

训练的每一步都在做同一件事：用当前参数算出梯度，再沿梯度的反方向挪一小步。这就是**梯度下降**（gradient descent，GD）：

::: eq 梯度下降 | (M7.1)
$$\theta_{t+1}=\theta_t-\eta\,\nabla\mathcal L(\theta_t)$$
$\theta_t$ 是第 $t$ 步的参数；$\nabla\mathcal L(\theta_t)$ 指向损失上升最快的方向（[M06 §2](m06-calculus-backprop.html#p2)），所以减去它是往下走；$\eta$ 决定走多远。
:::

整页只回答一个问题：**$\eta$ 能取多大？** 直觉上步子越大越快，可是步子太大会跨过谷底、冲到对面更高的地方。要把“太大”说成一个数，需要一个能精确计算的模型。最合适的就是二次函数：任何光滑函数在极小点附近都长得像它。

## 为什么是二次函数

::: pre 二阶泰勒展开、Hessian 与特征分解 | 学过多元微积分与对称矩阵对角化的可以跳过
一元函数在极小点 $x^*$ 附近有 $f(x^*+\delta)\approx f(x^*)+f'(x^*)\delta+\tfrac12f''(x^*)\delta^2$，而极小点处 $f'(x^*)=0$，所以附近的形状由二阶导数 $f''$ 决定：它越大，碗越陡。多元时二阶导数变成一个 $N\times N$ 的对称矩阵，叫 **Hessian**：$H_{ij}=\partial^2\mathcal L/\partial\theta_i\partial\theta_j$，展开式是 $\mathcal L(\theta^*+\boldsymbol\delta)\approx\mathcal L(\theta^*)+\tfrac12\boldsymbol\delta^\top H\boldsymbol\delta$。

对称矩阵总可以写成 $H=Q\,\diag(h_1,\dots,h_N)\,Q^\top$，$Q$ 是正交矩阵（列是互相垂直的单位向量 $\mathbf q_i$，[M03 §1](m03-rotation-complex.html#p1)），$h_i$ 是特征值。换到坐标 $z_i=\mathbf q_i^\top\boldsymbol\delta$ 下，二次型变成 $\tfrac12\sum_ih_iz_i^2$：$N$ 个互不干扰的一维抛物线。$h_i$ 是沿方向 $\mathbf q_i$ 的**曲率**。极小点处 $H$ 半正定，所有 $h_i\ge0$。
:::

所以下面研究 $\mathcal L(\theta)=\tfrac12\theta^\top H\theta$（把极小点平移到原点，常数项去掉）。它的梯度是 $\nabla\mathcal L=H\theta$。这一页还假设 $H$ **正定**，即所有 $h_i>0$：有 $h_i=0$ 的平坦方向时，那个方向上梯度恒为零、误差永远不动，下面的条件数也会变成无穷大。

**含义。** **条件数**（condition number）$\kappa=h_{\max}/h_{\min}$：最陡方向的曲率是最平方向的多少倍。

**理解。** 等高线是椭圆，$\kappa$ 是长轴与短轴长度之比的平方。$\kappa=1$ 时等高线是圆，梯度直指圆心；$\kappa$ 很大时是一道又窄又长的山谷，梯度几乎垂直于谷底，指向对面的谷壁，而不是谷的出口。

**数学。** 把 (M7.1) 代入二次模型，换到特征坐标，每个方向各走各的：

::: deriv 梯度下降在二次函数上的精确解
$\theta_{t+1}=\theta_t-\eta H\theta_t=(I-\eta H)\theta_t$ || 梯度是 $H\theta_t$
$Q^\top\theta_{t+1}=(I-\eta\,\diag(h))\,Q^\top\theta_t$ || 两边左乘 $Q^\top$，在 $H$ 与 $\theta_t$ 之间插入 $QQ^\top=I$，再用 $Q^\top HQ=\diag(h)$
$z_{i,t+1}=(1-\eta h_i)\,z_{i,t}$ || 对角矩阵乘向量：每个坐标单独乘一个数
$z_{i,t}=(1-\eta h_i)^t\,z_{i,0}$ || 递推 $t$ 次
:::

::: eq 每个特征方向的收敛因子 | (M7.2)
$$z_{i,t}=(1-\eta h_i)^t\,z_{i,0}$$
沿第 $i$ 个方向，误差每步乘 $1-\eta h_i$。它的绝对值决定快慢，符号决定是单调逼近（正）还是左右来回（负）。
:::

## 稳定的上限：$\eta<2/h_{\max}$

(M7.2) 收敛当且仅当每个方向都有 $\abs{1-\eta h_i}<1$，即 $0<\eta h_i<2$。最苛刻的是曲率最大的方向：

::: eq 稳定条件 | (M7.3)
$$0<\eta<\frac{2}{h_{\max}}$$
超过这个值，最陡的方向每步把误差乘上一个绝对值大于 1 的负数：来回翻转，越翻越远。
:::

按 $\eta h$ 的大小，一个方向上有四种行为：$\eta h<1$ 单调逼近；$\eta h=1$ 一步到底；$1<\eta h<2$ 来回摆动但幅度缩小；$\eta h>2$ 来回摆动且幅度放大（发散）。

::: ex 一维，曲率 $h=4$，四个步长
稳定上限是 $2/h=0.5$。取 $x_0=1$，看 10 步后的 $x_{10}=(1-4\eta)^{10}$：

| $\eta$ | $1-4\eta$ | 行为 | $x_{10}$ |
|---|---|---|---|
| 0.1 | 0.6 | 单调逼近 | $0.6^{10}=0.00605$ |
| 0.25 | 0 | 一步到底 | 0 |
| 0.45 | $-0.8$ | 摆动收敛 | $0.8^{10}=0.107$ |
| 0.55 | $-1.2$ | 摆动发散 | $1.2^{10}=6.19$ |

$\eta=0.45$ 比 $\eta=0.1$ 大 4.5 倍，10 步后的误差却大了 18 倍：步子大不等于快。
:::

## 最优步长与收敛因子 $(\kappa-1)/(\kappa+1)$

整体的收敛速度由最慢的方向决定，也就是 $\max_i\abs{1-\eta h_i}$。作为 $h$ 的函数，$\abs{1-\eta h}$ 在区间 $[h_{\min},h_{\max}]$ 上的最大值只可能出现在两端，所以要选 $\eta$ 让两端一样大：

::: deriv 最优步长
$\rho(\eta)=\max\big(\abs{1-\eta h_{\min}},\ \abs{1-\eta h_{\max}}\big)$ || $1-\eta h$ 对 $h$ 是直线，绝对值的最大值在区间端点
$1-\eta h_{\min}=\eta h_{\max}-1$ || 增大 $\eta$ 让左项变小、右项变大；两者相等时最大值最小
$\eta^*=\dfrac{2}{h_{\min}+h_{\max}}$ || 解上一行
$\rho^*=1-\dfrac{2h_{\min}}{h_{\min}+h_{\max}}=\dfrac{h_{\max}-h_{\min}}{h_{\max}+h_{\min}}=\dfrac{\kappa-1}{\kappa+1}$ || 代回，上下同除以 $h_{\min}$
:::

::: eq 梯度下降的最优收敛因子 | (M7.4)
$$\eta^*=\frac{2}{h_{\min}+h_{\max}},\qquad \rho^*=\frac{\kappa-1}{\kappa+1}\approx1-\frac{2}{\kappa}$$
误差缩小到原来的 $\varepsilon_0$ 倍需要约 $\ln(1/\varepsilon_0)/(1-\rho^*)\approx\tfrac{\kappa}{2}\ln(1/\varepsilon_0)$ 步：**步数与条件数成正比**。
:::

把 $\abs{1-\eta h}$ 看成曲率 $h$ 的函数画出来，它是一个 V 形折线，尖端在 $h=1/\eta$。曲率区间 $[h_{\min},h_{\max}]$ 里最坏的方向一定在区间的某一端：$\eta$ 偏小时尖端偏右（甚至跑到区间外面），左端（平缓方向）最高；$\eta$ 偏大时尖端偏左，右端（陡峭方向）最高，超过 1 就发散。最优步长让两端一样高。§3 的图 M7.3 左图把这三种情形画在了一起。

::: ex 条件数的账单
把误差缩小到 $10^{-6}$，要让 $(\rho^*)^t\le10^{-6}$，即 $t\ge\ln10^{-6}/\ln\rho^*$，取不小于它的整数（本课的步数都这样向上取整）：

- $\kappa=10$：$\rho^*=9/11=0.818$，69 步。
- $\kappa=100$：$\rho^*=99/101=0.9802$，691 步。
- $\kappa=10^4$：$\rho^*=9999/10001=0.99980$，69,078 步（$\rho^*$ 离 1 太近，验算时要用分数或足够多的位数，只取 0.9998 会算成 69,071）。

条件数每乘 10，步数乘 10。步长已经是最优的了，慢的原因是最平的方向每步只挪 $\eta^*h_{\min}\approx2/\kappa$ 那么一点，而 $\eta$ 又不敢再大，因为最陡的方向会炸。
:::

::: fig figures/m07-gd-quadratic.svg | 图 M7.1 同一个病态二次函数，三种步长
**先看右图三条线的斜率：绿线（最优步长 2/11）每步把损失乘 0.669，蓝线（步长太小）下降得慢，红线（步长 0.21，超过 $2/h_{\max}=0.2$）每步乘 1.21，一路发散。** 左图是前几步的位置。函数是 $f=\tfrac12(\theta_1^2+10\theta_2^2)$，$\kappa=10$，起点 $(-4,1.5)$。绿线在陡峭的 $\theta_2$ 方向上来回跳（因子 $-0.818$），在平缓的 $\theta_1$ 方向上稳步前进（因子 $+0.818$）；红线在 $\theta_1$ 方向其实是收敛的（因子 0.79），但 $\theta_2$ 方向的因子是 $-1.1$，一个方向炸掉，整体就炸掉。所有点由脚本按 (M7.2) 计算。
:::

::: ex 手算图 M7.1 的第一步
$H=\diag(1,10)$，$\theta_0=(-4,1.5)$，$f(\theta_0)=\tfrac12(16+10\times2.25)=19.25$。取 $\eta=2/11$：两个方向的因子是 $1-2/11=0.818$ 与 $1-20/11=-0.818$，

$$\theta_1=(0.818\times(-4),\ -0.818\times1.5)=(-3.273,\ -1.227),$$

$$f(\theta_1)=\tfrac12(10.71+10\times1.506)=12.89.$$

两个因子的绝对值相同，所以 $f$ 每步恰好乘 $0.818^2=0.669$，这就是右图绿线的斜率。
:::

::: warn 这个模型在哪里失效
真实的损失曲面不是一个固定的二次函数：Hessian 随训练变化，而且 $N$ 有几十亿维，$h_{\max}$ 与 $h_{\min}$ 没人算得出来。这一页给的是**定性**的结论，读者应带走三件事：（1）最大的稳定步长由最陡的方向决定；（2）进展的速度由最平的方向决定；（3）两者之比 $\kappa$ 越大越难办。后面几种改进都和“不同方向尺度悬殊”有关：动量（§3）在二次函数上把步数从正比于 $\kappa$ 降到正比于 $\sqrt\kappa$（§3 给出推导）；Adam（§4）给每个坐标配自己的步长；Muon（§6）把更新矩阵在各个方向上的大小拉平（后两者与条件数的关系是启发式的，§4、§7 会说清哪些有证明）。
:::

::: key 本页要点
- 在二次函数上，梯度下降沿每个特征方向把误差乘 $1-\eta h_i$ (M7.2)；稳定要求 $\eta<2/h_{\max}$ (M7.3)。
- 最优步长 $2/(h_{\min}+h_{\max})$，收敛因子 $(\kappa-1)/(\kappa+1)$，步数与条件数 $\kappa$ 成正比 (M7.4)。
- 步长太大不是“走得快”，而是在最陡的方向上来回翻、越翻越大。
:::

# 随机梯度：噪声地板与学习率衰减 | 为什么降学习率损失会陡降，batch 大小怎样进入公式 | 25

§1 假设每一步都拿到精确的梯度。真实训练里，梯度只在一个 batch 上算：即使是 DeepSeek-V4.1-Flash 每步 1.006 亿个 token 的大 batch，也只是它 45 万亿 token 训练量的约 45 万分之一 [@deepseekai2026v41]。于是每一步的方向都带着噪声。这一页回答：**噪声会让训练停在哪里？学习率和 batch 大小怎样决定这个位置？**

## 随机梯度：无偏，但有方差

**含义。** **随机梯度下降**（stochastic gradient descent，SGD）：用一小批样本的平均梯度代替全体数据的梯度。

**理解。** 每个样本的梯度都是“真梯度加一点噪声”。取 $B$ 个样本平均，噪声互相抵消一部分，方向更准，但永远不会完全准。

**数学。** 记 $\ell(x;\theta)$ 为单个样本 $x$ 上的损失（一个数），第 $i$ 个样本的梯度 $\nabla\ell(x_i;\theta)\in\R^N$。样本随机抽取时，它的期望是真梯度 $\mathbf g=\nabla\mathcal L(\theta)$，协方差矩阵为 $\Sigma\in\R^{N\times N}$（衡量单个样本梯度围绕真梯度的散布）。一个 batch 的平均梯度 $\hat{\mathbf g}=\frac1B\sum_{i=1}^B\nabla\ell(x_i;\theta)$ 满足：

::: eq batch 梯度的均值与协方差 | (M7.5)
$$\E[\hat{\mathbf g}]=\mathbf g,\qquad \Cov[\hat{\mathbf g}]=\frac{\Sigma}{B}$$
样本独立同分布时成立。均值不变（无偏）；方差按 $1/B$ 缩小（独立变量平均的方差，见 [M04 §3](m04-probability.html#p3)），标准差按 $1/\sqrt B$ 缩小。
:::

::: ana 类比：在晃动的山坡上蒙眼下山
**映射。** 参数 = 你站的位置；损失 = 海拔；梯度 = 脚底感到的坡度；学习率 = 步幅；batch 的噪声 = 地面在晃，你每次感到的坡度都有误差；batch 越大 = 多踩几下取平均，感觉越准。

**收益。** 它让本页的结论变得自然：离谷底很远时，坡度大、晃动相对小，大步走没问题；到了谷底附近，真实坡度几乎为零，你感到的“坡度”主要是晃动，步幅越大，被晃得离谷底越远。所以要走到底，最后必须收小步幅。

**失效的地方。** 真实的“山”有几十亿个方向，而且每换一个 batch，地形本身都略有不同；类比里的“晃动”是加在一个固定地形上的，这是简化。§3 会给这个人加上惯性（动量），§4 让他在每个方向上用不同的步幅，这两样类比都能接得上；但类比不能说明 Muon 那种“矩阵层面”的操作。
:::

## 噪声地板：它有多高

有噪声时，损失降到某个高度就不再往下走，只在那里上下抖动。本课把这个高度叫**噪声地板**（noise floor）。用 §1 的办法，在一维二次函数 $f(x)=\tfrac12hx^2$ 上把噪声加进去：每一步的梯度是 $hx_t+\xi_t$，噪声 $\xi_t$ 各步独立、均值 0、方差 $\sigma^2/B$（$\sigma^2$ 是单个样本梯度的方差）。

::: deriv 带噪声的梯度下降：均值与方差的递推
$x_{t+1}=x_t-\eta(hx_t+\xi_t)=(1-\eta h)x_t-\eta\xi_t$ || 代入 (M7.1)
$\E[x_{t+1}]=(1-\eta h)\,\E[x_t]$ || $\E[\xi_t]=0$：均值的行为和无噪声时一模一样
$\Var[x_{t+1}]=(1-\eta h)^2\Var[x_t]+\eta^2\sigma^2/B$ || $\xi_t$ 与 $x_t$ 独立（$x_t$ 只依赖更早的噪声），独立变量之和的方差相加
$v_\infty=(1-\eta h)^2v_\infty+\eta^2\sigma^2/B$ || 稳定时方差不再变化，记为 $v_\infty$（需要 $\abs{1-\eta h}<1$）
$v_\infty=\dfrac{\eta^2\sigma^2}{B\,\big(1-(1-\eta h)^2\big)}=\dfrac{\eta\,\sigma^2}{B\,h\,(2-\eta h)}$ || 解出 $v_\infty$；$1-(1-\eta h)^2=\eta h(2-\eta h)$
$\E[f(x_\infty)]=\tfrac12h\big(\E[x_\infty]^2+v_\infty\big)=\dfrac{\eta\,\sigma^2}{2B\,(2-\eta h)}$ || 均值趋于 0，$\E[x^2]=\E[x]^2+\Var[x]$
:::

::: eq 噪声地板 | (M7.6)
$$\E[f]-f^*\ \longrightarrow\ \frac{\eta\,\sigma^2}{2B\,(2-\eta h)}\ \approx\ \frac{\eta\,\sigma^2}{4B}\qquad(\eta h\ll1)$$
损失降到这个高度就不再降，只在它附近抖动。它**正比于 $\eta/B$**，而且在 $\eta h\ll1$ 时与曲率 $h$ 几乎无关。
:::

这个公式把前面类比里的直觉变成了数：学习率减半，地板减半；batch 翻倍，地板也减半。另一方面，均值以因子 $1-\eta h$ 逼近谷底，学习率越小，逼近得越慢。**大学习率走得快但停得高，小学习率停得低但走得慢**，这就是要“先大后小”的根本原因。

::: ex 算一个地板，再用模拟核对
取 $h=1$、$\sigma^2=1$、$B=1$：

- $\eta=0.1$：$v_\infty=0.1/(1\times1.9)=0.05263$，地板 $\tfrac12v_\infty=0.02632$；近似式 $\eta\sigma^2/4$ 给出 0.025。
- $\eta=0.01$：$v_\infty=0.01/1.99=0.005025$，地板 $0.002513$。

学习率小 10 倍，地板低 10 倍。用 2 万条独立的链各跑 3000 步（seed 0）统计 $x$ 的方差，得到 0.05265 与 0.005014，与公式一致。
:::

::: fig figures/m07-sgd-noise.svg | 图 M7.2 带噪声的二次函数：学习率决定地板的高度，降档后地板随之下降
**看绿线在第 2000 步的那一下：学习率从 0.1 降到 0.01 之后，100 步内期望损失从 0.0517 掉到 0.0302。** 这里的函数有一个平缓方向（$h=0.02$）和一个陡峭方向（$h=1$），起点 $(10,3)$，噪声 $\sigma^2=1$、$B=1$。蓝线（$\eta=0.1$ 恒定）很快到达自己的地板 0.0513，然后停住；紫线（$\eta=0.01$ 恒定）的地板只有 0.0050，但 3000 步后还在平缓方向上慢慢爬（0.305）；绿线先用大学习率走到蓝线的地板（2000 步前与蓝线重合），再降档：陡峭方向的噪声在几十步内就被消掉（它的弛豫时间约 $1/(2\eta h)=50$ 步），平缓方向的噪声消得慢，所以只降了一部分。浅蓝细线是 $\eta=0.1$ 的一条真实样本轨迹，期望值只是它的平均。粗线按 §2 的递推精确计算，不是拟合。
:::

这正是[第 6 课 §4](../deepseek/lesson-06-deepseek-llm.html#p4)里“为什么一降档损失就陡降”的数学版本：DeepSeek LLM 的多步调度在 80% 与 90% 的训练量处把学习率各乘 $\sqrt{1/10}\approx0.316$ [@dsllm]，每一次降档都把噪声地板按比例压低，于是损失在降档处往下跳。当然，图 M7.2 是二次函数加独立同分布噪声的玩具模型，大模型的损失曲面与噪声都复杂得多；它解释的是方向，不是具体的跳幅。

## Robbins–Monro 条件：学习率要怎样衰减才能收敛到底

如果学习率一直不降，(M7.6) 说损失永远停在地板上。如果降得太快，又可能还没走到谷底就“冻住”了。1951 年 Robbins 与 Monro 研究带噪声观测的求根问题时给出的步长条件，正好刻画了这两头 [@robbins1951; @lai2021sa]：

::: eq Robbins–Monro 条件 | (M7.7)
$$\sum_{t=1}^{\infty}\eta_t=\infty,\qquad \sum_{t=1}^{\infty}\eta_t^2<\infty$$
第一条：步长加起来是无穷大，从多远的地方出发都走得到。第二条：步长的平方和有限，噪声累计的方差有限，最终能停下来。
:::

两条缺一不可，而且都能用本页一维二次函数的递推看清楚（下面是笔者的示意推导，不是原定理的证明）。第一条管均值：学习率随步数变化时，均值每步乘的是 $1-\eta_th$。

::: deriv 第一条：步长之和有限，均值会停在半路
$\E[x_T]=\E[x_0]\prod_{t=1}^{T}(1-\eta_th)$ || 第 $t$ 步用 $\eta_t$：$\E[x_t]=(1-\eta_th)\E[x_{t-1}]$，连乘 $T$ 次；设每步 $0<\eta_th<1$
$\ln\prod_{t}(1-\eta_th)=\sum_{t}\ln(1-\eta_th)\le-h\sum_{t}\eta_t$ || 对数把乘积变成和；$\ln(1-u)\le-u$
$\sum_t\eta_t=\infty\ \Rightarrow\ \prod_t(1-\eta_th)\to0$ || 上一行右边趋于 $-\infty$：从多远出发，均值都回到谷底
$\sum_t\eta_t<\infty\ \Rightarrow\ \prod_t(1-\eta_th)\ge c\,e^{-2h\sum_t\eta_t}>0$ || $0\le u\le\tfrac12$ 时 $\ln(1-u)\ge-2u$；和有限则 $\eta_t\to0$，只有有限几步 $\eta_th>\tfrac12$，它们的因子之积 $c>0$
:::

第二条管方差：噪声对位置的贡献是 $\sum_t\eta_t\xi_t$ 这样的和，它的方差正比于 $\sum_t\eta_t^2$；平方和有限，噪声才不会无休止地把参数推来推去。原始的 Robbins–Monro 定理针对的是一维求根问题，证明的是均方收敛；这里的陈述转述自 Lai 与 Yuan 的综述（原文未读）[@lai2021sa]。

::: ex 四种学习率，谁满足条件
- $\eta_t=1/t$：前 $10^6$ 项的和是 14.39（像 $\ln t$ 一样无限增长），平方和收敛到 $\pi^2/6=1.645$。两条都满足。
- $\eta_t=1/\sqrt t$：前 $10^6$ 项的和约 1998.5，像 $2\sqrt t$ 一样无限增长（满足第一条），平方和就是 $\sum1/t$，也发散：**不满足第二条**。
- $\eta_t=0.01$ 恒定：平方和 $=10^{-4}\times t$，随步数线性增长，不满足第二条。
- $\eta_t=1/t^2$（从 $t=2$ 起）：平方和有限，但和也有限，只有 $\pi^2/6-1=0.645$：**不满足第一条**。取 $h=1$，每步乘 $1-1/t^2=\frac{(t-1)(t+1)}{t\cdot t}$，连乘时相邻项相消，$\prod_{t=2}^{T}(1-1/t^2)=\frac{T+1}{2T}\to\frac12$：步数再多，均值也只走完一半就冻住了。这就是“降得太快”的样子。
:::

::: warn 大模型的调度不在这个定理的保证之内
第 6、18、19 课的调度都有很长一段学习率恒定的阶段，然后降到一个**不为零**的终值（峰值的 10%）；第 2 课 GPT-3 的余弦衰减也停在峰值的 10%，之后保持不变。把这样的调度往后延长，它们都不满足 (M7.7) 的第二条；何况真实训练只有有限步，Robbins–Monro 定理对它们什么也不保证。这个条件在这里的价值是提供直觉：稳定期负责走得远（第一条），衰减期负责压低噪声（第二条的精神）。把它当作 LLM 训练收敛的证明是误用。
:::

::: note 这一页用在哪里
[第 2 课 §5](../deepseek/lesson-02-gpt-training.html#p5)：学习率为什么要衰减。[第 6 课 §4](../deepseek/lesson-06-deepseek-llm.html#p4)：多步调度降档时损失陡降；batch 越大每步越准（$1/B$ 的方差），这是 batch 与学习率要一起随算力调整的起点（本课 §5）。
:::

::: key 本页要点
- batch 平均梯度无偏，协方差为 $\Sigma/B$ (M7.5)。
- 带噪声的梯度下降最终停在噪声地板 $\approx\eta\sigma^2/(4B)$ 上 (M7.6)：地板正比于 $\eta/B$。大学习率走得快、停得高，小学习率停得低、走得慢，所以要先大后小。
- Robbins–Monro 条件 $\sum\eta_t=\infty$、$\sum\eta_t^2<\infty$ (M7.7) 说明了“为什么要衰减”，但大模型的实际调度并不满足它。
:::

# 动量：重球法 | 让过去的步子带着你走，为什么能把 κ 变成 √κ | 25

回到图 M7.1 的那道窄长山谷，梯度下降在那里有两个毛病：在陡峭方向上它每步翻一次面，来回打转；在平缓方向上它每步只挪一点。能不能让“来回翻面”的分量互相抵消，让“一直同向”的分量越积越多？办法是**记住上一步是往哪走的**。

## 定义：两种等价写法

**含义。** **动量**（momentum），又叫**重球法**（heavy-ball method）：每一步除了沿负梯度走，还沿上一步的位移方向再走 $\mu$ 倍。

**理解。** 接着 §2 的类比：下山的人现在推着一个沉重的球。球有惯性，一直同一个方向的坡会让它越滚越快；左右交替的坡推它一下、又拉它一下，净效果很小。**边界**：真的球会冲过谷底再滚回来，重球法也会，动量太大时这种“冲过头”会让收敛变慢（本页最后一节）。

**数学。** Polyak 在 1964 年提出这个方法 [@polyak1964]；Sutskever 等人用统一的记号写出了它和 Nesterov 的变体 [@sutskever2013]。下面两种写法在今天的代码里都常见：

::: eq 重球动量 | (M7.8)
$$\theta_{t+1}=\theta_t-\eta\,\mathbf g_t+\mu\,(\theta_t-\theta_{t-1})\qquad\Longleftrightarrow\qquad \mathbf m_t=\mu\,\mathbf m_{t-1}+\mathbf g_t,\ \ \theta_{t+1}=\theta_t-\eta\,\mathbf m_t$$
$\mathbf g_t=\nabla\mathcal L(\theta_t)$；$\mu\in[0,1)$ 是动量系数，$\mu=0$ 退回梯度下降；$\mathbf m_t\in\R^N$ 是**动量缓冲**，$\mathbf m_0=\mathbf 0$，和参数一样大，是优化器要额外存的状态。
:::

::: deriv 两种写法为什么相同
$\theta_{t+1}-\theta_t=-\eta\,\mathbf m_t=-\eta\,(\mu\,\mathbf m_{t-1}+\mathbf g_t)$ || 缓冲写法的两行合在一起
$=\mu\,(-\eta\,\mathbf m_{t-1})-\eta\,\mathbf g_t=\mu\,(\theta_t-\theta_{t-1})-\eta\,\mathbf g_t$ || 上一步也满足 $\theta_t-\theta_{t-1}=-\eta\,\mathbf m_{t-1}$（第一步约定 $\theta_0=\theta_1$，与 $\mathbf m_0=\mathbf 0$ 一致：起步时没有惯性）
:::

把缓冲展开，$\mathbf m_t=\sum_{k\le t}\mu^{t-k}\mathbf g_k$：动量是过去所有梯度的**指数加权和**，越旧的权重越小。两个极端情形说明了它的作用。梯度恒为 $\mathbf g$ 时，$\mathbf m_t\to\mathbf g/(1-\mu)$，等效步长放大 $1/(1-\mu)$ 倍（$\mu=0.9$ 时 10 倍，$0.95$ 时 20 倍）。梯度每步变号（$+\mathbf g,-\mathbf g,\dots$）时，$\mathbf m_t$ 在 $\pm\mathbf g/(1+\mu)$ 之间摆动，反而比单步的梯度还小。前者是平缓方向上的加速，后者是陡峭方向上的减震。

::: note 第三种写法：指数滑动平均
Adam（§4）与 Muon 的参考代码把动量写成**指数滑动平均**（exponential moving average，EMA）：$\mathbf m_t=\beta\,\mathbf m_{t-1}+(1-\beta)\,\mathbf g_t$。它恰好是缓冲写法（取 $\mu=\beta$）的 $(1-\beta)$ 倍：方向相同，只是尺度差一个常数，相当于学习率乘了 $1-\beta$。对 Muon 这个常数无关紧要：它在正交化之前要除以 Frobenius 范数，两种写法给出同一个更新（§7，笔者的推导）。对 Adam 则不然：Adam 除的是**梯度平方**的滑动平均，不是动量自己的大小，一阶矩必须用 EMA 写法（再配上 §4 的偏差修正）才是梯度的无偏估计；换成缓冲写法，步长会放大 $1/(1-\beta_1)$ 倍。
:::

## 在二次函数上精确求解

和 §1 一样换到特征坐标，每个方向各自满足一个**二阶**递推。二阶递推的标准解法是找特征根。

::: pre 二阶线性递推与特征根 | 学过线性差分方程的可以跳过
形如 $z_{t+1}=p\,z_t-q\,z_{t-1}$（$p,q$ 是常数；下面的重球法里 $p=1+\mu-\eta h$、$q=\mu$）的递推叫二阶线性递推：下一项由前两项决定，所以要给出 $z_0,z_1$ 两个初值。猜 $z_t=r^t$ 代入，两边除以 $r^{t-1}$，得到**特征方程** $r^2-pr+q=0$。它有两个不同的根 $r_1,r_2$ 时，每个满足递推的序列都能写成 $z_t=c_1r_1^t+c_2r_2^t$，常数 $c_1,c_2$ 由两个初值定出。于是 $t$ 大了以后，$\abs{z_t}$ 大约按 $\max(\abs{r_1},\abs{r_2})^t$ 变化：**模较大的那个根就是收敛因子**。

根是一对共轭复数 $r_{1,2}=\abs r\,e^{\pm i\omega}$ 时（复数的模与辐角见 [M03 §3](m03-rotation-complex.html#p3)），实序列的 $c_1,c_2$ 也互为共轭，两项合起来是 $z_t=2\abs{c_1}\,\abs r^t\cos(\omega t+\varphi)$：一边按 $\abs r^t$ 收缩，一边以每步 $\omega$ 弧度的节奏来回摆动。两根相等（$r_1=r_2=r$）时，通解变成 $(c_1+c_2t)\,r^t$，多出一个随 $t$ 线性增长的因子。
:::

::: deriv 重球法在一个特征方向上
$z_{t+1}=z_t-\eta h\,z_t+\mu\,(z_t-z_{t-1})=(1+\mu-\eta h)\,z_t-\mu\,z_{t-1}$ || (M7.8) 的第一种写法，梯度 $=hz_t$
$z_t=r^t\ \Rightarrow\ r^2-(1+\mu-\eta h)\,r+\mu=0$ || 试探解代入，两边除以 $r^{t-1}$
$r_1r_2=\mu,\qquad r_1+r_2=1+\mu-\eta h$ || 韦达定理
$(1+\mu-\eta h)^2<4\mu\ \Rightarrow\ r_{1,2}\ \text{为共轭复数},\ \abs{r_1}=\abs{r_2}=\sqrt\mu$ || 判别式为负；共轭复数模相等，乘积为 $\mu$
$(1-\sqrt\mu)^2<\eta h<(1+\sqrt\mu)^2$ || 把 $\abs{1+\mu-\eta h}<2\sqrt\mu$ 拆成两个不等式
:::

::: eq 重球法的特征方程与“复根区” | (M7.9)
$$r^2-(1+\mu-\eta h)\,r+\mu=0;\qquad (1-\sqrt\mu)^2<\eta h<(1+\sqrt\mu)^2\ \Rightarrow\ \abs r=\sqrt\mu$$
复根区里误差按 $\sqrt\mu$ 的速度收缩，并且**与曲率 $h$ 无关**；复数根意味着轨迹一边收缩一边旋转（来回摆动）。
:::

这一行是加速的关键。梯度下降的收敛因子 $1-\eta h$ 随曲率变化，平缓方向上它接近 1；重球法只要把所有曲率都放进复根区，每个方向就都以同一个 $\sqrt\mu$ 收缩。复根区之外呢？两根变成两个不相等的实数，乘积仍是 $\mu$，所以较大的那个模一定大于 $\sqrt\mu$（两个模相等的不等实根只能是 $\pm r$，和为 0，这时方程是 $r^2+\mu=0$，根又成了复数）。也就是说，对固定的 $\mu$，$\sqrt\mu$ 已经是一个方向上能拿到的最快收缩，复根区正是拿到它的地方。剩下的就是选 $\mu$ 与 $\eta$，让区间 $[h_{\min},h_{\max}]$ 恰好装进复根区，同时让 $\sqrt\mu$ 尽量小：

::: deriv 最优的 μ 与 η
$\eta\,h_{\min}=(1-\sqrt\mu)^2,\qquad \eta\,h_{\max}=(1+\sqrt\mu)^2$ || 复根区的两端对准曲率区间的两端（$\sqrt\mu$ 越小复根区越窄，所以最优时两端都卡住）
$\kappa=\dfrac{h_{\max}}{h_{\min}}=\Big(\dfrac{1+\sqrt\mu}{1-\sqrt\mu}\Big)^2\ \Rightarrow\ \sqrt\kappa=\dfrac{1+\sqrt\mu}{1-\sqrt\mu}$ || 两式相除，开方
$\sqrt{\mu^*}=\dfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$ || 解出 $\sqrt\mu$
$\eta^*=\dfrac{(1+\sqrt{\mu^*})^2}{h_{\max}}=\dfrac{4}{(\sqrt{h_{\max}}+\sqrt{h_{\min}})^2}$ || 代回第一行的右式；$1+\sqrt{\mu^*}=2\sqrt\kappa/(\sqrt\kappa+1)$
:::

::: eq 重球法的最优参数与收敛因子 | (M7.10)
$$\mu^*=\Big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\Big)^2,\qquad \eta^*=\frac{4}{(\sqrt{h_{\max}}+\sqrt{h_{\min}})^2},\qquad \rho^*=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\approx1-\frac{2}{\sqrt\kappa}$$
和 (M7.4) 对比：$\kappa$ 换成了 $\sqrt\kappa$，所需步数从正比于 $\kappa$ 降到正比于 $\sqrt\kappa$。区间端点处两根相等，误差多出一个随 $t$ 线性增长的因子，不改变“每步乘 $\rho^*$”的量级。
:::

上面只在“所有曲率都装进复根区”的选法里求了最优，没有证明别的选法不会更好。笔者对 $\kappa=10$ 与 $100$ 在 $(\mu,\eta)$ 上各做了 400×400 的网格扫描，找到的最小收敛因子是 0.522 与 0.820，与 (M7.10) 的 0.5195、0.8182 只差网格的粗细。

::: fig figures/m07-rate-curves.svg | 图 M7.3 每个方向的收敛因子随曲率变化：梯度下降的 V 形，重球的平台
**先看右图的绿色平台：最优重球在整个曲率区间 $[1,10]$ 上，每个方向都以同一个 0.5195 收缩；同样的区间里，梯度下降（黑色 V 形）在两端高达 0.818。** 左图回到 §1：$\abs{1-\eta h}$ 是 $h$ 的折线，区间里最坏的方向总在某一端。步长太小（蓝，0.05）时最坏在 $h_{\min}=1$，因子 0.95；太大（红，0.21）时最坏在 $h_{\max}=10$，因子 1.1，超过 1 就发散；最优步长 2/11（黑）让两端一样高，这就是 (M7.4) 的推导。右图橙线是 $\mu=0.9$：整个区间都在复根区里，可是平台高达 0.949。绿线在区间两端之外离开复根区，根变成实数，较大的模超过 $\sqrt{\mu^*}$，曲线翘起。函数与图 M7.1 相同（$h_{\min}=1$，$h_{\max}=10$）；所有曲线由脚本按 (M7.2) 与 (M7.9) 逐点计算。
:::

::: ex 同样的条件数，动量省多少步
把误差缩小到 $10^{-6}$（对照 §1 的账单）：

| $\kappa$ | $\mu^*$ | 重球 $\rho^*$ | 重球步数 | 梯度下降步数 |
|---|---|---|---|---|
| 10 | 0.270 | 0.5195 | 22 | 69 |
| 100 | 0.669 | 0.8182 | 69 | 691 |
| $10^4$ | 0.961 | 0.9802 | 691 | 69,078 |

比值约为 $\sqrt\kappa$（3.1、10、100 倍）。重球的步数只按 $\rho^*$ 估计：曲率区间两端的方向上两根相等，误差还带着一个随 $t$ 线性增长的因子 (M7.10)，实际所需的步数会再多几步。再看 $\kappa=10$、$h_{\min}=1$ 时的步长：$\eta^*=4/(1+\sqrt{10})^2=0.231$，**比梯度下降的稳定上限 $2/h_{\max}=0.2$ 还大**。单独用这个步长会发散；有了动量，陡峭方向上的来回翻转被 $-\mu z_{t-1}$ 这一项拉住了。
:::

::: fig figures/m07-momentum.svg | 图 M7.4 同一个二次函数：梯度下降、最优重球、动量过大的重球
**看右图的三种斜率：最优重球（绿）每步乘 0.520，30 步就到 $3\times10^{-7}$；梯度下降（黑）每步乘 0.818；动量过大（橙，$\mu=0.9$）每步只乘 0.949，比不用动量还慢。** 左图是前 25 步的位置。绿线第一步冲得比黑线更远（它的步长 0.231 超过了梯度下降的稳定上限），之后很快被动量拉回谷底；橙线像一个很重的球，冲过极小点、绕一大圈、再冲回来。函数、起点与图 M7.1 相同；所有点由脚本按 (M7.8) 计算。
:::

::: warn 一处常见写法：Sutskever 等人转述的 μ
Sutskever 等人转述 Polyak 的结论时写：取 $\mu=(\sqrt R-1)/(\sqrt R+1)$（$R$ 即条件数）时，迭代次数比最速下降少 $\sqrt R$ 倍 [@sutskever2013]。按上面的推导，$(\sqrt\kappa-1)/(\sqrt\kappa+1)$ 是**收敛因子**，最优的动量系数是它的**平方**。如果真把动量取成 $0.5195$（$\kappa=10$），每个方向的收敛因子至少是 $\sqrt{0.5195}=0.721$（复根的模恰好是它，实根时较大的模更大），无论 $\eta$ 怎么选：仍比梯度下降的 0.818 快，但到 $10^{-6}$ 要 43 步而不是 22 步。这是笔者的推导与数值核对；Polyak 1964 的原文本课没有读到，无法确定这是原文的写法还是转述时的差异。
:::

## Nesterov：先按惯性走，再看坡度

Nesterov 的加速梯度（Nesterov's accelerated gradient，NAG）只改一处：梯度不在当前位置算，而在“按惯性先走一步”之后的位置算。按 Sutskever 等人的写法是 $\mathbf v_{t+1}=\mu\mathbf v_t-\eta\nabla\mathcal L(\theta_t+\mu\mathbf v_t)$，$\theta_{t+1}=\theta_t+\mathbf v_{t+1}$（$\mathbf v$ 是位移）[@sutskever2013]。实现里没人真的多算一次梯度，而是换一个变量：令 $\boldsymbol\phi_t=\theta_t+\mu\mathbf v_t$ 为“按惯性预走一步”的位置，把它当成存下来的参数，于是 $\theta_t=\boldsymbol\phi_t-\mu\mathbf v_t$。

::: deriv 把 Nesterov 改写成“只在存储的参数处求梯度”
$\boldsymbol\phi_{t+1}=\theta_{t+1}+\mu\mathbf v_{t+1}=\theta_t+(1+\mu)\mathbf v_{t+1}$ || $\boldsymbol\phi$ 的定义；$\theta_{t+1}=\theta_t+\mathbf v_{t+1}$
$=\boldsymbol\phi_t-\mu\mathbf v_t+(1+\mu)\mathbf v_{t+1}$ || 代入 $\theta_t=\boldsymbol\phi_t-\mu\mathbf v_t$
$=\boldsymbol\phi_t+\mu\mathbf v_{t+1}-\eta\nabla\mathcal L(\boldsymbol\phi_t)$ || Nesterov 的定义：$\mathbf v_{t+1}-\mu\mathbf v_t=-\eta\nabla\mathcal L(\boldsymbol\phi_t)$
$\boldsymbol\phi_{t+1}=\boldsymbol\phi_t-\eta\big(\nabla\mathcal L(\boldsymbol\phi_t)+\mu\,\mathbf m_{t+1}\big)$ || 令 $\mathbf v=-\eta\mathbf m$，则 $\mathbf m_{t+1}=\mu\mathbf m_t+\nabla\mathcal L(\boldsymbol\phi_t)$
:::

::: eq Nesterov 动量的实现形式 | (M7.11)
$$\mathbf m_t=\mu\,\mathbf m_{t-1}+\mathbf g_t,\qquad \theta_{t+1}=\theta_t-\eta\,\big(\mathbf g_t+\mu\,\mathbf m_t\big)$$
这是上面推导的最后一行，只是把存储的参数 $\boldsymbol\phi$ 改回记作 $\theta$，并把 $\mathbf m$ 的下标整体减 1，让 $\mathbf m_t$ 恰好包含第 $t$ 步的梯度 $\mathbf g_t=\nabla\mathcal L(\theta_t)$。和 (M7.8) 只差最后一步用的方向：不是 $\mathbf m_t$，而是 $\mathbf g_t+\mu\mathbf m_t$，相当于“再往前看一步动量”。Muon 与 DeepSeek-V4 把 $\mu\mathbf M_t+\mathbf G_t$ 送进正交化，用的就是这个形式（§7）。
:::

::: warn 失败模式：动量太大，以及噪声
**动量太大。** $\mu$ 大到让所有曲率都进了复根区之后，收敛因子被钉在 $\sqrt\mu$，再调学习率也没用：$\mu=0.9$ 时每步只乘 0.949（图 M7.4 的橙线）。轨迹表现为绕着极小点打转、冲过头。

**噪声。** 上面的 $\sqrt\kappa$ 加速是无噪声梯度、极小点附近的渐近结论。Sutskever 等人同一节就提醒：理论预测随机梯度下，这种渐近的局部加速优势会丧失；他们接着论证，训练深度网络时更要紧的是进入极小点附近之前的“过渡阶段”，那时各步梯度的下降方向大体一致、没有被噪声淹没，动量在那里仍然有用 [@sutskever2013]。直观地看，动量把大约 $1/(1-\mu)$ 步的梯度加在一起，噪声也跟着被放大到相应的倍数。所以大模型训练里的动量系数通常固定在 0.9–0.95，而不是按一个谁也不知道的 $\kappa$ 去算 $\mu^*$；这时它更像一个跨步平均的“滤波器”（这是笔者对惯例的解读，不是定理）。
:::

::: note 这一页用在哪里
[第 2 课 §5](../deepseek/lesson-02-gpt-training.html#p5)：Adam 的一阶矩就是 EMA 写法的动量。[第 18 课 §5](../deepseek/lesson-18-v4.html#p5)：Muon 的动量缓冲 $\mathbf M_t=\mu\mathbf M_{t-1}+\mathbf G_t$ 与 Nesterov 形式 $\mu\mathbf M_t+\mathbf G_t$，DeepSeek-V4 取 $\mu=0.95$。
:::

::: key 本页要点
- 重球动量 (M7.8)：在负梯度之外再加上 $\mu$ 倍的上一步位移；缓冲 $\mathbf m_t$ 是过去梯度的指数加权和，同向的分量累积、反向的分量抵消。
- 在二次函数上，复根区内误差以 $\sqrt\mu$ 收缩且与曲率无关 (M7.9)；最优 $\mu^*=((\sqrt\kappa-1)/(\sqrt\kappa+1))^2$，收敛因子 $(\sqrt\kappa-1)/(\sqrt\kappa+1)$，步数从 $\propto\kappa$ 降到 $\propto\sqrt\kappa$ (M7.10)。
- Nesterov 的实现形式用 $\mathbf g_t+\mu\mathbf m_t$ 作方向 (M7.11)。动量太大会打转；随机梯度下，极小点附近的渐近加速优势会丧失。
:::

# Adam 与 AdamW：逐坐标步长、偏差修正与解耦权重衰减 | 为什么除以 1 − β 的 t 次方，为什么权重衰减不能写进梯度 | 35

§1 的结论是：所有坐标共用一个 $\eta$，它被最陡的方向卡住，又被最平的方向拖慢。假如 Hessian 是对角的，最理想的做法是给第 $i$ 个坐标配步长 $1/h_i$：每个方向一步到底，条件数就不再起作用。这类“给不同坐标乘不同系数”的做法叫**对角预条件**（diagonal preconditioning）。困难在于，大模型的 $h_i$ 算不出来。Adam 的办法是退而求其次：**用每个坐标梯度的典型大小，代替它的曲率**，来决定这个坐标走多远。

## Adam 的一步

**含义。** Adam（adaptive moment estimation）为每个坐标维护两个指数滑动平均：梯度的（一阶矩）和梯度平方的（二阶矩）；用前者作方向，用后者的平方根作这个坐标自己的尺度。

**理解。** 一阶矩就是上一页 EMA 写法的动量。二阶矩的平方根是“这个坐标的梯度通常有多大”。两者相除得到一个没有量纲的数：梯度一贯同号时接近 $\pm1$，忽正忽负时接近 0。于是每个坐标每步大约走 $\eta$ 那么远，不管它的梯度是 $10^{-6}$ 量级还是 $10^{2}$ 量级。取极端 $\beta_1=\beta_2=0$（并忽略 $\epsilon$），Adam 就变成 $\theta\leftarrow\theta-\eta\,\sign(\mathbf g)$：只看符号、不看大小。

**数学。** Kingma 与 Ba 的算法 1（ICLR 2015 版，arXiv v9）[@kingma2015adam]，所有运算逐坐标进行，$\mathbf g_t^2$ 是逐元素平方：

::: eq Adam | (M7.12)
$$\begin{aligned}\mathbf m_t&=\beta_1\,\mathbf m_{t-1}+(1-\beta_1)\,\mathbf g_t, & \mathbf v_t&=\beta_2\,\mathbf v_{t-1}+(1-\beta_2)\,\mathbf g_t^2,\\ \hat{\mathbf m}_t&=\frac{\mathbf m_t}{1-\beta_1^{\,t}}, & \hat{\mathbf v}_t&=\frac{\mathbf v_t}{1-\beta_2^{\,t}},\qquad \theta_t=\theta_{t-1}-\eta\,\frac{\hat{\mathbf m}_t}{\sqrt{\hat{\mathbf v}_t}+\epsilon}\end{aligned}$$
$\mathbf g_t$ 在 $\theta_{t-1}$ 处计算（Adam 的步数从 $t=1$ 数起，所以写成 $\theta_{t-1}\to\theta_t$）；$\mathbf m_0=\mathbf v_0=\mathbf 0$；$\beta_1,\beta_2\in[0,1)$；$\beta_1^{\,t}$ 是 $t$ 次方。论文的默认值：$\eta=0.001$、$\beta_1=0.9$、$\beta_2=0.999$、$\epsilon=10^{-8}$。
:::

::: warn 版本漂移：Adam 论文 v1 的 β 是另一种写法
arXiv 上 Adam 的第一版（2014 年 12 月）把衰减率写成“$1-\beta$”：$\mathbf m_t\leftarrow\beta_1\mathbf g_t+(1-\beta_1)\mathbf m_{t-1}$，默认 $\beta_1=0.1$、$\beta_2=0.001$、学习率 0.0002；ICLR 2015 版（v9）才是今天通行的 0.9、0.999、0.001 [@kingma2015adam]。引用 Adam 的默认值时要写明版本，否则很容易把 $\beta_1=0.1$ 抄进代码里。
:::

## 偏差修正：为什么除以 $1-\beta^t$

**偏差修正**（bias correction）处理的是起步问题。两个滑动平均都从 0 起步，前几步里，它们还没来得及“攒够”权重，会系统性地偏小。偏多少，可以算出来；Adam 论文 §3 对二阶矩做了这个推导 [@kingma2015adam]，一阶矩完全相同。

::: deriv 二阶矩的期望（Adam 论文 §3）
$\mathbf v_t=(1-\beta_2)\sum_{i=1}^{t}\beta_2^{\,t-i}\,\mathbf g_i^2$ || 把 (M7.12) 的递推从 $\mathbf v_0=\mathbf 0$ 展开 $t$ 次
$\E[\mathbf v_t]=(1-\beta_2)\sum_{i=1}^{t}\beta_2^{\,t-i}\,\E[\mathbf g_i^2]$ || 期望对求和是线性的
$=\E[\mathbf g_t^2]\,(1-\beta_2)\sum_{i=1}^{t}\beta_2^{\,t-i}+\boldsymbol\zeta$ || 把每个 $\E[\mathbf g_i^2]$ 写成 $\E[\mathbf g_t^2]$ 加差值，差值的加权和记为 $\boldsymbol\zeta$
$=\E[\mathbf g_t^2]\,(1-\beta_2^{\,t})+\boldsymbol\zeta$ || 等比数列：$(1-\beta_2)\sum_{j=0}^{t-1}\beta_2^{\,j}=1-\beta_2^{\,t}$
:::

::: eq 滑动平均的偏差 | (M7.13)
$$\E[\mathbf v_t]=(1-\beta_2^{\,t})\,\E[\mathbf g_t^2]+\boldsymbol\zeta,\qquad \E[\mathbf m_t]=(1-\beta_1^{\,t})\,\E[\mathbf g_t]+\boldsymbol\zeta'$$
梯度分布不随时间变时 $\boldsymbol\zeta=\boldsymbol\zeta'=\mathbf 0$；否则，只要衰减率让很久以前的梯度权重足够小，$\boldsymbol\zeta$ 也很小（Adam 论文的论证）。于是除以 $1-\beta^{\,t}$ 就恰好去掉了“从零起步”带来的偏差。
:::

$1-\beta^t$ 是一个滑动平均在第 $t$ 步“攒到”的总权重：权重 $(1-\beta)\beta^{t-i}$ 对 $i=1..t$ 求和。偏差修正就是把这个不足 1 的总权重重新归一化成 1。它对二阶矩特别重要，因为 $\beta_2$ 离 1 很近：

::: ex 修正因子有多小，不修正会怎样
默认 $\beta_1=0.9$、$\beta_2=0.999$：

| $t$ | $1-\beta_1^t$ | $1-\beta_2^t$ | 不修正的步长 ÷ 修正后的步长 $=(1-\beta_1^t)/\sqrt{1-\beta_2^t}$ |
|---|---|---|---|
| 1 | 0.1 | 0.001 | 3.16 |
| 12 | 0.718 | 0.0119 | 6.57（最大） |
| 100 | 1.000 | 0.0952 | 3.24 |
| 1000 | 1.000 | 0.632 | 1.26 |

第 1000 步时二阶矩的修正因子还只有 0.632。拿常数梯度 $g=10^{-3}$ 核对第一步：$m_1=(1-0.9)\times10^{-3}=10^{-4}$，$v_1=0.001\times10^{-6}=10^{-9}$，不修正时 $m_1/\sqrt{v_1}=10^{-4}/3.162\times10^{-5}=3.162$；修正后 $\hat m_1=10^{-3}$、$\hat v_1=10^{-6}$，比值恰为 1。不做修正，训练最开始的几十步会走出几倍于设定值的步子，正是最容易出事的时候。反过来，若 $\beta_2=0.95$（DeepSeek 的取值），第一步的比值是 $0.1/\sqrt{0.05}=0.447$：不修正的话步子反而偏小。
:::

::: fig figures/m07-adam-bias.svg | 图 M7.5 偏差修正因子，以及省掉修正时步长被放大或缩小的倍数
**看右图的红色驼峰：用默认的 $\beta_2=0.999$ 而不做偏差修正，第 12 步的步长是修正后的 6.57 倍，要到上千步才回落到 1 附近。** 左图是三个修正因子 $1-\beta^t$：$\beta_1=0.9$ 约 20 步就接近 1，$\beta_2=0.999$ 要上千步。右图橙线是 $\beta_2=0.95$：一开始偏小（0.447），很快回到 1。所有曲线由脚本逐点计算。
:::

**步长的上界。** 偏差修正之后，Adam 每个坐标每步的位移大致不超过 $\eta$。Adam 论文 §2.1 给出更精确的界（设 $\epsilon=0$）：当 $1-\beta_1>\sqrt{1-\beta_2}$ 时 $\abs{\Delta\theta}\le\eta(1-\beta_1)/\sqrt{1-\beta_2}$，否则 $\abs{\Delta\theta}\le\eta$，前一种情形只在最极端的稀疏梯度下出现 [@kingma2015adam]。默认值下 $(1-\beta_1)/\sqrt{1-\beta_2}=3.16$，落在前一种情形；$\beta_2=0.95$ 时是 0.447，落在后一种。这个“每步大致走 $\eta$ 左右”的性质，和 Adam 对梯度尺度不敏感是同一件事的两面：把所有梯度乘 $c$，$\hat{\mathbf m}_t$ 乘 $c$、$\hat{\mathbf v}_t$ 乘 $c^2$，比值不变 [@kingma2015adam]。§5 讲梯度裁剪时还会用到它。

:::: warn 论文的这两个“上界”并不严格
论文自己的结论写的是“近似有界”（$\abs{\Delta\theta}\lessapprox\eta$）；上面两个数刻画的是两种典型情形：只有当前一步有梯度（最稀疏），以及梯度大致平稳（$\abs{\E[g]}/\sqrt{\E[g^2]}\le1$）。对任意的梯度序列，它们都不是真正的上界。设 $\epsilon=0$、离开始很远（偏差修正因子取 1），单个坐标的确切上确界可以这样算（笔者推导，下标 $k$ 表示往前数第 $k$ 步）：

::: deriv Adam 一步的确切上确界
$m_t=(1-\beta_1)\sum_{k\ge0}a_kb_k,\quad a_k=\big(\beta_1/\sqrt{\beta_2}\big)^k,\ b_k=\beta_2^{\,k/2}g_{t-k}$ || 展开一阶矩的滑动平均，每项拆成两个因子
$m_t^2\le(1-\beta_1)^2\,\textstyle\sum_ka_k^2\,\sum_kb_k^2$ || Cauchy–Schwarz 不等式
$\textstyle\sum_ka_k^2=\dfrac{1}{1-\beta_1^2/\beta_2},\qquad \sum_kb_k^2=\dfrac{v_t}{1-\beta_2}$ || 等比数列，需 $\beta_1^2<\beta_2$；$v_t$ 照样展开
$\dfrac{\abs{m_t}}{\sqrt{v_t}}\le\dfrac{1-\beta_1}{\sqrt{(1-\beta_2)(1-\beta_1^2/\beta_2)}}$ || 代入、开方；$b_k\propto a_k$ 时取等号
:::

默认值 $(0.9,0.999)$ 下右边是 7.27，比论文的 3.16 大一倍多；$\beta_2=0.95$ 时是 1.165，也超过了 1。取到等号的梯度是 $g_{t-k}\propto(\beta_1/\beta_2)^k$：梯度每步放大 $\beta_2/\beta_1$ 倍（默认值约 11%，$\beta_2=0.95$ 时约 5.6%）并持续很久，$m$ 跟得上新的大梯度，$v$ 却被更旧、更小的梯度拖在后面。有限步、做了偏差修正也一样（把上面的无穷级数换成前 $t$ 项再乘修正因子）：$\beta_2=0.95$ 时从第 2 步起上确界就超过 1，默认值下从第 210 步起超过 3.16。这种持续上涨的梯度在训练中并不典型，所以“每步大致不超过 $\eta$”作为经验描述仍然可用，只是它不是定理。
::::

**$\epsilon$ 与状态。** $\epsilon$ 防止除以零，也给“梯度极小的坐标会被放大多少”设了下限。DeepSeek-V4 的 AdamW 把它设成 $10^{-20}$，$\beta_2=0.95$ [@deepseekai2026v4]；这么小的数在 FP16 里会下溢成 0，只能在 FP32（或 BF16）的优化器状态里起作用（数值格式见 [M08 §3](m08-floating-point.html#p3)）。另外，$\mathbf m_t,\mathbf v_t$ 各和参数一样大：假如 1.6 万亿个参数（DeepSeek-V4-Pro 的总参数量 [@deepseekai2026v4]）全用 Adam、状态用 FP32 存，就是 $1.6\times10^{12}\times8$ 字节 $=12.8$ TB（V4 实际只把少数参数交给 AdamW，见 §7）。

## 权重衰减与 L2 正则：在 SGD 里是一回事

**含义。** **权重衰减**（weight decay）：每一步把所有权重按一个小比例往 0 拉。**L2 正则**：往损失里加一项 $\tfrac{\lambda'}2\norm\theta^2$，惩罚大的权重。

**理解。** 两者的动机相同：不让权重无节制地长大。在普通 SGD 里，它们还是同一个操作。

**数学。** Loshchilov 与 Hutter 把 Hanson 与 Pratt 的权重衰减写成 $\theta_{t+1}=(1-\lambda)\theta_t-\eta\nabla f_t(\theta_t)$（他们的式 (1)）[@loshchilov2019adamw]。L2 正则的梯度是 $\lambda'\theta$，走一步：

::: deriv SGD 下，L2 正则 = 权重衰减（AdamW 论文命题 1）
$\theta_{t+1}=\theta_t-\eta\big(\nabla f_t(\theta_t)+\lambda'\theta_t\big)$ || 对 $f_t+\tfrac{\lambda'}2\norm\theta^2$ 做一步 SGD
$=(1-\eta\lambda')\,\theta_t-\eta\nabla f_t(\theta_t)$ || 把 $\theta_t$ 的项合并
$\lambda=\eta\lambda'\quad\text{即}\quad\lambda'=\lambda/\eta$ || 与权重衰减的式子逐项对比
:::

所以在 SGD 里，“往梯度里加 $\lambda'\theta$”就是权重衰减，许多代码库因此把权重衰减实现成前者。注意等价需要 $\lambda'=\lambda/\eta$：两个系数不是同一个数，学习率变了，对应关系也跟着变。

## 在 Adam 里不是一回事

把 L2 的梯度 $\lambda'\theta$ 加进 $\mathbf g_t$ 再交给 Adam，它就要经过 (M7.12) 的除法。

::: deriv Adam + L2 的实际收缩量（笔者按 AdamW 论文 §2 的思路展开）
$\mathbf g_t=\nabla f_t+\lambda'\theta$ || “Adam + L2”：L2 的梯度混进损失的梯度
$\hat{\mathbf m}_t\approx\overline{\nabla f}+\lambda'\theta$ || 设 $\theta$ 变得慢，L2 部分在滑动平均里近似不变；$\overline{\nabla f}$ 是损失梯度的平均
$\Delta\theta\approx-\eta\,\dfrac{\overline{\nabla f}}{\sqrt{\hat{\mathbf v}_t}}-\eta\,\dfrac{\lambda'\theta}{\sqrt{\hat{\mathbf v}_t}}$ || 代入 (M7.12)，忽略 $\epsilon$；第二项是 L2 带来的收缩
$\text{第 }i\text{ 个坐标每步收缩比例}\approx\eta\lambda'/\sqrt{\hat v_{t,i}}$ || 除以的是这个坐标自己的梯度尺度，各坐标不同
:::

::: eq Adam + L2 的有效衰减 | (M7.14)
$$\text{Adam + L2：}\ \ \frac{\abs{\Delta\theta_i}_{\text{L2}}}{\abs{\theta_i}}\approx\frac{\eta\lambda'}{\sqrt{\hat v_{t,i}}},\qquad\text{解耦（AdamW）：}\ \ \frac{\abs{\Delta\theta_i}_{\text{wd}}}{\abs{\theta_i}}=\eta\lambda$$
梯度一贯很大的坐标（$\sqrt{\hat v}$ 大）几乎不被 L2 拉动；梯度很小的坐标反而被拉得很狠。解耦衰减对所有坐标一视同仁。
:::

论文的原话是，梯度大的权重 “do not get regularized as much”，并证明了（命题 2）：只要预条件矩阵不是单位阵的倍数，就不存在任何 $\lambda'$ 能让“自适应方法 + L2”等价于解耦的权重衰减 [@loshchilov2019adamw]。命题 3 从另一面说明两者的关系：在固定的对角预条件 $\diag(\mathbf s)^{-1}$ 下（$\mathbf s$ 的各分量为正，扮演 $\sqrt{\hat{\mathbf v}}$ 的角色），每步衰减 $\lambda$ 的解耦衰减等价于一个按梯度尺度加权的 L2 项 $\frac{\lambda'}{2}\norm{\theta\odot\sqrt{\mathbf s}}^2$（$\lambda'=\lambda/\eta$，同命题 1），也就是**梯度尺度越大的坐标，被惩罚得越重**，恰好与 Adam + L2 相反。验算：这一项的梯度是 $\lambda'\,\mathbf s\odot\theta$，经过预条件变成 $\eta\,\diag(\mathbf s)^{-1}\lambda'\,\mathbf s\odot\theta=\eta\lambda'\theta=\lambda\theta$，正好是每步的衰减量。论文命题 3 的正文把系数印成 $\lambda'/(2\alpha)$，与它附录里的证明（用的是 $\lambda'/2$）不一致，按证明应为 $\lambda'/2$。解决办法就是把衰减从梯度里拿出来，直接作用在参数上：

::: eq AdamW（本课与 PyTorch 的写法） | (M7.15)
$$\theta_t=\theta_{t-1}-\eta_t\left(\frac{\hat{\mathbf m}_t}{\sqrt{\hat{\mathbf v}_t}+\epsilon}+\lambda\,\theta_{t-1}\right)$$
$\hat{\mathbf m}_t,\hat{\mathbf v}_t$ 只用损失的梯度算；$\eta_t$ 是含调度的实际学习率；衰减项每步把每个权重乘 $1-\eta_t\lambda$。
:::

::: fig figures/m07-adamw-decay.svg | 图 M7.6 同一个 λ：Adam + L2 与 AdamW 每步把权重拉回多少（模拟）
**看红点在竖直虚线 $s=\lambda w=0.1$ 处拐弯：往右，梯度越大，红点（Adam + L2）拉得越少，在 $s=1$ 处与绿点（AdamW）相交，$s=10$ 时只剩绿点的约十分之一；往左，红点是一条比绿点高 10 倍的平台。绿点从头到尾排成一条水平线。** 设置：一个标量权重 $w_0=1$，损失梯度是纯噪声 $s\cdot N(0,1)$，$\eta=10^{-3}$、$\lambda=\lambda'=0.1$、默认 $\beta$，走 200 步，每个 $s$ 重复 10 万次（seed 0），统计平均每步 $w$ 减少多少。梯度很小时，L2 项 $\lambda w\approx0.1$ 本身就是 Adam 看到的全部梯度，被归一化成“每步走满 $\eta$”；梯度很大时，它被淹没在 $\sqrt{\hat v}\approx s$ 里。穿过各点的红、绿虚线是 (M7.14) 的粗略预测 $\eta\lambda w/\sqrt{s^2+(\lambda w)^2}$ 与 $\eta\lambda w$，和模拟几乎重合。模拟由脚本完成；最右端几个点的统计误差约 7%。
:::

::: warn “weight_decay = 0.1”是哪一种 0.1
AdamW 论文的算法 2 写的是 $\theta_t\leftarrow\theta_{t-1}-\eta_t\big(\alpha\hat{\mathbf m}_t/(\sqrt{\hat{\mathbf v}_t}+\epsilon)+\lambda\theta_{t-1}\big)$：$\alpha$ 是基础学习率，$\eta_t$ 只是调度的**乘子**，衰减项**不乘 $\alpha$** [@loshchilov2019adamw]。PyTorch 的 `AdamW` 则先做 $\theta\leftarrow\theta-\gamma\lambda\theta$，$\gamma$ 是含调度的学习率，默认 `weight_decay=1e-2` [@pytorch-adamw]，这就是 (M7.15)。同一个 $\lambda$ 在两种写法下每步的收缩量差一个 $\alpha$ 的倍数：论文的 $\lambda$ 大约对应 PyTorch 的“学习率 × weight_decay”。另外，AdamW 论文 v1、v2 的标题是 “Fixing Weight Decay Regularization in Adam”，v3（ICLR 2019）才改成现在的题目 [@loshchilov2019adamw]。
:::

::: ex 权重衰减 0.1 每步拉多少
按 (M7.15)，$\eta=3\times10^{-4}$、$\lambda=0.1$：每步把每个权重乘 $1-3\times10^{-5}=0.99997$。假如学习率恒定、损失的梯度恰好为零，$10^5$ 步后权重变成原来的 $0.99997^{10^5}=0.0498$。实际上损失的梯度会不断把权重推开，衰减与之达成某种平衡；这个数只说明：一股每步只有十万分之三的力，积累十万步后并不弱。DeepSeek LLM 与 DeepSeek-V4 的 AdamW 都用 $\lambda=0.1$、$\beta_1=0.9$、$\beta_2=0.95$ [@dsllm; @deepseekai2026v4]。
:::

::: note 这一页用在哪里
[第 2 课 §5](../deepseek/lesson-02-gpt-training.html#p5)：AdamW 的每一行、偏差修正、L2 与解耦衰减。[第 6 课 §4](../deepseek/lesson-06-deepseek-llm.html#p4)：DeepSeek LLM 的 AdamW 设置。[第 18 课 §5](../deepseek/lesson-18-v4.html#p5)：V4 仍用 AdamW 更新嵌入、预测头、RMSNorm 权重等参数，并把 Muon 的更新缩放到“和 AdamW 一样的 RMS”，以便共用同一套学习率与权重衰减（§7）。
:::

::: key 本页要点
- Adam (M7.12)：一阶矩作方向、二阶矩的平方根作每个坐标的尺度，每个坐标每步大约走 $\eta$，与梯度的量级无关。
- 滑动平均从 0 起步，第 $t$ 步只攒到 $1-\beta^t$ 的权重 (M7.13)；除以它就去掉了偏差。$\beta_2=0.999$ 时不修正，第 12 步的步长会是 6.57 倍。
- SGD 里 L2 正则等于权重衰减（$\lambda'=\lambda/\eta$）；Adam 里 L2 被 $1/\sqrt{\hat v}$ 缩放，梯度大的权重几乎不被正则 (M7.14)。AdamW 把衰减直接作用在参数上 (M7.15)。“衰减 0.1”要问清乘不乘学习率。
:::

# 学习率调度、batch size 与梯度裁剪 | warmup、cosine、多步与 WSD；batch 翻倍时学习率怎么变；裁剪裁掉的是什么 | 30

§2 给出了“先大后小”的理由：大学习率走得快，小学习率停得低。这一页把它变成具体的**学习率调度**（learning-rate schedule，学习率随步数变化的函数），并回答两个紧挨着的问题：batch 变大时学习率该怎么变，以及偶发的巨大梯度怎么挡。

## warmup：开头为什么要慢

**含义。** **warmup**（预热）：训练开始的一段步数里，学习率从很小的值（通常是 0）线性升到峰值 $\eta_{\max}$。

**理解。** 训练刚开始时，参数是随机的，优化器的滑动平均也还没攒够样本，梯度方向变化很快。一个直观的担心是：这时用峰值学习率，§1 那种“某个方向的因子超过 1”的情况更容易出现（这是笔者的直觉说法，不是定理）。Goyal 等人在把 ImageNet 的 batch 扩大到 8k 时明确说明了一种机制：他们的线性缩放规则（下文）依赖“相邻几步的梯度差不多”这一假设，而训练初期网络变化很快，假设不成立，于是用 5 个 epoch 的渐进 warmup 来补救，并发现结果对 warmup 的长度不敏感 [@goyal2017]。[第 2 课 §5](../deepseek/lesson-02-gpt-training.html#p5)还讨论了 Post-LN 与 warmup 的关系。

**数学。** 在前 $s_w$ 步取 $\eta(s)=\eta_{\max}\,s/s_w$。DeepSeek LLM、V4、V4.1 都用 2000 步 warmup，V4 与 V4.1 的报告写明是线性升高 [@dsllm; @deepseekai2026v4; @deepseekai2026v41]。V4.1 的 batch 固定为 1.006 亿 token，2000 步就是 2012 亿 token，只占 45T 训练量的 0.447%（笔者按论文数字算）。

## 衰减的三种形状

**cosine。** 第 2 课的主角：warmup 之后，学习率沿半个余弦波从 $\eta_{\max}$ 平滑地降到 $\eta_{\min}$。

::: eq warmup 之后的 cosine 衰减 | (M7.16)
$$\eta(s)=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\big(1+\cos\pi\tau\big),\qquad \tau=\frac{s-s_w}{S-s_w}\in[0,1]$$
$S$ 是计划的总步数（或总 token 数）。$\tau$ 从 0 到 1 时，$\tfrac12(1+\cos\pi\tau)$ 从 1 平滑降到 0，两头降得慢、中间降得快。
:::

它的问题是**每一步都依赖计划总量 $S$**：训练到一半想加量，已经走过的那段曲线就不对了（[第 6 课 §4](../deepseek/lesson-06-deepseek-llm.html#p4)详细算过）。

**多步（multi-step）。** DeepSeek LLM 的做法：warmup 之后保持峰值，到 80% 训练量乘 $10^{-1/2}\approx0.316$，到 90% 再乘一次，变成峰值的 10% [@dsllm]。前 80% 与总量无关，可以复用。

**WSD（Warmup–Stable–Decay）。** MiniCPM 把这种“长平台 + 短衰减”写成一般形式 [@hu2024minicpm]：

::: eq WSD 调度（MiniCPM 式 (1)，本课记号） | (M7.17)
$$\eta(s)=\begin{cases}\eta_{\max}\,s/s_w, & s<s_w\\ \eta_{\max}, & s_w\le s<s_{\mathrm{dec}}\\ f(s-s_{\mathrm{dec}})\,\eta_{\max}, & s_{\mathrm{dec}}\le s\le S\end{cases}\qquad 0<f\le1\ \text{递减}$$
$s_w$：warmup 结束；$s_{\mathrm{dec}}$：稳定期结束、衰减开始（原文记作 $T$，与本课程的序列长度 $T$ 冲突，所以换了名字）；$S$：总步数。$f$ 的形状可以是指数、线性或 cosine。
:::

MiniCPM 报告了两个观察：衰减阶段损失会骤降；衰减段只占总 token 的约 10% 就够了 [@hu2024minicpm]。按 §2 的模型，前一条正是“噪声地板随学习率下降”的表现。MiniCPM 自己用的是指数衰减，论文把它印成 $f(s-T)=0.5^{(s-S)/T}$，并说 $T$ 取 5000 步；这个式子的记号有问题（衰减期里 $s-S<0$，会让 $f>1$，与式 (1) 要求的 $f\le1$ 矛盾），合理的读法是**半衰期** 5000 步：$f(u)=0.5^{u/5000}$。

::: ex 读懂两份真实的调度
**MiniCPM 的衰减段**（按半衰期读，稳定期峰值 0.01）：进入衰减 $u=1000$ 步时 $\eta=0.01\times0.5^{0.2}=0.00871$，2500 步时 $0.00707$，5000 步时 $0.005$，10000 步时 $0.0025$。

**DeepSeek-V4.1-Flash**：2000 步 warmup 后，学习率 $2.6\times10^{-4}$ 一直保持到第 28T 个 token；28T 到 40T 之间按 cosine 降到 $2.6\times10^{-5}$；40T 到 45T 保持不变 [@deepseekai2026v41]。按 token 算，稳定、衰减、末段各占 $28/45=62.2\%$、$12/45=26.7\%$、$5/45=11.1\%$。衰减段中点 34T 处 $\tau=0.5$，$\eta=2.6\times10^{-5}+\tfrac12(2.34\times10^{-4})=1.43\times10^{-4}$。这是一个衰减段取 cosine 形状的 WSD，衰减段比 MiniCPM 说的“10% 就够”长得多。DeepSeek-V4-Flash 的写法类似：“大部分时间”保持 $2.7\times10^{-4}$，末尾按 cosine 降到 $2.7\times10^{-5}$，共 32T token [@deepseekai2026v4]。
:::

::: fig figures/m07-lr-schedules.svg | 图 M7.7 三种调度放在同一根“训练进度”轴上，以及 MiniCPM 的指数衰减
**看绿色曲线：DeepSeek-V4.1-Flash 在前 62% 的训练里一直停在峰值，比 cosine（灰色虚线）高出一大截，然后在浅绿色的衰减带里降到 10%。** 蓝色台阶是 DeepSeek LLM 的多步调度 [@dsllm]；灰色虚线是降到 10% 的通用 cosine。三者都在训练末尾把学习率降到峰值的 10%，差别在于“什么时候降、降得多急”。插图放大了 V4.1 的 warmup：2000 步只占 0.447%，按论文写的线性升高画。右图是 MiniCPM 衰减段的 $0.01\times0.5^{u/5000}$。曲线由脚本按 (M7.16)、(M7.17) 与论文给出的数字计算 [@deepseekai2026v41; @hu2024minicpm]。
:::

## batch 翻倍，学习率怎么变

### 线性缩放规则

Goyal 等人的规则只有一句：batch 乘以 $k$，学习率也乘以 $k$，其余超参数不变 [@goyal2017]。它的推导值得一看，因为它同时说明了规则在哪里会失效。

::: deriv 线性缩放规则（Goyal 等人式 (3)(4)）
$\theta_{t+k}=\theta_t-\eta\,\frac1n\sum_{j<k}\sum_{x\in\mathcal B_j}\nabla\ell(x,\theta_{t+j})$ || 小 batch：用 $k$ 个大小为 $n$ 的 batch 各走一步，学习率 $\eta$
$\hat\theta_{t+1}=\theta_t-\hat\eta\,\frac1{kn}\sum_{j<k}\sum_{x\in\mathcal B_j}\nabla\ell(x,\theta_t)$ || 大 batch：把这 $kn$ 个样本合成一个 batch，只走一步，学习率 $\hat\eta$
$\nabla\ell(x,\theta_{t+j})\approx\nabla\ell(x,\theta_t)\ \Rightarrow\ \hat\eta=k\eta\ \text{时}\ \hat\theta_{t+1}\approx\theta_{t+k}$ || 假设这 $k$ 步里梯度变化不大，两个和逐项相同，只差系数 $\eta/n$ 与 $\hat\eta/(kn)$
:::

::: eq 线性缩放规则 | (M7.18)
$$B\to kB\quad\Longrightarrow\quad\eta\to k\eta$$
一步大 batch 顶替 $k$ 步小 batch。条件是这 $k$ 步里梯度几乎不变，所以训练初期（变化快）与 batch 过大时会失效。
:::

作者自己列出了两种失效：训练初期网络变化快（用 warmup 解决）；batch 不能无限增大，超过某个值精度迅速下降，ImageNet 上这个值约为 8k [@goyal2017]。他们的设置是 $\eta=0.1\times kn/256$：batch 256 时 0.1，8192 时 3.2。§2 的噪声地板给了同一规则的另一种读法：地板正比于 $\eta/B$，$\eta$ 与 $B$ 同乘 $k$，地板不变（这是笔者把两处结论放在一起的解读）。注意这条规则是为 SGD 推的，搬到 Adam 类优化器上并不当然成立。

### 梯度噪声尺度：batch 大到多少就不划算了

线性缩放说“batch 翻倍，一步顶两步”，可它不可能一直成立：batch 大到梯度几乎没有噪声之后，再加样本也只是在重复计算同一个方向。McCandlish 等人把“多大算大”变成了一个可以计算的量 [@mccandlish2018]。思路是在二次近似下，问“用 batch 梯度 $\hat{\mathbf g}$ 走一步，最多能让损失降多少”：

::: deriv 最优步长随 batch 的变化（McCandlish 等人式 (2.5)–(2.8)）
$\E\big[\mathcal L(\theta-\eta\hat{\mathbf g})\big]\approx\mathcal L-\eta\,\mathbf g^\top\E[\hat{\mathbf g}]+\tfrac12\eta^2\,\E[\hat{\mathbf g}^\top H\hat{\mathbf g}]$ || 在 $\theta$ 处做二阶泰勒展开再取期望
$\E[\hat{\mathbf g}^\top H\hat{\mathbf g}]=\mathbf g^\top H\mathbf g+\tr\big(H\,\Cov[\hat{\mathbf g}]\big)=\mathbf g^\top H\mathbf g+\tr(H\Sigma)/B$ || 二次型的期望（证明见下）；再用 (M7.5)
$\E[\Delta\mathcal L]=-\eta\norm{\mathbf g}^2+\tfrac12\eta^2\big(\mathbf g^\top H\mathbf g+\tr(H\Sigma)/B\big)$ || 代入 $\E[\hat{\mathbf g}]=\mathbf g$：这是 $\eta$ 的二次函数
$\eta_{\mathrm{opt}}=\dfrac{\norm{\mathbf g}^2}{\mathbf g^\top H\mathbf g+\tr(H\Sigma)/B}=\dfrac{\eta_\infty}{1+\mathcal B_{\mathrm{noise}}/B}$ || 对 $\eta$ 求导令其为零；分子分母同除以 $\mathbf g^\top H\mathbf g$
$\min_\eta\E[\Delta\mathcal L]=-\tfrac12\dfrac{\norm{\mathbf g}^4}{\mathbf g^\top H\mathbf g+\tr(H\Sigma)/B}$ || 代回；$-ax+\tfrac12bx^2$ 的最小值是 $-a^2/(2b)$
$\Delta\mathcal L_{\mathrm{opt}}:=-\min_\eta\E[\Delta\mathcal L]=\dfrac{\Delta\mathcal L_{\max}}{1+\mathcal B_{\mathrm{noise}}/B}$ || 取负号，记为一步的“降幅”；分子分母同除以 $\mathbf g^\top H\mathbf g$
:::

这里 $\eta_\infty=\norm{\mathbf g}^2/\mathbf g^\top H\mathbf g$ 是没有噪声（$B\to\infty$）时的最优步长，论文记作 $\epsilon_{\max}$；$\Delta\mathcal L_{\max}=\tfrac12\norm{\mathbf g}^4/\mathbf g^\top H\mathbf g$ 是没有噪声时一步能降的最多。第 (2) 行的二次型期望这样来：写 $\hat{\mathbf g}=\mathbf g+\boldsymbol\varepsilon$，$\boldsymbol\varepsilon$ 是均值为 0 的噪声，则 $\hat{\mathbf g}^\top H\hat{\mathbf g}=\mathbf g^\top H\mathbf g+2\mathbf g^\top H\boldsymbol\varepsilon+\boldsymbol\varepsilon^\top H\boldsymbol\varepsilon$；中间项的期望是 0；最后一项是一个数，等于它自己的迹，$\boldsymbol\varepsilon^\top H\boldsymbol\varepsilon=\tr(H\boldsymbol\varepsilon\boldsymbol\varepsilon^\top)$，取期望得 $\tr(H\,\Cov[\hat{\mathbf g}])$。

::: eq 梯度噪声尺度 | (M7.19)
$$\mathcal B_{\mathrm{noise}}=\frac{\tr(H\Sigma)}{\mathbf g^\top H\mathbf g},\qquad \eta_{\mathrm{opt}}(B)=\frac{\eta_\infty}{1+\mathcal B_{\mathrm{noise}}/B},\qquad \mathcal B_{\mathrm{simple}}=\frac{\tr(\Sigma)}{\norm{\mathbf g}^2}$$
$\mathcal B_{\mathrm{noise}}$ 是一个样本数：噪声与信号之比。Hessian 是单位阵的倍数时它化为 $\mathcal B_{\mathrm{simple}}$，只需要梯度的方差与均值就能估计。
:::

读法很简单。$B\ll\mathcal B_{\mathrm{noise}}$ 时，$\eta_{\mathrm{opt}}\approx\eta_\infty B/\mathcal B_{\mathrm{noise}}$ 与 $B$ 成正比：**这就是线性缩放规则成立的区间**。$B\gg\mathcal B_{\mathrm{noise}}$ 时，$\eta_{\mathrm{opt}}\to\eta_\infty$，再加 batch 几乎没有收益。把一整段训练的步数 $S$ 与用掉的样本数 $E$ 放在一起，论文推出两者之间是一条双曲线：

::: eq 步数与样本数的权衡（McCandlish 等人式 (2.11)(2.12)） | (M7.20)
$$\Big(\frac{S}{S_{\min}}-1\Big)\Big(\frac{E}{E_{\min}}-1\Big)=1,\qquad \mathcal B_{\mathrm{crit}}=\frac{E_{\min}}{S_{\min}}$$
$S_{\min}$：batch 无限大时最少要走的步数；$E_{\min}$：batch 无限小时最少要用的样本数。用 $B=\mathcal B_{\mathrm{crit}}$ 训练时，步数与样本数都是各自最小值的 2 倍；模型预测 $\mathcal B_{\mathrm{crit}}\approx\mathcal B_{\mathrm{noise}}$。
:::

::: fig figures/m07-batch-noise.svg | 图 M7.8 batch 大小的收益会饱和
**看右图绿线：batch 从 $\mathcal B_{\mathrm{noise}}/10$ 加到 $\mathcal B_{\mathrm{noise}}$，每步收益从 1/11 涨到 1/2；再加 10 倍，只涨到 10/11。** 左图是 (M7.20) 的双曲线：往左上走（小 batch）省样本、费步数，往右下走（大 batch）省步数、费样本，黑点 $(2,2)$ 是临界 batch。两图都按 McCandlish 等人的式 (2.7)、(2.11) 直接计算，横纵轴都是相对量，不依赖任何具体模型 [@mccandlish2018]。
:::

::: ex 三个 batch 的收益
由 (M7.19)，每步最多能降的损失是 $\Delta\mathcal L_{\max}/(1+\mathcal B_{\mathrm{noise}}/B)$：

- $B=\mathcal B_{\mathrm{noise}}/10$：$1/11=0.091$ 倍。batch 加倍，收益几乎加倍。
- $B=\mathcal B_{\mathrm{noise}}$：$0.5$ 倍。
- $B=10\,\mathcal B_{\mathrm{noise}}$：$10/11=0.909$ 倍。batch 再加倍，收益只多几个百分点，样本却多用了一倍。

在临界 batch 上训练，步数和样本数都是最小值的 2 倍：图 M7.8 左图的 $(2,2)$ 点满足 $(2-1)(2-1)=1$。
:::

::: warn 两个都叫“噪声尺度”的量，以及这个模型的假设
Smith 与 Le 也定义了一个“噪声尺度” $g=\eta(N_{\mathrm{data}}/B-1)\approx\eta N_{\mathrm{data}}/B$（$\eta$ 是学习率，$N_{\mathrm{data}}$ 是训练集大小；原文把学习率记作 $\epsilon$），它像一个温度，推出最优 batch 与学习率成正比 [@smith2018bayesian]。它和 $\mathcal B_{\mathrm{noise}}$ 不是同一个量：后者是一个样本数。McCandlish 等人自己列出了模型的几条假设与注意事项，包括只看一步的短视近似、病态条件、用 $\mathcal B_{\mathrm{simple}}$ 代替 $\mathcal B_{\mathrm{noise}}$、以及学习率未调好时的偏差 [@mccandlish2018]。DeepSeek LLM 没有用这个模型：论文说 McCandlish 等人早先的经验结论在他们的初步实验里适用性有限，于是在网格搜索里直接拟合出最优 batch 与学习率随算力的幂律：算力每乘 10，最优 batch 约乘 2.12、最优学习率约乘 0.75（[第 6 课 §6](../deepseek/lesson-06-deepseek-llm.html#p6)）[@dsllm]。DeepSeek-V4-Flash 在训练中把 batch 从较小的值逐步加到 7550 万 token 后保持 [@deepseekai2026v4]，V4.1-Flash 则全程固定 1.006 亿 token [@deepseekai2026v41]。
:::

## 梯度裁剪：给一步的长度设上限

**含义。** **梯度裁剪**（gradient clipping）：如果所有参数的梯度拼成的向量太长，就把它等比例缩短到一个上限。

**理解。** 偶尔某个 batch 会产生异常大的梯度（数据里的一段怪文本、一次数值上的尖峰）。裁剪只改长度、不改方向，把这一步的冲击限制在阈值之内。

**数学。** 把所有参数的梯度拼成 $\mathbf g\in\R^N$，算它的 L2 范数（**全局范数**），超过阈值 $c$ 就整体缩放：

::: eq 按全局范数裁剪 | (M7.21)
$$\mathbf g\ \leftarrow\ \mathbf g\cdot\min\!\Big(1,\ \frac{c}{\norm{\mathbf g}}\Big),\qquad \norm{\mathbf g}=\Big(\sum_{i=1}^{N}g_i^2\Big)^{1/2}$$
$\norm{\mathbf g}\le c$ 时不变；否则整个向量乘同一个系数 $c/\norm{\mathbf g}$。DeepSeek LLM 取 $c=1.0$ [@dsllm]。
:::

::: ex 为什么用全局范数，而不是逐坐标截断
$\mathbf g=(3,4)$，$\norm{\mathbf g}=5$，$c=1$：全局裁剪得 $(0.6,0.8)$，方向仍是与横轴成 $53.1^\circ$。若改成“每个分量各自截到 $[-1,1]$”，得到 $(1,1)$，方向变成 $45^\circ$，梯度的方向被扭曲了。全局裁剪只是把这一步变短。
:::

**已经有 Adam 了，为什么还要裁剪？** §4 说过，Adam 每个坐标每步大约只走 $\eta$，本身就挡住了一部分冲击。但它挡不住尖峰的**余波**（下面是笔者的推导与数值核对，只作示意）。设某个坐标平时的梯度恒为 1，某一步突然来了 100，$\beta_1=0.9$、$\beta_2=0.95$。那一步 $m$ 与 $v$ 都被尖峰主导，归一化步长约为 $(1-\beta_1)\cdot100/\sqrt{(1-\beta_2)\cdot100^2}=0.447$ 倍 $\eta$（连同平时的那份算是 $0.487\eta$），没有超过 §4 说的“每步大致 $\eta$”：这一下 Adam 确实挡住了。

麻烦在后面。尖峰在 $m$ 里按 $0.9^k$ 衰减、在 $v$ 里按 $0.95^k$ 衰减，第 $k$ 步里尖峰贡献的步长约为 $0.447\,(0.9/\sqrt{0.95})^k\,\eta$，第 10 步还有 $0.20\eta$，方向仍是尖峰那一步的方向。更长的影响来自 $v$：它被抬到约 501，要按 $0.95^k$ 慢慢回落，所以正常梯度的步长在第 20 步只有平时的 7%，到第 100 步才恢复到一半。一次尖峰让这个坐标几乎停摆上百步。尖峰若和平时的梯度反号，情况更糟：一阶矩变成 $m_k=1-10.1\times0.9^k$，要到 $k=22$ 才转正，所以前 22 步这个坐标一直在往反方向走。先裁剪，尖峰进入滑动平均之前就被缩到正常尺度，这两种余波都不会发生（图 M7.9）。至于阈值为什么取 1.0，DeepSeek LLM 的论文只给了数值，没有给出理由 [@dsllm]。

::: fig figures/m07-adam-spike.svg | 图 M7.9 一次梯度尖峰在 Adam 里的余波，以及先裁剪会怎样
**先看左图的绿色面积：同号尖峰之后，正常梯度推动的那一份只剩平时的 4%–16%（$k\le50$），第 100 步才回到一半；$k=0$ 到 150 这个坐标一共只走了 62η，平时是 151η。** 黄色面积是还留在一阶矩里的尖峰，按 $0.447\,(0.9/\sqrt{0.95})^k$ 衰减，第 10 步还有 0.20η。右图是反号的尖峰：红线在 $k=0$ 到 21 一直是负的，这个坐标先倒退，到第 50 步累计位移还是 $-1.2\eta$；绿线是先按全局范数裁剪到 $c=1$（单坐标时全局范数就是 $\abs g$）再交给 Adam，只在前 20 步略慢，150 步共走 149η。设置：平时梯度恒为 1，$m=v=1$ 的稳态（离训练开始很远，偏差修正因子已接近 1），$\beta_1=0.9$、$\beta_2=0.95$、$\epsilon=0$。这是笔者的玩具模型，所有点由脚本逐步计算。
:::

::: note 这一页用在哪里
[第 2 课 §5](../deepseek/lesson-02-gpt-training.html#p5)：warmup + cosine、梯度裁剪。[第 6 课 §4、§6](../deepseek/lesson-06-deepseek-llm.html#p4)：多步调度的复用、batch 与学习率的幂律（本页的线性缩放与噪声尺度是它的理论背景）。[第 18 课 §6](../deepseek/lesson-18-v4.html#p6)、[第 19 课 §7](../deepseek/lesson-19-v4-1-flash.html#p7)：V4 与 V4.1 的“长平台 + cosine 衰减”调度与 batch 设置。
:::

::: key 本页要点
- warmup 在最初的一段步数里把学习率从 0 升到峰值；DeepSeek 各代都用 2000 步。
- cosine (M7.16) 依赖计划总量；多步与 WSD (M7.17) 把大部分训练放在与总量无关的平台上，末段集中降学习率，降的时候损失陡降（§2 的噪声地板）。
- 线性缩放 (M7.18)：batch 乘 $k$、学习率乘 $k$，只在 $B\ll\mathcal B_{\mathrm{noise}}$ 时近似成立 (M7.19)；步数与样本数满足双曲线 (M7.20)，临界 batch 处两者都是最小值的 2 倍。
- 梯度裁剪按全局范数等比例缩短 (M7.21)，只改长度不改方向。Adam 挡得住尖峰那一步，挡不住它留在滑动平均里的余波（同号时停摆上百步，反号时先倒退二十多步，图 M7.9）；裁剪在尖峰进入滑动平均之前就把它缩小。
:::

# 正交化：SVD、极分解与牛顿–舒尔茨迭代 | Muon 的核心运算：保留更新矩阵的方向，把所有奇异值变成 1 | 35

前面的优化器都把参数看成一长串互不相干的数。可是 Transformer 里绝大多数参数是**矩阵**：一个线性层 $\mathbf y=W\mathbf x$ 的 $W\in\R^{n\times m}$ 把 $m$ 维输入变成 $n$ 维输出。它的梯度 $\mathbf G$ 也是一个 $n\times m$ 的矩阵，而矩阵有逐元素看不到的结构：它把哪些输入方向送到哪些输出方向，每个方向放大多少。Muon 的做法是：**保留更新矩阵的方向结构，把每个方向的放大倍数都换成 1**。这一页讲清楚这句话的数学，以及怎样只用矩阵乘法把它算出来。

::: pre 奇异值分解（SVD） | 学过 M02 的可以跳过
任何矩阵 $G\in\R^{n\times m}$（秩为 $r$）都可以写成 $G=U\Sigma V^\top=\sum_{i=1}^{r}\sigma_i\mathbf u_i\mathbf v_i^\top$：$V$ 的列 $\mathbf v_i\in\R^m$ 是互相垂直的单位输入方向，$U$ 的列 $\mathbf u_i\in\R^n$ 是互相垂直的单位输出方向，$\sigma_1\ge\cdots\ge\sigma_r>0$ 是**奇异值**。这一课用这种只保留非零奇异值的紧凑形式：$U\in\R^{n\times r}$、$\Sigma=\diag(\sigma_1,\dots,\sigma_r)\in\R^{r\times r}$、$V\in\R^{m\times r}$，$U^\top U=V^\top V=I_r$。读法：$G\mathbf v_i=\sigma_i\mathbf u_i$，矩阵把第 $i$ 个输入方向送到第 $i$ 个输出方向，并放大 $\sigma_i$ 倍。两个范数：**谱范数**（spectral norm）$\norm G_2=\sigma_1$，是“单位输入最多被放大多少”；**Frobenius 范数** $\norm G_F=(\sum_{ij}G_{ij}^2)^{1/2}=(\sum_i\sigma_i^2)^{1/2}$，所以 $\sigma_1\le\norm G_F$。详见 [M02 §3–§4](m02-rank-svd-lowrank.html#p3)。
:::

## 正交化：把奇异值全部换成 1

**含义。** 矩阵 $G$ 的**正交化**（orthogonalization）$\mathrm{Ortho}(G)=UV^\top$：保留 $G$ 的奇异向量，把所有非零奇异值换成 1。

**理解。** $G$ 把单位圆（高维时是单位球）压成一个椭圆，椭圆的各个半轴长就是奇异值。$UV^\top$ 把单位圆送到一个同样大小的圆，只是转了个方向：“哪个输入方向去哪个输出方向”保留了，“每个方向放大多少”被抹平了。它是“最接近 $G$ 的那个正交矩阵”。

**数学。** Jordan 在 Muon 的博客里用的正是这个定义 [@jordan2024muon]：

::: eq 正交化 | (M7.22)
$$\mathrm{Ortho}(G)=\operatorname*{argmin}_{O}\Big\{\norm{O-G}_F:\ O^\top O=I\ \text{或}\ OO^\top=I\Big\}=UV^\top$$
满足 $O^\top O=I$（$n\ge m$，列正交归一）或 $OO^\top=I$（$n<m$，行正交归一）的矩阵叫**半正交**矩阵。$G$ 满秩时极小元唯一 [@bernstein2024anthology]。
:::

::: deriv 为什么最近的正交矩阵是 $UV^\top$（方阵、满秩情形）
$\norm{O-G}_F^2=\norm O_F^2+\norm G_F^2-2\,\tr(O^\top G)$ || 展开平方，$\norm X_F^2=\tr(X^\top X)$
$\norm O_F^2=\tr(O^\top O)=n$ || $O$ 正交：此项固定，只需让 $\tr(O^\top G)$ 最大
$\tr(O^\top G)=\tr(O^\top U\Sigma V^\top)=\tr(\Sigma\,V^\top O^\top U)$ || 迹的循环性质 $\tr(ABC)=\tr(BCA)$
$=\sum_i\sigma_i Z_{ii},\qquad Z=U^\top OV$ || $\Sigma$ 对角；$Z^\top$ 与 $Z$ 的对角元相同
$\abs{Z_{ii}}\le1\ \Rightarrow\ \tr(O^\top G)\le\sum_i\sigma_i$ || $Z$ 是正交矩阵，每列都是单位向量
$\text{等号}\iff Z_{ii}=1\ (\forall i)\iff Z=I\iff O=UV^\top$ || 单位列向量有一个分量是 1，其余只能是 0
:::

长方阵同理：只要把“$Z$ 是正交矩阵”换成“$\abs{\mathbf u_i^\top O\mathbf v_i}\le\norm{O\mathbf v_i}\le1$”（半正交矩阵不会拉长任何向量）。上面的最大值 $\sum_i\sigma_i$ 有个名字，叫**核范数**（nuclear norm），下面还会遇到。

## 极分解：旋转 × 伸缩

把 $G$ 的 SVD 从中间拆开，得到它的**极分解**（polar decomposition）：

::: eq 极分解 | (M7.23)
$$G=\underbrace{(UV^\top)}_{Q\,=\,\mathrm{Ortho}(G)}\ \underbrace{(V\Sigma V^\top)}_{S\,=\,(G^\top G)^{1/2}},\qquad \mathrm{Ortho}(G)=G\,(G^\top G)^{-1/2}$$
$Q\in\R^{n\times m}$ 是（半）正交的“旋转/镜像”，$S\in\R^{m\times m}$ 是对称半正定的“伸缩”（这里 $G$ 满秩，$r=\min(n,m)$）。第二个式子由 $(G^\top G)^{-1/2}=V\Sigma^{-1}V^\top$ 得到，要求 $G^\top G$ 可逆，即 $n\ge m$ 且 $G$ 列满秩（此时 $V$ 是 $m\times m$ 正交矩阵）；$n<m$ 时 $G^\top G$ 奇异，改用 $\mathrm{Ortho}(G)=(GG^\top)^{-1/2}G$。
:::

名字来自复数：$z=e^{i\varphi}\abs z$，一个单位模的“相位”乘一个非负的“模长”（[M03 §3–§4](m03-rotation-complex.html#p3)）。正交化就是矩阵版的 $z/\abs z$：只留相位，丢掉模长。这个类比有两处边界：复数只有一个模长，矩阵有 $r$ 个奇异值，正交化丢掉的是 $r$ 个数之间的相对大小；$z=0$ 时相位没有定义，矩阵有零奇异值（不满秩）时“相位”同样不唯一（练习 10）。对照 §4 的 Adam：取 $\beta_1=\beta_2=0$ 时它的更新是 $\sign(\mathbf g)$，逐个元素“只留符号、丢掉大小”；Muon 做的是同一件事，只不过单位不是元素，而是奇异方向（这个对照是笔者的说法）。

::: ex 一个 2×2 的极分解
$G=\begin{pmatrix}3&0\\4&5\end{pmatrix}$。先求 $G^\top G=\begin{pmatrix}25&20\\20&25\end{pmatrix}$，它的特征值是 45 与 5，特征向量 $\mathbf v_1=(1,1)/\sqrt2$、$\mathbf v_2=(1,-1)/\sqrt2$，所以奇异值 $\sigma_1=\sqrt{45}=3\sqrt5=6.708$，$\sigma_2=\sqrt5=2.236$。输出方向 $\mathbf u_i=G\mathbf v_i/\sigma_i$：$\mathbf u_1=(3,9)/(\sqrt2\cdot3\sqrt5)=(1,3)/\sqrt{10}$，$\mathbf u_2=(3,-1)/\sqrt{10}$。于是

$$Q=\mathbf u_1\mathbf v_1^\top+\mathbf u_2\mathbf v_2^\top=\frac1{\sqrt{20}}\begin{pmatrix}4&-2\\2&4\end{pmatrix}=\frac1{\sqrt5}\begin{pmatrix}2&-1\\1&2\end{pmatrix},\qquad S=\sqrt5\begin{pmatrix}2&1\\1&2\end{pmatrix}.$$

$Q$ 是逆时针旋转 $\arctan(1/2)=26.57^\circ$；验算 $QS=\begin{pmatrix}4-1&2-2\\2+2&1+4\end{pmatrix}=G$。
:::

::: fig figures/m07-polar.svg | 图 M7.10 正交化对一个 2×2 更新矩阵做了什么：单位圆的像
**从左往右看单位圆的像怎样变：$G/\norm G_F$ 把它压成长短轴为 0.949 和 0.316 的细长椭圆；正交化之后，它变回单位圆，只剩一个 26.57° 的旋转。** 蓝、紫箭头是两个输入方向 $\mathbf v_1,\mathbf v_2$ 被送到的位置。第三格是下文牛顿–舒尔茨迭代的第一步：短轴从 0.316 被拉长到 0.945，长轴从 0.949 压到 0.752，两个方向的地位几乎倒了过来；第四格是 DeepSeek-V4 的 8+2 步之后，两个奇异值都在 1.0002–1.0005 之间。所有曲线由脚本对这个矩阵逐步计算。
:::

## 为什么要正交化：谱范数下的最速下降

用梯度下降时，我们默认“步长”按欧氏长度（Frobenius 范数）来量。换一种量法，最好的方向就会变。对一个线性层，更新 $\Delta W$ 对输出的影响满足 $\norm{\Delta W\mathbf x}\le\norm{\Delta W}_2\norm{\mathbf x}$：谱范数正是“这一步最多能把某个输入的输出改变多少”。于是可以问：**在谱范数的意义下，最速下降的方向是什么？**

::: deriv 谱范数下的最速下降（Bernstein–Newhouse 命题 5 的单层情形，笔者逐步展开）
$\min_{\Delta W}\ \tr(G^\top\Delta W)+\tfrac1{2\eta}\norm{\Delta W}_2^2$ || $G,\Delta W\in\R^{n\times m}$；损失的一阶变化 $\langle G,\Delta W\rangle=\tr(G^\top\Delta W)$，加上按谱范数计的步长惩罚
$\tr(G^\top\Delta W)=\sum_i\sigma_i\,\mathbf u_i^\top\Delta W\,\mathbf v_i\ \ge\ -\norm{\Delta W}_2\sum_i\sigma_i$ || 代入 $G=\sum_i\sigma_i\mathbf u_i\mathbf v_i^\top$；单位向量 $\mathbf u,\mathbf v$ 满足 $\abs{\mathbf u^\top\Delta W\mathbf v}\le\norm{\Delta W}_2$
$\norm{\Delta W}_2=\delta\ \text{固定时，下界由}\ \Delta W=-\delta\,UV^\top\ \text{取到}$ || $\mathbf u_i^\top UV^\top\mathbf v_i=1$，且 $\norm{UV^\top}_2=1$；$\delta\ge0$ 是步子的谱范数
$\min_\delta\big(-\delta\textstyle\sum_i\sigma_i+\tfrac{\delta^2}{2\eta}\big)\ \Rightarrow\ \delta=\eta\sum_i\sigma_i$ || 再对步长 $\delta$ 求一元二次函数的极小（这里的 $\delta$ 与秩 $r$ 无关）
:::

::: eq 谱范数下的最速下降方向 | (M7.24)
$$\Delta W^*=-\eta\,\Big(\sum_i\sigma_i\Big)\,UV^\top=-\eta\,\norm G_*\,\mathrm{Ortho}(G)$$
方向恰好是 $-\mathrm{Ortho}(G)$；$\norm G_*=\sum_i\sigma_i$ 是核范数，只决定走多远。
:::

上面的推导是完整的证明，但要说准它证明了什么。Bernstein 与 Newhouse 的命题 4、5 讲的是“离 $G$ 最近的半正交矩阵是 $UV^\top$”和“谱范数（多层时取各层的最大值）下的最速下降是 $-\eta U_lV_l^\top$”，论文把它们用来解释**不累积的 Shampoo**，正文并没有提到 Muon [@bernstein2024anthology]；把关掉动量的 Muon 看成这种最速下降，是 Jordan 的博客给出的对应 [@jordan2024muon]。它证明的是“在谱范数这把尺子下，$-UV^\top$ 是最陡的方向”；**它没有证明**用这把尺子训练神经网络会更快，那是下一页要讲的实测与推测。同样的推理换成 $\ell_\infty$ 范数（每个坐标最多走 $\delta$），最陡方向就是 $-\delta\,\sign(\mathbf g)$，也就是 $\beta_1=\beta_2=0$、$\epsilon=0$ 的 Adam（笔者推导，练习 2）。

::: ex 在谱范数为 1 的更新里，谁让损失降得最多
继续用 $G=\begin{pmatrix}3&0\\4&5\end{pmatrix}$。候选一：$-UV^\top$，谱范数 1，一阶降幅 $\tr(G^\top UV^\top)=\sigma_1+\sigma_2=4\sqrt5=8.944$。候选二：逐元素取符号再缩放到谱范数 1，$\sign(G)=\begin{pmatrix}1&0\\1&1\end{pmatrix}$ 的谱范数是 1.618，降幅 $(3+4+5)/1.618=7.416$。同样“大小”的一步，正交化方向多降 21%。反过来按 $\ell_\infty$ 范数比较，符号方向会赢：选哪个方向最好，取决于用哪把尺子量步长。
:::

## 牛顿–舒尔茨迭代：只用矩阵乘法算 $UV^\top$

对每个权重矩阵每步做一次 SVD，在 GPU 上太慢，Jordan 的博客明确说这是不用 SVD 的原因 [@jordan2024muon]。**牛顿–舒尔茨迭代**（Newton–Schulz iteration）只用矩阵乘法就能逼近 $UV^\top$，关键在于下面这个恒等式：$X$ 的奇数次“多项式”只作用在它的奇异值上。

::: deriv 奇次矩阵多项式只改奇异值
$XX^\top X=U\Sigma V^\top V\Sigma U^\top U\Sigma V^\top=U\Sigma^3V^\top$ || $X=U\Sigma V^\top$，$V^\top V=I$，$U^\top U=I$
$(XX^\top)^2X=U\Sigma^5V^\top$ || 同理，每多一对 $XX^\top$ 就多两次 $\Sigma$
$aX+b\,(XX^\top)X+c\,(XX^\top)^2X=U\,(a\Sigma+b\Sigma^3+c\Sigma^5)\,V^\top$ || 线性组合；$U$、$V$ 原封不动
:::

::: eq 一步牛顿–舒尔茨迭代 | (M7.25)
$$X_k=aX_{k-1}+b\,(X_{k-1}X_{k-1}^\top)X_{k-1}+c\,(X_{k-1}X_{k-1}^\top)^2X_{k-1}=U\,\varphi(\Sigma_{k-1})\,V^\top$$
其中 $\varphi(x)=ax+bx^3+cx^5$ 逐个作用在奇异值上。起点 $X_0=G/\norm G_F$，保证所有奇异值落在 $(0,1]$。每一步只有矩阵乘法，而且奇异向量从头到尾不变，只有奇异值 $x$ 被标量函数 $\varphi$ 反复作用。
:::

于是问题变成了一个一元问题：**找一个奇多项式 $\varphi$，让 $(0,1]$ 里的任何 $x$ 被迭代几次之后都接近 1。** 三个候选：

- **经典三次** $f(x)=1.5x-0.5x^3$：$f(1)=1$，$f'(1)=0$，在 1 附近收敛极快。Bernstein 与 Newhouse 在附录里说它改编自 Higham 的书，在 $0<x<\sqrt3$ 上把 $x$ 推向 1 [@bernstein2024anthology]。缺点是 $f'(0)=1.5$：很小的奇异值每步只放大 1.5 倍，从 0.01 出发 12 步后才到 0.893。
- **五次** $\varphi_B(x)=2x-1.5x^3+0.5x^5$：同样 $\varphi_B(1)=1$、$\varphi_B'(1)=0$，而 $\varphi_B'(0)=2$，从 0.01 出发 8 步到 1.002。Jordan 称这组系数“完全够用”。
- **Jordan 的五次** $\varphi_J$：为了少走几步，他把条件放宽为“迭代的极限落在 $[0.7,1.3]$ 里”，在这个约束下让 $a=\varphi'(0)$ 尽量大，用一种临时的梯度方法得到

::: eq Jordan 的系数 | (M7.26)
$$(a,b,c)=(3.4445,\ -4.7750,\ 2.0315),\qquad \varphi_J'(0)=3.4445,\qquad \varphi_J(1)=0.701$$
训练 Transformer 时 5 步就够；博客说奇异值偏离 1 达 0.3 左右也不影响损失曲线（经验观察）[@jordan2024muon]。
:::

代价是 $\varphi_J$ 不再把奇异值送到 1：$\varphi_J(1)=0.701$，峰值 $1.2024$（在 $x=0.5545$），反复迭代后落在一个吸引的 4 周期 $0.682\to1.134\to0.753\to1.047\to0.682$ 上，范围约 $[0.68,1.13]$；5 步之后，输入 $\ge10^{-3}$ 的奇异值落在 $[0.47,1.20]$。参考代码的注释把输出说成近似 $US'V^\top$、$S'$ 像是在 $[0.5,1.5]$ 里均匀分布 [@muon-code]。所以准确的说法是：Muon 做的是**近似**正交化。

::: fig figures/m07-ns-map.svg | 图 M7.11 牛顿–舒尔茨迭代对奇异值的作用
**先看右图：一个 0.01 的小奇异值，Jordan 系数（橙）4 步就冲到 1 附近，然后在 0.7 与 1.1 之间来回跳；五次 $(2,-1.5,0.5)$（蓝）要 8 步但稳稳停在 1；经典三次（灰）12 步还没到。** 左图是 $\varphi_J^N(x)$，颜色越深 $N$ 越大（1、2、3、5）：迭代几次之后，除了 0 附近很窄的一段，$(0,1]$ 都被压进橙色带 $[0.68,1.13]$（$N=5$ 时 $x\ge0.05$ 全在带内），但不会收成一条线；绿色带是博客的设计目标 $[0.7,1.3]$。绿色粗线是 DeepSeek-V4 的“8 步 Jordan + 2 步 $(2,-1.5,0.5)$”：几乎是 $y=1$ 的水平线，输入 $x\ge10^{-3}$ 时输出在 $[1.000000,1.000538]$ 里。所有曲线由脚本逐点迭代计算。
:::

::: ex 手算一步牛顿–舒尔茨（接着 2×2 的例子）
$\norm G_F=\sqrt{9+16+25}=\sqrt{50}$，$X_0=G/\sqrt{50}=\begin{pmatrix}0.4243&0\\0.5657&0.7071\end{pmatrix}$，奇异值 $3/\sqrt{10}=0.9487$ 与 $1/\sqrt{10}=0.3162$。

第一步，先算 $A=X_0X_0^\top=GG^\top/50$，$GG^\top=\begin{pmatrix}9&12\\12&41\end{pmatrix}$：

$$A=\begin{pmatrix}0.18&0.24\\0.24&0.82\end{pmatrix},\quad A^2=\begin{pmatrix}0.09&0.24\\0.24&0.73\end{pmatrix},\quad bA+cA^2=\begin{pmatrix}-0.6767&-0.6584\\-0.6584&-2.4325\end{pmatrix}.$$

再算 $X_1=aX_0+(bA+cA^2)X_0$：

$$X_1=\begin{pmatrix}1.4614&0\\1.9485&2.4356\end{pmatrix}+\begin{pmatrix}-0.6596&-0.4656\\-1.6554&-1.7200\end{pmatrix}=\begin{pmatrix}0.8018&-0.4656\\0.2931&0.7156\end{pmatrix}.$$

用标量函数核对：$\varphi_J(0.9487)=3.2678-4.0771+1.5612=0.7518$，$\varphi_J(0.3162)=1.0892-0.1510+0.0064=0.9447$，正是 $X_1$ 的两个奇异值。继续迭代，两个奇异值分别走

| 步数 $k$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 沿 $\mathbf u_1$ | 0.9487 | 0.7518 | 1.0484 | 0.6819 | 1.1343 | 0.7530 |
| 沿 $\mathbf u_2$ | 0.3162 | 0.9447 | 0.7568 | 1.0414 | 0.6825 | 1.1337 |

5 步之后 $X_5$ 与 $UV^\top$ 的最大元素差还有 0.196。换成 DeepSeek-V4 的做法：前 8 步用 Jordan 系数（第 8 步奇异值是 1.1343 与 0.6819），后 2 步用 $(2,-1.5,0.5)$，奇异值变成 $(1.0183,0.9619)$，再变成 $(1.0002,1.0005)$，$X_{10}=\begin{pmatrix}0.8948&-0.4475\\0.4472&0.8947\end{pmatrix}$，与 $UV^\top=\begin{pmatrix}0.8944&-0.4472\\0.4472&0.8944\end{pmatrix}$ 的最大差 $4\times10^{-4}$。
:::

最后两步为什么这么准：$\varphi_B(1)=1$、$\varphi_B'(1)=0$，把 $x=1+e$ 代进去做泰勒展开，误差 $e$ 变成约 $e^2/2$（$\varphi_B''(1)=1$），这是牛顿法式的二次收敛：误差 $\pm0.01$ 一步就变成约 $0.00005$（$\varphi_B(1.01)=1.0000535$，$\varphi_B(0.99)=1.0000465$）；而 8 步 Jordan 系数已经把奇异值赶进了 $[0.68,1.13]$，离 1 足够近（归一化后不小于约 $1.3\times10^{-4}$ 的奇异值都是如此；更小的奇异值 8 步还长不到 1 附近，这是任何固定步数的迭代都有的限制）。两组系数各干一半：前者把小奇异值快速抬起来，后者把结果精确地收到 1。DeepSeek-V4 的报告说后两步把奇异值“精确地稳定在 1” [@deepseekai2026v4]。

::: key 本页要点
- 正交化 $\mathrm{Ortho}(G)=UV^\top$：保留奇异向量、奇异值全换成 1，是离 $G$ 最近的半正交矩阵 (M7.22)；它是极分解 $G=QS$ 里的“旋转”部分 (M7.23)，矩阵版的 $z/\abs z$。
- 在谱范数下量步长时，最速下降方向恰好是 $-UV^\top$ (M7.24)（已证明，但证明的只是“这把尺子下最陡”）。
- 奇次矩阵多项式只作用在奇异值上 (M7.25)；牛顿–舒尔茨迭代因此只要矩阵乘法。Jordan 系数 (M7.26) 让小奇异值长得快，但不收敛到 1；再加两步 $(2,-1.5,0.5)$ 就收到 1 附近。
:::

# Muon：从 Jordan 的博客到 DeepSeek-V4 | 一步 Muon 的完整算法、更新 RMS 的缩放、代价，以及哪些是证明、哪些是观察 | 25

上一页只讲了一个运算。把它变成能训练千亿参数模型的优化器，还需要回答四个问题：动量放在哪里？更新的大小怎么定？哪些参数用它？代价多大？

## 定义：动量 + 正交化

**含义。** **Muon**（MomentUm Orthogonalized by Newton–Schulz）：对每个二维权重矩阵，先按 SGD 动量得到更新方向，再用牛顿–舒尔茨迭代把它近似正交化，最后按正交化后的矩阵更新参数。

**理解。** 动量负责跨步平均（§3），正交化负责把这个平均方向里各个奇异方向的大小拉平（§6）。顺序很重要：Jordan 的博客对比过“先正交化梯度、再做动量”的 Orthogonal-SGDM，发现把动量放在正交化之前效果更好（博客的经验观察）[@jordan2024muon]。

**数学。** Muon 没有论文，定义来自 Jordan 的博客（2024-12-08）与参考代码 [@jordan2024muon; @muon-code]。它只用于**隐藏层的二维参数**；博客建议标量、向量参数以及输入层、输出层用 AdamW 一类的方法，并说 Nesterov 式动量在所有测试里都略好，已设为默认。规模化的写法见 Moonshot AI 的 Liu 等人 [@liu2025muon]，DeepSeek-V4 沿用了其中的权重衰减、Nesterov 技巧与 RMS 缩放 [@deepseekai2026v4]：

::: eq 一步 Muon（DeepSeek-V4 算法 1，本课记号） | (M7.27)
$$\begin{aligned}\mathbf M_t&=\mu\,\mathbf M_{t-1}+\mathbf G_t\\ \mathbf O_t&=\gamma_{\mathrm{rms}}\sqrt{\max(n,m)}\cdot\mathrm{NS}_{8+2}\big(\mu\,\mathbf M_t+\mathbf G_t\big)\\ W_t&=W_{t-1}(1-\eta\lambda)-\eta\,\mathbf O_t\end{aligned}$$
$\mathbf G_t\in\R^{n\times m}$：这个矩阵的梯度；$\mathbf M_t$：动量缓冲，$\mu=0.95$；$\mu\mathbf M_t+\mathbf G_t$ 是 §3 的 Nesterov 形式 (M7.11)；$\mathrm{NS}_{8+2}$：先除以 Frobenius 范数，再做 8 步 Jordan 系数、2 步 $(2,-1.5,0.5)$ 的迭代；$\gamma_{\mathrm{rms}}=0.18$，配上因子 $\sqrt{\max(n,m)}$ 后它就是 $\mathbf O_t$ 的 RMS（正交化精确、被正交化的矩阵满秩时恰好相等；8+2 步之后，不太小的奇异值与 1 相差不到 $10^{-3}$，见 §6；下一节的 (M7.28)）；$\lambda=0.1$ 是解耦的权重衰减，写成乘法形式 $W(1-\eta\lambda)$，与 (M7.15) 等价。
:::

::: note 同一个 Muon 的几种写法
- **动量**：参考代码写成 EMA，$\mathbf M\leftarrow\beta\mathbf M+(1-\beta)\mathbf G$，送进迭代的是 $(1-\beta)\mathbf G+\beta\mathbf M$ [@muon-code]；Moonlight 与 DeepSeek-V4 写成缓冲形式。两者只差一个正的常数倍（§3 的注），而迭代前要除以 Frobenius 范数，所以得到**同一个** $\mathbf O_t$。
- **权重衰减**：Moonlight 写 $-\eta\lambda W$，DeepSeek-V4 写 $W(1-\eta\lambda)$，是同一件事。参考代码默认 `weight_decay=0`；Moonlight 指出，没有权重衰减时权重与层输出的 RMS 会一直增长，超出 bf16 的高精度范围 [@liu2025muon]。
- **形状缩放**：三种约定，下一节细讲。
:::

## 更新该多大：RMS 的缩放

**含义。** 一个矩阵的 **RMS**（root mean square，均方根）：所有元素平方的平均，再开方。

**理解。** 它是“一个典型元素有多大”，和矩阵有多少个元素无关。学习率乘的是更新方向：AdamW 的方向 $\hat{\mathbf m}_t/\sqrt{\hat{\mathbf v}_t}$ 每个元素的大小大致不超过 1（§4），Muon 的方向是正交化后的矩阵。想让两者共用一个学习率，就要让这两个方向的 RMS 可比。

**数学。** $X\in\R^{n\times m}$ 时 $\mathrm{RMS}(X)=\big(\frac1{nm}\sum_{ij}X_{ij}^2\big)^{1/2}=\norm X_F/\sqrt{nm}$。例如 $\begin{pmatrix}3&0\\4&5\end{pmatrix}$ 的 RMS 是 $\sqrt{50/4}=3.54$。

正交化之后，更新的大小完全由形状决定。设 $W\in\R^{n\times m}$（Moonlight 原文记作 $A\times B$；这里沿用 (M7.27) 的 $n\times m$，把 $B$ 留给 batch），被正交化的矩阵 $G\in\R^{n\times m}$ 满秩，于是 $UV^\top\in\R^{n\times m}$ 有 $\min(n,m)$ 个奇异值为 1：

::: deriv 正交化更新的 RMS（Moonlight 引理 1）
$\norm{UV^\top}_F^2=\sum_i1^2=\min(n,m)$ || Frobenius 范数的平方 = 奇异值的平方和
$\mathrm{RMS}(UV^\top)=\dfrac{\norm{UV^\top}_F}{\sqrt{nm}}=\sqrt{\dfrac{\min(n,m)}{nm}}=\dfrac1{\sqrt{\max(n,m)}}$ || RMS 的定义；$nm=\min(n,m)\cdot\max(n,m)$
:::

::: eq 更新 RMS 与缩放因子 | (M7.28)
$$\begin{aligned}\mathrm{RMS}\big(\mathrm{Ortho}(G)\big)&=\frac{1}{\sqrt{\max(n,m)}}\\ \mathrm{RMS}\big(\gamma_{\mathrm{rms}}\sqrt{\max(n,m)}\ \mathrm{Ortho}(G)\big)&=\gamma_{\mathrm{rms}}\end{aligned}$$
乘上这个因子，更新的 RMS 就与形状无关 [@liu2025muon]；Moonlight 取 0.2，DeepSeek-V4 取 0.18 [@deepseekai2026v4]。
:::

为什么要缩放到 0.2 左右？Moonlight 的理由是：AdamW 更新的 RMS 经验上在 0.2–0.4 之间，把 Muon 的更新调到同一量级，就可以**直接沿用 AdamW 的学习率与权重衰减**，一个模型里两种优化器共用一套超参数 [@liu2025muon]。不缩放的话问题出在形状上：$\max(n,m)$ 很大（稠密 MLP）时更新太小，很小（把每个 KV 头当成独立矩阵）时又太大。参考代码用的是第三种约定 $\sqrt{\max(1,\text{行}/\text{列})}$，更新的 RMS 大约是 $1/\sqrt{\text{列数}}$ [@muon-code]。

::: ex 一个 [7168, 18432] 的矩阵
这是 DeepSeek-V3 前 3 层稠密 FFN 的形状（隐藏维 7168、中间维 18432，见[第 11 课 §1](../deepseek/lesson-11-v3-architecture.html#p1)）。DeepSeek-V4 每一层都是 MoE、没有稠密 FFN（V4-Flash 的隐藏维是 4096，V4-Pro 是 7168，专家的中间维是 2048 与 3072 [@deepseekai2026v4]），这里只借 V3 的这个形状当一个“大矩阵”的代表。正交化后 RMS $=1/\sqrt{18432}=0.00737$。Moonlight 的缩放因子 $0.2\sqrt{18432}=27.15$，DeepSeek-V4 是 $0.18\sqrt{18432}=24.44$，缩放后 RMS 分别是 0.2 与 0.18。参考代码的因子是 $\sqrt{\max(1,7168/18432)}=1$，RMS 仍是 0.00737。同一个学习率，三种约定下每步的实际改动可以差 20 多倍，读别人的超参数时要先认清是哪一种。
:::

::: fig figures/m07-muon-pipeline.svg | 图 M7.12 DeepSeek-V4 的一步 Muon，以及参数的分工
**顺着上排的箭头读：梯度进入动量缓冲（黄色，是优化器的状态），Nesterov 组合后除以 Frobenius 范数，经过 8+2 步牛顿–舒尔茨，再乘 $0.18\sqrt{\max(n,m)}$，最后与解耦衰减一起更新 $W$。** 下排是 DeepSeek-V4 的分工：嵌入、预测头、mHC 的静态偏置与门控因子、全部 RMSNorm 权重仍用 AdamW，其余模块用 Muon；报告还说明同形状的矩阵可以合成一批做迭代 [@deepseekai2026v4]。中间一行借用 V3 稠密 FFN 的 [7168, 18432] 作例子（V4 没有稠密 FFN），RMS 数字由脚本按 (M7.28) 计算。
:::

## 用在哪些参数上

Muon 只处理二维矩阵，这是正交化的定义决定的：向量没有“奇异方向”可言。嵌入表与输出头虽然也是矩阵，Jordan 的博客仍建议交给 AdamW，说这样效果最好（经验观察）[@jordan2024muon]。DeepSeek-V4 的分工见图 M7.12；报告还说，由于注意力的 Q 与 KV 条目上直接有 RMSNorm，它不需要 QK-Clip 这类防止注意力 logit 过大的补丁 [@deepseekai2026v4]。DeepSeek-V4.1-Flash 再细分了一次：线性变换用 Muon（Q、K 的权重按头分别正交化），RMSNorm 与其他非矩阵参数用 AdamW，所有嵌入、预测头与 Engram 表改用一种“带 Sinkhorn 平衡的动量更新”：交替把更新矩阵的行和列除以各自的 $\ell_2$ 范数，共 11 步，学习率再乘 0.18 [@deepseekai2026v41]。交替归一化的数学在 [M11 §7](m11-doubly-stochastic-sinkhorn.html#p7)。

## 代价：多少 FLOPs，多少显存

设 $W\in\R^{n\times m}$、$m\le n$。实现时先把矩阵转置成“矮胖”的 $m\times n$，让 $XX^\top$ 只有 $m\times m$：算它要 $2nm^2$，算它的平方要 $2m^3$，再乘回 $X$ 又是 $2nm^2$。这正是 Jordan 的博客给出的计数。拿它和这一层本来的训练计算量比（一个 $n\times m$ 线性层处理 $B$ 个 token，前向加反向约 $6nmB$ FLOPs，见 [M10 §1](m10-compute-arithmetic.html#p1)）：

::: deriv 牛顿–舒尔茨开销的上界（笔者按博客的计数展开）
$T\cdot2(2nm^2+m^3)=Tm^2(4n+2m)$ || $T$ 步迭代，每步 $2nm^2+2m^3+2nm^2$
$\dfrac{Tm^2(4n+2m)}{6nmB}=\dfrac{Tm}{B}\cdot\dfrac{4n+2m}{6n}$ || 分子分母约去 $m$，把 $Tm/B$ 提出来
$\dfrac{4n+2m}{6n}\le1\quad(m\le n)$ || 方阵 $m=n$ 时等于 1；越“瘦长”越小
:::

::: eq Muon 的额外计算 | (M7.29)
$$\text{每步迭代}\ 2\,(2nm^2+m^3)\ \text{FLOPs},\qquad \frac{\text{NS 开销}}{\text{这一层一步训练的 }6nmB}\ \le\ \frac{T\,m}{B}$$
$T$ 是迭代步数，$B$ 是 batch 的 token 数；$6nmB$ 是这个线性层前向加反向的矩阵乘法 FLOPs。方阵时取等号，长方阵更小。
:::

博客的两个例子：NanoGPT（$m=768$，$B=524{,}288$，$T=5$）的上界 $5\times768/524288=0.73\%$，Llama 405B（$m=16384$，$B=1.6\times10^7$）约 $0.51\%$，博客分别写作 0.7% 与 0.5% [@jordan2024muon]。

::: ex V4 的 batch 下，一个大矩阵的开销
仍用上面 $[7168,18432]$ 的矩阵（V3 稠密 FFN 的形状），$m=7168$、$n=18432$：每步迭代 $2(2\times18432\times7168^2+7168^3)=4.52\times10^{12}$ FLOPs，10 步 $4.52\times10^{13}$。这一层一步训练是 $6\times18432\times7168\times7.55\times10^7=5.99\times10^{16}$（batch 取 V4-Flash 的 7550 万 token）。开销 $=0.076\%$，上界 $Tm/B=10\times7168/7.55\times10^7=0.095\%$，两者之比正是上面的 $(4n+2m)/(6n)=0.796$。batch 越大，这笔账越便宜。显存方面，Muon 每个参数只存一个动量，Adam 要存两个：1.6 万亿参数、FP32 状态下是 6.4 TB 对 12.8 TB。工程上真正麻烦的是另一件事：正交化需要**完整的**梯度矩阵，与把参数切碎分到各卡的做法冲突，第 18 课再讲 DeepSeek 怎样处理。
:::

## 为什么 Muon 可能更好：分清证明、实测与推测

::: thm 已证明（数学上成立，但要看清它说的是什么）
1. $UV^\top$ 是离 $G$ 最近的半正交矩阵 (M7.22)；在谱范数下量步长，最速下降的方向是 $-UV^\top$ (M7.24)。原论文讲的是不累积的 Shampoo，Muon 的对应来自 Jordan 的博客 [@bernstein2024anthology; @jordan2024muon]。
2. 奇次矩阵多项式只改奇异值 (M7.25)；Jordan 系数不收敛到 1，$(2,-1.5,0.5)$ 在 1 附近二次收敛（§6 的计算）。
3. 满秩矩阵正交化后的 RMS 是 $1/\sqrt{\max(n,m)}$ (M7.28)。

这些结论都**没有**说：用 Muon 训练神经网络会比 AdamW 收敛得更快。本课读过的来源里没有这样的定理。
:::

**实测。** Moonlight 报告，在计算最优的训练设置下，Muon 的计算效率约为 AdamW 的 2 倍 [@liu2025muon]。Jordan 报告奇异值偏离 1 达 0.3 左右也不伤损失曲线 [@jordan2024muon]。DeepSeek-V4 说它对大多数模块改用 Muon，是因为收敛更快、训练更稳定，并说牛顿–舒尔茨迭代在 BF16 矩阵乘法下保持稳定 [@deepseekai2026v4]。这些都是特定设置下的实验结果。

**推测。** Jordan 通过手工检查观察到，Transformer 里二维参数的 SGD 动量更新与 Adam 更新通常有非常高的条件数：几乎是低秩的，所有神经元的更新都被少数几个方向主导。他**推测**（博客原文用的是 speculate）正交化的作用是放大那些幅度小、却对学习很重要的“稀有方向” [@jordan2024muon]。图 M7.10 与 §6 的手算例子让这个说法有了具体的样子：第一步迭代就把 0.316 的奇异值抬到了 0.945。但它是否就是 Muon 更快的原因，目前只是一个合理的假说。

::: warn 失败模式与常见误解
- “Muon 把更新变成 $UV^\top$”：只对 DeepSeek-V4 的 8+2 步近似成立；纯 Jordan 系数的输出奇异值在约 $[0.68,1.13]$（多步）或 $[0.47,1.20]$（5 步）里。
- “Muon 适用于所有参数”：只用于隐藏层的二维矩阵；嵌入、输出头、向量与标量参数用 AdamW（V4.1 还把嵌入与预测头交给了 Sinkhorn 平衡更新）。
- 不加权重衰减：权重与输出的 RMS 持续增长（Moonlight）。
- 照搬学习率而不看形状缩放的约定：三种约定下同一学习率的实际步长可以差一个数量级以上。
:::

::: note 这一页用在哪里
[第 18 课 §5](../deepseek/lesson-18-v4.html#p5)：DeepSeek-V4 的 Muon（算法 1、混合 8+2 步、RMS 0.18、参数分工、与 ZeRO 的冲突）。[第 19 课 §7](../deepseek/lesson-19-v4-1-flash.html#p7)：V4.1-Flash 的训练日程；它的按头 Muon 与 Sinkhorn 平衡更新第 19 课没有展开，本页“用在哪些参数上”一节已经交代，交替归一化的数学见 [M11 §7](m11-doubly-stochastic-sinkhorn.html#p7)。
:::

::: key 本页要点
- 一步 Muon (M7.27)：动量缓冲 → Nesterov 组合 → Frobenius 归一化 → 牛顿–舒尔茨（V4 为 8+2 步）→ 乘 $\gamma_{\mathrm{rms}}\sqrt{\max(n,m)}$ → 连同解耦衰减一起更新。
- 正交化后 RMS 只由形状决定 (M7.28)；缩放到 0.2（Moonlight）或 0.18（V4），就能与 AdamW 共用学习率和权重衰减。
- 额外计算至多 $Tm/B$ (M7.29)：$m=7168$、V4-Flash 的 7550 万 token batch、10 步迭代时不到 0.1%；显存比 Adam 少一份状态。
- 已证明的是几何（最近的正交矩阵、谱范数下最陡）；“更快”是实测；“放大稀有方向”是推测。
:::

# 练习与答案 | 11 题：概念、计算、推导 | 50

先自己做，再展开答案。计算题用计算器即可，每题都能用几行 NumPy 核对。

## 概念题

:::: quiz 练习 1 · 条件数、对角步长与坐标轴
损失 $\mathcal L=\tfrac12\theta^\top H\theta$，$H=\diag(1,100)$。（a）只调学习率，梯度下降最快要多少步才能把误差缩到 $10^{-6}$？（b）如果允许每个坐标有自己的步长 $\eta_i$，最少几步？（c）把这个碗旋转 $45^\circ$（$H$ 的特征值不变，但不再是对角矩阵），（b）的办法还灵吗？这说明 Adam 这类“逐坐标”缩放有什么局限？
::: answer 看答案
（a）$\kappa=100$，由 (M7.4) 最优收敛因子 $99/101=0.9802$，需要 $\ln10^{-6}/\ln(99/101)=690.8$，向上取整 691 步。

（b）取 $\eta_1=1/1$、$\eta_2=1/100$，每个坐标的因子 $1-\eta_ih_i=0$，一步到底。

（c）不灵了。旋转后曲率最大、最小的方向是两条对角线，不再对齐坐标轴；逐坐标的步长只能沿坐标轴伸缩，改变不了“沿对角线方向尺度悬殊”这件事。Adam 的缩放是按坐标轴做的（对角预条件），只有当难走的方向恰好与坐标轴对齐时才最有效。Muon 在矩阵自己的奇异方向上操作，不依赖坐标轴，这是两者的一个结构差别（至于哪个在神经网络上更好，是实验问题）。
:::
::::

:::: quiz 练习 2 · “只留方向”的三种版本
（a）$\beta_1=\beta_2=0$、$\epsilon=0$ 时，Adam (M7.12) 的更新是什么？（b）证明：约束每个坐标最多移动 $\delta$（$\norm{\boldsymbol\Delta}_\infty\le\delta$）时，使一阶变化 $\mathbf g^\top\boldsymbol\Delta$ 最小的 $\boldsymbol\Delta\in\R^N$ 是 $-\delta\,\sign(\mathbf g)$。（c）一个 $1\times m$ 的“矩阵”（行向量）$\mathbf g^\top$ 正交化后是什么？与 $\sign(\mathbf g)$ 有什么不同？
::: answer 看答案
（a）$\mathbf m_t=\mathbf g_t$，$\mathbf v_t=\mathbf g_t^2$，没有偏差（$1-0^t=1$），更新 $-\eta\,\mathbf g_t/\abs{\mathbf g_t}=-\eta\,\sign(\mathbf g_t)$，每个坐标恰好走 $\eta$。

（b）$\mathbf g^\top\boldsymbol\Delta=\sum_ig_i\Delta_i\ge-\sum_i\abs{g_i}\,\abs{\Delta_i}\ge-\delta\sum_i\abs{g_i}$，第一步是每一项 $g_i\Delta_i\ge-\abs{g_i}\abs{\Delta_i}$，第二步用 $\abs{\Delta_i}\le\delta$。取 $\Delta_i=-\delta\,\sign(g_i)$ 时两个不等号同时取等。所以 $\ell_\infty$ 范数下最陡的方向是符号方向，对应 (M7.24) 在谱范数下的 $-UV^\top$。

（c）秩为 1：$\mathbf g^\top=\sigma_1\cdot1\cdot\mathbf v_1^\top$，$\sigma_1=\norm{\mathbf g}$，$\mathbf v_1=\mathbf g/\norm{\mathbf g}$，所以 $\mathrm{Ortho}(\mathbf g^\top)=\mathbf g^\top/\norm{\mathbf g}$：只是把整个向量缩放到单位长度，各分量之间的相对大小全部保留。$\sign(\mathbf g)$ 则把每个分量都变成 $\pm1$，抹掉了相对大小。可见“只留方向”要看以什么为单位：元素、奇异方向，还是整个向量。
:::
::::

:::: quiz 练习 3 · 读 V4.1 的调度
DeepSeek-V4.1-Flash 的学习率：warmup 2000 步，$2.6\times10^{-4}$ 保持到 28T token，28T–40T cosine 降到 $2.6\times10^{-5}$，40T–45T 保持。（a）它满足 Robbins–Monro 条件 (M7.7) 吗？（b）用 §2 的噪声地板说明稳定期、衰减期、末段各自在做什么。（c）与 cosine 相比，它在“续训”上有什么好处？
::: answer 看答案
（a）不满足。末段学习率恒为 $2.6\times10^{-5}$，$\sum_t\eta_t^2$ 随步数线性增长，第二条不成立；而且训练本来就是有限步。定理对它不保证什么。

（b）稳定期：大学习率让参数快速走远（$\sum\eta_t$ 大），代价是停在较高的噪声地板上。衰减期：地板正比于 $\eta$ (M7.6)，学习率降到十分之一，地板也跟着降，损失随之下降。末段：学习率不再变，参数在新的、较低的地板附近继续被数据推动。这是二次模型给出的直觉，不是对 V4.1 的精确描述。

（c）稳定期的学习率与计划总量无关：训练到 28T 之前的任何检查点都可以接着训，或者从那里分叉出不同长度的衰减段（第 6 课对多步调度算过这笔账）。cosine 的每一步都依赖总量，做不到。
:::
::::

## 计算题

:::: quiz 练习 4 · 梯度下降与重球
$H=\diag(2,50)$。（a）梯度下降的稳定上限、最优步长、收敛因子，以及把误差缩到 $10^{-6}$ 的步数。（b）重球法的最优 $\mu^*$、$\eta^*$、收敛因子与步数。（c）$\eta^*$ 是否超过了（a）的稳定上限？
::: answer 看答案
（a）$\eta<2/50=0.04$；$\eta^*=2/(2+50)=0.0385$；$\kappa=25$，$\rho^*=24/26=0.923$；$\ln10^{-6}/\ln(24/26)=172.6$，向上取整 173 步。

（b）$\sqrt\kappa=5$，$\rho^*=(5-1)/(5+1)=2/3$，$\mu^*=4/9=0.444$，$\eta^*=4/(\sqrt{50}+\sqrt2)^2=4/(7.071+1.414)^2=4/72=0.0556$；$\ln10^{-6}/\ln(2/3)=34.07$，向上取整 35 步，约少 5 倍（$173/35=4.9\approx\sqrt\kappa$）。

（c）超过了：$0.0556>0.04$。单独用这个步长，陡峭方向的因子是 $1-0.0556\times50=-1.78$，会发散；动量项把它拉了回来。
:::
::::

:::: quiz 练习 5 · 噪声地板与降档
一维 $f(x)=\tfrac12hx^2$，$h=0.5$，单个样本梯度的方差 $\sigma^2=4$，batch $B=16$。（a）$\eta=0.2$ 时，稳态方差 $v_\infty$ 与噪声地板各是多少？近似式 $\eta\sigma^2/(4B)$ 给出多少？（b）把 $\eta$ 降到 0.02，新的地板是多少？方差每步向新稳态靠近多少，大约多少步才贴近到 5% 以内？（c）另一种办法是保持 $\eta=0.2$，把 batch 加到 160。地板是多少？与（b）相比，两种办法各付出什么代价？
::: answer 看答案
（a）每步噪声的方差 $\sigma^2/B=0.25$。代入 §2 推导出的稳态方差：

$$v_\infty=\frac{\eta\sigma^2}{Bh(2-\eta h)}=\frac{0.2\times4}{16\times0.5\times1.9}=0.05263,$$

$$\text{地板}=\tfrac12hv_\infty=0.25\times0.05263=0.01316.$$

近似式给出 $0.2\times4/64=0.0125$，小 5%，因为 $\eta h=0.1$ 还不算很小。

（b）同一个公式，$\eta=0.02$：

$$v_\infty=\frac{0.02\times4}{16\times0.5\times1.99}=0.005025,\qquad \text{地板}=0.001256,$$

约为原来的十分之一。把方差递推 $\Var[x_{t+1}]=(1-\eta h)^2\Var[x_t]+\eta^2\sigma^2/B$ 与稳态方程 $v_\infty=(1-\eta h)^2v_\infty+\eta^2\sigma^2/B$ 相减，噪声项消掉：

$$\Var[x_{t+1}]-v_\infty=(1-\eta h)^2\,\big(\Var[x_t]-v_\infty\big).$$

与新稳态的差距每步乘 $0.99^2=0.9801$，缩到 5% 要 $\ln20/(-\ln0.9801)\approx150$ 步，约三个弛豫时间 $1/(2\eta h)=50$ 步。

（c）$\eta=0.2$、$B=160$：

$$v_\infty=\frac{0.2\times4}{160\times0.5\times1.9}=0.005263,\qquad \text{地板}=0.001316,$$

也约为原来的十分之一：地板只依赖 $\eta/B$。代价不同。（b）每步的计算量不变，但均值每步只乘 $1-\eta h=0.99$ 而不是 0.9，离谷底还远时慢得多：把均值缩到千分之一，因子 0.9 要 66 步，0.99 要 688 步。（c）保住了走得快的均值，代价是每步要算 10 倍的样本。这就是 §5 里 batch 与学习率要一起选的原因。
:::
::::

:::: quiz 练习 6 · Adam 的前两步
$\beta_1=0.9$、$\beta_2=0.95$、$\epsilon=0$，某个坐标的梯度依次是 $g_1=0.02$、$g_2=0.01$。求 $t=1,2$ 时的 $m_t,v_t,\hat m_t,\hat v_t$、更新比值 $\hat m_t/\sqrt{\hat v_t}$，以及不做偏差修正的比值 $m_t/\sqrt{v_t}$。
::: answer 看答案
$t=1$：$m_1=0.1\times0.02=0.002$，$v_1=0.05\times0.0004=2\times10^{-5}$；$\hat m_1=0.002/0.1=0.02$，$\hat v_1=2\times10^{-5}/0.05=4\times10^{-4}$；比值 $0.02/0.02=1$。不修正：$0.002/\sqrt{2\times10^{-5}}=0.002/0.004472=0.447$。

$t=2$：$m_2=0.9\times0.002+0.1\times0.01=0.0028$，$v_2=0.95\times2\times10^{-5}+0.05\times10^{-4}=2.4\times10^{-5}$；$1-\beta_1^2=0.19$，$1-\beta_2^2=0.0975$；$\hat m_2=0.01474$，$\hat v_2=2.462\times10^{-4}$，$\sqrt{\hat v_2}=0.01569$，比值 $0.939$。不修正：$0.0028/0.004899=0.572$。

修正后的比值接近 1（梯度同号），不修正时前两步分别只有修正后的 45% 与 61%（$0.447/1$、$0.572/0.939$）：$\beta_2=0.95$ 时偏差的方向与默认 $\beta_2=0.999$ 相反（对照图 M7.5 右图的橙线）。
:::
::::

:::: quiz 练习 7 · 同一个“λ = 0.1”
学习率 $\eta=2.6\times10^{-4}$、$\lambda=0.1$。（a）按 PyTorch / 本课 (M7.15) 的约定，每步权重乘多少？若梯度恰好为零，V4.1 稳定期约 $28\times10^{12}/1.006\times10^8$ 步之后乘了多少？（b）有人照 AdamW 论文算法 2 的写法实现（衰减项乘的是调度乘子，取 1，不乘基础学习率），又把 $\lambda=0.1$ 原样抄过来，每步乘多少？10 步后呢？
::: answer 看答案
（a）每步乘 $1-2.6\times10^{-5}=0.999974$。稳定期约 $2.78\times10^5$ 步，$0.999974^{278330}=e^{-278330\times2.6\times10^{-5}}\approx e^{-7.24}=7.2\times10^{-4}$。实际中梯度会不断把权重推开，两者达成平衡；这个数只说明衰减在长训练里是一股不可忽略的力。

（b）每步乘 $1-0.1=0.9$，10 步后 $0.9^{10}=0.349$：权重在几十步内就被拉到接近零，训练会被毁掉。两种约定的 $\lambda$ 相差一个基础学习率的倍数；从论文约定换到 PyTorch 约定，要用 $\lambda_{\text{PyTorch}}=\lambda_{\text{论文}}/\alpha$。
:::
::::

:::: quiz 练习 8 · batch 与学习率
（a）按 Goyal 等人的设置 $\eta=0.1\times B/256$，batch 4096 的学习率是多少？（b）按 (M7.19)，$B=\mathcal B_{\mathrm{noise}}/4$ 时把 batch 加倍，每步最多能降的损失变成原来的几倍？$B=4\,\mathcal B_{\mathrm{noise}}$ 时呢？（c）两种情形下，加倍 batch 对“达到同一损失所用的样本数”分别意味着什么？
::: answer 看答案
（a）$0.1\times4096/256=1.6$。

（b）比例 $1/(1+\mathcal B_{\mathrm{noise}}/B)$：$B/\mathcal B_{\mathrm{noise}}=0.25\to0.2$，加倍后 $0.5\to1/3=0.333$，是原来的 1.67 倍；$4\to0.8$，加倍后 $8\to0.889$，只是原来的 1.11 倍。

（c）步数反比于每步收益：由 (M7.19) 与 (M7.20)，$S/S_{\min}=1+\mathcal B_{\mathrm{noise}}/B$，样本数 $E=BS$ 满足 $E/E_{\min}=1+B/\mathcal B_{\mathrm{noise}}$。第一种情形，$S/S_{\min}$ 从 5 降到 3（步数少 40%），$E/E_{\min}$ 从 1.25 升到 1.5（样本只多 20%），划算。第二种情形，$S/S_{\min}$ 从 1.25 降到 1.125（步数只少 10%），$E/E_{\min}$ 从 5 升到 9（样本多用 80%），不划算。这正是 (M7.20) 双曲线的两端。
:::
::::

:::: quiz 练习 9 · 一个 [7168, 1536] 矩阵的 Muon
$W\in\R^{7168\times1536}$，batch $B=7.55\times10^7$ token，混合牛顿–舒尔茨 $T=10$ 步。（a）正交化后的 RMS 是多少？DeepSeek-V4 的缩放因子是多少？（b）每步迭代的 FLOPs、10 步合计、这一层一步训练的 $6nmB$，以及开销占比与上界 $Tm/B$。
::: answer 看答案
（a）$\max=7168$，RMS $=1/\sqrt{7168}=0.01181$；缩放因子 $0.18\sqrt{7168}=15.24$，缩放后 RMS 为 0.18。

（b）$n=7168$、$m=1536$：每步 $2(2\times7168\times1536^2+1536^3)=2(3.382\times10^{10}+3.624\times10^9)=7.49\times10^{10}$；10 步 $7.49\times10^{11}$。$6\times7168\times1536\times7.55\times10^7=4.99\times10^{15}$。占比 $1.5\times10^{-4}=0.015\%$；上界 $10\times1536/7.55\times10^7=0.020\%$。长方阵的实际开销比上界小。
:::
::::

## 推导题

:::: quiz 练习 10 · 秩 1 的更新
$G=\begin{pmatrix}4&0\\3&0\end{pmatrix}$。（a）写出它的 SVD、$\norm G_F$ 与 $G/\norm G_F$ 的奇异值。（b）用 Jordan 系数迭代 3 步，非零奇异值依次是多少？零奇异值会变吗？（c）正交化后得到的矩阵是什么？(M7.22) 的极小元在这里唯一吗？
::: answer 看答案
（a）只有一个非零奇异值：$G=5\,\mathbf u\mathbf v^\top$，$\mathbf u=(0.8,0.6)$，$\mathbf v=(1,0)$，$\sigma_1=5$，$\sigma_2=0$。$\norm G_F=\sqrt{16+9}=5=\sigma_1$，所以归一化后的奇异值**恰好是 1**。

（b）$1\to\varphi_J(1)=0.701\to\varphi_J(0.701)=1.114\to\varphi_J(1.114)=0.721$。零奇异值不变：$\varphi(0)=0$，奇多项式把 0 送到 0。这个例子最直观地说明 Jordan 系数不会把奇异值停在 1：即使输入已经是 1，它也会把它推开。

（c）迭代给出 $\varphi_J^N(1)\,\mathbf u\mathbf v^\top$，理想的正交化是 $\mathbf u\mathbf v^\top=\begin{pmatrix}0.8&0\\0.6&0\end{pmatrix}$。$G$ 不满秩，(M7.22) 的极小元不唯一：记 $\mathbf u_\perp=(-0.6,0.8)$、$\mathbf v_\perp=(0,1)$，$O_\pm=\mathbf u\mathbf v^\top\pm\mathbf u_\perp\mathbf v_\perp^\top$ 都是正交矩阵，到 $G$ 的距离平方都是 $2+25-2\times5=17$（更大的矩阵里可选的更多）。牛顿–舒尔茨给出的是“零奇异值保持为零”的 $\mathbf u\mathbf v^\top$，它其实不是正交矩阵。
:::
::::

:::: quiz 练习 11 · 最后两步为什么这么准
$\varphi_B(x)=2x-1.5x^3+0.5x^5$。（a）验证 $\varphi_B(1)=1$、$\varphi_B'(1)=0$、$\varphi_B''(1)=1$，并由泰勒展开说明 $\varphi_B(1+e)\approx1+e^2/2$。（b）从 Jordan 迭代的两个极端值 1.1343 与 0.6819 出发，各算两步 $\varphi_B$。（c）既然 $\varphi_B$ 这么准，为什么不从一开始就只用它？
::: answer 看答案
（a）$\varphi_B(1)=2-1.5+0.5=1$；$\varphi_B'(x)=2-4.5x^2+2.5x^4$，$\varphi_B'(1)=0$；$\varphi_B''(x)=-9x+10x^3$，$\varphi_B''(1)=1$。泰勒展开 $\varphi_B(1+e)=1+0\cdot e+\tfrac12e^2+O(e^3)$：误差平方地缩小，这就是二次收敛。

（b）$1.1343\to1.0183\to1.00019$；$0.6819\to0.9619\to1.00054$。第一步离 $1+e^2/2$ 的预测还差得远：$e=0.134$ 时预测 1.0090、实际 1.0183；$e=-0.318$ 时预测 1.0506、实际 0.9619，连误差的正负都不对。原因是 $e$ 还不够小，三次项不能忽略：$\varphi_B'''(x)=-9+30x^2$，$\varphi_B'''(1)=21$，泰勒展开再加一项 $\tfrac{21}{6}e^3=3.5e^3$，两个预测变成 1.0175 与 0.938，接近实际值。$\varphi_B$ 是多项式，展开是精确的有限和：$\varphi_B(1+e)=1+\tfrac12e^2+3.5e^3+2.5e^4+0.5e^5$；后者还差的 0.024 主要来自四次项（$2.5e^4=0.026$）。第二步的起点误差只有 $0.018$ 与 $-0.038$，二次项主导，$e^2/2$ 给出 $1.00017$ 与 $1.00073$，再加三次项就是 $1.00019$ 与 $1.00053$。两步之后误差在 $10^{-3}$ 以内，与 V4 的输出范围 $[1.000000,1.000538]$ 一致。

（c）$\varphi_B'(0)=2$，而 $\varphi_J'(0)=3.4445$：很小的奇异值用 $\varphi_B$ 要走更多步才能长到 1 附近（从 0.01 出发要 8 步，Jordan 系数 4 步就到）。DeepSeek-V4 的 8+2 让两组系数各做擅长的事：先用 $\varphi_J$ 把所有奇异值快速赶进 $[0.68,1.13]$，再用 $\varphi_B$ 精确收尾。
:::
::::

```refs
- id: robbins1951
  title: "A Stochastic Approximation Method"
  a: Herbert Robbins, Sutton Monro (1951). The Annals of Mathematical Statistics 22(3):400–407, doi:10.1214/aoms/1177729586
  url: https://doi.org/10.1214/aoms/1177729586
  finding: 用带噪声的观测求回归函数的根，证明步长满足 Σa_n = ∞、Σa_n² < ∞ 时迭代均方收敛（本课按 Lai 与 Yuan 的转述引用，原文未读）。
  matters: 本课 §2 用它说明学习率为什么要衰减、衰减到什么程度，并指出大模型的常数期与非零终值不在它的保证之内。
- id: lai2021sa
  title: "Stochastic Approximation: from Statistical Origin to Big-Data, Multidisciplinary Applications"
  a: Tze Leung Lai, Hongsong Yuan (2021). Statistical Science 36(2), doi:10.1214/20-STS784（本课读的是作者主页的预印本，页眉 “Submitted to Statistical Science”，文件日期 2019-12-09）
  url: https://tzelai.ckirby.su.domains/pubs/2021_StochApprox.pdf
  finding: 综述随机逼近的起源，给出 Robbins–Monro 递推的步长条件及两条条件各自的作用：平方和控制噪声，和为无穷保证到达根。
  matters: Robbins–Monro 原刊页面对脚本不可访问，本课 (M7.7) 的陈述以这篇综述 §2.1 为可读的二手出处。
- id: dsllm
  title: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"
  a: DeepSeek-AI（Bi, X. 等 86 人）(2024). arXiv:2401.02954v1
  url: https://arxiv.org/abs/2401.02954
  finding: 用多步学习率调度（80% 与 90% 训练量处各乘 0.316）代替 cosine，AdamW（β₁ 0.9、β₂ 0.95、衰减 0.1）与梯度裁剪 1.0，并拟合出最优学习率与 batch size 随算力的幂律。
  matters: 本课用它作“降档陡降”、梯度裁剪阈值和 batch/学习率缩放的 DeepSeek 实例（调度、AdamW 设置、裁剪阈值与幂律指数已对 arXiv 全文核对）。
- id: deepseekai2026v41
  title: "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"
  a: DeepSeek-AI (2026). Technical report, arXiv:2609.19969v1
  url: https://arxiv.org/abs/2609.19969
  finding: §4.2.2 给出训练配方：batch 固定 100.6M token，2000 步线性 warmup 后学习率 2.6×10⁻⁴ 保持到 28T token，28T–40T cosine 降到 2.6×10⁻⁵，40T–45T 保持；AdamW 与 Muon 超参数同 V4；§2.5：Q、K 权重按头做 Muon，嵌入、预测头与 Engram 表改用 Sinkhorn 平衡的动量更新（K=11，学习率乘 0.18）。
  matters: 本课用它作 WSD 形状调度的真实例子（§5），以及 Muon 在课程终点模型里的用法（§7）。
- id: deepseekai2026v4
  title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
  a: DeepSeek-AI (2026). Technical report, arXiv:2606.19348v1
  url: https://arxiv.org/abs/2606.19348
  finding: §2.4 与算法 1：除嵌入、预测头、mHC 的静态偏置与门控因子、RMSNorm 权重用 AdamW 外，其余模块用 Muon（Nesterov 动量、混合牛顿–舒尔茨 8+2 步、更新 RMS 缩放到 0.18、权重衰减 0.1）；§4.2.2：AdamW 取 β₁ 0.9、β₂ 0.95、ε 10⁻²⁰、衰减 0.1，2000 步线性 warmup，学习率 2.7×10⁻⁴、末尾 cosine 到 2.7×10⁻⁵，batch 逐步增到 75.5M token；§4.2.1：V4-Pro 1.6T 参数、隐藏维 7168，V4-Flash 284B、隐藏维 4096，每层都是 MoE。
  matters: 本课 §4 的 AdamW 超参数、§5 的调度与 batch、§7 的 Muon 设置都以它为 DeepSeek 实例；“NS 不收敛到 1、所以再加两步”这一设计也出自这里。
- id: polyak1964
  title: "Some Methods of Speeding up the Convergence of Iteration Methods"
  a: B. T. Polyak (1964). USSR Computational Mathematics and Mathematical Physics 4(5):1–17, doi:10.1016/0041-5553(64)90137-5
  url: https://doi.org/10.1016/0041-5553(64)90137-5
  finding: 提出重球（动量）法，并证明在极小点附近适当选参数时，所需迭代次数比最速下降少约 √R 倍（按 Sutskever 等 2013 的转述，原文未读）。
  matters: 本课 §3 动量一节的原始出处；Adam 的一阶矩与 Muon 的动量缓冲都从这里来。
- id: sutskever2013
  title: "On the Importance of Initialization and Momentum in Deep Learning"
  a: Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton (2013). ICML 2013, JMLR W&CP 28
  url: https://proceedings.mlr.press/v28/sutskever13.pdf
  finding: 用统一记号写出经典动量与 Nesterov 加速梯度（式 1–4），两者只差速度更新时梯度的取点；并提醒随机梯度下渐近的局部加速优势会丧失。
  matters: 本课的动量与 Nesterov 公式以它为准；§3 还指出它转述 Polyak 时写的 μ 值按推导应为收敛因子。
- id: kingma2015adam
  title: "Adam: A Method for Stochastic Optimization"
  a: Diederik P. Kingma, Jimmy Ba (2015). ICLR 2015, arXiv:1412.6980v9
  url: https://arxiv.org/abs/1412.6980v9
  finding: 提出用一阶、二阶矩的指数滑动平均并做偏差修正的自适应优化器（算法 1，默认 α 0.001、β₁ 0.9、β₂ 0.999、ε 10⁻⁸），§3 推导偏差修正，§2.1 说明有效步长“近似有界”并给出两种典型情形的界。
  matters: 本课 §4 的 Adam 算法、偏差修正推导与步长上界都以 v9 为准；v1 的参数化不同，课文专门提醒。
- id: loshchilov2019adamw
  title: "Decoupled Weight Decay Regularization"
  a: Ilya Loshchilov, Frank Hutter (2019). ICLR 2019, arXiv:1711.05101v3
  url: https://arxiv.org/abs/1711.05101
  finding: 证明权重衰减与 L2 正则只在 SGD 下等价（需 λ'=λ/α，命题 1），在自适应方法下不等价（命题 2），并提出把衰减从梯度更新中解耦的 AdamW（算法 2）。
  matters: 本课 §4 讲“为什么用 AdamW 而不是 Adam + L2”的依据，也是“论文的衰减乘调度乘子、不乘学习率”这一约定差别的出处。
- id: pytorch-adamw
  kind: artefact
  title: "torch.optim.AdamW (torch/optim/adamw.py)"
  a: PyTorch contributors. GitHub repository (main branch), read 2026-09-24
  url: https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
  finding: 默认 lr 1e-3、betas (0.9, 0.999)、eps 1e-8、weight_decay 1e-2；衰减写作 θ ← θ − lr·λ·θ，其中 lr 含调度。
  matters: 本课用它说明实践中的 weight_decay 乘了学习率，与 AdamW 论文算法 2 的写法不同。
- id: goyal2017
  title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
  a: Priya Goyal, Piotr Dollár, Ross Girshick, et al. (2017). Technical report, arXiv:1706.02677v2
  url: https://arxiv.org/abs/1706.02677
  finding: 提出线性缩放规则（batch 乘 k、学习率乘 k）与 5 个 epoch 的渐进 warmup，把 ResNet-50 的 batch 扩到 8k 而精度不降，并写明规则在训练初期与 batch 过大时失效。
  matters: 本课 §5 的线性缩放推导（式 3、4）与 warmup 的一个具体动机都来自这里。
- id: mccandlish2018
  title: "An Empirical Model of Large-Batch Training"
  a: Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team (2018). arXiv:1812.06162v1
  url: https://arxiv.org/abs/1812.06162
  finding: 定义梯度噪声尺度 B_noise = tr(HΣ)/(GᵀHG) 及其简化 B_simple = tr(Σ)/|G|²，由此推出最优步长随 batch 饱和、步数与样本数的双曲线权衡和临界 batch。
  matters: 本课 §5 用它回答“batch 大到多少不再划算”，并说明线性缩放规则只是小 batch 一端的近似。
- id: smith2018bayesian
  title: "A Bayesian Perspective on Generalization and Stochastic Gradient Descent"
  a: Samuel L. Smith, Quoc V. Le (2018). ICLR 2018, arXiv:1710.06451v3
  url: https://arxiv.org/abs/1710.06451
  finding: 把 SGD 看成带噪声尺度 g ≈ εN/B 的随机过程（ε 为学习率、N 为训练集大小），推出最优 batch 与学习率、训练集大小成正比。
  matters: 本课用它提醒“噪声尺度”一词有两种不同的含义，避免与 McCandlish 的 B_noise 混淆。
- id: hu2024minicpm
  title: "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies"
  a: Shengding Hu, Yuge Tu, Xu Han, et al. (2024). arXiv:2404.06395v3
  url: https://arxiv.org/abs/2404.06395
  finding: 提出 Warmup–Stable–Decay 学习率调度（式 1），观察到衰减阶段损失骤降、约 10% token 的衰减就够，稳定期的检查点可以低成本地复用。
  matters: 本课 §5 的 WSD 公式与“衰减段损失骤降”的出处；DeepSeek-V4/V4.1 的“长平台 + 末段 cosine”就是这种形状。
- id: bernstein2024anthology
  title: "Old Optimizer, New Norm: An Anthology"
  a: Jeremy Bernstein, Laker Newhouse (2024). arXiv:2409.20325v2
  url: https://arxiv.org/abs/2409.20325
  finding: 把 Adam、Shampoo、Prodigy 解释为某种范数下的最速下降；命题 4：Frobenius 意义下离 G 最近的半正交矩阵是 UVᵀ；命题 5：谱范数下的最速下降方向是 −UVᵀ（论文的对象是不累积的 Shampoo）；附录说明牛顿–舒尔茨三次迭代改编自 Higham。
  matters: 本课 §6 “为什么要正交化”里有定理支撑的那部分几何；Muon 与它的对应要同时引 Jordan 的博客。
- id: liu2025muon
  title: "Muon is Scalable for LLM Training"
  a: Jingyuan Liu, Jianlin Su, Xingcheng Yao, et al. (2025). arXiv:2502.16982v1
  url: https://arxiv.org/abs/2502.16982
  finding: 给 Muon 加上 AdamW 式权重衰减，并把更新按 0.2·√max(A,B) 缩放使 RMS 与 AdamW 一致（引理 1 给出正交化更新的 RMS），从而直接复用 AdamW 超参数；在计算最优训练下约 2 倍于 AdamW 的计算效率。
  matters: 本课 §7 的形状缩放、权重衰减的必要性与“约 2 倍”的实测都以它为出处；DeepSeek-V4 明言沿用它的做法。
- id: jordan2024muon
  kind: artefact
  title: "Muon: An optimizer for hidden layers in neural networks (blog post)"
  a: Keller Jordan (2024-12-08). kellerjordan.github.io，读取于 2026-09-24
  url: https://kellerjordan.github.io/posts/muon/
  finding: 定义 Muon：对隐藏层二维参数的 SGD 动量更新做 5 步牛顿–舒尔茨近似正交化，系数 (3.4445, −4.7750, 2.0315) 是在极限落在 [0.7, 1.3] 的约束下让 φ′(0) 最大化得到的；给出 FLOP 开销上界 Tm/B，并推测正交化放大了更新里的“稀有方向”。
  matters: 本课 §6、§7 的 Muon 定义、系数来历、开销分析与“推测”部分的原始出处；没有对应论文，只能按制品引用。
- id: muon-code
  kind: artefact
  title: "KellerJordan/Muon — muon.py"
  a: Keller Jordan et al. GitHub repository (master branch), read 2026-09-24
  url: https://github.com/KellerJordan/Muon
  finding: 参考实现：BF16 下先做 Frobenius 归一化再做 5 步五次迭代，注释说输出约为 US′Vᵀ、S′ 近似在 [0.5, 1.5] 里均匀；EMA 形式的 Nesterov 动量，更新按 √max(1, 行/列) 缩放，默认 lr 0.02、动量 0.95、权重衰减 0。
  matters: 本课用它说明论文与代码在动量写法、形状缩放上的差别，以及“近似正交化”的实际程度。
```
