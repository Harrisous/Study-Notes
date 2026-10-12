---
title: M06 · 微积分与反向传播：从导数到梯度高速公路
short: 反向传播推导
kicker: 数学补课 · M06
description: 导数即局部线性近似、偏导与梯度、多元链式法则与带形状的雅可比、计算图上的反向传播（逐个数字走一遍）、softmax 的雅可比与交叉熵梯度 p − y、线性层的反向 dW = XᵀdY 与“训练约 6N 次运算”、残差 x + f(x) 为什么是梯度高速公路，配可手算的例题与练习。
lede: 训练一个大模型，每一步都在回答同一个问题：把几亿个参数各自挪一点点，损失会怎样变？这一课从“导数就是放大后看到的那条直线”讲起，一路讲到反向传播：为什么一次反向就能得到所有参数的偏导数，softmax 加交叉熵的梯度为什么只是 **p − y**，一个线性层的反向为什么恰好是两次和前向一样大的矩阵乘法，以及残差连接 $\mathbf x+f(\mathbf x)$ 为什么能让梯度穿过一百层而不消失。
meta: 预计阅读 180 分钟，练习另需约 45 分钟 · 前置：M01；高中导数 · 服务于第 1、2、16 课
index: index.html
prev: m05-information-theory.html | M05 · 熵与 KL
next: m07-optimization.html | M07 · 优化器
---

## 这一课回答什么问题

第 2 课说“训练就是沿着负梯度走一小步”；第 1 课说“残差连接给梯度修了一条高速公路”；第 16 课整课都在担心残差流上的梯度会不会爆炸。这三句话背后是同一套数学：导数、链式法则、雅可比矩阵、反向传播。这一课把它从头讲一遍，每一步都配一个可以用计算器复核的数字。

读完这一课，下面这些问题你都能自己回答：

- 导数到底是什么？为什么说它是“局部线性近似”？这和“沿负梯度走一步，损失就会下降”有什么关系？
- GPT-2 small 有 1.24 亿个参数。为什么一次反向传播（代价只是前向的两三倍）就能算出全部 1.24 亿个偏导数，而不必把模型跑 1.24 亿遍？
- softmax 的各个输出互相牵连，它的导数是一个矩阵。可是交叉熵对 logits 的梯度为什么偏偏化简成简单的 $\mathbf p-\mathbf y$？
- 线性层 $Y=XW$ 的反向传播为什么是 $dW=X^\top dY$？为什么训练时每个 token 大约要做 $6N$ 次浮点运算，而推理只要 $2N$？
- 残差连接 $\mathbf x+f(\mathbf x)$ 求导后多出来的那个单位矩阵，为什么能让梯度穿过一百层？它在什么情况下也救不了？

## 这一课在科技树上的位置

这一课直接服务三课。[第 1 课](../deepseek/lesson-01-transformer-anatomy.html)：残差流的“梯度高速公路”(1.27)、softmax 的饱和、前向约 $2N$ 次运算。[第 2 课](../deepseek/lesson-02-gpt-training.html)：梯度从交叉熵对 logits 的导数 $\mathbf p-\mathbf y$ 开始往回流 (2.7)，训练算力 $C\approx6ND$。[第 16 课](../deepseek/lesson-16-mhc.html)：残差流上的连乘——Hyper-Connections 为什么会让增益爆炸，mHC 为什么要约束残差上的混合矩阵。

它也被很多课顺手借用：第 3 课奖励模型的 $-\log\sigma(r_w-r_l)$ 要对 sigmoid 求导；第 7、10 课的门控用 softmax，第 11 课把 V3 的门控换成 sigmoid 时拿两者的雅可比作比较，都要用到 softmax 的雅可比；第 12 课的 Zero Bubble 把线性层的反向拆成“对输入”和“对权重”两半，拆的正是本课 §6 的两次矩阵乘法；第 13 课要对 $\log\sigma$ 求导；第 14 课 indexer 的 KL 损失对 logits 的梯度也是“预测减目标”。上一课 [M05](m05-information-theory.html) 定义了交叉熵，这一课求它的导数；下一课 [M07](m07-optimization.html) 拿着梯度去更新参数。

## 读完你能……

1. 把导数读成“局部线性近似”，手算常见函数（含 sigmoid、log、exp）的导数，用中心差分核对，并说出差分的步长为什么不能太小。
2. 求多元函数的偏导数与梯度，说明负梯度为什么是下降最快的方向，写出对向量、矩阵求导的结果（结果与参数同形状）。
3. 写出复合函数每一步的雅可比，按形状把它们乘起来；解释训练为什么用反向模式（向量–雅可比积）而不用前向模式。
4. 在一张计算图上手工完成前向与反向，正确处理“一个变量被用了两次”的情形。
5. 推导 softmax 的雅可比 $\diag(\mathbf p)-\mathbf p\mathbf p^\top$ 与交叉熵梯度 $\mathbf p-\mathbf y$；推导线性层的 $dX$、$dW$、$d\mathbf b$ 并用形状检查；数出“反向约是前向的两倍”。
6. 用 $(I+J)$ 的连乘解释残差为什么保住梯度，并指出它的前提：分支要小，捷径上不能乘东西。

::: note 本课的记号
- 标量用小写斜体 $x$；向量用粗体小写，默认是列向量，$\mathbf x\in\R^n$；矩阵用大写 $W$。形状写成 $[m,n]$（$m$ 行 $n$ 列），与 NumPy / PyTorch 的 `shape` 一致。
- 损失 $\mathcal L$ 永远是**一个标量**。$\bar v$（读作 “v bar”）表示损失对某个中间量 $v$ 的导数 $\partial\mathcal L/\partial v$，与 $v$ 同形状；代码里常写成 `dv` 或 `v.grad`，本课讲线性层时也写 $dW$、$dX$。
- 梯度 $\nabla f$ 是与输入同形状的列向量；雅可比 $J$ 的形状是 [输出维数, 输入维数]。
- 其余沿用整门课的约定：$\mathbf z$ 是 logits，$\mathbf p=\softmax(\mathbf z)$，$\mathbf y$ 是独热标签，$V$ 是词表大小，$d$ 是隐藏维度，$L$ 是层数（损失用花体 $\mathcal L$，不会混）。
:::

[[legend]]

[[toc]]

# 导数：局部线性近似 | 导数是什么、怎样从定义推出来、怎样用数值核对 | 25

先把问题摆出来。训练模型时，我们手里有一个损失 $\mathcal L(\theta)$：参数 $\theta$ 取某组值，喂进一批数据，得到一个数，表示模型“猜得有多差”（[M05](m05-information-theory.html) 的交叉熵）。优化器每一步只做一件事：把参数挪一点点，让这个数变小。所以它需要知道：**把某一个参数增加一个很小的量 $h$，损失大约会变多少？** 如果答案是“大约 $+3h$”，就该让这个参数减小；如果是“大约 $-0.5h$”，就该让它增大。“输入变一点，输出跟着变多少倍”的这个比率，就是导数。这一页只看一个输入、一个输出的函数，把导数讲透；下一页再推广到很多个参数。

## 导数的三层意思

**含义。** 函数 $f$ 在点 $x$ 的导数 $f'(x)$ 是一个数：在 $x$ 附近，输入每增加一点，输出大约增加它的 $f'(x)$ 倍。

**理解。** 拿放大镜看一条光滑的曲线。放大得越多，视野里的那一小段就越像一条直线；导数就是这条“看起来的直线”的斜率。图 M6.1 把 sigmoid 曲线在 $z=1$ 附近连续放大两次：窗口宽 4 时，曲线和直线还明显分得开；窗口宽 0.04 时，两者的差只剩窗口宽度的 0.046%。这就是“局部线性”：凑得足够近，任何光滑函数看起来都是一条直线。这个画面也告诉你它在哪里失效：函数有“折角”的地方（比如下文的 ReLU 在 0 处），放大多少倍都还是一个角，看不成一条直线，那里就没有导数。

**数学。** 这条直线的斜率用极限来定义。

::: pre 极限与 o(h) | 熟悉极限记号的可以跳过
$\lim_{h\to0}g(h)=c$ 读作“$h$ 趋于 0 时 $g(h)$ 趋于 $c$”：只要 $h$ 足够小，$g(h)$ 与 $c$ 的差就能小于你事先指定的任何正数。本课还会用一个记号 $o(h)$（读作“小 o h”）：它代表某个比 $h$ **更快**趋于 0 的量，也就是 $o(h)/h\to0$。例如 $h^2$ 是 $o(h)$：$h=0.01$ 时 $h^2/h=0.01$，$h=0.001$ 时只剩 $0.001$。
:::

::: eq 导数的定义 | (M6.1)
$$f'(x)=\frac{df}{dx}(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$
分子是输出的变化，分母是输入的变化，比值是过两点的“割线”的斜率；$h\to0$ 时割线转成切线。
:::

把 (M6.1) 换一种写法，就得到这一课用得最多的形式：

::: eq 局部线性近似 | (M6.2)
$$f(x+h)=f(x)+f'(x)\,h+o(h)$$
输出的变化 ≈ 导数 × 输入的变化，剩下的误差比 $h$ 小得更快。
:::

(M6.2) 和 (M6.1) 说的是同一件事：(M6.1) 说 $\frac{f(x+h)-f(x)}{h}-f'(x)\to0$，两边乘上 $h$，就是说误差 $f(x+h)-f(x)-f'(x)h$ 是 $o(h)$。换成这种写法的好处是，导数从一个极限变成了一个**预测**：知道 $f(x)$ 和 $f'(x)$，不用重新计算 $f$，就能预测输入挪一点之后的输出。

::: ex 用导数预测函数值
$f(x)=x^2$，它的导数是 $f'(x)=2x$（下面马上推导）。在 $x=3$：$f(3)=9$，$f'(3)=6$。

- $h=0.01$：(M6.2) 预测 $f(3.01)\approx9+6\times0.01=9.06$；真值 $3.01^2=9.0601$，误差 $0.0001=h^2$。
- $h=0.001$：预测 $9.006$；真值 $9.006001$，误差 $10^{-6}$。

$h$ 缩小 10 倍，误差缩小 100 倍。误差与 $h$ 的比值从 0.01 降到 0.001，也在趋于 0，这正是“误差是 $o(h)$”的样子。
:::

::: fig figures/m06-tangent-zoom.svg | 图 M6.1 放大之后，曲线和它的切线越来越分不开
**看每个窗口下方的红字：窗口每放大 10 倍，误差占窗口宽度的比例也缩小约 10 倍（4.3% → 0.46% → 0.046%）。** 实线是 $\sigma(z)$，蓝色虚线是它在 $z_0=1$ 处的切线 $\sigma(1)+\sigma'(1)(z-1)$，淡红色是两者之差。切线的斜率 0.1966 就是 $\sigma'(1)$，下文会推出它等于 $\sigma(1)\big(1-\sigma(1)\big)$。所有数值由脚本逐点计算。
:::

## 从定义推出常用导数

有了 (M6.1)，常用函数的导数都能推出来。先做最简单的一个示范。

::: deriv (x²)′ = 2x
$\dfrac{(x+h)^2-x^2}{h}=\dfrac{x^2+2xh+h^2-x^2}{h}$ || 按 (M6.1) 写出割线斜率，把平方展开
$=2x+h$ || 约掉 $h$（此时 $h\ne0$）
$\to2x\quad(h\to0)$ || 取极限，含 $h$ 的项消失
:::

再做两个要借助基本极限的例子。$\frac{e^h-1}{h}\to1$ 与 $\frac{\ln(1+h)}{h}\to1$ 是微积分课本里的两个基本极限（取 $h=0.001$，两者分别是 1.0005 与 0.9995，可以用计算器感受一下）；有了它们，$e^x$ 与 $\ln x$ 的导数各只要一行：

::: deriv (eˣ)′ = eˣ 与 (ln x)′ = 1/x
$\dfrac{e^{x+h}-e^x}{h}=e^x\cdot\dfrac{e^h-1}{h}\to e^x$ || 提出公因子 $e^x$，再用 $\frac{e^h-1}{h}\to1$
$\dfrac{\ln(x+h)-\ln x}{h}=\dfrac1x\cdot\dfrac{\ln(1+k)}{k}\to\dfrac1x,\quad k=\dfrac hx$ || $\ln(x+h)-\ln x=\ln(1+h/x)$；$h\to0$ 时 $k\to0$，用 $\frac{\ln(1+k)}{k}\to1$
:::

下表汇总本课要用的几个导数。$\sin$、$\cos$ 的两行还要用到 $\frac{\sin h}{h}\to1$；$\tanh$ 那一行等讲完 sigmoid，用一次链式法则就能推出来（见本页 sigmoid 小节末尾）。

| 函数 $f(x)$ | 导数 $f'(x)$ | 备注 |
|---|---|---|
| $c$（常数） | $0$ | 不随 $x$ 变化 |
| $x^n$ | $nx^{n-1}$ | $n$ 可以是任意实数（$n$ 是负整数时要求 $x\ne0$，不是整数时要求 $x>0$），例如 $\sqrt x=x^{1/2}$ 的导数是 $\frac1{2\sqrt x}$ |
| $e^x$ | $e^x$ | 满足“导数等于自己”的函数只有 $Ce^x$ |
| $\ln x$ | $1/x$ | 自然对数，要求 $x>0$ |
| $\sin x$，$\cos x$ | $\cos x$，$-\sin x$ | §4 的计算图例题要用 |
| $\tanh x$ | $1-\tanh^2x$ | §7 的随机网络要用 |

## 三条组合规则与一元链式法则

神经网络是用简单函数拼起来的：加起来、乘起来、一层套一层。所以除了上表，只需要三条规则。**加法**：$(f+g)'=f'+g'$。**乘法**：$(fg)'=f'g+fg'$。**复合**：$\big(g(f(x))\big)'=g'\big(f(x)\big)\,f'(x)$，这条叫**链式法则**（chain rule）。三条都用局部线性近似 (M6.2) 推最透明。加法规则只要把 $f$、$g$ 的两个 (M6.2) 相加，$h$ 前面的系数就是 $f'+g'$。乘法规则多一步：

::: deriv 乘法规则 (fg)′ = f′g + fg′
$f(x+h)\,g(x+h)=\big(f+f'h+o(h)\big)\big(g+g'h+o(h)\big)$ || 两个因子各用一次 (M6.2)；$f,g,f',g'$ 都在 $x$ 处取值
$=fg+(f'g+fg')\,h+f'g'\,h^2+o(h)$ || 乘开；凡是乘了 $o(h)$ 的项仍是 $o(h)$
$=fg+(f'g+fg')\,h+o(h)$ || $h^2$ 也是 $o(h)$（见上面的前置框）
$(fg)'=f'g+fg'$ || 与 (M6.2) 对照：$h$ 前面的系数就是乘积的导数
:::

第三条是整课的主角，推法相同：

::: deriv 一元链式法则
$\Delta u=f(x+h)-f(x)=f'(x)\,h+o(h)$ || 记中间量 $u=f(x)$；输入挪 $h$，中间量挪 $\Delta u$，这是内层的 (M6.2)
$g(u+\Delta u)=g(u)+g'(u)\,\Delta u+o(\Delta u)$ || 外层函数 $g$ 在 $u$ 处的 (M6.2)
$=g(u)+g'(u)f'(x)\,h+o(h)$ || 代入第 1 行；$\Delta u/h\to f'(x)$，所以 $\abs{\Delta u}$ 不超过 $h$ 的常数倍（$f'(x)=0$ 时 $\Delta u$ 甚至更小），$o(\Delta u)$ 也就是 $o(h)$
$\big(g\circ f\big)'(x)=g'\big(f(x)\big)\,f'(x)$ || 与 (M6.2) 对照：$h$ 前面的系数就是复合函数的导数
:::

::: eq 一元链式法则 | (M6.3)
$$\frac{d}{dx}\,g\big(f(x)\big)=g'\big(f(x)\big)\cdot f'(x),\qquad\text{简写为}\quad\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}$$
变化率沿着计算链**相乘**：$x$ 挪一点，$u$ 跟着挪 $f'$ 倍；$u$ 挪一点，$y$ 跟着挪 $g'$ 倍。
:::

::: ex 链式法则算例：softplus 的导数是 sigmoid
$\mathrm{softplus}(x)=\ln(1+e^x)$。令 $u=1+e^x$，$y=\ln u$。$\frac{dy}{du}=\frac1u$，$\frac{du}{dx}=e^x$，相乘得 $\frac{e^x}{1+e^x}$；分子分母同除以 $e^x$，得 $\frac1{1+e^{-x}}$，这正是下面要讲的 sigmoid $\sigma(x)$。在 $x=0$：函数值 $\ln2=0.6931$，导数 $\frac{1}{1+1}=0.5$。用下面的 (M6.5) 取 $h=10^{-3}$ 核对，得 0.500000。
:::

## sigmoid 的导数：第 3、13 课要用

**sigmoid**（又叫 logistic 函数）$\sigma(z)=1/(1+e^{-z})$ 把任意实数压进 $(0,1)$：$\sigma(0)=0.5$，$z\to+\infty$ 时趋于 1，$z\to-\infty$ 时趋于 0。第 3 课的奖励模型用 $\sigma(r_w-r_l)$ 表示“回答 $w$ 比回答 $l$ 好”的概率，第 13 课的推导要对 $\log\sigma$ 求导。它的导数有一个非常好用的形式：

::: deriv σ′ = σ(1 − σ)
$\sigma(z)=(1+e^{-z})^{-1}$ || 写成幂的形式，外层是 $u^{-1}$，内层是 $u=1+e^{-z}$
$\sigma'(z)=-(1+e^{-z})^{-2}\cdot(-e^{-z})=\dfrac{e^{-z}}{(1+e^{-z})^2}$ || 链式法则：外层导数 $-u^{-2}$，内层导数 $-e^{-z}$
$=\dfrac{1}{1+e^{-z}}\cdot\dfrac{e^{-z}}{1+e^{-z}}$ || 拆成两个因子
$=\sigma(z)\big(1-\sigma(z)\big)$ || 第二个因子 $=\frac{(1+e^{-z})-1}{1+e^{-z}}=1-\sigma(z)$
:::

::: eq sigmoid 及其对数的导数 | (M6.4)
$$\begin{aligned}\sigma'(z)&=\sigma(z)\big(1-\sigma(z)\big)\\ \frac{d}{dz}\log\sigma(z)&=1-\sigma(z)\\ \frac{d}{dz}\log\big(1-\sigma(z)\big)&=-\sigma(z)\end{aligned}$$
后两个再用一次链式法则：$\frac{d}{dz}\log\sigma=\frac{\sigma'}{\sigma}=1-\sigma$；$\frac{d}{dz}\log(1-\sigma)=\frac{-\sigma'}{1-\sigma}=-\sigma$。本课的 $\log$ 都是自然对数。
:::

::: ex 数一数 sigmoid 的导数
- $z=0$：$\sigma=0.5$，$\sigma'=0.25$，$\frac{d}{dz}\log\sigma=0.5$。
- $z=2$：$\sigma=0.880797$，$\sigma'=0.880797\times0.119203=0.104994$，$\frac{d}{dz}\log\sigma=0.119203$。
- $z=5$：$\sigma=0.993307$，$\sigma'=0.006648$，$\frac{d}{dz}\log\sigma=0.006693$。

读出两个规律。第一，$\sigma'$ 在 $z=0$ 取最大值 $0.25$（$s(1-s)$ 在 $s=\tfrac12$ 时最大），离 0 越远越小：输入很大时 sigmoid **饱和**，导数几乎为 0。第二，$\frac{d}{dz}\log\sigma(z)=1-\sigma(z)$ 在模型已经很有把握（$\sigma\to1$）时趋于 0。第 3 课奖励模型的损失 $-\log\sigma(r_w-r_l)$ 对差值 $r_w-r_l$ 的导数是 $-(1-\sigma)$：一对已经排对、而且差距很大的回答，几乎不再提供梯度。
:::

有了 $\sigma'$，上面表格里 $\tanh$ 的那一行也只差一步。tanh 其实是把 sigmoid 横向压缩一半、纵向拉到 $(-1,1)$：

::: deriv tanh′ = 1 − tanh²（借 sigmoid 一步得到）
$\tanh x=\dfrac{e^x-e^{-x}}{e^x+e^{-x}}=\dfrac{1-e^{-2x}}{1+e^{-2x}}$ || 分子分母同除以 $e^x$
$=\dfrac{2}{1+e^{-2x}}-1=2\sigma(2x)-1$ || 记 $a=e^{-2x}$，$\frac{1-a}{1+a}=\frac{2}{1+a}-1$；再对照 $\sigma$ 的定义
$\tanh'x=2\,\sigma'(2x)\cdot2=4s(1-s),\quad s=\sigma(2x)$ || 链式法则：外层导数 $2\sigma'$，内层 $2x$ 的导数 2；再用 (M6.4)
$1-\tanh^2x=1-(2s-1)^2=4s-4s^2$ || 把第 2 行代入，展开平方
$=4s(1-s)=\tanh'x$ || 与第 3 行相同，所以 $\tanh'=1-\tanh^2$
:::

在 $x=0$，$\tanh'0=1$，是 sigmoid 最大斜率 0.25 的 4 倍；离原点远了它同样饱和，§3 的例题里 $\tanh'(2)$ 只剩 0.0707。

## 数值导数：用差分核对你的求导

**问题。** 手推的导数写错了怎么办？代码里的反向传播有 bug 怎么办？最朴素的检验是回到定义 (M6.1)，取一个小的 $h$，直接算差商。

::: eq 前向差分与中心差分 | (M6.5)
$$\frac{f(x+h)-f(x)}{h}=f'(x)+O(h),\qquad \frac{f(x+h)-f(x-h)}{2h}=f'(x)+O(h^2)$$
$O(h)$ 读作“至多与 $h$ 同量级的项”：$h\to0$ 时它的绝对值不超过 $h$ 的某个常数倍；$O(h^2)$ 同理。中心差分在 $x$ 两侧对称取点，一阶误差互相抵消 [@baydin2018]。
:::

为什么中心差分准得多？要多展开一项才看得出来。

::: pre 二阶展开 | 熟悉泰勒公式的可以跳过
(M6.2) 只保留到一次项。对可以多次求导的函数，还能继续往下写：$f(x+h)=f(x)+f'(x)h+\tfrac12f''(x)h^2+\tfrac16f'''(x)h^3+\cdots$。$f''$ 是“导数的导数”，量的是曲线弯得多厉害；$f'''$ 同理。本页只用它比较两种差分的误差。
:::

::: deriv 两种差分的误差
$\dfrac{f(x+h)-f(x)}{h}=f'(x)+\tfrac12f''(x)\,h+\cdots$ || 把二阶展开代入前向差分，约掉 $h$：误差的首项正比于 $h$
$f(x+h)-f(x-h)=2f'(x)\,h+\tfrac13f'''(x)\,h^3+\cdots$ || 把 $h$ 与 $-h$ 的两个展开相减：偶数次项（含 $f''$ 的那一项）正负抵消
$\dfrac{f(x+h)-f(x-h)}{2h}=f'(x)+\tfrac16f'''(x)\,h^2+\cdots$ || 除以 $2h$：误差的首项正比于 $h^2$
:::

::: ex 两种差分在 eˣ 上的表现
$f(x)=e^x$，在 $x=1$ 处真导数是 $e=2.718282$。取 $h=10^{-3}$：前向差分误差 $1.36\times10^{-3}$（与 $\tfrac12eh=1.36\times10^{-3}$ 吻合），中心差分误差 $4.53\times10^{-7}$（与 $\tfrac16eh^2=4.53\times10^{-7}$ 吻合）。同样是两次函数求值，中心差分准了约三千倍。
:::

**但是 $h$ 不能无限小。** 计算机里的数只有有限位有效数字（[M08](m08-floating-point.html)）。$h$ 很小时 $f(x+h)$ 与 $f(x)$ 几乎相等，相减把前面相同的有效数字全部抵掉，剩下的主要是舍入噪声；再除以很小的 $h$，噪声被放大。所以数值导数的误差由两部分组成：**截断误差**随 $h$ 减小而减小，**舍入误差**随 $h$ 减小而增大 [@baydin2018]。

::: fig figures/m06-finite-diff.svg | 图 M6.2 数值导数的误差随步长 h 呈 V 形 | wide
**看两条曲线的谷底：$h$ 太大时截断误差占主导（右侧的直线），太小时舍入误差占主导（左侧的锯齿），最好的 $h$ 在中间。** 函数是截断的 logistic 映射 $f(x)=64x(1-x)(1-2x)^2(1-8x+8x^2)^2$，取 $x_0=0.2$（真导数 9.0661）。在 float64 下，前向差分最好只能到约 $6\times10^{-7}$（$h\approx10^{-8}$），中心差分约 $3\times10^{-9}$（$h\approx10^{-6}$），离 float64 自身约 $10^{-16}$ 的精度差得很远。据《Automatic Differentiation in Machine Learning: a Survey》图 3 重绘，全部数值由脚本重新计算 [@baydin2018]。
:::

::: note 梯度检查（gradient check）
写完一个新层的反向传播，通常用中心差分逐个核对几个参数的偏导数：用 float64，$h$ 取 $10^{-6}$ 到 $10^{-5}$ 左右（图 M6.2 的谷底附近），比较相对误差 $\abs{a-b}/\max(\abs a,\abs b)$。本课的每个算例都这样核对过。它只能用来检验，不能用来训练：一个有 $n$ 个参数的模型，用差分求一次完整梯度要做 $2n$ 次函数求值 [@baydin2018]。GPT-2 small 有 124,439,808 个参数（第 1 课），那就是约 2.5 亿次前向。§3 和 §4 会讲，反向传播只花大约两三次前向的代价就能得到全部偏导数。
:::

::: warn 失败模式：不可导的点与饱和
ReLU 函数 $\max(0,x)$ 在 $x=0$ 处左边斜率是 0、右边斜率是 1，(M6.1) 的极限不存在，导数没有定义。激活值恰好等于 0 的情形在训练中很少见，实现里对这一点约定取某一侧的值即可（笔者的说明，各框架具体取哪一侧这里不展开）。更要紧的是**饱和**：在 sigmoid、tanh 的平坦区和 ReLU 的负半轴，导数是 0 或接近 0，梯度从这里几乎传不过去。§7 会看到“每层乘一个小于 1 的数”在几十层之后变成什么。
:::

::: key 本页要点
- 导数是局部线性近似的斜率：$f(x+h)=f(x)+f'(x)h+o(h)$。它把“输入挪一点，输出变多少”变成一个可以直接用的预测。
- 链式法则：沿着计算链，变化率相乘，$\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}$。
- $\sigma'=\sigma(1-\sigma)\le\tfrac14$，$\frac{d}{dz}\log\sigma=1-\sigma$。
- 中心差分的误差是 $O(h^2)$，是检查求导的标准工具；$h$ 太小会被舍入误差吞掉；它太慢，不能用来训练。
:::

::: note 这一页用在哪里
[第 3 课 §2](../deepseek/lesson-03-alignment-rlhf.html#p2)：奖励模型的损失 $-\log\sigma(r_w-r_l)$，求导用 (M6.4)。[第 13 课 §3](../deepseek/lesson-13-r1-reasoning.html#p3)：一步 GRPO 的算例对 $\log\sigma$ 求导。[第 1 课 §4](../deepseek/lesson-01-transformer-anatomy.html#p4)：softmax 的饱和让梯度变小，是本页 sigmoid 饱和的多维版本（它的雅可比在本课 §5）。
:::

# 偏导数与梯度 | 很多个输入：偏导数、梯度、方向导数与“最陡下降” | 25

一个语言模型的损失是上亿个参数的函数。上一页的导数只处理一个输入；现在的问题变成：**参数有 $n$ 个，应该让每一个往哪边挪、挪多少？** 这一页把导数推广到多个输入，得到梯度，并说明为什么“沿负梯度走”是让损失下降最快的走法。

## 偏导数：一次只动一个输入

**含义。** 多元函数 $f(x_1,\dots,x_n)$ 对 $x_i$ 的**偏导数**（partial derivative）$\partial f/\partial x_i$，是把其余输入都冻结、只让 $x_i$ 变化时的导数。

**理解。** 想象调音台上的一排推子。偏导数回答的是：只推第 $i$ 个推子、别的都不碰，音量变化得有多快。弯弯的符号 $\partial$ 提醒你：此刻其余变量都被当成常数。

**数学。** 把 (M6.1) 里的“挪 $h$”换成“只沿第 $i$ 个坐标挪 $h$”：

::: eq 偏导数 | (M6.6)
$$\frac{\partial f}{\partial x_i}(\mathbf x)=\lim_{h\to0}\frac{f(\mathbf x+h\,\mathbf e_i)-f(\mathbf x)}{h}$$
$\mathbf e_i\in\R^n$ 是第 $i$ 个分量为 1、其余为 0 的单位向量。所以求偏导的方法就是：把别的变量当常数，按一元函数求导。
:::

::: ex 求两个偏导数
$f(x,y)=x^2y+3y$，在点 $(2,1)$。对 $x$ 求偏导时把 $y$ 当常数：$\partial f/\partial x=2xy=4$。对 $y$ 求偏导时把 $x$ 当常数：$\partial f/\partial y=x^2+3=7$。含义：从 $(2,1)$ 出发，只把 $x$ 增加 0.01，$f$ 大约增加 0.04；只把 $y$ 增加 0.01，$f$ 大约增加 0.07。
:::

## 梯度：把所有偏导数排成一个向量

如果所有输入**同时**挪一点呢？设位移是 $\boldsymbol\Delta=(\Delta_1,\dots,\Delta_n)$。对光滑的函数，一阶近似下各个方向的贡献可以分别算、再加起来：$f(\mathbf x+\boldsymbol\Delta)\approx f(\mathbf x)+\sum_i\frac{\partial f}{\partial x_i}\Delta_i$。理由是把这次位移拆成 $n$ 小步、每步只动一个坐标：第 $i$ 步让 $f$ 变化约 $\frac{\partial f}{\partial x_i}\Delta_i$。严格说，这个偏导数是在途中的点取值，不是在 $\mathbf x$ 处；只要偏导数随位置连续变化，两者只差一个小量，乘上 $\Delta_i$ 以后比 $\norm{\boldsymbol\Delta}$ 小得更快。拿上例核对：从 $(2,1)$ 同时挪 $(0.01,0.01)$，预测 $f$ 增加 $4\times0.01+7\times0.01=0.11$；真值 $f(2.01,1.01)-f(2,1)=7.110501-7=0.110501$，差 $5\times10^{-4}$，是二阶小量。

这个和正是一个点积（[M01 §1](m01-vectors-matrices.html#p1)），于是把偏导数排成一个向量：

::: eq 梯度与多元局部线性近似 | (M6.7)
$$\nabla f(\mathbf x)=\begin{pmatrix}\partial f/\partial x_1\\ \vdots\\ \partial f/\partial x_n\end{pmatrix}\in\R^n,\qquad f(\mathbf x+\boldsymbol\Delta)=f(\mathbf x)+\nabla f(\mathbf x)^\top\boldsymbol\Delta+o(\norm{\boldsymbol\Delta})$$
梯度（gradient）$\nabla f$ 与 $\mathbf x$ 同形状；输出的变化 ≈ 梯度与位移的点积。
:::

**含义。** 梯度 $\nabla f(\mathbf x)\in\R^n$ 是一个与输入同形状的向量，第 $i$ 个分量是 $\partial f/\partial x_i$。**理解。** 把 $f$ 想成一张地形图上的海拔：在每一点，梯度是一支箭头，指向最陡的上坡方向，箭头的长度就是这个方向上的坡度（下一小节证明）。训练时这张“地形”有上亿个维度，画不出来，但每一点的箭头照样算得出来。

(M6.7) 的第二式在数学上正是“$f$ 在 $\mathbf x$ 处**可微**”的定义。它比“每个偏导数都存在”要求更强，但本课遇到的函数（多项式、$\exp$、$\log$、sigmoid、softmax 以及它们的复合）在定义域内都满足它；例外只有 ReLU 这类函数的折点。

## 方向导数：负梯度为什么下降最快

沿任意一个单位方向 $\mathbf u$（$\norm{\mathbf u}=1$）走一小步 $t$，由 (M6.7)，$f(\mathbf x+t\mathbf u)\approx f(\mathbf x)+t\,\nabla f^\top\mathbf u$。所以沿 $\mathbf u$ 方向的变化率是：

::: eq 方向导数 | (M6.8)
$$D_{\mathbf u}f(\mathbf x)=\nabla f(\mathbf x)^\top\mathbf u=\norm{\nabla f(\mathbf x)}\cos\varphi$$
$\varphi$ 是 $\mathbf u$ 与梯度的夹角。第二个等号是点积的几何形式（[M01](m01-vectors-matrices.html#p1) 的 (M1.3)；那里把夹角记作 $\theta$，本页把 $\theta$ 留给 (M6.9) 里的模型参数）。
:::

$\cos\varphi$ 只能在 $-1$ 到 $1$ 之间取值，由此从 (M6.8) 读出三件事（设 $\nabla f(\mathbf x)\ne\mathbf 0$）。$\varphi=0$，即 $\mathbf u$ 与梯度同向时，变化率最大，等于 $\norm{\nabla f}$：**梯度方向是上升最快的方向**，梯度的长度就是那个最快的速率。$\varphi=180^\circ$ 时变化率最小，等于 $-\norm{\nabla f}$：**负梯度方向下降最快**。$\varphi=90^\circ$ 时变化率为 0：与梯度垂直的方向上 $f$ 暂时不变，这些方向正是等高线（$f$ 取常数的曲线）的切线方向，所以在梯度不为零的每一点，**梯度都垂直于等高线**。（梯度为零的点，比如下图 $f$ 的最低点原点，等高线缩成一个点，谈不上方向。）第一条也可以不借助夹角，直接由柯西–施瓦茨不等式 $\abs{\nabla f^\top\mathbf u}\le\norm{\nabla f}\norm{\mathbf u}$ 得到（[M01](m01-vectors-matrices.html) 练习 7）。

::: fig figures/m06-gradient-contours.svg | 图 M6.3 等高线、梯度与两条梯度下降路径 | wide
**看点 P 处的红色箭头：梯度 $(4,6)$ 垂直于经过 P 的等高线（粗灰线 $f=7$），指向 $f$ 增大的一侧。** 红色虚线是等高线在 P 处的切线，沿它走的方向导数是 0。函数是 $f(x,y)=x^2+3y^2$，横纵坐标比例相同，所以“垂直”在图上看起来也是垂直的。蓝色路径是学习率 $\eta=0.1$ 的梯度下降，平稳地滑向原点；紫色路径 $\eta=0.3$ 在 $y$ 方向每步乘以 $1-6\eta=-0.8$，来回摆动。右侧的每个数都可以手算复核，路径上的点由脚本逐步计算。
:::

::: ex 方向导数与一步梯度下降
$f(x,y)=x^2+3y^2$，在 P $=(2,1)$：$f=7$，$\nabla f=(2x,6y)=(4,6)$，$\norm{\nabla f}=\sqrt{52}=7.2111$。

- 沿 $\mathbf u=(1,0)$：变化率 $4$。沿 $\mathbf u=(0.6,0.8)$：$4\times0.6+6\times0.8=7.2$，已经很接近上限。沿 $\mathbf u=\nabla f/\norm{\nabla f}=(0.5547,0.8321)$：恰好 $7.2111$，任何方向都不会更大。
- 梯度下降一步，$\eta=0.1$：新点 $(2,1)-0.1\times(4,6)=(1.6,0.4)$，$f$ 从 7 降到 $1.6^2+3\times0.4^2=3.04$。
- (M6.7) 的线性近似预测的是 $7-0.1\times\norm{\nabla f}^2=7-5.2=1.8$，比真值 3.04 乐观了 1.24。差出来的这部分来自曲率（二阶项）：线性近似只对“小步”可靠，步子一大就不准了。$\eta$ 取多大合适，是 [M07](m07-optimization.html) 的主题。
:::

**“最陡”是拿一把尺子量出来的。** 上面比较的是所有满足 $\norm{\mathbf u}=1$ 的方向，而 $\norm{\cdot}$ 是普通的欧氏长度。所以准确的说法是：**用欧氏长度量步子时**，负梯度方向下降最快。换一把尺子，最陡的方向就会变：把参数看成矩阵、用谱范数量步长时，最陡的方向不再是负梯度本身，[M07 §6](m07-optimization.html#p6) 的 Muon 走的正是这个方向。这也是地形图比喻失真的地方：地图默认往东走一米和往北走一米代价相同；参数空间里并没有这样一把天生的尺子，选哪一把是优化器设计的一部分。

::: eq 梯度下降的一步 | (M6.9)
$$\theta\leftarrow\theta-\eta\,\nabla\mathcal L(\theta),\qquad \mathcal L\big(\theta-\eta\nabla\mathcal L(\theta)\big)=\mathcal L(\theta)-\eta\,\norm{\nabla\mathcal L(\theta)}^2+o(\eta)$$
$\eta>0$ 是学习率。只要梯度不为零、$\eta$ 足够小，一阶项保证损失下降。
:::

## 对向量、矩阵求导：结果与参数同形状

反向传播里最常见的偏导数，是一个标量对一整个向量或矩阵求导。约定只有一条：**结果与被求导的对象同形状**，每个位置放“损失对这个元素的偏导数”。对一个 $[768,3072]$ 的权重矩阵求导，得到的也是一个 $[768,3072]$ 的矩阵，正好可以和权重逐元素相减。下面四个结果在后面反复出现：

::: eq 四个常用的向量 / 矩阵导数 | (M6.10)
$$\begin{aligned}\nabla_{\mathbf x}(\mathbf a^\top\mathbf x)&=\mathbf a, &\qquad \nabla_{\mathbf x}\norm{\mathbf x}^2&=2\mathbf x,\\ \nabla_{\mathbf x}(\mathbf x^\top A\mathbf x)&=(A+A^\top)\mathbf x, & \frac{\partial(\mathbf u^\top W\mathbf v)}{\partial W}&=\mathbf u\mathbf v^\top\end{aligned}$$
$\mathbf a,\mathbf x\in\R^n$，$A\in\R^{n\times n}$；$W\in\R^{m\times n}$，$\mathbf u\in\R^m$，$\mathbf v\in\R^n$，结果 $\mathbf u\mathbf v^\top$ 与 $W$ 同为 $[m,n]$。
:::

它们都可以用同一个办法推：把标量写成带下标的求和，对某一个元素求偏导，再把结果排回原来的形状。以第三个和第四个为例：

::: deriv 两个矩阵导数
$\mathbf x^\top A\mathbf x=\sum_{i}\sum_{j}x_iA_{ij}x_j$ || 按矩阵乘法 (M1.5) 展开成双重求和
$\dfrac{\partial}{\partial x_k}\sum_{i,j}x_iA_{ij}x_j=\sum_jA_{kj}x_j+\sum_ix_iA_{ik}$ || $x_k$ 出现在 $i=k$ 与 $j=k$ 两处，用乘法规则各求一次再相加
$=(A\mathbf x)_k+(A^\top\mathbf x)_k$ || 第一项是 $A$ 的第 $k$ 行点 $\mathbf x$，第二项是 $A$ 的第 $k$ 列点 $\mathbf x$
$\mathbf u^\top W\mathbf v=\sum_i\sum_ju_iW_{ij}v_j$ || 同样展开
$\dfrac{\partial(\mathbf u^\top W\mathbf v)}{\partial W_{ij}}=u_iv_j$ || 每个 $W_{ij}$ 只出现一次，系数是 $u_iv_j$；排回 $[m,n]$ 的形状就是外积 $\mathbf u\mathbf v^\top$
:::

::: ex 不对称的 A
$A=\begin{pmatrix}1&2\\0&3\end{pmatrix}$，$\mathbf x=(1,-1)$。$A\mathbf x=(-1,-3)$，$\mathbf x^\top A\mathbf x=-1+3=2$。梯度 $(A+A^\top)\mathbf x=\begin{pmatrix}2&2\\2&6\end{pmatrix}\begin{pmatrix}1\\-1\end{pmatrix}=(0,-4)$。中心差分核对：$(0.000000,-4.000000)$。
:::

::: err 把 ∇(xᵀAx) 记成 2Ax
只有 $A$ 对称（$A=A^\top$）时 $(A+A^\top)\mathbf x$ 才等于 $2A\mathbf x$。上例中 $2A\mathbf x=(-2,-6)$，和正确答案 $(0,-4)$ 差得很远。$\nabla\norm{\mathbf x}^2=2\mathbf x$ 是 $A=I$ 的特例，所以没问题。
:::

[M01](m01-vectors-matrices.html#p1) 提过点积对查询的导数 $\partial(\mathbf q^\top\mathbf k)/\partial\mathbf q=\mathbf k$，就是 (M6.10) 第一式取 $\mathbf a=\mathbf k$。最后一式 $\partial(\mathbf u^\top W\mathbf v)/\partial W=\mathbf u\mathbf v^\top$ 是线性层反向传播的种子：§6 会看到，一个 batch 的 $dW$ 就是许多个这样的外积之和。

::: key 本页要点
- 偏导数：冻结其余变量，按一元函数求导。梯度把全部偏导数排成与输入同形状的向量，$f(\mathbf x+\boldsymbol\Delta)\approx f(\mathbf x)+\nabla f^\top\boldsymbol\Delta$。
- 方向导数 $\nabla f^\top\mathbf u=\norm{\nabla f}\cos\varphi$：梯度方向上升最快，负梯度方向下降最快，与梯度垂直的方向（等高线）不变。
- 梯度下降 $\theta\leftarrow\theta-\eta\nabla\mathcal L$ 的“保证下降”只在小步时成立。
- 标量对向量、矩阵求导，结果与被求导对象同形状；$\partial(\mathbf u^\top W\mathbf v)/\partial W=\mathbf u\mathbf v^\top$ 是 $dW$ 的雏形。
:::

::: note 这一页用在哪里
[第 2 课 §5](../deepseek/lesson-02-gpt-training.html#p5)：每一步训练就是 (M6.9) 加上 AdamW 的修正。[M07](m07-optimization.html)：学习率、动量、Adam 都是在“怎样用梯度走这一步”上做文章。
:::

# 链式法则与雅可比 | 向量进、向量出：雅可比矩阵、多元链式法则，以及训练为什么“从后往前”乘 | 25

神经网络的一层把一个向量变成另一个向量：GPT-2 small 的 FFN 第一层把 768 维的隐藏状态变成 3072 维（[第 1 课 §5](../deepseek/lesson-01-transformer-anatomy.html#p5)）。这样的函数的“导数”是什么？整个网络是几十个这样的函数一层套一层，一元的链式法则 (M6.3) 又该怎样推广？这一页回答这两个问题，还要回答一个工程上更要紧的问题：同一串导数相乘，**从哪一头开始乘**，代价可以差出几百倍。

## 雅可比：向量函数的导数是一个矩阵

**含义。** 函数 $\mathbf f:\R^n\to\R^m$ 在点 $\mathbf x$ 的**雅可比矩阵**（Jacobian matrix）$J_{\mathbf f}(\mathbf x)$ 是一个 $m\times n$ 的矩阵，第 $i$ 行第 $j$ 列放着 $\partial f_i/\partial x_j$：第 $j$ 个输入挪一点，第 $i$ 个输出变多少。

**理解。** 把 (M6.7) 对每一个输出分量各写一遍，再排成一列，就是一个“矩阵乘向量”。所以在 $\mathbf x$ 附近，$\mathbf f$ 看起来就是一个线性映射（[M01 §2](m01-vectors-matrices.html#p2) 说的“矩阵是一台变换向量的机器”），雅可比就是这台“局部机器”的矩阵。它的第 $i$ 行是第 $i$ 个输出的梯度（横过来写），第 $j$ 列是“只推第 $j$ 个推子时，所有输出一起怎么动”。

**数学。**

::: eq 雅可比矩阵与局部线性近似 | (M6.11)
$$J_{\mathbf f}(\mathbf x)=\begin{pmatrix}\frac{\partial f_1}{\partial x_1}&\cdots&\frac{\partial f_1}{\partial x_n}\\ \vdots&\ddots&\vdots\\ \frac{\partial f_m}{\partial x_1}&\cdots&\frac{\partial f_m}{\partial x_n}\end{pmatrix}\in\R^{m\times n},\qquad \mathbf f(\mathbf x+\boldsymbol\Delta)=\mathbf f(\mathbf x)+J_{\mathbf f}(\mathbf x)\,\boldsymbol\Delta+o(\norm{\boldsymbol\Delta})$$
形状永远是 [输出维数, 输入维数]。$m=1$ 时它只有一行，正好是梯度的转置：$J_f=\nabla f^\top$。
:::

三种雅可比在神经网络里反复出现，每种一句话就能说清：

- **线性层** $\mathbf f(\mathbf x)=W\mathbf x+\mathbf b$：$J=W$。线性函数的最佳线性近似就是它自己，偏置 $\mathbf b$ 不影响导数。
- **逐元素函数** $\mathbf f(\mathbf x)=\big(\phi(x_1),\dots,\phi(x_n)\big)$（GELU、tanh、sigmoid 都是）：第 $i$ 个输出只依赖第 $i$ 个输入，所以 $J=\diag\big(\phi'(x_1),\dots,\phi'(x_n)\big)$，只有对角线非零。
- **标量函数**（比如损失）：$J=\nabla f^\top$，一行。

::: ex 手算一个雅可比
$\mathbf f(x_1,x_2)=(x_1+x_2,\;x_1x_2,\;x_2^2)$ 是 $\R^2\to\R^3$，所以 $J$ 是 $[3,2]$。逐行求偏导：第一行 $(1,1)$，第二行 $(x_2,x_1)$，第三行 $(0,2x_2)$。在 $\mathbf x=(1,2)$：

$$J_{\mathbf f}(1,2)=\begin{pmatrix}1&1\\2&1\\0&4\end{pmatrix},\qquad \mathbf f(1,2)=(3,2,4).$$

预测：输入挪 $\boldsymbol\Delta=(0.01,0)$，输出约挪 $J\boldsymbol\Delta=(0.01,0.02,0)$。真值 $\mathbf f(1.01,2)-\mathbf f(1,2)=(0.01,0.02,0)$，这里恰好一致，因为 $\mathbf f$ 对 $x_1$ 是线性的；沿 $x_2$ 挪就会看到 $o(\norm{\boldsymbol\Delta})$ 的误差（第三个分量 $x_2^2$ 是弯的）。
:::

## 多元链式法则：雅可比相乘

一元时，链式法则说“变化率相乘”。多元时，把 (M6.3) 的推导原样重做一遍，数换成矩阵：

::: deriv 多元链式法则
$\mathbf f(\mathbf x+\boldsymbol\Delta)=\mathbf u+J_{\mathbf f}\,\boldsymbol\Delta+o(\norm{\boldsymbol\Delta})$ || 内层：记 $\mathbf u=\mathbf f(\mathbf x)$，用 (M6.11)
$\mathbf g(\mathbf u+\boldsymbol\delta)=\mathbf g(\mathbf u)+J_{\mathbf g}\,\boldsymbol\delta+o(\norm{\boldsymbol\delta})$ || 外层 $\mathbf g$ 在 $\mathbf u$ 处的 (M6.11)
$\mathbf g\big(\mathbf f(\mathbf x+\boldsymbol\Delta)\big)=\mathbf g(\mathbf u)+J_{\mathbf g}J_{\mathbf f}\,\boldsymbol\Delta+o(\norm{\boldsymbol\Delta})$ || 把第 1 行的位移 $\boldsymbol\delta=J_{\mathbf f}\boldsymbol\Delta+o(\cdot)$ 代入第 2 行
$J_{\mathbf g\circ\mathbf f}(\mathbf x)=J_{\mathbf g}(\mathbf u)\,J_{\mathbf f}(\mathbf x)$ || 与 (M6.11) 对照：$\boldsymbol\Delta$ 前面的矩阵就是复合函数的雅可比
:::

::: eq 多元链式法则 | (M6.12)
$$\underbrace{J_{\mathbf g\circ\mathbf f}(\mathbf x)}_{[p,n]}=\underbrace{J_{\mathbf g}\big(\mathbf f(\mathbf x)\big)}_{[p,m]}\;\underbrace{J_{\mathbf f}(\mathbf x)}_{[m,n]},\qquad \frac{\partial y_k}{\partial x_j}=\sum_{i=1}^{m}\frac{\partial y_k}{\partial u_i}\,\frac{\partial u_i}{\partial x_j}$$
一元的“变化率相乘”变成“雅可比相乘”；形状按 [M01](m01-vectors-matrices.html#p3) 的规则对上，中间维数 $m$ 被求和吃掉。
:::

右边的逐元素写法值得多看一眼：$x_j$ 影响 $y_k$ 的途径，是经过**每一个**中间量 $u_i$；每条途径贡献“两段变化率之积”，所有途径的贡献加起来。§4 会把它画成计算图上的“路径求和”。

::: ex 用雅可比相乘求梯度
接上例，再套一层标量函数 $y=g(\mathbf u)=u_1u_2+u_3$（$\R^3\to\R$）。在 $\mathbf u=(3,2,4)$：$J_g=\nabla g^\top=(u_2,\,u_1,\,1)=(2,3,1)$，形状 $[1,3]$。

$$J_{g\circ\mathbf f}=(2,3,1)\begin{pmatrix}1&1\\2&1\\0&4\end{pmatrix}=(2+6+0,\;2+3+4)=(8,9).$$

形状 $[1,3]\times[3,2]\to[1,2]$。直接验证：$y=(x_1+x_2)x_1x_2+x_2^2=x_1^2x_2+x_1x_2^2+x_2^2$，$\partial y/\partial x_1=2x_1x_2+x_2^2=8$，$\partial y/\partial x_2=x_1^2+2x_1x_2+2x_2=9$。中心差分核对也是 $(8.000000,\,9.000000)$。
:::

::: ana 类比：兑换链
**映射。** 把导数想成“兑换率”：1 单位 $x$ 换成 $f'$ 单位 $u$，1 单位 $u$ 换成 $g'$ 单位 $y$，于是 1 单位 $x$ 换成 $g'f'$ 单位 $y$。链式法则就是“兑换率沿兑换链相乘”。钱可以同时走几条渠道换过去时（$x_j$ 经过多个 $u_i$ 影响 $y_k$），总所得是各渠道之和，对应 (M6.12) 里的 $\sum_i$。

**收益。** 两条规则的来由都能从兑换里读出来：串联的兑换率相乘，并联的渠道相加。

**失效的地方。** 真实的兑换率是常数，导数却随所在的点而变：同一个 $\sigma$，在 $z=0$ 处“兑换率”是 0.25，在 $z=5$ 处只有 0.0066。多维时“兑换率”是矩阵，而矩阵乘法有顺序：$J_{\mathbf g}J_{\mathbf f}$ 一般不等于 $J_{\mathbf f}J_{\mathbf g}$，后者往往连形状都对不上。
:::

## 一长串雅可比：从哪一头开始乘

一个网络是一条长链：$\mathbf x\to\mathbf h_1\to\mathbf h_2\to\cdots\to\mathbf h_L\to\mathcal L$，最后一步输出一个标量损失。反复用 (M6.12)：

$$(\nabla_{\mathbf x}\mathcal L)^\top=\mathbf g^\top J_LJ_{L-1}\cdots J_1,\qquad \mathbf g=\frac{\partial\mathcal L}{\partial\mathbf h_L},\ \ J_l=\frac{\partial\mathbf h_l}{\partial\mathbf h_{l-1}}\ \ (\mathbf h_0=\mathbf x).$$

矩阵乘法满足结合律（[M01 §6](m01-vectors-matrices.html#p6)），怎么加括号结果都一样，但代价可以天差地别。

- **从右往左**乘，叫**前向模式**（forward mode）：先算 $J_2J_1$，一路搬运一整块矩阵，每一步都是矩阵乘矩阵。它回答的是“输入的每个方向怎样影响后面的一切”；要得到对全部 $n$ 个输入的导数，相当于跑 $n$ 次前向 [@baydin2018]。
- **从左往右**乘，叫**反向模式**（reverse mode）：先算 $\mathbf g^\top J_L$，结果仍是一行；再乘 $J_{L-1}$，还是一行……每一步都是“行向量乘矩阵”，叫**向量–雅可比积**（vector–Jacobian product，VJP）。它回答的是“最后这一个标量，对前面每一个量有多敏感”。

::: eq 向量–雅可比积：反向传播的基本一步 | (M6.13)
$$\bar{\mathbf u}=J_{\mathbf f}(\mathbf u)^\top\,\bar{\mathbf h},\qquad \bar u_j=\sum_i\bar h_i\,\frac{\partial h_i}{\partial u_j},\qquad\text{其中}\ \ \bar{\mathbf h}=\frac{\partial\mathcal L}{\partial\mathbf h},\ \ \bar{\mathbf u}=\frac{\partial\mathcal L}{\partial\mathbf u}$$
$\mathbf h=\mathbf f(\mathbf u)$ 是某一步运算。已知损失对输出的导数 $\bar{\mathbf h}$（与 $\mathbf h$ 同形状），乘上雅可比的转置，得到损失对输入的导数 $\bar{\mathbf u}$（与 $\mathbf u$ 同形状）。来由：把 (M6.12) 的外层取成“从 $\mathbf h$ 到损失”，它的雅可比是一行 $\bar{\mathbf h}^\top$，于是 $\bar{\mathbf u}^\top=\bar{\mathbf h}^\top J_{\mathbf f}$，两边转置即得。
:::

::: fig figures/m06-jacobian-order.svg | 图 M6.4 同一串雅可比，两种乘法顺序 | wide
**看两行里中间结果的形状：反向模式每乘一步仍是一根细条（一行），前向模式一路都是整块方阵。** 设三层、每层宽 768（GPT-2 small 的隐藏维度），$\mathbf g^\top=\partial\mathcal L/\partial\mathbf h_3$ 是 $[1,768]$ 的一行。按 $2mnk$（[M01 §5](m01-vectors-matrices.html#p5)）计数：反向模式三步共 $3.54\times10^6$ 次运算，前向模式 $1.81\times10^9$ 次，相差约 512 倍。数值由脚本计算。
:::

训练恰好是反向模式最占便宜的情形：输出只有一个标量损失，输入（参数）却有上亿个。Baydin 等人的综述给出了一般的账：设求一次函数值要 $\mathrm{ops}(f)$ 次运算，对 $\R^n\to\R^m$ 的函数，前向模式求完整雅可比约需 $n\cdot c\cdot\mathrm{ops}(f)$，反向模式约需 $m\cdot c\cdot\mathrm{ops}(f)$，常数 $c<6$、通常在 2 到 3 之间 [@baydin2018]。损失函数的 $m=1$，所以一次反向就拿到全部梯度。反过来，输入很少、输出很多的时候（比如只想知道一个超参数对上万个输出的影响），前向模式更划算。

**实现上从不把雅可比写出来。** 每种运算只需实现“给我 $\bar{\mathbf h}$，还你 $J^\top\bar{\mathbf h}$”这一个函数。线性层是 $W^\top\bar{\mathbf h}$；逐元素函数的雅可比是对角阵，于是 $J^\top\bar{\mathbf h}=\phi'(\mathbf u)\odot\bar{\mathbf h}$（$\odot$ 是逐元素相乘），只要 $n$ 次乘法，根本不用造一个 $n\times n$ 的矩阵。PyTorch 的教程把自己的自动求导引擎描述为“an engine for computing vector-Jacobian product”[@pytorch-autograd]。

::: ex 逐元素运算的反向：不造对角矩阵
$\mathbf h=\tanh(\mathbf u)$，$\mathbf u=(0.5,\,-1,\,2)$，上游传来 $\bar{\mathbf h}=(0.2,\,-0.5,\,1.0)$。

$\tanh'(u)=1-\tanh^2u$：$\tanh(\mathbf u)=(0.462117,\,-0.761594,\,0.964028)$，所以 $\tanh'(\mathbf u)=(0.786448,\,0.419974,\,0.070651)$。逐元素相乘：$\bar{\mathbf u}=(0.157290,\,-0.209987,\,0.070651)$。

第三个分量的输入 $u=2$ 已经进入 tanh 的饱和区，局部导数只有 0.07，上游的梯度 1.0 过来只剩 0.07。GPT-2 small 的 FFN 中间层有 3072 维，若把对角雅可比真的写成矩阵，有 $3072^2=9{,}437{,}184$ 个元素，其中只有 3072 个不是零。
:::

::: key 本页要点
- $\mathbf f:\R^n\to\R^m$ 的雅可比是 $[m,n]$ 的矩阵，$\mathbf f(\mathbf x+\boldsymbol\Delta)\approx\mathbf f(\mathbf x)+J\boldsymbol\Delta$。线性层 $J=W$，逐元素函数 $J=\diag(\phi')$，标量函数 $J=\nabla f^\top$。
- 多元链式法则：$J_{\mathbf g\circ\mathbf f}=J_{\mathbf g}J_{\mathbf f}$；逐元素看，是对所有中间路径求和。
- 标量损失对上亿参数求导，用反向模式：从损失那头开始，每步做一次向量–雅可比积 $\bar{\mathbf u}=J^\top\bar{\mathbf h}$，代价是前向的常数倍（$c<6$，通常 2–3）。
:::

::: note 这一页用在哪里
[第 1 课 §4](../deepseek/lesson-01-transformer-anatomy.html#p4)：用雅可比与链式法则 $J_{g\circ f}=J_gJ_f$ 解释 softmax 饱和为什么让梯度变小。[第 16 课 §1](../deepseek/lesson-16-mhc.html#p1)：残差流上逐层雅可比的连乘（本课 §7 专门讲它）。
:::

# 计算图与反向传播 | 在一张小图上把前向、反向逐个数字走一遍 | 25

(M6.13) 告诉我们怎样倒着穿过**一步**运算。可真实的计算不是一条单线：同一个权重在每个位置都被用一次，同一个输入可能同时喂给好几个运算。我们需要一套记账方法，保证每条路径的贡献都算到、而且只算一次。这套方法就是在**计算图**上做反向传播。

## 计算图：把计算拆成基本运算

**含义。** 计算图（computational graph）把一次计算画成一张有向无环图：每个节点是一个中间量，每个中间量由它的“父节点”经过一次基本运算得到，而每种基本运算的导数我们都会求（§1 的表）。

**理解。** 以 Baydin 等人综述里的例子 $y=\ln x_1+x_1x_2-\sin x_2$ 为例，在 $(x_1,x_2)=(2,5)$ 处 [@baydin2018]。计算机算它时，其实是按下表一行一行执行的；这张表叫求值轨迹（evaluation trace）。

| 中间量 | 运算 | 在 (2, 5) 的值 |
|---|---|---|
| $v_{-1}$ | $x_1$ | 2 |
| $v_0$ | $x_2$ | 5 |
| $v_1$ | $\ln v_{-1}$ | 0.6931 |
| $v_2$ | $v_{-1}\times v_0$ | 10 |
| $v_3$ | $\sin v_0$ | −0.9589 |
| $v_4$ | $v_1+v_2$ | 10.6931 |
| $v_5$ | $v_4-v_3$ | 11.6521 |

$y=v_5$。注意 $x_1$（即 $v_{-1}$）被 $v_1$、$v_2$ 两个运算用到，$x_2$ 被 $v_2$、$v_3$ 用到：这就是“分叉”。

**数学。** 给每个中间量配一个**伴随**（adjoint）$\bar v_i=\partial y/\partial v_i$：它回答“$v_i$ 变一点，最终输出变多少”。$y$ 只能通过 $v_i$ 的子节点（直接用到 $v_i$ 的那些运算）感受到 $v_i$ 的变化，所以对这些子节点用多元链式法则 (M6.12)：

::: eq 反向传播的递推 | (M6.14)
$$\bar v_i=\sum_{j\in\mathrm{children}(i)}\bar v_j\,\frac{\partial v_j}{\partial v_i},\qquad \bar v_{\mathrm{out}}=\frac{\partial y}{\partial y}=1$$
按与前向**相反**的顺序计算：轮到 $v_i$ 时，它的所有子节点的伴随都已经算好了。
:::

每种基本运算的反向可以背成一张口诀表。写法 “$\bar a\mathrel{+}=\ldots$” 表示“把这一项**累加**到 $\bar a$ 上”，因为 $a$ 可能还被别的运算用到，别处的贡献稍后也要加进来。

| 前向 | 反向 | 口诀 |
|---|---|---|
| $c=a+b$ | $\bar a\mathrel{+}=\bar c$，$\bar b\mathrel{+}=\bar c$ | 加法：原样分发 |
| $c=a-b$ | $\bar a\mathrel{+}=\bar c$，$\bar b\mathrel{+}=-\bar c$ | 减法：分发，减数取负 |
| $c=a\times b$ | $\bar a\mathrel{+}=\bar c\,b$，$\bar b\mathrel{+}=\bar c\,a$ | 乘法：交换相乘 |
| $c=\phi(a)$ | $\bar a\mathrel{+}=\bar c\,\phi'(a)$ | 一元函数：乘局部导数 |
| $a$ 被用了多次 | 各处的贡献相加 | 分叉：求和 |

::: deriv 反向走一趟：从 v̄₅ = 1 出发
$\bar v_5=1$ || 起点：$y=v_5$
$\bar v_4=\bar v_5\cdot1=1$ || $v_5=v_4-v_3$：减法，原样分发……
$\bar v_3=\bar v_5\cdot(-1)=-1$ || ……减数取负
$\bar v_1=\bar v_4=1,\ \ \bar v_2=\bar v_4=1$ || $v_4=v_1+v_2$：加法，原样分发
$\bar v_0\mathrel{+}=\bar v_3\cos v_0=-0.2837$ || $v_3=\sin v_0$：乘局部导数 $\cos5=0.2837$
$\bar v_{-1}\mathrel{+}=\bar v_2\,v_0=5$ || $v_2=v_{-1}v_0$：乘法，交换相乘……
$\bar v_0\mathrel{+}=\bar v_2\,v_{-1}=2$ || ……另一个因子同理；此时 $\bar v_0=-0.2837+2=1.7163$
$\bar v_{-1}\mathrel{+}=\bar v_1/v_{-1}=0.5$ || $v_1=\ln v_{-1}$：局部导数 $1/v_{-1}$；此时 $\bar v_{-1}=5+0.5=5.5$
:::

一趟反向同时得到两个偏导数：$\partial y/\partial x_1=\bar v_{-1}=5.5$，$\partial y/\partial x_2=\bar v_0=1.7163$。解析核对：$\partial y/\partial x_1=1/x_1+x_2=5.5$，$\partial y/\partial x_2=x_1-\cos x_2=2-0.2837=1.7163$。中心差分（$h=10^{-5}$）给出 5.49999999997 与 1.71633781445，与解析值 5.5、1.71633781454 在第十位上吻合。

::: err 在分叉处用“赋值”代替“累加”
手写反向传播最常见的 bug，是把 $\bar a\mathrel{+}=\ldots$ 写成 $\bar a=\ldots$：后写进去的贡献把先写进去的覆盖掉。上面的推导里，$\bar v_{-1}$ 先收到乘法那条路的 5，再收到 $\ln$ 那条路的 0.5；如果用赋值，最后留下的是 0.5 而不是 5.5。$\bar v_0$ 同理会得到 2 而不是 1.7163。两个偏导数全错，而且程序不会报任何错，只有梯度检查（§1 的中心差分）能抓住它。
:::

::: ana 类比：反向传播 = 倒查每个人的“影响力”
**映射。** 把最终输出 $y$ 想成一份成绩单。伴随 $\bar v_i$ 是中间人 $v_i$ 的“影响力”：他的工作多做一个单位，最终成绩变多少。每个节点只向直接用了自己产出的下家打听：“你的影响力是多少？我的产出在你那儿被放大几倍（局部导数）？”两者相乘，就是经由这位下家的影响力；有几个下家，就把几份加起来。

**收益。** 它把两条规则讲清楚了：为什么必须**倒着**算（要等所有下家都报上自己的影响力，才能算自己的）；为什么分叉处要**相加**（一个人同时给几个下家干活，影响力是几条渠道之和）。

**失效的地方。** 这里的“影响力”只是在当前这一点的线性化：参数一变，局部导数跟着变，整张表要重算。它也不像分钱那样守恒：$\bar v_{-1}=5.5$ 比输出本身的伴随 $\bar v_5=1$ 还大，伴随还可以是负的（$\bar v_3=-1$）。
:::

::: fig figures/m06-backprop-graph.svg | 图 M6.5 计算图上的一次前向与一次反向 | wide
**看左边两个输入框下面的红字：$x_1$、$x_2$ 各被用了两次，它们的梯度是两条路径的贡献之和。** 蓝色是前向值（从左往右算），红色是伴随 $\bar v=\partial y/\partial v$（从 $\bar v_5=1$ 出发从右往左算），箭头上是局部导数 $\partial(\text{终点})/\partial(\text{起点})$。任何一个伴随都等于“所有出边上的（子节点伴随 × 局部导数）之和”，这就是 (M6.14)。据《Automatic Differentiation in Machine Learning: a Survey》图 4 与表 3 重绘 [@baydin2018]，数值由脚本计算并与解析梯度核对。
:::

## 为什么要先前向，并把中间值存起来

看上面反向的每一行：乘法的反向要用到另一个因子的**前向值**（$v_0=5$、$v_{-1}=2$），$\ln$ 的反向要用 $v_{-1}$，$\sin$ 的反向要用 $\cos v_0$。所以反向传播之前必须先做一次前向，并把反向要用的中间值留着。在大模型里，这些中间值就是每一层的**激活**（activation），它们占用的显存随层数、序列长度、batch 大小一起增长。Baydin 等人指出，反向模式的代价正在存储：最坏情况下与运算次数成正比，可以用 checkpointing 之类的办法缓解 [@baydin2018]。“重算”（recomputation）就是其中一种：前向时丢掉一部分便宜的中间值，反向时再算一遍。DeepSeek-V3 就在反向时重算 RMSNorm 和 MLA 的上投影，用少量额外计算换显存（[第 12 课 §7](../deepseek/lesson-12-v3-infrastructure.html#p7)）。

**一次反向花多少？** 反向时每条边只经过一次，每条边上做的是常数量的工作（一次乘法、一次累加），所以反向的代价是前向的常数倍。上一页 Baydin 等人给的 $c$ 算的是“前向加反向”一共相当于几次前向。对以矩阵乘法为主的网络，单是反向就约是前向的 2 倍，前向加反向约 3 倍，§6 会把它数出来。

::: note 一点历史
反向模式与反向传播的历史相互交织。Baydin 等人的梳理是：反向模式最早的公开描述通常引 Linnainmaa 1970 年的硕士论文，Werbos 1974 年把它写成了按依赖关系排序的离散形式；机器学习界又独立地重新发现过几次，最终因 Rumelhart、Hinton 与 Williams 1986 年发表在 Nature 上的论文而广为人知 [@baydin2018; @rumelhart1986]。今天的框架把这套记账自动化了：PyTorch 在前向时把每一步运算记进一张有向无环图，反向时按链式法则把梯度逐层传回，并**累加**到各参数的 `.grad` 里 [@pytorch-autograd]。“累加”这一点有两个直接后果：每一步训练开始前要把旧梯度清零；想用小显存模拟大 batch 时，可以连续做几次反向、让梯度自己加起来再更新一次。
:::

::: err 把反向传播当成“训练算法”
反向传播只负责**算梯度**：给定当前参数和一批数据，精确地（在浮点精度内）算出 $\nabla\mathcal L$。拿到梯度之后怎样更新参数，是优化器的事（SGD、AdamW、Muon，见 [M07](m07-optimization.html)）。它也不是 §1 的数值差分（没有步长 $h$），不是符号求导（不生成导数公式），而是反向模式的自动微分 [@baydin2018]。
:::

::: key 本页要点
- 计算图把一次计算拆成基本运算。伴随 $\bar v=\partial y/\partial v$ 按前向的逆序计算：$\bar v_i=\sum_{j}\bar v_j\,\partial v_j/\partial v_i$，求和跑遍 $v_i$ 的所有子节点。
- 口诀：加法原样分发，乘法交换相乘，一元函数乘局部导数，一个量被用了几次就把几处的贡献加起来。
- 反向要用前向的中间值，所以先前向、存激活；存不下就重算。一次反向的代价是前向的常数倍。
:::

::: note 这一页用在哪里
[第 2 课 §3](../deepseek/lesson-02-gpt-training.html#p3)：teacher forcing 一次前向算出 $T$ 个位置的损失；每个权重在 $T$ 个位置都被用了一次，按本页“分叉处求和”，一次反向就把 $T$ 份贡献加在一起。[第 9 课 §6](../deepseek/lesson-09-v2-mla.html#p6)：压缩查询为什么能省训练时的激活显存。[第 12 课 §7](../deepseek/lesson-12-v3-infrastructure.html#p7)：DeepSeek-V3 在反向时重算 RMSNorm 与 MLA 的上投影。
:::

# softmax 与交叉熵的梯度 | softmax 的雅可比、p − y，以及两者为什么要一起算 | 25

语言模型的最顶端都是同一个结构：最后一层输出 $V$ 个 logits $\mathbf z\in\R^V$，softmax 把它变成下一个 token 的概率分布 $\mathbf p$，损失是真实 token $y$ 的负对数概率 $\ell=-\log p_y$（[M05](m05-information-theory.html) 的交叉熵）。这里不加粗的 $y$ 是真实 token 在词表里的编号；加粗的 $\mathbf y\in\R^V$ 是它的独热向量，第 $y$ 格为 1、其余为 0。反向传播就从这里出发，所以这是被计算次数最多的一个导数。难点在于 softmax 的每个输出都依赖**全部** logits（分母是它们的和），它的雅可比是一个满满的 $V\times V$ 矩阵：GPT-2 的 $V=50257$ [@gpt2-config]，这个矩阵每个位置就有 25 亿个元素。可是第 2 课告诉我们，交叉熵对 logits 的梯度只是 $\mathbf p-\mathbf y$，一个长度为 $V$ 的向量。这一页把中间的来龙去脉补全。

## softmax 的雅可比

::: pre softmax 与克罗内克 δ | 会的可以跳过；softmax 作为分布的性质见 [M04](m04-probability.html)
$\softmax$ 把任意实数向量变成概率分布：$p_i=e^{z_i}/\sum_ke^{z_k}$，每个 $p_i>0$，$\sum_ip_i=1$。给所有 logits 同加一个常数 $c$，分子分母同乘 $e^c$，结果不变。**克罗内克 δ**（Kronecker delta）$\delta_{ij}$ 在 $i=j$ 时为 1、否则为 0；单位矩阵 $I$ 的第 $(i,j)$ 个元素就是 $\delta_{ij}$。
:::

先对 $\log p_i$ 求导（对数把商拆成差，求导更省事），再换回 $p_i$：

::: deriv softmax 的偏导数
$\log p_i=z_i-\log\sum_ke^{z_k}$ || 取对数：商变成差
$\dfrac{\partial\log p_i}{\partial z_j}=\delta_{ij}-\dfrac{e^{z_j}}{\sum_ke^{z_k}}=\delta_{ij}-p_j$ || 第一项只在 $j=i$ 时有导数 1；第二项用链式法则：$\log$ 的导数 $1/(\cdot)$ 乘里面对 $z_j$ 的导数 $e^{z_j}$
$\dfrac{\partial p_i}{\partial z_j}=p_i\,\dfrac{\partial\log p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)$ || 链式法则：$\frac{\partial\log p_i}{\partial z_j}=\frac1{p_i}\frac{\partial p_i}{\partial z_j}$，两边乘 $p_i$
:::

::: eq softmax 的雅可比 | (M6.15)
$$\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j),\qquad J_{\softmax}(\mathbf z)=\diag(\mathbf p)-\mathbf p\mathbf p^\top\in\R^{V\times V}$$
对角元 $p_i(1-p_i)>0$：自己的 logit 变大，自己的概率变大。非对角元 $-p_ip_j<0$：别人的 logit 变大，自己的概率变小。
:::

这个矩阵有四条性质，每条都有一句话的理由。**对称**：$J_{ij}=J_{ji}=-p_ip_j$。**每行之和为 0**：$\sum_jp_i(\delta_{ij}-p_j)=p_i-p_i\sum_jp_j=0$，即 $J\mathbf 1=\mathbf 0$。这正是“所有 logits 同加一个常数，softmax 不变”的导数版本：沿 $\mathbf 1$ 方向挪 logits，概率一动不动。所以 $J$ 有一个特征值是 0，不可逆。**每列之和也为 0**（由对称性）：概率之和永远是 1，各概率的变化必须互相抵消。**半正定**：对任意 $\mathbf v$，$\mathbf v^\top J\mathbf v=\sum_ip_iv_i^2-\big(\sum_ip_iv_i\big)^2$，这正是“按分布 $\mathbf p$ 随机取一个 $v_i$”的方差（[M04](m04-probability.html)），不会是负数。

::: ex z = (2, 1, 0) 的雅可比
$e^2=7.389$，$e^1=2.718$，$e^0=1$，和为 $11.107$，所以 $\mathbf p=(0.665241,\,0.244728,\,0.090031)$。对角元 $p_i(1-p_i)=(0.222695,\,0.184836,\,0.081925)$；非对角元 $-p_0p_1=-0.162803$，$-p_0p_2=-0.059892$，$-p_1p_2=-0.022033$。第一行之和 $0.222695-0.162803-0.059892=0$，另两行同理。三个特征值是 $0$、$0.118553$、$0.370904$：一个 0（对应方向 $\mathbf 1$），其余为正，与半正定、不可逆两条性质一致。
:::

::: fig figures/m06-softmax-jacobian.svg | 图 M6.6 softmax 的雅可比与交叉熵的梯度（z = (2, 1, 0)，真类别为 0） | wide
**看中间的热力图：只有对角线是蓝色（正），其余都是红色（负），每一行加起来恰好是 0。** 左栏是 logits 与 softmax 概率；中栏是雅可比 $\partial p_i/\partial z_j$，绝对值越大颜色越深，对角元加粗；右栏是交叉熵对 logits 的梯度 $\mathbf p-\mathbf y$：真类别那一格是 $p_0-1=-0.335$，其余两格就等于各自的概率，三格之和为 0。所有数值由脚本计算。
:::

## 交叉熵对 logits 的梯度：p − y

现在把 softmax 和它上面的损失 $\ell=-\log p_y$ 接起来。可以有两条路，结论相同。第一条路直接展开 $\ell$，第 2 课用的就是它：$\ell=-z_y+\log\sum_ke^{z_k}$，对 $z_j$ 求导得 $-\delta_{jy}+p_j$（[第 2 课 §3](../deepseek/lesson-02-gpt-training.html#p3)）。第二条路严格按反向传播的步骤走，先对 $\mathbf p$ 求导，再乘 softmax 雅可比的转置 (M6.13)，可以看清那个 $V\times V$ 矩阵是怎样“塌缩”的：

::: deriv 经过雅可比的反向传播
$\bar{\mathbf p}=\dfrac{\partial\ell}{\partial\mathbf p}=-\dfrac{\mathbf y}{\mathbf p}$ || $\ell=-\log p_y=-\sum_ky_k\log p_k$，逐元素求导；只有第 $y$ 格非零，等于 $-1/p_y$
$\bar{\mathbf z}=J^\top\bar{\mathbf p}=\big(\diag(\mathbf p)-\mathbf p\mathbf p^\top\big)\bar{\mathbf p}$ || 向量–雅可比积 (M6.13)；$J$ 对称，$J^\top=J$
$=\mathbf p\odot\bar{\mathbf p}-\mathbf p\,(\mathbf p^\top\bar{\mathbf p})$ || 分配律：$\diag(\mathbf p)\bar{\mathbf p}$ 就是逐元素相乘
$=-\mathbf y-\mathbf p\cdot(-1)$ || $\mathbf p\odot(-\mathbf y/\mathbf p)=-\mathbf y$；$\mathbf p^\top\bar{\mathbf p}=-p_y\cdot\frac1{p_y}=-1$
$=\mathbf p-\mathbf y$ || 整理
:::

::: eq 交叉熵对 logits 的梯度 | (M6.16)
$$\frac{\partial\ell}{\partial\mathbf z}=\mathbf p-\mathbf y,\qquad \ell=-\log p_y,\ \ \mathbf p=\softmax(\mathbf z),\ \ \mathbf y=\text{独热标签}$$
“预测减目标”。它的各分量之和为 0；各分量的绝对值之和（$L_1$ 范数）是 $2(1-p_y)$，因为真类别那一格贡献 $1-p_y$，其余各格的 $p_k$ 加起来也是 $1-p_y$：猜得越准，梯度越小。
:::

::: ex p − y 与沿负梯度走一步
接上例，真类别是 0：$\ell=-\ln0.665241=0.407606$ nats。梯度 $\mathbf p-\mathbf y=(-0.334759,\,0.244728,\,0.090031)$，和为 0，绝对值之和 $0.669518=2(1-0.665241)$。

沿负梯度把 logits 改一步（步长 1，只为演示）：$\mathbf z'=\mathbf z-(\mathbf p-\mathbf y)=(2.334759,\,0.755272,\,-0.090031)$，新概率 $\mathbf p'=(0.772453,\,0.159188,\,0.068360)$，损失降到 $0.258184$。真类别的 logit 被推高，另外两个被压低，而且原来概率越大的错误答案被压得越多（0.245 对 0.090）。
:::

## 为什么要把 softmax 和交叉熵合在一起算

**省内存、省计算。** 如果先把雅可比造出来再乘，GPT-2 的每个位置要存 $50257^2=2{,}525{,}766{,}049$ 个数，按 4 字节算约 10.1 GB；一条 1024 个 token 的序列就是约 10.3 TB，而 1024 个 $\mathbf p-\mathbf y$ 一共只占约 206 MB。实际上只需要 (M6.16) 的 $O(V)$ 个减法。即使 softmax 夹在网络中间（注意力权重、MoE 的门控），它的反向也只需要 $O(V)$：把上面推导的第 3 行一般化，就是

::: eq softmax 的向量–雅可比积 | (M6.17)
$$J_{\softmax}^\top\,\bar{\mathbf p}=\mathbf p\odot\big(\bar{\mathbf p}-(\mathbf p^\top\bar{\mathbf p})\,\mathbf 1\big)$$
先算一个标量 $\mathbf p^\top\bar{\mathbf p}$，再逐元素运算。代价 $O(V)$，从不造出 $V\times V$ 的矩阵。
:::

::: ex 用 (M6.17) 代替矩阵乘法
$\mathbf p=(0.665241,0.244728,0.090031)$，上游梯度 $\bar{\mathbf p}=(0.3,-1.2,0.5)$。$\mathbf p^\top\bar{\mathbf p}=0.199572-0.293674+0.045015=-0.049087$。于是 $\bar{\mathbf z}=\mathbf p\odot(\bar{\mathbf p}+0.049087)=(0.232227,\,-0.281661,\,0.049435)$，与直接用例题里的 $3\times3$ 矩阵相乘的结果一致。
:::

**数值稳定。** $e^{z}$ 很容易溢出：float64 在 $z>709.78$、FP32 与 BF16 在 $z>88.72$、FP16 在 $z>11.09$ 时，$e^z$ 就超过了格式能表示的最大数（用各格式的最大值取对数算出，格式参数见 [M08](m08-floating-point.html)）。反过来，$p_y$ 太小会被舍入成 0，$\log0=-\infty$。解决办法是**不单独算出 $\mathbf p$ 再取对数**，而是直接用 $\ell=-z_y+\log\sum_ke^{z_k}$，并对后一项用 **log-sum-exp 技巧**：

$$\log\sum_ke^{z_k}=m+\log\sum_ke^{z_k-m},\qquad m=\max_kz_k.$$

减去最大值后，最大的那一项是 $e^0=1$，不会溢出；和至少是 1，取对数也不会出问题。

::: ex 同一个分布，差 998 的 logits
$\mathbf z=(1000,999,998)$。直接算 $e^{1000}$ 在 float64 里溢出成无穷，$\infty/\infty$ 得到 NaN。减去最大值 1000 之后是 $(0,-1,-2)$，与 $(2,1,0)$ 只差一个常数，所以 $\mathbf p$ 完全相同：$(0.665241,0.244728,0.090031)$；$\log\sum_ke^{z_k}=1000.407606$，真类别为 0 时 $\ell=-1000+1000.407606=0.407606$，与上例一致。梯度照样是 $\mathbf p-\mathbf y$，不受平移影响（行和为 0 的另一种说法）。
:::

所以在实践中，交叉熵函数接收的是 **logits** 而不是概率：softmax、取对数、求导三步一起完成，既不造雅可比，也不让中间量溢出。

## 为什么分类用交叉熵，而不用平方误差

(M6.16) 还有一个不那么显眼的好处：**不会因为 softmax 饱和而失去梯度**。设模型很自信地猜错了：$\mathbf z=(10,0,0)$，真类别却是 1。$\mathbf p=(0.999909,\,0.000045,\,0.000045)$。

- 交叉熵：损失 $10.0001$，梯度 $\mathbf p-\mathbf y=(0.999909,\,-0.999955,\,0.000045)$，长度 $1.414$，几乎贴着上限 $\sqrt2$（$\norm{\mathbf p-\mathbf y}^2=(1-p_y)^2+\sum_{k\ne y}p_k^2\le2(1-p_y)^2<2$，因为 $\sum_{k\ne y}p_k^2\le\big(\sum_{k\ne y}p_k\big)^2=(1-p_y)^2$；各 $p_k>0$，所以这个上限只能无限逼近，取不到）。错得越离谱，推得越用力。
- 平方误差 $\tfrac12\norm{\mathbf p-\mathbf y}^2$：损失 $0.9999$，但它对 $\mathbf z$ 的梯度要乘 softmax 的雅可比，$J^\top(\mathbf p-\mathbf y)$，而饱和时 $J$ 的所有元素都接近 0（$p_i(1-p_i)\approx0$），结果梯度长度只有 $1.7\times10^{-4}$，小了约 8300 倍。模型错得很离谱，却几乎学不动。

交叉熵里的 $\log$ 恰好抵消了 softmax 的指数：$\bar{\mathbf p}=-\mathbf y/\mathbf p$ 在 $p_y$ 很小时很大，正好补上雅可比的“小”，这就是上面推导第 4 行发生的事。图 M6.7 把领先幅度从 0 连续调到 12，看两种损失的梯度怎样变化。

::: fig figures/m06-ce-vs-mse.svg | 图 M6.7 softmax 饱和时，两种损失对 logits 的梯度 | wide
**看两条实线往右走的方向：猜错时，交叉熵（蓝）的梯度升到紧贴上限 $\sqrt2$ 后一直保持，平方误差（红）的梯度却一路掉下去。** 横轴是 logits $\mathbf z=(m,0,0)$ 里的 $m$，$m$ 越大，模型越确信答案是类别 0。实线的真类别是 1（越来越自信地猜错），虚线的真类别是 0（越来越自信地猜对）；猜对时两种损失的梯度都应该变小，也确实都变小了。竖虚线 $m=10$ 就是正文的例子：交叉熵 1.414，平方误差 $1.7\times10^{-4}$。平方误差在“很有把握地猜错”时的梯度（$1.7\times10^{-4}$），与交叉熵在“很有把握地猜对”时的梯度（$1.1\times10^{-4}$）差不多大：一个离谱的错误，平方误差只给它交叉熵留给“已经答对、稍作微调”的那点推力。纵轴为对数刻度，全部数值由脚本计算。
:::

## 两个变体：温度、sigmoid 与一般的目标分布

**温度。** 若 $\mathbf p=\softmax(\mathbf s/\tau)$，$\tau>0$ 是温度，则 $\mathbf z=\mathbf s/\tau$，$\partial z_j/\partial s_j=1/\tau$，由链式法则 $\partial\ell/\partial\mathbf s=(\mathbf p-\mathbf y)/\tau$。例：$\mathbf s=(2,1,0)$，真类别仍为 0，$\tau=0.5$ 时 $\mathbf p=(0.866813,0.117310,0.015876)$，梯度 $(-0.266373,\,0.234621,\,0.031752)$；$\tau=2$ 时 $\mathbf p=(0.506480,0.307196,0.186324)$，梯度 $(-0.246760,\,0.153598,\,0.093162)$。

**一般的目标分布。** 目标不是独热向量、而是任意分布 $\mathbf q$（$\sum_kq_k=1$）时，重复第一条路的推导：$-\sum_kq_k\log p_k=-\mathbf q^\top\mathbf z+\log\sum_ke^{z_k}$（用了 $\sum_kq_k=1$），对 $\mathbf z$ 求导就是 $\mathbf p-\mathbf q$。KL 散度 $\KL(\mathbf q\|\mathbf p)$ 与交叉熵只差 $\mathbf q$ 的熵，一个与 $\mathbf z$ 无关的常数，所以梯度也是 $\mathbf p-\mathbf q$。第 14 课 DSA 的 indexer 用 KL 去对齐主注意力的分布，梯度就是这个“预测减目标”（[第 14 课 §6](../deepseek/lesson-14-sparse-attention.html#p6)）。

**sigmoid 与二分类。** 只有两类时，把两个 logit 记作 $(z,0)$，$\softmax$ 的第一个分量是 $\frac{e^z}{e^z+1}=\sigma(z)$，所以 sigmoid 就是两类的 softmax。标签 $y\in\{0,1\}$ 时，**二元交叉熵**（binary cross-entropy，BCE）$\ell=-[y\log\sigma(z)+(1-y)\log(1-\sigma(z))]$ 用 (M6.4) 求导：$-y(1-\sigma)+(1-y)\sigma=\sigma(z)-y$，还是“预测减目标”。

::: eq “预测减目标”的三种形式 | (M6.18)
$$\begin{aligned}\frac{\partial}{\partial\mathbf z}\Big(-\sum_kq_k\log\softmax(\mathbf z)_k\Big)&=\mathbf p-\mathbf q\\[6pt] \frac{\partial}{\partial z}\,\mathrm{BCE}\big(\sigma(z),y\big)&=\sigma(z)-y\\[6pt] \frac{\partial}{\partial\Delta}\big(-\log\sigma(\Delta)\big)&=\sigma(\Delta)-1\end{aligned}$$
第三式是第二式取 $y=1$：第 3 课 Bradley–Terry 损失 $-\log\sigma(r_w-r_l)$ 对差值 $\Delta=r_w-r_l$ 的导数。
:::

::: ex 二分类的两种标签
$z=2$，$\sigma(z)=0.880797$。标签 $y=1$：梯度 $\sigma-1=-0.119203$，沿负梯度把 $z$ 往上推，但推得不多，因为已经猜对了八成八。标签 $y=0$：梯度 $0.880797$，推得很用力。与 (M6.16) 的多分类情形完全对应。
:::

::: note 这一页用在哪里
[第 2 课 §3](../deepseek/lesson-02-gpt-training.html#p3)：梯度的起点 (2.7)。[第 1 课 §7](../deepseek/lesson-01-transformer-anatomy.html#p7)：输出层的 softmax。[第 3 课 §2](../deepseek/lesson-03-alignment-rlhf.html#p2)：Bradley–Terry 奖励模型的 sigmoid 损失。[第 7 课 §3](../deepseek/lesson-07-deepseekmoe.html#p3)、[第 10 课 §3](../deepseek/lesson-10-v2-moe-training.html#p3)：均衡损失的梯度要穿过门控 softmax 的雅可比 (M6.15)。[第 11 课 §2](../deepseek/lesson-11-v3-architecture.html#p2)：V3 把门控换成 sigmoid，比较的正是 (M6.15) 这个稠密雅可比与 sigmoid 的对角雅可比。[第 14 课 §6](../deepseek/lesson-14-sparse-attention.html#p6)：KL 对齐的梯度 $\mathbf p-\mathbf q$。
:::

::: key 本页要点
- softmax 的雅可比 $\diag(\mathbf p)-\mathbf p\mathbf p^\top$：对称、行和为 0（平移不变）、半正定、不可逆。
- 交叉熵对 logits 的梯度 $\mathbf p-\mathbf y$；一般目标分布是 $\mathbf p-\mathbf q$，sigmoid 二分类是 $\sigma-y$。
- 实现时 softmax 与交叉熵合起来算：用 log-sum-exp 防溢出，用 $O(V)$ 的公式代替 $V\times V$ 的雅可比。
- $\log$ 抵消了 softmax 的饱和：自信地猜错时，交叉熵的梯度接近最大，平方误差的梯度却几乎为 0。
:::

# 线性层的反向 | dX = dY·Wᵀ，dW = Xᵀ·dY：形状、对 batch 求和、反向是前向的两倍 | 25

大模型里绝大部分运算是线性层的矩阵乘法：注意力的四个投影、FFN 的两层、输出层（[第 1 课 §8](../deepseek/lesson-01-transformer-anatomy.html#p8)）。每个线性层在反向时要交出两样东西：对**输入**的梯度（继续往前一层传），和对**权重**的梯度（交给优化器更新参数）。这一页推出这两个公式，用形状核对它们，再数出“训练一个 token 约 $6N$ 次运算”里的 6 从哪里来。

## 一个 token：列向量写法

先看论文里常用的写法（DeepSeek 的论文也这样写）：一个 token 的输入 $\mathbf x\in\R^n$，权重 $W\in\R^{m\times n}$，偏置 $\mathbf b\in\R^m$，输出 $\mathbf y=W\mathbf x+\mathbf b\in\R^m$（这一页的 $\mathbf y$、$Y$ 都是线性层的输出，不是 §5 的独热标签）。上游传来 $\bar{\mathbf y}=\partial\mathcal L/\partial\mathbf y\in\R^m$。

- **对输入**：雅可比是 $W$，由 (M6.13)，$\bar{\mathbf x}=W^\top\bar{\mathbf y}$，形状 $[n,m]\times[m]\to[n]$。
- **对权重**：$y_i=\sum_jW_{ij}x_j+b_i$，$W_{ij}$ 只出现在 $y_i$ 里，局部导数是 $x_j$，所以 $\partial\mathcal L/\partial W_{ij}=\bar y_ix_j$。排回 $[m,n]$ 的形状，就是外积 $\bar W=\bar{\mathbf y}\,\mathbf x^\top$，正是 (M6.10) 最后一式的样子。
- **对偏置**：$\bar{\mathbf b}=\bar{\mathbf y}$。

一个 token 的 $\bar W=\bar{\mathbf y}\mathbf x^\top$ 是一个外积，秩至多为 1（两个向量都非零时恰为 1；[M01 §2](m01-vectors-matrices.html#p2) 的“外积之和”读法）。

## 一个 batch：行向量写法（代码里的样子）

代码里一次处理一批 token（设共 $B$ 个），并把它们按**行**堆起来：$X\in\R^{B\times n}$ 的第 $t$ 行是第 $t$ 个 token 的输入。很多实现把权重存成 [输入维, 输出维]，计算 $Y=XW+\mathbf 1\mathbf b^\top$，$W\in\R^{n\times m}$，$Y\in\R^{B\times m}$（GPT-2 的 `Conv1D` 就是这样存的，见 [M01 §3](m01-vectors-matrices.html#p3)）。上游传来 $dY=\partial\mathcal L/\partial Y$，与 $Y$ 同形。

::: deriv 线性层的三个梯度（行写法）
$Y_{tj}=\sum_iX_{ti}W_{ij}+b_j$ || 逐元素写出前向；$t$ 是 batch 里 token 的编号
$\dfrac{\partial\mathcal L}{\partial W_{ij}}=\sum_t\dfrac{\partial\mathcal L}{\partial Y_{tj}}\,\dfrac{\partial Y_{tj}}{\partial W_{ij}}=\sum_tdY_{tj}\,X_{ti}$ || 同一个 $W_{ij}$ 被**每个** token 用了一次（分叉），对 $t$ 求和；局部导数是 $X_{ti}$
$=(X^\top dY)_{ij}$ || 对 $t$ 求和正是矩阵乘法 $X^\top dY$ 的第 $(i,j)$ 格
$\dfrac{\partial\mathcal L}{\partial X_{ti}}=\sum_jdY_{tj}\,W_{ij}=(dY\,W^\top)_{ti}$ || $X_{ti}$ 只影响同一个 token 的各个输出 $Y_{tj}$，对 $j$ 求和
$\dfrac{\partial\mathcal L}{\partial b_j}=\sum_tdY_{tj}$ || 偏置被加到每个 token 上：对 batch 求和
:::

::: eq 线性层的反向（行写法 Y = XW + b） | (M6.19)
$$\begin{aligned}dX&=dY\,W^\top &&[B,m]\times[m,n]\to[B,n]\\ dW&=X^\top dY &&[n,B]\times[B,m]\to[n,m]\\ d\mathbf b&=\textstyle\sum_{t=1}^{B}dY_{t,:} &&[m]\end{aligned}$$
$dW$ 是 $B$ 个秩 1 外积之和：每个 token 贡献一个 $\mathbf x_t\,(d\mathbf y_t)^\top$。
:::

**形状自检**（不是推导，但能抓住大部分笔误）：$dW$ 必须与 $W$ 同形 $[n,m]$，手里只有 $X$（$[B,n]$）和 $dY$（$[B,m]$），唯一能乘出 $[n,m]$ 的组合是 $X^\top dY$。$dX$ 必须是 $[B,n]$，只能由 $dY$（$[B,m]$）右乘一个 $[m,n]$ 的矩阵得到，那就是 $W^\top$。

**两种存法的翻译。** PyTorch 的 `nn.Linear` 把权重存成 [输出维, 输入维]，计算 $Y=XA^\top+\mathbf b$ [@torch-linear]。令 $W=A^\top$ 代入 (M6.19)：$dX=dY\,A$，$dA=(X^\top dY)^\top=dY^\top X$。第 12 课讲 Zero Bubble 时写的 $\mathrm dX=\mathrm dY\,W$、$\mathrm dW=\mathrm dY^\top X$ 用的就是这种存法（[第 12 课 §5](../deepseek/lesson-12-v3-infrastructure.html#p5)）。

::: ex 一个 2×3 的线性层，逐个数字
$B=2$ 个 token，$n=3$，$m=2$：

$$X=\begin{pmatrix}1&0&2\\-1&1&0\end{pmatrix},\quad W=\begin{pmatrix}1&-1\\0&2\\1&1\end{pmatrix},\quad \mathbf b=(0,1),\quad Y^\star=\begin{pmatrix}2&2\\0&3\end{pmatrix}.$$

$Y^\star$ 是目标矩阵（这里不用字母 $T$，它在本课程里留给序列长度）。前向：$Y=XW+\mathbf b=\begin{pmatrix}3&2\\-1&4\end{pmatrix}$。取损失 $\mathcal L=\tfrac12\norm{Y-Y^\star}^2$（各元素平方和的一半），$\mathcal L=\tfrac12(1+0+1+1)=1.5$，于是 $dY=Y-Y^\star=\begin{pmatrix}1&0\\-1&1\end{pmatrix}$。

反向：

$$dW=X^\top dY=\begin{pmatrix}1&-1\\0&1\\2&0\end{pmatrix}\begin{pmatrix}1&0\\-1&1\end{pmatrix}=\begin{pmatrix}2&-1\\-1&1\\2&0\end{pmatrix},$$

$$dX=dY\,W^\top=\begin{pmatrix}1&0\\-1&1\end{pmatrix}\begin{pmatrix}1&0&1\\-1&2&1\end{pmatrix}=\begin{pmatrix}1&0&1\\-2&2&0\end{pmatrix}.$$

$d\mathbf b$ 是 $dY$ 按列求和：$(1-1,\;0+1)=(0,1)$。

核对 $dW$ 左上角的元素：$W_{11}$ 被两个 token 用到，贡献 $dY_{11}X_{11}+dY_{21}X_{21}=1\cdot1+(-1)(-1)=2$。对 $W$ 的六个元素逐个做中心差分，得到的矩阵与 $dW$ 一致到 $10^{-9}$。
:::

::: fig figures/m06-linear-backward.svg | 图 M6.8 线性层的三次矩阵乘法，按真实形状等比例画 | wide
**看三行的运算量：前向一次乘法，反向两次，三次一样大。** 形状取 GPT-2 small 的 FFN 上投影 $[768,3072]$、一批 1024 个 token。前向 $Y=XW$；反向 ① $dX=dY\,W^\top$ 交给前一层，反向 ② $dW=X^\top dY$ 交给优化器。每次乘法 $2\times1024\times768\times3072=4.83\times10^9$ 次运算，合计 $1.45\times10^{10}$，正好是 $6\times$ 参数量 $\times$ token 数。几何由脚本按维数计算。
:::

## 反向是前向的两倍：6N 的由来

按 $2mnk$ 数（[M01 §5](m01-vectors-matrices.html#p5)）：前向 $Y=XW$ 是 $2Bnm$；反向的 $dX$ 与 $dW$ 各是一次同样大小的矩阵乘法，各 $2Bnm$，合计 $4Bnm$。所以一个线性层的反向恰好是前向的两倍，一步训练是前向的三倍。这个层有 $nm$ 个参数，每个 token 就是“每个参数 2 次（前向）+ 4 次（反向）= 6 次”运算。把所有线性层加起来，训练一个 token 约 $6N$ 次运算、推理约 $2N$ 次，$N$ 是这些线性层的参数量（Kaplan 等人的 $N$ 不计词表嵌入与位置嵌入，与嵌入共享的反嵌入也就不在其中，见下例）。Kaplan 等人正是按“反向约为前向的两倍”定义了 $C\approx6N$ 每 token [@kaplan2020]，第 2 课的 $C\approx6ND$ 由此而来（[第 2 课 §6](../deepseek/lesson-02-gpt-training.html#p6)）。

::: ex 数一数 GPT-2 small
FFN 上投影 $[768,3072]$、一批 1024 个 token：前向 $4.83\times10^9$，反向 $9.66\times10^9$。整个模型的非嵌入参数 85,056,000 个（第 1 课），每个 token 训练约 $6\times85{,}056{,}000=5.10\times10^8$ 次运算。这个估算忽略了注意力打分等与序列长度有关的项，[第 2 课 §6](../deepseek/lesson-02-gpt-training.html#p6) 讨论了它们什么时候不能忽略。它也没算输出层：反嵌入矩阵与嵌入共享（[第 1 课 §7](../deepseek/lesson-01-transformer-anatomy.html#p7)），不在 85,056,000 里，但每个 token 照样要做一次 $[768]\times[768,50257]$ 的乘法，训练时是 $6\times768\times50257=2.32\times10^8$ 次。对 GPT-2 small 这种词表大、模型小的情形，这一项接近上面主项的一半，不能当成零头。
:::

这个“两半”的结构在系统层面也被利用：$dX$ 马上要交给前一层（前一层的反向在等它），$dW$ 却可以晚一点算。Zero Bubble 流水线调度就把反向拆成这两半，推迟 $dW$ 去填流水线的空档（[第 12 课 §5](../deepseek/lesson-12-v3-infrastructure.html#p5)）。

## 其他常见运算的反向：速查表

有了“逐元素乘局部导数”“矩阵乘法两条公式”“分叉求和”这三条，Transformer 里常见运算的反向都能写出来。

| 前向 | 反向 | 说明 |
|---|---|---|
| $Y=XW$ | $dX=dY\,W^\top$，$dW=X^\top dY$ | (M6.19) |
| $\mathbf h=\phi(\mathbf u)$（GELU、tanh、ReLU） | $d\mathbf u=d\mathbf h\odot\phi'(\mathbf u)$ | 对角雅可比；ReLU 的 $\phi'$ 是 0/1 掩码 |
| $\mathbf p=\softmax(\mathbf z)$ | $d\mathbf z=\mathbf p\odot(d\mathbf p-(\mathbf p^\top d\mathbf p)\mathbf 1)$ | (M6.17)，注意力里按行做 |
| $S=QK^\top/\sqrt{d_h}$ | $dQ=dS\,K/\sqrt{d_h}$，$dK=dS^\top Q/\sqrt{d_h}$ | 把 $K^\top$ 当作 (M6.19) 的 $W$；练习 12 |
| $O=PV$ | $dP=dO\,V^\top$，$dV=P^\top dO$ | 同上 |
| $\mathbf h=\mathbf x+f(\mathbf x)$ | $d\mathbf x=d\mathbf h+J_f^\top d\mathbf h$ | 残差，下一页 |
| 同一个参数在多处使用 | 各处梯度相加 | 如权重共享：嵌入矩阵既查表又做输出层，两处的梯度相加（[第 1 课 §7](../deepseek/lesson-01-transformer-anatomy.html#p7)） |

::: key 本页要点
- 线性层 $Y=XW$：$dX=dY\,W^\top$，$dW=X^\top dY$，$d\mathbf b$ 对 batch 求和；形状只有一种排法对得上。
- $dW$ 是每个 token 的外积之和，因为同一个 $W$ 被每个 token 用了一次。
- 反向两次矩阵乘法，各与前向一样大：训练每 token 约 $6N$ 次运算，推理约 $2N$ 次。
:::

::: note 这一页用在哪里
[第 1 课 §8](../deepseek/lesson-01-transformer-anatomy.html#p8)：参数量与每个 token 的运算量，“训练约 $6N$”的预告。[第 2 课 §6](../deepseek/lesson-02-gpt-training.html#p6)：训练算力 $C\approx6ND$。[第 12 课 §4](../deepseek/lesson-12-v3-infrastructure.html#p4)：FP8 训练把这三次矩阵乘法（报告称 Fprop、Dgrad、Wgrad）的输入都换成 8 位。[第 12 课 §5](../deepseek/lesson-12-v3-infrastructure.html#p5)：Zero Bubble 把 $dX$ 与 $dW$ 拆开调度。[第 16 课 §3](../deepseek/lesson-16-mhc.html#p3)：“前向左乘 $H$，反向左乘 $H^\top$”，就是本页列写法的 $\bar{\mathbf x}=W^\top\bar{\mathbf y}$ 对 $X$ 的每一列各用一次。
:::

# 残差：梯度高速公路 | x + f(x) 的导数、深层网络里的连乘、残差为什么能保住梯度 | 30

§3 说过，梯度从损失传回第一层，要依次乘上每一层雅可比的转置。层数一多，这就是几十、上百个矩阵的连乘。先用标量感受一下连乘有多可怕：每层让梯度乘 0.9，20 层后剩 $0.9^{20}=0.1216$，100 层后剩 $2.66\times10^{-5}$；每层乘 1.1，100 层后变成 $13{,}781$ 倍。sigmoid 的导数最大只有 0.25（§1），即使权重矩阵本身不缩小梯度（谱范数为 1，比如正交矩阵），十层 sigmoid 叠起来梯度也至多剩 $0.25^{10}=9.5\times10^{-7}$（这一步用到下面的不等式 (M6.20)）。前一种叫**梯度消失**（vanishing gradient），后一种叫**梯度爆炸**（exploding gradient）。一个普通的深层网络要想训练得动，每一层的“增益”都必须非常接近 1，这是一条刀刃。残差连接把刀刃变成了一条宽路。

## 连乘为什么危险

::: pre 谱范数：矩阵最多把向量拉长几倍 | M02 §4 系统地讲过；这里只用几条性质
矩阵 $A$ 的**谱范数** $\norm{A}_2=\max_{\mathbf x\ne\mathbf 0}\norm{A\mathbf x}/\norm{\mathbf x}$，是它能把一个向量拉长的最大倍数（[M02 §4](m02-rank-svd-lowrank.html#p4)）。由定义立得 $\norm{A\mathbf g}\le\norm{A}_2\norm{\mathbf g}$；连用两次得 $\norm{AB}_2\le\norm{A}_2\norm{B}_2$。另外 $\norm{A^\top}_2=\norm{A}_2$。
:::

设第 $l$ 层的雅可比是 $J_l$，顶层传下来的梯度是 $\bar{\mathbf h}_L$。反复用 (M6.13) 与上面的不等式：

::: eq 连乘的上界 | (M6.20)
$$\bar{\mathbf h}_0=J_1^\top J_2^\top\cdots J_L^\top\,\bar{\mathbf h}_L,\qquad \norm{\bar{\mathbf h}_0}\le\Big(\prod_{l=1}^{L}\norm{J_l}_2\Big)\norm{\bar{\mathbf h}_L}$$
只要每层都有 $\norm{J_l}_2\le0.9$，100 层后梯度就一定不超过顶层的 $2.66\times10^{-5}$。这是上界：它能证明“一定消失”，不能证明“一定爆炸”，爆炸要靠下面的实验来看。
:::

## 残差连接的导数：多出来的那个 I

残差连接让每一层不替换输入，而是往输入上加一个增量：$\mathbf h_l=\mathbf h_{l-1}+f_l(\mathbf h_{l-1})$（[第 1 课 §6](../deepseek/lesson-01-transformer-anatomy.html#p6)）。对 $\mathbf h_{l-1}$ 求雅可比，恒等映射 $\mathbf h\mapsto\mathbf h$ 的雅可比是单位矩阵 $I$，加法规则让两项分别求导再相加：

::: eq 残差块的雅可比与反向 | (M6.21)
$$\frac{\partial\mathbf h_l}{\partial\mathbf h_{l-1}}=I+J_{f_l},\qquad \bar{\mathbf h}_{l-1}=\bar{\mathbf h}_l+J_{f_l}^\top\,\bar{\mathbf h}_l$$
反向时，上游的梯度**原封不动**地传下去，分支只额外加上一份修正 $J_{f_l}^\top\bar{\mathbf h}_l$。
:::

把 $L$ 个这样的块串起来，连乘里的每个因子都是 $I+J_k$。从这里到本页结束，简记 $J_k=J_{f_k}$：它只是第 $k$ 个**分支**的雅可比，不是 (M6.20) 里整层的雅可比（整层的是 $I+J_k$）。展开这个乘积：

::: deriv 残差连乘的展开（以三层为例）
$\dfrac{\partial\mathbf h_3}{\partial\mathbf h_0}=(I+J_3)(I+J_2)(I+J_1)$ || 链式法则 (M6.12)，每个因子来自 (M6.21)
$=(I+J_3)\big(I+J_1+J_2+J_2J_1\big)$ || 先乘开后两个因子
$=I+(J_1+J_2+J_3)+(J_2J_1+J_3J_1+J_3J_2)+J_3J_2J_1$ || 再乘开：每一项对应“每层要么走捷径（取 $I$），要么走分支（取 $J_k$）”的一种选法
:::

::: eq 梯度高速公路 | (M6.22)
$$\frac{\partial\mathbf h_L}{\partial\mathbf h_l}=\prod_{k=l+1}^{L}(I+J_k)=I+\sum_{k}J_k+\sum_{k>k'}J_kJ_{k'}+\cdots+J_LJ_{L-1}\cdots J_{l+1}$$
连乘按链式法则的顺序排：$k$ 大的因子在左（$(I+J_L)\cdots(I+J_{l+1})$），展开后每一项里的 $J$ 也是下标大的在左，所以二次项写成 $J_kJ_{k'}$、$k>k'$。共 $2^{L-l}$ 条路径之和，第一项永远是 $I$：无论中间的层把梯度缩小多少，都有一条“一层也不经过”的路把顶层的梯度原样送到第 $l$ 层。没有残差时，只剩最后一项。
:::

He 等人在分析残差网络时写出了同样的结构：$\frac{\partial\mathcal E}{\partial\mathbf x_l}=\frac{\partial\mathcal E}{\partial\mathbf x_L}\big(1+\frac{\partial}{\partial\mathbf x_l}\sum_{i=l}^{L-1}\mathcal F(\mathbf x_i,\mathcal W_i)\big)$。论文的记号与本课的对应是：$\mathbf x_l$ 是第 $l$ 层的残差流（本课的 $\mathbf h_l$），$\mathcal F(\cdot,\mathcal W_i)$ 是带权重 $\mathcal W_i$ 的分支（本课的 $f$；论文从 $\mathbf x_i$ 算到 $\mathbf x_{i+1}$，下标比本课小 1），$\mathcal E$ 是损失；式中的“1”对向量应读作单位矩阵。他们指出第一项传递信息时不经过任何权重层，因此梯度“does not vanish even when the weights are arbitrarily small”[@he2016identity]。第 1 课的 (1.27) 就是这个式子。

::: fig figures/m06-residual-paths.svg | 图 M6.9 残差把雅可比的连乘展开成许多条路径之和 | wide
**看底部展开式里绿色的 I：它对应“三层全部走捷径”的那条路，不经过任何一个 $J$。** 上：无残差的三层网络，从 $\mathbf h_0$ 到 $\mathbf h_3$ 只有一条路，雅可比是 $J_3J_2J_1$。中：有残差时，每层的输出是“传送带上的原件 + 分支的增量”，反向时梯度沿绿色传送带原样传回，每经过一个 ⊕ 再从分支加回一份修正。下：乘积展开成 8 项，经过 $k$ 个分支的路径有 $\binom3k$ 条。示意图，路径计数由脚本枚举。
:::

::: ana 类比：残差流 = 传送带
**映射。** 残差流是一条传送带，每一层是带旁的一个工位：从带上取件（读 $\mathbf h_{l-1}$），加工，把一个增量放回带上（$+f_l$）。没人动的件原样往下走，对应恒等映射。反向时，“这件货最后卖了多少钱”（梯度）沿着传送带原路退回，退到每个工位时，工位再按自己的加工方式附上一份修正（$J_{f_l}^\top\bar{\mathbf h}$）。

**收益。** 它解释了为什么即使某些工位做得很差（$J_{f_l}$ 很小或很乱），信息和梯度仍能到达最底层：传送带本身不经过任何工位。

**失效的地方。** 真实的残差流是连续向量空间里的加法，工位放回的增量可以与带上的内容**抵消**：若某层的 $J_{f}$ 恰好有一个接近 $-1$ 的特征值，$I+J_f$ 在那个方向上就接近 0，那条路也会被堵住（He 等人的论证是：一个 mini-batch 里所有样本同时发生这种抵消的可能性很小）。另外，残差流上只有一次次累加，没有哪一步把已有的内容整体缩小，所以前向时它的尺度通常随层数增长，需要 LayerNorm 一类的操作控制（第 1 课）。
:::

## 数字：一百层的随机网络

公式说“有一条恒等通路”，实际能差多少？用一个随机初始化的网络直接算：宽 256、深 100，输入 $\mathbf h_0$ 的各分量独立地取自标准正态 $\mathcal N(0,1)$，每层权重 $W_l$ 的元素独立地取自 $\mathcal N(0,s^2/256)$，$s$ 是权重的尺度；无残差时 $\mathbf h_l=\tanh(W_l\mathbf h_{l-1})$，有残差时 $\mathbf h_l=\mathbf h_{l-1}+\alpha\,W_l\tanh(\mathbf h_{l-1})$，$\alpha$ 是分支的缩放。损失取顶层输出在一个随机单位向量上的投影，所以顶层梯度的长度是 1；然后用 (M6.13) 精确地一层层反传，记下每一层梯度的长度。

::: fig figures/m06-residual-gradnorm.svg | 图 M6.10 100 层随机网络里每一层的梯度范数 | wide
**看最左端（第 0 层）五个圆点的纵坐标：无残差时从 $6.8\times10^{-11}$ 到 $5.4\times10^6$，横跨 17 个数量级；有残差、分支缩小到 $\alpha=0.1$ 时是 1.22。** 无残差的网络只有在权重尺度 $s$ 恰到好处时才不至于消失或爆炸，而即使 $s=1$，梯度到第 0 层也只剩 5%。有残差时恒等通路托底；分支不缩小（$\alpha=1$）时，各分支的贡献仍会一层层累积，到底层放大了约 150 倍。浅色带是 5 个随机种子的范围：第 0 层的值依次为 $6.8\times10^{-11}$–$2.5\times10^{-10}$、0.053–0.19、$1.6\times10^6$–$8.9\times10^6$、146–231、1.17–1.32。全部由脚本计算（笔者的模拟，不是论文数据）。
:::

为什么 $\alpha=1$ 时仍会放大，而 $\alpha=0.1$ 时几乎不变？下面是笔者的粗略估计（近似推理，不是定理）。记分支 $\alpha W_l\tanh(\mathbf h_{l-1})$ 的雅可比为 $J=\alpha W_l\,\diag\big(\tanh'(\mathbf h_{l-1})\big)$，并用 $\E[\tanh'^2]$ 表示 $\tanh'$ 的平方在各分量上的平均。

::: deriv 每个残差块把梯度放大多少（笔者的粗略估计）
$\norm{\bar{\mathbf h}_{l-1}}^2=\norm{\bar{\mathbf h}_l+J^\top\bar{\mathbf h}_l}^2$ || 反向公式 (M6.21)
$=\norm{\bar{\mathbf h}_l}^2+2\,\bar{\mathbf h}_l^\top J^\top\bar{\mathbf h}_l+\norm{J^\top\bar{\mathbf h}_l}^2$ || 展开：$\norm{\mathbf a+\mathbf b}^2=\norm{\mathbf a}^2+2\mathbf a^\top\mathbf b+\norm{\mathbf b}^2$
$\approx\norm{\bar{\mathbf h}_l}^2+\norm{J^\top\bar{\mathbf h}_l}^2$ || 近似：随机初始化的 $W_l$ 正负对称，交叉项正负相消，平均约为 0
$\norm{J^\top\bar{\mathbf h}_l}^2\approx c\,\norm{\bar{\mathbf h}_l}^2$ || 其中 $c=\alpha^2s^2\,\E[\tanh'^2]$：$W_l^\top\bar{\mathbf h}_l$ 的 256 个分量各有方差 $\frac{s^2}{256}\norm{\bar{\mathbf h}_l}^2$，平方和约 $s^2\norm{\bar{\mathbf h}_l}^2$；再逐元素乘 $\alpha\tanh'$
$\norm{\bar{\mathbf h}_0}\approx(1+c)^{L/2}\,\norm{\bar{\mathbf h}_L}$ || 每层放大约 $\sqrt{1+c}$ 倍，连乘 $L=100$ 层
:::

代入数字。$\alpha=0.1$、$s=1$ 时，tanh 的输入大致是标准正态，$\E[\tanh'^2]\approx0.46$，$c\approx0.0046$，$(1.0046)^{50}\approx1.26$，与模拟的 1.22 吻合。$\alpha=1$ 时 $\alpha^2$ 大了一百倍；不过前向时 $\mathbf h$ 越传越大，tanh 渐渐饱和，$\E[\tanh'^2]$ 从第一层的 0.47 降到最后一层的 0.05（平均约 0.12），$c$ 也跟着逐层变小。把每层各自的 $\sqrt{1+c_l}$ 逐层乘起来，估计约 210 倍，与模拟的 146（5 个种子 146–231）是同一个量级。所以残差只保证有一条恒等通路，**分支必须足够小**，整条链才稳定。GPT-2 的报告在初始化时把残差层的权重乘以 $1/\sqrt{N}$，这里的 $N$ 是残差层的个数，不是 §6 的参数量 [@radford2019]（Hugging Face 与 nanoGPT 取 $N=2L$，见[第 1 课 §6](../deepseek/lesson-01-transformer-anatomy.html#p6)）。报告给的理由是控制残差流上随深度的累积；从上面的估计看，它同时也让每个分支在初始化时变小，起的正是 $\alpha$ 的作用（笔者的解读）。

## 捷径上不能乘东西

恒等通路之所以管用，是因为捷径上的局部导数**恰好是 $I$**。如果在捷径上乘一个标量 $\lambda$，(M6.22) 的第一项就变成 $\prod\lambda_i$：He 等人指出，所有 $\lambda_i>1$ 时它指数级变大，所有 $\lambda_i<1$ 时指数级变小直至消失，梯度被迫改走权重层，训练变难 [@he2016identity]。$\lambda=0.9$、100 层，捷径就只剩 $2.66\times10^{-5}$，和没有残差差不多。

第 16 课的 Hyper-Connections 把一条残差流扩成 $n$ 条，捷径上乘的不再是 $I$，而是一个可学的混合矩阵 $H$。同样的道理：$L$ 层之后，捷径的贡献是 $H$ 的连乘。一个看起来很温和的混合矩阵

$$H=\begin{pmatrix}1.2&0.3\\0.1&0.9\end{pmatrix},$$

每个元素都不大，连乘 60 次后谱范数 $\norm{H^{60}}_2$ 却达到 $2.84\times10^6$（[M11 §7](m11-doubly-stochastic-sinkhorn.html#p7)；这是说明现象的玩具例子）。真实模型里，mHC 的论文报告，27B 模型用 HC 时复合映射的增益最大接近 3000，而把 $H$ 约束为双随机矩阵（谱范数不超过 1）之后降低了约三个数量级 [@xie2025mhc]。论文的“增益”量的是复合映射最大的绝对行和（前向）与列和（反向），不是谱范数，所以这两个数不能直接相比。约束的数学在 [M11](m11-doubly-stochastic-sinkhorn.html)，模型的故事在 [第 16 课](../deepseek/lesson-16-mhc.html)。

## Pre-LN 与 Post-LN：LayerNorm 放在哪里

同一个道理还决定了 LayerNorm 该放在哪里。原版 Transformer 把它放在加法之后（Post-LN）：$\mathbf h_l=\LayerNorm\big(\mathbf h_{l-1}+f(\mathbf h_{l-1})\big)$，雅可比是 $J_{\mathrm{LN}}(I+J_f)$，每一层的恒等项都要先乘一个 $J_{\mathrm{LN}}$，连乘里不再有干净的 $I$。GPT-2 采用的 Pre-LN 把它放进分支里（[第 1 课 §6](../deepseek/lesson-01-transformer-anatomy.html#p6)）：$\mathbf h_l=\mathbf h_{l-1}+f\big(\LayerNorm(\mathbf h_{l-1})\big)$，雅可比是 $I+J_fJ_{\mathrm{LN}}$，恒等通路完好。这与 He 等人的条件一致：加法之后的操作也应当是恒等映射 [@he2016identity]。这一段是笔者按雅可比的结构推出的。Xiong 等人在简化假设下对两种放法初始化时梯度大小的分析，见[第 1 课 §6](../deepseek/lesson-01-transformer-anatomy.html#p6)；两种放法各自付出的代价，见[第 16 课 §1](../deepseek/lesson-16-mhc.html#p1)。

::: warn 残差不是万能的
在一维里看得最清楚。若每层分支的导数都是 $-0.1$，残差块的导数就是 $1-0.1=0.9$，20 层后同样只剩 $0.1216$：分支一致地抵消了恒等项。残差真正保证的是：**分支小的时候**，乘积接近 1，而不管这些小量是正是负（20 层 $1.01^{20}=1.22$，$0.99^{20}=0.82$）；没有残差时，同样“小”的层会让梯度变成 $0.01^{20}=10^{-40}$。
:::

::: key 本页要点
- 深层网络的梯度是雅可比的连乘，$\norm{\bar{\mathbf h}_0}\le\prod\norm{J_l}_2\norm{\bar{\mathbf h}_L}$：每层稍小于 1 就消失，稍大于 1 就爆炸。
- 残差块的雅可比是 $I+J_f$，反向时 $\bar{\mathbf h}_{l-1}=\bar{\mathbf h}_l+J_f^\top\bar{\mathbf h}_l$；连乘展开成 $2^L$ 条路径，其中一条是 $I$。
- 恒等通路要求两件事：分支足够小（GPT-2 按残差层数缩小分支的初始化），捷径上不乘任何东西（标量 $\lambda$、混合矩阵 $H$、Post-LN 都会破坏它；mHC 用双随机约束让连乘不再放大，恢复的是“不放大、均值守恒”这些性质，捷径本身并不等于 $I$，见[第 16 课 §4](../deepseek/lesson-16-mhc.html#p4)）。
:::

::: note 这一页用在哪里
[第 1 课 §6](../deepseek/lesson-01-transformer-anatomy.html#p6)：残差流、梯度高速公路 (1.27) 与 Pre-LN。[第 16 课 §1](../deepseek/lesson-16-mhc.html#p1)、[§3](../deepseek/lesson-16-mhc.html#p3)：残差流上的那个“1”，以及 Hyper-Connections 把它换成混合矩阵的连乘之后，增益怎样失控。[M11](m11-doubly-stochastic-sinkhorn.html)：双随机矩阵的谱范数为什么恰好是 1，连乘多少次都不会放大。
:::

# 练习与答案 | 12 题：概念、计算、推导 | 45

先自己做，再展开答案。计算题都能用纸笔或计算器完成；想核对的话，每题都可以用几行 NumPy 加中心差分验证。

## 概念题

:::: quiz 练习 1 · 什么时候前向模式更划算
§3 说训练用反向模式。（a）用 Baydin 等人给的代价估计说明原因。（b）举一个前向模式更划算的例子。
::: answer 看答案
（a）对 $\mathbf f:\R^n\to\R^m$，前向模式求完整雅可比约需 $n\cdot c\cdot\mathrm{ops}(f)$，反向模式约需 $m\cdot c\cdot\mathrm{ops}(f)$，$c<6$。训练时 $m=1$（一个标量损失），$n$ 是参数量（GPT-2 small 约 1.24 亿），反向一次就够，前向要 $n$ 次。

（b）输入少、输出多的时候。例如想知道 softmax 的温度 $\tau$（一个标量）变一点，词表里 50257 个概率各变多少：前向模式沿 $\tau$ 方向推一次（一个雅可比–向量积）就得到全部 50257 个导数；反向模式要对每个输出各做一次，共 50257 次。
:::
::::

:::: quiz 练习 2 · softmax 雅可比的行和
（a）由 $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$ 证明雅可比每一行的和为 0，并用 $\mathbf z=(2,1,0)$ 的数值矩阵验证第二行。（b）这件事与“logits 同加一个常数，softmax 不变”是什么关系？（c）由此说明：对 logits 做一步梯度下降，$\sum_kz_k$ 不会改变。
::: answer 看答案
（a）第 $i$ 行之和 $\sum_jp_i(\delta_{ij}-p_j)=p_i-p_i\sum_jp_j=p_i-p_i=0$。第二行：$-0.162803+0.184836-0.022033=0$。

（b）行和为 0 就是 $J\mathbf 1=\mathbf 0$：沿 $\mathbf 1$ 方向（所有 logits 同加一个常数）挪 logits，概率的一阶变化为 0。平移不变是一个精确的性质，行和为 0 是它的导数版本。

（c）交叉熵的梯度 $\mathbf p-\mathbf y$ 各分量之和为 $1-1=0$。一步更新 $\mathbf z\leftarrow\mathbf z-\eta(\mathbf p-\mathbf y)$ 让 $\sum_kz_k$ 改变 $-\eta\sum_k(p_k-y_k)=0$。一般地，损失只通过 softmax 依赖 $\mathbf z$ 时，$\bar{\mathbf z}=J^\top\bar{\mathbf p}$，而 $J$ 对称、行和为 0，所以 $\mathbf 1^\top\bar{\mathbf z}=(J\mathbf 1)^\top\bar{\mathbf p}=0$。
:::
::::

:::: quiz 练习 3 · Post-LN 与 Pre-LN 的雅可比
写出 Post-LN 块 $\mathbf h_l=\LayerNorm(\mathbf h_{l-1}+f(\mathbf h_{l-1}))$ 与 Pre-LN 块 $\mathbf h_l=\mathbf h_{l-1}+f(\LayerNorm(\mathbf h_{l-1}))$ 对 $\mathbf h_{l-1}$ 的雅可比（记 LayerNorm 的雅可比为 $J_{\mathrm{LN}}$，在各自的输入处取值）。$L$ 层连乘后，哪一种的展开式里含有单独的 $I$？
::: answer 看答案
Post-LN：链式法则，外层是 LN，内层是 $\mathbf h+f(\mathbf h)$，雅可比 $J_{\mathrm{LN}}\,(I+J_f)$。Pre-LN：恒等项求导得 $I$，分支是 $f\circ\LayerNorm$，雅可比 $I+J_f\,J_{\mathrm{LN}}$。

$L$ 层连乘：Pre-LN 是 $\prod(I+J_{f}J_{\mathrm{LN}})$，展开后第一项是 $I$；Post-LN 是 $\prod J_{\mathrm{LN}}(I+J_f)$，展开后“全走捷径”那一项是 $J_{\mathrm{LN}}^{(L)}\cdots J_{\mathrm{LN}}^{(1)}$，恒等通路每层都要经过一次 LayerNorm 的雅可比，不再是干净的 $I$。
:::
::::

## 计算题

:::: quiz 练习 4 · 线性近似与差分
$f(x)=\sqrt x$。（a）用 (M6.2) 在 $x=4$ 处估计 $\sqrt{4.1}$，误差多大？（b）取 $h=0.1$，分别用前向差分与中心差分估计 $f'(4)$，与真值比较。
::: answer 看答案
（a）$f'(x)=\frac1{2\sqrt x}$，$f'(4)=0.25$。估计 $\sqrt{4.1}\approx2+0.25\times0.1=2.025$；真值 $2.024846$，误差 $1.5\times10^{-4}$，量级正是 $h^2=0.01$ 乘一个系数（$\tfrac12\abs{f''(4)}=\tfrac1{64}$，$0.01/64=1.6\times10^{-4}$）。

（b）前向：$(\sqrt{4.1}-2)/0.1=0.248457$，误差 $1.5\times10^{-3}$。中心：$(\sqrt{4.1}-\sqrt{3.9})/0.2=0.250020$，误差 $2.0\times10^{-5}$。同样两次求值，中心差分准了约 80 倍。
:::
::::

:::: quiz 练习 5 · 梯度、方向导数、一步下降
$f(x,y)=e^xy^2$，在点 $(0,3)$。（a）求 $\nabla f$ 与 $\norm{\nabla f}$。（b）沿单位方向 $(-0.8,0.6)$ 的方向导数是多少？下降最快的单位方向是什么？（c）学习率 $\eta=0.01$ 走一步梯度下降，新的函数值是多少？线性近似预测的是多少？
::: answer 看答案
（a）$\partial f/\partial x=e^xy^2=9$，$\partial f/\partial y=2e^xy=6$。$\nabla f=(9,6)$，$\norm{\nabla f}=\sqrt{117}=10.8167$。

（b）$9\times(-0.8)+6\times0.6=-3.6$：沿这个方向 $f$ 在下降。下降最快的方向是 $-\nabla f/\norm{\nabla f}=(-0.8321,-0.5547)$，速率 $-10.8167$。

（c）新点 $(0,3)-0.01\times(9,6)=(-0.09,2.94)$，$f=e^{-0.09}\times2.94^2=7.8997$。线性预测 $9-0.01\times117=7.83$。真值略高于预测，差出的 0.07 来自二阶项。
:::
::::

:::: quiz 练习 6 · 带温度的交叉熵
logits $\mathbf s=(2,1,0)$，温度 $\tau=2$，真类别为 0。求 $\mathbf p=\softmax(\mathbf s/\tau)$、损失 $\ell$，以及 $\partial\ell/\partial\mathbf s$。与 $\tau=1$（§5 的例题）比较，梯度的真类别分量变大还是变小？
::: answer 看答案
$\mathbf s/\tau=(1,0.5,0)$，$e^{1}=2.71828$，$e^{0.5}=1.64872$，$e^0=1$，和为 $5.36700$，$\mathbf p=(0.506480,0.307196,0.186324)$。$\ell=-\ln0.506480=0.680270$。

$\partial\ell/\partial\mathbf s=(\mathbf p-\mathbf y)/\tau=(-0.493520,0.307196,0.186324)/2=(-0.246760,0.153598,0.093162)$。

$\tau=1$ 时真类别分量是 $-0.334759$；现在是 $-0.246760$，绝对值变小。温度升高有两个相反的效果：分布变平、$1-p_y$ 变大（从 0.335 升到 0.494），但整体再除以 $\tau=2$，后者占了上风。
:::
::::

:::: quiz 练习 7 · 线性层的反向
$Y=XW+\mathbf b$，其中

$$X=\begin{pmatrix}1&2\\3&-1\end{pmatrix},\quad W=\begin{pmatrix}1&0&-1\\2&1&0\end{pmatrix},\quad dY=\begin{pmatrix}1&0&2\\0&-1&1\end{pmatrix}.$$

（a）写出 $dX$、$dW$、$d\mathbf b$ 的形状并计算。（b）GPT-2 small 把 Q、K、V 三个投影合成一个 $[768,2304]$ 的矩阵。一批 1024 个 token，它的前向与反向各要多少次运算？
::: answer 看答案
（a）$X$ 是 $[2,2]$、$W$ 是 $[2,3]$、$dY$ 是 $[2,3]$。$dX=dY\,W^\top$ 是 $[2,3]\times[3,2]\to[2,2]$：

$$dX=\begin{pmatrix}1\cdot1+0\cdot0+2\cdot(-1)&1\cdot2+0\cdot1+2\cdot0\\0\cdot1+(-1)\cdot0+1\cdot(-1)&0\cdot2+(-1)\cdot1+1\cdot0\end{pmatrix}=\begin{pmatrix}-1&2\\-1&-1\end{pmatrix}.$$

$dW=X^\top dY$ 是 $[2,2]\times[2,3]\to[2,3]$：$X^\top=\begin{pmatrix}1&3\\2&-1\end{pmatrix}$，

$$dW=\begin{pmatrix}1&-3&5\\2&1&3\end{pmatrix}.$$

$d\mathbf b$ 是 $dY$ 按列求和：$(1,-1,3)$，形状 $[3]$。

（b）前向 $2\times1024\times768\times2304=3.62\times10^9$；反向两次同样大小的乘法，$7.25\times10^9$。
:::
::::

:::: quiz 练习 8 · 连乘与残差（一维）
（a）20 层、每层导数 0.9 的普通网络，梯度剩多少？（b）20 个残差块，每块分支的导数都是 $-0.1$，结果如何？为什么残差没帮上忙？（c）20 个残差块，分支导数都是 $+0.01$ 或都是 $-0.01$；对比 20 层每层导数 0.01 的普通网络。
::: answer 看答案
（a）$0.9^{20}=0.1216$。

（b）每块的导数是 $1+(-0.1)=0.9$，结果同样是 $0.1216$。分支每一层都朝同一个方向抵消恒等项，恒等通路被一致地削弱了。残差并不保证“梯度不变”，它保证的是乘积展开里有一项 $I$；如果其余项系统性地与它相消，也救不了。

（c）$1.01^{20}=1.2202$，$0.99^{20}=0.8179$：分支很小时，不论正负，乘积都在 1 附近。普通网络 $0.01^{20}=10^{-40}$，梯度彻底消失。这就是“分支要小”的意思（§7 的模拟与 GPT-2 的 $1/\sqrt N$ 初始化）。
:::
::::

:::: quiz 练习 9 · 手算一张带分叉的计算图
一个极小的“网络”：输入 $x=1$，权重 $w=0.5$，目标 $y^\star=1$。前向是 $u=wx$，$a=\tanh u$，$\hat y=wa$，损失 $\ell=\tfrac12(\hat y-y^\star)^2$。注意 $w$ 被用了两次：先乘输入，再乘输出（与第 1 课嵌入矩阵既查表又做输出层的权重共享同一个道理）。（a）列出求值轨迹，算出 $\ell$。（b）从 $\bar\ell=1$ 出发反向走一趟，求 $\partial\ell/\partial w$ 与 $\partial\ell/\partial x$，再直接对 $w$ 求导核对。（c）如果在 $\bar w$ 上用“赋值”代替“累加”，会得到什么？
::: answer 看答案
（a）$u=0.5$，$a=\tanh0.5=0.462117$，$\hat y=0.5\times0.462117=0.231059$，$\ell=\tfrac12(0.231059-1)^2=0.295635$。

（b）按前向的逆序，套 §4 的口诀：

- $\ell=\tfrac12(\hat y-y^\star)^2$：$\bar{\hat y}=\hat y-y^\star=-0.768941$。
- $\hat y=w\cdot a$ 是乘法，交换相乘：$\bar w\mathrel{+}=\bar{\hat y}\,a=-0.355341$，$\bar a=\bar{\hat y}\,w=-0.384471$。
- $a=\tanh u$，乘局部导数 $1-a^2=0.786448$：$\bar u=-0.302366$。
- $u=w\cdot x$，交换相乘：$\bar w\mathrel{+}=\bar u\,x=-0.302366$，$\bar x=\bar u\,w=-0.151183$。

所以 $\partial\ell/\partial w=-0.355341-0.302366=-0.657707$，$\partial\ell/\partial x=-0.151183$。直接求导核对：$\hat y=w\tanh(wx)$，由乘法规则与链式法则，$\frac{d\hat y}{dw}=\tanh(wx)+wx\big(1-\tanh^2(wx)\big)=0.462117+0.393224=0.855341$，于是 $\frac{\partial\ell}{\partial w}=(\hat y-y^\star)\times0.855341=-0.657707$，一致。乘法规则的两项，正好对应计算图上 $w$ 的两次使用。步长 $10^{-5}$ 的中心差分给出同样的 $-0.657707$ 与 $-0.151183$。

（c）$\bar w$ 的两份贡献，一份来自“乘输出”那一处（$-0.355341$），一份来自“乘输入”那一处（$-0.302366$）。按上面的顺序用赋值，最后留下 $-0.302366$，只有正确值的 46%；换个顺序则留下 $-0.355341$。两种都错，程序却不会报错，这正是 §4 那个常见错误。
:::
::::

## 推导题

:::: quiz 练习 10 · sigmoid、二元交叉熵与两类 softmax
（a）证明 $\softmax\big((z,0)\big)$ 的第一个分量等于 $\sigma(z)$。（b）从 $\sigma'=\sigma(1-\sigma)$ 出发，推出二元交叉熵 $\ell=-[y\log\sigma(z)+(1-y)\log(1-\sigma(z))]$ 对 $z$ 的导数是 $\sigma(z)-y$。（c）第 3 课的奖励模型损失 $-\log\sigma(r_w-r_l)$ 对 $r_w$ 和对 $r_l$ 的导数各是多少？
::: answer 看答案
（a）$\frac{e^z}{e^z+e^0}=\frac{e^z}{e^z+1}=\frac{1}{1+e^{-z}}=\sigma(z)$，最后一步分子分母同除以 $e^z$。

（b）由 (M6.4)，$\frac{d}{dz}\log\sigma=1-\sigma$，$\frac{d}{dz}\log(1-\sigma)=-\sigma$。所以 $\frac{d\ell}{dz}=-[y(1-\sigma)+(1-y)(-\sigma)]=-y+y\sigma+\sigma-y\sigma=\sigma-y$。

（c）令 $\Delta=r_w-r_l$，$\frac{d}{d\Delta}(-\log\sigma(\Delta))=-(1-\sigma(\Delta))=\sigma(\Delta)-1$。链式法则：$\partial\Delta/\partial r_w=1$，$\partial\Delta/\partial r_l=-1$，所以对 $r_w$ 的导数是 $\sigma(\Delta)-1$（负，梯度下降会抬高 $r_w$），对 $r_l$ 的导数是 $1-\sigma(\Delta)$（正，会压低 $r_l$）。
:::
::::

:::: quiz 练习 11 · log-sum-exp 的梯度与海森矩阵
记 $\mathrm{LSE}(\mathbf z)=\log\sum_ke^{z_k}$。（a）证明 $\nabla\mathrm{LSE}(\mathbf z)=\softmax(\mathbf z)$。（b）证明它的二阶导数矩阵（海森矩阵，Hessian，$H_{ij}=\partial^2\mathrm{LSE}/\partial z_i\partial z_j$）就是 softmax 的雅可比 $\diag(\mathbf p)-\mathbf p\mathbf p^\top$。（c）由此说明交叉熵 $\ell=-z_y+\mathrm{LSE}(\mathbf z)$ 是 $\mathbf z$ 的凸函数（海森矩阵半正定）。
::: answer 看答案
（a）链式法则：$\frac{\partial}{\partial z_i}\log\sum_ke^{z_k}=\frac{e^{z_i}}{\sum_ke^{z_k}}=p_i$。

（b）$H_{ij}=\frac{\partial}{\partial z_j}\big(\frac{\partial\mathrm{LSE}}{\partial z_i}\big)=\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)$，就是 (M6.15)。用二阶中心差分在 $\mathbf z=(2,1,0)$ 处数值计算海森矩阵，得到的正是 §5 例题的那个矩阵（对角 0.222695、0.184836、0.081925，非对角 −0.162803、−0.059892、−0.022033）。

（c）$-z_y$ 是线性的，海森矩阵为 0；所以 $\ell$ 的海森矩阵就是 $\diag(\mathbf p)-\mathbf p\mathbf p^\top$。§5 证明过它半正定（$\mathbf v^\top J\mathbf v$ 是一个方差）。海森矩阵处处半正定的函数是凸函数，所以交叉熵对 logits 是凸的：只看最后一层的 logits，不存在坏的局部极小。（整个网络对参数的损失一般不是凸的，因为 logits 本身是参数的非线性函数。）
:::
::::

:::: quiz 练习 12 · 注意力打分的反向
一个头里，$Q,K\in\R^{T\times d_h}$（每行一个 token），打分 $S=QK^\top\in\R^{T\times T}$（为简单起见省去 $1/\sqrt{d_h}$）。上游传来 $dS$。（a）推导 $dQ=dS\,K$、$dK=dS^\top Q$，写出形状。（b）为什么 $dK$ 的第 $j$ 行要对所有查询 $t$ 求和？
::: answer 看答案
（a）逐元素：$S_{tj}=\sum_aQ_{ta}K_{ja}$。$Q_{ta}$ 出现在第 $t$ 行的每一个 $S_{tj}$ 里，局部导数 $K_{ja}$，所以 $\partial\mathcal L/\partial Q_{ta}=\sum_jdS_{tj}K_{ja}=(dS\,K)_{ta}$，形状 $[T,T]\times[T,d_h]\to[T,d_h]$。$K_{ja}$ 出现在第 $j$ 列的每一个 $S_{tj}$ 里，局部导数 $Q_{ta}$，所以 $\partial\mathcal L/\partial K_{ja}=\sum_tdS_{tj}Q_{ta}=(dS^\top Q)_{ja}$，形状同样是 $[T,d_h]$。也可以直接套 (M6.19)：把 $K^\top$ 看成线性层的权重 $W$，$dQ=dS\,(K^\top)^\top=dS\,K$，$d(K^\top)=Q^\top dS$，转置得 $dK=dS^\top Q$。对随机的 $4\times3$ 矩阵做中心差分，与公式的最大误差约 $10^{-9}$。

（b）第 $j$ 个键被所有查询 $t$ 用来打分（因果掩码下是所有 $t\ge j$），这是 §4 的“分叉”：一个量被用了几次，它的梯度就是几处贡献之和。第 4 课的 KV cache 说的是同一件事的前向版本：一个 token 的键会被之后所有的查询反复读取。
:::
::::

```refs
- id: baydin2018
  title: "Automatic Differentiation in Machine Learning: a Survey"
  a: Baydin, A. G., Pearlmutter, B. A., Radul, A. A., Siskind, J. M. (2018). Journal of Machine Learning Research 18(153):1–43. arXiv:1502.05767（读的是 v4）
  url: https://arxiv.org/abs/1502.05767
  finding: 系统比较数值差分、符号求导与自动微分；§2.1 用图 3 展示差分误差随步长的 V 形（截断 vs 舍入），§3 用 $y=\ln x_1+x_1x_2-\sin x_2$ 的计算图（图 4、表 3）演示前向与反向模式，并指出反向模式一次就得到标量函数的整个梯度、代价是求值的常数倍（$c<6$，通常 2–3）。
  matters: 本课的数值导数图、计算图例题与“为什么训练用反向模式”都以它为出处，也借它交代反向传播的历史。
- id: rumelhart1986
  title: "Learning representations by back-propagating errors"
  a: Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986). Nature 323(6088):533–536. doi:10.1038/323533a0（只核对了 Crossref 元数据，正文未读）
  url: https://doi.org/10.1038/323533a0
  finding: 题目即主旨：通过把误差反向传播来学习网络的内部表示。按 Baydin 等人的梳理，是这篇论文让反向传播在机器学习界广为人知（本课只核对了题录，正文未读）。
  matters: 本课交代反向传播的来历时引它；算法本身的出处与更早的先行工作以 Baydin 等人的综述为准。
- id: pytorch-autograd
  kind: artefact
  title: "A Gentle Introduction to torch.autograd（PyTorch 教程）"
  a: PyTorch（教程 2.14 版）. 读取于 2026-09-24
  url: https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
  finding: 称 torch.autograd 是计算向量–雅可比积 $J^\top\vec v$ 的引擎；前向时把运算记录成有向无环图，反向时按链式法则把梯度传回叶子张量，并累加到各自的 .grad 属性里。
  matters: 本课用它说明主流框架实现的正是 (M6.13) 的向量–雅可比积，以及“梯度会累加、每步要清零”的来由。
- id: kaplan2020
  title: "Scaling Laws for Neural Language Models"
  a: Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., Amodei, D. (2020). arXiv:2001.08361（读的是 v1）
  url: https://arxiv.org/abs/2001.08361
  finding: §2.1 把前向每 token 的运算量估为约 $2N$（$N$ 不含词表与位置嵌入），并以“反向约为前向的两倍”把训练的非嵌入算力定义为每 token $C\approx6N$；§1.1 的记号表把整个训练记为 $C\approx6NBS$。
  matters: 本课 §6 从线性层的两次反向矩阵乘法推出“反向是前向的两倍”，与它的定义对上，也是第 2 课 $6ND$ 的来源。
- id: torch-linear
  kind: artefact
  title: PyTorch 文档：torch.nn.Linear
  a: PyTorch（文档 2.14 版）. 读取于 2026-09-24（与 M01 同一出处）
  url: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
  finding: nn.Linear 计算 $y=xA^\top+b$；可学习权重的形状是 (out_features, in_features)。
  matters: 本课用它把 (M6.19) 的行写法翻译成 PyTorch 的存法，得到第 12 课用的 $dX=dY\,W$、$dW=dY^\top X$。
- id: gpt2-config
  kind: artefact
  title: GPT-2 small 模型配置（Hugging Face openai-community/gpt2, config.json）
  a: OpenAI / Hugging Face. 读取于 2026-09-24（与 M01 同一出处）
  url: https://huggingface.co/openai-community/gpt2/blob/main/config.json
  finding: GPT-2 small 的 n_embd = 768、n_head = 12、n_layer = 12、n_positions = 1024、vocab_size = 50257。
  matters: 本课“按真实形状”算的例子（768 维、50257 个词）都取自这份配置，与第 1 课、M01 一致。
- id: radford2019
  kind: artefact
  title: "Language Models are Unsupervised Multitask Learners"
  a: Radford, Wu, Child, Luan, Amodei, Sutskever (2019). OpenAI 技术报告，无 arXiv（与第 1 课同一出处）
  url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  finding: §2.3 说为了照顾残差路径上随深度的累积，初始化时把残差层的权重乘以 $1/\sqrt N$，$N$ 是残差层的数目。
  matters: 本课 §7 用它说明“分支要小”在真实模型里的一种做法，并借此交代这里的 $N$ 与 §6 的参数量 $N$ 不是同一个量。
- id: he2016identity
  title: "Identity Mappings in Deep Residual Networks"
  a: He, K., Zhang, X., Ren, S., Sun, J. (2016). ECCV 2016. arXiv:1603.05027（读的是 v3）
  url: https://arxiv.org/abs/1603.05027
  finding: 式 (4)(5) 表明恒等捷径让任意两层之间的梯度含一项不经过权重层的直通项；式 (7)(8) 表明捷径乘以标量 $\lambda$ 时这一项变成 $\prod\lambda_i$，随深度指数级放大或消失；并要求加法之后的操作也是恒等映射。
  matters: 本课 §7 的“梯度高速公路”(M6.22)、“捷径上不能乘东西”以及 Pre-LN 与 Post-LN 的比较都以它为依据。
- id: xie2025mhc
  title: "mHC: Manifold-Constrained Hyper-Connections"
  a: Xie, Z., Wei, Y., Cao, H., Zhao, C., Deng, C., Li, J., et al. (2025). arXiv:2512.24880（读的是 v2）
  url: https://arxiv.org/abs/2512.24880
  finding: 用 Sinkhorn–Knopp 把超连接的残差混合矩阵投影成双随机矩阵，使复合映射不再放大；§5.4 报告 HC 的复合增益最大接近 3000，mHC 把它降低约三个数量级。
  matters: 本课 §7 用它说明“捷径上乘矩阵”的危险与修复办法，是通往第 16 课和 M11 的桥。
```
