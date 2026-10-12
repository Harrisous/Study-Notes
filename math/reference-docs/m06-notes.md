# M06 补充核实：微积分与反向传播

> 写 `math/src/m06-calculus-backprop.md` 时新核实或新算出的内容（2026-09-24）。共享研究笔记 `math-refs-doc.md` 对 M06 只给了 N41、N42（softmax 与 $p-y$）、
> F06-1、F06-2、W16、W17，其余教科书结论（导数法则、链式法则、线性层反传）不需要文献；下面是正文里用到的**外部事实**与**派生数字**。
> 外部来源给出不超过 15 个英文词的原文短语与位置；派生数字给出表达式，脚本在写课者的 scratch 目录（`m06-calculus-backprop/nums.py`、`resid_exp.py`），
> 图的数值由 `math/figures/gen/m06_*.py` 重新计算。

## 1. 外部来源（本次亲自读过）

| 短键 | 文献 | 读了什么 | 用在 |
|---|---|---|---|
| `baydin2018` | Baydin, Pearlmutter, Radul, Siskind, “Automatic Differentiation in Machine Learning: a Survey”，arXiv:1502.05767 **v4**（2018-02-05）；arXiv 页 journal-ref：JMLR 18(153):1–43, 2018 | arXiv HTML 全文：§2.1 式 (1)(2) 与图 3，§3 图 4 与表 3，§3.2 反向模式代价，§3.3 历史 | §1 数值导数图、§3 前向/反向模式、§4 计算图例题 |
| `he2016identity` | He, Zhang, Ren, Sun, “Identity Mappings in Deep Residual Networks”，arXiv:1603.05027 **v3**（2016-07-25），comments “ECCV 2016 camera-ready” | arXiv HTML 全文：§2 式 (4)(5)，§3 式 (7)(8) | §7 残差 |
| `kaplan2020` | Kaplan et al., “Scaling Laws for Neural Language Models”，arXiv:2001.08361 v1 | §2.1 | §6 反向约为前向两倍 → $6N$ |
| `rumelhart1986` | Rumelhart, Hinton, Williams, “Learning representations by back-propagating errors”，Nature 323(6088):533–536，1986-10 | **只核对了 Crossref 元数据**（DOI 10.1038/323533a0）；nature.com 对脚本返回 150 字节拦截页，按规则未绕过 | §4 历史注记（内容经 `baydin2018` §3.3 转述） |
| `pytorch-autograd` | PyTorch 教程 “A Gentle Introduction to torch.autograd”（docs.pytorch.org，2026-09-24 读取） | “Optional Reading: Vector Calculus using autograd” 一节与 “Computational Graph” 一节 | §3 VJP、§4 梯度累加 |

**引文与位置：**

- **B1 数值导数的两种误差（`baydin2018` §2.1）。** 前向差分 $\frac{f(x+h)-f(x)}{h}$ 的截断误差 $O(h)$；中心差分 $\frac{f(x+h)-f(x-h)}{2h}+O(h^2)$（式 (2)），“the first-order errors cancel”。
  截断误差随 $h\to0$ 趋于 0，“as h is decreased, round-off error increases and becomes dominant (Figure 3)”。
  图 3：函数 $f(x)=64x(1-x)(1-2x)^2(1-8x+8x^2)^2$（截断的 logistic 映射），在 $x_0=0.2$ 处画 $E_{\rm forward}(h)$ 与 $E_{\rm center}(h)$。
  用数值差分求 $\R^n\to\R^m$ 的雅可比要 $2mn$ 次函数求值（§2.1），梯度的 $O(n)$ 代价是它在机器学习里的主要障碍（同节，转述）。
- **B2 计算图例题（`baydin2018` 图 4、表 3）。** $y=f(x_1,x_2)=\ln x_1+x_1x_2-\sin x_2$，在 $(2,5)$：$v_1=\ln2$，$v_2=10$，$v_3=\sin5$，$v_4=0.693+10$，$v_5=10.693+0.959$，
  $y=11.652$；伴随（adjoint）$\bar v_5=1$ 起倒推，$\bar x_1=5.5$，$\bar x_2=1.716$；$\bar v_0$ 由两条路径的贡献相加（表 3 右栏）。
- **B3 反向模式的代价（`baydin2018` §3.2）。** $f:\R^n\to\R$ 时一次反向就得到整个梯度，前向模式要 $n$ 次；一般地前向模式算雅可比需 $n\,c\,\mathrm{ops}(f)$，
  反向模式需 $m\,c\,\mathrm{ops}(f)$，“c is a constant guaranteed to be c<6 and typically c∼[2,3]”（引 Griewank & Walther 2008）。
  反向模式可直接算转置雅可比乘向量 $J_f^\top\mathbf r$（令 $\bar{\mathbf y}=\mathbf r$）。代价是存储：最坏情形与运算数成正比，靠 checkpointing 等缓解（同节，转述）。
- **B4 历史（`baydin2018` §3.3，转述）。** 反向模式最早的公开描述常引 Linnainmaa (1970) 硕士论文；Werbos (1974)；Speelpenning (1980) 给出第一个真正自动的实现；
  机器学习界多次重新发明（如 Parker 1985），“brought to fame by Rumelhart et al. (1986)”。
- **B5 He 等式 (4)(5)（`he2016identity` §2）。** $\mathbf x_L=\mathbf x_l+\sum_{i=l}^{L-1}\mathcal F(\mathbf x_i,\mathcal W_i)$；
  $\frac{\partial\mathcal E}{\partial\mathbf x_l}=\frac{\partial\mathcal E}{\partial\mathbf x_L}\big(1+\frac{\partial}{\partial\mathbf x_l}\sum_{i=l}^{L-1}\mathcal F(\mathbf x_i,\mathcal W_i)\big)$。
  论文说第一项“propagates information directly without concerning any weight layers”，并称梯度 “does not vanish even when the weights are arbitrarily small”。
  前提（§2）：捷径是恒等映射，且加法之后的函数 $f$ 也是恒等映射。原文的“1”对向量应读作单位矩阵（笔者注）。
- **B6 He 等式 (7)(8)（`he2016identity` §3）。** 捷径换成标量 $\lambda_l$ 时第一项变成 $\prod_{i=l}^{L-1}\lambda_i$：全部 $\lambda_i>1$ 时“exponentially large”，全部 $<1$ 时“exponentially small and vanish”。
- **B7 Kaplan §2.1。** “Accounting for the backwards pass (approximately twice the compute as the forwards pass)” 定义 $C\approx6N$ 每 token。
  （与 lesson-02-doc D18 一致。）
- **B8 PyTorch 教程。** “torch.autograd is an engine for computing vector-Jacobian product”，即给定 $\vec v$ 计算 $J^\top\cdot\vec v$；
  反向时 autograd 按链式法则把梯度 “accumulates them in the respective tensor’s .grad attribute”。

## 2. 交叉引用（本次未重核）

- GPT-2 small：$d=768$、$V=50257$（M01 的 `gpt2-config` 卡片）；FFN 中间维 3072、总参数 124,439,808、非嵌入 85,056,000（lesson-01-doc §4.1）。
- `torch.nn.Linear` 计算 $y=xA^\top+b$，权重形状 (out_features, in_features)（M01 的 `torch-linear` 卡片）。
- 残差原论文 He et al. 2016 CVPR（lesson-01-doc `he2016`）。GPT-2 残差分支初始化乘 $1/\sqrt N$（lesson-01 §8，正文只链接不复述数字）。
- mHC：HC 复合增益最大近 3000、mHC 降约三个数量级（math-refs-doc C16）；无约束混合矩阵 $H=\begin{pmatrix}1.2&0.3\\0.1&0.9\end{pmatrix}$ 的 $\|H^{60}\|_2=2.84\times10^6$（N30）。
- 浮点格式最大值：FP32 $3.4028\times10^{38}$、FP16 65504（math-refs-doc N1、N3）。

## 3. 派生数字（Python 3.11 + numpy 1.26 复算）

| 编号 | 量 | 表达式 | 值 |
|---|---|---|---|
| D1 | 切线近似 | $3.01^2$ vs $9+6\times0.01$ | 9.0601 vs 9.06，误差 $10^{-4}=h^2$ |
| D2 | sigmoid 及导数 | $\sigma(z)$，$\sigma(1-\sigma)$，$1-\sigma$ | $z=0$：0.5、0.25、0.5；$z=2$：0.880797、0.104994、0.119203；$z=5$：0.993307、0.006648、0.006693 |
| D3 | $\sqrt{4.1}$ 的线性近似 | $2+0.1/4$ | 2.025 vs 2.024846 |
| D4 | Baydin 图 3 的函数 | $f(0.2)$，$f'(0.2)$（多项式精确求导） | 0.289014，9.0660864 |
| D5 | 数值导数误差（float64，$x_0=0.2$） | 前向 / 中心 | $h=10^{-2}$：0.272 / 0.0657；$10^{-4}$：3.36e-3 / 6.58e-6；$10^{-6}$：3.37e-5 / 2.41e-9；$10^{-8}$：2.96e-7 / 3.15e-7；$10^{-10}$：6.29e-5 / 2.23e-5；$10^{-12}$：8.22e-3 / 2.72e-4 |
| D6 | $e^x$ 在 1 处 | 前向 / 中心，$h=10^{-3}$ | 1.36e-3 / 4.53e-7 |
| D7 | 梯度与方向导数 | $f=x^2+3y^2$ 在 $(2,1)$ | $\nabla f=(4,6)$，$\|\nabla f\|=7.2111$；沿 $(0.6,0.8)$：7.2；沿 $(1,0)$：4 |
| D8 | 一步梯度下降 | $\eta=0.1$ | $(1.6,0.4)$，$f$：7 → 3.04；线性预测 1.8，差 1.24 = 二阶项 $\tfrac{\eta^2}2\mathbf g^\top H\mathbf g$ |
| D9 | 链式法则算例 | $\mathbf u=(x_1+x_2,x_1x_2,x_2^2)$，$y=u_1u_2+u_3$，$\mathbf x=(1,2)$ | $\mathbf u=(3,2,4)$，$y=10$，$\nabla y=(8,9)$，差分核对一致 |
| D10 | Baydin 例的差分核对 | 中心差分 $h=10^{-5}$ | 5.49999999997、1.71633781445（解析 5.5、1.71633781454） |
| D11 | softmax 与 CE（同 N41、N42） | $\mathbf z=(2,1,0)$ | 雅可比特征值 0、0.118553、0.370904（半正定，秩 2） |
| D12 | 雅可比 × 向量 | $\mathbf g=(0.3,-1.2,0.5)$ | $J\mathbf g=\mathbf p\odot(\mathbf g-\mathbf p^\top\mathbf g)=(0.232227,-0.281661,0.049435)$ |
| D13 | 完整 softmax 雅可比的大小（GPT-2 词表） | $50257^2$，×4 字节 | 2,525,766,049 个元素，10.10 GB（9.41 GiB）每个位置 |
| D14 | $e^z$ 溢出阈值 | $\ln(\max)$ | float64 709.78，FP32 88.72，FP16 11.09 |
| D15 | 温度 | $\mathbf z/\tau$ | $\tau=0.5$：$\mathbf p=(0.866813,0.117310,0.015876)$，$\partial\ell/\partial\mathbf z=(-0.266373,0.234621,0.031752)$；$\tau=2$：$(0.506480,0.307196,0.186324)$，$(-0.246760,0.153598,0.093162)$ |
| D16 | 线性层算例 | $X=\begin{pmatrix}1&0&2\\-1&1&0\end{pmatrix}$，$W=\begin{pmatrix}1&-1\\0&2\\1&1\end{pmatrix}$，$\mathbf b=(0,1)$，目标 $T=\begin{pmatrix}2&2\\0&3\end{pmatrix}$ | $Y=\begin{pmatrix}3&2\\-1&4\end{pmatrix}$，$G=\begin{pmatrix}1&0\\-1&1\end{pmatrix}$，$L=1.5$；$dW=\begin{pmatrix}2&-1\\-1&1\\2&0\end{pmatrix}$，$dX=\begin{pmatrix}1&0&1\\-2&2&0\end{pmatrix}$，$d\mathbf b=(0,1)$；差分核对一致 |
| D17 | GPT-2 FFN 上投影的 FLOPs（1024 token） | $2Bnm$，$4Bnm$ | 前向 4.83e9，反向 9.66e9，合计 1.45e10 $=6\times(768\cdot3072)\times1024$ |
| D18 | 连乘 | $0.9^k$，$1.1^k$，$0.25^k$ | $0.9^{20}=0.1216$，$0.9^{24}=0.0798$，$0.9^{50}=0.00515$，$0.9^{100}=2.66\times10^{-5}$；$1.1^{50}=117.4$，$1.1^{100}=13{,}781$；$0.25^{10}=9.54\times10^{-7}$ |
| D19 | 随机网络的梯度范数（笔者模拟，图 M6.10 的脚本 `m06_residual_gradnorm.py`；审校时插入图 M6.7 后由 M6.9 改号） | 宽 256、深 100，tanh，$W$ 的元素 $\sim\mathcal N(0,s^2/256)$（正文记权重尺度为 $s$），读出向量单位长，seed 0 | 无残差 $s=0.8$：第 0 层梯度范数 $6.8\times10^{-11}$（5 个种子 $6.8\times10^{-11}$–$2.5\times10^{-10}$）；$s=1$：0.053（0.053–0.19）；$s=2$：$5.4\times10^6$（$1.6\times10^6$–$8.9\times10^6$）；残差 $\mathbf h+\alpha W\tanh\mathbf h$，$\alpha=1$：146（146–231）；$\alpha=0.1$：1.22（1.17–1.32） |
| D20 | tanh 的向量–雅可比积 | $\mathbf u=(0.5,-1,2)$，$\bar{\mathbf h}=(0.2,-0.5,1)$ | $\tanh'=(0.786448,0.419974,0.070651)$，$\bar{\mathbf u}=(0.157290,-0.209987,0.070651)$ |
| D21 | 沿 $-(\mathbf p-\mathbf y)$ 改 logits 一步 | $\mathbf z=(2,1,0)$，步长 1 | $\mathbf z'=(2.334759,0.755272,-0.090031)$，$\mathbf p'=(0.772453,0.159188,0.068360)$，损失 0.407606 → 0.258184 |
| D22 | 饱和时交叉熵 vs 平方误差 | $\mathbf z=(10,0,0)$，真类别 1 | $\mathbf p=(0.999909,0.000045,0.000045)$；CE 损失 10.0001、梯度长度 1.4141；平方误差损失 0.99986、梯度 $J^\top(\mathbf p-\mathbf y)$ 长度 $1.70\times10^{-4}$ |
| D23 | 一般目标分布 | $\mathbf q=(0.7,0.2,0.1)$ | $\mathbf p-\mathbf q=(-0.034759,0.044728,-0.009969)$，差分核对一致 |
| D24 | 练习数字 | 见各题 | $\sqrt{4.1}$ 前向差分 0.248457、中心 0.250020；$e^xy^2$ 在 $(0,3)$：梯度 $(9,6)$，一步后 7.8997（线性预测 7.83）；$\tau=2$ 时 $\ell=0.680270$；练习 7 的 $dW=\begin{pmatrix}1&-3&5\\2&1&3\end{pmatrix}$、$dX=\begin{pmatrix}-1&2\\-1&-1\end{pmatrix}$、$d\mathbf b=(1,-1,3)$；QKV 投影 $[768,2304]$、1024 token：前向 $3.62\times10^9$、反向 $7.25\times10^9$；1024 个位置的完整 softmax 雅可比 $1.03\times10^{13}$ 字节 vs $\mathbf p-\mathbf y$ 205,852,672 字节；$1.01^{20}=1.2202$、$0.99^{20}=0.8179$；LSE 海森矩阵的二阶差分等于 N42 的矩阵；注意力打分反向公式对随机 $4\times3$ 矩阵的差分误差 $1.0\times10^{-9}$ |
| D25 | 残差放大的粗略估计（笔者推导） | $(1+c)^{L/2}$，$c\approx\alpha^2s^2\E[\tanh'^2]$ | $\E[\tanh'(u)^2]$，$u\sim\mathcal N(0,1)$：0.4637（20 万样本）；$\alpha=0.1$、$s=1$：$c\approx0.0046$，$(1.0046)^{50}\approx1.26$，模拟 1.22；逐层用该层实际的 $\E[\tanh'(\mathbf h_l)^2]$ 算 $\prod_l(1+c_l)^{1/2}$（seed 0）：$\alpha=0.1$ 得 1.243，$\alpha=1$ 得 214（$\E[\tanh'^2]$ 从 0.467 降到 0.052，平均 0.115），模拟 146 |

## 4. 审校补充（2026-09-24，教学审校时新增的派生数字）

| 编号 | 量 | 表达式 | 值 |
|---|---|---|---|
| D26 | 两个基本极限在 $h=10^{-3}$ | $(e^h-1)/h$，$\ln(1+h)/h$ | 1.0005002，0.9995003 |
| D27 | 多元一阶近似的核对 | $f=x^2y+3y$，$(2,1)\to(2.01,1.01)$ | 预测增量 $4\times0.01+7\times0.01=0.11$；真值 $7.110501-7=0.110501$，差 $5.0\times10^{-4}$ |
| D28 | 分叉处误用赋值（Baydin 例） | 按 §4 推导的顺序，用 `=` 代替 `+=` | $\bar v_{-1}=0.5$（正确 5.5），$\bar v_0=2$（正确 1.7163） |
| D29 | 饱和时交叉熵 vs 平方误差（图 M6.7 的脚本 `m06_ce_vs_mse.py`） | $\mathbf z=(m,0,0)$，梯度的 2-范数 | 真类别 1：$m=0$ 时 CE 0.8165、MSE 0.2722；$m=4$：1.377、0.06169；$m=10$：1.4141、$1.698\times10^{-4}$（比值 8327）。真类别 0：$m=10$ 时 CE $1.112\times10^{-4}$、MSE $1.514\times10^{-8}$ |
| D30 | 无约束混合矩阵的连乘（复核 N30） | $\norm{H^{60}}_2$，$H=\begin{pmatrix}1.2&0.3\\0.1&0.9\end{pmatrix}$ | $2.836\times10^6$ |

## 5. 数学审校（2026-09-24）

全部派生数字（D1–D30）与 10 个生成脚本用 Python 3 + numpy 重算过，脚本输出与现有 SVG 逐字节一致。新增与改动：

| 编号 | 量 / 事实 | 表达式或来源 | 值 |
|---|---|---|---|
| D31 | GPT-2 small 反嵌入（输出层）每 token 训练运算量 | $6\times768\times50257$ | $2.32\times10^8$（主项 $6N=5.10\times10^8$ 的 45%）；Kaplan 的 $N$ 不含它，正文 §6 例题补了这一句 |
| D32 | $\E[\tanh'(u)^2]$，$u\sim\mathcal N(0,1)$（复核 D25） | 200 万样本 | 0.4646；$(1+0.01\times0.4646)^{50}=1.261$ |
| G1 | GPT-2 报告的残差初始化（`radford2019` §2.3，本次重读 PDF 文本） | “accounts for the accumulation on the residual path with model depth”；“N is the number of residual layers” | 正文 §7 据此说明 $1/\sqrt N$ 里的 $N$ 是残差层数，不是 §6 的参数量 |

记号改动：§2 方向导数的夹角由 $\theta$ 改为 $\varphi$（同页 (M6.9) 用 $\theta$ 表示参数）；§6 线性层例题的目标矩阵由 $T$ 改为 $Y^\star$（$T$ 留给序列长度，练习 12 在用）；§7 从 (M6.21) 起的 $J_k$ 明确为分支雅可比 $J_{f_k}$，与 (M6.20) 的整层雅可比区分。

## 6. 教学审校第二轮（2026-09-24）

| 编号 | 量 / 事实 | 表达式或来源 | 值 |
|---|---|---|---|
| D33 | 新练习 9：带分叉的计算图（$w$ 用两次） | $x=1$，$w=0.5$，$y^\star=1$；$u=wx$，$a=\tanh u$，$\hat y=wa$，$\ell=\tfrac12(\hat y-y^\star)^2$ | $a=0.462117$，$\hat y=0.231059$，$\ell=0.295635$；$\bar{\hat y}=-0.768941$；$\bar w$ 两份贡献 $-0.355341$（乘输出）与 $-0.302366$（乘输入），合计 $-0.657707$；$\bar x=-0.151183$；中心差分（步长 $10^{-5}$）$-0.6577071$、$-0.1511831$；解析 $d\hat y/dw=0.855341$ |
| D34 | tanh 与 sigmoid 的关系（§1 新增推导） | $\tanh x=2\sigma(2x)-1$，$\tanh'=4\sigma(2x)(1-\sigma(2x))=1-\tanh^2$ | $\tanh'(0)=1$；$\tanh'(2)=0.070651$（与 D20 一致） |
| G2 | M07 §6 与 M11 的交叉引用 | `math/src/m07-optimization.md` §6“谱范数下的最速下降”；`math/src/m11-doubly-stochastic-sinkhorn.md`“谱范数恰为 1” | §2 新增“最陡依赖于量步长的尺子”一段、§7“这一页用在哪里”据此链接 |

结构改动：原练习 9（显式造 softmax 雅可比的内存，D13、D24）与 §5 正文重复，改为 D33 的手算计算图题，覆盖“读完你能”第 4 条；原题的 1024 个位置的数字（10.3 TB vs 206 MB）并入 §5 正文。§1、§3、§4、§6、§7 各加一个“这一页用在哪里”，链接到用到它的 DeepSeek 课（锚点用 `check_links.py` 核过）。图 M6.9 的残差流记号由 $\mathbf x_l$ 改为本课的 $\mathbf h_l$，正文对 He 等人式中的 $\mathbf x_l$、$\mathcal F$、$\mathcal E$ 注明对应关系。

## 7. 数学审校第二轮（2026-09-24）

全部派生数字与 10 个生成脚本再次用 Python 3 + numpy 复算，脚本输出与现有 SVG 逐字节一致；D25 的逐层估计复算为 $\alpha=0.1$：1.243，$\alpha=1$：214.3（$\E[\tanh'^2]$ 0.467 → 0.052，平均 0.115）。

| 编号 | 量 / 事实 | 表达式 | 值 |
|---|---|---|---|
| D12′ | D12 的中间量（勘误） | $\mathbf p^\top\bar{\mathbf p}$，$\bar{\mathbf p}=(0.3,-1.2,0.5)$ | 三项 0.199572、−0.293674、0.045015，和 $-0.0490866\approx-0.049087$（原文印 0.045016、−0.049085；最终结果 $(0.232227,-0.281661,0.049435)$ 不变） |
| D35 | $\norm{\mathbf p-\mathbf y}$ 的上限 | $(1-p_y)^2+\sum_{k\ne y}p_k^2\le2(1-p_y)^2<2$ | 上确界 $\sqrt2$，取不到（各 $p_k>0$）；正文由“能取到的最大值”改为“上限，只能逼近” |

记号与措辞：§1 表格 $x^n$ 补上“$n$ 为负整数时 $x\ne0$”；§6 注明本页的 $\mathbf y$、$Y$ 是线性层输出，不是 §5 的独热标签；§7 传送带类比的边界把与上一句矛盾的“只加不减”改为“只有累加、没有整体缩小”。

## 8. 补充核实：事实与引用审校（2026-09-25）

逐条回到一手来源重查正文里的外部事实（arXiv 用 `arxiv.py grep`，LaTeXML 全文；其余用 curl 抓原页）。全部派生数字（D1–D35）与 10 个生成脚本再次复算：脚本输出与现有 SVG 逐字节一致，
正文与练习答案里的每个数都与复算一致（复算脚本在审校者 scratch 目录 `facts-m06-calculus-backprop/recheck.py`）。练习 1–12 由审校者独立重解，答案无误。

| 短键 | 核实了什么（原文短语 / 位置） | 结论 |
|---|---|---|
| `baydin2018` | 图 3 标题：截断 logistic 映射、$x_0=0.2$、$E_{\rm forward}$ 与 $E_{\rm center}$；“the first-order errors cancel”；雅可比“requires 2mn evaluations”；“c is a constant guaranteed to be c<6 and typically c∼[2,3]”；存储“in proportion to the number of operations”（最坏情形）与 checkpointing；图 4、表 3 的例子与 $\bar v_5=1$；历史段：Linnainmaa (1970) “often cited as the first published description”，“topologically sorted in terms of dependency by Werbos (1974)”，“brought to fame by Rumelhart et al. (1986)”；arXiv journal-ref “JMLR 18(153):1--43, 2018” | 正文 §1、§3、§4 的转述与卡片全部准确 |
| `he2016identity` | 式 (5) 后：“without concerning any weight layers”，“cannot be always -1 for all samples in a mini-batch”，“does not vanish even when the weights are arbitrarily small”；式 (8) 后：$\lambda_i>1$ “exponentially large”，$<1$ “exponentially small and vanish … forces it to flow through the weight layers”；§1 “if both h(x_l) and f(y_l) are identity mappings”；参考文献标注 ECCV | 准确 |
| `kaplan2020` | §1.1 记号表：“N – … excluding all vocabulary and positional embeddings”，“C≈6NBS”；§2.1：“Accounting for the backwards pass (approximately twice the compute as the forwards pass)”，$C\approx6N$；表 1 的 De-embed 行只有 FLOPs、不计入 $N$ | 卡片原写“§2.1 … 整个训练 $C\approx6NBS$”，6NBS 实在 §1.1 记号表，已改；正文“不计嵌入与反嵌入”改为原文口径“不计词表嵌入与位置嵌入” |
| `xie2025mhc` | 作者顺序 Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li …；§3.1：图 3(b) 的 Amax Gain Magnitude（复合映射最大绝对行和 / 列和，27B 模型）“peaks of 3000”；§5.4：“maximum gain magnitude of nearly 3000 in HC … reduces it by three orders of magnitude”，mHC 复合最大约 1.6；§4.1 “bounded by 1” | 准确；正文补注“增益”是行和 / 列和而不是谱范数，与玩具例子的 $\norm{H^{60}}_2$ 不可直接相比，并注明 27B |
| `radford2019` | PDF §2.3：“accounts for the accumulation on the residual path with model depth”，“1/√N where N is the number of residual layers”，LayerNorm 前移“similar to a pre-activation residual network (He et al., 2016)”，词表 50,257 | 准确 |
| `rumelhart1986` | Crossref：Nature 323(6088):533–536，1986-10，Rumelhart / Hinton / Williams | 准确（正文仍未读） |
| `pytorch-autograd` | 页面标题“PyTorch Tutorials 2.14.0+cu130”，“Last Updated: Oct 01, 2025”；“torch.autograd is an engine for computing vector-Jacobian product”；“accumulates them in the respective tensor’s .grad attribute”；DAG 记录 | 准确 |
| `torch-linear` | 2.14 文档：$y=xA^T+b$；weight 形状 (out_features, in_features) | 准确 |
| `gpt2-config` | config.json：n_embd 768、n_head 12、n_layer 12、n_positions 1024、vocab_size 50257 | 准确 |
| V3（第 12 课的出处 2412.19437） | “We recompute all RMSNorm operations and MLA up-projections during back-propagation” | §4 正文的转述准确 |

跨课链接：`check_links.py` 0 个坏链接。按页题逐个核对了锚点指向的内容。改动三处：

- 封面与 §5“这一页用在哪里”原写“第 7、10、11 课的门控用 softmax”，但 V3（第 11 课）用 sigmoid 亲和度；改为“第 7、10 课用 softmax，第 11 课比较 softmax 与 sigmoid 的雅可比”（第 11 课 §2 式 (11.5)）。
- §6“这一页用在哪里”的 FP8 三次 GEMM 由第 12 课 §2 改指 §4（“一个 Linear 的三次矩阵乘法”，式 (12.13)）。
- §7 原写“两种放法在训练中的实测差别，留给第 16 课”；Xiong 等人的梯度分析实际在第 1 课 §6，第 16 课 §1 讲两难。改为分别链接。另把 §6 例题里“第 2 课讨论了”加上 §6 锚点，把 $H^{60}$ 的 M11 链接指到 §7。
