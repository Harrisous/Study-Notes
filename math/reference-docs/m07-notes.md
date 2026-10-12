# M07 补充核实笔记（优化器：从 SGD 到 AdamW 再到 Muon）

> 写 `math/src/m07-optimization.md` 时自己核实或自己推导的内容。共享研究笔记 `math-refs-doc.md` 已有的条目
> 不再重复，只记它没有的、以及写课时发现要更正或补充的地方。日期 2026-09-24。
> 标记沿用共享笔记：【笔者推导】= 本笔记自己的推导；【交叉引用】= 数字来自仓库里另一份研究笔记，本次未重核。
> 所有派生数字的脚本：scratch 目录 `m07-optimization/calc.py`，以及 `math/figures/gen/m07_*.py`（图里的数字都由它们算出）。

## 1. 新核实的引文

### V1 Jordan 博客：为什么要正交化（经验动机，原文是“推测”）

来源：Keller Jordan, “Muon: An optimizer for hidden layers in neural networks”，kellerjordan.github.io/posts/muon/，
页面日期 December 8, 2024；2026-09-24 用 WebFetch 重新抓取核对（共享 scratch 里另有一份同日保存的正文 `muon_blog.txt`，文字一致）。

- 小节 “Why is it good to orthogonalize the update?” 的经验动机（原句，15 词以内的片段）：
  “updates … typically have very high condition number”（对象：transformer 里二维参数的 SGD 动量更新与 Adam 更新，依据是“手工检查”）；
  作者随后写 “We speculate that orthogonalization effectively increases the scale of other ‘rare directions’” ——
  **博客自己用的是 speculate（推测）**，课文据此标为“推测”。
- 同节第一句先承认 “It just is OK?” 也是一种回答，并把理论动机指向 Bernstein & Newhouse 对 Shampoo 的分析（与共享笔记 C53、K8 一致）。
- 系数设计一节：“ε can be as high as around 0.3 without harming the loss curve”（经验观察），目标区间写作 [0.7, 1.3]（与 C44 一致）。
- 博客还说明不用 SVD 的原因是太慢（“far too slow”），用 NS 是因为能在 bfloat16 下稳定运行。
- 系数一节：基线系数 (2, −1.5, 0.5) “work perfectly fine for orthogonalizing the update”，调系数只是为了少走几步。
- “Orthogonal-SGDM”一节：Tuddenham et al. (2022) 先用 SVD 正交化梯度、再对结果做动量；博客说 Muon 把动量移到正交化之前，
  “which we find performs better empirically”（课文 §7 标为“博客的经验观察”）。
- 名称：“Muon (MomentUm Orthogonalized by Newton-Schulz)”。

### V2 Sutskever et al. 2013 转述 Polyak 的动量取值——课文按推导改写

来源：`sutskever2013` PMLR PDF §2（共享 scratch `sutskever13.txt` 第 187–198 行，pypdf 抽取）：
Polyak (1964) 证明 CM 所需迭代次数比最速下降少 √R 倍，“where R is the condition number … and µ is set to (√R−1)/(√R+1)”。

【笔者推导，课文 §3 全文给出】对二次函数 $f=\tfrac12\theta^\top H\theta$，重球 $\theta_{t+1}=\theta_t-\eta\nabla f+\mu(\theta_t-\theta_{t-1})$
在每个特征方向上的特征多项式为 $z^2-(1+\mu-\eta h)z+\mu=0$；两根为共轭复数时 $|z|=\sqrt\mu$。
要求区间 $[h_{\min},h_{\max}]$ 上都落在复根区，并让 $\sqrt\mu$ 最小，得
$$\sqrt{\mu^*}=\frac{\sqrt\kappa-1}{\sqrt\kappa+1},\qquad \mu^*=\Big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\Big)^2,\qquad \eta^*=\frac{4}{(\sqrt{h_{\max}}+\sqrt{h_{\min}})^2},$$
收敛因子为 $(\sqrt\kappa-1)/(\sqrt\kappa+1)$。也就是说，Sutskever 等写的 $(\sqrt R-1)/(\sqrt R+1)$ 是**收敛因子**；按这个推导，对应的动量系数是它的**平方**。
数值核对（`calc.py`，以及前一位作者留下的 `m07/check1.py`，按 2001 个曲率点扫描谱半径）：

| κ | 收敛因子 ρ | μ* = ρ² | η*（h_min = 1） | 谱半径实测 | 若 μ 取 ρ 本身，最好的收敛因子 |
|---|---|---|---|---|---|
| 10 | 0.5195 | 0.2699 | 0.2309 | 0.5195 | 0.7208（=√ρ） |
| 100 | 0.8182 | 0.6694 | 0.03306 | 0.8182 | 0.9045 |
| 10⁴ | 0.9802 | 0.9608 | 3.92×10⁻⁴ | 0.9802 | 0.9900 |

课文只在“全部曲率装进复根区”的选法里推导最优；为确认没有别的 $(\mu,\eta)$ 更好，另做了 400×400 网格扫描（`scan.py`，$\mu\in[0,0.99]$，$\eta\in(0,8/\kappa]$，
在 400 个曲率点上取谱半径的最大值）：$\kappa=10$ 时最小 0.5224（在 $\mu=0.273$、$\eta=0.229$），$\kappa=100$ 时 0.8200（$\mu=0.672$、$\eta=0.0325$），与理论值 0.5195、0.8182 只差网格误差。

所以共享笔记 N56 的“迭代次数”（21 / 69 / 691 步对 69 / 691 / 69078 步）是对的——它们用的是收敛因子 ρ；但同行标成“$\mu^*$=0.5195”的那个数应读作收敛因子，
对应动量是 0.2699。Polyak 1964 原文未读（共享笔记 §8 第 8 条），所以课文写成“按推导，动量应取平方；Sutskever 等的写法给出的是收敛因子”，并注明【笔者推导】。

## 2. 笔者推导（课文用，标注在正文里）

- **N-SGD1 噪声地板。** 一维 $f=\tfrac12hx^2$，随机梯度 $hx+\xi$，$\E\xi=0$，$\Var\xi=\sigma^2/B$：
  $\E x_{t+1}=(1-\eta h)\E x_t$，$\Var x_{t+1}=(1-\eta h)^2\Var x_t+\eta^2\sigma^2/B$，稳态 $v_\infty=\dfrac{\eta\sigma^2}{Bh(2-\eta h)}$，
  超额损失 $\tfrac12hv_\infty=\dfrac{\eta\sigma^2}{2B(2-\eta h)}\approx\dfrac{\eta\sigma^2}{4B}$。
  数值：$h=\sigma^2=B=1$，$\eta=0.1$：$v_\infty=0.052632$，地板 0.026316；$\eta=0.01$：0.005025 与 0.002513。
  模拟核对（2 万条链，3000 步，seed 0）：$\eta=0.1$ 时方差 0.052647，$\eta=0.01$ 时 0.005014。
  图 M7.2 用两维 $h=(0.02,1)$、$\theta_0=(10,3)$：地板 0.05134（η=0.1）与 0.005013（η=0.01）；第 2000 步降档后 100 步期望损失 0.0517→0.0302，第 3000 步 0.0203。
- **N-GD1 图 M7.1 的数。** $H=\mathrm{diag}(1,10)$、$\theta_0=(-4,1.5)$、$f_0=19.25$；$\eta=2/11$ 时一步后 $\theta_1=(-3.2727,-1.2273)$，$f_{40}=2.05\times10^{-6}$；
  $\eta=0.21$ 时 $f_{40}=2.30\times10^4$；$\eta=0.05$ 时 $f_{40}=0.132$。
- **N-HB1 图 M7.3 的数。** 同一函数，最优重球 $\mu=0.2699$、$\eta=0.2309$：$\|\theta_{30}\|=2.7\times10^{-7}$；GD（$\eta=2/11$）$\|\theta_{30}\|=0.0104$；
  $\mu=0.9$、$\eta=2/11$：$\|\theta_{60}\|=0.058$（两个方向都进入复根区，收敛因子 $\sqrt{0.9}=0.949$）。注意 $\eta^*=0.2309>2/h_{\max}=0.2$：没有动量时这个步长会发散。
- **N-AD1 偏差修正（补 N50）。** “不修正的步长 ÷ 修正后的步长” $=(1-\beta_1^t)/\sqrt{1-\beta_2^t}$：$\beta_2=0.999$ 时 $t=1$ 为 3.162，最大在 $t=12$（6.569 倍），$t=1000$ 为 1.258；
  $\beta_2=0.95$ 时 $t=1$ 为 0.447（偏小而不是偏大）。
- **N-AD2 尖峰在 Adam 里的停留。**【笔者推导】单坐标、平时 $|g|\approx1$、某步来一个 100 的尖峰，$\beta_1=0.9$、$\beta_2=0.95$：尖峰那一步的归一化步长被 C37 的界限在
  $(1-\beta_1)/\sqrt{1-\beta_2}=0.447$ 倍 η 以内，但之后约 $k$ 步里尖峰方向的步长仍约为 $0.447\,(0.9/\sqrt{0.95})^k$ 倍 η（$k=10$ 时 0.20）。这是课文说“Adam 不能替代裁剪”的数字依据之一（只是示意）。
- **N-AW1 L2 与解耦衰减的模拟（重做 N54）。** 设置同 N54，但每个点 10 万次重复（seed 0，`m07_adamw_decay.py`），标准误约 $2.3\times10^{-7}$：
  平均每步收缩（L2 / AdamW）：$s=10^{-4}$：9.58e-4 / 9.9e-5；$s=0.1$：6.75e-4 / 9.8e-5；$s=1$：9.80e-5 / 9.93e-5；$s=10$：9.86e-6 / 9.91e-5；$s=31.6$：3.27e-6 / 9.89e-5。
  与粗略预测 $\eta\lambda w/\sqrt{s^2+(\lambda w)^2}$ 一致。$s=10$ 处约小 10 倍（N54 的两次 5 千次级模拟分别得 15 倍与 7 倍，重复次数多了以后稳定在约 10 倍）；课文只引数量级。
- **N-NS1 2×2 手算例。** $G=\begin{pmatrix}3&0\\4&5\end{pmatrix}$：$G^\top G=\begin{pmatrix}25&20\\20&25\end{pmatrix}$，特征值 45、5，$\sigma=(3\sqrt5,\sqrt5)=(6.7082,2.2361)$，$\|G\|_F=\sqrt{50}$；
  $\mathbf v_1=(1,1)/\sqrt2$，$\mathbf v_2=(1,-1)/\sqrt2$，$\mathbf u_1=(1,3)/\sqrt{10}$，$\mathbf u_2=(3,-1)/\sqrt{10}$；
  极分解 $G=QS$，$Q=UV^\top=\frac1{\sqrt5}\begin{pmatrix}2&-1\\1&2\end{pmatrix}$（逆时针转 26.565°），$S=V\Sigma V^\top=\sqrt5\begin{pmatrix}2&1\\1&2\end{pmatrix}$。
  NS 第一步（Jordan 系数）：$X_0X_0^\top=\begin{pmatrix}0.18&0.24\\0.24&0.82\end{pmatrix}$，其平方 $\begin{pmatrix}0.09&0.24\\0.24&0.73\end{pmatrix}$，
  $X_1=\begin{pmatrix}0.8018&-0.4656\\0.2931&0.7156\end{pmatrix}$，奇异值 0.9447（沿 $\mathbf u_2$）与 0.7518（沿 $\mathbf u_1$）。
  归一化奇异值的轨迹：$3/\sqrt{10}=0.9487\to0.7518\to1.0484\to0.6819\to1.1343\to0.7530$（5 步）；$1/\sqrt{10}=0.3162\to0.9447\to0.7568\to1.0414\to0.6825\to1.1337$。
  5 步后 $\max|X_5-UV^\top|=0.196$；V4 混合 8+2 步后奇异值 (1.00054, 1.00019)，$\max|X_{10}-UV^\top|=4.0\times10^{-4}$。
- **N-V41 V4.1 调度的派生量。**【笔者算】batch 固定 100.6M token 时，2000 步 warmup = 201.2B token = 45T 的 0.447%；全程约 $45\times10^{12}/100.6\times10^6\approx4.47\times10^5$ 步（假设 batch 全程固定，论文说是固定的）；
  28T–40T cosine 段中点 34T 处 $\eta=1.43\times10^{-4}$。warmup 的形状 V4.1 没有写，图 M7.6 的插图按线性画并注明。
- **N-MU1** 随机 256×1024 高斯阵正交化后 RMS = 0.03125 = $1/\sqrt{1024}$（引理 1 的数值核对，补 N60）。
- **N-MU2 谱范数下的比较（课文 §6 例题）。** $G=\begin{pmatrix}3&0\\4&5\end{pmatrix}$：$\tr(G^\top UV^\top)=\sigma_1+\sigma_2=4\sqrt5=8.944$；
  $\sign(G)=\begin{pmatrix}1&0\\1&1\end{pmatrix}$ 的谱范数 1.618（黄金比），$\langle G,\sign(G)\rangle/1.618=12/1.618=7.416$。
- **N-MU3 练习用数。** $[7168,1536]$：RMS $1/\sqrt{7168}=0.01181$，缩放 $0.18\sqrt{7168}=15.24$；每步 NS $2(2\cdot7168\cdot1536^2+1536^3)=7.489\times10^{10}$，
  10 步 $7.489\times10^{11}$，$6nmB$（$B=7.55\times10^7$）$=4.988\times10^{15}$，占比 0.0150%，上界 $10\times1536/7.55\times10^7=0.0203\%$。
  秩 1 的 $G=\begin{pmatrix}4&0\\3&0\end{pmatrix}$：$\norm G_F=\sigma_1=5$，Jordan 迭代 $1\to0.701\to1.1136\to0.7207\to1.0900\to0.6964$。
  $\varphi_B$ 两步：$1.1343\to1.01833\to1.00019$；$0.6819\to0.96191\to1.00054$；$\varphi_B''(1)=1$。
  $H=\mathrm{diag}(2,50)$：GD $\eta<0.04$，$\eta^*=0.03846$，$\rho=0.9231$，173 步；重球 $\mu^*=0.4444$、$\eta^*=0.05556$、$\rho=2/3$、34 步。
  Adam（0.9/0.95）梯度 0.02、0.01：$\hat m_2=0.014737$，$\hat v_2=2.4615\times10^{-4}$，比值 0.9393；不修正 0.4472、0.5715。
- **N-AD3 EMA 与缓冲写法在 Adam 里不等价。**【笔者推导，课文 §3 注】Adam 的分母是梯度平方的滑动平均，与一阶矩的尺度无关，所以把一阶矩从 EMA 换成缓冲写法会让步长放大 $1/(1-\beta_1)$ 倍；
  共享笔记 E21 说“两种写法给出同一个 $\mathbf O_t$”只对 Muon（有 Frobenius 归一化）成立，课文照此区分。

### 2.1 教学审读补充（2026-09-24，第二轮）

脚本：scratch 目录 `teach-m07-optimization/verify.py`，以及新图的 `math/figures/gen/m07_rate_curves.py`。

- **N-AD2 更正（尖峰的余波）。**【笔者推导 + 数值核对】单坐标、平时梯度恒为 1（$m=v=1$ 的稳态）、某步来 100，$\beta_1=0.9$、$\beta_2=0.95$：
  那一步 $m=10.9$、$v=500.95$，归一化步长 0.487（尖峰部分 $(1-\beta_1)/\sqrt{1-\beta_2}=0.447$），**没有**超过 Adam §2.1 的上界 $\eta$（$\beta_2=0.95$ 属于“否则 $\le\alpha$”的情形；旧课文写“受上界约束、不超过 $0.447\eta$”不准，已改）。
  尖峰部分第 $k$ 步 $0.447(0.9/\sqrt{0.95})^k$：$k=10$ 时 0.2015。正常梯度部分的步长（相对平时 1）：$k=10$ 0.056，$k=20$ 0.074，$k=50$ 0.159，$k=100$ 0.5025（首次 $\ge0.5$），$k=150$ 0.90。
  旧课文“把正常梯度的步长压小了十几步”低估了，已改为“第 20 步只有平时的 7%，第 100 步才恢复到一半”。
- **N-RM1 Robbins–Monro 第一条的示意推导与 $1/t^2$ 例。**【笔者推导】$\E x_T=\E x_0\prod_t(1-\eta_th)$；$\ln(1-u)\le-u$ 给出和发散则乘积趋 0；$0\le u\le\tfrac12$ 时 $\ln(1-u)\ge-2u$ 给出和有限则乘积有正下界。
  $\eta_t=1/t^2$（$t\ge2$）、$h=1$：$\prod_{t=2}^T(1-1/t^2)=(T+1)/(2T)\to\tfrac12$（数值：$T=10^6$ 时 0.5000005）；$\sum_{t\ge2}1/t^2=\pi^2/6-1=0.6449$。
- **N-HB2 实根时较大的模 $>\sqrt\mu$。**【笔者推导】两根实且不等、乘积 $\mu$ 时 $\max|r|>\sqrt\mu$（模相等的不等实根只能是 $\pm r$，和为 0，此时方程 $r^2+\mu=0$ 无实根，矛盾）。
  图 M7.3 的数：最优重球（$\mu^*=0.2699$，$\eta^*=0.2309$）复根区恰为 $h\in[1,10]$，平台 0.5195；$h=0.5$ 时 0.8288，$h=11$ 时 1.000，$h=12$ 时 1.292；$\mu=0.9$、$\eta=2/11$ 的复根区 $h\in[0.0145,20.89]$，平台 0.9487。
- **N-NS2 8+2 步的奇异值顺序。** 沿 $\mathbf u_1$：第 8 步 1.13426 → 1.01832 → 1.00019；沿 $\mathbf u_2$：0.68186 → 0.96189 → 1.00054。旧课文把第 10 步写成 $(1.0005,1.0002)$（顺序颠倒），已改为 $(1.0002,1.0005)$。
  $\varphi_B$ 的泰勒核对：$\varphi_B'''(x)=-9+30x^2$，$\varphi_B'''(1)=21$；$e=0.1343$：$1+e^2/2=1.0090$，加 $3.5e^3$ 得 1.0175（实际 1.01833）；$e=-0.3181$：1.0506，加三次项 0.9379（实际 0.96191）；
  第二步 $e=0.0183$：1.00017 / 1.00019（实际 1.00019）；$e=-0.0381$：1.00073 / 1.00053（实际 1.00054）。$\varphi_B(1.01)=1.0000535$，$\varphi_B(0.99)=1.0000465$。
- **N-MU4 NS 开销上界的展开。**【笔者推导】$T\cdot2(2nm^2+m^3)/(6nmB)=(Tm/B)\cdot(4n+2m)/(6n)$；$[7168,18432]$ 时因子 0.7963，$0.095\%\times0.796=0.0756\%$，与课文 0.076% 一致。
- **N-EX5 练习 5（噪声地板与降档）。** $h=0.5$、$\sigma^2=4$、$B=16$：$\eta=0.2$ 时 $v_\infty=0.052632$、地板 0.013158（近似式 0.0125）；$\eta=0.02$ 时 0.0050251 / 0.0012563；
  $\eta=0.2$、$B=160$ 时 0.0052632 / 0.0013158。方差差距每步乘 $0.99^2=0.9801$，缩到 5% 要 149.04 步（取约 150）；均值缩到 $10^{-3}$：因子 0.9 要 66 步，0.99 要 688 步。

### 2.2 教学审读补充（2026-09-24，第三轮）

脚本：`math/figures/gen/m07_adam_spike.py`（新图 M7.9，原 M7.9–M7.11 顺延为 M7.10–M7.12），scratch 目录 `teach-m07-optimization/r3/spike_calc.py`。

- **N-AD4 尖峰余波的完整账（图 M7.9）。**【笔者推导 + 数值核对】单坐标、平时梯度恒为 1、$m=v=1$ 稳态、$\beta_1=0.9$、$\beta_2=0.95$、$\epsilon=0$：
  同号尖峰 $+100$：$m_k=1+9.9\times0.9^k$，$v_k=1+499.95\times0.95^k$；每步位移（单位 η）$k=0$ 0.4870，$k=10$ 0.2569（其中尖峰份额 0.1992），$k=20$ 0.1641（正常份额 0.0745），$k=50$ 0.1673，$k=100$ 0.5027；
  正常份额在 $k\le50$ 内为 0.0447–0.1592；$\sum_{k=0}^{150}=62.42$（无尖峰 151）。
  反号尖峰 $-100$：$m_k=1-10.1\times0.9^k$，$k=0$ 为 $-0.4066$，$k=0..21$ 共 22 步为负（$k=22$ 时 $+0.0004$），$\sum_{k=0}^{50}=-1.16$，$\sum_{k=0}^{150}=50.77$。
  先裁剪到 $c=1$（尖峰以 $\pm1$ 进入 Adam）：同号不变；反号 $k=0$ 为 0.80、$k=20$ 为 0.9757，$\sum_{k=0}^{150}=149.00$。
- **N-RM2** $\sum_{t=1}^{10^6}1/\sqrt t\approx2\sqrt{10^6}+\zeta(1/2)=2000-1.4604=1998.54$（旧课文写“和是 1999”，改为“约 1998.5”）。
- **N-NS3** 图 M7.10（原 M7.9）第四格的奇异值标签按 $G$ 的奇异方向排序计算：沿 $\mathbf u_1$ 1.0002、沿 $\mathbf u_2$ 1.0005（旧图写“1.0005、1.0002”，与正文顺序相反，已改为脚本算出）。

### 2.3 数学审读补充（2026-09-24，第四轮）

脚本：scratch 目录 `math-m07-optimization/verify1.py`、`verify2.py`、`verify3.py`、`bound_t.py`（重算课文、练习与图里的全部数字；12 个生成脚本重跑后 SVG 与原文件逐字节相同）。

- **N-AD5 Adam §2.1 的“上界”不严格。**【笔者推导 + 数值核对】$\epsilon=0$、稳态时用 Cauchy–Schwarz：
  $\abs{m_t}/\sqrt{v_t}\le(1-\beta_1)/\sqrt{(1-\beta_2)(1-\beta_1^2/\beta_2)}$（需 $\beta_1^2<\beta_2$），等号在 $g_{t-k}\propto(\beta_1/\beta_2)^k$。
  默认 (0.9, 0.999)：7.2703（论文的“界”3.162）；(0.9, 0.95)：1.1650（论文“$\le\alpha$”）。直接构造该梯度序列跑 Adam 得到同样的数。
  有限步带偏差修正：$\beta_2=0.95$ 时第 2 步起上确界 $>1$（$t=5$：1.0029），默认值第 210 步起 $>3.162$（$t=100$：2.243，$t=1000$：5.781）。
  论文原文（v9 §2.1）结论写的是 $|\Delta_t|\lessapprox\alpha$，课文据此把两个数说成“典型情形”，并加了 warn 与推导。
- **N-AW2 AdamW 命题 3 的系数有排印错误。** `adamw`（v3，LaTeXML 抽取）命题 3 正文写 $f^{sreg}=f+\frac{\lambda'}{2\alpha}\|\theta\odot\sqrt s\|^2$、$\lambda'=\lambda/\alpha$；
  附录的证明写 $f+\frac{\lambda'}{2}\|\theta\odot\sqrt s\|^2$，并由 $\alpha\lambda'=\lambda$ 得到等价。按证明验算，正确的是 $\lambda'/2$（正文的写法多除了一个 $\alpha$）。
  共享笔记 C39 照抄了正文的 $\lambda'/(2\alpha)$；课文旧稿写成 $\frac{\lambda'}{2\eta}$，已改为 $\frac{\lambda'}{2}$（$\lambda'=\lambda/\eta$）并注明论文正文与证明不一致。
- **N-ST1 步数统一向上取整。** $\ln10^{-6}/\ln\rho$：GD 68.85 / 690.75 / 69077.55 → 69 / 691 / 69078；重球 $\kappa=10$ 为 21.10 → **22**（旧稿 21，按四舍五入），比值 69/22 = 3.1（旧稿 3.3）；
  练习 4 重球 34.07 → **35**（旧稿 34），GD 172.6 → 173；Sutskever 取法 $\mu=0.5195$ 的最佳因子 0.7208 → 42.19 → **43**（旧稿 42）。练习 5 本来就向上取整（687.3 → 688）。
  另：只取 $\rho=0.9998$ 会算成 69,071 步（69070.6 向上取整），$0.980$ 会算成 684 步（683.8），课文改为给出分数 $9999/10001$、$99/101$。
- **N-EX8 练习 8(c) 的量化。** $S/S_{\min}=1+\mathcal B_{noise}/B$、$E/E_{\min}=1+B/\mathcal B_{noise}$：$B/\mathcal B_{noise}$ 从 0.25 到 0.5，步数 5 → 3（少 40%，旧稿“几乎减半”），样本 1.25 → 1.5（多 20%）；
  从 4 到 8，步数 1.25 → 1.125（少 10%），样本 5 → 9（多 80%）。
- **N-NS4 Jordan 迭代的输入下限。** 8 步 Jordan 后落进 $[0.68,1.1345]$ 的输入是 $x\gtrsim1.27\times10^{-4}$；8+2 步后 $|y-1|\le10^{-3}$ 的输入是 $x\gtrsim1.15\times10^{-4}$；
  $N=5$ 时 $x\ge0.05$ 已全部在 $[0.6818,1.1344]$ 内（线性网格上 99.5% 的点）。图 M7.11 说明与 §6 正文据此加了“输入不太小”的限定。
  $\varphi_J$ 的不动点 0.868（$\varphi_J'=-1.58$，排斥），4 周期乘子 $-0.143$（吸引）。
- **N-HB3** 400×400 网格扫描（本轮自己重做，网格 $\mu\in[0,0.99]$、$\eta\in[10^{-4},8/\kappa]$、400 个曲率点）：$\kappa=10$ 最小 0.5200，$\kappa=100$ 最小 0.8200，与课文“0.522 与 0.820，只差网格粗细”一致。
  SGD 地板的模拟重跑（2 万链、3000 步，抽样顺序与原稿不同）：0.0519 与 0.00504，与理论 0.05263、0.005025 相差在 1.4% 以内（方差估计的标准误约 1%）。
- **N-SV1 其他订正（无数值变化）。** (M7.23) 的 $G(G^\top G)^{-1/2}$ 需要 $n\ge m$ 且列满秩，$n<m$ 时用 $(GG^\top)^{-1/2}G$；(M7.28) 的“满秩”指被正交化的矩阵而非 $W$；
  §6 最速下降推导里的步长半径改记 $\delta$（原记 $r$，与同页 SVD 的秩 $r$ 冲突，练习 2 同改）；Robbins–Monro 第一条的下界补上常数 $c>0$；§1 补“$H$ 正定”的假设；
  McCandlish 推导最后一行的符号（期望变化为负，降幅为正）分成两行写清。

## 3. 交叉引用（本次未重核）

- DeepSeek-V4 §2.4 说它对大多数模块用 Muon 的理由是收敛更快、训练更稳（lesson-18-doc C34）；同形状参数合并成批做 NS、NS 在 BF16 矩阵乘下稳定（lesson-18-doc C40，V4 §3.4.1）。
- DeepSeek LLM：AdamW（0.9, 0.95, 0.1）、梯度裁剪 1.0、多步调度 80%/90% ×0.316/×0.1、超参数幂律 $\eta_{opt}=0.3118C^{-0.125}$、$B_{opt}=0.2920C^{0.3271}$（lesson-06-doc 第 12 条、表 239–240 行、式 (1)）。
  以上两条已在第 4 节直接对原文重核。

## 4. 补充核实（2026-09-25，事实审读）

方法：`arxiv.py text/grep`（2606.19348v1、2609.19969v1、2401.02954v1、1711.05101v3、1706.02677v2、2404.06395v3、2502.16982v1），WebFetch 重抓 Jordan 博客与 Crossref，
Sutskever 2013 与 Adam v9 用共享 scratch 里的 PDF 抽取文本。数值脚本：scratch 目录 `facts-m07-optimization/verify.py`（重算课文与 11 道练习的全部数字，均一致）。

- **F1 V4.1 的 warmup 是线性的（更正 N-V41）。** `dsv41` §4.2.2 原句 “linearly warmed up over the first 2000 steps”。N-V41 说“warmup 的形状 V4.1 没有写”不对；
  图 M7.7 说明原写“形状论文没写，这里按线性画”，已改为“按论文写的线性升高画”（图本身不变，生成脚本的说明同改）。V4 §4.2.2 同样写 “linearly warmed up in the first 2000 steps”；DeepSeek LLM 只写 “after 2000 warmup steps”，没说形状。
- **F2 [7168, 18432] 不是 V4 的矩阵。** 它是 DeepSeek-V3 前 3 层稠密 FFN 的形状（隐藏维 7168、`intermediate_size` 18432；【交叉引用】lesson-05-doc 第 454、899 行，lesson-11-doc 第 384 行）。
  `dsv4` §4.2.1：V4-Flash 43 层、$d=4096$、每个专家中间维 2048；V4-Pro 61 层、$d=7168$、专家中间维 3072；两者 “We employ MoE layers in all Transformer blocks”，没有稠密 FFN。
  课文旧稿把这个形状放在“DeepSeek-V4 规模下的开销”例题与图 M7.12（标题是 V4 的 Muon）里，读者会以为是 V4 的矩阵。已改：两道例题与图 M7.12 都注明“借用 V3 稠密 FFN 的形状”，例题标题改为“V4 的 batch 下，一个大矩阵的开销”，§7 要点改为“$m=7168$、V4-Flash 的 7550 万 token batch 时不到 0.1%”。数字本身不变。
- **F3 1.6 万亿参数 = DeepSeek-V4-Pro 的总参数量**（`dsv4` §1 与 §4.2.1：“DeepSeek-V4-Pro comprises 1.6T total parameters, of which 49B are activated”；V4-Flash 284B / 13B）。课文 §4 的 12.8 TB 算例补上这个出处，并注明 V4 实际只把少数参数交给 AdamW。
- **F4 V4 的 RMS 缩放理由。** `dsv4` §2.4：rescale the update RMS “for reutilization of our AdamW hyper-parameters”；§4.2.2：“for reutilization of the AdamW learning rate”。课文 §4 注“以便共用同一套学习率与权重衰减”有据。
  §2.4 另有：前 8 步 Jordan 系数 “bringing the singular values close to 1”，后 2 步 “stabilize the singular values precisely at 1”；因 Q 与 KV 条目上直接有 RMSNorm，不用 QK-Clip；§3.4.1：Muon 需要完整梯度矩阵，与 ZeRO 冲突，同形状参数合并批处理，NS 在 BF16 矩阵乘下稳定。
- **F5 DeepSeek LLM（2401.02954v1）直接核对。** §2.3 “The gradient clipping during the training phase is set to 1.0.”（只给数值，无理由）；AdamW $\beta_1=0.9$、$\beta_2=0.95$、weight_decay 0.1；
  多步调度 80% 处降到 31.6%、90% 处降到 10%；§3.1 幂律 $0.3118\,C^{-0.1250}$ 与 $0.2920\,C^{0.3271}$（$10^{0.3271}=2.124$，$10^{-0.125}=0.750$）。
  §3.1 还说 McCandlish 等早先的经验结论 “have limited applicability in our preliminary experiments”，课文 §5 的“DeepSeek LLM 没有用这个模型”据此补了一句理由。
- **F6 AdamW 原话。** v3 §2：Adam + L2 时 “weights that tend to have large gradients in f do not get regularized as much as they would with decoupled weight decay”。命题 3 正文 $\frac{\lambda'}{2\alpha}$、附录证明 $\frac{\lambda'}{2}$ 的不一致再次确认（N-AW2）。
- **F7 Lai & Yuan 已正式发表。** Crossref：Statistical Science 36(2)，2021（published-print 2021-04-01），DOI 10.1214/20-STS784，作者 Tze Leung Lai（Stanford）、Hongsong Yuan（上海财经大学）。
  共享笔记 §8 第 7 条的“是否及何时正式发表未核”由此解决；课文参考卡写上期刊出处，并保留“本课读的是预印本”。
- **F8 Sutskever 2013 §2 的完整论点。** 同一段先说“理论预测”随机情形下渐近局部收敛率的优势会丧失（引 Orr 1996、Wiegerinck et al. 1999），接着论证训练深度网络时更要紧的是局部收敛之前的“过渡阶段”（transient phase），
  那时下降方向在相邻梯度估计之间持续、不被噪声淹没。课文旧稿只写了前半句，已补后半句，避免把文献说成“动量在随机梯度下没用”。
- **F9 Jordan 博客（2026-09-25 WebFetch 重抓）。** $\mathrm{Ortho}(G)=\arg\min_O\{\|O-G\|_F: O^\top O=I\ \text{或}\ OO^\top=I\}$ 是正文里的文字公式（不是图片）；SVD “far too slow”；
  “manual inspection … very high condition number … almost low-rank … dominated by just a few directions”，随后 “We speculate …”；$\varepsilon$ 约 0.3 不伤损失曲线；约束 $[0.7,1.3]$；“an ad-hoc gradient based approach”；5 步足够；
  嵌入与最终分类头用 AdamW “to attain the best performance”；Nesterov “works a bit better … in every case we have tested”；每步 NS $2(2nm^2+m^3)$、基线 $6nmB$；$5\cdot768/524288=0.7\%$，$5\cdot16384/16000000=0.5\%$；
  “Muon with momentum turned off” 可以看成 “accumulation-free” Shampoo；页面日期 December 8, 2024；$(2,-1.5,0.5)$ “already works”。课文 §6–§7 的对应陈述全部一致。
- **F10 Goyal §2.1–§2.2。** 线性缩放失效的两种情形（训练初期 “network is changing rapidly”；batch 不能无限增大）；渐进 warmup 从 $\eta$ 起每次迭代加常数，5 个 epoch 后到 $k\eta$，“results are robust to the exact duration of warmup”。
- **F11 V4.1 §2.5 与 §4.2.2。** 按头 Muon 用于 Query 与 Key 权重（§2.5 第一段只说 Query，“Basic Configurations” 写 Query and Key）；Muon 还用于 Engram 投影层与视觉–语言投影；
  AdamW 用于归一化层权重与其他非矩阵参数（含偏置与缩放因子）；Engram 表、词嵌入、预测头用“动量 + Sinkhorn 平衡”（算法 1：奇数步归一化行、偶数步归一化列，$K=11$，乘 $\sqrt n$，学习率乘 $\gamma=0.18$，不加权重衰减）。
- **F12 跨课链接。** 第 19 课只在 §7 讲 V4.1 的训练日程（学习率、batch、45T），**没有**展开按头 Muon 与 Sinkhorn 平衡更新；V4.1 的 Sinkhorn 平衡更新在 M11 §7 “DeepSeek 里的另一处 Sinkhorn”。
  课文 §7 的“这一页用在哪里”旧稿说第 19 课讲这两样，已改。其余跨课链接补了 `#pN`（第 2 课 §5 优化、第 6 课 §4 多步调度与 AdamW、§6 幂律、第 18 课 §5 Muon、§6 训练日程、第 11 课 §1 V3 规格；M02 §3、M03 §1/§3、M04 §3、M06 §2、M08 §3、M10 §1、M11 §7），`check_links.py` 0 个坏链。
- **F13 第 2 课的调度不是“长平台”。** 第 2 课记录的 GPT-1 是余弦退火到 0，GPT-3 是 260B token 内余弦降到 10% 再保持；课文 §2 的 warn 旧稿说“第 2、6、18 课的调度都有很长一段恒定或几乎恒定的阶段”，已改为只对第 6、18、19 课这样说，第 2 课单独写 GPT-3 停在 10%。
- **F14 数值复核（`verify.py`）。** $\varphi_J$ 从 0.01 出发：0.0344、0.1184、0.4001、1.0931（第 4 步到 1 附近），之后在 0.699–1.116 之间跳；$\varphi_B$ 从 0.01 出发第 8 步 1.0017；三次式 12 步 0.8934；
  2×2 手算的 $A$、$A^2$、$bA+cA^2$、$aX_0$、$(bA+cA^2)X_0$、$X_1$、$X_{10}$ 与课文逐位一致（$\max|X_5-UV^\top|=0.1956$，$\max|X_{10}-UV^\top|=4.03\times10^{-4}$）；
  练习 1–11 的全部数字一致（例：练习 4 重球 34.07 → 35 步；练习 5 方差差距 149.04 步；练习 7 稳定期 278,330 步、$7.20\times10^{-4}$；练习 9 占比 0.0150%、上界 0.0203%；练习 10 距离平方 17；练习 11 四次项 $2.5e^4=0.0256$）。
- **F15 MiniCPM（2404.06395v3）。** 最大学习率 0.01、batch 3.93M（§6.2）；“10% Steps are Enough”（§4.3）；指数退火式照原文印作 $0.5^{(s-S)/T}$、$T$ 为 5000 步（20B tokens），E19 的半衰期读法仍成立。
- **F16 Moonlight（2502.16982v1）。** AdamW 更新 RMS “usually around 0.2 to 0.4”；$\max(A,B)$ 太大（dense MLP）更新太小、太小（把 GQA/MLA 的每个 KV 头当独立参数）更新太大；不加权重衰减时权重与层输出的 RMS 超出 bf16 的高精度范围；
  摘要 “∼2× computational efficiency”，正文“约 52% 训练 FLOPs 即可追平 AdamW”。
