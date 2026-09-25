# M11 补充核实与复算记录（双随机矩阵与 Sinkhorn 迭代）

> 2026-09-24。本文件只记 `math-refs-doc.md` 里**没有**、而 M11 正文用到的事实与数字。其余一律以 `math-refs-doc.md`
> （C9–C18、§2.2、§3.5、N27–N35、W41–W45、E3–E7）为准。引文不超过 15 个英文词，其余为中文转述并给出位置。
> 标记沿用研究笔记：【笔者推导】= 本课自己的推导或读法；【复算】= 本课脚本算出的数。

## 1. 自己回原文核对过的陈述

| 编号 | 陈述 | 来源与位置 | 原文短语（≤15 词） |
|---|---|---|---|
| S1 | mHC 只做 20 次迭代，单层反向增益略偏离 1；复合情形偏离变大但有界，最大约 1.6 | `mhc`（2512.24880v2）§5.4，图 7 说明段 | “reaching a maximum value of approximately 1.6” |
| S2 | mHC 称双随机约束让特征均值得到保持、信号范数受到约束 | `mhc` §1 | “the feature mean is conserved” |
| S3 | mHC 称 $n=1$ 时双随机条件退化为标量 1，恢复恒等映射 | `mhc` §4.1 | “degenerates to the scalar 1” |
| S4 | DeepSeek-V4 把残差映射记作 $B_l$，把双随机矩阵流形（Birkhoff 多面体）记作 $\mathcal M$；投影先取 $\exp$ 再交替做列、行归一化 | `dsv4`（2606.19348v1）§2.2（式 (8) 前后） | “constrain the residual mapping matrix B_l to the manifold of doubly stochastic matrices” 的前半句（转述其余） |
| S5 | Sinkhorn–Knopp 引理 1：行随机矩阵的列和之积不超过 1，等号仅当每个列和为 1；证明用算术–几何平均不等式 | `sk1967` p. 345 LEMMA 1 | “with equality only if each”（原文后接 $\beta_k=1$） |
| S6 | 收敛证明的核心：取一条正对角线，令 $s_n$ 为缩放因子沿这条对角线的乘积，由引理 1 得 $s_n\le s_n'\le s_{n+1}$ 且有上界，于是列和之积趋于 1，进而每个列和趋于 1 | `sk1967` p. 346 | 转述（原文符号 $s_n=\prod_i x_{i,n}y_{\sigma(i),n}$） |
| S7 | 没有正对角线时迭代不可能有双随机极限，理由是 Birkhoff 定理 | `sk1967` p. 347 | “no limit to the iteration is possible without at least one positive diagonal” |
| S8 | V4.1 的 Sinkhorn 平衡：$m$ 为较大维度（嵌入表与预测头即词表大小），$n$ 为隐藏维度；工作流同 Muon，只是用 Sinkhorn 平衡代替 Newton–Schulz；用 Nesterov 动量、不加权重衰减；动量系数与学习率修正同 Muon；$\gamma=0.18$ 是为了与 Adam 的更新幅度相当；范数 $\le\tau\bar\rho$ 的行先置零 | `dsv41`（2609.19969v1）§2.5 算法 1、式 (7) 及其后一段；§4.2.2 | “to match the update magnitude of Adam” |
| S9 | Sinkhorn 1964 的结论（正矩阵：$D_1AD_2$ 双随机唯一，$D_1,D_2$ 差一个标量因子唯一；非负矩阵迭代不一定收敛）——本课只经 `sk1967` 引言转引 | `sk1967` p. 343 引言 | “the iteration does not always converge”（原文为 for nonnegative matrices） |

## 2. 本课新增算例的复算（Python 3.11 + numpy，脚本在 `math/figures/gen/m11_*.py` 与 scratch 目录）

| 编号 | 量 | 值 |
|---|---|---|
| R1 | 天气链 $P=\begin{pmatrix}0.8&0.15&0.05\\0.2&0.6&0.2\\0.1&0.3&0.6\end{pmatrix}$（教学用例，非来自文献） | 列和 1.1、1.05、0.85；从 $(1,0,0)$ 出发 $\mathbf p_1=(0.8,0.15,0.05)$，$\mathbf p_2=(0.675,0.225,0.1)$，$\mathbf p_3=(0.595,0.26625,0.13875)$；平稳分布 $(4/9,1/3,2/9)$ |
| R2 | 主算例 $A=\begin{pmatrix}0.6&0.4&0\\0.1&0.3&0.6\\0.3&0.3&0.4\end{pmatrix}$ | $A^2=\begin{pmatrix}0.40&0.36&0.24\\0.27&0.31&0.42\\0.33&0.33&0.34\end{pmatrix}$；$A(9,0,3)^\top=(5.4,2.7,3.9)$，长度 9.487→7.187；奇异值 1、0.5646、0.0354；特征值 1、0.2、0.1；$\norm{A^{10}-J_3}_2=5.41\times10^{-7}$ |
| R3 | 主算例的 Birkhoff 分解（贪心，精确分数） | $0.3P_{(1,2,3)}+0.3P_{(2,3,1)}+0.3P_{(1,3,2)}+0.1P_{(2,1,3)}$；余量行列和 0.7、0.4、0.1；正对角线恰为这 4 条 |
| R4 | 单位圆的像（SVD） | $\begin{pmatrix}0.8&0.2\\0.2&0.8\end{pmatrix}$：1、0.6；$\begin{pmatrix}0.9&0.1\\0.6&0.4\end{pmatrix}$：1.1265、0.2663；$H=\begin{pmatrix}1.2&0.3\\0.1&0.9\end{pmatrix}$：1.3048、0.8048 |
| R5 | 行和–列和界 $\sqrt{\norm{A}_1\norm{A}_\infty}$ | $R=\mathbf 1\mathbf e_1^\top$（$n=4$）：2（=实际）；$\begin{pmatrix}0.9&0.1\\0.6&0.4\end{pmatrix}$：1.2247；$H$：1.3964 |
| R6 | $\norm{H^L}_2$ | $L$=1：1.305；10：12.77；20：150；27：840.5；28：1075；30：1759；60：$2.836\times10^6$（与 N30 一致）；$\ln1000/\ln1.2791=28.06$ |
| R7 | $3\times3$ SK 算例 $\begin{pmatrix}2&1&1\\1&3&1\\1&4&6\end{pmatrix}$（精确分数） | 第 1 轮：$\mathcal T_c$ 列除 4、8、8，行和 $\tfrac34,\tfrac34,\tfrac32$；$M^{(1)}$ 列和 $\tfrac76,1,\tfrac56$。第 2 轮：行和 $\tfrac{197}{210},\tfrac{69}{70},\tfrac{113}{105}$；$M^{(2)}$ 列和 1.0317、0.9946、0.9736。误差 0.167、0.0317、0.00575、0.00103、$1.83\times10^{-4}$，$t=10$：$3.2\times10^{-8}$，$t=20$：$8.9\times10^{-16}$；相邻比 0.190→0.1772。极限 $(0.59500,0.18159,0.22341;0.27917,0.51119,0.20964;0.12583,0.30722,0.56695)$，$D_1\propto(2.36431,2.21860,1)$，$D_2\propto(0.12583,0.07680,0.09449)$；交叉比 6、12、4.5 不变；$\norm{M^{(1)}}_2=1.01115$，$\norm{M^{(2)}}_2=1.00035$ |
| R8 | 势函数 $\Phi=M_{11}M_{22}M_{33}$（R7 的算例） | $\tfrac9{64}$→$\tfrac16$→$\tfrac6{35}$→0.172265→0.172413→0.172439→…→0.172445（单调） |
| R9 | $\begin{pmatrix}1&2\\3&4\end{pmatrix}$ 前两轮（N32 只有极限） | $M^{(1)}=\begin{pmatrix}3/7&4/7\\9/17&8/17\end{pmatrix}$，列和 $\tfrac{114}{119},\tfrac{124}{119}$；$M^{(2)}=\begin{pmatrix}31/69&38/69\\93/169&76/169\end{pmatrix}$，列误差 $5/11661=4.29\times10^{-4}$；之后每轮约 ×0.0102 |
| R10 | 练习 4：$\begin{pmatrix}4&1\\1&1\end{pmatrix}$ | $M^{(1)}=\begin{pmatrix}8/13&5/13\\2/7&5/7\end{pmatrix}$，列和 $\tfrac{82}{91},\tfrac{100}{91}$；极限 $a=\tfrac23$；第 3 轮列误差 0.00122 |
| R11 | 第 16 课算例 $\begin{pmatrix}4&1&1\\1&4&1\\2&2&2\end{pmatrix}$ 的第 1 轮 | $M^{(1)}$ 列和 $\tfrac{136}{135},\tfrac{136}{135},\tfrac{133}{135}$（与 lesson-16 一致） |
| R12 | 无正对角线的 $\begin{pmatrix}1&1&1\\1&0&0\\1&0&0\end{pmatrix}$ | 第 1 行 $(\varepsilon_t,\tfrac{1-\varepsilon_t}2,\tfrac{1-\varepsilon_t}2)$，$\varepsilon_{t+1}=\varepsilon_t/(3\varepsilon_t+4)$，$\varepsilon_t=1/(2\cdot4^t-1)$；列和 →$(2,\tfrac12,\tfrac12)$；行步、列步后的两个极限矩阵交替 |
| R13 | $\begin{pmatrix}1&1\\e^{-k}&1\end{pmatrix}$ 20 轮后的列误差（与 lesson-16-doc 一致） | $k$=2：$3.1\times10^{-14}$；5：$3.9\times10^{-4}$；10：0.0232；20：0.0244 |
| R14 | N33 矩阵放大 $s$ 倍后各轮的列误差（N34 只有 $t=20$） | $s=3$：0.942、0.559、0.368、0.210、0.0574（$t$=1,2,3,5,10）；$s=5$：1.13、0.806、0.554、0.313、0.138；$s=10$：1.17、1.05、0.939、0.660、0.224 |
| R15 | 连乘模拟（`m11_products.py`，numpy 种子 0，1000 次，$n=4$，60 层） | $I+0.2G$：谱范数中位 1.42、3.11、5.37、8.76、31.7（$L$=1,10,20,30,60），60 层第 10/90 百分位 6.33/154；逐行 softmax$(3I+Z)$：中位 1.047，全体最大 1.445；同一 logit 的 SK20：中位 1.0000011，单层最大 1.0013；充分迭代（200 轮）的双随机乘积与 $J_4$ 的距离中位：0.086（$L$=10）、0.0051（20）、$4.2\times10^{-8}$（60） |
| R16 | 循环矩阵 $D$ 的抹平 | $\norm{D^L-J_3}_2=0.2646^L$（$D$ 正规）：$L=10$ 时 $1.68\times10^{-6}$；$D^{10}$ 元素在 0.333332–0.333334 |
| R17 | 练习 1 (d) $\begin{pmatrix}1.2&-0.2\\-0.2&1.2\end{pmatrix}$ | 行列和为 1，特征值 1.4、1.0，谱范数 1.4；(b) $\begin{pmatrix}0.7&0.3\\0.4&0.6\end{pmatrix}$ 谱范数 1.0055 |
| R18 | 主算例反复乘 $(9,0,3)^\top$ 时偏离部分长度的相邻比（数学审阅复算，2026-09-25） | 0.2952、0.2329、0.2142、0.2067、0.2032、0.2016 → 0.2（第二特征值）；第一步不是 0.2，所以图 M11.3 说明改为“逐步趋向 0.2” |

## 3. 笔者推导（正文已标注）

- **行和–列和界** $\norm{A}_2^2\le\norm{A}_1\norm{A}_\infty$：用 $A^\top A$ 的最大特征向量取 ∞-范数。由此行随机矩阵 $\norm{A}_2\le\sqrt n$，$R=\mathbf 1\mathbf e_1^\top$ 取等。
- **贪心 Birkhoff 分解不会卡住**：余量除以公共行列和后是双随机矩阵，由定理本身得出一条正对角线（这不是定理的证明）；步数 $\le n^2-n+1$。
- **维数 $(n-1)^2$ 中 $2n-1$ 个约束独立**的说明：左上 $(n-1)\times(n-1)$ 块自由，其余由行、列和确定。
- **$\mathcal B_3$ 的平面投影**：$M\mapsto z(M)+e^{i\pi/3}w(M)$，$z=\tfrac13\sum M_{ij}\omega^{j-i}$，$w=\tfrac13\sum M_{ij}\omega^{i+j}$，把 3 个循环移位送到 $1,\omega,\omega^2$、3 个对换送到另外 3 个六边形顶点，$J_3$ 送到 0（线性，保持凸组合权重，不保距离）。
- **$(A-J_n)(B-J_n)=AB-J_n$**，因而 $\norm{P_L-J_n}_2\le\prod\sigma_2(H_k)$；$\norm{H-J_n}_2=\sigma_2(H)$ 用了 $\mathbf 1/\sqrt n$ 同时是 $H$、$H^\top$ 的不动向量。
- **有限步 SK 的最坏情形有界**：每轮以行归一化收尾，截断输出精确行随机，乘积也行随机，故谱范数 $\le\sqrt n$（与层数无关）。
- **“接近恒等”与“20 轮就准”的张力**：沿用第 16 课的分析，论文未讨论。
- **V4.1 平衡更新 = 对 $\hat G$ 逐元素平方做边际（行 1、列 $m/n$）的 Sinkhorn 缩放**：沿用 `math-refs-doc` C18 的读法（其 §8 第 25 条已列为笔者解读）。

## 4. 仍未核实（正文已回避或明确说明）

- Birkhoff–von Neumann 定理中 von Neumann 的文献（`math-refs-doc` §8 第 4 条）：正文写“von Neumann 那一部分的文献本课未能核实”。
- 每个双随机矩阵都有正对角线的组合学证明（通常借 Hall / König 类定理）：本课没有核到一手来源，正文只说“需要组合数学里的一步，本课不展开”。
- ~~SK 收敛速率的定量理论（`math-refs-doc` §8 第 5 条）：正文只报告算出来的比值（0.177、0.0102），不给速率公式。~~
  已解决（2026-09-25 事实审阅）：见下面 §5 的 V6，正文 §6 改为引 Knight 2008 定理 4.5，并用两个算例核对。
- “完全不可分解”的定义（`math-refs-doc` §8 第 6 条）：正文只在定理框里提到这个词，并注明不展开。
  （补：Knight 2008 §2 给出了定义，见 V6；Marcus–Minc 原书仍未读。）

## 5. 补充核实（事实审阅，2026-09-25）

逐条回原文重查（mHC v2、V4 v1、V4.1 v1 用 `arxiv.py text` 取 LaTeXML 全文并在页头确认版本号；Sinkhorn–Knopp 1967 从 msp.org 下载 PDF 用 pypdf 抽文本；
Knight 2008 从 Strathprints 下载作者预印本）。正文里全部数字另用 numpy / `fractions` 复算，结果与 §2 一致；10 个生成脚本重跑后 SVG 与仓库里的逐字节相同。

| 编号 | 陈述 | 来源与位置 | 原文短语（≤15 词）或说明 |
|---|---|---|---|
| V1 | mHC 式 (6) 就是双随机约束集；其后列出的三条性质依次是 Norm Preservation、Compositional Closure、Geometric Interpretation via the Birkhoff Polytope，都只陈述、不给证明 | `mhc` v2 §4.1 | “The spectral norm of a doubly stochastic matrix is bounded by 1” |
| V2 | mHC 式 (9) 之后：迭代在 $t_{\max}\to\infty$ 时收敛到双随机矩阵，取 20 作为实用值 | `mhc` v2 §4.2 | “We choose t_max = 20 as a practical value” |
| V3 | mHC 图 3、图 7 的对象是 27B 模型；横轴把每个 Transformer 块拆成注意力、FFN 两“层”，所以 60 = 30 块 × 2 个子层（正文开头改写成“60 个子层”） | `mhc` v2 图 3 说明 | “unrolls each standard Transformer block into two independent layers” |
| V4 | V4 论文说 mHC 加强的是“相邻 Transformer 块之间”的残差连接，式 (1) 的 $\mathcal F_l$ 举例为 MoE 层；“每块的注意力与 MoE 子层各有一套”来自发布的代码与 config（【交叉引用】`deepseek/reference-docs/lesson-18-doc.md` §2.1、`lesson-16-doc`），论文正文没有这句话。正文据此把“每个子层”改为“见第 18 课”的代码说明 | `dsv4` v1 §2.2；§4.2.1 两个模型都写 4 与 20 | “strengthen the conventional residual connections between adjacent Transformer blocks” |
| V5 | V4.1 把 mHC 改为 Single-Pass mHC（输入混合系数错后一个块，$A_{l-1}$ 代替 $A_l$），扩展因子 4、SK 20 次（§4.2.1）；Sinkhorn 平衡取代 Adam 的理由是 Engram 参数用 Adam 会大增优化器状态显存，新办法只需一份动量；置零条件是 $\rho_i\le\tau\bar\rho$（不超过，$\bar\rho$ 为 $\hat G$ 各行 $\ell_2$ 范数的平均）；$K=11$ 写在 §4.2.2，算法 1 只要求 $K$ 为奇数 | `dsv41` v1 §2.4.1、§2.5 算法 1 第 4、6 行、§4.2.1、§4.2.2 | “applying Adam to the newly introduced Engram parameters substantially increases the optimizer-state memory footprint” |
| V6 | Knight 2008 定理 4.5：完全不可分解的非负矩阵上 SK 线性收敛，从某轮起缩放向量误差每轮至多乘极限 $P$ 的 $\sigma_2^2$；§6：渐近速率就等于它，且很快贴近。§2 定义完全不可分解：找不到置换 $P,Q$ 使 $PAQ=\begin{pmatrix}A_1&0\\A_2&A_3\end{pmatrix}$（$A_1$ 为方块）。书目：SIAM J. Matrix Anal. Appl. 30(1):261–275，doi:10.1137/060659624（Strathprints 条目页 eprints 元数据）。与 `lesson-16-doc` 第 9 条一致 | Knight 2008 预印本 `skapp.pdf` 定理 4.5、§6、§2 | “the rate of convergence is asymptotically equal to the square of the second singular value” |
| V7 | 【复算】Knight 速率在本课算例上的核对：$3\times3$ 算例极限的奇异值 1、0.42091、0.27313，$\sigma_2^2=0.17717$，列和误差相邻比逐步趋于 0.1772；$\begin{pmatrix}1&2\\3&4\end{pmatrix}$ 极限 $\sigma_2=\abs{2a-1}=0.10102$，$\sigma_2^2=0.010205$，列和误差比 0.010205；N33 矩阵放大 $s=1,3,5,10$ 倍后极限的 $\sigma_2^2$ 为 0.2771、0.7674、0.9259、0.9972（迭代 3000 轮取极限） | scratch `knight_check.py` | — |
| V8 | Sinkhorn–Knopp 1967：从“各个和都趋于 1”到“矩阵收敛”，原文先说极限点都是双随机的，再用引理 2 证明任意两个极限点对角等价，最后用定理的唯一性部分推出只有一个极限（pp. 346–347） | `sk1967` pp. 346–347 | “two such limit points are diagonally equivalent” |
| V9 | 交叉引用核对：第 19 课没有讲 V4.1 的 Sinkhorn 平衡更新（全文无 “Sinkhorn”），它的 Single-Pass mHC 在第 7 页；第 18 课的 mHC 在第 5 页。正文的课间链接据此改正 | `deepseek/src/lesson-19-v4-1-flash.md`、`lesson-18-v4.md` | — |
