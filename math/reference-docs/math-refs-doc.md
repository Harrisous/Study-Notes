# 第 M01–M11 课研究笔记：数学补课的命名结果、数值格式常数、优化器算法与易错点

> 研究日期 2026-09-24。本文件是数学补课线（`math/src/m01…m11-*.md`）的唯一数字与引文来源：正文里的定理陈述、
> 算法常数、数值格式参数、优化器超参数都必须能在这里找到出处。DeepSeek 专属的规格（KV 字节、模型维度、训练成本等）
> 以 `deepseek/reference-docs/lesson-NN-doc.md` 为准，本文只在数学课需要举例时交叉引用，并注明“交叉引用，未在此重核”。
>
> **核对方法**：arXiv LaTeXML 全文（`arxiv.py text/grep`）逐句检索；无 HTML 渲染的论文下载 PDF 用 pypdf 抽文本；
> 扫描版 PDF（Stewart 1993）抽出页面图像逐页阅读；出版社元数据一律用 Crossref API 核对；GitHub 代码用 raw 地址下载后 grep；
> 所有派生数字用 Python 复算（脚本在 scratch 目录，表达式抄在 §4）。
>
> **引文政策**：为遵守版权约束，本文只保留少量不超过 15 个英文词的原文短语（凡是“措辞本身就是要核对的事实”的地方），
> 其余一律中文转述并给出精确位置（节 / 式 / 表 / 页 / 行）。公式与数值表是事实，按原样转录。写课者需要原句时按位置回原文查。
>
> **标记**：【笔者推导】= 本笔记自己的推导或解读，不是任何来源的原话；【交叉引用】= 数字来自仓库里另一份研究笔记，本次未重核；
> 【未读原件】= 原始文献本次无法取得，陈述来自列明的二手来源。
>
> **共同编辑**：Harry 在本文件里留审阅意见时写 `<!-- 批注(Harry): … -->`，回复写 `<!-- 回复(Claude): … -->`
> （约定见仓库根目录 `CLAUDE.md`）；`grep -rn "批注(" math/reference-docs` 列出未处理的批注。

---

## 0. 覆盖范围与所读版本

### 0.1 论文、书与标准

| 短键 | 标题 | 作者 | 编号 / 所读版本 | 出处 / 状态 | 日期 | 读取方式 |
|---|---|---|---|---|---|---|
| `stewart1993` | On the Early History of the Singular Value Decomposition | G. W. Stewart | DOI 10.1137/1035134 | SIAM Review 35(4):551–566（Crossref 核对） | 1993-12 | 扫描 PDF（UC Davis 课程页镜像），逐页读图：pp. 551, 559–563, 565–566；核查时另用带文本层的 SIAM 版（jontalle.web.engr.illinois.edu 镜像）grep 了 pp. 560–563 的原句 |
| `eckart1936` | The Approximation of One Matrix by Another of Lower Rank | Carl Eckart, Gale Young | DOI 10.1007/BF02288367 | Psychometrika 1(3):211–218（Crossref） | 1936-09 | **只读到摘要**（Springer 页面）；全文付费 |
| `mirsky1960` | Symmetric Gauge Functions and Unitarily Invariant Norms | L. Mirsky | DOI 10.1093/qmath/11.1.50 | Quarterly J. Math. 11(1):50–59（Crossref） | 1960 | 【未读原件】OUP 403；内容来自 `stewart1993` §7 |
| `schmidt1907` | Zur Theorie der linearen und nichtlinearen Integralgleichungen（I. Teil） | Erhard Schmidt | DOI 10.1007/BF01449770 | Math. Ann. 63(4):433–476（Crossref） | 1907-12 | 【未读原件】内容来自 `stewart1993` §5 |
| `mml2020` | Mathematics for Machine Learning | M. P. Deisenroth, A. A. Faisal, C. S. Ong | 免费 PDF（页脚 “Draft (2024-01-15)”） | Cambridge University Press 2020 | 2020 | PDF 文本：定理 4.22、4.24、4.25（§4.6 “Matrix Approximation”，pp. 131–132），式 (4.64)(4.93)–(4.99)，(7.30) 附近 |
| `birkhoff1946` | Tres observaciones sobre el algebra lineal | G. Birkhoff | 无 DOI | Univ. Nac. Tucumán Rev. Ser. A 5 | 1946 | 【未读原件】陈述来自 `sk1967` p. 345 与 `mhc` §4.1；页码见 §8 |
| `sinkhorn1964` | A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices | Richard Sinkhorn | DOI 10.1214/aoms/1177703591 | Ann. Math. Statist. 35(2):876–879（Crossref） | 1964-06 | 【未读原件】Project Euclid 有机器人拦截（未绕过）；陈述来自 `sk1967` p. 343 引言 |
| `sk1967` | Concerning Nonnegative Matrices and Doubly Stochastic Matrices | Richard Sinkhorn, Paul Knopp | DOI 10.2140/pjm.1967.21.343 | Pacific J. Math. 21(2):343–348（Crossref；MSP 开放获取） | 1967（收稿 1966-02-20） | 全文 PDF（msp.org） |
| `mhc` | mHC: Manifold-Constrained Hyper-Connections | Zhenda Xie 等 20 人（DeepSeek） | 2512.24880 **v2** | 预印本 | 2025-12-31 | LaTeXML 全文 |
| `dsv4` | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | DeepSeek-AI | 2606.19348 **v1** | 技术报告 | API 显示 2026-04-26（与编号前缀不符，见 §8） | LaTeXML 全文，只读 §2.2、§2.4、§4.2.2 |
| `dsv41` | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | DeepSeek-AI | 2609.19969 **v1** | 技术报告 | 2026-09-17 | LaTeXML 全文，只读 §2.4.4、§2.5、§4.2.2 |
| `jensen1906` | Sur les fonctions convexes et les inégalités entre les valeurs moyennes | J. L. W. V. Jensen | DOI 10.1007/BF02418571 | Acta Math. 30:175–193（Crossref） | 1906 | 只核对元数据；定理陈述用 `mackay2003` |
| `mackay2003` | Information Theory, Inference, and Learning Algorithms | David J.C. MacKay | 作者网站免费 PDF | Cambridge University Press 2003 | 2003 | PDF 文本：§2.6 式 (2.45)(2.46)，§2.7 式 (2.47)(2.48)，pp. 34–35 |
| `williams1992` | Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning | Ronald J. Williams | DOI 10.1007/BF00992696 | Machine Learning 8(3–4):229–256（Crossref） | 1992-05 | Springer PDF（OCR 文本层）：§3–§5，定理 1、2 |
| `sutton2000` | Policy Gradient Methods for Reinforcement Learning with Function Approximation | R. S. Sutton, D. McAllester, S. Singh, Y. Mansour | 无 DOI | Advances in NIPS 12, pp. 1057–1063, MIT Press 2000（PDF 页眉） | 会议 1999 / 出版 2000 | PDF（UPenn 课程镜像）：§1 定义、定理 1 式 (2) |
| `gae` | High-Dimensional Continuous Control Using Generalized Advantage Estimation | J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel | 1506.02438 **v6** | ICLR 2016（Abbeel 主页 bib；另见 lesson-03-doc 读到的 PDF 页眉） | v1 2015-06-08 / v6 2018-10-20 | LaTeXML（arxiv.org/html 渲染即 v6） |
| `adam` | Adam: A Method for Stochastic Optimization | D. P. Kingma, J. Ba | 1412.6980 **v9**，并对照 **v1** | ICLR 2015（v9 页眉） | v1 2014-12-22 / v9 2017-01-30 | **v9 与 v1 两个 PDF**（v9 的 HTML 只是嵌入 PDF，见 §0.4） |
| `adamw` | Decoupled Weight Decay Regularization | I. Loshchilov, F. Hutter | 1711.05101 **v3** | ICLR 2019（arXiv comment） | v1 2017-11-14 / v3 2019-01-04 | LaTeXML（v3）；v1/v2 只看 abs 页标题 |
| `moonlight` | Muon is Scalable for LLM Training | Jingyuan Liu, Jianlin Su 等 28 人（Moonshot AI） | 2502.16982 **v1** | 预印本 | 2025-02-24 | LaTeXML 全文 |
| `bn2024` | Old Optimizer, New Norm: An Anthology | Jeremy Bernstein, Laker Newhouse | 2409.20325 **v2** | 预印本 | 2024-09-30 | LaTeXML：命题 1、4、5，附录 |
| `lai2021` | Stochastic Approximation: from Statistical Origin to Big-Data, Multidisciplinary Applications | Tze Leung Lai, Hongsong Yuan | 无 | 作者主页 PDF 页眉 “Submitted to Statistical Science”，文件日期 2019-12-09 | — | PDF §2.1（用于转述 Robbins–Monro 条件） |
| `robbins1951` | A Stochastic Approximation Method | Herbert Robbins, Sutton Monro | DOI 10.1214/aoms/1177729586 | Ann. Math. Statist. 22(3):400–407（Crossref） | 1951-09 | 【未读原件】Project Euclid 拦截；陈述来自 `lai2021` |
| `polyak1964` | Some methods of speeding up the convergence of iteration methods | B. T. Polyak | DOI 10.1016/0041-5553(64)90137-5 | USSR Comput. Math. Math. Phys. 4(5):1–17（Crossref） | 1964 | 【未读原件】陈述来自 `sutskever2013` §2 |
| `sutskever2013` | On the Importance of Initialization and Momentum in Deep Learning | I. Sutskever, J. Martens, G. Dahl, G. Hinton | 无 | ICML 2013，JMLR W&CP 28（PDF 页脚） | 2013 | PMLR PDF §2 式 (1)–(4) |
| `goyal2017` | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour | P. Goyal, P. Dollár, R. Girshick 等 9 人 | 1706.02677 **v2** | 技术报告 | v2 2018-04-30 | LaTeXML：§2.1、§2.2、§4.1、§5 |
| `mccandlish2018` | An Empirical Model of Large-Batch Training | S. McCandlish, J. Kaplan, D. Amodei, OpenAI Dota Team | 1812.06162 **v1** | 预印本 | 2018-12-14 | LaTeXML：§2.2–§2.4 式 (2.1)–(2.12) |
| `smithle2018` | A Bayesian Perspective on Generalization and Stochastic Gradient Descent | Samuel L. Smith, Quoc V. Le | 1710.06451 **v3** | ICLR 2018（arXiv comment） | 2017-10-17 | grep：噪声尺度 $g$ |
| `minicpm` | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies | Shengding Hu 等 25 人 | 2404.06395 **v3** | arXiv comment 写 “revise according to peer review”，出处本次未核 | 2024-04-09 | LaTeXML：§4.2 式 (1)、§4.3、§6.2 |
| `roofline` | Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures | S. Williams, A. Waterman, D. Patterson | UCB/EECS-2008-134 | 技术报告（2008-10-17）；期刊版 CACM 52(4):65–76，DOI 10.1145/1498765.1498785，Crossref 题名只记 “Roofline” | 2008 / 2009 | **读的是技术报告**，期刊版未读 |
| `thakur2005` | Optimization of Collective Communication Operations in MPICH | R. Thakur, R. Rabenseifner, W. Gropp | DOI 10.1177/1094342005051521 | IJHPCA 19(1):49–66（Crossref） | 2005-02 | HLRS 镜像 PDF（ANL 原址 403）：§3、§4.1、§4.3、§4.5、§5.3 |
| `patarasuk2009` | Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations | Pitch Patarasuk, Xin Yuan | DOI 10.1016/j.jpdc.2008.09.002 | JPDC 69(2):117–124（Crossref） | 2009-02 | 作者主页预印本 PDF：§3 引理 3–5 |
| `goldberg1991` | What Every Computer Scientist Should Know About Floating-Point Arithmetic | David Goldberg | DOI 10.1145/103162.103163 | ACM Computing Surveys 23(1):5–48（Crossref） | 1991-03 | Oracle 文档站的重排版全文：“Relative Error and Ulps”、表 D-1 |
| `ieee754` | IEEE Standard for Floating-Point Arithmetic（2008 / 2019） | IEEE | DOI 10.1109/IEEESTD.2008.4610935；10.1109/IEEESTD.2019.8766229 | IEEE 标准（付费） | 2008 / 2019 | 【未读原件】只用 Crossref 核对了两个 DOI 的题名；参数来自 `goldberg1991`（754-1985 的 single/double）、`bf16` 表 1、`fp8` 与 numpy |
| `bf16` | A Study of BFLOAT16 for Deep Learning Training | D. Kalamkar 等 19 人（Intel、Facebook） | 1905.12322 **v3** | 预印本 | 2019-05-29（v3 2019-06-13） | LaTeXML：§1、§3 表 1 |
| `fp8` | FP8 Formats for Deep Learning | P. Micikevicius 等 15 人 | 2209.05433 **v2** | 预印本 | v1 2022-09-12 / v2 2022-09-29 | LaTeXML：§2、§3 表 1、§3.1、§3.2 |
| `mx` | Microscaling Data Formats for Deep Learning | B. Darvish Rouhani 等 33 人 | 2310.10537 **v3** | 预印本 | 2023-10-16 | LaTeXML：§2、§2.1、§2.2 表 1、§3 算法 1 |
| `nvfp4` | Pretraining Large Language Models with NVFP4 | NVIDIA（Abecassis 等约 90 人） | 2509.25149 **v2** | 预印本（v2 注明修正式 2 的笔误） | 2025-09-29 | LaTeXML：§2、表 1、附录 B.1–B.4 |

### 0.2 制品（artefact，无论文；2026-09-24 读取）

| 短键 | 内容 | 读了什么 |
|---|---|---|
| `jordan-blog` | Keller Jordan, “Muon: An optimizer for hidden layers in neural networks”，kellerjordan.github.io/posts/muon/ | 全文（页面日期 “December 8, 2024”）：定义代码、NS 系数设计、运行时间分析、适用参数 |
| `muon-code` | github.com/KellerJordan/Muon `muon.py`（master 分支） | `zeropower_via_newtonschulz5`、`muon_update`、`Muon.step` |
| `pytorch-adamw` | github.com/pytorch/pytorch `torch/optim/adamw.py`（main 分支） | 默认参数与 docstring 中的算法 |
| `microxcaling` | github.com/microsoft/microxcaling `mx/formats.py`、`mx/mx_ops.py`（MX 论文作者的仿真库） | 各元素格式的 ebits/mbits/emax/max_norm；共享指数的截断与溢出处理 |
| `joschu-kl` | John Schulman, “Approximating KL Divergence”，joschu.net/blog/kl-approx.html（页面 “Posted on 2020/03/07”） | k1、k2、k3 定义与两组实验数字 |
| `abbeel-bib` | people.eecs.berkeley.edu/~pabbeel/abbeel_pubs.bib | `Schulmanetal_ICLR2016` 条目（GAE 的会议出处） |
| `numpy-finfo` | numpy 1.26.4 的 `np.finfo(float16/32/64)`（本机） | eps、epsneg、max、tiny、smallest_subnormal |
| `ocp-mx-spec` | OCP Microscaling Formats (MX) Specification v1.0 | 【未读原件】opencompute.org 返回 HTTP 403；规范条款经 `mx`、`nvfp4`、`microxcaling` 间接核对 |

### 0.3 按课分配

| 数学课 | 服务的 DeepSeek 课 | 本文涉及的来源 |
|---|---|---|
| M01 矩阵与形状 | 1, 4, 9 | 教科书内容，无命名定理；$2mnk$ 在 §4 复算 |
| M02 低秩与 SVD | 9, 18 | `stewart1993`、`eckart1936`、`mirsky1960`、`schmidt1907`、`mml2020`；MLA 维度【交叉引用】lesson-09-doc |
| M03 旋转与复数 | 5 | 欧拉公式与旋转复合为教科书结论，不需引用；RoPE 常数【交叉引用】lesson-05-doc |
| M04 概率与方差 | 1, 4, 8 | 教科书；$\sqrt{d_k}$ 脚注【交叉引用】lesson-01-doc；重要性采样在 §4 复算 |
| M05 熵与 KL | 2, 3, 8, 14 | `mackay2003`、`jensen1906`、`joschu-kl` |
| M06 反向传播 | 1, 2, 16 | 教科书（softmax 雅可比、$p-y$ 在 §4 复算） |
| M07 优化器 | 2, 6, 18 | `robbins1951`/`lai2021`、`polyak1964`/`sutskever2013`、`adam`、`adamw`、`pytorch-adamw`、`jordan-blog`、`muon-code`、`moonlight`、`bn2024`、`dsv4`、`dsv41`、`goyal2017`、`mccandlish2018`、`smithle2018`、`minicpm` |
| M08 浮点与量化 | 12, 15, 19 | `goldberg1991`、`ieee754`、`bf16`、`fp8`、`mx`、`microxcaling`、`nvfp4`、`dsv41`、`numpy-finfo` |
| M09 策略梯度 | 3, 8, 13, 15 | `williams1992`、`sutton2000`、`gae`；PPO/GRPO 细节【交叉引用】lesson-03-doc、lesson-08-doc |
| M10 算力算术 | 2, 4, 12 | `roofline`、`thakur2005`、`patarasuk2009`、`goyal2017` §4.1；硬件数字【交叉引用】lesson-12-doc、mamba lesson-01-doc |
| M11 Sinkhorn 与双随机 | 16（及 18、19） | `birkhoff1946`、`sinkhorn1964`、`sk1967`、`mhc`、`dsv4` §2.2、`dsv41` §2.5 |

### 0.4 工具陷阱（本次研究过程发现，影响以后所有 session）

- `arxiv.py text 1412.6980` 返回的是 **Adam v1**（2014-12-22）而不是 v9。原因：脚本先试 `arxiv.org/html/<id>`，v9 的 HTML
  只是“See pages 1-last of 0_adam_main.pdf”的 PDF 嵌入页，（推断）被脚本的长度检查判为无效，于是退到 `arxiv.org/html/<id>v1`。v1 的算法用
  “$1-\beta$”参数化（见 §7 E16），直接照抄会把默认值写成 $\beta_1=0.1$、$\beta_2=0.001$、$\alpha=0.0002$。
  **凡是 `text` 输出里的版本号，都要在渲染页头 “arXiv:xxxxvN” 上确认。** 本次逐个确认：GAE→v6、Goyal→v2、McCandlish→v1、
  bf16→v3、FP8→v2、AdamW→v3 均为最新版；只有 Adam 例外，改用 v9 PDF。
- Project Euclid（Ann. Math. Statist.）、dblp、opencompute.org、OUP 对脚本返回拦截页或 403。按安全规则不绕过，相关陈述改用二手来源并标【未读原件】。

---

## 1. 核心论断与原文引文

编号 C1–C75，按数学课分组。位置写成“短键 §节 / 式 / 表 / 页”。引号内是不超过 15 词的原文短语，其余为中文转述。

### 1.1 M02 低秩与 SVD：Schmidt、Eckart–Young、Mirsky

- **C1 SVD 存在性（教科书陈述）。** 任意 $A\in\R^{m\times n}$（秩 $r$）都可分解为 $A=U\Sigma V^\top$，$U\in\R^{m\times m}$、$V\in\R^{n\times n}$
  正交，$\Sigma$ 为 $m\times n$ 对角、$\Sigma_{ii}=\sigma_i\ge0$，约定 $\sigma_1\ge\sigma_2\ge\dots$（`mml2020` 定理 4.22，式 (4.64)）。
  存在性的历史：Beltrami、Jordan、Sylvester（有限维，经双线性型）与 Schmidt、Weyl（积分方程）五人（`stewart1993` 摘要与 §1）。
- **C2 谱范数 = 最大奇异值。** `mml2020` 定义 4.23（式 (4.93)）与定理 4.24（书中把证明留作习题 4.12）。
- **C3 Eckart–Young 1936 的问题是最小二乘（Frobenius）意义。** 摘要：把问题写成最小二乘问题时，正规方程写不出来，因为近似矩阵的元素彼此
  不独立；先化为“canonic form”再解；“It is found that the problem always has a solution which is usually unique.”（`eckart1936` 摘要）
  ——“usually unique”对应 $\sigma_k>\sigma_{k+1}$ 时唯一（【笔者】，见 §7 E2）。
- **C4 这个定理最早属于 Schmidt 1907。** Stewart 按 Schmidt 的推导用矩阵记号重述：求形如 $\sum_{i=1}^k\mathbf x_i\mathbf y_i^\top$ 的最佳近似，
  即“秩不超过 $k$”的最佳近似；最优解是截断 SVD $A_k=\sum_{i\le k}\sigma_i\mathbf u_i\mathbf v_i^\top$，误差平方为 $\|A\|^2-\sum_{i\le k}\sigma_i^2$
  （`stewart1993` §5，pp. 559–560，式 (5.2)(5.3)；从 trace 展开可知所用范数是 Frobenius 范数）。Stewart 称它可以算作“SVD 的基本定理”（p. 561，转述）。
- **C5 “Eckart–Young 定理”这个名字是误称。** Stewart：Eckart 与 Young（1936、1939）把分解推广到长方阵，并重新发现了 Schmidt 的近似定理，该定理
  “often (and incorrectly) called the Eckart–Young theorem”（`stewart1993` §7 Envoi，p. 563）。
- **C6 Mirsky 1960 推广到所有酉不变范数（含谱范数）。** Mirsky “showed that A_k of (5.2) is a minimizing matrix in any unitarily invariant norm”
  （`stewart1993` §7 “Approximation theorems”，p. 563；原句 14 词）。同页定义：酉不变范数满足 $\|U^HAV\|_u=\|A\|_u$；von Neumann (1937) 证明每个酉不变范数
  都是奇异值的对称规范函数。谱范数与 Frobenius 范数都是酉不变范数，所以两者下截断 SVD 都最优。【未读原件】
- **C7 谱范数版本的最优误差 $=\sigma_{k+1}$，证明走“零空间维数”。** `mml2020` 定理 4.25 式 (4.95)，证明见式 (4.96)–(4.99)：若 $\|A-B\|_2<\sigma_{k+1}$，
  则 $B$ 的零空间（至少 $n-k$ 维）与 $\mathbf v_1,\dots,\mathbf v_{k+1}$ 张成的 $k+1$ 维子空间有非零交，矛盾（秩–零化度定理）。
  Weyl 的另一证法：$\sigma_1(A-B_k)\ge\sigma_{k+1}(A)$（`stewart1993` §6 式 (6.2)）与 $\sigma_{i+j-1}\le\sigma_i'+\sigma_j''$（式 (6.3)）推出
  $\sigma_i(A-B_k)\ge\sigma_{k+i}$，于是 $\|A-B_k\|_F^2\ge\sigma_{k+1}^2+\dots+\sigma_n^2$（p. 562）。
- **C8 `mml2020` 把谱范数版本归给 Eckart & Young 1936。** 定理 4.25 的标题是 “Eckart-Young Theorem (Eckart and Young, 1936)”，陈述却是谱范数的
  （式 (4.94) 用 $\|\cdot\|_2$）。与 C3、C5、C6 对照，这是“相邻命题替换”：1936 年论文是最小二乘版本（见 §7 E1）。

### 1.2 M11 双随机矩阵、Birkhoff、Sinkhorn–Knopp

- **C9 Birkhoff 定理（凸包表述）。** Sinkhorn–Knopp 在证明中引用：Birkhoff [1] 的定理说 $N$ 阶双随机矩阵之集是
  “the convex hull of the N x N permutation matrices”（`sk1967` p. 345）。`mhc` §4.1 “Geometric Interpretation via the Birkhoff Polytope”
  给出同一表述，并据此把残差映射解释为置换的凸组合。【未读原件】
- **C10 双随机矩阵的谱范数不超过 1（mHC 的用法）。** “The spectral norm of a doubly stochastic matrix is bounded by 1”，因此映射非扩张（`mhc` §4.1
  “Norm Preservation”）。同节“Compositional Closure”：双随机矩阵对乘法封闭，跨层乘积仍双随机；$n=1$ 时退化为标量 1，即恒等映射。
- **C11 Sinkhorn 1964 的结论（经 `sk1967` 转述）。** 每个**正**方阵 $A$ 对应唯一的形如 $D_1AD_2$ 的双随机矩阵（$D_1,D_2$ 为正对角阵，本身在相差一个
  标量因子的意义下唯一），可由交替归一化行与列的迭代极限得到；但对**非负**矩阵，迭代不一定收敛，即使收敛 $D_1,D_2$ 也不一定存在
  （`sk1967` p. 343 引言）。【未读原件】
- **C12 Sinkhorn–Knopp 1967 定理（精确条件）。** 设 $A$ 为 $N\times N$ 非负矩阵。(i) 存在形如 $D_1AD_2$ 的双随机矩阵（$D_1,D_2$ 为主对角线为正的对角阵）
  当且仅当 $A$ 有**完全支撑**（total support）；若存在则唯一；$D_1,D_2$ 在相差标量倍意义下唯一当且仅当 $A$ **完全不可分解**（fully indecomposable）。
  (ii) 交替归一化行与列的迭代收敛到双随机极限，当且仅当 $A$ 有**支撑**（support）；若 $A$ 有完全支撑，极限就是 (i) 中的 $D_1AD_2$；
  若有支撑但非完全支撑，极限不能写成 $D_1AD_2$（`sk1967` p. 344 THEOREM）。摘要的说法：迭代收敛到双随机极限
  “if and only if the matrix A contains at least one positive diagonal”。
- **C13 定义。** 对置换 $\sigma$，序列 $a_{1\sigma(1)},\dots,a_{N\sigma(N)}$ 称为 $A$ 的一条对角线；非负方阵若 $A\ne0$ 且每个正元素都在某条正对角线上，
  称为有完全支撑；含至少一条正对角线称为有支撑（`sk1967` p. 343 DEFINITIONS）。“完全不可分解”沿用 Marcus–Minc 的记号，论文正文未重新定义
  【笔者补充：通常定义为不含 $s\times(N-s)$ 的零子矩阵，$1\le s\le N-1$；需教科书核对，见 §8】。
- **C14 支撑但非完全支撑时，不在正对角线上的正元素会被迭代压到 0**（`sk1967` p. 347 末段，借 Birkhoff 定理论证）。
- **C15 必要性来自 Birkhoff。** 完全支撑对 $D_1AD_2$ 存在的必要性是 Birkhoff 定理的直接推论（`sk1967` p. 345，转述）。
- **C16 mHC 的 Sinkhorn 用法。** 先对 $\tilde{\mathcal H}^{res}_l$ 取 $\exp$ 得正矩阵，再交替做行、列归一化；论文称之为把 $\mathcal H^{res}_l$ “entropically project”
  到 Birkhoff 多面体（`mhc` §1）；式 (9) 为 $\mathbf M^{(t)}=\mathcal T_r(\mathcal T_c(\mathbf M^{(t-1)}))$，“We choose t_max = 20 as a practical value”（`mhc` §4.2）。
  实际只做 20 次迭代，所以反向梯度增益（列和）略偏离 1（`mhc` §5.4 “Stability Analysis”，图 7 说明）；HC 的复合增益最大近 3000，mHC 降低约三个数量级（同节）。
  式 (8)：$\mathcal H^{pre}_l=\sigma(\tilde{\mathcal H}^{pre}_l)$、$\mathcal H^{post}_l=2\sigma(\tilde{\mathcal H}^{post}_l)$、$\mathcal H^{res}_l=\text{Sinkhorn-Knopp}(\tilde{\mathcal H}^{res}_l)$（§4.2）；
  $n=4$ 时额外训练时间开销 6.7%（§4.3）。
- **C17 V4 与 V4.1 的 mHC 设置。** `dsv4` §2.2 式 (8)：$M^{(t)}=\mathcal T_r(\mathcal T_c(M^{(t-1)}))$，$t_{max}=20$；§4.2.1 两个模型都写扩展因子 $n_{hc}=4$、Sinkhorn–Knopp 迭代 20 次。
  `dsv41` §4.2.1 同为 4 与 20。
- **C18 V4.1 的第二处 Sinkhorn：优化器。** `dsv41` §2.5 用“带 Sinkhorn 平衡的动量更新”更新 Engram 表、词嵌入与预测头：
  交替把更新矩阵的**行**与**列**除以各自的 $\ell_2$ 范数，共 $K$ 步（$K$ 为奇数，最后一步是行），再乘 $\sqrt n$ 使行 RMS 为 1（算法 1，式 (7)）；
  $K=11$、$\tau=10^{-3}$、$\varepsilon=10^{-20}$，学习率修正 $\gamma=0.18$（§4.2.2）。【笔者推导】这等价于对 $U_{ij}^2$ 做边际为“行和 1、列和 $m/n$”的 Sinkhorn 缩放
  （行归一化后总平方和为 $m$，列若平衡则每列平方和为 $m/n$，乘 $\sqrt n$ 后行 RMS 与列 RMS 都约为 1，与式 (7) 一致）。

### 1.3 M05 Jensen、Gibbs 与 KL 估计器

- **C19 Gibbs 不等式。** $D_{KL}(P\|Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}$（式 (2.45)）满足 $D_{KL}(P\|Q)\ge0$（式 (2.46)），
  “with equality only if P = Q”；一般不对称，所以虽常被称为“KL 距离”，并非严格意义的距离（`mackay2003` §2.6，p. 34，转述）。
- **C20 Jensen 不等式。** 凸函数定义式 (2.47)；若 $f$ 凸、$x$ 为随机变量，则 $\E[f(x)]\ge f(\E[x])$（式 (2.48)）；若 $f$ 严格凸且等号成立，则 $x$ 为常数；
  凹函数不等号反向（`mackay2003` §2.7，p. 35）。原始文献 Jensen 1906，Acta Math. 30:175–193（只核对元数据）。`mml2020` 在式 (7.30) 之后的注记说，
  非负加权和的一整类不等式都叫 Jensen 不等式（转述）。
- **C21 KL 的三个估计器（`joschu-kl`）。** 样本 $x\sim q$，$r=p(x)/q(x)$，目标 $\mathrm{KL}[q,p]$：$k_1=-\log r$（无偏，方差大，约一半样本为负）；
  $k_2=\tfrac12(\log r)^2$（有偏，方差小）；$k_3=(r-1)-\log r$（无偏且恒非负，因 $\log x\le x-1$；它是 $\log$ 与其切线的竖直距离，属于 Bregman 散度）。
  实验：$q=N(0,1)$、$p=N(0.1,1)$，真 KL 0.005，bias/true：k1 0、k2 0.002、k3 0；stdev/true：20、1.42、1.42。$p=N(1,1)$（真 KL 0.5）时 bias/true 0、0.25、0，
  stdev/true 2、1.73、1.7。另给 $\mathrm{KL}[p,q]$ 的估计 $r\log r-(r-1)$。GRPO 如何用 k3 见【交叉引用】lesson-08-doc C18、C39。

### 1.4 M09 REINFORCE、策略梯度定理、GAE

- **C22 REINFORCE 的定义。** 每个试次后，每个参数的增量 $\Delta w_{ij}=\alpha_{ij}(r-b_{ij})e_{ij}$，$e_{ij}=\partial\ln g_i/\partial w_{ij}$ 称为特征资格
  （characteristic eligibility）；要求基线 $b_{ij}$ 在给定 $W$ 与 $\mathbf x^i$ 时与 $y_i$ 条件独立，学习率因子非负、至多依赖 $\mathbf w^i$ 与 $t$
  （`williams1992` §4，pp. 233–234）。名字是缩写：“REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility”。
- **C23 Williams 定理 1。** 对任何 REINFORCE 算法，$\E\{\Delta W\mid W\}$ 与 $\nabla_W\E\{r\mid W\}$ 的内积非负；若所有 $\alpha_{ij}>0$，内积为零仅当梯度为零；
  若 $\alpha_{ij}=\alpha$ 与 $i,j$ 无关，则 $\E\{\Delta W\mid W\}=\alpha\nabla_W\E\{r\mid W\}$，等价于说 $(r-b_{ij})\partial\ln g_i/\partial w_{ij}$ 是
  $\partial\E\{r\mid W\}/\partial w_{ij}$ 的无偏估计（`williams1992` §4 定理 1，p. 234；证明在附录 A）。
- **C24 Williams 定理 2（回合版）。** 每回合 $k$ 步、回合末只给一个 $r$ 时，$\Delta w_{ij}=\alpha_{ij}(r-b_{ij})\sum_{t=1}^ke_{ij}(t)$，结论与定理 1 相同（`williams1992` §5，p. 237）。
- **C25 Bernoulli 单元例。** $g_i=1-p_i$（$y_i=0$）或 $p_i$（$y_i=1$），特征资格 $\partial\ln g_i/\partial p_i=(y_i-p_i)/(p_i(1-p_i))$（式 (4)(5)，p. 234）。
- **C26 策略梯度定理（Sutton et al. 定理 1）。** 对任何 MDP，平均回报与起始状态两种表述下都有 $\partial\rho/\partial\theta=\sum_sd^\pi(s)\sum_a\frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)$
  （`sutton2000` 式 (2)）。起始状态表述中 $d^\pi(s)=\sum_{t=0}^\infty\gamma^t\Pr\{s_t=s\mid s_0,\pi\}$，是**未归一化的折扣加权**（§1）。
  关键性质：式中没有 $\partial d^\pi(s)/\partial\theta$ 这类项，即策略变化对状态分布的影响不出现，便于采样估计（定理 1 之后一段，转述）。
- **C27 归属。** Sutton 等写明：这种表达对平均回报情形由 Marbach & Tsitsiklis (1998) 首先讨论，基于 Jaakkola–Singh–Jordan (1995) 与 Cao–Chen (1997)；
  他们推广到起始状态表述并给出更直接的证明；Williams (1988, 1992) 的 REINFORCE 理论也可看作蕴含式 (2)（`sutton2000` 定理 1 之后一段，转述）。
- **C28 策略梯度的六种 $\Psi_t$ 与优势函数。** $g=\E[\sum_t\Psi_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)]$（`gae` 式 (1)），$\Psi_t$ 可取总回报、动作之后的回报、减基线版本、
  $Q^\pi$、$A^\pi$、TD 残差；$A^\pi=Q^\pi-V^\pi$（式 (3)）；取 $\Psi_t=A^\pi$ 方差“几乎最低”（§2，转述）。
- **C29 γ 在 GAE 里是方差控制参数。** 问题本身不折扣，$\gamma$ 作为降低方差、引入偏差的算法参数出现（`gae` §2）；$g^\gamma$ 本身就是未折扣梯度的有偏估计（脚注 1）。
- **C30 GAE 定义与化简。** $\delta^V_t=r_t+\gamma V(s_{t+1})-V(s_t)$；$k$ 步估计 $\hat A^{(k)}_t=\sum_{l=0}^{k-1}\gamma^l\delta^V_{t+l}$（式 (11)–(14)）；GAE 是它们的指数加权平均，
  化简为 $\hat A^{GAE(\gamma,\lambda)}_t=\sum_{l=0}^\infty(\gamma\lambda)^l\delta^V_{t+l}$（式 (16)）；$\lambda=0$ 为 $\delta_t$（式 (17)），$\lambda=1$ 为 $\sum_l\gamma^lr_{t+l}-V(s_t)$（式 (18)）（`gae` §3）。
- **C31 偏差–方差。** GAE($\gamma$,1) 无论 $V$ 准不准都是 γ-just，但方差大；GAE($\gamma$,0) 只在 $V=V^{\pi,\gamma}$ 时 γ-just；
  λ<1 只在价值函数不准时才引入偏差；经验上最优 λ 远低于最优 γ（`gae` §3）。实验最佳区间 $\gamma\in[0.96,0.99]$、$\lambda\in[0.92,0.99]$（cart-pole，图 2），
  结论节称 λ 取 $[0.9,0.99]$ 通常最好（§7）。
- **C32 GAE 的来历。** 作者说明该公式此前已由 Kimura & Kobayashi (1998) 与 Wawrzyński (2009) 在在线 actor-critic 中提出，本文贡献是分析以及在批量信赖域算法中的应用（`gae` §1，转述）。

### 1.5 M07 优化器：SGD、动量、Adam、AdamW、Muon、学习率与 batch

- **C33 Robbins–Monro 条件。** 递推 $x_{n+1}=x_n-a_nY_n$ 取权重 $a_n>0$，$\sum a_n^2<\infty$，$\sum a_n=\infty$；前者使 $\sum a_n\epsilon_n$ 在 $L^2$ 与几乎处处意义下收敛（对 i.i.d. 零均值有限方差误差等），
  后者保证 $x_n-\theta$ 的极限为 0；Robbins–Monro 原文通过 $\E(x_{n+1}-\theta)^2$ 的递推证明了 $L^2$ 收敛（`lai2021` §2.1，p. 3，转述）。【未读原件】
- **C34 经典动量（Polyak 1964）。** $v_{t+1}=\mu v_t-\varepsilon\nabla f(\theta_t)$，$\theta_{t+1}=\theta_t+v_{t+1}$（`sutskever2013` 式 (1)(2)）；Nesterov：梯度在 $\theta_t+\mu v_t$ 处取（式 (3)(4)）。
  Polyak 证明在极小点附近，取 $\mu=(\sqrt R-1)/(\sqrt R+1)$（$R$ 为曲率条件数）时所需迭代次数比最速下降少 $\sqrt R$ 倍（`sutskever2013` §2，转述）；
  同节提醒：经典收敛理论针对无噪声梯度，随机情形下渐近局部速度的优势会丧失。【未读原件】
- **C35 Adam 算法 1（v9）。** 算法 1 标题给出的“在所测问题上良好的默认值”为 $\alpha=0.001$、$\beta_1=0.9$、$\beta_2=0.999$、$\epsilon=10^{-8}$；
  $\beta_1,\beta_2\in[0,1)$；偏差修正 $\hat m_t=m_t/(1-\beta_1^t)$、$\hat v_t=v_t/(1-\beta_2^t)$；更新 $\theta_t=\theta_{t-1}-\alpha\hat m_t/(\sqrt{\hat v_t}+\epsilon)$（`adam` v9 p. 2 算法 1）。
- **C36 Adam 偏差修正的推导。** $v_t=(1-\beta_2)\sum_{i=1}^t\beta_2^{t-i}g_i^2$，取期望得 $\E[v_t]=\E[g_t^2](1-\beta_2^t)+\zeta$（式 (1)–(4)），平稳时 $\zeta=0$，故除以 $1-\beta_2^t$（`adam` §3）。
- **C37 Adam 的有效步长界。** 设 $\epsilon=0$，$\Delta_t=\alpha\hat m_t/\sqrt{\hat v_t}$；当 $(1-\beta_1)>\sqrt{1-\beta_2}$ 时 $|\Delta_t|\le\alpha(1-\beta_1)/\sqrt{1-\beta_2}$，否则 $|\Delta_t|\le\alpha$；
  前者只在最严重的稀疏情形出现（`adam` §2.1，转述）。【笔者】默认值下 $(1-\beta_1)/\sqrt{1-\beta_2}=3.162>1$，正属于前一种情形。
- **C38 Adam v1 的参数化不同。** v1 算法 1 写 $\beta_1,\beta_2\in(0,1]$，$m_t\leftarrow\beta_1g_t+(1-\beta_1)m_{t-1}$，修正项 $1-(1-\beta_1)^t$，默认 $\alpha=0.0002$、$\beta_1=0.1$、$\beta_2=0.001$、$\epsilon=10^{-8}$
  （`adam` v1 p. 2）。即 v1 的 $\beta$ 等于 v9 的 $1-\beta$，默认学习率也不同（见 §7 E16）。
- **C39 AdamW：L2 与权重衰减只在 SGD 下等价。** Hanson & Pratt 式权重衰减 $\theta_{t+1}=(1-\lambda)\theta_t-\alpha\nabla f_t(\theta_t)$（`adamw` 式 (1)）；
  命题 1：标准 SGD 用衰减 $\lambda$ 等价于对 $f_t+\frac{\lambda'}2\|\theta\|_2^2$ 做无衰减 SGD，$\lambda'=\lambda/\alpha$；命题 2（标题 “Weight decay ≠ L2 reg for adaptive gradients”）：
  预条件 $M_t\ne kI$ 时不存在这样的 $\lambda'$；命题 3：固定对角预条件 $\mathrm{diag}(s)^{-1}$ 时，解耦衰减等价于尺度调整的 L2 项 $\frac{\lambda'}{2\alpha}\|\theta\odot\sqrt s\|_2^2$（`adamw` §2）。
- **C40 AdamW 算法 2。** 默认 $\alpha=0.001,\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$；第 6 行（L2 版）$g_t\leftarrow\nabla f_t(\theta_{t-1})+\lambda\theta_{t-1}$；
  第 12 行（AdamW）$\theta_t\leftarrow\theta_{t-1}-\eta_t(\alpha\hat m_t/(\sqrt{\hat v_t}+\epsilon)+\lambda\theta_{t-1})$，$\eta_t$ 由 SetScheduleMultiplier 给出（`adamw` 算法 2）。
  注意：衰减项乘的是**调度乘子** $\eta_t$，**不乘** $\alpha$。
- **C41 PyTorch 的 AdamW 与论文写法不同。** 默认 `lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2`；docstring 算法第一步 $\theta_t\leftarrow\theta_{t-1}-\gamma\lambda\theta_{t-1}$，
  其中 $\gamma$ 是 lr（`pytorch-adamw`）。即 PyTorch 的每步衰减量 = lr × weight_decay，论文 = 调度乘子 × λ（见 §7 E18）。
- **C42 AdamW 的版本漂移。** arXiv v1（2017-11-14）与 v2（2018-02-14）题为 “Fixing Weight Decay Regularization in Adam”，v3（2019-01-04）改为 “Decoupled Weight Decay Regularization”（ICLR 2019）（arXiv abs 页）。
- **C43 Muon 的定义（博客）。** Muon 只用于隐藏层的 2D 参数：先得到 SGD-momentum 的更新，再对它做 Newton–Schulz 迭代作为后处理；NS 迭代把更新替换为最近的半正交矩阵，即 $UV^\top$
  （`jordan-blog` “Definition” 与 “Design”，转述）。标量、向量参数以及输入、输出层应该用 AdamW 一类标准方法，嵌入层与最终分类头用 AdamW 效果最好；
  Nesterov 式动量在所有测试中都略好，已设为默认（同页，转述）。
- **C44 NS 系数的来历。** 一步 NS $G'=aG+b(GG^\top)G+c(GG^\top)^2G=U(aS+bS^3+cS^5)V^\top$，$N$ 步等于对奇异值逐元施加 $\varphi^N$，$\varphi(x)=ax+bx^3+cx^5$；
  先用 $G/\|G\|_F$ 把奇异值压进 $[0,1]$；朴素系数 $(2,-1.5,0.5)$ 就收敛；为减少步数，让 $a=\varphi'(0)$ 尽量大，目标是
  “maximize a subject to lim_{N→∞} φ^N(x) ∈ [0.7, 1.3]”，用临时的梯度法得到 $(3.4445,-4.7750,2.0315)$；训练 Transformer 时 5 步足够；三阶、七阶多项式没有进一步降低墙钟开销（`jordan-blog`）。
- **C45 Muon 的代价。** 对 $n\times m$ 参数（设 $m\le n$），每步 NS 需 $2(2nm^2+m^3)$ 次矩阵乘 FLOPs，方阵时至多 $6nm^2$；线性层一步训练的基线 FLOPs 为 $6nmB$；
  故开销至多 $Tm/B$；NanoGPT（$m=768$，$B=524288$）约 0.7%，Llama 405B（$m=16384$，$B=16\times10^6$）约 0.5%（`jordan-blog` “Runtime analysis”）。
- **C46 代码与博客的出入。** `muon-code` docstring：为减少步数，系数选得即使不再在整个区间上收敛到 1 也继续增大 0 处斜率，因此输出不是 $UV^\top$ 而是约 $US'V^\top$，
  “S_{ii}' ~ Uniform(0.5, 1.5)”，且不影响模型表现（转述）；博客写的是 $[0.7,1.3]$。代码另有两处与 Moonlight 论文写法不同：动量写成 EMA
  `momentum.lerp_(grad, 1-beta)`（即 $M\leftarrow\beta M+(1-\beta)G$），形状缩放 `max(1, rows/cols)**0.5`；`Muon` 默认 `lr=0.02, weight_decay=0, momentum=0.95`，
  衰减写作 `p.mul_(1 - lr*weight_decay)`（`muon-code`）。
- **C47 Moonlight 的 Muon（式 (1)–(4)）。** 式 (1)：$M_t=\mu M_{t-1}+\nabla\mathcal L_t(W_{t-1})$，$O_t=\text{Newton-Schulz}(M_t)$，$W_t=W_{t-1}-\eta_tO_t$；脚注 1：实际按 Jordan 用 Nesterov 式，
  把 $\mu M_t+\nabla\mathcal L_t(W_{t-1})$ 送进 NS。式 (2)：$X_0=M_t/\|M_t\|_F$，$X_k=aX_{k-1}+b(X_{k-1}X_{k-1}^\top)X_{k-1}+c(X_{k-1}X_{k-1}^\top)^2X_{k-1}$，沿用 Jordan 系数。
  式 (3)：加入 AdamW 式权重衰减 $W_t=W_{t-1}-\eta_t(O_t+\lambda W_{t-1})$。引理 1：满秩 $[A,B]$ 参数的理论更新 RMS 为 “sqrt(1/max(A,B))”。式 (4)：
  $W_t=W_{t-1}-\eta_t(0.2\cdot O_t\cdot\sqrt{\max(A,B)}+\lambda W_{t-1})$，使 Muon 更新 RMS 与 AdamW 的典型值 0.2–0.4 相当，从而直接复用 AdamW 的学习率与权重衰减（`moonlight` §2.1–§2.2）。
  NS 步数取 5（10 步更准但不更好），动量 0.95（§2.2 末）。摘要：计算最优训练下约 $2\times$ 于 AdamW 的计算效率。
- **C48 为什么要加权重衰减。** 不加时权重与层输出的 RMS 持续增长，超出 bf16 的高精度范围（`moonlight` §2.2 “Weight Decay”，转述）；原始实现没有权重衰减（同节脚注）。
- **C49 为什么要按形状缩放。** $\max(A,B)$ 很大（稠密 MLP）时更新太小，很小（把 GQA/MLA 的每个 KV 头当独立参数）时更新太大（`moonlight` §2.2，转述）。
- **C50 DeepSeek-V4 的 Muon（算法 1，式 (28)）。** 嵌入、预测头、mHC 的静态偏置与门控因子、所有 RMSNorm 权重用 AdamW，其余模块用 Muon；沿用 Liu et al. 的权重衰减、Nesterov 技巧与 RMS 重缩放；
  不同点是“混合 Newton–Schulz”：共 10 次迭代，前 8 次用 $(3.4445,-4.7750,2.0315)$ 快速收敛，后 2 次用 $(2,-1.5,0.5)$，
  论文说这两步把奇异值精确稳定在 1（`dsv4` §2.4，转述）。算法 1：$M_t=\mu M_{t-1}+G_t$；$O_t'=\mathrm{HybridNewtonSchulz}(\mu M_t+G_t)$；
  $O_t=O_t'\cdot\sqrt{\max(n,m)}\cdot\gamma$；$W_t=W_{t-1}(1-\eta\lambda)-\eta O_t$。因注意力的 Q 与 KV 条目上直接有 RMSNorm，不用 QK-Clip（§2.4）。另称 NS 在 BF16 矩阵乘下保持稳定（§3.4.1 附近，转述）。
- **C51 V4-Flash 的优化器超参数。** AdamW：$\beta_1=0.9$、$\beta_2=0.95$、$\varepsilon=10^{-20}$、weight_decay 0.1；Muon：动量 0.95、权重衰减 0.1、每个更新矩阵的 RMS 重缩放到 0.18；
  batch 从小逐步增到 75.5M tokens 后保持；前 2000 步线性 warmup，大部分时间 $2.7\times10^{-4}$，末尾按 cosine 衰减到 $2.7\times10^{-5}$；共 32T tokens（`dsv4` §4.2.2）。
- **C52 V4.1-Flash 的优化器与调度。** 线性变换用 Muon（Q、K 权重用 head-wise Muon），RMSNorm 与其他非矩阵参数用 AdamW，所有嵌入与预测头用 Sinkhorn 平衡更新；AdamW 与 Muon 超参数同 V4
  （0.9 / 0.95 / $10^{-20}$ / 0.1；动量 0.95、衰减 0.1、RMS 0.18）；batch 固定 100.6M tokens；2000 步 warmup 后保持 $2.6\times10^{-4}$ 到 28T tokens，28T–40T 之间 cosine 衰减到 $2.6\times10^{-5}$，
  40T–45T 保持（`dsv41` §2.5、§4.2.2）。
- **C53 Bernstein–Newhouse：正交化更新的几何（论文讲的是 Shampoo，不是 Muon）。** 命题 4、5 在原文里是关于“无累积的 Shampoo”的结论，v2 正文没有出现 “Muon” 一词（只在致谢里感谢 Keller Jordan）。把它们用来解释 Muon 是这一步推理：Muon 关掉动量时的更新就是 $UV^\top$，Jordan 博客也称之为“无累积的 Shampoo”（引 Anil 2024b）。写课时要说“这一几何解释来自 Bernstein–Newhouse 对 Shampoo 的分析，Muon 的对应由 Jordan 博客给出”（核查记录 K8）。命题 1：$\arg\min_{\Delta w}[g^\top\Delta w+\frac\lambda2\|\Delta w\|^2]=-\frac{\|g\|^\dagger}{\lambda}\arg\max_{\|t\|=1}g^\top t$（式 (1)，$\dagger$ 为对偶范数）；
  命题 4：Frobenius 范数下离 $G$ 最近的半正交矩阵是 $UV^\top$（式 (17)），唯一当且仅当 $G$ 满秩；命题 5：谱范数（多层取最大）下最速下降的解为 $\Delta W_l=-\eta U_lV_l^\top$，
  $\eta=\frac1\lambda\sum_l\mathrm{tr}\Sigma_l$（式 (18)(19)）。附录：其 NS 迭代改编自 Higham (2008) 式 5.22，三次式 $f(x)=\tfrac32x-\tfrac12x^3$ 在 $0<x<\sqrt3$ 上把 $x$ 推向 1（`bn2024`）。
- **C54 线性缩放规则。** “When the minibatch size is multiplied by k, multiply the learning rate by k.”（`goyal2017` §2.1），其余超参数不变；依据是在 $\nabla l(x,w_t)\approx\nabla l(x,w_{t+j})$ 的假设下式 (3)(4) 近似相等；
  作者自己列出假设不成立的两种情形：训练初期网络变化快（用 warmup 解决）；batch 不能无限增大，超过某点精度迅速下降，ImageNet 上这个点约为 8k（§2.1，转述）。
- **C55 渐进 warmup。** 从 $\eta$ 起每次迭代加常数，5 个 epoch 后达到 $\hat\eta=k\eta$；结果对 warmup 时长不敏感（`goyal2017` §2.2）。ImageNet 设置 $\eta=0.1\cdot\frac{kn}{256}$，8k batch 时参考学习率 3.2（§5）。
- **C56 梯度噪声尺度。** 真梯度 $G$、Hessian $H$、单样本梯度协方差 $\Sigma$；$\epsilon_{opt}(B)=\epsilon_{max}/(1+\mathcal B_{noise}/B)$（式 (2.6)），$\Delta L_{opt}(B)=\Delta L_{max}/(1+\mathcal B_{noise}/B)$（式 (2.7)），
  $\mathcal B_{noise}=\mathrm{tr}(H\Sigma)/(G^\top HG)$（式 (2.8)）；Hessian 为单位阵倍数时化为 $\mathcal B_{simple}=\mathrm{tr}(\Sigma)/|G|^2$（式 (2.9)）；
  步数–样本数权衡 $S/S_{min}-1=(E/E_{min}-1)^{-1}$（式 (2.11)），$\mathcal B_{crit}=E_{min}/S_{min}$（式 (2.12)），模型预测 $\mathcal B_{crit}\approx\mathcal B_{noise}$（`mccandlish2018` §2.2–§2.3）。
  §2.4 列出四条假设与注意事项（短视偏差、病态条件、简化噪声尺度、学习率未调好）。
- **C57 Smith & Le 的“噪声尺度”是另一个量。** $g=\epsilon(N/B-1)\approx\epsilon N/B$（$\epsilon$ 学习率、$N$ 训练集大小、$B$ batch），推出 $B_{opt}\propto\epsilon N$；带动量 $m$ 时 $g\approx\epsilon N/(B(1-m))$（`smithle2018` 摘要、§5 与附录 D，转述）。
- **C58 WSD 调度。** $\mathrm{WSD}(T;s)=\frac sW\eta$（$s<W$），$\eta$（$W<s<T$），$f(s-T)\eta$（$T<s<S$），$0<f\le1$ 递减（`minicpm` 式 (1)，§4.2）；
  衰减阶段损失骤降；衰减阶段占总 token 的 10% 就足够（§4.3，转述）；MiniCPM 实际用指数退火，文中写作 $f(s-T)=0.5^{(s-S)/T}$，“$T$ 取 5000 步（20B tokens）”
  （§6.2；此式记号有误，见 §7 E19）。

### 1.6 M08 浮点格式与块缩放

- **C59 IEEE 754 单 / 双精度参数（754-1985）。** 单精度 $p=24$、$e_{max}=+127$、$e_{min}=-126$，双精度 $p=53$、$e_{max}=+1023$、$e_{min}=-1022$（`goldberg1991` 表 D-1）；
  单精度指数域 8 位、偏置 127，双精度偏置 1023（同文 “Exponent” 一节）；$|e_{min}|<e_{max}$ 是为了让最小数的倒数不溢出（转述）。
- **C60 “machine epsilon”在 Goldberg 里是半个 ulp。** 就近舍入的相对误差上界 $\epsilon=(\beta/2)\beta^{-p}$ 被称为 machine epsilon（`goldberg1991` “Relative Error and Ulps”，式 (2) 之后）。
  二进制时 $\epsilon=2^{-p}$，单精度 $5.96\times10^{-8}$；而 numpy / C 的 `eps` 是 $2^{-(p-1)}=1.19\times10^{-7}$（`numpy-finfo`）。两种约定相差 2 倍（§7 E33）。
- **C61 bfloat16 与 FP16 的参数表。** FP32 (1,8,23)：最大正规数 $3.40\times10^{38}$、最小正规数 $1.17\times10^{-38}$、最小次正规数 $1.40\times10^{-45}$；FP16 (1,5,10)：$6.55\times10^4$、$6.10\times10^{-5}$、$5.96\times10^{-8}$；
  BFLOAT16 (1,8,7)：$3.38\times10^{38}$、$1.17\times10^{-38}$、次正规 “N/A”；三者累加器均为 float32（`bf16` §3 表 1）。正文称其为截断的全精度值，“8 bits of mantissa”（此处把隐含位算在内），动态范围与 FP32 相当（§3）。
- **C62 bfloat16 不是 Kalamkar 等发明的。** 论文说明 BFLOAT16 最早作为降低通信量的低精度存储格式出现在 DistBelief 与 TensorFlow 中，之后主要在 Google 生态中用于训练（`bf16` §1，转述）。
- **C63 FP8 的两种编码（表 1）。** E4M3：偏置 7，无无穷，NaN 为 S.1111.111，零为 S.0000.000，最大正规数 S.1111.110 $=1.75\times2^8=448$，最小正规数 $2^{-6}$，最大次正规数 $0.875\times2^{-6}$，最小次正规数 $2^{-9}$；
  E5M2：偏置 15，无穷 S.11111.00，NaN S.11111.{01,10,11}，最大正规数 S.11110.11 $=1.75\times2^{15}=57{,}344$，最小正规数 $2^{-14}$，最大次正规数 $0.75\times2^{-14}$，最小次正规数 $2^{-16}$（`fp8` §3 表 1）。
- **C64 E4M3 为什么不守 IEEE 惯例。** E4M3 通过不表示无穷、NaN 只保留一种尾数位型来扩展动态范围（摘要，转述）；这把动态范围从 17 个 binade 扩到 18 个，多出 256、288、…、448 这 7 个值，
  不改的话最大值是 240；为保持正负对称保留了 ±0 与 ±NaN，否则还能多一个 480（`fp8` §3.1，转述）。E5M2 完全遵守 IEEE 惯例，可看成少了尾数位的 IEEE 半精度（§3）；32 个 binade（含次正规）足够（§3.1）。
- **C65 FP8 的使用约定。** 推荐 E4M3 用于权重与激活，E5M2 用于梯度（`fp8` §3）；转为 E4M3 时，更宽类型的无穷与 NaN 都变成 NaN；溢出值饱和到最大可表示值；缩放因子一般选成让张量最大幅值接近格式最大值（§2，转述）。
  【交叉引用】DeepSeek-V3 偏离这一推荐，所有张量都用 E4M3（lesson-12-doc C47）。
- **C66 MX 块的定义。** 一个 MX 块由共享缩放 $X$ 与 $k$ 个元素 $P_i$ 组成，表示的数为 $v_i=XP_i$；$X$ 为 NaN 时整块为 NaN；$|XP_i|$ 超过 FP32 最大值时由实现决定；共享缩放不编码 Inf（`mx` §2、§2.1）。
- **C67 具体 MX 格式（表 1）。** MXFP8（E4M3/E5M2）、MXFP6（E2M3/E3M2）、MXFP4（E2M1）、MXINT8，块大小都是 32，缩放格式都是 E8M0（8 位）；
  所有具体 MX 格式都用 8 位指数格式 E8M0 做共享缩放，其可表示指数是 FP32 的超集（`mx` §2.2 表 1）。
- **C68 MX 转换算法（非规范性示例）。** $shared\_exp\leftarrow\lfloor\log_2\max_i|V_i|\rfloor-emax_{elem}$，$X\leftarrow2^{shared\_exp}$，$P_i=\mathrm{quantize}(V_i/X)$，超出范围的正规数钳到最大可表示值；
  FP32 次正规输入置零是论文为简化所做、规范未描述（`mx` §3 算法 1）。规范允许其他实现定义的转换方案（同节）。
- **C69 FP4 E2M1 的全部取值。** MXFP4 每个元素 1 位符号、2 位指数、1 位尾数，可表示 ±0、±0.5、±1、±1.5、±2、±3、±4、±6；MX 的缩放是无符号 E8M0（UE8M0），表示 $2^{-127}$ 到 $2^{127}$ 的 2 的幂；
  Mishra et al. (2025) 发现把缩放向上取到下一个 UE8M0 值更好，可避免饱和（`nvfp4` §2，转述）。
- **C70 NVFP4 的三处改动。** 块大小从 32 减到 16；块缩放用 E4M3 而不是 UE8M0 存储；外加一个张量级 FP32 缩放（`nvfp4` §2）。
  全局编码缩放 $s_{enc}=6\cdot448/amax_x$（附录 B.1），块解码缩放 $s_{dec,b}=amax_b/6$，存为 $\mathrm{e4m3}(s_{dec,b}\cdot s_{enc})$（B.2）。MXFP4 的 2 的幂缩放会浪费 FP4 取值：块 amax $=3+\delta$ 时解码缩放向上取到 1，块内只用到 $\le3$ 的格点（B.4）。
- **C71 DeepSeek 的 FP4 选择。** V4 已对 indexer 的 Q、K 做 FP4 QAT，并采用 OCP 标准 MXFP4 以兼容更多硬件，尽管其他格式实验精度更高；V4.1 的主 KV cache 选用 E2M1、每 16 个通道一个 E4M3 缩放，
  即 NVFP4 的块结构但**去掉第二级全局缩放**，可表示幅值上限 $448\times6=2688$，远大于 KV 的幅值界（RMSNorm 后 512 维潜向量的 $\ell_2$ 范数约不超过 $\sqrt{512}\approx22.6$，训练中观测最大约 10）（`dsv41` §2.4.4）。同段还写了“训练后 RMSNorm 权重的最大幅值约为 11”；论文没有说明这个权重怎样进入 $\approx\sqrt{512}$ 的上界（若逐通道乘上权重，最坏情形应是 $11\sqrt{512}$），所以写课时只引“上限 2688、观测最大约 10”，不要复述这条上界推理（见 §8）。
- **C72 微缩放仿真库的格式参数。** `microxcaling` `formats.py`：fp6_e3m2（ebits 3，emax 4，max_norm 28）、fp6_e2m3（ebits 2，emax 2，max_norm 7.5）、fp4_e2m1（ebits 2，emax 2，max_norm 6），
  注释写明 fp4 与 fp6 “no NaN/Inf”，fp8_e4m3 的 max_norm 单独设为 $2^{emax}\times1.75$；`mx_ops.py` 把共享指数限制在 $[-127,127]$，超出上界置 NaN、低于下界置 $-127$（随后被冲成 0）。

### 1.7 M10 屋顶线与集合通信

- **C73 Roofline 的定义。** operational intensity 定义为“operations per byte of DRAM traffic”，流量按缓存过滤后到主存的字节计；作者刻意不用 arithmetic intensity，
  因为后者（与 machine balance）度量处理器与缓存之间的流量（`roofline` TR §2，转述）。可达性能 = Min(峰值浮点性能, 峰值内存带宽 × 运算强度)；两条线的交点称为 ridge point，其横坐标是达到峰值所需的最小运算强度（TR §3）。
  【交叉引用】这一术语之争已记入 mamba lesson-01-doc 第 11 条勘误。
- **C74 集合通信代价模型。** 点对点消息时间 $\alpha+n\beta$（$\alpha$ 延迟、$\beta$ 每字节传输时间、$n$ 字节），归约每字节计算代价 $\gamma$；假设与通信对数、距离无关，链路双向，网卡单端口（`thakur2005` §3）。
  环形 allgather $T_{ring}=(p-1)\alpha+\frac{p-1}p n\beta$，带宽项不能再降（§4.1）；递归倍增 allgather $\lg p\,\alpha+\frac{p-1}pn\beta$（§4.1.1）；长消息 all-to-all（成对交换）$T_{long}=(p-1)\alpha+n\beta$（§4.3）；
  Rabenseifner 归约 $2\lg p\,\alpha+2\frac{p-1}pn\beta+\frac{p-1}pn\gamma$（§4.5）；环形 allreduce $T_{all,ring}=2(p-1)\alpha+2n\beta+n\gamma-\frac1p(2n\beta+n\gamma)$（§5.3）。注意 Thakur §5.3 的“Ring Algorithm”在 reduce-scatter 阶段用的是**成对交换**（§4.4，$T=(p-1)\alpha+\frac{p-1}pn\beta+\frac{p-1}pn\gamma$），只有 allgather 阶段走环；它的代价与“环上 reduce-scatter + 环上 allgather”相同，所以公式可以直接当作环形 allreduce 的代价用，但不要说 Thakur 描述的就是纯环算法（核查记录 K7）。
- **C75 环形 allreduce 的带宽最优性。** 引理 3：单元素 all-reduce 在 $N$ 个进程上最少需要通信 $2(N-1)$ 个部分结果；引理 4：不压缩、元素独立时，至少一个进程要通信 $\lceil2\frac{N-1}NX\rceil\times itsize$ 字节；
  引理 5：线性通信模型下最优时间约为“发送全部数据时间”的 2 倍（`patarasuk2009` §3，转述）。Goyal 等的实现：递归减半倍增与环（bucket）算法每台服务器收发 $2\frac{p-1}pb$ 字节；前者 $2\log_2p$ 步、后者 $2(p-1)$ 步；
  32 台服务器（256 GPU）上前者比环快 3 倍（`goyal2017` §4.1）。这个 3 倍是他们系统里的实测：同段说明减半倍增在缓冲区至多约一百万个元素时明显更好（服务器多时阈值更高），不是任意消息大小下的结论。

---

## 2. 公式（原文转录 + 课程记号对照）

每条先按原文记号照抄（带原式号），再给“课程记号”版本。课程记号以 `style-guide.md` §2 为准：学习率 $\eta$、参数 $\theta$、batch $B$、参数量 $N$、token 数 $D$、
PPO/GRPO 裁剪 $\varepsilon$、KL 系数 $\beta$、奖励 $r$、优势 $\hat A$。原文记号与之冲突的地方在 §2.12 统一列出。

### 2.1 SVD 与最优低秩近似（M02）

原文（`mml2020` 式 (4.64)、(4.94)、(4.95)）：

$$A=U\Sigma V^\top,\qquad U\in\R^{m\times m},\ V\in\R^{n\times n}\ \text{正交},\ \Sigma_{ii}=\sigma_i\ge0,\ \Sigma_{ij}=0\ (i\ne j)\tag{4.64}$$

$$\hat A(k)=\sum_{i=1}^k\sigma_i\mathbf u_i\mathbf v_i^\top,\qquad \hat A(k)=\operatorname*{argmin}_{\mathrm{rk}(B)=k}\|A-B\|_2\ \text{(4.94)},\qquad \|A-\hat A(k)\|_2=\sigma_{k+1}\ \text{(4.95)}$$

原文（`stewart1993` §5，Frobenius 范数；Stewart 在 (5.3) 左边印作 $\|\cdot\|$，按上下文应为平方，见 §8）：

$$A_k=\sum_{i=1}^k\sigma_i\mathbf u_i\mathbf v_i^\top\tag{5.2}$$
$$\Big\|A-A_k\Big\|_F^2=\|A\|_F^2-\sum_{i=1}^k\sigma_i^2,\qquad \Big\|A-\sum_{i=1}^k\mathbf x_i\mathbf y_i^\top\Big\|_F^2\ge\|A\|_F^2-\sum_{i=1}^k\sigma_i^2\tag{5.3}$$

Weyl（`stewart1993` §6）：$\sigma_1(A-B_k)\ge\sigma_{k+1}(A)$（6.2）；$A=A'+A''\Rightarrow\sigma_{i+j-1}\le\sigma_i'+\sigma_j''$（6.3）。

**课程记号的完整陈述（M02 用）【笔者整理，依据 C4–C8】**：设 $A\in\R^{m\times n}$，奇异值 $\sigma_1\ge\dots\ge\sigma_r>0$，$A_k=\sum_{i\le k}\sigma_i\mathbf u_i\mathbf v_i^\top$，$k<r$。则
$$\min_{\rank B\le k}\|A-B\|_2=\sigma_{k+1},\qquad \min_{\rank B\le k}\|A-B\|_F=\Big(\sum_{i>k}\sigma_i^2\Big)^{1/2},$$
两个最小值都由 $A_k$ 取到；对任意酉不变范数 $A_k$ 都是极小元（Mirsky）。Frobenius 范数下当 $\sigma_k>\sigma_{k+1}$ 时极小元唯一；谱范数下极小元一般**不唯一**（§4 算例 N23）。
约束写“$\rank B\le k$”而不是“$=k$”（Schmidt 的原始问题就是“秩不超过 $k$”，C4）。

### 2.2 双随机矩阵、Birkhoff、Sinkhorn–Knopp（M11）

定义：$A\in\R^{n\times n}$，$a_{ij}\ge0$，$A\mathbf 1=\mathbf 1$，$A^\top\mathbf 1=\mathbf 1$。Birkhoff（经 `sk1967` p. 345）：
$$\mathcal B_n=\{A\ \text{双随机}\}=\operatorname{conv}\{P_\sigma:\sigma\in S_n\}.$$

Sinkhorn–Knopp 迭代（`sk1967` p. 346 的写法，$\alpha,\beta$ 为当前行、列和的倒数）：
$$x_{i,0}=1,\quad y_{j,0}=\Big(\sum_i a_{ij}\Big)^{-1},\quad x_{i,n+1}=\alpha_{i,n}^{-1}x_{i,n},\quad y_{j,n+1}=\beta_{j,n}^{-1}y_{j,n}$$
（扫描/抽取文本对这一段有损，式子的完整形状以原文为准；含义是交替让 $(x_iA_{ij}y_j)$ 行随机、列随机。）

mHC（`mhc` §4.2 式 (9)；`dsv4` §2.2 式 (8) 同形）：
$$\mathbf M^{(0)}=\exp(\tilde{\mathcal H}^{res}_l),\qquad \mathbf M^{(t)}=\mathcal T_r\big(\mathcal T_c(\mathbf M^{(t-1)})\big),\qquad \mathcal H^{res}_l=\mathbf M^{(t_{max})},\ t_{max}=20.$$
$\mathcal T_c$ 把每列除以列和，$\mathcal T_r$ 把每行除以行和；**最后一步是行归一化，所以有限步输出的行和精确为 1，列和只近似为 1**（§4 算例 N33）。

**课程记号的谱范数界【笔者推导，M11 正文可直接用】**：对双随机 $A$ 与任意 $\mathbf x$，由 Cauchy–Schwarz（权重 $a_{ij}$、行和为 1）
$$\|A\mathbf x\|_2^2=\sum_i\Big(\sum_ja_{ij}x_j\Big)^2\le\sum_i\Big(\sum_ja_{ij}\Big)\Big(\sum_ja_{ij}x_j^2\Big)=\sum_jx_j^2\sum_ia_{ij}=\|\mathbf x\|_2^2,$$
最后一步用了列和为 1；又 $A\mathbf 1=\mathbf 1$，故 $\|A\|_2=1$（等号可取到，不是 $<1$）。另一条路：Birkhoff 分解 + 三角不等式 $\|\sum_k t_kP_k\|_2\le\sum_kt_k\|P_k\|_2=1$。
只有行随机不够：全部行都是 $\mathbf e_1^\top$ 的 $n\times n$ 矩阵行和为 1，但 $\|\cdot\|_2=\sqrt n$（§4 N29）。

V4.1 的 Sinkhorn 平衡更新（`dsv41` §2.5 算法 1、式 (7)），原文记号：
$$M_t\leftarrow\beta M_{t-1}+(1-\beta)G_t,\quad \hat G_t\leftarrow\beta M_t+(1-\beta)G_t,\quad U^{(k)}_{i,:}\leftarrow\frac{U^{(k-1)}_{i,:}}{\|U^{(k-1)}_{i,:}\|_2+\varepsilon}\ (k\ \text{奇}),\quad U^{(k)}_{:,j}\leftarrow\frac{U^{(k-1)}_{:,j}}{\|U^{(k-1)}_{:,j}\|_2+\varepsilon}\ (k\ \text{偶})$$
$$\Delta_t=\sqrt n\,U^{(K)}=\sqrt n\,D_r\hat G_tD_c,\qquad \tfrac1n\sum_j(\Delta_t)_{ij}^2\approx1,\quad \tfrac1m\sum_i(\Delta_t)_{ij}^2\approx1,\qquad W_{t+1}=W_t-\gamma\eta_t\Delta_t.\tag{7}$$
注意这里的 $\beta$ 是动量系数、$\gamma=0.18$ 是学习率修正，都与课程记号的 KL 系数 $\beta$、折扣 $\gamma$ 冲突（§2.12）。

### 2.3 Jensen、Gibbs 与 KL 估计器（M05）

$$f(\lambda x_1+(1-\lambda)x_2)\le\lambda f(x_1)+(1-\lambda)f(x_2)\tag{MacKay 2.47}$$
$$\E[f(x)]\ge f(\E[x])\quad(f\ \text{凸})\tag{MacKay 2.48}$$
$$D_{KL}(P\|Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}\ (2.45),\qquad D_{KL}(P\|Q)\ge0\ (2.46),\ \text{等号当且仅当}\ P=Q.$$
Gibbs 不等式由 Jensen 推出【教科书推导，M05 正文用】：$-D_{KL}(P\|Q)=\E_P[\log(Q/P)]\le\log\E_P[Q/P]=\log\sum_{x:P(x)>0}Q(x)\le\log1=0$。
**最后一个不等号要求 $Q$ 是归一化分布**；$Q$ 不归一时结论可以不成立（§4 N38）。

KL 估计器（`joschu-kl`），$x\sim q$，$r=p(x)/q(x)$，估计 $\mathrm{KL}[q,p]$：
$$k_1=-\log r,\qquad k_2=\tfrac12(\log r)^2,\qquad k_3=(r-1)-\log r;\qquad \mathrm{KL}[p,q]\ \text{的估计}:\ r\log r-(r-1).$$
课程记号（GRPO，lesson-08）：$q\to\pi_\theta$，$p\to\pi_{ref}$，$r=\pi_{ref}(o_t\mid\cdot)/\pi_\theta(o_t\mid\cdot)$。

### 2.4 REINFORCE、策略梯度定理（M09）

Williams（§4、§5）：
$$\Delta w_{ij}=\alpha_{ij}(r-b_{ij})e_{ij},\quad e_{ij}=\frac{\partial\ln g_i}{\partial w_{ij}};\qquad \Delta w_{ij}=\alpha_{ij}(r-b_{ij})\sum_{t=1}^ke_{ij}(t)\ \text{(回合版)}$$
$$\text{定理 1：}\ \langle\E\{\Delta W\mid W\},\nabla_W\E\{r\mid W\}\rangle\ge0;\quad \alpha_{ij}\equiv\alpha\Rightarrow\E\{\Delta W\mid W\}=\alpha\nabla_W\E\{r\mid W\}.$$
Sutton et al.（式 (2)）：
$$\frac{\partial\rho}{\partial\theta}=\sum_sd^\pi(s)\sum_a\frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a),\qquad d^\pi(s)=\sum_{t=0}^\infty\gamma^t\Pr\{s_t=s\mid s_0,\pi\}\ \text{(起始状态表述)}.$$

课程记号：$w\to\theta$，$\alpha_{ij}\to\eta$，$g_i\to\pi_\theta$，$e\to\nabla_\theta\log\pi_\theta$，$\rho(\pi)\to J(\theta)$；
$$\nabla_\theta J(\theta)=\E_{\tau\sim\pi_\theta}\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(G_t-b(s_t)\big)\Big].$$
（$G_t$ 表示回报时与 GRPO 组大小 $G$ 冲突，M09 建议回报写 $R_t$。）

**基线不改变期望（M09 正文用）【教科书推导】**：$\E_{a\sim\pi}[\nabla\log\pi(a\mid s)\,b(s)]=b(s)\sum_a\nabla\pi(a\mid s)=b(s)\nabla\sum_a\pi(a\mid s)=b(s)\nabla1=0$。
Williams 的条件更一般：只要求 $b_{ij}$ 在给定 $W,\mathbf x$ 时与动作 $y_i$ 条件独立（C22）。

### 2.5 GAE（M09）

原文（`gae` §3）：
$$\delta^V_t=r_t+\gamma V(s_{t+1})-V(s_t)\qquad\text{（§3，式 (10) 之前给出的定义）}$$
$$\hat A^{(k)}_t=\sum_{l=0}^{k-1}\gamma^l\delta^V_{t+l}=-V(s_t)+r_t+\gamma r_{t+1}+\dots+\gamma^{k-1}r_{t+k-1}+\gamma^kV(s_{t+k})\tag{14}$$
$$\hat A^{GAE(\gamma,\lambda)}_t=(1-\lambda)\big(\hat A^{(1)}_t+\lambda\hat A^{(2)}_t+\lambda^2\hat A^{(3)}_t+\dots\big)=\sum_{l=0}^\infty(\gamma\lambda)^l\delta^V_{t+l}\tag{16}$$
$$\text{GAE}(\gamma,0):\ \hat A_t=\delta_t\ (17);\qquad \text{GAE}(\gamma,1):\ \hat A_t=\sum_{l=0}^\infty\gamma^l\delta_{t+l}=\sum_{l=0}^\infty\gamma^lr_{t+l}-V(s_t)\ (18)$$
实现用的倒推递归【教科书推导】：$\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$，终止处 $\hat A_T=\delta_T$（§4 N70）。课程记号：$\hat A_t$ 同 style-guide；$\lambda$ 与权重衰减 $\lambda$ 冲突（§2.12）。

### 2.6 SGD、动量、Adam、AdamW（M07）

Robbins–Monro（经 `lai2021`）：$x_{n+1}=x_n-a_nY_n$，$a_n>0$，$\sum_na_n=\infty$，$\sum_na_n^2<\infty$。课程记号：$\theta_{t+1}=\theta_t-\eta_t\hat{\mathbf g}_t$，$\sum\eta_t=\infty$，$\sum\eta_t^2<\infty$。

经典动量与 Nesterov（`sutskever2013` 式 (1)–(4)）：
$$v_{t+1}=\mu v_t-\varepsilon\nabla f(\theta_t),\ \ \theta_{t+1}=\theta_t+v_{t+1};\qquad v_{t+1}=\mu v_t-\varepsilon\nabla f(\theta_t+\mu v_t),\ \ \theta_{t+1}=\theta_t+v_{t+1}.$$
课程记号：$\varepsilon\to\eta$（$\varepsilon$ 留给 PPO 裁剪）。等价的“EMA 写法” $m_t=\beta m_{t-1}+(1-\beta)g_t$ 与上式等价：取 $v_t=-\eta m_t$ 可得 $v_{t+1}=\beta v_t-\eta(1-\beta)g_t$，即重球的 $\varepsilon$ 对应 EMA 写法的 $\eta(1-\beta)$【笔者推导】。

Adam v9（算法 1）：
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\quad \hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\quad \theta_t=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$
等价高效写法（§2）：$\alpha_t=\alpha\sqrt{1-\beta_2^t}/(1-\beta_1^t)$，$\theta_t=\theta_{t-1}-\alpha_tm_t/(\sqrt{v_t}+\hat\epsilon)$（注意 $\hat\epsilon$ 与 $\epsilon$ 不是同一个数）。
偏差修正（§3 式 (1)–(4)）：$v_t=(1-\beta_2)\sum_{i=1}^t\beta_2^{t-i}g_i^2$，$\E[v_t]=\E[g_t^2](1-\beta_2^t)+\zeta$。

Adam v1（仅作版本漂移记录）：$m_t=\beta_1g_t+(1-\beta_1)m_{t-1}$，$\hat m_t=m_t/(1-(1-\beta_1)^t)$，默认 $\beta_1=0.1$、$\beta_2=0.001$、$\alpha=0.0002$。

AdamW（`adamw` 式 (1)，算法 2 第 6、12 行）：
$$\theta_{t+1}=(1-\lambda)\theta_t-\alpha\nabla f_t(\theta_t)\tag{1}$$
$$\text{L2 版（第 6 行）}:\ g_t\leftarrow\nabla f_t(\theta_{t-1})+\lambda\theta_{t-1};\qquad \text{AdamW（第 12 行）}:\ \theta_t\leftarrow\theta_{t-1}-\eta_t\Big(\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\theta_{t-1}\Big).$$
PyTorch（`pytorch-adamw`）：$\theta_t\leftarrow\theta_{t-1}-\gamma\lambda\theta_{t-1}$，随后 $\theta_t\leftarrow\theta_t-\gamma\hat m_t/(\sqrt{\hat v_t}+\epsilon)$（$\gamma$=lr）。

**课程记号（M07 正文统一用这一版）**：学习率 $\eta_t$（含调度），权重衰减系数 $\lambda_{\mathrm{wd}}$，Adam 的数值常数写 $\epsilon_{\mathrm{A}}$：
$$\theta_t=\theta_{t-1}-\eta_t\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon_{\mathrm A}}+\lambda_{\mathrm{wd}}\theta_{t-1}\Big)\quad(\text{PyTorch 约定：衰减量} =\eta_t\lambda_{\mathrm{wd}}).$$
与论文互换：课程的 $\eta_t$ = 论文的 $\alpha\cdot\eta_t^{\text{论文}}$；每步衰减量论文为 $\eta_t^{\text{论文}}\lambda$，课程为 $\eta_t\lambda_{\mathrm{wd}}=\alpha\eta_t^{\text{论文}}\lambda_{\mathrm{wd}}$，两者相等要求 $\lambda_{\mathrm{wd}}=\lambda/\alpha$【笔者推导】。

### 2.7 Muon 与正交化（M07）

Jordan 博客（NS 一步，`jordan-blog` “Design”）：
$$G'=aG+b(GG^\top)G+c(GG^\top)^2G=U(aS+bS^3+cS^5)V^\top,\qquad \varphi(x)=ax+bx^3+cx^5,\ (a,b,c)=(3.4445,-4.7750,2.0315).$$
Moonlight（式 (1)–(4)）：
$$M_t=\mu M_{t-1}+\nabla\mathcal L_t(W_{t-1}),\quad O_t=\text{Newton-Schulz}(M_t),\quad W_t=W_{t-1}-\eta_tO_t\tag{1}$$
$$X_0=\frac{M_t}{\|M_t\|_F},\quad X_k=aX_{k-1}+b(X_{k-1}X_{k-1}^\top)X_{k-1}+c(X_{k-1}X_{k-1}^\top)^2X_{k-1}\tag{2}$$
$$W_t=W_{t-1}-\eta_t(O_t+\lambda W_{t-1})\tag{3}$$
$$W_t=W_{t-1}-\eta_t\big(0.2\cdot O_t\cdot\sqrt{\max(A,B)}+\lambda W_{t-1}\big)\tag{4}$$
引理 1：满秩 $[A,B]$ 时 $\mathrm{RMS}(UV^\top)=\sqrt{1/\max(A,B)}$。【笔者推导】$UV^\top$ 有 $\min(A,B)$ 个奇异值为 1，$\|UV^\top\|_F^2=\min(A,B)$，
$\mathrm{RMS}=\sqrt{\min(A,B)/(AB)}=1/\sqrt{\max(A,B)}$。

DeepSeek-V4（算法 1、式 (28)）：
$$M_t=\mu M_{t-1}+G_t,\quad O_t'=\mathrm{HybridNS}(\mu M_t+G_t),\quad O_t=O_t'\sqrt{\max(n,m)}\,\gamma,\quad W_t=W_{t-1}(1-\eta\lambda)-\eta O_t;$$
$$M_k=aM_{k-1}+b(M_{k-1}M_{k-1}^\top)M_{k-1}+c(M_{k-1}M_{k-1}^\top)^2M_{k-1}\tag{28}$$
前 8 步 $(3.4445,-4.7750,2.0315)$，后 2 步 $(2,-1.5,0.5)$；$\gamma$ 使更新 RMS 为 0.18（§4.2.2）。

Bernstein–Newhouse 命题 5（式 (18)(19)）：
$$\operatorname*{argmin}_{\Delta W_1..\Delta W_L}\Big[\sum_l\langle G_l,\Delta W_l\rangle+\frac\lambda2\max_l\|\Delta W_l\|_{\ell_2\to\ell_2}^2\Big]:\quad \Delta W_l=-\eta U_lV_l^\top,\quad \eta=\frac1\lambda\sum_l\mathrm{tr}\Sigma_l.$$

**课程记号（M07）**：梯度矩阵 $\mathbf G_t$，动量 $\mathbf M_t$（粗体，与 DeepSeek LLM 的标量 $M$ 区分），正交化算子 $\mathrm{Ortho}(\mathbf X)=UV^\top$，更新 RMS 目标 $\rho_{\mathrm{rms}}$（Moonlight 0.2，DeepSeek 0.18）：
$$\mathbf M_t=\mu\mathbf M_{t-1}+\mathbf G_t,\quad \mathbf O_t=\mathrm{NS}_T(\mu\mathbf M_t+\mathbf G_t)\approx\mathrm{Ortho}(\cdot),\quad \theta_t=\theta_{t-1}(1-\eta_t\lambda_{\mathrm{wd}})-\eta_t\,\rho_{\mathrm{rms}}\sqrt{\max(n,m)}\,\mathbf O_t.$$

### 2.8 学习率调度与 batch（M07）

线性缩放（`goyal2017` 式 (3)(4)）：
$$w_{t+k}=w_t-\eta\frac1n\sum_{j<k}\sum_{x\in\mathcal B_j}\nabla l(x,w_{t+j})\ (3),\qquad \hat w_{t+1}=w_t-\hat\eta\frac1{kn}\sum_{j<k}\sum_{x\in\mathcal B_j}\nabla l(x,w_t)\ (4);$$
若 $\nabla l(x,w_t)\approx\nabla l(x,w_{t+j})$，取 $\hat\eta=k\eta$ 则 $\hat w_{t+1}\approx w_{t+k}$。ImageNet：$\eta=0.1\cdot kn/256$。

噪声尺度（`mccandlish2018`）：
$$G_{est}=\frac1B\sum_{i=1}^B\nabla_\theta L_{x_i}(\theta)\ (2.1),\quad \mathrm{cov}(G_{est})=\frac1B\Sigma\ (2.2),\quad \E[L(\theta-\epsilon G_{est})]=L-\epsilon|G|^2+\frac12\epsilon^2\Big(G^\top HG+\frac{\mathrm{tr}(H\Sigma)}B\Big)\ (2.5)$$
$$\epsilon_{opt}(B)=\frac{\epsilon_{max}}{1+\mathcal B_{noise}/B},\ \epsilon_{max}=\frac{|G|^2}{G^\top HG}\ (2.6);\quad \Delta L_{opt}=\frac{\Delta L_{max}}{1+\mathcal B_{noise}/B},\ \Delta L_{max}=\frac12\frac{|G|^4}{G^\top HG}\ (2.7)$$
$$\mathcal B_{noise}=\frac{\mathrm{tr}(H\Sigma)}{G^\top HG}\ (2.8),\qquad \mathcal B_{simple}=\frac{\mathrm{tr}(\Sigma)}{|G|^2}\ (2.9),\qquad \frac S{S_{min}}-1=\Big(\frac E{E_{min}}-1\Big)^{-1}\ (2.11),\qquad \mathcal B_{crit}=\frac{E_{min}}{S_{min}}\ (2.12)$$
课程记号：$\epsilon\to\eta$（步长），$G\to\mathbf g=\nabla\mathcal L$（避开 GRPO 的 $G$），$H$ 保留（Hessian，与 mHC 的 $\mathcal H$ 用字体区分），参数维数原文写 $D$，课程改写 $P$。

WSD（`minicpm` 式 (1)）：
$$\mathrm{WSD}(T;s)=\begin{cases}\frac sW\eta,&s<W\\ \eta,&W<s<T\\ f(s-T)\eta,&T<s<S\end{cases},\qquad 0<f(s-T)\le1\ \text{递减}.$$
课程记号：峰值学习率 $\eta_{\max}$，warmup 结束步 $W$，稳定期结束步 $s_{\mathrm{dec}}$（原文 $T$ 与序列长度 $T$ 冲突），总步数 $S$；指数退火建议写 $f(u)=2^{-u/h}$，$h$ 为半衰期步数（MiniCPM $h=5000$）。

### 2.9 浮点数的数值公式（M08）

正规数与次正规数（二进制，$E$ 位指数、$M$ 位尾数、偏置 $b$）【教科书定义，与 `goldberg1991`、`fp8` 表 1 一致】：
$$x=(-1)^s\,(1.m_1m_2\ldots m_M)_2\times2^{e-b}\ (1\le e\le e_{\max}^{\mathrm{code}}),\qquad x=(-1)^s\,(0.m_1\ldots m_M)_2\times2^{1-b}\ (e=0).$$
IEEE 类格式 $b=2^{E-1}-1$；最大正规数 $(2-2^{-M})2^{e_{max}}$（E4M3 例外：顶码仍是正规数，只去掉 S.1111.111，故 $1.75\times2^8$）；最小正规数 $2^{1-b}$；最小次正规数 $2^{1-b-M}$；
$\epsilon_{\mathrm{mach}}=2^{-M}$（相邻两数在 1 附近的间距，numpy 约定），单位舍入 $u=2^{-(M+1)}$（Goldberg 约定，$p=M+1$）。

MX 块（`mx` §2、§3 算法 1）：$v_i=XP_i$；$shared\_exp=\lfloor\log_2\max_i|V_i|\rfloor-emax_{elem}$，$X=2^{shared\_exp}$，$P_i=\mathrm{quantize}(V_i/X)$（钳位）。
NVFP4（`nvfp4` B.1、B.2）：$s_{enc}=\frac{6\cdot448}{amax_x}$，$s_{dec}=1/s_{enc}$；$s_{dec,b}=\frac{amax_b}6$，$s_{dec,b,e4m3}=\mathrm{e4m3}(s_{dec,b}\cdot s_{enc})$；$s_{enc,b}=1/(\mathrm{fp32}(s_{dec,b,e4m3})\cdot s_{dec})$；$\hat x_i=q(x_i\cdot s_{enc,b})$。
课程记号：缩放因子统一写 $s$（style-guide 类比 8 的“放大镜”），块大小 $k_{\mathrm{blk}}$（避开 top-$k$ 的 $k$）。

### 2.10 屋顶线与通信（M10）

Roofline（TR §3）：$\text{Attainable GFlops/s}=\min(\text{Peak FP},\ \text{Peak BW}\times I)$，ridge point $I^*=\text{Peak FP}/\text{Peak BW}$。
Thakur（§3–§5）：点对点 $\alpha+n\beta$；ring allgather $(p-1)\alpha+\frac{p-1}pn\beta$；Rabenseifner reduce $2\lg p\,\alpha+2\frac{p-1}pn\beta+\frac{p-1}pn\gamma$；
ring allreduce $2(p-1)\alpha+2\frac{p-1}pn\beta+\frac{p-1}pn\gamma$【把 §5.3 的 $2n\beta+n\gamma-\frac1p(2n\beta+n\gamma)$ 合并】；长消息 all-to-all $(p-1)\alpha+n\beta$。
下界（`patarasuk2009` 引理 4）：**至少有一个进程**要通信 $\lceil2\frac{N-1}NX\rceil\cdot itsize$ 字节（引理讲的是“存在一个进程”，不是“每个进程”；环形算法让每个进程都恰好达到这个量，所以是最优的）。
矩阵乘 FLOPs【教科书】：$C=AB$，$A\in\R^{m\times k}$、$B\in\R^{k\times n}$：$mn$ 个输出、每个 $k$ 次乘 + $k-1$ 次加 $\approx2mnk$。
课程记号：进程数 $p$（与概率 $p$ 冲突时写 $n_{\mathrm{gpu}}$），消息字节数写 $S_{\mathrm{msg}}$（原文 $n$ 与 mHC 的 $n$、SK 的 $N$ 冲突）；$\gamma$（归约计算代价）与折扣 $\gamma$ 冲突，M10 写 $\gamma_{\mathrm{red}}$。

### 2.11 欧拉公式与旋转复合（M03）

教科书结论，不需要文献引用：$e^{i\phi}=\cos\phi+i\sin\phi$；$e^{i\alpha}e^{i\beta}=e^{i(\alpha+\beta)}$ 与 $R(\alpha)R(\beta)=R(\alpha+\beta)$（$R(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\ \sin\phi&\cos\phi\end{pmatrix}$）等价；
$R(\phi)^\top=R(-\phi)$，所以 $\langle R(m\theta)\mathbf q,R(n\theta)\mathbf k\rangle=\mathbf q^\top R((n-m)\theta)\mathbf k$ 只依赖 $n-m$（数值验证见 §4 N47）。RoPE 的原文公式与频率常数以 lesson-05-doc 为准（【交叉引用】该文第 4–5 条：$\theta_i=10000^{-2(i-1)/d}$，$i=1..d/2$；等价的 0 起下标写法 $10000^{-2i/d}$，$i=0..d/2-1$）。

### 2.12 记号冲突总表（写课时必须处理）

| 原文符号 | 出处与含义 | 与课程记号冲突 | 建议写法 |
|---|---|---|---|
| $\alpha$ | Adam/AdamW 步长；Williams 学习率因子 | style-guide 学习率是 $\eta$ | 统一写 $\eta$ |
| $\eta_t$ | AdamW 的调度乘子（不是学习率） | $\eta$ = 学习率 | 写 $s_t$ 或 $c_t$ |
| $\beta_1,\beta_2$ | Adam 矩衰减率 | $\beta$ = KL 系数 | 必须带下标；V4.1 算法 1 的动量 $\beta$ 改写 $\mu$ |
| $\epsilon$ | Adam 数值常数；McCandlish 步长；Sutskever 学习率 | $\varepsilon$ = PPO 裁剪 | Adam 常数写 $\epsilon_{\mathrm A}$；步长写 $\eta$ |
| $\lambda$ | AdamW 衰减；GAE 参数；BN 命题 1 的锐度；Muon 衰减 | 无定义，但一课内多义 | 衰减写 $\lambda_{\mathrm{wd}}$，GAE 保留 $\lambda$，锐度写 $\kappa$ |
| $\gamma$ | GAE/Sutton 折扣；PyTorch lr；V4 RMS 因子；V4.1 学习率修正；Thakur 归约代价 | — | 只把 $\gamma$ 留给折扣；RMS 目标写 $\rho_{\mathrm{rms}}$ |
| $\mu$ | 动量（Sutskever、Moonlight、V4） | — | 保留 |
| $M$, $M_t$ | Moonlight/V4 动量矩阵 | DeepSeek LLM 的 $M$（每 token 非嵌入 FLOPs） | 粗体 $\mathbf M_t$ |
| $G$, $G_t$ | McCandlish 真梯度；V4 梯度矩阵 | GRPO 组大小 $G$ | $\mathbf g$ / $\mathbf G_t$（粗体） |
| $D$ | McCandlish 参数维数；数据集大小（脚注） | $D$ = 训练 token 数 | $P$ |
| $N$ | Smith–Le 训练集大小；SK 矩阵阶数 | $N$ = 参数量 | $N_{\mathrm{data}}$；矩阵阶数写 $n$ |
| $n$ | Thakur 消息字节；mHC 扩展率 | — | $S_{\mathrm{msg}}$；扩展率写 $n_{hc}$（V4 记号） |
| $T$ | MiniCPM 稳定期结束步 / 半衰期 | $T$ = 序列长度 | $s_{\mathrm{dec}}$、$h$ |
| $k$ | Goyal worker 数；MX 块大小；Eckart–Young 目标秩 | top-$k$ | 块大小 $k_{\mathrm{blk}}$；秩保留 $k$ |
| $H$ | McCandlish Hessian | mHC $\mathcal H$ | 保留（字体区分） |
| $\rho(\pi)$ | Sutton 性能度量 | — | $J(\theta)$ |
| $w_{ij}$, $g_i$ | Williams 权重、概率质量函数 | $\mathbf g$ = 梯度 | $\theta$、$\pi_\theta$ |

---

## 3. 规格与超参数表

### 3.1 数值格式总表（M08）

“来源”列：**印**=该数字在来源中原样印出；**算**=由位布局按 §2.9 公式复算（Python 见 §4.1），并与来源交叉核对。$\epsilon$ 列用 numpy 约定 $2^{-(p-1)}$，$u$ 列用 Goldberg 约定 $2^{-p}$。

| 格式 | s/e/m | 偏置 | $p$ | 最大正规数 | 最小正规数 | 最小次正规数 | $\epsilon=2^{-M}$ | $u=2^{-(M+1)}$ | 特殊值 | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|
| binary64 | 1/11/52 | 1023 | 53 | $1.7977\times10^{308}$ | $2^{-1022}=2.2251\times10^{-308}$ | $2^{-1074}\approx4.94\times10^{-324}$ | $2.22\times10^{-16}$ | $1.11\times10^{-16}$ | ±Inf，NaN | $p$、$e_{max}$、$e_{min}$、偏置：印 `goldberg1991` 表 D-1；其余算；numpy finfo 一致 |
| binary32 | 1/8/23 | 127 | 24 | $3.4028\times10^{38}$ | $2^{-126}=1.1755\times10^{-38}$ | $2^{-149}=1.4013\times10^{-45}$ | $1.19\times10^{-7}$ | $5.96\times10^{-8}$ | ±Inf，NaN | 同上；$3.40\times10^{38}$/$1.17\times10^{-38}$/$1.40\times10^{-45}$ 印 `bf16` 表 1 |
| binary16（FP16） | 1/5/10 | 15 | 11 | 65504 | $2^{-14}=6.1035\times10^{-5}$ | $2^{-24}=5.9605\times10^{-8}$ | $9.77\times10^{-4}$ | $4.88\times10^{-4}$ | ±Inf，NaN | 位布局与 $6.55\times10^4$、$6.10\times10^{-5}$、$5.96\times10^{-8}$ 印 `bf16` 表 1；偏置 15 为推算：`fp8` 称 E5M2 相当于少了尾数位的 IEEE 半精度、偏置 15（§3、§3.2），numpy `nexp=5`、最小正规数 $2^{-14}$，故偏置 $=1-(-14)=15$；numpy finfo 其余值一致 |
| bfloat16 | 1/8/7 | 127 | 8 | $3.3895\times10^{38}$ | $2^{-126}=1.1755\times10^{-38}$ | 编码上 $2^{-133}=9.18\times10^{-41}$；`bf16` 表 1 写 “N/A” | $7.81\times10^{-3}$ | $3.91\times10^{-3}$ | ±Inf，NaN（同 FP32 规则）【笔者按截断 FP32 推断】 | 位布局、$3.38\times10^{38}$、$1.17\times10^{-38}$ 印 `bf16` 表 1；其余算 |
| FP8 E5M2 | 1/5/2 | 15 | 3 | 57,344 | $2^{-14}$ | $2^{-16}=1.526\times10^{-5}$ | 0.25 | 0.125 | ±Inf（S.11111.00），NaN（S.11111.{01,10,11}） | 全部印 `fp8` 表 1 |
| FP8 E4M3 | 1/4/3 | 7 | 4 | 448 | $2^{-6}=0.015625$ | $2^{-9}=0.001953$ | 0.125 | 0.0625 | 无 Inf；NaN 仅 S.1111.111（±两个编码） | 全部印 `fp8` 表 1 |
| FP6 E3M2 | 1/3/2 | 3 | 3 | 28 | $2^{-2}=0.25$ | $2^{-4}=0.0625$ | 0.25 | 0.125 | 无 Inf/NaN | 最大值与“无 NaN/Inf”印 `microxcaling`；偏置与其余算【OCP 规范未读】 |
| FP6 E2M3 | 1/2/3 | 1 | 4 | 7.5 | $2^{0}=1$ | $2^{-3}=0.125$ | 0.125 | 0.0625 | 无 Inf/NaN | 同上 |
| FP4 E2M1 | 1/2/1 | 1 | 2 | 6 | $2^{0}=1$ | $2^{-1}=0.5$ | 0.5 | 0.25 | 无 Inf/NaN；全部取值 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} | 取值印 `nvfp4` §2；最大值与“no NaN/Inf”印 `microxcaling` |
| E8M0（UE8M0，缩放） | 0/8/0 | 127 | — | $2^{127}$ | $2^{-127}$ | — | — | — | 无符号、无零、无 Inf；溢出/0xFF 为 NaN【推断】 | “8-bit exponent”印 `mx` §2.2；范围 $2^{-127}$–$2^{127}$ 印 `nvfp4` §2；$[-127,127]$ 与溢出置 NaN 见 `microxcaling` `mx_ops.py` |

配套的块格式（每元素平均位数 = 元素位数 + 缩放位数 / 块大小，**算**）：

| 块格式 | 元素 | 块大小 | 块缩放 | 额外缩放 | 位/元素 | 来源 |
|---|---|---|---|---|---|---|
| MXFP8 | E4M3 或 E5M2 | 32 | E8M0 | 无 | 8.25 | `mx` 表 1 |
| MXFP6 | E2M3 或 E3M2 | 32 | E8M0 | 无 | 6.25 | `mx` 表 1 |
| MXFP4 | E2M1 | 32 | E8M0 | 无 | 4.25 | `mx` 表 1；`nvfp4` §2 |
| MXINT8 | INT8 | 32 | E8M0 | 无 | 8.25 | `mx` 表 1 |
| NVFP4 | E2M1 | 16 | E4M3 | 每张量 FP32 | 4.5（+ 可忽略的 32 位/张量） | `nvfp4` §2、附录 B |
| V4.1 FP4 KV cache | E2M1 | 16（通道） | E4M3 | **无** | 4.5 | `dsv41` §2.4.4 |
| V4 indexer Q/K | MXFP4 | 32 | E8M0 | 无 | 4.25 | `dsv41` §2.4.4 对 V4 的回顾 |
| DeepSeek-V3 FP8（激活 / 权重） | E4M3 | 1×128 / 128×128 | FP32 | — | 8.25 / 约 8.002 | 【交叉引用】lesson-12-doc C41；位数【算】 |
| DeepSeek-V3.1 UE8M0 缩放 | — | — | UE8M0 | — | — | 【交叉引用】lesson-15-doc（本文未重核） |

### 3.2 优化器超参数总表（M07）

| 设置 | $\eta$（lr） | $\beta_1$ | $\beta_2$ | $\epsilon$ | 权重衰减 | 动量 / NS | 来源 |
|---|---|---|---|---|---|---|---|
| Adam 论文 v9 默认 | 0.001 | 0.9 | 0.999 | $10^{-8}$ | — | — | `adam` 算法 1 |
| Adam 论文 v1 默认（旧参数化） | 0.0002 | 0.1（=新 0.9） | 0.001（=新 0.999） | $10^{-8}$ | — | — | `adam` v1 算法 1 |
| AdamW 论文算法 2 | 0.001 | 0.9 | 0.999 | $10^{-8}$ | $\lambda$（乘调度乘子 $\eta_t$） | — | `adamw` 算法 2 |
| PyTorch `AdamW` 默认 | 1e-3 | 0.9 | 0.999 | 1e-8 | 1e-2（乘 lr） | — | `pytorch-adamw` |
| DeepSeek-V4-Flash AdamW | 峰值 $2.7\times10^{-4}$ | 0.9 | 0.95 | $10^{-20}$ | 0.1 | — | `dsv4` §4.2.2 |
| DeepSeek-V4.1-Flash AdamW | 峰值 $2.6\times10^{-4}$ | 0.9 | 0.95 | $10^{-20}$ | 0.1 | — | `dsv41` §4.2.2 |
| Muon（Jordan 代码默认） | 0.02 | — | — | — | 0（AdamW 式，乘 lr） | 动量 0.95，Nesterov，NS 5 步 $(3.4445,-4.7750,2.0315)$，缩放 $\sqrt{\max(1,\text{rows}/\text{cols})}$ | `muon-code` |
| Muon（Moonlight） | 与 AdamW 共用 | — | — | — | 与 AdamW 共用（Moonlight 训练 0.1） | 动量 0.95，Nesterov，NS 5 步，缩放 $0.2\sqrt{\max(A,B)}$ | `moonlight` §2.2、§4 |
| Muon（DeepSeek-V4/V4.1） | 与 AdamW 共用 | — | — | — | 0.1 | 动量 0.95，Nesterov，混合 NS 8+2 步，RMS 0.18 | `dsv4` §2.4、§4.2.2；`dsv41` §4.2.2 |
| Sinkhorn 平衡更新（V4.1） | $0.18\eta_t$ | — | — | $10^{-20}$ | 无 | 动量 0.95（Nesterov），$K=11$，$\tau=10^{-3}$ | `dsv41` §2.5、§4.2.2 |
| 经典动量（Polyak 最优，二次函数） | — | — | — | — | — | $\mu^*=(\sqrt R-1)/(\sqrt R+1)$ | `sutskever2013` §2 |

各模型哪些参数用哪个优化器：

| 模型 | Muon | AdamW | 其他 |
|---|---|---|---|
| Jordan 建议 | 隐藏层 2D 权重（卷积展平） | 嵌入、输出头、标量与向量参数 | — |
| DeepSeek-V4 | 其余全部模块 | 嵌入、预测头、mHC 静态偏置与门控因子、全部 RMSNorm 权重 | — |
| DeepSeek-V4.1 | 线性变换（Q/K 用 head-wise）、Engram 投影、视觉–语言投影 | RMSNorm 与其他非矩阵参数 | 嵌入、预测头、Engram 表：Sinkhorn 平衡更新 |

### 3.3 学习率调度与 batch（M07）

| 调度 | warmup | 稳定期 | 衰减 | 来源 |
|---|---|---|---|---|
| Goyal ImageNet | 5 epoch 线性，从 0.1 到 $0.1\cdot kn/256$ | — | 原调度（阶梯） | `goyal2017` §2.2、§5 |
| MiniCPM WSD | $W$ 步线性（影响小，常省略） | 峰值 0.01（MiniCPM 稳定期，batch 3.93M） | 指数退火，半衰期 5000 步（20B tokens）；约 10% token 足够 | `minicpm` §4.2、§4.3、§6.2 |
| DeepSeek LLM 多步 | 2000 步 | 到 80% token | 80% 处 ×0.316，90% 处 ×0.1 | 【交叉引用】lesson-06-doc 第 12 条 |
| DeepSeek-V4-Flash | 2000 步线性 | $2.7\times10^{-4}$，“大部分时间” | 末尾 cosine 到 $2.7\times10^{-5}$；batch 升到 75.5M tokens 后保持；共 32T tokens | `dsv4` §4.2.2 |
| DeepSeek-V4.1-Flash | 2000 步线性 | $2.6\times10^{-4}$ 到 28T tokens | 28T–40T cosine 到 $2.6\times10^{-5}$，40T–45T 保持；batch 固定 100.6M tokens | `dsv41` §4.2.2 |

### 3.4 集合通信代价（M10，`thakur2005` 模型，$p$ 进程、$n$ 字节）

| 操作 / 算法 | 延迟项 | 带宽项 | 计算项 | 位置 |
|---|---|---|---|---|
| 点对点 | $\alpha$ | $n\beta$ | — | §3 |
| allgather，环 | $(p-1)\alpha$ | $\frac{p-1}pn\beta$ | — | §4.1 |
| allgather，递归倍增 | $\lg p\,\alpha$ | $\frac{p-1}pn\beta$ | — | §4.1.1 |
| all-to-all，长消息成对交换 | $(p-1)\alpha$ | $n\beta$ | — | §4.3 |
| reduce，二项树 | $\lceil\lg p\rceil\alpha$ | $\lceil\lg p\rceil n\beta$ | $\lceil\lg p\rceil n\gamma$ | §4.5 |
| reduce，Rabenseifner | $2\lg p\,\alpha$ | $2\frac{p-1}pn\beta$ | $\frac{p-1}pn\gamma$ | §4.5 |
| allreduce，减半倍增（$p$ 为 2 的幂） | $2\lg p\,\alpha$ | $2\frac{p-1}pn\beta$ | $\frac{p-1}pn\gamma$ | §5.1（原文写成 $2n\beta+n\gamma-\frac1p(\cdot)$） |
| allreduce，“环”（MPICH 版） | $2(p-1)\alpha$ | $2\frac{p-1}pn\beta$ | $\frac{p-1}pn\gamma$ | §5.3（reduce-scatter 用的是成对交换，allgather 才是环；代价与纯环算法相同） |
| allreduce 带宽下界 | — | 至少一个进程 $\ge2\frac{p-1}pn$ 字节 | — | `patarasuk2009` 引理 4（“存在一个进程”，不是每个进程） |

### 3.5 mHC / Sinkhorn 设置（M11）

| 项 | 值 | 来源 |
|---|---|---|
| 扩展率 $n$（$n_{hc}$） | 4 | `mhc` §5.1；`dsv4` §4.2.1；`dsv41` §4.2.1 |
| Sinkhorn 迭代次数 $t_{max}$ | 20 | `mhc` §4.2；`dsv4` §2.2、§4.2.1；`dsv41` §4.2.1 |
| 迭代顺序 | $\mathcal T_r(\mathcal T_c(\cdot))$：先列后行 | `mhc` 式 (9)；`dsv4` 式 (8) |
| 正性保证 | $\exp(\tilde{\mathcal H}^{res})$ | 同上 |
| $\mathcal H^{pre}$、$\mathcal H^{post}$ | $\sigma(\cdot)$、$2\sigma(\cdot)$ | `mhc` 式 (8) |
| 训练时间开销 | 6.7%（$n=4$） | `mhc` §4.3 |
| V4.1 Sinkhorn 平衡（优化器） | $K=11$ 步行列 $\ell_2$ 归一化，$\tau=10^{-3}$，$\varepsilon=10^{-20}$，$\gamma=0.18$ | `dsv41` §2.5、§4.2.2 |

### 3.6 GAE 与策略梯度的常用取值（M09）

| 项 | 值 | 来源 |
|---|---|---|
| GAE 最佳区间（cart-pole） | $\gamma\in[0.96,0.99]$，$\lambda\in[0.92,0.99]$ | `gae` 图 2 附近 |
| GAE 一般建议 | $\lambda\in[0.9,0.99]$ | `gae` §7 |
| 3D 运动任务 | $\gamma=0.995$，$\lambda\in\{0,0.96\}$ 对比，0.96 最佳（四足） | `gae` §6.3 |
| InstructGPT PPO | $\gamma=1$，$\lambda=0.95$ | 【交叉引用】lesson-03-doc C35 |
| PPO 裁剪 | $\varepsilon=0.2$（论文举例） | 【交叉引用】lesson-08-doc C38 |

---

## 4. 复算的数字

全部在本机 Python 3.11.8 + numpy 1.26.4 上算出；表达式照抄，读者可以逐条粘贴复现。浮点格式的舍入用“就近、平局取偶”（bf16 用位运算 `(a + 0x7FFF + lsb) & 0xFFFF0000`；FP8/FP4 用枚举全部可表示值后取最近）。

### 4.1 数值格式（M08）

| 编号 | 量 | 表达式 | 值 |
|---|---|---|---|
| N1 | FP32 最大正规数 | `(2-2**-23)*2**127` | 3.4028e38 |
| N2 | BF16 最大正规数 | `(2-2**-7)*2**127` | 3.3895e38（`bf16` 表 1 印 3.38e38，截断而非四舍五入） |
| N3 | FP16 最大正规数 | `(2-2**-10)*2**15` | 65504（`bf16` 表 1 印 6.55e4） |
| N4 | E4M3 最大值 / 若按 IEEE 保留顶码 | `1.75*2**8`；`1.875*2**7` | 448 / 240（`fp8` §3.1 印 240）。IEEE 式布局下最大正规数是指数码 1110、尾数 111，即 $(2-2^{-3})\times2^7=1.875\times2^7$；旧版此处写的 `1.75*2**7` 等于 224，是错的（核查记录 K1） |
| N5 | E5M2 最大值 | `1.75*2**15` | 57344 |
| N6 | E4M3 顶部 8 个值 | 枚举 | 240, 256, 288, 320, 352, 384, 416, 448（256–448 是 E4M3 比 IEEE 版多出的 7 个） |
| N7 | 编码计数 | E4M3：有限值 254 个编码 + NaN 2 个 = 256；E5M2：有限 248 + Inf 2 + NaN 6 = 256 | 核对通过 |
| N8 | binade 数（按 2 的幂个数计） | E4M3 $2^{-9}..2^{8}$；IEEE 版 $2^{-9}..2^{7}$；E5M2 $2^{-16}..2^{15}$ | 18 / 17 / 32（与 `fp8` §3.1 一致）；$\log_2(\max/\min_{sub})$=17.81 / 16.91 / 31.81 |
| N9 | 0.1 的各格式表示与相对误差 | `np.float32(0.1)`、`np.float16(0.1)`、bf16 位运算、E4M3/E5M2 最近值 | FP32 0.10000000149（1.5e-8）；FP16 0.0999756（2.44e-4）；BF16 0.10009766（9.77e-4）；E4M3 0.1015625（1.56%）；E5M2 0.09375（6.25%） |
| N10 | 1 附近与各量级的 ulp | $2^{e-M}$ | BF16：1→0.0078125，4→0.03125，256→2，1024→8；FP16：1→9.77e-4，1024→1；FP32：1→1.19e-7 |
| N11 | 吞没（swamping） | `np.float16(1024)+np.float16(1)` 等 | FP16：1024+1=1025，2048+1=2048；BF16：256+1=256，1024+1=1024；FP32：$2^{24}+1=2^{24}$ |
| N12 | 顺序累加 10000 个 1 | 逐次相加并舍入到该格式 | BF16 停在 256；FP16 停在 2048 |
| N13 | 4096 个 U(0,0.01) 用 BF16 累加 | seed 0，逐次 bf16 舍入 | 精确和 20.333，BF16 累加得 4.0（相对误差 80.3%）；在 4 处 ulp=1/32，半 ulp 0.0156 > 单个加数上限 0.01，于是之后全部被舍掉 |
| N14 | $\epsilon$ 的两种约定 | numpy `finfo(float32).eps`、`epsneg` | 1.1920929e-07 与 5.9604645e-08（= $2^{-23}$ 与 $2^{-24}$）；float16：9.77e-4 与 4.88e-4 |
| N15 | $10^{-20}$ 在各格式 | `np.float16(1e-20)`、`np.float32(1e-20)` | FP16 下为 0（小于最小次正规 5.96e-8）；FP32 下为 9.9999997e-21（V4 的 AdamW $\varepsilon=10^{-20}$ 只能在 FP32/BF16 状态下起作用） |
| N16 | 每元素位数 | `4+8/32`，`4+8/16`，`8+8/32`，`8+32/128`，`8+32/(128*128)` | MXFP4 4.25；NVFP4 4.5；MXFP8 8.25；V3 激活 8.25；V3 权重 8.00195 |
| N17 | NVFP4 / V4.1 KV 的幅值上限 | `6*448` | 2688（`dsv41` §2.4.4 印 “448×6=2688”；`nvfp4` 附录 B.1 只印 $s_{enc}=6\cdot448/amax_x$，没有写出乘积 2688，见核查记录 K3） |
| N18 | V4.1 KV 幅值界 | `sqrt(512)` | 22.63（`dsv41` 印 “≈22.6”） |
| N19 | MX（OCP 算法 1）对 FP4 的钳位 | $e=\lfloor\log_2 amax\rfloor-2$，$X=2^e$ | amax=7：$e=0$，7/1=7>6 → 存成 6（−14.3%）；amax=0.9：$e=-3$，$X=0.125$，7.2 → 6 → 0.75（−16.7%）；amax=6：精确；**块最大值的尾数 >1.5 时必钳位** |
| N20 | 缩放向上取整（NVIDIA 做法）的浪费 | $s=2^{\lceil\log_2(amax/6)\rceil}$ | amax=7：$s=2$，可用格点 ≤7 为 {0,1,2,3,4,6}；amax=3.01：$s=1$，可用 {0,0.5,1,1.5,2,3}（与 `nvfp4` B.4 的 $3+\delta$ 例一致） |
| N21 | NVFP4（忽略全局缩放、E4M3 最近舍入）的块缩放 | $s=\mathrm{e4m3}(amax/6)$ | amax=7：$s=1.125$，$7/1.125=6.22$（仍略超 6，饱和）；amax=3.01：$s=0.5$，6.02 |
| N22 | 细粒度缩放的效果（1024 个 N(0,1) + 1 个离群值） | seed 0，最近舍入，统计不含离群值所在块 | E2M1 离群值 50：整张量缩放 96.8% 被冲成 0、中位相对误差 100%；16 元素块缩放 6.2% 冲零、中位误差 10.0%。E4M3 离群值 50：两者中位误差都约 2.1%（E4M3 的 18 个 binade 容得下）；离群值 $10^5$：整张量 16.3% 冲零、中位误差 16.3%，128 元素块 0% 冲零、2.1% |

### 4.2 线性代数与 Sinkhorn（M02、M11）

| 编号 | 量 | 表达式 | 值 |
|---|---|---|---|
| N23 | $A=\mathrm{diag}(3,2,1)$ 的秩 1 近似 | `np.linalg.norm(A-B,2)`、`'fro'` | $B=\mathrm{diag}(3,0,0)$：谱 2，F $\sqrt5=2.236$；$B=\mathrm{diag}(2.5,0,0)$、$(3.5,0,0)$、$(4,0,0)$：谱都等于 2，F 为 2.291、2.291、2.449 → **谱范数极小元不唯一，F 范数唯一** |
| N24 | 随机 $6\times4$ 矩阵 `np.random.default_rng(1).standard_normal((6,4))` | SVD 截断 | $\sigma$=(2.0769, 1.6715, 1.4254, 0.5616)；$k=1,2,3$ 时 $\|M-A_k\|_2$=1.6715、1.4254、0.5616（=$\sigma_{k+1}$），$\|M-A_k\|_F$=2.2674、1.5321、0.5616（=尾部平方和开方） |
| N25 | 2 万个随机秩 2 竞争者（给定 $X$ 时 $Y$ 取最小二乘最优） | 同上 | 最好 1.6860 > Eckart–Young 1.5321 |
| N26 | MLA 式低秩的参数账（V2 维度，【交叉引用】lesson-09-doc） | `16384*5120`；`512*5120+16384*512` | 满秩 $W^K$ 83,886,080；$W^{DKV}+W^{UK}$ 11,010,048；比 7.62 |
| N27 | $\mathcal B_4$ 的维数、顶点、Carathéodory 界 | `(n-1)**2`，`n!`，`(n-1)**2+1` | 9，24，10 |
| N28 | 24 个置换阵的随机凸组合 | Dirichlet 权重 | 行和、列和全为 1，$\|D\|_2=1.0$，$\sqrt{\|D\|_1\|D\|_\infty}=1.0$；两个这样的矩阵之积仍双随机 |
| N29 | 只行随机不够 | 全部行为 $\mathbf e_1^\top$ 的 $4\times4$ | $\|R\|_2=2=\sqrt4$；$\|R^{10}\|_2=2$ |
| N30 | 无约束混合矩阵的连乘 | $H=\begin{pmatrix}1.2&0.3\\0.1&0.9\end{pmatrix}$ | $\|H\|_2=1.3048$，谱半径 1.2791，$\|H^{60}\|_2=2.84\times10^6$（HC 式的爆炸；对比 mHC 图 3 的“近 3000”） |
| N31 | 循环双随机阵的 Birkhoff 分解 | $D=0.5I+0.3P+0.2P^2$，$P$ 为 3 循环 | $D=\begin{pmatrix}0.5&0.3&0.2\\0.2&0.5&0.3\\0.3&0.2&0.5\end{pmatrix}$；特征值 1、$0.25\pm0.0866i$（模 0.2646）；$\|D\|_2=1$；$\max|D^{10}-J/3|=1.1\times10^{-6}$ |
| N32 | 2×2 正矩阵 $\begin{pmatrix}1&2\\3&4\end{pmatrix}$ 的 Sinkhorn 极限 | 交叉比 $a_{11}a_{22}/(a_{12}a_{21})$ 在对角缩放下不变 | 极限 $\begin{pmatrix}a&1-a\\1-a&a\end{pmatrix}$，$a/(1-a)=\sqrt{4/6}$，$a=0.449490$；1 次迭代后 $\begin{pmatrix}0.42857&0.57143\\0.52941&0.47059\end{pmatrix}$；6 次后列误差 4.7e-12 |
| N33 | 4×4 mHC 式迭代（$\tilde H\sim N(0,1)$，矩阵数值抄在 §5 F11-3，$\exp$ 后先列后行） | 记录 $\max|\text{行和}-1|$ 与 $\max|\text{列和}-1|$ | $t$=1：行 1e-16，列 0.484；$t$=2：0.145；3：0.0417；5：3.25e-3；10：5.31e-6；20：1.42e-11；$\|M^{(20)}\|_2=1.0$ |
| N34 | 同一矩阵 logits 放大 $s$ 倍后 20 次迭代 | $\exp(s\tilde H)$ | $s$=3：列误差 4.18e-3，$\|\cdot\|_2=1.0000138$；$s$=5：0.0494，1.00386；$s$=10：0.0841，1.01430 → **有限步输出的谱范数可以略大于 1** |
| N35 | 有支撑无完全支撑 $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ | 交替归一化 | 右上角：1 次 0.3333，10 次 0.04762，100 次 0.00498，1000 次 0.00050（≈ $1/(2t+1)$，次线性收敛到单位阵；极限不是 $D_1AD_2$ 形式） |

### 4.3 概率、信息论、微积分（M04、M05、M06）

| 编号 | 量 | 表达式 | 值 |
|---|---|---|---|
| N36 | KL 不对称 | $p=(0.5,0.3,0.2)$，$q=(0.4,0.4,0.2)$ | $\mathrm{KL}(p\|q)=0.025267$，$\mathrm{KL}(q\|p)=0.025815$；$p=(0.9,0.1)$、$q=(0.5,0.5)$：0.368064 与 0.510826 |
| N37 | 交叉熵 = 熵 + KL | $p=(0.5,0.3,0.2)$，$q=(0.2,0.3,0.5)$ | $H(p)=1.029653$，$CE(p,q)=1.304540$，差 0.274887 = $\mathrm{KL}(p\|q)$（此例 $p,q$ 互为镜像，两个方向的 KL 恰好相等，**不要**用它讲不对称） |
| N38 | 未归一化的 $Q$ 让“KL”变负 | $\sum p\log(p/\tilde q)$，$\tilde q=(0.6,0.6,0.6)$ | −0.518827 |
| N39 | Jensen（log 凹） | $X\in\{1,4\}$ 等概率 | $\E[\log X]=0.693147$ < $\log\E[X]=0.916291$ |
| N40 | k1/k2/k3 单样本值 | $r=0.5,1,2$ | $r$=0.5：0.6931 / 0.2402 / 0.1931；$r$=1：0/0/0；$r$=2：−0.6931 / 0.2402 / 0.3069（k1 可为负，k3 恒 ≥ 0） |
| N41 | softmax 与交叉熵梯度 | $z=(2,1,0)$，标签 0 | $p=(0.665241,0.244728,0.090031)$，$CE=0.407606$ nats = 0.588051 bits，困惑度 1.503215；$\partial CE/\partial z=p-y=(-0.334759,0.244728,0.090031)$ |
| N42 | softmax 雅可比 | $\mathrm{diag}(p)-pp^\top$ | 对角 (0.222695, 0.184836, 0.081925)，非对角 −0.162803、−0.059892、−0.022033；每行和为 0 |
| N43 | 温度 | $z/T$ | $T$=0.5：(0.8668, 0.1173, 0.0159)；$T$=2：(0.5065, 0.3072, 0.1863) |
| N44 | 点积方差 | 20 万次 $\sum_{i\le128}q_ik_i$，$q,k\sim N(0,1)$ | 样本方差 127.06（理论 128，标准差 $\sqrt{128}=11.31$；$d=64$ 时 8）【交叉引用】lesson-01-doc 的同类数字 |
| N45 | 重要性采样 | 目标 $p=N(1,1)$，提议 $q=N(0,1)$，$w=e^{x-1/2}$，$10^6$ 样本，seed 0 | $\hat\E_p[x]=1.00201$；平均权重 1.00145；ESS/N=0.3697（理论 $e^{-1}=0.3679$） |
| N46 | RoPE 频率（$d=128$，底 10000） | $\theta_i=10000^{-2i/d}$，$i=0..63$ | $\theta_0=1$，$\theta_1=0.86596$，$\theta_{63}=1.1548\times10^{-4}$；波长 $2\pi/\theta$：6.28 到 54,410 token |
| N47 | 旋转只依赖相对位置 | $\mathbf q=(1,2)$，$\mathbf k=(0.5,-1)$，$\theta=0.1$ | $(m,n)$=(5,2)、(13,10)、(103,100) 的 $\langle R(m\theta)\mathbf q,R(n\theta)\mathbf k\rangle$ 都是 −2.024045；$R(30°)R(45°)-R(75°)$ 最大误差约 $10^{-16}$（机器精度量级；具体数值随计算顺序变，本次复算为 1.7e-16） |
| N48 | 注意力打分 FLOPs（V2 形状：$n_h=128$、$d_h=128$，$T=4096$） | `2*T*T*dh*nh` | $QK^\top$ 每层 5.50e11（$AV$ 同量，合计 1.10e12；因果只需约一半） |

### 4.4 优化器与调度（M07）

| 编号 | 量 | 表达式 | 值 |
|---|---|---|---|
| N49 | Adam 偏差修正因子 | $1-\beta_1^t$，$1-\beta_2^t$（0.9, 0.999） | $t$=1：0.1 / 0.001；10：0.6513 / 0.00995；100：0.99997 / 0.0952；1000：1.0 / 0.6323；10000：1.0 / 0.99995 |
| N50 | 高效写法的步长乘子 | $\sqrt{1-\beta_2^t}/(1-\beta_1^t)$ | $t$=1：0.3162；2：0.2353；10：0.1532；100：0.3086；1000：0.7952；10000：0.99998；整数 $t$ 上的最小值在 $t=12$，0.1522 |
| N51 | 常数梯度 $g=10^{-3}$ 的第一步 | 不修正 $m_1/\sqrt{v_1}$；修正后 | 3.1623 与 1.0 → 不做偏差修正，第一步大 $\sqrt{1000}/10=3.16$ 倍 |
| N52 | Adam 有效步长界 | $(1-\beta_1)/\sqrt{1-\beta_2}$ | 3.1623（>1，默认值落在 `adam` §2.1 的第一种情形） |
| N53 | 解耦衰减的累计效果 | `(1-3e-4*0.1)**1e5` | 每步 ×0.99997，10 万步后 ×0.0498（常数学习率、无梯度时） |
| N54 | Adam 里 L2 与解耦衰减的差（模拟） | 单个权重 $w_0=1$，梯度 = $s\cdot N(0,1)$ 纯噪声，lr=1e-3，$\lambda$=0.1，默认 $\beta$，200 步、50 次重复（seed 0），统计平均每步 $w$ 的下降 | $s$=1e-4：L2-in-Adam 9.58e-4，AdamW 9.80e-5；$s$=1e-2：9.54e-4 vs 9.43e-5；$s$=1：9.46e-5 vs 9.96e-5；$s$=10：6.59e-6 vs 9.62e-5 → AdamW 恒约 lr·λ·w=1e-4；L2 版在梯度小时约等于 lr（被 Adam 归一化成“符号步”，大 10 倍），梯度大时被压小（约小一个数量级）——即 `adamw` §2 所说“梯度大的权重被 L2 正则化得少”。**注意 $s=10$ 这一点受随机序列影响很大**：核查时用同样设置、不同的抽样顺序独立复算得 L2 版 1.47e-5（约小 7 倍，而不是 15 倍）；前三个点复算为 9.58e-4、9.52e-4、1.03e-4，与上面一致。写课时只引方向和数量级，不引“15 倍” |
| N55 | Robbins–Monro 条件 | 前 $10^6$ 项部分和 | $\sum1/n=14.3927$（发散，≈ $\ln n+0.5772$），$\sum1/n^2=1.644933$（收敛到 $\pi^2/6$）；$\sum n^{-1/2}=1999$，$\sum(n^{-1/2})^2=14.39$（后者也发散，**$1/\sqrt n$ 不满足第二个条件**）；常数 0.01：$\sum a_n^2=100$ 且随 $n$ 线性增长 |
| N56 | Polyak 最优动量 | $\mu^*=(\sqrt R-1)/(\sqrt R+1)$；到 $10^{-6}$ 的迭代数 $\ln10^{-6}/\ln\rho$ | $R$=10：$\mu^*$=0.5195，GD 69 步 vs 重球 21 步；$R$=100：0.8182，691 vs 69；$R=10^4$：0.9802，69078 vs 691（比值约 $\sqrt R$） |
| N57 | NS 多项式 | $\varphi_J(x)=3.4445x-4.7750x^3+2.0315x^5$ | $\varphi_J(1)=0.7010$；$\varphi_J'(0)=3.4445$；$\max_{[0,1]}\varphi_J=1.2024$（在 $x=0.5545$）；正不动点 0.8680、1.2637；朴素 $\varphi_B=(2,-1.5,0.5)$：$\varphi_B(1)=1$，$\varphi_B'(1)=0$ |
| N58 | NS 迭代后奇异值的范围（输入 $x\in[10^{-3},1]$） | 逐元迭代 | Jordan 系数 5 步：[0.4705, 1.2024]；8 步及以后：[0.6818, 1.1344]。长期行为是一个**吸引 4 周期** 0.6819 → 1.1343 → 0.7529 → 1.0469 → 0.6819（从 0.001、0.01、0.3、0.5、0.9、1.0 出发迭代 2000 步都落到这个周期上），区间端点就是周期里的最小、最大值。旧版写的“2 周期 0.7529 ↔ 1.1343”不对：$\varphi_J(1.1343)=0.7530$，但 $\varphi_J(0.7529)=1.0469$ 而不是 1.1343。$\varphi_J$ 另有一个不稳定 2 周期 {0.7093, 1.1040}（$(\varphi_J^2)'=-1.28$），不动点 0.8680 也不稳定（$\varphi_J'=-1.58$）（核查记录 K2）；输入 ≥0.0015 的奇异值 5 步后都 ≥0.68；V4 混合 8+2 步：[1.000000, 1.000538]（输入 ≥$10^{-4}$ 时上界 1.002017） |
| N59 | 经典三次 NS 的慢 | $f(x)=1.5x-0.5x^3$，从 0.01 起 12 步 | 0.8934（Jordan 五次式 4–5 步即进入 [0.68, 1.2]） |
| N60 | Muon 更新 RMS（引理 1） | 随机高斯阵 SVD 后 $UV^\top$ | [1024,4096]：0.015625 = $1/\sqrt{4096}$；[128,7168]：0.011811 = $1/\sqrt{7168}$ |
| N61 | 三种形状缩放 | Moonlight $0.2\sqrt{\max}$；V4 $0.18\sqrt{\max}$；Jordan 代码 $\sqrt{\max(1,r/c)}$ | [7168,18432]：27.15 / 24.44 / 1.0（Jordan 代码下 RMS 仍为 0.00737）；[7168,7168]：16.93 / 15.24 / 1.0 |
| N62 | Muon 的 FLOP 开销 | 每步 NS $2(2nm^2+m^3)$，基线 $6nmB$ | [7168,18432] 每步 NS 4.525e12；$B$=524,288：5 步 5.44%、10 步 10.89%；$B=1.6\times10^7$：0.18% / 0.36%；$B=7.55\times10^7$（V4 batch）：0.038% / 0.076%；博客的上界 $Tm/B$ 对应 6.84% / 0.224% / 0.047%（T=5） |
| N63 | 博客的两个例子 | `5*768/524288`，`5*16384/16e6` | 0.732%、0.512%（博客写 0.7%、0.5%） |
| N64 | 优化器状态显存（1.6T 参数，FP32 状态） | `1.6e12*8`，`1.6e12*4` | Adam $m+v$ 12.8 TB；Muon 动量 6.4 TB |
| N65 | 线性缩放 | $0.1\cdot kn/256$ | 256→0.1，1024→0.4，8192→3.2；8k batch 每 epoch 156.4 次迭代，5 epoch warmup ≈ 782 次 |
| N66 | 噪声尺度的几个点 | 式 (2.6)(2.7) | $B=\mathcal B_{noise}$：$\epsilon_{opt}=\epsilon_{max}/2$；$B=\mathcal B_{noise}/10$：$\Delta L=\Delta L_{max}/11$；$B=10\mathcal B_{noise}$：$\Delta L=0.909\Delta L_{max}$；$B=\mathcal B_{crit}$ 时 $S=2S_{min}$、$E=2E_{min}$ |
| N67 | MiniCPM 指数退火（按半衰期 5000 步读） | $0.01\times0.5^{u/5000}$ | $u$=0：0.01；1000：0.008706；2500：0.007071；5000：0.005；10000：0.0025 |
| N68 | DeepSeek LLM 多步系数 | `sqrt(0.1)` | 0.316228 |
| N69 | V4.1 调度的比例 | 28/45、(40−28)/45、(45−40)/45 | 稳定 62.2%、衰减 26.7%、末段保持 11.1%（按 token） |

### 4.5 强化学习（M09）

| 编号 | 量 | 表达式 | 值 |
|---|---|---|---|
| N70 | GAE 三步算例 | $r=(0,0,1)$，$V=(0.5,0.6,0.8,0)$，$\gamma=0.99$，$\lambda=0.95$ | $\delta=(0.094,0.192,0.200)$；GAE=(0.451484, 0.3801, 0.2)；GAE($\gamma$,1)=(0.4801, 0.39, 0.2)=MC 回报减 $V$；GAE($\gamma$,0)=$\delta$ |
| N71 | 有效视界 | $1/(1-\gamma\lambda)$，$1/(1-\gamma)$ | 16.81 与 100 |
| N72 | REINFORCE 的基线（Bernoulli，$p=0.3$，奖励 $\mathbb 1[y=1]+10$，$2\times10^5$ 样本） | $(r-b)(y-p)/(p(1-p))$ | $b$=0：均值 0.989、方差 544.97；$b$=10：0.999、2.33；$b$=10.3：0.9996、0.76（真梯度 1） |
| N73 | PPO 裁剪（$\varepsilon=0.2$） | $\min(rA,\mathrm{clip}(r,0.8,1.2)A)$ | $r$=1.3,$A$=1：1.2；$r$=1.3,$A$=−1：−1.3；$r$=0.7,$A$=1：0.7；$r$=0.7,$A$=−1：−0.8 |
| N74 | GRPO 组均值基线含自身的缩放【笔者推导】 | $(G-1)/G$ | $G$=16：0.9375；$G$=64：0.984375 |

### 4.6 算力算术与通信（M10）

| 编号 | 量 | 表达式 | 值 |
|---|---|---|---|
| N75 | GEMM 的 FLOPs 与算术强度（BF16，读 $A,B$ 写 $C$ 各一次） | $2mnk\,/\,2(mk+kn+mn)$ | $(m,n,k)$=(4096,2048,7168)：1.203e11 FLOPs，1146.9 FLOP/B；(1,7168,7168)：1.0；(64,7168,7168)：62.9；方阵 4096：1365.3（=$n/3$） |
| N76 | 屋顶线（A100，【交叉引用】mamba lesson-01-doc：312 TFLOP/s、2.039 TB/s） | $\min(312,2.039I)$ | ridge 153.0 FLOP/B；$I$=1：2.0 TFLOP/s；64：130.5；≥153：312（H100 SXM 用 989/3.35 得 ridge 295） |
| N77 | 环形 allreduce 的每卡发送量与时间（14 GB 梯度） | $2\frac{p-1}pn/\mathrm{BW}$；延迟 $2(p-1)\alpha$，$\alpha=5\,\mu$s【假设】 | $p$=2：1.0n，160 GB/s 下 0.0875 s，50 GB/s 下 0.28 s；$p$=8：1.75n，0.153 s / 0.49 s；$p$=64：1.969n，0.172 / 0.551 s；$p$=1024：1.998n，0.175 / 0.560 s，延迟项 10.2 ms |
| N78 | 朴素“归约到根再广播” vs 环（$p$=8） | 根的流量 $\approx2(p-1)n$ | 14n vs 每卡 1.75n |
| N79 | 延迟步数 | 环 $2(p-1)$，减半倍增 $2\lg p$ | $p$=1024：2046 步 vs 20 步 |
| N80 | MoE all-to-all 每 token 字节（V3 维度 $h=7168$，**示意**：FP8 分发 + 每 128 元素一个 FP32 缩放，BF16 合并） | `7168+56*4`，`7168*2` | 分发每份 7392 B，合并每份 14336 B；8 份：57.8 KiB / 112.0 KiB；4 份（节点受限）：28.9 / 56.0 KiB（DeepEP 实际格式见 lesson-12-doc，本文未核） |

---

## 5. 图表清单与重绘建议

### 5.1 来源里值得改绘的图（清单）

| 来源图 | 内容 | 用在 | 备注 |
|---|---|---|---|
| `jordan-blog` Figure 3 | 朴素系数 $(2,-1.5,0.5)$ 下 $\varphi^N(x)$ 的曲线 | M07 | 自己按公式算，不要截图（数据见 F07-3） |
| `jordan-blog` Figure 4 | 调过的系数 $(3.4445,-4.7750,2.0315)$ 的曲线，0 附近更陡 | M07 | 同上 |
| `mccandlish2018` Figure 1 | 步数–样本数的双曲线权衡 | M07 | 由式 (2.11) 直接画（F07-5） |
| `mccandlish2018` Figure 3 | 大步长 + 小 batch 在二次函数上的不稳定；$\mathcal B_{noise}$ 是“转折点” | M07 | 右半边由式 (2.7) 画 |
| `minicpm` Figure 5/6 | WSD 衰减阶段损失骤降；10% 衰减足够 | M07 | 只能示意（损失数据不公开），学习率曲线本身可精确画（F07-4） |
| `roofline` Figure 1a | 屋顶线（Opteron X2） | M10 | 改用 A100/H800 数字重画（F10-2） |
| `fp8` 表 1 | E4M3 / E5M2 位布局与特殊值 | M08 | 画成位域条（F08-1） |
| `mx` Figure 1 | 一个 MX 块：1 个共享缩放 + $k$ 个元素 | M08 | F08-4 |
| `nvfp4` Figure 1 | 16×32 矩阵的 NVFP4 存储：每 16 个元素一个 E4M3 缩放 | M08 | F08-4 右半 |
| `mhc` Figure 3 | HC 单层与复合映射的增益随层数爆炸 | M11 | 论文数据不公开；用 N30 的 2×2 例子画“同一现象的玩具版” |
| `mhc` Figure 7 | mHC 的前向（行和）/反向（列和）增益接近 1 | M11 | 用 N33/N34 的数值画（F11-3） |
| `thakur2005` Figure 1 | 递归倍增 allgather 的三步 | M10 | F10-3 |
| `patarasuk2009` §4 的环形算法（图号未核） | 环上 reduce-scatter + allgather | M10 | F10-3 |
| `stewart1993` | 无图 | M02 | 只用于历史注记 |

### 5.2 建议本系列画的图（29 张，按课）

每张给出“要看什么”与画图所需的全部数值；几何由 `math/figures/gen/mNN_*.py` 从这些数计算。颜色语义按 style-guide §4。这里只列需要真实数字的“账单图”与数值示意，course-map 要求每课 5–10 张，其余机制示意图由写课者补充。

**M02（4 张）**
- **F02-1 截断 SVD 的误差阶梯。** 横轴 $k=0..4$，纵轴误差；两条折线：谱范数误差 $\sigma_{k+1}$ = (2.0769, 1.6715, 1.4254, 0.5616, 0) 与 F 范数误差 $\sqrt{\sum_{i>k}\sigma_i^2}$ = (3.0748, 2.2674, 1.5321, 0.5616, 0)（N24 的矩阵；$k=0$ 时 F 误差 $=\|M\|_F=\sqrt{\sum\sigma_i^2}$，由 σ 算出 3.0748【算，与直接求 Frobenius 范数一致】）。看点：两种范数下同一个 $A_k$ 都最优，但误差大小不同。
- **F02-2 谱范数极小元不唯一。** $A=\mathrm{diag}(3,2,1)$；横轴 $b\in[1,5]$ 表示 $B=\mathrm{diag}(b,0,0)$；两条曲线：$\|A-B\|_2=\max(|3-b|,2)$（在 $b\in[1,5]$ 上恒为 2 的平台）与 $\|A-B\|_F=\sqrt{(3-b)^2+5}$（在 $b=3$ 处唯一最小 $\sqrt5$）。
- **F02-3 秩的几何：秩 1 矩阵把平面压成一条线。** $2\times2$ 例：$A=\sigma_1\mathbf u_1\mathbf v_1^\top$，$\sigma_1=2$，$\mathbf v_1=(\cos30°,\sin30°)$，$\mathbf u_1=(\cos100°,\sin100°)$；画单位圆上 24 个点的像（全落在 $\mathbf u_1$ 方向的线段 $[-2,2]$ 上）。
- **F02-4 低秩分解的参数账。** 柱状图：$W^K$ 满秩 83.9M vs $W^{DKV}$ 2.6M + $W^{UK}$ 8.4M（N26）；标注 $d=5120$、$d_c=512$、$n_hd_h=16384$。服务第 9 课。

**M03（2 张）**
- **F03-1 旋转的复合 = 角度相加。** 单位圆上 $e^{i30°}$、$e^{i45°}$ 及乘积 $e^{i75°}=(0.2588,0.9659)$；另画 $R(30°)$、$R(45°)$ 作用在 $(1,0)$ 上的两次旋转轨迹。
- **F03-2 RoPE 的多尺度时钟。** $d=128$、底 10000：横轴频率编号 $i=0..63$（对数纵轴），$\theta_i=10000^{-2i/128}$ 从 1 到 $1.1548\times10^{-4}$；右轴波长 $2\pi/\theta_i$ 从 6.28 到 54,410（N46）。

**M04（2 张）**
- **F04-1 点积的方差随维数线性增长。** 横轴 $d\in\{16,32,64,128,256\}$，纵轴标准差 $\sqrt d$=(4, 5.66, 8, 11.31, 16)；叠加除以 $\sqrt d$ 后恒为 1 的水平线。
- **F04-2 重要性权重的退化。** $p=N(\mu,1)$、$q=N(0,1)$，ESS/N 理论值 $e^{-\mu^2}$：$\mu$=0, 0.5, 1, 1.5, 2 → 1, 0.7788, 0.3679, 0.1054, 0.0183（$\mu=1$ 的蒙特卡洛 0.3697 作为一个点，N45）。【笔者推导：$\E_q[w^2]=e^{\mu^2}$】

**M05（2 张）**
- **F05-1 KL 不对称。** 两点分布 $p=(a,1-a)$ 对 $q=(0.5,0.5)$：横轴 $a\in(0,1)$，两条曲线 $\mathrm{KL}(p\|q)$ 与 $\mathrm{KL}(q\|p)$，标出 $a=0.9$ 处 0.3681 与 0.5108（N36）。
- **F05-2 k1、k2、k3 作为 $r$ 的函数。** 横轴 $r\in[0.1,4]$（对数），三条曲线 $-\log r$、$\frac12\log^2r$、$(r-1)-\log r$；标 $r=0.5,1,2$ 的点（N40）。看点：k1 可为负，k3 是切线与 log 的竖直距离。

**M06（2 张）**
- **F06-1 softmax 雅可比热力图。** $z=(2,1,0)$ 的 $3\times3$ 矩阵（N42），行和为 0。
- **F06-2 残差的梯度高速公路。** 20 层，每层雅可比谱范数 0.9（无残差）vs $I+0.1J$（有残差）：梯度范数 $0.9^{l}$ 与 $\approx1$ 的对比（$0.9^{20}=0.1216$【算：`0.9**20`】）。服务第 16 课。

**M07（5 张）**
- **F07-1 Adam 偏差修正因子。** 横轴 $t=1..10^4$（对数），曲线 $1-0.9^t$ 与 $1-0.999^t$（N49），以及步长乘子 $\sqrt{1-\beta_2^t}/(1-\beta_1^t)$（N50，非单调：最低点在 $t=12$，值 0.1522；$t=10$ 处为 0.1532。旧版写“$t=10$ 处最低”不准，见核查记录 K5）。
- **F07-2 L2 与解耦衰减。** 横轴梯度噪声尺度 $s\in\{10^{-4},10^{-2},1,10\}$（对数），纵轴平均每步收缩：L2-in-Adam（9.58e-4, 9.54e-4, 9.46e-5, 6.59e-6）与 AdamW（约 1e-4 的水平线）（N54 的模拟值；最右一点对随机序列敏感，两次独立模拟分别得 6.59e-6 与 1.47e-5，图上只标数量级，见 N54）。
- **F07-3 Newton–Schulz 的奇异值映射。** 横轴 $x\in[0,1]$；画 $\varphi_J^N(x)$，$N=1..5$，以及 V4 混合 8+2 步的结果；标注 $\varphi_J(1)=0.701$、峰值 1.2024 @0.5545、极限带 [0.6818, 1.1344]、博客目标带 [0.7,1.3]、代码注释带 [0.5,1.5]（N57–N58）。对照画朴素 $(2,-1.5,0.5)$ 与经典三次 $(1.5,-0.5)$。
- **F07-4 学习率调度对比。** 同一横轴（归一化训练进度 0–1）：cosine、WSD（MiniCPM：10% 衰减，指数半衰期 5000 步）、DeepSeek LLM 多步（80% 处 ×0.316，90% 处 ×0.1）、V4.1-Flash（token 坐标：0–28T 为 $2.6\times10^{-4}$，28T–40T cosine 到 $2.6\times10^{-5}$，40T–45T 平）。warmup 在这个尺度上不可见，另画插图。
- **F07-5 噪声尺度的双曲线。** 式 (2.11)：$S/S_{min}=1+1/(E/E_{min}-1)$，$E/E_{min}\in(1,6]$；标 $B=\mathcal B_{crit}$ 的点 (2,2)；另一子图 $\Delta L_{opt}/\Delta L_{max}=1/(1+\mathcal B_{noise}/B)$，横轴 $B/\mathcal B_{noise}\in[10^{-2},10^2]$（对数）。

**M08（4 张）**
- **F08-1 位布局条。** FP32(1/8/23)、BF16(1/8/7)、FP16(1/5/10)、E5M2(1/5/2)、E4M3(1/4/3)、E2M1(1/2/1)、E8M0(0/8/0) 的位域，按位数等比例；标偏置与最大值（§3.1）。
- **F08-2 数轴上的可表示值。** E4M3 全部 127 个非负有限值与 E2M1 的 8 个值，对数横轴 $[2^{-9},448]$；标出 240 以上 7 个“借来的”值（N6）。
- **F08-3 吞没与累加误差。** 左：BF16 顺序累加 1 的轨迹在 256 停住、FP16 在 2048 停住（N12）；右：4096 个 U(0,0.01) 的 BF16 累加 vs FP32 累加（最终 4.0 vs 20.33，N13）。服务第 12 课“提升累加精度”。
- **F08-4 块缩放。** 左：一个 MXFP4 块（32 元素 + 1 个 E8M0）；中：NVFP4（16 元素 + 1 个 E4M3 + 张量级 FP32）；右：N22 的对比柱——E2M1 整张量缩放 96.8% 冲零 vs 16 元素块 6.2%。服务第 12、19 课。

**M09（2 张）**
- **F09-1 GAE 的权重。** 横轴 $l=0..60$，$\delta_{t+l}$ 的权重 $(\gamma\lambda)^l$：$(\gamma,\lambda)$=(0.99,0.95)、(0.99,1)、(0.99,0)、(1,0.95)；有效视界 16.8 与 100（N71）。
- **F09-2 基线降低方差。** 三根柱：$b$=0、10、10.3 时估计的方差 544.97、2.33、0.76，均值都 ≈1（N72）。服务第 8 课的组均值基线。

**M10（3 张）**
- **F10-1 GEMM 的算术强度随 $m$ 变化。** $n=k=7168$，$m$ 从 1 到 4096（对数），强度 $2mnk/(2(mk+kn+mn))$：$m$=1 → 1.0，64 → 62.9，4096 → 1911.5【算：`2*4096*7168*7168/(2*(4096*7168+7168*7168+4096*7168))`】；叠加 A100 ridge 153 的竖线。
- **F10-2 屋顶线。** A100：$\min(312,2.039I)$，$I\in[0.25,2048]$ 对数轴；标 $I$=1、64、153、1365 四个点（N76）。
- **F10-3 环形 allreduce。** $p=4$ 的 reduce-scatter（3 步）+ allgather（3 步），每步每卡传 $n/4$；右侧曲线：每卡流量 $2(p-1)/p$ 随 $p$ 趋于 2（N77）。

**M11（3 张）**
- **F11-1 Birkhoff 多面体（3×3）。** $\mathcal B_3$ 维数 4，6 个顶点；投影示意 + N31 的 $D=0.5I+0.3P+0.2P^2$ 作为内部点。
- **F11-2 连乘的增益：无约束 vs 双随机。** 纵轴对数：$\|H^L\|_2$（N30，$L=60$ 时 $2.84\times10^6$）vs $\|D^L\|_2\equiv1$；标 mHC 图 3 的“近 3000”作文字注记。
- **F11-3 Sinkhorn 迭代的收敛。** 用 N33 的矩阵：
  $\tilde H=\begin{pmatrix}-0.88445&-0.25963&-0.23522&-0.00985\\0.42466&-0.11329&-0.84359&-0.63865\\-0.62088&-0.48862&-1.06303&0.55300\\0.19527&0.90549&3.10495&2.35742\end{pmatrix}$，
  20 次后 $M^{(20)}=\begin{pmatrix}0.19008&0.34250&0.22246&0.24496\\0.52059&0.29325&0.08955&0.09661\\0.23628&0.26015&0.09284&0.41072\\0.05304&0.10410&0.59515&0.24771\end{pmatrix}$；
  纵轴对数的列误差：$t$=1,2,3,5,10,20 → 0.484, 0.145, 0.0417, 3.25e-3, 5.31e-6, 1.42e-11；叠加 logits 放大 3、5、10 倍时 20 次后的列误差 4.18e-3、0.0494、0.0841（N34），以及 $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ 的 $1/(2t+1)$ 慢收敛（N35）。

---

## 6. 算例与练习素材

每条：题目 → 所需数字 → 答案（数字都来自 §4）→ 服务的 DeepSeek 课。类型标记：〔概〕概念题、〔算〕计算题、〔推〕推导题。

### M01 矩阵与形状

- **W1〔算〕注意力打分的 FLOPs。** V2 形状 $n_h=128$、$d_h=128$，序列 $T=4096$：一层 $QK^\top$ 要多少 FLOPs？$AV$ 呢？
  答：每头 $2T^2d_h$，全部头 $2\cdot4096^2\cdot128\cdot128=5.50\times10^{11}$；$AV$ 同量，合计 $1.10\times10^{12}$；因果掩码下有效约一半（N48）。→ 第 4、9、14 课。
- **W2〔算〕同一个矩阵乘在 decode 与 prefill 下的算术强度。** $W\in\R^{7168\times7168}$（BF16），输入 $m$ 个 token。$m=1$、64、4096 时强度各是多少？答：1.0、62.9、1911.5 FLOP/B（N75、F10-1）。
  追问：A100 的 ridge 是 153，哪些情况受带宽限制？答：$m=1$ 与 $m=64$。→ 第 4 课。

### M02 低秩与 SVD

- **W3〔算〕截断 SVD 的误差。** $\sigma=(2.0769,1.6715,1.4254,0.5616)$，求秩 2 近似的谱误差与 F 误差。答：$\sigma_3=1.4254$；$\sqrt{1.4254^2+0.5616^2}=1.5321$（N24）。→ 第 9 课。
- **W4〔概〕谱范数下最优秩 1 近似唯一吗？** $A=\mathrm{diag}(3,2,1)$，比较 $B=\mathrm{diag}(3,0,0)$ 与 $\mathrm{diag}(3.5,0,0)$。答：谱误差都是 2，都是极小元；F 误差 2.236 与 2.291，只有前者在 F 意义下最优（N23）。
  这说明谱范数版本的 “argmin” 不能写成唯一的（§7 E2）。
- **W5〔概〕这个定理叫什么名字、谁证明的？** 答：Frobenius 版本最早是 Schmidt 1907，Eckart–Young 1936 重新发现；谱范数与一切酉不变范数是 Mirsky 1960（C4–C6）。
  写作建议：正文称“Eckart–Young–Mirsky 定理”并加一句历史注（Schmidt）。→ 第 9、18 课。
- **W6〔算〕低秩分解省多少参数。** $d=5120$、$d_c=512$、$n_hd_h=16384$：满秩 $W^K$ 与 $W^{UK}W^{DKV}$ 的参数量之比？答：83,886,080 / 11,010,048 = 7.62（N26）。追问：$W^{UK}W^{DKV}$ 的秩至多多少？答：512。→ 第 9 课。

### M03 旋转与复数

- **W7〔算〕旋转复合。** 验证 $e^{i30°}e^{i45°}=e^{i75°}=(0.2588,0.9659)$，并用矩阵乘法验证 $R(30°)R(45°)=R(75°)$（误差约 $10^{-16}$，即浮点舍入，N47）。→ 第 5 课。
- **W8〔推〕相对位置性质。** 证明 $\langle R(m\theta)\mathbf q,R(n\theta)\mathbf k\rangle=\mathbf q^\top R((n-m)\theta)\mathbf k$，并用 $\mathbf q=(1,2)$、$\mathbf k=(0.5,-1)$、$\theta=0.1$ 验证 $(5,2)$、$(13,10)$、$(103,100)$ 三对的值都是 −2.024045（N47）。→ 第 5 课。
- **W9〔算〕RoPE 最慢的时钟。** $d=128$、底 10000：最低频 $\theta_{63}$ 与其波长？答：$1.1548\times10^{-4}$ rad/token，波长 54,410 token（N46）——4K 训练长度只转了约 7.5%【算：4096/54410】。→ 第 5 课（上下文扩展的动机）。

### M04 概率与方差

- **W10〔算〕点积方差。** $q_i,k_i$ 独立标准正态，$d=128$，$\mathbf q\cdot\mathbf k$ 的标准差？答：$\sqrt{128}=11.31$；20 万次模拟得方差 127.06（N44）。→ 第 1 课。
- **W11〔算〕重要性采样的代价。** 用 $q=N(0,1)$ 的样本估计 $p=N(1,1)$ 下的 $\E[x]$：权重 $w=e^{x-1/2}$，有效样本比例？答：理论 $e^{-1}=0.368$，模拟 0.370，估计值 1.002（N45）。
  追问：均值差 2 时呢？答：$e^{-4}=0.018$（F04-2）。→ 第 8、15 课（off-policy 与重要性比）。

### M05 熵与 KL

- **W12〔算〕交叉熵、困惑度与梯度。** logits $(2,1,0)$、正确类 0：交叉熵、困惑度、对 logits 的梯度？答：0.4076 nats（0.5881 bits），困惑度 1.5032，梯度 $(-0.3348,0.2447,0.0900)$（N41）。→ 第 2 课。
- **W13〔算〕KL 不对称。** $p=(0.9,0.1)$、$q=(0.5,0.5)$：$\mathrm{KL}(p\|q)$ 与 $\mathrm{KL}(q\|p)$？答：0.3681 与 0.5108（N36）。→ 第 3、8、14 课（前向 / 反向 KL）。
- **W14〔推〕Gibbs 不等式为什么要求 $Q$ 归一化？** 用 Jensen 推 $\mathrm{KL}\ge0$，指出哪一步用了 $\sum Q=1$；给出反例 $\tilde q=(0.6,0.6,0.6)$ 时 $\sum p\log(p/\tilde q)=-0.519$（N38）。→ 第 8 课（k1 单样本为负但期望非负）。
- **W15〔算〕三个 KL 估计器。** $r=0.5$ 与 $r=2$ 时 k1、k2、k3？答：0.6931/0.2402/0.1931 与 −0.6931/0.2402/0.3069（N40）；并引用 `joschu-kl` 的实验表（C21）说明 k2 有偏、k3 无偏。→ 第 8 课。

### M06 反向传播

- **W16〔推〕softmax 雅可比的行和为 0。** 由 $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$ 证明，并用 $z=(2,1,0)$ 的数值矩阵验证（N42）。解释：logits 同加一个常数，softmax 不变。→ 第 1、2 课。
- **W17〔算〕残差的梯度高速公路。** 20 层每层把梯度乘 0.9：剩多少？答：$0.9^{20}=0.1216$；加恒等残差后每层雅可比 $I+J_f$，梯度至少保留恒等通路（F06-2）。→ 第 16 课。

### M07 优化器

- **W18〔算〕偏差修正有多重要。** 常数梯度 $10^{-3}$，默认 $\beta$：第一步不修正的 $m_1/\sqrt{v_1}$ 是多少？答：3.162，修正后恰为 1（N51）；第 1000 步 $1-\beta_2^t$ 仍只有 0.632（N49）。→ 第 2 课。
- **W19〔概〕AdamW 的“0.1”是多少？** 同样写 `weight_decay=0.1`，PyTorch 每步衰减 $\eta_t\cdot0.1$，论文算法 2 每步衰减 $\eta_t^{\text{(sched)}}\cdot\lambda$（不乘 $\alpha$）。lr=3e-4 时 PyTorch 每步 ×0.99997，10 万步 ×0.0498（N53）。→ 第 2、6、18 课。
- **W20〔推〕Muon 的更新 RMS。** 证明满秩 $[A,B]$ 的 $UV^\top$ RMS $=1/\sqrt{\max(A,B)}$，并算 [7168,18432] 时 Moonlight 与 V4 的缩放因子（27.15 与 24.44）以及缩放后的 RMS（0.2 与 0.18）（N60–N61）。→ 第 18 课。
- **W21〔算〕Newton–Schulz 不收敛到 1。** 求 $\varphi_J(1)$，并说明 5 步后奇异值落在哪个区间；V4 为什么再加 2 步 $(2,-1.5,0.5)$？答：$\varphi_J(1)=0.701$；输入奇异值 $\ge10^{-3}$ 时 5 步后落在 $[0.47,1.20]$，长期在 $[0.68,1.13]$ 振荡（4 周期 0.682→1.134→0.753→1.047→0.682）；$(2,-1.5,0.5)$ 满足 $\varphi(1)=1$、$\varphi'(1)=0$，两步后收到 $[1.0000,1.0005]$（N57–N58）。→ 第 18 课。
- **W22〔算〕Muon 的额外 FLOPs。** [7168,18432] 权重、V4 的 75.5M token batch、10 步 NS：开销占比？答：0.076%（上界 $Tm/B=0.095\%$）；NanoGPT 设置下 5 步约 5.4%（N62）。→ 第 18 课。
- **W23〔算〕线性缩放。** batch 256 时 lr 0.1，batch 8192 时？答：3.2；每 epoch 约 156 步，5 epoch warmup 约 782 步（N65）。→ 第 6 课（DeepSeek LLM 的 batch/lr scaling law 对照）。
- **W24〔算〕噪声尺度。** $B=\mathcal B_{noise}/10$ 与 $10\mathcal B_{noise}$ 时每步的损失改进是最大值的几分之几？答：1/11 与 10/11；$B=\mathcal B_{crit}$ 时步数与样本数都是最小值的 2 倍（N66）。→ 第 6 课。
- **W25〔概〕Robbins–Monro 条件。** 判断 $a_n=1/n$、$1/\sqrt n$、常数 0.01 是否满足；答：只有 $1/n$ 两条都满足；$1/\sqrt n$ 的平方和发散（N55）；常数学习率不满足 $\sum a_n^2<\infty$，所以 LLM 训练的常数期（WSD 的 stable）不在 RM 定理的保证范围内，衰减期的作用可以借此直觉解释【笔者】。→ 第 2、6 课。
- **W26〔算〕重球的加速。** 条件数 $10^4$ 的二次函数，最速下降与最优重球到 $10^{-6}$ 各需多少步？答：约 69,078 与 691（N56）。→ 第 2 课。
- **W27〔算〕读懂 V4.1 的调度。** 45T token 中稳定、衰减、末段各占多少？答：62.2%、26.7%、11.1%（N69）；与 MiniCPM“10% 衰减足够”对比。→ 第 18、19 课。

### M08 浮点与量化

- **W28〔算〕0.1 存成什么。** 在 FP32、FP16、BF16、E4M3、E5M2 下的值与相对误差？答：见 N9（E4M3 为 0.1015625，误差 1.56%）。→ 第 12 课。
- **W29〔算〕E4M3 的最大值为什么是 448 而不是 240 或 480？** 答：去掉无穷、只留 S.1111.111 为 NaN 后顶码成为正规数，$1.75\times2^8=448$；若守 IEEE 惯例，顶码全留给 Inf/NaN，最大值是 $1.875\times2^7=240$（注意不是 $1.75\times2^7$，那等于 224）；若零与 NaN 都只留一个编码（放弃正负对称），还能多出 480（C64、N4）。→ 第 12 课。
- **W30〔算〕吞没。** BF16 里 $256+1$、$1024+1$ 等于多少？顺序累加 10000 个 1 得多少？答：256、1024、256（N11–N12）。追问：4096 个 U(0,0.01) 的 BF16 累加误差？答：80%（N13）。→ 第 12 课（DeepSeek-V3 为什么每 128 个元素提升到 FP32 累加）。
- **W31〔算〕MXFP4 块。** 块内最大值 7 与 0.9，按 OCP 算法 1 量化后最大值变成多少？答：6 与 0.75（N19）；按 NVIDIA“缩放向上取整”：7 → 缩放 2、可用格点 {0,1,2,3,4,6}（N20）。→ 第 19 课。
- **W32〔算〕位/元素。** MXFP4、NVFP4、V4.1 KV（E2M1+每 16 通道 E4M3）、V3 激活 FP8（1×128 FP32 缩放）？答：4.25、4.5、4.5、8.25（N16）。→ 第 12、19 课。
- **W33〔概〕细粒度缩放为什么救了 FP4。** 解释 N22：E2M1 只有约 3.6 个 binade，一个 50× 的离群值用整张量缩放会把 96.8% 的值冲成 0；E4M3 有 18 个 binade，要 $10^5\times$ 的离群值才出现同样问题。→ 第 12 课（1×128 / 128×128）、第 19 课。
- **W34〔概〕两种 machine epsilon。** FP32 的 `eps` 是 $1.19\times10^{-7}$ 还是 $5.96\times10^{-8}$？答：看约定（N14、C60）；正文统一写“1 与下一个数的间距 $2^{-(p-1)}$”并注明另一种。

### M09 策略梯度

- **W35〔算〕GAE 手算。** $r=(0,0,1)$、$V=(0.5,0.6,0.8,0)$、$\gamma=0.99$、$\lambda=0.95$：求 $\delta_t$ 与 $\hat A_t$。答：$\delta=(0.094,0.192,0.2)$，$\hat A=(0.4515,0.3801,0.2)$；$\lambda=1$ 时 $(0.4801,0.39,0.2)$ 等于 MC 回报减 $V$（N70）。→ 第 3 课。
- **W36〔算〕基线不改均值、只降方差。** Bernoulli 例（N72）：$b=0$、10、10.3 的方差 544.97、2.33、0.76，均值都 ≈1。→ 第 8 课。
- **W37〔推〕GRPO 的组均值基线含自身。** 设 $b_i=\frac1G\sum_jr_j$，证明 $\E[(r_i-b_i)\nabla\log\pi(o_i)]=\frac{G-1}G\E[r_i\nabla\log\pi(o_i)]$（其他样本与 $o_i$ 独立，只剩 $r_i/G$ 一项）；$G=16$ 时因子 0.9375（N74）。
  注意：除以组标准差后不再是这个简单结论（【笔者推导】，写课时标注）。→ 第 8 课。
- **W38〔算〕PPO 裁剪四种情形。** $\varepsilon=0.2$：$(r,A)$=(1.3,1)、(1.3,−1)、(0.7,1)、(0.7,−1) 的目标值？答：1.2、−1.3、0.7、−0.8（N73）。→ 第 3、8 课。

### M10 算力算术

- **W39〔算〕环形 allreduce。** 14 GB 梯度、8 卡、160 GB/s：时间？1024 卡、50 GB/s 呢？答：0.153 s；0.560 s + 延迟项 10.2 ms（假设 $\alpha=5\,\mu$s）（N77）。追问：为什么卡数翻倍时间几乎不变？答：每卡流量 $2(p-1)/p\cdot n\to2n$。→ 第 12 课。
- **W40〔算〕MoE 的 all-to-all 字节（示意）。** $h=7168$，FP8 分发（每 128 元素一个 FP32 缩放）+ BF16 合并，每 token 发往 8 个专家：每 token 字节？答：分发 57.8 KiB、合并 112.0 KiB；节点受限（4 份跨节点）时 28.9 与 56.0 KiB（N80）。→ 第 11、12 课。

### M11 Sinkhorn 与双随机

- **W41〔算〕2×2 的 Sinkhorn 极限（有闭式）。** $A=\begin{pmatrix}1&2\\3&4\end{pmatrix}$：利用交叉比 $a_{11}a_{22}/(a_{12}a_{21})$ 在 $D_1AD_2$ 下不变，求极限。答：$a=0.449490$（N32）。→ 第 16 课。
- **W42〔推〕双随机矩阵的谱范数恰为 1。** 用 §2.2 的 Cauchy–Schwarz 证明 $\le1$，再用 $\mathbf 1$ 证明等号；反例：行随机但 $\|R\|_2=2$（N29）。→ 第 16 课。
- **W43〔算〕有支撑但无完全支撑。** $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ 交替归一化 10、100、1000 次后右上角是多少？答：0.0476、0.00498、0.00050（N35）；极限是单位阵，不是 $D_1AD_2$ 形式（C12、C14）。→ 第 16 课。
- **W44〔算〕20 次够不够。** N33 的 4×4 例 20 次后列误差 1.4e-11；logits 放大 10 倍时 0.084，谱范数 1.0143（N34）。讨论：mHC 用 $\exp$ 保证正性（所以有完全支撑、必收敛），但收敛速度取决于矩阵元素的悬殊程度。→ 第 16 课。
- **W45〔算〕Birkhoff 分解。** 写出 $D=\begin{pmatrix}0.5&0.3&0.2\\0.2&0.5&0.3\\0.3&0.2&0.5\end{pmatrix}$ 的置换分解（$0.5I+0.3P+0.2P^2$），并求 $D^{10}$ 与 $J/3$ 的最大差（1.1e-6，N31）——“反复混合趋于平均”。→ 第 16 课。

---

## 7. 勘误与常见误解

每条：常见说法 → 正确说法 → 依据。**写课时不要回退这些更正。**

### 线性代数（M02、M11）

- **E1 “Eckart–Young 1936 证明了谱范数下的最优低秩近似。”** 错位归属。1936 年论文解的是最小二乘（Frobenius）问题（`eckart1936` 摘要）；这个 Frobenius 结论 Schmidt 1907 已给出，
  Stewart 明说“Eckart–Young 定理”是误称（C4、C5）；谱范数（以及所有酉不变范数）的版本是 Mirsky 1960（C6）。`mml2020` 定理 4.25 把谱范数陈述挂在 Eckart & Young 名下（C8），不要照抄这个归属。
  正文写法建议：“最优低秩近似定理（Schmidt 1907；Eckart–Young 1936 重新发现；Mirsky 1960 推广到一切酉不变范数，含谱范数）”。
- **E2 “截断 SVD 是唯一的最优秩 $k$ 近似。”** 只在 Frobenius 范数且 $\sigma_k>\sigma_{k+1}$ 时成立（“usually unique”，C3）。谱范数下极小元一般不唯一：$\mathrm{diag}(3,2,1)$ 的秩 1 近似 $\mathrm{diag}(b,0,0)$ 对所有 $b\in[1,5]$ 谱误差都是 2（N23、F02-2）。
  另：`mml2020` 式 (4.94) 的约束写 “$\mathrm{rk}(B)=k$”，标准写法是“$\le k$”（Schmidt 的原问题就是“秩不超过 $k$”，C4）。
- **E3（M11）“双随机矩阵的谱范数小于 1。”** 等于 1，不是小于 1：$A\mathbf 1=\mathbf 1$（§2.2）。mHC 的原话是“bounded by 1”（C10），准确。“非扩张”不等于“收缩”。
- **E4（M11）“行和为 1 就够稳定了。”** 只约束行和的矩阵谱范数可以到 $\sqrt n$（N29）；mHC 需要行、列两个条件。
- **E5（M11）“Sinkhorn 迭代对任何非负矩阵都收敛到 $D_1AD_2$。”** 精确条件（`sk1967` 定理，C12）：收敛到双随机极限 ⇔ 有支撑（至少一条正对角线）；极限是 $D_1AD_2$ ⇔ 完全支撑；
  $D_1,D_2$ 唯一（差标量）⇔ 完全不可分解。正矩阵（mHC 的 $\exp$ 输出）三条都满足。有支撑无完全支撑时收敛是次线性的（N35）。
- **E6（M11）“mHC 输出的矩阵是双随机的。”** 只是近似：20 次、最后一步做行归一化，所以行和精确为 1、列和有误差（mHC 自己在 §5.4 承认反向增益略偏离 1，C16）；logits 悬殊时误差可达 $10^{-2}$，谱范数可略超 1（N34）。
- **E7（M11）Birkhoff 1946 的页码。** `sk1967` 参考文献写 147–150；本任务说明与不少二手文献写 147–151。未能核对原刊（§8）。

### 信息论（M05）

- **E8 “KL 散度是距离。”** 不对称（N36），不满足三角不等式；MacKay 明确说它不是严格的距离（C19）。另外，N37 的例子 $p=(0.5,0.3,0.2)$、$q=(0.2,0.3,0.5)$ 两个方向 KL 恰好相等（互为镜像），举不对称例子时别选这种对。
- **E9 “KL ≥ 0 对任何 $Q$ 成立。”** 要求 $Q$ 归一化（§2.3 推导的最后一步）；未归一化时可为负（N38）。单样本的 $k_1=-\log r$ 可以为负，但期望非负（C21）。
- **E10 “Gibbs 不等式等号只在 $P=Q$ 时成立”这句话的逻辑。** MacKay 写“only if”（C19），完整的是“当且仅当”（$P=Q$ 时显然为 0）。

### 强化学习（M09）

- **E11 “策略梯度定理是 Sutton et al. 2000 首创。”** 论文自己说平均回报情形由 Marbach & Tsitsiklis (1998) 首先讨论，并指出 Williams 的 REINFORCE 理论也蕴含它（C27）。可以写“Sutton 等 (2000) 给出通行形式与证明”。
- **E12 “$d^\pi$ 是状态的平稳分布。”** 平均回报表述里是平稳分布；起始状态（折扣）表述里是 $\sum_t\gamma^t\Pr\{s_t=s\}$，未归一化（总和 $1/(1-\gamma)$），而且实践中的采样通常不带 $\gamma^t$ 权重（C26；后一点的文献未核，见 §8）。
- **E13 “REINFORCE 的基线可以是任何东西。”** Williams 的条件是：给定 $W$ 与输入时，基线与该单元的输出（动作）条件独立（C22）。依赖动作本身的“基线”会引入偏差。GRPO 的组均值包含样本自身，
  均值那部分的效果是把梯度乘以 $(G-1)/G$（W37，【笔者推导】），除以标准差之后的影响更复杂。
- **E14 “GAE 的 λ=1 是无偏的。”** 只相对于**折扣梯度** $g^\gamma$ 无偏；$g^\gamma$ 本身是对未折扣目标梯度的有偏估计（`gae` 脚注 1，C29）。
- **E15 “GAE 是 Schulman 等发明的公式。”** 作者注明公式已由 Kimura & Kobayashi (1998)、Wawrzyński (2009) 提出，他们的贡献是分析与应用（C32）。

### 优化器（M07）

- **E16 Adam 默认值的版本漂移。** v1（2014-12）把 $\beta$ 写成“1−衰减率”：默认 $\beta_1=0.1$、$\beta_2=0.001$、$\alpha=0.0002$；v9（ICLR 2015 版）是 $\beta_1=0.9$、$\beta_2=0.999$、$\alpha=0.001$（C35、C38）。
  两版 $\epsilon$ 都是 $10^{-8}$。**本仓库的 `arxiv.py text 1412.6980` 拿到的是 v1**（§0.4）。v9 §3 末尾仍有一句 v1 遗留的措辞：为稀疏梯度“选择小的 $\beta_2$”——按 v9 的参数化应为“接近 1 的 $\beta_2$”
  （同文 §6.4 已改为 “close to 1”）。引用 Adam 请写 “Kingma & Ba, ICLR 2015（arXiv:1412.6980v9）”。
- **E17 “LLM 用 Adam 默认值。”** DeepSeek-V4/V4.1 的 AdamW 是 $\beta_2=0.95$、$\varepsilon=10^{-20}$、衰减 0.1（C51、C52）；$10^{-20}$ 在 FP16 里是 0（N15）。“默认值”是论文实验的默认，不是大模型的惯例。
- **E18 AdamW 的衰减乘什么。** 论文算法 2 的衰减项乘调度乘子 $\eta_t$，**不乘**学习率 $\alpha$（C40）；PyTorch 乘 lr（C41）。所以“weight_decay=0.1”在两种写法下不是同一个数：论文的 $\lambda$ ≈ PyTorch 的 lr × weight_decay。
  也不要说“AdamW 就是 L2 正则”：只有 SGD 下两者等价，且需 $\lambda'=\lambda/\alpha$（命题 1）；Adam 下不等价（命题 2）（C39）。v1/v2 的标题是 “Fixing Weight Decay Regularization in Adam”（C42）。
- **E19 MiniCPM 的衰减公式。** 论文 §6.2 印作 $f(s-T)=0.5^{(s-S)/T}$ 且“$T$ 为 5000 步”：$T$ 同时当“稳定期结束步”和“半衰期”，指数里的 $s-S$ 在衰减期为负，会让 $f>1$，与式 (1) 要求的 $0<f\le1$ 矛盾。
  合理读法是 $f(u)=0.5^{u/h}$，$u=s-T$，半衰期 $h=5000$ 步（N67 按此读法算）。写课时注明“按半衰期理解”。
- **E20 “Muon 把更新变成 $UV^\top$。”** Jordan 系数的 $\varphi_J(1)=0.701$，迭代不收敛到 1，奇异值落在约 $[0.68,1.13]$ 振荡（N57–N58）；博客的设计目标是 $[0.7,1.3]$，代码注释说像 Uniform(0.5,1.5)（C44、C46）。
  DeepSeek-V4 专门在最后加两步 $(2,-1.5,0.5)$ 把奇异值拉到 1（C50）。正文应说“近似正交化”。
- **E21 Muon 的形状缩放有三种约定。** Jordan 代码 $\sqrt{\max(1,\text{rows}/\text{cols})}$（更新 RMS 约为 $1/\sqrt{\text{cols}}$）；Moonlight $0.2\sqrt{\max(A,B)}$（RMS 0.2）；DeepSeek $\gamma\sqrt{\max(n,m)}$，RMS 0.18（N61）。
  另外动量写法不同（Jordan 代码 EMA：$M\leftarrow\beta M+(1-\beta)G$，送入 NS 的是 $(1-\beta)G+\beta M$；Moonlight/V4：$M\leftarrow\mu M+G$，送入 $G+\mu M$）。同一梯度历史下前者恰为后者的 $(1-\beta)$ 倍，而 $X/\|X\|_F$ 归一化与正交化都与正的尺度无关，所以两种写法得到**同一个** $\mathbf O_t$【笔者推导】；差别只在不做正交化的优化器（如 SGD 动量）里表现为学习率缩放。
- **E22 “Muon 适用于所有参数。”** 只用于隐藏层 2D 权重；嵌入、输出头、标量/向量参数用 AdamW（C43）；DeepSeek-V4 还把 mHC 的静态偏置与门控、RMSNorm 权重交给 AdamW（C50），V4.1 把嵌入与预测头交给 Sinkhorn 平衡更新（C52）。
- **E23 “Muon 出自 Jordan 2024 的论文。”** 没有论文，是博客与代码（`jordan-blog`、`muon-code`，按制品引用）；博客日期 2024-12-08，博客引用的最早 X 帖为 2024-10-04。规模化论文是 Liu et al. 2025（`moonlight`）。
- **E24 Muon 开销的两个例子。** 博客写 0.7% 与 0.5%，精确值 0.732% 与 0.512%（N63）；$Tm/B$ 是方阵时的上界，长方阵实际更小（N62）。
- **E25 “线性缩放规则是定理。”** 是经验法则，作者自己列出失效条件（训练初期、batch 过大）（C54）。它的推导假设 SGD；对 Adam 类优化器不能直接搬。
- **E26 两个“噪声尺度”。** McCandlish 的 $\mathcal B_{noise}$ 是一个 batch 大小（样本数）；Smith & Le 的 $g\approx\epsilon N/B$ 是类似温度的量（C56、C57）。不要混用。$\mathcal B_{simple}$ 只在 Hessian 为单位阵倍数时等于 $\mathcal B_{noise}$。
- **E27 “Robbins–Monro 条件保证 LLM 训练收敛。”** 常数学习率不满足 $\sum a_n^2<\infty$（N55）；RM 原文是单变量求根、证明的是 $L^2$ 收敛（C33）。$1/\sqrt n$ 也不满足第二个条件。
- **E28 “动量总能加速 $\sqrt R$ 倍。”** Polyak 的结论是极小点附近、二次近似、无噪声梯度下的渐近局部速度；随机梯度下优势会丧失（C34）。

### 数值格式（M08）

- **E29 “E4M3 最大值 240 / 480。”** 是 448：无无穷、只有 S.1111.111 为 NaN（C63、C64）。240 是 IEEE 式保留顶码时的值，480 是连 ±NaN 对称都放弃时的值。
- **E30 “E4M3 与 E5M2 都有无穷。”** 只有 E5M2 有（S.11111.00）；E4M3 的无穷在转换时变成 NaN，溢出默认饱和（C63、C65）。
- **E31 “bfloat16 没有次正规数。”** 编码上与 FP32 一样，指数全 0 的位型可以表示次正规数（最小 $2^{-133}$，【算】）；Kalamkar 表 1 的 “N/A” 论文没有解释原因，正文也不再出现 subnormal / denormal 字样；【笔者推断】它反映的是他们的硬件或仿真约定，不是格式本身（C61）。另外“8 bits of mantissa”把隐含位算进去了，存储的尾数是 7 位。
- **E32 “bfloat16 是 Intel 这篇论文提出的。”** 论文自述最早出现在 DistBelief 与 TensorFlow 中，之后主要在 Google 生态中使用（C62；论文这一段没有提到 TPU，旧版写的 “TPU” 是添加，已删）；Kalamkar 等是第一篇系统的训练实证研究。
- **E33 两种 machine epsilon。** numpy/C 的 `eps` = $2^{-(p-1)}$（FP32 $1.19\times10^{-7}$）；Goldberg（以及许多数值分析教材的单位舍入 $u$）= $2^{-p}$（$5.96\times10^{-8}$）（C60、N14）。正文要写明用哪一个。
- **E34 “IEEE 754 规定了 bfloat16 / FP8 / FP4。”** 它们不是 IEEE 754-2008/2019 的格式：FP8 论文把 E4M3/E5M2 作为新的交换格式提出，并说明 E5M2“遵循 IEEE 惯例”（C63–C64）；MX 格式由 OCP 规范定义（C66–C67）。（IEEE 754 原文本次未读，此条依据的是这些论文的自我定位。）
- **E35 “MXFP4 是 4 位。”** 每元素平均 4.25 位（加 1/32 个 E8M0），NVFP4 4.5 位（N16）。E8M0 没有符号、没有尾数、不能表示 0 与 Inf，只表示 2 的幂（C67、C69）。
- **E36 “MX 转换不会溢出。”** OCP 示例算法 1 用 $\lfloor\log_2 amax\rfloor-emax_{elem}$，块最大值的尾数大于元素格式最大尾数（FP4 为 1.5）时会被钳位：7→6、0.9→0.75（N19）；NVIDIA 与 Mishra et al. 改为缩放向上取整（C69、N20）。
- **E37 “V4.1 的 FP4 KV cache 是 NVFP4 / MXFP4。”** 两者都不是：E2M1 + 每 16 通道一个 E4M3 缩放、**没有**第二级全局缩放（C71）；V4 的 indexer Q/K 才是 MXFP4。
- **E38 “FP8 格式推荐所有张量用 E4M3。”** FP8 论文推荐权重/激活 E4M3、梯度 E5M2（C65）；DeepSeek-V3 全用 E4M3 是偏离推荐、靠细粒度缩放弥补（【交叉引用】lesson-12-doc C47）。

### 算力与通信（M10）

- **E39 “Roofline 定义了 arithmetic intensity。”** Roofline 用的是 operational intensity（DRAM 流量），并刻意区别于 arithmetic intensity（C73；mamba lesson-01-doc 勘误 11 已记）。
- **E40 “环形 allreduce 的时间与卡数无关。”** 只有带宽项趋于常数 $2n\beta$；延迟项 $2(p-1)\alpha$ 线性增长（1024 卡时 2046 步，N79）。Goyal 等实测减半倍增在 32 台服务器上比环快 3 倍（C75）。
- **E41 “环形 allreduce 是 Patarasuk & Yuan 发明的。”** 他们证明了带宽下界并给出在树拓扑上达到下界的环形算法（C75）；环（bucket）算法本身更早（Goyal 引用 [2]，未核，见 §8）。

### 研究流程

- **E42 `arxiv.py text` 的版本回退。** 见 §0.4：无可用 HTML 的最新版会静默退回 v1。每次都在页头确认版本。

### 核查补充（2026-09-24 对抗核查新增，详见文末核查记录）

- **E43 “IEEE 式 E4M3 的最大值 $1.75\times2^7=240$。”** 算式错：$1.75\times2^7=224$。IEEE 式布局下最大正规数是 $1.875\times2^7=240$（指数码 1110、尾数 111）；E4M3 实际的 448 才是 $1.75\times2^8$（尾数 111 被 NaN 占用，最大尾数只到 110）。
- **E44 “Jordan 系数的 NS 迭代最终在 0.753 与 1.134 之间 2 周期振荡。”** 数值迭代显示吸引子是 **4 周期** 0.682 → 1.134 → 0.753 → 1.047（N58）。区间 $[0.68,1.13]$ 仍然对。
- **E45 “Patarasuk–Yuan 证明每个进程都至少要发 $2\frac{p-1}pn$。”** 引理 4 说的是“至少有一个进程”（C75、§2.10）。

---

## 8. 未能核实 / UNVERIFIED

正文若要用下列内容，要么不写，要么明确写“未能从原文核实”。

1. **Mirsky 1960 原文**（OUP 403）。“对一切酉不变范数成立”只来自 Stewart 1993 的转述（C6）；Mirsky 论文里的定理编号、证明方式、是否单独处理谱范数，均未见。
2. **Eckart & Young 1936 全文**（付费）。只读到摘要；论文里的具体陈述形式、是否讨论唯一性条件、符号，未见。Eckart–Young 1939（BAMS 45:118–121）未读。
3. **Schmidt 1907 原文**（德文）。近似定理的陈述来自 Stewart §5 的矩阵化重述。
4. **Birkhoff 1946 原文**（西班牙文，无 DOI）：页码 147–150（`sk1967` 参考文献）与 147–151（常见引法、本任务说明）不一致；定理在原文中的确切表述（凸包 / 极点）未见。
   “Birkhoff–von Neumann”中 von Neumann 的贡献（通常引 1953 年关于分配问题的论文）未核。
5. **Sinkhorn 1964 原文**（Project Euclid 机器人拦截，未绕过）。C11 来自 `sk1967` 引言。收敛速度（线性收敛及其常数）的文献（如 Franklin–Lorenz 1989、Knight 2008）未核，正文不要给速率常数。
6. **“完全不可分解”的定义**：`sk1967` 沿用 Marcus–Minc 记号，本次未见原定义（C13 的定义是笔者补充）。
7. **Robbins–Monro 1951 原文**（同 5）。C33 来自 Lai & Yuan 的预印本（文件日期 2019-12-09，页眉 “Submitted to Statistical Science”）；该综述是否及何时正式发表未核。原文是否为“$M(x)=\alpha$ 求根”的表述、定理编号，未见。
8. **Polyak 1964 原文**（Elsevier 付费）。C34 来自 Sutskever et al. 2013 §2 的转述。
9. **IEEE 754-2008 / 2019 正文**（付费）。binary16 参数（偏置 15、$e_{max}=15$）只由 `bf16` 表 1、`fp8` §3.2、numpy 间接核对；binary32/64 来自 Goldberg 的 754-1985 表。
   754-2008 是否定义了 binary16 为“交换格式”、754-2019 的改动，未核。TF32（1/8/10）的位布局本次**没有**一手来源，已从 §3.1 表中删去。
10. **OCP MX v1.0 规范**（opencompute.org 403）。E8M0 的 NaN 编码（0xFF）、偏置 127、FP6/FP4 的偏置、规范发布日期与版本号，全部来自 `mx` 论文、`nvfp4` 论文、`microxcaling` 代码的间接证据；
   E8M0 “0xFF = NaN” 是推断（代码把溢出置 NaN，NVFP4 论文给出范围 $2^{-127}$–$2^{127}$，二者合起来才推出 255 个有限值 + 1 个 NaN）。
11. **NVIDIA 的 NVFP4 博客**（“Introducing NVFP4…”）未读；NVFP4 细节来自 `nvfp4` 论文（v2 修正了式 2 的笔误，本文未用式 2）。
12. **bfloat16 次正规数的硬件支持**：E31 只说编码允许，NVIDIA/Intel/TPU 实际是否冲零，未核。
13. **Adam 收敛证明的缺陷**：Adam v9 定理 4.1 的条件 $\beta_1^2/\sqrt{\beta_2}<1$ 已读到，但“该证明有误、存在反例”（常引 Reddi, Kale, Kumar, ICLR 2018）本次**未核**，正文不要写。
14. **折扣状态分布与实际采样不一致**（E12 后半句）：常引 Thomas 2014、Nota & Thomas 2020，本次未核。
15. **环（bucket）allreduce 的最早出处**：Goyal 引用 [2]，Thakur 的 §5.3 未给首创者；未核。
16. **Roofline 的 CACM 期刊版**：只读了 2008 技术报告；期刊版措辞、图号可能不同。Crossref 题名为 “Roofline”（C73）。
17. **Stewart 1993 式 (5.3) 左边缺平方**：扫描图上左边印作 $\|A-\sum\mathbf x_i\mathbf y_i^\top\|$ 而右边是平方和，按上下文应为平方；属于读扫描件的判断，置信度中等。核查时用带文本层的镜像再看，上标在文本层里也丢失了，仍无法定论。
18. **GAE 的会议出处**：来自 Abbeel 主页 bib 与 lesson-03-doc 读到的 PDF 页眉；本次读的 LaTeXML v6 渲染里没有 “ICLR” 字样。
19. **MiniCPM 的出版状态**：arXiv comment 只写 “revise according to peer review”，会议/期刊未核。
20. **Kalamkar 2019、FP8 2022、MX 2023、McCandlish 2018、Moonlight 2025、Bernstein–Newhouse 2024** 均按 arXiv 预印本引用，正式出处未核。
21. **DeepSeek-V4 的 arXiv 编号与日期**：`arxiv.py meta 2606.19348` 显示日期 2026-04-26，与 “2606”（2026 年 6 月提交）的编号前缀矛盾；留给第 18 课研究笔记核对（本文只用了它的 §2.2、§2.4、§4.2.2 内容）。
22. **硬件数字**（A100 312 TFLOP/s、2.039 TB/s；H100 SXM 989/3.35；H800 NVLink 160 GB/s、IB 50 GB/s）全部是【交叉引用】，本次未重核。
23. **V3 all-to-all 的字节示意（N80）** 假设“FP8 + 每 128 元素一个 FP32 缩放”作为分发格式；DeepEP 的实际打包格式未核。
24. **GRPO 组均值的 $(G-1)/G$ 缩放（W37、N74）** 是笔者推导，未见文献；除以组标准差后的性质未分析。
25. **V4.1 Sinkhorn 平衡更新 = 对 $U^2$ 做非均匀边际 Sinkhorn**（C18）是笔者解读，论文没有这样表述。
26. **“Muon 的动量两种写法给出同一个 $\mathbf O_t$”**（E21）是笔者推导；只在 NS 前的 Frobenius 归一化存在时成立（Jordan 代码与 Moonlight 式 (2) 都有这一步）。
27. **Jordan 博客里 Muon 定义的公式**是图片，文本抽取拿不到；C43 的定义取自博客的文字说明与代码，而不是那张公式图。
28. ~~**`mml2020` 定理 4.25 所在的节号**未确认~~ 已在核查中确认：§4.6 “Matrix Approximation”，pp. 131–132（本条保留编号，内容已解决）。
29. **V4.1 KV 幅值上界的推理**（C71）：论文同段同时给出“RMSNorm 权重最大约 11”与“逐通道最大值约不超过 $\sqrt{512}$”，两者如何衔接论文没有交代；本文只采信“上限 2688、观测最大约 10”。
30. **Muon 的 4 周期**（N58）是对标量多项式 $\varphi_J$ 的数值迭代结论（双精度）。实际 Muon 在 bf16 下只做 5 步，还到不了这个周期；bf16 舍入会不会改变长期行为，没有试。

---

## 9. 参考文献卡片草稿

论文在前，制品（`kind: artefact`）在后。每张卡的 `finding` 与 `matters` 不同：前者说来源确立了什么，后者说数学课为什么引它。

### 9.1 论文、书与标准

```
id: schmidt1907
title: Zur Theorie der linearen und nichtlinearen Integralgleichungen. I. Teil
a: Erhard Schmidt (1907), Mathematische Annalen 63(4):433–476, doi:10.1007/BF01449770
url: https://doi.org/10.1007/BF01449770
finding: 在积分方程的框架下给出奇异值展开，并证明用 k 个秩一项做最小二乘最佳近似时，最优解是前 k 项截断（按 Stewart 1993 的转述）。
matters: M02 讲“最优低秩近似”时用于纠正归属：这个定理最早属于 Schmidt，而不是 Eckart–Young（原文未读，引用时注明经 Stewart 转述）。
```

```
id: eckart1936
title: The Approximation of One Matrix by Another of Lower Rank
a: Carl Eckart, Gale Young (1936), Psychometrika 1(3):211–218, doi:10.1007/BF02288367
url: https://doi.org/10.1007/BF02288367
finding: 把“用低秩矩阵近似给定矩阵”写成最小二乘问题，借正则形式（SVD）求解，并指出解总存在、通常唯一。
matters: M02 的最小二乘（Frobenius）版本以它命名；本课借它的摘要说明 1936 年处理的是最小二乘而不是谱范数。
```

```
id: mirsky1960
title: Symmetric Gauge Functions and Unitarily Invariant Norms
a: L. Mirsky (1960), The Quarterly Journal of Mathematics 11(1):50–59, doi:10.1093/qmath/11.1.50
url: https://doi.org/10.1093/qmath/11.1.50
finding: 证明截断 SVD 在任意酉不变范数下都是最优低秩近似（按 Stewart 1993 的转述）。
matters: M02 里谱范数版本（误差等于 σ_{k+1}）的正确出处；讲 MLA 压缩误差用谱范数时引它。
```

```
id: stewart1993
title: On the Early History of the Singular Value Decomposition
a: G. W. Stewart (1993), SIAM Review 35(4):551–566, doi:10.1137/1035134
url: https://doi.org/10.1137/1035134
finding: 梳理 Beltrami、Jordan、Sylvester、Schmidt、Weyl 对 SVD 的贡献，指出近似定理属于 Schmidt、“Eckart–Young 定理”是误称，Mirsky 推广到一切酉不变范数。
matters: M02 历史注记与三方归属（Schmidt / Eckart–Young / Mirsky）的依据，也是本文 E1 勘误的来源。
```

```
id: mml2020
title: Mathematics for Machine Learning
a: M. P. Deisenroth, A. A. Faisal, C. S. Ong (2020), Cambridge University Press（免费 PDF，页脚 Draft 2024-01-15）
url: https://mml-book.github.io/
finding: 以定理 4.22 陈述 SVD 存在性，定理 4.24 给出谱范数等于最大奇异值，定理 4.25 给出谱范数下的最优秩 k 近似及零空间维数证明。
matters: M02 可直接给学生的开放教材出处；但其定理 4.25 把谱范数版本归给 Eckart & Young，引用时要改正归属（E1）。
```

```
id: birkhoff1946
title: Tres observaciones sobre el algebra lineal
a: G. Birkhoff (1946), Univ. Nac. Tucumán Rev. Ser. A 5:147–150（页码据 Sinkhorn–Knopp 1967；常见引法为 147–151）
url: （无在线版本；经 Sinkhorn & Knopp 1967 引用）
finding: n 阶双随机矩阵之集恰是 n×n 置换矩阵的凸包（按 Sinkhorn–Knopp 1967 p.345 的转述）。
matters: M11 讲 mHC 把残差映射解释为“置换的凸组合”、以及谱范数 ≤ 1 的第二种证明时引它（原文未读）。
```

```
id: sinkhorn1964
title: A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices
a: Richard Sinkhorn (1964), The Annals of Mathematical Statistics 35(2):876–879, doi:10.1214/aoms/1177703591
url: https://doi.org/10.1214/aoms/1177703591
finding: 每个正方阵都唯一对应一个形如 D1AD2 的双随机矩阵，可由交替归一化行与列的迭代得到（按 Sinkhorn–Knopp 1967 引言的转述）。
matters: M11 里“mHC 先取 exp 再迭代”为什么一定收敛的最简出处：exp 的输出是正矩阵。
```

```
id: sinkhorn1967
title: Concerning Nonnegative Matrices and Doubly Stochastic Matrices
a: Richard Sinkhorn, Paul Knopp (1967), Pacific Journal of Mathematics 21(2):343–348, doi:10.2140/pjm.1967.21.343
url: https://msp.org/pjm/1967/21-2/p14.xhtml
finding: 对非负方阵给出精确条件：交替归一化收敛到双随机极限当且仅当有支撑；极限可写成 D1AD2 当且仅当有完全支撑，此时唯一，缩放矩阵唯一当且仅当完全不可分解。
matters: M11 的主定理出处；也提供“支撑但非完全支撑”的反例思路，用来讲 Sinkhorn 什么时候慢、什么时候不能写成对角缩放。
```

```
id: xie2025mhc
title: mHC: Manifold-Constrained Hyper-Connections
a: Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, et al. (2025), arXiv preprint, arXiv:2512.24880v2
url: https://arxiv.org/abs/2512.24880
finding: 用 Sinkhorn–Knopp（20 次、先列后行）把超连接的残差矩阵投影到双随机矩阵，得到非扩张、乘法封闭的残差混合，n=4 时训练开销 6.7%。
matters: M11 每一节都要回指的应用：谱范数 ≤ 1、乘法封闭、Birkhoff 解释都来自它的 §4.1，有限步误差来自 §5.4。
```

```
id: deepseekai2026v4
title: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
a: DeepSeek-AI (2026), technical report, arXiv:2606.19348v1
url: https://arxiv.org/abs/2606.19348
finding: 大部分模块用 Muon（混合 Newton–Schulz 8+2 步、RMS 重缩放 0.18、动量 0.95、衰减 0.1），其余用 AdamW（β2 0.95、ε 1e-20），mHC 取 n_hc=4、Sinkhorn 20 次。
matters: M07 讲 Muon 与 AdamW 的混用、M11 讲 Sinkhorn 次数时的 DeepSeek 实例；也是“NS 不收敛到 1、所以加两步”这一设计的出处。
```

```
id: deepseekai2026v41
title: DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression
a: DeepSeek-AI (2026), technical report, arXiv:2609.19969v1
url: https://arxiv.org/abs/2609.19969
finding: 主 KV cache 用 E2M1 加每 16 通道一个 E4M3 缩放（无全局缩放）做 QAT；嵌入、预测头与 Engram 表改用带 Sinkhorn 平衡的动量更新（K=11，γ=0.18）。
matters: M08 的块缩放与 M11 的 Sinkhorn 都能在课程终点模型里找到落脚点；也提供 WSD 形状的真实调度（28T 稳定、12T 衰减）。
```

```
id: jensen1906
title: Sur les fonctions convexes et les inégalités entre les valeurs moyennes
a: J. L. W. V. Jensen (1906), Acta Mathematica 30:175–193, doi:10.1007/BF02418571
url: https://doi.org/10.1007/BF02418571
finding: 建立凸函数与均值之间的一般不等式，即今天所称的 Jensen 不等式（只核对了书目）。
matters: M05 证明 KL 非负时给出不等式的原始出处；定理的现代陈述用 MacKay。
```

```
id: mackay2003
title: Information Theory, Inference, and Learning Algorithms
a: David J.C. MacKay (2003), Cambridge University Press（作者网站免费 PDF）
url: https://www.inference.org.uk/itprnn/book.pdf
finding: §2.6 定义相对熵并陈述 Gibbs 不等式 D_KL(P||Q) ≥ 0，§2.7 陈述 Jensen 不等式及严格凸时的等号条件。
matters: M05 的教科书出处，读者可以免费查到；课程里 Gibbs 不等式的式号 (2.46)、Jensen 的 (2.48) 以此为准。
```

```
id: williams1992
title: Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning
a: Ronald J. Williams (1992), Machine Learning 8(3–4):229–256, doi:10.1007/BF00992696
url: https://doi.org/10.1007/BF00992696
finding: 定义 REINFORCE 类算法 Δw = α(r − b)∂ln g/∂w，并证明（定理 1、2）其期望更新与期望奖励的梯度同向，常数学习率时恰为梯度。
matters: M09 对数导数技巧与“基线不改变期望”的原始出处，也是 GRPO 用组均值当基线时要满足的条件（基线与动作条件独立）的来源。
```

```
id: sutton2000
title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
a: Richard S. Sutton, David A. McAllester, Satinder P. Singh, Yishay Mansour (2000), Advances in Neural Information Processing Systems 12, pp. 1057–1063（NIPS 1999）
url: https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html
finding: 定理 1 证明策略梯度可写成 Σ_s d^π(s) Σ_a ∂π/∂θ · Q^π(s,a)，不含状态分布对参数的导数，平均回报与起始状态两种表述都成立。
matters: M09 推导策略梯度定理时对照的原式 (2)；课程还要引它自己的说明，交代 Marbach–Tsitsiklis 的先行工作与 d^π 的折扣定义。
```

```
id: schulman2016gae
title: High-Dimensional Continuous Control Using Generalized Advantage Estimation
a: John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2016), ICLR 2016, arXiv:1506.02438v6
url: https://arxiv.org/abs/1506.02438
finding: 把 k 步优势估计做指数加权平均得到 GAE(γ,λ)=Σ(γλ)^l δ_{t+l}，λ 在偏差与方差之间折中，λ<1 只在价值函数不准时引入偏差。
matters: M09 的 GAE 公式、倒推递归与 γ/λ 的取值区间都以它为准；第 3 课 PPO 的优势估计直接用到。
```

```
id: robbins1951
title: A Stochastic Approximation Method
a: Herbert Robbins, Sutton Monro (1951), The Annals of Mathematical Statistics 22(3):400–407, doi:10.1214/aoms/1177729586
url: https://doi.org/10.1214/aoms/1177729586
finding: 用噪声观测求回归函数的根，步长满足 Σa_n = ∞、Σa_n² < ∞ 时证明均方收敛（按 Lai & Yuan 的转述）。
matters: M07 讲 SGD 的学习率条件与“常数学习率为什么需要衰减阶段”时的原始出处（原文未读）。
```

```
id: lai2021sa
title: Stochastic Approximation: from Statistical Origin to Big-Data, Multidisciplinary Applications
a: Tze Leung Lai, Hongsong Yuan（预印本，页眉 “Submitted to Statistical Science”，文件日期 2019-12-09）
url: https://tzelai.ckirby.su.domains/pubs/2021_StochApprox.pdf
finding: 综述随机逼近的起源，给出 Robbins–Monro 递推的步长条件及其作用（Σa_n² 控制噪声、Σa_n 保证到达根）。
matters: M07 用它作 Robbins–Monro 条件的可读二手出处，因为原刊页面对脚本不可访问。
```

```
id: polyak1964
title: Some Methods of Speeding up the Convergence of Iteration Methods
a: B. T. Polyak (1964), USSR Computational Mathematics and Mathematical Physics 4(5):1–17, doi:10.1016/0041-5553(64)90137-5
url: https://doi.org/10.1016/0041-5553(64)90137-5
finding: 提出重球（动量）法，在极小点附近取 μ=(√R−1)/(√R+1) 时迭代次数比最速下降少约 √R 倍（按 Sutskever et al. 2013 的转述）。
matters: M07 动量一节的原始出处；课程借它说明 Adam 的一阶矩与 Muon 的动量缓冲从哪里来（原文未读）。
```

```
id: sutskever2013
title: On the Importance of Initialization and Momentum in Deep Learning
a: Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton (2013), ICML 2013, JMLR W&CP 28
url: https://proceedings.mlr.press/v28/sutskever13.pdf
finding: 用统一记号写出经典动量与 Nesterov 加速梯度（式 1–4），并说明两者只差速度更新时梯度的取点。
matters: M07 的动量公式与“Nesterov 技巧”的出处；Muon 与 V4 算法 1 都用 Nesterov 式动量。
```

```
id: kingma2015adam
title: Adam: A Method for Stochastic Optimization
a: Diederik P. Kingma, Jimmy Ba (2015), ICLR 2015, arXiv:1412.6980v9
url: https://arxiv.org/abs/1412.6980v9
finding: 提出用一阶、二阶矩的指数滑动平均并做偏差修正的自适应优化器，默认 α=0.001、β1=0.9、β2=0.999、ε=1e-8。
matters: M07 的 Adam 算法与偏差修正推导以 v9 为准；课程要特别提醒 v1 的参数化不同（E16）。
```

```
id: loshchilov2019adamw
title: Decoupled Weight Decay Regularization
a: Ilya Loshchilov, Frank Hutter (2019), ICLR 2019, arXiv:1711.05101v3
url: https://arxiv.org/abs/1711.05101
finding: 证明权重衰减与 L2 正则只在 SGD 下等价（需 λ'=λ/α），在自适应方法下不等价，并提出把衰减从梯度更新中解耦的 AdamW。
matters: M07 讲“为什么 LLM 用 AdamW 而不是 Adam+L2”，以及论文与 PyTorch 在衰减是否乘学习率上的差别（E18）。
```

```
id: liu2025muon
title: Muon is Scalable for LLM Training
a: Jingyuan Liu, Jianlin Su, Xingcheng Yao, et al. (2025), arXiv preprint, arXiv:2502.16982v1
url: https://arxiv.org/abs/2502.16982
finding: 给 Muon 加上 AdamW 式权重衰减，并把更新按 0.2√max(A,B) 缩放使 RMS 与 AdamW 一致，从而直接复用 AdamW 超参数，在计算最优训练下约 2 倍于 AdamW 的效率。
matters: M07 讲 Muon 的形状缩放（引理 1）与权重衰减时的主出处，DeepSeek-V4 明言沿用它的做法。
```

```
id: bernstein2024anthology
title: Old Optimizer, New Norm: An Anthology
a: Jeremy Bernstein, Laker Newhouse (2024), arXiv preprint, arXiv:2409.20325v2
url: https://arxiv.org/abs/2409.20325
finding: 把 Adam、Shampoo、Prodigy 解释为某种范数下的最速下降；谱范数下的最速下降方向是 UV^T，离 G 最近的半正交矩阵也是 UV^T。
matters: M07 用来回答“为什么要正交化”：谱范数下的最速下降方向就是 UV^T，这给 Newton–Schulz 一个几何意义（论文的表述对象是无累积的 Shampoo，正文未提 Muon；对应到 Muon 要同时引 Jordan 博客，见 C53）。
```

```
id: goyal2017
title: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
a: Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He (2017), technical report, arXiv:1706.02677v2
url: https://arxiv.org/abs/1706.02677
finding: 提出线性缩放规则（batch 乘 k、学习率乘 k）与 5 个 epoch 的渐进 warmup，把 ResNet-50 的 batch 扩到 8k 而精度不降。
matters: M07 讲 batch 与学习率的关系、warmup 的来历；它的 §4.1 还给了 M10 用的 allreduce 流量 2(p−1)/p·b 与实测比较。
```

```
id: mccandlish2018
title: An Empirical Model of Large-Batch Training
a: Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team (2018), arXiv preprint, arXiv:1812.06162v1
url: https://arxiv.org/abs/1812.06162
finding: 定义梯度噪声尺度 B_noise=tr(HΣ)/(G^T H G) 及其简化 B_simple=tr(Σ)/|G|²，并由此推出步数与样本数的双曲线权衡和临界 batch。
matters: M07 讲“batch 大到多少不再划算”时的数学模型，为第 6 课 DeepSeek LLM 的 batch/学习率 scaling law 铺垫。
```

```
id: smith2018bayesian
title: A Bayesian Perspective on Generalization and Stochastic Gradient Descent
a: Samuel L. Smith, Quoc V. Le (2018), ICLR 2018, arXiv:1710.06451v3
url: https://arxiv.org/abs/1710.06451
finding: 把 SGD 看作带噪声尺度 g≈εN/B 的随机微分方程，推出最优 batch 与学习率、训练集大小成正比。
matters: M07 用来提醒“噪声尺度”一词有两种含义，避免与 McCandlish 的 B_noise 混淆（E26）。
```

```
id: hu2024minicpm
title: MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies
a: Shengding Hu, Yuge Tu, Xu Han, et al. (2024), arXiv:2404.06395v3
url: https://arxiv.org/abs/2404.06395
finding: 提出 Warmup-Stable-Decay 学习率调度，发现衰减阶段损失骤降、约 10% token 的衰减就够，并可用稳定期检查点低成本测 scaling law。
matters: M07 讲 WSD 的出处；DeepSeek-V4/V4.1 的“warmup + 长稳定 + 末段 cosine 衰减”就是这种形状。
```

```
id: williams2009roofline
title: Roofline（Crossref 记录的期刊版题名；技术报告题名为 Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures）
a: Samuel Williams, Andrew Waterman, David Patterson (2009), Communications of the ACM 52(4):65–76, doi:10.1145/1498765.1498785（本文读的是 2008 年技术报告 UCB/EECS-2008-134）
url: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html
finding: 以“可达性能 = min(峰值算力, 带宽 × 运算强度)”给出屋顶线模型，运算强度按每字节 DRAM 流量计。
matters: M10 的屋顶线与 ridge point 的出处，同时用来纠正“operational vs arithmetic intensity”的术语混用（E39）。
```

```
id: thakur2005
title: Optimization of Collective Communication Operations in MPICH
a: Rajeev Thakur, Rolf Rabenseifner, William Gropp (2005), International Journal of High Performance Computing Applications 19(1):49–66, doi:10.1177/1094342005051521
url: https://doi.org/10.1177/1094342005051521
finding: 在 α+nβ(+nγ) 代价模型下分析 allgather、broadcast、all-to-all、reduce-scatter、reduce、allreduce 的多种算法，给出环、递归倍增、Rabenseifner 等算法的时间公式。
matters: M10 的集合通信代价表（§3.4）全部取自这里，服务第 12 课的 all-to-all 与数据并行梯度同步。
```

```
id: patarasuk2009
title: Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations
a: Pitch Patarasuk, Xin Yuan (2009), Journal of Parallel and Distributed Computing 69(2):117–124, doi:10.1016/j.jpdc.2008.09.002
url: https://doi.org/10.1016/j.jpdc.2008.09.002
finding: 证明 all-reduce 中至少一个进程要通信约 2(N−1)/N 倍的数据量，并给出在树拓扑上达到这一下界的环形算法。
matters: M10 用它说明环形 allreduce 的 2(p−1)/p·n 不是“碰巧”，而是带宽意义下的最优。
```

```
id: goldberg1991
title: What Every Computer Scientist Should Know About Floating-Point Arithmetic
a: David Goldberg (1991), ACM Computing Surveys 23(1):5–48, doi:10.1145/103162.103163
url: https://doi.org/10.1145/103162.103163
finding: 系统讲解舍入误差、ulp 与相对误差、IEEE 754（1985）的格式参数（表 D-1）与特殊值，并把 machine epsilon 定义为 (β/2)β^{-p}。
matters: M08 的单/双精度参数与“两种 machine epsilon 约定”（E33）的出处。
```

```
id: ieee754
title: IEEE Standard for Floating-Point Arithmetic (IEEE Std 754-2008 / 754-2019)
a: IEEE (2008, 2019), doi:10.1109/IEEESTD.2008.4610935; doi:10.1109/IEEESTD.2019.8766229
url: https://doi.org/10.1109/IEEESTD.2019.8766229
finding: 定义 binary16/32/64/128 等交换格式、舍入规则与特殊值（本文未读原文，只核对了 DOI 与题名）。
matters: M08 提到“IEEE 754”时的规范出处；具体参数在课程里引 Goldberg、Kalamkar 与 numpy，并注明标准原文未读。
```

```
id: kalamkar2019bf16
title: A Study of BFLOAT16 for Deep Learning Training
a: Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, et al. (2019), arXiv preprint, arXiv:1905.12322v3
url: https://arxiv.org/abs/1905.12322
finding: 系统评估 bfloat16（1/8/7 位，动态范围与 FP32 相当）在多类模型上的混合精度训练，BF16 输入加 FP32 累加无需调超参即可达到 FP32 精度。
matters: M08 的 BF16/FP16/FP32 参数表（表 1）出处；课程同时要注明 bf16 源自 Google、表中次正规 “N/A” 是实现约定（E31、E32）。
```

```
id: micikevicius2022fp8
title: FP8 Formats for Deep Learning
a: Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, et al. (2022), arXiv preprint, arXiv:2209.05433v2
url: https://arxiv.org/abs/2209.05433
finding: 提出 E4M3 与 E5M2 两种 8 位浮点交换格式：E5M2 遵循 IEEE 惯例，E4M3 去掉无穷、只留一种 NaN 把最大值扩到 448。
matters: M08 的 FP8 参数表与特殊值编码的唯一一手出处，服务第 12 课的 FP8 训练。
```

```
id: rouhani2023mx
title: Microscaling Data Formats for Deep Learning
a: Bita Darvish Rouhani, Ritchie Zhao, Ankit More, et al. (2023), arXiv preprint, arXiv:2310.10537v3
url: https://arxiv.org/abs/2310.10537
finding: 定义 MX 块（k 个元素共享一个 E8M0 缩放，值为 X·P_i），列出 MXFP8/6/4 与 MXINT8（块大小 32），并给出一种从 FP32 转换的算法。
matters: M08 讲块缩放与 E8M0 的出处（OCP 规范原文未读，以此代替）；V4 的 indexer 与 V3.1 的 UE8M0 都与之相关。
```

```
id: nvidia2025nvfp4
title: Pretraining Large Language Models with NVFP4
a: NVIDIA, Felix Abecassis, et al. (2025), arXiv preprint, arXiv:2509.25149v2
url: https://arxiv.org/abs/2509.25149
finding: 描述 NVFP4（E2M1 元素、16 元素块、E4M3 块缩放加 FP32 张量缩放），列出 E2M1 的全部取值，并说明 MXFP4 的 2 的幂缩放会浪费 FP4 格点。
matters: M08 讲 FP4 取值与两级缩放的出处；V4.1 的 FP4 KV cache 明言“沿用 NVFP4 但去掉全局缩放”。
```

### 9.2 制品（kind: artefact）

```
id: jordan2024muon
title: Muon: An optimizer for hidden layers in neural networks
a: Keller Jordan (2024-12-08), blog post（引用条目列出合作者 Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein）
url: https://kellerjordan.github.io/posts/muon/
kind: artefact
finding: 定义 Muon：对 SGD 动量更新做 5 步 Newton–Schulz 近似正交化，系数 (3.4445, −4.7750, 2.0315) 是在 φ^N(x) 落在 [0.7, 1.3] 的约束下使 φ'(0) 最大化得到的，FLOP 开销至多 Tm/B。
matters: M07 的 Muon 定义、系数来历与开销分析的原始出处；没有对应论文，只能按制品引用。
```

```
id: muon-code
title: KellerJordan/Muon — muon.py
a: Keller Jordan et al. (2024–2025), GitHub repository (master branch, read 2026-09-24)
url: https://github.com/KellerJordan/Muon
kind: artefact
finding: 参考实现：BF16 下做 Frobenius 归一化与 5 步五次 NS，更新按 √max(1, rows/cols) 缩放，EMA 形式的 Nesterov 动量，默认 lr 0.02、动量 0.95、衰减 0。
matters: M07 讲“论文与代码的差别”时的对照（形状缩放、动量写法、奇异值落在约 0.5–1.5 的注释）。
```

```
id: pytorch-adamw
title: torch.optim.AdamW (torch/optim/adamw.py)
a: PyTorch contributors, GitHub repository (main branch, read 2026-09-24)
url: https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
kind: artefact
finding: 默认 lr 1e-3、betas (0.9, 0.999)、eps 1e-8、weight_decay 1e-2，衰减写作 θ ← θ − lr·λ·θ。
matters: M07 用它说明实践中的“weight_decay”乘了学习率，与 AdamW 论文算法 2 的写法不同（E18）。
```

```
id: microxcaling
title: microsoft/microxcaling — mx/formats.py, mx/mx_ops.py
a: Microsoft (MX paper authors), GitHub repository (main branch, read 2026-09-24)
url: https://github.com/microsoft/microxcaling
kind: artefact
finding: MX 格式的仿真库：给出 FP6 E3M2 / E2M3 与 FP4 E2M1 的 emax 与最大值（28、7.5、6），注明它们没有 NaN/Inf，共享指数限制在 [−127, 127]。
matters: OCP 规范无法访问时，M08 的 FP6/FP4 参数表以它和 MX 论文互相印证。
```

```
id: schulman2020kl
title: Approximating KL Divergence
a: John Schulman (2020-03-07), blog post
url: http://joschu.net/blog/kl-approx.html
kind: artefact
finding: 给出 KL 的三个蒙特卡洛估计 k1、k2、k3，其中 k3=(r−1)−log r 既无偏又恒非负，并用两组高斯实验比较偏差与标准差。
matters: M05 讲“KL 的估计”时的出处，直接服务第 8 课 GRPO 的 KL 项。
```

```
id: numpy-finfo
title: numpy.finfo (NumPy 1.26.4)
a: NumPy developers, library documentation / local run
url: https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
kind: artefact
finding: 给出 float16/32/64 的 eps、epsneg、max、tiny（最小正规数）与 smallest_subnormal。
matters: M08 让读者自己复核格式参数的工具，也是“eps = 2^{-(p-1)}”这一约定的实际来源（E33）。
```

```
id: ocp-mx-spec
title: OCP Microscaling Formats (MX) Specification Version 1.0
a: Open Compute Project (2023), specification（题名、版本与年份据 NVFP4 论文参考文献）
url: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
kind: artefact
finding: MX 格式的规范性定义（E8M0 缩放、块大小 32、各元素格式的编码）——本文未能读到原文（HTTP 403），内容经 MX 论文与仿真库间接核对。
matters: M08 讲 MXFP4/UE8M0 时应引用的规范本体；写课前最好由人工下载核对 E8M0 的 NaN 编码与 FP6/FP4 的偏置。
```

---

## 核查记录（对抗式复核，2026-09-24）

复核方法：挑出最可能被写课者照抄的约 30 组论断（规格数字、转录的公式、标题性结论、日期、归属），逐条回原始来源重查，不依赖本文已有的转述。
arXiv 论文用 `arxiv.py text/grep`（先在页头确认版本号）；Adam 两个版本、Sinkhorn–Knopp 1967、Williams 1992、Sutton 2000、Thakur 2005、Patarasuk–Yuan 2009、MML 书、Stewart 1993
下载 PDF 用 pypdf 抽文本；Muon 博客、`muon.py`、PyTorch `adamw.py`、Schulman KL 博客、Goldberg（Oracle 重排版）、Eckart–Young 摘要（Springer）用 curl 抓原页。
§4 的数字用 Python 3.11 + numpy 1.26 全部重算，脚本在 session scratch 目录（recompute1–5.py、ns_orbit.py、sim.py）。

### 改了什么（K 编号在正文里有回指）

- **K1 N4、W29：IEEE 式 E4M3 最大值的算式写错。** `1.75*2**7` 等于 224，不是 240。正确算式是 $1.875\times2^7=240$（指数码 1110、尾数 111）。448 与 240 这两个数本身对（`fp8` 表 1、§3.1）。W29 里“再砍掉 ±NaN 对称可得 480”按 `fp8` §3.1 改成“零与 NaN 都只留一个编码”。
- **K2 N58、W21：NS 长期行为不是 2 周期。** 用 Jordan 系数从多个起点迭代 2000 步，都收敛到 4 周期 0.6819 → 1.1343 → 0.7529 → 1.0469。旧版的“0.7529 ↔ 1.1343”只对了一半：$\varphi_J(1.1343)=0.7530$，但 $\varphi_J(0.7529)=1.0469$。区间 [0.6818, 1.1344]、$\varphi_J(1)=0.701$、峰值 1.2024 @ 0.5545、不动点 0.8680 / 1.2637、混合 8+2 步后的 [1.000000, 1.000538] 都复算一致。新增 E44。
- **K3 N17：NVFP4 论文没有印 2688。** 附录 B.1 只写 $s_{enc}=6\cdot448/amax_x$；“448×6=2688”只出现在 `dsv41` §2.4.4。
- **K4 §2.10、§3.4：Patarasuk–Yuan 引理 4 的量词。** 原文是 “there exists at least one process”，旧版写成“每进程至少”。已改，新增 E45。C75 原本写对了。
- **K5 F07-1、N50：步长乘子的最低点。** $\sqrt{1-0.999^t}/(1-0.9^t)$ 在整数 $t$ 上的最小值在 $t=12$（0.1522），不在 $t=10$（0.1532）。
- **K6 N54、F07-2：“小 15 倍”不稳。** 用同样设置独立重做模拟，前三个点一致（9.58e-4 / 9.52e-4 / 1.03e-4），$s=10$ 一点得 1.47e-5，约小 7 倍，旧版是 6.59e-6。结论方向不变，已改成只引数量级。
- **K7 C74、§3.4：Thakur 的 “Ring Algorithm” 不是纯环。** §5.3 的 reduce-scatter 用成对交换（§4.4），allgather 才走环；两者代价公式相同，所以 $2(p-1)\alpha+2\frac{p-1}pn\beta+\frac{p-1}pn\gamma$ 可以照用，但归属要写准。
- **K8 C53 与参考卡：Bernstein–Newhouse 讲的是 Shampoo。** v2 正文没有 “Muon” 一词。命题 4、5 的对象是无累积的 Shampoo；Muon 与它的对应来自 Jordan 博客（引 Anil 2024b）。属于“相邻命题替换”，已加标注。
- **K9 E31、E32：bfloat16。** `bf16` 表 1 的 “N/A” 论文没给理由，改标【笔者推断】；E32 删去原文没有提到的 “TPU”。
- **K10 N47、W7：** $R(30°)R(45°)-R(75°)$ 的误差是浮点舍入，本次复算 1.7e-16（旧版 1.1e-16），改写成“约 $10^{-16}$”。
- **K11 C71：** 补上 `dsv41` §2.4.4 同段“RMSNorm 权重最大约 11”这句，并说明论文的上界推理有缺口（新增 §8 第 29 条）。
- **K12 C75：** Goyal 的“快 3 倍”补上原文的适用范围（缓冲区至多约一百万元素）。
- **K13 §8 第 28 条已解决：** `mml2020` 定理 4.25 在 §4.6 “Matrix Approximation”，pp. 131–132。

### 核对无误（未改动）

- **SVD 归属。** `eckart1936` 摘要原句 “always has a solution which is usually unique”（Springer）。`stewart1993` p. 561 “fundamental theorem”，p. 563 “often (and incorrectly) called the Eckart-Young theorem”，Mirsky “minimizing matrix in any unitarily invariant norm”，§6 式 (6.2)(6.3)。`mml2020` 定理 4.25 标题 “Eckart-Young Theorem (Eckart and Young, 1936)”，式 (4.94) 约束写 rk(B)=k，用谱范数。
- **Sinkhorn–Knopp 1967。** p. 344 THEOREM（支撑 / 完全支撑 / 完全不可分解三条），p. 343 定义，p. 345 Birkhoff 凸包原句，p. 347 非正对角线元素趋于 0，参考文献 Birkhoff “147-150”，收稿 1966-02-20。
- **mHC（2512.24880v2）。** 式 (8)(9)，$t_{max}=20$，“bounded by 1”（§4.1），6.7%（§1、§4.3），§5.4 “nearly 3000”。
- **DeepSeek-V4（2606.19348v1）。** 算法 1、式 (28)，混合 NS 8+2 步，“stabilize the singular values precisely at 1”；§4.2.2：AdamW 0.9 / 0.95 / 1e-20 / 0.1，Muon 0.95 / 0.1 / RMS 0.18，75.5M batch，2000 步 warmup，2.7e-4 → 2.7e-5，32T tokens；§2.2 式 (8)。
- **DeepSeek-V4.1（2609.19969v1）。** §2.4.4：E2M1 加每 16 通道一个 E4M3、无全局缩放、2688、≈22.6、观测约 10；§2.5 算法 1 与式 (7)，$\gamma=0.18$；§4.2.2：$K=11$、$\tau=10^{-3}$、$\varepsilon=10^{-20}$、100.6M batch，2.6e-4 保持到 28T，28T–40T cosine 到 2.6e-5，40T–45T 保持。
- **Adam。** v9 PDF：算法 1 默认值，$\beta\in[0,1)$，§2.1 步长界，§3 式 (1)–(4)，§3 末遗留的 “small value of β2”，§6.4 的 “close to 1”。v1 PDF：$\beta\in(0,1]$，$m_t\leftarrow\beta_1g_t+(1-\beta_1)m_{t-1}$，修正项 $1-(1-\beta_1)^t$，默认 0.0002 / 0.1 / 0.001 / 1e-8。
- **AdamW（1711.05101v3）。** 算法 2 第 6、11、12 行逐符号核对；命题 1–3；v1、v2 的标题 “Fixing Weight Decay Regularization in Adam” 与提交日期（arXiv abs 页）。PyTorch `AdamW` 默认值与 docstring 里的 $\theta_t\leftarrow\theta_{t-1}-\gamma\lambda\theta_{t-1}$。
- **Muon。** 博客（2024-12-08）：系数、$[0.7,1.3]$、$2(2nm^2+m^3)$、$6nmB$、0.7% 与 0.5%。`muon.py`：“Uniform(0.5, 1.5)”，`lerp_` 形式的 EMA 动量，`max(1, rows/cols)**0.5`，默认 lr 0.02 / wd 0 / 动量 0.95，bf16。Moonlight（2502.16982v1）：式 (1)–(4)，引理 1，$N=5$，动量 0.95，“∼2×”。
- **浮点格式。** `fp8` 表 1 全部条目，§3.1 的 17→18 binade、240、480，§3 的推荐用法。`bf16` 表 1、“8 bits of mantissa”、DistBelief/TensorFlow。`mx` 表 1、算法 1 与三条说明。`nvfp4` §2 的 E2M1 取值、UE8M0 范围、Mishra 的向上取整，附录 B.1–B.4。Goldberg 表 D-1 与 $\epsilon=(\beta/2)\beta^{-p}$。
- **RL。** Williams §4 定义、REINFORCE 缩写、定理 1。Sutton 2000 定理 1 与其后一段的 Marbach–Tsitsiklis 归属、$d^\pi$ 的折扣定义。GAE（1506.02438v6）式 (16)，§6.3.1 cart-pole 区间，§7 的 [0.9, 0.99]，Kimura & Kobayashi、Wawrzyński。Schulman KL 博客的两张表。
- **学习率与 batch。** Goyal（1706.02677v2）线性缩放原句、~8k、$0.1\cdot kn/256$、3.2、$2\frac{p-1}pb$、$2\log_2p$ 与 $2(p-1)$ 步。McCandlish（1812.06162v1）式 (2.5)–(2.9)(2.11)(2.12) 逐符号核对。Smith & Le 的 $g=\epsilon(N/B-1)$ 与 $\epsilon N/(B(1-m))$。MiniCPM（2404.06395v3）式 (1)，§4.3 “10%”，§6.2 “$0.5^{(s-S)/T}$，T 为 5000 步（20B tokens）”，E19 的解读成立。
- **通信。** Thakur §4.1 环形 allgather、§4.4 成对交换、§5.1 减半倍增、§5.3。Patarasuk–Yuan 引理 3–5。
- **§4 的数字。** N1–N80 中凡能从公开参数重算的都重算了（N25、N28、N44、N45 未重跑，它们依赖随机抽样且只作示意）。除 K1、K2、K5、K6、K10 外全部一致，包括 N22 的 96.8% / 6.2% / 10.0% 与 N72 的方差 544.97 / 2.33 / 0.76（seed 0 完全复现）。
