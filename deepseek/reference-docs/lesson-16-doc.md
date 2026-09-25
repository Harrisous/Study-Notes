# 第 16 课研究笔记：残差流的新几何——Hyper-Connections 与 mHC

> 本文件是 `deepseek/src/lesson-16-mhc.md` 的研究底稿，只收“核对过的引文、复算过的数字、勘误、未能核实清单”，
> 不是讲义正文。建立于 2026-09-24。引文保留英文原文（为控制篇幅与版权风险，只摘关键短语，每条注明节号 /
> 表号 / 式号 / 所读版本），其余用中文转述。所有派生数字都在 Python 里复算过（第 4 节给出表达式）。
> 凡是笔者自己的推理，一律标注“笔者推导”或“笔者推测”。图中数值凡是从论文 SVG 矢量坐标里“数字化”读出的，
> 都标注“数字化”，它们不是论文印刷的数字。
>
> 本课的时间线：HC（ByteDance，arXiv 2024-09，ICLR 2025）→ mHC（DeepSeek，arXiv 2025-12-31）→
> DeepSeek-V4 把 mHC 用进 1.6T/284B 模型（2026-04）→ DeepSeek-V4.1-Flash 改成 Single-Pass mHC（2026-09）。

---

## 0. 覆盖范围与所读版本

### 0.1 论文

| 短键 | 标题 | 作者 | arXiv / 版本 | 发表状态 | 日期 | 本课用途 |
|---|---|---|---|---|---|---|
| `zhu2025hc` | Hyper-Connections | Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou（ByteDance Seed-Foundation-Model Team） | 2409.19606，**读 v3**；另读 v1 做对照 | **ICLR 2025**（proceedings.iclr.cc 页面确认：“International Conference on Learning Representations 2025 (ICLR 2025)”；V4 论文参考文献写作 “ICLR 2025, Singapore, April 24-28, 2025”） | v1 2024-09-29；v2 2024-11-28；v3 2025-03-18 | HC 矩阵、静态/动态 HC、初始化、Pre/Post-Norm 的 HC 表示、序列-并行对偶、1B/7B/MoE 结果、附录 B 的参数/FLOPs/显存账 |
| `xie2026mhc` | mHC: Manifold-Constrained Hyper-Connections | Zhenda Xie\*†, Yixuan Wei\*, Huanqi Cao\*, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang（DeepSeek-AI；\* core contributors，† 通讯作者 xie.zhenda@deepseek.com） | 2512.24880，**读 v2**；v1 全文 diff 过 | 预印本（技术报告，无会议信息，arXiv comments 字段为空） | v1 2025-12-31；v2 2026-01-05 | 全部 mHC 公式、不稳定性分析、Sinkhorn 投影、基础设施、实验、附录表 5 |
| `dsv4` | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | DeepSeek-AI | 2606.19348v1 | 预印本 | v1 提交记录 2026-04-26（见 §8 关于编号的疑点） | §2.2 mHC 在 V4 中的写法（A/B/C 记号）、§2.4 优化器分工、§3.3 确定性、§3.4.2 实现、§4.2.1 超参数 |
| `dsv41` | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | DeepSeek-AI | 2609.19969v1 | 预印本 | 2026-09-17 | §2.4.1 Single-Pass mHC 与 Mega-mHC 访存账、§4.2.1 超参数 |
| `he2016identity` | Identity Mappings in Deep Residual Networks | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun | 1603.05027v3 | ECCV 2016（arXiv comments：“ECCV 2016 camera-ready”） | v1 2016-03-16 | 式 (4)(5)(7)(8)：恒等捷径 vs 标量缩放捷径 |
| `xiong2020preln` | On Layer Normalization in the Transformer Architecture | Ruibin Xiong 等 10 人 | 2002.04745v2 | ICML 2020（arXiv journal 字段） | v1 2020-02-12 | 定理 1：Post-LN / Pre-LN 最后一层梯度的量级 |
| `liu2020admin` | Understanding the Difficulty of Training Transformers | Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, Jiawei Han | 2004.08249v3 | EMNLP 2020 | v1 2020-04-17 | Pre-LN 稳但弱、Post-LN 强但易发散的实验数字 |
| `vaswani2017` | Attention Is All You Need | Vaswani 等 8 人 | 1706.03762v7 | NeurIPS 2017 | 2017-06-12 | 只用来核对：原版 Transformer 是 Post-LN |
| `sinkhorn1967` | Concerning nonnegative matrices and doubly stochastic matrices | Richard Dennis Sinkhorn, Paul Joseph Knopp | 无 arXiv | Pacific Journal of Mathematics **21(2): 343–348**, 1967, doi 10.2140/pjm.1967.21.343（msp.org 页面确认） | 1967 | 交替行列缩放收敛的充要条件 |
| `sinkhorn1964` | A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices | Richard Sinkhorn | 无 arXiv | Annals of Mathematical Statistics **35(2): 876–879**, 1964-06, doi 10.1214/aoms/1177703591（Project Euclid 页面确认） | 1964 | 正矩阵情形的存在唯一性（mHC 用 exp 保证正，所以真正用到的是这一篇的情形） |
| `birkhoff1946` | Tres observaciones sobre el algebra lineal | Garrett Birkhoff | 无 | Univ. Nac. Tucumán Revista Ser. A **5: 147–151**, 1946（**只从二手索引核对**，见 §8） | 1946 | Birkhoff–von Neumann 定理出处 |
| `mena2018gumbel` | Learning Latent Permutations with Gumbel-Sinkhorn Networks | Gonzalo Mena, David Belanger, Scott Linderman, Jasper Snoek | 1802.08665v1 | ICLR 2018（arXiv journal 字段） | 2018-02-23 | 定理 1：Sinkhorn 算子 = 熵正则化的线性指派；给 mHC 的 “entropically project” 一个精确含义 |
| `tilelang2025` | TileLang: A Composable Tiled Programming Model for AI Systems | Lei Wang 等 11 人 | 2504.17577v2 | 预印本 | 2025-04-24 | mHC 大部分核用它写（只核了元数据） |
| `dsv3` | DeepSeek-V3 Technical Report | DeepSeek-AI | 2412.19437v2 | 预印本 | 2024-12-27 | DualPipe 的出处（mHC 引作 Liu et al., 2024b）；本笔记未重读其内容 |
| `guo2026ohc` | oHC: Orthogonal Hyper-Connections on SO(4) via Quaternions | Haoqiang Guo 等 8 人 | 2609.02672v1 | 预印本（未经同行评审） | 2026-09-02 | 后续工作；**只读了摘要**，用来佐证“双随机混合会耗尽流间差异”的判断 |

### 0.2 制品（无论文，`kind: artefact`）

| 短键 | 内容 | 读取方式 / 时间 |
|---|---|---|
| `hf-v4flash-config` | `huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json` | 2026-09-24 原样下载 |
| `hf-v4pro-config` | `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json` | 同上 |
| `hf-v41flash-config` | `huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json` | 同上 |
| `hf-v4flash-code` | V4-Flash 仓库 `inference/model.py`（类 `Block.hc_pre/hc_post`、`ParallelHead.hc_head`、`Transformer.forward`）与 `inference/kernel.py`（`hc_split_sinkhorn_kernel`，TileLang 写的） | 同上，逐行读过 |
| `hf-v41flash-code` | V4.1-Flash 仓库 `inference/model.py`（`Block.hc_mixes/hc_pre/hc_post/forward`、`make_identity_pre_mix`）与 `inference/kernel.py` | 同上，逐行读过 |
| `iclr2025-hc-page` | proceedings.iclr.cc 上 HC 的摘要页 | 2026-09-24 |

### 0.3 版本差异（全文 diff 结果）

- **mHC v1 → v2**：正文、公式、表格、图完全相同；唯一差别是作者列表在 v2 **加入了 Kuai Yu**（v1 19 人，v2 20 人）。
  所以 v1 与 v2 的一切数字可以互换引用，但作者名单以 v2 为准。
- **HC v1 → v3**：§2 的所有公式（式 (1)–(14)）逐字相同；相对 v1，**从 v2（2024-11-28）起**新增了**附录 A**（带 HC 的 Transformer 结构图，图 8）与**附录 B**（参数量、FLOPs、显存账，含表 7–9），v3 保留
  （核查时对 v2 的 arXiv HTML 检索：附录 A/B 标题、表 1 新标题、表 7 的 0.0000026 / 0.0013124 两处笔误都已在 v2 出现；v2 与 v3 未做全文 diff），
  原附录因此整体后移两个字母（v1 的附录 A “MoE 1B/7B” 在 v3 变成附录 C，v1 附录 J “1B Model Experiments” 变成 v3 附录 L）；v1 表 1 标题为 “Performance comparison of
  different expansion rates on 500 B tokens.”，v3 改为 “Ablation study on expansion rates n with training on 500 B tokens.”。
  实验数字在 v1 与 v3 之间未发现改动（只抽查了表 1–6）。引用参数/显存账时必须引 v2 及以后的版本（本笔记引 v3），或 ICLR 版。

---

## 1. 核心论断与原文引文

编号 C1–C40。每条：论断（中文）→ 原文短摘（英文）→ 位置。

### 1A. 残差连接与恒等映射（背景）

- **C1｜恒等捷径让任意两层之间“信号直通”。** He 等把残差单元递推得到 $\mathbf x_L=\mathbf x_l+\sum_{i=l}^{L-1}\mathcal F(\mathbf x_i,\mathcal W_i)$，
  并说明反向梯度含一项不经过任何权重层的直通项。
  原文：信号可以 “directly propagated from any unit to another, both forward and backward”
  ——`he2016identity` v3，§2 “Discussions”，紧跟式 (4)(5)。
- **C2｜把捷径换成标量缩放 $\lambda_l$，乘积会指数爆炸或消失。**
  原文：“if λ_i > 1 for all i, this factor can be exponentially large”（后半句对 λ_i < 1 说 “exponentially small and vanish”）。
  ——`he2016identity` v3，§3，式 (7)(8) 之后。**这正是 mHC 式 (4) 的标量版本**（见 §2.6）。
- **C3｜对捷径做乘性操作会伤害优化。**
  原文：“Multiplicative manipulations … on the shortcuts can hamper information propagation”（省略号处列举了缩放、门控、1×1 卷积、dropout）
  ——`he2016identity` v3，§3.2 “Discussions”（§3 末）。常数缩放 λ=0.5 的 ResNet-110 测试误差 12.35%（表 1，F 同时乘 0.5 的那一行）。
- **C4｜原版 Transformer 是 Post-LN。** 原文：“the output of each sub-layer is LayerNorm(x + Sublayer(x))”
  ——`vaswani2017` v7，§3.1。（用于 §7 勘误 E10。）
- **C5｜Post-LN 最后一层梯度与深度无关、Pre-LN 随 $\sqrt L$ 缩小。** Xiong 定理 1：Post-LN 最后一个 FFN 的梯度
  $\|\partial\tilde{\mathcal L}/\partial W^{2,L}\|_F\le\mathcal O(d\sqrt{\ln d})$，Pre-LN 为 $\mathcal O(d\sqrt{\ln d/L})$；
  原因是 Pre-LN 中 “the scale of the input to the final layer normalization is linear in L”。
  ——`xiong2020preln` v2，§3.3 定理 1 及其后一段。定理的前提：$\|x^{post,5}_{L,i}\|^2_2$ 与 $\|x^{pre}_{L+1,i}\|^2_2$ 是 $(\epsilon,\delta)$-bounded，
  结论以概率 $\ge 0.99-\delta-\frac{\epsilon}{0.9+\epsilon}$ 成立，且只针对初始化时刻。
- **C6｜Pre-LN 稳但弱、Post-LN 强但易发散（实测）。**
  原文：“Pre-LN converges in all 15 settings, and Post-LN diverges in 7 out of 15 settings”，后半句：Post-LN 收敛时
  “outperforms Pre-LN in 7 out of 8 settings”。——`liu2020admin` v3，§3 开头。同节还说 Post-LN 有梯度消失而 Pre-LN 没有，
  但梯度消失 “is not the direct reason” 导致 Post-LN 不稳定。
- **C7｜HC 把这对矛盾叫作“跷跷板”。** 原文：“the seesaw effect between gradient vanishing and representation collapse”
  ——`zhu2025hc` v3，摘要；§1 第二段展开：“like two ends of a seesaw”。

### 1B. Hyper-Connections（Zhu et al.）

- **C8｜HC 的核心：把输入复制成 $n$ 份（扩展率 expansion rate），用一个 $(n+1)\times(n+1)$ 矩阵决定每层怎么读、怎么写、流之间怎么混。**
  矩阵见 §2.2 式 (HC-1)；初始隐藏矩阵 $\mathbf H^0=(\mathbf h^0\ \cdots\ \mathbf h^0)^\intercal\in\R^{n\times d}$；最后一层把 $n$ 行 **求和** 再进最终归一化与 unembedding。
  ——`zhu2025hc` v3，§2.1。
- **C9｜深度连接 / 宽度连接的分解。** $\mathcal{DC}=(\mathbf B;\ \mathrm{diag}(\mathbf A_r))\in\R^{2\times n}$，$\mathcal{WC}=(\mathbf A_m\ \mathbf A_r)\in\R^{n\times(n+1)}$。
  ——`zhu2025hc` v3，式 (6)(7)，图 2(c)(d)。
- **C10｜动态 HC（DHC）= 静态矩阵 + 输入相关的小扰动。** 先归一化，再线性变换，再 tanh，再乘一个“小的初值可学”的缩放因子。
  原文：“apply the tanh activation function after it”，再乘一个初值很小的可学习因子。
  ——`zhu2025hc` v3，§2.2，式 (10)–(13)。PyTorch 伪代码里缩放因子初值 `torch.ones(1) * 0.01`，归一化用 `LayerNorm(dim)`（附录 J，算法 2）。
- **C11｜初始化等价于 Pre-Norm 残差。** 动态权重 $\mathbf W_\beta,\mathbf W_m,\mathbf W_r$ 初始化为 0；静态部分 $\mathbf B^k=\mathbf 1_{1\times n}$，
  $\mathbf A_m^k=\mathbf e_{k\bmod n}$，$\mathbf A_r^k=\mathbf e_{n\times n}$（单位阵）。原文：“equivalent to the Pre-Norm residual connections”
  ——`zhu2025hc` v3，§2.3，式 (14)。
- **C12｜Pre-Norm 与 Post-Norm 都是“不可训练的 HC”（$n=1$）。** 式 (15)(16)，证明在附录 G。
- **C13｜$n=2$ 时 HC 可以表示“串行”与“并行（PTB）”两种层排布，学到的可以是二者的软混合。** 式 (17)–(19)，图 4，附录 H。
- **C14｜$n=1$ 不行，$n>1$ 才有收益；$n=4$ 最好，$n=8$ 几乎不再涨。**
  原文：n=4 最好，“the increase to n=8 providing minimal additional benefits.”
  ——`zhu2025hc` v3，§4.1，表 1。$n=1$ 时 DHC 比基线差（V2 loss 2.819 vs 2.811）。
- **C15｜$n=4$ 时 DHC 明显好于 SHC，$n=2$ 时二者相近。** ——§4.1 “Static and dynamic hyper-connections”，表 2。
- **C16｜HC 论文自己报告的是“更稳”，不是“不稳”。** 原文：“no spikes observed in any DHC experiments”（§4.1）；
  7B 上 “the baseline model exhibits frequent spikes, while our model with DHCs shows no spikes”（§4.3）。
  （与 mHC 在 27B 观察到的不稳定形成对照，见 §7 E3。）
- **C17｜OLMoE-1B-7B 上的收益（500B tokens）。** 收敛快 1.8 倍（图 1 标题：“converges 1.8 times faster”）；
  ARC-Challenge 41.8 → 47.8（+6.0），MMLU Var 38.5 → 39.7（+1.2），训练 loss 约 −0.027，C4-en 验证 loss −0.028（§4.4，表 6）。
- **C18｜7B 稠密（OLMo-7B-DHC×4）。** V2 loss 2.581 → 2.559（−0.022），V2 PPL 14.316 → 14.023（−0.293），
  下游平均 70.1 → 71.0；FLOPs 13.36 → 13.38 G/token（表 5）。
- **C19｜HC 的参数、FLOPs 几乎为零，但显存不是。** 参数增加 +0.00002% ~ +0.03349%（表 7），FLOPs +0.038% ~ +0.208%（表 8），
  实测显存 +9.7% ~ +28.28%（表 9，8 GPU）。原文（显存）：“adds an extra memory overhead of 2nsbd_model L”，
  并称重计算后可降到 $nsbd_{model}$。——`zhu2025hc` v3，附录 B（**v1 没有这一附录**）。
- **C20｜学到的连接呈 Λ 形：既有 Post-Norm 式的“近邻衰减”，又有 Pre-Norm 式的“底层被频繁访问”。**
  原文：“a free mixture of Pre- and Post-Norm architecture.”——§4.5。
- **C21｜HC 也改善视觉任务。** DiT-XL/2-SHC×2（675M）FID 2.18，接近 DiT-1B/2（983M）的 2.13（表 10）；ViT-L/16 top-1 77.25 → 78.38（SHC×2）/ 79.94（DHC×2）（表 11）。

### 1C. mHC 的问题陈述（Xie et al.）

- **C22｜HC 的单层写法与三个映射的形状。** $\mathbf x_{l+1}=\mathcal H^{\rm res}_l\mathbf x_l+\mathcal H^{{\rm post}\top}_l\mathcal F(\mathcal H^{\rm pre}_l\mathbf x_l,\mathcal W_l)$，
  $\mathcal H^{\rm res}_l\in\R^{n\times n}$，$\mathcal H^{\rm pre}_l,\mathcal H^{\rm post}_l\in\R^{1\times n}$，$\mathbf x_l\in\R^{n\times C}$。
  原文强调 HC “without altering the computational overhead of individual units regarding FLOPs”。——`xie2026mhc` v2，§1，式 (3)。
- **C23｜多层展开后，复合映射 $\prod\mathcal H^{\rm res}$ 不再保持特征的全局均值，导致无界放大或衰减。**
  原文：“fails to preserve the global mean of the features.” ——§1，式 (4) 之后。
- **C24｜消融：三个映射里 $\mathcal H^{\rm res}$ 贡献最大。** 表 1（“preliminary experiments”，模型规模与 token 数均未写）：只开 $\mathcal H^{\rm res}$
  → loss 差 −0.022；再开 $\mathcal H^{\rm pre}$ → −0.025；三个全开 → −0.027。关掉的映射用固定值代替：
  $\mathcal H^{\rm pre}=1/n$ 均匀、$\mathcal H^{\rm post}=$ 全 1、$\mathcal H^{\rm res}=\mathbf I$。——§3，表 1 及其标题。
- **C25｜27B 上 HC 在约 12k 步出现 loss 突增，与梯度范数不稳定高度相关。**
  原文：“HC exhibits an unexpected loss surge around the 12k step” ——§3.1，图 2。
- **C26｜Amax Gain Magnitude 的定义。** 前向：复合映射“maximum absolute value of the row sums”；反向：“maximum absolute column sum”。
  对一个选定序列的所有 token 取平均。——§3.1 正文与图 3 标题。**注意是“行和的绝对值”，不是“绝对值的行和”**（笔者用图 8 的数字验证过，见 §4 N27）。
- **C27｜HC 复合增益峰值约 3000。** 原文：“extreme values with peaks of 3000”（§3.1）；§5.4 又说 “nearly 3000”。
  数字化图 3(b)：反向复合增益最大 ≈ 2825（在子层 $l=52$）。
- **C28｜HC 的访存（I/O）约按 $n$ 倍增长，另有显存与流水线通信开销。** 原文：“increases the memory access cost by a factor approximately proportional to n”；
  以及 HC “requires n-fold more communication cost in pipeline parallelism”。——§3.2，表 2。

### 1D. mHC 的方法

- **C29｜约束：把 $\mathcal H^{\rm res}_l$ 限制为双随机矩阵（非负、行和列和都为 1），即 Birkhoff 多面体。** 式 (6)。
  $n=1$ 时退化为标量 1（原文：“recovering the original identity mapping”）。——§4.1。
- **C30｜三条性质（论文的原话标题）：Norm Preservation（$\|\mathcal H^{\rm res}_l\|_2\le1$，“non-expansive”）、Compositional Closure（乘法封闭）、
  Geometric Interpretation via the Birkhoff Polytope（置换矩阵的凸包，“a convex combination of permutations”）。**
  另有一句未给证明的断言：反复作用 “tends to increase the mixing of information across streams monotonically”。——§4.1 列表 1–3。
  （这些性质的证明本论文没有给出，笔者在 §2.7 补全；“monotonically” 的精确含义见 §7 E7。）
- **C31｜$\mathcal H^{\rm pre}$、$\mathcal H^{\rm post}$ 也加非负约束，防止正负系数相互抵消。** 原文：“prevents signal cancellation” ——§4.1 末段。
- **C32｜参数化：把 $n$ 条流展平成 $1\times nC$ 向量、RMSNorm、线性投影得到动态部分，加静态偏置；再用 sigmoid / 2·sigmoid / Sinkhorn-Knopp 施加约束。**
  展平的理由：“to preserve full context information.”——§4.2，式 (7)(8)。**注意 mHC 的式 (7) 已经没有 tanh**（HC 的式 (5) 有）。
- **C33｜Sinkhorn-Knopp：先取指数变正，再交替做列归一化、行归一化；$t_{\max}\to\infty$ 时收敛到双随机矩阵；实用取 $t_{\max}=20$。**
  原文：“We choose t_max=20 as a practical value in our experiments.”——§4.2，式 (9)。**迭代顺序是 $\mathbf M^{(t)}=\mathcal T_r(\mathcal T_c(\mathbf M^{(t-1)}))$，即每轮最后一步是行归一化。**
- **C34｜“entropically project”。** 原文 §1：“to entropically project $\mathcal H^{\rm res}_l$ onto the Birkhoff polytope”。
  这句话的精确数学含义由 `mena2018gumbel` 定理 1 给出（§2.5）：$S(X)=\arg\max_{P\in\mathcal B_N}\langle P,X\rangle_F+h(P)$。mHC 论文本身没有写这个式子。

### 1E. mHC 的基础设施

- **C35｜核融合与混合精度。** 把 RMSNorm 的“除以范数”挪到矩阵乘之后（“maintains mathematical equivalence”）；RMSNorm 权重吸收进 $\varphi_l$；
  三个系数核 + 两个应用核；$\varphi_l$ 用 tfloat32，$\vec{\mathbf x}_l$ 用 bfloat16，系数用 float32（式 (10)–(19)）；
  $\mathcal F_{\rm post,res}$ 核把读从 $(3n+1)C$ 降到 $(n+1)C$、写从 $3nC$ 降到 $nC$；
  除式 (14)–(15) 外大部分核用 TileLang 实现。Sinkhorn 放在单个核里，反向用自定义核在片上重算中间结果。——§4.3.1。
- **C36｜重计算。** 前向后丢弃 mHC 核的中间激活，反向时只重跑 mHC 核（不重跑 $\mathcal F$）；每 $L_r$ 层只存块首输入 $\mathbf x_{l_0}$；
  最优块长 $L_r^*\approx\sqrt{nL/(n+2)}$（式 (20)）；实践中把重计算块边界与流水线 stage 边界对齐（“synchronize the recomputation boundaries with the pipeline stages”）。——§4.3.2，表 3。
- **C37｜DualPipe 扩展。** MLP 层的 $\mathcal F_{\rm post,res}$ 核放到**高优先级计算流**；注意力层的长时操作**不用 persistent kernel**，以便被抢占；
  重计算与流水线通信解耦（stage 首激活已本地缓存）。——§4.3.3，图 4。
- **C38｜开销：$n=4$ 时额外训练时间 6.7%。** 原文 §1：“only a 6.7% additional time overhead when expansion rate n=4”；§4.3：“a marginal training overhead of only 6.7%”。
  §1 的原句以 “In-house large-scale training indicates …” 起头，§4.3 只说 “in large-scale models”；两处都**没有写是哪个模型、多少卡**（§8）。

### 1F. mHC 的实验

- **C39｜设置与结果。** 3B / 9B / 27B（“MoE architectures inspired by DeepSeek-V3”，$n=4$），外加一个 1T tokens 的 3B（token scaling）。
  27B：最终 loss 比基线低 0.021（§5.2，“a final loss reduction of 0.021”）；表 4 八个基准 mHC 全部高于基线，8 个里 7 个高于 HC（MATH 例外：26.0 vs 26.4）；
  相对 HC，BBH +2.1、DROP +2.3（论文写作 “2.1% … 2.3%”，实为百分点，见 §7 E12）。
  compute scaling：优势在更大算力上“only marginal attenuation”（§5.3，图 6）。
  稳定性：20 次 SK 迭代只得到近似解，单层反向增益略偏离 1；复合反向增益最大约 1.6（“reaching a maximum value of approximately 1.6”），
  比 HC 的近 3000 低三个数量级（§5.4，图 7）。

### 1G. DeepSeek-V4 / V4.1 中的 mHC

- **C40｜V4 摘要把 mHC 列为三项“architecture and optimization”关键升级之一**（另两项是 CSA/HCA 混合注意力与 Muon 优化器，Muon 属优化而非架构；摘要：“(2) Manifold-Constrained Hyper-Connections (mHC) that enhance conventional residual connections”），
  §2.2 用 $A_l$（输入映射，$1\times n_{hc}$）、$B_l$（残差变换，$n_{hc}\times n_{hc}$）、$C_l$（输出映射，$n_{hc}\times1$）重写（式 (1)–(8)）；
  V4 的理由句：HC “will frequently exhibit numerical instability when stacking multiple layers”；
  $A_l$、$C_l$ “constrained to be non-negative and bounded via a Sigmoid function”；
  优化器分工：mHC 的 **static biases and gating factors** 用 AdamW（§2.4 “Basic Configurations”），其余矩阵用 Muon；
  确定性：“mHC involves a matrix multiplication with an output dimension of only 24”，小 batch 时 split-k 需确定性归约（§3.3）；
  实现（§3.4.2）：融合核、选择性重计算（“recompute most hidden states between layers and all normalized layer inputs”）、调整 DualPipe 1F1B 重叠，
  开销 “only 6.7% of the overlapped 1F1B pipeline stage”；
  超参数（§4.2.1，Flash 与 Pro 两处原句相同）：扩展率 $n_{hc}=4$，“the number of Sinkhorn-Knopp iterations t_max is set to 20.”
- **C41｜V4.1-Flash：Single-Pass mHC。** V4 的多趟（multi-pass）实现有三个串行核，激活访存共 $(4n+4)d$（原文：“twice the lower bound”）；
  理想下界 $(2n+2)d$。办法是把输入混合系数“推迟一块”：第 $l$ 块用第 $l-1$ 块预测的 $A_{l-1}$（式 (6)），原文：“shifts the input-mixing coefficients by one block”；
  “Empirically, this shift incurs negligible performance degradation.”；预训练仍用多核实现；部署时用单核 Mega-mHC，mHC 版 $(3n+2)d$、Single-Pass 版 $(2n+2)d$，
  “halving the activation memory traffic of our original implementation.”——`dsv41` v1，§2.4.1。超参数同 V4：扩展率 4、SK 20 次（§4.2.1）。

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.0 课程记号建议（与 style-guide §2 对齐）

mHC 论文、HC 论文、V4 论文三套记号互不相同，而且好几个字母和全局符号表冲突。建议本课统一如下（写课时在首次出现处给出对照）：

| 课程记号 | 含义 / 形状 | mHC 论文 | HC 论文 | V4 / V4.1 论文 | V4 代码 | 冲突说明 |
|---|---|---|---|---|---|---|
| $n$ | 扩展率（残差流条数），DeepSeek 全部取 4 | $n$ | $n$ | $n_{hc}$ / $n$ | `hc_mult` | — |
| $d$ | 模型隐藏维 | $C$ | $d$ / $d_{model}$ | $d$ | `dim` | **mHC 的 $C$ 与全局 $C$（训练算力）冲突，课程用 $d$** |
| $X_l\in\R^{n\times d}$ | 第 $l$ 个子层之前的 $n$ 条流，**每行一条流** | $\mathbf x_l$ | $\mathbf H^{k-1}$（行 = 流） | $X_l$ | `x: [b,s,hc,d]` | 全局约定“向量是列向量”；这里必须明说“按行堆叠” |
| $H_l^{\rm pre}\in\R^{1\times n}$ | 读出（n 流 → 1 个层输入） | $\mathcal H^{\rm pre}_l$ | $\mathbf A_m^\intercal$ | $A_l$ | `pre` | — |
| $H_l^{\rm post}\in\R^{1\times n}$ | 写入（层输出 → n 流），公式里以转置出现 | $\mathcal H^{\rm post}_l$ | $\mathbf B$ | $C_l=(H^{\rm post}_l)^\top\in\R^{n\times1}$ | `post` | **V4 的 $C_l$ 与全局 $C$ 冲突；HC 的 $\mathbf B$ 与全局 $B$（batch）冲突** |
| $H_l^{\rm res}\in\R^{n\times n}$ | 流间混合 | $\mathcal H^{\rm res}_l$ | $\mathbf A_r^\intercal$ | $B_l$ | `comb` 的**转置**（见 §2.8） | **V4 的 $B_l$ 与 batch $B$ 冲突** |
| $\tilde H_l^{\cdot}$ | 约束前的原始系数 | $\tilde{\mathcal H}$ | — | $\tilde A,\tilde B,\tilde C$ | `mixes*scale+base` | — |
| $W_l^{\rm pre},W_l^{\rm post}\in\R^{nd\times n}$, $W_l^{\rm res}\in\R^{nd\times n^2}$ | 动态部分的投影 | $\varphi_l$ | $\mathbf W_m,\mathbf W_\beta,\mathbf W_r\in\R^{d\times\cdot}$ | $W_l$ | `hc_*_fn`（形状 `[24, 4d]`） | HC 论文 / mHC 式 (5) 用 $\theta$，与全局 $\theta$（参数）冲突 |
| $S_l^{\rm pre},S_l^{\rm post},S_l^{\rm res}$ | 静态偏置 | $\mathbf b_l$ | 静态矩阵本身 | $S_l$ | `hc_*_base` | **mHC 的 $\mathbf b$ 与全局 $b_i$（无辅助损失均衡偏置）冲突，建议沿用 V4 的 $S$** |
| $\alpha_l^{\rm pre},\alpha_l^{\rm post},\alpha_l^{\rm res}$ | 门控标量，初值 0.01 | 同 | $s_\alpha,s_\beta$ | 同 | `hc_*_scale`（3 个） | — |
| $\mathcal F_l$ | 第 $l$ 个子层（注意力或 FFN/MoE），**内含自己的 pre-norm** | $\mathcal F(\cdot,\mathcal W_l)$ | $\mathcal T$ | $\mathcal F_l$ | `attn`/`ffn` 前接 `*_norm` | HC 的 $\mathcal T$ 与全局 $T$（序列长）易混 |
| $\mathcal B_n$ | $n\times n$ 双随机矩阵集合（Birkhoff 多面体） | $\mathcal M^{\rm res}$ | — | $\mathcal M$ | — | — |
| $\mathcal T_r,\mathcal T_c$ | 行 / 列归一化 | 同 | — | 同 | — | 与 $T$ 不同字体，可保留 |
| $t_{\max}$ | SK 迭代次数（=20） | 同 | — | 同 | `hc_sinkhorn_iters` | — |
| $\sigma(\cdot)$ | sigmoid | 同 | — | 同 | `T.sigmoid` | HC 式 (16) 的 $\sigma_i,\sigma_o$ 是标准差，讲到 Post-Norm 时要换字母 |
| $\mathbf 1_n$, $J_n=\tfrac1n\mathbf 1\mathbf 1^\top$ | 全 1 向量；均匀平均矩阵 | $\mathbf 1_n$ | $\mathbf 1_{1\times n}$ | $\mathbf 1_n$ | — | $J_n$ 为笔者引入 |

层号约定：mHC 图 3/7/8 把每个 Transformer block 拆成两个“层”（注意力、FFN），27B 的 30 个 block 对应横轴 $l=1..60$
（图 3 标题：“unrolls each standard Transformer block into two independent layers (Attention and FFN)”）。V4 代码里每个 `Block`
确实有两套 mHC 参数（`hc_attn_*` 与 `hc_ffn_*`）。V4.1 正文说 “block” 时指的也是这种子层（代码注释：“Each sub-block's own
`hc_mixes` produces the mix for the *next* one”）。**课程建议统一称“子层”，$l$ 为子层编号。**

### 2.1 残差连接（mHC 式 (1)(2)，He 2016b 式 (3)–(8)）

mHC v2 §1：

$$\mathbf{x}_{l+1}=\mathbf{x}_{l}+\mathcal{F}(\mathbf{x}_{l},\mathcal{W}_{l}) \tag{mHC-1}$$

$$\mathbf{x}_{L}=\mathbf{x}_{l}+\sum_{i=l}^{L-1}\mathcal{F}(\mathbf{x}_{i},\mathcal{W}_{i}) \tag{mHC-2}$$

He 2016b v3（式号为原文式号）：

$$\frac{\partial{\mathcal{E}}}{\partial{\mathbf{x}_{l}}}=\frac{\partial{\mathcal{E}}}{\partial{\mathbf{x}_{L}}}\left(1+\frac{\partial}{\partial{\mathbf{x}_{l}}}\sum_{i=l}^{L-1}\mathcal{F}(\mathbf{x}_{i},\mathcal{W}_{i})\right) \tag{He-5}$$

$$\mathbf{x}_{l+1}=\lambda_{l}\mathbf{x}_{l}+\mathcal{F}(\mathbf{x}_{l},\mathcal{W}_{l}),\qquad
\mathbf{x}_{L}=\Big(\prod_{i=l}^{L-1}\lambda_{i}\Big)\mathbf{x}_{l}+\sum_{i=l}^{L-1}\hat{\mathcal{F}}(\mathbf{x}_{i},\mathcal{W}_{i}) \tag{He-6, He-7}$$

$$\frac{\partial{\mathcal{E}}}{\partial{\mathbf{x}_{l}}}=\frac{\partial{\mathcal{E}}}{\partial{\mathbf{x}_{L}}}\left(\Big(\prod_{i=l}^{L-1}\lambda_{i}\Big)+\frac{\partial}{\partial{\mathbf{x}_{l}}}\sum_{i=l}^{L-1}\hat{\mathcal{F}}(\mathbf{x}_{i},\mathcal{W}_{i})\right) \tag{He-8}$$

课程记号：$\mathbf x_l\to\mathbf x_l\in\R^d$（单流时仍是列向量）。

### 2.2 Hyper-Connections（HC v3 式 (1)–(14)）

$$\mathcal{HC}=\begin{pmatrix}\mathbf{0}_{1\times 1}&\mathbf{B}\\ \mathbf{A_{m}}&\mathbf{A_{r}}\end{pmatrix}=\begin{pmatrix}0&\beta_{1}&\beta_{2}&\cdots&\beta_{n}\\ \alpha_{1,0}&\alpha_{1,1}&\alpha_{1,2}&\cdots&\alpha_{1,n}\\ \vdots&\vdots&\vdots&\ddots&\vdots\\ \alpha_{n,0}&\alpha_{n,1}&\alpha_{n,2}&\cdots&\alpha_{n,n}\end{pmatrix}\in\mathbb{R}^{(n+1)\times(n+1)} \tag{HC-1}$$

$$\hat{\mathbf H}=\mathcal{HC}(\mathcal{T},\mathbf{H})=\mathbf{B}^{\intercal}\mathcal{T}(\mathbf{H}^{\intercal}\mathbf{A_{m}})^{\intercal}+\mathbf{A_{r}}^{\intercal}\mathbf{H} \tag{HC-2}$$

$$\mathbf{h}_{0}^{\intercal}=\mathbf{A_{m}}^{\intercal}\mathbf{H},\qquad \mathbf{H'}=\mathbf{A_{r}}^{\intercal}\mathbf{H},\qquad \hat{\mathbf H}=\mathbf{B}^{\intercal}(\mathcal{T}\mathbf{h}_{0})^{\intercal}+\mathbf{H'} \tag{HC-3,4,5}$$

动态 HC（式 (10)–(13)）：

$$\overline{\mathbf H}=\texttt{norm}(\mathbf H),\quad
\mathcal B(\mathbf H)=s_\beta\circ\tanh(\overline{\mathbf H}\mathbf W_\beta)^\intercal+\mathbf B\in\R^{1\times n},$$
$$\mathcal A_m(\mathbf H)=s_\alpha\circ\tanh(\overline{\mathbf H}\mathbf W_m)+\mathbf A_m\in\R^{n\times1},\quad
\mathcal A_r(\mathbf H)=s_\alpha\circ\tanh(\overline{\mathbf H}\mathbf W_r)+\mathbf A_r\in\R^{n\times n} \tag{HC-10..13}$$

初始化（式 (14)）：$\begin{pmatrix}\mathbf 0_{1\times1}&\mathbf B^k\\ \mathbf A_m^k&\mathbf A_r^k\end{pmatrix}=\begin{pmatrix}\mathbf 0_{1\times1}&\mathbf 1_{1\times n}\\ \mathbf e_{k\bmod n}&\mathbf e_{n\times n}\end{pmatrix}$。

Pre-Norm / Post-Norm（式 (15)(16)，$n=1$）：

$$\mathcal{HC}_{PreNorm}=\begin{pmatrix}0&1\\1&1\end{pmatrix},\qquad
\mathcal{HC}_{PostNorm}=\begin{pmatrix}0&\frac{1}{\sqrt{\sigma_{i}^{2}+\sigma_{o}^{2}+2\sigma_{io}}}\\ 1&\frac{1}{\sqrt{\sigma_{i}^{2}+\sigma_{o}^{2}+2\sigma_{io}}}\end{pmatrix} \tag{HC-15,16}$$

序列 / 并行（式 (17)–(19)，$n=2$）：

$$\mathcal{HC}=\begin{pmatrix}0&1&1\\1&1&0\\0&0&1\end{pmatrix}\ (\text{串行}),\qquad
\mathcal{HC}_{odd}=\begin{pmatrix}0&1&0\\1&1&1\\1&1&1\end{pmatrix},\ \mathcal{HC}_{even}=\begin{pmatrix}0&0&1\\0&1&0\\1&0&1\end{pmatrix}\ (\text{两层并行}) \tag{HC-17..19}$$

展开成稠密连接（附录 F，式 (27)）：$c^{(0)}_{kj}=\mathbf B^j\big(\prod_{t=j+1}^{k-1}\mathbf A_r^t\big)\mathbf A_m^k$。

参数量（附录 B，式 (21)(22)(25)(26)）：$|\theta_{SHC}|=n(n+2)$；$|\theta_{DHC}|=|\theta_{norm}|+d_{model}(n+2)+n(n+2)+2$；
$P_{extra}=|\theta|\times2\times L$（每层两个 HC 模块）。显存：额外 $2nsbd_{model}L$，重计算后 $nsbd_{model}$（Korthikanti 的基准为 $sbd_{model}L(34+5as/d_{model})$）。

**HC ↔ mHC ↔ 课程映射（笔者推导，逐项核对形状）**：HC 的 $\mathbf H\in\R^{n\times d}$ 与 mHC 的 $\mathbf x_l\in\R^{n\times C}$ 同为“行 = 流”。
由式 (HC-2)：$\mathbf H^\intercal\mathbf A_m$ 是 $d\times1$ 的层输入，即 $H^{\rm pre}=\mathbf A_m^\intercal$；$\mathbf B^\intercal(\cdot)^\intercal$ 把 $1\times d$ 的层输出写到 $n$ 行，即 $H^{\rm post}=\mathbf B$；
$\mathbf A_r^\intercal\mathbf H$ 即 $H^{\rm res}=\mathbf A_r^\intercal$（所以 $\alpha_{i,j}$，$i,j\ge1$，是“从流 $i$ 到新流 $j$”的权重）。
HC 初始化因此对应 $H^{\rm pre}=\mathbf e_{k\bmod n}^\top$（只读一条流）、$H^{\rm post}=\mathbf 1^\top$（写回所有流）、$H^{\rm res}=\mathbf I$。
动态部分的结构差别：HC 里 $\mathcal A_r(\mathbf H)$ 的第 $i$ 行只由第 $i$ 条流 $\overline{\mathbf h}_i$ 决定（$\overline{\mathbf H}\mathbf W_r$ 逐行作用）；
mHC 式 (7) 用展平后的整条 $nC$ 向量，每个系数都依赖所有流。

### 2.3 HC 的多层展开与 mHC 的约束（mHC v2 式 (3)–(6)）

$$\mathbf{x}_{l+1}=\mathcal{H}_{l}^{\mathrm{res}}\mathbf{x}_{l}+\mathcal{H}_{l}^{\mathrm{post}\,\top}\mathcal{F}(\mathcal{H}_{l}^{\mathrm{pre}}\mathbf{x}_{l},\mathcal{W}_{l}) \tag{mHC-3}$$

$$\mathbf{x}_{L}=\left(\prod_{i=1}^{L-l}\mathcal{H}_{L-i}^{\mathrm{res}}\right)\mathbf{x}_{l}+\sum_{i=l}^{L-1}\left(\prod_{j=1}^{L-1-i}\mathcal{H}_{L-j}^{\mathrm{res}}\right)\mathcal{H}_{i}^{\mathrm{post}\,\top}\mathcal{F}(\mathcal{H}_{i}^{\mathrm{pre}}\mathbf{x}_{i},\mathcal{W}_{i}) \tag{mHC-4}$$

读法：$\prod_{i=1}^{L-l}\mathcal H^{\rm res}_{L-i}=\mathcal H^{\rm res}_{L-1}\mathcal H^{\rm res}_{L-2}\cdots\mathcal H^{\rm res}_{l}$（$i=1$ 的因子在最左）。

$$\mathcal{P}_{\mathcal{M}^{\mathrm{res}}}(\mathcal{H}^{\mathrm{res}}_{l})\coloneq\left\{\mathcal{H}^{\mathrm{res}}_{l}\in\mathbb{R}^{n\times n}\mid\mathcal{H}^{\mathrm{res}}_{l}\mathbf{1}_{n}=\mathbf{1}_{n},\ \mathbf{1}^{\top}_{n}\mathcal{H}^{\mathrm{res}}_{l}=\mathbf{1}^{\top}_{n},\ \mathcal{H}^{\mathrm{res}}_{l}\geqslant 0\right\} \tag{mHC-6}$$

课程记号：$X_{l+1}=H^{\rm res}_lX_l+(H^{\rm post}_l)^\top\mathcal F_l(H^{\rm pre}_lX_l)$，$H^{\rm res}_l\in\mathcal B_n$。

mHC 对 HC 系数的复述（式 (5)，注意这是 HC 的写法，**有 tanh**）：

$$\tilde{\mathbf x}_l=\text{RMSNorm}(\mathbf x_l),\quad
\mathcal{H}^{\mathrm{pre}}_{l}=\alpha_{l}^{\mathrm{pre}}\cdot\tanh(\theta^{\mathrm{pre}}_{l}\tilde{\mathbf{x}}^{\top}_{l})+\mathbf{b}_{l}^{\mathrm{pre}},\ \ldots,\
\mathcal{H}^{\mathrm{res}}_{l}=\alpha_{l}^{\mathrm{res}}\cdot\tanh(\theta^{\mathrm{res}}_{l}\tilde{\mathbf{x}}^{\top}_{l})+\mathbf{b}_{l}^{\mathrm{res}} \tag{mHC-5}$$

其中 $\theta^{\rm pre}_l,\theta^{\rm post}_l\in\R^{1\times C}$，$\theta^{\rm res}_l\in\R^{n\times C}$，$\mathbf b^{\rm res}_l\in\R^{n\times n}$，$\alpha$ “initialized to small values”。
（mHC 这里写 RMSNorm，HC 代码用 LayerNorm、正文写 `norm`；见 §7 E14。）

### 2.4 mHC 的参数化与投影（式 (7)–(9)）

$$\begin{cases}\vec{\mathbf{x}}^{\prime}_{l}=\text{RMSNorm}(\vec{\mathbf{x}}_{l})\\
\tilde{\mathcal{H}}^{\mathrm{pre}}_{l}=\alpha_{l}^{\mathrm{pre}}\cdot(\vec{\mathbf{x}}^{\prime}_{l}\varphi^{\mathrm{pre}}_{l})+\mathbf{b}_{l}^{\mathrm{pre}}\\
\tilde{\mathcal{H}}^{\mathrm{post}}_{l}=\alpha_{l}^{\mathrm{post}}\cdot(\vec{\mathbf{x}}^{\prime}_{l}\varphi^{\mathrm{post}}_{l})+\mathbf{b}_{l}^{\mathrm{post}}\\
\tilde{\mathcal{H}}^{\mathrm{res}}_{l}=\alpha_{l}^{\mathrm{res}}\cdot\text{mat}(\vec{\mathbf{x}}^{\prime}_{l}\varphi^{\mathrm{res}}_{l})+\mathbf{b}_{l}^{\mathrm{res}}\end{cases} \tag{mHC-7}$$

$\vec{\mathbf x}_l=\mathrm{vec}(\mathbf x_l)\in\R^{1\times nC}$；$\varphi^{\rm pre}_l,\varphi^{\rm post}_l\in\R^{nC\times n}$，$\varphi^{\rm res}_l\in\R^{nC\times n^2}$；$\mathrm{mat}:\R^{1\times n^2}\to\R^{n\times n}$。

$$\begin{cases}\mathcal{H}^{\mathrm{pre}}_{l}=\sigma(\tilde{\mathcal{H}}^{\mathrm{pre}}_{l})\\
\mathcal{H}^{\mathrm{post}}_{l}=2\sigma(\tilde{\mathcal{H}}^{\mathrm{post}}_{l})\\
\mathcal{H}^{\mathrm{res}}_{l}=\text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}^{\mathrm{res}}_{l})\end{cases} \tag{mHC-8}$$

$$\mathbf{M}^{(0)}=\exp(\tilde{\mathcal{H}}^{\mathrm{res}}_{l}),\qquad \mathbf{M}^{(t)}=\mathcal{T}_{r}\left(\mathcal{T}_{c}(\mathbf{M}^{(t-1)})\right),\qquad \mathcal H^{\rm res}_l=\mathbf M^{(t_{\max})},\ t_{\max}=20 \tag{mHC-9}$$

值域（笔者推导）：$H^{\rm pre}\in(0,1)^{1\times n}$，$H^{\rm post}\in(0,2)^{1\times n}$。初始化时若 $\alpha\approx0.01$ 且静态偏置为 0，则 $H^{\rm pre}\approx(0.5,\ldots,0.5)$、$H^{\rm post}\approx(1,\ldots,1)$；
**但论文没有给出 $\mathbf b$ 的初值**（§8）。因为 $\mathcal F_l$ 内含 pre-norm（V4 代码：`x = self.attn_norm(x)` 紧跟在 `hc_pre` 之后），
$H^{\rm pre}$ 的整体缩放对层输入无影响，只有方向起作用（笔者推导）。

核融合版（式 (10)–(19)；$\varphi_l$ 与 $\mathbf b_l$ 把三组拼在一起，RMSNorm 权重吸收进 $\varphi_l$）：

$$\varphi_l:\text{tfloat32}\ [nC,\,n^2+2n];\quad \vec{\mathbf x}_l:\text{bfloat16}\ [1,nC];\quad \alpha:\text{float32};\quad \mathbf b_l:\text{float32}\ [1,n^2+2n] \tag{mHC-10..13}$$
$$[\tilde{\tilde{\mathcal H}}^{\rm pre}_l,\tilde{\tilde{\mathcal H}}^{\rm post}_l,\tilde{\tilde{\mathcal H}}^{\rm res}_l]=\vec{\mathbf x}_l\varphi_l,\qquad r=\|\vec{\mathbf x}_l\|_2/\sqrt{nC} \tag{mHC-14,15}$$
$$[\tilde{\mathcal H}^{\rm pre}_l,\tilde{\mathcal H}^{\rm post}_l,\tilde{\mathcal H}^{\rm res}_l]=1/r\,[\alpha^{\rm pre}_l\tilde{\tilde{\mathcal H}}^{\rm pre}_l,\alpha^{\rm post}_l\tilde{\tilde{\mathcal H}}^{\rm post}_l,\alpha^{\rm res}_l\tilde{\tilde{\mathcal H}}^{\rm res}_l]+\mathbf b_l \tag{mHC-16}$$
式 (17)–(19) 同式 (8)。应用核：$\mathcal F_{\rm pre}\coloneq\mathcal H^{\rm pre}_l\mathbf x_l$，$\mathcal F_{\rm post,res}\coloneq\mathcal H^{\rm res}_l\mathbf x_l+\mathcal H^{{\rm post}\top}_l\mathcal F(\cdot,\cdot)$。

“先乘后除”为什么等价（笔者推导）：$\mathrm{RMSNorm}(\vec{\mathbf x})\varphi=\frac{\vec{\mathbf x}\odot\mathbf g}{r}\varphi=\frac1r\,\vec{\mathbf x}\,(\mathrm{diag}(\mathbf g)\varphi)$，$r$ 是标量，可以提到矩阵乘外面；
$\mathrm{diag}(\mathbf g)\varphi$ 离线合并。V4 代码与此一致：`mixes = F.linear(x, hc_fn) * rsqrt`，`rsqrt = torch.rsqrt(x.square().mean(-1) + norm_eps)`。

### 2.5 Sinkhorn 算子的熵正则化刻画（Mena 2018 定理 1）

`mena2018gumbel` v1 式 (1)：$S^0(X)=\exp(X)$，$S^l(X)=\mathcal T_c(\mathcal T_r(S^{l-1}(X)))$，$S(X)=\lim_{l\to\infty}S^l(X)$
（**注意它的顺序是先行后列，与 mHC 式 (9) 相反**）；定理 1：

$$S(X/\tau)=\arg\max_{P\in\mathcal{B}_{N}}\ \langle P,X\rangle_{F}+\tau\, h(P),\qquad h(P)=-\sum_{i,j}P_{i,j}\log P_{i,j},\qquad M(X)=\lim_{\tau\to0^{+}}S(X/\tau) \tag{Mena-Thm1}$$

$M(X)=\arg\max_{P\in\mathcal P_N}\langle P,X\rangle_F$ 是线性指派（选一个置换）。
**定理的前提（不要漏）**：第一式（式 (3)）对任意 $X$ 成立，$h$ 只对双随机 $P$ 定义；第二式（式 (4)）的 $\tau\to0^+$ 极限另需假设
“$X$ 的元素独立地取自关于 Lebesgue 测度绝对连续的分布”，并且只是**几乎必然**成立（保证最优置换唯一）。原文随后说明定理 1 “cannot be realized in practice”，
实际总用截断到 $L$ 次的不完全 Sinkhorn 算子。——`mena2018gumbel` v1，§2 定理 1 及其后一段。
对 mHC（$\tau=1$，$X=\tilde H^{\rm res}$）的含义（笔者据定理 1 推出）：$H^{\rm res}_l$ 是 Birkhoff 多面体里“与 $\tilde H^{\rm res}$ 内积最大、同时熵最大”的折中点；
logits 越悬殊（相当于 $\tau$ 越小），结果越靠近某个置换矩阵，SK 收敛也越慢（§4 N20–N22）。

Sinkhorn–Knopp 1967 的定理（摘要原意）：非负矩阵 $A$ 的交替行列缩放序列收敛到双随机极限，当且仅当 $A$ 至少含一条正对角线
（原文：“if and only if the matrix A contains at least one positive diagonal”）；存在主对角线为正的对角阵 $D_1,D_2$，使 $D_1AD_2$ **既双随机又是该迭代的极限**，其充要条件是 $A\ne0$ 且每个正元素都在某条正对角线上；
$D_1AD_2$ 的形式唯一，而 $D_1,D_2$ 在相差一个正数倍的意义下唯一**当且仅当** $A$ 完全不可分解（原文是 “if and only if”，不只是“充分”）。mHC 先取 $\exp$，矩阵严格为正，条件自动满足（这正是 Sinkhorn 1964 处理的情形）。

### 2.6 从 He 2016b 到 mHC：同一个乘积（笔者推导，课上可作为桥梁）

$n=1$ 时 $H^{\rm res}_l$ 退化成标量 $\lambda_l$，式 (mHC-4) 就是 He 式 (7)。所以 mHC 要解决的是 He 2016b 在 2016 年就指出的问题的矩阵版：
标量情形里“乘积不爆炸也不消失”的唯一稳妥选择是 $\lambda\equiv1$；矩阵情形里，mHC 的选择是“每个因子都是双随机矩阵”，
它包含 $\mathbf I$，但也允许流之间交换信息。

### 2.7 双随机矩阵的性质（mHC §4.1 只陈述不证明；以下为笔者补全的证明，供 `::: deriv` 使用）

设 $A,B\in\mathcal B_n$。

1. **乘法封闭**：$AB\ge0$；$(AB)\mathbf 1=A(B\mathbf 1)=A\mathbf 1=\mathbf 1$；$\mathbf 1^\top(AB)=(\mathbf 1^\top A)B=\mathbf 1^\top B=\mathbf 1^\top$。
2. **谱范数 $\le1$**：$\|A\|_2\le\sqrt{\|A\|_1\|A\|_\infty}$（1-范数 = 最大绝对列和 = 1，∞-范数 = 最大绝对行和 = 1），故 $\|A\|_2\le1$；又 $A\mathbf 1=\mathbf 1$，所以 $\|A\|_2=1$。
3. **均值（总量）守恒**：对 $X\in\R^{n\times d}$，$\mathbf 1^\top(AX)=\mathbf 1^\top X$，即“$n$ 条流逐坐标求和”不变（列和为 1 的作用）；
   行和为 1 让每条输出流都是输入流的凸组合（原文 “functions as a convex combination of the input features”，§1）。
4. **反向同样守恒**：反向梯度 $\partial\mathcal L/\partial X_l=A^\top\,\partial\mathcal L/\partial X_{l+1}$（只看残差支路），$A^\top\in\mathcal B_n$，所以梯度也是凸组合、梯度总量守恒。
5. **“混合单调增加”的精确含义**：记 $\Pi=I-J_n$ 为去均值投影。$A\mathbf 1=\mathbf 1$ 且 $\mathbf 1^\top A=\mathbf 1^\top$ ⟹ $A$ 把 $\mathbf 1^\perp$ 映到 $\mathbf 1^\perp$，
   且 $\|A\Pi X\|_F\le\|\Pi X\|_F$。所以“各流偏离均值的总量”**非增**（等号在 $A$ 为置换矩阵时成立），严格减少需要 $\sigma_2(A)<1$。
   这是“弱单调”，不是严格单调；`guo2026ohc` 摘要从同一角度批评 mHC：流间差异随深度被耗尽。
6. **只有行和为 1 时的反向上界**（解释图 7 的 1.6）：若每个因子非负且行和严格为 1（SK 以 $\mathcal T_r$ 收尾时成立），
   则乘积 $P$ 也非负、行和为 1，于是 $P$ 的任一列和 $\le$ 全部元素之和 $=n$。故**在精确算术下前向增益恒为 1，反向增益 $\le n=4$**。
   **但论文的实测并非精确 1**：图 7(a) 单层前向数字化为 1.0000–1.0005，图 7(b) 复合前向为 1.0001–1.0066（SVG 坐标换算，核查时复算），
   图 8 中 mHC $\prod_{1..60}$ 的行和标签印着 1.00, 1.01, 1.01, 1.00。论文没有解释这点偏差（精度、eps 或取平均方式均有可能，见 §8）。
   课上应说“前向增益按构造≈1（精确算术下 = 1）”，不要说“实测恰为 1”。
   反例：$\begin{pmatrix}1&0\\1&0\end{pmatrix}$ 行和为 1，列和 $(2,0)$，谱范数 $\sqrt2$——只有行随机不够。
7. **维数与顶点**：$\mathcal B_n$ 由 $n^2$ 个非负元素与 $2n$ 个线性等式定义，其中恰有一个冗余（行和总和 = 列和总和），故维数 $n^2-(2n-1)=(n-1)^2$；
   顶点是 $n!$ 个置换矩阵（Birkhoff–von Neumann）。$n=3$：4 维、6 个顶点；$n=4$：9 维、24 个顶点。
8. **“流形”的措辞**：$\mathcal B_n$ 是凸多面体，有面、棱、顶点，不是光滑流形；它的相对内部（全部元素严格为正的双随机矩阵）才是 $(n-1)^2$ 维光滑流形。
   SK($\exp(\cdot)$) 的极限（$t\to\infty$）总在相对内部，永远到不了顶点（包括 $\mathbf I$），只能逼近；截断到 $t_{\max}=20$ 的输出元素严格为正，但列和只近似为 1，严格说并不在 $\mathcal B_n$ 上（见 E4）。

### 2.8 V4 / V4.1 的写法（`dsv4` §2.2 式 (1)–(8)；`dsv41` §2.4.1 式 (2)–(6)）

$$X_{l+1}=B_{l}X_{l}+C_{l}\mathcal{F}_{l}(A_{l}X_{l}) \tag{V4-1}$$
$$B_{l}\in\mathcal{M}\coloneq\{M\in\mathbb{R}^{n\times n}\mid M\mathbf{1}_{n}=\mathbf{1}_{n},\;\mathbf{1}_{n}^{T}M=\mathbf{1}_{n}^{T},\;M\geqslant 0\} \tag{V4-2}$$
$$\hat{X}_{l}=\operatorname{RMSNorm}(\operatorname{vec}(X_{l}))\in\R^{1\times n_{hc}d},\quad
\tilde{A}_{l}=\alpha_{l}^{\mathrm{pre}}(\hat{X}_{l}W^{\mathrm{pre}}_{l})+S_{l}^{\mathrm{pre}},\quad
\tilde{B}_{l}=\alpha_{l}^{\mathrm{res}}\operatorname{Mat}(\hat{X}_{l}W^{\mathrm{res}}_{l})+S_{l}^{\mathrm{res}},\quad
\tilde{C}_{l}=\alpha_{l}^{\mathrm{post}}(\hat{X}_{l}W^{\mathrm{post}}_{l})^{T}+S_{l}^{\mathrm{post}} \tag{V4-3,4,5}$$
$$A_{l}=\sigma(\tilde{A}_{l}),\qquad C_{l}=2\sigma(\tilde{C}_{l}),\qquad M^{(0)}=\exp(\tilde B_l),\ M^{(t)}=\mathcal{T}_{r}(\mathcal{T}_{c}(M^{(t-1)})),\ B_l=M^{(t_{\max})} \tag{V4-6,7,8}$$

V4.1（$n$ 条流，$(A_l,B_l,C_l)=\mathcal H(X_l)$）：

$$X_{l+1}=B_{l}X_{l}+C_{l}\,\mathcal{F}_{l}(A_{l}X_{l}),\qquad(A_{l},B_{l},C_{l})=\mathcal{H}(X_{l}) \tag{V4.1-2}$$
多趟实现：$X_l=B_{l-1}X_{l-1}+C_{l-1}Y_{l-1}$（读 $(n+1)d$）；$(A_l,B_l,C_l)=\mathcal H(X_l)$（读 $nd$）；$\hat X_l=A_lX_l$（读 $nd$）；三核共写 $(n+1)d$；
加上 $\mathcal F_l$ 的 pre-norm，总计 $(4n+4)d$。 (V4.1-3,4,5)

$$X_{l+1}=B_{l}X_{l}+C_{l}\,\mathcal{F}_{l}(A_{l-1}X_{l}),\qquad(A_{l},B_{l},C_{l})=\mathcal{H}(X_{l}) \tag{V4.1-6, Single-Pass}$$

**代码与论文的对照（`hf-v4flash-code`、`hf-v41flash-code`，笔者逐行核对）：**

- 系数向量 `mixes` 长 $(2+n)n=24$，顺序 `[pre(n), post(n), comb(n·n)]`；`hc_scale` 是 3 个 $\alpha$，`hc_base` 是 24 个静态偏置；`hc_*_fn` 形状 `[24, n·d]`（V4-Flash：`[24, 16384]`）。
- `pre = sigmoid(mixes*scale0 + base) + eps`（**多了 `hc_eps = 1e-6`**，论文式 (8) 没有）；`post = 2*sigmoid(...)`（与论文同）。
- Sinkhorn：第一步是对 `comb` 做**按行 softmax**（减最大值后取 exp 再除行和）并加 `eps`，然后除以（列和 + eps）；之后再做 `sinkhorn_iters-1 = 19` 轮“除以（行和+eps）、除以（列和+eps）”。合计 20 轮（行,列）对。
- `hc_post`：`y = post[...,None]*x[...,None,:] + sum(comb[...,:,:,None]*residual[...,:,None,:], dim=2)`，即 $y_j=\text{post}_j\,\mathcal F+\sum_i\text{comb}_{i,j}\,x_i$。
  所以论文的 $H^{\rm res}=\text{comb}^\top$。据此换算：代码先归一化 comb 的行（= $H^{\rm res}$ 的列，$\mathcal T_c$），后归一化 comb 的列（= $H^{\rm res}$ 的行，$\mathcal T_r$），**与论文式 (9) 的顺序一致**，
  最后一步使 $H^{\rm res}$ 的行和为 $s/(s+10^{-6})$，即精确到约 $10^{-6}$（前向增益≈1），列和只是近似（反向增益有偏差）——与 mHC 图 7(a) 的现象一致。
- 嵌入之后 `h.unsqueeze(2).repeat(1,1,hc_mult,1)`：把同一向量复制成 $n$ 条流（同 HC 的 $\mathbf H^0$）。
- V4 的输出头：`hc_head` 用 `sigmoid(mixes*scale+base)+eps` 的加权和把 $n$ 条流合成一条（**不是 HC 论文算法 1 的“按行求和”**），再 RMSNorm、再 logits；MTP 块有自己的一套 `hc_head_*`。
- V4.1 的 Single-Pass：`Block.forward` 里注意力子层用上一子层传来的 `pre_mix`，本子层的 `hc_mixes` 产生的 `attn_pre` 留给 FFN 子层用，FFN 的 `ffn_pre` 返回给下一个 block；
  第一个子层用 `make_identity_pre_mix`（one-hot，只取第 0 条流，注释 “initial one-hot mix”），最后用最后一个 `pre_mix` 收拢后接 `norm` 与 `head`。

### 2.9 访存与显存公式（mHC 表 2、表 3、式 (20)）

表 2（每 token、一个残差层的前向，不含 $\mathcal F$ 内部 I/O）：

| 方法 | 操作 | 读（元素） | 写（元素） |
|---|---|---|---|
| 残差 | Residual Merge | $2C$ | $C$ |
| HC | 计算 $\mathcal H^{\rm pre},\mathcal H^{\rm post},\mathcal H^{\rm res}$ | $nC$ | $n^2+2n$ |
| HC | $\mathcal H^{\rm pre}$ | $nC+n$ | $C$ |
| HC | $\mathcal H^{\rm post}$ | $C+n$ | $nC$ |
| HC | $\mathcal H^{\rm res}$ | $nC+n^2$ | $nC$ |
| HC | Residual Merge | $2nC$ | $nC$ |
| HC | **合计** | $(5n+1)C+n^2+2n$ | $(3n+1)C+n^2+2n$ |

逐行求和复核通过（§4 N10）。

表 3（每 token，保留给反向的激活）：$\mathbf x_{l_0}$：$nC$，每 $L_r$ 层存一次；$\mathcal F(\mathcal H^{\rm pre}_l\mathbf x_l,\mathcal W_l)$：$C$，每层存；
$\mathbf x_l$（$nC$）、$\mathcal H^{\rm pre}_l\mathbf x_l$（$C$）、$\mathrm{RMSNorm}(\mathcal H^{\rm pre}_l\mathbf x_l)$（$C$）：块内临时重算。

$$L_{r}^{*}=\arg\min_{L_{r}}\left[nC\times\left\lceil\frac{L}{L_{r}}\right\rceil+(n+2)C\times L_{r}\right]\approx\sqrt{\frac{nL}{n+2}} \tag{mHC-20}$$

推导（笔者补）：去掉取整，$f(L_r)=nCL/L_r+(n+2)CL_r$，$f'=0\Rightarrow L_r^2=nL/(n+2)$。

---

## 3. 规格与超参数表

### 3.1 mHC 实验模型（`xie2026mhc` v2，附录 A.1 表 5；合并单元格已展开）

| 属性 | 3B | 9B | 27B | 3B（1T tokens） | 来源 |
|---|---|---|---|---|---|
| Vocab Params | 331M | 496M | 662M | 331M | 表 5 |
| Active Params | 612M | 1.66B | 4.14B | 612M | 表 5 |
| Total Params | 2.97B | 9.18B | 27.0B | 2.97B | 表 5 |
| Layers | 12 | 18 | 30 | 12 | 表 5 |
| Leading Dense Layers | 1 | 1 | 1 | 1 | 表 5（合并格） |
| Routed / Active / Shared Experts | 64 / 6 / 2 | 64 / 6 / 2 | 72 / 6 / 2 | 64 / 6 / 2 | 表 5 |
| Dimension（即 $C$） | 1280 | 1920 | 2560 | 1280 | 表 5 |
| FFN Dimension | 896 | 1280 | 1536 | 896 | 表 5 |
| Load Balancing | Loss-Free (Wang et al., 2024) | 同 | 同 | 同 | 表 5 |
| Attention | MLA，16 / 24 / 32 头，头维 128，KV Rank 512，RoPE 维 64，θ=10000 | | | 16 头 | 表 5 |
| Norm | RMSNorm，ε = 1e-20 | | | | 表 5 |
| **mHC/HC 扩展率 $n$** | **4** | **4** | **4** | **4** | 表 5；§5.1 |
| **门控因子初值 $\alpha$** | **0.01** | 同 | 同 | 同 | 表 5 |
| **SK $t_{\max}$** | **20** | 同 | 同 | 同 | 表 5；§4.2 |
| Sequence Length | 4096 | | | | 表 5 |
| Vocab Size | 129280 | | | | 表 5 |
| Batch Size（序列） | 320 | 512 | 1280 | 2560 | 表 5 |
| Training Steps | 30000 | 50000 | 50000 | 100000 | 表 5 |
| Training Tokens | 39.3B | 105B | 262B | 1.05T | 表 5（复算见 N30） |
| Warmup Steps | 2000 | | | | 表 5 |
| Optimizer | AdamW，β=(0.9, 0.95)，ε=1e-20，weight decay 0.1 | | | | 表 5 |
| Base LR | 8.6e-4 | 5.9e-4 | 4.0e-4 | 9.0e-4 | 表 5 |
| LR Scheduler | Step；decay step ratio [0.8×, 0.9×]；decay rate [0.316, 0.1] | | | | 表 5（0.316 与 0.1 是否累乘未说明，§8） |

注：图 3/7/8 的 27B 横轴 1..60 = 30 层 × 2 子层（§2.0）。

### 3.2 HC 在 DeepSeek 生产模型中的设置（论文 vs config.json vs 代码）

| 模型 | 层数（block） | $d$ | 子层数 = 2×block | $n$ | $t_{\max}$ | hc_eps | 论文出处 | config.json | 一致？ |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 43 | 4096 | 86（另有 MTP 1 块） | 4 | 20 | — | `dsv4` §4.2.1 | `hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-06, hidden_size=4096, num_hidden_layers=43` | 一致 |
| DeepSeek-V4-Pro | 61 | 7168 | 122 | 4 | 20 | — | `dsv4` §4.2.1 | `hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-06, hidden_size=7168, num_hidden_layers=61` | 一致 |
| DeepSeek-V4.1-Flash | 40 | 5120 | 80 | 4 | 20 | — | `dsv41` §4.2.1（只说扩展率 4、SK 20 次） | `text_config`：`hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-06, hidden_size=5120, num_hidden_layers=40` | 一致 |

补充：config 的 `num_nextn_predict_layers`：V4-Flash = 1、V4-Pro = 1、V4.1-Flash = 3（V4.1 的 3 个草稿块属 DSpark，见 `dsv41` §2.4.3），表中“子层数”都不含这些块。V4-Flash `rms_norm_eps=1e-06`；V4.1-Flash `rms_norm_eps=1e-20`（与 mHC 表 5 的 1e-20 相同）。`hc_eps` 在论文里没有出现（代码专有，见 §7 E15）。
V4 的 mHC 静态偏置与门控因子用 AdamW，动态投影矩阵随“其余模块”用 Muon（`dsv4` §2.4）；V4.1 为 “AdamW … for the weights of all RMSNorm modules and other non-matrix parameters”（`dsv41` §4.2.2）。

### 3.3 HC 论文的实验规模

| 项目 | 值 | 出处 |
|---|---|---|
| 稠密基线 | OLMo-1B（ablation）、OLMo-7B；数据 dolma v1.5 sample；全部 500B tokens | `zhu2025hc` §4 “Experiment Settings” |
| MoE 基线 | OLMoE-1B-7B（激活 1.3B / 总 7B），OLMoE-MIX，500B tokens | 同上 |
| 默认方法 | DHC，$n=4$，含 tanh（后缀 -DHC）；-SHC 为静态 | §4.1 首句 |
| 维度 | OLMo-1B：$d=2048$，16 层（附录 B 例子 $2048\times\ldots\times2\times16$） | 附录 B |
| 权重衰减 | 静态部分不用 weight decay，动态部分用 | §4 “Implementation” |
| 输出层缩放 | 所有层输出模块（FFN 第二个线性层、注意力输出投影）的初始化 std 按 $\sqrt n$ 缩放（原文未写乘还是除，§8） | 同上 |
| 最终合流 | 最后一个 hyper hidden 矩阵按行求和，再 norm + unembedding | §2.1；算法 1 |

### 3.4 mHC 与 HC 的系数计算方式对照

| 项目 | HC（`zhu2025hc` 式 (10)–(13)、算法 2） | mHC（`xie2026mhc` 式 (7)(8)） | V4 代码 |
|---|---|---|---|
| 归一化 | `norm`（代码：LayerNorm(dim)，逐流） | RMSNorm，作用在展平的 $nC$ 向量上 | 除以 $\sqrt{\text{mean}(x^2)+\epsilon}$，放在矩阵乘之后 |
| 动态投影输入 | 每条流各自的 $d$ 维向量 | 全部 $nC$ | 全部 $n d$ |
| 激活 | tanh | 无 tanh；pre 用 σ、post 用 2σ、res 用 SK | 同 mHC，外加 eps |
| 门控 $\alpha$ 初值 | 0.01 | 0.01（表 5） | 权重文件载入，看不到初值 |
| $H^{\rm res}$ 约束 | 无 | 双随机（SK 20 次） | SK 20 次 |
| 静态初值 | $\mathbf B=\mathbf 1$，$\mathbf A_m=\mathbf e_{k\bmod n}$，$\mathbf A_r=\mathbf I$ | **未给出** | V4.1 首层 pre-mix 为 one-hot $\mathbf e_0$ |

---

## 4. 复算的数字

所有脚本在 scratch 目录（`L16/sim/*.py`），不入库。表达式可直接粘进 `python -c`。

### 4.1 HC 论文的账

- **N1** SHC 额外参数（OLMo-1B-SHC×4）：`4*(4+2)*2*16` = **768**（论文式 (22) 后例子同）；SHC×2：`2*4*2*16` = 256。
- **N2** DHC 额外参数：`(0+2048*6+4*6+2)*2*16` = **394,048**（论文同）；DHC×2：`(2048*4+2*4+2)*32` = 262,464。
- **N3** 用表 7 的“总参数差”反推：SHC×2 `(1.17676467-1.17676442)*1e9` ≈ 250；SHC×4 ≈ 760；DHC×2 ≈ 262,460；DHC×4 ≈ 394,040；
  OLMo-7B-DHC×4 `(6.88967027-6.88809574)*1e9` ≈ **1,574,530**；OLMoE-DHC×4 ≈ 394,050。
  与公式比：OLMo-7B（$d=4096$、$L=32$，这两个值是笔者从差值反推的，与 OLMo-7B 公开配置是否一致未核）`(4096*6+26)*2*32` = **1,574,528**，吻合总参数差，
  **但表 7 “HC Params(B)” 列写 0.0013124**；SHC 两行该列写 0.0000026 / 0.0000077，是公式值 0.000000256 / 0.000000768 的 10 倍（§7 E8）。
- **N4** OLMoE 若按正文 “$|\theta_{norm}|=d_{model}$” 计：`(2048+2048*6+26)*32` = 459,584，与表 7 总参数差 394,050 不符；按 $|\theta_{norm}|=0$ 计才吻合（§7 E9）。
- **N5** 表 8 FLOPs 增幅：`(2.3583/2.3536-1)*100` = 0.200%（DHC×4），`(13.3844/13.3647-1)*100` = 0.147%（7B），`(2.3629/2.3580-1)*100` = 0.208%（OLMoE）——与表一致。
- **N6** 表 9 显存增幅：`47.55/41.11` → +15.67%；`51.85/41.11` → +26.13%（表写 26.0%。按表中的 GB 数，四舍五入和截断都得 26.1%，所以 26.0% 与表内数据不自洽，可能是笔误或基于未舍入的原始 GB 数；DHC×4 的 51.86 → 26.15% 表写 26.1%，自洽）；`33.70/26.27` → +28.28%；`34.65/31.59` → +9.69%。
- **N7** ViT 的“相对提升”：Base `77.60-76.38` = 1.22 **百分点**（相对提升 `(77.60/76.38-1)*100` = 1.60%）；`77.26-76.38` = 0.88 pp（相对 1.15%）；
  Large `78.38-77.25` = 1.13 pp（相对 1.46%）、`79.94-77.25` = 2.69 pp（相对 3.48%）。论文把百分点叫作 “relative increases”（§7 E11）。
- **N8** 表 1 的“V3 Eval Loss 降 0.029”：DHC×8 W/O tanh `2.544-2.514` = **0.030**（V2：`2.811-2.777` = 0.034 与文字一致）。差 0.001，可能是未四舍五入的原始值。
- **N9** HC Post-Norm 的直通系数：若 $\sigma_i=\sigma_o=1,\sigma_{io}=0$，每层 $1/\sqrt2$；60 层后 `(2**-0.5)**60` = **9.31e-10**（=$2^{-30}$）。
  这是“Post-Norm 深层梯度消失”在 HC 视角下的一个算例（假设是笔者设的，不是论文数字）。

### 4.2 mHC 的访存、参数、FLOPs

- **N10** 表 2 合计复核：读 $nC+(nC+n)+(C+n)+(nC+n^2)+2nC=(5n+1)C+n^2+2n$；写 $(n^2+2n)+C+nC+nC+nC=(3n+1)C+n^2+2n$。$n=4$：读 $21C+24$，写 $13C+24$。
- **N11** 与普通残差（读 $2C$ 写 $C$，共 $3C$）之比：`(34*C+48)/(3*C)` → **11.34×**（$C$ 为千量级时与 $C$ 几乎无关）。
  bf16 每 token 每子层字节：27B（$C=2560$）残差 15,360 B vs HC 174,176 B；V4-Flash（$d=4096$）24,576 B vs **278,624 B**；V4-Pro（7168）43,008 B vs 487,520 B；V4.1-Flash（5120）30,720 B vs 348,256 B。
  （表 2 只含残差流维护，不含 $\mathcal F$ 内部；这里的字节数是笔者按 bf16 换算。）
- **N12** 融合 $\mathcal F_{\rm post,res}$：读 $(3n+1)C\to(n+1)C$ 即 13C → 5C（2.6× 少）；写 $3nC\to nC$ 即 12C → 4C（3× 少）。
- **N13** V4 的 “output dimension of only 24”：`n*n+2*n` = **24**（$n=4$）；代码 `mix_hc = (2 + hc_mult) * hc_mult` = 24。
- **N14** 每子层 mHC 参数 = $nd(n^2+2n)+(n^2+2n)+3$：
  27B `4*2560*24+24+3` = 245,787，×60 = 14,747,220（占 27.0B 的 0.055%）；
  V4-Flash `4*4096*24+27` = 393,243，×86 = **33,818,898**（占 284B 的 0.012%）；
  V4-Pro `4*7168*24+27` = 688,155，×122 = 83,954,910（占 1.6T 的 0.0052%）；
  V4.1-Flash `4*5120*24+27` = 491,547，×80 = 39,323,760（占 552B 的 0.0071%）。（未计 MTP/输出头的 `hc_head_*`：每个 $n\cdot nd+n+1$。）
- **N15** V4-Flash 每 token 每子层 mHC FLOPs（笔者估算，乘加算 2）：投影 `2*4*4096*24` = 786,432；$H^{\rm pre}X$ `2*4*4096` = 32,768；$H^{\rm res}X$ `2*16*4096` = 131,072；
  post 写回 32,768；SK 约 `20*2*16*2` = 1,280；合计 984,320；×86 = 84,651,520 ≈ 8.5e7。对比前向 $\approx2N_{act}=2.6\times10^{10}$：**约 0.33%**。
  结论：mHC 的代价不在 FLOPs，而在访存（N11）与流水线通信（C28）。
- **N16** V4.1 Single-Pass 访存（$n=4$）：多趟 $(4n+4)d=20d$；两趟 $(3n+2)d=14d$；单趟 = 下界 $(2n+2)d=10d$；`20/10` = **2.0**（“halving”）。
  V4.1-Flash（$d=5120$，bf16）每 token 每子层：204,800 B → 143,360 B → 102,400 B；×80 子层：16.38 MB → 11.47 MB → 8.19 MB（按 bf16 是笔者假设；V4.1 正文说 Mega-mHC 含 “FP8 conversion”，实际字节数可能更小）。
- **N17** 重计算块长（式 (20)，$n=4$）：
  $L=60$：`sqrt(4*60/6)` = 6.32，整数最优 $L_r=6$，显存 `4*ceil(60/6)+6*6` = **76C**；对比“每层都存 $\mathbf x_l$” `4*60` = 240C、“三种激活全存” `6*60` = 360C。
  $L=86$（V4-Flash 子层）：7.57 → 整数最优 8，92C；$L=122$（V4-Pro）：9.02 → 9，110C；$L=80$（V4.1-Flash）：7.30 → 8，88C。
  （这里把“层”当子层数用；论文未说明 $L$ 指 block 还是子层，§8。）

### 4.3 增益（gain）的数字

- **N18** 标量类比：`1.05**60` = 18.68；`1.1**60` = 304.5；`0.9**60` = 0.00180；`0.95**60` = 0.0461；
  **`3000**(1/60)` = 1.1428**——每个子层只要平均放大 14.3%，60 个子层就到 3000；`2825**(1/60)` = 1.1416；`509.1**(1/60)` = 1.1095。
- **N19** 三个数量级：`2825/1.649` = 1713（数字化值），`log10` = 3.23；按论文文字 `3000/1.6` = 1875，`log10` = 3.27。
- **N20** 随机无约束矩阵乘积（笔者模拟，`sim_gain.py`，$n=4$，60 个因子，$H=\mathbf I+\sigma G$，$G_{ij}\sim\mathcal N(0,1)$，2000 次，种子 0）：

  | $\sigma$ | 前向 Amax 中位 / p90 / 最大 | 反向 Amax 中位 / p90 / 最大 | 谱范数中位 / p90 / 最大 |
  |---|---|---|---|
  | 0.05 | 1.74 / 2.91 / 7.28 | 1.75 / 2.94 / 7.08 | 2.00 / 2.93 / 5.36 |
  | 0.1 | 2.80 / 7.64 / 34.3 | 2.85 / 7.68 / 33.7 | 4.15 / 9.34 / 35.0 |
  | 0.2 | 17.8 / 114 / 5.01e3 | 17.2 / 105 / 1.85e3 | 32.7 / 160 / 4.99e3 |
  | 0.3 | 354 / 4.8e3 / 1.95e5 | 345 / 4.76e3 / 2.87e5 | 741 / 7.55e3 / 3.1e5 |

  深度曲线（种子 42，前向 Amax 中位数，深度 0,5,…,60）：
  σ=0.1：1.0, 1.44, 1.61, 1.74, 1.85, 1.96, 2.10, 2.25, 2.32, 2.44, 2.57, 2.68, 2.89；
  σ=0.2：1.0, 1.87, 2.33, 2.79, 3.49, 4.34, 5.38, 6.44, 7.75, 9.80, 11.9, 14.8, 18.2（p90：1.0, 3.29, 4.89, 7.45, 10.8, 14.3, 19.4, 27.5, 38.3, 50.2, 66.9, 92.2, 124）；
  σ=0.3：1.0, 2.53, 3.97, 6.40, 10.1, 16.4, 24.7, 37.9, 62.8, 97.5, 155, 239, 405（p90：…, 1482, 2713, 5081）。
  读法：扰动只有 0.2–0.3 的量级，60 层后的中位增益就到 10¹–10²，尾部到 10³ 以上，与 HC 在 27B 上观察到的 3000 同一量级。**这是示意模拟，不是对 HC 训练动态的复现。**
- **N21** SK 投影后（同上设置，$H=\mathrm{SK}_{20}(\exp(\beta\mathbf I+sG))$，500 次）：前向 Amax 最大 **1.000000**（行归一化收尾）；
  反向 Amax 中位 / 最大：(β,s)=(0,1) 1.0000/1.0001；(3,1) 1.0016/1.0146；(0,3) 1.0035/1.0408；(5,3) 1.0183/1.1109；(0,6) 1.0246/1.1208；(8,4) 1.0400/1.1658；谱范数最大 ≤ 1.0055。
  近单位阵情形 (β,s)=(5,2)（单层矩阵对角约 0.96–0.97，形似图 8 的 $\mathcal P(\mathcal H_{30})$）：反向 Amax 中位随深度 1.0, 1.019, 1.022, … 稳定在 1.023，p90 稳定在 1.044；
  $\max|P-J_4|$ 中位：深度 5 为 0.507，30 为 0.020，60 为 0.0072（i.i.d. 因子下乘积很快趋于均匀矩阵；真实模型的图 8 在 31–60 子层的乘积仍保留 0.62/0.80 的对角，说明学到的因子并非 i.i.d.）。
- **N22** 单层 SK 截断误差（反向增益 = 最大列和，$n=4$，logits $sG$，5000 次）：

  | logit 尺度 $s$ | t=1 中位/p99 | t=3 | t=5 | t=10 | t=20 |
  |---|---|---|---|---|---|
  | 0.5 | 1.040/1.143 | 1.0002/1.0033 | 1.0000/1.0001 | 1.0000/1.0000 | 1.0000/1.0000 |
  | 1 | 1.128/1.427 | 1.005/1.052 | 1.0003/1.011 | 1.0000/1.0004 | 1.0000/1.0000 |
  | 2 | 1.300/1.924 | 1.050/1.277 | 1.012/1.105 | 1.0004/1.027 | 1.0000/1.0065 |
  | 4 | 1.469/2.366 | 1.148/1.751 | 1.067/1.372 | 1.017/1.101 | 1.0021/1.037 |
  | 8 | 1.545/2.703 | 1.216/2.041 | 1.124/1.978 | 1.051/1.402 | 1.019/1.076 |

  结论：logits 越悬殊（越接近置换矩阵），20 次迭代的残差越大；图 7(a) 单层反向增益 1.01–1.06 对应的就是这种“尖锐”的 $\tilde H^{\rm res}$（笔者推测）。

### 4.4 Sinkhorn 算例

- **N23** 3×3 算例 $A=\begin{pmatrix}4&1&1\\1&4&1\\2&2&2\end{pmatrix}$（笔者选的；行和 6,6,6 已经相等，列和 7,7,4 不等——“只做行归一化不够”）：
  $\mathcal T_c(A)=\begin{pmatrix}4/7&1/7&1/4\\1/7&4/7&1/4\\2/7&2/7&1/2\end{pmatrix}$，行和 $27/28,27/28,15/14$；
  $\mathbf M^{(1)}=\mathcal T_r\mathcal T_c(A)=\begin{pmatrix}16/27&4/27&7/27\\4/27&16/27&7/27\\4/15&4/15&7/15\end{pmatrix}$，列和 $136/135,136/135,133/135$（= 1.00741, 1.00741, 0.98519）。
  最大列和误差：t=1 1.48e-2；t=2 6.46e-4；t=3 2.81e-5；t=4 1.23e-6；t=5 5.34e-8；t=6 2.33e-9；t=7 1.01e-10；t=8 4.41e-12；t=10 8.4e-15。
  相邻误差比稳定在 **0.0436**（线性收敛）。
  极限 $=\begin{pmatrix}0.58899&0.14725&0.26376\\0.14725&0.58899&0.26376\\0.26376&0.26376&0.47247\end{pmatrix}$（$t=20$ 与极限五位小数一致）；
  $D_1\propto(1.11652,1.11652,1)$，$D_2\propto(0.13188,0.13188,0.23624)$。极限的奇异值 $1,\ 0.44174,\ 0.20871$。
- **N24** 该极限的 Birkhoff 分解（贪心，置换写作“第 i 行的 1 在第 p(i) 列”）：
  $0.47247\,\mathbf I+0.14725\,P_{(1,2,0)}+0.14725\,P_{(2,0,1)}+0.11652\,P_{(0,2,1)}+0.11652\,P_{(2,1,0)}$，权重和 = 1.000000。
  （分解不唯一；这是一种。）
- **N25** 不满足“完全支撑”的 2×2：$A=\begin{pmatrix}1&1\\0&1\end{pmatrix}$。精确迭代（分数算）：$\mathbf M^{(t)}=\begin{pmatrix}\frac{2t}{2t+1}&\frac{1}{2t+1}\\0&1\end{pmatrix}$，
  列和 $\frac{2t}{2t+1},\frac{2t+2}{2t+1}$（t=1..7 逐项验证；归纳证明：对 $\begin{pmatrix}a&b\\0&1\end{pmatrix}$ 做 $\mathcal T_c$ 得右上 $\frac{b}{1+b}$，再做 $\mathcal T_r$ 得 $\frac{1}{2t+3}$）。
  $t=20$：列和偏差 $1/41$ = **0.0244**；收敛到 $\mathbf I$ 只有 $O(1/t)$ 速度。
  近退化的正矩阵 $\begin{pmatrix}1&1\\e^{-k}&1\end{pmatrix}$ 迭代 20 次后的列和偏差：k=2 → 0.0000；k=5 → 0.0004；k=10 → 0.0232；k=20 → 0.0244。
  含义：logit 差达到 ~10 时，20 次 SK 就留下约 2.4% 的列和误差——和图 7(a) 的量级一致（笔者推测的解释）。

### 4.5 从图 8 数字复核

图 8 是 PNG，矩阵元素与行/列和标签都印在图上（两位小数），笔者放大后逐格转录（见 §5.2 图 8 条目）。

- **N26** HC $\mathcal H^{\rm res}_1$ 行和：`5.43+3*4.43` = 18.72（标签 18.73，舍入）；列和 −6.81 ×4。$\mathcal H^{\rm res}_1-\mathbf I$ 的四列完全相同，即 $\mathcal H^{\rm res}_1=\mathbf I+\mathbf u\mathbf 1^\top$，$\mathbf u=(4.43,-4.07,-3.95,-4.22)^\top$（观察，原因论文未讨论）。
- **N27** Amax 的定义核对：若按“绝对值的列和”，$\mathcal H^{\rm res}_1$ 第一列为 `5.43+4.07+3.95+4.22` = 17.67；图 3(a) $l=1$ 的反向值（数字化）是 **6.81**，等于“列和的绝对值”。故 Amax = $\max_j|\sum_i P_{ij}|$。
- **N28** 图 3 与图 8 互相印证：HC $\prod_{1..30}$ 前向 6.46–6.47（图 8 标签 6.47；图 3(b) $l=30$ 数字化 6.47）；$\prod_{31..60}$ 反向 489（图 8 列和 −489.0；图 3(b) $l=31$ 数字化 489）；
  $\prod_{1..60}$ 前向 509.1（图 8；图 3(b) $l=60$ 数字化 509）；反向 259.2（图 8；图 3(b) $l=1$ 数字化 259）。mHC $\prod_{31..60}$ 反向 1.50（图 8）与图 7(b) $l=31$ 数字化 1.504 一致。
- **N29** 乘积一致性（练习素材）：HC 图 8 中 $\prod_{31..60}\times\prod_{1..30}$ 的第一行算得 $(-132.3,-110.5,-115.1,-111.0)$，图上 $\prod_{1..60}$ 第一行 $(-132.8,-112.2,-117.0,-113.3)$，相差 ≤ 2%；
  mHC 算得第一行 $(0.240,0.262,0.231,0.272)$ vs 图上 $(0.24,0.26,0.23,0.27)$。不完全相等的原因：图中每个矩阵都是“对一个序列的所有 token 取平均”，平均的乘积 ≠ 乘积的平均。
  HC $\prod_{1..60}$ 的奇异值 487.9, 0.149, 0.057, 0.003——**几乎是秩 1**；mHC $\prod_{1..60}$ 与 $J_4$（全 0.25）的最大差 0.04。

### 4.6 实验设置复核

- **N30** tokens = batch × 4096 × steps：`320*4096*30000` = 3.932e10（39.3B）；`512*4096*50000` = 1.049e11（105B）；`1280*4096*50000` = 2.621e11（262B）；`2560*4096*100000` = 1.049e12（1.05T）。全部一致。
- **N31** “dataset size proportional to its parameters” 的比例：tokens / 激活参数 = 64.3、63.2、63.3；tokens / 总参数 = 13.2、11.4、9.7。**按激活参数成比例（约 63–64 tokens/激活参数）**（笔者推断，论文没说是哪种参数）。
- **N32** $6N_{act}D$：3B 1.44e20；9B 1.04e21；27B 6.51e21；3B-1T 3.85e21。图 6(a) 横坐标数字化为 1.64e20、1.24e21、7.94e21，约为 $6N_{act}D$ 的 1.14–1.22 倍（多出部分大概是注意力等非参数 FLOPs，笔者推测）。
- **N33** LR 衰减步：`0.8*50000` = 40,000，`0.9*50000` = 45,000；图 5(b) 的梯度范数在 40k 处陡降（数字化：基线 0.057@30k → 0.036@40k），与 step 调度吻合。`1/sqrt(10)` = 0.3162 ≈ 0.316。
- **N34** 表 4（27B）差值：mHC−基线：BBH +7.2，DROP +6.9，GSM8K +7.1，HellaSwag +1.0，MATH +4.0，MMLU +4.4，PIQA +2.0，TriviaQA +3.3；
  mHC−HC：+2.1，+2.3，+0.6，+0.4，**−0.4**，+0.4，+0.6，+1.3。八项平均：基线 53.125，HC 56.700，mHC 57.6125（≈57.61）。

---

## 5. 图表清单与重绘建议

### 5.1 mHC 论文（`xie2026mhc` v2）的图

| 图 | 标题（摘要） | 展示内容 | 本课是否重绘 |
|---|---|---|---|
| 1 | Illustrations of Residual Connection Paradigms | (a) 残差 (b) HC：Res/Pre/Post Mapping，$h^{\rm res},h^{\rm in},h^{\rm out},h^{\rm post}$ (c) mHC：三个映射都换成 $\mathcal P_{\mathcal M}(\cdot)$ | **是**（主结构图） |
| 2 | Training Instability of HC（27B） | (a) HC 相对 mHC 的 loss 差；(b) 梯度范数 | 是（合并进稳定性图） |
| 3 | Propagation Instability of HC（27B） | (a) 单层 $\mathcal H^{\rm res}_l$ 的前向/反向 Amax（对数轴 10⁰–10¹）；(b) 复合映射（对数轴 10¹–10⁵） | **是** |
| 4 | Communication-Computation Overlapping for mHC | 扩展的 DualPipe：普通计算流、通信流、高优先级计算流 | 是（示意） |
| 5 | Training Stability of mHC（27B） | (a) mHC、HC 相对基线的 loss 差；(b) 三者梯度范数 | 是 |
| 6 | Scaling properties of mHC | (a) compute scaling（3B/9B/27B）(b) token scaling（3B，1T） | 是 |
| 7 | Propagation Stability of mHC（27B） | 同图 3，线性轴 0–2 | **是**（与图 3 并排） |
| 8 | Visualizations of Learnable Mappings | HC 与 mHC 各 6 个 4×4 矩阵（单层 1/30/60，复合 1–30、31–60、1–60），带行和/列和 | **是** |

表：1（HC 组件消融）、2（每 token 访存）、3（重计算激活）、4（27B 下游）、5（模型规格）。

### 5.2 HC 论文（`zhu2025hc` v3）的图

| 图 | 内容 | 重绘？ |
|---|---|---|
| 1 | OLMoE-1B-7B vs DHC×4：训练 loss、C4-en 验证 loss、HellaSwag、ARC-C 曲线 | 否（只引数字） |
| 2 | $n=2$ 的 HC：(a) 残差 (b) HC（β₁,β₂,α₁,₀,α₂,₀,α₁,₁,α₂,₁,α₁,₂,α₂,₂）(c) 深度连接 (d) 宽度连接 | **是**（HC 矩阵图） |
| 3 | 相邻层输入的余弦相似度（OLMo-1B，Pre-Norm vs HC） | 否 |
| 4 | $n=2$ 的串行 / 并行排布 | **是** |
| 5 | 不同扩展率的训练 loss（有 / 无 tanh） | 否 |
| 6 | 7B 曲线 | 否 |
| 7 | 连接矩阵可视化（HC 与 Pre-Norm/Post-Norm/PTB/two-hop 基线） | 可选 |
| 8 | 带 HC 的 Transformer | 否（用 mHC 图 1 代替） |
| 9–18 | 附录曲线、ViT、DHC 权重分布、展开连接矩阵等 | 否 |

### 5.3 本课建议的 15 张图（每张附绘制所需的数据）

1. **图 16.1 残差 / HC / mHC 三联结构图**（据 mHC 图 1 重绘，机制图）。标真实形状：$X_l\in\R^{4\times d}$（画 4 条带），$H^{\rm pre}\in\R^{1\times4}$，$H^{\rm post}\in\R^{1\times4}$，$H^{\rm res}\in\R^{4\times4}$；
   mHC 面板里三个映射框旁分别写 “σ(·)∈(0,1)”、“2σ(·)∈(0,2)”、“Sinkhorn×20 → 双随机”。颜色：流 = `--fig-ink` 细线，层 $\mathcal F$ = `--fig-blue`，约束 = `--fig-green`，HC 无约束 = `--fig-red`。
2. **图 16.2 HC 矩阵的分块读法**（据 HC 式 (1) 与图 2 重绘）。画 $(n+1)\times(n+1)=3\times3$（$n=2$）网格：左上 0；第一行 $\beta_1,\beta_2$（= $H^{\rm post}$，写回）；第一列 $\alpha_{1,0},\alpha_{2,0}$（= $H^{\rm pre\top}$，读出）；右下 2×2 $\alpha_{i,j}$（= $H^{\rm res\top}$，流间）。
   配初始化值：$\beta=(1,1)$，$\alpha_{\cdot,0}=\mathbf e_{k\bmod2}$，右下 = $\mathbf I$。
3. **图 16.3 Pre-Norm / Post-Norm / 串行 / 并行都是 HC 的特例**：四个小矩阵 $\begin{pmatrix}0&1\\1&1\end{pmatrix}$、Post-Norm（元素 $1/\sqrt{\sigma_i^2+\sigma_o^2+2\sigma_{io}}$，数值例 $1/\sqrt2=0.707$）、
   串行 $\begin{pmatrix}0&1&1\\1&1&0\\0&0&1\end{pmatrix}$、并行 odd/even 两个矩阵（§2.2），旁边配 HC 图 4 式的两层数据流小图。
4. **图 16.4 乘积的指数律（账单图，计算得出）**：横轴深度 0–60，纵轴对数；曲线 $g^L$，$g\in\{0.9,0.95,1,1.05,1.1,1.1428\}$；标出 $1.1428^{60}\approx3008$、$1.1^{60}=304.5$、$0.9^{60}=0.0018$；水平虚线 3000。
5. **图 16.5 HC 的增益爆炸（据 mHC 图 3 重绘，数据为数字化值）**。横轴子层 $l=1..60$。
   (a) 单层（对数轴）前向：18.7, 1.02, 0.911, 0.954, 0.927, 0.644, 0.785, 0.68, 1.03, 0.809, 1.17, 0.838, 1.19, 1.12, 1.32, 0.819, 1.01, 1.04, 1.39, 0.89, 1.38, 0.677, 1.34, 0.957, 1.16, 0.769, 1.07, 0.848, 1.16, 0.955, 1.37, 0.992, 1.59, 1.21, 1.16, 1.43, 1.18, 0.839, 1.11, 1.02, 1.03, 1.1, 1.64, 1.14, 1.24, 1.06, 1.19, 1.2, 1.31, 0.905, 1.07, 1.55, 1.87, 1.63, 1.47, 1.64, 2.81, 2.68, 9.02, 22.5；
   反向：6.81, 0.996, 0.871, 0.84, 1.01, 0.741, 0.937, 0.916, 1.02, 1.19, 0.94, 0.91, 0.985, 0.945, 1, 0.963, 0.785, 0.843, 1.28, 0.925, 1.28, 0.702, 1.14, 1.23, 1.19, 0.943, 1.42, 0.95, 1.21, 0.828, 1.3, 0.734, 1.25, 0.974, 1.31, 1.12, 1.02, 0.921, 1.19, 1.24, 1.04, 0.944, 1.77, 1.19, 1.25, 0.968, 1.22, 1.13, 1.55, 0.796, 1.09, 1.68, 1.53, 1.4, 1.42, 1.74, 2.64, 2.83, 6.56, 12。
   (b) 复合（对数轴 10¹–10⁵）前向 $\prod_{i=1}^{l}\mathcal H_{l+1-i}$：18.7, 26.7, 27.6, 28.8, 27.8, 26.6, 25, 22.7, 19.7, 20, 18.1, 15.2, 13.6, 9.83, 8.99, 8.99, 8.82, 9.25, 8.97, 9.64, 9.29, 8.34, 7.7, 8.2, 8.38, 7.25, 7.78, 7.48, 7.19, 6.47, 6.4, 6.87, 6.46, 6.34, 6.72, 6.69, 6.71, 6.53, 6.68, 5.64, 4.95, 4.93, 4.7, 5.06, 4.83, 5.24, 5.64, 5.65, 5.63, 5.23, 5.15, 6.11, 7.19, 8.4, 9.15, 11.5, 12.2, 15.2, 34.7, 509；
   反向 $\prod_{i=1}^{61-l}\mathcal H_{61-i}$：259, 27.6, 24.7, 23.9, 25.6, 27, 29.9, 32, 35.9, 39.8, 46.3, 52.8, 62.7, 70.4, 82.9, 90.9, 106, 115, 118, 94.2, 141, 190, 199, 205, 244, 263, 288, 298, 420, 390, 489, 550, 584, 720, 838, 909, 1200, 1270, 1440, 1330, 1570, 1460, 1800, 2740, 2640, 2420, 2650, 2490, 2790, 2510, 2780, 2820, 2040, 1480, 1090, 804, 486, 216, 79.4, 12。
   标注峰值 ≈2825（$l=52$），论文文字“peaks of 3000”。
6. **图 16.6 mHC 的增益守恒（据 mHC 图 7 重绘，线性轴 0–2）**。前向单层与复合都 ≈1.000（数字化范围 1.0000–1.0005 与 1.0001–1.0066）；
   单层反向：1.037, 1.027, 1.027, 1.033, 1.032, 1.054, 1.061, 1.054, 1.041, 1.045, 1.036, 1.032, 1.044, 1.035, 1.03, 1.041, 1.044, 1.032, 1.045, 1.025, 1.018, 1.021, 1.021, 1.012, 1.011, 1.013, 1.023, 1.022, 1.032, 1.04, 1.039, 1.049, 1.039, 1.044, 1.04, 1.038, 1.034, 1.022, 1.03, 1.03, 1.03, 1.021, 1.036, 1.034, 1.034, 1.023, 1.025, 1.026, 1.021, 1.025, 1.028, 1.027, 1.024, 1.034, 1.019, 1.033, 1.031, 1.051, 1.022, 1.006；
   复合反向：1.11, 1.071, 1.233, 1.211, 1.182, 1.325, 1.378, 1.451, 1.507, 1.528, 1.555, 1.553, 1.554, 1.563, 1.566, 1.553, 1.599, 1.623, 1.624, 1.649, 1.645, 1.646, 1.646, 1.641, 1.636, 1.631, 1.623, 1.599, 1.58, 1.555, 1.504, 1.501, 1.52, 1.513, 1.484, 1.46, 1.434, 1.41, 1.395, 1.376, 1.353, 1.331, 1.315, 1.284, 1.279, 1.27, 1.258, 1.255, 1.247, 1.232, 1.22, 1.211, 1.2, 1.177, 1.141, 1.126, 1.091, 1.062, 1.027, 1.006。
   建议与图 16.5 左右并排，并画一条 $y=n=4$ 的虚线标“行随机乘积的反向上界”（§2.7 第 6 条）。
7. **图 16.7 模拟：无约束 vs 双随机（计算得出）**。用 N20/N21 的深度曲线：σ=0.1/0.2/0.3 三条中位线（对数轴），σ=0.2 的 p10–p90 阴影可用 p90 数据；mHC(β=5,s=2) 前向 1.0、反向中位 ≈1.023、p90 ≈1.044。
   说明写清“示意模拟，i.i.d. 高斯扰动”。
8. **图 16.8 Sinkhorn 分步图（机制图）**：用 N23 的 3×3 算例画 4 格：$A$（行和 6,6,6 / 列和 7,7,4）→ $\mathcal T_c$（分数矩阵，行和 27/28,27/28,15/14）→ $\mathcal T_r$（$\mathbf M^{(1)}$，列和 136/135,…）→ 极限（五位小数）；
   下方小图：最大列和误差对 $t$ 的对数图（1.48e-2, 6.46e-4, 2.81e-5, 1.23e-6, 5.34e-8, 2.33e-9, 1.01e-10, 4.41e-12），并与 2×2 反例 $\frac{1}{2t+1}$ 的慢收敛（t=1..20）画在同一坐标，对照“线性收敛 vs $O(1/t)$”。
9. **图 16.9 Birkhoff 多面体（示意）**：左：$\mathcal B_2$ 是线段 $\{\begin{pmatrix}a&1-a\\1-a&a\end{pmatrix}\}$，两端 $\mathbf I$ 与交换矩阵，中点 $J_2$；
   右：$\mathcal B_3$ 是 4 维、6 个顶点，**画不出来**，改画它的一个 2 维精确切片——循环矩阵三角形 $a\mathbf I+b\mathbf C+c\mathbf C^2$（$a+b+c=1$），三个顶点 $\mathbf I,\mathbf C,\mathbf C^2$，重心 $J_3$；
   再标注恒等式 $\mathbf I+\mathbf C+\mathbf C^2=J\cdot3=P_{(12)}+P_{(13)}+P_{(23)}$（两个三角形共享重心）。可加 N24 的分解权重条形图（0.472, 0.147, 0.147, 0.117, 0.117）。
10. **图 16.10 学到的映射热力图（据 mHC 图 8 重绘，数值取自图上印刷值）**。两行六列：
    HC：$\mathcal H_1$ = [[5.43,4.43,4.43,4.43],[−4.07,−3.07,−4.07,−4.07],[−3.95,−3.95,−2.95,−3.95],[−4.22,−4.22,−4.22,−3.22]]（行和 18.73, −15.29, −14.79, −15.88；列和 −6.81×4）；
    $\mathcal H_{30}$ = [[0.94,−0.07,−0.05,0.02],[−0.08,0.89,−0.07,−0.07],[−0.10,−0.14,0.81,−0.07],[0.06,0.05,−0.03,0.87]]（行和 0.84, 0.67, 0.49, 0.96；列和 0.83, 0.73, 0.66, 0.75）；
    $\mathcal H_{60}$ = [[−5.58,−3.74,−5.71,−6.60],[−6.06,−2.27,−5.33,−6.57],[6.08,3.12,6.53,6.77],[−6.41,−3.97,−5.72,−5.49]]（行和 −21.64, −20.22, 22.50, −21.59；列和 −11.97, −6.86, −10.23, −11.89）；
    $\prod_{1..30}$ = [[−0.38,−0.33,−0.34,−0.31],[1.81,1.56,1.58,1.51],[0.01,−0.00,0.01,0.01],[−0.23,−0.19,−0.20,−0.19]]（行和 −1.35, 6.47, 0.03, −0.81；列和 1.22, 1.04, 1.06, 1.02）；
    $\prod_{31..60}$ = [[−69.9,−68.3,−255.3,142.1],[−69.1,−66.1,−247.4,139.6],[74.8,72.7,268.9,−151.8],[−71.2,−71.8,−255.2,143.3]]（行和 −251.4, −243.0, 264.6, −254.8；列和 −135.4, −133.4, −489.0, 273.3）；
    $\prod_{1..60}$ = [[−132.8,−112.2,−117.0,−113.3],[−129.3,−109.3,−113.9,−110.3],[142.3,120.2,125.3,121.3],[−139.3,−117.8,−122.6,−118.7]]（行和 −475.3, −462.8, 509.1, −498.5；列和 −259.2, −219.1, −228.2, −221.0）。
    mHC：$\mathcal P(\mathcal H_1)$ = [[0.67,0.09,0.03,0.22],[0.26,0.48,0.26,0.00],[0.03,0.24,0.00,0.73],[0.03,0.20,0.69,0.09]]（行和 1.00×4；列和 0.98, 1.00, 0.98, 1.04）；
    $\mathcal P(\mathcal H_{30})$ = [[0.96,0.01,0.00,0.04],[0.00,0.97,0.03,0.00],[0.00,0.00,1.00,0.00],[0.00,0.04,0.01,0.95]]（列和 0.96, 1.02, 1.04, 0.99）；
    $\mathcal P(\mathcal H_{60})$ = [[0.92,0.06,0.01,0.01],[0.05,0.81,0.01,0.13],[0.00,0.01,0.97,0.02],[0.03,0.13,0.00,0.84]]（列和 1.00, 1.01, 0.99, 1.00）；
    $\prod_{1..30}$ = [[0.30,0.25,0.22,0.24],[0.24,0.25,0.25,0.26],[0.20,0.24,0.28,0.28],[0.17,0.32,0.18,0.34]]（列和 0.90, 1.06, 0.93, 1.11）；
    $\prod_{31..60}$ = [[0.35,0.28,0.17,0.20],[0.03,0.62,0.29,0.07],[0.01,0.17,0.80,0.02],[0.02,0.42,0.25,0.31]]（列和 0.41, 1.50, 1.50, 0.60）；
    $\prod_{1..60}$ = [[0.24,0.26,0.23,0.27],[0.23,0.25,0.26,0.27],[0.21,0.25,0.27,0.28],[0.21,0.27,0.24,0.29]]（行和 1.00, 1.01, 1.01, 1.00；列和 0.88, 1.03, 1.00, 1.11）。
    HC 与 mHC 用两套色标（HC 发散色标对称于 0；mHC 顺序色标 0–1）。
11. **图 16.11 训练稳定性（据 mHC 图 2、图 5 重绘，数字化采样点，示意精度）**：
    (a) 相对基线的 loss 差（步数 5k,10k,12k,15k,20k,30k,40k,50k）：HC −0.0443, −0.0353, −0.0333, −0.0283, −0.0256, −0.0211, −0.0211, −0.0166；mHC −0.0435, −0.0361, −0.0342, −0.0324, −0.0310, −0.0275, −0.0253, −0.0218。
    (b) HC 减 mHC 的 loss 差（图 2a）：−0.0008, 0.0008, 0.0009, 0.0041, 0.0054, 0.0064, 0.0043, 0.0052（12k 之后抬升，与 C25 对应）。
    (c) 梯度范数（图 5b，5k…50k，并加 45k）：基线 0.091, 0.082, 0.072, 0.069, 0.064, 0.057, 0.036, 0.034(45k), 0.039；HC 0.106, 0.098, 0.103, 0.120, 0.127, 0.144, 0.077, 0.114(45k), 0.133；mHC 0.143, 0.111, 0.102, 0.097, 0.092, 0.107, 0.041, 0.037(45k), 0.050。
    竖虚线标 40k、45k（LR 衰减点）。曲线本身噪声大，建议只画平滑后的趋势并注明“据图数字化”。
12. **图 16.12 规模化曲线（据 mHC 图 6 重绘，数字化值）**：(a) compute scaling：FLOPs 1.64e20 / 1.24e21 / 7.94e21（3B/9B/27B），绝对 loss 差 −0.0292 / −0.0233 / −0.0212，相对 loss 比 98.51% / 98.68% / 98.66%；
    (b) token scaling（3B，1T）：FLOPs 0.874e21, 1.75e21, 2.62e21, 3.50e21, 4.37e21，绝对差 −0.0242, −0.0206, −0.0186, −0.0173, −0.0153，相对比 98.72%, 98.88%, 98.98%, 99.05%, 99.13%。
13. **图 16.13 访存账（账单图，计算得出）**：每 token 每子层元素数（以 $d$ 为单位，$n=4$）：残差 3；HC 未融合 34（+48 个标量）；mHC 融合后的 post-res 核 9（读 5 + 写 4）；
    V4 多趟 20、Mega-mHC(mHC) 14、Single-Pass 10（= 下界）。第二个面板用 V4.1-Flash $d=5120$、bf16 的字节数：30,720 / 348,256 / 204,800 / 143,360 / 102,400 B。
14. **图 16.14 重计算块长的 U 形曲线（计算得出）**：$f(L_r)=4\lceil60/L_r\rceil+6L_r$（单位 $C$），$L_r=1..30$；最小值 $L_r=6$ 处 76C；连续近似 $\sqrt{40}=6.32$；参考线 240C（每层存 $\mathbf x_l$）。
15. **图 16.15 Single-Pass mHC 的依赖图（据 V4.1 §2.4.1 与代码绘制，机制图）**：上图 V4 多趟：$X_l$ ← 残差更新（读 $(n+1)d$）→ 系数 $\mathcal H(X_l)$（读 $nd$，需在隐藏维上归约完才得到 $A_l$）→ 输入混合 $A_lX_l$（再读 $nd$）；
    下图 Single-Pass：同一次按 tile 遍历中完成残差更新、用 $A_{l-1}$ 做输入混合、累加下一块的系数；标 “读 $(n+1)d$ + 写 $(n+1)d$”。
    DualPipe 重叠示意（mHC 图 4）可作为本图的附图：普通计算流依次为 MLP(B)、MLP(W)、MLP(F)、$\mathcal F^{M}_{pre}$(B)、$\mathcal F^{A}_{post,res}$(B)、ATTN(B)、$\mathcal F^{A}_{pre}$(B)、ATTN(W)、Whole Stage Recompute(B)、$\mathcal F^{A}_{pre}$(F)、ATTN(F)、$\mathcal F^{A}_{post,res}$(F)、$\mathcal F^{M}_{pre}$(F)；
    通信流：DISPATCH(F)、DISPATCH(B)、COMBINE(F)、PP Send Recv(F)、PP Send Recv(B)、COMBINE(B)；高优先级计算流：$\mathcal F^{M}_{post,res}$(F)（位于 COMBINE(F) 与 PP Send Recv(F) 之间）、$\mathcal F^{M}_{post,res}$(B)（位于 PP Send Recv(B) 与 COMBINE(B) 之间）。论文注明块长仅示意。

---

## 6. 算例与练习素材（附答案）

1. **（计算）3000 倍需要每层放大多少？** 60 个子层复合增益 3000，若每层相同，单层增益 $g=3000^{1/60}=1.1428$。再问：$g=1.05$ 时 60 层后是多少？答：18.7。$g=0.95$？答：0.046。
   要点：单层看起来“接近 1”的矩阵，乘 60 次就不接近了（图 3(a) 的单层值大多在 0.64–1.87 之间，复合却到 2825）。
2. **（计算）读图 8 求增益。** 给出 HC $\mathcal H^{\rm res}_1$ 的 16 个元素，求前向增益（最大 |行和|）与反向增益（最大 |列和|）。答：18.72（图标 18.73）、6.81。
   追问：如果把定义误写成“绝对值之和”，反向增益是多少？答：17.67——所以论文的 Amax 是“和的绝对值”。
3. **（推导）双随机的三条性质。** 证明：(a) 乘法封闭；(b) $\|A\|_2=1$；(c) $\mathbf 1^\top AX=\mathbf 1^\top X$。答：见 §2.7 第 1–3 条（$\|A\|_2\le\sqrt{\|A\|_1\|A\|_\infty}$）。
4. **（推导）为什么 mHC 的前向增益（在精确算术下）恰好是 1，而反向不是？上界是多少？** 答：SK 以行归一化收尾 ⟹ 每个 $H^{\rm res}$ 行和精确为 1 ⟹ 乘积行和为 1 ⟹ 前向增益 = 1；列和只是近似 1。
   （注意：论文实测的复合前向增益最高约 1.007（图 7(b) 数字化），图 8 标签有 1.01，所以这是精确算术下的结论，不是实测值。）
   非负且行和为 1 的矩阵乘积，任一列和 ≤ 全部元素和 = $n$，故反向增益 ≤ 4。论文实测最大约 1.6，远低于上界。反例 $\begin{pmatrix}1&0\\1&0\end{pmatrix}$ 说明只有行随机时谱范数可达 $\sqrt2$。
5. **（手算）Sinkhorn 一步。** 对 $A=\begin{pmatrix}4&1&1\\1&4&1\\2&2&2\end{pmatrix}$ 先列归一化、再行归一化，写出 $\mathbf M^{(1)}$ 和它的列和。
   答：$\mathbf M^{(1)}=\begin{pmatrix}16/27&4/27&7/27\\4/27&16/27&7/27\\4/15&4/15&7/15\end{pmatrix}$，列和 $136/135,136/135,133/135$。追问：误差每轮大约乘多少？答：约 0.0436（t=2 时 6.46e-4）。
6. **（推导 + 计算）Sinkhorn 什么时候慢？** 对 $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ 用归纳法证明 $\mathbf M^{(t)}$ 右上元素为 $1/(2t+1)$，求 $t=20$ 时的列和误差。答：$1/41=0.0244$。
   联系：$\exp(\tilde H)$ 中若某些元素比别的小 $e^{-10}$ 倍，情形接近这个反例（k=10 时 20 次后误差 0.0232）。
7. **（概念）Birkhoff 多面体的维数与“3×3 示意”。** $\mathcal B_4$ 几维、几个顶点？$\mathcal B_3$ 呢？能不能在纸上画 $\mathcal B_3$？
   答：$(n-1)^2$ 维、$n!$ 个顶点：$\mathcal B_4$ 9 维 24 顶点；$\mathcal B_3$ 4 维 6 顶点，不能直接画；可画循环矩阵切片（三角形 $\mathbf I,\mathbf C,\mathbf C^2$），并验证 $\mathbf I+\mathbf C+\mathbf C^2=P_{(12)}+P_{(13)}+P_{(23)}=$ 全 1 矩阵。
8. **（计算）Birkhoff 分解。** 把 N23 的极限矩阵写成置换矩阵的凸组合。答：见 N24（权重 0.47247, 0.14725, 0.14725, 0.11652, 0.11652）。
9. **（计算）访存账。** $n=4$、$d=4096$、bf16：普通残差每 token 每子层读写多少字节？未融合 HC 呢？答：24,576 B vs 278,624 B（11.34×）。
   追问：V4.1 的 Single-Pass 相对 V4 多趟实现省多少？答：$(4n+4)d\to(2n+2)d$，20d → 10d，省一半。
10. **（计算）参数与 FLOPs 都是零头。** V4-Flash 一个子层的 mHC 参数量？全模型？占 284B 多少？每 token FLOPs 占比？
    答：393,243；×86 = 33.8M；0.012%；约 8.5e7 FLOPs / 2.6e10 ≈ 0.33%。所以 6.7% 的时间开销来自访存和通信，不来自算术。
11. **（计算）重计算块长。** $n=4$，$L=60$：由式 (20) 求 $L_r^*$，并算整数最优时的激活显存（单位 $C$），与“每层都存 $\mathbf x_l$”比较。答：6.32；$L_r=6$ 时 $4\cdot10+6\cdot6=76C$ vs $240C$，约 3.2 倍。
12. **（概念）为什么约束 $H^{\rm res}$ 而不是 $H^{\rm pre}/H^{\rm post}$ 为双随机？** 答（要点）：式 (mHC-4) 里跨层**连乘**的只有 $H^{\rm res}$；$H^{\rm pre}$、$H^{\rm post}$ 每条路径上只出现一次，
    只需非负防抵消（C31）；另外 $\mathcal F$ 内有 pre-norm，$H^{\rm pre}$ 的整体尺度不影响层输入（§2.4，笔者推导）。

---

## 7. 勘误与常见误解

- **E1｜“mHC 让残差恒等映射可达 / 恢复了恒等映射”——要说精确。** $\mathbf I$ 是 $\mathcal B_n$ 的顶点，但 SK($\exp(\cdot)$) 的输出元素严格为正，只能无限逼近 $\mathbf I$ 而到不了；V4 代码还在 softmax 后加 `eps=1e-6`。
  mHC 恢复的是“恒等映射的性质”（均值守恒、非扩张、任意深度复合仍守恒），不是 $H^{\rm res}=\mathbf I$。课程地图里“恒等映射可达”应改为“可任意逼近”。
- **E2｜“Norm Preservation” 是误称。** 论文标题写 Norm Preservation，内容其实是 $\|H\|_2\le1$（非扩张）。双随机矩阵可以强烈压缩（$J_n$ 把所有偏离均值的分量压成 0）。
  图 8 中 mHC 60 层复合几乎就是 $J_4$。准确说法：“范数不增、均值守恒”。
- **E3｜“HC 本来就不稳定”——不准确。** HC 原论文在 1B/7B/OLMoE 上报告 DHC “no spikes”，比基线还稳（C16）。不稳定是 DeepSeek 在 27B MoE（约 12k 步）观察到的（C25），
  V4 论文也说是“stacking multiple layers”时“frequently”出现。说法应带上规模与来源。
- **E4｜“mHC 的 Sinkhorn 输出是双随机矩阵”——只是近似。** $t_{\max}=20$，单层反向增益 1.01–1.06，复合最大约 1.6（C39）；因此“谱范数 ≤ 1”对实际矩阵也只是近似成立（图 8 的 $\prod_{31..60}$ 按印刷值算谱范数 1.14）。
  前向增益在精确算术下为 1，是因为迭代以行归一化收尾。但实测并不恰为 1：图 7(b) 的复合前向增益数字化最高约 1.0066，图 8 中 mHC $\prod_{1..60}$ 有两行的行和标签印着 1.01（论文未解释）。
- **E5｜SK 迭代顺序在不同来源里相反。** mHC 式 (9)、V4 式 (8)：$\mathcal T_r(\mathcal T_c(\cdot))$（先列后行）；Mena 2018 式 (1)：$\mathcal T_c(\mathcal T_r(\cdot))$（先行后列）。
  V4 代码存的是 `comb` $=H^{\rm res\top}$，代码里“先行后列”在 $H^{\rm res}$ 坐标下正是“先列后行”，与 mHC 一致。写课时要说明：顺序决定有限步后哪一组和是精确的。
- **E6｜Amax Gain 不是矩阵 ∞-范数。** 是“行和（列和）的绝对值”的最大值，符号相消会让它比 ∞-范数小（N27）。对非负矩阵二者相同，所以对 mHC 无差别，对 HC 有差别。
- **E7｜“mixing … monotonically”只在弱意义下成立。** 严格表述：各流偏离均值的 Frobenius 范数非增（§2.7 第 5 条）；置换矩阵（含 $\mathbf I$）时不减少。论文没有给证明。
- **E8｜HC 论文表 7 的 “HC Params(B)” 列有笔误。** SHC×2 / SHC×4 两行是公式值的 10 倍（0.0000026 vs 0.000000256；0.0000077 vs 0.000000768）；OLMo-7B-DHC×4 行写 0.0013124，
  而总参数差（1,574,530）与公式（$d=4096,L=32$：1,574,528）都指向 0.0015745。总参数列与 Δ% 列是自洽的。引用参数量时用公式或总参数差，不要用该列。
- **E9｜HC 附录 B 说 OLMoE 的 $|\theta_{norm}|=d_{model}$，但表 7 的 OLMoE 数字只与 $|\theta_{norm}|=0$ 吻合**（N4）。二者必有一处与实际实现不符；本笔记无法判定。
- **E10｜mHC §3.2 把 “widely adopted pre-norm Transformer” 引到 Vaswani et al. 2017，但原版 Transformer 是 Post-LN**（C4）。Pre-LN 的分析应引 Xiong et al. 2020。
- **E11｜HC 的 ViT 结果把“百分点”称为“relative increases”。** 1.22% / 0.88% / 1.13% / 2.69% 都是 top-1 的绝对差；相对提升应为 1.60% / 1.15% / 1.46% / 3.48%（N7）。
- **E12｜mHC 表 4 的 “2.1% on BBH、2.3% on DROP” 同样是百分点**（EM / F1 的绝对差），不是相对百分比。
- **E13｜HC 引 Liu et al. 2020 支持“Pre-Norm 导致表示坍缩（representation collapse）”，但 Liu 2020 全文没有 “collapse” 一词**（v3 全文检索 0 处）；
  它的说法是 Pre-LN 对残差支路依赖弱，“often produces inferior models”。“表示坍缩”这一说法在 HC 中主要靠自己的图 3（相邻层余弦相似度）支撑。属于“邻近论断替换”。
- **E14｜mHC 复述 HC 时写 RMSNorm，HC 原文写 `norm`、代码用 LayerNorm。** 无害，但写课时不要说“HC 用 RMSNorm”。
- **E15｜代码与论文的小差异**：(a) `pre = σ(·) + 1e-6`；(b) Sinkhorn 首步是逐行 softmax + eps，除法分母加 eps；(c) V4 输出头用 sigmoid 加权合流，V4.1 用最后一个 pre-mix 合流，而 HC 论文是按行求和；
  (d) V4.1 首个子层的 pre-mix 是 one-hot $\mathbf e_0$。均不改变结论，但“照论文实现”与“照代码实现”数值不会逐位相同。
- **E16｜“6.7% 开销”的分母不同。** mHC 论文：相对训练时间的 “additional time overhead”；V4 论文：“6.7% of the overlapped 1F1B pipeline stage”。数字相同，口径未必相同，写课时引原句并注明出处。
- **E17｜HC 的 FLOPs 很小 ≠ HC 很便宜。** FLOPs +0.2%（表 8），但显存 +9.7%~+28%（表 9），访存约 11 倍（N11）；mHC 的 6.7% 是在融合核、重计算、DualPipe 改造之后。
- **E18｜课程地图写“Birkhoff 多面体（3×3 示意）”。** $\mathcal B_3$ 是 4 维的，不能如实画成平面或立体图；用 $\mathcal B_2$（线段）加 $\mathcal B_3$ 的循环切片（三角形），并在图注中说明（§5.3 图 16.9）。
- **E19｜HC 论文内部的交叉引用错误（据 arXiv v3 的 LaTeXML HTML；PDF 版编号未逐一核对）。** 图 2 标题与 §1 都写 “The Transformer with HC is shown in Fig. 17”，而 v3 中该图是图 8（图 17 是训练曲线；这句话 v1 没有，是 v2/v3 加的）；
  §1 与 §2.2 说实验在 “§ 5”，实际在 §4（§5 是 Related Work；v1 已如此）；图 2 标题列出 “α0,0, α0,1”，式 (1) 中不存在这两个下标（α 的行号从 1 开始）。
- **E20｜HC 表 3 与表 1/2 的 PPL 不一致。** 同一配置 DHC×4（含 tanh）：表 1/2 的 V2 PPL 17.509、V3 PPL 13.826，表 3 末行为 17.835、13.807；DHC×4 W/O tanh：表 1/2 为 17.451/13.844，表 3 为 17.773/13.823。
  loss 与下游平均一致，只有 PPL 列不同。另有 DHC×2 W/O tanh 的 V3 loss：表 1 为 2.537、表 2 与表 4 为 2.529；DHC×4 的 V3 loss：表 1 为 2.514、表 2 为 2.515。
- **E21｜“mHC 就是把残差流变宽”——不完整。** 变宽是 HC 的贡献；mHC 的贡献是约束 + 基础设施。mHC 表 1 的消融显示收益的大头（−0.022/−0.027）来自 $H^{\rm res}$ 的流间交换。
- **E22｜V4 与 mHC 引 HC 的年份不同**：mHC 写 Zhu et al., 2024（arXiv），V4 写 Zhu et al., 2025（ICLR 2025）。是同一篇论文。
- **E23｜“DeepSeek 发明了 Hyper-Connections”——错。** HC 来自 ByteDance Seed（`zhu2025hc`）；DeepSeek 的工作是 mHC。

---

## 8. 未能核实 / UNVERIFIED

1. **mHC 静态偏置 $\mathbf b^{\rm pre},\mathbf b^{\rm post},\mathbf b^{\rm res}$ 的初值。** 论文与 V4 论文都没写；代码只加载训练好的权重（`torch.empty`）。$H^{\rm res}$ 在初始化时是否接近 $\mathbf I$（需要大对角 logits）未知。
2. **6.7% 是在哪个模型、多少卡、什么并行配置下测的。** mHC 只说 “in-house large-scale training”；V4 说是“overlapped 1F1B pipeline stage”的墙钟开销。二者是否同一次测量未知。
3. **mHC 表 1 的消融规模**（“preliminary experiments”）未写模型大小与 token 数。
4. **mHC 实验里 HC 基线的具体实现**：是否用式 (5)（逐流、带 tanh），还是用式 (7) 的展平投影只去掉约束，论文未说明。
5. **Amax 的“selected sequence”是哪一条、多长。** 图 3/7/8 都基于它；无法复现。
6. **学习率衰减率 [0.316, 0.1] 是否累乘**（最终 0.1 还是 0.0316 倍）。表 5 未说明；DeepSeek 系其他论文的多步调度惯例可参考但本笔记未核。
7. **式 (20) 中的 $L$ 指 block 数还是子层数**，论文未说明（N17 按子层算）。
8. **Birkhoff 1946 的原始文献未见。** 卷期页码 “Univ. Nac. Tucumán Revista Ser. A 5: 147–151 (1946)” 只来自 Wikipedia（Birkhoff polytope 条目）等二手索引；ScienceOpen 记录页返回 403。von Neumann 1953 的独立证明、Kőnig 1916 的等价结果同样只见二手来源。
9. **Sinkhorn 收敛速率的定量理论**（例如以极限矩阵次大奇异值的平方为收敛因子的结论）本笔记没有核到一手来源；N23 观察到的 0.0436 与 $\sigma_2^2=0.195$ 不相等，所以不要在课里写“收敛因子 = $\sigma_2^2$”。
10. **DeepSeek-V4 arXiv 编号 2606.19348 与 v1 提交时间 2026-04-26 不一致**（2606 前缀对应 2026 年 6 月）。arXiv 页面原样如此，原因未查明。
11. **HC 论文“输出模块初始化 std 按 $\sqrt n$ 缩放”的方向**（乘还是除）。原文 “by a factor of √n”，从“保持最终求和后的 std 不变”推断应是除以 $\sqrt n$，但未见代码确认。
12. **OLMo-7B 的 $d=4096$、$L=32$** 是笔者从表 7 差值反推并与公式吻合，未去 OLMo 论文核对。
13. **图中数字化的数值**（§5.3 图 16.5/16.6/16.11/16.12 的曲线点）来自 SVG 路径坐标与网格线标定，精度受线宽与平滑影响；图 16.11 的采样点取“最近顶点”，噪声大。它们可以画图，但**不要当作论文数字引用**。
    图 8 的矩阵元素是图上印刷的两位小数，由笔者放大读出（已用行和/列和标签交叉校验），仍可能有个别读错。
14. **mHC 的 Single-Pass 版本对质量的影响**只有 V4.1 一句 “negligible performance degradation”，没有数字。
15. **“memory wall” 的出处。** mHC 把它引到 Dao et al. 2022（FlashAttention），该术语更早的出处通常认为是 Wulf & McKee 1995——此为笔者记忆，**未核实**，不要写进课文。
16. **mHC 是否有同行评审版本**：截至 2026-09-24 arXiv comments 为空，未查到会议录用信息（没有专门检索 OpenReview）。
17. **oHC（2609.02672）只读了摘要**，其定理与实验未核；只能作为“后续有人从 X 角度批评”的提示，不能引其数字。
18. **TileLang、DeepSeek-V3（DualPipe）只核了元数据**，DualPipe 的气泡与显存细节属于第 12 课笔记。
19. **mHC 实测前向增益为何略大于 1。** 图 7(b) 复合前向数字化最高约 1.0066，图 8 的 $\prod_{1..60}$ 行和标签有 1.01。
    按式 (9)（以 $\mathcal T_r$ 收尾），每个因子的行和应精确为 1，按 token 取平均也不改变行和，所以偏差来源不明（bf16 精度？训练核与式 (9) 不同？标签的计算方式？）。论文没有说明。

---

## 9. 参考文献卡片草稿

```
id: zhu2025hc
title: Hyper-Connections
a: Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou (2025). ICLR 2025. arXiv:2409.19606 (v3 read)
url: https://arxiv.org/abs/2409.19606
finding: 把残差流复制成 n 条，用可学习（静态或输入相关）的 (n+1)×(n+1) 矩阵同时决定层的读出、写回与流间混合；n=4 时在 OLMo/OLMoE 上稳定优于 Pre-Norm 残差，参数与 FLOPs 增量可忽略。
matters: 它是 mHC 要约束的对象：本课的三映射记号、初始化、Pre/Post-Norm 作为特例、以及“无约束残差矩阵”的来源都出自这里。
```

```
id: xie2026mhc
title: mHC: Manifold-Constrained Hyper-Connections
a: Zhenda Xie, Yixuan Wei, Huanqi Cao, et al. (DeepSeek-AI) (2025). Preprint. arXiv:2512.24880 (v2 read)
url: https://arxiv.org/abs/2512.24880
finding: 把 HC 的残差映射用 Sinkhorn-Knopp（20 次）投影到双随机矩阵，读出/写回用 sigmoid 非负化，使 27B MoE 上复合增益从近 3000 降到约 1.6，并以核融合、重计算与 DualPipe 改造把 n=4 的额外训练时间压到 6.7%。
matters: 本课主体：公式、稳定性分析、基础设施账和 3B/9B/27B 实验都以它为准，也是 DeepSeek-V4 采用 mHC 的直接依据。
```

```
id: dsv4
title: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
a: DeepSeek-AI (2026). Preprint. arXiv:2606.19348 (v1 read)
url: https://arxiv.org/abs/2606.19348
finding: V4-Pro（1.6T/49B）与 V4-Flash（284B/13B）在每个子层使用 mHC（n_hc=4，t_max=20），mHC 的静态偏置与门控因子用 AdamW 训练，墙钟开销为重叠后 1F1B 流水段的 6.7%。
matters: 证明 mHC 不是论文玩具而是进入了生产级模型，并提供 A/B/C 记号、确定性（输出维 24 的小矩阵乘）等工程细节。
```

```
id: dsv41
title: DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression
a: DeepSeek-AI (2026). Preprint. arXiv:2609.19969 (v1 read)
url: https://arxiv.org/abs/2609.19969
finding: 提出 Single-Pass mHC：输入混合改用上一子层预测的系数，使部署时残差流只需读写各一次，激活访存从 (4n+4)d 降到下界 (2n+2)d，据称质量几乎无损。
matters: 让本课的“访存账”延伸到时间线终点，说明 mHC 的瓶颈在带宽而非算力，并为第 19 课铺路。
```

```
id: he2016identity
title: Identity Mappings in Deep Residual Networks
a: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016). ECCV 2016. arXiv:1603.05027 (v3 read)
url: https://arxiv.org/abs/1603.05027
finding: 恒等捷径使任意两层之间存在不经权重层的前向与反向直通项；若捷径乘以标量 λ，其连乘会随深度指数放大或消失并造成优化困难。
matters: mHC 式 (4) 正是其式 (7) 的矩阵版，本课用它说明“为什么复合映射必须守恒”。
```

```
id: xiong2020preln
title: On Layer Normalization in the Transformer Architecture
a: Ruibin Xiong et al. (2020). ICML 2020. arXiv:2002.04745 (v2 read)
url: https://arxiv.org/abs/2002.04745
finding: 初始化时 Post-LN 最后一层 FFN 的梯度量级为 O(d√ln d)、与层数无关，而 Pre-LN 为 O(d√(ln d / L))，因为 Pre-LN 最终 LayerNorm 的输入规模随 L 线性增长。
matters: 给“Pre-LN 与 Post-LN 的两难”提供有前提的定理，替代 mHC 对 Vaswani 的误引。
```

```
id: liu2020admin
title: Understanding the Difficulty of Training Transformers
a: Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, Jiawei Han (2020). EMNLP 2020. arXiv:2004.08249 (v3 read)
url: https://arxiv.org/abs/2004.08249
finding: 在 15 组超参数中 Pre-LN 全部收敛而 Post-LN 有 7 组发散，但 Post-LN 收敛时 8 组中有 7 组优于 Pre-LN；Post-LN 更依赖残差支路。
matters: 用实测数字支撑“跷跷板”叙事，同时提醒 HC 对它的“表示坍缩”引用并不贴切。
```

```
id: vaswani2017
title: Attention Is All You Need
a: Ashish Vaswani et al. (2017). NeurIPS 2017. arXiv:1706.03762 (v7 read)
url: https://arxiv.org/abs/1706.03762
finding: 原版 Transformer 的每个子层输出为 LayerNorm(x + Sublayer(x))，即 Post-LN。
matters: 只用于勘误：纠正 mHC 把 pre-norm Transformer 归到本文。
```

```
id: sinkhorn1967
title: Concerning nonnegative matrices and doubly stochastic matrices
a: Richard Sinkhorn, Paul Knopp (1967). Pacific Journal of Mathematics 21(2):343–348. doi:10.2140/pjm.1967.21.343
url: https://msp.org/pjm/1967/21-2/p14.xhtml
finding: 非负矩阵交替行列缩放收敛到双随机极限当且仅当它含一条正对角线；存在 D1AD2 双随机的充要条件是每个正元素都位于某条正对角线上。
matters: mHC 所用算法的命名出处；也解释了为什么 mHC 先取 exp——正矩阵自动满足条件。
```

```
id: sinkhorn1964
title: A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices
a: Richard Sinkhorn (1964). Annals of Mathematical Statistics 35(2):876–879. doi:10.1214/aoms/1177703591
url: https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-2/A-Relationship-Between-Arbitrary-Positive-Matrices-and-Doubly-Stochastic-Matrices/10.1214/aoms/1177703591.full
finding: 任意严格正方阵都可以左右各乘一个正对角阵变成双随机矩阵（据题名与 Mena 2018 的转述；正文未读）。
matters: mHC 的 M(0)=exp(H̃) 严格为正，真正用到的正是这一篇覆盖的情形。
```

```
id: birkhoff1946
title: Tres observaciones sobre el algebra lineal
a: Garrett Birkhoff (1946). Univ. Nac. Tucumán Revista Ser. A 5:147–151. (bibliographic data from secondary sources only)
url: https://en.wikipedia.org/wiki/Birkhoff_polytope
finding: 双随机矩阵集合是置换矩阵的凸包，其顶点恰为 n! 个置换矩阵（Birkhoff–von Neumann 定理）。
matters: 支撑 mHC 的“残差映射 = 置换的凸组合”这一几何解释以及本课的 Birkhoff 多面体图。
kind: 原文未见，卡片按二手索引填写
```

```
id: mena2018gumbel
title: Learning Latent Permutations with Gumbel-Sinkhorn Networks
a: Gonzalo Mena, David Belanger, Scott Linderman, Jasper Snoek (2018). ICLR 2018. arXiv:1802.08665 (v1 read)
url: https://arxiv.org/abs/1802.08665
finding: 定理 1 证明 Sinkhorn 算子 S(X/τ) 等于 Birkhoff 多面体上 ⟨P,X⟩ + τ·熵 的最大化解，τ→0 时趋于线性指派（置换）。
matters: 给 mHC 一句话带过的“entropically project”一个可写进课文的精确定义，并解释 logits 越尖锐 SK 越接近置换、也越难收敛。
```

```
id: tilelang2025
title: TileLang: A Composable Tiled Programming Model for AI Systems
a: Lei Wang et al. (2025). Preprint. arXiv:2504.17577 (metadata only)
url: https://arxiv.org/abs/2504.17577
finding: 一种分块（tile）式 GPU 核编程模型（仅核对了题名与作者）。
matters: mHC 的大部分融合核与 V4 推理代码中的 Sinkhorn 核都用它编写，是“基础设施”一节的工具出处。
```

```
id: dsv3
title: DeepSeek-V3 Technical Report
a: DeepSeek-AI (2024). Preprint. arXiv:2412.19437 (metadata only for this lesson)
url: https://arxiv.org/abs/2412.19437
finding: 提出 DualPipe 流水线调度（细节见第 12 课笔记）。
matters: mHC 在其上扩展了通信-计算重叠；mHC 实验模型也按 V3 架构搭建。
```

```
id: guo2026ohc
title: oHC: Orthogonal Hyper-Connections on SO(4) via Quaternions
a: Haoqiang Guo et al. (2026). Preprint (not peer reviewed). arXiv:2609.02672 (abstract only)
url: https://arxiv.org/abs/2609.02672
finding: 摘要称双随机约束只能靠缩小流间差异来降低范数、使流随深度趋同，因而改把残差矩阵限制在旋转群 SO(n)，n=4 时用一对单位四元数闭式参数化。
matters: 作为“开放问题”：mHC 的守恒是以耗尽流间多样性为代价的，这一批评与本课图 8 的观察一致。
```

```
id: hf-v4flash-code
title: DeepSeek-V4-Flash inference code (inference/model.py, inference/kernel.py) and config.json
a: DeepSeek-AI (2026). Hugging Face repository deepseek-ai/DeepSeek-V4-Flash
url: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
finding: 代码中每个 block 有注意力与 FFN 两套 mHC 参数（24×4d 投影、24 个偏置、3 个门控），Sinkhorn 以逐行 softmax 起步共 20 轮，并在 pre 与分母中加 1e-6；config 为 hc_mult=4、hc_sinkhorn_iters=20。
matters: 用于核对论文与实现的一致性（迭代顺序、系数布局、输出头合流方式），并发现论文未写的 eps 细节。
kind: artefact
```

```
id: hf-v41flash-code
title: DeepSeek-V4.1-Flash inference code and config.json
a: DeepSeek-AI (2026). Hugging Face repository deepseek-ai/DeepSeek-V4.1-Flash
url: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
finding: Block.forward 中每个子层使用上一子层产生的 pre-mix（首层为 one-hot），与 V4.1 论文的 Single-Pass 公式一致；config 为 hc_mult=4、hc_sinkhorn_iters=20、hidden_size=5120、40 层。
matters: 证实 Single-Pass mHC 已进入发布的推理代码，而不只是论文描述。
kind: artefact
```

```
id: iclr2025-hc-page
title: Hyper-Connections — ICLR 2025 proceedings abstract page
a: ICLR (2025). proceedings.iclr.cc
url: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f1e8ff97057e0c3f6bcb7018c78d4df1-Abstract-Conference.html
finding: 记载 HC 为 ICLR 2025 会议论文，作者 8 人与 arXiv 一致。
matters: 确认 HC 的发表状态（课程引用时写 ICLR 2025 而非“预印本”）。
kind: artefact
```

---

## 核查记录（2026-09-24，对抗式复核）

复核方法：用 `arxiv.py text/grep` 拉取 mHC v2、HC v3、DeepSeek-V4 v1、V4.1 v1、He 2016 v3、Xiong 2020 v2、Liu 2020、Vaswani v7、Mena 2018 v1 全文逐句比对；
arXiv abs 页核对版本历史与作者数；HF 原样下载 V4-Flash / V4-Pro / V4.1-Flash 的 `config.json` 与 `inference/model.py`、`kernel.py`；
msp.org 与 Project Euclid 核对 Sinkhorn 两篇；proceedings.iclr.cc 核对 HC 的会议信息；mHC 图 3、图 7 的 SVG 按网格线标定后重新数字化，图 8 PNG 目视逐格比对；
所有派生数字在 Python 中复算（精确分数算 Sinkhorn 算例）。

**确认无误（原样保留）**

| # | 论断 | 核对结果与来源 |
|---|---|---|
| 1 | mHC 2512.24880：v1 2025-12-31、v2 2026-01-05；v1 19 位作者、v2 20 位，新增 Kuai Yu | arXiv abs 页的提交历史与 `citation_author` 计数 |
| 2 | HC 2409.19606：v1 2024-09-29 / v2 2024-11-28 / v3 2025-03-18，8 位作者，ICLR 2025 | arXiv abs 页；proceedings.iclr.cc 原文 “International Conference on Learning Representations 2025 (ICLR 2025)” |
| 3 | mHC 式 (1)–(9)、(10)–(20) 的转录 | 与 v2 LaTeXML 逐符号一致（含 $\mathcal T_r(\mathcal T_c(\cdot))$ 的顺序、$\varphi$ 形状、$r=\lVert\vec x\rVert_2/\sqrt{nC}$） |
| 4 | “peaks of 3000” / “nearly 3000” / “approximately 1.6” / “around the 12k step” / “final loss reduction of 0.021” / “t_max=20 as a practical value” | mHC v2 §3.1、§5.4、§5.2、§4.2 原句 |
| 5 | 6.7% 的两处原句 | mHC §1 与 §4.3；V4 §3.4.2 原句 “6.7% of the overlapped 1F1B pipeline stage” |
| 6 | 表 2 各行读写元素与合计 | mHC v2 表 2；复算合计 $(5n+1)C+n^2+2n$ / $(3n+1)C+n^2+2n$ |
| 7 | 融合核把读从 $(3n+1)C$ 降到 $(n+1)C$、写从 $3nC$ 降到 $nC$；除式 (14)–(15) 外用 TileLang；MLP 的 $\mathcal F_{\rm post,res}$ 放高优先级流 | mHC §4.3.1、§4.3.3 原句 |
| 8 | 表 4 八项分数、mHC−基线 / mHC−HC 差值（MATH −0.4） | mHC 表 4；Python 复算 |
| 9 | 表 5 全部单元格（27B：30 层、$C=2560$、72/6/2 专家、batch 1280、50k 步、262B tokens、$\alpha$=0.01、$t_{\max}$=20） | mHC 附录 A.1 表 5；tokens = batch×4096×steps 复算一致 |
| 10 | Amax 是“和的绝对值”（N27、E6） | 图 3 SVG 重新数字化：$l=1$ 前向 18.732、反向 6.811，图例确认灰 = 前向、蓝 = 反向；图 8 $\mathcal H_1$ 列和 −6.81，绝对值列和为 17.67 |
| 11 | 复合反向峰值 ≈2825（$l=52$），509.1 / 259.2 / 489.0 / 6.47 | 图 3(b) SVG 复算 2824.7 @ $l=52$，其余各点与 §5.3 列表一致 |
| 12 | 图 7：mHC 复合反向最大 1.649 | 图 7 SVG 复算 1.6492 |
| 13 | 图 8 的 12 个矩阵转录 | PNG 目视逐格比对无误；行和/列和与标签在舍入内一致；E4 的谱范数 1.14 与 N29 的乘积第一行复算一致 |
| 14 | HC 表 7 的“HC Params(B)”列笔误（E8）、OLMoE 的 $\theta_{norm}$ 参数量矛盾（E9） | 复算：SHC×2/×4 公式值 256 / 768 vs 表列 2600 / 7700；7B 总参数差 1,574,530 与公式 1,574,528 吻合，与表列 1,312,400 不符；OLMoE 按 θ_norm = d_model 得 459,584，按 θ_norm = 0 得 394,048 ≈ 总参数差 394,050 |
| 15 | HC “no spikes”（§4.1 与 §4.3）、表 1（n=1 V2 loss 2.819 vs 2.811）、表 5/6、ViT 百分点 vs 相对提升（E11）、表 1–3 的 PPL 不一致（E20） | HC v3 原文与表；Python 复算 1.60% / 1.15% / 1.46% / 3.48% |
| 16 | V4 式 (1)–(8)；Flash 43 层 $d=4096$、Pro 61 层 $d=7168$；$n_{hc}=4$、$t_{\max}=20$；“output dimension of only 24”；静态偏置与门控用 AdamW | V4 v1 §2.2、§4.2.1、§3.3、§2.4 原句；HF config.json |
| 17 | V4.1 Single-Pass：$(4n+4)d$ “twice the lower bound”、下界 $(2n+2)d$、Mega-mHC $(3n+2)d$、“negligible performance degradation”、“halving …” | V4.1 v1 §2.4.1 原句；V4.1-Flash config（40 层、5120、hc_mult 4、iters 20）与 `make_identity_pre_mix` 代码 |
| 18 | 代码：$H^{\rm res}=\text{comb}^\top$，首步行 softmax + eps，再 19 轮（行、列），合计 20 对，在 $H^{\rm res}$ 坐标下与式 (9) 顺序一致；`pre` 加 1e-6；`hc_head` 用 sigmoid 加权和 | V4-Flash `kernel.py` 第 386–425 行、`model.py` `hc_pre/hc_post/hc_head` |
| 19 | Sinkhorn–Knopp 1967 PJM 21(2):343–348，doi 10.2140/pjm.1967.21.343；Sinkhorn 1964 AMS 35(2):876–879，1964 年 6 月 | msp.org 摘要页、Project Euclid 页 |
| 20 | He 2016 式 (3)–(8) 与三处引文；Xiong 定理 1（含 $(\epsilon,\delta)$ 前提与概率界）；Liu 2020 “15 settings / 7 out of 8”，全文 “collapse” 0 处；Vaswani “LayerNorm(x + Sublayer(x))”；mHC 把 pre-norm 引到 Vaswani（§3.2） | 各论文全文 grep |
| 21 | 复算 N1–N6、N9、N11、N14–N19、N23–N25、N30–N34 | Python；全部一致（N6、N34 的小修见下） |

**本次修改**

1. **“前向增益恰为 1”过强**（§2.7 第 6 条、§2.8 代码对照、§6 练习 4、E4）：改为“精确算术下为 1”，并补上论文实测的偏差：图 7(a) 单层前向 1.0000–1.0005、图 7(b) 复合前向最高约 1.0066（SVG 复算），图 8 mHC $\prod_{1..60}$ 行和标签有 1.01。原因列入 §8 第 19 条。代码里行和为 $s/(s+10^{-6})$，也只是精确到约 $10^{-6}$。
2. **§2.7 第 8 条**：“SK 输出总在相对内部”改为指 $t\to\infty$ 的极限；截断到 20 次的输出严格说不在 $\mathcal B_n$ 上。
3. **§2.5 Mena 定理 1 漏了前提**：$M(X)=\lim_{\tau\to0^+}S(X/\tau)$ 需要“元素独立取自绝对连续分布”，并且只是几乎必然成立。已补上（`mena2018gumbel` v1 §2）。
4. **§2.5 Sinkhorn–Knopp 1967**：$D_1,D_2$ 唯一性原文是“当且仅当 $A$ 完全不可分解”，笔记原来只写了“充分”方向；$D_1AD_2$ 的条件原文是“既双随机又是迭代极限”。已按 msp.org 摘要改正。
5. **§0.3 版本漂移**：附录 A/B、表 1 改名、表 7 的笔误**从 v2 起**就存在，不是 v3 才有（对 v2 HTML 检索确认）；“必须引 v3”改为“v2 及以后”。
6. **C38**：原写“两处都说是 in-house large-scale training”，但 §4.3 原文只有 “in large-scale models”，已改为逐字对应。
7. **C40**：“三大架构升级之一”改为摘要原意“三项 architecture and optimization 升级之一”（第三项 Muon 是优化器）。
8. **C3 位置**：“§3 末”精确到 “§3.2 Discussions”。
9. **N6**：原写“四舍五入口径略不同”，但 51.85/41.11 = +26.125%，无论四舍五入还是截断都得 26.1%，所以表 9 的 26.0% 与表内 GB 数不自洽，已据实改写。
10. **N34**：mHC 八项平均 57.612 改为 57.6125（≈57.61）。
11. **§3.2 补充**：补上 config 的 `num_nextn_predict_layers`（V4-Flash 1、V4-Pro 1、V4.1-Flash 3），说明子层数不含这些块。

**仍未能核实**：原 §8 的 1–18 条保持不变，另加第 19 条（实测前向增益 > 1 的原因）。HC v2 与 v3 之间没有做全文 diff；E19 中“图 17 是训练曲线、该句 v1 没有”只核了 v1 与 v2 的 HTML 检索（v1 无 “Fig. 17”，v2 有），“图 17 的内容”没有逐图核对。

---

## 补充核实（写课时笔者的复算与推导，2026-09-24）

写 `lesson-16-mhc.md` 时没有引入新的外部来源；下面记录的是课文里用到、而上文没有逐字给出的**计算与推导**，全部可由
`deepseek/figures/gen/l16_*.py` 或下列表达式复现。凡属推导，课文里都标注了“笔者推导/笔者补”。

1. **随机近单位矩阵乘积的增长率（课文 §3，笔者推导）。** 对单位向量 $\mathbf v\in\R^n$，$\norm{(I+\sigma G)\mathbf v}^2=1+2\sigma z+\sigma^2w$，
   $z=\mathbf v^\top G\mathbf v\sim\mathcal N(0,1)$，$w=\norm{G\mathbf v}^2\sim\chi^2_n$；二阶展开得 $\E\log\norm{(I+\sigma G)\mathbf v}\approx\frac{n-2}{2}\sigma^2$（$n=4$ 时为 $\sigma^2$）。
   蒙特卡洛（numpy，种子 1，20 万次）：$\sigma=0.1,0.2,0.3$ 时 $\E\log$ 为 0.00999、0.03974、0.09020；60 次连乘后单位向量范数的中位数 1.86、12.0、225，对照 $e^{60\sigma^2}=1.82,11.0,221$。
2. **图 16.6 的模拟数字（`l16_sim_products.py`，种子 0，2000 次）。** 无约束 $H=I+\sigma G$ 在深度 60 的前向 Amax 中位数：$\sigma=0.1$ 为 2.83，0.2 为 16.39，0.3 为 370.15（第 90 百分位 5118）。
   行随机 rowsoftmax$(5I+2G)$：前向恒为 1，反向中位 1.766、p90 2.426、最大 3.768。双随机 SK$_{20}(\exp(5I+2G))$：前向最大 1.000000000000，反向中位 1.0223、p90 1.0419、最大 1.0860。
   与 N20/N21 的数字不同（种子与参数设置不同），课文引用的是本脚本的输出。
3. **谱范数 ≤ 1 的另一种证明（课文 (16.18)）。** 加权柯西–施瓦茨：$\norm{A\mathbf x}^2=\sum_i(\sum_ja_{ij}x_j)^2\le\sum_i(\sum_ja_{ij})(\sum_ja_{ij}x_j^2)=\sum_jx_j^2\sum_ia_{ij}=\norm{\mathbf x}^2$，依次用到非负、行和、列和；与 §2.7 第 2 条的 $\sqrt{\norm{A}_1\norm{A}_\infty}$ 证明等价。
4. **SK 保持交叉比（课文 §5）。** 行列缩放下 $\frac{m_{ij}m_{kl}}{m_{il}m_{kj}}$ 不变。N23 的算例：$A$ 的 $\frac{a_{11}a_{22}}{a_{12}a_{21}}=16$，极限的 $(0.58899/0.14725)^2=16.0$。
5. **熵正则问题的拉格朗日条件（课文 §5，笔者推导）。** $\partial\mathcal L/\partial P_{ij}=X_{ij}-\log P_{ij}-1-u_i-v_j=0\Rightarrow P=D_1\exp(X)D_2$；与 Sinkhorn 定理的唯一性合起来即 Mena 定理 1 在 $\tau=1$ 的形式。
6. **课文自拟的小算例（全部为笔者选取的数，非论文数字）。**
   - $n=2,d=3$ 的 HC 子层：$X_l$ 行 $(1,0,2),(3,1,0)$，$H^{\rm pre}=(0.5,0.5)$，$H^{\rm res}=\begin{pmatrix}0.9&0.1\\0.2&0.8\end{pmatrix}$，$\mathcal F$ 输出 $(0.1,-0.2,0.3)$ → 新流 $(1.3,-0.1,2.1),(2.7,0.6,0.7)$；混合后流之和 $(3.8,0.9,2.2)$ ≠ $(4,1,2)$。
   - 双随机 $A=\begin{pmatrix}0.7&0.2&0.1\\0.2&0.5&0.3\\0.1&0.3&0.6\end{pmatrix}$：奇异值 1、0.5732、0.2268；$\mathbf x=(6,3,0)$ → $(4.8,2.7,1.5)$ → $(4.05,2.76,2.19)$，偏差范数 4.243 → 2.362 → 1.348；$A^{10}$ 的元素在 0.3316–0.3357。
   - 练习 7：$A=\begin{pmatrix}2&1\\1&1\end{pmatrix}$，$M^{(1)}=\begin{pmatrix}4/7&3/7\\2/5&3/5\end{pmatrix}$，列和 $34/35,36/35$；极限 $a=2-\sqrt2=0.5858$（交叉比 2）。
   - 练习 6：图 8 的 $H^{\rm res}_{60}$ 按印出元素算行和 $-21.63,-20.23,22.50,-21.59$（标签 $-21.64,-20.22,22.50,-21.59$，差在两位小数舍入），列和 $-11.97,-6.86,-10.23,-11.89$；第 1 列绝对值之和 24.13。
7. **图 16.7（`l16_birkhoff.py`）的贪心 Birkhoff 分解**与 N24 完全相同：0.47247·I + 0.14725·C + 0.14725·C² + 0.11652·P(0,2,1) + 0.11652·P(2,1,0)。
8. **图 16.13 中 mHC 各矩阵的反向增益**直接取图 8 印出的列和标签（1.04、1.04、1.01、1.11、1.50、1.11）；由印出元素重算会因舍入差 0.01（如 $\prod_{1..30}$ 为 1.12）。

### 补充核实（教学复审，2026-09-24）

9. **SK 收敛速率（新来源 `knight2008`，解决 §8 第 9 条）。** Philip A. Knight, “The Sinkhorn-Knopp Algorithm: Convergence and
   Applications”, SIAM J. Matrix Anal. Appl. 30(1):261–275 (2008), doi:10.1137/060659624（书目信息据 strathprints.strath.ac.uk/19685；
   正文读的是同页的作者预印本 PDF）。原文：
   - 定理 4.5：“If A is fully indecomposable then the SK algorithm will converge linearly to vectors r∗ and c∗ such that
     D(r∗)AD(c∗) = P where P is doubly stochastic. Furthermore, there exists K ∈ Z such that if k ≥ K,
     ‖[r_{k+1}; c_{k+1}] − [r∗; c∗]‖ ≤ σ2² ‖[r_k; c_k] − [r∗; c∗]‖, where σ2 is the second singular value of P.”
   - §6：“the rate of convergence is asymptotically equal to the square of the second singular value of P. Generally, we
     have found that this asymptotic convergence rate is approached fairly quickly.”
   - §2 的定义：“A matrix is fully indecomposable if it is impossible to find permutation matrices P and Q such that
     PAQ = [A1 0; A2 A3] with A1 square.”

   §8 第 9 条的疑点（N23 的 0.0436 ≠ $\sigma_2^2=0.195$）的解释（笔者推导 + numpy 核对）：N23 的 $A$ 在“同时对调行 1、2 与列 1、2”下
   不变，SK 保持这个对称，误差不含 $\sigma_2=0.44174$ 的奇异向量 $(1,-1,0)/\sqrt2$ 方向，起作用的是 $\sigma_3^2=0.20871^2=0.043561$；
   实测相邻误差比 0.043561（t = 4..9）。无对称时比值等于 $\sigma_2^2$：numpy `default_rng(0)`，logit $=s\cdot$ 标准正态 $4\times4$，
   依次抽 $s=1,2,4$ 各 3 个，$s=2$ 的三个样本 $\sigma_2^2$ 为 0.8340、0.4580、0.5970，第 15–25 轮误差比为 0.8341、0.4580、0.5970。
10. **Knight 定理的另外三个算例（课文 §5、练习 7，笔者计算）。**
   - $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ 不是完全不可分解的（对调两行、两列得 $\begin{pmatrix}1&0\\1&1\end{pmatrix}$），定理不适用，与 N25 的 $O(1/t)$ 一致。
   - $\begin{pmatrix}1&1\\e^{-k}&1\end{pmatrix}$：极限 $\begin{pmatrix}1-\delta&\delta\\\delta&1-\delta\end{pmatrix}$，$\delta=1/(1+e^{k/2})$（交叉比 $e^k$）；$k=10$：$\delta=0.006693$，
     $\sigma_2^2=(1-2\delta)^2=0.97341$，数值迭代第 150–250 轮相邻误差比 0.9734；第 100 轮误差 0.0018，第 200 轮 1.2e-4。
   - 练习 7 的 $\begin{pmatrix}2&1\\1&1\end{pmatrix}$：$M^{(2)}=\begin{pmatrix}24/41&17/41\\12/29&17/29\end{pmatrix}$（分数精确计算），列和 $1188/1189,1190/1189$；
     第 1、2 轮误差 $1/35$、$1/1189$，比值 0.029437；$\sigma_2^2=(3-2\sqrt2)^2=17-12\sqrt2=0.029437$。
11. **对角 logit 算例（课文 §5，笔者计算）。** logit $=cI$（$4\times4$）一次归一化即双随机：对角 $e^c/(e^c+3)$，特征值 1 与三重 $(e^c-1)/(e^c+3)$。
   $c=2$：对角 0.7112、$\sigma_2=0.6150$、$\sigma_2^2=0.3782$；$c=5$：0.9802、0.9736、0.9479（$0.9479^{20}=0.343$）；$c=6$：0.9926、0.9902、0.9804。
   加扰动的模拟：numpy `default_rng(0)`，依次 $c=0,2,5$ 各 200 个 $cI+G$（$G$ 标准正态），20 轮 SK 后最大列和误差中位数
   $1.70\times10^{-13}$、$3.72\times10^{-7}$、$9.79\times10^{-3}$（p90 分别 $4.8\times10^{-9}$、$1.3\times10^{-4}$、$1.6\times10^{-2}$），极限 $\sigma_2^2$ 中位数 0.244、0.524、0.958。
12. **算术强度（课文 §6，笔者估算）。** V4-Flash 每 token 每子层 mHC 运算约 984,320 次（课文 §5 算例），未融合访存 $(34d+48)\times2=278{,}624$ 字节
   → 3.53 FLOP/字节；$20d$ 口径 163,840 字节 → 6.01。H100 SXM 脊点 295 FLOP/字节取自第 4 课（其研究笔记的出处）。
13. **总账表 mHC 一列的 18d（课文 §7，笔者推算）。** 按 V4.1 §2.4.1 对三个串行核的描述：更新残差读 $(n+1)d$ 写 $nd$，算系数读 $nd$，输入混合读 $nd$ 写 $d$，
   合计 $(4n+2)d=18d$；加 pre-norm 一读一写 $2d$ 即 V4.1 论文的 $(4n+4)d$。
14. **“单层增益都是 1，乘积却放大”的 2×2 算例（课文 §3、练习 5，笔者构造，numpy 核对）。** $A=\begin{pmatrix}1.1&-0.1\\-0.1&1.1\end{pmatrix}$（行和、列和都是 1，
   $\norm{A}_2=1.2$），$B=\operatorname{diag}(1,0.8)$（行和、列和 1 与 0.8）：两者 Amax 增益都是 1。$AB=\begin{pmatrix}1.1&-0.08\\-0.1&0.88\end{pmatrix}$，行和 1.02、0.78，
   特征值 1.13177、0.84823。全 1 输入：$(1,1)\to(1,0.8)\to(1.02,0.78)\to(1.02,0.624)\to(1.0596,0.5844)$。$(AB)^{30}$（60 层）前向增益 24.84，反向 21.95，谱范数 41.10；
   $(AB)^{10}$：2.16 / 1.94。
15. **(16.21) 之后“离 $J_n$ 多远”的界（课文 §4，笔者推导）。** $P-J_n=P\Pi=(H_L\Pi)\cdots(H_1\Pi)$，$\norm{H_k\Pi}_2=\sigma_2(H_k)$（$H_k^\top H_k\mathbf 1=\mathbf 1$，
   其余奇异向量与 $\mathbf 1$ 正交），故 $\norm{H_L\cdots H_1-J_n}_2\le\prod_k\sigma_2(H_k)$。核对：§4 开头的 $3\times3$ 对称双随机矩阵特征值 $1,(4\pm\sqrt3)/10=0.57321,0.22679$；
   $\norm{A^{k}-J_3}_2=0.57321^k$（对称，取等号），$k=10$ 时 0.003829；$A^{10}$ 元素在 0.331590 与 0.335715 之间（最大偏差 0.00238）。

### 补充核实（数学复审，2026-09-24）

16. **全课数字复算。** 所有生成脚本重跑后 SVG 逐字节不变；§1–§7 与练习中的数字用 Python 逐个复算，除下面几条外全部无误。
17. **图 7 单层反向增益的区间（课文 §7 第 1 条细节）。** 数字化列表（`l16_mhc_gain.py`）中第 20–60 个子层的单层反向增益为 1.006–1.051（全部 60 层为 1.006–1.061），
   连乘 3.112；原文写“1.01–1.06”指的是全部 60 层，已改为第 20–60 层的区间。
18. **“损失对 mHC 的 $H^{\rm res}_1$ 梯度为 0”的准确含义（课文 §7，笔者推导）。** $X_1=\mathbf 1\mathbf x^\top$ 时 $\partial\mathcal L/\partial H^{\rm res}_1=G_2X_1^\top=(G_2\mathbf x)\mathbf 1^\top$，一般不为 0；
   但它与任何 $\Delta\mathbf 1=\mathbf 0$ 的改动内积为 0，而 SK 以行归一化收尾、输出行和恒为 1，所以对 logit（及 $W^{\rm res}_1,S^{\rm res}_1,\alpha^{\rm res}_1$）的梯度恰为 0。
19. **Amax 与单位向量增长之比约 1.5 的来源（课文 §3，笔者模拟）。** numpy `default_rng(0)`，20 万个 $\R^4$ 均匀随机单位向量：$\max_i|u_i|$ 中位数 0.773、均值 0.779；
   $2\times0.77\approx1.55$，与 `l16_sim_products.py` 打印的 Amax 中位数 / $\norm{P_{60}\mathbf v}$ 中位数 = 1.55、1.50、1.60 一致。
20. **(16.5) 的前提。** “所有 $\lambda_i>1$ 则指数增长”需要 $\lambda_i\ge\lambda>1$ 一致成立；反例 $\prod_{i\ge1}(1+1/i^2)=\sinh\pi/\pi\approx3.68$ 有限。
21. **非负矩阵的增益乘法不等式（练习 5 答案）。** 非负矩阵的最大行和 = $\norm{\cdot}_\infty$、最大列和 = $\norm{\cdot}_1$，都是诱导范数，满足乘法不等式，所以“单层增益相乘”是复合增益的上界；
   有正有负时不成立（§3 的 2×2 算例：单层都是 1，60 层后 24.84）。

### 补充核实（事实复审，2026-09-25）

逐条核对课文中的数字、引文与归属；用 `arxiv.py grep` 回到原文，所有生成脚本重跑后 SVG 逐字节不变（除下面第 27 条改动的图注）。

22. **Xiong 2020 与 warm-up（课文 §1）。** `xiong2020preln` v2 摘要：“the expected gradients of the parameters near the output layer are large. Therefore, using a large
    learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful for avoiding this problem.” §1 同意。课文“所以必须用很小的学习率热身”有据。
23. **DualPipe 的重叠（课文 §6 与 `dsv3` 卡片）。** `dsv3` v2 §3.2.1：“The key idea of DualPipe is to overlap the computation and communication within a pair of individual forward and
    backward chunks”，每个 chunk 分为 attention、all-to-all dispatch、MLP、all-to-all combine，另有 PP communication 组件；“we can ensure that both all-to-all and PP communication
    can be fully hidden during execution.” 卡片的 finding 因此有据，`a` 字段由“只核对了元数据”改为“只核对了 §3.2.1 关于 DualPipe 重叠的几句”。
24. **Sinkhorn 1964 的唯一性（课文 §5 定理框、`sinkhorn1964` 卡片）。** Project Euclid 页面只显示元数据（35(2):876–879，1964 年 6 月），不显示摘要。唯一性据 `mena2018gumbel` v1
    附录 A 开头转述的 Sinkhorn 定理：“there exists two diagonal matrices D1, D2, with positive diagonals, so that P = D1 A D2 is a doubly stochastic matrix. These D1, D2 are unique up to
    a scalar factor.”（出处注为 Sinkhorn 1964、Sinkhorn & Knopp 1967、Knight 2008）。$D_1,D_2$ 在相差一个数倍的意义下唯一，$D_1AD_2$ 就唯一。原文正文仍未读。
25. **HC 基线也用 α = 0.01、n = 4（课文 §7）。** `xie2026mhc` v2 表 5 的行名是 “mHC/HC Expansion Rate n” 与 “mHC/HC Gating Factor Init α”；§5.1：“the expansion rate n for both HC and mHC is set to 4.”
26. **V4 与 V4.1 的原句复核。** `dsv4` v1：“We maintain the AdamW … optimizer for the embedding module, the prediction head module, the static biases and gating factors of mHC modules,
    and the weights of all RMSNorm modules. All other modules are updated with Muon.”；§3.3：“mHC involves a matrix multiplication with an output dimension of only 24. For very small
    batch sizes, we are compelled to use the split-k … algorithm, whose naive implementation will cause non-determinism.”；§3.4.2 “6.7% of the overlapped 1F1B pipeline stage”。
    V4 把 $S^{\rm post}_l$ 写成 $n_{hc}\times1$（与 $C_l$ 同形），课文用 mHC 的 $1\times n$ 写法，二者只差转置。
27. **V4.1 的“读、写各 $(n+1)d$”是原文，不是笔者解读。** `dsv41` v1 §2.4.1：“The residual is thereby read once and written once, attaining the (n+1)d reads and (n+1)d writes of
    the ideal map and halving the activation memory traffic of our original implementation.” 图 16.14 的图注与 `l16_single_pass.py` 中的对应文字已改为注明出处。
28. **对上面第 1 条的更正（课文 §3 引用的是生成脚本的数）。** 课文“单位向量经过 60 个矩阵后的长度中位数 1.83、11.0、231”来自 `l16_sim_products.py`（`default_rng(1)`，2 万次），
    脚本输出 1.83、10.96、231.45，Amax 中位数 / 长度中位数 = 1.55、1.50、1.60；第 1 条记的 1.86、12.0、225（种子 1、20 万次）是另一次运行，课文不引用它。
29. **本次对课文的事实性改动。** (a) 封面问题原写“参数与 FLOPs 都不到全模型的千分之一”，但 FLOPs 占比是 0.33%（N15），改为“参数不到千分之一、FLOPs 不到百分之一”。
    (b) 导语与“科技树位置”原写 HC “第一次”动了残差捷径 / “第一次”把残差流变成可学习对象：本笔记与所读文献都没有确立这种优先权，删去“第一次”，导语限定为“主流模型里”。
    (c) “DeepSeek 的每一代模型都改了注意力……”改为“先后改了”（并非每一代都改了注意力）。(d) §4 误解框原写“Sinkhorn 投影的结果恰好都落在（相对）内部”，按核查记录第 2 条改为：
    极限在相对内部，截断 20 轮的输出列和只近似为 1；练习 2(b) 同改。(e) `sinkhorn1967` 卡片的 matters 原说它“解释了”反例为什么只有 $O(1/t)$，该文只给出收敛与 $D_1AD_2$ 形式的条件，
    $O(1/t)$ 是课文自己的计算，已改写；`knight2008` 卡片把“至多乘 $\sigma_2^2$”（定理 4.5）与“渐近速率等于它”（§6）分开写。
