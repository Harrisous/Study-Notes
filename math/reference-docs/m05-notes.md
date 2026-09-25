# M05 补充核实：信息论课自己核对的事实

> 本文件只记录 `math-refs-doc.md` 以外、写 M05 时自己核对过的事实（来源 + 位置 + 不超过 15 词的原文短语）。
> 共用的 M05 条目（Gibbs、Jensen、k1/k2/k3、N36–N41、E8–E10）仍以 `math-refs-doc.md` 为准。核对日期 2026-09-24。
> 读取方式：WebFetch 取回 PDF 后在 session scratch 目录用 pypdf 抽文本、grep 原句；书目信息用 Crossref API。

## S1 Shannon 1948（`shannon1948`）

读的是 Harvard 镜像的重排版 PDF（people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf），
首页印 “Reprinted with corrections from The Bell System Technical Journal, Vol. 27, pp. 379–423, 623–656, July, October, 1948”。
下面的页码是这份重排版的页码。

- **bit 这个词。** 引言（重排版 p. 1）：以 2 为底时单位 “may be called binary digits, or more briefly bits”，
  “a word suggested by J. W. Tukey”。同段：以 10 为底叫 decimal digits。
- **熵的三条要求与定理 2。** §6 “Choice, Uncertainty and Entropy”（p. 10）列出对 $H(p_1,\dots,p_n)$ 的三条要求：
  (1) 对 $p_i$ 连续；(2) 各 $p_i=1/n$ 时是 $n$ 的单调增函数；(3) 一个选择拆成两次相继的选择时，原来的 $H$ 应是各次 $H$ 的加权和。
  图 6 的例子：$H(\tfrac12,\tfrac13,\tfrac16)=H(\tfrac12,\tfrac12)+\tfrac12H(\tfrac23,\tfrac13)$，系数 $\tfrac12$ 是因为第二次选择只在一半的时候发生。
  定理 2（p. 11）：满足这三条的 $H$ 只能是 $H=-K\sum_{i=1}^np_i\log p_i$，$K$ 为正常数；证明在附录 2（p. 28 起）。
  Shannon 自己说定理与假设“in no way necessary for the present theory”，只为给后面的定义增加可信度。
- **熵的性质。** p. 11：$H=0$ 当且仅当只有一个 $p_i$ 为 1、其余为 0；给定 $n$ 时，所有 $p_i$ 相等（$1/n$）时 $H$ 最大，等于 $\log n$。
- **图 7 = 二元熵曲线。** p. 11：“The entropy in the case of two possibilities with probabilities p and q = 1 − p” 画在图 7，
  纵轴标 “BITS”，横轴 $p$ 从 0 到 1.0。本课图 M5.2 据此重绘（曲线由脚本计算）。
- **A、B、C、D 的例子。** §10 “Discussion and Examples”（pp. 17–18）：独立产生 A、B、C、D，概率 $\tfrac12,\tfrac14,\tfrac18,\tfrac18$，
  $H=\tfrac74$ bits/符号；编码 A 0、B 10、C 110、D 111，平均 $\tfrac74$ 个二进制位，恰好达到极限。本课图 M5.3 的 (a) 就是这个码的“问题树”写法。
- **定理 9（无噪声信道基本定理）。** §9（p. 16）：熵为 $H$（bits/符号）的信源、容量为 $C$（bits/秒）的信道，可以编码到每秒 $C/H-\epsilon$ 个符号，
  但不可能超过 $C/H$。本课只用它的推论形式：平均每个符号至少要 $H$ 个二进制位（笔者的转述）。
- **Shannon 的 “relative entropy” 不是 KL。** §7（p. 14）：“The ratio of the entropy of a source to the maximum value it could have
  while still restricted to the same symbols will be called its relative entropy”；1 减去它叫冗余度（redundancy）。
  即 Shannon 的 relative entropy $=H/H_{\max}$；今天“相对熵”通常指 KL 散度，两者不是一回事（本课 §4 的 `err` 框）。

## S2 Minka 2005（`minka2005`）

Thomas Minka, “Divergence measures and message passing”, Microsoft Research Technical Report MSR-TR-2005-173, December 7, 2005（首页原样）。
读的是 microsoft.com 上的 PDF（wp-content/uploads/2016/02/tr-2005-173.pdf），17 页。

- §2 “Divergence measures”（p. 2）：要用简单族 $\mathcal F$（如高斯）里的 $q$ 近似复杂分布 $p$，最小化 $D(p\,\|\,q)$；式 (1) 是带修正项、可用于未归一化分布的 KL；
  注明这个散度对 $p,q$ 不对称。式 (2) 的 α-散度满足 $\lim_{\alpha\to0}D_\alpha(p\|q)=\mathrm{KL}(q\|p)$（式 (4)）、$\lim_{\alpha\to1}D_\alpha(p\|q)=\mathrm{KL}(p\|q)$（式 (6)）。
- 图 1（p. 3）标题：“The Gaussian q which minimizes α-divergence to p (a mixture of two Gaussians), for varying α”；
  α 趋于 −∞ 偏好只匹配一个峰，α 趋于 ∞ 偏好覆盖整个分布。图 2：质量、均值、标准差在 α = 1 时与真值一致（即 KL(p‖q) 做矩匹配）。
  **注意**：Minka 的 $p$ 是“一高一窄 + 一矮一宽”两个高斯；本课图 M5.7 用的是两个等高等宽的高斯，数值全部是本课脚本自己算的，只借用了“单个高斯拟合双峰混合”这一设定。
- p. 3：α ≤ 0 的散度 “are zero-forcing because p(x) = 0 forces q(x) = 0”；负 α 越大，越偏向只表示质量最大的那个峰（“not the mode which is highest”）；
  zero-forcing 倾向于低估 $p$ 的方差。
- p. 4：沿用 Frey et al. (2000) 的术语，α ≥ 1 的散度叫 **inclusive**（α < 1 叫 exclusive），要求 $p>0$ 处 $q>0$；
  “If two identical Gaussians are separated enough, an exclusive divergence prefers to represent only one of them, while an inclusive divergence prefers to stretch across both.”
  （这句 25 词，只在这里存档，正文转述。）
- 摘要：平均场方法与信念传播的区别在于最小化的损失不同，“‘exclusive’ versus ‘inclusive’ Kullback-Leibler divergence”。
- 本课采用的中文说法：KL(p‖q)（前向，inclusive）= “质量覆盖”；KL(q‖p)（反向，exclusive / zero-forcing）= “挑峰”。
  “mode-covering / mode-seeking”这对英文说法在 Minka 文中只出现了 “mode-seeking property”（p. 3，指 α ≪ 0），没有 “mode-covering” 一词。

## S3 书目信息（Crossref，未读正文）

- **Kullback & Leibler 1951。** “On Information and Sufficiency”, S. Kullback, R. A. Leibler, *The Annals of Mathematical Statistics* 22(1):79–86,
  1951 年 3 月，doi:10.1214/aoms/1177729694（Crossref API）。正文未读（Project Euclid 对脚本拦截，未绕过）。本课只用它作“KL 这个名字的来历”，不引其中的任何论断。
- **Jelinek, Mercer, Bahl, Baker 1977。** “Perplexity—a measure of the difficulty of speech recognition tasks”, *The Journal of the Acoustical Society of America* 62(S1):S63，
  1977-12-01，doi:10.1121/1.2016299（Crossref API）。这是一篇会议摘要，正文未读。本课只用它作“困惑度这个名字的来历”。

## S4 未能核实（写课时回避）

- MacKay 书的网站（inference.org.uk）对 WebFetch 返回 403，未能核对 §2.4 里“Shannon information content $h(x)$”的式号。本课的信息量与熵的定义改引 Shannon 1948，
  MacKay 只引研究笔记已核对的 §2.6（2.45）(2.46) 与 §2.7（2.47）(2.48)。
- Bishop《Pattern Recognition and Machine Learning》§10.1.2 的前向/反向 KL 图（PDF 超过 WebFetch 的 10 MB 上限）未能读到，本课不引。
- Schulman 博客这次两次连接被重置，未能重读；k1/k2/k3 的内容全部以研究笔记 C21（2026-09-24 已核对）与第 8 课为准。

## S5 交叉引用（来自 DeepSeek 课的研究笔记，本次未重核）

- GPT-1 在 BooksCorpus 上的 token 级困惑度 18.4（lesson-02-doc D4）；GPT-2 按规范单位（字符、字节或词）报告（同条）。
- GPT-2 词表 50,257（M01 的 `gpt2-config` 卡片）；DeepSeekMath 词表 102,400（lesson-08-doc 表，“vocabulary size of 100K”取整前的配置值）。
- DeepSeekMath 式 (4) 的 KL 估计器就是 k3，论文称 “unbiased estimator (Schulman, 2020)”、“guaranteed to be positive”；KL 系数 0.04（lesson-08-doc C18、C26）。
- DeepSeek-V3.2 §3.1 “Unbiased KL Estimate”：用 $\pi_\theta/\pi_{old}$ 的重要性比修正 k3（lesson-15-doc C22）；数学领域用很弱的 KL 甚至不用效果更好（C25）。
- Tang & Munos 2025 式 (3)：把 k3 当损失求导，期望梯度是 $\nabla\mathrm{KL}(\pi_{ref},\pi)$（lesson-15-doc C24）。
- DSA 预热阶段的 indexer 损失 $\sum_t\KL(p_{t,:}\,\|\,\softmax(I_{t,:}))$（DeepSeek-V3.2-Exp 报告式 (3)，第 14 课 (14.20)）。
- InstructGPT 在每个 token 的奖励上加相对 SFT 模型的 KL 惩罚（第 3 课 (3.7)(3.8)，lesson-03-doc）。

## S6 本课自己算的数字（脚本在 `math/figures/gen/m05_*.py`；其余在 session scratch `m05-information-theory/nums.py`、`kdisc.py`）

| 量 | 表达式 | 值 |
|---|---|---|
| 信息量 | $-\log_2 p$ | 1/2 → 1；1/6 → 2.585；0.1 → 3.322；0.01 → 6.644 bits = 4.605 nats；1/50257 → 15.617 bits = 10.825 nats；1/102400 → 16.644 bits = 11.537 nats |
| 单位换算 | $1/\ln2$ | 1 nat = 1.4427 bits；1 bit = 0.6931 nats |
| Shannon 图 6 | $H(\tfrac12,\tfrac13,\tfrac16)$ vs $1+\tfrac12H(\tfrac23,\tfrac13)$ | 1.4591 = 1.4591（$H(\tfrac23,\tfrac13)=0.9183$） |
| 二元熵 | $H_b(p)$ bits / nats | 0.5 → 1 / 0.6931；0.75 → 0.8113 / 0.5623；0.9 → 0.4690 / 0.3251；0.99 → 0.0808 / 0.0560；0.999 → 0.0114；$H_b=0.5$ 在 $p=0.110$ 与 $0.890$ |
| A/B/C/D | $H(p)$，$H(p,u)$，$H(p,q_{rev})$ | 1.75；2（KL 0.25）；2.625（KL 0.875）；$2^{1.75}=3.364$ |
| softmax (2,1,0) | 熵；温度 0.5 / 2 | 0.8324 nats = 1.2009 bits；0.4411 / 1.0202 nats |
| 困惑度 | 9 个 0.5 + 1 个 0.001 | 平均 1.3146 nats，PPL 3.723；全 0.5 时 2.000；算术平均概率 0.4501，倒数 2.222；坏 token 占总意外 53% |
| 损失差与 PPL 比 | $e^{\Delta}$ | 0.01 → 1.0101；0.1 → 1.1052；0.5 → 1.6487 |
| 分词器 | 300 nats / 100 或 130 token | 3.0 / 2.308 nats/token → PPL 20.09 / 10.05；450 字节 → 0.962 bits/字节 |
| 三角不等式反例 | $a=(0.9,0.1)$、$b=(0.7,0.3)$、$c=(0.5,0.5)$ | KL(a‖c) = 0.3681 > KL(a‖b) + KL(b‖c) = 0.1163 + 0.0823 = 0.1986 |
| 前向/反向拟合（图 M5.7） | $p=\tfrac12N(-2,0.6^2)+\tfrac12N(2,0.6^2)$ | 前向解 $N(0,2.088^2)$：KL(p‖q) 0.5552、KL(q‖p) 1.2732；反向解 $N(-1.997,0.605^2)$（2026-09 复核：旧值 −1.998 是 0.0025 网格点 −1.9975 的舍入，黄金分割细化后 μ = −1.9971）：KL(q‖p) 0.6918、KL(p‖q) 10.23；反向剖面在 μ = 0 另有局部极小 1.1384（σ = 1.79），μ = ±1.10 处局部极大 1.2433 |
| 三元例的估计器 | $q=(0.5,0.3,0.2)$，$p=(0.4,0.4,0.2)$ | $w=(0.8,1.333,1)$；KL(q‖p) = 0.025267；标准差 k1 0.2216、k2 0.01433、k3 0.01596；E[k2] = 0.024862；前向估计 $w\ln w-(w-1)$ 的期望 0.025815 = KL(p‖q) |
| 高斯闭式 | $q=N(0,1)$，$p=N(\mu,1)$ | 相对标准差 k1/k2/k3：μ = 0.1 → 20 / 1.418 / 1.417；μ = 1 → 2 / 1.732 / 1.695；μ = 2 → 1 / 2.449 / 3.521 |
| 1000 词表例（图 M5.9） | $q=\softmax(z)$，$p=\softmax(z+sg)$，seed 0 | $H(q)=5.236$ nats；峰度 $\kappa=3.328$，$\sqrt{\kappa-1}=1.526$；KL = 0.0029 时相对标准差 k1 26.2、k2 1.50、k3 1.52，k2 偏差 −0.6%；KL = 0.527 时 1.86 / 1.55 / 1.48，k2 偏差 +17.9% |
| 样本数 | $(\text{相对标准差}/0.1)^2$ | KL = 0.005 的高斯例：k1 要 40,000 个样本才把相对误差压到 10%，k3 约 200 个 |
| 序列 KL 玩具例（图 M5.8） | 5 词词表，$q$ 逐个采样“你 很 好 。”，分布写在 `m05_seq_kl.py` | 逐位置精确 KL(q‖p) 0.165 / 0.065 / 0.119 / 0.062，和 0.411；采样词的 $w$ = 0.667 / 1.30 / 1.00 / 0.824；k1 之和 0.337（第 2 步 −0.262），k3 之和 0.127（第 3 步为 0）；每列 $\E_q[k_3]$ = 该列精确 KL |
| 最优控制变量系数（练习 12，本课推导） | $\lambda^*=-\Cov_q(k_1,w)/\Var_q(w)$ | 三元例 $\lambda^*=0.958$，最小标准差 0.0126（k3 为 0.0160）；高斯例 $\lambda^*=\mu^2/(e^{\mu^2}-1)$：μ = 0.1 → 0.995（相对标准差 1.413，k3 1.417），μ = 1 → 0.582（1.293，k3 1.695）；$\lambda=0.58$、$w=1.5$ 时 $-\ln w+\lambda(w-1)=-0.115$ |
| 反向 KL 的吸引域（2026-09 数学复核） | 对 $(\mu,\ln\sigma)$ 做梯度下降（有限差分），双峰例 | 从 $\mu_0=-0.05$ 出发（$\sigma_0=0.3\ldots2$）都停在 $\mathcal N(0,1.79^2)$，KL 1.1384；$(\mu_0,\sigma_0)=(-0.5,0.6)$ → 左峰 0.6918，$(-0.5,1.0)$ → 中间 1.1384；$(-1.3,\sigma_0\le2)$ → 左峰。旧稿“μ 略小于 0 会滑到左峰”错误，已改 |
| k2 与 k3 的相对标准差交叉点（2026-09 数学复核） | 高斯闭式；1000 词表例 | 高斯例 $\mu=1.2047$（KL 0.7256）以后 k3 > k2；词表例 KL ≈ 0.379（$s=0.779$，第二次数学复核用 brentq 求根；旧值 0.39 偏大）以后 k3 < k2（KL 2.46 时 1.066 对 2.350）；0.38–0.73 之间两例都是 k3 更稳。词表例 k2 相对偏差最低 −0.93%（KL ≈ 0.016–0.02） |
| 困惑度类比的数（§3 类比框） | 练习 4 的 logits (3, 1, 0, −1) | 模型自身熵 0.595 nats → $e^{0.595}=1.81$；真 token 交叉熵 2.185 nats → $e^{2.185}=8.89$ |
| 练习 8 的来源拆分（第二次数学复核） | $p=0.7N(-2,0.6^2)+0.3N(2,0.6^2)$，取 $q$ 恰为一个分量 | $\KL(\mathcal N_L\|p)=0.35589$（$-\ln0.7=0.35667$，数值最优 0.35586）；$\KL(\mathcal N_R\|p)=1.20204$（$-\ln0.3=1.20397$，数值最优 1.20188）：估计与数值解之差几乎全来自另一个峰的尾巴 |
| softmax (2,1,0) 熵的三项（第二次数学复核） | $-p_i\ln p_i$ | 0.27116、0.34448、0.21676（旧稿四位小数 0.3446、0.2167 舍入有误，已改为五位） |
| 硬币的最大似然（§2 例题，2026-09 教学复核补） | $\hat p=(0.7,0.3)$，$\mathcal L(q)=-(0.7\ln q+0.3\ln(1-q))$ | $q=0.7$：0.6109 $=H(\hat p)$；$q=0.5$：0.6931（差 0.0823 = KL(p̂‖(0.5,0.5))，与练习 3 的 KL(b‖c) 相同）；$q=0.9$：0.7645（差 0.1537，与练习 3 的 KL(b‖a) 相同）；真硬币公平时 $q=0.7$ 的期望损失 0.7803，比 $\ln2$ 多 0.0872（= 练习 3 的 KL(c‖b)） |

## S7 补充核实（2026-09-25 事实与引文复核）

> 方法：Shannon 重排版 PDF 与 Minka PDF 用 curl 取回后 pypdf 抽文本 grep；Schulman 博客用 curl 取回原页（S4 里“两次连接被重置”这次没有复现）；
> arXiv 论文用 `arxiv.py grep/text`（页头确认版本）。所有正文数字与 12 道练习的答案用 scratch `facts-m05-information-theory/recheck.py` 从头重算，全部一致；
> 十个生成脚本重跑，SVG 与重跑前逐字节相同。

- **Shannon 自己算过 A–D 信源的 relative entropy。** 重排版 p. 18（§10 A、B、C、D 例子的末尾）：最大可能熵 $\log4=2$，“Hence the relative entropy is 7/8”。
  旧稿 §4 `err` 框把 $7/8$ 标成“笔者按定义算的”，不对，已改为引 Shannon。
- **定理 2 之后那句话的后半句。** p. 11：定理与假设 “in no way necessary for the present theory”，之后一句 “The real justification of these definitions, however, will reside in their implications”。
  正文 §1“定义真正的理由在于它推出的结果”有据。p. 11 同页还有性质 1（$H=0$ 当且仅当一个 $p_i$ 为 1）与性质 2（$p_i$ 相等时 $H$ 最大且等于 $\log n$）以及图 7。
- **Minka 的“质量最大的峰”说的是哪一段 α。** p. 3：$\alpha\le0$ 的散度 zero-forcing，“q will keep the areas of largest total mass”（对所有 $\alpha\le0$，含 $\alpha=0$ 即 KL(q‖p)）；
  “represents only one mode, the one with largest mass”（不是最高的峰）说的是 $\alpha$ 为很大的负数时的最优近似，且是在他那个“一高而窄、一矮而宽”的例子里。
  p. 10 §5 把 KL(q‖p) 叫 exclusive KL（平均场），p. 11 §6 把 KL(p‖q) 叫 inclusive KL（BP、EP）。图 2 标题：质量、均值、标准差在 α = 1 时都与真值一致。
  Minka 的近似族是“single (scaled) Gaussian”（带尺度因子、不要求归一）。据此改了练习 8 (b) 的归属措辞与 `minka2005` 卡片的 finding（旧稿把 KL(q‖p) 说成 α 族的“一端”，并把 α ≪ 0 的结论挂在 zero-forcing 整类上）。
- **Schulman 博客（重读原页，页面 “Posted on 2020/03/07”）。** $r=p/q$，样本来自 $q$；k1 “negative for half of the samples”；k2 的期望是 $f(x)=\tfrac12(\log x)^2$ 的 f-散度，
  而博客对 f-散度的定义写的是“for a convex function f”（本课指出这个 $f$ 在 $x>e$ 处不凸，是本课自己的补充）；可微 $f$ 的 f-散度在 $q\approx p$ 时二阶上都像 KL（两者 $f''(1)=1$）。
  控制变量：使方差最小的 $\lambda$ “depends on p and q and is hard to calculate analytically”，于是取 $\lambda=1$ 使估计量恒为正，k3 是 $\log$ 与其切线的竖直距离（Bregman 散度）。
  $\mathrm{KL}[p,q]$ 的估计 $r\log r-(r-1)$。两张表（KL 0.005：bias/true 0 / 0.002 / 0，stdev/true 20 / 1.42 / 1.42；KL 0.5：0 / 0.25 / 0，2 / 1.73 / 1.7）与 C21 一致。
  练习 12 (e) 据此补引博客。
- **Tang & Munos（arXiv:2506.09477v1，2025-06-11）。** 作者 Yunhao Tang（“Work done while at Meta GenAI”）、Rémi Munos（Meta FAIR）。式 (3)：
  $\E[\widehat{\nabla\mathrm{KL}}_{\text{var-reduced}}]=\nabla\mathrm{KL}(\pi_{ref},\pi)\ne\nabla\mathrm{KL}(\pi,\pi_{ref})$；他们把 k3 叫 “variance reduced estimate”。
  **叫法冲突**：他们以 $\mathrm{KL}(\pi,\pi_{ref})$ 为基准，把 $\mathrm{KL}(\pi_{ref},\pi)$ 称为 “reverse KL”（§4.1 “sequence-level reverse KL”；结论 “ends up producing a reverse KL regularization”）；
  本课按“目标在前 = 前向”把同一个量叫前向。正文 §6 已加一句提醒。他们的 8B 实验里两种估计在低 KL 区行为相近，作者自己说这是推测（“We speculate”）。
- **DeepSeekMath（arXiv:2402.03300v3，2024-04-27）。** 作者 11 人与卡片一致（Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo）；式 (3) 是 GRPO 目标，式 (4) 是 KL 估计
  $\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1$，前一句 “unbiased estimator (Schulman, 2020)”，后一句 “guaranteed to be positive”。表 1 注 “vocabulary size of 100K”（102,400 是 config 值，见 lesson-08-doc）。
- **DeepSeek-V3.2（arXiv:2512.02556v1，2025-12-02）。** 题名 “DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models”；§3.1 “Scaling GRPO” 下的小标题 “Unbiased KL Estimate”：
  样本来自 $\pi_{old}$，用 $\pi_\theta/\pi_{old}$ 的重要性比 “correct the K3 estimator (Schulman, 2020)”；称修正后 “the gradient of this KL estimator becomes unbiased”；
  原 k3 在 $\pi_\theta\ll\pi_{ref}$ 时 “disproportionately large, unbounded weights”；数学等领域用很弱的 KL 甚至不用效果更好。
- **跨课链接的页码。** 第 15 课 p5 = “稳定化（一）：无偏 KL 估计”（原链接无页码，已补 `#p5`）；M06 p5 = “softmax 与交叉熵的梯度”（(M5.7) 的注与封面改链到 `#p5`）；
  M08 p2 = “IEEE 754 的位布局”（FP16 最小次正规数 $2^{-24}$ 在这一页，已补 `#p2`）；第 14 课 (14.19)：目标分布 = 各头注意力按头求和再 L1 归一化 = 头平均（lesson-14-doc 第 128 行），
  §5 的 DSA 说明补了这一句。第 3 课 §3 写明“RLHF 用 KL(πθ‖πref)……可以用策略自己的样本来估计”，§5“第一”的引用有据。
- **“为什么要单样本估计”的归属。** 第 8 课第 646 行把“参考模型要为每个位置保留整张分布”这一理由标为笔者解释、论文未讨论；本课 §6 同一段原来没有标，已补标注。
