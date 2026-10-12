# M09 写课补充笔记：策略梯度到 PPO

> 本文件只记录**写 M09 时新增**的核实与计算；共享研究笔记是 `math/reference-docs/math-refs-doc.md`（§1.4、§2.4–§2.5、
> §3.6、§4.5、§6 M09、§7 E11–E15 与文末核查记录），本文件不改动它。PPO、GRPO、V3.2 的细节按【交叉引用】取自
> `deepseek/reference-docs/lesson-03-doc.md`、`lesson-08-doc.md`、`lesson-15-doc.md`，本次未重核原文。
> 写于 2026-09-24。标记同共享笔记：【笔者推导】= 本课自己的推导或数值实验，不是任何来源的原话。

## 1. 补充核实（一手来源）

### 1.1 TRPO：性能差分恒等式与局部近似（`schulman2015trpo`）

- 来源：John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel,
  “Trust Region Policy Optimization”，arXiv:1502.05477 **v5**（`arxiv.py meta` 页头确认 v5；comment 字段 “16 pages, ICML 2015”）。
  读取方式：`arxiv.py grep 1502.05477 ...`（LaTeXML 全文），2026-09-24。
- **式 (1)（§2）**：$\eta(\tilde\pi)=\eta(\pi)+\E_{s_0,a_0,\dots\sim\tilde\pi}\big[\sum_{t=0}^\infty\gamma^tA_\pi(s_t,a_t)\big]$。
  原文引出这条恒等式时写明出处：“see Kakade & Langford (2002) or Appendix A for proof”。
  其中 $\eta$ 是期望折扣回报，$\E_{\dots\sim\tilde\pi}$ 表示动作按 $\tilde\pi$ 采样。
- **$\rho_\pi$ 的定义（§2）**：“(unnormalized) discounted visitation frequencies”，$\rho_\pi(s)=P(s_0=s)+\gamma P(s_1=s)+\dots$。
  与 `sutton2000` 的 $d^\pi$（共享笔记 C26）是同一个量。
- **式 (2)**：$\eta(\tilde\pi)=\eta(\pi)+\sum_s\rho_{\tilde\pi}(s)\sum_a\tilde\pi(a|s)A_\pi(s,a)$；随后说明若每个状态的期望优势都非负，则性能不降（转述）。
- **式 (3)，局部近似**：$L_\pi(\tilde\pi)=\eta(\pi)+\sum_s\rho_\pi(s)\sum_a\tilde\pi(a|s)A_\pi(s,a)$，
  原文说明它用 $\rho_\pi$ 而不是 $\rho_{\tilde\pi}$，“ignoring changes in state visitation density”。
- **式 (4) 前后**：对可微的参数化策略，$L_\pi$ 与 $\eta$ 在 $\theta_0$ 处一阶相同（值相等、梯度相等），原文写 “matches η to first order”，
  并注明见 Kakade & Langford (2002)；紧接着指出：足够小的一步若改进 $L$ 也会改进 $\eta$，但没有告诉我们步子能迈多大（转述）。
- **式 (14)（§5）**：把对动作的求和换成重要性采样估计，采样分布记为 $q$；single path 方法中 $q(a|s)=\pi_{\theta_{old}}(a|s)$。
- 课文用法：M9 §5 的“性能差分恒等式”与“替代目标只在一阶上可信”。Kakade & Langford 2002 原文**未读**，只按 TRPO 的引用转述，课文写“TRPO 论文引 Kakade & Langford (2002)”。
- **（事实审阅补充，2026-09-25，`arxiv.py grep 1502.05477`）实践中的约束与解法**：§4 用 “heuristic approximation which considers the average KL divergence”
  代替逐状态的最大 KL；§5 与附录 C：搜索方向由近似解 $Ax=g$ 得到，$A$ 是 Fisher 信息矩阵，即 “the quadratic approximation to the KL divergence constraint”，
  用共轭梯度加线搜索求解（“conjugate gradient algorithm followed by a line search”）；FIM 取 $D_{KL}(\pi_{\theta_{old}}(\cdot|s)\|\pi_\theta(\cdot|s))$ 的 Hessian。
  课文 §6 “信赖域”一段据此写“目标线性近似、KL 约束二次近似（Fisher）、共轭梯度加线搜索”。

### 1.2 PPO 的几处原句（`schulman2017ppo`，事实审阅补充，2026-09-25）

- 来源：arXiv:1707.06347 **v2** PDF（页脚 “arXiv:1707.06347v2 [cs.LG] 28 Aug 2017”）。无 HTML 渲染（`arxiv.py grep` 报 “No HTML rendering”），
  用 WebFetch 取得 PDF、pypdf 抽文本（抽取稿在 scratch，未入库）。
- **CPI 的来历（§3，式 (6) 之后）**：“The superscript CPI refers to conservative policy iteration [KL02], where this objective was proposed.” 课文 §5 “下标 CPI 取自 conservative policy iteration” 据此。
- **TRPO 的约束与解法（§2.2，式 (3)(4)）**：约束 $\hat\E_t[\mathrm{KL}[\pi_{\theta_{old}}(\cdot|s_t),\pi_\theta(\cdot|s_t)]]\le\delta$；
  “approximately solved using the conjugate gradient algorithm, after making a linear approximation to the objective and a quadratic approximation to the constraint.”
- **TRPO 的缺点（§1）**：TRPO “is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing”（转述时只保留这一短语，课文写“相对复杂……不兼容”）。
- **表 1 的 7 个环境**（§6.1 脚注 2）：HalfCheetah、Hopper、InvertedDoublePendulum、InvertedPendulum、Reacher、Swimmer、Walker2d（均为 “-v1”），与课文“7 个连续控制环境”一致。

### 1.3 GAE 论文里没有倒推递归（`schulman2016gae`，事实审阅补充，2026-09-25）

- `arxiv.py grep 1506.02438 'recurs|backward'` 零命中：v6 只给出式 (16) 的求和形式，没有写 $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$。课文 (M9.18) 本就没有给它挂引用；
  参考文献卡片原写“倒推递归……以它为准”，已改为“倒推递归是由式 (16) 直接推出的实现形式，论文没有写出”。共享笔记 §9 的 `schulman2016gae` 卡片草稿有同样的措辞（未改共享笔记）。
- 同次 grep 确认 §3 原句 “Empirically, we find that the best value of λ is much lower than the best value of γ”（课文“GAE 论文发现最好的 λ 远低于最好的 γ”）。

### 1.4 其他一手核对（事实审阅补充，2026-09-25）

- `sutton2000` 卡片的 URL（proceedings.neurips.cc …/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html）用 WebFetch 打开：题名、四位作者（含中间名缩写）、
  “Advances in Neural Information Processing Systems 12 (NIPS 1999)” 与卡片一致。
- DeepSeek-R1 的期刊版题名与 arXiv 不同：“DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning”，Nature 645:633–638【交叉引用】`deepseek/reference-docs/lesson-13-doc.md` 表头 `r1-nature`。
  参考文献卡片的出处行改为“arXiv:2501.12948v2；期刊版 Nature 645:633–638，题为……”，避免把 arXiv 题名配上 Nature 卷页。
- V3.2 的优势是否除以标准差：`lesson-15-doc` §7 E2、§8 U3 记录论文前一句说 “normalizing the outcome reward within each group”，公式却只减均值，未见代码可核。
  课文 §7 与 `dsv32` 卡片原写“把优势改成只减组均值、不除标准差”，已改成“印出的优势公式只减组均值……论文前后表述不一”。
- V3.2 的 $\pi_{old}$：原文 “π_old here denotes the sampling probability directly returned by the inference framework” 出现在序列掩码式 (8)(9) 处，式 (6) 的重要性比用的是同一个 $\pi_{old}$
  （`lesson-15-doc` §2.1、§2.3 转录）。课文 §5 改成“目标函数里的 $\pi_{old}$ 是推理引擎返回的采样概率……重要性比的分母用的也是它”，不再说“所以重要性比分母直接取……”（原句把论文针对掩码 KL 的说明换成了对重要性比的说明）。
- DeepSeekMath 关于价值网络的原话位置：“价值函数通常与策略同等规模”“每个 token 都准确的价值函数更难”在 DSM §4.1.1（`lesson-08-doc` C15），第 8 课把它放在 §1（p1），
  M9 §4 的“这一页用在哪里”原来链到第 8 课 §3，已改为 §1。

### 1.5 共享笔记里发现、本课已避开的问题（未改共享笔记，供维护者处理）

- `math-refs-doc.md` §3.6 表写“InstructGPT PPO：$\gamma=1$，$\lambda=0.95$（【交叉引用】lesson-03-doc C35）”。C35 是 **Stiennon 2020** 的附录 B.1；
  InstructGPT（lesson-03-doc C12）只写 “No discount is applied when estimating the generalized advantage”，**没有给 λ**。本课正文把 $\lambda=0.95$ 归给 Stiennon、InstructGPT 只说不折扣，是对的；
  图 M9.7 绿线的图例原写“语言模型 RLHF 的常见取值”，已改成“Stiennon 等人的取值”。
- `math-refs-doc.md` §9 的 `williams1992` 卡片草稿说它是“M09 对数导数技巧……的原始出处”。得分函数 / 似然比方法本身不始于 Williams（本次未核更早的文献，所以课文不写首创者）；
  本课卡片改成“REINFORCE 估计量（得分函数在强化学习里的用法）……的原始出处”。卡片 finding 的“学习率相同时恰为梯度”也补全为“恰为学习率乘以梯度”（C23）。

## 2. 本课自己的推导与数值（【笔者推导】，脚本在 scratch 与 `math/figures/gen/m09_*.py`）

所有数字用 Python 3.11.8 + numpy 1.26.4 复算。

### 2.1 玩具 MDP（“2+3=” 两个 token）

设定（笔者构造，非任何论文数据）：第一个 token 取 “5”（对）或 “6”（错），$\pi(5|q)=0.6$；第二个 token 取 “！”（肯定）或 “吧”（犹豫），
$\pi(！|q5)=0.7$、$\pi(！|q6)=0.5$。回报只在结尾给：“5！”=1，“5吧”=0.5，“6！”=0，“6吧”=0.2。$\gamma=1$。

| 量 | 值 |
|---|---|
| $V(q5)$、$V(q6)$、$V(q)=J$ | 0.85、0.10、0.55 |
| $Q(q,5)$、$Q(q,6)$ | 0.85、0.10 |
| $A(q,5)$、$A(q,6)$ | +0.30、−0.45 |
| $A(q5,！)$、$A(q5,吧)$、$A(q6,！)$、$A(q6,吧)$ | +0.15、−0.35、−0.10、+0.10 |
| 四条回答概率 | 0.42、0.18、0.20、0.20 |
| 精确梯度 $\partial J/\partial z_{s,a}=d(s)\pi(a|s)A(s,a)$ | $z_{q,5}$ 0.18，$z_{q,6}$ −0.18，$z_{q5,！}$ 0.063，$z_{q5,吧}$ −0.063，$z_{q6,！}$ −0.02，$z_{q6,吧}$ 0.02（中心差分 $h=10^{-6}$ 一致） |
| REINFORCE 对 $z_{q,5}$ 的单样本估计，$b=0$ | 0.4、0.2、0、−0.12；均值 0.18，方差 0.04488 |
| 同上，$b=V(q)=0.55$ | 0.18、−0.02、0.33、0.21；均值 0.18，方差 0.01188 |
| 同上，方差最优常数基线 $b^*=\E[s^2R]/\E[s^2]$ | $b^*=0.40$，方差 0.00648 |
| 精确 $V$ 下的 TD 残差 | 每条回答每一步 $\delta_t=A(s_t,a_t)$ 恰好相等（转移确定性，见 2.3） |

### 2.2 Bernoulli 基线实验（共享笔记 N72 的闭式）

$p=0.3$，$r=\mathbb 1[y=1]+10$，$g_b=(r-b)(y-p)/(p(1-p))$（对 $p$ 求导的 REINFORCE，Williams 的 Bernoulli 单元，C25）。
- 闭式：$\E[g_b]=1$ 对所有 $b$；$\Var[g_b]=(b-b^*)^2/(p(1-p))$，$b^*=11-p=10.7$。
- 取值：$b$=0：545.19；5：154.71；10：2.3333；10.3（平均奖励）：0.7619；10.7：0；12：8.0476。
- 模拟（`default_rng(0)`，$2\times10^5$ 样本）：$b$=0 均值 0.9890、方差 544.97；10：0.9993、2.332；10.3：0.9996、0.7616（与 N72 一致）。
- 看点：平均奖励 10.3 是好基线但不是最优；两种结果、一个参数时，最优基线能把方差压到 0（$g$ 两个取值都等于 1）。

### 2.3 确定性转移下 TD 残差等于优势

语言模型的“转移”是把 token 接到前缀后面，$s_{t+1}$ 由 $(s_t,a_t)$ 唯一确定，所以 $Q^\pi(s_t,a_t)=r_t+\gamma V^\pi(s_{t+1})$ 不需要对下一状态取期望；
若 $V=V^\pi$，则 $\delta_t=A^\pi(s_t,a_t)$ 逐样本成立，不只是期望意义下（GAE 论文式 (10) 是期望形式）。
推论：LLM 场景里 GAE 的 λ=0 端的噪声只来自 $V$ 的误差，λ=1 端的噪声来自未来 token 的随机性。

### 2.4 序列级重要性比的有效样本量

模型：每个 token 的对数比 $\ell_t=\log(\pi_\theta/\pi_{\theta_{old}})$ 独立同分布 $N(-\sigma^2/2,\sigma^2)$（使 $\E[e^{\ell_t}]=1$），
序列比 $W=\prod_te^{\ell_t}$。则 $\E[W]=1$，$\E[W^2]=e^{T\sigma^2}$，有效样本比例 $\mathrm{ESS}/N\approx1/\E[W^2]=e^{-T\sigma^2}$，每 token 的
$\KL(\pi_{\theta_{old}}\|\pi_\theta)=\sigma^2/2$。
- $\sigma=0.05$（每 token KL 0.00125）：$T$=100 → 0.779；1000 → 0.0821；4000 → $4.5\times10^{-5}$。
- $\sigma=0.1$：$T$=100 → 0.368；1000 → $4.5\times10^{-5}$。$\sigma=0.02$：$T$=1000 → 0.670；4000 → 0.202。
- 与 M04 的 F04-2 同形：均值差 $\mu$ 的高斯提议 ESS 为 $e^{-\mu^2}$，这里 $T\sigma^2$ 扮演 $\mu^2$。
- 注：用 2 万条样本直接估 ESS 得 0.047（理论 0.082），重尾让 ESS 的样本估计本身也不稳，课文不引模拟值。

### 2.5 小批次上的替代目标（PPO 为什么要裁剪）

玩具 MDP，批次只有两条回答：“5吧”（$R=0.5$）与 “6吧”（$R=0.2$），基线取批次均值 0.35，于是两条回答的所有 token 优势为 ±0.15。
沿经验替代目标 $\hat L^{CPI}$ 在 $\theta_{old}$ 处的梯度方向（单位化，6 个 logit）走一条直线，步长 $\alpha$：
- $\nabla\hat L^{CPI}$ 与真梯度 $\nabla J$ 的夹角余弦 0.498：它把 “吧” 在 “5” 之后的概率也往上推（样本噪声）。
- 真实改进 $J-J_{old}$：$\alpha$=0.5：+0.0556；1.0：+0.0842；最大 +0.0893（$\alpha\approx1.35$）；3：+0.0339；4：−0.0029；$\alpha\to\infty$：−0.0500（$J\to0.5$）。
- $\hat L^{CPI}$ 单调上升到上确界 0.1875；$\hat L^{CLIP}$（$\varepsilon=0.2$）在 $\alpha\approx0.757$ 处达到平台 0.0300，之后梯度为 0；此时 $J-J_{old}=+0.0737$。
- 另：沿**真**梯度方向走（精确优势），$J-J_{old}$ 与精确 $L^{CPI}$ 在 $\alpha$=0.5 时为 0.1266 与 0.1237、$\alpha$=1 时 0.2258 与 0.2166：一阶相同，二阶起分开（状态分布变化，TRPO 式 (3)）。

### 2.6 其他

- GAE（N70）的有限视界权重：$t=0$ 处 $\hat A^{(1)},\hat A^{(2)},\hat A^{(3)}$ 的权重为 $0.05,0.0475,0.9025$（尾部质量全部落在最后一个 $k$ 步估计上），加权和 0.451484 与递推一致。
- $(\gamma\lambda)^l$ 的半衰期：$\gamma=0.99,\lambda=0.95$ 时 $\ln0.5/\ln0.9405=11.3$ 步；有效视界 $1/(1-\gamma\lambda)=16.8$（N71）。
- PPO 裁剪另两点：$(\rho,A)=(1.1,+1)$ → 1.1，梯度照常；$(0.9,-1)$ → −0.9，梯度照常。

### 2.7 课文里的其他数字（全部复算）

- 长回答里 λ<1 的衰减：$0.95^{100}=0.00592$，$0.95^{1000}=5.29\times10^{-23}$（§4 失败模式；把 R1 附录的 λ 对比归因于此是笔者分析，论文未这样解释）。
- 价值网络估错（$V(\text{2+3=5})=0.70$）时第一个 token 的优势：λ=0 → 0.15，λ=1 → 0.45，λ=0.95 → 0.435（真值 0.30）。
- 伯努利基线：$b=0$ 与 $b=10.3$ 把标准误压到 0.1 所需样本 54,519 与 77（=Var/0.01）；两者方差比 545.19/0.7619 = 715.6。
- 序列比 ESS：$\sigma=0.05$、$T=500$ → $e^{-1.25}=0.287$；$\sigma=0.1$ 时 ESS/N=1% 需 $T=\ln100/0.01=460.5$；$\sigma=0.02$、$T=4000$ → $e^{-1.6}=0.202$；模型内 ESS/N $=e^{-2\,\mathrm{KL}_{\text{整段}}}$（$\mathrm{KL}_{\text{整段}}=T\sigma^2/2$）。
- 高斯 IS：$N(0.5,1)$ 对 $N(0,1)$ 的 ESS/N $=e^{-0.25}=0.779$（$\E[w^2]=e^{\mu^2}$）。
- 练习 4（把 $\pi(\text{！}\mid\text{2+3=6})$ 改为 0.2）：$V=0.16$，$J=0.574$，$A=+0.276/-0.414$，$\partial J/\partial z_{q,5}=0.1656$，“2+3=6”两个 logit 的梯度 ±0.0128。
- 练习 5（$p=0.5$）：$b^*=10.5$，方差 441 / 1 / 0，所需样本 44,100。
- 练习 6（$\gamma=1$，λ=0.9）：$\delta=(0.1,0.2,0.2)$，$\hat A=(0.442,0.38,0.2)$；λ=1 时 $\hat A_0=0.5$。
- 练习 7：裁剪值 (0.6, 0.72, −0.44, −0.8)，$L^{CLIP}=0.02$，$L^{CPI}=0.03875$；$\varepsilon=0.1$ 小练习：(1.1, 0.85, −1.15, −0.9)。
- 组均值基线因子 $(G-1)/G$：G=4 → 0.75，16 → 0.9375，64 → 0.984375（共享笔记 N74，笔者推导）。

### 2.8 教学审阅时新增的数字（2026-09-24，【笔者推导】，全部复算）

- 图 M9.3（`m09_score_push.py`，§2 “得分函数的推与拉”）：第一步 $\pi=(0.6,0.4)$，$z_5$ 分量的得分 $+0.4/-0.6$；每支箭头 = $\pi(a)\times$权重$\times s(a)$。
  权重 1：$+0.240/-0.240$，合 0；权重 $Q=(0.85,0.10)$：$+0.204/-0.024$，合 $0.180$；权重 $Q-0.55$：$+0.072/+0.108$，合 $0.180$；
  权重 $Q+10$：$+2.604/-2.424$，合 $0.180$。单次采样（只看第一步、按 $Q+10$ 加权）的估计值 $10.85\times0.4=4.34$ 或 $10.10\times(-0.6)=-6.06$。
  脚本内 assert 全部通过。
- §7 例题（G = 4 的组）：回答“5！”“5吧”“6吧”“5！”，奖励 $(1,0.5,0.2,1)$，均值 $0.675$，只减均值的优势 $(+0.325,-0.175,-0.475,+0.325)$；
  与图 M9.2 的真实优势对照，8 个 token 中“5吧”的“5”（真值 +0.30）与“6吧”的“吧”（真值 +0.10）被推错方向。
- §5 前置框：$\mathrm{ESS}=(\sum w)^2/\sum w^2\approx N/\E[w^2]$（M04 §7 的 (M4.29)）；正态指数矩 $\E[e^{cx}]=e^{cm+c^2s^2/2}$（配方，标准结果）。
- 审阅时把课文里的“共享笔记 Nxx / Wxx / F09-1”出处标记改成了 HTML 注释（不进页面，仍留在源文件里供核对）。

### 2.9 第二次教学审阅（2026-09-24 晚，只改表述与记号，没有新增事实）

- 记号：§3 最优基线里得分函数的分量由 $s$ 改为 $u$（$s$ 全课表示状态），练习 10 的向量 $\mathbf s$ 改为 $\mathbf u$；§5 前置框的正态分布写成 $N(m,v)$（$v$ 为方差），
  指数矩 $\E[e^{cx}]=e^{cm+c^2v/2}$。上面 2.1 表里的 $\E[s^2R]/\E[s^2]$ 即课文的 $\E[Ru^2]/\E[u^2]$。
- 新增推导（标准结果，逐行可验）：伯努利得分函数 $\frac{d}{dp}\log\Pr(y)=\frac yp-\frac{1-y}{1-p}=\frac{y-p}{p(1-p)}$；
  $b^*=\frac{11(1-p)+10p}{(1-p)+p}=11-p$（分子分母同乘 $p(1-p)$）。
- (M9.10) 旁补充：式子把每个状态的 logits 当独立参数（表格型 softmax，玩具模型 6 个 logit）；真实 LLM 的 logits 共享网络，按链式法则回传。

### 2.10 数学审阅（2026-09-24 深夜，逐式重推、逐数复算）

全部行间公式与 `::: deriv` 逐行重推，课文、例题、练习答案、表格与 11 个生成脚本里的数字用 Python 3.11 + numpy 重算
（脚本在 scratch，未入库）。除下面一处外全部一致。

- **勘误：共享笔记 N45 的高斯重要性采样模拟值复现不出。** 目标 $\mathcal N(1,1)$、提议 $\mathcal N(0,1)$、$10^6$ 样本：
  `default_rng(0).standard_normal` 得 $\frac1N\sum w_ix_i=1.00331$、平均权重 1.00149、ESS/N 0.3679；
  `np.random.seed(0)`/`RandomState(0)` 得 1.00212、1.00125、0.3684；`random.seed(0)`+`gauss` 得 1.00197、1.00102、0.3684；
  float32 版 1.00406、1.00175、0.3688。N45 的 1.00201 / 1.00145 / 0.3697 与哪一种都对不上（共享笔记文末也注明 N45 未重跑）。
  课文 §5 改用可复现的 `default_rng(0)` 数值：1.0033、1.0015、0.3679（理论 $1/e=0.3679$）。
- **补充的条件与精确化**（【笔者推导】）：
  - 伯努利基线例：方差比不减基线小的充要条件是 $\abs{b-b^*}<b^*$，即 $0<b<21.4$；$b=22$ 时 $11.3^2/0.21=608.0>545.2$。
  - “只用之后的回报”方差**通常**更小但不必然：保留过去的奖励等于减去基线 $-\sum_{k<t}\gamma^kr_k$。反例：两步，$r_0\equiv-10$，
    第二步是 $p=0.3$ 的伯努利动作、$r_1=10+\mathbb 1[y=1]$。对第二步参数，完整回报给出 $b=10$ 的估计（方差 2.33），
    只用之后的回报给出 $b=0$ 的估计（方差 545）。图 M9.11 相应改成“方差通常变小”。
  - $\delta_t-A^\pi=\gamma e(s_{t+1})-e(s_t)$ 逐样本成立要求确定性转移（语言模型满足）；一般 MDP 只在条件期望下成立。
  - 折扣访问量 $d^\pi$ 的总和 $1/(1-\gamma)$ 只在回合不终止时成立。
  - (M9.25) 的右式是期望：逐 token 惩罚之和的**期望**等于 $-\beta\,\KL$。
  - 裁剪项的折点：$A>0$ 时在 $\rho=1+\varepsilon$、$A<0$ 时在 $\rho=1-\varepsilon$；$A=0$ 的项恒为 0。
  - 图 M9.9 的两条替代目标是在两条样本上算的经验值，不是期望（原说明写成“三条曲线都用精确期望”）。
  - §5 例题“替代目标低估改进”的解释复核：$\alpha=0.5$ 时 $(d^{new}(q5)-0.6)\,(\text{imp}_{q5}-\text{imp}_{q6})=0.1445\times(0.0232-0.0037)=0.0028$，
    正是 0.1266 与 0.1237 之差。
- **记号**：正态分布改写 $\mathcal N$（同 M04，避开全局表里的 $N$）；V3.2 的 $\pi_{old}$ 保留论文写法并说明它是推理引擎的采样概率，
  不同于 $\pi_{\theta_{old}}$；Schulman 博客的比值 $r$ 在参考文献卡片里改写成 $w$；图 M9.1 去掉易与价值 $V$ 混淆的“$|V|$”。

### 2.11 事实审阅（2026-09-25）：数字与练习复核

- 用 Python 3.11 + numpy 从头重算（脚本在 scratch 的 `facts-m09-rl-math/`，未入库）：玩具模型的 $V/Q/A$、六个 logit 的精确梯度（中心差分）、三种基线下的单样本估计与方差、
  伯努利例子的闭式方差与 `default_rng(0)` 模拟、三步 GAE（$\gamma=0.99$ 与 $\gamma=1$）、图 M9.6 的行列两种读法、高斯重要性采样的 `default_rng(0)` 模拟、序列比 ESS 表、
  PPO 裁剪四种情形与练习 7、图 M9.9 的五个关键数（余弦 0.498、峰值 +0.0893 @ α=1.346、过零 α=3.905、极限 −0.050、裁剪平台 α=0.756 处真实改进 +0.0736）、
  §5 例题的 0.1266/0.1237 与 0.2258/0.2166。全部与课文一致。
- 12 道练习与 6 个小练习逐题从头解过，答案全部一致，没有改动。唯一的舍入口径：伯努利例子 $b=10.3$ 时 $n=\Var/0.01=76.2$，课文写“只要 77 个”（向上取整），$b=0$ 写“约 54,519 个”（54,519.05），两者口径不同但都不算错。

## 3. 课文用到的交叉引用（未在此重核）

- lesson-03-doc：C10（InstructGPT 的 bandit 环境、逐 token KL、价值函数由 RM 初始化），C12（β=0.02、ε=0.2、GAE 不折扣），C35（Stiennon γ=1、λ=0.95），
  C39、C41、C42、C45（PPO 论文：多步优化破坏性更新、裁剪式 (7) 与悲观下界、对旧策略的自适应 KL、表 1 消融），§2.3 的逐 token KL 推导。
- lesson-08-doc：C15–C18、C20、C26（GRPO 去掉 critic、KL 进损失、k3、结果监督优势、G=64、β=0.04、单次更新），C40（R1 附录：PPO 在 λ=0.95 时差于 GRPO、λ=1.0 时接近；
  V3.2 只减均值；Dr. GRPO 两种偏差），§2.5 (b)(c)(f)(g)(h)。
- lesson-15-doc：C21–C23、C26–C28（V3.2 的无偏 KL、off-policy 两个来源与序列掩码、Keep Routing、Keep Sampling Mask）。
