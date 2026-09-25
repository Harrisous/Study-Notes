# 第 02 课研究笔记：GPT 的训练——预训练目标、优化与 scaling laws

> 对应课文：`deepseek/src/lesson-02-gpt-training.md`（尚未写）。建立于 2026-09-24。
> 本文件只存放**核对过的**引文、公式、规格、复算数字、勘误和“未能核实”清单，不是课文。
> 核对方法同 `lesson-01-doc.md`：arXiv LaTeXML 全文 grep；OpenAI 报告与早期版本取 PDF 抽文字；Chinchilla 额外取了
> **arXiv TeX 源码**（`arxiv.org/e-print/2203.15556`）和 **NeurIPS 2022 论文集 PDF**；所有派生数字在 Python 里复算（§4 给表达式）。
>
> **引文政策**：英文原文只摘 ≤15 词的关键短语，附节/表/式号与版本；公式、数字、超参数完整转录。
>
> 与第 1 课的分工：架构、参数量、前向 FLOPs、BPE、LN 位置在 `lesson-01-doc.md`；本课管训练目标、数据、优化器、
> 学习率调度、$6ND$、Kaplan 与 Chinchilla 的 scaling law，并为第 6 课（DeepSeek LLM 的 $M$ 与超参数 scaling law）铺路。

---

## 0. 覆盖范围与所读版本

| 短键 | 标题 | 作者 | arXiv / 出处 与所读版本 | 发表状态 | 日期 |
|---|---|---|---|---|---|
| radford2018 (GPT-1) | Improving Language Understanding by Generative Pre-Training | Radford, Narasimhan, Salimans, Sutskever | OpenAI PDF（元数据 CreationDate 2018-06-08） | OpenAI 技术报告 | 2018-06 |
| radford2019 (GPT-2) | Language Models are Unsupervised Multitask Learners | Radford, Wu, Child, Luan, Amodei, Sutskever | OpenAI PDF（元数据 CreationDate 2019-02-14） | OpenAI 技术报告 | 2019-02 |
| brown2020 (GPT-3) | Language Models are Few-Shot Learners | Brown, Mann, Ryder, Subbiah, Kaplan, Dhariwal 等 31 人 | 2005.14165 **v4**（arXiv HTML；表 2.1、2.2、D.1，附录 A、B） | NeurIPS 2020 | v1 2020-05-28；v4 2020-07-22 |
| kaplan2020 | Scaling Laws for Neural Language Models | Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei | 2001.08361 **v1**（唯一版本；HTML + PDF） | arXiv 预印本 | 2020-01-23 |
| hoffmann2022 (Chinchilla) | Training Compute-Optimal Large Language Models | Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai, Rutherford 等 22 人 | 2203.15556 **v1**（唯一 arXiv 版本；HTML + TeX 源码）；另读 NeurIPS 2022 论文集 PDF（15 页） | NeurIPS 2022。论文集网页标题为 “An empirical analysis of compute-optimal large language model training”，PDF 标题仍为 arXiv 标题（§7-F12） | 2022-03-29 |
| besiroglu2024 | Chinchilla Scaling: A replication attempt | Besiroglu, Erdil, Barnett, You（Epoch AI） | 2404.10102 **v2** | arXiv 预印本 | 2024-04-15（v1） |
| loshchilov2019 (AdamW) | Decoupled Weight Decay Regularization | Loshchilov, Hutter | 1711.05101 **v3**；**v1 标题 “Fixing Weight Decay Regularization in Adam”**（GPT-1 引的就是 v1） | ICLR 2019（arXiv comments） | v1 2017-11-14；v2 2018-02-14；v3 2019-01-04 |
| vaswani2017 | Attention Is All You Need | Vaswani 等 8 人 | 1706.03762 v7（只用 §5.3 式 (3) 学习率与 §5.4 正则） | NIPS 2017 | 2017-06-12（v1） |
| xiong2020 | On Layer Normalization in the Transformer Architecture | Xiong 等 | 2002.04745 v2（只用 warm-up 部分） | ICML 2020 | 2020-02-12（v1） |
| palm2022 | PaLM: Scaling Language Modeling with Pathways | Chowdhery 等 | 2204.02311 v5，附录 B | — | 2022-04-05 |
| deepseekllm2024 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism | DeepSeek-AI | 2401.02954 **v1**，只读 §3.2 的 $M$ 定义（第 6 课详读） | arXiv 预印本 | 2024-01-05 |
| torchadamw | PyTorch `torch/optim/adam.py`、`adamw.py` | PyTorch | GitHub main，2026-09-24 读取 | 制品 | — |
| gpt2code | openai/gpt-2 仓库 | OpenAI | GitHub | 制品 | — |

---

## 1. 核心论断与原文引文

### 1.1 训练目标：自回归分解、最大似然、交叉熵、困惑度

**D1 自回归分解（GPT-2 式 (1)，§2）。** 语言有自然顺序，因此把联合概率分解为条件概率之积，引 Jelinek & Mercer 1980、Bengio et al. 2003。
“factorize the joint probabilities over symbols as the product of conditional probabilities”（§2，节选）。
- 式 (1) **如印刷**：$p(x)=\prod_{i=1}^{n}p(s_n\mid s_1,\dots,s_{n-1})$——乘积下标是 $i$，因子里却写 $s_n$（pypdf 抽取确认）。应读作 $p(s_i\mid s_1,\dots,s_{i-1})$（§7-F15）。

**D2 GPT-1 的预训练目标（式 (1)，§3.1）。** 最大化 $L_1(\mathcal U)=\sum_i\log P(u_i\mid u_{i-k},\dots,u_{i-1};\Theta)$，$k$ 为上下文窗口，
“standard language modeling objective to maximize the following likelihood”（节选）。参数用 SGD 类方法训练。

**D3 交叉熵即论文的主指标（Kaplan §2）。** “We optimize the autoregressive log-likelihood (i.e. cross-entropy loss)”，在 1024-token 上下文上平均，
单位是 nats（§6.3 的 $L^*\sim1.7$ nats/token）。GPT-3 图 3.1 纵轴同样是 “cross-entropy validation loss”。

**D4 困惑度要说明单位。** GPT-1 在 BooksCorpus 上报告 “a very low token level perplexity of 18.4”（§4.1）；
GPT-2 强调各基准报告的是“按规范预测单位（字符、字节或词）平均的负对数概率”的缩放或指数化版本，WebText LM 用可逆去分词器后按规范单位重算（§3.1）。
Vaswani 表 3 注明 “perplexities are per-wordpiece … should not be compared to per-word perplexities”（表 3 标题节选）。

**D5 一次前向算出 $T$ 个损失、文档打包。** GPT-3 附录 B：训练时总是用满 $n_{ctx}=2048$，把短文档拼接进同一序列，
“Sequences with multiple documents are not masked in any special way”，文档之间用 end of text token 分隔（节选）。
即：一个 2048-token 序列通过因果掩码一次得到 2048 个下一词损失（掩码机制见第 1 课 C5）。

### 1.2 数据管线

**D6 GPT-1：BooksCorpus。** 7,000 多本未出版书籍，含长段连续文本，便于学习长程依赖；对比 1B Word Benchmark “shuffled at a sentence level - destroying long-range structure”（§4.1 节选）。ftfy 清洗 + spaCy 分词。

**D7 GPT-2：WebText（§2.1）。** 抓取 Reddit 上至少获得 3 karma 的外链（“a heuristic indicator”）；45M 链接的文本子集；只含 2017 年 12 月之前的链接；
去重与启发式清洗后 “slightly over 8 million documents for a total of 40 GB of text”；删除了全部 Wikipedia 文档（避免与评测集重叠）。

**D8 GPT-3：三步提质（§2.2 + 附录 A）。**
1. 以 WebText 等精选语料为正例、未过滤 Common Crawl 为负例训练逻辑回归分类器（Spark tokenizer + HashingTF 特征），按
   `np.random.pareto(α) > 1 − document_score` 保留文档，$\alpha=9$（附录 A）。
2. 用 Spark MinHashLSH（10 个哈希）在每个数据集内部做模糊去重，并从 CC 中模糊去除 WebText；整体数据量平均减少 10%（附录 A）。
3. 混入已知高质量语料：扩充版 WebText（WebText2）、Books1、Books2、英文 Wikipedia（§2.2）。
- CC：41 个月度分片（2016–2019），过滤前 45TB 压缩文本、过滤后 570GB，“roughly equivalent to 400 billion byte-pair-encoded tokens”（§2.2）。
- 采样不按大小比例：高质量数据集被采样 2–3 次，CC 与 Books2 少于 1 次（§2.2）。
- 评测污染：用 13-gram 重叠过滤，但 “a bug in the filtering caused us to ignore some overlaps”，因成本未重训（§2.2、附录 C）。

**D9 GPT-3 表 2.2 的数据配比（如印刷）：** CC(filtered) 410B tokens / 60% / 0.44 epochs；WebText2 19B / 22% / 2.9；Books1 12B / 8% / 1.9；Books2 55B / 8% / 0.43；Wikipedia 3B / 3% / 3.4。
表题说部分数据集最多被看 3.4 次。——**权重列与 epoch 列不能互相复算**（§4.4、§7-F9）。

### 1.3 GPT-1 → GPT-2 → GPT-3 的训练设置与主张

**D10 GPT-1：预训练 + 判别式微调（§3.2、§4.1）。** 微调目标 $L_3(\mathcal C)=L_2(\mathcal C)+\lambda\cdot L_1(\mathcal C)$，即把语言模型目标作辅助项，
$\lambda=0.5$（“λ was set to 0.5”，pypdf 确认符号）；微调 lr 6.25e-5、batch 32、3 epochs 通常足够、0.2% 训练步数线性 warmup；
预训练：Adam，最大 lr 2.5e-4，前 2000 次更新从 0 线性升温，再余弦退火到 0；100 epochs，minibatch 64 条 512-token 连续序列。
结果：12 个任务中 9 个超过当时最优（摘要）。

**D11 GPT-2：零样本（§1、§3）。** 最大模型 1.5B，在 8 个语言建模数据集中 7 个取得零样本最优，但 “still underfits WebText”（摘要）；
各模型学习率 “manually tuned for the best perplexity on a 5% held-out sample of WebText”（§3），**具体数值、步数、训练 token 数均未报告**。

**D12 GPT-3：规模 + 上下文学习（§1–2）。** 175B；“in-context learning” 指预训练得到的能力在推理时的前向传播内被调用（图 1.1 标题）；
few-shot 指推理时给 $K$ 个示例、不更新权重，$K$ 通常 10–100，受 $n_{ctx}=2048$ 限制（§2）。所有模型训练 300B tokens（表 2.1 标题）。

**D13 GPT-3 优化细节（附录 B，逐项）：** Adam $\beta_1=0.9$，$\beta_2=0.95$，$\epsilon=10^{-8}$；全局梯度范数裁剪到 1.0；
余弦衰减到峰值的 10%，历时 260B tokens，之后以 10% 继续；前 375M tokens 线性 warmup；
batch size 在前 4–12B tokens 内从 32k tokens 线性增加到满值（依模型大小）；数据无放回采样（直到 epoch 边界）；weight decay 0.1（引 [68] = Loshchilov & Hutter）。
§2.3：用梯度噪声尺度（gradient noise scale）指导 batch size 选择；大模型用更大 batch、更小学习率；训练在 V100 上。

### 1.4 优化器与调度

**D14 AdamW：L2 正则 ≠ weight decay（对 Adam）。**
- 命题 1：对标准 SGD，weight decay $\lambda$ 等价于 L2 系数 $\lambda'=\lambda/\alpha$——两个超参数被学习率耦合（§2）。
- 命题 2：对自适应梯度算法，不存在 L2 系数使之等价于解耦 weight decay（§2）。
- 直觉：L2 的梯度会被 Adam 的 $\sqrt{\hat v}$ 一起缩放，“weights that tend to have large gradients … do not get regularized as much”（§2 节选）。
- 算法 2 第 12 行（AdamW）：$\theta_t\leftarrow\theta_{t-1}-\eta_t\big(\alpha\hat m_t/(\sqrt{\hat v_t}+\epsilon)+\lambda\theta_{t-1}\big)$——衰减项乘的是**调度乘子 $\eta_t$**，不乘 $\alpha$。
- 附录 B.1 “normalized weight decay”：$\lambda=\lambda_{norm}\sqrt{b/(BT)}$。
- 默认超参（算法 2 第 1 行）：$\alpha=0.001,\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$。

**D15 PyTorch 的 AdamW 与论文参数化不同（代码核对）。** `adam.py` 第 418–420 行：`if decoupled_weight_decay: param.mul_(1 - lr * weight_decay)`；
`AdamW` 默认 `weight_decay=1e-2`，并强制 `decoupled_weight_decay=True`。所以 PyTorch 的每步衰减率是 $\text{lr}\times\text{wd}$，
而论文里是 $\eta_t\lambda$（与基础学习率 $\alpha$ 解耦）。“weight decay 0.1”在两种约定下含义不同（§7-F10）。

**D16 Warmup 的两种写法。** Vaswani 式 (3)：$lrate=d_{model}^{-0.5}\min(step^{-0.5},\,step\cdot warmup^{-1.5})$，warmup 4000 步，之后按步数平方根倒数衰减（§5.3）。
Xiong：Post-LN 需要 warm-up（IWSLT14 De-En 不用 warm-up 时 Adam 的 BLEU 只有 8.45，用则约 34），Pre-LN 可以去掉（§3.2、摘要）；
并反驳“warm-up 只是为了 Adam 的方差”——SGD 也需要（§3.2 末段，引 Liu et al. 2019a 的 RAdam 观点）。

**D17 混合精度（Chinchilla §4.1）。** 前向、反向用 bfloat16，分布式优化器状态里保存 float32 权重副本（引 ZeRO/Rajbhandari et al. 2020）。
另：Chinchilla 改用 AdamW 而非 Gopher 的 Adam；脚注 8 说 AdamW 模型的训练表现要到余弦周期约 80% 处才超过 Adam 模型（“around 80% of the way through the cosine cycle”），但最终明显更好（图 A7）。
GPT-3 论文**没有写**数值精度（§8）。

### 1.5 训练算力 $C\approx6ND$

**D18 Kaplan 的推导（§2.1）。** 前向每 token $\approx2N$ 次加乘（“the factor of two comes from the multiply-accumulate operation”），
反向约为前向两倍，所以 “we then define the estimated non-embedding compute as $C\approx6N$ floating point operators per training token”（节选）。
前提：$d_{model}\gg n_{ctx}/12$，上下文相关项可忽略；$N$ 是**非嵌入**参数。

**D19 GPT-3 表 D.1 的逐列逻辑（标题）。** 每个激活参数每 token 前向做一次乘一次加（2 FLOPs）；反向乘 3（计算对参数与对激活的梯度，各与前向相当）→ 每参数每 token 6 FLOPs；
再乘参数量与训练 token 数；忽略注意力，因为 “it typically uses less than 10% of the total compute”（附录 D 首段节选）。
1 PF-day $=8.64\times10^{19}$ FLOPs。T5 因编码器–解码器只激活一半参数，每参数每 token 3 FLOPs。

**D20 Chinchilla 附录 F：精确计数与 $6ND$ 相差 ≤10%。** 按嵌入、QKV、QK logits、softmax、softmax@V、输出投影、FFN、logits 逐项计数，反向取前向两倍；
表 A4：73M → 1.03，305M → 1.10，552M → 1.08，1.1B → 1.04，1.6B → 1.03，6.8B → 0.99（Ours/6ND）。Gopher 的算力用此法为 $6.3\times10^{23}$，Rae et al. 报告 $5.76\times10^{23}$。

**D21 DeepSeek LLM 用 $M$ 取代 $6N$（2401.02954 v1 §3.2，第 6 课详读）。** “non-embedding FLOPs/token $M$”包含注意力开销、不含词表计算，$C=MD$；
$6N_1=72n_{layer}d_{model}^2$，$6N_2=72n_{layer}d_{model}^2+6n_{vocab}d_{model}$，$M=72n_{layer}d_{model}^2+12n_{layer}d_{model}l_{seq}$。

### 1.6 Kaplan et al. 2020

**D22 三条单变量幂律（§1.2，式 (1.1)–(1.3)）**，常数见 §2、§3。适用条件各不相同：
(1.1) “trained to convergence on sufficiently large datasets”；(1.2) “large models trained with a limited dataset with early stopping”；
(1.3) 计算受限、数据足够、模型大小最优、batch 足够小（节选）。跨越 $C_{min}$ 八个量级、$N$ 六个量级、$D$ 两个多量级（§1.2）。

**D23 常数没有绝对意义。** $N_c,C_c^{min},D_c$ “depend on the vocabulary size and tokenization and hence do not have a fundamental meaning”（§1.2 节选）；
数值针对 WebText2（$n_{vocab}=50257$，1024 上下文）。例：参数翻倍 → 损失乘 $2^{-\alpha_N}=0.95$（§1.2 原文给出 0.95）。

**D24 形状几乎无关（§3.1）。** 固定 $N$ 时深宽比可变 40 倍而性能只轻微变化；$(n_{layer},d_{model})=(6,4288)$ 与 GPT-2 的 $(48,1600)$ 损失相差 3% 以内（图 5 标题）。
包含嵌入参数时性能看起来强烈依赖层数，排除后收敛到同一趋势（图 6）。

**D25 联合式 $L(N,D)$ 与过拟合（§4）。** 式 (1.5)/(4.1)；三条设计原则（可按词表缩放、两端极限、在 $D=\infty$ 处解析——第三条“significantly weaker”）；
表 2 的**联合拟合**：$\alpha_N=0.076$，$\alpha_D=0.103$，$N_c=6.4\times10^{13}$，$D_c=1.8\times10^{13}$（与式 (1.2) 的 0.095、$5.4\times10^{13}$ 不同，因为是全式拟合）；
过拟合程度只依赖 $N^{\alpha_N/\alpha_D}/D$；§1.1：模型每增大 8 倍，数据只需增大约 5 倍（“only need to increase the data by roughly 5x”）。

**D26 训练设置（§2.2）。** Adam，固定 $2.5\times10^5$ 步，batch 512 条 × 1024 tokens；>1B 参数的模型因显存改用 Adafactor；
3000 步线性 warmup + 余弦衰减到 0；“results at convergence were largely independent of learning rate schedule”（节选）。

**D27 算力最优分配（§6）。** 经验拟合 $N(C_{min})\propto C_{min}^{0.73}$（式 (6.1)），$B_{crit}\propto C_{min}^{0.24}$，$S_{min}\propto C_{min}^{0.03}$（式 (6.2)）；
“we should predominantly increase the model size $N$”（§6.1 节选）；图 14：计算量每增 10 倍，最优模型增 5 倍、数据只增 2 倍。
§1.1 结论：算力最优训练是 “training very large models and stopping significantly short of convergence”（节选）。

**D28 理论预测与经验的差别（§6.2）。** 由 $L(N,S_{min})$ 推出 $\alpha_C^{min}=1/(1/\alpha_S+1/\alpha_B+1/\alpha_N)\approx0.054$（式 (6.4)），$N\propto C_{min}^{0.71}$（式 (6.5)），
与经验 0.050、0.73 “in excellent agreement”。

**D29 矛盾与猜想（§6.3）。** 数据若只按 $D\propto C_{min}^{0.26}$ 增长，最终会与 $L(D)$ 相交；交点
$C^*\sim10^4$ PF-days、$N^*\sim10^{12}$、$D^*\sim10^{12}$、$L^*\sim1.7$ nats/token（式 (6.8)），“highly uncertain, varying by an order or magnitude”；
作者“conjecture”交点可能是 Transformer 的最佳性能或文本熵的估计——**推测**。

**D30 临界 batch（§5.1、式 (1.4)）。** $B_{crit}(L)=B_*/L^{1/\alpha_B}$，$B_*\sim2\cdot10^8$ tokens，$\alpha_B\sim0.21$；损失每降 13%，临界 batch 约翻倍（图 10 标题）。
§1.1：收敛时最大模型的理想 batch “roughly 1-2 million tokens”。

### 1.7 Chinchilla（Hoffmann et al. 2022）

**D31 问题陈述（式 (1)，§1）。** 固定 FLOPs 预算 $C$，求 $N_{opt}(C),D_{opt}(C)=\argmin_{N,D:\,\mathrm{FLOPs}(N,D)=C}L(N,D)$。
训练了 400 多个模型（“over 400 language models”），参数 7000 万到 160 亿以上，训练 token 50 亿到 5000 亿（摘要）；
§1 正文写的是 “under 70M to over 16B”、“5B to over 400B tokens”——摘要与正文的 token 上限（500B vs 400B+）略有出入。

**D32 核心结论。** “for every doubling of model size the number of training tokens should also be doubled”（摘要）；
按 Gopher 的算力预算，最优模型应小 4 倍、多训 4 倍 token（§1）。

**D33 三种方法（§3.1–3.3）与指数（表 2，括号为 10%/90% 自助法分位数）：**
| 方法 | $a$（$N_{opt}\propto C^a$） | $b$（$D_{opt}\propto C^b$） |
|---|---|---|
| 1. Minimum over training curves | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3. Parametric modelling of the loss | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| Kaplan et al. (2020) | 0.73 | 0.27 |
- 方法 1：每个规模训 4 种余弦周期长度（学习率衰减 10 倍，周期跨 16 倍），在 1500 个对数间隔的 FLOP 值上取最低损失包络（§3.1）；所选点都在训练的最后 15%（脚注）。
- 方法 2：9 个固定 FLOP 预算（$6\times10^{18}$ 到 $3\times10^{21}$），每个预算下扫模型大小，拟合抛物线取谷底（§3.2）。
- 方法 3：用 Huber 损失（$\delta=10^{-3}$）+ L-BFGS 拟合式 (2) 的对数（§3.3、附录 D.2）；方法 3 在大算力下预测更小的模型，作者归因于前沿的负曲率（§3.4）。

**D34 参数化损失的拟合值（附录 D.2）。** $L(N,D)=E+A/N^{0.34}+B/D^{0.28}$，$E=1.69$，$A=406.4$，$B=410.7$；作者把 $E$ 解释为“entropy of natural text”（§3.3，第一项的解释）。
**TeX 源码注释里的未四舍五入值**（`main.tex` 第 899–903 行，笔者亲自核对）：`loss_offset 0.5267228`、`param_offset 6.0073404`、`param_slope 0.33917084`、
`token_offset 6.0179186`、`token_slope 0.2849083` → $E=e^{0.5267}=1.6934$，$A=e^{6.0073}=406.4$，$B=e^{6.0179}=410.7$，$\alpha=0.3392$，$\beta=0.2849$。
图 4 标题：按 Gopher 预算，方法 3 预测的最优规模为 40B（与未四舍五入值复算吻合，§4.3）。

**D35 表 3（方法 1 的投影，如印刷）：** 400M → 1.92e19 FLOPs、8.0B tokens；1B → 1.21e20、20.2B；10B → 1.23e22、205.1B；67B → 5.76e23、1.5T；
175B → 3.85e24、3.7T；280B → 9.90e24、5.9T；520B → 3.43e25、11.0T；1T → 1.27e26、21.2T；10T → 1.30e28、216.2T。
**同一页正文却写** 175B 应用 $4.41\times10^{24}$ FLOPs、“over 4.2 trillion tokens”，280B 用约 $10^{25}$ FLOPs、6.8T tokens（arXiv v1 §3.4；TeX 第 421 行）——与表不符（§7-F6）。
NeurIPS 版把该表移到附录 A3，并删掉了这句正文。

**D36 Chinchilla 规格（表 4，§4.1）。**
| 模型 | 层 | 头 | Key/Value size | $d_{model}$ | Max LR | Batch（tokens） |
|---|---|---|---|---|---|---|
| Gopher 280B | 80 | 128 | 128 | 16,384 | $4\times10^{-5}$ | 3M → 6M |
| Chinchilla 70B | 80 | 64 | 128 | 8,192 | $1\times10^{-4}$ | 1.5M → 3M |
FFN 恒为 $4d_{model}$；两者都在训练中途把 batch 翻倍（表 4 标题）。其他差异：AdamW 代替 Adam；SentencePiece 不做 NFKC 归一化（词表 32,000，94.15% 与 Gopher 相同）；
bf16 计算 + fp32 权重副本；MassiveText 配比略改；TPUv3/TPUv4、JAX、Haiku。

**D37 与 Kaplan 分歧的原因（§2，Chinchilla 自述）。** Kaplan 对所有模型用固定 token 数与固定学习率调度；“For a fixed learning rate cosine schedule to 130B tokens”，
中间点的损失高估了“周期与 $D$ 匹配”时的损失，从而低估小数据训练的效果，导致“模型应比数据增长更快”的结论；另外 Chinchilla 的多数模型大于 500M，而 Kaplan 的多数运行“significantly smaller”、其中**许多**（“many”，不是“多数”）小于 100M，而前沿有轻微曲率（§2 节选）。
130B 与 Kaplan §2.2 的 $2.5\times10^5$ 步 × $2^{19}$ tokens $=1.31\times10^{11}$ 吻合（§4.2）。

**D38 结果。** Chinchilla 在同等算力下全面优于 Gopher（280B）、GPT-3（175B）、Jurassic-1（178B）、MT-NLG（530B）；MMLU 平均 67.5%，比 Gopher 高 7% 以上（摘要）。
表 1（当时的大模型）：LaMDA 137B/168B tokens；GPT-3 175B/300B；Jurassic 178B/300B；Gopher 280B/300B；MT-NLG 530B/270B；Chinchilla 70B/1.4T。

**D39 “约 20 tokens/参数”不是论文原话。** arXiv 全文与 TeX 源码中搜索 “per param”“tokens per”“20 tokens” 均为 0 处。
它是由 Chinchilla 本身（1.4T/70B = 20）与表 3 各行（20.0–22.4）得出的经验法则（§4.3）。

### 1.8 Besiroglu et al. 2024 对方法 3 的复现

**D40** 从论文图中重建数据后重拟合方法 3，结论：原估计与方法 1、2 不一致、拟合不好、置信区间窄得不合理——
“intervals this narrow would require over 600,000 experiments”（摘要节选）。
- 重拟合（表 1）：$A=482.01$、$B=2085.43$、$E=1.8172$、$\alpha=0.3478$、$\beta=0.3658$，$a=\beta/(\alpha+\beta)=0.5126$（标准误 0.02）；Hoffmann 对应列：$A,B$ 为正文 406.4、410.7，$E,\alpha,\beta$ 为 TeX 注释值 1.6934、0.3392、0.2849，
  而 $a$ 列写 0.454——这是 Hoffmann 表 2 区间 (0.454, 0.455) 的下端，**不是**由 TeX 值算出的 $0.2849/(0.3392+0.2849)=0.4565$。
- 按 Hoffmann 的方法 3 参数，最优策略约 **70 tokens/参数**，与 Chinchilla 实际的 20 不符；重拟合给出约 20（点估计 25.6）——“consistent with the 20 tokens per parameter rule of thumb”（§ 附表后 “Optimal scaling”）。
- 原因之一：正文参数四舍五入（尤其 $\beta$ 0.2849→0.28）带来偏差，$D=10^{11}$ 时有限数据项约偏大 13%。
- 另一原因（二手）：v2 称 Hoffmann 的一位主要作者（Borgeaud 2024）澄清，L-BFGS-B 因对 Huber 损失**取平均而非求和**导致损失尺度问题、优化过早停止，
  自助法时参数几乎不离开初值，所以区间极窄。——这一澄清本轮只见于 Besiroglu 的转述（§8）。

---

## 2. 公式（原文转录 + 课程记号对照）

**记号对照**（style-guide §2）：$N$ 参数量、$D$ 训练 token 数、$C$ 训练算力（FLOPs）、$\eta$ 学习率、$B$ batch（说明单位）、$\theta$ 参数、$\mathcal L$ 损失、$T$ 序列长度、$V$ 词表大小。
Kaplan 的 $N$ 是**非嵌入参数**，Chinchilla 的 $N$ **含嵌入**（附录 F 首句），两者不能直接比较常数。AdamW 论文的学习率写 $\alpha$、调度乘子 $\eta_t$，课程统一写 $\eta_t=\eta_{max}\cdot s(t)$，在转录处注明。

**F1 自回归分解（GPT-2 式 (1)，如印刷，含笔误）**
$$p(x)=\prod_{i=1}^{n}p(s_n\mid s_1,\dots,s_{n-1})\tag{1}$$
课程记号：$p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$。

**F2 GPT-1 目标（式 (1)(3)(4)(5)）**
$$L_1(\mathcal U)=\sum_i\log P(u_i\mid u_{i-k},\dots,u_{i-1};\Theta)\tag{1}$$
$$P(y\mid x^1,\dots,x^m)=\mathrm{softmax}(h_l^mW_y)\tag{3}\qquad L_2(\mathcal C)=\sum_{(x,y)}\log P(y\mid x^1,\dots,x^m)\tag{4}$$
$$L_3(\mathcal C)=L_2(\mathcal C)+\lambda*L_1(\mathcal C)\tag{5}$$

**F3 最大似然 ⇔ 交叉熵 ⇔ 困惑度（课程推导，非论文原式）**
$$\mathcal L(\theta)=-\frac1T\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}),\qquad
\argmax_\theta\prod_t p_\theta(x_t\mid x_{<t})=\argmin_\theta\mathcal L(\theta),\qquad \mathrm{PPL}=e^{\mathcal L}.$$
对 logits $\mathbf z_t$：$\partial\mathcal L_t/\partial\mathbf z_t=\mathbf p_t-\mathbf e_{x_t}$（数学课 M06）。以 2 为底时单位为 bits。

**F4 Vaswani 学习率（式 (3)）**
$$lrate=d_{\text{model}}^{-0.5}\cdot\min\big(step\_num^{-0.5},\;step\_num\cdot warmup\_steps^{-1.5}\big)\tag{3}$$
峰值出现在 $step=warmup$，值为 $d^{-0.5}\,warmup^{-0.5}$。

**F5 AdamW（Loshchilov & Hutter v3 算法 2；式 (1) 为 Hanson & Pratt 的原始 weight decay）**
$$\theta_{t+1}=(1-\lambda)\theta_t-\alpha\nabla f_t(\theta_t)\tag{1}$$
算法 2（第 6–12 行；原文用底色区分两种变体：第 6 行的 $+\lambda\theta_{t-1}$ 属于“Adam + L2”，第 12 行的 $+\lambda\theta_{t-1}$ 属于 AdamW，
LaTeXML 渲染中后者为 `\pagecolor{SpringGreen}`；两种变体各只用其中一项）：
$$\begin{aligned}
&g_t\leftarrow\nabla f_t(\theta_{t-1})\;\color{purple}{+\lambda\theta_{t-1}}\\
&m_t\leftarrow\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t\leftarrow\beta_2v_{t-1}+(1-\beta_2)g_t^2\\
&\hat m_t\leftarrow m_t/(1-\beta_1^t),\quad \hat v_t\leftarrow v_t/(1-\beta_2^t),\quad \eta_t\leftarrow\mathrm{SetScheduleMultiplier}(t)\\
&\theta_t\leftarrow\theta_{t-1}-\eta_t\Big(\alpha\hat m_t/(\sqrt{\hat v_t}+\epsilon)\;\color{green}{+\lambda\theta_{t-1}}\Big)
\end{aligned}$$
PyTorch（`adam.py`）：先 `param.mul_(1 - lr*weight_decay)` 再做 Adam 更新 → 课程写法 $\theta\leftarrow(1-\eta_t\lambda_{pt})\theta-\eta_t\,\hat m/(\sqrt{\hat v}+\epsilon)$，
其中 $\eta_t$ 是含调度的**实际学习率**；对应关系 $\lambda_{paper}\cdot(\eta_t)_{paper}=\text{lr}_t\cdot\lambda_{pt}$。

**F6 $6ND$（Kaplan §2.1 + GPT-3 表 D.1 的逻辑，课程整理）**
$$C_{train}\approx\underbrace{2N}_{\text{前向}}\cdot D+\underbrace{4N}_{\text{反向}}\cdot D=6ND.$$
反向 $4N$：每个线性层 $Y=XW$ 反向要算 $\partial\mathcal L/\partial X=(\partial\mathcal L/\partial Y)W^\top$ 与 $\partial\mathcal L/\partial W=X^\top(\partial\mathcal L/\partial Y)$ 两个同规模矩阵乘（GPT-3 表 D.1 标题的说法）。
带注意力（PaLM 附录 B）：每 token $6N+12LHQT$（$Q$ = 每头维度，$T$ = 序列长度）；DeepSeek LLM：$C=MD$，$M=72n_{layer}d^2+12n_{layer}d\,l_{seq}$。

**F7 Kaplan 单变量幂律（式 (1.1)–(1.3)）**
$$L(N)=\left(N_c/N\right)^{\alpha_N};\quad\alpha_N\sim0.076,\;N_c\sim8.8\times10^{13}\text{ (non-embedding parameters)}\tag{1.1}$$
$$L(D)=\left(D_c/D\right)^{\alpha_D};\quad\alpha_D\sim0.095,\;D_c\sim5.4\times10^{13}\text{ (tokens)}\tag{1.2}$$
$$L(C_{min})=\left(C_c^{min}/C_{min}\right)^{\alpha_C^{min}};\quad\alpha_C^{min}\sim0.050,\;C_c^{min}\sim3.1\times10^{8}\text{ (PF-days)}\tag{1.3}$$
$$B_{crit}(L)=\frac{B_*}{L^{1/\alpha_B}},\quad B_*\sim2\cdot10^8\text{ tokens},\;\alpha_B\sim0.21\tag{1.4}$$
损失单位 nats/token；$C$ 单位 PF-days（$=8.64\times10^{19}$ FLOPs）。

**F8 Kaplan 联合式与学习曲线（式 (1.5)(1.6)）**
$$L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}}+\frac{D_c}{D}\right]^{\alpha_D}\tag{1.5}$$
$$L(N,S)=\left(\frac{N_c}{N}\right)^{\alpha_N}+\left(\frac{S_c}{S_{min}(S)}\right)^{\alpha_S},\quad S_c\approx2.1\times10^3,\;\alpha_S\approx0.76\tag{1.6}$$
脚注 3 提到的对称替代式 $L=[(N_c/N)^{\alpha_N}+(D_c/D)^{\alpha_D}]^\beta$ 被弃用，因为没有 $1/D$ 的整数幂展开。

**F9 Kaplan 算力分配（式 (1.7)(1.8)(6.1)–(6.8)）**
$$N\propto C^{\alpha_C^{min}/\alpha_N},\quad B\propto C^{\alpha_C^{min}/\alpha_B},\quad S\propto C^{\alpha_C^{min}/\alpha_S},\quad D=B\cdot S\tag{1.7}$$
$$\alpha_C^{min}=1/\left(1/\alpha_S+1/\alpha_B+1/\alpha_N\right)\tag{1.8}$$
$$N(C_{min})\propto(C_{min})^{0.73}\tag{6.1}\qquad S_{min}\propto(C_{min})^{0.03}\tag{6.2}$$
$$\alpha_C^{min}\equiv\frac{1}{1/\alpha_S+1/\alpha_B+1/\alpha_N}\approx0.054\tag{6.4}\qquad N(C_{min})\propto(C_{min})^{\alpha_C^{min}/\alpha_N}\approx(C_{min})^{0.71}\tag{6.5}$$
$$D\propto N^{0.74}\propto C_{min}^{0.54}\tag{6.6}\qquad D(C_{min})=\frac{2C_{min}}{6N(C_{min})}\approx(4\times10^{10}\text{ tokens})(C_{min}/\text{PF-Day})^{0.26}\tag{6.7}$$
$$C^*\sim10^4\text{ PF-Days},\;N^*\sim10^{12}\text{ parameters},\;D^*\sim10^{12}\text{ tokens},\;L^*\sim1.7\text{ nats/token}\tag{6.8}$$
定义：$C_{min}\equiv6NB_{crit}S$（§6.1）。表 6（算力最优，$C_{min}$ 以 PF-days 计）：
$N_{opt}=N_e\cdot C_{min}^{p_N}$，$p_N=0.73$，$N_e=1.3\cdot10^9$；$B\ll B_{crit}=B_eC_{min}^{p_B}$，$p_B=0.24$，$B_e=2.0\cdot10^6$；
$S_{min}=S_eC_{min}^{p_S}$（下界），$p_S=0.03$，$S_e=5.4\cdot10^3$；$D_{opt}=D_eC_{min}^{p_D}$（1 epoch），$p_D=0.27$，$D_e=2\cdot10^{10}$。
课程记号：$N_{opt}(C)\propto C^{a}$，Kaplan $a=0.73$；Chinchilla $a\approx0.5$。

**F10 Chinchilla（式 (1)–(5)，arXiv v1）**
$$N_{opt}(C),D_{opt}(C)=\argmin_{N,D\text{ s.t. }\mathrm{FLOPs}(N,D)=C}L(N,D)\tag{1}$$
$$\hat L(N,D)\triangleq E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}\tag{2}$$
$$\min_{A,B,E,\alpha,\beta}\sum_{\text{Runs }i}\mathrm{Huber}_\delta\Big(\log\hat L(N_i,D_i)-\log L_i\Big)\tag{3}$$
$$N_{opt}(C)=G\left(\frac C6\right)^a,\;D_{opt}(C)=G^{-1}\left(\frac C6\right)^b,\;G=\left(\frac{\alpha A}{\beta B}\right)^{\frac1{\alpha+\beta}},\;a=\frac{\beta}{\alpha+\beta},\;b=\frac{\alpha}{\alpha+\beta}\tag{4}$$
附录 D.2 实际求解（LSE 形式）：
$$\min_{a,b,e,\alpha,\beta}\sum_{\text{Run }i}\mathrm{Huber}_\delta\Big(\mathrm{LSE}\big(a-\alpha\log N_i,\;b-\beta\log D_i,\;e\big)-\log L_i\Big),\quad A,B,E=e^a,e^b,e^e$$
（这里的 $a,b$ 是对数常数，与式 (4) 的指数 $a,b$ 同名——论文如此，课文要换字母。）
式 (4) 由在约束 $6ND=C$ 下对式 (2) 求极小得到：令 $D=C/(6N)$，$\partial\hat L/\partial N=0\Rightarrow\alpha AN^{-\alpha}=\beta BD^{-\beta}$，解出即得（第 2 课可逐行推导，数学课 M07）。
注意式 (3) 正文写 “balance the two terms in Equation (3)”，指的应是式 (2)（§7-F16）。

---

## 3. 规格与超参数表

### 3.1 GPT-1 / GPT-2 / GPT-3 对比（全部来自各报告；“—”= 报告未写）

| 项 | GPT-1（2018） | GPT-2 xl（2019） | GPT-3 175B（2020） |
|---|---|---|---|
| 层 / $d$ / 头 | 12 / 768 / 12 | 48 / 1600 / 25（头数仅代码） | 96 / 12288 / 96×128 |
| $d_{ff}$ | 3072 | 6400（代码 `nx*4`） | $4d$ = 49152 |
| 上下文 | 512 | 1024 | 2048 |
| 词表 | BPE 40,000 merges（HF 40478） | 50,257（字节级） | 未写（沿用 GPT-2） |
| LN | “used extensively” | Pre-LN + 最终 LN | pre-normalization（同 GPT-2） |
| 位置 | learned | learned（`wpe`） | 未另说明（同 GPT-2） |
| 注意力 | 稠密 | 稠密 | 稠密与局部带状稀疏交替 |
| 参数 | 未写（复算 116.5M） | 1542M（印刷）/ 1558M（仓库）/ 1,557,611,200（复算） | 175.0B（表 2.1）/ 174,600M（表 D.1） |
| 数据 | BooksCorpus（7,000+ 本） | WebText（8M+ 文档，40GB） | CC 过滤 + WebText2 + Books1/2 + Wikipedia，300B tokens |
| batch | 64 × 512 tokens = 32,768 tokens | 512 序列（× 1024 = 524,288 tokens） | 3.2M tokens |
| 学习率 | max 2.5e-4，2000 步 warmup，余弦到 0 | 手工调（未报告） | 0.6e-4，375M tokens warmup，余弦到 10%（260B） |
| 优化器 | Adam + “modified L2”（w=0.01，引 AdamW v1） | — | Adam(0.9, 0.95, 1e-8)，clip 1.0，wd 0.1 |
| 训练量 | 100 epochs | — | 300B tokens；3.14e23 FLOPs（3,640 PF-days） |
| 主张 | 预训练 + 微调，9/12 任务最优 | 零样本，7/8 LM 数据集最优 | few-shot / in-context learning |

### 3.2 GPT-3 表 2.1（v4，完整；批量单位为 tokens）

| 模型 | $n_{params}$ | $n_{layers}$ | $d_{model}$ | $n_{heads}$ | $d_{head}$ | Batch | LR | 300B/Batch（步，忽略 ramp） |
|---|---|---|---|---|---|---|---|---|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M | $6.0\times10^{-4}$ | 600,000 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | $3.0\times10^{-4}$ | 600,000 |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | $2.5\times10^{-4}$ | 600,000 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | $2.0\times10^{-4}$ | 300,000 |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | $1.6\times10^{-4}$ | 300,000 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | $1.2\times10^{-4}$ | 150,000 |
| GPT-3 13B | 13.0B | 40 | 5140（应为 5120） | 40 | 128 | 2M | $1.0\times10^{-4}$ | 150,000 |
| GPT-3 175B | 175.0B | 96 | 12288 | 96 | 128 | 3.2M | $0.6\times10^{-4}$ | 93,750 |

规律：模型越大 batch 越大、学习率越小（§2.3 引 Kaplan 与 McCandlish et al.）。

### 3.3 GPT-3 表 D.1（训练算力，如印刷）

| 模型 | PF-days | FLOPs | Params(M) | Tokens(B) | FLOPs/param/token | bwd mult | fwd FLOPs/active param/token | 激活比例 |
|---|---|---|---|---|---|---|---|---|
| GPT-3 Small | 2.60 | 2.25E+20 | 125 | 300 | 6 | 3 | 2 | 1.0 |
| GPT-3 Medium | 7.42 | 6.41E+20 | 356 | 300 | 6 | 3 | 2 | 1.0 |
| GPT-3 Large | 15.8 | 1.37E+21 | 760 | 300 | 6 | 3 | 2 | 1.0 |
| GPT-3 XL | 27.5 | 2.38E+21 | 1,320 | 300 | 6 | 3 | 2 | 1.0 |
| GPT-3 2.7B | 55.2 | 4.77E+21 | 2,650 | 300 | 6 | 3 | 2 | 1.0 |
| GPT-3 6.7B | 139 | 1.20E+22 | 6,660 | 300 | 6 | 3 | 2 | 1.0 |
| GPT-3 13B | 268 | 2.31E+22 | 12,850 | 300 | 6 | 3 | 2 | 1.0 |
| GPT-3 175B | 3,640 | 3.14E+23 | 174,600 | 300 | 6 | 3 | 2 | 1.0 |
| 对照：BERT-Large | 6.16 | 5.33E+20 | 355 | 250 | 6 | 3 | 2 | 1.0 |
| 对照：T5-11B | 382 | 3.30E+22 | 11,000 | 1,000 | 3 | 3 | 1 | 0.5 |
| 对照：RoBERTa-Large | 49.3 | 4.26E+21 | 355 | 2,000 | 6 | 3 | 2 | 1.0 |

### 3.4 GPT-3 表 2.2 数据配比（如印刷）

| 数据集 | tokens | 训练配比 | 300B 时的 epoch |
|---|---|---|---|
| Common Crawl (filtered) | 410B | 60% | 0.44 |
| WebText2 | 19B | 22% | 2.9 |
| Books1 | 12B | 8% | 1.9 |
| Books2 | 55B | 8% | 0.43 |
| Wikipedia | 3B | 3% | 3.4 |

### 3.5 Kaplan 拟合常数（表 4、5、6、2）

| 量 | 值 | 位置 |
|---|---|---|
| $\alpha_N$, $N_c$ | 0.076, $8.8\times10^{13}$ params（非嵌入） | 式 (1.1)，表 5 |
| $\alpha_D$, $D_c$ | 0.095, $5.4\times10^{13}$ tokens | 式 (1.2)，表 5 |
| $\alpha_C$, $C_c$（固定 batch 的经验趋势） | 0.057, $1.6\times10^7$ PF-days | 表 5 |
| $\alpha_C^{min}$, $C_c^{min}$ | 0.050, $3.1\times10^8$ PF-days | 式 (1.3)，表 5 |
| $\alpha_B$, $B_*$ | 0.21, $2.1\times10^8$ tokens（式 (1.4) 写 $\sim2\cdot10^8$） | 表 5 |
| $\alpha_S$, $S_c$ | 0.76, $2.1\times10^3$ steps | 式 (1.6)，表 5 |
| $L(N,D)$ 联合拟合 | $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$, $D_c=1.8\times10^{13}$ | 表 2 |
| $N_{opt}$ | $1.3\cdot10^9\,C_{min}^{0.73}$ | 表 6 |
| $B_{crit}$ | $2.0\cdot10^6\,C_{min}^{0.24}$ tokens | 表 6 |
| $S_{min}$ | $5.4\cdot10^3\,C_{min}^{0.03}$ steps | 表 6 |
| $D_{opt}$ | $2\cdot10^{10}\,C_{min}^{0.27}$ tokens | 表 6 |
| 训练设置 | Adam（>1B 用 Adafactor），$2.5\times10^5$ 步，512×1024 tokens，3000 步 warmup，余弦到 0 | §2.2 |
| 数据 | WebText2：20.3M 文档，96 GB，$1.62\times10^{10}$ 词，$2.29\times10^{10}$ tokens，测试保留 $6.6\times10^8$ | §2.3 |
| 规模范围 | 非嵌入 768 到 1.5B 参数；数据 22M 到 23B tokens；batch 多为 $2^{19}$ | §3 |

### 3.6 Chinchilla 方法 3 的三组参数

| 来源 | $E$ | $A$ | $B$ | $\alpha$ | $\beta$ | $a=\beta/(\alpha+\beta)$ |
|---|---|---|---|---|---|---|
| 论文正文（附录 D.2，四舍五入） | 1.69 | 406.4 | 410.7 | 0.34 | 0.28 | 0.4516 |
| TeX 源码注释（未四舍五入） | 1.6934 | 406.40 | 410.72 | 0.33917 | 0.28491 | 0.4565 |
| Besiroglu 2024 重拟合 | 1.8172 (0.03) | 482.01 (124.58) | 2085.43 (1293.23) | 0.3478 (0.02) | 0.3658 (0.02) | 0.5126 (0.02) |

---

## 4. 复算的数字

脚本：scratch 目录 `flops.py`、`chin.py`、`misc.py`、`gpt3.py`。

### 4.1 训练算力与步数

| 量 | 表达式 | 值 |
|---|---|---|
| 1 PF-day | `1e15*86400` | $8.64\times10^{19}$ FLOPs |
| GPT-3 175B 训练算力 | `6*174.6e9*300e9` | $3.1428\times10^{23}$ = 3,637.5 PF-days（表 D.1：3.14E+23、3.64E+03 ✓） |
| 加上注意力（Kaplan 口径 ×3 用于训练） | `(6*N + 3*2*96*2048*12288)*300e9` | $3.186\times10^{23}$，只多 1.4% |
| GPT-3 Small / Medium / XL | `6*P*300e9` | 2.250e20（2.604 PF-days）/ 6.408e20（7.417）/ 2.376e21（27.50）——与表 D.1 吻合 |
| Chinchilla 70B × 1.4T | `6*70e9*1.4e12` | $5.88\times10^{23}$ |
| Gopher 280B × 300B | `6*280e9*300e9` | $5.04\times10^{23}$（论文取 Rae et al. 的 $5.76\times10^{23}$，附录 F 自己的算法得 $6.3\times10^{23}$） |
| GPT-3 175B 步数（忽略 batch ramp） | `300e9/3.2e6` | 93,750 |
| GPT-3 warmup 步数（若按满 batch） | `375e6/3.2e6` | 117.2 步——说明 warmup 远短于 batch ramp（4–12B tokens），两者叠加时步数无法从论文精确还原 |
| GPT-3 余弦段 | `260e9/3.2e6` | 81,250 步（同上，仅量级） |
| 3.2M tokens / 2048 | | 1,562.5 序列（非整数；若为 1536 序列则 3,145,728 tokens——推测，§8） |
| GPT-1 每个 minibatch | `64*512` | 32,768 tokens |
| GPT-2 每个 batch | `512*1024` | 524,288 tokens |
| Kaplan 默认一次训练的 token 数 | `2.5e5*512*1024` | $1.31\times10^{11}$ ≈ Chinchilla 说的 “130B tokens” ✓；是 WebText2（$2.29\times10^{10}$）的 5.7 倍 |
| Adam 训练显存（仅 fp32 权重+梯度+m+v，不含激活） | `16*124439808` | 1.99 GB（GPT-2 small）；这是算术，不是论文数字 |

### 4.2 Kaplan 数值

| 量 | 表达式 | 值 |
|---|---|---|
| 参数翻倍的损失比 | `2**-0.076` | 0.9487（论文 “0.95” ✓） |
| 数据翻倍 / 算力翻倍 | `2**-0.095`；`2**-0.050` | 0.9363；0.9659 |
| $L(N)$ 于 $N$=1e3/1e5/1e7/1e9 | `(8.8e13/N)**0.076` | 6.789 / 4.784 / 3.371 / 2.376 nats |
| $L(N)$ 于 85M（GPT-2 small 非嵌入）/ 1.5B / 175B | 同上 | 2.865 / 2.304 / 1.604 |
| $L(D)$ 于 22M / 100M / 1B / 23B / 300B / 1.4T | `(5.4e13/D)**0.095` | 4.046 / 3.504 / 2.816 / 2.090 / 1.638 / 1.415 |
| $L(C_{min})$ 于 1e-9 / 1e-6 / 1e-3 / 1 / 3640 / 1e4 PF-days | `(3.1e8/C)**0.050` | 7.491 / 5.304 / 3.755 / 2.658 / 1.764 / 1.677 |
| $\alpha_N/\alpha_D$，式 (1.1)(1.2) 的值 | `0.076/0.095` | **0.800** |
| $\alpha_N/\alpha_D$，表 2 联合拟合 | `0.076/0.103` | **0.738** ≈ 论文的 0.74 → 0.74 来自表 2，不来自式 (1.2) |
| 模型 8 倍需数据几倍 | `8**0.738` | 4.64（论文 “roughly 5x”） |
| 算力 10 倍：$N$、$D$ | `10**0.73`；`10**0.27` | 5.37；1.86（图 14 “5x”“2x” ✓） |
| 算力 $10^9$ 倍：$N$、$B$、$S$、$D$ | `1e9**0.73` 等 | 3.72e6；144.5；1.86；269 |
| 式 (1.8)/(6.4) 用表 5 的四舍五入值 | `1/(1/0.76+1/0.21+1/0.076)` | **0.0520**（论文印 ≈0.054） |
| 式 (6.5) 指数 | `0.0520/0.076` | 0.684（论文印 ≈0.71，用的是 0.054/0.076=0.71） |
| 式 (1.7) 的 $N$ 指数若直接用 $\alpha_C^{min}=0.050$ | `0.050/0.076` | 0.658 |
| $B_{crit}$ 随损失：$L$=4/3/2.5/2 nats | `2.1e8/L**(1/0.21)` | 2.85e5 / 1.12e6 / 2.67e6 / 7.74e6 tokens |
| 损失降 13% 时 $B_{crit}$ 的倍数 | `(1/0.87)**(1/0.21)` | 1.94（图 10 “approximately doubles” ✓） |
| 表 6 于 $C_{min}$=3,640 PF-days（GPT-3 的量级） | `1.3e9*C**0.73`；`2e10*C**0.27` | $N_{opt}=5.17\times10^{11}$；$D_{opt}=1.83\times10^{11}$；$D/N=0.35$ |
| 同上于 1e4 PF-days | | $N_{opt}=1.08\times10^{12}$，$D_{opt}=2.41\times10^{11}$ |
| 式 (6.7) 用表 6 的 $N_e$ 复算 | `2*8.64e19/(6*1.3e9)` | $2.2\times10^{10}\,C^{0.27}$，与表 6 的 $2\times10^{10}C^{0.27}$ 一致，但**与式 (6.7) 印的 $4\times10^{10}C^{0.26}$ 差 2 倍**（§7-F5） |
| 注意：$6N_{opt}D_{opt}$ 并不等于 $C_{min}$ | `6*N*D/8.64e19` at C=3640 | 6,570 PF-days ≈ $1.8C_{min}$——因为表 6 的 $D$ 对应在临界 batch 训练（$C=2C_{min}$，§6.3） |
| 用表 2 拟合的 $L(N,D)$：85M 参数 × 22B tokens；174B × 300B | 式 (1.5) + 表 2 | 2.807；1.661 nats（仅 WebText2 分词下有意义） |

### 4.3 Chinchilla 数值

| 量 | 表达式 | 值 |
|---|---|---|
| Chinchilla 的 tokens/参数 | `1.4e12/70e9` | **20.0** |
| 表 3 各行 tokens/参数 | $D/N$ | 400M: 20.00；1B: 20.20；10B: 20.51；67B: 22.39；175B: 21.14；280B: 21.07；520B: 21.15；1T: 21.20；10T: 21.62 |
| 表 3 FLOPs 与 $6ND$ 的比 | `6*N*D/F` | 1.000、1.002、1.000、**1.047**（67B 行：1.5T 是四舍五入，精确应约 1.43T）、1.009、1.001、1.001、1.002、0.998 |
| 表 3 “Gopher 单位” | `F/5.76e23` | 400M: 算得 1/30,000（表印 1/29,968，差 0.1%）；175B: 6.684（表印 6.7 ✓）；1T: 220.5（表印 221.3，差 0.4%）——两处微差都可能来自未四舍五入的 FLOPs，未核实 |
| 正文的 175B 说法 | `6*175e9*4.2e12` | $4.41\times10^{24}$（与正文自洽，但与表 3 的 3.85e24/3.7T 不符）；tokens/参数 24.0 |
| 正文的 280B 说法 | `6*280e9*6.8e12` | $1.14\times10^{25}$（正文 “approximately $10^{25}$”） |
| 20 tokens/参数法则在 GPT-3 算力（3.14e23） | $N=\sqrt{C/120}$ | $N=51.2$B，$D=1.02$T |
| 同法则在 Gopher 算力（5.76e23） | 同上 | $N=69.3$B，$D=1.39$T → 恰是 Chinchilla |
| 方法 3（正文四舍五入参数）在 $C$=5.76e23 | 式 (4) | $N_{opt}=32.2$B，$D_{opt}=2.98$T，**$D/N=92.6$** |
| 方法 3（TeX 未四舍五入）在 5.76e23 | 式 (4) | $N_{opt}=40.4$B（图 4 标题 “40B” ✓），$D_{opt}=2.38$T，$D/N=58.9$ |
| 方法 3（Besiroglu）在 5.76e23 | 式 (4) | $N_{opt}=72.3$B，$D_{opt}=1.33$T，$D/N=18.4$ |
| 三组参数在 $C$=1e21 / 3.14e23 / 1e24 / 1e25 的 $D/N$ | 式 (4) | 四舍五入：50.1 / 87.4 / 97.7 / 122.1；TeX：33.9 / 55.9 / 61.8 / 75.5；Besiroglu：21.6 / 18.7 / 18.1 / 17.1 |
| $\hat L$(Chinchilla 70B,1.4T) vs $\hat L$(Gopher 280B,300B) vs $\hat L$(GPT-3 175B,300B) | 式 (2) | 四舍五入：1.9366 / 1.9933 / 2.0023；TeX：1.9208 / 1.9672 / 1.9764；Besiroglu：1.9739 / 1.9996 / 2.0085——三组参数都预测 Chinchilla 更低 |
| Kaplan 与 Chinchilla 在 $10^{21}$ FLOPs（附录 D.4 图 A4） | `1e21/(6*N)` | 方法 1 推荐 2.80B → 59.5B tokens；Kaplan 推荐 4.74B → 35.2B tokens |
| 算力 10 倍时 $N$ 的倍数 | `10**0.5` vs `10**0.73` | Chinchilla 3.16 vs Kaplan 5.37 |
| 表 1 各模型 tokens/参数 | $D/N$ | LaMDA 1.23；GPT-3 1.71；Jurassic 1.69；Gopher 1.07；MT-NLG 0.51；Chinchilla 20.0 |
| Besiroglu：$\beta$ 四舍五入造成的偏差 | `(1e11)**0.0049 - 1` | 0.132 ≈ 13% ✓ |

### 4.4 数据与损失

| 量 | 表达式 | 值 |
|---|---|---|
| GPT-3 表 2.2：权重×300B÷tokens | `w*300/q` | CC 0.439（印 0.44 ✓）；WebText2 **3.474（印 2.9 ✗）**；Books1 2.000（印 1.9）；Books2 0.436（印 0.43 ✓）；Wikipedia **3.000（印 3.4 ✗）** |
| 由 epoch 列反推的权重 | `e*q/300` | 60.1% / 18.4% / 7.6% / 7.9% / 3.4%，合计 97.4%；印刷权重合计 101% |
| GPT-3 Pareto 过滤的保留概率 | $P(X>1-s)=(2-s)^{-9}$（numpy `pareto` 为 Lomax 分布；200 万次模拟核对） | $s$=1: 1.000；0.9: 0.424；0.5: 0.0260；0.1: 0.00310；0: 0.00195 |
| 交叉熵示例：4 个 token 概率 [0.5, 0.25, 0.1, 0.8] | `-mean(log p)` | 1.1513 nats → PPL 3.162（= 1.661 bits） |
| GPT-1 PPL 18.4 对应的损失 | `log(18.4)` | 2.912 nats/token |
| 均匀分布在 50,257 词表上的损失 | `log(50257)` | 10.825 nats（未训练模型的初始损失量级） |
| 损失 ↔ 困惑度 | `exp(L)` | 1→2.72；2→7.39；3→20.09；3.3→27.11 |

### 4.5 学习率曲线（绘图数据）

| 调度 | 点 | 值 |
|---|---|---|
| Vaswani 式 (3)，$d=512$，warmup 4000 | step 1 / 1000 / 4000 / 10000 / 50000 / 100000 / 300000 | 1.75e-7 / 1.75e-4 / **6.99e-4（峰值）** / 4.42e-4 / 1.98e-4 / 1.40e-4 / 8.07e-5 |
| GPT-3 175B（峰值 0.6e-4，warmup 375M tokens，余弦到 10% 于 260B，之后恒定；按 token 计，忽略 batch ramp） | 0 / 0.1875B / 0.375B / 50B / 100B / 130B / 200B / 260B / 300B | 0 / 3.0e-5 / 6.0e-5 / 5.53e-5 / 4.26e-5 / 3.31e-5 / 1.28e-5 / 6.0e-6 / 6.0e-6 |

余弦段公式（笔者按“余弦衰减到 10%”实现）：$\eta(\tau)=\eta_{max}\big[0.1+0.9\cdot\tfrac12(1+\cos\pi\tau)\big]$，$\tau=(D-0.375\text{B})/(260\text{B}-0.375\text{B})$。
论文没说余弦段是否从 warmup 结束处起算，也没给公式——绘图时标“示意，按附录 B 文字实现”。

---

## 5. 图表清单与重绘建议

### 5.1 论文原图清单

| 来源 | 图号 | 内容 | 建议 |
|---|---|---|---|
| Kaplan | 图 1 | 损失对算力（PF-days，非嵌入）/ 数据（tokens）/ 参数（非嵌入）三联幂律 | **由式 (1.1)–(1.3) 重新计算绘制**（图内拟合常数未能读出，§8） |
| Kaplan | 图 2 | $10^3$–$10^9$ 参数模型的学习曲线，大模型更省样本 | 不重绘，可文字引用 |
| Kaplan | 图 3 | 算力增加 10 亿倍时，模型/batch/步数各分得多少 | 用 §4.2 “1e9 倍”数值重画成堆叠对数条 |
| Kaplan | 图 4 | 左：$L(N,D)$ 早停损失；右：$L(N,S)$ 学习曲线拟合 | 左图可用表 2 常数计算重绘 |
| Kaplan | 图 5、6 | 形状无关性；含/不含嵌入参数的对比 | 文字引用 |
| Kaplan | 图 10 | 临界 batch 随损失的幂律（每降 13% 翻倍） | 可按式 (1.4) 计算重绘 |
| Kaplan | 图 13、14 | $L(C_{min})$；最优 $N$、步数随 $C_{min}$ | 图 14 左按表 6 公式重绘 |
| Kaplan | 图 15 | $L(C_{min})$ 与 $L(D(C_{min}))$ 的交点（矛盾） | 可按 §6.3 的两条幂律示意重绘 |
| GPT-3 | 图 2.2 | 各模型总训练算力（PF-days） | 用表 D.1 数值重绘 |
| GPT-3 | 图 3.1 | 验证损失对算力的幂律（多延伸两个量级） | 不重绘（拟合常数未核实） |
| GPT-3 | 图 2.1 / 1.1 | zero/one/few-shot 对比；元学习示意 | 可重画示意 |
| Chinchilla | 图 1 | 三种方法 + Kaplan 的最优 $N$–FLOPs 叠加 | **重绘**：方法 1 用表 3 点，Kaplan 用表 6 公式（注意单位换算与 $N$ 定义差异） |
| Chinchilla | 图 2 | 方法 1：训练曲线包络 | 示意 |
| Chinchilla | 图 3 | 方法 2：IsoFLOP 谷底 | **用式 (2) 计算“示意 IsoFLOP 曲线”**，注明不是论文数据 |
| Chinchilla | 图 4 | 方法 3：等损失等高线 + 高效前沿 | **重绘**（按 §3.6 参数计算） |
| Chinchilla | 图 A4 | $10^{21}$ FLOPs 下 2.80B vs 4.74B 的正面对比 | 文字引用 |
| Besiroglu | 图 5 | 最优 tokens/参数随算力（两组参数 + 80% 区间） | 按 §4.3 三组参数重算重绘（不画区间） |
| Xiong | 图 2 | 有/无 warm-up 的 BLEU 与验证损失 | 文字引用 8.45 vs ~34 |

### 5.2 本课建议绘制的 13 张图（给出全部数值）

1. **图 2.1 自回归分解与 teacher forcing。** 一条 8-token 序列（可用第 1 课的 GPT-2 分词结果，如 `The|Ġunbelievable|Ġtoken|ization`），上方 8 个预测框，每个只看左边；标出 8 个损失 $-\log p(x_t|x_{<t})$ 一次前向同时得到。
2. **图 2.2 交叉熵、困惑度与 bits。** 横轴损失 0–11 nats，纵轴 PPL（对数），标注点：均匀分布 10.825（PPL 50,257）、GPT-1 的 2.91（PPL 18.4）、示例 1.151（PPL 3.16）。
3. **图 2.3 GPT-3 的数据管线。** CC 45TB → 分类器 + Pareto($\alpha=9$) 重采样 → 570GB（≈400B tokens）→ MinHashLSH 去重（−10%）→ 与 WebText2/Books1/Books2/Wikipedia 混合（表 2.2 配比）→ 13-gram 去污染（有 bug）。右侧小图：保留概率 $(2-s)^{-9}$ 曲线（§4.4 数值）。
4. **图 2.4 GPT-3 数据配比。** 表 2.2 的两列并排：采样权重（60/22/8/8/3）vs 数据量占比（410/19/12/55/3 共 499B → 82.2%/3.8%/2.4%/11.0%/0.6%），并标每个数据集的 epoch（0.44/2.9/1.9/0.43/3.4）。
5. **图 2.5 学习率调度三联。** (a) Vaswani 式 (3)（§4.5 数值）；(b) GPT-3 175B（按 token，§4.5 数值）；(c) GPT-1 示意（2000 步线性升温 + 余弦到 0，总步数未知，横轴用相对进度）。
6. **图 2.6 $6ND$ 的来历。** 一个线性层 $Y=XW$：前向 1 次 matmul（$2N$/token），反向 2 次 matmul（$\partial X$、$\partial W$，$4N$/token）；右侧饼图：GPT-3 175B 每 token 训练 FLOPs 中权重 $6N=1047.6$ GFLOP vs 注意力 $3\times2LTd=14.5$ GFLOP（1.4%）。
7. **图 2.7 GPT-1/2/3 对比表图。** 直接用 §3.1 表，按“结构 / 数据 / 优化 / 主张”四组着色。
8. **图 2.8 GPT-3 八个规模。** 双轴散点：横轴参数（125M–175B，对数），纵轴学习率（6e-4 → 0.6e-4）与 batch（0.5M → 3.2M tokens），数值取表 2.1。
9. **图 2.9 Kaplan 三条幂律。** 三个对数坐标面板：$L(N)$，$N\in[10^3,10^9]$；$L(D)$，$D\in[2.2\times10^7,2.3\times10^{10}]$；$L(C_{min})$，$C\in[10^{-9},10^{1}]$ PF-days；曲线由式 (1.1)–(1.3) 计算（§4.2 给了端点值），标注“常数依赖分词，无绝对意义”。
10. **图 2.10 Kaplan vs Chinchilla：最优模型规模随算力。** 横轴 FLOPs $10^{18}$–$10^{26}$；Chinchilla 方法 1 用表 3 九个点（FLOPs, 参数）；Kaplan 线 $N=1.3\times10^9(C/8.64\times10^{19})^{0.73}$（注意：Kaplan 的 $N$ 为非嵌入、$C$ 为 $C_{min}$，只能作量级比较，图注写明）；
    标注 GPT-3（3.14e23, 175B）、Gopher（5.76e23, 280B）、Chinchilla（5.88e23, 70B）、MT-NLG（$6\times530\text{e}9\times270\text{e}9=8.59\times10^{23}$, 530B）。
11. **图 2.11 tokens/参数柱状图。** 表 1 六个模型：1.23、1.71、1.69、1.07、0.51、20.0（对数纵轴），加一条 “20” 参考线（标注“经验法则，非论文原话”）。
12. **图 2.12 Chinchilla 参数化损失的等高线与前沿。** 用 TeX 未四舍五入参数：$N\in[10^8,10^{12}]$，$D\in[10^9,10^{13}]$，等高线 $\hat L$；叠加 IsoFLOP 线 $C=6ND$ 取 $10^{19},10^{21},10^{23},5.76\times10^{23},10^{25}$；前沿 $N_{opt}=G(C/6)^{0.4565}$，$G=1.300$。
    旁边 IsoFLOP 切片：$C=10^{21}$ 时 $\hat L(N,C/6N)$ 对 $N\in[10^8,10^{11}]$，谷底在 $N_{opt}=2.22$B（TeX 参数）。
13. **图 2.13 方法 3 的 tokens/参数“三种说法”。** 横轴 $C\in[10^{20},10^{26}]$，三条曲线 $D_{opt}/N_{opt}$：四舍五入参数（1e21 时 50.1 → 1e25 时 122.1）、TeX 参数（33.9 → 75.5）、Besiroglu（21.6 → 17.1）；水平参考线 20。

---

## 6. 算例与练习素材（附答案）

1. **（推导）MLE = 最小化交叉熵。** 对数似然 $\sum_t\log p_\theta(x_t|x_{<t})$ 取负并除以 $T$ 即平均交叉熵；单调变换不改变最优点。追问：困惑度 $e^{\mathcal L}$ 的含义（等效的“均匀选项数”）。
2. **（计算）交叉熵与困惑度。** 4 个 token 的模型概率 0.5、0.25、0.1、0.8。答：$\mathcal L=1.1513$ nats，PPL 3.162，1.661 bits。
3. **（计算）GPT-3 训练算力。** 答：$6\times174.6\text{e}9\times300\text{e}9=3.14\times10^{23}$ FLOPs $=3{,}637$ PF-days；加注意力（Kaplan 口径）只多 1.4%。
4. **（概念 + 计算）$6ND$ 为什么是 6？** 答：前向每参数每 token 1 乘 1 加 = 2；反向算对激活与对权重的梯度两次 matmul = 4；合计 6（GPT-3 表 D.1 标题）。T5 为什么是 3？（编码器–解码器每个 token 只激活一半参数。）
5. **（计算）Kaplan：参数翻倍损失降多少？** 答：$2^{-0.076}=0.949$，即 5.1%。数据翻倍：6.4%；算力翻倍：3.4%。
6. **（计算）Kaplan 的分配法则。** 算力 ×10 时 $N$、$D$ 各乘多少？答：$10^{0.73}=5.37$、$10^{0.27}=1.86$。Chinchilla：各 3.16。
7. **（计算）20 tokens/参数法则。** 给 $C=3.14\times10^{23}$（GPT-3 的算力），求 $N,D$。答：$6N\cdot20N=C\Rightarrow N=\sqrt{C/120}=51.2$B，$D=1.02$T。对照 GPT-3 的实际选择 175B/300B。
8. **（推导）方法 3 的前沿。** 从 $\hat L=E+AN^{-\alpha}+BD^{-\beta}$ 与 $6ND=C$ 推出式 (4)。答要点：代入 $D=C/6N$，求导得 $\alpha AN^{-\alpha}=\beta BD^{-\beta}$，解出 $N_{opt}=G(C/6)^{\beta/(\alpha+\beta)}$，$G=(\alpha A/\beta B)^{1/(\alpha+\beta)}$。
9. **（计算 + 批判）“20 tokens/参数”从哪来？** 用正文参数（1.69, 406.4, 410.7, 0.34, 0.28）算 Gopher 算力下的 $D/N$。答：92.6，而不是 20；用 TeX 未四舍五入参数为 58.9（$N=40.4$B，与图 4 标题 “40B” 一致）；Chinchilla 实际选的 20 与方法 1、2 一致。结论：四舍五入 + 拟合问题（Besiroglu 2024）。
10. **（计算）学习率。** Vaswani 式 (3) 在 $d=512$、warmup 4000 时的峰值学习率？答：$512^{-0.5}\times4000^{-0.5}=6.99\times10^{-4}$，出现在第 4000 步；第 100,000 步为 $1.40\times10^{-4}$。
11. **（计算）GPT-3 的步数。** 175B：$300\text{B}/3.2\text{M}=93{,}750$ 步（忽略 batch ramp）；warmup 375M tokens 若按满 batch 只有 117 步——说明前期 batch 递增阶段不能按满 batch 计。
12. **（批判）GPT-3 表 2.2 能自洽吗？** 用 “权重 × 300B ÷ 数据量” 复算 epoch。答：WebText2 得 3.47（印 2.9），Wikipedia 3.0（印 3.4）；权重列合计 101%。结论：表中至少有一列是四舍五入/另有口径，引用时两列都照抄并加注。
13. **（计算）Pareto 过滤。** 分类器给某文档打 0.5 分，被保留的概率？答：$(2-0.5)^{-9}=2.6\%$；打 0.9 分：42.4%。
14. **（概念）AdamW 与 L2 的区别。** 答要点：L2 梯度 $\lambda\theta$ 会被 $1/\sqrt{\hat v}$ 缩放，梯度大的权重被正则得少；解耦 weight decay 对所有权重以同一比例收缩。追问：PyTorch `AdamW(lr=6e-5, weight_decay=0.1)` 每步把权重乘多少？答：$1-6\times10^{-6}$（在峰值学习率时）。

---

## 7. 勘误与常见误解

- **F1 “Chinchilla 论文说每个参数 20 个 token。”** 全文（含 TeX 源码）没有这句话。论文说的是“模型大小翻倍、token 也翻倍”（$a\approx b\approx0.5$），20 来自 1.4T/70B 与表 3 的 20.0–22.4。
- **F2 “Chinchilla 的参数化拟合 $E=1.69,\ldots$ 推出 20 tokens/参数。”** 相反：用正文四舍五入值在 Gopher 算力下得 ~93，用 TeX 原值 ~59；Besiroglu 称约 70（其图 5）。只有方法 1、2 与 Besiroglu 的重拟合给出 ~20。
- **F3 方法 3 的置信区间自相矛盾。** 表 2 写 $a=0.46\,(0.454,0.455)$：点估计 0.46（TeX 值 0.4565）落在自己的 80% 区间外；Besiroglu 估计区间应宽约 50 倍。
- **F4 “$D\propto N^{0.74}$ 来自式 (1.1)(1.2)。”** 这句话**是 Kaplan 自己写的**：§1.2 原文 “Equation (1.1) and (1.2) together suggest … $D\propto N^{\alpha_N/\alpha_D}\sim N^{0.74}$”。
  但按式 (1.1)(1.2) 的常数，0.076/0.095 = 0.80；0.74 只能由表 2 联合拟合 0.076/0.103 = 0.738 得到。论文在同一句里把式 (1.1)(1.2) 的记号与表 2 的数值拼在了一起。
  课文写 0.74 时注明“来自表 2 的联合拟合”，不要写“由式 (1.1)(1.2) 得出”。
- **F5 Kaplan 式 (6.7) 与表 6 相差 2 倍。** 按表 6 的 $N_e=1.3\times10^9$ 复算 $2C_{min}/6N=2.2\times10^{10}C^{0.27}$，与表 6 的 $D_e=2\times10^{10}$ 一致，而式 (6.7) 印 $4\times10^{10}C^{0.26}$。原因论文未说明（§8）。
  同类小差异：式 (6.4) 印 ≈0.054，用表 5 的四舍五入值复算为 0.052；$B_*$ 式 (1.4) 印 $2\cdot10^8$、表 5 印 $2.1\times10^8$。
- **F6 Chinchilla 表 3 与正文不一致（arXiv v1）。** 表：175B → 3.85e24 FLOPs、3.7T tokens；正文：4.41e24、“over 4.2 trillion”。280B：表 9.90e24、5.9T；正文 ~1e25、6.8T。NeurIPS 版删掉了这句正文。
- **F7 “Kaplan 主张把模型训练到收敛。”** 恰好相反：算力最优是训练很大的模型并在收敛前很早停止（§1.1）。
- **F8 “Kaplan 的 $N$ 与 Chinchilla 的 $N$ 是一回事。”** Kaplan 排除嵌入参数（并说这样幂律更干净），Chinchilla 计入嵌入（附录 F 首句）；两者的 $C$ 也不同（Kaplan 的 $C_{min}$ 以 PF-days 计、不含嵌入与注意力）。把两家常数混用会出错。
- **F9 GPT-3 表 2.2 的 epoch 列与权重列不能互相复算**（WebText2 3.47 vs 2.9；Wikipedia 3.0 vs 3.4；权重合计 101%）。
- **F10 “weight decay 0.1”不是唯一定义的数。** AdamW 论文：每步衰减 $\eta_t\lambda$（与基础学习率 $\alpha$ 解耦）；PyTorch：每步衰减 $\text{lr}\cdot\text{wd}$。GPT-3 的实现未公开，0.1 是哪种约定不知道。
- **F11 “GPT-1 用的是 L2 正则。”** GPT-1 写 “modified version of L2 regularization proposed in [37]”，[37] 是 AdamW 的 v1（标题 “Fixing Weight Decay Regularization in Adam”）——即解耦 weight decay，w=0.01。
- **F12 Chinchilla 的出版信息。** NeurIPS 2022 论文集网页标题为 “An empirical analysis of compute-optimal large language model training”，PDF 与 arXiv 标题为 “Training Compute-Optimal Large Language Models”。引用时用 PDF 标题并注明 NeurIPS 2022。
- **F13 “GPT-3 的 300B tokens 是一个 epoch。”** 不是：CC 只看了 0.44 遍，Wikipedia 3.4 遍（表 2.2）；另外 GPT-3 只用了过滤后 CC 的 40%（附录 C 提到）。
- **F14 “$6ND$ 包含注意力。”** Kaplan 与 GPT-3 都明确忽略了注意力（上下文相关项）；Chinchilla 附录 F 的精确计数包含注意力，与 $6ND$ 之比 0.99–1.10。
- **F15 GPT-2 式 (1) 的下标笔误**：$\prod_{i=1}^n p(s_n|s_1,\ldots,s_{n-1})$，应为 $s_i|s_{<i}$。
- **F16 Chinchilla §3.3 “balance the two terms in Equation (3)”。** 式 (3) 是 Huber 拟合目标，含 $N$、$D$ 两项的是式 (2)——引用号笔误。
- **F17 “$L^*\approx1.7$ nats 是自然语言的熵”、“$E=1.69$ 是文本熵”。** 前者是 Kaplan 明说的猜想且“highly uncertain”；后者是 Chinchilla 对第一项的解释，依赖分词与数据集（MassiveText、SentencePiece 32k），不能与 Kaplan 的 nats 直接比较。二者数值接近纯属巧合，不应并列为“两次测量”。
- **F18 “Kaplan 错在方法。”** Chinchilla 自己给的原因是：固定学习率调度（到 130B tokens）导致中间损失被高估，以及 Kaplan 的模型多数很小。它没有说 Kaplan 的幂律拟合本身错了。
- **F19 GPT-3 Table 2.1 的 13B 行 $d_{model}=5140$ 与 XL 行头宽。** 见第 1 课笔记 §4.2：5120 与表 D.1 精确吻合。
- **F20 GPT-3 的 “alternating dense and locally banded sparse attention” 常被忽略。** 很多“GPT-3 = 放大的 GPT-2”的说法不准确；但稀疏细节论文未给。

---

## 8. 未能核实 / UNVERIFIED

1. **“teacher forcing”一词的出处**（通常归于 Williams & Zipser 1989）：本轮未查原文，课文若用此归属须另行核对。
2. **GPT-3 的数值精度**（fp16 混合精度？）与**每步序列数**（3.2M tokens ÷ 2048 = 1562.5，非整数；1536 序列 = 3,145,728 tokens 是常见说法但未见原文）。
3. **GPT-3 各规模 batch ramp 的具体长度**（论文只说 4–12B tokens，“depending on the model size”）以及余弦段的起点。
4. **GPT-1 总训练步数与 BooksCorpus 的 token 数**：报告只说 100 epochs、batch 64×512。
5. **GPT-2 的学习率、训练步数、训练 token 数**：报告只说“manually tuned”。
6. **Kaplan 图 1 图内拟合式的常数**（图为矢量路径，无法抽文字、本机无渲染器）；**GPT-3 图 3.1 的拟合幂律常数**同理。
7. **Kaplan 式 (6.7) 的 $4\times10^{10}$ 与表 6 的 $2\times10^{10}$ 差 2 倍的原因**；式 (6.4) 的 0.054 用了什么未四舍五入的值。
8. **Kaplan 默认训练 $1.31\times10^{11}$ tokens 而 WebText2 只有 $2.29\times10^{10}$**：按论文数字推得约 5.7 遍，论文未讨论重复轮数；是否所有运行都跑满 $2.5\times10^5$ 步也未写。
9. **Borgeaud（2024）对 Huber 损失取平均导致早停的澄清**：只见于 Besiroglu v2 的转述，原始出处（可能是社交媒体帖子）未查。
10. **Chinchilla NeurIPS 版与 arXiv v1 的其他差异**：只核对了表 3 移位、175B 句删除、方法 3 表 2 数字相同、图 4 “40B” 相同；未逐段比对。
11. **Kaplan–Chinchilla 分歧的后续解释**（例如 NeurIPS 2024 的 “Resolving Discrepancies in Compute-Optimal Scaling of Language Models”，在检索结果中出现）：本轮未读，不得在课文中引用其结论。
12. **“16 字节/参数”的混合精度 Adam 显存规则**（ZeRO 论文）：本轮只做了 fp32 四份的算术，未核对 ZeRO 原文。
13. **GPT-3 “gradient noise scale” 的具体测量值**：论文未给。
14. **MT-NLG 的训练算力**（图 2.10 用的 $8.59\times10^{23}$ 是 $6ND$ 算术，不是其论文报告值）。
15. **Chinchilla 表 3 中 “1T → 221.3 Gopher 单位”与 $1.27\times10^{26}/5.76\times10^{23}=220.5$ 的微小差异**：可能用了未四舍五入的 FLOPs，未核实。

---

## 9. 参考文献卡片草稿

```
- id: radford2018
  title: Improving Language Understanding by Generative Pre-Training
  a: Radford, Narasimhan, Salimans, Sutskever (2018), OpenAI technical report, no arXiv
  url: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
  finding: 在 BooksCorpus 上以标准语言模型目标预训练 12 层 Transformer，再加线性头并以语言模型为辅助目标（λ=0.5）做判别式微调。
  matters: 课文“预训练 + 微调”范式的起点，也给出最早一套完整的 GPT 训练超参数（2.5e-4、2000 步 warmup、余弦到 0）。
  kind: artefact

- id: radford2019
  title: Language Models are Unsupervised Multitask Learners
  a: Radford, Wu, Child, Luan, Amodei, Sutskever (2019), OpenAI technical report, no arXiv
  url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  finding: 用 Reddit 高赞外链构建 40GB 的 WebText 并训练至 1.5B 参数，模型不经微调即在 8 个语言建模数据集中的 7 个上达到最优。
  matters: 提供课文自回归分解式 (1) 与“数据质量靠人类筛选的启发式”这一数据管线思想的出处。
  kind: artefact

- id: brown2020
  title: Language Models are Few-Shot Learners
  a: Brown, Mann, Ryder, Subbiah, Kaplan, Dhariwal, et al. (2020), NeurIPS 2020, arXiv:2005.14165 (read v4)
  url: https://arxiv.org/abs/2005.14165
  finding: 8 个规模统一训练 300B tokens，175B 模型只靠上下文示例即可完成多种任务，附录 B 与 D 给出完整优化设置与 6ND 算力表。
  matters: 课文训练超参数（β2=0.95、裁剪 1.0、余弦到 10%、batch ramp、wd 0.1）、数据配比与 6ND 逐列逻辑的来源。

- id: kaplan2020
  title: Scaling Laws for Neural Language Models
  a: Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei (2020), arXiv preprint, arXiv:2001.08361 (v1)
  url: https://arxiv.org/abs/2001.08361
  finding: 语言模型损失分别对非嵌入参数、数据和最优算力呈幂律，并得出算力最优应主要增大模型（N∝C^0.73）且远在收敛前停止。
  matters: 课文 scaling law 第一站与 C≈6N 推导的出处，其结论被 Chinchilla 修正，是“先算账”思想的起点。

- id: hoffmann2022
  title: Training Compute-Optimal Large Language Models
  a: Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai, Rutherford, et al. (2022), NeurIPS 2022, arXiv:2203.15556 (v1; TeX source and NeurIPS PDF also read)
  url: https://arxiv.org/abs/2203.15556
  finding: 三种独立方法都显示最优模型规模与训练 token 应随算力等比例增长，按此训练的 70B Chinchilla（1.4T tokens）以相同算力胜过 280B 的 Gopher。
  matters: 课文 Chinchilla 法则、参数化损失 L=E+A/N^α+B/D^β 以及“20 tokens/参数”经验法则的真正来源与边界。

- id: besiroglu2024
  title: "Chinchilla Scaling: A replication attempt"
  a: Besiroglu, Erdil, Barnett, You (2024), arXiv preprint, arXiv:2404.10102 (read v2)
  url: https://arxiv.org/abs/2404.10102
  finding: 从论文图表重建数据重拟合方法 3，发现原参数拟合不佳、区间窄得不可能，重拟合后方法 3 与方法 1、2 一致（约 20 tokens/参数）。
  matters: 课文必须说明“20”来自方法 1、2 而非正文的参数化常数，这篇论文是该勘误的依据。

- id: loshchilov2019
  title: Decoupled Weight Decay Regularization
  a: Loshchilov, Hutter (2019), ICLR 2019, arXiv:1711.05101 (read v3; v1 titled "Fixing Weight Decay Regularization in Adam")
  url: https://arxiv.org/abs/1711.05101
  finding: 证明对 Adam 而言 L2 正则与 weight decay 不等价，并提出把衰减从梯度更新中解耦的 AdamW。
  matters: GPT-1 的“modified L2”、GPT-3 的 wd 0.1、Chinchilla 改用 AdamW 都指向它，课文优化器一节的核心公式在其算法 2。

- id: vaswani2017
  title: Attention Is All You Need
  a: Vaswani et al. (2017), NIPS 2017, arXiv:1706.03762 (v7, §5.3–5.4 only)
  url: https://arxiv.org/abs/1706.03762
  finding: 用 d_model^-0.5 · min(step^-0.5, step · warmup^-1.5) 的学习率（warmup 4000 步）配合 Adam(0.9, 0.98, 1e-9) 训练 Transformer。
  matters: 课文 warmup 调度的第一个公式实例，用于和 GPT 系列的线性 warmup + 余弦对照。

- id: xiong2020
  title: On Layer Normalization in the Transformer Architecture
  a: Xiong et al. (2020), ICML 2020, arXiv:2002.04745 (v2)
  url: https://arxiv.org/abs/2002.04745
  finding: Post-LN 的末层梯度在初始化时很大所以必须 warm-up，而 Pre-LN 可以去掉 warm-up 并更快收敛。
  matters: 解释课文“为什么要 warmup”，并把 warmup 与第 1 课的 LN 位置联系起来。

- id: palm2022
  title: "PaLM: Scaling Language Modeling with Pathways"
  a: Chowdhery et al. (2022), arXiv:2204.02311 (v5, Appendix B)
  url: https://arxiv.org/abs/2204.02311
  finding: 训练 FLOPs 取 6N 加注意力 matmul 12LHQT，并据此定义与硬件无关的模型 FLOPs 利用率 MFU。
  matters: 给课文 6ND 之外的“含注意力”口径，并与 DeepSeek LLM 的 M 公式对齐。

- id: deepseekllm2024
  title: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"
  a: DeepSeek-AI (2024), arXiv preprint, arXiv:2401.02954 (v1, §3.2 only)
  url: https://arxiv.org/abs/2401.02954
  finding: 用含注意力、不含词表计算的非嵌入 FLOPs/token M 代替 6N，使 C=MD 并改进最优分配与损失预测。
  matters: 本课 6ND 与 Chinchilla 方法 2 的直接后继，第 6 课从这里接上。

- id: torchadamw
  title: PyTorch torch/optim/adam.py and adamw.py
  a: PyTorch contributors (accessed 2026-09-24), GitHub
  url: https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py
  finding: 解耦 weight decay 实现为 param.mul_(1 - lr * weight_decay)，AdamW 默认 weight_decay=1e-2。
  matters: 说明代码里的 weight decay 与 AdamW 论文的参数化差一个学习率因子，课文给数值时必须说明约定。
  kind: artefact
```

---

## 10. 核查记录（对抗式复核，2026-09-24）

复核方法：选出 20 组最可能被课文复述的论断，重新从一手来源取证：arXiv LaTeXML 全文（`arxiv.py text/grep`）、Chinchilla **TeX 源码**
（`arxiv.org/e-print/2203.15556`，`main.tex`）与 **NeurIPS 2022 论文集页面及 PDF**、GPT-1/GPT-2 OpenAI PDF（`pdftotext` + `pypdf`）、
PyTorch `adam.py`/`adamw.py` raw。所有派生数字在 Python 里重算。

### 10.1 改动

| # | 位置 | 原文 | 改为 | 依据 |
|---|---|---|---|---|
| 1 | F4 | 把“$D\propto N^{0.74}$ 来自式 (1.1)(1.2)”列为读者的误解 | 注明这句话是 Kaplan §1.2 **自己写的**（“Equation (1.1) and (1.2) together suggest … $\sim N^{0.74}$”），但数值只能由表 2 的 0.076/0.103 得到 | 2001.08361 v1 §1.2（LaTeXML 第 347 行附近） |
| 2 | D37 | “Kaplan 的多数模型小于 100M” | “Kaplan 的多数运行 significantly smaller，其中**许多**小于 100M”（对照：Chinchilla 多数模型 >500M） | 2203.15556 `main.tex` §2：“many being less than 100M parameters” |
| 3 | D40 | “Hoffmann 对应值（取自 TeX 注释）… $a=0.454$” | 说明 Besiroglu 表中 Hoffmann 的 $a=0.454$ 是 Hoffmann 表 2 区间下端，不是 TeX 值算出的 0.4565；$A,B$ 列是正文值 | 2404.10102 v2 表 1；本文件 §3.6 |
| 4 | §4.3 “Gopher 单位”行 | “400M: 1/29,968 → 3.333e-5 ✓” | 算得 1/30,000，表印 1/29,968（差 0.1%），与 1T 行一样属“微差、原因未核实” | `1.92e19/5.76e23 = 3.333e-5 = 1/30,000` |

### 10.2 复核后确认无误（节选关键证据）

1. **D39/F1**：arXiv 全文 grep “20 tokens|tokens per param|per parameter” 与 TeX 源码 grep “tokens per|per param|20 tokens” 均 0 处。✓
2. **D33 表 2**：方法 1 0.50 (0.488, 0.502)/0.50 (0.501, 0.512)，方法 2 0.49 (0.462, 0.534)/0.51 (0.483, 0.529)，方法 3 0.46 (0.454, 0.455)/0.54 (0.542, 0.543)；标题 “10th and 90th percentiles … bootstrapping”；NeurIPS PDF 数值相同。✓
3. **D33 方法细节**：4 种余弦周期、跨 16×、1500 个对数间隔 FLOP 值、选点在最后 15%；方法 2 “9 different training FLOP counts”（$6\times10^{18}$ 到 $3\times10^{21}$）；Huber $\delta=10^{-3}$、L-BFGS。✓
4. **D34**：式 (10) $E+A/N^{0.34}+B/D^{0.28}$，$E=1.69$、$A=406.4$、$B=410.7$；TeX 第 899–903 行 `0.5267228 / 6.0073404 / 0.33917084 / 6.0179186 / 0.2849083`；第 360 行图 4 标题 “40B parameters”。✓
5. **§4.3 方法 3 重算**：正文值 $N_{opt}=32.2$B、$D/N=92.6$；TeX 值 40.4B、58.9；Besiroglu 72.3B、18.4；四个算力点的 $D/N$ 与三组 $\hat L$ 全部一致。✓
6. **D35/F6**：TeX 第 406–414 行表 3 九行原样；第 421 行正文 “$4.41\times10^{24}$ … over 4.2 trillion tokens … approximately $10^{25}$ … 6.8 trillion”；NeurIPS PDF（15 页）为 “Table A3”，无该句；论文集网页标题 “An empirical analysis of compute-optimal large language model training”，PDF 标题仍为 arXiv 标题。重算 tokens/参数与 $6ND/F$（67B 行 1.047）一致。✓
7. **D36、D17**：表 4 两行（80/128/128/16,384/4e-5/3M→6M；80/64/128/8,192/1e-4/1.5M→3M）与标题；bfloat16 + float32 副本；脚注 “around 80% of the way through the cosine cycle”；SentencePiece 不做 NFKC、94.15%、32,000。✓
8. **D20**：表 A4 六个比值 1.03/1.10/1.08/1.04/1.03/0.99；$6.3\times10^{23}$ vs $5.76\times10^{23}$；“we also count embeddings matrices”。✓
9. **D31、D32、D38**：摘要 “70 million to over 16 billion … 5 to 500 billion tokens”，§1 “under 70M … 5B to over 400B”；“4 times smaller … 4 times more tokens”；MMLU 67.5%、“greater than a 7% improvement”；表 1 六行。✓
10. **D40 Besiroglu**：“over 600,000 experiments”；表 1 482.01/2085.43/1.8172/0.3478/0.3658/0.5126；“approximately 70 tokens per parameter”；“25.6 tokens per parameters”；约 50 倍宽的区间；Borgeaud (2024) 为 Twitter，经 Besiroglu 转述“averaged the Huber loss values … instead of summing”。✓
11. **F7–F9 Kaplan**：式 (1.1)–(1.8)、表 5（含 $\alpha_C=0.057$、$C_c=1.6\times10^7$、$B_*=2.1\times10^8$）、表 2 联合拟合、表 6 四行、式 (6.1)–(6.8) 的编号与数值（0.054、0.71、$N^{0.74}\propto C^{0.54}$、$4\times10^{10}C^{0.26}$、$C^*,N^*,D^*,L^*$、“order or magnitude”）。重算 0.0520、0.684、0.800、0.738、$2.2\times10^{10}$、6,572 PF-days、1.94。✓
12. **D18、D26、表 3.5**：“factor of two comes from the multiply-accumulate”、“$d_{model}\gg n_{ctx}/12$”、“$C\approx6N$”；Adam、$2.5\times10^5$ 步、512×1024、>1B 用 Adafactor、3000 步 warmup + 余弦到 0；WebText2 20.3M 文档、$1.62\times10^{10}$ 词、$2.29\times10^{10}$ tokens、测试 $6.6\times10^8$。$1.31\times10^{11}$ tokens = 5.72 × WebText2。✓
13. **GPT-3**：表 2.1 八行（含 5140）、表 2.2 五行与标题 “up to 3.4 times”、表 D.1 八行与标题逻辑、附录 D “less than 10%”；附录 B 全部超参；附录 A $\alpha=9$、MinHashLSH 10 hashes、“decreased dataset size by an average of 10%”；附录 C “we only trained on 40% of our filtered Common Crawl documents”；13-gram 与 bug；$K$ “typically fits 10 to 100”；文档拼接 “not masked in any special way”；参考文献 [68] = Loshchilov & Hutter。重算表 2.2：3.474、3.000、合计 101%。✓
14. **D14/F5 AdamW**：算法 2 第 12 行 $\theta_t\leftarrow\theta_{t-1}-\eta_t(\alpha\hat m_t/(\sqrt{\hat v_t}+\epsilon)+\lambda\theta_{t-1})$，衰减项在 `\pagecolor{SpringGreen}` 中；式 (1)；命题 1 $\lambda'=\lambda/\alpha$；第 1 行默认值；附录 B.1 $\lambda=\lambda_{norm}\sqrt{b/(BT)}$；arXiv 提交历史 v1 2017-11-14 / v2 2018-02-14 / v3 2019-01-04。✓
15. **D15 PyTorch**：`adam.py` 第 418–420 行 `if decoupled_weight_decay: … param.mul_(1 - lr * weight_decay)`；`adamw.py` 默认 `weight_decay: float = 1e-2`、`decoupled_weight_decay=True`。✓
16. **D2、D10 GPT-1**：$\lambda$ “was set to 0.5”、6.25e-5、batch 32、3 epochs、0.2% warmup；参考文献 [37] “Fixing weight decay regularization in adam”。✓
17. **D1/F1/F15 GPT-2 式 (1)**：`pypdf` 抽出 “p(x) = n∏ i=1 p(sn|s1, ..., sn−1) (1)”，笔误属实。D7 WebText（3 karma、45 million links、Dec 2017、8 million documents / 40 GB、删除 Wikipedia）与 D11（7 out of 8、underfits、manually tuned、5% held-out）。✓
18. **D16**：Vaswani 式 (3) 峰值 6.99e-4 及 §4.5 各点；Xiong IWSLT14 De-En 8.45 vs “around 34”，SGD 同样需要 warm-up，并反驳 Liu et al. (2019a) 的方差解释。✓
19. **D21**：DeepSeek LLM §3.2 $6N_1=72n_{layer}d^2$、$6N_2=\ldots+6n_{vocab}d$、$M=72n_{layer}d^2+12n_{layer}d\,l_{seq}$。✓
20. **§4.4–§4.5 其他重算**：Pareto 保留概率 $(2-s)^{-9}$、交叉熵示例 1.1513/3.162/1.661、$\ln18.4$、$\ln50257$、GPT-3 学习率曲线九个点、PyTorch 衰减示例 $1-6\times10^{-6}$、MT-NLG $8.586\times10^{23}$。✓

### 10.3 本轮新增的未核实项

- Besiroglu 说 Hoffmann 参数对应“approximately 70 tokens per parameter”，未说明在哪个算力点、用正文值还是 TeX 值；本文件用 TeX 值在 $10^{24}$–$10^{25}$ FLOPs 得 62–76，用正文值得 98–122，量级相符，但无法确认他们的计算口径。


---

## 11. 补充核实（写课时追加，2026-09-24）

写 `src/lesson-02-gpt-training.md` 时用到、而上文没有覆盖的事实与复算。引文 ≤15 词。

### 11.1 浮点格式（课文 §6 混合精度）

| 事实 | 来源与原文 | 本机复核 |
|---|---|---|
| bfloat16 = 1 位符号 + 8 位指数 + 7 位显式尾数（精度 8 位） | Wikipedia “Bfloat16 floating-point format”：“Sign bit: 1 bit, Exponent width: 8 bits, Significand precision: 8 bits (7 explicitly stored…)” | — |
| bfloat16 是 binary32 的截短，保留全部指数位 | 同上：“the exponent bits are preserved while the significand field can be reduced by truncation” | NumPy 按位模拟（取 float32 高 16 位、就近偶数舍入）：bf16 最大值 3.3895e38 ✓ |
| bfloat16 与 float32 动态范围相同；TPU 矩阵乘用 bf16、累加用 fp32 | Google Cloud TPU 文档 `docs.cloud.google.com/tpu/docs/bfloat16`：“The dynamic range of bfloat16 and float32 are equivalent”；“matrix multiplication operations with bfloat16 values and accumulations with IEEE float32 values” | — |
| FP32：23 位尾数、8 位指数、eps 2^−23 = 1.19e−7、max 3.40e38；FP16：10 位尾数、5 位指数、eps 2^−10 = 0.000977、max 65504 | `numpy.finfo(np.float32)`、`numpy.finfo(np.float16)`（NumPy 1.26.4） | ✓ |
| bf16(0.02) = 0.02001953125；0.02 附近间隔 2^−13 = 1.2207e−4，半间隔 6.1035e−5 | 按位模拟 | ✓ |
| 每步加 6e−5、共 1000 步：FP32 累加得 0.0800190；每步舍回 BF16 仍为 0.0200195 | 按位模拟 | ✓ |
| 70,000：FP16 溢出为 inf，BF16 为 70,144；1e−8：FP16 下溢为 0，BF16 为 1.0012e−8；1 + 1e−4 在 BF16 中舍回 1.0 | 按位模拟 + `np.float16` | ✓ |
| FP32 最小正规数 1.1755e−38，最小次正规数约 1.4e−45；$0.1^{2048}$ 在 FP32 连乘中为 0，$\sum\ln0.1=-4715.7$ | `numpy.finfo(np.float32)` 与直接计算 | ✓ |

说明：“16 字节/参数”（BF16 权重 2 + BF16 梯度 2 + FP32 主权重 4 + m 4 + v 4）是课文里笔者自己的算术，已在课文中标注；ZeRO 原文仍未核对（§8-12 不变）。Chinchilla 只写了前向/反向用 bf16、优化器状态里有 fp32 权重副本，**没有写梯度的格式**，课文和图 2.10 都没有断言梯度用什么格式。

### 11.2 第 1 课研究笔记中复用的事实（已在 `lesson-01-doc.md` 核对）

- GPT-2 small 精确参数量 124,439,808；GPT-1 结构复算 116,534,784（约 1.17 亿）。
- GPT-2 分词：“The unbelievable tokenization” → `The` 464、`Ġunbelievable` 24479、`Ġtoken` 11241、`ization` 1634；“Hello world” → 15496、995；id 50256 = `<|endoftext|>`。
- 初始化：GPT-1 报告 “N(0, 0.02)”；HF 配置 `initializer_range` 0.02（作标准差）。
- GPT-1 的 LN 位置报告未写，HF `modeling_openai.py` 为 Post-LN（课文表中如此标注）。
- GPT-3 的交替稠密/局部带状稀疏注意力（“similar to the Sparse Transformer”），细节未给。

### 11.3 第 6 课研究笔记中借用的结论（已在 `lesson-06-doc.md` 核对）

- DeepSeek LLM §3.3：数据质量越高，模型扩展指数 $a$ 越大、数据指数 $b$ 越小，“increased compute budget should be allocated more to the model”；并称这 “might also explain” 早先规模化研究在分配上的分歧。课文 §4、§7、§8 只转述这一定性结论，不引用具体指数。
- 最优 batch 与学习率随算力的幂律：课文 §8 只说“第 6 课会拟合”，没有引用具体常数。

### 11.4 课文中新增的复算（Python 3.11 + NumPy 1.26.4）

| 量 | 表达式 | 值 |
|---|---|---|
| 四词玩具例：$\mathbf z=(2,1,0,-1)$，真 token 为第 2 个 | softmax | $\mathbf p=(0.6439,0.2369,0.0871,0.0321)$；$\mathbf p-\mathbf y=(0.6439,-0.7631,0.0871,0.0321)$；$\ell=1.4402$ |
| 同上，沿负梯度走一步（步长 1） | $\mathbf z'=\mathbf z-(\mathbf p-\mathbf y)$ | $\mathbf p'=(0.3533,0.5308,0.0834,0.0324)$；$\ell'=0.6334$ |
| Adam（$\beta_1=0.9,\beta_2=0.95$）前三步的 $\hat m/\sqrt{\hat v}$ | 梯度恒 0.01 / 恒 100 / ±1e−3 交替 | 1,1,1 / 1,1,1 / 1.0, −0.0526, 0.3358 |
| 不做偏差修正的第一步 | $(1-\beta_1)/\sqrt{1-\beta_2}$ | 0.447（$\beta_2=0.95$）；3.162（$\beta_2=0.999$） |
| GPT-3 学习率（课文式 (2.16) 的实现） | 0.1B / 10B / 130B / 200B | 1.6e−5 / 5.98e−5 / 3.31e−5 / 1.28e−5 |
| 注意力占比 $n_{ctx}/(12d)$ | GPT-3 175B；GPT-3 XL（$d=2048$） | 1.39%；8.3% |
| GPT-3 XL 训练算力 | `6*1.32e9*300e9` | 2.376e21 = 27.5 PF-days；$10^{17}$ FLOP/s 下 6.6 小时 |
| 打包利用率示例 | 700+1500+900 token 各占一条 2048 序列 | 3100/6144 = 50.5% |
| MinHash 示例 | $\lvert A\cap B\rvert=80$，$\lvert A\cup B\rvert=100$；无关文章 2/178 | $J=0.8$；$J=0.0112$ |
| 逐前缀计算与一次前向之比 | $(T+1)/2$，$T=2048$ | 1024.5 |
| $M$ 与 $6N$（第 6 课式） | GPT-3：96 层、$d=12288$、$l_{seq}=2048$；GPT-2 small：12 层、768、1024、$V=50257$ | GPT-3：$6N_1=1.0437\times10^{12}$，$M=1.0727\times10^{12}$（+2.78%）；GPT-2 small：$6N_1=5.096\times10^8$，$M=6.229\times10^8$（+22.2%），$6N_2=7.412\times10^8$（+45.4%） |
| 20 token 法则 / 方法 3（TeX）/ Kaplan 表 6，于 $C=3.14\times10^{23}$ | (2.28)(2.27)(2.24) | 51.2B & 1.02T / 30.6B & 1.71T（D/N 55.9）/ 5.16e11 & 1.83e11 |
| $\hat L$（TeX 参数） | GPT-3 175B/300B；最优；20 法则 | 1.9764；1.9406；1.9437 |
| $C=10^{24}$ | 20 法则；方法 3（TeX） | 91.3B & 1.83T；51.9B & 3.21T（D/N 61.8） |
| 最优点两项之比（$C=10^{21}$，TeX） | $A/N^\alpha$ : $B/D^\beta$ | 0.2749 : 0.3272 = 0.840 = $\beta/\alpha$ ✓ |
| Kaplan：损失降 10% 所需参数倍数 | `0.9**(-1/0.076)` | 4.000 |
| 练习 4 | 概率 0.4, 0.2, 0.9, 0.05 | 1.4067 nats，2.029 bits，PPL 4.082 |
| 练习 5 | $\mathbf z=(1,3,0)$，真 token 第 2 个 | $\mathbf p=(0.1142,0.8438,0.0420)$，$\ell=0.1698$ |
| §2 例题：有真实不确定性的位置（教学复核时追加） | $q=(0.5,0.5)$，$p_A=(0.8,0.2)$ | $H(q,p_A)=0.9163$，$H(q)=\ln2=0.6931$，$\KL(q\|p_A)=0.2231$ |
| §3 log-sum-exp（教学复核时追加） | FP32 最大值、FP16 最大值、FP32 最小次正规数取对数；$\mathbf z=(1000,999,998)$ | $\ln(3.4028\times10^{38})=88.72$，$\ln65504=11.09$，$\ln(1.4\times10^{-45})=-103.3$；$\ell=1.4076$ |
| §5 梯度尖峰（教学复核时追加） | $\beta_1=0.9,\beta_2=0.95$，稳态 $m=v=1$、梯度恒 1；尖峰一步 $g=100$（不裁剪）或 $g=2$（按全局范数 0.5→50 裁到 1.0） | 不裁剪：比值 0.487 / +1 步 0.454 / +10 0.257 / +20 0.164 / +50 0.167 / +100 0.503，101 步合计 25.8，第 150 步才回到 ≥0.9；裁剪：1.026 / 1.020 / 0.991 / 0.986 / 0.995 / 1.000，合计 100.5；$m$ 半衰期 6.58 步、$v$ 半衰期 13.5 步 |
| §5 梯度尖峰的单位（数学复核时追加） | 全局范数 $\ge$ 任一分量，所以“梯度恒 1、全局范数 0.5”不能取字面值 | 课文改为“把这个参数平时的梯度记作 1 个单位”；忽略 $\epsilon$ 时 $\hat m/\sqrt{\hat v}$ 对整串梯度同乘常数不变，表中数字不变 |
| 练习 7(c) 比值与份额（数学复核时追加） | $n_{ctx}/(12d)$ 是注意力项 ÷ 参数项；份额 $=r/(1+r)$ | GPT-3 XL：比值 8.33%，份额 7.69%；GPT-3 175B：比值 1.389%，份额 1.370% |
| Kaplan $L(C_{\min})$ 的拟合区间（数学复核时追加；ar5iv 2001.08361） | §1.2 “These relations hold across eight orders of magnitude in C_min”；图 13 标题 “we exclude 1-layer networks in the power-law fits”（1 层网络的“鼓包”在约 $10^{-5}$ PF-days） | 论文没有给出 $C_{\min}$ 的拟合区间；图 2.11 的算力实线改为示意八个数量级 $10^{-7}$–$10^{1}$ PF-days，上端约为 Kaplan 最大配置的 $6ND=6\times1.5\times10^9\times1.31\times10^{11}=1.18\times10^{21}$ FLOPs $\approx13.7$ PF-days |

### 11.5 未采用的说法

- “exposure bias” 只作为术语名出现，未注明出处（Ranzato 等 2016 未核对）。
- Pareto 抽样“为了保留多样性”的动机：论文附录 A 的动机本轮未核对，课文只描述公式算出的效果，不陈述动机。
- Chinchilla 是否在其他数据集（C4、GitHub）上复核过分配：未核对，课文只说 $a\approx b\approx0.5$ 来自其 MassiveText 训练数据。

### 11.6 事实复核（第二轮，2026-09-24；arXiv LaTeXML grep）

| 课文中的说法 | 一手来源与原文（≤15 词） | 结论 / 课文改动 |
|---|---|---|
| Kaplan 的 $C_{\min}=C/(1+B/B_{crit})$ 是 §5.1 式 (5.5) | 2001.08361 §5.1 “Adjustment for Training at $B_{crit}(L)$”：$C_{\min}(C)\equiv C/(1+B/B_{crit}(L))$，“minimum compute, at $B\ll B_{crit}$”，式号 (5.5)；$C=6NBS$ | ✓ |
| “按临界 batch 训练时 $C=2C_{\min}$”；式 (6.7) 在 §6.3 | §6.3：“If we train at the critical batch size (i.e. $C=2C_{\min}$)”，紧接式 (6.7) | ✓；课文补一句“式 (6.7) 印 $4\times10^{10}C^{0.26}$，与表 6 差 2 倍，论文未解释”（§7-F5） |
| Kaplan 前向计数 $2N+2n_{layer}n_{ctx}d$ | 表 1：$C_{forward}=2N+2n_{layer}n_{ctx}d_{attn}$；§2.1 正文取 $d_{model}$ | ✓ |
| “过拟合程度 depends predominantly on $N^{0.74}/D$” | 图 9 标题：“The extent of overfitting depends predominantly on the ratio $N^{\alpha_N/\alpha_D}/D$”；§1.1 的写法是 “depends predictably on the ratio $N^{0.74}/D$” | ✓（引语出自图 9 标题） |
| Kaplan 的分词 | §2：WebText2 “tokenized using byte-pair encoding … $n_{vocab}=50257$”；**没有**说“GPT-2 的分词器” | 图 2.11 底注由“GPT-2 分词器”改为“它的 BPE 分词” |
| $L^\ast$ 的猜想 | §6.3：交点 “provides an estimate of the point at which Transformer language models reach maximal performance”；另一种解读 “rough estimate for the entropy-per-token” | 课文补上“性能上限”这层意思 |
| Chinchilla 的损失单位 | 2203.15556 全文 grep “nats” 与 “natural log” 均 0 处，只有表题 “bits per byte”（下游数据集） | **论文未注明损失单位**。课文 §2 改为“本课按自然对数读 Chinchilla 的损失，笔者的假设”；图 2.12 右图纵轴由“预测损失（nats）”改为“预测损失 $\hat L$” |
| Chinchilla 的 MMLU | 摘要：67.5%，“greater than a 7% improvement over Gopher”；正文 MMLU 一段与表 6：Chinchilla 5-shot 67.6%、Gopher 60.0%（“improving upon Gopher by 7.6%”） | 摘要与表差 0.1；课文写“67.5%（摘要；表 6 印 67.6%），比 Gopher 的 60.0% 高 7 个百分点以上” |
| Chinchilla 的混合精度 | §4.1 只写前向/反向 bfloat16、分布式优化器状态里的 float32 权重副本 | 图 2.10 原标题“Chinchilla 的做法”把 FP32 的 m、v 也算在 Chinchilla 名下；改为“m、v（FP32，笔者假设）”，说明文字同步 |
| Xiong：Post-LN 需要 warmup 的原因 | 2002.04745 摘要：“the expected gradients of the parameters near the output layer are large” | ✓ |
| PaLM 的 MFU | 2204.02311 正文训练效率一节：MFU 是 “implementation-independent” 的效率指标，公式在附录 B | ✓ |
| DeepSeek LLM 的三份数据 | 2401.02954 §3.3：“three different datasets: early in-house data, current in-house data, and OpenWebText2”；表 4 后 “This finding might also explain the significant differences …” | ✓ |
| “GPT 系列都用 warmup + 余弦” | GPT-1（D10）、GPT-3（D13）如此；GPT-2 未报告调度（D11） | 课文改为“GPT-1 和 GPT-3 都用……；GPT-2 没有报告” |
| GPT-2 的 1,558M、Pre-LN + ln_f、初始化 0.02 | `lesson-01-doc.md` C18/C20/E6/E7（openai/gpt-2 的 DEVELOPERS.md 与 model.py；wte 与 conv1d 0.02、wpe 0.01） | 课文原把“GPT-2 的官方代码和配置”只引 hfgpt2（Hugging Face 并非官方代码）；新增 gpt2code 卡片，分开引用 |
| 数值复算 | Python 3.11 + NumPy 1.26.4 | $\log_{10}50257^{2048}=9628$ 位；FP32 连乘 0.1 在第 46 次变成 0；$(1-6\times10^{-6})^{93750}=0.570$；Kaplan 表 6 在 $C_{\min}=3634$ 时 $N=5.16\times10^{11}$、$D=1.83\times10^{11}$、$6ND=1.81C$；方法 3（TeX）在 $10^{19}/10^{23}$ 的谷底 271M / 18.15B；$\arctan(4/3)=53.13^\circ$；$(0.424/0.026)=16.3$ ✓ |
