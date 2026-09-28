# 第 01 课研究笔记：Transformer 解剖——GPT 的一次前向传播

> 对应课文：`deepseek/src/lesson-01-transformer-anatomy.md`（已于 2026-09-24 写成）。建立于 2026-09-24。
> 本文件只存放**核对过的**引文、公式、规格、复算数字、勘误和“未能核实”清单，不是课文。
> 核对方法：arXiv 论文用 `.cursor/skills/academic-research/scripts/arxiv.py text/grep` 取 LaTeXML 全文后 grep；
> 老论文的早期版本（v1–v5）和 OpenAI 报告取 PDF 用 `pdftotext`/`pypdf` 抽文字；代码与配置取 GitHub raw /
> Hugging Face raw。所有派生数字都在 Python 里复算过（表达式见 §4）。
>
> **引文政策**：英文原文引文一律只摘**不超过 15 个词的关键短语**，并给出节/表/公式号与版本；需要看整段时，
> 按给出的位置用 `arxiv.py grep <id> '<短语>' -C 3` 复查。公式、数字、超参数属于事实，完整转录。
>
> 与第 2 课的分工：本课管“架构、形状、参数量、每 token 前向 FLOPs、BPE、LN 位置”；
> 训练目标、优化器、学习率、6ND、scaling law 放在 `lesson-02-doc.md`。两份文档共用的来源（GPT-1/2/3、Kaplan）
> 在两边各记与本课有关的部分。

---

## 0. 覆盖范围与所读版本

| 短键 | 标题 | 作者 | arXiv / 出处 与所读版本 | 发表状态 | 日期 |
|---|---|---|---|---|---|
| vaswani2017 | Attention Is All You Need | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin | 1706.03762。主读 **v7**（arXiv HTML，2023-08-02）；另取 **v1、v2、v3、v4、v5、v7 的 PDF** 核对版本漂移 | NIPS 2017（GPT-1 参考文献 [62] 写作 “Advances in Neural Information Processing Systems, pages 6000–6010, 2017”） | v1 2017-06-12；v2 06-19；v3 06-20；v4 06-30；v5 2017-12-06；v6 2023-07-24；v7 2023-08-02（arXiv 提交历史） |
| radford2018 (GPT-1) | Improving Language Understanding by Generative Pre-Training | Radford, Narasimhan, Salimans, Sutskever | 无 arXiv。OpenAI PDF `cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf`（12 页；PDF 元数据 CreationDate 2018-06-08） | OpenAI 技术报告（未经同行评审） | 2018-06 |
| radford2019 (GPT-2) | Language Models are Unsupervised Multitask Learners | Radford, Wu, Child, Luan, Amodei, Sutskever | 无 arXiv。OpenAI PDF `cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf`（24 页；元数据 CreationDate 2019-02-14） | OpenAI 技术报告。PDF 元数据 Subject 字段写着 “Proceedings of the International Conference on Machine Learning 2019”，那是 LaTeX 模板残留，**不是** ICML 论文（见 §7-E12） | 2019-02 |
| gpt2code | openai/gpt-2 仓库（`README.md`、`src/model.py`、`src/encoder.py`、`DEVELOPERS.md`） | OpenAI | GitHub master 分支（仓库状态 “Archive”），2026-09-24 读取 | 制品（kind: artefact） | — |
| hfgpt2 | Hugging Face `openai-community/gpt2{,-medium,-large,-xl}` 与 `openai-gpt` 的 `config.json`、`vocab.json`、`merges.txt`；transformers `modeling_gpt2.py`、`activations.py` | Hugging Face | 2026-09-24 读取 main 分支 | 制品 | — |
| brown2020 (GPT-3) | Language Models are Few-Shot Learners | Brown, Mann, Ryder, Subbiah, Kaplan, Dhariwal 等 31 人 | 2005.14165 **v4**（arXiv HTML） | NeurIPS 2020（proceedings.neurips.cc 有该文） | v1 2020-05-28；v4 2020-07-22 |
| kaplan2020 | Scaling Laws for Neural Language Models | Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei | 2001.08361 **v1**（唯一版本；arXiv HTML + PDF） | arXiv 预印本（未见正式发表） | 2020-01-23 |
| sennrich2016 | Neural Machine Translation of Rare Words with Subword Units | Sennrich, Haddow, Birch | 1508.07909 **v5** | ACL 2016（arXiv comments: “accepted at ACL 2016”） | v1 2015-08-31；v5 2016-06-10 |
| hendrycks2016 | Gaussian Error Linear Units (GELUs) | Hendrycks, Gimpel | 1606.08415 **v5**；v1 标题为 “Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units”（GPT-1 引的就是这个旧标题） | arXiv 预印本（comments: “Trimmed version of 2016 draft”） | v1 2016-06-27；v5 2023-06-06 |
| ba2016 | Layer Normalization | Ba, Kiros, Hinton | 1607.06450 **v1**（唯一版本） | arXiv 预印本 | 2016-07-21 |
| he2016 | Deep Residual Learning for Image Recognition | He, Zhang, Ren, Sun | 1512.03385 **v1** | CVPR 2016（Vaswani 参考文献 [11]：pages 770–778） | 2015-12-10 |
| xiong2020 | On Layer Normalization in the Transformer Architecture | Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, Liu | 2002.04745 **v2** | ICML 2020（arXiv jref: “Published on ICML 2020”） | v1 2020-02-12；v2 2020-06-29 |
| press2017 | Using the Output Embedding to Improve Language Models | Press, Wolf | 1608.05859 **v3** | EACL 2017（comments） | v1 2016-08-20；v3 2017-02-21 |
| geva2021 | Transformer Feed-Forward Layers Are Key-Value Memories | Geva, Schuster, Berant, Levy | 2012.14913 **v2** | EMNLP 2021（comments） | v1 2020-12-29；v2 2021-09-05 |
| elhage2021 | A Mathematical Framework for Transformer Circuits | Elhage, Nanda, Olsson, Henighan, Joseph, Mann 等（Anthropic） | 无 arXiv。`transformer-circuits.pub/2021/framework/index.html`，页面标注 “Published Dec 22, 2021” | 制品（Transformer Circuits Thread 网文） | 2021-12-22 |
| t2t | tensor2tensor `models/transformer.py`、`layers/common_hparams.py` | Google | GitHub master，2026-09-24 读取 | 制品 | — |
| nanogpt | karpathy/nanoGPT `model.py` | Karpathy | GitHub master，2026-09-24 读取 | 制品（仅作“社区通行实现”的旁证） | — |
| palm2022 | PaLM: Scaling Language Modeling with Pathways | Chowdhery 等 | 2204.02311 **v5**，只读 Appendix B 的 FLOPs 公式 | JMLR 2023（未在本轮核对） | 2022-04-05 |

作者名单核对：vaswani2017（8 人，arXiv 元数据）、kaplan2020（10 人，GPT-3 参考文献 [57] 列全）、brown2020（6 + 25 = 31 人，元数据）、
xiong2020（10 人：Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu，v2 首页）、
其余 arXiv 论文作者均与 `arxiv.py meta` 输出一致。

---

## 1. 核心论断与原文引文

每条：论断（中文）→ 短引文（英文，≤15 词）→ 位置与版本。

### 1.1 Vaswani et al. 2017（1706.03762）

**C1 缩放点积注意力的定义。** 输入是维度 $d_k$ 的查询与键、维度 $d_v$ 的值；点积除以 $\sqrt{d_k}$ 后做 softmax，
得到对值的加权。
- “We call our particular attention "Scaled Dot-Product Attention"” — §3.2.1，式 (1)，v7。

**C2 为什么除以 $\sqrt{d_k}$：v1 只有一句猜测，方差论证是 v2 才加的脚注 4。**
- v1（2017-06-12）§3.2.1 原文只说点积 “growing too large in magnitude to result in useful gradients”（v1 PDF）。
- v2 起改写为 softmax 进入 “regions where it has extremely small gradients”，并加脚注，脚注核心句：
  “their dot product … has mean 0 and variance $d_k$” — §3.2.1 **脚注 4**（v2、v3、v4、v5、v7 的 PDF 都是 4 号）。
- 脚注的**假设**必须一起讲：q、k 的分量是“independent random variables with mean 0 and variance 1”（同一脚注）。
- arXiv HTML（v7 LaTeXML）把它渲染成脚注 “1”，因为作者贡献等 `\thanks` 脚注不编号——引用时写“脚注 4（PDF 编号）”。

**C3 为什么是点积而不是加性注意力。** 两者理论复杂度相近，点积可以用高度优化的矩阵乘法，
“much faster and more space-efficient in practice” — §3.2.1。并且“小 $d_k$ 时两者相近，大 $d_k$ 时加性注意力胜过**未缩放**的点积”，
引 [3]（Britz et al. 2017, 1703.03906）——这正是引入缩放的实验动机。

**C4 多头注意力。** $h=8$，$d_k=d_v=d_{model}/h=64$；因为每头维度变小，总计算量与全维单头相近。
- “jointly attend to information from different representation subspaces at different positions” — §3.2.2。
- “With a single attention head, averaging inhibits this.” — §3.2.2。
- 表 3 (A) 行（v7）：单头（$h=1,d_k=d_v=512$）BLEU 24.9，比最好设置低 0.9；$h=32$（$d_k=16$）也掉到 25.4。
  论文原话：“quality also drops off with too many heads” — §6.2。

**C5 因果掩码。** 解码器自注意力里，把 softmax 输入中“非法连接”置为 $-\infty$。
- “masking out (setting to $-\infty$) all values in the input of the softmax” — §3.2.3。
- 另见 §3.1 解码器段：掩码 + 输出嵌入右移一位，使位置 $i$ 的预测只依赖 $<i$ 的已知输出。

**C6 FFN。** 两个线性变换夹一个 ReLU，逐位置独立；$d_{model}=512$，$d_{ff}=2048$（即 4 倍）。
- “two linear transformations with a ReLU activation in between” — §3.3，式 (2)。
- 论文**没有**解释为什么取 4 倍（只给了数值）。表 3 (C) 行改 $d_{ff}$ 为 1024/4096：PPL 5.12/4.75，BLEU 25.4/26.2。

**C7 嵌入共享与 $\sqrt{d_{model}}$ 放大。** 两个嵌入层与 pre-softmax 线性层共用一个权重矩阵，“similar to [30]”（[30] = Press & Wolf）；
嵌入层里把权重乘 $\sqrt{d_{model}}$ — §3.4。
- 注意：GPT-2 代码**没有**乘 $\sqrt{d}$（`h = tf.gather(wte, X) + tf.gather(wpe, …)`，见 §1.3 C16）。

**C8 正弦位置编码。** 见 §2 式 E4。波长构成从 $2\pi$ 到 $10000\cdot2\pi$ 的几何级数；作者假设它便于学习相对位置
（对固定偏移 $k$，$PE_{pos+k}$ 是 $PE_{pos}$ 的线性函数）。可学习位置嵌入“nearly identical results”（§3.5；表 3 (E) 行：
PPL 4.92、BLEU 25.7，base 为 4.92、25.8）。选正弦的理由是“may allow the model to extrapolate”——**假设，未验证**。

**C9 Post-LN。** 每个子层输出是 $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$ — §3.1。这就是后来所说的 Post-LN。
- 但 tensor2tensor 代码后来默认改成了 Pre-LN（§1.6 C28），论文与代码不一致。

**C10 复杂度表（表 1，v7）。** 自注意力每层 $O(n^2\cdot d)$、顺序操作 $O(1)$、最长路径 $O(1)$；循环层 $O(n\cdot d^2)$、$O(n)$、$O(n)$；
卷积 $O(k\cdot n\cdot d^2)$、$O(1)$、$O(\log_k n)$；受限自注意力 $O(r\cdot n\cdot d)$、$O(1)$、$O(n/r)$。
论文据此说序列长度 $n$ 小于表示维度 $d$ 时自注意力比循环层快（§4，节选：“faster than recurrent layers when the sequence length n is smaller”）。

**C11 基础/大模型规格。** base：$N=6$，$d_{model}=512$，$d_{ff}=2048$，$h=8$，$d_k=d_v=64$，$P_{drop}=0.1$，$\epsilon_{ls}=0.1$，100K 步，65M 参数；
big：$N=6$，1024，4096，16 头，$P_{drop}=0.3$，300K 步，213M 参数（表 3，v7）。base 每步 0.4 s、共 12 小时；big 每步 1.0 s、3.5 天（§5.2，8 块 P100）。

**C12 版本漂移：EN-FR BLEU（七个版本的 PDF 逐一核对，见 §10 核查记录）。**

| 版本 | 摘要 | 表 2（Transformer big） | §6.1 正文 | 其他 |
|---|---|---|---|---|
| v1（2017-06-12） | 41.0 | 41.0 | 41.0 | 摘要 “3.5 days” |
| v2（06-19） | **41.2** | 41.0 | **41.17** | 摘要 “**4.5** days”；§6.1 另写 EN-FR big “did not share source and target embeddings, dff = 8192” |
| v3（06-20） | 41.0 | 41.0 | **41.17** | §6.1 同 v2（dff = 8192、不共享嵌入） |
| v4（06-30） | 41.0 | 41.0 | 41.0 | 删去 dff = 8192 那句 |
| v5（2017-12-06）–v7 | **41.8** | **41.8** | 41.0 | — |

所以**不是**“v1–v4 一律 41.0”：v2、v3 出现过 41.2 / 41.17。v5 起摘要与表 2 为 41.8，§6.1 正文到 v7 仍写 “achieves a BLEU score of 41.0”。
课文引用 EN-FR 成绩时写“41.8（v5 起摘要与表 2；正文残留 41.0）”。

### 1.2 GPT-1（Radford et al. 2018，OpenAI 报告）

**C13 GPT-1 模型规格（§4.1 “Model specifications” 一段，逐项核对）：**
| 项 | 论文原文短语 | 结论 |
|---|---|---|
| 层数/类型 | “12-layer decoder-only transformer with masked self-attention heads” | 12 层，仅解码器 ✓ |
| 宽度/头数 | “768 dimensional states and 12 attention heads” | $d=768$，12 头 ✓ |
| FFN | “we used 3072 dimensional inner states” | $d_{ff}=3072=4d$ ✓ |
| 上下文 | “contiguous sequences of 512 tokens” | 512 ✓ |
| 数据 | “We use the BooksCorpus dataset”（§4.1 Unsupervised pre-training），“over 7,000 unique unpublished books” | BooksCorpus ✓ |
| 激活 | “Gaussian Error Linear Unit (GELU)” | GELU ✓ |
| 位置 | “learned position embeddings instead of the sinusoidal version” | 可学习位置嵌入 ✓ |
| 词表 | “bytepair encoding (BPE) vocabulary with 40,000 merges” | 4 万次合并；HF `config.json` 的 `vocab_size` = **40478** |
| 初始化 | “a simple weight initialization of N(0, 0.02) was sufficient” | $\mathcal N(0,0.02)$（0.02 是标准差还是方差？原文写 N(0, 0.02)，代码 `initializer_range: 0.02` 作标准差用） |
| 优化 | Adam，max lr 2.5e-4，前 2000 次更新线性升温，余弦退火到 0；100 epochs，minibatch 64 | 见第 2 课 |
| 正则 | residual/embedding/attention dropout 0.1；“modified version of L2 regularization proposed in [37], with w = 0.01” | [37] = Loshchilov & Hutter “Fixing weight decay regularization in adam”（即 AdamW 的 v1 标题） |
| 预处理 | ftfy 清洗 + spaCy 分词器 | ✓ |

**C14 GPT-1 的前向公式直接体现权重共享：** 式 (2) 的输出层是 $P(u)=\mathrm{softmax}(h_nW_e^T)$，用的是**输入嵌入矩阵 $W_e$ 的转置**（§3.1）。
GPT-1 报告**没有**给出参数总数；“117M”这个数字来自 GPT-2 报告表 2（“The smallest model is equivalent to the original GPT”，GPT-2 §3）。

**C15 GPT-1 的 LN 位置。** 报告只说 “layernorm [2] is used extensively throughout the model”，没有写 Pre/Post。
HF `modeling_openai.py` 的 `Block.forward` 为 `n = self.ln_1(x + a)`、`h = self.ln_2(n + m)`，即 Post-LN，且全文无 `ln_f`；
其 `ACT_FNS["gelu"]` 映射到 `gelu_new`（tanh 近似）。这是 HF 移植代码的事实，报告本身未写（原始 OpenAI 代码未比对，§8）。

### 1.3 GPT-2（Radford et al. 2019，OpenAI 报告）与官方代码

**C16 相对 GPT-1 的改动（§2.3，四句全部核对）：**
1. “Layer normalization … was moved to the input of each sub-block”，类比 pre-activation ResNet（He et al. 2016）。
2. “an additional layer normalization was added after the final self-attention block”。
   代码里这个 `ln_f` 位于**最后一个块（含 MLP）之后**、unembedding 之前（`h = norm(h, 'ln_f')`），不是字面上的“最后一个自注意力之后”——见 §7-E7。
3. “We scale the weights of residual layers at initialization by a factor of 1/√N”，$N$ 为残差层数（PDF 经 pypdf 抽取确认根号存在）。
4. “The vocabulary is expanded to 50,257”；上下文 512 → 1024；batch size 512。

**C17 GPT-2 报告表 2 的四个规模（如印刷）：** Parameters 117M / 345M / 762M / 1542M；Layers 12 / 24 / 36 / 48；$d_{model}$ 768 / 1024 / 1280 / 1600。
报告**没有**给头数；头数来自代码配置：12 / 16 / 20 / 25（HF `n_head`），因此每头维度都是 64。

**C18 官方仓库承认参数量印错了。** README：“our original parameter counts were wrong due to an error”（括注说错误出现在此前的博客和论文里），
并说小、中模型曾被称为 117M、345M；`DEVELOPERS.md` 里的下载名是 **124M / 355M / 774M / 1558M**。复算见 §4.1，四个新名字都与精确计数吻合。

**C19 字节级 BPE。** 基础词表只需 256 个字节（“a byte-level version of BPE only requires a base vocabulary of size 256”，§2.2）；
禁止跨字符类别合并，但对空格开例外（§2.2）。HF 词表文件核对：`merges.txt` 首行 `#version: 0.2` 之后恰好 **50,000** 条合并，
`vocab.json` 共 **50,257** 项，id 50256 = `<|endoftext|>` → $50{,}257=256+50{,}000+1$。

**C20 官方代码的实现细节（`src/model.py`，与课文“形状追踪”直接相关）：**
- `block`：`a = attn(norm(x,'ln_1'))`；`x = x + a`；`m = mlp(norm(x,'ln_2'), n_state=nx*4)`；`x = x + m` → Pre-LN，FFN 宽 4d。
- `attn`：一个 `conv1d(x,'c_attn', n_state*3)` 同时产生 Q、K、V（带偏置），再 `split_heads` 成 `[batch, heads, seq, features]`。
- 缩放：`w = w * tf.rsqrt(v.shape[-1])`，即除以 $\sqrt{d_v}=\sqrt{64}=8$（GPT-2 中 $d_k=d_v$，故与论文一致）。
- 掩码：`w = w*b - 1e10*(1-b)`，用 $-10^{10}$ 而不是 $-\infty$；注释 “information flows from src to dst”。
- 位置/词嵌入初始化标准差：`wpe` 0.01，`wte` 0.02；其余 `conv1d` 权重 0.02——**没有** 1/√N 缩放（见 §7-E6）。
- LN：`epsilon=1e-5`，在 `rsqrt(s + epsilon)` 里。
- GELU：tanh 近似 `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`。
- 输出：`logits = tf.matmul(h_flat, wte, transpose_b=True)` → 与输入嵌入共享（weight tying）。
- 缓存形状 `past_shape = [batch, n_layer, 2, n_head, sequence, n_embd // n_head]` —— KV cache（第 4 课）在 2019 年的官方代码里已经存在。

**C21 HF 实现把 1/√N 解释为 $N=2\times$层数。** `modeling_gpt2.py` `_init_weights`：`std = initializer_range / math.sqrt(2 * n_layer)` 作用于每个
`c_proj.weight`，注释 “2 residual paths (attention and MLP) per Transformer Block”。nanoGPT 同样 `0.02/math.sqrt(2 * n_layer)`。
这是社区实现对报告的**解读**，报告原文只说 “the number of residual layers”。

### 1.4 GPT-3（Brown et al. 2020，2005.14165 v4）——仅架构部分

**C22 架构沿用 GPT-2，唯一例外是交替的稠密/局部带状稀疏注意力。** “alternating dense and locally banded sparse attention patterns”，
“similar to the Sparse Transformer [15]” — §2.1。**带宽、哪些层稀疏均未给出**（§8）。

**C23 FFN 恒为 4 倍、上下文 2048。** “we always have the feedforward layer four times the size of the bottleneck layer”；
“All models use a context window of $n_{ctx}=2048$ tokens” — §2.1。架构参数“chosen based on computational efficiency and load-balancing”。
表 2.1 全表见第 2 课研究笔记 §3；本课用到的参数量核对见 §4.2。

**C24 GPT-3 论文没有写词表大小**，只说沿用 GPT-2 的 “reversible tokenization”（§2.1）。“GPT-3 词表 50,257”是推断，不是 GPT-3 论文的陈述。

### 1.5 各组件的来源论文

**C25 BPE（Sennrich et al. 2016，v5 §3.2）。** 用字符表初始化符号表，每个词末尾加一个词尾符号；反复把最频繁的符号对 (A,B) 替换为新符号 AB；
最终符号表大小 = 初始字符表大小 + 合并次数，
合并次数是 “the only hyperparameter of the algorithm”。不考虑跨词边界的对。
算法 1（Python 代码）里的玩具词典是 `{'l o w </w>':5, 'l o w e r </w>':2, 'n e w e s t </w>':6, 'w i d e s t </w>':3}`，
而**图 1 的词典是另一个**：{‘low’, ‘lowest’, ‘newer’, ‘wider’}（见 §7-E9）。

**C26 GELU（Hendrycks & Gimpel，v5 §2）。** $\mathrm{GELU}(x)=x\Phi(x)$，是“按输入大小加权而不是按符号门控”（“weights inputs by their value, rather than gates inputs by their sign”，摘要）。
两种近似：tanh 式与 $x\sigma(1.702x)$；作者“used the former in every experiment”。

**C27 LayerNorm（Ba et al. 2016，v1 §3）。** 统计量在同一层所有隐藏单元上计算（式 (3)），“all the hidden units in a layer share the same normalization terms”；
训练与测试计算完全相同。**原式没有 $\epsilon$**；实现（GPT-2 代码）在方差上加 $10^{-5}$。

**C28 残差（He et al. 2016，v1 §3.1–3.2）。** 让堆叠层拟合残差 $\mathcal F(\mathbf x):=\mathcal H(\mathbf x)-\mathbf x$，式 (1) $\mathbf y=\mathcal F(\mathbf x,\{W_i\})+\mathbf x$；
恒等捷径 “add neither extra parameter nor computational complexity”（§1）。

**C29 Pre-LN 的理论（Xiong et al. 2020，v2）。**
- 定义：Post-LN 把 LN 放在残差块之间；Pre-LN 把 LN 放进残差分支、位于所有非线性变换之前，并在预测前加一个最终 LN（§3.1 末段、表 1）。
- 定理 1：初始化时，Post-LN 最后一层 FFN 第二个权重的梯度范数上界 $O(d\sqrt{\ln d})$，与层数 $L$ 无关；Pre-LN 为 $O(d\sqrt{\ln d/L})$（§3.3）。
- **定理的假设（必须一起讲）**：单头注意力；$W^Q,W^K$ 初始化为零（注意力退化为均匀平均）；所有权重 $d\times d$；
  **输入向量也从同一高斯分布采样**（“we assume the input vectors are also sampled from the same Gaussian distribution”，其余权重按 Xavier 初始化）；
  FFN 用 ReLU（表 1）；定理 1 本身另要求 $\|x^{post,5}_{L,i}\|_2^2$、$\|x^{pre}_{L+1,i}\|_2^2$ 满足 $(\epsilon,\delta)$-有界（§3.3 “we study a simpler setting”）。
- 引理 2：Post-LN 中 $\E\|x^{post,5}_{l,i}\|_2^2=\tfrac32 d$；Pre-LN 中 $(1+\tfrac l2)d\le\E\|x^{pre}_{l,i}\|_2^2\le(1+\tfrac{3l}{2})d$——Pre-LN 残差流的尺度随深度线性增长。
- 结论：Post-LN 必须 warm-up；Pre-LN 可以去掉 warm-up（摘要）。实验：IWSLT14 De-En 上 Post-LN 不用 warm-up、Adam 训练 BLEU 只有 8.45，用 warm-up 约 34（§3.2）。
- 归属：Xiong 自己说 Pre-LN “was implemented in several systems”，引 Vaswani et al. 2018（tensor2tensor）、Klein et al. 2018、Liu et al. 2019b（§3.3），
  并说 Wang et al. 2019 发现层数增加时 Pre-LN 更好。**Pre-LN 不是 GPT-2 或 Xiong 发明的。**

**C30 权重共享（Press & Wolf 2017，v3）。** 建议把输入嵌入 $U$ 与输出嵌入 $V$ 绑成 $U=V$（§1 (iii)）；能降低多种语言模型的困惑度；
在神经机器翻译里可把模型大小降到原来的 “less than half … without harming their performance”（摘要）。
归属：此前 Mnih & Hinton 2009 的 log-bilinear 模型已用过但未解释；Inan et al. 2016 为独立同期工作（§2）。

**C31 FFN = 键值记忆（Geva et al. 2021，v2 §2）。** 忽略偏置，$\mathrm{FF}(\mathbf x)=f(\mathbf x\cdot K^\top)\cdot V$（式 (1)），与神经记忆
$\mathrm{MN}(\mathbf x)=\mathrm{softmax}(\mathbf x\cdot K^\top)\cdot V$（式 (2)）只差非线性是否归一化。
- 键：“each key correlates with textual patterns in the training examples”；值在输出词表上诱导出一个分布（摘要）。
- 低层键偏“浅层模式”（n-gram），高层偏“语义模式”（§3.2：层 1–9 vs 10–16）。
- 实验对象是 Baevski & Auli 2019 的 16 层 WikiText-103 模型，$d=1024$，$d_m=4096$，共 65,536 个键（§3.1）。
- “Feed-forward layers constitute two-thirds of a transformer model’s parameters”（摘要首句；自注意力占 1/3，即每层 $4d^2$，§1）。

**C32 残差流框架（Elhage et al. 2021，制品）。**
- 残差流是 “the sum of the output of all the previous layers and the original embedding”（14 词节选，“Transformer Overview”节）。
- 把残差流看成通信信道：“it doesn't do any processing itself and all layers communicate through it”。
- 线性、可加的结构意味着残差流 “doesn't have a "privileged basis"”。
- 注意力头的 “concatenate and multiply” 写法与“各头独立输出再相加”数学等价（Summary of Results）。
- 维度被比作 “memory” 或 “bandwidth”；某些 MLP 神经元/头可能在做“memory management”（清除别人写的维度）——作者用词为 “hints”，属推测。

### 1.6 参数量与 FLOPs 公式、代码漂移

**C33 Kaplan 表 1 与式 (2.1)(2.2)（2001.08361 v1 §2.1）。** 非嵌入参数 $N\approx2d_{model}n_{layer}(2d_{attn}+d_{ff})=12n_{layer}d_{model}^2$（取 $d_{attn}=d_{ff}/4=d_{model}$）；
前向每 token $C_{forward}\approx2N+2n_{layer}n_{ctx}d_{model}$；当 $d_{model}>n_{ctx}/12$ 时上下文相关的计算占比较小。
表 1 注明 “Sub-leading terms such as nonlinearities, biases, and layer normalization are omitted”。

**C34 tensor2tensor 的 LN 位置随版本漂移（代码核对）。** `common_hparams.basic_params1` 默认 `layer_preprocess_sequence="none"`、
`layer_postprocess_sequence="dan"`（dropout→add→norm，即 Post-LN，与论文一致）；`transformer_base_v1` 继承之；
`transformer_base_v2` 改为 `"n"` / `"da"`（Pre-LN）；当前 `transformer_base()` → `v3` → `v2`，所以**默认的 base 是 Pre-LN**。
`transformer_base_v1` 还设了 `shared_embedding_and_softmax_weights = True`、`learning_rate_warmup_steps = 4000`、`optimizer_adam_beta2 = 0.98`（与论文吻合）。

**C35 FLOPs 统计口径不统一（注意力项差 2 倍）。**
- Kaplan 表 1 的 “Attention: Mask” 每 token 前向 $2n_{layer}n_{ctx}d_{attn}$。
- Chinchilla 附录 F 按整段序列计：QK logits $2\,T^2(d_h n_h)$ 与 “Softmax @ query reductions” $2\,T^2(d_h n_h)$ 各一份（另加 softmax $3n_hT^2$），
  即每 token 每层 $\approx4Td_{attn}$。
- PaLM 附录 B：训练时注意力 matmul 每 token $6LH(2QT)$，即前向 $4LTd_{attn}$，总 $6N+12LHQT$。
- 所以 Kaplan 的注意力项是另两家的一半。一种自洽解释是 Kaplan 的数字等于“因果掩码下平均只看一半上下文”，但**论文没有这样说**（§8）。
  DeepSeek LLM 的 $M=72n_{layer}d^2+12n_{layer}d\,l_{seq}$（2401.02954 v1 §3.2）用的是 PaLM 口径（$12=3\times4$），第 6 课要接上。

---

## 2. 公式（原文转录 + 课程记号对照）

**记号对照总表**（style-guide §2）：

| 论文记号 | 课程记号 | 说明 |
|---|---|---|
| Vaswani $n$（序列长）/ Kaplan $n_{ctx}$ / GPT-1 窗口 $k$ | $T$（讨论上下文时 $L_{ctx}$） | GPT-1 的 $k$ 是上下文窗口 |
| $d_{model}$ | $d$ | |
| $h$（Vaswani 头数）/ $n_{heads}$ | $n_h$ | |
| $d_k=d_v$ / $d_{head}$ | $d_h$ | 本课所有模型都有 $d_k=d_v$ |
| $N$（Vaswani 层数）/ $n_{layer}$ / GPT-1 的 $n$ | $L$（KV 公式里 $l$） | 注意 Vaswani 的 $N=6$ 是层数，Kaplan 的 $N$ 是参数量 |
| $d_{attn}$（Kaplan） | $n_hd_h$ | 标准模型里 $=d$ |
| $d_{ff}$ / Geva $d_m$ | $d_{ff}$ | |
| $n_{vocab}$ | $V$ | |
| 行向量 $x W$ | 列向量 $W\mathbf h_t$ | 课程约定列向量，$W_{课}=W_{论文}^\top$ |

**E1 缩放点积注意力（Vaswani 式 (1)，v7 §3.2.1）**
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V\tag{1}$$
课程记号（单头 $i$，位置 $t$，因果）：
$$\mathbf o_{t,i}=\sum_{j=1}^{t}\alpha_{t,j,i}\,\mathbf v_{j,i},\qquad
\alpha_{t,j,i}=\frac{\exp\!\big(\mathbf q_{t,i}^\top\mathbf k_{j,i}/\sqrt{d_h}\big)}{\sum_{s=1}^{t}\exp\!\big(\mathbf q_{t,i}^\top\mathbf k_{s,i}/\sqrt{d_h}\big)}.$$
矩阵形式里 $Q\in\R^{T\times d_h}$ 按行堆 token（写课文时要明说“这里每行一个 token”）。

**E2 多头（Vaswani §3.2.2，无编号）**
$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O,\quad
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)$$
$$W_i^Q\in\R^{d_{model}\times d_k},\;W_i^K\in\R^{d_{model}\times d_k},\;W_i^V\in\R^{d_{model}\times d_v},\;W^O\in\R^{hd_v\times d_{model}}$$
课程记号：$\mathbf q_{t,i}=W^Q_i\mathbf h_t$，$W^Q_i\in\R^{d_h\times d}$，拼起来 $W^Q\in\R^{n_hd_h\times d}$（$K$、$V$ 同理）；
$\mathbf u_t=W^O[\mathbf o_{t,1};\dots;\mathbf o_{t,n_h}]$，$W^O\in\R^{d\times n_hd_h}$。
Elhage 的等价写法：$W^O=[W^O_1,\dots,W^O_{n_h}]$ 按列分块，$\mathbf u_t=\sum_{i}W^O_i\mathbf o_{t,i}$（各头独立写回残差流）。

**E3 FFN（Vaswani 式 (2)）**
$$\mathrm{FFN}(x)=\max(0,\,xW_1+b_1)W_2+b_2\tag{2}$$
GPT 系列：ReLU 换成 GELU。课程记号：$\FFN(\mathbf x)=W_2\,\mathrm{GELU}(W_1\mathbf x+\mathbf b_1)+\mathbf b_2$，$W_1\in\R^{d_{ff}\times d}$，$W_2\in\R^{d\times d_{ff}}$。

**E4 正弦位置编码（Vaswani §3.5，无编号）**
$$PE_{(pos,2i)}=\sin\!\big(pos/10000^{2i/d_{model}}\big),\qquad PE_{(pos,2i+1)}=\cos\!\big(pos/10000^{2i/d_{model}}\big)$$
这里的 $i$ 是**维度对的下标**（不是头下标）。课文里建议改写成 $\omega_m=10000^{-2m/d}$，第 $m$ 对维度的波长 $2\pi/\omega_m$。

**E5 GPT-1 目标与前向（GPT-1 式 (1)(2)，§3.1；符号经 pypdf 抽取确认）**
$$L_1(\mathcal U)=\sum_i\log P(u_i\mid u_{i-k},\dots,u_{i-1};\Theta)\tag{1}$$
$$h_0=UW_e+W_p,\qquad h_l=\mathrm{transformer\_block}(h_{l-1})\;\forall i\in[1,n],\qquad P(u)=\mathrm{softmax}(h_nW_e^T)\tag{2}$$
（如印刷：量词写的是 $\forall i$，但下标是 $l$——排版小误，应读作 $\forall l\in[1,n]$；$n$ 是层数。）
课程记号：$\mathbf h^{(0)}_t=E[x_t]+P[t]$，$E\in\R^{V\times d}$，$P\in\R^{T_{max}\times d}$；logits $\mathbf z_t=E\,\mathbf h^{(L)}_t\in\R^V$。

**E6 GPT-2 块（报告 §2.3 文字 + `model.py`，课程记号）**
$$\mathbf h'_t=\mathbf h^{(l-1)}_t+\big[\mathrm{MHA}\big(\LayerNorm_1(\mathbf h^{(l-1)})\big)\big]_t,\qquad
\mathbf h^{(l)}_t=\mathbf h'_t+\FFN\big(\LayerNorm_2(\mathbf h'_t)\big)$$
$$\mathbf z_t=E\,\LayerNorm_f\big(\mathbf h^{(L)}_t\big)$$
初始化：残差分支的输出投影 $W^O$、$W_2$ 的标准差 $0.02/\sqrt{N_{res}}$，$N_{res}$ = 残差层数（HF/nanoGPT 取 $2L$）。

**E7 GELU（Hendrycks & Gimpel v5 §2）**
$$\mathrm{GELU}(x)=xP(X\le x)=x\Phi(x)=x\cdot\tfrac12\big[1+\mathrm{erf}(x/\sqrt2)\big]$$
近似：$0.5x\big(1+\tanh[\sqrt{2/\pi}(x+0.044715x^3)]\big)$ 或 $x\sigma(1.702x)$。

**E8 LayerNorm（Ba et al. v1 式 (3)(4)）**
$$\mu^l=\frac1H\sum_{i=1}^{H}a_i^l,\qquad\sigma^l=\sqrt{\frac1H\sum_{i=1}^{H}\big(a_i^l-\mu^l\big)^2}\tag{3}$$
$$\mathbf h^t=f\!\left[\frac{\mathbf g}{\sigma^t}\odot(\mathbf a^t-\mu^t)+\mathbf b\right]\tag{4}$$
Xiong 的写法（§3.1）：$\mathrm{LayerNorm}(v)=\gamma\frac{v-\mu}{\sigma}+\beta$。实现（GPT-2）：$\gamma\odot(\mathbf x-\mu)/\sqrt{\sigma^2+10^{-5}}+\beta$。
课程记号：$\LayerNorm(\mathbf x)=\boldsymbol\gamma\odot\frac{\mathbf x-\mu\mathbf 1}{\sqrt{\sigma^2+\epsilon}}+\boldsymbol\beta$，$\mu,\sigma^2$ 在 $d$ 个分量上求。

**E9 残差（He et al. v1 式 (1)）**
$$\mathbf y=\mathcal F(\mathbf x,\{W_i\})+\mathbf x\tag{1}$$

**E10 Post-LN 与 Pre-LN（Xiong 表 1，逐行转录，$l$ 层、位置 $i$）**

| Post-LN | Pre-LN |
|---|---|
| $x^{post,1}_{l,i}=\mathrm{MultiHeadAtt}(x^{post}_{l,i},[x^{post}_{l,1},\dots,x^{post}_{l,n}])$ | $x^{pre,1}_{l,i}=\mathrm{LayerNorm}(x^{pre}_{l,i})$ |
| $x^{post,2}_{l,i}=x^{post}_{l,i}+x^{post,1}_{l,i}$ | $x^{pre,2}_{l,i}=\mathrm{MultiHeadAtt}(x^{pre,1}_{l,i},[x^{pre,1}_{l,1},\dots,x^{pre,1}_{l,n}])$ |
| $x^{post,3}_{l,i}=\mathrm{LayerNorm}(x^{post,2}_{l,i})$ | $x^{pre,3}_{l,i}=x^{pre}_{l,i}+x^{pre,2}_{l,i}$ |
| $x^{post,4}_{l,i}=\mathrm{ReLU}(x^{post,3}_{l,i}W^{1,l}+b^{1,l})W^{2,l}+b^{2,l}$ | $x^{pre,4}_{l,i}=\mathrm{LayerNorm}(x^{pre,3}_{l,i})$ |
| $x^{post,5}_{l,i}=x^{post,3}_{l,i}+x^{post,4}_{l,i}$ | $x^{pre,5}_{l,i}=\mathrm{ReLU}(x^{pre,4}_{l,i}W^{1,l}+b^{1,l})W^{2,l}+b^{2,l}$ |
| $x^{post}_{l+1,i}=\mathrm{LayerNorm}(x^{post,5}_{l,i})$ | $x^{pre}_{l+1,i}=x^{pre,5}_{l,i}+x^{pre,3}_{l,i}$ |
| | Final LayerNorm：$x^{pre}_{Final,i}\leftarrow\mathrm{LayerNorm}(x^{pre}_{L+1,i})$ |

**E11 Xiong 定理 1（§3.3）** 在上面的假设下，以概率至少 $0.99-\delta-\frac{\epsilon}{0.9+\epsilon}$：
$$\text{Post-LN: }\Big\|\frac{\partial\tilde{\mathcal L}}{\partial W^{2,L}}\Big\|_F\le\mathcal O\big(d\sqrt{\ln d}\big),\qquad
\text{Pre-LN: }\Big\|\frac{\partial\tilde{\mathcal L}}{\partial W^{2,L}}\Big\|_F\le\mathcal O\Big(d\sqrt{\tfrac{\ln d}{L}}\Big).$$
引理 3（直觉的来源）：$\|\mathbf J_{LN}(x)\|_2=\mathcal O(\sqrt d/\|x\|_2)$——LN 的雅可比与输入范数成反比，Pre-LN 最终 LN 的输入范数 $\propto\sqrt L$。

**E12 Geva 式 (1)(2)**
$$\mathrm{FF}(\mathbf x)=f(\mathbf x\cdot K^\top)\cdot V\tag{1}\qquad
\mathrm{MN}(\mathbf x)=\mathrm{softmax}(\mathbf x\cdot K^\top)\cdot V\tag{2}$$
$K,V\in\R^{d_m\times d}$（行 = 一条记忆）。课程记号：$K=W_1$（$d_{ff}\times d$），$V=W_2^\top$（$d_{ff}\times d$）；$\FFN(\mathbf x)=\sum_{m=1}^{d_{ff}}f(\mathbf k_m^\top\mathbf x)\,\mathbf v_m$。

**E13 Kaplan 表 1（v1 §2.1，逐行转录；符号照印刷）**

| Operation | Parameters | FLOPs per Token |
|---|---|---|
| Embed | $(n_{vocab}+n_{ctx})d_{model}$ | $4d_{model}$ |
| Attention: QKV | $n_{layer}d_{model}3d_{attn}$ | $2n_{layer}d_{model}3d_{attn}$ |
| Attention: Mask | — | $2n_{layer}n_{ctx}d_{attn}$ |
| Attention: Project | $n_{layer}d_{attn}d_{model}$ | $2n_{layer}d_{attn}d_{embd}$（印刷如此，$d_{embd}$ 应为 $d_{model}$） |
| Feedforward | $n_{layer}2d_{model}d_{ff}$ | $2n_{layer}2d_{model}d_{ff}$ |
| De-embed | — | $2d_{model}n_{vocab}$ |
| Total (Non-Embedding) | $N=2d_{model}n_{layer}(2d_{attn}+d_{ff})$ | $C_{forward}=2N+2n_{layer}n_{ctx}d_{attn}$ |

$$N\approx2d_{model}n_{layer}(2d_{attn}+d_{ff})=12n_{layer}d_{model}^2\quad(d_{attn}=d_{ff}/4=d_{model})\tag{2.1}$$
$$C_{forward}\approx2N+2n_{layer}n_{ctx}d_{model}\tag{2.2}$$
课程记号：$N\approx12Ld^2$，$C_{fwd}\approx2N+2LTd$。
注意两点：(a) $N$ **不含**嵌入，且 $C_{forward}$ 的 “Total” 行**不含** De-embed 的 $2dV$；(b) 表中 Mask 行用 $d_{attn}$，式 (2.2) 用 $d_{model}$。

**E14 精确参数量（笔者推导，GPT-2 结构：带偏置、Pre-LN、共享嵌入、有最终 LN）**
$$N_{total}=\underbrace{Vd}_{wte}+\underbrace{T_{max}d}_{wpe}+L\big(\underbrace{4d^2+4d}_{\text{attn}}+\underbrace{8d^2+5d}_{\text{mlp}}+\underbrace{4d}_{2\times\text{LN}}\big)+\underbrace{2d}_{ln_f}
=(V+T_{max})d+L(12d^2+13d)+2d.$$
（attn：$W_{qkv}$ $3d^2+3d$ + $W^O$ $d^2+d$；mlp：$W_1$ $4d^2+4d$ + $W_2$ $4d^2+d$。）

---

## 3. 规格与超参数表

### 3.1 Transformer（Vaswani 表 3，v7）

| 模型 | $N$(层) | $d_{model}$ | $d_{ff}$ | $h$ | $d_k=d_v$ | $P_{drop}$ | $\epsilon_{ls}$ | 步数 | 参数 | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 6 | 512 | 2048 | 8 | 64 | 0.1 | 0.1 | 100K | 65M | 表 3 |
| big | 6 | 1024 | 4096 | 16 | 64（由 1024/16 推出，表中空白=同 base） | 0.3 | 0.1 | 300K | 213M | 表 3 |

优化器（§5.3）：Adam $\beta_1=0.9$，$\beta_2=0.98$，$\epsilon=10^{-9}$，warmup 4000 步（学习率公式见第 2 课）。
词表：EN-DE “about 37000 tokens” 的共享 BPE；EN-FR 32000 word-piece（§5.1）。

### 3.2 GPT-1

| 项 | 论文（GPT-1 报告 §4.1） | HF `openai-gpt/config.json` | 备注 |
|---|---|---|---|
| 层数 | 12 | `n_layer` 12 | ✓ |
| $d$ | 768 | `n_embd` 768 | ✓ |
| 头数 | 12 | `n_head` 12 | $d_h=64$（推出） |
| $d_{ff}$ | 3072 | —（代码中 4×n_embd） | ✓ |
| 上下文 | 512 | `n_ctx`/`n_positions` 512 | ✓ |
| 词表 | BPE 40,000 merges | `vocab_size` 40478 | 差 478，笔者推测为基础字符等符号数（未核实） |
| 激活 | GELU | `afn` “gelu” | ✓ |
| 位置 | learned | — | ✓ |
| 初始化 | N(0, 0.02) | `initializer_range` 0.02 | ✓ |
| dropout | 0.1（residual/embedding/attention） | `resid_pdrop`/`embd_pdrop`/`attn_pdrop` 0.1 | ✓ |
| LN $\epsilon$ | 未写 | `layer_norm_epsilon` 1e-5 | 仅代码 |
| 参数量 | **未写** | — | 复算 116,534,784（§4.1） |

### 3.3 GPT-2 四个规模（报告表 2 vs 代码配置 vs 复算）

| 名称（报告 / 仓库） | 层 $L$ | $d$ | 头 $n_h$（仅代码） | $d_h$ | 报告表 2 参数 | 复算精确参数（共享嵌入） | 仓库下载名 | 非嵌入参数 | $12Ld^2$ |
|---|---|---|---|---|---|---|---|---|---|
| small | 12 | 768 | 12 | 64 | 117M | 124,439,808 | 124M | 85,056,000 | 84,934,656 |
| medium | 24 | 1024 | 16 | 64 | 345M | 354,823,168 | 355M | 302,311,424 | 301,989,888 |
| large | 36 | 1280 | 20 | 64 | 762M | 774,030,080 | 774M | 708,390,400 | 707,788,800 |
| xl（“GPT-2”） | 48 | 1600 | 25 | 64 | 1542M | 1,557,611,200 | 1558M | 1,475,561,600 | 1,474,560,000 |

四个规模共同项（HF config）：`vocab_size` 50257，`n_ctx`=`n_positions` 1024，`activation_function` “gelu_new”（= tanh 近似，
HF `activations.py` 注释 “identical to OpenAI GPT”），`layer_norm_epsilon` 1e-5，`initializer_range` 0.02，三种 dropout 0.1，
`bos_token_id`=`eos_token_id`=50256。报告：batch size 512（序列），上下文 1024。

### 3.4 GPT-3 架构（表 2.1，v4；参数核对见 §4.2）

| 模型 | $n_{params}$ | $n_{layers}$ | $d_{model}$ | $n_{heads}$ | $d_{head}$ | $n_{heads}d_{head}$ | 表 D.1 Params(M) |
|---|---|---|---|---|---|---|---|
| Small | 125M | 12 | 768 | 12 | 64 | 768 | 125 |
| Medium | 350M | 24 | 1024 | 16 | 64 | 1024 | 356 |
| Large | 760M | 24 | 1536 | 16 | 96 | 1536 | 760 |
| XL | 1.3B | 24 | 2048 | 24 | 128 | **3072 ≠ 2048** | 1,320 |
| 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 2560 | 2,650 |
| 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 4096 | 6,660 |
| 13B | 13.0B | 40 | **5140**（印刷） | 40 | 128 | **5120 ≠ 5140** | 12,850 |
| 175B | 175.0B | 96 | 12288 | 96 | 128 | 12288 | 174,600 |

共同：$d_{ff}=4d_{model}$，$n_{ctx}=2048$（§2.1）。

### 3.5 GPT-2 small 的全流程形状追踪（batch $B$、长度 $T\le1024$；来自 `model.py` 的运算顺序，形状由配置推出）

| 步 | 运算 | 输出形状 | GPT-2 small 数值（$B=1,T=1024$） | 参数 |
|---|---|---|---|---|
| 0 | token id | $B\times T$（int） | $1\times1024$ | — |
| 1 | `wte` 查表 | $B\times T\times d$ | $1\times1024\times768$ | $50257\times768=38{,}597{,}376$ |
| 2 | `wpe` 查表并相加 | $B\times T\times d$ | 同上 | $1024\times768=786{,}432$ |
| 3 | 每块：`ln_1` | $B\times T\times d$ | 同上 | $2\times768$ |
| 4 | `c_attn`（融合 QKV） | $B\times T\times3d$ | $1\times1024\times2304$ | $768\times2304+2304=1{,}771{,}776$ |
| 5 | split + `split_heads` | 3 个 $B\times n_h\times T\times d_h$ | $1\times12\times1024\times64$ | — |
| 6 | $QK^\top\cdot d_h^{-1/2}$ | $B\times n_h\times T\times T$ | $1\times12\times1024\times1024$（12,582,912 个数） | — |
| 7 | 掩码 + softmax | 同上 | 下三角有效 | — |
| 8 | $\times V$ | $B\times n_h\times T\times d_h$ | $1\times12\times1024\times64$ | — |
| 9 | `merge_heads` | $B\times T\times d$ | $1\times1024\times768$ | — |
| 10 | `c_proj`（$W^O$） | $B\times T\times d$ | 同上 | $768\times768+768=590{,}592$ |
| 11 | 残差相加 | $B\times T\times d$ | 同上 | — |
| 12 | `ln_2` | 同上 | 同上 | $2\times768$ |
| 13 | `c_fc` + GELU | $B\times T\times4d$ | $1\times1024\times3072$ | $768\times3072+3072=2{,}362{,}368$ |
| 14 | `c_proj`（$W_2$） | $B\times T\times d$ | $1\times1024\times768$ | $3072\times768+768=2{,}360{,}064$ |
| 15 | 残差相加 | $B\times T\times d$ | 同上 | — |
| 16 | 重复 3–15 共 12 次 | | | 每块 7,087,872 |
| 17 | `ln_f` | $B\times T\times d$ | 同上 | $2\times768$ |
| 18 | `h @ wte^T` | $B\times T\times V$ | $1\times1024\times50257$（51,463,168 个数；fp32 约 205.9 MB） | 0（共享） |
| 19 | softmax（训练时进交叉熵） | $B\times T\times V$ | | — |

HF 的 `Conv1D` 权重形状是 `(in, out)`（行向量约定），nanoGPT 的 `nn.Linear` 是 `(out, in)`，从 HF 加载时要转置 `c_attn/c_proj/c_fc` 四个权重
（nanoGPT `from_pretrained` 的 `transposed` 列表）——课文画形状图时写明约定。

---

## 4. 复算的数字

所有数字由 scratch 目录下的 `params.py`、`gpt3.py`、`flops.py`、`misc.py`、`bpe.py`、`gpt2tok.py` 算出；下列表达式可直接 `python -c` 复现。

### 4.1 GPT-1/GPT-2 参数量

| 量 | 表达式 | 值 |
|---|---|---|
| GPT-2 small 每块参数 | `12*768**2 + 13*768` | 7,087,872 |
| 其中 attn / mlp / LN | `768*2304+2304 + 768*768+768`；`768*3072+3072+3072*768+768`；`4*768` | 2,362,368；4,722,432；3,072 |
| GPT-2 small 总参数 | `(50257+1024)*768 + 12*(12*768**2+13*768) + 2*768` | **124,439,808** |
| 去掉 wpe（nanoGPT “non_embedding” 口径） | `124439808 - 1024*768` | 123,653,376 |
| 若不共享输出矩阵 | `124439808 + 50257*768` | 163,037,184 |
| GPT-2 small 非嵌入参数 | `12*(12*768**2+13*768) + 2*768` | 85,056,000 |
| $12Ld^2$ 近似 | `12*12*768**2` | 84,934,656（比非嵌入精确值小 0.14%） |
| 嵌入占总参数比例 small/medium/large/xl | `(50257+1024)*d / total` | 31.65% / 14.80% / 8.48% / 5.27% |
| GPT-2 medium / large / xl 总参数 | 同式 | 354,823,168 / 774,030,080 / 1,557,611,200 |
| GPT-1 总参数（V=40478，T=512，无最终 LN） | `(40478+512)*768 + 12*(12*768**2+13*768)` | **116,534,784** ≈ 117M |
| **假说**：报告表 2 的数字 = 用 GPT-1 词表 40,478 算 GPT-2 | `(40478+1024)*d + L*(12d^2+13d) + 2d` | 116.930M / 344.809M / 761.513M / 1541.965M → 四舍五入 **117 / 345 / 762 / 1542**，与表 2 四个数**全部吻合** |

最后一行是笔者的推测（数值上四个全中），OpenAI 只说 “due to an error”，从未说明错在哪里。课文可以作为“侦探练习”，但必须标“笔者推测”。

### 4.2 GPT-3 表 2.1 的核对

精确式（带偏置、共享嵌入、$V=50257$、$T_{max}=2048$、$d_{ff}=4d$）：`(V+T)*d + L*(d*3*da+3*da + da*d+d + 8*d*d+5*d + 4*d) + 2*d`，$d_a=n_hd_h$。

| 模型 | $12Ld^2$ | $+(V+T)d$ | 精确（$d_a=d$） | 精确（$d_a=n_hd_h$） | 表 D.1 |
|---|---|---|---|---|---|
| Small | 0.085B | 0.125B | 0.125B | 同 | 125M |
| Medium | 0.302B | 0.356B | 0.356B | 同 | 356M（表 2.1 印 350M） |
| Large | 0.679B | 0.760B | 0.760B | 同 | 760M |
| XL | 1.208B | 1.315B | **1.316B** | 1.517B | **1,320M** → 支持 $d_a=2048$ |
| 2.7B | 2.517B | 2.650B | 2.652B | 同 | 2,650M |
| 6.7B | 6.442B | 6.657B | 6.658B | 同 | 6,660M |
| 13B（$d=5140$ 如印） | 12.681B | 12.950B | 12.953B | 12.936B | 12,850M |
| 13B（$d=5120$） | `12*40*5120**2 + 52305*5120` | **12.851B** | | | **12,850M** → 支持 $d=5120$ |
| 175B | 173.946B | 174.589B | **174.604B** | 同 | **174,600M** |

结论：表 2.1 的 13B 行 $d_{model}=5140$ 是笔误，应为 5120（$=40\times128$，且与表 D.1 的 12,850M 精确吻合）；XL 行 “24 头 × 128” 与 $d_{model}=2048$ 矛盾，
参数量支持注意力宽度 2048（哪一项印错论文未说明，§8）。

### 4.3 每 token 前向 FLOPs

| 量 | 表达式 | 值 |
|---|---|---|
| GPT-2 small $2N$（Kaplan 近似 $N=12Ld^2$） | `2*12*12*768**2` | 169,869,312 ≈ 169.9 MFLOP |
| 每层权重矩阵乘精确 FLOPs | `2*(768*2304 + 768*768 + 2*768*3072)` | 14,155,776；×12 = 169.9M（与 $2\cdot12Ld^2$ 相同，因为二者都不含偏置） |
| 注意力项（Kaplan 口径）$2LTd$，$T=1024$ | `2*12*1024*768` | 18,874,368（占 $C_{fwd}$ 的 10.0%） |
| $C_{forward}=2N+2LTd$ | | 188,743,680 |
| De-embed $2dV$（Kaplan 表 1 单列，不计入 total） | `2*768*50257` | 77,194,752 |
| 含 unembedding 的前向 | `188743680 + 77194752` | 265,938,432 |
| 注意力项（PaLM/Chinchilla 口径）$4LTd$ | `4*12*1024*768` | 37,748,736 |
| 注意力与权重项相等的上下文长度 | $2LTd=24Ld^2\Rightarrow T=12d$ | GPT-2 small 9,216；GPT-3 175B 147,456（对应 Kaplan 的 $d_{model}>n_{ctx}/12$） |
| 注意力占比随 $T$（$d=768$，Kaplan 口径） | $T/(12d+T)$ | $T$=1024: 10.0%；8192: 47.1%；32768: 78.0%；131072: 93.4% |
| GPT-3 175B：$2N$ | `2*174.6e9` | 349.2 GFLOP |
| GPT-3 175B：$2LTd$，$T=2048$ | `2*96*2048*12288` | 4.83 GFLOP（占 1.36%） |
| 共享嵌入模型的巧合 | 精确 matmul FLOPs = $2\times$(非嵌入权重) + $2dV$ = `169869312+77194752` = 247,064,064；$2\times$(总参数−wpe) = `2*123653376` = 247,306,752 | 相差 0.1%（差的是偏置与 LN），所以“$2N_{total}$（不含位置表）≈ 真实矩阵乘 FLOPs”对共享嵌入的小模型比“$2N_{非嵌入}$”准得多 |

### 4.4 注意力数值

| 量 | 表达式 | 值 |
|---|---|---|
| $d_k=64$ 时点积方差/标准差（脚注 4 的假设下） | 理论 $d_k$；Monte Carlo 20 万次 | 63.77（理论 64），标准差 7.985（理论 8）；除以 $\sqrt{64}$ 后方差 0.996 |
| $d_k=128$ | 同上 | 128.09（理论 128），标准差 11.318（理论 11.314） |
| softmax 饱和：512 个 logit 服从 $\mathcal N(0,s^2)$ 时平均最大权重 | Monte Carlo 2 万次 | $s=1$：0.027；$s=8$（即 $d_k=64$ 不缩放）：0.729 |
| 同上，softmax 雅可比 $\mathrm{diag}(p)-pp^\top$ 的 Frobenius 范数均值 | 200 个样本 | $s=1$：0.0715；$s=8$：0.2494 |
| 温度示例 $z=[2,1,0.1]$，$\mathrm{softmax}(z/\tau)$ | numpy | $\tau=2$: [0.5017, 0.3043, 0.1940]；1: [0.6590, 0.2424, 0.0986]；0.5: [0.8638, 0.1169, 0.0193]；0.25: [0.9815, 0.0180, 0.0005] |
| 同一 $z$ 乘以 8（相当于不缩放、$d_k=64$） | | [0.9997, 0.0003, 0.0000] |
| 因果掩码示例：分数矩阵 $S=[[1,.5,-.3,2],[.2,1.1,.4,-1],[.3,0,.9,.5],[1.2,-.4,.1,.8]]$ 掩上三角后逐行 softmax | numpy | 行 1 [1,0,0,0]；行 2 [0.2891,0.7109,0,0]；行 3 [0.2807,0.2079,0.5114,0]；行 4 [0.4535,0.0916,0.1510,0.3040] |

说明：“softmax 雅可比范数变大”只说明**平均**情形；饱和时绝大多数行接近 one-hot，梯度几乎只流向一个位置——课文讲“梯度极小”时要区分
“对非最大项的梯度”与“整体范数”，避免与上表数字矛盾。可以画“最大权重随 $s$ 变化”的曲线（§5）。

### 4.5 GELU、位置编码、LN

| 量 | 表达式 | 值 |
|---|---|---|
| GELU(1) 精确 / tanh 近似 / sigmoid 近似 | `norm.cdf(1)`；`0.5*(1+tanh(sqrt(2/pi)*1.044715))`；`1/(1+exp(-1.702))` | 0.841345 / 0.841192 / 0.845796 |
| tanh 近似在 [-6,6] 上最大误差 | 网格 12 万点 | 4.73e-4（在 $x\approx-2.70$） |
| sigmoid 近似最大误差 | 同上 | 2.03e-2（在 $x\approx-2.27$） |
| GELU 最小值 | 网格 | −0.16997，在 $x\approx-0.752$ |
| GELU(−1)、GELU(2)、GELU(−3) | $x\Phi(x)$ | −0.15866、1.95450、−0.00405 |
| 正弦 PE 最短/最长波长（$d=512$） | $2\pi$；$2\pi\cdot10000^{510/512}$ | 6.283；60,611（极限 $2\pi\cdot10^4=62{,}832$） |
| $PE(pos=1)$ 前四维 | $\sin1,\cos1,\sin(10000^{-2/512}),\cos(\cdot)$ | 0.8415, 0.5403, 0.8219, 0.5697 |
| LN 每层参数 | $2d$ | GPT-2 small：1,536/个；每块 2 个 + 最终 1 个 → 12×3,072+1,536 = 38,400 |

### 4.6 其他（为第 4 课铺垫）

| 量 | 表达式 | 值 |
|---|---|---|
| GPT-2 small KV cache 每 token（fp16） | `2*12*768*2` | 36,864 B = 36 KiB |
| GPT-3 175B KV cache 每 token（fp16） | `2*96*12288*2` | 4,718,592 B = 4.5 MiB |
| Transformer base 参数复算（37,000 共享词表、三处共享、带偏置与 LN） | 编码器 `6*(4d²+4d+2d·dff+dff+d+4d)` + 解码器 `6*(8d²+8d+2d·dff+dff+d+6d)` + `37000*512` | 63,082,496（论文 65M；词表只说 “about 37000”，差距可由词表实际大小解释，未核实） |
| Transformer big 同法 | $d=1024$ | 214,245,376（论文 213M） |

### 4.7 BPE 手算

Sennrich 算法 1 原样运行（`bpe.py`，Python 的 `max` 遇到平局取先出现者）于 `{low:5, lower:2, newest:6, widest:3}`，前 10 次合并及频数：
`(e,s)9 → (es,t)9 → (est,</w>)9 → (l,o)7 → (lo,w)7 → (n,e)6 → (ne,w)6 → (new,est</w>)6 → (low,</w>)5 → (w,i)3`；
10 次后词表状态 `{'low</w>':5, 'low e r </w>':2, 'newest</w>':6, 'wi d est</w>':3}`。

图 1 的词典（各频数 1）`{low, lowest, newer, wider}` 前 4 次：`(l,o)2 → (lo,w)2 → (e,r)2 → (er,</w>)2`（平局很多，顺序依赖实现；论文图 1 的顺序
`r· → l o → lo w → e r·` 是另一种平局打破方式，同样合法）。论文说 OOV “lower” 会切成 “low er·”。

GPT-2 真实分词（`gpt2tok.py`：忠实移植 `encoder.py`，只把 `\p{L}`/`\p{N}` 换成 ASCII 字符类，所以只对纯 ASCII 文本等价）：

| 文本 | token | id |
|---|---|---|
| `Hello world` | `Hello`, `Ġworld` | 15496, 995 |
| `The unbelievable tokenization` | `The`, `Ġunbelievable`, `Ġtoken`, `ization` | 464, 24479, 11241, 1634 |
| ` lower newest widest` | `Ġlower`, `Ġnewest`, `Ġwidest` | 2793, 15530, 46232 |
| `DeepSeek` | `Deep`, `Se`, `ek` | 29744, 4653, 988 |

`Ġunbelievable` 的合并轨迹（rank 越小越先合并）：`(l,e)37 → (Ġ,u)78 → (a,b)141 → (e,l)161 → (i,e)238 → (ab,le)284 → (Ġu,n)299 → (b,el)6411 → (ie,v)10947
→ (Ġun,bel)17235 → (Ġunbel,iev)20298 → (Ġunbeliev,able)24223`。`Ġtokenization` 停在 `Ġtoken` + `ization`，因为没有 `(Ġtoken, ization)` 这条合并。
`Ġ` 是 `bytes_to_unicode` 把空格字节 0x20 映射成的可见字符（U+0120）。merges 表第一条是 `Ġ t`（rank 0）。

---

## 5. 图表清单与重绘建议

### 5.1 论文原图清单（与本课相关者）

| 来源 | 图号 | 标题/内容 | 是否重绘 |
|---|---|---|---|
| Vaswani | 图 1 | “The Transformer - model architecture”，编码器–解码器全图 | 重绘成**仅解码器**的 GPT 版（去掉交叉注意力），并标 Post-LN |
| Vaswani | 图 2 | 左：缩放点积注意力（MatMul→Scale→Mask(opt.)→SoftMax→MatMul）；右：多头 | 重绘，按课程颜色（Q 蓝、K 紫、V 绿、掩码红） |
| Vaswani | 图 3–5 | 第 5 层（共 6 层）编码器自注意力可视化：长距离依赖 ‘making…more difficult’、指代消解（‘its’，头 5、6）、结构相关的头 | 不重绘，可文字引用 |
| GPT-1 | 图 1 | 左：Transformer 结构与训练目标（Text & Position Embed → 12× 块 → Text Prediction / Task Classifier）；右：各任务的输入变换 | 左半边可作为 GPT-1 结构参考重绘 |
| GPT-2 | 图 1–4 | 零样本性能随模型大小；CBT；Winograd；WebText 训练/测试困惑度随大小 | 不重绘（第 2 课可提） |
| Xiong | 图 1 | (a) Post-LN 层；(b) Pre-LN 层 | **重绘**（本课核心对比图） |
| Xiong | 图 3 | 6-6 Transformer 各层 $W^1,W^2$ 梯度范数；不同层数下 $W^{2,L}$ 梯度范数（Pre vs Post） | 只取定性结论，数据不可得，不重绘数值 |
| Sennrich | 图 1 | 玩具词典的 4 次合并 | 重绘为“合并轨迹”图，建议用 §4.7 的算法 1 词典（可复算） |
| Geva | 图 1 | FFN 模拟键值记忆：输入乘键得记忆系数（例 0.2），加权值上的词表分布 | 重绘成课程版“键值记忆”示意 |
| Elhage | “High-Level Architecture”图 | 残差流 + 各层读写 | 用 style-guide 类比 6“传送带”重画，不照抄 |
| Kaplan | 表 1 | 参数与 FLOPs 计数表 | 转成课程记号的表格图 |

### 5.2 本课建议绘制的 14 张图（每张给出绘图所需的全部数值）

1. **图 1.1 GPT-2 small 整体架构（仅解码器、Pre-LN）。** 自下而上：token id → wte(50257×768) + wpe(1024×768) → 12× [LN₁ → MHA(12 头×64) → ⊕ → LN₂ → FFN(768→3072→768, GELU) → ⊕] → LN_f → ×wteᵀ → logits(50257)。
   残差主干用 `--fig-ink` 粗线，分支用细线。标注“共享”虚线连接 wte 与输出层。
2. **图 1.2 BPE 合并轨迹。** 左：Sennrich 算法 1 词典四个词（频数 5/2/6/3）逐步合并的前 5 步（数据 §4.7）；右：GPT-2 对 `Ġunbelievable` 的 12 次合并阶梯（rank 值标在箭头上）。
3. **图 1.3 嵌入查表与位置编码热力图。** 正弦 PE：pos 0–127 × 维度 0–127（$d=512$ 下的前 128 维），值域 [−1,1]，用发散色；旁边小图画第 $m$ 对维度的波长 $2\pi\cdot10000^{2m/512}$（已复算：$m$=0: 6.283；10: 9.004；50: 37.97；100: 229.4；200: 8,378.8；255: 60,611），纵轴用对数。
4. **图 1.4 缩放点积注意力计算图（单头）。** 形状：$Q,K,V\in\R^{T\times64}$，$QK^\top\in\R^{T\times T}$，除以 8，加掩码，softmax，乘 $V$ 得 $T\times64$。颜色：Q 蓝、K 紫、V 绿、掩码红。
5. **图 1.5 为什么除以 $\sqrt{d_k}$。** 左：$q\cdot k$ 的分布（$d_k=64$：均值 0、标准差 8 vs 缩放后 1）的两条高斯密度；右：512 个 logit 的平均最大 softmax 权重随标准差 $s\in[0.25,16]$ 变化（$s=1$ 处 0.027，$s=8$ 处 0.729；需在 Python 中按 §4.4 方法跑一条曲线）。
6. **图 1.6 softmax 温度。** $z=[2,1,0.1]$ 在 $\tau=2,1,0.5,0.25$ 下的柱状图（数值 §4.4）。
7. **图 1.7 因果掩码矩阵。** 4×4（或 6×6）：上三角红色 “−∞”，下三角是 §4.4 示例的权重（行和为 1）。附一句：训练时 $T$ 个位置一次前向并行算出。
8. **图 1.8 多头拆分与 $W^O$。** 768 维向量切成 12 段 × 64；每头独立注意力；拼接后乘 $W^O$(768×768)；同时画 Elhage 的等价形式（12 个 $W^O_i$ 各自写回、再求和）。
9. **图 1.9 FFN 与 GELU。** 左：768→3072→768 的“键值记忆”示意（Geva 式 (1)）；右：GELU 精确、tanh 近似、ReLU 在 [−4,4] 的曲线，标出最小值点 (−0.752, −0.170)。
10. **图 1.10 残差流 = 传送带。** 12 个工位（每块两个：注意力、FFN）从主干读（经 LN）、把增量加回；标注“残差流 = 嵌入 + 所有层输出之和”（Elhage 原句，§1 C32）。
11. **图 1.11 Post-LN vs Pre-LN。** 两列并排，按 Xiong 表 1 的步骤编号 1–5；下方小表：定理 1 的两个梯度上界、引理 2 的隐藏状态范数（Post 恒为 $1.5d$；Pre 在 $(1+l/2)d$ 与 $(1+3l/2)d$ 之间，画 $l=0..12$ 的带状区域，$d$ 归一化为 1）。
12. **图 1.12 参数都去哪儿了（GPT-2 small）。** 堆叠条：wte 38.60M、wpe 0.79M、注意力 12×2.362M=28.35M、FFN 12×4.722M=56.67M、LN 0.038M；总 124.44M。旁边四个规模的嵌入占比条（31.6%、14.8%、8.5%、5.3%）。
13. **图 1.13 形状追踪表格图。** 直接用 §3.5 表，按步骤着色（注意力步骤蓝紫绿，FFN 灰，残差黑）。
14. **图 1.14 每 token 前向 FLOPs 分解（GPT-2 small，$T=1024$）。** 权重矩阵乘 169.9M、unembedding 77.2M、注意力 18.9M（Kaplan 口径；另用虚线框标 PaLM 口径 37.7M）；右侧小图：注意力占比 $T/(12d+T)$ 随 $T$ 从 128 到 131072 的曲线，交叉点 $T=9216$。
15. （可选）**图 1.15 $12Ld^2$ 近似 vs 实际。** 散点：横轴 $12Ld^2$，纵轴精确参数；GPT-2 四个（总参数与非嵌入两组点）+ GPT-3 八个（用 §4.2 的数），对角线参考。

---

## 6. 算例与练习素材（附答案）

1. **（计算）GPT-2 small 精确参数量。** 给 $L=12,d=768,V=50257,T_{max}=1024$、带偏置、Pre-LN、共享嵌入、有最终 LN。
   答：$(50257+1024)\cdot768+12(12\cdot768^2+13\cdot768)+2\cdot768=124{,}439{,}808$。其中非嵌入 85,056,000；若不共享输出矩阵再加 38,597,376 → 163,037,184。
2. **（侦探）“117M”从哪里来？** 用 GPT-1 的词表 40,478 代替 50,257 重算四个 GPT-2 规模。
   答：116.93M、344.81M、761.51M、1541.97M，四舍五入恰为报告表 2 的 117/345/762/1542。结论须标“笔者推测”，官方只说有错。
3. **（计算）$12Ld^2$ 能差多少？** GPT-3 175B：$12\cdot96\cdot12288^2=173.95$B；加 $(50257+2048)\cdot12288=0.643$B 得 174.59B；表 D.1 为 174,600M。
   GPT-2 small：$12Ld^2=84.9$M，但总参数 124.4M——小模型里嵌入占 31.6%，“$12Ld^2$”严重低估总数。
4. **（核对）GPT-3 13B 的 $d_{model}$。** 表 2.1 印 5140、40 头 × 128。问：哪个数与表 D.1 的 12,850M 一致？
   答：$d=5120$ 给出 12,850.7M（精确到百万位吻合），$d=5140$ 给出 12,950M；且 $40\times128=5120$。5140 是笔误。
5. **（推导）$\sqrt{d_k}$。** 设 $q_i,k_i$ 独立、均值 0、方差 1，求 $\Var(\mathbf q^\top\mathbf k)$；$d_k=128$ 时标准差多少？
   答：$\Var=\sum_i\E[q_i^2]\E[k_i^2]=d_k$；标准差 $\sqrt{128}=11.31$。除以 $\sqrt{d_k}$ 后方差回到 1。
6. **（计算）每 token 前向 FLOPs。** GPT-2 small，$T=1024$，Kaplan 口径。
   答：$2N+2LTd=169.87\text{M}+18.87\text{M}=188.74$M；再加 unembedding $2dV=77.19$M 得 265.94M。问学生：为什么 Kaplan 的 total 不含 $2dV$？（因为他们的 $N$ 定义就不含嵌入，式 (2.1)。）
7. **（推导）什么时候注意力项追上权重项？** 令 $2LTd=2\cdot12Ld^2$，得 $T=12d$。GPT-2 small：9,216；GPT-3：147,456。这就是 Kaplan “$d_{model}>n_{ctx}/12$” 条件的来历。
8. **（概念）掩码为什么让训练可以并行？** 答要点：一次前向对 $T$ 个位置同时算出 $T$ 个下一词分布；上三角 $-\infty$ 保证位置 $t$ 看不到 $>t$；
   GPT-2 代码用 $-10^{10}$ 而非 $-\infty$，softmax 后权重为 $e^{-10^{10}}\approx0$。
9. **（计算）softmax 温度。** $z=[2,1,0.1]$，求 $\tau=1$ 与 $\tau=0.25$ 的分布。答：[0.659, 0.242, 0.099]；[0.9815, 0.0180, 0.0005]。
   追问：乘以 8 相当于 $\tau=1/8$，得 [0.9997, 0.0003, 0.0000]——这就是“不缩放 $d_k=64$”的效果。
10. **（计算）GELU。** 求 GELU(1)、GELU(−1) 和 tanh 近似在 $x=1$ 的误差。答：0.84134、−0.15866；误差 $1.5\times10^{-4}$（0.841345−0.841192）。
11. **（手算）BPE。** 对 `{low:5, lower:2, newest:6, widest:3}` 做 3 次合并。答：(e,s) 9 次 → (es,t) 9 次 → (est,</w>) 9 次；此时 “newest” 变成 `n e w est</w>`。
12. **（计算）LN 与残差不增加多少参数。** GPT-2 small 的所有 LN 参数：$25\times1536=38{,}400$，占总数 0.031%；残差连接 0 参数（He et al. 原话“neither extra parameter nor computational complexity”）。
13. **（预告第 4 课）KV cache。** GPT-2 small fp16 每 token $2\cdot12\cdot768\cdot2=36{,}864$ B；1024 个 token 为 36 MiB。GPT-3 175B 每 token 4.5 MiB。

---

## 7. 勘误与常见误解

- **E1 “$\sqrt{d_k}$ 的方差论证是原始论文就有的。”** 不对。v1（2017-06-12）只有“点积太大、梯度无用”的猜测；方差论证是 v2（2017-06-19）加的**脚注 4**。
  arXiv HTML 把它显示成脚注 1，引用时写 PDF 编号 4。（与 mamba 课程 lesson-01-doc 勘误 12 一致。）
- **E2 “脚注证明了缩放是必要的。”** 脚注只算了一个方差，前提是分量独立、单位方差；它解释“为什么点积会大”，不证明缩放最优。
  论文自己的用词是 “We suspect”。
- **E3 “Transformer 论文用的是 Pre-LN。”** 论文 §3.1 明写 $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$（Post-LN）；但 tensor2tensor 的默认 `transformer_base`（v2 起）是 Pre-LN。
  论文与官方代码不一致，按版本说。
- **E4 “Pre-LN 是 GPT-2 发明的 / 是 Xiong 提出的。”** GPT-2 报告自己说是仿照 pre-activation ResNet；Xiong 说 Pre-LN 已在 tensor2tensor、OpenNMT 等系统中实现。
  最早出处本轮未能确定（§8）。
- **E5 “GPT-2 small 是 117M 参数。”** 报告表 2 印 117M，但官方仓库承认计数错误，现名 124M；精确值 124,439,808。117M 是 GPT-1 的规模
  （复算 116.5M），GPT-2 报告说最小模型 “equivalent to the original GPT”，指结构规模而非参数精确相等（词表和上下文都变了）。
- **E6 “GPT-2 官方代码里有 1/√N 初始化。”** `openai/gpt-2` 的 `model.py` 是推理代码，所有 `conv1d` 权重都是 `stddev=0.02`，没有 1/√N；
  1/√N 只在报告文字里。HF/nanoGPT 按 $N=2L$ 实现，属于对“residual layers”的解读。
- **E7 “GPT-2 的最终 LN 在最后一个自注意力之后。”** 报告字面如此（“after the final self-attention block”），但代码的 `ln_f` 在全部 12 个块（含最后的 MLP）之后、输出层之前。
  以代码为准：最终 LN 在整个堆栈之后。
- **E8 “GPT 也把嵌入乘 $\sqrt{d_{model}}$。”** 那是 Vaswani §3.4 的做法；GPT-2 代码直接 `wte[X] + wpe[pos]`，不缩放。
- **E9 Sennrich 论文里算法 1 与图 1 用的是两个不同词典。** 算法 1：{low, lower, newest, widest}（带频数）；图 1 标题：{‘low’, ‘lowest’, ‘newer’, ‘wider’}。
  很多教程把图 1 的合并结果当成算法 1 的输出，二者对不上。手算例请用算法 1 词典（可复算，§4.7）。
- **E10 “GPT-3 的词表是 50,257”写成 GPT-3 论文的陈述。** GPT-3 论文没写词表大小，只说沿用 GPT-2 的可逆分词。
- **E11 GPT-3 表 2.1 有两处内部矛盾。** 13B 行 $d_{model}$ 印 5140（应为 5120）；XL 行 24 头 × 128 = 3072 ≠ 2048，参数量支持注意力宽度 2048。转录表格时加注，不要静默改正。
- **E12 “GPT-2 是 ICML 2019 论文。”** PDF 的元数据 Subject 是 ICML 模板残留；它是 OpenAI 技术报告（以本轮所见，无正式会议版本，§8）。
- **E13 Vaswani EN-FR BLEU。** 41.8（v5 起摘要与表 2）vs 41.0（v1、v4 全文；v5–v7 的 §6.1 正文）；中间的 v2 摘要写 41.2、v2/v3 的 §6.1 写 41.17（见 C12 表）。
  引用时说明版本。
- **E14 Kaplan 表 1 的 “$d_{embd}$”。** Attention: Project 行 FLOPs 印作 $2n_{layer}d_{attn}d_{embd}$，应为 $d_{model}$；且表中 Mask 行用 $d_{attn}$、式 (2.2) 用 $d_{model}$。转录保留原样并加注。
- **E15 注意力 FLOPs 的“2 倍之谜”。** Kaplan 的注意力项 $2LTd$ 是 PaLM/Chinchilla/DeepSeek LLM 口径 $4LTd$ 的一半。课文给公式时必须说用的是哪家口径。
- **E16 “$6N$ 里的 $N$ 就是模型参数量。”** Kaplan 的 $N$ 是**非嵌入**参数，而且 $C_{forward}$ 不含 unembedding 的 $2dV$；对 GPT-2 small，这一项占前向 FLOPs 的 29%（77.2/265.9）。
- **E17 “FFN 为什么是 4 倍：因为理论最优。”** 没有论文给出理由：Vaswani 只给数值 2048=4×512；GPT-3 说“always … four times”；Kaplan 称其为 “standard”。它是惯例。
  （第 5 课 SwiGLU 的 $\tfrac83d$ 也是在“保持参数量不变”的约束下由惯例推出。）
- **E18 “LayerNorm 原式里有 $\epsilon$。”** Ba et al. 式 (3)(4) 没有 $\epsilon$；$\epsilon=10^{-5}$ 是实现细节（GPT-2 代码、HF config）。
- **E19 GPT-1 式 (2) 与 GPT-2 式 (1) 的下标笔误。** GPT-1：$h_l=\mathrm{transformer\_block}(h_{l-1})\,\forall i\in[1,n]$（量词应为 $l$）；GPT-2 式 (1) 见第 2 课笔记。转录时保留原样并注明。
- **E20 “注意力头越多越好。”** Vaswani 表 3 (A)：$h=16$ 与 $h=8$ 持平（BLEU 25.8），$h=32$ 降到 25.4；作者原话 quality drops off with too many heads。

---

## 8. 未能核实 / UNVERIFIED

1. **Pre-LN 的最早出处。** Xiong 列了 Vaswani et al. 2018（tensor2tensor）、Klein et al. 2018、Liu et al. 2019b、Baevski & Auli 2018、Child et al. 2019、Wang et al. 2019，
   本轮没有逐篇核对谁最早；tensor2tensor 中 `transformer_base_v2` 引入的**日期**也未核实。
2. **GPT-2 报告表 2 参数量错误的真实原因。** “用 GPT-1 词表 40,478 计数”的假说四个数全中，但无官方说明。
3. **GPT-1 词表 40,478 = 40,000 + 478 的构成**（478 个基础符号/特殊符号？）未核实；报告只说 40,000 merges。
4. **GPT-1 的 LN 位置。** 报告未写；HF 实现为 Post-LN，未与 OpenAI 原始 `finetune-transformer-lm` 代码逐行比对。
5. **GPT-3 局部带状稀疏注意力的带宽、在哪些层使用**：论文未给。
6. **GPT-3 表 2.1 XL 行到底是头数还是每头维度印错**（24×128 vs $d=2048$）。参数量只能说明注意力宽度是 2048。
7. **GPT-2 是否在任何会议正式发表**：只核对了 PDF 元数据与 GPT-3 参考文献 [117]（写作 “2019”，无会议）。
8. **Kaplan 图 1 图内拟合式的常数**（图中标注的 $L=(C_{min}/\cdot)^{-0.050}$ 等）：图为矢量路径，本机无 PDF 渲染器，无法读出；课文只用式 (1.1)–(1.3) 与表 4/5 的数值。
9. **Kaplan 的注意力 FLOPs 为何是 $2n_{layer}n_{ctx}d_{attn}$**（是否因果平均）：论文未解释。
10. **Transformer base 65M 的精确计数**：按 37,000 共享词表复算 63.1M；实际 BPE 词表大小未知（“about 37000”）。
11. **Xiong et al. 的 ICML 2020 论文集页码/卷号**：只有 arXiv jref “Published on ICML 2020”，未在 PMLR 核对。
12. **Vaswani 的 NIPS 页码 6000–6010** 只来自 GPT-1 的参考文献条目，未在 NeurIPS 论文集核对。
13. **GELU 的“随机正则化期望”解释是否是 GPT 选用 GELU 的原因**：GPT-1 报告只写了“用了 GELU”，没有给理由。
14. **Elhage 页面的“High-Level Architecture”图的确切编号/标题**：网页图无编号，本轮只核对了正文句子。
15. **GPT-1 原始训练代码用的 GELU 是 tanh 近似还是 erf 精确版**：HF 移植映射到 tanh 近似（`gelu_new`），OpenAI 原始代码未读。

---

## 9. 参考文献卡片草稿

```
- id: vaswani2017
  title: Attention Is All You Need
  a: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017), NIPS 2017, arXiv:1706.03762 (read v7; v1–v5 PDFs for drift)
  url: https://arxiv.org/abs/1706.03762
  finding: 提出只用注意力的编码器–解码器，定义缩放点积注意力、多头注意力、逐位置 FFN、正弦位置编码与 Post-LN 残差结构。
  matters: 本课所有组件的原始定义与 √d_k 脚注的出处，也是“论文 Post-LN、代码 Pre-LN”版本漂移的起点。

- id: radford2018
  title: Improving Language Understanding by Generative Pre-Training
  a: Radford, Narasimhan, Salimans, Sutskever (2018), OpenAI technical report, no arXiv
  url: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
  finding: 12 层仅解码器 Transformer（768 维、12 头、3072 FFN、512 上下文、GELU、可学习位置、BPE 4 万次合并）在 BooksCorpus 上预训练后微调。
  matters: 课文“GPT 结构”的第一个具体实例，式 (2) 直接展示了输出层复用输入嵌入矩阵。
  kind: artefact

- id: radford2019
  title: Language Models are Unsupervised Multitask Learners
  a: Radford, Wu, Child, Luan, Amodei, Sutskever (2019), OpenAI technical report, no arXiv
  url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  finding: 把 LN 移到每个子层输入、加最终 LN、残差层初始化乘 1/√N、词表扩到 50,257、上下文扩到 1024，训练 117M–1542M（印刷值）四个规模。
  matters: 本课形状追踪与参数量核对的样板模型，也是 Pre-LN 进入 GPT 谱系的节点。
  kind: artefact

- id: gpt2code
  title: openai/gpt-2 repository (README, src/model.py, src/encoder.py, DEVELOPERS.md)
  a: OpenAI (2019), GitHub, archived
  url: https://github.com/openai/gpt-2
  finding: 官方推理代码给出 Pre-LN 块、融合 QKV、−1e10 掩码、tanh-GELU、共享输出矩阵，README 承认报告中的参数量有误并改称 124M/355M/774M/1558M。
  matters: 本课“论文 vs 代码”核对的依据，也是 1/√N 初始化只存在于报告文字的证据。
  kind: artefact

- id: hfgpt2
  title: Hugging Face openai-community/gpt2* configs, vocab/merges files, transformers modeling_gpt2.py
  a: Hugging Face (accessed 2026-09-24)
  url: https://huggingface.co/openai-community/gpt2
  finding: 四个 GPT-2 规模的 n_layer/n_embd/n_head/vocab 配置，merges 恰为 50,000 条，词表 50,257，残差投影初始化 std=0.02/√(2L)。
  matters: 头数、ε、激活函数等报告没写的工程事实只能从这里取。
  kind: artefact

- id: brown2020
  title: Language Models are Few-Shot Learners
  a: Brown, Mann, Ryder, Subbiah, Kaplan, Dhariwal, et al. (2020), NeurIPS 2020, arXiv:2005.14165 (read v4)
  url: https://arxiv.org/abs/2005.14165
  finding: 175B 模型沿用 GPT-2 架构，仅改为稠密与局部带状稀疏注意力交替，8 个规模的形状列于表 2.1，FFN 恒为 4 倍、上下文 2048。
  matters: 用来检验 12Ld² 近似并暴露表 2.1 的两处笔误（13B 的 5140、XL 的头宽）。

- id: kaplan2020
  title: Scaling Laws for Neural Language Models
  a: Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei (2020), arXiv preprint, arXiv:2001.08361 (v1)
  url: https://arxiv.org/abs/2001.08361
  finding: 表 1 给出 Transformer 各部分参数与每 token 前向 FLOPs，得出 N≈12 n_layer d_model² 与 C_forward≈2N+2 n_layer n_ctx d_attn。
  matters: 本课参数量与 FLOPs 公式的标准出处，且其注意力项口径与 PaLM/DeepSeek 相差 2 倍，需要课文点明。

- id: sennrich2016
  title: Neural Machine Translation of Rare Words with Subword Units
  a: Sennrich, Haddow, Birch (2016), ACL 2016, arXiv:1508.07909 (read v5)
  url: https://arxiv.org/abs/1508.07909
  finding: 把压缩算法 BPE 改造成子词切分：从字符出发反复合并最频繁的符号对，合并次数是唯一超参数。
  matters: 课文 BPE 手算例的来源，且其算法 1 与图 1 用了不同词典，手算时必须选算法 1 的那组。

- id: hendrycks2016
  title: Gaussian Error Linear Units (GELUs)
  a: Hendrycks, Gimpel (2016), arXiv preprint, arXiv:1606.08415 (read v5; v1 title "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units")
  url: https://arxiv.org/abs/1606.08415
  finding: GELU(x)=xΦ(x)，可看作随机正则的期望，并给出 tanh 与 1.702-sigmoid 两种近似。
  matters: 解释 GPT 的 FFN 非线性，以及 GPT-2 代码里那行 tanh 公式的来源。

- id: ba2016
  title: Layer Normalization
  a: Ba, Kiros, Hinton (2016), arXiv preprint, arXiv:1607.06450 (v1)
  url: https://arxiv.org/abs/1607.06450
  finding: 在同一层全部隐藏单元上计算均值与标准差并做逐元素仿射，训练与测试计算相同。
  matters: 本课“LN = 尺度控制”一节的公式来源，并说明 ε 是实现添加的。

- id: he2016
  title: Deep Residual Learning for Image Recognition
  a: He, Zhang, Ren, Sun (2016), CVPR 2016, arXiv:1512.03385 (v1)
  url: https://arxiv.org/abs/1512.03385
  finding: 让层拟合残差 F(x)=H(x)−x 并加恒等捷径 y=F(x)+x，不增参数即可训练很深的网络。
  matters: 残差流观点和第 16 课 mHC 的起点，课文“梯度高速公路”需要它的式 (1)。

- id: xiong2020
  title: On Layer Normalization in the Transformer Architecture
  a: Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, Liu (2020), ICML 2020, arXiv:2002.04745 (read v2)
  url: https://arxiv.org/abs/2002.04745
  finding: 在简化假设下证明初始化时 Post-LN 末层梯度为 O(d√ln d)、Pre-LN 为 O(d√(ln d/L))，并实验表明 Pre-LN 可去掉 warm-up。
  matters: 给 Post-LN vs Pre-LN 对比图提供定理与引理数字，同时提醒课文 Pre-LN 并非 GPT-2 首创。

- id: press2017
  title: Using the Output Embedding to Improve Language Models
  a: Press, Wolf (2017), EACL 2017, arXiv:1608.05859 (read v3)
  url: https://arxiv.org/abs/1608.05859
  finding: 输出嵌入本身是有效词向量，把输入与输出嵌入绑定可降低语言模型困惑度，并能把翻译模型体积减半以上。
  matters: weight tying 的引用出处（Vaswani §3.4 引的 [30]），也支撑课文“共享省下 38.6M 参数”的算例。

- id: geva2021
  title: Transformer Feed-Forward Layers Are Key-Value Memories
  a: Geva, Schuster, Berant, Levy (2021), EMNLP 2021, arXiv:2012.14913 (read v2)
  url: https://arxiv.org/abs/2012.14913
  finding: FFN 可写成 f(x·Kᵀ)·V 的键值记忆，键对应可解释的输入模式，值在词表上诱导出下一词分布。
  matters: 课文“FFN = 键值记忆”一段的数学与实证依据，也是第 17 课 Engram“条件记忆”的对照。

- id: elhage2021
  title: A Mathematical Framework for Transformer Circuits
  a: Elhage, Nanda, Olsson, Henighan, Joseph, Mann, et al. (2021), Transformer Circuits Thread (Anthropic), no arXiv
  url: https://transformer-circuits.pub/2021/framework/index.html
  finding: 把残差流看作所有层读写的线性通信信道，残差流等于嵌入与各层输出之和且没有特权基，各注意力头独立相加。
  matters: 课文“残差 = 增量更新 / 传送带”观点的出处，并为第 16 课超连接铺路。
  kind: artefact

- id: t2t
  title: tensor2tensor models/transformer.py and layers/common_hparams.py
  a: Google (accessed 2026-09-24), GitHub
  url: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  finding: basic_params1 与 transformer_base_v1 为 Post-LN（"none"/"dan"），transformer_base_v2 起改为 Pre-LN（"n"/"da"），默认 transformer_base 继承 v2。
  matters: 证明 Transformer 的官方代码与论文在 LN 位置上不一致，课文讲 Post/Pre-LN 时要按版本说。
  kind: artefact

- id: palm2022
  title: PaLM: Scaling Language Modeling with Pathways
  a: Chowdhery, Narang, Devlin, et al. (2022), arXiv:2204.02311 (read v5, Appendix B only)
  url: https://arxiv.org/abs/2204.02311
  finding: 训练 FLOPs 按 6N 加上注意力 matmul 的 6LH(2QT)，即 6N+12LHQT，并据此定义 MFU。
  matters: 提供与 Kaplan 不同的注意力 FLOPs 口径，DeepSeek LLM 的 M 公式与之一致。

- id: nanogpt
  title: karpathy/nanoGPT model.py
  a: Karpathy (accessed 2026-09-24), GitHub
  url: https://github.com/karpathy/nanoGPT/blob/master/model.py
  finding: 社区参考实现按 0.02/√(2L) 初始化残差投影、共享 wte 与 lm_head，并用 6N+12LHQT 估算 MFU。
  matters: 旁证 HF 对 1/√N 的解读是社区通行做法，且给出从 HF Conv1D 到 nn.Linear 的转置约定。
  kind: artefact
```

---

## 10. 核查记录（对抗式复核，2026-09-24）

复核方法：从 §1–§7 选出 20 组最可能被课文复述的论断，**不看本文件的结论**，重新从一手来源取证：arXiv LaTeXML 全文（`arxiv.py text/grep`）、
Vaswani **v1–v7 全部七个 PDF**（`pdftotext`）、GPT-1/GPT-2 OpenAI PDF（`pdftotext` + `pypdf` 查特殊符号）、GitHub raw（openai/gpt-2、tensor2tensor、
nanoGPT、transformers）、Hugging Face raw（config/merges/vocab）、Chinchilla TeX 源码、arXiv 摘要页的提交历史。所有派生数字在 Python 里重算。

### 10.1 改动

| # | 位置 | 原文 | 改为 | 依据 |
|---|---|---|---|---|
| 1 | C12、E13 | “v1–v4 的摘要与表 2 写 41.0” | 七版本对照表：v2 摘要 **41.2**（并写 “4.5 days”），v2/v3 §6.1 写 **41.17** 且多一句 EN-FR big “did not share source and target embeddings, dff = 8192”；v4 回到 41.0；v5–v7 摘要/表 2 为 41.8、§6.1 仍 41.0 | 1706.03762 v1–v7 PDF，`pdftotext` 后 grep “BLEU score of”“28.4”“dff = 8192” |
| 2 | C29 定理假设 | 缺一条假设 | 补上“输入向量也从同一高斯分布采样（其余权重 Xavier）”，并注明 $(\epsilon,\delta)$-有界是定理 1 自身的条件 | 2002.04745 v2 §3.3 “we study a simpler setting” 段第三条；定理 1 陈述 |

### 10.2 复核后确认无误（节选关键证据）

1. **C2 √d_k 脚注**：v1 §3.2.1 只有 “We suspect this to be caused by the dot products growing too large…”；v2–v7 均为脚注 **4**，含“independent random variables with mean 0 and variance 1”。✓
2. **C11 / §3.1 表 3（v7）**：base 65M、big 213M；(A) 单头 5.29/24.9、$h=32$ 5.01/25.4；(C) $d_{ff}$ 1024 → 5.12/25.4、4096 → 4.75/26.2；(E) 4.92/25.7；§5.2 0.4 s/100K 步/12 h，1.0 s/300K 步/3.5 天；Adam(0.9, 0.98, 1e-9)、warmup 4000。✓
3. **C9、C8、C10**：v7 “LayerNorm(x + Sublayer(x))”，波长 “geometric progression from 2π to 10000·2π”，“nearly identical results”，表 1 复杂度各项。✓
4. **Vaswani 版本日期**：arXiv 提交历史 v1 2017-06-12 … v7 2023-08-02，与 §0 一致。✓
5. **C13 / E5 GPT-1 §4.1**：12 层、768、12 头、3072、512 tokens、2.5e-4、2000 updates、100 epochs × 64、N(0, 0.02)、40,000 merges、dropout 0.1、w = 0.01、GELU、learned position、ftfy/spaCy；式 (2) 的 “∀i∈[1,n]” 笔误（`pypdf`）；报告全文无参数总数。✓
6. **C16/C17 GPT-2 §2.3 与表 2**：四句原文全部核到；`pypdf` 抽出 “1/\n√\nN”，根号确实存在；表 2 为 117M/345M/762M/1542M、12/24/36/48、768/1024/1280/1600；§3 “The smallest model is equivalent to the original GPT”。✓
7. **C18**：README 原句 “our original parameter counts were wrong due to an error (in our previous blog posts and paper)”；DEVELOPERS.md 为 124M/355M/774M/1558M。✓
8. **C20 `model.py`**：Pre-LN 块、`c_attn` 融合 QKV、`rsqrt(v.shape[-1])`、`1e10` 掩码、`wpe` 0.01 / `wte` 0.02、`conv1d` 默认 0.02 且全文无 $1/\sqrt N$、`epsilon=1e-5`、tanh-GELU、`ln_f` 在全部块之后、`logits = matmul(h, wte, transpose_b=True)`、`past_shape`。✓（E6、E7、E8 随之成立）
9. **HF 配置与词表**：`n_head` 12/16/20/25、`gelu_new`、1e-5、0.02、50257；`openai-gpt` 为 40478、`afn: gelu`、`n_positions` 512；`merges.txt` 共 50,001 行 = 版本头 + 50,000 条；`vocab.json` 50,257 项，50256 = `<|endoftext|>`。✓
10. **C21、C15**：HF `std = initializer_range / math.sqrt(2 * n_layer)`，注释 “2 residual paths”；nanoGPT `0.02/math.sqrt(2 * config.n_layer)`、`transposed` 列表、`6*N + 12*L*H*Q*T`；HF `modeling_openai` 为 `ln_1(x + a)`、`ln_2(n + m)`，`"gelu": gelu_new`。✓
11. **§4.1 参数量**：124,439,808 / 354,823,168 / 774,030,080 / 1,557,611,200；非嵌入 85,056,000 等；GPT-1 116,534,784；“GPT-1 词表”假说 116.930/344.809/761.513/1541.965M。全部重算一致。✓
12. **§3.4 / §4.2 GPT-3**：表 2.1 原样印 13B 行 5140、XL 行 24 头 × 128；表 D.1 125/356/760/1,320/2,650/6,660/12,850/174,600；重算 $d=5120$ 得 12.853B、$d=5140$ 得 12.953B，XL $d_a=2048$ 得 1.316B、$d_a=3072$ 得 1.517B。✓
13. **E13 Kaplan 表 1、式 (2.1)(2.2)**：逐符号比对 LaTeXML，包括 Project 行的 $d_{embd}$、Mask 行 $d_{attn}$ vs 式 (2.2) $d_{model}$、标题 “Sub-leading terms … omitted”。✓
14. **E10、E11 Xiong**：表 1 左右两栏逐行一致；定理 1 概率 $0.99-\delta-\frac{\epsilon}{0.9+\epsilon}$ 与两个上界；引理 2 的 $\frac32d$ 与 $(1+\frac l2)d\le\cdot\le(1+\frac{3l}2)d$；引理 3；8.45 vs “around 34”；Pre-LN “implemented in several systems” 引 Vaswani 2018、Klein 2018（OpenNMT，AMTA 2018）、Liu 2019b。✓
15. **C34 tensor2tensor**：`basic_params1` 为 `"none"`/`"dan"`；`transformer_base_v2` 为 `"n"`/`"da"`；`transformer_base()` → `v3` → `v2`；v1 设 warmup 4000、beta2 0.98、共享嵌入。✓
16. **C35 FLOPs 口径**：Chinchilla 附录 F 的 QK logits 与 “Softmax @ query reductions” 各 $2T^2(d_hn_h)$、softmax $3n_hT^2$，“we also count embeddings matrices in the total parameter count”；PaLM 附录 B “$6LH(2QT)$”“$6N+12LHQT$”；DeepSeek LLM §3.2 的 $6N_1$、$6N_2$、$M$ 三式。✓
17. **C25、E9、§4.7 BPE**：算法 1 词典与图 1 词典 {low, lowest, newer, wider} 原文不同；把算法 1 原样运行 10 次，合并序列与 §4.7 完全一致。✓
18. **§4.3–§4.6 数值**：FLOPs 分解（169.87M / 18.87M / 77.19M / 265.94M，$T=12d$）、GELU 值与近似误差、PE 波长、softmax 温度与掩码示例、Monte Carlo（方差 64.2、最大权重 0.027/0.730）、KV cache、Transformer base/big 复算 63,082,496 / 214,245,376。✓
19. **C26、C30、C31、C32**：Hendrycks “we used the former in every experiment”；Press & Wolf “less than half”（摘要与 §1 (vi)）、$U=V$ 为 §1 (iii)；Geva “two-thirds”、$d_m\cdot16=65{,}536$；Elhage 五处引文与 “Published Dec 22, 2021”。✓
20. **§0 元数据**：GPT-2 PDF 元数据 Subject = “Proceedings of the International Conference on Machine Learning 2019”、CreationDate 2019-02-14；GPT-3 v1 2020-05-28、v4 2020-07-22、作者 “Tom B. Brown and 30 other authors”；GELU v1 2016-06-27、v5 2023-06-06；GPT-1 参考文献 [18] 用 GELU 旧标题。✓

### 10.3 本轮新增的未核实项

- Vaswani v2 摘要里的 “4.5 days” 与 “41.2” 是否对应另一套 EN-FR 配置（v2/v3 §6.1 的 $d_{ff}=8192$、不共享嵌入）：论文后续版本删掉了这些说法，没有解释。

---

## 11. 补充核实（第 1 课写作时，2026-09-24）

写课时**没有查阅新的外部来源**；课文中的事实性陈述全部来自本文件 §1–§10。下面记录的是写作中新增的三类内容：借用的参考文献卡片、自己复算的数字、以及课文里标为“笔者推导/推论”的论证。

### 11.1 借用的参考文献

- **hoffmann2022**（Chinchilla）的元数据取自 `lesson-02-doc.md` §9 的卡片（该笔记已核对：NeurIPS 2022，arXiv:2203.15556 v1）；课文只用它支撑“注意力 FLOPs 的 $4LTd$ 口径”，依据是本文件 C35 与 §10.2 第 16 条（附录 F 的 QK logits 与 softmax@V 各 $2T^2(d_hn_h)$）。

### 11.2 写作中复算的数字（Python 复算；生成图的脚本在 `deepseek/figures/gen/l01_*.py`，均带 assert）

| 用途 | 表达式 / 输入 | 值 |
|---|---|---|
| §3 手算注意力（自设数据） | $Q=[(1,1),(0,2),(2,1)]$，$K=[(0,1),(2,0),(1,1)]$，$V=[(1,0),(0,1),(1,1)]$，$d_k=2$ | $QK^\top=[[1,2,2],[2,0,2],[1,4,3]]$；权重行 2 $(0.8044,0.1956)$，行 3 $(0.0743,0.6200,0.3057)$；$\mathbf o_3=(0.3800,0.9257)$ |
| 同上，对照 | 行 3 不缩放；行 1 不加掩码 | $(0.0351,0.7054,0.2595)$；$(0.1978,0.4011,0.4011)$ |
| 同上，几何 | $\cos(\mathbf q_3,\mathbf k_2)$、$\cos(\mathbf q_3,\mathbf k_3)$ | 0.894、0.949 |
| 练习 5 | $\mathbf q_3=(0,2)$ | 权重 $(0.4458,0.1084,0.4458)$，$\mathbf o_3=(0.8916,0.5542)$ |
| §4 两个头的手算（教学复核时新增） | 同一组 $Q,K,V$ 拆成 2 个 $d_h=1$ 的头（头 1 取第 1 坐标、头 2 取第 2 坐标），第 3 行；$W^O=[[1,0.5],[-1,1]]$ | 头 1 分数 $(0,4,2)$，权重 $(0.0159,0.8668,0.1173)$，$o=0.1332$；头 2 分数 $(1,0,1)$，权重 $(0.4223,0.1554,0.4223)$，$o=0.5777$；$\mathbf u_3=(0.4220,0.4445)$ |
| 图 1.9 | §4.4 的 4×4 示例第 1 行不加掩码 | $(0.218,0.132,0.059,0.591)$，未来位置合计 78.2% |
| 图 1.8 | 20 万对 64 维标准正态向量（seed 0）；512 个 $\mathcal N(0,s^2)$ 分数 × 2000 次（seed 1） | 方差 63.77 / 缩放后 0.996；最大权重 $s=1$：0.027，$s=8$：0.726（§4.4 另一随机种子为 0.729，课文写“约 0.73”） |
| §5 FFN 玩具例 | $\mathbf k=[(1,0),(0,1),(-1,-1)]$，$\mathbf v=[(1,1),(-1,2),(3,0)]$，$\mathbf x=(1.5,0.5)$ | 系数 $(1.39979,0.34573,-0.04550)$；输出 $(0.91756,2.09125)$ |
| §6 LayerNorm 例 | $(2,4,6,8)$ 与 $(1,2,3,4)$ | 都得 $(-1.3416,-0.4472,0.4472,1.3416)$，长度 $2=\sqrt4$ |
| §2 / §7 输出层玩具例（图 1.4） | 5 词、$d=3$ 的 $E$，$\tilde{\mathbf h}=(-0.4,2.0,0.6)$ | logits $(-0.04,0.12,1.78,0.32,0.22)$，log-sum-exp 2.3647，$p=(0.090,0.106,0.557,0.129,0.117)$；$\tilde{\mathbf h}$ 加倍后 $p(\text{sat})=0.862$ |
| §4 softmax 雅可比 | 512 个 $\mathcal N(0,s^2)$ 分数，200 次（seed 0），$\norm{\mathrm{diag}(p)-pp^\top}_F$ 的均值 | $s=1$：0.0717；$s=8$：0.2369（§4.4 另一次为 0.0715 / 0.2494）。数学复核时改用 2000 次（seed 0）：$s=1$：0.0713，$s=8$：0.2473；200 次的 10 个种子在 0.218–0.265 之间波动，课文改写“2,000 组……约 0.25 对 0.07” |
| §6 | $0.9^{24}$；$0.02/\sqrt{24}$；$\sqrt{768}$ | 0.0798；0.00408；27.71 |
| §7 | GPT-2 xl 的 wte $50257\times1600$ | 80,411,200（总数的 5.16%） |
| §8 | PaLM 口径交叉点 $4LTd=24Ld^2$ | $T=6d$；GPT-2 small 4,608 |
| 练习 4 | GPT-2 medium 分项 | 嵌入 52,511,744；每块 12,596,224；不共享时 406,286,336 |
| 练习 6 | GPT-2 small，$T=512$ | 注意力 9,437,184（Kaplan）/ 18,874,368（PaLM）；合计 256,501,248；占比 5.26% |
| 练习 7 | GPT-3 175B，2048 token 的 KV cache（fp16） | 9,663,676,416 B = 9 GiB |

### 11.3 课文中标为“笔者推导 / 推论 / 推测”的论证

1. Kaplan 的注意力项 $2LTd$ 等于因果模型中每层 $4td$ 对 $t$ 取平均（本文件 §8 第 9 条：论文未说明）。
2. $1/\sqrt{2L}$ 初始化的动机：$2L$ 份独立增量之和的方差（GPT-2 报告只写了做法，未在本文件中核对其理由的原文）。
3. LayerNorm 在 $\boldsymbol\gamma,\boldsymbol\beta$ 之前的输出落在半径 $\sqrt d$ 的球面上（纯代数推导）。
4. Pre-LN 中单个子层增量占残差流的比例随深度变小（由引理 2 推出的直观推论，非 Xiong 等人的结论）。
5. 多层因果模型能从掩码形状间接获得一些位置信息（由方程组推出，未引文献）。练习 12 证明了其中可严格陈述的一半：没有位置编码时，第一层在位置 $t$ 的输出对前缀 $x_1,\dots,x_{t-1}$ 的排列不变；第二层起不再成立。
6. “117M 来自 GPT-1 词表”的假说（§4.1 已记为推测），课文作为练习 8 并标“推测”。
7. 写作中删去的一处未核实说法：“DeepSeek 在 2019 年的语料里不存在所以被切成三段”，改为只陈述分词结果。

### 11.4 数学复核（2026-09-24）

逐条重推课文的推导、用 Python 复算课文与图中的全部数字（15 个生成脚本重跑后 SVG 逐字节不变，只有 l01_layernorm.py 按下面第 5 条改过）。改动：

1. **“吻合到百万位”不准确。** GPT-3 175B 按 (1.30) 精确为 174,604,259,328，表 D.1 为 174,600M，差约 4M：吻合的是四位有效数字，不是百万位。13B（$d=5120$）的简式 $12Ld^2+(V+T_{max})d$ 为 12,850.7M（四舍五入 12,851M），含偏置与 LN 为 12,853.4M（§10.2 第 12 条），与 12,850M 分别差 0.006%、0.03%；$d=5140$ 差 0.78%。课文与练习 11 改为给出相对差。§6 第 4 题答案里“精确到百万位吻合”同样应读作“四位有效数字吻合”。
2. **GPT-3 注意力占比。** $T/(12d+T)=2048/149{,}504=1.37\%$；1.36% 是用 $2N=349.2\times10^9$（表 D.1 的 $N$）作权重项的结果。课文 §8 两个数都给出并说明口径。
3. **softmax 雅可比的 Frobenius 范数**：见 §11.2 表，$s=8$ 由“约 0.24”改为“约 0.25”（2,000 次）。
4. **补上的假设**：温度 $\tau\to0$ 趋于独热要求最大值唯一；LayerNorm 输出落在半径 $\sqrt d$ 的球面上要求忽略 $\epsilon$ 且 $\sigma>0$（练习 10 同）；方差推导里“乘积相互独立”由全部 $2d_k$ 个分量相互独立推出；梯度高速公路的乘积公式在“只看同一位置”时是近似，对整段序列的 $Td$ 维雅可比严格成立，恒等项保证的是一条路径而非总和的大小。
5. **图 1.13 (c) 的 Post-LN 量**：引理 2 的 $\tfrac32d$ 是 $x^{post,5}_{l}$（送进第二个 LN 之前的和，$l>0$），LN 之后的范数平方恒为 $d$。图注与图中标签改为“LN 前的和”，Post-LN 折线从 $l=1$ 画起。
6. **记号**：(1.17) 的求和下标 $s$ 与分数 $s_{t,j}$ 撞车，改为 $k$；掩码推导里的行向量 $\mathbf m$ 与 FFN 输出 $\mathbf m_t$ 撞车，改写为 $S_{tk}+M_{tk}$；点积夹角 $\theta$ 与模型参数 $\theta$ 撞车，改为 $\varphi$；FFN 组件卡与要点里的 $b_m$ 统一为 $b_{1,m}$ 并补上 $+\mathbf b_2$；(1.19) 注明本课的 $M$（掩码）与全局符号表中第 6 课的 $M$（每 token 非嵌入 FLOPs）无关。
7. **措辞**：§3 “上三角的一半被掩掉”改为严格上三角 $1024\times1023/2=523{,}776$ 格（49.95%）；§2 “嵌入梯度稀疏”限定为输入端（权重共享后输出端梯度稠密，§7）；图 1.14 图注 “FFN 恰好是注意力的两倍、两者是块的全部”改为不计偏置时恰为 $8d^2:4d^2$、另有每块 3,072 个 LN 参数。

### 11.5 事实与引用复核（2026-09-24）

逐条列出课文里的数字、规格、引文和“论文说”，对照本文件 §1–§10 与 §11；本文件没有覆盖的三处，用 `arxiv.py grep` 回到原文核对（下面给出原文位置）。全部 `[@key]` 逐句检查是否支持它所附的那句话。

**新核实的事实（一手来源）**

1. **Press & Wolf 的归属说明**（1608.05859 v3，§2 “Related Work” 段）：weight tying “was previously used in the log-bilinear model of [Mnih and Hinton 2009]”，但 “the decision to use it was not explained, and its effect on the model’s performance was not tested”；Inan et al. 2016 “Independently and concurrently” 给出另一种解释。与 C30 一致；课文 §7 据此把“这个做法的出处是 Press 与 Wolf”改为“原论文引用的是 Press 与 Wolf”，并补上更早用法与同期工作。
2. **Xiong 等对其他层的分析**（2002.04745 v2，§3.3 末 “Extended theory to other layers/parameters”）：Post-LN 的梯度范数 “large for the parameters near the output and will be likely to decay as the layer index l decreases”，Pre-LN “likely to stay the same for any layer”，结果是 “preliminary”，放在补充材料；§3.4 “On the extended theory”：6-6 模型各层 FFN 的实测期望梯度，Post-LN “grows along with the layer index”，Pre-LN “almost keeps the same”。课文练习 3 的答案原写“（24 个 LN 雅可比的乘积）是 Xiong 等人分析的出发点之一”，原文没有这样说，改为引用上述两句（初步理论 + 实测）。同段 §1 引言另把 Pre-LN 归到 Baevski & Auli 2018、Child et al. 2019、Wang et al. 2019（与 C29 的列表互补）。
3. **Chinchilla 附录 F 的口径**（2203.15556 v1，附录 F）：逐项表是“Total forward pass FLOPs”，并写 “As in Kaplan et al. (2020) we assume that the backward pass has twice the FLOPs of the forward pass”；“Key @ Query logits”与“Softmax @ query reductions”各为 $2\times\text{seq\_len}^2\times(\text{key\_size}\times\text{num\_heads})$，按整段序列计。hoffmann2022 卡片的 finding 由“逐项计数训练 FLOPs”改为“逐项计数一次前向的 FLOPs（反向按前向的两倍计）”。

**据本文件已有条目改正的课文（引用或措辞与笔记不符）**

4. §6 “OpenAI 公开的 GPT-2 代码……里面所有权重都用 0.02”：与 C20 矛盾（`wpe` 为 0.01），改为“每个线性层的权重按 0.02、只有 `wpe` 是 0.01”。
5. §1 “GPT-1……并让输出层直接乘以输入嵌入矩阵的转置”被写成 GPT-1 相对 Transformer 的改动；但 Vaswani §3.4（C7）已让两个嵌入层与 pre-softmax 线性层共享。改为“输出层沿用原 Transformer 的权重共享”，引 radford2018 + vaswani2017。图 1.2 同步：(a) 的 “Linear” 标为 “Linear（共享嵌入）”，(b) 的改动清单由“输出乘 $W_e^\top$”改为“输出层仍共享嵌入”（`l01_lineage.py`）。
6. 最终 LN 的位置：§1、§6 两处“在最后一块 / 整个堆栈之后加一个 LayerNorm”只引了 radford2019，而报告字面是 “after the final self-attention block”（E7），位置依据是代码；两处都补引 gpt2code。
7. §1 形状检查例的 token 编号 464/24479/11241/1634 原引 radford2019 + hfgpt2；报告里没有这些编号，它们来自官方 `encoder.py` 的移植加 HF 词表文件（§4.7），改引 gpt2code + hfgpt2。§2 “空格字节 0x20 映射成 Ġ” 补引 gpt2code（`bytes_to_unicode`）。
8. 参考文献卡片：hfgpt2 的标题与 finding 补上 `openai-gpt` 配置、`modeling_openai.py`（GPT-1 移植为 Post-LN，C15）与 `activations.py`（`gelu_new`），因为图 1.2 图注与 §5 用它支撑这两件事；gpt2code 的 finding 把“模型改称 124M/…”落到 DEVELOPERS.md（C18）；press2017 的 matters 由“权重共享的出处”改为“最常被引用的出处”。
9. 图 1.13 (b) 的标题“Pre-LN（GPT-2 起）”容易读成 GPT-2 首创，与 E4 冲突，改为“Pre-LN（GPT-2 采用）”（`l01_layernorm.py`）。

**核对后保留不改**：GPT-2 small 形状与 124,439,808、四个规模与嵌入占比、Vaswani 表 3 (A)(C)(E) 与 §3.2.1/§3.2.2/§3.5 的引文、√d_k 脚注的版本（v2、PDF 脚注 4）、GELU 数值与近似误差、Geva 的 16 层 / 65,536 键 / “two-thirds”、Xiong 定理 1 与引理 2/3 的假设和 8.45 vs 约 34、Kaplan 式 (2.1)(2.2) 与 $d_{model}>n_{ctx}/12$、GPT-3 175B 的 174,604,259,328 vs 表 D.1 174,600M、13B 行 5140/5120、KV cache 36 KiB / 4.5 MiB、全部跨课链接（`lesson-00/04/05/07/09/11/14/16` 为 writer-brief 中的真实 slug；M01 §1–§6、M04 §2–§4、M06 §3/§5/§7 的页题与引用处内容一致）。


### 11.6 数学复核（第二轮，2026-09-24）

重推全部 `::: deriv` 与行间公式（逐项核对形状、下标、转置与归一化常数），用 Python 复算正文、表格、练习答案与 15 个生成脚本里的数字；15 个脚本重跑后 SVG 逐字节不变，没有改动脚本。改动只在课文：

1. **softmax 雅可比的 Frobenius 范数（§4“三处需要诚实”）。** §11.4 第 3 条把 $s=8$ 由“约 0.24”改成“约 0.25”，依据是单个种子的 2,000 次抽样（0.2473）。2,000 次的标准误约 0.0035，单次估计会在 0.235–0.25 之间摆动。改用 20,000 次、4 个种子：$s=8$ 为 0.2393 / 0.2420 / 0.2438 / 0.2431（每个标准误约 0.0011），$s=1$ 为 0.0713。课文改回“约 0.24 对 0.071（20,000 组）”。公式 $\norm{\diag(p)-pp^\top}_F^2=\sum p_i^2-2\sum p_i^3+(\sum p_i^2)^2$。
2. **梯度高速公路的乘积不是近似（§6）。** §11.4 第 4 条说“只看同一位置”时乘积公式是近似；其实对 $\partial\mathbf h^{(L)}_t/\partial\mathbf h^{(l)}_t$（$J_l$ 取只让 $\mathbf h^{(l-1)}_t$ 变动时的雅可比）它是精确的：因果掩码下整段序列的逐层雅可比是分块下三角的，乘积的对角块等于各对角块之积；$\mathbf h^{(l)}_t$ 的扰动只传到位置 $\ge t$，而位置 $t$ 只读位置 $\le t$。损失对 $\mathbf h^{(l)}_t$ 的总梯度另含经过 $s>t$ 的项。课文括注据此改写。
3. **“除以 $\sqrt{d_k}$ 与缩小 $W^Q$ 初始值只差比例是否固定”（§4）说得过满。** 两者只在初始化时的前向等价；固定因子同时把 $\partial\mathcal L/\partial W^Q=\mathbf g\bar{\mathbf h}^\top/\sqrt{d_k}$ 缩小 $\sqrt{d_k}$ 倍，同样学习率下每步对分数的改动不同（SGD 下相差 $d_k$ 倍）。课文补上这一点。
4. **GELU tanh 近似的误差（§5 要点）。** “误差不超过 $4.7\times10^{-4}$”与实际最大值 $4.7324\times10^{-4}$ 矛盾，改为“最大误差约 $4.73\times10^{-4}$”。在 $[-30,30]$ 上复算：tanh 近似的最大误差在 $x\approx-2.699$，sigmoid 近似 $x\sigma(1.702x)$ 的最大误差 $2.0335\times10^{-2}$ 在 $x\approx2.270$；$\abs x\to\infty$ 时两者误差都趋于 0，所以 $[-6,6]$ 上的最大值就是全局最大值，§5 正文补上位置与这句话。
5. **输出层玩具例（§7）。** 五个三位小数概率相加是 0.999，课文原写“加起来等于 1”，改为“精确值加起来等于 1（四舍五入后相加得 0.999）”。
6. **记号。** Xiong 定理 1 的 $\epsilon$ 与 LayerNorm 的 $\epsilon=10^{-5}$ 同形，定理框里注明两者无关；§7 引 GPT-1 输出公式时 $W_e^T$ 统一为 $W_e^\top$，并注明是行写法。

**复算后确认无误、保持不改**：方程组 (1.1)–(1.13) 的形状与下标；独热乘矩阵、正弦位移旋转、点积方差、softmax 导数、掩码归一化、因果归纳、分块合头、FFN 键值展开、残差望远镜求和、log-softmax 各推导；§3 手算五步与两处对照、练习 2/5 的 softmax、两头例（$\mathbf u_3=(0.4220,0.4445)$）、FFN 玩具例、LayerNorm 例与练习 10；温度表五行；$\omega_1=0.96466$、$(\sin,\cos)=(0.8219,0.5697)$、最长波长 60,611、图 1.5 左图 $m=0$ 振荡 20.4 次与 $m=128$ 走 0.204 个周期；四个 GPT-2 规模、GPT-1 与“40,478 词表”四个数、GPT-3 175B 与 13B 两种 $d$ 的参数量；每 token FLOPs（$T=1024$ 与 $T=512$）、29%、10.0%、5.26%、1.37%/1.36%、0.1% 的巧合；KV cache 36 KiB / 36 MiB / 288 MiB / 4.5 MiB / 9 GiB；logits 张量 205.9 MB；图 1.8 的 63.77 / 0.996 / 0.027 / 0.726，图 1.9 的 0.782 与 0.132。

### 11.7 事实与引用复核（第二轮，2026-09-24）

重新逐条列出课文、图（15 个 SVG 的文字全部抽出比对）与参考文献卡片里的数字、规格、引文、归属和“论文说”，对照本文件 §1–§11；
没有覆盖或说法不够准的，回到一手来源核对。检查每个 `[@key]` 是否支持它所附的那句话，每个跨课链接是否指向真实 slug 与合理的 `#pN`。

**新核实的事实（一手来源）**

1. **GPT-1 报告图 1（左）画的是 Post-LN。** 从 OpenAI PDF 第 4 页抽出嵌入的图片（`pypdf` 的 `page.images`，Im1）逐框读：
   自下而上为 “Text & Position Embed” → [“Masked Multi Self Attention” → ⊕（残差）→ “Layer Norm” → “Feed Forward” → ⊕ → “Layer Norm”] × “12x”
   → “Text Prediction” / “Task Classifier”。也就是说，报告**正文**没写 LN 位置（C15 仍对），但**图 1 画出了** LN 在每个残差相加之后，
   且块外没有最终 LN。这部分回答了 §8 第 4 条（报告与 HF 移植一致；OpenAI 原始 `finetune-transformer-lm` 代码仍未比对）。
   课文图 1.2 图注原写“报告没有写 LayerNorm 放在哪里，图中的 Post-LN 依据的是 Hugging Face 的移植代码”，改为“报告正文没写，但图 1 画在残差相加之后，
   HF 移植也如此”；radford2018 卡片 finding 补上这一点；hfgpt2 卡片 matters 删去“GPT-1 的 LN 位置只能从这里读到”；`l01_lineage.py` 的文档字符串同步
   （SVG 重生成后逐字节不变）。
2. **tensor2tensor 何时改成 Pre-LN。** GitHub API 列出标签并按标签读取 raw 文件：`v1.0.3`（标签提交 2017-06-20）的 `transformer.py` 里
   `residual_fn(x, y)` 返回 `layer_norm(x + tf.nn.dropout(y, …))`，即 Post-LN；`v1.2.0`–`v1.2.4`（2017-08-26 至 09-30）的 `basic_params1` 为
   `"none"`/`"dan"`，还没有 `transformer_base_v2`；`v1.2.5`（标签提交 2017-10-16）首次出现 `transformer_base_v2`（`"n"`/`"da"`，warmup 8000、lr 0.2），
   `transformer_base()` 直接 `return transformer_base_v2()`。同文件的 `transformer_n_da` 文档字符串写着 “seems to cure failure-to-learn bugs”
   与 “Probably this should become the default”。这回答了 §8 第 1 条里“`transformer_base_v2` 引入日期”一半（Pre-LN 的最早出处仍未定）。
   课文 §6 的“后来把默认配置改成了 Pre-LN”由此有了日期，t2t 卡片 finding 补上 v1.0.3 / v1.2.5。
3. **Vaswani 表 3 (A) 与 §6.2 措辞**（1706.03762 v7）：“keeping the amount of computation constant”；“single-head attention is 0.9 BLEU worse than the best setting”；
   (A) 行四组为 $h=1$：5.29/24.9，$h=4$：5.00/25.5，$h=16$：4.91/25.8，$h=32$：5.01/25.4（基线 $h=8$：4.92/25.8）。§3.2.1 v7 仍是 “We suspect that for large values of $d_k$ …”。
   课文 §4 的“保持总计算量不变”“16 个头与 8 个头持平”“论文自己用的词是 We suspect”均成立。
4. **Geva 等的“浅层模式”具体指什么**（2012.14913 v2 §3.1–3.2）：人工标注把模式分为 “shallow”（“e.g. recurring n-grams”）与 “semantic”（“recurring topic”）；
   层 1–9 “dominated by shallow patterns, often with prefixes that share the last word”，层 10–16 更多语义模式。课文 §5 原写“（如特定的词或 n-gram）”，
   改为“（如重复出现的 n-gram、以同一个词结尾的前缀）”，并补上层号与“（如同一个话题）”。
5. **He et al. 的“不增加计算”**（1512.03385 v1 §3.2）：“neither extra parameter nor computation complexity”，括注 “except for the negligible element-wise addition”。
   课文“也几乎不增加计算”与之一致。
6. **Ba et al. 与 batch normalization 的区别**（1607.06450 v1 摘要）：“Unlike batch normalization, layer normalization performs exactly the same computation at training and test times.”
   课文 §6“这是它与 batch normalization 的区别”成立。
7. **PaLM 的 MFU 在哪里定义**（2204.02311 v5）：MFU 在 §4.1 提出（“We propose a new metric … called model FLOPs utilization (MFU)”），附录 B 开头
   “We elaborate the calculation of model FLOPs utilization (MFU) as discussed in Section 4.1”，给出 $6N$、“$6LH(2QT)$” 与 $R=P/(6N+12LHQT)$，并说反向含前向两倍的矩阵乘（引 Kaplan）。
   palm2022 卡片原写“据此定义模型 FLOPs 利用率”，改为“用它计算正文 §4.1 提出的 MFU”，`a:` 字段改为“读 v5 的附录 B 与 §4.1”。

**据其他课的研究笔记改正**

8. §8 “解码受带宽限制”原写“DeepSeek 反复压缩 KV cache（第 9、14、18、19 课）”。`lesson-14-doc.md` 第 512 行与勘误 11 明确：DSA **不减少** KV cache 的存储
   （MLA 条目全部保留，还多存 indexer 键），减少的是每步读取与计算。改为“一再想办法少存、少读 KV cache：第 9、18、19 课压缩缓存本身，第 14 课的稀疏注意力
   不减少缓存、只让每一步少读一些”。（第 18、19 课的压缩依据 `course-map.md` 第 18、19 课条目：CSA/HCA 压缩、跨层复用与 FP4 KV。）

**据本文件已有条目补全引用**

9. §2“可学习的位置表（GPT-1、GPT-2）”只引了 radford2018，GPT-2 那一半（`wpe` 1024 行）来自官方代码（C20），补引 gpt2code。

**核对后保留不改**：跨课链接全部是 writer-brief 中的真实 slug（`lesson-00/04/05/07/09/11/14/16`），数学链接 M01 §1–§6（点积 / 三种读法 / 形状思维 / 分块 / 数 FLOPs / 投影）、
M04 §2–§4（期望与方差 / 独立和的方差 / softmax 与温度）、M06 §3/§5/§7（链式法则与雅可比 / softmax 雅可比 / 残差梯度高速公路）与引用处内容一致；
第 16 课“Post-LN 与 Pre-LN 的两难是 mHC 的出发点之一”有 `lesson-16-doc.md` C6、C7（Liu 2020 的“稳但弱 / 强但易发散”、HC 摘要的 “seesaw effect”）支持；
15 张图中的文字标签（形状、参数、波长、合并轨迹、掩码权重、GELU 最值、FLOPs 分解）与本文件及正文逐一一致；其余条目同 §11.5 的“保留不改”清单。

---

## 补充核实（黑板课改写）

2026-09-26，第 1 课按 `style-blackboard.md` 改写为“黑板课”（改写前的版本在 git 提交 9a24abd）。改写**没有引入新的外部事实**：正文里的规格、引文与归属全部沿用 §1–§11；
新增的只有三类东西：黑板上的小例子（随手取的数，已在课文“本课的记号”里声明）、用本文件已有来源复现的分词结果、以及由已有数字推出的算式。
下表逐项列出改写时新写进课文的数字，全部在 Python 里重算过（一次性核算脚本未入库；四张新图的生成脚本 `deepseek/figures/gen/l01b_*.py` 带 assert，重跑可复现）。

### B1 分词：用 GPT-2 的分词逻辑跑在它自己的词表上

方法与 §4.7 相同：按官方 `encoder.py` 移植 `bytes_to_unicode` 与 `bpe()`（C20、§10.2 第 8 条），读 Hugging Face 的 `vocab.json`（50,257 项）与 `merges.txt`（50,000 条）（§10.2 第 9 条）。
预切分正则里的 `\p{L}` 用 Python `re` 的 `[^\W\d_]` 代替；下面三段文本都不含标点与数字，两种写法切出的块相同。UTF-8 字节数用 `str.encode('utf-8')` 得到。

| 文本 | UTF-8 字节 | 切出的 token（字节分组） | 编号 | 用在 |
|---|---|---|---|---|
| `The cat sat`（11 个字符） | — | `The`、`Ġcat`、`Ġsat` | 464、3797、3332 | §2 黑板、§2 表 |
| 大模型 | E5 A4 A7 · E6 A8 A1 · E5 9E 8B | [E5 A4 A7]、[E6]、[A8]、[A1]、[E5]、[9E]、[8B]，共 7 个 | 32014、162、101、94、161、252、233 | §2 黑板、§2 表、封面 |
| 你好 | E4 BD A0 · E5 A5 BD | [E4 BD]、[A0]、[E5 A5]、[BD]，共 4 个 | 19526、254、25001、121 | §2 错误框 |

“大”的两次合并：`('å','¤')`（E5+A4）名次 13,527，`('å¤','§')`（+A7）名次 31,758；“模”“型”各自的相邻字节对在合并表里都没有条目。
课文的说法“哪些字节对能被粘起来，取决于它们在构造合并表的语料里出现得多不多”是 BPE 算法本身的定义（C25），**没有**对 GPT-2 训练语料的语种构成作任何断言。
“一个常用汉字在 UTF-8 里占 3 个字节”：U+4E00–U+9FFF 的汉字编码为 3 字节（`'大'.encode('utf-8')` 等逐字验证）。

### B2 其余新数字（表达式 → 值）

| 页 | 用途 | 表达式 | 值 |
|---|---|---|---|
| §1 | 5 词世界的概率（取自 §7 玩具输出层，§11.2） | $\softmax(E\tilde{\mathbf h})$，$\tilde{\mathbf h}=(-0.4,2,0.6)$ | $(0.090,0.106,0.557,0.129,0.117)$，三位小数相加 0.999 |
| §2 | 独热乘 $E$（图 1.4 (a) 的表） | $V\cdot d$ 次乘法，其中 $(V-1)d$ 次乘 0 | 15 次，12 次乘 0；$E^\top\mathbf e_{\text{cat}}=(0.9,0.3,-0.2)$ |
| §2 | 4 维正弦编码 | $\omega_0=1$，$\omega_1=10000^{-2/4}=0.01$；波长 $2\pi$、$2\pi/0.01$ | 6.283、628.3 |
| §2 | 同上，读数 | $(\sin t,\cos t,\sin0.01t,\cos0.01t)$，$t=1,2,3,9$ | $(0.8415,0.5403,0.0100,1.0000)$；$(0.9093,-0.4161,0.0200,0.9998)$；$(0.1411,-0.9900,0.0300,0.9996)$；$(0.4121,-0.9111,0.0899,0.9960)$ |
| §2 | 旋转代回 | $2\sin1\cos1$；$\cos^21-\sin^21$ | 0.9093；$0.2919-0.7081=-0.4161$ |
| §3 | 均匀平均 | $1/7$ | 0.1429 |
| §3 | 打分不对称 | $\mathbf q_2\cdot\mathbf k_3=0\cdot1+2\cdot1$ | 2（对照 $\mathbf q_3\cdot\mathbf k_2=4$） |
| §3 | 软检索 | $0.0743+0.3057$ | 0.38 |
| §4 | 开场抛硬币 | 64 个独立 ±1 之和 $S$：$\sqrt{64}$；$P(\abs S\le8)$；$P(\abs S\ge32)$（二项分布精确求和） | 8；0.7396；$7.73\times10^{-5}$ |
| §4 | $d_k=4$ 的点积分布 | $2^4=16$ 种符号组合按和分组 | 值 −4/−2/0/2/4 各 1/4/6/4/1 种；$\Var=64/16=4$；除以 2 后 $\Var=1$ |
| §4 | softmax 雅可比对角线 $p_i(1-p_i)$ | $\mathbf z=(2,1,0.1)$，$\tau=1$ 与 $\tau=1/8$ | $(0.2247,0.1837,0.0889)$；$p=(0.99966,0.00034,2.5\times10^{-7})$，对角线 $(0.00034,0.00034,2.5\times10^{-7})$ |
| §4 | 掩码后 cat 行 | $e^{1.4142}+e^0$ | 5.1133 → $(0.8044,0.1956,0)$（与 §11.2 一致） |
| §5 | 线性叠加 | $3\cdot(2x)=6x$；$3\cdot\mathrm{ReLU}(2x)$ 在 $x=\pm1$ | 6 与 0 |
| §5 | FFN 系数之和 | $1.39979+0.34573-0.04550$ | 1.70002 |
| §6 | 3 层残差的 8 条路 | $0.1^3$；$1.1^3=\sum_k\binom3k0.1^k$ | 0.001；$1+0.3+0.03+0.001=1.331$ |
| §6 | $2L$ 份增量之和的标准差 | $\sqrt{24}$ | 4.899 |
| §6 | 三份“答卷” | $(20,40,60,80)$：$\mu,\sigma^2,\sigma$；$(102,\dots,108)$：$\mu,\sigma^2$ | 50、500、22.361；105、5；三者输出都是 $(-1.3416,-0.4472,0.4472,1.3416)$，加 $\epsilon=10^{-5}$ 后前四位不变 |
| §8 | 玩具 GPT（$V=10,T_{max}=4,d=4,L=1$） | wte $10\cdot4$，wpe $4\cdot4$，c_attn $4\cdot12+12$，$W^O$ $4\cdot4+4$，$W_1$ $4\cdot16+16$，$W_2$ $16\cdot4+4$，3 个 LN 各 $2\cdot4$ | 40、16、60、20、80、68、8；块 244；合计 308 = (1.30) |
| §8 | 同上，近似与占比 | 非嵌入 $244+8$ 对 $12Ld^2$；$13d/12d^2$；嵌入 $56/308$；注意力 $60+20$，FFN $80+68$，LN $3\times8$ | 252 对 192（小 23.8%）；0.271；18.2%；80、148、24 |
| §8 | 每个权重两次运算 | $2\times3$ 矩阵带偏置乘向量 | 6 次乘法 + 6 次加法 = 12 |
| §8 | 因果平均的注意力（$T=4$，§11.3 第 1 条的解释） | $4d(1+2+3+4)/4$；$2(T+1)d$；$4Td$ | $10d$；$10d$；$16d$ |
| §8 | GPT-2 small 的嵌入占比 | $38{,}597{,}376/124{,}439{,}808$；加上 wpe | 31.0%；31.6%（与 §4.1 一致） |

### B3 改写中重新组织、但内容未变的地方

1. 图号：新增图 1.7（`l01b-board-attention`）、1.14（`l01b-residual-paths`）、1.15（`l01b-layernorm-board`）、1.17（`l01b-tiny-params`）；原图 1.1–1.6 编号不变（第 5 课引用的“第 1 课图 1.5”仍指正弦位置编码），
   原 1.7→1.8、1.8→1.9、1.9→1.10、1.10→1.11、1.11→1.12、1.12→1.13、1.13→1.16、1.14→1.18、1.15→1.19；对应生成脚本的文档字符串同步改号（SVG 重生成后逐字节不变）。
2. 公式编号 (1.1)–(1.32) 与页序 §1–§9 不变（其他课链接的 `#p2`–`#p8` 仍指向同一主题）。
3. 原来夹在正文或警告框里的四段史实改放进“小故事”框，文字与引用不变：BPE 原为压缩算法（sennrich2016 卡片）、方差脚注 v2 才加入（C2）、残差连接出自图像识别论文（C28，论文标题）、权重共享的更早用法（C30）；另把“117M 与 124M”（C18）放进 §8 的小故事框。
4. 生活类比只作开场，不构成事实断言：输入法候选栏、接半句话、读小说时往回翻、抛硬币、背单词时盖住答案的纸、作文比赛的评委、乘法口诀、传话游戏、你比划我猜、模型名字里的“7B”。
