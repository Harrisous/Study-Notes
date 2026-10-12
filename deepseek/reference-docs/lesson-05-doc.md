# 第 05 课研究笔记：现代 Transformer 积木——RoPE、RMSNorm、SwiGLU 与上下文扩展（PI / NTK-aware / YaRN / DeepSeek 的用法）

> 对应课文：`deepseek/src/lesson-05-modern-block.md`（尚未撰写）。本文件只存放**核对过的**引文、公式、规格、复算数字、
> 勘误与未能核实清单，供写课文时逐条引用。整理日期 2026-09-24。
>
> 核对方式：arXiv 论文用 `arxiv.py text/grep` 拉 LaTeXML 全文后逐句比对；超参数同时读论文表格与 Hugging Face
> `config.json`、官方推理代码；所有派生数字都在 Python 里重算（表达式写在 §4）。
>
> **引文长度约定**：受版权约束，本文件中的英文原文引文一律只摘关键短语（每处不超过约 15 个英文词），其余用中文转述；
> 公式、数字、超参数属于事实，完整转录。需要更长上下文时回到原文对应位置（位置都已标明）。

---

## 0. 覆盖范围与所读版本

### 0.1 论文

| key | 标题 | 作者 | arXiv / 版本 | 发表状态 | 日期 |
|---|---|---|---|---|---|
| `su2021roformer` | RoFormer: Enhanced Transformer with Rotary Position Embedding | Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu（arXiv 顺序；期刊版顺序为 Su, Ahmed, Lu, Pan, Bo, Liu） | 2104.09864，**读 v5**（2023-11-08，comment “fixed some typos”） | Neurocomputing 568 (2024) 127063（ScienceDirect/ACM DL 记录，DOI 10.1016/j.neucom.2023.127063） | v1 2021-04-20 |
| `zhang2019rmsnorm` | Root Mean Square Layer Normalization | Biao Zhang, Rico Sennrich | 1910.07467，**读 v1**（唯一版本） | NeurIPS 2019（arXiv comment） | 2019-10-16 |
| `shazeer2020glu` | GLU Variants Improve Transformer | Noam Shazeer | 2002.05202，**读 v1**（唯一版本） | arXiv 预印本（未见正式发表） | 2020-02-12 |
| `touvron2023llama` | LLaMA: Open and Efficient Foundation Language Models | Touvron, Lavril, Izacard, … Lample（14 人） | 2302.13971，**读 v1**（唯一版本） | arXiv 预印本 | 2023-02-27 |
| `chen2023pi` | Extending Context Window of Large Language Models via Positional Interpolation（HTML 正文标题写作 “Position Interpolation”） | Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian | 2306.15595，**读 v2**（comment “Fix template issues”，HTML 2023-06-28） | arXiv 预印本（未找到同行评审版本） | v1 2023-06-27 |
| `peng2023yarn` | YaRN: Efficient Context Window Extension of Large Language Models | Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole | 2309.00071，**主读 v3**（2026-02-06），**对照 v1**（2023-08-31）与 **v2**（2023-11-01） | ICLR 2024（proceedings.iclr.cc、OpenReview id wHBfxhZu1u） | v1 2023-08-31 |
| `deepseek2024v2` | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek-AI | 2405.04434，**主读 v5**；§3.1.4 在 v1–v5 **逐字相同**（已逐版 grep） | arXiv 技术报告 | v1 2024-05-07 |
| `deepseek2024v3` | DeepSeek-V3 Technical Report | DeepSeek-AI | 2412.19437，**主读 v2**（2025-02-18）；§4.3 与 v1 逐字相同 | arXiv 技术报告 | v1 2024-12-27 |
| `deepseek2024llm` | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism | DeepSeek-AI | 2401.02954 v1（只核对了 §2.2 “micro design” 与 Table 2） | arXiv 技术报告 | 2024-01-05 |
| `chowdhery2022palm` | PaLM: Scaling Language Modeling with Pathways | Chowdhery et al. | 2204.02311 v5（只核对 §2 的 SwiGLU / No Biases / RoPE 三条与 Table 1 注） | JMLR 2023（未在本次核对） | 2022-04-05 |
| `radford2019gpt2` | Language Models are Unsupervised Multitask Learners | Radford et al. | 无 arXiv；读 OpenAI 官方 PDF §2.3 | 技术报告 | 2019 |

### 0.2 制品（artefact，无论文或以代码/配置为准）

| key | 内容 | 读取方式 |
|---|---|---|
| `bloc97-2023ntk` | Reddit r/LocalLLaMA 帖子 “NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation.”，作者 u/bloc97 | reddit.com 直连被 403；读 **Wayback Machine 2023-06-29 08:58 UTC 快照**（`web.archive.org/web/20230629085816id_/…/14lz7j5/comment/jpyyras/`），快照页含帖子正文；核查时另读帖子 URL 本身的 Wayback 快照，`<shreddit-post>` 的 `created-timestamp` 为 2023-06-29T08:21:29 UTC |
| `su2021blog` | 苏剑林，《Transformer升级之路：2、博采众长的旋转式位置编码》，科学空间 kexue.fm/archives/8265，页面日期 2021-03-23 | 直连 403；读 Wayback 2021 快照 |
| `yarn-code` | github.com/jquesnelle/yarn，`scaled_rope/LlamaYaRNScaledRotaryEmbedding.py`、`LlamaNTKScaledRotaryEmbedding.py`（master 分支，2026-09-24 拉取） | raw.githubusercontent |
| `ds-v2-config` / `ds-v2lite-config` / `ds-v3-config` | HF `deepseek-ai/DeepSeek-V2`、`DeepSeek-V2-Lite`、`DeepSeek-V3` 的 `config.json` 与 `modeling_deepseek.py` | huggingface.co/…/raw/main |
| `ds-v3-code` | github.com/deepseek-ai/DeepSeek-V3 `inference/model.py`、`inference/configs/config_671B.json` | raw.githubusercontent |
| `ds-later-configs` | HF `DeepSeek-V3.1`、`V3.2-Exp`、`V3.2`、`V4-Flash`、`V4-Pro`、`V4.1-Flash` 的 `config.json`（仅作前瞻指针） | huggingface.co |
| `ds-v31-card` | HF `DeepSeek-V3.1-Base` README（两阶段长上下文扩展的 token 数） | huggingface.co |
| `llama-v1-code` | github.com/meta-llama/llama 分支 `llama_v1`，`llama/model.py` | raw.githubusercontent |
| `hf-llama-convert` | huggingface/transformers `convert_llama_weights_to_hf.py`（`permute` 函数） | raw.githubusercontent |
| `huggyllama-configs` | HF `huggyllama/llama-{7b,13b,30b,65b}` 的 `config.json`（社区转存，非 Meta 官方，只作交叉核对） | huggingface.co |

### 0.3 版本漂移一览（写课文前必读）

- **YaRN 三个版本差别很大**：v1 把温度写成 $\sqrt t\approx0.1\ln s+1$（公式 27，$t$ 是**乘**在 logits 上的因子）；v2/v3 改写成
  $\sqrt{1/t}=0.1\ln s+1$（$t$ 是**除**在 logits 上的温度）。两者数值上是同一个因子。公式编号：v2 的 (18)(20)(21)(22) 对应 v3 的 (11)(13)(14)(15)；
  v3 新增了 Figure 1（方法关系图），图号整体后移（v2 的 Figure 2 = v3 的 Figure 4）。v3 的式 (9) 把 PI 写成 $g(m)=s\cdot m$，是笔误（v2 写 $m/s$，v3 附录 A.1 式 (16) 写 $mL/L'$）。详见 §7。
- **DeepSeek-V2 §3.1.4** 在 v1–v5 五个版本中完全相同（YaRN 段落逐版 grep 一致）。
- **DeepSeek-V3 §4.3** 在 v1、v2 中相同。
- **RoFormer** v5 仅“fixed some typos”，本文件的引文与公式号均取 v5。

---

## 1. 核心论断与原文引文

格式：编号. 论断（中文）｜原文短引｜位置｜状态。状态：**原文**＝论文原话；**代码**＝以代码/配置为准；**推导**＝本笔记自己算的。

### 1.1 RoPE（RoFormer）

1. **出发点是“内积只依赖相对位置”这一要求**，即寻找 $f_q,f_k$ 使 $\langle f_q(\mathbf x_m,m),f_k(\mathbf x_n,n)\rangle=g(\mathbf x_m,\mathbf x_n,m-n)$。
   ｜“we hope that the inner product encodes position information only in the relative form”｜§3.1，式 (11)｜原文。
2. **二维情形的解是复数乘以 $e^{im\theta}$**：$f_q(\mathbf x_m,m)=(W_q\mathbf x_m)e^{im\theta}$。论文说“证明了 a solution”，不是“唯一解”。
   ｜“to prove … that a solution to our formulation Equation 11 is”｜§3.2.1，式 (12)｜原文。
3. **推导（§3.4.1）只对 $d=2$ 做**，中间有选择而非推论：取“初始条件”(22)，从 (26a) 取“a straightforward solution” $R_q(\mathbf x_q,m)=R_q(\mathbf x_q,0)$（式 27），
   令 $n=m+1$ 得相位差为常数（式 29），于是 $\phi(m)=m\theta+\gamma$ 是等差数列（式 30），最后**令 $\gamma=0$**（式 33 前一句）。
   ｜“a straightforward solution of R_f could be formed from Equation 26a”｜§3.4.1｜原文。课文中要把“选择”和“推论”分开讲（style-guide §1 第 5 条）。
4. **高维推广靠“内积的线性”把 $d$ 维拆成 $d/2$ 个二维子空间**，每个子空间一个频率 $\theta_i$，构成块对角旋转矩阵 $R^d_{\Theta,m}$。
   ｜“divide the d-dimension space into d/2 sub-spaces”｜§3.2.2，式 (14)(15)｜原文。高维情形没有唯一性论证。
5. **频率取值**：§3.2.2 写 $\Theta=\{\theta_i=10000^{-2(i-1)/d},\,i\in[1,2,\dots,d/2]\}$（1 起下标）；§3.3 与 §3.4.3 写 $\theta_i=10000^{-2i/d}$（配合 0 起下标 $i=0..d/2-1$）。两者是同一组数。
   ｜“Following Vaswani et al. (2017), we set θ_i = 10000^{−2i/d}”｜§3.3｜原文。
6. **旋转矩阵正交，保模长**。｜“is an orthogonal matrix, which ensures stability”｜§3.2.2 式 (16) 之后｜原文。
7. **RoPE 是乘性的，与此前所有加性位置编码不同。**｜“our approach is multiplicative”｜§3.2.2 末段｜原文。
8. **高效实现**：不做稀疏矩阵乘法，而用两次逐元素乘加（式 34）。｜“a more computational efficient realization”｜§3.4.2｜原文。
9. **远程衰减**：论文称“可以证明”该频率设置带来远程衰减；§3.4.3 实际给出的是经 Abel 变换得到的**上界** $\big(\max_i|h_{i+1}-h_i|\big)\sum_i|S_{i+1}|$，
   并**用图 2 数值展示** $\frac{1}{d/2}\sum_{i=1}^{d/2}|S_i|$ 随相对距离下降。
   ｜“One can prove that this setting provides a long-term decay property”｜§3.3｜原文。｜衰减的是上界，不是注意力本身，见 §7 E3。
10. **图 2 的维度是 $d=128$**：论文图中未标 $d$；按 $d=128$、$b=10000$ 重算与图中曲线吻合（$m=5$ 时 20.85、$m=50$ 时 12.63、$m=250$ 时约 6.55，图中 y 轴 20→约 7，x 轴 0–256），
    苏剑林博客同一图的图注写明“RoPE的远程衰减性（d=128）”。｜推导 + `su2021blog`。
11. **RoPE 可用于线性注意力**：把旋转作用在非负特征映射之后，分母保持不变（式 19）。｜“we keep the denominator unchanged to avoid the risk of dividing zero”｜§3.3｜原文。
12. **RoPE 的中文出处早于论文**：博客页面日期 2021-03-23，把 RoPE 的出发点概括为“通过绝对位置编码的方式实现相对位置编码”。｜`su2021blog`｜制品。

### 1.2 RMSNorm

13. **假设：LayerNorm 成功靠的是“重缩放不变性”，不是“重中心化不变性”。**
    ｜“the re-scaling invariance is the reason for success of LayerNorm”｜§4 第 1 段｜原文。
14. **定义**：$\bar a_i=\frac{a_i}{\mathrm{RMS}(\mathbf a)}g_i$，$\mathrm{RMS}(\mathbf a)=\sqrt{\frac1n\sum_i a_i^2}$（式 4）；论文式中**没有 $\epsilon$**，$\epsilon$ 是实现加的。｜§4｜原文 + 代码。
15. **均值为零时 RMSNorm 与 LayerNorm 完全相同。**｜“When the mean of summed inputs is zero, RMSNorm is exactly equal to LayerNorm.”｜§4｜原文。
16. **不变性**：对整个权重矩阵或输入的统一缩放不变（式 6、7），对单个权重向量各自缩放、以及对平移（re-centering）不具不变性（Table 1）。｜§4.1｜原文。
17. **提速 7%–64%**（不同模型与实现，2019 年的框架与硬件）。｜“reduces the running time by 7%∼64%”｜摘要；§7 结论段同数｜原文。
18. **pRMSNorm**：只用前 $p\%$ 元素估计 RMS；理论上更快但作者未观察到一致的实际加速。｜§5、§7｜原文。

### 1.3 SwiGLU（GLU Variants）

19. **SwiGLU 定义**：$\mathrm{SwiGLU}(x,W,V,b,c,\beta)=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$（式 5）；用于 FFN 时去掉偏置、取 $\beta=1$：
    $\mathrm{FFN}_{\mathrm{SwiGLU}}(x,W,V,W_2)=(\mathrm{Swish}_1(xW)\otimes xV)W_2$（式 6）。｜§2｜原文。
20. **三个矩阵换两个，为保持参数量与计算量不变，把 $d_{ff}$ 乘以 2/3**（原文措辞 “reduce … by a factor of 2/3”，意思是**变为** 2/3 倍，不是减少 2/3）。
    ｜“To keep the number of parameters and the amount of computation constant”｜§2 末段｜原文。实验里 $d_{ff}$ 从 3072 变为 2048（§3.1）。
21. **结果**：T5-base 规模 524,288 步后，SwiGLU 的 heldout log-perplexity 1.636，ReLU 基线 1.677，GEGLU 1.633 最佳（Table 1）。｜§3.2 Table 1｜原文。
22. **作者明说没有解释为什么有效。**｜“we attribute their success, as all else, to divine benevolence”｜§4 Conclusions｜原文。

### 1.4 LLaMA 块

23. **Pre-normalization + RMSNorm**：对每个子层的**输入**做归一化，并使用 RMSNorm。｜“we normalize the input of each transformer sub-layer, instead of normalizing the output”｜§2.2｜原文（方括号致谢写的是 [GPT3]）。
24. **SwiGLU，隐藏维 $\tfrac23\cdot4d$。**｜“We use a dimension of 2/3 4d instead of 4d as in PaLM.”｜§2.2｜原文（致谢 [PaLM]）。
25. **RoPE 取代绝对位置嵌入，每层都加。**｜“We remove the absolute positional embeddings”｜§2.2（致谢 [GPTNeo]）｜原文。
26. **LLaMA 论文正文没有写“无偏置”，也没有写 2048 的上下文长度**：`bias=False` 来自 `llama-v1-code`（所有线性层）。2K 上下文的**主要依据**是 PI 图 1 说明（“Consider a Llama model pre-trained with a 2048 context window length”）与 YaRN §4.1 脚注（“LLaMA models have a pretrained context size of 2k tokens”），两者都是第三方论文的陈述；`llama-v1-code` 里的 `max_seq_len: int = 2048` 只是推理用 `ModelArgs` 的默认值（决定 KV 缓存与 `freqs_cis` 的长度，`example.py` 的默认 `max_seq_len` 是 512），只能算旁证，不是训练上下文的声明。｜代码 + 原文。
27. **“无偏置”的明文出处是 PaLM**：“No biases were used in any of the dense kernels or layer norms.”｜PaLM §2｜原文。PaLM 的 $d_{ff}$ 在 SwiGLU 下仍是 $4d$（Table 1 注：“d_ff is always 4 × d_model”）。
28. **Pre-LN 更早见于 GPT-2**：“Layer normalization … was moved to the input of each sub-block”。｜GPT-2 §2.3｜原文。
29. **DeepSeek LLM 沿用 LLaMA 微结构**：Pre-Norm + RMSNorm，SwiGLU，中间维 $\tfrac83 d_{model}$，RoPE。｜“adopting a Pre-Norm structure with RMSNorm”｜2401.02954 §2.2｜原文。

### 1.5 位置插值（PI）

30. **直接外推会炸**：困惑度可到 $>10^3$，与未训练模型相当。｜§2.2｜原文。
31. **PI 的做法**：把位置下标线性压缩回预训练范围，$\mathbf f'(\mathbf x,m)=\mathbf f(\mathbf x,mL/L')$（式 4）。｜“we directly down-scale the position indices”｜§1｜原文。
32. **插值误差界（Theorem 2.1）**：$|a(s)-a_{\text{linear}}(s)|\le d(\max_j|h_j|)\frac{(s-s_1)(s_2-s)}{8\ln c}$；取 $c=10000$ 得 $\le \frac{d}{32\ln c}\max_j|h_j|\approx\frac{d\max_j|h_j|}{294.73}$（式 5、7）。
    与 RoFormer §3.4.3 的外推界比较，“至少小 $2\cdot294.73\sim600\times$”。｜§2.3｜原文。
33. **PI 自己批评了 RoFormer 的远程衰减界**：该上界“may be too loose”，“and thus vacuous”。｜§2.2｜原文。
34. **微调成本**：多数设置 1000 步；7B/13B 学习率 $2\times10^{-5}$，33B/65B 为 $10^{-5}$；AdamW $\beta_1=0.9,\beta_2=0.95$，warmup 20 步，weight decay 0。｜§3.1｜原文。
35. **扩展倍数**：摘要与实验都是 2048 → 32768（16 倍）；§3 开头却写 “up to 32 times of the original size”。｜§3｜原文（内部不一致，见 §7）。

### 1.6 NTK-aware（bloc97，制品）

36. **思路**：改 RoPE 的**底数 base** 而不是缩放位置。｜“changes the base of the RoPE instead of the scale”｜`bloc97-2023ntk` 帖子正文｜制品。
37. **结果主张**：LLaMA 7B 在不微调的情况下即可用到 4096 上下文，困惑度退化很小；用 alpha=8 与 SuperHOT 的 scale=4 比较。｜帖子正文｜制品（帖子中的曲线未能以数据形式获得）。
38. **公式**：帖子正文未写出底数公式（代码在 Colab）。可核对的公式来自 YaRN 附录 A.2 与 `yarn-code`：$b'=b\cdot s^{|D|/(|D|-2)}$；代码 `base = base * alpha ** (dim / (dim-2))`。｜YaRN v3 附录 A.2 式 (19)(22)｜原文 + 代码。
39. **NTK-aware 的缺点（YaRN 的评价）**：部分维度会被轻微外推到“越界”值，因此用它微调不如 PI；最佳底数往往需经验搜索；Code Llama 把底数手动设为 1M。
    ｜“some dimensions are slightly extrapolated to "out-of-bound" values”｜YaRN v3 附录 A.2；§3.1 脚注｜原文。

### 1.7 YaRN

40. **PI 对所有维度一视同仁地拉伸。**｜“Whereas PI stretches all RoPE dimensions equally”｜v3 §3 开头｜原文。
41. **波长 $\lambda_d=2\pi/\theta_d$、比值 $r(d)=L/\lambda_d$**：$r$ 是该维度在预训练上下文 $L$ 内转过的整圈数。｜v3 式 (8)(10)｜原文。
42. **分段原则**：波长远小于 $L$ 的维度不插值；波长 $\ge L$ 的维度只插值不外推；中间维度两者兼有。
    ｜“it is important to not touch the dimensions that only encode relative positional information”｜v3 §3.2｜原文。
43. **LLaMA 家族推荐 $\alpha=1,\beta=32$。**｜“good values for α and β are α=1 and β=32”｜v3 §3.2（v2 同）｜原文。
44. **注意力温度**：softmax 改为 $\mathrm{softmax}\big(\frac{\mathbf q_m^T\mathbf k_n}{t\sqrt{|D|}}\big)$（v3 式 14）；实现上把 $\mathbf q,\mathbf k$ 都乘 $\sqrt{1/t}$（“length scaling” 技巧），零额外开销。
    ｜“scales both q_m and k_n by a constant factor √(1/t)”｜v3 §3.3｜原文。
45. **推荐值 $\sqrt{1/t}=0.1\ln s+1$**，由 LLaMA 7b/13b/33b/65b **不微调**时在不同 $s$ 下的最低困惑度拟合而来，对 Llama 2 也“fairly well”。｜v3 式 (15) 及其后一段｜原文。
46. **YaRN 的定义 = 注意力缩放 + NTK-by-parts**（Definition 2）。它**不改底数**（改底数的是 NTK-aware）。｜v3 §3.3｜原文。
47. **Dynamic Scaling**：每次前向用 $s=\max(1,l'/L)$；与 KV cache 同用时应缓存旋转**之前**的 K。｜“cache the kv-embeddings before applying rotary position embeddings”｜v3 §3.4｜原文。
48. **训练成本**：Llama 2 7B/13B，$s=16$ 训练 400 步（全局 batch 64，PG19 切成 64k 段）；$s=32$ 从 $s=16$ 检查点再训 200 步，仍只用 64k 数据，却能外推到 128k（Table 1）。学习率 $2\times10^{-5}$，无 weight decay，warmup 20 步，AdamW (0.9, 0.95)。｜v3 §4.1｜原文。
49. **摘要主张**：比以往方法少用 10 倍 token、2.5 倍训练步数。｜摘要｜原文。对照对象见 v1 §3.4：10 倍对 Code Llama、2.5 倍对 PI（400 vs 1000 步）；10 倍 token 的具体算法未复核，见 §8 第 6 条。

### 1.8 DeepSeek 的做法

50. **V2 为什么需要解耦 RoPE**：RoPE 夹在 $W^Q$ 与 $W^{UK}$ 之间使吸收失效。｜“RoPE is incompatible with low-rank KV compression.”｜V2 §2.1.3｜原文。详细推导归第 9 课。
51. **V2：YaRN 只作用在承载 RoPE 的解耦键 $\mathbf k_t^R$ 上**（相对于内容部分 $\mathbf k^C$ 而言）。｜“specifically applied to the decoupled shared key”｜V2 §3.1.4｜原文。代码中 $\mathbf q^R$ 与 $\mathbf k^R$ 都用同一套 YaRN 频率（§2.7）。
52. **V2 超参数**：$s=40$，$\alpha=1$，$\beta=32$，目标最大上下文 160K，期望在 128K 表现良好。｜“the target maximum context length to 160K”｜V2 §3.1.4｜原文。
53. **V2 改了长度缩放系数**：$\sqrt t=0.0707\ln s+1$。｜“Slightly diverging from original YaRN, due to our distinct attention mechanism”｜V2 §3.1.4｜原文。
54. **V2 只在 32K 上训了 1000 步，batch 576 条序列**，评测到 128K 仍稳健（图 4 NIAH，图中标题写的是 “DeepSeek-V2 Base 128K”）。｜“train the model for 1000 steps, with a sequence length of 32K”｜V2 §3.1.4｜原文。
55. **V3：两阶段、各 1000 步，4K → 32K → 128K。**｜“two additional training phases, each comprising 1000 steps”｜V3 §4.3｜原文。
56. **V3：两阶段超参数相同**，$s=40,\alpha=1,\beta=32$，$\sqrt t=0.1\ln s+1$；阶段一 32K、batch 1920；阶段二 128K、batch 480；学习率均为 $7.3\times10^{-6}$，与预训练末段学习率一致。
    ｜“matching the final learning rate from the pre-training stage”｜V3 §4.3｜原文。
57. **V3 自称“与 V2 的 YaRN 配置一致”，但温度系数从 0.0707 变为 0.1**；`config.json` 印证：V2 `mscale_all_dim=0.707`，V3 `mscale_all_dim=1.0`（§2.7、§7 E11）。｜“The YaRN configuration is consistent with that used in DeepSeek-V2”｜V3 §4.3｜原文 + 配置。
58. **V3 上下文扩展成本 119K H800 GPU 小时**（总 2,788K 中的 4.3%，按 2 美元/小时为 0.238M 美元）。｜V3 Table 1｜原文。
59. **V3.1 延续同一两阶段方法并加量**：32K 阶段增至 10 倍（630B token），128K 阶段增至 3.3 倍（209B token）。｜“The 32K extension phase has been increased 10-fold to 630B tokens”｜`ds-v31-card`｜制品。与 §4 R19 复算的 V3 两阶段各 62.9B token 相互印证。

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 RoFormer（v5）原式

（原文用行向量/列向量混用、位置下标 $m,n$、维度 $d$、虚数单位与下标都写 $i$；照印刷转录。）

$$\langle f_q(\boldsymbol x_m,m),f_k(\boldsymbol x_n,n)\rangle=g(\boldsymbol x_m,\boldsymbol x_n,m-n).\tag{11}$$

$$f_q(\boldsymbol x_m,m)=(\boldsymbol W_q\boldsymbol x_m)e^{im\theta},\quad f_k(\boldsymbol x_n,n)=(\boldsymbol W_k\boldsymbol x_n)e^{in\theta},\quad g(\boldsymbol x_m,\boldsymbol x_n,m-n)=\operatorname{Re}\big[(\boldsymbol W_q\boldsymbol x_m)(\boldsymbol W_k\boldsymbol x_n)^{*}e^{i(m-n)\theta}\big]\tag{12}$$

$$f_{\{q,k\}}(\boldsymbol x_m,m)=\begin{pmatrix}\cos m\theta&-\sin m\theta\\ \sin m\theta&\cos m\theta\end{pmatrix}\begin{pmatrix}W^{(11)}_{\{q,k\}}&W^{(12)}_{\{q,k\}}\\ W^{(21)}_{\{q,k\}}&W^{(22)}_{\{q,k\}}\end{pmatrix}\begin{pmatrix}x^{(1)}_m\\ x^{(2)}_m\end{pmatrix}\tag{13}$$

$$f_{\{q,k\}}(\boldsymbol x_m,m)=\boldsymbol R^{d}_{\Theta,m}\boldsymbol W_{\{q,k\}}\boldsymbol x_m\tag{14}$$

$$\boldsymbol R^{d}_{\Theta,m}=\begin{pmatrix}\cos m\theta_1&-\sin m\theta_1&0&0&\cdots&0&0\\ \sin m\theta_1&\cos m\theta_1&0&0&\cdots&0&0\\ 0&0&\cos m\theta_2&-\sin m\theta_2&\cdots&0&0\\ 0&0&\sin m\theta_2&\cos m\theta_2&\cdots&0&0\\ \vdots&\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\ 0&0&0&0&\cdots&\cos m\theta_{d/2}&-\sin m\theta_{d/2}\\ 0&0&0&0&\cdots&\sin m\theta_{d/2}&\cos m\theta_{d/2}\end{pmatrix},\quad \Theta=\{\theta_i=10000^{-2(i-1)/d},\ i\in[1,2,\dots,d/2]\}\tag{15}$$

$$\boldsymbol q_m^{\intercal}\boldsymbol k_n=(\boldsymbol R^d_{\Theta,m}\boldsymbol W_q\boldsymbol x_m)^{\intercal}(\boldsymbol R^d_{\Theta,n}\boldsymbol W_k\boldsymbol x_n)=\boldsymbol x^{\intercal}\boldsymbol W_q R^d_{\Theta,n-m}\boldsymbol W_k\boldsymbol x_n,\qquad \boldsymbol R^d_{\Theta,n-m}=(\boldsymbol R^d_{\Theta,m})^{\intercal}\boldsymbol R^d_{\Theta,n}\tag{16}$$

（式 16 最右端按印刷转录，少了下标 $m$ 与 $\boldsymbol W_q$ 的转置，正确应为 $\boldsymbol x_m^{\intercal}\boldsymbol W_q^{\intercal}\boldsymbol R^d_{\Theta,n-m}\boldsymbol W_k\boldsymbol x_n$，见 §7 E1。）

§3.4.1 的二维推导（注意此处相对位置写成 $n-m$，与式 11 的 $m-n$ 方向相反）：

$$\boldsymbol q^{\intercal}_m\boldsymbol k_n=\langle f_q(\boldsymbol x_m,m),f_k(\boldsymbol x_n,n)\rangle=g(\boldsymbol x_m,\boldsymbol x_n,n-m)\tag{21}$$
$$\boldsymbol q=f_q(\boldsymbol x_q,0),\quad \boldsymbol k=f_k(\boldsymbol x_k,0)\tag{22}$$
$$f_q(\boldsymbol x_q,m)=R_q(\boldsymbol x_q,m)e^{i\Theta_q(\boldsymbol x_q,m)},\ f_k(\boldsymbol x_k,n)=R_k(\boldsymbol x_k,n)e^{i\Theta_k(\boldsymbol x_k,n)},\ g=R_g(\cdot,n-m)e^{i\Theta_g(\cdot,n-m)}\tag{23}$$
$$R_q(\boldsymbol x_q,m)R_k(\boldsymbol x_k,n)=R_g(\boldsymbol x_q,\boldsymbol x_k,n-m),\qquad \Theta_k(\boldsymbol x_k,n)-\Theta_q(\boldsymbol x_q,m)=\Theta_g(\boldsymbol x_q,\boldsymbol x_k,n-m)\tag{24}$$
$$\boldsymbol q=\|\boldsymbol q\|e^{i\theta_q}=R_q(\boldsymbol x_q,0)e^{i\Theta_q(\boldsymbol x_q,0)},\qquad \boldsymbol k=\|\boldsymbol k\|e^{i\theta_k}=R_k(\boldsymbol x_k,0)e^{i\Theta_k(\boldsymbol x_k,0)}\tag{25}$$
$$R_q(\boldsymbol x_q,m)R_k(\boldsymbol x_k,m)=R_g(\boldsymbol x_q,\boldsymbol x_k,0)=R_q(\boldsymbol x_q,0)R_k(\boldsymbol x_k,0)=\|\boldsymbol q\|\|\boldsymbol k\|\tag{26a}$$
$$\Theta_k(\boldsymbol x_k,m)-\Theta_q(\boldsymbol x_q,m)=\Theta_g(\boldsymbol x_q,\boldsymbol x_k,0)=\Theta_k(\boldsymbol x_k,0)-\Theta_q(\boldsymbol x_q,0)=\theta_k-\theta_q\tag{26b}$$
$$R_q(\boldsymbol x_q,m)=R_q(\boldsymbol x_q,0)=\|\boldsymbol q\|,\quad R_k(\boldsymbol x_k,n)=R_k(\boldsymbol x_k,0)=\|\boldsymbol k\|,\quad R_g(\boldsymbol x_q,\boldsymbol x_k,n-m)=R_g(\boldsymbol x_q,\boldsymbol x_k,0)=\|\boldsymbol q\|\|\boldsymbol k\|\tag{27}$$
（式 26a、26b、27 的中间项是核查时按原文补全的；原文 26a/26b 由“在式 24 中令 $m=n$ 并代入初始条件式 25”得到。）
$$\Theta_f(\boldsymbol x_{\{q,k\}},m)=\phi(m)+\theta_{\{q,k\}}\tag{28}$$
$$\phi(m+1)-\phi(m)=\Theta_g(\boldsymbol x_q,\boldsymbol x_k,1)+\theta_q-\theta_k\tag{29}$$
$$\phi(m)=m\theta+\gamma\tag{30}$$
$$f_q(\boldsymbol x_q,m)=\|\boldsymbol q\|e^{i\theta_q+m\theta+\gamma}=\boldsymbol q e^{i(m\theta+\gamma)},\quad f_k(\boldsymbol x_k,n)=\|\boldsymbol k\|e^{i\theta_k+n\theta+\gamma}=\boldsymbol k e^{i(n\theta+\gamma)}\tag{31}$$
（式 31 中间一项按印刷转录，指数少了括号与 $i$：应为 $e^{i(\theta_q+m\theta+\gamma)}$。）
$$\boldsymbol q=f_q(\boldsymbol x_m,0)=\boldsymbol W_q\boldsymbol x_n,\quad \boldsymbol k=f_k(\boldsymbol x_n,0)=\boldsymbol W_k\boldsymbol x_n\tag{32}$$
（式 32 第一式按印刷转录，$\boldsymbol x_n$ 应为 $\boldsymbol x_m$。）
$$f_q(\boldsymbol x_m,m)=(\boldsymbol W_q\boldsymbol x_m)e^{im\theta},\quad f_k(\boldsymbol x_n,n)=(\boldsymbol W_k\boldsymbol x_n)e^{in\theta}\tag{33}$$

高效实现（§3.4.2）：

$$\boldsymbol R^d_{\Theta,m}\boldsymbol x=\begin{pmatrix}x_1\\x_2\\x_3\\x_4\\\vdots\\x_{d-1}\\x_d\end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_1\\\cos m\theta_1\\\cos m\theta_2\\\cos m\theta_2\\\vdots\\\cos m\theta_{d/2}\\\cos m\theta_{d/2}\end{pmatrix}+\begin{pmatrix}-x_2\\x_1\\-x_4\\x_3\\\vdots\\-x_d\\x_{d-1}\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_1\\\sin m\theta_1\\\sin m\theta_2\\\sin m\theta_2\\\vdots\\\sin m\theta_{d/2}\\\sin m\theta_{d/2}\end{pmatrix}\tag{34}$$

远程衰减（§3.4.3）：

$$(\boldsymbol R^d_{\Theta,m}\boldsymbol W_q\boldsymbol x_m)^{\intercal}(\boldsymbol R^d_{\Theta,n}\boldsymbol W_k\boldsymbol x_n)=\operatorname{Re}\Big[\sum_{i=0}^{d/2-1}\boldsymbol q_{[2i:2i+1]}\boldsymbol k^{*}_{[2i:2i+1]}e^{i(m-n)\theta_i}\Big]\tag{35}$$

记 $h_i=\boldsymbol q_{[2i:2i+1]}\boldsymbol k^*_{[2i:2i+1]}$，$S_j=\sum_{i=0}^{j-1}e^{i(m-n)\theta_i}$，$h_{d/2}=0$，$S_0=0$：

$$\sum_{i=0}^{d/2-1}\boldsymbol q_{[2i:2i+1]}\boldsymbol k^{*}_{[2i:2i+1]}e^{i(m-n)\theta_i}=\sum_{i=0}^{d/2-1}h_i(S_{i+1}-S_i)=-\sum_{i=0}^{d/2-1}S_{i+1}(h_{i+1}-h_i)\tag{36}$$

$$\Big|\sum_{i=0}^{d/2-1}\boldsymbol q_{[2i:2i+1]}\boldsymbol k^{*}_{[2i:2i+1]}e^{i(m-n)\theta_i}\Big|=\Big|\sum_{i=0}^{d/2-1}S_{i+1}(h_{i+1}-h_i)\Big|\le\sum_{i=0}^{d/2-1}|S_{i+1}||h_{i+1}-h_i|\le\Big(\max_i|h_{i+1}-h_i|\Big)\sum_{i=0}^{d/2-1}|S_{i+1}|\tag{37}$$

图 2 画的量：$\frac{1}{d/2}\sum_{i=1}^{d/2}|S_i|$，$\theta_i=10000^{-2i/d}$。

线性注意力（§3.3）：

$$\operatorname{Attention}(\mathbf Q,\mathbf K,\mathbf V)_m=\frac{\sum_{n=1}^{N}\big(\boldsymbol R^d_{\Theta,m}\phi(\boldsymbol q_m)\big)^{\intercal}\big(\boldsymbol R^d_{\Theta,n}\varphi(\boldsymbol k_n)\big)\boldsymbol v_n}{\sum_{n=1}^{N}\phi(\boldsymbol q_m)^{\intercal}\varphi(\boldsymbol k_n)}\tag{19}$$

### 2.2 RoPE 的课程记号版

记号对照（style-guide §2）：

| 原文 | 课程 | 说明 |
|---|---|---|
| 位置 $m$（查询）、$n$（键） | $t$（当前）、$j$（历史，$j\le t$） | 因果掩码下相对距离写 $t-j\ge0$ |
| $d$（每头维度） | $d_h$；MLA 解耦部分写 $d_h^R$ | 不要与模型宽度 $d$ 混 |
| $\boldsymbol x_m$（“word embedding”） | $\mathbf h_t$（进入注意力层的隐藏状态） | Pre-Norm 下实际是 $\RMSNorm(\mathbf h_t)$ |
| $\boldsymbol W_q,\boldsymbol W_k$ | $W^Q,W^K$（单个头的切片） | 形状 $\R^{d_h\times d}$ |
| $\boldsymbol R^d_{\Theta,m}$ | $\mathbf R_{\Theta,t}\in\R^{d_h\times d_h}$ | 块对角正交矩阵 |
| $\theta_i$，$i=1..d/2$（§3.2.2）或 $0..d/2-1$（§3.4.3） | $\theta_i=b^{-2i/d_h}$，$i=0,\dots,d_h/2-1$，$b=10000$ | 全课统一 0 起下标；$b$ 即配置中的 `rope_theta` |
| 虚数单位 $i$ | $\mathrm i$（正体） | 避免与频率下标 $i$、头下标 $i$ 冲突（PI 论文就是这么写的） |

课程版公式（建议课文使用）：

$$\mathbf q_t=\mathbf R_{\Theta,t}W^Q\mathbf h_t,\qquad \mathbf k_j=\mathbf R_{\Theta,j}W^K\mathbf h_j,\qquad \mathbf R_{\Theta,t}=\diag\Big(\begin{bmatrix}\cos t\theta_i&-\sin t\theta_i\\ \sin t\theta_i&\cos t\theta_i\end{bmatrix}\Big)_{i=0}^{d_h/2-1}$$

$$\mathbf q_t^\top\mathbf k_j=(W^Q\mathbf h_t)^\top\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,j}(W^K\mathbf h_j)=(W^Q\mathbf h_t)^\top\mathbf R_{\Theta,j-t}(W^K\mathbf h_j)$$

复数形式：令 $\tilde q_i=q_{2i}+\mathrm i\,q_{2i+1}$，$\tilde k_i=k_{2i}+\mathrm i\,k_{2i+1}$（旋转前的分量，0 起下标），则

$$\mathbf q_t^\top\mathbf k_j=\operatorname{Re}\sum_{i=0}^{d_h/2-1}\tilde q_i\,\overline{\tilde k_i}\,e^{\mathrm i(t-j)\theta_i}$$

（推导：二维实向量 $z,w$ 看成复数时，$z^\top w=\operatorname{Re}(z\bar w)$；$\tilde q_ie^{\mathrm i t\theta_i}\cdot\overline{\tilde k_ie^{\mathrm i j\theta_i}}=\tilde q_i\overline{\tilde k_i}e^{\mathrm i(t-j)\theta_i}$。已在 Python 中数值验证：随机 $\mathbf q,\mathbf k\in\R^{128}$，$(t,j)=(37,11)$ 与 $(137,111)$ 得到相同内积 1.692714…，且等于 $(\mathbf R_{\Theta,26}\mathbf q)^\top\mathbf k$。）

**配对方式的两种实现（论文 vs 代码）**：
- 论文 / LLaMA v1 参考代码 / DeepSeek-V3 官方推理代码：**相邻配对** $(x_{2i},x_{2i+1})$，用 `view_as_complex(x.view(..., -1, 2))` 实现。
- Hugging Face LLaMA：**前后半配对** $(x_i,x_{i+d_h/2})$，用 `rotate_half`；转换脚本对 $W^Q,W^K$ 的行做了置换（`permute`，注释 “permute for sliced rotary”）使两者数学等价。
- DeepSeek 的 HF `modeling_deepseek.py`：权重保持相邻配对，在 `apply_rotary_pos_emb` 里先把 $\mathbf q,\mathbf k$ 从 `(d//2, 2)` 重排成前后半布局，再用 `rotate_half`。
- 结论：两种写法只差一个固定的维度置换，内积结果相同；但**不能**把一种布局训练的权重直接用另一种实现推理。

### 2.3 位置插值 PI（v2）原式

$$\mathbf f(\mathbf x,m)=\big[(x_0+\mathrm i x_1)e^{\mathrm i m\theta_0},(x_2+\mathrm i x_3)e^{\mathrm i m\theta_1},\dots,(x_{d-2}+\mathrm i x_{d-1})e^{\mathrm i m\theta_{d/2-1}}\big]^\top,\quad \theta_j=10000^{-2j/d}\tag{1}$$

$$a(m,n)=\mathrm{Re}\langle\mathbf f(\mathbf q,m),\mathbf f(\mathbf k,n)\rangle=\mathrm{Re}\Big[\sum_{j=0}^{d/2-1}(q_{2j}+\mathrm iq_{2j+1})(k_{2j}-\mathrm ik_{2j+1})e^{\mathrm i(m-n)\theta_j}\Big]$$
$$=\sum_{j=0}^{d/2-1}(q_{2j}k_{2j}+q_{2j+1}k_{2j+1})\cos((m-n)\theta_j)+(q_{2j}k_{2j+1}-q_{2j+1}k_{2j})\sin((m-n)\theta_j)=:a(m-n)\tag{2}$$

$$a(s)=\mathrm{Re}\Big[\sum_{j=0}^{d/2-1}h_je^{\mathrm is\theta_j}\Big],\quad h_j:=(q_{2j}+\mathrm iq_{2j+1})(k_{2j}-\mathrm ik_{2j+1})\tag{3}$$

$$\mathbf f'(\mathbf x,m)=\mathbf f\Big(\mathbf x,\frac{mL}{L'}\Big)\tag{4}$$

Theorem 2.1（Interpolation bound）：对 $a(s)=\mathrm{Re}[\sum_j h_je^{\mathrm is\theta_j}]$，$\theta_j=c^{-2j/d}$，$s\in[s_1,s_2]$，

$$|a(s)-a_{\mathrm{linear}}(s)|\le d\Big(\max_j|h_j|\Big)\frac{(s-s_1)(s_2-s)}{8\ln c}\tag{5}$$
$$a_{\mathrm{linear}}(s):=(1-\lambda(s))a(s_1)+\lambda(s)a(s_2),\quad \lambda(s):=\frac{s-s_1}{s_2-s_1}\tag{6}$$
$$|a(s)-a_{\mathrm{linear}}(s)|\le\frac{d}{32\ln c}\max_j|h_j|\approx\frac{d\max_j|h_j|}{294.73}\tag{7}$$
$$|a(s)|\le\Big(\max_j|h_j-h_{j+1}|\Big)\sum_{k=0}^{d/2-1}|A_{k+1}(s)|\le2\Big(\max_j|h_j|\Big)\sum_{k=0}^{d/2-1}|A_{k+1}(s)|,\quad A_k(s):=\sum_{j=0}^{k-1}e^{\mathrm is\theta_j}\tag{8}$$

注意：**PI 里的 $s$ 是相对距离**（positional span），YaRN 里的 $s$ 是缩放倍数。课文里必须换记号：相对距离写 $\Delta=t-j$，缩放倍数写 $s$。

课程记号：$L_{ctx}$ = 预训练上下文（PI 的 $L$），$L'_{ctx}$ = 目标上下文；$s=L'_{ctx}/L_{ctx}$；PI 等价于 $\theta_i\to\theta_i/s$（对所有 $i$）或位置 $t\to t/s$。

### 2.4 NTK-aware（YaRN v3 附录 A.2 的形式化；v2 为 §3.1 式 14–16）

$$g(m)=m\tag{17}$$
$$h(\theta_d)={b'}^{-2d/|D|}\tag{18}$$
$$b'=b\cdot s^{\frac{|D|}{|D|-2}}\tag{19}$$
$$s=\frac{L'}{L}\tag{20}$$
推导：令最后一个维度（原文说 “the last dimension $d\in D$ is $|D|-2$”）的波长恰好等于线性插值 $s$ 倍后的波长：
$${b'}^{\frac{|D|-2}{|D|}}=s\cdot b^{\frac{|D|-2}{|D|}}\tag{21}\qquad\Rightarrow\qquad b'=b\cdot s^{\frac{|D|}{|D|-2}}\tag{22}$$

课程记号：第 $i$ 对频率变为 $\theta_i'=(b')^{-2i/d_h}=\theta_i\cdot s^{-2i/(d_h-2)}$。于是 $i=0$ 不变（$\theta_0'=\theta_0=1$），$i=d_h/2-1$ 恰好缩小 $s$ 倍，中间按几何级数过渡。代码（`yarn-code` 的 `LlamaNTKScaledRotaryEmbedding`）：`base = base * alpha ** (dim / (dim-2))`，这里 `alpha` 就是 $s$。

### 2.5 YaRN 原式（v3 编号；方括号内为 v2 编号）

$$\mathrm{softmax}\Big(\frac{\boldsymbol q_m^T\boldsymbol k_n}{\sqrt{|D|}}\Big)\tag{2 [2]}$$
$$f_{\boldsymbol W}(\boldsymbol x_m,m,\boldsymbol\theta)=e^{im\boldsymbol\theta}\boldsymbol W\boldsymbol x_m,\quad \boldsymbol\theta=\mathrm{diag}(\theta_1,\cdots,\theta_{|D|/2}),\ \theta_d=b^{-2d/|D|},\ b=10000\tag{5 [5]}$$
$$f'_{\boldsymbol W}(\boldsymbol x_m,m,\boldsymbol\theta)=f_{\boldsymbol W}(\boldsymbol x_m,g(m),\boldsymbol h(\boldsymbol\theta))\tag{7 [12]}$$
$$\lambda_d=\frac{2\pi}{\theta_d}=2\pi b^{\frac{2d}{|D|}}\tag{8 [13]}$$
$$\text{PI: } g(m)=s\cdot m,\ \boldsymbol h(\boldsymbol\theta)=\boldsymbol\theta\quad\text{（v3 印刷如此；v2 正文为 } g(m)=m/s\text{）}\tag{9}$$
$$r(d)=\frac{L}{\lambda_d}=\frac{L}{2\pi b^{\frac{2d}{|D|}}}\tag{10 [17]}$$
（v2 式 17 的分母印成 $2\pi b'^{2d/|D|}$，带撇，见 §7 E6。）
$$\gamma(r)=\begin{cases}0,&\text{if }r<\alpha\\1,&\text{if }r>\beta\\ \dfrac{r-\alpha}{\beta-\alpha},&\text{otherwise}\end{cases}\tag{11 [18]}$$
NTK-by-parts（Definition 1 [v2 Definition 2]）：
$$g(m)=m\tag{12 [19]}$$
$$h(\theta_d)=\Big(1-\gamma\big(r(d)\big)\Big)\frac{\theta_d}{s}+\gamma\big(r(d)\big)\theta_d\tag{13 [20]}$$
注意力温度与推荐值：
$$\mathrm{softmax}\Big(\frac{\boldsymbol q_m^T\boldsymbol k_n}{t\sqrt{|D|}}\Big)\tag{14 [21]}$$
$$\sqrt{\frac1t}=0.1\ln(s)+1\tag{15 [22]}$$
v1 的对应式：$\sqrt t\approx0.1\ln(s)+1$（v1 式 27；v1 把 $t$ 定义为乘在注意力 logits 上的因子）。
附录 A.3：$s=8$ 时 $\sqrt{1/t}=0.1\ln(s)+1\approx1.208$（式 23）；困惑度相对变化 $\frac{\mathrm{ppl}(t)-\mathrm{ppl}(t=1)}{\mathrm{ppl}(t=1)}$（式 24）。
Dynamic Scaling：$s=\max(1,l'/L)$（§3.4 第 2 种方式）。

**课程记号版（建议）**：

| YaRN 原文 | 课程 | 冲突说明 |
|---|---|---|
| $|D|$ | $d_h$（LLaMA）或 $d_h^R$（DeepSeek） | |
| $d$（维度下标） | $i$（频率对下标，0 起） | YaRN 附录 A.2 又把 $d$ 当“原始维度号 0,2,…,|D|−2”用，见 §7 E9 |
| $L$，$L'$ | $L_{ctx}$，$L'_{ctx}$ | style-guide 规定 $L$ 是层数 |
| $s$ | $s$ | style-guide 把 $s$ 列为位置下标；本课位置只用 $t,j$，$s$ 专指缩放倍数 |
| $t$（温度） | 建议改写为 $\mu:=\sqrt{1/t}$（即代码里的 `mscale`），logits 乘 $\mu^2$ | $t$ 在全课是当前位置 |
| $\alpha,\beta$ | $\alpha,\beta$（“转圈数阈值”），代码名 `beta_slow`$=\alpha$、`beta_fast`$=\beta$ | 全课另有 GRPO 的 $\beta$、V3 的平衡损失 $\alpha$；本课首次出现时说明 |
| $\gamma(r)$ | $\gamma(r)$ | V3 用 $\gamma$ 表示偏置更新速度；RoFormer 用 $\gamma$ 表示相位常数 |

$$r_i=\frac{L_{ctx}}{\lambda_i}=\frac{L_{ctx}\,\theta_i}{2\pi},\qquad \theta_i'=\Big[\frac{1-\gamma(r_i)}{s}+\gamma(r_i)\Big]\theta_i,\qquad \mu=0.1\ln s+1$$
$$\alpha_{t,j}\propto\exp\Big(\mu^2\,\frac{\mathbf q_t'^{\top}\mathbf k_j'}{\sqrt{d_h}}\Big)\quad(\mathbf q',\mathbf k'\text{ 用 }\theta_i'\text{ 旋转})$$

### 2.6 YaRN 的代码实现（`yarn-code` 与 DeepSeek 代码同源）

代码把“比值 $r$”反解成“频率对下标”（函数 `find_correction_dim`），公式为

$$i(r)=\frac{d_h\,\ln\!\big(L_{ctx}/(2\pi r)\big)}{2\ln b}$$

（由 $r=L_{ctx}/(2\pi b^{2i/d_h})$ 直接解出，与论文式 10 一致。）然后：

- `low = floor(i(β))`，`high = ceil(i(α))`，并裁到 $[0,d_h-1]$；
- `ramp(i) = clip((i - low)/(high - low), 0, 1)`（**对下标 $i$ 线性**），`mask = 1 - ramp`；
- `inv_freq = (θ_i/s)·(1 - mask) + θ_i·mask`，即 `mask` 扮演论文的 $\gamma$；
- 原始 YaRN 代码：`mscale = 0.1·ln(s) + 1`，乘到缓存的 `cos`、`sin` 上（$\mathbf q,\mathbf k$ 各乘一次 → logits 乘 $\mu^2$）。

**论文与代码的差别**：论文的 $\gamma$ 对 $r$ 线性（$r$ 随 $i$ 指数变化），代码的 ramp 对下标 $i$ 线性，外加 floor/ceil 取整。两者“全保留 / 全插值 / 过渡”的**分界完全相同**（DeepSeek 设置下都是 11 / 9 / 12 对），但过渡区的混合权重差很多（例如 $i=16$：论文 $\theta'/\theta=0.1986$，代码 0.5500）。全表见 §4 R8。

### 2.7 DeepSeek-V2/V3：MLA 中的 RoPE 与 YaRN

V2 §2.1.3（式 14–19；V3 §2.1.1 的式 3、4、8–11 与之相同，对应关系：V2 (14)(15)(16)(17)(18)(19) = V3 (8)(3)(9)(4)(10)(11)）：

$$[\mathbf q_{t,1}^R;\mathbf q_{t,2}^R;\dots;\mathbf q_{t,n_h}^R]=\mathbf q_t^R=\operatorname{RoPE}(W^{QR}\mathbf c_t^Q)\tag{14}$$
$$\mathbf k_t^R=\operatorname{RoPE}(W^{KR}\mathbf h_t)\tag{15}$$
$$\mathbf q_{t,i}=[\mathbf q_{t,i}^C;\mathbf q_{t,i}^R]\tag{16}$$
$$\mathbf k_{t,i}=[\mathbf k_{t,i}^C;\mathbf k_t^R]\tag{17}$$
$$\mathbf o_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_j\Big(\frac{\mathbf q_{t,i}^T\mathbf k_{j,i}}{\sqrt{d_h+d_h^R}}\Big)\mathbf v_{j,i}^C\tag{18}$$
$$\mathbf u_t=W^O[\mathbf o_{t,1};\mathbf o_{t,2};\dots;\mathbf o_{t,n_h}]\tag{19}$$
其中 $W^{QR}\in\R^{d_h^Rn_h\times d_c'}$，$W^{KR}\in\R^{d_h^R\times d}$；缓存 $(d_c+d_h^R)l$ 个元素。已是课程记号，无需换。

DeepSeek 的 YaRN（V2 §3.1.4 / V3 §4.3）：

$$\text{V2: }\sqrt t=0.0707\ln s+1,\qquad \text{V3: }\sqrt t=0.1\ln s+1,\qquad s=40,\ \alpha=1,\ \beta=32$$

**DeepSeek 的 $\sqrt t$ 就是 YaRN v2/v3 的 $\sqrt{1/t}$（= 课程的 $\mu$）**，沿用的是 YaRN v1 的记号方向。

代码中的实际做法（`ds-v2-config`、`ds-v3-config`、HF `modeling_deepseek.py`，V2 与 V3 两份代码此处一致）：

- `yarn_get_mscale(scale, mscale) = 0.1·mscale·ln(scale) + 1`（scale ≤ 1 时返回 1）。
- 旋转部分：`cos`、`sin` 乘 `_mscale = yarn_get_mscale(s, mscale) / yarn_get_mscale(s, mscale_all_dim)`；两份配置都有 `mscale == mscale_all_dim`，所以 `_mscale = 1`，**旋转本身不缩放**。
- 注意力缩放：`softmax_scale = q_head_dim**-0.5 * m**2`，`m = yarn_get_mscale(s, mscale_all_dim)`，`q_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`。
  所以温度因子**作用在全部 192 维**（内容部分 $\mathbf q^C\!\cdot\mathbf k^C$ 与 RoPE 部分 $\mathbf q^R\!\cdot\mathbf k^R$ 都乘 $\mu^2$），而不是像原始 YaRN 那样通过 `cos/sin` 只缩放被旋转的维度。
  这正是论文“due to our distinct attention mechanism”一句的代码含义（笔者据代码解读；论文未展开）。
- V2：`mscale_all_dim = 0.707` → $\mu=0.0707\ln40+1=1.2608$，logits 乘 1.5896；V3：`mscale_all_dim = 1.0` → $\mu=1.3689$，logits 乘 1.8739。
- 频率：`qk_rope_head_dim = 64` 的 32 对频率按 §2.6 的代码 ramp 修改；$\mathbf q^R$ 与 $\mathbf k^R$ 用同一组修改后的频率（HF：`q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, ...)`；官方推理代码：`q_pe`、`k_pe` 都调用 `apply_rotary_emb(·, freqs_cis)`）。
- 官方 `inference/model.py` 只有当 `max_seq_len > original_seq_len (=4096)` 时才启用 YaRN 频率与 mscale；默认 `max_seq_len = 4096*4 = 16384`，因此默认就启用。YaRN 是**静态**的（固定 $s=40$，与输入长度无关），不是 Dynamic 版本。

### 2.8 RMSNorm 原式（v1）

$$a_i=\sum_{j=1}^{m}w_{ij}x_j,\quad y_i=f(a_i+b_i)\tag{1}$$
$$\bar a_i=\frac{a_i-\mu}{\sigma}g_i,\quad y_i=f(\bar a_i+b_i)\tag{2}$$
$$\mu=\frac1n\sum_{i=1}^{n}a_i,\quad \sigma=\sqrt{\frac1n\sum_{i=1}^{n}(a_i-\mu)^2}\tag{3}$$
$$\bar a_i=\frac{a_i}{\mathrm{RMS}(\mathbf a)}g_i,\quad\text{where }\mathrm{RMS}(\mathbf a)=\sqrt{\frac1n\sum_{i=1}^{n}a_i^2}\tag{4}$$
$$\mathbf y=f\Big(\frac{\mathbf{Wx}}{\mathrm{RMS}(\mathbf a)}\odot\mathbf g+\mathbf b\Big)\tag{5}$$
$$\mathrm{RMS}(\alpha\mathbf x)=\alpha\,\mathrm{RMS}(\mathbf x)\tag{6}$$
$$\mathbf y'=f\Big(\frac{\mathbf W'\mathbf x}{\mathrm{RMS}(\mathbf a')}\odot\mathbf g+\mathbf b\Big)=f\Big(\frac{\delta\mathbf{Wx}}{\delta\,\mathrm{RMS}(\mathbf a)}\odot\mathbf g+\mathbf b\Big)=\mathbf y\tag{7}$$
（式 6 对 $\alpha>0$ 成立；$\alpha<0$ 时应为 $|\alpha|$，论文未写这个条件。）

课程记号（Transformer 中的用法，作用在残差流上，而非论文的“神经元求和输入”）：

$$\RMSNorm(\mathbf x)=\frac{\mathbf x}{\sqrt{\frac1d\sum_{k=1}^{d}x_k^2+\epsilon}}\odot\mathbf g,\qquad \mathbf x,\mathbf g\in\R^d,\ \epsilon=10^{-6}\ (\text{DeepSeek 配置 } \texttt{rms\_norm\_eps})$$
$$\LayerNorm(\mathbf x)=\frac{\mathbf x-\bar x\mathbf 1}{\sqrt{\frac1d\sum_k(x_k-\bar x)^2+\epsilon}}\odot\mathbf g+\mathbf b$$

代码：LLaMA v1 `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` 后乘 `weight`（先转 float32 计算）；DeepSeek HF 版同式（注释称等价于 T5LayerNorm），官方推理代码直接调 `F.rms_norm`。都**只有增益 $\mathbf g$，没有偏置**。

### 2.9 GLU 家族原式（v1；行向量约定 $xW$）

$$\mathrm{FFN}(x,W_1,W_2,b_1,b_2)=\max(0,xW_1+b_1)W_2+b_2\tag{1}$$
$$\mathrm{FFN}_{\mathrm{ReLU}}(x,W_1,W_2)=\max(xW_1,0)W_2\tag{2}$$
$$\mathrm{FFN}_{\mathrm{GELU}}(x,W_1,W_2)=\mathrm{GELU}(xW_1)W_2,\quad \mathrm{FFN}_{\mathrm{Swish}}(x,W_1,W_2)=\mathrm{Swish}_1(xW_1)W_2\tag{3}$$
其中 $\mathrm{GELU}(x)=x\Phi(x)$，$\mathrm{Swish}_\beta(x)=x\sigma(\beta x)$。
$$\mathrm{GLU}(x,W,V,b,c)=\sigma(xW+b)\otimes(xV+c),\quad \mathrm{Bilinear}(x,W,V,b,c)=(xW+b)\otimes(xV+c)\tag{4}$$
$$\mathrm{ReGLU}=\max(0,xW+b)\otimes(xV+c),\ \mathrm{GEGLU}=\mathrm{GELU}(xW+b)\otimes(xV+c),\ \mathrm{SwiGLU}(x,W,V,b,c,\beta)=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)\tag{5}$$
$$\mathrm{FFN}_{\mathrm{GLU}}=(\sigma(xW)\otimes xV)W_2,\ \mathrm{FFN}_{\mathrm{Bilinear}}=(xW\otimes xV)W_2,\ \mathrm{FFN}_{\mathrm{ReGLU}}=(\max(0,xW)\otimes xV)W_2,$$
$$\mathrm{FFN}_{\mathrm{GEGLU}}=(\mathrm{GELU}(xW)\otimes xV)W_2,\ \mathrm{FFN}_{\mathrm{SwiGLU}}(x,W,V,W_2)=(\mathrm{Swish}_1(xW)\otimes xV)W_2\tag{6}$$

课程记号（列向量）：

$$\FFN_{\mathrm{SwiGLU}}(\mathbf x)=W_2\big(\mathrm{SiLU}(W_1\mathbf x)\odot W_3\mathbf x\big),\quad W_1,W_3\in\R^{d_{ff}\times d},\ W_2\in\R^{d\times d_{ff}},\quad \mathrm{SiLU}(z)=z\,\sigma(z)=\mathrm{Swish}_1(z)$$

命名对照：论文 $W$ ↔ LLaMA 代码 `w1` ↔ HF `gate_proj`；论文 $V$ ↔ `w3` ↔ `up_proj`；论文 $W_2$ ↔ `w2` ↔ `down_proj`。LLaMA 与 DeepSeek 代码都是 `w2(F.silu(w1(x)) * w3(x))`，配置 `hidden_act: "silu"`。

参数匹配：普通 FFN $2\cdot d\cdot 4d=8d^2$；SwiGLU $3\cdot d\cdot d_{ff}$；令二者相等得 $d_{ff}=\tfrac83d=\tfrac23\cdot4d$。

### 2.10 LLaMA 块（由 `llama-v1-code` 写出，论文只有文字描述）

$$\mathbf h=\mathbf x+\Attn\big(\RMSNorm_{\text{attn}}(\mathbf x)\big),\qquad \mathbf x_{\text{out}}=\mathbf h+\FFN_{\mathrm{SwiGLU}}\big(\RMSNorm_{\text{ffn}}(\mathbf h)\big)$$

最后一层之后再接一个 $\RMSNorm$，然后是输出投影（`self.norm`、`self.output`）。注意力内部：RoPE 只作用于 $\mathbf q,\mathbf k$（`apply_rotary_emb(xq, xk, ...)`），不作用于 $\mathbf v$；所有线性层 `bias=False`。
FFN 隐藏维：`hidden_dim = int(2 * 4d / 3)`，再向上取整到 `multiple_of = 256` 的倍数。

---

## 3. 规格与超参数表

### 3.1 LLaMA-1（论文 Table 2 + 代码/配置）

| 模型（Table 2 参数量） | $d$ | $n_h$ | $L$（层） | $d_h=d/n_h$ | 学习率 | batch | tokens | $d_{ff}$（代码规则，huggyllama 配置一致） | 上下文 |
|---|---|---|---|---|---|---|---|---|---|
| 6.7B | 4096 | 32 | 32 | 128 | 3.0e-4 | 4M | 1.0T | 11008 | 2048 |
| 13.0B | 5120 | 40 | 40 | 128 | 3.0e-4 | 4M | 1.0T | 13824 | 2048 |
| 32.5B | 6656 | 52 | 60 | 128 | 1.5e-4 | 4M | 1.4T | 17920 | 2048 |
| 65.2B | 8192 | 64 | 80 | 128 | 1.5e-4 | 4M | 1.4T | 22016 | 2048 |

来源：前八列 LLaMA v1 Table 2；$d_h$ 为推导；$d_{ff}$ 由 `llama-v1-code` 规则计算，并与 `huggyllama-configs` 的 `intermediate_size` 逐一吻合；上下文 2048 以 PI 图 1 说明和 YaRN §4.1 脚注为据（`llama-v1-code` 的 `max_seq_len: int = 2048` 只是推理默认值，旁证；huggyllama 配置 `max_sequence_length` 2048 亦为社区转存），**LLaMA 论文正文未写**。
其他：RoPE $b=10000$（代码 `theta: float = 10000.0`）；RMSNorm $\epsilon$：代码 `ModelArgs.norm_eps` 默认 1e-5、`RMSNorm` 类默认 1e-6，huggyllama 配置 7B/13B/30B 为 1e-6、65B 为 1e-5（官方 `params.json` 未核对）。优化：AdamW (0.9, 0.95)、cosine 到 10%、weight decay 0.1、梯度裁剪 1.0、warmup 2000 步（§2.3）。

### 3.2 DeepSeek LLM（2401.02954 Table 2 + HF 配置）

| 模型 | $L$ | $d$ | $n_h$ | $n_{kv}$ | 上下文 | batch（序列） | 学习率 | tokens | $d_{ff}$（配置） | $d_{ff}/d$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 7B | 30 | 4096 | 32 | 32 | 4096 | 2304 | 4.2e-4 | 2.0T | 11008 | 2.6875 |
| 67B | 95 | 8192 | 64 | 8 | 4096 | 4608 | 3.2e-4 | 2.0T | 22016 | 2.6875 |

配置另记：`rope_theta 10000`、`rms_norm_eps 1e-6`、`hidden_act silu`、`rope_scaling null`。

### 3.3 DeepSeek-V2 / V2-Lite / V3 与位置、归一化、FFN 相关的超参数

| 项目 | V2 论文 | V2 config | V2-Lite config | V3 论文 | V3 config |
|---|---|---|---|---|---|
| 层数 / $d$ | 60 / 5120（§3.1.2） | 60 / 5120 | 27 / 2048 | 61 / 7168（§4.2） | 61 / 7168 |
| $n_h$、$d_h$ | 128、128 | 128、`qk_nope_head_dim` 128 | 16、128 | 128、128 | 128、128 |
| $d_h^R$ | 64 | `qk_rope_head_dim` 64 | 64 | 64 | 64 |
| $d_c$ / $d_c'$ | 512 / 1536 | 512 / 1536 | 512 / null | 512 / 1536 | 512 / 1536 |
| RoPE 底数 $b$ | 未写 | `rope_theta` 10000 | 10000 | 未写 | 10000 |
| 预训练上下文 | 4K（§3.1.2） | `original_max_position_embeddings` 4096 | 4096 | 4K（§4.2） | 4096 |
| YaRN $s$ | 40 | `factor` 40 | 40 | 40 | 40 |
| YaRN $\alpha$ / $\beta$ | 1 / 32 | `beta_slow` 1 / `beta_fast` 32 | 1 / 32 | 1 / 32 | 1 / 32 |
| 温度系数 | $\sqrt t=0.0707\ln s+1$ | `mscale` = `mscale_all_dim` = 0.707 | 0.707 | $\sqrt t=0.1\ln s+1$ | 1.0 / 1.0 |
| 最大位置 | “160K”目标 | `max_position_embeddings` 163840 | 163840 | 128K | 163840 |
| 扩展训练 | 1000 步 × 32K × batch 576 | — | V2 附录只说 “we also perform long context extension”，无细节；附录对比表中 V2-Lite 的 Context Length 记为 32K（配置却与 V2 同为 163840 / factor 40） | 两阶段各 1000 步：32K×1920、128K×480，lr 7.3e-6 | — |
| RMSNorm $\epsilon$ | 未写 | 1e-6 | 1e-6 | 未写 | 1e-6 |
| 激活 | 未写 | `silu` | `silu` | SwiGLU（§3.3.3） | `silu` |
| 稠密 FFN $d_{ff}$ | 未写 | 12288（仅第 1 层，`first_k_dense_replace` 1） | 10944 | 未写 | 18432（前 3 层） |
| 专家 $d_{ff}$ | 1536 | 1536 | 1408 | 2048 | 2048 |
| 初始化 std | 0.006 | `initializer_range` 0.02 | 0.02 | 0.006 | 0.02 |

（最后一行是论文与配置的不一致：配置里的 `initializer_range` 对推理无影响，可能只是 HF 模板默认值；不属于本课重点，第 10/11 课可引用。）

### 3.4 YaRN 论文的实验与推荐超参数

| 项目 | 值 | 位置 |
|---|---|---|
| $\alpha,\beta$（Llama 家族） | 1, 32 | v3 §3.2 |
| $\sqrt{1/t}$ | $0.1\ln s+1$ | v3 式 (15) |
| 128k 模型 | Llama 2 7B/13B，$s=16$ 训练 400 步；$s=32$ 再训 200 步 | v3 §4.1 |
| 数据 | PG19，切成 64k 段（ablation 用 32k 段） | v3 §4.1 |
| 优化器 | AdamW $\beta_1=0.9,\beta_2=0.95$，lr $2\times10^{-5}$，无 weight decay，warmup 20 步，全局 batch 64 | v3 §4.1 |
| Table 1 PPL（Proof-pile，S=256） | 7B $s=32$：8192→3.56，16384→3.04，32768→2.70，65536→2.45，131072→2.37；7B $s=16$ 在 131072 处 $>10^1$ | v3 Table 1 |
| Table 4 训练时长（A100 GPU 小时） | LLaMA YaRN 2k×16：128；Llama 2 YaRN 4k×16：256；4k×32：256+128；PI（Chen et al.）2k×8：640；NTK-aware（Rozière et al.）4k×88.6：6400；Xiong et al. 4k×44.2：64000 | v3 Table 4 |

### 3.5 PI 论文的训练设置

7B–65B LLaMA；扩到 8192/16384/32768；PI 默认 1000 步、直接微调 10000 步；Pile 数据（另比较 RedPajama）；7B/13B/33B 扩到 8192 时 32 张 A100、全局 batch 64，其他情况 128 张 A100、batch 128；AdamW (0.9, 0.95)，lr 2e-5（7B/13B）或 1e-5（33B/65B），warmup 20 步从最大学习率的 10% 起，weight decay 0（§3.1）。

### 3.6 GLU 论文的实验设置

T5-base 编码器-解码器，各 12 层，$d_{model}=768$，$h=12$，$d_k=d_v=64$；基线 $d_{ff}=3072$，GLU 变体 $d_{ff}=2048$；预训练 524,288 步，batch 128 条，输入 512、输出 114 token；Adafactor + 反平方根学习率；预训练不用 dropout（§3.1–3.2）。

### 3.7 前瞻指针（仅配置，论文未在本课核对；第 15、18、19 课需各自核实）

| 模型 | `rope_scaling` | 其他 |
|---|---|---|
| V3.1 / V3.2-Exp / V3.2 | 与 V3 相同（factor 40，beta 32/1，mscale 1.0/1.0，original 4096） | `max_position_embeddings` 163840 |
| V4-Flash / V4-Pro | `yarn`，factor 16，beta_fast 32，beta_slow 1，`original_max_position_embeddings` 65536，**无 mscale 字段** | `max_position_embeddings` 1,048,576；`qk_rope_head_dim` 64；`rope_theta` 10000；另有 `compress_rope_theta` 160000；V4-Flash 与 V4-Pro 都有 `swiglu_limit` 10.0（核查时重读两份 config 确认） |
| V4.1-Flash | `text_config` 内同 V4（factor 16，original 65536） | `max_position_embeddings` 1,048,576 |

---

## 4. 复算的数字

全部在 Python 3.11 + NumPy 中计算（脚本存于会话临时目录，未入库）。

**R1. RoPE 频率与波长（$b=10000$）**
- `theta_i = 10000**(-2*i/d)`，`lambda_i = 2*pi/theta_i`。
- $d_h=128$：$\theta_0=1$，$\theta_1=0.865964$，$\theta_{63}=1.15478\times10^{-4}$；$\lambda$ 从 $2\pi=6.2832$ 到 **54,410.1** token；相邻频率比 $10000^{-2/128}=0.865964$。
- $d_h^R=64$：$\theta_1=0.749894$，$\theta_{31}=1.33352\times10^{-4}$；$\lambda$ 从 6.2832 到 **47,117.2** token；相邻比 $10000^{-2/64}=0.749894$。
- 每 4 对取样（$d_h=128$）：$i$=0,4,8,…,60,63 的 $\lambda$ = 6.28, 11.17, 19.87, 35.33, 62.83, 111.73, 198.69, 353.33, 628.32, 1117.33, 1986.92, 3533.29, 6283.19, 11173.26, 19869.18, 35332.95, 54410.14。
- $d_h^R=64$，$i$=0,4,…,28,31：6.28, 19.87, 62.83, 198.69, 628.32, 1986.92, 6283.19, 19869.18, 47117.24。
- 注意 $d_h^R=64$ 的第 $i$ 对等于 $d_h=128$ 的第 $2i$ 对（因为 $2i/64=4i/128$）。

**R2. 在预训练上下文内“转不满一圈”的维度数**（$\lambda_i>L_{ctx}$，即 $r_i<1$）
- $d_h=128$，$L=2048$：23 对（$i=41..63$）；$L=4096$：18 对（$i=46..63$）。
- $d_h^R=64$，$L=4096$：9 对（$i=23..31$）；$L=2048$：11 对。
- 最低频一对在 $t=L$ 处转过的角度：$d_h=128$、$L=4096$：`4096*10000**(-126/128)` = 0.4730 rad = 27.10°；$d_h^R=64$、$L=4096$：0.5462 rad = 31.30°；$d_h=128$、$L=2048$：13.55°。

**R3. YaRN 分界的下标**（$d_h^R=64$，$L=4096$，$b=10000$）
- $r=\beta=32$ 处：`64*ln(4096/(32*2*pi))/(2*ln(1e4))` = **10.4722**；$r=\alpha=1$ 处：`64*ln(4096/(2*pi))/(2*ln(1e4))` = **22.5134**。
- 代码取 `low=floor(10.4722)=10`，`high=ceil(22.5134)=23`。
- 因此：$i=0..10$（11 对，22 维）**完全不插值（外推）**；$i=23..31$（9 对，18 维）**完全插值**（频率 ÷40）；$i=11..22$（12 对，24 维）**过渡**。论文式 11 与代码给出相同的三段划分。
- 对照 $d_h=128$，$L=4096$，$s=40$：分界 20.94 / 45.03 → 21 对不动、18 对全插值、25 对过渡。$d_h=128$，$L=2048$（LLaMA-1）：17 / 23 / 24 对。
- 用波长表述：$\beta=32$ ↔ $\lambda=L/32=128$ token；$\alpha=1$ ↔ $\lambda=L=4096$ token。

**R4. 温度因子 $\mu$ 与 logits 乘数 $\mu^2$**
- YaRN / V3：`0.1*ln(40)+1` = **1.3689**，$\mu^2$ = **1.8739**（等价温度 $t=1/\mu^2=0.5337$）。
- V2：`0.0707*ln(40)+1` = **1.2608**，$\mu^2$ = **1.5896**。
- V3/V2 logits 乘数之比 1.8739/1.5896 = 1.1788。
- $s=8$：`0.1*ln(8)+1` = 1.2079（论文写 ≈1.208 ✓）。
- 其他 $s$：2→1.0693，4→1.1386，16→1.2773，32→1.3466，64→1.4159，128→1.4852（YaRN 系数）；V2 系数：2→1.0490，4→1.0980，8→1.1470，16→1.1960，32→1.2450，64→1.2940。
- `0.1*0.707` = 0.0707；$0.1/\sqrt2=0.070711$（两者一致到 4 位，**DeepSeek 为何取 0.707 未见说明**，见 §8）。

**R5. DeepSeek 的 softmax 缩放系数**
- 无 YaRN：`192**-0.5` = 0.072169。
- V2：`192**-0.5 * 1.2608**2` = 0.114721；V3：`192**-0.5 * 1.3689**2` = 0.135234。

**R6. NTK-aware 新底数 $b'=b\,s^{D/(D-2)}$**
- $D=128$：$s$=4→40,889.9；8→82,684.6；16→167,198.7；40→424,120.7。
- $D=64$：$s$=4→41,829.4；8→85,550.4；16→174,969.6；40→450,546.2。
- 验证：$(b'/b)^{(D-2)/D}=s$ 精确成立（最低频一对恰好缩小 $s$ 倍）。
- $D=64,s=40$ 每对的缩放 $\theta'_i/\theta_i=s^{-2i/62}$：$i$=0→1.0000，4→0.6213，8→0.3860，12→0.2398，16→0.1490，20→0.0926，23→0.0648，24→0.0575，28→0.0357，31→0.0250。

**R7. NTK-aware 的“越界外推”**（$D=64$，$L=4096$，$s=40$，$L'=163840$）
- $i=23$（$\lambda=4711.7>L$，本应只插值）：NTK 缩放 0.0648 > 1/40=0.025；在 $L'$ 处转角是预训练最大转角的 `163840*0.0648/4096` = **2.59 倍** → 仍在外推。
- 只有 $i=31$（缩放恰为 0.025）不越界。YaRN 附录 A.2 所说“some dimensions are slightly extrapolated”即此。

**R8. YaRN 逐对缩放表（DeepSeek：$d_h^R=64$，$L=4096$，$s=40$，$\alpha=1,\beta=32$）**

| $i$ | $\theta_i$ | $\lambda_i$ | $r_i$ | 论文 $\gamma$ | 论文 $\theta'/\theta$ | 代码 mask | 代码 $\theta'/\theta$ | 代码 $\lambda'_i$ |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.00000 | 6.28 | 651.90 | 1 | 1 | 1 | 1 | 6.3 |
| 5 | 2.37137e-1 | 26.50 | 154.59 | 1 | 1 | 1 | 1 | 26.5 |
| 10 | 5.62341e-2 | 111.73 | 36.659 | 1 | 1 | 1 | 1 | 111.7 |
| 11 | 4.21697e-2 | 149.00 | 27.490 | 0.8545 | 0.85816 | 0.9231 | 0.92500 | 161.1 |
| 12 | 3.16228e-2 | 198.69 | 20.615 | 0.6327 | 0.64192 | 0.8462 | 0.85000 | 233.8 |
| 13 | 2.37137e-2 | 264.96 | 15.459 | 0.4664 | 0.47976 | 0.7692 | 0.77500 | 341.9 |
| 14 | 1.77828e-2 | 353.33 | 11.593 | 0.3417 | 0.35815 | 0.6923 | 0.70000 | 504.8 |
| 15 | 1.33352e-2 | 471.17 | 8.6932 | 0.2482 | 0.26696 | 0.6154 | 0.62500 | 753.9 |
| 16 | 1.00000e-2 | 628.32 | 6.5190 | 0.1780 | 0.19858 | 0.5385 | 0.55000 | 1142.4 |
| 17 | 7.49894e-3 | 837.88 | 4.8886 | 0.1254 | 0.14730 | 0.4615 | 0.47500 | 1763.9 |
| 18 | 5.62341e-3 | 1117.33 | 3.6659 | 0.0860 | 0.10885 | 0.3846 | 0.40000 | 2793.3 |
| 19 | 4.21697e-3 | 1489.98 | 2.7490 | 0.0564 | 0.08001 | 0.3077 | 0.32500 | 4584.5 |
| 20 | 3.16228e-3 | 1986.92 | 2.0615 | 0.0342 | 0.05839 | 0.2308 | 0.25000 | 7947.7 |
| 21 | 2.37137e-3 | 2649.60 | 1.5459 | 0.0176 | 0.04217 | 0.1538 | 0.17500 | 15140.6 |
| 22 | 1.77828e-3 | 3533.29 | 1.1593 | 0.0051 | 0.03001 | 0.0769 | 0.10000 | 35332.9 |
| 23 | 1.33352e-3 | 4711.72 | 0.8693 | 0 | 0.025 | 0 | 0.025 | 188,469.0 |
| 26 | 5.62341e-4 | 11173.26 | 0.3666 | 0 | 0.025 | 0 | 0.025 | 446,930.4 |
| 31 | 1.33352e-4 | 47117.24 | 0.0869 | 0 | 0.025 | 0 | 0.025 | 1,884,689.7 |

（$i=1..9$ 与 $i=24..30$ 分别同 0 与 23 的规律；完整 32 行可用 §2.6 的公式重算。表达式：`r = 4096/(2*pi*1e4**(2*i/64))`；论文 `g = clip((r-1)/31, 0, 1)`（$r<1$ 取 0、$r>32$ 取 1），`scale_p = (1-g)/40 + g`；代码 `mask = 1 - clip((i-10)/13, 0, 1)`，`scale_c = (1-mask)/40 + mask`。）

**R9. RoPE 远程衰减上界（RoFormer 图 2 的量）** `mean(abs(cumsum(exp(1j*m*theta))))`
- $d=128$：$m$=0→32.500，1→31.538，2→28.956，5→20.845，10→17.954，20→15.242，50→12.629，100→10.227，200→7.245，256→6.543，500→6.190，1000→4.471，2000→5.047，4096→4.883，8192→4.902，16384→5.111，32768→3.803，65536→4.892。
- $d=64$：0→16.5，5→10.428，50→5.733，100→7.465，256→3.844，1000→4.090，4096→5.287。
- 观察：$m=0$ 时等于 $(d/2+1)/2$；约 1000 之后**不再系统性下降**：局部均值停在约 5（$d=128$，逐 512 窗口均值：1024 起 5.43，2048 起 5.00，3072 起 4.94），但**逐点值起伏很大**——$m\in[1000,4096]$ 内最小 2.23、最大 11.39（5%–95% 分位 2.83–8.27，中位数 4.68）；$[16384,65536]$ 内 1.21–14.41。上面列出的整数取样点恰好都落在 4–5 附近，不能据此说“在 4–5 附近起伏”。即使在 0–256 内曲线也不是单调的（200–300 之间 6.10–10.27）。
  （核查时补算：`m=np.arange(a,b+1)` 全量计算后取 min/max/分位数。）

**R10. 同向向量的 RoPE 得分随距离的变化**（$\mathbf q=\mathbf k=\mathbf 1$，得分 $=\sum_i2\cos(\Delta\theta_i)$，除以 $d$ 归一）
- $d=128$：$\Delta$=0→1.0000，1→0.9702，2→0.8966，5→0.7373，10→0.6691，20→0.6083，50→0.5462，100→0.4772，200→0.3057，500→0.2985，1000→0.1590，2000→0.0291，4096→−0.0529，16384→−0.1423，65536→−0.0075。
- 随机 $\mathbf q$、$\mathbf k=\mathbf q$ 的蒙特卡洛（2000 次）：$\E[\langle\mathbf R_\Delta\mathbf q,\mathbf q\rangle]/d$ = 1.0019（Δ=0），0.6704（10），0.4768（100），0.1611（1000），−0.0507（4096），与上一行一致。
- 独立随机 $\mathbf q,\mathbf k$ 的期望得分在任何距离都是 0（旋转不改变零均值）。

**R11. 相对性质与保模长的数值检查**
- 随机 $\mathbf q,\mathbf k\in\R^{128}$：$\langle\mathbf R_{37}\mathbf q,\mathbf R_{11}\mathbf k\rangle=\langle\mathbf R_{137}\mathbf q,\mathbf R_{111}\mathbf k\rangle=\langle\mathbf R_{26}\mathbf q,\mathbf k\rangle=1.692714…$；$\|\mathbf q\|=\|\mathbf R_{12345}\mathbf q\|=10.78394…$。

**R12. PI 误差界的数字**
- `32*ln(10000)` = **294.7309**（论文 294.73 ✓）；`2*294.73` = **589.46**（论文“∼600×”）。
- $B(s)=\sum_{k=0}^{63}|A_{k+1}(s)|$（$d=128$）：$s$=0→2080，100→654.5，1000→286.1，2048→231.6，4096→312.5，10000→246.9，均 ≥ $d=128$，与论文“numerically it is at least larger than d”一致（抽样检查，非证明）。

**R13. SwiGLU 参数匹配**
- T5-base：`2*768*3072` = 4,718,592 = `3*768*2048` ✓（2048 = 3072 × 2/3）。
- LLaMA：$\tfrac83d$ 与代码值：7B 10922.67→**11008**；13B 13653.33→**13824**；33B 17749.33→**17920**；65B 21845.33→**22016**。
- 取整后 SwiGLU 参数 / $8d^2$：7B 135,266,304 / 134,217,728 = **1.0078**；13B 1.0125；33B 1.0096；65B 1.0078。$d_{ff}/d$：2.6875、2.7000、2.6923、2.6875。
- PaLM（$d_{ff}=4d$ + SwiGLU）：$3\cdot d\cdot4d=12d^2$，比 $8d^2$ 多 50%。

**R14. DeepSeek 各模型 FFN 宽度比**
- DeepSeek LLM 7B/67B：11008/4096 = 22016/8192 = **2.6875**（与 LLaMA 同规则）。
- V2 稠密层 12288/5120 = **2.4000**；V3 稠密层 18432/7168 = **2.5714**；均**不是** 8/3。
- 专家：V2 1536/5120 = 0.3000；V3 2048/7168 = 0.2857。
- 单个 SwiGLU 的参数量 $3dd_{ff}$：V2 稠密层 188,743,680；V3 稠密层 396,361,728；V2 专家 23,592,960；V3 专家 44,040,192。

**R15. SwiGLU 每 token 每层 FLOPs（$2\times3dd_{ff}$）**
- LLaMA-7B：270,532,608；DeepSeek LLM 67B：1,082,130,432；V3 单个专家：88,080,384。

**R16. RMSNorm / LayerNorm 手算例**（$\mathbf x=(1,2,3,4)$，$\mathbf g=\mathbf1$，$\epsilon=0$）
- $\frac14\sum x^2=7.5$，RMS = 2.7386；RMSNorm = (0.3651, 0.7303, 1.0954, 1.4606)，输出的 RMS = 1。
- LayerNorm：均值 2.5，方差 1.25，标准差 1.1180；输出 (−1.3416, −0.4472, 0.4472, 1.3416)。
- 平移 $\mathbf x+10$：LayerNorm 输出不变；RMSNorm 变为 (0.8765, 0.9562, 1.0359, 1.1155)。
- 缩放 $3\mathbf x$：RMSNorm 输出不变 (0.3651, …)。
- 每个归一化层参数：RMSNorm $d$（仅增益），带偏置的 LayerNorm $2d$；$d=4096$ 时 4096 vs 8192。

**R17. 激活函数数值**（ReLU / GELU / SiLU）
- $x=-2$：0 / −0.0455 / −0.2384；$x=-1$：0 / −0.1587 / −0.2689；$x=0.5$：0.5 / 0.3457 / 0.3112；$x=1$：1 / 0.8413 / 0.7311；$x=2$：2 / 1.9545 / 1.7616；$x=4$：4 / 3.9999 / 3.9281。
- SiLU 最小值在 $x=-1.2785$，值 −0.2785（网格搜索步长 1e-5）。

**R18. DeepSeek 上下文长度的算术**
- `4096*40` = 163,840 = 160K（K=1024）= 配置 `max_position_embeddings` ✓；128K = 131,072。
- V4 配置：`65536*16` = 1,048,576 ✓。

**R19. 扩展阶段的 token 数**
- V2：`1000*32768*576` = **18,874,368,000**（≈18.9B），占 8.1T 预训练的 0.233%。
- V3 阶段一：`1000*32768*1920` = **62,914,560,000**；阶段二：`1000*131072*480` = **62,914,560,000**（两阶段每步 token 数相同：62,914,560）；合计 125.8B，占 14.8T 的 0.850%。
- 交叉核对 V3.1 模型卡：630B / 62.91B = 10.01（“10-fold” ✓）；209B / 62.91B = 3.32（“3.3x” ✓）。

**R20. V3 扩展阶段的成本**
- 119K GPU 小时 × 2 美元 = 238,000 美元（Table 1 “$0.238M” ✓）；占总 2,788K 的 **4.27%**。
- 若按预训练的 180K GPU 小时 / 万亿 token 计，125.8B token 只需 `0.1258*180e3` ≈ 22,649 GPU 小时；实际 119K，约 **5.25 倍**（长序列注意力更贵；笔者估算，未扣除其他开销，仅作量级参考）。

**R21. MLA 缓存中 RoPE 部分的占比**（与第 9 课交叉）
- V2：$(512+64)\times60$ = 34,560 元素/token = 69,120 字节（BF16）；V3：$(512+64)\times61$ = 35,136 元素 = 70,272 字节；RoPE 部分 64/576 = **11.1%**。

**R22. 二维旋转手算**（$\theta=0.3$，$\mathbf q=(1,0.5)$，$\mathbf k=(0.8,-0.2)$）
- $(t,j)=(5,2)$：$\mathbf R_{1.5}\mathbf q=(-0.4280,1.0329)$，$\mathbf R_{0.6}\mathbf k=(0.7732,0.2866)$，内积 **−0.0349**；$(13,10)$ 与 $(3,0)$ 得到同一值 −0.0349；不旋转时 $\mathbf q^\top\mathbf k=0.7$。
- $\theta=0.5$，$\mathbf q=\mathbf k=(1,0)$：得分 $\cos(0.5\Delta)$，$\Delta=2$ 时 0.5403，$\Delta=4$ 时 −0.4161。

**R23. 多频率“时钟”读数**（$d_h^R=64$，角度 mod 360°，$i$=0,4,…,28）
- $t=1$：57.3, 18.1, 5.7, 1.8, 0.6, 0.2, 0.1, 0.0。
- $t=100$：329.6, 11.9, 213.0, 181.2, 57.3, 18.1, 5.7, 1.8。
- $t=101$：26.9, 30.0, 218.7, 183.0, 57.9, 18.3, 5.8, 1.8。
- $t=4096$：323.5, 53.4, 68.4, 221.3, 186.8, 22.1, 234.7, 74.2。
- （快针每步转 57.3°，相邻位置靠快针区分；慢针在 4096 内转不满一圈，只能靠慢针区分“远近/绝对位置”。）

---

## 5. 图表清单与重绘建议

### 5.1 论文原图清单

| 来源 | 图号 | 标题/内容 | 是否建议重绘 |
|---|---|---|---|
| RoFormer v5 | Fig. 1 | “Implementation of Rotary Position Embedding(RoPE)”：上半部分 $d=2$ 时 $(x_1,x_2)$ 以角度 $m\theta_1$ 旋转成 $(x'_1,x'_2)$；下半部分 6 个 token（Enhanced Transformer with Rotary Position Embedding，位置 1–6）的 q/k 向量按 $\theta_1..\theta_{d/2}$ 分块着色，旋转后得到 “Position Encoded Query / Key” | 是（改绘为图 5.2-1、5.2-2） |
| RoFormer v5 | Fig. 2 | “Long-term decay of RoPE”：y 轴 “relative upper bound”（约 20 → 7），x 轴 “relative distance” 0–256；$d=128$（推导，见 §1 第 10 条） | 是（图 5.2-4，并延长到 4096 以显示下降停止、只剩起伏） |
| RoFormer v5 | Fig. 3 | BERT vs RoFormer 预训练损失；Performer 有无 RoPE 的损失 | 否 |
| PI v2 | Fig. 1 | 2048 预训练窗口；外推到 4096（红点）vs 把 [0,4096] 压回 [0,2048]（插值） | 是（图 5.2-6） |
| PI v2 | Fig. 2 | 用 $d=128$ 的三角基拟合得分函数：[0,2048] 内约在 [−1,1]，外推处可超过 8000；插值区间内平滑 | 可选（示意重绘，数值不可复现：论文用随机点最小二乘，附录 C.1 有代码） |
| PI v2 | Fig. 5（附录 B） | $B(s)/d$ 随 $s$ 下降但始终 ≥1 | 可选（R12 数据可画） |
| YaRN v3 | Fig. 1 | 方法关系图：RoPE → PI、NTK-aware → NTK-by-parts（“apply ramp function over dim”“keep the insight from NTK: scale dim differently!”）→ 加 “temperature scaling of attention” → YaRN | 是（图 5.2-7 的上方小图） |
| YaRN v3 | Fig. 2 | LLaMA 7B 扩到 32k 的训练损失（右图放大） | 否 |
| YaRN v3 | Fig. 3 | 滑窗 PPL 与 passkey 准确率，四种插值法对比 | 否 |
| YaRN v3 | Fig. 4 | $s=8$ 时 LLaMA 7b 困惑度 vs $1/\sqrt t$（1.05–1.5），阴影为 1 个标准差；曲线最低点在约 1.25–1.3，1.2–1.35 之间很平（目测）；图内标题写 “56 16k-token documents”，图注写 896 | 可选（只能示意；目测数值不可当数据） |
| YaRN v3 | Fig. 5、6 | 各位置段的 PPL 变化百分比；各 $1/\sqrt t$ 下取得最低 PPL 的样本数 | 否 |
| YaRN v3 | Fig. 7、8 | 1.28M token 文档滑窗 PPL；Dynamic-PI / Dynamic-YaRN 对比 | 否 |
| DeepSeek-V2 | Fig. 4 | NIAH 热力图，标题 “Pressure Testing DeepSeek-V2 Base 128K Context”，x 轴 1K–128K，y 轴深度 0–100%；约 35K 处（深度约 27–36%、64%）与约 104K（45%）有几个非满分格（目测） | 可选（示意） |
| DeepSeek-V3 | Fig. 8 | NIAH 热力图（SFT 之后），2K–128K 全部满分 | 可选 |
| RMSNorm | Fig. 1–7 | RNNSearch/阅读理解/检索等训练曲线 | 否 |
| GLU Variants | — | 无图，只有 Table 1–4 | — |
| LLaMA | Fig. 1 | 7B–65B 训练损失 vs token | 否 |

### 5.2 本课建议的 13 张图（数据都已给出，可直接在 Python 里算几何）

1. **图 5.1　二维旋转：RoPE 的最小单元**。平面上画 $\mathbf q=(1,0.5)$ 与 $\mathbf k=(0.8,-0.2)$（$\theta=0.3$）；三组 $(t,j)=(3,0),(5,2),(13,10)$ 分别画旋转后的向量对（R22 的坐标），标注“夹角相同 → 内积都是 −0.0349”。颜色：q 用 `--fig-blue`，k 用 `--fig-violet`，旋转弧用 `--fig-red` 细线。
2. **图 5.2　块对角旋转矩阵 $\mathbf R_{\Theta,t}$ 的结构**。$d_h=8$ 的 8×8 网格，4 个 2×2 块分别标 $\theta_0..\theta_3$（$=1, 0.1, 0.01, 0.001$，因 $10000^{-2i/8}=10^{-i}$），其余为 0。
3. **图 5.3　多频率时钟**。8 个表盘（$d_h^R=64$ 的 $i$=0,4,…,28），分三行画 $t=100$、$t=101$、$t=4096$ 的指针角度（R23）。每个表盘下标注波长（R1：6.28, 19.87, 62.83, 198.69, 628.32, 1986.92, 6283.19, 19869.18）。
4. **图 5.4　远程衰减（重绘 RoFormer 图 2 并延长）**。x 轴相对距离 0–4096（前段线性 0–256 小图 + 对数全图），y 轴 $\frac{1}{d/2}\sum|S_i|$，$d=128$（R9 取样点；实际绘制用 `m=np.arange(0,4097)` 全算）。叠加灰线 $d=64$。标注：约 1000 之后不再系统性下降——窗口均值约 5，但逐点值在约 2–11 之间剧烈起伏（R9）；建议同时画 512 点滑动均值，避免读者把锯齿误读成“平台”。
5. **图 5.5　同向向量的得分随距离**。$\sum_i2\cos(\Delta\theta_i)/d$，$d=128$，$\Delta$ 从 0 到 65536（对数 x 轴，R10 取样点），并在同一图上画 0 线表示“独立随机 q,k 的期望”。
6. **图 5.6　外推 vs 位置插值（改绘 PI 图 1）**。上：预训练位置点 0..2048（蓝），外推新增 2048..4096（红，标“没见过”）；下：4096 个位置按 $t/2$ 压进 [0,2048]（蓝+绿交错），标“点变密、没有越界”。
7. **图 5.7　波长阶梯与 YaRN 的三段**。横轴频率对 $i$（$d_h^R=64$，0–31），纵轴 $\log_{10}\lambda_i$（R1）；水平虚线 $\lambda=128$（$r=\beta=32$）与 $\lambda=4096$（$r=\alpha=1$）；三段着色：$i\le10$ 绿“保留”、$11..22$ 琥珀“过渡”、$\ge23$ 红“÷40”。第二条折线画 YaRN 后的 $\lambda'_i$（R8 代码列）。
8. **图 5.8　每对频率的缩放系数 $\theta'_i/\theta_i$：四种方法**。$d_h^R=64$，$s=40$，$L=4096$，纵轴对数（0.02–1）：PI 恒为 0.025；NTK-aware $40^{-2i/62}$（R6）；NTK-by-parts 论文 $\gamma$（R8 论文列）；代码 ramp（R8 代码列）。重点标出 $i=23..30$ 的 NTK-aware 高于 0.025（越界外推，R7）。
9. **图 5.9　斜坡函数 $\gamma(r)$**。x 轴 $r$（对数，0.05–1000），$\gamma$ 分段线性（$\alpha=1,\beta=32$）；把 32 个频率对的 $(r_i,\gamma_i)$ 画成点（R8 论文列），另用空心点画代码 mask 值，直观显示“同样的分界、不同的过渡形状”。
10. **图 5.10　注意力温度曲线**。x 轴 $s$（1–128，对数），两条曲线 $\mu=0.1\ln s+1$ 与 $0.0707\ln s+1$（R4），右轴画 $\mu^2$；标注 $s=8$（1.208）、$s=40$（V3：1.3689 / 1.8739；V2：1.2608 / 1.5896）。
11. **图 5.11　LLaMA / DeepSeek 式 Pre-RMSNorm 块结构**。残差主干（`--fig-ink`）→ RMSNorm → 注意力（q、k 上画 RoPE 小旋转符号，`--fig-red` 细线；v 不旋转）→ 加回主干 → RMSNorm → SwiGLU（三个矩阵 $W_1$ gate、$W_3$ up、$W_2$ down，SiLU 与逐元素乘）→ 加回；末端 final RMSNorm。旁注“所有线性层无偏置”。
12. **图 5.12　RMSNorm vs LayerNorm 的几何**。用 R16 的 $\mathbf x=(1,2,3,4)$ 画四组条形：原始、LayerNorm、RMSNorm、平移 +10 后两者的输出；说明 RMSNorm 把向量投到半径 $\sqrt d$ 的球面上（$\|\cdot\|_2=\sqrt d\cdot\mathrm{RMS}=2$，因输出 RMS=1、$d=4$）。
13. **图 5.13　SwiGLU 参数匹配与宽度比**。左：两个 $d\times4d$ 方块 vs 三个 $d\times\tfrac83d$ 方块（面积相等，$8d^2$）；右：各模型 $d_{ff}/d$ 条形图——LLaMA 7B 2.6875、DeepSeek LLM 2.6875、V2 稠密 2.4、V3 稠密 2.5714、PaLM 4（R13、R14），参考线 8/3=2.6667。
14. （可选）**图 5.14　DeepSeek 上下文扩展时间线**。V2：4K 预训练 8.1T → 32K×1000 步（batch 576，18.9B token）→ 评测到 128K；V3：4K 预训练 14.8T → 32K×1000 步（batch 1920，62.9B）→ 128K×1000 步（batch 480，62.9B），lr 7.3e-6，119K GPU 小时；V3.1：630B + 209B（R19）。
15. （可选）**图 5.15　激活函数**：ReLU、GELU、SiLU 在 [−4,4]（R17），标出 SiLU 最小值 (−1.2785, −0.2785)。

---

## 6. 算例与练习素材（附答案）

1. **（计算）RoPE 频率与波长。** 对 $d_h=128$、$b=10000$，求 $\theta_0,\theta_{32},\theta_{63}$ 与对应波长。
   答：$\theta_0=1$，$\lambda_0=2\pi\approx6.28$；$\theta_{32}=10000^{-64/128}=0.01$，$\lambda_{32}=628.3$；$\theta_{63}=10000^{-126/128}=1.1548\times10^{-4}$，$\lambda_{63}\approx54{,}410$。
2. **（计算）哪些维度在 4K 内转不满一圈？** $d_h^R=64$、$L_{ctx}=4096$。
   答：$\lambda_i>4096\iff10000^{i/32}>4096/2\pi=651.9\iff i>22.51$，即 $i=23..31$ 共 9 对（18 维）；最低频一对在 $t=4096$ 只转 31.3°。
3. **（计算）DeepSeek 的 YaRN 三段划分。** 用 $\alpha=1,\beta=32$、$s=40$。
   答：$r_i=\beta$ 于 $i=10.47$，$r_i=\alpha$ 于 $i=22.51$；$i=0..10$ 不变（11 对），$i=23..31$ 频率 ÷40（9 对），$i=11..22$ 过渡（12 对）。追问：$i=16$ 的新频率是多少？论文公式 $0.19858\times0.01=1.986\times10^{-3}$；DeepSeek 代码 $0.55\times0.01=5.5\times10^{-3}$。
4. **（计算）温度因子。** 分别按 V2 与 V3 的公式，$s=40$ 时 logits 被放大多少倍？
   答：V2 $\mu=1.2608$，放大 $\mu^2=1.5896$；V3 $\mu=1.3689$，放大 1.8739。追问：这是“升温”还是“降温”？答：logits 变大，softmax 变尖锐，等价于温度 $t=1/\mu^2<1$，是**降温**。
5. **（推导）为什么 RoPE 的内积只依赖 $t-j$？** 用 $\mathbf R_{\Theta,t}^\top\mathbf R_{\Theta,j}=\mathbf R_{\Theta,j-t}$（每个 2×2 块：$R(-t\theta)R(j\theta)=R((j-t)\theta)$，旋转角相加）写出三行推导。并用 R22 的数值（三组 $(t,j)$ 都得 −0.0349）验证。
6. **（计算）PI 的“600 倍”从哪来。** 答：插值界 $\frac{d}{32\ln c}\max|h_j|$，外推界 $\ge2d\max|h_j|$（因 $B(s)\ge d$），比值 $\ge64\ln10000=589.46\approx600$。
7. **（计算）NTK-aware 的底数。** $d_h=128$、$s=8$，求 $b'$；并验证最低频一对恰好缩小 8 倍。
   答：$b'=10000\cdot8^{128/126}=82{,}684.6$；$(b'/b)^{126/128}=8$ ✓。追问：$d_h^R=64$、$s=40$ 时，$i=23$ 的频率缩放了多少？是否越界？答：$40^{-46/62}=0.0648$，大于 1/40；在 163,840 位置处转角是预训练最大值的 2.59 倍，越界外推。
8. **（计算）SwiGLU 的 $\tfrac83$。** 证明参数相等时 $d_{ff}=\tfrac83d$；对 LLaMA-7B 算出代码实际 $d_{ff}$ 与参数误差。
   答：$8d^2=3dd_{ff}\Rightarrow d_{ff}=8d/3=10922.67$；向上取 256 的倍数得 11008；参数多 0.78%。追问：DeepSeek-V3 稠密层 18432 是否遵守这一规则？答：否，$18432/7168=2.571<8/3$。
9. **（计算）RMSNorm vs LayerNorm。** $\mathbf x=(1,2,3,4)$，$\mathbf g=\mathbf 1$，$\epsilon=0$。答见 R16；追问：$\mathbf x+10$ 时谁变谁不变？答：LayerNorm 不变，RMSNorm 变成 (0.8765, 0.9562, 1.0359, 1.1155)——这就是“放弃重中心化不变性”的代价。
10. **（计算）DeepSeek-V3 两阶段扩展的 token 数。** 答：每阶段 $1000\times32768\times1920=1000\times131072\times480=6.29\times10^{10}$；合计 1.258e11，占预训练 0.85%。追问：V3.1 模型卡说 32K 阶段 630B 是“10 倍”，与你的结果一致吗？答：630/62.9=10.01，一致。
11. **（概念）** YaRN 为什么要“不碰”高频维度？答：高频维度（$\lambda\ll L$）在预训练中已经转过很多圈，只携带相对位置信息，且负责区分相邻 token；把它们压缩 $s$ 倍会让相邻 token 的角度差缩小 $s$ 倍，模型分不清近邻顺序（YaRN §3.2）。低频维度（$\lambda\ge L$）从未转满一圈，其角度可被用作绝对位置，外推会产生没见过的角度，所以必须插值。
12. **（概念 / 陷阱）** “RoPE 保证注意力随距离衰减”对吗？答：不对。RoFormer 只证明了一个**上界**的平均量在下降，且只画到 256；该量约 1000 之后不再系统性下降，均值约 5、逐点在 2–11 之间起伏（R9）；PI 论文直言该界可能“vacuous”；独立随机 q、k 的期望得分在任何距离都是 0（R10）。衰减只在 q、k 方向相关时体现为“平均趋势”。

---

## 7. 勘误与常见误解

**原文排印错误 / 内部不一致**

- **E1. RoFormer 式 (16)**：印作 $\boldsymbol x^{\intercal}\boldsymbol W_qR^d_{\Theta,n-m}\boldsymbol W_k\boldsymbol x_n$，缺下标 $m$ 与转置，应为 $\boldsymbol x_m^{\intercal}\boldsymbol W_q^{\intercal}\boldsymbol R^d_{\Theta,n-m}\boldsymbol W_k\boldsymbol x_n$。式 (31) 中间项缺 $i$ 与括号；式 (32) 的 $\boldsymbol x_n$ 应为 $\boldsymbol x_m$。（v5 仍在。）
- **E2. RoFormer 的相对位置方向不统一**：式 (11)(12)(35) 用 $m-n$，式 (21)(24) 用 $n-m$；两种都对（取决于共轭哪一边），但课文要选一种（建议 $t-j$）并说明。频率下标也不统一：§3.2.2 为 1 起 $10000^{-2(i-1)/d}$，§3.3/§3.4.3 为 $10000^{-2i/d}$（0 起）。
- **E3. “远程衰减”被过度解读**：RoFormer §3.3 说“One can prove”，但 §3.4.3 只证明了上界不等式，衰减本身靠图 2 的数值展示（只到距离 256）。重算发现该量在约 1000 之后不再系统性下降（均值约 5，逐点在 2–11 之间起伏，R9）；PI 论文认为该界“may be too loose”乃至“vacuous”。正确说法：**RoPE 让 q、k 相关时的平均得分随距离呈下降趋势，但不保证单个注意力权重随距离递减。**
- **E4. RoFormer 的推导不是唯一性证明**：§3.4.1 只针对 $d=2$，并在 (27) 取“a straightforward solution”、在 (33) 前令 $\gamma=0$；高维的块对角结构与 $\theta_i$ 的取值是设计选择。课文避免写“唯一解”“必然推出”。
- **E5. YaRN v3 式 (9)** 把 PI 写成 $g(m)=s\cdot m$，与 v2 正文 $g(m)=m/s$、v3 附录 A.1 式 (16) $mL/L'$ 矛盾；正确是 $m/s$。
- **E6. YaRN v2 式 (17)** 的分母印成 $2\pi b'^{2d/|D|}$（带撇），v3 式 (10) 改为 $b$；应为原底数 $b$（代码 `find_correction_dim` 用的就是原 base）。
- **E7. YaRN 温度记号在版本间反转**：v1 写 $\sqrt t\approx0.1\ln s+1$，且把 $t$ 描述为乘在 logits 上的因子（“multiplying … by a temperature t>1”），却同时说目标是 “increase the "temperature" of the attention logits”（v1 §3.4）——乘以 $t>1$ 实际上是让 softmax 更尖锐，与“升温”的说法方向相反；v2/v3 改成 $\sqrt{1/t}=0.1\ln s+1$、$t$ 为除数。三者数值上都是“把 q、k 各乘 $0.1\ln s+1$”。**DeepSeek 论文沿用 v1 的 $\sqrt t$ 写法。**
- **E8. YaRN v3 图 4 标题与图注不一致**：图内标题 “56 16k-token documents”，图注与附录 A.3 正文为 896 篇。
- **E9. YaRN 的维度下标含义不统一**：§2.1 的 $\theta_d=b^{-2d/|D|}$ 里 $d$ 是“频率对”下标；附录 A.2 却说“the last dimension $d\in D$ is $|D|-2$”，此时 $d$ 是原始维度号（0, 2, …, |D|−2），对应 $\theta=b^{-d/|D|}$。$b'$ 公式本身在代码约定（`arange(0, dim, 2)/dim`）下正确。
- **E10. PI 的扩展倍数**：§3 开头写“up to 32 times of the original size”，但全文实验都是 2048→32768（16 倍）。
- **E11. DeepSeek-V3 说“YaRN 配置与 V2 一致”只对 $s,\alpha,\beta$ 成立**：温度系数 V2 为 0.0707、V3 为 0.1（两份论文原文 + `mscale_all_dim` 0.707→1.0 三方印证）。写课文时不要把 V2 的 0.0707 套到 V3 上，反之亦然。
- **E12. “YaRN 只作用在 $\mathbf k^R$ 上”的准确含义**：论文意在“只作用在承载 RoPE 的解耦部分、不碰内容部分 $\mathbf k^C$”。代码里 (a) $\mathbf q^R$ 与 $\mathbf k^R$ 都用修改后的频率（相对位置编码必须两边一致）；(b) 温度因子 $\mu^2$ 通过 `softmax_scale` 作用在**全部 192 维**的 logits 上，并不只作用于 RoPE 部分。
- **E13. GLU 论文的“reduce by a factor of 2/3”** 是乘以 2/3（3072→2048），不是“减少三分之二”。任务说明中“d_ff reduced by 2/3”的说法容易被误读。
- **E14. LLaMA 论文没写“无偏置”，也没写 2048 上下文**；“无偏置”来自官方代码（`bias=False`），2048 来自 PI、YaRN 的陈述（官方代码里的 `max_seq_len=2048` 只是推理默认值，旁证）。“无偏置”的明文出处是 PaLM §2 “No Biases”。课程地图第 5 课“无偏置”一项应引 PaLM + LLaMA 代码，而不是 LLaMA 论文。
- **E15. LLaMA 把 Pre-normalization 致谢给 [GPT3]**，但把 LayerNorm 移到子层输入早在 GPT-2 §2.3 就有。LLaMA 的方括号是“灵感来源”，不是“首创者”。
- **E16. 代码 ramp ≠ 论文 ramp**：论文 $\gamma$ 对 $r$ 线性，YaRN 原始代码与 DeepSeek 代码对频率下标线性（再 floor/ceil）。分界相同，过渡区权重不同（R8）。说“DeepSeek 用了论文式 (11)”不严谨，应说“用了 YaRN 参考实现”。
- **E17. RMSNorm 论文式 (4) 没有 $\epsilon$**，式 (6) 只对正的缩放因子成立；实现（LLaMA、DeepSeek）都在根号内加 $\epsilon$（DeepSeek 1e-6），并在 float32 中计算。RMSNorm 论文是在 RNN/Transformer 的“神经元求和输入”上提出的；LLaMA 式用法是对**残差流**做 Pre-Norm，属于后来的组合。

**常见误解**

- **M1. “YaRN 就是 NTK-aware / YaRN 改了 RoPE 底数。”** 错。改底数的是 NTK-aware；YaRN = NTK-by-parts（按波长分段插值频率）+ 注意力缩放（YaRN v3 Definition 2）。
- **M2. “YaRN 的温度是升温。”** 错。$\mu^2>1$ 放大 logits，softmax 更尖锐，等价温度 $t<1$。
- **M3. “PI 是外推。”** 错。PI 把位置压缩进已训练范围，是**插值**；外推指直接使用没见过的位置。
- **M4. “RoPE 加在输入嵌入上 / 也作用于 V。”** 错。RoPE 是乘性的、每层都作用于 q 和 k，不作用于 v（RoFormer 式 14、19；LLaMA 代码只旋转 `xq`、`xk`）。
- **M5. “SwiGLU 的隐藏维就是 8d/3。”** 只对“参数与 $4d$ FFN 相等”这一设计成立；LLaMA 实际向上取整到 256 的倍数（11008 等），PaLM 用 $4d$，DeepSeek-V2/V3 的稠密层分别为 2.4d、2.571d，MoE 专家更窄。
- **M6. “RMSNorm 只是 LayerNorm 去掉偏置。”** 不准确。核心是去掉**均值中心化**（不减均值、用 RMS 而非标准差）；LLaMA 式实现同时也没有偏置。均值为零时两者相等。
- **M7. “DeepSeek 的 128K 是训练到 160K 得来的。”** 不准确。V2 只在 32K 上训练 1000 步，按 $s=40$ 设定目标 160K、期望 128K 可用；V3 第二阶段才真正在 128K 序列上训练。配置中的 163,840 是 $4096\times40$。
- **M8. “DeepSeek 用 Dynamic YaRN。”** 未见证据。代码是静态 $s=40$；官方推理代码只在 `max_seq_len > 4096` 时启用 YaRN（默认 16384，故默认启用）。
- **M9. “RoPE 的两种实现（相邻配对 vs 前后半配对）结果不同。”** 数学上等价，只差固定置换；但权重必须与实现配套（HF 转换脚本的 `permute`、DeepSeek HF 代码的运行时重排）。
- **M10. “V2 的 NIAH 全绿。”** 图 4 标题显示是 Base 模型，热力图中约 35K、104K 处有少数非满分格（目测）；V3 图 8（SFT 后）才全部满分。论文正文“performs well across all context window lengths up to 128K”是概括。

---

## 8. 未能核实 / UNVERIFIED

1. **bloc97 帖子的发帖时间（核查后已大体确认）**：reddit.com 直连 403。核查时重读帖子 URL 的 Wayback 快照，`created-timestamp="2023-06-29T08:21:29.413000+0000"` 是 `<shreddit-post>` 元素本身的属性（不是评论），故发帖时间应为 **2023-06-29 08:21 UTC**；仍只是经 Wayback 间接读取。帖子图表的数值、Colab 代码未读到。
2. **NTK-aware 公式的最早出处**：帖子正文没有写公式；本笔记的 $b'=b\,s^{D/(D-2)}$ 来自 YaRN 附录 A.2 与 `yarn-code`，未能确认与 bloc97 原始 Colab 完全一致。
3. **DeepSeek 为何把系数取为 0.0707（=0.1×0.707≈0.1/√2），V3 又改回 0.1**：两份论文都只说“minimizing the perplexity”，无消融或解释。“0.707≈1/√2”是否有意为之，属笔者推测。
4. **DeepSeek 内部训练框架（HAI-LLM）的 YaRN 实现**：只看到开源推理代码（HF 与官方 `inference/model.py`）。训练时是否也用“对下标线性”的 ramp、是否同样把温度放进 `softmax_scale`，无法确认。
5. **V2 的 32K 扩展阶段数据构成与学习率**：V2 §3.1.4 只给步数、长度、batch，未给学习率与数据来源。
6. **YaRN 摘要“10x less tokens and 2.5x less training steps than previous methods”的比较口径（部分解决）**：YaRN v1 §3.4 写明两个倍数的对照对象——“10x reduction from [25]”（[25] = Code Llama）与“2.5x reduction in training steps from [7]”（[7] = PI）；v3 附录也说 YaRN 只训 400 步、PI 训 1000 步（1000/400 = 2.5 ✓）。**10 倍 token 的算法**（Code Llama 用了多少 token、YaRN 用了多少）仍未复核；Table 4 中 Together.ai 一行的 GPU 小时为“?”。
7. **YaRN 图 4 的最优 $1/\sqrt t$**：目测曲线最低在 1.25–1.3 附近，与公式给出的 1.208 略有出入；图只有图片，无数据，未能定量核对。
8. **LLaMA-1 各尺寸官方 `params.json` 中的 `norm_eps`、`multiple_of`**：只核对了社区转存的 huggyllama 配置（65B 为 1e-5，其余 1e-6）；官方文件未读到（Meta 仓库需申请）。
9. **Llama 2 的配置**：`meta-llama/Llama-2-7b-hf` 需登录（HTTP 401），未读到；“Llama 2 预训练 4K”只以 YaRN §4.1 脚注为据。
10. **RoFormer 期刊版（Neurocomputing 2024）与 arXiv v5 在公式/排印上的差异**：未读期刊全文。作者顺序已在核查时用 Crossref（DOI 10.1016/j.neucom.2023.127063，vol. 568，article 127063，2024-02）确认为 Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, Yunfeng Liu。
11. **PI 论文的发表状态**：未找到同行评审版本（按 arXiv 预印本处理）。
12. **RMSNorm 7%–64% 的提速**：2019 年框架与硬件上的测量，未复现；在现代融合核（如 `F.rms_norm`、Triton 核）上的差距未知。
13. **V4 / V4.1 的 RoPE 与 YaRN 设置**（factor 16、original 65536、`compress_rope_theta` 160000、无 mscale 字段）：只读了 `config.json`，未读论文与代码，含义留给第 18、19 课核实。
14. **苏剑林博客的首发日期**：页面显示 2021-03-23（Wayback 快照），与 RoFormer arXiv v1（2021-04-20）的先后关系据此判断；博客中“之前简要介绍过、当时称为‘融合式’”所指的更早文章未追溯。
15. **V2 图 4 的非满分格位置**：目测读图，未取得底层数据。核查时重看 arXiv v5 的原图（`needle_in_a_haystack.png`）：35K 列的深度 27%、36%、64% 三格为橙红色（按色标约 3 分），104K 列深度 45% 一格为黄绿色（约 7 分），其余全为 10 分；分数仍是按颜色估读。
16. **YaRN 温度为什么有效的机理**：v1 的解释（插值让注意力“spikier”、熵下降，因此要“升温”）与它实际做的操作（logits 乘 $\mu^2>1$，使分布更尖锐）方向相反；v2/v3 删去了这段解释，只保留“经验拟合 + 对各位置影响一致”（附录 A.3）。“插值使注意力变平、故需锐化补偿”是笔者推测，未见论文证据；课文应把 $\mu$ 表述为**经验规律**。
17. **DeepSeek-V2-Lite 的长上下文扩展**：V2 附录只有一句 “we also perform long context extension”，其对比表写 32K，而 HF 配置与 V2 相同（`max_position_embeddings` 163840、factor 40、mscale 0.707）。Lite 实际训练/评测到多长、是否同一套 YaRN 训练流程，未能核实。

---

## 9. 参考文献卡片草稿

```
[su2021roformer] RoFormer: Enhanced Transformer with Rotary Position Embedding
    a: Su, Lu, Pan, Murtadha, Wen, Liu (2021; Neurocomputing 568:127063, 2024). arXiv:2104.09864v5
    url: https://arxiv.org/abs/2104.09864
    finding: 从“q·k 只依赖相对位置”的要求出发，在二维用复数旋转构造出一个解，并推广为块对角旋转矩阵，频率 θ_i=10000^{-2i/d}。
    matters: 本课 RoPE 推导、频率“多尺度时钟”、远程衰减图与其局限（只是上界）都以此为原始出处。

[zhang2019rmsnorm] Root Mean Square Layer Normalization
    a: Zhang, Sennrich (2019, NeurIPS 2019). arXiv:1910.07467v1
    url: https://arxiv.org/abs/1910.07467
    finding: 去掉 LayerNorm 的均值中心化、只按 RMS 缩放，保留缩放不变性，在多种任务上精度相当而运行时间减少 7%–64%。
    matters: 解释 LLaMA/DeepSeek 为什么用 RMSNorm，以及“放弃平移不变性”具体丢掉了什么。

[shazeer2020glu] GLU Variants Improve Transformer
    a: Shazeer (2020, arXiv preprint). arXiv:2002.05202v1
    url: https://arxiv.org/abs/2002.05202
    finding: 在 T5 的 FFN 中用 GLU 变体替换 ReLU，SwiGLU/GEGLU 在参数与计算量匹配（d_ff×2/3）下困惑度最好。
    matters: SwiGLU 定义与 “2/3·4d” 参数匹配算术的原始出处；作者明言没有理论解释。

[touvron2023llama] LLaMA: Open and Efficient Foundation Language Models
    a: Touvron et al. (2023, arXiv preprint). arXiv:2302.13971v1
    url: https://arxiv.org/abs/2302.13971
    finding: 把 Pre-RMSNorm、SwiGLU（2/3·4d）和 RoPE 组合成一个开源模型家族（7B–65B，Table 2）。
    matters: “现代 Transformer 块”的标准模板；DeepSeek LLM 明确说沿用了它的微结构。

[chen2023pi] Extending Context Window of Large Language Models via Positional Interpolation
    a: Chen, Wong, Chen, Tian (2023, arXiv preprint). arXiv:2306.15595v2
    url: https://arxiv.org/abs/2306.15595
    finding: 把位置下标按 L/L' 线性压缩后微调约 1000 步，可把 LLaMA 扩到 32768，并给出插值误差界比外推界小约 600 倍。
    matters: 上下文扩展的第一个基线；它的“所有维度一样压缩”正是 NTK-aware 与 YaRN 要修正的问题。

[bloc97-2023ntk] NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation
    a: u/bloc97 (2023-06-29 08:21 UTC per the archived page's post timestamp, Reddit r/LocalLLaMA post). kind: artefact
    url: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ (read via Wayback snapshot 2023-06-29)
    finding: 提出改 RoPE 底数而非缩放位置，使高频维度几乎不变、低频维度被压缩，LLaMA 7B 不微调即可用到约 4K 以上。
    matters: YaRN 的直接前身；本课用它说明“不同频率应区别对待”这一思想的来源（按制品引用）。

[peng2023yarn] YaRN: Efficient Context Window Extension of Large Language Models
    a: Peng, Quesnelle, Fan, Shippole (2023, ICLR 2024). arXiv:2309.00071v3 (v1, v2 also read)
    url: https://arxiv.org/abs/2309.00071
    finding: 按波长把 RoPE 维度分为不插值/过渡/全插值三段（斜坡 γ(r)，α=1、β=32），再把 q、k 各乘 0.1 ln s+1，以约 0.1% 预训练数据扩展上下文。
    matters: DeepSeek-V2/V3 扩到 128K 所用的方法，其公式、默认超参数与版本间记号变化是本课的核心。

[deepseek2024v2] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
    a: DeepSeek-AI (2024, arXiv technical report). arXiv:2405.04434v5
    url: https://arxiv.org/abs/2405.04434
    finding: 在解耦 RoPE 的 k^R 上应用 YaRN（s=40, α=1, β=32，目标 160K），温度系数改为 0.0707 ln s+1，仅以 32K 训练 1000 步即在 128K 上通过 NIAH。
    matters: 说明 YaRN 如何与 MLA 的解耦 RoPE 结合，以及 DeepSeek 对原版 YaRN 的唯一改动。

[deepseek2024v3] DeepSeek-V3 Technical Report
    a: DeepSeek-AI (2024, arXiv technical report). arXiv:2412.19437v2
    url: https://arxiv.org/abs/2412.19437
    finding: 用两阶段 YaRN 扩展（4K→32K→128K，各 1000 步，batch 1920/480，lr 7.3e-6）以 119K GPU 小时获得 128K 上下文，温度系数为 0.1 ln s+1。
    matters: 提供本课复算的扩展 token 数与成本，并暴露“与 V2 一致”说法中温度系数的出入。

[deepseek2024llm] DeepSeek LLM: Scaling Open-Source Language Models with Longtermism
    a: DeepSeek-AI (2024, arXiv technical report). arXiv:2401.02954v1
    url: https://arxiv.org/abs/2401.02954
    finding: 7B/67B 沿用 LLaMA 的 Pre-Norm RMSNorm、SwiGLU（8/3 d）与 RoPE，67B 用 GQA，上下文 4096。
    matters: 把本课的“LLaMA 块”与 DeepSeek 科技树的起点直接连起来。

[chowdhery2022palm] PaLM: Scaling Language Modeling with Pathways
    a: Chowdhery et al. (2022). arXiv:2204.02311v5
    url: https://arxiv.org/abs/2204.02311
    finding: 使用 SwiGLU（d_ff 仍为 4d）、RoPE，并在所有稠密层与 LayerNorm 中去掉偏置以提升大模型训练稳定性。
    matters: “无偏置”设计的明文出处，也是 LLaMA 把 SwiGLU 宽度改为 2/3·4d 时的对照对象。

[radford2019gpt2] Language Models are Unsupervised Multitask Learners
    a: Radford et al. (2019, OpenAI technical report). kind: artefact (no arXiv)
    url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    finding: 把 LayerNorm 移到每个子层的输入，并在最后一个注意力块后加一层 LayerNorm。
    matters: 纠正“Pre-Norm 始于 GPT-3/LLaMA”的印象，承接第 1 课的 Pre-LN 讨论。

[su2021blog] Transformer升级之路：2、博采众长的旋转式位置编码
    a: 苏剑林 (2021-03-23, 科学空间 kexue.fm/archives/8265). kind: artefact
    url: https://kexue.fm/archives/8265
    finding: RoPE 的中文首发说明，提出“以绝对位置编码的方式实现相对位置编码”，远程衰减图注明 d=128。
    matters: 给出 RoFormer 图 2 未标注的维度，并记录 RoPE 早于论文的出处。

[yarn-code] jquesnelle/yarn reference implementation
    a: Quesnelle, Peng et al. (2023, GitHub). kind: artefact
    url: https://github.com/jquesnelle/yarn
    finding: 用 find_correction_range 把 β、α 换成频率下标，再用对下标线性的 ramp 混合 θ 与 θ/s，并把 mscale 乘进 cos/sin。
    matters: DeepSeek 代码的直接来源，也是“论文 ramp 与代码 ramp 不同”这一勘误的证据。

[ds-configs-code] DeepSeek-V2 / V2-Lite / V3 config.json and modeling_deepseek.py; DeepSeek-V3 inference/model.py
    a: DeepSeek-AI (2024, Hugging Face / GitHub). kind: artefact
    url: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
    finding: rope_scaling 为 yarn、factor 40、beta_fast 32、beta_slow 1、original 4096；V2 的 mscale_all_dim=0.707，V3 为 1.0，温度通过 softmax_scale 作用于全部 192 维。
    matters: 用代码核实两份论文的温度系数，并说明“只作用于 k^R”在实现上的准确含义。

[ds-v31-card] DeepSeek-V3.1-Base model card
    a: DeepSeek-AI (2025, Hugging Face). kind: artefact
    url: https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base
    finding: V3.1 沿用 V3 的两阶段扩展，32K 阶段增至 630B token（10 倍），128K 阶段增至 209B token（3.3 倍）。
    matters: 反向印证本笔记算出的 V3 两阶段各约 62.9B token，并展示扩展方法的延续。

[llama-v1-code] meta-llama/llama (branch llama_v1) llama/model.py
    a: Meta AI (2023, GitHub). kind: artefact
    url: https://github.com/meta-llama/llama/blob/llama_v1/llama/model.py
    finding: RMSNorm（rsqrt(mean(x²)+eps)）、复数形式的相邻配对 RoPE、w2(silu(w1 x)·w3 x) 的 SwiGLU、hidden_dim 取整到 256 的倍数、所有线性层 bias=False；ModelArgs 默认 max_seq_len 2048（推理默认值，example.py 用 512）。
    matters: LLaMA 论文没写的实现细节（无偏置、11008）以它为准；2048 上下文以 PI/YaRN 的陈述为主，它只作旁证。

[hf-llama-convert] huggingface/transformers convert_llama_weights_to_hf.py
    a: Hugging Face (GitHub). kind: artefact
    url: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
    finding: 转换时对 W_q、W_k 的行做 permute（“permute for sliced rotary”），把相邻配对改成前后半配对。
    matters: 解释为什么网上两种 RoPE 代码写法都“对”，以及权重与实现必须配套。
```

---

## 10. 核查记录（2026-09-24，独立复核）

做法：对下列论断逐条回到一手来源重读，不采信研究笔记原有的转述。论文用 `arxiv.py text/grep`（LaTeXML 全文）；需要比版本时分别拉 v1/v2/v3 全文并 `diff` 相关段落；配置与代码用 `curl` 直接读 HF `raw/main` 与 GitHub raw；图只能看原图 PNG；所有派生数字在 Python 里从头重算。

### 10.1 核对后确认无误（节选最重要的 24 组）

| # | 论断 | 一手来源（核查时实际读的） | 结果 |
|---|---|---|---|
| 1 | V2 §3.1.4：YaRN 只作用于 $\mathbf k_t^R$；$s=40,\alpha=1,\beta=32$；目标 160K、期望 128K；$\sqrt t=0.0707\ln s+1$；1000 步 × 32K × batch 576 | 2405.04434 v5 与 v1 全文，段落 `diff` 相同 | ✓ |
| 2 | V3 §4.3：两阶段各 1000 步；32K×1920、128K×480；lr $7.3\times10^{-6}$（= 预训练末段 lr，§4.2 亦确认）；$\sqrt t=0.1\ln s+1$；“consistent with that used in DeepSeek-V2” | 2412.19437 v2 与 v1，`diff` 相同 | ✓ |
| 3 | V3 Table 1：2664K + 119K + 5K = 2788K GPU 小时，$0.238M，$2/小时；每万亿 token 180K GPU 小时 | 2412.19437 v2 | ✓ |
| 4 | YaRN v3 式 (8)(9)(10)(11)(12)(13)(14)(15) 逐符号；Definition 1 = NTK-by-parts、2 = YaRN、3 = NTK-aware；$\alpha=1,\beta=32$；式 (9) 印作 $g(m)=s\cdot m$；附录 A.1 式 (16) 为 $mL/L'$ | 2309.00071 v3 | ✓（笔误确认存在） |
| 5 | YaRN v2：式 (17) 分母带撇 $b'$；v2 (18)(20)(21)(22) = v3 (11)(13)(14)(15)；v2 Definition 1/2/3 = NTK-aware / NTK-by-parts / YaRN；正文 PI 为 $g(m)=m/s$；v2 Fig. 2 = v3 Fig. 4 | 2309.00071 v2 | ✓ |
| 6 | YaRN v1 式 (27) $\sqrt t\approx0.1\ln s+1$；“multiplying … by a temperature $t>1$”；“increase the "temperature"”；插值使注意力“spikier” | 2309.00071 v1 §3.4 | ✓（v1 的解释方向与操作方向确实相反） |
| 7 | NTK-aware：v3 附录 A.2 式 (17)–(22)、“out-of-bound”、“the last dimension $d\in D$ is $|D|-2$”；代码 `base * alpha ** (dim / (dim-2))` | 2309.00071 v3；`jquesnelle/yarn` `LlamaNTKScaledRotaryEmbedding.py` | ✓ |
| 8 | YaRN §4.1：$s=16$ 400 步、batch 64、PG19 64k 段、lr $2\times10^{-5}$、$s=32$ 再训 200 步；Table 1 PPL；Table 4 GPU 小时；脚注“LLaMA 2k / Llama 2 4k” | 2309.00071 v3 | ✓ |
| 9 | 版本日期：YaRN v1 2023-08-31 / v2 2023-11-01 / v3 2026-02-06；RoFormer v5 2023-11-08；PI v1 06-27 / v2 06-28；V2 v5 2024-06-19；V3 v2 2025-02-18；YaRN 收于 ICLR 2024 | arXiv abs 页 submission history；proceedings.iclr.cc 2024 目录 | ✓ |
| 10 | PI 式 (4)–(8)、Theorem 2.1、294.73、“$2\cdot294.73\sim600\times$”、“may be too loose … vacuous”、“up to 32 times”、§3.1 训练设置、外推 PPL $>10^3$、图 1 的 2048 | 2306.15595 v2；`32*ln(1e4)=294.7309` | ✓ |
| 11 | RoFormer 式 (11)(12)(13)(15)(16)(21)–(25)(28)–(37)，含式 (16)(31)(32) 的印刷错误；§3.2.2 与 §3.3 两种 $\theta_i$ 下标；“a straightforward solution”、令 $\gamma=0$ | 2104.09864 v5 全文 | ✓（式 26a/26b/27 见 10.2） |
| 12 | RMSNorm 式 (4)(7)、Table 1 不变性（RMSNorm 行：权重矩阵缩放 ✓、权重向量缩放 ✗、数据平移 ✗）、“exactly equal to LayerNorm”、7%∼64% | 1910.07467 v1 | ✓ |
| 13 | GLU 式 (5)(6)、“by a factor of 2/3”、3072→2048、Table 1（524,288 步：ReLU 1.677、SwiGLU 1.636、GEGLU 1.633 最佳，ReGLU 1.645）、“divine benevolence” | 2002.05202 v1 | ✓ |
| 14 | LLaMA §2.2 三条（方括号是“inspiration”）、Table 2 全部数字、§2.3 优化设置；正文无“bias”架构描述、无 2048 上下文 | 2302.13971 v1 全文 grep | ✓ |
| 15 | PaLM “No biases were used in any of the dense kernels or layer norms.”；Table 1 注 $d_{ff}$ always $4\times d_{model}$ | 2204.02311 v5 | ✓ |
| 16 | GPT-2 §2.3：LayerNorm 移到每个 sub-block 的输入，最后自注意力块后加一层 LayerNorm | OpenAI 官方 PDF（pypdf 抽文本） | ✓ |
| 17 | DeepSeek LLM §2.2 “Pre-Norm structure with RMSNorm”、$\tfrac83d_{model}$、Table 2；HF 配置 11008 / 22016、`rope_theta` 10000、eps 1e-6 | 2401.02954 v1；HF `deepseek-llm-{7b,67b}-base` | ✓ |
| 18 | V2 / V2-Lite / V3 `config.json` 的全部 `rope_scaling` 字段、层数/宽度/头数、`intermediate_size` 12288/10944/18432、专家宽度、`initializer_range` 0.02 | HF raw config.json | ✓ |
| 19 | HF `modeling_deepseek.py`：`yarn_get_mscale`、`_mscale` = 比值、`softmax_scale = q_head_dim**-0.5 * mscale**2`、ramp 对下标线性且 floor/ceil、裁到 `[0, dim-1]`、`apply_rotary_pos_emb` 先重排 `(d//2,2)`；V2 与 V3 的 YaRN 代码 `diff` 仅类名不同 | HF raw modeling_deepseek.py（V2、V3） | ✓ |
| 20 | 官方 `inference/model.py`：`max_seq_len` 默认 4096*4、`original_seq_len` 4096、仅 `max_seq_len > original_seq_len` 时启用 YaRN 与 mscale²、相邻配对 `view_as_complex` | github.com/deepseek-ai/DeepSeek-V3 | ✓ |
| 21 | V3.1-Base 模型卡：32K 阶段“10-fold to 630B tokens”、128K 阶段“3.3x to 209B tokens”；V3.1/V3.2 `rope_scaling` 同 V3 | HF README 与 config | ✓ |
| 22 | V2 图 4 标题 “Pressure Testing DeepSeek-V2 Base 128K Context”；V3 图 8 全部满分且图注说在 SFT 之后 | arXiv HTML 原图 PNG | ✓ |
| 23 | YaRN v3 图 4 图内标题 “56 16k-token documents”（图注为 896）；曲线最低点约 1.25–1.3 | arXiv HTML 原图 `mscale_vs_ppl_o.png` | ✓（目测） |
| 24 | 复算：R1–R8（含 R8 全表逐格）、R10、R12–R23 全部重算一致；R3 三段划分（11/12/9；21/25/18；17/24/23）论文式与代码式一致；R4 $\mu$=1.3689/1.2608、$\mu^2$=1.8739/1.5896；R5 0.114721/0.135234；R7 2.5908；R19 18,874,368,000 与 62,914,560,000×2；R20 4.27%、5.25 倍 | Python 3.11 + NumPy | ✓ |

### 10.2 发现并已修改的问题

1. **RoFormer 式 (26a)(26b)(27) 转录不完整**：原笔记省略了原文的中间项 $R_q(\boldsymbol x_q,0)R_k(\boldsymbol x_k,0)$、$\Theta_k(\boldsymbol x_k,0)-\Theta_q(\boldsymbol x_q,0)$ 以及式 27 三行各自的“$=R_\cdot(\cdot,0)$”，并把 $R_g$ 的参数写成“$\cdot$”。已按 2104.09864v5 §3.4.1 逐符号补全（§2.1）。
2. **V3 MLA 公式编号对应写错**：原写“V3 §2.1.1 的式 3、8–11 相同”，漏了 $\mathbf k_{t,i}=[\mathbf k^C_{t,i};\mathbf k^R_t]$ 所在的 V3 式 (4)。已改为“式 3、4、8–11”并给出逐式对应（§2.7）。来源：2412.19437v2 式 (1)–(11)。
3. **R9“约 1000 之后在 4–5 附近起伏”不成立**：列出的整数取样点碰巧都在 4–5，但全量计算（$d=128$）显示 $[1000,4096]$ 内逐点值在 2.23–11.39 之间（5%–95% 分位 2.83–8.27），$[16384,65536]$ 内 1.21–14.41；只有滑动窗口均值稳定在约 5。已改 R9 的“观察”、图 5.4 的标注建议、§5.1 图表清单、§6 练习 12 与 §7 E3 的措辞。
4. **`swiglu_limit` 被写成 V4-Flash 独有**：重读配置，V4-Flash 与 V4-Pro 都有 `swiglu_limit: 10.0`。已改 §3.7。
5. **“LLaMA 2048 上下文来自官方代码”说法过强**：`llama_v1` 分支 `model.py` 里的 `max_seq_len: int = 2048` 只是推理 `ModelArgs` 的默认值（决定 KV 缓存与 `freqs_cis` 长度），`example.py` 的默认 `max_seq_len` 是 512；它不是训练上下文的声明。2K 的依据改为以 PI 图 1 说明与 YaRN §4.1 脚注为主、代码为旁证。已改 §1.4 第 26 条、§3.1 表注、§7 E14、`llama-v1-code` 卡片。

### 10.3 核查时顺带收窄的“未能核实”条目

- §8 第 1 条（bloc97 发帖时间）：帖子 URL 的 Wayback 快照中，`created-timestamp="2023-06-29T08:21:29.413000+0000"` 挂在 `<shreddit-post>` 元素上，是帖子本身的时间；原笔记猜测它“可能属于评论”不对。仍属间接读取。
- §8 第 6 条（YaRN “10x / 2.5x”）：v1 §3.4 写明对照对象（10 倍对 Code Llama、2.5 倍对 PI），v3 附录写 400 步 vs 1000 步；10 倍 token 的具体算法仍未复核。
- §8 第 10 条（RoFormer 期刊版作者顺序）：已用 Crossref 确认。
- §8 第 15 条（V2 NIAH 非满分格）：重看原图，给出了按色标估读的分数（约 3 分与约 7 分）。

### 10.4 本次仍未能独立复核的

- RoFormer 图 2 的坐标范围（“y 轴 20→约 7，x 轴 0–256”）：arXiv 没有 HTML 图片，PDF 中是矢量图，未能读图；只复算了曲线本身（$m=5$ 20.845、$m=50$ 12.629、$m=250$ 6.548 与笔记一致）。
- 苏剑林博客（kexue.fm/archives/8265）的日期与“d=128”图注：本次未重新打开。
- R11 的随机向量数值（1.692714…、10.78394…）：依赖未记录的随机种子，无法逐位复现；性质本身（相对性、保模长）已由 R22 的确定性算例复核。
- HF `convert_llama_weights_to_hf.py` 只确认了 `permute` 与注释 “permute for sliced rotary” 存在，未逐行读置换逻辑。
- PaLM 的 JMLR 发表信息、PI 是否有同行评审版本：未复核。

---

## 11. 补充核实（撰写第 5 课时追加，2026-09-24）

写课文时自己去一手来源核实、或自己在 Python 里算出的事实。引文仍按不超过约 15 个英文词摘录。

### 11.1 GPT-2（图 5.1、§1 表格与 §4 的 GPT-2 一侧）

| # | 事实 | 来源（实际读取的） | 原文短引 / 字段 |
|---|---|---|---|
| S1 | GPT-2 的上下文从 512 增加到 1024 个 token | OpenAI 官方 PDF（`radford2019gpt2`）§2.3，pypdf 抽文本 | “We also increase the context size from 512 to 1024 tokens” |
| S2 | GPT-2 small 配置：`n_positions` 1024、`n_embd` 768、`n_head` 12、`n_layer` 12、`activation_function` "gelu_new"、`layer_norm_epsilon` 1e-05 | `huggingface.co/openai-community/gpt2/raw/main/config.json` | 字段原值 |
| S3 | 位置表是可学习的 `nn.Embedding(max_position_embeddings, embed_dim)`，加在词嵌入上；前馈中间维在 `n_inner` 为 null 时取 `4 * hidden_size`；LayerNorm 用 `nn.LayerNorm`（默认带偏置）；投影层 `Conv1D` 有 `self.bias = nn.Parameter(torch.zeros(nf))` | transformers main 分支 `models/gpt2/modeling_gpt2.py` 与 `pytorch_utils.py`（curl raw.githubusercontent） | `self.wpe = nn.Embedding(...)`；`inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size` |

课文中这三条以新卡片 `hf-gpt2`（config + modeling 代码，artefact）和已有的 `radford2019gpt2` 引用。

### 11.2 偏置与词表（§1 表格、§5 参数账）

| # | 事实 | 来源 | 原文短引 / 字段 |
|---|---|---|---|
| S4 | DeepSeek-V3 官方推理代码的 `Linear` 默认不带偏置；注意力 `wq/wq_a/wq_b/wkv_a/wkv_b/wo` 与 MLP `w1/w2/w3` 构造时都不传 `bias` | `github.com/deepseek-ai/DeepSeek-V3` `inference/model.py`（curl raw） | docstring “Whether to include a bias term. Defaults to False.”；签名 `bias: bool = False` |
| S5 | DeepSeek LLM 7B 的 HF 配置声明架构为 `LlamaForCausalLM`，`num_hidden_layers` 30、`intermediate_size` 11008、`max_position_embeddings` 4096、`rms_norm_eps` 1e-06、`rope_theta` 10000、`rope_scaling` null；配置中没有 bias 字段 | `huggingface.co/deepseek-ai/deepseek-llm-7b-base/raw/main/config.json` | 字段原值。课文据此只写“沿用 LLaMA 结构”，**没有**断言 DeepSeek LLM 无偏置（HF LLaMA 实现的默认值未在本次核对） |
| S6 | LLaMA-7B 社区转存配置：`vocab_size` 32000、`tie_word_embeddings` false、`hidden_size` 4096、`intermediate_size` 11008、`num_hidden_layers` 32、`rms_norm_eps` 1e-06 | `huggingface.co/huggyllama/llama-7b/raw/main/config.json`（非 Meta 官方） | 字段原值 |

### 11.3 原始 Transformer 的正弦位置编码（§1、§3）

| # | 事实 | 来源 | 原文短引 |
|---|---|---|---|
| S7 | 正弦位置编码的波长构成从 $2\pi$ 到 $10000\cdot2\pi$ 的几何级数；作者设想固定偏移 $k$ 下 $PE_{pos+k}$ 是 $PE_{pos}$ 的线性函数，便于按相对位置注意 | arXiv:1706.03762 §3.5（LaTeXML 全文） | “hypothesized it would allow the model to easily learn to attend by relative positions” |

课文以新卡片 `vaswani2017` 引用。

### 11.4 课文中新算的数（Python 3.11 + NumPy；脚本在 `deepseek/figures/gen/l05_*.py` 或下列表达式）

- **R11 的随机种子找到了**（对应 §10.4 第三条“无法逐位复现”）：`rng = np.random.default_rng(0); q = rng.standard_normal(128); k = rng.standard_normal(128)`，相邻配对、$b=10000$，得 $\mathbf q^\top\mathbf k=0.346338$；$\Delta=26$ 的打分：$(26,0)$ 1.6927147759190864，$(37,11)$ 1.6927147759190904，$(137,111)$ 1.6927147759190857，$(1037,1011)$ 1.6927147759189332，$(100037,100011)$ 1.6927147759177315（最大差 $1.4\times10^{-12}$，浮点舍入）；复数公式同得 1.6927147759190833；$\norm{\mathbf q}=\norm{\mathbf R_{12345}\mathbf q}=10.783940328088976$。图 5.5 的生成脚本对 $48\times48$ 得分矩阵的每条对角线断言极差 $<10^{-9}$。
- **LLaMA-7B 参数逐项**：每层 $4d^2=67{,}108{,}864$，SwiGLU $3\cdot4096\cdot11008=135{,}266{,}304$，两个 RMSNorm $8{,}192$，合计 $202{,}383{,}360$；32 层 $6{,}476{,}267{,}520$；加最后 RMSNorm 4096 与不共享的嵌入/输出 $2\times32000\times4096=262{,}144{,}000$，总计 $6{,}738{,}415{,}616$（表 2 记 6.7B）。前馈占每层 66.8%。若每层带偏置：$4\cdot4096+2\cdot11008+4096+2\cdot4096=50{,}688$，32 层 1,622,016，占 0.024%。RMSNorm 增益共 $65\times4096=266{,}240$。
- **宽度比补充**：$\tfrac83\times7168=19114.67$，按 LLaMA 规则会取 19200；V3 稠密层实际 18432，参数 $396{,}361{,}728$ 比 $8d^2=411{,}041{,}792$ 少 3.57%；V2 稠密层 $188{,}743{,}680$ 比 $209{,}715{,}200$ 少 10%。DeepSeek LLM 67B：$21845.33\to22016$，多 0.78%。
- **128 维头套用 DeepSeek 的 YaRN 设置**（练习 8）：$i(32)=20.94$、$i(1)=45.03$，代码 low/high = 20/46，三段 21 / 25 / 18 对（与 §4 R3 的 128 维结果一致）；$i=30$：$\lambda=471.17$，$r=8.6932$，论文缩放 0.26696，代码 0.625，与 64 维 $i=15$ 完全相同。
- **二维 RoPE 练习**：$\theta=0.5$，$\mathbf q=(1,0)$，$\mathbf k=(0.6,0.8)$：$(4,1)$ 与 $(10,7)$ 都是 0.8404，$(1,4)$ 为 −0.7556。
- **归一化练习**：$\mathbf x=(3,-1,1,1)$：LayerNorm $(1.4142,-1.4142,0,0)$，RMSNorm $(1.7321,-0.5774,0.5774,0.5774)$；$\mathbf x+2$ 的 RMSNorm $(1.5076,0.3015,0.9045,0.9045)$。
- **温度的效果**：logits $(2,0)$ 的 softmax 第一项 0.8808；乘 1.5896 后 0.9600，乘 1.8739 后 0.9770。
- **远程衰减图**：$U(\Delta)$ 在 $[1000,4096]$ 上最小 2.23、最大 11.39（与 R9 一致）；图 5.6(b) 用居中 512 点滑动平均。
- **PI 例子的角度**：LLaMA（$d_h=128$）最慢一对在 2048 处 13.55°，外推到 4096 为 27.10°；最快一对每 token 57.30°，$s=2$ 插值后 28.65°，$s=16$ 后 3.58°，$s=40$ 后 1.43°。
- **MHA 对照**：V2 尺寸的标准 MHA 每 token $2\times128\times128\times60\times2=3{,}932{,}160$ 字节，是 MLA 69,120 字节的 56.9 倍。

### 11.5 笔者自己的推导（课文中已标注“笔者”）

- **模长可变的一族解**（§2 warn 框、练习 11）：若 $R_q(\mathbf q,m)=\norm{\mathbf q}a(m)$、$R_k(\mathbf k,m)=\norm{\mathbf k}/a(m)$，“$t\ne j$ 时模的乘积只依赖 $t-j$”在整数位置上迫使 $a(m)=\rho^m$，打分多出因子 $\rho^{t-j}$。RoFormer 的“straightforward solution”对应 $\rho=1$。未查到文献把这一族作为 RoFormer 推导的直接推广来陈述，只作为教学推导。
- **同向向量平均得分的闭式**（§3 (5.11)）：$\E[\mathbf q^\top\mathbf R_{\Theta,\Delta}\mathbf q]=\tr\mathbf R_{\Theta,\Delta}=\sum_i2\cos(\Delta\theta_i)$（$\E[\mathbf q\mathbf q^\top]=I$），与 R10 的数值一致。
- **吸收失败的最小反例**（§8）：$W^{UK}=(1,0)^\top$，$\mathbf q'=(0,1)$，$\theta=\pi/2$，$t=3$：各缓存位置 $j=0,1,2,3$ 的吸收系数为 $1,0,-1,0$。

### 11.6 本次撰写中注意到、但未改动研究笔记正文的问题

- 本课未使用 §5.2 建议图 13（激活函数曲线）与图 14（扩展时间线）：前者改成 §5 的数值表，后者改成 §7 的 V2/V3 对照表，以把全课控制在 15 张图。
- M03（`math/m03-rotation-complex.html`）尚未写成，课文对它的链接是课级链接（没有 `#pN`），`check_links.py` 目前报 1 个缺失文件。
  （2026-09-24 教学审校更新：M03 已写成，课文改为链接到具体页：§2 旋转矩阵、§3 复数、§4 欧拉公式、§5 相对角、§6 d 维旋转、§7 RoPE 算例。）

### 11.7 教学审校追加（2026-09-24）

课文新增的失败模式段落与算例所依据的事实。

| # | 事实 | 来源（实际读取的） | 原文短引 |
|---|---|---|---|
| S8 | 静态缩放（整个推理过程固定 $s=L'/L$）的两个毛病：长度小于 $L$ 时表现可能打折扣，超过 $L'$ 时突然变差；动态缩放 $s=\max(1,l'/L)$ 让模型平滑退化 | YaRN arXiv HTML（v3）§3.4；v2 同段逐字相同 | “a performance discount at a length less than L”；“an abrupt degradation when the sequence length is longer than L′”；“gracefully degrade instead of immediately breaking” |
| S9 | 附录 A.3 固定 $s=8$ 扫描 $1/\sqrt t$（LLaMA 7B，896 篇 16K 文档）；作者观察：最佳 $t$ 在样本与位置间大体一致，且随 $s$ 贴近式 (15) | YaRN v3 附录 A.3 正文与图 4–6 图注 | “the best value of t is mostly consistent across different samples and different positions” |

- 图 4 曲线的最低点位置（约 1.25–1.3，1.2–1.35 很平）仍只是 §6 表中的**目测**，课文照此标注“目测”。“偏大为什么有害”课文标为笔者的理解，论文没有给解释。
- **新算的数**（Python 3.11）：$\theta_{31}=1.33352\times10^{-4}$（64 维）；YaRN 后在 $t=163{,}840$ 的角度 $163840\,\theta_{31}/40=0.5462$ rad $=31.30°$（等于预训练最大角 $4096\,\theta_{31}$）；$t=200{,}000$ 时 $0.6668$ rad $=38.20°$；练习 9(d) 的 $t=327{,}680$ 时 $8192\,\theta_{31}=1.0924$ rad $=62.59°$，是预训练最大角的 2 倍。参考代码斜坡下被改动的对是 $i=11,\dots,31$，共 21 对（$i=10$ 的缩放为 1，$i=11$ 为 0.925，$i\ge23$ 为 0.025）。§1 的单频例子改用 RoPE（$\theta=0.5$，$\mathbf q=\mathbf k=(1,0)$）后，三处打分都是 $\cos0.5=0.87758$。$\mathrm{SiLU}(4)=3.92806$。
- “DeepSeek-V3 官方推理代码只要 `max_seq_len > 4096` 就启用 YaRN（默认 16384）”沿用本笔记 §2.7 已核实的条目，课文以 `ds-configs-code` 引用。“DeepSeek 论文里没有静态 YaRN 对短文本影响的单独对照”是笔者在 V2、V3 论文中没有找到，不是论文的陈述。

### 11.8 数学审校追加（2026-09-24）

逐条重推课文的推导、用 Python 3.11 + NumPy 重算全部数字后发现并已改正的问题。所有图的生成脚本重跑后与仓库里的 SVG 逐字节相同（改动的图除外）。

- **E18. PI 定理 2.1 的常数偏乐观（笔者核算，论文勘误）。** 读的是 arXiv:2306.15595v2 附录 A（PDF 经 pypdf 抽文本）。证明式 (13) 把 $\abs{a''}$ 放大到 $\max_j\abs{h_j}\sum_j c^{-4j/d}\le\max_j\abs{h_j}/(1-c^{-4/d})$，这一步成立；式 (14) 写 “when x < 0 and c > 1, c^x ≤ 1 + x ln c”，由此得 $1/(1-c^{-4/d})\le d/(4\ln c)$。但 $e^y\ge1+y$ 对所有 $y$ 成立（凸性），不等式方向反了，实际 $1/(1-c^{-4/d})\ge d/(4\ln c)$。
  - 数值：$d=128$、$c=10^4$：$\sum_{j=0}^{63}\theta_j^2=3.99831$，$d/(4\ln c)=3.47436$；$d=64$：2.28489 对 1.73718。
  - 反例（定理被违反）：$h_j=e^{-\mathrm i\theta_j/2}$（$\tilde k_j=1$、$\tilde q_j=e^{-\mathrm i\theta_j/2}$），$\Delta_1=0,\Delta_2=1$ 的中点误差 $\sum_j(1-\cos\frac{\theta_j}{2})=0.49388$（$d=128$），定理上界 $d/(32\ln c)=0.43429$。该误差也是 $\abs{h_j}\le1$ 时中点误差的最大值。
  - 改正：去掉式 (14) 的放大，$\abs{a-a_{\rm linear}}\le\max_j\abs{h_j}\big(\sum_j\theta_j^2\big)\frac{(s-s_1)(s_2-s)}{2}$，且 $\sum_j\theta_j^2<\frac{d}{4\ln c}+1$。相邻整数之间 $\le0.49979\max_j\abs{h_j}$，与外推界 $2d\max_j\abs{h_j}$ 之比 512.2（论文 589.5，“~600×”）。结论（插值比外推安全几百倍）不变。课文 §6 在定理框后加了警示框，§6 要点与 `chen2023pi` 卡片同步改为“约 500 倍（论文说约 600 倍）”。
- **(5.11) 的适用条件。** $\E[\mathbf q^\top\mathbf R_\Delta\mathbf k]=\tr(\mathbf R_\Delta\E[\mathbf k\mathbf q^\top])$；只有 $\E[\mathbf k\mathbf q^\top]=\kappa I$（$\kappa>0$）时才与“同向”曲线同形。课文原写“相关的 q、k 平均打分下降”，已补上“各对维度上同样正相关”这一假设（§3 式 (5.11) 注、err 框、要点、练习 2）。
- **同向平均得分第一次穿过 0 在 $\Delta=1707$**（$\Delta=1706$ 时 0.00107，1707 时 −0.00780），不是图 5.6 说明原写的“4096 附近”；$\Delta$ 从 11 到 12 有第一次回升（0.66163 → 0.66221）。
- **蒙特卡洛核对改为可复现**：`default_rng(0)` 抽 2000×128 标准正态，令 $\mathbf k=\mathbf q$：$\Delta$=0,10,100,1000,4096 的均值 1.0017、0.6683、0.4769、0.1576、−0.0537（种子 1：0.9970、0.6667、0.4758、0.1582、−0.0549）。R10 的 0.6704 种子不明，课文改用种子 0 的数。
- **$U(\Delta)$ 的逐点范围**：$[1000,4096]$ 为 2.23–11.39，$[1000,65536]$ 为 1.21–14.41；512 点滑动平均在 $\Delta$=1000, 2000, 4096, 8192, 16384, 32768 处为 5.66、4.97、4.72、4.58、4.37、5.08。练习 2 原写“逐点在 2 到 11 之间”只对 $[1000,4096]$ 成立，已加区间。
- **§1 的计算量比**：一条 128K 序列的打分 FLOPs 是一条 4K 序列的 1024 倍；同样 token 总数下是 32 倍。课文原句只写 1024 倍，已补上按 token 计的 32 倍。
- **吸收失败反例**：系数 $+1,0,-1,0$ 只有三个不同值，课文原写“四个不同的系数”，已改为“系数随 $j$ 变化”。图 5.15 第二行原写“$W^{UK}$ 的转置再乘 $R_{j-t}$”，旋转放错了一侧（那样应是 $R_{t-j}$），改为 (5.24) 中夹在中间的 $R_{j-t}W^{UK}$，查询标为旋转前的 $\mathbf q'_t$。
- **舍入链**：若读者把课文显示的四舍五入中间值相乘/相加会在末位得到不同结果的地方，已多写一位中间值：§1 例（$(1004,1003)$ 用 4 位小数相加得 3.0749，精确值 3.07482）；§7 $i=16$ 的 $\gamma=0.17803$；$s=8$ 的 $\mu=1.20794$（$1.2079^2=1.4590$，精确 $\mu^2=1.45913$）；等价温度写 $1/\mu^2=0.5337$（$1/1.8739=0.53365$）；(5.23) 用 $\mu^2=1.589626/1.873854$ 与 $1/\sqrt{192}=0.0721688$；练习 5 的 $e^{1.5\mathrm i}=0.070737+0.997495\mathrm i$；练习 8 的 $r_{30}=8.6932$、$\gamma=0.248168$；练习 9 的 V3 用 $\mu^2=1.87385$。

### 11.9 第二轮数学审校（2026-09-24）

用 Python 3.11 + NumPy 重新推导课文全部 `deriv` 与行间公式、重算全部数字（含练习答案与 15 个生成脚本；脚本重跑后与仓库 SVG 逐字节一致）。数字全部复核无误；改动的是几处“性质”的措辞与假设：

- **PI 的“外推界至少 $2d\max_j\abs{h_j}$”只是数值观察，且只在有限范围内成立。** PI 式 (8) 的外推界是 $2\max_j\abs{h_j}B(\Delta)$，$B(\Delta)=\sum_{k=0}^{d/2-1}\abs{A_{k+1}(\Delta)}=\tfrac d2U(\Delta)$。$d=128$ 全量计算：$\Delta\in[0,2048]$ 最小 180.2（$\Delta=1893$），$[2048,4096]$ 最小 142.8（$\Delta=3643$；步长 0.05 细网格 142.70），均 $\ge d$；但 $[4096,32768]$ 最小 80.5（$\Delta=22297$，有 887 个整数点 $<d$），$[32768,65536]$ 最小 77.2（$\Delta=61061$）。于是改正常数后“外推界/插值界”在 $\Delta\le4096$ 内 $\ge4.0017\times142.7\approx571$，更远处可低到约 309。课文 §6 警示框补了这一句，结论“安全几百倍”不变。
- **$U(\Delta)$ 的 512 点滑动平均**（居中窗口，与图 5.6b 相同）在 $[1000,4096]$ 为 4.60–5.67，在 $[4096,65536]$ 为 3.97–6.24（最大在 $\Delta\approx63460$）；$\Delta=256$ 时 8.67。原课文“停在 5 附近不再下降”改为“不再系统性下降，在 4.0–6.2 之间徘徊”，图 5.6b 的标注改为“之后基本持平”，err 框、要点与练习 2 同步。
- **“慢针编码绝对位置”是 YaRN 论文的推测**（arXiv HTML v3 §3.2，2026-09-24 重读：“we presume having unique position pairs implies that the absolute positional information remains intact”），而 RoPE 打分在结构上只依赖 $t-j$。课文 §3 的类比与 §7 的列表改为明说“这是推测”，并解释“绝对”指训练范围内这把尺子没有重复刻度。
- **(5.11) 的推导只用到二阶矩 $\E[\mathbf q\mathbf q^\top]=I$**，不需要独立或零均值；课文原写“各分量独立、均值 0、方差 1，即 $\E[\mathbf q\mathbf q^\top]=I$”，“即”暗示等价，已改为“例如”。
- **远程衰减上界管的是复数和的模**；打分是其实部，$\abs{\operatorname{Re}z}\le\abs z$。课文在推导后补了这一句。

### 11.10 事实与引用审校（2026-09-24）

逐条列出课文的事实性陈述，与本笔记对照；笔记里没有的，回到一手来源核实（arXiv 用 `arxiv.py grep/text`，HF 与 GitHub 用 `curl` 读 raw 文件，Reddit 帖子读 Wayback 快照）。新核实的事实如下，课文已据此修改。

| # | 事实 | 来源（实际读取的） | 原文短引 / 字段 |
|---|---|---|---|
| S10 | DeepSeek LLM 的发布权重没有偏置：7B 的 `pytorch_model.bin.index.json` 共 273 个张量（30 层 × 9 + 嵌入、最后归一化、`lm_head`），67B 共 858 个（95 × 9 + 3），名字里都没有 `bias`；7B 的每层张量只有 `q/k/v/o_proj.weight`、`gate/up/down_proj.weight` 与两个 `layernorm.weight`。7B 索引 `total_size` 13,820,731,392 字节（BF16 约 6.91B 参数） | `huggingface.co/deepseek-ai/deepseek-llm-{7b,67b}-base/raw/main/pytorch_model.bin.index.json` | 字段原值 |
| S11 | DeepSeek LLM 67B 配置：`LlamaForCausalLM`、95 层、`hidden_size` 8192、`intermediate_size` 22016、64 头、`num_key_value_heads` 8、`rms_norm_eps` 1e-6、`rope_theta` 10000、`max_position_embeddings` 4096、`vocab_size` 102400（7B 同为 102400，30 层、11008、32/32 头） | `huggingface.co/deepseek-ai/deepseek-llm-{7b,67b}-base/raw/main/config.json` | 字段原值 |
| S12 | DeepSeek LLM 论文全文没有写 RMSNorm 的 $\epsilon$、RoPE 底数、11008/22016 或“无偏置”；§2.2 只写 Pre-Norm + RMSNorm、SwiGLU（$\tfrac83d_{model}$）、RoPE、67B 用 GQA | arXiv:2401.02954 全文 grep（`10000`、`11008`、`epsilon`、`bias`） | “The micro design of DeepSeek LLM largely follows the design of LLaMA” |
| S13 | LLaMA v1 代码 `multiple_of` 有注释说明取整目的 | `github.com/meta-llama/llama` 分支 `llama_v1` `llama/model.py` 第 26 行 | “make SwiGLU hidden layer size multiple of large power of 2” |
| S14 | DeepSeek-V3 官方推理代码的 `RMSNorm.forward` 直接 `return F.rms_norm(x, (self.dim,), self.weight, self.eps)`，没有先转 float32；HF `modeling_deepseek.py`（V2、V3）的 RMSNorm 先 `hidden_states.to(torch.float32)` 再 `rsqrt`。LLaMA v1 为 `self._norm(x.float()).type_as(x)`。**修正 §7 E17** 的“实现（LLaMA、DeepSeek）都在 float32 中计算”：只对 LLaMA v1 与 DeepSeek 的 HF 代码成立 | `raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/inference/model.py`；HF `deepseek-ai/DeepSeek-V{2,3}` `modeling_deepseek.py` | 代码原文 |
| S15 | DeepSeek-V3 官方推理代码缓存的是**旋转之后**的键：`k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)` 之后，absorb 模式写 `self.pe_cache[...] = k_pe.squeeze(2)`，naive 模式写入的整段 `k` 也含已旋转的 `k_pe`；YaRN 频率在 `seqlen > args.original_seq_len` 时启用，`softmax_scale` 在 `max_seq_len > original_seq_len` 时乘 `mscale²`（与 §2.7 一致） | 同上 `inference/model.py` 第 360–490 行 | 代码原文 |
| S16 | PaLM 的“无偏置”一条后面紧跟着稳定性的实测说法 | arXiv:2204.02311 v5 §2 | “We found this to result in increased training stability for large models.” |
| S17 | YaRN 的全称带 “method” | arXiv:2309.00071 v3 摘要与 §1 | “YaRN (Yet another RoPE extensioN method)” |
| S18 | bloc97 帖子：正文说 LLaMA 7B 用到 4096 连微调都不需要、困惑度退化很小；标题说 8k+；图是 LLaMA 7b 在 40 条 12k+ 长的提示（gov_report 子集）上的平均困惑度，alpha=8 对 SuperHOT 的 scale=4。细化 §1.6 第 37 条（该条写“即可用到 4096 上下文”，标题的 8k+ 是更强的说法） | Wayback `20230629085816id_` 快照（帖子 + 评论页） | “you don't even need to fine tune the LLaMA 7B model for 4096 context size” |
| S19 | PI 的实验设置：7B/13B/33B/65B “to various context window of sizes up to 32768”；表中出现的长度为 8192、16384、32768 | arXiv:2306.15595 v2 §3.1 | “to various context window of sizes up to 32768” |
| S20 | DeepSeek-V2 §3.1.4 没有单独定义 $t$，只说调整“length scaling factor”以调节注意力熵，并给出 $\sqrt t=0.0707\ln s+1$ | arXiv:2405.04434 §3.1.4 | “adjust the length scaling factor to modulate the attention entropy” |
| S21 | RoFormer §3.3 对远程衰减的直觉说明：距离远的 token 对应当联系更弱 | arXiv:2104.09864 v5 §3.3 | “a pair of tokens with a long relative distance should have less connection” |

课文据此做的修改：封面导语“GPT-2 之后几乎所有开源大模型”改为“从 LLaMA（2023）起，许多开源大模型”（GPT-NeoX、BLOOM、OPT 等 2022 年前后的模型并没有同时换四块积木，Qwen 系列保留了 QKV 偏置——这一句是笔者的常识性判断，没有逐个核对配置，所以课文改成了不作全称断言的写法）；§1 表格中 DeepSeek LLM 的偏置一格由“沿用 LLaMA 结构”改为“无（发布的权重里没有偏置张量）”，并新增 `ds-llm-hf` 卡片（S10、S11），§1 表格来源、图 5.9 说明与 §8“前四样全部就位”一句改引该卡片；§4 的“都先转成 float32”改为只对 LLaMA 与 DeepSeek 的 HF 代码成立（S14）；§4“说明在它们的规模上没有可见损失”改标为笔者推断；§5“代码里没有注释说明理由”改为引用 `multiple_of` 注释（S13）；§6 PI 例子不再说“主实验是 $s=16$”（S19）；§6 bloc97 的转述按帖子原文改（S18）；§7 YaRN 全称补 “method”（S17）；§1 痛点二注明 1258 亿 token 是按论文步数与 batch 算出的；`peng2023yarn` 卡片的“约 400 步扩到 64K–128K”改为“400 步到 64K、再 200 步到 128K”（§1.7 第 48 条）；图 5.1 说明补引 `llama-v1-code`（11008 与“无偏置”只见于代码）。
