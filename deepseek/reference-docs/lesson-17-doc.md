# 第 17 课研究笔记：记忆的另外两种形态——Engram 条件记忆与 DeepSeek-OCR 视觉压缩

> 本文件是 `deepseek/src/lesson-17-engram-ocr.md` 的研究底稿，只收“核对过的引文、复算过的数字、勘误、未能核实清单”，
> 不是讲义正文。建立于 2026-09-24。所有引文保留英文原文，并注明节号 / 表号 / 式号 / 所读版本。
> 所有派生数字都在 Python 里复算过（第 4 节给出表达式）。凡是笔者自己的推理，一律标注“笔者推测”或“笔者推导”；
> 从论文图上目测读出的数值一律标注“目测”，只能用于示意图，不能当作论文数字引用。
> 与第 16 课（mHC）、第 19 课（V4.1-Flash）笔记有交叉：Engram 在 V4.1 中的集成方式记在本文件 §1.3，
> V4.1 的其余架构留给第 19 课笔记。

---

## 0. 覆盖范围与所读版本

### 0.1 论文

| 短键 | 标题 | 作者 | arXiv / 版本 | 发表状态 | 日期 | 本课用途 |
|---|---|---|---|---|---|---|
| `engram` | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models | Xin Cheng, Rui Tian（v2 起共同一作）, Wangding Zeng, Damai Dai, … , Huishuai Zhang, Dongyan Zhao, Wenfeng Liang（v2 共 21 人；v1 共 14 人）；单位 Peking University + DeepSeek-AI | 2601.07372，**读 v2**，并与 v1 全文 diff | **ACL 2026 长文**（见 `engram-acl`）；arXiv 仍为预印本 | v1 2026-01-12；v2 2026-07-12（HTML 页脚 “12 Jul 2026”） | 全部 Engram 公式、表 1–6、图 1–7、稀疏分配律、系统实验 |
| `engram-acl` | 同名（正文把模块改称 **Deep Sparse Embedding (DSE)**） | 14 人（与 arXiv v1 同一名单；Dongyan Zhao 另挂 “National Engineering Research Center of New Electronic Publishing Technologies”） | —（ACL Anthology 2026.acl-long.226，DOI 10.18653/v1/2026.acl-long.226） | Proceedings of the 64th Annual Meeting of the ACL (Volume 1: Long Papers), pp. 4968–4990 | 2026-07（会议 July 2–7, 2026, San Diego） | 版本漂移核对：数字与 arXiv 一致，但命名、措辞、结构不同（§7 E1） |
| `dsv41` | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | DeepSeek-AI | 2609.19969v1 | 预印本（技术报告） | 2026-09-17 | **V4.1 使用 Engram**：§2.1、§2.4.2、§2.5、§3.1.3、§4.2.1–4.2.2 |
| `dsv4` | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | DeepSeek-AI | 2606.19348v1 | 预印本 | HTML 页脚 “26 Apr 2026” | **V4 未使用 Engram**，只在 §6 未来工作里提到 |
| `ocr` | DeepSeek-OCR: Contexts Optical Compression | Haoran Wei, Yaofeng Sun, Yukun Li（DeepSeek-AI） | 2510.18234v1（唯一版本） | 预印本（技术报告） | 2025-10-21 | DeepEncoder、表 1–4、图 1/3/13、遗忘机制 |
| `ocr2` | DeepSeek-OCR 2: Visual Causal Flow | Haoran Wei, Yaofeng Sun, Yukun Li | 2601.20552v1（唯一版本） | 预印本 | 2026-01-28 | DeepEncoder V2、式 (1)(2)、表 1–4 |
| `ocr-crutch` | Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR | Yunhao Liang 等 10 人 | 2601.03714v2 | 预印本（第三方） | v1 2026-01-07 | 对“光学压缩”结论的反驳性检验（只读了摘要） |
| `engram-nine` | A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics | Tao Lin | 2601.16531v2 | 预印本（第三方） | v1 2026-01-23 | “哈希碰撞是不是瓶颈”的第三方实验（只读了摘要） |
| `dsvl` | DeepSeek-VL: Towards Real-World Vision-Language Understanding | Haoyu Lu 等 | 2403.05525v2 | 预印本 | 2024-03-08 | 多模态谱系：SAM-B + SigLIP-L 混合编码器，两层 stride-2 卷积（DeepEncoder 压缩器的祖先） |
| `dsvl2` | DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding | Zhiyu Wu 等 27 人 | 2412.10302v1 | 预印本 | 2024-12-13 | 动态切块；DeepSeekMoE 3B/16B/27B（0.57B/2.4B/4.1B 激活）——OCR 解码器与 Engram 基线的来源 |
| `janus` | Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation | Chengyue Wu 等 11 人 | 2410.13848v1 | 预印本 | 2024-10-17 | 谱系一行 |
| `januspro` | Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling | Xiaokang Chen 等 8 人 | 2501.17811v1 | 预印本 | 2025-01-29 | 谱系一行 |

### 0.2 制品（无论文，`kind: artefact`）

| 短键 | 内容 | 读取方式 / 时间 |
|---|---|---|
| `gh-engram` | `github.com/deepseek-ai/Engram`：README、`engram_demo_v1.py`（演示版实现）、`Engram_paper.pdf`（v1 内容）、`figures/*.png` | 2026-09-24 raw 下载；首次提交 `init` 2026-01-12T09:11Z，最后提交 2026-01-14 |
| `hf-v41-config` | `huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json` 与 `inference/config.json` | 2026-09-24 原样下载 |
| `hf-v41-code` | 同仓库 `inference/engram.py`（哈希与分词器压缩）、`inference/model.py`（`Engram` 模块、`ParallelEngramEmbedding`） | 2026-09-24 原样下载，逐行读 |
| `hf-v41-card` | 同仓库 `README.md`（模型卡） | 2026-09-24 |
| `hf-ocr-config` / `hf-ocr-code` | `huggingface.co/deepseek-ai/DeepSeek-OCR` 的 `config.json`、`modeling_deepseekocr.py` | 2026-09-24 原样下载 |
| `gh-ocr` / `gh-ocr2` | `github.com/deepseek-ai/DeepSeek-OCR`、`DeepSeek-OCR-2` README | 2026-09-24；OCR 首次提交 2025-10-20，OCR-2 首次提交 2026-01-27 |
| `hf-vl2-config` | `huggingface.co/deepseek-ai/deepseek-vl2/raw/main/config.json`（27B LLM 的专家维度） | 2026-09-24 |

### 0.3 Engram 论文的版本史（全文 diff 结果）

| 版本 | 日期 | 与前一版的差异（本课相关的用粗体） |
|---|---|---|
| arXiv v1 | 2026-01-12 | 初版，14 位作者，无共同一作标记。代码同日开源 |
| arXiv v2 | 2026-07-12 | **作者从 14 人增至 21 人**（新增 Rui Tian 为共同一作，另加 Chengqi Deng, Shangyan Zhou, Chenggang Zhao, Zhengyan Zhang, Yixuan Wei, M.Y Xu）；§2.3 末句从 “its specific placement is governed by the system-level latency constraints” 改为 “**its specific placement is optimized to balance modeling effectiveness against the system-level latency constraints**”；§2.5 首段改写；§7 新增 N-Grammer、STEM、L3 和 “High-Cardinality Categorical Embeddings”（推荐系统哈希嵌入）一段，Memory Network 段加了 CoG；引用格式微调（如 Muon 的引用由 Team et al. 2025 改为 Liu et al. 2025b）。**所有数字、表、式与 v1 逐字相同** |
| ACL 2026 | 2026-07 | **模块改名为 Deep Sparse Embedding (DSE)**，正文中 “Engram” 出现 0 次、“DSE” 出现在 155 行里（按整词计 159 次；pypdf 抽取，计数随抽取方式浮动，核查记录 V2）；篇幅压缩，结构重排（附录 A–J）；**删去了 §6.2 的逐层插入扫描与组件消融（arXiv 图 5）**，也删去了“ρ≈40% 仍与纯 MoE 持平”一句；无限内存曲线的措辞从 “strict power law (linear in log-space)” 改为 “roughly log-linearly”；表 1 的 Dense-4B 总参数一格印成 3.8B（arXiv 与 ACL 附录表 3 都是 4.1B）；吞吐开销补写了 4B 的 1.9%；分词器压缩表（ACL 表 4）的标题写 “Overall compression ratio: 76.57%”——名称仍是 compression ratio，数值却是 arXiv 表 6 “23.43%” 的补数（即保留比例）；正文 §2.2 两版都说有效词表减少 23%（arXiv “a 23% reduction”，ACL “reduces … by 23%”） |

- **写课时的引用约定**：机制和编号以 arXiv v2 为准（它有 §6.2 的消融和图 5）；需要“同行评审出处”时引 ACL 版本，
  并在正文注明“ACL 版称 DSE”。V4.1 技术报告仍称 Engram，并同时引用了 arXiv 版（Cheng et al., 2026b）和 ACL 版（2026c）。

### 0.4 “V4 / V4.1 用没用 Engram”——精确答案

- **DeepSeek-V4（2606.19348v1）：没有用。** 全文 grep “Engram / engram / conditional memory” 只命中参考文献条目；
  唯一的引用位置在 §6 “Conclusion, Limitations, and Future Directions”，把它列为未来方向（引文见 C-V4）。
- **DeepSeek-V4.1-Flash（2609.19969v1）：用了。** 196B Engram 参数，两个模块，放在第 1、14 层（从 0 计数），
  N-gram 阶 {2,3,4}，每阶 8 个哈希头，去掉了短卷积，嵌入表用 Sinkhorn 平衡的动量更新训练（引文见 C-V41-1 至 C-V41-6）。
  `config.json` 与 `inference/engram.py` 和论文一致，并补充了论文没写的细节（§3.3–§3.4、§4.3）。
- **V4.1 没有用 DeepSeek-OCR 的 DeepEncoder。** V4.1 的视觉编码器是从头训练的 DeepSeek-ViT（patch 14，3×3 pixel-unshuffle），
  全文不引用 DeepSeek-OCR（grep “OCR” 只命中训练数据描述）。

---

## 1. 核心论断与原文引文

### 1.1 Engram（`engram` arXiv v2，除非另注）

**E-C1. 问题：Transformer 没有“查表”原语，只能用计算模拟检索。**（§1）
> "Since standard Transformers (Vaswani et al., 2017) lack a native knowledge lookup primitive, current LLMs are forced to simulate retrieval through computation. For instance, resolving a common multi-token entity requires consuming multiple early layers of attention and feed-forward networks (Ghandeharioun et al., 2024; Jin et al., 2025) (see Table 3). This process essentially amounts to an expensive runtime reconstruction of a static lookup table, wasting valuable sequential depth on trivial operations that could otherwise be allocated to higher-level reasoning."

**E-C2. 条件记忆 vs 条件计算。**（§1）
> "Whereas conditional computation sparsely activates parameters to process dynamic logic (Shazeer et al., 2017; Bengio et al., 2013), conditional memory relies on sparse lookup operations to retrieve static embeddings for fixed knowledge."

**E-C3. 四个“现代化改造”。**（§1）
> "we propose Engram, a conditional memory module grounded in the classic N-gram structure but equipped with modern adaptations such as tokenizer compression, multi-head hashing, contextualized gating, and multi-branch integration"

**E-C4. 分词器压缩：满射 $\mathcal P:V\to V'$，词表有效大小减 23%。**（§2.2）
> "we pre-compute a surjective function $\mathcal{P}:V\to V^{\prime}$ that collapses raw token IDs into canonical identifiers based on normalized textual equivalence (using NFKC (Whistler, 2025), lowercasing, etc.). In practice, this process achieves a 23% reduction in the effective vocabulary size for a 128k tokenizer (see Appendix C)."

附录 C 表 6 标题：
> "The table illustrates Top-5 merged tokens by Tokenizer Compression and the overall compression ratio is 23.43% for our 128k tokenizer."

**E-C5. 多头哈希：每个 N-gram 阶用 $K$ 个哈希头、素数大小的表、乘法-异或哈希。**（§2.2）
> "To mitigate collisions, we employ $K$ distinct hash heads for each $N$-gram order $n$. Each head $k$ maps the compressed context to an index within an embedding table $\mathbf{E}_{n,k}$ (of prime size $M_{n,k}$) via a deterministic function $\varphi_{n,k}$"

> "In practice, $\varphi_{n,k}$ is implemented as a lightweight multiplicative-XOR hash."

**E-C6. 上下文感知门控：隐藏状态当 Query，检索到的记忆当 Key/Value。**（§2.3）
> "we utilize the current hidden state $\mathbf{h}_{t}$—which has aggregated global context via preceding attention layers—as a dynamic Query, while the retrieved memory $\mathbf{e}_{t}$ serves as the source for both Key and Value projections"

> "This design enforces semantic alignment: if the retrieved memory $\mathbf{e}_{t}$ contradicts the current context $\mathbf{h}_{t}$, the gate $\alpha_{t}$ tends toward zero, effectively suppressing the noise."

**E-C7. 短因果卷积：核宽 4、膨胀 = 最大 N-gram 阶、SiLU。**（§2.3）
> "Using a kernel size $w$ (set to 4), dilation $\delta$ (set to the max $N$-gram order) and SiLU activation (Elfwing et al., 2018), the final output $\mathbf{Y}$ is computed as:"

**E-C8. 只放在少数几层，残差相加。**（§2.3 末，v2 措辞）
> "The Engram module is integrated into the backbone via a residual connection: $\mathbf{H}^{(\ell)}\leftarrow\mathbf{H}^{(\ell)}+\mathbf{Y}$, followed by the standard Attention and MoE. Crucially, Engram is not applied to every layer; its specific placement is optimized to balance modeling effectiveness against the system-level latency constraints detailed in Section 2.5."

**E-C9. 与多分支（mHC）的集成：一张表、一个 $W_V$ 共享，$M$ 个分支各有 $W_K^{(m)}$；可融合成一个 FP8 GEMM。**（§2.4）
> "a single sparse embedding table and a Value projection matrix $\mathbf{W}_{V}$ are shared across all $M$ branches, whereas $M$ distinct Key projection matrices $\{\mathbf{W}_{K}^{(m)}\}_{m=1}^{M}$ are employed to enable branch-specific gating behaviors."

> "This design allows the linear projections (one $\mathbf{W}_{V}$ and $M$ distinct $\mathbf{W}_{K}^{(m)}$) to be fused into a single dense FP8 matrix multiplication, maximizing the compute utilization of modern GPUs. Unless otherwise stated, all experiments utilize this integration with Manifold-Constrained Hyper-Connections ($M=4$) (Xie et al., 2025)."

**E-C10. 确定性寻址 → 训练时分片 + All-to-All；推理时从主机内存异步预取。**（§2.5）
> "During training, to accommodate large-scale embedding tables, we employ standard model parallelism by sharding the tables across available GPUs. An All-to-All communication primitive is used to gather active rows in the forward pass and dispatch gradients in the backward pass"

> "Since memory indices are known prior to the forward pass, the system can asynchronously retrieve embeddings from abundant host memory via PCIe. To effectively mask communication latency, the Engram module is placed at specific layers within the backbone, leveraging the computation of preceding layers as a buffer to prevent GPU stalls."

**E-C11. 放置位置是“软硬件协同”的折中。**（§2.5）
> "while placing Engram deeper extends the compute window available for hiding latency, our ablation in Section 6.2 shows that modeling performance favors early intervention to offload local pattern reconstruction. Therefore, the optimal placement must simultaneously satisfy both modeling and system latency constraints."

**E-C12. 多级缓存（Zipf）是设想，不是实测。**（§2.5 与 §6.4）
> "This statistical property motivates a Multi-Level Cache Hierarchy: frequently accessed embeddings can be cached in faster storage tiers (e.g., GPU HBM or Host DRAM), while the long tail of rare patterns resides in slower, high-capacity media (e.g., NVMe SSD)."

> "While the hierarchical design in Section 2.5 exploits Zipfian locality to cache frequent items in HBM, our experimental setup forces all retrievals to traverse the PCIe bus from host memory."

**E-C13. 稀疏分配问题的定义：固定总参数与激活参数，只在“不激活的参数预算”里分配。**（§3.1）
> "$P_{\mathrm{sparse}}\triangleq P_{\mathrm{tot}}-P_{\mathrm{act}}$: the inactive parameters, which represents the “free” parameter budget available for scaling model size without incurring computational cost (e.g., unselected experts or unretrieved embeddings)."

> "We define the allocation ratio $\rho\in[0,1]$ as the fraction of the inactive-parameter budget assigned to MoE expert capacity"

**E-C14. 实验规模与稀疏度。**（§3.1 Experimental protocol）
> "We evaluate this trade-off at two compute regimes and maintain a constant sparsity ratio $P_{\mathrm{tot}}/P_{\mathrm{act}}\approx 10$ across both settings: $C=2\times 10^{20}$ FLOPs: $P_{\mathrm{tot}}\approx 5.7$B and $P_{\mathrm{act}}=568$M. The baseline ($\rho=1$) has a total of 106 experts. $C=6\times 10^{20}$ FLOPs: $P_{\mathrm{tot}}\approx 9.9$B and $P_{\mathrm{act}}=993$M. The baseline ($\rho=1$) has a total of 99 experts."

**E-C15. U 形曲线；最优约在 ρ≈75%–80%，即把 20%–25% 的稀疏预算给 Engram。**（§3.1 Results）
> "Figure 3 (left) reveals a consistent U-shaped relationship between validation loss and the allocation ratio $\rho$. Remarkably, the Engram model achieves comparable performance to the pure MoE baseline ($\rho=100\%$) even when the MoE allocation is reduced to just $\rho\approx 40\%$ (i.e., a total of 46 experts for the 5.7B model and 43 experts for the 9.9B model). Furthermore, the pure MoE baseline proves suboptimal: reallocating roughly 20%–25% of the sparse parameter budget to Engram yields the best performance. Quantitatively, in the 10B regime ($C=6\times 10^{20}$), validation loss improves from 1.7248 (at $\rho=100\%$) to 1.7109 near the optimum of $\rho\approx 80\%$ ($\Delta=0.0139$). Crucially, the location of this optimum is stable across regimes ($\rho\approx 75\%$–$80\%$), suggesting a robust allocation preference across the examined scales (under fixed sparsity)."

  注意限定语 “under fixed sparsity” 和 “across the examined scales”：只在两个算力点、固定 $P_{tot}/P_{act}\approx10$ 下测过。

**E-C16. U 形两端的解释（作者的解释，不是证明）。**（§3.1）
> "MoE-dominated ($\rho\to 100\%$): The model lacks dedicated memory for static patterns, forcing it to inefficiently reconstruct them through depth and computation."

> "Engram-dominated ($\rho\to 0\%$): The model loses conditional computation capacity, hurting tasks that require dynamic, context-dependent reasoning; memory cannot replace computation in this regime."

**E-C17. 无限内存区：槽数 $2.58\times10^5$ → $1.0\times10^7$，验证损失对 log(槽数) 近似线性。**（§3.2）
> "We utilize a fixed MoE backbone with $P_{\mathrm{tot}}\approx 3$B and $P_{\mathrm{act}}=568$M, trained for 100B tokens to ensure convergence. On top of this backbone, we attach an Engram table and sweep the number of slots $M$ from $2.58\times 10^{5}$ to $1.0\times 10^{7}$ (adding up to $\approx 13$ billion parameters)."

> "Across the explored range, the curve follows a strict power law (linear in log-space), indicating that Engram provides a predictable scaling knob: larger memory continues to pay off without requiring additional computation."

  图 3 标题的说法是 “Validation loss exhibits a log-linear trend with respect to the number of embeddings”；ACL 版改为
  “validation loss decreases roughly log-linearly as memory grows”。见勘误 E4。

**E-C18. 27B 规模主实验：四个模型、262B tokens、激活参数都是 3.8B。**（表 1 标题）
> "All models are trained for 262B tokens and are matched in activated parameters (3.8B). Engram-27B is iso-parameters with MoE-27B by reallocating parameters from routed experts (72 → 55) to a 5.7B-parameter Engram memory. Engram-40B further increases Engram memory (18.5B parameters) while keeping the activated-parameter budget fixed."

**E-C19. Engram-27B 的具体配置。**（§4.1）
> "We reduce the number of routed experts from 72 to 55 and reallocate the freed parameters to a 5.7B-parameter embedding module ($\rho=74.3\%$), keeping the total model size constant at 26.7B. Regarding the Engram configuration, we instantiate the module at layers 2 and 15 and set the maximum $N$-gram size to 3, the number of heads to 8, and the dimension to 1280. For optimization, the embedding parameters are updated using Adam (Kingma, 2014) with a learning rate scaled by $5\times$ and no weight decay, while the convolution parameters are initialized to zero to strictly preserve the identity mapping at the start of training."

**E-C20. 收益不止在知识类，推理/代码/数学更大。**（§4.2，正文数字）
> "Interestingly, these gains are not limited to knowledge-intensive tasks (e.g., MMLU: +3.0, MMLU-Pro: +1.8, CMMLU: +4.0), where memory capacity is intuitively beneficial. We observe even more significant improvements in general-reasoning domains (e.g., BBH: +5.0, ARC-Challenge: +3.7, DROP: +3.3), as well as code and mathematical reasoning (e.g., HumanEval: +3.0, MBPP: +1.6, GSM8K: +2.2, MATH: +2.4)."

  **摘要与 §1 写的是 “MMLU +3.4”，§4.2 写 “MMLU: +3.0”**，表 1 支持 +3.0（见勘误 E2）。

**E-C21. Engram-40B 没有全面胜过 Engram-27B，作者归因于欠训练。**（§4.2）
> "Although it does not yet strictly dominate Engram-27B on every task, this is likely an artifact of under-training. We observe that the training loss gap between Engram-40B and the baselines continues to widen towards the end of training, suggesting that the expanded memory capacity has not yet fully saturated within the current token budget."

**E-C22. 长上下文：等损失（46k 步）对比下 Multi-Query NIAH 97.0 vs 84.2。**（§5.2）
> "When comparing Engram-27B (46k) against the fully trained MoE-27B (50k)—models aligned on pre-training loss—Engram demonstrates significant gains. Specifically, it outperforms the baseline on complex retrieval tasks (e.g., Multi-Query NIAH: 97.0 vs. 84.2; VT: 87.2 vs. 77.0)."

  注意：**摘要里的 “Multi-Query NIAH: 84.2 → 97.0” 是等损失（46k）或等 FLOPs（50k）两种比较共用的数字**（46k 与 50k 的 MQ 都是 97.0），
  而 §1 的 “Variable Tracking: 89.0 vs. 77.0” 是等 FLOPs（50k）那一行；§5.2 的 “VT: 87.2 vs. 77.0” 是等损失那一行。
  同一个“VT 提升”在论文里有两个数，引用时要写明是哪一行。

**E-C23. 长上下文能力与基座能力耦合——比较架构要对齐基座损失，而不是对齐步数。**（§5.2）
> "This suggests that long-context performance is intrinsically coupled with the general modeling ability of the base model. Consequently, a rigorous architectural comparison must control for this confounding variable by aligning base model loss, rather than merely aligning training steps."

**E-C24. “Engram 等价于加深网络”：LogitLens 与 CKA。**（§6.1.2）
> "For instance, the representations formed at layer 5 of Engram-27B align most closely with those at approximately layer 12 of the MoE baseline."

> "This validates our central hypothesis: by bypassing early-stage feature composition via explicit lookups, Engram is functionally equivalent to increasing the model's effective depth."

  “functionally equivalent” 是作者对表征相似度的解读（实测是 CKA 相似度矩阵的对角线上移），不是定理。

**E-C25. 放在哪一层：单模块时第 2 层最好；同样 1.6B 拆成第 2、6 层两块更好。**（§6.2，只在 arXiv 版）
> "Unless otherwise specified, the backbone is a 12-layer 3B MoE model (0.56B activated parameters) trained for 100B tokens. Figure 5 reports validation loss. The dashed orange line denotes the 3B MoE baseline (Val Loss $=1.808$)."

> "Our reference model uses $\{2,3\}$-grams and inserts Engram at Layers 2 and 6, achieving Val Loss $=1.768$, a substantial improvement over the MoE baseline ($\Delta=0.04$)."

> "The sweep shows that Layer 2 achieves the best single-layer performance (Val Loss $=1.770$), outperforming Layer 1 and degrading as the insertion point moves deeper. This indicates that one round of attention is already sufficient to provide a meaningfully contextualized $\mathbf{h}_{t}$ for gating, while still being early enough to replace the backbone's bottom-layer local aggregation."

**E-C26. 哪些组件重要：多分支融合、上下文门控、分词器压缩最重要；去掉卷积只略差；加 4-gram 略差。**（§6.2）
> "We find that three components yield the most significant gains: (i) branch-specific fusion within the multi-branch backbone, (ii) context-aware gating, and (iii) tokenizer compression."

> "Other changes have smaller effects: removing the lightweight depthwise convolution only marginally degrades performance. Allocating capacity to 4-grams is slightly suboptimal under a fixed 1.6B budget—likely because it dilutes capacity from the more frequent 2/3-gram patterns—though we do not rule out that higher-order $N$-grams become beneficial at larger memory scales."

  这句话后来被 V4.1 “兑现”：V4.1 用了 {2,3,4} 阶并去掉了卷积（C-V41-3）。

**E-C27. 推理时把 Engram 输出整个关掉：事实类只保留 29–44%，阅读理解保留 81–93%。**（§6.3）
> "Factual knowledge benchmarks suffer a catastrophic collapse, retaining only 29–44% of the original performance (e.g., TriviaQA at 29%), confirming that the Engram module acts as the primary repository for parametric knowledge. Conversely, reading comprehension tasks are remarkably resilient, retaining 81–93% (e.g., C3 at 93%)"

  作者自己的限定：
> "Crucially, this post-hoc ablation induces a training–inference inconsistency, potentially introducing noise in complex, mixed-capability tasks."

**E-C28. 100B 参数的 Engram 表整个放在主机内存，吞吐只降 ≤2.8%。**（§6.4、表 4）
> "We insert a massive 100B-parameter Engram layer into the second Transformer block, with the entire embedding table resident in host DRAM. During inference, the system prefetches embeddings for the Engram layer asynchronously, overlapping the PCIe transfer with the computation of the first block."

> "As detailed in Table 4, offloading a 100B-parameter embedding table incurs a negligible throughput penalty, peaking at only 2.8% on the 8B backbone."

> "Crucially, the effective communication volume per step scales with the number of activated slots rather than the total embedding table size."

  实验条件：nano-vLLM 改的推理框架，**稠密**骨干（Dense-4B、Dense-8B，不是 MoE），NVIDIA H800，512 条序列，长度 Uniform(100,1024)。

**E-C29. 门控可视化：门控在“局部固定搭配的最后一个 token”上打开。**（§6.5）
> "In English, we observe strong activations on multi-token named entities (e.g., “Alexander the Great”, “the Milky Way”) and formulaic phrases (e.g., “By the way”, “Princess of Wales”). This behavior generalizes effectively across languages. In the Chinese examples, Engram identifies and retrieves distinct idiomatic expressions and historical entities, such as “Four Great Inventions” (四大发明) and “Zhang Zhongjing” (张仲景)."

  脚注 2：27B 模型有 mHC $M=4$、两层 Engram，所以每个 token 有 8 个门控标量，作者只挑“与语义模式匹配最相关”的那个画图：
> "We observe that not every branch encodes interpretable activation patterns. For the clarity of this visualization, we select and display the gating values most strongly correlated with semantic pattern matching."

**E-C30. 与已有工作的区别（作者自述）：严格等参数、等 FLOPs 比较；嵌入放在深层以便通信与计算重叠。**（§7）
> "Existing approaches place embeddings strictly at the input layer (Layer 0), which inherently serializes memory access and computation (Yu et al., 2025; Huang et al., 2025a). Engram, conversely, strategically injects memory into deeper layers to enable communication-computation overlap."

### 1.2 Engram 的开源演示代码（`gh-engram`，`engram_demo_v1.py`）

**E-K1. 代码自称演示版。**（README §5）
> "The provided code is a demonstration version intended to illustrate the data flow. It mocks standard components (like Attention/MoE/mHC) to focus on the Engram module."

**E-K2. 门控在 sigmoid 之前多了一步“带符号平方根”，论文式 (4) 没有。**（`Engram.forward`）
```python
gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
gate = gate.sigmoid().unsqueeze(-1)
```

**E-K3. 多头哈希的实现：同一个“混合整数”对不同素数取模；各头之间只差模数。**（`NgramHashMapping._get_ngram_hashes`）
```python
mix = (tokens[0] * multipliers[0])
for k in range(1, n):
    mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
...
for j in range(num_heads_for_this_ngram):
    mod = int(head_vocab_sizes[j])
    head_hash = mix % mod
```
  乘子是每个 Engram 层各一组、每个回看位置（0…N−1）一个的随机奇数（`r * 2 + 1`，种子 `seed + 10007 * layer_id`），上界 `(int64_max // 压缩词表大小) // 2` 保证 `token_id * multiplier` 不溢出 int64；表大小是从 `engram_vocab_size - 1`
  往上找的**全模型互不相同的素数**（`find_next_prime(..., seen_primes)`）。

**E-K4. 分词器压缩的规范化链比论文说的多：NFKC → NFD → 去重音 → 小写 → 空白折叠。**（`CompressedTokenizer.__init__`）
```python
normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), ...
```
  论文只写 “NFKC (Whistler, 2025), lowercasing, etc.”；附录表 6 把 'á'、'ä' 并到 'a'，与代码的去重音一致。

### 1.3 DeepSeek-V4 / V4.1 对 Engram 的使用

> 许可说明：`dsv4`、`dsv41` 使用 arXiv 非独占许可（不是 CC BY），这里只摘不超过 15 词的短语，其余用中文转述并给出节号；
> Engram、DeepSeek-OCR、OCR 2、VL2、Janus、Janus-Pro 的 arXiv 版与 ACL Anthology 版均为 CC BY 4.0，可带出处整句引用。

**C-V4.（`dsv4` §6 “Conclusion, Limitations, and Future Directions”，唯一提及处）**
转述：在 MoE 与稀疏注意力之外，将沿新的维度探索模型稀疏性，例子是 “more sparse embedding modules (Cheng et al., 2026)”，目标是提升计算与内存效率。
——V4 的架构章节里没有 Engram。

**C-V41-1. V4.1 集成了 Engram。**（`dsv41` §1）
> "we integrate the Engram (Cheng et al., 2026b) conditional memory module to strengthen model capabilities"

**C-V41-2. 552B 骨干参数之外另有 196B Engram 参数。**（`dsv41` §2.1）
> "552B backbone parameters and 196B Engram parameters"

  同句还说 prefill 每 token 激活 8B、decode 激活 16B。
  模型卡（`hf-v41-card`）同样写：“Engram conditional memory (196B parameters, sparsely accessed via token-based lookup)”。
  摘要只说 “552B backbone parameters”，不提 196B；模型卡规格表的 “# Backbone Params” 一行也是 552B。

**C-V41-3. 沿用原设计，两处修改：去掉短卷积；嵌入表改用动量 + Sinkhorn 平衡更新。**（`dsv41` §2.4.2）
转述：沿用原 Engram 的分词器压缩、多头哈希、上下文门控与多分支集成；改动一是去掉短因果卷积，理由是
“its performance gains do not justify the added complexity in our inference stack”；改动二是嵌入表用动量更新加 Sinkhorn 平衡（详见其 §2.5）。

**C-V41-4. 具体配置。**（`dsv41` §2.4.2，转述 + 短引）
- 196B Engram 参数平均分给两个模块；每个模块用 N-gram 阶 {2,3,4}、8 个哈希头、每阶总嵌入维 2048。
- 每个头的表 “approximately 16M entries, with table sizes chosen to be distinct primes”。
- 嵌入表与 K/V 投影都用 FP8。
- 放在第 1、14 层（从 0 计），理由是 “to balance memory usage across training pipeline stages”。
- 推理时靠确定性寻址从主机内存预取，方式是 “background RDMA transfers”，第一个模块的预取与第 1 个 Transformer 块的计算重叠。

  注意放置理由变了：Engram 论文讲“建模效果 vs 延迟隐藏”，V4.1 讲训练流水段之间的显存平衡。

**C-V41-5. 为什么不用 Adam：优化器状态太大。**（`dsv41` §2.5）
> "applying Adam to the newly introduced Engram parameters substantially increases the optimizer-state memory footprint"

  转述：因此 Engram 嵌入表、token 嵌入和预测头都改用“动量更新 + Sinkhorn 平衡”；它像 Muon 一样
  “requires only a momentum buffer while empirically outperforming Adam”。

**C-V41-6. 训练与推理系统。**（`dsv41` §3.1.3，转述 + 短引）
- 训练：表 “partitioned by row across dedicated process groups of engram parallel size”；组大小权衡每卡显存与查表通信范围；
  优化器状态再在各副本间切分；在每个流水段开始处理本训练步的微批之前，为整个本地 batch 发起预取（原文 “before each pipeline stage begins processing microbatches for the current training step”）；嵌入梯度在骨干反传后送回所属 rank；预取与梯度传输和视觉编码器的前/反向重叠。
- 精度：“Embeddings are stored and fetched in FP8”，取回的数值与缩放因子直接送入后续 GEMM。
- RL：“During RL rollouts, Engram embedding tables remain resident in GPU memory.” 理由是减轻主机内存压力、避免主机内存碎片导致的 OOM。

**C-V41-7. 超参数。**（`dsv41` §4.2.2）
Sinkhorn 平衡更新与 Muon 用相同的动量系数与学习率修正因子，并设 $K=11$、$\tau=10^{-3}$、$\varepsilon=10^{-20}$；
“the learning rate of Engram is scaled by $5\times$”（沿用 Engram 论文）。

**C-V41-8. V4.1 代码里的门控同样有带符号平方根，并注明与训练内核一致；图像 token 不参与 n-gram。**（`hf-v41-code`，`model.py` `Engram.forward` 与 `Transformer.forward`）
```python
# signed sqrt before the sigmoid, matching the training kernel
gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
...
# image tokens take no part in an n-gram and get no engram contribution; text-only needs no mask
```
  V4.1 代码中 `Engram.forward` 返回 `h + gate * value`，没有卷积，与 C-V41-3 一致。

### 1.4 DeepSeek-OCR（`ocr` v1）

**O-C1. 定位：一次“能否用光学二维映射压缩长上下文”的初步探索。**（摘要）
> "We present DeepSeek-OCR as an initial investigation into the feasibility of compressing long contexts via optical 2D mapping. DeepSeek-OCR consists of two components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder."

**O-C2. 头条数字：<10× 时约 97%，20× 时约 60%。**（摘要）
> "Experiments show that when the number of text tokens is within 10 times that of vision tokens (i.e., a compression ratio < 10×), the model can achieve decoding (OCR) precision of 97%. Even at a compression ratio of 20×, the OCR accuracy still remains at about 60%. This shows considerable promise for research areas such as historical long-context compression and memory forgetting mechanisms in LLMs."

**O-C3. 更细的三段说法（§1 贡献一）。**
> "Our method achieves 96%+ OCR decoding precision at 9-10× text compression, ∼90% at 10-12× compression, and ∼60% at 20× compression on Fox [21] benchmarks featuring diverse document layouts (with actual accuracy being even higher when accounting for formatting differences between output and ground truth), as shown in Figure 1(a)."

**O-C4. 压缩比的定义。**（图 1 标题）
> "Figure (a) shows the compression ratio (number of text tokens in ground truth/number of vision tokens model used) testing on Fox [21] benchmark"

**O-C5. DeepEncoder：SAM-base（窗口注意力，80M）→ 16× 卷积压缩器 → CLIP-large（全局注意力，300M）。**（§3.1、§3.2.1）
> "DeepEncoder is approximately 380M in parameters, mainly composed of an 80M SAM-base [17] and a 300M CLIP-large [29] connected in series. The decoder adopts a 3B MoE [19, 20] architecture with 570M activated parameters."

> "Between the two components, we borrow from Vary [36] and use a 2-layer convolutional module to perform 16× downsampling of vision tokens. Each convolutional layer has a kernel size of 3, stride of 2, padding of 1, and channels increase from 256 to 1024. Assuming we input a 1024×1024 image, the DeepEncoder will segment it into 1024/16×1024/16=4096 patch tokens. Since the first half of encoder is dominated by window attention and only 80M, the activation is acceptable. Before entering global attention, the 4096 tokens go through the compression module and the token count becomes 4096/16=256, thus making the overall activation memory controllable."

> "For CLIP, we remove the first patch embedding layer since its input is no longer images but output tokens from the previous pipeline."

**O-C6. 多分辨率模式。**（§3.2.2）
> "Native resolution supports four sub-modes: Tiny, Small, Base, and Large, with corresponding resolutions and token counts of 512×512 (64), 640×640 (100), 1024×1024 (256), and 1280×1280 (400) respectively."

> "The vision token number output by the DeepEncoder under Gundam mode is: $n\times 100+256$, where $n$ is the number of tiles. For images with both width and height smaller than 640, $n$ is set to 0, i.e., Gundam mode will degrade to Base mode."

> "It's worth noting that due to our relatively large native resolutions, images won't be fragmented too much under dynamic resolution (the number of tiles is controlled within the range of 2 to 9)."

> "Note that Gundam-master mode (1024×1024 local views+1280×1280 global view) is obtained through continued training on a trained DeepSeek-OCR model."

**O-C7. 解码器：64 选 6 路由专家 + 2 共享专家，约 570M 激活。**（§3.3）
> "During inference, the model activates 6 out of 64 routed experts and 2 shared experts, with about 570M activated parameters. The 3B DeepSeekMoE is very suitable for domain-centric (OCR for us) VLM research, as it obtains the expressive capability of a 3B model while enjoying the inference efficiency of a 500M small model."

**O-C8. 数据配比与序列长度。**（§3.4.4）
> "To ensure the model's language capabilities, we introduced 10% of in-house text-only pretrain data, with all data processed to a length of 8192 tokens, which is also the sequence length for DeepSeek-OCR. In summary, when training DeepSeek-OCR, OCR data accounts for 70%, general vision data accounts for 20%, and text-only data accounts for 10%."

**O-C9. 训练规模。**（§3.5.2）
> "We use 20 nodes (each with 8 A100-40G GPUs) for training, with a data parallelism (DP) of 40 and a global batch size of 640. We use the AdamW optimizer with a step-based scheduler and an initial learning rate of 3e-5. For text-only data, the training speed is 90B tokens/day, while for multimodal data, the training speed is 70B tokens/day."

**O-C10. 压缩实验设置：Fox 英文、600–1300 个文本 token、共 100 页；只测 Tiny/Small。**（§4.1）
> "We use the English document portion of Fox, tokenize the ground truth text with DeepSeek-OCR's tokenizer (vocabulary size of approximately 129k), and select documents with 600-1300 tokens for testing, which happens to be 100 pages. Since the number of text tokens is not large, we only need to test performance in Tiny and Small modes, where Tiny mode corresponds to 64 tokens and Small mode corresponds to 100 tokens."

**O-C11. 超过 10× 后下降的两个原因；模糊被重新解释为“遗忘机制”的特性。**（§4.1）
> "When the compression ratio exceeds 10×, performance begins to decline, which may have two reasons: one is that the layout of long documents becomes more complex, and another reason may be that long texts become blurred at 512×512 or 640×640 resolution. The first issue can be solved by rendering texts onto a single layout page, while we believe the second issue will become a feature of the forgetting mechanism. When compressing tokens by nearly 20×, we find that precision can still approach 60%."

**O-C12. 实用性能（OmniDocBench）。**（§4.2）
> "Requiring only 100 vision tokens (640×640 resolution), DeepSeek-OCR surpasses GOT-OCR2.0 [38] which uses 256 tokens; with 400 tokens (285 valid tokens, 1280×1280 resolution), it achieves on-par performance with state-of-the-arts on this benchmark. Using fewer than 800 tokens (Gundam mode), DeepSeek-OCR outperforms MinerU2.0 [34] which needs nearly 7,000 vision tokens."

> "For newspapers, Gundam or even Gundam-master mode is required to achieve acceptable edit distances, because the text tokens in newspapers are 4-5,000, far exceeding the 10× compression of other modes."

**O-C13. 生产吞吐。**（摘要与 §1）
> "In production, DeepSeek-OCR can generate training data for LLMs/VLMs at a scale of 200k+ pages per day (a single A100-40G)."

> "In production, DeepSeek-OCR can generate 33 million pages of data per day for LLMs or VLMs using 20 nodes (each with 8 A100-40G GPUs)."

**O-C14. 遗忘机制设想（图 13 标题与 §5）。**
> "Forgetting mechanisms constitute one of the most fundamental characteristics of human memory. The contexts optical compression approach can simulate this mechanism by rendering previous rounds of historical text onto images for initial compression, then progressively resizing older images to achieve multi-level compression, where token counts gradually decrease and text becomes increasingly blurred, thereby accomplishing textual forgetting."

> "These findings suggest promising directions for future applications, such as implementing optical processing for dialogue histories beyond $k$ rounds in multi-turn conversations to achieve 10× compression efficiency."

**O-C15. 作者自己承认：OCR 不足以证明“上下文光学压缩”成立。**（§6）
> "Of course, OCR alone is insufficient to fully validate true context optical compression and we will conduct digital-optical text interleaved pretraining, needle-in-a-haystack testing, and other evaluations in the future."

**O-C16. 不是聊天模型。**（图 12 标题）
> "Note that since we do not include SFT (Supervised Fine-Tuning) stage, the model is not a chatbot, and some capabilities need completion prompts to be activated."

### 1.5 DeepSeek-OCR 的开源代码（`hf-ocr-code`、`hf-ocr-config`、`gh-ocr`）

**O-K1. DeepEncoder 的输出是 CLIP 特征（去掉 CLS）与 SAM 压缩特征在通道维拼接，再线性投影到 1280。**（`modeling_deepseekocr.py`）
```python
global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
global_features = self.projector(global_features)
```
  `config.json`：`projector_config` = `{"input_dim": 2048, "n_embed": 1280, "projector_type": "linear"}`。论文正文没有写这个拼接。

**O-K2. 实际送入 LLM 的图像占位 token 比表 1 多：每行末尾一个 `image_newline`，全局视图后一个 `view_seperator`。**
```python
tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
tokenized_image += [image_token_id]
```
  其中 `patch_size = 16`、`downsample_ratio = 4`（每个空间维 4 倍，合计 16 倍）。所以 Tiny 实际占 73 个位置、Small 111、Base 273、Large 421（§4.5）。

**O-K3. 代码里的“有效 token”用 `int()`（向下取整），论文式 (1) 用上取整；代码里的“压缩比”用模型输出文本的 token 数做分子。**
```python
ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
valid_img_tokens += int(256 * ratio)
...
print('compression ratio: ', round(pure_texts_outputs_token_length/valid_img_tokens, 2))
```

**O-K4. 开源模型支持的模式不含 Gundam-M。**（`gh-ocr` README “Support-Modes”）列出 Tiny/Small/Base/Large 和 “Gundam: n×640×640 + 1×1024×1024”，没有 Gundam-master。

### 1.6 DeepSeek-OCR 2（`ocr2` v1）

**O2-C1. 核心想法：编码器按语义“重排”视觉 token，而不是按光栅顺序。**（摘要）
> "We present DeepSeek-OCR 2 to investigate the feasibility of a novel encoder—DeepEncoder V2—capable of dynamically reordering visual tokens upon image semantics. Conventional vision-language models (VLMs) invariably process visual tokens in a rigid raster-scan order (top-left to bottom-right) with fixed positional encoding when fed into LLMs."

**O2-C2. 用 LLM 结构（Qwen2-0.5B）替换 CLIP；视觉 token 双向注意力、因果流查询因果注意力；只把查询输出送给解码器。**（§1、§3.2.2）
> "we replace the CLIP [37] component in DeepEncoder [53] with a compact LLM [48] architecture"

> "We instantiate this architecture using Qwen2-0.5B [48], whose 500M parameters are comparable to CLIP ViT (300M) without introducing excessive computational overhead. The decoder-only architecture with prefix-concatenation of visual tokens proves crucial: extra experiments with cross-attention in an mBART-style [26] encoder-decoder structure fail to converge."

**O2-C3. 查询数 = 视觉 token 数；全局视图 1024² → 256，局部 768² → 144，0–6 个局部视图，总数 256–1120。**（§3.2.3）
> "the number of causal query tokens equals the number of visual tokens, computed as $\frac{W\times H}{16^{2}\times 16}$"

> "Therefore, the total number of reordered visual tokens fed to the LLM is $k\times 144+256$, ranging from $[256,1120]$. This maximum token count (1120) is lower than DeepSeek-OCR's 1156 (Gundam mode) and matches Gemini-3-Pro's maximum visual token budget."

**O2-C4. 压缩器沿用 DeepEncoder，只把最后一层卷积输出维从 1024 改为 896。**（§3.2.1）
> "The output dimension of the final convolutional layer is reduced from 1024 in DeepEncoder to 896 to align with the subsequent pipeline."

**O2-C5. 解码器不变，但激活参数写成“约 500M”。**（§3.3）
> "we retain DeepSeek-OCR's decoder − a 3B-parameter MoE structure with about 500M active parameters."

**O2-C6. 结果：OmniDocBench v1.5 总分 91.09（+3.73），阅读顺序编辑距离 0.085 → 0.057；生产重复率下降。**（§5.1、§5.3）
> "As shown in Table 1, DeepSeek-OCR 2 achieves advanced performance of 91.09% while using the smallest upper limit of visual tokens (V-token$_{max}$). Compared to the DeepSeek-OCR baseline, it demonstrates a 3.73% improvement under similar train data sources"

> "reducing the repetition rate from 6.25% to 4.17% for online user-log images, and from 3.69% to 2.88% for PDF data production."

**O2-C7. 退步之处：报纸类文本编辑距离 >0.13。**（§5.2）
> "For text recognition Edit Distance (ED), DeepSeek-OCR 2 outperforms DeepSeek-OCR in most cases, but there are also notable weaknesses, such as newspapers, where it performs >0.13 ED."

### 1.7 第三方检验（只读摘要，方向性引用）

**X-C1. `ocr-crutch`：语义扰动后 DeepSeek-OCR 从约 90% 跌到约 20%；约 1 万文本 token 处全面崩溃。**（摘要）
> "performance plummets from approximately 90% to 20%"

  （条件：用句子级、词级语义扰动去掉语言先验之后。）另一处短引：
> "a total model collapse around 10,000 text tokens"

  作者据此认为当前的光学压缩可能反而加重长上下文瓶颈（转述）。

**X-C2. `engram-nine`：高频 n-gram 用无碰撞的完美哈希，等参数下并不稳定地更好；碰撞可能起正则化作用。**（摘要）
> "the collision-free design does not consistently improve validation loss"

  （严格等参数设置下。）作者还观察到无碰撞配置更早出现“热/冷位置损失翻转”，并据此提出
> "collisions act as implicit regularization"

### 1.8 DeepSeek 多模态谱系（每条一行，均已核对）

| 模型 | 出处 | 一句话（已核对） | 引文 |
|---|---|---|---|
| DeepSeek-VL（1.3B/7B） | 2403.05525v2 | SAM-B（1024²）+ SigLIP-L（384²）混合编码器，把 1024² 图压成 576 个 token；SAM 特征 64×64×256 先插值到 96×96，再过两层 stride-2 卷积成 24×24×1024 | "condenses a 1024×1024 resolution image … into 576 tokens"（§1）；"two convolutional layers with a stride of 2"（§3.1 VL Adaptor，输出 24×24×1024）（该文为 arXiv 非独占许可，只作短引） |
| DeepSeek-VL2（Tiny/Small/标准） | 2412.10302v1 | 动态切块 + 2×2 pixel shuffle（每块 27×27 → 196 token），语言部分换成 DeepSeekMoE（3B/16B/27B，激活 0.57B/2.4B/4.1B；Small 与标准版用 MLA） | "we implement a 2×2 pixel shuffle operation to compress each tile's visual tokens from 27×27 to 14×14=196 tokens"（§2 Vision-Language Adaptor）；"These LLMs have 0.57B, 2.4B, and 4.1B activated parameters respectively."（§1） |
| Janus（1.3B） | 2410.13848v1 | 理解用 SigLIP-L/16-384、生成用码本 16,384 的 VQ tokenizer，两条视觉编码通路共用一个自回归 Transformer | "we decouple visual encoding into separate pathways, while still leveraging a single, unified transformer architecture for processing"（摘要） |
| Janus-Pro（1B/7B） | 2501.17811v1 | 更好的训练策略 + 更多数据 + 放大到 7B；Janus-Pro-7B MMBench 79.2、GenEval 0.80 | "Janus-Pro-7B achieved a score of 79.2 on the multimodal understanding benchmark MMBench"；"Janus-Pro-7B scores 0.80"（§1） |
| DeepSeek-OCR | 2510.18234v1 | SAM-base + 16× 卷积 + CLIP-large 串联（DeepEncoder），解码器 DeepSeek3B-MoE-A570M；研究“视觉 token 能装下多少文本 token” | O-C1、O-C5 |
| DeepSeek-OCR 2 | 2601.20552v1 | DeepEncoder V2：用 Qwen2-0.5B 结构替换 CLIP，加“因果流查询”重排视觉 token | O2-C1、O2-C2 |
| DeepSeek-V4.1-Flash 的视觉部分 | 2609.19969v1 | 从头训练的 DeepSeek-ViT（32 层、1024 维、patch 14）+ 3×3 pixel-unshuffle + MLP 投影；不用 DeepEncoder | "A 3×3 pixel-unshuffle operation then rearranges each local neighborhood along the channel dimension"（§2.1.1）；"We train a vision encoder named DeepSeek-ViT from scratch"（§2.1.1） |

  谱系注记（笔者推导，已用配置核对）：
  - OCR 的解码器与 VL2-Tiny 的语言模型是同一规格：VL2 表 1 的 Tiny 列为 12 层、嵌入 1,280、10 头、MHA、64 路由 + 2 共享、top-6；
    OCR 的 `config.json` 完全相同（`num_hidden_layers 12`、`hidden_size 1280`、`num_attention_heads 10`、`use_mla false`、
    `n_routed_experts 64`、`n_shared_experts 2`、`num_experts_per_tok 6`），VL2 §1 与其训练超参数表写该 3B 模型激活 0.57B。
  - Engram 论文的 MoE-27B 基线与 VL2 标准版的 27B 语言模型规格一致：30 层、隐藏 2560、MLA、32 头、72 路由 + 2 共享、top-6
    （VL2 表 1；VL2 标准版用 sigmoid 路由 + 专家偏置修正，与 Engram 表 5 的 “Loss Free” 均衡一致；词表同为 129,280）；VL2 的 `config.json` 给出 `moe_intermediate_size 1536`、`first_k_dense_replace 1`。Engram 论文没写专家中间维，
    本笔记 §4.2 用 1536 复算出 ρ=74.3%，可作为旁证，但**“两者是同一个骨干”仍是笔者推测**。
  - DeepSeek-VL（SAM-B + 两层 stride-2 卷积）→ DeepEncoder（SAM-base + 两层 stride-2 卷积，16×）是一脉相承的压缩器设计；
    DeepSeek-OCR 自己说是 “borrow from Vary [36]”。
  - Janus-Pro §2.4 说 “The previous version of Janus validates the effectiveness of visual encoding decoupling using a 1.5B LLM”，
    而 Janus 论文 §4.1 写 “DeepSeek-LLM (1.3B)”——同一个模型两个数，见勘误 E14。

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 Engram（`engram` v2，式号为原文式号）

**式 (1) 哈希检索**（§2.2）

$$z_{t,n,k}\triangleq\varphi_{n,k}(g_{t,n}),\quad\mathbf{e}_{t,n,k}=\mathbf{E}_{n,k}[z_{t,n,k}].\tag{1}$$

其中 $g_{t,n}=(x'_{t-n+1},\dots,x'_{t})$，$x'_t=\mathcal P(x_t)$ 是压缩后的规范化 ID。

**式 (2) 拼接**

$$\mathbf{e}_{t}\triangleq\mathop{\|}_{n=2}^{N}\mathop{\|}_{k=1}^{K}\mathbf{e}_{t,n,k}.\tag{2}$$

$\mathbf e_t\in\mathbb R^{d_{\text{mem}}}$。27B 配置：$N=3$（阶 2、3），$K=8$，$d_{\text{mem}}=1280$，故每个 $\mathbf e_{t,n,k}\in\mathbb R^{80}$（笔者推导：$1280/(2\times8)$，与代码 `D = n_embed_per_ngram // n_head_per_ngram` 的约定一致）。

**式 (3) 记忆的 Key/Value 投影**（§2.3）

$$\mathbf{k}_{t}=\mathbf{W}_{K}\mathbf{e}_{t},\quad\mathbf{v}_{t}=\mathbf{W}_{V}\mathbf{e}_{t}\tag{3}$$

**式 (4) 标量门控**

$$\alpha_{t}=\sigma\left(\frac{\text{RMSNorm}(\mathbf{h}_{t})^{\top}\text{RMSNorm}(\mathbf{k}_{t})}{\sqrt{d}}\right).\tag{4}$$

门控后 $\tilde{\mathbf v}_t=\alpha_t\cdot\mathbf v_t$。

**式 (5) 短卷积 + 残差**（$\tilde{\mathbf V}\in\mathbb R^{T\times d}$，按行堆叠）

$$\mathbf{Y}=\text{SiLU}\left(\text{Conv1D}(\text{RMSNorm}(\tilde{\mathbf{V}}))\right)+\tilde{\mathbf{V}},\tag{5}$$

再 $\mathbf H^{(\ell)}\leftarrow\mathbf H^{(\ell)}+\mathbf Y$。

**式 (6) 多分支门控**（§2.4）

$$\alpha_{t}^{(m)}=\sigma\left(\frac{\text{RMSNorm}(\mathbf{h}_{t}^{(m)})^{\top}\text{RMSNorm}(\mathbf{W}_{K}^{(m)}\mathbf{e}_{t})}{\sqrt{d}}\right).\tag{6}$$

$\mathbf u_t^{(m)}=\alpha_t^{(m)}\cdot(\mathbf W_V\mathbf e_t)$。

**式 (7) 分配比**（§3.1）

$$P_{\mathrm{MoE}}^{(\mathrm{sparse})}=\rho\,P_{\mathrm{sparse}},\qquad P_{\mathrm{Engram}}=(1-\rho)\,P_{\mathrm{sparse}}.\tag{7}$$

$P_{\mathrm{sparse}}\triangleq P_{\mathrm{tot}}-P_{\mathrm{act}}$；$P_{\mathrm{tot}}$ “excluding vocabulary embedding and LM head”。（ACL 版把 $P_{\text{sparse}}$ 记作 $P_{\text{free}}$。）

**式 (8) CKA**（§6.1.2）

$$\text{CKA}(K,L)=\frac{\text{HSIC}(K,L)}{\sqrt{\text{HSIC}(K,K)\text{HSIC}(L,L)}}\tag{8}$$

$K=XX^\top$，$L=YY^\top$（线性核 Gram 矩阵）。

**式 (9) 软对齐指数**

$$a_{j}=\frac{\sum_{i\in\mathcal{I}_{j}}S_{i,j}\cdot i}{\sum_{i\in\mathcal{I}_{j}}S_{i,j}},\quad\text{where }\mathcal{I}_{j}=\mathop{\text{argtop}k}_{i}(S_{i,j}).\tag{9}$$

$k=5$。

**代码中的门控（论文没写）**——`gh-engram` 与 `hf-v41-code` 两份实现都是

$$\alpha_t=\sigma\!\Big(\operatorname{sign}(s_t)\sqrt{\max(|s_t|,10^{-6})}\Big),\qquad s_t=\frac{\text{RMSNorm}(\mathbf h_t)^\top\text{RMSNorm}(\mathbf k_t)}{\sqrt d}.$$

V4.1 代码里 RMSNorm 的两个可学习缩放合并成一个逐元素权重 `q_weight * k_weight`（注释 “only ever used as a product”），归一化按“每个 token、每个 hc 副本”在 $d$ 维上做。

**代码中的哈希（论文只写“乘法-异或”）**——笔者据 `engram.py` 写成公式：

$$\text{mix}_{t,n}=\bigoplus_{i=0}^{n-1}\big(x'_{t-i}\cdot c^{(\ell)}_i\big),\qquad z_{t,n,k}=\text{mix}_{t,n}\bmod p^{(\ell)}_{n,k}\;+\;\text{offset}_{n,k},$$

其中 $\oplus$ 是按位异或，$c^{(\ell)}_i$ 是第 $\ell$ 个 Engram 层的随机奇数乘子（种子 `10007 * layer_id`），$p^{(\ell)}_{n,k}$ 是全模型互不相同的素数，
offset 是把所有 (阶, 头) 的表拼成一张大表时的行偏移。位置 $t-i<0$ 或图像 token 处用 pad（V4.1 代码称 DEAD 截断）。
注意：**同一阶的 $K$ 个头共用同一个 $\text{mix}_{t,n}$，只是模不同**。

**V4.1 的 Sinkhorn 平衡动量更新**（`dsv41` §2.5，算法 1 与式 (7)）

算法 1 第 2–3 行（Nesterov 动量）：
$$M_{t}\leftarrow\beta M_{t-1}+(1-\beta)G_{t},\qquad \widehat{G}_{t}\leftarrow\beta M_{t}+(1-\beta)G_{t}$$
第 6 行：若 $\rho_i\le\tau\bar\rho$（$\rho_i=\lVert\widehat G_{t,i,:}\rVert_2$）则把该行置零；第 7–17 行：$K$（奇数）步交替做行、列 $\ell_2$ 归一化
$$U^{(k)}_{i,:}\leftarrow U^{(k-1)}_{i,:}/(\lVert U^{(k-1)}_{i,:}\rVert_{2}+\varepsilon)\ (k\text{ 奇}),\qquad U^{(k)}_{:,j}\leftarrow U^{(k-1)}_{:,j}/(\lVert U^{(k-1)}_{:,j}\rVert_{2}+\varepsilon)\ (k\text{ 偶});$$
第 18–20 行：$\Delta_t\leftarrow\sqrt n\,U^{(K)}$，$\tilde\eta_t\leftarrow\gamma\eta_t$，$W_{t+1}\leftarrow W_t-\tilde\eta_t\Delta_t$。

$$\Delta_{t}=\sqrt{n}\,U^{(K)}=\sqrt{n}\,D_{r}\widehat{G}_{t}D_{c},\qquad\frac{1}{n}\sum_{j=1}^{n}(\Delta_{t})_{ij}^{2}\approx 1,\qquad\frac{1}{m}\sum_{i=1}^{m}(\Delta_{t})_{ij}^{2}\approx 1,\tag{7, V4.1}$$

$m$ = 行数（词表 / n-gram 槽数），$n$ = 隐藏维；$\gamma=0.18$，$K=11$，$\tau=10^{-3}$，$\varepsilon=10^{-20}$。
（第 16 课 mHC 的 Sinkhorn–Knopp 是把矩阵投影到双随机矩阵；这里是把**更新量**的行、列 RMS 拉平，两者同名不同用途。）

### 2.2 课程记号对照（Engram / V4.1 优化器）

| 原文记号 | 含义 | 课程记号建议 | 理由 |
|---|---|---|---|
| $K$（哈希头数） | 每阶的哈希头数 | $K_{\text{hash}}$ | 课程的 $K_r$ 是激活路由专家数 |
| $N$（最大阶） | 最大 N-gram 阶 | $N_{\text{gram}}$ | 课程的 $N$ 是参数量 |
| $M_{n,k}$ | 第 $(n,k)$ 张表的大小（素数） | $p_{n,k}$ | 课程的 $M$ 是每 token 非嵌入 FLOPs；论文里 $M$ 还同时表示分支数和“槽数” |
| $M$（§2.4） | mHC 分支数 | $n_{hc}$（与第 16 课笔记核对后统一） | V4.1 论文写 $n$ 条残差流，代码写 `hc_mult` |
| $M$（§3.2） | 嵌入槽数 | $S$ 或 “槽数” | 同上 |
| $\mathbf W_K,\mathbf W_V$ | 记忆的 Key/Value 投影 | $W^{K}_{\text{mem}},W^{V}_{\text{mem}}$ | 避免与注意力的 $W^K,W^V$ 混淆 |
| $\mathbf k_t,\mathbf v_t$（式 3） | 记忆 Key/Value | $\mathbf k^{\text{mem}}_t,\mathbf v^{\text{mem}}_t$ | 同上 |
| $\mathbf h_t$ | 当前隐藏状态（Query） | $\mathbf h_t\in\R^d$ | 与课程一致 |
| $\mathbf H^{(\ell)}\in\R^{T\times d}$ | 第 $\ell$ 层隐藏状态（按行堆叠） | $H^{(l)}$，写明“每行一个 token” | 课程层号用 $l$ |
| $d_{\text{mem}}$ | 拼接后记忆向量维度 | 保留 | |
| $\alpha_t$ | 标量门控 $\in(0,1)$ | 保留 | |
| $P_{\text{tot}},P_{\text{act}},P_{\text{sparse}}$ | 总/激活/不激活参数 | $N_{\text{tot}},N_{\text{act}},N_{\text{sparse}}$ | 课程 $N$ = 参数量；注明论文记号 |
| $\rho$ | 分配给 MoE 的稀疏预算比例 | 保留 | 课程未占用 |
| 算法 1 的 $\beta$ | 动量系数 | $\mu$ | 课程 $\beta$ = KL 系数 |
| 算法 1 的 $K$ | 归一化步数（奇数） | $K_{\text{sk}}$ | 同上避免冲突 |
| 算法 1 的 $m,n$ | 矩阵行数、列数 | 就地说明 | |

### 2.3 DeepSeek-OCR（`ocr` v1）

**式 (1) 有效视觉 token 数**（§3.2.2，Base/Large 模式填充后）

$$N_{valid}=\lceil N_{actual}\times[1-((max(w,h)-min(w,h))/(max(w,h)))]\rceil\tag{1}$$

化简（笔者推导）：$1-\frac{\max-\min}{\max}=\frac{\min(w,h)}{\max(w,h)}$，所以 $N_{valid}=\lceil N_{actual}\cdot\min(w,h)/\max(w,h)\rceil$。

**式 (2) 解码器的映射**（§3.3）

$$f_{\text{dec}}:\mathbb{R}^{n\times d_{\text{latent}}}\rightarrow\mathbb{R}^{N\times d_{\text{text}}};\quad\hat{\mathbf{X}}=f_{\text{dec}}(\mathbf{Z})\quad\text{where }n\leq N\tag{2}$$

**压缩比**（图 1 标题的文字定义，笔者写成式子）：$r=\dfrac{\#\text{ground-truth text tokens}}{\#\text{vision tokens}}$。

**每模式 token 数**（笔者推导，与表 1 吻合）：分辨率 $R$、patch 16、卷积 16× 压缩 ⇒ $n_v=\dfrac{(R/16)^2}{16}=\left(\dfrac{R}{64}\right)^2$。

课程记号：$n\to n_v$（视觉 token 数），$N\to n_t$（文本 token 数，避免与参数量 $N$ 冲突），$r=n_t/n_v$。

### 2.4 DeepSeek-OCR 2（`ocr2` v1）

**查询数**（§3.2.3）：$\dfrac{W\times H}{16^{2}\times 16}$。

**式 (1) 注意力掩码**（§3.2.4）

$$M=\begin{bmatrix}\mathbf{1}_{m\times m}&\mathbf{0}_{m\times n}\\ \mathbf{1}_{n\times m}&\text{LowerTri}(n)\end{bmatrix},\quad\text{where }n=m\tag{1}$$

$m$ = 视觉 token 数，$n$ = 因果查询数，LowerTri 含对角线。

**式 (2) 前向**（§3.3）

$$\mathbf{O}=\mathcal{D}\left(\pi_{Q}\left(\mathcal{T}^{L}\left(\mathcal{E}(\mathbf{I})\oplus\mathbf{Q}_{0};\mathbf{M}\right)\right)\right)\tag{2}$$

$\mathcal E$：视觉分词器（SAM + 卷积）；$\mathbf Q_0\in\R^{n\times d}$：可学习因果查询；$\oplus$：序列拼接；$\pi_Q$：取最后 $n$ 个位置；$\mathcal D$：语言解码器。
原文写 $\mathbf M\in\{0,1\}^{2n\times2n}$。课程记号：$m\to n_v$，$n\to n_q(=n_v)$。

---

## 3. 规格与超参数表

### 3.1 Engram 27B 规模实验（`engram` 表 1、附录 A 表 5；ACL 版表 3 同）

| 项 | Dense-4B | MoE-27B | Engram-27B | Engram-40B | 出处 |
|---|---|---|---|---|---|
| 总参数 | 4.1B | 26.7B | 26.7B | 39.5B | 表 1、表 5（ACL 表 1 把 Dense 印成 3.8B） |
| 激活参数（不含 token 嵌入） | 3.8B | 3.8B | 3.8B | 3.8B | 表 1 |
| 训练 token | 262B | 262B | 262B | 262B | 表 1 |
| 层数 / 隐藏维 | 30 / 2560 | 同 | 同 | 同 | 表 5 |
| 注意力 | MLA，32 头 | 同 | 同 | 同 | §4.1、表 5 |
| 残差 | mHC，扩展率 4 | 同 | 同 | 同 | §4.1、表 5 |
| 前置稠密层 | — | 1 | 1 | 1 | 表 5 |
| 路由专家 / 激活 / 共享 | — | 72 / 6 / 2 | 55 / 6 / 2 | 55 / 6 / 2 | 表 1、表 5 |
| 负载均衡 | — | Loss Free (Wang et al., 2024a) | 同 | 同 | 表 5 |
| RoPE θ | 10000 | 同 | 同 | 同 | 表 5 |
| 序列长度 / batch / 步数 | 4096 / 1280 / 50000 | 同 | 同 | 同 | 表 5（1280×4096×50000 = 262.1B token，§4.1 复算） |
| 词表 | 129280 | 同 | 同 | 同 | 表 5 |
| 骨干优化器 / 基础 LR / 调度 / WD | Muon / 4e-4 / Step Decay / 0.1 | 同 | 同 | 同 | 表 5 |
| Engram 参数 | — | — | 5.7B | 18.5B | 表 1 |
| $d_{\text{mem}}$ | — | — | 1280 | 1280 | 表 5 |
| Engram Vocab Size | — | — | 2,262,400 | 7,239,680 | 表 5（笔者解读：每个 (阶, 头) 表的起始大小，见 §4.2） |
| 哈希头数 / 阶 / 层 | — | — | 8 / [2,3] / [2,15] | 同 | 表 5 |
| 分词器压缩 / 与 mHC 结合 / 卷积零初始化 | — | — | True / True / True | 同 | 表 5 |
| Engram 优化器 / LR 倍数 / WD | — | — | Adam / ×5 / 0.0 | 同 | 表 5、§4.1 |
| 分配比 ρ | 1（纯 MoE） | — | 74.3% | — | §4.1 |

### 3.2 其他 Engram 实验设置（`engram`）

| 实验 | 设置 | 出处 |
|---|---|---|
| 稀疏分配（两点） | $C=2\times10^{20}$：$P_{tot}\approx5.7$B、$P_{act}=568$M、基线 106 个专家；$C=6\times10^{20}$：$P_{tot}\approx9.9$B、$P_{act}=993$M、基线 99 个专家；$P_{tot}/P_{act}\approx10$ | §3.1 |
| 无限内存 | 固定 MoE 骨干 $P_{tot}\approx3$B、$P_{act}=568$M、100B token；槽数 $2.58\times10^5$→$1.0\times10^7$（至多约 13B 参数）；对照 OverEncoding | §3.2 |
| 结构消融 | 12 层 3B MoE（0.56B 激活）、100B token，基线 val loss 1.808；+1.6B Engram，参考配置 {2,3}-gram、第 2 与第 6 层，val loss 1.768；单模块第 2 层 1.770 | §6.2 |
| 长上下文扩展 | YaRN，32768 上下文，5,000 步（30B token）；$s=10,\alpha=1,\beta=32$，缩放因子 $f=0.707$；比较 MoE-27B(50k) 与 Engram-27B(41k/46k/50k) | §5.1 |
| 卸载吞吐 | nano-vLLM；Dense-4B / Dense-8B；100B 参数 Engram 放第 2 个 Transformer 块，整表在主机 DRAM；H800；512 条序列，长度 Uniform(100,1024) | §6.4、表 4 |
| CKA | Few-NERD 数据集，取命名实体最后一个 token 的隐藏状态；无偏 HSIC 小批估计 | §6.1.2 |

### 3.3 Engram 实现：论文 vs 演示代码 vs V4.1

| 项 | Engram 论文（27B） | `engram_demo_v1.py` | V4.1 论文 | V4.1 `config.json` / 代码 |
|---|---|---|---|---|
| N-gram 阶 | {2,3} | `max_ngram_size=3` → {2,3} | {2,3,4} | `engram_max_ngram_size: 4` |
| 每阶哈希头 | 8 | 8 | 8 | `engram_n_heads: 8` |
| 每头维度 | 80（笔者推导） | 512/8 = 64 | 2048/8 = 256（笔者推导） | `engram_head_dim: 256` |
| 每阶维度 | 640（推导） | 512 | 2048 | — |
| 表起始大小 | 2,262,400（表 5） | `129280*5` | ≈16M | `engram_vocab_size: 16000000` |
| 表大小取法 | “prime size” | 向上找未用过的素数 | “distinct primes” | 同（`find_next_prime`） |
| 每模块总行数 | — | — | — | `engram_num_embeddings: [384006168, 384016682]` |
| 层号 | “layers 2 and 15” | `layer_ids=[1, 15]`（0 起） | “layers 1 and 14 (zero-indexed)” | `engram_layer_ids: [1, 14]` |
| 门控 | 式 (4)，无平方根 | 带符号平方根 | “original design” | 带符号平方根（注释 “matching the training kernel”） |
| 短卷积 | 有（w=4, δ=N） | 有（`dilation=max_ngram_size`） | **去掉** | 无 |
| 精度 | 投影融合为 FP8 GEMM | fp32/bf16 演示 | 表与 K/V 投影 FP8 | 表 `float8_e4m3fn`，每 32 维一个 `float8_e8m0fnu` 缩放 |
| 嵌入优化器 | Adam，LR×5，WD 0 | — | 动量 + Sinkhorn 平衡，LR×5，无 WD | — |
| 图像 token | — | — | 未写 | 不参与 n-gram，门控强制为 0 |
| 压缩后词表 | 128k 词表减 23%（23.43%） | 运行时计算 | — | `engram_compressed_vocab_size: 99092` |
| 骨干隐藏维 $d$ | 2560 | 1024（演示） | 5120 | `hidden_size: 5120` |
| 分支数 | mHC $M=4$ | `hc_mult=4` | mHC 扩展 4 | `hc_mult: 4` |

### 3.4 V4.1 中与 Engram 相关的规格（`dsv41`，`hf-v41-config`）

| 项 | 值 | 出处 |
|---|---|---|
| 骨干参数 / Engram 参数 | 552B / 196B | §2.1；模型卡 |
| 激活参数 | prefill 8B / decode 16B | §2.1 |
| 层数 / 隐藏维 | 40 / 5120 | §4.2.1；config |
| Engram 模块数 / 位置 | 2 / 第 1、14 层（0 起） | §2.4.2；config |
| 阶 / 头 / 每阶维度 / 每头表大小 | {2,3,4} / 8 / 2048 / ≈16M 素数 | §2.4.2 |
| 训练：表按行切分到 engram 并行组；优化器状态再按副本切分；预取在每个流水段开始前对整个本地 batch 发起；梯度在反传后回送 | — | §3.1.3 |
| 推理：主机内存经后台 RDMA 预取，第一模块的预取与第 1 个 Transformer 块重叠 | — | §2.4.2 |
| RL rollout：表常驻 GPU | — | §3.1.3 |
| Sinkhorn 平衡更新：$K=11$、$\tau=10^{-3}$、$\varepsilon=10^{-20}$、动量与 LR 修正同 Muon（动量 0.95、RMS 0.18） | — | §4.2.2 |
| Engram LR 倍数 | ×5 | §4.2.2 |
| 预训练 token | 45T（多模态） | §4.2.2 |

### 3.5 DeepSeek-OCR 规格（`ocr` 与 `hf-ocr-config`）

**分辨率模式（表 1）**

| 模式 | 分辨率 | 视觉 token（论文） | 处理 | 代码实际占位（含换行/分隔，§4.5） |
|---|---|---|---|---|
| Tiny | 512 | 64 | resize | 73 |
| Small | 640 | 100 | resize | 111 |
| Base | 1024 | 256 | padding | 273 |
| Large | 1280 | 400 | padding | 421 |
| Gundam | 640 瓦片 + 1024 全局 | $n\times100+256$（$n$=2–9） | resize + padding | 见 §4.5 |
| Gundam-M | 1024 瓦片 + 1280 全局 | $n\times256+400$ | resize + padding；另行续训；开源版未提供 | — |

**DeepEncoder 与解码器**

| 项 | 论文 | config / 代码 |
|---|---|---|
| DeepEncoder 总参数 | ≈380M（80M SAM-base + 300M CLIP-large） | — |
| SAM | SAM-base，patch 16，窗口注意力为主 | `sam_vit_b`：12 层、宽 768、12 头、全局注意力层 [2,5,8,11]、`downsample_channels [512, 1024]` |
| 压缩器 | 2 层卷积，k=3、s=2、p=1，通道 256→1024，16× | 通道 256→512→1024（config）；`downsample_ratio = 4`（每维） |
| CLIP | CLIP-large，去掉首层 patch embedding | `clip-l-14-224`：24 层、宽 1024、16 头 |
| 投影 | 未写 | 线性，2048（CLIP 1024 ⊕ SAM 1024）→ 1280 |
| 解码器 | DeepSeek-3B-MoE，64 选 6 + 2 共享，≈570M 激活，12 层 | 12 层、hidden 1280、10 头 MHA（`use_mla: false`）、`moe_intermediate_size 896`、首层稠密 `intermediate_size 6848`、词表 129,280、`max_position_embeddings 8192` |
| 序列长度 | 8192 | 同上 |
| 数据配比 | OCR 70% / 通用视觉 20% / 纯文本 10% | — |
| DeepEncoder 训练 | 2 epoch，batch 1280，AdamW，cosine，LR 5e-5，序列 4096 | — |
| 整体训练 | 20 节点 × 8 A100-40G，PP=4，DP=40，全局 batch 640，AdamW，LR 3e-5；文本 90B tok/天、多模态 70B tok/天 | — |
| 冻结 | SAM + 压缩器（PP0）冻结；CLIP（PP1）训练 | — |

**Fox 压缩实验（表 2，英文文档，600–1300 文本 token，共 100 页）**

| 文本 token | 64 视觉 token：精度 | 压缩比 | 100 视觉 token：精度 | 压缩比 | 页数 |
|---|---|---|---|---|---|
| 600–700 | 96.5% | 10.5× | 98.5% | 6.7× | 7 |
| 700–800 | 93.8% | 11.8× | 97.3% | 7.5× | 28 |
| 800–900 | 83.8% | 13.2× | 96.8% | 8.5× | 28 |
| 900–1000 | 85.9% | 15.1× | 96.8% | 9.7× | 14 |
| 1000–1100 | 79.3% | 16.5× | 91.5% | 10.6× | 11 |
| 1100–1200 | 76.4% | 17.7× | 89.8% | 11.3× | 8 |
| 1200–1300 | 59.1% | 19.7× | 87.1% | 12.6× | 4 |

**OmniDocBench（表 3，编辑距离，越低越好；只摘本课要用的行）**

| 模型 | token/页 | EN overall | ZH overall |
|---|---|---|---|
| GOT-OCR2.0 | 256 | 0.287 | 0.411 |
| MinerU2.0 | 6790 | 0.133 | 0.238 |
| Qwen2.5-VL-72B | 3949 | 0.214 | 0.261 |
| dots.ocr†200dpi | 5545 | 0.125 | 0.16 |
| DeepSeek-OCR Tiny | 64 | 0.386 | 0.361 |
| DeepSeek-OCR Small | 100 | 0.221 | 0.284 |
| DeepSeek-OCR Base | 256 (182) | 0.137 | 0.24 |
| DeepSeek-OCR Large | 400 (285) | 0.138 | 0.208 |
| DeepSeek-OCR Gundam | 795 | 0.127 | 0.181 |
| DeepSeek-OCR Gundam-M†200dpi | 1853 | 0.123 | 0.157 |

**按文档类别（表 4，overall 编辑距离；摘录）**：Slides 列 Tiny 0.116；Newspaper 列 Tiny 0.94、Small 0.744、Base 0.645、Large 0.353、Gundam 0.122、Gundam-M 0.099；
Financial Report 列 Gundam 0.289（明显高于 Base 的 0.027，论文未解释；表内模式名印作 “Guandam-M”）。

### 3.6 DeepSeek-OCR 2 规格（`ocr2`）

| 项 | 值 | 出处 |
|---|---|---|
| 视觉分词器 | 80M SAM-base + 两层卷积（16×），末层输出 896 维 | §3.2.1 |
| LLM 式编码器 | Qwen2-0.5B 结构（500M）；视觉 token 双向、查询因果 | §3.2.2 |
| 查询数 | = 视觉 token 数；全局 1024² → 256，局部 768² → 144 | §3.2.3 |
| 局部视图数 $k$ | 0–6（两边都 <768 时不裁剪） | §3.2.3 |
| 送入解码器 token | $k\times144+256\in[256,1120]$ | §3.2.3、图 4 |
| 解码器 | 沿用 DeepSeek-OCR 的 3B MoE（写作“约 500M 激活”、“DeepSeek-3B-A500M”） | §3.3、§4.2.2 |
| 数据 | 同源；OCR 数据占 80%；OCR 1.0 按文本:公式:表格 = 3:1:1 采样 | §4.1 |
| 阶段 1 | 编码器预训练：160 A100，batch 640，40k 步，8K 打包，约 1 亿图文对，LR 1e-4→1e-6 | §4.2.1 |
| 阶段 2 | 查询增强：冻结 SAM+卷积，联合训 LLM 编码器与解码器；PP=4、DP=40；batch 1280；15k 步；LR 5e-5→1e-6 | §4.2.2 |
| 阶段 3 | 冻结编码器只训解码器：20k 步，LR 1e-6→5e-8 | §4.2.3 |
| OmniDocBench v1.5 | 总分 91.09（OCR 87.36）；TextEdit 0.048（0.073）；FormulaCDM 90.31（84.14）；TableTEDs 87.75（85.25）；R-order Edit 0.057（0.085）；V-token$_{max}$ 1120（1156） | 表 1 |
| 编辑距离（表 2） | Overall 0.100（OCR 0.129；Gemini-3 Pro 0.115 @1120；Seed-1.8 0.106 @5120） | 表 2 |
| 生产重复率（表 4） | 在线日志 6.25% → 4.17%；PDF 3.69% → 2.88% | 表 4 |

---

## 4. 复算的数字

所有表达式都在 Python 3.11 中运行；素数用确定性 Miller–Rabin（底 2–37，对 64 位整数确定）实现，与代码里的 `sympy.isprime` 等价。

### 4.1 Engram 表 1 的差值（Engram-27B − MoE-27B）

| 基准 | 差值 | 基准 | 差值 | 基准 | 差值 |
|---|---|---|---|---|---|
| MMLU | **+3.0** | MMLU-Redux | **+3.4** | MMLU-Pro | +1.8 |
| CMMLU | +4.0 | C-Eval | +4.7 | AGIEval | +3.2 |
| ARC-Easy | +2.5 | ARC-Challenge | +3.7 | TriviaQA | +1.9 |
| TriviaQA-ZH | +1.5 | PopQA | +0.2 | CCPM | +7.5 |
| BBH | +5.0 | HellaSwag | +0.9 | PIQA | +1.6 |
| WinoGrande | +0.2 | DROP | +3.3 | RACE-M | +1.9 |
| RACE-H | +2.8 | C3 | +3.5 | HumanEval | +3.0 |
| MBPP | +1.6 | CruxEval-i | +1.5 | CruxEval-o | +0.9 |
| GSM8K | +2.2 | MGSM | +2.6 | MATH | +2.4 |

- `60.4-57.4 = 3.0`（MMLU），`64.0-60.6 = 3.4`（MMLU-Redux）→ 摘要的 “MMLU +3.4” 与表不符（勘误 E2）。
- Engram-27B 在全部 27 项上 ≥ MoE-27B（程序检查：差值为负的列表为空）。
- Engram-40B 低于 Engram-27B 的 4 项：C3（−1.8）、HumanEval（−2.4）、MBPP（−2.0）、MATH（−0.1）；MBPP 上 Engram-40B 甚至低于 MoE-27B（46.2 vs 46.6）。
- 知识类的反差：TriviaQA +1.9、PopQA +0.2（“知识”增益并不大），而 CCPM +7.5、BBH +5.0 更大——这正是论文“推理增益更大”的依据。

### 4.2 参数账（Engram-27B / 40B）

| 量 | 表达式 | 值 |
|---|---|---|
| 训练 token 核对 | `1280*4096*50000` | 262,144,000,000 ≈ 262B ✓ |
| 稀疏度 | `5.7/0.568`，`9.9/0.993` | 10.04，9.97 ✓（≈10） |
| ρ≈40% 时专家数 | 设激活专家 6 个不计入稀疏预算：`6+0.4*(106-6)`，`6+0.4*(99-6)` | 46.0，43.2 → 与论文 46、43 吻合（笔者推导；若设激活 8 个则得 47.2、44.4，不吻合） |
| ρ（专家计数法） | `(55-6)/(72-6)` | 0.7424 |
| 每个专家参数（笔者推断用 VL2 的 1536） | `3*2560*1536` | 11,796,480 |
| Engram-27B 表行数（素数布局） | 2 层 × 2 阶 × 8 头，从 2,262,400 起找素数 | 72,404,232 行（每层 36,199,972 / 36,204,260） |
| Engram-27B 参数 | `72404232*80` | 5,792,338,560 ≈ **5.79B**（论文写 5.7B） |
| Engram-40B 参数 | 从 7,239,680 起，同法，`231678892*80` | 18,534,311,360 ≈ 18.53B（论文 18.5B ✓） |
| 被拿掉的 17 个专家 | `17*29*11796480` | 5,815,664,640 ≈ 5.82B（与 5.79B 相差 0.4%） |
| ρ（精确） | `P_MoE=49*29*11796480=16,762,798,080`；`P_MoE/(P_MoE+5,792,338,560)` | **0.7432 = 74.3%** ✓ |
| token 嵌入 | `129280*2560` | 330,956,800 |
| MoE-27B 总参 | `3.8e9 + 66*29*11796480 + 330956800` | 26.71B ✓（说明表中“总参数”含 token 嵌入） |
| Engram-27B 总参 | `3.8e9 + 16762798080 + 5792338560 + 330956800` | 26.69B ✓ |
| Engram-40B 总参 | 同法 | 39.43B（论文 39.5B，≈） |
| Dense-4B 总参 | `3.8e9 + 330956800` | 4.13B ✓（4.1B） |
| 无限内存参数范围 | `2.58e5*1280`，`1.0e7*1280` | 0.33B → 12.8B（论文 “up to ≈13 billion” ✓；笔者推导：一个“槽”对应 $d_{mem}=1280$ 维） |

  以上“专家中间维 1536、29 个 MoE 层”是笔者据 VL2 配置和表 5（前置稠密层 1）推断的；ρ=74.3% 与总参 26.7B 两处独立吻合，
  可信度较高，但论文本身没给专家中间维。

### 4.3 V4.1 的 Engram 账

| 量 | 表达式 | 值 |
|---|---|---|
| 48 个素数表大小 | 从 16,000,000 起、2 模块 × 3 阶 × 8 头、全局不重复 | 模块 0：16,000,057 … 16,000,463；模块 1：16,000,477 … 16,000,889 |
| 每模块行数 | 24 个素数之和 | **384,006,168 / 384,016,682**——与 `config.json` 的 `engram_num_embeddings` **逐位相同** ✓ |
| Engram 参数 | `768022850*256` | 196,613,849,600 ≈ **196.6B** ✓（论文 “196B”） |
| 近似式 | `2*3*8*16e6*256` | 196.608B |
| 占全部参数 | `196.6/(552+196.6)` | 26.3% |
| 存储（FP8 + 每 32 维 1 字节 UE8M0 缩放） | `768022850*(256+8)` | 202.76 GB = 188.8 GiB（若用 BF16：393.2 GB） |
| 每 token 需取的字节 | `2*24*(256+8)` | 12,672 B（纯数值 12,288 B） |
| 每模块 K/V 投影参数 | `6144*5120*5`（输入 3×2048，输出 4 个 Key + 1 个 Value） | 157.3M；两模块 314.6M ≈ decode 激活 16B 的 2.0%（是否计入 552B / 16B，论文未说） |
| 压缩后词表 | `99092/129280` | 76.65%（即减少 23.35%；论文说 128k 词表减 23.43%，保留 76.57% → `0.7657*129280 = 98,990`）——两者差约 100 个 ID，原因未知（分词器版本或 `len(tokenizer)` 不同，见 §8） |

### 4.4 门控与卷积

| 量 | 表达式 | 值 |
|---|---|---|
| RMSNorm 后的点积范围 | RMSNorm 输出 RMS=1 ⇒ 范数 $\sqrt d$，点积 $\in[-d,d]$，除以 $\sqrt d$ | $|s|\le\sqrt d$：d=2560 时 50.60；d=5120 时 71.55（忽略可学习缩放，笔者推导） |
| 带符号平方根后 | $d^{1/4}$ | 7.11 / 8.46 |
| sigmoid 可达范围 | `1/(1+exp(-7.11))` 等 | [0.00081, 0.99919] / [0.00021, 0.99979] |
| sigmoid(x) vs sigmoid(ssqrt(x)) | x = −4, −1, −0.25, 0, 0.25, 1, 4, 9, 16 | 0.0180/0.1192，0.2689/0.2689，0.4378/0.3775，0.5/0.5，0.5622/0.6225，0.7311/0.7311，0.9820/0.8808，0.9999/0.9526，1.0000/0.9820 |
| 卷积感受野（27B：w=4、δ=3） | `(4-1)*3+1` | 10 个位置（t, t−3, t−6, t−9）；加上三元组本身的跨度 → 覆盖 t−11…t 共 12 个 token |

  带符号平方根的作用（笔者推导）：在 $|s|<1$ 处放大、$|s|>1$ 处压缩，交点在 $|s|=1$；让门控在大点积时不那么快饱和（x=4 时 0.88 而非 0.98）。
  论文没解释，代码也只说与训练内核一致。

### 4.5 DeepSeek-OCR token 算术

| 量 | 表达式 | 值 |
|---|---|---|
| 每模式 token | `(R//16)**2 // 16` | 512→64，640→100，1024→256，1280→400 ✓；OCR2 局部 768→144 ✓ |
| 网格边长 | $\sqrt{n_v}$ | 8、10、16、20（OCR2 局部 12） |
| 代码实际占位（每行 +1 换行，再 +1 分隔） | `(s+1)*s+1` | 73、111、273、421（OCR2 风格的 144 → 157，仅示意） |
| Gundam token | `n*100+256`，n=2,4,6,9 | 456、656、856、**1156**（=OCR 2 论文说的 “DeepSeek-OCR's 1156”） ✓ |
| Gundam-M token | `n*256+400`，n=2,9 | 912、2704 |
| OCR2 token | `k*144+256`，k=0…6 | 256 … **1120** ✓ |
| SAM 段 vs CLIP 段的注意力矩阵大小 | `4096**2/256**2` | 256 倍（若 CLIP 直接吃 4096 个 token，全局注意力的分数矩阵大 256 倍） |
| 表 2 页数 | `7+28+28+14+11+8+4` | 100 ✓ |
| 表 2 隐含平均文本 token | 压缩比 × 视觉 token | 64 列：672、755、845、966、1056、1133、1261；100 列：670、750、850、970、1060、1130、1260（两列一致，说明压缩比是按每档平均文本 token 算的） |
| “<10× 约 97%” | <10× 的格只有 100-token 列前四档：`(98.5*7+97.3*28+96.8*28+96.8*14)/77` | **97.14%**（按页加权，77 页）；不加权均值 97.35% |
| “20× 约 60%” | 64-token 列 1200–1300 档 | 59.1% @ 19.7×，**只有 4 页** |
| “10–12× 约 90%” | 100 列 10.6×、11.3×；64 列 10.5×、11.8× | 91.5%、89.8%、96.5%、93.8%（10.5× 那格达 96.5%，说明“~90%”是粗略概括） |
| 有效 token（A4 纵向 1654×2339） | `ceil(256*1654/2339)`，`ceil(400*1654/2339)` | 182、283（笔者观察：Base 的 182 恰好等于表 3 的平均值 “256(182)”；Large 表中为 285） |
| 16:9 横图 1920×1080 | `ceil(256*1080/1920)` | 144 |
| 生产吞吐一致性 | `33e6/(20*8)` | 206,250 页/GPU/天 = 2.39 页/秒 ✓（与 “200k+ pages per day (a single A100-40G)” 一致） |
| README 的 PDF 并发 “~2500tokens/s(an A100-40G)” 换算 | `2500*86400/2e5` | 每页约 1,080 个输出 token（笔者推算，只是量级核对） |
| 训练吞吐 | `90e9/160/86400`，`70e9/160/86400` | 6,510 / 5,064 token/s/GPU |
| 解码器参数（据 config 重建） | 嵌入与 LM 头 `2*129280*1280`；注意力 `12*4*1280^2`；首层稠密 `3*1280*6848`；11 个 MoE 层 × 66 专家 × `3*1280*896` | 总 ≈ **2.93B** ✓（“3B”）；激活（非嵌入）407.7M，加 LM 头 **573.2M** ✓（“A570M”）——笔者推导，未计 RMSNorm 与路由器 |

### 4.6 DeepSeek-OCR 2 的差值

| 量 | 表达式 | 值 |
|---|---|---|
| 总分提升 | `91.09-87.36` | 3.73 **个百分点**（论文写 “3.73%”，是绝对差） |
| 阅读顺序编辑距离 | `0.057-0.085`，相对 | −0.028，−32.9% |
| Overall 编辑距离（表 2） | `(0.100-0.129)/0.129` | −22.5% |
| 重复率 | `4.17-6.25`，`2.88-3.69` | −2.08 pp（相对 −33.3%），−0.81 pp（相对 −22.0%） |
| Gundam 与 MinerU2.0 的 token 比 | `6790/795` | 8.54× |

### 4.7 哈希碰撞（笔者推导；U 为假设值，论文没给不同 n-gram 的个数）

**模型**：把 $U$ 个不同的 n-gram 均匀散列到大小为 $p$ 的表。某个给定 n-gram 在一个头里“与至少一个别的 n-gram 共享槽”的概率
$1-(1-1/p)^{U-1}\approx1-e^{-U/p}$；每头期望碰撞对数 $\binom U2/p\approx U^2/(2p)$。

| 表大小 $p$ | $U$ | 负载 $U/p$ | 一个头里共享槽的概率 | 8 个头全都被占（按独立近似） | 每头碰撞对数 |
|---|---|---|---|---|---|
| 2,262,409（27B，2-gram 头 0） | $10^6$ | 0.442 | 0.357 | 2.65e-4 | 2.21e5 |
| 同上 | $10^7$ | 4.42 | 0.988 | 0.908 | 2.21e7 |
| 同上 | $10^8$ | 44.2 | ≈1 | ≈1 | 2.21e9 |
| 16,000,057（V4.1） | $10^6$ | 0.0625 | 0.0606 | 1.8e-10 | 3.12e4（=31,249.9） |
| 同上 | $10^7$ | 0.625 | 0.465 | 2.18e-3 | 3.12e6 |
| 同上 | $10^8$ | 6.25 | 0.998 | 0.985 | 3.12e8 |

**要点（笔者推导）**：当 $U\gg p$ 时每个槽都被许多 n-gram 共享，单看“有没有碰撞”没有意义；多头的作用是让**槽位组合**几乎唯一：
- 两个不同 n-gram 在某一对头上同时碰撞的概率约 $1/(p_ip_j)$；在 8 个头中至少 2 个头碰撞：`comb(8,2)/p**2` = 5.47e-12（27B）/ 1.09e-13（V4.1）。
  即便 $U=10^{10}$，每个 n-gram 平均只有 0.055（27B）/ 0.0011（V4.1）个“至少两头相撞”的伙伴。
- 更强的结论来自代码结构（前提：所用哈希与 `engram_demo_v1.py` / V4.1 `inference/engram.py` 相同；论文只写 “multiplicative-XOR”，27B 训练代码未公开）：同一阶所有头共用同一个 $\text{mix}\in[0,2^{63})$，只是模不同。若 $\text{mix}\ne\text{mix}'$ 却在 3 个头上都相撞，
  则 $\text{mix}-\text{mix}'$ 能被 3 个不同素数之积整除；27B：`2262409*2262413*2262437 = 1.158e19 > 2**63 = 9.223e18`；
  V4.1：三素数之积 4.10e21。所以**除非 mix 完全相同，最多只有 2 个头能同时碰撞**（中国剩余定理 + 整数范围）。
- 可能的 n-gram 空间（压缩词表 99,092）：2-gram 9.82e9，3-gram 9.73e14，4-gram 9.64e19——远大于任何一张表，碰撞是常态，
  所以论文才需要上下文门控来“抑制噪声”（E-C6），第三方 `engram-nine` 还发现碰撞可能起正则化作用（X-C2）。

### 4.8 系统与吞吐

| 量 | 表达式 | 值 |
|---|---|---|
| 4B 吞吐损失 | `1-8858.28/9031.62` | **1.919%** |
| 8B 吞吐损失 | `1-6140.02/6315.52` | **2.779%**（论文 “2.8%” ✓；ACL 版补 “1.9% on the 4B” ✓） |
| U 形改进 | `1.7248-1.7109`，相对 | 0.0139，0.81% |
| 长上下文 “82%” | `41/50` | 0.82 ✓ |
| 若按 27B 的 Engram 配置、BF16 取数（假设） | `2*16*80*2` B/token × 9031.62 tok/s | 5,120 B/token → 46.2 MB/s（示意：主机→GPU 的流量只取决于“每 token 取几行”，与表大小无关） |
| 若按 V4.1 配置 | `12672*9031.62` | 114.4 MB/s（示意） |
| Engram-27B 投影的 FLOPs | 每层 `5*1280*2560` 参数，2 层，×2 FLOPs | ≈65.5 MFLOPs/token，约为 `2*3.8e9` 的 0.86%（笔者推算） |

### 4.9 从图上读出的数值（只用于重绘示意；不是论文正文数字）

> 核查更新：图 3、图 5、图 6 在 arXiv v2 HTML 里是矢量 SVG（`https://arxiv.org/html/2601.07372v2/scaling_law.svg`、`arch_ablation.svg`、
> `sensitivity_analysis.svg`）。核查时直接读取标记点的坐标，用坐标轴刻度线换算成数值；刻度标签由字形序列解码（图 6 的字形号就是 ASCII 码）。
> 换算的误差来自刻度线本身，约 ±0.0001 损失，比原先目测（±0.001）精确一个量级。下面先给原目测值，再给 SVG 反算值；两者相差 ≤0.0005。
> 这些数仍然是**从图反算**的，引用时要写“据图 3 读取”，不能当作论文数字。

**图 3 左（U 形）**：
- $6\times10^{20}$（紫色圆点，左轴，刻度 1.710–1.745、步长 0.005）：ρ = 40, 50, 60, 70, 75, 80, 85, 90, 95, 100 (%) → 目测 1.7247, 1.7196, 1.7137, 1.7123, 1.7126, **1.7109**, 1.7095, 1.7125, 1.7130, **1.7248**
  （加粗的两个与正文一致）。SVG 反算：1.7247, 1.7196, 1.7136, 1.7122, 1.7126, **1.7109**, **1.7095**, 1.7125, 1.7130, **1.7248**；
  这一组点的横坐标并不正好落在整十百分比上，而是 ρ≈39.7, 50.5, 60.2, 69.8, 75.2, 79.5, 84.9, 90.3, 94.6, 100（%）（笔者推测：对应整数个专家）。
- $2\times10^{20}$（青色菱形，右轴，刻度 1.795–1.830、步长 0.005）：同样 ρ → 目测 1.8276, 1.8227, 1.8191, 1.8172, 1.8177, 1.8170, 1.8156, 1.8172, 1.8202, 1.8293；
  SVG 反算 1.8275, 1.8227, 1.8190, 1.8172, 1.8177, 1.8170, **1.8155**, 1.8172, 1.8201, 1.8292（横坐标正好在整十/整五百分比上）。
- 二次拟合（`np.polyfit`）：6e20 顶点 ρ≈0.738–0.739、损失 1.7106–1.7107；2e20 顶点 ρ≈0.724–0.725、损失 1.8161–1.8162。
  **两个算力点的最低观测点都在 ρ≈85%**（SVG 反算确认：6e20 为 ρ≈84.9% 处 1.7095，2e20 为 85% 处 1.8155），
  论文正文把 1.7109（ρ≈80%）称为 “near the optimum”，并说最优在 ρ≈75%–80%。结论“最优在 70%–85% 一带、纯 MoE 明显更差”是稳的，精确位置不稳。

**图 3 右（无限内存）**：Engram 8 个点（槽数, 损失）目测 ≈ (2.56e5, 1.7885), (6.45e5, 1.7762), (1.285e6, 1.7681), (1.94e6, 1.7656), (2.58e6, 1.7604),
(3.88e6, 1.7529), (6.50e6, 1.7483), (1.04e7, 1.7432)；SVG 反算 (2.586e5, 1.7883), (6.464e5, 1.7760), (1.293e6, 1.7678), (1.939e6, 1.7654), (2.586e6, 1.7603),
(3.878e6, 1.7527), (6.464e6, 1.7481), (1.034e7, 1.7432)。纯 MoE 目测 ≈ 1.8087，SVG 反算 **1.8082**（与 §6.2 的 3B 基线 1.808 一致，笔者推测是同一骨干）；
OverEncoding 4 点目测 ≈ (1.29e6, 1.7823), (2.58e6, 1.7715), (5.0e6, 1.7673), (1.27e7, 1.7506)，SVG 反算 (1.293e6, 1.7820), (2.586e6, 1.7711), (5.171e6, 1.7670), (1.293e7, 1.7505)。
线性拟合（SVG 值）：每十倍槽数损失下降 **0.0284**（`np.polyfit(log10 M, L, 1)`）；对数-对数斜率 −0.00699（在这么窄的范围内“幂律”与“对数线性”无法区分）。

**图 4(a)（LogitLens KL，目测）**：MoE-27B 第 0–17 层约 10.0 → 6.4，第 18 层骤降到约 3.9，末层约 0.5；Engram-27B 从第 2 层起比 MoE 低约 0.5–1.0，
第 0 层反而略高（约 10.35 vs 10.0）；第 18 层之后三条曲线基本重合。

**图 6（关掉 Engram 后保留的性能，pypdf 从 v1 PDF 抽出的文字，标签与数值按抽取顺序配对，未逐像素核对）**：
C3 93%，RACE-Middle 89%，RACE-High 84%，DROP 81%，HellaSwag 85%，ARC-Challenge 81%，PIQA 81%，CMMLU 78%，MMLU 75%，MMLU-Pro 72%，
CruxEval 76%，MBPP 68%，HumanEval 58%，BBH 67%，GSM8K 62%，MGSM 44%，MATH 36%，TriviaQA-ZH 44%，PopQA 44%，TriviaQA 29%；
类别标签：Reading Comprehension、Commonsense Reasoning、Knowledge-Intensive Reasoning、Code、Algorithmic Reasoning、Factual Knowledge。
（与正文 “29–44%”“81–93%”“TriviaQA at 29%”“C3 at 93%” 一致。）
**核查更新**：已用 `sensitivity_analysis.svg` 确认标签与柱子的对应。横轴标签的字形号是 ASCII 码，解码后的顺序与上面完全一致；
柱高（以 100% 灰色底柱为满格）依次为 93.6、89.6、84.3、81.8、85.9、81.9、81.1、78.8、75.5、72.3、76.1、68.5、58.2、67.1、62.6、44.9、36.7、44.9、44.6、29.8（%），
六种底色把柱子分成 4+3+3+3+4+3 组，与六个类别一致。图上的百分比标签看来是**截断取整**（如 DROP 81.8% 标 81%、HellaSwag 85.9% 标 85%），不是四舍五入（笔者推断）。
ACL 版附录 G 图 9 的抽取文字给出同一组标签与数值。

**图 5（层扫描）**：正文只给三点（基线 1.808；单模块第 2 层 1.770；第 2+6 层 1.768）。核查时用 `arch_ablation.svg` 反算出其余各点
（y 轴为断轴：上段只画基线 1.808，下段刻度 1.7675–1.7825、步长 0.0025，标签印成 1.768/1.770/1.773/…/1.783）：
- 单模块插入层 1–12：1.7730, **1.7705**, 1.7748, 1.7762, 1.7786, 1.7774, 1.7793, 1.7802, 1.7808, 1.7810, 1.7824, 1.7822。
  第 2 层最好（与正文 1.770 一致）；总体随深度变差，但不严格单调（第 5 层比第 6 层差，第 11 层比第 12 层差）。
- 参考配置（第 2+6 层，绿色菱形与水平线）：1.7678（正文 1.768）。
- 组件消融（右侧标记，按标签解码）：w/o multi branch 1.7743、w/o token compress 1.7735、w/o gating 1.7724、+4-gram 1.7695、w/o short conv 1.7683。
  排序与正文一致：多分支、分词器压缩、门控三项退化最大；去掉卷积只差约 0.0005；加 4-gram 差约 0.0017。
- 这些都是从矢量图反算的数，引用时写“据图 5 读取”。ACL 版已删去此图。

---

## 5. 图表清单与重绘建议

### 5.1 论文原图清单

**Engram（arXiv v2）**

| 图/表 | 标题（原文要点） | 内容 | 本课用法 |
|---|---|---|---|
| 图 1 | “The Engram Architecture.” | 左：Vocab Embedding → Transformer Block → [Engram → Attention → MoE]，每个子层都是残差相加；右：以 “Only Alexander the Great could tame the horse Bucephalus.” 为例，“the Great”→ 2-gram、“Alexander the Great”→ 3-gram，各经 $h$ 个 Hash 头查 2-Gram/3-Gram Embedding → Concat → 两个 Linear（Key、Value）→ 与 Input Hidden 做 Scaled Dot Product → 乘 → Conv | **重绘**（图 F1、F2） |
| 图 2 | “System implementation of Engram.” | (a) 训练：表分片到各 GPU，All-to-All 取活跃行；(b) 推理：表卸载到主机内存，主机异步预取，与前面 Transformer 块的计算重叠 | **重绘**（图 F9） |
| 图 3 | “Sparsity allocation and Engram scaling.” | 左：两个算力点的 U 形；右：无限内存下损失对槽数（对数轴）近似直线，Engram 在 OverEncoding 之下 | **重绘**（图 F5、F6，数据用 §4.9 目测值并标注） |
| 图 4 | “Analysis of representational alignment and convergence speed.” | (a) 逐层 LogitLens KL；(b)(c) CKA 热图与软对齐曲线（白虚线） | 可选重绘 (a)（图 F8 的一部分） |
| 图 5 | “Architecture ablation results.” | 层扫描曲线（第 1–12 层）+ 右侧组件消融标记（w/o gating、w/o multi branch、w/o token compress、+4-gram、w/o short conv） | 只能画三个已知点（§8） |
| 图 6 | “Retained performance under Engram ablation.” | 20 个基准关掉 Engram 后保留的百分比，六类 | **重绘**（图 F8） |
| 图 7 | “Visualization of the gating mechanism of Engram.” | 英文、中文句子上 $\alpha_t$ 的热图 | 示意性复刻（不用原句，自造例句并标“示意”） |
| 图 8 | “Last 10k pre-training benchmark curve.”（附录 B） | 训练最后 1 万步的基准曲线 | 不重绘 |
| 表 1–6 | 主结果、长上下文、实体解析例（转引 Ghandeharioun et al.）、吞吐、超参数、分词器压缩 top-5 | — | 表 1、2、4、5、6 的数字已录入 §3 |

**DeepSeek-OCR**

| 图 | 标题要点 | 本课用法 |
|---|---|---|
| 图 1 | (a) Fox 上的压缩比-精度；(b) OmniDocBench 性能 vs token 数 | **重绘**（图 F12，用表 2 数据；(b) 用表 3 数据画散点） |
| 图 2 | 三类常见视觉编码器（Vary 双塔、InternVL 切块、Qwen2-VL NaViT）的缺陷 | 可选 |
| 图 3 | DeepSeek-OCR 架构：SAM（窗口注意力）→ 16× 压缩 → CLIP（全局注意力）→ DeepSeek-3B-MoE | **重绘**（图 F10） |
| 图 4 | 多分辨率模式 | **重绘**（图 F11） |
| 图 5–6 | OCR 1.0 精标注格式；图表→HTML、几何→字典 | 不重绘 |
| 图 7–12 | 深度解析、多语言、通用视觉的定性例子 | 不重绘 |
| 图 13 | 遗忘机制：越久远的历史渲染得越小越模糊 | **重绘**（图 F13） |

**DeepSeek-OCR 2**

| 图 | 标题要点 | 本课用法 |
|---|---|---|
| 图 1 | 用 LLM 式结构替换 CLIP；视觉 token 双向、查询因果 | 并入图 F14 |
| 图 2 | DETR 解码器与 BLIP-2 Q-former 的并行查询 | 不重绘 |
| 图 3 | DeepSeek-OCR 2 架构 | 并入图 F14 |
| 图 4 | token 计数：0–6 个局部视图，256–1120 | 并入图 F11 |
| 图 5 | DeepEncoder V2 的注意力掩码 | **重绘**（图 F14） |

**DeepSeek-V4.1**：图 3（整体架构）标出 Engram 在第 1、14 层（0 起）的位置——只作为图 F1 的第二个面板引用，完整重绘留给第 19 课。

### 5.2 本课建议的 15 张图（每张给出画图所需的全部数值）

**F1. Engram 在骨干中的位置（重绘 Engram 图 1 左 + V4.1 对照）**
- 面板 A（Engram-27B）：30 个块竖排；第 2、15 块（论文编号）里在 Attention 之前插入 Engram 子层；每块内部顺序 Engram → Attention → MoE，均为残差相加；
  标注 “$d=2560$，MLA 32 头，mHC ×4，MoE 2 共享 + 55 路由选 6”。
- 面板 B（V4.1-Flash）：40 层（20 编码器 + 20 解码器），Engram 在第 1、14 层（0 起，即第 2、15 层），标注 “196B，{2,3,4}-gram，8 头，无卷积”。
- 颜色：Engram/记忆用 `--fig-amber`；MoE 路由专家 `--fig-blue`、共享专家 `--fig-violet`。

**F2. 从 token 到记忆向量：分词器压缩 + 多头哈希（重绘图 1 右，改成可算的例子）**
- 输入：“Alexander the Great”三个 token → 压缩后 ID $x'_{t-2},x'_{t-1},x'_t$（示意：大小写、前导空格、重音被并掉；附录表 6 的 top-5 合并：' ' 合并 163 个原 token、'a' 54、'o' 40、'e' 35、'i' 30）。
- 2-gram (the, Great) 和 3-gram (Alexander, the, Great) 各算一个 mix：$\text{mix}=x'_t c_0\oplus x'_{t-1}c_1(\oplus x'_{t-2}c_2)$。
- 每个 mix 对 8 个不同素数取模（27B：2-gram 头 0–7 为 2,262,409 / 2,262,413 / 2,262,437 / 2,262,443 / 2,262,451 / 2,262,457 / 2,262,461 / 2,262,467；
  3-gram 头为 2,262,499 / 2,262,521 / 2,262,529 / 2,262,541 / 2,262,563 / 2,262,569 / 2,262,593 / 2,262,619——这是第一个 Engram 层；第二层接着往上找）。
- 取回 16 行，每行 80 维，拼成 $\mathbf e_t\in\R^{1280}$。V4.1 版本：24 行 × 256 维 = 6144 维。
- 颜色：哈希/索引 `--fig-violet`，表 `--fig-amber`，拼接后的向量 `--fig-orange`。

**F3. 上下文门控计算图 + 带符号平方根曲线**
- 左：$\mathbf h_t$（`--fig-blue`）→ RMSNorm；$\mathbf e_t\to W_K\to$ RMSNorm；点积 $/\sqrt d$ → [带符号平方根] → σ → $\alpha_t$；$\alpha_t\cdot W_V\mathbf e_t$ → (+卷积) → 加回残差。
- 右：曲线 $\sigma(s)$ 与 $\sigma(\operatorname{sign}(s)\sqrt{|s|})$，$s\in[-16,16]$；标出 §4.4 的 9 个点（如 $s=4$：0.9820 vs 0.8808），
  并标出 $d=2560$ 时 $|s|\le50.6$、平方根后 $\le7.11$ 的竖线。图注写明“平方根一步来自开源代码，论文式 (4) 没有”。

**F4. 多头哈希的碰撞几何（计算图）**
- 横轴：负载 $U/p$，对数 0.01–100；纵轴：概率（对数）。三条线：一个头共享槽 $1-e^{-U/p}$；8 头全被占 $(1-e^{-U/p})^8$；
  与**某个特定**别的 n-gram 在 ≥2 个头相撞 $\binom82/p^2$ 乘以 $U$（期望伙伴数），分别画 $p=2{,}262{,}409$ 和 $16{,}000{,}057$。
- 关键点（§4.7）：$p=2.26\text{M}$、$U=10^7$：0.988 / 0.908；$U=10^{10}$ 时期望 ≥2 头伙伴 0.055。旁注“三素数之积 1.158e19 > 2^63 ⇒ 最多两头同撞”。

**F5. 稀疏分配 U 形曲线（重绘图 3 左）**
- 数据：§4.9 两条序列（用 SVG 反算值）；只有 (100%, 1.7248) 与 (80%, 1.7109) 为正文数字，用实心标记，其余空心并注“据图读取”。
- 叠加二次拟合（顶点 0.739 / 0.724）；阴影标出论文所说 ρ≈75%–80%；在 ρ=40% 处标注“与纯 MoE 持平（46 / 43 个专家）”。
- 双纵轴：6e20 用左轴 1.708–1.746，2e20 用右轴 1.814–1.830。

**F6. 无限内存：损失对槽数（重绘图 3 右）**
- 横轴对数 2e5–2e7（同时标参数量：×1280 → 0.33B–12.8B）；Engram 8 点、OverEncoding 4 点、纯 MoE 1 点（§4.9）；
  Engram 拟合线斜率 −0.0284 / 十倍。图注写“对数线性，不是严格幂律”。

**F7. 参数账本堆叠柱（计算图）**
- 四根柱：Dense-4B、MoE-27B、Engram-27B、Engram-40B。分段：激活 3.8B（`--fig-blue`）；不激活的路由专家 22.58B / 16.76B / 16.76B（`--fig-muted`）；
  Engram 表 0 / 0 / 5.79B / 18.53B（`--fig-amber`）；token 嵌入 0.33B（`--fig-grid`）。总高 4.13 / 26.71 / 26.69 / 39.43（§4.2）。
- 顶部标注 ρ：1 → 0.743。第五根柱（另一尺度或插图）：V4.1 = 552B 骨干 + 196.6B Engram。

**F8. Engram 的收益与“关掉它会怎样”（两联条形图）**
- 左：表 1 差值（Engram-27B − MoE-27B），27 项，按“知识 / 推理 / 阅读 / 代码数学”分组（§4.1）；突出 CCPM +7.5、BBH +5.0、C-Eval +4.7、CMMLU +4.0、ARC-C +3.7，
  以及 PopQA +0.2、WinoGrande +0.2。
- 右：图 6 的保留百分比（§4.9 的 20 个值），同样六类着色；横线标 100%。
- 图注提醒：右图是事后消融（训练-推理不一致），不能读作“知识有 x% 存在 Engram 里”。

**F9. 训练与推理的系统图 + 吞吐**
- 上：训练，表按行分片到 $G$ 张卡，前向 All-to-All 收集活跃行，反向把梯度发回所属卡（V4.1：专门的 engram 并行组，梯度在骨干反传后回送）。
- 下：推理时间线：token ID 已知 → 立刻算哈希 → 主机内存（DRAM）经 PCIe/RDMA 异步取行 ‖ GPU 在算第 1 个块 → 第 2 块用上 Engram 输出。
- 右下小表：4B 9,031.62 → 8,858.28 tok/s（−1.92%）；8B 6,315.52 → 6,140.02（−2.78%）；标“稠密骨干，nano-vLLM，H800，100B 表全在主机内存”。
- 旁注每 token 流量：取 32 行（27B 配置）或 48 行（V4.1），与表大小无关。

**F10. DeepEncoder 数据流（重绘 OCR 图 3，加上代码里的拼接）**
- 1024×1024 图 → patch 16 → 64×64 = 4096 token → SAM-base（80M，12 层，窗口注意力为主，第 2/5/8/11 层全局）→ 两层 3×3 stride-2 卷积（256→512→1024）→ 16×16 = 256 token
  → CLIP-large（300M，24 层全局注意力，去掉 patch embedding）→ 与 SAM 特征拼接（1024 ⊕ 1024 = 2048）→ 线性投影 → 1280 → DeepSeek-3B-MoE（12 层，64 选 6 + 2 共享，≈570M 激活）。
- 标注两个注意力矩阵大小：4096² 只在窗口内算；256² 全局；比值 256。

**F11. 分辨率模式与 token 数（计算柱状图）**
- 柱：Tiny 64、Small 100、Base 256、Large 400、Gundam（n=2…9：456…1156）、Gundam-M（n=2：912，n=9：2704）；OCR 2：256…1120。
- 每根柱上叠一小段“代码实际占位多出的换行/分隔 token”：+9、+11、+17、+21。
- 次坐标：对一页 1,000 文本 token 的文档，各模式压缩比 15.6×、10×、3.9×、2.5×。

**F12. 压缩比-精度曲线（重绘 OCR 图 1(a)，用表 2）**
- 横轴压缩比 6–20；两条折线：64 token（10.5, 96.5）(11.8, 93.8)(13.2, 83.8)(15.1, 85.9)(16.5, 79.3)(17.7, 76.4)(19.7, 59.1)；
  100 token（6.7, 98.5)(7.5, 97.3)(8.5, 96.8)(9.7, 96.8)(10.6, 91.5)(11.3, 89.8)(12.6, 87.1)。
- 点的面积 ∝ 页数（7, 28, 28, 14, 11, 8, 4）；竖虚线 10×；在 (19.7, 59.1) 旁注 “n=4 页”。

**F13. 光学遗忘示意（重绘 OCR 图 13，用可算的数字）**
- 横轴：对话轮次（越往左越久远）；每轮 1,000 文本 token。最近 2 轮保留文本（2,000 token），更早的轮次依次渲染为 Large 400 → Base 256 → Small 100 → Tiny 64 token。
- 下方标注各档压缩比 2.5×、3.9×、10×、15.6× 以及表 2 对应的大致精度区间（≤10×：约 97%；15×：约 86%（64 token 档 15.1×=85.9%）；约 20×：约 60%）。
- 图注：这是作者提出的设想（§5 Discussion），论文没有做过这种多轮实验。

**F14. DeepEncoder V2 的注意力掩码（重绘 OCR2 图 5）**
- 用 $m=n=4$ 的 8×8 小矩阵画：左上 4×4 全 1（视觉 token 双向），右上 4×4 全 0，左下 4×4 全 1（查询看全部视觉 token），右下 4×4 下三角（查询之间因果）。
- 旁注真实规模：全局视图 $m=n=256$ ⇒ 512×512 掩码；每个局部视图 $m=n=144$ ⇒ 288×288；只有后一半（查询）的输出送给解码器。

**F15. DeepSeek 多模态与记忆谱系时间线**
- 时间轴：DeepSeek-VL（2024-03，576 token/图）→ Janus（2024-10）→ DeepSeek-VL2（2024-12，每块 196 token）→ Janus-Pro（2025-01）→ DeepSeek-OCR（2025-10-21）
  → Engram（2026-01-12）→ DeepSeek-OCR 2（2026-01-28）→ DeepSeek-V4（2026-04，无 Engram）→ Engram ACL 版（2026-07，称 DSE）→ DeepSeek-V4.1-Flash（2026-09-17，196B Engram，DeepSeek-ViT）。
- 两条泳道：“条件记忆（参数）”与“视觉压缩上下文（输入）”，强调两者在 V4.1 里只有前者被采用。

---

## 6. 算例与练习素材（附答案）

**练习 1（计算）V4.1 的 Engram 有多大？**
已知：2 个模块，每模块阶 {2,3,4}、每阶 8 个头、每头表约 1.6×10⁷ 行、每行 256 维，FP8 存储且每 32 维一个 1 字节缩放。
求参数量与存储量。
- 答：行数 $2\times3\times8\times1.6\times10^7=7.68\times10^8$；参数 $7.68\times10^8\times256=1.966\times10^{11}$（196.6B）。
  精确值（素数表）：768,022,850 行 → 196,613,849,600 参数，与 `config.json` 的 [384,006,168, 384,016,682] 一致。
  存储 $768{,}022{,}850\times(256+8)=202.8$ GB（188.8 GiB）；BF16 则 393.2 GB。

**练习 2（计算）每个 token 要从主机内存搬多少字节？和表多大有没有关系？**
- 答：V4.1 每 token 取 $2\times3\times8=48$ 行，每行 264 B（含缩放）⇒ 12,672 B；Engram-27B 配置取 $2\times2\times8=32$ 行 × 80 维，若按 BF16 为 5,120 B。
  与表的行数无关：Engram-27B 与 Engram-40B 的表相差 3 倍多（5.79B vs 18.53B），每 token 都是 32 行；每 token 取几行只由“层数 × 阶数 × 头数”决定。这就是论文 “the effective communication volume per step scales with the number of activated slots rather than the total embedding table size” 的算术。
  延伸：按表 4 的 9,031.62 tok/s，V4.1 式的取数约 114 MB/s。

**练习 3（计算）从专家数推出分配比 ρ。**
MoE-27B：72 个路由专家激活 6 个；Engram-27B：55 个路由专家激活 6 个，总参数不变。求 ρ。
- 答：不激活的专家预算从 $72-6=66$ 份变成 $55-6=49$ 份，其余给 Engram，所以 $\rho=49/66=0.742$。
  用精确参数（专家 $3\times2560\times1536$、29 个 MoE 层、Engram 5,792,338,560）得 0.7432，与论文 74.3% 相符。
  追问：“最优 ρ≈75%–80%”意味着给 Engram 多少？答：20%–25% 的**不激活**参数预算，不是总参数的 20%–25%。

**练习 4（推导）为什么 27B 配置里两个不同的三元组最多只能在 2 个哈希头上同时撞车？**
- 答：同一阶的 8 个头共用同一个整数 $\text{mix}\in[0,2^{63})$，第 $k$ 个头的索引是 $\text{mix}\bmod p_k$。若 $\text{mix}\neq\text{mix}'$ 在头 $i,j,l$ 同时相撞，
  则 $p_ip_jp_l\mid(\text{mix}-\text{mix}')$。最小的三个素数之积 $2{,}262{,}409\times2{,}262{,}413\times2{,}262{,}437=1.158\times10^{19}>2^{63}=9.22\times10^{18}>|\text{mix}-\text{mix}'|$，
  所以不可能，除非 $\text{mix}=\text{mix}'$。（V4.1：三素数之积 $4.10\times10^{21}$，结论同。）
  注意前提：这里的“27B 配置”指按演示代码的哈希结构与素数布局（从 2,262,400 起找素数）来实例化；Engram 论文只写 “multiplicative-XOR hash”，
  27B 实验实际用的哈希代码没有公开。V4.1 的 `inference/engram.py` 与演示代码结构相同（逐头 `rolling % primes` 再加 offset），所以 V4.1 的结论有代码依据。

**练习 5（计算）哈希负载。**假设训练数据里有 $10^8$ 个不同的（压缩后）三元组，V4.1 的一个头表大小 16,000,057。
求负载、一个头里“有人同槽”的概率、以及与某个特定三元组在 ≥2 个头相撞的期望伙伴数。
- 答：负载 6.25；$1-e^{-6.25}\approx0.998$；期望伙伴 $(10^8-1)\times\binom82/16{,}000{,}057^2\approx1.1\times10^{-5}$。
  结论：单个槽几乎都被共享，但 8 个槽的“组合”几乎唯一——所以拼接后的 $\mathbf e_t$ 仍能区分不同的 n-gram，而门控负责压掉“撞错”的噪声。

**练习 6（计算）门控的取值范围。**V4.1 的 $d=5120$，忽略可学习缩放。求 $s_t$ 的范围、带符号平方根后的范围、$\alpha_t$ 的范围；再比较 $s=4$ 时有无平方根的 $\alpha$。
- 答：RMSNorm 后两向量范数都是 $\sqrt d$，$|s|\le\sqrt{5120}=71.55$；平方根后 $\le8.46$；$\alpha\in[0.00021,0.99979]$。
  $s=4$：$\sigma(4)=0.982$，$\sigma(2)=0.881$。

**练习 7（计算）Engram-27B 的短卷积能“看到”多远？**核宽 4、膨胀 = 最大阶 3，输入是三元组嵌入。
- 答：卷积取 $t,t-3,t-6,t-9$ 四个位置，感受野 $(4-1)\times3+1=10$；每个位置本身又汇总了 3 个 token，所以总共覆盖 $t-11\dots t$ 共 12 个 token。V4.1 去掉了卷积，只剩 4 个 token（4-gram）。

**练习 8（计算）DeepSeek-OCR 的 token 数。**
(a) 1280×1280 的 Large 模式输出多少视觉 token？代码里实际占多少个位置？
(b) 一张 A4 纵向扫描（1654×2339 像素）用 Base 模式，有效 token 按式 (1) 是多少？
(c) Gundam 模式最多多少 token？
- 答：(a) $1280/16=80$，$80^2=6400$ patch，$/16=400$；网格 20×20，每行加 1 个换行、最后加 1 个分隔：$21\times20+1=421$。
  (b) $\lceil256\times1654/2339\rceil=182$（代码用 `int` 得 181）；恰好等于表 3 Base 行括号里的平均有效 token 182。
  (c) 瓦片数上限 9：$9\times100+256=1156$。

**练习 9（计算）从表 2 读压缩比。**64 个视觉 token、文本 1200–1300 token 的一档，压缩比 19.7×、精度 59.1%。
(a) 这一档平均多少文本 token？(b) “<10× 时约 97%”按页加权是多少？(c) 为什么 20× 那一点要谨慎引用？
- 答：(a) $19.7\times64\approx1261$（100-token 列 $12.6\times100=1260$，一致）。
  (b) <10× 的格只有 100-token 列前四档：$(98.5\times7+97.3\times28+96.8\times28+96.8\times14)/77=97.14\%$。
  (c) 只有 4 页；而且 64-token 列并不单调（13.2× 时 83.8%，15.1× 时反而 85.9%），小样本噪声明显。

**练习 10（设计计算）光学遗忘的 KV 账。**一段对话已进行 10 轮，每轮 1,000 文本 token。方案 A：全部保留为文本；方案 B：最近 2 轮保留文本，更早 8 轮每轮渲染为一张 Small（100 token）图片。
按 V4.1-Flash 的全局 KV 890 字节/token（`dsv41` 摘要）计算两方案的 KV 占用与压缩倍数，并说出方案 B 的风险。
- 答：A：10,000 token → 8.90 MB；B：2,000 + 8×100 = 2,800 token → 2.49 MB；3.57 倍。风险：每张图压缩比 10×，按表 2 精度约 90%–97%（1000–1100 档 91.5%）；
  而且这是 OCR 精度，不是“理解与推理”的保真度（O-C15；第三方 X-C1 报告去掉语言先验后大幅下降）。另注：假设视觉 token 在 LLM 里与文本 token 的 KV 成本相同（笔者假设）。

**练习 11（概念）Engram 省了 KV cache 吗？**
- 答：没有。Engram 是**参数化**记忆：用最近几个 token 的 ID 作键去查一张训练出来的表，查到的向量加到残差流里；它不存储、也不压缩上下文的 K/V。
  它对长上下文的帮助是间接的：把局部固定搭配交给查表，让注意力“腾出容量”（E-C22、E-C23），论文在等损失设置下测到 MQ-NIAH 84.2→97.0。
  真正“把上下文压小”的是 DeepSeek-OCR 的设想（把历史渲染成图片）以及第 18–19 课的 CSA/HCA/CSA2。

**练习 12（概念）为什么 Engram 能卸载到主机内存而 MoE 专家不方便？**
- 答：Engram 的行号只取决于输入 token ID（确定性寻址），在第 $\ell$ 层开始计算之前就能算出，因此可以提前预取并与前面层的计算重叠；
  MoE 选哪个专家取决于该层的隐藏状态，只有算到那一层才知道（E-C10、§2.5 原文 “unlike MoE which relies on runtime hidden states for dynamic routing”）。
  放置的代价：越往深处放，预取窗口越长，但建模上越早越好（第 2 层最佳，E-C25），所以是折中。

---

## 7. 勘误与常见误解

**E1. 名字漂移：Engram = DSE。** ACL 2026 正式版把模块改称 “Deep Sparse Embedding (DSE)”，全文不出现 “Engram”；arXiv v1/v2、GitHub 仓库和 V4.1 报告都叫 Engram。
作者名单也不同：arXiv v2 21 人（Rui Tian 为共同一作），ACL 版与 arXiv v1 同为 14 人。写课时用 Engram，脚注说明 ACL 版名称。

**E2. “MMLU +3.4” 与表不符。** 摘要与 §1 写 MMLU +3.4，§4.2 写 +3.0；表 1：57.4 → 60.4 = +3.0。+3.4 是 MMLU-Redux（60.6 → 64.0）。ACL 版引言同样写 +3.4、§4.2 写 +3.0。
课文应引 +3.0（MMLU）或 +3.4（MMLU-Redux），不要混。

**E3. ρ 是给 MoE 的份额，不是给 Engram 的份额；最优点也不是“定律”。** 常见误读是“把 75%–80% 的参数给 Engram”。正确：ρ 是**不激活参数预算**里给 MoE 专家的比例，
最优 ρ≈75%–80% ⇒ Engram 占这部分预算的 20%–25%；换成总参数（Engram-27B）是 5.79B / 26.7B ≈ 22%，巧合地接近。
论文只在两个算力点、固定 $P_{tot}/P_{act}\approx10$ 下测过（“under fixed sparsity”），“U 形 scaling law” 是经验曲线；两个算力点的最低观测点都在 ρ≈85%（由图 3 的矢量坐标反算：6e20 为 1.7095，2e20 为 1.8155，§4.9），二次拟合顶点约 0.72–0.74。

**E4. “strict power law (linear in log-space)” 用词不准。** 损失对 $\log M$ 呈直线是**对数线性**关系，不是幂律（幂律是 $\log L$ 对 $\log M$ 呈直线）。
图 3 标题说 “log-linear trend”，ACL 版改成 “roughly log-linearly”。在 2.6×10⁵–1.0×10⁷ 这么窄、损失只变 0.045 的范围内两者也无法区分（§4.9 对数-对数斜率 −0.007）。

**E5. 论文 vs 代码：门控里有带符号平方根。** 式 (4)(6) 是 $\sigma(\cdot)$ 直接作用于缩放点积；GitHub 演示代码与 V4.1 推理代码都先做 $\operatorname{sign}(s)\sqrt{|s|}$ 再 sigmoid，
V4.1 代码注释 “matching the training kernel”。课文给式 (4) 时应加脚注。

**E6. 论文 vs 代码：分词器压缩不只是 NFKC + 小写。** 代码链为 NFKC → NFD → StripAccents → Lowercase → 空白折叠（并用私用区字符保护单个空格 token）。
V4.1 `config.json` 给出压缩后词表 99,092（相对 129,280 减少 23.35%），论文说 128k 词表减少 23.43%（保留 76.57%，约 98,990）——约 100 个 ID 的差没有来源解释。

**E7. 层号的 0 起 / 1 起。** Engram 论文写 “layers 2 and 15”；演示代码 `layer_ids=[1, 15]`（0 起，即第 2、16 层）；V4.1 写 “layers 1 and 14 (zero-indexed)”（即第 2、15 层）。
论文的 “2 and 15” 与 V4.1 的 [1,14] 一致（若论文是 1 起计数，笔者推测），演示代码的 15 与二者都差一层。课文统一说“第 2 层与第 15 层（从 1 数）”。

**E8. “Engram-27B”不是 27B 的记忆表。** 27B 指总参数 26.7B（含 token 嵌入），其中 Engram 表只有 5.7B（按素数布局复算 5.79B）；Engram-40B 的表 18.5B。
同理，V4.1 的 “552B” 是骨干参数，不含 196B Engram；全部参数约 748.6B。

**E9. Engram 不是上下文记忆，也不省 KV cache。** 它是“以局部 n-gram 为键的参数化查表”，每个位置只需要最近 $N_{gram}-1$ 个 token 的 ID（V4.1 代码把压缩后的 ID 缓存下来跨 prefill/decode 使用）。
课程地图把第 17 课概括为“不是所有上下文都需要以完整 KV 的形式记住”——对 DeepSeek-OCR 成立；对 Engram 应改说“不是所有知识都要靠逐层计算重建”。

**E10. “<3% 开销”的适用范围。** 这是 nano-vLLM 原型、**稠密** 4B/8B 骨干、100B 表整个放主机内存、全部经 PCIe 取数的实测；没有测 MoE 骨干（作者说是为避开专家并行通信的干扰）。
多级缓存（HBM/DRAM/SSD，按 Zipf 分层）是设想，论文没有实测。V4.1 的 RL rollout 反而把表常驻 GPU（C-V41-6）。

**E11. “等价于加深网络”是解读，不是定理。** 证据是 LogitLens KL 更低、CKA 对角线上移（Engram-27B 第 5 层 ≈ MoE 第 12 层，一个例子）。“functionally equivalent” 应写成“表征上类似更深的层”。
另：图 4(a) 第 0 层 Engram-27B 的 KL 反而略高（目测 10.35 vs 10.0）。

**E12. Engram-40B 并非处处更好。** 相对 Engram-27B，C3、HumanEval、MBPP、MATH 四项下降，MBPP 甚至低于 MoE-27B；作者归因于欠训练（E-C21），这是推测。

**E13. “关掉 Engram 事实类只剩 29%–44%”不等于“知识的 56%–71% 存在 Engram 里”。** 这是训练后硬关掉一个一直在用的模块，作者自己说会引入训练-推理不一致（E-C27）。

**E14. Janus 的语言模型是 1.3B 还是 1.5B？** Janus 论文 §4.1：“DeepSeek-LLM (1.3B)”；Janus-Pro §2.4：“The previous version of Janus … using a 1.5B LLM”，并说表 1 列出 “the hyperparameters of both the 1.5B and 7B LLMs”（列名为 Janus-Pro-1B / Janus-Pro-7B）。
同一底座两个数，可能是是否计入嵌入的差别（笔者推测，未核实）。

**E15. DeepSeek-OCR 的 “97% / 60%” 是什么。** 是 Fox 英文部分、600–1300 文本 token、100 页、只用 Tiny/Small、提示 “Free OCR.” 下的 OCR 精度；
“<10× 约 97%” 对应 100-token 列四档（按页加权 97.14%），“20× 约 60%” 是 4 页的 59.1%。论文自己说输出格式与 Fox 不完全匹配，“actual performance would be somewhat higher”。
它衡量的是“能不能逐字读回来”，不是“能不能在压缩后的上下文上推理”——作者承认 “OCR alone is insufficient to fully validate true context optical compression”。
第三方（X-C1）报告去掉语言先验后从约 90% 跌到约 20%，并在约 1 万文本 token 处崩溃。“视觉 token 能把上下文压缩 10 倍而近乎无损”不能当作已证实的结论讲。

**E16. 压缩比分母没数换行与分隔 token。** 论文的 64/100/256/400 是编码器输出的视觉 token；开源代码送进 LLM 时每行多一个 `image_newline`、最后多一个 `view_seperator`：
Tiny 实际 73 个位置、Small 111。按实际占位算，表 2 的“10.5×”约为 9.2×（`672/73`）。另：代码的 `test_compress` 用**输出**文本的 token 数做分子。

**E17. 解码器激活参数：570M 还是 500M？** OCR 论文：“DeepSeek3B-MoE-A570M”“about 570M activated parameters”；OCR 2 论文：“about 500M active parameters”“DeepSeek-3B-A500M”；
VL2：“0.57B”。按 config 重建：非嵌入激活 407.7M，加 LM 头 573.2M（笔者推导）——570M 应是含 LM 头的口径；“500M”是 OCR 2 的约数。

**E18. DeepEncoder 不只是“SAM → 压缩 → CLIP”串联。** 开源代码把 CLIP 输出（去 CLS）与 SAM 压缩特征在通道维**拼接**（1024 ⊕ 1024 = 2048）再线性投影到 1280，
即 SAM 特征绕过 CLIP 直达解码器；论文正文与图 3 没画这条旁路。DeepSeek-VL 的混合编码器同样是拼接 SAM-B 与 SigLIP-L 特征。

**E19. OCR 2 的 “3.73% improvement” 是百分点。** OmniDocBench v1.5 总分 87.36 → 91.09，是绝对差 3.73 分，相对提升 4.3%。表 1 表头把 “R-order Edit” 标成 ↑（越高越好），
但它是编辑距离，越低越好（差值行写 ↓0.028）。

**E20. Gundam-M 没有开源；Gundam 的 “795” 是平均值。** README 的支持模式只到 Gundam；表 3 的 795、1853 是 OmniDocBench 上每页平均 token；Base/Large 括号里的 182/285 是平均有效 token。
表 4 模式名印作 “Guandam-M”；其中 Gundam 在 Financial Report 上 0.289，远差于 Base 的 0.027，论文没解释。

**E21. “DeepSeek-V4 用了 Engram” 是错的；“V4.1 用了 DeepSeek-OCR 的编码器” 也是错的。** V4 只在未来工作里提到 “more sparse embedding modules (Cheng et al., 2026)”；
V4.1 集成了 Engram（196B），视觉部分是从头训练的 DeepSeek-ViT，不引用 DeepSeek-OCR。

**E22. “多头哈希 = K 个独立哈希函数” 不准确。** 实现是同一个混合整数对 K 个不同素数取模（多模数），头与头之间的碰撞近似独立（CRT），但在 64 位整数范围内至多 2 头同时碰撞（§4.7）。
另外，表 5 的 “Engram Vocab Size 2,262,400” 本身不是素数，它是每个 (阶, 头) 找素数的起点。

**E23. 引用长上下文数字要写明哪一行。** “MQ-NIAH 84.2 → 97.0”在等损失（46k）与等 FLOPs（50k）两行相同；VT 在 46k 是 87.2、在 50k 是 89.0（基线 77.0）。论文 §1 用 89.0，§5.2 用 87.2。

**E24. V4.1 的 Engram 放置理由与原论文不同。** 原论文：建模要早、延迟隐藏要深，折中；V4.1：“to balance memory usage across training pipeline stages”。不要把原论文的理由安到 V4.1 上。

**E25. 精度 vs 准确率的措辞。** OCR 摘要前半句用 “decoding (OCR) precision of 97%”，后半句用 “OCR accuracy … about 60%”，表 2 列名是 “Precision”；论文没有给出 precision 的计算定义（§8）。

---

## 8. 未能核实 / UNVERIFIED

1. **Engram-27B 的专家中间维**：论文未给；本笔记用 VL2 27B 语言模型 `config.json` 的 1536 复算出 ρ=74.3% 与总参 26.7B，均吻合，但“Engram 的 MoE-27B 就是 VL2 的 27B 骨干”仍是推测。
2. **5.7B vs 5.79B**：按演示代码的素数布局约定（参数 = 行数 × 每头维度），Engram-27B 为 5.79B，论文写 5.7B（40B 版 18.53B 与 18.5B 吻合）。是截断还是另有口径，未能确认。
3. **训练语料中不同 n-gram 的数量**：论文未给，§4.7 与练习 5 的 $U$ 都是假设值。
4. **图 5 各层数值与组件消融标记的具体损失**：核查时已从 arXiv HTML 的矢量图 `arch_ablation.svg` 反算（§4.9），三个正文数字都能复现（1.808、1.7705、1.7678）。
   仍未解决的只是：这些数不是论文正文数字，论文没有给出表格；标记中心按外框中点估计，误差约 ±0.0002。ACL 版已删去此图。
5. **图 3 的数据点**：除 1.7248、1.7109 外都是从图读出的；已用 `scaling_law.svg` 的矢量坐标复核（§4.9），误差约 ±0.0001。仍不是论文数字。
6. **图 6 标签与数值的配对**：已用 `sensitivity_analysis.svg` 确认（§4.9）。剩下的小疑问是图上百分比为何是截断取整，这是笔者推断，论文未说明。
7. **V4.1 的 8B/16B 激活参数与 552B 骨干是否包含 Engram 的 K/V 投影**（每模块 157.3M，笔者推算）：报告未说明。
8. **V4.1 中 Engram 的收益**：V4.1 报告没有给出“去掉 Engram”的消融，无法量化它对 V4.1 的贡献。
9. **压缩后词表 99,092 vs 约 98,990**：需要下载 DeepSeek-V3 / V4.1 分词器实际运行 `build_compressed_token_map` 才能判断，本次未做（未下载分词器文件）。
10. **DeepSeek-OCR 的 “Precision” 定义**：论文未定义计算方式（字符级？编辑距离换算？），表 2 的数值无法独立复现。
11. **PCIe / RDMA 带宽**：论文未给测试平台的主机-GPU 带宽，§4.8 只给了每秒字节数，没有与带宽比较；任何带宽数字都需另找厂商资料核对。
12. **100B 卸载实验中 Engram 层的具体配置**（阶、头、维度）：论文未给，所以无法算出该实验的真实每 token 流量。
13. **OCR 2 卷积输出 896 维是否为了匹配 Qwen2-0.5B 的隐藏维**：论文只说 “to align with the subsequent pipeline”；Qwen2-0.5B 的隐藏维未在本次核对。
14. **ACL 版与 arXiv 的完整 diff**：只对关键数字（表 1、表 2、表 6 吞吐、分配实验、配置表）和措辞做了比对，附录 D–J 的全文未逐句 diff，图未检查。
15. **“Engram”一词的由来**（神经科学中的“记忆痕迹”）：论文没有解释，不要写成论文的说法。
16. **`ocr-crutch` 与 `engram-nine`**：只读了摘要，方法、数据、是否被复现均未核实；两者都是未经评审的预印本。
17. **DeepSeek-OCR 表 4 中 Gundam 在 Financial Report 上的 0.289**：是否排印错误，未能确认。
18. **Engram-27B 训练时的哈希乘子种子**：演示代码与 V4.1 代码都用 `10007 * layer_id`（演示再加 `seed=0`），论文未说明 27B 实验用的种子。
19. **V4 论文 arXiv 编号 2606 而 HTML 页脚日期为 2026-04-26** 的不一致：属于第 18 课范围，此处只记录，未调查。
20. **DeepSeek-OCR 结果是否被第三方复现**：除 `ocr-crutch` 的反驳性测试外，未查到独立复现。
21. **Janus 1.3B vs 1.5B 的原因**（是否计入嵌入）：未核实。

---

## 9. 参考文献卡片草稿

```
[engram] Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
    a: Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai, et al. (2026), preprint (arXiv v2); published at ACL 2026. arXiv:2601.07372v2
    url: https://arxiv.org/abs/2601.07372v2
    finding: 提出 Engram：用压缩后的 2/3-gram 经多头乘法-异或哈希查素数大小的嵌入表，再用隐藏状态做门控加回残差；在固定总参数与激活参数下把约 20%–25% 的不激活预算从 MoE 专家挪给 Engram，27B 模型全面优于等参数等 FLOPs 的 MoE。
    matters: 本课 Engram 部分的全部公式（式 1–9）、表 1/2/4/5/6、U 形分配律与 100B 卸载实验都以它 v2 的编号为准，且只有 arXiv 版含 §6.2 的逐层消融。

[engram-acl] Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
    a: Xin Cheng, Wangding Zeng, Damai Dai, et al. (2026), ACL 2026 (Long Papers), pp. 4968–4990. DOI 10.18653/v1/2026.acl-long.226
    url: https://aclanthology.org/2026.acl-long.226/
    finding: 同一工作的同行评审版本，模块改称 Deep Sparse Embedding (DSE)，数字与 arXiv 一致，补写了 4B 骨干 1.9% 的吞吐开销并把无限内存曲线改述为“近似对数线性”。
    matters: 课文需要同行评审出处时引它，并借它说明版本漂移（改名、删去层扫描、措辞修正）。

[dsv41] DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression
    a: DeepSeek-AI (2026), preprint. arXiv:2609.19969v1
    url: https://arxiv.org/abs/2609.19969
    finding: V4.1-Flash 在 552B 骨干外集成 196B Engram（两模块、{2,3,4}-gram、8 头、约 16M 行/头、第 1 和 14 层（从 0 计）），去掉短卷积，并用动量加 Sinkhorn 平衡的更新代替 Adam 训练嵌入表。
    matters: 它回答了“DeepSeek 旗舰模型到底用没用 Engram”，并提供了 Engram 从论文到生产的具体改动，是本课与第 19 课的接口。

[dsv4] DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
    a: DeepSeek-AI (2026), preprint. arXiv:2606.19348v1
    url: https://arxiv.org/abs/2606.19348
    finding: V4 的架构不含 Engram，只在结论的未来方向里提出要探索“more sparse embedding modules (Cheng et al., 2026)”。
    matters: 用来纠正“V4 已采用 Engram”的误传，并标出 Engram 进入 DeepSeek 旗舰线的时间点是 V4.1。

[ocr] DeepSeek-OCR: Contexts Optical Compression
    a: Haoran Wei, Yaofeng Sun, Yukun Li (2025), preprint. arXiv:2510.18234v1
    url: https://arxiv.org/abs/2510.18234
    finding: DeepEncoder（SAM-base + 16× 卷积 + CLIP-large）配 3B MoE 解码器，在 Fox 英文页上压缩比低于 10× 时 OCR 精度约 97%、接近 20× 时约 60%，并在 OmniDocBench 上以 100 token 超过 GOT-OCR2.0、以不到 800 token 超过 MinerU2.0。
    matters: 本课“视觉压缩上下文”一半的全部规格、token 算术、压缩比-精度曲线和遗忘机制设想都来自它。

[ocr2] DeepSeek-OCR 2: Visual Causal Flow
    a: Haoran Wei, Yaofeng Sun, Yukun Li (2026), preprint. arXiv:2601.20552v1
    url: https://arxiv.org/abs/2601.20552
    finding: 用 Qwen2-0.5B 结构的编码器替换 CLIP，加入与视觉 token 等数量的因果流查询来重排视觉信息，在 256–1120 个视觉 token 下把 OmniDocBench v1.5 总分从 87.36 提到 91.09。
    matters: 说明 DeepSeek-OCR 系列的下一步重点从“压多少”转向“按什么顺序读”，并提供注意力掩码这一可画、可算的例子。

[ocr-crutch] Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR
    a: Yunhao Liang et al. (2026), preprint. arXiv:2601.03714v2
    url: https://arxiv.org/abs/2601.03714
    finding: 用句子级和词级语义扰动去掉语言先验后，DeepSeek-OCR 的表现从约 90% 跌到约 20%，并在约 1 万文本 token 处全面崩溃。
    matters: 给课文提供对“10 倍近乎无损光学压缩”的反方证据，避免把一篇技术报告的结论当成定论。

[engram-nine] A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics
    a: Tao Lin (2026), preprint. arXiv:2601.16531v2
    url: https://arxiv.org/abs/2601.16531
    finding: 给高频 n-gram 换上无碰撞的最小完美哈希，在等参数下并不能稳定降低验证损失，碰撞可能起隐式正则化作用。
    matters: 支撑课文“哈希碰撞不可怕、门控才是关键”的讨论，同时标明这是第三方未评审的结果。

[dsvl] DeepSeek-VL: Towards Real-World Vision-Language Understanding
    a: Haoyu Lu et al. (2024), preprint. arXiv:2403.05525v2
    url: https://arxiv.org/abs/2403.05525
    finding: 用 SAM-B（1024²）与 SigLIP-L（384²）的混合编码器，经两层 stride-2 卷积把 1024² 图像压成 576 个视觉 token。
    matters: 它的 SAM + 卷积压缩器是 DeepEncoder 的直接前身，是谱系时间线的起点。

[dsvl2] DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding
    a: Zhiyu Wu et al. (2024), preprint. arXiv:2412.10302v1
    url: https://arxiv.org/abs/2412.10302
    finding: 动态切块加 2×2 pixel shuffle（每块 196 token），语言部分换成激活 0.57B/2.4B/4.1B 的 DeepSeekMoE 3B/16B/27B。
    matters: 它的 3B MoE 就是 DeepSeek-OCR 的解码器规格，27B 语言模型的配置又为本笔记复算 Engram 的 ρ 提供了专家维度旁证。

[janus] Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation
    a: Chengyue Wu et al. (2024), preprint. arXiv:2410.13848v1
    url: https://arxiv.org/abs/2410.13848
    finding: 理解走 SigLIP 编码、生成走码本 16,384 的 VQ tokenizer，两条视觉通路共用一个自回归 Transformer（DeepSeek-LLM 1.3B）。
    matters: 作为谱系中“统一理解与生成”一支的一行说明，交代 DeepSeek 多模态并非只有 OCR 一条线。

[januspro] Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling
    a: Xiaokang Chen et al. (2025), preprint. arXiv:2501.17811v1
    url: https://arxiv.org/abs/2501.17811
    finding: 改进训练策略、扩充数据并放大到 7B，Janus-Pro-7B 在 MMBench 得 79.2、在 GenEval 得 0.80。
    matters: 谱系时间线上的一个节点，同时提供“1.3B 还是 1.5B”这一版本漂移例子（勘误 E14）。

[gh-engram] deepseek-ai/Engram (GitHub repository; README, engram_demo_v1.py, Engram_paper.pdf)
    a: DeepSeek-AI (2026), artefact
    url: https://github.com/deepseek-ai/Engram
    finding: 演示代码给出乘法-异或哈希、按素数取模的多头表、分词器规范化链和带符号平方根的门控，并声明只是数据流演示。
    matters: 是“论文 vs 代码”差异（门控平方根、规范化链、层号）的一手证据。
    kind: artefact

[hf-v41] deepseek-ai/DeepSeek-V4.1-Flash (Hugging Face: config.json, inference/engram.py, inference/model.py, README.md)
    a: DeepSeek-AI (2026), artefact
    url: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
    finding: 配置给出 engram_layer_ids [1,14]、engram_num_embeddings [384006168, 384016682]、压缩词表 99,092，推理代码显示 FP8 表加 UE8M0 缩放、带符号平方根门控且图像 token 不参与 n-gram。
    matters: 让课文能从配置逐位复算出 196.6B，并确认生产实现与论文描述的差异。
    kind: artefact

[hf-ocr] deepseek-ai/DeepSeek-OCR (Hugging Face: config.json, modeling_deepseekocr.py; GitHub README)
    a: DeepSeek-AI (2025), artefact
    url: https://huggingface.co/deepseek-ai/DeepSeek-OCR
    finding: 代码显示 DeepEncoder 输出为 CLIP 与 SAM 特征拼接后的 2048 维线性投影，并在每行视觉 token 后插入换行 token、全局视图后插入分隔 token。
    matters: 用来修正论文中的 token 计数与“纯串联”描述，并支撑解码器 570M 激活参数的复算。
    kind: artefact

[hf-vl2-config] deepseek-ai/deepseek-vl2 config.json
    a: DeepSeek-AI (2024), artefact
    url: https://huggingface.co/deepseek-ai/deepseek-vl2/raw/main/config.json
    finding: 27B 语言模型为 hidden 2560、72 路由加 2 共享专家、top-6、moe_intermediate_size 1536、首层稠密。
    matters: 为 Engram-27B 的 ρ=74.3% 与 26.7B 总参复算提供专家维度（推断性旁证）。
    kind: artefact
```

---

## 10. 核查记录（2026-09-24，对抗式复核）

> 做法：挑出读者最可能转述的 21 组高风险论断（规格数字、逐符号转录的公式、头条结果、日期、署名），逐条回到一手来源重新核对：
> arXiv LaTeXML 全文（`arxiv.py text` 后本地 grep；Engram 另取 v1 全文与 v2 做 diff）、ACL Anthology 页面与 PDF（pypdf 抽取）、
> Hugging Face `config.json` / `inference/*.py` / `modeling_deepseekocr.py` / 模型卡、GitHub README 与提交记录（API），
> 以及 arXiv HTML 中的矢量图 SVG。所有派生数字在 Python 3.11 里重算。默认怀疑：来源里找不到的，一律放进 §8。

### 10.1 核对过的论断（✓ = 与一手来源一致）

| # | 论断 | 核对来源 | 结果 |
|---|---|---|---|
| K1 | Engram arXiv v1 2026-01-12（14 人）、v2 2026-07-12（21 人，Xin Cheng 与 Rui Tian 共同一作）；v1→v2 只改措辞与引用，数字、表、式不变 | arXiv 提交历史；v1 与 v2 HTML 全文 diff | ✓（补记：v2 在 Memory Network 段加了 CoG，已写入 §0.3） |
| K2 | ACL 版：14 人、pp. 4968–4990、DOI、San Diego 2026 年 7 月；称 DSE、全文无 “Engram”；表 1 Dense 印成 3.8B、附录表 3 为 4.1B；“roughly log-linearly”；4B 1.9%、8B 2.8%；引言 MMLU +3.4、§4.2 +3.0；无 §6.2 层扫描与“ρ≈40% 持平”句 | ACL Anthology 页面、卷页面、PDF 抽取 | ✓（DSE 计数与“保留率”措辞已修正，见 V2、V3） |
| K3 | 摘要 / §1 “MMLU +3.4” 与表 1（57.4→60.4 = +3.0）不符，+3.4 属 MMLU-Redux（60.6→64.0） | arXiv v2 摘要、§1、§4.2、表 1 | ✓；表 1 全部 27 项差值重算，与 §4.1 一致，且 27 项均为正 |
| K4 | ρ 的定义（不激活参数预算里给 MoE 的份额）与式 (7) | §3.1 原文 | ✓ 逐符号一致 |
| K5 | 两个算力点的协议（2e20：5.7B/568M/106 专家；6e20：9.9B/993M/99 专家）；1.7248→1.7109、Δ=0.0139；ρ≈40% 时 46 / 43 个专家 | §3.1 | ✓；46.0 / 43.2 重算吻合 |
| K6 | “strict power law (linear in log-space)” 原句；图 3 标题 “log-linear trend”；槽数 2.58e5→1.0e7、≈13B 参数 | §3.2、图 3 标题 | ✓；每十倍 −0.0284、对数-对数斜率 −0.00699 重算吻合 |
| K7 | 表 1 标题（262B、3.8B、72→55、5.7B、18.5B）、§4.1 配置（第 2、15 层，N=3，8 头，1280 维，Adam，LR×5）、表 5 全部超参数 | 表 1、§4.1、附录 A 表 5 | ✓ |
| K8 | 式 (1)–(6)、(8)、(9) 的转录 | §2.2–§2.4、§6.1.2 | ✓ 逐符号一致 |
| K9 | 长上下文表 2：MQ 84.2 / 97.0（46k）/ 97.0（50k）；VT 77.0 / 87.2 / 89.0；§1 用 89.0、§5.2 用 87.2；82% = 41/50 | 表 2、§1、§5.2 | ✓ |
| K10 | §6.2：12 层 3B MoE（0.56B 激活）、100B token；1.808 / 1.770 / 1.768 | §6.2 | ✓；并从 `arch_ablation.svg` 反算出全部层扫描与消融点（V1） |
| K11 | §6.3：事实类保留 29–44%（TriviaQA 29%），阅读理解 81–93%（C3 93%）；图 6 标签配对 | §6.3、`sensitivity_analysis.svg`、ACL 图 9 | ✓；配对已由 SVG 字形解码确认（V1） |
| K12 | 表 4：9,031.62→8,858.28、6,315.52→6,140.02；H800、nano-vLLM、稠密骨干、512 条序列 | §6.4、表 4 | ✓；1.919% / 2.779% 重算吻合 |
| K13 | 素数表布局：27B 为 5,792,338,560，40B 为 18,534,311,360；V4.1 每模块 384,006,168 / 384,016,682，共 196,613,849,600；ρ=0.7432 | 演示代码 `calculate_vocab_size_across_layers` 的算法（WebFetch 原样转录）+ 本地 Miller–Rabin 重算；V4.1 `config.json` | ✓ 逐位一致 |
| K14 | V4.1：552B 骨干 + 196B Engram；prefill 8B / decode 16B；§2.4.2 配置（{2,3,4}、8 头、每阶 2048、约 16M 素数表、FP8、第 1/14 层、RDMA）；§2.5 弃用 Adam 的理由；§3.1.3；§4.2.2（K=11、τ=1e-3、ε=1e-20、γ=0.18、LR×5）；算法 1 与式 (7)；Cheng 2026b 为 arXiv 版、2026c 为 ACL 版 | 2609.19969v1 全文；模型卡；`config.json` | ✓（C-V41-6 的预取时机措辞已按原文收紧，见 V4） |
| K15 | V4 未用 Engram，只在 §6 未来方向提到 “more sparse embedding modules (Cheng et al., 2026)” | 2606.19348v1 全文 grep | ✓（全文不出现 “Engram” 一词） |
| K16 | 代码：演示版与 V4.1 门控都有带符号平方根（V4.1 注释 “matching the training kernel”）；规范化链 NFKC→NFD→StripAccents→Lowercase→空白折叠；乘法-异或 + 逐头取模；乘子上界用压缩词表大小；演示 `layer_ids=[1, 15]`；README 的“演示版”声明；仓库 init 2026-01-12T09:11Z、最后提交 2026-01-14 | GitHub raw 与 API；HF `inference/engram.py`、`model.py` | ✓ |
| K17 | DeepSeek-OCR：摘要 97% / 约 60%；DeepEncoder ≈380M = 80M + 300M；卷积 k3 s2 p1、256→1024；四种原生模式 64/100/256/400；Gundam n×100+256（n=2–9）、Gundam-M n×256+400；解码器 64 选 6 + 2 共享、约 570M 激活、12 层；式 (1) 为上取整；表 2 全部 14 格与页数；表 3、表 4 摘录行 | 2510.18234v1 全文 | ✓；97.14%（按页加权）重算吻合 |
| K18 | OCR 代码：CLIP（去 CLS）与 SAM 特征拼接后 2048→1280 线性投影；非切块分支的占位 (s+1)·s+1 = 73/111/273/421；代码用 `int()`；`config.json` 各字段；解码器参数重建 2.93B / 407.7M / 573.2M | HF `modeling_deepseekocr.py`、`config.json`；GitHub README | ✓ |
| K19 | OCR 2：k×144+256 ∈ [256,1120]；末层卷积 896 维；Qwen2-0.5B（500M）；“about 500M active”；91.09 vs 87.36（+3.73 个百分点，相对 +4.27%）；R-order 0.085→0.057；表 1 表头 R-orderEdit 印成 ↑；表 2、表 4 数值；训练三阶段参数 | 2601.20552v1 全文 | ✓ |
| K20 | 谱系：VL 576 token、SAM 64×64×256 插值到 96×96、两层 stride-2 卷积、与 SigLIP 拼接成 2048 维；VL2 激活 0.57/2.4/4.1B、27×27→196、表 1 Tiny 与标准版配置；Janus §4.1 “1.3B” vs Janus-Pro §2.4 “1.5B”；Janus-Pro-7B MMBench 79.2、GenEval 0.80；各文作者数与日期 | 各文 arXiv 全文与元数据 | ✓ |
| K21 | 第三方：`ocr-crutch`（10 人）“约 90% 跌到 20%”“约 1 万 token 崩溃”；`engram-nine` “does not consistently improve”“implicit regularization” | arXiv 摘要 | ✓（仍只读了摘要） |

### 10.2 本次修改

| 编号 | 位置 | 原内容 | 改为 | 理由与来源 |
|---|---|---|---|---|
| V1 | §4.9、§8 第 4–6 条、F5、E3 | 图 3 数据点标“目测 ±0.001”；图 5 其余层“未能读出”；图 6 配对“未逐像素核对”；纯 MoE ≈1.8087 | 改用 arXiv v2 HTML 矢量图（`scaling_law.svg`、`arch_ablation.svg`、`sensitivity_analysis.svg`）的坐标反算；补上图 5 全部 12 层与 5 个消融点；确认图 6 配对；纯 MoE 1.8082；注明 6e20 各点横坐标不在整十 ρ 上 | 原目测值基本正确（差 ≤0.0005），但现在有可复现的来源；“最低观测点在 ρ≈85%”由目测升级为坐标确认，两个算力点都是如此 |
| V2 | §0.3 ACL 行 | “DSE” 出现 155 次 | 出现在 155 行（按整词 159 次） | 复算 pypdf 抽取：155 是含 DSE 的行数，不是出现次数 |
| V3 | §0.3 ACL 行 | ACL “改用保留率 76.57% 的说法” | ACL 表 4 标题仍叫 “Overall compression ratio”，数值是 23.43% 的补数；正文两版都说减少 23% | “保留率”是笔者的解读，不是 ACL 原文措辞 |
| V4 | §1.3 C-V41-6 | “每个训练步开始前为整个本地 batch 发起预取” | “在每个流水段开始处理本训练步的微批之前”，附原文短引 | 原文 §3.1.3 说的是流水段层面的时机 |
| V5 | §4.7 表 | V4.1 每头碰撞对数 3.13e4 / 3.13e6 / 3.13e8 | 3.12e4（=31,249.9）/ 3.12e6 / 3.12e8 | $10^{12}/(2\times16{,}000{,}057)=31{,}249.9$，三位有效数字是 3.12 |
| V6 | §4.7 要点、练习 4 | “27B 配置里最多只有 2 个头同撞”写成定论 | 加前提：此结论依赖演示代码 / V4.1 代码的哈希结构；27B 实验代码未公开，论文只写 “multiplicative-XOR” | 邻近论断替换：代码事实被直接安到论文的 27B 实验上 |
| V7 | §0.3 v2 行 | §7 新增项列表 | 补记 CoG 与 Muon 引用改动 | v1/v2 全文 diff |

没有发现头条数字、公式转录或署名上的实质错误；研究笔记原有的勘误（E1–E25）经复核全部成立。

### 10.3 复核后仍存疑（已并入或对应 §8）

- 图 5 组件消融标记与标签的对应，是按“标签基线与标记中心的纵坐标最接近”配的（五个标记都比各自标签基线高约 2.6pt，而标记之间相距 ≥10pt），可信但属推断。
- V4.1 的 552B / 8B / 16B 是否计入 Engram K/V 投影、5.7B vs 5.79B、MoE-27B 专家中间维 1536：本次没有新证据，维持 §8 原判。
- ACL 版与 arXiv v2 只比对了本表列出的条目，附录 D–J 未逐句 diff。

---

## 补充核实（写课时，2026-09-24）

> 写 `lesson-17-engram-ocr.md` 时用到、而上文没有的事实。每条附来源与原文短引；纯算术只写表达式。

**S1. Adam 为每个参数维护两个状态（一阶矩与二阶矩）。**（Kingma & Ba, “Adam: A Method for Stochastic Optimization”, arXiv:1412.6980，算法 1；本次用 pypdf 从 arXiv PDF 抽取）
> "m0← 0 (Initialize 1st moment vector)" / "v0← 0 (Initialize 2nd moment vector)"

> "mt←β1·mt−1 + (1−β1)·gt (Update biased first moment estimate)"；"vt←β2·vt−1 + (1−β2)·g2t (Update biased second raw moment estimate)"

  课文 §5 据此写“196.6B 参数的 Adam 状态是 3,932 亿个状态量，只存动量则减半”（`2*196.6e9`，笔者推算；与 V4.1 “substantially increases the optimizer-state memory footprint” 的定性说法一致，V4.1 报告没有给出具体字节数）。

**S2. UE8M0（`float8_e8m0fnu`）只有 8 位指数、没有符号位和尾数，所以缩放因子只能是 2 的整数次幂。**（PyTorch 2.14 文档 `tensor_attributes.html`，dtype 表）
> "8-bit floating point, S-E-M 0-8-0, from https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf"

  “S-E-M 0-8-0” 即 0 位符号、8 位指数、0 位尾数。OCP MX 规范 PDF 本次直接访问返回 403，未能读原文。

**S3. A4 纵向 200dpi 约为 1654×2339 像素。**（ISO 216：A4 = 210 mm × 297 mm；笔者换算）
`210/25.4*200 = 1653.5`、`297/25.4*200 = 2338.6`。上文 §4.5 与练习 8 的 “A4 纵向 1654×2339” 即此分辨率；课文 §6 据此写“按 200dpi 扫描”。

**S4. 课文里新增的派生数字（均为笔者在 Python 中复算，不是论文数字）**
| 量 | 表达式 | 值 | 用在 |
|---|---|---|---|
| 门控点积的方差（初始化假设：各分量独立、零均值、单位方差） | $\sum_{i=1}^d\E[a_i^2]\E[b_i^2]/d$ | 1 | §3 推导 |
| $\sigma(-50.6)$ | `1/(1+exp(50.6))` | 1.06e-22 | §3 |
| sigmoid 梯度 | $\sigma'(4)$；$\sigma'(2)/4$；$\sigma'(16)$；$\sigma'(4)/8$ | 0.0177；0.0262；1.13e-7；0.0022 | 练习 12 |
| 图 3 左二次拟合系数（6e20，用 §4.9 的 SVG 反算值） | `np.polyfit(rho, L, 2)` | a = 0.1350，b = −0.1991，顶点 0.738 | §4 推导 |
| 对数线性的等效幂律指数 | `0.0284/ln(10)/1.765` | 0.00699 | §4 推导（与 §4.9 直接双对数拟合 −0.00699 一致） |
| Sinkhorn 平衡的 2×2 玩具例 | $\widehat G=[[3,4],[0.6,0.8]]$，先行后列 ℓ2 归一化，乘 $\sqrt2$ | 两步收敛到全 1 | §5 算例 |
| Engram 表与百万 token 会话的 KV | `202.76e9/(1e6*890)` | 227.8 个会话 | §8 算例 |
| 光学遗忘算例（2 轮文本 + 2×Large/Base/Small/Tiny） | `2*1000+2*400+2*256+2*100+2*64`，`×890` | 3,640 token，3.24 MB，省 2.75 倍 | §7、图 17.14 |
| V4.1 三个最小 2-gram 表素数之积 | `16000057*16000079*16000081` | 4.096e21（两素数之积 2.56e14 < 2^63） | 练习 11 |
| 1920×1080 用 Base 的有效 token | `ceil(256*1080/1920)` | 144 | 练习 8 |

**S5. 教学复核时（2026-09-24）新增的派生数字与说法**（均为笔者推算或定义性说明，不是论文数字）

| 量 / 说法 | 表达式或依据 | 值 | 用在 |
|---|---|---|---|
| 一次 decode 步（512 个并发请求）按 V4.1 配置要取的字节 | `512*12672` | 6.49e6 B ≈ 6.5 MB | §5 “预取藏不住时会怎样” |
| prefill 一条 10 万 token 提示要取的字节 | `1e5*12672`；与上一行之比 `1e5/512` | 1.27e9 B；约 195 倍 | 同上 |
| 预取可隐藏的条件 $n_{\text{step}}B_{\text{token}}/W_{\text{link}}\le t_{\text{hide}}$ | 笔者的形式化，论文未给出；没有代入任何带宽数字（论文未报告平台带宽，见 §8 第 11 条） | — | 同上 |
| 代码门控在 $s=4$ 处对 $s$ 的梯度 | `sigmoid'(2)/(2*2)` 对 `sigmoid'(4)` | 0.0262 对 0.0177（约 1.5 倍） | §3 正文（原文误把 $\sigma'(2)$ 直接与 $\sigma'(4)$ 比较，已改为含链式因子的数） |
| Engram-40B 的 ρ（与 Engram-27B 同样保留 49 个不激活专家/层） | `16762798080/(16762798080+18534311360)` | 0.4749 | 练习 3(b) |
| 百万 token 会话按约 10× 光学压缩（设想）后与 Engram 表相等的会话数 | `202.76/(0.89/10)` | 约 2,278 | 图 17.16 |
| 窗口注意力的分数个数 | $n/w$ 个窗口 × $w^2$ = $n\cdot w$ | — | §6 前置知识 |
| 卷积输出边长 | $\lfloor(s+2\cdot1-3)/2\rfloor+1$，`s=64` | 32 | §6 推导 |
| SAM `global_attn_indexes [2,5,8,11]` 是从 0 计的块序号 | SAM 图像编码器在 `for i in range(depth)` 里按 `i in global_attn_indexes` 决定是否用窗口注意力（从 0 计），即 12 层里第 3/6/9/12 层（从 1 数）为全局 | 4 层全局 | §6 正文与图 17.11 标注“从 0 计” |
| SAM-base 原本为图像分割训练、CLIP-large 原本用图文配对（对比学习）训练 | 两个模型的通行定义（Segment Anything；Contrastive Language–Image Pre-training），本次未另开原文核对 | — | §6 前置知识（只作定性说明，不含数字） |

**S6. 数学复核（2026-09-24）改正的派生数字与表述**（纯算术或记号，不涉及新的事实来源）

| 位置 | 原来 | 改为 | 依据 |
|---|---|---|---|
| §4.9、E4 的双对数斜率 | −0.00699 | −0.00698 | `np.polyfit(log S, log L, 1)` 用 §4.9 的 SVG 反算值得 −0.006983；−0.00699 是 `0.0284/ln10/1.765` 的等效指数，两者不是同一个量 |
| 课文 §4 要点 | $\rho=49/66\approx74.3\%$ | 按专家数 74.2%，按参数 74.3% | `49/66 = 0.7424`；74.3% 来自参数比 `16762798080/(16762798080+5792338560) = 0.7432` |
| 课文 §3 方差推导的假设 | “各分量独立” | $\mathbf a,\mathbf b$ 相互独立，分量零均值、同分布、两两不相关 | RMSNorm 固定平方和为 $d$，分量不可能严格独立；推导只用到不相关（方向在球面上均匀分布时严格成立），$\E[a_i^2]=1$ 来自同分布加 $\sum_i a_i^2=d$ |
| 课文 §2 迷你哈希例 | “这里的混合值都小于 64” | 补上前提：规范编号 0–12，$x'c_i\le60$，异或后 $<64$ | 否则“不可能在两个头上同时相撞”没有依据；`max((a*3)^(b*5) for a,b in 0..12) = 63` |
| 图 17.13 圆的大小 | 半径 $3+1.9\sqrt n$（面积不与页数成正比） | 半径 $2.5\sqrt n$ | 图注写“圆面积与该档页数成正比” |
| 课文 §5 prefill 与 decode 的取数之比 | 约 200 倍 | $10^5/512\approx195$ 倍 | 与 S5 表一致 |
| 记号 | 损失写 $L$（与层数 $L_{\text{MoE}},L_{\text{E}}$ 同字母）；专家数写 $E$（与表 $E_{n,k}$ 同字母）；OCR 2 掩码写 $m,n$（$n_v$ 已是视觉 token 数，$n$ 已是 N-gram 阶） | 损失 $\mathcal L$（style-guide §2）；路由专家数 $N_r$；掩码 $n_v,n_q$ | style-guide §2 与本课记号说明 |

**S7. 第二次教学复核（2026-09-24）新增的定义与推导**

- **长上下文两项任务的定义**（课文 §4 的浮窗）。RULER（Hsieh et al., “RULER: What's the Real Context Size of Your Long-Context Language Models?”, arXiv:2404.06654，HTML 版 §3.1、§3.2，本次用 WebFetch 读取）：
  > Multi-query NIAH: "Multiple 'needles' are inserted into the 'haystack'. All 'needles' with distinct keys need to be retrieved."

  > Variable Tracking: "A variable X₁ is initialized with a value V, followed by a linear chain of variable name binding statements (e.g., X₂=X₁, X₃=X₂,...), which are inserted at various positions of the input."

  Engram 论文的 “Multi-Query NIAH / VT” 沿用这两个任务名（E-C22）；课文只写定性定义，不含数字。
- **V4.1 嵌入表更新的逐步写法**（课文 §5 新增的 `::: deriv`）：逐行转录上文 §2.1 “V4.1 的 Sinkhorn 平衡动量更新”（算法 1 第 2–3、6、7–17、18–20 行），记号按 §2.2 改写（$\beta\to\mu$、$K\to K_{\text{sk}}$、行范数 $\rho_i\to\norm{\widehat G_{t,i,:}}_2$、平均行范数 $\bar\rho\to\bar g$，以免与分配比 $\rho$ 冲突）；$\mu$、$\gamma$ 的数值取 §3.4 表（同 Muon：0.95、0.18）。
- **“行、列均方根同时约为 1”的解释**（笔者推导，纯代数）：$K_{\text{sk}}=11$ 为奇数，末步按行归一，所以 $U^{(K)}$ 每行单位长度、全矩阵平方和为 $m$；$\Delta=\sqrt n\,U^{(K)}$ 每行平方和为 $n$（行 RMS 恰为 1），全矩阵平方和为 $mn$；若前面的列步骤已使各列大小相近，每列平方和约为 $m$，列 RMS 约为 1。$m=n$ 时（课文 2×2 例）两者都精确成立。
- **浮窗里的通用术语**（PCIe、RDMA、All-to-All、隐式正则化）只作定性说明，不含数字与出处性论断。

**S8. 第二次数学复核（2026-09-24）补上的假设与派生数字**（纯算术或推导，不涉及新的事实来源）

| 位置 | 改动 | 依据 |
|---|---|---|
| 课文 (17.10) 的说明 | 补上：对混合值不同的伙伴，28 个“恰好在这两个头上相撞”的事件互不相容（不可能 3 头同撞），所以并集上界取等号；混合值完全相同的伙伴每个约 $2^{-63}\approx1.1\times10^{-19}$，可忽略 | §4.7 的整除论证；`2**-63 = 1.08e-19` |
| 课文 §4 二次拟合 | 系数写到 5 位：6e20 为 $a=0.13496,\ b=-0.19913$，顶点 0.738；2e20 为 $a=0.11978,\ b=-0.17346$，顶点 0.724 | 原来写 $a=0.1350,\ b=-0.1991$，读者用计算器得 $0.1991/0.27=0.737$，与 0.738 对不上；`np.polyfit` 用 §4.9 的 SVG 反算值 |
| 课文 §4 “ρ≈40% 时 46/43 个专家” | 写明纯 MoE 基线有 106 / 99 个路由专家（E-C14），“每 token 激活 6 个”标为笔者推断 | E-C14 原文；§4.2 |
| 课文 §5 “行、列均方根同时约为 1” | 补上前提：忽略 $\varepsilon$、没有行被阈值置零。被置零的行保持为零，若占比为 $f$，列均方根约为 $\sqrt{1-f}$ | 推导：末步按行归一后全矩阵平方和为 $(1-f)mn$；数值模拟（4000×256 随机矩阵、11 步交替归一）：$f=25\%$ 时列均方根 0.864，$\sqrt{0.75}=0.866$ |
| 课文 §5 阈值 | “范数不到它 τ 倍”改为“不超过”，与算法 1 第 6 行的 $\le$ 一致 | §2.1 转录 |
| 记号 | 幂律指数 $\gamma\to\kappa$（$\gamma$ 已是 RMSNorm 缩放与 Muon 缩放常数）；窗口注意力的窗口大小 $w\to n_{\text{win}}$（$w$ 已是卷积核宽与图像宽） | 本课记号一致性 |
| 图 17.8 说明 | “把 5.79B 的不激活专家换成 Engram 表”改为“把每层 17 个不激活专家（共 5.82B）换成 5.79B 的 Engram 表”，总高“几乎相同”（26.71B 对 26.69B） | `17*29*11796480 = 5,815,664,640`；§4.2 |

**S9. 事实与引用复核（2026-09-25）**

> 逐条列出课文里的数字、规格、日期、引文与归属，对照本笔记（含 §10 核查记录）；本笔记没有的，回到一手来源核对
> （`arxiv.py grep` 读 LaTeXML 全文），或引用本课程其他课已核实的研究笔记。课文里全部派生数字另在 Python 中重算过一遍，均与课文一致。

新核实的原文（一手来源）：

- **mHC 集成的适用范围**（`engram` v2 §2.4）：“Unless otherwise stated, all experiments utilize this integration with Manifold-Constrained Hyper-Connections ($M=4$)”。
  课文 §1、§3 原写“Engram 的全部实验都建立在 mHC 上”，已按原文改为“除非另行注明”（100B 卸载实验用的是稠密的 Dense-4B/8B 骨干，论文没说它们是否带 mHC）。
- **OverEncoding 是什么**（`engram` v2 §3.2）：“OverEncoding (Huang et al., 2025a), which integrates $N$-gram embeddings via averaging with the vocabulary embedding”。
  课文 §4 例题据此把 OverEncoding 写成“在输入层把 N-gram 嵌入与词嵌入取平均”。
- **层扫描实验是“加上” 1.6B，不是等参数替换**（`engram` v2 §6.2）：“We augment the backbone with a fixed 1.6B-parameter Engram memory.” 与课文 §3“在……MoE 上加 1.6B 参数的 Engram”一致。
- **门控在固定搭配的末尾打开**（`engram` v2 §6.5 与图 7 标题）：“The gating mechanism consistently activates (shown in red) upon completing local, static patterns.”；
  图 7 标题：“a high activation on a specific token $x_t$ implies that the preceding tokens culminating in that token … are recognized as a static pattern”。支持课文 §3“门控在局部固定搭配的最后一个 token 上打开”。
- **Fox 实验的提示词**（`ocr` §4.1）：“We use the prompt without layout: "<image>\nFree OCR." to control the model’s output format.” 支持课文 §7 错误框与图 17.13 的 “Free OCR.”。
- **V4.1 的 Muon 与 Sinkhorn 平衡更新的超参数**（`dsv41` §4.2.2，非 CC BY，只作短引与转述）：Muon 动量 0.95、权重衰减 0.1、每个更新矩阵的 RMS 缩放到 0.18；
  Sinkhorn 平衡更新 “use the same momentum coefficient and learning-rate correction factor as for Muon”，并设 $K=11,\ \tau=10^{-3},\ \varepsilon=10^{-20}$；§2.5：它 “also uses Nesterov momentum but does not apply weight decay”，
  $\gamma=0.18$ “close to the factor 0.2 used in Moonlight”。支持课文 §5 的 $\mu=0.95$、$\gamma=0.18$（原先只在 §3.4 表中出现，未附原文位置）。
- **作者顺序**（arXiv LaTeXML 首页）：v2 为 “Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai, Qinyu Chen, …”（前两人带共同一作标记）；v1 为 “Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, …”。
  与课文参考文献卡片 `engram`、`engram-acl`（ACL 名单同 v1，见 §0.1）一致。

引用本课程其他课研究笔记的事实（那些笔记已对一手来源核对）：

| 课文中的说法 | 出处 |
|---|---|
| V4 对大多数模块用 Muon（§5 前置知识“Muon 是 V4 用在大多数权重矩阵上的优化器”） | `lesson-18-doc.md` C34（`dsv4` §2.4 “majority of modules”） |
| V4.1-Flash 用纯 CSA2（不再有 HCA）、多层共享一份全局 KV、主 KV 存成 FP4（§8 结尾、练习 1） | `lesson-19-doc.md` C13、C22、C27 |
| V4.1-Flash 全局 KV 890 字节/token | `lesson-19-doc.md` C5（`dsv41` 摘要） |
| V4 的日期写 2026-04（时间线表） | 本笔记 §0.1（HTML 页脚 “26 Apr 2026”）；`lesson-18-doc.md` §0（Preview 2026-04-24 发布，arXiv v1 元数据 2026-04-26） |
| DeepSeek 分词器词表 129,280 | 本笔记 §3.1（表 5）；`lesson-18-doc.md` E16（config `vocab_size`） |

本次对课文的修改：

| 位置 | 原来 | 改为 | 理由 |
|---|---|---|---|
| 封面“位置”段、§3 多分支集成 | “Engram 的全部实验都建立在 / 跑在 mHC 上” | “除非另行注明”，并附原文短引 | 过度概括；原文有限定语（见上） |
| 图 17.6 说明 | “Engram-27B 和 V4.1-Flash 都把 Engram 放在第 2 层和第 15 层” | 写明 V4.1 的 `[1, 14]` 从 0 计；论文的“第 2、15 层”按从 1 数来画是笔者推测 | 勘误 E7：论文未说明计数起点，正文已标推测，图注原先写成定论 |
| §4 例题 OverEncoding | “把嵌入放在输入层的 OverEncoding” | 补上原文定义（与词嵌入取平均）并附短引 | 原说法无原文支撑，改为可核对的定义 |
| §2 式 (17.5) 后 | 填充编号补齐（无出处） | “开源代码用填充编号补齐”，引 `gh-engram` | 这是代码行为（§2.1 “位置 $t-i<0$ … 用 pad”），论文未写 |
| §6 解码器参数重建 | 无出处 | 列出所用的配置字段并引 `hf-ocr` | 相邻论断替换：重建用的是配置文件，原句只引了论文与 VL2 |
| §6 开头 | “一页 A4 文档分词后大约是一千个文本 token” | 加上 Fox 实验页面为 600–1300 token 的依据 | 原为无出处的约数 |
| §7 第三方证据 | “这说明高精度里有相当一部分来自语言先验” | “该文作者据此认为……” | 该预印本只读了摘要，结论应归于其作者，不应写成本课的判断 |
| §8 结尾 | “V4.1-Flash 再加上跨层复用与 FP4 的 KV” | “V4.1-Flash 改用纯 CSA2，让多层共享同一份全局 KV，并把主 KV 存成 FP4” | “再加上”暗示 V4.1 保留了 HCA；V4.1 是纯 CSA2（`lesson-19-doc.md` C22） |
| 练习 1 答案 | “CSA2、FP4、跨层复用” | “CSA2（多层共享全局 KV）、FP4 主 KV”，并链接第 19 课 | 同上，措辞对齐 |
| 时间线表 | V4 “2026” | “2026-04” | 表头说日期精确到 arXiv 首版或官方发布日 |
| §4 二次拟合推导 | $0.19913/0.26991$ | $0.19913/0.26992$ | $2a=2\times0.13496=0.26992$，与上一格写出的 $a$ 对齐 |
| 参考文献 `engram` 的 matters | “公式（式 1–9）” | “公式（原文式 (1)–(7)）” | 本课没有用到原文式 (8)(9)（CKA、软对齐指数），卡片不能说以它们为准 |
