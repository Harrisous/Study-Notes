# 第 14 课研究笔记：稀疏注意力——从 NSA 到 DSA

> 服务对象：`deepseek/src/lesson-14-sparse-attention.md`（short：DSA 稀疏注意力）。
> 建立：2026-09-24。所有引文均从一手来源抓取原文后核对（arXiv LaTeXML 全文、arXiv PDF、ACL Anthology
> 正式版 PDF、DeepSeek GitHub 仓库 PDF、Hugging Face `config.json` 与推理代码、FlashMLA 仓库）。
> 所有派生数字都在 Python 中复算过，表达式写在 §4。凡是“笔者推测”的地方都单独标注。
> 引文保持英文原文；为控制篇幅与版权风险，引文尽量只截取承载事实的最短从句，其余内容用中文转述并给出位置。
> NSA 的 ACL 正式版按 CC BY 4.0 发布（ACL Anthology 2016 年后的统一许可）；V3.2-Exp 技术报告放在 MIT 许可的
> GitHub 仓库中；V3.2 arXiv 论文为 arXiv 非独占许可，其 DSA 小节与 V3.2-Exp 报告逐字相同，引用时优先标 V3.2-Exp 报告。

---

## 0. 覆盖范围与所读版本

| 键 | 标题 | 作者 | 编号 / 版本 | 发表状态 | 日期 | 读法 |
|---|---|---|---|---|---|---|
| **NSA** | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng（DeepSeek-AI / 北大 / UW） | arXiv 2502.11089 **v2**（主读）；另读 v1 PDF 做 diff | **ACL 2025 长文**，Proceedings pp. 23078–23097，DOI 10.18653/v1/2025.acl-long.1126，**Best Paper**（四篇最佳论文之一） | v1 2025-02-16；v2 2025-02-27；ACL 2025 年 7 月（维也纳） | v2 LaTeXML 全文 + v2 PDF + v1 PDF + ACL 正式版 PDF 逐句比对；图 1、2、3、4、6、7、8 的原始 SVG/PNG 均打开看过，图 6 的柱高从 SVG 几何反解 |
| **V32Exp** | DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention | DeepSeek-AI | **无 arXiv**；`DeepSeek_V3_2.pdf`（6 页）于 github.com/deepseek-ai/DeepSeek-V3.2-Exp | 技术报告（artefact，未经同行评审） | PDF 元数据 CreationDate 2025-09-29 08:49:11Z；API 发布 2025-09-29 | pdftotext 全文 + 在 GitHub PDF 查看器中目视核对图 1、图 3、附录图 4 与公式 (1)–(4)；图 3 的曲线顶点从 PDF 矢量流中直接解析 |
| **V32** | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models | DeepSeek-AI（arXiv 作者列表共 264 条 = 机构名 “DeepSeek-AI” + Aixin Liu 等 263 位个人） | arXiv 2512.02556 **v1**（截至 2026-09-24 仅 v1） | 预印本 | 2025-12-02（API 发布 2025-12-01） | LaTeXML 全文 + PDF；§2（DSA）、§2.3（推理成本）、附录 A（MLA 的 MHA/MQA 模式）；图 3 矢量数据与 V32Exp 逐点比对 |
| V3（辅助） | DeepSeek-V3 Technical Report | DeepSeek-AI | arXiv 2412.19437 **v2** | 预印本 | v2 2025-02-18 | 只取 MLA 公式 (1)–(11) 与 §4.2 维度 |
| V2（辅助） | DeepSeek-V2 | DeepSeek-AI | arXiv 2405.04434 **v5** | 预印本 | v5 2024-06-19 | 只作 MLA 出处引用 |
| 制品 | HF `config.json`（V3、V3.1-Terminus、V3.2-Exp、V3.2）；V3.2-Exp `inference/model.py`、`kernel.py`、`config_671B_v3.2.json`（V3.2 仓库的 `model.py` 与之**逐字节相同**）；HF 提交历史；HF safetensors 参数计数 | DeepSeek-AI | — | artefact | 2025-09-29 起；“fix indexer rope” 提交 2025-11-17 | 原文下载后 grep / diff |
| 制品 | FlashMLA README 与博客 `docs/20250929-hopper-fp8-sparse-deep-dive.md` | DeepSeek-AI | — | artefact | 稀疏核发布 2025-09-29（README 当前版本已含 2026-09-10 的 V4.1 条目） | 原文下载 |
| 制品 | DeepSeek API 新闻页 news250922 / news250929（含价格图）/ news251201 | DeepSeek | — | artefact | 2025-09-22 / 09-29 / 12-01 | WebFetch + 价格图目视 |
| 制品 | ACL 2025 Awards 页、ACL Anthology 条目页 | ACL | — | 官方 | 2025 | WebFetch / 下载 |

**版本差异摘要**（细节见 §7）：
- NSA v1→v2（v1 的公式细节据 pdftotext 抽取的字形判断，未目视 v1 PDF）：式 (7) 的下标起点 `1⩽i` 改为 `0⩽i`；式 (8) 中 $\mathbf p_t^{cmp}$ 的维数由 $\lfloor\frac{t-l}{d}\rfloor$ 改为 $\lfloor\frac{t-l}{d}\rfloor+1$；式 (9) 下标里的两个 “+” 改为 “−”，前提条件由 “$d\mid l$ 且 $d\mid l'$” 增加 “$l\leqslant l'$”；§6.1 对“辅助损失选择”基线的描述补写为“mean-pooling + KL 监督”；相关工作大幅扩写。数字（l、d、l′、n、w、260B/270B、表 1–4、9.0×/6.0×/11.6×）三个版本一致。
- NSA ACL 正式版：章节重排（arXiv §2 “Rethinking” → 附录 B；§3 → §2；§4 → §3；§5 → §4；§7 相关工作 → 附录 A；骨干细节 → 附录 C；损失曲线 → 附录 D），**图号也随之变化**（NIAH 在 arXiv 为图 5、在 ACL 为图 4；kernel 速度图 arXiv 图 6 → ACL 图 5；预训练损失 arXiv 图 4 → ACL 图 6）；ACL 表 4 **删去了 32768 一列**；图 6 标题改为 “marginally lower loss values”；新增 Limitations 段。本笔记的节号、式号、图号**一律按 arXiv v2**，需要时括注 ACL 编号。
- V32Exp → V32：DSA 描述逐字相同（仅 “V3.2-Exp” 改为 “V3.2”），并明说 “DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp.”（V32 §2.1）。图号变化：V32Exp 图 1（架构）→ V32 图 2；V32Exp 图 3（成本）→ V32 图 3（**曲线数据逐点相同，只把图例由 V3.2-Exp 改为 V3.2**）；V32Exp 附录图 4 → V32 附录图 7。V32Exp 的“对等评估”只有表 1 和 RL 曲线（图 2）；V32 改为 ChatbotArena Elo（2025-11-10）、AA-LCR、Fiction.liveBench。

---

## 1. 核心论断与原文引文

### 1A. NSA（Yuan et al. 2025，arXiv v2，节号按 arXiv）

**C1. 动机：解码 64K 时注意力占延迟 70–80%（无出处的“理论估计”）。**
§1：“Theoretical estimates indicate that attention computation with softmax architectures accounts for 70–80% of total latency when decoding 64k-length contexts”。论文没有给出这个估计的推导或引用 → 只能当作作者的说法，见 §8。

**C2. 现有稀疏方法的两个“幻觉”。** §2.1 标题 “The Illusion of Efficient Inference”，列两点：阶段受限的稀疏（phase-restricted sparsity，H2O 只稀疏 decode、MInference 只稀疏 prefill）；与 GQA/MQA 不兼容——Quest 类方法每个头各自选 KV，在 GQA 下
“the memory access volume of KV-cache corresponds to the union of selections from all query heads within the same GQA group.”（§2.1）
→ 这是 NSA 与 DSA 都坚持“同一组查询头共享同一批 KV”的根本原因。

**C3. “可训练稀疏”的神话。** §2.2 “The Myth of Trainable Sparsity”：事后（post-hoc）稀疏会让模型偏离预训练轨迹；转引 Chen et al. 2024b（MagicPIG）：“top 20% attention can only cover 70% of the total attention scores”（§2.2，二手数字，见 §8）。另两点：离散操作不可导（ClusterKV 的 k-means、MagicPIG 的 SimHash）；token 粒度选择导致非连续访存、无法用 FlashAttention（HashAttention）。

**C4. 总框架：用按查询构造的紧凑 KV 集合替换原 KV。** §3.2 式 (3)(4)(5)。三种映射 $\mathcal C=\{\text{cmp},\text{slc},\text{win}\}$，门控
“derived from input features via an MLP and sigmoid activation.”（§3.2，指 $g_t^c\in[0,1]$）
并要求 $N_t\ll t$（式 (6)）。注意：三个门是**各自独立的 sigmoid**，不是 softmax，三者之和不必为 1。

**C5. 压缩分支：块长 $l$、步长 $d$，$d<l$ 让相邻块重叠。** §3.3.1 式 (7)；$\varphi$ 是 “a learnable MLP with intra-block position encoding”；原文：“Usually, we adopt d<l to mitigate information fragmentation.”（§3.3.1）

**C6. 选择分支按块选，理由是硬件与注意力的空间连续性。** §3.3.2：
“modern GPU architectures exhibit significantly higher throughput for continuous block accesses compared to random index-based reads.”
并以 §6.2 / 图 8 的注意力图为证：注意力分数呈块状聚集。

**C7. 选择分数“白嫖”压缩分支的注意力分数，不加新参数。** §3.3.2 式 (8)(9)：$l'=l=d$ 时直接相等；否则按空间重叠求和。

**C8. GQA 组内共享选择。** §3.3.2 式 (10)：
“consistent block selection across these heads has to be ensured to minimize KV cache loading during decoding.”
组内 $H$ 个头的分数相加后再做 top-$n$（式 (11)(12)）。

**C9. 滑动窗口分支防“捷径学习”，三分支 KV 各自独立。** §3.3.3：
“local patterns typically adapt faster and can dominate the learning process”；
“we provide independent keys and values for three branches.”
→ 教学要点：窗口分支不是为了省算力，而是为了让压缩/选择分支学到长程模式。

**C10. 核设计：以 GQA 组为单位装载查询（group-centric）。** §3.4：
“for each position on the query sequence, we load all query heads within a GQA group (they share the same sparse KV blocks) into SRAM.”
三要点：Group-Centric Data Loading（一次装 $Q\in\R^{[h,d_k]}$ 与共享索引 $\mathcal I_t$）；Shared KV Fetching（按 $\mathcal I_t$ 顺序装连续 KV 块，$B_k\mid l'$）；Outer Loop on Grid（查询/输出循环交给 Triton grid）。压缩与窗口分支直接复用 FlashAttention-2 核。只针对 GQA/MQA（“Given MHA is memory-intensive and inefficient for decoding”）。

**C11. 实验骨干。** §4.1（ACL 附录 C）：27B 总参数 / 3B 激活的 GQA+MoE；30 层，隐藏维 2560；GQA 组数 4、共 64 头；$d_q=d_k=192$、$d_v=128$；DeepSeekMoE 72 路由 + 2 共享、top-6；第一层用 SwiGLU MLP 代替 MoE。

**C12. NSA 超参数（本课核心数字，已核实）。** §4.1：
“compression block size l=32, sliding stride d=16, selected block size l′=64, selected block count n=16 (including fixed activating the 1 initial block and 2 local blocks), and sliding window size w=512.”
§5 效率分析重申同一组值，并给 GQA 组 $g=4$、每组头数 $h=16$、$d_k=192$、$d_v=128$。
→ **16 个选择块里有 3 个是固定的**（1 个开头块 + 2 个本地块），真正由分数动态选出的是 13 个。

**C13. 训练数据量：260B 还是 270B？论文自相矛盾。** §1：“Pretraining on a 27B-parameter transformer backbone with 260B tokens”；§4.1：“pretrained on 270B tokens of 8k-length texts”。v1、v2、ACL 三版都保留这一矛盾。随后在 32K 上继续训练 + SFT，用 YaRN 做长上下文适配。见 §7。

**C14. 通用基准（表 1）。** 9 项中 7 项 NSA ≥ Full Attention，平均 0.456 vs 0.443；DROP +0.042、GSM8K +0.034（§4.3）。复算见 §4.4。

**C15. 长上下文。** 64K 大海捞针全位置满分（图 5）；LongBench（表 2）平均 0.469，比 Full Attention 高 0.032、比 Exact-Top 高 0.046。为公平，所有稀疏基线的 token 预算设为 2560，
“which corresponds to the average number of tokens activated in NSA when handling 32k sequence lengths.”（§4.3）；预算内含开头 128 个 token 与 512 个本地 token。复算：按式 (7)(12) 对 32K 内所有位置取平均得 2541（精确均值 2540.96），与 2560 差 0.7%（§4.1 复算）。

**C16. 推理链 SFT（表 3）。** 用 DeepSeek-R1 蒸馏的 10B tokens、32K 长的数学推理轨迹做 SFT；AIME 24，温度 0.7、top-p 0.95、每题 16 个样本取平均：8K 限长 0.046 → 0.121（+0.075），16K 限长 0.092 → 0.146（+0.054）。

**C17. 训练速度（图 6，A100 上 Triton 对 Triton-FA2）。** §5.1：“up to 9.0× forward and 6.0× backward speedup at 64k context-length.” 这是**注意力核**的前向/反向耗时比，不是端到端训练吞吐。

**C18. 解码加速 11.6× 是由访存量推算的“期望加速”，不是实测延迟。** §5.2：每步最多装
$\lfloor\frac{s-l}{d}\rfloor$ 个压缩 token、$nl'$ 个选择 token、$w$ 个邻近 token；表 4 标题：
“the expected speedup is approximately linear with the volume of memory access.”
表 4 行名就叫 “Expected Speedup”。而 §5.2 正文与图 1 右图把它写成 “achieving up to 11.6× speedup”。见 §7 勘误。

**C19. 放弃的两条路线（§6.1，DSA 前史的关键证据）。**
(a) 基于辅助损失的可学习选择：为每个 token 加额外查询、为每块加代表键，对块内注意力分数做 mean-pooling 当监督信号，“and use KL divergence to supervise block importance prediction”（v2 新增句）；
(b) Quest 式启发式无参数选择；并试过“先用 Full Attention 训 1000 步再切换到启发式选择”的冷启动：
“Full Attention is applied for the initial 1000 steps before transitioning to the heuristic blockwise selection.”
结论：两者损失都更差（图 7，3B 模型）。放弃 (a) 的理由：
“relies on auxiliary loss, which increases operator overhead and often degrades model performance”。
→ **历史注脚（笔者观察）**：DSA 恰恰是“用 KL 监督一个独立打分器 + 先稠密 1000 步预热”——正是 NSA 当年试过并放弃的组合，只是换成了 token 粒度、ReLU+FP8 的 indexer，并且把 indexer 的梯度与主模型**分离**（detach）。课文可以把这当作“同一个想法在不同工程条件下翻盘”的例子，但要标注为笔者的解读，论文没有这样说。

**C20. ACL 版局限性。** ACL Limitations：“we focus on sparsity within attention maps of a single layer. Exploring cross-layer sparsity remains an intriguing avenue for future research.” 另承认 Triton 实现相对原生 CUDA 有抽象开销。→ 可接第 19 课 V4.1-Flash 的跨层 KV 复用。

### 1B. DSA（V32Exp 技术报告 = V32 §2，引文逐字相同）

**C21. DSA 是 V3.1-Terminus 之上唯一的架构改动，通过续训引入。** V32Exp §1：“the only architectural modification of DeepSeek-V3.2-Exp is the introduction of DeepSeek Sparse Attention (DSA) through continued training.” V32 §2.1 同句，并加一句 “DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp.”

**C22. 两个组件。** “The prototype of DSA primarily consists of two components: a lightning indexer and a fine-grained token selection mechanism.”（V32Exp §1 / V32 §2.1）

**C23. lightning indexer 的打分公式（式 (1)）与“为什么是 ReLU、为什么便宜”。**
“We choose ReLU as the activation function for throughput consideration.”
“Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable.”
$\mathbf q^I_{t,j}\in\R^{d^I}$ 与 $w^I_{t,j}\in\R$ 由查询 token $\mathbf h_t$ 导出，$\mathbf k^I_s\in\R^{d^I}$ 由前文 token $\mathbf h_s$ 导出。论文**没有**给出 $H^I$、$d^I$ 的数值——数值来自 `config.json`：`index_n_heads = 64`、`index_head_dim = 128`（§3.3）。

**C24. token 级 top-k 选择（式 (2)）。** “retrieves only the key-value entries $\{\mathbf c_s\}$ corresponding to the top-k index scores.” 注意：这里的 $\mathbf c_s$ 是 MLA 的“KV 条目”，即图中拼接的 $[\mathbf c_s^{KV};\mathbf k_s^R]$（576 维），不只是 $\mathbf c_s^{KV}$。$k$ 的数值不在架构小节，而在训练小节：“select 2048 key-value tokens for each query token”；`config.json`：`index_topk = 2048`。

**C25. 在 MLA 的 MQA 模式上实例化，理由直接引 NSA。**
“At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency (Yuan et al., 2025). Therefore, we implement DSA based on the MQA (Shazeer, 2019) mode of MLA”，
“where each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token.”（V32Exp §1 / V32 §2.1）
附录（V32Exp 图 4 / V32 图 7 标题）：“For DeepSeek-V3.1-Terminus, the MHA mode is used for training and prefilling, while the MQA mode is used for decoding.”
→ 一个查询 token 的 128 个头**共用同一组 2048 个被选 token**；这与 NSA“同一 GQA 组的 16 个头共用同一组块”是同一原则推到极致（组 = 全部头）。

**C26. 续训起点与数据。** “Starting from a base checkpoint of DeepSeek-V3.1-Terminus, whose context length has been extended to 128K”；两阶段的数据分布 “is totally aligned with the 128K long context extension data used for DeepSeek-V3.1-Terminus.”（V32Exp §2.1）

**C27. 稠密预热阶段：只训 indexer。**
“we keep dense attention and freeze all model parameters except for the lightning indexer.”
目标分布：对第 $t$ 个查询，把主注意力分数**按头求和**，再沿序列维做 L1 归一化得到 $p_{t,:}\in\R^t$；损失为式 (3) 的 KL。
超参数：“we use a learning rate of $10^{-3}$. We train the indexer for only 1000 steps, with each step consisting of 16 sequences of 128K tokens, resulting in a total of 2.1B tokens.”

**C28. 稀疏训练阶段：全参数，indexer 仍用 KL 但只在被选集合上。**
“we introduce the fine-grained token selection mechanism and optimize all model parameters to adapt the model to the sparse pattern of DSA.” 损失为式 (4)，只在 $\mathcal S_t=\{s\mid I_{t,s}\in\text{Top-k}(I_{t,:})\}$ 上算。

**C29. 梯度分离。**
“we detach the indexer input from the computational graph for separate optimization. The training signal of the indexer is from only $\mathcal L^I$, while the optimization of the main model is according to only the language modeling loss.”
→ 主模型不会为了“让 indexer 好学”而改变表示；indexer 也拿不到语言模型损失的梯度（top-k 本来就不可导）。

**C30. 稀疏阶段超参数。** “we use a learning rate of $7.3\times10^{-6}$, and select 2048 key-value tokens for each query token. We train both the main model and the indexer for 15000 steps, with each step consisting of 480 sequences of 128K tokens, resulting in a total of 943.7B tokens.” 复算：只有取 128K = 131,072 才得到 943.7B（§4.2）。

**C31. 后训练也用稀疏注意力。** “The post-training of DeepSeek-V3.2-Exp also employs sparse attention in the same way as the sparse continued pre-training stage.”（V32Exp §2.2；V32 §3 同句）。V32Exp 为了隔离 DSA 的影响，后训练流水线、算法、数据与 V3.1-Terminus 相同：“we maintain the same post-training pipeline, algorithm, and data as used for DeepSeek-V3.1-Terminus”。

**C32. 复杂度。** “DSA reduces the core attention complexity of the main model from $O(L^2)$ to $O(Lk)$, where $k$ ($\ll L$) is the number of selected tokens. Although the lightning indexer still has a complexity of $O(L^2)$, it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus.”（V32Exp §3 / V32 §2.3）

**C33. 成本图的口径。** “These costs are estimated from benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU hour.” 以及
“for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions.”
→ 图 3 是**内部成本估计**（GPU 租价折算），**不是 API 价格**。

**C34. 质量对等（V32Exp 表 1）。** V3.1-Terminus → V3.2-Exp：MMLU-Pro 85.0 → 85.0；GPQA-Diamond 80.7 → 79.9；HLE 21.7 → 19.8；BrowseComp 38.5 → 40.1；BrowseComp-zh 45.0 → 47.9；SimpleQA 96.8 → 97.1；LiveCodeBench 74.9 → 74.1；Codeforces 2046 → 2121；Aider-Polyglot 76.1 → 74.5；SWE Verified 68.4 → 67.8；SWE-bench Multilingual 57.8 → 57.9；Terminal-bench 36.7 → 37.7；AIME 2025 88.4 → 89.3；HMMT 2025 86.1 → 83.6。表题解释 GPQA/HLE/HMMT 下降：“because DeepSeek-V3.2-Exp generates fewer reasoning tokens”，且用 token 数相当的中间检查点时差距消失（作者说法，未给数据）。

**C35. V32 的补充对等证据。** ChatbotArena：两者后训练策略相同，2025-11-10 的 Elo “closely matched”；AA-LCR 上 V3.2-Exp 在推理模式下高 4 分；Fiction.liveBench 多项指标更好（V32 §2.2）。均为第三方或间接评测，未给具体分数表。

**C36. 作者自己的保留。** V32Exp §3 “Future Validation in Real World”：仍在做大规模真实场景测试，“to uncover potential limitations of the sparse attention architecture.”（V32 删去了这一段。）

**C37. “首次实现细粒度稀疏注意力”是营销说法。** GitHub README：“DeepSeek Sparse Attention (DSA) achieves fine-grained sparse attention for the first time”。技术报告正文没有这个说法；NSA 论文里的 Exact-Top、HashAttention、MInference/TokenSelect 等都是 token 级选择（多为推理期方法）。课文若引用，应写成“DeepSeek 在 README 中称……”，见 §7。

### 1C. 核、部署与定价（artefact）

**C38. FlashMLA 稀疏核。** README：2025-09-29 发布 token 级稀疏注意力核，“achieve up to 640 TFlops during prefilling and 410 TFlops during decoding”（H800 SXM5）。支持矩阵：Sparse Decoding 与 Sparse Prefill 均为 **MQA 模式**（`head_dim_k = 576`、`head_dim_v = 512`）；Dense Prefill 为 MHA 模式（`head_dim_k = 192/128`、`head_dim_v = 128`）。

**C39. V3.2 的 FP8 KV cache 格式：每 token 656 字节。** FlashMLA README：前 512 字节是 512 个 `float8_e4m3`（NoPE 部分），接着 16 字节 = 4 个 `float32` 缩放因子（每 128 个值一个），最后 128 字节 = 64 个 `bfloat16`（RoPE 部分，“not quantized for accuracy”）。稠密解码核读 bf16/fp16 KV；`is_fp8_kvcache=True` 只与稀疏注意力一起支持。

**C40. FP8 稀疏解码核的性能与“≈3000 交叉点”。** FlashMLA 博客（2025-09-29）：每 SM 每周期 4096 MMA FLOPs（`989 TFlops / 1830 MHz / 132 SMs`）；每 KV token MMA 约 34 周期、反量化约 50 周期 → 核是 “dequantization-bound”；用 2-CTA cluster “crossover” 分摊反量化，从 250 提到 410 TFLOPS（batch 128、128 头、s_q=2、topk=2048）；并说该配置下核的执行时间 “is comparable to that of the dense decoding kernel when the sequence length is around 3000.” 博客另给 128K 单请求的 KV：`576 × 2 × 62 × 128 × 1024 = 8.72 GiB`（博客只给算式、不解释各因子；把 62 读作 61 层 + 1 个 MTP 层、把 ×2 读作 BF16 两字节，是笔者的解读）。

**C41. 推理 demo 的 RoPE 布局 bug（2025-11-17 修复）。** README：“previous versions of the inference demo code contained an implementation discrepancy in Rotary Position Embedding (RoPE) within the indexer module”；indexer 的 RoPE 输入要求 non-interleaved 布局，而 MLA 模块要求 interleaved。HF 提交历史：2025-11-17 “fix indexer rope”。

**C42. API 降价。** DeepSeek 新闻 2025-09-29：“DeepSeek API prices drop 50%+, effective immediately.” 价格图（每百万 token，2025-09-29 10:00 UTC 生效）：输入缓存命中 $0.07 → $0.028，缓存未命中 $0.56 → $0.28，输出 $1.68 → $0.42；V3.1-Terminus 临时 API 保留到 2025-10-15 15:59 UTC。V3.1-Terminus 发布于 2025-09-22（news250922）。

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 NSA（arXiv v2 式号；ACL 版式号相同）

背景注意力（式 (1)(2)，NSA 原记号，$i$ 是**键的位置**）：
$$
\mathbf o_t=\operatorname{Attn}\left(\mathbf q_t,\mathbf k_{:t},\mathbf v_{:t}\right)\tag{1}
$$
$$
\operatorname{Attn}\left(\mathbf q_t,\mathbf k_{:t},\mathbf v_{:t}\right)=\sum_{i=1}^{t}\frac{\alpha_{t,i}\mathbf v_i}{\sum_{j=1}^{t}\alpha_{t,j}},\quad \alpha_{t,i}=e^{\frac{\mathbf q_t^\top\mathbf k_i}{\sqrt{d_k}}}\tag{2}
$$
重映射框架：
$$
\tilde K_t=f_K(\mathbf q_t,\mathbf k_{:t},\mathbf v_{:t}),\quad \tilde V_t=f_V(\mathbf q_t,\mathbf k_{:t},\mathbf v_{:t})\tag{3}
$$
$$
\mathbf o^*_t=\operatorname{Attn}\left(\mathbf q_t,\tilde K_t,\tilde V_t\right)\tag{4}
$$
$$
\mathbf o^*_t=\sum_{c\in\mathcal C}g_t^c\cdot\operatorname{Attn}(\mathbf q_t,\tilde K_t^c,\tilde V_t^c),\qquad \mathcal C=\{\text{cmp},\text{slc},\text{win}\},\ g_t^c\in[0,1]\tag{5}
$$
$$
N_t=\sum_{c\in\mathcal C}\operatorname{size}[\tilde K_t^c]\tag{6}
$$
压缩（v2；v1 为 $1\leqslant i$）：
$$
\tilde K_t^{\text{cmp}}=f_K^{\text{cmp}}(\mathbf k_{:t})=\left\{\varphi(\mathbf k_{id+1:id+l})\ \middle|\ 0\leqslant i\leqslant\left\lfloor\frac{t-l}{d}\right\rfloor\right\}\tag{7}
$$
正文称 $\tilde K_t^{\text{cmp}}\in\R^{d_k\times\lfloor\frac{t-l}{d}\rfloor}$——与 $0\leqslant i\leqslant\lfloor\frac{t-l}{d}\rfloor$ 共 $\lfloor\frac{t-l}{d}\rfloor+1$ 个元素差一（原文内部不一致，见 §7）。

选择分数（v2 维数为 $+1$；v1 无 $+1$）：
$$
\mathbf p_t^{\text{cmp}}=\operatorname{Softmax}\left(\mathbf q_t^{T}\tilde K_t^{\text{cmp}}\right),\qquad \mathbf p_t^{\text{cmp}}\in\R^{\lfloor\frac{t-l}{d}\rfloor+1}\tag{8}
$$
当 $l'=l=d$ 时 $\mathbf p_t^{\text{slc}}=\mathbf p_t^{\text{cmp}}$；一般情形（条件 $l\leqslant l'$，$d\mid l$，$d\mid l'$）：
$$
\mathbf p_t^{\text{slc}}[j]=\sum_{m=0}^{\frac{l'}{d}-1}\sum_{n=0}^{\frac{l}{d}-1}\mathbf p_t^{\text{cmp}}\left[\frac{l'}{d}j-m-n\right]\tag{9}
$$
（v1 下标为 $\frac{l'}{d}j+m+n$。）GQA 组内共享：
$$
{\mathbf p_t^{\text{slc}}}'=\sum_{h=1}^{H}\mathbf p_t^{\text{slc},(h)}\tag{10}
$$
top-$n$：
$$
\mathcal I_t=\{i\mid \operatorname{rank}({\mathbf p_t^{\text{slc}}}'[i])\leqslant n\}\tag{11}
$$
$$
\tilde K_t^{\text{slc}}=\operatorname{Cat}\left[\{\mathbf k_{il'+1:(i+1)l'}\mid i\in\mathcal I_t\}\right],\qquad \tilde K_t^{\text{slc}}\in\R^{d_k\times nl'}\tag{12}
$$
滑动窗口（§3.3.3，无式号）：$\tilde K_t^{\text{win}}=\mathbf k_{t-w:t},\ \tilde V_t^{\text{win}}=\mathbf v_{t-w:t}$。

解码访存上界（§5.2，无式号）：每步最多 $\lfloor\frac{s-l}{d}\rfloor+nl'+w$ 个 token，$s$ 为已缓存长度。

**式 (9) 的权重模式（笔者推导，可作练习）**：$l=32,d=16,l'=64$ 时 $\frac{l'}{d}=4$、$\frac{l}{d}=2$，$m+n$ 取 0..4 的次数为 $(1,2,2,2,1)$，即
$\mathbf p^{\text{slc}}[j]=\mathbf p^{\text{cmp}}[4j]+2\mathbf p^{\text{cmp}}[4j-1]+2\mathbf p^{\text{cmp}}[4j-2]+2\mathbf p^{\text{cmp}}[4j-3]+\mathbf p^{\text{cmp}}[4j-4]$。
直觉：一个 64 长的选择块与 5 个（长 32、步长 16 的）压缩块有重叠，中间 3 个完全包含、两端 2 个半重叠，按重叠加权。**但**按式 (7)/(12) 的下标约定逐一核对，完全落在选择块 $j$（键 $64j+1..64j+64$）内的压缩块下标是 $4j,4j+1,4j+2$，而式 (9) 取的是 $4j-4..4j$——下标约定差一个块（v1 的 “+” 版本同样对不齐）。课文讲“按重叠加权”的意思即可，不要逐下标照抄；见 §7。

**NSA 记号 → 课程记号**（style-guide §2 冲突处理建议）：

| NSA 原记号 | 含义 | 与课程记号的冲突 | 建议课文写法 |
|---|---|---|---|
| $l$ | 压缩块长度（32） | 课程 $l$ = 层数 | $\ell_c$ |
| $d$ | 压缩步长（16） | 课程 $d$ = 隐藏维 | $\delta_c$ |
| $l'$ | 选择块大小（64） | — | $\ell_s$ |
| $n$ | 选择块数（16，含 3 个固定块） | — | $n_s$ |
| $w$ | 滑动窗口（512） | 无（DSA 的 $w^I_{t,j}$ 带上标区分） | $w$ |
| $d_k,d_v$ | 每头 QK / V 维（192 / 128） | 课程每头维 $d_h$ | $d_h^{QK}$、$d_h^{V}$，或直接写数字 |
| $i$（式 2）| 键位置 | 课程历史位置用 $j$（DeepSeek 记号里 $i$ 是头） | 键位置写 $j$ 或 $s$ |
| $i$（式 11、12）| 选择块编号 | 同上 | $b$ |
| $j$（式 9）| 选择块编号 | 同上 | $b$ |
| $h$、$H$ | 组内头编号 / 组内头数（16） | 课程 $\mathbf h$ 是隐藏状态 | 头编号 $i$，组内头数 $n_h/n_g$ |
| $g$（§5）| GQA 组数（4） | 与门控 $g_t^c$ 同字母 | $n_g$ |
| $g_t^c$ | 分支门控 | 课程 $g_{i,t}$ = MoE 门控值 | 保留 $g_t^{c}$，首次出现声明“与 MoE 门控无关” |
| $s$（§5.2）| 已缓存序列长 | 课程 $s$ 是位置下标 | $T$ |
| $t$ | 当前位置 / 已见长度 | 一致 | $t$ |

### 2.2 DSA（V32Exp 式号 = V32 式号，均为 (1)–(4)）

$$
I_{t,s}=\sum_{j=1}^{H^I}w_{t,j}^I\cdot\operatorname{ReLU}\left(\mathbf q^I_{t,j}\cdot\mathbf k^I_s\right)\tag{1}
$$
$\mathbf h_t,\mathbf h_s\in\R^d$；$\mathbf q^I_{t,j}\in\R^{d^I}$、$w^I_{t,j}\in\R$ 来自 $\mathbf h_t$；$\mathbf k^I_s\in\R^{d^I}$ 来自 $\mathbf h_s$；$H^I$ 为 indexer 头数。
$$
\mathbf u_t=\operatorname{Attn}\left(\mathbf h_t,\ \left\{\mathbf c_s\ \middle|\ I_{t,s}\in\operatorname{Top\text{-}k}(I_{t,:})\right\}\right)\tag{2}
$$
稠密预热：
$$
\mathcal L^I=\sum_t\mathbb D_{\mathrm{KL}}\left(p_{t,:}\,\middle\|\,\operatorname{Softmax}(I_{t,:})\right)\tag{3}
$$
稀疏训练，$\mathcal S_t=\{s\mid I_{t,s}\in\operatorname{Top\text{-}k}(I_{t,:})\}$：
$$
\mathcal L^I=\sum_t\mathbb D_{\mathrm{KL}}\left(p_{t,\mathcal S_t}\,\middle\|\,\operatorname{Softmax}(I_{t,\mathcal S_t})\right)\tag{4}
$$

**$p_{t,:}$ 的显式形式（笔者推导，不是原文公式）**：设主注意力第 $i$ 头对位置 $s$ 的概率为 $a_{t,i,s}$，$\sum_s a_{t,i,s}=1$。“按头求和再 L1 归一化”
$$
p_{t,s}=\frac{\sum_{i=1}^{n_h}a_{t,i,s}}{\sum_{s'}\sum_{i=1}^{n_h}a_{t,i,s'}}=\frac{\sum_i a_{t,i,s}}{n_h}=\frac1{n_h}\sum_{i=1}^{n_h}a_{t,i,s},
$$
即 128 个头注意力分布的**算术平均**。KL 方向是 $\KL(p\,\|\,q_{\text{indexer}})$（以主注意力为“真分布”的前向 KL），它惩罚“主注意力看重、indexer 却给低分”的 token——正好对应“漏选”这一代价最大的错误（笔者解读）。式 (4) 中 $p_{t,\mathcal S_t}$ 是否在 $\mathcal S_t$ 上重新归一化，原文没写；稀疏阶段主注意力本来只在 $\mathcal S_t$ 上算 softmax，天然归一，故大概率是（列入 §8）。

**代码层面的 indexer（`inference/model.py` 类 `Indexer` + `kernel.py` 的 `fp8_index`；不是论文公式，按代码转写）**：
$$
\begin{aligned}
\mathbf q^I_{t,j}&=\operatorname{RoPE}_{[:64]}\!\left(W^{IQ}_j\,\mathbf c_t^{Q}\right),&W^{IQ}&\in\R^{(H^I d^I)\times d_c'}=\R^{8192\times1536}\\
\mathbf k^I_s&=\operatorname{RoPE}_{[:64]}\!\left(\operatorname{LayerNorm}(W^{IK}\mathbf h_s)\right),&W^{IK}&\in\R^{d^I\times d}=\R^{128\times7168}\\
w_{t,j}&=(W^{Iw}\mathbf h_t)_j,&W^{Iw}&\in\R^{H^I\times d}=\R^{64\times7168}\\
I_{t,s}&=\sum_{j=1}^{H^I}\frac{w_{t,j}}{\sqrt{H^I}\sqrt{d^I}}\operatorname{ReLU}\!\left(\mathcal H\mathbf q^I_{t,j}\cdot\mathcal H\mathbf k^I_s\right)
\end{aligned}
$$
其中 $\mathcal H=H_{128}/\sqrt{128}$ 是归一化 Hadamard 变换（`rotate_activation`），之后 $\mathcal H\mathbf q$、$\mathcal H\mathbf k$ 按 128 维一块量化为 FP8 E4M3（缩放格式 UE8M0），点积在 FP8 上做、FP32 累加，ReLU 之后乘权重与缩放再按头求和。与论文式 (1) 的差别：(i) 多了 $1/\sqrt{H^I}$、$1/\sqrt{d^I}$ 两个常数缩放（对 top-k 排序无影响，对 Softmax$(I)$ 的温度有影响）；(ii) 查询来自 MLA 的查询潜向量 $\mathbf c_t^Q$（代码里的 `qr`，即 $\operatorname{RMSNorm}(W^{DQ}\mathbf h_t)$），不是直接来自 $\mathbf h_t$；(iii) 键是**单个**向量，被 64 个 indexer 头共享（indexer 自身也是 MQA 结构）；(iv) 只对前 64 维加 RoPE（`rope_head_dim = qk_rope_head_dim = 64`），非交错布局；(v) Hadamard 旋转是正交变换，不改变点积（$\mathcal H^\top\mathcal H=I$），笔者推测其作用是把离群值摊平以利 FP8 量化（论文未提）。

**MLA 相关式（V3 arXiv v2 式 (1)–(11)，课程第 9 课已讲，这里只列 DSA 用到的）**：
$\mathbf c_t^{KV}=W^{DKV}\mathbf h_t$ (1)；$\mathbf k_t^C=W^{UK}\mathbf c_t^{KV}$ (2)；$\mathbf k_t^R=\operatorname{RoPE}(W^{KR}\mathbf h_t)$ (3)；$\mathbf k_{t,i}=[\mathbf k_{t,i}^C;\mathbf k_t^R]$ (4)；$\mathbf v_t^C=W^{UV}\mathbf c_t^{KV}$ (5)；$\mathbf c_t^Q=W^{DQ}\mathbf h_t$ (6)；$\mathbf q_t^C=W^{UQ}\mathbf c_t^Q$ (7)；$\mathbf q_t^R=\operatorname{RoPE}(W^{QR}\mathbf c_t^Q)$ (8)；$\mathbf q_{t,i}=[\mathbf q_{t,i}^C;\mathbf q_{t,i}^R]$ (9)；
$$
\mathbf o_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_j\!\left(\frac{\mathbf q_{t,i}^T\mathbf k_{j,i}}{\sqrt{d_h+d_h^R}}\right)\mathbf v_{j,i}^C\ (10),\qquad \mathbf u_t=W^O[\mathbf o_{t,1};\dots;\mathbf o_{t,n_h}]\ (11)
$$

**MQA 模式（吸收形式；笔者按 V32 附录图 7 与 `model.py` 的 decode 分支写出，不是论文式）**：
$$
\mathbf q_{t,i}^A=(W^{UK}_i)^\top\mathbf q_{t,i}^C\in\R^{d_c},\qquad
\mathbf o^C_{t,i}=\sum_{s\in\mathcal S_t}\operatorname{Softmax}_s\!\left(\frac{[\mathbf q_{t,i}^A;\mathbf q_{t,i}^R]^\top[\mathbf c_s^{KV};\mathbf k_s^R]}{\sqrt{d_h+d_h^R}}\right)\mathbf c_s^{KV},\qquad
\mathbf o_{t,i}=W^{UV}_i\mathbf o^C_{t,i}
$$
（$W^{UK}_i,W^{UV}_i\in\R^{d_h\times d_c}$ 为第 $i$ 头的切片。）于是每个被选条目 $[\mathbf c_s^{KV};\mathbf k_s^R]\in\R^{576}$ 同时充当 128 个头的“键”（576 维）和“值”（前 512 维）——这就是 “each latent vector ... shared across all query heads” 的字面含义。代码中 softmax 缩放还乘了 YaRN 的 `mscale²`（`mscale = 0.1·ln(40)+1`；条件是 `max_seq_len > original_seq_len = 4096`，demo 默认 `max_seq_len = 16384`，故生效）。

**DSA 记号 → 课程记号**：

| 原记号 | 含义 | 冲突 | 建议 |
|---|---|---|---|
| $\mathbf h_t$, $d$ | 注意力层输入、隐藏维 7168 | 一致 | 保留 |
| $I_{t,s}$ | index score | 无 | 保留 |
| $H^I$, $d^I$ | indexer 头数 64、头维 128 | 无 | 保留 |
| $j$ | **indexer 头编号** | 课程 $j$ = 历史位置 | 本课历史位置统一用 $s$（与论文一致），首次出现说明 $j$ 在本课是 indexer 头 |
| $k$（top-k 的 k） | 被选 token 数 2048 | 与键向量 $\mathbf k$ 同字母（粗细不同） | 写 $k=2048$，首次出现说明；或写 $k_{\text{sel}}$ |
| $L$（$O(L^2)$ 中） | 序列长度 | 课程 $L$ = 层数 | 写 $O(T^2)\to O(Tk)$，或 $L_{ctx}$ |
| $\mathbf c_s$ | MLA 的 KV 条目 | 易误读为 $\mathbf c_s^{KV}$ | 写 $[\mathbf c_s^{KV};\mathbf k_s^R]$ |
| $\mathbb D_{\mathrm{KL}}$ | KL 散度 | — | 宏 `\KL` |
| $p_{t,:}$, $\mathcal S_t$, $\mathcal L^I$ | 目标分布、被选集合、indexer 损失 | 无 | 保留 |

### 2.3 本课使用的计算量模型（笔者定义，§4 全部由此复算）

每层、每个查询 token、位置 $T$（能看到 $T$ 个键）的**核心注意力** FLOPs（每次乘加记 2 FLOPs；只算 $QK^\top$ 与 $PV$ 两个矩阵乘，不算 softmax、ReLU、top-k 选择、投影）：
- 稠密，MLA 的 MHA 模式（Terminus 训练/prefill）：$F_{\text{MHA}}(T)=2n_hT\,[(d_h+d_h^R)+d_h]=2\cdot128\cdot T\cdot(192+128)=81{,}920\,T$
- 稠密，MQA 模式（Terminus decode）：$F_{\text{MQA}}(T)=2n_hT\,[(d_c+d_h^R)+d_c]=2\cdot128\cdot T\cdot(576+512)=278{,}528\,T$
- DSA 核心（MQA 模式，只看 $\min(T,k)$ 个键）：$F_{\text{core}}(T)=278{,}528\,\min(T,2048)$
- lightning indexer（只有 QK，FP8）：$F_{\text{idx}}(T)=2H^Id^I\,T=2\cdot64\cdot128\cdot T=16{,}384\,T$
- 与 $T$ 无关的常数项（不计入上面）：indexer 三个投影 $2\times13{,}959{,}424\approx27.9$M；MQA 模式的吸收乘法 $W^{UK}$、$W^{UV}$ 各 $2n_hd_hd_c=16.8$M。

---

## 3. 规格与超参数表

### 3.1 NSA 超参数

| 量 | 值 | 来源 |
|---|---|---|
| 压缩块长 $l$ | 32 | NSA §4.1、§5（ACL §3.1、§4） |
| 压缩步长 $d$ | 16 | 同上 |
| 选择块大小 $l'$ | 64 | 同上 |
| 选择块数 $n$ | 16（含固定的 1 个开头块 + 2 个本地块） | §4.1（括号说明只在 §4.1 出现，§5 未重复） |
| 滑动窗口 $w$ | 512 | §4.1、§5 |
| 门控 | 每分支一个 MLP + sigmoid，$g_t^c\in[0,1]$ | §3.2 |
| 压缩函数 $\varphi$ | 带块内位置编码的可学习 MLP | §3.3.1 |
| 三分支 KV | 各自独立 | §3.3.3 |
| 实现 | Triton；压缩/窗口复用 FlashAttention-2 核；选择分支自研核 | §3.4 |
| 核块大小 $B_k$ | 满足 $B_k\mid l'$（数值未给） | §3.4 |

### 3.2 NSA 实验骨干与训练

| 量 | 值 | 来源 |
|---|---|---|
| 总 / 激活参数 | 27B / 3B | §4.1 |
| 层数 / 隐藏维 | 30 / 2560 | §4.1 |
| 注意力 | GQA，4 组，共 64 头（每组 16 头） | §4.1、§5 |
| 每头维 | $d_q=d_k=192$，$d_v=128$ | §4.1 |
| MoE | DeepSeekMoE，72 路由 + 2 共享，top-6；第 1 层用 SwiGLU MLP | §4.1 |
| 预训练 | **270B**（§4.1）/ **260B**（§1）tokens，8K 长 | 论文内部矛盾 |
| 长上下文适配 | 32K 续训 + SFT，YaRN | §4.1 |
| 推理 SFT | R1 蒸馏 10B tokens，32K 长数学推理轨迹 | §4.3 |
| 速度测试硬件 | 8×A100，Triton 对 Triton-FA2 | §5 |
| 消融模型 | 3B（图 7） | §6.1 |

### 3.3 DSA / V3.2 规格：论文、config、代码三方对照

| 量 | 论文 | `config.json`（V3.2-Exp 与 V3.2 相同） | 推理代码 | 备注 |
|---|---|---|---|---|
| indexer 头数 $H^I$ | “a small number of heads”（无数值） | `index_n_heads: 64` | `index_n_heads = 64` | 主注意力 `num_attention_heads: 128` |
| indexer 头维 $d^I$ | 无数值 | `index_head_dim: 128` | 128 | 其中前 64 维加 RoPE |
| top-$k$ | 2048（V32 §2.1.1 / V32Exp §2.1 稀疏阶段） | `index_topk: 2048` | `min(index_topk, end_pos)` | $T\le2048$ 时等于全选 |
| indexer 精度 | “can be implemented in FP8” | — | q、k 均 FP8 E4M3，128 元素一块，UE8M0 缩放；`weights_proj` 存为 bf16 | 代码另有 Hadamard 旋转 |
| indexer 激活 | ReLU | — | `T.max(logits, 0)` | |
| indexer 参数/层 | 无 | — | $1536\cdot8192+7168\cdot128+7168\cdot64+2\cdot128=13{,}959{,}424$ | ×62（含 MTP 层）= 865,484,288，与 HF 按精度分项的参数差额逐项吻合（total 差额多出的 51,088 是 FP32 缩放张量，§4.6） |
| 注意力实现模式 | MLA 的 MQA 模式 | — | demo 的 prefill 分支用 **MHA 模式 + 掩码**，decode 分支用 MQA 模式；**两个分支都先对全部可见位置算稠密分数，再加 −∞ 掩码**（decode 分支对 `:end_pos` 全算） | 论文另说短序列 prefill 用 masked MHA 模式 |
| 主 MLA 维度 | 与 V3.1-Terminus 相同 | $n_h=128$，$d_h=128$（`qk_nope_head_dim`、`v_head_dim`），$d_h^R=64$，$d_c=512$（`kv_lora_rank`），$d_c'=1536$（`q_lora_rank`） | 同 | V3 论文 §4.2 数值相同 |
| 层数 | — | 61（+1 MTP 层，`num_nextn_predict_layers: 1`） | 每个 MLA 层各有一个 `Indexer` | |
| 隐藏维 $d$ | — | 7168 | | |
| 位置上限 | 上下文 128K（续训起点） | `max_position_embeddings: 163840`，YaRN factor 40，original 4096 | | 服务上下文按论文为 128K |
| KV cache 精度 | — | — | demo 模拟 FP8（`act_quant` 后再反量化） | FlashMLA：656 B/token/层 |
| `model_type` | — | `deepseek_v32` | | Terminus 为 `deepseek_v3` |
| 权重缩放格式 | — | `quantization_config.scale_fmt: ue8m0`（嵌套在 `quantization_config` 内，顶层无此键） | `inference/config_671B_v3.2.json` 顶层 `scale_fmt: ue8m0` | Terminus 的 `quantization_config` 无此字段 |

### 3.4 DSA 两阶段续训

| 阶段 | 可训练参数 | 注意力 | 损失 | lr | 步数 | 每步 | tokens（论文） | 复算 |
|---|---|---|---|---|---|---|---|---|
| 稠密预热 | 只有 indexer | 稠密 | 式 (3) KL（对全部位置） | $10^{-3}$ | 1000 | 16 × 128K | 2.1B | $1000\cdot16\cdot131072=2.097\times10^9$ |
| 稀疏训练 | 全部（主模型 + indexer，梯度分离） | top-2048 | 主模型：LM 损失；indexer：式 (4) KL（只在 $\mathcal S_t$） | $7.3\times10^{-6}$ | 15000 | 480 × 128K | 943.7B | $15000\cdot480\cdot131072=9.437\times10^{11}$ |
| 后训练 | 全部 | 与稀疏阶段相同 | 专家蒸馏 + 混合 RL（GRPO） | — | — | — | — | — |

两阶段共用“与 V3.1-Terminus 128K 长上下文扩展数据完全对齐”的数据分布。

### 3.5 KV cache 字节（每 token、每层）

| 格式 | 字节 | 来源 |
|---|---|---|
| MLA，BF16，$(d_c+d_h^R)\times2$ | 1152 | 由维度计算；FlashMLA 稠密解码读 bf16 |
| MLA，V3.2 FP8 “with scale” | 656 = 512 + 16 + 128 | FlashMLA README |
| indexer 键，FP8 + 1 个 FP32 缩放 | 132 = 128 + 4 | `model.py`：`k_cache` 为 `float8_e4m3fn`，`k_scale_cache` 为 float32，`head_dim // block_size = 1` |
| V4 / V4.1 / V4.1 fp4（对照，属第 18/19 课） | 584 / 528 / 288 | FlashMLA README |

### 3.6 时间线

| 日期 | 事件 | 来源 |
|---|---|---|
| 2025-02-16 | NSA arXiv v1 | arXiv |
| 2025-02-27 | NSA arXiv v2 | arXiv |
| 2025-07 | NSA 获 ACL 2025 Best Paper（四篇之一） | ACL 2025 Awards 页、Anthology 条目 |
| 2025-09-22 | DeepSeek-V3.1-Terminus 发布 | api-docs news250922 |
| 2025-09-29 | V3.2-Exp 发布（权重、推理 demo、技术报告 PDF、FlashMLA 稀疏核）；API 降价 50%+。README 同时指向 TileLang 示例与 DeepGEMM PR #200（indexer logits 核），这两者的合入日期未核 | news250929；PDF 元数据；HF 提交；FlashMLA README News 条目 |
| 2025-10-15 | V3.1-Terminus 临时 API 下线 | news250929 |
| 2025-11-17 | 推理 demo 修复 indexer RoPE 布局 | HF 提交 “fix indexer rope”；README |
| 2025-12-01 / 12-02 | V3.2 与 V3.2-Speciale 发布；arXiv 2512.02556 v1 | news251201；arXiv |

### 3.7 参数量（HF safetensors 元数据，2026-09-24 抓取）

| 模型 | BF16 | F8_E4M3 | F32 | total |
|---|---|---|---|---|
| V3.1-Terminus | 3,917,742,080 | 680,571,043,840 | 42,600,080 | 684,531,386,000 |
| V3.2-Exp | 3,946,184,704 | 681,408,069,632 | 42,667,040 | 685,396,921,376 |
| V3.2 | 3,946,184,704 | 681,408,069,632 | 1,075,456 | 685,355,329,792 |
| V3 | 3,918,786,560 | 680,571,043,840 | 41,555,600 | 684,531,386,000 |

---

## 4. 复算的数字

全部用 Python 算过；表达式可直接粘贴复现。

### 4.1 NSA

| 量 | 表达式 | 值 |
|---|---|---|
| 表 4，8K 访存 token | `8192/16 + 16*64 + 512` | 2048（精确上界 $\lfloor(8192-32)/16\rfloor+1024+512=2046$；按式 (8) 的 +1 为 2047） |
| 表 4，16K | `16384/16 + 1024 + 512` | 2560 |
| 表 4，32K | `32768/16 + 1536` | 3584 |
| 表 4，64K | `65536/16 + 1536` | 5632 |
| 期望加速 | `8192/2048, 16384/2560, 32768/3584, 65536/5632` | 4.0×, 6.4×, 9.14×（表印 9.1×）, 11.64×（表印 11.6×） |
| 64K 稀疏率 | `5632/65536` | 8.59% |
| 32K 内按位置平均的激活 token | `mean(floor((t-32)/16)+1 + min(1024,t) + min(512,t))`，$t=32..32768$ | 2541（论文称 2560） |
| 每查询键数 @4K/32K/128K（外推） | `t/16+1024+512` | 1792 / 3584 / 9728 |
| 式 (9) 权重 | `Counter(m+n for m in range(4) for n in range(2))` | {0:1, 1:2, 2:2, 3:2, 4:1} |
| 64 头 = 组数 × 组内头 | `4*16` | 64 |
| 表 1 平均 | `mean([.567,.279,.576,.497,.486,.263,.503,.482,.335])` / NSA 同 | 0.4431 / 0.4558（表印 0.443 / 0.456）；NSA 胜 7/9 |
| 表 2 平均 | Full / NSA / Exact-Top | 0.4370 / 0.4687 / 0.4234（表印 0.437 / 0.469 / 0.423） |
| 表 2 增量 | NSA − Full | HPQ +0.087，2Wiki +0.051，LCC +0.069，PassR-en +0.075；MFQA-en −0.009，GovRpt −0.017，PassR-zh −0.010 |
| 表 3 增量 | `0.121-0.046, 0.146-0.092` | +0.075, +0.054 |

**图 6 柱高（从 SVG 几何反解；刻度：前向 200 ms = 32.571 pt，反向 500 ms = 31.667 pt）**：

| 长度 | FA2 前向 ms | NSA 选择核 | NSA 压缩+窗口 | NSA 合计 | 加速（图上标注） | FA2 反向 ms | NSA 选择 | NSA 压缩+窗口 | NSA 合计 | 加速（标注） |
|---|---|---|---|---|---|---|---|---|---|---|
| 8K | 16.3 | 5.5 | 2.3 | 7.8 | 2.1× | 42.5 | 28.3 | 10.3 | 38.5 | 1.1× |
| 16K | 63.8 | 11.1 | 5.8 | 16.9 | 3.8× | 165.6 | 59.8 | 24.2 | 84.0 | 2.0× |
| 32K | 249.5 | 23.5 | 15.8 | 39.3 | 6.3× | 647.8 | 124.4 | 64.1 | 188.5 | 3.4× |
| 64K | 998.1 | 47.0 | 63.3 | 110.3 | 9.0× | 2591.5 | 250.9 | 180.5 | 431.4 | 6.0× |

复算加速：前向 2.09/3.78/6.35/9.05，反向 1.10/1.97/3.44/6.01，与图中标注一致。FA2 每翻倍耗时 ×3.91/3.91/4.00（二次方）；NSA 选择核每翻倍约 ×2（线性），压缩+窗口在 64K 前向已超过选择核（63.3 > 47.0 ms）——压缩分支本身是 $O(T^2/\delta_c)$。

### 4.2 DSA 训练与规格

| 量 | 表达式 | 值 |
|---|---|---|
| 预热 tokens | `1000*16*131072` | 2,097,152,000 ≈ 2.1B |
| 若 128K 取 128,000 | `1000*16*128000` | 2.048B（会印成 2.0B） |
| 稀疏阶段 tokens | `15000*480*131072` | 943,718,400,000 ≈ 943.7B ✓ |
| 若 128K 取 128,000 | `15000*480*128000` | 921.6B ✗ → 论文的 128K = 131,072 |
| 稀疏/预热 token 比 | `(15000*480)/(1000*16)` | 450 倍；预热只占续训的 0.22% |
| 两阶段 batch（token） | `16*131072`, `480*131072` | 2.10M, 62.9M tokens/step |
| lr 比 | `1e-3/7.3e-6` | 137 倍 |
| 选择比例 @128K | `2048/131072` | 1.5625% |
| indexer 参数/层 | `1536*64*128 + 7168*128 + 7168*64 + 2*128` | 13,959,424 |
| ×62 层（含 MTP） | | 865,484,288 ≈ 0.87B（占 685B 的 0.13%） |

### 4.3 FLOPs：稠密 vs DSA（V3 维度，每层、每查询 token、位置 $T$）

| $T$ | 稠密 MHA 模式 | 稠密 MQA 模式 | DSA 核心 | indexer | DSA 合计 | MHA/DSA | MQA/DSA |
|---|---|---|---|---|---|---|---|
| 4,096 | 335.5 M | 1,140.9 M | 570.4 M | 67.1 M | 637.5 M | **0.53** | 1.79 |
| 8,192 | 671.1 M | 2,281.7 M | 570.4 M | 134.2 M | 704.6 M | 0.95 | 3.24 |
| 32,768 | 2,684.4 M | 9,126.8 M | 570.4 M | 536.9 M | 1,107.3 M | 2.42 | 8.24 |
| 65,536 | 5,368.7 M | 18,253.6 M | 570.4 M | 1,073.7 M | 1,644.2 M | 3.27 | 11.10 |
| 131,072 | 10,737.4 M | 36,507.2 M | 570.4 M | 2,147.5 M | 2,717.9 M | **3.95** | 13.43 |

×61 层（每 token，GFLOPs）：@4K MHA 20.47 / MQA 69.59 / DSA 核心 34.80 / indexer 4.09 / DSA 合计 38.89；@32K 163.75 / 556.74 / 34.80 / 32.75 / 67.55；@128K 654.98 / 2226.94 / 34.80 / 131.00 / 165.79。

| 交叉点 / 比值 | 表达式 | 值 |
|---|---|---|
| 每键 FLOPs：MQA/MHA 模式 | `(576+512)/(192+128)` | 3.4 |
| indexer / MHA 模式（每键） | `16384/81920` | 0.20 |
| indexer / MQA 模式（每键） | `16384/278528` | 0.0588（1/17） |
| DSA 核心 = 稠密 MHA（不计 indexer） | `278528*2048/81920` | $T$ = 6,963 |
| DSA 核心+indexer = 稠密 MHA | `278528*2048/(81920-16384)` | $T$ = 8,704 |
| DSA 核心+indexer = 稠密 MQA（decode 口径） | `278528*2048/(278528-16384)` | $T$ = 2,176 |
| indexer = DSA 核心 | `278528*2048/16384` | $T$ = 34,816（此后 indexer FLOPs 超过核心注意力） |
| 整段 prefill（因果，逐位置求和）每层 | 稠密 MHA $81920\cdot T(T+1)/2$；DSA 分段求和 | 4K：0.687 vs 1.890 TFLOPs（DSA 贵 2.75×）；32K：43.98 vs 26.90（1.63×）；128K：703.7 vs 214.9（3.27×） |
| 同上 ×61 | | 128K：42,925 vs 13,110 TFLOPs/序列 |
| 参数 FLOPs 基线 | `2*37e9` | 74 GFLOPs/token（V3 激活 37B） |
| 注意力占比 @4K/32K/128K（稠密 MHA） | `61*F/(61*F+74e9)` | 21.7% / 68.9% / 89.8% |
| 注意力占比（DSA，含 indexer） | 同上 | 34.4% / 47.7% / 69.1% |
| MQA 吸收常数项 | `2*128*128*512` | 16.8M（$W^{UK}$）+ 16.8M（$W^{UV}$）/ token / 层 |

**解读（笔者推测，需在课文中标注）**：
1. 在 MQA 模式下，每个被选键的 FLOPs 是 MHA 模式的 3.4 倍，所以短序列时“稀疏”反而更贵：只看核心注意力，交叉点约 7.0K；这与论文“短序列 prefill 改用 masked MHA 模式模拟 DSA”以及图 3(a) V3.2 曲线在 6.589K 处的拐点（§4.5）相互印证。
2. decode 口径下（两边都是 MQA 模式），交叉点只有约 2.2K；图 3(b) 的拐点在 2.056K ≈ $k=2048$。
3. 128K 时 indexer 的 FLOPs（2.15G/层）已是 DSA 核心（0.57G/层）的 3.8 倍——“$O(Tk)$”只描述核心注意力；indexer 的 $O(T^2)$ 在更长上下文下会重新成为主项（这是第 18 课 V4 CSA 先沿序列压缩再稀疏选择的动机之一，需在第 18 课核实）。

### 4.4 KV cache 与 decode 访存

| 量 | 表达式 | 值 |
|---|---|---|
| MLA BF16 每 token（61 层） | `576*2*61` | 70,272 B = 68.6 KiB |
| MLA FP8 每 token（61 层） | `656*61` | 40,016 B |
| indexer 键每 token（61 层） | `132*61` | 8,052 B；为 FP8 MLA 的 20.1% |
| FlashMLA 博客 128K 单请求 | `576*2*62*128*1024` | 9,361,686,528 B = 8.72 GiB ✓（62 层 BF16） |
| 128K 单请求，FP8，61 层 | `656*61*131072/2**30` | 4.88 GiB；加 indexer 键 5.87 GiB |
| decode 每步每层读字节 @128K：稠密 FP8 vs DSA | `656*L` vs `656*2048+132*L` | 86.0 MB vs 18.6 MB → 4.61× |
| 同上，稠密 BF16（Terminus 实际稠密核读 bf16） | `1152*L` | 151.0 MB → 8.10× |
| 同上 @32K | | FP8 3.79×；BF16 6.66× |
| H800 每 SM 每周期 MMA FLOPs | `989e12/1.83e9/132` | 4094（博客取 4096） |
| 每 KV token MMA 周期（64 头/CTA） | `64*(576+512)*2/4096` | 34.0 |
| 每 token 反量化周期 | `(1/64+1/64+1/16+1/256)*512` | 50.0 |

**要点**：DSA 不减少 KV cache 的**存储**（任何历史 token 都可能被选中，所以 MLA 条目一个都不能丢，还要多存 indexer 键）；它减少的是每步**读取和计算**的量。

### 4.5 成本图（V32Exp 图 3 = V32 图 3）的精确顶点与派生量

从两份 PDF 的矢量内容流解析出的**全部**折线顶点（x 为坐标轴上的 “K” 单位，y 为美元/百万 token；两份 PDF 逐点相同）：

| 曲线 | 顶点 |
|---|---|
| (a) Prefilling，V3.1-Terminus | (0, 0.0529) → (128, 0.6658) |
| (a) Prefilling，V3.2(-Exp) | (0, 0.0611) → (6.589, 0.0999) → (128, 0.1871) |
| (b) Decoding，V3.1-Terminus | (0, 0.0705) → (128, 2.1543) |
| (b) Decoding，V3.2(-Exp) | (0, 0.0952) → (2.056, 0.1457) → (101.607, 0.2274) → (128, 0.2491) |

轴：prefill y 轴 0–0.7$，刻度 0.1$；decode y 轴 0–2.4$，刻度 0.4$；x 轴 0K–128K，刻度 32K。**曲线只有 2–4 个顶点，是分段直线**，说明图是由少量测点拟合/连线得到的模型化曲线，而不是密集实测。

| 派生量 | 值 |
|---|---|
| 128K 位置的成本比 | prefill 0.6658/0.1871 = **3.56×**；decode 2.1543/0.2491 = **8.65×** |
| 64K 位置 | prefill 2.55×；decode 5.66× |
| 32K 位置 | prefill 1.74×；decode 3.47× |
| 4K 位置 | prefill 0.85×；decode 0.92×（V3.2 更贵） |
| 交叉点（V3.2 开始便宜） | prefill ≈ 10.4K；decode ≈ 4.76K |
| 整段 0–128K 的平均成本 | prefill 0.3594 vs 0.1403（2.56×）；decode 1.1124 vs 0.1961（5.67×） |
| 斜率（$/百万 token 每 1K 位置） | prefill：Terminus 0.00479，V3.2 前段 0.00589、后段 0.00072；decode：Terminus 0.01628，V3.2 前段 0.02456、后段 0.00082 |
| 折算 GPU 时间（$2/GPU·h） | 0.0529$ → 95 GPU·s / 百万 token；prefill@128K Terminus 1.198 GPU·ms/token vs V3.2 0.337；decode@128K 3.878 vs 0.448 GPU·ms/token |
| 与 FLOPs 模型对照 | 128K 位置 prefill 成本比 3.56×，FLOPs 模型（稠密 MHA vs DSA 合计）3.95×；decode 成本比 8.65×，BF16 字节模型 8.10×（但 32K 处字节模型 6.66× 与成本比 3.47× 不符，说明成本还含大量与注意力无关的项，不宜逐点套用） |
| API 降价幅度 | 缓存命中 −60%（0.028/0.07=0.40），未命中 −50%，输出 −75%（0.42/1.68=0.25） |

### 4.6 参数差额核对（HF safetensors）

| 量 | 表达式 | 值 |
|---|---|---|
| F8 差额（V3.2-Exp − Terminus） | `681408069632-680571043840` | 837,025,792 |
| indexer FP8 权重 ×62 | `(1536*8192 + 7168*128)*62` | 837,025,792 ✓ **精确相等** |
| BF16 差额 | `3946184704-3917742080` | 28,442,624 = `7168*64*62`（`weights_proj`）✓ |
| F32 差额 | `42667040-42600080` | 66,960 = 62 × 1080；1080 = LayerNorm 256 + `wq_b` 缩放 768 + `wk` 缩放 56 ✓ |
| total 差额 | `685396921376-684531386000` | 865,535,376 = indexer 参数 865,484,288 + FP32 缩放张量 51,088（`62*(768+56)`，HF 把 FP8 权重的 scale 张量也计入参数） ✓ |

→ 这证明：(i) indexer 在 **62 层**都有（含 MTP 层）；(ii) indexer 的查询投影输入确实是 1536 维的查询潜向量；(iii) 除 indexer 外 V3.2 没有新增任何参数，印证“唯一的架构改动”。

---

## 5. 图表清单与重绘建议

### 5.1 论文原图清单

**NSA（arXiv v2 编号；括号内为 ACL 编号）**
| 图/表 | 标题要点 | 内容 | 课文用途 |
|---|---|---|---|
| 图 1（1） | 性能与效率对比 | 左：General 0.443/0.456、LongBench 0.437/0.469、Reasoning 0.092/0.146（= 表 3 的 16K 列）；右：Decode 11.6×、Forward 9.0×、Backward 6.0× | 可不重绘，只引用数字并注明 decode 为“期望”值 |
| 图 2（2） | NSA 架构 | 左：$\mathbf k_{:t},\mathbf v_{:t}$ 切成连续块 → Compression / Selection（top-n，Concat）/ Sliding 三分支 → 三个 Output → Gated Output；右：三种注意力掩码（Compressed / Selected / Sliding Attention Mask）；图例：绿色 = 需算注意力分数 | **重绘**（图 14.3） |
| 图 3（3） | 核设计 | Q：$N\times d_K\times h$ 按 GQA 组沿 Grid Loop；K：$d_K\times N$、V：$d_V\times N$ 沿 Inner Loop，按索引 “Select In” 装 $B_K$ 块到 SRAM；输出 $N\times h\times d_V$ | **重绘**（图 14.5） |
| 图 4（6） | 27B 预训练损失 | 0–60000 步，损失约 4.0 → 1.65，NSA 略低；原图标题拼作 “Taining Loss Curve” | 不重绘，文字提及 |
| 表 1 | 通用基准 | 9 项 | 引用 |
| 表 2 | LongBench | H2O / InfLLM / Quest / Exact-Top / Full / NSA | 可做条形图 |
| 图 5（4） | 64K 大海捞针 | 全绿（满分） | 不重绘 |
| 表 3 | AIME 24 | 8K / 16K 限长 | 引用 |
| 图 6（5） | 前向/反向耗时 | 见 §4.1 反解表 | **重绘**（图 14.7） |
| 表 4 | decode 访存量 | 8K–64K（ACL 删去 32K） | **重绘**（图 14.6） |
| 图 7 | 3B 消融损失 | Full / 辅助损失可学习选择 / 启发式无参数选择 / NSA，0–30000 步，终值约 2.37–2.40，NSA 最低 | 可做示意 |
| 图 8 | 注意力图可视化 | 27B Full Attention 模型；q 位置 0–256、kv 位置 0–1024；色标 −3 到 −1（对数刻度）；块状聚集 | 做示意（图 14.4 的依据） |

**V32Exp / V32**
| 图/表 | 内容 | 课文用途 |
|---|---|---|
| V32Exp 图 1 / V32 图 2 | DSA 在 MLA 下的注意力架构。自下而上：Input Hidden $\mathbf h_t$ → $\mathbf c_t^Q$ → $\{\mathbf q_{t,i}^C\}$（吸收后 $\{\mathbf q_{t,i}^A\}$）与 apply RoPE 的 $\{\mathbf q_{t,i}^R\}$ 拼接；$\mathbf c_t^{KV}$ 与 apply RoPE 的 $\mathbf k_t^R$ 拼接成 $\{[\mathbf c_t^{KV};\mathbf k_t^R]\}$；绿色部分：$\{\mathbf q_{t,j}^I\}$（partially apply RoPE）、$\mathbf k_t^I$（partially apply RoPE）、$\{w_{t,j}^I\}$ → Lightning Indexer → Top-k Selector → Multi-Query Attention (Core Attention) → $\{\mathbf o_{t,i}^C\}$ → $\{\mathbf o_{t,i}\}$ → Output Hidden $\mathbf u_t$ | **重绘**（图 14.8） |
| V32Exp 图 2 | RL 训练曲线（BrowseComp、SWE Verified），实线准确率、虚线平均输出 token 数；两模型曲线接近 | 可不重绘 |
| V32Exp 图 3 / V32 图 3 | 成本曲线（§4.5 顶点） | **重绘**（图 14.11） |
| V32Exp 图 4 / V32 图 7 | MLA 的 MHA 模式 vs MQA 模式；MHA 模式把 $\mathbf c^{KV}$ 经 $W^{UK}$、$W^{UV}$ 展开成每头键值；MQA 模式把 $W^{UK}$ 吸收进查询、$W^{UV}$ 移到输出 | **重绘**（图 14.9） |
| V32Exp 表 1 | 14 项基准对比 | 做对比表 |

### 5.2 建议本课绘制的图（14 张，数据齐全，可直接算几何）

1. **图 14.1 注意力 FLOPs 随位置的二次增长（接第 4 课）**。x：位置 $T$ ∈ {1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K}（对数轴）；y：每 token 全模型 GFLOPs（对数轴）。曲线：稠密 MHA 模式 $61\times81920T/10^9$；水平线：参数 FLOPs 74 GFLOPs。交点 $T=74\times10^9/(61\cdot81920)=14{,}808$（注意力超过其余计算之和的位置）。颜色：注意力线 `--fig-blue`，参数线 `--fig-muted`。

2. **图 14.2 稀疏模式四联（32×32 因果掩码示意）**。同一 32 token 序列：(a) 稠密因果；(b) 滑动窗口 $w=4$；(c) NSA 式块选择（块大小 4，每查询选 1 个开头块 + 2 个本地块 + 1 个动态块，窗口 4）；(d) DSA 式 token 级选择（$k=6$，被选位置可任意分散）。格子：被算 = `--fig-green` 12% 填充，被跳过 = 白，被掩 = `--fig-red` 细斜线。动态块/token 的位置由作者随机指定并在说明中标“示意”。

3. **图 14.3 NSA 三分支结构（据 NSA 图 2 重绘）**。输入 $\mathbf k_{:t},\mathbf v_{:t}$；分支 1：长 $\ell_c=32$、步长 $\delta_c=16$ 的重叠块 → $\varphi$ → 压缩键值（每 16 token 一个）；分支 2：由分支 1 的注意力分数经式 (9) 得块分数 → 组内求和（式 10）→ top-16 块（每块 64 token）；分支 3：最近 512 token；三输出经 sigmoid 门 $g^{\text{cmp}},g^{\text{slc}},g^{\text{win}}$ 加权求和。标注：三分支 KV 独立；虚线箭头“分数复用”从压缩分支指向选择分支。

4. **图 14.4 式 (9) 的重叠加权**。一条 0–192 的键轴；压缩块 $i=0..10$ 画成 [16i+1, 16i+32] 的横条（交错两行以示重叠）；选择块画成 [64b+1, 64b+64]；对某一选择块标出与之重叠的 5 个压缩块及权重 1,2,2,2,1。说明中注明原文下标约定差一（§2.1、§7）。

5. **图 14.5 NSA 组中心核（据 NSA 图 3 重绘）**。Grid Loop：按查询位置取一个 GQA 组的 $h=16$ 个头，$Q\in\R^{16\times192}$；Inner Loop：按 $\mathcal I_t$ 依次装 $B_k$ 行的 $K\in\R^{B_k\times192}$、$V\in\R^{B_k\times128}$ 到 SRAM；输出 $16\times128$ 写回 HBM。绿色 = SRAM，蓝色 = HBM（沿用原图语义，但课程配色里 SRAM/HBM 用 `--fig-amber` 深浅区分更一致）。

6. **图 14.6 NSA decode 访存量与期望加速（表 4）**。分组柱状：x = 8K/16K/32K/64K；Full = 8192/16384/32768/65536；NSA 堆叠三段：压缩 512/1024/2048/4096、选择 1024×4、窗口 512×4；柱顶标 4×/6.4×/9.1×/11.6×，并在标题注“按访存量推算的期望值”。

7. **图 14.7 NSA 核耗时（图 6 重绘）**。两个面板（前向、反向），数据用 §4.1 反解表；NSA 柱分“选择核”（`--fig-green`）与“压缩+窗口”（`--fig-violet`）两段；FA2 柱 `--fig-blue`。建议 y 轴用对数，否则 8K/16K 的 NSA 柱几乎看不见。

8. **图 14.8 DSA 数据流（据 V32 图 2 重绘，带形状）**。$\mathbf h_t\in\R^{7168}$ → $\mathbf c_t^Q\in\R^{1536}$ → 主查询 $128\times(128+64)$，吸收后 $128\times576$；$\mathbf h_t$ → $[\mathbf c_t^{KV};\mathbf k_t^R]\in\R^{576}$（`--fig-orange`，写入 KV cache `--fig-amber`）；indexer（绿色框，本课可用 `--fig-violet` 表示“打分”）：$\mathbf c_t^Q\to\mathbf q^I\in\R^{64\times128}$，$\mathbf h_t\to w\in\R^{64}$，$\mathbf h_s\to\mathbf k^I_s\in\R^{128}$（FP8，写入 indexer 键缓存 132 B/层）→ $I_{t,:}\in\R^{t}$ → Top-2048 → 从 KV cache 取 2048 个 576 维条目 → MQA 核心注意力（128 头共享）→ $\mathbf o^C\in\R^{128\times512}$ → $W^{UV}$ → $W^O$ → $\mathbf u_t\in\R^{7168}$。

9. **图 14.9 MLA 的 MHA 模式 vs MQA 模式（据 V32 附录图 7 重绘）**。左：缓存 576 维 → 每头展开为键 192、值 128；每键 FLOPs $2\cdot128\cdot320=81{,}920$。右：查询吸收 $W^{UK}$ → 每头 576 维；每键 FLOPs $2\cdot128\cdot1088=278{,}528$；每键读 576 个数、被 128 头共享。标注“×3.4 FLOPs，换来 KV 只读一次”。

10. **图 14.10 FLOPs 曲线与交叉点（本课核心计算图）**。x：$T$ 从 512 到 131072（对数）；y：每层每 token FLOPs（对数）。五条线：$81920T$（稠密 MHA）、$278528T$（稠密 MQA）、$278528\min(T,2048)$（DSA 核心，在 2048 处折平）、$16384T$（indexer）、DSA 合计。竖虚线标注交叉点 2,176（vs MQA）、6,963（核心 vs MHA）、8,704（合计 vs MHA）、34,816（indexer = 核心）。

11. **图 14.11 每百万 token 成本随位置（据 V32 图 3 重绘）**。两个面板，用 §4.5 的精确顶点画折线；标注交叉点 10.4K（prefill）、4.76K（decode），标注 128K 处 3.56× 与 8.65×，标注 prefill 拐点 6.589K（“masked MHA 模式”）与 decode 拐点 2.056K（≈ $k$）。说明中写明“按 H800 租价 $2/GPU·h 估算，不是 API 价格”。

12. **图 14.12 两阶段续训时间线**。横条长度按 token 数（对数刻度，否则预热看不见）：预热 2.1B（lr $10^{-3}$、只训 indexer、稠密、式 3）→ 稀疏 943.7B（lr $7.3\times10^{-6}$、全参数、top-2048、式 4）→ 后训练（稀疏）。起点标 “V3.1-Terminus（已扩到 128K）”。

13. **图 14.13 KL 对齐示意**。4 个候选 token 的柱状图：两头主注意力 $a_1=(0.10,0.70,0.15,0.05)$、$a_2=(0.20,0.40,0.35,0.05)$ → 平均 $p=(0.15,0.55,0.25,0.05)$；indexer $I=(1,3,1.5,0)$ → $\operatorname{Softmax}(I)=(0.0961,0.7101,0.1584,0.0354)$；$\KL(p\|\cdot)=0.0576$。稀疏阶段只看 top-2 $\{2,3\}$：~~$p_{\mathcal S}=(0.6875,0.3125)$，KL = 0.0491~~（已更正，见 §11.E：每个头先在 $\mathcal S$ 内归一再平均，$p_{\mathcal S}=(0.6784,0.3216)$，KL = 0.0557），$\operatorname{Softmax}(I_{\mathcal S})=(0.8176,0.1824)$。画一个“stop-gradient”记号表示 indexer 输入与主模型分离。

14. **图 14.14 每 token 每层缓存字节**。条形：MLA BF16 1152、MLA FP8 656、indexer 键 132、DSA 合计 788（= 656 + 132）；再给 128K 单请求 61 层总量：BF16 8.58 GiB、FP8 4.88 GiB、FP8+indexer 5.87 GiB。强调“DSA 让缓存变大而不是变小”。

---

## 6. 算例与练习素材

1. **（计算）NSA 在 64K 解码时读多少 token？** 用 $\ell_c=32,\delta_c=16,\ell_s=64,n_s=16,w=512$。答：压缩 $65536/16=4096$（精确上界 $\lfloor(65536-32)/16\rfloor=4094$），选择 $16\times64=1024$，窗口 512，合计约 5632；期望加速 $65536/5632=11.6$；稀疏率 8.6%。追问：为什么这只是“期望”加速？（decode 访存受限，时间 ∝ 读的字节；没算 kernel 启动、选择开销等；论文表 4 行名即 “Expected Speedup”。）

2. **（计算）验证 LongBench 基线的 2560 token 预算。** 对 $t=32..32768$ 求 $\lfloor(t-32)/16\rfloor+1+\min(1024,t)+\min(512,t)$ 的平均。答：约 2541，与论文 2560 差 0.7%；直观：压缩部分平均约 $16384/16=1024$，加选择 1024 与窗口 512 得 2560。

3. **（推导）式 (9) 的权重。** $\ell_s/\delta_c=4$、$\ell_c/\delta_c=2$，写出 $\mathbf p^{\text{slc}}[j]$ 展开式。答：$\mathbf p^{\text{cmp}}[4j]+2\mathbf p^{\text{cmp}}[4j-1]+2\mathbf p^{\text{cmp}}[4j-2]+2\mathbf p^{\text{cmp}}[4j-3]+\mathbf p^{\text{cmp}}[4j-4]$；解释：一个选择块与 5 个压缩块重叠，重叠程度不同。进阶：按式 (7)(12) 的下标约定检查哪些压缩块真正落在选择块里（答：$4j..4j+2$ 完全包含，$4j-1$ 与 $4j+3$ 半重叠——与式 (9) 差一个块）。

4. **（计算）DSA 的 token 数与“128K 到底是多少”。** 答：$1000\times16\times131072=2.097\times10^9$，$15000\times480\times131072=9.437\times10^{11}$；若 128K = 128,000 则得 2.048B 与 921.6B，与论文 943.7B 不符 → 论文的 128K = $2^{17}$。

5. **（推导）“按头求和再 L1 归一化”= 头平均。** 答：每头注意力行和为 1，$n_h$ 个头求和后行和为 $n_h$，除以 $n_h$ 即平均（§2.2）。追问：若改成“按头取最大再归一化”会有什么不同？（开放题：更偏向被某一个头强烈关注的 token。）

6. **（手算）lightning indexer。** $H^I=2$，$d^I=2$，$\mathbf q^I_{t,1}=(1,0)$，$\mathbf q^I_{t,2}=(0,1)$，$w=(0.5,1.0)$；键 $\mathbf k_1=(2,-1)$，$\mathbf k_2=(-1,3)$，$\mathbf k_3=(1,1)$，$\mathbf k_4=(-2,-2)$。求 $I_{t,s}$ 与 top-2。答：$I=(1.0,\ 3.0,\ 1.5,\ 0)$，选 $\{2,3\}$。追问：$w$ 可以为负吗？（可以，$w^I_{t,j}\in\R$，由线性投影得到，某个头可以“投反对票”；ReLU 只作用在点积上。）再追问：ReLU 让什么信息丢了？（负相关的 token 在该头上一律记 0，不能区分“无关”和“强烈相反”。）

7. **（计算）KL 损失。** 接上题，两头主注意力 $a_1=(0.10,0.70,0.15,0.05)$、$a_2=(0.20,0.40,0.35,0.05)$。求 $p$、$\operatorname{Softmax}(I)$ 与 $\KL(p\|\operatorname{Softmax}(I))$；再求稀疏阶段只在 $\{2,3\}$ 上的 KL。答：$p=(0.15,0.55,0.25,0.05)$；$\operatorname{Softmax}(I)=(0.0961,0.7101,0.1584,0.0354)$；KL = 0.0576；稀疏：$p_{\mathcal S}=(0.6784,0.3216)$（每个头在 $\{2,3\}$ 内归一后平均；旧稿的 $(0.6875,0.3125)$ 是先平均再截断归一，见 §11.E），$q_{\mathcal S}=(0.8176,0.1824)$，KL = 0.0557。

8. **（计算）每 token FLOPs 与交叉点。** V3 维度，只算核心注意力（每乘加 2 FLOPs）。答：见 §4.3 表；128K 时稠密 MHA 模式 10.74 GFLOPs/层，DSA 核心 0.570 + indexer 2.147 = 2.718 GFLOPs/层，比值 3.95；交叉点 6,963 / 8,704 / 2,176；indexer = 核心于 34,816。

9. **（概念 + 计算）为什么短序列 prefill 要用 masked MHA 模式？** 答：MQA 模式每键 278,528 FLOPs，MHA 模式 81,920，比值 3.4；当 $T\lesssim7$K 时稠密 MHA 的 $81920T$ 小于稀疏 MQA 的 $278528\times2048$；成本图 prefill 拐点 6.589K 与之吻合（笔者推测的解释）。

10. **（计算）DSA 的缓存是变大还是变小？** 答：每 token 每层 MLA FP8 656 B + indexer 键 132 B = 788 B，比不带 indexer 多 20.1%；128K 请求 61 层：4.88 → 5.87 GiB。decode 每步读：稠密 FP8 $656T$，DSA $656\cdot2048+132T$，128K 时 86.0 MB vs 18.6 MB（4.61×）。结论：省读取、不省存储。

11. **（证明）Hadamard 旋转不改变点积。** $\mathcal H=H_{128}/\sqrt{128}$，$H_{128}H_{128}^\top=128I$ ⇒ $\mathcal H^\top\mathcal H=I$ ⇒ $(\mathcal H\mathbf q)\cdot(\mathcal H\mathbf k)=\mathbf q^\top\mathcal H^\top\mathcal H\mathbf k=\mathbf q\cdot\mathbf k$。数值例：一个 128 维标准高斯向量把第 6 个分量改成 40，旋转后最大绝对值降到约 5–8（结果随随机抽样而变；200 次抽样范围 5.4–7.8、均值约 6.2，Python 复算；直观上 $40/\sqrt{128}\approx3.5$ 再叠加高斯分量的最大值）——离群值被摊平，更适合 FP8 的有限动态范围（量化收益为笔者推测，代码未注释；§11.E 的复算显示，这个例子里旋转对 FP8 E4M3 没有好处）。

12. **（读图）成本曲线。** 用 §4.5 顶点：(a) 128K 位置 prefill 与 decode 的成本比；(b) V3.2 在什么位置开始比 Terminus 便宜；(c) 一整段 128K prefill 的平均每百万 token 成本比。答：(a) 3.56×、8.65×；(b) prefill ≈ 10.4K、decode ≈ 4.8K；(c) 2.56×。追问：为什么 (c) 比 (a) 小？（平均要把短位置也算进去，那里两者接近甚至 V3.2 更贵。）

13. **（概念）$T\le2048$ 时 DSA 与稠密注意力的关系。** 答：代码中 `topk(min(index_topk, end_pos))` 会选中全部可见 token，DSA 退化为完全相同的稠密注意力（只多付了 indexer 的钱）；因此 DSA 对短上下文任务的质量影响在机制上为零，差异只来自续训与后训练。

14. **（比较）NSA vs DSA 设计对照表（让学生填）。** 粒度（64-token 块 vs 单 token）；选择打分来源（复用压缩分支注意力分数，无额外参数 vs 独立 indexer，约 0.87B 参数）；是否有局部窗口（有，512 vs 无强制窗口）；共享粒度（GQA 组 16 头 vs MLA 全部 128 头）；训练方式（从头原生训练 vs 从稠密检查点续训 + KL 蒸馏）；每查询预算（$T/16+1536$，随 $T$ 线性增长 vs 固定 2048）；打分成本（压缩分支本身 $O(T^2/16)$ 的完整注意力 vs indexer $O(T^2)$ 但 FP8、单键、无 softmax/V）。

---

## 7. 勘误与常见误解

1. **“NSA 是 ACL 2025 最佳论文”要加限定：是四篇 Best Paper 之一。** ACL 2025 Awards 页列出 4 篇 Best Paper，NSA 为其一；Anthology 条目 Award 栏为 “Best Paper”。课程地图写“ACL 2025 最佳论文”，课文应写“ACL 2025 最佳论文奖（四篇之一）”。

2. **NSA 训练 token 数自相矛盾：260B（§1）vs 270B（§4.1）。** 三个版本都未修正。课文用 §4.1 的实验设定原文（270B，8K 长），并脚注矛盾。

3. **NSA 的 11.6× 解码加速是“期望加速”，不是实测延迟。** 来自表 4 的访存量之比；表行名 “Expected Speedup”。图 1 与 §5.2 正文把它和实测的 9.0×/6.0× 并列，容易误读。9.0×/6.0× 也只是 A100 上 Triton 注意力核的对比，不是端到端训练加速。

4. **NSA 的 $n=16$ 不是 16 个“动态选出”的块。** 含固定的 1 个开头块 + 2 个本地块；并且本地块与 512 窗口有重叠，表 4 的计数是上界（“at most”）。

5. **NSA 式 (7)/(8)/(9)/(12) 的下标约定不一致。** (a) $\tilde K_t^{\text{cmp}}$ 声明的列数 $\lfloor\frac{t-l}{d}\rfloor$ 比式 (7) 的元素个数少 1；(b) 式 (9) 的下标范围与按式 (7)(12) 推出的“重叠压缩块”差一个块；(c) v1→v2 把式 (9) 的 “+m+n” 改成 “−m−n”、把式 (7) 的 $1\leqslant i$ 改成 $0\leqslant i$。课文讲机制（按重叠加权），不要逐下标照抄。另有笔误：式 (12) 后称 $\tilde K_t^{\text{slc}}$ 为 “compresion keys”（应为被选的原始键）。

6. **NSA v2 HTML 版 §6.1 写 “As shown in Figure 8, both methods exhibited inferior loss”，PDF（v1、v2）与 ACL 版均为 “Figure 7”。** 这是 LaTeXML 渲染的交叉引用错误；以 PDF 为准（图 7 是损失曲线，图 8 是注意力可视化）。

7. **NSA 在 arXiv 与 ACL 的图号、节号不同。** 如 NIAH：arXiv 图 5 / ACL 图 4；“Rethinking”：arXiv §2 / ACL 附录 B；ACL 表 4 缺 32K 列。引用时注明版本。

8. **“DSA 就是 NSA 上线版”——错。** 差别见 §6 第 14 题：token 级 vs 块级；独立 indexer vs 复用压缩分数；无窗口/压缩分支；续训 + KL 蒸馏 vs 原生训练；MLA MQA 模式全头共享 vs GQA 组共享。二者的共同点是“KV 条目必须被多个查询头共享”这一硬件原则，DSA 论文明确引 NSA 作为这一原则的出处（C25）。

9. **“lightning indexer 只有几个头”——不准确。** 论文只说 “a small number of heads”；`config.json` 为 **64 头 × 128 维**（主注意力 128 头）。它便宜的真正原因是：只有一个共享键向量（indexer 自身是 MQA 结构）、没有值和 softmax、在 FP8 上算、每键 FLOPs 只有 MQA 模式核心注意力的 1/17。课程地图“少量头”应改为“64 个头但每头极轻”。

10. **“DSA 把注意力复杂度降到 $O(Lk)$”只对核心注意力成立。** indexer 仍是 $O(L^2)$（论文原话）。按 §4.3，128K 时 indexer 的 FLOPs 已是核心注意力的 3.8 倍，交叉点 34,816。

11. **“DSA 减小了 KV cache”——错。** 所有 MLA 条目都必须保留（任何历史 token 都可能被选中），还要额外存 indexer 键（FP8，132 B/层/token，+20%）。DSA 减小的是每步读取量与计算量。

12. **“图 3 是 API 价格”——错。** 是按 H800 租价 2 USD/GPU·h 从实际服务基准折算的内部成本估计（C33）。API 价格另行公布：2025-09-29 起输入未命中 $0.28、命中 $0.028、输出 $0.42（降 50%–75%）。

13. **“DSA 在任何长度都更便宜”——错。** 按图 3 顶点，prefill 在约 10.4K 以下、decode 在约 4.8K 以下 V3.2 的估计成本略高于 Terminus（位置 0 处 prefill 0.0611 vs 0.0529，decode 0.0952 vs 0.0705）。

14. **“每个头各选 2048 个 token”——错。** 每个**查询 token** 选一组 2048 个，被该 token 的全部 128 个头共享（MQA 模式）。

15. **“indexer 和主模型一起用语言模型损失训练”——错。** indexer 输入被 detach，只接收 $\mathcal L^I$ 的梯度；主模型只接收 LM 损失的梯度（C29）。

16. **“稠密预热阶段训练全模型”——错。** 只训 indexer，其余参数全部冻结，注意力保持稠密（C27）。

17. **DSA 式 (2) 中的 $\mathbf c_s$ 不是 $\mathbf c_s^{KV}$。** 是 MLA 的完整 KV 条目 $[\mathbf c_s^{KV};\mathbf k_s^R]$（576 维），图中标为 $\{[\mathbf c_t^{KV};\mathbf k_t^R]\}$。

18. **论文与代码的差异（paper vs code）。** (a) 代码的 indexer 分数多了 $1/\sqrt{H^I}$ 与 $1/\sqrt{d^I}$ 缩放；(b) 代码对 q、k 做 Hadamard 旋转再量化，论文未提；(c) 代码的 indexer 查询来自查询潜向量 $\mathbf c_t^Q$，论文只说“derived from the query token $\mathbf h_t$”；(d) 代码只对 indexer 前 64 维加 RoPE，且布局为 non-interleaved（2025-11-17 之前的 demo 用错了布局；README 原话是 “potentially leading to degraded model performance”，即**可能**降低性能，未给量化结果）；(e) demo 的 prefill 分支用 MHA 模式 + 掩码实现 DSA，decode 分支用 MQA 模式，但**两个分支都先对全部可见位置算稠密分数再加 $-\infty$ 掩码**——数学等价但 prefill 与 decode 都**没有任何加速**，真正的稀疏核在 FlashMLA（稀疏 prefill 与 decode 均为 MQA 模式）与 DeepGEMM（indexer logits 核）；(f) demo 的 KV cache 以 bf16 存放、只“模拟” FP8 精度，部署时为 656 B 的 FP8 格式。

19. **“DSA 首次实现细粒度稀疏注意力”是 README 的宣传语，不是技术报告的结论。** token 级选择在推理期方法中早已存在（NSA 表 2 的 Exact-Top；§7.3 列出的 HashAttention、MInference、TokenSelect）。可以说“首个在前沿规模模型上端到端续训并部署的 token 级稀疏注意力”，但即便这一表述也需标“DeepSeek 自称”。

20. **V3.2 与 V3.2-Exp 的 DSA 没有区别。** V32 明说架构完全相同；成本图数据也逐点相同。别把 V3.2 当成“改进版 DSA”。

21. **V3.2-Exp 技术报告与 V3.2 论文在后训练描述上的漂移（属第 15 课，但顺带记录）。** V32Exp：专家蒸馏覆盖 “five specialized domains”（数学、竞赛编程、通用逻辑推理、智能体编程、智能体搜索）；V32：“six specialized domains”（增加 general agentic tasks，且“竞赛编程”改为 “programming”）。

22. **FlashMLA 博客 8.72 GiB 的算式是 576 × 2 × 62 × 128 × 1024，即 62 层 × 2 字节（笔者解读为含 1 个 MTP 层、BF16；博客未解释各因子）。** 按 61 层 BF16 为 8.58 GiB；按部署的 FP8 格式 61 层为 4.88 GiB。引用时说明口径。

23. **“128K”在 DeepSeek 文档中 = 131,072。** 由 943.7B 的复算确认（§4.2）。NSA 的“64k”同理按 65,536（表 4）。

24. **成本图不是密集测量曲线。** 矢量数据显示每条曲线只有 2–4 个顶点（分段直线）。课文重绘时应说明“据论文图 3 的折线重绘”，不要暗示有更细的实测点。

---

## 8. 未能核实 / UNVERIFIED

1. NSA §1 “注意力占 64K 解码延迟 70–80%”的“理论估计”——论文未给推导、硬件、模型配置或引用，无法复算。
2. NSA §2.2 转引的 “top 20% attention can only cover 70% of the total attention scores”——未回到 MagicPIG（arXiv 2410.16179）原文核对其条件（模型、层、长度）。
3. NSA 是否公开了官方训练/推理代码——未找到 DeepSeek 官方仓库；社区复现（如各类 “native-sparse-attention” 仓库）未核对，不能引用为官方实现。
4. NSA 三分支“独立的键和值”带来的 KV cache 增量——论文未量化；选择分支与窗口分支是否共享同一份原始 K/V 缓存不明。
5. NSA 门控 MLP 的结构（层数、输入是 $\mathbf h_t$ 还是 $\mathbf q_t$）与压缩 MLP $\varphi$ 的结构、块内位置编码的形式——论文未给细节。
6. NSA 核块大小 $B_k$ 的具体取值；压缩与窗口分支是否与选择分支融合在同一核内（图 6 分开计时，暗示未融合）。
7. DSA 的 $H^I=64$、$d^I=128$、Hadamard、LayerNorm、查询来自 $\mathbf c_t^Q$ 等细节**只来自推理代码与 config**，训练时是否完全相同无法从论文确认。
8. 式 (4) 的 $p_{t,\mathcal S_t}$ 是否在 $\mathcal S_t$ 上重新归一化；目标分布是按层各自对齐（每层 indexer 对齐本层主注意力）——代码显示每层各有 indexer，但训练代码未公开，论文未明说。
9. 预热/稀疏阶段的优化器、batch 内序列打包方式、warmup/衰减调度、indexer 在预热阶段的初始化方式——论文只给 lr、步数、每步序列数。
10. 训练时 top-$k$ 的反向与 MQA 模式稀疏核在训练中的实现（论文只说续训用稀疏注意力，FlashMLA 公开的是前向核；稀疏反向核未见公开）。
11. 成本图的具体测量方法（batch、并行方式、是否含 MTP 投机解码、PD 分离配置）及“K”是 1000 还是 1024——论文未说；图中拐点 6.589K、2.056K 的来源（是否就是 masked-MHA 切换点与 $k$）是笔者推测。
12. V32Exp 表 1 题注“用 token 数相当的中间检查点时差距消失”——未给数据。
13. V32 的 AA-LCR “高 4 分”、Fiction.liveBench “consistently outperforms”——第三方网页，未抓取原始榜单核对。
14. DeepGEMM PR #200（indexer logits 核）与 TileLang 示例的性能数字——未读。
15. V4 的 CSA/HCA 是否直接复用 lightning indexer（第 18 课需核）；V4.1 FlashMLA README 提到的 V4/V4.1 KV 格式（584/528/288 B）属后续课，只作对照，未深入核对。
16. SGLang/vLLM 等第三方推理框架中 DSA 实现与官方的一致性——未核。
17. NSA ACL 版与 arXiv v2 的全部正文差异——只比对了关键数字、公式、图号与若干句子，未做逐字全文 diff。
18. “NSA 被用于 DeepSeek 后续模型”这类说法——没有找到 DeepSeek 一手来源说明 NSA 本身被部署；V3.2 用的是 DSA 而不是 NSA，课文不要写“NSA 已上线”。
19. （核查轮新增）NSA v1 的式 (7) 下界 $1\leqslant i$、式 (8) 无 $+1$、式 (9) 的 “+m+n”：核查轮用 `pdftotext -layout` 独立比对 v1/v2 字形，结果一致（v1 为 “1 −”“+ +”，v2 为 “0 −”“− −”“+1”），但本机无 PDF 渲染器，**仍未目视核对**。
20. （核查轮新增）图 6 的图例与颜色对应（橙 = “Sparse Attention Kernel” = 选择核，绿 = 压缩 + 窗口）是按 SVG 图例色块的上下顺序与 PDF 抽出的图例文字顺序配对得到的；图例文字在 SVG 中是字形轮廓，未直接读出。
21. （核查轮新增）§5.1 中图 1 左图的三组数字、图 4 损失曲线“约 4.0 → 1.65”、图 7 终值“约 2.37–2.40”、图 8 色标范围——核查轮未重新从原图核对。

---

## 9. 参考文献卡片草稿

```
- id: yuan2025nsa
  title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
  a: "Yuan, Gao, Dai, Luo, Zhao, Zhang, Xie, Wei, Wang, Xiao, Wang, Ruan, Zhang, Liang, Zeng (2025), ACL 2025 (Long Papers) pp. 23078–23097, Best Paper; arXiv:2502.11089v2"
  url: https://aclanthology.org/2025.acl-long.1126/
  finding: "提出压缩、块选择、滑动窗口三分支加门控的原生可训练稀疏注意力，27B 模型上质量不低于全注意力，64K 时注意力核前向 9.0×、反向 6.0×，解码按访存推算 11.6×。"
  matters: "它确立了“KV 条目必须被多个查询头共享、按硬件友好的方式选择”的原则，DSA 直接引用它来论证为何在 MLA 的 MQA 模式上实现。"

- id: deepseek2025v32exp
  title: "DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention"
  a: "DeepSeek-AI (2025), technical report PDF in GitHub repository, no arXiv id"
  url: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
  kind: artefact
  finding: "在 V3.1-Terminus 上通过两阶段续训（2.1B tokens 只训 indexer 的稠密预热 + 943.7B tokens 的 top-2048 稀疏训练）引入 lightning indexer 与 token 级选择，基准基本持平。"
  matters: "它是 DSA 公式、训练超参数与成本曲线的首发来源，也是本课所有 DSA 引文的主引出处。"

- id: deepseek2025v32
  title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
  a: "DeepSeek-AI (2025), preprint, arXiv:2512.02556v1"
  url: https://arxiv.org/abs/2512.02556
  finding: "声明 V3.2 与 V3.2-Exp 架构完全相同，逐字沿用 DSA 描述，并补充 ChatbotArena、AA-LCR 等对等评估与 MLA 的 MHA/MQA 模式附录。"
  matters: "用于确认 DSA 在正式版中没有变化，并为“短序列用 masked MHA 模式”和 MQA 模式提供可引用的同行可见版本。"

- id: deepseek2024v3
  title: "DeepSeek-V3 Technical Report"
  a: "DeepSeek-AI (2024), preprint, arXiv:2412.19437v2"
  url: https://arxiv.org/abs/2412.19437
  finding: "给出 MLA 公式 (1)–(11) 与 V3 维度：61 层、d=7168、128 头、d_h=128、d_c=512、d_c'=1536、d_h^R=64、激活 37B。"
  matters: "本课所有 FLOPs 与字节复算都以这些维度为输入，DSA 正是在这套 MLA 上实例化的。"

- id: deepseek2024v2
  title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  a: "DeepSeek-AI (2024), preprint, arXiv:2405.04434v5"
  url: https://arxiv.org/abs/2405.04434
  finding: "提出 MLA 的低秩 KV 联合压缩与解耦 RoPE。"
  matters: "DSA 技术报告以它作为 MLA 的出处；本课回指第 9 课时引用。"

- id: hf2025v32expcode
  title: "DeepSeek-V3.2-Exp inference demo (model.py, kernel.py, config.json)"
  a: "DeepSeek-AI (2025), Hugging Face repository; indexer RoPE fix 2025-11-17"
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference
  kind: artefact
  finding: "代码给出 indexer 的具体实现：64 头×128 维、查询来自 c^Q、单个共享键、部分 RoPE、Hadamard 旋转、FP8 打分、top-2048。"
  matters: "论文没有给出 H^I、d^I 等数值，本课的规格表与 paper-vs-code 勘误全部依赖这份代码。"

- id: flashmla2025
  title: "FlashMLA: sparse and dense MLA kernels; blog 'A Deep Dive Into The Flash MLA FP8 Decoding Kernel on Hopper'"
  a: "DeepSeek-AI (2025), GitHub repository and docs"
  url: https://github.com/deepseek-ai/FlashMLA
  kind: artefact
  finding: "稀疏 prefill/decode 核均在 MQA 模式运行，V3.2 FP8 KV 为每 token 656 字节，稀疏解码核在 H800 上达 410 TFLOPS，约 3000 长度时与稠密解码核持平。"
  matters: "它把“省多少”落到真实内核与字节格式上，是本课 KV 字节表与“短序列稀疏不划算”讨论的依据。"

- id: deepseekapi2025v32exp
  title: "Introducing DeepSeek-V3.2-Exp (API news, 2025-09-29) and price card"
  a: "DeepSeek (2025), API documentation news page"
  url: https://api-docs.deepseek.com/news/news250929
  kind: artefact
  finding: "V3.2-Exp 发布当日 API 降价：输入未命中 $0.56→$0.28、命中 $0.07→$0.028、输出 $1.68→$0.42（每百万 token）。"
  matters: "用来把论文图 3 的“内部成本估计”与对外价格区分开，避免学生把两者混为一谈。"

- id: acl2025awards
  title: "ACL 2025 Awards"
  a: "Association for Computational Linguistics (2025), conference website"
  url: https://2025.aclweb.org/program/awards/
  kind: artefact
  finding: "ACL 2025 共评出四篇 Best Paper，NSA 为其中之一。"
  matters: "修正“NSA 是 ACL 2025 唯一最佳论文”的常见说法。"
```

---

## 10. 核查记录（2026-09-24，对抗式复核）

复核者独立重新抓取一手来源（不复用研究轮的缓存文件）：NSA arXiv v2 LaTeXML 全文与 v1/v2/ACL 三份 PDF（`pdftotext`）、V3.2-Exp 技术报告 PDF（GitHub raw，`pypdf` 解析矢量流）、V3.2 arXiv v1 LaTeXML 与 PDF、HF `config.json` / `inference/config_671B_v3.2.json` / `model.py` / `kernel.py` / 提交历史 / safetensors 元数据 API、FlashMLA README 与 2025-09-29 博客、DeepSeek news250922/250929（含价格图 `v3_2_price_en.jpeg`）/251201、ACL 2025 Awards 页与 Anthology 条目、arXiv 提交历史与 API 作者列表。所有派生数字用 Python 重算。

**逐项核对且确认无误（23 组）**
1. NSA 超参数 l=32、d=16、l′=64、n=16（含 1 开头块 + 2 本地块）、w=512（v2 §4.1 原句；§5 重申但无括号）。
2. NSA 骨干：27B/3B、30 层、隐藏维 2560、GQA 4 组 64 头、$d_q=d_k=192$、$d_v=128$、72 路由 + 2 共享 top-6、首层 SwiGLU（§4.1）。
3. 260B（§1）vs 270B（§4.1）矛盾：v1、v2、ACL 三份 PDF 均如此。
4. 9.0× 前向 / 6.0× 反向（§5.1，8×A100，Triton 对 Triton-FA2）；图 6 全部 16 个柱高按 SVG 几何（前向 200 ms = 32.571 pt，反向 500 ms = 31.667 pt）逐一重算，与 §4.1 反解表一致到 0.1 ms。
5. 表 4 “Expected Speedup” 4×/6.4×/9.1×/11.6× 与表题 “approximately linear with the volume of memory access”；ACL 表 4 确无 32768 列。
6. NSA 式 (1)–(12) 与 v2 LaTeX 逐符号比对一致（含式 (7) 维数声明的差一、式 (9) 的 “−m−n” 与前提 $l\leqslant l'$、两处 “compresion keys” 笔误）。
7. v2 HTML §6.1 “As shown in Figure 8”，v1/v2/ACL 三份 PDF 均为 “Figure 7”；v2 新增 “mean-pooling … KL divergence” 句（v1 无）；ACL 有 “marginally lower loss” 与 Limitations 原句。
8. ACL 2025 Best Paper 共 4 篇，NSA 为其一；Anthology：pp. 23078–23097、DOI 10.18653/v1/2025.acl-long.1126、Award: Best Paper、CC BY 4.0。arXiv v1 2025-02-16、v2 2025-02-27。
9. 表 1 平均 0.4431/0.4558、NSA 胜 7/9；表 2 平均 0.437/0.4687/0.4234 与 HPQ/2Wiki/LCC/PassR-en 增量；表 3 +0.075/+0.054；2560 预算原句。
10. DSA 式 (1)–(4) 与 V3.2 LaTeX 逐符号比对一致；V3.2-Exp PDF 同句同式号。
11. 预热：lr $10^{-3}$、1000 步 × 16 × 128K = 2.1B；稀疏：lr $7.3\times10^{-6}$、2048、15000 步 × 480 × 128K = 943.7B；仅 128K = 131,072 能复现 943.7B。
12. detach 原句；“exactly the same architecture”；“only architectural modification”；“small number of heads”。
13. `config.json`（V3.2-Exp 与 V3.2 相同）：`index_n_heads 64`、`index_head_dim 128`、`index_topk 2048`、128 头、61 层 + 1 nextn、163840、YaRN 40/4096。
14. `model.py`：`wq_b` 输入为 `q_lora_rank`（1536），单个 `wk` 键 + `LayerNorm`，`weights_proj` 注释 bf16，`n_heads**-0.5` 与 `head_dim**-0.5`，前 64 维 non-interleaved RoPE，`rotate_activation` = Hadamard × $128^{-1/2}$，`topk(min(index_topk,end_pos))`；`kernel.py::fp8_index` 为 ReLU → × 权重·q 缩放 → 按头求和 → × k 缩放。V3.2 的 `model.py` 与 V3.2-Exp 逐字节相同（38,639 B，`cmp`）。
15. HF safetensors：四个模型的 BF16/F8/F32/total 与 §3.7 完全一致；F8 差额 837,025,792 = `(1536*8192+7168*128)*62`，BF16 差额 = `7168*64*62`，F32 差额 = 62 × 1080。
16. 成本图：自行从 V3.2-Exp PDF（`/Im5`、`/Im6`）与 V3.2 arXiv PDF（`/Im4`、`/Im5`）解析折线，两份 PDF 原始坐标逐点相同、仅图例文字不同；顶点、3.56×/8.65×、64K/32K/4K 比值、交叉点 10.39K/4.76K、平均 2.56×/5.67×、斜率、GPU 时间全部重算一致。
17. 成本图口径原句（H800、2 USD/GPU·h、masked MHA）；V3.2-Exp 表 1 全部 14 组数字与表题。
18. FLOPs 模型（81,920 / 278,528 / 16,384 每键）、§4.3 表全部格、交叉点 6,963 / 8,704 / 2,176 / 34,816、整段 prefill 求和、注意力占比、14,808 交点。
19. KV 字节：FlashMLA README 的 656 B 格式（512 FP8 + 4×FP32 + 64×BF16）、`k_cache` FP8 + `k_scale_cache` FP32（`block_size = 128`）→ 132 B；4.88/5.87/8.58/8.72 GiB、86.0 vs 18.6 MB 等重算一致。
20. FlashMLA：640/410 TFlops（H800 SXM5）、稀疏 prefill/decode 均为 MQA 模式、`head_dim_k 576/head_dim_v 512`；博客 4096 FLOPs/SM/周期、34 与 50 周期、250→410 TFLOPS（batch 128、128 头、s_q=2、topk=2048）、“around 3000”原句。
21. API 价格图：$0.07→$0.028、$0.56→$0.28、$1.68→$0.42，“Effective from 10:00 UTC on September 29, 2025”；Terminus 临时 API 至 2025-10-15 15:59 UTC。
22. 时间线：news250922（2025/09/22）、HF 提交 2025-09-29（demo、README）与 2025-11-17 “fix indexer rope”、news251201、arXiv 2512.02556v1 2025-12-02；V3.2-Exp PDF CreationDate 2025-09-29 08:49:11Z。
23. 算例：indexer 手算 $I=(1,3,1.5,0)$、KL 0.0576 / 0.0491（后者已更正为 0.0557，见 §11.E）、式 (9) 权重 (1,2,2,2,1)、Hadamard 正交性。

**本轮修改（10 处）**
1. C15：32K 平均激活 token 由“2540，一致到 1%”改为“2541（精确 2540.96），差 0.7%”，与 §4.1、§6 第 2 题统一（Python 重算）。
2. §0 V32 行：“Aixin Liu 等 264 人”改为“作者列表 264 条 = 机构名 DeepSeek-AI + 263 位个人”（arXiv API 作者列表首条为 “DeepSeek-AI”）。
3. §3.3 `scale_fmt`：`config.json` 中该键嵌套在 `quantization_config` 内（顶层为 None），顶层键只出现在 `inference/config_671B_v3.2.json`；Terminus 的 `quantization_config` 无此键。
4. §3.3 与 §7 第 18(e) 条：demo 的 **decode** 分支同样先对全部 `:end_pos` 位置算稠密 MQA 分数再加 −∞ 掩码，因此 prefill 与 decode 都无加速；原文只说 prefill 无加速，会让人误以为 decode demo 是真稀疏（`model.py` L590–605）。
5. §7 第 18(d) 条：“官方承认会降低性能”改为 README 原话 “potentially leading to degraded model performance”（只说可能）。
6. C40 与 §7 第 22 条：8.72 GiB 算式中“62 = 61 层 + MTP、×2 = BF16”改标为笔者解读；博客只给算式。
7. §2.2 `mscale²`：补上生效条件（`max_seq_len > original_seq_len`；demo 默认 16384 > 4096）。
8. §3.3、§4.6：HF total 差额 865,535,376 比 indexer 参数 865,484,288 多出 51,088，补注为 62 × (768 + 56) 个 FP32 缩放张量，避免“total 差额吻合”的误读。
9. §3.3 top-k 出处：补为 “V32 §2.1.1 / V32Exp §2.1”（V32Exp 无 §2.1.1）。
10. §6 第 11 题：Hadamard 例“降到约 5.8”依赖未给出的随机种子，改为“约 5–8（200 次抽样 5.4–7.8，均值约 6.2）”。

**未能在本轮消除的疑点**：已并入 §8 第 19–21 条（v1 式号仅字形比对、图 6 颜色与图例配对靠顺序、图 1/4/7/8 的读图数字未重核）；§8 第 1–18 条维持原状。

---

## 11. 补充核实（写课时追加，2026-09-24）

写 `lesson-14-sparse-attention.md` 时用到、而上文没有逐条列出的事实与派生数字。一手来源的引文保持英文原文；派生数字均已在 Python 中复算。

**A. 一手来源引文**

1. **NSA 核设计与 FlashAttention 的对照**（NSA arXiv v2 §3.4，本地抓取的全文 `nsa.txt` 第 354 行）：
   “If we were to follow FlashAttention’s strategy of loading temporally continuous query blocks into SRAM, it would result in inefficient memory access since queries within a block may require disjoint KV blocks.”
   → 课文 §3“组中心的核”一节据此写“FlashAttention 让一段连续的查询共用 KV 块；NSA 改为让同一位置、同一组的 16 个头共用”。
2. **FP8 E4M3 的格式**：1 位符号、4 位指数、3 位尾数，最大正规数 $1.75\times2^8=448$。来源：数学课研究笔记 `math/reference-docs/math-refs-doc.md` C63（Micikevicius et al., “FP8 Formats for Deep Learning”, arXiv:2209.05433, 表 1），与第 12 课研究笔记中 V3 推理代码 `s = amax / 448.` 一致。课文 §4 的 FP8 前置框据此写“最大值约 448”；“相邻可表示数的相对差距约 6%–12%”是由 3 位尾数推出的：同一 binade 内间距为 $2^{-3}$ 倍的 $2^e$，相对值在 $2^{-3}/2=6.25\%$ 到 $2^{-3}=12.5\%$ 之间（笔者推导）。

**B. 课文中的派生数字（笔者计算，均已复算）**

| 课文位置 | 量 | 表达式 | 值 |
|---|---|---|---|
| §1 例题 | 整段 128K 预填充：注意力 / 其余部分 | `61*81920*131072*131073/2` / `74e9*131072` | 4.29e16 / 9.70e15 = 4.43 倍 |
| §1 例题 | 128K 解码一步读 BF16 KV（61 层） | `1152*131072*61` | 9.21e9 B ≈ 9.2 GB |
| §2 小练习 | $t=16{,}384$ 的精确 $N_t$ | `(16384-32)//16 + 1 + 1024 + 512` | 2559 |
| §3 例题 | 组内各选各的最坏并集（64K） | `3 + 16*13` 块；`211*64`；`13504/65536` | 211 块，13,504 token，20.6%；约为共享时的 13 倍 |
| §3 例题 | 组中心核的算术强度（按 BF16 假设） | `16*64*640 / (64*(192+128)*2)` | 16 FLOPs/B；单头读取为 `640/640` = 1 |
| §3 例题 | 块分数算例 | `1*.02+2*.10+2*.30+2*.05+1*.01` | 0.93 |
| §5 例题 | MQA 模式算术强度 | `278528/1152`，`278528/656` | 241.8（BF16），424.6（FP8 格式）FLOPs/B |
| §7 例题 | DSA 读取与稠密 FP8 读取的交叉点 | `656*2048/(656-132)` | 2563.9 |
| §7 失败模式 | 按 V3 维度外推到 1M（假设） | `16384*2**20`，`/ (278528*2048)` | 17.18 GFLOPs/层/查询，是 DSA 核心的 30.1 倍 |
| §6 例题 | KL 单项 | `p*ln(p/q)` | 0.0668, −0.1405, 0.1140, 0.0173；和 0.0576 |
| 练习 10 | 反向 KL $\KL(q\Vert p)$ | 同例 | −0.0428 + 0.1814 − 0.0723 − 0.0122 = 0.0541 |
| 练习 8 | 预填充交叉点（按顶点直线求交） | `0.0529+0.004788x = 0.0952+0.000718x` | x ≈ 10.39K |
| §6 | 学习率比、预热占比 | `1e-3/7.3e-6`；`2.097/(2.097+943.7)` | 137；0.22% |
| §1 推导（教学复核轮追加） | 稀疏注意力的输出误差上界 | $\tilde a_s=a_s/m$ ⇒ $\norm{\mathbf o-\tilde{\mathbf o}}\le2\varepsilon\,v_{\max}$，$\varepsilon=1-m$ | 纯数学推导（三角不等式），不是论文结论 |
| §1 例题 | 集中 / 弥散两种情形的上界 | `2*0.05`；`2048/131072`，`2*(1-2048/131072)` | 0.1 $v_{\max}$（95% 为假设数字）；1.56%，1.97 $v_{\max}$ |
| 练习 1(b) | 32K 处 80% / 均匀 | `2*0.2`；`2048/32768`，`2*(1-0.0625)` | 0.40 $v_{\max}$；6.25%，1.875 $v_{\max}$ |
| §4 | Hadamard 小例 | $H_2/\sqrt2\,(40,0)^\top$ | (28.28, 28.28)；Sylvester 构造 $H_{2n}=[H_n,H_n;H_n,-H_n]$ 为标准线性代数事实 |

**C. 课文里明确标为“笔者推测/解读”的地方**（便于以后核对或替换）：“数一篇文档里某个词出现多少次”作为弥散注意力任务的例子（§1，笔者举例）；ReLU 为何利于吞吐的解释；Hadamard 旋转服务于 FP8 量化；前向 KL 与“漏选”代价一致；detach 的设计动机；稀疏阶段 KL 只能调整已选 token 排序的分析；成本曲线拐点 6.589K/2.056K 与 6,963 / $k$ 的对应；预热阶段大学习率的理由；NSA 放弃路线在 DSA 中“翻盘”的原因；FP8 矩阵乘峰值吞吐高于 BF16（硬件常识，未对具体 GPU 核对）。

**D. 数学复核轮追加（2026-09-24）**：全部由 Python 复算，脚本不入库。

| 课文位置 | 量 | 方法 / 表达式 | 值 |
|---|---|---|---|
| §2 压缩分支 | 重叠块“总有一块完整包住一句话”的长度上限 | 块 $[16b+1,16b+32]$，对起点 $a=1..2000$ 暴力枚举 | 长度 ≤ 17（$=\delta_c+1$）恒成立，18 起失败；不重叠（步长 32）时只有长度 1 恒成立 |
| §4 FP8 前置框 | E4M3 正规区内相邻两数的相对差距 | 间距 $2^{-3}\cdot2^e$ 除以较小者 / 较大者 | 12.5%–6.67% / 11.1%–6.25%，合起来即 A.2 的“6.25% 到 12.5%”（课文原写“约 6%–12%”，改为这个区间）；舍入相对误差 ≤ $2^{-4}=6.25\%$ |
| §4 FP8 前置框 | E4M3 正规区跨度；冲零阈值 | $448/2^{-6}$；$2\times448/2^{-9}$ | 28,672 倍；小于最大值的 1/458,752 被舍入成 0 |
| §4 例题、练习 12 | 128 维 N(0,1) 向量、一个分量改成 40，每 128 个数一个缩放因子（最大值映到 448 / 127）；其余分量相对误差中位数，1000 次抽样 | 自写 E4M3（含次正规数、就近舍入）与 INT8 量化器；Sylvester $H_{128}/\sqrt{128}$ | INT8：旋转前 10.9%（12.5% 的分量为 0），旋转后 0.34%；E4M3：旋转前 2.16%、旋转后 2.18%，都没有 0。离群值 $10^4$：E4M3 不旋转 1.7% 为 0；$10^5$：17.1% 为 0（与 M08 §6 的 17.4% 一致），旋转后 0。**口径有误，已更正，见 §11.E**：旋转后的百分数是在旋转坐标里量的 |
| §7 质量 | V3.2-Exp − V3.1-Terminus（表 1） | 逐项相减 | HMMT −2.5、HLE −1.9、Aider-Polyglot −1.6、GPQA −0.8、LiveCodeBench −0.8、SWE Verified −0.6；除 Codeforces 外 13 项中 8 项在 ±1 以内 |
| §7 (14.24) | 交叉点公式的成立条件 | 推导中假设核心项已封顶 | 需 $F_{\text{dense}}>F_{\text{idx}}$ 且 $T^*>k$；2176、8704 均 > 2048 |
| §7 字节 | 以 BF16 稠密为基准的读取交叉点（对照用） | `656*2048/(1152-132)` | 1317；FlashMLA 博客的“约 3000”比较的是两个注意力核（不含 indexer、稀疏核受反量化限制），不能用来验证 2564 |
| 练习 8 | 预填充 V3.2 第一段上无交点 | 两端比较 | $x=0$：0.0611 > 0.0529；$x=6.589$：0.0999 > 0.0844 |

课文据此修改：§4 FP8 前置框与 Hadamard 例题不再说“离群值 40 会让 FP8 的其余分量量化变粗”（这是 INT8 式的推理，与 M08 §6 的结论相反），改为给出上表的 INT8 / E4M3 对照；“为了量化而旋转”仍标为笔者推测。

**E. 数学复核第二轮（2026-09-24）**：课文全部数字重新用 Python 复算（脚本不入库），除下列几处外全部一致。

| 课文位置 | 问题 | 方法 / 表达式 | 更正 |
|---|---|---|---|
| §4 例题、练习 12(b) | D 表的“旋转后 0.34% / 2.18%”是在**旋转后的坐标**里量的相对误差（那些坐标都在 3.5 左右），与“旋转前其余 127 个分量”的误差不可比 | 旋转后量化再乘回 $\mathcal H^\top$，在原坐标里比较 127 个普通分量；另用与随机 N(0,1) 键的点积误差 $\abs{\hat q\cdot\hat k-q\cdot k}/(\norm q\norm k)$ 交叉检查；1000 次抽样 | 离群值 40：INT8 10.8% → **1.4%**（点积误差 1.57e-3 → 4.5e-4）；E4M3 2.2% → **9.9%**（点积误差 1.25e-3 → 2.3e-3，旋转反而更差）。离群值 $10^5$：E4M3 不旋转 17% 为 0；旋转后每个坐标约 ±8840，E4M3 在 448 附近的格距 32 对应原单位约 630，普通分量整个被舍掉，原坐标误差中位数 100%——“旋转后一个也没有变成 0”只在旋转坐标里成立，不代表信息被保住。旋转后最大绝对值：1000 次抽样 5.18–7.81，均值 6.12 |
| §6 (14.22)、图 14.11(b) | 旧稿把稠密平均 $p$ 截到 $\mathcal S=\{2,3\}$ 再归一化，得 $(0.6875,0.3125)$、KL 0.0491；但课文对 (14.22) 的理解是“稀疏阶段主注意力每个头只在 $\mathcal S_t$ 上做 softmax 再平均”，两者一般不等 | 头 $i$ 的稀疏权重 $a_{i,s}/m_i$（§1 推导），$m_1=0.85$、$m_2=0.75$；截断归一化等于按 $m_i/\sum m$ 加权平均 | $p_{\mathcal S}=(0.6784,0.3216)$，KL = **0.0557**（梯度 $\pm0.139$）；图 14.11 生成脚本与说明已改 |
| §6 稀疏阶段分析 | “只能调整已选 token 的排序”可以写成精确的式子 | $\sum_{u\in\mathcal S_t}(q_u-p_u)=0$ | 损失对 $\mathcal S_t$ 内分数的整体平移不变；间接的参数共享效应另作说明 |
| §7 (14.23) 注 | 原写“与 $T$ 无关的投影、吸收乘法稠密与稀疏一样多”不成立：indexer 的三个投影只有 DSA 有 | `2*(1536*8192+7168*128+7168*64)`；MHA 模式展开 $W^{UK},W^{UV}$ 与 MQA 模式吸收各 `2*128*128*512` | indexer 投影 27.9M FLOPs/token/层 ≈ 稠密 MHA 模式 341 个键；展开与吸收两边都是 2 × 16.8M，确实相等 |
| §2 (14.7) | 原文 $\mathbf k_{t-w:t}$ 按课文“两端都含”的下标约定是 $w+1$ 个 | — | 改写为 $\mathbf k_{t-w+1:t}$ 并注明 |
| 练习 10(b) | 四项四舍五入后相加为 0.0540，与总和 0.0541 对不上 | 各项保留 5 位 | $-0.04279+0.18142-0.07226-0.01225=0.05412$ |
| 练习 11(b) | “权重和 = 8”缺前提 | 交换求和次序；键 $x$ 属于块 $b$ ⇔ $x/\delta_c-2\le b<x/\delta_c$ | 需 $j\ge1$ 且块 $4j+3$ 已凑满；否则 7 或更少 |
| 练习 9(b) | “取最大后 KL 目标更尖锐”不是一般成立的结论 | 反例：各头都近似均匀、其中一头集中时变尖；两头各集中在不同 token 时不变或变平 | 改为具体描述两种目标的差别 |
| 图 14.15 说明 | 解码的绿线有两个内部顶点（2.056K 与 101.607K），原说明只提一个 | 两段斜率 0.000821 / 0.000822 | 说明里补上 101.607K |

**F. 事实与引用复核轮（2026-09-24）**：逐条核对课文中的事实性陈述与 `[@key]` 挂靠；以下为本轮新抓取或新确认的一手来源（NSA 全文用 `arxiv.py text 2502.11089` 重新抓取的 v2 LaTeXML；其余为 raw 下载或 WebFetch）。

1. **NSA 图 5（大海捞针）只画了 NSA。** 正文：“As shown in Figure 5, NSA achieves perfect retrieval accuracy across all positions in 64k-context needle-in-a-haystack”；图题 “Needle-in-a-Haystack retrieval accuracy across context positions with 64k context length. NSA achieves perfect accuracy …”。论文没有给出全注意力模型的大海捞针结果 → 课文 §3 质量表中全注意力一格原写“全部位置满分”，改为“论文未给出”。
2. **NSA §2 的结构。** 2.1 “The Illusion of Efficient Inference”（Phase-Restricted Sparsity；Incompatibility with Advanced Attention Architecture），2.2 “The Myth of Trainable Sparsity”（Performance Degradation；Training Efficiency Demands，含 Non-Trainable Components 与 Inefficient Back-propagation），2.3 “Native Sparsity as an Imperative”。论文没有“选谁 / 怎么读 / 怎么训”的三分法 → 课文 §1 改为“这些论点可以归到下面三处（本课的归纳）”。
3. **Exact-Top 的定义**（NSA §4.1）：“Exact-Top, which first computes full attention score and select the top-n scores keys corresponding to each query and then calculates attention on these positions.” → 课文 §4 误读框据此说明它是 token 级选择。
4. **NSA §6.1 的两条放弃路线是分开试的。** 辅助损失路线：“introduce additional queries for each token and representative keys for each block … mean-pooling attention scores within each key block, and use KL divergence to supervise block importance prediction”；启发式路线：Quest 式 min-max 打分，另试 “Full Attention is applied for the initial 1000 steps before transitioning to the heuristic blockwise selection”。1000 步稠密冷启动只和启发式路线搭配过 → 课文 §3 原写“NSA 当年试过并放弃的组合”，改为“这两样 NSA 当年都试过（分别用在两条路里）、都放弃了”；NSA 参考文献卡片 matters 同步改写。
5. **NSA 图 8 的原文描述**：“we visualize the attention map from our pretrained 27B Full Attention model … attention scores tend to exhibit blockwise clustering characteristics, with nearby keys often showing similar attention scores.” 原文没有“大片区域权重接近零”的说法 → 课文 §1 改为“成块聚集，相邻的键常有相近的分数”。另：选择块的硬件理由原文还有 “blockwise computation enables optimal utilization of Tensor Cores”（§3.3.2），支持课文 §3 的张量核一句。
6. **NSA 效率实验硬件**（§5）：“on an 8-GPU A100 system”；引言：“compare the kernel speed on A100 GPUs with optimized Triton … implementations”。NSA 全文未说明数值精度（grep bf16/fp16/precision 无相关结果），课文 §3 算例中“按 BF16 假设”的标注保留。
7. **V3.2-Exp README**（GitHub `deepseek-ai/DeepSeek-V3.2-Exp` 与 HF 模型卡 README 内容相同，raw 下载比对）：“DeepSeek Sparse Attention (DSA) achieves fine-grained sparse attention for the first time”；2025.11.17 条目 “… an implementation discrepancy in Rotary Position Embedding (RoPE) within the indexer module, potentially leading to degraded model performance”；“indexer logit kernels (including paged versions) are available in DeepGEMM (PR #200). Sparse attention kernels are released in FlashMLA (PR #98).” → 课文 §4 的 README 说法改挂 `hf2025v32expcode`（卡片 url 改为仓库根、标题补上 README）；§5 的警告框改为“FlashMLA 的稀疏核只读只算被选位置；DeepGEMM 的 indexer logits 核加速的是对全部位置的打分”（原写“真正跳过未选位置的是 FlashMLA 的稀疏核与 DeepGEMM 的 indexer 打分核”，后者并不跳过任何位置）。
8. **FlashMLA 博客原文**（`docs/20250929-hopper-fp8-sparse-deep-dive.md`，标题 “A Deep Dive Into The Flash MLA FP8 Decoding Kernel on Hopper”）：“If we assign each CTA … to process 64 query heads, it only requires $64\times(576+512)\times2/4096\approx34$ cycles”；“We launch CTAs in clusters of size 2”；“410 TFLOPS in a compute-bound configuration (batch_size=128, num_heads=128, s_q=2, topk=2048) on H800 SXM5 … over the 250 TFLOPS achieved by our previous FP8 sparse decoding kernel”；“below the 640 TFLOPS peak of our previous bfloat16 dense decoding kernel”；“comparable to that of the dense decoding kernel when the sequence length is around 3000”。→ 课文 §7 “一个线程块处理 64 个头”“两个线程块协作分摊反量化”“稠密核读 BF16”均有原文支持。附带记录（课文未用）：博客开头称 “With the release of DeepSeek-V3.2, we have doubled the context length of our models from 64K tokens to 128K tokens”，与 V3.2-Exp 报告“V3.1-Terminus 已扩到 128K”的说法口径不同。
9. **FlashMLA README**（2026-09-24 版）：2025.09.29 条目 “achieve up to 640 TFlops during prefilling and 410 TFlops during decoding”；稀疏 prefill “up to 640 TFlops in forward computation on H800 SXM5”；模式表 Dense Decoding = MQA（V3/V3.1），Sparse Decoding / Sparse Prefill = MQA（V3.2/V4/V4.1），Dense Prefill = MHA（SM100）；“the dense decoding kernel reads bf16 / fp16 KV caches”。与 C38/C39 一致。
10. **news250922**：页面标题 “DeepSeek-V3.1-Terminus”，日期 2025/09/22，首句 “DeepSeek-V3.1 → DeepSeek-V3.1-Terminus”。课文 §4 的发布日期据此新增参考文献卡片 `deepseekapi2025terminus`。
11. **V3.2 附录图 7 题注**（`arxiv.py grep 2512.02556`）：“For DeepSeek-V3.1-Terminus, the MHA mode is used for training and prefilling, while the MQA mode is used for decoding.” 说的是 V3.1-Terminus，不是 V3 论文本身 → 课文 §1 改为“V3 这一代的 MLA……（V3.2 论文附录以同一架构的 V3.1-Terminus 为例）”，V3.2 卡片 finding 同步补上 V3.1-Terminus。同次 grep 确认 ChatbotArena（2025-11-10，“closely matched”）、AA-LCR（推理模式高四分）、Fiction.liveBench（“consistently outperforms”）与 masked MHA 原句。
12. **ACL Anthology BibTeX**（`2025.acl-long.1126.bib`）：15 位作者顺序与 arXiv 一致（ACL 写作 “Wei, Yuxing”，arXiv 写作 “Y. X. Wei”），pp. 23078–23097，DOI 10.18653/v1/2025.acl-long.1126，Vienna，2025 年 7 月。
13. **V4 的 CSA 动机**（据第 18 课研究笔记 C20、C23）：V4 论文给出的动机是“上下文极长时注意力成为主要计算瓶颈”；CSA 的 indexer 对压缩后的 indexer 键打分（每 4 个 token 一个）。论文没有说 CSA 是“为了解决 indexer 的 $O(T^2)$” → 课文封面改为“压缩之后打分器要打分的条目也少了几倍，正好缓解……（把两者联系起来是本课的读法）”。
14. **indexer 的初始化方式**仍未核实（见 §8 第 9 条）→ 课文 §6 原写“刚加上去的 indexer 是随机初始化的”，改为“报告没有说明它怎样初始化；按新加模块通常随机初始化来设想”。
