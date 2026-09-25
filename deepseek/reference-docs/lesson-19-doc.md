# 第 19 课研究笔记：DeepSeek-V4.1-Flash——把 KV cache 压到极限

> 服务对象：`deepseek/src/lesson-19-v4-1-flash.md`（short：V4.1-Flash）。建立：2026-09-24。
> 主来源 arXiv 2609.19969 **v1** 已通读全文（LaTeXML HTML 全文 + arXiv PDF 51 页 + Hugging Face 仓库里的
> `DeepSeek_V41_Tech_Report.pdf` 逐词 diff）。对照来源：V4 技术报告 arXiv 2606.19348 v1（§2.3、§3.5、§4.2、§5.3 精读）、
> Hugging Face 上 V4.1-Flash / V4-Flash / V4-Pro 的 `config.json`、V4.1 与 V4-Flash 的 `inference/` 参考代码、
> V4.1 全部 48 个 safetensors 分片的文件头（按张量形状数参数）、FlashMLA README（含 2026-09-10 的 V4.1 条目）、
> DeepSeek 新闻页与 API 文档（2026-09-24 抓取）。
> 所有派生数字都在 Python 中复算，表达式列在 §4；标“笔者推算/笔者判断”的是本笔记自己的推理，不是论文的说法。
> 引文保持英文原文，只截取承载事实的最短从句。V4.1 论文按 arXiv 非独占许可发布；HF 仓库与权重为 MIT 许可。
> 批注约定（与仓库 `CLAUDE.md` 一致）：在本文件里用 `<!-- 批注(Harry): … -->` 留言，回复写 `<!-- 回复(Claude): … -->`。

---

## 0. 覆盖范围与所读版本

| 键 | 标题 | 作者 | 编号 / 版本 | 状态 | 日期 | 读法 |
|---|---|---|---|---|---|---|
| **V41** | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | DeepSeek-AI（PDF 元数据列 591 个姓名；附录 A “listed alphabetically by their first name”） | arXiv 2609.19969 **v1**（截至 2026-09-24 仅 v1） | 预印本（技术报告，未经同行评审） | 提交 2026-09-17 09:43 UTC；HF 仓库 PDF 生成于 2026-09-10 05:49 UTC | LaTeXML 全文逐节读完；PDF 用于页码与矢量图文字；HF 版 PDF 与 arXiv PDF 做词级 diff；图 1(a)(b)、图 2、图 6 的 PNG 打开看过，图 2 曲线用像素数字化 |
| **V4** | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | DeepSeek-AI | arXiv 2606.19348 **v1** | 预印本，自称 “preview version” | 提交 2026-04-26 14:49 UTC | 精读 §2.3（CSA/HCA 公式 (9)–(27)）、§2.3.4、§3.5.1–3.5.2、§4.2.1–4.2.2、表 1、表 6–7 |
| YOCO | You Only Cache Once: Decoder-Decoder Architectures for Language Models | Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma 等 9 人 | arXiv 2405.05254 **v2** | NeurIPS 2024（据 V41 参考文献条目：Advances in NeurIPS 37, pp. 7339–7361；本次未打开会议页） | 2024-05 | grep §2（式 (1)–(3)）与摘要 |
| CLA | Reducing Transformer Key-Value Cache Size with Cross-Layer Attention | William Brandon 等 5 人 | arXiv 2405.12981 v1 | NeurIPS 2024（据 V41 参考文献） | 2024-05-21 | 只核对元数据（V41 引它为“层维度复用缓存”的先例） |
| IndexCache | IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse | Yushi Bai 等 8 人 | arXiv 2603.12201 v1 | 预印本 | 2026-03-12 | 摘要 |
| YOIO | You Only Index Once: Cross-Layer Sparse Attention with Shared Routing | Yutao Sun 等 5 人 | arXiv 2606.06467 v1 | 预印本 | 2026-06-04 | 元数据 |
| HySparse | HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing | Yizhao Gao 等 14 人 | arXiv 2602.03560 v1 | 预印本 | 2026-02-03 | 元数据 |
| HISA | HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention | Yufei Xu 等 14 人 | arXiv 2603.28458 **v4** | COLM 2026（arXiv comment 自述） | 2026-03-30 | 摘要（V41 §2.3.2 引它为“先块后 token”的先例） |
| PowerAttn | PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention | Lida Chen 等 11 人 | arXiv 2503.03588 v1 | 预印本 | 2025-03-05 | grep “receptive field”“propagation”及 §4.6（V41 §2.2 用它支持“SWA 有效感受野远小于理论值”） |
| MX | Microscaling Data Formats for Deep Learning | Bita Darvish Rouhani 等 33 人 | arXiv 2310.10537 **v3** | 预印本 | 2023-10 | 表 1（MXFP4 = E2M1 元素 + E8M0 共享缩放，块大小 32） |
| DSpark | DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation | Xin Cheng 等 33 人 | arXiv 2607.05147 v1 | 预印本 | 2026-07-06 | 摘要 |
| Engram | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models | Xin Cheng, Rui Tian, Wangding Zeng 等 21 人 | arXiv 2601.07372 **v2** | V41 参考文献另列 ACL 2026 长文 pp. 4968–4990（本次未核实） | 2026-01 | 元数据（细节留给第 17 课） |
| mHC | mHC: Manifold-Constrained Hyper-Connections | Zhenda Xie 等 20 人 | arXiv 2512.24880 v2 | 预印本 | 2025-12-31 | 元数据（细节留给第 16 课） |
| 阶梯图的旁证 | GPT-3（2005.14165 **v4** 表 2.1）；Llama 3（2407.21783 **v3** 表 3）；DeepSeek-V2（2405.04434 **v5** §3.2.2 “6 bits”）；DeepSeek LLM 67B / V2 / V3 / V3.2 的 HF `config.json` | — | — | — | — | grep 表格行；config 原文下载 |
| 制品 | HF `deepseek-ai/DeepSeek-V4.1-Flash`：`config.json`、`inference/{config.json,model.py,kernel.py,image_processor.py,README.md}`、`encoding/README.md`、`README.md`（模型卡）、48 个 safetensors 分片头 | DeepSeek-AI | 提交 `dba1be0a…`，仓库创建 2026-09-10 02:17 UTC | artefact，MIT | — | 原文下载后 grep；分片头用 HTTP Range 读取 |
| 制品 | HF `DeepSeek-V4-Flash` 与 `DeepSeek-V4-Pro` 的 `config.json`，`DeepSeek-V4-Flash/inference/model.py` | DeepSeek-AI | V4-Flash 仓库 lastModified 2026-06-22 | artefact | — | 原文下载 |
| 制品 | FlashMLA README（“2026.09.10 Release of DeepSeek v4.1's Attention Kernels”） | DeepSeek-AI | GitHub `main` 分支，2026-09-24 抓取 | artefact | — | 原文下载 |
| 制品 | DeepSeek 新闻页 “Introducing DeepSeek-V4.1-Flash”；API Change Log；Models & Pricing | DeepSeek | 2026-09-24 抓取 | artefact | 2026-09-10 | WebFetch（经摘要模型转述，数字需发稿前人工复看，见 §8） |

**版本差异摘要**
- arXiv v1 PDF 与 HF 仓库 PDF（早 7 天）逐词 diff：只有两处——“YoCo” 两次改成 “YOCO”，以及删掉一个多余的 “for”。数字、公式、表格完全相同。
- 模型卡（HF `README.md`）与论文表 3 有一处不一致：**NL2Repo-Bench 的 DS-V4.1-Flash 分数，论文 65.4，模型卡 64.0**。模型卡把闭源列名写成 “Opus-5.0”，论文写 “Opus-5”。
- V41 表 1、表 3 里 V4-Flash / V4-Pro 的分数**不等于** V4 论文自己的分数（论文未说明原因）。后训练表（表 3）的差异有制品佐证：V4-Flash、V4-Pro 两列与 API Change Log 里 V4-Flash-0731 更新和 V4-Pro GA（2026-08-13）两条公告的分数逐项相同，即比较对象是更新后的检查点；基座表（表 1）的差异仍无解释。见 §7 E14。

---

## 1. 核心论断与原文引文

节号、式号、图号一律按 arXiv v1（LaTeXML 与 PDF 一致）。PDF 页码：§2.2/§2.3 在 p.9，§2.3.1 p.10，§2.3.2 p.11，
§2.4.1 p.12，§2.4.2–2.4.3 p.13，§2.4.4/§2.5 p.14，§3.2 p.18，§3.2.1 p.19，§3.2.2/§4.1 p.20，§4.2.1 p.21，
§4.2.2 p.22，表 1 p.23–24，§5.1 p.25，§5.1.4 p.29，表 2 p.30，表 3 p.33，表 4 p.34–35，§6 p.37，附录 C p.49。

### 1A. 动机与总账

**C1. 负载变成“输入重”，瓶颈从算力转向存储与搬运。**
摘要：“prefill remains computationally expensive, and large KV caches continue to strain HBM and SSD capacity and data-transfer bandwidth.”
§1：“Prior advances in sparse attention … have significantly reduced the computational cost of long-sequence processing, making persistent storage and data movement increasingly prominent bottlenecks.”

**C2. 两类 KV 的定义（本课最重要的术语区分）。** §1：
“The global branch maintains global KV, comprising main KV and indexer K, while SWA maintains local KV states.”
“For a fixed window size, SWA KV storage is bounded independently of sequence length. For sufficiently long sequences, global KV therefore dominates the runtime KV footprint, which is constrained by HBM capacity.”
→ **global KV = main KV + indexer K**，随序列长度线性增长，常驻 HBM；**SWA KV** 每层只保留最近 $n_{win}$ 个，长度无关。

**C3. 持久化 KV（persistent KV）的定义。** §1：“certain KV are persisted for prefix reuse, referred to as persistent KV caches, which are constrained by SSD and host memory capacity.”
摘要把两种口径对应到硬件：global KV “(always in HBM)”，persistent KV “(always on SSD or in host memory)”。

**C4. 规模与激活。** 摘要：“a multimodal Mixture-of-Experts (MoE) model with 552B backbone parameters and support for contexts of up to one million tokens”；“the model activates 16B parameters per token during decode but only 8B parameters during prefill”。§2.1 补充：“DeepSeek-V4.1-Flash has 552B backbone parameters and 196B Engram parameters”。
→ “552B” **不含** 196B 的 Engram 表（也不含 DSpark 与视觉编码器）；checkpoint 总逻辑参数约 763B（§4 复算）。

**C5. 890 字节/token，约为 V4-Flash 的 1/4。** 摘要：“These designs reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash.”
图 1(b) 标注（PNG 目视）：DeepSeek-V1 **389,120**（2023.11）→ DeepSeek-V3.2 **48,068**（2025.12）→ DeepSeek-V4-Flash **3,514**（2026.04）→ DeepSeek-V4.1-Flash **890**（2026.09）；箭头标注 “8.1× smaller”“13.7× smaller”“3.9× smaller”。图 1 标题：“approximately 4-fold and 437-fold reductions in per-token global KV cache size relative to DeepSeek-V4-Flash and DeepSeek-V1”。四个数全部可从配置复算，见 §4-A。

**C6. 持久化 KV 约为 V4-Flash 的 1/8，来自两个乘法因子。** §3.2.1：
“Two multiplicative factors account for this reduction: the persistent KV cache no longer stores SWA KV, which almost halves its size, and the global KV retained in it is further compressed to 1/4 of V4’s footprint”。
→ $\tfrac12\times\tfrac14=\tfrac18$。注意：“1/2”是 V4 线上负载的统计（“In the V4 deployment, SWA KV accounts for nearly half of the persistent KV cache capacity.”），不是架构常数。

**C7. 设计哲学：V4 = 以 SWA 做局部处理的主干 + 压缩的全局上下文。** §1：
“Conceptually, DeepSeek-V4 can be viewed as an SWA-based local-processing backbone augmented with compressed global context. This perspective motivates us to focus on simplifying the global branch while largely preserving the local attention design.”
→ 整篇论文的所有改动都落在“全局分支”上；局部 SWA 几乎原样保留（仍然每层都有、仍是 FP8、窗口仍是 128）。

**C8. 单 token 解码 FLOPs 几乎不随上下文增长。** §1：“extending the context length 256-fold, from 4K to 1M, increases its Decode FLOPs by only 1/4, significantly less than the growth observed for DeepSeek-V4-Flash.”
图 2 标题给出加权口径：BF16、FP8、FP4 运算分别按 1、0.5、0.25 计（LaTeXML 把它渲染成 “11, 0.50.5, 0.250.25”，是重复渲染，不是数值）。数字化读数与笔者的解析模型见 §4-F。

### 1B. 因果编码器–解码器（CED）

**C9. 来源是 YOCO。** §2.2：“we propose the Causal Encoder-Decoder (CED) architecture, inspired by YOCO (Sun et al., 2024). YOCO reduces prefill computation by allowing the upper half of the layers to directly share the KV cache generated by the lower half.”
并称 CED “introduces a series of structural improvements to enhance both the overall KV cache capacity and the computational depth of KV generation”，结论 “reduces nearly half of the prefill computation while maintaining performance comparable to the baseline”。

**C10. 解码器的全局 KV 从编码器最后一层的隐藏状态直接投影。** §2.2：“For the upper half layers (i.e., the decoder, $l>L/2$), the KV entries are not derived from their respective hidden states $H_l$. Instead, they are projected directly from the hidden state of the $(L/2)$-th layer, $H_{L/2}$, using layer-dependent projection weights”，即式 (1)。紧接着：“This design allows CED to compute only the first half of the layers during the prefill phase, acquiring the upper-layer global KV cache with minimal computational cost.”

**C11. 但 SWA 仍然逐层计算，所以需要“回放”。** §2.2：“For sliding window attention (SWA), CED maintains the conventional layer-wise computation across all layers. … However, maintaining this layer-wise computation necessitates an SWA replay process. During the prefill phase, computing the SWA KV cache for the decoder requires processing an additional $n_{win}\times L/2$ tokens”。
→ 这是 CED 与 YOCO 的关键区别：YOCO 的上半部分**没有**自注意力，只有对共享 KV 的交叉注意力（YOCO 式 (2)(3)），prefill 可以精确提前退出；CED 的解码器每层保留自己的 SWA，精确提前退出就不可能了。

**C12. 复杂度。** §2.2：“for a sequence length $N\gg n_{win}$, CED reduces the prefill complexity from $\mathcal O(NL)$ to $\mathcal O(NL/2+n_{win}\times L/2)\approx\mathcal O(NL/2)$, effectively halving the overall computation.”（这里的 $\mathcal O(NL/2+\dots)$ 已经把 Decoder SWA Bounded Replay 算进去：解码器只跑最后 $n_{win}$ 个 token。）

**C13. CED 与 CSA2 的分工。** §2.1：“CED constructs the decoder’s global key-value (KV) cache from encoder outputs, allowing most prompt tokens to bypass full decoder computation while retaining layer-local sliding-window attention. This nearly halves prefill computation”。“CSA2 shares global KV across layers to reduce cache storage and reuses sparse selections to reduce indexing work.”

**C14. 与 CSA2 组合后，式 (1) 只在一层上真正实例化。** §2.3.1：“When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the $(L/2)$-th layer, i.e. the last layer of the causal encoder. The Reindex and Reuse Modes are unchanged.” 解码器 20 层里只有第 20 层（0 起编号）是 Full Mode（`kv_source_layer_ids: [2, 8, 14, 20]`），所以整个解码器的全局 KV 只有**一份**，由 $W_{20}^{KV}$ 从 $H_{L/2}$ 投影而来。代码里第 20 层的 `Compressor`（`ratio == 1` 分支：`self.norm(self.wkv(x))`）吃的正是第 20 层的输入，即编码器输出经 mHC 输入混合与 `attn_norm` 之后的向量。

**C15. 8B / 16B 的来源（论文没有给分解）。** 论文只给结论（摘要、§2.1、§4.2.1、§6）。笔者按 safetensors 形状复算（§4-C）：解码时每 token 走 40 层 + 输出头 ≈ 16.5B（不含嵌入表 15.8B）；prefill 时每个提示 token 只走编码器 20 层 + 第 20 层的 KV 投影 ≈ 8.2B（不含嵌入表 7.6B）。与论文“8B / 16B”一致到约 5%。

### 1C. CSA2：跨层共享 KV 与索引

**C16. 三个相乘的压缩维度。** §2.3：“These costs can be reduced along three multiplicative dimensions: the entry size, where GQA … reduces the number of KV heads and MLA … shares a small latent across heads; the sequence dimension, where every $m$ tokens are compressed into one entry, like CSA and HCA in DeepSeek-V4 …; and the layer dimension, where some layers reuse the caches (Brandon et al., 2024) and selections of other layers instead of keeping their own, or are replaced altogether by more efficient layers.”
→ 这是本课的“总公式”：每 token 字节 = 每条目字节 × 每 token 条目数（序列维） × 存储份数（层维）。

**C17. 对先前层维度工作的批评。** §2.3：“index reuse alone saves no main KV storage, network-wide routing sharing limits performance, and hybrid designs still retain full attention layers; more importantly, none of these methods covers all three multiplicative dimensions.”（分别指 IndexCache、YOIO、HySparse。）

**C18. CSA2 相对 CSA 的三处简化。** §2.3：“In CSA, a compression ratio of $m$ produces each main KV entry from $2m$ original KV cache entries, with overlapping source entries for adjacent compressed entries. It also includes absolute positional embedding … CSA2 removes this overlap and absolute positional embedding. In addition, CSA2 obtains indexer K by projecting main KV entries, replacing CSA’s separate compression path from hidden states.”
代码佐证：V4-Flash 的 `Compressor` 有 `self.ape`（形状 `[ratio, coff*head_dim]`）与 `overlap = compress_ratio == 4`；indexer 自带一个 `Compressor(args, compress_ratio, self.head_dim, True)`（最后的 `True` 是 Hadamard 旋转）。V4.1 的 `Compressor` 没有 `ape`、没有 overlap；indexer 键是 `self.wk = Linear(args.head_dim, self.index_head_dim)` 作用在主 KV 的**未加 RoPE** 的潜向量上，也没有 Hadamard 旋转。

**C19. 压缩比 1 是 CSA2 的特例。** §2.3：“CSA2 also includes the uncompressed main KV setting as a special case with a compression ratio of 1.” 解码器 20 层全部是 $m=1$（§4.2.1）。

**C20. 三种模式（静态指定）。** §2.3.1：
- Full：“The layer computes its own main KV and indexer Q, projects indexer K from that main KV, and runs the indexer to produce fresh Top-K indices.”
- Reindex：“The layer reuses the most recent available main KV from a preceding layer together with its corresponding indexer K. The indexer computes its own query, rescores the reused keys, and produces fresh Top-K indices.”
- Reuse：“The layer reuses the most recent available main KV and the latest Top-K indices … It performs attention using this selection without computing indexer Q or evaluating index scores.”
小结句：“Sharing main KV and indexer K reduces cache storage, while reusing Top-K indices avoids additional indexer computation.”

**C21. 每层仍保留自己的 Q 与 SWA KV。** §1：“In all three modes, each layer retains its own global Q and SWA KV.” 图 4 标题：“All three modes compute main Q and SWA KV in the current layer.”

**C22. 不再有 HCA。** §1：“different from DeepSeek-V4 that employs the Compressed Sparse Attention (CSA)–Heavily Compressed Attention (HCA) hybrid architecture, DeepSeek-V4.1-Flash uses pure CSA2.”

**C23. 层配置（§4.2.1 原文要点）。** “We set the number of Transformer layers to 40 and the hidden dimension $d$ to 5120.”“For the first two layers, we use pure sliding window attention. The remaining 18 encoder layers use CSA2 with a compression rate of $m=2$. These layers are divided into three identically configured groups of six layers. In each group, the first layer operates in Full Mode, and the remaining five layers operate in Reuse Mode. The 20 decoder layers use CSA2 with a compression rate of $m=1$. These layers are divided into five groups of four layers. In the first group, the first layer operates in Full Mode, and the remaining three layers operate in Reuse Mode. The remaining four groups share the same configuration: the first layer operates in Reindex Mode, and the remaining three layers operate in Reuse Mode.”
config 逐字对应：`compress_ratios` 前 40 项为 `0,0` + 18×`2` + 20×`1`；`kv_source_layer_ids: [2, 8, 14, 20]`；`index_source_layer_ids: [2, 8, 14, 20, 24, 28, 32, 36]`。**全表见 §3-B。**

### 1D. 分层稀疏索引器（Hierarchical Sparse Indexer）

**C24. 只在解码器里用；把后续索引器的每查询成本从线性变常数。** §2.3.2：“We therefore introduce the Hierarchical Sparse Indexer, which is used only in the decoder of CED … For each query, the first layer assigned to Full Mode constructs a candidate pool that later re-indexing layers use as their search domain. For a fixed candidate-pool size, this changes the per-query cost of deeper indexers from linear in context length to constant.”

**C25. 候选池怎么造。** §2.3.2：“each block is assigned the maximum index score among its positions, and the blocks with the highest scores are selected. … For example, selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions.”§4.2.1：“we select a maximum of 2,048 blocks with 8 positions, yielding up to 16,384 candidate positions in total.”config：`candidate_source_layer_id: 20`、`candidate_topk_blocks: 2048`、`candidate_block_size: 8`。
代码细节（论文未写）：`select_candidate_blocks` 把“包含当前查询最新位置的那个块”强制钉入（`masked_fill(... == last, torch.inf)`），理由见代码注释：“it holds the most recent tokens but could otherwise be outscored by an older, full block”。

**C26. 训练感知，且在后训练阶段才引入；第一层仍全扫。** §2.3.2：“The mechanism is training-aware and introduced in post-training: the candidate restriction is applied identically during training and inference”；“The first Full Mode layer still scans the entire causally visible range.”
先例：“Prior work introduced indexer sparsity by scoring and pruning pooled block representations before token-level indexing (Xu et al., 2026b)”（HISA）。差别：HISA 用池化的块表示打分；V4.1 直接取第 20 层 token 级分数的块内最大值，“without adding any extra state”。

### 1E. FP4 主 KV 缓存

**C27. 两种 FP4 格式，各有理由。** §2.4.4：
- indexer：“DeepSeek-V4 already uses quantization-aware training (QAT) … for FP4 indexer queries and keys … We adopt the OCP-standard MXFP4 format … to support as many hardware platforms as possible, despite the higher accuracy of alternative formats in our experiments.”
- 主 KV：“We now extend QAT to the main KV cache, where FP4 reduces storage rather than accelerates matrix multiplication. Dequantizing cached values before attention allows us to use a more accurate format without requiring native matrix-multiplication support for that format”。
- 格式：“we select E2M1 with one E4M3 scale per 16 channels, following NVFP4 … but omitting its second-level global scale”。
代码与之一致：`fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`，注释 “Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.” E4M3 缩放取 `Cast(FP8, amax/6)`（就近舍入，不是 2 的幂），E8M0 缩放取 `fast_round_scale`（向上取到 2 的幂）；E4M3 路径把 amax 下限钳到 $6\times2^{-9}$，注释 “keep even an all-zero group's scale nonzero”。

**C28. 不要全局缩放的理由：动态范围够用。** §2.4.4：“the format supports magnitudes up to $448\times6=2688$, far above the cache’s magnitude bound. … the largest trained RMSNorm weight magnitude is approximately 1. After RMS normalization, the L2 norm of the 512-channel KV latent is at most approximately $\sqrt{512}$. RoPE preserves this norm, so the maximum absolute value across channels after rotation is also bounded by approximately $\sqrt{512}\approx22.6$. Besides, the maximum magnitude observed during training is around 10.”

**C29. 后训练阶段 QAT；RoPE 之后量化；RoPE 维也量化。** §2.4.4：“we introduce QAT during post-training. The non-RoPE and RoPE components use the same quantization format. We quantize the cache after RoPE: quantizing before RoPE yields only a marginal accuracy improvement in our experiments and would introduce additional overhead during decoding.”
对比 V4：“BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions.”（V4 §2.3.4）

**C30. SWA KV 保留 FP8。** §2.4.4：“We retain FP8 for the SWA KV cache due to its sensitivity to quantization.”

**C31. 省多少。** §2.4.4：“Compared with the FP8 main KV cache in DeepSeek-V4, this format nearly halves the storage footprint, both in HBM and when offloaded to SSD.”（复算：288/584 = 0.493。）

**C32. 反量化在哪里发生。** 论文：“Dequantizing cached values before attention”（§2.4.4）。FlashMLA README 对 FP8 缓存的说明：“It dequantizes the cache to bfloat16 and performs attention computation in bfloat16.”；V4.1 FP4 格式 “is only valid for `extra_k_cache`, and only when `k_cache` is in the V4.1 format … In pratice we expect the sliding window (SWA) kv cache to be in FP8 and the compress attention (CA) kv cache to be in FP4.”（README 原文拼写 “pratice”）。即：FP4/FP8 只是**存储**格式，注意力核在片上把它们解回 BF16 再做矩阵乘。V4.1 的稀疏解码核 “is only available on SM100”。

### 1F. 持久化缓存与 SWA Bounded Replay

**C33. V4 的线上做法。** §3.2.1：“In the V4 deployment, SWA KV accounts for nearly half of the persistent KV cache capacity. … SWA KV is cached at two specific points—the end of the prompt and the end of the output—to facilitate regeneration and multi-turn sessions … we configured a sufficiently large persistent KV cache on SSD such that, under typical workloads, both types of KV remain resident for over 72 hours.”

**C34. 为什么持久化 SWA KV 不划算。** §3.2.1：“SWA KV is reused only within a narrow, minute-scale window inside an active session and becomes dead once the session ends or the next turn begins. The V4 technical report proposed Zero SWA Caching … Exact recovery, however, requires a full forward pass over $L\times n_{win}$ tokens, whose cost proved prohibitive in production deployments.”

**C35. V4.1 的新分工。** §3.2.1 第 1 条：“SWA KV is no longer cached in the persistent KV cache and instead stored in a distributed memory pool provisioned from 10% of the host DRAM on each machine. … its short TTL (only minutes) allows expired entries to be recycled immediately … Global KV remains in the persistent KV cache with a guaranteed lifetime of at least 72 hours.”第 2 条：“it recovers the missing state by recomputing only $n_{win}$ tokens instead of a full $L\times n_{win}$-token forward pass. This bounded replay is the cornerstone of the design”。

**C36. Bounded Replay 的定义（本课要画的那张掩码）。** §3.2.2：“Since SWA dependencies accumulate across layers, exactly reconstructing the SWA KV of $L$ layers would require replaying $L\times n_{win}$ tokens. SWA Bounded Replay instead replays only the most recent $n_{win}$ tokens and truncates SWA to the replay segment, accepting approximate states: for a replay starting at position $s$, a query at position $i$ attends to SWA keys in $[\max(s,\,i-W+1),\,i]$.”（论文在这里没有定义 $W$；按上下文它就是 $n_{win}$，这是笔者判断。）

**C37. Encoder SWA Bounded Replay：近似、结果依赖命中位置。** §3.2.2：“The replayed tokens regenerate only SWA KV, reusing the cached global KV without recomputation or overwriting, while the uncached suffix generates both global KV and SWA KV. By design, the replayed prefix state is approximate, so the global KV and SWA KV computed for the uncached suffix depend on the cache-hit position and are not mathematically identical across positions.”质量：“our experimental evidence confirms that this bounded replay strategy barely compromises response quality.”（没有给数字，见 §8。）

**C38. Decoder SWA Bounded Replay：prefill 砍半的另一半功劳。** §3.2.2：“Decoder SWA Bounded Replay bounds the decoder forward pass to $n_{win}$ tokens, nearly halving total prefill computation.”“exactly reconstructing it requires running the $\frac L2$ decoder layers over the last $\frac L2\times n_{win}$ prompt tokens … at every prefill, we replay the last $n_{win}$ tokens of the prompt, feed their encoder outputs through the decoder layers under the same SWA truncation, and use the resulting decoder SWA KV only for decoding, not for prefix caching.”“For added safety, we additionally simulate the same replay during post-training for train-aware adaptation.”

### 1G. 架构精简与“高效扩展”

**C39. 论文原句与清单。** 摘要：“In addition, we streamline the DeepSeek-V4 architecture and introduce several efficient architectural extensions.”§2.4 标题即 “Efficient Architectural Extensions”，四小节：**Single-Pass mHC、Engram、DSpark、FP4 Main KV Cache**。精简（streamline）没有单独成节，散见各处；笔者汇总（每条都有出处）：
1. 纯 CSA2，去掉 HCA（§1，见 C22）。
2. 压缩器去掉重叠窗口与绝对位置嵌入（§2.3；V4 代码 `ape`/`overlap`，V4.1 代码无）。
3. indexer 键从主 KV 投影，去掉独立的压缩路径（§2.3）；代码里同时去掉了 indexer 的 Hadamard 旋转（V4 `rotate_activation`，V4.1 无；论文未提）。
4. 所有 FFN 都是标准 DeepSeekMoE（图 3 标题 “All feed-forward layers use standard DeepSeekMoE.”）；V4 前 3 个 MoE 层用 Hash 路由（V4 §4.2.1 “use the Hash routing strategy for the first 3 MoE layers”），V4.1 config 没有 `num_hash_layers`。
5. 预训练去掉 MTP（§2.1 “We omit the MTP module during backbone pre-training and use DSpark … for speculative decoding.”）。
6. 查询的逐头 RMSNorm 去掉：FlashMLA README “Q-norm (only used in V4, not V4.1)”；V4 代码 `q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)`，V4.1 代码无（仍保留 `q_norm` 作用在 1280 维查询潜向量上，以及 `kv_norm`）。论文未提。
7. 稀疏注意力从头训练、无稠密预热（§1 “Sparse attention is trained from scratch at a sequence length of 64K, without any dense attention warmup stages.”；V4 是 “first warmup the model with dense attention for the first 1T tokens”）。
8. Engram 去掉短因果卷积（§2.4.2 “we omit the short causal convolution because its performance gains do not justify the added complexity in our inference stack”）。
9. SWA 与主 KV 的 RoPE 维不再单独用 BF16（§2.4.4 C29；FlashMLA 的 V4.1 FP8 格式 “the 64 RoPE dimensions are quantized as well”）。
新增的还有：CED、分层稀疏索引器、多模态（DeepSeek-ViT）、按模态分开的无辅助损失均衡偏置、head-wise Muon、Sinkhorn 平衡更新。

**C40. Single-Pass mHC。** §2.4.1：理想映射 “requires $(n+1)d$ reads and $(n+1)d$ writes, giving a lower bound of $(2n+2)d$ on activation memory traffic”；V4 的三核实现 “Including the pre-norm in $\mathcal F_l$, the total activation memory traffic is $(4n+4)d$, twice the lower bound.”；两遍实现 “$(3n+2)d$”；Single-Pass 把输入混合系数错后一层（式 (6)），“Empirically, this shift incurs negligible performance degradation.”；部署核 Mega-mHC “The residual is thereby read once and written once … halving the activation memory traffic of our original implementation.”预训练时 “we keep the existing multi-kernel implementation”。代码：`Block.forward` 注释 “attention uses what the previous layer's FFN produced and the FFN uses what this attention produced”。

**C41. Engram 配置。** §2.4.2：“We allocate 196B Engram parameters evenly across two modules. Each module uses $N$-gram orders $\{2,3,4\}$, with 8 hash heads and a total embedding dimension of 2048 per order. Each head indexes a table of approximately 16M entries, with table sizes chosen to be distinct primes. Both the embedding tables and the key/value projections use FP8 precision. The modules are placed at layers 1 and 14 (zero-indexed)”；推理时 “deterministic addressing enables embeddings to be prefetched from host memory via background RDMA transfers”。config：`engram_layer_ids: [1, 14]`、`engram_num_embeddings: [384006168, 384016682]`、`engram_n_heads: 8`、`engram_head_dim: 256`、`engram_vocab_size: 16000000`。

**C42. DSpark。** §2.4.3：“The drafter comprises three Transformer blocks with a sliding attention window of 128 tokens. A single forward pass through these blocks computes base logits for five draft positions in parallel, while a lightweight Markov head models dependencies among the draft tokens.”训练：“DSpark is introduced in a dedicated stage after pre-training. In this stage, we train only DSpark while keeping the backbone frozen. During post-training, we continue to train DSpark alongside the backbone, without propagating gradients from the DSpark objective into the backbone.”config：`num_nextn_predict_layers: 3`、`dspark_block_size: 5`、`dspark_target_layer_ids: [37, 38, 39]`、`dspark_markov_rank: 256`、`dspark_n_routed_experts: 128`、`dspark_num_experts_per_tok: 3`；权重在 checkpoint 的 `mtp.*` 命名空间下（代码注释 “DSpark stage stored under the mtp.* checkpoint namespace.”）。DSpark 论文摘要：线上相对 MTP-1 “accelerates per-user generation speeds by 60 to 85 percent at matched throughput levels”（部署于 V4 服务系统）。

### 1H. 多模态

**C43. DeepSeek-ViT。** §2.1.1：“we replace standard absolute positional embeddings with 2D-RoPE … we replace the patch embedding layer’s convolution with a linear projection to ensure compatibility with the Muon optimizer. We also adopt RMSNorm … and SwiGLU”；“we apply a pixel-unshuffle operation with $3\times3$ downsampling to reduce the visual token count by a factor of nine, effectively supporting input resolutions up to approximately $1344\times1344$ pixels.”§4.2.1：“we set its number of layers to 32, the hidden dimension to 1024, the number of attention heads to 16, and the image patch size to 14. The vision MLP projector has 2 layers with a hidden dimension of 5120.”

**C44. 按模态分开的无辅助损失均衡。** §2.1.1：“we extend auxiliary-loss-free load balancing … by maintaining separate expert-wise correction biases for text and image tokens.”代码：`gate.bias` 与 `gate.bias_vl` 两套偏置；§4.2.2：“we set the bias update speed to 0.001 for both image and text tokens, while retaining a small sequence-level balance loss with a loss weight of 0.0001”。

**C45. ViT 两阶段训练。** §4.2.2：对比学习 “using the sigmoid contrastive loss introduced by SigLIP … on approximately 47B image-text pairs”，分辨率上限 $224\times224$；自回归阶段 “we connect the vision encoder to a 4B MoE LLM and train on 236B tokens”，分辨率限制在 $544\times544$ 到 $1344\times1344$。并入主干预训练后：“we keep the vision encoder frozen until the learning-rate decay stage”（§2.5）。

**C46. 数据配比。** §4.1：“After this substitution, the resulting corpus uses a $7{:}1$ token ratio of text-only to multimodal data.”；打包 “achieving a padding rate of at most $10^{-4}$”。

### 1I. 预训练与优化器

**C47. 45T、固定 batch、学习率。** §4.2.2：“We train DeepSeek-V4.1-Flash on 45T tokens of multimodal data with no instability. We keep the batch size fixed at 100.6 million tokens throughout training. The learning rate is linearly warmed up over the first 2000 steps and then maintained at $2.6\times10^{-4}$ until 28T tokens. Between 28T and 40T tokens, we decay the learning rate to $2.6\times10^{-5}$ following a cosine schedule. … We train the model from scratch with sparse attention at a sequence length of 64K and extend the sequence length to 1M at 34T tokens.”

**C48. 优化器。** §2.5：“we use head-wise Muon, where Query weights are split by head before applying the Muon update”（Basic Configurations 里写 “We use head-wise Muon for Query and Key weights.”——前后范围不同，见 §7）；Engram 表、词嵌入、预测头用 “a momentum-based update followed by Sinkhorn balancing”，“Like Muon, this approach requires only a momentum buffer while empirically outperforming Adam.”；“We set $\gamma=0.18$, which is close to the factor 0.2 used in Moonlight”。§4.2.2：AdamW $\beta_1=0.9,\beta_2=0.95,\varepsilon=10^{-20}$，weight decay 0.1；Muon 动量 0.95、weight decay 0.1、更新 RMS 缩到 0.18；Sinkhorn 更新 $K=11,\tau=10^{-3},\varepsilon=10^{-20}$；“the learning rate of Engram is scaled by $5\times$”。

### 1J. 后训练

**C49. 算法零创新，全靠数据与环境。** §1：“our post-training introduces no algorithmic innovation: the recipe follows the standard paradigm of supervised fine-tuning (SFT) followed by reinforcement learning (RL) and on-policy distillation (OPD)”。§5.1：“systematic improvements in the scale, diversity, and verifiability of synthesized data and environments account for essentially all of the observed gains.”§5.2.4：最后的全词表 OPD “using over 40 teacher models”。

**C50. 推理强度是 1–100 的标量。** §5.1.4：系统提示前缀 “Reasoning Effort: {effort} (range 1--100; higher values request more thorough reasoning)”，“responses from different effort levels are not directly compared”（按 $(x,b)$ 分子组做组内均值中心化）。表 2：API 档位 max/high/low 对应 $b=100/75/50$。`encoding/README.md` 的实际渲染文字略有不同（见 §7 E19），默认 `"high"`（75）。

**C51. 异步 RL 的三个工程点。** §5.2.1：最终用 “sample-level dispatch”；“we use concatenated routing-replay: for samples that span multiple checkpoints, we concatenate the expert routing produced at each rollout segment”；§5.2.2：限制最大 off-policy 比例 + 对过旧 token 做损失掩码；§5.2.3：“rollout states such as KV cache and expert routing are persisted at token granularity”。

**C52. DSec 沙箱规模。** §5.1.3：“V4.1 training further increased demand to millions of concurrent sandbox instances”；“it increases the supported density from roughly 1,000 to more than 2,500 concurrent live containers per physical node”。

### 1K. 评测

**C53. 基座（表 1）。** 表 1 标题：“Scores with a gap not exceeding 0.3 are considered to be at the same level.” §1 的概括：“DeepSeek-V4.1-Flash-Base achieves world knowledge, reasoning and coding abilities comparable to DeepSeek-V4-Pro-Base, and delivers 5%–10% improvements on held-out evaluations, using only 1/3 total parameters and 1/4 activated parameters.”（“5%–10%”与图 6 三个集合的**平均**降幅吻合：相对 V4-Pro-Base 5.3%，相对 V4-Flash-Base 9.6%，但单个集合的值不全在区间内；“1/4 activated”与 8B、16B 都对不上。见 §7 E15–E16。）图 6（BPB，越低越好）：Internal Docs 0.617 / 0.59 / **0.564**；Internal Code Repos 0.1562 / 0.1494 / **0.1443**；Academic Materials 0.4929 / 0.4677 / **0.4305**（V4-Flash-Base / V4-Pro-Base / V4.1-Flash-Base）。

**C54. 后训练模型（表 3，全部 Max 档）。** 头条：Codeforces 3471（V4-Pro 3348，V4-Flash 3289）；MathArena Apex 65.6（与 Kimi-K3 并列）；GPQA Diamond 90.9；Terminal-Bench 2.1 90.6；DeepSWE v1.1 74.2（Opus-5 74.0）；AutomationBench 54.8；Agents’ Last Exam 31.8；CyberGym 88.1。短板：Terminal-Bench 4.0 31.2（Opus-5 51.8），HLE 36.8（文本子集 39.1），ProgramBench 20.3（Opus-5 37.0）。§1 原话：“a gap with giant models remains on science-oriented agentic tasks, such as Terminal-Bench 4.0 … that require expert-level domain knowledge.”

**C55. 推理强度曲线。** §5.3.2：“Raising the effort from 25 to 100 improves the average Pass@1 on eight reasoning-intensive benchmarks from 67.1% to 76.3%, on DeepSWE v1.1 from 66.0% to 74.2%, and on Terminal-Bench 2.1 from 82.4% to 90.6%, at the cost of roughly $2.5\times$ more output tokens.”附录 B.3：“from 4.6k to 11.4k tokens per response on AIME 2026 and from 29.1k to 86.1k on MathArena Apex 2025”，“MathArena Apex 2025 gains $+40.3$ points (25.3%→65.6%)”。

**C56. 跨脚手架（表 4，Max 档）。** DeepSWE v1.1：Claude Code 69.8、Codex 65.6、OpenCode 65.5、Pi 66.2、mini-SWE **74.2**、DSH Minimal 72.6 / Standard 70.5 / PTC 67.6；Terminal-Bench 2.1：88.0 / 84.1 / 85.0 / 86.1 / 90.3 / **90.6** / 85.8 / 85.8。

**C57. 多智能体（初步）。** §5.3.5：ProgramBench（172 个 golden 任务）Almost@1 多智能体从 1 小时 13.59% 到 8 小时峰值 30.04%，单智能体 12.79% 与 20.39%；FrontierSWE v2 Mean@5 多智能体 13.50%→32.90%（20 小时），单智能体 10.50%→28.20%。

### 1L. 部署与基础设施

**C58. 内核数与分离部署。** §3.2：“the vast majority of Transformer layers—those whose CSA2 operates in Reuse Mode—execute with only 15 kernels during prefill and 11 during decode”（Reuse 层 30/40，§3-B）；“At the deployment level, we adopt Encoder–Prefill–Decode (EPD) disaggregation, enabling vision encoding, prefill, and decoding to scale independently”。§1：“We also separate long-lived global KV storage from short-lived encoder SWA KV in host memory”。

**C59. FlashMLA 的三种 V4/V4.1 字节格式（制品，逐字）。** “**V4**: 584 Bytes per token. The data row is 448 Bytes of quantized NoPE (`float8_e4m3`) followed by 128 Bytes, i.e. the 64 `bfloat16` RoPE values (not quantized). The scale row is 8 Bytes, of which the first 7 are `float8_e8m0` scales and the 8th byte is padding; each scale covers 64 consecutive `float8_e4m3` values”；“**V4.1**: 528 Bytes per token. The data row is 512 Bytes of `float8_e4m3` … The scale row is 16 Bytes of `float8_e8m0`, each scale covering 32 consecutive `float8_e4m3` values.”；“**V4.1 fp4**: 288 Bytes per token. The data row is 256 Bytes containing 512 `e2m1` values, 2 values per byte (the even-indexed one in the low nibble). The scale row is 32 Bytes of `float8_e4m3`, each scale covering 16 consecutive `e2m1` values.”融合核 “achieves up to 1430 TFlops during prefill and 670 TFlops during decoding on B200”。

**C60. API（制品）。** 新闻页（`api-docs.deepseek.com/news/news260910`）：“552B-parameter MoE”，“just 8B active parameters for input, 16B for output”，KV “1/4 the HBM”“1/8 the SSD storage”，模型名 `deepseek-flash`，“Off-peak rates are 50% of peak rates.”。价格页（每百万 token，off-peak / peak）：缓存命中 $0.003 / $0.006，未命中 $0.15 / $0.3，输出 $0.6 / $1.2；`deepseek-v4-pro`（模型版本 DeepSeek-V4-Pro-0813）为 $0.022 / $0.044、$0.66 / $1.32、$1.98 / $3.96；上下文 1M，“MAXIMUM: 384K”；价格页自己的措辞是 “Off-peak rates are half of the peak rates.”；旧名 “deepseek-v4-flash and deepseek-v4-flash-vision-exp are still accepted, but the corresponding models have been retired”。（2026-09-24 用原始 HTML 抓取核对，不经摘要模型。）关于 V4-Pro，新闻页与 Change Log 说法相反（§7 E20）。

### 1M. 局限（论文自述）

**C61.** §6：“Potential selection errors in CSA2 and approximate state reconstruction in SWA Bounded Replay may still cause capability degradation in untested boundary cases. Going forward, we will continue to expand our stress-testing and evaluation stack, with particular attention to sparse retrieval over long contexts and SWA state reconstruction at cache-resumption boundaries.”

### 1N. 对照：V4 论文（2606.19348 v1）里被本课引用的事实

**C62. V4 的 KV 存储格式。** V4 §2.3.4：“BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions. This hybrid representation reduces the KV cache size by nearly half compared with pure BF16 storage. Second, attention computation within the lightning indexer is performed in FP4 precision”。

**C63. V4 的 SWA 落盘三策略。** V4 §3.5.2：“For the SWA KV entries, since they are not compressed and exist in every layer, their volume is approximately 8 times larger than the compressed CSA and HCA KV entries.”三策略 Full SWA Caching / Periodic Checkpointing / Zero SWA Caching；后者 “recomputing the last $n_{win}\cdot L$ tokens is enough to restore the last $n_{win}$ SWA KV entries for an $L$-layer model.”（“约 8 倍”的复算：43 / 5.406 = 7.95，§4-A。）

**C64. V4 的规格。** V4 §4.2.1（Flash）：“We set the number of Transformer layers to 43 and the hidden dimension $d$ to 4096. For the first two layers, we use pure sliding window attention. For the subsequent layers, CSA and HCA are used in an interleaved manner.”CSA $m=4$、indexer 64 头 × 128 维、top-k 512；HCA $m'=128$；64 个查询头、$c=512$、$d_c=1024$；$g=8$、$d_g=1024$；$n_{win}=128$；1 共享 + 256 路由专家、中间维 2048、激活 6；MTP 深度 1；284B / 13B。Pro：61 层、$d=7168$、前两层 HCA、top-k 1024、128 头、$d_c=1536$、$g=16$、384 路由专家、中间维 3072；1.6T / 49B。

**C65. V4 的效率口径。** V4 摘要：“In the one-million-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2.”§1：V4-Flash “achieves only 10% of the single-token FLOPs and 7% of the KV cache size compared with DeepSeek-V3.2.”（复算：5031.4/48068 = 10.5%，3514.25/48068 = 7.3%，§4-A。）

### 1O. 对照：YOCO（2405.05254 v2）

**C66.** YOCO §2：“Self-decoder (Section 2.1) uses efficient self-attention (e.g., sliding-window attention). In comparison, cross-decoder (Section 2.2) uses global cross-attention to attend to the shared KV caches produced by the output of the self-decoder.”§2.2：“The KV caches $\hat K,\hat V$ are reused by all the $\frac L2$ cross-decoder modules”。摘要：“the computation flow enables prefilling to early exit without changing the final output”。
（注意 YOCO §1 有一句 “enables prefilling to early exit before entering the self-decoder”——按上下文应为 cross-decoder，疑似笔误，引用时用摘要那句。）

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 V4.1 论文中带编号的公式（按原文 LaTeX 转录）

**式 (1)，CED（§2.2）**
$$C_{l}=H_{L/2}W_{l}^{KV},\quad Z_{l}=H_{L/2}W_{l}^{Z},\quad l>\frac{L}{2},\tag{1}$$
“where $C$ and $Z$ represent the KV entries and their corresponding compression weights, respectively.”
（论文无编号的复杂度式：$\mathcal O(NL)\to\mathcal O(NL/2+n_{\mathrm{win}}\times L/2)\approx\mathcal O(NL/2)$。）

**式 (2)，mHC（§2.4.1，沿用 V4 式 (1)）**
$$X_{l+1}=B_{l}X_{l}+C_{l}\,\mathcal{F}_{l}(A_{l}X_{l}),\qquad(A_{l},B_{l},C_{l})=\mathcal{H}(X_{l}),\tag{2}$$
$A_l\in\R^{1\times n}$，$C_l\in\R^{n\times1}$，$B_l\in\R^{n\times n}$；“token-wise coefficients predicted from $X_l$”。

**式 (3)–(5)，V4 的三核实现**
$$X_{l}=B_{l-1}X_{l-1}+C_{l-1}Y_{l-1}\quad\text{(Residual update, contraction over }n)\tag{3}$$
$$(A_{l},B_{l},C_{l})=\mathcal{H}(X_{l})\quad\text{(Coefficients, contraction over }nd)\tag{4}$$
$$\hat{X}_{l}=A_{l}X_{l}\quad\text{(Input mixing, contraction over }n)\tag{5}$$
访存量（正文无编号）：下界 $(2n+2)d$；三核 + pre-norm $(4n+4)d$；两遍 $(3n+2)d$。

**式 (6)，Single-Pass mHC**
$$X_{l+1}=B_{l}X_{l}+C_{l}\,\mathcal{F}_{l}(A_{l-1}X_{l}),\qquad(A_{l},B_{l},C_{l})=\mathcal{H}(X_{l}).\tag{6}$$

**式 (7)，Sinkhorn 平衡更新（§2.5，算法 1 之后）**
$$\Delta_{t}=\sqrt{n}\,U^{(K)}=\sqrt{n}\,D_{r}\widehat{G}_{t}D_{c},\qquad\frac{1}{n}\sum_{j=1}^{n}(\Delta_{t})_{ij}^{2}\approx 1,\qquad\frac{1}{m}\sum_{i=1}^{m}(\Delta_{t})_{ij}^{2}\approx 1,\tag{7}$$
算法 1 主体（逐行转录）：$M_t\leftarrow\beta M_{t-1}+(1-\beta)G_t$；$\widehat G_t\leftarrow\beta M_t+(1-\beta)G_t$（Nesterov）；$\rho_i\leftarrow\lVert\widehat G_{t,i,:}\rVert_2$，$\bar\rho\leftarrow\frac1m\sum_i\rho_i$；若 $\rho_i\le\tau\bar\rho$ 则该行置 0；$k=1..K$：奇数步行归一化 $U^{(k)}_{i,:}\leftarrow U^{(k-1)}_{i,:}/(\lVert U^{(k-1)}_{i,:}\rVert_2+\varepsilon)$，偶数步列归一化；$\Delta_t\leftarrow\sqrt n\,U^{(K)}$；$\tilde\eta_t\leftarrow\gamma\eta_t$；$W_{t+1}\leftarrow W_t-\tilde\eta_t\Delta_t$。$K$ 取奇数（=11）保证最后一步是行归一化。

**式 (8)–(10)，推理强度（§5.1.4）**
$$z_{b,j}\sim\pi_{\theta}(\cdot\mid x,b),\qquad b\in\mathcal{B},\quad j=1,\ldots,M_{b}.\tag{8}$$
$$r_{b,j}^{\mathrm{len}}=-\min\left\{C_{\max},\;k(b)\frac{\ell_{b,j}}{L_{\mathrm{norm}}}\right\},\tag{9}$$
$$k(b)=k_{0}\exp\left(-\frac{b-b_{\min}}{\tau}\right),\qquad\tau=\lambda\overline{\Delta b},\tag{10}$$
原文性质：“Increasing $b$ by $\tau$ multiplies the penalty coefficient by $e^{-1}$.”

**式 (11)–(17)，附录 C 的边际效用推导**
$$r^{\mathrm{len}}(\ell,b)=-\min\left\{C_{\max},k(b)\frac{\ell}{L_{\mathrm{norm}}}\right\},\tag{11}$$
$$k(b)=k_{0}\exp\left(-\frac{b-b_{\min}}{\tau}\right),\tag{12}$$
$$\ell_{x}^{*}(b)\in\arg\max_{\ell\geqslant 0}\left[p_{x}(\ell)-k(b)\frac{\ell}{L_{\mathrm{norm}}}\right].\tag{13}$$
$$p_{x}^{\prime}\!\left(\ell_{x}^{*}(b)\right)=\frac{k(b)}{L_{\mathrm{norm}}},\tag{14}$$
$$p_{x}^{\prime}(\ell)\approx a_{x}\exp\left(-\frac{\ell}{s_{x}}\right),\tag{15}$$
$$\ell_{x}^{*}(b)\approx C_{x}-s_{x}\log k_{0}+\frac{s_{x}}{\tau}(b-b_{\min}),\qquad C_x=s_x\log(a_xL_{\mathrm{norm}}),\tag{16}$$
$$\ell_{x}^{*}(b_{2})-\ell_{x}^{*}(b_{1})\approx\frac{s_{x}}{\tau}(b_{2}-b_{1}).\tag{17}$$
笔者逐行核对 (16)：由 (14)(15) 得 $a_xe^{-\ell^*/s_x}=k(b)/L_{norm}$ ⇒ $\ell^*=s_x\log(a_xL_{norm})-s_x\log k(b)$，代入 (12) 的 $\log k(b)=\log k_0-(b-b_{\min})/\tau$ 即得 (16)。✔。论文自己的限定：“This derivation is a local reward-level approximation, not a claim that measured average lengths must be linear or pointwise monotonic.”

### 2.2 V4.1 论文中无编号但要用的式子

- **SWA Bounded Replay 的可见区间**（§3.2.2）：从位置 $s$ 开始回放时，位置 $i$ 的查询看到 SWA 键 $[\max(s,\,i-W+1),\,i]$。
- **精确重建的回放长度**：编码器侧 $L\times n_{win}$；解码器侧 $\frac L2\times n_{win}$；Bounded Replay：$n_{win}$。
- **FP4 动态范围**（§2.4.4）：$448\times6=2688$；$\lVert\mathbf c\rVert_2\lesssim\sqrt{512}\approx22.6$。
- **图像分片的“IO 藏在计算后”条件**（§3.1.1）：
$$\frac{N\times\rho}{B_{\mathrm{IO}}}<\frac{N\times C}{B_{\mathrm{GPU}}}\quad\Longleftrightarrow\quad\rho<\frac{B_{\mathrm{IO}}}{B_{\mathrm{GPU}}}\,C$$
（$N$ token 数，$\rho$ 每 token 原始字节，$C$ 每 token 计算量；这里的 $N$、$C$ 与课程记号冲突，见 2.6。）
- **对比学习的通信–计算重叠调度**（§3.1.1）：$\text{Forward}(V)\rightarrow(\text{Forward}(T)\,\|\,\text{AllGather}(V))\rightarrow\nabla_{\text{Text}}\rightarrow(\text{Backward}(T)\,\|\,\text{AllGather}(T))\rightarrow\nabla_{\text{Vision}}\rightarrow\text{Backward}(V)$。

### 2.3 V4 论文中被 CSA2 修改的公式（2606.19348 v1，逐字转录，用于对比）

CSA 压缩（V4 式 (9)–(12)）：
$$C^{a}=H\cdot W^{aKV},\quad C^{b}=H\cdot W^{bKV},\tag{V4-9}$$
$$Z^{a}=H\cdot W^{aZ},\quad Z^{b}=H\cdot W^{bZ},\tag{V4-10}$$
$$[S^{a}_{mi:m(i+1)-1};S^{b}_{m(i-1):mi-1}]=\operatorname{Softmax}_{\text{row}}([Z^{a}_{mi:m(i+1)-1}+B^{a};Z^{b}_{m(i-1):mi-1}+B^{b}]),\tag{V4-11}$$
$$C^{\text{Comp}}_{i}=\sum_{j=mi}^{m(i+1)-1}S^{a}_{j}\odot C^{a}_{j}+\sum_{j=m(i-1)}^{mi-1}S^{b}_{j}\odot C^{b}_{j},\tag{V4-12}$$
indexer（V4 式 (13)–(17)）：$\mathbf c_t^Q=\mathbf h_t\cdot W^{DQ}$；$\mathbf q_t^I=\mathbf c_t^Q\cdot W^{IUQ}$；$\mathbf w_t^I=\mathbf h_t\cdot W^w$；
$$I_{t,s}=\sum_{h=1}^{n_h^I}w_{t,h}^I\cdot\text{ReLU}\left(\mathbf q^I_{t,h}\cdot K^{\text{IComp}}_s\right),\tag{V4-16}$$
$$\mathcal C^{\text{SprsComp}}_t=\left\{C^{\text{Comp}}_s\ \Big|\ I_{t,s}\in\operatorname{Top-k}(I_{t,:})\right\}.\tag{V4-17}$$
HCA 压缩（V4 式 (22)(23)）：$S_{m'i:m'(i+1)-1}=\operatorname{Softmax}_{\text{row}}(Z_{m'i:m'(i+1)-1}+B)$，$C^{\text{Comp}}_i=\sum_{j=m'i}^{m'(i+1)-1}S_j\odot C_j$。
注意力汇（V4 式 (27)，V4.1 代码保留 `attn_sink`）：$s_{h,i,j}=\dfrac{\operatorname{Exp}(z_{h,i,j})}{\sum_k\operatorname{Exp}(z_{h,i,k})+\operatorname{Exp}(z'_h)}$。

### 2.4 CSA2 的公式——论文没写，由 V4.1 参考代码还原（标注“代码还原”，写进课文时必须这样说明）

令 $\mathbf h_t\in\R^{5120}$ 为第 $l$ 层注意力输入（已 `attn_norm`），$m\in\{1,2\}$。
- 压缩（`Compressor.forward`）：$m=1$ 时 $\mathbf c_t=\RMSNorm(W^{KV}\mathbf h_t)\in\R^{512}$；$m=2$ 时
$$\mathbf c^{\text{Comp}}_i=\RMSNorm\Big(\sum_{j=mi}^{m(i+1)-1}\softmax_j\big(\mathbf z_{mi},\dots,\mathbf z_{m(i+1)-1}\big)\odot\mathbf c_j\Big),\quad \mathbf c_j=W^{KV}\mathbf h_j,\ \mathbf z_j=W^{Z}\mathbf h_j,$$
softmax 对组内 $m$ 个 token **逐通道**做；与 V4-11/12 相比：没有 $B^a,B^b$（位置偏置），没有 $b$ 支路（重叠）。
- RoPE：对 $\mathbf c^{\text{Comp}}_i$ 的后 64 维按位置 $mi$ 旋转（代码注释 “a latent stands for the first token of its group, so group j takes position j * ratio”），然后 FP4（E2M1 + 每 16 通道一个 E4M3）量化写入缓存。
- indexer 键（只有 KV 源层有）：$K^I_i=\mathrm{RoPE}\big(\RMSNorm(W^{IK}\mathbf c^{\text{Comp}}_i)\big)\in\R^{128}$，用的是**加 RoPE 之前**的 $\mathbf c^{\text{Comp}}_i$，MXFP4 量化。$W^{IK}\in\R^{128\times512}$（safetensors `indexer.wk.weight` 形状 `[128, 512]`）。
- indexer 分数（代码多了两个缩放）：
$$I_{t,s}=\sum_{h=1}^{32}\frac{w_{t,h}}{\sqrt{128}\sqrt{32}}\,\mathrm{ReLU}\big(\mathbf q^I_{t,h}\cdot K^I_s\big),\qquad \mathbf q^I_t=W^{IUQ}\,\RMSNorm(W^{DQ}\mathbf h_t),\ \mathbf w_t=W^w\mathbf h_t.$$
- 分层索引（第 20 层）：块分数 $\beta_{t,b}=\max_{s\in\text{block }b}I^{(20)}_{t,s}$（块长 8）；取 top-2048 块（外加钉入最新块）形成布尔掩码；第 24/28/32/36 层把掩码外的 $I_{t,s}$ 置 $-\infty$ 再取 top-512。
- 核心注意力：每层查询 $\mathbf q_{t,i}=W^{UQ}_i\RMSNorm(W^{DQ}\mathbf h_t)$（64 头 × 512 维，V4.1 **没有**逐头 Q-norm），键值集合 = 本层 SWA 窗口 128 个（FP8）∪ 所选 512 个压缩条目（FP4 解回），带注意力汇；输出对后 64 维做 $-i$ 位置的逆旋转，再走 8 组分组输出投影（每组 4096→1024，再 8192→5120）。

### 2.5 YOCO 的对照公式（2405.05254 v2 §2.2）

$$\hat{K}=\operatorname{LN}(X^{L/2})W_{K},\quad\hat{V}=\operatorname{LN}(X^{L/2})W_{V}\tag{YOCO-2}$$
$$\hat{Q}^{l}=\operatorname{LN}(X^{l})W_{Q}^{l},\qquad Y^{l}=\operatorname{Attention}(\hat{Q}^{l},\hat{K},\hat{V})+X^{l},\qquad X^{l+1}=\operatorname{SwiGLU}(\operatorname{LN}(Y^{l}))+Y^{l}\tag{YOCO-3}$$
（YOCO 原文：$W_K,W_V,W_Q^l\in\R^{d\times d}$；“Causal masking is also used for cross-attention.”）
对照 V4.1 式 (1)：YOCO 所有上层共用**一对** $W_K,W_V$；V4.1 写成“layer-dependent” $W_l^{KV}$，但在 CSA2 配置下实际只有第 20 层有这组权重，其余 19 个解码层复用它——**效果上又回到了 YOCO 的“一份全局 KV”**，只是这份 KV 是 $m=1$ 的 FP4 主 KV + indexer 键，且解码器每层另有自己的 SWA。

### 2.6 记号对照（论文 → 课程 style-guide §2）

| 论文记号 | 含义 | 课程记号 / 处理 | 冲突提醒 |
|---|---|---|---|
| $L$，$l$ | 层数，层号 | 同（style-guide：KV 公式沿用 $l$） | 附录 C 的 $\ell$ 是推理 token 数，不是层号 |
| $N$（§2.2、§3.1.1） | 序列长度 / token 数 | 写 $T$ 或 $L_{ctx}$ | **课程 $N$ 是参数量** |
| $H_l$ | 第 $l$ 层隐藏状态（行 = token） | 单 token 写 $\mathbf h_t^{(l)}\in\R^{d}$，整段写 $H^{(l)}\in\R^{T\times d}$ 并说明按行堆叠 | — |
| $C_l$（式 1） | 第 $l$ 层的 KV 条目（既当 K 又当 V） | 建议 $\mathbf c_s^{KV}\in\R^{512}$（$s$ 为压缩块号）；说明 V4 起“KV 条目维 = 头维 $c=d_h=512$”，与 MLA 的 $d_c=512$ 数值相同但角色不同（这里没有上投影） | **与式 (2) 的 mHC 输出映射 $C_l$ 同名** |
| $Z_l$ | 压缩权重 logits | $\mathbf z_j$ | — |
| $W_l^{KV}$ | KV 投影 | 类比 MLA 的 $W^{DKV}$（只有“下投影”，没有 $W^{UK},W^{UV}$；吸收在查询/输出侧） | — |
| $m$ | 压缩比 | $m$ | 算法 1 的 $m$ 是矩阵行数 |
| $c$ | 头维 / 条目维 | $d_h$（=512） | 课程 $d_c$ 专指 MLA 潜在维 |
| $n_{win}$，$W$（§3.2.2） | SWA 窗口 | $n_{win}$ | $W$ 与权重矩阵同字母 |
| $n$（mHC） | 残差流条数 | $n_{hc}$（V4 记号） | 与 $n_h$ 易混 |
| $A_l,B_l,C_l$ | mHC 输入混合 / 残差映射 / 输出映射 | 第 16 课的 $H^{pre}_l,H^{res}_l,H^{post}_l$ | $C_l$ 与 KV 条目同名 |
| $b$ | 推理强度 1–100 | 写 $e$ 或 $b^{\text{eff}}$ | 课程 $b_i$ 是 MoE 专家偏置 |
| $\tau$ | 惩罚衰减尺度 | $\tau_{\text{eff}}$ | 算法 1 的 $\tau$ 是掩行阈值（=$10^{-3}$） |
| $M_b$ | 每个强度档采样数 | 对应 GRPO 的组大小 $G$（分子组） | — |
| $k(b)$ | 长度惩罚系数 | $k(e)$ | 与 top-$k$ 同字母 |
| $\gamma$ | 学习率校正（0.18） | $\gamma$ | V3 无辅助损失均衡的偏置步长也叫 $\gamma$ |

---

## 3. 规格与超参数表

### 3-A. V4.1-Flash 与 V4-Flash / V4-Pro 对照（每格注来源；P = 该模型论文正文，C = HF `config.json`，K = 参考代码，F = FlashMLA README）

| 项 | V4.1-Flash | V4-Flash | V4-Pro |
|---|---|---|---|
| 层数 | 40（P §4.2.1；C `num_hidden_layers`） | 43（P；C） | 61（P；C） |
| 隐藏维 $d$ | 5120（P；C） | 4096 | 7168 |
| 注意力布局 | 2 SWA + 18 CSA2($m$=2) + 20 CSA2($m$=1)（P；C `compress_ratios`） | 2 SWA + 21 CSA($m$=4) + 20 HCA($m'$=128)（C；P 只说“interleaved”） | 2 HCA + 30 CSA + 29 HCA（C；P “first two layers, we use HCA”）→ HCA 共 31 |
| 全局 KV 份数（存储维） | 4（C `kv_source_layer_ids` [2,8,14,20]） | 41（每个 CSA/HCA 层各一份） | 61 |
| 自带 indexer 的层 | 8（C `index_source_layer_ids`） | 21（全部 CSA 层） | 30 |
| 查询头 × 头维 | 64 × 512（P；C） | 64 × 512 | 128 × 512 |
| RoPE 维 | 64（C `qk_rope_head_dim`；K） | 64（V4 P §2.3.3） | 64 |
| 查询压缩维 $d_c$ | 1280（P；C `q_lora_rank`） | 1024 | 1536 |
| 输出分组 $g$ × $d_g$ | 8 × 1024（P；C） | 8 × 1024 | 16 × 1024 |
| indexer 头 × 维 | 32 × 128（P；C） | 64 × 128 | 64 × 128 |
| 注意力 top-k | 512（P；C） | 512 | 1024 |
| 候选池 | 2048 块 × 8 = 16,384（仅解码器，P；C） | — | — |
| $n_{win}$ | 128（P；C `sliding_window`） | 128 | 128 |
| 逐头 Q-norm | 无（F “only used in V4, not V4.1”；K） | 有（K） | 有 |
| MoE | 1 共享 + 384 路由，中间维 2304，激活 6（P；C） | 1 + 256，2048，激活 6 | 1 + 384，3072，激活 6 |
| Hash 路由层 | 无（P 图 3；C 无字段） | 前 3 层（P；C `num_hash_layers: 3`） | 前 3 层 |
| 亲和函数 / 路由缩放 | sqrtsoftplus / 1.5（C） | sqrtsoftplus / 1.5（C） | sqrtsoftplus / 2.5（C） |
| SwiGLU 钳位 | 10（P “clamping … at a threshold of 10”；C `swiglu_limit`） | 10（C） | 10（C） |
| mHC | $n$=4，Sinkhorn 20 次，Single-Pass（P；C） | 4，20（P；C） | 4，20 |
| 投机解码 | DSpark 3 块（P；C `num_nextn_predict_layers: 3`） | MTP 深度 1 | MTP 深度 1 |
| Engram | 196B，两模块，层 1 与 14（P；C） | 无 | 无 |
| 视觉 | DeepSeek-ViT 32 层 / 1024 / 16 头 / patch 14，投影 MLP 2 层 5120（P；C） | 无 | 无 |
| 参数 | 552B 主干 + 196B Engram（P） | 284B（P） | 1.6T（P） |
| 激活 | prefill 8B / decode 16B（P） | 13B | 49B |
| 上下文 | 1M；YaRN factor 16，原长 65,536（C） | 1M；同（C） | 1M；同 |
| RoPE 基频 | 10,000；压缩层 160,000（C `compress_rope_theta`；K：纯 SWA 层不加 YaRN） | 同（C） | 同 |
| `rms_norm_eps` | $10^{-20}$（C；K） | $10^{-6}$（C） | $10^{-6}$（C） |
| 权重量化 | FP8 E4M3，32×32 块，UE8M0 缩放；专家 FP4（C `weight_block_size [32,32]`；K） | FP8，128×128 块，UE8M0；专家 FP4（C） | 同 V4-Flash |
| 主 KV 缓存 | FP4：256 B E2M1 + 32 B E4M3 = **288 B/条目**（P §2.4.4；F） | FP8 NoPE 448 B + BF16 RoPE 128 B + 8 B 缩放 = **584 B**（F；V4 P §2.3.4） | 584 B |
| indexer 键缓存 | MXFP4：64 B + 4 B E8M0 = **68 B/条目**（K `fp4_block_size = 32`；P “MXFP4”） | FP4，块 32：68 B（V4 K `fp4_block_size = 32`） | 68 B |
| SWA KV 缓存 | FP8 全 512 维 + 16 B E8M0 = **528 B/条目**（F；K `act_quant(kv, fp8_block_size, ...)`，`fp8_block_size = 32`；P “retain FP8”） | 584 B（F；V4 K `act_quant(kv[..., :-rd], 64, ...)`） | 584 B |
| 预训练 token | 45T 多模态（P） | 32T（V4 P §4.2.2） | 33T（V4 P §4.2.2；摘要写 “more than 32T”） |
| batch | 固定 100.6M token（P） | 递增到 75.5M（V4 P） | 最大 94.4M |
| 学习率 | 2000 步预热 → 2.6e-4 至 28T → 余弦至 2.6e-5（28T–40T）→ 保持到 45T（P） | 2.7e-4 → 2.7e-5（V4 P） | 2.0e-4 → 2.0e-5 |
| 序列长度与稀疏 | 64K 起步即稀疏；34T 时扩到 1M（P） | 4K→16K→64K→1M；前 1T 稠密预热（V4 P） | 同，稠密阶段更长 |
| 均衡 | 偏置更新速度 0.001（图、文各一套）；序列级损失 1e-4（P） | 0.001；1e-4（V4 P） | 同 |

### 3-B. V4.1-Flash 逐层配置（C `compress_ratios` / `kv_source_layer_ids` / `index_source_layer_ids` / `candidate_source_layer_id` / `engram_layer_ids` / `dspark_target_layer_ids` + P §4.2.1）

| 层（0 起） | 角色 | 注意力 | $m$ | CSA2 模式 | 写全局 KV？ | 跑 indexer？ | 备注 |
|---|---|---|---|---|---|---|---|
| 0 | 编码器 | 纯 SWA | — | — | 否 | 否 | |
| 1 | 编码器 | 纯 SWA | — | — | 否 | 否 | Engram 模块 1 |
| 2 | 编码器 | CSA2 | 2 | Full | **是**（KV 源 #1） | 是 | |
| 3–7 | 编码器 | CSA2 | 2 | Reuse | 否 | 否 | 复用层 2 的 KV 与 top-k |
| 8 | 编码器 | CSA2 | 2 | Full | **是**（#2） | 是 | |
| 9–13 | 编码器 | CSA2 | 2 | Reuse | 否 | 否 | |
| 14 | 编码器 | CSA2 | 2 | Full | **是**（#3） | 是 | Engram 模块 2 |
| 15–19 | 编码器 | CSA2 | 2 | Reuse | 否 | 否 | 第 19 层输出 = $H_{L/2}$ |
| 20 | 解码器 | CSA2 | 1 | Full | **是**（#4，由 $H_{L/2}$ 投影） | 是（全扫；造候选池） | `candidate_source_layer_id` |
| 21–23 | 解码器 | CSA2 | 1 | Reuse | 否 | 否 | |
| 24 / 28 / 32 / 36 | 解码器 | CSA2 | 1 | Reindex | 否 | 是（只在 16,384 候选里打分） | |
| 25–27 / 29–31 / 33–35 / 37–39 | 解码器 | CSA2 | 1 | Reuse | 否 | 否 | 37–39 是 DSpark 的取特征层 |
| 40–42（`mtp.0–2`） | DSpark 草稿块 | 纯 SWA（窗口 128） | — | — | 否 | 否 | 每块 128 路由专家激活 3 |

计数：纯 SWA 2、Full 4、Reindex 4、Reuse 30（“vast majority”= 75%）。`compress_ratios` 数组长 43：前 40 项是主干，最后 3 个 0 是 DSpark 块——**不要把 43 当层数**。safetensors 佐证：`compressor.wkv` 4 个（层 2/8/14/20），`compressor.wgate` 3 个（层 20 是 $m=1$ 无门），`indexer.wk` 4 个，`indexer.wq_b` 8 个。

### 3-C. 参数量（由 48 个 safetensors 分片头的张量形状求和；FP4 专家权重以 `I8` 存储、每字节两个值，已 ×2；缩放张量不计）

| 组 | 参数 | 说明 |
|---|---|---|
| 路由专家（40 层 × 384） | 543.58B | 每专家 3 × 5120 × 2304 = 35.39M |
| 共享专家（40 层） | 1.42B | |
| 注意力（含 compressor / indexer） | 5.13B | 每层均值 128.2M |
| 其他（norm、mHC、gate、sink） | 0.12B | |
| 词嵌入 / 输出头 | 0.662B / 0.662B | 不共享（`tie_word_embeddings: false`） |
| **主干合计** | **551.57B**（加 Engram 投影 551.88B） | 论文 “552B backbone” |
| Engram 表 | 196.61B | 两表行数 384,006,168 与 384,016,682，每行 256 维，FP8 |
| Engram 投影 | 0.315B | 两个 `wkv`：6144 → 25,600 |
| DSpark（3 块） | 14.23B | 每块 128 路由专家 |
| 视觉（ViT + aligner） | 0.485B | |
| **checkpoint 总计** | **763.21B** | 与 HF API `safetensors.total` = 763,205,315,794 一致 |

激活（笔者复算，§4-C）：decode ≈ 16.48B（含嵌入与输出头；不含嵌入 15.82B），prefill ≈ 8.24B（含嵌入；不含 7.58B）。论文 8B / 16B。

### 3-D. KV 条目字节格式（每条目）

| 模型 / 缓存 | 数据 | 缩放 | 合计 | 每元素比特 | 来源 |
|---|---|---|---|---|---|
| V3.2 MLA（FP8） | 512 FP8 + 64 BF16 | 4 × FP32（每 128 个） | 656 B | 9.11（按 576 维） | F |
| V3.2 indexer 键 | 128 FP8 | 1 × FP32 | 132 B | 8.25 | 第 14 课笔记（`model.py`） |
| V4 主 KV / SWA | 448 FP8 + 64 BF16 | 7 × E8M0（每 64 个）+ 1 B 填充 | 584 B | 9.125 | F；V4 P |
| V4 / V4.1 indexer 键 | 128 × E2M1 = 64 B | 4 × E8M0（每 32 个） | 68 B | 4.25 | K；MX 表 1 |
| **V4.1 主 KV（FP4）** | 512 × E2M1 = 256 B | 32 × E4M3（每 16 个） | **288 B** | 4.5 | P §2.4.4；F；K |
| V4.1 SWA KV（FP8） | 512 FP8（含 RoPE 维） | 16 × E8M0（每 32 个） | 528 B | 8.25 | F；K |

### 3-E. 训练与后训练超参数（全部 P）

| 项 | 值 | 位置 |
|---|---|---|
| AdamW | $\beta_1$ 0.9，$\beta_2$ 0.95，$\varepsilon$ $10^{-20}$，wd 0.1；用于 norm 权重、偏置、缩放因子等非矩阵参数，但 wd 只施加于 norm 权重（§2.5：“normalization-layer weights are also subject to weight decay, whereas biases and scaling factors are not”） | §4.2.2、§2.5 |
| Muon | 动量 0.95，wd 0.1，更新 RMS 缩放 0.18；Q/K 权重 head-wise | §4.2.2、§2.5 |
| Sinkhorn 平衡更新 | 用于 Engram 表、词嵌入、预测头；$K=11$，$\tau=10^{-3}$，$\varepsilon=10^{-20}$，$\gamma=0.18$，Nesterov，无 wd | §2.5、§4.2.2 |
| Engram 学习率 | ×5 | §4.2.2 |
| 视觉编码器 | 冻结到学习率衰减开始，之后以更小学习率联合训练 | §2.5 |
| 数据 | 45T；文本 : 多模态 = 7 : 1；padding ≤ $10^{-4}$ | §4.1 |
| ViT 预训练 | SigLIP 损失，约 47B 图文对，224²；自回归阶段接 4B MoE，236B token，544²–1344² | §4.2.2 |
| 推理强度 API 档位 | max = 100，high = 75，low = 50 | 表 2 |
| OPD 教师数 | “over 40” | §5.2.4 |
| 评测采样 | 推理题 temperature 1.0、top-p 1.0；代码智能体 temperature 1.0、top-p 0.95、1M 上下文；视觉智能体 512K 上下文 | §5.3.1 |
| 脚手架对比 | DeepSWE 每任务 N = 8，TB2.1 N = 3，max_steps = 500 | 表 4 注 |

---

## 4. 复算的数字

所有数字由 Python 算出（脚本：scratch `calc.py`、`qsim.py`，表达式照抄于此）。

### 4-A. 全局 KV 字节/token（配置 → 890）

| 量 | 表达式 | 值 |
|---|---|---|
| V4.1 主 KV 条目 | `512//2 + 512//16` | 288 B |
| indexer 键条目（V4/V4.1） | `128//2 + 128//32` | 68 B |
| V4.1 每 token 全局条目数 | `3*(1/2) + 1*1`（编码器 3 份 $m$=2 + 解码器 1 份 $m$=1） | 2.5 |
| **V4.1 全局 KV** | `2.5*(288+68)` | **890 B** ✔（= 摘要、图 1(b)） |
| 其中编码器 / 解码器 | `1.5*356` / `1*356` | 534 B（60%） / 356 B（40%） |
| V4 主 KV 条目 | `448 + 64*2 + 8` | 584 B |
| V4-Flash CSA / HCA 层数 | `compress_ratios` 计数 | 21 / 20 |
| **V4-Flash 全局 KV** | `(21/4 + 20/128)*584 + (21/4)*68` | **3,514.25 B** ✔（图 1(b) 3,514） |
| 其中 CSA 主 KV / HCA 主 KV / indexer | `5.25*584` / `0.15625*584` / `5.25*68` | 3,066 / 91.25 / 357 B |
| V4-Pro 全局 KV | `(30/4 + 31/128)*584 + (30/4)*68` | 5,031.44 B |
| V3.2 全局 KV | `61*(656+132)` | 48,068 B ✔（图 1(b)） |
| DeepSeek LLM 67B（图中 “V1”） | `2*95*8*128*2` | 389,120 B ✔（图 1(b)） |
| V4-Flash / V4.1 | `3514.25/890` | 3.949（“3.9× smaller”“roughly 1/4”） |
| V1 / V4.1 | `389120/890` | 437.2（“437-fold”） |
| V1 / V3.2 | `389120/48068` | 8.095（“8.1×”） |
| V3.2 / V4-Flash | `48068/3514.25` | 13.68（“13.7×”） |
| V4-Pro / V3.2 | `5031.44/48068` | 10.47%（V4 摘要 “10%”） |
| V4-Flash / V3.2 | `3514.25/48068` | 7.31%（V4 §1 “7%”） |
| V4.1 主 KV / V4 主 KV（每条目） | `288/584` | 0.493（“nearly halves”） |
| V4.1 若主 KV 仍用 FP8（528 B） | `2.5*(528+68)` | 1,490 B |
| V4.1 若不跨层共享（38 份各存） | `(18/2 + 20/1)*356` | 10,324 B（共享省 11.6×） |
| 分解：先改结构（保持 V4 精度） | `2.5*(584+68)` | 1,630 B → 结构因子 `3514.25/1630` = 2.156× |
| 再改精度 | `1630/890` | 1.831×；结构占对数降幅 `ln2.156/ln3.949` = 56%，精度 44% |
| 统一按 BF16 比架构 | V4-Flash `5.40625*1024+5.25*256`；V4.1 `2.5*(1024+256)` | 6,880 B vs 3,200 B（2.15×） |

### 4-B. 代际阶梯（每 token 全局 KV；“原生精度”= 该代实际部署/论文口径）

| 模型 | 结构 | 表达式 | 字节/token | 精度口径 | 1M（$2^{20}$）token 时 |
|---|---|---|---|---|---|
| GPT-3 175B | MHA，96 层，96 头 × 128（2005.14165 表 2.1） | `2*96*96*128*2` | 4,718,592 | FP16/BF16（论文未写推理精度；按 2 字节） | 4,608 GiB |
| Llama-3-70B | GQA-8，80 层，头维 8192/64 = 128（2407.21783 表 3；config） | `2*80*8*128*2` | 327,680 | BF16 | 320 GiB |
| DeepSeek LLM 67B | GQA-8，95 层（config） | `2*95*8*128*2` | 389,120 | BF16 | 380 GiB |
| DeepSeek-V2 | MLA，60 层，512 + 64（config） | `60*576*2` | 69,120 | BF16；论文部署 “6 bits on average” → `60*576*6/8` = 25,920 | 67.5 GiB（6 bit：25.3 GiB） |
| DeepSeek-V3 | MLA，61 层（config；不计 MTP 层） | `61*576*2` | 70,272 | BF16 | 68.6 GiB |
| DeepSeek-V3.2 | MLA + FP8 indexer 键，61 层 | `61*(656+132)` | 48,068 | FP8（FlashMLA 656 B 格式） | 46.9 GiB |
| DeepSeek-V4-Pro | 30 CSA + 31 HCA | 见 4-A | 5,031 | FP8+BF16 主 KV，FP4 indexer | 4.91 GiB |
| DeepSeek-V4-Flash | 21 CSA + 20 HCA | 见 4-A | 3,514 | 同上 | 3.43 GiB |
| **DeepSeek-V4.1-Flash** | 4 份 CSA2（3×$m$2 + 1×$m$1） | 见 4-A | **890** | FP4 主 KV，MXFP4 indexer | **0.87 GiB** |

相对 V4.1 的倍数：GPT-3 5,302×；Llama-3-70B 368×；67B 437×；V2 BF16 77.7×；V3 79.0×；V3.2 54.0×；V4-Pro 5.65×；V4-Flash 3.95×。
注意：以上**不含** SWA 状态（每请求常数，见 4-D），也不含 CSA/HCA 尚未凑满一组的尾部状态。GPT-3 有一半层是 “locally banded sparse attention”（2005.14165 §2.1），若只缓存局部窗口，其实际缓存会更小；表中按全部 96 层全量缓存计，是上界。

### 4-C. 激活参数（8B / 16B）

| 量 | 表达式（safetensors 形状求和） | 值 |
|---|---|---|
| 每层激活路由专家 | `6/384 * 543.58e9 / 40` | 212.3M |
| 每层共享专家 | `1.4156e9/40` | 35.4M |
| 每层注意力均值 | `5.127e9/40` | 128.2M |
| decode：40 层 + 其他 + 嵌入 + 头 | `40*(212.3M+35.4M) + 5.127B + 0.118B + 0.662B + 0.662B` | 16.48B（去掉嵌入查表 15.82B） |
| prefill：编码器 20 层 + 第 20 层 compressor/indexer 键 + 嵌入 | 按层号 < 20 求和 + 2.69M | 8.24B（去掉嵌入 7.58B） |
| Engram 每 token 读取 | `2 模块 * 24 行 * 256` | 12,288 个表参数（另有投影 0.315B 全激活） |
| 主干 / V4-Pro 总参数 | `552/1600` | 0.345（“1/3 total parameters”） |
| 激活 / V4-Pro 激活 | `16/49`，`8/49`，`12/49` | 0.327，0.163，0.245（“1/4 activated”只与 8B/16B 的平均 12B 吻合，见 §7 E16） |

### 4-D. SWA 状态、回放与 prefill

| 量 | 表达式 | 值 |
|---|---|---|
| V4-Flash SWA 状态（每请求） | `43*128*584` | 3,214,336 B = 3.07 MiB ≈ 915 个 token 的全局 KV |
| V4-Pro SWA 状态 | `61*128*584` | 4,559,872 B = 4.35 MiB |
| V4.1 SWA 状态（主干） | `40*128*528` | 2,703,360 B = 2.58 MiB ≈ **3,037** 个 token 的全局 KV |
| V4.1 + DSpark 3 块 | `43*128*528` | 2,906,112 B = 2.77 MiB |
| V4 “SWA 约为压缩条目 8 倍” | `43/(21/4+20/128)` | 7.95（按条目数；按字节含 indexer 7.15） |
| 精确回放（V4-Flash，Zero SWA Caching） | `43*128` | 5,504 token |
| 精确回放（V4.1 编码器侧 $L\times n_{win}$） | `40*128` | 5,120 token |
| 精确回放（V4.1 解码器侧 $\tfrac L2 n_{win}$） | `20*128` | 2,560 token |
| Bounded Replay | $n_{win}$ | 128 token（40× / 20× 少） |
| CED prefill 相对计算量 | `(N*20+128*20)/(N*40)` = `0.5 + 64/N` | N=1K：0.5625；4K：0.5156；32K：0.5020；128K：0.5005；1M：0.50006 |
| 持久化 1/8 | `(1/2)*(890/3514.25)` | 0.1266 ≈ 1/7.9（若 V4 持久化中 SWA 恰占一半） |

教学点（笔者推算）：V4.1 的全局 KV 小到 890 B 后，SWA 状态（2.58 MiB）相当于 3,037 个 token 的全局 KV；对短回合、多轮的智能体会话，SWA 反而成了持久化缓存的大头——这正是 V4.1 把 SWA 从 SSD 挪到 10% 主机内存池、并用 Bounded Replay 兜底的动机。

### 4-E. 分层索引器与稀疏选择

| 量 | 表达式 | 值 |
|---|---|---|
| 候选池 | `2048*8` | 16,384 位置；1M 上下文时占 `16384/2**20` = 1.5625% |
| V4.1 每个解码 token 被打分的条目（1M） | `3*N/2 + N + 4*16384` | 2,686,976（无分层时 `3*N/2 + N + 4*N` = 6,815,744） |
| 同上（128K） | | 393,216（无分层 851,968） |
| V4-Flash 同口径（1M） | `21*N/4` | 5,505,024 |
| × indexer 头数 | V4.1 `2686976*32`；V4-Flash `5505024*64` | 8.60e7 vs 3.52e8（4.1×） |
| 编码器 top-512 覆盖的原始 token | `512*2` | 1,024（$m$=2）；解码器 512（$m$=1） |

### 4-F. 解码 FLOPs（图 2）

图 2 数字化读数（像素→对数坐标；网格线校准：10 在 y=1277，每十倍 477 px；4K 在 x=504，每 ×4 为 586 px；误差约 ±3%），单位 GFLOPs（加权：BF16 1、FP8 0.5、FP4 0.25）：

| 上下文 | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V1 | 146 | 159 | 185 | 236 | 338 | 542 | 953 | 1,776 | 3,359 |
| DeepSeek-V3.2 | 77 | 79 | 83 | 91 | 108 | 138 | 206 | 333 | 591 |
| DeepSeek-V4-Flash | 16.4 | 16.6 | 16.9 | 17.6 | 19.0 | 22.0 | 27.2 | 38.3 | 60.1 |
| DeepSeek-V4.1-Flash | 20.1 | 20.2 | 20.3 | 20.4 | 20.5 | 20.8 | 21.5 | 22.8 | 25.6 |

- 论文 “only 1/4”：`25.6/20.1` = 1.27 ✔。
- **笔者的解析增长项**（$N$ 为上下文 token 数，每 token 的“随长度增长部分”）：
  - V4.1：只有 indexer 随 $N$ 长（核心注意力固定 512 + 128 个条目）：`0.25*2*32*128*(1.5*N + N)` = 5,120·N → 1M 时 5.37 GFLOPs（读数增量 5.5）。
  - V4-Flash：indexer `0.25*2*64*128*(21*N/4)` = 21,504·N，加 HCA 稠密注意力 `1.0*2*2*64*512*(20*N/128)` = 20,480·N → 1M 时 44.0 GFLOPs（读数增量 43.7）。
  - V3.2：FP8 indexer `0.5*2*64*128*61*N` = 499,712·N → 524 GFLOPs（读数增量 514）。
  - V1（67B）：`2*2*95*8192*N` → 3,264 GFLOPs（读数增量 3,213）。
  - 交叉点：`(20.1-16.4)e9/(41984-5120)` ≈ 100,369 token ≈ 98K——**短于约 100K 时 V4.1 的解码 FLOPs 反而比 V4-Flash 高**（16B vs 13B 激活）。
- 常数项粗估（笔者假设 FP8×FP8 与 FP4×FP8 GEMM 都按 0.5 计、注意力核按 BF16 计）：V4.1 = GEMM 16.69 + 核心注意力 `(38*640+2*128)*131072` 3.22 ≈ 19.9（读数 20.1）；V4-Flash ≈ 15.5（读数 16.4）。只作量级参考。

### 4-G. FP4 数值

| 量 | 表达式 | 值 |
|---|---|---|
| 最大可表示幅值（无全局缩放） | `448*6` | 2,688 |
| 主 KV 幅值上界 | `sqrt(512)` | 22.63（训练中实测约 10） |
| E2M1 正值集合 | — | {0, 0.5, 1, 1.5, 2, 3, 4, 6} |
| E4M3 最小次正规数 | `2**-9` | 0.001953（代码把 amax 下限钳到 `6*2**-9`，即缩放 ≥ $2^{-9}$） |
| 单位 RMS 高斯向量（2000 × 512）相对 RMSE | `qsim.py` | E2M1 + E4M3/16：9.51%（余弦 0.9955）；MXFP4 E8M0/32：11.53%（0.9934）；E8M0/16：11.27%；FP8 E4M3 + E8M0/64：2.66%（0.99965） |
| 算例 16 维块（§6 练习 7） | `qsim.py` | E4M3 缩放 0.375 → 相对 RMSE 8.91%；E8M0 缩放 0.5 → 12.09% |

高斯模拟只说明格式之间的相对精度（真实 KV 分布未知）；它与论文 “despite the higher accuracy of alternative formats in our experiments” 的方向一致。

### 4-H. 训练、数据、评测、价格

| 量 | 表达式 | 值 |
|---|---|---|
| 文本 / 多模态 token | `45*7/8`，`45/8` | 39.375T / 5.625T |
| 预训练步数 | `45e12/100.6e6` | ≈ 447,316 步 |
| 1M 扩展发生在 | 34T / 45T | 75.6% 处（学习率衰减段 28T–40T 之内） |
| Engram 参数 | `(384006168+384016682)*256` | 196.61B ✔ |
| Engram 每表平均行数 | `384006168/24` | 16,000,257（“approximately 16M”，24 = 3 阶 × 8 头） |
| 图像 token（方图，每行末一个换行 token，首尾各 1） | `2 + n*(n+1)` | 31×31 网格：994（≤1024 上限）；32×32：1,058（超限，代码会缩小到 31 格 = 1,302 px） |
| 544² 最小图 | `ceil(544/14)`=39 patch → `ceil(39/3)`=13 格 | 184 token |
| mHC 访存（$n$=4，$d$=5120，单位：元素） | `(4n+4)d`，`(3n+2)d`，`(2n+2)d` | 102,400 → 71,680 → 51,200 |
| BPB 降幅（图 6） | `(old-new)/old` | 相对 V4-Flash-Base：8.59% / 7.62% / 12.66%（三集平均 9.62%）；相对 V4-Pro-Base：4.41% / 3.41% / 7.95%（平均 5.26%） |
| 推理强度 25→100 输出长度 | `11.4/4.6`，`86.1/29.1` | 2.48×，2.96×（附录 B.3 “2.0–3.1×”） |
| MathArena Apex 增益 | `65.6-25.3` | 40.3 ✔ |
| API 缓存命中 / 未命中价格比 | `0.003/0.15`；V4-Pro `0.022/0.66` | 2.0%；3.3% |
| V4-Pro / V4.1-Flash 价格比 | 未命中 `0.66/0.15`；输出 `1.98/0.60` | 4.4×；3.3× |

---

## 5. 图表清单与重绘建议

### 5-A. 论文图表清单（arXiv v1；PDF 页码）

| 编号 | 页 | 标题要点 | 显示内容 | 本课用途 |
|---|---|---|---|---|
| 图 1(a) | 1 | agentic benchmarks | 4 组柱：TB3.0（V4.1 30.0 / Kimi-K3 17.7 / GLM-5.3 28.3 / Opus5 43.3 / GPT5.6-Sol 34.4）；DeepSWE v1.1（74.2 / 67.5 / 66.9 / 74.0 / 73.0）；CyberGym（88.1 / 80.0 / 84.5 / — / 84.5，Opus 无）；AutomationBench（54.8 / 46.7 / 48.8 / 50.3 / 45.8） | 只在“能力没掉”处提一句，不重绘 |
| 图 1(b) | 1 | Global KV Cache Per Token (Bytes) | 横条：V1 389,120（2023.11）、V3.2 48,068（2025.12）、V4-Flash 3,514（2026.04）、V4.1-Flash 890（2026.09）；箭头 8.1×、13.7×、3.9× | **重绘并扩展成阶梯图（F1）** |
| 图 2 | 5 | Single-token Decode FLOPs vs context | 4 条曲线，对数纵轴 10–3000 GFLOPs，横轴 4K–1024K（×4 刻度） | **重绘（F9），数据见 §4-F** |
| 图 3 | 7 | Overall architecture | 左：编码器（SWA×2 → [CSA2(2,Full)+MoE, CSA2(2,Reuse)+MoE ×5] ×3），右：解码器（CSA2(1,Full)、Reuse×3；[Reindex, Reuse×3]×4），底部 Engram / 文本嵌入 / 视觉嵌入 / Single-Pass HC，右上 DSpark，右下分层索引器与候选池 | **重绘（F2、F3）** |
| 图 4 | 10 | Three operating modes of CSA2 | 三栏：Full / Reindex / Reuse；绿 = 本层计算，黄 = 复用自最近 Full 层的主 KV 与 indexer K，红 = 复用的 top-k | **重绘（F4）** |
| 图 5 | 11 | Hierarchical Sparse Indexer | Full 层对全部位置打分，选 top-512 并按块最大分挑块（蓝框），组成共享候选池；Reindex 层只在池内选 top-512 | **重绘（F5）** |
| 算法 1 | 14–15 | Momentum update with Sinkhorn balancing | 伪代码 20 行 | 只在“优化器”侧栏提及 |
| 表 1 | 23–24 | 基座对比 | 19 个基准 + 4 个多模态 | 摘 5 行放进一张小表 |
| 图 6 | 25 | BPB on held-out sets | 3 组柱（见 C53） | 可选，配 §7 E15 讨论 |
| 图 7 | 27 | RL scaling on code-agent benchmarks | RL 步数 vs 分数，断开的线段 = 模型合并后重启的新 RL 轮次；延长到 1M 上下文后 TB v3.0 继续涨 | 不重绘（无数据） |
| 图 8 | 28 | RL across Claude Code versions / heterogeneous scaffolds | 在 DeepSWE v1.1 上的曲线 | 不重绘 |
| 表 2 | 30 | API 档位 ↔ $b$ | max 100 / high 75 / low 50 | 放进正文小表 |
| 表 3 | 33 | 与开闭源模型对比 | 21 行 × 7 列 | 摘 6–8 行 |
| 图 9 | 35 | Performance and output length vs effort | 3 个面板（8 个推理基准平均、DeepSWE、TB2.1），实线 Pass@1、虚线长度 | 用端点数字画简图（F13） |
| 表 4 / 表 5 | 34 / 47 | 跨脚手架 / Claude Code 各版本 | 见 C56 | 可选 |
| 图 10 | 36 | Multi-agent vs single-agent | 两面板，按墙钟截止时间 | 可选，端点数字见 C57 |
| 图 11 / 图 12 | 49 / 50 | 附录：强度 × 脚手架；8 个推理基准逐个曲线 | — | 不用 |

注：arXiv HTML 只有图 1(a)(b)、图 2、图 6 是 PNG；图 3、4、5、7–12 是 PDF 矢量图，本机没有 `pdftoppm`，只提取了其中文字标签（上表的布局描述来自这些标签与图注）。

### 5-B. 本课建议绘制的图（F1–F13 必画，F14–F15 可选；颜色按 style-guide §4）

**F1 KV 字节/token 代际阶梯图（计算得出，对数横轴）**。条目与数值（字节）：GPT-3 175B 4,718,592（FP16）；Llama-3-70B 327,680（BF16）；DeepSeek LLM 67B 389,120（BF16）；V2 69,120（BF16），附一个浅色子条 25,920（6 bit 部署）；V3 70,272（BF16）；V3.2 48,068（FP8）；V4-Pro 5,031（FP8+BF16 / FP4 idx）；V4-Flash 3,514；V4.1-Flash 890。每条右侧标“×倍于 V4.1”（5,302 / 368 / 437 / 77.7 / 79.0 / 54.0 / 5.65 / 3.95 / 1）。颜色：缓存条用 `--fig-amber`，V4.1 用 `--fig-orange` 强调；精度写在条内小字。横轴 $10^2$–$10^7$。另起一行小注：“不含 SWA 状态（V4.1 每请求 2.58 MiB）”。

**F2 CED 结构图（示意）**。40 个小方块排成两列：左列层 0–19（编码器），右列层 20–39（解码器）。标出：$H_{L/2}$（层 19 输出）→ 箭头 → 层 20 的 $W_{20}^{KV}$（`--fig-orange`）→ “解码器全局 KV（1 份，$m$=1）”（`--fig-amber`）→ 虚线分发给层 21–39。每个方块内再画一个小 “SWA” 标记（每层都有，`--fig-amber` 细框）。上方两条时间线：prefill（提示的 $T$ 个 token 只走左列 + 层 20 的投影；最后 128 个 token 走完右列）；decode（每个新 token 走全部 40 层 + 输出头）。旁注：prefill 8.24B / decode 16.48B（§4-C）。

**F3 逐层模式色条（计算得出）**。一行 40 格 + 3 格 DSpark：灰 = 纯 SWA（0,1,40–42），深橙 = Full（2,8,14,20），紫 = Reindex（24,28,32,36），浅色 = Reuse（其余 30 格）；格下标 $m$（2 或 1）；Engram 标记在 1 与 14；DSpark 取特征层 37–39 加小点。数据即 §3-B。

**F4 CSA2 三种模式（据论文图 4 重绘）**。三栏，每栏自下而上：Main Q、SWA KV（本层，绿/蓝）→ Main KV、Indexer K（Full：本层算；Reindex/Reuse：复用，`--fig-amber`）→ Indexer（Full、Reindex 有；Reuse 无）→ Top-K Indices（Reuse：复用，`--fig-red` 细框表示“来自别层”）→ Selection → Selected Main KV ⊕ SWA KV → Core Attention → Output。标注：“每层都有自己的 Q 与 SWA KV”。

**F5 分层稀疏索引器（据论文图 5 重绘，带真实数字）**。一行位置方格（示意 64 格，块长 8 → 8 块）；第 20 层：全部位置打分，取 top-512（绿点）与 top-2048 块（蓝框；示意图中取 3 块）+ 最新块强制保留；第 24/28/32/36 层：只在蓝框内打分，各自选 top-512。旁注真实规模：1M 上下文 1,048,576 位置 → 候选池 16,384（1.56%）；被打分条目/解码 token 2,686,976（无分层 6,815,744）。

**F6 890 字节的拆账（计算得出，堆叠条）**。V4-Flash 3,514.25 = CSA 主 KV 3,066 + HCA 主 KV 91.25 + indexer 357；V4.1 890 = 编码器主 KV 432（`1.5*288`）+ 编码器 idx 102（`1.5*68`）+ 解码器主 KV 288 + 解码器 idx 68。中间加一根“V4.1 结构、V4 精度”1,630 的过渡条，标“结构 2.16×”“精度 1.83×”。

**F7 字节布局图（示意，按 F 的格式）**。四行，每格 1 字节按比例：V3.2 656（512 FP8 | 16 FP32 缩放 | 128 BF16 RoPE）；V4 584（448 FP8 | 128 BF16 RoPE || 8 缩放行）；V4.1 SWA 528（512 FP8 || 16 E8M0）；V4.1 主 KV 288（256 = 512 个 E2M1，每字节两值，偶数下标在低 4 位 || 32 E4M3）。低精度数据用 `--fig-orange`，缩放用 `--fig-violet`，BF16 用 `--fig-blue`。

**F8 E2M1 数轴与分块量化（计算得出）**。上：E2M1 可表示值 ±{0,0.5,1,1.5,2,3,4,6} 画在数轴上（间距不均）。下：§6 练习 7 的 16 维向量 x2（数值见练习）原值（黑点）vs E4M3 缩放 0.375 的反量化值（橙点）vs E8M0 缩放 0.5 的反量化值（红叉）；标注相对 RMSE 8.91% vs 12.09%。

**F9 解码 FLOPs 随上下文（据论文图 2 重绘，数据见 §4-F）**。对数纵轴 10–4000，横轴 4K–1M（对数，刻度 4/16/64/256/1024K）。四条线：V1 灰虚线、V3.2 深灰点划线、V4-Flash 浅蓝、V4.1 蓝。可叠加笔者的解析增长项（细虚线）：$F_{4.1}(N)=20.1\text{G}+5120N$，$F_{4F}(N)=16.4\text{G}+41984N$；标交叉点约 100K。

**F10 CED 的 prefill 省多少（计算得出）**。横轴提示长度 $T$（1K–1M，对数），纵轴“相对全量 prefill 的计算比例” $0.5+64/T$；再画一条“若解码器 SWA 精确重建” $0.5+\tfrac{20\cdot128\cdot20}{40T}=0.5+1280/T$ 对比（笔者推算：精确重建要让解码器 20 层处理最后 2,560 个 token）。标注 $T$=4K 时 0.516 vs 0.8125。

**F11 存储层级图（示意）**。三层盒子：HBM（全局 KV 890 B/token + 活跃请求的 SWA 状态 2.58 MiB/请求）；主机 DRAM（每机 10% 做分布式 SWA 池，TTL 数分钟；Engram 表经 RDMA 预取）；SSD（持久化全局 KV，保证 ≥72 小时，LRU）。两条路径：前缀命中 + SWA 命中 → 直接续算；前缀命中 + SWA 未命中 → Encoder SWA Bounded Replay（回放 128 token）。对比面板：V4 的 SSD 里同时有全局 KV 与 SWA KV（各约一半）。颜色：存储 `--fig-amber`。

**F12 Bounded Replay 的感受野（示意 + 公式）**。纵轴层号（示意 4 层），横轴位置；精确重建时最后一个 token 的 SWA 依赖是一个倒三角，底宽 $L\times n_{win}$（V4.1：5,120）；Bounded Replay 只取宽 $n_{win}$=128 的矩形，左边界处每个查询的可见区间 $[\max(s,i-W+1),i]$ 被截断（截断区用 `--fig-red` 斜线）。标注：全局 KV 部分精确（直接读缓存），误差只来自截断的 SWA。

**F13 持久化 1/8 的两步（计算得出）**。三根条：V4 持久化 = 全局 1.0 + SWA ≈ 1.0（示意“nearly half”）；去掉 SWA → 1.0；全局再 ÷3.95 → 0.253；最终相对 V4 = 0.127 ≈ 1/8。

**F14（可选）Single-Pass mHC 的依赖与访存**。两张小依赖图：式 (3)(4)(5) 三核串行（$A_l$ 要等 (4) 的全维归约），式 (6) 用 $A_{l-1}$ 后三步融成一核；右侧柱状：$(4n+4)d$ = 102,400 → $(3n+2)d$ = 71,680 → $(2n+2)d$ = 51,200 元素/token（$n$=4，$d$=5120）。

**F15（可选）推理强度旋钮**。左：$k(b)=k_0e^{-(b-b_{\min})/\tau}$ 的形状（$k_0,\tau$ 论文未给，用归一化坐标）；右：图 9 端点——8 个推理基准平均 67.1%→76.3%，DeepSWE 66.0%→74.2%，TB2.1 82.4%→90.6%，输出长度约 ×2.5；API 三档竖线 50/75/100。

---

## 6. 算例与练习素材（附答案）

1. **（计算）从 config 算 890。** 已知主 KV 512 维 E2M1 + 每 16 维一个 E4M3 缩放；indexer 键 128 维 E2M1 + 每 32 维一个 E8M0；KV 源层 [2,8,14,20]，压缩比分别为 2,2,2,1。求每 token 全局 KV 字节。
   答：条目 288 B + 68 B = 356 B；每 token 条目数 $3\times\tfrac12+1=2.5$；$2.5\times356=890$ B。编码器占 534 B（60%），解码器 356 B（40%）。

2. **（计算）V4-Flash 的 3,514 从哪来？** 43 层，前 2 层纯 SWA，其后 21 个 CSA（$m$=4）与 20 个 HCA（$m'$=128）交替；主 KV 584 B，indexer 键 68 B，只有 CSA 层有 indexer。
   答：$(21/4+20/128)\times584+(21/4)\times68=3066+91.25+357=3514.25$ B。追问：HCA 只占 2.6%，为什么 V4.1 把它删了几乎不损失压缩率？（大头在 CSA 的 21 份存储；V4.1 靠跨层共享把份数降到 4。）

3. **（计算 / 反事实）不共享会怎样？** 若 V4.1 的 38 个 CSA2 层每层都存自己的主 KV 与 indexer 键：答 $(18/2+20)\times356=10{,}324$ B，是 890 的 11.6 倍，比 V4-Flash 还大 2.9 倍。结论：V4.1 的 $m$ 更小（2 与 1，比 V4 的 4 与 128 “更不压缩”），全靠层维共享扳回来。

4. **（计算）1M token 的一次请求要多少 HBM？** 答（$2^{20}$ token）：V4.1 0.87 GiB；V4-Flash 3.43 GiB；V3.2 46.9 GiB；DeepSeek 67B 380 GiB；GPT-3 4,608 GiB。再加上 V4.1 的 SWA 状态 2.58 MiB（可忽略）。追问：多长的上下文时 SWA 状态与全局 KV 一样大？$2{,}703{,}360/890\approx3{,}037$ token。

5. **（计算）CED 省了多少 prefill？** 40 层，$n_{win}$=128，提示 $T$ 个 token。答：相对计算量 $\frac{20T+20\cdot128}{40T}=0.5+\frac{64}{T}$；$T$=4,096 时 0.516，$T$=131,072 时 0.5005。若解码器 SWA 要精确重建（让解码器处理最后 $\tfrac L2 n_{win}=2560$ 个 token），$T$=4,096 时变成 $0.5+\frac{20\cdot2560}{40\cdot4096}=0.8125$——这就是 Decoder SWA Bounded Replay 存在的理由。

6. **（推导）为什么持久化 KV 是 1/8？** 答：$\frac{\text{V4.1 持久化}}{\text{V4 持久化}}=\frac{G_{4.1}}{G_4+S_4}$，线上 $S_4\approx G_4$（“nearly half”），$G_{4.1}=G_4/3.95$ ⇒ $\approx\frac{1}{2\times3.95}=0.127$。强调“1/2”是负载统计，不是架构常数；单轮长文档（SWA 占比小）时比值更接近 1/4。

7. **（手算）FP4 量化一个 16 维块。** x2 = [0.089, −0.634, 1.29, 0.298, −2.2, 0.052, 0.492, −0.902, 0.246, 1.566, −0.358, 0.679, −0.015, 0.783, −1.223, 0.433]。(a) E4M3 缩放：amax/6 = 0.3667 → 最近的 E4M3 值 0.375；编码 = [0, −1.5, 3, 1, −6, 0, 1.5, −2, 0.5, 4, −1, 2, −0, 2, −3, 1]；反量化 = 编码 × 0.375；相对 RMSE 8.91%。(b) 若用 MXFP4 的 E8M0（向上取 2 的幂）缩放 0.5：编码 [0, −1.5, 3, 0.5, −4, 0, 1, −2, 0.5, 3, −0.5, 1.5, −0, 1.5, −2, 1]，相对 RMSE 12.09%。问：为什么主 KV 可以选 (a) 而 indexer 只能用 (b)？（主 KV 只存储、进注意力前解回 BF16，格式随便选；indexer 要直接做 FP4 矩阵乘，得用硬件原生支持的 OCP MXFP4。）

8. **（判断）去掉全局缩放安全吗？** 单缩放能表示的最大幅值 $448\times6=2688$；RMSNorm 后 512 维向量 $\lVert\mathbf c\rVert_2\le\sqrt{512}\cdot\max|\gamma|\approx22.6$，单个分量不会超过范数；实测最大约 10。答：上界离 2688 还差两个数量级，所以二级全局缩放多余。追问：下界呢？（E4M3 最小次正规 $2^{-9}$，代码把缩放钳在它之上，全零块也不会出现 0 缩放。）

9. **（计算）解码 FLOPs 为什么几乎平？** 用 §4-F 的加权口径。V4.1：核心注意力每层固定 512 + 128 个条目；随 $N$ 增长的只有 4 个 KV 源层的 indexer 全扫（3 个 $m$=2 + 1 个 $m$=1，共 $2.5N$ 条）与 4 个 Reindex 层的候选池（常数 16,384）。增长 $=0.25\times2\times32\times128\times2.5N=5120N$，1M 时 5.4 GFLOPs，对 20.1 的底数是 +27%。V4-Flash：21 个 CSA 层各扫 $N/4$（64 头）+ 20 个 HCA 层稠密注意 $N/128$ 个条目 → $41{,}984N$，1M 时 +44 GFLOPs。交叉点约 100K：短上下文时 V4.1 反而更贵（激活 16B vs 13B）。

10. **（计算）8B 与 16B。** 用 §3-C 的形状：每层激活 = 6 个路由专家 212.3M + 共享 35.4M + 注意力 ≈128.2M（+ norm/mHC/gate）。答：decode ≈ 40 层 + 头 ≈ 16.5B；prefill ≈ 20 层 + 嵌入 ≈ 8.2B。追问：Engram 的 196B 算不算激活？（每 token 只读 2 × 24 行 × 256 = 12,288 个表参数，外加 0.315B 投影；论文的 8B/16B 显然不含表。）

11. **（计算）分层索引器省了多少打分？** 1M 上下文，每个解码 token：有分层 $1.5N+N+4\times16384=2{,}686{,}976$ 条；无分层（Reindex 也全扫）$1.5N+N+4N=6{,}815{,}744$ 条（2.54×）。追问：为什么编码器不用分层？（编码器 Full 层之间全是 Reuse，没有 Reindex 层可省；论文只说 “used only in the decoder”。）

12. **（计算）一张图多少 token？** patch 14，3×3 pixel-unshuffle，每行末一个换行 token，首尾各一个标记，上限 1,024 token。答：1344² → 96 patch → 32 格 → $2+32\times33=1058>1024$，代码缩到 31 格（1,302 px），994 token；544² → 39 patch → 13 格 → 184 token。与论文 “up to approximately 1344×1344” 对照：精确上限因换行 token 略低于 1344。

13. **（概念）CED 是 T5 那样的编码器–解码器吗？** 答：不是。两半都是因果的、处理同一条 token 流；“解码器”并不交叉注意一个独立的源序列，而是读由编码器输出投影出来的全局 KV，同时每层保留自己的 SWA。它是 YOCO 式的“把 decoder-only 模型切成两半”，名字里的 encoder/decoder 指的是 prefill 时跑不跑。

---

## 7. 勘误与常见误解

- **E1 “V4.1-Flash 比 V4-Flash 小”——错。** 主干 552B vs 284B，外加 196B Engram 表、14.2B DSpark、0.49B 视觉；checkpoint 总逻辑参数 763.2B。论文 §1 自己也说 “Despite being considerably larger than DeepSeek-V4-Flash”。“Flash”指激活小（decode 16B）与服务便宜，不指总参数。
- **E2 “prefill 只激活 8B，所以模型读提示时只用了一半能力”——不准确。** 编码器 20 层完整处理每个提示 token；解码器通过从 $H_{L/2}$ 投影出的全局 KV 看到全部上下文；最后 128 个 token 还会跑完解码器（Decoder SWA Bounded Replay）。省掉的是“提示 token 在解码器里的计算”，这部分结果本来就只用于它们自己的 SWA KV 与（不需要的）输出。
- **E3 “890 B/token 是全部 KV”——错。** 890 是 global KV（主 KV + indexer 键）。SWA KV 另算：每请求常数 2.58 MiB（40 层 × 128 × 528 B），短上下文时比全局 KV 还大。
- **E4 “FP4 KV 意味着注意力用 FP4 算”——错。** 主 KV 只以 FP4 存储，进注意力前反量化（FlashMLA 对 FP8 缓存的说法是解回 BF16 并以 BF16 计算）；只有 indexer 的打分是真 FP4 矩阵乘，所以它必须用硬件广泛支持的 MXFP4。
- **E5 “V4.1 的 FP4 就是 MXFP4”——只对一半。** indexer：MXFP4（E2M1 + 每 32 个一个 E8M0）；主 KV：E2M1 + 每 16 个一个 E4M3（NVFP4 去掉二级全局缩放）。SWA KV 仍是 FP8。
- **E6 “SWA Bounded Replay 是精确的”——错。** 论文明说近似：“not mathematically identical across positions”；同一段后缀，从不同命中位置续算会得到不同的 KV。质量影响只有定性描述（“barely compromises”“negligible”），没有公布数字。
- **E7 “1/8 = 架构把持久化 KV 压到 1/8”——半对。** 1/8 = 1/2（SSD 上不再存 SWA，前提是 V4 线上 SWA 恰好约占一半）× 1/4（全局 KV 本身的压缩）。前一半是部署策略 + 负载统计，不是架构常数。
- **E8 “V4.1 解码在所有长度上都比 V4-Flash 便宜”——错。** 图 2 与笔者复算：约 100K 以下 V4.1 的单 token 解码 FLOPs 更高（4K 时 20.1 vs 16.4 GFLOPs）。V4.1 的优势在长上下文与 prefill（8B）和 KV 存储。
- **E9 “分层稀疏索引器让索引变成 O(1)”——错。** 只有解码器里第 24/28/32/36 层的 Reindex 变成常数；层 2/8/14（编码器 Full，$N/2$ 条）与层 20（解码器 Full，$N$ 条）每个解码 token 仍全扫。
- **E10 “CSA2 = CLA（跨层注意力）”——不完整。** CSA2 把“共享 KV 存储”与“共享 top-k 选择”解耦：Reindex 层共享 KV 但重新选择，Reuse 层两样都共享；每层始终保留自己的 Q 与 SWA KV。
- **E11 “V4.1 有 MTP”——措辞要小心。** 预训练去掉了 MTP；DSpark 在预训练之后单独训练（主干冻结），权重放在 checkpoint 的 `mtp.*` 命名空间，config 字段仍叫 `num_nextn_predict_layers: 3`。
- **E12 数 config 的坑。** `compress_ratios` 有 43 项（含 3 个 DSpark 块），V4-Flash 的有 44 项（含 1 个 MTP 层）；`num_key_value_heads: 1` 表示共享 KV 的 MQA 形态，不是“1 个 GQA 组”；`quantization_config.weight_block_size` V4.1 是 [32,32]，V4 是 [128,128]——不要沿用 V3/V4 的 128×128 记忆。
- **E13 式 (1) 的“layer-dependent $W_l^{KV}$”在 CSA2 配置下只有一份实物。** 解码器里只有第 20 层是 Full，其余 19 层复用它；所以“每个解码层各自从 $H_{L/2}$ 投影 KV”的读法与发布的模型不符。
- **E14 不要混用两篇论文里的 V4 分数。** V4 论文表 7（V4-Flash Max / V4-Pro Max）：GPQA Diamond 88.1 / 90.1，HLE 34.8 / 37.7，Codeforces 3052 / 3206；V4.1 论文表 3 给的是 89.9 / 92.4，37.8† / 42.7†（† = HLE 纯文本子集），3289 / 3348。基座表也有差异：AGIEval 在 V4 论文是 0-shot 82.6 / 83.1，在 V4.1 论文是 3–5-shot 83.9 / 84.4；MultiLoKo 指标由 EM（42.2 / 51.1）换成 LLM-Judge（42.6 / 50.9）。论文没说明原因。后训练表的差异可以用 API Change Log 解释（制品，2026-09-24 抓取）：2026-07-31 条目说 “DeepSeek-V4-Flash-0731 keeps the same model architecture and size as DeepSeek-V4-Flash-Preview, and was only re-post-trained”，2026-08-13 条目发布 V4-Pro GA 版。V4.1 表 3 的 V4-Flash 列（TB2.1 82.7、NL2Repo 54.2、CyberGym 76.7、DeepSWE 54.4、ALE 25.2）与 0731 条目逐项相同；V4-Pro 列（HLE 42.7、HLE w/ tools 60.0、TB2.1 87.9、NL2Repo 61.5、CyberGym 83.3、DeepSWE 62.7、ALE 25.7）与 08-13 GA 条目逐项相同。所以 V4.1 表 3 比较的最可能是这两个更新后的检查点，而不是 V4 论文里的 preview 版（笔者推断，论文未写）。例外是 AutomationBench：Change Log 的 “Automation Bench (Public)” 为 25.1 / 31.8，表 3 为 37.7 / 43.2，可能是不同子集。基座表（AGIEval、MultiLoKo）的差异仍无解释，因为重新后训练不改变基座。课文引用 V4 分数时必须注明出自哪篇、哪个检查点。
- **E15 “held-out 上提升 5%–10%”只在平均意义上对得上图 6。** 图 6 的 BPB 降幅相对 V4-Flash-Base 为 8.6% / 7.6% / 12.7%，相对 V4-Pro-Base 为 4.4% / 3.4% / 8.0%，单个集合的值不全在 5%–10% 内。但三集平均分别是 9.6%（对 V4-Flash-Base）与 5.3%（对 V4-Pro-Base），正好落在区间两端，所以“5%–10%”很可能是“对 Pro 约 5%、对 Flash 约 10%”的平均说法（笔者推断，论文没写计算方式，§8）。课文要写单集数字时，给出范围 3.4%–12.7% 并注明基线。
- **E16 “1/4 activated parameters”对不上 16B 或 8B。** 16/49 = 0.33，8/49 = 0.16；只有两者平均 12B 时是 0.245。课文写“decode 激活约为 V4-Pro 的 1/3”更稳妥。
- **E17 论文内部小矛盾。** (a) §1 说 Mega-mHC 相对 “original four-kernel implementation” 减半访存，§2.4.1 说 V4 是 “three kernels”（加上 pre-norm 才四步）；(b) §2.5 先说 head-wise Muon 只拆 Query 权重，后面 Basic Configurations 说 “for Query and Key weights”；(c) Engram 在 §1/§2.1 引 Cheng et al. 2026b（arXiv 2601.07372），在 §2.4.2 引 2026c（ACL 2026 论文集），参考文献把同一篇列了两次，作者名单也与 arXiv v2（第二作者 Rui Tian）不同。（初稿这里还有一条 (d)，说参考文献把 TileKernels 标成 “Cited by: §2.5”。核查时发现 LaTeXML 版 TileKernels 条目写的是 “Cited by: §3.2.”，与正文一致，所以该条已删除。）
- **E18 “CED 的想法是 DeepSeek 原创”——错。** 论文自己写 “inspired by YOCO”。差别在于：YOCO 上半部分没有自注意力、所有上层共享一对 $W_K,W_V$（YOCO 式 (2)(3)），因此 prefill 能精确提前退出；CED 解码器每层保留 SWA，于是需要（近似的）回放。
- **E19 推理强度的提示文本，论文与代码不一致。** 论文 §5.1.4：“(range 1--100; higher values request more thorough reasoning)”；`encoding/README.md` 渲染为 “(range 1-100, the higher the value, the more thorough the reasoning)”，且只在 `thinking_mode="thinking"`、对话开头渲染。以代码为准。
- **E20 API 制品互相矛盾。** DeepSeek 新闻页：“Starting at 04:00 UTC on Sept 14, 2026, all deepseek-v4-pro requests will route to V4.1-Flash”；API Change Log 同日条目：“we have decided to continue providing API services for DeepSeek V4 Pro after September 14, 2026”（价格页也仍列 `deepseek-v4-pro`）。两句都已在 2026-09-24 从原始 HTML 逐字核对。以 Change Log / 价格页为准，并注明抓取日期。
- **E21 模型卡与论文的一个分数不同。** NL2Repo-Bench：论文表 3 为 65.4，HF 模型卡为 64.0。
- **E22 PowerAttention 的引用支撑度（笔者判断）。** V4.1 §2.2 说 Chen et al. 2025 “has shown that the actual effective receptive field of SWA is much smaller than the theoretical $n_{win}\times L/2$”。PowerAttention 的图 1 标题却写 “The actual token retrieval capability closely matches the theoretical receptive field growth for both patterns”，只在 §4.6 的探针实验中说 “Neither the sliding window attention nor the untrained PowerAttention effectively retrieves the correct passkey”。它支持“SWA 的感受野只线性增长、长距离传递很弱”，但不直接等于 V4.1 的表述。课文引用时写“据 V4.1 作者对 PowerAttention 的解读”，或直接引 V4.1 自己的实验结论。
- **E23 “HCA 被 CSA2 的某种模式替代了”——错。** CSA2 没有“重压缩”模式；V4.1 的压缩比反而变小（2 与 1）。长上下文的稠密注意力成本（V4 的 HCA）被“每层固定 512 + 128 个条目”的稀疏注意力取代。
- **E24 V4 的 “SWA 约是压缩条目的 8 倍”与 V4.1 的 “SWA 近一半”不矛盾。** 前者是“每个 token 都存 SWA”（Full SWA Caching）时的条目比（43 / 5.41 = 7.95）；后者是 V4 实际部署只在“提示结尾、输出结尾”两个点存 SWA 时的容量占比。

---

## 8. 未能核实 / UNVERIFIED

1. **Bounded Replay 与 FP4 KV 的质量代价**：论文只有定性词（“negligible”“barely compromises”“only marginal performance degradation”），没有任何表格或消融数字；“quantizing before RoPE yields only a marginal accuracy improvement”也无数字。
2. **“SWA KV accounts for nearly half”** 是 V4 线上负载的统计，没有给出分布、时间窗或会话长度；1/8 的精确程度随负载而变。
3. **CED “maintaining performance comparable to the baseline”** 与 “Single-Pass … negligible performance degradation”：没有给出基线是什么、在什么规模上做的消融。
4. **8B / 16B 的计数口径**（是否含嵌入、输出头、Engram 投影、DSpark）论文未说明；§4-C 的 8.24B / 16.48B 是笔者按形状复算。
5. **“5%–10% improvements on held-out evaluations”** 的出处集合与计算方式未说明。它与图 6 三集平均降幅（对 Pro 5.3%、对 Flash 9.6%）吻合，但这只是笔者的对应，论文没确认（§7 E15）。
6. **推理强度训练的超参数** $k_0$、$\lambda$、$C_{\max}$、$L_{norm}$、训练用的强度集合 $\mathcal B$ 与 $M_b$ 均未公布。
7. **图 3、4、5、7–12 的矢量图未能渲染目视**（本机无 `pdftoppm`，浏览器面板打开 arXiv PDF 会触发下载，未重试），布局描述来自 PDF 文字层与图注；重绘前需有人在 PDF 阅读器里看一眼。
8. **API 新闻页、Change Log、价格页**：初稿经 WebFetch 摘要模型转述。核查时（2026-09-24）已用原始 HTML 逐字核对价格、384K、V4-Pro 两种说法和退役措辞，见 C60、E20。仍未取得 V4-Flash 旧价格（用于算“降价幅度”）；页面内容会变，发稿前仍需按日期再看一次。
9. **YOCO、CLA 的会议出处**（NeurIPS 2024，页码 7339–7361 与 86927–86957）只来自 V4.1 的参考文献条目，未打开会议页核对；Engram 的 ACL 2026 出处同理；HISA 的 COLM 2026 只来自 arXiv comment。
10. **GPT-3 的推理精度**：论文没写，表中按 2 字节；其“局部带状稀疏层”是否可以只缓存窗口，未核实。
11. **Llama-3-70B 的 KV**：层数/头数来自 Llama 3 论文表 3 与社区镜像的 config（NousResearch、unsloth），官方 `meta-llama` 仓库需登录，未直接读取。
12. **图 1(b) 的日期标签**（V1 “2023.11”、V3.2 “2025.12”、V4-Flash “2026.04”）照录图片，未与各自发布公告逐一核对。
13. **FlashMLA 里 V4.1 FP4 稀疏解码的性能数字**：README 只给了融合核（1430 / 670 TFLOPS on B200）和 “only available on SM100”，没有 FP4 与 FP8 缓存的解码速度对比。
14. **DSpark 的 60%–85% 加速**是在 V4 服务系统上相对 MTP-1 的数字（DSpark 摘要），V4.1 论文没有给 V4.1 上的接受长度或加速比。
15. **Engram 表“sizes chosen to be distinct primes”**：只核对了两模块的总行数（384,006,168 与 384,016,682），24 张子表各自的素数大小没有从代码 `engram.py` 里逐个算出。
16. **V4 arXiv 编号的异常**：2606.19348 的编号前缀是 2606（2026 年 6 月），但 abs 页显示 “Submitted on 26 Apr 2026”。照录，未深究。
17. **Figure 2 的加权规则细节**（FP4×FP8 的专家 GEMM 按 0.25 还是 0.5 计）论文未说明；§4-F 的常数项粗估假设按 0.5，增长项与图吻合到 ±3%，但常数项只吻合到 1%（V4.1）与 5%（V4-Flash）。
18. **V4.1 表 3 用的 V4 检查点**：V4-Flash、V4-Pro 两列与 Change Log 的 0731 / 08-13 GA 公告分数逐项相同（E14），但论文没说用了哪个检查点；AutomationBench 两处不一致的原因未查明。

---

## 9. 参考文献卡片草稿

- id: deepseek2026v41flash
  title: "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"
  a: "DeepSeek-AI (2026), preprint, arXiv:2609.19969v1"
  url: https://arxiv.org/abs/2609.19969
  finding: "用 CED（解码器全局 KV 由编码器输出投影）、CSA2 跨层共享主 KV 与 indexer 键、FP4 主 KV 缓存，把全局 KV 压到 890 字节/token，并以 SWA Bounded Replay 把持久化 KV 降到 V4-Flash 的约 1/8。"
  matters: "本课所有架构、配置与效率数字的主出处；课文每个 V4.1 数字都应回指它的节号。"

- id: deepseek2026v4
  title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
  a: "DeepSeek-AI (2026), preprint (preview version), arXiv:2606.19348v1"
  url: https://arxiv.org/abs/2606.19348
  finding: "提出 CSA/HCA 混合注意力、FP8+BF16 的 KV 存储格式与三种 SWA 落盘策略，V4-Flash 为 43 层、284B/13B。"
  matters: "V4.1 的每个“1/4”“1/8”都以它为基线；CSA2 的简化必须对照它的式 (9)–(17) 才讲得清。"

- id: sun2024yoco
  title: "You Only Cache Once: Decoder-Decoder Architectures for Language Models"
  a: "Yutao Sun et al. (2024), NeurIPS 2024 (per V4.1 bibliography), arXiv:2405.05254v2"
  url: https://arxiv.org/abs/2405.05254
  finding: "上半部分交叉解码器共享由自解码器输出一次性生成的全局 KV，从而只缓存一次并允许 prefill 提前退出。"
  matters: "CED 的直接来源；讲清 YOCO 与 CED 的差别（解码器是否保留自己的 SWA）才能解释为什么 V4.1 需要回放。"

- id: brandon2024cla
  title: "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention"
  a: "William Brandon et al. (2024), NeurIPS 2024 (per V4.1 bibliography), arXiv:2405.12981v1"
  url: https://arxiv.org/abs/2405.12981
  finding: "让相邻层共享同一份 KV 以减少缓存。"
  matters: "V4.1 把“层维度”列为第三个压缩维度时引用的先例，用来说明 CSA2 的新意在于把 KV 共享与 top-k 共享解耦。"

- id: bai2026indexcache
  title: "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse"
  a: "Yushi Bai et al. (2026), preprint, arXiv:2603.12201v1"
  url: https://arxiv.org/abs/2603.12201
  finding: "在 DSA 模型上让多数层复用最近 Full 层的 top-k 索引，可去掉 75% 的 indexer 计算。"
  matters: "它是 CSA2 Reuse 模式的近亲；V4.1 批评它“saves no main KV storage”，正好点出 CSA2 多做的那一步。"

- id: sun2026yoio
  title: "You Only Index Once: Cross-Layer Sparse Attention with Shared Routing"
  a: "Yutao Sun et al. (2026), preprint, arXiv:2606.06467v1"
  url: https://arxiv.org/abs/2606.06467
  finding: "全网只算一次稀疏路由并在所有层共享。"
  matters: "V4.1 以它为反例说明“network-wide routing sharing limits performance”，是 Reindex 模式存在的理由。"

- id: gao2026hysparse
  title: "HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing"
  a: "Yizhao Gao et al. (2026), preprint, arXiv:2602.03560v1"
  url: https://arxiv.org/abs/2602.03560
  finding: "让稀疏层复用稠密层的 KV cache。"
  matters: "V4.1 批评其仍保留全注意力层，用来对比 V4.1“纯 CSA2、无稠密层”的选择。"

- id: xu2026hisa
  title: "HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention"
  a: "Yufei Xu et al. (2026), COLM 2026 (per arXiv comment), arXiv:2603.28458v4"
  url: https://arxiv.org/abs/2603.28458
  finding: "先用池化块表示粗筛、再在保留块内做 token 级索引，免训练替换 DSA 的 indexer。"
  matters: "分层稀疏索引器的先例；对比点是 V4.1 直接用上一层 token 分数的块内最大值，且训练感知。"

- id: chen2025powerattention
  title: "PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention"
  a: "Lida Chen et al. (2025), preprint, arXiv:2503.03588v1"
  url: https://arxiv.org/abs/2503.03588
  finding: "从感受野角度分析静态稀疏模式，指出滑动窗口的感受野随层数只线性扩张。"
  matters: "V4.1 用它为 Bounded Replay 辩护；本笔记 E22 指出其支撑度有限，课文需谨慎措辞。"

- id: rouhani2023mx
  title: "Microscaling Data Formats for Deep Learning"
  a: "Bita Darvish Rouhani et al. (2023), preprint, arXiv:2310.10537v3"
  url: https://arxiv.org/abs/2310.10537
  finding: "定义 MX 块格式：块大小 32、共享 E8M0 缩放，MXFP4 的元素为 E2M1。"
  matters: "V4.1 indexer 采用的 OCP 标准格式的出处，用来对比主 KV 所用的“E2M1 + 每 16 个 E4M3”格式。"

- id: cheng2026dspark
  title: "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
  a: "Xin Cheng et al. (2026), preprint, arXiv:2607.05147v1"
  url: https://arxiv.org/abs/2607.05147
  finding: "半自回归草稿 + 置信度调度的验证长度，在 V4 线上相对 MTP-1 提速 60%–85%（同吞吐）。"
  matters: "解释 V4.1 为什么在预训练里去掉 MTP、改为事后单独训练的投机解码模块。"

- id: cheng2026engram
  title: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"
  a: "Xin Cheng et al. (2026), preprint, arXiv:2601.07372v2 (V4.1 also cites an ACL 2026 version, unverified)"
  url: https://arxiv.org/abs/2601.07372
  finding: "用 N-gram 哈希查表作为与 MoE 互补的条件记忆。"
  matters: "V4.1 的 196B Engram 参数来自它；本课只需说明“这部分不计入 552B、可放主机内存预取”，细节回指第 17 课。"

- id: xie2025mhc
  title: "mHC: Manifold-Constrained Hyper-Connections"
  a: "Zhenda Xie et al. (2025), preprint, arXiv:2512.24880v2"
  url: https://arxiv.org/abs/2512.24880
  finding: "把残差混合矩阵约束为双随机矩阵以稳定多流残差。"
  matters: "Single-Pass mHC 是在它之上的部署改造；本课只讲访存从 $(4n+4)d$ 降到 $(2n+2)d$，数学回指第 16 课。"

- id: brown2020gpt3
  title: "Language Models are Few-Shot Learners"
  a: "Tom B. Brown et al. (2020), NeurIPS 2020, arXiv:2005.14165v4"
  url: https://arxiv.org/abs/2005.14165
  finding: "表 2.1 给出 GPT-3 175B：96 层、d_model 12288、96 头 × 128。"
  matters: "阶梯图的起点（4,718,592 字节/token），让学生看到五年里约 5,300 倍的差距。"

- id: grattafiori2024llama3
  title: "The Llama 3 Herd of Models"
  a: "Aaron Grattafiori et al. (2024), preprint, arXiv:2407.21783v3"
  url: https://arxiv.org/abs/2407.21783
  finding: "表 3 给出 70B：80 层、8192 维、64 头、8 个 KV 头。"
  matters: "阶梯图里的 GQA 代表（327,680 字节/token），说明“只减头数”能走多远。"

- id: deepseek2024v2
  title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  a: "DeepSeek-AI (2024), preprint, arXiv:2405.04434v5"
  url: https://arxiv.org/abs/2405.04434
  finding: "MLA 每层缓存 512 + 64 维，部署时 KV 平均量化到 6 bit。"
  matters: "阶梯图中 MLA 一级（69,120 / 25,920 字节）的出处，并提醒学生“精度”早在 V2 就参与了 KV 账单。"

- id: hf2026v41flash
  title: "deepseek-ai/DeepSeek-V4.1-Flash (model card, config.json, inference code, encoding, safetensors)"
  a: "DeepSeek-AI (2026), Hugging Face repository, commit dba1be0a, created 2026-09-10"
  url: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
  kind: artefact
  finding: "config 给出逐层压缩比与 KV/索引源层 [2,8,14,20] / [2,8,14,20,24,28,32,36]，代码给出 FP4 量化（主 KV 每 16 个 E4M3、indexer 每 32 个 E8M0）与候选块选择的实现。"
  matters: "890 字节、8B/16B、763B 总参数等复算全部依赖它；论文没写的实现细节（钉入最新块、去掉 Q-norm 与 Hadamard）只能引这里。"

- id: hf2026v4flash
  title: "deepseek-ai/DeepSeek-V4-Flash and DeepSeek-V4-Pro (config.json, inference/model.py)"
  a: "DeepSeek-AI (2026), Hugging Face repositories"
  url: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
  kind: artefact
  finding: "config 的 compress_ratios 显示 V4-Flash 为 21 个 CSA + 20 个 HCA、V4-Pro 为 30 + 31，代码显示压缩器含绝对位置偏置与重叠窗口、indexer 有独立压缩路径与 Hadamard 旋转。"
  matters: "复算 3,514 与 5,031 字节、以及列出 V4.1“精简了什么”的对照依据。"

- id: flashmla2026v41
  title: "FlashMLA README (2026.09.10 release of DeepSeek V4.1 attention kernels)"
  a: "DeepSeek-AI (2026), GitHub repository"
  url: https://github.com/deepseek-ai/FlashMLA
  kind: artefact
  finding: "给出 V4 584 B、V4.1 FP8 528 B、V4.1 FP4 288 B 三种每 token KV 行格式，并说明 V4.1 不再使用 Q-norm、V4.1 稀疏解码仅支持 SM100。"
  matters: "把论文的“E2M1 + E4M3/16”落实成可数的字节布局，是 F7 字节布局图和 890 复算的直接依据。"

- id: deepseek2026v41news
  title: "Introducing DeepSeek-V4.1-Flash (news page); API Change Log 2026-09-10; Models & Pricing"
  a: "DeepSeek (2026), website and API documentation, fetched 2026-09-24"
  url: https://api-docs.deepseek.com/quick_start/pricing
  kind: artefact
  finding: "模型名 deepseek-flash，off-peak 每百万 token 缓存命中 $0.003、未命中 $0.15、输出 $0.60，峰时翻倍；V4-Flash 退役。"
  matters: "把“KV 压缩”连到用户看得到的缓存命中价格上，同时提醒学生制品会变（V4-Pro 路由说法前后矛盾）。"

---

## 核查记录（对抗式事实核查，2026-09-24）

范围：从本笔记挑出 20 余条最高风险的论断（规格数字、转录公式、头条结果、日期、归属），逐条回到一手来源重新核对。
工具：`arxiv.py text/grep`（V41 2609.19969v1、V4 2606.19348v1、YOCO 2405.05254v2、PowerAttention 2503.03588v1、MX 2310.10537v3、DSpark 2607.05147v1、Engram 2601.07372v2、HISA 2603.28458v4 的元数据）；arXiv HTML 里图 1(b)、图 2、图 6 的原始 PNG（目视）；HF `config.json` 原文（V4.1-Flash / V4-Flash / V4-Pro / DeepSeek LLM 67B / V2 / V3 / V3.2）；V4.1 全部 48 个 safetensors 分片头（HTTP Range）与 HF API `safetensors`；V4.1 与 V4-Flash 的 `inference/model.py`、V4.1 `inference/kernel.py`、`encoding/README.md`、模型卡；FlashMLA README 原文；DeepSeek 价格页、Change Log、新闻页 news260910 的原始 HTML。算术全部用 Python 重算。

**确认无误（未改）**
1. 摘要：552B 主干、1M 上下文、decode 16B / prefill 8B、890 B/token、约 1/4、约 1/8、45T。
2. 图 1(b) 四个数（389,120 / 48,068 / 3,514 / 890）、三个倍数（8.1× / 13.7× / 3.9×）和日期标签（目视）；图 1 标题的 “4-fold and 437-fold”。四个字节数都按配置重算一致（389,120 = 2·95·8·128·2；48,068 = 61·(656+132)；3,514.25；890 = 2.5·356）。
3. 式 (1) 逐符号一致（$C_l=H_{L/2}W_l^{KV}$，$Z_l=H_{L/2}W_l^Z$，$l>L/2$）；复杂度 $\mathcal O(NL)\to\mathcal O(NL/2+n_{win}\times L/2)$ 一致。
4. §4.2.1 层配置（40 层、$d$=5120、2 SWA + 18 层 $m$=2 分 3 组 × 6 + 20 层 $m$=1 分 5 组 × 4）、indexer 32×128、top-k 512、64 头 × 512、$d_c$=1280、2,048 块 × 8 = 16,384；config 的 `kv_source_layer_ids` [2,8,14,20]、`index_source_layer_ids`、`candidate_*`、43 项 `compress_ratios`、`rms_norm_eps` 1e-20、`weight_block_size` [32,32] 逐项一致。
5. §2.4.4 FP4：E2M1 + 每 16 通道一个 E4M3（NVFP4 去掉全局缩放）、448×6 = 2688、√512 ≈ 22.6、实测约 10、RoPE 后量化、SWA 保留 FP8、“nearly halves”（288/584 = 0.493）。代码的 E4M3 缩放 `Cast(FP8, amax/6)`、钳位 `6*(2**-9)`、E8M0 `fast_round_scale` 均一致。
6. FlashMLA README 的 584 / 528 / 288 B 三种格式、“Q-norm (only used in V4, not V4.1)”、“only available on SM100”、1430 / 670 TFlops，全部逐字一致。
7. §3.2.1–3.2.2：“nearly half”、72 小时、10% 主机 DRAM、TTL 分钟级、可见区间 $[\max(s,i-W+1),i]$、解码器侧 $\frac L2\times n_{win}$ 精确重建。
8. §4.2.2 训练：45T、batch 100.6M、2000 步预热、2.6e-4 到 28T、余弦到 2.6e-5（40T）、保持到 45T、64K 起步稀疏、34T 扩到 1M；§4.1 的 7:1。
9. 表 3 头条：Codeforces 3471（V4-Pro 3348 / V4-Flash 3289）、GPQA 90.9、MathArena Apex 65.6、TB2.1 90.6、DeepSWE 74.2（Opus-5 74.0）、TB4.0 31.2（Opus-5 51.8）、ProgramBench 20.3、NL2Repo 65.4；模型卡 NL2Repo 64.0、“Opus-5.0”。
10. §5.3.2 强度曲线（67.1→76.3、66.0→74.2、82.4→90.6、约 2.5×）；表 2 的 100 / 75 / 50；式 (9)(10) 逐符号一致；`encoding/README.md` 的提示文字与默认 “high”（75）一致。
11. V4 论文：43 层、$d$=4096、284B / 13B；“27% … 10%”（Pro）与 “10% … 7%”（Flash）；BF16 RoPE + FP8 的 KV 格式；“approximately 8 times”；Hash 路由前 3 层；表 7 的 GPQA 88.1 / 90.1、HLE 34.8 / 37.7、Codeforces 3052 / 3206；表 1 的 AGIEval（0-shot 82.6 / 83.1）与 MultiLoKo（EM 42.2 / 51.1）。
12. 48 个分片头重新求和：逻辑参数总计 763,205,315,794，与 HF API 完全一致（HF 的 “I8” 桶已按每字节 2 个 FP4 值计）。分组：路由专家 543.58B、共享 1.42B、注意力 5.13B、Engram 表 196.61B、DSpark 14.23B、视觉 0.485B。激活量 decode 16.48B（不含嵌入 15.82B）、prefill 8.24B（不含嵌入 7.58B）。专家权重确为打包存储（如 `w1` 为 I8 `[2304, 2560]`，逻辑形状 [2304, 5120]）。
13. PowerAttention（E22）：图 1 标题 “closely matches the theoretical receptive field growth” 与 §4.6 “Neither the sliding window attention nor the untrained PowerAttention effectively retrieves the correct passkey” 逐字一致。
14. YOCO 式 (2)(3) 与摘要 “early exit without changing the final output” 一致；§1 的 “before entering the self-decoder” 笔误确实存在。
15. 图 2：目视复读 4K 处 V4.1 ≈ 20、V4-Flash ≈ 16.5，1M 处约 26 与 60；两线交叉在约 99K（像素估计），与解析交叉点 100,369 token 一致。解析增长项 5,120·N 与 41,984·N 重算无误。
16. mHC：式 (6) 与访存 $(2n+2)d$ / $(3n+2)d$ / $(4n+4)d$，以及 “four-kernel” 与 “three kernels” 的内部不一致（E17a），均确认；Engram 引用 2026b / 2026c 重复（E17c）确认。
17. 练习 7 的两组 FP4 编码与相对 RMSE（8.91% / 12.09%）重算一致；§4-G 高斯模拟重跑得 9.51% / 11.54% / 11.28%，与原值差别在随机种子噪声内。

**修改（错误或说得过头）**
1. **E15 / C53 / §8-5 / §4-H**：原文说 “5%–10%” 与图 6 “对不上”。图 6 三集平均降幅是 9.62%（对 V4-Flash-Base）与 5.26%（对 V4-Pro-Base），恰好落在区间两端，所以该说法与图 6 **在平均意义上一致**，只是单集值不全在区间内。已改为这个表述，并补上平均值。来源：图 6 PNG（arXiv HTML `pretrain_inhouse_ppl.png`）+ Python。
2. **E17(d) 已删除**：原文称参考文献把 TileKernels 标为 “Cited by: §2.5”。V41 v1 的 LaTeXML 参考文献里 TileKernels 条目写的是 “Cited by: §3.2.”，与正文 §3.2 的引用一致。来源：2609.19969v1 HTML 参考文献 “Wang et al. (2026a)” 条目。
3. **E14 / §0 版本差异**：原文说原因不明，并暗示与 V4-Flash HF 仓库 06-22 的更新有关。Change Log 原文显示比较对象很可能是 V4-Flash-0731（“only re-post-trained”）与 V4-Pro GA（08-13）：表 3 两列分数与这两条公告逐项相同。已补充这一证据，标为笔者推断，并注明 AutomationBench 的例外和基座表仍无解释。来源：`api-docs.deepseek.com/updates`（2026-09-24）。
4. **C60**：“Off-peak rates are 50% of peak rates” 原文放在价格页名下，实际出自新闻页；价格页的原话是 “Off-peak rates are half of the peak rates.”。已改正出处，补上 V4-Pro 价格与退役原话。价格、384K 已由原始 HTML 核实，§8-8 相应降级。
5. **§3-E AdamW 行**：原文读起来像 wd 0.1 施加于 norm 权重、偏置和缩放因子全部；§2.5 原文是只有 norm 权重做 wd，偏置和缩放因子不做。已改。
6. **§3-A SWA KV 行**：代码引文 `act_quant(kv, 32, ...)` 不是原文；实际是 `act_quant(kv, fp8_block_size, ...)`，其中 `fp8_block_size = 32`。已改为原文。
7. **C16**：引文在从句中间截断，丢了 “, or are replaced altogether by more efficient layers.”。已补全，因为这一分句是层维度定义的一部分。
8. **C36**：原文把 “$W$ 就是 $n_{win}$” 写成事实；论文在该处未定义 $W$。已标为笔者判断。
9. **§2.5 YOCO 式 (3)**：补上第三行 $X^{l+1}=\operatorname{SwiGLU}(\operatorname{LN}(Y^l))+Y^l$ 和权重形状，转录现在完整。
10. **E1**：补上论文自己的原话 “Despite being considerably larger than DeepSeek-V4-Flash”（§1），不再只靠笔者复算。

**本次没有复核的**（保留原状，可信度按原研究代理的记录）：HF 版 PDF 与 arXiv PDF 的逐词 diff；精确到分钟的提交时间；图 3–5、7–12 的矢量图布局；`image_processor.py` 的图像 token 算例（练习 12）；Engram 24 张子表的素数大小；GPT-3 / Llama-3 表格行（只重算了算术）。


---

## 补充核实（写课时补查，2026-09-24）

> 写第 19 课正文时补查或新推导的内容。一手来源是 arXiv 2609.19969v1 的 PDF（scratch `v41.pdf`）与 LaTeXML 文本（scratch `v41.txt`）。
> 本机没有 `pdftoppm`；笔者在无头 Chrome 里用 jsDelivr 上的 pdf.js 3.11.174 把 PDF 第 7、10、11 页渲染成 PNG，目视了图 3、4、5。§8-7 所说“矢量图未能目视”的问题，对这三张图已经解决。

### S1. 图 3、4、5 的实际布局（目视 PDF 渲染）

- **图 3**（p.7）：左框 “Causal Encoder”，自下而上为 Embedding → [SWA + MoE] ×2 → {CSA2(2, Full) + MoE，[CSA2(2, Reuse) + MoE] ×5} ×3，顶上输出 “Enc. Hidden States”。右框 “Decoder”，自下而上为 CSA2(1, Full) + MoE（左侧箭头标 “CED”，来自 Enc. Hidden States）→ [CSA2(1, Reuse) + MoE] ×3 → {CSA2(1, Reindex) + MoE，[CSA2(1, Reuse) + MoE] ×3} ×4，顶上是 DSpark。右下 “Hierarchical Sparse Indexer” 从第一个解码层引出到 “Candidate Pool”，再回到 Reindex 层。底部是 Engram、Text Embedding、Vision Embedding（其下是 Vision Encoder），编码器底部标 “Single-Pass mHC”。图注：“The 40-layer network is divided into a causal encoder and a decoder, each with 20 layers. All feed-forward layers use standard DeepSeekMoE.”
- **图 4**（p.10）：三栏 Full / Reindex / Reuse。图注原文：“Green blocks indicate quantities computed in the current layer; yellow blocks indicate main KV and indexer K reused from the most recent Full Mode layer; while red blocks indicate Top-K indices reused from the most recent index-producing (Full or Reindex Mode) layer. All three modes compute main Q and SWA KV in the current layer.” 本课图 19.4 按课程颜色改为：本层计算 = 蓝，复用主 KV/索引键 = 黄（缓存色），复用 top-k = 紫。
- **图 5**（p.11）：左 “Full Mode Indexer / Score on Full Positions”，绿格 = 选中位置，蓝框 = 按块最高分选中的块；中 “Shared Candidate Pool”；右两个 “Reindex Mode Indexer / Score on Candidate Pos.”，各自输出 Top-512。图注：“The decoder's first CSA2 layer in Full mode selects its own Top-512 indices and builds a shared candidate pool from the selected blocks for subsequent layers.”

### S2. 补充引文（LaTeXML 文本 grep，逐字）

- §2.3（p.10，CSA2 的两处简化之后）：“Both designs simplify the implementation and increase the training efficiency.”
- §2.3.1（p.11）Reuse 模式全文：“The layer reuses the most recent available main KV and the latest Top-K indices computed against that main KV by a preceding layer in Full or Reindex Mode. It performs attention using this selection without computing indexer Q or evaluating index scores.”
- §4.2.2：“We keep the learning rate at this value from 40T to 45T tokens.”（C47 的省略号处；学习率在 40T–45T 保持 $2.6\times10^{-5}$。）
- §2.1：“with multimodal data incorporated from the start of language-model pre-training”。
- §2.5：“During pre-training, we keep the vision encoder frozen until the learning-rate decay stage, while its final normalization layer and the vision–language projector remain trainable. At the onset of learning-rate decay, we unfreeze the vision encoder and optimize it jointly with the LLM with a smaller learning rate.”
- §2.1.1（按模态分开的均衡）：“each token uses the correction biases associated with its modality for expert selection, while the original routing scores are retained for weighting the selected expert outputs. After each training step, the two sets of biases are updated independently according to their respective expert loads.”
- §4.1（多模态数据）：“Our multimodal pre-training dataset primarily comprises three types of data: image-text pairs, interleaved image-text data, and domain-specific data.”；“we refrained from large-scale data synthesis”；领域数据包括 “visual grounding and pointing”“optical character recognition (OCR)”“the acquisition of long-tail knowledge”，另有 “image-code pairs and computer-use trajectories”。
- §4.2.2（ViT 两阶段）：“Although using higher resolutions in this phase yields notable gains, empirical results show that these benefits contribute little to the final model.”；第二阶段数据 “image captions, alt text, charts, and OCR”；“After this stage, we discard the LLM and retain only the optimi[zed vision encoder]”（原文此处截断在 grep 窗口外）。
- §3.1.1（图像 IO 判据）：图像 “sharded across the CP ranks with load balancing, and each image is loaded exactly once”；“Storage throughput therefore becomes a bottleneck only for small models with low per-token compute, as in ablations, while production-scale models remain compute-bound.”
- §2.4.1（Mega-mHC）：“The kernel implements mHC with $(3n+2)d$ activation reads and writes and Single-Pass mHC with $(2n+2)d$ activation reads and writes.”；预训练 “we keep the existing multi-kernel implementation”。
- §2.4.2（Engram 预取）：“with prefetching for the first module overlapping computation in the first Transformer block.”

### S3. 课文里笔者自己的推算（课文已逐处标注“笔者”）

| 内容 | 表达式 | 值 | 课文位置 |
|---|---|---|---|
| 精确重建解码器 SWA 要读入的编码器输出位置数 | `128 + 19*127` | 2,541（论文写 $\frac L2 n_{win}$ = 2,560） | §2 推导、练习 12 |
| 同一口径下 V4-Flash 43 层 | `128 + 42*127` | 5,462（论文写 43×128 = 5,504） | 练习 12 |
| V4.1 编码器侧精确重建（按感受野推理） | `20*128` | ≈ 2,560（论文一般式 $L\times n_{win}$ 给 5,120） | §6 算例 |
| 候选池必含层 20 自己的 top-512 | 块分 ≥ τ 的块至多 512 个 ≤ 2,048 | 证明见 §4 | §4 推导 |
| 若无分层索引器，解码增长项 | `0.25*2*128*32*6.5` | 13,312·T；1M 时约 34.1 G；与 V4-Flash 交叉点约 129,000 token | 练习 10 |
| 持久化比值一般式 | `(1-σ)*890/3514.25` | σ = 0.5 / 0.3 / 0.6 → 0.127 / 0.177 / 0.101；长文档例 σ = 0.1 → 0.228 | §6、练习 11 |
| 一张图的视觉 token（不计换行与起止标记） | `ceil(H/42)*ceil(W/42)` | 1344² → 1,024；544² → 169 | §7 (19.20) |
| 智能体会话示例（参数为笔者假设） | $T_0$=20,000，$K$=50，$o$=400，$a$=2,000 | 输入 3,940,000，输出 20,000（197:1），命中 97.0%，账单 \$0.041 vs \$0.603 | §1、图 19.1 |
| 会话的主干计算 | `2*8.24e9*118000`，`2*16.48e9*20000` | prefill 1.94e15，decode 0.66e15 | 图 19.1 |
| 练习 9 的 FP4 块 | `l19_fp4.quant_block` | E4M3 缩放 0.5625、相对 RMSE 9.81%；E8M0 缩放 1、14.30% | 练习 9 |
| 128K 与 1M 的请求字节 | `890*131072`，`3514.25*131072` | 111.25 MiB / 439.3 MiB | 练习 7 |
| 编码器 SWA 状态 | `20*128*528` | 1,351,680 B ≈ 1,519 个 token 的全局 KV | §6 |
| 第 50 轮命中时读回的全局 KV（审阅时补） | `3514.25*135600`，`890*135600` | 476,532,300 B（454 MiB）/ 120,684,000 B（115 MiB） | §1 第三张账单 |
| 每层注意力参数的拆分（审阅时补；维度取自 3-A） | $W^{DQ}$ `5120*1280`；$W^{UQ}$ `1280*64*512`；分组输出 `8*1024*4096` + `5120*8192`；SWA KV `5120*512`；压缩器 `wkv`/`wgate` 各 `5120*512`；索引器 `wq_b` `1280*32*128`、`weights_proj` `5120*32`、`wk` `512*128` | 每层基础 126,615,552；40 层 + 4 个压缩器（层 20 只有 `wkv`）+ 8 个索引器查询 + 4 个索引器键 = 5,126,488,064 ≈ 3-C 的 5.13B；KV 投影占基础的 2.1% | §2 例题“复算 16B 与 8B”之后 |
| FP4 打分误差粗估（审阅时补） | 独立误差模型 $\epsilon/\sqrt{512}$；scratch 模拟 400 对 512 维高斯向量，主 KV 格式量化 | 粗估 0.42%；模拟：逐元素相对 RMSE 9.53%，打分误差 RMS 为 $\lVert\mathbf q\rVert\lVert\mathbf e\rVert$ 的 0.45% | §5 “每个数错 9%，注意力分数错多少” |

会话、字节、FP4 与交叉点等关键值在 `deepseek/figures/gen/l19_num.py` 与 `l19_fp4.py` 里复算并用 `assert` 固定，其余是手算（已用 Python 复核）；图 19.1–19.15 的几何全部由 `l19_*.py` 计算。

### S4. 课文借用的其他课程参考文献卡片

阶梯图（图 19.15）还引用了 DeepSeek LLM（arXiv 2401.02954；67B 为 95 层、8 个 KV 头，见第 4、6 课卡片与 HF config）、DeepSeek-V3（arXiv 2412.19437；61 层 MLA，$d_c$ = 512、$d_h^R$ = 64，见第 9、14 课卡片）与 DeepSeek-V3.2（arXiv 2512.02556，见第 14 课卡片）。三者的数字都在本笔记 §4-B 里复算过。

### S5. 数学审阅的更正（2026-09-24）

课文里全部推导逐行重推、全部数字用 Python 重算（scratch `math-lesson-19-v4-1-flash/verify.py`）。以下几处有改动，其余数字全部复核一致。

| 位置 | 原来 | 现在 | 依据 |
|---|---|---|---|
| §4-G “FP8 E4M3 + E8M0/64：2.66%” 与课文 §5 “V4 的 FP8 格式约 2.7%” | 2.7% | 约 2.5% | 2.66% 是把 512 维全按 FP8 量化的结果；V4 的真实格式是 448 维 FP8（每 64 个一个 E8M0，向上取 2 的幂）+ 64 维 BF16，按这个格式模拟 1,000 个单位 RMS 高斯向量得 2.48%（全 512 维 FP8 得 2.65%，与 §4-G 一致） |
| S3 “打分误差 RMS 为 ‖q‖‖e‖ 的 0.45%” | 0.45% | 约 0.42%（6 个种子 0.40%–0.43%） | 400 对 512 维高斯向量，主 KV 格式量化；逐元素相对 RMSE 9.47%–9.52%，与独立误差模型 $\epsilon/\sqrt{512}=0.42\%$ 吻合 |
| 课文 (19.4) 的 $l>\tfrac L2$ | 未说明层号起点 | 注明按从 1 数起理解（= 本课层 20–39），$H_{L/2}$ = 本课层 19 的输出 | 课文其余部分层号从 0 数；只有从 1 数起时，$l>L/2$ 与“$H_{L/2}$ 是编码器输出”才同时成立 |
| 课文 §5 E2M1 | “正值只有 8 个” | “非负值只有 8 个（7 个正值加 0）” | 0 不是正值；$7+7+1=15$ |
| 课文 §6 误差推导第 3、4 行 | $\hat{\mathbf h}^{(l)}_i\ne\mathbf h^{(l)}_i\ (l\ge1)$；$\hat{\mathbf u}\ne\mathbf u$ 配“SWA KV 近似” | $l\ge2$ 对所有 $i$ 成立（$i=h-1$ 在层 0 的窗口完整，层 1 输入仍精确）；第 4 行改为 SWA KV $\hat{\mathbf k}^{(l)}_i=W^K_l\hat{\mathbf h}^{(l)}_i$ | 逐层追踪依赖 |
| 课文练习 10(b) | 交叉点约 129,000 | 保留，并注明去掉分层后 $F_0$ 还要减去候选池打分常数 $4\times16{,}384\times2{,}048\approx0.13$ G，交叉点约 124,000 | 该修正小于图 2 读数误差（±0.6 G） |
| 课文 §7 图像 IO 判据 | $B_{\mathrm{GPU}}$ 叫“GPU 的带宽” | 保留论文原词，补一句量纲：$B_{\mathrm{GPU}}$ 必须是 FLOP/s | $N\times C/B_{\mathrm{GPU}}$ 要是秒 |
| 课文 (19.10)、(19.22) 记号 | $\mathcal S$（层集合，与同页 top-k 集合 $\mathcal S_t$ 撞名）；$\beta$（每数字节，与 §4 块分 $\beta_{t,b}$ 撞名） | $\mathcal L_{\text{kv}}$；$b_{\text{num}}$ | 同一课内同一符号只有一个含义 |

### S6. 事实审阅补查（2026-09-24）

逐条核对课文里的事实、引文与引用挂靠。工具：`arxiv.py grep`（V41 2609.19969、V3.2 2512.02556、DeepSeek LLM 2401.02954、Llama 3 2407.21783）；
数学补课研究笔记 `math/reference-docs/math-refs-doc.md`（NVFP4）；第 14 课研究笔记（V3.2 字节格式）。以下引文逐字取自 LaTeXML 文本。

**新核实的引文（课文据此改写或补注）**

| 课文位置 | 原文（逐字） | 出处 |
|---|---|---|
| 封面“V4 留下三个问题”第二条 | “In agentic workflows, frequent tool calls generate extensive prefill requests, imposing severe computational overhead when KV caches miss.” 论文**没有**给出输入 : 输出的比例；课文的“一两百倍”改为指向 §1 的假设会话（197 : 1） | V41 §2.2 |
| §7 精简表之后“15 / 11 个内核” | “Through reasonable kernel fusion, we encapsulate the intricate operations … in a small number of fused kernels—including the fused-RoPE-attention-RoPE-cast kernel in FlashMLA …, the Mega-Gate, Mega-mHC, and Mega-MoE kernels in DeepGEMM …, the kernels in TileKernels …, and the TopK kernel in DeepSelect …. As a result, the vast majority of Transformer layers … execute with only 15 kernels during prefill and 11 during decode” | V41 §3.2 |
| §7 Single-Pass mHC 例题 | “Input mixing cannot be fused into this pass because $A_l$ is not available until the reduction over all hidden tiles is complete. It therefore requires a second read of $X_l$. … This two-pass implementation would require $(3n+2)d$ activation reads and writes in total, one additional read of $X_l$ compared with the lower bound.” | V41 §2.4.1 |
| §7 数据 | “As our text-only and multimodal data were processed through distinct pipelines, we constructed the final training corpus as the union of both data sources. For overlapping samples, we replace the text-only versions with their multimodal counterparts and use the larger epoch count of the two configurations.”；“For interleaved data, we build this subset predominantly from webpages and PDFs.”；“we refrained from large-scale data synthesis; instead, we prioritized cleaning and utilizing the data in its native form” | V41 §4.1 |
| §7 视觉编码器第一阶段 | “Because the subsequent autoregressive stage specifically handles high-resolution extrapolation, scaling up resolutions during contrastive pretraining significantly increases computational overhead without much overall improvement.”；图文对 “sourced from alt-text data” | V41 §4.2.2 |
| §7 按模态分开的均衡 | “Image and text tokens exhibit distinct representation distributions and may induce different expert-routing preferences in MoEs. Balancing their aggregate load may therefore obscure modality-specific imbalance.” | V41 §2.1.1 |
| §7 图像加载例题 | “a single ultra-long, image-dense sequence can exhaust one host’s I/O, CPU, and memory during loading” | V41 §3.1.1 |
| §5 两种 FP4 | “DeepSeek-V4 already uses quantization-aware training (QAT) … for FP4 indexer queries and keys … We adopt the OCP-standard MXFP4 format … to support as many hardware platforms as possible”（V4 代码的索引器同为块 32，见 3-A） | V41 §2.4.4 |
| §5 “RoPE 之后量化” | 论文只说 “would introduce additional overhead during decoding”，没有说明开销是什么；课文“读出后要再旋转一遍”已标为笔者的理解 | V41 §2.4.4 |
| §6 V4 部署 | “Within this cache, global KV and SWA KV are managed independently, governed by an LRU eviction policy.”（课文未用，供图 19.11 参考） | V41 §3.2.1 |
| §5 NVFP4 的二级缩放 | NVFP4 = 16 元素块 + E4M3 块缩放 + **每张量 FP32** 缩放 | 数学补课笔记 C70（NVFP4 论文 2509.25149 v2 §2） |

**参考文献卡片与引用挂靠的更正**

1. **`deepseek2025v32` 卡片**：原 finding 写“推理时 MLA 缓存用 FP8 存储”。V3.2 论文（2512.02556 v1）全文只有一处 FP8：“Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable.” 656 字节的 FP8 MLA 格式出自 FlashMLA README，132 字节的索引器键出自 V3.2 `model.py`（第 14 课笔记表：`k_cache` FP8 + `k_scale_cache` FP32）。卡片与阶梯表的来源注已据此拆开。
2. **`deepseek2026v41news` 卡片**：课文 §8 用它支撑“表 3 的 V4 两列与更新日志 0731 / 08-13 条目逐项相同”（E14），原 finding 只写了价格。已补上更新日志两条的内容。
3. **§1 规格表**：“数字来自两篇论文与配置文件”后面的引用漏了 V4.1 论文本身，已补。
4. **DeepSeek LLM 67B**：2401.02954 v1 表 2 为 95 层、$d$=8192、64 个查询头、8 个 KV 头（与 HF config 一致，§4-B）；§2.2 原话 “DeepSeek LLM 67B has 95 layers. These layer adjustments … also facilitate model pipeline partitioning”，以及 “the 67B model uses Grouped-Query Attention (GQA)”。
5. **Llama 3 70B**：2407.21783 表 3 复核 “Attention Heads 64”“Key/Value Heads 8”。
6. **CLA 的措辞**：课文原写 CLA “是层维度的起点”，这是历史判断，V4.1 只是把它当作层维度的一个先例引用。已改为“V4.1 论文为层维度举的先例”。

### S7. 跨课一致性核查（F-timeline-specs，2026-09-25）

1. **DSpark 的加速区间只对 V4-Flash 是 60%–85%。** `arxiv.py grep 2607.05147`：摘要 “accelerates per-user generation speeds by 60%–85% at matched throughput levels”；§1 “60%–85% (V4-Flash) and 57%–78% (V4-Pro) at matched aggregate throughput capacities”；§5 分别写 “60% to 85%”（Flash）与 “57% to 78%”（Pro）。本笔记 §8 第 14 条与参考卡把摘要区间写成了“V4 线上”的区间（与 `lesson-00-doc.md` §10.2 第 4 条同一处邻近论断替换）。课文 §7 正文与 `cheng2026dspark` 卡片已改为分开写两个区间。
