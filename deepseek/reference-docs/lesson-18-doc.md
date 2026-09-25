# 第 18 课研究笔记：DeepSeek-V4——百万 token 上下文（CSA/HCA 混合注意力、mHC、Muon）

> 本文件是 `deepseek/src/lesson-18-v4.md` 的研究底稿，只收“核对过的事实、转录的公式、复算过的数字、勘误、未能核实清单”，
> 不是讲义正文。建立于 2026-09-24。所有派生数字都在 Python 里复算过（第 4 节给出表达式；脚本在研究者的 scratch 目录，
> 未放进仓库，表达式足以复现）。凡是笔者自己的推理，一律标注“笔者推测”或“笔者推导”。
> 同日做过一轮对抗式复核（28 条高风险论断回到一手来源重查），核对范围、改动与理由见第 10 节“核查记录”；标“核查者”的内容来自那一轮。
>
> **引文处理方式（重要）**：本次研究会话受版权规则约束——对受版权保护的论文正文，每份产出最多只能逐字引用一处、且不超过
> 15 个英文词。因此本笔记对 V4 论文、V3.2 论文、Muon 两篇文献等**一律采用“中文意译 + 精确定位（节号/式号/表号/图号）+
> grep 关键词”**的写法；写课者需要原句时，用 `python .cursor/skills/academic-research/scripts/arxiv.py grep <id> '<关键词>' -C 2`
> 自取（关键词已在每条论断后给出，均已实测能命中）。公式、超参数、表格中的数字属于事实，照原样转录。
> MIT 许可的制品（Hugging Face 模型卡、`config.json`、`inference/*.py`、`encoding/README.md`）只摘极短的代码行做对照。
> 全文唯一的一处论文逐字引文见 C3。
>
> 与其他课的分工：DSA/lightning indexer 的来历与 V3.2 的训练细节在第 14 课笔记（`lesson-14-doc.md`，其 C39 记录了
> V3.2 的 656 B/token FP8 KV 格式，本笔记与之一致）；mHC 的完整数学在第 16 课笔记；V4.1-Flash 在第 19 课。本笔记只写
> V4 如何**使用**这些部件，以及 V4 自己的新东西。
>
> 共同编辑：本文件是本地 Markdown，可直接改；批注写成 `<!-- 批注(Harry): … -->`，回复写成 `<!-- 回复(Claude): … -->`
> （约定见仓库根目录 `CLAUDE.md`）。任何事实改动先在这里改、附来源，再改讲义。

---

## 0. 覆盖范围与所读版本

### 0.1 论文

| 短键 | 标题 | 作者 | arXiv / 版本 | 状态 | 日期 | 本课用途 |
|---|---|---|---|---|---|---|
| `dsv4` | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | DeepSeek-AI（arXiv 元数据：DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu 及另外 313 人） | **2606.19348，读 v1（全文逐节读完）** | 预印本（技术报告，自称 “preview version”） | arXiv v1 2026-04-26 | 全部架构公式、超参数、基础设施、训练、后训练、评测 |
| `dsv32` | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models | DeepSeek-AI | 2512.02556v1 | 预印本 | 2025-12-02 | 对照基线：DSA、top-k=2048、V3.2 只支持 128K |
| `dsv3` | DeepSeek-V3 Technical Report | DeepSeek-AI | 2412.19437v2 | 预印本 | 2024-12-27 | V4 “未说明的细节沿用 V3” 的出处 |
| `mhc` | mHC: Manifold-Constrained Hyper-Connections | Zhenda Xie 等 20 人 | 2512.24880v2 | 预印本 | v1 2025-12-31 | 与 V4 §2.2 的记号对照；6.7% 开销；$t_{\max}=20$（详见第 16 课笔记） |
| `muon-liu` | Muon is Scalable for LLM Training | Jingyuan Liu, Jianlin Su 等 28 人（机构未在本次核对） | 2502.16982v1（arXiv 上只有 v1） | 预印本 | 2025-02-24 | Muon 标准形式、权重衰减、update RMS 匹配（0.2·√max(A,B)）、Lemma 1 |
| `kimi-k2` | Kimi K2: Open Agentic Intelligence | Kimi Team（Bai et al.） | 2507.20534v2 | 预印本 | 2025-07 | **QK-Clip 的真正出处**（勘误 E7） |
| `gptoss` | gpt-oss-120b & gpt-oss-20b Model Card | OpenAI | 2508.10925v1 | 模型卡（arXiv） | 2025-08-08 | V4 引用的 SwiGLU clamping、attention sink、MXFP4（4.25 bit/参数）、128 token 带状窗口 |
| `minillm` | MiniLLM: On-Policy Distillation of Large Language Models | Yuxian Gu, Li Dong, Furu Wei, Minlie Huang | 2306.08543v6 | ICLR 2024（V4 参考文献写法） | 2023-06-14 | OPD（反向 KL 在线蒸馏）的学术出处之一 |
| `hashlayers` | Hash Layers For Large Sparse Models | Stephen Roller 等 | 2106.04426v3 | NeurIPS 2021 | 2021-06-08 | 前 3 层 MoE 的 Hash 路由 |
| `sinks` | Efficient Streaming Language Models with Attention Sinks | Guangxuan Xiao 等 | 2309.17453v4 | ICLR 2024 | 2023-09-29 | attention sink 的出处 |
| `engram` | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs | Xin Cheng 等 | 2601.07372v2 | 预印本 | 2026-01-12 | 仅作为 V4 结论里“未来方向”的指针（第 17 课主讲） |

### 0.2 制品（无论文，`kind: artefact`）

| 短键 | 内容 | 读取方式 / 时间 |
|---|---|---|
| `hf-v4pro` / `hf-v4flash` | `huggingface.co/deepseek-ai/DeepSeek-V4-Pro`（commit `b5968e91`，创建 2026-04-22，最后修改 2026-06-22）与 `…/DeepSeek-V4-Flash`（commit `60d8d707`）：README 模型卡、`config.json`、`generation_config.json`、`inference/{config.json, model.py, kernel.py, generate.py, convert.py, README.md}`、`encoding/README.md`、`model.safetensors.index.json` | 2026-09-24 用 HF API + raw 原样下载；Pro 与 Flash 的 `model.py`/`kernel.py` 逐字节相同 |
| `hf-v4-base` | `DeepSeek-V4-Pro-Base`、`DeepSeek-V4-Flash-Base` 的 `config.json` | 同上；与指令模型只差 `expert_dtype`（Base 为 `fp8`） |
| `hf-v4pro-0813` / `hf-v4flash-0731` | GA 版模型卡与 `config.json`（创建 2026-08-13 / 2026-07-31） | 同上 |
| `hf-dspark` | `DeepSeek-V4-Pro-DSpark`、`DeepSeek-V4-Flash-DSpark` 模型卡与 config（创建 2026-06-27） | 同上 |
| `hf-v4-vision` | `DeepSeek-V4-Flash-Vision-Exp` 模型卡（HF 创建 2026-08-31） | 同上 |
| `hf-v32-config` | `DeepSeek-V3.2/config.json` | 同上 |
| `flashmla` | `github.com/deepseek-ai/FlashMLA` README（当前 main，含 V4 的 584 B 与 V3.2 的 656 B KV 格式说明） | raw 下载 |
| `deepgemm` | `github.com/deepseek-ai/DeepGEMM` README（Mega MoE、FP4 indexer、V3.2 `fp8_mqa_logits` 的 KV 格式） | raw 下载 |
| `ds-changelog` | `api-docs.deepseek.com/updates`（Change Log） | 2026-09-24 原样下载 |
| `ds-news-0424` / `ds-news-0813` | `api-docs.deepseek.com/news/news260424`、`news260813` | 同上 |
| `ds-pricing` / `ds-thinking` | `api-docs.deepseek.com/quick_start/pricing`、`/guides/thinking_mode` | 同上（价格是“截至 2026-09-24”的快照） |
| `muon-blog` | Keller Jordan, “Muon: An optimizer for hidden layers in neural networks”, kellerjordan.github.io/posts/muon（页面日期 2024-12-08） | 原样下载 HTML |

### 0.3 版本与时间线（全部来自制品的日期字段，笔者整理）

| 日期 | 事件 | 来源 |
|---|---|---|
| 2025-12-01 | V3.2 正式版上线（API 的 `deepseek-chat`/`deepseek-reasoner` 指向 V3.2） | `ds-changelog` |
| 2026-04-22 | HF 上 V4-Pro/Flash（含 Base）仓库创建 | HF API `createdAt` |
| **2026-04-24** | **V4 Preview 发布并开源**；API 新增 `deepseek-v4-pro`/`deepseek-v4-flash`；1M 上下文成为官方服务默认；旧名 `deepseek-chat`/`deepseek-reasoner` 宣布三个月后（2026-07-24）停用，期间分别指向 V4-Flash 的非思考/思考模式 | `ds-changelog`、`ds-news-0424` |
| 2026-04-26 | arXiv 2606.19348v1 | arXiv 元数据 |
| 2026-06-27 | V4-Pro-DSpark / V4-Flash-DSpark（同一 checkpoint + 投机解码模块） | HF API |
| 2026-07-31 | V4-Flash 正式版（`DeepSeek-V4-Flash-0731`）API 公测；change log 说它与 Preview **架构和规模相同，只是重做了后训练** | `ds-changelog`、`hf-v4flash-0731` |
| **2026-08-13** | **V4-Pro GA（`DeepSeek-V4-Pro-0813`）**：APP/Web/API 同步；Agent 能力大幅增强；API 原生支持 OpenAI Responses API；思考强度三档 low/high/max | `ds-changelog`、`ds-news-0813`、`hf-v4pro-0813` |
| 2026-08-16 16:00 UTC | 新价格生效：峰谷定价，谷时为峰时的一半 | `ds-changelog` |
| 2026-08-21 | `DeepSeek-V4-Flash-Vision-Exp` 上 API（HF 仓库 2026-08-31 创建） | `ds-changelog`、HF API |
| 2026-09-10 | V4.1-Flash 发布；V4-Flash 与 Vision-Exp 退役（旧名暂时路由到 V4.1-Flash）；V4-Pro 在 2026-09-14 之后继续提供 | `ds-changelog` |

- **写课时的版本口径**：论文 v1 描述的是 **Preview**（2026-04）。GA 版（Flash-0731、Pro-0813）按官方说法与 Preview 同架构，
  config 的差异只有 DSpark 相关字段（见 §3.6）。所以本课的架构、KV、FLOPs 数字对 Preview 与 GA 都适用；**评测数字只对 Preview 适用**。
- 2026-04-24 新闻页把技术报告链接到 HF 仓库里的 `DeepSeek_V4.pdf`；该文件 2026-09-24 已返回 404（仓库文件列表里也没有），
  引用一律用 arXiv 2606.19348。

---

## 1. 核心论断与原文引文

> 每条格式：论断（中文意译）→ 位置 → `grep` 关键词（对 `dsv4` 即 `arxiv.py grep 2606.19348 '<关键词>'`）。
> 除 C3 外均非逐字引文（原因见文件开头“引文处理方式”）；grep 关键词全部在 v1 全文上实测命中。

### A. 总体

**C1. V4 是一个“预览版”系列，两款 MoE：V4-Pro 总参数 1.6T / 激活 49B，V4-Flash 284B / 13B，都支持 100 万 token 上下文。**
摘要；§1 第二段。grep：`preview version`。

**C2. 三项主要升级：CSA+HCA 混合注意力、mHC、Muon 优化器；保留 DeepSeekMoE 与 MTP。** 摘要；§1；§2 开头
（“其余未说明的细节沿用 V3 的设定”）。grep：`All other unspecified details`。

**C3. 头条效率数字（全文唯一逐字引文，摘要）：** V4-Pro 在 1M 上下文下 “requires only 27% of single-token inference FLOPs and 10% of KV cache”（相对 V3.2）。
§1 补充了两个限定：FLOPs 是**“等效 FP8 FLOPs”**口径；图 1 右侧的曲线是**估算**值。grep：`equivalent FP8 FLOPs`、`estimated single-token`。

**C4. V4-Flash 在 1M 时只需 V3.2 的 10% 单 token FLOPs、7% KV cache。** §1。grep：`7% of the KV cache`。
图 1 上的标注是 Pro “3.7× lower / 9.5× smaller”、Flash “9.8× lower / 13.7× smaller”（1/3.7=27.0%，1/9.5=10.5%，1/9.8=10.2%，1/13.7=7.3%）。

**C5. 路由专家参数用 FP4；FP4×FP8 的峰值算力在现有硬件上与 FP8×FP8 相同，但论文说未来硬件理论上可“高效 1/3”。** §1。grep：`1/3`。
（笔者推测：操作数位宽 (8+8)/(4+8)=1.333，可能是“多 1/3”的来由；论文未给推导，这只是一种与数字相符的解释，不是论文的说法。）

**C6. 训练数据：Flash 32T token、Pro 33T token；预训练后原生支持 1M。** §1、§4.2.2。grep：`33T tokens`。
摘要与模型卡写的是 “more than 32T”。

**C7. V4-Flash-Base 在多数基准上超过 V3.2-Base；V4-Pro-Base 是 DeepSeek 最强基座。** §1、§4.3.2、表 1。grep：`Flash-Base already surpasses`。

**C8. 后训练两段式：先各领域独立培养专家（SFT + GRPO），再用在线策略蒸馏（OPD，反向 KL）合并成一个统一模型。** §1 末段、§5.1。grep：`reverse KL loss`。

### B. MoE（§2.1）

**C9. 亲和度激活函数从 V3 的 Sigmoid 改为 $\operatorname{Sqrt}(\operatorname{Softplus}(\cdot))$。** §2.1。grep：`Softplus`。
代码对照：`config.json` `"scoring_func": "sqrtsoftplus"`；`model.py` `Gate.forward` 计算 `F.softplus(scores).sqrt()`，再加偏置选 top-k，
门控值用**未加偏置**的分数、按所选专家归一化后乘 `route_scale`（Pro 2.5、Flash 1.5；论文未提 `route_scale`）。

**C10. 负载均衡：沿用无辅助损失策略，外加一个很小的序列级均衡损失；偏置更新速度 0.001、均衡损失权重 0.0001。** §2.1、§4.2.2。
grep：`bias update speed`。

**C11. 去掉 V3 的“每 token 最多发往 $M$ 个节点”限制，改为重新设计并行策略。** §2.1。grep：`routing target nodes`。
代码对照：V4 的 config 没有 `n_group`/`topk_group`（V3.2 的 config 有 `n_group: 8`、`topk_group: 4`）。

**C12. 用 Hash 路由的 MoE 层替换 V3 开头几层的稠密 FFN；Hash 路由按输入 token ID 的预定义哈希决定专家。** §2.1；§4.2.1 给出“前 3 个 MoE 层”。
grep：`Hash routing`。代码：`num_hash_layers: 3`；`Gate` 在哈希层用查表 `tid2eid[input_ids]`（形状 vocab×6）选专家，但门控值仍来自打分。

**C13. MTP 策略与 V3 完全相同，深度 1。** §2.1、§4.2.1。grep：`without modification`。代码：`num_nextn_predict_layers: 1`。

### C. mHC（§2.2）

**C14. 标准 HC 把残差流宽度扩大 $n_{hc}$ 倍（$\R^{d}\to\R^{n_{hc}\times d}$），引入输入映射 $A_l$、残差变换 $B_l$、输出映射 $C_l$；层内部仍是 $d$ 维。** §2.2 式 (1)。grep：`three linear mappings`。

**C15. 作者观察到朴素 HC 叠很多层时训练经常数值不稳定。** §2.2。grep：`numerical instability`。

**C16. mHC 把 $B_l$ 约束到双随机矩阵流形（Birkhoff 多面体），谱范数 $\le 1$、乘法封闭；$A_l$、$C_l$ 用 Sigmoid 约束为非负有界。** §2.2 式 (2)、(6)、(7)。grep：`Birkhoff polytope`。

**C17. 映射参数动态生成：输入展平后 RMSNorm，线性投影得动态部分，加静态偏置 $S$，门控因子 $\alpha$ 初始化为小值。** §2.2 式 (3)–(5)。grep：`initialized to small values`。

**C18. Sinkhorn-Knopp：先取指数，再交替列/行归一化，$t_{\max}=20$。** §2.2 式 (8)；§4.2.1 两款模型都 $n_{hc}=4$、$t_{\max}=20$。grep：`practical value`。
代码：`hc_mult: 4`、`hc_sinkhorn_iters: 20`、`hc_eps: 1e-06`。

**C19. mHC 的工程开销：融合核 + 选择性重计算 + 调整 DualPipe 1F1B 重叠，把墙钟开销压到重叠后 1F1B 流水段的 6.7%。** §3.4.2。grep：`6.7%`。
（mHC 论文 §1、§4.3 也报 6.7%，但表述为“$n=4$ 时额外训练时间”——同一个数，两处口径略不同。）

### D. 混合注意力 CSA / HCA（§2.3）

**C20. 动机：上下文极长时注意力成为主要计算瓶颈；V4 设计 CSA 与 HCA 两种高效注意力并交错使用。** §2.3 开头。grep：`interleaved hybrid configuration`。

**C21. CSA = 先把每 $m$ 个 token 的 KV 压成一个条目，再做 DSA 式稀疏选择（每个查询只看 $k$ 个压缩条目）；HCA = 把每 $m'$（$\gg m$）个 token 压成一个条目、不做稀疏。** §1、§2.3。grep：`cache of every`。

**C22. CSA 的压缩：两路 KV 候选 $C^a,C^b$ 与两路压缩权重 $Z^a,Z^b$；条目 $i$ 由当前块的 $C^a$ 与前一块的 $C^b$ 共 $2m$ 个候选、按行 softmax 加权求和得到；相邻条目的输入区间有重叠，但序列长度仍压到 $1/m$。** §2.3.1 式 (9)–(12)。
grep：`overlapped`。$i=0$ 时前一块用 $-\infty$（权重）/ 0（值）填充。

**C23. lightning indexer：对压缩后的 indexer 键 $K^{\mathrm{IComp}}$（同样的压缩操作，维度 $c^I$）打分；indexer 查询低秩生成、与主注意力**共享**查询潜向量 $\mathbf c_t^Q$；分数 = 各头 ReLU 点积的加权和；top-k 选出压缩条目。** §2.3.1 式 (13)–(17)。grep：`Lightning Indexer`、`shared with that used for the indexer`。

**C24. 核心注意力是“共享 KV 的 MQA”：每个被选中的压缩条目同时充当键和值。** §2.3.1 式 (18)–(19)。grep：`serves as both attention key and value`。

**C25. 分组输出投影：$c\,n_h$ 很大，直接投到 $d$ 太贵；先把 $n_h$ 个头分成 $g$ 组，每组投到 $d_g$ 维（$d_g < c\,n_h/g$），再拼接投到 $d$。** §2.3.1。grep：`Grouped Output Projection`。

**C26. HCA 的压缩与 CSA 类似，但压缩率 $m'$ 更大、不重叠；然后对全部压缩条目做共享 KV 的 MQA 与分组输出投影。** §2.3.2 式 (20)–(26)。grep：`does not perform overlapped compression`。

**C27. Q 与 KV 条目在核心注意力前各做一次 RMSNorm（每个查询头；KV 只有一个头），以防注意力 logit 爆炸。** §2.3.3。grep：`Entry Normalization`。

**C28. 部分 RoPE：只对查询、KV 条目的**最后 64 维**加 RoPE；因为 KV 条目同时当值，输出会带上绝对位置，所以再对每个头输出的最后 64 维施加“负位置”的 RoPE，使输出只含相对位置。** §2.3.3。grep：`last 64 dimensions`。
（论文写“position $-i$”，$i$ 在式 (19) 中是**头**的下标；而同一小节的式 (27) 又用 $i$ 表示**查询 token**——记号混用。代码用的是查询 token 的位置 $t$ 做逆旋转——见勘误 E8。）

**C29. 附加滑动窗口分支：为了严格因果，查询只能看已完成的压缩块，看不到自己所在块里的其他 token；因此给 CSA、HCA 都加一个 $n_{\text{win}}$ 个未压缩最近 token 的滑窗分支，与压缩条目一起进入核心注意力。** §2.3.3。grep：`Sliding Window Attention`。
§4.2.1：$n_{\text{win}}=128$（两款模型）。

**C30. attention sink：每个头一个可学习的 sink logit $z'_h$，其指数加到 softmax 分母上，使该头的注意力总和可以小于 1 甚至接近 0。** §2.3.3 式 (27)。grep：`Attention Sink`。

**C31. 效率来源四条：(1) KV 条目混合精度存储——RoPE 维 BF16、其余 FP8，比纯 BF16 小“近一半”；(2) indexer 内的注意力计算用 FP4；(3) top-k 比 V3.2 小，利好中短文本；(4) 最重要的是压缩与混合。** §2.3.4。grep：`mixed storage format`、`smaller attention top-k`。

**C32. 以 BF16 GQA8、头维 128 为基线，1M 时 V4 的 KV cache 约为其 2%。** §2.3.4。grep：`GQA8`。（复算见 §4.2：Pro 2.01%、Flash 2.00%。）

**C33. 层的排布：Flash 共 43 层，前 2 层纯滑窗注意力，其后 CSA/HCA 交错；Pro 共 61 层，前 2 层 HCA，其后交错。** §4.2.1。grep：`pure sliding window attention`、`For the first two layers, we use HCA`。
代码 `compress_ratios`（0=纯 SWA，4=CSA，128=HCA；最后一个元素属于 MTP 层）逐层确认：Flash = `[0,0,4,128,4,…,128,4,0]`（44 项：2 SWA + 21 CSA + 20 HCA + MTP 的 0）；
Pro = `[128,128,4,128,…,128,4,0]`（62 项：31 HCA + 30 CSA + MTP 的 0）。

### E. Muon（§2.4、§4.2.2、算法 1）

**C34. V4 对“大多数模块”用 Muon，理由是收敛更快、训练更稳。** §2.4。grep：`majority of modules`。

**C35. 仍用 AdamW 的部分：embedding、预测头、mHC 的静态偏置与门控因子、所有 RMSNorm 权重；其余一律 Muon。** §2.4。grep：`static biases and gating factors`。
（§4.2.2 两处重复这份清单时**漏写了 mHC 的静态偏置与门控因子**——以 §2.4 为准，见 E13。）

**C36. 沿用 Liu et al. 的三件事：对 Muon 参数也做权重衰减、Nesterov 技巧、把更新矩阵的 RMS 重新缩放以复用 AdamW 超参数；不同之处是用“混合牛顿–舒尔茨迭代”。** §2.4。grep：`reutilization of our AdamW`。

**C37. 混合 NS：共 10 步，前 8 步用 $(a,b,c)=(3.4445,-4.7750,2.0315)$ 快速把奇异值推近 1，后 2 步用 $(2,-1.5,0.5)$ 把奇异值精确稳定在 1。** §2.4 式 (28)。grep：`Hybrid Newton-Schulz`。

**C38. 因为能直接对 Q 和 KV 条目做 RMSNorm，注意力 logit 不会爆炸，所以 V4 **不用** QK-Clip。** §2.4。grep：`QK-Clip`。（论文把 QK-Clip 归给 Liu et al. 2025，出处有误，见 E7。）

**C39. 超参数：AdamW $\beta_1=0.9,\beta_2=0.95,\varepsilon=10^{-20}$，权重衰减 0.1；Muon 动量 0.95、权重衰减 0.1，每个更新矩阵的 RMS 缩放到 0.18。Pro 与 Flash 相同。** §4.2.2。grep：`0.18`。

**C40. Muon 与 ZeRO 的冲突与解法：Muon 需要完整梯度矩阵；稠密参数限制 ZeRO 并行规模、用背包算法分配矩阵（每 rank 不超过 5 个矩阵，padding 内存开销 <10%）；MoE 参数按专家独立优化、展平后均分；同形状参数合并批量做 NS；NS 用 BF16 矩阵乘仍稳定，于是把 MoE 梯度以随机舍入量化成 BF16 同步（通信量减半），并用“all-to-all + 本地 FP32 求和”替代树/环形 reduce-scatter。** §3.4.1。grep：`knapsack`、`stochastic rounding`。

### F. 基础设施（§3）

**C41. 专家并行的细粒度通信-计算重叠：把专家分成若干“波”（wave），一波通信完成就立刻计算；相对强的非融合基线，一般推理 1.50–1.73×，RL rollout / 高速 agent 服务等延迟敏感场景最高 1.96×；已在 NVIDIA GPU 与华为昇腾 NPU 上验证；CUDA 版 mega-kernel “MegaMoE” 开源在 DeepGEMM（PR #304）。** §3.1、图 5。grep：`1.96`、`MegaMoE`。
图 5（SVG 渲染后读出）给的**理论**加速：Comet 1.42×、V4 方案 1.92×（Flash 配置）。

**C42. 计算/通信比的平衡点：当峰值算力 $C$ 与互联带宽 $B$ 满足 $C/B \le V_{\text{comp}}/V_{\text{comm}}$ 时通信可完全隐藏；对 V4-Pro，每个 token-专家对 $6hd$ FLOPs、$3h$ 字节（FP8 dispatch + BF16 combine），化简为 $C/B\le 2d=6144$ FLOPs/Byte，即每 1 GB/s 带宽可掩盖 6.1 TFLOP/s。** §3.1 “Observations and Proposals”。grep：`6144`。
注意此处 $h$=隐藏维、$d$=专家中间维（3072）——与 §2 的 $d$ 冲突（E15）。

**C43. 给硬件厂商的四条建议：瞄准计算/通信比平衡点而非无限加带宽；功耗余量；更低延迟的跨 GPU 信号（dispatch 现用 pull 方式）；建议用不含指数和除法的逐元素激活替代 SwiGLU。** §3.1。grep：`Power Budget`。

**C44. TileLang：用融合核替代数百个细粒度 ATen 算子；Host Codegen 把每次调用的 CPU 侧检查从几十–几百微秒降到 <1 微秒；集成 Z3 SMT 求解器（QF_NIA）做整数分析；默认关闭 fast-math，追求与手写 CUDA 位级一致。** §3.2。grep：`Host Codegen`、`QF_NIA`。

**C45. 批不变与确定性核：解码注意力用“双核”策略（整序列单 SM + 末波多 SM 且累加顺序一致）；cuBLAS 全部换成 DeepGEMM，多数场景放弃 split-k；注意力反向改为每 SM 独立缓冲再确定性求和；MoE 反向用 token 顺序预处理与缓冲隔离；mHC 中输出维只有 24 的矩阵乘用“分块输出 + 确定性归约”。** §3.3。grep：`output dimension of only 24`。
（笔者核对：24 = $(2+n_{hc})\,n_{hc}$ = 4 + 4 + 16，对应 $A_l$ 4 个、$C_l$ 4 个、$B_l$ 16 个动态系数；代码 `mix_hc = (2 + hc_mult) * hc_mult`。）

**C46. 上下文并行（CP）两阶段通信：每个 rank 先把最后 $m$ 个未压缩 KV 发给下一个 rank，本地压出固定长度 $s/m+1$ 个条目（含 padding）；再 all-gather，并用融合的 select-and-pad 重排成总长 $\text{cp\_size}\cdot s/m$。训练样本是多序列打包，每段独立压缩，尾部不足 $m$ 的 token 丢弃。** §3.4.3。grep：`two-stage communication`。

**C47. 张量级激活检查点：用 TorchFX 追踪计算图，对标注张量反向找最小重计算子图；无额外训练开销、零拷贝复用存储指针。** §3.4.4。grep：`TorchFX`。

**C48. 推理 KV cache 两部分：“经典 KV cache”放 CSA/HCA 的压缩条目（每块覆盖 $\operatorname{lcm}(m,m')$ 个原始 token，产生 $k_1=\operatorname{lcm}/m$ 个 CSA 条目和 $k_2=\operatorname{lcm}/m'$ 个 HCA 条目）；“状态 cache”每请求一块固定大小，放 SWA 的最近 $n_{\text{win}}$ 个条目和尚未凑满一块的未压缩尾部。** §3.5.1、图 6。grep：`State Cache`、`lcm`。
笔者复算：$\operatorname{lcm}(4,128)=128$，$k_1=32$，$k_2=1$。0813 模型卡给的 vLLM 命令用 `--block-size 256`（=2×128，是 lcm 的倍数）。

**C49. 混合注意力违反了 PagedAttention 的基本假设（缓存策略各异 + 高性能核的对齐要求），V4 把 SWA 与未压缩尾部当作“状态空间模型”式的定长状态来管理，并与稀疏注意力核联合设计块布局。** §3.5.1。grep：`PagedAttention`。

**C50. 磁盘 KV cache（前缀复用）：CSA/HCA 的压缩条目全部落盘，命中时复用到最后一个完整压缩块，尾部不完整块需重算；SWA 的未压缩条目每层都有，体积约为压缩条目的 8 倍，提供三种策略：全量缓存、每 $p$ 个 token 周期性存检查点、完全不存（重算最后 $n_{\text{win}}\cdot L$ 个 token 即可恢复）。** §3.5.2。grep：`approximately 8 times`、`Zero SWA Caching`。

### G. 预训练（§4）

**C51. 数据：在 V3 语料上加强多样性与长文；过滤批量自动生成/模板化网页（防模型坍塌）；中期训练加入 agentic 数据增强编码；更大的多语料；重点整理科学论文、技术报告等长文档；总量超过 32T。** §4.1。grep：`model collapse`。

**C52. 分词器沿用 V3（新增少量上下文构造用特殊 token），词表仍为 128K；沿用 token-splitting 与 FIM；按来源打包文档减少截断；与 V3 不同，预训练使用样本级注意力掩码。** §4.1。grep：`sample-level attention masking`。
代码：`vocab_size: 129280`（“128K”是约数，见 E16）。

**C53. 训练长度与稀疏化日程：从 4K 起，逐步扩到 16K、64K、1M；Flash 前 1T token 用稠密注意力预热，在 64K 长度时引入稀疏注意力并保持到结束；引入稀疏时先短暂预热 lightning indexer；Pro 的稠密阶段更长，引入稀疏的方法相同（两阶段）。** §4.2.2。grep：`first 1T tokens`。

**C54. Batch 与学习率：Flash batch 从小逐步增到 75.5M token 并在大部分训练中保持；学习率 2000 步线性预热后保持 $2.7\times10^{-4}$，末期余弦衰减到 $2.7\times10^{-5}$；Pro 最大 batch 94.4M，峰值/末值 $2.0\times10^{-4}$/$2.0\times10^{-5}$。MTP 损失权重大部分时间 0.3，学习率开始衰减时改为 0.1。** §4.2.2。grep：`75.5M`、`94.4M`。

**C55. 训练不稳定：尖峰总与 MoE 层的离群值相关，路由机制会放大离群值；单纯回滚治标不治本。两招：Anticipatory Routing 与 SwiGLU Clamping；作者承认机理尚不清楚。** §4.2.3、§6。grep：`Anticipatory Routing`。

**C56. Anticipatory Routing：第 $t$ 步用当前参数 $\theta_t$ 算特征，但路由索引用历史参数 $\theta_{t-\Delta t}$ 计算；实现上在 $t-\Delta t$ 步提前取第 $t$ 步的数据、预先算好并缓存路由；额外墙钟开销约 20%；只在检测到 loss spike 时自动短回滚并开启，一段时间后恢复常规训练，因此总开销可忽略。** §4.2.3。grep：`approximately 20%`。（$\Delta t$ 的取值论文未给。）

**C57. SwiGLU Clamping：Flash 与 Pro 全程把 SwiGLU 的线性分量截断到 $[-10,10]$、门分量上限截断到 10。** §4.2.3。grep：`SwiGLU Clamping`。代码：`swiglu_limit: 10.0`；`Expert.forward` 对 `up` 双边截断、对 `gate` 只截上界，然后 `F.silu(gate) * up`。

### H. 后训练（§5）

**C58. 流程大体沿用 V3.2，但关键替换：把混合 RL 阶段**整体**换成 OPD。** §5.1。grep：`entirely replaced by On-Policy Distillation`。

**C59. 专家训练：每个领域先微调，再用领域提示与奖励做 GRPO，超参数与此前工作接近。** §5.1.1。grep：`Specialist Training`。

**C60. 推理强度：不同 RL 配置训练出面向不同推理量的专家；两款模型都支持三档（Non-think / Think High / Think Max）；每档在 RL 时用不同的长度惩罚与上下文窗口；Think Max 在系统提示最前面注入一段指令。** §5.1.1、表 2、表 3。grep：`Reasoning Efforts`、`Think Max`。
表 3 的注入指令在 MIT 许可的 `encoding/README.md` 里原样给出，其开头一句是 `Reasoning Effort: Absolute maximum with no shortcuts permitted.`（README 说它被放在 prompt 最开头、系统消息之前）。

**C61. 生成式奖励模型（GRM）：难验证任务不再训练标量奖励模型，而用带评分细则（rubric）的 RL 数据 + GRM 评判；并直接对 GRM 做 RL——actor 本身就充当 GRM，评判能力与生成能力联合优化，只需少量人工标注。** §5.1.1。grep：`Generative Reward Model`。

**C62. 工具调用新格式：特殊 token “|DSML|” + XML 风格调用，减少转义失败。** §5.1.1、表 4。grep：`DSML`。encoding README 给出了 `<｜DSML｜tool_calls>` / `<｜DSML｜invoke name=…>` / `<｜DSML｜parameter name=… string="true|false">` 的完整格式。

**C63. 交错思考（interleaved thinking）：有工具调用的场景里，**跨用户轮次**也完整保留全部推理内容（V3.2 在新用户消息到来时丢弃）；普通对话仍丢弃旧轮推理。** §5.1.1、图 7。grep：`Interleaved Thinking`。
API 文档对应规则：请求带 `tools` 参数时，之前所有轮的 `reasoning_content` 都要回传并拼进上下文；不带 `tools` 时回传也会被忽略（`ds-thinking`）。

**C64. Quick Instruction：把“要不要搜索”“起标题”“生成搜索词”“权威性需求”“领域”“是否读取 URL”等辅助任务做成专用特殊 token 直接接在输入后面，复用已算好的 KV cache，免去小模型的重复 prefill、降低 TTFT。** §5.1.1、表 5。grep：`Quick Instruction`。

**C65. OPD 目标 = 对 $N$ 个专家的反向 KL 加权和，需要从学生采样轨迹；10 个以上教师蒸馏一个学生；不用“逐 token KL 估计当优势”的省资源做法（方差大、不稳定），而做全词表 logit 蒸馏。** §5.1.2 式 (29)。grep：`more than ten teacher`、`full-vocabulary logit distillation`。

**C66. FP4 QAT（后训练阶段）：对 MoE 专家权重与 CSA indexer 的 QK 路径做 MXFP4 量化感知训练；indexer 的 QK 激活以 FP4 缓存、读取、相乘；index score 从 FP32 量化到 BF16，top-k 选择器提速 2×、KV 条目召回率保持 99.7%。** §5.2.1。grep：`99.7%`。

**C67. FP4→FP8 反量化无损：FP32 主权重 → FP4 → 反量化为 FP8 计算；只要每个 128×128 FP8 块内各 1×32 FP4 子块的缩放因子最大/最小比不超过某阈值，FP8（E4M3）多出的 2 个指数位就能吸收细粒度缩放；作者实测当前权重满足条件，因此 QAT 完全复用现有 FP8 训练框架；反向相当于直通估计（STE）；推理与 RL rollout 直接用原生 FP4 权重。** §5.2.1。grep：`lossless`。
（阈值论文未给；笔者推导为 $2^{14}$，见 §4.9。）

**C68. 全词表 OPD 的工程：教师权重放集中式分布式存储、按需 ZeRO 式分片加载；只缓存教师最后一层隐藏状态，训练时再过预测头重建 logits；按教师索引排序样本，保证每个教师头每个 mini-batch 只加载一次、显存里同时最多一个；精确 KL 用 TileLang 核计算。** §5.2.2。grep：`last-layer teacher hidden states`。

**C69. 可抢占、容错的 rollout：逐 token 的预写日志（WAL）+ 保存 KV；从头重生成未完成请求在数学上不正确——短回答更容易在中断中“存活”，会引入长度偏差。** §5.2.3。grep：`length bias`。

**C70. 沙箱平台 DSec：三个 Rust 组件（Apiserver、Edge、Watcher），跑在 3FS 上，单集群管理数十万并发沙箱；四种执行底座（函数调用、容器、Firecracker microVM、QEMU fullVM）共用一个 Python SDK。** §5.2.5。grep：`DSec`。

### I. 评测（§4.3、§5.3、§5.4）

**C71. 评测设置：推理/知识任务温度 1.0，Non-think/High/Max 三档的上下文窗口分别为 8K/128K/384K；代码 agent 最多 500 步、512K 上下文。** §5.3.1。grep：`384K`、`512K`。模型卡另建议 Think Max 至少 384K 上下文。

**C72. Codeforces 评分：14 场 Div.1（2025-05 至 2025-11，114 题），每题生成 32 个候选、无放回抽 10 个按随机顺序提交；每个抽样提交序列算出比赛得分 → 排名 → 估计评分，对所有抽样与排列取评分的**期望**（不是先对排名取期望），再对 14 场取平均；V4-Pro-Max 3206 分，在人类参赛者榜上排第 23。** §5.3.1、§5.3.2、表 6。grep：`23rd`。

**C73. 知识：Pro-Max 在 SimpleQA-Verified 上比现有开源基线高 20 个百分点，但仍落后 Gemini-3.1-Pro；推理：略逊 GPT-5.4 与 Gemini-3.1-Pro，作者估计落后前沿闭源约 3–6 个月。** §1、§5.3.2。grep：`3 to 6 months`。

**C74. 1M 上下文：MRCR 上 Pro 超过 Gemini-3.1-Pro、落后 Claude Opus 4.6；128K 以内检索非常稳定，超过 128K 后可见下降。** §5.3.2、图 9。grep：`beyond the 128K mark`。
图 9（MRCR 8-needle，SVG 渲染读数）：Pro-Max 在 8K/16K/32K/64K/128K/256K/512K/1024K 为 0.90/0.85/0.94/0.90/0.92/0.82/0.66/0.59；Flash-Max 为 0.91/0.84/0.87/0.85/0.87/0.76/0.60/0.49。

**C75. 形式化数学：Putnam-200 Pass@8（精简工具）上优于 Seed-Prover 等；混合形式-非形式的高算力流程在 Putnam-2025 上达到 120/120。** §5.3.2、图 8。grep：`120/120`。

**C76. 局限与未来：为控制风险保留了许多预验证过的组件与技巧，架构相对复杂，未来要精简；Anticipatory Routing 与 SwiGLU Clamping 的原理尚不清楚；将探索更稀疏的 embedding 模块（引 Engram）、低延迟架构、多模态。** §6。grep：`relatively complex`。

### J. 制品（模型卡、API）

**C77. 模型卡精度栏：Base 为 “FP8 Mixed”，指令模型为 “FP4 + FP8 Mixed”（MoE 专家 FP4、其余大多 FP8）。** `hf-v4pro` README 的 Model Downloads 表。与 config 一致：Base `expert_dtype: fp8`，指令模型 `expert_dtype: fp4`。

**C78. 推荐采样：temperature 1.0、top_p 1.0（Preview 卡）；0813 卡改为 agent 场景 top_p 0.95、其他 1.0，high/max 档建议最大输出 384K。** `hf-v4pro`、`hf-v4pro-0813`。

**C79. GA 版 0813 “建立在 V4-Pro（Preview）的模型结构上，并附加了 DSpark 投机解码模块”；0731 与 DSpark 版同结构；DSpark 版“不是新模型”，是同一 checkpoint 加投机解码模块（详见 DeepSpec 仓库）。** `hf-v4pro-0813`、`hf-v4flash-0731`、`hf-dspark`。

**C80. 思考强度（API，GA 之后）：三档 low/high/max，默认 high；请求的 minimal→low、medium→high、xhigh→high、ultra→max 映射。** `ds-thinking`、`ds-changelog` 2026-08-13 条。

---

## 2. 公式（原文转录 + 课程记号对照）

> 转录依据 arXiv v1 的 LaTeXML 源（`arxiv.py text 2606.19348`），式号即论文式号。**V4 论文用行向量约定**：$\mathbf h_t\in\R^{1\times d}$，
> 投影写 $\mathbf h_t\cdot W$，$W\in\R^{d\times(\cdot)}$；整段序列按行堆成 $H\in\R^{n\times d}$。课程默认列向量（style-guide §2），
> 改写时 $W$ 要转置、乘法顺序要反过来。

### 2.0 记号冲突总表（写课前必读）

| V4 原符号 | V4 含义 | 与课程/别处的冲突 | 建议课程写法 |
|---|---|---|---|
| $n$ | 序列长度（§2.3） | mHC 式 (2) 里 $n$ 又是矩阵阶数（$=n_{hc}$）；算法 1 里 $W\in\R^{n\times m}$ | 序列长度写 $T$（或 $L_{ctx}$）；mHC 写 $n_{hc}$ |
| $d$ | 隐藏维（§2）；**§3.1 里却是专家中间维**（3072） | 同一篇论文内冲突 | §3.1 改写成 $d_{ff}$（专家中间维）与 $d$（隐藏维） |
| $h$ | 式 (16) 里是 indexer 头下标；§3.1 里是隐藏维 | 同一篇论文内冲突 | 隐藏维一律 $d$；头下标用 $j$ |
| $d_c$ | **查询**压缩维（1024/1536） | 课程与 V2 论文里 $d_c$ 是 **KV** 压缩维，$d_c'$ 才是查询压缩维 | 写 $d_c'$ |
| $c$ | KV 条目维（=512，也是注意力头维） | 与 V3 的 $d_c=512$ 数值相同但含义不同：V4 的条目**直接**当 K 和 V，没有上投影 | 写 $c$（首次出现说明“=512，K 与 V 共用”） |
| $C^a,C^b,C,C^{\text{Comp}}$ | KV 候选/压缩条目矩阵 | $C_l$ 又是 mHC 输出映射；§3.1 的 $C$ 是峰值算力 | mHC 用第 16 课的 $H^{post}_l$ |
| $A_l,B_l,C_l$ | mHC 的输入/残差/输出映射 | mHC 论文与第 16 课用 $\mathcal H^{pre}_l,\mathcal H^{res}_l,\mathcal H^{post}_l$ | 用 $H^{pre}_l,H^{res}_l,H^{post}_l$，并注明 V4 的对应 |
| $B^a,B^b,B$ | 可学习的块内位置偏置（$m\times c$） | $B_l$（mHC）、batch size $B$、§3.1 带宽 $B$ | 写 $P^a,P^b$（position bias） |
| $m,\;m'$ | CSA、HCA 压缩率（4、128） | 算法 1 的 $W\in\R^{n\times m}$ | Muon 的矩阵形状改写 $W\in\R^{p\times q}$ |
| $k$ / Top-k | 被选压缩条目数（512/1024） | 与键向量 $\mathbf k$ 同字母 | 写 $k_{\text{sel}}$ |
| $I_{t,s}$ | index score | — | 沿用 |
| $\mathbf w_t^I,W^w$ | indexer 头权重 | — | 沿用 |
| $g,d_g$ | 输出分组数、每组中间维 | — | 沿用 |
| $\eta,\mu,\lambda,\gamma$ | Muon 学习率、动量、权重衰减、update RMS 缩放 | $\gamma$ 在第 11 课是偏置更新速度 | Muon 的 $\gamma$ 写 $\gamma_{\text{rms}}$ |

### 2.1 mHC（§2.2，式 (1)–(8)）

$$X_{l+1}=B_{l}X_{l}+C_{l}\mathcal{F}_{l}(A_{l}X_{l}) \tag{1}$$

其中 $X_l=[\mathbf x_{l,1};\ldots;\mathbf x_{l,n_{hc}}]^T\in\R^{n_{hc}\times d}$，$A_l\in\R^{1\times n_{hc}}$，$B_l\in\R^{n_{hc}\times n_{hc}}$，$C_l\in\R^{n_{hc}\times 1}$，$\mathcal F_l$ 是第 $l$ 层（如 MoE），输入输出都是 $d$ 维。

$$B_{l}\in\mathcal{M}\coloneq\{M\in\mathbb{R}^{n\times n}\mid M\mathbf{1}_{n}=\mathbf{1}_{n},\;\mathbf{1}_{n}^{T}M=\mathbf{1}_{n}^{T},\;M\geqslant 0\} \tag{2}$$

动态参数化：$\hat X_l=\operatorname{RMSNorm}(\operatorname{vec}(X_l))\in\R^{1\times n_{hc}d}$，

$$\tilde{A}_{l}=\alpha_{l}^{\mathrm{pre}}\cdot(\hat{X}_{l}W^{\mathrm{pre}}_{l})+S_{l}^{\mathrm{pre}} \tag{3}$$
$$\tilde{B}_{l}=\alpha_{l}^{\mathrm{res}}\cdot\operatorname{Mat}(\hat{X}_{l}W^{\mathrm{res}}_{l})+S_{l}^{\mathrm{res}} \tag{4}$$
$$\tilde{C}_{l}=\alpha_{l}^{\mathrm{post}}\cdot(\hat{X}_{l}W^{\mathrm{post}}_{l})^{T}+S_{l}^{\mathrm{post}} \tag{5}$$

$W^{\mathrm{pre}}_l,W^{\mathrm{post}}_l\in\R^{n_{hc}d\times n_{hc}}$，$W^{\mathrm{res}}_l\in\R^{n_{hc}d\times n_{hc}^2}$；$S^{\mathrm{pre}}_l\in\R^{1\times n_{hc}}$，$S^{\mathrm{post}}_l\in\R^{n_{hc}\times1}$，$S^{\mathrm{res}}_l\in\R^{n_{hc}\times n_{hc}}$；$\alpha^{\mathrm{pre}}_l,\alpha^{\mathrm{res}}_l,\alpha^{\mathrm{post}}_l\in\R$。

$$A_{l}=\sigma(\tilde{A}_{l}) \tag{6}$$
$$C_{l}=2\sigma(\tilde{C}_{l}) \tag{7}$$
$$M^{(t)}=\mathcal{T}_{r}(\mathcal{T}_{c}(M^{(t-1)})),\quad M^{(0)}=\exp(\tilde B_l),\quad B_l=M^{(t_{\max})},\ t_{\max}=20 \tag{8}$$

**课程记号对照**：$A_l\to H^{pre}_l$，$B_l\to H^{res}_l$，$C_l\to H^{post}_l$（第 16 课）。

**代码对照（`model.py` `Block.hc_pre/hc_post`，`kernel.py` `hc_split_sinkhorn_kernel`）**：
- 一个线性层一次性产出 $(2+n_{hc})n_{hc}=24$ 个“mix”系数（`hc_attn_fn`/`hc_ffn_fn` 形状 $24\times 4d$），乘以 $\operatorname{rsqrt}(\operatorname{mean}(x^2))$——即**不带可学习权重**的 RMSNorm 放到线性层之后（数学等价）；`hc_scale` 3 个数 = $\alpha^{pre},\alpha^{post},\alpha^{res}$；`hc_base` 24 个数 = 静态偏置 $S$。
- $A_l=\sigma(\cdot)+\epsilon$（代码多加 `hc_eps`=1e-6）；$C_l=2\sigma(\cdot)$。
- Sinkhorn 顺序：代码先做一次**行** softmax（指数 + 行归一化，再加 eps），接着列归一化，然后再做 19 轮“行、列”——共 20 次行归一化、20 次列归一化，**最后一步是列归一化**；论文式 (8) 每轮是先列后行、最后一步是行归一化。第 16 课笔记 C33 记录 mHC 论文与 V4 论文写法一致。
- `hc_post` 实际计算 $\mathbf y_k=\sum_j \texttt{comb}[j,k]\,\mathbf x_j$，即 $X_{l+1}=\texttt{comb}^{\top}X_l+\ldots$。
- **上两条合起来看，代码与论文是等价的（核查者推导，2026-09-24）**：令 $\texttt{comb}=B_l^{\top}$，则 `hc_post` 正是 $X_{l+1}=B_lX_l+\ldots$；论文对 $B_l$“先列后行”的迭代，换到转置矩阵上就是代码对 `comb`“先行后列”的迭代；论文最后一步的行归一化 = 代码最后一步的列归一化，两者都保证“每条输出残差流的混合权重精确和为 1”。所以两者除 eps 外逐步相同，不是实质差异；只是写“代码逐行对照”时 `comb[j,k]` 对应论文的 $B_l[k,j]$。
- 注意力子层和 MoE 子层**各有一套** mHC（`hc_attn_*`、`hc_ffn_*`），所以“每层两次 mHC”。最后的输出头前还有一次只含 $A$ 型汇聚（`hc_head`，sigmoid + eps）。

### 2.2 CSA：压缩 KV 条目（§2.3.1，式 (9)–(12)）

$$C^{a}=H\cdot W^{aKV},\quad C^{b}=H\cdot W^{bKV} \tag{9}$$
$$Z^{a}=H\cdot W^{aZ},\quad Z^{b}=H\cdot W^{bZ} \tag{10}$$

$H\in\R^{n\times d}$；$C^a,C^b,Z^a,Z^b\in\R^{n\times c}$；$W^{aKV},W^{bKV},W^{aZ},W^{bZ}\in\R^{d\times c}$；位置偏置 $B^a,B^b\in\R^{m\times c}$。

$$[S^{a}_{mi:m(i+1)-1};S^{b}_{m(i-1):mi-1}]=\operatorname{Softmax}_{\text{row}}([Z^{a}_{mi:m(i+1)-1}+B^{a};Z^{b}_{m(i-1):mi-1}+B^{b}]) \tag{11}$$
$$C^{\text{Comp}}_{i}=\sum_{j=mi}^{m(i+1)-1}S^{a}_{j}\odot C^{a}_{j}+\sum_{j=m(i-1)}^{mi-1}S^{b}_{j}\odot C^{b}_{j} \tag{12}$$

- $\operatorname{Softmax}_{\text{row}}$ 在 $2m$ 个元素上归一化——**对 $c$ 个通道各自独立**做（$Z$ 是 $n\times c$ 的，不是每 token 一个标量）。代码 `(kv * score.softmax(dim=2)).sum(dim=2)`，`score` 形状 `[b, n/m, 2m, c]`，softmax 沿 token 轴。所以压缩是“**逐通道的门控加权池化**”，不是“按 token 的标量加权平均”（勘误 E4）。
- $i=0$ 时 $Z^b$ 用 $-\infty$、$C^b$ 用 0 填充。
- 条目 $i$ 读 $2m=8$ 个 token（$[m(i-1),\,m(i+1)-1]$），步长 $m=4$，所以相邻条目的读取区间重叠 4 个 token；条目数仍是 $n/m$。
- 代码细节：`Compressor.wkv`/`wgate` 输出 $2c=1024$ 维，**后半**是当前块的 $C^a$/$Z^a$，**前半**是给下一个条目用的 $C^b$/$Z^b$（`overlap_transform`）；`ape` 形状 $m\times 2c$ 即 $[B^a,B^b]$。压缩在 FP32 里算，之后做 RMSNorm（`self.norm`）、对最后 64 维加 RoPE、对前 448 维做 FP8 模拟量化。

**课程记号（列向量）**：$\mathbf c^a_j=W^{aKV}\mathbf h_j\in\R^{c}$，$\mathbf z^a_j=W^{aZ}\mathbf h_j$；对每个通道 $r=1..c$：
$$[\mathbf s_j]_r=\frac{\exp([\mathbf z_j+\mathbf p_{j}]_r)}{\sum_{j'\in\mathcal W_i}\exp([\mathbf z_{j'}+\mathbf p_{j'}]_r)},\qquad \mathbf e_i=\sum_{j\in\mathcal W_i}\mathbf s_j\odot\mathbf c_j,$$
其中 $\mathcal W_i$ 是条目 $i$ 的 $2m$ 个 token（前 $m$ 个取 $b$ 路、后 $m$ 个取 $a$ 路），$\mathbf p_j$ 是按块内偏移取的位置偏置行。建议把压缩条目写成 $\mathbf e_i\in\R^{c}$（entry），避免与 $C_l$、$\mathbf c^{KV}_t$ 撞车。

### 2.3 CSA：lightning indexer 与 top-k（式 (13)–(17)）

$$\mathbf{c}_{t}^{Q}=\mathbf{h}_{t}\cdot W^{DQ} \tag{13}$$
$$[\mathbf{q}_{t,1}^{I};\mathbf{q}_{t,2}^{I};...;\mathbf{q}_{t,n_{h}^{I}}^{I}]=\mathbf{q}_{t}^{I}=\mathbf{c}_{t}^{Q}\cdot W^{IUQ} \tag{14}$$
$$[w_{t,1}^{I};w_{t,2}^{I};...;w_{t,n_{h}^{I}}^{I}]=\mathbf{w}_{t}^{I}=\mathbf{h}_{t}\cdot W^{w} \tag{15}$$
$$I_{t,s}=\sum_{h=1}^{n_{h}^{I}}w_{t,h}^{I}\cdot\text{ReLU}\left(\mathbf{q}^{I}_{t,h}\cdot K^{\text{IComp}}_{s}\right) \tag{16}$$
$$\mathcal{C}^{\text{SprsComp}}_{t}=\left\{C^{\text{Comp}}_{s}\ \Big|\ I_{t,s}\in\operatorname{Top-k}(I_{t,:})\right\} \tag{17}$$

$W^{DQ}\in\R^{d\times d_c}$，$W^{IUQ}\in\R^{d_c\times c^In^I_h}$，$W^w\in\R^{d\times n^I_h}$；论文写的可见范围是 $s<\operatorname{Floor}(t/m)$。

**代码对照（`Indexer.forward`）**：
1. indexer 查询来自**归一化后**的查询潜向量 `qr = q_norm(wq_a(x))`（即 $\operatorname{RMSNorm}(\mathbf c^Q_t)$），与主注意力共享；
2. 查询与 indexer 键的最后 64 维加 RoPE，然后都做 **Hadamard 旋转**（`rotate_activation`，归一化 Hadamard，正交变换不改点积），再按 32 维一块做 FP4（E2M1）模拟量化，缩放因子 UE8M0——论文只说 “FP4”，没提 Hadamard（第 14 课笔记记录 V3.2 的 indexer 同样有 Hadamard）；
3. 头权重乘常数 $c_I^{-1/2}\,(n^I_h)^{-1/2}$（`softmax_scale * n_heads ** -0.5`），论文式 (16) 没写；
4. 可见范围：代码掩码为 $s<\lfloor (t+1)/m\rfloor$（$t$ 从 0 计），即**以 $t$ 结尾的那一块也可见**；论文写 $s<\lfloor t/m\rfloor$。两者只在 $(t+1)\bmod m=0$ 时不同，都满足因果（勘误 E9）。
5. top-k 取 `min(index_topk, 可见条目数)`；当可见条目不足 $k$ 时就是“全选”。

**课程记号**：$\mathbf c^{Q}_t=W^{DQ}\mathbf h_t\in\R^{d_c'}$；$\mathbf q^I_{t,j}=W^{IUQ}_j\mathbf c^Q_t\in\R^{c^I}$；$w^I_{t,j}=(W^w\mathbf h_t)_j$；
$I_{t,s}=\sum_{j=1}^{n^I_h}w^I_{t,j}\,\operatorname{ReLU}(\mathbf q^{I\top}_{t,j}\mathbf k^I_s)$。与第 14 课 DSA 式 (1) 的唯一结构差别：$\mathbf k^I_s$ 是**压缩后**的 indexer 键（每 4 个 token 一个），所以打分次数是 $t/4$ 而不是 $t$。

### 2.4 CSA：共享 KV 的 MQA 与分组输出投影（式 (18)–(19)）

$$[\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}]=\mathbf{q}_{t}=\mathbf{c}_{t}^{Q}\cdot W^{UQ} \tag{18}$$
$$\mathbf{o}_{t,i}=\operatorname{CoreAttn}\left(\texttt{query=}\mathbf{q}_{t,i},\ \texttt{key=}\mathcal{C}^{\text{SprsComp}}_{t},\ \texttt{value=}\mathcal{C}^{\text{SprsComp}}_{t}\right) \tag{19}$$

$W^{UQ}\in\R^{d_c\times c\,n_h}$，$\mathbf o_{t,i}\in\R^{c}$。分组输出投影（无式号，§2.3.1 正文）：把 $\mathbf o_t\in\R^{c\,n_h}$ 切成 $g$ 组 $\mathbf o^G_{t,i}\in\R^{c\,n_h/g}$，每组投到 $\mathbf o^{G'}_{t,i}\in\R^{d_g}$（$d_g<c\,n_h/g$），拼接 $[\mathbf o^{G'}_{t,1};\ldots;\mathbf o^{G'}_{t,g}]\in\R^{d_g g}$ 再投到 $\hat{\mathbf o}_t\in\R^{d}$。

代码：`wo_a` 形状 $(g\cdot d_g)\times(c\,n_h/g)$，按组做 einsum `bsgd,grd->bsgr`（**块对角**的第一次投影）；`wo_b` 形状 $d\times(g\,d_g)$。模型卡把 $d_g$ 叫 `o_lora_rank`（=1024），$g$ 叫 `o_groups`。

**笔者合成的“一个 CSA 查询的完整核心注意力”**（把 §2.3.3 的附加部件放回去；课程可用作总公式）：
$$\mathcal K_t=\underbrace{\{\mathbf e_s: s\in\operatorname{TopK}(I_{t,:})\}}_{\le k_{\text{sel}}\text{ 个压缩条目}}\ \cup\ \underbrace{\{\mathbf u_j: t-n_{\text{win}}<j\le t\}}_{n_{\text{win}}\text{ 个未压缩滑窗条目}},$$
$$a_{t,i,x}=\frac{\exp(\tilde{\mathbf q}_{t,i}^{\top}\tilde{\mathbf x}/\sqrt{c})}{\sum_{\mathbf y\in\mathcal K_t}\exp(\tilde{\mathbf q}_{t,i}^{\top}\tilde{\mathbf y}/\sqrt{c})+\exp(z'_i)},\qquad
\mathbf o_{t,i}=R_{-t}\Big(\sum_{\mathbf x\in\mathcal K_t}a_{t,i,x}\,\tilde{\mathbf x}\Big),$$
其中波浪号表示“RMSNorm 后、最后 64 维按各自位置做了 RoPE”，$R_{-t}$ 表示对最后 64 维做位置 $-t$ 的旋转。滑窗条目 $\mathbf u_j$ 来自单独的投影 `wkv`（$d\to c$）+ `kv_norm`，不是压缩器的输出。softmax 缩放是 $c^{-1/2}=512^{-1/2}$（代码 `softmax_scale = head_dim ** -0.5`）。

### 2.5 HCA（§2.3.2，式 (20)–(26)）

$$C=H\cdot W^{KV} \tag{20}$$
$$Z=H\cdot W^{Z} \tag{21}$$
$$S_{m'i:m'(i+1)-1}=\operatorname{Softmax}_{\text{row}}(Z_{m'i:m'(i+1)-1}+B) \tag{22}$$
$$C^{\text{Comp}}_{i}=\sum_{j=m'i}^{m'(i+1)-1}S_{j}\odot C_{j} \tag{23}$$
$$\mathbf{c}_{t}^{Q}=\mathbf{h}_{t}\cdot W^{DQ} \tag{24}$$
$$[\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}]=\mathbf{q}_{t}=\mathbf{c}_{t}^{Q}\cdot W^{UQ} \tag{25}$$
$$\mathbf{o}_{t,i}=\operatorname{CoreAttn}\left(\texttt{query=}\mathbf{q}_{t,i},\ \texttt{key=}C^{\text{Comp}},\ \texttt{value=}C^{\text{Comp}}\right) \tag{26}$$

$W^{KV},W^Z\in\R^{d\times c}$，$B\in\R^{m'\times c}$，$C^{\text{Comp}}\in\R^{(n/m')\times c}$。HCA 不重叠、不做 top-k：查询对**全部**已完成的压缩块 + 滑窗条目做稠密注意力（代码 `get_compress_topk_idxs` 返回全部可见索引）。

### 2.6 其他细节（§2.3.3）

- **Q/KV 归一化**：代码里查询按头做 $\mathbf q\leftarrow \mathbf q/\sqrt{\operatorname{mean}(\mathbf q^2)+\epsilon}$（**没有可学习权重**），KV 条目用带权重的 RMSNorm（`kv_norm` / 压缩器的 `norm`）。
- **部分 RoPE 的相对位置性质（笔者推导，供 `::: deriv` 用）**：设 $R_\theta$ 为二维旋转，查询最后 64 维 $\tilde{\mathbf q}=R_{t}\mathbf q^{r}$，条目 $\tilde{\mathbf e}_s=R_{p_s}\mathbf e^{r}_s$（$p_s$ 为条目的位置）。
  (i) 打分：$\tilde{\mathbf q}^\top\tilde{\mathbf e}_s=\mathbf q^{r\top}R_{t}^{\top}R_{p_s}\mathbf e^{r}_s=\mathbf q^{r\top}R_{p_s-t}\mathbf e^{r}_s$，只依赖 $p_s-t$；
  (ii) 输出：$R_{-t}\sum_s a_s\tilde{\mathbf e}_s=\sum_s a_s R_{p_s-t}\mathbf e^{r}_s$，每个条目对输出的贡献也只依赖相对距离——这就是论文说的“输出也携带相对位置”。
  压缩条目的位置 $p_s$：代码取**块首 token 的位置** $p_s=m\,s$（`freqs_cis[:cutoff:ratio]`；decode 时 `start_pos + 1 - ratio`）；论文未写。
- **RoPE 基数（代码，论文未写）**：有压缩的层（CSA/HCA）用 `compress_rope_theta = 160000` 加 YaRN（`factor 16`，`original_max_position_embeddings 65536`，$16\times65536=1{,}048{,}576$ = `max_position_embeddings`，`beta_fast 32`、`beta_slow 1`）；**纯滑窗层**（Flash 的前 2 层、两款模型的 MTP 层）用 `rope_theta = 10000` 且不加 YaRN。
- **attention sink**：
$$s_{h,i,j}=\frac{\operatorname{Exp}(z_{h,i,j})}{\sum_{k}\operatorname{Exp}(z_{h,i,k})+\operatorname{Exp}(z'_{h})} \tag{27}$$
  代码 `sparse_attn_kernel` 在在线 softmax 结束后执行 `sum_exp += exp(attn_sink - scores_max)`，与式 (27) 一致。含义：$\sum_j s_{h,i,j}=1-\frac{e^{z'_h}}{\sum_k e^{z_{h,i,k}}+e^{z'_h}}<1$。

### 2.7 Muon（§2.4，算法 1 与式 (28)）

算法 1（逐行转录；输入：学习率 $\eta$、动量 $\mu$、权重衰减 $\lambda$、update 缩放因子 $\gamma$；对每个“逻辑上独立”的权重 $W\in\R^{n\times m}$）：
$$G_t=\nabla_W\mathcal L_t(W_{t-1})$$
$$M_t=\mu M_{t-1}+G_t$$
$$O'_t=\operatorname{HybridNewtonSchulz}(\mu M_t+G_t)\qquad(\text{Nesterov 技巧})$$
$$O_t=O'_t\cdot\sqrt{\max(n,m)}\cdot\gamma$$
$$W_t=W_{t-1}\cdot(1-\eta\lambda)-\eta O_t$$

牛顿–舒尔茨迭代（$M_0=M/\lVert M\rVert_F$）：
$$M_{k}=aM_{k-1}+b(M_{k-1}M_{k-1}^{T})M_{k-1}+c(M_{k-1}M_{k-1}^{T})^{2}M_{k-1} \tag{28}$$
V4：10 步，前 8 步 $(a,b,c)=(3.4445,-4.7750,2.0315)$，后 2 步 $(2,-1.5,0.5)$；$\gamma$ 使 update RMS=0.18（§4.2.2）。

**标准形式（`muon-liu` §2.1 式 (1)–(2)，意译转录）**：$M_t=\mu M_{t-1}+\nabla\mathcal L_t(W_{t-1})$，$O_t=\text{Newton-Schulz}(M_t)$（脚注：实际按 Jordan 的做法把 $\mu M_t+\nabla\mathcal L_t$ 送进 NS，即 Nesterov），$W_t=W_{t-1}-\eta_tO_t$；NS 求的是 $(M_tM_t^{\top})^{-1/2}M_t=UV^{\top}$；式 (2) 与 V4 式 (28) 相同，系数沿用 Jordan 的 $(3.4445,-4.7750,2.0315)$。
**Lemma 1**（`muon-liu` §2.2）：形状 $[A,B]$ 的满秩矩阵，Muon update 的理论 RMS 为 $\sqrt{1/\max(A,B)}$。
**调整后的更新**（`muon-liu` §2.2 “Matching update RMS of AdamW”）：
$$W_t=W_{t-1}-\eta_t\big(0.2\cdot O_t\cdot\sqrt{\max(A,B)}+\lambda W_{t-1}\big)$$
理由：AdamW 的 update RMS 经验上在 0.2–0.4，把 Muon 的 update RMS 调到这个量级即可直接复用 AdamW 的学习率和权重衰减。**V4 把 0.2 换成 0.18**（E10）。

**Jordan 博客（`muon-blog`，意译）**：Muon 只用于隐藏层的二维参数；标量/向量参数、以及输入层（embedding）和输出层（分类头）应该用 AdamW 之类；NS 五步、系数 $(3.4445,-4.7750,2.0315)$，先除以 Frobenius 范数、在 bfloat16 里算、矩阵“高”时先转置；基线系数 $(2,-1.5,0.5)$ 也能收敛但更慢；调参目标是让 $\varphi(x)=ax+bx^3+cx^5$ 在 $[0,1]$ 上迭代后落入约 $[0.7,1.3]$、同时让 $a=\varphi'(0)$ 尽量大；每步 NS 的额外 FLOPs 对 $n\times m$（$m\le n$）矩阵为 $2(2nm^2+m^3)$。

**课程记号**：把 $W\in\R^{n\times m}$ 改写为 $W\in\R^{p\times q}$；$\varphi_{A}(x)=3.4445x-4.7750x^3+2.0315x^5$，$\varphi_B(x)=2x-1.5x^3+0.5x^5$；对 $M=U\Sigma V^\top$，NS 只作用于奇异值：$\varphi(M)=U\varphi(\Sigma)V^\top$（笔者推导：$(MM^\top)^kM=U\Sigma^{2k+1}V^\top$）。

### 2.8 OPD（§5.1.2，式 (29)）

$$\mathcal{L}_{\text{OPD}}(\theta)=\sum_{i=1}^{N}w_{i}\cdot\text{D}_{\text{KL}}\left(\pi_{\theta}\parallel\pi_{E_{i}}\right) \tag{29}$$

反向 KL 需从学生 $\pi_\theta$ 采样轨迹。论文批评的“常见简化”：把逐位置的全词表 KL 换成 token 级估计，并把 $\texttt{sg}\big[\log\frac{\pi_{E_i}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}\big]$ 当作每 token 优势塞进策略梯度损失——省资源但方差大；V4 改为在每个位置用完整 logits 精确算
$\mathrm{D}_{\mathrm{KL}}(\pi_\theta(\cdot|x,y_{<t})\,\|\,\pi_{E_i}(\cdot|x,y_{<t}))=\sum_{v\in\mathcal V}\pi_\theta(v|\cdot)\log\frac{\pi_\theta(v|\cdot)}{\pi_{E_i}(v|\cdot)}$（笔者展开）。

### 2.9 计算/通信平衡（§3.1，无式号）

$$\frac{C}{B}\le\frac{V_{\text{comp}}}{V_{\text{comm}}}=\frac{6hd}{3h}=2d=6144\ \mathrm{FLOPs/Byte}\quad(d=3072,\ \text{V4-Pro})$$
（$6hd$：SwiGLU 的 gate、up、down 三个 $h\times d$ 矩阵乘，每个乘加 2 FLOPs；$3h$ 字节：FP8 dispatch $h$ 字节 + BF16 combine $2h$ 字节。）

### 2.10 KV 字节与单 token FLOPs 的解析式（笔者推导，§4 复算用）

记 $b_e$ = 一个主 KV 条目的字节（V4 为 584），$b_I$ = 一个 indexer 键的字节，$\mathcal L_C,\mathcal L_H$ 为 CSA/HCA 层集合。每 token 摊到的**增长型** KV（不含常数项的滑窗与尾部状态）：
$$\text{KV/token}=\sum_{\ell\in\mathcal L_C}\frac{b_e+b_I}{m}+\sum_{\ell\in\mathcal L_H}\frac{b_e}{m'}$$
位置 $t$ 处单个 token 的前向 FLOPs（每乘加 2 FLOPs；只算投影 $2N_{\text{act}}$、核心注意力 $QK^\top$ 与 $PV$、indexer 打分）：
$$F(t)=2N_{\text{act}}+w_{\text{BF16}}\Big[\sum_{\ell\in\mathcal L_C}2n_h(2c)\big(\min(k_{\text{sel}},\tfrac{t}{m})+n_{\text{win}}\big)+\sum_{\ell\in\mathcal L_H}2n_h(2c)\big(\tfrac{t}{m'}+n_{\text{win}}\big)+\sum_{\ell\in\mathcal L_S}2n_h(2c)\,n_{\text{win}}\Big]+w_{\text{FP4}}\sum_{\ell\in\mathcal L_C}2n^I_hc^I\tfrac{t}{m}$$
V3.2（MLA 的 MQA 模式，QK 用 576 维、PV 用 512 维；indexer FP8）：
$$F_{3.2}(t)=2N_{\text{act}}+w_{\text{BF16}}\cdot 61\cdot n_h(2\cdot576+2\cdot512)\min(t,2048)+w_{\text{FP8}}\cdot 61\cdot 2n^I_hc^I\,t$$
权重 $w$ 是“等效 FP8 FLOPs”的换算系数。§4.4 显示：取 $w_{\text{BF16}}=2,\ w_{\text{FP8}}=1,\ w_{\text{FP4}}=0.5$ 时三条曲线与论文图 1 的截距、斜率都吻合（**笔者推测**，论文没写换算规则）。

---

## 3. 规格与超参数表

### 3.1 架构（论文 §4.2.1 vs `config.json`；V3.2 列供对照）

| 项 | V4-Pro | V4-Flash | V3.2（对照） | 来源 |
|---|---|---|---|---|
| Transformer 层数 | 61 | 43 | 61 | §4.2.1；config `num_hidden_layers`；`hf-v32-config` |
| 隐藏维 $d$ | 7168 | 4096 | 7168 | §4.2.1；config `hidden_size` |
| 注意力排布 | 前 2 层 HCA，之后 CSA/HCA 交错 → **30 CSA + 31 HCA** | 前 2 层纯 SWA，之后交错 → **21 CSA + 20 HCA + 2 SWA** | 61 层 MLA+DSA | §4.2.1；config `compress_ratios`（笔者逐项计数） |
| 查询头数 $n_h$ | 128 | 64 | 128 | §4.2.1；config `num_attention_heads` |
| KV 头数 | 1（共享 KV 的 MQA） | 1 | MLA（解码时 MQA 模式） | config `num_key_value_heads: 1`；V3.2 config 名义上写 128 |
| 条目维/头维 $c$ | 512 | 512 | MQA 模式 K 576 / V 512 | §4.2.1；config `head_dim`；FlashMLA README |
| RoPE 维 | 最后 64 维 | 最后 64 维 | 64（解耦 RoPE） | §2.3.3；config `qk_rope_head_dim` |
| 查询压缩维（论文 $d_c$，课程 $d_c'$） | 1536 | 1024 | 1536 | §4.2.1；config `q_lora_rank` |
| CSA 压缩率 $m$ | 4（重叠） | 4（重叠） | — | §4.2.1；代码 `overlap = compress_ratio == 4` |
| HCA 压缩率 $m'$ | 128 | 128 | — | §4.2.1 |
| 稀疏 top-k | 1024 个压缩条目 | 512 个压缩条目 | 2048 个 token | §4.2.1；config `index_topk`；`dsv32` §2.1.1 |
| indexer 头数 $n^I_h$ / 头维 $c^I$ | 64 / 128 | 64 / 128 | 64 / 128 | §4.2.1；config `index_n_heads`/`index_head_dim` |
| 输出分组 $g$ / $d_g$ | 16 / 1024 | 8 / 1024 | — | §4.2.1；config `o_groups`/`o_lora_rank` |
| 滑窗 $n_{\text{win}}$ | 128 | 128 | — | §4.2.1；config `sliding_window` |
| attention sink | 每头 1 个 | 每头 1 个 | 无 | §2.3.3；代码 `attn_sink` |
| MoE 覆盖 | 全部层；前 3 层 Hash 路由 | 同左 | 前 3 层稠密 FFN（中间维 18432） | §2.1、§4.2.1；config `num_hash_layers`；V3.2 `first_k_dense_replace: 3` |
| 路由专家 / 共享专家 / 激活 | 384 / 1 / 6 | 256 / 1 / 6 | 256 / 1 / 8 | §4.2.1；config |
| 专家中间维 | 3072 | 2048 | 2048 | §4.2.1；config `moe_intermediate_size` |
| 共享专家中间维 | 3072（同路由专家） | 2048 | 2048 | 代码 `shared_experts = Expert(dim, moe_inter_dim)`；论文未写 |
| 亲和度函数 | $\sqrt{\operatorname{softplus}}$ | 同左 | sigmoid | §2.1；config `scoring_func` |
| `routed_scaling_factor` | 2.5 | 1.5 | 2.5 | config（论文未写） |
| 节点受限路由 | 无 | 无 | `n_group 8` / `topk_group 4` | §2.1；config |
| MTP 深度 | 1 | 1 | 1 | §4.2.1；config `num_nextn_predict_layers` |
| mHC $n_{hc}$ / $t_{\max}$ | 4 / 20 | 4 / 20 | 无 | §4.2.1；config `hc_mult`/`hc_sinkhorn_iters` |
| SwiGLU 截断 | 线性分量 $[-10,10]$，门分量 $\le 10$ | 同左 | 无 | §4.2.3；config `swiglu_limit: 10.0` |
| 词表 | “128K”（config 129,280） | 同左 | 129,280 | §4.1；config `vocab_size` |
| 最大位置 | 1,048,576 | 1,048,576 | 163,840（论文说只支持 128K） | config；`dsv32` §4 |
| RoPE | 压缩层：θ=160000 + YaRN（×16，原长 65536）；纯 SWA 层：θ=10000 无 YaRN | 同左 | θ=10000 + YaRN（×40，原长 4096） | config / `model.py`（论文未写） |
| 总参数 | 1.6T（论文）；复算 1,573.0B 不含 MTP、1,598.8B 含 MTP | 284B；复算 284.3B 不含 MTP、290.9B 含 | 671B | 摘要；§4.4 复算 |
| 激活参数 | 49B；复算 48.85B（不含输入 embedding）/ 49.78B（含） | 13B；复算 13.27B / 13.80B | 37B | 同上 |
| 发布权重精度 | 路由专家 FP4（MXFP4：E2M1，每 32 个一个 UE8M0 缩放），其余多为 FP8 E4M3（128×128 块，UE8M0 缩放） | 同左 | FP8（UE8M0 缩放） | 模型卡；config `expert_dtype`、`quantization_config`；`model.py` `Linear` |
| Base 权重精度 | FP8 Mixed（`expert_dtype: fp8`） | 同左 | — | 模型卡；Base config |
| checkpoint 总字节 | 864,704,792,696 B（≈864.7 GB） | 159,609,485,896 B（≈159.6 GB） | — | `model.safetensors.index.json` `metadata.total_size` |

### 3.2 每层注意力模块的参数形状（`model.py` `Attention.__init__`，行向量形状写成 in→out）

| 张量 | 形状（Pro） | 形状（Flash） | 说明 |
|---|---|---|---|
| `wq_a`（$W^{DQ}$） | 7168→1536 | 4096→1024 | 查询下投影，后接 `q_norm` |
| `wq_b`（$W^{UQ}$） | 1536→128×512=65536 | 1024→64×512=32768 | 查询上投影 |
| `wkv` | 7168→512 | 4096→512 | **滑窗分支**的未压缩条目 |
| `compressor.wkv`/`wgate` | CSA：7168→1024（两路 $a,b$）；HCA：7168→512 | 同理（4096→…） | $W^{aKV},W^{bKV}$ / $W^{aZ},W^{bZ}$；HCA 只有一路 |
| `compressor.ape` | CSA：4×1024；HCA：128×512 | 同 | 位置偏置 $B^a,B^b$ / $B$ |
| `indexer.wq_b`（$W^{IUQ}$） | 1536→64×128 | 1024→64×128 | 仅 CSA 层 |
| `indexer.weights_proj`（$W^w$） | 7168→64 | 4096→64 | 仅 CSA 层 |
| `indexer.compressor` | wkv/wgate 7168→256，ape 4×256 | 4096→256 | indexer 键的压缩器（$c^I=128$，重叠两路） |
| `wo_a` | 16 组 × (4096→1024) | 8 组 × (4096→1024) | 分组第一次投影，checkpoint 为 FP8，参考代码按 BF16 einsum |
| `wo_b` | 16384→7168 | 8192→4096 | 拼接后投到 $d$ |
| `attn_sink` | 128 | 64 | 每头一个 logit |

### 3.3 预训练超参数（§4.2.2）

| 项 | V4-Flash | V4-Pro |
|---|---|---|
| 训练 token | 32T | 33T |
| 优化器 | Muon（大部分）+ AdamW（embedding、预测头、RMSNorm；§2.4 另含 mHC 静态偏置与门控因子） | 同左 |
| AdamW | $\beta_1=0.9,\ \beta_2=0.95,\ \varepsilon=10^{-20}$，wd 0.1 | 同左 |
| Muon | 动量 0.95，wd 0.1，update RMS 缩放到 0.18；Nesterov；混合 NS 8+2 步 | 同左 |
| batch（token） | 由小增到 **75.5M** 后保持 | 最大 **94.4M** |
| 学习率 | 线性预热 2000 步 → $2.7\times10^{-4}$ 保持 → 末期余弦衰减到 $2.7\times10^{-5}$ | 峰值 $2.0\times10^{-4}$ → 末值 $2.0\times10^{-5}$（其余“大体相同”） |
| 序列长度 | 4K → 16K → 64K → 1M | 同左 |
| 稠密→稀疏 | 前 1T token 稠密注意力预热；64K 阶段引入稀疏（先短暂预热 indexer），之后一直稀疏 | 稠密阶段更长（长度未给），引入方式相同 |
| 无辅助损失偏置更新速度 | 0.001 | 0.001 |
| 序列级均衡损失权重 | 0.0001 | 0.0001 |
| MTP 损失权重 | 0.3；学习率开始衰减时改为 0.1 | 同左 |
| 稳定性 | Anticipatory Routing（尖峰时触发）+ SwiGLU Clamping（全程） | 同左 |
| 数据处理 | 沿用 V3 的预处理、token-splitting、FIM；样本级注意力掩码 | 同左 |

### 3.4 优化器分工（§2.4 为准）

| 用 AdamW | 用 Muon |
|---|---|
| embedding（129,280×$d$） | 所有注意力投影：`wq_a`、`wq_b`、`wkv`、压缩器 `wkv`/`wgate`、indexer `wq_b`/`weights_proj`、`wo_a`、`wo_b` |
| 预测头（129,280×$d$） | 所有专家（每个专家的 gate/up/down 各自独立正交化）与共享专家 |
| 所有 RMSNorm 权重 | mHC 的动态投影 $W^{pre},W^{res},W^{post}$（论文只把 mHC 的**静态偏置与门控因子**列给 AdamW） |
| mHC 静态偏置 $S$ 与门控因子 $\alpha$ | 路由器打分矩阵（笔者推测：它是二维矩阵且不在 AdamW 清单上） |
| （未说明：attention sink、压缩器位置偏置 `ape`、Hash 查表不训练） | MTP 模块内的同类矩阵（笔者推测） |

### 3.5 数值精度地图（论文 + 代码）

| 位置 | 精度 | 来源 |
|---|---|---|
| KV 条目（CSA/HCA 压缩条目、SWA 条目） | 前 448 维 FP8 E4M3（每 64 维一个 UE8M0 缩放）+ 最后 64 维 BF16；生产格式每条目 584 B（448 + 128 + 8 字节缩放，其中 1 字节 padding） | §2.3.4；FlashMLA README；`model.py` `act_quant(kv[..., :-rd], 64, …)` |
| indexer 的 Q、K（缓存、读取、相乘） | FP4（MXFP4），Hadamard 旋转后量化，每 32 个一个 UE8M0 缩放 | §2.3.4、§5.2.1；`model.py` `fp4_act_quant` |
| index score | 后训练 QAT 时 FP32 → BF16（top-k 选择提速 2×，召回 99.7%） | §5.2.1 |
| 路由专家权重 | 预训练：FP8（Base 发布为 FP8）；后训练 QAT：FP32 主权重 → FP4 → 无损反量化到 FP8 计算；推理/rollout：原生 FP4 | §5.2.1；模型卡；config |
| 其他线性层 | FP8 E4M3，128×128 块缩放 | config `quantization_config` |
| 核心注意力计算 | 反量化到 BF16 做乘法（FlashMLA：FP8 KV cache、BF16 矩阵乘） | FlashMLA README；`kernel.py` `sparse_attn_kernel` 的 BF16 输入 |
| 压缩器内部 | FP32（`wkv`/`wgate` checkpoint 为 BF16，代码转成 FP32 计算） | `model.py` 注释 |
| NS 迭代 | BF16 矩阵乘（稳定） | §3.4.1 |
| MoE 梯度的数据并行同步 | BF16（随机舍入），all-to-all 后本地 FP32 求和 | §3.4.1 |
| EP 通信 | dispatch FP8、combine BF16 | §3.1 |
| RMSNorm、lm_head、mHC | checkpoint 中 RMSNorm 与 lm_head 为 BF16（代码注释）；参考代码把它们和 mHC 参数都放在 FP32 里计算 | `model.py` 注释与 `set_dtype(torch.float32)` |

### 3.6 Preview → GA / DSpark 的 config 差异（`diff` 结果）

| 模型 | 与 Preview config 的差异 |
|---|---|
| V4-Pro-0813（GA） | `compress_ratios` 末尾多两个 `0`（共 64 项）；新增 `dspark_block_size: 5`、`dspark_noise_token_id: 128799`、`dspark_target_layer_ids: [58, 59, 60]`、`dspark_markov_rank: 512` |
| V4-Pro-DSpark | 与 0813 的 config **完全相同** |
| V4-Flash-0731（GA） | 同理：`compress_ratios` 多两个 `0`；`dspark_target_layer_ids: [40, 41, 42]`、`dspark_markov_rank: 256`，其余 DSpark 字段同上 |
| V4-Pro-Base / Flash-Base | 只差 `expert_dtype: fp8` |

（多出的两个 `0` 应是 DSpark 模块的两层纯滑窗注意力——**笔者推测**，DeepSpec 仓库未核对。）

### 3.7 后训练、评测与部署设置

| 项 | 值 | 来源 |
|---|---|---|
| 专家训练 | 各领域 SFT → GRPO（超参数接近 R1/V3.2） | §5.1.1 |
| 推理档位（Preview） | Non-think（`</think>` 直接接总结）/ Think High / Think Max（系统提示最前注入指令） | 表 2、表 3；模型卡 |
| 推理档位（GA API） | `reasoning_effort` = low / high / max，默认 high | `ds-thinking`、`ds-changelog` |
| OPD | >10 个教师；反向 KL；全词表 logit 蒸馏 | §5.1.2 |
| 评测温度 / 上下文 | 1.0；Non-think 8K、High 128K、Max 384K | §5.3.1 |
| 代码/搜索 agent | 最多 500 步，最大上下文 512K；BrowseComp 用与 V3.2 相同的“全部丢弃”上下文管理 | §5.3.1 |
| 推荐采样 | Preview：temperature 1.0、top_p 1.0；GA：agent 场景 top_p 0.95 | 模型卡 |
| vLLM（0813 卡示例） | 单个 4×GB300 节点，`--kv-cache-dtype fp8 --block-size 256`，`use_fp4_indexer_cache: true`，DSpark `num_speculative_tokens: 7` | `hf-v4pro-0813` |
| API 价格（截至 2026-09-24，每百万 token，美元） | V4-Pro-0813：缓存命中 0.022（谷）/0.044（峰）；未命中 0.66/1.32；输出 1.98/3.96；上下文 1M，最大输出 384K | `ds-pricing` |

### 3.8 基准要点（数字取自 MIT 许可的模型卡，与论文表 1/6/7 一致；只列课上可能用到的）

**基座（表 1，内部框架统一设置；差距 ≤0.3 视为同一水平）**

| 基准 | V3.2-Base | V4-Flash-Base | V4-Pro-Base |
|---|---|---|---|
| 激活 / 总参数 | 37B / 671B | 13B / 284B | 49B / 1.6T |
| MMLU-Pro (5-shot) | 65.5 | 68.3 | 73.5 |
| Simple-QA verified (25-shot) | 28.3 | 30.1 | 55.2 |
| FACTS Parametric (25-shot) | 27.1 | 33.9 | 62.6 |
| HumanEval (0-shot) | 62.8 | 69.5 | 76.8 |
| BigCodeBench (3-shot) | **63.9** | 56.8 | 59.2 |
| MATH (4-shot) | 60.5 | 57.4 | 64.5 |
| LongBench-V2 (1-shot) | 40.2 | 44.7 | 51.5 |

（反例值得课上提：BigCodeBench 63.9→56.8、MATH 60.5→57.4、BBH 87.6→86.9、TriviaQA 83.3→82.8、HellaSwag 86.4→85.7、CLUEWSC 83.5→82.2、GSM8K 91.1→90.8，这些基准上 Flash-Base 低于 V3.2-Base；Pro-Base 在 BigCodeBench（59.2）、CMath（90.9 vs 92.6）上也低于 V3.2-Base——“多数基准超过”不是“全部”。）

**指令模型（表 7，三档对比）**

| 基准 | Flash Non-Think | Flash High | Flash Max | Pro Non-Think | Pro High | Pro Max |
|---|---|---|---|---|---|---|
| GPQA Diamond | 71.2 | 87.4 | 88.1 | 72.9 | 89.1 | 90.1 |
| HLE | 8.1 | 29.4 | 34.8 | 7.7 | 34.5 | 37.7 |
| HMMT 2026 Feb | 40.8 | 91.9 | 94.8 | 31.7 | 94.0 | 95.2 |
| Codeforces (rating) | – | 2816 | 3052 | – | 2919 | 3206 |
| MRCR 1M (MMR) | 37.5 | 76.9 | 78.7 | 44.7 | 83.3 | 83.5 |
| CorpusQA 1M | 15.5 | 59.3 | 60.5 | 35.6 | 56.5 | 62.0 |
| SWE Verified | 73.7 | 78.6 | 79.0 | 73.6 | 79.4 | 80.6 |
| Terminal Bench 2.0 | 49.1 | 56.6 | 56.9 | 59.1 | 63.3 | 67.9 |

**与闭源/开源对比（表 6，Pro-Max 一列与各家最强档）**：Pro-Max 在 LiveCodeBench（93.5）、Codeforces（3206）、Apex Shortlist（90.2）三项最高；SimpleQA-Verified 57.9 vs Gemini-3.1-Pro 75.6；MRCR 1M 83.5 vs Opus-4.6 92.9、Gemini-3.1-Pro 76.3。

**GA 版（模型卡 `hf-v4pro-0813`，与 Preview 不同的一套基准）**：Terminal Bench **2.1**：0813 为 87.9，Preview Pro 为 72.1，Flash-0731 为 82.7；HLE（无/有工具）：0813 42.7/60.0，Preview Pro 37.7/48.2。

---

## 4. 复算的数字（Python 表达式 → 值）

> 约定：“1M” = $2^{20}=1{,}048{,}576$ token（论文图 1 横轴以 K=1024 为单位，§4.4 的拟合支持这一点）；GB = $10^9$ 字节，GiB = $2^{30}$。

### 4.1 参数量（按 `config.json` + `model.py` 逐项计数）

| 量 | 表达式（要点） | 值 |
|---|---|---|
| 每个专家参数（Pro） | `3*7168*3072` | 66,060,288 ≈ 66.06M |
| 每个专家参数（Flash） | `3*4096*2048` | 25,165,824 ≈ 25.17M |
| 路由专家总参（Pro） | `384*66.06e6*61` | 1,547.4B（占总参 98.4%） |
| 路由专家总参（Flash） | `256*25.17e6*43` | 277.0B |
| 注意力（所有层合计） | 逐层累加 `Attention`+`Compressor`+`Indexer` 参数 | Pro 19.47B；Flash 5.09B |
| embedding / 预测头 | `129280*7168`；`129280*4096` | Pro 各 0.927B；Flash 各 0.530B |
| 主干总参（不含 MTP） | 各层 + embedding + head | **Pro 1,573.0B；Flash 284.3B** |
| 含 MTP | + MTP 块（Pro 25.84B，Flash 6.61B） | Pro 1,598.8B；Flash 290.9B |
| 激活（不含输入 embedding） | 注意力 + 6 路由 + 1 共享 + 路由器 + mHC + norm + head | **Pro 48.85B；Flash 13.27B** |
| 激活（含 embedding） | 上行 + embedding | Pro 49.78B；Flash 13.80B |
| mHC 动态投影每层 | `2*24*4*7168` / `2*24*4*4096` | Pro 1,376,256；Flash 786,432 |
| 分组输出投影 vs 直接投影（Pro） | `(16*1024*4096 + 16384*7168) / (65536*7168)` | 184.5M vs 469.8M → **39.3%** |
| 同上（Flash） | `(8*1024*4096 + 8192*4096) / (32768*4096)` | 67.1M vs 134.2M → **50.0%** |
| checkpoint 字节/参数（含 MTP） | `864.7e9/1598.8e9`；`159.6e9/290.9e9` | Pro 0.541；Flash 0.549（≈MXFP4 的 4.25 bit = 0.531 B，加上少量 FP8 权重） |
| MXFP4 每参数位数 | `4 + 8/32` | 4.25 bit |

结论：“284B”精确对上“不含 MTP 的主干”；“1.6T”是 1.573T（不含 MTP）或 1.599T（含 MTP）的四舍五入，论文没说口径。“49B/13B”与“不含输入 embedding 的激活参数”吻合（**笔者推测**的口径）。

### 4.2 KV cache 字节/token

| 量 | 表达式 | 值 |
|---|---|---|
| V3.2 主条目 | `512 + 4*4 + 64*2`（FP8 NoPE + 4 个 FP32 缩放 + BF16 RoPE） | 656 B |
| V3.2 indexer 键 | `128 + 4`（FP8 + 1 个 FP32 缩放；DeepGEMM `fp8_mqa_logits` 的 KV 格式） | 132 B |
| **V3.2 每 token** | `61*(656+132)` | **48,068 B** |
| V4 主条目 | `448 + 64*2 + 8` | 584 B |
| V4 indexer 键 | `128//2 + 128//32`（FP4 打包 + 4 个 UE8M0 缩放） | 68 B（参考代码格式，见 UNVERIFIED） |
| 每个 CSA 层摊到每 token | `(584+68)/4` | 163 B |
| 每个 HCA 层摊到每 token | `584/128` | 4.5625 B |
| **V4-Pro 每 token** | `30*(584+68)/4 + 31*584/128` | **5,031.44 B**（CSA 主 4,380 + CSA indexer 510 + HCA 141.44） |
| **V4-Flash 每 token** | `21*(584+68)/4 + 20*584/128` | **3,514.25 B**（3,066 + 357 + 91.25） |
| @1M（$2^{20}$） | `b*2**20/1e9` | V3.2 **50.40 GB**（46.94 GiB）；Pro **5.276 GB**（4.914 GiB）；Flash **3.685 GB**（3.432 GiB） |
| @$10^6$ | `b*1e6/1e9` | 48.07 / 5.031 / 3.514 GB |
| @128K | `b*131072/1e9` | 6.300 / 0.660 / 0.461 GB |
| 比值 | `48068/5031.44`；`48068/3514.25` | **9.55×（Pro 为 V3.2 的 10.47%）**；**13.68×（Flash 7.31%）** ← 与图 1 的 9.5×、13.7× 和正文 10%、7% 一致 |
| BF16 GQA8、头维 128 的基线 | `2*8*128*2*61`；`…*43` | 249,856 B（61 层）；176,128 B（43 层） |
| V4 / 基线 | `5031.44/249856`；`3514.25/176128` | **2.01%；2.00%** ← 论文“约 2%” |
| 纯 BF16 存 V4 条目 | `512*2` | 1024 B；混合精度 584 B = 57.0%（论文“近一半”） |
| V3 MLA 纯 BF16 | `576*2*61` | 70,272 B/token（对照阶梯图用） |
| SWA 条目若每 token 都落盘 | `61*584`；`43*584` | Pro 35,624 B；Flash 25,112 B |
| SWA / 压缩条目 | 含 indexer：`35624/5031.44`、`25112/3514.25`；不含 indexer：`35624/(30*146+31*4.5625)`、`25112/(21*146+20*4.5625)` | 7.08× / 7.15×；**7.88× / 7.95×** ← 论文“约 8 倍”对应不含 indexer 键的口径（笔者推测） |
| 每请求滑窗状态（常数） | `61*128*584`；`43*128*584` | Pro 4,559,872 B ≈ 4.56 MB；Flash 3.21 MB |
| Zero SWA Caching 重算量 | `128*61`；`128*43` | Pro 7,808 token；Flash 5,504 token |
| KV 块 | `lcm(4,128)`，`128/4`，`128/128` | 128 token/块，$k_1=32$，$k_2=1$ |

### 4.3 单 token FLOPs（§2.10 的模型，等效 FP8 权重 $w_{\text{BF16}}=2,w_{\text{FP8}}=1,w_{\text{FP4}}=0.5$）

每个注意力条目每头的核心 FLOPs：V4 `2*512 + 2*512 = 2048`；V3.2 `2*576 + 2*512 = 2176`；indexer 每条目每头 `2*128 = 256`。

| 分项 @ $t=2^{20}$ | V3.2 | V4-Pro | V4-Flash |
|---|---|---|---|
| 投影 $2N_{\text{act}}$ | `2*37e9` = 0.0740 T | `2*49e9` = 0.0980 T | `2*13e9` = 0.0260 T |
| CSA/稀疏核心注意力（×2） | `2*61*128*2176*2048` = 0.0696 T | `2*30*128*2048*(1024+128)` = 0.0181 T | `2*21*64*2048*(512+128)` = 0.0035 T |
| HCA 稠密核心注意力（×2） | — | `2*31*128*2048*(8192+128)` = **0.1352 T** | `2*20*64*2048*(8192+128)` = 0.0436 T |
| 纯 SWA 层（×2） | — | — | `2*2*64*2048*128` = 0.0001 T |
| indexer | FP8×1：`61*64*256*2**20` = **1.0480 T** | FP4×0.5：`0.5*30*64*256*2**18` = 0.0644 T | `0.5*21*64*256*2**18` = 0.0451 T |
| **合计** | **1.1916 T** | **0.3158 T**（26.5%，3.77×） | **0.1183 T**（9.9%，10.07×） |

- 同一模型在 $t=10^6$：V3.2 1.1430 T，Pro 0.3066 T（**26.8%，3.73×**），Flash 0.1142 T（**10.0%，10.01×**）← 与论文正文“27% / 10%”吻合；与图 1 标注只是**接近**：Pro 3.73–3.77× vs 标注 3.7×，Flash 10.0–10.07× vs 标注 **9.8×**（差约 2–3%）。所以笔者模型能复现量级与排序，但与论文内部的估算并不完全相同，课上不要说“精确复现了图 1”。
- 不加精度权重（全部 ×1）：@$2^{20}$ Pro 26.2%（3.81×）但 **Flash 12.1%（8.27×）**，与论文的 10%/9.8× 对不上——这是推断“等效 FP8”换算规则的主要证据（§4.4）。
- @128K（V3.2 官方最大上下文）：V3.2 0.2746 T，Pro 0.1429 T（**52.0%，1.92×**），Flash 0.0413 T（15.0%）。**FLOPs 优势随长度增长；KV 优势（9.55×）与长度无关**（两者都是过原点的直线）。
- 交叉点：$t\approx 911$ 之前 Pro 的单 token FLOPs 反而高于 V3.2（激活参数 49B > 37B）；之后 V4-Pro 更省。
- 按等效 FP8 口径，1M 时 Pro 注意力的最大单项是 **HCA 的稠密注意力（0.135 T）**，超过 CSA 的 indexer（0.064 T）——“重压缩但稠密”的层在极长上下文下反而成了主要注意力开销（笔者推导）；按原始 FLOPs 则 indexer 0.129 T > HCA 0.068 T，排序随口径翻转。

### 4.4 与论文图 1 的数值拟合（像素读数 → 模型）

图 1 右侧两幅子图从 `dsv4_performance.png`（2048×1109）按坐标轴像素换算读出（FLOPs 轴 0–1.2 T 对应 y=466.5→17.5 px；KV 轴 0–50 GB 对应 y=1033→584 px；横轴 0–1024 K 对应 x=1461→2038 px）：

| 曲线 @ 位置 | 读数 | 模型 | | 曲线 @ 位置 | 读数 | 模型 |
|---|---|---|---|---|---|---|
| V3.2 FLOPs @256K | 0.411 T | 0.406 T | | V3.2 KV @886K | 43.6 GB | 43.61 GB |
| V3.2 FLOPs @768K | 0.934 T | 0.930 T | | Pro KV @768K | 3.953 GB | 3.957 GB |
| V3.2 FLOPs @1010K | 1.184 T | 1.177 T | | Pro KV @886K | 4.566 GB | 4.565 GB |
| Pro FLOPs @256K | 0.169 T | 0.168 T | | Flash KV @768K | 2.728 GB | 2.764 GB |
| Pro FLOPs @513K | 0.219 T | 0.217 T | | Flash KV @1020K | 3.73 GB | 3.67 GB |
| Flash FLOPs @256K | 0.0522 T | 0.0523 T | | V3.2 FLOPs 截距（外推） | ≈0.145 T | 0.1436 T |
| Flash FLOPs @1010K | 0.1177 T | 0.1171 T | | Pro FLOPs 截距（外推） | ≈0.118 T | 0.118 T |

- 曲线在 $t$ 很小时的“竖直跳变”也被模型解释：V3.2 在 $t\le2048$ 时核心注意力随 $t$ 增长到 top-k 上限（从 0.074 跳到 0.144 T），Pro 在 $t\le4096$（$=1024\times4$）时同理。
- 由拟合得出的**笔者推测**：(a) 图 1 的“GB”是 $10^9$ 字节、“K”是 1024；(b) KV 曲线只计增长型的压缩条目（CSA 主条目 + indexer 键 + HCA 条目），不计滑窗与尾部状态、不计 MTP 层；(c) FLOPs 的“等效 FP8”按 BF16 算 2 份、FP4 算 0.5 份；V3.2 的 indexer 按 FP8、核心注意力按 BF16 计。

### 4.5 Muon 与牛顿–舒尔茨

| 量 | 表达式 | 值 |
|---|---|---|
| $\varphi_A(1)$ | `3.4445 - 4.7750 + 2.0315` | 0.7010（不收敛到 1） |
| $\varphi_A$ 的正不动点 | 解 `2.0315x^4 - 4.7750x^2 + 2.4445 = 0` | 0.868、1.264 |
| $\max_{[0,1]}\varphi_A$ | 数值 | 1.2024（在 $x=0.554$） |
| $\varphi_B(1)$，$\varphi_B'(1)$ | `2-1.5+0.5`，`2 - 4.5 + 2.5` | 1，**0**（二阶收敛） |
| 8 步 $\varphi_A$ 后 $x_0\in[10^{-3},1]$ 的范围 | 数值扫描 2000 点 | [0.682, 1.134] |
| 再 2 步 $\varphi_B$ 后 | 同上 | **[1.000000, 1.000538]** |
| Jordan 原版 5 步 $\varphi_A$ 后 | 对随机 256×64 矩阵（Frobenius 归一化后奇异值约在 0.06–0.19） | 奇异值 ∈ [0.682, 1.134]（“近似正交”，误差 ~0.3）。**注意 5 步够不够取决于初始奇异值**：[0.682, 1.134] 是 $\varphi_A$ 的不变区间（该区间在 $\varphi_A$ 下的像仍是它自己，核查者复算），奇异值一旦进入就只在其中振荡、不再收敛到 1。这个随机矩阵的奇异值 5 步内已进入；但对 $x_0\in[10^{-3},1]$ 扫描，5 步后是 [0.471, 1.202]，6 步后 [0.682, 1.202]，要 7 步才全部进入 [0.682, 1.134]。别写成“Jordan 5 步总落在 [0.68, 1.13]” |
| V4 混合 10 步后 | 同一随机矩阵 | 奇异值 ∈ [1.00000, 1.00054] |
| 轨迹 $x_0=10^{-3}$ | 逐步 | 0.001, 0.0034, 0.0119, 0.0409, 0.1404, 0.4705, 1.1702, 0.8368, 0.9179 ‖ 1.0015, 1.0000 |
| 轨迹 $x_0=0.05$ | 逐步 | 0.05, 0.1716, 0.5673, 1.2016, 0.9436, 0.7581, 1.0395, 0.6828, 1.1334 ‖ 1.0180, 1.0002 |
| 轨迹 $x_0=1$ | 逐步 | 1, 0.701, 1.1136, 0.7207, 1.0900, 0.6964, 1.1188, 0.7277, 1.0810 ‖ 1.0053, 1.0000 |
| $x_0=10^{-4}$ | 8A+2B | 8A 后 1.2023，最终 1.0020（极小奇异值仍能被拉上来） |
| 满秩 $p\times q$ 正交化矩阵的 RMS | `sqrt(min(p,q)/(p*q))` = `1/sqrt(max(p,q))` | 7168×1536：0.01181；1536×65536：0.00391 |
| 乘 $\sqrt{\max}\cdot0.18$ 后 | 任意形状 | **0.180**（与形状无关——这就是缩放的目的） |
| NS 每步 FLOPs（7168×1536） | `2*(2*7168*1536**2 + 1536**3)` | 7.49e10；10 步 7.49e11 |

### 4.6 训练规模与日程

| 量 | 表达式 | 值 |
|---|---|---|
| Flash 训练 FLOPs（$6ND$ 近似，不含注意力） | `6*13e9*32e12` | 2.50e24 |
| Pro 训练 FLOPs | `6*49e9*33e12` | 9.70e24 |
| V3 对照 | `6*37e9*14.8e12` | 3.29e24（Pro ≈ 2.95× V3） |
| batch 分解 | `18432*4096`；`23040*4096` | 75,497,472（=75.5M）；94,371,840（=94.4M）——恰为 4K 序列的整数倍（笔者观察） |
| 若全程最大 batch 的步数下界 | `32e12/75497472`；`33e12/94371840` | Flash ≈ 423,855 步；Pro ≈ 349,681 步（实际有 batch 递增阶段，步数更多） |
| 学习率衰减倍数 | `2.7e-4/2.7e-5` | 10× |
| YaRN 目标长度 | `16*65536` | 1,048,576 |

### 4.7 基础设施数字

| 量 | 表达式 | 值 |
|---|---|---|
| Pro 每 token-专家对 FLOPs | `6*7168*3072` | 1.321e8 |
| Pro 每 token-专家对通信 | `3*7168` | 21,504 B |
| 平衡点 | `2*3072`；`2*2048` | Pro 6144 FLOPs/B；Flash 4096 FLOPs/B |
| 每 GB/s 可掩盖的算力 | `1e9*6144` | 6.144 TFLOP/s（论文 “6.1”） |
| mHC 小矩阵乘输出维 | `(2+4)*4` | 24 |
| 1M 时 CSA 条目数 / HCA 条目数 | `2**20/4`；`2**20/128` | 262,144；8,192 |
| top-k 占可见条目比例 @1M | `1024/262144`；`512/262144` | Pro 0.391%；Flash 0.195% |
| 每查询真正读到的原始 token 跨度 | `1024*4 + 128`；`512*4 + 128` | Pro 4,224；Flash 2,176（V3.2 为 2,048） |
| index score 每查询内存 @1M | `4*262144` vs `2*262144` | FP32 1 MiB → BF16 512 KiB |
| FP4×FP8 相对 FP8×FP8 的操作数位宽 | `(8+8)/(4+8)` | 1.333（论文“1/3 更高效”的可能来由） |

### 4.8 训练稳定性、后训练与评测里的小数字

| 量 | 值 | 来源 |
|---|---|---|
| Anticipatory Routing 开启时的墙钟开销 | ≈20% | §4.2.3 |
| mHC 墙钟开销 | 重叠 1F1B 流水段的 6.7% | §3.4.2 |
| Muon ZeRO padding 内存开销 | <10%（每 rank ≤5 个矩阵） | §3.4.1 |
| EP 实测加速 | 1.50–1.73×（一般推理）；≤1.96×（RL rollout 等） | §3.1 |
| EP 理论加速（图 5） | Comet 1.42×；V4 1.92× | 图 5（SVG 渲染读数） |
| Host Codegen | 每次调用 CPU 检查：数十–数百 μs → <1 μs | §3.2 |
| index score BF16 | top-k 2× 提速，召回 99.7% | §5.2.1 |
| Codeforces | 14 场、114 题；32 候选/题、提交 10 个；Pro-Max 3206、人类榜第 23 | §5.3.1–5.3.2 |
| 开发者调查（N=85） | 52% 愿意作为主力编码模型，39% 倾向愿意，<9% 不愿意 | §5.4.4 |
| 内部 R&D 编码基准（30 题）通过率 | Haiku 4.5 13、Sonnet 4.5 47、V4-Pro-Max 67、Opus 4.5 70、Opus 4.5 Thinking 73、Opus 4.6 Thinking 80 | 表 8 |
| agentic search vs RAG（869 题） | agent 胜 61.7%、RAG 胜 18.3%、平 20.0%；平均工具调用 16.2 次，prefill 13,649 vs 10,453 token，输出 1,526 vs 1,308 | 表 9、表 10 |

### 4.9 FP4→FP8 无损反量化的阈值（笔者推导，论文只说“某个阈值”）

E2M1 的非零幅值为 $\{0.5,1,1.5,2,3,4,6\}$；E4M3 最大 448、最小次正规数 $2^{-9}$，尾数 3 位。若 FP8 块缩放与 FP4 子块缩放都是 2 的幂（config `scale_fmt: ue8m0`），子块相对块缩放的指数差 $e$ 必须让全部 7 个幅值乘 $2^e$ 后都能被 E4M3 精确表示。穷举得 $e\in[-8,6]$（`6*2**6=384 ≤ 448`，`6*2**7=768 > 448`；`0.5*2**-8 = 2**-9` 恰为最小次正规数），跨度 14，即**同一 128×128 块内各 1×32 子块缩放因子的最大/最小比 ≤ $2^{14}=16384$** 时反量化无损。

---

## 5. 图表清单与重绘建议

### 5.1 论文的图表清单（v1）

| 编号 | 标题大意 | 内容 | 本课是否重绘 |
|---|---|---|---|
| 图 1 | 左：Pro-Max 与对手的基准柱状图；右：单 token FLOPs 与累计 KV cache 随长度变化 | 右侧两幅是本课“算总账”的核心；标注 3.7×/9.8× lower、9.5×/13.7× smaller | **重绘右侧（按 §4.3 模型计算）** |
| 图 2 | V4 整体架构 | Embedding → 扩宽的残差流 →（Pre-Block Mixing → CSA/HCA → Post-Block Mixing，⊕ Residual Mixing）→（同样包住 DeepSeekMoE）×L → 预测头（LM loss）→ MTP 模块（MTP loss） | **重绘**（加真实维度） |
| 图 3 | CSA 核心结构 | KV token 隐状态 → token 级压缩器 → 压缩 KV 条目 → top-k 选择器（由 lightning indexer：压缩 indexer 键 + indexer 查询 → MQA 打分 → index scores 驱动）→ 与滑窗 KV 条目拼接 → 共享 KV 的 MQA；查询 token 隐状态分出 indexer queries 与 queries | **重绘** |
| 图 4 | HCA 核心结构 | 同图 3 但无 indexer、压缩率 $m'$ | **重绘**（可与图 3 并排） |
| 图 5 | EP 方案对比 | (a) 朴素串行 Dispatch→L1→Act→L2→Combine；(b) Comet 两段重叠（理论 1.42×）；(c) V4 按专家“波”细粒度流水（理论 1.92×，Flash 配置） | 可选（时间轴示意） |
| 图 6 | KV cache 布局 | 左：状态 cache（每请求：各层 SWA KV + 各 CSA/HCA 层的未压缩尾部状态）；右：经典 KV cache（每块：Layer-2 CSA indexer KV 与主 KV 各 $k_1$ 个，Layer-3 HCA KV $k_2$ 个……） | **重绘**（填入 $k_1=32,k_2=1$） |
| 图 7 | 思考内容管理 | (a) 带工具：所有轮次的 thinking 都保留进后续输入；(b) 不带工具：新用户消息到来时丢弃旧 thinking | 可选 |
| 图 8 | 形式化数学 | 左：Putnam-200 Pass@8；右：Putnam-2025 120/120 | 否 |
| 图 9 | MRCR 8-needle 随长度 | Pro-Max 与 Flash-Max 的 8K–1024K 曲线（数据见 C74） | **重绘**（小图） |
| 图 10 | HLE / Terminal Bench 2.0 随总 token 的推理强度曲线 | V4-Pro、V4-Flash 的 None/High/Max 与 V3.2（Think、Speciale） | 可选（坐标需目测，见 UNVERIFIED） |
| 图 11–15 | 白领任务胜率、维度评分、示例输出 | 定性 | 否 |
| 表 1 | 基座对比 | 见 §3.8 | 摘表 |
| 表 2–3 | 三档推理模式、Think Max 注入指令 | 见 C60 | 摘表 |
| 表 4 | 工具调用格式（DSML） | 见 C62 | 否 |
| 表 5 | Quick Instruction 特殊 token | action、title、query、authority、domain、extracted_url + read_url 六类特殊 token（原文写作尖括号加竖线的形式，放进表格会破坏 Markdown 表格，故省略符号） | 可选（表格） |
| 表 6–7 | 与前沿模型对比 / 三档对比 | 见 §3.8 | 摘表 |
| 表 8–14 | 内部 R&D 编码、搜索、中文写作 | 见 §4.8 | 否 |

### 5.2 本课建议的图（13 张，每张给出可直接计算几何的数据）

**F1 V4 一层的结构（据论文图 2 重绘，加维度）**——机制图。
自下而上：token id → embedding（129,280×7168）→ 复制成 4 条残差流 $X_l\in\R^{4\times7168}$；
注意力子层：$\hat X=\operatorname{RMSNorm}(\operatorname{vec}X_l)\in\R^{1\times28672}$ → 24 个系数 → $A_l$（1×4，σ）、$C_l$（4×1，2σ）、$B_l$（4×4，Sinkhorn 20 次）；
$A_lX_l\in\R^{7168}$ → RMSNorm → CSA 或 HCA → $C_l\otimes$输出 + $B_lX_l$ → MoE 子层同样结构（1 共享 + 384 路由选 6，中间维 3072）→ ×61 → hc_head 汇聚 → 预测头；MTP 深度 1。颜色：残差流 `--fig-ink`，路由专家 `--fig-blue`，共享专家 `--fig-violet`。

**F2 层类型条带图**——账单/结构图。一行一个模型，每个小格一层：
Pro（61 + MTP）：`H H C H C H … C | M`（层 0–1 HCA，层 2、4、…、60 CSA，层 3、5、…、59 HCA，MTP 为 SWA）；
Flash（43 + MTP）：`S S C H C H … C | M`（层 0–1 SWA，偶数层 2–42 CSA，奇数层 3–41 HCA）。
颜色：CSA `--fig-blue`，HCA `--fig-orange`（压缩表示），SWA `--fig-muted`。标注计数：Pro 30C/31H；Flash 21C/20H/2S。

**F3 CSA 的重叠压缩（按真实 $m=4$）**——机制图。横轴 token 0–19；条目 $i$ 的读取窗 = token $[4(i-1),4i+3]$：
条目 0：token 0–3（$a$ 路）+ 填充（$b$ 路，权重 $-\infty$）；条目 1：token 0–3（$b$）+ 4–7（$a$）；条目 2：4–7（$b$）+ 8–11（$a$）……
每条目下画“逐通道 softmax（8 个权重 × 512 通道）”的小热图示意；RoPE 位置标在块首 $4i$。颜色：压缩条目 `--fig-orange`，原始 KV `--fig-green`/`--fig-violet`。

**F4 CSA 数据流（据论文图 3 重绘，Pro 维度）**——机制图。
查询侧：$\mathbf h_t$(7168) → $W^{DQ}$ → $\mathbf c^Q_t$(1536) → RMSNorm → ① $W^{UQ}$ → 128 头 × 512 → 每头 RMSNorm + RoPE(最后 64 维)；② $W^{IUQ}$ → 64 × 128（FP4）；$\mathbf h_t\to W^w\to$ 64 个头权重。
KV 侧：压缩器 → 条目 512 维（448 FP8 + 64 BF16，584 B）；indexer 压缩器 → 128 维键（FP4，68 B）。
选择：$I_{t,s}$ over $\lfloor(t+1)/4\rfloor$ 个条目 → top-1024 → ∪ 128 个滑窗条目 → 共享 KV MQA（+ sink）→ 128×512 → 逆 RoPE → 16 组 × (4096→1024) → 16384 → 7168。

**F5 HCA 数据流（据论文图 4 重绘）**——与 F4 并排，去掉 indexer，压缩器为 128→1 不重叠，全部 $\lfloor(t+1)/128\rfloor$ 条目 + 128 滑窗稠密注意力。

**F6 一个查询到底看见了什么（$t=1000$ 的玩具例子）**——机制图。
0-based 位置 $t=1000$：SWA 覆盖 873–1000（128 个）；CSA 可见条目 $s<\lfloor1001/4\rfloor=250$（覆盖 token 0–999，其中条目 249 读 992–999）；若 $k_{\text{sel}}=1024$ 则 250 个全选（$t<4096$ 时 CSA 等于“压缩后的稠密”）；HCA 可见 $\lfloor1001/128\rfloor=7$ 个块（token 0–895），块 7（896–1023）尚未完成 → token 896–1000 只能靠 SWA（873–1000）补上，**896 之前、873 之后的空隙不存在**（873<896），说明 $n_{\text{win}}=128=m'$ 恰好无缝衔接（笔者观察：块内最多 127 个未完成 token，窗口 128 恰好覆盖）。
画三条横带：原始 token 轴、CSA 条目（每 4 个一格）、HCA 块（每 128 个一格），用 `--fig-red` 标“不可见”。

**F7 部分 RoPE 与输出逆旋转**——机制图。512 维条目 = 448 NoPE（FP8，`--fig-orange`）+ 64 RoPE（BF16，`--fig-red` 细线）；查询在位置 $t$ 旋转，条目在块首位置 $p_s$ 旋转，打分只剩 $R_{p_s-t}$；输出乘 $R_{-t}$ 后每个条目的贡献带 $R_{p_s-t}$。配 §2.6 的两行推导。

**F8 KV 字节/token 阶梯（对数轴，计算得出）**——账单图。数据（B/token）：
BF16 GQA8 头维 128（61 层）249,856；V3 MLA 纯 BF16（576×2×61）70,272；V3.2 FP8 MLA + indexer 48,068；V4-Pro 5,031.44；V4-Flash 3,514.25。
可叠加分项：V3.2 = 主 40,016 + indexer 8,052；Pro = CSA 主 4,380 + CSA indexer 510 + HCA 141.44；Flash = 3,066 + 357 + 91.25。

**F9 累计 KV cache vs 长度（重绘图 1 右下，计算得出）**——账单图。$x$ = 0–1024K（K=1024），$y$ = GB（$10^9$）：
直线斜率（GB/K token）：V3.2 `48068*1024/1e9 = 0.049222`；Pro `0.0051522`；Flash `0.0035986`。端点 @1024K：50.40、5.276、3.685 GB。标 9.55×、13.68×。另画一条虚线“BF16 GQA8（61 层）”：斜率 0.25585 GB/K，@1024K 262.0 GB（需断轴或对数轴）。

**F10 单 token FLOPs vs 位置（重绘图 1 右上，计算得出）**——账单图。用 §4.3 的表（等效 FP8，T=$10^{12}$）：

| t(K) | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 768 | 1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V3.2 | 0.0919 | 0.1098 | 0.1456 | 0.1477 | 0.1518 | 0.1600 | 0.1763 | 0.2091 | 0.2746 | 0.4056 | 0.6676 | 0.9296 | 1.1916 |
| Pro | 0.1042 | 0.1063 | 0.1105 | 0.1190 | 0.1197 | 0.1213 | 0.1244 | 0.1305 | 0.1429 | 0.1676 | 0.2170 | 0.2664 | 0.3158 |
| Flash | 0.0282 | 0.0289 | 0.0304 | 0.0306 | 0.0309 | 0.0316 | 0.0330 | 0.0358 | 0.0413 | 0.0523 | 0.0743 | 0.0963 | 0.1183 |

旁边配 @1M 的分项堆叠柱（见 §4.3 表）：V3.2 = 投影 0.074 + 核心 0.070 + indexer 1.048；Pro = 0.098 + CSA 0.018 + HCA 0.135 + indexer 0.064；Flash = 0.026 + 0.0035 + 0.0437 + 0.0451。图注必须写“笔者按公开架构估算，等效 FP8 换算为笔者推测”。

**F11 牛顿–舒尔茨多项式与混合迭代（计算得出）**——机制/账单图。
左：$x\in[0,1.2]$ 上画 $y=\varphi_A(x)=3.4445x-4.7750x^3+2.0315x^5$、$y=\varphi_B(x)=2x-1.5x^3+0.5x^5$、$y=x$；标 $\varphi_A(1)=0.701$、$\max\varphi_A=1.2024@0.554$、$\varphi_B(1)=1$ 且切线水平。
右：奇异值轨迹（步 0–10），用 §4.5 的三条数据（$x_0=10^{-3},0.05,1$），第 8 步后换色（`--fig-green`）表示切换到 $\varphi_B$；阴影带 [0.7,1.3]。

**F12 Muon vs AdamW 的分工 + update RMS**——机制图。左列 AdamW：embedding、预测头、RMSNorm、mHC 的 $S$ 与 $\alpha$；右列 Muon：其余所有矩阵。底部小条：不同形状（7168×1536、1536×65536、3072×7168）正交化后 RMS 为 0.0118 / 0.0039 / 0.0118，乘 $\sqrt{\max}\cdot0.18$ 后都等于 0.18。

**F13 后训练流水线**——机制图。V4-Base（FP8）→ 各领域专家（数学、代码、agent、指令遵循……；每个：SFT → GRPO，奖励来自规则/测试或 GRM；同一领域按 Non-think/High/Max 用不同长度惩罚与上下文训练）→ 多教师 OPD（>10 个教师，学生采样轨迹，全词表反向 KL，式 (29)）→ 统一模型；旁注 FP4 QAT（专家权重 + indexer QK）贯穿后训练，rollout 用原生 FP4。

**F14（可选）KV cache 布局（据图 6 重绘）**：每请求一块状态 cache（SWA：Pro 61 层 × 128 条目 × 584 B = 4.56 MB；尾部状态）；经典 cache 每块 128 个原始 token → 每个 CSA 层 32 个主条目 + 32 个 indexer 键，每个 HCA 层 1 个条目。

**F15（可选）发布时间线**：2025-12-01 V3.2 → 2026-04-24 V4 Preview（04-26 arXiv）→ 06-27 DSpark → 07-31 Flash GA → 08-13 Pro GA → 08-16 新价格 → 08-21 Vision-Exp → 09-10 V4.1-Flash（V4-Flash 退役）。

---

## 6. 算例与练习素材（带答案）

1. **（计算）V4-Pro 每 token 的 KV 字节。** 已知 30 个 CSA 层、31 个 HCA 层，主条目 584 B、indexer 键 68 B，$m=4$、$m'=128$。
   答：$30\times(584+68)/4+31\times584/128=4890+141.44=5031.44$ B；1M（$2^{20}$）token 为 5.276 GB。
2. **（计算）与 V3.2 和 GQA8 比。** V3.2 每层 656 B 主条目 + 132 B indexer 键、61 层。
   答：V3.2 = 48,068 B/token；比值 48,068/5,031.44 = 9.55×（10.5%）；GQA8 BF16 头维 128、61 层为 249,856 B，V4-Pro 为其 2.01%。
3. **（概念+计算）为什么 KV 的比值与长度无关，而 FLOPs 的比值随长度变好？**
   答：两者的 KV 都 ∝ $t$（过原点直线），比值恒为 9.55×；FLOPs 有常数项（投影 $2N_{\text{act}}$：Pro 0.098 T > V3.2 0.074 T）和线性项（V3.2 indexer 每 token 1.0 M FLOPs/上下文 token vs Pro 约 0.19 M），所以短上下文时 Pro 甚至更贵（$t<911$），128K 时 52%，1M 时 26.5%。
4. **（计算）CSA 的稀疏度。** 1M 上下文、Pro 的 top-1024。
   答：可见条目 $2^{20}/4=262{,}144$，选中比例 0.39%；每个条目概括 4 个 token（读 8 个），故读到的原始跨度约 4,096 token + 128 滑窗 = 4,224；V3.2 为 2,048 个 token。
5. **（计算）HCA 在 1M 时的稠密注意力 FLOPs（原始 FLOPs，不加权）。**
   答：每层每头每条目 $2\cdot512+2\cdot512=2048$；条目 $2^{20}/128+128=8320$；Pro：$31\times128\times2048\times8320=6.76\times10^{10}$。原始 FLOPs 下它小于 CSA indexer 的 $30\times64\times256\times2^{18}=1.29\times10^{11}$；但按等效 FP8（HCA 核心注意力在 BF16 记 ×2、indexer 在 FP4 记 ×0.5）则是 0.135 T 对 0.064 T，HCA 反成最大的注意力项——结论依赖口径，这正是 E2 要提醒的。
6. **（计算）混合精度条目为什么“近一半”。**
   答：纯 BF16 512 维 = 1024 B；V4 = 448（FP8）+ 128（64 维 BF16）+ 8（7 个 UE8M0 缩放 + 1 字节 padding）= 584 B，为 57.0%，省 43%。
7. **（推导）部分 RoPE 输出逆旋转为何得到相对位置。**
   答：见 §2.6：$R_{-t}\sum_s a_sR_{p_s}\mathbf e^r_s=\sum_s a_sR_{p_s-t}\mathbf e^r_s$，并且打分 $\mathbf q^{r\top}R_t^\top R_{p_s}\mathbf e^r_s=\mathbf q^{r\top}R_{p_s-t}\mathbf e^r_s$；用到 $R_a^\top=R_{-a}$、$R_aR_b=R_{a+b}$。
8. **（计算）Muon 的 update RMS 缩放。** 形状 7168×1536 的矩阵正交化后 $O'=UV^\top$。
   答：$\lVert UV^\top\rVert_F^2=\min(p,q)=1536$，RMS $=\sqrt{1536/(7168\cdot1536)}=1/\sqrt{7168}=0.01181$；乘 $\sqrt{7168}\times0.18$ 得 0.18。对 1536×65536 同样得 0.18——缩放抵消了形状的影响（Liu et al. Lemma 1）。
9. **（推导+计算）为什么 NS 最后要换系数。**
   答：$\varphi_A(1)=0.701$、正不动点 0.868 与 1.264，迭代只会在约 [0.68,1.13] 里振荡；$\varphi_B(1)=1$ 且 $\varphi_B'(1)=0$，在 1 附近二阶收敛：8 步 $\varphi_A$ 后再 2 步 $\varphi_B$，$x_0\in[10^{-3},1]$ 全部落在 [1.0000, 1.0005]。
10. **（计算）EP 的带宽平衡点。** 某加速卡稠密 FP8 算力 1,000 TFLOP/s（假设值），跑 V4-Pro 的 MoE。
    答：需 $B\ge C/(2d)=10^{15}/6144\approx1.63\times10^{11}$ B/s ≈ 163 GB/s；超过它再加带宽收益递减。（Flash 的 $2d=4096$，需 ≈ 244 GB/s——专家越“瘦”越吃带宽。）
11. **（计算）FP4→FP8 反量化的无损条件。**
    答：见 §4.9：子块缩放比 ≤ $2^{14}$；关键两端 $6\cdot2^6=384\le448$ 与 $0.5\cdot2^{-8}=2^{-9}$。
12. **（概念）从头重生成被中断的 rollout 为什么有偏。**
    答：短回答在中断前完成的概率更高，“幸存”样本的长度分布偏短；把没完成的丢掉重来，相当于按长度做了有偏的拒绝采样（§5.2.3）。V4 用逐 token 的 WAL 续写，或在批不变+确定性的推理栈上用同一随机种子重放。
13. **（计算，备用）参数量核对。** Flash：每专家 $3\times4096\times2048=25.17$M，256 专家 × 43 层 = 277.0B；加上注意力 5.09B、embedding+head 1.06B 等，主干 284.3B。

---

## 7. 勘误与常见误解

**E1. “V4 只要 V3.2 的 27% 算力和 10% KV”——必须带上三个限定。** (1) 只在 **1M 上下文**、只指 **V4-Pro**；(2) FLOPs 是“等效 FP8 FLOPs”、且是**估算**（§1）；(3) V3.2 官方只支持 128K（`dsv32` §4 明说），1M 处的 V3.2 是按架构外推的假想点。按 §4.3 的模型，128K 时 Pro 的 FLOPs 约为 V3.2 的 52%，而 KV 比例按图 1 口径（只计随长度增长的压缩条目）在任何长度都是约 10.5%。若把每请求约 4.56 MB 的滑窗状态也算进去，短上下文时比例会高一些：4K 时 12.8%，32K 时 10.8%，128K 以上约 10.5%（核查者复算：`(5031.44*t + 61*128*584)/(48068*t)`）。

**E2. 图 1 的 FLOPs 口径不是原始 FLOPs。** 论文 §1 明说是“equivalent FP8 FLOPs”。按笔者模型，原始 FLOPs 下 Flash 只省到 12.1%（8.27×），离论文的 10%/9.8× 较远；按 BF16×2、FP8×1、FP4×0.5 的“等效 FP8”换算，三条曲线的斜率和截距都能近似拟合（§4.4；Flash 比值 10.07× vs 标注 9.8×，仍差约 3%）。这个换算规则是笔者从数据反推的**一种**可行解释，论文未写，也没有证明它是唯一的，课上应标“笔者推测”。

**E3. 图 1 的 KV 只算“随长度增长的部分”。** 拟合显示它等于 CSA 主条目 + indexer 键 + HCA 条目，不含每请求 4.56 MB（Pro）的滑窗状态和尾部状态；单位 GB=$10^9$ 字节、1M=$2^{20}$ token。

**E4. CSA 的压缩不是“每 4 个 token 取平均”。** (1) 权重是 softmax 出来的、依输入而变（$Z=HW^Z$）；(2) 是**逐通道**的 softmax（每个通道各有一组 8 个权重）；(3) 每个条目读 8 个 token（当前块 + 前一块，两套投影 $a$/$b$），步长 4；(4) 还有可学习的块内位置偏置。压缩率仍是 1/4。

**E5. HCA 不是稀疏注意力。** 它对全部 $t/128$ 个压缩块做**稠密**注意力；在 1M 时按等效 FP8 口径，HCA 的核心注意力（Pro 31 层合计 0.135 T）是 V4-Pro 最大的注意力开销项，超过 CSA 的 indexer（0.064 T）；按原始 FLOPs 则 indexer（0.129 T）大于 HCA（0.068 T）。

**E6. V4 的注意力不是 V2/V3 意义上的 MLA。** 没有 $W^{UK}/W^{UV}$ 上投影、没有“矩阵吸收”、没有单独的解耦 RoPE 键：512 维条目本身同时当 K 和 V（共享 KV 的 MQA），RoPE 放在同一向量的最后 64 维。与 MLA 共享的只有“查询低秩压缩”和“解码时 MQA 形态”。注意 `model.py` 的类注释仍写 “Multi-head Latent Attention (MLA) with sliding window”——代码注释沿用旧名，别据此说 V4 用 MLA。FlashMLA README 把 V4 归为 MQA 模式、`head_dim_k=512`（V3 系列是 576）。

**E7. QK-Clip 不是 Liu et al. 2025（2502.16982）提出的。** V4 §2.4 在说明不用 QK-Clip 时引用 “(Liu et al., 2025)”；但 2502.16982v1 全文没有 “QK-Clip”/clip 机制（只在附录讨论了最大注意力 logit 的增长），QK-Clip 出自 **Kimi K2 技术报告（2507.20534）** 的 MuonClip。课上引用 QK-Clip 请用 `kimi-k2`。

**E8. 输出逆 RoPE 的“位置 $-i$”应理解为“位置 $-t$”。** V4 §2.3.3 写对每个 $\mathbf o_{t,i}$ 施加位置 $-i$ 的 RoPE，而 $i$ 在式 (19) 中是**头下标**；参考代码 `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` 用的是查询 token 的位置（`freqs_cis[start_pos:start_pos+seqlen]` 的共轭）。按 §2.6 的推导，只有用 $-t$ 才得到相对位置。另注意同一小节的式 (27) 把 $i$ 用作**查询 token** 下标，所以“$-i$”更像是记号混用（若按式 (27) 的记号读，$-i$ 就是查询位置，与代码一致），而不是公式本身错了。FlashMLA README 也把这一步称为 “O-RoPE (conjugate)”。判断：记号混用（笔者判断）；课上统一写 $R_{-t}$。

**E9. 压缩块可见性的差一。** 论文 §2.3.1 写 $s<\operatorname{Floor}(t/m)$；参考代码（0-based $t$）是 $s<\lfloor(t+1)/m\rfloor$，即“以当前 token 结尾的那一块”也可见。两者都满足因果；可能只是 $t$ 从 1 还是从 0 计数的差别。画 F6 时按代码。

**E10. V4 的 Muon update RMS 是 0.18，不是 Liu et al. 的 0.2。** 两者都是把正交化后的更新乘 $\sqrt{\max(\text{行},\text{列})}$ 再乘常数；V4 算法 1 把权重衰减写成 $W(1-\eta\lambda)$ 的乘法形式，与 Liu et al. 的 $-\eta\lambda W$ 等价。

**E11. NS 的“5 步 + (3.4445,−4.7750,2.0315)”不会把奇异值精确推到 1。** Jordan 的设计本就只要求落在约 [0.7,1.3]；V4 的改动是 8 步快系数 + 2 步 $(2,-1.5,0.5)$ 精修，使奇异值误差 <$6\times10^{-4}$（§4.5）。$(2,-1.5,0.5)$ 正是 Jordan 博客里的“基线系数”。

**E12. “Pro-Max”不是另一个模型。** 它是 V4-Pro 的最大推理强度档（Think Max）；Flash-Max 同理。GA 后 API 的档位名是 low/high/max。

**E13. V4 最后一步不是 RL。** 后训练的“混合 RL 阶段被整体换成 OPD”；GRPO 只出现在各领域专家的训练里。OPD 用的是全词表的反向 KL（不是 token 级估计当优势）。

**E14. AdamW 清单以 §2.4 为准。** §4.2.2 两次列 AdamW 参数时漏了 mHC 的静态偏置与门控因子；§2.4 写了。另外 attention sink、压缩器位置偏置 `ape` 归哪个优化器论文没说。

**E15. §3.1 的 $d$ 不是隐藏维。** “$6hd$ FLOPs、$3h$ 字节 → $C/B\le2d=6144$” 里 $h$=隐藏维 7168、$d$=专家中间维 3072。直接套 §2 的 $d=7168$ 会得到错的 14336。

**E16. “词表 128K” = 129,280 个条目。** 与 V3 相同（config `vocab_size`），128K 是约数（$128\times1024=131{,}072$）。

**E17. “1.6T / 284B” 的口径。** 按 config 复算，284.3B 恰为不含 MTP 的主干；Pro 不含 MTP 1.573T、含 MTP 1.599T，“1.6T”对两者都算四舍五入。别写成“恰好 1.6T”。

**E18. Hash 路由层也是 MoE 层。** V4 去掉了 V3 开头的稠密 FFN，前 3 层是“按 token ID 查表选 6 个专家”的 MoE（仍有共享专家与门控加权），不是稠密层。

**E19. 发布的 Base 模型没有 FP4。** FP4 专家权重来自后训练阶段的 QAT；Base 的 `expert_dtype` 是 `fp8`，模型卡精度栏写 “FP8 Mixed”。

**E20. “MRCR 1M = 83.5” 不是“在 1M 长度上 83.5 分”。** 图 9 的 MRCR 8-needle 在 1024K 处 Pro-Max 只有 0.59（Flash-Max 0.49），128K 以内约 0.84–0.94（Flash-Max 在 16K 处为 0.84）；表 6/7 的 “MRCR 1M (MMR)” 显然是另一种汇总口径（论文未定义，见 UNVERIFIED）。它也不是图 9 八个点的简单平均：Pro-Max 平均 0.8225、Flash-Max 平均 0.774，都不等于 83.5 / 78.7（核查者复算）。图 9 纵轴标的是 “Average MMR”。

**E21. 技术报告 PDF 的旧链接已失效。** 2026-04-24 新闻页链接的 HF `DeepSeek_V4.pdf` 已 404；引用 arXiv 2606.19348v1。

**E22. 模型卡之间的第三方分数不一致（telephone 风险）。** Opus-4.8 的 Cybergym：Flash-0731 卡写 83.1，Pro-0813 卡与 Vision-Exp 卡写 78.3（0813 卡里 83.1 属于另一个模型那一列）。凡引用 GA 卡里的对手分数，以最新的卡为准并注明来源。

**E23. mHC 代码存的是 $B_l^\top$，但与论文等价。** `hc_post` 计算 $\mathbf y_k=\sum_j \texttt{comb}[j,k]\mathbf x_j$，Sinkhorn 以列归一化收尾（论文以行归一化收尾）。这两处“差异”互相抵消：把 `comb` 当作 $B_l^\top$，代码的迭代就是论文式 (8) 作用在转置矩阵上，前向结果除 eps 外完全相同（见 §2.1 代码对照第 5 条，核查者推导）。所以别把它当成“代码与论文不一致”来讲，只需在逐行对照时注明 `comb[j,k]` ↔ $B_l[k,j]$。

**E24. V4 参考文献的小瑕疵。** mHC 在 V4 参考文献里标作 “Xie et al. (2026)”，arXiv v1 实为 2025-12-31；Engram 条目的作者列表与 arXiv v2 不同（V4 列 14 人，arXiv v2 更多）。引用时用 arXiv 元数据。

---

## 8. 未能核实 / UNVERIFIED

1. **预训练的数值精度**：论文没有明确写“V4 预训练用 FP8 混合精度”。间接证据：§2 说未说明的细节沿用 V3；§5.2.1 说 QAT “复用现有 FP8 训练框架”；Base 权重以 FP8 发布。写课时只能写“沿用 V3 的 FP8 训练框架（论文间接表述）”。
2. **训练成本、GPU 小时数、训练硬件型号**：论文全文没有（grep `GPU hours|H800|B200` 无命中；只说 EP 方案在 NVIDIA GPU 与华为昇腾 NPU 上验证过）。不得写任何成本数字。
3. **“等效 FP8 FLOPs”的换算规则**（BF16×2、FP4×0.5）：笔者从图 1 反推（§4.4），论文未写。
4. **indexer 键缓存的生产格式**：68 B/条目（FP4 64 B + 4 个 UE8M0 缩放）来自参考代码与图 1 拟合；vLLM/FlashMLA 的实际布局（是否有 padding）未核对。参考代码注释说它的实现用 BF16 存储、只做量化模拟。
5. **图 1 是否计入 MTP 层、滑窗与尾部状态**：拟合显示不计（KV 与 61/43 个主干层吻合），但论文未说明。
6. **Anticipatory Routing 的 $\Delta t$**、触发阈值、“一段时间”的长度：未给。
7. **稠密→稀疏的具体 token 数**：Flash 前 1T 稠密；Pro “更长”但未给数字；各长度阶段（4K/16K/64K/1M）各训多少 token 未给；batch 从多小开始递增未给。
8. **Muon 对 attention sink、压缩器位置偏置 `ape`、路由器矩阵、MTP 模块的处理**：未说明（§3.4 中标为推测）。
9. **Think Max 注入指令的全文**只在 MIT 许可的 `encoding/README.md` 与论文表 3 中出现；本笔记只摘了首句。各档 RL 的长度惩罚形式与数值未给。
10. **GRM 的训练细节**（rubric 数据规模、对 GRM 做 RL 的奖励来源）：只有定性描述。
11. **OPD 的教师数量、各教师权重 $w_i$ 的取值**：只说“10 个以上”“按相对重要性”。
12. **Thinking Machines 的 “On-policy distillation” 博客（Lu & Lab 2025）**：未抓取核对；只核对了 MiniLLM（2306.08543）的元数据。
13. **TileLang 论文（Wang et al., ICLR 2026）**：未核对其 arXiv 号与内容。
14. **DSpark / DeepSpec**：只核对了 config 字段与模型卡的一句话说明；投机解码机制、接受率、`dspark_markov_rank` 的含义都未核对（多出的两个 `0` 层是 DSpark 模块——笔者推测）。
15. **GA 版（0731/0813）是否改动了除 DSpark 以外的任何权重结构**：config 只差 DSpark 字段；change log 说 Flash-0731 “只重新做了后训练”；0813 卡说“建立在 Preview 结构上”。GA 的后训练流程是否仍是“专家 + OPD”未见说明。
16. **图 10 的横坐标（总 token 数）**：只能目测（如 Pro High ≈35K、Max ≈53K；Flash Max ≈86K；V3.2 Think ≈21K），不得当精确数字引用。
17. **MRCR “1M” 与 CorpusQA “1M” 的汇总口径**（E20）：论文未定义。
18. **1M 上下文的服务端实际配置**（例如是否所有请求都真用满 1M、SWA 磁盘策略线上选哪种）：未给。
19. **§3.1 的实测加速“相对强的非融合基线”具体是哪个基线**：未给。
20. **Muon-liu 作者机构**：未在本次核对。
21. **V4 的 KV 条目为何在 RoPE 前后都有归一化、`q` 的 RMSNorm 为何不带权重**：代码如此，论文未解释动机。

---

## 9. 参考文献卡片草稿

```
id: deepseekai2026v4
title: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
a: DeepSeek-AI (2026), technical report (preview). arXiv:2606.19348v1
url: https://arxiv.org/abs/2606.19348
finding: V4-Pro（1.6T/49B）与 V4-Flash（284B/13B）用 CSA+HCA 混合注意力、mHC 与 Muon，在 32–33T token 上预训练并原生支持 1M 上下文。
matters: 本课全部架构公式、超参数、效率数字与后训练流程的一手来源，也是若干笔误（位置 −i、QK-Clip 出处）的出处。

id: deepseekai2025v32
title: DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
a: DeepSeek-AI (2025), technical report. arXiv:2512.02556v1
url: https://arxiv.org/abs/2512.02556
finding: 提出 DSA（lightning indexer + top-2048 token 选择），并说明 V3.2 最大上下文为 128K。
matters: V4 的 27%/10% 都以 V3.2 为基线，且 1M 处的 V3.2 只能是外推——这一限定要靠它来说明。

id: deepseekai2024v3
title: DeepSeek-V3 Technical Report
a: DeepSeek-AI (2024), technical report. arXiv:2412.19437v2
url: https://arxiv.org/abs/2412.19437
finding: 定义了 V4 沿用的 DeepSeekMoE、无辅助损失均衡、MTP 与 FP8 训练框架。
matters: V4 声明“未说明的细节沿用 V3”，所以 V4 省略的训练细节只能回到这里查。

id: xie2025mhc
title: mHC: Manifold-Constrained Hyper-Connections
a: Xie, Wei, Cao, et al. (2025), preprint. arXiv:2512.24880v2
url: https://arxiv.org/abs/2512.24880
finding: 用 Sinkhorn-Knopp 把超连接的残差矩阵投影到双随机矩阵，n=4 时训练开销约 6.7%。
matters: V4 §2.2 的 mHC 是它的直接集成，本课只讲“怎么接进 V4”，数学留给第 16 课。

id: liu2025muon
title: Muon is Scalable for LLM Training
a: Liu, Su, Yao, et al. (2025), preprint. arXiv:2502.16982v1
url: https://arxiv.org/abs/2502.16982
finding: 给 Muon 加权重衰减并按 0.2·√max(A,B) 缩放更新，使其能复用 AdamW 超参数并在大模型上约 2× 计算效率。
matters: V4 算法 1 的权重衰减、Nesterov 与 update RMS 缩放都“沿用”它，但常数从 0.2 改为 0.18；也用来纠正 V4 对 QK-Clip 的误引。

id: jordan2024muon
title: Muon: An optimizer for hidden layers in neural networks
a: Keller Jordan et al. (2024-12-08), blog post
url: https://kellerjordan.github.io/posts/muon/
finding: 定义 Muon（动量 + 5 步 NS 正交化，系数 3.4445/−4.7750/2.0315），并建议 embedding 与输出层仍用 AdamW。
matters: V4 混合 NS 前 8 步的系数与“基线系数 (2,−1.5,0.5)”都出自这里，是解释“为什么要两段”的起点。
kind: artefact

id: kimiteam2025k2
title: Kimi K2: Open Agentic Intelligence
a: Kimi Team (Bai et al.) (2025), technical report. arXiv:2507.20534v2
url: https://arxiv.org/abs/2507.20534
finding: 提出 MuonClip：在 Muon 更新后按最大注意力 logit 重缩放 W_q、W_k（QK-Clip），1T 参数模型 15.5T token 预训练无 loss spike。
matters: QK-Clip 的真正出处；V4 说自己靠 Q/KV 的 RMSNorm 就不需要它，对比时要引这一篇。

id: openai2025gptoss
title: gpt-oss-120b & gpt-oss-20b Model Card
a: OpenAI (2025). arXiv:2508.10925v1
url: https://arxiv.org/abs/2508.10925
finding: 采用 128 token 带状窗口与稠密层交替、每头可学习的 softmax 分母偏置（attention sink）、带截断的 SwiGLU，以及 4.25 bit/参数的 MXFP4 MoE 权重。
matters: V4 的 SwiGLU clamping、attention sink 与 FP4 专家权重都引用它，说明这些是“借来的已验证部件”。

id: gu2024minillm
title: MiniLLM: On-Policy Distillation of Large Language Models
a: Gu, Dong, Wei, Huang (2024), ICLR 2024. arXiv:2306.08543v6
url: https://arxiv.org/abs/2306.08543
finding: 把标准知识蒸馏的前向 KL 换成反向 KL，并推导出在学生自身采样上优化该目标的 on-policy 方法。
matters: V4 用 OPD 取代混合 RL 阶段，反向 KL 在线蒸馏的学术源头之一。

id: roller2021hash
title: Hash Layers For Large Sparse Models
a: Roller, Sukhbaatar, Szlam, Weston (2021), NeurIPS 2021. arXiv:2106.04426v3
url: https://arxiv.org/abs/2106.04426
finding: 按当前 token 哈希到不同的前馈权重，无需路由参数和负载均衡损失，效果不逊于 Switch/BASE 等学习式路由。
matters: V4 前 3 个 MoE 层就是这种路由，替换了 V3 开头的稠密层。

id: xiao2024sinks
title: Efficient Streaming Language Models with Attention Sinks
a: Xiao, Tian, Chen, Han, Lewis (2024), ICLR 2024. arXiv:2309.17453v4
url: https://arxiv.org/abs/2309.17453
finding: 发现模型会把很强的注意力分数给开头 token（即使语义上不重要），保留这些 token 的 KV 就能让窗口注意力在超长流式输入下恢复性能；预训练时加专用 sink token 还能更好。
matters: V4 式 (27) 的可学习 sink logit 是这一现象的“显式化”版本。

id: cheng2026engram
title: Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
a: Cheng, Tian, Zeng, et al. (2026), preprint. arXiv:2601.07372v2
url: https://arxiv.org/abs/2601.07372
finding: 提出 Engram——把经典 N-gram embedding 现代化为 O(1) 查表的“条件记忆”，作为与 MoE 条件计算互补的稀疏维度。
matters: V4 结论把“更稀疏的 embedding 模块”列为未来方向并引用它，是连接第 17 课与第 19 课的线索。

id: hf2026dsv4card
title: DeepSeek-V4-Pro / DeepSeek-V4-Flash model cards, config.json and inference code
a: DeepSeek-AI (2026), Hugging Face repositories (MIT License), commit b5968e91 / 60d8d707
url: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
finding: 给出逐层 compress_ratios、RoPE/YaRN 设置、FP4/FP8 权重格式与可运行的 CSA/HCA/mHC 参考实现。
matters: 论文省略或写错的细节（输出逆 RoPE 位置、Hadamard 旋转、Sinkhorn 顺序、route_scale）都以它为准，并据此复算参数量。
kind: artefact

id: hf2026dsv4ga
title: DeepSeek-V4-Pro-0813 and DeepSeek-V4-Flash-0731 model cards
a: DeepSeek-AI (2026), Hugging Face repositories (MIT License)
url: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813
finding: GA 版沿用 Preview 结构并附加 DSpark 投机解码模块，Agent 基准（如 Terminal Bench 2.1 87.9）大幅高于 Preview。
matters: 区分“论文里的 Preview”与“2026-08 之后线上的 GA”，评测数字不能混用。
kind: artefact

id: deepseek2026changelog
title: DeepSeek API Docs — Change Log and News (V4 Preview 2026-04-24, V4-Pro GA 2026-08-13)
a: DeepSeek (2026), API documentation
url: https://api-docs.deepseek.com/updates
finding: 记录了 V4 Preview 上线、旧模型名退役、Flash/Pro 正式版、峰谷定价与 V4.1-Flash 替代 V4-Flash 的日期。
matters: 本课时间线与“Preview vs GA”口径的唯一官方依据。
kind: artefact

id: flashmla2026readme
title: FlashMLA README (sparse attention kernels; KV cache formats for V3.2 / V4 / V4.1)
a: DeepSeek-AI (2025–2026), GitHub repository
url: https://github.com/deepseek-ai/FlashMLA
finding: 规定 V3.2 每 token 每层 656 B、V4 每条目 584 B（448 FP8 + 64 BF16 + 8 字节缩放）的 KV 存储格式。
matters: KV 字节复算与图 1 拟合的精度依据，论文只说了“RoPE 维 BF16、其余 FP8”。
kind: artefact

id: deepgemm2026readme
title: DeepGEMM README (Mega MoE, FP4 indexer, V3.2 MQA logits kernels)
a: DeepSeek-AI (2025–2026), GitHub repository
url: https://github.com/deepseek-ai/DeepGEMM
finding: 给出 V3.2 indexer 的 FP8 键 + 每 token 一个 float 缩放的格式，并发布了 V4 的 Mega MoE 与 FP4 indexer 内核。
matters: V3.2 每 token 132 B indexer 键的来源，以及 §3.1 MegaMoE 开源的对应制品。
kind: artefact
```

---

## 10. 核查记录（对抗式复核，2026-09-24）

> 复核方式：挑出读者最可能转述的 28 条论断，逐条回到一手来源重新核对，不沿用研究底稿的结论。来源：`arxiv.py text/grep 2606.19348`（v1 LaTeXML 全文）、
> `arxiv.py grep 2502.16982 / 2507.20534 / 2512.02556 / 2512.24880`、HF `raw/main` 的 `config.json`、`inference/model.py`、`inference/kernel.py`、
> `encoding/README.md`、四张模型卡、HF API（`createdAt`/`sha`）、FlashMLA 与 DeepGEMM README、DeepSeek API 文档（Change Log、thinking_mode、pricing）、
> Jordan 博客；图 1、图 5、图 9 在浏览器里打开 arXiv HTML 的原图/SVG 目视读数。所有算术用 Python 重算。

### 10.1 核对无误（原文照旧）

| # | 论断 | 核对结果与来源 |
|---|---|---|
| 1 | 摘要：Pro 1.6T/49B、Flash 284B/13B、1M 上下文、“more than 32T”、27% FLOPs / 10% KV | 摘要逐字一致（C3 引文无误） |
| 2 | §1：“equivalent FP8 FLOPs”、“estimated”；Flash 10% / 7%；FP4×FP8 “1/3 more efficient” | §1 第 427–431 行一致 |
| 3 | §4.2.1 全部规格（层数、$d$、$n_h$、$d_c$、top-k、$g$/$d_g$、$m$/$m'$、indexer 64×128、$n_{\text{win}}$=128、专家 384/256+1、激活 6、中间维 3072/2048、MTP 1、$n_{hc}$=4、$t_{\max}$=20、前 3 层 Hash、前 2 层 HCA / 纯 SWA） | 论文与两份 `config.json` 逐项一致 |
| 4 | `compress_ratios` 计数：Pro 30 CSA + 31 HCA；Flash 21 CSA + 20 HCA + 2 SWA；末项为 MTP 的 0 | 用 Python 数 config 列表，一致 |
| 5 | 式 (1)–(8)（mHC）、(9)–(12)（CSA 压缩）、(13)–(19)、(20)–(26)、(27)、(28)、(29) 与算法 1 | 与 v1 LaTeX 源逐符号比对，一致（含 $C_l=2\sigma$、$s<\operatorname{Floor}(t/m)$、$\sqrt{\max(n,m)}\cdot\gamma$、$W(1-\eta\lambda)-\eta O$） |
| 6 | 混合 NS：8 步 (3.4445, −4.7750, 2.0315) + 2 步 (2, −1.5, 0.5) | §2.4 一致；Python 复算 8A 后 [0.6818, 1.1344]、+2B 后 [1.0000, 1.00054] |
| 7 | QK-Clip 误引（E7） | V4 参考文献里只有一篇 Liu et al. (2025)，即 “Muon is scalable…”；`grep 2502.16982 'QK-Clip\|clip'` 无命中；`grep 2507.20534 'QK-Clip'` 命中（MuonClip） |
| 8 | Liu et al.：Lemma 1 的 $\sqrt{1/\max(A,B)}$、$0.2\cdot O\cdot\sqrt{\max(A,B)}$、Nesterov 脚注、系数沿用 Jordan；v1、28 位作者 | 2502.16982v1 §2.1–2.2 一致 |
| 9 | 超参数：AdamW (0.9, 0.95, 1e-20, wd 0.1)；Muon 动量 0.95、wd 0.1、RMS 0.18；32T/33T；75.5M/94.4M；LR 2.7e-4→2.7e-5、2.0e-4→2.0e-5；2000 步预热；4K→16K→64K→1M；前 1T 稠密；偏置速度 0.001；均衡损失 1e-4；MTP 0.3→0.1 | §4.2.2 逐项一致；§4.2.2 两处 AdamW 清单确实漏了 mHC 偏置与门控（E14） |
| 10 | 输出逆 RoPE 用查询位置（E8） | `model.py` 第 486、534 行：`freqs_cis[start_pos:start_pos+seqlen]` + `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`；FlashMLA 称 “O-RoPE (conjugate)” |
| 11 | 压缩块可见性 $s<\lfloor(t+1)/m\rfloor$（E9） | `model.py` 第 274、425、429 行的掩码 `>= arange(1, seqlen+1)//ratio` |
| 12 | 逐通道 softmax 压缩、重叠两路、`ape` 形状、压缩位置取块首（E4、§2.6） | `Compressor.forward`：`score` 形状 `[b, n/m, 2m, c]`、`softmax(dim=2)`；`overlap_transform`；`freqs_cis[:cutoff:ratio]` |
| 13 | 类注释仍写 “MLA”、Q 无权重 RMSNorm、滑窗 `wkv`、RoPE θ（压缩层 160000+YaRN，纯 SWA 层 10000 无 YaRN）、indexer 头权重缩放、Hadamard | `model.py` 第 437–534、395–418 行一致；Pro 与 Flash 的 `model.py`/`kernel.py` md5 相同 |
| 14 | KV 格式 584 B = 448 + 128 + 8（7 个 e8m0 + 1 padding）；V3.2 656 B；V3.2 indexer 键 128 + 4 B；MQA 模式 `head_dim_k` 576 vs 512 | FlashMLA README 第 81、131–137 行；DeepGEMM README `fp8_mqa_logits` 输入说明 |
| 15 | KV 复算：48,068 / 5,031.44 / 3,514.25 B；@$2^{20}$ 50.40 / 5.276 / 3.685 GB；9.55×、13.68×；GQA8 基线 2.01% / 2.00%；584/1024=57.0%；SWA/压缩 7.88×/7.95× | Python 全部重算一致 |
| 16 | FLOPs 模型：@$2^{20}$ 1.1916 / 0.3158 / 0.1183 T；@128K 52.0%；交叉点 911；原始口径 Flash 8.27×；F10 表 13 列 | Python 全部重算一致（见 10.2 第 2 条对“一致”措辞的修正） |
| 17 | 参数量：Pro 1,573.0B（含 MTP 1,598.8B）、激活 48.85B/49.78B；Flash 284.3B/290.9B、13.27B/13.80B；分组输出投影 39.3%/50.0% | 按 `model.py` 的参数定义独立重写计数脚本，结果一致 |
| 18 | §3.1：$6hd$、$3h$、$2d=6144$、6.1 TFLOP/s per GB/s；1.50–1.73×、1.96×；PR #304；图 5 理论加速 1.42× / 1.92×（Flash 配置） | §3.1 正文与图 5 SVG 一致 |
| 19 | 6.7%（mHC）、约 20%（Anticipatory Routing）、<10% padding、每 rank ≤5 个矩阵、BF16 随机舍入 + all-to-all + FP32 本地求和 | §3.4.1、§3.4.2、§4.2.3 一致；mHC 论文 6.7% 也核对 |
| 20 | FP4 QAT：MXFP4、专家权重 + indexer QK、FP32→BF16 index score、2×、99.7%、无损反量化、STE | §5.2.1 一致；§4.9 的 $2^{14}$ 阈值用 Python 穷举 E4M3 可表示值重算一致 |
| 21 | 表 1、表 6、表 7 的全部引用数字；Codeforces 3206、第 23；SimpleQA 高 20 个百分点；3–6 个月；Putnam-2025 120/120 | 论文表格与正文一致 |
| 22 | 图 9 八个长度的 MRCR 8-needle 读数（Pro-Max 0.59、Flash-Max 0.49 @1024K 等） | 在浏览器中打开 `mrcr.svg`，数据标签与底稿完全一致 |
| 23 | 图 1 标注 3.7× / 9.8× lower、9.5× / 13.7× smaller | 打开 `dsv4_performance.png` 目视一致 |
| 24 | V3.2 只支持 128K、top-k 2048 | 2512.02556 §4.1 明说最大上下文只有 128K（grep `maximum context length`）；top-k 2048 见 §2 训练段落 |
| 25 | OPD 式 (29)、十个以上教师模型（grep `more than ten teacher`）、token 级估计的 sg 写法、全词表蒸馏、混合 RL 阶段被整体替换 | §5.1、§5.1.2 一致 |
| 26 | 时间线：2026-04-24 发布与旧名 07-24 停用、07-31 Flash-0731“同架构同规模、只重做后训练”、08-13 Pro GA（Responses API、low/high/max）、08-16 16:00 UTC 新价、08-21 Vision-Exp、09-10 V4.1-Flash 与 V4-Pro 在 09-14 之后继续提供；HF `createdAt` 与 commit | Change Log 与 HF API 一致 |
| 27 | 模型卡：精度栏（Base FP8 Mixed、指令 FP4+FP8 Mixed）、采样建议、0813 结构说明与 DSpark 字段、Terminal Bench 2.1 / HLE 数字、Cybergym 83.1 vs 78.3（E22）；技术报告 PDF 链接 404（E21） | 四张卡与 0813 `config.json` 一致；PDF 链接 HEAD 请求返回 404 |
| 28 | API：effort 映射（minimal→low … ultra→max）、带/不带 tools 的 `reasoning_content` 规则、V4-Pro 峰谷价格与 1M/384K | thinking_mode 与 pricing 页面一致 |

### 10.2 本次改动

1. **mHC 代码与论文其实等价（§2.1 代码对照、E23）。** 底稿把“`hc_post` 用 $B^\top$”和“Sinkhorn 以列归一化收尾”列成两处差异。核查：`kernel.py` 第 401–424 行先行 softmax、再列，之后 19 轮“行、列”；`model.py` 第 685 行 $\mathbf y_k=\sum_j\texttt{comb}[j,k]\mathbf x_j$。把 `comb` 视为 $B_l^\top$，代码就是论文式 (8) 作用在转置矩阵上，两处差异互相抵消。已改写为“等价，只是下标转置”。
2. **“与图 1 一致”说过头了（§4.3、E2）。** 笔者模型的 Flash 比值是 10.0–10.07×，图 1 标注 9.8×，差约 3%；Pro 3.73–3.77× vs 3.7×。已改为“接近”，并把 E2 的“才能同时拟合”改为“一种可行解释，未证明唯一”。
3. **Jordan 5 步 NS 的区间（§4.5 表）。** [0.682, 1.134] 只对该随机矩阵（初始奇异值 0.06–0.19）成立；对 $x_0\in[10^{-3},1]$，5 步后是 [0.471, 1.202]。核查者还验证了 [0.682, 1.134] 是 $\varphi_A$ 的不变区间，所以它是振荡带而不是“5 步的结果”。已加注，防止研究摘要里“Jordan 5 步落在约 [0.68, 1.13]”的说法流进讲义。
4. **KV 比值“任何长度都约 10.5%”（E1）** 只在图 1 口径下成立（只计增长型条目）。计入每请求 4.56 MB 滑窗状态后，4K 时是 12.8%，32K 时是 10.8%。已加限定。
5. **Codeforces 评分方法（C72）。** 原写“按期望排名换算评分”。论文 §5.3.1 是先把每个抽样序列换算成评分，再对评分取期望，最后对 14 场取平均。已改。
6. **位置 $-i$（C28、E8）。** 补充：同一小节的式 (27) 用 $i$ 表示查询 token，所以更可能是记号混用而非公式错误；FlashMLA 也把这一步叫 “O-RoPE (conjugate)”。结论（课上写 $R_{-t}$）不变。
7. **“1/3 更高效”的来由（C5）** 从“笔者推导”改为“笔者推测”：论文没有推导，位宽比只是与数字相符的一种解释。
8. **MRCR 1M（E20）** 补充：它也不是图 9 八个点的简单平均（Pro 0.8225、Flash 0.774），纵轴标 “Average MMR”。

### 10.3 本次未复核（沿用底稿，写课时如要引用需自行再查）

- 图 1 的像素读数与拟合表（§4.4）只核了标注，没有重做像素换算；图 10 横坐标未看。
- Flash-0731、两款 DSpark 与 Base 的 `config.json` 差异（§3.6）只核了 Pro-0813 一份。
- §3.2–§3.3（TileLang 细节、批不变核）、§5.1.1 的 Quick Instruction / DSML / 交错思考、§5.2.2–§5.2.5、表 8–10 与开发者调查只抽查了个别数字（Host Codegen <1 μs、N=85 的 52%/39%/<9%、DSec 三组件，均一致）。
- 研究底稿第 8 节 UNVERIFIED 的各项维持原状；本次复核没有找到能把其中任何一项升级为“已核实”的一手来源（indexer 键 68 B 的生产格式在 DeepGEMM README 中也没有写明）。

---

## 11. 补充核实（写课者，2026-09-24）

> 写第 18 课时新增的数字。没有新的外部来源：全部是笔者（写课者）用 Python 对本笔记已有的公式、维度与系数复算或推导的结果，
> 表达式照录，便于复现。讲义里凡用到这些数字的地方都标了“笔者复算/推导/观察”。

| 量 | 表达式 | 值 | 讲义位置 |
|---|---|---|---|
| V3.2 原始 FLOPs @$2^{20}$（不加精度权重） | `2*37e9 + 61*128*2176*2048 + 61*64*256*2**20` | 0.074 + 0.0348 + 1.048 = 1.1568 T，indexer 占 90.6% | §1 图 18.1、式 (18.2) |
| 同上 @8K/32K/128K/256K/512K | 同式 | 0.117 / 0.142 / 0.240 / 0.371 / 0.633 T | 图 18.1 |
| V3.2 每解码一步读 indexer 键 @$2^{20}$ | `132*61*2**20` | 8.44e9 B；核心注意力读 `2048*656*61` = 8.2e7 B | §1 例题 |
| Flash 路由专家占总参 | `256*3*4096*2048*43 / 284.3e9` | 277.0B / 284.3B = 97.4% | §1 例题 |
| CSA 层对全部压缩条目做稠密注意力（Pro，@$2^{20}$，每层每查询） | `128*2048*2**18`；含窗口 `(2**18+128)*128*2048` | 6.87e10；6.88e10 | §3 开头、练习 2 |
| CSA 层核心注意力（Pro，每层每查询） | `1152*128*2048` | 3.02e8 | §3 小练习 |
| HCA 层核心注意力（Pro，每层每查询，@$2^{20}$） | `8320*128*2048` | 2.18e9 | §4 例题 |
| HCA 与 CSA 层每 token 字节比 | `163/4.5625` | 35.7（“1/36”） | §4 例题 |
| attention sink 玩具例 | 打分 (2,1,0)，$z'=1$ | 权重 (0.534, 0.197, 0.072)，和 0.803；无 sink 时 (0.665, 0.245, 0.090) | §3 例题 |
| CSA 压缩玩具例（$m=2,c=2$） | 见讲义表格 | $\mathbf e_1=(0.8,\ 19/9)$ | §2 例题 |
| 部分 RoPE 玩具例 | θ=0.1，(t,p)=(10,4) 与 (110,104) | 打分 cos(−0.6)=0.825；逆旋转后输出 (0.825, −0.565)；不逆旋转 (0.921, 0.389) 与 (−0.561, −0.828)（原记 (−0.575, −0.818) 有误，见 11.3） | §4 例题、图 18.9 |
| $\varphi_B(1+\varepsilon)$ 展开 | 二项式展开 | $1+0.5\varepsilon^2+3.5\varepsilon^3+2.5\varepsilon^4+0.5\varepsilon^5$ | §5 推导 |
| 牛顿–舒尔茨手算例 $M=\begin{pmatrix}1&2\\2&1\end{pmatrix}$ | `numpy` 按式 (28) 迭代 8A+2B | $X_1$ 对角 −0.0964、非对角 0.8483；$X_8$ 0.2262 / 0.9081；$X_{10}$ −0.0002 / 1.0004；奇异值 (1.0005, 1.0002)；目标 $UV^\top=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | §5 例题与表 |
| 对角例 diag(3,4)/5 | $\varphi_A(0.6),\varphi_A(0.8)$ 后接 $\varphi_B$ | 1.1933, 0.9765 → 1.0476, 1.0002 → 1.0015 | 练习 9 |
| $\varphi_A(2)$ | `3.4445*2 - 4.775*8 + 2.0315*32` | 33.70（不归一化会发散） | 练习 12 |
| 牛顿–舒尔茨开销占比（Pro 的 $W^{DQ}$） | `7.49e11 / (6*7168*1536*94371840)` | 1.2e-4（约 0.012%） | §5 例题 |
| mHC 动态投影 61 层合计（Pro） | `1376256*61` | 8.4e7 | §5 例题 |
| 单样本 KL 估计的方差（3 词玩具例） | 学生 (0.7,0.2,0.1)，教师 (0.5,0.4,0.1) | 精确反向 KL 0.0969 nat；单样本估计方差 0.166、标准差 0.41 | §6 例题 |
| 状态缓存计入后 32K 处的比值 | `(5031.4375*32768 + 4559872)/(48068*32768)` | 10.8%（4K 12.8% 已在 E1） | §7 例题 |
| GQA8 → V3.2 的倍数 | `249856/48068` | 5.20 | 图 18.13 说明 |
| 斜率（等效 FP8，每个上下文 token） | V3.2 indexer `61*16384`；Pro HCA `2*31*2048`、indexer `0.5*30*16384/4` | 999,424；126,976 + 61,440 = 188,416 | 练习 1 |
| E2M1 → E4M3 无损区间 | 穷举（本笔记 §4.9） | $e\in[-8,6]$，比值 ≤ $2^{14}$ | §6 例题（沿用 §4.9） |

另：讲义 §6 引用了 MIT 许可的 `encoding/README.md` 中 Think Max 注入指令的首句（本笔记 C60 已摘录），未引用论文正文的任何句子。

### 11.1 教学复核补充（2026-09-24）

> 复核讲义时新增的数字，同样没有新的外部来源，全部是对本笔记已有公式与维度的复算。

| 量 | 表达式 | 值 | 讲义位置 |
|---|---|---|---|
| sigmoid 与 $\sqrt{\operatorname{softplus}}$ 对照 | `1/(1+exp(-x))`；`sqrt(log1p(exp(x)))`，$x=-4,0,4,8$ | 0.018 / 0.5 / 0.982 / 0.9997；0.135 / 0.833 / 2.005 / 2.828 | §1 亲和度 |
| attention sink 取 $z'=5$ | `s=e^2+e^1+1=11.107`；`s/(s+e^5)` | 分母 159.5，真实条目权重和 0.070 | §3 例题 |
| 每块字节（Pro，块 = lcm(4,128) = 128 token） | `30*32*(584+68) + 31*1*584` | 644,024 B = `128*5031.4375` | §7 图 18.15 |
| CSA 层每块字节 | `32*(584+68)` | 20,864 B | 图 18.15 |
| SWA 条目每 token（Pro，61 层） | `61*584`；压缩条目每 token（不含 indexer 键）`30*146+31*4.5625` | 35,624 B 对 4,521.4 B，比值 7.88（论文“约 8 倍”，C50） | §7 正文 |
| 图 18.15 示意请求 | 已读 582 token | HCA 尾部 `582%128=70`，CSA 尾部 `582%4=2`，窗口 454–581 | 图 18.15 |
| $n_{\text{win}}\times L$ 的来历（笔者解释 C50） | 第 $\ell$ 层窗口条目依赖第 $\ell-1$ 层前 127 个位置的窗口条目，逐层向前叠加 | 精确需要 `128 + 127*60 = 7748`，论文的 `128*61 = 7808` 是上界（见 11.3） | §7 正文 |

### 11.2 教学复核补充（第二轮，2026-09-24）

> 第二轮教学复核新增的数字与说明。同样没有新的外部来源：数字是对本笔记已有公式、维度与系数的复算；
> 标“笔者解读”的是写课者的推理，讲义里也照此标注。另：本轮在 §3 新增一张图（分组输出投影），
> 原图 18.4–18.16 顺延为 18.5–18.17，本节以上各表里的图号已同步改过。

| 量 | 表达式 | 值 | 讲义位置 |
|---|---|---|---|
| CSA 压缩玩具例的多对一 | 通道 1 权重 (0.2, 0.2, 0.4, 0.2)，把 token 0、1 的候选 1 与 0 对调 | `0.2*0 + 0.2*1 + 0.4*2 + 0.2*(-1)` = 0.8，与对调前相同 | §2 warn“压缩丢掉了什么” |
| 每个条目合并的候选个数 | CSA `8*512`；HCA `128*512` | 4096 → 512；65,536 → 512 | 同上 |
| 分组输出投影三块参数（Pro） | `7168*65536`；`16*1024*4096`；`7168*16384` | 469.8M；67.1M；117.4M；`(67.1+117.4)/469.8` = 39.3%；`67.1/469.8` = 0.143（约 1/7） | §3 图 18.4 |
| 混合牛顿–舒尔茨对极小奇异值 | 按式 (28) 对标量迭代 8 步 $\varphi_A$ + 2 步 $\varphi_B$，起点 $\sigma_0=\sigma_i/\norm{M}_F$ | $10^{-5}\to0.686$；$10^{-6}\to0.0791$；$10^{-7}\to0.00793$（$10^{-4}\to1.002$，与 §4 一致） | §5 类比“调音台”的失效处 |
| OPD 与 GRPO 的监督信号个数（示意） | 回答长 $10^4$ token：GRPO 1 个优势（(15.11)，全体 token 共享）；OPD `1e4*129280` | 1 对 1.29e9 个数（只是按定义计数，不等于信息量） | §6“为什么最后一步换成了蒸馏” |
| OPD 替代“离线蒸馏 + 混合 RL”（笔者解读） | 对照第 15 课 §4：V3.2 = 专家 RL → 专家数据 SFT → 一次混合 RL；V4 = 专家 SFT+GRPO → OPD（C58、C60、C65） | 论文只说“混合 RL 阶段被整体换成 OPD”（C58），未给理由 | 同上 |
| 部分 RoPE 取 64 维的旁证（笔者观察） | V2/V3 解耦 RoPE 键 $d_h^R=64$；V4 条目里只有 RoPE 的 64 维以 BF16 存（C31、FlashMLA 格式） | 论文未解释为什么是 64 维 | §4“做法” |

### 11.3 数学复核（第三轮，2026-09-24）

> 逐条重算讲义中的推导与数字后做的更正与补充。没有新的外部来源，全部是 Python 复算或推导。

| 量 | 表达式 | 值 | 讲义位置 |
|---|---|---|---|
| 部分 RoPE 玩具例情形 B 不逆旋转的输出（**更正**） | `(cos(10.4), sin(10.4))` | (−0.561, −0.828)；原稿 (−0.575, −0.818) 是算错的 | §4 例题 |
| 部分 RoPE“只依赖相对位置”的前提（笔者补充） | 块边界固定在 4、128 的整数倍 | 条目内容相同时成立；整段平移只在平移量为 128 的倍数时不改变条目 | §4 (18.15) 后、练习 11 |
| $\varphi_A$ 振荡带为何不变 | $\varphi_A'(\sigma)=0$ 于 $\sigma=0.5545,\ 1.0501$；`phiA(1.0501)`、`phiA(0.6818)` | 局部极小 0.6818、`phiA(0.6818)=1.1344`，区间 [0.682, 1.134] 的像恰为自身；不动点 0.868 处斜率 −1.58（排斥） | §5 |
| $\varphi_A$ 从 $10^{-3}$ 出发 | 逐步迭代 | 0.0034, 0.0119, 0.0409, 0.1404, 0.4705, **1.1702**, 0.8368, 0.9179 | §5 |
| 手算例 $X_{10}$ 与 $UV^\top$ 的差 | `X10 - [[0,1],[1,0]]` | 逐元素最大 3.6e-4；谱范数 5.4e-4 | §5 表后 |
| 不归一化时的发散 | `phiA(33.697)` | 8.8e7 | 练习 12 |
| attention sink 的缩放 | `11.107/13.825` | 0.8034：有 sink 时输出恰为无 sink 时的 0.803 倍 | §3 例题 |
| $6ND$：Pro / V3 | `9.702e24 / 3.2856e24` | 2.95（原稿写 2.9） | §6 例题 |
| 图 1 标注与复算之差（Flash） | `10.0/9.8`、`10.07/9.8` | 2.0%–2.8% | §7 |
| MRCR 128K 以内的最低点 | Flash-Max 16K | 0.84（原稿写 0.85–0.94） | §7 图 18.17、要点 |
| 窗口状态重算的精确长度 | `128 + 127*60` | 7748 ≤ 7808 | §7 正文 |
| V3.2 在 128K 处 indexer 占比（原始口径） | `999424*131072 / 0.2398e12` | 54.6% | 图 18.1 说明 |

### 11.4 事实与引用复核（第四轮，2026-09-25）

> 逐条核对讲义里的事实、归属与引用。新增的外部核实来源：`arxiv.py grep/text/meta`（v1/v2 全文与元数据）、
> Hugging Face 数据集 `openai/mrcr` 的说明页与 API、第 19 课研究笔记（V4.1 的持久化 KV）。讲义里据此做的改动也列在这里。

| # | 论断 | 核对结果与来源 | 讲义改动 |
|---|---|---|---|
| 1 | V3 的节点受限路由 $M=4$ | `arxiv.py grep 2412.19437 'at most 4 nodes'`：§4.2 “each token will be sent to at most 4 nodes (i.e., M=4)” | 无（原文正确） |
| 2 | V3.2 论文是否给出 indexer 64 头 × 128 维、MQA 模式 576/512、激活 37B | `grep 2512.02556`：论文只说 indexer “a small number of heads”（第 171 行）；附录 A 图 7 说 V3.1-Terminus 训练/prefill 用 MHA 模式、解码用 MQA 模式；全文无 “576”“37B”“671B”。64×128 来自 `config.json`（第 14 课笔记第 665、707 行）；576/512 来自 FlashMLA README；37B/61 层/128 头来自 V3 论文 | §1 例 (18.2) 前的维度改为分别引 V3、V3.2（MQA 模式）、FlashMLA，并注明 indexer 维度来自 config；V3.2 参考文献卡的 finding 去掉“64 头 × 128 维” |
| 3 | 专家培养的领域 | `grep 2606.19348 'target domain'`（§1 第 439 行）：“such as mathematics, coding, agent, and instruction following” | 无（讲义 §6 与图 18.12 的四个领域有据） |
| 4 | PagedAttention 的“基本假设”被违反在哪里 | §3.5.1：两个障碍——各层缓存策略不同（如 SWA）、高性能注意力内核的约束（含对齐要求）；论文没说“按 token 分页” | §7 改写为论文列的两个障碍；第 4 课链接由 #p2 改为讲 PagedAttention 的 #p3 |
| 5 | SWA 条目“约 8 倍” | §3.5.2：比较对象是 “the compressed CSA and HCA KV entries”，没提 indexer 键；不含 indexer 键 7.88×，含则 7.08× | §7 补一句口径说明（不含 indexer 键为研究笔记推测） |
| 6 | FP4 QAT 复用的 FP8 框架 | §5.2.1：“fully reuse the existing FP8 training framework”——没说是 V3 的那一套（与第 8 节 UNVERIFIED #1 一致） | §6 由“V3 的 FP8 训练框架”改为“现有的 FP8 训练框架”，并注明论文未明说 |
| 7 | Flash 前 1T token 的“稠密注意力”是什么 | §4.2.2 只说 “warmup the model with dense attention for the first 1T tokens”，没说此时压缩分支是否启用 | §6 的“压缩条目全部参与、不做 top-k”改标为笔者的读法 |
| 8 | attention sink 与 SwiGLU clamping 的引用 | §2.3.3 sink 引 Xiao et al. (2024) 与 OpenAI (2025)；§4.2.3 clamping 引 OpenAI (2025)；§5.2.1 专家权重占显存引 OpenAI (2025) | 无 |
| 9 | 全词表 OPD 为何只缓存隐藏状态 | §5.2.2：词表超过 100k 时为所有教师物化 logits “prohibitive, even when spooled to disk” | 无 |
| 10 | MRCR 是什么 | HF 数据集说明页（标题 “OpenAI MRCR: Long context multiple needle in a haystack benchmark”；MRCR = “Multi-round co-reference resolution”）：合成长多轮对话里混入 2/4/8 个相同请求，要求在第 $k$ 个回答前加哈希前缀并复现它，用 difflib `SequenceMatcher` 比值打分，缺前缀记 0；长度分箱 4K–8K … 524K–1M；与 GPT-4.1 发布关联；HF API `createdAt` 2025-04-11。V4 论文把它引作 “OpenAI, 2024e” | §7 补上 MRCR 的定义，新增参考文献卡 `openai2025mrcr`（artefact） |
| 11 | V4.1 对滑动窗口持久化缓存做了什么 | 第 19 课笔记 C4（V4.1 论文）：“the persistent KV cache no longer stores SWA KV”，SWA KV 改放主机内存池、丢了只回放 $n_{\text{win}}$ 个 token | 封面“位置”段与 §7 末段：由“再砍掉大半 / 持久化缓存有界重放”改为“不再进入持久化缓存，丢了只回放 128 个 token” |
| 12 | 正式版与预览版的关系 | C79、§3.6：Flash-0731 官方说“同架构同规模、只重做后训练”；Pro-0813 只说“建立在 Preview 的模型结构上 + DSpark”；两份 GA config 相对 Preview 多 DSpark 字段与 `compress_ratios` 末尾两个 0 | 封面版本口径、§7 时间线后与要点：不再说 Pro 正式版“改的是后训练”，改为“权重不同（Flash 官方说明是重做后训练）”；config 差异补上两个 0 |
| 13 | 十一篇 arXiv 文献的 id、版本、日期、作者 | `arxiv.py meta`：2606.19348v1（2026-04-26）、2512.02556v1、2412.19437v2、2601.07372v2、2512.24880v2、2502.16982v1、2507.20534v2、2306.08543v6（ICLR 2024）、2309.17453v4（ICLR 2024）、2508.10925v1、2106.04426v3；标题与第一作者均与参考文献卡一致 | 无 |
| 14 | Hash 路由一句的引用 | 查表形状（词表 × 6）与“门控值仍来自打分”来自代码（C12），Roller et al. 只支持“按 token 哈希分配专家” | §1 该句改引 `hf2026dsv4card`，Roller 单独成句 |
| 15 | 讲义中未见于本笔记的算例数字 | Python 复算：牛顿–舒尔茨手算例第 2 步 $X_2$ 对角 0.1458、非对角 0.9026，奇异值 1.0484/0.7568；第 9 步 0.0282/0.9901，奇异值 1.0183/0.9619；两位学生的 KL 小练习 0.582/0.639（反向）、1.238/0.303（前向）；$\varphi_B$ 端点 1.134→1.018→1.0002、0.682→0.962→1.0005 | 无（均与讲义一致） |
