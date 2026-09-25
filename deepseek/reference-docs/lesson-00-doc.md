# 第 00 课研究笔记：DeepSeek 科技树的时间线与规格总表（服务第 0 课总览与第 20 课复盘）

> 本文件是第 0 课 `lesson-00-tech-tree` 与第 20 课 `lesson-20-retrospective` 的核对底稿，不是课文。建立于 2026-09-24。
> 所有引文均为原文英文、逐字照抄的短片段（为避免大段转载，每条引文只截取承载事实的那一句或半句，其余用中文转述）；
> 所有派生数字都在 Python 中复算过，表达式见 §4（脚本存于笔者的临时目录，不进仓库）。
> 凡是没能在一手来源里找到的说法，一律放进 §8，不进课文。机制细节（MLA、DSA、CSA/HCA、CED 的推导）分别属于第 9、14、18、19 课，
> 本笔记只核对“哪一天、哪个编号、多少参数、多长上下文、每 token 多少 KV 字节”这一层，以及跨代比较所需的少量公式。
> 批注约定（本地共同编辑）：对本笔记的疑问或修改意见直接写成 `<!-- 批注(Harry): … -->`，回复写 `<!-- 回复(Claude): … -->`；任何事实改动先改这里再改课文。
> 与其它研究笔记重叠的条目（GPT-2/3 规格、V2 的 93.3%、V3.2 的 656 字节）已与 `lesson-01/04/09/14-doc.md` 交叉核对，结论一致处直接引用其编号。

---

## 0. 覆盖范围与所读版本

### 0.1 日期的定义（全篇统一，写课时必须照此区分）

| 叫法 | 定义 | 取数来源（优先级从高到低） |
|---|---|---|
| **论文 v1 日期** | arXiv “Submission history” 里 v1 的提交时间（UTC） | arXiv abs 页（本笔记逐篇抓取）；arXiv API 的 `published` 字段与之相同 |
| **公开发布日** | 权重或服务第一次对公众可用的日子（UTC） | ① DeepSeek API 新闻页 / Change Log 的日期；② GitHub README 里的 “released” 条目；③ Hugging Face（HF）上 “Release …” 提交或 GitHub 最早提交；④ HF 仓库 `createdAt` |
| **HF createdAt** | HF 仓库**创建**时间 | HF API。仓库常常先私有创建、后公开，所以它只是公开发布日的**下界参考**，不能当发布日（反例：V2 仓库 2024-04-22 创建，README 写 05-06 发布） |

### 0.2 读过的论文（有 arXiv 编号）

| 键 | 标题（简） | 作者 | arXiv 编号与所读版本 | v1 日期 → 最新版日期 | 发表状态 |
|---|---|---|---|---|---|
| **vaswani2017** | Attention Is All You Need | Vaswani, Shazeer 等 8 人 | 1706.03762（只核元数据；正文已在 `lesson-01-doc.md` 以 v7 核对） | 2017-06-12 → v7 | NIPS 2017（见 lesson-01-doc） |
| **brown2020** | Language Models are Few-Shot Learners（GPT-3） | Brown 等 31 人 | 2005.14165（元数据；正文见 lesson-01/02-doc，v4） | 2020-05-28 → v4 2020-07-22 | NeurIPS 2020（见 lesson-02-doc） |
| **ouyang2022** | Training language models to follow instructions with human feedback（InstructGPT） | Ouyang 等 20 人 | 2203.02155 **v1**（唯一版本） | 2022-03-04 | NeurIPS 2022 Main Conference Track（Advances in NeurIPS 35；proceedings.neurips.cc 页，核查补，见 §10） |
| **dscoder** | DeepSeek-Coder: When the Large Language Model Meets Programming | Daya Guo 等 13 人 | 2401.14196 **v2**（全文 grep） | 2024-01-25 → 2024-01-26 | 预印本 |
| **dsllm** | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism | DeepSeek-AI（86 人） | 2401.02954 **v1**（全文 grep，表 2） | 2024-01-05（唯一版本） | 预印本 |
| **dsmoe** | DeepSeekMoE: Towards Ultimate Expert Specialization in MoE LMs | Damai Dai 等 17 人 | 2401.06066 **v1**（全文 grep，§5） | 2024-01-11 | 预印本 |
| **dsmath** | DeepSeekMath: Pushing the Limits of Mathematical Reasoning | Zhihong Shao 等 11 人 | 2402.03300 **v3**（摘要 + grep） | 2024-02-05 → v3 2024-04-27 | 预印本 |
| **dsvl** | DeepSeek-VL: Towards Real-World Vision-Language Understanding | Haoyu Lu 等 15 人 | 2403.05525 **v2**（摘要） | 2024-03-08 → 2024-03-11 | 预印本 |
| **dsv2** | DeepSeek-V2: A Strong, Economical, and Efficient MoE LM | DeepSeek-AI（156 人） | 2405.04434 **v5**（全文 grep：表 1、§3.1、附录 B） | 2024-05-07 → v5 2024-06-19（共 5 版） | 预印本 |
| **prover1** | DeepSeek-Prover: Advancing Theorem Proving … Synthetic Data | Huajian Xin 等 9 人 | 2405.14333 **v1**（摘要） | 2024-05-23 | 预印本 |
| **dscoderv2** | DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models | DeepSeek-AI（39 人） | 2406.11931 **v1**（摘要 + grep） | 2024-06-17 | 预印本 |
| **esft** | Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning | Zihan Wang 等 6 人 | 2407.01906 **v2**（摘要 + grep） | 2024-07-02 → 2024-07-05 | EMNLP 2024 Main（GitHub README，§0.3） |
| **prover15** | DeepSeek-Prover-V1.5: … RL and Monte-Carlo Tree Search | Huajian Xin 等 17 人 | 2408.08152 **v1**（摘要） | 2024-08-15 | 预印本 |
| **fireflyer** | Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design | Wei An 等 52 人 | 2408.14158 **v2**（摘要 + comments） | 2024-08-26 → 2024-08-31 | SC'24（arXiv comments） |
| **alf** | Auxiliary-Loss-Free Load Balancing Strategy for MoE | Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, Damai Dai | 2408.15664 **v1**（摘要） | 2024-08-28 | 预印本 |
| **janus** | Janus: Decoupling Visual Encoding for Unified Multimodal … | Chengyue Wu 等 11 人 | 2410.13848 **v1**（摘要） | 2024-10-17 | 预印本 |
| **janusflow** | JanusFlow: Harmonizing Autoregression and Rectified Flow … | Yiyang Ma 等 14 人 | 2411.07975 **v2**（摘要） | 2024-11-12 → 2025-03-24 | CVPR 2025（arXiv comments） |
| **dsvl2** | DeepSeek-VL2: MoE Vision-Language Models … | Zhiyu Wu 等 27 人 | 2412.10302 **v1**（摘要 + grep） | 2024-12-13 | 预印本 |
| **dsv3** | DeepSeek-V3 Technical Report | DeepSeek-AI | 2412.19437 **v2**（全文 grep：表 1、§4.2） | 2024-12-27 → v2 2025-02-18 | 预印本 |
| **dsr1** | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | DeepSeek-AI | 2501.12948 **v2**（全文 grep：附录 B.4.4 Training Cost 表 7、G.2）；v1 另行 grep 比对 | 2025-01-22 → **v2 2026-01-04** | Nature 645:633–638，2025-09-17 在线（标题不同，见 E6） |
| **januspro** | Janus-Pro: Unified Multimodal Understanding and Generation … | Xiaokang Chen 等 8 人 | 2501.17811 **v1**（摘要 + grep） | 2025-01-29 | 预印本 |
| **codeio** | CodeI/O: Condensing Reasoning Patterns via Code I/O Prediction | Junlong Li 等 6 人 | 2502.07316 **v4**（摘要；作者单位 grep） | 2025-02-11 → 2025-05-21 | ICML 2025（arXiv comments） |
| **nsa** | Native Sparse Attention: Hardware-Aligned and Natively Trainable … | Jingyang Yuan 等 15 人 | 2502.11089 **v2**（摘要 + grep） | 2025-02-16 → 2025-02-27 | ACL 2025 长文 pp. 23078–23097，**Best Paper**（§0.3） |
| **grm** | Inference-Time Scaling for Generalist Reward Modeling | Zijun Liu 等 8 人 | 2504.02495 **v3**（摘要；单位 grep） | 2025-04-03 → 2025-09-25 | 预印本（comments：under review） |
| **proverv2** | DeepSeek-Prover-V2: … RL for Subgoal Decomposition | Z. Z. Ren 等 18 人 | 2504.21801 **v2**（摘要） | 2025-04-30 → 2025-07-18 | 预印本 |
| **insights** | Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware | Chenggang Zhao 等 15 人 | 2505.09343 **v2**（摘要；正文见 lesson-04-doc） | 2025-05-14 → 2025-12-23 | ISCA '25 Industry Track，DOI 10.1145/3695053.3731412 |
| **dsocr** | DeepSeek-OCR: Contexts Optical Compression | Haoran Wei, Yaofeng Sun, Yukun Li | 2510.18234 **v1**（摘要） | 2025-10-21 | 预印本 |
| **dsmathv2** | DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning | Zhihong Shao 等 9 人 | 2511.22570 **v1**（摘要；详见 lesson-15-doc） | 2025-11-27 | 预印本 |
| **dsv32** | DeepSeek-V3.2: Pushing the Frontier of Open LLMs | DeepSeek-AI（263 人） | 2512.02556 **v1**（全文 grep） | 2025-12-02 | 预印本 |
| **mhc** | mHC: Manifold-Constrained Hyper-Connections | Zhenda Xie 等 20 人 | 2512.24880 **v2**（摘要 + grep） | 2025-12-31 → 2026-01-05 | 预印本 |
| **engram** | Conditional Memory via Scalable Lookup: A New Axis of Sparsity | arXiv v2 列 Xin Cheng 等 21 人；ACL 版 14 人（E14） | 2601.07372 **v2**（摘要） | 2026-01-12 → 2026-07-12 | ACL 2026 长文 pp. 4968–4990（ACL Anthology） |
| **dsocr2** | DeepSeek-OCR 2: Visual Causal Flow | Haoran Wei, Yaofeng Sun, Yukun Li | 2601.20552 **v1**（摘要） | 2026-01-28 | 预印本 |
| **dsv4** | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | DeepSeek-AI（318 人） | **2606.19348 v1**（LaTeXML 全文，逐节读 §1、§2、§3.5、§4.1–4.2；图 1 PNG 目视） | 提交 **2026-04-26 14:49 UTC**；编号前缀却是 2606（E1） | 预印本（论文自称 “preview version”） |
| **dspark** | DSpark: Confidence-Scheduled Speculative Decoding with Semi-AR Generation | Xin Cheng 等 33 人 | 2607.05147 **v1**（摘要） | 2026-07-06 | 预印本 |
| **dsv41** | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | DeepSeek-AI（591 人） | **2609.19969 v1**（LaTeXML 全文，逐节读 §1–§4.2；图 1(b)、图 2 PNG 目视） | 2026-09-17 09:43 UTC | 预印本 |
| **dsec** | DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Agentic Training | Jialiang Huang 等 131 人 | 2609.22978 **v1**（摘要；单位 grep：DeepSeek-AI） | 2026-09-19 | 预印本（comments：早期两页摘要投 ATC 2026） |

**检索方式**：`arxiv.py search 'au:"DeepSeek-AI"'`（8 篇，全部为主线技术报告）、`au:"Wenfeng Liang"`（17 篇，补出 DSpark、DSec）、
`ti:DeepSeek`（大量第三方论文，逐条排除，见 E17）、核心作者 OR 检索（Damai Dai / Chong Ruan / Zhihong Shao / Huazuo Gao /
Chenggang Zhao / Haoran Wei / Xiaokang Chen / Daya Guo / Runxin Xu，同名噪声逐条人工排除）。
另有 DeepSeek 作者参与但非 DeepSeek 主导的论文（如 Math-Shepherd 2312.08935、RLCoder、RepoReasoner 等）不列入时间线。
**作者人数口径**（核查修正，§10）：表中“DeepSeek-AI（N 人）”是 arXiv API 作者字段里的**真人**数；该字段另含团体名 “DeepSeek-AI” 一项，dsllm 与 dsv41 还多一个 “:” 占位项，故 arXiv 页面上的条目数分别比 N 多 1 或 2（例：dsv41 共 593 项，其中真人 591）。

### 0.3 读过的制品（无论文，`kind: artefact`）

| 键 | 内容 | URL | 读取方式 |
|---|---|---|---|
| **NEWS** | DeepSeek API 新闻页 18 篇：news0725、0802、0905、1120、1210、1226、250115、250120、250325、250528、250821、250922、250929、251201、**260424、260813、260821、260910** | https://api-docs.deepseek.com/news/news260910 等 | 2026-09-24 抓 HTML 抽 `<article>`；注意 `news250120` 无尾斜杠时被重定向到 Quick Start 页，需加尾斜杠 |
| **CHANGELOG** | API “Change Log”（2024-05-17 至 2026-09-10 共 **21 个日期条目**、22 个小节；原记 “24 条” 有误，2026-09-24 核查重新计数，§10） | https://api-docs.deepseek.com/updates/ | 同上 |
| **HF-ORG** | deepseek-ai 组织下 105 个模型仓库及其 `createdAt` | https://huggingface.co/api/models?author=deepseek-ai | HF API |
| **HF-COMMITS** | V4-Flash、V4-Pro、V4.1-Flash 的提交历史 | https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash/commits/main 等 | HF API |
| **HF-CFG** | `config.json`：deepseek-llm-7b/67b-base、deepseek-moe-16b-base、DeepSeek-V2、V2-Lite、Coder-V2-Instruct、V3、R1、V3.1、V3.2-Exp、V3.2、V4-Flash、V4-Pro、V4.1-Flash、V4-Flash-0731、V4-Flash-Vision-Exp、coder-33b、math-7b | https://huggingface.co/deepseek-ai/<model>/raw/main/config.json | raw 文件 |
| **HF-CARD** | 模型卡：V2、V2-Lite、V3、V3.1、V3.2、V3.2-Exp、V3.2-Speciale、R1-0528、V3-0324、V2.5、V2.5-1210、V2-Chat-0628、Math-V2、OCR、OCR-2、V4-Flash、V4-Pro、V4-Flash-0731、V4-Pro-0813、V4-Flash-Vision-Exp、V4-Flash-DSpark、V4.1-Flash | 同上 `README.md` | raw 文件 |
| **HF-CODE** | V4-Flash 与 V4.1-Flash 的 `inference/model.py`、`inference/config.json` | 同上 | raw 文件，grep 量化格式 |
| **GH-ORG** | github.com/deepseek-ai 39 个仓库及 `created_at`；9 个仓库的最早提交日期 | https://api.github.com/orgs/deepseek-ai/repos | GitHub API |
| **GH-README** | 26 个仓库 README（V2 的发布条目、VL/VL2/Janus/OCR/ESFT/DeepSelect 新闻、open-infra-index 开源周） | https://github.com/deepseek-ai/<repo> | raw 文件 |
| **FLASHMLA** | FlashMLA README：V3.2 / V4 / V4.1 的每 token KV 字节格式；2026-09-10 V4.1 内核新闻 | https://github.com/deepseek-ai/FlashMLA | raw 文件（2026-09-24 版） |
| **OAI-WB** | OpenAI 四篇公告的 Wayback 快照：GPT-1（快照 2019-03-11）、GPT-2（2019-03-11）、InstructGPT（2022-01-29）、ChatGPT（2022-12-05） | http://web.archive.org/web/20221205235144/https://openai.com/blog/chatgpt/ 等 | openai.com 直连返回 403，改用 Wayback |
| **NATURE** | R1 的 Nature 文章页（收稿、接收、在线日期） | https://www.nature.com/articles/s41586-025-09422-z | HTML 元数据 |
| **ACL25** | ACL 2025 奖项页（Best Paper 列表）与 ACL Anthology 2025.acl-long.1126 | https://2025.aclweb.org/program/awards/ | HTML |
| **ACL26** | ACL Anthology 2026.acl-long.226（Engram） | https://aclanthology.org/2026.acl-long.226/ | HTML 元数据 |

### 0.4 没有读的东西（范围声明）

- DeepSeek 官方 X/微信公众号帖子：未抓取（不稳定、难以存档）。凡只在这些渠道出现的日期，一律以 API 新闻页 / GitHub / HF 为准。
- HF 上的 V4 技术报告 PDF（2026-06-22 已从 HF 仓库删除，故 404，E2）与 V4.1 技术报告 PDF（`DeepSeek_V41_Tech_Report.pdf`，1,809,802 字节）：未下载比对，以 arXiv v1 为准（§8）。

---

## 1. 核心论断与原文引文

编号规则：G = GPT 侧，T = DeepSeek 时间线，S = 主线规格，K = KV/FLOPs 代际账单，A = 被放弃或被替换的路线（第 20 课用）。

### 1.1 GPT 侧（第 0 课的“史前史”）

**G1. Transformer：arXiv v1 2017-06-12。** abs 页 Submission history：v1 “Mon, 12 Jun 2017”。NIPS 2017 见 `lesson-01-doc.md` 的参考文献卡片 vaswani2017。

**G2. GPT-1：OpenAI 公告 2018-06-11，没有 arXiv 版。** OAI-WB（快照 2019-03-11）页面日期 “June 11, 2018”，首句：
“We've obtained state-of-the-art results on a suite of diverse language tasks”。论文 PDF 链接为
`s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf`（`lesson-02-doc.md` 记该 PDF 元数据 CreationDate 为 2018-06-08；作者表见 `lesson-01-doc.md`）。

**G3. GPT-2：OpenAI 公告 2019-02-14，没有 arXiv 版。** OAI-WB（快照 2019-03-11）日期 “February 14, 2019”，首句：
“We’ve trained a large-scale unsupervised language model which generates coherent paragraphs of text”。
规格（1.5B、Pre-LN 等）已在 `lesson-01-doc.md` C17 核对。

**G4. GPT-3：arXiv v1 2020-05-28；175B；上下文 2048。** 元数据 v1 2020-05-28；“All models use a context window of $n_{ctx}=2048$ tokens”
（§2.1，见 `lesson-01-doc.md` C23）。

**G5. InstructGPT：公告与 PDF 2022-01-27，arXiv 2022-03-04。** OAI-WB（快照 2022-01-29）页面日期 “January 27, 2022”，
页面已链接 `cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf`；
首句 “We’ve trained language models that are much better at following user intentions than GPT-3”。
arXiv 2203.02155 v1 为 2022-03-04。→ 课程地图写“InstructGPT Mar 2022”只对 arXiv 成立，首次公开是 1 月（E7）。

**G6. ChatGPT：2022-11-30。** OAI-WB（快照 2022-12-05）页面日期 “November 30, 2022”；首句
“We’ve trained a model called ChatGPT which interacts in a conversational way.”

### 1.2 DeepSeek 发布时间线（逐条一手来源）

**T1. DeepSeek-Coder（最早公开的 DeepSeek 语言模型）：GitHub 最早提交 2023-11-02；论文 2024-01-25。** GH-ORG：仓库 `DeepSeek-Coder` 创建 2023-10-20（私有期），
最早提交 “init project” 2023-11-02T14:07:09Z；HF 仓库 2023-10-23 至 11-01 创建。论文摘要：
“sizes from 1.3B to 33B, trained from scratch on 2 trillion tokens”，“a fill-in-the-blank task with a 16K window”。
公开发布日取 2023-11-02 仍是**代理值**（没有官方公告页，§8）。

**T2. DeepSeek LLM 7B/67B：2023-11-29 发布，论文 2024-01-05。** GH-ORG：`DeepSeek-LLM` 仓库创建与最早提交均为 2023-11-29；
HF 四个仓库 createdAt 2023-11-29。dsv41 图 1(b) 把它标为 “DeepSeek-V1 / 2023.11”；V2 的 GitHub README 表头写
“DeepSeek-V1 (Dense-67B)”——“DeepSeek-V1”是官方后来给 DeepSeek LLM 67B 起的别名，**没有**名为 V1 的独立发布（E8）。

**T3. DeepSeekMoE 16B：论文与 GitHub 同日 2024-01-11。** arXiv v1 2024-01-11；GitHub 最早提交 “initial commit” 2024-01-11T02:35:17Z；
HF 仓库 01-08/01-09 创建（先私有）。

**T4. DeepSeek-Coder v1.5 7B：HF 2024-01-25。** dscoder §5：从 DeepSeek-LLM-7B Base 再预训练 2T tokens，“resulting in DeepSeek-Coder-v1.5 7B”（截取）。它是 DeepSeekMath 的起点（T5）。

**T5. DeepSeekMath 7B（GRPO 出处）：2024-02-05。** arXiv、GitHub、HF 三处同日。摘要：
“continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens”。

**T6. DeepSeek-VL 1.3B/7B：arXiv 2024-03-08，GitHub 标注 03-11 发布。** GH-README：“2024-03-11: DeepSeek-VL family released”（HTML 标签内文字）。

**T7. DeepSeek-V2：2024-05-06 发布，arXiv 2024-05-07。** GH-README（DeepSeek-V2）：“2024.05.06: We released the DeepSeek-V2.”；
HF 仓库 2024-04-22 创建（先私有）。**V2-Lite：“2024.05.16: We released the DeepSeek-V2-Lite.”**
API：CHANGELOG 2024-05-17 “deepseek-chat model has been upgraded to DeepSeek-V2-0517”。

**T8. DeepSeek-Prover（V1）：arXiv 2024-05-23，权重 HF 2024-08-16 才建库。** 摘要：
在 DeepSeekMath 7B 上微调，合成数据 “comprises 8 million formal statements with proofs”（截取）。

**T9. DeepSeek-Coder-V2（236B/21B 与 16B/2.4B）：HF/GitHub 2024-06-14，arXiv 06-17。** 摘要：
“further pre-trained from an intermediate checkpoint of DeepSeek-V2 with additional 6 trillion tokens”；
“extending the context length from 16K to 128K”。CHANGELOG 2024-06-14：deepseek-coder 升级为 “DeepSeek-Coder-V2-0614”。

**T10. V2-Chat-0628 与 Coder-V2-0724。** CHANGELOG：2024-06-28 “upgraded to DeepSeek-V2-0628”；2024-07-24 “upgraded to DeepSeek-Coder-V2-0724”。
NEWS 0905 交代 0628 的来历：“we upgraded DeepSeek-V2-Chat by replacing its base model with the Coder-V2-base”。
HF：V2-Chat-0628 仓库 2024-07-18；Coder-V2-Instruct-0724 仓库 2024-09-05。

**T11. ESFT：arXiv 2024-07-02；HF/GitHub 07-04；EMNLP 2024。** esft 正文：“We use the backbone architecture of DeepSeek-V2-Lite”（截取）；
GH-README：“2024.9.20: … ESFT has been accepted to the EMNLP 2024 Main Conference”（截取，原文带 emoji 与粗体）。

**T12. DeepSeek-Prover-V1.5：2024-08-15（arXiv 与 HF 同日）。** 摘要：“Pre-trained on DeepSeekMath-Base”；提出 RMaxTS（MCTS 变体），
miniF2F 测试集 “($63.5\%$)”。

**T13. Fire-Flyer AI-HPC：arXiv 2024-08-26；SC'24。** 摘要：“we deployed the Fire-Flyer 2 with 10,000 PCIe A100 GPUs”。

**T14. 无辅助损失负载均衡：arXiv 2024-08-28。** 摘要：“MoE models with up to 3B parameters trained on up to 200B tokens”。
这是 V3 的 aux-free 均衡的出处（第 11 课）。

**T15. DeepSeek-V2.5：2024-09-05。** NEWS 0905：“a powerful combination of DeepSeek-V2-0628 and DeepSeek-Coder-V2-0724”。
**V2.5-1210：2024-12-10**，NEWS 1210：“With the release of DeepSeek-V2.5-1210, the V2.5 series comes to an end.”（截取）

**T16. Janus（1.3B）：arXiv 2024-10-17；GitHub 最早提交 10-18。JanusFlow：arXiv 2024-11-12（CVPR 2025）。
Janus-Pro（1B/7B）：GitHub “2025.01.27: Janus-Pro is released”，arXiv 01-29。** Janus-Pro 正文：
“we utilize DeepSeek-LLM (1.5B and 7B) … as the base language model”（截取；1.5B 版 DeepSeek-LLM 从未单独公开，§8）。

**T17. DeepSeek-R1-Lite-Preview：2024-11-20（仅网页，无权重）。** NEWS 1120：“DeepSeek-R1-Lite-Preview is now live”；
“Open-source models & API coming soon!”。

**T18. DeepSeek-VL2（Tiny/Small/Base）：2024-12-13（arXiv、GitHub、HF 同日）。** 摘要：
“with 1.0B, 2.8B and 4.5B activated parameters respectively”；正文 §2：语言部分 “These LLMs have 0.57B, 2.4B, and 4.1B activated parameters”（E12）。

**T19. DeepSeek-V3：2024-12-26 发布（HF 仓库 12-25 建），arXiv 12-27。** NEWS 1226（标题 “Introducing DeepSeek-V3 2024/12/26”）列出
“671B MoE parameters”“37B activated parameters”“Trained on 14.8T high-quality tokens”；GitHub 最早提交 2024-12-26T09:52:41Z。

**T20. DeepSeek-R1 与 R1-Zero、6 个蒸馏模型：2025-01-20，arXiv 01-22；Nature 2025-09-17。** NEWS 250120（需加尾斜杠访问）：
“Distilled from DeepSeek-R1, 6 small models fully open-sourced”；HF 八个仓库同日 01-20 建。dsr1 v2：
“DeepSeek-R1 and DeepSeek-R1-Zero are trained on top of DeepSeek-V3-Base”。NATURE：Received 14 February 2025，Accepted 17 July 2025，
Published 17 September 2025；卷 645、首页 633。

**T21. 开源周：2025-02-24 起每日一个仓库。** GH-README（open-infra-index）：“Starting **this week** , Feb 24, 2025 we'll open-source 5 repos”；
Day 1 FlashMLA、Day 2 DeepEP、Day 3 DeepGEMM、Day 4 DualPipe + EPLB + profile-data、Day 5 3FS + smallpond、Day 6 推理系统概述
（“Cost profit margin **545%**”）。GH-ORG 创建日期：DeepGEMM 02-13、DeepEP 02-17、FlashMLA 02-21、DualPipe/EPLB 02-26、3FS 02-27（均为私有创建日）。

**T22. CodeI/O：2025-02-11（ICML 2025）。NSA：2025-02-16（ACL 2025 Best Paper）。** ACL25 奖项页 “Best Paper” 下列出
“Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention”（与另 3 篇并列）。NSA 实验骨干：
“featuring 27B total parameters with 3B active parameters”。CodeI/O 的作者单位含 DeepSeek-AI 与 HKUST（数据与模型发布在 hkust-nlp 名下）。

**T23. DeepSeek-V3-0324：HF 与 CHANGELOG 2025-03-24；新闻页标题 2025/03/25。** NEWS 250325：“Models are now released under the MIT License”（截取）。

**T24. DeepSeek-GRM（SPCT）：arXiv 2025-04-03。** 权重放在个人空间 `huggingface.co/collections/BBQGOD/…`，不在 deepseek-ai 组织下（arXiv comments）。

**T25. DeepSeek-Prover-V2（671B 与 7B）：2025-04-30。** 摘要：“reaching 88.9% pass ratio on the MiniF2F-test”；README：
“DeepSeek-Prover-V2-671B is trained on top of DeepSeek-V3-Base.”

**T26. Insights into DeepSeek-V3（ISCA '25）：arXiv 2025-05-14。** 摘要：“DeepSeek-V3, trained on 2,048 NVIDIA H800 GPUs”。

**T27. DeepSeek-R1-0528：2025-05-28；R1-0528-Qwen3-8B 05-29。** CHANGELOG 2025-05-28 给出 AIME 2025 “70.0 → 87.5 (+17.5)”。

**T28. DeepSeek-V3.1：V3.1-Base HF 2025-08-19，V3.1 2025-08-21。** NEWS 250821：“Hybrid inference: Think & Non-Think — one model, two modes”；
“V3.1 Base: 840B tokens continued pretraining for long context extension on top of V3”。HF-CARD（V3.1）：
“The 32K extension phase has been increased 10-fold to 630B tokens”，“the 128K extension phase has been extended by 3.3x to 209B tokens”；
“trained using the **UE8M0 FP8 scale data format**”（截取）。

**T29. DeepSeek-V3.1-Terminus：2025-09-22。** CHANGELOG：“Language consistency: Reduced occurrences of Chinese-English mixing”（截取）。

**T30. DeepSeek-V3.2-Exp（DSA 首发）：2025-09-29。** NEWS 250929：“Built on V3.1-Terminus, it debuts DeepSeek Sparse Attention (DSA)”；
“API prices cut by 50%+!”。技术报告只在 GitHub（`DeepSeek_V3_2.pdf`，无 arXiv 编号；HF 卡引用标题
“DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention”）。

**T31. DeepSeek-OCR：GitHub “2025/10/20 … We release DeepSeek-OCR”，arXiv 10-21。** 摘要：
“compression ratio < 10x), the model can achieve decoding (OCR) precision of 97%”；解码器 “DeepSeek3B-MoE-A570M”。

**T32. LPLB（线性规划专家负载均衡器，早期研究）：GitHub 2025-11-19 建库，无论文。**

**T33. DeepSeekMath-V2：2025-11-27（arXiv 与 HF 同日）。** HF-CARD：“DeepSeekMath-V2 is built on top of DeepSeek-V3.2-Exp-Base.”；
摘要：“a near-perfect 118/120 on Putnam 2024 with scaled test-time compute”（截取）。

**T34. DeepSeek-V3.2 与 V3.2-Speciale：2025-12-01；论文 arXiv 12-02。** NEWS 251201：
“V3.2-Speciale attains gold-level results in IMO, CMO, ICPC World Finals & IOI 2025.”；Speciale 临时端点至 2025-12-15 15:59 UTC。
dsv32 §2.1：以 “a base checkpoint of DeepSeek-V3.1-Terminus” 为起点续训（截取）。

**T35. mHC：arXiv 2025-12-31，无代码仓库。** 正文：“with a marginal training overhead of only 6.7%”（n = 4，§4.3 Efficient Infrastructure Design 首段）。

**T36. Engram：arXiv 与 GitHub 2026-01-12；ACL 2026。** 摘要：“we scale Engram to 27B parameters”。ACL26：pp. 4968–4990，出版月 2026/7。

**T37. DeepSeek-OCR 2：GitHub/HF 2026-01-27，arXiv 01-28。** 摘要：“DeepEncoder V2-capable of dynamically reordering visual tokens”（截取，原文无空格）。

**T38. DeepSeek-V4 预览版（V4-Pro 1.6T/49B、V4-Flash 284B/13B）：2026-04-24。** NEWS 260424：
“DeepSeek-V4 Preview is officially live & open-sourced!”；“1M context is now the default across all official DeepSeek services.”；
HF-COMMITS：V4-Flash 仓库 “Release DeepSeek-V4” 提交于 2026-04-24T02:24:45Z；arXiv 提交 04-26（编号 2606.19348，E1）。
旧 API 名 deepseek-chat / deepseek-reasoner 定于 2026-07-24 下线（CHANGELOG 2026-04-24）。

**T39. DeepSpec / DSpark（投机解码）：GitHub 2026-06-26，HF `V4-Flash-DSpark`、`V4-Pro-DSpark` 06-27，arXiv 07-06。**
HF-CARD（V4-Flash-DSpark）：“**DeepSeek-V4-Flash-DSpark** is **not** a new model.”；dspark 摘要：相对生产基线 MTP-1，
“accelerates per-user generation speeds by 60 to 85 percent”。**限定**（核查补，§10）：正文 §1 把区间分开写——V4-Flash 为 “60%–85%”，V4-Pro 为 “57%–78%”（均为 “at matched aggregate throughput capacities”）；摘要的 60–85% 只对应 Flash。

**T40. V4-Flash 正式版（0731）：2026-07-31。** CHANGELOG 2026-07-31：“The official release of the DeepSeek-V4-Flash API is now in public beta.”；
“keeps the same model architecture and size as DeepSeek-V4-Flash-Preview, and was only re-post-trained”（截取）。

**T41. V4-Pro 正式版（0813，GA）：2026-08-13。** NEWS 260813：“We’re launching DeepSeek-V4-Pro today!”；推理强度 “low / high / max” 三档，
峰谷定价自 2026-08-16 16:00 UTC 生效。HF-CARD（V4-Pro-0813）：“built on the DeepSeek-V4-Pro (Preview) model structure”（截取）。

**T42. V4-Flash-Vision-Exp（首个 V4 多模态）：API 2026-08-21，HF 仓库 08-31 建。** NEWS 260821：“This experimental multimodal model matches
DeepSeek-V4-Flash on text capabilities”（截取）；图像 “up to 384 tokens each”。

**T43. DeepSeek-V4.1-Flash：2026-09-10 发布（API、HF、FlashMLA 内核同日），arXiv 09-17。** NEWS 260910：“552B-parameter MoE.”；
“just 8B active parameters for input, 16B for output”；“1/4 the HBM”“1/8 the SSD storage”；“Set your model to deepseek-flash.”；
V4-Flash 与 V4-Flash-Vision-Exp 退役。FLASHMLA：“2026.09.10 Release of DeepSeek v4.1's Attention Kernels”。

**T44. DSec（沙箱基础设施）：arXiv 2026-09-19。** 摘要：“serving about 3 million sandboxes per day”；单位 DeepSeek-AI（部分作者兼清华）。

### 1.3 主线规格（每条都有论文原文）

**S1. DeepSeek LLM 67B：95 层、$d_{model}$ 8192、64 头、8 个 KV 头、上下文 4096、2.0T tokens。** dsllm 表 2 行
“67B | 95 | 8192 | 64 | 8 | 4096 | 4608 | 3.2e-4 | 2.0T”（列：Params、$n_{layers}$、$d_{model}$、$n_{heads}$、$n_{kv\_heads}$、Context Length、
Sequence Batch Size、Learning Rate、Tokens）；§2.2：“the 67B model uses Grouped-Query Attention (GQA)”。7B 行：30 层、4096、32/32 头、4096、2304、4.2e-4、2.0T。

**S2. DeepSeekMoE 16B：28 层、隐藏维 2048、16 头 × 128、2 共享 + 64 路由、激活 6 个、16.4B/2.8B、2T tokens、4K。** dsmoe §5.1.2（Hyper-Parameters）：
“we set the number of Transformer layers to 28 and the hidden dimension to 2048”；“2 shared experts and 6 out of 64 routed experts”（截取）；
“approximately 16.4B total parameters”“activated parameters around 2.8B”。上下文 4K 见 dsv2 附录 B 表（“4K 4K 32K” 三列）与 config。

**S3. DeepSeek-V2：60 层、5120、MLA（$n_h$ = 128，$d_h$ = 128，$d_c$ = 512，$d_c'$ = 1536，$d_h^R$ = 64）、2 共享 + 160 路由激活 6、236B/21B、8.1T、128K。**
dsv2 §3.1.2 逐句给出（“We set the number of Transformer layers to 60 and the hidden dimension to 5120.” 等）；
摘要 “236B total parameters, of which 21B are activated for each token”，并支持 128K 上下文（截取）。

**S4. DeepSeek-V2-Lite：27 层、2048、16 头 × 128、$d_c$ = 512、**不压缩查询**、$d_h^R$ = 64、2 + 64 激活 6、15.7B/2.4B、5.7T、32K。**
dsv2 附录 B.1：“DeepSeek-V2-Lite has 27 layers and a hidden dimension of 2048.”；“it does not compress the queries”；
B.1 “Training Details” 段：“train DeepSeek-V2-Lite on 5.7T tokens”。GitHub README 表写 “16B | 2.4B | 32k”。

**S5. DeepSeek-V3：61 层、7168、MLA 同 V2 维度、1 共享 + 256 路由激活 8、前 3 层稠密、MTP 深度 1、671B/37B、14.8T、128K。** dsv3 §4.2：
“We set the number of Transformer layers to 61 and the hidden dimension to 7168.”；“8 experts will be activated for each token”（截取）；
“The multi-token prediction depth $D$ is set to 1”（截取）。两阶段扩到 32K、128K（§1）。

**S6. DeepSeek-V3.2：结构同 V3 + DSA（indexer 64 头 × 128，top-k 2048），128K。** dsv32 §2.1：稀疏训练阶段
“select 2048 key-value tokens for each query token”；“resulting in a total of 943.7B tokens”；§4：“supports a maximum context length of only 128K”。
indexer 头数/维度取自 HF-CFG（`index_n_heads` 64、`index_head_dim` 128、`index_topk` 2048）。

**S7. DeepSeek-V4-Flash：43 层、4096、前 2 层纯 SWA、其余 CSA（m = 4，top-k 512）与 HCA（m' = 128）交替、64 头、头维 c = 512、
查询压缩 1024、输出分 8 组、SWA 窗口 128、1 共享 + 256 路由（中间维 2048）激活 6、前 3 个 MoE 层哈希路由、MTP 深度 1、mHC $n_{hc}$ = 4、284B/13B、32T、1M。**
dsv4 §4.2.1 逐句（“number of Transformer layers to 43 and the hidden dimension $d$ to 4096”；
“DeepSeek-V4-Flash comprises 284B total parameters, of which 13B are activated for each token.”）。

**S8. DeepSeek-V4-Pro：61 层、7168、前 2 层 HCA、其余 CSA（top-k 1024）/HCA 交替、128 头、c = 512、查询压缩 1536、输出分 16 组、
1 + 384 路由（中间维 3072）激活 6、1.6T/49B、33T、1M。** dsv4 §4.2.1（“For the first two layers, we use HCA.”；
“384 routed experts, where the intermediate hidden dimension of each expert is 3072”）；§1：
“we train DeepSeek-V4-Flash on 32T tokens and DeepSeek-V4-Pro on 33T tokens”。

**S9. DeepSeek-V4.1-Flash：40 层（因果编码器 20 + 解码器 20）、5120、前 2 层纯 SWA；编码器其余 18 层 CSA2（m = 2，三组各 1 Full + 5 Reuse），
解码器 20 层 CSA2（m = 1，五组各 4 层：第 1 组 Full + 3 Reuse，其余 4 组 Reindex + 3 Reuse）；indexer 32 头 × 128、top-k 512、
候选池 2048 块 × 8；64 头、c = 512、查询压缩 1280、输出 8 组；SWA 128；1 + 384 路由（2304）激活 6；Engram 196B；552B backbone；
prefill 8B / decode 16B 激活；45T 多模态 tokens；1M。** dsv41 §4.2.1 与 §2.1：
“552B backbone parameters and 196B Engram parameters”；“with 8B activated per token during prefill and 16B during decode”；
§4.2.2：“We train DeepSeek-V4.1-Flash on 45T tokens of multimodal data with no instability.”

**S10. 版本族的 base 血缘（写族谱图用）。** Coder-v1.5 7B ← DeepSeek-LLM-7B Base（T4）；DeepSeekMath 7B ← Coder-v1.5 7B（T5）；
Prover-V1 ← DeepSeekMath 7B（T8）；Prover-V1.5 ← DeepSeekMath-Base（T12）；Coder-V2 ← V2 中间检查点（T9）；V2-Chat-0628 ← Coder-V2-base（T10）；
V2.5 = V2-0628 + Coder-V2-0724（T15）；R1/R1-Zero ← V3-Base（T20）；Prover-V2-671B ← V3-Base（T25）；V3.1-Base ← V3 base（T28）；
V3.2-Exp ← V3.1-Terminus（T30）；V3.2 ← V3.1-Terminus 检查点（T34）；Math-V2 ← V3.2-Exp-Base（T33）；V4-Flash-0731 ← V4-Flash 预览版重做后训练（T40）；
V4-Flash-Vision-Exp ← V4-Flash 结构加视觉模块续训（T42）；VL2 语言部分 ← DeepSeekMoE 3B/16B/27B（T18）；ESFT ← V2-Lite 骨干（T11）；
OCR 解码器 = DeepSeek3B-MoE-A570M（T31）。**V4.1-Flash 从头训练**（HF-CARD：“trained from scratch on a multimodal corpus”，截取）。

### 1.4 KV cache 与 FLOPs 的代际账单

**K1. V2 表 1 的通式与 “2.25 组 GQA”。** dsv2 表 1 注：“For DeepSeek-V2, $d_c$ is set to $4d_h$ and $d_h^R$ is set to $\frac{d_h}{2}$.”；
“So, its KV cache is equal to GQA with only 2.25 groups”。

**K2. V2 “KV cache 降 93.3%” 依赖 6 bit 量化。** 摘要：“reduces the KV cache by 93.3%”；§3.2.3（Training and Inference Efficiency）：
“compress each element in its KV cache into 6 bits on average”。复算见 N6（BF16 对 BF16 只有 82.2%）。与 `lesson-09-doc.md` 一致。

**K3. V3.2 的 KV 字节格式：MLA 每 token 每层 656 字节 + indexer 键 132 字节。** FLASHMLA：“a page block is `page_block_size`
token-major rows of 656 Bytes each”（512 FP8 + 16 字节 4 个 FP32 缩放 + 128 字节 64 个 BF16 RoPE）；indexer 键 132 = 128 FP8 + 4 字节 FP32 缩放
（`lesson-14-doc.md` C39 与其 §3 字节表已核）。61 层合计 48,068 字节，恰为 dsv41 图 1(b) 的 V3.2 柱长（N2）。

**K4. V4 的 KV 条目格式：RoPE 维 BF16、其余 FP8；每条 584 字节。** dsv4 §2.3.4：“BF16 precision is used for the rotary positional embedding (RoPE) dimensions”；
FLASHMLA：“**V4**: 584 Bytes per token.”——数据行 448 字节 FP8（NoPE）+ 128 字节 BF16（64 维 RoPE），另 8 字节缩放行（7 个 E8M0 + 1 字节填充）。indexer 键为 FP4：dsv4 §1 “FP4 quantization-aware training for MoE expert weights and the indexer QK path”（截取）。

**K5. V4 相对 V3.2 的 1M 上下文效率。** dsv4 §1：V4-Pro “only 27% of the single-token FLOPs (measured in equivalent FP8 FLOPs)” 与 10% KV（截取），V4-Flash “achieves only 10% of the single-token FLOPs and 7% of the KV cache size”（截取）；图 1 右标注 “3.7× lower / 9.8× lower”（FLOPs）
与 “9.5× smaller / 13.7× smaller”（KV）。复算 N3、N4。

**K6. V4 相对 “BF16 GQA8、头维 128” 基线约 2%。** dsv4 §2.3.4：“approximately 2% times of that baseline in the 1M-context setting”（原文如此，“2% times”）。复算 N5。

**K7. V4.1 的全局 KV 890 字节/token，约 V4-Flash 的 1/4；持久化 KV 约 1/8。** dsv41 摘要：
“reduce its global KV cache footprint (always in HBM) to 890 bytes per token”；§3.2.1 给出 1/8 的因式：持久化 KV 不再存 SWA KV
（“which almost halves its size”）× 全局 KV 压到 1/4。图 1(b) 柱长（目视读数，数字印在图上）：DeepSeek-V1 389,120（2023.11）、
DeepSeek-V3.2 48,068（2025.12）、DeepSeek-V4-Flash 3,514（2026.04）、DeepSeek-V4.1-Flash 890（2026.09），箭头标 “8.1× / 13.7× / 3.9× smaller”；
图注：“approximately 4-fold and 437-fold reductions in per-token global KV cache size”。

**K8. V4.1 的 FP4 主 KV 格式：E2M1，每 16 通道一个 E4M3 缩放，每条 288 字节；indexer 为 MXFP4（每 32 个一个 E8M0）。** dsv41 §2.4.4：
“we select E2M1 with one E4M3 scale per 16 channels”；FLASHMLA：“**V4.1 fp4**: 288 Bytes per token”；HF-CODE（V4.1 `model.py` 第 759 行注释）：
“Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.” → indexer 键 128 维 = 64 字节 + 4 字节缩放 = 68 字节（推算，N1 验证它与 3,514、890 同时吻合）。
论文本身也给了格式（核查补）：dsv41 §2.4.4 “DeepSeek-V4 already uses quantization-aware training (QAT) … for FP4 indexer queries and keys”，并称 “We adopt the OCP-standard MXFP4 format”（OCP MX 规范即每 32 个元素共享一个 E8M0 缩放）——因此 68 字节的推算有论文 + 代码两条依据，但 “68” 这个数仍未被直接印出（§8）。

**K9. 两张“单 token FLOPs”图的单位不同。** dsv4 图 1 用 “equivalent FP8 FLOPs”；dsv41 图 2 图注：
“weighting BF16, FP8, and FP4 operations by 1, 0.5, and 0.25”（BF16 当量）。同一个 V3.2 在 1M 处，前者约 1.2 T、后者约 0.6 T（目视），相差 2 倍正是单位差（E4）。

**K10. V4.1 decode FLOPs 几乎不随上下文增长。** dsv41 §1：上下文从 4K 到 1M（256 倍）“increases its Decode FLOPs by only 1/4”。

### 1.5 被放弃或被替换的路线（第 20 课）

**A1. PRM 与 MCTS（R1）。** dsr1 v2 附录 G.2（v1 为 §4.2）“Unsuccessful Attempts”：“this does not imply that these approaches are incapable of developing effective reasoning models”；
PRM 三条限制（步难定义、步对错难判、奖励黑客）；MCTS：搜索空间指数大、细粒度价值模型难训。对照：Prover-V1.5 在形式证明里**用了** MCTS 变体 RMaxTS（T12）。

**A2. token 丢弃（V2 用，V3 弃）。** dsv2 §2.2.4：“we introduce a device-level token-dropping strategy during training”；dsv3 §2.1.2：
“DeepSeek-V3 does not drop any tokens during training.”

**A3. 辅助损失 → 无辅助损失 + 极小序列级损失（V3 起），V4 仍保留。** dsv3：“we also employ a complementary sequence-wise balance loss”（截取）；
dsv4 §2.1：aux-free “augmented by a slight sequence-wise balance loss”（截取）；V4/V4.1 的损失权重 0.0001（§4.2.2）。

**A4. 节点受限路由（V3 的 M = 4）在 V4 取消。** dsv4 §2.1：“we remove the constraint on the number of routing target nodes”。

**A5. 前几层稠密 FFN → 哈希路由 MoE（V4）→ “All feed-forward layers use standard DeepSeekMoE”（V4.1 图 3 图注）。** dsv4 §2.1：前几层改为
“MoE layers that employ Hash routing”（截取）；V4.1 的 config 没有 `num_hash_layers` 字段（V4 两个 config 都是 3）。V4.1 是否完全取消哈希路由，正文未明说（§8）。

**A6. MTP（V3 引入、V4 沿用）→ V4.1 预训练不再带 MTP，改用单独训练的 DSpark。** dsv41 §2.1：“We omit the MTP module during backbone pre-training and use DSpark”（截取）。

**A7. CSA/HCA 混合 → 纯 CSA2；CSA 的重叠压缩与绝对位置偏置被删。** dsv41 §1：“DeepSeek-V4.1-Flash uses pure CSA2”；§2.3：“CSA2 removes this overlap and absolute positional embedding.”

**A8. V4 提出的 “Zero SWA Caching” 在生产中太贵，V4.1 改为近似的 SWA Bounded Replay。** dsv41 §3.2.1：精确恢复要跑 $L\times n_{win}$ 个 token，
“whose cost proved prohibitive in production deployments”。

**A9. 稠密注意力热身（V3.2、V4）→ V4.1 从头稀疏。** dsv41 §1：在 64K 序列长度上从头训练稀疏注意力，“without any dense attention warmup stages”（截取）

**A10. 后训练算法创新到 V4.1 暂停。** dsv41 §1：“our post-training introduces no algorithmic innovation”（改动全在数据与环境）。

**A11. Q-norm 只在 V4 用。** FLASHMLA：“Q-norm (only used in V4, not V4.1)”。

---

## 2. 公式（原文转录 + 课程记号对照）

第 0 课只需要“跨代记账”的公式；第 20 课复盘要把每代的关键式并排放。机制推导留给各课。下面先转录原文（式号为原文式号），
再给课程记号（`style-guide.md` §2）对照。凡是**笔者整理**而非论文印刷的式子，单独标出。

### 2.1 KV cache 每 token 元素数（dsv2 表 1，原文转录）

| Attention Mechanism | KV Cache per Token (# Element) | Capability |
|---|---|---|
| Multi-Head Attention (MHA) | $2n_{h}d_{h}l$ | Strong |
| Grouped-Query Attention (GQA) | $2n_{g}d_{h}l$ | Moderate |
| Multi-Query Attention (MQA) | $2d_{h}l$ | Weak |
| MLA (Ours) | $(d_{c}+d_{h}^{R})l\approx\frac{9}{2}d_{h}l$ | Stronger |

表注原文：KV 量 “measured by the number of elements, regardless of the storage precision”。
**记号对照**：与课程符号表完全一致（$n_h,d_h,n_g,l,d_c,d_h^R$）。换成字节要乘每元素字节数 $b$（BF16 为 2）。
V2 取 $d_c=4d_h=512$、$d_h^R=d_h/2=64$，故 $(512+64)/(2\cdot128)=2.25$ 组 GQA（N7）。

### 2.2 DSA 的 lightning indexer 与 top-k 选择（dsv32 式 (1)(2)）

$$I_{t,s}=\sum_{j=1}^{H^{I}}w_{t,j}^{I}\cdot\text{ReLU}\left(\mathbf{q}^{I}_{t,j}\cdot\mathbf{k}^{I}_{s}\right),\tag{1}$$

$$\mathbf{u}_{t}=\text{Attn}\left(\mathbf{h}_t,\ \left\{\mathbf{c}_s\ \middle|\ I_{t,s}\in\text{Top-k}\left(I_{t,:}\right)\right\}\right).\tag{2}$$

**记号对照**：$H^I$（indexer 头数，V3.2 = 64）、$d^I$（indexer 头维，128）是 V3.2 记号；V4 把同两个量写成 $n_h^I$、$c^I$（§2.3 式 (16)）。
课程建议统一用 V3.2 的 $H^I,d^I$，在第 18 课首次出现 V4 记号时注明对应。$\mathbf u_t$ 与课程 “注意力层输出” 同义。

### 2.3 DeepSeek-V4（dsv4 v1）

**(a) Hyper-Connections 更新（dsv4 式 (1)）**

$$X_{l+1}=B_{l}X_{l}+C_{l}\mathcal{F}_{l}(A_{l}X_{l}),\tag{1}$$

其中 $X_l\in\mathbb{R}^{n_{\text{hc}}\times d}$，$A_l\in\mathbb{R}^{1\times n_{\text{hc}}}$，$B_l\in\mathbb{R}^{n_{\text{hc}}\times n_{\text{hc}}}$，$C_l\in\mathbb{R}^{n_{\text{hc}}\times 1}$。
**记号对照**：mHC 论文（2512.24880 式 (3)）写作 $\mathbf{x}_{l+1}=\mathcal{H}_{l}^{\mathrm{res}}\mathbf{x}_{l}+\mathcal{H}_{l}^{\mathrm{post}\,\top}\mathcal{F}(\mathcal{H}_{l}^{\mathrm{pre}}\mathbf{x}_{l},\mathcal{W}_{l})$，
故 $A_l\leftrightarrow\mathcal H_l^{pre}$，$B_l\leftrightarrow\mathcal H_l^{res}$，$C_l\leftrightarrow\mathcal H_l^{post\top}$，$n_{hc}\leftrightarrow n$，V4 的 $d\leftrightarrow$ mHC 的 $C$。
课程第 16 课沿用 mHC 论文的 $H^{pre},H^{post},H^{res}$（课程地图如此写），第 18 课引 V4 时给这张对照。V4 取 $n_{hc}=4$、Sinkhorn 迭代 $t_{max}=20$。

**(b) CSA 的压缩（dsv4 式 (11)(12)，$m$ 为压缩率）**

$$[S^{a}_{mi:m(i+1)-1};S^{b}_{m(i-1):mi-1}]=\operatorname{Softmax}_{\text{row}}([Z^{a}_{mi:m(i+1)-1}+B^{a};Z^{b}_{m(i-1):mi-1}+B^{b}]),\tag{11}$$

$$C^{\text{Comp}}_{i}=\sum_{j=mi}^{m(i+1)-1}S^{a}_{j}\odot C^{a}_{j}+\sum_{j=m(i-1)}^{mi-1}S^{b}_{j}\odot C^{b}_{j},\tag{12}$$

原文随后说明每个压缩条目来自 $2m$ 个原始条目、相邻条目有重叠，但 “CSA in fact compresses the sequence length to $\frac{1}{m}$ times”。
**(c) CSA 的 indexer 与 top-k（式 (16)(17)）**：$I_{t,s}=\sum_{h=1}^{n_{h}^{I}}w_{t,h}^{I}\cdot\text{ReLU}\left(\mathbf{q}^{I}_{t,h}\cdot K^{\text{IComp}}_{s}\right)$（16），
$\mathcal{C}^{\text{SprsComp}}_{t}=\left\{C^{\text{Comp}}_{s}\ \Big|\ I_{t,s}\in\operatorname{Top-k}(I_{t,:})\right\}$（17）——即式 (2.2) 的 DSA 作用在压缩后的条目上。

**(d) HCA 的压缩（式 (22)(23)，$m'\gg m$，不重叠、不稀疏）**

$$S_{m'i:m'(i+1)-1}=\operatorname{Softmax}_{\text{row}}(Z_{m'i:m'(i+1)-1}+B),\qquad C^{\text{Comp}}_{i}=\sum_{j=m'i}^{m'(i+1)-1}S_{j}\odot C_{j}.\tag{22–23}$$

**(e) Attention sink（式 (27)）**：$s_{h,i,j}=\dfrac{\operatorname{Exp}(z_{h,i,j})}{\sum_{k}\operatorname{Exp}(z_{h,i,k})+\operatorname{Exp}(z^{\prime}_{h})}$。

**(f) Muon（dsv4 算法 1，第 4–8 行）**：$G_t=\nabla_W\mathcal L_t(W_{t-1})$；$M_t=\mu M_{t-1}+G_t$；
$O'_t=\operatorname{HybridNewtonSchulz}(\mu M_t+G_t)$；$O_t=O'_t\cdot\sqrt{\max(n,m)}\cdot\gamma$；$W_t=W_{t-1}\cdot(1-\eta\lambda)-\eta O_t$。
超参数（§4.2.2）：动量 0.95、权重衰减 0.1、更新 RMS 重标到 0.18；嵌入、预测头、RMSNorm 权重仍用 AdamW（$\beta_1=0.9,\beta_2=0.95,\varepsilon=10^{-20}$）。

**V4 记号冲突（写第 18 课必须处理）**：
1. V4 的 $d_c$ 是**查询**压缩维（Flash 1024、Pro 1536），而 V2/V3 与课程的 $d_c$ 是 **KV** 压缩维（512），查询压缩维是 $d_c'$。→ 课程写 V4 时用 $d_c'$。
2. V4 的 $c$ 是“头维 = KV 条目维”（512，末 64 维施 RoPE），同时充当 K 与 V（shared-KV MQA）。它在数值上等于 MLA 的 $d_c$，但 RoPE 维**含在** 512 之内，
   而 MLA 是 $d_c+d_h^R=576$ 追加。建议课程保留 $c$ 并在首次出现时点明差别，不要复用 $d_c$。
3. $m,m'$（压缩率）、$n_{win}$（SWA 窗口）、$n_{hc}$（残差流条数）是课程符号表没有的新符号，需补进 `style-guide.md` §2（§8 待办）。
4. $B_l$（式 (1) 的残差矩阵）与 $B^a,B^b,B$（式 (11)(22) 的位置偏置）同字母，课文中应改写或加注。

### 2.4 DeepSeek-V4.1-Flash（dsv41 v1）

**(a) 因果编码器–解码器（式 (1)）**

$$C_{l}=H_{L/2}W_{l}^{KV},\quad Z_{l}=H_{L/2}W_{l}^{Z},\quad l>\frac{L}{2},\tag{1}$$

即解码器各层的全局 KV（及压缩权重）都从第 $L/2$ 层（编码器最后一层）的隐藏状态投影出来；SWA 的 K、V 仍来自本层 $H_l$。
§2.2 给出 prefill 复杂度：$\mathcal{O}(NL)\;\to\;\mathcal{O}(NL/2+n_{\mathrm{win}}\times L/2)\approx\mathcal{O}(NL/2)$（$N\gg n_{win}$）。

**(b) Single-Pass mHC（式 (2) → 式 (6)）**

$$X_{l+1}=B_{l}X_{l}+C_{l}\,\mathcal{F}_{l}(A_{l}X_{l}),\qquad(A_{l},B_{l},C_{l})=\mathcal{H}(X_{l}),\tag{2}$$

$$X_{l+1}=B_{l}X_{l}+C_{l}\,\mathcal{F}_{l}(A_{l-1}X_{l}),\qquad(A_{l},B_{l},C_{l})=\mathcal{H}(X_{l}).\tag{6}$$

区别只在输入混合用上一块的 $A_{l-1}$。激活内存读写量（§2.4.1）：理想映射下界 $(2n+2)d$；V4 的三核实现 $(4n+4)d$（下界的 2 倍）；两遍实现 $(3n+2)d$；
Single-Pass + Mega-mHC 达到 $(2n+2)d$。

**(c) FP4 主 KV 的动态范围论证（§2.4.4）**：E2M1 最大 6、E4M3 缩放最大 448，“the format supports magnitudes up to $448\times 6=2688$”；
**前提**（核查补）：论文先说明 “the largest trained RMSNorm weight magnitude is approximately 1”，在此前提下 RMSNorm 后 512 维潜向量的 L2 范数至多约 $\sqrt{512}$，
RoPE 保范数，所以旋转后各通道的最大绝对值也不超过约 $\sqrt{512}\approx 22.6$；训练中观测最大值约 10，故省略 NVFP4 的第二级全局缩放 “causes no measurable decrease in accuracy”。
（若 RMSNorm 权重远大于 1，这个上界就不成立——转述时不能丢掉这个假设。）

**(d) SWA Bounded Replay（§3.2.2）**：从位置 $s$ 开始回放时，位置 $i$ 的查询只看 SWA 键 $[\max(s,\,i-W+1),\,i]$；
精确重建需回放 $L\times n_{win}$ 个 token，有界回放只回放 $n_{win}$ 个。

### 2.5 跨代 KV 记账公式（**笔者整理**，由论文描述 + FlashMLA 字节格式推出，经 §4 N1 与论文图 1(b) 四个数全部吻合）

记 $B_{kv}$ 为每 token 的**全局** KV 字节数（随上下文线性增长的那部分；SWA 窗口与未满一组的尾巴是每请求常数，不计入）。

$$\text{稠密 MHA/GQA：}\quad B_{kv}=2\,n_{kv}\,d_h\,l\,b\qquad(n_{kv}=n_h\text{ 为 MHA},\ n_{kv}=n_g\text{ 为 GQA})\tag{0.1}$$

$$\text{MLA（V2/V3）：}\quad B_{kv}=(d_c+d_h^R)\,l\,b\tag{0.2}$$

$$\text{MLA + DSA（V3.2）：}\quad B_{kv}=l\,(B_{\text{MLA}}+B_{\text{idx}}),\quad B_{\text{MLA}}=656,\ B_{\text{idx}}=132\tag{0.3}$$

$$\text{CSA/HCA（V4）：}\quad B_{kv}=\sum_{\ell\in\text{CSA}}\frac{B_{\text{main}}+B_{\text{idx}}}{m}+\sum_{\ell\in\text{HCA}}\frac{B_{\text{main}}}{m'},\quad B_{\text{main}}=584,\ B_{\text{idx}}=68\tag{0.4}$$

$$\text{CSA2（V4.1）：}\quad B_{kv}=\sum_{\ell\in\text{Full}}\frac{B_{\text{main}}+B_{\text{idx}}}{m_\ell},\quad B_{\text{main}}=288\ (\text{FP4}),\ B_{\text{idx}}=68\tag{0.5}$$

单请求累计：$M(T)\approx T\cdot B_{kv}$（再加 SWA 等常数项）。式 (0.5) 只对 Full 模式层求和，因为 Reindex/Reuse 层复用前面 Full 层的主 KV 与 indexer K
（dsv41 §2.3.1；config `kv_source_layer_ids` = [2, 8, 14, 20] 正好 4 层）。
**说明**：$B_{\text{idx}}=68$ 是推算值（128 维 FP4 = 64 字节 + 每 32 维一个 E8M0 = 4 字节），论文与 FlashMLA README 都没有直接印这个数（§8）；
依据是 dsv41 §2.4.4 称 V4 起 indexer Q/K 用 OCP 标准 MXFP4，以及 V4.1 `model.py` 注释 “the indexer uses 32 with E8M0”（K8）。

### 2.6 单 token decode FLOPs 的粗算（**笔者整理**）

Kaplan 2020 的前向式 $C_{forward}\approx2N+2n_{layer}n_{ctx}d_{attn}$（`lesson-01-doc.md` C33）。解码第 $T$ 个 token 时，笔者把两个与 $T$ 成正比的矩阵乘都计入：
$$C_{decode}(T)\approx 2N_{act}+4\,n_h d_h\,T\,l\qquad(QK^\top\text{ 与 }AV\text{ 各 }2n_hd_hT\text{ 每层})\tag{0.6}$$
对 DeepSeek LLM 67B（BF16，权重 1）：$T=4096$ 时 146.8 GFLOPs，$T=2^{20}$ 时 3,398 GFLOPs，与 dsv41 图 2 的 “DeepSeek-V1” 曲线目视读数（约 150 与约 3,400）吻合（N12）；
若只用 Kaplan 的 $2n_{layer}n_{ctx}d_{attn}$（笔者理解为只计了一个与上下文成正比的矩阵乘），1M 处只有约 1,766 GFLOPs，对不上。→ 图 2 的记账包含 QK 与 AV 两项。
其余曲线（V3.2、V4-Flash、V4.1）涉及 FP8/FP4 权重折算与稀疏选择，论文没给逐项公式，本笔记只给目视读数（§5.1）。

### 2.7 训练算力的数量级（**笔者整理**，仅作量级比较）

$C\approx 6N_{act}D$（Kaplan/Chinchilla 口径，见 `lesson-02-doc.md`）。MoE 用激活参数；V4.1 用 decode 口径 16B（训练时每个位置都要算解码器，笔者推测）。
结果见 N20；这只是量级，DeepSeek LLM 自己主张用 “非嵌入 FLOPs/token $M$” 代替 $6N$（第 6 课），不能拿 6ND 与论文的 GPU 小时互相换算。

---

## 3. 规格与超参数表

### 3.1 GPT 侧时间线

| 事件 | 日期（UTC） | 编号 / 链接 | 规格要点 | 来源 |
|---|---|---|---|---|
| Transformer | 2017-06-12（arXiv v1） | 1706.03762 | base 65M，$d_{model}$ 512，$d_{ff}$ 2048 | G1；lesson-01-doc C11 |
| GPT-1 | 2018-06-11（公告） | 无 arXiv；OpenAI PDF | 12 层、768、上下文 512（lesson-01-doc 表） | G2；OAI-WB |
| GPT-2 | 2019-02-14（公告） | 无 arXiv；OpenAI PDF | 最大 1.5B（表 2 印 1542M），上下文 1024 | G3；lesson-01-doc C17 |
| GPT-3 | 2020-05-28（arXiv v1） | 2005.14165 | 175B，$n_{ctx}$ 2048 | G4；lesson-01-doc C23 |
| InstructGPT | 2022-01-27（公告 + PDF）；2022-03-04（arXiv v1） | 2203.02155 | SFT → RM → PPO | G5；OAI-WB |
| ChatGPT | 2022-11-30 | 公告页 | — | G6；OAI-WB |

### 3.2 DeepSeek 发布总表（按公开发布日排序；“主线”列用课程地图 A–F，M = 多模态旁支，P = 形式证明旁支，I = 基础设施）

| # | 公开发布日 | 论文 v1 | 名称 | arXiv | 总 / 激活参数 | 上下文 | 一句话创新 | 主线 | 来源 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2023-11-02（GitHub 最早提交，代理值） | 2024-01-25 | DeepSeek-Coder 1.3B/6.7B/33B | 2401.14196 | 稠密 1.3B–33B | 16K | 仓库级代码语料 + FIM 填空 | D | T1 |
| 2 | 2023-11-29 | 2024-01-05 | DeepSeek LLM 7B/67B（后称 “DeepSeek-V1”） | 2401.02954 | 稠密 7B / 67B | 4K | 超参数 scaling law；67B 用 GQA、加深到 95 层 | F、A | T2、S1 |
| 3 | 2024-01-11 | 2024-01-11 | DeepSeekMoE 16B | 2401.06066 | 16.4B / 2.8B | 4K | 细粒度专家 + 共享专家 | B | T3、S2 |
| 4 | 2024-01-25（HF） | — | DeepSeek-Coder-v1.5 7B | （见 2401.14196 §5） | 7B | — | 从 LLM-7B 续训 2T | D | T4 |
| 5 | 2024-02-05 | 2024-02-05 | DeepSeekMath 7B | 2402.03300 | 7B | 4K（config） | 120B 数学语料；**GRPO** | D | T5 |
| 6 | 2024-03-11（GitHub） | 2024-03-08 | DeepSeek-VL 1.3B/7B | 2403.05525 | 1.3B / 7B | — | 混合视觉编码器，1024×1024 | M | T6 |
| 7 | 2024-05-06 | 2024-05-07 | **DeepSeek-V2** | 2405.04434 | 236B / 21B | 128K | **MLA**；设备受限路由 | A、B | T7、S3 |
| 8 | 2024-05-16 | （同上，附录 B） | DeepSeek-V2-Lite | 同上 | 15.7B / 2.4B | 32K | 小号 MLA + MoE | A、B | T7、S4 |
| 9 | （权重 2024-08-16 建库） | 2024-05-23 | DeepSeek-Prover（V1） | 2405.14333 | 7B | — | 800 万条 Lean 4 合成证明 | P | T8 |
| 10 | 2024-06-14 | 2024-06-17 | DeepSeek-Coder-V2（+Lite） | 2406.11931 | 236B/21B；16B/2.4B | 128K | V2 中间检查点续训 6T，338 种语言 | D | T9 |
| 11 | 2024-06-28（API）；HF 07-18 | — | DeepSeek-V2-Chat-0628 | — | 236B / 21B | — | 换 Coder-V2 base 重做 Chat | D | T10 |
| 12 | 2024-07-04（HF/GitHub） | 2024-07-02 | ESFT | 2407.01906 | V2-Lite 骨干 | — | 只微调与任务相关的专家 | B | T11 |
| 13 | 2024-07-24（API） | — | DeepSeek-Coder-V2-0724 | — | 236B / 21B | — | 对齐优化 | D | T10 |
| 14 | 2024-08-15 | 2024-08-15 | DeepSeek-Prover-V1.5 | 2408.08152 | 7B | — | RLPAF + RMaxTS（MCTS） | P | T12 |
| 15 | — | 2024-08-26 | Fire-Flyer AI-HPC（SC'24） | 2408.14158 | — | — | 1 万张 PCIe A100 集群软硬协同；HFReduce、3FS | C、I | T13 |
| 16 | — | 2024-08-28 | Auxiliary-Loss-Free Load Balancing | 2408.15664 | ≤3B 实验 | — | 专家偏置代替辅助损失 | B | T14 |
| 17 | 2024-09-05 | — | DeepSeek-V2.5 | — | 236B / 21B | — | Chat 与 Coder 合并 | D | T15 |
| 18 | 2024-10-18（GitHub） | 2024-10-17 | Janus 1.3B | 2410.13848 | 1.3B | — | 理解/生成解耦视觉编码 | M | T16 |
| 19 | 2024-11-12（HF） | 2024-11-12 | JanusFlow 1.3B（CVPR 2025） | 2411.07975 | 1.3B | — | 自回归 + rectified flow | M | T16 |
| 20 | 2024-11-20 | — | DeepSeek-R1-Lite-Preview（仅网页） | — | 未公开 | — | 首个公开的长思维链推理服务 | D | T17 |
| 21 | 2024-12-10 | — | DeepSeek-V2.5-1210（V2 系列终版） | — | 236B / 21B | — | — | D | T15 |
| 22 | 2024-12-13 | 2024-12-13 | DeepSeek-VL2 Tiny/Small/Base | 2412.10302 | 3B/16B/27B 级 MoE；激活 1.0/2.8/4.5B | — | 动态切图 + MLA-MoE 语言模型 | M、A | T18 |
| 23 | 2024-12-26 | 2024-12-27 | **DeepSeek-V3** | 2412.19437 | 671B / 37B | 128K | 无辅助损失均衡、**MTP**、**FP8** 训练、DualPipe | B、C、D | T19、S5 |
| 24 | 2025-01-20 | 2025-01-22 | **DeepSeek-R1**、R1-Zero、6 个蒸馏模型 | 2501.12948 | 671B / 37B | 128K | 纯 RL 激发推理（Nature 2025-09-17） | D | T20 |
| 25 | 2025-01-27（GitHub） | 2025-01-29 | Janus-Pro 1B/7B | 2501.17811 | 1B / 7B | 4096 | Janus 放大数据与模型 | M | T16 |
| 26 | — | 2025-02-11 | CodeI/O（ICML 2025） | 2502.07316 | — | — | 代码输入输出预测凝练推理模式 | D | T22 |
| 27 | — | 2025-02-16 | **NSA**（ACL 2025 Best Paper） | 2502.11089 | 实验 27B / 3B | 64K 测速 | 压缩 + 选择 + 滑窗三分支可训练稀疏注意力 | A | T22 |
| 28 | 2025-02-24 起 | — | 开源周：FlashMLA、DeepEP、DeepGEMM、DualPipe/EPLB、3FS | — | — | — | 生产级基础设施开源 | C、I | T21 |
| 29 | 2025-03-24 | — | DeepSeek-V3-0324（改 MIT 许可） | — | 671B / 37B（同 V3 结构） | 卡片未写；config 同 V3（163,840） | 推理与前端能力提升 | D | T23 |
| 30 | — | 2025-04-03 | DeepSeek-GRM（SPCT） | 2504.02495 | — | — | 生成式奖励模型的推理期扩展 | D | T24 |
| 31 | 2025-04-30 | 2025-04-30 | DeepSeek-Prover-V2 671B/7B | 2504.21801 | 671B / 7B | — | 子目标分解 + RL，miniF2F 88.9% | P | T25 |
| 32 | — | 2025-05-14 | Insights into DeepSeek-V3（ISCA '25） | 2505.09343 | — | — | 硬件–模型协同的反思 | C | T26 |
| 33 | 2025-05-28 | — | DeepSeek-R1-0528 | — | 671B / 37B（同 V3 结构） | 卡片未写；config 同 V3 | 更深推理、函数调用 | D | T27 |
| 34 | 2025-08-21（Base 08-19） | — | DeepSeek-V3.1 | — | 671B / 37B | 128K | 一模两式（think / non-think）；UE8M0 FP8 | D、C | T28 |
| 35 | 2025-09-22 | — | DeepSeek-V3.1-Terminus | — | 671B / 37B（同 V3 结构） | config 同 V3；按 V3.1 为 128K（未单独核卡片） | 语言一致性、智能体 | D | T29 |
| 36 | 2025-09-29 | （GitHub PDF，无 arXiv） | DeepSeek-V3.2-Exp | — | 671B / 37B | 128K | **DSA** 首发 | A | T30 |
| 37 | 2025-10-20（GitHub） | 2025-10-21 | DeepSeek-OCR | 2510.18234 | 解码器 3B-MoE-A570M | — | 视觉 token 压缩文本上下文（<10× 时 97%） | E | T31 |
| 38 | 2025-11-19（GitHub） | — | LPLB | — | — | — | 线性规划专家负载均衡（早期研究） | B、I | T32 |
| 39 | 2025-11-27 | 2025-11-27 | DeepSeekMath-V2 | 2511.22570 | 基于 V3.2-Exp-Base | — | 可自验证的证明生成 | D | T33 |
| 40 | 2025-12-01 | 2025-12-02 | **DeepSeek-V3.2**、V3.2-Speciale | 2512.02556 | 671B / 37B | 128K | 规模化 RL、思考融入工具调用 | A、D | T34、S6 |
| 41 | — | 2025-12-31 | **mHC** | 2512.24880 | 实验 27B | — | 双随机约束的超连接，开销 6.7% | E | T35 |
| 42 | 2026-01-12 | 2026-01-12 | **Engram**（ACL 2026） | 2601.07372 | 实验 Engram-27B | — | N-gram 条件记忆，新稀疏轴 | E、B | T36 |
| 43 | 2026-01-27 | 2026-01-28 | DeepSeek-OCR 2 | 2601.20552 | — | — | DeepEncoder V2 按语义重排视觉 token | E、M | T37 |
| 44 | 2026-04-24 | 2026-04-26（编号 2606.19348） | **DeepSeek-V4 预览版**：V4-Pro、V4-Flash（+Base） | 2606.19348 | 1.6T/49B；284B/13B | 1M | CSA/HCA 混合注意力、mHC、Muon、FP4 专家 | A、E、F、C | T38、S7、S8 |
| 45 | 2026-06-26（GitHub）/06-27（HF） | 2026-07-06 | DeepSpec / **DSpark**（V4-Flash/Pro-DSpark） | 2607.05147 | 同 V4 + 草稿模块 | — | 半自回归草稿 + 置信度调度验证 | C、D | T39 |
| 46 | 2026-07-31 | — | DeepSeek-V4-Flash-0731（V4-Flash 正式版） | — | 284B / 13B | 1M | 同结构重做后训练 | D | T40 |
| 47 | 2026-08-13 | — | DeepSeek-V4-Pro-0813（**V4-Pro GA**） | — | 1.6T / 49B | 1M | 智能体增强；low/high/max 三档 | D | T41 |
| 48 | 2026-08-21（API；HF 08-31） | — | DeepSeek-V4-Flash-Vision-Exp | — | 同 V4-Flash + 视觉 | 1M（config） | V4 首个多模态实验模型 | M | T42 |
| 49 | 2026-09-10 | 2026-09-17 | **DeepSeek-V4.1-Flash** | 2609.19969 | 552B backbone（+196B Engram）/ prefill 8B、decode 16B | 1M | CED、CSA2 跨层复用、FP4 KV、SWA 有界回放 | A、E、C | T43、S9 |
| 50 | — | 2026-09-19 | DSec 沙箱平台 | 2609.22978 | — | — | 智能体 RL 的弹性沙箱 | C、I | T44 |

**排除项**（有意不进表）：DeepSeek APP（2025-01-15，产品）、API 功能更新（2024-07-25 JSON/FIM、2024-08-02 硬盘缓存）、DreamCraft3D（2023-10，3D 生成，不属本课范围）、
各类工具仓库（awesome-*、smallpond、profile-data、TileKernels、DeepJIT、DeepSelect、deepseek-harness、deepseek-recipe）只在课文“基础设施”旁注里提。
谣言与未发布项见 §7 E16–E18。

### 3.3 主线模型规格总表（每格注来源；P = 论文，C = config.json，R = README/模型卡，N = §4 复算）

| 模型 | 层数 | $d_{model}$ | 注意力 | KV 字节/token（全局） | 专家 | 训练 tokens | 上下文 |
|---|---|---|---|---|---|---|---|
| DeepSeek LLM 67B | 95（P 表 2；C） | 8192（P；C） | GQA：64 查询头 / 8 KV 头，$d_h$ 128（P；C） | 389,120（BF16，N1；= dsv41 图 1(b)） | 稠密 | 2.0T（P 表 2） | 4096（P 表 2；C） |
| DeepSeekMoE 16B | 28（P §5.1.2；C） | 2048（P；C） | MHA 16 头 × 128（P；C） | 229,376（BF16，N1） | 2 共享 + 64 路由，激活 6；专家为标准 FFN 的 0.25 倍（P）/ `moe_intermediate_size` 1408（C） | 2T（P） | 4K（dsv2 附录 B 表；C 4096） |
| DeepSeek-V2 | 60（P §3.1.2；C） | 5120（P；C） | MLA：128 头，$d_h$ 128，$d_c$ 512，$d_c'$ 1536，$d_h^R$ 64（P；C） | 69,120（BF16）/ 25,920（6 bit 均值）（N1、N6） | 2 共享 + 160 路由，激活 6，中间维 1536；首层稠密（P；C） | 8.1T（P） | 128K（P；R；C 写 163,840，E10） |
| DeepSeek-V2-Lite | 27（P 附录 B.1；C） | 2048（P；C） | MLA：16 头，$d_c$ 512，**无查询压缩**，$d_h^R$ 64（P；C `q_lora_rank` null） | 31,104（BF16，N1） | 2 + 64，激活 6，中间维 1408（P；C） | 5.7T（P 附录 B.1） | 32K（P 附录 B 表；R；C 163,840） |
| DeepSeek-V3 | 61（P §4.2；C） | 7168（P；C） | MLA，维度同 V2（P；C） | 70,272（BF16）/ 40,016（FP8 656 B/层）（N1；ISCA 表 1 印 70.272 KB） | 1 共享 + 256 路由，激活 8，中间维 2048；前 3 层稠密；节点受限 M = 4（P；C） | 14.8T（P） | 128K（P；R；C 163,840） |
| DeepSeek-V3.2 | 61（C） | 7168（C） | MLA + DSA：indexer 64 头 × 128，top-k 2048（P §2.1；C） | 48,068（N1；= dsv41 图 1(b)） | 同 V3（C） | V3 的 14.8T + V3.1 续训 0.839T + V3.2 续训 2.1B + 943.7B（P、R；N9，约 16.6T，笔者合计） | 128K（P §4；C 163,840） |
| DeepSeek-V4-Flash | 43（P §4.2.1；C） | 4096（P；C） | 前 2 层纯 SWA；21 层 CSA（m = 4，top-k 512）+ 20 层 HCA（m' = 128）交替；64 头，c = 512；SWA 128（P；C `compress_ratios`） | 3,514（N1；= dsv41 图 1(b)） | 1 + 256，激活 6，中间维 2048；前 3 个 MoE 层哈希路由（P；C） | 32T（P §1、§4.2.2） | 1M（P；C 1,048,576） |
| DeepSeek-V4-Pro | 61（P；C） | 7168（P；C） | 前 2 层 HCA；30 层 CSA（top-k 1024）+ 31 层 HCA；128 头，c = 512（P；C） | 5,031（N1，笔者推算，论文未印） | 1 + 384，激活 6，中间维 3072（P；C） | 33T（P §1、§4.2.2） | 1M（P；C） |
| DeepSeek-V4.1-Flash | 40 = 20 编码器 + 20 解码器（P §4.2.1；C） | 5120（P；C） | 前 2 层纯 SWA；18 层 CSA2 m = 2（3 Full + 15 Reuse）；20 层 CSA2 m = 1（1 Full + 4 Reindex + 15 Reuse）；indexer 32 头 × 128，top-k 512；FP4 主 KV（P；C） | 890（P 摘要；N1） | 1 + 384，激活 6，中间维 2304（P；C）；另有 Engram 196B（P；C 推算 N14） | 45T 多模态（P §4.2.2） | 1M（P；C） |

参数量：67B 与 16B 取论文；V2 236B/21B、V2-Lite 15.7B/2.4B、V3 671B/37B、V4-Flash 284B/13B、V4-Pro 1.6T/49B、V4.1 552B（backbone）/ 8B（prefill）/ 16B（decode）均为论文原句。
V3.2 论文与 README 都没有单列参数量（config 与 V3 同结构，按 671B/37B 记；网上常见的 “685B” 口径未核，§8）。

### 3.4 其它版本的规格速查（第 0 课总表可能要列）

| 模型 | 总/激活 | 上下文 | 来源 |
|---|---|---|---|
| DeepSeek LLM 7B | 稠密 7B；30 层、4096、MHA 32 头 | 4096 | dsllm 表 2；C |
| DeepSeek-Coder 33B | 62 层、7168、56 头 / 8 KV 头 | 16,384（RoPE 线性缩放因子 1 → 4，基频 10000 → 100000） | C；dscoder §3.6 Long Context |
| DeepSeek-Coder-V2 | 236B/21B；16B/2.4B | 128K | R 表；摘要 |
| DeepSeek-R1 / R1-Zero | 671B / 37B | 128K | GitHub README 表 |
| R1 蒸馏模型 | Qwen2.5-Math-1.5B/7B、Llama-3.1-8B、Qwen2.5-14B/32B、Llama-3.3-70B-Instruct 为底座 | — | R1 README 表 |
| DeepSeek-V3.1 / Base | 671B / 37B | 128K | HF-CARD 表 |
| V3.2-Exp | 同 V3 + DSA（config 与 V3.2 相同字段值） | 128K（按 V3.2 论文；C 163,840） | C |
| V4-Flash-Vision-Exp | 43 层、4096、256 专家激活 6（同 V4-Flash） | 1M（C） | C（`num_nextn_predict_layers` 3） |
| DeepSeek-VL2 | Tiny 3.37B/1.0B；Small 16.1B/2.8B；Base 27.5B/4.5B（整模型）；语言部分激活 0.57B/2.4B/4.1B | — | 摘要；§2；README 注释（E12） |

### 3.5 论文 vs config / 版本漂移

| 项 | 论文 | config / 代码 / 其它 | 判断 |
|---|---|---|---|
| V2/V3/V3.2 上下文 | 128K | `max_position_embeddings` 163,840 = 4096 × 40（YaRN factor 40） | config 是位置编码上限，不是宣称支持的上下文（E10） |
| DeepSeek LLM 67B FFN 中间维 | $\frac{8}{3}d_{model}$ = 21,845.3 | 22,016（= 86 × 256） | 取整到 256 的倍数 |
| DeepSeekMoE 16B 专家大小 | “0.25 times the size of a standard FFN” | `moe_intermediate_size` 1408；首层稠密 `intermediate_size` 10,944 | 1408 = 0.25 × 5632；5632 与 10,944 的关系未核（§8） |
| V4-Flash config | — | 2026-04-27 提交补了 `"expert_dtype": "fp4"`（与 04-24 首版 diff 仅此一行） | 发布后三天修过 config |
| V4 技术报告 | arXiv v1 提交 04-26（此后无 v2） | HF 上 `DeepSeek_V4.pdf` 在 04-24 当天至少有 3 次 PDF 提交（02:39、02:50、约 10:00 UTC；README 自 02:25 的 “Release DeepSeek-V4” 起即链接该 PDF，首份 PDF 是否随 Release 提交上传未逐一核 diff），05-06 再更新一次；**06-22 的 “Update technical report” 提交（V4-Pro b5968e9）实为删除该 PDF**（diff 显示 “DeepSeek_V4.pdf DELETED”，删前 4,480,407 字节），同时把 README 链接改指 arXiv。新闻页链接因此 404（核查更正，§10） | 读 arXiv v1；注意 HF 的 05-06 版晚于 arXiv v1，二者未比对（E2、§8） |
| V4.1 的 MTP | “We omit the MTP module during backbone pre-training” | config `num_nextn_predict_layers` 3 + `dspark_*` 字段 | 这 3 层是 DSpark 草稿块（§2.4.3 “three Transformer blocks”），不是 V3 式 MTP |
| V4.1 RMSNorm eps | — | `rms_norm_eps` 1e-20（V4 为 1e-6） | 只记录，不解释（§8） |
| V4.1 Engram 作者引用 | 同一篇 Engram 被引为 Cheng et al. 2026b（arXiv）与 2026c（ACL） | ACL 版 14 位作者；arXiv v2 列 21 位（多出 Rui Tian 等） | 作者表随版本变化（E14） |
| V3.2 indexer 键格式 | 论文只说 “can be implemented in FP8” | 代码：FP8 + FP32 缩放（132 字节） | 见 lesson-14-doc |
| VL2 激活参数 | 摘要 1.0B/2.8B/4.5B | README 注释 “activated 2.4B”（small）、“activated 4.2B”（base） | README 注释应指语言部分（笔者判断），且 4.2 ≠ 论文的 4.1（E12） |
| V4.1-Flash 对 V4-Pro 的 “1/4 activated” | “using only 1/3 total parameters and 1/4 activated parameters” | 552/1600 = 0.345；16/49 = 0.327；8/49 = 0.163 | “1/4” 与 decode 或 prefill 任一口径都对不上，只与二者平均 12/49 = 0.245 接近（E11） |

---

## 4. 复算的数字（Python 表达式 → 结果）

全部在 Python 3.11 中算过。字节单位：B；GB = $10^9$ B，GiB = $2^{30}$ B。

**N1 每 token 全局 KV 字节（式 (0.1)–(0.5)）**

| 模型 | 表达式 | 结果 | 对照 |
|---|---|---|---|
| GPT-3 175B（MHA，BF16） | `2*96*128*96*2` | 4,718,592 | 与 lesson-04-doc 图 4.3 数据一致 |
| DeepSeek LLM 7B（MHA） | `2*32*128*30*2` | 491,520 | 比 67B 还大（E9） |
| DeepSeek LLM 67B（GQA-8） | `2*8*128*95*2` | **389,120** | = dsv41 图 1(b) “DeepSeek-V1” |
| DeepSeekMoE 16B（MHA） | `2*16*128*28*2` | 229,376 | — |
| V2（MLA，BF16） | `(512+64)*60*2` | 69,120 | lesson-09-doc 一致 |
| V2（6 bit 均值） | `(512+64)*60*6/8` | 25,920 | 用于 93.3% |
| V2-Lite（BF16） | `(512+64)*27*2` | 31,104 | — |
| V3（BF16） | `(512+64)*61*2` | 70,272 | = ISCA 表 1 “70.272 KB” |
| V3（FP8 656 B/层格式） | `656*61` | 40,016 | — |
| V3.2（656 + 132） | `(656+132)*61` | **48,068** | = dsv41 图 1(b) |
| V4-Flash | `21*(584+68)/4 + 20*584/128` | **3,514.25** | = 图 1(b) 的 3,514 |
| V4-Pro | `30*(584+68)/4 + 31*584/128` | 5,031.44 | 论文未印，笔者推算 |
| V4.1-Flash | `3*(288+68)/2 + 1*(288+68)/1` | **890.00** | = 摘要 890 |

分解：V4-Flash 主 KV 部分 `21*584/4+20*584/128` = 3,157.25，indexer 部分 `21*68/4` = 357；V4.1 主 KV `2.5*288` = 720，indexer `2.5*68` = 170。
若 V4.1 主 KV 用 FP8 的 528 字节格式：`2.5*(528+68)` = 1,490 → FP4 单独贡献 `1490/890` = 1.67 倍，跨层复用等结构改动贡献 `3514.25/1490` = 2.36 倍（笔者分解，E15）。

**N2 dsv41 图 1(b) 的倍数**：`389120/48068` = 8.095（“8.1×”）；`48068/3514.25` = 13.678（“13.7×”）；`3514.25/890` = 3.949（“3.9×”、正文 “1/4”）；
`389120/890` = 437.21（“437-fold”）。四个柱长与三段倍数**全部复现**。

**N3 V4 相对 V3.2 的 KV**：`5031.44/48068` = 10.47%（“10%”）；`3514.25/48068` = 7.31%（“7%”）；倒数 9.55（图 1 “9.5× smaller”）、13.68（“13.7× smaller”）。

**N4 V4 相对 V3.2 的 FLOPs**：`1/3.7` = 27.0%（“27%”）；`1/9.8` = 10.2%（“10%”）。（FLOPs 本身只能目视，§5.1。）

**N5 “约 2% 的 GQA8 基线”**：基线 `2*8*128*2*L`：Pro（L = 61）249,856 → `5031.44/249856` = 2.014%；Flash（L = 43）176,128 → `3514.25/176128` = 1.995%。

**N6 V2 的 93.3%**：`1-25920/389120` = **93.34%**；若都按 BF16 `1-69120/389120` = 82.24%；V2 按 1 字节 FP8 `1-34560/389120` = 91.12%。元素数之比 `34560/194560` = 17.76%。

**N7 V2 = 2.25 组 GQA**：`(512+64)/(2*128)` = 2.25。

**N8 V3 训练成本**：`2664+119+5` = 2,788K GPU 小时；`2788*2` = 5,576K 美元 = $5.576M；每万亿 token 180K 小时 × 14.8 = 2,664K；`180e3/2048/24` = 3.66 天（原文 “3.7 days”）。

**N9 续训 token**：V3.1 `630+209` = 839B（新闻写 “840B”）；旧阶段规模 `630/10` = 63B、`209/3.3` = 63.3B；V3.2 indexer 预热 `1000*16*131072` = 2.097e9（“2.1B”），
稀疏阶段 `15000*480*131072` = 9.437e11（“943.7B”）；V3.2 累计约 `14.8+0.839+0.0021+0.9437` = 16.585T（笔者合计，论文未给）。

**N10 R1 训练成本（dsr1 表 7）**：`512*198` = 101,376 ≈ 101K；`512*80` = 40,960 ≈ 41K；`101+5+41` = 147K；×2 = $294K。

**N11 V2 省 42.5% 训练成本**：`1-172.8/300.6` = 42.51%（每万亿 token：67B 300.6K 小时，V2 172.8K 小时，dsv2 §3.2.3）。

**N12 DeepSeek LLM 67B 的 decode FLOPs（式 (0.6)）**：`2*67e9 + 4*8192*T*95`：T = 4096 → 146.8 GFLOPs；T = 1,048,576 → 3,398.2；T = 10^6 → 3,247.0。
只用 Kaplan 单项 `2*67e9 + 2*95*1048576*8192` = 1,766.1，与图 2 不符。

**N13 单请求累计 KV（1M = 1,048,576 token）**

| 模型 | 1M 时 | 128K 时 | 80 GB 能放几个 1M 请求（只算 KV） |
|---|---|---|---|
| DeepSeek LLM 67B | 408.02 GB = 380.00 GiB | 47.50 GiB | `80e9/(389120*1048576)` = 0.20 |
| V3.2 | 50.40 GB = 46.94 GiB | 5.87 GiB | 1.59 |
| V4-Pro | 5.28 GB = 4.91 GiB | 0.61 GiB | 15.16 |
| V4-Flash | 3.68 GB = 3.43 GiB | 0.43 GiB | 21.71 |
| V4.1-Flash | 0.93 GB = 0.87 GiB | 0.11 GiB | 85.72 |

V3.2 的 50.4 GB 与 dsv4 图 1 右下 V3.2 曲线在 1024K 处的约 48–50（纵轴 “GB”；核查时按像素读约 48.6）在目视误差内一致；图上无法分辨 GB 与 GiB（46.9）口径。

**N14 Engram 参数**：config `engram_num_embeddings` [384,006,168, 384,016,682] × `engram_head_dim` 256：`(384006168+384016682)*256` = 1.9661e11 ≈ **196.6B**（论文 “196B”）；
每模块 3 阶 × 8 头 = 24 张表，平均每表 `384006168/24` = 16,000,257 行（论文 “approximately 16M entries”，表长取互异素数）。

**N15 V4.1 “1/3 总参、1/4 激活”**：`552/1600` = 0.345；`16/49` = 0.327；`8/49` = 0.163；`12/49` = 0.245（E11）。

**N16 V4.1 小数字**：FP4 上限 `448*6` = 2,688；`sqrt(512)` = 22.63；候选池 `2048*8` = 16,384；上下文倍数 `1048576/4096` = 256。

**N17 SWA 回放长度**：精确重建 V4-Flash `43*128` = 5,504、V4-Pro `61*128` = 7,808、V4.1 `40*128` = 5,120；V4.1 解码器精确 `20*128` = 2,560；有界回放只要 128。

**N18 V4 “SWA KV 约为压缩 KV 的 8 倍”**：每 token 每层 SWA 按 584 B 计：Flash `584*43/3157.25` = 7.95、Pro `584*61/4521.44` = 7.88（只比主 KV）；
连 indexer 一起比则为 7.15、7.08。原文 “approximately 8 times” 对应前者。

**N19 参数量粗算（笔者，只计路由专家 + 共享专家 + 注意力主投影 + 嵌入/输出头；忽略 indexer、压缩器、mHC、norm、MTP/DSpark、Engram、视觉编码器）**

| 模型 | 表达式要点 | 总参 | 激活 |
|---|---|---|---|
| V4-Flash | 路由 `43*256*3*4096*2048` = 277.0B；共享 1.08B；注意力 4.60B；嵌入 + 头 1.06B | ≈ 283.8B（论文 284B） | `43*6*3*4096*2048` + 共享 + 注意力 + 头 ≈ 12.7B（论文 13B） |
| V4-Pro | 路由 `61*384*3*7168*3072` = 1,547.4B；共享 4.03B；注意力 18.29B；嵌入 + 头 1.85B | ≈ 1,571.6B（论文 1.6T） | ≈ 47.4B（论文 49B） |
| V4.1-Flash | 路由 `40*384*3*5120*2304` = 543.6B；共享 1.42B；注意力 5.06B；嵌入 + 头 1.32B | ≈ 551.4B（论文 552B backbone） | decode ≈ 15.6B（论文 16B）；prefill 只算 20 个编码器层 ≈ 8.1B（论文 8B） |

注意力每层 = $d\cdot d_c'+d_c'\cdot n_hc$（Q）+ $g\cdot\frac{n_h}{g}c\cdot d_g+g\,d_g\cdot d$（分组输出）+ $d\cdot c$（KV），取 $c=512$、$d_g=1024$。这张表说明
“552B 不含 Engram 的 196B”——若含，总数约 748B（`552+196`）。

**N20 数据/参数比与 6ND（笔者）**

| 模型 | D / N_total | D / N_act | $6N_{act}D$ |
|---|---|---|---|
| DeepSeek LLM 67B | 29.9 | 29.9 | 8.04e23 |
| DeepSeekMoE 16B | 122.0 | 714.3 | 3.36e22 |
| V2 | 34.3 | 385.7 | 1.02e24 |
| V2-Lite | 363.1 | 2,375 | 8.21e22 |
| V3 | 22.1 | 400.0 | 3.29e24 |
| V4-Flash | 112.7 | 2,461.5 | 2.50e24 |
| V4-Pro | 20.6 | 673.5 | 9.70e24 |
| V4.1-Flash | 81.5 | 2,812.5（按 16B） | 4.32e24 |

**N21 间隔天数**：Transformer → ChatGPT 1,997 天；ChatGPT → DeepSeek LLM 364 天；DeepSeek LLM → V2 发布 159 天（`2024-05-06 − 2023-11-29`）；V2 → V3 234 天；
**V3 → R1 25 天**；R1 → V3.2-Exp 252 天；V3.2 → V4 预览 144 天；V4 预览 → V4-Pro GA 111 天；V4-Pro GA → V4.1-Flash 28 天；
DeepSeek LLM → V4.1-Flash **1,016 天**；Transformer → V4.1-Flash 3,377 天；V4 HF 发布 → arXiv 提交 2 天；V4.1 HF → arXiv 7 天；R1 Nature 收稿 → 在线 215 天。

**N22 KV 年化缩减**：`437.21**(365/1016)` = 8.89 倍/年（从 DeepSeek LLM 到 V4.1-Flash，笔者口径）。

**N23 V4.1 decode FLOPs**：上下文 ×256 时 FLOPs +25%（原文 “by only 1/4”）；对照 67B 同区间 `3398.2/146.8` = 23.1 倍。

**N24 VL2 视觉部分的激活量（笔者推算）**：`1.0-0.57` = 0.43B，`2.8-2.4` = 0.40B，`4.5-4.1` = 0.40B → 整模型激活 − 语言激活 ≈ 0.4B，即视觉编码器规模。

---

## 5. 图表清单与重绘建议

### 5.1 论文中可重绘的图（只重绘数据，不照搬配色）

| 来源 | 图号 | 图注（原文要点） | 展示什么 | 可提取的数据 |
|---|---|---|---|---|
| dsv41 | 图 1(b) | “Global KV cache size per token (in bytes) across generations of DeepSeek models” | 四代 KV 柱状图 | 389,120（2023.11）、48,068（2025.12）、3,514（2026.04）、890（2026.09）；8.1× / 13.7× / 3.9×（数字印在图上） |
| dsv41 | 图 2 | 单 token decode FLOPs vs 上下文（对数–对数，4K–1024K），BF16/FP8/FP4 按 1/0.5/0.25 加权 | 四条曲线 | 目视（±10%）：V1 约 147→约 3,400；V3.2 约 76→约 600；V4-Flash 约 16→约 60；V4.1 约 20→约 25（GFLOPs，4K→1M）。V1 曲线可用式 (0.6) 精确画 |
| dsv41 | 图 3 | 40 层网络 “divided into a causal encoder and a decoder, each with 20 layers” | V4.1 结构 | 层型序列见 §3.3 |
| dsv41 | 图 4 | CSA2 三种模式 Full / Reindex / Reuse | 模式示意 | — |
| dsv41 | 图 5 | 分层稀疏 indexer | 候选池 | 2048 块 × 8 = 16,384 候选，最终 top-512 |
| dsv4 | 图 1 右 | “inference FLOPs and KV cache size of DeepSeek-V4 series and DeepSeek-V3.2” | 两条累计曲线 | FLOPs（FP8 当量，T）：V3.2 在 1024K 约 1.2，Pro 约 0.32，Flash 约 0.12；标注 3.7× / 9.8×；KV（GB）：V3.2 约 48–50，Pro 约 5.3，Flash 约 3.7；标注 9.5× / 13.7×（四个倍数标注已在原图核对） |
| dsv4 | 图 2 | V4 总体结构（CSA/HCA + DeepSeekMoE + mHC） | 结构 | — |
| dsv4 | 图 3、4 | CSA、HCA 核心结构 | 压缩 + 选择 | m = 4，m' = 128，top-k 512/1024，SWA 128 |
| dsv4 | 图 6 | KV cache 布局：经典 KV + 状态缓存 | 存储层级 | 每块覆盖 lcm(m, m') = 128 个原始 token（`lcm(4,128)`），产出 32 个 CSA 条目、1 个 HCA 条目（笔者代入） |
| dsv2 | 图 1(b) | “Training costs and inference efficiency of DeepSeek 67B (Dense) and DeepSeek-V2.” | 三组柱 | 42.5%、93.3%、5.76 倍（数字来自正文） |
| dsv2 | 表 1 | 四种注意力的 KV 通式 | 表格 | §2.1 |

### 5.2 第 0 课建议的图（8–15 张，给出画图所需的全部数据）

1. **图 0.1 科技树总图（时间轴 × 六条主线）**。横轴 2023-10 至 2026-10（月刻度）；纵向六条泳道 A–F，外加 M/P/I 三条细泳道。节点 = §3.2 表 50 行，
   坐标取“公开发布日”（无发布日的论文取 v1 日期，用空心点）；节点标签“名称 · 总/激活 · 上下文”。主干节点（V2、V3、R1、V3.2、V4、V4.1）加粗。
   颜色按泳道：A 用 `--fig-blue`，B 用 `--fig-violet`，C 用 `--fig-orange`，D 用 `--fig-green`，E 用 `--fig-amber`，F 用 `--fig-ink`；旁支用 `--fig-muted`。
2. **图 0.2 GPT 史前史条带**。2017-06-12、2018-06-11、2019-02-14、2020-05-28、2022-01-27（虚点 03-04 arXiv）、2022-11-30；标 “1,997 天”；右端接 2023-11-29 DeepSeek LLM（“364 天”）。
3. **图 0.3 每 token KV 字节阶梯（对数纵轴）**。数据 N1：GPT-3 4,718,592；DeepSeek 7B 491,520；67B 389,120；MoE 16B 229,376；V2 69,120（BF16）/ 25,920（6 bit）；
   V3 70,272 / 40,016（FP8）；V3.2 48,068；V4-Pro 5,031；V4-Flash 3,514；V4.1 890。横轴按发布日。颜色 `--fig-amber`（缓存语义）。
4. **图 0.4 重绘 dsv41 图 1(b)**（线性横轴柱状）：四根柱 + 三段倍数 + 总倍数 437×。标注“据 DeepSeek-V4.1-Flash 论文图 1(b) 重绘”。
5. **图 0.5 单请求累计 KV 随上下文增长**（0–1M，线性）：直线斜率 = N1 字节；在 1M 处标 408 GB / 50.4 GB / 5.28 GB / 3.68 GB / 0.93 GB；
   画一条 80 GB 水平虚线（`--fig-red`），交点即“单卡放不下”的上下文：67B 在 `80e9/389120` = 205,592 token，V3.2 在 1,664,309（>1M，不相交）。
6. **图 0.6 总参数 vs 激活参数（双对数散点）**：DeepSeekMoE 16.4/2.8、V2-Lite 15.7/2.4、V2 236/21、V3 671/37、V4-Flash 284/13、V4-Pro 1600/49、V4.1 552/16（decode）与 552/8（prefill，用竖线连成一段）；
   参考线“激活比 = 1/10、1/20、1/40”。稠密 67B 放在对角线上。
7. **图 0.7 训练 token 柱状**：67B 2.0T、MoE 16B 2T、V2 8.1T、V2-Lite 5.7T、V3 14.8T、V3.2 约 16.6T（堆叠：14.8 + 0.839 + 0.946）、V4-Flash 32T、V4-Pro 33T、V4.1 45T（多模态，文本:多模态 = 7:1）。
8. **图 0.8 注意力演进条带（机制图）**：MHA（67B 前）→ GQA（67B，8 组）→ MLA（V2，512 + 64）→ MLA + DSA（V3.2，top-2048）→ CSA/HCA（V4，m = 4/128，top-512/1024）→ CSA2（V4.1，m = 2/1，跨层复用，FP4）。
   每格标“每 token 每层写入多少字节”：BF16 GQA8 4,096；MLA BF16 1,152；V3.2 656 + 132；V4 CSA 层 `(584+68)/4` = 163、HCA 层 `584/128` = 4.56；V4.1 Full 层 m = 2 为 178、m = 1 为 356，Reuse 层 0。
9. **图 0.9 专家配置演进表格图**：MoE 16B 2 + 64（激活 6，中间维 1408）；V2 2 + 160（6，1536）；V3 1 + 256（8，2048）；V4-Flash 1 + 256（6，2048）；V4-Pro 1 + 384（6，3072）；V4.1 1 + 384（6，2304）。
   用小方块阵列画：共享 `--fig-violet`，激活路由 `--fig-blue`，未激活 `--fig-muted`。
10. **图 0.10 族谱（base 血缘）**：按 S10 连线。三棵树：LLM-7B → Coder-v1.5 → Math-7B → Prover V1/V1.5；V2 → Coder-V2 → V2-0628 → V2.5 → V2.5-1210；
    V3-Base → R1-Zero / R1 / Prover-V2-671B；V3 → V3-0324 → V3.1 → Terminus → V3.2-Exp → V3.2 / Speciale、Math-V2；V4 预览 → Flash-0731 / Pro-0813 / Flash-Vision-Exp；V4.1-Flash 独立从头训练（单独一棵）。
11. **图 0.11 发布节奏**：相邻主干发布间隔柱（N21）：159、234、25、252、63、144、111、28 天。
12. **图 0.12 训练成本卡片**：V3 堆叠 2,664K / 119K / 5K GPU 小时（$5.576M）；R1 101K / 5K / 41K（$294K）；并排文字“不含前期研究与消融”。

### 5.3 第 20 课建议的图

13. **图 20.1 六条主线汇总矩阵**：行 = 主线 A–F，列 = V2、V3、R1、V3.2、V4、V4.1；格内写该代在此主线的关键改动（取自 §3.2 与 §1.5）。
14. **图 20.2 被替换的路线时间带**：token 丢弃（V2 起 → V3 止）、辅助损失（V2 → V3 改 aux-free）、节点受限路由（V3 → V4 取消）、MTP（V3 → V4 → V4.1 换 DSpark）、
    HCA（V4 → V4.1 取消）、稠密热身（V3.2/V4 → V4.1 取消）、PRM/MCTS（R1 试过未采用）。
15. **图 20.3 两张 FLOPs 图的单位换算示意**：同一 V3.2 在 1M 处，FP8 当量约 1.2 T 与 BF16 当量约 0.6 T，箭头标 “×0.5”。

---

## 6. 算例与练习素材（附答案）

1. **（计算）“DeepSeek-V1” 的 389,120 字节从哪里来？** 答：DeepSeek LLM 67B，95 层、8 个 KV 头、头维 128、BF16：$2\cdot8\cdot128\cdot95\cdot2=389{,}120$ B = 380 KiB。
   追问：7B 呢？$2\cdot32\cdot128\cdot30\cdot2=491{,}520$ B——小模型的 KV 反而更大，因为它是 MHA。
2. **（计算）复现 V3.2 的 48,068。** 答：每层 MLA FP8 条目 $512+16+128=656$ B，indexer 键 $128+4=132$ B，61 层：$788\times61=48{,}068$。追问：DSA 让 KV 变大还是变小？变大 20.1%（相对 40,016），它省的是读取量与计算。
3. **（计算）从 config 的 `compress_ratios` 复现 V4-Flash 的 3,514。** 答：列表共 44 项，前 43 项（主干层）里有 21 个 4、20 个 128、2 个 0（纯 SWA 层）；第 44 项的 0 属 MTP 层，不计；$21\cdot(584+68)/4+20\cdot584/128=3{,}423+91.25=3{,}514.25$。
4. **（计算）复现 V4.1 的 890 与 “437 倍”。** 答：`kv_source_layer_ids` = [2, 8, 14, 20]，前三层 m = 2、第 20 层 m = 1；每个 Full 层条目 $288+68=356$ B；$356\cdot(3/2+1)=890$；$389{,}120/890=437.2$。
5. **（计算）1M 上下文时一张 80 GB 显存（只放 KV）能容纳几个请求？** 答：V3.2 1.59 个；V4-Flash 21.7 个；V4.1 85.7 个；67B 连一个都放不下（0.20）。追问：为什么这对智能体重要？（长会话、前缀缓存命中计费，见 NEWS 260910 “Cache-hit charges often account for a large share of agent costs.”）
6. **（辨析）V2 “KV 降 93.3%” 成立的前提是什么？** 答：67B 用 BF16，V2 的 KV 平均压到 6 bit；$1-25{,}920/389{,}120=93.3\%$。两边都按 BF16 只有 82.2%。
7. **（计算）V3 的 $5.576M 怎么来的、不包括什么？** 答：$(2664+119+5)\text{K}\times\$2=\$5.576\text{M}$；不含 “prior research and ablation experiments”。每万亿 token 180K 小时，2048 卡上 3.66 天。
8. **（计算）R1 的 $294K。** 答：R1-Zero $512\times198\approx101$K 小时，SFT 数据 5K，R1 $512\times80\approx41$K，合计 147K × $2 = $294K。追问：它为什么不能和 V3 的 $5.576M 相加当作“R1 总成本”？（R1 建在 V3-Base 上，基座成本另计；两者口径都只含正式训练。）
9. **（日期推理）InstructGPT 是 2022 年 1 月还是 3 月？** 答：公告与论文 PDF 2022-01-27，arXiv v1 2022-03-04。时间线图应标 1 月并注 arXiv 日期。
10. **（日期推理）为什么 V4 论文编号是 2606.19348，提交日期却是 2026-04-26？** 答：arXiv 编号的年月前缀反映编号分配（公开）月份，提交后若经历较长审核，编号会晚于提交日；
    HF 上的 V4 报告 2026-04-24 就已公开。课文写 V4 时间点用 2026-04-24（发布）并注 arXiv 编号即可（机制细节属推测，§8）。
11. **（估算）V4.1 为什么 prefill 只激活 8B？** 答：CED 下 prefill 只需跑 20 个编码器层；按 N19 粗算，40 层 decode 约 15.6B，20 层加嵌入约 8.1B。
12. **（分解）V4.1 相对 V4-Flash 的 3.9 倍从哪来？** 答：若主 KV 仍用 FP8（528 B），V4.1 为 $2.5\times(528+68)=1{,}490$ B；FP4 再压到 890（1.67 倍），其余 2.36 倍来自 CSA2 的跨层共享与压缩率配置（笔者分解，仅作直觉）。
13. **（年化）从 DeepSeek LLM 到 V4.1-Flash，每 token 全局 KV 每年缩小多少倍？** 答：1,016 天缩小 437.2 倍，$437.2^{365/1016}\approx8.9$ 倍/年。
14. **（综合，第 20 课）列出三处“被放弃的设计”并各给出一句原文依据。** 答：token 丢弃（V3 “does not drop any tokens during training”）；节点受限路由（V4 “remove the constraint on the number of routing target nodes”）；
    V4.1 预训练去掉 MTP（“We omit the MTP module during backbone pre-training”）。

---

## 7. 勘误与常见误解

**E1. “V4 论文是 2026 年 6 月发的”——不对。** 编号 2606.19348 的前缀是 2606，但 arXiv 提交记录是 “Sun, 26 Apr 2026 14:49:33 UTC”，而权重与技术报告在
2026-04-24 已在 HF 公开（“Release DeepSeek-V4” 提交 02:24 UTC）。时间线用 2026-04-24（发布）/ 2026-04-26（arXiv 提交），并注明编号前缀的反常。

**E2. V4 新闻页的技术报告链接已失效。** NEWS 260424 链接 `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf`，2026-09-24 实测 `blob` 与 `resolve` 地址均 404；
原因是 2026-06-22 的提交（V4-Pro `b5968e9`，标题仍叫 “Update technical report”）把该 PDF 从仓库删除，并把 README 链接改为 arXiv 2606.19348（V4-Flash 同日同样处理）。
HF 上 PDF 真正的版本只有 04-24（当日至少 3 次提交）与 05-06 两批，其中 05-06 版晚于 arXiv v1（04-26）。课文与参考文献一律引 arXiv。

**E3. “V4.1-Flash 是 552B 模型”——要加限定。** 552B 是 “backbone parameters”，另有 196B Engram 参数（§2.1 原句 “552B backbone parameters and 196B Engram parameters”），
视觉编码器与 DSpark 也不在内。合计约 748B（`552+196`，笔者）。激活参数分 prefill 8B / decode 16B，不能只写一个数。

**E4. 两张“单 token FLOPs”图不能直接比。** V4 图 1 用 FP8 当量（“measured in equivalent FP8 FLOPs”），V4.1 图 2 用 BF16 当量（BF16/FP8/FP4 = 1/0.5/0.25）。
同是 V3.2 在 1M，前者约 1.2 T，后者约 0.6 T。课文若并排引用，必须换算到同一口径。

**E5. “MLA 把 KV 压了 93.3%”——不准确。** 93.3% 是 V2 对 67B 的**部署态**比较，V2 的 KV 平均压到 6 bit；架构本身（两边都 BF16）是 82.2%（N6）。与 lesson-09-doc 结论一致。

**E6. R1 的 Nature 版不是新模型、标题也不同。** Nature 标题 “DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning”，2025-09-17 在线；
arXiv 2501.12948 的 v2（2026-01-04）是扩写版：v1 全文 grep 不到 “294”“GPU hours”“660B”，即训练成本表 7（附录 B.4.4）是 v2 新增；“Unsuccessful Attempts” 在 v1 是正文 §4.2，v2 移到附录 G.2；v2 摘要也改写成 “Here we show …” 的期刊体（未与 Nature 摘要逐字比对）。引用时写清读的是哪一版。

**E7. “InstructGPT 2022 年 3 月”——只对 arXiv 成立。** OpenAI 公告与论文 PDF 2022-01-27；arXiv v1 2022-03-04。

**E8. “DeepSeek-V1”不是一次独立发布。** 它是官方在 V2 README（“DeepSeek-V1 (Dense-67B)”）与 V4.1 图 1(b) 里对 DeepSeek LLM 67B 的回溯命名。
而 DeepSeek 最早公开的语言模型是 DeepSeek-Coder（GitHub 最早提交 2023-11-02），早于 DeepSeek LLM（2023-11-29）。

**E9. “小模型的 KV cache 更小”——不一定。** DeepSeek LLM 7B（MHA）每 token 491,520 B，比 67B（GQA-8）的 389,120 B 还大（N1）。

**E10. config 的 `max_position_embeddings` = 163,840 不是“支持 160K 上下文”。** V2/V3/V3.x 论文与模型卡都写 128K；163,840 = 4096 × 40 是 YaRN 位置上限。

**E11. V4.1 “1/4 activated parameters” 的口径不清。** 相对 V4-Pro 的 49B，decode 16B 是 1/3、prefill 8B 是 1/6，只有两者平均 12B 才接近 1/4（N15）。课文不要转述这个比值，直接给 8B/16B/49B。

**E12. VL2 的激活参数有三套数。** 摘要 1.0B/2.8B/4.5B（整模型）；正文语言部分 0.57B/2.4B/4.1B；GitHub README 注释 small “activated 2.4B”、base “activated 4.2B”。前两套相差约 0.4B（视觉部分，N24）；README 的 4.2 与论文 4.1 不一致。

**E13. 成本数字的口径。** V3 的 $5.576M 只含正式训练（“excluding the costs associated with prior research and ablation experiments”）；R1 的 $294K 只含 R1-Zero、SFT 数据构造与 R1 训练，
**不含** V3-Base 本身。另：dsr1 附录说 “confidently scale up to 660B R1-Zero and R1”，而论文正文与模型卡写 671B——原文如此、未解释差异，课文统一用 671B/37B。

**E14. Engram 的作者表随版本变化。** ACL 2026 版 14 位作者（ACL Anthology 元数据），arXiv v2（2026-07-12）列 21 位；V4.1 参考文献把同一工作分别引为 Cheng et al. 2026b（arXiv）与 2026c（ACL）。

**E15. “V4.1 的 KV 缩小 4 倍是因为用了 FP4”——只对一小半。** 按 N1 分解，FP4（相对 FP8 的 528 B 格式）约贡献 1.67 倍，CSA2 跨层共享与压缩率配置约贡献 2.36 倍（笔者分解）。

**E16. “DeepSeek-R2” 属谣言。** 截至 2026-09-24：API 新闻页（2024-07 至 2026-09 共 18 篇）、Change Log、HF 组织 105 个仓库、GitHub 组织 39 个仓库中都没有 R2。时间线不收录。

**E17. 标题含 “DeepSeek” 的第三方论文不是 DeepSeek 的工作。** 例：2608.30877 “Deploying DeepSeek 175B Locally …”（官方从未发布 175B 模型）、2607.20145（SLAI T-Rex，昇腾上后训练 V4）、
2606.09079（FlashMemory-DeepSeek-V4）、2609.07008（RedKnot-MLA）、2609.15627（V4-Flash 在 AMD gfx90a 上）、2609.24698（树形投机解码适配 V4）。不引作 DeepSeek 一手来源。

**E18. 官方文档自己改过口径：V4-Pro 是否退役。** NEWS 260910：“Starting at 04:00 UTC on Sept 14, 2026, all deepseek-v4-pro requests will route to V4.1-Flash”；
而 CHANGELOG 2026-09-10 条目（抓取于 09-24）写 “continue providing API services for DeepSeek V4 Pro after September 14, 2026”。两页同日期、内容相反，后者应是事后更新。
另 NEWS 260910 “This will continue until V4.1-Pro launches.”——**V4.1-Pro 尚未发布**，不进时间线。

**E19. V3-0324 的日期差一天。** HF 仓库与 CHANGELOG 为 2025-03-24，新闻页标题 “2025/03/25”。时间线取 03-24（模型名即 0324）。

**E20. DeepSeek-Coder 的语言数：87 还是 86？** dscoder 正文 “retain only 87 programming languages”（截取）；dscoderv2 摘要 “expands its support for programming languages from 86 to 338”。两篇自相矛盾，课文写 “87 种（V2 论文称 86）”。

**E21. CHANGELOG 的 R1 条目有笔误。** 原文 “You can invoke DeepSeek-V3 by specifying model='deepseek-reasoner'”（应为 R1）。只影响引用时的措辞。

**E22. HF `createdAt` 不是发布日。** V2 仓库 2024-04-22 建、05-06 发布；V4 04-22 建、04-24 发布；OCR 10-17 建、10-20 发布；Vision-Exp 的 API 先于 HF（08-21 vs 08-31）。

**E23. “V4 预训练 32T tokens”只对 Flash。** Pro 是 33T；摘要写 “more than 32T”。V4.1 是 45T **多模态** tokens（文本 : 多模态 = 7 : 1）。

**E24. “V4 全部参数用 Muon”——不对。** 嵌入、预测头、RMSNorm 权重仍用 AdamW；V4.1 进一步把嵌入、预测头、Engram 表改用 “Sinkhorn-balanced update”，并对 Q/K 用 head-wise Muon。

**E25. V4-Flash 与 V4-Pro 的前两层不同。** Flash 前两层纯 SWA（`compress_ratios` 前两个 0）；Pro 前两层是 HCA（前两个 128）。

**E26. 访问陷阱。** `api-docs.deepseek.com/news/news250120` 不带尾斜杠会落到 Quick Start 页；openai.com 公告直连返回 403，需走 Wayback 快照。

**E27. “MCTS 在 DeepSeek 失败了”要限定领域。** R1 附录 G.2 说 MCTS 在通用推理的 RL 里没走通，并声明 “this does not imply that these approaches are incapable”；
而 Prover-V1.5 在 Lean 4 形式证明里用 MCTS 变体 RMaxTS 取得当时最好成绩。

**E28. Janus/Janus-Pro 的 1.5B 语言模型从未单独发布。** Janus-Pro 正文说用 “DeepSeek-LLM (1.5B and 7B)”；HF 组织里没有 1.5B 的 DeepSeek-LLM。

---

## 8. 未能核实 / UNVERIFIED

1. **DeepSeek-Coder 的公开发布日。** 只找到 GitHub 最早提交 2023-11-02 与 HF 建库日期（10-23 至 11-01），没有官方公告页（第三方网站写 11-02，未采信为一手来源）。课文写 “2023 年 11 月初”最稳。
2. **2606.19348 为何是 2606 前缀。** “编号按公开（announce）月份分配、提交后长时间待审”是笔者的解释，未找到 arXiv 对这一篇的公告日期记录。
3. **HF 上 V4 报告的两批版本（04-24、05-06；06-22 是删除，不是新版本，见 E2）与 arXiv v1 的差异；V4.1 HF PDF 与 arXiv v1 的差异。** 未下载比对（V4 PDF 已从 HF 删除；历史版本仍可经 LFS 历史取回，但本次未做）。
4. **V4/V4.1 indexer 键 68 字节。** 由 dsv41 §2.4.4（indexer Q/K 用 OCP 标准 MXFP4）与 V4.1 代码注释（每 32 维一个 E8M0）推算，并与论文 3,514、890 两个数同时吻合；但论文、FlashMLA README 均未直接印出 “68”。
5. **V4-Pro 每 token 5,031 字节。** 笔者按同一公式推算，论文只给 “10%”“9.5×”，与之吻合但不是原文数字。
6. **所有图表目视读数**（dsv41 图 2、dsv4 图 1 右）误差约 ±10%，不能当精确值写进课文。
7. ~~InstructGPT 的会议版本（NeurIPS 2022？）本次未核。~~ **已解决**（核查补）：NeurIPS proceedings 页列为 Advances in Neural Information Processing Systems 35（NeurIPS 2022）Main Conference Track，标题与 arXiv 相同。
8. **“V3.2 有 685B 参数”** 之类的说法（网上常见）未核；论文与模型卡未单列参数量。
9. **V4.1 是否完全取消哈希路由。** 只有 config 缺 `num_hash_layers` 与图 3 图注 “All feed-forward layers use standard DeepSeekMoE” 两条间接证据。
10. **DeepSeekMoE 16B 的 “标准 FFN” 中间维 5632**：由 1408 = 0.25 × 5632 反推，论文未印 5632；首层稠密 FFN 的 10,944 与之关系未核。
11. **V4.1 `rms_norm_eps` = 1e-20** 的含义（V4 为 1e-6）未核。
12. **V3-0324、R1-0528、V3.1-Terminus 的上下文长度**：模型卡未写；config 与 V3 相同。
13. **V2.5、V2.5-1210、V2-Chat-0628 的参数量**：模型卡未写，默认与 V2 相同（236B/21B）属推断。
14. **各预印本是否已正式发表**（DeepSeek-VL、Janus、Janus-Pro、DeepSeekMath、VL2、OCR、OCR 2、mHC、DSpark、DSec、V2、V3、V3.2、V4、V4.1）：只看了 arXiv 元数据的 journal/comments 字段。
15. **V4-Flash-0731 是否随权重附带 DSpark 模块。** 模型卡说 “it comes with a speculative decoding module attached”，但 config `num_nextn_predict_layers` 为 1（Vision-Exp 为 3），未核权重索引。
16. **V2 的公开发布时刻**：README 写 2024-05-06，arXiv v1 提交于 05-07 15:56 UTC；二者时区关系（北京时间 05-06 晚还是 05-07）未核。
17. **官方 X / 微信 / 知乎帖子**一概未读；若课文要引用 “开源周 Day 6 是哪一天”，需另核（README 未写日期）。
18. ~~Janus-Pro 表中的 “4096” 是 README 表格的列值，列名未逐字核对。~~ **已解决**（核查补）：januspro 论文表 1 行名 “Context Window”，1B 与 7B 均为 4096；§3 原文 “DeepSeek-LLM (1.5B and 7B) with a maximum supported sequence length of 4096”。
19. **课程符号表需要新增** $m$、$m'$、$n_{win}$、$n_{hc}$、$c$（V4 头维）等符号——这是给 `style-guide.md` 的待办，不是事实核对项，列此提醒。

---

## 9. 参考文献卡片草稿

格式与 `lesson-markdown` SKILL 的 ```refs 块一致；`id` 即正文 `[@id]` 的键。论文与技术报告 38 条（含 GPT-1/GPT-2 两份无 arXiv 的 OpenAI 报告），制品 12 条。第 0 课只需其中主线部分，第 20 课可全用。

```refs
- id: vaswani2017
  title: "Attention Is All You Need"
  a: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017), NIPS 2017, arXiv:1706.03762
  url: https://arxiv.org/abs/1706.03762
  finding: 2017-06-12 提交 arXiv，提出完全基于注意力的 Transformer。
  matters: 科技树的根节点，时间轴从这一天起算（到 V4.1-Flash 共 3,377 天）。
- id: radford2018
  title: "Improving Language Understanding by Generative Pre-Training"
  a: Radford, Narasimhan, Salimans, Sutskever (2018), OpenAI technical report（无 arXiv）
  url: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
  finding: OpenAI 于 2018-06-11 公告，预训练 + 微调的 GPT-1。
  matters: 时间线 GPT 侧第一个 “GPT” 节点，说明它没有 arXiv 版，日期取公告页。
- id: radford2019
  title: "Language Models are Unsupervised Multitask Learners"
  a: Radford, Wu, Child, Luan, Amodei, Sutskever (2019), OpenAI technical report（无 arXiv）
  url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  finding: 2019-02-14 公告 GPT-2（最大 1.5B），展示零样本能力。
  matters: 时间线节点之一，规格引用 lesson-01-doc 的核对结果。
- id: brown2020
  title: "Language Models are Few-Shot Learners"
  a: Brown et al. (2020), NeurIPS 2020, arXiv:2005.14165
  url: https://arxiv.org/abs/2005.14165
  finding: 175B 的 GPT-3，所有模型上下文 2048，2020-05-28 提交。
  matters: KV 阶梯图的起点（每 token 4,718,592 字节）。
- id: ouyang2022
  title: "Training language models to follow instructions with human feedback"
  a: Ouyang et al. (2022), NeurIPS 2022, arXiv:2203.02155v1
  url: https://arxiv.org/abs/2203.02155
  finding: InstructGPT 的三阶段 SFT–RM–PPO；公告与 PDF 2022-01-27，arXiv 2022-03-04。
  matters: 用来纠正 “InstructGPT = 2022 年 3 月” 的说法，并连接第 3 课。
- id: dscoder
  title: "DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence"
  a: Guo, Zhu, Yang, Xie, Dong, Zhang, et al. (2024), preprint, arXiv:2401.14196v2
  url: https://arxiv.org/abs/2401.14196
  finding: 1.3B–33B 代码模型，2T tokens 从头训练，16K 窗口与 FIM 填空；§5 从 LLM-7B 续训出 Coder-v1.5。
  matters: DeepSeek 最早公开的语言模型，也是 DeepSeekMath 血缘的起点。
- id: dsllm
  title: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"
  a: DeepSeek-AI (2024), preprint, arXiv:2401.02954v1
  url: https://arxiv.org/abs/2401.02954
  finding: 7B/67B、2T tokens；67B 为 95 层、GQA-8、上下文 4096（表 2）。
  matters: 规格总表第一行，也是 V4.1 图 1(b) 中 389,120 字节 “DeepSeek-V1” 的出处。
- id: dsmoe
  title: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"
  a: Dai, Deng, Zhao, Xu, Gao, Chen, et al. (2024), preprint, arXiv:2401.06066v1
  url: https://arxiv.org/abs/2401.06066
  finding: 细粒度 + 共享专家；16B 版 28 层、2 + 64 专家激活 6、16.4B/2.8B、2T tokens。
  matters: 主线 B 的起点，规格表第二行。
- id: dsmath
  title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
  a: Shao, Wang, Zhu, Xu, Song, Bi, et al. (2024), preprint, arXiv:2402.03300v3
  url: https://arxiv.org/abs/2402.03300
  finding: 从 Coder-v1.5 7B 续训 120B 数学 token，并提出 GRPO。
  matters: 主线 D 上 GRPO 的首次出现，时间线 2024-02-05 节点。
- id: dsvl
  title: "DeepSeek-VL: Towards Real-World Vision-Language Understanding"
  a: Lu, Liu, Zhang, Wang, Dong, Liu, et al. (2024), preprint, arXiv:2403.05525v2
  url: https://arxiv.org/abs/2403.05525
  finding: 1.3B 与 7B 视觉语言模型，混合编码器处理 1024×1024 图像。
  matters: 多模态旁支的起点，时间线节点。
- id: dsv2
  title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  a: DeepSeek-AI (2024), preprint, arXiv:2405.04434v5
  url: https://arxiv.org/abs/2405.04434
  finding: MLA + DeepSeekMoE，236B/21B、8.1T、128K；表 1 给出四种注意力的 KV 通式；附录 B 给出 V2-Lite。
  matters: 规格表 V2 与 V2-Lite 两行，及 “93.3% 需要 6 bit” 的勘误依据。
- id: prover1
  title: "DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data"
  a: Xin, Guo, Shao, Ren, Zhu, Liu, et al. (2024), preprint, arXiv:2405.14333v1
  url: https://arxiv.org/abs/2405.14333
  finding: 在 800 万条合成 Lean 4 证明上微调 DeepSeekMath 7B。
  matters: 形式证明旁支的起点，时间线节点。
- id: dscoderv2
  title: "DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence"
  a: DeepSeek-AI (2024), preprint, arXiv:2406.11931v1
  url: https://arxiv.org/abs/2406.11931
  finding: 从 V2 中间检查点续训 6T tokens，上下文 16K → 128K，语言 86 → 338 种。
  matters: V2.5 合并前的代码分支，时间线节点；另用于 “86 vs 87” 勘误。
- id: esft
  title: "Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models"
  a: Wang, Chen, Dai, Xu, Li, Wu (2024), EMNLP 2024, arXiv:2407.01906v2
  url: https://arxiv.org/abs/2407.01906
  finding: 只微调与任务最相关的专家即可匹配全参微调，实验骨干为 V2-Lite。
  matters: 主线 B 的旁支节点，说明细粒度专家对下游适配的好处。
- id: prover15
  title: "DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search"
  a: Xin, Ren, Song, Shao, Zhao, Wang, et al. (2024), preprint, arXiv:2408.08152v1
  url: https://arxiv.org/abs/2408.08152
  finding: RLPAF 加 MCTS 变体 RMaxTS，miniF2F 测试 63.5%。
  matters: 第 20 课对照：MCTS 在形式证明里有效，而 R1 在通用推理里放弃了它。
- id: fireflyer
  title: "Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning"
  a: An, Bi, Chen, Chen, Deng, Ding, et al. (2024), SC'24, arXiv:2408.14158v2
  url: https://arxiv.org/abs/2408.14158
  finding: 1 万张 PCIe A100 的 Fire-Flyer 2 以约一半成本接近 DGX-A100 性能，配套 HFReduce、3FS。
  matters: 主线 C 的基础设施起点，也是 DSec 所用 3FS 的来源。
- id: alf
  title: "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"
  a: Wang, Gao, Zhao, Sun, Dai (2024), preprint, arXiv:2408.15664v1
  url: https://arxiv.org/abs/2408.15664
  finding: 用按负载更新的专家偏置代替辅助损失，在 ≤3B、≤200B tokens 上验证。
  matters: V3 起无辅助损失均衡的出处，第 20 课 “辅助损失被替换” 一条的依据。
- id: janus
  title: "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation"
  a: Wu, Chen, Wu, Ma, Liu, Pan, et al. (2024), preprint, arXiv:2410.13848v1
  url: https://arxiv.org/abs/2410.13848
  finding: 理解与生成用两条独立视觉编码通路、共享一个 Transformer。
  matters: 多模态旁支时间线节点。
- id: janusflow
  title: "JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation"
  a: Ma, Liu, Chen, Liu, Wu, Wu, et al. (2024), CVPR 2025, arXiv:2411.07975v2
  url: https://arxiv.org/abs/2411.07975
  finding: 在语言模型框架内直接训练 rectified flow 做图像生成。
  matters: 多模态旁支时间线节点。
- id: dsvl2
  title: "DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding"
  a: Wu, Chen, Pan, Liu, Liu, Dai, et al. (2024), preprint, arXiv:2412.10302v1
  url: https://arxiv.org/abs/2412.10302
  finding: 动态切图 + MLA-MoE 语言模型，三档激活 1.0B/2.8B/4.5B（语言部分 0.57B/2.4B/4.1B）。
  matters: MLA 进入多模态的节点，也是激活参数口径勘误（E12）的来源。
- id: dsv3
  title: "DeepSeek-V3 Technical Report"
  a: DeepSeek-AI (2024), preprint, arXiv:2412.19437v2
  url: https://arxiv.org/abs/2412.19437
  finding: 671B/37B、61 层、1 + 256 专家激活 8、14.8T tokens，全程训练 2.788M H800 小时（$5.576M，不含前期研究）。
  matters: 规格表 V3 行、成本卡片与 “不丢 token” 一条的依据。
- id: dsr1
  title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
  a: DeepSeek-AI (2025), Nature 645:633–638, arXiv:2501.12948v2
  url: https://arxiv.org/abs/2501.12948
  finding: 在 V3-Base 上纯 RL 得到 R1-Zero，再经多阶段得到 R1；表 7 训练成本 147K H800 小时（$294K）；附录 G.2 记录 PRM/MCTS 未成功。
  matters: 时间线 2025-01-20 主干节点，第 20 课 “放弃的路线” 的主要来源。
- id: januspro
  title: "Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling"
  a: Chen, Wu, Liu, Pan, Liu, Xie, et al. (2025), preprint, arXiv:2501.17811v1
  url: https://arxiv.org/abs/2501.17811
  finding: 把 Janus 放大到 7B，基座为 DeepSeek-LLM 1.5B/7B。
  matters: 多模态旁支节点；也揭示一个从未单独发布的 1.5B DeepSeek-LLM。
- id: codeio
  title: "CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction"
  a: Li, Guo, Yang, Xu, Wu, He (2025), ICML 2025, arXiv:2502.07316v4
  url: https://arxiv.org/abs/2502.07316
  finding: 把代码改写成输入/输出预测任务，用自然语言 CoT 训练，提升多类推理。
  matters: 主线 D 旁支节点（R1 之后的推理数据研究）。
- id: nsa
  title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
  a: Yuan, Gao, Dai, Luo, Zhao, Zhang, et al. (2025), ACL 2025 (Best Paper), arXiv:2502.11089v2
  url: https://arxiv.org/abs/2502.11089
  finding: 压缩、选择、滑窗三分支的可训练稀疏注意力，在 27B/3B 骨干上不输全注意力。
  matters: 主线 A 从 MLA 走向 DSA、CSA 的中间节点，时间线 2025-02-16。
- id: grm
  title: "Inference-Time Scaling for Generalist Reward Modeling"
  a: Liu, Wang, Xu, Ma, Ruan, Li, et al. (2025), preprint, arXiv:2504.02495v3
  url: https://arxiv.org/abs/2504.02495
  finding: 用 SPCT 训练生成式奖励模型 DeepSeek-GRM，并靠并行采样 + 元 RM 做推理期扩展。
  matters: 主线 D 旁支节点，说明奖励模型也在走 “推理期扩展”。
- id: proverv2
  title: "DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition"
  a: Ren, Shao, Song, Xin, Wang, Zhao, et al. (2025), preprint, arXiv:2504.21801v2
  url: https://arxiv.org/abs/2504.21801
  finding: 以 V3 分解子目标做冷启动，671B 版 miniF2F-test 88.9%。
  matters: 形式证明旁支接入 V3 主干的节点。
- id: insights
  title: "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures"
  a: Zhao, Deng, Ruan, Dai, Gao, Li, et al. (2025), ISCA '25 Industry Track, arXiv:2505.09343v2
  url: https://arxiv.org/abs/2505.09343
  finding: 总结 V3 在 2,048 张 H800 上的软硬件协同，表 1 给出 V3 每 token KV 70.272 KB。
  matters: 交叉验证 N1 的 V3 数字，并作为主线 C 的时间线节点。
- id: dsocr
  title: "DeepSeek-OCR: Contexts Optical Compression"
  a: Wei, Sun, Li (2025), preprint, arXiv:2510.18234v1
  url: https://arxiv.org/abs/2510.18234
  finding: 文本 token 不超过视觉 token 10 倍时 OCR 精度约 97%，20 倍时约 60%。
  matters: 主线 E “视觉压缩上下文” 的节点，时间线 2025-10-20。
- id: dsmathv2
  title: "DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning"
  a: Shao, Luo, Lu, Ren, Hu, Ye, et al. (2025), preprint, arXiv:2511.22570v1
  url: https://arxiv.org/abs/2511.22570
  finding: 基于 V3.2-Exp-Base 训练可自验证的证明生成器，Putnam 2024 得 118/120。
  matters: 时间线节点，展示 V3.2-Exp 基座的派生模型。
- id: dsv32
  title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
  a: DeepSeek-AI (2025), preprint, arXiv:2512.02556v1
  url: https://arxiv.org/abs/2512.02556
  finding: 以 V3.1-Terminus 为起点加入 DSA（top-2048），续训 2.1B + 943.7B tokens，上下文 128K。
  matters: 规格表 V3.2 行与 KV 48,068 字节的结构依据。
- id: mhc
  title: "mHC: Manifold-Constrained Hyper-Connections"
  a: Xie, Wei, Cao, Zhao, Deng, Li, et al. (2025), preprint, arXiv:2512.24880v2
  url: https://arxiv.org/abs/2512.24880
  finding: 把超连接的残差矩阵约束为双随机矩阵，n = 4 时训练开销 6.7%。
  matters: 主线 E 节点；V4 直接采用、V4.1 改为 Single-Pass mHC。
- id: engram
  title: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"
  a: Cheng, Zeng, Dai, Chen, Wang, Xie, et al. (2026), ACL 2026, arXiv:2601.07372v2
  url: https://arxiv.org/abs/2601.07372
  finding: 以 N-gram 哈希查表作条件记忆，发现 MoE 与记忆分配的 U 形规律，放大到 Engram-27B。
  matters: 主线 E 节点；V4.1 以 196B 参数规模把它用进生产模型。
- id: dsocr2
  title: "DeepSeek-OCR 2: Visual Causal Flow"
  a: Wei, Sun, Li (2026), preprint, arXiv:2601.20552v1
  url: https://arxiv.org/abs/2601.20552
  finding: DeepEncoder V2 按图像语义动态重排视觉 token 再交给 LLM。
  matters: 视觉压缩支线的延续节点，时间线 2026-01-27。
- id: dsv4
  title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
  a: DeepSeek-AI (2026), preprint, arXiv:2606.19348v1
  url: https://arxiv.org/abs/2606.19348
  finding: 预览版 V4-Pro 1.6T/49B 与 V4-Flash 284B/13B，CSA/HCA 混合注意力、mHC、Muon，1M 上下文下 Pro 仅需 V3.2 的 27% FLOPs 与 10% KV。
  matters: 规格表 V4 两行、KV 3,514 字节的结构依据，及 “编号前缀 2606 但 4 月提交” 的勘误。
- id: dspark
  title: "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
  a: Cheng, Yu, Shao, Li, Xiong, Qian, et al. (2026), preprint, arXiv:2607.05147v1
  url: https://arxiv.org/abs/2607.05147
  finding: 半自回归草稿 + 置信度调度验证，在 V4 线上服务中、同等总吞吐下比 MTP-1 基线把单用户生成速度提升 60%–85%（V4-Flash）与 57%–78%（V4-Pro）。
  matters: 解释 V4.1 为何在预训练中去掉 MTP，时间线 2026-07 节点。
- id: dsv41
  title: "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"
  a: DeepSeek-AI (2026), preprint, arXiv:2609.19969v1
  url: https://arxiv.org/abs/2609.19969
  finding: 552B backbone（另 196B Engram）、prefill 8B / decode 16B 激活，CED + CSA2 + FP4 KV 使全局 KV 为 890 字节/token，45T 多模态 tokens。
  matters: 课程时间线终点；图 1(b) 的四代 KV 数字是第 0 课 KV 阶梯图的一手数据。
- id: dsec
  title: "DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale"
  a: Huang, Tang, Chen, Liu, Chen, Cheng, et al. (2026), preprint, arXiv:2609.22978v1
  url: https://arxiv.org/abs/2609.22978
  finding: 统一 SDK 暴露四类沙箱后端，单单元约 160 节点、每天约 300 万个沙箱。
  matters: 时间线上最新的 DeepSeek 论文（2026-09-19），第 20 课讲智能体 RL 基础设施时引用。
- id: dsnews
  title: "DeepSeek API News (18 release posts, 2024-07 to 2026-09)"
  a: DeepSeek (2024–2026)
  url: https://api-docs.deepseek.com/news/news260910
  finding: 逐篇给出 V2.5、R1-Lite、V3、R1、V3-0324、R1-0528、V3.1、Terminus、V3.2-Exp、V3.2、V4 预览、V4-Pro GA、V4-Flash-Vision-Exp、V4.1-Flash 的发布日与要点。
  matters: 时间线 “公开发布日” 列的首选来源。
  kind: artefact
- id: dschangelog
  title: "DeepSeek API Change Log"
  a: DeepSeek (2024–2026)
  url: https://api-docs.deepseek.com/updates/
  finding: 2024-05-17 至 2026-09-10 共 21 个日期条目（其中 2024-07-25 新 API 功能、2024-08-02 硬盘缓存两条不是模型切换），含 V2-0517、Coder-V2-0614、V2-0628、Coder-V2-0724、V4-Flash-0731。
  matters: 补齐没有新闻页的版本日期，并暴露与新闻页不一致之处（E18）。
  kind: artefact
- id: hforg
  title: "deepseek-ai organization on Hugging Face (model list, configs, model cards, commit histories)"
  a: DeepSeek-AI (2023–2026)
  url: https://huggingface.co/deepseek-ai
  finding: 105 个模型仓库的建库时间，各主线模型 config.json 的层数、维度、专家、`compress_ratios` 等字段。
  matters: 规格表 “C” 列的来源，也用来复现 V4/V4.1 的 KV 字节。
  kind: artefact
- id: ghorg
  title: "deepseek-ai organization on GitHub (39 repositories)"
  a: DeepSeek-AI (2023–2026)
  url: https://github.com/deepseek-ai
  finding: 仓库创建日与最早提交，V2 README 的 “2024.05.06 / 05.16” 发布条目，开源周 README。
  matters: 为没有新闻页的早期版本（Coder、LLM、MoE、V2）提供发布日期证据。
  kind: artefact
- id: flashmla
  title: "FlashMLA README (KV cache byte formats for V3.2 / V4 / V4.1)"
  a: DeepSeek-AI (2025–2026)
  url: https://github.com/deepseek-ai/FlashMLA
  finding: V3.2 每 token 每层 656 字节；V4 584 字节；V4.1 FP8 528 字节、FP4 288 字节。
  matters: 式 (0.3)–(0.5) 的字节常数来源，使 48,068 / 3,514 / 890 可复现。
  kind: artefact
- id: dsv41code
  title: "DeepSeek-V4.1-Flash inference/model.py"
  a: DeepSeek-AI (2026)
  url: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py
  finding: 注释写明压缩 KV 用每 16 个一组的 E4M3 缩放、indexer 用每 32 个一组的 E8M0 缩放；只有 kv_source 层自己压缩 KV。
  matters: 推出 indexer 键 68 字节，是 890 与 3,514 能同时复现的关键一环。
  kind: artefact
- id: dsv4card
  title: "DeepSeek-V4-Flash / V4-Pro / V4-Flash-0731 / V4-Pro-0813 / V4-Flash-Vision-Exp model cards"
  a: DeepSeek-AI (2026)
  url: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813
  finding: 预览版参数表（284B/13B、1.6T/49B、1M、FP4 + FP8 混合）与正式版 “same model structure, re-post-trained” 说明。
  matters: 区分 V4 “预览版” 与 “正式版 / GA” 两个时间点。
  kind: artefact
- id: dsv3card
  title: "DeepSeek-V3.1 model card"
  a: DeepSeek-AI (2025)
  url: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
  finding: V3.1-Base 在 V3 基础上把 32K 阶段扩到 630B tokens、128K 阶段扩到 209B tokens，并采用 UE8M0 FP8 缩放格式。
  matters: 复算新闻页 “840B tokens” 与 V3.2 累计训练量的依据。
  kind: artefact
- id: oaiwb
  title: "OpenAI announcements via Wayback Machine (GPT-1, GPT-2, InstructGPT, ChatGPT)"
  a: OpenAI (2018–2022)
  url: http://web.archive.org/web/20221205235144/https://openai.com/blog/chatgpt/
  finding: 四篇公告日期分别为 2018-06-11、2019-02-14、2022-01-27、2022-11-30。
  matters: GPT 侧时间线唯一可存档的一手日期来源（openai.com 直连 403）。
  kind: artefact
- id: naturer1
  title: "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning (Nature article page)"
  a: DeepSeek-AI (2025), Nature 645, 633–638, doi:10.1038/s41586-025-09422-z
  url: https://www.nature.com/articles/s41586-025-09422-z
  finding: 2025-02-14 收稿、07-17 接收、09-17 在线发表，标题与 arXiv 版不同。
  matters: 时间线里 R1 的第二个日期，以及 E6 勘误的依据。
  kind: artefact
- id: acl2025awards
  title: "ACL 2025 Awards page"
  a: ACL (2025)
  url: https://2025.aclweb.org/program/awards/
  finding: NSA 列于 Best Paper 四篇之一。
  matters: 支持 “NSA 获 ACL 2025 最佳论文” 的表述，课程地图第 14 课也用到。
  kind: artefact
- id: openinfra
  title: "open-infra-index README (DeepSeek Open-Source Week, Feb 2025)"
  a: DeepSeek-AI (2025)
  url: https://github.com/deepseek-ai/open-infra-index
  finding: 2025-02-24 起连续开源 FlashMLA、DeepEP、DeepGEMM、DualPipe/EPLB、3FS，Day 6 公布推理系统成本利润率 545%。
  matters: 时间线 “开源周” 节点与主线 C 的制品清单。
  kind: artefact
```

---

## 10. 核查记录（2026-09-24，对抗式复核）

复核人按 `academic-research` SKILL 的 Reconcile 四项（版本漂移、论文 vs 代码、传话失真、邻近论断替换）逐条重查。所有核对都**直接对一手来源重做**，不沿用本笔记原来的结论：
arXiv 用 `arxiv.py grep/text/meta` 与 export API；arXiv 图用原图（arxiv.org/html/…/figures/*.png）目视；HF 用 `config.json`、模型卡、commits API 与提交 diff 页；
GitHub 用 REST API；DeepSeek API 文档抓 HTML 抽 `<article>`；OpenAI 公告经 Wayback `id_` 快照；Nature 取页面 `citation_*` 元数据；§4 全部算式在 Python 3.11 重算。

### 10.1 核对过且确认无误的条目

| # | 论断 | 一手来源与结果 |
|---|---|---|
| 1 | V4 arXiv v1 提交 “Sun, 26 Apr 2026 14:49:33 UTC”，编号 2606.19348，无 v2（E1） | arXiv abs 页 Submission history ✓ |
| 2 | V4 HF “Release DeepSeek-V4” 提交 2026-04-24T02:24:45Z（Flash；Pro 02:25:01） | HF commits API ✓ |
| 3 | V4-Flash 43 层 / 4096 / 284B / 13B / 32T；V4-Pro 61 层 / 7168 / 1.6T / 49B / 33T；CSA m=4、HCA m'=128、top-k 512/1024、64/128 头、c=512、$d_c$ 1024/1536、g=8/16、SWA 128、1+256(2048)/1+384(3072) 激活 6、前 3 MoE 层哈希、MTP 1、$n_{hc}$=4、$t_{max}$=20（S7、S8） | dsv4 §4.2.1 逐句 ✓；config `compress_ratios` 计数 Flash 21×4 + 20×128 + 2×0、Pro 30×4 + 31×128 ✓ |
| 4 | V4 1M 处 Pro 27% FLOPs（equivalent FP8）/10% KV，Flash 10%/7%；图 1 标注 3.7×/9.8×/9.5×/13.7×（K5） | dsv4 §1 原句 ✓；原图目视四个标注 ✓ |
| 5 | “approximately 2% times of that baseline”（K6）与 “approximately 8 times”（N18） | dsv4 §2.3.4、§3 原句 ✓ |
| 6 | dsv4 式 (1)(11)(12)(16)(17)(22)(23)(27) 与算法 1 第 4–8 行（§2.3） | LaTeXML 源逐符号比对 ✓（上下标、Softmax 下标 row、Top-k 写法、$\sqrt{\max(n,m)}$、$(1-\eta\lambda)$ 全部一致） |
| 7 | V4.1 arXiv v1 “Thu, 17 Sep 2026 09:43:10 UTC” | arXiv abs 页 ✓ |
| 8 | V4.1 “552B backbone parameters and 196B Engram parameters”，prefill 8B / decode 16B，45T 多模态 tokens，40 层（20+20）等 S9 全部超参数 | dsv41 §2.1、§4.2.1、§4.2.2 逐句 ✓；config `kv_source_layer_ids` [2,8,14,20]、`index_n_heads` 32、`q_lora_rank` 1280、`o_lora_rank` 1024、`moe_intermediate_size` 2304 ✓ |
| 9 | V4.1 图 1(b) 四根柱 389,120 / 48,068 / 3,514 / 890 与 8.1× / 13.7× / 3.9×；图注 “approximately 4-fold and 437-fold” | 原图目视 ✓；图注原句 ✓ |
| 10 | V4.1 图 2 “weighting BF16, FP8, and FP4 operations by 1, 0.5, and 0.25”；四条曲线目视读数 | 图注原句 ✓；按对数刻度像素重读：V1 ≈149→≈3,430，V3.2 ≈78→≈600，V4-Flash ≈17→≈61，V4.1 ≈21→≈26（GFLOPs），均在 §5.1 所写 ±10% 内 ✓ |
| 11 | V4.1 “using only 1/3 total parameters and 1/4 activated parameters”（对比 V4-Pro-Base，E11） | dsv41 §1 原句 ✓；比值 N15 重算 ✓ |
| 12 | dsv41 式 (1)(2)(6)、prefill 复杂度式、mHC 读写量 $(2n+2)d$/$(4n+4)d$/$(3n+2)d$、FP4 “448×6=2688”、SWA Bounded Replay 区间 $[\max(s,i-W+1),i]$ | 逐符号比对 ✓（§2.4(c) 补了论文的前提，见 10.2） |
| 13 | FlashMLA：V3.2 656 B（512 FP8 + 4×FP32 + 64×BF16）；V4 584 B（448 FP8 + 128 BF16 + 7 个 E8M0 + 1 填充）；V4.1 FP8 528 B；V4.1 FP4 288 B（256 + 32 个 E4M3）；“Q-norm (only used in V4, not V4.1)”；2026.09.10 内核新闻 | FlashMLA README ✓ |
| 14 | V2：93.3%、“6 bits on average”、42.5%（300.6K vs 172.8K）、5.76 倍；表 1 四行通式；“$d_c$ is set to $4d_h$ … $\frac{d_h}{2}$”；“2.25 groups”；V2/V2-Lite 超参数 | dsv2 摘要、§2.1.2 表 1、§3.1.2、§3.2.3、附录 B.1 ✓；arXiv v1 “Tue, 7 May 2024 15:56:43 UTC”，v5 2024-06-19 ✓ |
| 15 | DeepSeek LLM 表 2 两行（7B：30/4096/32/32/4096/2304/4.2e-4/2.0T；67B：95/8192/64/8/4096/4608/3.2e-4/2.0T）与 “67B model uses GQA” | dsllm 表 2 与 §2.2 ✓；HF config ✓ |
| 16 | V3：61 层 / 7168 / 激活 8 且至多 4 节点；2.788M H800 小时 = $5.576M；每万亿 token 180K 小时、2048 卡 3.7 天；不含前期研究；“does not drop any tokens” | dsv3 原句 ✓ |
| 17 | R1：v1 2025-01-22 15:19:35 UTC，v2 2026-01-04 03:57:36 UTC；v2 表 7（101K/5K/41K/147K；$202K/$10K/$82K/$294K）；“660B R1-Zero and R1”；v1 无 “GPU hours”“660B”，v1 “Unsuccessful Attempts” 为 §4.2、v2 为附录 G.2；Nature 标题、2025-02-14 收稿、07-17 接收、09-17 在线、645:633–638 | arXiv abs 页；v1、v2 分别 grep ✓；Nature `citation_*` 元数据 ✓ |
| 18 | V3.2 式 (1)(2)；“select 2048 key-value tokens”；2.1B 与 943.7B；“maximum context length of only 128K”；基于 V3.1-Terminus；config 64/128/2048 | dsv32 ✓；HF config ✓ |
| 19 | GPT 侧：GPT-1 “June 11, 2018”、GPT-2 “February 14, 2019”、InstructGPT “January 27, 2022”（页面已链 PDF）、ChatGPT “November 30, 2022”，四个首句；Transformer / GPT-3 / InstructGPT arXiv v1 日期 | Wayback `id_` 快照 ✓；arXiv 元数据 ✓ |
| 20 | API 新闻页 260424（1M 默认、Tech Report 链接）、260910（552B、8B/16B、1/4 HBM、1/8 SSD、Sept 14 路由、V4.1-Pro、`deepseek-flash`）、260813、260821、251201、250929、250821、250325、250120、1226、1210、1120、0905 的引文；Change Log 中 E18 两段相反表述、E21 笔误、T27 AIME “70.0 → 87.5 (+17.5)” | 逐页抓 HTML ✓ |
| 21 | NSA 为 ACL 2025 四篇 Best Paper 之一、pp. 23078–23097；Engram ACL 2026 pp. 4968–4990、14 位作者、2026 年 7 月；arXiv v2 21 人 | ACL 奖项页与 Anthology ✓；arXiv API ✓ |
| 22 | mHC “6.7%”（n = 4）与 mHC 式 (3)；V4 式 (1) 的 $A_l,B_l,C_l$ 与 $\mathcal H^{pre},\mathcal H^{res},\mathcal H^{post\top}$ 对应（$\mathcal H^{post}\in\mathbb R^{1\times n}$） | mhc 全文 ✓ |
| 23 | 早期发布日：DeepSeek-Coder 仓库 2023-10-20 建、最早提交 “init project” 2023-11-02T14:07:09Z；DeepSeek-LLM 2023-11-29；DeepSeekMoE 最早提交 2024-01-11T02:35:17Z；V2 README “2024.05.06/05.16” 与 “DeepSeek-V1 (Dense-67B)”；HF createdAt（V2 2024-04-22、V3 2024-12-25、Vision-Exp 2026-08-31、V4-Flash-DSpark 2026-06-27）；HF 组织 105 个模型、GitHub 组织 39 个仓库、无 R2 | GitHub / HF API ✓ |
| 24 | 其它论文引文：Coder “87 programming languages”、“16K window”、v1.5 从 LLM-7B 续训 2T；Coder-V2 “from 86 to 338”；VL2 1.0/2.8/4.5B 与 0.57/2.4/4.1B、README 注释 3.37B/16.1B/27.5B 与 “activated 2.4B/4.2B”；Math 120B；Prover 800 万；Prover-V1.5 63.5%；Fire-Flyer 10,000 PCIe A100；ALF ≤3B/≤200B；Prover-V2 88.9%；Insights 2,048 H800 与表 1 70.272 KB；OCR 97% 与 A570M；Math-V2 118/120；DSec 160 节点、300 万/天；各论文 v1 与最新版日期 | arXiv 全文 / 摘要 / API ✓ |
| 25 | §4 N1–N24 全部数字 | Python 重算，逐项与笔记一致 ✓（仅 N13 的 67B “0.20” 为 0.196 四舍五入） |

### 10.2 本次修改（改了什么、为什么、依据）

1. **Change Log 条目数 “24 条” → 21 个日期条目（22 个小节）**（§0.3、参考卡 dschangelog）。依据：2026-09-24 抓 https://api-docs.deepseek.com/updates/ ，`<article>` 内 `h2` “Date: …” 恰 21 个（2024-05-17 至 2026-09-10），`h3` 22 个；其中 2024-07-25（New API Features）与 2024-08-02（Context Caching on Disk）不是模型切换，卡片原写 “模型切换记录” 也一并改正。
2. **V4 技术报告在 HF 上的版本史更正**（§3.5、E2、§0.4、§8 第 3 条）。原写 “04-24、05-06、06-22 三次更新”。依据：V4-Pro 提交 `b5968e9` 与 V4-Flash 提交 `60d8d70`（均 2026-06-22，标题 “Update technical report”）的 diff 页都显示 “DeepSeek_V4.pdf DELETED”（删前 LFS 大小 4,480,407 字节），同时 README 链接由 HF PDF 改为 arxiv.org/abs/2606.19348；两个仓库当前文件树中已无 PDF。所以 06-22 是**删除**，这也是新闻页链接 404 的直接原因；真正的 PDF 版本只有 04-24 与 05-06 两批，且 05-06 版晚于 arXiv v1。研究代理报告中的 “HF report updated on 04-24, 05-06 and 06-22” 同样属于这个误读。
3. **作者人数少算了团体条目**（§0.2 六行）：dsllm 88→86、dsv2 157→156、dscoderv2 40→39、dsv32 264→263、dsv4 319→318、dsv41 593→591。依据：arXiv export API 作者字段；原数把 “DeepSeek-AI” 团体名（dsllm、dsv41 还有一个 “:” 占位）也算成了人。表下加了口径说明。
4. **DSpark 60%–85% 被推广到整个 V4**（T39、参考卡 dspark）。依据：dspark 2607.05147 §1 “60%–85% (V4-Flash) and 57%–78% (V4-Pro) at matched aggregate throughput capacities”；摘要只写 60–85 percent。属邻近论断替换：Flash 的区间被写成 “V4 线上服务” 的区间。
5. **补上 FP4 动态范围论证的前提**（§2.4(c)）。原文只写 “L2 范数至多约 √512”，漏了论文的假设 “the largest trained RMSNorm weight magnitude is approximately 1”；按 SKILL “定理要带假设” 补入。
6. **indexer 68 字节的依据加强**（K8、§2.5 说明、§8 第 4 条）。dsv41 §2.4.4 原文已说 V4 的 indexer Q/K 用 FP4 QAT，并 “adopt the OCP-standard MXFP4 format”（每 32 个元素一个 E8M0），与代码注释互证；“68” 本身仍未印出，保留在 §8。
7. **两条 UNVERIFIED 已解决**：§8 第 7 条 InstructGPT 会议版本 = NeurIPS 2022 Main Conference Track（proceedings.neurips.cc，Advances in NeurIPS 35），同步改 §0.2 与参考卡；§8 第 18 条 Janus-Pro 的 4096 = 论文表 1 “Context Window”，且 §3 原文 “maximum supported sequence length of 4096”。
8. **目视读数微调**（N13 末句、§5.1 dsv4 图 1 行）：V3.2 累计 KV 在 1024K 处按像素重读约 48.6 GB，原写 “约 49–50/约 50” 改为 “约 48–50”，并注明图上无法区分 GB 与 GiB。结论（与 N13 的 50.4 GB 在目视误差内一致）不变。

### 10.3 复核后仍未能确认

- arXiv 编号 2606 前缀与 04-26 提交日的关系：arXiv 官方说明页（info.arxiv.org/help/arxiv_identifier）只说 YYMM 是年月，没写编号在提交时还是公告时分配，笔记里 “公告时分配、长时间待审” 的解释仍是推测（§8 第 2 条保留）。
- V4 首份 PDF 是否随 04-24 02:25 的 Release 提交一起上传：只核了 02:33 提交（仅改 README 链接），Release 提交的 diff 未逐一打开。
- HF 上 V4 的 05-06 版 PDF 与 arXiv v1 的差异、V4.1 HF PDF 与 arXiv v1 的差异：未取回比对。
- “数字印在图上” 这一类只针对 dsv41 图 1(b) 与 dsv4 图 1 做了原图核对；dsv2 图 1(b)、dsv41 图 3–5 未看原图。
- lesson-01/02/04/09/14-doc 的交叉引用（GPT-1 PDF CreationDate、V3.2 indexer 132 字节等）本次未重查，沿用那些笔记自己的核查结论。



