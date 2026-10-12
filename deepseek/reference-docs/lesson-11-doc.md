# 第 11 课研究笔记：DeepSeek-V3（上）——sigmoid 门控、无辅助损失负载均衡与多 token 预测（MTP）

> 用途：第 11 课 `lesson-11-v3-architecture` 的全部论文数字、公式、超参数、勘误的唯一来源。
> 研究日期 2026-09-24。所有数字都来自下列一手来源，派生数字全部用 Python 复算（见 §4）。
> 引文遵守版权约束：每条原文只摘关键短语（英文原样，≤15 词），其余用中文转述并给出精确位置，
> 写课时如需整句请回到所注位置核对。公式按 LaTeX 源码逐字转录（V3 取自 arXiv v2 的 LaTeX 源 `main.tex`）。
> 与第 7 课（DeepSeekMoE）、第 9 课（MLA）、第 10 课（V2 路由/均衡损失）、第 12 课（FP8/DualPipe/部署）有交叉，
> 本笔记只在需要对照时引用它们的内容。

---

## 0. 覆盖范围与所读版本

### 0.1 论文

| 简称 | 题目 | 作者 | arXiv id / 所读版本 | 状态 / 出处 | 日期 |
|---|---|---|---|---|---|
| **V3** | DeepSeek-V3 Technical Report | DeepSeek-AI（arXiv 作者栏共 200 条：机构名 “DeepSeek-AI” + Aixin Liu 等 199 人） | 2412.19437，**读 v2**（同时取 v1 全文做 diff） | 技术报告，arXiv 预印本 | v1 2024-12-27；v2 2025-02-18 |
| **LossFree** | Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts | Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, Damai Dai | 2408.15664，**v1（唯一版本）** | arXiv 预印本；OpenReview 显示投 ICLR 2025，venueid = `ICLR.cc/2025/Conference/Rejected_Submission` | 2024-08-28 |
| **Gloeckle** | Better & Faster Large Language Models via Multi-token Prediction | Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve | 2404.19737，**v1（唯一版本）** | ICML 2024，PMLR 235:15706–15734（PMLR 页面元数据核对） | arXiv 2024-04-30；PMLR 2024-07-08 |
| **V2**（对照） | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek-AI | 2405.04434，读 **v5** | arXiv 预印本 | v5 2024-06-19 |
| **Insights**（对照） | Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures | Chenggang Zhao 等 15 人 | 2505.09343，读 **v2**（v1 对照了 MTP 与带宽两段，措辞相同） | ISCA ’25 Industry Track，DOI 10.1145/3695053.3731412 | v1 2025-05-14；v2 2025-12-23 |
| EAGLE（仅元数据） | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | Yuhui Li 等 | 2401.15077 v3 | — | 2024-01-26 |
| Leviathan（仅元数据） | Fast Inference from Transformers via Speculative Decoding | Leviathan, Kalman, Matias | 2211.17192 v2 | ICML 2023 Oral | 2022-11-30 |
| Switch（仅元数据） | Switch Transformers | Fedus, Zoph, Shazeer | 2101.03961 v3 | JMLR | 2021-01-11 |
| DeepSeekMoE（仅元数据） | DeepSeekMoE: Towards Ultimate Expert Specialization in MoE LMs | Damai Dai 等 | 2401.06066 v1 | — | 2024-01-11 |

**读法**：V3 用 arXiv LaTeXML HTML（v1、v2 各一份）+ arXiv 源码包（v2 的 `main.tex`，公式逐字转录来源）+ v2 PDF（只用来定页码）。
LossFree、Gloeckle、V2、Insights 用 LaTeXML HTML（公式取 `alttext`）。论文图用 HTML 附带的 PNG 或源码包里的 PDF 图的文字层查看，未拷入仓库。

**V3 v1 → v2 的差异（逐行 diff 两个 LaTeXML 文本得出）**：与本课有关的 §2（架构）、§4.2（超参数）、§4.5（消融）、§5.4.3（MTP 评估）、附录 C **完全相同**。
全部差异只有三处：
1. §3.5.2 “Higher FP8 GEMM Accumulation Precision in Tensor Cores” 一段整段改写（v1 说“at least 34-bit precision is required”，v2 改为描述 14 位累加并在 CUDA core 上每 128 次乘加提升到 FP32）——属于第 12 课。
2. §3.5.1 一个排版错误：v1 “tensor cores remain entirely -utilized.” → v2 “under-utilized”。
3. 表 6（chat 模型对比）DeepSeek-V3 的 DROP (3-shot F1) 单元格 v1 渲染为 “91.” → v2 “91.6”。

### 0.2 制品（artefact，无论文；抓取日期 2026-09-24）

| 简称 | 内容 | URL |
|---|---|---|
| HF-config | DeepSeek-V3 `config.json`（与 V3-Base 逐字相同；V3-0324 只差 `transformers_version`） | https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json |
| HF-modeling | `modeling_deepseek.py`（`MoEGate` 类，仅推理） | https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/modeling_deepseek.py |
| HF-index | `model.safetensors.index.json`（权重张量清单） | https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/model.safetensors.index.json |
| GH-model | 官方推理代码 `inference/model.py`（`Gate`、`MoE` 类）及其提交历史 | https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py |
| GH-config | `inference/configs/config_671B.json`（另有 16B、236B、v3.1 配置） | https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference/configs |
| GH-README / README_WEIGHTS | 仓库 README 与权重说明 | https://github.com/deepseek-ai/DeepSeek-V3 |
| vLLM-MTP | `vllm/model_executor/models/deepseek_mtp.py`（main 分支最后提交 2026-09-24） | https://github.com/vllm-project/vllm |
| SGLang-NextN | `python/sglang/srt/models/deepseek_nextn.py`（最后提交 2026-09-19） | https://github.com/sgl-project/sglang |
| 后续配置 | V3.1、V3.2-Exp、V3.2、R1、V4-Flash、V4-Pro 的 `config.json`（仅用于“版本漂移”对照） | https://huggingface.co/deepseek-ai/ |
| V2-config | DeepSeek-V2 `config.json`（对照 V2 路由字段） | https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json |

**注意**：DeepSeek 没有公开 V3 的**训练**代码。偏置更新、序列级均衡损失、MTP 损失都只有论文文字，没有代码可对照；
HF 与 GitHub 的代码都是推理代码（HF 的 `MoEGate.forward` 里有 `assert not self.training`）。

---

## 1. 核心论断与原文引文

记号：`V3 §x / p.N` = V3 v2 PDF 第 N 页。引文为英文原文短语。

### A. 规格与总体

**C1 总参数 / 激活参数。** V3 摘要：“671B total parameters with 37B activated for each token”。§4.2（p.23）同句重复。
**C2 层数与宽度。** V3 §4.2 p.22：“number of Transformer layers to 61 and the hidden dimension to 7168”。
**C3 MLA 维度。** V3 §4.2：$n_h=128$，$d_h=128$，“KV compression dimension $d_{c}$ is set to 512”，$d_c'=1536$，解耦 RoPE 每头 $d_h^R=64$。
**C4 MoE 规格。** V3 §4.2：除前三层外全部 FFN 换成 MoE（“all FFNs except for the first three layers”）；
每个 MoE 层 “1 shared expert and 256 routed experts”，专家中间维度 2048；“8 experts will be activated for each token”；每 token 最多 4 个节点。
**C5 MTP 深度。** V3 §4.2：“multi-token prediction depth $D$ is set to 1”，即除下一个 token 外再多预测 1 个。
**C6 数据与上下文。** V3 §4.2：预训练最大序列 4K，“pre-train DeepSeek-V3 on 14.8T tokens”。§4.3 p.23：用 YaRN 两阶段各 1000 步，“from 4K to 32K and then to 128K”。
**C7 与 V2 的唯一架构差异（MoE 部分）。** V3 §2.1：“an exception is that we additionally introduce an auxiliary-loss-free load balancing strategy”（与 V2 相比）。
另见 C9：门控函数也从 softmax 改成 sigmoid（§2.1.2 自己称为 “Slightly different”）。
**C8 V2 的遗留设定。** V3 §4.2：沿用 V2 在压缩潜在向量后加 RMSNorm、在“width bottlenecks”处乘额外缩放因子。V2 §4.1 把 width bottleneck 解释为压缩潜在向量和“the intermediate hidden states of routed experts”。

### B. DeepSeekMoE 门控（sigmoid + 选中后归一化）

**C9 sigmoid 亲和度与归一化。** V3 §2.1.2，式 (12)–(15)（p.8）之后、p.9 页首：“DeepSeek-V3 uses the sigmoid function to compute the affinity scores”，并且 “applies a normalization among all selected affinity scores”。
**C10 V2 用 softmax。** V2 §2.2.1 式 (22)：$s_{i,t}=\operatorname{Softmax}_i(\mathbf u_t^T\mathbf e_i)$，且 V2 的 $g_{i,t}$ 直接等于 $s_{i,t}$（无选中后归一化，式 (21)）。
**C11 sigmoid 的来源与理由。** V3 本身没给理由。LossFree §4.1：“the sigmoid baseline performs better than the softmax baseline”；附录 C.1 称 softmax 门控在相近均衡程度下困惑度更高、对不均衡更敏感；C.2 称 softmax 的归一化使偏置调节更难（一个专家的分差受其他专家分数影响）。
**C12 MoE 门控保持高精度。** V3 §3.3.1：FP8 训练中保持 BF16/FP32 的组件清单包含 “MoE gating modules”。

### C. 无辅助损失负载均衡（偏置 $b_i$）

**C13 动机。** V3 §2.1.2：不均衡会导致 routing collapse 并降低专家并行效率；传统做法靠辅助损失，但 “too large an auxiliary loss will impair the model performance”（引 LossFree）。
**C14 偏置只管选、不管权重。** V3 §2.1.2 p.9，式 (16) 之后：“Note that the bias term is only used for routing.” 门控值仍由原始 $s_{i,t}$ 得出。
**C15 统计范围 = 整个 batch。** V3 §2.1.2：“keep monitoring the expert load on the whole batch of each training step”。
**C16 更新规则。** V3 §2.1.2：每步结束时，过载专家 “decrease the bias term by $\gamma$”，欠载专家增加 $\gamma$；$\gamma$ 称为 “bias update speed”。（V3 没写“过载”的阈值；LossFree 算法 1 用均值作阈值，见 C31。）
**C17 $\gamma$ 的取值（已核对）。** V3 §4.2 p.23：“0.001 for the first 14.3T tokens, and to 0.0 for the remaining 500B”。即最后 500B tokens 偏置冻结。
**C18 效果声明。** V3 §2.1.2：动态调节使训练全程保持均衡，并且比纯辅助损失的模型性能更好。

### D. 互补的序列级均衡损失

**C19 仍然有一个辅助损失。** V3 §2.1.2 p.9：“to prevent extreme imbalance within any single sequence”，另加序列级均衡损失 $\mathcal L_{\rm Bal}$（式 (17)–(20)），$\alpha$ “assigned an extremely small value”。
**C20 $\alpha$ 的取值（已核对）。** V3 §4.2 p.23：“we set $\alpha$ to 0.0001, just to avoid extreme imbalance within any single sequence”（后半句为原文，整句 15 词内）。
**C21 $T$ 的含义。** V3 式 (20) 之后：$T$ 为一个序列中的 token 数。式 (18) 的指示函数里的 Topk 集合**不带偏置**（见 §7 E13）。

### E. 节点受限路由、不丢 token

**C22 节点受限路由。** V3 §2.1.2 p.10：“each token will be sent to at most $M$ nodes”；节点按 “the sum of the highest $\frac{K_r}{M}$ affinity scores” 挑选。V3 §4.2：“(i.e., $M=4$)”，路由专家均匀部署在 “64 GPUs belonging to 8 nodes”。
**C23 带宽依据与 “13 个专家”。** V3 §3.2.2 p.13：NVLink 160 GB/s，“roughly 3.2 times that of IB (50 GB/s)”；每 token 平均每节点可选 3.2 个专家而不增加 NVLink 开销，因此可把激活数扩到 “a maximum of 13 experts (4 nodes $\times$ 3.2 experts/node)” 而通信成本不变。§3.2.2 还说 20 个 SM 足以打满 IB 与 NVLink 带宽。
**C24 不丢 token。** V3 §2.1.2 p.10：“DeepSeek-V3 does not drop any tokens during training”；推理时靠部署策略也不丢。
**C25 对照：V2 丢 token。** V2 §2.2.4：设备级 token 丢弃，容量因子 1.0，约 10% 训练序列永不丢；V2 §3.1.2 “We employ the token-dropping strategy during training”，评估时不丢。
**C26 对照：V2 设备受限路由。** V2 §2.2.2：“select $M$ devices that have experts with the highest affinity scores in them”，再在这些设备内做 top-K；V2 取 $M=3$（V2 §3.1.2）；并称 $M\geqslant 3$ 时与不受限 top-K 大致持平。
**C27 Insights 的节点受限路由叙述。** Insights §4.3：H800 节点内外带宽比 “approximately 4:1”（NVLink 名义 200 GB/s、实际约 160；IB 400Gbps 网卡 50 GB/s，按 40 GB/s 有效计）；256 个路由专家分 8 组、每组 32 个、每组放一个节点；token 的 IB 通信代价从 $8t$ 降到 $Mt$。

### F. 消融

**C28 无辅助损失 vs 纯辅助损失（表 5，p.27）。** V3 §4.5.2：小模型 15.7B 总参 / 2.4B 激活 / 1.33T tokens；大模型 228.7B / 20.9B / 578B tokens。
基线 “purely use auxiliary losses” 并使用 “sigmoid gating function with top-K affinity normalization”；辅助损失强度分别同 V2-Lite 与 V2。结论：“consistently achieves better model performance on most of the evaluation benchmarks”。完整数字见 §3.6。
**C29 batch 级 vs 序列级（§4.5.3，p.27–28）。** 关键区别是均衡范围：“batch-wise versus sequence-wise”；batch 级 “imposes a more flexible constraint”，允许专家按领域特化。
1B MoE 验证损失：序列级辅助损失 2.258，无辅助损失 2.253，batch 级辅助损失 2.253；3B MoE：2.085 vs 2.080 vs 2.080。
代价：“two potential challenges in efficiency”——(1) 个别序列或小 batch 内不均衡；(2) 推理时领域迁移导致不均衡。前者靠大规模 EP/DP 保证 micro-batch 足够大，后者靠冗余专家部署（§3.4）。
**C30 专家特化图（图 9，p.28；附录 C 图 11，p.48–53）。** 16B 辅助损失基线 vs 16B 无辅助损失模型，在 Pile 测试集三个领域（Wikipedia (en)、Github、DM Mathematics）上的相对专家负载；无辅助损失模型 “shows greater expert specialization patterns”。相对负载 = 实际负载 / 理论均衡负载。

### G. MTP（多 token 预测）

**C31 受 Gloeckle 启发，但改为顺序 + 完整因果链。** V3 §2.2 p.10：“Inspired by Gloeckle et al. (2024)”；区别在于 Gloeckle 用独立输出头并行预测，V3 “sequentially predict additional tokens and keep the complete causal chain at each prediction depth”（15 词）。
**C32 两个动机（原文用 “may”）。** V3 §2.2：MTP “densifies the training signals and may improve data efficiency”；还 “may enable the model to pre-plan its representations”。二者都是作者的假设性表述，不是被证明的结论。
**C33 模块组成。** V3 §2.2：第 $k$ 个 MTP 模块 = 共享嵌入层 $\operatorname{Emb}(\cdot)$ + 共享输出头 $\operatorname{OutHead}(\cdot)$ + 一个 Transformer 块 $\operatorname{TRM}_k(\cdot)$ + 投影矩阵 $M_k\in\R^{d\times 2d}$；$k=1$ 时 $\mathbf h_i^{k-1}$ 是主模型给出的表示。嵌入层和输出头都 “shared with the main model”。
**C34 与 EAGLE 的关系。** V3 §2.2：保持因果链的原则与 EAGLE 相似，但 EAGLE 的主要目标是投机解码，而 V3 用 MTP 改进训练。
**C35 $\lambda$ 取值（已核对）。** V3 §4.2 p.23：MTP 损失权重 “0.3 for the first 10T tokens, and to 0.1 for the remaining 4.8T tokens”。
**C36 推理时可丢弃。** V3 §2.2：“directly discard the MTP modules and the main model can function independently”；也可改作投机解码。
**C37 MTP 消融（表 4，p.26）。** 小模型 15.7B / 1.33T tokens；大模型 228.7B / **540B** tokens（注意与表 5 的 578B 不同）；在基线上加 1 层深度的 MTP 模块；推理时丢弃，“inference costs of the compared models are exactly the same”。结论：大多数基准提升。完整数字见 §3.6。
**C38 投机解码接受率（§5.4.3，p.35，已核对）。** V3 通过 MTP 预测后 2 个 token；“acceptance rate of the second token prediction ranges between 85% and 90%”（跨不同生成主题），“delivering 1.8 times TPS”。
**C39 Insights 的不同数字。** Insights §2.3.3：“acceptance rate of 80% to 90% for predicting the second subsequent token”，同样 1.8x TPS；并说这种做法 “slightly hurting the throughput”，但显著改善端到端延迟；每个 MTP 模块只用单层。
**C40 物理共享。** V3 §3.2.3：借助 DualPipe，把最浅层（含嵌入）与最深层（含输出头）放在同一个 PP rank 上，MTP 模块与主模型的嵌入与输出头在参数和梯度上物理共享。

### H. LossFree 论文（Wang et al. 2024）

**C41 核心论点：梯度干扰。** LossFree 摘要：大辅助损失 “will introduce non-negligible interference gradients into training”，从而损害性能。这是**经验论断**（图 2 的 $\alpha$ 扫描），论文没有给出定理或梯度分解。
**C42 偏置加在门控分数上，只影响 top-K。** LossFree §3 式 (3)；并说明偏置 “not added to the $g_{i,t}$”。
**C43 算法 1（偏置更新）。** 初始化 $b_i=0$；每个 batch 训练后统计每个专家分到的 token 数 $c_i$ 与平均 $\overline{c_i}$；误差 $e_i=\overline{c_i}-c_i$；更新 $b_i=b_i+u*\mathrm{sign}(e_i)$。（算法里两步都编号为 “4.”，是原文排版错误。）
**C44 为何用历史负载。** LossFree §3：用当前序列的负载信息 “will break the causal constraint of language modeling”，会泄漏未来 token 信息。
**C45 更新率的取舍（§4.3、图 4）。** $u=0.0001$ 收敛慢，$u=0.01$ 后期偏置波动、均衡变差，“An appropriate choice is $u=0.001$”。与 V3 的 $\gamma=0.001$ 一致。
**C46 更新规则变体（表 3）。** 把 sign 改成 $b_i=b_i+u*e_i$ 略改善均衡但困惑度不更好，因此保留 sign 版。乘性偏置（式 (5)，初始化为 1）略差（表 4）。
**C47 MaxVio 指标。** LossFree §4.1 式 (4)；分 $\text{MaxVio}_{\text{global}}$（整个验证集计负载）与 $\text{MaxVio}_{\text{batch}}$（每个训练 batch 计）。
**C48 主结果（表 2）。** 1B：Loss-Controlled 困惑度 9.56 / MaxVio_global 0.72；Loss-Free 9.50 / 0.04。3B：7.97 / 0.52 vs 7.92 / 0.04。基线 $\alpha=0.001$。
**C49 与专家并行的兼容性（§5.1）。** 计算 batch 越大，Loss-Free 的均衡越好；辅助损失法在大 batch 下大致停在常数水平。
**C50 Expert Choice 的未来信息泄漏（§5.2、附录 D）。** 稀疏率 $R=K/N$ 的 EC 路由每 token 可泄漏 “more than $K\log_{2}\frac{1-R}{R}$ bits”；9 层、16 专家、平均 2 个专家的模型超过 50 bits。块大小 512 时出现约 10% 的异常损失下降。
**C51 实验规模。** 1B 模型训 100B tokens，3B 模型训 200B tokens（§4.1 与附录 B）；32K BPE 词表；DeepSeekMoE 骨架，2 共享 + 64 路由、激活 6 个（附录 A 表 5）。

### I. Gloeckle et al. 2024（并行独立头 MTP）

**C52 架构。** Gloeckle 摘要：预测后续 $n$ 个 token，“using $n$ independent output heads, operating on top of a shared model trunk”；§2：头是 transformer 层 $f_{h_i}$，共享反嵌入 $f_u$。
**C53 参数匹配的比较。** Gloeckle §3：加 $n-1$ 层到预测头时，“we remove $n-1$ layers from the shared model trunk”，保证总参数相同。（DeepSeek 的 MTP 消融**不是**这样做的，见 §7 E19。）
**C54 省显存的前反向顺序。** Gloeckle §2：主干前向后，逐个头做前向+反向，把峰值显存从 $O(nV+d)$ 降到 $O(V+d)$。
**C55 规模依赖。** 图 3 标题：“Multi-token prediction models are worse than the baseline for small model sizes”，大模型才受益。13B 模型：HumanEval 多解 12%、MBPP 多解 17%（摘要；表 S7 复算见 §4）。
**C56 最优 $n$。** §3.4：7B、200B tokens 代码数据上 $n=4$ 在 HumanEval/MBPP 全面最好；APPS/Intro 上 $n=6$ 领先；字节级最优为 8 字节。
**C57 自投机解码。** §3.2：4-token 模型在代码上加速 3.0 倍，“an average of 2.5 accepted tokens out of 3 suggestions”；文本 2.7 倍；8 字节模型 6.4 倍。表 S2 给出按使用头数的加速（见 §3.7）。
**C58 自然语言上的保留。** §3.7：7B/200B 自然语言，2-token 与基线持平，4-token 在选择题类基准上略退；摘要类生成任务 $n=2$、$n=4$ 都有提升。
**C59 因果/反因果头变体已被试过。** 附录 B：因果变体中后面的头叠在前面的头之上（式见 §2.9）；表 S4 标题称替代架构 “showed no significant improvement with respect to Parallel architecture”。**因此“把头串起来”本身不是 DeepSeek 首创；DeepSeek 的新东西是每一深度额外喂入真实的第 $i+k$ 个 token 的嵌入（见 §7 E6）。**

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 MLA 回顾（V3 §2.1.1，式 (1)–(11)，p.7–8；与 V2 相同，详见第 9 课）

```latex
\boxed{\color{blue} \mathbf{c}_{t}^{KV}} &= W^{DKV} \mathbf{h}_{t}, \tag{1}\\
[\mathbf{k}_{t, 1}^{C};\mathbf{k}_{t, 2}^{C};...;\mathbf{k}_{t, n_{h}}^{C}] = \mathbf{k}_{t}^{C} &= W^{UK} \mathbf{c}_{t}^{KV}, \tag{2}\\
\boxed{\color{blue}\mathbf{k}_{t}^{R}} &= \operatorname{RoPE}({W^{KR}} \mathbf{h}_{t}), \tag{3}\\
\mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}], \tag{4}\\
[\mathbf{v}_{t, 1}^{C};\mathbf{v}_{t, 2}^{C};...;\mathbf{v}_{t, n_{h}}^{C}] = \mathbf{v}_{t}^{C} &= W^{UV} \mathbf{c}_{t}^{KV}, \tag{5}\\
\mathbf{c}_{t}^{Q} &= W^{DQ} \mathbf{h}_{t}, \tag{6}\\
[\mathbf{q}_{t, 1}^{C};\mathbf{q}_{t, 2}^{C};...;\mathbf{q}_{t, n_{h}}^{C}] = \mathbf{q}_{t}^{C} &= W^{UQ} \mathbf{c}_{t}^{Q}, \tag{7}\\
[\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} &= \operatorname{RoPE}({W^{QR}} \mathbf{c}_{t}^{Q}), \tag{8}\\
\mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}], \tag{9}\\
\mathbf{o}_{t, i} &= \sum_{j=1}^{t} \operatorname{Softmax}_j(\frac{\mathbf{q}_{t, i}^T \mathbf{k}_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}) \mathbf{v}_{j, i}^{C}, \tag{10}\\
\mathbf{u}_{t} &= W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}], \tag{11}
```

形状（原文）：$W^{DKV}\in\R^{d_c\times d}$，$W^{UK},W^{UV}\in\R^{d_hn_h\times d_c}$，$W^{KR}\in\R^{d_h^R\times d}$，
$W^{DQ}\in\R^{d_c'\times d}$，$W^{UQ}\in\R^{d_hn_h\times d_c'}$，$W^{QR}\in\R^{d_h^Rn_h\times d_c'}$，$W^O\in\R^{d\times d_hn_h}$。
只缓存蓝框的 $\mathbf c_t^{KV}$ 与 $\mathbf k_t^R$。课程记号与论文一致（style-guide §2 就是按 DeepSeek 记号定的）。

### 2.2 DeepSeekMoE（V3 式 (12)–(15)，p.8）

```latex
\mathbf{h}_{t}^{\prime} & = \mathbf{u}_{t} + \sum_{i=1}^{N_{s}} {\operatorname{FFN}^{(s)}_{i}\left( \mathbf{u}_{t} \right)} + \sum_{i=1}^{N_r} {g_{i,t} \operatorname{FFN}^{(r)}_{i}\left( \mathbf{u}_{t} \right)}, \tag{12}\\
g_{i,t} & = \frac{g^{\prime}_{i,t}}{\sum_{j=1}^{N_r} g^{\prime}_{j,t}}, \tag{13}\\
g^{\prime}_{i,t} & = \begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | 1 \leq j \leq N_r \}, K_{r}), \\
0, & \text{otherwise},
\end{cases} \tag{14}\\
s_{i,t} & = \operatorname{Sigmoid} \left( {\mathbf{u}_{t}}^{T} \mathbf{e}_{i} \right), \tag{15}
```

符号（原文）：$N_s,N_r$ 共享/路由专家数；$K_r$ 激活的路由专家数；$g_{i,t}$ 门控值；$s_{i,t}$ token–专家亲和度；$\mathbf e_i$ 第 $i$ 个路由专家的中心向量（centroid）；$\operatorname{Topk}(\cdot,K)$ 为当前 token 对全体路由专家亲和度中最高 $K$ 个组成的集合。
**课程记号**：与 style-guide 完全一致（$N_s,N_r,K_r,s_{i,t},g_{i,t},\mathbf e_i$）；`\operatorname{Topk}` 用课程宏 `\TopK`；残差 $\mathbf u_t+\cdots$ 写在 MoE 公式内部这一点要提醒读者（式 (12) 的 $\mathbf h_t'$ 已含残差）。
形状（笔者补全，原文未写）：$\mathbf u_t\in\R^{d}$，$\mathbf e_i\in\R^{d}$，$s_{i,t}\in(0,1)$ 标量。V3 取 $N_s=1,N_r=256,K_r=8$。

**V2 的对照式（V2 式 (21)–(22)）**——注意 V2 **没有**式 (13) 这一步：

```latex
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t}\in\operatorname{Topk}(\{s_{j,t}|1\leqslant j\leqslant N_{r}\},K_{r}),\\ 0,&\text{otherwise},\end{cases} \tag{V2-21}\\
s_{i,t} = \operatorname{Softmax}_{i}\left({\mathbf{u}_{t}}^{T}\mathbf{e}_{i}\right), \tag{V2-22}
```

### 2.3 无辅助损失均衡（V3 式 (16)，p.9）

```latex
g^{\prime}_{i,t} & = \begin{cases}
s_{i,t}, & s_{i,t} + b_i \in \operatorname{Topk} (\{ s_{j, t} + b_j | 1 \leq j \leq N_r \}, K_{r}), \\
0, & \text{otherwise}.
\end{cases} \tag{16}
```

式 (16) 替换式 (14)。选专家用 $s+b$，门控值仍取 $s$，再经式 (13) 归一化。
**更新规则**（V3 只有文字，笔者按文字写成式子；与 LossFree 算法 1 的 sign 规则一致）：

$$
b_i \leftarrow b_i - \gamma\,\sign\!\big(\mathrm{load}_i - \overline{\mathrm{load}}\big),\qquad \gamma=\begin{cases}0.001,&\text{前 14.3T tokens}\\0,&\text{最后 500B tokens}\end{cases}
$$

其中 $\mathrm{load}_i$ 在“整个训练 batch”上统计。$\overline{\mathrm{load}}$ 作阈值是 LossFree 的定义，V3 没明写（列入 §8）。

### 2.4 互补序列级均衡损失（V3 式 (17)–(20)，p.9）

```latex
\mathcal{L}_{\mathrm{Bal}} & = \alpha \sum_{i=1}^{N_r}{f_i P_i}, \tag{17}\\
f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathds{1} & \left( s_{i,t} \in \operatorname{Topk} ( \{ s_{j, t} | 1 \leq j \leq N_r \}, K_{r} ) \right), \tag{18}\\
s^{\prime}_{i,t} & = \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}}, \tag{19}\\
P_i & = \frac{1}{T} \sum_{t=1}^{T}{s^{\prime}_{i,t}}, \tag{20}
```

源码中式 (18) 上方有一行被注释掉的旧写法：`f_i = \frac{N_r}{K_r T} \sum \mathds{1}( \text{Expert $i$ } \text{belongs to the Top-$K_r$ set for Token $t$} )`——
即作者曾用文字“属于该 token 的 Top-$K_r$ 集合”表述，定稿改成了不带偏置的 $\operatorname{Topk}(\{s_{j,t}\})$。实际训练用的是哪一个集合，无法从公开材料确认（§8）。

要点：
- $s'_{i,t}$ 是把 sigmoid 分数在**全部** $N_r$ 个专家上归一化（不是只在选中的上），使 $P_i$ 构成一个分布，和 V2 的 softmax 版 $P_i$（V2 式 (25)：$P_i=\frac1T\sum_t s_{i,t}$）同构。
- 归一化：每个 token 恰好选 $K_r$ 个，所以 $\sum_i f_i = N_r$；完全均衡时 $f_i\equiv 1$。
- 课程记号：$\mathds{1}$ 可写 $\mathbb 1$；$T$ 与课程“序列长度”一致。

**对照 V2 的专家级均衡损失（V2 式 (23)–(25)）**：

```latex
\mathcal{L}_{\mathrm{ExpBal}} = \alpha_{1}\sum_{i=1}^{N_{r}}{f_{i}P_{i}}, \quad
f_{i} = \frac{N_{r}}{K_{r}T}\sum_{t=1}^{T}{\mathds{1}(\text{Token $t$ selects Expert $i$})}, \quad
P_{i} = \frac{1}{T}\sum_{t=1}^{T}{s_{i,t}}
```

V2 另有设备级（$\alpha_2$）和通信均衡损失（$\alpha_3$），V3 全部去掉，只保留一个 $\alpha=10^{-4}$ 的序列级损失。

### 2.5 节点受限路由（V3 只有文字；笔者形式化）

设路由专家按节点划分为 $\mathcal E_1,\dots,\mathcal E_8$（每个 32 个）。对 token $t$：

$$
\text{NodeScore}_n(t)=\sum_{\text{最高 }K_r/M\text{ 个}\ i\in\mathcal E_n} s_{i,t}\quad(K_r/M=8/4=2),\qquad
\mathcal N_t=\TopK(\{\text{NodeScore}_n(t)\},M),
$$

然后只在 $\bigcup_{n\in\mathcal N_t}\mathcal E_n$ 内做式 (16) 的 top-$K_r$。论文没说节点打分用 $s$ 还是 $s+b$；推理代码用 $s+b$（§2.9）。
V2 的设备打分是“设备内最高的一个亲和度”（V2 §2.2.2 文字；GitHub 代码中 `bias is None` 分支 `scores.amax(dim=-1)`）。

### 2.6 MTP（V3 式 (21)–(25)，p.10–11）

```latex
\mathbf{h}_i^{\prime k} = M_k [\operatorname{RMSNorm}(\mathbf{h}_i^{k-1}) ; \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))], \tag{21}\\
\mathbf{h}_{1:T-k}^{k} = \operatorname{TRM}_k(\mathbf{h}_{1:T-k}^{\prime k}), \tag{22}\\
P_{i+k+1}^{k} = \operatorname{OutHead}(\mathbf{h}_{i}^{k}). \tag{23}\\
\mathcal{L}_{\text{MTP}}^{k} = \operatorname{CrossEntropy}(P_{2 + k:T + 1}^{k}, t_{2 + k:T + 1}) = -\frac{1}{T} \sum_{i=2 + k}^{T + 1} \log P_i^k [t_i], \tag{24}\\
\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{\text{MTP}}^{k}. \tag{25}
```

符号（原文）：$t_i$ 第 $i$ 个输入 token；$\mathbf h_i^{k-1}\in\R^d$ 第 $i$ 个 token 在第 $k-1$ 深度的表示（$k=1$ 时为主模型的表示）；$M_k\in\R^{d\times 2d}$；
$_{i:j}$ 为含两端的切片；$P^k_{i+1+k}\in\R^{V}$，$V$ 为词表大小；$\operatorname{OutHead}$ 线性映射到 logits 后接 Softmax。
正文文字写的是 “$P_{i+1+k}^{k}$”，式 (23) 写 “$P_{i+k+1}^{k}$”，同一对象。

**下标自洽性检查（笔者）**：式 (22) 对 $i=1..T-k$ 计算；由式 (23)，被预测位置 $i+k+1\in[k+2,\,T+1]$，与式 (24) 的求和范围一致；
式 (21) 需要 $\operatorname{Emb}(t_{i+k})$，$i+k\le T$，恰在输入内。式 (24) 求和有 $T-k$ 项却除以 $T$（不是 $T-k$），
$T=4096,k=1$ 时系数差 $4095/4096$，无实质影响，但写课时照抄即可、不要“更正”。
总训练目标（笔者按原文文字组合）：$\mathcal L=\mathcal L_{\text{Main}}+\mathcal L_{\text{MTP}}\,(+\mathcal L_{\rm Bal})$；原文图 3 把主损失标为 $\mathcal L_{\text{Main}}$。

**课程记号冲突与建议**（写课前必须决定）：

| 论文符号 | 论文含义 | 与课程冲突 | 建议 |
|---|---|---|---|
| $t_i$ | 第 $i$ 个 token（离散 id） | 课程 $t$ 是位置下标 | 写 $x_i$（Gloeckle 也用 $x_t$），位置用 $i$ |
| $D$ | MTP 深度 | style-guide 中 $D$ = 训练 token 数 | 写 $D_{\rm MTP}$ |
| $M_k$ | MTP 投影矩阵 $\R^{d\times2d}$ | $M$ 在同一论文又是节点数；style-guide 中 $M$ = 每 token FLOPs | 写 $W^{\rm MTP}_k$ 或 $\mathbf M_k$；节点数写 $M_{\rm node}$ |
| $P_i$（式 20） vs $P_i^k$（式 23） | 平均亲和度 vs MTP 预测分布 | 同一字母 | 均衡里保留 $P_i$，MTP 分布写 $\mathbf p^{(k)}_{i}$ |
| $\mathbf h_i^{k}$ | 深度 $k$ 的表示 | 上标 $k$ 像幂 | 写 $\mathbf h_i^{(k)}$ |
| $\mathbf h_t'$（式 12）vs $\mathbf h_i'^k$（式 21） | MoE 输出 vs MTP 组合输入 | 都用撇号 | MTP 的写 $\tilde{\mathbf h}_i^{(k)}$ |
| $\alpha$ | 均衡因子（式 17） | 同论文 §4.3 YaRN 也有 $\alpha=1,\beta=32$ | 均衡处写 $\alpha_{\rm Bal}$ |
| LossFree 的 $u$ | 偏置更新率 | 与 $\mathbf u_t$（FFN 输入）冲突 | 统一用 V3 的 $\gamma$ |
| LossFree 的 $e_i$ | 负载误差 $\overline{c_i}-c_i$ | 与专家中心 $\mathbf e_i$ 冲突（同一篇论文里就冲突） | 写 $\delta_i$ |
| LossFree 的 $N,K$ | 专家数、激活数 | — | 映射为 $N_r,K_r$ |

### 2.7 LossFree 论文公式（2408.15664 v1）

```latex
% (1) MoE 层（一个编号覆盖三行）
\mathbf{h}_{t}=\mathbf{u}_{t}+\sum_{i=1}^{N}g_{i,t}\operatorname{FFN}_{i}\left(\mathbf{u}_{t}\right),\quad
g_{i,t}=\begin{cases}s_{i,t},&s_{i,t}\in\operatorname{Topk}\left(\left\{s_{j,t}\mid 1\leq j\leq N\right\},K\right),\\ 0,&\text{ otherwise, }\end{cases}\quad
s_{i,t}=G\left(\mathbf{u}_{t}{}^{T}\mathbf{e}_{i}\right) \tag{1}
% (2) 辅助损失
\mathcal{L}_{\text{Balance }}=\alpha\sum_{i=1}^{N}f_{i}P_{i},\quad
f_{i}=\frac{N}{KT}\sum_{t=1}^{T}\mathbbm{1}(\text{ Token }t\text{ selects Expert }i),\quad
P_{i}=\frac{1}{T}\sum_{t=1}^{T}s_{i,t} \tag{2}
% (3) 带偏置的选择
g_{i,t}=\begin{cases}s_{i,t},&s_{i,t}+b_{i}\in\operatorname{Topk}\left(\left\{s_{j,t}+b_{j}\mid 1\leq j\leq N\right\},K\right),\\ 0,&\text{ otherwise. }\end{cases} \tag{3}
% (4) 最大违背度
\text{MaxVio}=\frac{\max_{i}\text{Load}_{i}-\overline{\text{Load}_{i}}}{\overline{\text{Load}_{i}}} \tag{4}
% (5) 乘性偏置变体
g_{i,t}=\begin{cases}s_{i,t},&s_{i,t}*{b}_{i}\in\operatorname{Topk}\left(\left\{s_{j,t}*{b}_{j}\mid 1\leq j\leq N\right\},K\right),\\ 0,&\text{ otherwise, }\end{cases} \tag{5}
% (7) EC 路由泄漏下界（附录 D.1）
I=\left.\log_{2}\binom{\frac{KT}{N}}{T}^{N}\right/T > N\left.\log_{2}\frac{((1-\frac{K}{N})T)^{\frac{K}{N}T}}{(\frac{K}{N}T)^{\frac{K}{N}T}}\right/T = K\log_{2}\frac{1-R}{R} \tag{7}
```

算法 1（原文逐步）：1. 初始化 $b_i=0$；2. 用式 (3) 的门控在该 batch 上训练；3. 统计 $c_i$ 与 $\overline{c_i}$；4. $e_i=\overline{c_i}-c_i$；4.（原文重号）$b_i=b_i+u*\mathrm{sign}(e_i)$。
注意：LossFree 式 (3) 中 $g_{i,t}=s_{i,t}$，**没有**选中后归一化（V3 式 (13) 才有）。式 (7) 的二项式记号 $\binom{KT/N}{T}$ 按原文照录（按常规应理解为“从 $T$ 个中选 $KT/N$ 个”，原文上下位置写法与常规相反——笔者注，列入 §7 E24）。

### 2.8 Gloeckle 公式（2404.19737 v1）

```latex
L_{1}=-\sum_{t}\log P_{\theta}(x_{t+1}\mid x_{t:1}), \tag{1}\\
L_{n}=-\sum_{t}\log P_{\theta}(x_{t+n:t+1}\mid x_{t:1}). \tag{2}\\
% 无编号：共享主干 + 独立头的分解
L_{n}=-\sum_{t}\log P_{\theta}(x_{t+n:t+1}\mid z_{t:1})\cdot P_{\theta}(z_{t:1}\mid x_{t:1})
     =-\sum_{t}\sum_{i=1}^{n}\log P_{\theta}(x_{t+i}\mid z_{t:1})\cdot P_{\theta}(z_{t:1}\mid x_{t:1}).\\
% 无编号：每个头
P_{\theta}(x_{t+i}\mid x_{t:1})=\text{softmax}(f_{u}(f_{h_{i}}(f_{s}(x_{t:1})))),\quad i=1,\ldots,n\\
% 附录 B：因果变体 / 反因果变体
P_{\theta}(x_{t+i}|\cdot)=\mathrm{softmax}\circ f_{u}\circ f_{h_{i}}\circ f_{h_{i-1}}\cdots\circ f_{h_{1}}\circ f_{s}\\
P_{\theta}(x_{t+i}|\cdot)=\mathrm{softmax}\circ f_{u}\circ f_{h_{i}}\circ f_{h_{i+1}}\cdots\circ f_{h_{n}}\circ f_{s}
```

解读：Gloeckle 的第 $i$ 个头只看 $z_{t:1}$（主干在位置 $t$ 的表示），**不看** $x_{t+1},\dots,x_{t+i-1}$；各头给出的是 $x_{t+i}$ 的边缘分布，彼此条件独立（给定 $z$）。
V3 的第 $k$ 深度在位置 $i$ 额外得到真实 token $t_{i+k}$ 的嵌入，因此预测的是 $p(t_{i+k+1}\mid t_{\le i+k})$ 的一个近似——即“完整因果链”。
（分解式中“$\cdot P_\theta(z\mid x)$”这一写法在数学上并不规范——$z$ 是确定性函数——照录，不要在课上当作推导依据。）
§5.1 的“选择点隐式权重”结论：$n$-token 预测给选择点权重 $\frac{n(n+1)}{2}$，给无关紧要点权重 $n$。§5.2：$H(X)+H(Y)=H(X\mid Y)+2I(X;Y)+H(Y\mid X)$，即 2-token 预测把 $I(X;Y)$ 的权重翻倍。

### 2.9 代码里的门控（HF `MoEGate.forward`；GitHub `Gate.forward`）写成数学

对一个 token（$\mathbf u\in\R^{7168}$）。数值精度两份代码不同：HF 版把输入和门控权重都转成 float32 再算 logits；GitHub 版 `linear(x, self.weight)` 在 BF16 中算 logits 和 sigmoid（`generate.py` 设 `torch.set_default_dtype(torch.bfloat16)`；发布权重中 `mlp.gate.weight` 也是 BF16），加上 FP32 偏置之后才提升为 FP32：

$$
\begin{aligned}
\mathbf s&=\sigma(W_g\mathbf u)\in(0,1)^{256},\quad W_g\in\R^{256\times 7168}\ (\text{行即 }\mathbf e_i^\top,\ \text{无 bias 项}),\\
\tilde{\mathbf s}&=\mathbf s+\mathbf b\quad(\mathbf b=\texttt{e\_score\_correction\_bias}),\\
G_n&=\text{第 }n\text{ 组 32 个 }\tilde s\text{ 中最大两个之和}\quad(n=1..8,\ \texttt{topk(2)} \text{ 写死}),\\
\mathcal N&=\TopK(\{G_n\},4),\qquad \text{组外 }\tilde s\leftarrow-\infty,\\
\mathcal S&=\TopK(\tilde{\mathbf s},8),\qquad g_i=2.5\cdot\frac{s_i}{\sum_{j\in\mathcal S}s_j}\ (i\in\mathcal S).
\end{aligned}
$$

MoE 输出（GitHub `MoE.forward`）：$\mathbf y=\sum_{i\in\mathcal S}g_i\,\FFN^{(r)}_i(\mathbf u)+\FFN^{(s)}(\mathbf u)$，残差在 `Block` 里加。
即代码比论文多一个常数 $2.5$（`routed_scaling_factor`），论文式 (12)–(13) 中没有；笔者推测它就是 V2/V3 所说的“routed experts 中间隐状态处的额外缩放因子”——因为专家的下投影是线性的，缩放中间隐状态与缩放门控值数学等价——但论文没给数值，属推测。

### 2.10 $\mathcal L_{\rm Bal}$ 对 token 表示的梯度（笔者推导，用有限差分数值验证，最大误差 $1.6\times10^{-14}$）

把 $f_i$ 视为常数（指示函数不可导），记 $S_t=\sum_j s_{j,t}$，$z_{k,t}=\mathbf u_t^\top\mathbf e_k$：

$$
\frac{\partial\mathcal L_{\rm Bal}}{\partial s'_{i,t}}=\frac{\alpha f_i}{T},\qquad
\frac{\partial s'_{i,t}}{\partial s_{k,t}}=\frac{\delta_{ik}-s'_{i,t}}{S_t},\qquad
\frac{\partial s_{k,t}}{\partial z_{k,t}}=s_{k,t}(1-s_{k,t}),
$$

$$
\boxed{\ \frac{\partial\mathcal L_{\rm Bal}}{\partial\mathbf u_t}=\frac{\alpha}{T}\sum_{k=1}^{N_r}\frac{f_k-\sum_i f_i s'_{i,t}}{S_t}\,s_{k,t}(1-s_{k,t})\,\mathbf e_k\ }
$$

课上用途：这一项直接进入 $\mathbf u_t$，再反传进注意力层和更早的层——这就是 LossFree 所说“干扰梯度”的具体形态：
它推动 token 表示本身去迎合均衡，而不只是调整路由器。偏置 $b_i$ 不产生任何梯度。（这是笔者的推导与解读；LossFree 没有写出这个梯度。）

---

## 3. 规格与超参数表

### 3.1 架构（论文 vs 代码）

| 量 | V3 论文 | HF `config.json` | GitHub `config_671B.json` | 备注 |
|---|---|---|---|---|
| 层数 $l$ | 61（§4.2） | `num_hidden_layers` 61 | `n_layers` 61 | 一致 |
| 隐藏维 $d$ | 7168 | `hidden_size` 7168 | `dim` 7168 | 一致 |
| 头数 $n_h$ | 128 | `num_attention_heads` 128（`num_key_value_heads` 128） | `n_heads` 128 | |
| 每头维 $d_h$ | 128 | `qk_nope_head_dim` 128，`v_head_dim` 128 | 同 | |
| $d_h^R$ | 64 | `qk_rope_head_dim` 64 | 同 | |
| $d_c$ | 512 | `kv_lora_rank` 512 | 同 | |
| $d_c'$ | 1536 | `q_lora_rank` 1536 | 同 | |
| 稠密层 | 前 3 层（“except for the first three layers”） | `first_k_dense_replace` 3，`moe_layer_freq` 1 | `n_dense_layers` 3 | |
| 稠密 FFN 中间维 | **论文未给** | `intermediate_size` 18432 | `inter_dim` 18432 | $18432=9\times2048$，与“1 共享 + 8 路由”的激活宽度相同（算术事实；设计意图属推测） |
| $N_s$ / $N_r$ | 1 / 256 | `n_shared_experts` 1 / `n_routed_experts` 256 | 同 | |
| $K_r$ | 8 | `num_experts_per_tok` 8 | `n_activated_experts` 8 | |
| 专家中间维 | 2048 | `moe_intermediate_size` 2048 | `moe_inter_dim` 2048 | |
| 亲和度函数 | Sigmoid（式 15） | `scoring_func` "sigmoid" | `score_func` "sigmoid" | |
| 选中后归一化 | 有（式 13） | `norm_topk_prob` true | 代码中 sigmoid 时恒归一化 | |
| 路由缩放 | **论文未写数值** | `routed_scaling_factor` 2.5 | `route_scale` 2.5 | 见 §2.9 |
| 路由限制 | 节点受限 $M=4$，8 节点 | `n_group` 8，`topk_group` 4，`topk_method` "noaux_tc" | `n_expert_groups` 8，`n_limited_groups` 4 | 代码叫“组”，一组 = 一个节点的 32 个专家 |
| 偏置 $b_i$ | 有 | 张量 `mlp.gate.e_score_correction_bias`（层 3–61 每个 MoE 层一份） | `Gate.bias`（仅 `dim == 7168` 时创建，FP32） | 见 §7 E22 |
| MTP 深度 $D$ | 1 | `num_nextn_predict_layers` 1 | 推理代码未实现 MTP | 层号 61 |
| 词表 | “extended vocabulary of 128K tokens”（§4.1） | `vocab_size` 129280 | 同 | $129280=1010\times128$；“128K”是约数 |
| 初始化标准差 | 0.006（§4.2） | `initializer_range` 0.02 | — | 配置值只在从头初始化时有意义，加载预训练权重时无关 |
| 最大位置 | 128K（§4.3） | `max_position_embeddings` 163840 | — | $163840=4096\times40$（YaRN factor 40） |
| YaRN | $s=40,\alpha=1,\beta=32$，$\sqrt t=0.1\ln s+1$，只作用于 $\mathbf k^R_t$（§4.3） | `rope_scaling`: factor 40, beta_fast 32, beta_slow 1, mscale 1.0, mscale_all_dim 1.0, original 4096 | — | |
| 权重精度 | FP8 训练（§3.3） | `quantization_config`: fp8, e4m3, 块 128×128, dynamic 激活 | `dtype` "fp8" | 第 12 课 |
| 嵌入/输出头共享 | MTP 与主模型共享 | `tie_word_embeddings` false（主模型的输入嵌入与输出头**不**绑定） | — | 注意区分两种“共享” |
| 总参 / 激活 | 671B / 37B | README_WEIGHTS：主模型 671B，激活 **36.7B**（含嵌入 0.9B 与输出头 0.9B）；MTP 11.5B 独有、激活 2.4B；HF 总量 685B | — | 复算见 §4 |

### 3.2 训练超参数（V3 §4.2–4.3，p.22–23）

| 超参数 | 值 | 位置 |
|---|---|---|
| 偏置更新速度 $\gamma$ | 0.001（前 14.3T tokens），0.0（最后 500B tokens） | §4.2 |
| 序列级均衡因子 $\alpha$ | 0.0001 | §4.2 |
| MTP 权重 $\lambda$ | 0.3（前 10T），0.1（其余 4.8T） | §4.2 |
| MTP 深度 $D$ | 1 | §4.2 |
| 节点上限 $M$ | 4（8 节点、64 GPU） | §4.2 |
| 优化器 | AdamW，$\beta_1=0.9,\beta_2=0.95$，weight_decay 0.1 | §4.2 |
| 学习率 | 前 2K 步线性升到 $2.2\times10^{-4}$；恒定到 10T；4.3T 内余弦降到 $2.2\times10^{-5}$；最后 500B 中前 333B 保持 $2.2\times10^{-5}$、后 167B 为 $7.3\times10^{-6}$ | §4.2 |
| 梯度裁剪 | 1.0 | §4.2 |
| batch size | 前 469B tokens 内从 3072 逐渐增到 15360，此后 15360（单位：序列） | §4.2 |
| 预训练序列长 | 4K | §4.2 |
| 预训练 tokens | 14.8T | §4.2 |
| 上下文扩展 | 两阶段各 1000 步：32K（batch 1920）→ 128K（batch 480），lr $7.3\times10^{-6}$ | §4.3 |
| 并行 | 16 路 PP、跨 8 节点的 64 路 EP、ZeRO-1 DP，不用 TP | §3.2 |

### 3.3 V2 vs V3 的 MoE 设定对照

| 项 | V2（论文 / config） | V3（论文 / config） |
|---|---|---|
| 路由专家 / 共享专家 / 激活 | 160 / 2 / 6（V2 §3.1.2；config 同） | 256 / 1 / 8 |
| 专家中间维 | 1536 | 2048 |
| 稠密层 | 第 1 层（config `first_k_dense_replace` 1） | 前 3 层 |
| 稠密 FFN 中间维 | 12288 $=8\times1536$ | 18432 $=9\times2048$ |
| 亲和度 | Softmax（式 V2-22） | Sigmoid（式 15） |
| 门控归一化 | 无（config `norm_topk_prob` false） | 选中后归一化（`norm_topk_prob` true） |
| 路由缩放（config） | `routed_scaling_factor` 16.0 | 2.5 |
| 受限路由 | 设备受限 $M=3$，按设备内最高亲和度选设备（config `topk_method` "group_limited_greedy"，`n_group` 8，`topk_group` 3） | 节点受限 $M=4$，按节点内最高 2 个之和选节点（"noaux_tc"，8，4） |
| 均衡手段 | 三个辅助损失：$\alpha_1=0.003,\alpha_2=0.05,\alpha_3=0.02$（V2 §3.1.2）；V2-Lite 只用 $\alpha_1=0.001$（V2 附录 B.1） | 偏置 $b_i$（$\gamma=0.001$）+ 序列级 $\alpha=0.0001$ |
| token 丢弃 | 训练时设备级丢弃（容量因子 1.0），评估不丢 | 训练、推理都不丢 |
| config 中的均衡字段 | `aux_loss_alpha` 0.001，`seq_aux` true | 无此类字段 |

### 3.4 LossFree 实验模型（附录 A 表 5、附录 B）

| | 1B | 3B |
|---|---|---|
| 词表 | 32064 | 32064 |
| 隐藏维 | 1024 | 1280 |
| 注意力头 | 8 | 10 |
| MoE 层数 | 9 | 11 |
| 粒度 $d_{ff}/d_{expert}$ | 16/3 | 4 |
| 共享 / 路由 / 激活 | 2 / 64 / 6 | 2 / 64 / 6 |
| 训练 tokens | 100B（batch 1152 × 40000 步 × 2048） | 200B（batch 1728 × 56514 步 × 2048） |
| 学习率 | 余弦：1e-3 → 1e-4，warmup 1000 | 多步：[7.8e-4, 2.47e-4, 7.8e-5]，阶段步 [45211, 50862, 56514]，warmup 2000 |
| 初始化 std / 序列长 | 0.006 / 2048 | 同 |
| 门控 | sigmoid（主实验） | sigmoid |
| 基线 $\alpha$ | 0.001 | 0.001 |
| 偏置更新率 $u$ | 0.001（只在 1B 上调参，3B 直接继承） | 0.001 |

### 3.5 版本漂移：后续模型的路由字段（HF config，2026-09-24 抓取）

| 模型 | $N_r$ | $K_r$ | `n_group`/`topk_group` | `scoring_func` | `routed_scaling_factor` | `topk_method` | MTP 层 |
|---|---|---|---|---|---|---|---|
| V3 / V3-Base / V3-0324 / R1 | 256 | 8 | 8 / 4 | sigmoid | 2.5 | noaux_tc | 1 |
| V3.1 / V3.2-Exp / V3.2 | 256 | 8 | 8 / 4 | sigmoid | 2.5 | noaux_tc | 1 |
| V4-Flash | 256 | 6 | 无此字段 | **sqrtsoftplus** | 1.5 | noaux_tc | 1（另有 `num_hash_layers` 3） |
| V4-Pro | 384 | 6 | 无此字段 | **sqrtsoftplus** | 2.5 | noaux_tc | 1（`num_hash_layers` 3） |

结论：无辅助损失均衡（`noaux_tc`）与 1 层 MTP 从 V3 一直沿用到 V4；V4 改了打分函数、去掉了组限制字段、加了哈希层——这些属第 18 课，须在第 18 课的研究笔记中核对机制，此处只记录字段。

### 3.6 消融表原始数字

**V3 表 5（无辅助损失 vs 纯辅助损失，p.27）**

| 基准（指标） | shots | 小·Aux-Based | 小·Aux-Free | 大·Aux-Based | 大·Aux-Free |
|---|---|---|---|---|---|
| 激活参数 | – | 2.4B | 2.4B | 20.9B | 20.9B |
| 总参数 | – | 15.7B | 15.7B | 228.7B | 228.7B |
| 训练 tokens | – | 1.33T | 1.33T | 578B | 578B |
| Pile-test (BPB) ↓ | – | 0.727 | 0.724 | 0.656 | 0.652 |
| BBH (EM) | 3 | 37.3 | 39.3 | 66.7 | 67.9 |
| MMLU (EM) | 5 | 51.0 | 51.8 | 68.3 | 67.2 |
| DROP (F1) | 1 | 38.1 | 39.0 | 67.1 | 67.1 |
| TriviaQA (EM) | 5 | 58.3 | 58.5 | 66.7 | 67.7 |
| NaturalQuestions (EM) | 5 | 23.2 | 23.4 | 27.1 | 28.1 |
| HumanEval (Pass@1) | 0 | 22.0 | 22.6 | 40.2 | 46.3 |
| MBPP (Pass@1) | 3 | 36.6 | 35.8 | 59.2 | 61.2 |
| GSM8K (EM) | 8 | 27.1 | 29.6 | 70.7 | 74.5 |
| MATH (EM) | 4 | 10.9 | 11.1 | 37.2 | 39.6 |

**V3 表 4（MTP 消融，p.26）**

| 基准（指标） | shots | 小·Baseline | 小·w/ MTP | 大·Baseline | 大·w/ MTP |
|---|---|---|---|---|---|
| 激活参数（推理） | – | 2.4B | 2.4B | 20.9B | 20.9B |
| 总参数（推理） | – | 15.7B | 15.7B | 228.7B | 228.7B |
| 训练 tokens | – | 1.33T | 1.33T | 540B | 540B |
| Pile-test (BPB) ↓ | – | 0.729 | 0.729 | 0.658 | 0.657 |
| BBH (EM) | 3 | 39.0 | 41.4 | 70.0 | 70.7 |
| MMLU (EM) | 5 | 50.0 | 53.3 | 67.5 | 66.6 |
| DROP (F1) | 1 | 39.2 | 41.3 | 68.5 | 70.6 |
| TriviaQA (EM) | 5 | 56.9 | 57.7 | 67.0 | 67.3 |
| NaturalQuestions (EM) | 5 | 22.7 | 22.3 | 27.2 | 28.5 |
| HumanEval (Pass@1) | 0 | 20.7 | 26.8 | 44.5 | 53.7 |
| MBPP (Pass@1) | 3 | 35.8 | 36.8 | 61.6 | 62.2 |
| GSM8K (EM) | 8 | 25.4 | 31.4 | 72.3 | 74.0 |
| MATH (EM) | 4 | 10.7 | 12.6 | 38.6 | 39.8 |

注意：表 4 与表 5 的“小/大 MoE 基线”不是同一批运行（大模型 token 数 540B vs 578B；小模型 BBH 39.0 vs 37.3），不能跨表相加。

### 3.7 Gloeckle 的相关表

**表 S7（按规模，MBPP / HumanEval pass@1，$n=1$ vs $n=4$）**：0.3B 1.8→1.0 / 1.9→1.2；0.6B 4.7→3.0 / 2.9→2.7；1.3B 6.8→7.4 / 4.6→4.8；3B 11.1→12.7 / 7.2→7.2；6.7B 23.9→26.0 / 12.8→13.8；13B 26.0→30.5 / 14.1→15.8。（13B 的 $n=2$：30.5 / 15.2。）
**表 1（7B，200B tokens 代码，32k 词表，pass@1 MBPP / HumanEval / APPS-Intro）**：$n=1$ 30.0 / 22.8 / 2.8；$n=2$ 30.3 / 22.2 / 2.1；$n=4$ 33.8 / 24.0 / 1.6；$n=6$ 31.9 / 20.6 / 3.5；$n=8$ 30.7 / 20.0 / 3.5。
**表 S2（自投机解码，7B，按使用头数；相对加速 / 每次前向 token 数）**：Wikipedia 2 头 1.79/1.88，3 头 2.35/2.57，4 头 2.74/3.12；Books 1.77/1.87，2.32/2.56，2.67/3.09；Code 1.85/1.94，2.54/2.78，3.05/3.50。
**表 S4（替代架构，$n=4$，7B/200B，MBPP@1 / HumanEval@1）**：基线 $n=1$ 30.0 / 22.8；linear 并行 33.6 / 21.9；transformer 反因果 30.8 / 20.9；因果 31.9 / 20.9；并行 33.8 / 24.0；并行 +3 层 33.3 / 22.4。

### 3.8 部署相关（V3 §3.4，第 12 课详讲，此处只记与均衡有关的）

- 预填充最小单元 4 节点 32 GPU，MoE 用 EP32；**冗余专家**：高负载专家复制部署，按线上统计周期调整（如每 10 分钟）；预填充设 32 个冗余专家，每 GPU 原 8 个专家外再放 1 个冗余专家；正在探索“每 GPU 放 16 个、每步激活 9 个”的动态冗余。
- 解码：把共享专家当作路由专家，“each token will select 9 experts”（共享专家视为必选的高负载专家）；最小单元 40 节点 320 GPU，EP320，每 GPU 一个专家，64 个 GPU 放冗余专家与共享专家。

---

## 4. 复算的数字

所有值用 Python 算出（脚本在 scratchpad，未入库）。

### 4.1 参数量（按 HF `config.json` 逐项累加；SwiGLU 专家 = 3 个矩阵）

| 量 | 表达式 | 值 |
|---|---|---|
| 每层 MLA 参数 | `7168*1536 + 1536 + 1536*128*(128+64) + 7168*(512+64) + 512 + 512*128*(128+128) + 128*128*7168` | 187,107,328 ≈ 187.1M |
| 单个专家 | `3*7168*2048` | 44,040,192 ≈ 44.04M |
| 稠密 FFN（每层） | `3*7168*18432` | 396,361,728 ≈ 396.4M |
| 门控（每 MoE 层） | `256*7168 + 256` | 1,835,264 |
| 嵌入 = 输出头 | `129280*7168` | 926,679,040 ≈ 0.927B |
| 注意力合计（61 层） | `61*187107328` | 11.41B |
| 稠密 FFN 合计（3 层） | `3*396361728` | 1.19B |
| 共享专家合计（58 层） | `58*44040192` | 2.55B |
| 路由专家（激活，58×8） | `58*8*44040192` | 20.43B |
| 路由专家（全部，58×256） | `58*256*44040192` | 653.91B（占总参 97.4%） |
| **主模型总参** | 上述之和 + 门控 + 归一化 | **671.03B**（论文 671B ✓） |
| **主模型激活参数** | 含嵌入与输出头 | **37.55B**；不含嵌入 36.63B；二者都不含 35.70B |
| README_WEIGHTS 口径 | “36.7B（含 0.9B 嵌入与 0.9B 输出头）” | 与笔者 37.55B 差 0.85B，未能解释（§8） |
| MTP 模块独有参数 | `2*7168 + 2*7168*7168 + 187107328 + 2*7168 + (257*44040192 + 1835264) + 7168` | 11.61B（README 11.5B ✓） |
| 其中 `eh_proj`（$M_1$） | `7168*14336` | 102,760,448 ≈ 102.8M |
| MTP 激活（含共享嵌入与输出头） | | 2.54B（README 2.4B，同样偏高约 0.14B） |
| HF 仓库总量 | `671.03 + 11.61 + 2*0.927` | 684.5B ≈ **685B**（README ✓）——因为 `model.layers.61.embed_tokens` 与 `model.layers.61.shared_head.head` 在 safetensors 中各存了一份（HF-index 核对） |
| MTP 占总参 | `11.61/671.03` | 1.7% |
| 每 token 前向 FLOPs（主模型，不计注意力分数、不计嵌入查表） | `2*(37.55e9 - 0.927e9)` | ≈ 73.3 GFLOPs |
| MTP 额外前向 FLOPs | `2*(0.688e9 + 0.927e9)` | ≈ 3.2 GFLOPs，约为主模型的 4.4%（笔者估算，训练时 MTP 的额外开销量级） |

### 4.2 KV cache（MLA 回顾）

| 量 | 表达式 | 值 |
|---|---|---|
| 每 token 缓存元素 | `(512+64)*61` | 35,136 |
| BF16 字节 / token | `35136*2` | 70,272 B = 68.625 KiB |
| 等价 GQA 组数 | `(512+64)/(2*128)` | 2.25（与 V2 相同，因 $d_c=4d_h$、$d_h^R=d_h/2$ 未变） |
| 同层数 MHA（$2n_hd_hl$） | `2*128*128*61*2` 字节 | 3,997,696 B = 3.8125 MiB；比 MLA 大 **56.9 倍** |
| 4K / 32K / 128K 上下文（单序列，BF16） | `70272*L/2**30` | 0.268 / 2.14 / 8.58 GiB |

### 4.3 路由与负载均衡

| 量 | 表达式 | 值 |
|---|---|---|
| 每节点专家 / 每 GPU 专家 | `256/8`，`256/64` | 32，4 |
| 节点打分取几个 | `K_r/M = 8/4` | 2（代码 `topk(2)` 写死） |
| NVLink/IB 带宽比 | `160/50` | 3.2 |
| 同通信成本下的专家上限 | `4*3.2` | 12.8（论文写 “13”，是四舍五入） |
| 不受限 8-of-256 组合数 | `comb(256,8)` | 409,663,695,276,000 ≈ $4.10\times10^{14}$ |
| 至多跨 4 个节点的组合数 | 容斥：$\sum_{j=1}^{4}\binom 8j\sum_{i}(-1)^i\binom ji\binom{32(j-i)}{8}$ | ≈ $7.16\times10^{13}$，占 **17.48%** |
| 随机 8 个专家覆盖节点数的分布 | 同上逐项 | 1 个 0.00002%，2 个 0.03%，3 个 1.63%，4 个 15.82%，5 个 41.70%，6 个 33.27%，7 个 7.28%，8 个 0.27% |
| 随机 8 个专家的期望节点数 | `8*(1-comb(224,8)/comb(256,8))` | 5.29 |
| V2 组合数（对照） | `comb(160,6)` | $2.12\times10^{10}$ |
| $\mathcal L_{\rm Bal}$ 完全均衡时 | $\alpha\cdot N_r\cdot1\cdot\frac1{N_r}$ | $\alpha=10^{-4}$ |
| $\mathcal L_{\rm Bal}$ 坍缩到固定 8 个专家（$P$ 在其上均匀） | $\alpha\cdot 8\cdot\frac{N_r}{K_r}\cdot\frac18=\alpha N_r/K_r$ | $32\alpha=3.2\times10^{-3}$ |
| $\sum_i f_i$ | 恒等式 | $=N_r=256$ |
| 满 batch 每步 tokens | `15360*4096` | 62,914,560 ≈ 62.9M |
| 每步每专家期望分配数 | `62914560*8/256` | 1,966,080 |
| 每步每 GPU（4 专家）期望分配数 | `×4` | 7,864,320 |
| 满 batch 阶段的步数 | `(14.8e12-469e9)/62914560` | ≈ 227,785 |
| batch 爬坡阶段步数（假设按步线性增长，取平均 batch 9216；**假设**） | `469e9/(9216*4096)` | ≈ 12,424 |
| 总步数（估） | | ≈ 240,000 |
| 最后 500B（$\gamma=0$）步数 | `500e9/62914560` | ≈ 7,947 |
| 100 步对应 tokens | `100*62914560` | 6.29B：偏置每 6.3B tokens 至多移动 0.1 |
| BF16 间隔 vs $\gamma$ | $2^{e-7}$ | $[0.125,0.25)$：0.00098；$[0.25,0.5)$：0.00195；$[0.5,1)$：0.0039；$[1,2)$：0.0078 → $|b|\ge0.5$ 时 BF16 存不下 0.001 的增量（被舍入掉），这是偏置要用 FP32 的数值理由（GitHub 代码 2025-08-26 改为 FP32）。实际发布的偏置在 $[2.03,8.05]$，对应 BF16 间隔 0.0156–0.0625，是 $\gamma$ 的 16–62 倍；发布 checkpoint 中偏置以 F32 存储（safetensors 头部核对） |

### 4.4 MTP

| 量 | 表达式 | 值 |
|---|---|---|
| 接受率 $p$ 下每步期望产出 token 数（1 个草稿 token） | $1+p$ | $p=0.80/0.85/0.90$ → 1.80 / 1.85 / 1.90 |
| 实测 1.8× 与理论上限之比 | `1.8/(1+p)` | 1.000 / 0.973 / 0.947（差距来自 MTP 模块与验证的额外开销；笔者解释） |
| 式 (24) 项数与系数 | $T-k$ 项、系数 $1/T$ | $T=4096,k=1$：4095/4096 = 0.99976 |
| MTP 梯度相对权重 | $\lambda/D$ | 0.3（前 10T）→ 0.1 |
| Gloeckle 13B 相对提升 | `(30.5-26.0)/26.0`，`(15.8-14.1)/14.1` | MBPP +17.3%，HumanEval +12.1%（摘要 17%、12% ✓） |
| Gloeckle 图 1 标注 | $n=4$ 相对 $n=1$ 的 MBPP@1 差 | 0.6B：−1.7；13B：+4.5（与表 S7 一致） |
| Gloeckle 代码 2 头每次前向 token 数 | 表 S2 | 1.94（≈ 第 2 头 94% 被接受；笔者换算） |

### 4.5 消融与 LossFree 的派生数字

| 量 | 表达式 | 值 |
|---|---|---|
| 表 5 差值（Free − Based），小 / 大 | 逐行相减 | BPB −0.003 / −0.004；BBH +2.0 / +1.2；MMLU +0.8 / **−1.1**；DROP +0.9 / 0.0；TriviaQA +0.2 / +1.0；NQ +0.2 / +1.0；HumanEval +0.6 / **+6.1**；MBPP **−0.8** / +2.0；GSM8K +2.5 / +3.8；MATH +0.2 / +2.4 |
| 表 5 计数 | 10 项中变好的 | 小：9 好 1 差；大：8 好 1 平（DROP）1 差（MMLU） |
| 表 4 差值（MTP − Base），小 / 大 | 逐行相减 | BPB 0.000 / −0.001；BBH +2.4 / +0.7；MMLU +3.3 / **−0.9**；DROP +2.1 / +2.1；TriviaQA +0.8 / +0.3；NQ **−0.4** / +1.3；HumanEval **+6.1** / **+9.2**；MBPP +1.0 / +0.6；GSM8K +6.0 / +1.7；MATH +1.9 / +1.2 |
| 表 4 计数 | | 小：8 好 1 平（BPB）1 差（NQ）；大：9 好 1 差（MMLU） |
| §4.5.3 验证损失 → 困惑度 | `exp(2.258), exp(2.253)` | 9.564 vs 9.516，比值 1.0050；3B：8.045 vs 8.004，同为 1.0050 |
| LossFree 1B 训练 tokens | `1152*2048*40000` | 94.37B（论文称 100B） |
| LossFree 3B 训练 tokens | `1728*2048*56514` | 200.0B（结论段误写 300B，见 §7） |
| LossFree 验证集 | `30*1152*2048`，`20*1728*2048` | 70.78M（论文 “71M” ✓） |
| LossFree 表 2 困惑度比 | `9.56/9.50`，`7.97/7.92` | 1.0063，1.0063 |
| MaxVio 与“最慢专家”的关系 | $1+\text{MaxVio}$ | 0.72 → 最忙专家负载为均值的 1.72 倍；0.04 → 1.04 倍（专家并行时一步的时间由最忙的设备决定——笔者解释） |
| EC 泄漏 | `2*log2(7)`，`9*2*log2(7)` | 每层 5.61 bits，9 层 50.5 bits（论文 “more than 50 bits” ✓） |
| LossFree 图 1 玩具例 | 负载 [5,2,1,4]，均值 3，$u=0.1$（$u=0.1$ 和 Top-2 都是从图中数字反推的：负载和 12 = 6 token × 2，偏置每格变 0.1；图上没有标这两个值） | $e=[-2,1,2,-1]$，新偏置 $[0.1,-0.2,-0.1,-0.1]+0.1\,\sign(e)=[0.0,-0.1,0.0,-0.2]$（与图 ✓）；MaxVio $=(5-3)/3=0.667$ |

### 4.6 预训练 token 账（核对 §4.2 的分段是否自洽）

- 学习率分段：$10\text{T}+4.3\text{T}+0.5\text{T}=14.8\text{T}$ ✓；$333\text{B}+167\text{B}=500\text{B}$ ✓。
- $\gamma$ 分段：$14.3\text{T}+0.5\text{T}=14.8\text{T}$ ✓，且 $\gamma$ 冻结的 500B 恰与最后两个恒定学习率段重合。
- $\lambda$ 分段：$10\text{T}+4.8\text{T}=14.8\text{T}$ ✓，$\lambda$ 下调点与余弦衰减起点（10T）重合。
- 上下文扩展每阶段 tokens：`32768*1920*1000` = `131072*480*1000` = 62.9B（两阶段相同，笔者复算）。

---

## 5. 图表清单与重绘建议

### 5.1 论文图清单

| 图 | 来源 / 页 | 标题要点 | 显示什么 | 本课是否重绘 |
|---|---|---|---|---|
| V3 图 2 | p.7 | Illustration of the basic architecture | 左：Transformer Block ×L（RMSNorm→Attention→RMSNorm→FFN，两条残差）；右上：DeepSeekMoE（共享专家 1..$N_s$ 绿色，路由专家 1..$N_r$ 浅蓝，Router + Top-$K_r$ 柱状图，门控乘法 ⊗）；右下：MLA（斜线填充表示推理时缓存的 $\mathbf c_t^{KV}$ 与 $\mathbf k_t^R$） | 是（图 11.1） |
| V3 图 3 | p.10 | Illustration of our MTP implementation | 主模型 + MTP Module 1（“Next2 Token Prediction”）+ MTP Module 2（“Next3”）；每模块：Embedding Layer（Shared）→ RMSNorm ×2 → concatenation → Linear Projection → Transformer Block → Output Head（Shared）→ Cross-Entropy；损失 $\mathcal L_{\rm Main},\mathcal L^1_{\rm MTP},\mathcal L^2_{\rm MTP}$。输入/目标：主模型输入 $t_1..t_4$、目标 $t_2..t_5$；模块 1 嵌入 $t_2..t_5$、目标 $t_3..t_6$；模块 2 嵌入 $t_3..t_6$、目标 $t_4..t_7$（文字层提取）。注意图画了 $D=2$，而 V3 实际用 $D=1$ | 是（图 11.9） |
| V3 图 9 | p.28 | Expert load … on three domains in the Pile test set | 4 条热图：Aux-Loss-Based / Aux-Loss-Free × 第 9 层 / 第 18 层；横轴 64 个专家，纵轴 3 个领域；色标 0–10 的相对负载 | 示意重绘（图 11.8） |
| V3 图 11 | 附录 C，p.48–53 | 同上，全部层 | 各层同样的热图；PNG 面板实际覆盖第 1–26 层（文件名 1-6、7-12、13-18、19-24、25-26） | 否 |
| V3 图 4/5/6/7/10 | §3 | DualPipe、FP8 | — | 第 12 课 |
| LossFree 图 1 | §1 | 训练中加偏置做 Top-K，训练后按负载反馈更新偏置 | 玩具：6 token × 4 专家门控分数；偏置 [0.1,−0.2,−0.1,−0.1]；负载 [5,2,1,4]；均值线 3；新偏置 [−0.0,−0.1,−0.0,−0.2] | 是（图 11.4） |
| LossFree 图 2 | §2.2 | 均衡与性能的两难 | 1B 模型，横轴 MaxVio_global，纵轴困惑度；$\alpha$=0.01、0.001、0.0001、0 四点连成 U 形虚线，Loss-Free 为左下角星号 | 是（图 11.5） |
| LossFree 图 3/4/5 | §4–5 | MaxVio_batch 训练曲线；更新率影响；计算 batch 大小影响 | 曲线（未读取具体数值） | 图 4 用笔者模拟替代（图 11.6） |
| LossFree 图 6 | §5.2 | EC 的未来 token 泄漏示例 | — | 可选旁注 |
| Gloeckle 图 1 | §1 | 上：共享主干 + 4 个头，输入 1–4，4-token 目标（2345、3456、4567、5678）；下：MBPP pass@1 随规模的增益条形图（0.3B 1.8、0.6B 4.7、1.3B 6.8、3B 11.1、6.7B 23.9、13B 26.0 为基线；刻度 −1.7 与 +4.5） | 是（图 11.10 的左半） |
| Gloeckle 图 2 / S11 | §2 / 附录 B | 前反向顺序（省显存） | — | 否 |

### 5.2 本课建议绘制的图（13 张；数据均已给全）

**图 11.1 DeepSeek-V3 总体架构（据 V3 图 2 重绘并加数字）。**
竖向 61 层：第 0–2 层标“稠密 FFN（中间维 18432）”，第 3–60 层标“MoE（1 共享 + 256 路由，激活 8，专家中间维 2048）”，顶部接“MTP 模块（层号 61，训练用、推理可丢）”。
注意力块标 MLA：$d=7168$，$n_h=128$，$d_h=128$，$d_c=512$，$d_c'=1536$，$d_h^R=64$，缓存 $(512+64)$ 个数/层。
颜色：共享专家 `--fig-violet`，激活的路由专家 `--fig-blue`，未激活 `--fig-muted`，缓存 `--fig-amber`，$\mathbf c^{KV}$ `--fig-orange`。

**图 11.2 门控流水线：V2 vs V3 两行对照（数据流图）。**
V2 行：$\mathbf u_t$ → $W_g\mathbf u_t$（160 个 logit）→ softmax → 按“设备内最大值”选 3 个设备 → top-6 → $g=s$（不归一化）→ ×16（config）。
V3 行：$\mathbf u_t$ → 256 个 logit → sigmoid → $+\mathbf b$（只进入选择支路，用 `--fig-red` 虚线表示“不进入权重”）→ 每节点取最大 2 个之和 → 选 4 个节点 → top-8 → 取**未加偏置**的 $s$ → 除以 8 个之和 → ×2.5（config）。

**图 11.3 一个完整路由算例（柱状图，16 个专家 = 4 节点 × 4）。** 数据（笔者构造，见 §6 练习 4）：
logits = [1.2, −0.3, 0.4, −1.0 | 0.9, 0.8, −0.5, −0.2 | −0.7, 0.1, 1.5, −1.2 | 0.0, −0.4, 0.3, 0.6]；
$s=\sigma(\cdot)$ = [0.7685, 0.4256, 0.5987, 0.2689 | 0.7109, 0.6900, 0.3775, 0.4502 | 0.3318, 0.5250, 0.8176, 0.2315 | 0.5000, 0.4013, 0.5744, 0.6457]；
$\mathbf b$ = [−0.05, 0, 0.02, 0 | 0.03, 0, 0, 0 | 0, 0, −0.10, 0 | 0.04, 0, 0, 0.06]；
$s+b$ = [0.7185, 0.4256, 0.6187, 0.2689 | 0.7409, 0.6900, 0.3775, 0.4502 | 0.3318, 0.5250, 0.7176, 0.2315 | 0.5400, 0.4013, 0.5744, 0.7057]；
节点分（最大 2 个之和）= [1.3372, 1.4309, 1.2426, 1.2801] → 选节点 {0, 1}；$K_r=4$、$M=2$；选中专家 {0, 2, 4, 5}；
门控 $g$ = [0.2776, 0.2163, 0.2568, 0.2493]（和为 1），×2.5 = [0.6941, 0.5407, 0.6421, 0.6231]。
对照：不加偏置、不限节点的 top-4 = {0, 4, 5, 10}（跨 3 个节点）；加偏置不限节点 = {0, 4, 10, 15}（跨 4 个节点）。
看点：亲和度最高的专家 10（0.8176）被排除，**完全是节点限制造成的**：只加偏置、不限节点时它仍入选（{0, 4, 10, 15}）；不加偏置、只限节点时它照样落选（节点 2 的分 1.3426 仍低于节点 0、1）。偏置 −0.10 只是把节点 2 的分进一步压低，并不是它落选的原因（见练习 4 的追问）。

**图 11.4 偏置反馈回路（据 LossFree 图 1 重绘）。** 左框“训练”：6×4 门控分数矩阵
[[0.5,0.3,0.2,0.1],[0.4,0.3,0.1,0.1],[0.1,0.4,0.1,0.4],[0.3,0.2,0.2,0.3],[0.2,0.1,0.2,0.5],[0.4,0.2,0.2,0.2]]，偏置 [0.1,−0.2,−0.1,−0.1]，Top-2 选择 → 负载 [5,2,1,4]；
右框“偏置更新”：均值线 3，箭头 ↓↑↑↓，新偏置 [0.0,−0.1,0.0,−0.2]（$u=0.1$）。
**警告**：该矩阵加偏置后有并列（例如 token 0 的专家 1、2 都是 0.1），负载 [5,2,1,4] 依赖并列的打破方式；课上若要让读者手算，请改用图 11.3 那种无并列的数据，或在图注说明“原图为示意”。

**图 11.5 均衡–性能两难（据 LossFree 图 2 重绘）。** 散点（横 MaxVio_global，纵困惑度）：
$\alpha=0.01$：(≈0.10, ≈9.69)；$\alpha=0.001$：(0.72, 9.56)（表 2 精确值）；$\alpha=0.0001$：(≈2.22, ≈9.527)；$\alpha=0$：(≈5.07, ≈9.54)；Loss-Free：(0.04, 9.50)（表 2 精确值）。
带“≈”的是从图上读出的估计值，图注须写“数值读自原图，约数”。纵轴 9.49–9.70。

**图 11.6 偏置更新速度的影响（笔者模拟，非论文数据）。** 设定：16 专家、top-2、$d=32$、专家中心 $\mathbf e_i\sim\mathcal N(0,I/d)$、固定热度偏移 $c_i$ 从 −1 到 +1 线性、每步 4096 个随机 token、$s=\sigma(\mathbf u^\top\mathbf e_i+c_i)$、sign 规则、seed 0。
MaxVio_batch 在第 1/10/30/100/300/1000/3000 步：
$\gamma=10^{-4}$：1.727, 1.668, 1.701, 1.576, 1.410, 0.615, 0.051；
$\gamma=10^{-3}$：1.727, 1.557, 1.453, 0.670, 0.129, 0.098, 0.055；
$\gamma=10^{-2}$：1.727, 0.615, 0.113, 0.121, 0.152, 0.102, 0.100。
看点：小 $\gamma$ 收敛慢；大 $\gamma$ 快但稳态更抖（0.10 vs 0.055），与 LossFree §4.3 的定性结论一致。另一个看点：$\gamma=10^{-2}$ 时最终偏置全部为正（0.21–0.61）——sign 规则不守恒 $\sum_i b_i$，但给所有 $b_i$ 加同一常数不改变 top-k 排序，只有差值 $b_i-b_j$ 有意义。
（写课时可在 Python 里用上述设定重新生成完整曲线，横轴对数。）

**图 11.7 节点受限路由与 IB/NVLink 两级转发（示意）。** 8 个节点 × 8 GPU；每 GPU 4 个专家（256/64）；一个 token 选中节点 {1, 3, 4, 6}（示例），IB 每节点只发一次（带宽 50 GB/s），到达后经 NVLink（160 GB/s）转发到目标 GPU；旁注“IB 代价 ∝ 节点数：不受限最多 $8t$，受限 $\le 4t$”（Insights §4.3）。
可附小柱状图：随机选 8 个专家覆盖节点数的分布（§4.3 数字：1–8 个节点依次 0.00002%、0.03%、1.63%、15.82%、41.70%、33.27%、7.28%、0.27%；期望 5.29）；标出 $\le4$ 的部分合计 17.48%。

**图 11.8 专家特化热图（示意，据 V3 图 9 的形式）。** 两行子图：“辅助损失（序列级）”与“无辅助损失（batch 级）”；横轴 64 个专家，纵轴 Wikipedia (en) / Github / DM Mathematics；色标 0–10（相对负载 = 实际/均衡）。
论文原图只能读出定性模式（例如第 9 层无辅助损失模型中 DM Mathematics 有一个专家接近色标上限 10、若干专家 6–8；辅助损失模型大多在 0–3），**不要编造具体数值**；用程序生成的示意数据时必须在图注写“示意”。
配合一张小柱状图：§4.5.3 验证损失 1B：2.258 / 2.253 / 2.253；3B：2.085 / 2.080 / 2.080（序列级辅助 / 无辅助 / batch 级辅助）。

**图 11.9 MTP 模块结构与 token 对齐（据 V3 图 3 重绘，改为 $D=1$ 并加对齐表）。**
取 $T=6$：主模型输入 $t_1..t_6$，在位置 $i$ 输出 $\mathbf h_i^{0}$，预测 $t_{i+1}$（目标 $t_2..t_7$）；
MTP 深度 1：位置 $i=1..5$ 输入 $[\RMSNorm(\mathbf h_i^0);\RMSNorm(\operatorname{Emb}(t_{i+1}))]$ → $M_1$（$7168\times14336$）→ $\operatorname{TRM}_1$ → 共享输出头，预测 $t_{i+2}$（目标 $t_3..t_7$，共 $T-k=5$ 项，式 (24) 求和 $i=3..7$）。
共享的嵌入层和输出头用同一颜色并画虚线连回主模型；代码里拼接顺序是 [嵌入; 隐状态]（见 §7 E14），图注说明。

**图 11.10 两种 MTP 的对照（左 Gloeckle，右 DeepSeek）。**
左：共享主干 $f_s$ → $z_t$ → $n$ 个并行头 $f_{h_1..h_n}$（各一层 transformer）→ 共享反嵌入 $f_u$ → 预测 $x_{t+1},\dots,x_{t+n}$，每个头只看 $z_t$；标注“参数匹配：加 $n-1$ 层头就从主干减 $n-1$ 层”。
右：主模型 → 深度 1（输入加 $\operatorname{Emb}(t_{i+1})$）→ 深度 2（输入加 $\operatorname{Emb}(t_{i+2})$）……；标注“顺序、完整因果链、额外参数（V3 为 11.6B）、推理可丢或作草稿”。
中间可放 Gloeckle 附录 B 的“因果变体”小图：头串联但**不**喂真实中间 token，并注明表 S4 该变体 MBPP@1 31.9 < 并行 33.8。

**图 11.11 MTP 用于投机解码的时间线。** 每步：主模型验证上一步草稿并产生 1 个新 token + MTP 产生 1 个草稿；接受概率 $p$。期望每步 $1+p$ 个 token：$p=0.85\to1.85$，$p=0.90\to1.90$；论文实测 1.8× TPS（V3 §5.4.3），Insights 给 $p\in[0.80,0.90]$。可叠加 Gloeckle 表 S2 代码列（2/3/4 头：1.85/2.54/3.05 倍）作为“多个并行头”的对照。

**图 11.12 训练日程时间轴（横轴 0–14.8T tokens）。** 四条带：
学习率（0→2.2e-4 于 2K 步；恒定至 10T；余弦至 2.2e-5 于 14.3T；2.2e-5 至 14.633T；7.3e-6 至 14.8T）；
batch（3072→15360 于前 469B，其后 15360）；$\lambda$（0.3 至 10T，其后 0.1）；$\gamma$（0.001 至 14.3T，其后 0）；
另标 $\alpha=10^{-4}$ 全程。看点：$\lambda$ 下调与余弦起点重合，$\gamma$ 冻结与最后 500B 的低学习率段重合。

**图 11.13 消融差值哑铃图（表 4、表 5）。** 横轴差值，纵轴 10 个基准，每个基准两个点（小 / 大），数据见 §4.5；零线；BPB 行注意“越低越好”，建议取负号统一方向。

---

## 6. 算例与练习素材

1. **（计算）参数账。** 由 §3.1 的配置算出 V3 的总参数与激活参数。
   答：每层 MLA 187.1M；单专家 44.04M；总参 $=671.03$B；激活（含嵌入与输出头）37.55B。讨论：论文 “37B”、README “36.7B”，口径不同（嵌入查表不花 FLOPs，是否计入“激活”是约定）。
2. **（计算）稠密层宽度的巧合。** 为什么前 3 层稠密 FFN 的中间维是 18432？答：$18432=9\times2048$，与 MoE 层每 token 实际经过的 1 个共享 + 8 个路由专家的总中间维相同，稠密层与 MoE 层每 token 的 FFN 计算量相等；V2 同理 $12288=8\times1536$（2 共享 + 6 路由）。（设计意图为推测，算术是事实。）
3. **（概念）偏置能不能进门控值？** 若把 $s+b$ 同时用作门控值会怎样？答：$b$ 会改变专家输出的加权，从而改变前向函数，$b$ 的调节就不再是“只影响分诊”，还会与语言模型目标相互作用（偏置本身没有梯度，却会改变其他参数的梯度方向）；V3 与 LossFree 都明确 $b$ 只进入 Top-K 选择。
4. **（计算）完整路由算例。** 用 §5.2 图 11.3 的 16 个 logit 与偏置，$K_r=4$、4 节点 × 4 专家、$M=2$，按代码流程算选中的专家与门控值。
   答：节点分 [1.3372, 1.4309, 1.2426, 1.2801] → 节点 {0,1}；专家 {0,2,4,5}；$g$=[0.2776, 0.2163, 0.2568, 0.2493]；乘 2.5 后 [0.6941, 0.5407, 0.6421, 0.6231]。追问：若去掉偏置但保留节点限制？（节点分变为 $s$ 的最大两个之和：[1.3672, 1.4009, 1.3426, 1.2201] → 节点 {0,1}，专家 {0,2,4,5}，与有偏置时相同——请读者自己验算，结论：本例中专家 10 落选主要是节点限制造成的。）
5. **（计算）偏置更新一步。** 负载 [5,2,1,4]，旧偏置 [0.1,−0.2,−0.1,−0.1]，$u=0.1$。答：均值 3，$e=[-2,1,2,-1]$，新偏置 [0.0,−0.1,0.0,−0.2]；MaxVio $=2/3\approx0.667$。
6. **（计算）$\mathcal L_{\rm Bal}$ 的量级。** $\alpha=10^{-4}$、$N_r=256$、$K_r=8$。答：完全均衡时 $10^{-4}$；某序列所有 token 都选同样 8 个专家且 $P$ 在其上均匀时 $3.2\times10^{-3}$（$=\alpha N_r/K_r$）；与语言模型交叉熵（~2 nats）相比微乎其微，所以它只是“防极端”的护栏。
7. **（推导）$\mathcal L_{\rm Bal}$ 的梯度。** 推导 §2.10 的框中公式，并说明它为何进入 $\mathbf u_t$。可给读者一个数值检验脚本的思路（有限差分）。
8. **（计算）节点受限损失了多少组合？** 答：$\binom{256}{8}\approx4.10\times10^{14}$；至多跨 4 节点 $\approx7.16\times10^{13}$，占 17.48%；随机选 8 个专家平均跨 5.29 个节点。讨论：限制看似砍掉 82.5% 的组合，但 V2 已报告 $M\ge3$ 时性能与不受限大致持平（V2 §2.2.2），V3 取更宽松的 $M=4$ 个节点（每节点 32 个专家）。
9. **（计算）“13 个专家”从哪来？** 答：NVLink/IB = 160/50 = 3.2；IB 每节点发一次，NVLink 在节点内还能再转发约 3.2 份而不成为瓶颈；$4\times3.2=12.8\approx13$。
10. **（计算）偏置的精度。** 若 $b_i=0.6$ 用 BF16 存，加 $\gamma=0.001$ 会怎样？答：$[0.5,1)$ 内 BF16 间隔 $2^{-8}=0.0039$，0.001 小于半个间隔 0.00195，增量被舍入掉，偏置永远不动；因此偏置应以 FP32 存储与累加（GitHub 推理代码 2025-08-26 的提交把 `Gate.bias` 改为 FP32；训练端的精度论文未说明）。
11. **（计算）投机解码。** 接受率 $p=0.85$，每步 1 个草稿。答：期望每步 1.85 个 token；论文实测 1.8×。如果套用“草稿免费”的简化模型，差距约 2.7%（$1.8/1.85=0.973$）；$p=0.9$ 时期望 1.9 个，差距约 5.3%。这只是笔者的粗算：论文没给测量条件，也没定义接受率怎么统计，不能据此反推真实开销。追问：为什么只有“延迟”收益而吞吐未必提高？（Insights：“slightly hurting the throughput”——验证与草稿占用算力。）
12. **（计算）每步的负载规模。** 满 batch（15360 × 4096 tokens）时每个专家每步期望分到多少 token？答：62,914,560 × 8 / 256 = 1,966,080；每 GPU（4 个专家）7,864,320。讨论：batch 级统计的样本量极大，所以 batch 级均衡对单个序列的约束很松——这正是 §4.5.3 所说的“更灵活”，也解释了为何还需要序列级护栏。
13. **（概念）为什么不用 Expert Choice？** 答：EC 按专家挑 token，同一 batch/块内未来 token 会影响过去 token 的分配，泄漏信息；LossFree 给出每层泄漏下界 $K\log_2\frac{1-R}{R}$，$K=2,N=16$ 时每层 5.61 bits、9 层 50.5 bits；块大小 512 时出现约 10% 的异常损失下降。
14. **（概念）MTP 的公平比较。** V3 表 4 的“w/ MTP”模型与基线推理成本相同，训练成本相同吗？答：不同——训练时多了一个 MTP 模块（笔者估算前向多约 4.4% FLOPs，另加 11.6B 参数的优化器状态）；Gloeckle 则通过从主干减层做到参数匹配。所以表 4 的提升应读作“同推理成本下”的提升。

---

## 7. 勘误与常见误解

**E1 “V3 完全没有辅助损失。”** 错。V3 “主要”依赖无辅助损失策略，但保留了一个互补的序列级均衡损失，$\alpha=0.0001$（§2.1.2、§4.2）。“auxiliary-loss-free”说的是主机制。
**E2 “偏置 $b_i$ 会改变专家输出的权重。”** 错。偏置只用于 Top-K 选择；门控值取原始 $s_{i,t}$ 再归一化（V3 式 (16) 后的说明；代码 `scores.gather(1, topk_idx)` 取未加偏置的分数）。
**E3 “$\gamma=0.001$ 贯穿全程。”** 不完整。最后 500B tokens $\gamma=0$（偏置冻结）。同理“$\lambda=0.3$”只在前 10T，其后 0.1。
**E4 “MTP 接受率 85–90%”与“80–90%”。** 两个一手来源不一致：V3 §5.4.3 写 85%–90%；DeepSeek 自己的 Insights（ISCA ’25，v1 与 v2 相同）写 80%–90%。两者都给 1.8× TPS。写课时两个都引并注明出处，不要只写一个区间当作定论。
**E5 “1.8× 是免费的吞吐提升。”** 不准确。1.8× 是生成 TPS，V3 称之为 “decoding speed”（论文没有给出测量时的 batch 设置，见 §8），Insights 明说这种做法 “slightly hurting the throughput”，收益主要在延迟。
**E6 “DeepSeek 首创了顺序 / 因果链式 MTP。”** 不准确。Gloeckle 附录 B 已试过“因果”变体（后面的头叠在前面的头上），表 S4 显示并不优于并行头；EAGLE 也有类似的因果链（V3 自己承认原则相似）。V3 的区别在于：每一深度额外输入真实第 $i+k$ 个 token 的嵌入、每一深度是完整的 Transformer 块（在 V3 里就是带 256 专家的 MoE 层）、嵌入与输出头与主模型共享、目标首先是改进训练。
**E7 “节点受限路由 = V2 的设备受限路由。”** 不同：V2 按设备、$M=3$、设备分 = 设备内最高亲和度；V3 按节点（每节点 8 GPU、32 个专家）、$M=4$、节点分 = 节点内最高 $K_r/M=2$ 个亲和度之和。代码两者都叫 “group”（`n_group`、`topk_group`），V2 的 config 为 `group_limited_greedy`、`topk_group` 3。
**E8 “V3 有 671B 参数，HF 上的权重就是 671B。”** HF 权重总量 685B = 671B 主模型 + 约 11.6B MTP 独有参数 + MTP 层里重复存储的一份嵌入与输出头（各 0.93B）。README 的 “14B MTP” 即 685−671。
**E9 “sigmoid 门控是 V3 首创。”** LossFree（2024-08）主实验就用 sigmoid，并报告 sigmoid 基线优于 softmax 基线；V3 引用了它。V3 报告本身没有给出改用 sigmoid 的理由。
**E10 “LossFree 是同行评审论文。”** 它是 arXiv 预印本（只有 v1）；OpenReview 记录其 ICLR 2025 投稿（Submission3529，forum `y1iU5czYpE`）的 venueid 为 `Rejected_Submission`。OpenReview 搜索 API 能看到三份 Official Review，评分分别是 5、3、3。其方法的大规模有效性主要由 V3（技术报告，也非同行评审）背书；V3 表 5 是 DeepSeek 自己的消融。
**E11 LossFree 结论段的 token 数。** 结论写 3B 模型训练 “300B tokens”，但 §1、§4.1 与附录 B 都是 200B，且 $1728\times2048\times56514=200.0$B。以 200B 为准。1B 模型标称 100B，而按 batch × 序列长 × 步数算是 $1152\times2048\times40000=94.4$B（论文取了整数）。
**E12 LossFree 的记号与排版错误。** 算法 1 用 $e_i$ 表示负载误差，与同文式 (1) 的专家中心 $\mathbf e_i$ 撞名；算法 1 有两个 “4.”；表 4 “Addative Bias” 拼写错误。课程改用 $\gamma$ 与 $\delta_i$。
**E13 V3 式 (18) 的 Topk 集合不带偏置。** 实际路由用的是带偏置的式 (16)，而 $f_i$ 按式 (18) 统计的是不带偏置的 top-K。源码中被注释掉的旧写法是“属于该 token 的 Top-$K_r$ 集合”（未说明带不带偏置）。这可能是有意（统计“若无偏置会怎样分配”）也可能是笔误；训练代码未公开，不可确定（§8）。写课时照抄式 (18)，并加一句说明。
**E14 MTP 拼接顺序：论文 vs 推理代码。** V3 式 (21) 是 $[\RMSNorm(\mathbf h);\RMSNorm(\operatorname{Emb})]$（隐状态在前）；vLLM `deepseek_mtp.py` 与 SGLang `deepseek_nextn.py` 都是 `cat([enorm(embeds), hnorm(hidden)])`（嵌入在前）。二者必须与发布的 `eh_proj` 权重匹配才能得到可用的草稿，所以发布权重实际按“嵌入在前”排列（强推断）。对学习来说顺序无关（$M_k$ 是学出来的），但读代码时会困惑，应在课上点明。
**E15 `routed_scaling_factor=2.5` 论文里没有。** 代码在归一化门控后乘 2.5（V2 代码乘 16.0 且不归一化）。论文只在文字里说在 “width bottlenecks”（含路由专家的中间隐状态）乘“额外缩放因子”，未给数值；二者数学等价是笔者推断。
**E16 初始化标准差与词表。** 论文：初始化 std 0.006、词表 “128K”；config：`initializer_range` 0.02、`vocab_size` 129280。前者 config 值不影响已训练权重，后者“128K”为约数。不要把 0.02 当成 V3 的初始化。
**E17 附录 C 子图标签。** 图 11 的子标题写 “(a) Layers 1-7 … (e) Layers 25-27”，而对应面板（及其文件名）实际是第 1–6、7–12、13–18、19–24、25–26 层。以面板为准。标签很像 Python 的半开区间 `range(1,7)`…`range(25,27)` 被照抄成了闭区间写法（笔者推测）。热图一共覆盖 26 层；这 26 层是不是 16B 对照模型的全部 MoE 层，论文没说，不要写成定论。
**E18 “13 个专家”是 12.8。** V3 §3.2.2 的 $4\times3.2=12.8$ 被写成 “a maximum of 13 experts”。
**E19 MTP 消融不是算力匹配的。** V3 表 4 在基线上“append”一个 MTP 模块，其余不变；推理成本相同，训练成本更高。Gloeckle 则是参数匹配（从主干减层）。表 4 的结论应表述为“同推理成本下 MTP 更好”。
**E20 NVLink:IB 带宽比的两个说法。** V3：160 vs 50 GB/s，“roughly 3.2 times”；Insights：“approximately 4:1”（NVLink 名义 200、实际约 160；IB 50，有效按 40）。两者可以同时成立（名义 vs 实际），引用时说明口径。
**E21 表 4 与表 5 的基线不同。** 大 MoE 的 token 数分别为 540B 与 578B，小 MoE 的基线分数也不同。不要把两张表的“Baseline”当成同一个模型。
**E22 GitHub 推理代码的历史 bug 与特例。** (a) 2025-02-14 的提交 “fix scores mask” 之前，组外专家的分数被乘以 0（`scores * mask`）而不是置为 $-\infty$。修复后改为 `masked_fill_(…, -inf)`。理论上，若某些组内专家的 $s+b<0$，分数为 0 的组外专家会压过它们，破坏节点限制。**但对发布的 V3 权重，这个 bug 实际上不起作用**：核查时读取了全部 59 个 `e_score_correction_bias`，数值都在 $[2.03, 8.05]$，所以 $s+b>2>0$ 恒成立，组内专家总能压过被置 0 的组外专家（核查者据发布权重推断；别的权重上可能不成立）。(b) `Gate.bias` 只在 `dim == 7168` 时创建（即只对 671B 配置启用偏置），16B/236B 配置走 V2 式的 `amax` 分组打分。(c) 2025-08-26 把偏置改为 FP32。
**E23 Gloeckle 自身的不一致。** 摘要与 §3.1 说实验到 13B，结论却写 “up to 7B parameters and 1T tokens”；表 1 行标签 “313B bytes” 与标题及 §3.3 的 “314B bytes” 不一致。
**E24 LossFree 式 (7) 的二项式写法。** 原文 $\binom{\frac{KT}{N}}{T}$ 上下位置与常规“从 $T$ 中选 $KT/N$”相反（笔者判断为记号问题，不影响结论 $K\log_2\frac{1-R}{R}$）；课上若引用请写成常规形式并注明。
**E25 “MTP 在推理时总是预测两个 token。”** 默认推理直接丢弃 MTP 模块；只有用作投机解码时才产生第二个 token 的草稿（V3 §2.2、§5.4.3）。DeepSeek 官方推理仓库本身没有实现 MTP（README 写明社区支持仍在开发中，时间点为发布时）。
**E26 “batch 级均衡严格更好。”** V3 §4.5.3 同时指出两个效率风险（小 batch/单序列内不均衡、推理时领域迁移），分别靠大 micro-batch 与冗余专家部署解决；这不是免费的。
**E27 “sign 规则让 $\sum_i b_i=0$。”** 不成立。sign 更新不守恒总和（模拟中 $\gamma=10^{-2}$ 时偏置整体漂到 0.21–0.61），但给所有偏置加同一常数不改变 Top-K，所以无害；课上应强调“只有偏置之差有意义”。**发布的权重也证实了这一点**（核查时用 HTTP Range 读了 HF safetensors 中全部 59 个 `e_score_correction_bias`，都是 F32，形状 [256]）：所有偏置都是正数，范围 $[2.03, 8.05]$，各层均值从约 2.23（第 59 层）到约 8.04（第 48 层）不等；层内极差却只有 0.013（第 4 层）到 0.221（第 59 层），中位数 0.045。也就是说，整体漂移比层内差值大一到两个数量级，而真正起作用的只是那点层内差值。
**E28 “辅助损失伤害性能”的证据强度。** LossFree 的证据是 1B/3B 上的 $\alpha$ 扫描（困惑度差约 0.6%）与 V3 表 5；“干扰梯度”是机制性解释而非被分离测量的量。§4.5.3 还显示：同样的 batch 级约束用辅助损失实现也能达到相同验证损失（2.253 / 2.080）——说明关键可能在“均衡的范围”而不在“是否用损失”。这一点 V3 自己的实验就支持，课上应当讲出来。

---

## 8. 未能核实 / UNVERIFIED

1. **$f_i$ 用哪个 Topk 集合。** V3 式 (18) 印的是不带偏置的 Topk；训练代码未公开，实际实现不可知。
2. **“过载/欠载”的阈值。** V3 只说过载减 $\gamma$、欠载加 $\gamma$，未定义阈值（LossFree 用均值）；负载恰等于均值时（sign(0)=0？）如何处理，两篇都未说明。
3. **节点选择是否使用加偏置的分数（训练时）。** 推理代码（HF、GitHub）用 $s+b$；论文只说 “affinity scores”。
4. **偏置更新是在全局 batch 上还是每个 EP/DP 组上统计。** V3 说“整个 batch”，未说明跨数据并行如何聚合。
5. **偏置在训练中的数值精度**（FP32 与否）。已确认的只有两点：推理代码 2025-08-26 改成了 FP32；发布 checkpoint 里偏置以 F32 存储，而门控权重是 BF16（safetensors 头部）。训练时用什么精度，仍然不知道。
6. **README_WEIGHTS 的 36.7B / 2.4B 口径。** 笔者按 config 逐项得 37.55B / 2.54B，差约 0.85B / 0.14B，未找到解释。
7. **表 4（MTP 消融）基线用的是哪种均衡方法**（辅助损失还是无辅助损失）——论文未说。
8. **§5.4.3 接受率与 1.8× TPS 的测量条件**（采样温度、batch、硬件、生成主题的定义）——论文未给。
9. **MTP 模块输入的主模型表示是最终 RMSNorm 之前还是之后的隐状态。** 论文只说 “representation given by the main model”；vLLM 代码注释提到把 “post-final-norm hidden” 回收给下一步草稿，但主模型→MTP 的首步接口未逐行核对。
10. **MTP 模块在后训练（SFT/RL）阶段是否继续训练。** V3 报告未见说明（未系统检索第 5 节以外的全部文字）。
11. **V3 为何改用 sigmoid、为何选中后归一化、为何缩放 2.5。** 无官方理由；LossFree 附录 C 只给出 sigmoid 优于 softmax 的经验结果。
12. **图 9 / 图 11 的具体数值。** 只能读热图颜色，不能给精确数字。
13. **LossFree 图 2 除表 2 两点外的坐标。** 读图估计值（≈0.10/9.69，≈2.22/9.527，≈5.07/9.54）。
14. **LossFree 在 ICLR 2025 被拒的理由。** 按 forum 列出笔记的 API 返回错误，但 `notes/search` 能看到三份 Official Review（评分 5 / 3 / 3）。其中一份批评参考文献太少（只引了约 10 篇）。没有取到最终 decision/meta-review 的文字，所以被拒的正式理由仍未核实。
15. **V3 预训练 batch 爬坡的具体形状**（“gradually increased”），因此总步数只能估计（§4.3 按线性假设约 24 万步）。
16. ~~HF checkpoint 中偏置 $b_i$ 的实际分布~~ **已在核查中解决**：用 HTTP Range 读取全部 59 个张量（层 3–61），全部为 F32，数值在 $[2.03, 8.05]$，层内极差 0.013–0.221（见 E27）。仍未知的是：这些偏置是训练结束时的原值，还是后训练阶段（SFT/RL）又改过的值。
17. **V4 的 `sqrtsoftplus`、`num_hash_layers` 机制。** 本笔记只记录 config 字段，机制归第 18 课核对。
18. **EAGLE、Leviathan、Switch、DeepSeekMoE 的内容。** 本笔记只核对了它们的元数据和被 V3/LossFree 引用的方式，未读正文。

---

## 9. 参考文献卡片草稿

```
id: deepseek-v3
title: DeepSeek-V3 Technical Report
a: DeepSeek-AI (Aixin Liu et al.), 2024, arXiv technical report, arXiv:2412.19437 (read v2, 2025-02-18; v1 2024-12-27)
url: https://arxiv.org/abs/2412.19437
finding: 671B 总参 / 37B 激活的 MoE 模型，在 V2 的 MLA + DeepSeekMoE 之上改用 sigmoid 门控、以偏置实现无辅助损失均衡（γ=0.001→0）、保留 α=1e-4 的序列级护栏损失、节点受限 M=4、不丢 token，并加入 D=1 的顺序式 MTP（λ=0.3→0.1），消融显示两者在多数基准上更好。
matters: 本课每个公式（式 12–25）、每个超参数和两张消融表的唯一一手出处；第 12 课的 FP8/DualPipe 也出自同一报告。
kind: paper
```

```
id: wang2024lossfree
title: Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts
a: Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, Damai Dai, 2024, arXiv preprint (ICLR 2025 submission, rejected per OpenReview), arXiv:2408.15664 (v1)
url: https://arxiv.org/abs/2408.15664
finding: 在门控分数上加逐专家偏置、只影响 Top-K，并按上一 batch 负载用 sign 规则（u=0.001）更新，在 1B/3B 模型上同时得到更低困惑度（9.50 vs 9.56；7.92 vs 7.97）和更好均衡（MaxVio_global 0.04 vs 0.72/0.52）。
matters: V3 偏置机制、更新速度取值、MaxVio 指标、“干扰梯度”论点和 sigmoid 优于 softmax 的经验依据都来自这里；课上讲“为什么”时引它，讲“V3 用了什么”时引 V3。
kind: paper
```

```
id: gloeckle2024mtp
title: Better & Faster Large Language Models via Multi-token Prediction
a: Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve, 2024, ICML 2024 (PMLR 235:15706–15734), arXiv:2404.19737 (v1)
url: https://proceedings.mlr.press/v235/gloeckle24a.html
finding: 共享主干上接 n 个独立并行输出头同时预测未来 n 个 token，在参数匹配下对大模型的代码生成明显有益（13B：MBPP +17%、HumanEval +12%），小模型反而变差，4-token 模型自投机解码可加速约 3 倍。
matters: 它是 V3 MTP 的直接灵感与对照组：课上用它说明“并行独立头”与 V3“顺序完整因果链”的差别，并用其附录 B 纠正“因果链式 MTP 是 DeepSeek 首创”的误解。
kind: paper
```

```
id: deepseek-v2
title: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
a: DeepSeek-AI, 2024, arXiv preprint, arXiv:2405.04434 (read v5)
url: https://arxiv.org/abs/2405.04434
finding: V2 使用 softmax 亲和度、设备受限路由（M=3，按设备内最高亲和度选设备）、专家级/设备级/通信三种均衡损失（α1=0.003、α2=0.05、α3=0.02）和设备级 token 丢弃。
matters: 本课所有“V3 改了什么”的对照基准；V3 表 5 的辅助损失基线强度也直接取自 V2 与 V2-Lite。
kind: paper
```

```
id: deepseek-insights-v3
title: Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures
a: Chenggang Zhao et al., 2025, ISCA '25 Industry Track (DOI 10.1145/3695053.3731412), arXiv:2505.09343 (read v2)
url: https://arxiv.org/abs/2505.09343
finding: 从硬件协同设计角度解释节点受限路由（8 组×32 专家、每 token 至多 4 节点、IB 代价从 8t 降到 Mt），并报告 MTP 第二个 token 的接受率为 80%–90%、TPS 提升 1.8 倍但略损吞吐。
matters: 为节点受限路由提供 V3 报告未写的动机算账，并提供与 V3 §5.4.3（85%–90%）不一致的第二个接受率来源，课上须并列引用。
kind: paper
```

```
id: eagle
title: EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
a: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang, 2024, arXiv:2401.15077 (v3)
url: https://arxiv.org/abs/2401.15077
finding: （仅核对元数据）V3 §2.2 称其“保持预测因果链”的原则与 EAGLE 相似，但 EAGLE 的主要目标是投机解码。
matters: 用来说明 V3 MTP 的结构在投机解码文献中已有先例，V3 的新意在于把它当作训练目标。
kind: paper
```

```
id: leviathan2023spec
title: Fast Inference from Transformers via Speculative Decoding
a: Yaniv Leviathan, Matan Kalman, Yossi Matias, 2023, ICML 2023 (Oral), arXiv:2211.17192 (v2)
url: https://arxiv.org/abs/2211.17192
finding: （仅核对元数据）V3 §5.4.3 引用它作为投机解码框架，用 MTP 模块产生草稿 token。
matters: 讲 MTP 推理加速时的背景引用；投机解码本身在第 4 课展开。
kind: paper
```

```
id: switch-transformer
title: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
a: William Fedus, Barret Zoph, Noam Shazeer, 2021/2022, JMLR, arXiv:2101.03961 (v3)
url: https://arxiv.org/abs/2101.03961
finding: （仅核对元数据）V3 与 LossFree 都把它（与 GShard 一起）引为“传统辅助损失负载均衡”的出处。
matters: 第 7 课讲 f_i·P_i 型辅助损失时的原始出处；本课只作回指。
kind: paper
```

```
id: deepseekmoe
title: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
a: Damai Dai et al., 2024, arXiv:2401.06066 (v1)
url: https://arxiv.org/abs/2401.06066
finding: （仅核对元数据）V3 的 FFN 采用其细粒度专家 + 共享专家架构；LossFree 的实验骨架也是它。
matters: 第 7 课的主论文；本课式 (12) 的结构来源。
kind: paper
```

```
id: hf-deepseek-v3-config
title: deepseek-ai/DeepSeek-V3 config.json, modeling_deepseek.py, model.safetensors.index.json
a: DeepSeek-AI, Hugging Face model repository (fetched 2026-09-24)
url: https://huggingface.co/deepseek-ai/DeepSeek-V3
finding: 发布配置为 256 路由 + 1 共享、每 token 8 个、n_group=8/topk_group=4、sigmoid、noaux_tc、routed_scaling_factor=2.5、num_nextn_predict_layers=1；门控代码按“组内最大两个加偏置分数之和”选组，用未加偏置的分数归一化后乘 2.5。
matters: 用来核对论文规格、揭示论文未写的实现细节（2.5 缩放、组限制写法、MTP 层存储方式），是“论文 vs 代码”对照的依据。
kind: artefact
```

```
id: gh-deepseek-v3-inference
title: deepseek-ai/DeepSeek-V3 GitHub repository (inference/model.py, inference/configs, README, README_WEIGHTS)
a: DeepSeek-AI, GitHub (fetched 2026-09-24; relevant commits 2025-02-14 "fix scores mask", 2025-08-26 "fp32 gate bias")
url: https://github.com/deepseek-ai/DeepSeek-V3
finding: 官方推理实现的 Gate 与 HF 版本一致，但偏置只在 dim==7168 时启用，早期版本的组掩码用乘 0 而非 −∞，后改为 FP32 偏置；README_WEIGHTS 给出主模型 671B（激活 36.7B）与 MTP 11.5B 的拆分。
matters: 提供参数口径（685B 的来源）与代码演化史，支撑本笔记的勘误 E8、E22。
kind: artefact
```

```
id: vllm-sglang-mtp
title: vLLM deepseek_mtp.py and SGLang deepseek_nextn.py
a: vLLM project / SGLang project, GitHub (fetched 2026-09-24)
url: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py ; https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_nextn.py
finding: 两个主流推理引擎都按 cat([enorm(嵌入), hnorm(隐状态)]) → eh_proj → 一个解码层 → 共享输出头 实现 V3 的 MTP 草稿，拼接顺序与论文式 (21) 相反。
matters: 课上讲 MTP 结构时用于说明“论文记号与可运行实现的对应关系”，并提醒读者读代码时的顺序差异。
kind: artefact
```

```
id: hf-later-configs
title: config.json of DeepSeek-V3.1, V3.2-Exp, V3.2, R1, V4-Flash, V4-Pro
a: DeepSeek-AI, Hugging Face (fetched 2026-09-24)
url: https://huggingface.co/deepseek-ai
finding: V3.1/V3.2/R1 的路由字段与 V3 完全相同；V4-Flash/Pro 仍为 noaux_tc 与 1 层 MTP，但打分函数改为 sqrtsoftplus、激活 6 个、去掉组限制字段并新增 3 个哈希层。
matters: 说明本课的两项设计（无辅助损失均衡、MTP）一直沿用到 V4，并为第 18 课标出需要核对的变化点。
kind: artefact
```

---

## 10. 核查记录（2026-09-24，对抗式复核）

复核者没有沿用研究阶段的缓存，而是重新抓取了一手来源：V3 v1/v2 的 LaTeXML 文本、V3 v2 的 LaTeX 源码包和 PDF（用来定页码）、LossFree 源码包（含图 1 PNG）、Gloeckle、V2 v5、Insights v1/v2、PMLR 页面、OpenReview API、HF 的 `config.json`/`modeling_deepseek.py`/safetensors index 与分片头部、GitHub `inference/model.py` 及其提交 diff、vLLM `deepseek_mtp.py`、SGLang `deepseek_nextn.py`。§4 的派生数字全部用 Python 重算过。

**逐项确认无误（20 组高风险论断）**
1. V3 规格（§4.2 p.22）：61 层、$d=7168$、$n_h=d_h=128$、$d_c=512$、$d_c'=1536$、$d_h^R=64$、前 3 层稠密、1+256 专家、中间维 2048、激活 8 个、至多 4 节点、$D=1$、init std 0.006、671B/37B。
2. $\gamma$ = 0.001（前 14.3T）→ 0.0（最后 500B）；$\alpha$ = 0.0001；$\lambda$ = 0.3（前 10T）→ 0.1（其余 4.8T）。三者都在 §4.2 p.23，原文逐字核对。
3. 式 (12)–(16)、(17)–(20)（含被注释掉的 $f_i$ 旧写法）、(21)–(25)：对照 `main.tex` 逐符号一致。
4. §5.4.3（p.35）：85%–90%，1.8 倍 TPS。Insights §2.3.3 的 v1 与 v2 措辞相同：80%–90%、1.8x、“slightly hurting the throughput”、MTP 模块只有单层。
5. §3.2.2（p.13）：NVLink 160 GB/s、IB 50 GB/s、“roughly 3.2 times”、“a maximum of 13 experts”、20 个 SM。Insights §4.3：“approximately 4:1”，名义 200、实际约 160，有效带宽按 40 GB/s 算，8t → Mt。
6. 表 4 与表 5 的全部 40 个单元格、token 数（540B / 578B）、§4.5.3 的 2.258/2.253/2.253 与 2.085/2.080/2.080：对照源码表格核实；§4.5 的差值与计数重算后一致。
7. v1→v2 的差异：重新 diff 两版 LaTeXML，只有 §3.5.1 的错字、§3.5.2 的 FP8 累加段落、表 6（chat 模型表，源码 `tables/chat_evaluation.tex`）DROP 单元格三处。
8. LossFree：表 2（9.56/0.72、9.50/0.04、7.97/0.52、7.92/0.04）；$u=0.001$；算法 1（两个 “4.”）；结论段误写 “300B”；附录 B 的 batch、步数、学习率；附录 D 的 “more than 50 bits”；约 10% 的异常损失下降；附录 C.1/C.2；表 5 的模型配置。图 1 核对了 PNG 原图：负载 [5,2,1,4]，偏置 [0.1,−0.2,−0.1,−0.1] → [−0.0,−0.1,−0.0,−0.2]。
9. OpenReview：ICLR 2025 Submission3529，venueid = `ICLR.cc/2025/Conference/Rejected_Submission`。
10. Gloeckle：表 S4（causal 31.9/20.9 vs parallel 33.8/24.0）；表 S7 全部数字；表 1、表 S2；摘要中的 12%/17%；“remove $n-1$ layers”；$O(nV+d)\to O(V+d)$；3.0× 与 “2.5 accepted tokens out of 3”；图 3 标题；结论段 “up to 7B”；表 1 行标签 313B vs 314B；附录 B 的因果与反因果式。PMLR 页码 15706–15734，日期 2024/07/08。
11. V2：式 (20)–(25)；$M=3$；$\alpha_1/\alpha_2/\alpha_3$ = 0.003/0.05/0.02；V2-Lite $\alpha_1=0.001$；容量因子 1.0；约 10% 的序列不丢 token；“$M\geqslant3$” 大致持平；width bottleneck 的定义。
12. HF `config.json`（V3 / V3-Base / V3-0324 / R1 / V3.1 / V3.2-Exp / V3.2 / V4-Flash / V4-Pro）以及 V2 `config.json` 的所有字段，与 §3.1、§3.3、§3.5 一致。
13. HF `MoEGate`：`topk(2)` 写死；组外分数用 `-inf` 屏蔽；门控值取未加偏置的 `scores.gather`；归一化后乘 `routed_scaling_factor`；`assert not self.training`。V2 `group_limited_greedy` 用 `.max`，不归一化时乘 16.0。
14. GitHub 提交：`1398800ebf`（2025-02-14，“fix scores mask”，`scores * mask` → `masked_fill_(-inf)`）；`4592be48c0`（2025-08-26，“fp32 gate bias”）；`bias` 仅在 `dim == 7168` 时创建；`convert.py` 跳过 `model.layers.61`，即官方推理代码不实现 MTP。
15. safetensors index：`model.layers.61.{embed_tokens, shared_head.head, eh_proj, enorm, hnorm, shared_head.norm}` 都在，层 61 含 256 个路由专家和 1 个共享专家，共 59 个 `e_score_correction_bias`。README 给出 685B = 671B + 14B；README_WEIGHTS 给出 36.7B / 11.5B / 2.4B。
16. vLLM：`torch.cat([inputs_embeds, previous_hidden_states])`；SGLang：`cat(enorm(embeds), hnorm(prev_hidden))`。两者都是嵌入在前，与论文式 (21) 的顺序相反。最后提交日期 2026-09-24 与 2026-09-19 核对无误。
17. 附录 C：源码里的子图标签是 “Layers 1-7 / 7-13 / 13-19 / 19-25 / 25-27”，图 PDF 文字层是 Layer 1–6 … 25–26。图 9 横轴 64 个专家，色标 0–10。
18. §4 的算术全部重算：MLA 187,107,328；专家 44,040,192；总参 671.03B；激活 37.55/36.63/35.70B；MTP 11.61B；HF 684.5B；KV 35,136 / 70,272 B / 56.9×；$\binom{256}{8}$、≤4 节点 17.48%、期望 5.29；步数 227,785 / 12,424 / 7,947；困惑度比值；LossFree 94.37B / 200.0B / 70.78M；EC 5.61 / 50.5 bits；图 11.3 的路由算例；$\mathcal L_{\rm Bal}$ 梯度公式（链式法则逐项核对）。
19. 各 arXiv 版本日期：V3 v1 2024-12-27、v2 2025-02-18；LossFree v1 2024-08-28；Gloeckle v1 2024-04-30；V2 v5 2024-06-19；Insights v1 2025-05-14、v2 2025-12-23，DOI 核对无误。
20. V3 的部署与并行设置（§3.2、§3.4）：PP16 / EP64 / ZeRO-1、不用 TP；预填充 4 节点 32 GPU、32 个冗余专家；解码 40 节点 320 GPU、EP320、64 个 GPU 放冗余与共享专家；每 10 分钟调整；“9 experts”。

**改动（错误或过度表述）**
- §2.1 式 (7)(8)：原稿把 $[\mathbf q_{t,1};\dots]$ 简写了，与“逐字转录”的说法不符。已按 `main.tex` 恢复 $\mathbf q_{t,2}$ 项。
- C9 页码：式 (12)–(15) 在 p.8，但所引的 “uses the sigmoid function…” 那句在 p.9 页首（PDF 逐页文字核对）。
- §0.1 V3 作者数：arXiv 作者栏共 200 条，其中第一条是机构名 “DeepSeek-AI”，所以个人作者是 199 人。
- §2.5 交叉引用：“推理代码用 $s+b$（§2.10）”应为 §2.9。
- §2.9 精度：原稿说 HF 与 GitHub 的门控“全部在 float32 中计算”，这只对 HF 成立。GitHub 版用 BF16 算 logits 和 sigmoid（`set_default_dtype(bfloat16)`，发布的 `gate.weight` 是 BF16），加 FP32 偏置后才提升到 FP32。
- §5.2 图 11.3 的“看点”：原稿说专家 10 “因偏置与所在节点落选而被排除”。重算发现，只加偏置、不限节点时专家 10 仍入选；不加偏置、只限节点时照样落选。所以排除它的完全是节点限制。原稿把原因归错了，已改正（与练习 4 一致）。
- E22(a)：原稿推断旧代码“乘 0”的 bug 可能破坏节点限制。读取发布偏置后发现 $b\in[2.03,8.05]$，因此 $s+b>2$，这个 bug 对发布的 V3 权重不起作用。已加限定。
- E17：“16B 对照模型共 26 个 MoE 层”是推断，已改为“热图覆盖 26 层”，并注明标签像半开区间（笔者推测）。
- E11、E5、练习 11：“实际 94.4B”“单请求解码速度”“说明额外开销约 2.7%”都是把推算写成了事实，已改成有条件的表述。
- §4.5 LossFree 图 1：注明 $u=0.1$ 与 Top-2 是从图中数字反推的，图上没有标。

**新增证据（原属 §8 的项，现已部分解决）**
- §8-16 偏置分布：59 个张量全部为 F32，数值在 $[2.03,8.05]$，层内极差 0.013–0.221（中位数 0.045），已写入 E27 和 §4.3。这是 E27“sign 规则不守恒总和”的实证，也说明发布偏置远超 BF16 能分辨 $\gamma$ 的范围。
- §8-5：确认发布 checkpoint 中偏置以 F32 存储；训练时的精度仍然未知。
- §8-14：通过 OpenReview 搜索 API 看到三份评审，评分 5/3/3。decision 文本仍未取到。

**保留在 §8、仍未核实的**：$f_i$ 用的是哪个 Top-K 集合、过载阈值、训练时节点打分是否带偏置、跨 rank 如何聚合负载、README 的 36.7B 口径、表 4 基线用的均衡方法、1.8× 的测量条件、MTP 输入是否经过最终 norm、MTP 在后训练阶段是否继续训练、sigmoid/2.5 的设计理由、热图数值、LossFree 图 2 的读图值、batch 爬坡的形状、V4 机制、EAGLE 等论文的正文。


---

## 11. 补充核实（写课时，2026-09-24）

写第 11 课时新核实或新算出的内容。外部来源给出原文短语（≤15 词）与位置；派生数字给出算法，脚本在 `deepseek/figures/gen/l11_*.py`（图用）或写课者的 scratchpad（算例用，未入库）。

### 11.1 Leviathan 等人的投机采样规则与期望产出（新读正文）

来源：arXiv:2211.17192 的 HTML 全文（抓取 2026-09-24）。

- **接受规则**（§2.3 “Speculative Sampling”、算法 1）：草稿 token $x\sim q$，“keeping it if q(x)≤p(x)”，否则 “reject the sample with probability 1−p(x)/q(x)”。
- **拒绝后的重采样**（§2.3）：从 $p'(x)=\mathrm{norm}(\max(0,\,p(x)-q(x)))$ 采样。
- **分布不变**（§2.3，证明在附录 A.1 “Correctness of Speculative Sampling”）：输出服从 $p$。
- **期望产出**（§3.1 “Number of Generated Tokens”，式 (1)）：$\E(\#\text{generated tokens})=\dfrac{1-\alpha^{\gamma+1}}{1-\alpha}$，$\alpha$ 为期望接受率，$\gamma$ 为草稿数；推导假设各位置接受独立同分布。
- 课上用途：第 11 课 §7 的前置知识框与式 (11.26)。原 §0.1 把 Leviathan 列为“仅元数据”，现已核对上述三点。

### 11.2 玩具模拟的复现与新增统计（笔者模拟，非论文数据）

脚本 `deepseek/figures/gen/l11_sim_balance.py` 按 §5.2 图 11.6 的设定（16 专家、top-2、$d=32$、$\mathbf e_i\sim\mathcal N(0,I/d)$、热度 $c_i\in[-1,1]$ 线性、每步 4096 个 $\mathbf u\sim\mathcal N(0,I)$、seed 0）重跑，第 1/10/30/100/300/1000/3000 步的 MaxVio 与 §5.2 记录的三组数**逐位一致**（脚本内有断言）。新增：

| 量 | $\gamma=10^{-4}$ | $\gamma=10^{-3}$ | $\gamma=10^{-2}$ |
|---|---|---|---|
| 最后 100 步平均 MaxVio | 0.080 | 0.083 | 0.153 |
| 3000 步后偏置范围 | $[-0.175,\,0.218]$ | $[-0.153,\,0.239]$ | $[0.210,\,0.610]$ |
| 偏置极差 | 0.393 | 0.392 | 0.400 |

采样噪声地板：每个 token 从 16 个专家里均匀随机选 2 个、每步 4096 个 token，2000 次模拟的平均 MaxVio 为 **0.076**（seed 1）。

### 11.3 第 11 课算例与练习的派生数字（笔者复算，Python）

- **softmax / sigmoid 对照**（§2，图 11.3，构造数字）：$\mathbf z=(2.0,1.5,0.5,0,-0.5,-1.0)$；softmax $=(0.4769,0.2893,0.1064,0.0645,0.0391,0.0237)$；sigmoid $=(0.8808,0.8176,0.6225,0.5,0.3775,0.2689)$，和 3.467；Top-2 门控 V2 $(0.4769,0.2893)$（和 0.766），V3 $(0.5186,0.4814)$；把 $z_3$ 改为 1.4 后 V2 变为 $(0.4128,0.2504)$，V3 不变。
- **偏置耦合**（§2，构造数字，$K=1$，$b_2-b_1=0.05$）：token A $z=(2,1.8,0,0,0,0)$：softmax $s_1-s_2=0.0768$；token B $z=(2,1.8,1.7,1.7,1.7,1.7)$：$0.0379$；sigmoid 两者都是 $0.0226$。
- **$f_iP_i$ 损失的梯度**（§3）：$\partial\mathcal L/\partial z_{k,t}=\frac{\alpha}{T}s_{k,t}(f_k-\bar f_t)$（softmax 雅可比代入）；算例 $f=(3,1,0,0)$、$z=(1,0.5,0,-0.5)$：$s=(0.4551,0.2760,0.1674,0.1015)$，$\bar f=1.6412$，$s_k(f_k-\bar f)=(0.6183,-0.1770,-0.2747,-0.1666)$，与有限差分一致。
- **§2.10 的 $\mathcal L_{\rm Bal}$ 梯度公式**：随机向量（$N=6$、$d=5$、seed 1）有限差分复核，最大误差 $1.3\times10^{-10}$。
- **偏置的对偶解释**（§4，笔者解读）：容量约束 $\sum_tx_{t,i}\le C$ 的拉格朗日乘子 $\lambda_i$；固定 $\lambda$ 时每个 token 对 $s_{i,t}-\lambda_i$ 取 Top-K；对偶次梯度 $C-\mathrm{Load}_i$；$b_i=-\lambda_i$ 且取 sign 步长即得 V3 的更新规则。V3 与 LossFree 都没有这样表述。
- **练习数字**：Expert Choice 泄漏下界 $N=256,K=8$：$8\log_231=39.6$ 比特/层（58 层约 2299 比特）；4 节点×4 专家选 4 个：$\binom{16}{4}=1820$，至多 2 个节点 $412$（22.6%）；图 11.10 改 $M=1$：节点内全部 4 个 $s+b$ 之和为 $(2.0317,2.2586,1.8059,2.2214)$，选节点 1，$g=(0.3190,0.3096,0.1694,0.2020)$；两个草稿 $p=0.85$：$2.5725$，第二个条件接受率 0.6：$2.36$；Gloeckle 表 S2 反推逐位置接受率（假设加头不改变前面的接受率）$p_1=0.94$，$p_2\approx0.894$，$p_3\approx0.857$。
- **MTP 模块参数逐项**：$14{,}336+102{,}760{,}448+187{,}107{,}328+14{,}336+11{,}318{,}329{,}344+1{,}835{,}264+7{,}168=11{,}610{,}068{,}224$（与 §4.1 的 11.61B 一致）。

### 11.4 写课时遇到的工具问题（给后来者）

- 在本环境的 Bash 工具里，命令文本中的 `\\` 会先被折叠成 `\`（包括带引号的 heredoc）。用 heredoc 写 Python 时，普通字符串里的 `"\\times"` 会变成 `"\times"`，Python 再把 `\t` 解释成制表符，TeX 就成了 `imes`。改用 Write 工具写脚本文件，或在 heredoc 里用原始字符串并只写一个反斜杠。
- `check_lesson.py` 的控制字符检查不含制表符（`\x09`），所以上面这种错误不会被报出（KaTeX 也不报错，只是渲染成 “imes”）。本课是在截图里发现的，已修正。

### 11.5 教学复核时补充（2026-09-24）

- **`noaux_tc` 分支同时管偏置与组限制**（Hugging Face `deepseek-ai/DeepSeek-V3` 的 `modeling_deepseek.py`，`MoEGate`，抓取 2026-09-24）：`__init__` 里 `if self.topk_method == "noaux_tc": self.e_score_correction_bias = nn.Parameter(...)`，即偏置只在该取值下创建；`forward` 的同一分支里先 `scores_for_choice = scores ... + self.e_score_correction_bias`，再 `.topk(2, dim=-1)[0].sum(dim=-1)` 得组分、`topk(group_scores, k=self.topk_group)` 选组，最后 `topk_weight = scores.gather(1, topk_idx)`（用未加偏置的分数取门控值）。课上用途：§5“论文与代码对照”表把 `topk_method` 移到“偏置”行并注明组限制同在此分支。
- **§4“价格”算例**（笔者构造）：2 专家、$K=1$、4 个 token 的 $(s_1,s_2)=(0.9,0.2),(0.8,0.5),(0.7,0.6),(0.6,0.4)$，偏爱差 $(0.7,0.3,0.1,0.2)$；$\gamma=0.04$、$\sign(0)=0$ 时 $b_2-b_1$ 依次为 $0,0.08,0.16,0.24$，负载 $(4,0),(4,0),(3,1),(2,2)$ 后停住；容量 2 的 6 种分法中最优总亲和度 $3.0-0.3=2.7$，与第 4 步一致。$\gamma=0.2$ 时负载在 $(4,0)$ 与 $(1,3)$ 之间来回跳。

### 11.6 数学复核（2026-09-24）

逐条重推了课中的推导、重算了全部数字（Python，脚本在复核者的 scratchpad，未入库），并重跑了全部 `l11_*.py`（输出与仓库中的 SVG 逐字节相同）。改动如下：

- **V2 专家级均衡损失的统计范围。** 第 10 课研究笔记引 V2 §2.2.3 原文：“$T$ denotes the number of tokens in a sequence”，HF 配置 `seq_aux = true` 也表明按序列计算。所以 V2 的 $\mathcal L_{\rm ExpBal}$ 与 V3 的 $\mathcal L_{\rm Bal}$ **统计范围相同**（都是一条序列）。原稿 §5 说二者“只有两处不同：统计范围是一条序列，而不是一个 batch；……”，前半句错；已改为“结构与范围相同，不同在 sigmoid 归一化、系数 $0.003\to10^{-4}$、$f_i$ 用不带偏置的 Top-K”。§3 式 (11.8) 的 $T$ 改注为“一条序列的 token 数”，§3 算例的“这一批”改为“这条序列”。
- **BF16 算例。** $0.6$ 不能用 BF16 精确表示（最近的是 $0.6015625$），原稿写的“下一个可表示数 $0.6039$”也不是 BF16 数（$0.6015625$ 的下一个是 $0.60546875$）。改用可精确表示的 $b=0.625$：下一个数 $0.62890625$，$0.626$ 舍回 $0.625$。间隔与 $\gamma$ 之比由“16 到 62 倍”改为精确的 15.6 到 62.5 倍。练习 8(c)：FP32 间隔 $2^{-22}\approx2.38\times10^{-7}$ 约为 $\gamma$ 的 $1/4194$（原稿“小四个数量级”“精确累加”都不准，$0.001$ 本身也不能精确表示）。练习 8(b)：恰好半个间隔 $0.0078125$ 时按舍入到偶数回到 $3.0$（已写明结果）。
- **拉格朗日解读。** 原稿说“取 $C$ 为平均负载时容量约束接近等式、去掉 $\lambda_i\ge0$ 是近似”。实际上负载之和恒为 $N_r\overline{\mathrm{Load}}$，所以 $C=\overline{\mathrm{Load}}$（整数）时约束**必然全部取等号**，乘子本来就可正可负，不是近似。另补：该分配问题（二部图 $b$-匹配）的约束矩阵全单模，线性松弛最优值等于整数最优值，拉格朗日对偶无间隙（线性规划标准结论）。对偶函数由 $g(\lambda)$ 改名 $\Phi(\lambda)$，避免与门控值 $g_{i,t}$ 同名。
- **玩具模拟的描述。** 重跑 `l11_sim_balance.py`：$\gamma=10^{-2}$ 最后 100 步的 MaxVio 10%/90% 分位数为 0.109/0.200（最小 0.080、最大 0.332，均值 0.153），原稿“在 0.10–0.15 附近抖动”与均值 0.153 自相矛盾，改为“0.11–0.20 之间（80% 的步）”。图注“噪声地板，任何控制器都不可能低于它”与正文自己的单步值 0.051、0.055 矛盾，改为“稳态只能落在它附近，单步可以偶尔低于它”。$\gamma=10^{-4}$“至少几千步”改为可推的下界：偏置差约 0.39、每步最多拉开 $2\times10^{-4}$，至少约 1960 步（模拟中热、冷两专家负载都进入 $1\pm0.1$ 在第 1988 步）。$\gamma=10^{-3}$ 时最热门专家的负载在第 133 步、最冷门在第 203 步进入 $1\pm0.1$；3000 步后偏置与热度的相关系数 $-0.96$（大体单调，并非严格单调，图注加“大体上”）。
- **小数舍入。** §2 小练习 $K=3$ 的 softmax 门控之和按未舍入值是 $0.8726$（三位小数相加得 0.872），改写四位小数；练习 4(a) 节点 2 的和按未舍入值是 $1.80584$（四个四位小数相加得 1.8059），已注明。
- **Gloeckle 表 S2 的读法。** 1.94/2.78/3.50 是“每次前向产出的 token 数”（含主模型自己的 1 个；§3.2 原文 4 头时“3 个草稿平均接受 2.5 个”，$1+2.5=3.5$），原稿写成“每次前向接受”，已改（正文、图 11.13 说明、练习 12）。
- **符号冲突。** §7 前置框里大模型/草稿分布由 $p,q$ 改为 $P,Q$，因为同页 $p$ 是接受率；封面记号说明补上 $\delta_{ij}$ 与 $\delta_i$、$\lambda_i$ 与 $\lambda$、§7 的 $p$ 与 $N$。
- **其他。** Gloeckle 分解推导的“代入 (1)(2)”改为“代入第一行 / 第一、二行”（推导框行号与式号易混）；§3“梯度下降沿 $-\partial\mathcal L/\partial\mathbf u_t$ 走”改为经参数雅可比 $J$ 的严格一阶论证 $(\nabla_{\mathbf u}\mathcal L)^\top\Delta\mathbf u_t\approx-\eta\norm{J^\top\nabla_{\mathbf u}\mathcal L}^2\le0$；NVLink 的“3.2 个专家”改写为时间相等（$3.2/160=1/50$），图 11.9 的文字同步修改；“V3 的节点限制比 V2 宽松得多”改为给出依据：可选专家 4/8 对 3/8，随机组合满足限制的比例 17.48% 对 10.7%（V2 按配置 8 组、160 专家、选 6、至多 3 组，笔者用同样的容斥计算）；练习 2(b)“每 token 选的专家越多泄漏越多”只在 $R$ 固定时成立，已加条件；练习 11 的“推力小了 30 倍”改为“系数小了 30 倍”（两者的 $T$ 都是序列长度，但梯度大小还取决于权重）。
- **复核无误的部分（摘要）。** 参数账 671.03B / 37.55B / 36.63B / 73.25 GFLOPs、MTP 11,610,068,224、HF 684.49B；MaxVio 与 batch 规模；softmax/sigmoid 算例与 Ⓒ 的 13.4% 缩小；偏置耦合算例（0.0768、0.0379、0.0226）；式 (11.9)(11.10) 与 4 专家算例（有限差分误差 $1.2\times10^{-10}$）；练习 11 的梯度公式（有限差分相对误差 $2\times10^{-9}$）；两个端点 $\alpha$、$\alpha N_r/K_r$；价格算例两种步长；组合计数 17.48%、5.29、各 $j$ 的比例与练习 7 的 412/1820；路由算例与两个反事实、练习 4；MTP 下标自洽与 $T=6$、$D=2$ 对齐；投机解码 $\E[N]$、97.3%/94.7%、2.82/3.65、练习 9 与 12；KV cache 35,136 / 70,272 B / 56.9 倍 / 0.268、2.14、8.58 GiB；训练日程分段；两张消融表的差值与计数。

### 11.7 数学复核（二，2026-09-24）

再次独立重算全部数字（Python，复核者 scratchpad），重跑全部 `l11_*.py`（改动前输出与仓库 SVG 逐字节相同），并用有限差分核对式 (11.10) 两项与练习 11（误差 $\le1.2\times10^{-12}$）。数字无误；改动的是几处命题与记号：

- **$f_iP_i$ 损失的“取值范围”。** 原稿称推导给出“损失的取值范围”，并把 $\alpha$、$\alpha N_r/K_r$ 叫作“两个端点”。$\alpha N_r/K_r$ 确是上界（$f_i\le N_r/K_r$、$\sum_iP_i=1$），但 $\alpha$ 不是下界：$N_r=3$、$K_r=1$、$T=2$，亲和度 $(0.50,0.02,0.48)$、$(0.02,0.50,0.48)$ 各选专家 1、2，$f=(1.5,1.5,0)$，$P=(0.26,0.26,0.48)$，$\sum f_iP_i=0.78<1$。已改为“两个极端情形”，补上上界证明与这个反例；§3 小练习“不均衡越严重推力越大”改为式 (11.9) 给出的精确说法 $\frac{\alpha}{T}s_{k,t}\abs{f_k-\bar f_t}$。
- **§1 的 1.72 倍。** 一卡 4 个专家，最忙的卡相对均值的倍数 $\le1+\mathrm{MaxVio}$，只有同卡专家一样忙时才取到。“耗时就是 1.72 倍”改为“至多约 1.72 倍”。
- **§4 对偶推导。** “$\partial\Phi/\partial\lambda_i=C-\mathrm{Load}_i$”在折点处不是导数，改写为 $(C-\mathrm{Load}_i)_i\in\partial\Phi(\lambda)$；“全单模 ⇒ 无对偶间隙”补上中间一步：固定 $\lambda$ 时每个 token 的“选 $K$ 个”子问题放宽后仍取 0/1 解，所以拉格朗日对偶值等于线性松弛值，再由全单模等于整数最优值。
- **玩具模拟的描述。** $\gamma=10^{-2}$“此后一直在 0.11–0.20 之间”与表中第 1000、3000 步的 0.102、0.100 矛盾，改为“在 0.1–0.2 一带抖动，最后 100 步约八成落在 0.11–0.20”（重算为 79%）。图 11.7 说明“第 100–200 步之间汇合”改为实算的约第 133、203 步进入 $1\pm0.1$。Ⓒ 的“两条线只差一个常数”改为“约差一个常数 0.37”（逐专家差值 0.356–0.377，标准差 0.006），生成脚本同步改为计算并断言这个平移。
- **记号。** Expert Choice 泄漏下界的 $K/N$ 改为 $K_r/N_r$（$N$ 在课程符号表里是参数量）；Insights 的 IB 代价 $8t\to Mt$ 改写为 $8\tau\to M\tau$（$t$ 是位置下标）；softmax 雅可比“满矩阵”改为“稠密矩阵（不满秩，每行和为 0）”；MTP 记号注里“$P_i$ 在 §5 已用”改为“§3、§5”；8 个专家散在 8 个节点时跨 IB 的份数由“8 份”改为 7 份（本节点不走 IB），要点框改为“至多 4 份”。

### 11.8 事实与引用复核（2026-09-24）

逐条对照本笔记与一手来源核对了课中的事实论断、引用与跨课链接。新核实的内容（抓取日期 2026-09-24）：

- **EAGLE 正文**（arXiv:2401.15077 v3 的 LaTeXML 全文，§3 “Methods”；原 §0.1 列为“仅元数据”，§8-18 列为未读）：草稿模型由三部分组成：“the Embedding layer, LM Head, and Autoregression Head”；嵌入层与 LM head 用目标模型的参数，“do not necessitate additional training”；输入是特征序列与 “a token sequence advanced by one time step”，嵌入后拼成 $2\times$hidden_dim，自回归头 = 一个 FC 层（降回 hidden_dim）+ 一个解码层；训练损失为特征回归 “Smooth L1” 加分类损失 $L=L_{reg}+w_{cls}L_{cls}$，$w_{cls}=0.1$；70B 目标模型只训练一个不到 1B 参数的解码层。**影响**：E6 与原稿 §7 所说“V3 的新意在于每个深度额外输入真实第 $i+k$ 个 token 的嵌入、嵌入与输出头共享、完整的 Transformer 块”——这些对 EAGLE 都不成立（EAGLE 也喂提前一步的真实 token 嵌入、也共享嵌入与输出头、也用一个完整解码层）；只有“相对 Gloeckle 的因果变体喂入真实 token”成立。课中已改为：相对 Gloeckle 的区别是喂入真实 token；相对 EAGLE 的区别是 V3 与主模型从头联合训练（交叉熵、梯度回传主模型、共享参数一起更新），EAGLE 是在固定目标模型上事后训练的草稿头（特征回归 + 分类）。要点框同步修改。
- **Insights 的“4:1”口径**（arXiv:2505.09343，§4.3 首段）：“approximately 4:1. Specifically, NVLink provides 200GB/s … about 160GB/s can actually be achieved … each 400Gbps IB NIC delivers only 50GB/s … use 40GB/s for effective bandwidth”。即名义 200:50 与实际 160:40 都是 4:1；V3 的 3.2 倍是 NVLink 实际值 160 对 IB 名义值 50。原稿写“名义带宽比 4:1（……IB 按有效 40 GB/s 计）”，把两种口径混在一起，已改。
- **Insights 的延迟措辞**（§2.3.3）：“Although slightly hurting the throughput, this approach significantly improves the end-to-end generation latency.” 与课中表述一致。
- **LossFree 的 MaxVio 按层平均**（§4.1）：“we report the MaxVio averaged across all layers”。课中 §1 补了这一句（0.72、0.04 等都是各层平均值）。LossFree 的“interference gradients”在摘要、引言中出现多次，原稿“指控只有一句话”改为“归结起来是一句话”。
- **V3 “Slightly different” 的范围**（§2.1.2）：原文 “Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function … and applies a normalization among all selected affinity scores”——“略有不同”同时指 sigmoid 与选中后归一化，课中 §1 已补上后半句。V3 §2.1 “Compared with DeepSeek-V2, an exception is that we additionally introduce an auxiliary-loss-free load balancing strategy” 核对无误。
- **GitHub 推理代码的其他配置**（`inference/configs/config_16B.json`、`config_236B.json`、`inference/model.py` 的 `ModelArgs`）：16B 配置没有 `n_expert_groups` 字段，`ModelArgs` 缺省 `n_expert_groups = 1`、`score_func = "softmax"`，`route_scale` 1.0，即**不分组**；236B 配置 `n_expert_groups` 8、`n_limited_groups` 3、`route_scale` 16.0，走 `scores.amax(dim=-1)` 的组打分。E22(b) 原写“16B/236B 配置走 V2 式的 amax 分组打分”对 16B 不准确（`if self.n_groups > 1` 分支根本不进入），课中已改。
- **V2 的 `config.json`**（HF `deepseek-ai/DeepSeek-V2`）：`scoring_func` "softmax"、`norm_topk_prob` false、`routed_scaling_factor` 16.0、`topk_method` "group_limited_greedy"、`n_group` 8、`topk_group` 3、`n_routed_experts` 160、`n_shared_experts` 2、`num_experts_per_tok` 6。课中“V2 乘 16、不归一化”原来只挂 V2 论文与 V3 配置的引用，现新增参考文献卡片 `hf-v2` 并引用。
- **V2 的 token 丢弃范围**（V2 §2.2.4）：“we drop tokens with the lowest affinity scores on each device until reaching the computational budget”——按设备丢，被丢的只是该 token 在超载设备上的专家计算。原稿“超出容量的 token 在这一层被丢掉，不经过任何路由专家”“只剩残差和共享专家”过度，与第 10 课 §4 的说法也不一致，已改。
- **V4 与节点限制**：V4-Flash/Pro 的配置没有 `n_group`/`topk_group`（§3.5），第 18 课 §1 据 V4 报告写明 V4 去掉了“每 token 至多 $M$ 个节点”的限制。原稿封面说“无辅助损失均衡与节点受限路由……一直沿用到 V4”，后半句不成立，已改为只有前者沿用。
- **稠密 FFN 中间维 18432 的出处**：论文未给（§3.1），只见于配置。原稿 §1 说“表中数字都出自 V3 报告”，已注明 18432 取自配置。
- **跨课链接**：原稿“V2 的两类手段（辅助损失、丢 token）见第 10 课 §2–§3”，而第 10 课 §2 是设备受限路由、§3 是三种均衡损失、§4 是 token 丢弃，已改为 §3–§4；“交叉熵约 2 nats（第 2 课）”在第 2 课中没有这句话，改为引用 V3 §4.5.3 的 1B/3B 验证损失（2.253、2.080）作量级依据，并链接第 2 课 §2 的 nats 定义；M06 的“链式法则”链接由 §2 改为 §2–§3。
- **小的一致性修正**：§2 要点框“softmax 的雅可比是满的”改为“稠密的”（与式 (11.5) 的注一致，它不满秩）；§5 要点框“IB 流量从最多 8 份降到至多 4 份”改为“跨 IB 的份数从最多 7 份降到至多 4 份（Insights 按 8 个节点都计入，记作 $8\tau\to M\tau$）”，与正文一致；参考文献卡片 `switch` 的 matters、`hf-v3`（补权重分片与偏置取值）、`gh-v3`（逻辑一致但数值精度不同，见 §2.9）、`eagle`（改为正文核实后的内容）已相应更新。
