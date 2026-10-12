# DeepSeek 科技树 + 数学补课：课程地图

> 本文件是两条课程线（`deepseek/` 与 `math/`）的总规划。每一课的作者（人或 agent）开写前必须读完本文件和
> `style-guide.md`。改动课程范围时先改这里。建立于 2026-09-24。

## 读者与目标

- **读者**：有一定 ML/AI 背景的本科生。训练过小网络，知道什么是梯度下降、矩阵乘法、softmax；
  但 Transformer 内部细节、推理系统、MoE、强化学习对齐等术语**还不熟**，本科数学可能有漏洞。
- **目标**：从“知道 GPT 是 Transformer”进阶到“能读懂并复述 DeepSeek 每篇技术报告的每个公式、
  能算出每个设计省下了多少字节/FLOPs、能说出每个设计为什么这样做以及代价是什么”。
- **写法**：当作给自己孩子写的教材。不怕慢，不跳步。每个术语从“含义 → 理解（直觉/类比）→ 数学”展开。
- **时间线终点**：DeepSeek-V4.1-Flash（arXiv 2609.19969，2026-09-17）。

## 六条主线（整门课反复回到这六条线）

| 主线 | 演进 | 主要课次 |
|---|---|---|
| A. 注意力与 KV cache 经济学 | MHA → MQA/GQA → **MLA** → NSA → **DSA** → **CSA/HCA** → CSA2 + FP4 KV + 跨层复用 | 1, 4, 5, 9, 14, 18, 19 |
| B. 前馈层的稀疏化（MoE） | 稠密 FFN → GShard/Switch → **DeepSeekMoE**（细粒度 + 共享）→ 设备/节点受限路由 → **无辅助损失均衡** → V4 | 7, 10, 11, 18 |
| C. 数值与系统 | BF16 → **FP8**（细粒度量化）→ FP4；DualPipe、DeepEP、专家并行、PD 分离 | 12, 18, 19 |
| D. 训练目标与后训练 | 下一词预测 → **MTP**；SFT → RLHF/PPO → DPO → **GRPO** → **R1** 纯 RL → V3.2 规模化 RL → 智能体 | 2, 3, 8, 11, 13, 15, 18 |
| E. 残差流与记忆 | 残差连接 → Pre-LN → Hyper-Connections → **mHC**；**Engram** 条件记忆；视觉压缩上下文 | 1, 16, 17, 18 |
| F. 规模化规律与优化器 | Kaplan / Chinchilla → DeepSeek LLM 超参数 scaling law → AdamW → **Muon** | 2, 6, 18 |

## DeepSeek 科技树课程（deepseek/）

文件名 `deepseek/src/lesson-NN-slug.md`，图 `deepseek/figures/lNN-*.svg`，研究笔记 `deepseek/reference-docs/lesson-NN-doc.md`。
篇幅目标：每课 6–9 页（`#` 分页），1.2 万–2.5 万汉字，8–15 张 SVG 图，20–50 个行间公式，
6–12 道带答案的练习。总览、复盘两课可以短一些。

### 第 0 课 · 科技树总览：从 GPT 到 DeepSeek-V4.1 — `lesson-00-tech-tree`（short：DeepSeek 科技树）
- 一张大图：时间轴 × 六条主线，每个节点是一篇论文/一个模型（发布日期、参数量、激活参数、上下文长度）。
- 每条主线用一段话讲“问题是什么、DeepSeek 怎么一步步解决的”。
- 怎么读这门课：推荐顺序、数学补课对照表（哪一课需要哪几节数学）、图例颜色约定。
- 模型规格总表：DeepSeek LLM 7B/67B、MoE 16B、V2（236B/21B）、V2-Lite、V3（671B/37B）、R1、V3.1、V3.2、V4-Flash（284B/13B）、V4-Pro（1.6T/49B）、V4.1-Flash（552B backbone，decode 16B / prefill 8B 激活）——数字必须来自研究笔记。

### 第 1 课 · Transformer 解剖：GPT 的一次前向传播 — `lesson-01-transformer-anatomy`（short：GPT 前向传播）
- 从文本到 token（BPE 的直觉与一个手算例子）、embedding 查表、位置编码（正弦 / 可学习）。
- **缩放点积注意力**：从“检索”需求推导 $\softmax(QK^\top/\sqrt{d_k})V$；为什么是点积；$\sqrt{d_k}$ 的方差推导；softmax 的温度与饱和。
- 因果掩码（causal mask）为什么让训练可以并行；多头注意力（为什么要多个头、$W^O$ 做什么）。
- FFN（为什么是 4d、GELU）、残差连接（梯度高速公路，残差流 residual stream 的观点）、LayerNorm（Post-LN vs Pre-LN，GPT-2 的改动）。
- 输出层：unembedding、权重共享（weight tying）、logits → softmax。
- **全流程形状追踪**：以 GPT-2 small（d=768, 12 层, 12 头, V=50257）为例，从 token id 到 logits 每一步张量形状。
- 参数量公式 $\approx 12 L d^2$ 的推导与 GPT-2/GPT-3 的核对；每 token 前向 FLOPs $\approx 2N$。
- 每个组件的“数学/逻辑意义”单独成段：注意力 = 内容寻址的加权平均；FFN = 键值记忆（key-value memory 观点）；残差 = 增量更新；LN = 尺度控制。
- 图：整体架构图、注意力计算图、因果掩码矩阵、多头拆分、残差流示意、Pre/Post-LN 对比、形状追踪表格图。

### 第 2 课 · GPT 的训练：预训练、优化与 scaling laws — `lesson-02-gpt-training`（short：GPT 预训练）
- 自回归分解 $p(x)=\prod_t p(x_t\mid x_{<t})$；最大似然 ⇔ 交叉熵；困惑度。
- Teacher forcing 与“一次前向算出 T 个损失”；训练数据管线（去重、质量过滤的思想）。
- GPT-1（预训练 + 微调）→ GPT-2（zero-shot，Pre-LN，1.5B）→ GPT-3（175B，in-context learning，few-shot）。
- 优化：AdamW、warmup + cosine、梯度裁剪、混合精度（BF16）、训练 FLOPs $\approx 6ND$ 的推导。
- Scaling laws：Kaplan 2020（幂律形式、$N_{opt}\propto C^{0.73}$ 等，数字须核对）→ Chinchilla 2022（$N$ 与 $D$ 同比例，约 20 tokens/参数）。为第 6 课 DeepSeek LLM 的超参数 scaling law 铺路。
- 图：损失曲线与学习率调度、scaling law 对数坐标图（由公式计算）、6ND 的 FLOPs 分解图、GPT-1/2/3 对比表。

### 第 3 课 · 从预训练模型到助手：SFT、奖励模型、RLHF 与 DPO — `lesson-03-alignment-rlhf`（short：RLHF 与对齐）
- InstructGPT 三步：SFT、奖励模型（Bradley–Terry 偏好模型推导）、PPO（带 KL 惩罚的目标）。
- PPO 的裁剪目标、价值网络（critic）与 GAE 的直觉（详细数学在数学课 M09）。
- DPO：从 KL 约束下的最优策略闭式解推导出 DPO 损失（逐行推导）。
- 为第 8 课 GRPO 铺路：PPO 的四个模型（policy/ref/reward/value）与显存代价。
- 图：三阶段流程图、Bradley–Terry sigmoid、PPO 裁剪函数图、RLHF 中四个模型的交互图。

### 第 4 课 · 推理：自回归解码、KV cache 与带宽墙 — `lesson-04-inference-kv-cache`（short：KV cache 与解码）
- 生成是一个循环：prefill（并行处理提示）与 decode（每步一个 token）。
- **KV cache 从零推导**：为什么 K、V 可以缓存而 Q 不需要；为什么缓存是精确的（因果掩码保证历史不变）；不缓存时的 $O(T^2)$ 重复计算。
- 内存公式：每 token $2\,n_h d_h\,l$ 个元素；以 GPT-3、Llama 系列、DeepSeek LLM 67B 为例算字节数。
- 算术强度与 roofline：decode 为什么受带宽限制（每读一次权重/缓存只做很少计算）；batch 的作用。
- MQA、GQA：减少 KV 头数；质量与缓存的权衡。为第 9 课 MLA 铺路：“能不能不减头数也把缓存压小？”
- 采样：温度、top-k、top-p；投机解码（speculative decoding）的接受-拒绝推导（为第 11 课 MTP 铺路）。
- PagedAttention / 连续批处理只点到为止。
- 图：prefill/decode 时间线、KV cache 增长图、缓存 vs 重算的 FLOPs 对比（计算得出）、roofline 图、MHA/MQA/GQA 头共享示意、各模型 KV 字节柱状图（计算得出）。

### 第 5 课 · 现代 Transformer 积木：RoPE、RMSNorm、SwiGLU 与上下文扩展 — `lesson-05-modern-block`（short：现代 Transformer 积木）
- LLaMA 式块：Pre-RMSNorm、SwiGLU（为什么隐藏维取 $\tfrac{8}{3}d$）、无偏置。
- **RoPE**：从“希望 $\langle f(q,m), f(k,n)\rangle$ 只依赖 $m-n$”出发，用复数推导旋转；频率 $\theta_i=b^{-2i/d}$ 的意义（多尺度时钟）；远程衰减。
- 上下文扩展：位置插值（PI）、NTK-aware、**YaRN**（DeepSeek-V2/V3 用它扩到 128K；公式与超参数须核对）。
- RMSNorm vs LayerNorm；为什么去掉均值中心化。
- 图：RoPE 在二维平面上的旋转、多频率“时钟”图、相对位置点积随距离的变化（计算得出）、YaRN 的频率分段插值曲线（计算得出）、LLaMA 块结构图。

### 第 6 课 · DeepSeek LLM：以长期主义做 scaling — `lesson-06-deepseek-llm`（short：DeepSeek LLM）
- 2024 年 1 月：7B / 67B，2T tokens 中英双语。数据管线（去重、过滤、重混）。
- 架构：LLaMA 式；67B 用 GQA；“加深而非加宽”的选择；多步（multi-step）学习率调度器及其对续训的好处。
- **超参数 scaling law**：最优 batch size 与学习率随算力 $C$ 的幂律；用非嵌入 FLOPs/token $M$ 表示模型规模（$C=MD$），与 $6N$ 的差别推导。
- 最优模型/数据分配依赖数据质量（数据越好越该给模型更多算力）。
- 对齐：SFT + DPO；以及 DeepSeek-Coder（FIM 填空训练、仓库级数据）作为同期旁支简介。
- 图：multi-step vs cosine 学习率曲线、超参数 scaling law 拟合图（按论文公式计算）、$6N$ 与 $M$ 的比较、数据管线流程图。

### 第 7 课 · DeepSeekMoE：细粒度专家与共享专家 — `lesson-07-deepseekmoe`（short：DeepSeekMoE）
- MoE 基础：门控网络、top-k 路由、GShard / Switch Transformer、负载均衡辅助损失、容量因子与 token 丢弃。
- 两个问题：知识混杂（knowledge hybridity）与知识冗余（knowledge redundancy）。
- **细粒度专家切分**：把 $N$ 个专家切成 $mN$ 个、激活 $mK$ 个，参数与计算不变；组合数的爆炸（论文中的 120 vs 4,426,165,368 例子须核对）。
- **共享专家隔离**。
- 专家级与设备级均衡损失公式。
- 规模：2B 验证、16B、145B 的结果（数字须核对）；DeepSeek-V2-Lite / DeepSeekMoE 16B 规格。
- 图：稠密 FFN vs 传统 top-2 MoE vs 细粒度 vs 细粒度+共享（重绘论文图 2）、组合数增长图（计算得出）、路由分配热力图示意、均衡损失机制图。

### 第 8 课 · DeepSeekMath 与 GRPO — `lesson-08-deepseekmath-grpo`（short：GRPO）
- 数学语料管线：以 OpenWebMath 为种子、fastText 分类器、多轮迭代召回（120B tokens，须核对）。
- 从 PPO 到 **GRPO**：去掉价值网络，用组内奖励标准化作优势；目标函数逐项解释；KL 项的无偏估计器（k3）推导与性质（非负、无偏）。
- 结果监督 vs 过程监督；迭代 RL。
- “统一范式”：SFT、RFT、DPO、PPO、GRPO 的梯度都可写成 $\E[\mathrm{GC}\cdot\nabla\log\pi]$，逐一推导梯度系数。
- 论文的观察：RL 提升 Maj@K 而非 Pass@K（须核对表述）。
- 图：GRPO vs PPO 结构对比（重绘）、组内优势标准化示例图、k1/k2/k3 估计器对比曲线（计算得出）、数据管线迭代图。

### 第 9 课 · DeepSeek-V2（上）：多头潜在注意力 MLA — `lesson-09-v2-mla`（short：MLA 潜在注意力）
- 动机：第 4 课的 KV cache 账单；GQA/MQA 的质量代价。
- **低秩 KV 联合压缩**：$\mathbf c_t^{KV}=W^{DKV}\mathbf h_t$，$\mathbf k^C=W^{UK}\mathbf c$，$\mathbf v^C=W^{UV}\mathbf c$。
- **矩阵吸收**（absorb）：推理时 $W^{UK}$ 并入 $W^Q$、$W^{UV}$ 并入 $W^O$，逐行推导，说明为什么缓存 $\mathbf c$ 就够。
- 为什么 RoPE 破坏吸收（旋转矩阵夹在中间且依赖位置），**解耦 RoPE**：额外的共享键 $\mathbf k_t^R$ 与查询 $\mathbf q_{t,i}^R$。
- 缓存对比表：MHA $2n_hd_hl$、GQA $2n_gd_hl$、MQA $2d_hl$、MLA $(d_c+d_h^R)l$，V2 的取值（$d_c=4d_h$、$d_h^R=d_h/2$ → $\tfrac92 d_hl$，须核对）；换算成字节与“相当于几组 GQA”。
- 查询的低秩压缩（训练省激活）；MLA 的 MQA 模式 vs MHA 模式（为第 14 课 DSA 铺路）。
- FlashMLA 等推理内核简述。
- 图：MHA/GQA/MQA/MLA 四联对比（重绘论文图 3）、MLA 完整数据流图、吸收前后计算图、解耦 RoPE 结构、KV 字节柱状图（计算得出）。

### 第 10 课 · DeepSeek-V2（下）：规模化 MoE、长上下文与对齐 — `lesson-10-v2-moe-training`（short：DeepSeek-V2 训练）
- V2 规格：236B 总参数 / 21B 激活，60 层，160 路由专家 + 2 共享，激活 6 个（须核对）。
- **设备受限路由**（每 token 最多发往 $M$ 个设备）、三种均衡损失（专家级、设备级、通信均衡）、token 丢弃策略。
- 训练：8.1T tokens；YaRN 扩到 128K；节省 42.5% 训练成本、KV cache 降 93.3%、吞吐 5.76 倍（须核对表述）。
- 对齐：SFT + GRPO（两阶段 RL：推理 → 人类偏好）；DeepSeek-Coder-V2 与 V2-Lite 简介。
- 图：V2 整体架构图（重绘论文图 2）、设备受限路由示意、三种均衡损失作用对象示意、长上下文“大海捞针”热力图示意。

### 第 11 课 · DeepSeek-V3（上）：无辅助损失负载均衡与多 token 预测 — `lesson-11-v3-architecture`（short：DeepSeek-V3 架构）
- V3 规格：671B / 37B，61 层，MLA + DeepSeekMoE（256 路由 + 1 共享，激活 8 个），14.8T tokens（须核对）。
- sigmoid 亲和度 + 归一化门控（与 V2 softmax 门控的差别）。
- **无辅助损失负载均衡**：每个专家一个偏置 $b_i$ 只参与 top-k 选择、不参与门控值；更新规则 $b_i \leftarrow b_i \pm \gamma$；为什么辅助损失会伤害性能（梯度干扰）；来源论文 Wang et al. 2024（arXiv 2408.15664）。
- 互补的序列级均衡损失（极小 $\alpha$）、节点受限路由（$M=4$）、不丢 token。
- **MTP**：$D$ 个顺序模块，每个预测更远的 token，保持完整因果链；损失 $\mathcal L_{MTP}=\tfrac{\lambda}{D}\sum_k\mathcal L^k$；推理时可丢弃或用作投机解码（接受率须核对）。
- 图：V3 架构图、偏置调节的反馈回路示意、负载均衡 vs 性能曲线示意、MTP 模块结构（重绘论文图 3）、MTP 与 Gloeckle 并行头的对比。

### 第 12 课 · DeepSeek-V3（下）：FP8 训练、DualPipe 与“557.6 万美元” — `lesson-12-v3-infrastructure`（short：V3 训练基建）
- 训练成本账：2.788M H800 GPU 小时，每小时 2 美元（须核对拆分：预训练、上下文扩展、后训练），这个数字**不包括**什么（研究、消融、人员）。
- **FP8 混合精度**：E4M3 格式；细粒度量化（激活 1×128 tile、权重 128×128 block）；提升累加精度（每 $N_C$ 个元素提升到 FP32，CUDA core）；哪些算子保持高精度。
- **DualPipe**：流水线并行的气泡；前向/反向与通信重叠；与 1F1B、ZB1P 的气泡与显存对比（论文表 2 须核对）。
- 跨节点 all-to-all（IB + NVLink，每 token 最多 4 个节点），warp 专门化；显存节省技巧。
- 推理部署：prefill/decode 分离、冗余专家、专家并行规模（须核对）。
- 开源周基础设施：FlashMLA、DeepEP、DeepGEMM、DualPipe、EPLB、3FS（作为制品引用）；ISCA 2025 “Insights into DeepSeek-V3” 论文中对硬件的反思（须核对）。
- 图：FP8 数值格式位布局、细粒度量化 tile/block 示意、累加精度提升示意、DualPipe 调度图（重绘）、气泡对比表、PD 分离部署图。

### 第 13 课 · DeepSeek-R1：用强化学习激发推理 — `lesson-13-r1-reasoning`（short：DeepSeek-R1）
- **R1-Zero**：V3-Base 上直接 GRPO；基于规则的奖励（准确性 + 格式）；模板；训练中回答长度增长与“aha moment”；问题（可读性差、语言混杂）。
- **R1 四阶段流水线**：冷启动 SFT → 面向推理的 RL（语言一致性奖励）→ 拒绝采样 + SFT（约 80 万样本，须核对）→ 全场景 RL。
- 蒸馏：小模型直接蒸馏优于小模型自己做 RL（论文结论须核对）。
- 失败尝试：PRM、MCTS 为什么没用上。
- Nature 版本（2025）的补充信息：训练成本、安全评估等（须核对）。
- 图：R1-Zero 与 R1 训练流程图、回答长度随步数增长（示意或据论文数据重绘并标注来源）、奖励设计图、蒸馏关系图。

### 第 14 课 · 稀疏注意力：从 NSA 到 DSA — `lesson-14-sparse-attention`（short：DSA 稀疏注意力）
- 长上下文的 $O(L^2)$ 账单（接第 4 课）。
- **NSA**（Native Sparse Attention，ACL 2025 最佳论文）：压缩、选择、滑动窗口三分支 + 门控；硬件对齐的块选择；可训练的稀疏。
- **DSA**（DeepSeek-V3.2-Exp，2025-09）：lightning indexer（少量头、ReLU、FP8）打分 → 每个查询选 top-$k$（$k=2048$，须核对）个 KV；基于 MLA 的 MQA 模式实现。
- 两阶段续训：稠密预热（只训 indexer，用 KL 对齐主注意力分布）→ 稀疏训练；复杂度从 $O(L^2)$ 到 $O(Lk)$（indexer 仍是 $O(L^2)$ 但极便宜）。
- 成本曲线（论文图中的每百万 token 价格）。
- 图：NSA 三分支结构（重绘）、DSA 数据流（重绘）、注意力稀疏模式示意、FLOPs 随上下文长度的曲线（计算得出）、KL 对齐训练示意。

### 第 15 课 · DeepSeek-V3.1 / V3.2：混合思考、规模化 RL 与智能体 — `lesson-15-v3-2-agents`（short：V3.2 与智能体）
- V3.1：一个模型两种模式（thinking / non-thinking），UE8M0 FP8 缩放格式（须核对），工具调用。
- V3.2 的 RL：规模化 GRPO 的稳定性改动（无偏 KL 估计、off-policy 序列掩码、保持 MoE 路由、保持采样掩码等，须逐条核对）；后训练算力占比。
- 专家蒸馏（specialist distillation）、大规模智能体任务合成管线、工具使用中的思考上下文管理。
- V3.2-Speciale 与竞赛成绩（IMO/IOI，须核对表述）；DeepSeekMath-V2 自我验证简介。
- 图：混合思考模板示意、V3.2 后训练流水线图、智能体任务合成流程图、off-policy 掩码示意。

### 第 16 课 · 残差流的新几何：Hyper-Connections 与 mHC — `lesson-16-mhc`（short：mHC 超连接）
- 残差连接的恒等映射为什么重要（梯度传播、信号尺度）；Pre-LN 与 Post-LN 的两难。
- **Hyper-Connections**（ByteDance 2024）：把残差流扩成 $n$ 条，$H^{pre}$、$H^{post}$、$H^{res}$ 三个映射；表达力与不稳定性。
- **mHC**：把 $H^{res}$ 投影到双随机矩阵（Birkhoff 多面体）上，用 Sinkhorn–Knopp 迭代；性质：范数不增、乘积封闭、恒等映射可达；实现开销（须核对）。
- 基础设施：核融合、重计算、与 DualPipe 的重叠。
- 图：残差 / HC / mHC 三联结构图（重绘）、多层复合增益爆炸 vs 守恒（计算得出）、Sinkhorn 迭代收敛动画式分步图、Birkhoff 多面体（3×3 示意）。

### 第 17 课 · 记忆的另外两种形态：Engram 条件记忆与 DeepSeek-OCR 视觉压缩 — `lesson-17-engram-ocr`（short：Engram 与 OCR）
- **Engram**（2026-01，须核对标题/编号/作者）：N-gram 哈希查表作为“条件记忆”，与 MoE“条件计算”互补的新稀疏轴；上下文感知门控；MoE 与 Engram 的参数分配规律（U 形曲线，须核对）；可放到主机内存。
- **DeepSeek-OCR**（2025-10）：用视觉 token 压缩文本上下文（压缩比与精度的关系，须核对）；DeepEncoder；对长上下文“遗忘机制”的设想。DeepSeek-OCR 2（若存在，须核对）。
- 两者都在回答：“不是所有上下文都需要以完整 KV 的形式记住”。
- 图：Engram 模块在 Transformer 中的位置（重绘）、哈希查表示意、压缩比-精度曲线（据论文数据重绘并标注）、OCR 编码器结构图。

### 第 18 课 · DeepSeek-V4：百万 token 上下文 — `lesson-18-v4`（short：DeepSeek-V4）
- V4-Pro（1.6T / 49B）与 V4-Flash（284B / 13B），1M 上下文；32T+ tokens。
- **混合注意力**：CSA（沿序列维压缩 KV 后做 DSA）+ HCA（更激进压缩但保持稠密注意力）；层间如何排布；为何在 1M 上下文时只需 V3.2 的 27% 单 token FLOPs 和 10% KV cache（须核对并复算）。
- mHC 的实际集成方式；**Muon** 优化器（牛顿–舒尔茨正交化，与 AdamW 的混用方式，须核对）。
- 训练与后训练：专家模型独立训练再蒸馏合并（须核对细节）、推理强度档位（Pro-Max 等）。
- 图：V4 整体架构（重绘）、CSA/HCA 压缩与选择示意、KV cache 与 FLOPs 随上下文增长的对比曲线（计算得出）、Muon 更新几何图。

### 第 19 课 · DeepSeek-V4.1-Flash：把 KV cache 压到极限 — `lesson-19-v4-1-flash`（short：V4.1-Flash）
- 动机：长程智能体让负载“输入重”（input-heavy）；prefill 计算、KV 存储（HBM/SSD）与传输带宽是新瓶颈。
- **因果编码器–解码器（CED）**：decode 激活 16B、prefill 只激活 8B（须核对机制）。
- **CSA2**：跨层 KV cache 复用；**FP4 KV 缓存**；全局 KV 890 字节/token（约 V4-Flash 的 1/4，复算）。
- **SWA Bounded Replay**：持久化 KV（SSD/主机内存）降到约 1/8。
- 多模态 45T tokens 预训练；架构精简与扩展（须核对）。
- 图：CED 结构图、跨层复用示意、KV 字节随代际下降的阶梯图（计算得出，从 GPT-3 到 V4.1-Flash）、存储层级（HBM/主机/SSD）图。

### 第 20 课 · 总复盘：六条主线与开放问题 — `lesson-20-retrospective`（short：科技树复盘）
- 每条主线一页：演进表 + 一张汇总图 + “设计原则”总结（例如“先算账再设计”：每个创新背后的字节/FLOPs 账单）。
- 被放弃的路线与失败尝试（PRM、MCTS、token 丢弃、辅助损失……）。
- 开放问题；综合练习（跨课计算题）。

## 数学补课（math/）

文件名 `math/src/mNN-slug.md`，图 `math/figures/mNN-*.svg`。篇幅：每课 5–8 页，0.8 万–1.6 万汉字，
5–10 张图，大量手算例子和练习。**每节都要说明它在 DeepSeek 课程的哪一课被用到**，并提供反向链接。

| 课 | 文件 | 内容 | 服务于 |
|---|---|---|---|
| M01 | `m01-vectors-matrices`（short：矩阵与形状） | 向量、点积与余弦相似度、矩阵乘法的三种读法（线性映射 / 批量点积 / 列组合）、形状思维与 einsum、分块矩阵、矩阵乘法 FLOPs $=2mnk$ | 1, 4, 9 |
| M02 | `m02-rank-svd-lowrank`（short：低秩与 SVD） | 秩、列空间、低秩分解 $W=UV$、SVD、Eckart–Young 最优低秩近似、矩阵结合律与“吸收”技巧、谱范数与 Frobenius 范数 | 9, 18 |
| M03 | `m03-rotation-complex`（short：旋转与复数） | 正交矩阵、二维旋转、复数与欧拉公式、旋转的复合、相对位置性质、频率与波长 | 5 |
| M04 | `m04-probability`（short：概率与方差） | 随机变量、期望、方差、独立和的方差（$\sqrt{d_k}$）、softmax 作为分布、温度、采样、无偏估计、蒙特卡洛、重要性采样 | 1, 4, 8 |
| M05 | `m05-information-theory`（short：熵与 KL） | 信息量、熵、交叉熵 = 负对数似然、KL 散度及其非负性（Jensen）、前向/反向 KL、困惑度、k1/k2/k3 估计器 | 2, 3, 8, 14 |
| M06 | `m06-calculus-backprop`（short：反向传播） | 导数、偏导、梯度、链式法则、雅可比、softmax 的雅可比、交叉熵对 logits 的梯度 $p-y$、线性层的反传、计算图、残差的梯度高速公路 | 1, 2, 16 |
| M07 | `m07-optimization`（short：优化器） | GD/SGD、动量、Adam 的偏差修正、AdamW 解耦权重衰减、学习率 warmup/cosine/多步/WSD、batch size 与噪声尺度、梯度裁剪、Muon（牛顿–舒尔茨正交化） | 2, 6, 18 |
| M08 | `m08-floating-point`（short：浮点与量化） | 二进制、IEEE 754、FP32/BF16/FP16/FP8（E4M3、E5M2）/FP4（E2M1）、动态范围与精度、舍入误差、累加误差、缩放因子与细粒度量化、UE8M0/MX 格式 | 12, 15, 19 |
| M09 | `m09-rl-math`（short：策略梯度） | MDP 与语言模型的对应、回报、策略梯度定理（对数导数技巧）逐行推导、基线不改变期望的证明、优势、GAE、重要性采样与 off-policy、PPO 裁剪、KL 正则 | 3, 8, 13, 15 |
| M10 | `m10-compute-arithmetic`（short：算力算术） | FLOPs 计数、内存层级与带宽、算术强度、roofline、并行方式（数据/张量/流水线/专家）、集合通信（all-reduce、all-to-all）代价 | 2, 4, 12 |
| M11 | `m11-doubly-stochastic-sinkhorn`（short：Sinkhorn 与双随机） | 随机矩阵、双随机矩阵、Birkhoff–von Neumann 定理、谱范数 ≤ 1 的证明、Sinkhorn–Knopp 迭代与收敛直觉 | 16 |

## 统一约定

- 符号、术语、图的颜色语义、贯穿全课的类比：见 `style-guide.md`。**不得各课自造符号。**
- 课与课之间用相对链接互相引用：`[第 9 课 §3](lesson-09-v2-mla.html#p3)`；数学课：`[M02 §2](../math/m02-rank-svd-lowrank.html#p2)`。
- 每课封面的 `meta` 写“预计阅读 N 分钟 · 前置：第 x 课，数学 Mxx”。
- 所有论文数字、公式、超参数都来自 `reference-docs/lesson-NN-doc.md` 中核对过的条目，不凭记忆写。
