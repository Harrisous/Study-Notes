# 写作与符号规范（DeepSeek 科技树 + 数学补课）

> 所有课共用。与 `.cursor/skills/professor/SKILL.md`（讲解契约）、`.cursor/skills/lesson-markdown/SKILL.md`
> （源文件格式）、`.cursor/skills/academic-research/SKILL.md`（核对流程）一起读。本文件是中文课程的具体化。

## 1. 讲解契约（中文版）

1. **白话在前，术语在后，两者都给。** 例：“每走一步梯度就乘上 0.9，一百步之后几乎什么都不剩了，
   这就是**梯度消失**（vanishing gradient）。”术语第一次出现写成“中文（English）”，之后只用中文或
   业界通用的英文缩写（MLA、MoE、KV cache 这类业界直接用英文的，保留英文）。
2. **每个符号第一次出现都要交代三件事：形状、类型、含义。** 例：“$\mathbf c_t^{KV}\in\R^{d_c}$——第 $t$
   个 token 的压缩潜在向量，一列 $d_c$ 个数，是 MLA 真正写进缓存的东西。”
3. **先有问题，再给公式。** 公式回答读者心里已经有的问题；读者还没有这个问题，就先把问题造出来。
4. **推导不跳步。** 用 `::: deriv`，每一行写理由。禁止“可以证明”“显然”“易证”“不难看出”“众所周知”。
   要么给出推导，要么精确引用（论文、定理号、页码/节号）。
5. **五级阶梯**，每个重要概念都要走完：
   ① 问题（没有它会出什么事）→ ② 图景（类比 + 映射 + 类比失效的地方）→ ③ 形式化（定义，符号全交代）
   → ④ 推导（逐行 + 理由；是“选择”而非“推论”的地方要说明，还有什么别的选法）
   → ⑤ 数字（一个真实数值的算例，读者可以自己验算；然后讲失败模式：什么时候会坏、坏了长什么样）。
6. **前置知识就地补。** 用 `::: pre`，只讲这里需要的那一点，并链接到数学课的对应小节；写明“已经会的可以跳过”。
7. **一段一个意思，第一句就说出来。**
8. **诚实**：区分 **已证明**（有定理）、**实测**（论文实验/基准）、**推测**（作者自己的推理，标注“笔者推测”）。
   论文与开源代码不一致时明说。历史偶然就说是历史偶然。
9. **读者读到一半停下，也学到了完整的东西。** 每页结尾有小结或 `::: key`。
10. **语气**：像一位耐心的老师当面讲课，平实、具体、不煽情。不用“美妙的洞见”之类的形容词代替机制。

## 2. 全局符号表

向量默认**列向量**，粗体小写；矩阵大写。整段序列按行堆成矩阵时（每行一个 token），会明说。

| 符号 | 含义 | 备注 |
|---|---|---|
| $T$ | 序列长度（token 数） | 讨论上下文长度时也写 $L_{ctx}$ 以免与层数混淆 |
| $t, s, j$ | token 位置下标 | $t$ 当前位置，$j\le t$ 历史位置 |
| $\mathcal V$, $V=\lvert\mathcal V\rvert$ | 词表及其大小 | |
| $d$ | 模型隐藏维度（$d_{model}$） | |
| $l$ 或 $L$ | 层数 | KV cache 公式里沿用 DeepSeek 论文的 $l$ |
| $n_h$ | 注意力头数 | |
| $d_h$ | 每头维度 | 常有 $d=n_hd_h$ |
| $n_g$ | GQA 的 KV 组数 | |
| $\mathbf h_t\in\R^{d}$ | 第 $t$ 个 token 进入注意力层的隐藏状态 | DeepSeek 论文记号 |
| $\mathbf q_t,\mathbf k_t,\mathbf v_t$ | 查询、键、值 | 第 $i$ 个头写 $\mathbf q_{t,i}$ |
| $W^Q,W^K,W^V,W^O$ | 投影矩阵 | 形状写在首次出现处，如 $W^Q\in\R^{n_hd_h\times d}$ |
| $\mathbf u_t$ | 注意力层输出（残差分支） | DeepSeek 论文记号 |
| $d_{ff}$ | FFN 中间维度 | |
| $N_s,N_r,K_r$ | 共享专家数、路由专家数、每 token 激活的路由专家数 | DeepSeek-V3 记号 |
| $s_{i,t}$, $g_{i,t}$, $\mathbf e_i$ | 亲和度、门控值、专家中心向量 | |
| $b_i$ | 无辅助损失均衡中的专家偏置 | |
| $d_c$, $d_c'$ | MLA 的 KV 压缩维度、查询压缩维度 | |
| $d_h^R$ | MLA 解耦 RoPE 的每头维度 | |
| $\mathbf c_t^{KV}$, $\mathbf c_t^{Q}$ | KV / 查询的压缩潜在向量 | |
| $W^{DKV},W^{UK},W^{UV},W^{KR},W^{DQ},W^{UQ},W^{QR}$ | MLA 的下投影（D）、上投影（U）、RoPE 投影（R） | |
| $\theta$ | 模型参数 | RoPE 的频率写 $\theta_i$，上下文可区分时说明 |
| $\eta$ | 学习率 | |
| $B$ | batch size（按 token 或按序列要说明） | |
| $N$, $D$, $C$ | 参数量、训练 token 数、训练算力（FLOPs） | |
| $M$ | 每 token 非嵌入 FLOPs（DeepSeek LLM 记号） | |
| $\pi_\theta$, $\pi_{\theta_{old}}$, $\pi_{ref}$ | 当前策略、采样用旧策略、参考策略 | |
| $q$, $o$, $o_{i,t}$ | 问题（prompt）、回答、第 $i$ 个回答的第 $t$ 个 token | GRPO 记号 |
| $G$ | GRPO 的组大小 | |
| $r$, $\hat A$ | 奖励、优势估计 | |
| $\varepsilon$, $\beta$ | PPO/GRPO 裁剪系数、KL 系数 | |
| $\mathcal L$ | 损失 | |

宏（构建器已定义）：`\R \E \N \norm{} \abs{} \softmax \diag \Attn \RMSNorm \LayerNorm \FFN \KL \T \argmax \argmin \TopK \clip \Var \Cov \tr \rank \sign`。转置统一写 `^\top` 或 `\T`。

## 3. 术语表（中文 ↔ English）

首次出现用“中文（English）”，业界惯用英文的保留英文。

| 中文 | English | | 中文 | English |
|---|---|---|---|---|
| 词元 / token | token | | 注意力 | attention |
| 分词器 | tokenizer | | 缩放点积注意力 | scaled dot-product attention |
| 词嵌入 | embedding | | 多头注意力 | multi-head attention (MHA) |
| 位置编码 | positional encoding | | 因果掩码 | causal mask |
| 旋转位置编码 | rotary position embedding (RoPE) | | 前馈网络 | feed-forward network (FFN) |
| 残差连接 / 残差流 | residual connection / residual stream | | 层归一化 | layer normalization |
| 均方根归一化 | RMSNorm | | 门控线性单元 | gated linear unit (GLU / SwiGLU) |
| 逻辑值 | logits | | 反嵌入 | unembedding |
| 权重共享 | weight tying | | 自回归 | autoregressive |
| 教师强制 | teacher forcing | | 交叉熵 | cross-entropy |
| 困惑度 | perplexity | | 最大似然估计 | maximum likelihood estimation (MLE) |
| 规模化规律 | scaling law | | 算力 | compute (FLOPs) |
| 预填充 / 解码 | prefill / decode | | KV 缓存 | KV cache |
| 算术强度 | arithmetic intensity | | 屋顶线模型 | roofline model |
| 显存带宽 | memory bandwidth (HBM) | | 投机解码 | speculative decoding |
| 多查询 / 分组查询注意力 | MQA / GQA | | 多头潜在注意力 | multi-head latent attention (MLA) |
| 低秩压缩 | low-rank compression | | 矩阵吸收 | weight absorption |
| 解耦 RoPE | decoupled RoPE | | 稀疏注意力 | sparse attention |
| 混合专家 | mixture of experts (MoE) | | 路由 / 路由器 | routing / router |
| 门控值 | gating value | | 亲和度 | affinity score |
| 共享专家 | shared expert | | 路由专家 | routed expert |
| 细粒度专家切分 | fine-grained expert segmentation | | 负载均衡 | load balancing |
| 辅助损失 | auxiliary loss | | 专家并行 | expert parallelism (EP) |
| 多 token 预测 | multi-token prediction (MTP) | | 流水线并行 | pipeline parallelism (PP) |
| 流水线气泡 | pipeline bubble | | 混合精度 | mixed precision |
| 量化 / 反量化 | quantization / dequantization | | 缩放因子 | scaling factor |
| 监督微调 | supervised fine-tuning (SFT) | | 奖励模型 | reward model (RM) |
| 基于人类反馈的强化学习 | RLHF | | 近端策略优化 | PPO |
| 直接偏好优化 | DPO | | 组相对策略优化 | GRPO |
| 策略 / 价值函数 | policy / value function | | 优势 | advantage |
| 重要性采样 | importance sampling | | KL 散度 | KL divergence |
| 拒绝采样 | rejection sampling | | 蒸馏 | distillation |
| 冷启动 | cold start | | 思维链 | chain of thought (CoT) |
| 过程奖励模型 | process reward model (PRM) | | 结果奖励 | outcome reward |
| 超连接 | hyper-connections (HC) | | 流形约束超连接 | manifold-constrained HC (mHC) |
| 双随机矩阵 | doubly stochastic matrix | | 条件记忆 | conditional memory |
| 上下文长度 | context length | | 长上下文扩展 | context extension |

## 4. 图的颜色语义（全课统一）

SVG 只用 `var(--fig-*, #fallback)`。同一个概念在所有课里用同一种颜色：

| 颜色 token | 语义 |
|---|---|
| `--fig-blue` | 查询 Q、当前 token、“正在计算的东西”、激活的路由专家 |
| `--fig-violet` | 键 K、路由打分、共享专家 |
| `--fig-green` | 值 V、输出、正奖励、“好的/保留的” |
| `--fig-amber` | 缓存 / 存储 / 记忆（KV cache、状态、内存层级） |
| `--fig-orange` | 压缩后的潜在表示（$\mathbf c^{KV}$）、量化后的低精度数据 |
| `--fig-red` | 被掩掉 / 被丢弃 / 错误 / 负奖励 / 位置旋转（RoPE 分支，用细线区分） |
| `--fig-muted` / `--fig-rule` / `--fig-grid` | 未激活专家、背景结构、坐标网格 |
| `--fig-ink` | 文字、主要线条、箭头 |

浅色填充：`fill="var(--fig-blue, #1F5FA8)" fill-opacity="0.12"`。箭头用 `<marker>`，颜色同线条。
每张图有 `::: fig` 标题（“图 N.k 标题”）和说明文字；说明文字第一句说“这张图要看什么”。
改绘自论文的图，说明里写“据 XXX 论文图 k 重绘”并引用。

## 5. 贯穿全课的类比（复用，不要每课造新的）

每个类比都有：映射（逐项）→ 收益（它让什么变得显然）→ 边界（它在哪里失效，必须写出）。

1. **注意力 = 图书馆检索。** 查询 = 你的检索问题；键 = 书脊标签；值 = 书的内容；点积 = 问题和标签的匹配度；
   softmax = 按匹配度分配阅读时间；输出 = 按时间加权的读书笔记。**边界**：真检索是挑一本，注意力是把所有书
   加权混合；标签和内容都是学出来的，而且同一本书对不同问题可以贴不同标签（每个头各有 $W^K$）。
2. **KV cache = 会议纪要本。** 每个发言者（token）说完，秘书把他的“标签 + 要点”（K、V）记在本子上；后来的人
   只翻本子，不用重听录音。**边界**：纪要是逐字精确的副本，所以它省的是重复计算，不省空间，空间随发言人数线性增长。
3. **MLA = 速记。** 纪要不再逐字抄，而是记成更短的速记符号（潜在向量 $\mathbf c^{KV}$），要用时临时展开；
   “矩阵吸收”= 训练出一种能直接读速记的眼光，不必展开。**边界**：速记是学出来的线性压缩，会损失信息，
   模型必须在训练中学会用它；位置信息（RoPE）没法写进速记，只能另记一小栏。
4. **MoE = 医院。** 路由器 = 分诊台；路由专家 = 专科医生；共享专家 = 每个病人都要看的全科医生；
   细粒度切分 = 把“内科”拆成心内、消化、呼吸……，组合更灵活；负载均衡 = 别让所有病人都挤到一个科室；
   偏置 $b_i$ = 分诊台看到某科排队太长，就把它的“推荐分”暂时调低。**边界**：分诊按 token 进行，同一句话里
   的不同词会去不同科室；“专科”是训练中自发形成的，不是人为指定的。
5. **GRPO = 同题多答、组内排名。** 同一道题让学生写 $G$ 份答卷，每份按“比这组平均分高多少个标准差”给分，
   不用另请一位“预测分数的老师”（价值网络 critic）。**边界**：只有同一道题内比较才有意义；如果一组全对或全错，
   优势全为零，这道题就不提供学习信号。
6. **残差流 = 传送带。** 每层是传送带旁的工位，从带上取件、加工、把增量放回带上；恒等映射 = 没人动的件原样
   往下走。Hyper-Connections = 多条并行传送带再加上带与带之间的调度；mHC = 调度必须“守恒”（双随机矩阵：
   每条带送出去的总量和收进来的总量都是 1），所以件既不会越传越多，也不会凭空消失。**边界**：真实残差流是连续
   向量空间里的加法，“件”的比喻不表达方向和叠加。
7. **稀疏注意力 = 先查目录再精读。** lightning indexer = 便宜的目录索引，先粗略给所有页打分；top-$k$ = 只翻开
   得分最高的 $k$ 页精读。**边界**：目录本身也要逐页扫一遍（仍是 $O(L^2)$，只是非常便宜）。
8. **低精度数值 = 短尺子 + 分段放大镜。** FP8 = 刻度很少的尺子；缩放因子 = 给每一小段配一个放大镜，
   让这一段的数落在尺子最精细的区间里；细粒度量化 = 放大镜配得更密。**边界**：放大镜只能整体缩放，
   同一段里大小悬殊的数仍然会有一个被量得很粗。

## 6. 版式细节

- 中文正文用全角标点；中英文之间、中文与公式之间留一个半角空格（“维度 $d$ 的向量”）。
- 数字：大数用千分位逗号或科学计数法并注明单位（“131,072 字节 = 128 KiB”）；区分 KB/KiB、GB/GiB，
  论文用哪个就照抄并说明。
- 每页至少一张图或一个表；每个重要结论后面跟一个 `::: ex` 算例。
- 练习题：每课末尾一页“练习与答案”，题目分三类：概念题、计算题（需要动手算）、推导题。答案写完整过程。
- 引用：论文用 arXiv 号或出版信息；模型卡、GitHub 仓库、博客标 `kind: artefact`。
- 每课的研究笔记 `reference-docs/lesson-NN-doc.md` 保存所有核对过的引文、数字、勘误，以及“未能核实”清单。
