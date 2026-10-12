# M10 补充核实（写课时自行核对的事实）

> 主研究笔记是 `math-refs-doc.md`（§1.7 C73–C75、§2.10、§3.4、§4.6 N75–N80、§7 E39–E41、§8 第 15、16、22、23 条）。
> 本文件只记录**那里没有、写 M10 时另外核实**的事实，以及从 DeepSeek 课研究笔记交叉引用过来的数字。
> 核对日期 2026-09-24。引号内是不超过 15 词的原文短语，其余为中文转述。

## A. 本次新核实（一手来源）

| 编号 | 事实 | 来源与位置 | 原文短语 / 读法 |
|---|---|---|---|
| A1 | A100 数据表（JUN21）：BF16 Tensor Core 312 TFLOPS，稀疏 624*；80GB SXM 显存带宽 2,039 GB/s（80GB PCIe 1,935，40GB 1,555）；SXM 的 NVLink 600 GB/s，PCIe Gen4 64 GB/s | NVIDIA A100 datasheet（scratch 中的 `a100.pdf`，pypdf 抽文本） | “GPU Memory Bandwidth 1,555GB/s 1,935GB/s 1,555GB/s 2,039GB/s”；“* With sparsity” |
| A2 | H100 SXM5 的存储层级：GMEM（HBM）80 GiB @ 3.35 TB/s；L2 50 MiB @ 12 TB/s；SMEM 每 SM 228 KiB、全卡 31 TB/s；RMEM 每 SM 256 KiB | FlashAttention-3，arXiv:2407.08608**v2**，§2.2 表 1 | “80 GiB @ 3.35 TB/s”；“50 MiB @ 12 TB/s”；“228 KiB per SM, 31TB/s per GPU” |
| A3 | 31 TB/s 的来历：每 SM 每时钟 128 字节（引 Luo et al.）× 132 个 SM × 1830 MHz | 同上，脚注 4 | “128 bytes per clock cycle per SM” |
| A4 | H100 SXM5 的 FP16 矩阵乘峰值 989 TFLOPS，而指数等特殊函数只有 3.9 TFLOPS（16 次/SM/时钟 × 132 × 1830 MHz） | 同上，§3.1 及脚注 5 | “989 TFLOPS of FP16 matmul but only 3.9 TFLOPS” |
| A5 | A100：40–80 GB HBM，1.5–2.0 TB/s；108 个 SM 各 192 KB 片上 SRAM，带宽约 19 TB/s | FlashAttention，arXiv:2205.14135，§2.1（`arxiv.py grep` 核对） | “192KB of on-chip SRAM per each of 108” |
| A6 | Megatron-LM 张量并行：MLP 第一个 GEMM 按列切、第二个按行切，前向只需一次 all-reduce（g 算子）、反向一次（f 算子）；注意力按头切 Q/K/V、输出投影按行切；一个 Transformer 层前向两次 all-reduce、反向两次 | Megatron-LM，arXiv:1909.08053**v4**，§3 与图 3、图 4 | “only two all-reduces in the forward path and two in the backward path” |
| A7 | 标准数据并行的梯度 all-reduce 通常实现为 reduce-scatter + all-gather，各搬运 Ψ 个元素，每步共 2Ψ；ZeRO 的 $P_{os+g}$ 仍是 2Ψ，$P_{os+g+p}$ 为 3Ψ（1.5 倍） | ZeRO，arXiv:1910.02054**v3**，§7.1、§7.2（scratch 中的 `1910.02054.txt`） | “the standard DP incurs 2Ψ data movement during each training step” |
| A8 | 混合精度 Adam 每参数 16 字节：fp16 参数 2 + fp16 梯度 2 + fp32 参数、动量、方差各 4（K=12）；$P_{os}$ 大 $N_d$ 时约 4Ψ，$P_{os+g}$ 约 2Ψ，$P_{os+g+p}$ 为 16Ψ/N_d | 同上，§3.1、§5.1–§5.3 | “2Ψ+2Ψ+KΨ=16Ψ bytes of memory requirement” |
| A9 | Megatron-LM 带激活重算时，每个 Transformer 块前向 2 次、重算 2 次、反向 2 次 all-reduce，每次大小 batch×seq×hidden；all-reduce 的通信量按 2×消息大小计 | 同上，§8（ZeRO-R 的通信分析） | “communication volume of an all-reduce is 2×message_size” |
| A10 | GPipe 的气泡：每个加速器的空闲时间为 $O(\frac{K-1}{M+K-1})$（$K$ 段、$M$ 个 micro-batch）；实验中 $M\ge4K$ 时气泡开销可以忽略 | GPipe，arXiv:1811.06965**v5**，§2.3 | “negligible when M ≥ 4 × K” |
| A11 | MFU 的定义：观测吞吐（token/s）与“系统以峰值 FLOPs 运行时的理论最大吞吐”之比；理论最大值只计前向+反向所需运算、不计重算；每 token $6N+12LHQT$；PaLM 540B 在 6144 块 TPU v4 上 MFU 46.2%（不计注意力 45.7%），HFU 57.8% | PaLM，arXiv:2204.02311**v5**，§4.1、附录 B | “and not rematerialization” |
| A13 | H800 SXM5：显存带宽峰值 3.35 TB/s，峰值 990 TFLOPS；降频到约 1600 MHz 时实际峰值约 865 TFLOPS；新 FlashMLA 内核在 H800 上达到降频后峰值的约 80% 与 3 TB/s 带宽 | FlashMLA 仓库 `docs/20250422-new-kernel-deep-dive.md`（DeepSeek，2025-04-22；curl 原始 Markdown，第 13、57 行） | “peak memory bandwidth of 3.35 TB/s and peak FLOPs of 990 TFlops”。这是 DeepSeek 的说法，**不是** NVIDIA 规格表；与 lesson-09-doc 的引文一致 |
| A14 | ZeRO 全称 Zero Redundancy Optimizer；$P_{os+g}$ 用梯度的 reduce-scatter（Ψ）代替 all-reduce，各进程更新自己那块参数后再 all-gather 参数（Ψ），合计 2Ψ，与标准 DP 相同；$P_{os+g+p}$ 前向时边算边收集各层参数（Ψ），反向按相反顺序再收集一次（Ψ），加上梯度 reduce-scatter，共 3Ψ | ZeRO，arXiv:1910.02054，摘要、§7.2.1、§7.2.2（2026-09-24 用 `arxiv.py grep` 复核） | “an all-gather is performed to collect all the updated parameters”；“this all-gather needs to happen once again for the backward propagation” |
| A15 | 递归减半–倍增 all-reduce = reduce-scatter + allgather；reduce-scatter 第 1 步两两配对（0 与 1、2 与 3），互发各自缓冲区的一半并归约；之后每步目标距离加倍、收发数据减半 | Goyal et al. 2017，arXiv:1706.02677，§4.1（2026-09-24 用 `arxiv.py grep` 核对） | “the distance to the destination rank is doubled while the data sent and received is halved” |
| A16 | GPipe 把“$M\ge4K$ 时气泡可以忽略”部分归因于反向重算可以提前调度（按公式 $m=4P$ 时气泡仍约 $\frac{P-1}{5P-1}\approx20\%$，所以这是实验观察而非公式推论） | GPipe，arXiv:1811.06965，§2.3（2026-09-24 数学审校时用 `arxiv.py grep` 核对） | “partly because re-computation during the backward pass can be scheduled earlier” |
| A17 | Goyal 等的两句话是分开的：减半倍增“for buffer sizes up to a million elements”明显优于环（服务器多时更高）；另一句是 32 台服务器（256 GPU）上比环快 3 倍。正文不要把 3 倍写成“一百万元素以内”的条件结论 | Goyal et al. 2017，arXiv:1706.02677，§4.1（同上核对） | “On 32 servers (256 GPUs), using halving/doubling led to a speedup of 3×” |
| A18 | V3 主模型里没有 $7168\times7168$ 的方阵：MLA 各投影为 $7168\times1536$、$1536\times24576$、$7168\times576$、$512\times32768$、$16384\times7168$（输出投影），稠密 FFN $7168\times18432$，专家 $7168\times2048$。本课的“$7168\times7168$”只是取隐藏维的方阵示例 | 【交叉引用】lesson-12-doc §4.3 的参数分解（本课用 Python 按这些形状重算得每层 187,105,280） | 笔者核对 |
| A12 | Thakur 的 all-to-all 记号：$n$ 是“一个进程要发给（或收自）所有其他进程的数据总量”；长消息成对交换 $T_{long}=(p-1)\alpha+n\beta$；短消息 Bruck 算法 $\lg p\,\alpha+\frac n2\lg p\,\beta$ | Thakur et al. 2005，§4.3（scratch 中的 `thakur.txt`） | “the total amount of data a process needs to send” |

## B. 交叉引用（数字来自 DeepSeek 课研究笔记，本次未重核原文）

| 编号 | 事实 | 出处 |
|---|---|---|
| B1 | H100 SXM：官网只印稀疏 BF16 1,979；稠密 989 取自 FA3；80GB、3.35 TB/s、NVLink 900 GB/s | lesson-04-doc C73、§3.2 |
| B2 | H800：FP64 与 NVLink 被降，NVLink 900 → 400 GB/s；每节点 8 张 400G CX7 | lesson-04-doc C74；lesson-12-doc I7 |
| B3 | V3 报告：NVLink 160 GB/s，约为 IB 50 GB/s 的 3.2 倍；Insights：名义 200、可达约 160，IB 按有效 40 | lesson-12-doc C24、C30 |
| B4 | V3 集群 2048 张 H800，每节点 8 卡；训练 PP16、EP64（跨 8 节点）、ZeRO-1 DP、不用 TP；H800 132 个 SM，通信用 20 个 | lesson-12-doc C7、C10、C29、§3.2 |
| B5 | Insights 表 4（MPFT）：每天 272.80B token，每步 19.926 s；1F 1.13、bubble 2.06、1B 1.99、1W 0.48、1F1B 13.95、opt 0.29 s；MFU（causal）38.94%，按 BF16 峰值；由此反推 BF16 稠密峰值约 988 TFLOPS | lesson-12-doc I12、§4.2、§4.3 |
| B6 | V3 每 token 训练 FLOPs（按 config 重建）：矩阵部分 $6\times36.625$B = 219.75 GFLOPs，注意力（causal，平均上下文 2048）30.70，合计 250.45（Insights 表 2 印 250） | lesson-12-doc §4.3 |
| B7 | V3 每层注意力参数 187,105,280；稠密 FFN 396,361,728；单个专家 44,040,192；门控 1,835,008；嵌入 = 输出头 = 926,679,040；61 层，前 3 层稠密，58 层 MoE（8 路由 + 1 共享） | lesson-12-doc §4.3 |
| B8 | V3 decode 阶段每个专家的 batch 通常不超过 256 个 token，瓶颈在访存 | lesson-04-doc C27 |
| B9 | Insights §2.3.2 的通信上界：$(1+2)\,\text{B}\times32\times9\times7\text{K}/50\,\text{GB/s}=120.96\,\mu$s；每层 ×2（双 micro-batch 重叠）；61 层 14.76 ms；900 GB/s 时 6.72 µs、0.82 ms | lesson-12-doc §2.8 |
| B10 | V3 MLA 维度：$d_c=512$，$d_h^R=64$，$n_h=128$，KV cache $(512+64)\times61\times2=70{,}272$ 字节/token | lesson-04-doc §4.3；lesson-09-doc |
| B11 | 1F1B 气泡 $(P-1)(T_F+T_B)$，占比 $(P-1)/(m+P-1)$；V3 表 2 | lesson-12-doc §2.1、§4.6；第 12 课 (12.15) |
| B12 | DeepEP V1 实测（公司自测）：节点内 EP8 153/158 GB/s，跨节点 EP16–64 为 43–58 GB/s | lesson-12-doc A2 |

## C. 本课自己的推导与估算（写课时标注“笔者推导 / 笔者估算”）

- **GEMM 强度的调和平均形式**：$I=\frac{2mnk}{s(mk+kn+mn)}=\frac{2}{s}\big(\frac1m+\frac1n+\frac1k\big)^{-1}$，并有 $\frac{2}{3s}\min(m,n,k)\le I\le\frac2s\min(m,n,k)$（代数，可直接验证）。
- **roofline 由时间下界推出**，以及“不重叠时最多慢 2 倍”：$\max(a,b)\le a+b\le2\max(a,b)$。
- **MLA 解码注意力的强度**（V3 维度，吸收后的 MQA 模式，BF16 缓存）：每个缓存 token、每个头打分 $2\times576$、加权求和 $2\times512$ 次运算，128 个头共 278,528 次；读 $576\times2=1152$ 字节；$I\approx241.8$ FLOP/B。未计入吸收后的投影、softmax 与 RoPE 的额外开销。
- **张量并行的“通信强度”**：一个 $12h^2$ 参数的稠密层，每 token 训练运算 $72h^2/p$（每卡），通信 $4\times\frac{2(p-1)}{p}\times2h$ 字节（前向 2 次、反向 2 次 all-reduce，BF16，环形），比值 $\frac{4.5h}{p-1}$ FLOP/字节；$h=7168$、$p=8$ 时 4608。与链路的“脊点” $F/\mathrm{BW}$ 比较：989 TFLOP/s ÷ 160 GB/s = 6181，÷ 50 GB/s = 19,780。只是量级论证，不计重叠与注意力打分。
- **各级存储的“脊点”**（取 989 TFLOP/s）：SMEM 31 TB/s → 31.9；L2 12 TB/s → 82.4；HBM 3.35 TB/s → 295.2；NVLink 160 GB/s → 6181；IB 50 GB/s → 19,780 FLOP/字节。
- **V3 专家矩阵的强度**（BF16，$k=7168$，$n=2048$）：$m=1$ 时 1.0；$m=64$ 时 61.5；$m=128$ 时 118.5；$m=256$ 时 220.6；$m=4096$ 时 1146.9；上限 1592.9。$7168\times7168$：$m=64$ 为 62.9，$m=4096$ 为 1911.5，上限 3584。越过 H800 脊点 295.5 的最小 $m$：投影 323、专家 363（`m10_intensity.py` 输出）。
- **专家并行的通信强度**：一层 MoE 前向每 token 计算 $2\times(9\times44.04+187.1+1.8)\times10^6\approx1.17\times10^9$；节点受限时跨 IB 至多 $4\times(7392+14{,}336)=86{,}912$ 字节（N80 的示意格式）；比值约 13,500，低于 IB 链路脊点 19,800。与 V3 报告的“计算与通信约 1:1”（lesson-12-doc C13）同一量级。
- **V3 一步的满速时间**：$62{,}914{,}560\times250.45\times10^9/2048/990\times10^{12}=7.77$ s；实测 19.926 s；比值 39.0%。
- **20/132 个 SM 做通信的影响**（笔者推断，假设矩阵乘吞吐与 SM 数成正比）：计算可用 84.8% 的 SM。
- **张量并行在 H100 上的对照**（笔者推算，假设可达 NVLink 带宽与名义值同比例 900/400 = 2.25 倍）：链路脊点约 6190/2.25 ≈ 2750 < 4608。
- **数据并行隐藏通信的条件**：$1.5b\ge F/\mathrm{BW}$，990 TFLOP/s、50 GB/s 时 $b\ge13{,}200$；V3 每卡每步 $62{,}914{,}560/2048=30{,}720$ 个 token。
- **（教学审校补充，2026-09-24）重叠的时间线**（图 M10.6，`m10_overlap.py`）：MLA 解码例题 $a=5.567\times10^{11}/990\times10^{12}=0.562$ ms（计算），$b=2{,}302{,}672{,}896/3.35\times10^{12}=0.687$ ms（搬运；正文原写 0.688，按精确字节数改为 0.687）；不重叠 1.250 ms；切 8 块双缓冲 $b+a/8=0.758$ ms。8 块只是示意。
- **半带宽消息大小**（§2 小练习）：$S/(\alpha+S\beta)=\mathrm{BW}/2\iff S_{1/2}=\alpha\cdot\mathrm{BW}$；把 Insights 实测的 64 字节端到端延迟当作 $\alpha$：IB $2.8\,\mu$s × 50 GB/s = 140 KB，NVLink $3.33\,\mu$s × 160 GB/s ≈ 533 KB；7392 字节单独发时有效带宽 $7392/(2.8+0.148)\,\mu\text{s}\approx2.5$ GB/s（5%）。只是 α–β 模型的推算。
- **路由专家的参数占比**（§5 专家并行）：$256\times44{,}040{,}192=1.127\times10^{10}$ 每层，×58 层 = 653.9B，约占 671B 的 97%（按 B7 的配置计数；671B 的其余部分：注意力 11.41B、稠密 FFN 1.19B、共享专家 2.55B、门控 0.11B、嵌入与输出头 1.85B，合计 671.03B，与报告的 671B 一致；653.9/671.0 = 97.4%）。
- **（数学审校补充，2026-09-24）MLA 解码强度约为 $2n_h$ 的来历**：吸收后 512 维压缩向量既当键又当值，每个头用两次，64 维 RoPE 键每头用一次；每个缓存数平均被用 $(512\times2\times128+64\times128)/576=241.8$ 次，BF16 下“使用次数”恰等于 FLOP/字节。所以 GQA 公式 $n_h/n_g$ 在 $n_g=1$ 时给的 128 不适用于 MLA。
- **（数学审校补充，2026-09-24）数据并行的 $1.5b$ 只对稠密模型成立**：MoE 若每卡放完整 671B 模型，比值为 $1.5b\times37/671\approx2500$（$b=30{,}720$），低于 IB 脊点；V3 实际是 PP16 × EP64，$2048/16=128$ 路数据并行，每卡每步流过自己那段层的 token $62{,}914{,}560/128=491{,}520$，稠密部分比值 $\approx74$ 万；每个路由专家在全集群 $2048/(16\times64)=2$ 份，只在 2 份间同步梯度。正文原写“纯数据并行、每卡一份完整模型的保守估计”，对 MoE 不成立，已改。
- **（数学审校补充，2026-09-24）朴素 all-reduce 的对照**：$p=8$ 时根卡先收 $7S$ 再发 $7S$，时间 $14S\beta$；环形每卡同时收发各 $1.75S$，时间 $1.75S\beta$，快 8 倍（正文原把根卡的“收发合计”与环的“只发”相比）。
- **§7 的 MFU 天花板**（笔者估算）：$(1-0.103)\times(112/132)\approx0.76$，把气泡与让给通信的 SM 当作相互独立的折扣连乘，只是量级论证。

## D. 仍未核实

- H800 的 HBM 容量、稠密 BF16/FP8 峰值的 NVIDIA 原始数据（页面不可得，见 lesson-12-doc §8 第 1 条）。正文用 DeepSeek FlashMLA 博客的 3.35 TB/s 与 990 TFLOPS（A13），并与“由 MFU 反推约 988 TFLOPS”互相印证；图中注明来源是 DeepSeek 的博客。
- NVLink 的 900 / 400 / 200 / 160 GB/s 各是单向还是双向合计（lesson-12-doc §8 第 2 条）。正文的计算一律用 V3 报告给的可达值 160 GB/s。
- 环形 all-reduce 的最早出处（math-refs-doc §8 第 15 条）。

## E. 补充核实（事实审校，2026-09-25）

> 本轮逐条核对正文的数字、归属与引用；下面只记**本轮直接回一手来源重查过**的条目（B 表里原为交叉引用、这次重核过的也记在这里）。
> arXiv 用 `arxiv.py grep`；Thakur 用 ANL 原址 PDF（`web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf`，本轮可以访问）加 pypdf 抽文本；FlashMLA 博客与 V3 `config.json` 用 curl 抓原文件。

| 编号 | 事实 | 来源与位置 | 原文短语 / 读法 |
|---|---|---|---|
| E1 | all-to-all 长消息成对交换：$p$ 为 2 的幂时第 $k$ 步（$1\le k<p$）与 `rank ^ k`（异或）直接交换；非 2 的幂时第 $k$ 步收自 rank−k、发往 rank+k；两者 $T_{long}=(p-1)\alpha+n\beta$。Bruck 的 $\lg p\,\alpha+\frac n2\lg p\,\beta$ 是 2 的幂时的式子 | `thakur2005` §4.3 | “(exclusive-or operation) and exchanges data directly”。正文图 M10.11 的说明原来没写“2 的幂”这个条件，已补 |
| E2 | Thakur §3 的模型假设（与通信对数、距离无关；链路双向；网卡单端口）；§4.5 二项树归约 $T_{tree}=\lceil\lg p\rceil(\alpha+n\beta+n\gamma)$；§5.3 “Ring Algorithm” 的 reduce-scatter 用成对交换、allgather 走环 | `thakur2005` §3、§4.5、§5.3 | “single ported”；“uses a pairwise-exchange algorithm for the reduce-scatter phase” |
| E3 | V3 报告：每万亿 token 180K H800 GPU 小时、2048 卡 3.7 天、预训练 2664K；PP16、跨 8 节点的 EP64、ZeRO-1 DP；每层路由专家均匀放在 8 个节点的 64 张卡上；batch 在前 469B token 内从 3072 增到 15360、之后保持 15360；预训练最大序列长 4K；decode 时每个专家的 batch “usually within 256 tokens”；20/132 个 SM 做通信；NVLink 160 GB/s ≈ IB 50 GB/s 的 3.2 倍；每 token 至多 4 个节点；计算通信比约 1:1；dispatch 前量化为 FP8、combine 保持 BF16；PD 分离的理由是同时满足 SLO 与高吞吐 | `v3` arXiv:2412.19437，§1、§3.2、§3.2.2、§3.3.3、§3.4、§3.5.1、§4.2 | “usually within 256 tokens”；“keeps 15360 in the remaining training”；“Service-Level Objective (SLO) for online services and high throughput” |
| E4 | Insights §2.3.2：每张卡放一个专家、一次约处理 32 个 token；120.96 µs 是 EP 中“两次 all-to-all”的合计；“Network latency is not included”；14.76 ms ≈ 67 token/s；900 GB/s 时 0.82 ms ≈ 1200 token/s。§4.1：H800 NVLink 900 → 400 GB/s；§4.2：训练不用 TP。表 5 是 “CPU side end-to-end latency” 64B：IB 同 leaf 2.8 µs、跨 leaf 3.7 µs，NVLink 3.33 µs。ISCA ’25 的 DOI 10.1145/3695053.3731412（arXiv abs 页元数据） | `insights` arXiv:2505.09343 | “each device holds one expert’s parameters”；“Tensor Parallelism is avoided during training” |
| E5 | FA3 表 1 的 RMEM 行只写 “256 KiB per SM”，没有带宽；31 TB/s 用的 1830 MHz 是 boost clock；989 TFLOPS 也按 1830 MHz 计 | `fa3` arXiv:2407.08608，§2.2 表 1、脚注 4、附录基准设置 | “the boost clock of 1830 MHz” |
| E6 | FlashMLA 博客：计算访存比 $\approx2h_qs_q$；H800 SXM5 3.35 TB/s、990 TFLOPS，降频约 1600 MHz 时约 865；新内核 “up to 80% Tensor Core utilization (of the throttled theoretical peak) and 3 TB/s”（两个数分别是受算力限制与受带宽限制的设置）；K 块 $64\times576$、输出 $64\times512$ | `flashmla-blog` 第 3、11、13、54、57 行 | 标题 “A Deep-Dive Into the New Flash MLA Kernel” 与参考卡一致 |
| E7 | V3 `config.json`（重读）：hidden 7168、61 层、128 头、q_lora 1536、kv_lora 512、nope 128、rope 64、v 128、vocab 129280、intermediate 18432、moe_intermediate 2048、256 路由 + 1 共享、每 token 8 个、first_k_dense_replace 3、tie_word_embeddings false。重算：每层注意力 187,105,280；$N_{act}=36{,}624{,}596{,}992$（加嵌入 37.55B）；总参数 671.03B；路由专家 653.9B（97.4%）；全部参数参与时每 token 4051.3 GFLOPs（16.18 倍）；一层（注意力 + 9 专家 + 门控）585,302,016 参数 $=11.39d^2$，$I_{\text{TP}}=4374$；EP 前向 1.1706 GFLOPs，÷86,912 = 13,469 | huggingface.co/deepseek-ai/DeepSeek-V3 `config.json`（scratch 中 `recompute.py`） | 与 B6、B7 一致 |
| E8 | Goyal 的减半–倍增：第 1 步 0 与 1、2 与 3 配对，“rank 0 sends the second half of its buffer to 1 and receives the first half” | `goyal2017` §4.1 | 支持正文“0 号发后一半、收前一半” |
| E9 | FlashAttention 用 “arithmetic intensity” 一词，并引 Roofline 为出处 | `flashattention` arXiv:2205.14135 §2.1、§5 | “commonly measured by the arithmetic intensity [85]” |

**本轮改动（正文）**：§2 “这一页用在哪里”原把第 14 课写成“KV cache 在 HBM、主机内存与 SSD 之间的取舍”，第 14 课讲的是稀疏读取，已改，并给第 19 课补了链接；§4 “这一页用在哪里”原写“所以 V3 把两个阶段分开部署”，报告给的理由是 SLO 与吞吐（E3），已改成报告理由 + 笔者解读；§5 “V3 报告等工程文献叫 ZeRO-1、2、3”改为只把 “ZeRO-1” 归给 V3 报告；§5 的 62,914,560 补了出处（E3 的 batch 15360 × 4K）；“实测气泡 10.3%”补上“表 4 标为 bubble 的一行、论文未另行定义”；图 M10.11 与 (M10.20) 补“$p$ 为 2 的幂”（E1）；Insights 的解码上界例题补“不计网络延迟”“每卡一个专家”“两次 all-to-all 合计”（E4）；参考卡 `v3-report` 的 matters 删去正文没有引它的“流水线气泡公式”，`insights-isca25` 补 DOI。
