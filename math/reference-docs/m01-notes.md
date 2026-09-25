# M01 补充核实（向量、矩阵乘法与形状思维）

M01 试写作者在写课时逐条对照原文核实过以下事实（2026-09-24）。它们不在共享的 `math-refs-doc.md` 里，记录在此供审阅者复核。

- **Vaswani et al., arXiv:1706.03762v7, §3.2.1–3.2.2**：缩放点积注意力公式；点积注意力与加性注意力的比较；
  脚注中“点积方差为 $d_k$”的论证（该脚注在 v1 中不存在，见 mamba/mamba-reference-docs/lesson-01-doc.md 勘误 12）；
  多头投影的形状 $W_i^Q\in\R^{d_{model}\times d_k}$、$W^O\in\R^{hd_v\times d_{model}}$，基础模型 $h=8$、$d_k=d_v=64$。
- **DeepSeek-V2, arXiv:2405.04434v5, §2.1.1–2.1.3**：MLA 方程与形状；矩阵吸收的原句；RoPE 与低秩 KV 压缩不兼容（矩阵乘法不交换）的原句。
  **§3.1.2**：$n_h=d_h=128$，$d_c=512$，$d_c'=1536$，$d_h^R=64$，60 层，$d=5120$。
- **GPT-2 small（Hugging Face `openai-community/gpt2` config.json）**：`n_embd` 768，`n_layer` 12，`n_head` 12，`n_positions` 1024，`vocab_size` 50257。
- **PyTorch `nn.Linear` 文档**：$y = xA^\top + b$，权重按 (out_features, in_features) 存放。
- **Hugging Face `Conv1D`（transformers/pytorch_utils.py）**：权重按 (in, out) 存放；GPT-2 的 `c_attn` 一次算出 QKV 后 `split`，再 `view` + `transpose` 拆成多头。
- **NumPy `einsum` 文档**：下标约定（输出中缺席的下标被求和）。
- 标注为“推测”而非事实的一句：一次大矩阵乘法比三次小矩阵乘法更能用满硬件（未给出测量）。

## 补充核实（Teach 审阅，2026-09-24）

- **§6 浮点结合律的数值**（笔者实测，NumPy 1.26，float64，`default_rng(0..2)`）：$d_h=128$、$d_c=512$、$T=4096$，
  $W\sim N(0,1)/\sqrt{d_c}$，$\mathbf q$、$C$ 为标准正态；`(C @ W.T) @ q` 与 `C @ (W.T @ q)` 的最大绝对差
  $3.1\times10^{-14}$–$3.9\times10^{-14}$，分数最大绝对值约 38–44，相对差 $\approx8\times10^{-16}$–$9\times10^{-16}$。
  正文原写“差别在 $10^{-13}$ 量级”，改为“最多差约 $4\times10^{-14}$、相对差约 $10^{-15}$”。
- **练习 11**：研究笔记 §6 W1 / N48（`2*T*T*dh*nh`，$T=4096$、$n_h=d_h=128$）得 $5.50\times10^{11}$；复算
  $549{,}755{,}813{,}888$。$4T^2d$（$d=5120$）$=343{,}597{,}383{,}680$；$n_hd_h/d=16384/5120=3.2$。$d=5120$ 与
  $d_h^R=64$ 见上条 DeepSeek-V2 §3.1.2。
- **练习 12（GQA）**：无外部事实，配置是假设的；`np.einsum('grte,gse->grts', Q.reshape(n_g,r,T,d_h), K)` 与逐头
  `Q[h] @ K[h//r].T` 一致（最大差 $2\times10^{-14}$），`einsum_path` 报告的朴素 FLOPs $=2\cdot32\cdot T^2\cdot128$。
- **图 M1.7（两个小头）**：$\mathrm{softmax}(2,0)=(0.8808,0.1192)$，$\mathrm{softmax}(2,2)=(0.5,0.5)$，由
  `figures/gen/m01_two_heads.py` 计算。

## 补充核实（Math 审阅，2026-09-24）

- 全课每个算例、表格、练习答案与图中数字都用 NumPy 重算过一遍（float64），全部一致；十个图脚本重跑后与仓库里的 SVG
  逐字节相同（改动前）。§3 的 einsum 代码在小尺寸上与“大矩阵 $W^Q$ 按行切头”“拼接后乘 $W^O$”逐元素一致。
- 吸收账单的极限：$\text{Ⓐ}/\text{Ⓑ}=(2Td_cd_h+2Td_h)/(2d_hd_c+2Td_c)\to d_h(1+1/d_c)=128.25$（精确值，正文原写
  “$\approx128.3$”、图里写“$\approx128$”，统一为 $128.25$）。练习 6 的比值恰为 $2064=2^{11}+2^4$。
- 浮点结合律实验复现（`default_rng(0..2)`，同上条设置）：最大绝对差 $3.1$–$3.9\times10^{-14}$，分数最大绝对值 37.6–44.1，
  相对差 $8.3$–$8.9\times10^{-16}$。正文补写了 $W^{UK}_i$ 按 $1/\sqrt{d_c}$ 缩放这一设置（不缩放时分数是几百的量级）。
- 记号：§2 的两个示例矩阵由 $P,Q$ 改名 $F,G$、拉伸矩阵由 $S$ 改名 $D$，以免与同页的注意力权重 $P$、查询 $Q$、打分表 $S$
  撞名；练习 12 的组内序号由 $j$（全课的键位置下标）改为 $a$；图 M1.5 的输入由 $X$ 改为与正文表格一致的 $H$。

## 补充核实（Facts 审阅，2026-09-24）

逐条回原始来源重查（arXiv 用 `arxiv.py grep`，先在页头确认版本；制品用 curl 抓 raw 文件或 WebFetch 读文档页）：

- **Vaswani et al.（页头 “arXiv:1706.03762v7 … 02 Aug 2023”）§3.2.1**：加性注意力用只有单个隐藏层的前馈网络打分
  （正文原写“带隐藏层的小前馈网络”，已改准）；两者理论复杂度相近，点积注意力因能用优化过的矩阵乘法而实践中更快、更省内存；
  大 $d_k$ 下 softmax 梯度极小是论文的推测（原文以 “We suspect” 起句），脚注的方差论证只是说明。正文原把“梯度消失所以除以
  $\sqrt{d_k}$”写成定论，已补写“证明 / 推测”的区分；参考卡补“读的是 v7”与“方差脚注 v2 起才有”（依据
  mamba/mamba-reference-docs/lesson-01-doc.md 勘误 12）。§3.2.2：因每头维度缩小，多头总代价与全维单头相近（正文 §4 转述无误）。
- **DeepSeek-V2（页头 “arXiv:2405.04434v5 … 19 Jun 2024”）**：§2.1.1 $W^Q,W^K,W^V\in\R^{d_hn_h\times d}$；§2.1.2 $W^{DKV}\in\R^{d_c\times d}$、
  $W^{UK},W^{UV}\in\R^{d_hn_h\times d_c}$、吸收原句；§2.1.3：与当前生成 token 相关的 RoPE 矩阵夹在 $W^Q$ 与 $W^{UK}$ 之间、
  矩阵乘法不可交换，故不能吸收；解耦的 RoPE 键也要缓存，总缓存 $(d_c+d_h^R)l$ 个元素。据此在 §6“问题”段补写：每 token
  还要缓存 $d_h^R=64$ 维共享 RoPE 键，合计 576 个数/层。
  §3.1.2 的 60 层、$d=5120$、$n_h=d_h=128$、$d_c=512$、$d_c'=1536$、$d_h^R=64$ 与上文一致。
- **GPT-2 config.json**（`huggingface.co/openai-community/gpt2/raw/main/config.json`，2026-09-24 重新下载）：`n_embd` 768、
  `n_head` 12、`n_layer` 12、`n_positions` 1024、`n_ctx` 1024、`vocab_size` 50257。
- **Hugging Face transformers main 分支**（raw 文件，2026-09-24）：`pytorch_utils.py` 的 `Conv1D.__init__(self, nf, nx)`，
  `self.weight = nn.Parameter(torch.empty(nx, nf))`，前向 `torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)`，
  docstring “Basically works like a linear layer but the weights are transposed.”；`modeling_gpt2.py` 的
  `self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)`（权重 $[768,2304]$）、`self.c_proj = Conv1D(self.embed_dim, self.embed_dim)`，
  拆头 `shape_q = (*query_states.shape[:-1], -1, self.head_dim)`、`query_states.view(shape_q).transpose(1, 2)`。
- **PyTorch `nn.Linear`**：`docs.pytorch.org/docs/stable/…` 重定向到 `/docs/2.14/…`，参考卡的“文档 2.14 版”无误；
  “$y=xA^T+b$”、权重 “(out_features, in_features)”、输入 $(*,H_{in})$ / 输出 $(*,H_{out})$。
- **NumPy `einsum`**（文档页为 v2.5）：显式模式用 `->` 指定输出；示例含 `'ij,jk->ik'`、`'ij->ji'`、`'i,i'`、`'i,j'`。
- **正文全部算例、表格、图中数字与 12 道练习**用 Python 3.11 + NumPy 独立重算（`facts-m01-vectors-matrices/recompute.py`，
  scratch 目录），全部一致；§3 的 einsum 代码按原尺寸运行，输出形状与注释一致，`U` 与“拼接后乘 $W^O$”相对差 $1.4\times10^{-15}$。
- **标注调整**：§4“一次大乘法比三次小乘法更能用满硬件”由“经验上的常识”改标“笔者推测（无测量）”；§5“业界统一用 $2mnk$”改为
  “通常用”，并注明第 1 课的 $N$ 是不含嵌入表的参数量（第 1 课取 $N\approx12Ld^2$）。

## 补充核实（Math 复审，2026-09-24）

- 再次独立重算全部算例、表格、练习与图中数字（NumPy，float64），并逐字节重跑十个图脚本：全部一致，SVG 无变化。
  §3 的 einsum 代码按原尺寸运行，与“大矩阵按行切头、拼接后乘 $W^O$”相对差 $1.2\times10^{-15}$；练习 12 的 GQA einsum 与逐头
  计算一致；§4 注释的 $\lceil i\,n_g/n_h\rceil$ 与练习 12 的 $\lfloor h/r\rfloor$ 对 $n_h=32,n_g=8$ 的每个头逐一核对一致。
- **预乘的条件**（纯推导，无外部事实）：$A\in\R^{m\times k}$、$B\in\R^{k\times n}$ 合并后每个向量 $2mn$ 次，分两步 $2k(m+n)$ 次。
  正文原把“固定乘积提前算好”直接等同于吸收并说“加载模型时算一次就够”，与 M02 §7（三项 MLA 固定乘积都应分开乘）冲突。
  补写条件 $mn<k(m+n)$；§6 查询侧（$[d_c,d_h]\times[d_h,d]$，$d=5120$）：合并 $2d_cd=5{,}242{,}880$，分开
  $2d_hd+2d_cd_h=1{,}441{,}792$，比值 3.64；练习 9 输出侧（$[d,d_h]\times[d_h,d_c]$）数值相同。本页要点同步改写。
- **记号**：练习 10 的投影矩阵 $P$ 改为 $\Pi$（§2 已声明 $P$ 留给注意力权重，练习 9 紧挨着用 $P^{(i)}_{tj}$）；§6 的坐标向量
  $\mathbf c$ 改为 $\boldsymbol\alpha$（同页 $\mathbf c_j$ 是 MLA 的缓存向量；$\boldsymbol\alpha$ 正好是直线情形 $\alpha$ 的 $k$ 维版本）；
  §6 开头注明 $\mathbf u,\mathbf u_1,\dots,U$ 是方向，与注意力层输出 $\mathbf u_t$ 无关；记号框补“$d_k$ 就是 $d_h$”；
  封面与 §1 的“从其余 token / 前面的 token 取信息”补上“包括它自己”（因果掩码下 $P_{tt}$ 一般非零）；§3 注释中 KV cache 的
  形状改说成“所有头的 $K_i$ 叠起来是 $[n_h,T,d_h]$”。

## 补充核实（Facts 复审，2026-09-24）

对 Math 复审之后的正文再做一遍事实与引用核对（scratch 目录 `facts-m01-vectors-matrices/recompute2.py`）：

- **原文重查**：`arxiv.py grep 1706.03762`（v7 LaTeXML）：§3.2.1 “a feed-forward network with a single hidden layer”、
  “similar in theoretical complexity … much faster and more space-efficient in practice”、“We suspect that for large values of
  $d_k$ …”、脚注 “mean 0 and variance $d_k$”、“To counteract this effect, we scale the dot products by $1/\sqrt{d_k}$”；§3.2.2
  “$d_k=d_v=d_{model}/h=64$”“the total computational cost is similar to that of single-head attention with full dimensionality”。
  `arxiv.py grep 2405.04434`（页头 “arXiv:2405.04434v5 … 19 Jun 2024”）：$W^{DKV}\in\R^{d_c\times d}$、$W^{UK},W^{UV}\in\R^{d_hn_h\times d_c}$、
  吸收原句、$W^{DQ}\in\R^{d_c'\times d}$ 与 $W^{UQ}\in\R^{d_hn_h\times d_c'}$、RoPE 原句（“a RoPE matrix related to the currently generating
  token will lie between $W^Q$ and $W^{UK}$ and matrix multiplication does not obey a commutative law”）、§3.1.2 的 60 层、5120、
  $n_h=d_h=128$、$d_c=512$、$d_c'=1536$、$d_h^R=64$。GPT-2 `config.json`、`pytorch_utils.py`（`Conv1D` 权重 `torch.empty(nx, nf)`、
  docstring）、`modeling_gpt2.py`（`c_attn`/`c_proj`、`shape_q` + `transpose(1, 2)`）、`nn.Linear` 2.14 文档页，均于本日重新抓取，与正文一致。
- **更正上一条记录**：NumPy `einsum` 文档（v2.5）的例子里**没有**字面的 `'ij,jk->ik'`，只有隐式的 `'ij,jk'` 与显式的
  `'ij,jh->ih'`（另有 `'ij->ji'`、`'i,i'`、`'i,j'`、`'i->'`）。参考卡已改成文档里实际出现的写法；正文“只有两条规则”改为
  “本课用到的只有两条规则（显式写出 `->` 的写法）”，因为文档另有“同一操作数里重复的下标取对角线”、省略号广播等规则。
- **补写 V2 真实的查询路径**（§6）：V2 的查询由 $d_c'=1536$ 维的 $\mathbf c^Q_t$ 经 $W^{UQ}$ 展开。合并
  $(W^{UK}_i)^\top W^{UQ}_i$（$[d_c,d_h]\times[d_h,d_c']$）后每头每 token $2d_cd_c'=1{,}572{,}864$ 次，分开乘
  $2d_hd_c'+2d_cd_h=393{,}216+131{,}072=524{,}288$ 次，比值 3.0；条件 $mn<k(m+n)$ 为 $786{,}432<262{,}144$，不成立，结论与正文
  用 $d=5120$ 的示意相同。参考卡补上 $W^{DQ}$、$W^{UQ}$ 与 $d_c'=1536$。
- **参考卡**：`vaswani2017` 补出版信息 NIPS 2017（与第 1 课、M04 的卡一致）。
- **全部算例、图中数字、12 道练习与 5 个小练习**独立重算，全部一致（含练习 6 的 2064、练习 11 的 $3.2$、练习 12 的 GQA einsum
  与逐头计算差 $3.6\times10^{-15}$、§4 $\lceil i\,n_g/n_h\rceil$ 与 $\lfloor h/r\rfloor$ 对 32 个头逐一一致）。浮点结合律实验换一种抽样顺序
  （先 $\mathbf q$、再 $C$、再 $W$，`default_rng(0..2)`）得最大绝对差 $2.9$–$3.2\times10^{-14}$、分数最大绝对值 36–41、相对差
  $7.4$–$8.3\times10^{-16}$，与正文“最多差约 $4\times10^{-14}$、相对差约 $10^{-15}$”相符。十个图脚本重跑，SVG 逐字节不变。
- **跨课链接**：`check_links.py` 0 个坏链；逐个核对了目标页的标题（第 1 课 §3/§4/§7/§8，第 4 课 §2/§3/§5，第 9 课 §2/§3，
  M02 §1/§2/§3/§7，M04 §3/§4，M06 §2，M08 §5，M10 §3/§4），内容都对得上。
