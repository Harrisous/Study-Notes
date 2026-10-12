# M03 补充核实（正交矩阵、旋转与复数）

M03 作者在写课时（2026-09-24）自己核实或计算的事实。它们不在共享的 `math-refs-doc.md` 里，记录在此供审阅者复核。
`math-refs-doc.md` §2.11 说明欧拉公式、旋转复合等是教科书结论、不需要文献；本课正文里这类结论都给了完整推导，没有引用文献。

## 1. 一手来源（原文短引，均不超过 15 个英文词）

| # | 事实 | 来源（实际读取的） | 原文短引 / 字段 |
|---|---|---|---|
| S1 | RoFormer 的块对角旋转矩阵，频率 $\Theta=\{\theta_i=10000^{-2(i-1)/d}\}$（1 起下标） | arXiv:2104.09864v5 §3.2.2 式 (15)，`arxiv.py grep`（页头确认 “arXiv:2104.09864v5 … 08 Nov 2023”） | “rotary matrix with pre-defined parameters” |
| S2 | 相对位置恒等式 $R^d_{\Theta,n-m}=(R^d_{\Theta,m})^\top R^d_{\Theta,n}$ | 同上，式 (16) 之后 | 公式原样 |
| S3 | 旋转矩阵正交，论文认为这保证编码位置时的稳定 | 同上，式 (16) 之后 | “is an orthogonal matrix, which ensures stability” |
| S4 | RoPE 是乘性的 | 同上，§3.2.2 末段 | “our approach is multiplicative” |
| S5 | 频率沿用原始 Transformer，0 起下标写法 | 同上，§3.3 | “Following Vaswani et al. (2017), we set θi=10000^{−2i/d}” |
| S6 | RoPE 与低秩 KV 压缩不兼容 | arXiv:2405.04434v5 §2.1.3（页头确认 v5，19 Jun 2024） | “RoPE is incompatible with low-rank KV compression.” |
| S7 | 不兼容的理由：RoPE 矩阵夹在 $W^Q$ 与 $W^{UK}$ 之间，矩阵乘法不可交换 | 同上 | “matrix multiplication does not obey a commutative law” |
| S8 | 解耦 RoPE 的每头维度 $d_h^R=64$ | 同上，§3.1.2 | “we set the per-head dimension d_h^R to 64” |
| S9 | DeepSeek-V3 配置：`kv_lora_rank` 512、`num_attention_heads` 128、`qk_nope_head_dim` 128、`qk_rope_head_dim` 64、`v_head_dim` 128、`rope_theta` 10000、`rope_scaling.original_max_position_embeddings` 4096 | `huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json`（curl） | 字段原值 |
| S10 | DeepSeek-V2 配置：`kv_lora_rank` 512、`qk_rope_head_dim` 64、`rope_theta` 10000、`original_max_position_embeddings` 4096 | `huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json`（curl） | 字段原值 |
| S11 | DeepSeek LLM 7B 配置：`hidden_size` 4096、`num_attention_heads` 32（每头 128 维）、`num_key_value_heads` 32、`max_position_embeddings` 4096、`rope_theta` 10000.0 | `huggingface.co/deepseek-ai/deepseek-llm-7b-base/raw/main/config.json`（curl） | 字段原值 |

交叉引用（未重核，与本课计算一致）：`math-refs-doc.md` N46（$d=128$ 的频率与波长）、N47（$\mathbf q=(1,2)$、$\mathbf k=(0.5,-1)$、$\theta=0.1$ 的 −2.024045，$R(30°)R(45°)-R(75°)$ 的误差约 $10^{-16}$）；
`deepseek/reference-docs/lesson-05-doc.md` R1、R2（慢针对数与角度）、R10（同向平均得分）、R23（$t=100$ 时快针 329.6°）。

## 2. 本课新算的数（Python 3.11 + NumPy）

图的数字全部在 `math/figures/gen/m03_*.py` 里计算并带 `assert`。正文里另外用到的数：

- **浮点漂移（§1 失败模式）**：$\theta=0.1$，把 $\mathbf e_1$ 用 $\mathbf R(0.1)$ 连乘 $t$ 次，与 $(\cos t\theta,\sin t\theta)$ 比较。
  float32：$t=10^5$ 时长度偏离 1 为 $1.17\times10^{-3}$，与精确位置相差 $1.23\times10^{-3}$；float64：$5.4\times10^{-12}$。
  直接算 $\cos(t\theta)$、$\sin(t\theta)$（数学审阅时重算，2026-09-24，对**全部** $t=1,\dots,10^5$）：角度用 float64 算、结果存 float32，
  离精确位置最多 $4.21\times10^{-8}$；角度也在 float32 里相乘（`float32(t)*float32(0.1)`），最多 $5.86\times10^{-4}$（在 $t=87516$），
  长度离 1 最多 $5.4\times10^{-8}$。原稿写的“float32 下误差只在 $10^{-8}$ 量级”只在 $t=10,10^3,10^4,10^5$ 这几个点成立：
  $0.1t$ 在那里恰好是 float32 能精确表示的整数，角度没有舍入。正文已改成分两种情形陈述。
  连乘的脚本（照抄可复现）：
  ```python
  import numpy as np, math
  th = 0.1
  for dt in (np.float32, np.float64):
      R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=dt)
      v = np.array([1, 0], dtype=dt)
      for t in range(1, 100001):
          v = R @ v
      d = np.array([math.cos(t * th), math.sin(t * th)])
      print(dt.__name__, np.linalg.norm(v.astype(float)) - 1, np.linalg.norm(v - d))
  ```
  直接计算的脚本：
  ```python
  import numpy as np
  ts = np.arange(1, 100001)
  exact = np.stack([np.cos(ts * 0.1), np.sin(ts * 0.1)], 1)            # float64 参照
  a32 = ts.astype(np.float32) * np.float32(0.1)                         # 角度在 float32 里相乘
  d32 = np.stack([np.cos(a32), np.sin(a32)], 1).astype(np.float32)
  e32 = np.linalg.norm(d32.astype(float) - exact, axis=1)
  e64 = np.linalg.norm(exact.astype(np.float32).astype(float) - exact, axis=1)   # 角度 float64，结果存 float32
  print(e64.max(), e32.max(), ts[e32.argmax()], np.abs(np.linalg.norm(d32.astype(float), axis=1) - 1).max())
  ```
- **$R(30°)R(45°)-R(75°)$**：最大误差 $1.67\times10^{-16}$（与 N47 / 核查记录 K10 的“约 $10^{-16}$”一致）。
- **欧拉级数（图 M3.4、§4 例题与小练习）**：$\varphi=2$ 前 8 项之和 $-0.42222+0.90794\mathrm i$，离 $e^{2\mathrm i}$ 0.00623；前 12 项 $8.47\times10^{-6}$。$\varphi=\pi/2$ 前 5 项 $0.0200+0.9248\mathrm i$，前 7 项离 $\mathrm i$ $4.6\times10^{-3}$。
- **多频率打分（图 M3.9）**：$d=128$、$b=10000$、$\mathbf q=\mathbf k=\mathbf 1$，$\frac2d\sum_i\cos(\Delta\theta_i)$：$\Delta=13$ 为 0.6600（单频率 $\cos6.5=0.9766$）；对全部整数 $\Delta\in[100,65536]$ 的最大值 0.4972（在 $\Delta=101$），$[4096,65536]$ 内最大 0.2659、最小 −0.2818。10、100、1000、4096 处的值与 lesson-05-doc R10 一致。
- **4 维 RoPE 算例（§7、图 M3.10）**：$b=10000$，$\theta=(1,0.01)$，$\mathbf q=(1,2,2,1)$、$\mathbf k=(0.5,-1,1,1)$。$(5,2)$、$(105,102)$、$(1005,1002)$、$(3,0)$ 均为 4.231394（两对分别 1.202749、3.028646）；$(2,5)$ 为 4.735883。
- **吸收查询的玩具例子（§7）**：$W=\begin{pmatrix}1&0\\0&1\\1&1\\0&1\end{pmatrix}$，$t=5$。$\tilde{\mathbf q}^{(j)}=W^\top\mathbf R_{\Theta,t-j}\mathbf q$：$j=2,3,4,5$ 依次为 $(0.6969,1.1898)$、$(-0.2551,3.0964)$、$(0.8473,4.9319)$、$(3,5)$；取 $\mathbf c_2=(1,-1)$、$\mathbf c_3=(0.5,2)$、$\mathbf c_4=(-1,1)$、$\mathbf c_5=(2,0.5)$，正确打分 −0.492909、6.065235、4.084665、8.5，与直接旋转后点积逐位相同。$\mathbf R_{\Theta,1}(1,0,1,0)=(0.54030,0.84147,0.99995,0.01000)$，$a+b=1.38177\ne0.99995$。
- **等比与等差频率的对照（§6“为什么取等比数列”、图 M3.8 灰色 ×、练习 12；审阅时补充，2026-09-24）**：首尾频率固定，
  $d=128$：等比（RoPE）波长 $<100$ / $100$–$4096$ / $>4096$ 的对数为 20 / 26 / 18；等差（步长 $(1-1.1548\times10^{-4})/63=0.015871$）为 60 / 3 / 1，倒数第二对 $\theta_{62}=0.01599$、$\lambda_{62}=393$。
  $d=64$：等比 10 / 13 / 9；等差（$\delta=0.032254$）30 / 1 / 1，$\theta_{30}=0.03239$、$\lambda_{30}=194$。生成脚本 `m03_frequencies.py` 里有对应的 `assert`。
  这是笔者为说明“等比是一种选择”而做的比较，RoFormer 只写了沿用原始 Transformer 的设定（S5）。
- **有向面积算例（§3 定义框与例题）**：$\mathbf q=(1,2)$、$\mathbf k=(0.5,-1)$，$\det[\mathbf q\ \mathbf k]=-2$，$\norm{\mathbf q}\norm{\mathbf k}\sin126.87°=2.5\times0.8=2=\operatorname{Im}(q\bar k)$；$\mathbf q$ 与 $\mathbf k$ 方向差 $126.87°$。
- **§3 例题改用图 M3.1 的 $\mathbf x=(2,1)$**：$(0.6+0.8\mathrm i)(2+\mathrm i)=0.4+2.2\mathrm i$，与 §1 的 $U\mathbf x=(0.4,2.2)$ 一致。
- **练习**：练习 7 的打分 $\sin3-\sin0.03=0.111125$；练习 5 的 $44/2\pi=7.00282$；练习 6 的 $163840\theta_{31}=21.848$ rad $=3.477$ 圈；练习 1 的 $\norm{M(1,1)}=1.7889$、$\norm{M(1,-1)}=0.8944$；练习 3 的 $\mathbf R_x\mathbf R_z\mathbf e_1=\mathbf e_3$、$\mathbf R_z\mathbf R_x\mathbf e_1=\mathbf e_2$；练习 10 的 $F(a)^\top F(b)=\mathbf R(a-b)$ 数值核对误差 0。

## 3. 笔者自己的推导或解读（正文已标注）

- **§5 “能与所有二维旋转交换的矩阵恰是 $aI+cJ$”**：标准线性代数结论，正文给了完整推导（由 $AJ=JA$ 比较四格）。
- **§5 笔者补充**：“把旋转搬到缓存一侧”需要 $\mathbf R_jW^{UK}_i=W^{UK}_i\mathbf R'_{j}$；单个头、$W$ 可逆时 $\mathbf R'_j=W^{-1}\mathbf R_jW$ 存在但一般不是旋转（满行秩的宽矩阵 $W$ 可取 $\mathbf R'_j=W^+\mathbf R_jW$，$WW^+=I$）；真正的障碍是 $\mathbf c_j$ 被 128 个头共享而 $\mathbf R'_j$ 随头而变。DeepSeek-V2 论文只给出“矩阵乘法不可交换”一句（S7），这段分析是笔者的，正文已写明。
- **§7 玩具例子**是为本课构造的（$W$、$\mathbf c_j$ 都是随手取的小整数），不来自任何论文；它演示的是“吸收查询随 $j$ 变”这一点，以及在该例中旋转后的键会离开 $W$ 的列空间。真实 MLA 的维度下（$W^{UK}_i\in\R^{128\times512}$，一般满行秩）后一点不成立，正文因此另外指出“$\mathbf c_j$ 被所有头共享”这一障碍。
- **练习 10（用镜像代替旋转）**是笔者设计的教学题，推导完整，未见文献。

## 4. 撰写中注意到、未改动共享文档的问题

- `build_lesson.py`：`::: eq` 标题里出现 `|`（例如写 `|w|`）时，构建脚本抛出 `ValueError: too many values to unpack` 的 Python 回溯，而不是给出指向源文件行号的错误信息。本课改写了标题绕开。
- `::: deriv` 的每一行左侧必须是一个完整的 `$…$`，否则构建失败（提示信息清楚）；本课把文字说明都挪到了理由一栏或 `\text{}` 里。

## 5. 数学审阅的勘误（2026-09-24）

逐条重推了全部 `::: deriv` 与编号公式，并用 NumPy 重算了正文、表格、练习答案与生成脚本里的每个数（除下面第 1 条外全部吻合）。改动：

1. **§1 浮点失败模式**：原稿称“直接算 $\cos(t\theta)$、$\sin(t\theta)$，float32 下误差只在 $10^{-8}$ 量级”，只在 $0.1t$ 恰为整数的几个 $t$ 上成立；改为分“角度用 float64 算”（$\le4.3\times10^{-8}$）与“角度也用 float32 相乘”（$t\le10^5$ 时最多 $5.9\times10^{-4}$，长度不漂）两种情形，数字见 §2。
2. **§1 半正交矩阵**：原稿说 $U\in\R^{d\times k}$ 时“推导的第一行失败”；其实第一行（$U\mathbf x=\mathbf 0\Rightarrow\mathbf x=\mathbf 0$）对列正交归一的高矩阵仍成立，失败的是第二行（非方阵谈不上逆）；改正，并补上 $UU^\top\mathbf v=\mathbf 0$ 的反例。类比框里“瘦矩阵把高维压到低维”改成“$U$ 保长度地嵌入、$U^\top$ 才压扁”。
3. **§7 与 §5 笔者补充**：原稿“真实的 MLA 里还有另一个障碍”暗示列空间障碍在真实 MLA 里也成立；按本笔记第 3 节，单头 $W^{UK}_i$ 是满行秩的宽矩阵，该障碍不存在，正文改为明说。
4. **记号**：预训练长度 $L$ 改为 $L_{ctx}$（style-guide 第 2 节，与第 5 课一致）；记号框补充 $d$ 是被旋转向量的维数（RoFormer 记号，不是模型隐藏维度），以及 §1–§5 的 $a,b,c,d$ 只是临时元素。
5. **小处**：§2 “$(-b,a)$ 在逆时针一侧”补上对所有非零 $(a,b)$ 成立的理由（连续性）；棣莫弗式注明 $t$ 为正整数；§4 例题一处虚部写成 $(-0.4330+0.4330)\mathrm i$（原稿符号顺序颠倒，值不变）；图 M3.4 说明“缩短为 $2/n$ 倍”改为“长度乘 $2/n$（$n\ge3$ 起才缩短）”；练习 2 补上“相位对 $2\pi$ 取余”的条件；(M3.21) 注明取对数时 $\ln b>0$。
6. **图 M3.9 生成脚本**：(b) 在 $\Delta\lesssim26$ 处有些像素列里没有整数 $\Delta$，原脚本改用列左端的非整数 $\Delta$ 取值，与说明“每列内全部整数 $\Delta$”不符；改为跳过这些列。图 M3.10 说明“左右两个面板的红色弧线”改为“红色标注”（右面板只有红字，没有弧线）。

## 6. 第二轮教学审阅（2026-09-24）

没有引入新的外部事实；新增内容都是推导或算术，用 NumPy 核对过（脚本在审阅者的临时目录，要点如下）。

- **§1**：给剪切 $S$ 起名并解释；新增“2×2 矩阵的行列式”前置框（含义 → 理解 → 数学），引用 M01 图 M1.3（矩阵列为 $(2,1)$、$(-1,1)$，$\det=3$）；补 $\det S=1$ 说明“行列式为 1 不等于正交”。要点框改为“二维正交矩阵里 $+1$ 是旋转、$-1$ 是镜像”，避免把高维的 $\det=-1$ 一律叫“镜像”。
- **§2**：$\det=-1$ 情形“沿角度 $\varphi/2$ 的直线镜像”补验算：$U(\cos\frac\varphi2,\sin\frac\varphi2)$ 不变、$U(-\sin\frac\varphi2,\cos\frac\varphi2)$ 反号（对 13 个 $\varphi$ 数值核对）。
- **§3**：原稿“复数法只做了 4 次实数乘法”暗示比矩阵法少算；其实矩阵乘向量也是 4 次乘法。改为“乘法次数相同，省下的是记账：2 个数对 4 格”。
- **§4 推导二**：原稿一句“复数的级数逐项求导给出同样的结果”改为写出逐项求导的三步（$\sum_{n\ge1}\mathrm i^nn\varphi^{n-1}/n!=\mathrm i\sum(\mathrm i\varphi)^{n-1}/(n-1)!=\mathrm if$），数值核对 $\varphi=1.3$ 截断 40 项误差 $<10^{-12}$。
- **§5**：相对角恒等式原来没有“图景”一级，新增类比“同一条环形跑道上速度相同的两个人”，写明失效处：速度不同（练习 2）；旋转之后若再乘一般矩阵 $A$，打分含 $A^\top A$，不再只依赖 $t-j$（例：$A=\begin{pmatrix}2&1\\0&1\end{pmatrix}$，$\mathbf q=(1,2)$、$\mathbf k=(0.5,-1)$、$\theta=0.1$，$(5,2)$ 与 $(13,10)$ 的打分为 $-0.9389$ 与 $-4.0093$，而不乘 $A$ 时都是 $-2.024045$）。
- **§6**：新增注意框“哪两维配成一对是一个约定”：前后半配对的旋转是 $P^\top\mathbf R_{\Theta,t}P$（$P$ 为置换矩阵，正交），相对位置性质照样成立（$d=8$ 数值核对），混用权重才会出错；事实部分引用第 5 课 §3（lesson-05 的 warn 框与其研究笔记）。
- **§7**：“同一个量级”补真实尺寸的乘加数：$W^{UK}_i\in\R^{128\times512}$，每头每个 $j$ 的 $(W^{UK}_i)^\top\mathbf v$ 与 $W^{UK}_i\mathbf c_j$ 都是 $512\times128=65{,}536$ 次乘加。
- **图 M3.9**：(a) 的“12.57”标签原来贴在横轴上方，背景框盖住了红线在 $\Delta=18.85$ 处的第二个波谷；改为放在第一条红点线的正中、打分 $-0.8$ 高度（该处 $\Delta\in[9.2,16]$ 内红线不低于 $-0.15$，脚本里加了 `assert`）。

## 7. 第三轮数学审阅（2026-09-24）

重推了全部 `::: deriv` 与编号公式，用 NumPy 独立重算了正文、表格、练习答案与 10 个生成脚本里的数（包括 §1 浮点漂移：float32 连乘 $1.17\times10^{-3}$ / $1.23\times10^{-3}$，float64 $5.4\times10^{-12}$，直接算 $4.21\times10^{-8}$ 与 $5.86\times10^{-4}$，全部吻合）。没有引入新的外部事实。改动：

1. **§1 与“这一页用在哪里”（实质错误）**：原稿说“RoPE 不改变任意两个查询（或两个键）之间的夹角”“也不改变同一侧向量之间的夹角”。极化恒等式只对**同一个**正交矩阵成立；位置不同的两个查询被 $\mathbf R_{\Theta,t}$、$\mathbf R_{\Theta,t'}$ 转了不同角度，夹角一般会变（这正是 §5 的相对位置信号）。改为“同一个位置的旋转保持经它变换的任意两个向量的夹角”。
2. **§6 慢针（实质错误）**：原稿说 $\lambda_i>4096$ 的 9 对慢针“在训练中只见过 $0°$ 到约 $30°$”，其实只有最慢的 $i=31$ 如此。$d=64$、$t=4096$ 时各对转角：$i=23$ 312.96°、24 234.68°、25 175.99°、26 131.97°、27 98.97°、28 74.21°、29 55.65°、30 41.73°、31 31.30°。正文改为写出 $i=23$、27、31 三个值。
3. **记号**：§5 的拉伸 $\diag(2,1)$ 原来也叫 $S$，与 §1 的剪切 $S$ 同名；改记 $\Lambda$（M01 记作 $D$，但本课的 $D$ 是镜像 $\diag(1,-1)$），图 M3.6 的生成脚本同步改标签。§1 与 §3 把夹角 $\cos\theta$ 改写成 $\cos\angle(\cdot,\cdot)$，本课的 $\theta$ 只表示 RoPE 转速。
4. **舍入不自洽（读者用计算器会对不上）**：§4 例题 $\sqrt{10}(\cos108.43°+\mathrm i\sin108.43°)$ 用 4 位输入算得 $-0.9999+3.0001\mathrm i$，改为 $\approx-1.000+3.000\mathrm i$；§5 例题 $30°+45°$ 原稿 $0.6124+0.3536$ 写成 $0.9659$（实为 0.9660），改用精确值 $(\sqrt6\mp\sqrt2)/4=0.258819$、$0.965926$；§5 例题 $0.0993-0.9801$ 写成 $-0.8807$（实为 $-0.8808$），改用 5 位 $0.09933-0.98007$；练习 7 $0.14112-0.02999$ 写成 0.11112，改为 7 位 $0.1411200-0.0299955=0.1111245$；练习 8 $0.0200+0.99995$ 写成 1.0199，改为 7 位 $0.0199997+0.9999500=1.0199497$；练习 12 $1-30\times0.032254=0.03238$ 写成 0.03239，注明用 $\delta=0.03225376$；练习 4 $53.13°+63.43°=116.56°\ne116.57°$，注明是舍入（精确 $116.565°$）。
5. **表述精度**：极化恒等式的逆命题限定为方阵；Muon 的 $UV^\top$ 对宽矩阵是各行正交归一；指数律推导里二项式定理用到交换律、结合律与分配律（原稿只写交换律）；跑道类比“$t-j=\pm3$ 不同”改为“一般不同”（$q\bar k$ 为实数时相同）；练习 4(b)、7(c) 的虚部引用原写 (M3.10)，但 (M3.10) 不含虚部，且虚部是有向面积的**相反数**，改为引用 §3 的推导；§7 列空间论证注明取 $j=1$；§5 的 $\mathbf c_j$ 注明即第 9 课的 $\mathbf c^{KV}_j$。

## 8. 补充核实：事实与引用审阅（2026-09-24）

逐条列出正文的事实性论断（数字、规格、归属、“论文说”），对照本笔记第 1 节、`math-refs-doc.md`（含核查记录）与 `deepseek/reference-docs/lesson-05-doc.md`；笔记里没有的，回一手来源核实。

### 8.1 新核实的事实（一手来源）

| # | 事实 | 来源（实际读取的） | 原文短引 / 字段 |
|---|---|---|---|
| S12 | V2 §2.1.3 的完整理由：与当前生成 token 有关的 RoPE 矩阵夹在 $W^Q$ 与 $W^{UK}$ 之间；后果是推理时必须为所有前缀 token 重算键 | arXiv:2405.04434v5 §2.1.3，`arxiv.py grep` | “a RoPE matrix related to the currently generating token”；“recompute the keys for all the prefix tokens” |
| S13 | V2 的 MLA：$n_h=128$、$d_h=128$、$d_c=512$（支撑 §5、§7 的“128 个头共享 $\mathbf c_j$”与 $W^{UK}_i\in\R^{128\times512}$） | 同上 §3.1.2 | “number of attention heads n_h to 128 and the per-head dimension d_h to 128” |
| S14 | V2 用 YaRN 把上下文从 4K 扩到 128K，只作用于解耦的共享键 $\mathbf k^R_t$；$s=40$、$\alpha=1$、$\beta=32$，目标最大长度 160K（支撑 §6 的“预训练长度 4096”、练习 6 的 163,840 与 §6 末“这一页用在哪里”） | 同上 §3.1.4 | “from 4K to 128K”；“specifically applied to the decoupled shared key” |
| S15 | 配置复查（curl raw `config.json`）：V2、V3 均为 `kv_lora_rank` 512、`num_attention_heads` 128、`qk_nope_head_dim` 128、`qk_rope_head_dim` 64、`rope_theta` 10000、`original_max_position_embeddings` 4096、`max_position_embeddings` 163840、`factor` 40、`beta_fast` 32、`beta_slow` 1；7B：`hidden_size` 4096、`num_attention_heads` 32、`max_position_embeddings` 4096、`rope_theta` 10000.0。与 S9–S11 一致 | `huggingface.co/deepseek-ai/{DeepSeek-V2,DeepSeek-V3,deepseek-llm-7b-base}/raw/main/config.json` | 字段原值 |
| S16 | RoFormer 选频率时除了“沿用 Vaswani”，还声称这一设定带来远程衰减性质（§3.4.3 给出证明的是一个上界，见第 5 课 §3） | arXiv:2104.09864v5 §3.3 | “this setting provides a long-term decay property” |
| S17 | RoFormer 引言把 Radford et al. (2019)（GPT-2）列为可训练的绝对位置编码（与 lesson-05-doc S3 的 `wpe` 一致，支撑 §1 的“GPT-2 给每个位置学一个向量”） | 同上，引言 | “a trainable absolute position encoding” |
| S18 | RoPE 官方实现先算角度表再取 cos/sin：LLaMA v1 `precompute_freqs_cis` 为 `freqs = torch.outer(t, freqs).float()`，`torch.polar(torch.ones_like(freqs), freqs)  # complex64`；DeepSeek-V3 `precompute_freqs_cis` 取 `dim = args.qk_rope_head_dim`，`freqs` 为 float32，`seqlen = args.max_seq_len` 超过 `original_seq_len` 时按 YaRN 修改 `freqs`，再 `torch.outer(t, freqs)`、`torch.polar(...)`。两者的角度都在 float32 里相乘（§1 失败模式的第二种情形） | `raw.githubusercontent.com/meta-llama/llama/llama_v1/llama/model.py` 第 47–52 行；`raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/inference/model.py` 第 297–375 行（curl） | 代码原文 |

### 8.2 核对无误（未改动）

- 引用：`[@roformer]` 6 处（相对位置、正交与稳定、式 (15)、式 (16) 之后的恒等式、沿用 Vaswani 两处，其中一处本轮补了远程衰减的声称）、`[@deepseekv2]` 3 处、`[@deepseek-configs]` 2 处、`[@rope-code]` 1 处（本轮新增），均能在 S1–S11 与上表找到原文。参考卡片的题名、作者（RoFormer 用 arXiv 作者顺序；期刊版信息见 lesson-05-doc）、年份、编号、URL 都对。
- 跨课链接：`check_links.py` 0 个坏链；逐个打开目标页核对了内容：第 5 课 §1（GPT-2 位置表）、§2（(5.3)(5.4)(5.5)、复数前置框）、§3（(5.6)–(5.9)、(5.11)、图 5.3 的 329.6°、代码 `q = rope(W_q @ h_t, t)`、配对注意框、远程衰减）、§6–§7（PI、NTK-aware、YaRN、163,840）；第 9 课 §4（二维相对性推导第一行是转置 = 反向旋转，后两行用和角公式；(9.12)）、§5（解耦 RoPE）；第 10 课 §6（YaRN）；M01 §1（温度与嗓门）、§2（(M1.6)、图 M1.3 的列 $(2,1)$、$(-1,1)$ 与面积 3、$D=\diag(2,1)$ 与 $R(90°)$ 的例子）、§3、§4、§6（$U^\top U=I$、$UU^\top$ 是投影、矩阵吸收）；M07 §6（$\mathrm{Ortho}(G)=UV^\top$）。
- 数字：用 NumPy 重算了正文、表格、图的文字标签与全部练习答案（脚本在审阅者 scratch 目录 `facts-m03-rotation-complex/verify.py`），与正文一致。
- 练习 1–12 从头独立解了一遍，答案与过程都正确。

### 8.3 本轮改动

1. **§1 浮点失败模式**：原稿“所以 RoPE 的实现都是先算角度”是没有来源的全称判断（只指向第 5 课笔者自写的代码）。改为引用 S18 核实过的 LLaMA v1 与 DeepSeek-V3 官方代码，并指出它们的角度在 float32 里相乘；新增 `rope-code` 制品卡片。
2. **§1 半正交注意框**：原稿说 Muon“把更新矩阵换成 $UV^\top$”，没有交代代码与博客的出入（`math-refs-doc.md` C46：牛顿–舒尔茨迭代只得到约 $US'V^\top$）；补一句“实际只算出近似”，并把链接指到 M07 §6。
3. **§6“为什么取等比数列”**：原稿“RoFormer 论文本身只说沿用原始 Transformer 的设定”漏了论文紧接着的远程衰减声称（S16）；补上，并链接第 5 课 §3 对这一声称的分析。
4. **§5**：“DeepSeek-V2 论文的原话是”后面是中文转述，改为“原话译成中文是”。
5. **练习 2**：原答案说查询、键必须同频率同配对“也是第 5 课强调两种配对写法的权重不能混用的原因”；后者讲的是配对方式与训练时不一致（查询与键仍用同一种配对），是另一回事，改写为“另一层要求”。
6. **§1 开头**：GPT-2 的可学习位置表补上指向第 5 课 §1 的链接（事实见 lesson-05-doc S3 与 S17）。
