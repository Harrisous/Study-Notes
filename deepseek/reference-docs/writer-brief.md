# 写课手册（给每一课的作者）

> 读者：写某一课 `deepseek/src/lesson-NN-*.md` 或 `math/src/mNN-*.md` 的人或 agent。
> 开写前按顺序读：`course-map.md`（自己那一课 + 前后两课）、`style-guide.md`、
> `.cursor/skills/professor/SKILL.md`、`.cursor/skills/lesson-markdown/SKILL.md`、
> 本课的研究笔记 `reference-docs/lesson-NN-doc.md`（数学课读 `math/reference-docs/math-refs-doc.md`）。

## 你要交付什么

1. `src/<slug>.md`：本课 Markdown 源文件（格式见 lesson-markdown SKILL）。
2. `figures/<lNN|mNN>-*.svg`：本课所有图，一图一文件。
3. `figures/gen/<lNN|mNN>_*.py`：凡是数据驱动的图（曲线、柱状图、热力图、按比例的示意），生成它的脚本。
   脚本只用 Python 标准库（可以用 numpy），直接写出 SVG 文件；运行 `python figures/gen/xxx.py` 可复现。
4. 构建产物 `<slug>.html`（运行构建脚本得到），并通过检查。
5. 如果你为了写课自己核实了研究笔记里没有的事实，把它追加到研究笔记的“补充核实”小节（附来源）。

## 质量标准

- **篇幅**：DeepSeek 课 6–9 页、1.2 万–2.5 万汉字；数学课 5–8 页、0.8 万–1.6 万汉字。宁可写透，不要凑字：
  每一段都要推进理解。
- **图**：DeepSeek 课 8–15 张，数学课 5–10 张。至少一半是“机制图”（数据流、结构、计算步骤），
  其余是“账单图”（按真实数字计算的柱状图/曲线）。每张图说明文字第一句写“看什么”。
- **公式**：DeepSeek 课 20–50 个行间公式；每个重要公式用 `::: eq` 编号（编号格式 `(课号.序号)`，如 `(9.4)`）；
  关键推导用 `::: deriv`，每行有理由。
- **例题**：每个核心概念至少一个 `::: ex`，用真实数字（来自研究笔记），读者能用计算器复核。
- **练习**：最后一页（参考文献页之前）“练习与答案”，6–12 题，分概念题/计算题/推导题；每题用
  `:::: quiz` + `::: answer`，答案写完整过程。正文中间也可以穿插 1–3 个小练习。
- **五级阶梯**（professor 契约）：问题 → 图景（类比，写明失效边界）→ 形式化 → 推导 → 数字与失败模式。
  类比优先复用 style-guide 第 5 节的那 8 个。
- **前置知识**：用 `::: pre`，就地讲清，并链接数学课：`[M02 §3](../math/m02-rank-svd-lowrank.html#p3)`。
  数学课则反过来链接到用到它的 DeepSeek 课。
- **每个术语**：含义（一句白话）→ 理解（直觉/例子）→ 数学（定义/公式）。第一次出现可以加 `{{术语::一句话解释}}`
  浮窗。
- **事实**：所有数字、公式、日期、规格只能来自研究笔记（或你自己用工具核实过的来源）。研究笔记“未能核实”里
  的内容，要么不写，要么明确写“未能从原文核实”。区分 已证明 / 实测 / 推测（笔者推测）。
- **引用**：从研究笔记第 9 节拿参考文献卡片，放进本课的 ```refs 块；正文用 `[@key]`。每条 ref 都要被引用。
- **禁用词**：显然、易证、不难看出、容易看出、众所周知、“可以证明”（除非紧跟着证明或精确出处）。
  也不要“美妙的”“优雅的洞见”这种用形容词代替机制的写法。

## 页面结构建议（DeepSeek 课）

- 封面：导语（这课回答什么问题）、“这一课在科技树上的位置”（一小段 + 上一课留下的问题）、
  “读完你能……”（3–5 条可检验的能力）、`[[legend]]`、`[[toc]]`。
- §1：问题。上一课/上一代模型的账单，让读者感到痛。
- §2–§(n-2)：机制，一页一个核心机制，每页走完五级阶梯。
- §(n-1)：“算总账”——把本课所有设计放回一张对比表/一张图里，给出真实数字。
- §n：练习与答案。
- 参考文献页由构建器自动生成。

**费曼学习层（2026-10 起每课都要有）**：按 `style-feynman.md` 加上封面的“开课前，先写下你的理解”
（`predict {#before-1..3}`）、每个讲内容的页末一个“费曼时刻”（原来页末的“这一页讲了三件事”移进它的参考讲法）、
以及练习页之后的最后一页“费曼复盘”。新页放在最后，已有页码不变。`check_lesson.py` 会提醒漏掉的页。

## 图（SVG）规范要点

- 见 lesson-markdown SKILL「Figures」。颜色只用 `var(--fig-*, #fallback)`，语义见 style-guide 第 4 节。
- viewBox 宽 640–760；文字 ≥ 12（刻度 ≥ 11）；中文标签用正常字重；不要让文字贴边或互相重叠。
- 结构图要画出**真实的维度**（例如在张量方块旁标 $d_c=512$），不要画成抽象方框。
- 改绘论文图时，说明文字写“据《……》图 k 重绘”并引用。不要照抄论文的配色，用本课程的颜色语义。
- 数据图：坐标轴有单位与刻度，对数轴要标明；曲线/柱子的几何由脚本从数据计算。
- 文字里不要用 LaTeX（SVG 里不渲染 KaTeX）；需要下标就用 `<tspan baseline-shift="sub" font-size="…">`，
  希腊字母直接用 Unicode（θ、α、ε、√）。

## 编号与细节约定（来自 M01 试写）

- 编号：DeepSeek 课公式 `(9.4)`、图“图 9.4”；数学课公式 `(M2.4)`、图“图 M2.4”。
- `$$…$$` 独占一段，前后各空一行，否则手机上不会得到横向滚动容器。
- SVG 里不要用 `clipPath` 藏几何：图以 `overflow: visible` 绘制，被裁掉的部分仍会撑宽页面。在生成脚本里直接把线段截断。
- 在 Python 字符串里写 TeX 时用原始字符串 `r"..."`：普通字符串里反斜杠加 t 会变成制表符，把 top 前的反斜杠吃掉，TeX 的转置符号就变成了字母 “op”（检查器会报 control character）。
- 参考文献卡片的 finding / matters 可以写 `$…$` 行内公式。
- 页内跳转写 `[§3](#p3)`；跨课跳转写 `[第 9 课 §3](lesson-09-v2-mla.html#p3)`。
- 做完一课后参照 `math/src/m01-vectors-matrices.md`（已完成的示范课）检查自己的密度与格式：它有 17 个算例、
  10 个逐行推导、22 个编号公式、9 张图、10 道带完整答案的练习。DeepSeek 课应当至少同样充实。

## 工作流程

```bash
cd <repo>
S=.cursor/skills/lesson-markdown/scripts
python deepseek/figures/gen/l09_kv_bytes.py                  # 生成数据图
python $S/build_lesson.py deepseek/src/lesson-09-v2-mla.md   # 构建
python $S/check_lesson.py deepseek/src/lesson-09-v2-mla.md   # 必须 OK（需要联网加载 KaTeX）
python $S/shoot.py deepseek/lesson-09-v2-mla.html --out <scratch>/shots            # 每页截图
python $S/shoot.py deepseek/lesson-09-v2-mla.html --pages 2 --dark --out <scratch>/shots
python $S/shoot.py deepseek/lesson-09-v2-mla.html --pages 3 --width 420 --out <scratch>/shots
```

- 源文件很长，**分多次写**：先写前两页，构建检查，再用编辑追加后面的页，每次追加后构建一次。
  不要试图一次性输出整个文件。
- 截图用 Read 工具打开 PNG 逐页看：公式是否渲染、SVG 文字是否被裁切或重叠、表格是否溢出、深色模式是否可读。
  发现问题就改，改完再截图确认。
- 截图和临时文件放在你的 scratch 目录，不要放进仓库。

## 所有课的文件名（用于互相链接）

DeepSeek（`deepseek/`）：
`lesson-00-tech-tree` · `lesson-01-transformer-anatomy` · `lesson-02-gpt-training` · `lesson-03-alignment-rlhf` ·
`lesson-04-inference-kv-cache` · `lesson-05-modern-block` · `lesson-06-deepseek-llm` · `lesson-07-deepseekmoe` ·
`lesson-08-deepseekmath-grpo` · `lesson-09-v2-mla` · `lesson-10-v2-moe-training` · `lesson-11-v3-architecture` ·
`lesson-12-v3-infrastructure` · `lesson-13-r1-reasoning` · `lesson-14-sparse-attention` · `lesson-15-v3-2-agents` ·
`lesson-16-mhc` · `lesson-17-engram-ocr` · `lesson-18-v4` · `lesson-19-v4-1-flash` · `lesson-20-retrospective`

数学（`math/`）：
`m01-vectors-matrices` · `m02-rank-svd-lowrank` · `m03-rotation-complex` · `m04-probability` ·
`m05-information-theory` · `m06-calculus-backprop` · `m07-optimization` · `m08-floating-point` · `m09-rl-math` ·
`m10-compute-arithmetic` · `m11-doubly-stochastic-sinkhorn`

链接写法：同目录 `lesson-09-v2-mla.html#p3`；跨目录 `../math/m02-rank-svd-lowrank.html#p2`、
`../deepseek/lesson-09-v2-mla.html`。`#pN` 跳到第 N 页（封面是 p0）。课程目录：DeepSeek 课 `index: index.html`，
数学课 `index: index.html`（各自目录下的 index.html）。
