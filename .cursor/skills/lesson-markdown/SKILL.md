---
name: lesson-markdown
description: Authors and builds the Markdown-sourced, mostly-Chinese study lessons (deepseek/, math/) into the repo's single-file paged HTML, with KaTeX math, standalone SVG figures inlined at build time, and a claude.ai Artifact variant for review comments. Use when writing or editing any file under */src/*.md, adding a figure under */figures/, building or checking a lesson, or publishing a lesson for comments.
---

# Markdown lessons (source → single-file HTML)

The hand-authored HTML lessons (mamba/, reinforcement-learning/) use `lesson-html-tailwind`.
The DeepSeek course and the math refresher are **authored in Markdown** so they can be
co-edited in any editor, and compiled into the same design system.

```
deepseek/
  src/lesson-08-mla.md          <- the source; edit this
  figures/l08-mla-dataflow.svg  <- standalone SVG, inlined into the HTML at build time
  figures/gen/l08_kv_bytes.py   <- script that computed a data-driven figure
  reference-docs/lesson-08-doc.md  <- research notes: verified quotes, numbers, errata
  lesson-08-mla.html            <- build output (committed, served by GitHub Pages)
math/  (same layout)
```

## Build, check, look

```bash
S=.cursor/skills/lesson-markdown/scripts
python $S/build_lesson.py deepseek/src/lesson-08-mla.md          # -> deepseek/lesson-08-mla.html
python $S/check_lesson.py deepseek/src/lesson-08-mla.md          # build + SVG + KaTeX DOM checks
python $S/shoot.py deepseek/lesson-08-mla.html --pages 1-3       # PNGs to inspect (add --dark, --width 420)
python $S/check_links.py deepseek/*.html math/*.html             # cross-lesson links and #pN targets
python $S/build_index.py deepseek/src/index.yaml math/src/index.yaml   # course index pages
python $S/build_lesson.py deepseek/src/lesson-08-mla.md --artifact /tmp/l08.html   # Artifact variant
```

`check_lesson.py` exits non-zero on: build errors, unresolved or uncited references, SVGs
without `viewBox` or with hard-coded colours, KaTeX parse errors, undefined macros, and raw
`\(`/`\[` left untypeset. It needs network access for KaTeX (jsDelivr). No Tailwind
binary is needed: the CSS is assembled from `lesson-html-tailwind/assets/lesson.css`
(design tokens and components, shared), `assets/preflight.css` and `assets/lesson-zh.css`.

## Source format

### Front matter

```yaml
---
title: 第 8 课 · DeepSeek-V2（上）：多头潜在注意力 MLA     # <title> and cover h1
short: MLA 潜在注意力        # 2-4 word Artifact name
kicker: DeepSeek 科技树 · 第 8 课
description: 一句话摘要（meta description）
lede: 封面导语，可以有 $行内公式$ 和 **粗体**。
meta: 预计阅读 120 分钟 · 前置：第 3、4 课，数学 M02
index: index.html                      # course index, relative to the built HTML
prev: lesson-07-grpo.html | 第 7 课 · GRPO
next: lesson-09-v2-moe.html | 第 9 课 · V2 的 MoE
---
```

### Pages

Text before the first `# ` heading is the cover. Each `# ` heading starts a page:

```
# 标题 | 目录里的一句话说明 | 预计分钟数
```

Pages are numbered §1, §2, … automatically; do not type the § yourself. Inside a page use
`##` (becomes h3) and `###` (h4). A References page is generated from the refs block.
On the cover, `[[legend]]` inserts the callout legend and `[[toc]]` the table of contents
(appended automatically if omitted). Keep math out of `#` titles.

### Math

`$…$` inline, `$$…$$` display (may span lines; give it its own paragraph with blank lines around it,
otherwise it gets no horizontal-scroll wrapper on phones). A literal dollar is `\$` (e.g. `\$5.6M`).
Reference-card `finding`/`matters` lines may contain inline math too.
Macros available: `\R \E \N \norm{} \abs{} \softmax \diag \Attn \RMSNorm \LayerNorm \FFN
\KL \T` (transpose, use as `W\T`) `\argmax \argmin \TopK \clip \Var \Cov \tr \rank \sign`.
Never put `<`/`&` issues aside: the builder escapes math for you.

### Containers

Fenced with three or more colons; nest by using more colons on the outer fence.

| fence | renders as | use for |
|---|---|---|
| `::: def 标题` | blue box | definitions, formal setup |
| `::: thm 标题` | violet box | a proved or paper-stated result, with source and hypotheses |
| `::: key 标题` | green box | the takeaway |
| `::: warn 标题` | orange box | a subtlety, a hidden assumption |
| `::: err 标题` | red box | a common wrong claim and its correction |
| `::: note 标题` | grey box | aside |
| `::: ex 标题` | ink-ruled box | worked example with real numbers |
| `::: ana 标题` | dashed box | analogy: mapping, payoff, and the seam where it breaks |
| `::: pre 标题 \| 跳过提示` | amber aside | prerequisite at the point of need |
| `:::: quiz 标题` + nested `::: answer 看答案` | exercise with collapsible answer |
| `::: eq 标签 \| (8.3)` | numbered display equation; body = one `$$…$$` plus an optional note line. `eq.o` `eq.v` `eq.g` recolour the rule |
| `::: deriv 标题` | line-by-line derivation; each row is exactly one `$expr$ \|\| 理由` (words go in `\text{}` or the reason); keep rows short, a wide row squeezes the reason column |
| `::: fig figures/x.svg \| 图 8.2 标题 \| wide fit` | inlined SVG; body = caption. Figures scroll sideways on phones by default (min width 36rem); `fit` opts a simple figure out, `wide` bleeds 1rem into the margins |
| `:::: cols` + `::: col` ×2 | two columns |
| `::: todo` | orange dashed marker (unfinished work) |

### Inline

- Citation: `[@deepseekv2]` or `[@a; @b]`, resolved against the refs block, numbered by
  first use.
- Tooltip gloss: `{{术语::一句话解释（纯文本，不要公式）}}`.
- Bold with `**…**` (safe next to CJK punctuation); avoid `*italic*` in Chinese text.
- Review notes for co-editing: `<!-- 批注(Harry): … -->` and `<!-- 回复(Claude): … -->`.
  HTML comments never reach the built page.
- `> …` blockquotes render as a quoted passage (use for verbatim source quotes).
- `\KL` already carries its subscript (it is `\mathop{\mathbb D_{KL}}\nolimits`), so `\KL_t` works; a literal
  dollar inside math is `\$` as well.
- A built page accepts query flags for inspection: `?all` (every page at once), `?open` (answers expanded),
  `?dark` / `?light`. `shoot.py` uses `?open` and slices tall pages into strips.

### References

One fenced block, anywhere (by convention at the end):

````
```refs
- id: deepseekv2
  title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
  a: DeepSeek-AI (2024). arXiv:2405.04434
  url: https://arxiv.org/abs/2405.04434
  finding: 一句话：这篇文献确立了什么。
  matters: 一句话：本课为什么需要它（不能和 finding 重复）。
- id: flashmla
  kind: artefact                 # model card, repo, blog: no paper
  title: FlashMLA (GitHub repository)
  ...
```
````

Every ref must be cited; every citation must resolve. The build fails otherwise.

## Figures

Each figure is a standalone `.svg` file under `figures/`, named `l08-…svg` / `m02-…svg`.

1. Root `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 W H" font-family="system-ui, 'PingFang SC', 'Microsoft YaHei', sans-serif">`.
   Width/height attributes are stripped at build time; design at W≈640–760 so 1 unit ≈ 1 px.
2. Colours **only** as `var(--fig-ink, #1C1B19)` with the light-theme fallback, so the file
   renders standalone and follows the theme once inlined. Tokens: `--fig-ink #1C1B19`,
   `--fig-muted #74716A`, `--fig-rule #DCD7CC`, `--fig-grid #EDE9E0`, `--fig-surface #FFFFFF`,
   `--fig-blue #1F5FA8`, `--fig-violet #6733C4`, `--fig-green #16704A`, `--fig-red #A3231C`,
   `--fig-orange #B4460C`, `--fig-amber #8A6508`. Tinted fills: `fill="var(--fig-blue, #1F5FA8)" fill-opacity="0.12"`.
3. Text ≥ 12 units (labels), ≥ 11 (tick labels). Chinese labels are fine.
4. Data-driven geometry (curves, bars, heatmaps, scaled axes) is **computed** by a script in
   `figures/gen/` that writes the SVG; never eyeball a curve. Schematics may be hand-written.
5. Ids are namespaced at build time, so `<marker id="arrow">` may repeat across files.
6. Colour carries meaning consistently across the whole course (see the course style guide).
7. Do not hide geometry with `clipPath`: figures render with `overflow: visible`, so clipped-away
   shapes still widen the page on phones. Clip lines in the generator script instead.

## Verify before calling it done

- `check_lesson.py` passes (it runs the build, the SVG lint and the KaTeX DOM scan).
- Screenshot every page with `shoot.py`, light and dark, at least one page at `--width 420`;
  look for clipped SVG text, overflowing tables, crushed derivations.
- The companion `reference-docs/lesson-XX-doc.md` lists every number and quote used, where
  it came from, and every correction made (see `academic-research`).
