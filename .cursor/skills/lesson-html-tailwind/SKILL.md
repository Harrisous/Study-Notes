---
name: lesson-html-tailwind
description: Builds and edits the single-file, multi-page HTML study lessons in this repository using Tailwind v4 compiled and inlined via the standalone CLI, KaTeX for math, and hand-authored inline SVG for figures. Use when creating a new lesson page, editing an existing lesson's markup or styles, adding figures, equations, callouts, tables, or references, or when the user mentions Tailwind, the lesson design system, or rendering/verifying a lesson HTML.
---

# Lesson HTML (Tailwind v4, single file)

Each lesson is **one `.html` file** with no runtime asset dependency except the
KaTeX CDN. Tailwind is compiled at authoring time and inlined. Figures are
hand-authored inline SVG. Opening the file from disk works; GitHub Pages serves
it unchanged.

## Layout of a lesson file

```
<head>
  <style id="tw"></style>          <!-- compiled Tailwind, injected by build -->
  <link  KaTeX css (SRI) >
  <script defer KaTeX js + auto-render (SRI) >
</head>
<body>
  <div class="bar">…</div>          nav: page pips, dark toggle, print
  <div class="prog"><i></i></div>   reading progress
  <main class="shell prose-lesson">
    <section class="page on" data-t="Cover">      … </section>
    <section class="page"    data-t="§1 …">       … </section>
    …
    <section class="page"    data-t="References"> … </section>
  </main>
  <script> paging + tooltips + KaTeX render </script>
</body>
```

One `<section class="page">` per section. Only the active one is displayed;
`@media print` shows them all in order. `data-t` supplies the pager labels, so
adding a section needs no other bookkeeping.

## Build

```bash
B=.cursor/skills/lesson-html-tailwind/scripts/build_css.py
python3 $B mamba/lesson-01-the-problem-with-sequences.html
```

This scans that file for utility classes, compiles `assets/lesson.css`, and
replaces the body of `<style id="tw">`. **Re-run after every change to a
`class` attribute.** One-time binary install (no Node required):

```bash
curl -sL -o ~/.local/bin/tailwindcss \
  https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-macos-arm64
chmod +x ~/.local/bin/tailwindcss
```

## Non-negotiable design rules

1. **Semantic tokens only.** Use `text-ink`, `text-ink-2`, `text-muted`,
   `bg-paper`, `bg-surface`, `bg-surface-2`, `border-rule`, and the accent
   tokens `blue / violet / green / amber / red / orange` (each with a `-soft`
   background). Never `text-slate-600`, never `bg-gray-100` — raw palette
   utilities do not flip in dark mode and are the one thing that reliably
   breaks these pages.
2. **Accent colours carry meaning.** blue = definition/formalism, violet =
   theorem/formal result, green = key takeaway, amber = prerequisite aside,
   orange = caution or subtlety, red = common error / correction. Never pick an
   accent for looks.
3. **Inline SVG only** for figures — no canvas, no chart library, no `<img>`.
   Colour SVG strokes and fills with `var(--fig-ink)`, `var(--fig-muted)`,
   `var(--fig-rule)`, `var(--fig-grid)`, `var(--fig-blue)`, … so figures follow
   the theme. A hardcoded `#333` is a dark-mode bug.
4. **Compute figure geometry, do not eyeball it.** Generate the path data with
   a Python script and paste the result. A plot of `0.9^n` whose curve is drawn
   by hand is a lie in a document about numbers.
5. **Every SVG needs `viewBox`** and no `width`/`height` attributes; the
   container sizes it.
6. **Math in KaTeX delimiters**: `\( … \)` inline, `\[ … \]` display. Escape
   `\` correctly in HTML, and never put `<` or `&` unescaped inside math.
7. **No em dashes** in prose; use a spaced hyphen or restructure.

## Component vocabulary

Full CSS in `assets/lesson.css`. The pieces to reach for:

| markup | use |
|---|---|
| `<div class="box box-def"><span class="box-h">Definition</span>…` | definitions, formal setup |
| `box-thm` / `box-key` / `box-warn` / `box-err` / `box-note` | theorem, takeaway, caution, correction, aside |
| `<aside class="pre"><span class="box-h">Prerequisite<span class="skip">skip if you know X</span></span>…` | background at the point of need |
| `<div class="eq"><span class="eq-l">label</span>\[…\]<span class="eq-n">note</span><span class="eq-t">(1)</span></div>` | display equation, `.eq.o/.v/.g` recolours the rule |
| `<div class="deriv">` + `<table>` with `td.step`, `td` (expression), `td.why` | line-by-line derivation with a reason column |
| `<div class="tw"><table class="tbl">` + `td.n` / `td.c`, `tr.good` / `tr.bad` | tables; `.n` is numeric right-aligned mono |
| `<figure class="fig">` + `<svg viewBox>` + `<figcaption>` | figures |
| `<a class="cite" href="#r7">7</a>` and `<div class="ref" id="r7">` | citation and reference card |
| `<span class="gl" data-d="…">term</span>` | hover gloss |

Reference cards carry a **Finding** line and a **Matters** line, not just a
bibliographic string — see the `academic-research` skill.

## Verify before declaring done

The Browser pane serves stale snapshots for `file://` URLs here. Screenshot
with headless Chrome and then Read the PNG:

```bash
python3 - <<'PY'
import urllib.parse, os
p = os.path.abspath("mamba/lesson-01-the-problem-with-sequences.html")
print("file://" + urllib.parse.quote(p))
PY

"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
  --disable-gpu --hide-scrollbars --virtual-time-budget=4000 \
  --window-size=1100,3000 --screenshot=/tmp/p1.png "file://<encoded>#p1"
```

`--virtual-time-budget` matters: without it the screenshot can land before
KaTeX has typeset. Use `sips -c` / `sips -z --cropOffset` to zoom into a
suspect region (there is no PIL on this machine).

Check, per page: no raw `\(`…`\)` left visible (KaTeX failed), no horizontal
overflow, no clipped SVG text, tables not crushed, dark mode readable.

## Checklist

- [ ] `python3 $B <file>` re-run after the last markup edit
- [ ] no raw palette utilities; no hardcoded colours in SVG
- [ ] every figure's geometry computed, every axis labelled with units
- [ ] every equation numbered if referred to later
- [ ] every `.cite` resolves to an existing `.ref` id, and every ref is cited
- [ ] screenshotted every page, light and dark
- [ ] prints as one continuous document
