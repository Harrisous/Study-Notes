# DeepSeek 科技树 (deepseek/)

The course: lessons 0–20 from a standard GPT to DeepSeek-V4.1-Flash, plus the maths refresher in `math/`.
- Sources are `src/lesson-NN-*.md`, built to `lesson-NN-*.html` (see the root CLAUDE.md for build and check commands).
- Research notes are `reference-docs/lesson-NN-doc.md`.
- Figures are drawn by scripts in `figures/gen/`.
- `reference-docs/course-map.md`, `style-guide.md` and `writer-brief.md` fix the syllabus, notation and terms.

## Lesson videos: the current version (2026-10-01)

**One episode per lesson page.** `video/lNN-sM-js/` holds:
- `script.json`: narration, chapters with level and the question they answer, `questions`, `roadmap`;
- `formulas.txt`: LaTeX, with `\p{name}{...}` parts;
- `intro.tsx`: the opening;
- `boards*.tsx`: one board per chapter;
- `index.tsx`: the entry point and the sound cues;
- `storyboard.md`.

The pipeline and its tools are in `.cursor/skills/lesson-video/SKILL.md`. The reference episode is `video/l01-s4-js` (lesson 1 §4, 2026-10-01).

**The shape of every episode.**
- **Opening, about a minute, one continuous camera move.** It goes from the whole tech tree to this lesson's place on it, then into the page, where the page's questions are pinned on its key formula or figure, and ends on a chapter map grouped by question.
- **Chapters.** Each chapter answers one question and is tagged 零基础 or 工程师向.
- **The last chapter.** It answers the questions with check marks, then backs out to the map and lights the next page.
- **Orientation.** A breadcrumb, a minimap and the "Q1/Q2/Q3" tags keep viewers from getting lost.

**Design.**
- Style: "科技极简 + a little cyberpunk", approved 2026-09-29 on the [style canvas](https://claude.ai/artifact/4xaeYdbBf4sspP6Rxw6uLU).
- Colour: a grey scale plus `#00E5FF`; no scanlines.
- Type: Latin Modern formulas, Noto Sans SC text.
- Boards follow the professor ladder: question → picture → formula → derivation → a number you can check.
- Motion carries meaning: things transform, they don't just appear.
- Data-shaped pictures are computed, not drawn by hand.

**Narration.**
- Write in the lesson's blackboard voice (`reference-docs/style-blackboard.md`), then polish with the 说人话 rules (shuorenhua, local copy in `D:\StudyNotes-video\tools\shuorenhua\`).
- Every number and attribution comes from the lesson page and its research note. Anything illustrative is labelled 示意 in the storyboard.

**Voice.**
- **The version Harry watches is the 曼波 voice.** It runs locally on the GPU through GPT-SoVITS and is built with `--voice gsv:mambo`.
  - The voice imitates a real voice actress, so it is for personal viewing only. Its audio and videos never go into git, an Artifact page or any upload; they stay in `D:\StudyNotes-video\out\<slug>-mambo\`.
  - English is rewritten for this voice the way a Mandarin speaker reads it (`scripts/translit.py`); for example, GQA becomes 吉丘诶. Subtitles keep the English.
  - Whisper scores every sentence, and sentences above 12% CER get two more takes; the clearest take is kept.
- **Kokoro `zm_010`** (local, Apache-2.0) is the default build and the only voice for anything that may be shared.

**Checks before delivering.**
- Contact sheets (`sheet`).
- Full-resolution frames at the end of every narration line.
- Whisper listen-back (`qa`). Fix polyphones with spoken-form pairs such as `[行|航]`.
- Loudness of −16 LUFS (done by `render`).
- A phone copy under 30 MiB for viewing away from the desk.

**Status.**
- Done: `l01-s4-js` (lesson 1 §4).
- Done (2026-10-01): lesson 0, one episode per page, `l00-s1-js` … `l00-s11-js` (§11 is the exercises page).
  - Each 曼波 cut has a phone copy in its `out\<slug>-mambo\` folder.
  - The Kokoro base data is built, because each episode imports it, but the Kokoro cuts are not rendered.
  - Lesson 0 shares `video/l00-shared/`:
    - `scene.tsx`: the opening that flies from the syllabus into the page;
    - `chain.tsx`: the improvement-chain board;
    - `charts.tsx`;
    - `sfx.ts`.
