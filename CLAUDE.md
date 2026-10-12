# Study-Notes

Self-contained lecture notes published with GitHub Pages. Two authoring pipelines live side by side:

- **Hand-authored HTML** (`mamba/`, `reinforcement-learning/`): see `.cursor/skills/lesson-html-tailwind/SKILL.md`.
- **Markdown-sourced lessons** (`deepseek/`, `math/`, mostly Chinese): edit `*/src/*.md`, never the built
  `*.html`. See `.cursor/skills/lesson-markdown/SKILL.md`. Build and check with
  `python .cursor/skills/lesson-markdown/scripts/build_lesson.py <src.md>` and `check_lesson.py <src.md>`.
- **Lesson videos**: see `.cursor/skills/lesson-video/SKILL.md`. There are two pipelines.
  - `deepseek/video/*.py`: Manim, voiced by ElevenLabs, which is billed. Lay out with `--dry` first.
  - `deepseek/video/*-js/`: Remotion (React), voiced locally by Kokoro, which is free. Lay out with `sheet` stills.
  Large files (venvs, TinyTeX, node_modules, caches, renders) live on `D:\StudyNotes-video`.

Writing rules for every lesson: `.cursor/skills/professor/SKILL.md` (explanation contract) and
`.cursor/skills/academic-research/SKILL.md` (never state a number or attribution from memory). The DeepSeek
course adds `deepseek/reference-docs/{course-map,style-guide,writer-brief}.md`, which fix the syllabus, notation,
terminology and figure colours for both `deepseek/` and `math/`.

## Co-editing and review comments

- Harry edits the Markdown sources directly. Review notes go in HTML comments, which never reach the built
  page: `<!-- 批注(Harry): … -->`, answered with `<!-- 回复(Claude): … -->` or by making the change and deleting
  the note. `grep -rn "批注(" deepseek/src math/src` lists open notes.
- Each built lesson can also be published as a private claude.ai Artifact for in-page comments:
  `build_lesson.py <src.md> --artifact <scratch>/<slug>.html`, then publish that file (same file path on every
  republish keeps the URL). The Artifact URLs are recorded in `deepseek/reference-docs/artifacts.md`.
  When a comment arrives, change the Markdown source, rebuild both variants, republish, and reply in the thread.
- Every factual change goes through the lesson's `reference-docs/lesson-NN-doc.md` first (source + quote).
