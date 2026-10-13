# Study Notes

Self-contained lecture notes, written to be learned from rather than skimmed. Every lesson is a single HTML page with
its own figures, formulas and worked examples. The site is published with GitHub Pages:
<https://harrisous.github.io/Study-Notes/>.

## Courses

| Course | Read on GitHub Pages | Interactive version (claude.ai) |
|---|---|---|
| **DeepSeek 科技树**: 21 lessons from a standard GPT forward pass to DeepSeek-V4.1-Flash (Chinese) | [deepseek/](https://harrisous.github.io/Study-Notes/deepseek/) | [课程目录](https://claude.ai/artifact/Dsrd1uuGzLMQ9Cbja9rmwr) |
| **第 1 课 · Transformer 解剖**: with the Feynman practice layer (write, teach back, check, review) | [lesson 1](https://harrisous.github.io/Study-Notes/deepseek/lesson-01-transformer-anatomy.html) | [第 1 课（费曼练习版）](https://claude.ai/artifact/JPjrAfp7YpgC6p6CYdEWDc) |
| **数学补课**: 11 lessons of the undergraduate maths the DeepSeek course needs (Chinese) | [math/](https://harrisous.github.io/Study-Notes/math/) | [数学补课目录](https://claude.ai/artifact/FRTcqR8vBKDJepW7i81RLz) |
| **Mamba** state space models | [mamba/](https://harrisous.github.io/Study-Notes/mamba/lesson-01-the-problem-with-sequences.html) | |
| **Reinforcement learning from scratch**, 16 lessons | [reinforcement-learning/](https://harrisous.github.io/Study-Notes/reinforcement-learning/reinforcement-learning-from-scratch.html) | |
| **System One 费曼课**: System One, Jev, RLCD calibration and training an open Jev-style judge, with Feynman cards (Chinese, in progress) | [system-one/](system-one/) | [System One 费曼课](https://claude.ai/artifact/2zWp8sHYmJP9jCU3xRnJaB) |

The claude.ai versions take in-page comments. In lesson 1, each 费曼时刻 box also has a 请 Claude 找漏洞 button that
checks your own explanation. The links are public but still being revised, so content may change.

## How the lessons are made

- Markdown-sourced lessons (`deepseek/`, `math/`): edit `*/src/*.md` and build with
  `python .cursor/skills/lesson-markdown/scripts/build_lesson.py <src.md>`. See `CLAUDE.md` and
  `.cursor/skills/lesson-markdown/SKILL.md`.
- Every number and quote traces to a per-lesson research note in `*/reference-docs/`.
- Lesson videos: `.cursor/skills/lesson-video/SKILL.md`.
