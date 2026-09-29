---
name: lesson-video
description: Turns a lesson page into a narrated 16:9 explainer video (Manim CE + ElevenLabs) in a 3Blue1Brown-style or chalkboard theme, with per-segment difficulty badges (零基础 / 工程师向), burned-in Chinese subtitles, an SRT file, a chapter list and per-chapter files for a web player. Use when writing or editing an episode under */video/*.py, changing the voice or theme, or building, checking or re-rendering a lesson video.
---

# Lesson videos (episode script → narrated blackboard video)

An **episode** is one Python file per lesson page (`deepseek/video/l01-s4.py` is the first). It
holds the narration and the board work together. `scripts/make_video.py` turns it into an mp4.
The code and the episode scripts live in the repo. Everything large lives on `D:\StudyNotes-video`
(override with `BB_VIDEO_ROOT`):

| Path | What |
|---|---|
| `venv\` | Python 3.11 with manim 0.21, faster-whisper, numpy, pillow |
| `TinyTeX\` | LaTeX for `MathTex` (not on PATH; the build adds it) |
| `cache\tts\` | one mp3 + timing json per narration line, keyed by text+voice+settings |
| `work\<slug>\<dry|preview|final>\` | Manim media, logs and timelines per segment |
| `out\<slug>\` | the mp4, `subs.srt`, `chapters.txt`, `sheet-*.png`, `summary.json` |

The ElevenLabs key is `ELEVENLABS_API_KEY` in the repo's `.env` (git-ignored).

## Build: cheap passes first

```bash
PY=D:/StudyNotes-video/venv/Scripts/python.exe
S=.cursor/skills/lesson-video/scripts
$PY $S/make_video.py all deepseek/video/l01-s4.py --dry      # silent placeholder voice, 480p: free, ~1 min
$PY $S/make_video.py tts deepseek/video/l01-s4.py            # bill only lines not in the cache
$PY $S/make_video.py qa  deepseek/video/l01-s4.py            # Whisper listen-back, prints lines above 12% CER
$PY $S/make_video.py all deepseek/video/l01-s4.py            # 1080p30 final (TTS already cached)
$PY $S/make_video.py render deepseek/video/l01-s4.py --only S05Saturation   # re-render one segment
$PY $S/make_video.py split deepseek/video/l01-s4.py          # one file per chapter (<19 MiB) + chapters.json
$PY $S/make_video.py page  deepseek/video/l01-s4.py --page-dir <scratchpad>/l01-s4-video   # the video page
```

`page` fills `assets/player.html` from the episode's `page_title`, `kicker`, `headline`, `lede` and
`source` fields. Publish its `index.html` as an Artifact, with `poster.jpg` and `chapters/*.mp4` as
`files` (each under 15 MB, the lot under 64 MB). Publishing reads only from the repo or the session
scratchpad, which is why `--page-dir` points there. The video pages are listed in
`deepseek/reference-docs/artifacts.md`.

An episode's `EPISODE` dict sets `theme` (`"3b1b"`, the default, or `"chalk"`), `voice` (an ElevenLabs
voice id) and `speak_numbers` (`"zh"`: every bare number is read the Chinese way while the subtitle keeps
the digits, so `64` is spoken 六十四, `2017 年` 二零一七年, `87%` 百分之八十七).

- **3b1b**: black ground, 思源黑体 (Noto Sans SC), Computer Modern maths and Manim's 3Blue1Brown palette
  (blue #58C4DD for queries, purple for keys, green for values, yellow for highlights, red for masked).
- **chalk**: green board, 楷体 and chalk colours, with a dust texture laid over the frame at assembly.

Per-chapter files are for the private video page (an Artifact with the `assets` capability, whose
uploads are capped at 20 MiB each). The page lists chapters with their level and can skip the
工程师向 ones.

1. **Lay out with `--dry`.** Narration is faked at 4.3 characters per second, so the whole episode
   renders in about 30 s for nothing. `render` prints layout warnings (anything outside the safe
   area). `sheet` writes contact sheets with one tile per board state: the frame before each
   `wipe()` and the end of each segment. Look at the sheets, not the video. They show every board
   in 2–3 images.
2. **Synthesise once.** `tts` bills only lines whose spoken text, voice or settings changed. Editing
   a board never re-bills. Rewording one line re-bills only that line.
3. **Listen back with `qa`.** Nobody on the pipeline side can hear the audio, so every clip is
   transcribed with Whisper (large-v3-turbo, CPU) and scored against its script. Results are cached
   next to the clip. Homophones (记/即/计) count as errors, so check flagged lines by reading them,
   and re-word the few that are really garbled.
4. **Render final.** Segments render in parallel, one Manim process each (`-j`, default 8).
   Assembly concatenates them and mixes the narration. It overlays the chalk-dust texture and the
   level strip, burns in the subtitles, embeds the chapters, and encodes with NVENC (falls back to
   libx264, or force it with `--cpu`).

## Writing an episode

```python
from blackboard import *

EPISODE = {"title": "第 1 课 §4 · 注意力（二）"}

class S01Coins(Segment):            # S<nn> = order and chapter number
    level, title = ZERO, "开场：抛 64 次硬币"     # ZERO = 零基础, ENG = 工程师向

    def body(self):
        with self.say("抛 64 次硬币，{a}和通常落在[±8|正负八]上下。") as v:
            line = self.put(L(r"和 $S$ 通常在 $\pm 8$ 上下"))   # next board line
            v.until("a")                                         # wait for the bookmark
            self.play(Write(line))
        self.wipe()                                             # erase, also a contact-sheet tile
        self.narrate("一句不配动画的话。")
```

- **Narration markup.** `{name}` is a bookmark for `v.until("name")`. `[shown|spoken]` puts one
  thing in the subtitle and has the voice say another. Use it for maths (`[√d_k|根号 d k]`) and for
  words the voice misreads: 行 as háng is safer as `[按行|一行一行地]`. With `speak_numbers: "zh"`
  you never need to write number pairs by hand. Leaving a `with self.say()` block waits for the clip to
  end. Keep every `say()`/`narrate()` text a **string literal**: the TTS and QA passes read them
  from the file without running Manim.
- **Helpers.** `T` (chalk text, KaiTi), `M` (MathTex), `L("text $maths$ text")` (one mixed line),
  `table`, `bars`, `code`, `why` (a derivation step's reason), `chalk_box`, `chalk_underline`,
  `badge`. `self.put(mob, gap=, x=, center=)` places the next line under the previous one.
- **Colours** follow the course's figure semantics (`style-guide.md` §4): `Q_BLUE` query,
  `K_VIOLET` key, `V_GREEN` value/good, `AMBER` cache and highlight, `ORANGE` compressed,
  `RED` masked/wrong, `MUTED` scaffolding, `INK` plain chalk.
- **Safe area.** Keep content inside x ±6.75 and y from −2.75 to 3.05. The strip below −2.75 is
  for subtitles, and the header sits above 3.05. Text runs about 0.0117 × font size units per
  Chinese character: 28 characters per line at size 40, 38 at size 30.

## Content rules

- The voice is the lesson's blackboard voice (`deepseek/reference-docs/style-blackboard.md`): 大家,
  short sentences, a question before each answer, the small hand-worked example before the formula.
- Every number and attribution comes from the lesson page and its research note. Nothing is added
  from memory (professor and academic-research skills). Mark an illustrative sample as 示意.
- **Levels.** 零基础 means anyone can follow it: analogies, hand-worked numbers, conclusions.
  工程师向 needs linear algebra, probability or deep-learning background: derivations,
  gradients, code, parameter counts. Every segment has exactly one level. The title card explains
  the two badges, and `chapters.txt` lists them for the video description.

## Voice

`scripts/tts.py` holds the default voice, model and settings: `eleven_v3`, speed 1.15, stability 0.5.
An episode's `voice` overrides the default voice.

The account is on ElevenLabs' free plan, which only allows the premade voices through the API.
Voice Library voices return HTTP 402, and designing or cloning a voice returns HTTP 403 ("only
available on a paid plan"). On 2026-09-28 the premade voices scored 19–25% CER reading Mandarin on
`eleven_multilingual_v2` and 3–5% on `eleven_v3` in the Whisper listen-back. `l01-s4` uses "Laura"
(lively; the highest Whisper confidence of those tested). "Daniel" (steady broadcaster) was the first
pick. Test clips are in `D:\StudyNotes-video\out\voice-test\`.

Changing the voice re-bills every line once. Don't imitate a real person's voice (a voice actor, or a
meme voice cloned from one) without their permission. A cloned voice of your own, or a Voice Library
voice, needs a paid plan.

The free plan allows 10,000 credits a month, and one 14-minute episode costs about 3,700. On 2026-09-28
the month's quota ran out (4 credits left) after the episode had been voiced twice plus the voice tests.
A quota error names the credits left, and lines already synthesised stay cached. Audio made
on the free plan may not be used commercially. When it is published, "elevenlabs.io" or "11.ai" must
appear in the title
([ElevenLabs help](https://elevenlabs.io/docs/help-center/legal/can-i-publish-the-content-i-generate-on-the-platform)).
Starter ($6/month on 2026-09-28) adds a commercial license, 30,000 credits and instant voice cloning
([pricing](https://elevenlabs.io/pricing)).
