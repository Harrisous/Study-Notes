---
name: lesson-video
description: Turns a lesson page into a narrated 16:9 explainer video, either Manim CE + ElevenLabs (*/video/*.py) or Remotion (React) + a local Kokoro voice (*/video/*-js/), with per-segment difficulty badges (零基础 / 工程师向), burned-in Chinese subtitles, an SRT file and a chapter list. JS episodes open with a zoom from the course map down to the page and its questions. Use when writing or editing an episode, its storyboard or narration, changing the voice or style, or building, checking or re-rendering a lesson video.
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

## Second pipeline: JavaScript picture, local voice

`l01-s4-js` is the same lesson page as a roughly 11-minute cut: a 66-second opening on the course map, then nine
segments. The picture is written in React and rendered by Remotion, and the voice is Kokoro running locally, so
nothing is billed. Its style is
"科技极简 + a little cyberpunk", approved on 2026-09-29 on the
[style canvas](https://claude.ai/artifact/4xaeYdbBf4sspP6Rxw6uLU):
- Colours: a grey scale plus one accent (`#00E5FF`).
- Effects: glow on the accent only, a magenta/cyan RGB split on big titles, HUD corners and
  readouts, a perspective grid on the cover, and a 3-frame glitch when a chapter starts. There are no
  scanlines.
- Formulas: Latin Modern, typeset by LaTeX.

| Piece | Where |
|---|---|
| Episode | `deepseek/video/<slug>-js/`: `script.json` (narration with `{mark}` and `[shown\|spoken]`, chapters with level and the question `q` they answer, `questions`, `roadmap`), `formulas.txt` (`id \| LaTeX`, parts as `\p{name}{...}`), `intro.tsx` (the opening), `boards*.tsx` (one board per chapter), `index.tsx` (entry and sound cues), `storyboard.md` |
| Kit | `js/kit/`: `core.tsx` (tokens, motion `ease`/`swing`/`up`/`focus`, `useBoard()` with `at('l3.a')` = frame of a bookmark, `Tex` with parts, `useTexBox`), `viz.tsx` (`Svg`, `Axes`, `Arc`, `Cells`, `Slider`, `Brace`, `Check`, `Leader`, `drawOn`, `curvePts`, `softmax`, `coinSum`), `roadmap.tsx` (camera, `Tree`, `Root`, `NodeTag`, `Minimap`), `opening.tsx` (`Question`, `ChapterMap`, `Breadcrumb`, `PageTabs`), `syllabus.tsx` (the course syllabus as a world: `syllabusLayout`, `Syllabus`, `fit`, `nest`, `pageTabs`, `tabPreview`), `index.tsx` (chapter card, chrome, `makeEpisode`) |
| Lesson 0 | `deepseek/video/l00-shared/`: `scene.tsx` (`Opening`: syllabus → lesson card → page tab → the page's world, one shot; `TreeScene`, `laneRect`, `FULLV`), `chain.tsx` (`Chain`: the improvement-chain board that walks one lane), `charts.tsx` (`LogBars`, `Span`), `sfx.ts` |
| Driver | `scripts/make_js_video.py`, run with `D:\StudyNotes-video\tts-venv\Scripts\python.exe` |
| Workspace | `D:\StudyNotes-video\remotion`: `node_modules`, `gen/<slug>/` (timeline.json, math.json), `public/` (voice clips, fonts), `stills.mjs` |

```bash
PY=D:/StudyNotes-video/tts-venv/Scripts/python.exe
S=.cursor/skills/lesson-video/scripts
$PY $S/make_js_video.py tts deepseek/video/l01-s4-js        # Kokoro per sentence, cached; ~4 min for 41 lines
$PY $S/make_js_video.py math deepseek/video/l01-s4-js       # formulas.txt -> path-only SVG, with part boxes
$PY $S/make_js_video.py roadmap deepseek/video/l01-s4-js    # the course map for the opening (from the lesson-0 figure data)
$PY $S/make_js_video.py sfx deepseek/video/l01-s4-js        # the synthesised sound palette (drone, whoosh, tick, ping, glitch)
$PY $S/make_js_video.py timeline deepseek/video/l01-s4-js   # the one timeline for picture, sound and subtitles
$PY $S/make_js_video.py sheet deepseek/video/l01-s4-js      # one tile per board state; 43 stills in ~20 s
$PY $S/make_js_video.py render deepseek/video/l01-s4-js     # 1080p30 mp4
$PY $S/make_js_video.py subs deepseek/video/l01-s4-js       # .srt and chapters.txt
$PY $S/make_js_video.py qa deepseek/video/l01-s4-js         # Whisper listen-back on the GPU
```

- **Timeline.** `timeline` turns clip lengths into frames. Each chapter gets a 2.2 s card, a 0.5 s
  lead-in, 0.3 s between lines, an optional `hold` after a line, and a 1 s tail. Everything reads
  that file: boards, audio placement, burned-in subtitles and the `.srt`. A narration edit therefore
  moves every later frame; re-run `timeline` and render again.
- **Bookmarks.** Kokoro gives no per-character timing. Each sentence is synthesised separately, so
  sentence boundaries are exact. Positions inside a sentence are estimated by character count, with
  commas weighted 1.6. Put bookmarks at punctuation when timing matters.
- **Layout loop.** Change a board, then run `sheet --only 03` (or `still --at 12.5,40`) and look at
  the PNGs. Stills do not need the audio, so they are quick. The sheet starts with one tile per opening line.
- **Subtitles.** Cues are cut at punctuation and measured by visual width (a Latin letter or digit counts as
  0.55 of a Chinese character), so "从 2017 年的 Transformer" stays one cue. A list of numbers joined by 、
  ("0.66、0.24、0.10") is never cut, and a short sentence tail stays with its line.
- **Polyphones.** Kokoro guesses readings: 行 (row) came out xíng, 调一调 diào, 为零 wèi. Spell the reading
  with a pair, `[行|航]`, `[调一调|条一条]`, `[为|围]`, and confirm with `qa` (Whisper writes homophones such
  as 头/投 or 乘积/乘机 either way, so read flagged lines before rewording).
- **SVG filters on straight lines.** A filter's region follows the element's bounding box, so a perfectly
  horizontal or vertical `<line>` with `filter="url(#vglow)"` has a zero-height box and disappears. Leave the
  glow off such lines, or draw them as rects.
- **Long chapter titles.** The chapter card sets the title at 104 px next to the big outlined number; keep titles
  to about ten Chinese characters, or the title runs into the number.
- **Remotion quirks.** The entry file must contain the literal `registerRoot(` call. Repo files
  resolve `react` and `remotion` through the workspace's `remotion.config.ts` and `stills.mjs`
  overrides. Geist Mono has a slashed zero, so large numerals use Noto Sans SC.
- **Voice.** Kokoro-82M-v1.1-zh is Apache-2.0 and has 103 voices (`zm_*` male, `zf_*` female). The
  default `zm_010` tied for the best Whisper confidence of four male voices. The voice read Mandarin
  at about 4.7% CER, level with ElevenLabs v3. Keep `speak_numbers: "zh"`. Watch for homophones the
  listener will hear: "分布太尖" was heard as "太监".
- **The opening (every episode).** One continuous shot from the whole course to the page, so viewers know
  where they are: the tech tree (lanes, nodes and dates from `deepseek/figures/gen/l00_tech_tree.py`), then the
  lesson's root diagram with links to what later replaces each box, then the box that opens into the page's
  formula, whose parts carry the page's questions, then a chapter map grouped by question. The camera flies with
  `camAt`/`zoomPath` (van Wijk–Nuij, as d3.interpolateZoom), and the worlds are drawn in screen space each frame
  so text stays crisp and switches detail with zoom. A breadcrumb (top left) and a minimap (top right) say where
  the viewer is. For a new lesson: set `roadmap` (course, lesson, page) and `questions` in `script.json`, give
  each chapter its `q`, draw the lesson's root diagram and its links in the episode's `intro.tsx`, and write the
  shots into `storyboard.md` first. The last chapter answers the questions and backs out to the map
  (`RoadmapView`), lighting the next page.
  Lesson 0 uses its own opening (`l00-shared/scene.tsx`): it starts on the syllabus, lights the lessons the page
  draws on (`lit`), opens the lesson-0 card into its page tabs, and dives through the page's tab window into the
  page's world (the tech tree), whose camera (`destView`) can then push to one lane. Zooming into a low lane brings
  the top lanes under the breadcrumb, so dim the other lanes after the zoom (`nodeIn`/`laneIn`).
- **Formula parts.** `\p{name}{...}` in `formulas.txt` typesets that piece in a marker colour. `math`
  turns it into a CSS hook and records its box, so a board can light it as the voice reaches it
  (`<Tex id parts={{name: lit()}}/>`) and point at it (`useTexBox()(id, size, 'name')`).
- **Narration polish.** Before synthesis, pass the narration through the 说人话 rules (shuorenhua v2.5.1, MIT,
  runtime files in `D:\StudyNotes-video	ools\shuorenhua\`). Keep facts, numbers, conditions and hedges
  (几乎总是, 大约, 我们猜测), drop pure signposting (要说明一点, 现在…), and add no new claims.
- **Sound.** `sfx` synthesises five quiet sounds with numpy, so there are no licences to track. The episode places them (drone under
  the opening, whooshes on camera moves, ticks, pings on questions and check marks, a glitch on chapter
  cards) about 20 dB under the voice. Nothing plays under explanation.
- **Rebuilding the workspace.** `js/workspace/` holds copies of the workspace's `package.json`,
  `remotion.config.ts`, `tsconfig.json` and `stills.mjs`. To rebuild the workspace, copy them to
  `D:\StudyNotes-video\remotion` and run `npm install remotion @remotion/cli @remotion/media-utils react react-dom`
  there (4.0.530 on 2026-09-29). Then put Geist Mono in `public/fonts/`.
- **Loudness.** `render` post-processes with ffmpeg: loudnorm brings the audio to -16 LUFS (Kokoro comes
  out near -28), the chapters are embedded, and the picture is copied. An 8.5-minute episode renders in
  about 9 minutes with `-j 10`.

### Other voices: `--voice`

`make_js_video.py <step> <episode> --voice <spec>` builds a separate variant. It has its own slug
(`<slug>-<voice>`), gen dir and output, so the default Kokoro version stays untouched. The variant's
timeline reaches Remotion as input props, so `index.tsx` needs no change.
- `--voice zf_001`: another Kokoro voice.
- `--voice gsv:<name>`: GPT-SoVITS with the weights in `D:\StudyNotes-video\voices\<name>\` (`voice.json`
  lists the sovits/gpt weights and the reference clip with its text). The engine is the official
  Windows package `GPT-SoVITS-v2pro-20250604`, unpacked to `D:\StudyNotes-video\gpt-sovits\`. The script
  starts its `api.py` on 127.0.0.1:9880 when needed and stops it on exit. Each sentence takes about
  4 s on the RTX 4080 SUPER.
- **English in a Chinese-only voice.** The 曼波 weights were fine-tuned on Chinese speech only, so English sent
  through GPT-SoVITS's English frontend comes out garbled. `scripts/translit.py` rewrites every Latin token the
  way a Mandarin speaker reads it (GQA → 吉丘诶, token → 托肯, H800 → 艾尺八百) before synthesis; subtitles keep the
  English. Unknown words are spelled letter by letter and listed by `translit.unknown()`: preview a new script
  with `to_spoken()` and add real words to `WORDS` before running `tts`. Avoid symbols the voice cannot read
  (`log2(a/448)`, `@`): say them in words or use a `[shown|spoken]` pair.
- **Retakes.** Every GSV sentence is scored by Whisper (CER against the spoken form). Sentences above 12% get two
  more takes, and the clearest take is kept (cached under `work/takes/`). Run one GSV job at a time: each job
  stops the shared engine on :9880 when it exits.

`voices/mambo` holds the 曼波 voice from the MamboTTS release. Harry asked for it for **personal
viewing only** (2026-10-01). It imitates a real voice actress, so its audio and videos stay on this
machine: no Artifact page, nothing committed, nothing published.

How third-party voice packages were vetted (2026-10-01; repeat this for any new one):
1. Download into `D:\StudyNotes-video\quarantine\` and execute nothing from it.
2. Check hashes against the official source. The engine 7z matched `lj1995/GPT-SoVITS-windows-package`
   on HuggingFace, and the bundled 7zr.exe matched 7-Zip 26.02 from `ip7z/7zip`.
3. Run a Defender custom scan (`MpCmdRun -Scan -ScanType 3 -File <dir>`) on the download and again on the
   unpacked tree. The only hit in the official engine was `PUA:Win32/FRProxy`: Gradio's
   `frpc_windows_amd64_v0.2` tunnel for share links. It is not used here and was deleted.
4. Pickle-scan the weights (`picklescan`, or list the `GLOBAL`s with pickletools). The mambo weights
   reference only tensor rebuilders, `OrderedDict` and GPT-SoVITS's `utils.HParams`.
5. Grep the client scripts for obfuscation, persistence and unexpected hosts. Run only the official
   engine and our own code; never the package's `.bat`/`.exe` files.
