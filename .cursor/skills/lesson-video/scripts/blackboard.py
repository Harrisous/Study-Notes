"""Blackboard kit for lesson videos (Manim Community Edition).

An episode is a Python file of `Segment` scenes, one per chapter. Each segment carries a level
badge (ZERO = anyone can follow, ENG = needs engineering background) and a title, and its
`body()` interleaves narration with board work:

    class S01Coins(Segment):
        level, title = ZERO, "抛 64 次硬币"
        def body(self):
            with self.say("抛 64 次硬币，{a}加起来通常是多少？") as v:
                q = self.put(T("64 次硬币之和 = ?"))
                v.until("a")
                self.play(Write(q))

`say()` starts the narration clip at the current scene time, `v.until(mark)` waits for a
bookmark, and leaving the block waits for the clip to end. Clip start times and texts, layout
warnings and contact-sheet times go to <BB_OUT>/<Scene>.timeline.json; make_video.py mixes the
audio and cuts the subtitles from them.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from manim import *  # noqa: F401,F403  (episode files do `from blackboard import *`)

import tts

# ---- chalk palette; hues follow the course's figure semantics (style-guide.md §4) ----
BOARD = "#1E3328"
INK = "#F1EFE6"      # plain chalk: text, lines
MUTED = "#8FA79A"    # faded chalk: scaffolding, grids, "earlier" work
Q_BLUE = "#8EC5F0"   # query / current token
K_VIOLET = "#C9ABF2" # key / scores
V_GREEN = "#A6DB9A"  # value / output / good
AMBER = "#F2CC6B"    # cache / memory; also the general highlight
ORANGE = "#F5A66B"   # compressed / low precision
RED = "#F08A8A"      # masked / discarded / wrong
HILITE = AMBER

FONT = os.environ.get("BB_FONT", "KaiTi")
CODE_FONT = "Consolas"
MATH_FONT_SIZE = 46
ZERO, ENG = "zero", "eng"
LEVELS = {ZERO: ("零基础", V_GREEN), ENG: ("工程师向", ORANGE)}

# Safe area in scene units (frame is 14.22 x 8). The strip below SAFE_BOTTOM is kept for subtitles.
SAFE_LEFT, SAFE_RIGHT, SAFE_TOP, SAFE_BOTTOM = -6.75, 6.75, 3.05, -2.75
HEADER_Y = 3.55

config.background_color = BOARD
TexTemplateLibrary.default.add_to_preamble(r"\usepackage{amsmath,amssymb}")


# ---------------------------------------------------------------- typography
def T(text: str, size: float = 40, color=INK, **kw) -> Text:
    """Chalk text in the board font (Chinese and Latin)."""
    kw.setdefault("line_spacing", 0.9)
    return Text(text, font=FONT, font_size=size, color=color, **kw)


def M(*tex: str, size: float = MATH_FONT_SIZE, color=INK, **kw) -> MathTex:
    """Chalk maths. Several strings make separately addressable parts (m[0], m[1], ...)."""
    return MathTex(*tex, font_size=size, color=color, **kw)


_INLINE = re.compile(r"\$([^$]+)\$")


def L(text: str, size: float = 38, color=INK, math_color=None, buff: float = 0.12) -> VGroup:
    """One line mixing text and $maths$, e.g. L("除以 $\\sqrt{d_k}$ 之后，方差回到 1")."""
    parts, pos = [], 0
    for m in _INLINE.finditer(text):
        if m.start() > pos and text[pos:m.start()].strip():
            parts.append(T(text[pos:m.start()].strip(), size, color))
        parts.append(M(m.group(1), size=size * 1.15, color=math_color or color))
        pos = m.end()
    if text[pos:].strip():
        parts.append(T(text[pos:].strip(), size, color))
    return VGroup(*parts).arrange(RIGHT, buff=buff)


def code(src: str, size: float = 26, color=INK) -> VGroup:
    """A code snippet in a faint box."""
    txt = Text(src, font=CODE_FONT, font_size=size, color=color)
    box = RoundedRectangle(width=txt.width + 0.5, height=txt.height + 0.35, corner_radius=0.1, stroke_color=MUTED,
                           stroke_width=1.5, fill_color=BLACK, fill_opacity=0.18)
    return VGroup(box, txt.move_to(box))


def why(mob: Mobject, reason: str, x: float = 2.6, size: float = 26, color=MUTED) -> VGroup:
    """A derivation step's reason (text with optional $maths$), small, at column x on mob's line."""
    r = L(reason, size, color)
    r.move_to([x + r.width / 2, mob.get_center()[1], 0])
    return r


def bars(values, labels=None, colors=None, max_h: float = 2.0, bar_w: float = 0.55, gap: float = 0.3,
         fmt: str = "{:.2f}", size: float = 26, vmax: float | None = None, texts=None) -> VGroup:
    """Vertical bars on a baseline with value labels above and names below.
    Returns VGroup(bars, values, names, baseline); index it as g[0][i], g[1][i], g[2][i]."""
    vmax = vmax or max(values) or 1
    colors = colors or [INK] * len(values)
    if isinstance(colors, str):
        colors = [colors] * len(values)
    bs, vs, ns = VGroup(), VGroup(), VGroup()
    for i, v in enumerate(values):
        x = i * (bar_w + gap)
        hgt = max(0.02, max_h * v / vmax)
        b = Rectangle(width=bar_w, height=hgt, stroke_width=2, stroke_color=colors[i], fill_color=colors[i],
                      fill_opacity=0.55).move_to([x, hgt / 2, 0])
        bs.add(b)
        vs.add(T(texts[i] if texts else fmt.format(v), size, colors[i]).next_to(b, UP, buff=0.08))
        if labels:
            ns.add(T(labels[i], size, INK).move_to([x, -0.28, 0]))
    base = Line([-bar_w * 0.9, 0, 0], [(len(values) - 1) * (bar_w + gap) + bar_w * 0.9, 0, 0], color=MUTED,
                stroke_width=2)
    return VGroup(bs, vs, ns, base)


def badge(level: str, size: float = 24) -> VGroup:
    """The level tag drawn in each segment's header (零基础 / 工程师向)."""
    name, color = LEVELS[level]
    t = T(name, size, color)
    box = RoundedRectangle(width=t.width + 0.4, height=t.height + 0.24, corner_radius=0.12, stroke_color=color,
                           stroke_width=2.5)
    return VGroup(box, t.move_to(box))


def chalk_box(mob: Mobject, color=HILITE, buff: float = 0.15, **kw) -> SurroundingRectangle:
    return SurroundingRectangle(mob, color=color, buff=buff, corner_radius=0.08, stroke_width=3, **kw)


def chalk_underline(mob: Mobject, color=HILITE, buff: float = 0.08) -> Line:
    return Line(mob.get_corner(DL) + DOWN * buff, mob.get_corner(DR) + DOWN * buff, color=color, stroke_width=4)


def table(rows: list[list[str]], col_colors=None, size: float = 30, h_buff: float = 0.55, v_buff: float = 0.22,
          header_color=MUTED) -> VGroup:
    """A light chalk table: cells may be plain text or $maths$. Returns a VGroup of rows of cells."""
    grid = [[L(c, size, header_color if r == 0 else (col_colors[j] if col_colors else INK)) for j, c in enumerate(row)]
            for r, row in enumerate(rows)]
    widths = [max(grid[r][j].width for r in range(len(grid))) for j in range(len(grid[0]))]
    heights = [max(c.height for c in row) for row in grid]
    out, y = VGroup(), 0.0
    for r, row in enumerate(grid):
        x, cells = 0.0, VGroup()
        for j, cell in enumerate(row):
            cell.move_to([x + widths[j] / 2, y - heights[r] / 2, 0])
            cells.add(cell)
            x += widths[j] + h_buff
        out.add(cells)
        y -= heights[r] + v_buff
    rule = Line(out[0].get_corner(DL) + LEFT * 0.2 + DOWN * v_buff / 2,
                out[0].get_corner(DR) + RIGHT * 0.2 + DOWN * v_buff / 2, color=MUTED, stroke_width=2)
    out.add(rule)
    return out


# ---------------------------------------------------------------- narration
class Voice:
    """Context manager returned by Segment.say(); see the module docstring."""

    def __init__(self, scene: "Segment", text: str, pause: float, **tts_kw):
        self.scene, self.text, self.pause = scene, text, pause
        self.line = tts.parse(text)
        self.meta = tts.synth(text, **tts_kw)

    def __enter__(self) -> "Voice":
        s = self.scene
        self.t0 = s.renderer.time
        # subtitles are cut from text + timing at assembly time, so they can change without a re-render
        s._timeline["clips"].append({"mp3": self.meta["mp3"], "t": self.t0, "dur": self.meta["duration"],
                                     "text": self.text})
        return self

    def at(self, mark: str) -> float:
        return self.t0 + tts.time_of(self.meta, self.line.marks[mark])

    def until(self, mark: str) -> None:
        dt = self.at(mark) - self.scene.renderer.time
        if dt > 1 / 60:
            self.scene.wait(dt)

    def left(self) -> float:
        return max(0.0, self.t0 + self.meta["duration"] - self.scene.renderer.time)

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            dt = self.left() + self.pause
            if dt > 1 / 60:
                self.scene.wait(dt)


# ---------------------------------------------------------------- scenes
class Segment(Scene):
    level = ZERO
    title = ""
    number = 0          # filled from the class name S01..., or set explicitly

    # layout cursor for put(): next line goes below the previous one, left-aligned
    def setup(self):
        self._timeline = {"scene": type(self).__name__, "level": self.level, "title": self.title,
                          "clips": [], "warnings": [], "snaps": []}
        self._cursor = None
        self._frame_items: set[int] = set()
        n = re.match(r"S(\d+)", type(self).__name__)
        self.number = self.number or (int(n.group(1)) if n else 0)

    def construct(self):
        self.header()
        self.body()
        self.wait(0.5)
        self.snap()
        self._dump()

    def body(self):
        raise NotImplementedError

    # ---- narration
    def say(self, text: str, pause: float = 0.25, **tts_kw) -> Voice:
        return Voice(self, text, pause, **tts_kw)

    def narrate(self, text: str, pause: float = 0.25, **tts_kw) -> None:
        """Narration over a board that does not change."""
        with self.say(text, pause, **tts_kw):
            pass

    # ---- chrome
    def header(self):
        name, color = LEVELS[self.level]
        tag_box = badge(self.level).to_corner(UR, buff=0.3).set_y(HEADER_Y)
        badge_t = tag_box[1]
        num = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"[self.number - 1] if 1 <= self.number <= 20 else ""
        big = T(f"{num} {self.title}".strip(), 60, INK)
        if big.width > 12.4:
            big.scale_to_fit_width(12.4)
        tag = T(name, 32, color).next_to(big, DOWN, buff=0.35)
        self.play(Write(big), FadeIn(tag), run_time=0.8)
        self.wait(0.5)
        small = T(f"{num} {self.title}".strip(), 26, MUTED).to_corner(UL, buff=0.3).set_y(HEADER_Y)
        rule = Line([SAFE_LEFT, HEADER_Y - 0.32, 0], [SAFE_RIGHT, HEADER_Y - 0.32, 0], color=MUTED, stroke_width=1.5,
                    stroke_opacity=0.5)
        self.chrome = VGroup(small, tag_box, rule)
        self._frame_items |= {id(m) for m in [*self.chrome.get_family(), big, tag]}
        self.play(ReplacementTransform(big, small), ReplacementTransform(tag, badge_t), Create(tag_box[0]),
                  Create(rule), run_time=0.7)

    # ---- layout helpers
    def put(self, mob: Mobject, gap: float = 0.32, x: float | None = None, first_y: float = SAFE_TOP - 0.1,
            center: bool = False) -> Mobject:
        """Position mob as the next board line (left-aligned at x, or centred). Does not animate."""
        top = first_y if self._cursor is None else self._cursor - gap
        mob.move_to([0, 0, 0])
        mob.shift(UP * (top - mob.get_top()[1]))
        if center:
            mob.set_x(0 if x is None else x)
        else:
            mob.shift(RIGHT * ((SAFE_LEFT + 0.25 if x is None else x) - mob.get_left()[0]))
        self._cursor = mob.get_bottom()[1]
        return mob

    def reset_cursor(self, y: float | None = None):
        self._cursor = y

    def wipe(self, *mobs: Mobject, run_time: float = 0.6):
        """Erase board content (everything but the header when called without arguments)."""
        targets = list(mobs) or [m for m in self.mobjects if id(m) not in self._frame_items]
        if not mobs:
            self.snap()
        if targets:
            self.play(*[FadeOut(m, shift=UP * 0.15) for m in targets], run_time=run_time)
        self._cursor = None

    def snap(self):
        """Mark the current board for the contact sheet (wipe() and the segment end do this too)."""
        self._timeline["snaps"].append(round(self.renderer.time - 0.05, 3))

    # ---- layout checks (warnings only; make_video.py prints them)
    def _check(self):
        t = round(self.renderer.time, 2)
        for m in self.mobjects:
            if id(m) in self._frame_items or not isinstance(m, VMobject) or m.width < 1e-3:
                continue
            l, r = m.get_left()[0], m.get_right()[0]
            b, top = m.get_bottom()[1], m.get_top()[1]
            if l < SAFE_LEFT - 0.05 or r > SAFE_RIGHT + 0.05 or top > SAFE_TOP + 0.35 or b < SAFE_BOTTOM - 0.05:
                self._warn(t, f"outside safe area: {type(m).__name__} x[{l:.2f},{r:.2f}] y[{b:.2f},{top:.2f}]"
                              f" {self._label(m)}")

    @staticmethod
    def _label(m: Mobject) -> str:
        txt = getattr(m, "text", None) or getattr(m, "tex_string", None) or ""
        if not txt and len(m.submobjects):
            for s in m.submobjects:
                txt = getattr(s, "text", None) or getattr(s, "tex_string", None) or ""
                if txt:
                    break
        return repr(txt[:24])

    def _warn(self, t, msg):
        w = self._timeline["warnings"]
        if not any(x["msg"] == msg for x in w):
            w.append({"t": t, "msg": msg})

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        self._check()

    def _dump(self):
        out = Path(os.environ.get("BB_OUT", "."))
        out.mkdir(parents=True, exist_ok=True)
        self._timeline["duration"] = round(self.renderer.time, 3)
        (out / f"{type(self).__name__}.timeline.json").write_text(
            json.dumps(self._timeline, ensure_ascii=False, indent=1), encoding="utf-8")
