"""Tiny SVG writer shared by the lesson-0 figure scripts (standard library only; adapted from l19_svg.py).

Colours are always emitted as var(--fig-*, #fallback) so a figure renders on its own and follows the
lesson theme once inlined. Text accepts a small markup:

    *q*        bold
    _{t,i}     subscript
    ^{Q}       superscript

Run any l00_*.py script from anywhere; it writes ../l00-*.svg next to this folder.
"""

from __future__ import annotations

import os
import re
import sys

try:  # the scripts print Chinese labels; do not crash on a cp1252 console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

PALETTE = {
    "ink": "#1C1B19", "muted": "#74716A", "rule": "#DCD7CC", "grid": "#EDE9E0",
    "surface": "#FFFFFF", "blue": "#1F5FA8", "violet": "#6733C4", "green": "#16704A",
    "red": "#A3231C", "orange": "#B4460C", "amber": "#8A6508",
}
FONT = "system-ui, 'PingFang SC', 'Microsoft YaHei', sans-serif"
FIG_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC_DIR = os.path.normpath(os.path.join(FIG_DIR, "..", "src"))


def c(name: str) -> str:
    return "var(--fig-%s, %s)" % (name, PALETTE[name])


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(v: float) -> str:
    s = "%.2f" % v
    s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def text_width(s: str, size: float, weight: int = 400) -> float:
    """Rough advance width of a mixed CJK/Latin string in user units."""
    s = re.sub(r"\*|_\{|\^\{|\}", "", s)
    w = 0.0
    for ch in s:
        o = ord(ch)
        if o >= 0x2E80 or ch in "（）：，、；。“”":
            w += 1.0
        elif ch in ",.:;'|!":
            w += 0.28
        elif ch == " ":
            w += 0.28
        elif ch in "ijlrtf()[]/-·":
            w += 0.36
        elif ch.isupper() or ch in "mwMW+":
            w += 0.66
        else:
            w += 0.56
    return w * size * (1.04 if weight >= 600 else 1.0)


TOKEN = re.compile(r"(\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})")


def rich(s: str, size: float) -> str:
    small = max(11, round(size * 0.74))
    out = []
    for part in TOKEN.split(s):
        if not part:
            continue
        if part.startswith("_{") or part.startswith("^{"):
            shift = "sub" if part.startswith("_{") else "super"
            out.append('<tspan baseline-shift="%s" font-size="%s">%s</tspan>' % (shift, fmt(small), esc(part[2:-1])))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append('<tspan font-weight="700">%s</tspan>' % esc(part[1:-1]))
        else:
            out.append(esc(part))
    return "".join(out)


class SVG:
    def __init__(self, w: int, h: int, title: str = "", min_width: int | None = None):
        self.w, self.h = w, h
        self.title = title
        self.min_width = min_width
        self.parts: list[str] = []
        self.markers: set[str] = set()

    def raw(self, s: str):
        self.parts.append(s)

    def text(self, x, y, s, size=13, anchor="start", color="ink", weight=None, opacity=None, rotate=None,
             halo=False):
        attrs = ['x="%s"' % fmt(x), 'y="%s"' % fmt(y), 'font-size="%s"' % fmt(size), 'fill="%s"' % c(color)]
        if halo:  # a background-coloured outline so dashed guide lines behind the label do not cut through it
            attrs.append('stroke="%s" stroke-width="3.5" stroke-linejoin="round" paint-order="stroke"' % c("surface"))
        if anchor != "start":
            attrs.append('text-anchor="%s"' % anchor)
        if weight:
            attrs.append('font-weight="%s"' % weight)
        if opacity is not None:
            attrs.append('fill-opacity="%s"' % fmt(opacity))
        if rotate is not None:
            attrs.append('transform="rotate(%s %s %s)"' % (fmt(rotate), fmt(x), fmt(y)))
        self.parts.append("<text %s>%s</text>" % (" ".join(attrs), rich(s, size)))

    def line(self, x1, y1, x2, y2, color="ink", width=1.5, dash=None, opacity=None, arrow=False, cap=None):
        attrs = ['x1="%s" y1="%s" x2="%s" y2="%s"' % (fmt(x1), fmt(y1), fmt(x2), fmt(y2)),
                 'stroke="%s"' % c(color), 'stroke-width="%s"' % fmt(width)]
        if dash:
            attrs.append('stroke-dasharray="%s"' % dash)
        if opacity is not None:
            attrs.append('stroke-opacity="%s"' % fmt(opacity))
        if cap:
            attrs.append('stroke-linecap="%s"' % cap)
        if arrow:
            self.markers.add(color)
            attrs.append('marker-end="url(#ah-%s)"' % color)
        self.parts.append("<line %s/>" % " ".join(attrs))

    def rect(self, x, y, w, h, fill=None, fill_opacity=None, stroke=None, width=1.2, rx=0, dash=None,
             stroke_opacity=None):
        attrs = ['x="%s" y="%s" width="%s" height="%s"' % (fmt(x), fmt(y), fmt(w), fmt(h))]
        attrs.append('fill="%s"' % (c(fill) if fill else "none"))
        if fill and fill_opacity is not None:
            attrs.append('fill-opacity="%s"' % fmt(fill_opacity))
        if stroke:
            attrs.append('stroke="%s" stroke-width="%s"' % (c(stroke), fmt(width)))
            if stroke_opacity is not None:
                attrs.append('stroke-opacity="%s"' % fmt(stroke_opacity))
        if rx:
            attrs.append('rx="%s"' % fmt(rx))
        if dash:
            attrs.append('stroke-dasharray="%s"' % dash)
        self.parts.append("<rect %s/>" % " ".join(attrs))

    def circle(self, x, y, r, fill=None, stroke=None, width=1.2, fill_opacity=None, dash=None):
        attrs = ['cx="%s" cy="%s" r="%s"' % (fmt(x), fmt(y), fmt(r)), 'fill="%s"' % (c(fill) if fill else "none")]
        if fill and fill_opacity is not None:
            attrs.append('fill-opacity="%s"' % fmt(fill_opacity))
        if stroke:
            attrs.append('stroke="%s" stroke-width="%s"' % (c(stroke), fmt(width)))
        if dash:
            attrs.append('stroke-dasharray="%s"' % dash)
        self.parts.append("<circle %s/>" % " ".join(attrs))

    def path(self, d, stroke="ink", width=1.5, fill=None, fill_opacity=None, dash=None, arrow=False,
             opacity=None, join=None):
        attrs = ['d="%s"' % d, 'fill="%s"' % (c(fill) if fill else "none")]
        if fill and fill_opacity is not None:
            attrs.append('fill-opacity="%s"' % fmt(fill_opacity))
        if stroke:
            attrs.append('stroke="%s" stroke-width="%s"' % (c(stroke), fmt(width)))
        if dash:
            attrs.append('stroke-dasharray="%s"' % dash)
        if opacity is not None:
            attrs.append('stroke-opacity="%s"' % fmt(opacity))
        if join:
            attrs.append('stroke-linejoin="%s"' % join)
        if arrow:
            self.markers.add(stroke)
            attrs.append('marker-end="url(#ah-%s)"' % stroke)
        self.parts.append("<path %s/>" % " ".join(attrs))

    def polygon(self, pts, fill=None, fill_opacity=None, stroke=None, width=1.2):
        attrs = ['points="%s"' % " ".join("%s,%s" % (fmt(x), fmt(y)) for x, y in pts),
                 'fill="%s"' % (c(fill) if fill else "none")]
        if fill and fill_opacity is not None:
            attrs.append('fill-opacity="%s"' % fmt(fill_opacity))
        if stroke:
            attrs.append('stroke="%s" stroke-width="%s"' % (c(stroke), fmt(width)))
        self.parts.append("<polygon %s/>" % " ".join(attrs))

    def arrow(self, x1, y1, x2, y2, color="ink", width=1.6, dash=None):
        self.line(x1, y1, x2, y2, color=color, width=width, dash=dash, arrow=True)

    def box(self, x, y, w, h, lines, color="ink", fill_opacity=0.10, size=13, rx=6, text_color=None,
            weight=None, dash=None):
        self.rect(x, y, w, h, fill=color, fill_opacity=fill_opacity, stroke=color, width=1.3, rx=rx, dash=dash)
        if isinstance(lines, str):
            lines = [lines]
        gap = size * 1.35
        y0 = y + h / 2 - (len(lines) - 1) * gap / 2 + size * 0.36
        for i, ln in enumerate(lines):
            self.text(x + w / 2, y0 + i * gap, ln, size=size, anchor="middle", color=text_color or "ink",
                      weight=weight)

    def save(self, name: str):
        defs = []
        for col in sorted(self.markers):
            defs.append('<marker id="ah-%s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
                        'markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
                        '<path d="M0 0 L10 5 L0 10 z" fill="%s"/></marker>' % (col, c(col)))
        style = ' style="min-width:%dpx"' % self.min_width if self.min_width else ""
        head = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="%s"%s>'
                % (self.w, self.h, FONT, style))
        title = "<title>%s</title>" % esc(self.title) if self.title else ""
        svg = "%s\n%s\n%s\n%s\n</svg>\n" % (head, title, "<defs>%s</defs>" % "".join(defs) if defs else "",
                                            "\n".join(self.parts))
        path = os.path.join(FIG_DIR, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg)
        print("wrote", os.path.relpath(path))
        return path
