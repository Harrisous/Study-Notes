"""Tiny SVG writer shared by the lesson-13 figure scripts (standard library only).

Colours are always emitted as var(--fig-*, #fallback) so a figure renders on its own and
follows the lesson theme once inlined. Text accepts a small markup:

    *q*        bold
    _{t,i}     subscript
    ^{Q}       superscript

Run any l13_*.py script from anywhere; it writes ../l13-*.svg next to this folder.
"""

from __future__ import annotations

import os
import re

PALETTE = {
    "ink": "#1C1B19", "muted": "#74716A", "rule": "#DCD7CC", "grid": "#EDE9E0",
    "surface": "#FFFFFF", "blue": "#1F5FA8", "violet": "#6733C4", "green": "#16704A",
    "red": "#A3231C", "orange": "#B4460C", "amber": "#8A6508",
}
FONT = "system-ui, 'PingFang SC', 'Microsoft YaHei', sans-serif"
FIG_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def c(name: str) -> str:
    return "var(--fig-%s, %s)" % (name, PALETTE[name])


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(v: float) -> str:
    s = "%.2f" % v
    s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


TOKEN = re.compile(r"(\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})")


def plain(s: str) -> str:
    """Strip the mini markup (for width estimates)."""
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"[_^]\{([^}]*)\}", r"\1", s)
    return s


def tw(s: str, size: float) -> float:
    """Rough rendered width of a label: CJK and full-width ~1 em, Latin ~0.56 em."""
    w = 0.0
    for ch in plain(s):
        o = ord(ch)
        if 0x2E80 <= o <= 0x9FFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F:
            w += 1.0
        elif ch in " .,:;'|!il":
            w += 0.3
        elif ch in "mwMW—":
            w += 0.85
        else:
            w += 0.57
    return w * size


def rich(s: str, size: float) -> str:
    small = max(11, round(size * 0.74))
    out = []
    for part in TOKEN.split(s):
        if not part:
            continue
        if part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append('<tspan font-weight="700">%s</tspan>' % esc(part[1:-1]))
        elif part.startswith("_{"):
            out.append('<tspan baseline-shift="%s" font-size="%s">%s</tspan>'
                       % (fmt(-0.22 * size), fmt(small), esc(part[2:-1])))
        elif part.startswith("^{"):
            out.append('<tspan baseline-shift="%s" font-size="%s">%s</tspan>'
                       % (fmt(0.36 * size), fmt(small), esc(part[2:-1])))
        else:
            out.append(esc(part))
    return "".join(out)


class SVG:
    def __init__(self, w: int, h: int, title: str = ""):
        self.w, self.h = w, h
        self.title = title
        self.parts: list[str] = []
        self.markers: set[str] = set()

    def raw(self, s: str):
        self.parts.append(s)

    def text(self, x, y, s, size=13, anchor="start", color="ink", weight=None, italic=False,
             opacity=None, rotate=None, family=None):
        attrs = ['x="%s"' % fmt(x), 'y="%s"' % fmt(y), 'font-size="%s"' % fmt(size),
                 'fill="%s"' % c(color)]
        if anchor != "start":
            attrs.append('text-anchor="%s"' % anchor)
        if weight:
            attrs.append('font-weight="%s"' % weight)
        if italic:
            attrs.append('font-style="italic"')
        if family:
            attrs.append('font-family="%s"' % family)
        if opacity is not None:
            attrs.append('fill-opacity="%s"' % fmt(opacity))
        if rotate is not None:
            attrs.append('transform="rotate(%s %s %s)"' % (fmt(rotate), fmt(x), fmt(y)))
        self.parts.append("<text %s>%s</text>" % (" ".join(attrs), rich(s, size)))

    def mono(self, x, y, s, size=12, color="ink", anchor="start", weight=None):
        self.text(x, y, s, size=size, color=color, anchor=anchor, weight=weight,
                  family="ui-monospace, 'Cascadia Mono', Consolas, monospace")

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

    def arrow(self, x1, y1, x2, y2, color="ink", width=1.6, dash=None, shorten=0.0):
        if shorten:
            dx, dy = x2 - x1, y2 - y1
            n = (dx * dx + dy * dy) ** 0.5 or 1.0
            x2 -= dx / n * shorten
            y2 -= dy / n * shorten
        self.line(x1, y1, x2, y2, color=color, width=width, dash=dash, arrow=True)

    def rect(self, x, y, w, h, fill=None, fill_opacity=None, stroke=None, width=1.2, rx=0,
             dash=None, stroke_opacity=None):
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
        attrs = ['cx="%s" cy="%s" r="%s"' % (fmt(x), fmt(y), fmt(r)),
                 'fill="%s"' % (c(fill) if fill else "none")]
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

    def polyline(self, pts, stroke="ink", width=1.5, dash=None, arrow=False, fill=None, fill_opacity=None):
        d = "M" + " L".join("%s %s" % (fmt(x), fmt(y)) for x, y in pts)
        self.path(d, stroke=stroke, width=width, dash=dash, arrow=arrow, fill=fill,
                  fill_opacity=fill_opacity, join="round")

    def box(self, x, y, w, h, color="ink", lines=(), size=13, fill_opacity=0.10, stroke_w=1.4, rx=8,
            title_weight="600", dash=None, line_gap=None, sub_color="ink", sub_size=None, fill=True):
        """Rounded box with centred lines; first line bold in the box colour, rest in sub_color."""
        self.rect(x, y, w, h, fill=color if fill else None, fill_opacity=fill_opacity if fill else None,
                  stroke=color, width=stroke_w, rx=rx, dash=dash)
        lines = list(lines)
        if not lines:
            return
        sub_size = sub_size or max(12, size - 1)
        gap = line_gap or (size + 4)
        total = size + (len(lines) - 1) * gap
        y0 = y + (h - total) / 2 + size * 0.86
        for i, s in enumerate(lines):
            if i == 0:
                self.text(x + w / 2, y0, s, size=size, anchor="middle", color=color if color != "muted" else "ink",
                          weight=title_weight)
            else:
                self.text(x + w / 2, y0 + i * gap, s, size=sub_size, anchor="middle", color=sub_color)

    def check(self, x, y, s=7, color="green", width=2.2):
        self.path("M%s %s l%s %s l%s %s" % (fmt(x - s), fmt(y), fmt(s * 0.7), fmt(s * 0.7), fmt(s * 1.3),
                                           fmt(-s * 1.4)), stroke=color, width=width, join="round")

    def cross(self, x, y, s=6, color="red", width=2.2):
        self.line(x - s, y - s, x + s, y + s, color=color, width=width, cap="round")
        self.line(x - s, y + s, x + s, y - s, color=color, width=width, cap="round")

    def save(self, name: str):
        defs = []
        for col in sorted(self.markers):
            defs.append('<marker id="ah-%s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" '
                        'markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
                        '<path d="M0 0 L10 5 L0 10 z" fill="%s"/></marker>' % (col, c(col)))
        head = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="%s">'
                % (self.w, self.h, FONT))
        title = "<title>%s</title>" % esc(self.title) if self.title else ""
        body = "\n".join(self.parts)
        svg = "%s\n%s\n%s\n%s\n</svg>\n" % (head, title, "<defs>%s</defs>" % "".join(defs) if defs else "", body)
        path = os.path.join(FIG_DIR, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg)
        print("wrote", os.path.relpath(path))
        return path


class Axes:
    """Linear or log axes mapping data to a pixel rectangle."""

    def __init__(self, x0, x1, y0, y1, xlim, ylim, xlog=False, ylog=False):
        import math
        self.m = math
        self.X0, self.X1, self.Y0, self.Y1 = x0, x1, y0, y1
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    def _t(self, v, lim, log):
        if log:
            return (self.m.log10(v) - self.m.log10(lim[0])) / (self.m.log10(lim[1]) - self.m.log10(lim[0]))
        return (v - lim[0]) / (lim[1] - lim[0])

    def px(self, v):
        return self.X0 + self._t(v, self.xlim, self.xlog) * (self.X1 - self.X0)

    def py(self, v):
        return self.Y1 - self._t(v, self.ylim, self.ylog) * (self.Y1 - self.Y0)
