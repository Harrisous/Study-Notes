"""Tiny SVG writer for the M04 figure scripts (standard library + numpy only).

A private copy of the M01 helper (so other lessons can change theirs freely) plus a small
Axes class for data plots. Colours are always emitted as var(--fig-*, #fallback), so each
figure renders on its own and follows the lesson theme once inlined. Text accepts a small
markup:

    *q*        bold (vectors)
    _{t,i}     subscript
    ^{Q}       superscript

Run any m04_*.py script from anywhere; it writes ../m04-*.svg next to this folder.
"""

from __future__ import annotations

import math
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


def rich(s: str, size: float) -> str:
    """Turn the mini markup into tspans. Sub/superscripts never go below 11 units."""
    small = max(11, round(size * 0.74))
    out = []
    for part in TOKEN.split(s):
        if not part:
            continue
        if part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append('<tspan font-weight="700">%s</tspan>' % esc(part[1:-1]))
        elif part.startswith("_{"):
            out.append('<tspan baseline-shift="sub" font-size="%s">%s</tspan>'
                       % (fmt(small), rich_inner(part[2:-1])))
        elif part.startswith("^{"):
            out.append('<tspan baseline-shift="super" font-size="%s">%s</tspan>'
                       % (fmt(small), rich_inner(part[2:-1])))
        else:
            out.append(esc(part))
    return "".join(out)


def rich_inner(s: str) -> str:
    return re.sub(r"\*([^*]+)\*", lambda m: '<tspan font-weight="700">%s</tspan>' % esc(m.group(1)), esc(s))


class SVG:
    def __init__(self, w: int, h: int, title: str = ""):
        self.w, self.h = w, h
        self.title = title
        self.parts: list[str] = []
        self.markers: set[str] = set()

    def raw(self, s: str):
        self.parts.append(s)

    def text(self, x, y, s, size=13, anchor="start", color="ink", weight=None, italic=False,
             opacity=None, rotate=None):
        attrs = ['x="%s"' % fmt(x), 'y="%s"' % fmt(y), 'font-size="%s"' % fmt(size),
                 'fill="%s"' % c(color)]
        if anchor != "start":
            attrs.append('text-anchor="%s"' % anchor)
        if weight:
            attrs.append('font-weight="%s"' % weight)
        if italic:
            attrs.append('font-style="italic"')
        if opacity is not None:
            attrs.append('fill-opacity="%s"' % fmt(opacity))
        if rotate is not None:
            attrs.append('transform="rotate(%s %s %s)"' % (fmt(rotate), fmt(x), fmt(y)))
        self.parts.append("<text %s>%s</text>" % (" ".join(attrs), rich(s, size)))

    def line(self, x1, y1, x2, y2, color="ink", width=1.5, dash=None, opacity=None, arrow=False,
             cap=None):
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

    def arrow(self, x1, y1, x2, y2, color="ink", width=1.8, dash=None, shorten=0.0):
        if shorten:
            dx, dy = x2 - x1, y2 - y1
            n = (dx * dx + dy * dy) ** 0.5 or 1.0
            x2 -= dx / n * shorten
            y2 -= dy / n * shorten
        self.line(x1, y1, x2, y2, color=color, width=width, dash=dash, arrow=True)

    def rect(self, x, y, w, h, fill=None, fill_opacity=None, stroke=None, width=1.2, rx=0,
             dash=None, stroke_opacity=None):
        attrs = ['x="%s" y="%s" width="%s" height="%s"' % (fmt(x), fmt(y), fmt(max(w, 0)), fmt(max(h, 0)))]
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

    def path(self, d, stroke="ink", width=1.5, fill=None, fill_opacity=None, dash=None,
             arrow=False, opacity=None, join=None):
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

    def polyline(self, pts, stroke="ink", width=1.8, dash=None, opacity=None, fill=None,
                 fill_opacity=None):
        if len(pts) < 2:
            return
        d = "M" + " L".join("%s %s" % (fmt(x), fmt(y)) for x, y in pts)
        self.path(d, stroke=stroke, width=width, dash=dash, opacity=opacity, fill=fill,
                  fill_opacity=fill_opacity, join="round")

    def polygon(self, pts, fill=None, fill_opacity=None, stroke=None, width=1.2, dash=None):
        attrs = ['points="%s"' % " ".join("%s,%s" % (fmt(x), fmt(y)) for x, y in pts),
                 'fill="%s"' % (c(fill) if fill else "none")]
        if fill and fill_opacity is not None:
            attrs.append('fill-opacity="%s"' % fmt(fill_opacity))
        if stroke:
            attrs.append('stroke="%s" stroke-width="%s"' % (c(stroke), fmt(width)))
        if dash:
            attrs.append('stroke-dasharray="%s"' % dash)
        self.parts.append("<polygon %s/>" % " ".join(attrs))

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
    """A rectangular plot area with linear or log10 scales.

    (x0, y0) is the top-left corner of the plotting box in SVG units, w/h its size.
    xr, yr are (min, max) data ranges. logx/logy switch that axis to log10.
    """

    def __init__(self, svg: SVG, x0, y0, w, h, xr, yr, logx=False, logy=False):
        self.s, self.x0, self.y0, self.w, self.h = svg, x0, y0, w, h
        self.xr, self.yr, self.logx, self.logy = xr, yr, logx, logy

    def _t(self, v, r, log):
        if log:
            return (math.log10(v) - math.log10(r[0])) / (math.log10(r[1]) - math.log10(r[0]))
        return (v - r[0]) / (r[1] - r[0])

    def X(self, v):
        return self.x0 + self.w * self._t(v, self.xr, self.logx)

    def Y(self, v):
        return self.y0 + self.h * (1 - self._t(v, self.yr, self.logy))

    def P(self, x, y):
        return self.X(x), self.Y(y)

    def frame(self, xticks=(), yticks=(), xfmt=None, yfmt=None, grid=True, xlabel=None,
              ylabel=None, tick_size=11, label_size=12, ylabel_dx=-40):
        s = self.s
        xb, yb = self.y0 + self.h, self.x0
        if grid:
            for t in yticks:
                y = self.Y(t)
                s.line(self.x0, y, self.x0 + self.w, y, color="grid", width=1)
            for t in xticks:
                x = self.X(t)
                s.line(x, self.y0, x, xb, color="grid", width=1)
        s.line(self.x0, xb, self.x0 + self.w, xb, color="muted", width=1.1)
        s.line(yb, self.y0, yb, xb, color="muted", width=1.1)
        for t in xticks:
            x = self.X(t)
            s.line(x, xb, x, xb + 4, color="muted", width=1)
            s.text(x, xb + 16, (xfmt or _auto)(t), size=tick_size, anchor="middle", color="muted")
        for t in yticks:
            y = self.Y(t)
            s.line(yb - 4, y, yb, y, color="muted", width=1)
            s.text(yb - 7, y + 4, (yfmt or _auto)(t), size=tick_size, anchor="end", color="muted")
        if xlabel:
            s.text(self.x0 + self.w / 2, xb + 34, xlabel, size=label_size, anchor="middle", color="ink")
        if ylabel:
            cx, cy = self.x0 + ylabel_dx, self.y0 + self.h / 2
            s.text(cx, cy, ylabel, size=label_size, anchor="middle", color="ink", rotate=-90)

    def curve(self, xs, ys, **kw):
        pts = [self.P(x, y) for x, y in zip(xs, ys)]
        self.s.polyline(pts, **kw)


def _auto(v):
    if v == 0:
        return "0"
    if abs(v) >= 1000 or abs(v) < 0.01:
        e = int(math.floor(math.log10(abs(v))))
        m = v / 10 ** e
        if abs(m - 1) < 1e-9:
            return "10^{%d}" % e
        return "%g×10^{%d}" % (m, e)
    return "%g" % v
