"""Tiny SVG writer shared by the M07 figure scripts (standard library only).

Colours are always emitted as var(--fig-*, #fallback) so a figure renders on its own
and follows the lesson theme once inlined. Text accepts a small markup:

    *q*        bold (vectors)
    _{t,i}     subscript
    ^{Q}       superscript

Run any m07_*.py script from anywhere; it writes ../m07-*.svg next to this folder.
(Same helper as m01_svg.py, copied so the M07 scripts do not depend on another lesson;
plotting helpers for axes and clipped polylines are appended at the end.)
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
    # bold inside a sub/superscript, e.g. _{*t*}; no nesting of scripts
    return re.sub(r"\*([^*]+)\*", lambda m: '<tspan font-weight="700">%s</tspan>' % esc(m.group(1)), esc(s))


class SVG:
    def __init__(self, w: int, h: int, title: str = ""):
        self.w, self.h = w, h
        self.title = title
        self.parts: list[str] = []
        self.markers: set[str] = set()

    # ---- primitives
    def raw(self, s: str):
        self.parts.append(s)

    def text(self, x, y, s, size=13, anchor="start", color="ink", weight=None, italic=False,
             opacity=None, rotate=None, halo=False):
        """halo=True paints a surface-coloured outline under the glyphs, so a label stays legible
        where a thin grid or guide line runs behind it (the outline follows the theme)."""
        attrs = ['x="%s"' % fmt(x), 'y="%s"' % fmt(y), 'font-size="%s"' % fmt(size),
                 'fill="%s"' % c(color)]
        if halo:
            attrs.append('stroke="%s" stroke-width="3.5" stroke-linejoin="round" paint-order="stroke"'
                         % c("surface"))
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

    # ---- composites
    def matrix(self, x, y, vals, cell=30, size=13, color="ink", fill=None, fill_opacity=0.10,
               highlight=None, hl_color="blue", brackets=True):
        """Draw a small numeric matrix. highlight: set of (i, j) cells to tint."""
        rows, cols = len(vals), len(vals[0])
        for i in range(rows):
            for j in range(cols):
                cx, cy = x + j * cell, y + i * cell
                if highlight and (i, j) in highlight:
                    self.rect(cx + 1, cy + 1, cell - 2, cell - 2, fill=hl_color, fill_opacity=0.18, rx=4)
                elif fill:
                    self.rect(cx + 1, cy + 1, cell - 2, cell - 2, fill=fill, fill_opacity=fill_opacity, rx=4)
                self.text(cx + cell / 2, cy + cell / 2 + size * 0.36, str(vals[i][j]), size=size,
                          anchor="middle", color=color)
        if brackets:
            h = rows * cell
            w = cols * cell
            t = 5
            self.path("M%s %s h-%s v%s h%s" % (fmt(x + 1), fmt(y), t, fmt(h), t), stroke="ink", width=1.3)
            self.path("M%s %s h%s v%s h-%s" % (fmt(x + w - 1), fmt(y), t, fmt(h), t), stroke="ink", width=1.3)
        return cols * cell, rows * cell

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


# ---- plotting helpers added for M07 (axes, clipped polylines)
import math as _math


class Axes:
    """Maps data (x, y) to screen for one plot panel; supports log axes.

    x0, x1, y0, y1 are the screen box (y0 = top). xlim/ylim are data ranges;
    xlog/ylog use log10 of the data.
    """

    def __init__(self, x0, y0, x1, y1, xlim, ylim, xlog=False, ylog=False):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    def _t(self, v, log):
        return _math.log10(v) if log else v

    def X(self, x):
        a, b = self._t(self.xlim[0], self.xlog), self._t(self.xlim[1], self.xlog)
        return self.x0 + (self._t(x, self.xlog) - a) / (b - a) * (self.x1 - self.x0)

    def Y(self, y):
        a, b = self._t(self.ylim[0], self.ylog), self._t(self.ylim[1], self.ylog)
        return self.y1 - (self._t(y, self.ylog) - a) / (b - a) * (self.y1 - self.y0)

    def __call__(self, x, y):
        return self.X(x), self.Y(y)

    def frame(self, svg, color="muted"):
        svg.line(self.x0, self.y1, self.x1, self.y1, color=color, width=1.2)
        svg.line(self.x0, self.y0, self.x0, self.y1, color=color, width=1.2)

    def xgrid(self, svg, ticks, labels=None, size=11, grid=True, dy=17):
        for i, t in enumerate(ticks):
            x = self.X(t)
            if grid:
                svg.line(x, self.y0, x, self.y1, color="grid", width=1)
            lab = labels[i] if labels else fmt(t)
            svg.text(x, self.y1 + dy, lab, size=size, anchor="middle", color="muted")

    def ygrid(self, svg, ticks, labels=None, size=11, grid=True, dx=7):
        for i, t in enumerate(ticks):
            y = self.Y(t)
            if grid:
                svg.line(self.x0, y, self.x1, y, color="grid", width=1)
            lab = labels[i] if labels else fmt(t)
            svg.text(self.x0 - dx, y + 4, lab, size=size, anchor="end", color="muted")


def _clip_segment(p, q, box):
    """Liang-Barsky clip of segment p-q to box (xmin, ymin, xmax, ymax) in screen units."""
    x0, y0 = p
    x1, y1 = q
    xmin, ymin, xmax, ymax = box
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if pp == 0:
            if qq < 0:
                return None
        else:
            r = qq / pp
            if pp < 0:
                if r > t1:
                    return None
                t0 = max(t0, r)
            else:
                if r < t0:
                    return None
                t1 = min(t1, r)
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def clipped_path(points, box):
    """SVG path data for a polyline (screen points) clipped to box; breaks where it leaves the box."""
    out = []
    pen = None
    for p, q in zip(points[:-1], points[1:]):
        seg = _clip_segment(p, q, box)
        if seg is None:
            pen = None
            continue
        a, b = seg
        if pen is None or abs(pen[0] - a[0]) > 1e-6 or abs(pen[1] - a[1]) > 1e-6:
            out.append("M%s %s" % (fmt(a[0]), fmt(a[1])))
        out.append("L%s %s" % (fmt(b[0]), fmt(b[1])))
        pen = b
    return " ".join(out)


def sci(v, digits=2):
    """1.5e-4 -> '1.5×10^{-4}' in the mini markup."""
    if v == 0:
        return "0"
    e = int(_math.floor(_math.log10(abs(v))))
    m = v / 10 ** e
    ms = ("%%.%dg" % digits) % m
    if ms == "1":
        return "10^{%d}" % e
    return "%s×10^{%d}" % (ms, e)
