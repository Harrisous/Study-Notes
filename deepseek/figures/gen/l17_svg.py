"""Tiny SVG writer shared by the lesson-17 figure scripts (standard library only; adapted from math/figures/gen/m01_svg.py).

Colours are always emitted as var(--fig-*, #fallback) so a figure renders on its own
and follows the lesson theme once inlined. Text accepts a small markup:

    *q*        bold (vectors)
    _{t,i}     subscript
    ^{Q}       superscript

Run any l17_*.py script from anywhere; it writes ../l17-*.svg next to this folder.
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
    prev_w = None           # estimated width of an immediately preceding sub/superscript
    pending_dx = 0.0        # forward shift owed after a stacked pair
    for part in TOKEN.split(s):
        if not part:
            continue
        if part.startswith("_{") or part.startswith("^{"):
            inner = part[2:-1]
            shift = "sub" if part.startswith("_{") else "super"
            w = text_width(re.sub(r"\*", "", inner), small)
            dx = ""
            if prev_w is not None:
                # stack this script over/under the previous one: step back by its width
                dx = ' dx="-%s"' % fmt(prev_w)
                pending_dx = max(prev_w, w) - w
                prev_w = None
            else:
                prev_w = w
            out.append('<tspan baseline-shift="%s" font-size="%s"%s>%s</tspan>'
                       % (shift, fmt(small), dx, rich_inner(inner)))
            continue
        lead = ' dx="%s"' % fmt(pending_dx) if pending_dx > 0.05 else ""
        pending_dx = 0.0
        prev_w = None
        if part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append('<tspan font-weight="700"%s>%s</tspan>' % (lead, esc(part[1:-1])))
        elif lead:
            out.append('<tspan%s>%s</tspan>' % (lead, esc(part)))
        else:
            out.append(esc(part))
    return "".join(out)


def text_width(s: str, size: float) -> float:
    """Rough advance width of a short Latin string (for stacking sub/superscripts)."""
    w = 0.0
    for ch in s:
        if ch in ",.:;'|!":
            w += 0.28
        elif ch in "ijlrtf()[]":
            w += 0.34
        elif ch.isupper() or ch in "mwMW":
            w += 0.68
        else:
            w += 0.56
    return w * size


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

    def box(self, x, y, w, h, lines, color="ink", fill_opacity=0.10, size=13, rx=6, stroke=True,
            text_color=None, weight=None, dash=None, line_gap=None):
        """Tinted rounded box with centred text lines (list of str or (str, dict) tuples)."""
        self.rect(x, y, w, h, fill=color, fill_opacity=fill_opacity,
                  stroke=color if stroke else None, width=1.3, rx=rx, dash=dash)
        if isinstance(lines, str):
            lines = [lines]
        gap = line_gap or size * 1.35
        n = len(lines)
        y0 = y + h / 2 - (n - 1) * gap / 2 + size * 0.36
        for i, ln in enumerate(lines):
            opts = {}
            if isinstance(ln, tuple):
                ln, opts = ln
            self.text(x + w / 2, y0 + i * gap, ln, size=opts.get("size", size), anchor="middle",
                      color=opts.get("color", text_color or "ink"), weight=opts.get("weight", weight))

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
