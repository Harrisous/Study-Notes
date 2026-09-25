"""Tiny SVG writer shared by the lesson-01 figure scripts (standard library only;
adapted from math/figures/gen/m01_svg.py).

Colours are always emitted as var(--fig-*, #fallback) so a figure renders on its own
and follows the lesson theme once inlined. Text accepts a small markup:

    *q*        bold (vectors)
    _{t,i}     subscript
    ^{Q}       superscript   (a ^{..} right after a _{..} is stacked above it)

Run any l01_*.py script from anywhere; it writes ../l01-*.svg next to this folder.
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


_NARROW = set("iljtfr.,;:!|'·()[] ")
_WIDE = set("MWmw⊤")


def text_width(s: str, size: float) -> float:
    """Rough advance width of s at font size `size` (system-ui metrics, good to ~15%)."""
    w = 0.0
    for ch in s:
        if ord(ch) > 0x2E7F:          # CJK and full-width punctuation
            w += 1.0
        elif ch in _NARROW:
            w += 0.30
        elif ch in _WIDE:
            w += 0.85
        elif ch.isupper():
            w += 0.66
        else:
            w += 0.55
    return w * size


def rich(s: str, size: float) -> str:
    """Turn the mini markup into tspans. Sub/superscripts never go below 11 units.

    A subscript directly followed by a superscript (or the reverse) is stacked: the second
    script is pulled back by the width of the first, and the text after the pair is pushed
    forward if the first script was the wider one, so e.g. W^{Q}_{i} does not run into the
    next glyph.
    """
    small = max(11, round(size * 0.74))
    out = []
    prev_w = None      # width of a script that the next script may stack under/over
    gap = 0.0          # forward shift owed to the next piece after a stacked pair
    for part in TOKEN.split(s):
        if not part:
            continue
        lead = ' dx="%s"' % fmt(gap) if gap > 0.5 else ""
        if part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append('<tspan font-weight="700"%s>%s</tspan>' % (lead, esc(part[1:-1])))
            prev_w, gap = None, 0.0
        elif part.startswith("_{") or part.startswith("^{"):
            inner = part[2:-1]
            shift = "sub" if part.startswith("_{") else "super"
            w = text_width(re.sub(r"\*", "", inner), small)
            if prev_w is not None:
                dx = ' dx="-%s"' % fmt(prev_w)
                gap = prev_w - w
                prev_w = None
            else:
                dx = lead
                prev_w, gap = w, 0.0
            out.append('<tspan baseline-shift="%s" font-size="%s"%s>%s</tspan>'
                       % (shift, fmt(small), dx, rich_inner(inner)))
        else:
            out.append('<tspan%s>%s</tspan>' % (lead, esc(part)) if lead else esc(part))
            prev_w, gap = None, 0.0
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
             opacity=None, rotate=None, family=None):
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
        if family:
            attrs.append('font-family="%s"' % family)
        self.parts.append("<text %s>%s</text>" % (" ".join(attrs), rich(s, size)))

    def mono(self, x, y, s, size=12, anchor="start", color="ink", weight=None):
        """Shape labels such as [T, 768] in a monospace face."""
        self.text(x, y, s, size=size, anchor=anchor, color=color, weight=weight,
                  family="ui-monospace, 'Cascadia Mono', Consolas, monospace")

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
    def box(self, x, y, w, h, lines, color="ink", fill_opacity=0.08, size=13, rx=6, stroke=True,
            text_color=None, weight=None, dash=None, line_gap=None, width=1.3):
        """Tinted rounded box with centred text lines (list of str, or (str, dict) tuples)."""
        self.rect(x, y, w, h, fill=color, fill_opacity=fill_opacity,
                  stroke=color if stroke else None, width=width, rx=rx, dash=dash)
        if isinstance(lines, (str, tuple)):
            lines = [lines]
        gap = line_gap or size * 1.38
        n = len(lines)
        y0 = y + h / 2 - (n - 1) * gap / 2 + size * 0.36
        for k, ln in enumerate(lines):
            opts = {}
            if isinstance(ln, tuple):
                ln, opts = ln
            self.text(x + w / 2, y0 + k * gap, ln, size=opts.get("size", size), anchor="middle",
                      color=opts.get("color", text_color or "ink"),
                      weight=opts.get("weight", weight))

    def plus(self, x, y, r=9, color="ink"):
        self.circle(x, y, r, fill="surface", stroke=color, width=1.6)
        self.line(x - r * 0.55, y, x + r * 0.55, y, color=color, width=1.6)
        self.line(x, y - r * 0.55, x, y + r * 0.55, color=color, width=1.6)

    def matrix(self, x, y, vals, cell=30, size=13, color="ink", fill=None, fill_opacity=0.10,
               highlight=None, hl_color="blue", brackets=True, cell_h=None):
        rows, cols = len(vals), len(vals[0])
        ch = cell_h or cell
        for i in range(rows):
            for j in range(cols):
                cx, cy = x + j * cell, y + i * ch
                if highlight and (i, j) in highlight:
                    col = highlight[(i, j)] if isinstance(highlight, dict) else hl_color
                    self.rect(cx + 1, cy + 1, cell - 2, ch - 2, fill=col, fill_opacity=0.18, rx=3)
                elif fill:
                    self.rect(cx + 1, cy + 1, cell - 2, ch - 2, fill=fill, fill_opacity=fill_opacity, rx=3)
                self.text(cx + cell / 2, cy + ch / 2 + size * 0.36, str(vals[i][j]), size=size,
                          anchor="middle", color=color)
        if brackets:
            h = rows * ch
            w = cols * cell
            t = 5
            self.path("M%s %s h-%s v%s h%s" % (fmt(x + 1), fmt(y), t, fmt(h), t), stroke="ink", width=1.3)
            self.path("M%s %s h%s v%s h-%s" % (fmt(x + w - 1), fmt(y), t, fmt(h), t), stroke="ink", width=1.3)
        return cols * cell, rows * ch

    def save(self, name: str):
        defs = []
        for col in sorted(self.markers):
            defs.append('<marker id="ah-%s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
                        'markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
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
