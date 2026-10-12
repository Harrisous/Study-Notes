"""Tiny SVG writer for the lesson-09 (MLA) figure scripts. Standard library only.

Adapted from math/figures/gen/m01_svg.py (copied, not imported, so the two lessons can evolve
independently). Colours are always emitted as var(--fig-*, #fallback) so a figure renders on
its own and follows the lesson theme once inlined. Text accepts a small markup:

    *q*        bold (vectors)
    _{t,i}     subscript
    ^{Q}       superscript

Every l09_*.py script writes ../l09-*.svg next to this folder:
    python deepseek/figures/gen/l09_kv_bytes.py
"""

from __future__ import annotations

import math
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):   # scripts print CJK labels; Windows consoles default to cp1252
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

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


def tw(s: str, size: float) -> float:
    """Rough rendered width of a label (markup stripped): CJK ~1.0em, others ~0.58em."""
    plain = re.sub(r"[*]|_\{|\^\{|\}", "", s)
    w = 0.0
    for ch in plain:
        w += size * (1.0 if ord(ch) > 0x2E80 else 0.58)
    return w


TOKEN = re.compile(r"(\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})")


def rich(s: str, size: float) -> str:
    """Turn the mini markup into tspans. Sub/superscripts never go below 11 units.

    Scripts are placed with explicit dy offsets (not baseline-shift) so every browser puts them
    at the same height; the offset is undone on the next run of normal text.
    """
    small = max(11, round(size * 0.74))
    d_sub, d_sup = round(size * 0.30, 1), -round(size * 0.40, 1)
    out = []
    cur = 0.0
    for part in TOKEN.split(s):
        if not part:
            continue
        if part.startswith("_{") or part.startswith("^{"):
            target = d_sub if part.startswith("_{") else d_sup
            dy = target - cur
            cur = target
            out.append('<tspan dy="%s" font-size="%s">%s</tspan>'
                       % (fmt(dy), fmt(small), rich_inner(part[2:-1])))
            continue
        dy = -cur
        cur = 0.0
        dy_attr = ' dy="%s"' % fmt(dy) if dy else ""
        if part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append('<tspan%s font-weight="700">%s</tspan>' % (dy_attr, esc(part[1:-1])))
        elif dy_attr:
            out.append('<tspan%s>%s</tspan>' % (dy_attr, esc(part)))
        else:
            out.append(esc(part))
    return "".join(out)


def rich_inner(s: str) -> str:
    return re.sub(r"\*([^*]+)\*", lambda m: '<tspan font-weight="700">%s</tspan>' % esc(m.group(1)), esc(s))


def comma(v) -> str:
    return format(int(round(v)), ",")


class LogAxis:
    """Map a positive value to pixels on a log10 scale."""

    def __init__(self, vmin, vmax, pmin, pmax):
        self.a, self.b = math.log10(vmin), math.log10(vmax)
        self.pmin, self.pmax = pmin, pmax

    def __call__(self, v):
        return self.pmin + (math.log10(v) - self.a) / (self.b - self.a) * (self.pmax - self.pmin)


class LinAxis:
    def __init__(self, vmin, vmax, pmin, pmax):
        self.vmin, self.vmax, self.pmin, self.pmax = vmin, vmax, pmin, pmax

    def __call__(self, v):
        return self.pmin + (v - self.vmin) / (self.vmax - self.vmin) * (self.pmax - self.pmin)


class SVG:
    def __init__(self, w: int, h: int, title: str = ""):
        self.w, self.h = w, h
        self.title = title
        self.parts: list[str] = []
        self.markers: set[str] = set()
        self.defs: list[str] = []

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

    def arrow(self, x1, y1, x2, y2, color="ink", width=1.6, dash=None, shorten=0.0):
        if shorten:
            dx, dy = x2 - x1, y2 - y1
            n = (dx * dx + dy * dy) ** 0.5 or 1.0
            x2 -= dx / n * shorten
            y2 -= dy / n * shorten
        self.line(x1, y1, x2, y2, color=color, width=width, dash=dash, arrow=True)

    def rect(self, x, y, w, h, fill=None, fill_opacity=None, stroke=None, width=1.2, rx=0,
             dash=None, stroke_opacity=None, fill_url=None):
        attrs = ['x="%s" y="%s" width="%s" height="%s"' % (fmt(x), fmt(y), fmt(w), fmt(h))]
        if fill_url:
            attrs.append('fill="url(#%s)"' % fill_url)
        else:
            attrs.append('fill="%s"' % (c(fill) if fill else "none"))
        if fill and fill_opacity is not None and not fill_url:
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

    def polyline(self, pts, stroke="ink", width=1.8, dash=None, opacity=None):
        d = "M" + " L".join("%s %s" % (fmt(x), fmt(y)) for x, y in pts)
        self.path(d, stroke=stroke, width=width, dash=dash, opacity=opacity, join="round")

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

    def hatch(self, pid: str, color: str, gap: int = 6, width: float = 1.4, opacity: float = 0.7):
        """Register a diagonal hatch pattern usable as rect(..., fill_url=pid)."""
        self.defs.append('<pattern id="%s" patternUnits="userSpaceOnUse" width="%d" height="%d" '
                         'patternTransform="rotate(45)"><rect width="%d" height="%d" fill="%s" '
                         'fill-opacity="0.12"/><line x1="0" y1="0" x2="0" y2="%d" stroke="%s" '
                         'stroke-width="%s" stroke-opacity="%s"/></pattern>'
                         % (pid, gap, gap, gap, gap, c(color), gap, c(color), fmt(width), fmt(opacity)))

    # ---- composites
    def box(self, x, y, w, h, label, color="ink", size=13, fill_opacity=0.12, weight="600",
            stroke_w=1.3, dash=None, label_color=None, sub=None, sub_size=11.5, rx=5,
            stroke=None, width=None):
        """A rounded box with a centred label (and optional second line)."""
        self.rect(x, y, w, h, fill=color, fill_opacity=fill_opacity, stroke=stroke or color,
                  width=width or stroke_w, rx=rx, dash=dash)
        lc = label_color or ("ink" if color in ("rule", "grid", "muted") else color)
        if sub:
            self.text(x + w / 2, y + h / 2 - 2, label, size=size, anchor="middle", color=lc, weight=weight)
            self.text(x + w / 2, y + h / 2 + sub_size + 1, sub, size=sub_size, anchor="middle", color="muted")
        else:
            self.text(x + w / 2, y + h / 2 + size * 0.36, label, size=size, anchor="middle", color=lc,
                      weight=weight)

    def save(self, name: str):
        defs = list(self.defs)
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
