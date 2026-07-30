#!/usr/bin/env python3
"""arXiv research CLI: search, metadata, full-text retrieval, and grep.

Standard library only (Python 3.8+). No installation required.

    python3 arxiv.py search 'ti:"selective state spaces"' -n 5
    python3 arxiv.py meta 2312.00752 2405.21060
    python3 arxiv.py text 2402.01032 --out /tmp/jelassi.txt
    python3 arxiv.py grep 2205.14314 'IO-optimal' -C 3
    python3 arxiv.py cite 2312.00752

`grep` is the workhorse: it pulls the LaTeXML full text once, caches it, and
prints only the matching lines, so an exact theorem statement can be confirmed
without reading a whole paper into context.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

API = "https://export.arxiv.org/api/query"
UA = "academic-research-skill/1.0 (single-user research tooling)"
CACHE = os.path.join(tempfile.gettempdir(), "arxiv-skill-cache")
ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


# ---------------------------------------------------------------- http


def _get(url: str, tries: int = 3, timeout: int = 45) -> bytes:
    last = None
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 403, 404):
                break
            time.sleep(1.5 * (attempt + 1))
    raise SystemExit("fetch failed: %s\n  %s" % (url, last))


def _cached(key: str, url: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9._-]", "_", key))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    body = _get(url).decode("utf-8", "replace")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return body


# ---------------------------------------------------------------- metadata


def norm_id(raw: str) -> str:
    """'arXiv:2312.00752v2' / a full URL / '2312.00752' -> '2312.00752v2'."""
    raw = raw.strip()
    raw = re.sub(r"^https?://arxiv\.org/(abs|pdf|html)/", "", raw)
    raw = re.sub(r"^arxiv:", "", raw, flags=re.I)
    return raw.replace(".pdf", "")


def bare_id(raw: str) -> str:
    return re.sub(r"v\d+$", "", norm_id(raw))


class Paper(dict):
    def line(self) -> str:
        return "arXiv:%s  %s  [%s]" % (self["id"], self["published"][:10], self["primary"])


def _parse_entry(e: ET.Element) -> Paper:
    def txt(tag: str) -> str:
        node = e.find("a:" + tag, ATOM)
        return re.sub(r"\s+", " ", (node.text or "").strip()) if node is not None else ""

    full = txt("id")
    authors = [
        re.sub(r"\s+", " ", (n.text or "").strip())
        for n in e.findall("a:author/a:name", ATOM)
    ]
    primary = e.find("x:primary_category", ATOM)
    doi = e.find("x:doi", ATOM)
    comment = e.find("x:comment", ATOM)
    journal = e.find("x:journal_ref", ATOM)
    pdf = ""
    for link in e.findall("a:link", ATOM):
        if link.get("title") == "pdf":
            pdf = link.get("href") or ""
    return Paper(
        id=full.rsplit("/", 1)[-1],
        title=txt("title"),
        authors=authors,
        summary=txt("summary"),
        published=txt("published"),
        updated=txt("updated"),
        primary=(primary.get("term") if primary is not None else ""),
        categories=[c.get("term") for c in e.findall("a:category", ATOM)],
        doi=(doi.text.strip() if doi is not None and doi.text else ""),
        comment=(re.sub(r"\s+", " ", comment.text.strip()) if comment is not None and comment.text else ""),
        journal=(re.sub(r"\s+", " ", journal.text.strip()) if journal is not None and journal.text else ""),
        abs="https://arxiv.org/abs/" + full.rsplit("/", 1)[-1],
        pdf=pdf,
    )


def query(search: str = "", ids: str = "", n: int = 10, start: int = 0,
          sort: str = "relevance") -> list:
    params = {
        "start": str(start),
        "max_results": str(n),
        "sortBy": {"relevance": "relevance", "new": "submittedDate",
                   "updated": "lastUpdatedDate"}[sort],
        "sortOrder": "descending",
    }
    if search:
        params["search_query"] = search
    if ids:
        params["id_list"] = ids
    root = ET.fromstring(_get(API + "?" + urllib.parse.urlencode(params)))
    return [_parse_entry(e) for e in root.findall("a:entry", ATOM)]


# ---------------------------------------------------------------- full text


class _Text(HTMLParser):
    SKIP = {"script", "style", "head", "svg", "noscript"}
    BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "table", "figure", "figcaption", "blockquote", "br"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.depth = max(0, self.depth - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.depth:
            self.out.append(data)

    def text(self) -> str:
        raw = html.unescape("".join(self.out))
        raw = re.sub(r"[ \t\xa0]+", " ", raw)
        raw = re.sub(r" ?\n ?", "\n", raw)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def fulltext(ident: str) -> str:
    """LaTeXML rendering as plain text. Tries arxiv.org/html then ar5iv."""
    ident = norm_id(ident)
    errors = []
    for url in ("https://arxiv.org/html/" + ident,
                "https://arxiv.org/html/" + bare_id(ident) + "v1",
                "https://ar5iv.labs.arxiv.org/html/" + bare_id(ident)):
        try:
            body = _cached("html-" + url.rsplit("/", 2)[-1] + ".html", url)
        except SystemExit as exc:
            errors.append(str(exc))
            continue
        parser = _Text()
        parser.feed(body)
        out = parser.text()
        # ar5iv error pages are short; a real paper is tens of thousands of chars
        if len(out) > 4000:
            return out
        errors.append("too short from " + url)
    raise SystemExit(
        "no HTML rendering available for %s (pre-2023 papers often have none).\n"
        "Fall back to the abs page, the published version, or the PDF.\n  %s"
        % (ident, "\n  ".join(errors))
    )


# ---------------------------------------------------------------- commands


def _show(p: Paper, abstract: bool = True, width: int = 96) -> None:
    print(p.line())
    print("  " + textwrap.fill(p["title"], width, subsequent_indent="  "))
    auth = p["authors"]
    shown = ", ".join(auth[:6]) + (" (+%d more)" % (len(auth) - 6) if len(auth) > 6 else "")
    print("  " + textwrap.fill(shown, width, initial_indent="  ", subsequent_indent="    ").strip())
    for label, key in (("journal", "journal"), ("doi", "doi"), ("comment", "comment")):
        if p[key]:
            print("  %-8s %s" % (label + ":", textwrap.shorten(p[key], 220)))
    print("  url:     " + p["abs"])
    if abstract:
        print(textwrap.fill(p["summary"], width, initial_indent="  > ",
                            subsequent_indent="  > "))
    print()


def cmd_search(a) -> None:
    for p in query(search=" ".join(a.query), n=a.n, start=a.start, sort=a.sort):
        _show(p, abstract=not a.brief)


def cmd_meta(a) -> None:
    for p in query(ids=",".join(norm_id(i) for i in a.ids), n=len(a.ids)):
        _show(p, abstract=not a.brief)


def cmd_text(a) -> None:
    body = fulltext(a.id)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(body)
        print("%s -> %s (%d chars)" % (norm_id(a.id), a.out, len(body)))
    else:
        print(body)


def cmd_grep(a) -> None:
    lines = fulltext(a.id).splitlines()
    flags = 0 if a.case else re.IGNORECASE
    try:
        rx = re.compile(a.pattern, flags)
    except re.error as exc:
        raise SystemExit("bad pattern: %s" % exc)
    hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
    if not hits:
        raise SystemExit("no match for %r in %s (%d lines of text)"
                         % (a.pattern, norm_id(a.id), len(lines)))
    shown: set = set()
    for i in hits[: a.max]:
        lo, hi = max(0, i - a.C), min(len(lines), i + a.C + 1)
        if lo in shown:
            print("  ...")
        for j in range(lo, hi):
            if j in shown:
                continue
            shown.add(j)
            print("%6d%s %s" % (j + 1, ":" if j == i else "-", lines[j]))
        print("--")
    print("# %d match(es), showed %d" % (len(hits), min(len(hits), a.max)))


def cmd_cite(a) -> None:
    """Reference-card lines ready to paste into a lesson's reference list."""
    for p in query(ids=",".join(norm_id(i) for i in a.ids), n=len(a.ids)):
        last = [x.split()[-1] for x in p["authors"]]
        who = last[0] if len(last) == 1 else (
            " & ".join(last) if len(last) == 2 else last[0] + " et al.")
        year = p["published"][:4]
        print("title:   %s" % p["title"])
        print("authors: %s (%s), %d authors" % (who, year, len(p["authors"])))
        print("venue:   %s" % (p["journal"] or p["comment"] or "preprint (see comment field)"))
        print("id:      arXiv:%s   submitted %s, current version %s"
              % (p["id"], p["published"][:10], p["updated"][:10]))
        print("url:     %s" % p["abs"])
        print("finding: <one sentence: what it establishes>")
        print("matters: <one sentence: why the lesson cites it>")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="arXiv API search; supports ti: au: abs: cat: all: AND OR ANDNOT")
    s.add_argument("query", nargs="+")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--sort", choices=["relevance", "new", "updated"], default="relevance")
    s.add_argument("--brief", action="store_true", help="omit abstracts")
    s.set_defaults(fn=cmd_search)

    m = sub.add_parser("meta", help="metadata for explicit ids")
    m.add_argument("ids", nargs="+")
    m.add_argument("--brief", action="store_true")
    m.set_defaults(fn=cmd_meta)

    t = sub.add_parser("text", help="full text (LaTeXML) as plain text")
    t.add_argument("id")
    t.add_argument("--out")
    t.set_defaults(fn=cmd_text)

    g = sub.add_parser("grep", help="search inside the full text; use this to confirm exact wording")
    g.add_argument("id")
    g.add_argument("pattern")
    g.add_argument("-C", type=int, default=2, help="context lines")
    g.add_argument("--max", type=int, default=12, help="max matches to print")
    g.add_argument("--case", action="store_true", help="case sensitive")
    g.set_defaults(fn=cmd_grep)

    c = sub.add_parser("cite", help="emit reference-card skeletons")
    c.add_argument("ids", nargs="+")
    c.set_defaults(fn=cmd_cite)

    a = ap.parse_args()
    try:
        a.fn(a)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
