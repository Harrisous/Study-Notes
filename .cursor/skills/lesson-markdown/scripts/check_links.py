#!/usr/bin/env python3
"""Check cross-lesson links in built HTML lessons.

    python check_links.py deepseek/*.html math/*.html index.html

For every <a href> that points at a local .html file, check that the file exists and, when the
link carries '#pN', that the target lesson has at least N+1 pages. Links to '#rN' / '#sN-M' anchors
are checked against element ids. External (http, https, mailto) links are ignored.
Exit status 1 if anything is broken.
"""

from __future__ import annotations

import os
import re
import sys
import urllib.parse


def pages_and_ids(path, cache={}):
    if path not in cache:
        with open(path, encoding="utf-8") as fh:
            t = fh.read()
        cache[path] = (len(re.findall(r'<section class="page', t)), set(re.findall(r'\bid="([^"]+)"', t)))
    return cache[path]


def main(files):
    bad = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        for href in re.findall(r'<a\b[^>]*\bhref="([^"]+)"', text):
            if re.match(r"^(https?:|mailto:|#$)", href):
                continue
            url, _, frag = href.partition("#")
            target = f if not url else os.path.normpath(os.path.join(os.path.dirname(f), urllib.parse.unquote(url)))
            if url and not url.endswith(".html"):
                continue
            if not os.path.exists(target):
                print("%s: missing file %s" % (f, href))
                bad += 1
                continue
            if not frag:
                continue
            n, ids = pages_and_ids(target)
            m = re.fullmatch(r"p(\d+)", frag)
            if m:
                if int(m.group(1)) >= max(n, 1):
                    print("%s: %s points past the last page (%d pages)" % (f, href, n))
                    bad += 1
            elif frag not in ids:
                print("%s: %s has no element id=%r" % (f, href, frag))
                bad += 1
    print("checked %d files, %d broken links" % (len(files), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    sys.exit(main(sys.argv[1:]))
