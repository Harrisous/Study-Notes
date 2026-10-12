#!/usr/bin/env python3
"""Build a course index page (deepseek/index.html, math/index.html) from a small YAML file.

    python build_index.py deepseek/src/index.yaml      # -> deepseek/index.html

The YAML holds: title, lede, back (link back to the site index), sections: [{name, blurb?, lessons:
[{file, tag, title, desc}]}], and optional aside (markdown-free HTML string shown under the hero).
A lesson whose built .html does not exist yet is listed but greyed out as "编写中".
Styling follows the site's root index.html so the whole site reads as one system.
"""

from __future__ import annotations

import html
import os
import re
import sys

import yaml

CSS = r"""
:root{
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", Roboto, Helvetica, Arial, sans-serif;
  --serif: "Noto Serif SC", "Source Han Serif SC", "Songti SC", ui-serif, Georgia, "Times New Roman", serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --bg:#FBFAF7; --surface:#FFFFFF; --surface-2:#F4F2ED; --ink:#1C1B19; --ink-2:#4A4843; --muted:#7A776F; --line:#E4E0D7;
  --blue:#2563EB; --blue-bg:#EBF1FE; --purple:#7038B8; --purple-bg:#F3EDFC; --green:#16704A; --green-bg:#E9F4EE;
  --amber:#8A6508; --amber-bg:#FCF7E8;
  --maxw: 860px; --r: 12px;
  --shadow: 0 1px 2px rgba(28,27,25,.05), 0 4px 14px rgba(28,27,25,.05);
  --shadow-lift: 0 2px 4px rgba(28,27,25,.06), 0 12px 28px rgba(28,27,25,.10);
}
html[data-theme="dark"]{
  --bg:#14161A; --surface:#1B1E24; --surface-2:#22262D; --ink:#ECEAE5; --ink-2:#C3C0B9; --muted:#8E8B84; --line:#2E333B;
  --blue:#7DA6FF; --blue-bg:#1B2436; --purple:#B490F0; --purple-bg:#261D38; --green:#5EC191; --green-bg:#12251C;
  --amber:#DFB44E; --amber-bg:#262010;
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 14px rgba(0,0,0,.25);
  --shadow-lift: 0 2px 6px rgba(0,0,0,.35), 0 14px 32px rgba(0,0,0,.32);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.7;-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 24px}
header{border-bottom:1px solid var(--line);background:var(--surface)}
.bar{max-width:var(--maxw);margin:0 auto;padding:14px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.brand{font-weight:650;display:flex;align-items:center;gap:10px}
.dot{width:10px;height:10px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--purple));flex:none}
.pill{font-size:.875rem;color:var(--muted);padding:6px 12px;border:1px solid var(--line);border-radius:999px;background:transparent;font:inherit;font-size:.875rem;cursor:pointer}
.pill:hover{color:var(--ink);border-color:var(--muted)}
.actions{display:flex;gap:10px;align-items:center}
.hero{padding:56px 0 28px}
h1{font-family:var(--serif);font-size:clamp(1.9rem,5vw,2.6rem);line-height:1.2;margin:0 0 14px;font-weight:700;text-wrap:balance}
.lede{color:var(--ink-2);font-size:1.06rem;max-width:62ch;margin:0}
.aside{margin:22px 0 0;padding:14px 18px;border:1px solid var(--line);border-radius:var(--r);background:var(--surface);color:var(--ink-2);font-size:.94rem}
.aside b{color:var(--ink)}
h2{font-size:.78rem;letter-spacing:.12em;color:var(--muted);font-weight:700;margin:46px 0 6px}
.blurb{color:var(--muted);font-size:.92rem;margin:0 0 14px;max-width:70ch}
.grid{display:grid;gap:12px}
@media (min-width:680px){.grid{grid-template-columns:repeat(2,1fr)}}
.card{display:block;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:16px 18px;box-shadow:var(--shadow);transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease}
a.card:hover{transform:translateY(-2px);box-shadow:var(--shadow-lift);border-color:var(--muted)}
a.card:focus-visible{outline:2px solid var(--blue);outline-offset:3px}
.card.wip{opacity:.55;box-shadow:none}
.tag{display:inline-block;font-size:.7rem;font-weight:700;letter-spacing:.06em;padding:2px 9px;border-radius:999px;margin-bottom:8px;color:var(--blue);background:var(--blue-bg)}
.tag.m{color:var(--green);background:var(--green-bg)}
.tag.x{color:var(--purple);background:var(--purple-bg)}
.card h3{font-family:var(--serif);font-size:1.08rem;line-height:1.35;margin:0 0 4px;font-weight:700}
.card p{margin:0;color:var(--muted);font-size:.9rem;line-height:1.6}
.wipnote{font-size:.75rem;color:var(--amber);margin-left:6px}
footer{margin-top:56px;border-top:1px solid var(--line);padding:22px 0 40px;color:var(--muted);font-size:.875rem}
@media (prefers-reduced-motion:reduce){.card{transition:none}}
"""

SCRIPT = r"""
(function(){
  var btn = document.getElementById('theme'), root = document.documentElement;
  function apply(t){ root.setAttribute('data-theme', t); btn.textContent = t === 'dark' ? '浅色' : '深色'; }
  var saved = null; try { saved = localStorage.getItem('study-notes-theme'); } catch (e) {}
  apply(saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
  btn.addEventListener('click', function(){
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    apply(next); try { localStorage.setItem('study-notes-theme', next); } catch (e) {}
  });
})();
"""


def esc(s):
    return html.escape(str(s), quote=True)


def build(yaml_path):
    with open(yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    src_dir = os.path.dirname(os.path.abspath(yaml_path))
    out_dir = os.path.dirname(src_dir) if os.path.basename(src_dir) == "src" else src_dir
    out = os.path.join(out_dir, "index.html")
    parts = []
    for sec in cfg["sections"]:
        parts.append("<h2>%s</h2>" % esc(sec["name"]))
        if sec.get("blurb"):
            parts.append('<p class="blurb">%s</p>' % esc(sec["blurb"]))
        parts.append('<div class="grid">')
        for l in sec["lessons"]:
            exists = os.path.exists(os.path.join(out_dir, l["file"]))
            cls = "tag " + l.get("tagclass", "")
            inner = ('<span class="%s">%s</span><h3>%s%s</h3><p>%s</p>'
                     % (cls.strip(), esc(l["tag"]), esc(l["title"]),
                        "" if exists else '<span class="wipnote">编写中</span>', esc(l["desc"])))
            if exists:
                parts.append('<a class="card" href="%s">%s</a>' % (esc(l["file"]), inner))
            else:
                parts.append('<div class="card wip">%s</div>' % inner)
        parts.append("</div>")
    aside = '<div class="aside">%s</div>' % cfg["aside"] if cfg.get("aside") else ""
    page = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title>
<meta name="description" content="%(desc)s">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&display=swap">
<!-- Built by .cursor/skills/lesson-markdown/scripts/build_index.py from src/index.yaml -->
<style>%(css)s</style>
</head>
<body>
<header><div class="bar"><a class="brand" href="%(back)s"><span class="dot"></span>Study Notes</a>
<span class="actions"><a class="pill" href="%(back)s">全部笔记</a><button class="pill" id="theme" type="button">深色</button></span></div></header>
<main class="wrap">
<section class="hero"><h1>%(title)s</h1><p class="lede">%(lede)s</p>%(aside)s</section>
%(body)s
</main>
<footer class="wrap">%(footer)s</footer>
<script>%(script)s</script>
</body>
</html>
""" % {"title": esc(cfg["title"]), "desc": esc(cfg.get("description", cfg["lede"])), "css": CSS,
       "back": esc(cfg.get("back", "../index.html")), "lede": esc(cfg["lede"]), "aside": aside,
       "body": "\n".join(parts), "footer": cfg.get("footer", ""), "script": SCRIPT}
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    print("%s -> %s" % (yaml_path, out))
    return cfg, out_dir


def build_artifact(yaml_path, artifact_out, link_map, short=None):
    """claude.ai Artifact variant of a course index: no document skeleton, theme follows the viewer,
    lesson cards link to the lessons' published Artifact URLs (unpublished ones are not links)."""
    with open(yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    src_dir = os.path.dirname(os.path.abspath(yaml_path))
    out_dir = os.path.dirname(src_dir) if os.path.basename(src_dir) == "src" else src_dir
    root = out_dir
    while not os.path.isdir(os.path.join(root, ".cursor")) and os.path.dirname(root) != root:
        root = os.path.dirname(root)

    def key(rel):
        return os.path.relpath(os.path.normpath(os.path.join(out_dir, rel)), root).replace(os.sep, "/")

    parts = []
    for sec in cfg["sections"]:
        parts.append("<h2>%s</h2>" % esc(sec["name"]))
        if sec.get("blurb"):
            parts.append('<p class="blurb">%s</p>' % esc(sec["blurb"]))
        parts.append('<div class="grid">')
        for l in sec["lessons"]:
            url = link_map.get(key(l["file"]))
            cls = ("tag " + l.get("tagclass", "")).strip()
            inner = '<span class="%s">%s</span><h3>%s</h3><p>%s</p>' % (cls, esc(l["tag"]), esc(l["title"]), esc(l["desc"]))
            parts.append('<a class="card" href="%s">%s</a>' % (esc(url), inner) if url else '<div class="card">%s</div>' % inner)
        parts.append("</div>")
    aside = cfg.get("aside", "")

    def relink(m):
        url = link_map.get(key(m.group(1)))
        return 'href="%s"' % esc(url) if url else 'href="%s"' % esc(m.group(1))
    aside = re.sub(r'href="([^"#:]+\.html)"', relink, aside)
    dark = re.search(r'html\[data-theme="dark"\]\{(.*?)\}', CSS, re.S).group(1)
    css = CSS.replace('html[data-theme="dark"]{' + dark + '}',
                      '@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;%s}}'
                      ':root[data-theme="dark"]{color-scheme:dark;%s}' % (dark, dark))
    page = ('<title>%s</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&display=swap">\n'
            '<style>%s</style>\n<header><div class="bar"><span class="brand"><span class="dot"></span>Study Notes</span></div></header>\n'
            '<main class="wrap"><section class="hero"><h1>%s</h1><p class="lede">%s</p>%s</section>\n%s\n</main>\n'
            '<footer class="wrap">%s</footer>\n'
            % (esc(short or cfg["title"]), css, esc(cfg["title"]), esc(cfg["lede"]),
               '<div class="aside">%s</div>' % aside if aside else "", "\n".join(parts), cfg.get("footer", "")))
    with open(artifact_out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    print("  artifact variant -> %s" % artifact_out)


if __name__ == "__main__":
    import argparse
    import json
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="+")
    ap.add_argument("--artifact", help="also write the Artifact variant to this path (single src only)")
    ap.add_argument("--link-map", help="JSON map of repo-relative lesson paths to Artifact URLs")
    ap.add_argument("--short", help="Artifact title (2-4 words)")
    a = ap.parse_args()
    for p in a.src:
        build(p)
    if a.artifact:
        lm = {}
        if a.link_map:
            with open(a.link_map, encoding="utf-8") as fh:
                lm = json.load(fh)
        build_artifact(a.src[0], a.artifact, lm, a.short)
