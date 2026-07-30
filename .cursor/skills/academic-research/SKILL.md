---
name: academic-research
description: Runs a plan-search-verify-aggregate loop over primary literature (arXiv, PMLR, OpenReview, ACL, journal pages) and produces citation-grade reference cards with verbatim quotes and URLs. Use when writing or fact-checking technical lesson material, when the user asks to find or verify papers, when a claim needs a citation, when building a reference list, or when any number, theorem, benchmark result, or attribution is about to be asserted.
---

# Academic research

Rule zero: **never state a number, theorem, or attribution from memory.** Memory
supplies the search query; the source supplies the claim. If a source cannot be
reached, the claim is written as unverified or dropped.

## The loop

```
Task progress:
- [ ] 1. PLAN     decompose into atomic claims, each independently checkable
- [ ] 2. SEARCH   find the primary source for each claim
- [ ] 3. VERIFY   read the source; quote it verbatim
- [ ] 4. RECONCILE  compare versions, check what secondary sources got wrong
- [ ] 5. AGGREGATE  write reference cards + an errata list
```

### 1. Plan

Write the claim list *before* searching. An atomic claim names one fact and one
would-be source:

- good: "Pascanu et al. 2013 state the vanishing-gradient sufficient condition
  using the largest singular value of W_rec, not the largest eigenvalue" — one
  fact, one source, checkable.
- bad: "RNNs have trouble with long sequences" — unfalsifiable, uncitable.

Classify each claim, because the verification bar differs:

| kind | bar |
|---|---|
| theorem / formal result | exact statement + its number + hypotheses |
| headline number | the number as printed, plus table/figure number |
| attribution ("X introduced Y") | the earliest source that actually does it |
| engineering fact (defaults, shapes) | the code, not the paper |
| "the field believes" | 2+ independent primary sources, or say "as of <date>" |

### 2. Search

Use `scripts/arxiv.py` (standard library only, no install):

```bash
S=.cursor/skills/academic-research/scripts/arxiv.py
python3 $S search 'ti:"selective state spaces"' -n 5
python3 $S search 'au:Gu_A AND abs:"state space"' -n 10 --sort new --brief
python3 $S meta 2312.00752 2405.21060
```

Field prefixes: `ti:` `au:` `abs:` `cat:` `all:`, combined with `AND` `OR`
`ANDNOT`. Quote multi-word phrases. `--sort new` for recency, default is
relevance.

Non-arXiv venues, in order of preference: publisher page (PMLR, ACL Anthology,
TACL, JMLR, IEEE, Neural Computation), then OpenReview for reviews and camera
copies, then the lab's own page. Never a blog post, never a secondary tutorial,
never another model's summary.

### 3. Verify

`grep` the full text instead of reading whole papers — it pulls the LaTeXML
rendering once, caches it, and prints only what matched:

```bash
python3 $S grep 2402.01032 'Theorem 1' -C 6
python3 $S grep 2205.14314 'IO.optimal|Proposition 3' -C 4
python3 $S text 2405.17394 --out /tmp/sarrof.txt   # then Grep/Read the file
```

Pre-2023 papers usually have no LaTeXML rendering. Fall back to the abs page,
the published HTML, or fetch the PDF. For a formal result, capture the
**hypotheses too** — a theorem quoted without its assumptions is a
misattribution waiting to happen.

Then recompute anything arithmetic. Do not copy a derived figure; derive it:

```bash
python3 -c "print(2*2*32*128*2048*512/1e12, 'TB')"
```

### 4. Reconcile

Four failure modes to check every time, in this order:

1. **Version drift.** arXiv v1 vs vN vs the published version differ, sometimes
   in the statement of a theorem. Say which version was read. When the published
   version disagrees with arXiv, the published version wins and the discrepancy
   is worth a footnote.
2. **Paper vs code.** Released implementations routinely differ from the paper's
   equations. When both exist, read both and record the disagreement rather than
   picking one.
3. **Telephone game.** A number that appears in many secondary sources and no
   primary one is a rumour. Trace it or drop it.
4. **Adjacent-claim substitution.** The most common real error: a true theorem
   cited for a nearby claim it does not support. If the theorem is about
   problem A and the text is about problem B, that is a defect even when both
   sentences are individually true.

### 5. Aggregate

Emit one reference card per source. `python3 $S cite <ids>` prints the skeleton:

```
[n] Title
    First-author et al. (year), venue. arXiv:id[vN]
    url
    Finding — one sentence: what it establishes.
    Matters — one sentence: why this document cites it.
```

`Finding` and `Matters` are mandatory and must differ. A card whose `Matters`
line restates `Finding` means the citation has no job in the document.

Also emit an **errata list**: every place where the obvious/common claim turned
out to be wrong, what the correct claim is, and the source. This is the highest
value artefact of the loop — write it to a companion doc so the next session
does not repeat the research or regress the correction.

## Honesty requirements

- An "UNVERIFIED / COULD NOT CONFIRM" section is required, and being empty is
  suspicious. Reporting a gap beats filling it.
- Distinguish "the paper claims" from "this is true". Report the claim's status
  (peer-reviewed, preprint, withdrawn, contested) and, if known, whether anyone
  has reproduced it.
- Benchmark numbers are the least reliable citable object: they depend on
  tuning, seeds, and harness version. Cite direction confidently, magnitude
  cautiously.
- Cite artefacts that have no paper (model cards, release notes) as artefacts,
  in a separate list. Never invent an arXiv id for one.

## Parallel gathering

For a broad sweep, dispatch one subagent per *cluster* of claims and require
verbatim quotes plus URLs in the report, so the returned material can be
spot-checked. Treat subagent output as a lead, not as a verified fact: re-run
`grep` yourself on any claim that will carry weight in the document.
