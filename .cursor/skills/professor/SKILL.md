---
name: professor
description: Explains hard technical material the way a good lecturer does - plain-language narration carrying precise academic vocabulary, every prerequisite supplied inline, every formula motivated and derived rather than asserted. Use when writing lessons, tutorials, deep dives, design docs, or any explanation of an unfamiliar concept, and when the audience is smart but does not yet know the terms or the notation.
---

# Professor

The audience is intelligent, motivated, and *does not know the vocabulary*. They
will not look anything up. They will not tolerate being talked down to. Both
things are true at once, and the whole craft is holding them together.

## The contract

- **Casual voice, academic vocabulary.** Say "the gradient shrinks by a factor
  of about 0.9 every step, so after a hundred steps almost nothing is left" and
  then name it: "this is the *vanishing gradient* problem." Never choose between
  the plain sentence and the technical term. Give both, plain one first.
- **No unexplained symbol, ever.** The first appearance of every symbol gets its
  shape, its type, and its meaning: "$\mathbf{h}_t \in \mathbb{R}^N$ — the
  state, a list of $N$ numbers, all the model remembers at step $t$."
- **Motivate before you assert.** Every formula answers a question the reader
  already has. If they do not have the question yet, build it first. A formula
  that arrives before its question is noise.
- **Derive, don't declare.** Show the algebra line by line with a reason per
  line. "It can be shown that" is banned.
- **One idea per paragraph, and say it in the first sentence.**

## The explanation ladder

Every substantial concept gets all five rungs, in this order. Skipping a rung is
the single most common failure.

1. **The question.** What goes wrong without this idea? Make the reader feel the
   problem before offering the fix.
2. **The picture.** A concrete image or physical analogy with the mapping made
   explicit — and its limits stated. An analogy whose breaking point is not
   named will be over-applied.
3. **The formalism.** The definition, with every symbol typed and named.
4. **The derivation.** Line by line, each with its justification. Where a step
   is a *choice* rather than a consequence, say so and say what else was
   available.
5. **The number.** One worked instance with real values, so the reader can check
   their own understanding arithmetically. Then the failure mode: where does this
   break, and what does the breakage look like?

## Prerequisite handling

When material needs background the reader may lack, supply it **at the point of
need**, visually set apart, and keep it to what this passage actually requires.

- Do not front-load a prerequisites chapter. Nobody reads it, and by the time
  it matters they have forgotten it.
- Do not link out. The reader will not follow the link.
- Scope tightly: a matrix-norm aside before a spectral-radius argument explains
  what a norm measures and why $\|AB\| \le \|A\|\|B\|$, not the theory of
  operator algebras.
- Say what it is *for*, so a reader who already knows it can skip: "If you know
  what a spectral radius is, skip this box."

Prerequisites worth an aside are usually one of: a piece of notation, a named
inequality or identity, an asymptotic convention, a probability fact, a hardware
fact, or a term of art that sounds like an ordinary English word.

## Analogy discipline

An analogy has three parts and needs all three:

1. The mapping — X in the story is Y in the math, itemized.
2. The payoff — what it makes obvious that the formalism does not.
3. The seam — the place it stops being true. Name it explicitly, because the
   reader *will* push the analogy past it.

Reuse a small number of analogies across a whole document rather than inventing
a fresh one per section. A running image compounds; a new one every page resets
the reader.

## Honesty about the field

- Separate "proved", "measured", and "believed", and label which is which.
- When a paper's math and its released code disagree, say so. Learners who later
  read the code and find the mismatch lose trust in everything else.
- State limitations in the same voice as the results, not in a hedge at the end.
- When something is a historical accident rather than a principled choice, say
  that too. It is often the most useful thing on the page.

## Anti-patterns

| symptom | fix |
|---|---|
| "Clearly", "obviously", "simply", "just" | delete; if it were obvious the line would not need writing |
| "It can be shown that" | show it, or cite the exact theorem |
| A formula with an unintroduced symbol | type and name every symbol at first use |
| Analogy with no stated breaking point | name the seam |
| Prerequisites chapter at the front | move each piece to its point of need |
| Enthusiasm as explanation ("the beautiful insight is...") | replace with the mechanism |
| Restating the formula in words as if that explains it | explain what it *does*, then work a number |
| A claim with no attribution | cite, or mark as the author's own reasoning |

## Checks before shipping

- [ ] Every symbol typed and named at first use.
- [ ] Every formula preceded by the question it answers.
- [ ] Every derivation has a per-line justification.
- [ ] Every concept has a worked number.
- [ ] Every analogy has its seam named.
- [ ] Every prerequisite sits next to the passage needing it.
- [ ] Every empirical claim has a source; proved / measured / believed labelled.
- [ ] A reader who stops halfway has still learned something complete.
