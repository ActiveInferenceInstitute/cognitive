---
title: Knowledge Base and Tooling Review — 2026-10-09
type: review
status: complete
---

# Knowledge base and tooling review — 2026-10-09

The earlier passes (see [`REVIEW_LOG_2026-08-02.md`](REVIEW_LOG_2026-08-02.md)
and [`REVIEW_LOG_2026-08-18.md`](REVIEW_LOG_2026-08-18.md)) audited prose and the
package. This pass looked for structural damage the gates did not measure.

## Baseline

Python 3.12: ruff and mypy clean, 243 tests passing at 92.6% branch coverage,
`validate_docs` and both link checkers clean. Every gate was green, so each
finding below is a property the gates did not check.

## Findings

### 1. `semantic_relations` links were not links

753 pages stored their relations as nested YAML lists (`- - - active_inference`).
YAML parsed them without complaint, which is why no gate fired, but only a
quoted `"[[page]]"` is a link to Obsidian. The relations were invisible to the
graph view and to backlinks. `normalize_frontmatter.py` rewrites the block and
nothing else, and the rewrite was checked for semantic equality against the
parsed original on every file before it was applied. The link checker now also
sees these links, which exposed 18 explicit ones that pointed at pages that did
not exist; six were repointed or removed and the rest got pages.

### 2. A lossy conversion in the previous pass

The 2026-08-02 pass converted 879 `[[links]]` inside fenced code to plain text
on the grounds that Obsidian does not render links in fences. True, but the link
checkers already skip fences, so nothing was gained, and the pages that teach
linking syntax (`linking_standards.md`, `linking_template.md`, and others)
ended up showing "Preferred: free_energy_principle / Avoid: Free Energy
Principle". 309 lines in markdown, yaml and unlabeled fences were restored.
Python fences were left alone: the links there were metadata in illustrative
docstrings.

### 3. An equation that contradicted the package

`mathematics/expected_free_energy.md` set epistemic value plus risk equal to
"ambiguity plus risk". The package's corrected convention (see `CHANGELOG.md`,
1.1.0) is risk plus ambiguity, with the information gain already inside the
ambiguity. The page now says so.

### 4. Documentation that described things that do not exist

`project_structure.md` listed directories that were never created;
`docs/api/` described classes that are not importable;
`docs/agents/agent_docs_readme.md` ran to 1,574 lines of fictional content;
`DOCUMENTATION_ROADMAP.md` was 809 lines of 2024 planning prose. Each was
rewritten against the code and the real tree, and the Python examples in
`docs/api/` were executed.

### 5. Missing and thin knowledge-base pages

The most-referenced unresolved concept names got pages (158), 72 one-paragraph pages
carrying the boilerplate "this entry is a foundation" were rewritten, and
near-duplicate names were added as Obsidian aliases of the page that already
covers them. Worked examples were computed with the package.

### 6. Code defects

Eleven, each with a regression test that fails on the old code; see
`CHANGELOG.md` `[Unreleased]`. Coverage went from 92.6% to 96.8%. The mutation
gate declares five more mutations and kills all of them.

## A correction to the working assumption

The pass started from a count of roughly 400 pages with no inbound link. That
count was wrong: it compared absolute and relative paths. Recomputed correctly,
four pages had no inbound link, and those are domain index pages. The generated
catalogs are therefore a discoverability aid (every page with its opening line),
not a repair of orphaned content, and `kb_index.py` documents them that way.

## Second pass — 2026-10-10

The first pass left two kinds of debt it had measured but not closed: link names
without pages, and pages that described things that do not exist.

### 7. CI had been red for seven weeks

`quality.yml` installed `pandoc-crossref` with apt. Ubuntu does not package it, so
the test job failed at `Unable to locate package` on every run since 2026-08-19 and
never reached pytest. The earlier backlog entry that called the HTML render
"closed" had never been seen to run on a runner. Pandoc and pandoc-crossref are
now installed as a pinned pair from their release pages (crossref 0.3.26 is built
against Pandoc 3.12.1).

### 8. 510 link names with three or more inbound links

Three agents classified each as write, alias or unlink after reading how it was
used. 200 became pages, 124 existing pages gained aliases, and 140 generic labels
were unlinked (493 links in 187 files). Two weak aliases were demoted by hand. No
name with three or more links is now unresolved, apart from those covered by an
alias.

Most of the inbound links came from lists written by a generator, whose glosses
were nonsense ("Talk ways", "Copy save"). Reading those lists is what exposed the
larger problem, below.

### 9. Garbled glosses

7,491 lines in 207 pages have the shape `- [[term]] - Gloss`. 4,987 lines in 47
pages carried glosses that did not describe the term ("Warn ways", "Brain think",
"Nature boss"). Nine agents rewrote them from the section context, kept 1,170 that
were already accurate and dropped two whose meaning was unclear. A script checked
every changed line against the previous version: the text before the gloss is
identical on all of them and the line counts match.

### 10. Documentation that described the repository wrongly

Three more audits covered the parts of `docs/` the first pass skipped. Roughly 120
false claims were removed from `docs/implementation` and `docs/research`, including
invented RxInfer API names; about 8,500 lines of fabricated scaffolding from
`docs/guides`; and the Things README's claim that code-less folders were active
implementations. The RxInfer notes were rewritten from the official documentation.
The benchmark page now describes the four things the runner times.

### 11. References

Every page written or expanded in the two passes was checked against PubMed,
Crossref and publisher records, about 900 works in all. 41 details were wrong:
truncated or incorrect titles, an author list, page ranges, three findings credited
to the wrong paper, and a missing co-author. None of the papers was invented. Several
claims were also corrected because the cited paper did not say what the page
attributed to it, for example a sentence that credited a review with experimental
findings.

### 12. Code defects, second round

All four suspects from the first pass were reproduced and fixed
(`AdaptiveControl` ignoring `prediction_error`, `SimplePOMDP` ignoring its seed and
temperature, unbounded precision learning, an inconsistent `goal_prior` scale),
normalization was made scale-invariant in one shared helper, and the link scripts
now use a line-based fence tracker. A cross-vendor-style audit of the diff found
three regressions in the new tracker (fences inside blockquotes and on list-item
lines, a non-text link target with an anchor crashing the checkers) and two smaller
ones, all fixed with tests. The audit ran as manual probes against the working
tree, not as a second model's review.

### 13. Code fences that never closed

The new tracker exposed four pages whose generator wrote the closing fence as a
fence with an info string, which cannot close a block, so the rest of each page
rendered as code. The old pattern had hidden this. `validate_docs` now fails on an
unterminated fence.

## Two incidents

The system volume filled up twice during the pass. The first time the cause was a
virtual environment and package cache this pass created in a temporary directory;
they were moved to the repository volume. The second time most of the space was
another session's test artifacts, which this pass did not touch. Deleting them was
not permitted, so the remaining work kept its scratch files on the repository
volume. Several agents lost their shell for a few minutes and resumed afterwards;
each rewrite was re-verified against the staged version.

## Verification record

- ruff, mypy, 504 tests with branch coverage over the whole source tree (96.9%),
  `compileall`, the 31-mutation gate, the no-render manuscript build and the
  benchmark runner pass locally.
- `validate_docs`, `verify_links` and `check_markdown_links` report no errors.
- Reference lists on new pages were checked against online records; see the open
  list in the backlog file for what that did not cover.
