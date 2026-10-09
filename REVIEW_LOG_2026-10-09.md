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

## Verification record

- ruff, mypy, 362 tests with branch coverage over the whole source tree,
  `compileall`, the 22-mutation gate, the no-render manuscript build and the
  benchmark runner pass locally.
- `validate_docs`, `verify_links` and `check_markdown_links` report no errors.
- Reference lists on new pages are unverified; see the open list in the backlog file.
