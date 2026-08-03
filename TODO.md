---
title: Cognitive Active Inference — Documentation Backlog
type: backlog
status: active
owner: Active Inference Institute
last_reviewed: 2026-08-02
---

# Documentation backlog

Repository: `ActiveInferenceInstitute/cognitive`

Last reviewed: 2026-08-02 in a repository-wide DOCS-DEEP pass. The review
covered the root documentation, `docs/`, `knowledge_base/`, code-adjacent
README/AGENTS.md files, manuscript sources, configuration, and CI.

## Severity definitions

- Minor — typo, malformed metadata, broken link, or formatting correction.
- Medium — stale section rewrite, inaccurate command/API guidance, or focused
  documentation restructure.
- Major — cross-cutting documentation-system overhaul or migration requiring a
  broad editorial decision and validation of many conceptual files.

## Minor

- ✓ Fix invalid YAML frontmatter caused by unquoted wiki-link labels containing
  punctuation. Files: `knowledge_base/citations/AGENTS.md`,
  `knowledge_base/research/concepts/AGENTS.md`. Completed in this pass; commit
  reference: see the `docs:` commit containing the review fixes.
- ✓ Re-run explicit wiki-link validation and preserve the existing zero broken
  file-link result. Files: repository-wide. Completed in this pass.

## Medium

- ✓ Replace fictional setup/imports, example URLs, obsolete Python floor, and
  nonexistent paths in `docs/repo_docs/getting_started.md`. Completed in this
  pass.
- ✓ Replace the configuration overview's nonexistent `ConfigManager`/schema API
  and unsupported defaults with the actual root and example configuration files.
  File: `docs/config/README.md`. Completed in this pass.
- ✓ Replace the high-traffic guides landing page and examples instructions with
  accurate navigation and an explicit conceptual-vs-executable boundary. Files:
  `docs/guides/README.md`, `docs/guides/AGENTS.md`, `docs/examples/AGENTS.md`.
  Completed in this pass.
- ✓ Replace the oversized documentation hub with a concise navigation spine that
  points only to current repository domains and entry points. File: `docs/README.md`.
  Completed in this pass.
- ✓ Knowledge-base deep pass: normalized frontmatter on 69 files, corrected
  stale document counts in 5 domain READMEs, collapsed 2 byte-identical
  duplicate pairs into `redirect` stubs, linked all 20 orphaned pages into
  navigation, cross-linked 23 duplicate-topic groups with mutual "See also"
  sections, replaced unrendered `{{date}}` placeholders, and added honest
  scope notes to the two pages with illustrative code. Files: `knowledge_base/`
  (131 files touched). Completed in this pass.

## Major

- ✓ Migrate the expansive pre-existing `AGENTS.md` pseudo-code into accurate
  instruction or overview documents. All `docs/` instruction files
  (`docs/AGENTS.md`, agents, config, development, examples, guides,
  learning_paths, implementation, rxinfer, repo_docs, research, templates,
  tools), all knowledge-base domain overviews (`knowledge_base/AGENTS.md`,
  BioFirm, agents, biology, cognitive, free_energy_principle, mathematics,
  ontology, hyperspatial, philosophy, systems), and the archived
  `code/Things/` overviews (Baseball_Game, KG_Multi_Agent, Path_Network,
  ActiveInferenceInstitute) now describe only real repository content.
  Completed in this pass.
- ✓ Add a first-class validator for standard Markdown links and anchors.
  `code/scripts/check_markdown_links.py` checks `[text](path)` and image
  references (file existence) and reports heading-slug anchor mismatches as
  advisory warnings; it is wired into `validate_docs.py` and covered by
  `code/tests/test_check_markdown_links.py`. Completed in this pass.

## Open / deferred

- Deferred: the individual `docs/tools/*.md` topic pages (for example
  `git_tools.md`, `development_tools.md`) still describe speculative tooling.
  They now carry scope banners pointing to the real `cognitive-*` entry
  points and `code/scripts/`; a dedicated triage pass should merge or remove
  pages whose topics have no real tooling. Rewriting all ~80 pages in one
  pass was judged disproportionate churn.
- Deferred: 345 knowledge-base files use an "airy" frontmatter style (blank
  line between every YAML key). It is valid and gate-clean; the 4 navigation
  index files were normalized, and a sweep of the rest is available as a
  mechanical formatting follow-up if desired.
- Deferred: knowledge-base concept pages may contain illustrative pseudocode
  (now framed by the notes added to the two heaviest offenders and by the
  authoring rules in `knowledge_base/AGENTS.md`). Page-by-page editorial
  review of ~700 scholarship pages is a standing maintenance activity, not a
  single-pass migration.

## Verification record

- `git fetch origin` and `git pull --ff-only origin main`: up to date.
- Baseline explicit wiki-link scan: 23,691 checked, 0 broken file links,
  11,112 unresolved concept links intentionally skipped.
- Baseline documentation validator: 2 invalid frontmatter errors; both fixed.
- No private paths, credentials, secrets, or sibling repositories were modified.
