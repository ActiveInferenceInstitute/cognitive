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

## Major

- Open / deferred: migrate the remaining expansive pre-existing `AGENTS.md` and
  knowledge-base pseudo-code into either verified executable examples or clearly
  labelled conceptual pseudocode. The remaining surface is large (many hundreds
  of pages) and includes scholarship and domain notes; deleting it or rewriting
  it mechanically would risk losing meaning. The new authoring rules in
  `docs/guides/AGENTS.md`, `docs/examples/AGENTS.md`, and `docs/api/AGENTS.md`
  establish the boundary for a dedicated follow-up editorial migration.
- Open / deferred: add a first-class Markdown-link/anchor validator for standard
  Markdown links. The repository currently gates explicit Obsidian wiki links;
  the remaining standard links include intentional build-time figure paths and a
  vendored Documenter.jl syntax that requires renderer-aware handling. A safe
  validator should be introduced with allowlists for those contracts rather than
  failing on valid generated references.

## Verification record

- `git fetch origin` and `git pull --ff-only origin main`: up to date.
- Baseline explicit wiki-link scan: 23,691 checked, 0 broken file links,
  11,112 unresolved concept links intentionally skipped.
- Baseline documentation validator: 2 invalid frontmatter errors; both fixed.
- No private paths, credentials, secrets, or sibling repositories were modified.
