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
  duplicate pairs into `redirect` pages, linked all 20 orphaned pages into
  navigation, cross-linked 23 duplicate-topic groups with mutual "See also"
  sections, replaced unrendered `{{date}}` values, and added honest
  scope notes to the two pages with illustrative code. Files: `knowledge_base/`
  (131 files touched). Completed in this pass.
- ✓ Docs-content pass: removed 16,936 lines of fictional class-based code
  from 34 documentation pages; rewrote `ai_documentation_style.md`,
  `simulation.md`, `ant_colony_example.md`, and the stale root-structure
  diagram in `folder_structure.md`; replaced fictional template examples with
  the real package API; and normalized the "airy" frontmatter style in 341
  knowledge-base files. Completed in this pass.
- ✓ Tools-folder triage: wrote 18 accurate tool pages mapping to real
  tooling (validators, benchmark, build, CI, linting, coverage, packaging,
  plotting, static analysis, testing, configuration, setup), deleted 42
  content-free empty or boilerplate pages and 2 obsolete hub pages, fixed surviving
  dead links, and added scope banners to the kept conceptual pages.
  Completed in this pass.

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

- ✓ Subagent-assisted knowledge-base audit: three parallel review
  subagents audited all ~700 KB pages; every verified finding was fixed —
  fictional API collisions renamed/framed (BioFirm schema, InferenceConfig,
  ControlMode, ActiveInferenceModel, overview.md), uncited statistics
  framed or removed, 879 code-fence wiki links converted to plain text,
  broken anchors fixed, invalid BioFirm fragments stripped, GenericPOMDP
  README corrected, JAX guide banner added. Gate-gap tooling:
  `verify_links.py` gained advisory `#fragment` anchor validation and
  `validate_docs.py` rejects template frontmatter values, with 6 new
  regression tests. Completed in this pass.

## Open / deferred

- Deferred: a handful of kept conceptual `docs/tools/` pages
  (`model_context_protocol.md`, `obsidian_usage.md`, `network_analysis.md`,
  `advanced_linking.md`, `automation_scripts.md`, `cursor_integration.md`)
  remain broad conceptual write-ups. They are gate-clean and no longer
  reference deleted pages; a future editorial pass may consolidate them.
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
