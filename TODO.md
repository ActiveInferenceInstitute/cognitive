---
title: Cognitive Active Inference — Documentation Backlog
type: backlog
status: active
owner: Active Inference Institute
last_reviewed: 2026-10-09
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

## Code and mathematics pass — 2026-08-18

Recorded in [`REVIEW_LOG_2026-08-18.md`](REVIEW_LOG_2026-08-18.md). Closed in
that pass: expected-free-energy double counting, a second disagreeing EFE
implementation, the continuous gradient's missing Jacobian transpose, two
homeostatic paths correct only when two spaces happened to have equal size, an
entropy axis mismatch, a coverage report filtered to five of twenty-five
modules, a documentation gate that evaded its own scan, a rewriter that
enforced the gate by renaming, an inert `exploration_weight`, and a published
figure that could not distinguish its own actions.

## Knowledge-base and tooling pass — 2026-10-09

Recorded in [`REVIEW_LOG_2026-10-09.md`](REVIEW_LOG_2026-10-09.md). Closed in
that pass: nested-list `semantic_relations` frontmatter on 753 pages, fenced
link examples flattened by the 2026-08-02 pass, an expected-free-energy
decomposition that contradicted the package, 158 missing concept pages, 72
one-paragraph pages, a project-structure map of directories that do not exist, fictional
API and agent pages in `docs/`, and eleven boundary and numerical defects in
`code/` (listed in `CHANGELOG.md` under `[Unreleased]`).

## Open

- Verify the references on the pages written or expanded on 2026-10-09; they
  were compiled from memory and not checked against the sources.
- About 7,300 distinct `[[concept]]` link targets still have no page, 510 of
  them linked three or more times. Write the ones that are real concepts and
  unlink the ones that are not (`new_methods`, `validation_framework`,
  `version_control`).
- About fifty pages open with a formula and appear in their catalog by title
  only; give them a `description:` in frontmatter.
- Suspected, unproven, in `code/`: `AdaptiveControl.compute_policy_prior`
  ignores `prediction_error`; `SimplePOMDP` accepts `inference.seed`,
  `temperature` and horizon keys it does not use; precision learning in
  `ContinuousActiveInference` is unbounded; `goal_prior` is scaled by
  `1/temperature` in variational and sampling modes but not in mean-field.
- Heading-anchor collection in `verify_links.py` and `check_markdown_links.py`
  uses a non-greedy fence regex, so a comment inside a nested or unterminated
  fence is still read as a heading. A line-based fence tracker would fix both.
- The `EPS` floor in the dispatcher, homeostatic, `MatrixOps` and
  `ensure_matrix_properties` normalizers still treats total mass below `1e-12`
  as empty; only `DiscreteGenerativeModel.posterior` and `SimplePOMDP` were fixed.
- `code/Things/README.md` lists code-less folders as active implementations.

## Open / deferred

- Closed 2026-08-19: mutation testing. Two harnesses were written in parallel
  and consolidated into one, `code/scripts/mutation_check.py`, reachable as
  `cognitive-mutation-check` and run as its own CI job. It declares 17
  mutations covering the corrected mathematics, the dispatcher parameters, and
  the documentation term policy; all are killed. Two properties from the merge
  are load-bearing: a fresh `PYTHONPYCACHEPREFIX` per run, because a same-width
  edit leaves a file's size and modification time unchanged and Python then
  loads stale bytecode, and a kill criterion that requires a reported test
  failure rather than any non-zero exit, because a broken import proves only
  that the file no longer parses. `code/tests/test_mutation_check.py` proves the
  harness can report a survivor.
- Closed 2026-08-19: coverage in the named gaps. Branch coverage across every
  configured module is 92.6% against a floor of 90, up from a measured 70% when
  the report still carried an include filter.
  `Things/Continuous_Generic/continuous_generic.py` is at 100%,
  `Things/Simple_POMDP` at 98%, and `verify_links` at 93%.
- Closed 2026-08-19: CI installs `pandoc` and `pandoc-crossref`, so the HTML
  render runs there instead of skipping, and the workflow asserts the rendered
  document carries no unresolved cross-reference or unhydrated variable. The
  PDF stage still skips because the runner has no XeLaTeX.
- Closed 2026-08-19: four dispatcher parameters that the suite executed but did
  not check are now pinned by `test_dispatcher_semantics.py` — the belief
  blend's `learning_rate`, the policy `temperature`, the precision clip bounds,
  and the policy aggregation. All four were found by the mutation gate.

- Deferred: decide whether `policy_aggregation` should default to `marginal`
  rather than `min` in a future major version. `marginal` is the standard
  Bayesian model average; `min` is the optimistic behaviour this package has
  always had, and 1.1.0 already changed the objective, so changing the
  aggregation in the same release would make the two effects hard to separate.
- Deferred: extend the mutation set beyond the mathematics into the
  documentation and link gates. One mutation currently covers
  `validate_docs.py`, and none covers `verify_links.py`.
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
