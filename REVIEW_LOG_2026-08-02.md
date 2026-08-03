# Documentation deep review — 2026-08-02

## Subagent-assisted knowledge-base audit (same day)

Dispatched three parallel review-only subagents over the full knowledge
base (cognitive+mathematics, biology+systems+philosophy+ontology,
research+citations+agents+BioFirm+FEP). Their findings were verified
against the files and fixed:

- Fictional API collisions: `class InferenceConfig`/`ControlMode`/
  `ActiveInferenceModel` redefined with different contracts than the real
  exports (renamed + framed as illustrative); BioFirm schema used
  nonexistent `InferenceMethod`/`PolicyType` members and a fictional
  `BioregionalStewardshipFactory` (rewritten with the real
  `ActiveInferenceFactory.create(config, model)`); `overview.md` claimed
  "our framework uses probabilistic programming" with a fictional API
  (reframed); `pomdp_structure.md` template frontmatter values and remaining
  fictional classes removed; `matrix_specifications.md` B-matrix example
  corrected to the package's `B[s_next, s_prev, a]` convention and E
  corrected from policy prior to action prior.
- Fabricated/uncited statistics: three spatial_web.md case-study outcome
  lists replaced with illustrative-scenario notes (earlier commit); two
  biology pages gained "indicative ranges, not repository measurements"
  notes; the "See the canonical package documentation" dangling phrase
  (26 occurrences in 33 files) replaced with a real pointer.
- 879 wiki links embedded inside code fences (which Obsidian never
  renders) converted to plain text across 59 files; the conversion
  deliberately skipped Python `[[0, 1, 2]]` list-index literals.
- Broken anchors fixed (`information_theory#kl-divergence` ->
  `#divergence-measures`, `variational_methods#variational-free-energy` ->
  `#variational-inference-framework`); 10 invalid BioFirm `#fragment`
  links stripped.
- Gate-gap tooling: `verify_links.py` now validates `#fragment` anchors
  against heading slugs (`anchor_warnings`, advisory) and
  `validate_docs.py` rejects template frontmatter values
  (`unique_identifier`, `timestamp`, `lorem_ipsum`); 6 new regression
  tests (13 in the two link-test files).
- Misc: GenericPOMDP README (docs-only framing, Parr 2019 -> 2022, stray
  `print(".3f")`, duplicate entry); JAX implementation guide banner;
  `[[[name]]]` triple-bracket frontmatter links normalized to block YAML
  in 5 files; citation "cited over N times" counts removed; two stale
  counts corrected; `metaverse.md` links format normalized;
  `cellular_intelligence.md` expanded from a minimal entry; two same-folder
  duplicate-stem pairs cross-linked; dangling "Re- " fragment fixed.

Final gate: 102 tests passed; ruff clean; mypy clean; `validate_docs.py`
ok (0 errors, 0 forbidden terms, 0 broken links, 0 anchor warnings).

## Docs-content and tooling pass (same day)

Audited all 288 `docs/` Markdown files for fabricated content (fake package
imports, fictional class definitions, example URLs) and removed or
corrected it:

- Removed 16,936 lines of fictional Python class code from 34 documentation
  pages (18 learning paths, 9 repo_docs pages, implementation patterns,
  research pages, guides, and 3 templates). All pages keep their educational
  prose, mermaid diagrams, and valid frontmatter.
- Replaced dangling template examples with the real package API (canonical
  dispatcher example) in `docs/templates/implementation_example.md`; removed
  `torch` (not a repository dependency) from templates.
- Rewrote `docs/repo_docs/ai_documentation_style.md` (previously ~700 lines
  with duplicated sections) as a concise, accurate standards page; rewrote
  `docs/repo_docs/simulation.md` (fictional `cognitive.simulation` ->
  real benchmark/manuscript tooling) and `docs/examples/ant_colony_example.md`
  (the ant-colony runtime is not shipped in this repository); fixed the
  fictional `cognitive_system.SemanticProcessor` example in
  `ai_semantic_processing.md`; corrected the stale root-structure diagram in
  `folder_structure.md`.
- docs/tools triage: wrote accurate pages for 18 topics with real tooling
  (validators, benchmark, build, CI, linting, coverage, packaging, plotting,
  static analysis, testing, configuration, setup), deleted 42 content-free
  empty or boilerplate pages and the two obsolete hub pages, fixed the surviving dead
  links (2 gate-breaking plus 5 fence-free references in kept pages), and
  added scope banners to `development_tools.md` and `git_workflow.md`.
- Normalized the "airy" frontmatter style (blank line between every YAML
  key) in 341 knowledge-base files via round-trip-verified YAML
  re-serialization.

Final gate: 96 tests passed; ruff clean; mypy clean; `validate_docs.py` ok
(0 errors, 0 forbidden terms, 0 broken links, 0 anchor warnings).

## Knowledge-base deep pass (same day)

Repository-wide knowledge-base review (`knowledge_base/`, 699 tracked files;
the `active-inference-journal/` and `journal-transcripts/` entries are
symlinks to external sibling repositories and were left untouched). Audits
run: frontmatter validity, wiki-link and standard-link resolution, inbound
link/orphan detection, duplicate stems and byte-identical files, stale
document-count claims, templater date values, and Python code blocks.

Implemented:

- Moved 53 misplaced YAML frontmatter blocks to the top of their files and
  added minimal frontmatter to 16 files that had none (69 files total; 6
  generated titles were fixed for YAML quoting).
- Replaced 10 unrendered `{{date}}` templater date values in the
  GenericPOMDP matrix/config/state-space notes with the folder creation date.
- Corrected stale document counts: cognitive 173->213, systems 31->46,
  biology 30+->45, philosophy 6->10, mathematics 140+->150+.
- Collapsed two byte-identical duplicate pairs into canonical pages with
  `redirect` frontmatter (systems/swarm_intelligence_implementation.md ->
  root; BioFirm/biofirm_active_inference_connections.md ->
  active_inference_connections.md).
- Linked all 20 orphaned pages (no inbound links) into navigation: 19 added
  to domain READMEs, `learning_roadmap` and `quality_assessment` added to
  the knowledge-base README.
- Cross-linked 23 duplicate-stem topic groups (48 files) with mutual
  "See also" sections; all pairs are distinct cross-domain treatments.
- Added illustrative-pseudocode scope notes to
  `cognitive/overview.md` and an honest scope banner to
  `biology/implementation_examples_social_insects.md` (its Ant Colony
  imports reference a runtime that is not shipped in this repository).
- Normalized frontmatter formatting of the 4 navigation index files.

Audit conclusion: the knowledge base contains no fabricated Python classes
(the earlier 1,317 "class" hits were Mermaid diagram declarations) and its
explicit wiki links resolve (23,502 checked, 0 broken).

Final gate: 96 tests passed; ruff clean; mypy clean; `validate_docs.py` ok
(0 errors, 0 forbidden terms, 0 broken links, 0 anchor warnings).

## Follow-up pass (same day)

Open major items from the first pass were implemented:

- Migrated all remaining expansive pre-existing `AGENTS.md` files to accurate
  instruction/overview documents: 11 in `docs/`, 11 knowledge-base domain
  overviews, 4 archived `code/Things/` overviews, plus accuracy fixes in
  `code/tests/AGENTS.md` and `CLAUDE.md`. Also rewrote 8 fictional
  `docs/**/README.md` navigation files, added the missing
  AGENTS.md/README.md pairs under `docs/research/`, added
  `docs/manuscript/AGENTS.md`, and added a scope banner to
  `docs/tools/git_tools.md`.
- Added `code/scripts/check_markdown_links.py` (standard Markdown link and
  anchor validator), wired it into `validate_docs.py`, added
  `code/tests/test_check_markdown_links.py` (10 tests), and documented the
  gate in `docs/development/README.md`.

Final gate (all measured): 96 tests passed; ruff clean; mypy clean;
`validate_docs.py` ok (0 errors, 0 forbidden terms, 0 broken wiki links, 0
broken standard links); `check_markdown_links.py` 67 links checked with 0
anchor warnings.

## Scope

Repository-wide review of Markdown documentation, README files, AGENTS.md files,
configuration documentation, API/example guidance, manuscript source, link
validation, and CI documentation. The repository contains 1,078 tracked Markdown
files and a Python package with executable documentation gates.

## Preflight

- Branch: `main`
- Remote default branch: `origin/main`
- Starting state: clean and up to date with `origin/main`
- Baseline explicit wiki-link gate: 23,691 links checked; 0 broken file links
- Baseline documentation validator: failed on 2 invalid YAML frontmatter files
- Baseline standard Markdown scan: 5 manuscript figure links are build-time outputs;
  no other actionable standard relative links found after excluding Documenter.jl
  `@ref`/`@id` syntax and LaTeX expressions.

## Findings and implementation

- Minor: two `semantic_relations.links` entries were unquoted YAML wiki links;
  punctuation in their display labels made frontmatter invalid. Fixed in the two
  affected `knowledge_base/**/AGENTS.md` files.
- Medium: the getting-started guide contained fictional package imports,
  example GitHub/community URLs, Python 3.8 guidance, and paths that do not
  exist in this repository. Replaced with a concise, executable guide grounded in
  the root README, `pyproject.toml`, and current test/build commands.
- Medium: configuration overview documented nonexistent `ConfigManager`, schemas,
  feature flags, and versioned defaults instead of the actual root `config.yaml` and
  `docs/config/simulation_config.yaml`. Replaced with an accurate configuration
  reference and explicitly separated repository configuration from runtime model
  configuration.
- Medium: the guides and examples landing pages mixed real navigation with large
  speculative pseudo-code sections. Replaced the guides landing page and examples
  agent guidance with accurate navigation/instructions; retained conceptual and
  research notes as non-executable material where appropriate.
- Major (deferred): the repository contains many older, expansive AGENTS.md and
  knowledge-base notes with illustrative pseudo-code that is not executable package
  API documentation. A complete content-model migration would require a dedicated
  editorial pass across the knowledge base; this review removes high-traffic false
  quickstarts and documents the remaining boundary in the project backlog rather
  than deleting scholarship or conceptual material wholesale.

## Verification

Final verification commands and measured results are recorded in the final report
and the project backlog. No secrets, private paths, or external repositories were
modified.
