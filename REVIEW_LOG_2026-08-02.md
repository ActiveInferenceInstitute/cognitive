# Documentation deep review — 2026-08-02

## Knowledge-base deep pass (same day)

Repository-wide knowledge-base review (`knowledge_base/`, 699 tracked files;
the `active-inference-journal/` and `journal-transcripts/` entries are
symlinks to external sibling repositories and were left untouched). Audits
run: frontmatter validity, wiki-link and standard-link resolution, inbound
link/orphan detection, duplicate stems and byte-identical files, stale
document-count claims, templater placeholders, and Python code blocks.

Implemented:

- Moved 53 misplaced YAML frontmatter blocks to the top of their files and
  added minimal frontmatter to 16 files that had none (69 files total; 6
  generated titles were fixed for YAML quoting).
- Replaced 10 unrendered `{{date}}` templater placeholders in the
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
