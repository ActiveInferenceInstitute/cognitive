---
title: Documentation roadmap
type: roadmap
status: stable
updated: 2026-10-09
tags:
  - roadmap
  - documentation
  - maintenance
semantic_relations:
  - type: implements
    links:
      - "[[repo_docs/documentation_standards]]"
      - "[[repo_docs/ai_documentation_style]]"
  - type: organizes
    links:
      - "[[README]]"
      - "[[AGENTS]]"
      - "[[repo_docs/content_management]]"
  - type: documents
    links:
      - "[[repo_docs/folder_structure]]"
      - "[[knowledge_base/linking_standards]]"
      - "[[repo_docs/obsidian_linking]]"
      - "[[knowledge_base/index]]"
---

# Documentation roadmap

A map of the documentation system as it exists, how the gates keep it
accurate, and the documentation work that remains. It describes the
repository; it is not a plan for features that do not exist.

## Layout

The repository is two things at once: a Python package (`cognitive` in
`code/tools/src`, `Things` in `code/Things`, scripts in `code/scripts`) and an
Obsidian vault of Markdown pages linked with `[[wiki links]]`.

| Location | Purpose |
|---|---|
| `README.md`, `CHANGELOG.md`, `CLAUDE.md`, `AGENTS.md` | Install, quality gates, release notes, agent instructions |
| `docs/` | Documentation for the implemented package and the process around it |
| `knowledge_base/` | Conceptual material: theory, domain context, citations |
| `code/Things/*/` | One folder per agent family; only `Simple_POMDP` and `Continuous_Generic` contain code, the rest are documentation |
| `docs/manuscript/` | The executable manuscript and its `config.yaml` |

### `docs/` folders

| Folder | Purpose |
|---|---|
| `api/` | Exports, signatures, executed examples, version policy |
| `config/` | The root `config.yaml` and which keys code reads |
| `development/` | Development loop, contribution rules |
| `agents/` | Navigation for agent documentation |
| `examples/` | Example entry points |
| `guides/` | Application notes and learning paths |
| `implementation/` | Implementation patterns and a vendored RxInfer documentation snapshot (`implementation/rxinfer/`) |
| `research/` | Research methods and application notes |
| `repo_docs/` | Documentation standards, linking, naming, testing guidelines |
| `templates/` | Page templates |
| `tools/` | One page per shipped tool or gate |
| `policy/` | `documentation_terms.yaml`, the term policy `validate_docs.py` enforces |
| `manuscript/` | Chapters, `config.yaml`, `layer_contract.yaml`, `references.bib` |

### `knowledge_base/` domains

`BioFirm`, `agents`, `biology`, `citations`, `cognitive`,
`free_energy_principle`, `mathematics`, `ontology`, `philosophy`, `research`,
`systems`, plus top-level pages (`index.md`, `glossary.md`,
`linking_standards.md`, `learning_roadmap.md`, `quality_assessment.md`). Each
domain has a `README.md` and an `AGENTS.md`. These pages are conceptual: they
may contain illustrative code and theory beyond the runtime, and code behavior
is defined by the package and its tests.

## Conventions

- Every content folder has `README.md` (overview and navigation) and
  `AGENTS.md` (instructions for authors and agents).
- Markdown files start with YAML frontmatter. Links in `semantic_relations` are
  quoted strings such as `"[[page]]"`; unquoted links parse as nested lists and
  disappear from Obsidian's graph and backlinks.
- Cross-references use `[[wiki links]]`; links to code or data files carry
  their extension.
- Pages describe only what exists. Commands are run before they are documented,
  and Python examples import from the installed packages.

## Gates

`validate_docs` is the aggregate gate and runs in CI
(`.github/workflows/quality.yml`, "Documentation and links"). `verify_links`
also runs there as its own step. The scripts live in `code/scripts/`.

| Gate | Command | Enforces |
|---|---|---|
| `validate_docs` | `python code/scripts/validate_docs.py --json` | Frontmatter parses as a mapping and has no template values; `semantic_relations` links are quoted; every `python` fence compiles; each name in `cognitive.__all__` exists; the term policy holds; explicit wiki links and standard Markdown links resolve; knowledge-base catalogs are current; manuscript references resolve |
| `verify_links` | `python code/scripts/verify_links.py . --json` | Wiki links that name a file resolve. Extensionless concept links are counted as skipped unless `--strict-wiki-links` is passed. Reports `anchor_warnings` for `#fragment` mismatches (advisory) |
| `check_markdown_links` | `python code/scripts/check_markdown_links.py . --json` | `[text](target)` links and `#anchor` fragments (anchors advisory). Skips the RxInfer snapshot and generated manuscript figures |
| `normalize_frontmatter` | `python code/scripts/normalize_frontmatter.py --check .` (`--write` fixes) | Keeps `semantic_relations` links as quoted wiki links. Only that block is rewritten |
| `kb_index` | `python code/scripts/kb_index.py --check .` (`--write` regenerates) | Every `knowledge_base` domain carries a generated `catalog.md` listing each page beneath it; `knowledge_base/catalog.md` lists the domains |

Notes on the gates:

- Catalogs are derived. Change a page's `title`, `description`, or opening
  paragraph and regenerate; never edit a `catalog.md` by hand. A page added
  without regenerating makes `validate_docs` fail.
- The term policy in `docs/policy/documentation_terms.yaml` rejects a fixed list
  of words that mark incomplete content, in documentation and in source files.
  Outstanding work goes in the tracked backlog at the repository root, not in
  page text.
- `validate_docs` compiles Python fences; it does not run them. Changed
  snippets must be run by hand.

## Build and test gates that touch documentation

- `cognitive-build-manuscript` builds the manuscript from
  `docs/manuscript/config.yaml`; `--no-render` skips Pandoc and XeLaTeX. CI
  asserts that the HTML render has no unresolved `{{` or `[@...]` references.
- `python -m scripts.mutation_check` checks that the test suite rejects declared
  source mutations; see `docs/tools/mutation_testing.md`.
- The development loop is listed in `docs/development/README.md`.

## Open documentation work

1. Knowledge-base catalogs: run `python code/scripts/kb_index.py --write .` and
   commit the generated `catalog.md` files; until then `validate_docs` reports
   them as missing, and READMEs that link to them do not resolve.
2. Concept links: roughly 9,900 extensionless wiki links in the vault (for
   example `[[some_concept]]`) do not resolve to a page. The default gate skips
   them; `verify_links.py --strict-wiki-links` lists them. Each needs a page,
   a retarget to an existing page, or removal.
3. Executable examples: about 120 Python fences under `docs/` and about 1,400
   under `knowledge_base/` are compile-checked only. Only the pages in
   `docs/api/` were run against the current API. The rest should be run, reduced
   to what the package supports, or marked as illustrative.
4. Conceptual pages (`docs/guides/`, `docs/examples/`, `docs/research/`,
   `code/Things/README.md`) have not had a claim-by-claim check against `code/`.
   Several describe agent families whose folders under `code/Things/` hold no
   code, and `code/Things/README.md` lists some of them as active
   implementations.
5. Console scripts: `pyproject.toml` declares eight; CI runs `--help` on seven
   and exercises `cognitive-mutation-check` as a separate job. An environment
   installed before `cognitive-kb-index` and `cognitive-normalize-frontmatter`
   were added needs `python -m pip install -e ".[dev]"` again.
6. `docs/config/simulation_config.yaml` is a parameter sketch that no code loads;
   either add a loader and a test, or remove it.
