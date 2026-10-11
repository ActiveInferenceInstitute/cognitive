---
title: Project Structure
type: reference
status: stable
updated: 2026-10-09
---

# Project Structure

The repository is two things at once: a Python package with its validation
tooling, and an Obsidian vault of linked Markdown. This page maps both. Every
directory named below exists; the quality gates in
[`README.md`](README.md#quality-gates) fail when a documented link stops
resolving.

## Top level

| Path | Purpose |
|------|---------|
| `code/` | Python source, tests, and tooling scripts |
| `knowledge_base/` | The theory vault: concepts, mathematics, biology, philosophy, ontology |
| `docs/` | Guides, API notes, the executable manuscript, templates, and repository policy |
| `config.yaml` | Default inference, visualization, and logging parameters |
| `pyproject.toml` | Package metadata, entry points, ruff, mypy, pytest, and coverage configuration |
| `CHANGELOG.md` | Release notes; entries that change a computed result are marked **behaviour change** |
| tracked backlog file | Open work only; the documentation policy forbids work markers elsewhere |
| `AGENTS.md`, `CLAUDE.md` | Operating guidance for AI agents working in the repository |
| `.github/workflows/quality.yml` | CI: tests, static checks, documentation gates, manuscript build, mutation gate, packaging |
| `.obsidian/` | Vault settings so the Markdown opens as a graph |

## `code/`

| Path | Contents |
|------|----------|
| `code/tools/src/` | Installed as the `cognitive` package |
| `code/tools/src/models/active_inference/` | `ActiveInferenceModel` base class, `ActiveInferenceDispatcher`, `DiscreteGenerativeModel`, and the homeostatic module |
| `code/tools/src/models/matrices/` | `MatrixOps`, `MatrixLoader`, `MatrixInitializer` |
| `code/tools/src/utils/` | Matrix helpers, node creation, network visualization |
| `code/tools/src/visualization/` | `MatrixPlotter`, `StateSpacePlotter`, `NetworkPlotter` |
| `code/tools/src/benchmarks.py` | The `cognitive-benchmark` runner |
| `code/Things/` | Installed as the `Things` package. `Simple_POMDP` and `Continuous_Generic` are executable; the other folders hold documentation for agent designs (`Generic_Thing`, `Ant_Colony`, `BioFirm`, `Path_Network`, `KG_Multi_Agent`, and others) |
| `code/scripts/` | Installed as `scripts`: `validate_docs`, `verify_links`, `check_markdown_links`, `normalize_frontmatter`, `kb_index`, `build_manuscript`, `mutation_check` |
| `code/tests/` | The centralized suite; `conftest.py` holds the shared fixtures and sets the Matplotlib `Agg` backend |

Console scripts declared in `pyproject.toml`: `cognitive-create-node`,
`cognitive-verify-links`, `cognitive-validate-docs`,
`cognitive-normalize-frontmatter`, `cognitive-kb-index`, `cognitive-benchmark`,
`cognitive-build-manuscript`, `cognitive-mutation-check`.

## `knowledge_base/`

Each domain folder carries a generated `catalog.md` listing every page beneath
it with its title and opening line; [`knowledge_base/catalog.md`](knowledge_base/catalog.md)
lists the domains. Start there to find a page, or at
[`knowledge_base/index.md`](knowledge_base/index.md) to read in order.

| Domain | Subject |
|--------|---------|
| `cognitive/` | Cognitive science, neuroscience, and psychology as they bear on Active Inference |
| `mathematics/` | Probability, information theory, optimization, dynamical systems, geometry, worked examples |
| `free_energy_principle/` | The free-energy principle by sub-field: mathematics, biology, systems, applications, implementations |
| `agents/` | Agent architectures: generic POMDP, continuous time, and related designs |
| `biology/` | Biological foundations, homeostasis, ecology, social insects |
| `systems/` | Complex systems, swarms, self-organization, governance and infrastructure topics |
| `philosophy/` | Epistemology and philosophy of science relevant to the framework |
| `ontology/` | Formal ontologies of the vault's own concepts |
| `research/` | Active research areas and their open questions |
| `citations/` | One note per cited work |
| `BioFirm/` | Biological firm theory and its Active Inference formulation |

Cross-cutting pages at the top of the folder: `glossary.md`,
`learning_roadmap.md`, `linking_standards.md`.

Conventions are in [`linking_standards.md`](knowledge_base/linking_standards.md):
YAML frontmatter on every page, `semantic_relations` links written as quoted
wiki links (`"[[page]]"`), and `[[wiki links]]` for internal references. The
frontmatter rule and catalog freshness are enforced by `validate_docs`.

## `docs/`

| Path | Contents |
|------|----------|
| `docs/guides/` | Learning paths and how-to guides |
| `docs/api/` | API notes for the package |
| `docs/examples/` | Examples, with an explicit boundary between conceptual and executable ones |
| `docs/manuscript/` | The executable manuscript source, built by `cognitive-build-manuscript` |
| `docs/implementation/` | Implementation notes, including RxInfer integration |
| `docs/research/` | Research notes and plans |
| `docs/repo_docs/` | How this repository is organized and maintained |
| `docs/templates/` | Page templates for new knowledge-base and documentation files |
| `docs/tools/` | One page per tool in the quality toolchain |
| `docs/config/`, `docs/development/`, `docs/agents/` | Configuration, contributor workflow, agent guidance |
| `docs/policy/` | The forbidden-term policy read by `validate_docs` |

## Conventions that apply everywhere

- Every folder has an `AGENTS.md` (technical detail) and a `README.md` (overview and navigation).
- Python files and methods use `snake_case`, public functions carry type hints and docstrings, and names avoid non-semantic adjectives such as `enhanced_` or `new_`.
- Tests use real computation; the suite writes artifacts only to temporary directories.
- Code is MIT licensed (see `LICENSE`); the documentation licence is stated in [`docs/README.md`](docs/README.md).
