# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cognitive Modeling Framework built on Active Inference principles. The repository is both a Python codebase and an Obsidian knowledge vault, with extensive markdown documentation interconnected via `[[wiki-style links]]`.

**Licenses**: Code is MIT; repository-owned knowledge-base content and graph exports are CC BY 4.0 (see LICENSE-KNOWLEDGE-BASE.md); other documentation is CC BY-NC-SA 4.0.

## Repository Architecture

```
cognitive/
  code/
    tools/src/          # Core Python framework
      models/active_inference/  # ActiveInferenceModel base class, dispatcher
      models/matrices/          # Matrix operations (MatrixOps, MatrixInitializer)
      utils/                    # matrix_utils, create_node, visualization helpers
      visualization/            # MatrixPlotter
      analysis/                 # Analysis tools
    Things/               # Two executable agents; eight documentation-only designs
      Simple_POMDP/       # Executable discrete educational agent
      Continuous_Generic/ # Executable continuous agent and visualizer
      Generic_Thing/      # Documentation only; also contains saved PNG images
      Generic_POMDP/      # Documentation-only design
      Ant_Colony/         # Documentation-only design
      BioFirm/            # Documentation-only design
      Path_Network/       # Documentation-only design
      KG_Multi_Agent/     # Documentation-only design
      Baseball_Game/      # Documentation-only design
      ActiveInferenceInstitute/ # Documentation-only design
    tests/               # Centralized test suite
    scripts/             # Utility scripts (e.g., verify_links.py)
  knowledge_base/        # Theoretical foundations (cognitive, math, biology, philosophy, ontology)
  docs/                  # Guides, API docs, examples, research, RxInfer integration docs
  config.yaml            # Global configuration (inference params, visualization, logging)
```

### Key Design Patterns

- **ActiveInferenceModel** (`code/tools/src/models/active_inference/base.py`):
  Abstract model interface for belief updating, policy inference, precision,
  and free energy. Standalone Things need not inherit from it.
- **ActiveInferenceDispatcher** (`dispatcher.py`): Implements `variational`,
  `sampling`, and `mean_field` inference with `PolicyType.DISCRETE` only.
  `Things.Continuous_Generic` is a standalone continuous implementation;
  hierarchical episodes compose controllers rather than introducing a
  dispatcher policy type.
- **Discrete matrix system**: A is the observation likelihood, B the
  transition tensor, C log observation preferences, D the state prior,
  and E the action prior. Shapes and probability constraints are validated.
- **Things boundary**: Only `Simple_POMDP` and `Continuous_Generic` are
  executable agents. The other eight folders are documentation-only designs;
  do not describe them as implemented agents or infer private environments
  from their names.

## Tests and Gates

Run from the repository root through the shared entry point:

```bash
./run.sh                 # defaults to test
./run.sh lint            # Ruff lint + format check + compileall
./run.sh types           # mypy
./run.sh test            # selected core-model tests only; no coverage
./run.sh test-full       # whole suite with branch coverage
./run.sh docs            # documentation, generated freshness, and OKN checks
./run.sh gnn             # validator + conformance; pinned runtime required
./run.sh manuscript      # no-render + HTML; --pdf adds PDF
```

The wrapper invokes
`uv run --locked --extra dev --python 3.12 python code/scripts/gate.py ARM`
using `uv.lock`. The GNN parser stays in an independent interpreter named by
`GNN_RUNTIME_PYTHON`; `./run.sh gnn --provision` explicitly permits network
provisioning. Do not treat an unavailable parser as a conformance pass.

Coverage scope and its floor come from `pyproject.toml`; do not narrow
`test-full` with `--cov=<path>`. Test configuration is in `pytest.ini` and
`pyproject.toml`; shared fixtures are in `code/tests/conftest.py`, which selects
Matplotlib's non-interactive `Agg` backend.

See [development resources](docs/development/README.md) for tool prerequisites,
benchmark and mutation commands. CI consolidates core checks in `tests` while
retaining the independent `gnn-conformance` job, display name
`gnn conformance (pinned)`. Hosted Actions execution is currently blocked by the
organization's billing lock; workflow definitions are not hosted-success
receipts. Measurements belong in the generated
[gate-measurements.md](docs/development/gate-measurements.md), sourced from
[gate-measurements.json](docs/development/gate-measurements.json).

## Dependencies

The shared gates use Python 3.12 and locked development dependencies. Core dependencies (per `pyproject.toml`): `click`, `jinja2`,
`matplotlib`, `networkx`, `numpy`, `Pillow`, `plotly`, `PyYAML`, `seaborn`.
Dependency and runtime authority is the root `pyproject.toml`, `uv.lock`,
and shared gate configuration. Documentation-only Things are not separate
executable environments.

## Conventions

### Documentation
- `docs/README.md` is curated navigation; `docs/documentation-index.md` is the generated flat index. Regenerate it with `uv run --locked --extra dev --python 3.12 python code/scripts/docs_index.py --write .`, then run `./run.sh docs`.
- Every folder must have both `AGENTS.md` (technical documentation) and `README.md` (overview/navigation).
- Markdown files use YAML frontmatter with `semantic_relations` for Obsidian graph connections.
- Use `[[wiki-link]]` syntax for internal cross-references. The repo is an Obsidian vault.

### Code Style
- Python: snake_case for files and methods. Type hints required. Complete docstrings on public methods.
- **No test fixture methods in tests** - always use real data/computation.
- Remove non-semantic adjectives from names ("enhanced_", "new_", "improved_" are banned; "continuous_time", "hierarchical_" are fine).

### Configuration
- Agent configurations use YAML files specifying matrix shapes, constraints, initialization methods, and inference parameters.
- Global config at `config.yaml` defines default Active Inference parameters (precision, temporal_horizon, learning_rate, etc.).
