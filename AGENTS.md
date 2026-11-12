---
title: Cognitive Agents Index
type: agents
status: stable
created: 2026-02-06
updated: 2026-10-10
tags:
  - agents
  - index
  - cognitive_modeling
semantic_relations:
  - type: organizes
    links:
      - "[[code/Things/README]]"
      - "[[code/tools/src/models/active_inference/AGENTS]]"
      - "[[knowledge_base/AGENTS]]"
---

# Cognitive Agents Index

Entry point for the agent material in this repository. It separates what can
be imported and run from what is design documentation only. The authoritative
list of importable agents is [[docs/agents/agent_docs_readme|the agent documentation clearinghouse]].

## Repository layout

- `code/tools/src/` — the installed `cognitive` package: settings, simulation,
  episodes, the dispatcher, models, visualization, benchmarks and the `cognitive`
  command.
- `code/Things/` — the installed `Things` package with the runnable
  `Simple_POMDP` and `Continuous_Generic` agents; the other folders are design
  notes.
- `code/scripts/` — the repository tooling and validation gates.
- `code/tests/` — the pytest suite.
- `docs/` — documentation tree; `docs/README.md` is the curated hub. The generated
  `docs/documentation-index.md` recursively lists long-lived Markdown, including
  manuscript and tools pages, excluding agent instructions and the index itself.
  `knowledge_base/` — the Obsidian vault; `examples/` — runnable scripts.

## Tests and gates

Use the shared root entry point; it invokes
`uv run --locked --extra dev --python 3.12 python code/scripts/gate.py ARM`
against `uv.lock`:

```bash
./run.sh                 # defaults to test
./run.sh lint            # Ruff lint + format check + compileall
./run.sh types           # mypy
./run.sh test            # selected core-model tests only; no coverage
./run.sh test-full       # whole suite with branch coverage
./run.sh docs            # documentation, generated freshness, and OKN gates
./run.sh gnn             # validator + conformance with pinned parser runtime
./run.sh manuscript      # no-render + HTML; --pdf adds PDF
```

`gnn` requires a separately provisioned `GNN_RUNTIME_PYTHON`; use
`./run.sh gnn --provision` only when network provisioning is intended.
`test-full` applies the branch-coverage floor in `pyproject.toml` across the
configured source tree. The [development guide](docs/development/README.md)
documents prerequisites, index generation, additional benchmark and mutation
checks, and the consolidated CI contract. Hosted Actions execution is currently
blocked by the organization's billing lock; do not infer hosted success from
configuration or local measurements.

## Agents that run

- **Discrete Active Inference runtime** (`code/tools/src/models/active_inference/`):
  `DiscreteGenerativeModel`, `ActiveInferenceDispatcher`, `HomeostaticFactory`,
  all importable from `cognitive`. [[code/tools/src/models/active_inference/AGENTS|Documentation]]
- **Simple POMDP** (`code/Things/Simple_POMDP/`): discrete Active Inference agent
  with YAML configuration. [[code/Things/Simple_POMDP/AGENTS|Documentation]]
- **Continuous Generic** (`code/Things/Continuous_Generic/`): continuous
  state-space agent, `Things.Continuous_Generic.ContinuousActiveInference`.
  [[code/Things/Continuous_Generic/AGENTS|Documentation]]

## Agent designs without code

These folders under `code/Things/` and `knowledge_base/` hold documentation
only. Treat them as design notes.

- **Generic POMDP** (`knowledge_base/agents/GenericPOMDP/`): configuration and
  state-space notes. [[knowledge_base/agents/GenericPOMDP/AGENTS|Documentation]]
- **Continuous time** (`knowledge_base/agents/Continuous_Time/`): continuous-time
  agent notes. [[knowledge_base/agents/AGENTS|Documentation]]
- **Ant Colony** (`code/Things/Ant_Colony/`): swarm and stigmergy design notes.
  [[code/Things/Ant_Colony/AGENTS|Documentation]]
- **BioFirm** (`knowledge_base/BioFirm/`): biological firm theory.
  [[knowledge_base/BioFirm/AGENTS|Documentation]]
- **Generic Thing** (`code/Things/Generic_Thing/`): base entity notes and saved images.
  [[code/Things/Generic_Thing/AGENTS|Documentation]]

The full folder index is [[code/Things/README]].

## Knowledge base

- [[knowledge_base/AGENTS|Active Inference Knowledge Base]]
- [[knowledge_base/cognitive/active_inference_agent|Active Inference Agent Theory]]
- [[knowledge_base/agents/architectures_overview|Agent Architectures Overview]]
- [[knowledge_base/agents/index|Agent Architecture Index]]

## Development tools

- [[code/tools/AGENTS|Cognitive tools instructions]]: the package source,
  console scripts, and test commands.

Most subdirectories of `code/Things/` and `knowledge_base/` have their own
`AGENTS.md` and `README.md`.
