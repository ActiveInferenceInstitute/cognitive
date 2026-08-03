---
title: Benchmarking Active Inference Agents
type: implementation_guide
id: fep_benchmarking_001
created: 2025-12-18
updated: 2026-08-02
tags:
  - free_energy_principle
  - benchmarking
  - active_inference
  - evaluation
aliases: [benchmarking_agents, agent_benchmarks]
semantic_relations:
  - type: implements
    links:
      - [[knowledge_base/free_energy_principle/implementations/python_framework]]
      - [[knowledge_base/free_energy_principle/implementations/simulation]]
  - type: relates
    links:
      - [[knowledge_base/free_energy_principle/implementations/simulation]]
      - [[docs/tools/benchmark_tools|Benchmark tools]]
---

# Benchmarking Active Inference Agents

This page documents how benchmarking is done in this repository. No
benchmark experiments with results are recorded in the knowledge base;
the repository's benchmark tooling and its measured results live in the
executable package and the manuscript.

## Repository benchmark tooling

```bash
cognitive-benchmark --repetitions 1
```

`cognitive-benchmark` (backed by `cognitive.benchmarks`,
`code/tools/src/benchmarks.py`) runs the repository's benchmark suite and
emits a JSON-serializable report. Deterministic runtime evidence —
including any measured numbers — is produced by the executable manuscript:

```bash
cognitive-build-manuscript --output build/manuscript
```

## Reporting conventions

- Measured results belong in the manuscript (`docs/manuscript/`), which
  hydrates its numeric values and figures from the package API.
- Knowledge-base pages must not contain invented statistics, tables, or
  error bars. Comparisons against external baselines (for example
  Q-learning or policy-gradient methods) are only permissible with real,
  cited sources or real runs.
- Earlier drafts of this page contained fabricated benchmark tables with
  invented values and uncertainties. Those tables were removed; they did
  not correspond to any experiment in this repository.
