---
title: Simulation
type: documentation
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - simulation
  - documentation
  - benchmarks
semantic_relations:
  - type: documents
    links:
      - '[[getting_started|Getting started]]'
      - '[[unit_testing|Testing]]'
      - '[[../../manuscript/README|Executable manuscript]]'
---

# Simulation

This repository validates its runtime behavior through deterministic
simulation: seeded models, reproducible benchmarks, and an executable
manuscript. There is no separate `cognitive.simulation` module; the
simulation surface is the package API plus the tooling below.

## Running the benchmarks

```bash
cognitive-benchmark --repetitions 1
```

`cognitive-benchmark` runs the repository's benchmark suite (discrete
inference, continuous agent, visualization, and knowledge-base tooling) and
emits a JSON-serializable report. See `cognitive.benchmarks` in the package
source for details.

## Deterministic model runs

The discrete dispatcher is seeded through `InferenceConfig(seed=...)`; the
root `README.md` example is the canonical minimal simulation. Model
parameters are supplied in Python (matrix arrays, configuration objects),
not read from a global simulation config file.

## Reproducible experiments

- `code/tests/` — behavior checks; tests write artifacts only to temporary
  directories.
- `docs/manuscript/` — the executable manuscript hydrates its numeric
  results and figures from the package API via
  `cognitive-build-manuscript --output build/manuscript`.
- `cognitive.benchmarks` (`code/tools/src/benchmarks.py`) — benchmark sources
  behind the `cognitive-benchmark` entry point.

## Related

- [[getting_started]] — install and first run.
- [[unit_testing]] — test conventions.
- `docs/manuscript/README.md` — end-to-end reproducible build.
