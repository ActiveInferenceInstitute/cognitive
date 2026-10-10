---
title: Simulation
type: documentation
status: stable
created: 2024-01-01
updated: 2026-10-10
tags:
  - simulation
  - documentation
  - benchmarks
semantic_relations:
  - type: documents
    links:
      - "[[docs/repo_docs/getting_started|Getting started]]"
      - "[[docs/repo_docs/unit_testing|Testing]]"
      - "[[docs/manuscript/README|Executable manuscript]]"
---

# Simulation

This repository checks its runtime behavior through seeded models, a small
benchmark runner, and an executable manuscript. There is no `cognitive.simulation`
module and no global simulation runner; the simulation surface is the package
API plus the tooling below.

## Benchmarks

```bash
cognitive-benchmark --repetitions 1 --output benchmarks.json
```

`cognitive-benchmark` (`code/tools/src/benchmarks.py`, flags `--repetitions`
and `--output`) times four operations: `MatrixOps.softmax` on a 64 by 64
matrix, one dispatcher belief update, one `SimplePOMDP` step, and one
`ContinuousActiveInference` step. It prints a JSON report and also writes it
to `--output` when given. CI runs it with one repetition.

## Seeded model runs

- The discrete dispatcher takes its seed from `InferenceConfig(seed=...)`; the
  root `README.md` example is the minimal runnable simulation.
- `SimplePOMDP` accepts an optional `seed` in its configuration and exposes
  `step`, `run`, and `reset`. `ContinuousActiveInference` takes a `seed`
  argument and exposes `step` and `reset`. See
  [[docs/api/api_reference|the API reference]].
- Model parameters are supplied as arrays or YAML/dict configuration. The
  repository `config.yaml` is read by `cognitive-create-node` and by the
  network visualizer in `cognitive.utils.visualization`; the model classes do
  not read it.

## Reproducible experiments

- `code/tests/` checks behavior and writes artifacts only to temporary
  directories.
- `docs/manuscript/` is the executable manuscript. `cognitive-build-manuscript
  --output build/manuscript` hydrates its numeric results and figures from the
  package API; `--no-pdf` skips LaTeX and `--no-render` skips Pandoc as well.

## Related

- [[docs/repo_docs/getting_started|Getting started]]: install and first run.
- [[docs/repo_docs/unit_testing|Unit testing]]: test layout and commands.
- [[docs/manuscript/README|Executable manuscript]]: end-to-end build.
