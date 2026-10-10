---
title: Benchmark tools
type: tool
status: stable
updated: 2026-10-10
created: 2026-08-02
tags:
  - benchmark
  - tools
  - performance
semantic_relations:
  - type: documents
    links:
      - "[[../../code/tools/src/benchmarks.py|Benchmark source]]"
---

# Benchmark tools

The repository benchmark suite is run through the installed `cognitive-benchmark`
entry point (backed by `cognitive.benchmarks`, `code/tools/src/benchmarks.py`):

```bash
cognitive-benchmark --repetitions 1
```

Options: `--repetitions N` (default 10, must be at least 1) and `--output PATH`,
which writes the JSON report to a file as well as stdout.

The runner measures four runtime micro-benchmarks and nothing else; it does not
time visualization, link checking, or any knowledge-base tooling:

| Name | Call timed |
|---|---|
| `matrix_operations` | `MatrixOps.softmax` on a 64x64 random matrix |
| `dispatcher_inference` | one `ActiveInferenceDispatcher.dispatch_belief_update` on a 2-state model |
| `simple_pomdp` | one `SimplePOMDP.step` |
| `continuous_inference` | one `ContinuousActiveInference.step` |

Each function is called once untimed as a warm-up, then `N` times under
`time.perf_counter`. The JSON report has the keys `schema_version` (1),
`repetitions`, and `benchmarks`, a list of objects with `name`, `repetitions`,
`total_seconds`, and `seconds_per_call`. Times depend on the machine, so the
output is for comparing runs on one machine. CI runs the command with
`--repetitions 1` to confirm it still executes.
