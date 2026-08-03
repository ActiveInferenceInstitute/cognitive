---
title: Benchmark tools
type: tool
status: stable
created: 2026-08-02
tags:
  - benchmark
  - tools
  - performance
semantic_relations:
  - type: documents
    links:
      - '[[../../code/tools/src/benchmarks.py|Benchmark source]]'
---

# Benchmark tools

The repository benchmark suite is run through the installed `cognitive-benchmark`
entry point (backed by `cognitive.benchmarks`, `code/tools/src/benchmarks.py`):

```bash
cognitive-benchmark --repetitions 1
```

It runs the repository's benchmarks (discrete inference, continuous agent,
visualization, and knowledge-base tooling) and emits a JSON-serializable
report. Use temporary or caller-selected output directories; benchmarks
never assume a repository output tree.
