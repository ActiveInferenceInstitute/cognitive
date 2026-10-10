---
title: Analysis tools
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/api_reference]]"
      - "[[docs/tools/benchmark_tools]]"
---

# Analysis tools

The `cognitive.analysis` package exists but contains only a module docstring: it
exports no functions. Analysis of results is done with the helpers in
`cognitive.utils.matrix_utils` (`compute_entropy`, `kl_divergence`, `softmax`,
`expected_free_energy`), with the plotting classes in
`cognitive.visualization.matrix_plots`, and with `cognitive-benchmark` for timings
([[docs/tools/benchmark_tools|benchmark tools]]). Signatures are in
[[docs/api/api_reference|the API reference]]. There is no statistics or
experiment-tracking module.
