---
title: Analysis package
type: documentation
status: active
created: 2025-01-01
updated: 2026-10-10
tags:
  - analysis
  - tools
semantic_relations:
  - type: documents
    links:
      - "[[code/tools/README|Tools Overview]]"
      - "[[code/tools/src/models/active_inference/README|Active Inference runtime]]"
---

# Analysis package

`code/tools/src/analysis/` contains only `__init__.py`, which holds a module
docstring. It defines no classes, functions, or submodules, and nothing in
the `cognitive` public API imports it.

Quantities that analysis code would normally compute are available elsewhere:

- Entropy, KL divergence, softmax, and expected free energy:
  `cognitive.utils.matrix_utils` ([[code/tools/src/utils/README|utilities]]).
- Variational, mean-field, and sampling belief updates and policy evaluation:
  `cognitive.ActiveInferenceDispatcher`
  ([[code/tools/src/models/active_inference/README|runtime]]).
- Belief-evolution, free-energy-landscape, and policy-evaluation plots:
  `cognitive.MatrixPlotter` and the plotters in
  `cognitive.visualization.matrix_plots` ([[code/tools/src/visualization/README|plotting]]).
- Reproducible figure and benchmark generation: `code/scripts/build_manuscript.py`
  and `cognitive-benchmark` (`code/tools/src/benchmarks.py`).
