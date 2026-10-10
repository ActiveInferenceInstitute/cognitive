---
title: Matrix operations instructions
type: agents
status: stable
created: 2025-01-01
updated: 2026-10-10
tags:
  - matrices
  - operations
  - agents
semantic_relations:
  - type: documents
    links:
      - "[[code/tools/src/models/matrices/README|Matrix operations]]"
      - "[[code/tools/src/models/active_inference/AGENTS|Active Inference Models]]"
      - "[[code/tools/AGENTS|Cognitive tools instructions]]"
---

# Matrix operations instructions

`code/tools/src/models/matrices/matrix_ops.py` defines four classes, exported
from `cognitive.models.matrices`. The method list for each is in
[[code/tools/src/models/matrices/README|the module README]].

## Contracts

- All four classes use static methods; none hold state.
- Inputs must be finite and numeric. Normalization methods name their axis
  (`normalize_columns` sums over axis 0, `normalize_rows` over axis 1).
- `MatrixLoader.validate_matrix` returns `False` for a matrix that violates
  the specification and raises `ValueError` for a constraint string it does
  not support. The supported constraints are `sum(cols) == 1.0`,
  `sum(rows) == 1.0`, and `all_values >= 0`.
- `MatrixLoader.load_matrix` reads `.npy` files with `allow_pickle=False`.
- `MatrixVisualizer` only prepares data dictionaries. Figures are drawn by
  `cognitive.visualization.matrix_plots.MatrixPlotter`.

Changes need a test in `code/tests/test_matrix_ops.py`.
