---
title: Matrix Operations
type: module_index
status: stable
created: 2026-02-06
updated: 2026-10-10
tags:
  - matrices
  - linear_algebra
  - active_inference
  - module_index
semantic_relations:
  - type: implements
    links:
      - "[[code/tools/src/models/README]]"
      - "[[code/tools/src/README]]"
---

# Matrix Operations

Matrix normalization, information-theoretic calculations, specification loading, and initialization used by the discrete Active Inference runtime. The four classes are exported from `cognitive.models.matrices`.

## Module contents

- **`matrix_ops.py`**: Core matrix operations module containing all classes

## Classes

### MatrixOps
Static methods for matrix algebra and information-theoretic calculations:
- `normalize_columns()` - Normalize matrix columns to sum to 1
- `normalize_rows()` - Normalize matrix rows to sum to 1
- `ensure_probability_distribution()` - Validate and normalize probability distributions
- `compute_entropy()` - Shannon entropy of a distribution
- `compute_kl_divergence()` - KL divergence between two distributions
- `softmax()` - Softmax transformation

### MatrixLoader
Utilities for loading and validating matrices from configuration:
- `load_spec()` - Read the YAML frontmatter of a Markdown specification file
- `load_matrix()` - Load a NumPy `.npy` array (no pickle)
- `validate_matrix()` - Check shape and constraints (`sum(cols) == 1.0`, `sum(rows) == 1.0`, `all_values >= 0`)

### MatrixInitializer
Factory methods for creating common matrix types:
- `random_stochastic()` - Random column-stochastic matrices (optional `rng`)
- `identity_based()` - Square column-stochastic matrix with `strength` on the diagonal
- `uniform()` - Constant array whose entries sum to 1 over the whole shape

### MatrixVisualizer
Data preparation for matrix visualization (figures are drawn by `MatrixPlotter`):
- `prepare_heatmap_data()` - Format matrix for heatmap display
- `prepare_bar_data()` - Format vector for bar chart display
- `prepare_multi_heatmap_data()` - Format 3D tensor for multi-panel heatmaps

## Related documentation

- [[code/tools/src/models/README|Models Overview]]
- [[code/tools/src/models/active_inference/README|Active Inference Models]]
- [[code/tools/src/visualization/README|Matrix plotting]]
- [[code/tools/src/models/matrices/AGENTS|Instructions]]
