---
title: Matrix Operations
type: module_index
status: stable
created: 2026-02-06
updated: 2026-02-08
tags:
  - matrices
  - linear_algebra
  - active_inference
  - module_index
semantic_relations:
  - type: implements
    links:
      - [[code/tools/src/models/README]]
      - [[code/tools/src/README]]
---

# Matrix Operations

Core matrix operations for Active Inference computations. This module provides matrix manipulation, normalization, information-theoretic calculations, and initialization utilities used throughout the framework.

## 📁 Module Contents

- **`matrix_ops.py`**: Core matrix operations module containing all classes

## 🔧 Classes

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
- `load_spec()` - Load matrix specification from YAML
- `load_matrix()` - Load matrix data from file
- `validate_matrix()` - Validate matrix against specification constraints

### MatrixInitializer
Factory methods for creating common matrix types:
- `random_stochastic()` - Random column-stochastic matrices
- `identity_based()` - Identity-biased stochastic matrices
- `uniform()` - Uniform distribution matrices

### MatrixVisualizer
Data preparation for matrix visualization:
- `prepare_heatmap_data()` - Format matrix for heatmap display
- `prepare_bar_data()` - Format vector for bar chart display
- `prepare_multi_heatmap_data()` - Format 3D tensor for multi-panel heatmaps

## 📚 Related Documentation

- [[code/tools/src/models/README|Models Overview]]
- [[code/tools/src/models/active_inference/README|Active Inference Models]]
- [[code/tools/src/visualization/README|Visualization Tools]]
