---
title: Matrix plotting API
type: documentation
status: stable
---

# Matrix plotting API

`cognitive.visualization.matrix_plots.MatrixPlotter` writes matrix and vector
figures to a caller-provided output directory. It supports heatmaps,
multi-slice heatmaps, bar charts, and explicit figure saving. The constructor
requires an output directory and style mapping so generated paths are never
implicit.

For data-only adapters that do not create figures, use
`cognitive.models.matrices.matrix_ops.MatrixVisualizer`. The manuscript figure
pipeline is implemented in `code/scripts/build_manuscript.py` and is the
canonical example of deterministic figure generation.
