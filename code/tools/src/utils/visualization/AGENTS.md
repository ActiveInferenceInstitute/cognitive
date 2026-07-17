---
title: Knowledge network visualization instructions
type: agents
status: stable
---

# Knowledge network visualization

`NetworkVisualizer` reads a configured knowledge-base directory and builds a
deterministic NetworkX graph from Markdown links. It understands link aliases,
canonicalizes node paths, handles empty graphs, and creates output directories
only at caller-selected paths.

The matrix plotter is separate: use
`cognitive.visualization.matrix_plots.MatrixPlotter` for figures and
`cognitive.models.matrices.matrix_ops.MatrixVisualizer` for data adapters.
Tests must inspect the produced graph or file rather than only checking that a
method returned.
