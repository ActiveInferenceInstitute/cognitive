---
title: Matrix plotting API
type: documentation
status: stable
updated: 2026-10-10
---

# Matrix plotting API

`cognitive.visualization.matrix_plots.MatrixPlotter` writes matrix and vector
figures to a caller-provided output directory. Its methods are
`plot_heatmap`, `plot_multi_heatmap`, `plot_bar`, `plot_belief_evolution`,
`plot_free_energy_landscape`, and `save_figure`. The constructor
`MatrixPlotter(output_dir, style_config)` requires an output directory and a
style mapping (`theme`, `figure_size`, `dpi`) so generated paths are never
implicit. The same module also defines `StateSpacePlotter` and `NetworkPlotter`
(import them from `cognitive.visualization.matrix_plots`; only `MatrixPlotter`
is exported from `cognitive`).

For data-only adapters that do not create figures, use
`cognitive.models.matrices.matrix_ops.MatrixVisualizer`. The manuscript figure
pipeline is implemented in `code/scripts/build_manuscript.py` and is the
canonical example of deterministic figure generation.
