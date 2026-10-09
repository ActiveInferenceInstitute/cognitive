---
title: Visualization Mathematics
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - visualization
  - plotting
  - overview
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/decision_visualization]]"
      - "[[knowledge_base/mathematics/learning_visualization]]"
      - "[[knowledge_base/mathematics/exploration_visualization]]"
      - "[[knowledge_base/mathematics/matrix_operations]]"
---

# Visualization Mathematics

## Overview
Visualization in this framework is a mapping from mathematical objects to graphical marks: stochastic matrices to heatmaps, probability vectors to bars, belief trajectories to curves, and scalar functions on the belief simplex to surfaces. Choosing the mapping well is a modelling decision, since the display must respect the object's constraints (non-negative, sums to one) to avoid misleading readers.

## Definition / Formalism
Implemented in `code/tools/src/visualization/matrix_plots.py` (class `MatrixPlotter`, exported as `cognitive.visualization.MatrixPlotter`), with data adapters in `models/matrices/matrix_ops.py` (`MatrixVisualizer`).

| Object | Mathematical form | Plot |
|---|---|---|
| Likelihood / transition slice | $M\in[0,1]^{m\times n}$, columns sum to 1 | `plot_heatmap` (seaborn) |
| Transition tensor | $B\in[0,1]^{n\times n\times |\mathcal A|}$ | `plot_multi_heatmap`, one panel per action |
| Belief or preference vector | $q\in\Delta^{n-1}$ | `plot_bar` |
| Belief trajectory | $q_t$, array of shape (time, states) | `plot_belief_evolution`, one line per state |
| Free energy over beliefs | $f:\Delta^2\to\mathbb R$ | `plot_free_energy_landscape` |

`plot_free_energy_landscape` makes the simplex geometry explicit. For three states it grids $(x,y)$ over the belief range, sets the third belief to $1-x-y$, evaluates the supplied function only where all three beliefs are non-negative, and leaves other grid points as NaN. The surface is therefore defined only on the triangle $x,y\ge0,\;x+y\le1$. Its z-axis label is "Expected Free Energy", so it is intended for plotting $G$, though any scalar function of a three-vector works. The `MatrixVisualizer` methods return dictionaries (`data`, tick ranges, or `slices`) and raise `ValueError` for wrong dimensionality (2-D for heatmaps, 3-D for multi-heatmaps).

Guidelines that follow from the mathematics: use a fixed colour scale of $[0,1]$ when comparing stochastic matrices, so intensity means probability; use a perceptually uniform colormap (the style config defaults to viridis); plot trajectories on a linear probability axis, and consider a log axis only for quantities spanning orders of magnitude, such as precision; for beliefs over more than three states, plot marginals or entropy rather than a surface.

## Role in Active Inference
Plots are diagnostic: a belief trajectory that fails to concentrate, or a heatmap whose columns do not look like distributions, signals a modelling or normalisation fault. They do not enter inference.

## Related concepts
- [[matrix_operations]] — the objects being displayed
- [[decision_visualization]] — policy and action plots
- [[learning_visualization]] — parameter and belief learning curves
- [[exploration_visualization]] — information-seeking diagnostics
- [[expected_free_energy]] — the landscape plotted over beliefs
- [[probability_distributions]] — what bars and simplex surfaces represent
