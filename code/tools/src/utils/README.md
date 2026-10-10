---
title: Cognitive utility modules
type: documentation
status: stable
updated: 2026-10-10
---

# Cognitive utility modules

The utility layer contains small, validated operations used by the runtime.

## Probability helpers

`cognitive.utils.matrix_utils` exports `ensure_matrix_properties`,
`compute_entropy`, `kl_divergence`, `softmax`, `expected_free_energy`, and
`normalize_mass`. The first five are also re-exported from `cognitive`. These functions require finite numeric input and
make the normalization axis explicit. Unknown matrix constraints and invalid
probability vectors are rejected.

## Node creation

`cognitive.utils.create_node.NodeCreator` loads a YAML configuration, resolves
paths relative to that file, selects a template by node type, validates a safe
single filename, and renders a complete Markdown node. Use the installed
command:

```bash
cognitive-create-node --help
```

## Network visualization

`cognitive.utils.visualization.network_viz.NetworkVisualizer` reads a
configured knowledge-base directory, resolves aliases in Obsidian links,
builds a deterministic NetworkX graph (seeded spring layout, seed from
`visualization.seed` in the configuration), and renders it as a Plotly figure.
`visualize(output_path)` writes an HTML file only when a path is given. Empty
graphs are valid and do not trigger layout failures. The module also runs as
`python -m cognitive.utils.visualization.network_viz --config config.yaml
--output graph.html`.
