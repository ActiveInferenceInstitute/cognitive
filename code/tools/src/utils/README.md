---
title: Cognitive utility modules
type: documentation
status: stable
---

# Cognitive utility modules

The utility layer contains small, validated operations used by the runtime.

## Probability helpers

`cognitive.utils.matrix_utils` exports `ensure_matrix_properties`,
`compute_entropy`, `kl_divergence`, `softmax`, and
`expected_free_energy`. These functions require finite numeric input and
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
builds a deterministic NetworkX graph, and writes explicitly requested output
files. Empty graphs are valid and do not trigger layout failures.
