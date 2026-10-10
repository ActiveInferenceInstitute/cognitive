---
title: Graph generation
type: tool
status: stable
updated: 2026-10-10
created: 2026-08-02
tags:
  - graphs
  - visualization
  - tools
semantic_relations:
  - type: documents
    links:
      - "[[../../code/tools/src/utils/visualization/network_viz.py|Network visualizer]]"
---

# Graph generation

There is no `graph_generator` module. The knowledge-graph renderer is
`NetworkVisualizer` in `cognitive.utils.visualization.network_viz` (source:
`code/tools/src/utils/visualization/network_viz.py`). It reads
`paths.knowledge_base` from `config.yaml`, builds a `networkx` graph from the
`[[wiki links]]` in the Markdown files, lays it out with a seeded spring
layout (`visualization.seed`), and renders a Plotly figure:

```bash
python -m cognitive.utils.visualization.network_viz --config config.yaml --output kb_graph.html
```

Options: `--config/-c`, `--output/-o`, `--width/-w`, `--height`. Write the
output to a temporary or caller-selected path. For graphs of about 500 nodes or
more, networkx's spring layout imports `scipy`, which is not a declared
dependency of this package; the whole-vault graph therefore fails in an
environment without it.

Obsidian's own graph view is separate and is driven by the same wiki links; see
`docs/repo_docs/obsidian_linking.md` for the linking conventions.
