---
title: Knowledge network visualization
type: module_index
status: stable
created: 2026-02-06
updated: 2026-10-10
tags:
  - module_index
  - documentation
---

# Knowledge network visualization

`cognitive.utils.visualization.network_viz.NetworkVisualizer` is the only
module in this package (exported from its `__init__.py`).

- The constructor takes a YAML configuration path (default `config.yaml`). The
  configuration must define `paths.knowledge_base`; a relative path resolves
  against the configuration file. `visualization.seed` (default 0) seeds the
  layout.
- `build_network()` returns a NetworkX graph with one node per Markdown file
  (canonical relative path without suffix) and an edge for each `[[wiki link]]`
  that resolves to another file.
- `visualize(output_path=None, width=1200, height=800)` returns a Plotly
  figure and writes HTML only when `output_path` is given.

Related: [[code/tools/src/utils/README|Utility modules]],
[[code/tools/src/utils/visualization/AGENTS|Instructions]].
