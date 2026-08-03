---
title: Graph generation
type: tool
status: stable
created: 2026-08-02
tags:
  - graphs
  - visualization
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../../code/tools/src/utils/visualization/network_viz.py|Network visualizer]]'
---

# Graph generation

The package's deterministic knowledge-graph visualization lives in
`cognitive.utils.visualization.network_viz` (source:
`code/tools/src/utils/visualization/network_viz.py`). It uses deterministic
layouts and handles empty graphs.

Knowledge-base graph views (Obsidian) are produced by Obsidian itself from
the `[[wiki links]]` in the Markdown files; see
`docs/repo_docs/obsidian_linking.md` for the linking conventions.
