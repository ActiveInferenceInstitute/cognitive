---
title: Graph Visualization
type: concept
status: stable
updated: 2026-10-09
tags:
- graphs
- visualization
- knowledge
semantic_relations:
  - type: relates
    links:
      - "[[graph_theory]]"
      - "[[network_analysis]]"
      - "[[visualization_tools]]"
      - "[[knowledge_base/mathematics/factor_graphs]]"
      - "[[knowledge_base/cognitive/visualization_suite]]"
---


# Graph Visualization

Graph visualization renders relational structure for interpretation, validation, and discovery. In cognitive modeling, it aids inspection of generative models, factor graphs, and concept ontologies.

## Practices

- Layouts: force-directed, hierarchical, radial for multi-scale structure

- Encoding: node color/shape by type; edge width by weight/precision

- Interactivity: filtering by community, degree, or uncertainty

See also: [[graph_theory]], [[network_analysis]], [[visualization_tools]].

## Layouts and Their Trade-offs

Force-directed layouts treat edges as springs and nodes as repelling charges. The Fruchterman-Reingold algorithm, which is what NetworkX's `spring_layout` implements, is the usual choice. It reveals clusters without prior structure but is stochastic, so a fixed seed is needed for reproducible figures, and beyond a few hundred nodes it produces a dense "hairball". Hierarchical (layered) layouts suit directed acyclic structure such as a hierarchical generative model, where level matters ([[knowledge_base/cognitive/hierarchical_models]]). Radial layouts place a focal node at the centre and show distance in hops.

## What Is Drawn in Cognitive Modeling

Three kinds of graph recur. The first is the model graph: a factor graph or Bayesian network in which variables and factors are nodes and message passing runs along edges ([[knowledge_base/mathematics/factor_graphs]], [[knowledge_base/mathematics/message_passing]]). The second is the interaction graph among agents, whose edge weights may encode coupling or precision. The third is a documentation graph, where pages are nodes and links are edges. Mixing the three in one figure is best avoided, because each has different semantics for an edge.

## In This Repository

The documentation graph is implemented by `NetworkVisualizer` in `code/tools/src/utils/visualization/network_viz.py`. What it does, as implemented:

- Reads `config.yaml`, takes `paths.knowledge_base`, and scans every Markdown file below it. Each file is a node, identified by its path relative to that directory without the extension.
- Extracts `[[wiki links]]`, strips aliases (`|`) and anchors (`#`), and resolves each target case-insensitively against node id, file stem, file name or relative path. Unresolved targets and self-links are dropped, and the result is an undirected NetworkX graph.
- Lays out nodes with `spring_layout`, seeded from `visualization.seed` (default 0), so a given vault renders identically on every run.
- Renders a Plotly figure with node ids as labels and writes standalone HTML when an output path is given. The module has a command-line `main` accepting `--config`, `--output`, `--width` and `--height`; no console-script entry is registered for it.

Its limits are worth knowing. Node type is taken from the parent directory name with a trailing "s" removed, and only the types `agent`, `belief`, `goal`, `action`, `observation` and `relationship` have colours, so most pages in this vault render grey. Node size is fixed. Of the options in the `visualization` block of `config.yaml`, only `seed` is read. Filtering by community, degree or uncertainty, mentioned in the practices above, is guidance, not a feature of this class.

For plots of agent-side structure, `NetworkPlotter` in `code/tools/src/visualization/matrix_plots.py` draws message-passing dynamics between nodes and an adjacency heatmap for belief networks; see [[knowledge_base/cognitive/visualization_suite]].
