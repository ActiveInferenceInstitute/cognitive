---
title: Network analysis
type: tool
status: stable
updated: 2026-10-10
created: 2026-08-02
tags:
  - network_analysis
  - graphs
  - tools
semantic_relations:
  - type: documents
    links:
      - "[[../repo_docs/obsidian_linking|Obsidian linking standards]]"
      - "[[graph_generator|Graph generation]]"
---

# Network analysis

Network analysis of the repository's knowledge graph:

- **Obsidian graph view** — visual exploration of the vault's wiki-link
  graph (`knowledge_base/`, `docs/`).
- **`cognitive.utils.visualization.network_viz`** — `NetworkVisualizer`,
  deterministic knowledge-graph rendering from the package
  (`code/tools/src/utils/visualization/network_viz.py`).
- **Link auditing** — `code/scripts/verify_links.py` (explicit wiki links)
  and `code/scripts/check_markdown_links.py` (standard Markdown links and
  anchors) provide the quantitative checks; `docs/repo_docs/linking_*.md`
  describe the methodology.

## Analysis notes

- Extensionless wiki links that match no file are intentional concept edges,
  not errors; `verify_links.py --strict-wiki-links` lists them.
- `docs/repo_docs/linking_analysis.md`, `linking_completeness.md`,
  `linking_validation.md`, and `linking_patterns.md` document the
  measurement approach.
- `knowledge_base/linking_standards.md` defines the linking conventions.
