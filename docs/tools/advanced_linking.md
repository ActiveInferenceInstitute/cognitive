---
title: Advanced linking
type: tool
status: stable
created: 2026-08-02
tags:
  - linking
  - wiki_links
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../repo_docs/obsidian_linking|Obsidian linking standards]]'
      - '[[../repo_docs/linking_validation|Link validation]]'
---

# Advanced linking

Advanced linking conventions used in this repository:

- **Path forms**: wiki links may be vault-root relative
  (`[[knowledge_base/cognitive/active_inference|Active Inference]]`) or
  folder relative (`[[active_inference]]`). The link validator resolves
  both; explicit file-like targets (containing a slash or a file suffix)
  must exist.
- **Aliased links**: `[[target|display text]]` for readable labels.
- **Concept links**: extensionless targets without a slash are concept
  edges by convention; they may resolve to a file (by stem) or remain
  unresolved graph edges.
- **Redirects**: duplicate or renamed pages use `redirect` frontmatter
  (for example `knowledge_base/citations/parr_2022.md` ->
  `[[parr_pezzulo_friston_2022]]`).
- **`semantic_relations`**: frontmatter declares typed relationships
  (`documents`, `organizes`, `relates`, ...) that Obsidian's graph view
  renders.

Authoring rules: `docs/repo_docs/obsidian_linking.md` and
`knowledge_base/linking_standards.md`. Validation:
`python code/scripts/verify_links.py . --json`.
