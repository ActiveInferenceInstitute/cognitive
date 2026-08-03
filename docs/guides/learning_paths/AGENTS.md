---
title: Learning paths documentation instructions
type: agents
status: stable
created: 2025-01-01
updated: 2026-08-02
tags:
  - agents
  - learning_paths
  - education
  - cognitive_development
semantic_relations:
  - type: organizes
    links:
      - '[[README|Learning paths index]]'
      - '[[index|Learning path index]]'
      - '[[catalog_of_learning_paths|Catalog of learning paths]]'
      - '[[../../examples/README|Examples]]'
---

# Learning paths documentation instructions

The learning paths are educational, conceptual documents that sequence topics
for study. They are not executable curricula: they reference knowledge-base
concepts and repository documentation rather than defining code.

## Authoring rules

- A learning path must be a Markdown page with valid YAML frontmatter and a
  `semantic_relations` section linking the topics it covers.
- Links must resolve to real files (knowledge-base concepts or repository
  documentation). Do not invent topic pages, citations, or external resources.
- The `catalog_of_learning_paths.md` page is the index of paths; add a new
  path there and in `index.md` when one is created.
- Do not include Python code blocks that claim to implement the path;
  executable examples live in `docs/examples/` and the manuscript.

## Validation

```bash
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
```
