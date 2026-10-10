---
title: Analysis package instructions
type: agents
status: active
created: 2025-01-01
updated: 2026-10-10
tags:
  - analysis
  - tools
  - agents
semantic_relations:
  - type: documents
    links:
      - "[[code/tools/src/analysis/README|Analysis package]]"
      - "[[code/tools/src/models/active_inference/AGENTS|Active Inference Models]]"
      - "[[code/tools/AGENTS|Cognitive tools instructions]]"
---

# Analysis package instructions

`code/tools/src/analysis/__init__.py` is currently an empty package (a module
docstring only). Documentation for it must not describe classes, metrics, or
subdirectories that do not exist.

To add analysis behaviour, follow the rules in
[[code/tools/AGENTS|Cognitive tools instructions]]: a real implementation, a
regression test in `code/tests/`, a documented import path, and an export from
`cognitive` if it is meant to be public. Reuse
`cognitive.utils.matrix_utils` for entropy, KL divergence, and expected free
energy rather than re-deriving them.
