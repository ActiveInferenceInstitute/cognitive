---
title: Hierarchical extensions
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/hierarchical_models]]"
      - "[[knowledge_base/cognitive/hierarchical_inference]]"
---

# Hierarchical extensions

The package has no hierarchical agent. `InferenceConfig` accepts only
`policy_type="discrete"`; any other value, including `"hierarchical"`, raises
`ValueError`.

The theory is documented in [[knowledge_base/cognitive/hierarchical_models]] and
[[knowledge_base/cognitive/hierarchical_inference]]. A hierarchical agent built on
this package would stack `DiscreteGenerativeModel` instances itself; no such
construction ships here.
