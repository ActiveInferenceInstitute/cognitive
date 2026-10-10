---
title: Implementation tools
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/api_reference]]"
      - "[[docs/implementation/README]]"
---

# Implementation tools

Building blocks for a model, all listed in [[docs/api/api_reference|the API reference]]:
`DiscreteGenerativeModel`, `ActiveInferenceDispatcher` with `ActiveInferenceFactory`,
`ActiveInferenceModel` as the abstract base for stateful models,
`cognitive.models.matrices.matrix_ops` (`MatrixOps`, `MatrixLoader`,
`MatrixInitializer`), `HomeostaticFactory`, and the agent families in `Things`.
Guidance on structuring an implementation is in
[[docs/implementation/README|the implementation index]].
