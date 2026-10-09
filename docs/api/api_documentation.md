---
title: API documentation overview
type: index
status: stable
updated: 2026-10-09
---

# API documentation overview

The runtime API is organized as follows:

1. `cognitive.models.active_inference`: `DiscreteGenerativeModel`,
   `InferenceConfig`, `ActiveInferenceDispatcher`, `ActiveInferenceFactory`, the
   `ActiveInferenceModel` base class with `ModelState` and its save/load
   methods, and the homeostatic family (`HomeostaticInference`,
   `HomeostaticFactory`, `HomeostaticControl`, `AdaptiveControl`).
2. `cognitive.models.matrices`: `MatrixOps`, `MatrixLoader`, `MatrixInitializer`,
   and `MatrixVisualizer` (plot data adapters).
3. `cognitive.utils`: probability helpers (`matrix_utils`), `NodeCreator`
   (`create_node`), and `NetworkVisualizer` (`cognitive.utils.visualization`).
4. `cognitive.visualization`: `MatrixPlotter`, `StateSpacePlotter`,
   `NetworkPlotter`.
5. `Things.Simple_POMDP` and `Things.Continuous_Generic`: the two installed
   agent families. Other directories under `code/Things/` are documentation
   only and are not importable.

Use [`api_reference.md`](api_reference.md) for signatures,
[`api_examples.md`](api_examples.md) for executed examples, and
[`api_versioning.md`](api_versioning.md) for version policy. Theory lives in
`knowledge_base/`; it is not a runtime import surface.
