---
title: Adaptation
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/adaptation_mechanisms]]"
      - "[[knowledge_base/cognitive/adaptive_systems]]"
      - "[[knowledge_base/cognitive/homeostatic_control_theory]]"
---

# Adaptation

The executable examples of adaptation in the package are small:

- `ActiveInferenceDispatcher.update_precision(prediction_error)` moves the
  precision toward the reciprocal of the prediction error (a 0.9/0.1 blend,
  clipped to the range 0.1 to 1e6).
- `ContinuousActiveInference` takes a `precision_learning_rate`.
- `HomeostaticFactory.create_adaptive(config)` returns a `HomeostaticInference`
  whose `update_parameters` method changes the model during a run.

Everything else on adaptation is theory:
[[knowledge_base/cognitive/adaptation_mechanisms]],
[[knowledge_base/cognitive/adaptive_systems]], and
[[knowledge_base/cognitive/homeostatic_control_theory]]. Signatures are in
[[docs/api/api_reference|the API reference]].
