---
title: Continuous architecture
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/continuous_time_models]]"
      - "[[knowledge_base/cognitive/continuous_time_active_inference]]"
      - "[[docs/api/api_reference]]"
---

# Continuous architecture

`Things.Continuous_Generic.ContinuousActiveInference` (source in
`code/Things/Continuous_Generic/continuous_generic.py`) keeps beliefs over
generalized coordinates up to `n_orders`, with observation and dynamics matrices,
precisions, and a precision learning rate. `step(observation)` returns the action
and the free energy; `reset` restores the beliefs. `ContinuousVisualizer` plots
the belief evolution and saves a GIF through Pillow.

The theory is in [[knowledge_base/mathematics/continuous_time_models]] and
[[knowledge_base/cognitive/continuous_time_active_inference]]; the full parameter
list is in [[docs/api/api_reference|the API reference]].
