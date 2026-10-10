---
title: POMDP architecture
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/pomdp_structure]]"
      - "[[knowledge_base/mathematics/active_inference_pomdp]]"
      - "[[docs/api/api_reference]]"
---

# POMDP architecture

Implemented in three layers, all documented in
[[docs/api/api_reference|the API reference]]:

- `DiscreteGenerativeModel`: the frozen `A, B, C, D, E` tuple with belief
  prediction, posterior, expected free energy, and policy evaluation.
- `ActiveInferenceDispatcher`: belief updates and policy inference under the
  `variational`, `sampling`, or `mean_field` method, for discrete policies only.
- `Things.Simple_POMDP.SimplePOMDP`: a YAML- or dict-configured agent that wraps
  these into `step`, `run`, `reset`, `save_state`, and `load_state`.

Matrix conventions and the structure of the model are in
[[knowledge_base/mathematics/pomdp_structure]].
