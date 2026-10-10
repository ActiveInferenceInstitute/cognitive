---
title: Continuous Generic agent
type: agents
status: stable
updated: 2026-10-10
tags:
  - continuous
  - active_inference
semantic_relations:
  - type: documented_by
    links:
      - "[[code/Things/Continuous_Generic/README]]"
      - "[[code/Things/Continuous_Generic/Continuous_Generic_README]]"
  - type: related_to
    links:
      - "[[knowledge_base/cognitive/continuous_time_active_inference]]"
      - "[[knowledge_base/mathematics/generalized_coordinates|Generalized coordinates]]"
---

# Continuous Generic agent

Executable. `from Things.Continuous_Generic import ContinuousActiveInference,
ContinuousState, ContinuousVisualizer`; source in
`code/Things/Continuous_Generic/`.

- **State**: a `(n_states, n_orders)` matrix of belief means in generalized
  coordinates, with matching precisions, held in `ContinuousState`.
- **Time**: discrete steps of size `dt`; the dynamics matrix gives the expected
  derivative at every order. This is a finite generalized-coordinate
  approximation, not an ODE solver.
- **Inference**: each `step` is one gradient-descent update (step size `alpha`)
  on the free energy that the agent reports, with optional learning of the
  observation precision, clipped to `[precision_min, precision_max]`.

Concept background: [[knowledge_base/agents/Continuous_Time/continuous_time_agent|Continuous-time agent architecture]]
and [[knowledge_base/mathematics/differential_equations|Differential equations]].
