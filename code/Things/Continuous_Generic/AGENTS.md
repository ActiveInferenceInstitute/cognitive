---
title: Continuous Generic Agent
type: agents
status: stable
tags:
  - continuous
  - active_inference
  - differential_equations
semantic_relations:
  - type: implements
    links:
      - [[knowledge_base/cognitive/continuous_time_active_inference]]
---

# Continuous Generic Agent

This agent implements Active Inference in continuous time/space using generalized coordinates. See [[code/Things/Continuous_Generic/README|Knowledge Base]] for full implementation details.

## Characteristics

- **State Space**: Continuous variables (position, velocity, etc.).
- **Time**: Continuous (differential equations).
- **Inference**: Variational filtering / predictive coding.

## Related

- [[knowledge_base/agents/Continuous_Time/continuous_time_agent|Continuous-Time Agent Architecture]]
- [[knowledge_base/mathematics/generalized_coordinates|Generalized Coordinates]]
- [[knowledge_base/mathematics/differential_equations|Differential Equations]]
