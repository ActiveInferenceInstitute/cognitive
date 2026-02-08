---
title: Reinforcement Learning (Comparison)
type: concept
status: stable
created: 2024-01-01
updated: 2026-02-07
tags:
  - reinforcement-learning
  - decision-making
  - comparison
semantic_relations:
  - type: relates
    links:
      - [[active_inference]]
      - [[../mathematics/optimal_control]]
      - [[../mathematics/expected_free_energy]]
---

# Reinforcement Learning (Comparison)

Reinforcement learning (RL) optimizes expected cumulative reward via value functions and policy improvement. In contrast, [[active_inference]] selects actions by minimizing [[../mathematics/expected_free_energy]], which unifies exploration (epistemic value) and exploitation (preference satisfaction) under a single probabilistic objective.

## Mappings

- Reward ↔ log prior preference over observations
- Value function ↔ negative expected free energy
- Policy gradient ↔ free energy gradient

## See Also

- [[../mathematics/optimal_control]] — Optimal control theory
- [[../mathematics/variational_free_energy]] — Variational free energy
