---
external_mappings:
  - target: http://www.wikidata.org/entity/Q830687
    relation: exactMatch
    evidence: https://www.wikidata.org/wiki/Q830687
    rationale: >-
      The source explicitly defines reinforcement learning as optimizing expected
      cumulative reward through value functions and policy improvement. The target
      defines action-and-reward learning that maximizes cumulative reward. The page
      compares this same method with Active Inference; it does not equate their
      objectives.
    reviewed_on: "2026-10-20"
title: Reinforcement Learning (Comparison)
type: concept
status: stable
created: 2024-01-01
updated: 2026-02-07
tags:
  - reinforcement-learning
  - decision-making
  - comparison
aliases:
  - reinforcement
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[../mathematics/optimal_control]]"
      - "[[../mathematics/expected_free_energy]]"
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
