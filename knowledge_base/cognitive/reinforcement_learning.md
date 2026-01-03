---

title: Reinforcement Learning (Comparison)

type: concept

status: draft

created: 2024-01-01

tags:

  - reinforcement_learning

  - decision_making

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

- [[../mathematics/optimal_control]]

- [[../mathematics/variational_free_energy]]

