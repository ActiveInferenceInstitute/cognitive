---
title: Matrix Specifications
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [active_inference, POMDP, matrices, generative_model]
semantic_relations:
  - type: specifies
    links: [[knowledge_base/mathematics/active_inference_pomdp]]
  - type: relates
    links: [[[observation_model]], [[transition_model]], [[system_definition]], [[inference_configuration]]]
---

# Matrix Specifications

The generative model in discrete Active Inference is defined by five matrices (A, B, C, D, E) that fully specify the agent's beliefs about how observations are generated, how states evolve, and what outcomes are preferred.

## Matrix Definitions

### A Matrix — Observation Model

Maps hidden states to observations: $p(o_t | s_t)$

```math
A_{ij} = p(o = i | s = j), \\quad \\sum_i A_{ij} = 1 \\; \\forall j
```

Properties: Column-stochastic, entries in [0,1].

### B Matrix — Transition Model

Maps state transitions under actions: $p(s_{t+1} | s_t, a_t)$

```math
B^{(a)}_{ij} = p(s_{t+1} = i | s_t = j, a_t = a), \\quad \\sum_i B^{(a)}_{ij} = 1
```

Properties: Column-stochastic per action, one matrix per action.

### C Matrix — Preference Model

Log prior preferences over observations: $\\ln p(o_t)$

```math
C_i = \\ln p(o = i), \\quad \\text{higher values = more preferred}
```

### D Vector — Initial State Prior

Prior beliefs about the initial state: $p(s_0)$

```math
D_i = p(s_0 = i), \\quad \\sum_i D_i = 1
```

### E Vector — Policy Prior (Habits)

Prior over policies before evidence: $p(\\pi)$

```math
E_k = p(\\pi = k), \\quad \\sum_k E_k = 1
```

## Example Specification

```python
import numpy as np

def create_t_maze_model():
    """Create a T-maze generative model."""
    num_states = 4      # left-arm, right-arm, center, cue
    num_obs = 4          # reward-left, reward-right, no-reward, cue
    num_actions = 3      # go-left, go-right, stay

    A = np.zeros((num_obs, num_states))
    A[0, 0] = 0.9; A[2, 0] = 0.1  # left arm mostly reward-left
    A[1, 1] = 0.9; A[2, 1] = 0.1  # right arm mostly reward-right
    A[2, 2] = 1.0                   # center: no reward
    A[3, 3] = 1.0                   # cue location: cue

    B = np.zeros((num_actions, num_states, num_states))
    B[0] = np.eye(num_states)[[0, 0, 0, 3]]  # go-left
    B[1] = np.eye(num_states)[[1, 1, 1, 3]]  # go-right
    B[2] = np.eye(num_states)                  # stay

    C = np.array([2.0, 2.0, -2.0, 0.0])  # prefer rewards

    D = np.array([0.0, 0.0, 1.0, 0.0])   # start at center

    return A, B, C, D
```

## Validation Requirements

| Matrix | Shape | Constraint | Validation |
| --- | --- | --- | --- |
| A | (obs, states) | Column-stochastic | `assert np.allclose(A.sum(0), 1)` |
| B | (actions, states, states) | Column-stochastic per action | `assert np.allclose(B.sum(1), 1)` |
| C | (obs,) | Real-valued log prefs | No strict constraint |
| D | (states,) | Sums to 1 | `assert np.isclose(D.sum(), 1)` |
| E | (policies,) | Sums to 1 | `assert np.isclose(E.sum(), 1)` |

## Related Topics

- [[observation_model]] — Detailed A matrix specification
- [[transition_model]] — Detailed B matrix specification
- [[system_definition]] — Full system specification
- [[inference_configuration]] — Inference parameters
- [[knowledge_base/mathematics/active_inference_pomdp]] — POMDP formulation

## References

- Da Costa, L., et al. (2020). Active inference on discrete state-spaces.
- Friston, K., et al. (2017). Active inference: A process theory.
