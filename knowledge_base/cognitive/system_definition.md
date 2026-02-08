---
title: System Definition
type: concept
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - system-definition
  - POMDP
  - generative-model
  - active-inference
  - specification
semantic_relations:
  - type: relates
    links:
      - [[generative_model]]
      - [[matrix_specifications]]
      - [[observation_model]]
      - [[transition_model]]
---

# System Definition

## Overview

A system definition in Active Inference specifies the complete Partially Observable Markov Decision Process (POMDP) generative model: the state space, observation space, action space, and all associated probability distributions. This is the foundational step before inference, learning, or control can proceed.

## POMDP Specification

### Tuple Notation

```math
\mathcal{M} = \langle \mathcal{S}, \mathcal{O}, \mathcal{U}, A, B, C, D, E, T \rangle
```

| Component | Symbol | Description |
| --- | --- | --- |
| State space | $\mathcal{S}$ | Set of hidden states (discrete or continuous) |
| Observation space | $\mathcal{O}$ | Set of possible observations |
| Action space | $\mathcal{U}$ | Set of available actions |
| Observation model | $A$ | $p(o \| s)$ likelihood mapping |
| Transition model | $B$ | $p(s' \| s, u)$ state dynamics |
| Preference prior | $C$ | $\ln p(o)$ preferred observations |
| State prior | $D$ | $p(s_0)$ initial state beliefs |
| Habit prior | $E$ | $p(\pi)$ default policy |
| Time horizon | $T$ | Planning depth |

## Implementation

```python
class SystemDefinition:
    def __init__(self, n_states, n_obs, n_actions, n_factors=1, T=5):
        self.n_states = n_states if isinstance(n_states, list) else [n_states]
        self.n_obs = n_obs if isinstance(n_obs, list) else [n_obs]
        self.n_actions = n_actions if isinstance(n_actions, list) else [n_actions]
        self.n_factors = n_factors
        self.T = T
        
        # Initialize model components
        self.A = [np.eye(o, s) for o, s in zip(self.n_obs, self.n_states)]
        self.B = [np.eye(s)[..., np.newaxis].repeat(a, axis=-1)
                  for s, a in zip(self.n_states, self.n_actions)]
        self.C = [np.zeros(o) for o in self.n_obs]
        self.D = [np.ones(s) / s for s in self.n_states]
        self.E = np.ones(n_actions[0]) / n_actions[0] if isinstance(n_actions, list) else np.ones(n_actions) / n_actions

    def validate(self):
        for i, a in enumerate(self.A):
            assert np.allclose(a.sum(axis=0), 1.0), f"A[{i}] columns must sum to 1"
        for i, b in enumerate(self.B):
            for a in range(b.shape[-1]):
                assert np.allclose(b[:,:,a].sum(axis=0), 1.0), f"B[{i}][action={a}] columns must sum to 1"
        return True
```

### Example: T-Maze System Definition

```python
tmaze = SystemDefinition(
    n_states=[4, 2],        # [location(4), context(2)]
    n_obs=[4, 2, 2],        # [location(4), reward(2), cue(2)]
    n_actions=[4],           # [up, down, left, right]
    T=3
)
```

## Design Considerations

1. **State factorization**: Separate independent factors to reduce computational cost
2. **Observation modalities**: Multi-modal observations in separate factors
3. **Action controllability**: Which state factors are action-dependent
4. **Time horizon**: Longer $T$ → more foresighted but costlier planning
5. **Prior specification**: Informative vs. flat priors for D and C

## Related Topics

- [[generative_model]] — Generative model theory
- [[matrix_specifications]] — Matrix specifications
- [[observation_model]] — Observation model (A matrix)
- [[transition_model]] — Transition model (B matrix)
- [[state_space_abstraction]] — State space design
