---
title: Transition Model
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [active-inference, transition, dynamics, generative-model, POMDP]
semantic_relations:
  - type: component_of
    links: [[matrix_specifications]]
  - type: relates
    links:
      - "[[observation_model]]"
      - "[[system_definition]]"
      - "[[../mathematics/active_inference_pomdp]]"
      - "[[learning_mechanisms]]"
---

# Transition Model

The transition model (B matrices) in Active Inference defines how hidden states evolve over time as a function of the agent's actions. Each action has its own transition matrix encoding the agent's beliefs about state dynamics.

## Mathematical Definition

```math
\begin{aligned}
& B^{(a)}_{ij} = p(s_{t+1} = i | s_t = j, a_t = a) \\
& \text{Constraint:} \quad \sum_i B^{(a)}_{ij} = 1 \quad \forall j, a \\
& \text{Dimensionality:} \quad B \in \mathbb{R}^{|A| \times |S| \times |S|}
\end{aligned}
```

### Controllability Analysis

A system is controllable if any target state is reachable from any initial state through some sequence of actions:

```math
\text{Controllable iff } \text{rank}\left(\left[B^{(a_1)}, B^{(a_2)}, \ldots, \prod_{k} B^{(a_k)}, \ldots\right]\right) = |S|
```

## Types of Transitions

### Deterministic Transitions

Permutation matrices where each action maps each state to exactly one successor:

```math
B^{(a)}_{ij} \in \{0, 1\}, \quad \sum_i B^{(a)}_{ij} = 1
```

### Stochastic Transitions

Soft transitions allowing probabilistic state evolution:

```python
def create_noisy_transition(deterministic_B, noise_level=0.1):
    """Add noise to a deterministic transition model."""
    n_states = deterministic_B.shape[0]
    B_noisy = deterministic_B * (1 - noise_level) + noise_level / n_states
    B_noisy /= B_noisy.sum(axis=0, keepdims=True)
    return B_noisy
```

### Context-Dependent Transitions

Factored B matrices where transitions depend on context:

```math
B^{(a, c)}_{ij} = p(s_{t+1} = i | s_t = j, a_t = a, c_t = c)
```

## Learning Transitions

B matrices are learned through Dirichlet concentration parameter updates:

```math
b^{(a)}_{ij,\text{new}} = b^{(a)}_{ij,\text{old}} + s_{t+1,i} \cdot s_{t,j} \cdot \mathbb{1}[a_t = a]
```

### Implementation

```python
class LearnableTransitionModel:
    def __init__(self, n_states, n_actions, prior_concentration=1.0):
        self.b = np.ones((n_actions, n_states, n_states)) * prior_concentration
        self.B = self.b / self.b.sum(axis=1, keepdims=True)

    def update(self, state_t, state_t1, action):
        self.b[action] += np.outer(state_t1, state_t)
        self.B[action] = self.b[action] / self.b[action].sum(axis=0, keepdims=True)

    def predict(self, state, action):
        return self.B[action] @ state
```

## Analysis Tools

| Property | Formula | Interpretation |
| --- | --- | --- |
| Entropy | $H_a = -\sum_{i,j} p(j) B^{(a)}_{ij} \ln B^{(a)}_{ij}$ | Transition uncertainty per action |
| Mixing time | $t_{mix} = \min\{t: ||B^t - \pi||_{TV} < \epsilon\}$ | Speed to stationary distribution |
| Spectral gap | $1 - |\lambda_2|$ | Convergence speed |
| Reversibility | $\pi_i B_{ij} = \pi_j B_{ji}$ | Detailed balance |

## Related Topics

- [[matrix_specifications]] — Full matrix specification
- [[observation_model]] — A matrix specification
- [[system_definition]] — System specification
- [[learning_mechanisms]] — Learning B matrices
- [[../mathematics/active_inference_pomdp]] — POMDP framework
