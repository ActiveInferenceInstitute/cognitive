---
description: "How expected free energy per action is computed and used to update the policy prior E through a softmax with precision, including implementation, properties and related concepts."
type: mathematical_concept
id: expected_free_energy_update_001
created: 2024-02-05
modified: 2024-02-05
tags:
- mathematics
- active-inference
- policy-selection
- free-energy
aliases:
- EFE-update
- policy-prior-update
---


# Expected Free Energy Update

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Mathematical Definition

The Expected Free Energy (G) for each action a is:

$G(a) = \underbrace{\mathbb{E}_{Q(o,s|a)}[\ln Q(s|a) - \ln P(s|a)]}_{\text{epistemic value}} + \underbrace{\mathbb{E}_{Q(o|a)}[-\ln P(o)]}_{\text{pragmatic value}}$

where:

- Q(s|a) is predicted state distribution under action a

- P(s|a) is prior state distribution

- Q(o|a) is predicted observation distribution

- P(o) is preferred observation distribution (from C matrix)

## Policy Prior Update

The policy prior E is updated using:

$E_{t+1} = (1-\alpha)E_t + \alpha\sigma(-\gamma G)$

where:

- $E_t$ is current policy prior

- $\alpha$ is learning rate (0 for static prior)

- $\gamma$ is precision parameter

- $\sigma$ is softmax function

- G is vector of Expected Free Energies

## Implementation

```python
import numpy as np
from cognitive.utils.matrix_utils import expected_free_energy, softmax

def update_policy_prior(A, B, C, E, beliefs, alpha=0.1, gamma=1.0):
    """E <- (1 - alpha) * E + alpha * softmax(-gamma * G).
    A: p(o|s) [n_obs, n_states]; B: p(s'|s,a) [n_states, n_states, n_actions];
    C: log preferences [n_obs]; E: policy prior [n_actions]; beliefs: q(s)."""
    G = np.array([expected_free_energy(A, B, C, beliefs, a)
                  for a in range(B.shape[2])])
    return (1 - alpha) * E + alpha * softmax(-gamma * G)
```

## Usage

Iterating the update with a fixed belief state drives the policy prior to the fixed point softmax(-γG):

```python
A = np.array([[0.9, 0.2], [0.1, 0.8]])                                   # illustrative
B = np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2)    # two actions
C = np.array([2.0, 0.0])
beliefs = np.array([0.8, 0.2])

E = np.ones(2) / 2                       # uniform initial policy prior
for _ in range(200):
    E = update_policy_prior(A, B, C, E, beliefs, alpha=0.1, gamma=1.0)

G = np.array([expected_free_energy(A, B, C, beliefs, a) for a in range(2)])
print(np.round(E, 3), np.allclose(E, softmax(-G)))   # fixed point is softmax(-gamma G)
```

## Properties

### Mathematical Properties

- probability conservation - Output is valid probability distribution

- policy convergence - Converges to optimal policy under right conditions

- [[learning_dynamics]] - Controlled by learning rate and precision

### Computational Properties

- [[numerical_stability]] - Uses log space for preferences

- [[computational_efficiency]] - Vectorized operations

- memory usage - O(n_actions) space complexity

## Related Concepts

- [[free_energy_principle]]

- [[active_inference]]

- [[policy_selection]]

- [[belief_updating]]

## Advanced Theoretical Framework

### Convergence Analysis and Stability Theory

**Theorem** (EFE Update Convergence): Under mild regularity conditions, the policy prior update converges to the Gibbs distribution:

$$\lim_{t \to \infty} E_t = \frac{\exp(-\gamma G^*)}{\sum_a \exp(-\gamma G^*_a)}$$

where $G^*$ is the fixed-point EFE vector.

**Proof**: Using Lyapunov function analysis and the contraction mapping principle for the update operator.

### Information-Theoretic Learning Bounds

**Theorem** (PAC-Bayes Bound for EFE Learning): With probability at least $1-\delta$, the generalization error of the learned policy satisfies:

$$\mathbb{E}_{D}[G(\pi)] \leq \hat{\mathbb{E}}[G(\pi)] + \sqrt{\frac{D_{KL}[Q(\pi) \| P(\pi)] + \ln(1/\delta)}{2n}}$$

where $D$ is the data distribution, $Q(\pi)$ is the learned policy distribution, and $P(\pi)$ is the prior.

### Stochastic Approximation Theory

**Theorem** (Robbins-Monro for EFE Updates): The EFE update sequence $\{E_t\}$ with learning rate $\alpha_t$ converges almost surely to the optimal policy if:

1. $\sum_{t=1}^{\infty} \alpha_t = \infty$

1. $\sum_{t=1}^{\infty} \alpha_t^2 < \infty$

1. The EFE function satisfies Lipschitz conditions

This provides theoretical foundation for adaptive learning rate schedules in EFE updates.

## References

- [[friston_2017]] - Mathematical foundations

- [[da_costa_2020]] - Active Inference implementation

- [[parr_2019]] - Policy learning

- robbins monro 1951 - Stochastic approximation theory

- mcallester 1999 - PAC-Bayes bounds

- borkar 2008 - Stochastic approximation algorithms

