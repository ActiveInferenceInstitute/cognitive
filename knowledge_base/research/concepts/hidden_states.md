---
title: Hidden States
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [hidden_states, POMDP, inference, generative_model]
semantic_relations:
  - type: component_of
    links: [[knowledge_base/mathematics/active_inference_pomdp]]
  - type: relates
    links: [[[state_estimation]], [[state_spaces]], [[knowledge_base/cognitive/perception_processing]], [[knowledge_base/mathematics/markov_blanket]]]
---

# Hidden States

Hidden states are the unobserved variables in a generative model that the agent infers from observations. In Active Inference, the agent maintains beliefs over hidden states and updates them through variational inference.

## Formal Definition

```math
\begin{aligned}
& s_t \in \mathcal{S} \quad \text{(hidden state space)} \\
& o_t = g(s_t) + \omega_t \quad \text{(observation function)} \\
& s_{t+1} = f(s_t, a_t) + \zeta_t \quad \text{(transition function)}
\end{aligned}
```

The agent's task is to infer $q(s_t) \approx p(s_t | o_{1:t})$.

## State Inference

### Discrete States

```math
q(s_t = j) = \frac{A_{o_t,j} \sum_i B^{(a_{t-1})}_{j,i} q(s_{t-1} = i)}{\sum_k A_{o_t,k} \sum_i B^{(a_{t-1})}_{k,i} q(s_{t-1} = i)}
```

### Continuous States

```math
\dot{\mu} = D\mu - \kappa \nabla_\mu F = D\mu + \kappa(\Pi_o \varepsilon_o + \Pi_s \varepsilon_s)
```

## Types of Hidden States

| Type | Description | Example |
| --- | --- | --- |
| External | Physical environment states | Object position |
| Internal | Agent's physiological states | Energy level |
| Social | Other agents' mental states | Intentions |
| Contextual | Task or environmental context | Rule set |

## Factored Hidden States

```python
class FactoredHiddenStates:
    def __init__(self, factors):
        self.factors = factors  # {'location': 4, 'context': 2, 'reward': 3}
        self.beliefs = {f: np.ones(d)/d for f, d in factors.items()}

    def update(self, observation, A_matrices, B_matrices, action):
        for factor in self.factors:
            likelihood = A_matrices[factor][:, :].T @ observation
            transition = B_matrices[factor][action] @ self.beliefs[factor]
            self.beliefs[factor] = likelihood * transition
            self.beliefs[factor] /= self.beliefs[factor].sum()
```

## Related Topics

- [[state_estimation]] — Methods for estimating states
- [[state_spaces]] — State space structures
- [[knowledge_base/cognitive/perception_processing]] — Perceptual inference
- [[knowledge_base/mathematics/active_inference_pomdp]] — POMDP formulation
- [[knowledge_base/mathematics/markov_blanket]] — Markov blanket boundaries
