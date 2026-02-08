---
title: State Estimation
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [state_estimation, filtering, inference, perception]
semantic_relations:
  - type: implements
    links: [[knowledge_base/mathematics/variational_inference]]
  - type: relates
    links: [[[hidden_states]], [[belief_evolution]], [[knowledge_base/cognitive/perception_processing]], [[knowledge_base/cognitive/belief_updating]]]
---

# State Estimation

Methods for estimating hidden states from observations in Active Inference, corresponding to the perceptual inference component of the perception-action loop.

## Discrete State Estimation

### Variational Message Passing

```math
\begin{aligned}
& q(s_t) \propto \sigma\left(\ln A^T o_t + \ln B(a_{t-1})^T q(s_{t-1}) + \ln B(a_t) q(s_{t+1})\right) \\
& \text{Iterative update until convergence:} \quad ||q^{(n+1)} - q^{(n)}|| < \epsilon
\end{aligned}
```

### Forward-Backward Algorithm

```python
def forward_backward(A, B, observations, actions, D):
    """Run forward-backward algorithm for state estimation."""
    T = len(observations)
    n_states = A.shape[1]
    
    # Forward pass
    alpha = np.zeros((T, n_states))
    alpha[0] = A[observations[0]] * D
    alpha[0] /= alpha[0].sum()
    for t in range(1, T):
        alpha[t] = A[observations[t]] * (B[actions[t-1]] @ alpha[t-1])
        alpha[t] /= alpha[t].sum()
    
    # Backward pass
    beta = np.zeros((T, n_states))
    beta[-1] = np.ones(n_states)
    for t in range(T-2, -1, -1):
        beta[t] = B[actions[t]].T @ (A[observations[t+1]] * beta[t+1])
        beta[t] /= beta[t].sum()
    
    # Smoothed beliefs
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma
```

## Continuous State Estimation

### Kalman Filtering (Linear Gaussian)

```math
\begin{aligned}
& \text{Predict:} \quad \hat{x}_{t|t-1} = A\hat{x}_{t-1|t-1} + Bu_t \\
& \text{Update:} \quad \hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t(z_t - H\hat{x}_{t|t-1}) \\
& \text{Kalman gain:} \quad K_t = P_{t|t-1}H^T(HP_{t|t-1}H^T + R)^{-1}
\end{aligned}
```

### Generalized Filtering (Active Inference)

```math
\dot{\tilde{\mu}} = D\tilde{\mu} - \kappa \frac{\partial F}{\partial \tilde{\mu}}
```

where $\tilde{\mu}$ includes generalized coordinates (position, velocity, acceleration, ...).

## Comparison of Methods

| Method | State Type | Nonlinear? | Computational Cost |
| --- | --- | --- | --- |
| Variational MP | Discrete | N/A | $O(|S|^2 \cdot T)$ |
| Forward-backward | Discrete | N/A | $O(|S|^2 \cdot T)$ |
| Kalman filter | Continuous | No | $O(d^3)$ |
| Extended Kalman | Continuous | Yes (local) | $O(d^3)$ |
| Particle filter | Any | Yes | $O(N \cdot d)$ |
| Generalized filter | Continuous | Yes | $O(d^2 \cdot k)$ |

## Related Topics

- [[hidden_states]] — Hidden state theory
- [[belief_evolution]] — Belief trajectory analysis
- [[knowledge_base/cognitive/perception_processing]] — Perceptual inference
- [[knowledge_base/cognitive/belief_updating]] — Belief updating mechanisms
- [[knowledge_base/mathematics/variational_inference]] — Variational inference
