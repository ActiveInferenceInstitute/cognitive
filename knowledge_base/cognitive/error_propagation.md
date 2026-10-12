---
title: Error Propagation Implementation
type: concept
status: stable
created: 2024-02-12
tags:
- implementation
- predictive-processing
- error-handling
semantic_relations:
  - type: implements
    links:
      - "[[predictive_processing]]"
  - type: relates
    links:
      - "[[predictive_network]]"
      - "[[precision_mechanisms]]"
---


# Error Propagation Implementation

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Overview

This guide details the implementation of error propagation mechanisms in predictive processing networks, focusing on both forward and backward message passing.

## Core Mechanisms

### Error Types

```python
import numpy as np

W = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 0.5]])   # generative weights: layer-1 causes -> layer-0 prediction
y = np.array([1.0, 0.8, 0.2])                        # sensory input (illustrative)
prior = np.zeros(2)                                  # top-down expectation for layer 1
pi_y, pi_x = 4.0, 1.0                                # precisions
mu = np.zeros(2)                                     # layer-1 belief

eps_y = y - W @ mu                                   # sensory (bottom-up) prediction error
eps_x = mu - prior                                   # state (top-down) prediction error
F = 0.5 * (pi_y * eps_y @ eps_y + pi_x * eps_x @ eps_x)
print(np.round(eps_y, 2), np.round(eps_x, 2), round(F, 3))
```

### Message Passing

```python
history = []
for _ in range(200):
    eps_y = y - W @ mu
    eps_x = mu - prior
    mu = mu + 0.05 * (pi_y * W.T @ eps_y - pi_x * eps_x)   # errors flow up, expectations update locally
    history.append(0.5 * (pi_y * eps_y @ eps_y + pi_x * eps_x @ eps_x))
print(history[0] > history[-1], np.round(mu, 2))           # free energy decreases
```

## Error Integration

### Update Rules

```python
eta = 0.05
for _ in range(100):
    eps_y = y - W @ mu
    mu = mu + 0.05 * (pi_y * W.T @ eps_y - pi_x * (mu - prior))
    W = W + eta * pi_y * np.outer(eps_y, mu)               # local Hebbian-like weight update
print(round(float(np.sum((y - W @ mu) ** 2)), 4))          # sensory error after learning
```

## Best Practices

### Error Handling

1. Validate error magnitudes

1. Check precision values

1. Monitor gradients

1. Handle edge cases

### Optimization

1. Batch processing

1. Memory management

1. Computational efficiency

1. Numerical stability

### Debugging

1. Visualize error flow

1. Track convergence

1. Monitor updates

1. Validate predictions

## Common Issues

### Stability

1. Gradient explosion

1. Vanishing errors

1. Precision instability

1. Update oscillations

### Solutions

1. Gradient clipping

1. Error normalization

1. Adaptive learning rates

1. Error gating

## Related Documentation

- [[predictive_network]]

- [[precision_mechanisms]]

- [[temporal_models]]

## References

- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2(1), 79-87.
- Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B, 360(1456), 815-836.
- Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and learning. Journal of Mathematical Psychology, 76, 198-211.
- Whittington, J. C. R., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. Neural Computation, 29(5), 1229-1262.
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience, 4, 215.
