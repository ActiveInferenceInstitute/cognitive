---
title: Gradient Descent
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [optimization, gradient, learning, numerical_methods]
semantic_relations:
  - type: implements
    links: [[knowledge_base/mathematics/optimization_theory]]
  - type: relates
    links: [[knowledge_base/mathematics/natural_gradients]], [[knowledge_base/mathematics/numerical_methods]], [[convergence_analysis]]]
---

# Gradient Descent

Gradient descent methods underpin the parameter and belief updates in Active Inference. Free energy is minimized through gradient flow on the variational free energy functional.

## Free Energy Gradient Flow

### Belief Updates

```math
\dot{\mu} = -\kappa \frac{\partial F}{\partial \mu}
```

### Parameter Updates

```math
\dot{\theta} = -\eta \frac{\partial F}{\partial \theta}
```

### Natural Gradient (Information Geometry)

```math
\dot{\theta} = -\eta \mathcal{G}^{-1} \frac{\partial F}{\partial \theta}
```

where $\mathcal{G}$ is the Fisher information matrix.

## Variants

| Method | Update Rule | Convergence | Memory |
| --- | --- | --- | --- |
| Vanilla GD | $\theta - \eta \nabla F$ | Linear | O(1) |
| Momentum | $\theta - \eta \nabla F + \alpha \Delta\theta$ | Faster | O(d) |
| Adam | Adaptive first/second moments | Fast | O(d) |
| Natural GD | $\theta - \eta G^{-1} \nabla F$ | Fastest | O(d²) |

## Implementation

```python
class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def step(self, params, gradient):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity - self.lr * gradient
        return params + self.velocity
```

## Related Topics

- [[knowledge_base/mathematics/natural_gradients]] — Natural gradient methods
- [[knowledge_base/mathematics/optimization_theory]] — Optimization theory
- [[knowledge_base/mathematics/numerical_methods]] — Numerical methods
- [[convergence_analysis]] — Convergence properties
