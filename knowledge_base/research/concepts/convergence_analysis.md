---
title: Convergence Analysis
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [convergence, variational_inference, optimization, analysis]
semantic_relations:
  - type: analyzes
    links: [[knowledge_base/mathematics/variational_inference]]
  - type: relates
    links: [[[stability_analysis]], [[belief_evolution]], [[knowledge_base/mathematics/numerical_stability]]]
---

# Convergence Analysis

Analysis of convergence properties in the variational inference algorithms underlying Active Inference, including convergence rates, conditions, and diagnostic tools.

## Convergence Criteria

### Fixed-Point Convergence

```math
\begin{aligned}
& \text{Convergence criterion:} \quad ||q^{(n+1)}(s) - q^{(n)}(s)|| < \epsilon \\
& \text{Free energy criterion:} \quad |F^{(n+1)} - F^{(n)}| < \delta \\
& \text{Gradient criterion:} \quad ||\nabla_\mu F|| < \eta
\end{aligned}
```

### Rate of Convergence

```math
\begin{aligned}
& \text{Linear:} \quad ||\mu^{(n+1)} - \mu^*|| \leq c \cdot ||\mu^{(n)} - \mu^*|| \\
& \text{Quadratic:} \quad ||\mu^{(n+1)} - \mu^*|| \leq c \cdot ||\mu^{(n)} - \mu^*||^2
\end{aligned}
```

## Diagnostics

```python
class ConvergenceDiagnostics:
    def __init__(self, tolerance=1e-6, max_iter=100):
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.history = []

    def check_convergence(self, current, previous):
        delta = np.abs(current - previous)
        self.history.append(delta)
        return delta < self.tolerance

    def estimate_rate(self):
        if len(self.history) < 3:
            return None
        ratios = [self.history[i+1] / (self.history[i] + 1e-16) for i in range(len(self.history)-1)]
        return np.mean(ratios[-5:])
```

## Factors Affecting Convergence

| Factor | Effect | Mitigation |
| --- | --- | --- |
| Model complexity | Slower convergence | Structured mean-field |
| Precision magnitude | Oscillation risk | Damping |
| Learning rate | Overshoot/undershoot | Adaptive scheduling |
| Initialization | Local minima | Multiple restarts |

## Related Topics

- [[stability_analysis]] — Stability of dynamic systems
- [[belief_evolution]] — Belief trajectory analysis
- [[knowledge_base/mathematics/variational_inference]] — Variational inference theory
- [[knowledge_base/mathematics/numerical_stability]] — Numerical stability
