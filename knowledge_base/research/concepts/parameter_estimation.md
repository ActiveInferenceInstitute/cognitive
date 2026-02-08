---
title: Parameter Estimation
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [estimation, parameters, learning, inference]
semantic_relations:
  - type: relates
    links: [[[bayesian_analysis]], [[knowledge_base/mathematics/bayesian_generative_models]], [[knowledge_base/cognitive/learning_mechanisms]]]
---

# Parameter Estimation

Methods for estimating generative model parameters in Active Inference, including maximum likelihood, Bayesian estimation, and online learning approaches.

## Estimation Methods

### Maximum Likelihood

```math
\hat{\theta}_{ML} = \argmax_\theta \ln p(o|\theta) = \argmax_\theta \sum_t \ln p(o_t|\theta)
```

### Maximum A Posteriori

```math
\hat{\theta}_{MAP} = \argmax_\theta [\ln p(o|\theta) + \ln p(\theta)]
```

### Full Bayesian (Variational)

```math
q^*(\theta) = \argmin_q D_{KL}[q(\theta)||p(\theta|o)]
```

For Dirichlet-Categorical models (standard in discrete Active Inference):

```math
\begin{aligned}
& \text{Prior:} \quad \theta \sim \text{Dir}(\alpha_0) \\
& \text{Posterior:} \quad \theta|o \sim \text{Dir}(\alpha_0 + n)
\end{aligned}
```

## Implementation

```python
class ParameterEstimator:
    def __init__(self, prior_concentration):
        self.alpha = prior_concentration.copy()

    def update(self, observations, states):
        for o, s in zip(observations, states):
            self.alpha[o, s] += 1.0

    def point_estimate(self, method='mean'):
        if method == 'mean':
            return self.alpha / self.alpha.sum(axis=0, keepdims=True)
        elif method == 'mode':
            return (self.alpha - 1) / (self.alpha.sum(axis=0, keepdims=True) - self.alpha.shape[0])
```

## Related Topics

- [[bayesian_analysis]] — Bayesian estimation framework
- [[knowledge_base/mathematics/bayesian_generative_models]] — Generative model theory
- [[knowledge_base/cognitive/learning_mechanisms]] — Learning in Active Inference
