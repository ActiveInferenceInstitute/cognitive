---
title: Bayesian Analysis
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [bayesian, statistics, inference, active_inference]
semantic_relations:
  - type: foundation
    links: [[knowledge_base/mathematics/bayes_theorem]]
  - type: relates
    links: [[[frequentist_analysis]], [[parameter_estimation]], [[model_comparison]], [[hypothesis_testing]], [[knowledge_base/mathematics/bayesian_networks]]]
---

# Bayesian Analysis

Bayesian analysis provides the statistical foundation for Active Inference, enabling principled reasoning under uncertainty through posterior computation, model comparison, and sequential updating.

## Core Framework

### Bayes' Theorem

```math
p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)} = \frac{p(D|\theta)p(\theta)}{\int p(D|\theta')p(\theta') d\theta'}
```

### Posterior Computation

For conjugate models, the posterior has a closed form:

```math
\text{Prior: } \theta \sim \text{Beta}(\alpha, \beta) \quad \Rightarrow \quad \text{Posterior: } \theta|D \sim \text{Beta}(\alpha + s, \beta + n - s)
```

For non-conjugate models, approximate inference is required:
- Variational inference (minimizing $D_{KL}[q(\theta)||p(\theta|D)]$)
- MCMC sampling
- Laplace approximation

## Implementation

```python
import numpy as np
from scipy import stats

class BayesianAnalyzer:
    def __init__(self, prior_params):
        self.prior = prior_params
        self.posterior = prior_params.copy()

    def update(self, data):
        likelihood = self.compute_likelihood(data)
        self.posterior = self.compute_posterior(likelihood)
        return self.posterior

    def model_evidence(self, data):
        return np.exp(-self.compute_free_energy(data))

    def bayes_factor(self, data, alternative_model):
        return self.model_evidence(data) / alternative_model.model_evidence(data)
```

## Key Concepts

| Concept | Formula | Role in Active Inference |
| --- | --- | --- |
| Posterior | $p(\theta|D)$ | Updated beliefs |
| Evidence | $p(D) = \int p(D|\theta)p(\theta)d\theta$ | Model comparison |
| Bayes factor | $BF_{12} = p(D|M_1)/p(D|M_2)$ | Model selection |
| Predictive | $p(D_{new}|D) = \int p(D_{new}|\theta)p(\theta|D)d\theta$ | Forecasting |

## Related Topics

- [[frequentist_analysis]] — Frequentist approaches for comparison
- [[model_comparison]] — Bayesian model comparison
- [[parameter_estimation]] — Parameter estimation methods
- [[hypothesis_testing]] — Statistical testing
- [[knowledge_base/mathematics/bayes_theorem]] — Bayes' theorem
- [[knowledge_base/mathematics/bayesian_networks]] — Graphical models

## References

- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.).
- Friston, K. (2010). The free-energy principle: a unified brain theory?
