---
title: Model Comparison
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [model_selection, bayesian, evidence, comparison, free_energy]
semantic_relations:
  - type: relates
    links: [[[bayesian_analysis]], [[goodness_of_fit]], [[knowledge_base/mathematics/variational_free_energy]], [[knowledge_base/cognitive/model_selection]], [[hypothesis_testing]]]
---

# Model Comparison

Bayesian model comparison in Active Inference uses the variational free energy (negative log model evidence) to select among competing generative models, balancing accuracy against complexity.

## Model Evidence

### Free Energy Bound

```math
\ln p(o|m) \geq -F_m = \underbrace{\mathbb{E}_{q(s)}[\ln p(o|s,m)]}_{\text{accuracy}} - \underbrace{D_{KL}[q(s)||p(s|m)]}_{\text{complexity}}
```

### Bayes Factor

```math
BF_{12} = \frac{p(o|m_1)}{p(o|m_2)} \approx \exp(F_2 - F_1)
```

### Interpretation Scale

| $\ln BF_{12}$ | $BF_{12}$ | Evidence for $m_1$ |
| --- | --- | --- |
| 0-1 | 1-3 | Anecdotal |
| 1-3 | 3-20 | Positive |
| 3-5 | 20-150 | Strong |
| >5 | >150 | Very strong |

## Family-Level Comparison

When comparing families of models (e.g., all models with learning vs. all without):

```math
p(\text{family}_k|o) = \sum_{m \in \text{family}_k} p(m|o) = \sum_{m \in \text{family}_k} \frac{p(o|m)p(m)}{\sum_{m'} p(o|m')p(m')}
```

## Implementation

```python
class BayesianModelComparison:
    def __init__(self, models):
        self.models = models

    def compare(self, data):
        free_energies = {}
        for name, model in self.models.items():
            free_energies[name] = model.compute_free_energy(data)

        best = min(free_energies, key=free_energies.get)
        posterior = self.compute_model_posterior(free_energies)
        bayes_factors = self.compute_bayes_factors(free_energies, best)

        return {
            'free_energies': free_energies,
            'best_model': best,
            'posterior': posterior,
            'bayes_factors': bayes_factors,
        }

    def compute_model_posterior(self, free_energies):
        F = np.array(list(free_energies.values()))
        log_evidence = -F
        log_posterior = log_evidence - np.max(log_evidence)
        posterior = np.exp(log_posterior) / np.sum(np.exp(log_posterior))
        return dict(zip(free_energies.keys(), posterior))

    def compute_bayes_factors(self, free_energies, reference):
        F_ref = free_energies[reference]
        return {name: np.exp(F_ref - F) for name, F in free_energies.items()}
```

### Common Model Comparison Scenarios

```mermaid
graph TD
    subgraph "Model Comparison Workflow"
        A[Define Model Space] --> B[Fit All Models]
        B --> C[Compute Free Energies]
        C --> D[Bayes Factors]
        D --> E[Model Posterior]
        E --> F[Bayesian Model Averaging]
    end
    style A fill:#f9d,stroke:#333
    style F fill:#bfb,stroke:#333
```

## Bayesian Model Averaging

When no single model is clearly best, weight predictions by posterior:

```math
p(o_{new}|o) = \sum_m p(o_{new}|m) p(m|o)
```

## Related Topics

- [[bayesian_analysis]] — Bayesian methods
- [[goodness_of_fit]] — Fit evaluation
- [[hypothesis_testing]] — Hypothesis testing
- [[knowledge_base/mathematics/variational_free_energy]] — Free energy bound
- [[knowledge_base/cognitive/model_selection]] — Model selection theory
