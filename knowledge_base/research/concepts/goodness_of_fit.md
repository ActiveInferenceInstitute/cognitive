---
title: Goodness of Fit
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [model_evaluation, statistics, fit, validation, bayesian]
semantic_relations:
  - type: relates
    links: [[[model_comparison]], [[bayesian_analysis]], [[validation_methods]], [[knowledge_base/mathematics/variational_free_energy]], [[statistical_analysis]]]
---

# Goodness of Fit

Metrics and methods for assessing how well an Active Inference generative model explains observed data, including accuracy-complexity tradeoffs, information criteria, posterior predictive checks, and residual analysis.

## Core Metrics

### Free Energy as Model Evidence

The variational free energy provides a natural goodness-of-fit measure:

```math
F = \underbrace{-\mathbb{E}_{q(s)}[\ln p(o|s)]}_{\text{accuracy (lower = better fit)}} + \underbrace{D_{KL}[q(s)||p(s)]}_{\text{complexity (Occam penalty)}}
```

Lower $F$ (higher evidence $-F$) indicates better fit with appropriate complexity. This naturally implements Occam's razor: simpler models are preferred unless the extra complexity is justified by improved fit.

### Information Criteria

```math
\begin{aligned}
& \text{AIC} = -2\ln L(\hat{\theta}) + 2k \\
& \text{BIC} = -2\ln L(\hat{\theta}) + k \ln n \\
& \text{DIC} = -2\ln L(\bar{\theta}) + 2p_D \\
& \text{WAIC} = -2\sum_i \ln \mathbb{E}_{\theta|D}[p(o_i|\theta)] + 2p_{\text{WAIC}}
\end{aligned}
```

### Comparison of Criteria

| Criterion | Complexity Penalty | Bayesian? | Use Case |
| --- | --- | --- | --- |
| AIC | $2k$ | No | Prediction-focused |
| BIC | $k \ln n$ | Approximation | Model identification |
| DIC | $2p_D$ | Semi | Hierarchical models |
| WAIC | $p_{\text{WAIC}}$ | Yes | General purpose |
| VFE ($-F$) | $D_{KL}$ | Yes | Active Inference native |

## Posterior Predictive Checks

### Visual Check

```python
def posterior_predictive_check(model, observed_data, n_samples=1000):
    """Generate posterior predictive samples and compare to observed data."""
    simulated = []
    for _ in range(n_samples):
        theta = model.sample_posterior()
        sim_data = model.generate_data(theta)
        simulated.append(sim_data)

    simulated = np.array(simulated)
    p_value = np.mean(simulated.mean(axis=1) >= observed_data.mean())

    return {
        'p_value': p_value,
        'observed_mean': observed_data.mean(),
        'predicted_mean': simulated.mean(),
        'predicted_ci_95': np.percentile(simulated.mean(axis=1), [2.5, 97.5]),
    }
```

### Residual Analysis

```math
r_t = o_t - \hat{o}_t = o_t - A \hat{s}_t
```

Well-fitting models should have residuals that are approximately:
- Zero-mean: $\mathbb{E}[r_t] \approx 0$
- Uncorrelated: $\text{corr}(r_t, r_{t+k}) \approx 0$ for $k > 0$
- Homoscedastic: $\text{Var}(r_t) \approx \text{const}$

## Related Topics

- [[model_comparison]] — Comparing alternative models
- [[bayesian_analysis]] — Bayesian model evaluation
- [[validation_methods]] — Validation techniques
- [[statistical_analysis]] — Statistical methods
- [[knowledge_base/mathematics/variational_free_energy]] — Free energy as evidence bound
