---
title: Hypothesis Testing
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [statistics, hypothesis, testing, inference, methodology]
semantic_relations:
  - type: relates
    links: [[[bayesian_analysis]], [[frequentist_analysis]], [[power_analysis]], [[model_comparison]], [[experiment_design]]]
---

# Hypothesis Testing

Statistical hypothesis testing for evaluating Active Inference models and experiments, integrating both frequentist significance testing and Bayesian hypothesis evaluation via Bayes factors.

## Frequentist Approach

### Standard Framework

```math
\begin{aligned}
& H_0: \theta = \theta_0 \quad \text{(null hypothesis — e.g., no benefit from learning)} \\
& H_1: \theta \neq \theta_0 \quad \text{(alternative — learning improves performance)} \\
& \alpha: \text{significance level (typically 0.05)} \\
& \beta: \text{Type II error rate (typically 0.2)} \\
& \text{Power} = 1 - \beta
\end{aligned}
```

### Decision Table

| | $H_0$ True | $H_1$ True |
| --- | --- | --- |
| Reject $H_0$ | Type I error ($\alpha$) | Correct (power) |
| Fail to reject | Correct ($1-\alpha$) | Type II error ($\beta$) |

### Common Tests for Active Inference

| Hypothesis | Test | Active Inference Example |
| --- | --- | --- |
| Two means differ | t-test | Mean free energy: learning vs no-learning |
| Multiple means | ANOVA | Performance across 4 precision levels |
| Pre/post change | Paired t | Before vs. after model update |
| Distribution shape | KS test | Belief distribution normality |
| Trend over time | Linear regression | Free energy convergence slope |

## Bayesian Approach

### Bayes Factors

```math
BF_{10} = \frac{p(D|H_1)}{p(D|H_0)} = \frac{\int p(D|\theta, H_1)p(\theta|H_1)d\theta}{\int p(D|\theta, H_0)p(\theta|H_0)d\theta}
```

### Evidence Interpretation

| $BF$ Range | $\ln BF$ | Evidence Strength | Recommendation |
| --- | --- | --- | --- |
| 1-3 | 0-1.1 | Anecdotal | Collect more data |
| 3-10 | 1.1-2.3 | Moderate | Worth noting |
| 10-30 | 2.3-3.4 | Strong | Reliable conclusion |
| 30-100 | 3.4-4.6 | Very strong | Convincing |
| >100 | >4.6 | Decisive | Definitive |

## Implementation

```python
from scipy import stats
import numpy as np

def comprehensive_hypothesis_test(group_a, group_b, alpha=0.05):
    """Run both frequentist and Bayesian hypothesis tests."""
    # Frequentist
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    effect_size = (np.mean(group_a) - np.mean(group_b)) / np.sqrt(
        (np.var(group_a) + np.var(group_b)) / 2)

    # Bayesian (using BIC approximation)
    n_a, n_b = len(group_a), len(group_b)
    n_total = n_a + n_b
    bic_h0 = n_total * np.log(np.var(np.concatenate([group_a, group_b])))
    bic_h1 = n_a * np.log(np.var(group_a)) + n_b * np.log(np.var(group_b))
    log_bf = (bic_h0 - bic_h1) / 2

    return {
        'frequentist': {'t': t_stat, 'p': p_value, 'd': effect_size,
                        'significant': p_value < alpha},
        'bayesian': {'log_BF': log_bf, 'BF': np.exp(log_bf),
                     'evidence': classify_evidence(np.exp(log_bf))},
    }

def classify_evidence(bf):
    if bf > 100: return 'Decisive'
    if bf > 30: return 'Very strong'
    if bf > 10: return 'Strong'
    if bf > 3: return 'Moderate'
    return 'Anecdotal'
```

## Active Inference-Specific Considerations

1. **Non-independence**: Trials within an agent session are autocorrelated — use mixed-effects models or summarize per-session
2. **Multiple comparisons**: When testing many conditions, apply Bonferroni or False Discovery Rate correction
3. **Effect size first**: Report Cohen's d or $\eta^2$ alongside p-values — statistical significance ≠ practical significance

## Related Topics

- [[bayesian_analysis]] — Bayesian statistical methods
- [[frequentist_analysis]] — Frequentist methods
- [[power_analysis]] — Sample size planning
- [[model_comparison]] — Model selection via hypothesis testing
- [[experiment_design]] — Designing experiments\n