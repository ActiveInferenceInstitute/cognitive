---
title: Frequentist Analysis
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [frequentist, statistics, hypothesis_testing, p_values]
semantic_relations:
  - type: contrasts_with
    links: [[bayesian_analysis]]
  - type: relates
    links: [[[hypothesis_testing]], [[power_analysis]], [[statistical_analysis]]]
---

# Frequentist Analysis

Frequentist statistical methods applied to Active Inference experiments, providing complementary perspectives to Bayesian analysis for hypothesis testing, confidence intervals, and effect size estimation.

## Core Methods

### Hypothesis Testing

```math
\begin{aligned}
& H_0: \theta = \theta_0 \quad \text{(null hypothesis)} \\
& H_1: \theta \neq \theta_0 \quad \text{(alternative)} \\
& \text{Test statistic:} \quad T = \frac{\hat{\theta} - \theta_0}{\text{SE}(\hat{\theta})} \\
& p\text{-value} = P(|T| \geq |t_{obs}| | H_0)
\end{aligned}
```

### Common Tests for Active Inference Experiments

| Test | Use Case | Assumptions |
| --- | --- | --- |
| Paired t-test | Before/after intervention | Normality, paired data |
| ANOVA | Multiple conditions | Normality, homoscedasticity |
| Wilcoxon | Non-normal paired data | Symmetry |
| Kruskal-Wallis | Non-normal groups | Independence |
| Permutation test | Distribution-free | Exchangeability |

### Effect Sizes

```math
\begin{aligned}
& \text{Cohen's } d = \frac{\bar{x}_1 - \bar{x}_2}{s_p} \\
& \eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}} \\
& r = \frac{Z}{\sqrt{N}}
\end{aligned}
```

## Implementation

```python
from scipy import stats

def compare_conditions(condition_a, condition_b):
    t_stat, p_value = stats.ttest_ind(condition_a, condition_b)
    effect_size = (np.mean(condition_a) - np.mean(condition_b)) / np.sqrt(
        (np.var(condition_a) + np.var(condition_b)) / 2)
    return {'t': t_stat, 'p': p_value, 'd': effect_size}
```

## Related Topics

- [[bayesian_analysis]] — Bayesian alternative
- [[hypothesis_testing]] — Hypothesis testing theory
- [[power_analysis]] — Sample size planning
- [[statistical_analysis]] — General statistical methods
