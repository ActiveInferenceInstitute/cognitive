---
title: Power Analysis
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [statistics, power, sample_size, experimental_design]
semantic_relations:
  - type: relates
    links: [[[hypothesis_testing]], [[experiment_design]], [[statistical_analysis]], [[frequentist_analysis]]]
---

# Power Analysis

Statistical power analysis for Active Inference experiments, determining the sample sizes needed to detect meaningful effects in agent performance comparisons and model evaluations.

## Core Framework

### Power Components

```math
\begin{aligned}
& \text{Power} = 1 - \beta = P(\text{reject } H_0 | H_1 \text{ true}) \\
& \text{Required sample size:} \quad n = \left(\frac{z_{1-\alpha/2} + z_{1-\beta}}{d}\right)^2 \cdot k
\end{aligned}
```

where $d$ is the effect size (Cohen's d), $\alpha$ is the significance level, $\beta$ is the type II error rate, and $k$ depends on the test design.

### Effect Size Guidelines

| Effect Size | Cohen's d | Typical Active Inference Context |
| --- | --- | --- |
| Small | 0.2 | Subtle parameter changes |
| Medium | 0.5 | Different exploration strategies |
| Large | 0.8 | Architectural differences |
| Very Large | 1.2 | With vs. without learning |

## Implementation

```python
from scipy import stats
import numpy as np

def power_analysis(effect_size, alpha=0.05, power=0.8, test='two-sample'):
    """Compute required sample size for given effect size and desired power."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    if test == 'two-sample':
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    elif test == 'paired':
        n = ((z_alpha + z_beta) / effect_size) ** 2
    elif test == 'one-sample':
        n = ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

def simulation_based_power(agent_factory_a, agent_factory_b, env, n_trials=100,
                           n_simulations=1000, alpha=0.05):
    """Estimate power via simulation for complex Active Inference comparisons."""
    rejections = 0
    for _ in range(n_simulations):
        results_a = [run_trial(agent_factory_a(), env) for _ in range(n_trials)]
        results_b = [run_trial(agent_factory_b(), env) for _ in range(n_trials)]
        _, p_value = stats.ttest_ind(results_a, results_b)
        if p_value < alpha:
            rejections += 1
    return rejections / n_simulations
```

### Sample Size Table for Active Inference Experiments

| Comparison | Expected d | n per group (power=0.8) | n per group (power=0.9) |
| --- | --- | --- | --- |
| Gamma: 1.0 vs 4.0 | 0.8 | 26 | 34 |
| With/without learning | 1.2 | 12 | 16 |
| Planning horizon 3 vs 10 | 0.5 | 64 | 86 |
| Noise σ=0.1 vs σ=0.5 | 0.6 | 45 | 60 |

## Practical Recommendations

1. **Always run power analysis before experiments** — avoids underpowered studies
2. **Use simulation-based power** for complex metrics (free energy convergence curves)
3. **Report effect sizes** alongside p-values
4. **Consider Bayesian approaches** — sequential Bayes factors allow flexible stopping

## Related Topics

- [[hypothesis_testing]] — Statistical testing methods
- [[experiment_design]] — Experiment design principles
- [[statistical_analysis]] — Statistical analysis overview
- [[frequentist_analysis]] — Frequentist methods
