---
title: Experiment Design
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [experimental_design, methodology, research, active_inference]
semantic_relations:
  - type: relates
    links: [[[control_variables]], [[hypothesis_testing]], [[power_analysis]], [[sampling_strategies]]]
---

# Experiment Design

Principles and methods for designing experiments to evaluate Active Inference models, including factorial designs, ablation studies, and simulation protocols.

## Factorial Experiment Design

### Full Factorial

```python
from itertools import product

def full_factorial_design(factors):
    levels = [factors[f] for f in factors]
    conditions = list(product(*levels))
    return [dict(zip(factors.keys(), c)) for c in conditions]

factors = {
    'gamma': [1.0, 4.0, 16.0],
    'learning_rate': [0.01, 0.1],
    'horizon': [3, 5, 10],
}
conditions = full_factorial_design(factors)
```

### Ablation Studies

Systematically removing components to assess their contribution:

| Condition | A matrix | B matrix | C matrix | Learning | Expected Effect |
| --- | --- | --- | --- | --- | --- |
| Full model | ✓ | ✓ | ✓ | ✓ | Baseline |
| No learning | ✓ | ✓ | ✓ | ✗ | No adaptation |
| No preferences | ✓ | ✓ | ✗ | ✓ | Random behavior |
| No observation | ✗ | ✓ | ✓ | ✓ | Open-loop control |

## Simulation Protocol

```python
class ExperimentProtocol:
    def __init__(self, conditions, n_seeds=10, n_trials=100):
        self.conditions = conditions
        self.n_seeds = n_seeds
        self.n_trials = n_trials

    def run(self):
        results = []
        for cond in self.conditions:
            for seed in range(self.n_seeds):
                agent = create_agent(cond, seed=seed)
                trial_data = run_trials(agent, self.n_trials)
                results.append({'condition': cond, 'seed': seed, 'data': trial_data})
        return results
```

## Related Topics

- [[control_variables]] — Variable control
- [[hypothesis_testing]] — Statistical testing
- [[power_analysis]] — Sample size determination
- [[sampling_strategies]] — Sampling methods
