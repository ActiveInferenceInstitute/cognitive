---
title: Control Variables
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [experimental_design, variables, control, methodology, research]
semantic_relations:
  - type: relates
    links: [[[experiment_design]], [[parameter_estimation]], [[power_analysis]], [[statistical_analysis]], [[knowledge_base/cognitive/active_inference]]]
---

# Control Variables

Specification and management of experimental variables in Active Inference research. Proper variable control isolates causal effects and ensures reproducible results.

## Variable Classification

### Independent Variables (Manipulated)

| Variable | Symbol | Typical Range | Effect On |
| --- | --- | --- | --- |
| Policy precision | $\gamma$ | 0.5 - 16 | Exploration-exploitation trade-off |
| Learning rate | $\alpha$ | 0.001 - 0.5 | Adaptation speed and stability |
| Planning horizon | $T$ | 1 - 20 | Look-ahead depth and policy quality |
| Observation noise | $\sigma_o$ | 0.01 - 1.0 | Sensory reliability |
| State dimension | $|S|$ | 2 - 100 | Problem complexity |
| Preference strength | $||C||$ | 0.1 - 10 | Goal-directedness |
| Prior confidence | $\alpha_D$ | 0.1 - 100 | Flexibility of initial beliefs |

### Dependent Variables (Measured)

| Metric | Formula | Measures |
| --- | --- | --- |
| Free energy | $F = -\text{ELBO}$ | Model fit quality |
| Expected free energy | $G(\pi)$ | Planning quality |
| Task performance | Application-specific | Goal achievement |
| Convergence time | Steps to $\Delta F < \epsilon$ | Inference efficiency |
| Belief accuracy | $D_{KL}[q(s)||\delta_{s_{true}}]$ | State estimation quality |
| Computational cost | Wall-clock time, iterations | Practical efficiency |

### Controlled Variables

Variables held constant across conditions to isolate effects:

```python
class ControlledExperiment:
    CONTROLLED = {
        'random_seed': 42,
        'n_trials': 100,
        'n_seeds': 10,
        'inference_iterations': 16,
        'convergence_threshold': 1e-6,
        'environment_structure': 'T-maze',
    }

    def __init__(self, independent_var, levels):
        self.independent_var = independent_var
        self.levels = levels

    def generate_conditions(self):
        conditions = []
        for level in self.levels:
            config = self.CONTROLLED.copy()
            config[self.independent_var] = level
            conditions.append(config)
        return conditions

    def validate_control(self, results):
        for var, expected in self.CONTROLLED.items():
            if var == self.independent_var:
                continue
            actual = set(r[var] for r in results)
            assert len(actual) == 1, f"Control variable {var} varied: {actual}"
```

### Confounding Variables

Common confounds in Active Inference experiments and mitigation strategies:

| Confound | Risk | Mitigation |
| --- | --- | --- |
| Random seed | Results depend on specific seed | Average over many seeds |
| Initialization | Starting beliefs affect trajectory | Standardize or randomize |
| Environment order | Learning depends on sequence | Counterbalance or randomize |
| Computational precision | Floating-point artifacts | Use stable implementations |

## Related Topics

- [[experiment_design]] — Experiment design principles
- [[parameter_estimation]] — Parameter estimation methods
- [[statistical_analysis]] — Statistical analysis methods
- [[power_analysis]] — Statistical power analysis
- [[knowledge_base/cognitive/active_inference]] — Core Active Inference framework\n