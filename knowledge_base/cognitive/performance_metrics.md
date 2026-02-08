---
title: Performance Metrics
type: concept
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - performance
  - evaluation
  - metrics
  - free-energy
  - active-inference
semantic_relations:
  - type: relates
    links:
      - [[quality_metrics]]
      - [[quality_assessment]]
      - [[simulation_studies]]
      - [[active_inference_agent]]
---

# Performance Metrics

## Overview

Performance metrics for Active Inference agents assess how well the agent perceives, learns, plans, and acts. Unlike traditional ML metrics focused on prediction accuracy, Active Inference metrics span multiple dimensions: free energy minimization, goal achievement, epistemic value, and computational efficiency.

## Core Metrics

### Free Energy Metrics

| Metric | Formula | Interpretation |
| --- | --- | --- |
| Variational free energy | $F = D_{KL}[q(s) \| p(s)] - \ln p(o)$ | Inference quality |
| Expected free energy | $G(\pi) = \sum_\tau \text{ambiguity} + \text{risk}$ | Planning quality |
| Free energy gradient | $\nabla F$ | Convergence rate |
| Accuracy | $-\mathbb{E}_q[\ln p(o\|s)]$ | Model fit to observations |
| Complexity | $D_{KL}[q(s) \| p(s)]$ | Deviation from priors |

### Task Performance

```python
class AgentMetrics:
    def __init__(self):
        self.free_energies = []
        self.rewards = []
        self.prediction_errors = []
        self.beliefs_entropy = []
    
    def log_step(self, agent, observation, reward=None):
        self.free_energies.append(agent.compute_free_energy())
        self.prediction_errors.append(agent.last_prediction_error)
        self.beliefs_entropy.append(entropy(agent.beliefs))
        if reward is not None:
            self.rewards.append(reward)
    
    def summary(self):
        return {
            'mean_free_energy': np.mean(self.free_energies),
            'cumulative_reward': np.sum(self.rewards),
            'mean_prediction_error': np.mean(self.prediction_errors),
            'final_entropy': self.beliefs_entropy[-1],
            'convergence_rate': self._compute_convergence_rate(),
        }
    
    def _compute_convergence_rate(self):
        fe = np.array(self.free_energies)
        if len(fe) < 2:
            return 0.0
        return -(fe[-1] - fe[0]) / len(fe)
```

### Epistemic and Pragmatic Performance

| Dimension | Metric | Good Performance |
| --- | --- | --- |
| Epistemic | Entropy reduction over time | Beliefs become more certain |
| Pragmatic | Preference achievement | Observations match C vector |
| Efficiency | Steps to convergence | Fast belief settling |
| Adaptability | Recovery after perturbation | Quick relearning |
| Robustness | Performance under noise | Stable with corrupted inputs |

## Related Topics

- [[quality_metrics]] — Quality metrics
- [[quality_assessment]] — Assessment framework
- [[simulation_studies]] — Simulation-based evaluation
- [[active_inference_agent]] — Agent architecture
