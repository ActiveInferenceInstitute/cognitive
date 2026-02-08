---
title: Causal Analysis
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [causality, intervention, counterfactual, active_inference]
semantic_relations:
  - type: relates
    links: [[knowledge_base/mathematics/bayesian_networks]], [[experiment_design]], [[hidden_states]], [[knowledge_base/mathematics/d_separation]]]
---

# Causal Analysis

Causal analysis methods for understanding and evaluating Active Inference models, including interventional reasoning, counterfactual analysis, and structural causal modeling.

## Structural Causal Models

### Definition

```math
\begin{aligned}
& \text{Structural equations:} \quad X_i = f_i(\text{Pa}(X_i), U_i) \\
& \text{Graph:} \quad G = (V, E) \text{ (DAG)} \\
& \text{Distribution:} \quad p(x) = \prod_i p(x_i | \text{pa}(x_i))
\end{aligned}
```

### Interventional Distributions (do-calculus)

```math
p(y | do(x)) = \sum_z p(y|x,z)p(z) \quad \text{(backdoor adjustment)}
```

## Causal Discovery in Active Inference

Active Inference agents can learn causal structure through interventionist exploration:

```python
class CausalExplorer:
    def __init__(self, variables, candidate_graphs):
        self.variables = variables
        self.candidate_graphs = candidate_graphs

    def select_intervention(self, beliefs):
        info_gains = []
        for intervention in self.possible_interventions():
            ig = self.expected_information_gain(intervention, beliefs)
            info_gains.append(ig)
        return self.possible_interventions()[np.argmax(info_gains)]

    def expected_information_gain(self, intervention, beliefs):
        H_prior = entropy(beliefs)
        H_posterior = self.expected_posterior_entropy(intervention, beliefs)
        return H_prior - H_posterior
```

## Counterfactual Reasoning

```math
p(Y_{X=x'} = y | X = x, Y = y') = \text{counterfactual query}
```

## Related Topics

- [[experiment_design]] — Designing causal experiments
- [[hidden_states]] — Understanding hidden causal states
- [[knowledge_base/mathematics/bayesian_networks]] — Bayesian network structure
- [[knowledge_base/mathematics/d_separation]] — Conditional independence
