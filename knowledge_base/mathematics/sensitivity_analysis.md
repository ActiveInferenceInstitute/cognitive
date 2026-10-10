---
title: Sensitivity Analysis
description: "Methods for measuring how Active Inference agent outputs respond to parameters, including local derivatives, Morris elementary effects and Sobol indices, with a toy agent example."
type: guide
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- analysis
- robustness
semantic_relations:
  - type: relates_to
    links:
      - "[[parameter_tuning]]"
      - "[[hyperparameter_optimization]]"
      - "[[worked_example_efe]]"
---


# Sensitivity Analysis

Quantifies how outputs change with parameters (e.g., $\gamma$, precisions). Includes local derivatives and global variance-based indices. In an Active Inference agent the outputs of interest are usually the policy distribution or the chosen action, and the inputs are the temperature or precision, the preference vector $C$, the likelihood $A$ and the current belief. Run a sensitivity analysis before tuning ([[parameter_tuning]]) to learn which inputs deserve attention, and before trusting a result to learn how fragile it is.

## Local sensitivity

For a scalar output $y = f(\theta)$ at a base point $\theta_0$, the local sensitivity is the gradient $\partial f / \partial \theta_i$, estimated by central differences $\big(f(\theta_0 + h e_i) - f(\theta_0 - h e_i)\big)/2h$. It is cheap (two evaluations per input), exact in the limit, and valid only near $\theta_0$. To compare inputs of different units, scale by $\theta_i / f$ (elasticity) or by a plausible range.

## One-at-a-time sweeps and elementary effects

Varying one input over its range while holding the rest fixed shows the shape of the response, including saturation and thresholds that a derivative misses. The Morris method repeats this from many random base points and summarises the mean and spread of the effects, giving a screening of which inputs matter and which interact, for far fewer evaluations than a full global analysis.

## Global variance-based indices

Treat inputs as independent random variables over their ranges and decompose the output variance. The first-order Sobol index $S_i = \mathrm{Var}\big(\mathbb{E}[y \mid \theta_i]\big)/\mathrm{Var}(y)$ is the share of variance explained by $\theta_i$ alone. The total index $S_{T_i}$ also counts every interaction involving $\theta_i$. If $\sum_i S_i \ll 1$ the model is dominated by interactions, and local derivatives or one-at-a-time sweeps will mislead. Saltelli's estimator for $S_i$ and Jansen's for $S_{T_i}$ need $N(d+2)$ model evaluations for $d$ inputs.

## Example: the move probability of the toy agent

Output: $P(\text{move}) = \mathrm{softmax}(-G/T)$ for the second action of [[worked_example_efe]], with inputs the preference strength $C_0$, the ambiguity entry $A_{0,1}$, the belief $q(s_0)$ and the temperature $T$. At the base point $C_0 = 2$, $A_{0,1} = 0.5$, $q(s_0) = 0.3$, $T = 1$ the output is $0.6087$, matching the worked example. Central-difference gradients there:

| input | $\partial P / \partial \text{input}$ |
|-------|------------------------|
| preference strength $C_0$ | $+0.0572$ |
| ambiguity entry $A_{0,1}$ | $-0.2474$ |
| belief $q(s_0)$ | $-0.2316$ |
| temperature $T$ | $-0.1052$ |

Per unit change the output is most sensitive to the likelihood entry and the belief, and least to preference strength. Units are not comparable, though: the ranges used below span $0.4$, $0.8$, $4$ and $3.75$ respectively. Global indices over the ranges $C_0 \in [0, 4]$, $A_{0,1} \in [0.3, 0.7]$, $q(s_0) \in [0.1, 0.9]$, $T \in [0.25, 4]$ (seed 0, $N = 2000$) tell a different story:

| input | first-order | total |
|-------|-------------|-------|
| preference strength | 0.176 | 0.394 |
| ambiguity entry | $-0.008$ | 0.063 |
| belief | 0.091 | 0.410 |
| temperature | 0.172 | 0.510 |

The first-order indices sum to about $0.43$, so most variance comes from interactions: totals are well above firsts. Over wide ranges temperature and preference strength dominate, and the ambiguity entry is negligible even though its local gradient was the largest. The small negative first-order value is estimator noise, which is a few hundredths at this sample size. A local analysis at one base point ranks the inputs differently from a global one, and neither is wrong; they answer different questions.

## Practical notes

- Choose ranges from plausible values, not from convenience: Sobol indices depend entirely on them.
- Use common random numbers or fix `InferenceConfig.seed` when the model is stochastic, otherwise noise leaks into the indices.
- Report the sample size and the estimator noise; indices below the noise floor should be read as zero.
- Output choice matters. A sensitivity of the policy distribution differs from that of $G$ itself, since the softmax saturates ([[precision_parameter]]).

## Reproducing the numbers

```python
import numpy as np

from cognitive.models.active_inference import DiscreteGenerativeModel
from cognitive.utils.matrix_utils import softmax

B = np.zeros((2, 2, 2))
B[:, :, 0] = np.eye(2)
B[:, :, 1] = [[0.9, 0.9], [0.1, 0.1]]


def p_move(preference, a01, belief0, temperature):
    """Probability of the 'move' action under softmax(-G / T)."""
    A = np.array([[0.9, a01], [0.1, 1.0 - a01]])
    model = DiscreteGenerativeModel(A, B, np.array([preference, 0.0]), np.full(2, 0.5), np.full(2, 0.5))
    beliefs = np.array([belief0, 1.0 - belief0])
    G = np.array([model.expected_free_energy(beliefs, a)[0] for a in range(2)])
    return float(softmax(-G, temperature=temperature)[1])


base = dict(preference=2.0, a01=0.5, belief0=0.3, temperature=1.0)
print("base", round(p_move(**base), 4))

# Local sensitivity: central finite differences of P(move) at the base point.
for name, h in (("preference", 1e-4), ("a01", 1e-4), ("belief0", 1e-4), ("temperature", 1e-4)):
    up, down = dict(base), dict(base)
    up[name] += h
    down[name] -= h
    print("d P(move) / d", name, round((p_move(**up) - p_move(**down)) / (2 * h), 4))

# Global sensitivity: Sobol indices with the Saltelli (first-order) and Jansen (total) estimators.
ranges = {"preference": (0.0, 4.0), "a01": (0.3, 0.7), "belief0": (0.1, 0.9), "temperature": (0.25, 4.0)}
names = list(ranges)
rng = np.random.default_rng(0)
N = 2000
low = np.array([ranges[n][0] for n in names])
high = np.array([ranges[n][1] for n in names])
M1 = low + (high - low) * rng.random((N, len(names)))
M2 = low + (high - low) * rng.random((N, len(names)))


def run(matrix):
    return np.array([p_move(**dict(zip(names, row))) for row in matrix])


f1, f2 = run(M1), run(M2)
variance = np.var(np.concatenate([f1, f2]))
for i, name in enumerate(names):
    mixed = M1.copy()
    mixed[:, i] = M2[:, i]
    fi = run(mixed)
    first = np.mean(f2 * (fi - f1)) / variance
    total = 0.5 * np.mean((f1 - fi) ** 2) / variance
    print(f"{name:12s} first-order {first:6.3f}   total {total:6.3f}")
```

## Links

- [[precision_parameter]]
- [[optimization_strategies]]
- [[parameter_tuning]] · [[hyperparameter_optimization]]
