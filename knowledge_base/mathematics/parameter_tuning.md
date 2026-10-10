---
title: Parameter Tuning
type: guide
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- tuning
- optimization
aliases:
  - optimization_parameters
semantic_relations:
  - type: relates_to
    links:
      - "[[sensitivity_analysis]]"
      - "[[temperature_parameter]]"
      - "[[convergence_control]]"
---


# Parameter Tuning

Systematic methods to pick learning rates, precisions, and other knobs via grid/random search and adaptive controllers. Tuning is needed because several quantities in an Active Inference agent are modelling choices, not facts about the world, and they change behaviour a great deal. [[sensitivity_analysis]] tells you which of them matter before you spend effort tuning.

## The knobs

| parameter | where | effect |
|-----------|-------|--------|
| `temperature` | `InferenceConfig` | scale of $\mathrm{softmax}(-G/T)$ over policies; lower is greedier |
| `learning_rate` | `InferenceConfig` | blend weight toward the posterior in `variational`, step size in `mean_field` |
| `ModelState.precision`, seeded from `precision_init` | state, `InferenceConfig` | gain in the `mean_field` belief update (step size $\lambda\gamma$), inverse temperature in `HomeostaticInference.select_action` |
| `discount_factor` | `InferenceConfig` | weight of later steps in $G(\pi)$; changes the scale of $G$ |
| `temporal_horizon` | `InferenceConfig` | planning depth; cost grows as $N_a^{H}$ |
| `exploration_weight` | `InferenceConfig`, `AdaptiveControl` | extra information-seeking bias |
| `num_samples`, `policy_limit` | `InferenceConfig` | accuracy and cost of sampling or enumeration |
| preferences `C` | generative model | the task itself; scale controls how strongly risk dominates |

Two of these are easy to conflate. In the policy inference of all three dispatcher methods the softmax scale is `temperature`; precision does not enter. `ModelState.precision` enters the `mean_field` belief update and the homeostatic action rule, and `HomeostaticInference` seeds it from `precision_init`. See [[precision_parameter]] and [[temperature_parameter]].

## Approaches

**Grid, random and quasi-random search.** A grid is exhaustive and wasteful in many dimensions. Random search covers each dimension more densely for the same number of runs when only a few parameters matter. Quasi-random sequences (Sobol, Halton) spread points more evenly than independent draws. All three are embarrassingly parallel and give a response surface you can plot.

**Adaptive controllers.** Rather than fix a value, regulate it toward a target. Two are natural here. The first holds the entropy of the policy distribution at a target value by adjusting $T$: the entropy of $\mathrm{softmax}(-G/T)$ is increasing in $T$, so bisection finds the root. For the expected free energies $G = (0.8056, 0.3639)$ of [[worked_example_efe]], whose maximum entropy is $\ln 2 = 0.6931$ nats, a target of $0.5$ nats gives $T = 0.3182$ and policy probabilities $(0.1997, 0.8003)$. The second tracks prediction error. `ActiveInferenceModel.update_precision` moves precision toward $1/(\text{error} + 10^{-8})$ with exponential smoothing, $\gamma \leftarrow \beta\gamma + (1-\beta)/(e + 10^{-8})$, $\beta = 0.9$. Starting at $\gamma = 1$ with errors $0.5, 0.1, 0.1, 0.1$ it takes the values $1.10, 1.99, 2.79, 3.51$: precision rises as predictions improve. `ActiveInferenceDispatcher.update_precision` does the same with a clip to $[0.1, 10^{6}]$. Such controllers remove one parameter but add their own (target entropy, $\beta$).

**Model-based search.** Bayesian optimisation and bandits spend fewer runs when each run is costly; see [[hyperparameter_optimization]].

## What to optimise

Tune against task outcome (reward collected, goal reached, steps to goal), not against $G$ or $F$. Those depend on the parameters being tuned: changing `discount_factor` or the scale of $C$ rescales $G$, so a lower free energy after a change need not mean better behaviour. Use several random seeds, since the sampling paths are stochastic (`InferenceConfig.seed` fixes the generator), and report the spread.

## Procedure

1. Fix the task and an outcome metric.
2. Run a coarse random or quasi-random search over log-scaled ranges for $T$, learning rate and discount.
3. Inspect sensitivity ([[sensitivity_analysis]]) and freeze parameters with no effect.
4. Refine the remaining ones with a controller or model-based search.
5. Re-validate on fresh seeds.

## Reproducing the numbers

```python
import numpy as np

from cognitive.utils.matrix_utils import softmax

G = np.array([0.8055918, 0.36385392])        # expected free energies from worked_example_efe


def entropy(p):
    return float(-np.sum(p * np.log(p)))


def temperature_for_entropy(G, target, low=1e-3, high=1e3, iterations=80):
    """Bisection in log space; the entropy of softmax(-G / T) increases with T."""
    for _ in range(iterations):
        mid = float(np.sqrt(low * high))
        if entropy(softmax(-G, temperature=mid)) < target:
            low = mid
        else:
            high = mid
    return mid


T = temperature_for_entropy(G, target=0.5)
print(round(T, 4), softmax(-G, temperature=T).round(4), round(entropy(softmax(-G, temperature=T)), 4))
print("maximum entropy", round(float(np.log(len(G))), 4))

# Smoothed precision from prediction error, as in ActiveInferenceModel.update_precision.
precision, beta = 1.0, 0.9
for error in (0.5, 0.1, 0.1, 0.1):
    precision = beta * precision + (1 - beta) / (error + 1e-8)
    print(round(precision, 4))
```

## Links

- [[precision_parameter]]
- [[hyperparameter_optimization]]
- [[sensitivity_analysis]] · [[temperature_parameter]] · [[convergence_control]]
