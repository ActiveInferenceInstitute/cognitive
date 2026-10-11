---
title: Exploration Strategies
type: mathematical_concept
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- exploration
- decision_making
semantic_relations:
  - type: relates_to
    links:
      - "[[expected_free_energy]]"
      - "[[mutual_information]]"
      - "[[exploration_visualization]]"
---


# Exploration Strategies

Approaches to information-seeking behavior: uncertainty sampling, expected information gain, and novelty bonuses. This page compares the standard strategies with what Active Inference does, and shows where each knob sits in the code.

## Standard strategies

| strategy | rule | kind of exploration |
|----------|------|---------------------|
| $\epsilon$-greedy | with probability $\epsilon$ take a random action | undirected |
| Boltzmann (softmax) | $P(a) \propto e^{-Q(a)/T}$ | undirected, graded by value |
| Upper confidence bound | add a bonus that shrinks with visit counts | directed at rarely tried actions |
| Thompson sampling | act optimally for a parameter sampled from the posterior | directed by posterior uncertainty |
| Uncertainty sampling | query where predictive uncertainty is largest | directed at ambiguous inputs |
| Novelty or count bonus | reward visiting unfamiliar states | directed at unfamiliar states |

Undirected methods inject randomness independent of what is unknown. Directed methods spend actions where information is to be had, which is why they usually win when exploration is costly.

## Exploration as expected information gain

In Active Inference exploration is not a separate mechanism. Expected free energy is risk plus ambiguity, and ambiguity equals $H[q(o)] - I(s;o)$, so lowering $G$ raises the expected information gain $I(s;o)$ about hidden states ([[epistemic_value]], [[mutual_information]]). The bonus is directed, scales with current uncertainty, and vanishes by itself when beliefs become sharp, because $I(s;o) = 0$ for a deterministic belief. Equivalently, $G = -I(s;o) - \mathbb{E}_{q(o)}[\ln p^\ast(o)]$: negative epistemic value minus pragmatic value ([[exploration_exploitation]]).

With flat preferences, so that $p^\ast(o)$ is uniform, the pragmatic term is constant: $G = \ln |O| - I(s;o)$. In the two-state model of [[worked_example_efe]] the "stay" action then has $G = 0.6118$ and "move" has $0.6501$, which is $\ln 2$ minus $0.0813$ and $0.0431$ nats of information gain. The agent explores because exploring is all there is to prefer; once preferences are non-flat (here $C = (2, 0)$) the pragmatic term dominates and "move" wins with $G = 0.3639$.

## Tuning exploration in this repository

- **`temperature`** in `InferenceConfig` is undirected randomness over the policy distribution, $\mathrm{softmax}(-G/T)$. Raising $T$ flattens choices without regard to what is informative.
- **`exploration_weight`** (range $[0, 1]$, default 0) subtracts an extra multiple of the information gain from each step's $G$, on top of the epistemic drive already inside ambiguity. At weight 1 the preferred-outcome model above changes from $G = (0.8056, 0.3639)$ to $(0.7243, 0.3208)$. Weight 1 reproduces the objective used before version 1.1.0, which counted information gain twice; see "Corrected mathematics" in `CHANGELOG.md`. Use it as a deliberate optimism bias, not as the baseline.
- **`AdaptiveControl(exploration_weight=...)`** in `homeostatic.py` mixes target tracking with a prediction-error term in its policy prior, and `HomeostaticInference.update_parameters` sets that weight to $\mathrm{clip}(0.5 - 0.4\,s, 0.1, 0.5)$ from a stability index $s$, so a stable controller explores less.
- **Horizon and discount** (`temporal_horizon`, `discount_factor`) set how far ahead information gain is credited.

## Limits

Information gain here is about states under a fixed likelihood. Exploration to learn the likelihood itself (novelty about parameters, via Dirichlet counts) is outside `DiscreteGenerativeModel`, whose matrices are fixed. In a very ambiguous world the gain can be small for every action, and the pragmatic term decides.

## Reproducing the numbers

```python
import numpy as np

from cognitive.models.active_inference import DiscreteGenerativeModel

A = np.array([[0.9, 0.5], [0.1, 0.5]])
B = np.zeros((2, 2, 2))
B[:, :, 0] = np.eye(2)
B[:, :, 1] = [[0.9, 0.9], [0.1, 0.1]]
beliefs = np.array([0.3, 0.7])


def model_with(C):
    return DiscreteGenerativeModel(A, B, np.asarray(C, dtype=float), np.full(2, 0.5), np.full(2, 0.5))


# Flat preferences: G = ln(2) - I(s; o), so ranking by G is ranking by information gain.
flat = model_with([0.0, 0.0])
for action in range(2):
    G, _risk, _ambiguity, info = flat.expected_free_energy(beliefs, action)
    print("flat", action, round(G, 4), round(np.log(2) - info, 4), round(info, 4))

# Preferences for outcome 0: extra epistemic weight on top of the canonical objective.
prefer = model_with([2.0, 0.0])
for weight in (0.0, 1.0):
    values = [prefer.evaluate_policy(beliefs, (a,), 1.0, epistemic_weight=weight) for a in range(2)]
    print("weight", weight, np.round(values, 4))
```

## Links

- [[epistemic_value]]
- [[exploration_exploitation]]
- [[exploration_methods]] · [[expected_free_energy]] · [[mutual_information]]

## References

- Sutton, R. S., & Barto, A. G. (1998). Reinforcement Learning: An Introduction. MIT Press.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2-3), 235-256.
- Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial on Thompson sampling. Foundations and Trends in Machine Learning, 11(1), 1-96.
- Friston, K., Rigoli, F., Ognibene, D., Mathys, C., FitzGerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. Cognitive Neuroscience, 6(4), 187-214.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
