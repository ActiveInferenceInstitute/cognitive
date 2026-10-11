---
title: Policy Search Methods (Active Inference Lens)
type: mathematical_concept
status: draft
created: 2025-08-08
updated: 2026-10-09
tags:
- policy-optimization
semantic_relations:
  - type: relates
    links:
      - "[[policy_selection]]"
      - "[[expected_free_energy]]"
      - "[[natural_gradients]]"
  - type: relates_to
    links:
      - "[[search_strategies]]"
      - "[[evolutionary_algorithms]]"
      - "[[stochastic_optimization]]"
---


# Policy Search Methods

A policy is a sequence of actions, and in Active Inference the quality of a policy is its expected free energy $G(\pi)$. Policy search is therefore minimisation of $G$ over a discrete set whose size, $N_a^{H}$ for $N_a$ actions and horizon $H$, grows exponentially.

## Policy inference as variational inference

The softmax rule is not a heuristic added on top of the search. Over distributions $q(\pi)$ on a finite policy set, the functional $\mathbb{E}_{q}[G(\pi)] - H[q(\pi)]$ is minimised by $q(\pi) \propto e^{-G(\pi)}$, and with a prior $E(\pi)$ and an inverse temperature the minimiser becomes $q(\pi) \propto E(\pi)\,e^{-G(\pi)/T}$. Every search method below either evaluates this distribution exactly (enumeration) or approximates it from samples. In the repository the same rule is applied to first-action values after aggregation: `ActiveInferenceDispatcher._mean_field_policy_inference` applies the action prior once through $\ln E - G/T$, and the `variational` path uses $-G/T$ alone; `temperature` is $T$.

## Families

**Enumerative.** Score every sequence with `DiscreteGenerativeModel.evaluate_policy` and normalise. Exact; cost $N_a^{H}$ evaluations of $H$ steps. This is what `ActiveInferenceDispatcher.calculate_expected_free_energy` does, subject to `policy_limit`. See [[search_strategies]] for pruning and beam search.

**Gradient-based.** Give the policy a differentiable parameterisation, for example independent categorical distributions per step with logits $\phi_t$, and descend $\mathbb{E}_{\pi \sim q_\phi}[G(\pi)]$ using the score-function estimator $\nabla_\phi \mathbb{E}[G] = \mathbb{E}[G(\pi)\,\nabla_\phi \ln q_\phi(\pi)]$. A natural-gradient step preconditions this by the Fisher information of $q_\phi$ ([[natural_gradients]]), making the update invariant to how the logits are parameterised. The estimator has high variance; subtract a baseline such as the mean score.

**Sampling-based.** The cross-entropy method (CEM) keeps a factorised sampling distribution over steps, draws sequences, scores them, and refits the distribution to the best fraction (the elites), usually with smoothing. Evolution strategies are close relatives; see [[evolutionary_algorithms]].

## A comparison on a harder instance

The two-state toy models elsewhere in this knowledge base are too easy to separate the methods: at horizon 10, half of all sequences tie for optimal. Here is a randomly generated 4-state, 3-observation, 3-action model with horizon 8, which has $3^8 = 6561$ sequences and a unique optimum. Exhaustive search gives $G^\ast = 8.2634$ at `(1, 0, 1, 1, 0, 1, 0, 1)`; the median sequence scores $9.6776$. With a budget of 320 evaluations (about 5% of the space), over 10 seeds:

| method | mean gap to $G^\ast$ | worst gap |
|--------|---------------------|-----------|
| CEM (8 iterations, 40 samples, 6 elites, smoothing 0.7) | 0.0651 | 0.1565 |
| uniform random search | 0.1680 | 0.4271 |

CEM concentrates its samples where earlier elites lay and is about 2.6 times closer to the optimum on average. It does not find the optimum reliably at this budget. These numbers are from one random instance and ten seeds, so treat them as an illustration of the mechanism rather than a benchmark.

## Practical notes

- Evaluate EFE with the discount factor and `exploration_weight` fixed before comparing methods; both change the scale of $G$.
- Search approximate $G$ on a shorter horizon first, then extend the best prefixes.
- With `PolicyAggregation.MIN` the action value is the best continuation; with `MARGINAL` it is a soft minimum over continuations, so a sampled search should keep enough diversity for the sum to be meaningful.
- Anneal the temperature during sampling-based search ([[annealing_schedules]], [[precision_parameter]]) and compare final rankings at the target temperature.

## Reproducing the numbers

```python
import numpy as np

from cognitive.models.active_inference import DiscreteGenerativeModel

rng = np.random.default_rng(3)
n_s, n_o, n_a, horizon = 4, 3, 3, 8
A = rng.dirichlet(np.ones(n_o) * 0.7, size=n_s).T            # columns are P(o | s)
B = np.stack([rng.dirichlet(np.ones(n_s) * 0.3, size=n_s).T for _ in range(n_a)], axis=2)
model = DiscreteGenerativeModel(A, B, np.array([2.0, 0.0, -1.0]), np.full(n_s, 1 / n_s),
                                np.full(n_a, 1 / n_a))
beliefs = np.full(n_s, 1 / n_s)


def score(policy):
    return model.evaluate_policy(beliefs, tuple(int(a) for a in policy), discount_factor=1.0)


policies = model.enumerate_policies(horizon)
scores = np.array([score(p) for p in policies])
print(len(policies), "best", round(float(scores.min()), 4), policies[int(scores.argmin())],
      "median", round(float(np.median(scores)), 4), "ties", int(np.sum(np.isclose(scores, scores.min()))))


def cem(seed, iterations=8, samples=40, elites=6, smooth=0.7):
    r = np.random.default_rng(seed)
    probs = np.full((horizon, n_a), 1 / n_a)
    best = np.inf
    for _ in range(iterations):
        draws = np.stack([r.choice(n_a, size=samples, p=probs[t]) for t in range(horizon)], axis=1)
        values = np.array([score(d) for d in draws])
        best = min(best, values.min())
        top = draws[np.argsort(values)[:elites]]
        freq = np.stack([np.bincount(top[:, t], minlength=n_a) / elites for t in range(horizon)])
        probs = smooth * freq + (1 - smooth) * probs
    return best


def random_search(seed, budget=8 * 40):
    r = np.random.default_rng(seed)
    return min(score(r.integers(0, n_a, size=horizon)) for _ in range(budget))


for name, fn in (("cem", cem), ("random", random_search)):
    results = np.array([fn(seed) for seed in range(10)])
    gap = results - scores.min()
    print(name, "mean gap", round(float(gap.mean()), 4), "max gap", round(float(gap.max()), 4))
print("budget", 8 * 40, "of", len(policies))
```

## See also

- [[policy_selection]] · [[expected_free_energy]] · [[natural_gradients]] · [[precision_parameter]]

## References

- Friston, K., Rigoli, F., Ognibene, D., Mathys, C., FitzGerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. Cognitive Neuroscience, 6(4), 187-214.
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: A process theory. Neural Computation, 29(1), 1-49.
- Rubinstein, R. Y. (1999). The cross-entropy method for combinatorial and continuous optimization. Methodology and Computing in Applied Probability, 1(2), 127-190.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4), 229-256.
- Amari, S. (1998). Natural gradient works efficiently in learning. Neural Computation, 10(2), 251-276.
