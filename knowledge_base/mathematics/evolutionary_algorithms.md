---
title: Evolutionary Algorithms
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - mathematics
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/mathematics/variational_methods]]"
      - "[[knowledge_base/biology/evolutionary_biology]]"
      - "[[knowledge_base/mathematics/policy_search_methods]]"
      - "[[knowledge_base/mathematics/natural_gradients]]"
      - "[[knowledge_base/mathematics/stochastic_optimization]]"
---

# Evolutionary Algorithms

Evolutionary algorithms optimise by maintaining a population of candidates, scoring them, and producing the next generation from the better ones. They need only function evaluations, no gradients, which makes them useful for non-differentiable objectives such as expected free energy over discrete action sequences.

## Main families

**Genetic algorithms** encode a candidate as a string, select parents in proportion to fitness, recombine them by crossover and perturb offspring by mutation. They suit combinatorial problems like policy sequences, where crossover swaps blocks of actions.

**Evolution strategies (ES)** work on real-valued parameters. In a $(\mu/\mu, \lambda)$-ES, $\lambda$ offspring are sampled from $\mathcal{N}(m, \sigma^2 I)$, the best $\mu$ are kept, and their average becomes the new mean $m$. Step size $\sigma$ is decayed or self-adapted.

**Natural evolution strategies (NES)** treat the sampling distribution $p_\phi$ as the object being optimised and follow the gradient of expected loss with the score-function identity $\nabla_\phi \mathbb{E}_{p_\phi}[f] = \mathbb{E}_{p_\phi}[f(x)\,\nabla_\phi \ln p_\phi(x)]$. Preconditioning by the inverse Fisher information gives the natural gradient ([[natural_gradients]]). The cross-entropy method in [[policy_search_methods]] is a close cousin that refits $p_\phi$ to the elite samples.

**Coevolution** evolves populations whose fitness depends on each other, as in predator and prey or competing agents, so the landscape moves.

## Selection is Bayesian updating

For a population with type frequencies $x_i$ and fitnesses $f_i$, the discrete replicator step is

$$x_i' = \frac{x_i f_i}{\sum_j x_j f_j}$$

This is Bayes' rule with frequencies as the prior and fitness as the likelihood; the mean fitness is the evidence. Reusing the numbers of [[worked_example_vfe]], prior $(0.6, 0.4)$ and fitness $(0.9, 0.2)$ (the likelihood of the observed outcome under each state) give $x' = (0.8710, 0.1290)$ and mean fitness $0.62$, which are the exact posterior and evidence of that page. Repeated selection without mutation therefore performs repeated Bayesian updating on a fixed set of hypotheses, and mutation supplies new hypotheses. Reading evolution as inference is an interpretation that links this page to [[knowledge_base/mathematics/variational_methods]] and [[knowledge_base/biology/evolutionary_biology]]; it is not something the code in this repository implements.

## Numerical illustration

A $(3/3, 12)$-ES minimising $\lVert x - (1, -2)\rVert^2$ from the origin with $\sigma$ decayed by 0.85 per generation reaches mean $(1.005, -1.999)$, loss $2.9 \times 10^{-5}$, after 30 generations (seed 0). The score-function estimator with $2 \times 10^5$ samples at $m = 0$, $\sigma = 1$ gives the gradient $(-2.01, 3.99)$ against the exact $(-2, 4)$.

## When to use them here

Use evolutionary search when the objective is a black box, the space is discrete or mixed, or evaluations parallelise. Expect it to need many more evaluations than gradient methods on smooth problems. For perception, where $F[q]$ is smooth, gradient and mirror-descent updates ([[optimization_strategies]]) are the better tool; for planning over long horizons ES and CEM are reasonable alternatives to enumeration ([[search_strategies]]).

## Reproducing the numbers

```python
import numpy as np

# 1. Discrete replicator step is Bayes' rule (fitness plays the likelihood).
prior = np.array([0.6, 0.4])                 # type frequencies = prior over states
fitness = np.array([0.9, 0.2])               # = A[0, :] from the VFE example
next_freq = prior * fitness / np.sum(prior * fitness)
print(next_freq, "mean fitness", np.sum(prior * fitness))   # posterior and p(o)

# 2. (mu/mu, lambda) evolution strategy with a deterministic step-size decay.
rng = np.random.default_rng(0)
target = np.array([1.0, -2.0])


def loss(x):
    return np.sum((x - target) ** 2, axis=-1)


mean, sigma, lam, mu = np.zeros(2), 1.0, 12, 3
for _ in range(30):
    population = mean + sigma * rng.standard_normal((lam, 2))
    mean = population[np.argsort(loss(population))[:mu]].mean(axis=0)
    sigma *= 0.85
print(mean.round(3), float(loss(mean)))

# 3. Score-function (natural evolution strategy) gradient of E[loss] w.r.t. the mean.
m, s = np.zeros(2), 1.0
samples = m + s * rng.standard_normal((200_000, 2))
grad = (loss(samples)[:, None] * (samples - m) / s**2).mean(axis=0)
print(grad.round(2), 2 * (m - target))
```

## Related

- [[knowledge_base/mathematics/variational_methods]]
- [[knowledge_base/biology/evolutionary_biology]]
- [[policy_search_methods]] · [[natural_gradients]] · [[stochastic_optimization]]
