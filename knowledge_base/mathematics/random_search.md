---
title: Random Search
description: Optimisation and exploration by sampling candidate solutions without using gradients or learned structure, its convergence guarantees, its curse of dimensionality, and uses in hyperparameter tuning and foraging models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - search
  - stochastic_methods
semantic_relations:
  - type: relates
    links:
      - "[[search_strategies]]"
      - "[[stochastic_optimization]]"
      - "[[hyperparameter_optimization]]"
      - "[[random_walk]]"
      - "[[exploration_exploitation]]"
wikidata: Q4925641
---

# Random Search

## Overview
Random search optimises a function $f$ over a domain $D$ by evaluating it at randomly generated points and keeping the best. In its simplest form, pure random search, points are sampled independently from a fixed distribution. Adaptive variants sample near the current best point, with a step distribution that may shrink or grow. It requires only function values, so it applies to non-differentiable, noisy or discrete objectives.

## Properties
- **Global convergence.** If $D$ is compact and the sampling distribution gives positive probability to every neighbourhood of the global optimum, the best value found converges to the optimum with probability one as the number of samples grows. Solis and Wets (1981) gave general conditions for this property for adaptive random search algorithms.
- **Dimension dependence.** To land within distance $\varepsilon$ of a point in the unit cube $[0,1]^d$ with uniform sampling, the success probability per sample is proportional to $\varepsilon^d$. The expected number of samples therefore grows exponentially in $d$, so pure random search is slow when many dimensions matter.
- **Effective low dimensionality.** In hyperparameter optimisation, when only a few hyperparameters strongly affect performance, random search explores more distinct values of each influential parameter than a grid with the same budget, and Bergstra and Bengio (2012) reported that, on the neural network tasks they studied, it found models as good as or better than grid search with much less computation ([[hyperparameter_optimization]]).

## Exploration without direction
Random search is the undirected end of the exploration-exploitation trade-off ([[exploration_exploitation]]). In biology, foraging has been modelled by random walks and heavy-tailed step distributions, and ants and other social insects are often described as combining random search with trail following when no information is available ([[social_insect_cognition]], [[random_walk]]). Which movement statistics real animals use is an empirical question and remains debated.

## Relation to Active Inference
Random search is a contrast case. Active Inference agents select actions by minimising expected free energy, whose epistemic term directs exploration toward informative outcomes, so exploration is guided by beliefs about hidden states. Undirected exploration appears only as noise in action selection, for example through the precision parameter of a softmax policy posterior ([[softmax_function]], [[exploration_strategies]]). The relation is that of a baseline to a directed scheme and implies no further theoretical link.

## Related concepts
- [[search_strategies]] — the broader family of search methods
- [[stochastic_optimization]] — optimisation with random elements
- [[hyperparameter_optimization]] — the main practical use of random search
- [[random_walk]] — the movement model for undirected exploration
- [[exploration_exploitation]] — the trade-off random search sits at one end of
- [[epsilon_greedy]] — a simple mix of random and greedy choice

## References
- Solis, F. J., & Wets, R. J.-B. (1981). Minimization by random search techniques. Mathematics of Operations Research, 6, 19-30.
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13, 281-305.
