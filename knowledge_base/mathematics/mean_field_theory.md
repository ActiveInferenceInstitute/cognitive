---
title: Mean Field Theory
description: Approximation that replaces interactions among many variables by an average field, used in statistical physics for phase transitions and in variational inference for factorised posteriors in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistical_physics
  - variational_methods
  - approximation
semantic_relations:
  - type: relates
    links:
      - "[[variational_inference]]"
      - "[[free_energy]]"
      - "[[statistical_physics]]"
      - "[[phase_transitions]]"
      - "[[variational_message_passing]]"
      - "[[markov_random_fields]]"
---

# Mean Field Theory

## Overview
Mean field theory approximates a system of many interacting components by treating each component as responding to the average influence of the others. Fluctuations around that average are neglected, which converts a coupled many-body problem into a set of single-variable problems solved self-consistently. The term has two closely related uses, in physics and in probabilistic inference.

## Statistical physics
For the Ising model with coupling $J$, external field $h$ and $z$ neighbours per site, the mean field approximation gives the self-consistent magnetisation
$$m=\tanh\big(\beta(Jzm+h)\big),\qquad \beta=1/k_BT.$$
With $h=0$, a nonzero solution appears below $T_c=zJ/k_B$, predicting a continuous phase transition. The approximation is exact for fully connected models in the large-system limit and is quantitatively poor in low dimensions, where it misses fluctuations and gives incorrect critical exponents. It is rigorously justified as a bound: the Bogoliubov inequality shows that the free energy of a factorised trial distribution is never below the true free energy.

## Variational inference
In Bayesian inference, the mean field family restricts the approximate posterior to a product $q(x)=\prod_i q_i(x_i)$. Minimising the Kullback-Leibler divergence to the true posterior gives coordinate updates
$$\ln q_i^*(x_i)=\mathbb{E}_{q_{-i}}[\ln p(x)]+\text{const},$$
each of which is computed from expectations over the other factors. This is the same variational principle as in physics, with the negative log joint playing the role of energy. The approximation tends to underestimate posterior uncertainty and to ignore correlations between variables.

## Relation to Active Inference
The link is a derivation. Active Inference commonly assumes a mean field factorisation of beliefs over hidden-state factors and over time steps, so that free energy can be minimised by local updates that resemble message passing. The same factorisation underlies variational message passing on factor graphs. Mean field assumptions are modelling choices that trade accuracy for tractability, and structured alternatives such as the Bethe approximation retain some dependencies.

## Related concepts
- [[variational_inference]] — the framework in which the mean field family is optimised
- [[free_energy]] — the objective minimised by the factorised approximation
- [[bethe_free_energy]] — an approximation that keeps pairwise dependencies
- [[variational_message_passing]] — local updates under the mean field assumption
- [[phase_transitions]] — the physical setting where the theory originated

## References
- Parisi, G. (1988). Statistical Field Theory. Addison-Wesley.
- Wainwright, M. J., & Jordan, M. I. (2008). Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning, 1, 1-305.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning, chapter 10. Springer.
