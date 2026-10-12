---
title: Sequential Monte Carlo
description: Particle-based methods that approximate a sequence of probability distributions by weighted samples updated through importance sampling and resampling, including the particle filter for state-space models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - monte_carlo
  - filtering
  - approximate_inference
semantic_relations:
  - type: relates
    links:
      - "[[particle_filtering]]"
      - "[[importance_sampling]]"
      - "[[monte_carlo_methods]]"
      - "[[variational_sequential_monte_carlo]]"
      - "[[belief_updating]]"
---

# Sequential Monte Carlo

## Overview
Sequential Monte Carlo (SMC) is a family of algorithms that approximate a sequence of distributions $\pi_1,\pi_2,\dots,\pi_T$ with a cloud of $N$ weighted samples, called particles. The classic application is filtering in a state-space model, where $\pi_t(x_{1:t}) = p(x_{1:t}\mid y_{1:t})$ is the posterior over hidden states given observations so far. Unlike [[markov_chain_monte_carlo]], which targets one fixed distribution, SMC moves forward in time and reuses earlier work at each step.

## Algorithm
Each step has three parts.
1. **Propose.** Extend each particle with $x_t^i\sim q(x_t\mid x_{t-1}^i, y_t)$.
2. **Weight.** Update the importance weight
$$w_t^i \propto w_{t-1}^i\,\frac{p(y_t\mid x_t^i)\,p(x_t^i\mid x_{t-1}^i)}{q(x_t^i\mid x_{t-1}^i,y_t)},$$
and normalise so the weights sum to one.
3. **Resample.** When the effective sample size $\mathrm{ESS}=1/\sum_i (w_t^i)^2$ falls below a threshold, draw particles in proportion to their weights and reset the weights to $1/N$.

Choosing the transition prior as the proposal gives the bootstrap filter of Gordon, Salmond and Smith, in which the weight is just the likelihood $p(y_t\mid x_t^i)$.

## Properties
The estimate of the marginal likelihood $p(y_{1:T})$, formed from the product of average unnormalised weights, is unbiased, and expectations of test functions converge as $N\to\infty$ under mild conditions. Resampling prevents weight collapse but causes path degeneracy: the ancestry of the particles coalesces, so smoothing distributions over distant past states are poorly represented. SMC samplers extend the idea to static problems by moving through a tempered or annealed sequence of distributions.

## Relation to Active Inference
Active Inference usually represents beliefs with categorical distributions or Gaussian approximations fitted by gradient descent on variational free energy. SMC is an alternative, sample-based way to perform the same belief updating, handling multimodal and non-Gaussian posteriors at higher computational cost. The relation is methodological. Some neural accounts propose that populations of neurons encode samples (see [[neural_sampling]]), and variational and SMC ideas are combined in [[variational_sequential_monte_carlo]], but standard Active Inference does not require particles.

## Related concepts
- [[particle_filtering]] — SMC applied to online state estimation
- [[importance_sampling]] — the weighting principle on which each SMC step rests
- [[monte_carlo_methods]] — the wider family of sampling approximations
- [[variational_sequential_monte_carlo]] — learning SMC proposals by maximising a variational bound
- [[belief_updating]] — the inferential role that particle updates play

## References
- Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F, 140, 107-113.
- Doucet, A., de Freitas, N., & Gordon, N. (Eds.) (2001). Sequential Monte Carlo Methods in Practice. Springer.
- Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential Monte Carlo samplers. Journal of the Royal Statistical Society: Series B, 68, 411-436.
